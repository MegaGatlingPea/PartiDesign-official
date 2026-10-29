"""Candidate provenance, residue-token identity and chemical invariants."""
from __future__ import annotations
import csv
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import re
import tempfile
from collections import Counter
from collections.abc import Mapping
from .selection import ChainSequence

_TOKEN = re.compile(r"\[[^\[\]]+\]|[A-Za-z]")

def tokens(sequence):
    result = tuple(_TOKEN.findall(sequence))
    if "".join(result) != sequence:
        raise ValueError(f"invalid residue-token sequence: {sequence!r}")
    return result

def jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return value

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="."+path.name, dir=path.parent)
    try:
        with os.fdopen(fd,"w") as stream:
            json.dump(jsonable(value),stream,ensure_ascii=False,allow_nan=False)
            stream.write("\n")
        os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["candidate_id","status"]
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(jsonable(v),separators=(",",":"))
                             if isinstance(v,(dict,tuple,list)) else v for k,v in row.items()})

@dataclass(frozen=True)
class Candidate:
    candidate_id:str
    task:str
    parent_id:str
    origin_id:str
    path:str
    row:dict
    source:str="gt"
    prediction_key:str=""
    score:float=0.0
    origin_path:str=""
    score_profile:str=""
    sequence_identity:str=""
    lineage_id:str=""
    def saved(self):
        return asdict(self)

def load_candidate(candidate,task_dir,*,atoms=None):
    from src.validate.inputs import load_input
    if atoms is None:
        record=load_input(candidate.path,candidate.row,task_dir)
    else:
        # Workflow rows inherit already-validated task conditions. Reuse the
        # parsed structure and the native metadata/constraint producers.
        from src.validate.inputs import (ValidationInput, _parse_design_chains,
            _parse_contig_positions, _parse_tip_positions, _fixed_position_index,
            _validate_constraint_identity, _cyclic_chains_from_graph)
        row=dict(candidate.row)
        metadata=_parse_design_chains(row["design_chains"],row_context=candidate.candidate_id)
        design=tuple(x["chain"] for x in metadata)
        atoms=_prepared_atoms(atoms,design)
        fixed=[x for x in json.loads(row.get("fixed_restypes") or "[]") if x.get("chain") in design]
        bonds=json.loads(row.get("bond_constraints") or "[]")
        contig=row.get("contig_string","")
        tips=row.get("tip_atoms","")
        record=ValidationInput(path=Path(candidate.path),task_dir=Path(task_dir),
            sample_id=Path(candidate.path).stem,row=row,atoms=atoms,design_chains=design,
            fixed_restypes=fixed,bond_constraints=bonds,contig_string=contig,tip_atoms=tips,
            cyclic_chains=_cyclic_chains_from_graph(atoms),
            cyclic_offset_chains=tuple(x["chain"] for x in metadata if x.get("cyclic_offset")),
            fixed_positions=_fixed_position_index(fixed),
            motif_positions=_parse_contig_positions(contig),tip_positions=_parse_tip_positions(tips),
            reasons=tuple(_validate_constraint_identity(atoms,fixed,bonds)))
    if record.reasons:
        raise ValueError(f"{candidate.candidate_id}: "+"; ".join(record.reasons))
    return record

def residues(atoms,chain):
    import numpy as np
    labels=np.asarray(getattr(atoms,"label_asym_id",atoms.chain_id)).astype(str)
    groups={}
    for i in np.flatnonzero(labels==chain):
        key=(str(atoms.res_id[i]),str(atoms.ins_code[i]))
        groups.setdefault(key,[]).append(int(i))
    return list(groups.values())

def position(atoms,i):
    try:
        value=int(getattr(atoms,"label_seq_id",atoms.res_id)[i])
    except (ValueError,TypeError):
        value=int(atoms.res_id[i])
    return value,str(atoms.ins_code[i])

def _prepared_atoms(atoms,design_chains):
    import numpy as np
    from src.validate.inputs import _finalize_loaded_identity
    if not hasattr(atoms,"token_uid") or bool(np.any(np.asarray(atoms.token_uid)<0)):
        return _finalize_loaded_identity(atoms,design_chains)
    labels=np.asarray(getattr(atoms,"label_asym_id",atoms.chain_id)).astype(str)
    atoms.is_design=np.isin(labels,design_chains)
    return atoms


def _bond_anchor(record,bond):
    from src.validate.inputs import _locator_rows
    from src.validate.sequences import _bond_order_name
    endpoints=[]
    for name in ("atom1","atom2"):
        rows=_locator_rows(record.atoms,{k:v for k,v in bond[name].items() if k not in {"atom_id","atom_uid","uid","atom_index","residue_uid","token_uid"}})
        if len(rows)!=1:
            raise ValueError("bond diversity anchor has no unique semantic atom")
        i=rows[0]
        endpoints.append((str(getattr(record.atoms,"label_asym_id",record.atoms.chain_id)[i]),
                          *position(record.atoms,i),str(record.atoms.atom_name[i])))
    return {"endpoints":sorted(endpoints),"order":_bond_order_name(bond.get("order")),
            "required_output":bond.get("required_output",True)}


def chain_sequences(record,variant=None):
    from src.validate.sequences import _AA3_TO_1,_RNA3_TO_1,_DNA3_TO_1,_fixed_modified_map
    from src.validate.inputs import _locator_rows
    atoms=record.atoms
    modified,_=_fixed_modified_map(record)
    maps={"protein":_AA3_TO_1,"rna":_RNA3_TO_1,"dna":_DNA3_TO_1}
    proposals={} if variant is None else variant.get("sequences",{})
    fixed_rows={row for entry in record.fixed_restypes for row in _locator_rows(atoms,entry)}
    graph=special_edges(record)
    result=[]
    for chain in record.design_chains:
        groups=residues(atoms,chain)
        if not groups:
            raise ValueError(f"empty design chain {chain!r}")
        modality=str(atoms.mol_type[groups[0][0]])
        if modality not in maps:
            raise ValueError("pipeline supports polymer designs only; pure ligand design is unsupported")
        proposal=tokens(str(proposals[chain])) if chain in proposals else None
        if proposal is not None and len(proposal)!=len(groups):
            raise ValueError(f"proposal changed chain {chain!r} length")
        values,anchors=[],[]
        for ordinal,rows in enumerate(groups):
            i=rows[0]
            pos=position(atoms,i)
            name=str(atoms.res_name[i]).upper()
            uid=int(atoms.residue_uid[i])
            token=f"[{modified[uid]}]" if uid in modified else maps[modality].get(name,f"[{name.strip('[]')}]")
            if proposal is not None and uid not in modified:
                token=proposal[ordinal]
            values.append(token)
            fixed=[]
            for field in ("fixed_positions","motif_positions","tip_positions"):
                if pos in getattr(record,field).get(chain,()):
                    fixed.append(f"{field}:{chain}:{pos}:{name}")
            if not fixed_rows.isdisjoint(rows):
                fixed.append(f"identity:{chain}:{pos}:{name}")
            for bond in record.bond_constraints:
                if ring_closure(record,bond):
                    continue
                if any(r in _locator_rows(atoms,bond[end]) for end in ("atom1","atom2") for r in rows):
                    fixed.append("bond:"+json.dumps(_bond_anchor(record,bond),sort_keys=True))
            for edge,order in graph:
                if _ring_edge(record,edge):
                    continue
                if any(end[0]==chain and end[1:3]==pos for end in edge):
                    fixed.append("graph:"+json.dumps([edge,order]))
            anchors.append(tuple(sorted(set(fixed))))
        cyclic=chain in (*record.cyclic_chains,*record.cyclic_offset_chains)
        result.append(ChainSequence(chain,modality,tuple(values),cyclic,tuple(anchors)))
    return tuple(result)

def refreshed_row(record,path,*,atoms=None):
    from src.design.utils.reader import read_structure
    atoms=_prepared_atoms(read_structure(path) if atoms is None else atoms,record.design_chains)
    current=replace(record,atoms=atoms,fixed_restypes=[{k:v for k,v in x.items() if k!="atom_id"} for x in record.fixed_restypes])
    chains=chain_sequences(current)
    row=dict(record.row)
    previous={x["chain"]:x for x in json.loads(row.get("design_chains") or "[]")}
    row["design_chains"]=json.dumps([
        {**previous.get(c.chain_id,{}),"chain":c.chain_id,"modality":c.modality,
         "sequence":"".join(c.tokens),"residue_count":len(c.tokens),
         **({"cyclic_offset":True} if c.chain_id in record.cyclic_offset_chains else {})}
        for c in chains],separators=(",",":"))
    # Public atom serials change when a free residue changes its sidechain.
    # Resolve inherited locators in the parent, then remap the same semantic
    # atom/residue occurrence in the restored child.
    from src.validate.inputs import _locator_rows
    def remap(locator):
        value=dict(locator)
        old=_locator_rows(record.atoms,value)
        if not old:
            raise ValueError(f"inherited locator cannot be resolved: {value}")
        i=old[0]
        value.update(chain=str(getattr(record.atoms,"label_asym_id",record.atoms.chain_id)[i]),
                     res_id=int(record.atoms.res_id[i]),ins_code=str(record.atoms.ins_code[i]))
        had_serial="atom_id" in value
        value.pop("atom_id",None)
        found=_locator_rows(atoms,value)
        if not found or (value.get("atom") and len(found)!=1):
            raise ValueError(f"refold lost an inherited occurrence: {value}")
        if had_serial:
            if len(found)!=1:
                raise ValueError(f"atom serial requires a unique occurrence: {value}")
            value["atom_id"]=found[0]+1
        return value
    row["fixed_restypes"]=json.dumps([remap(x) for x in record.fixed_restypes],separators=(",",":"))
    row["bond_constraints"]=json.dumps([
        {**bond,"atom1":remap(bond["atom1"]),"atom2":remap(bond["atom2"])}
        for bond in record.bond_constraints],separators=(",",":"))
    row.pop("design_sequence",None)
    row["filename"]=str(path)
    return row

def _context_atom_key(atoms, index):
    labels = getattr(atoms, "label_asym_id", atoms.chain_id)
    return (str(labels[index]), *position(atoms, index), str(atoms.res_name[index]),
            str(atoms.atom_name[index]), str(atoms.element[index]))


def _context_rows(record):
    import numpy as np
    atoms = record.atoms
    labels = np.asarray(getattr(atoms, "label_asym_id", atoms.chain_id)).astype(str)
    return [i for i in range(len(atoms)) if labels[i] not in record.design_chains
            and str(atoms.element[i]) not in {"H", "D", "T"}]


def _assert_context(parent, child):
    # Native refolding owns canonical polymer completion. Preserve known
    # source facts without rejecting newly completed standard terminal atoms.
    original={_context_atom_key(parent.atoms,i) for i in _context_rows(parent)}
    predicted={_context_atom_key(child.atoms,i) for i in _context_rows(child)}
    missing=original-predicted
    if missing:
        optional_oxt=set()
        present_names={key[:5] for key in predicted}
        for chain in {key[0] for key in missing if key[4:]==("OXT","O")}:
            terminal=residues(parent.atoms,chain)[-1]
            pos=position(parent.atoms,terminal[0])
            if (parent.atoms.mol_type[terminal[0]]=="protein" and
                    "OXT" not in parent.tip_positions.get(chain,{}).get(pos,())):
                optional_oxt.update(key for i in terminal
                    if (key:=_context_atom_key(parent.atoms,i))[4:]==("OXT","O")
                    and key[:5] not in present_names)
        if not missing.issubset(optional_oxt):
            raise ValueError("refold changed context atom identity: missing or replaced source atom")
    if {key[:4] for key in original}!={key[:4] for key in predicted}:
        raise ValueError("refold changed context residue identity")


def _cif_atom_locator(category, index, prefix=""):
    def value(name, default=""):
        return str(category[name].as_array()[index]) if name in category else default
    def clean(text):
        return "" if text in {"", ".", "?"} else text
    chain = clean(value(prefix + "label_asym_id")) or clean(value(prefix + "auth_asym_id"))
    sequence = clean(value(prefix + "label_seq_id")) or clean(value(prefix + "auth_seq_id"))
    insertion_name = "pdbx_" + prefix + "PDB_ins_code" if prefix else "pdbx_PDB_ins_code"
    try:
        sequence = int(sequence)
    except ValueError as exc:
        raise ValueError("ligand alias CIF occurrence has no integral residue locator") from exc
    return (chain, sequence, clean(value(insertion_name)),
            value(prefix + "label_comp_id"), value(prefix + "label_atom_id"))


def _rewrite_cif_aliases(path, atoms, replacements):
    """Rename only atom identifiers and their CIF references, preserving data."""
    from biotite.structure.io import pdbx
    cif = pdbx.CIFFile.read(path)
    if len(cif) != 1:
        raise ValueError("ligand alias restoration requires one CIF data block")
    block = next(iter(cif.values()))
    site = block["atom_site"]
    by_locator = {}
    components = {}
    instances = set()
    for row, new_name in replacements.items():
        chain, pos, insertion, comp, old_name, _ = _context_atom_key(atoms, row)
        locator = (chain, pos, insertion, comp, old_name)
        if locator in by_locator:
            raise ValueError("ligand alias occurrence is ambiguous")
        by_locator[locator] = new_name
        previous = components.setdefault(comp, {}).get(old_name)
        if previous is not None and previous != new_name:
            raise ValueError("ligand component aliases conflict between occurrences")
        components[comp][old_name] = new_name
        instances.add(locator[:4])
    columns = {name: site[name].as_array().tolist() for name in ("label_atom_id", "auth_atom_id") if name in site}
    found = Counter()
    for row in range(site.row_count):
        comp = str(site["label_comp_id"].as_array()[row])
        if comp not in components:
            continue
        locator = _cif_atom_locator(site, row)
        if locator[:4] not in instances:
            raise ValueError("ligand alias component is also used by an unrelated occurrence")
        if locator in by_locator:
            found[locator] += 1
            for values in columns.values():
                values[row] = by_locator[locator]
    if set(found) != set(by_locator) or any(count != 1 for count in found.values()):
        raise ValueError("ligand alias CIF atom occurrence is absent or ambiguous")
    for name, values in columns.items():
        site[name] = values
    for category_name, category in block.items():
        if (category_name.startswith("chem_comp") or category_name.startswith("pdbx_chem_comp")) and "comp_id" in category:
            fields = [name for name in category if re.fullmatch(
                r"(?:atom_id(?:_[1-4]|_centre)?|alt_atom_id|pdbx_component_atom_id)", name)]
            for name in fields:
                values = category[name].as_array().tolist()
                for row, component in enumerate(category["comp_id"].as_array()):
                    mapping = components.get(str(component), {})
                    values[row] = mapping.get(str(values[row]), values[row])
                category[name] = values
    if "struct_conn" in block:
        category = block["struct_conn"]
        for partner in ("ptnr1_", "ptnr2_"):
            label = partner + "label_atom_id"
            if label not in category:
                continue
            fields = {name: category[name].as_array().tolist()
                      for name in (label, partner + "auth_atom_id") if name in category}
            for row in range(category.row_count):
                component = str(category[partner + "label_comp_id"].as_array()[row])
                if component not in components:
                    continue
                locator = _cif_atom_locator(category, row, partner)
                if locator in by_locator:
                    for values in fields.values():
                        values[row] = by_locator[locator]
            for name, values in fields.items():
                category[name] = values
    fd, temporary = tempfile.mkstemp(prefix="." + path.stem, suffix=".cif", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(cif.serialize())
        os.replace(temporary, path)
        for row, name in replacements.items():
            atoms.atom_name[row] = name
    finally:
        Path(temporary).unlink(missing_ok=True)


def restore_context_aliases(parent_record,target_cif,*,atoms=None):
    """Restore public aliases using the native validator's chemical mapping."""
    import numpy as np
    from src.design.utils.reader import read_structure
    from src.validate.ligands import resolve_context_ligands
    from src.validate.ligand_metrics import atom_mapping
    source=parent_record.atoms
    labels=np.asarray(getattr(source,"label_asym_id",source.chain_id)).astype(str)
    chains=list(dict.fromkeys(labels[(source.mol_type=="ligand") &
                                    ~np.isin(labels,parent_record.design_chains)].tolist()))
    if not chains:
        return atoms
    target=read_structure(target_cif) if atoms is None else atoms
    entries,_=resolve_context_ligands(parent_record.task_dir,source,
                                      design_chains=parent_record.design_chains,chains=chains)
    replacements={}
    for chain in chains:
        mapping=atom_mapping(source,target,chain,chain,trusted_smiles=entries[chain].get("smiles"))
        replacements.update({b:str(source.atom_name[a]) for a,b in mapping.items()
                             if str(source.atom_name[a])!=str(target.atom_name[b])})
    if replacements:
        _rewrite_cif_aliases(Path(target_cif),target,replacements)
    return target


def assert_invariants(parent,child):
    import numpy as np
    from src.validate.inputs import _locator_rows
    if parent.design_chains!=child.design_chains:
        raise ValueError("refold changed design chain roles")
    for chain in parent.design_chains:
        if len(residues(parent.atoms,chain))!=len(residues(child.atoms,chain)):
            raise ValueError(f"refold changed chain {chain!r} length")
    if parent.cyclic_offset_chains!=child.cyclic_offset_chains:
        raise ValueError("cyclic_offset intent was lost")
    for entry in parent.fixed_restypes:
        a,b=_locator_rows(parent.atoms,entry),_locator_rows(child.atoms,{k:v for k,v in entry.items() if k!="atom_id"})
        if not a or not b or set(parent.atoms.res_name[a])!=set(child.atoms.res_name[b]):
            raise ValueError(f"fixed residue identity changed: {entry}")
    for chain,positions in parent.motif_positions.items():
        for pos in positions:
            def identity(record):
                return {str(record.atoms.res_name[g[0]]) for g in residues(record.atoms,chain)
                        if position(record.atoms,g[0])==pos}
            if not identity(parent) or identity(parent)!=identity(child):
                raise ValueError(f"motif identity changed: {chain}:{pos}")
    for bond in parent.bond_constraints:
        a,b=(_locator_rows(child.atoms,{k:v for k,v in bond[end].items() if k!="atom_id"}) for end in ("atom1","atom2"))
        if len(a)!=1 or len(b)!=1 or not any(
            {int(i),int(j)}=={a[0],b[0]} for i,j,_ in child.atoms.bonds.as_array()):
            raise ValueError(f"required chemical connection missing: {bond}")
    if special_edges(parent)!=special_edges(child):
        raise ValueError("refold changed the required special-bond graph")
    _assert_context(parent,child)



def special_edges(record):
    from src.validate.sequences import _graph_topology_conditions
    from src.design.core.records import TopologyCondition
    atoms=record.atoms
    indices={int(uid):i for i,uid in enumerate(atoms.atom_uid)}
    labels=getattr(atoms,"label_asym_id",atoms.chain_id)
    def endpoint(i):
        return (str(labels[i]),*position(atoms,i),str(atoms.atom_name[i]))
    edges=[]
    for edge in _graph_topology_conditions(record,TopologyCondition):
        a,b=indices[edge.atom_uid_a],indices[edge.atom_uid_b]
        if (atoms.mol_type[a]==atoms.mol_type[b]=="ligand" and
                not atoms.is_design[a] and not atoms.is_design[b] and
                atoms.molecule_uid[a]==atoms.molecule_uid[b]):
            continue
        edges.append((tuple(sorted((endpoint(a),endpoint(b)))),edge.order))
    return tuple(sorted(edges))

def has_special_polymer_bond(record):
    """Detect non-backbone polymer links from the current graph and ledger."""
    import numpy as np
    from src.validate.inputs import _locator_rows

    atoms=getattr(record,"atoms",None)
    if atoms is None:
        return False
    labels=np.asarray(getattr(atoms,"label_asym_id",atoms.chain_id)).astype(str)
    modalities=np.asarray(atoms.mol_type).astype(str)
    residue_uids=np.asarray(atoms.residue_uid)
    names=np.asarray(atoms.atom_name).astype(str)
    polymer=np.isin(modalities,("protein","rna","dna"))
    order={}
    lengths={}
    for row in np.flatnonzero(polymer):
        key=(str(labels[row]),int(residue_uids[row]))
        if key not in order:
            order[key]=lengths.get(key[0],0)
            lengths[key[0]]=order[key]+1

    def special(a,b):
        if not (polymer[a] and polymer[b]) or residue_uids[a]==residue_uids[b]:
            return False
        if labels[a]==labels[b]:
            forward=order[(str(labels[b]),int(residue_uids[b]))]-order[(str(labels[a]),int(residue_uids[a]))]
            pair=(str(names[a]),str(names[b]))
            if modalities[a]==modalities[b]=="protein":
                if (forward==1 and pair==("C","N")) or (forward==-1 and pair==("N","C")):
                    return False
            elif modalities[a] in {"rna","dna"} and modalities[b] in {"rna","dna"}:
                if (forward==1 and pair==("O3'","P")) or (forward==-1 and pair==("P","O3'")):
                    return False
        return True

    if atoms.bonds is not None:
        for a,b,_ in atoms.bonds.as_array():
            if special(int(a),int(b)):
                return True
    for bond in getattr(record,"bond_constraints",()):
        if not bond.get("required_output",True):
            continue
        a,b=(_locator_rows(atoms,{k:v for k,v in bond[end].items()
                                if k not in {"atom_id","atom_uid","uid","atom_index","residue_uid","token_uid"}})
             for end in ("atom1","atom2"))
        if len(a)==len(b)==1 and special(a[0],b[0]):
            return True
    return False


def ring_closure(record,bond):
    from src.validate.inputs import _locator_rows
    a,b=(_locator_rows(record.atoms,{k:v for k,v in bond[end].items() if k not in {"atom_id","atom_uid","uid","atom_index","residue_uid","token_uid"}}) for end in ("atom1","atom2"))
    if len(a)!=1 or len(b)!=1:
        return False
    atoms=record.atoms
    labels=getattr(atoms,"label_asym_id",atoms.chain_id)
    chain=str(labels[a[0]])
    if chain!=str(labels[b[0]]) or chain not in record.cyclic_chains:
        return False
    groups=residues(atoms,chain)
    return ({(a[0],str(atoms.atom_name[a[0]])),(b[0],str(atoms.atom_name[b[0]]))}
            == {(next((i for i in groups[0] if str(atoms.atom_name[i])=="N"),-1),"N"),
                (next((i for i in groups[-1] if str(atoms.atom_name[i])=="C"),-1),"C")})


def _ring_edge(record,edge):
    (left,right)=edge
    chain=left[0]
    if chain!=right[0] or chain not in record.cyclic_chains:
        return False
    groups=residues(record.atoms,chain)
    first=position(record.atoms,groups[0][0])
    last=position(record.atoms,groups[-1][0])
    return {left,right}=={(chain,*first,"N"),(chain,*last,"C")}
