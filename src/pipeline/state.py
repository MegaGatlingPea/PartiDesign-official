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
from .selection import ChainSequence, LigandIdentity

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


def failed_row(row):
    """Explicit native failures may legitimately have no output filename."""
    return bool(row.get("error")) or any(
        str(row.get(field,"")).strip().lower() in {"failed","failure","error"}
        for field in ("status","generation_status"))

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
        if row.get("heavy_role"):
            from src.validate.inputs import _antibody_chain_metadata
            metadata,antibody_fixed=_antibody_chain_metadata(row,atoms,candidate.candidate_id)
            if tuple(item["chain"] for item in metadata)!=design:
                raise ValueError("antibody chain roles differ from design metadata")
            # The public antibody policy and the resolved native ledger must agree.
            fixed_positions={(x["chain"],int(x.get("res_id",x.get("resid",0)))) for x in fixed}
            for item in antibody_fixed:
                if (item["chain"],int(item["res_id"])) not in fixed_positions:
                    fixed.append(item)
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


@dataclass(frozen=True, slots=True)
class _PreparedChainSequences:
    owner: object
    chains: tuple[ChainSequence | LigandIdentity, ...]
    mutable_positions: tuple[tuple[int, ...], ...]


def prepare_chain_sequences(record):
    """Derive anchors once while this input's atoms and conditions stay unchanged."""
    from src.validate.sequences import _AA3_TO_1,_RNA3_TO_1,_DNA3_TO_1,_fixed_modified_map
    from src.validate.inputs import _locator_rows
    atoms=record.atoms
    modified,_=_fixed_modified_map(record)
    maps={"protein":_AA3_TO_1,"rna":_RNA3_TO_1,"dna":_DNA3_TO_1}
    fixed_rows={row for entry in record.fixed_restypes for row in _locator_rows(atoms,entry)}
    graph=special_edges(record)
    result=[]
    mutable_positions=[]
    for chain in record.design_chains:
        groups=residues(atoms,chain)
        if not groups:
            raise ValueError(f"empty design chain {chain!r}")
        modality=str(atoms.mol_type[groups[0][0]])
        if modality == "ligand":
            entries=getattr(record,"ligand_entries",{}) or {}
            entry=entries.get(chain,{})
            if not entry.get("smiles"):
                raise ValueError(f"design ligand {chain!r} lacks authoritative SMILES metadata")
            result.append(LigandIdentity(chain,str(entry["smiles"])))
            mutable_positions.append(())
            continue
        if modality not in maps:
            raise ValueError(f"unsupported pipeline design modality: {modality}")
        values,anchors,mutable=[],[],[]
        for ordinal,rows in enumerate(groups):
            i=rows[0]
            pos=position(atoms,i)
            name=str(atoms.res_name[i]).upper()
            uid=int(atoms.residue_uid[i])
            token=f"[{modified[uid]}]" if uid in modified else maps[modality].get(name,f"[{name.strip('[]')}]")
            if uid not in modified:
                mutable.append(ordinal)
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
        mutable_positions.append(tuple(mutable))
    return _PreparedChainSequences(record,tuple(result),tuple(mutable_positions))


def chain_sequences(record,variant=None,*,prepared=None):
    if prepared is None:
        prepared=prepare_chain_sequences(record)
    elif prepared.owner is not record:
        raise ValueError("prepared chain sequences belong to a different input record")
    proposals={} if variant is None else variant.get("sequences",{})
    result=[]
    for chain,mutable in zip(prepared.chains,prepared.mutable_positions):
        if isinstance(chain,LigandIdentity):
            proposal=proposals.get(chain.chain_id)
            if proposal is None:
                result.append(chain)
            else:
                smiles=proposal.get("smiles") if isinstance(proposal,Mapping) else proposal
                result.append(LigandIdentity(chain.chain_id,str(smiles)))
            continue
        if chain.chain_id not in proposals:
            result.append(chain)
            continue
        proposal=tokens(str(proposals[chain.chain_id]))
        if len(proposal)!=len(chain.tokens):
            raise ValueError(f"proposal changed chain {chain.chain_id!r} length")
        values=list(chain.tokens)
        for ordinal in mutable:
            values[ordinal]=proposal[ordinal]
        result.append(replace(chain,tokens=tuple(values)))
    return tuple(result)

def refreshed_row(record,path,*,atoms=None):
    import numpy as np
    from src.design.utils.reader import read_structure
    atoms=_prepared_atoms(read_structure(path) if atoms is None else atoms,record.design_chains)
    current=replace(record,atoms=atoms,fixed_restypes=[{k:v for k,v in x.items() if k!="atom_id"} for x in record.fixed_restypes])
    chains=chain_sequences(current)
    row=dict(record.row)
    previous={x["chain"]:x for x in json.loads(row.get("design_chains") or "[]")}
    metadata=[]
    labels=np.asarray(getattr(atoms,"label_asym_id",atoms.chain_id)).astype(str)
    for c in chains:
        entry={**previous.get(c.chain_id,{}),"chain":c.chain_id,"modality":c.modality}
        if isinstance(c,LigandIdentity):
            selected=labels==c.chain_id
            # The source row owns chemistry; a refold changes coordinates only.
            # New FM children already supply their own native source row.
            entry.update(sequence=previous[c.chain_id]["sequence"],
                         residue_count=len(residues(atoms,c.chain_id)),
                         atom_count=int(np.count_nonzero(selected)),
                         heavy_atom_count=int(np.count_nonzero(selected &
                             ~np.isin(np.char.upper(atoms.element.astype(str)),("H","D","T")))))
        else:
            entry.update(sequence="".join(c.tokens),residue_count=len(c.tokens))
            if c.chain_id in record.cyclic_offset_chains:
                entry["cyclic_offset"]=True
        metadata.append(entry)
    row["design_chains"]=json.dumps(metadata,separators=(",",":"))
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
    if row.get("heavy_role"):
        from src.design.antibody.results import refresh_antibody_result
        context=row.get("_antibody_context")
        if isinstance(context,str):
            context=json.loads(context)
        if context is None and row.get("_antibody_context_path"):
            context=json.loads(Path(row["_antibody_context_path"]).read_text())
        if context is None:
            raise ValueError("antibody result is missing its native region/source context")
        row.update(refresh_antibody_result(row,atoms,context))
        row["filename"]=str(path)
        if row.get("_antibody_context_path"):
            row.pop("_antibody_context",None)
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


@dataclass(frozen=True, slots=True)
class _ContextBaseline:
    atom_keys: frozenset
    residue_keys: frozenset
    optional_oxt: frozenset


def _prepare_context(record):
    original=frozenset(_context_atom_key(record.atoms,i) for i in _context_rows(record))
    optional_oxt=set()
    for chain in {key[0] for key in original if key[4:]==("OXT","O")}:
        terminal=residues(record.atoms,chain)[-1]
        pos=position(record.atoms,terminal[0])
        if (record.atoms.mol_type[terminal[0]]=="protein" and
                "OXT" not in record.tip_positions.get(chain,{}).get(pos,())):
            optional_oxt.update(key for i in terminal
                if (key:=_context_atom_key(record.atoms,i))[4:]==("OXT","O"))
    return _ContextBaseline(original,frozenset(key[:4] for key in original),
                            frozenset(optional_oxt))


def _assert_context(parent, child, *, prepared=None):
    # Native refolding owns canonical polymer completion. Preserve known
    # source facts without rejecting newly completed standard terminal atoms.
    baseline=_prepare_context(parent) if prepared is None else prepared
    predicted={_context_atom_key(child.atoms,i) for i in _context_rows(child)}
    missing=baseline.atom_keys-predicted
    if missing:
        present_names={key[:5] for key in predicted}
        optional_oxt={key for key in baseline.optional_oxt if key[:5] not in present_names}
        if not missing.issubset(optional_oxt):
            raise ValueError("refold changed context atom identity: missing or replaced source atom")
    if baseline.residue_keys!={key[:4] for key in predicted}:
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


def _protected_ligand_chains(record,chains,labels):
    from src.validate.ligands import protected_ligand_chains
    return protected_ligand_chains(record,chains,labels)


def restore_context_aliases(parent_record,target_cif,*,atoms=None):
    """Restore public ligand aliases using the native chemical mapping."""
    import numpy as np
    from src.design.utils.reader import read_structure
    from src.validate.ligands import resolve_context_ligands
    from src.validate.ligand_metrics import atom_mapping
    source=parent_record.atoms
    labels=np.asarray(getattr(source,"label_asym_id",source.chain_id)).astype(str)
    chains=list(dict.fromkeys(labels[source.mol_type=="ligand"].tolist()))
    if not chains:
        return atoms
    target=read_structure(target_cif) if atoms is None else atoms
    context=[chain for chain in chains if chain not in parent_record.design_chains]
    entries=(resolve_context_ligands(parent_record.task_dir,source,
        design_chains=parent_record.design_chains,chains=context)[0] if context else {})
    designed=getattr(parent_record,"ligand_entries",{}) or {}
    for chain in chains:
        if chain in parent_record.design_chains:
            if not designed.get(chain,{}).get("smiles"):
                raise ValueError(f"design ligand {chain!r} lacks authoritative SMILES metadata")
            entries[chain]=designed[chain]
    protected=_protected_ligand_chains(parent_record,
        [chain for chain in chains if chain in parent_record.design_chains],labels)
    replacements={}
    for chain in chains:
        try:
            mapping=atom_mapping(source,target,chain,chain,trusted_smiles=entries[chain].get("smiles"),
                allow_symmetric_graph=chain in parent_record.design_chains and chain not in protected)
        except ValueError as exc:
            if chain in protected and "ambiguous" in str(exc):
                raise ValueError(f"protected design ligand {chain!r} atom correspondence is ambiguous") from exc
            raise
        replacements.update({b:str(source.atom_name[a]) for a,b in mapping.items()
                             if str(source.atom_name[a])!=str(target.atom_name[b])})
    if replacements:
        _rewrite_cif_aliases(Path(target_cif),target,replacements)
    return target


@dataclass(frozen=True, slots=True)
class _InvariantBaseline:
    owner: object
    design_lengths: tuple[int, ...]
    fixed_names: tuple[frozenset, ...]
    motif_names: tuple[tuple[str, tuple, frozenset], ...]
    special_edges: tuple
    context: _ContextBaseline
    ligand_identity: tuple


def _ligand_invariants(record):
    """Refolding must preserve each designed ligand's authoritative graph."""
    from .selection import sequence_key
    atoms=record.atoms
    entries=getattr(record,"ligand_entries",{}) or {}
    labels=getattr(atoms,"label_asym_id",atoms.chain_id)
    result=[]
    for chain in record.design_chains:
        rows=[i for i in range(len(atoms)) if str(labels[i])==chain
              and str(atoms.mol_type[i])=="ligand"]
        if not rows:
            continue
        if not entries.get(chain,{}).get("smiles"):
            raise ValueError(f"design ligand {chain!r} lacks authoritative SMILES metadata")
        identity=sequence_key((LigandIdentity(chain,entries[chain]["smiles"]),))
        locators={i:(str(atoms.atom_name[i]),str(atoms.element[i])) for i in rows}
        charges=getattr(atoms,"formal_charge",None)
        known=getattr(atoms,"formal_charge_known",None)
        inventory=tuple(sorted((locators[i],None if charges is None or charges[i] is None or
            (known is not None and not known[i]) else int(charges[i])) for i in rows))
        # Compare aromatic/Kekule witnesses in the trusted molecule's graph;
        # never change the stored source or prediction bond representation.
        from src.design.rewards.af3_chemistry import molecule_graph, trusted_smiles
        trusted=molecule_graph(trusted_smiles(entries[chain]["smiles"]))
        graph=None
        if any(order==9 for _,_,order in trusted.bonds):
            from src.validate.ligand_metrics import _heavy_indices, _trusted_smiles_graphs
            heavy,_=_heavy_indices(atoms,chain)
            graph,_=_trusted_smiles_graphs(atoms,atoms,heavy,heavy,entries[chain]["smiles"],
                                           comparison_only=True)
        edges=tuple(sorted((tuple(sorted((locators[int(a)],locators[int(b)]))),
                            int(graph[int(a)][int(b)]) if graph is not None and int(a) in graph and int(b) in graph
                            else int(order))
                           for a,b,order in ([] if atoms.bonds is None else atoms.bonds.as_array())
                           if int(a) in locators and int(b) in locators))
        result.append((identity,inventory,edges))
    return tuple(result)


def _same_ligand_invariants(original,current):
    if len(original)!=len(current):
        return False
    for (identity,inventory,edges),(new_identity,new_inventory,new_edges) in zip(original,current):
        old_atoms,new_atoms=dict(inventory),dict(new_inventory)
        if identity!=new_identity or edges!=new_edges or old_atoms.keys()!=new_atoms.keys():
            return False
        # Missing source charge annotation is not a zero-charge condition.
        if any(charge is not None and new_atoms[atom]!=charge for atom,charge in old_atoms.items()):
            return False
    return True


def _residue_identity_index(atoms,chain):
    result={}
    for rows in residues(atoms,chain):
        result.setdefault(position(atoms,rows[0]),set()).add(str(atoms.res_name[rows[0]]))
    return result


def prepare_invariants(parent):
    """Snapshot one unchanged parent's identities; derive every child afresh.

    The caller owns this value's lifetime and must prepare again after replacing
    the input record or changing its atom inventory, graph, or conditions.
    """
    from src.validate.inputs import _locator_rows
    fixed=tuple(frozenset(parent.atoms.res_name[_locator_rows(parent.atoms,entry)])
                for entry in parent.fixed_restypes)
    motifs=[]
    for chain,positions in parent.motif_positions.items():
        identities=_residue_identity_index(parent.atoms,chain)
        motifs.extend((chain,pos,frozenset(identities.get(pos,())))
                      for pos in positions)
    return _InvariantBaseline(parent,
        tuple(len(residues(parent.atoms,chain)) for chain in parent.design_chains),
        fixed,tuple(motifs),special_edges(parent),_prepare_context(parent),_ligand_invariants(parent))


def assert_invariants(parent,child,*,prepared=None):
    from src.validate.inputs import _locator_rows
    if prepared is not None and prepared.owner is not parent:
        raise ValueError("prepared invariants belong to a different input record")
    if parent.design_chains!=child.design_chains:
        raise ValueError("refold changed design chain roles")
    ligand_identity=_ligand_invariants(parent) if prepared is None else prepared.ligand_identity
    if not _same_ligand_invariants(ligand_identity,_ligand_invariants(child)):
        raise ValueError("refold changed designed ligand SMILES or atom graph")
    for index,chain in enumerate(parent.design_chains):
        length=(len(residues(parent.atoms,chain)) if prepared is None
                else prepared.design_lengths[index])
        if length!=len(residues(child.atoms,chain)):
            raise ValueError(f"refold changed chain {chain!r} length")
    if parent.cyclic_offset_chains!=child.cyclic_offset_chains:
        raise ValueError("cyclic_offset intent was lost")
    for index,entry in enumerate(parent.fixed_restypes):
        names=(frozenset(parent.atoms.res_name[_locator_rows(parent.atoms,entry)])
               if prepared is None else prepared.fixed_names[index])
        rows=_locator_rows(child.atoms,{k:v for k,v in entry.items() if k!="atom_id"})
        if not names or not rows or names!=set(child.atoms.res_name[rows]):
            raise ValueError(f"fixed residue identity changed: {entry}")
    if prepared is None:
        motifs=[]
        for chain,positions in parent.motif_positions.items():
            identities=_residue_identity_index(parent.atoms,chain)
            motifs.extend((chain,pos,frozenset(identities.get(pos,()))) for pos in positions)
    else:
        motifs=prepared.motif_names
    child_identities={}
    for chain,pos,names in motifs:
        if chain not in child_identities:
            child_identities[chain]=_residue_identity_index(child.atoms,chain)
        if not names or names!=child_identities[chain].get(pos,set()):
            raise ValueError(f"motif identity changed: {chain}:{pos}")
    if parent.bond_constraints:
        child_edges={frozenset((int(i),int(j))) for i,j,_ in child.atoms.bonds.as_array()}
        for bond in parent.bond_constraints:
            a,b=(_locator_rows(child.atoms,{k:v for k,v in bond[end].items() if k!="atom_id"})
                 for end in ("atom1","atom2"))
            if len(a)!=1 or len(b)!=1 or frozenset((a[0],b[0])) not in child_edges:
                raise ValueError(f"required chemical connection missing: {bond}")
    original_edges=special_edges(parent) if prepared is None else prepared.special_edges
    if original_edges!=special_edges(child):
        raise ValueError("refold changed the required special-bond graph")
    _assert_context(parent,child,prepared=None if prepared is None else prepared.context)



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
