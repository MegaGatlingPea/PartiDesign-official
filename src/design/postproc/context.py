"""Owner-local authoritative CCD chemistry for retained molecular context.

Only compact requested components survive the source snapshot read.  These facts
are never model inputs, public artifacts, or guesses about an unknown ligand.
"""
from __future__ import annotations

import gc
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem

from partidesign.paths import community_asset
from ..utils.chemistry import TrustedCCDComponent, trusted_ccd_private_molecule
from ..utils.reader import read_structure

_CONTEXT_COMPONENTS: dict[str, TrustedCCDComponent] = {}
_REQUESTED: set[str] = set()
_SOURCE_CHARGES: dict[Path, dict[tuple, int]] = {}
_SOURCE_NORMALIZED_CHARGES: dict[Path, dict[tuple, int]] = {}
_SOURCE_COMPONENTS: dict[Path, dict[tuple, str]] = {}


def _asset_path():
    return community_asset("esmfold2", Path(__file__).resolve().parents[3]) / "common" / "ccd.pkl"


def _load_components(names):
    names = {str(name).strip().upper() for name in names} - _REQUESTED
    if not names:
        return
    path = _asset_path()
    if not path.is_file():
        # Absence is reported for each affected sample, rather than preventing
        # native protein/NA or explicitly authored design ligands from relaxing.
        _REQUESTED.update(names)
        return
    with path.open("rb") as stream:
        mapping = pickle.load(stream)
    if not isinstance(mapping, dict):
        raise ValueError("release CCD chemistry asset is not a component mapping")
    for name in names:
        molecule = mapping.get(name)
        if isinstance(molecule, Chem.Mol):
            _CONTEXT_COMPONENTS[name] = TrustedCCDComponent(
                name, bytes(molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)))
    _REQUESTED.update(names)
    del mapping
    gc.collect()


def _source_charge_facts(path):
    path = Path(path).resolve()
    if path in _SOURCE_CHARGES:
        return _SOURCE_CHARGES[path]
    if not path.is_file():
        return {}
    facts = {}
    if path.suffix.lower() in {".pdb", ".ent"}:
        # Blank PDB charge columns are unknown. Only explicit nonzero author
        # charges constrain a matched CCD protonation state.
        for line in path.read_text().splitlines():
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            value = line[78:80].strip()
            if not value:
                continue
            # Some structure exporters serialize a neutral charge as "0 ",
            # rather than leaving both PDB columns blank.  Its meaning is exact.
            if value in {"0", "0+", "0-"}:
                charge = 0
            elif len(value) == 2 and value[0].isdigit() and value[1] in "+-":
                charge = int(value[0]) * (1 if value[1] == "+" else -1)
            elif len(value) == 2 and value[0] in "+-" and value[1].isdigit():
                charge = int(value)
            else:
                raise ValueError(f"invalid explicit PDB formal charge in {path}")
            key = (line[21:22].strip(), int(line[22:26]), line[26:27].strip(),
                   line[17:20].strip(), line[12:16].strip())
            previous = facts.get(key)
            if previous is not None and previous != charge:
                raise ValueError("ambiguous source conformer formal charge")
            facts[key] = charge
    elif path.suffix.lower() in {".cif", ".mmcif"}:
        from biotite.structure.io.pdbx import CIFFile
        block = next(iter(CIFFile.read(path).values()))
        if "atom_site" in block and "pdbx_formal_charge" in block["atom_site"]:
            site = block["atom_site"]
            columns = {}
            for field, fallback in (("auth_asym_id", "label_asym_id"),
                                    ("auth_seq_id", "label_seq_id"),
                                    ("auth_comp_id", "label_comp_id"),
                                    ("auth_atom_id", "label_atom_id")):
                columns[field] = site[field if field in site else fallback].as_array(str)
            insertion = (site["pdbx_PDB_ins_code"].as_array(str) if "pdbx_PDB_ins_code" in site
                         else [""] * len(columns["auth_asym_id"]))
            for index, value in enumerate(site["pdbx_formal_charge"].as_array(str)):
                if value in {"", ".", "?"}:
                    continue
                ins = insertion[index]
                key = (columns["auth_asym_id"][index], int(columns["auth_seq_id"][index]),
                       "" if ins in {".", "?"} else ins,
                       columns["auth_comp_id"][index], columns["auth_atom_id"][index])
                previous = facts.get(key)
                charge = int(value)
                if previous is not None and previous != charge:
                    raise ValueError("ambiguous source conformer formal charge")
                facts[key] = charge
    _SOURCE_CHARGES[path] = facts
    return facts


def _source_key(atoms, row):
    def text(field, fallback):
        if field in atoms.get_annotation_categories():
            value = str(atoms.get_annotation(field)[row]).strip()
            if value not in {"", ".", "?"}:
                return value
        return fallback
    return (text("auth_asym_id", str(atoms.chain_id[row])),
            int(text("auth_seq_id", str(atoms.res_id[row]))),
            text("auth_ins_code", str(atoms.ins_code[row])),
            str(atoms.res_name[row]), str(atoms.atom_name[row]))



def source_formal_charge_facts(atoms, source_path, *, rows=None):
    """Freeze explicit input charge facts using emitted normalized identities."""
    if source_path is None:
        return ()
    path = Path(source_path).resolve()
    facts = _source_charge_facts(path)
    normalized = _SOURCE_NORMALIZED_CHARGES.get(path, {})
    values = []
    for row in (range(len(atoms)) if rows is None else rows):
        auth_key = _source_key(atoms, row)
        key = (str(atoms.chain_id[row]), int(atoms.res_id[row]),
               str(atoms.ins_code[row]), str(atoms.res_name[row]),
               str(atoms.atom_name[row]))
        original_name = _SOURCE_COMPONENTS.get(path, {}).get(key[:3], key[3])
        lookup = (*key[:3], original_name, key[4]) if key[3].startswith("LIG") else key
        author_lookup = (*auth_key[:3], original_name, auth_key[4]) if key[3].startswith("LIG") else auth_key
        if auth_key[:3] == key[:3] and lookup in normalized:
            charge = normalized[lookup]
        elif author_lookup in facts:
            charge = facts[author_lookup]
        else:
            continue
        values.append((key, int(charge)))
    return tuple(values)


def _apply_source_charges(molecule, atoms, rows, source_path, *, component_name=None, frozen_facts=()):
    frozen = dict(frozen_facts)
    if source_path is None:
        facts, normalized = {}, {}
    else:
        source_path = Path(source_path).resolve()
        facts = _source_charge_facts(source_path)
        normalized = _SOURCE_NORMALIZED_CHARGES.get(source_path, {})
    if not facts and not normalized and not frozen:
        return molecule
    result = Chem.Mol(molecule)
    declared = {}
    for index, row in enumerate(rows):
        key = _source_key(atoms, row)
        if component_name is not None:
            key = (*key[:3], component_name, key[4])
        normal_key = (str(atoms.chain_id[row]), int(atoms.res_id[row]),
                      str(atoms.ins_code[row]), key[3], str(atoms.atom_name[row]))
        # The public writer normalizes author numbering. For those files, use
        # the original input's normalized identity, never the changed author ID.
        public_key = (str(atoms.chain_id[row]), int(atoms.res_id[row]),
                      str(atoms.ins_code[row]), str(atoms.res_name[row]),
                      str(atoms.atom_name[row]))
        if public_key in frozen:
            declared[index] = frozen[public_key]
        elif key[:3] == normal_key[:3] and normal_key in normalized:
            declared[index] = normalized[normal_key]
        elif key in facts:
            declared[index] = facts[key]
    oxyanion_groups = {}
    for index, charge in declared.items():
        atom = result.GetAtomWithIdx(index)
        if atom.GetFormalCharge() == charge:
            continue
        neighbors = list(atom.GetNeighbors())
        if (charge == -1 and atom.GetSymbol() == "O" and len(neighbors) == 1
                and neighbors[0].GetSymbol() in {"P", "S"}):
            # PDB sources often label the double-bonded resonance oxygen as
            # negative, whereas CCD places that negative charge on an equivalent
            # single-bonded nonbridging oxygen. Preserve the public CCD orders
            # and honor the explicitly declared oxyanion charge-state count.
            oxyanion_groups.setdefault(neighbors[0].GetIdx(), set()).add(index)
            continue
        allowed = (charge == -1 and atom.GetFormalCharge() == 0
                   and atom.GetSymbol() == "O" and atom.GetTotalNumHs() == 1
                   and len(neighbors) == 1 and neighbors[0].GetSymbol() == "C"
                   and result.GetBondBetweenAtoms(index, neighbors[0].GetIdx()).GetBondType() == Chem.BondType.SINGLE
                   and any(b.GetBondType() == Chem.BondType.DOUBLE
                           and b.GetOtherAtom(neighbors[0]).GetSymbol() == "O"
                           for b in neighbors[0].GetBonds()))
        if not allowed:
            raise ValueError(f"explicit source formal charge contradicts unsupported CCD state: {_source_key(atoms, rows[index])}")
        atom.SetFormalCharge(-1)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    for center_index in oxyanion_groups:
        center = result.GetAtomWithIdx(center_index)
        terminal = [a for a in center.GetNeighbors()
                    if a.GetSymbol() == "O" and a.GetDegree() == 1]
        if center.GetDegree() != 4 or len(terminal) < 2:
            raise ValueError("source oxyanion charge does not match a tetrahedral phosphate/sulfate")
        requested = sum(declared.get(a.GetIdx()) == -1 for a in terminal)
        present = sum(a.GetFormalCharge() == -1 for a in terminal)
        targets = [a for a in terminal if a.GetFormalCharge() == 0
                   and a.GetTotalNumHs() == 1 and declared.get(a.GetIdx()) != 0
                   and result.GetBondBetweenAtoms(center_index, a.GetIdx()).GetBondType() == Chem.BondType.SINGLE]
        targets.sort(key=lambda a: (declared.get(a.GetIdx()) != -1, str(atoms.atom_name[rows[a.GetIdx()]])))
        if requested - present > len(targets):
            raise ValueError("explicit source oxyanion charge state cannot match the retained heavy graph")
        for atom in targets[:max(0, requested-present)]:
            atom.SetFormalCharge(-1)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
    Chem.SanitizeMol(result)
    return result


def preload_context_chemistry(tasks, ccd_path=None):
    """Read each relevant source once and extract needed CCD molecules once."""
    names = set()
    seen = set()
    for task in tasks:
        if not task.postprocess or not task.override or task.input_path is None:
            continue
        design = {chain.chain_id for chain in task.design_chains}
        key = (task.input_path, tuple(sorted(design)))
        if key in seen:
            continue
        seen.add(key)
        source = read_structure(task.input_path, ccd_path=ccd_path)
        facts = _source_charge_facts(task.input_path)
        path = Path(task.input_path).resolve()
        normalized = _SOURCE_NORMALIZED_CHARGES.setdefault(path, {})
        components = _SOURCE_COMPONENTS.setdefault(path, {})
        for row in range(len(source)):
            auth_key = _source_key(source, row)
            normal_key = (str(source.chain_id[row]), int(source.res_id[row]),
                          str(source.ins_code[row]), str(source.res_name[row]), str(source.atom_name[row]))
            if auth_key in facts:
                normalized[normal_key] = facts[auth_key]
            residue_key = normal_key[:3]
            name = str(source.res_name[row]).upper()
            previous = components.get(residue_key)
            if previous is not None and previous != name:
                raise ValueError("source normalized component identity is ambiguous")
            components[residue_key] = name
        for uid in np.unique(source.component_uid):
            rows = np.flatnonzero(source.component_uid == uid)
            if len(rows) <= 1 or not np.all(source.mol_type[rows] == "ligand"):
                continue
            if all(str(source.chain_id[row]) in design for row in rows):
                continue
            names.update(str(source.res_name[row]).upper() for row in rows)
    _load_components(names)



def context_component_name(atoms, rows, *, source_path=None):
    """Resolve a public ligand alias through the original normalized source."""
    names = {str(atoms.res_name[row]).strip().upper() for row in rows}
    if len(names) != 1:
        raise ValueError("retained molecular component has multiple chemical identities")
    name = next(iter(names))
    if source_path is not None and name.startswith("LIG"):
        mapping = _SOURCE_COMPONENTS.get(Path(source_path).resolve(), {})
        original = {mapping.get((str(atoms.chain_id[row]), int(atoms.res_id[row]),
                                 str(atoms.ins_code[row]))) for row in rows}
        original.discard(None)
        if len(original) == 1:
            return original.pop()
    return name


def context_private_molecule(atoms, rows, *, source_path=None):
    """Use one exact named CCD graph, including its formal charge and H state."""
    rows = tuple(rows)
    original_name = str(atoms.res_name[rows[0]]).strip().upper()
    name = context_component_name(atoms, rows, source_path=source_path)
    _load_components({name})
    component = _CONTEXT_COMPONENTS.get(name)
    if component is None:
        raise ValueError(f"retained context {name!r} lacks authoritative CCD chemistry")
    matched_atoms = atoms
    if name != original_name:
        matched_atoms = atoms.copy()
        matched_atoms.res_name[list(rows)] = name
    molecule = trusted_ccd_private_molecule(matched_atoms, rows, name, component)
    return _apply_source_charges(molecule, atoms, rows, source_path, component_name=name)


def clear_context_chemistry_cache():
    _CONTEXT_COMPONENTS.clear()
    _REQUESTED.clear()
    _SOURCE_CHARGES.clear()
    _SOURCE_NORMALIZED_CHARGES.clear()
    _SOURCE_COMPONENTS.clear()
