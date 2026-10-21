"""Recover a private postprocess candidate from an existing public CIF."""
from __future__ import annotations
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import biotite.structure as struc
import numpy as np
from partidesign.paths import community_asset
from rdkit import Chem
from ..core.records import ExpandedRequest, SearchCandidate
from ..utils.chemistry import (
    load_trusted_ccd_components,
    private_molecule_from_authoring_graph,
    private_molecule_from_trusted_graph,
    project_trusted_ccd_component,
    trusted_ccd_private_molecule,
)
from ..utils.reader import read_structure
from .runner import PostprocessPreparationError


class ExistingOutputRecoveryError(PostprocessPreparationError):
    """Public output ownership or identity cannot be recovered safely."""


_BIO_ORDER = {
    int(struc.BondType.SINGLE): "single",
    int(struc.BondType.DOUBLE): "double",
    int(struc.BondType.TRIPLE): "triple",
    int(struc.BondType.AROMATIC): "aromatic",
    int(struc.BondType.AROMATIC_SINGLE): "aromatic",
    int(struc.BondType.AROMATIC_DOUBLE): "aromatic",
    int(struc.BondType.AROMATIC_TRIPLE): "aromatic",
}
_RD_ORDER = {
    Chem.BondType.SINGLE: "single",
    Chem.BondType.DOUBLE: "double",
    Chem.BondType.TRIPLE: "triple",
    Chem.BondType.AROMATIC: "aromatic",
}
_AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

_DESIGN_CHAIN_MODALITIES = frozenset({"protein", "rna", "dna", "ligand"})
_PEPTIDE_SUB_MODALITY = "peptide"


def _value(row: Any, key: str, default: Any = None) -> Any:
    return row.get(key, default) if isinstance(row, Mapping) else getattr(row, key, default)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _positive(value: Any, where: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ExistingOutputRecoveryError(f"{where} must be a positive integer") from exc
    if isinstance(value, bool) or result < 1:
        raise ExistingOutputRecoveryError(f"{where} must be a positive integer")
    return result


def _json(value: Any, where: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ExistingOutputRecoveryError(f"{where} is not valid JSON") from exc


def _normal_entries(row: Any) -> list[dict[str, Any]]:
    raw = _json(_value(row, "design_chains"), "design_chains")
    if not isinstance(raw, list) or not raw:
        raise ExistingOutputRecoveryError("normal result row lacks design_chains metadata")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ExistingOutputRecoveryError(f"design_chains[{index}] must be an object")
        missing = {"chain", "modality", "sequence", "residue_count"}.difference(item)
        if missing:
            raise ExistingOutputRecoveryError(f"design_chains[{index}] missing {sorted(missing)}")
        chain = _text(item["chain"])
        modality = _text(item["modality"]).lower()
        sequence = _text(item["sequence"])
        has_sub_modality = "sub_modality" in item
        sub_modality = item.get("sub_modality")
        if has_sub_modality and (
            not isinstance(sub_modality, str) or sub_modality != _PEPTIDE_SUB_MODALITY
        ):
            raise ExistingOutputRecoveryError(
                f"design_chains[{index}].sub_modality must be peptide when present"
            )
        if has_sub_modality and modality != "protein":
            raise ExistingOutputRecoveryError(
                f"design_chains[{index}].sub_modality=peptide requires modality=protein"
            )
        if not chain or chain in seen or modality not in _DESIGN_CHAIN_MODALITIES or not sequence:
            raise ExistingOutputRecoveryError(f"design_chains[{index}] has invalid ownership metadata")
        seen.add(chain)
        normalized = {
            "chain": chain,
            "modality": modality,
            "sequence": sequence,
            "residue_count": _positive(item["residue_count"], f"design_chains[{index}].residue_count"),
        }
        if has_sub_modality:
            normalized["sub_modality"] = sub_modality
        if "cyclic_offset" in item:
            if (
                modality != "protein" or type(item["cyclic_offset"]) is not bool
                or (item["cyclic_offset"] and normalized["residue_count"] < 3)
            ):
                raise ExistingOutputRecoveryError(f"design_chains[{index}].cyclic_offset is invalid")
            normalized["cyclic_offset"] = item["cyclic_offset"]
        result.append(normalized)
    return result


def _labels(atoms: Any) -> np.ndarray:
    categories = set(atoms.get_annotation_categories())
    return np.asarray(atoms.label_asym_id if "label_asym_id" in categories else atoms.chain_id).astype(str)


def _rows(atoms: Any, label: str) -> np.ndarray:
    result = np.flatnonzero(_labels(atoms) == str(label))
    if not len(result):
        raise ExistingOutputRecoveryError(f"ledger chain {label!r} is absent from the CIF")
    return result


def _count(atoms: Any, rows: Sequence[int], modality: str) -> int:
    rows = np.asarray(rows, dtype=np.int64)
    if modality == "ligand":
        uid = np.asarray(atoms.component_uid, dtype=np.int64)
        return len({(int(uid[r]), int(atoms.res_id[r]), str(atoms.ins_code[r])) for r in rows})
    return len({(int(atoms.res_id[r]), str(atoms.ins_code[r])) for r in rows})


def _normal_ownership(atoms: Any, task: Any, entries: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    declared = list(task.design_chains)
    if len(declared) != len(entries):
        raise ExistingOutputRecoveryError("design-chain count disagrees with task")
    if sorted(str(c.modality) for c in declared) != sorted(str(e["modality"]) for e in entries):
        raise ExistingOutputRecoveryError("design-chain modalities disagree with task")
    unused = list(entries)
    mapped: dict[str, Mapping[str, Any]] = {}

    def _matches_sub_modality(chain: Any, entry: Mapping[str, Any]) -> bool:
        expected = getattr(chain, "sub_modality", None)
        actual = entry.get("sub_modality")
        return actual is None or actual == expected

    for chain in declared:
        hit = [
            e for e in unused
            if str(e["chain"]) == str(chain.chain_id)
            and str(e["modality"]) == str(chain.modality)
            and _matches_sub_modality(chain, e)
        ]
        if len(hit) == 1:
            mapped[str(chain.chain_id)] = hit[0]
            unused.remove(hit[0])
    for chain in declared:
        chain_id = str(chain.chain_id)
        if chain_id in mapped:
            continue
        hit = [
            e for e in unused
            if str(e["modality"]) == str(chain.modality)
            and _matches_sub_modality(chain, e)
        ]
        pending = sum(
            str(c.modality) == str(chain.modality)
            and str(c.chain_id) not in mapped
            for c in declared
        )
        if len(hit) != 1 or pending != 1:
            raise ExistingOutputRecoveryError(f"cannot prove ownership of task chain {chain_id!r}")
        mapped[chain_id] = hit[0]
        unused.remove(hit[0])
    if unused:
        raise ExistingOutputRecoveryError("design-chain ownership is incomplete")
    atoms.is_design[:] = False
    for chain in declared:
        item = mapped[str(chain.chain_id)]
        chain_rows = _rows(atoms, str(item["chain"]))
        modality = str(item["modality"])
        if _count(atoms, chain_rows, modality) != int(item["residue_count"]):
            raise ExistingOutputRecoveryError(f"ledger residue_count disagrees with chain {item['chain']!r}")
        observed = set(np.asarray(atoms.mol_type)[chain_rows].astype(str))
        if modality == "ligand":
            atoms.mol_type[chain_rows] = "ligand"
        elif observed != {modality}:
            raise ExistingOutputRecoveryError(f"chain {item['chain']!r} is not modality {modality!r}")
        atoms.is_design[chain_rows] = True
    return mapped


def _antibody_ownership(atoms: Any, task: Any, row: Any) -> dict[str, Mapping[str, Any]]:
    spec = getattr(task, "antibody", None)
    designs = list(getattr(spec, "chains", ())) if spec is not None else []
    if not designs:
        raise ExistingOutputRecoveryError("antibody task has no role-chain specification")
    heavy_role = _text(_value(row, "heavy_role")).upper()
    heavy_chain = _text(_value(row, "heavy_chain"))
    if heavy_role not in {"VH", "VHH"} or not heavy_chain:
        raise ExistingOutputRecoveryError("antibody heavy ownership fields are incomplete")
    role_map = {str(d.role).upper(): d for d in designs}
    heavy = role_map.get(heavy_role) or role_map.get("VH")
    if heavy is None or str(heavy.chain_id) != heavy_chain:
        raise ExistingOutputRecoveryError("antibody heavy chain disagrees with task roles")
    light_chain = _text(_value(row, "light_chain"))
    light = role_map.get("VL")
    if light is not None and str(light.chain_id) != light_chain:
        raise ExistingOutputRecoveryError("antibody light chain disagrees with task roles")
    if light is None and light_chain:
        raise ExistingOutputRecoveryError("antibody row declares an unexpected light chain")
    result: dict[str, Mapping[str, Any]] = {}
    for design in designs:
        chain_id, role = str(design.chain_id), str(design.role).upper()
        chain_rows = _rows(atoms, chain_id)
        if set(np.asarray(atoms.mol_type)[chain_rows].astype(str)) != {"protein"}:
            raise ExistingOutputRecoveryError(f"antibody chain {chain_id!r} is not protein")
        result[chain_id] = {"chain": chain_id, "modality": "protein", "role": role}
    atoms.is_design[:] = False
    for item in result.values():
        atoms.is_design[_rows(atoms, str(item["chain"]))] = True
    return result


def _request(task: Any, row: Any) -> ExpandedRequest:
    filename = _text(_value(row, "filename"))
    if not filename:
        raise ExistingOutputRecoveryError("result row has no filename")
    raw_id = _value(row, "base_sample_id")
    if raw_id in (None, ""):
        match = re.match(r"sample([0-9]+)", Path(filename).stem)
        raw_id = match.group(1) if match else None
    base_id = _positive(raw_id, "base_sample_id")
    plans = tuple(getattr(task, "length_plans", ()))
    if not plans:
        raise ExistingOutputRecoveryError("task has no length plan")
    slot = base_id - 1
    sample_num = _positive(getattr(task, "sample_num", 1), "task.sample_num")
    plan_index = slot // sample_num
    if plan_index >= len(plans):
        raise ExistingOutputRecoveryError(
            f"base_sample_id {base_id} is outside the task expansion"
        )
    return ExpandedRequest(
        task=task,
        plan=plans[plan_index],
        sample_id=filename,
        task_slot_ordinal=slot,
        global_ordinal=slot,
    )


def _condition(task: Any, chain_id: str) -> Mapping[str, Any]:
    condition = getattr(task, "condition", {})
    values = condition.get("ligand", {}) if isinstance(condition, Mapping) else {}
    item = values.get(chain_id, {}) if isinstance(values, Mapping) else {}
    return item if isinstance(item, Mapping) else {}


def _mode(task: Any, chain_id: str) -> str:
    item = _condition(task, chain_id)
    mode = _text(item.get("mode")).lower()
    if mode:
        return mode
    condition = getattr(task, "condition", {})
    if isinstance(condition, Mapping) and chain_id in set(condition.get("backbone", ()) or ()):
        return "backbone"
    return "free"


def _order(value: int) -> str:
    try:
        return _BIO_ORDER[int(value)]
    except KeyError as exc:
        raise ValueError(f"unsupported public bond order {value}") from exc


def _component_edges(atoms: Any, rows: Sequence[int]) -> set[tuple[str, str, str]]:
    row_set = {int(r) for r in rows}
    names = {int(r): str(atoms.atom_name[int(r)]).strip() for r in rows}
    if not all(names.values()) or len(set(names.values())) != len(names):
        raise ValueError("ligand atom names are missing or ambiguous")
    if atoms.bonds is None:
        raise ValueError("existing ligand has no public graph")
    result: set[tuple[str, str, str]] = set()
    for raw_i, raw_j, raw_order in atoms.bonds.as_array():
        i, j = int(raw_i), int(raw_j)
        if i in row_set and j in row_set:
            result.add((*sorted((names[i], names[j])), _order(int(raw_order))))
    return result


def _rd_edges(molecule: Chem.Mol) -> set[tuple[int, int, str]]:
    return {(min(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())),
             max(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())), _RD_ORDER[b.GetBondType()])
            for b in molecule.GetBonds()}


def _smiles(value: Any) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(_text(value)) if _text(value) else None
    if molecule is None:
        raise ValueError("persisted ligand SMILES is invalid or empty")
    molecule = Chem.RemoveHs(molecule, sanitize=True)
    if not molecule.GetNumAtoms() or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("persisted ligand SMILES is empty or disconnected")
    return molecule


def _smiles_private(atoms: Any, rows: Sequence[int], value: Any) -> Chem.Mol:
    source = _smiles(value)
    rows = np.asarray(rows, dtype=np.int64)
    names = [str(atoms.atom_name[int(r)]).strip() for r in rows]
    expected = [f"A{i + 1}" for i in range(len(rows))]
    if len(rows) != source.GetNumAtoms() or set(names) != set(expected):
        raise ValueError("SMILES output lacks the compiler A1... atom ledger")
    by_name = {name: int(row) for name, row in zip(names, rows)}
    ordered_rows = np.asarray([by_name[name] for name in expected], dtype=np.int64)
    if [str(atoms.element[int(r)]).strip().upper() for r in ordered_rows] != [a.GetSymbol().upper() for a in source.GetAtoms()]:
        raise ValueError("SMILES output elements disagree with task SMILES")
    expected_edges = {(*sorted((f"A{i + 1}", f"A{j + 1}")), order) for i, j, order in _rd_edges(source)}
    if _component_edges(atoms, ordered_rows) != expected_edges:
        raise ValueError("SMILES output graph disagrees with task SMILES")
    return private_molecule_from_trusted_graph(atoms, ordered_rows, source)


def _target_graph(atoms: Any, rows: Sequence[int]) -> Chem.Mol:
    rows = np.asarray(rows, dtype=np.int64)
    table, editable = Chem.GetPeriodicTable(), Chem.RWMol()
    row_to_local: dict[int, int] = {}
    for row in rows:
        symbol = str(atoms.element[int(row)]).strip()
        number = int(table.GetAtomicNumber(symbol[:1].upper() + symbol[1:].lower()))
        if number <= 1:
            raise ValueError(f"unsupported ligand element {symbol!r}")
        row_to_local[int(row)] = int(editable.AddAtom(Chem.Atom(number)))
    if atoms.bonds is None:
        raise ValueError("existing ligand has no public graph")
    bond_types = {"single": Chem.BondType.SINGLE, "double": Chem.BondType.DOUBLE,
                  "triple": Chem.BondType.TRIPLE, "aromatic": Chem.BondType.AROMATIC}
    for raw_i, raw_j, raw_order in atoms.bonds.as_array():
        i, j = int(raw_i), int(raw_j)
        if i not in row_to_local or j not in row_to_local:
            continue
        order = _order(int(raw_order))
        editable.AddBond(row_to_local[i], row_to_local[j], bond_types[order])
        if order == "aromatic":
            editable.GetAtomWithIdx(row_to_local[i]).SetIsAromatic(True)
            editable.GetAtomWithIdx(row_to_local[j]).SetIsAromatic(True)
    target = editable.GetMol()
    try:
        Chem.SanitizeMol(target)
    except Exception as exc:
        raise ValueError("existing ligand graph cannot be sanitized") from exc
    return target


def _mappings(source: Chem.Mol, target: Chem.Mol) -> list[tuple[int, ...]]:
    if source.GetNumAtoms() != target.GetNumAtoms():
        return []
    source_adj: dict[int, dict[int, str]] = {i: {} for i in range(source.GetNumAtoms())}
    target_adj: dict[int, dict[int, str]] = {i: {} for i in range(target.GetNumAtoms())}
    for bond in source.GetBonds():
        i, j = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        source_adj[i][j] = source_adj[j][i] = _RD_ORDER[bond.GetBondType()]
    for bond in target.GetBonds():
        i, j = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        target_adj[i][j] = target_adj[j][i] = _RD_ORDER[bond.GetBondType()]
    candidates = {
        i: [j for j in range(target.GetNumAtoms())
            if source.GetAtomWithIdx(i).GetAtomicNum() == target.GetAtomWithIdx(j).GetAtomicNum()
            and bool(source.GetAtomWithIdx(i).GetIsAromatic()) == bool(target.GetAtomWithIdx(j).GetIsAromatic())
            and len(source_adj[i]) == len(target_adj[j])]
        for i in range(source.GetNumAtoms())
    }
    if any(not values for values in candidates.values()):
        return []
    answer: list[tuple[int, ...]] = []
    mapping: dict[int, int] = {}
    used: set[int] = set()

    def visit() -> None:
        if len(answer) >= 4096:
            raise ValueError("ligand graph has too many isomorphisms to audit charge/H transfer")
        if len(mapping) == source.GetNumAtoms():
            answer.append(tuple(mapping[i] for i in range(source.GetNumAtoms())))
            return
        remaining = [i for i in range(source.GetNumAtoms()) if i not in mapping]
        source_i = min(remaining, key=lambda i: (-sum(n in mapping for n in source_adj[i]), -len(source_adj[i]), i))
        for target_j in candidates[source_i]:
            if target_j in used:
                continue
            if any(n in mapping and target_adj[target_j].get(mapping[n]) != order for n, order in source_adj[source_i].items()):
                continue
            mapping[source_i] = target_j
            used.add(target_j)
            visit()
            used.remove(target_j)
            del mapping[source_i]

    visit()
    return answer


def _generated_private(atoms: Any, rows: Sequence[int], value: Any) -> Chem.Mol:
    source = _smiles(value)
    target = _target_graph(atoms, rows)
    mappings = _mappings(source, target)
    if not mappings:
        raise ValueError("persisted SMILES has no true element/edge/order isomorphism to CIF")
    rows = np.asarray(rows, dtype=np.int64)
    selected: Chem.Mol | None = None
    signature: str | None = None
    for mapping in mappings:
        charges, hydrogens = [0] * len(rows), [0] * len(rows)
        for source_i, target_i in enumerate(mapping):
            atom = source.GetAtomWithIdx(source_i)
            charges[target_i], hydrogens[target_i] = int(atom.GetFormalCharge()), int(atom.GetTotalNumHs())
        try:
            molecule = private_molecule_from_authoring_graph(atoms, rows, charges, trusted_hydrogen_counts=hydrogens)
            current = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        except Exception:
            continue
        if selected is None:
            selected, signature = molecule, current
        elif current != signature:
            raise ValueError("symmetric mappings transfer non-equivalent charge/H chemistry")
    if selected is None:
        raise ValueError("persisted SMILES cannot be sanitized on the CIF graph")
    return selected


def _reader_ccd_path(path: Path | None) -> Path | None:
    # The release private asset is a pickle; the structure reader accepts its
    # compact topology JSON or an input CCD CIF, but never the private pickle.
    if path is not None and Path(path).suffix.lower() in {".pkl", ".pickle"}:
        return None
    return path


def _ccd_path(path: Path | None) -> Path:
    if path is not None and Path(path).suffix.lower() in {".pkl", ".pickle"}:
        return Path(path)
    return community_asset("esmfold2", Path(__file__).resolve().parents[3]) / "common" / "ccd.pkl"


def _source_component(task: Any, chain_id: str, ccd_path: Path | None) -> str:
    source_path = getattr(task, "input_path", None)
    if source_path is None:
        raise ValueError("type_bond recovery requires the original task input")
    source = read_structure(Path(source_path), ccd_path=_reader_ccd_path(ccd_path))
    rows = np.flatnonzero(np.asarray(source.chain_id).astype(str) == chain_id)
    if not len(rows) or set(np.asarray(source.mol_type)[rows].astype(str)) != {"ligand"}:
        raise ValueError(f"task input has no ligand on chain {chain_id!r}")
    names = {str(v).strip().upper() for v in np.asarray(source.res_name)[rows]}
    if len(names) != 1 or not next(iter(names)):
        raise ValueError("task input ligand component identity is ambiguous")
    return next(iter(names))


def _trusted_omitted_atom_names(atoms: Any, rows: Sequence[int], task: Any, chain_id: str) -> tuple[str, ...]:
    """Recover only explicit bond-local omissions on the original task chain."""
    condition = getattr(task, "condition", {})
    declarations = condition.get("bond", []) if isinstance(condition, Mapping) else []
    if not isinstance(declarations, list):
        return ()
    rows = np.asarray(rows, dtype=np.int64)
    retained_names = set(np.asarray(atoms.atom_name)[rows].astype(str))
    labels = _labels(atoms)
    chain_names = np.asarray(atoms.atom_name)[labels == labels[int(rows[0])]].astype(str)
    omitted: set[str] = set()
    endpoint_names: set[str] = set()
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            continue
        endpoints = [value for key in ("atom1", "atom2")
                     if isinstance(value := declaration.get(key), Mapping)
                     and value.get("chain") == chain_id]
        endpoint_names.update(value["atom"] for value in endpoints
                              if isinstance(value.get("atom"), str))
        raw_omit = declaration.get("omit_atoms", [])
        selected = [value for value in raw_omit
                    if isinstance(value, Mapping) and value.get("chain") == chain_id] if isinstance(raw_omit, list) else []
        if not selected:
            continue
        if not endpoints:
            raise ValueError("ligand omission must belong to a bond endpoint component")
        for selector in (*endpoints, *selected):
            if set(selector) != {"chain", "atom"} or not isinstance(selector.get("atom"), str) or not selector["atom"]:
                raise ValueError("ligand omission and endpoint selectors require exact chain/atom names")
        if any(value["atom"] not in retained_names
               or np.count_nonzero(chain_names == value["atom"]) != 1 for value in endpoints):
            raise ValueError("ligand omission lacks a unique retained endpoint in this component")
        omitted.update(value["atom"] for value in selected)
    if omitted.intersection(endpoint_names):
        raise ValueError("ligand omission cannot remove a declared bond endpoint")
    return tuple(sorted(omitted))


def _trusted_private(atoms: Any, rows: Sequence[int], task: Any, chain_id: str, mode: str, ccd_path: Path | None) -> Chem.Mol:
    item = _condition(task, chain_id)
    component_id = _text(item.get("ccd")).upper() if mode == "ccd" else _source_component(task, chain_id, ccd_path)
    if not component_id:
        raise ValueError("trusted CCD component identity is empty")
    components = load_trusted_ccd_components(_ccd_path(ccd_path), {component_id})
    component = components.get(component_id)
    if component is None:
        raise ValueError(f"release CCD lacks component {component_id!r}")
    omitted_names = _trusted_omitted_atom_names(atoms, rows, task, chain_id)
    if omitted_names:
        component = project_trusted_ccd_component(component, omitted_names)
    ordered = trusted_ccd_private_molecule(atoms, rows, component_id, component)
    return private_molecule_from_trusted_graph(atoms, rows, ordered)


def _positions(value: Any) -> set[int]:
    result: set[int] = set()
    for token in re.split(r"[,; ]+", _text(value)):
        if not token:
            continue
        match = re.fullmatch(r"(-?[0-9]+)(?:\s*(?:-|\.\.)\s*(-?[0-9]+))?", token)
        if match is None:
            continue
        start, stop = int(match.group(1)), int(match.group(2) or match.group(1))
        result.update(range(start, stop + (1 if stop >= start else -1), 1 if stop >= start else -1))
    return result


class _ObservedLengthRNG:
    """Replay antibody CDR lengths observed in the public result row."""

    def __init__(self, values: Sequence[int]) -> None:
        self._values = iter(int(value) for value in values)

    def integers(self, low: int, high: int) -> int:
        try:
            value = next(self._values)
        except StopIteration as exc:
            raise ValueError("antibody compiler requested an unrecorded CDR length") from exc
        if value < int(low) or value >= int(high):
            raise ValueError(
                f"result CDR length {value} is outside task interval [{low}, {high})"
            )
        return value

    def randint(self, low: int, high: int) -> int:
        return self.integers(low, high)

    def assert_consumed(self) -> None:
        try:
            next(self._values)
        except StopIteration:
            return
        raise ValueError("result row contains fewer CDR lengths than the task compiler")


def _antibody_metadata(
    request: ExpandedRequest,
    atoms: Any,
    row: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task = request.task
    if getattr(task, "format", "normal") != "antibody":
        return {}, {}
    values: list[int] = []
    for design in getattr(task.antibody, "chains", ()):
        side = "heavy" if str(design.role) in {"VH", "VHH"} else "light"
        ranges = {
            index: _positions(_value(row, f"{side}_cdr{index}_range"))
            for index in (1, 2, 3)
        }
        selected = set(getattr(design, "cdrs", ()))
        for index in sorted(selected):
            observed = ranges[index]
            if not observed:
                raise ExistingOutputRecoveryError(
                    f"antibody result lacks {side}_cdr{index} range metadata"
                )
            values.append(len(observed))
    replay = _ObservedLengthRNG(values)
    try:
        from ..antibody.compiler import compile_antibody_request
        from ..core.compiler import compile_request

        compiled = compile_antibody_request(
            request,
            compile_normal=compile_request,
            rng=replay,
        )
        replay.assert_consumed()
        entries = list(compiled.antibody_metadata.get("entries", ()))
        actual = {
            (str(atoms.chain_id[index]), int(atoms.res_id[index]))
            for index in range(len(atoms))
            if bool(atoms.is_design[index])
        }
        compiled_keys = {
            (str(item.get("chain_id")), int(item.get("position")))
            for item in entries
        }
        if not entries or compiled_keys != actual:
            raise ValueError(
                "compiler antibody residue ledger does not match existing CIF "
                f"(compiled={len(compiled_keys)}, existing={len(actual)})"
            )
        return dict(compiled.antibody_metadata), dict(compiled.motif_output_fields)
    except ExistingOutputRecoveryError:
        raise
    except Exception as exc:
        raise ExistingOutputRecoveryError(
            f"antibody metadata cannot be reconstructed from the existing row: {exc}"
        ) from exc


def candidate_from_existing(base_cif: Path, row: Any, task: Any, *, ccd_path: Path | None = None) -> SearchCandidate:
    """Read one existing base CIF and rebuild only its private owner state."""
    supplied_cif = Path(base_cif).expanduser()
    if supplied_cif.suffix.lower() != ".cif" or supplied_cif.is_symlink() or not supplied_cif.is_file():
        raise ExistingOutputRecoveryError(
            f"existing base CIF is not a regular file: {supplied_cif}"
        )
    base_cif = supplied_cif.resolve()
    request = _request(task, row)
    declared = {str(c.chain_id): str(c.modality) for c in getattr(task, "design_chains", ()) if str(c.modality) in {"protein", "rna", "dna"}}
    try:
        atoms = read_structure(base_cif, ccd_path=_reader_ccd_path(ccd_path), declared_polymer_chain_modalities=declared)
    except Exception as exc:
        raise ExistingOutputRecoveryError(f"cannot read existing base CIF {base_cif}: {exc}") from exc
    ownership = _antibody_ownership(atoms, task, row) if getattr(task, "format", "normal") == "antibody" else _normal_ownership(atoms, task, _normal_entries(row))
    private: dict[int, Any] = {}
    errors: dict[int, str] = {}
    for chain in getattr(task, "design_chains", ()):
        if str(chain.modality) != "ligand":
            continue
        chain_id = str(chain.chain_id)
        item = ownership[chain_id]
        chain_rows = _rows(atoms, str(item["chain"]))
        molecule_uids = sorted({int(v) for v in np.asarray(atoms.molecule_uid)[chain_rows]})
        if not molecule_uids:
            raise ExistingOutputRecoveryError(f"design ligand {chain_id!r} has no molecule UID")
        mode, sequence = _mode(task, chain_id), item.get("sequence", "")
        for molecule_uid in molecule_uids:
            component_rows = np.asarray([i for i in chain_rows if int(atoms.molecule_uid[i]) == molecule_uid], dtype=np.int64)
            try:
                if mode == "smiles":
                    private[molecule_uid] = _smiles_private(atoms, component_rows, _condition(task, chain_id).get("smiles"))
                elif mode in {"free", "backbone"}:
                    private[molecule_uid] = _generated_private(atoms, component_rows, sequence)
                elif mode in {"ccd", "type_bond"}:
                    private[molecule_uid] = _trusted_private(atoms, component_rows, task, chain_id, mode, ccd_path)
                else:
                    raise ValueError(f"unsupported existing-output ligand mode {mode!r}")
            except Exception as exc:
                errors[molecule_uid] = f"{type(exc).__name__}: {exc}"
    lengths: list[int] = []
    labels = {key: str(value["chain"]) for key, value in ownership.items()}
    for chain in getattr(task, "design_chains", ()):
        chain_rows = _rows(atoms, labels[str(chain.chain_id)])
        lengths.append(len(chain_rows) if str(chain.modality) == "ligand" else _count(atoms, chain_rows, str(chain.modality)))
    antibody_metadata, motif_fields = _antibody_metadata(request, atoms, row)
    return SearchCandidate(
        candidate_id=f"{task.name}/{request.sample_id}/existing",
        candidate_ordinal=request.global_ordinal,
        base_request=request,
        actual_joint_length=tuple(lengths),
        atoms=atoms,
        private_ligand_molecules=private,
        private_ligand_errors=errors,
        motif_output_fields=motif_fields,
        antibody_metadata=antibody_metadata,
    )


__all__ = ["ExistingOutputRecoveryError", "candidate_from_existing"]
