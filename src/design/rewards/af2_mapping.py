"""Private AF2 occurrence indices and reversible source residue identity."""

from __future__ import annotations

import copy
import os
import string
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from biotite import structure as struc
from biotite.structure.io import pdb

from ..utils.source_mapping import polymer_break_before, polymer_residue_groups


_AA3 = frozenset((
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
))
_RESIDUE_FIELDS = frozenset((
    "chain_id", "res_id", "ins_code", "label_asym_id", "label_seq_id",
    "auth_asym_id", "auth_seq_id", "auth_ins_code", "label_entity_id",
    "entity_uid", "asym_uid", "component_uid", "molecule_uid", "residue_uid",
    "token_uid", "is_design", "mol_type",
    "source_format", "source_graph_complete", "source_break_before",
    "source_polymer_sequence", "source_polymer_position",
))


def af2_source_ledger(
    atoms: Any, chain_order: Sequence[str], design_chain: str,
) -> dict[str, Any]:
    """Assign one private position per source occurrence, independent of labels."""

    order = [str(chain) for chain in chain_order]
    if not order or len(set(order)) != len(order) or design_chain not in order:
        raise ValueError("AF2 source chain ledger is inconsistent")
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    if len(order) > len(alphabet):
        raise ValueError("AF2 private PDB cannot represent more than 62 chains")
    used = {chain for chain in order if len(chain) == 1 and chain in alphabet}
    available = iter(char for char in alphabet if char not in used)
    categories = set(atoms.get_annotation_categories())
    chains = []
    all_groups = []
    occurrences = []
    for chain in order:
        private_chain = chain if chain in used else next(available)
        groups = polymer_residue_groups(atoms, chain)
        if not groups:
            raise ValueError(f"AF2 source chain {chain!r} has no polymer residues")
        breaks = set() if chain == design_chain else polymer_break_before(atoms, groups)
        offset = 0
        records = []
        for ordinal, group in enumerate(groups):
            if ordinal in breaks:
                offset += 50
            private_id = ordinal + 1 + offset
            if private_id > 9999:
                raise ValueError("AF2 private residue index exceeds the PDB field")
            records.append({
                "row_indices": group["row_indices"], "locator": dict(group["locator"]),
                "private_res_id": private_id, "break_before": ordinal in breaks,
            })
            all_groups.append(group)
            occurrences.append((chain, ordinal))
        chains.append({"source_chain": chain, "private_chain": private_chain, "residues": records})
    counts = np.fromiter((len(group["row_indices"]) for group in all_groups), dtype=np.int64)
    rows = np.fromiter(
        (int(row) for group in all_groups for row in group["row_indices"]),
        dtype=np.int64, count=int(counts.sum()),
    )
    group_by_row = np.repeat(np.arange(len(all_groups), dtype=np.int64), counts)
    starts = np.concatenate(([0], np.cumsum(counts[:-1])))
    first_rows = rows[starts]
    if np.any(np.asarray(atoms.mol_type)[rows] != "protein"):
        raise ValueError("AF2 source ledger accepts protein residues only")
    first_values = {}
    for name in _RESIDUE_FIELDS & categories:
        values = np.asarray(atoms.get_annotation(name))
        first = values[first_rows]
        if not np.array_equal(values[rows], first[group_by_row]):
            raise ValueError(f"AF2 source residue has ambiguous {name}")
        first_values[name] = first
    records = [record for chain in chains for record in chain["residues"]]
    for index, record in enumerate(records):
        annotations = {}
        for name, values in first_values.items():
            value = values[index]
            annotations[name] = value.item() if isinstance(value, np.generic) else value
        annotations.update({name: value for name, value in record["locator"].items() if name in categories})
        record["annotations"] = annotations
    edges = []
    if atoms.bonds is not None:
        atom_groups = np.full(len(atoms), -1, dtype=np.int64)
        atom_groups[rows] = group_by_row
        bonds = np.asarray(atoms.bonds.as_array())
        if len(bonds):
            left_groups, right_groups = atom_groups[bonds[:, 0]], atom_groups[bonds[:, 1]]
            cross = (left_groups >= 0) & (right_groups >= 0) & (left_groups != right_groups)
            names = np.asarray(atoms.atom_name)
            for left, right, order_type in bonds[cross].tolist():
                a = (*occurrences[int(atom_groups[left])], str(names[left]))
                b = (*occurrences[int(atom_groups[right])], str(names[right]))
                edges.append([list(a), list(b), int(order_type)])
    return {"design_chain": design_chain, "chains": chains, "source_edges": edges}


def private_chain_id(ledger: Mapping[str, Any], source_chain: str) -> str:
    matches = [
        chain["private_chain"] for chain in ledger["chains"]
        if chain["source_chain"] == source_chain
    ]
    if len(matches) != 1:
        raise ValueError("AF2 private chain mapping is not unique")
    return str(matches[0])


def af2_preparation_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Send only native occurrence indices to the resident worker."""

    return {"chains": [
        {
            "source_chain": chain["source_chain"], "private_chain": chain["private_chain"],
            "residues": [{"private_res_id": residue["private_res_id"]} for residue in chain["residues"]],
        }
        for chain in ledger["chains"]
    ]}


def encode_private_af2_indices(atoms: Any, ledger: Mapping[str, Any]) -> None:
    """Apply private IDs to a copied template, never to public source atoms."""

    records = [(str(chain["private_chain"]), residue) for chain in ledger["chains"] for residue in chain["residues"]]
    counts = np.fromiter((len(residue["row_indices"]) for _chain, residue in records), dtype=np.int64)
    rows = np.fromiter(
        (int(row) for _chain, residue in records for row in residue["row_indices"]),
        dtype=np.int64, count=int(counts.sum()),
    )
    groups = np.repeat(np.arange(len(records), dtype=np.int64), counts)
    private_chains = np.asarray([chain for chain, _residue in records])
    private_ids = np.asarray([residue["private_res_id"] for _chain, residue in records], dtype=np.int64)
    chains = np.asarray(atoms.chain_id).copy()
    res_ids = np.asarray(atoms.res_id).copy()
    ins_codes = np.asarray(atoms.ins_code).copy()
    chains[rows] = private_chains[groups]
    res_ids[rows] = private_ids[groups]
    ins_codes[rows] = ""
    atoms.chain_id, atoms.res_id, atoms.ins_code = chains, res_ids, ins_codes


def context_unknowns(atoms: Any, design_chain: str) -> Any:
    """Keep unsupported context occurrences as native backbone-only UNK."""

    unknown = (np.asarray(atoms.chain_id).astype(str) != design_chain) & ~np.isin(
        atoms.res_name, tuple(_AA3)
    )
    if not np.any(unknown):
        return atoms
    keep = ~unknown | np.isin(atoms.atom_name, ("N", "CA", "C", "O", "OXT"))
    atoms.res_name[unknown] = "UNK"
    atoms.hetero[unknown] = False
    return atoms[keep]


def write_private_af2_pdb(
    atoms: Any, output_path: Path, *, ledger: Mapping[str, Any],
) -> None:
    """Write the exact occurrence ledger as a coordinate-only native template."""

    private = copy.deepcopy(atoms)
    encode_private_af2_indices(private, ledger)
    private = context_unknowns(
        private, private_chain_id(ledger, str(ledger["design_chain"]))
    )
    if hasattr(private, "coord_valid"):
        valid = np.asarray(private.coord_valid, dtype=bool)
        if not valid.all():
            private = private[valid]
    private.bonds = None
    private.set_annotation("atom_id", np.arange(1, len(private) + 1, dtype=np.int64))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file = pdb.PDBFile()
    pdb.set_structure(file, private, hybrid36=False)
    file.write(output_path)


def normalize_private_override(
    source: Any, override: Any, ledger: Mapping[str, Any],
) -> Any:
    """Validate a coordinate override against source occurrences before encoding."""

    private = copy.deepcopy(source)
    observed = list(dict.fromkeys(np.asarray(override.chain_id).astype(str).tolist()))
    source_ids = [str(chain["source_chain"]) for chain in ledger["chains"]]
    private_ids = [str(chain["private_chain"]) for chain in ledger["chains"]]
    if set(observed) == set(source_ids):
        override_ids = source_ids
    elif set(observed) == set(private_ids):
        override_ids = private_ids
    else:
        raise ValueError("AF2 private PDB override has incompatible chain identities")
    for chain, override_id in zip(ledger["chains"], override_ids, strict=True):
        groups = polymer_residue_groups(override, override_id)
        if len(groups) != len(chain["residues"]):
            raise ValueError("AF2 private PDB override changed a source residue count")
        for residue, group in zip(chain["residues"], groups, strict=True):
            source_rows = np.asarray(residue["row_indices"], dtype=np.int64)
            rows = np.asarray(group["row_indices"], dtype=np.int64)
            names = [str(value) for value in override.atom_name[rows]]
            if len(names) != len(set(names)):
                raise ValueError("AF2 private PDB override has duplicate atom occurrences")
            by_name = dict(zip(names, rows.tolist(), strict=True))
            if set(str(value) for value in source.atom_name[source_rows]) != set(by_name):
                raise ValueError("AF2 private PDB override changed a source atom mask")
            source_name = str(source.res_name[source_rows[0]])
            if str(override.res_name[rows[0]]) != source_name:
                raise ValueError("AF2 private PDB override changed source residue identity")
            for row in source_rows:
                template_row = by_name[str(source.atom_name[row])]
                if bool(source.coord_valid[row]) != bool(override.coord_valid[template_row]):
                    raise ValueError("AF2 private PDB override changed a source coordinate mask")
                private.coord[row] = override.coord[template_row]
                private.coord_valid[row] = override.coord_valid[template_row]
    return private


def validate_prepared_af2(model: Any, ledger: Mapping[str, Any], chain_order: Sequence[str]) -> None:
    """Require native preparation to preserve every private source occurrence."""

    chosen = [chain for source_chain in chain_order for chain in ledger["chains"] if chain["source_chain"] == source_chain]
    counts = [len(chain["residues"]) for chain in chosen]
    size = sum(counts)
    expected_ids = np.fromiter(
        (int(residue["private_res_id"]) for chain in chosen for residue in chain["residues"]),
        dtype=np.int64, count=size,
    )
    expected_chains = np.repeat(np.asarray([str(chain["private_chain"]) for chain in chosen]), counts)
    native = getattr(model, "_pdb", None)
    index = native.get("idx") if isinstance(native, Mapping) else None
    if not isinstance(index, Mapping):
        raise ValueError("AF2 native preparation has no source occurrence ledger")
    chains = np.asarray(index.get("chain"), dtype=str)
    residues = np.asarray(index.get("residue"))
    indices = np.asarray(model._inputs.get("residue_index"))
    if (chains.shape != (size,) or residues.shape != chains.shape or indices.shape != chains.shape
            or not np.array_equal(chains, expected_chains) or not np.array_equal(residues, expected_ids)):
        raise ValueError("AF2 native preparation changed the source occurrence mapping")
    within = np.ones(max(0, size - 1), dtype=bool)
    cursor = 0
    for count in counts[:-1]:
        cursor += count
        if cursor:
            within[cursor - 1] = False
    if not np.array_equal(np.diff(indices)[within], np.diff(expected_ids)[within]):
        raise ValueError("AF2 native preparation changed a context break interval")

def restore_af2_residue_identity(
    folded: Any, ledger: Mapping[str, Any], chain_order: Sequence[str],
) -> Any:
    """Map native prediction ordinals back to exact public source locators."""

    selected = [
        chain for source_chain in chain_order for chain in ledger["chains"]
        if chain["source_chain"] == source_chain
    ]
    records = [residue for chain in selected for residue in chain["residues"]]
    source_occurrences = [
        (str(chain["source_chain"]), ordinal)
        for chain in selected for ordinal in range(len(chain["residues"]))
    ]
    selected_occurrences = set(source_occurrences)
    groups = []
    for chain in dict.fromkeys(np.asarray(folded.chain_id).astype(str).tolist()):
        groups.extend(polymer_residue_groups(folded, chain))
    if len(records) != len(groups):
        raise ValueError("AF2 prediction changed the source residue occurrence count")
    categories = set(folded.get_annotation_categories())
    names = set().union(*(residue["annotations"] for residue in records))
    arrays = {}
    for name in names:
        if name in categories:
            arrays[name] = np.asarray(folded.get_annotation(name)).copy()
        else:
            values = [residue["annotations"][name] for residue in records]
            dtype = object if name == "source_polymer_sequence" else np.asarray(values).dtype
            arrays[name] = np.zeros(len(folded), dtype=dtype)
    row_ordinal = np.full(len(folded), -1, dtype=np.int64)
    for ordinal, (record, group) in enumerate(zip(records, groups, strict=True)):
        rows = np.asarray(group["row_indices"], dtype=np.int64)
        row_ordinal[rows] = ordinal
        for name, value in record["annotations"].items():
            if arrays[name].dtype == object:
                for row in rows:
                    arrays[name][row] = value
            else:
                arrays[name][rows] = value
    for name, values in arrays.items():
        folded.set_annotation(name, values)
    atom_rows = {}
    for occurrence, group in zip(source_occurrences, groups, strict=True):
        for row in group["row_indices"]:
            key = (*occurrence, str(folded.atom_name[row]))
            if key in atom_rows:
                raise ValueError("AF2 prediction has duplicate source atom occurrences")
            atom_rows[key] = int(row)
    # Native serialization can flatten multiple target chains.  Revoke its
    # inferred peptide edges wherever the source ledger declares a boundary.
    edges = []
    if folded.bonds is not None:
        for left, right, order in folded.bonds.as_array().tolist():
            a, b = int(row_ordinal[left]), int(row_ordinal[right])
            if a != b and {str(folded.atom_name[left]), str(folded.atom_name[right])} == {"C", "N"}:
                lo, hi = sorted((a, b))
                if (
                    str(folded.chain_id[left]) != str(folded.chain_id[right])
                    or (hi == lo + 1 and bool(records[hi]["break_before"]))
                ):
                    continue
            edges.append((left, right, order))
    by_pair = {tuple(sorted((left, right))): order for left, right, order in edges}
    for left, right, order in ledger.get("source_edges", ()):
        a, b = tuple(left), tuple(right)
        if a[:2] not in selected_occurrences or b[:2] not in selected_occurrences:
            continue
        if a not in atom_rows or b not in atom_rows:
            raise ValueError("AF2 prediction omitted an authoritative covalent endpoint")
        pair = tuple(sorted((atom_rows[a], atom_rows[b])))
        if order == int(struc.BondType.ANY) and pair in by_pair:
            continue
        by_pair[pair] = int(order)
    folded.bonds = struc.BondList(
        len(folded), np.asarray([(a, b, order) for (a, b), order in by_pair.items()], dtype=np.uint32).reshape(-1, 3),
    )
    return folded


def af2_output_suffix(ledger: Mapping[str, Any], chain_order: Sequence[str]) -> str:
    """Retain PDB names only when that format preserves complete source locators."""

    selected = set(chain_order)
    for left, right, order in ledger.get("source_edges", ()):
        if left[0] not in selected or right[0] not in selected:
            continue
        if left[0] != right[0]:
            return ".cif"
        first, second = sorted((left, right), key=lambda atom: int(atom[1]))
        if (int(second[1]) != int(first[1]) + 1
                or (first[2], second[2]) != ("C", "N") or int(order) != 1):
            return ".cif"
    used_auth_chains = set()
    for chain in ledger["chains"]:
        if chain["source_chain"] not in chain_order:
            continue
        used = set()
        source_chain = str(chain["source_chain"])
        if len(source_chain) != 1 or source_chain not in string.ascii_letters + string.digits:
            return ".cif"
        for record in chain["residues"]:
            if record.get("break_before", False):
                # PDB has no authoritative absent-backbone-edge record after folding.
                return ".cif"
            locator = record["locator"]
            source_format = str(locator.get("source_format", ""))
            auth_chain = str(locator.get("auth_asym_id") or source_chain)
            auth_id = str(locator.get("auth_seq_id") or locator["res_id"])
            ins = str(locator.get("auth_ins_code") or locator.get("ins_code") or "")
            if len(auth_chain) != 1 or auth_chain not in string.ascii_letters + string.digits:
                return ".cif"
            if source_format != "pdb" and (
                auth_chain != source_chain or auth_id != str(locator["res_id"])
                or ins != str(locator.get("ins_code") or "")
            ):
                return ".cif"
            try:
                number = int(auth_id)
            except ValueError:
                return ".cif"
            if str(number) != auth_id or not -999 <= number <= 9999 or len(ins) > 1:
                return ".cif"
            if (number, ins) in used:
                return ".cif"
            used.add((number, ins))
        if auth_chain in used_auth_chains:
            return ".cif"
        used_auth_chains.add(auth_chain)
    return ".pdb"


def preserve_native_pdb_confidence(native_path: Path, restored_path: Path) -> None:
    """Retain native occupancy/B factors through a verified atom occurrence map."""

    def atom_fields(lines):
        residues = {}
        fields = {}
        for line in lines:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 66:
                continue
            residue = (line[21], line[22:26], line[26])
            ordinal = residues.setdefault(residue, len(residues))
            key = (ordinal, line[12:16].strip())
            if key in fields:
                raise ValueError("AF2 PDB confidence map has duplicate atom occurrences")
            fields[key] = line[54:66]
        return fields

    native = atom_fields(native_path.read_text().splitlines(keepends=True))
    lines = restored_path.read_text().splitlines(keepends=True)
    restored = atom_fields(lines)
    if set(native) != set(restored):
        raise ValueError("AF2 PDB confidence source atom occurrence mapping differs")
    residues = {}
    rewritten = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 66:
            residue = (line[21], line[22:26], line[26])
            ordinal = residues.setdefault(residue, len(residues))
            line = line[:54] + native[(ordinal, line[12:16].strip())] + line[66:]
        rewritten.append(line)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=f".{restored_path.name}.", suffix=".tmp",
            dir=restored_path.parent, encoding="utf-8", delete=False,
        ) as file:
            temporary = Path(file.name)
            file.write("".join(rewritten))
        os.replace(temporary, restored_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
