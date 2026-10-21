"""Private source locators and observed/native polymer position plans.

Model indices are never public residue locators.  Only confirmed source
sequence mappings may introduce positions without observed source atoms.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..core.records import DesignInputError
from .atoms import COMMON_STANDARD_PARENT, PROTEIN_RES_ATOMS


COMPLETE_BACKBONE_CRITERIA = (
    "All observed inter-residue polymer covalent bonds are explicitly listed."
)
SOURCE_ANNOTATIONS = (
    "source_format", "source_graph_complete", "source_break_before",
    "source_polymer_sequence", "source_polymer_position",
)
_AA = dict(zip(
    ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
     "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"),
    "ARNDCQEGHILKMFPSTWYV",
    strict=True,
))


def _text(atoms: Any, field: str, row: int, default: str = "") -> str:
    values = getattr(atoms, field, None)
    return default if values is None else str(values[row]).strip()


def canonical_chain_ids(atoms: Any) -> np.ndarray:
    labels = getattr(atoms, "label_asym_id", atoms.chain_id)
    return np.asarray([
        str(label).strip() if str(label).strip() not in {"", ".", "?"}
        else str(atoms.chain_id[row])
        for row, label in enumerate(labels)
    ], dtype=str)


def source_locator(atoms: Any, row: int) -> dict[str, Any]:
    """Snapshot the current public identity before a private backend rename."""
    row = int(row)
    chain = _text(atoms, "label_asym_id", row)
    if chain in {"", ".", "?"}:
        chain = str(atoms.chain_id[row])
    position = int(atoms.res_id[row])
    modality = _text(atoms, "mol_type", row)
    insertion = _text(atoms, "ins_code", row)
    return {
        "chain_id": chain,
        "res_id": position,
        "ins_code": insertion,
        "label_asym_id": chain,
        "label_seq_id": str(position) if modality != "ligand" else ".",
        "auth_asym_id": _text(atoms, "auth_asym_id", row) or chain,
        "auth_seq_id": _text(atoms, "auth_seq_id", row) or str(position),
        "auth_ins_code": _text(atoms, "auth_ins_code", row) or insertion,
        "source_format": _text(atoms, "source_format", row),
    }


def polymer_residue_groups(atoms: Any, chain_id: str) -> list[dict[str, Any]]:
    chains = canonical_chain_ids(atoms)
    rows = np.flatnonzero((chains == str(chain_id)) & (np.asarray(atoms.mol_type) != "ligand"))
    if not len(rows):
        return []
    uids = np.asarray(atoms.residue_uid, dtype=np.int64)
    by_uid: dict[int, list[int]] = {}
    for row in rows.tolist():
        by_uid.setdefault(int(uids[row]), []).append(row)
    groups = []
    for uid, members in by_uid.items():
        first = members[0]
        tokens = np.asarray(atoms.token_uid[members], dtype=np.int64)
        group = {
            "row_indices": members, "residue_uid": uid,
            "token_uid": int(tokens.min()) if len(tokens) and (tokens >= 0).all() else -1,
            "locator": source_locator(atoms, first),
            "res_name": str(atoms.res_name[first]),
            "modality": str(atoms.mol_type[first]),
        }
        for name in SOURCE_ANNOTATIONS:
            values = getattr(atoms, name, None)
            if values is not None:
                group[name] = values[first]
        groups.append(group)
    if groups and all(group["token_uid"] >= 0 for group in groups):
        groups.sort(key=lambda group: group["token_uid"])
    return groups


def _backbone_rows(atoms: Any, group: Mapping[str, Any], end: str) -> list[int]:
    modality = group["modality"]
    name = ("C" if end == "left" else "N") if modality == "protein" else ("O3'" if end == "left" else "P")
    return [row for row in group["row_indices"] if str(atoms.atom_name[row]) == name]


def polymer_break_before(atoms: Any, groups: Sequence[Mapping[str, Any]]) -> set[int]:
    """Breaks in observed graph; explicit bonds always outrank geometry/IDs."""
    bond_list = getattr(atoms, "bonds", None)
    pairs = set() if bond_list is None else {
        tuple(sorted((int(left), int(right))))
        for left, right, _order in bond_list.as_array().tolist()
    }
    result: set[int] = set()
    complete = getattr(atoms, "source_graph_complete", None)
    recorded = getattr(atoms, "source_break_before", None)
    for index, (left, right) in enumerate(zip(groups, groups[1:]), start=1):
        lr, rr = left["row_indices"][0], right["row_indices"][0]
        ends = [(a, b) for a in _backbone_rows(atoms, left, "left") for b in _backbone_rows(atoms, right, "right")]
        old_break = recorded is not None and bool(recorded[rr])
        graph_is_complete = complete is not None and bool(complete[lr]) and bool(complete[rr])
        if any(tuple(sorted(edge)) in pairs for edge in ends) and (graph_is_complete or not old_break):
            continue
        if left["modality"] != right["modality"] or int(atoms.asym_uid[lr]) != int(atoms.asym_uid[rr]):
            result.add(index)
            continue
        if old_break:
            result.add(index)
            continue
        if graph_is_complete:
            result.add(index)
            continue
        if ends and pairs:
            result.add(index)
            continue
        # Compatibility for old context dumps lacking complete explicit graph.
        # Callers never use this fallback to split a newly designed chain.
        if left["modality"] == "protein":
            names = np.asarray(atoms.atom_name).astype(str)
            la = [r for r in left["row_indices"] if names[r] == "CA"]
            ra = [r for r in right["row_indices"] if names[r] == "CA"]
            if len(la) == len(ra) == 1:
                valid = getattr(atoms, "coord_valid", np.ones(len(atoms), dtype=bool))
                xyz = np.asarray(atoms.coord[[la[0], ra[0]]], dtype=float)
                if bool(valid[la[0]]) and bool(valid[ra[0]]) and np.isfinite(xyz).all() and np.linalg.norm(xyz[1] - xyz[0]) > 4.8:
                    result.add(index)
    return result


def _component_letter(component: str, modality: str) -> str:
    name = COMMON_STANDARD_PARENT.get(str(component).upper(), str(component).upper())
    if modality == "protein":
        return _AA.get(name, "X")
    if modality == "dna" and name.startswith("D"):
        name = name[1:]
    allowed = "ACGU" if modality == "rna" else "ACGT"
    return name if name in allowed else "N"


def plan_refold_positions(
    atoms: Any, *, design_chains: tuple[str, ...], chain_sequences: Mapping[str, str],
) -> dict[str, Any]:
    """Plan physical entities without fabricating missing source positions."""
    design = set(str(chain) for chain in design_chains)
    used = set(canonical_chain_ids(atoms).tolist())
    plans = []
    for chain in dict.fromkeys(canonical_chain_ids(atoms).tolist()):
        groups = polymer_residue_groups(atoms, chain)
        if not groups:
            continue
        modalities = {group["modality"] for group in groups}
        if len(modalities) != 1:
            raise DesignInputError(f"refold polymer chain {chain!r} has mixed modalities")
        modality = next(iter(modalities))
        observed = chain_sequences.get(chain)
        if not isinstance(observed, str) or len(observed) != len(groups):
            raise DesignInputError(f"refold chain {chain!r} observed sequence/position count differs")
        role = "design" if chain in design else "context"
        breaks = set() if role == "design" else polymer_break_before(atoms, groups)
        full_values = [group.get("source_polymer_sequence") for group in groups]
        present = [tuple(value) for value in full_values if value is not None and len(value)]
        full = present[0] if present and role == "context" else None
        if full is not None and any(value != full for value in present):
            raise DesignInputError(f"refold chain {chain!r} has conflicting full sequences")
        positions = [int(group.get("source_polymer_position", 0)) for group in groups]
        entities = []
        semantics = "design" if role == "design" else "observed_fragments"
        if role == "context" and full is not None:
            if any(position <= 0 or position > len(full) for position in positions) or len(set(positions)) != len(positions) or positions != sorted(positions):
                raise DesignInputError(f"refold chain {chain!r} full-sequence mapping is not unique/ordered")
            if any(int(group["locator"]["label_seq_id"]) != position
                   for group, position in zip(groups, positions, strict=True)):
                raise DesignInputError(f"refold chain {chain!r} full-sequence positions disagree with source labels")
            for ordinal, position in enumerate(positions):
                old, new = _component_letter(full[position - 1], modality), observed[ordinal]
                if old != new and old not in {"X", "N"} and new not in {"X", "N"}:
                    raise DesignInputError(f"refold chain {chain!r} source sequence disagrees at full position {position}")
            # A break between consecutive confirmed full positions is a real
            # fragment boundary. A hole in the full sequence is unresolved pose.
            fragment_breaks = {index for index in breaks if positions[index] == positions[index - 1] + 1}
            if not fragment_breaks:
                sequence = [_component_letter(component, modality) for component in full]
                reverse: list[int | None] = [None] * len(sequence)
                for ordinal, position in enumerate(positions):
                    sequence[position - 1] = observed[ordinal]
                    reverse[position - 1] = ordinal
                entities = [{"chain_id": chain, "sequence": "".join(sequence), "components": list(full), "observed_positions": reverse}]
                semantics = "complete_sequence"
            else:
                cuts = [0, *[positions[index] - 1 for index in sorted(fragment_breaks)], len(full)]
                for number, (start, stop) in enumerate(zip(cuts, cuts[1:])):
                    physical = chain if not number else f"{chain}__pdfrag{number}"
                    if number:
                        serial = number
                        while physical in used:
                            serial += 1
                            physical = f"{chain}__pdfrag{serial}"
                        used.add(physical)
                    sequence = [_component_letter(component, modality) for component in full[start:stop]]
                    reverse = [None] * len(sequence)
                    for ordinal, position in enumerate(positions):
                        if start < position <= stop:
                            sequence[position - start - 1] = observed[ordinal]
                            reverse[position - start - 1] = ordinal
                    entities.append({"chain_id": physical, "sequence": "".join(sequence), "components": list(full[start:stop]), "observed_positions": reverse})
                semantics = "complete_sequence"
        if not entities:
            bounds = [0, *sorted(breaks), len(groups)]
            for number, (start, stop) in enumerate(zip(bounds, bounds[1:])):
                if start == stop:
                    continue
                physical = chain
                if number:
                    serial = number
                    physical = f"{chain}__pdfrag{serial}"
                    while physical in used:
                        serial += 1
                        physical = f"{chain}__pdfrag{serial}"
                    used.add(physical)
                entities.append({"chain_id": physical, "sequence": observed[start:stop], "observed_positions": list(range(start, stop)), "components": [group["res_name"] for group in groups[start:stop]]})
        plans.append({"logical_chain_id": chain, "role": role, "modality": modality,
                      "observed": groups, "entities": entities, "input_semantics": semantics,
                      "break_before": sorted(breaks)})
    return {"chains": plans}


def attach_source_sequences(atoms: Any, *, block: Any = None, pdb_lines: Sequence[str] | None = None) -> None:
    """Read authoritative sequence information once at the source boundary."""
    sequences: dict[str, tuple[str, ...]] = {}
    if block is not None and "entity_poly_seq" in block:
        category = block["entity_poly_seq"]
        by_entity: dict[str, dict[int, str]] = {}
        for entity, number, component in zip(category["entity_id"].as_array(str), category["num"].as_array(int), category["mon_id"].as_array(str), strict=True):
            values = by_entity.setdefault(str(entity), {})
            if int(number) in values and values[int(number)] != str(component):
                raise DesignInputError(f"ambiguous full polymer sequence for entity {entity!r}")
            values[int(number)] = str(component).upper()
        for entity, values in by_entity.items():
            if sorted(values) != list(range(1, len(values) + 1)):
                raise DesignInputError(f"entity {entity!r} full sequence positions are not contiguous")
            sequences[entity] = tuple(values[index] for index in range(1, len(values) + 1))
    elif pdb_lines is not None:
        by_chain: dict[str, list[str]] = {}
        counts: dict[str, int] = {}
        for line in pdb_lines:
            if line.startswith("SEQRES"):
                chain = line[11:12].strip()
                count = int(line[13:17])
                if chain in counts and counts[chain] != count:
                    raise DesignInputError(f"conflicting SEQRES length for chain {chain!r}")
                counts[chain] = count
                by_chain.setdefault(chain, []).extend(line[19:70].split())
        for chain, values in by_chain.items():
            if len(values) != counts[chain]:
                raise DesignInputError(f"incomplete SEQRES for chain {chain!r}")
            sequences[chain] = tuple(component.upper() for component in values)
    if not sequences:
        return
    for chain in dict.fromkeys(canonical_chain_ids(atoms).tolist()):
        groups = polymer_residue_groups(atoms, chain)
        if not groups:
            continue
        first = groups[0]["row_indices"][0]
        key = str(atoms.label_entity_id[first]) if block is not None else str(atoms.auth_asym_id[first])
        full = sequences.get(key)
        if full is None:
            continue
        if block is not None:
            positions = [int(atoms.label_seq_id[group["row_indices"][0]]) for group in groups]
            if any(position <= 0 or position > len(full) for position in positions):
                raise DesignInputError(f"source chain {chain!r} label positions exceed its full sequence")
        else:
            normalized = [COMMON_STANDARD_PARENT.get(component, component) for component in full]
            observed = [COMMON_STANDARD_PARENT.get(group["res_name"], group["res_name"]) for group in groups]
            def match(reverse: bool) -> list[int]:
                result = []
                cursor = len(full) - 1 if reverse else 0
                for component in reversed(observed) if reverse else observed:
                    while 0 <= cursor < len(full) and normalized[cursor] != component:
                        cursor += -1 if reverse else 1
                    if not 0 <= cursor < len(full):
                        raise DesignInputError(f"source chain {chain!r} does not match its SEQRES")
                    result.append(cursor + 1)
                    cursor += -1 if reverse else 1
                return list(reversed(result)) if reverse else result
            positions = match(False)
            if positions != match(True):
                raise DesignInputError(f"source chain {chain!r} SEQRES/observed alignment is ambiguous; provide an explicit CIF sequence mapping")
        for group, position in zip(groups, positions, strict=True):
            actual = COMMON_STANDARD_PARENT.get(group["res_name"], group["res_name"])
            expected = COMMON_STANDARD_PARENT.get(full[position - 1], full[position - 1])
            if actual != expected:
                raise DesignInputError(f"source chain {chain!r} sequence identity differs at position {position}")
            for row in group["row_indices"]:
                atoms.source_polymer_sequence[row] = full
                atoms.source_polymer_position[row] = position
                if block is None:
                    # PDB author numbering stays in label_seq_id/auth fields;
                    # this is the confirmed legal CIF position, not resSeq.
                    atoms.res_id[row] = position


def mark_source_breaks(atoms: Any, declared_bonds: Sequence[tuple[int, int, int]], *, complete: bool) -> None:
    """Record observed source discontinuities before template inference."""
    declared = {tuple(sorted((int(a), int(b)))) for a, b, _ in declared_bonds}
    for chain in dict.fromkeys(canonical_chain_ids(atoms).tolist()):
        groups = polymer_residue_groups(atoms, chain)
        for left, right in zip(groups, groups[1:]):
            ends = [(a, b) for a in _backbone_rows(atoms, left, "left") for b in _backbone_rows(atoms, right, "right")]
            if any(tuple(sorted(edge)) in declared for edge in ends):
                continue
            lr, rr = left["row_indices"][0], right["row_indices"][0]
            p, q = int(atoms.source_polymer_position[lr]), int(atoms.source_polymer_position[rr])
            broken = complete or (p > 0 and q > p + 1)
            if not broken and ends and left["modality"] == "protein":
                a, b = ends[0]
                try:
                    author_gap = abs(int(atoms.auth_seq_id[a]) - int(atoms.auth_seq_id[b])) > 1
                except ValueError:
                    author_gap = False
                distance = float(np.linalg.norm(atoms.coord[a] - atoms.coord[b]))
                broken = author_gap and bool(atoms.coord_valid[a]) and bool(atoms.coord_valid[b]) and np.isfinite(distance) and distance > 2.0
            if broken:
                atoms.source_break_before[right["row_indices"]] = True
