"""Closed, source-first task crop projection.

The crop operates on the reader-normalized full source. It deliberately keeps
the source AtomUIDs and the reader's already-materialized BondList; only the
row indices of surviving edges are changed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

import biotite.structure as struc
import numpy as np

from ..utils.atoms import DNA_RES_ATOMS, PROTEIN_RES_ATOMS, RNA_RES_ATOMS
from .records import CropSpec, DesignSampleError


_SOURCE_SEGMENT = re.compile(
    r"^(?:\[([^\]]+)\]|([A-Za-z]))(\d+)([^\d\-/]*)"
    r"(?:-(\d+)([^\d\-/]*))?$"
)
_SOURCE_CHAIN = re.compile(r"^(?:\[([^\]]+)\]|([A-Za-z]))$")
_STANDARD_RESIDUES = {
    "protein": frozenset(PROTEIN_RES_ATOMS),
    "rna": frozenset(RNA_RES_ATOMS),
    "dna": frozenset(DNA_RES_ATOMS),
}
_POLYMER_MODALITIES = frozenset(_STANDARD_RESIDUES)
_COORDINATION = int(struc.BondType.COORDINATION)
_ANY = int(struc.BondType.ANY)


@dataclass(frozen=True)
class CropResult:
    source: struc.AtomArray
    declared_bonds: tuple[tuple[int, int, int], ...]
    # Keys use stable label-asym text because source asym_uid is replaced later
    # when the final authoring IR is assembled.
    forced_chain_breaks: tuple[tuple[str, int, str], ...] = ()


def _unique_residue_uids(atoms: struc.AtomArray, rows: np.ndarray) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for raw_row in np.asarray(rows, dtype=np.int64).tolist():
        uid = int(atoms.residue_uid[int(raw_row)])
        if uid not in seen:
            seen.add(uid)
            result.append(uid)
    return result


def _chain_residue_uids(atoms: struc.AtomArray, chain_id: str) -> list[int]:
    return _unique_residue_uids(atoms, np.flatnonzero(atoms.chain_id == chain_id))


def _source_residue_term(atoms: struc.AtomArray, term: str) -> list[int]:
    text = str(term).strip()
    chain_match = _SOURCE_CHAIN.fullmatch(text)
    if chain_match is not None:
        chain_id = chain_match.group(1) or chain_match.group(2)
        if not chain_id or "/" in chain_id:
            raise DesignSampleError(f"invalid source chain selector: {term!r}")
        residues = _chain_residue_uids(atoms, chain_id)
        if not residues:
            raise DesignSampleError(f"source selector chain {chain_id!r} does not exist")
        return residues
    match = _SOURCE_SEGMENT.fullmatch(text)
    if match is None:
        raise DesignSampleError(f"invalid source residue selector: {term!r}")
    chain_id = match.group(1) or match.group(2)
    if not chain_id or "]" in chain_id or "/" in chain_id:
        raise DesignSampleError(f"invalid source chain selector: {term!r}")
    start_number, start_ins = int(match.group(3)), match.group(4) or ""
    end_number = None if match.group(5) is None else int(match.group(5))
    end_ins = match.group(6) or ""
    residues = _chain_residue_uids(atoms, chain_id)
    if not residues:
        raise DesignSampleError(f"source selector chain {chain_id!r} does not exist")

    def endpoint(number: int, ins_code: str) -> int:
        found: list[int] = []
        for uid in residues:
            rows = np.flatnonzero(atoms.residue_uid == uid)
            label_seq_id = str(atoms.label_seq_id[int(rows[0])])
            try:
                structural_number = int(label_seq_id)
            except ValueError:
                structural_number = int(atoms.res_id[int(rows[0])])
            if structural_number != number:
                continue
            if ins_code and str(atoms.ins_code[int(rows[0])]) != ins_code:
                continue
            found.append(uid)
        if len(found) != 1:
            raise DesignSampleError(
                f"source selector endpoint {chain_id}{number}{ins_code} "
                f"matched {len(found)} residues"
            )
        return found[0]

    start_uid = endpoint(start_number, start_ins)
    if end_number is None:
        return [start_uid]
    end_uid = endpoint(end_number, end_ins)
    start_index, end_index = residues.index(start_uid), residues.index(end_uid)
    if start_index > end_index:
        raise DesignSampleError(f"source residue range is reversed: {term!r}")
    return residues[start_index : end_index + 1]


def source_selector(atoms: struc.AtomArray, text: str) -> list[int]:
    """Resolve the public source-residue selector grammar on a full source."""

    if not isinstance(text, str) or not text.strip():
        raise DesignSampleError("source selector must be a non-empty string")
    result: list[int] = []
    seen: set[int] = set()
    for term in text.split("/"):
        for uid in _source_residue_term(atoms, term):
            if uid not in seen:
                seen.add(uid)
                result.append(uid)
    return result


def _source_endpoint_row(
    atoms: struc.AtomArray,
    endpoint: object,
    *,
    design_chain_ids: frozenset[str],
    where: str,
) -> int | None:
    if not isinstance(endpoint, Mapping):
        raise DesignSampleError(f"{where} endpoint must be a mapping")
    allowed = {"chain", "resid", "ins_code", "atom"}
    if set(endpoint).difference(allowed) or not {"chain", "atom"} <= set(endpoint):
        raise DesignSampleError(
            f"{where} endpoint requires chain/atom and optional resid/ins_code"
        )
    chain = endpoint["chain"]
    atom_name = endpoint["atom"]
    if not isinstance(chain, str) or not isinstance(atom_name, str):
        raise DesignSampleError(f"{where} endpoint chain and atom must be strings")
    chain_rows = np.flatnonzero(atoms.chain_id == chain)
    if not len(chain_rows):
        # A de novo design endpoint is resolved later after materialization.
        return None
    if chain in design_chain_ids:
        # Design-chain residue semantics are resolved against the final design
        # materialization (including contig/new-length mapping), not the input
        # source.  The whole source design chain is mandatory independently.
        return None
    modalities = {str(value) for value in atoms.mol_type[chain_rows]}
    if len(modalities) != 1:
        raise DesignSampleError(f"{where} endpoint chain {chain!r} has mixed modalities")
    if modalities == {"ligand"}:
        if "resid" in endpoint or "ins_code" in endpoint:
            raise DesignSampleError(
                f"{where} designed ligand endpoints use stable atom name only"
            )
        mask = (atoms.chain_id == chain) & (atoms.atom_name == atom_name)
    else:
        if "resid" not in endpoint:
            raise DesignSampleError(f"{where} polymer endpoint requires resid")
        raw_resid = endpoint["resid"]
        if isinstance(raw_resid, bool) or not isinstance(raw_resid, int):
            raise DesignSampleError(f"{where} polymer resid must be an integer")
        if raw_resid <= 0:
            raise DesignSampleError(
                f"{where} context polymer resid must be positive"
            )
        mask = (atoms.chain_id == chain) & (atoms.label_seq_id == str(raw_resid))
        if "ins_code" in endpoint:
            mask &= atoms.ins_code == str(endpoint["ins_code"])
        mask &= atoms.atom_name == atom_name
    matches = np.flatnonzero(mask)
    if len(matches) != 1:
        raise DesignSampleError(
            f"{where} endpoint {dict(endpoint)!r} matched {len(matches)} source atoms"
        )
    return int(matches[0])


def _residue_rows(atoms: struc.AtomArray) -> dict[int, np.ndarray]:
    return {
        int(uid): np.flatnonzero(atoms.residue_uid == uid)
        for uid in np.unique(atoms.residue_uid)
    }


def _is_modified_polymer(atoms: struc.AtomArray, residue_uid: int) -> bool:
    rows = np.flatnonzero(atoms.residue_uid == int(residue_uid))
    if not len(rows):
        return False
    modality = str(atoms.mol_type[int(rows[0])])
    return (
        modality in _POLYMER_MODALITIES
        and str(atoms.res_name[int(rows[0])]).strip().upper()
        not in _STANDARD_RESIDUES[modality]
    )


def _physical_groups(atoms: struc.AtomArray) -> tuple[frozenset[int], ...]:
    """Return connected ligand component/molecule groups without using mol_id."""

    groups: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in np.flatnonzero(atoms.mol_type == "ligand"):
        residue_uid = int(atoms.residue_uid[int(row)])
        for name in ("component_uid", "molecule_uid"):
            value = int(atoms.get_annotation(name)[int(row)])
            if value >= 0:
                groups[(name, value)].add(residue_uid)
    result: list[frozenset[int]] = []
    unassigned = set().union(*groups.values()) if groups else set()
    while unassigned:
        component = {unassigned.pop()}
        changed = True
        while changed:
            changed = False
            for members in groups.values():
                if members.intersection(component) and not members <= component:
                    component.update(members)
                    unassigned.difference_update(members)
                    changed = True
        result.append(frozenset(component))
    return tuple(result)


def _residue_order_by_asym(atoms: struc.AtomArray) -> dict[int, list[int]]:
    result: dict[int, list[int]] = defaultdict(list)
    for row in range(len(atoms)):
        asym_uid = int(atoms.asym_uid[row])
        residue_uid = int(atoms.residue_uid[row])
        if residue_uid not in result[asym_uid]:
            result[asym_uid].append(residue_uid)
    return result


def _ordinary_backbone_edge(
    atoms: struc.AtomArray,
    row_i: int,
    row_j: int,
    residue_order: Mapping[int, Sequence[int]],
) -> bool:
    residue_i = int(atoms.residue_uid[row_i])
    residue_j = int(atoms.residue_uid[row_j])
    if residue_i == residue_j or int(atoms.asym_uid[row_i]) != int(atoms.asym_uid[row_j]):
        return False
    modality_i, modality_j = str(atoms.mol_type[row_i]), str(atoms.mol_type[row_j])
    if modality_i != modality_j or modality_i not in _STANDARD_RESIDUES:
        return False
    order = residue_order.get(int(atoms.asym_uid[row_i]), ())
    try:
        adjacent = abs(order.index(residue_i) - order.index(residue_j)) == 1
    except ValueError:
        return False
    if not adjacent:
        return False
    names = {str(atoms.atom_name[row_i]), str(atoms.atom_name[row_j])}
    if modality_i == "protein":
        return names == {"C", "N"}
    return names == {"O3'", "P"}


def _edge_closes(
    atoms: struc.AtomArray,
    row_i: int,
    row_j: int,
    residue_order: Mapping[int, Sequence[int]],
) -> bool:
    residue_i = int(atoms.residue_uid[row_i])
    residue_j = int(atoms.residue_uid[row_j])
    if residue_i == residue_j:
        return False
    if str(atoms.mol_type[row_i]) == "ligand" or str(atoms.mol_type[row_j]) == "ligand":
        return True
    if _is_modified_polymer(atoms, residue_i) or _is_modified_polymer(atoms, residue_j):
        return True
    return not _ordinary_backbone_edge(atoms, row_i, row_j, residue_order)


def _normalize_declared_bonds(
    atoms: struc.AtomArray,
    declared_bonds: Sequence[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], ...]:
    by_uid: dict[tuple[int, int], tuple[int, int, int]] = {}
    for raw_i, raw_j, raw_order in declared_bonds:
        row_i, row_j, order = int(raw_i), int(raw_j), int(raw_order)
        if row_i < 0 or row_j < 0 or row_i >= len(atoms) or row_j >= len(atoms):
            raise DesignSampleError("source declared bond endpoint is out of range")
        if row_i == row_j:
            raise DesignSampleError("source declared bond cannot be a self-edge")
        uid_pair = tuple(sorted((int(atoms.atom_uid[row_i]), int(atoms.atom_uid[row_j]))))
        previous = by_uid.get(uid_pair)
        if previous is None:
            by_uid[uid_pair] = (row_i, row_j, order)
            continue
        if previous[2] == order or previous[2] == _ANY:
            if previous[2] == _ANY and order != _ANY:
                by_uid[uid_pair] = (row_i, row_j, order)
            continue
        if order == _ANY:
            continue
        raise DesignSampleError(
            f"conflicting duplicate source declared bond orders for AtomUID pair {uid_pair}"
        )
    return tuple(by_uid[key] for key in sorted(by_uid))


def _forced_break_keys(
    full_source: struc.AtomArray,
    kept_residues: set[int],
    design_chain_ids: frozenset[str],
) -> tuple[tuple[str, int, str], ...]:
    result: set[tuple[str, int, str]] = set()
    residue_order = _residue_order_by_asym(full_source)
    for asym_uid, ordered in residue_order.items():
        if not ordered:
            continue
        representative = int(np.flatnonzero(full_source.asym_uid == asym_uid)[0])
        if str(full_source.chain_id[representative]) in design_chain_ids:
            continue
        kept = [uid for uid in ordered if uid in kept_residues]
        for left, right in zip(kept, kept[1:]):
            if ordered.index(right) - ordered.index(left) <= 1:
                continue
            rows = np.flatnonzero(full_source.residue_uid == left)
            row = int(rows[0])
            result.add(
                (
                    str(full_source.label_asym_id[row]),
                    int(full_source.res_id[row]),
                    str(full_source.ins_code[row]),
                )
            )
    return tuple(sorted(result))


def project_source_crop(
    source: struc.AtomArray,
    declared_bonds: Sequence[tuple[int, int, int]],
    spec: CropSpec,
    *,
    design_chain_ids: set[str] | frozenset[str],
    hotspot: str | None,
    condition: Mapping[str, object],
) -> CropResult:
    """Apply one task crop to a fully read/validated source structure."""

    if source is None or not isinstance(spec, CropSpec):
        raise DesignSampleError("task crop requires a fully read source structure")
    design_chain_ids = frozenset(str(value) for value in design_chain_ids)
    full_source = source
    residue_rows = _residue_rows(full_source)
    all_residues = set(residue_rows)
    normalized_declared = _normalize_declared_bonds(full_source, declared_bonds)
    residue_order = _residue_order_by_asym(full_source)

    mandatory: set[int] = set()
    for chain_id in design_chain_ids:
        mandatory.update(
            int(uid)
            for uid in np.unique(full_source.residue_uid[full_source.chain_id == chain_id])
        )
    if hotspot is not None:
        mandatory.update(source_selector(full_source, hotspot))

    explicit_edge_keys: set[tuple[int, int]] = set()
    declarations = condition.get("bond", []) or []
    if not isinstance(declarations, list):
        raise DesignSampleError("condition.bond must be a list")
    for index, declaration in enumerate(declarations):
        if not isinstance(declaration, Mapping):
            raise DesignSampleError(f"condition.bond[{index}] must be a mapping")
        rows = [
            _source_endpoint_row(
                full_source,
                declaration.get(name),
                design_chain_ids=design_chain_ids,
                where=f"condition.bond[{index}].{name}",
            )
            for name in ("atom1", "atom2")
        ]
        for row in rows:
            if row is not None:
                mandatory.add(int(full_source.residue_uid[row]))
        if rows[0] is not None and rows[1] is not None:
            explicit_edge_keys.add(
                tuple(
                    sorted(
                        (
                            int(full_source.atom_uid[rows[0]]),
                            int(full_source.atom_uid[rows[1]]),
                        )
                    )
                )
            )

    if spec.include is None:
        selected = set(all_residues)
    else:
        selected = set()
        for selector in spec.include:
            selected.update(source_selector(full_source, selector))
    selected.update(mandatory)

    for proximity in spec.proximity:
        try:
            radius = float(proximity.radius)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DesignSampleError("crop proximity radius must be finite and positive") from exc
        if not np.isfinite(radius) or radius <= 0.0:
            raise DesignSampleError("crop proximity radius must be finite and positive")
        anchor_residues = source_selector(full_source, proximity.anchor)
        anchor_rows = [
            row
            for residue_uid in anchor_residues
            for row in residue_rows[residue_uid].tolist()
            if bool(full_source.coord_valid[row])
            and str(full_source.element[row]).strip().upper() not in {"H", "D", "T"}
        ]
        if not anchor_rows:
            raise DesignSampleError(
                f"crop proximity anchor {proximity.anchor!r} has no valid heavy atoms"
            )
        target_rows = [
            row
            for row in range(len(full_source))
            if bool(full_source.coord_valid[row])
            and str(full_source.element[row]).strip().upper() not in {"H", "D", "T"}
        ]
        if target_rows:
            anchor_coords = np.asarray(full_source.coord[anchor_rows], dtype=np.float64)
            target_coords = np.asarray(full_source.coord[target_rows], dtype=np.float64)
            distances_sq = ((target_coords[:, None, :] - anchor_coords[None, :, :]) ** 2).sum(axis=2)
            selected.update(
                int(full_source.residue_uid[target_rows[index]])
                for index in np.flatnonzero(
                    np.any(distances_sq <= radius**2, axis=1)
                )
            )

    physical_groups = _physical_groups(full_source)
    closure_edges: set[tuple[int, int]] = set()
    edge_rows_by_uid: dict[tuple[int, int], tuple[int, int, int]] = {
        tuple(
            sorted(
                (
                    int(full_source.atom_uid[row_i]),
                    int(full_source.atom_uid[row_j]),
                )
            )
        ): (row_i, row_j, order)
        for row_i, row_j, order in normalized_declared
    }

    changed = True
    while changed:
        changed = False
        for group in physical_groups:
            if group.intersection(selected) and not group <= selected:
                selected.update(group)
                changed = True
        for uid_pair, (row_i, row_j, order) in edge_rows_by_uid.items():
            if order == _COORDINATION:
                continue
            if not _edge_closes(full_source, row_i, row_j, residue_order):
                continue
            residue_i = int(full_source.residue_uid[row_i])
            residue_j = int(full_source.residue_uid[row_j])
            if residue_i not in selected and residue_j not in selected:
                continue
            closure_edges.add(uid_pair)
            before = len(selected)
            selected.update((residue_i, residue_j))
            changed |= len(selected) != before

    excluded: set[int] = set()
    for selector in spec.exclude:
        excluded.update(source_selector(full_source, selector))
    mandatory_intersection = mandatory.intersection(excluded)
    if mandatory_intersection:
        raise DesignSampleError(
            "crop exclude removes mandatory residues: "
            f"{sorted(mandatory_intersection)}"
        )
    for group in physical_groups:
        overlap = group.intersection(excluded)
        if overlap and overlap != group and group.intersection(selected):
            raise DesignSampleError(
                "crop exclude intersects only part of an indivisible ligand/molecule "
                f"closure: excluded={sorted(overlap)}, component={sorted(group)}"
            )

    kept_residues = selected.difference(excluded)
    old_uid_to_new_row: dict[int, int] = {}
    keep_mask = np.asarray(
        [int(uid) in kept_residues for uid in full_source.residue_uid], dtype=bool
    )
    cropped = full_source[keep_mask]
    cropped.coord = np.asarray(cropped.coord, dtype=np.float32)
    for row, atom_uid in enumerate(np.asarray(cropped.atom_uid, dtype=np.int64)):
        old_uid_to_new_row[int(atom_uid)] = row

    remapped_source_bonds: list[tuple[int, int, int]] = []
    if full_source.bonds is not None:
        for raw_i, raw_j, raw_order in full_source.bonds.as_array():
            old_i, old_j = int(raw_i), int(raw_j)
            uid_i, uid_j = int(full_source.atom_uid[old_i]), int(full_source.atom_uid[old_j])
            new_i, new_j = old_uid_to_new_row.get(uid_i), old_uid_to_new_row.get(uid_j)
            if new_i is not None and new_j is not None:
                remapped_source_bonds.append((new_i, new_j, int(raw_order)))
    cropped.bonds = struc.BondList(
        len(cropped),
        np.asarray(remapped_source_bonds, dtype=np.uint32)
        if remapped_source_bonds
        else np.empty((0, 3), dtype=np.uint32),
    )

    remapped_declared: list[tuple[int, int, int]] = []
    for uid_pair, (old_i, old_j, order) in edge_rows_by_uid.items():
        new_i, new_j = old_uid_to_new_row.get(uid_pair[0]), old_uid_to_new_row.get(uid_pair[1])
        if new_i is None or new_j is None:
            residue_i = int(full_source.residue_uid[old_i])
            residue_j = int(full_source.residue_uid[old_j])
            active_closed = uid_pair in closure_edges or uid_pair in explicit_edge_keys
            if active_closed and (residue_i in kept_residues) != (residue_j in kept_residues):
                raise DesignSampleError(
                    "crop removes one endpoint of a required/closed source bond"
                )
            continue
        remapped_declared.append((new_i, new_j, order))

    forced_breaks = _forced_break_keys(full_source, kept_residues, design_chain_ids)
    return CropResult(
        source=cropped,
        declared_bonds=tuple(remapped_declared),
        forced_chain_breaks=forced_breaks,
    )


__all__ = ["CropResult", "project_source_crop", "source_selector"]
