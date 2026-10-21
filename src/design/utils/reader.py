from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping as ABCMapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import gzip
import json
from pathlib import Path
import re
from partidesign.paths import resource_root
from typing import Any, Mapping
import warnings
import weakref

import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdbx

from .atoms import (
    COMMON_STANDARD_PARENT,
    DNA_RES_ATOMS,
    PROTEIN_RES_ATOMS,
    RNA_RES_ATOMS,
    assign_identity_uids,
    create_atom_array,
    validate_atom_array,
)
from ..core.records import DesignInputError
from src.common.chemistry.ccd import ccd_topology_components, ccd_topology_ids
from src.common.chemistry.charges import parse_formal_charge
from .source_mapping import COMPLETE_BACKBONE_CRITERIA, attach_source_sequences, mark_source_breaks


_WATER = {"HOH", "WAT", "DOD", "H2O"}
_HYDROGEN_ELEMENTS = frozenset({"H", "D", "T"})
_ORDER_TO_BIOTITE = {
    "SING": int(struc.BondType.SINGLE),
    "SINGLE": int(struc.BondType.SINGLE),
    "DOUB": int(struc.BondType.DOUBLE),
    "DOUBLE": int(struc.BondType.DOUBLE),
    "TRIP": int(struc.BondType.TRIPLE),
    "TRIPLE": int(struc.BondType.TRIPLE),
    "AROM": int(struc.BondType.AROMATIC),
    "AROMATIC": int(struc.BondType.AROMATIC),
    "QUAD": int(struc.BondType.QUADRUPLE),
    "QUADRUPLE": int(struc.BondType.QUADRUPLE),
}
_EXACT_COVALENT_TYPES = frozenset(
    {
        int(struc.BondType.SINGLE),
        int(struc.BondType.DOUBLE),
        int(struc.BondType.TRIPLE),
        int(struc.BondType.AROMATIC),
        int(struc.BondType.AROMATIC_SINGLE),
        int(struc.BondType.AROMATIC_DOUBLE),
        int(struc.BondType.AROMATIC_TRIPLE),
    }
)
_ROW_TEXT_WIDTHS = {
    "chain_id": 32,
    "res_name": 16,
    "atom_name": 32,
    "element": 3,
    "label_entity_id": 128,
    "label_asym_id": 128,
    "label_seq_id": 64,
    "auth_asym_id": 128,
    "auth_seq_id": 64,
    "auth_ins_code": 8,
    "label_alt_id": 8,
}
_DEFAULT_COMPACT_CCD_PATH = (
    resource_root() / "configs" / "resources" / "chemistry" / "ccd.json.gz"
)
_MAX_COMPACT_CCD_COMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_COMPACT_CCD_UNCOMPRESSED_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class _CcdTopology:
    atom_names: frozenset[str]
    bonds: tuple[tuple[str, str, int], ...]
    atom_elements: tuple[tuple[str, str], ...] = ()


class _LazyCcdTopology:
    """Proxy that installs its concrete topology into the owning dict on demand."""

    __slots__ = ("_records", "_component_id", "_owner_ref", "_value")

    def __init__(self, records, component_id: str, owner_ref) -> None:
        self._records = records
        self._component_id = component_id
        self._owner_ref = owner_ref
        self._value: _CcdTopology | None = None

    def _materialize(self) -> _CcdTopology:
        value = self._value
        if value is None:
            record = self._records[self._component_id]
            value = _CcdTopology(
                atom_names=frozenset(record["atoms"]),
                bonds=tuple(
                    (str(left), str(right), int(order))
                    for left, right, order in record["bonds"]
                ),
                atom_elements=tuple(
                    (str(atom), str(element))
                    for atom, element in zip(record["atoms"], record["elements"], strict=True)
                ),
            )
            self._value = value
            owner = self._owner_ref()
            if owner is not None:
                dict.__setitem__(owner, self._component_id, value)
        return value

    def __getattr__(self, name: str):
        return getattr(self._materialize(), name)

    def __repr__(self) -> str:
        return repr(self._materialize())


class _LazyCcdTopologies(dict[str, _CcdTopology]):
    """Native dict index with lazy graph values backed by the shared CCD stream."""

    __slots__ = ("_records", "__weakref__")

    def __init__(self, records, component_ids: tuple[str, ...]) -> None:
        if len(records) != len(component_ids):
            raise DesignInputError(
                "CCD topology index and component inventory have different sizes"
            )
        dict.__init__(self)
        self._records = records
        owner_ref = weakref.ref(self)
        for component_id in component_ids:
            dict.__setitem__(
                self,
                component_id,
                _LazyCcdTopology(records, component_id, owner_ref),
            )


@dataclass(frozen=True)
class _AltlocResidueSelection:
    key: tuple[object, ...]
    label_asym_id: str
    label_seq_ids: frozenset[str]
    auth_locator_keys: frozenset[tuple[str, str]]
    chosen_res_name: str
    chosen_alt_id: str | None
    available_alt_ids: frozenset[str]
    source_atom_alt_ids: tuple[tuple[str, str, frozenset[str]], ...]
    selected_atom_names: frozenset[str]


def _normal_element(value: str, atom_name: str = "") -> str:
    element = str(value).strip().upper()
    if not element or element in {".", "?"}:
        letters = "".join(character for character in str(atom_name) if character.isalpha())
        element = letters[:1].upper()
    if not element or len(element) > 3:
        raise DesignInputError(f"invalid element {value!r} for atom {atom_name!r}")
    return element


def _normal_occupancy(value: object, *, where: str) -> Decimal:
    text = str(value).strip()
    if text in {"", ".", "?"}:
        return Decimal("1")
    try:
        occupancy = Decimal(text)
    except InvalidOperation as exc:
        raise DesignInputError(f"non-numeric occupancy at {where}: {value!r}") from exc
    if not occupancy.is_finite() or occupancy < 0:
        raise DesignInputError(f"invalid occupancy at {where}: {value!r}")
    return occupancy


def _normal_alt_id(value: object) -> str:
    alt_id = str(value).strip()
    return "" if alt_id in {"", ".", "?"} else alt_id


def _identifier_is_present(value: object) -> bool:
    return str(value).strip() not in {"", ".", "?"}


def _restore_source_atom_order(
    rows: list[dict[str, Any]],
    *,
    key: str,
    where: str,
) -> list[dict[str, Any]]:
    """Restore the file author's atom order after a physical row permutation.

    PDB atom serials and the numeric mmCIF ``_atom_site.id`` column are stable
    row identities.  Sorting by them makes a byte-level row permutation
    invisible without changing the historical first-seen chain/entity/token
    semantics for normally authored files.  If a CIF does not provide an
    integral ID, its physical row order remains authoritative as before.
    """

    if len(rows) < 2:
        return rows
    raw_values = [str(row.get(key, "")).strip() for row in rows]
    if any(value in {"", ".", "?"} for value in raw_values):
        return rows
    try:
        values = [int(value) for value in raw_values]
    except ValueError:
        return rows
    if len(set(values)) != len(values):
        raise DesignInputError(f"{where} contains duplicate atom row IDs")
    return [row for _value, row in sorted(zip(values, rows), key=lambda item: item[0])]


def _altloc_residue_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    """Return a physical residue/component identity, not a chemical identity.

    A valid CIF label sequence is authoritative for a polymer site and must
    therefore group microheterogeneous component names and drifting author
    locators.  PDB rows have no independent label sequence, so they always use
    the author residue locator.  Non-polymers normally lack label_seq_id; their
    author locator keeps distinct component instances apart.  If neither
    locator exists, the asym/entity fallback is deliberately fail-closed:
    different component IDs make the selected component ambiguous and repeated
    indistinguishable instances collide on atom IDs, rather than being silently
    split or merged.
    """

    asym_key = row["asym_key"]
    label_seq_id = str(row["label_seq_id"]).strip()
    if bool(row.get("label_seq_authoritative", False)) and _identifier_is_present(
        label_seq_id
    ):
        return (asym_key, "label", label_seq_id)

    auth_seq_id = str(row["auth_seq_id"]).strip()
    if _identifier_is_present(auth_seq_id):
        auth_asym_id = str(row["auth_asym_id"]).strip()
        if not _identifier_is_present(auth_asym_id):
            auth_asym_id = str(row["label_asym_id"]).strip()
        return (
            asym_key,
            "auth",
            auth_asym_id,
            auth_seq_id,
            str(row["auth_ins_code"]),
        )

    return (asym_key, "unlocated", str(row["entity_key"]))


def _select_altloc_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[_AltlocResidueSelection, ...]]:
    """Select one residue/component conformer while retaining shared blank rows."""

    grouped: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[_altloc_residue_key(row)].append(index)

    retained: set[int] = set()
    normalized_component_by_index: dict[int, dict[str, Any]] = {}
    ledger: list[_AltlocResidueSelection] = []
    for key, indices in grouped.items():
        source_res_names = {
            str(rows[index]["res_name"]) for index in indices
        }
        if key[1] == "unlocated" and len(source_res_names) != 1:
            raise DesignInputError(
                "alternate-location component identity is ambiguous for unlocated "
                f"physical residue/component {key}: source component IDs="
                f"{sorted(source_res_names)}"
            )
        blank = [index for index in indices if not str(rows[index]["label_alt_id"])]
        alt_ids = sorted(
            {
                str(rows[index]["label_alt_id"])
                for index in indices
                if str(rows[index]["label_alt_id"])
            }
        )
        if alt_ids:
            scores = {
                alt_id: sum(
                    (
                        rows[index]["occupancy"]
                        for index in indices
                        if str(rows[index]["label_alt_id"]) == alt_id
                        and str(rows[index]["element"]).upper()
                        not in _HYDROGEN_ELEMENTS
                    ),
                    start=Decimal("0"),
                )
                for alt_id in alt_ids
            }
            highest = max(scores.values())
            chosen_alt: str | None = min(
                alt_id for alt_id in alt_ids if scores[alt_id] == highest
            )
            chosen = [
                index
                for index in indices
                if str(rows[index]["label_alt_id"]) == chosen_alt
            ]
        else:
            chosen_alt = None
            chosen = []

        component_exemplar_indices = chosen if chosen else blank
        chosen_res_names = {
            str(rows[index]["res_name"]) for index in component_exemplar_indices
        }
        if len(chosen_res_names) != 1:
            raise DesignInputError(
                "alternate-location component identity is ambiguous for physical "
                f"residue/component {key}: selected component IDs="
                f"{sorted(chosen_res_names)}"
            )
        chosen_res_name = next(iter(chosen_res_names))
        component_exemplar = rows[component_exemplar_indices[0]]

        blank_names = [str(rows[index]["atom_name"]) for index in blank]
        chosen_names = [str(rows[index]["atom_name"]) for index in chosen]
        duplicate_blank = sorted(
            name for name, count in Counter(blank_names).items() if count > 1
        )
        duplicate_chosen = sorted(
            name for name, count in Counter(chosen_names).items() if count > 1
        )
        shared_duplicates = sorted(set(blank_names).intersection(chosen_names))
        if duplicate_blank or duplicate_chosen or shared_duplicates:
            raise DesignInputError(
                "alternate-location atom identity is ambiguous for residue/component "
                f"{key}: duplicate blank atom IDs={duplicate_blank}, "
                f"duplicate chosen atom IDs={duplicate_chosen}, "
                f"blank/chosen overlaps={shared_duplicates}"
            )
        retained.update(blank)
        retained.update(chosen)

        normalized_component = {
            "res_name": chosen_res_name,
            "mol_type": component_exemplar["mol_type"],
            "hetero": component_exemplar["hetero"],
            "auth_asym_id": component_exemplar["auth_asym_id"],
            "auth_seq_id": component_exemplar["auth_seq_id"],
            "auth_ins_code": component_exemplar["auth_ins_code"],
        }
        for index in blank + chosen:
            normalized_component_by_index[index] = normalized_component

        source_atom_alt_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
        for index in indices:
            source_atom_alt_ids[
                (str(rows[index]["res_name"]), str(rows[index]["atom_name"]))
            ].add(str(rows[index]["label_alt_id"]))
        exemplar = rows[indices[0]]
        ledger.append(
            _AltlocResidueSelection(
                key=key,
                label_asym_id=str(exemplar["label_asym_id"]),
                label_seq_ids=frozenset(
                    str(rows[index]["label_seq_id"]) for index in indices
                ),
                auth_locator_keys=frozenset(
                    (
                        (
                            str(rows[index]["auth_asym_id"]).strip()
                            if _identifier_is_present(rows[index]["auth_asym_id"])
                            else str(rows[index]["label_asym_id"]).strip()
                        ),
                        str(rows[index]["auth_seq_id"]).strip(),
                    )
                    for index in indices
                    if _identifier_is_present(rows[index]["auth_seq_id"])
                ),
                chosen_res_name=chosen_res_name,
                chosen_alt_id=chosen_alt,
                available_alt_ids=frozenset(alt_ids),
                source_atom_alt_ids=tuple(
                    (res_name, atom_name, frozenset(values))
                    for (res_name, atom_name), values in sorted(
                        source_atom_alt_ids.items()
                    )
                ),
                selected_atom_names=frozenset(blank_names + chosen_names),
            )
        )

    selected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index not in retained:
            continue
        normalized = dict(row)
        normalized.update(normalized_component_by_index[index])
        normalized["label_alt_id"] = ""
        selected.append(normalized)
    return selected, tuple(ledger)


def _modality(
    res_name: str,
    chem_type: str | None = None,
    *,
    parent_sources: Mapping[str, str | None] | None = None,
    declared_modality: str | None = None,
) -> str:
    name = str(res_name).strip().upper()
    type_name = "" if chem_type is None else str(chem_type).strip().upper()
    # A concrete CCD chemical type describes the component that is actually
    # present in this file.  It must be consulted before the small historical
    # name->parent fallback table: e.g. 5HM is ``RNA LINKING`` even though an
    # older fallback entry called it a DNA parent (DC).  Only an absent or
    # uninformative chemical type falls through to parent/name inference.
    if "SACCHARIDE" in type_name or "CARBOHYDRATE" in type_name:
        inferred = "ligand"
    elif "PEPTIDE" in type_name or "PROTEIN" in type_name:
        inferred = "protein"
    elif "DNA" in type_name or "DEOXYRIBO" in type_name:
        inferred = "dna"
    elif "RNA" in type_name or "RIBONUCLE" in type_name:
        inferred = "rna"
    else:
        if parent_sources is not None and name in parent_sources:
            # ``None`` is a deliberate marker for a concrete CCD declaration
            # with no unambiguous parent.  Its presence must suppress the
            # historical name fallback; only an absent mapping entry may use
            # COMMON_STANDARD_PARENT.
            parent = parent_sources[name]
            name = name if parent is None else str(parent).strip().upper()
        else:
            name = COMMON_STANDARD_PARENT.get(name, name)
        if name in PROTEIN_RES_ATOMS:
            inferred = "protein"
        elif name in DNA_RES_ATOMS:
            inferred = "dna"
        elif name in RNA_RES_ATOMS:
            inferred = "rna"
        else:
            inferred = "ligand"
    if declared_modality is None:
        return inferred
    declared = str(declared_modality).strip().lower()
    if declared not in {"protein", "rna", "dna", "ligand"}:
        raise DesignInputError(
            f"invalid declared chain modality {declared_modality!r}"
        )
    # Explicit ligand ownership is authoritative at task-input and private
    # reward-output reads.  An isolated ligand may use a polymer CCD ID such as
    # 5MC/MSE/ALA; residue chemistry alone cannot establish polymer membership.
    # The task compiler separately requires exactly one source component.
    if declared == "ligand":
        return declared
    if inferred == "ligand":
        return declared
    if inferred != declared:
        raise DesignInputError(
            f"residue {res_name!r} has {inferred} chemistry but chain is declared {declared}"
        )
    return inferred


def _factor_residue_positions(rows: list[dict[str, Any]], *, use_label_seq: bool) -> list[int]:
    positions: list[int] = []
    mapping: dict[tuple[Any, ...], int] = {}
    next_by_asym: dict[Any, int] = defaultdict(lambda: 1)
    for row in rows:
        asym = row["asym_key"]
        label_seq = str(row["label_seq_id"])
        if use_label_seq and label_seq not in {"", ".", "?"}:
            try:
                position = int(label_seq)
            except ValueError as exc:
                raise DesignInputError(f"non-integral label_seq_id: {label_seq!r}") from exc
            if position <= 0:
                raise DesignInputError(f"label_seq_id must be positive: {label_seq!r}")
            next_by_asym[asym] = max(next_by_asym[asym], position + 1)
        else:
            key = (
                asym,
                label_seq,
                row["auth_seq_id"],
                row["auth_ins_code"],
                row["res_name"],
            )
            if key not in mapping:
                mapping[key] = next_by_asym[asym]
                next_by_asym[asym] += 1
            position = mapping[key]
        positions.append(position)
    return positions


def _rows_to_atom_array(rows: list[dict[str, Any]], *, use_label_seq: bool) -> struc.AtomArray:
    kept = [
        row
        for row in rows
        if row["res_name"].upper() not in _WATER
        and row["element"].upper() not in _HYDROGEN_ELEMENTS
    ]
    if not kept:
        raise DesignInputError("input contains no non-water heavy atoms in the first model")
    for row_index, row in enumerate(kept):
        for field, width in _ROW_TEXT_WIDTHS.items():
            value = str(row[field])
            if len(value) > width:
                raise DesignInputError(
                    f"input row {row_index} field {field} exceeds width {width}"
                )
    atoms = create_atom_array(len(kept))
    atoms.source_format[:] = "cif" if use_label_seq else "pdb"
    res_positions = _factor_residue_positions(kept, use_label_seq=use_label_seq)
    atoms.coord = np.asarray([row["coord"] for row in kept], dtype=np.float32)
    atoms.coord_valid[:] = True
    atoms.chain_id = np.asarray([row["chain_id"] for row in kept], dtype="<U32")
    atoms.res_id = np.asarray(res_positions, dtype=np.int64)
    atoms.ins_code = np.asarray([row["auth_ins_code"] for row in kept], dtype="<U8")
    atoms.res_name = np.asarray([row["res_name"] for row in kept], dtype="<U16")
    atoms.atom_name = np.asarray([row["atom_name"] for row in kept], dtype="<U32")
    atoms.element = np.asarray([row["element"] for row in kept], dtype="<U3")
    atoms.formal_charge = np.asarray([row.get("formal_charge") for row in kept], dtype=object)
    atoms.hetero = np.asarray([row["hetero"] for row in kept], dtype=np.bool_)
    atoms.mol_type = np.asarray([row["mol_type"] for row in kept], dtype="<U8")
    atoms.label_entity_id = np.asarray([row["label_entity_id"] for row in kept], dtype="<U128")
    atoms.label_asym_id = np.asarray([row["label_asym_id"] for row in kept], dtype="<U128")
    atoms.label_seq_id = np.asarray([row["label_seq_id"] for row in kept], dtype="<U64")
    atoms.auth_asym_id = np.asarray([row["auth_asym_id"] for row in kept], dtype="<U128")
    atoms.auth_seq_id = np.asarray([row["auth_seq_id"] for row in kept], dtype="<U64")
    atoms.auth_ins_code = np.asarray([row["auth_ins_code"] for row in kept], dtype="<U8")
    atoms.label_alt_id = np.asarray([row["label_alt_id"] for row in kept], dtype="<U8")
    asym_keys = [row["asym_key"] for row in kept]
    residue_keys = [
        (row["asym_key"], position, row["auth_ins_code"])
        for row, position in zip(kept, res_positions)
    ]
    entity_keys = [row["entity_key"] for row in kept]
    component_keys = [
        (row["asym_key"], "polymer") if row["mol_type"] != "ligand" else (residue_key, row["res_name"])
        for row, residue_key in zip(kept, residue_keys)
    ]
    assign_identity_uids(
        atoms,
        entity_keys=entity_keys,
        asym_keys=asym_keys,
        component_keys=component_keys,
        molecule_keys=component_keys,
        residue_keys=residue_keys,
    )
    return atoms


def _merge_bonds(
    atom_count: int,
    bond_groups: list[list[tuple[int, int, int]]],
) -> struc.BondList:
    merged: dict[tuple[int, int], int] = {}
    for group in bond_groups:
        for atom_i, atom_j, order in group:
            if atom_i == atom_j:
                raise DesignInputError("self bonds are not valid structure facts")
            key = tuple(sorted((int(atom_i), int(atom_j))))
            current = merged.get(key)
            if current is not None and current != int(order):
                if current == int(struc.BondType.ANY):
                    merged[key] = int(order)
                    continue
                if int(order) == int(struc.BondType.ANY):
                    continue
                raise DesignInputError(f"conflicting bond orders for atom rows {key}")
            merged[key] = int(order)
    rows = np.asarray([(i, j, order) for (i, j), order in sorted(merged.items())], dtype=np.uint32)
    if rows.size == 0:
        rows = np.empty((0, 3), dtype=np.uint32)
    return struc.BondList(atom_count, rows)


def polymer_template_bond_is_supported(atoms: struc.AtomArray, row_i: int, row_j: int) -> bool:
    """Avoid inferring a peptide bond across an unresolved source segment.

    Author numbering can legitimately skip numbers (e.g. thrombin), so a
    numbering gap alone is insufficient. Require both an author gap and an
    incompatible observed C--N distance. Explicit file/task bonds are merged
    separately and generated design residues retain their intended topology.
    """
    if int(atoms.residue_uid[row_i]) == int(atoms.residue_uid[row_j]):
        return True
    if not (bool(atoms.is_design[row_i]) or bool(atoms.is_design[row_j])):
        later = row_i if int(atoms.residue_uid[row_i]) > int(atoms.residue_uid[row_j]) else row_j
        if bool(atoms.source_break_before[later]):
            return False
    if atoms.mol_type[row_i] != "protein" or atoms.mol_type[row_j] != "protein":
        return True
    if {str(atoms.atom_name[row_i]), str(atoms.atom_name[row_j])} != {"C", "N"}:
        return True
    if bool(atoms.is_design[row_i]) or bool(atoms.is_design[row_j]):
        return True
    try:
        author_gap = abs(int(atoms.auth_seq_id[row_i]) - int(atoms.auth_seq_id[row_j]))
    except (TypeError, ValueError):
        return True
    if author_gap <= 1 or not (bool(atoms.coord_valid[row_i]) and bool(atoms.coord_valid[row_j])):
        return True
    distance = float(np.linalg.norm(atoms.coord[row_i] - atoms.coord[row_j]))
    return not (np.isfinite(distance) and distance > 2.0)


def _polymer_template_bonds(atoms: struc.AtomArray) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for asym_uid in np.unique(atoms.asym_uid):
        indices = np.where((atoms.asym_uid == asym_uid) & (atoms.mol_type != "ligand"))[0]
        if not len(indices):
            continue
        subset = atoms[indices]
        try:
            bonds = struc.connect_via_residue_names(subset, inter_residue=True)
        except Exception:
            continue
        for local_i, local_j, order in bonds.as_array():
            row_i, row_j = int(indices[int(local_i)]), int(indices[int(local_j)])
            if polymer_template_bond_is_supported(atoms, row_i, row_j):
                result.append((row_i, row_j, int(order)))
    return result


def _template_order_matches(declared: int, template: int) -> bool:
    """Only an aromatic subtype retains the same known Kekule base order."""
    return int(declared) == int(template) or frozenset((int(declared), int(template))) in (
        frozenset((1, 5)), frozenset((2, 6)), frozenset((3, 7)),
    )


def _merge_template_and_declared_bonds(
    atom_count: int,
    template_bonds: list[tuple[int, int, int]],
    declared_bonds: list[tuple[int, int, int]],
) -> struc.BondList:
    # Concrete explicit facts must agree exactly, including aromaticity.
    # Only the template boundary admits the known base/subtype correspondence;
    # retain the explicit file representation rather than rewriting its facts.
    declared = _merge_bonds(atom_count, [declared_bonds])
    orders = {(int(i), int(j)): int(o) for i, j, o in declared.as_array()}
    missing = []
    for i, j, order in _merge_bonds(atom_count, [template_bonds]).as_array():
        key = (int(i), int(j))
        if key not in orders:
            missing.append((int(i), int(j), int(order)))
        elif orders[key] == int(struc.BondType.ANY):
            # Preserve the pre-existing unknown-positive template refinement.
            missing.append((int(i), int(j), int(order)))
        elif int(order) == int(struc.BondType.ANY):
            continue
        elif not _template_order_matches(orders[key], int(order)):
            raise DesignInputError(f"conflicting bond orders for atom rows {key}")
    return _merge_bonds(atom_count, [declared.as_array().tolist(), missing])


def _file_formal_charge(value: object, *, where: str, pdb_style: bool = False) -> int | None:
    """Keep file charge syntax strict while sharing runtime integer semantics."""
    text = str(value).strip()
    if not text or (not pdb_style and text in {".", "?"}):
        return None
    pattern = r"(?:0|[0-9][+-]|[+-][0-9])" if pdb_style else r"[+-]?[0-9]+"
    if re.fullmatch(pattern, text) is None:
        raise DesignInputError(f"invalid explicit formal charge at {where}: {value!r}")
    return parse_formal_charge(text, where=where)


def _read_pdb(
    path: Path,
    *,
    parent_sources: Mapping[str, str | None] | None = None,
    declared_polymer_chain_modalities: Mapping[str, str] | None = None,
) -> tuple[struc.AtomArray, list[tuple[int, int, int]], dict[int, int]]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    if not any(line.startswith(("ATOM  ", "HETATM")) for line in lines):
        raise DesignInputError(f"PDB content has no ATOM/HETATM rows: {path}")
    first_model: str | None = None
    active = True
    segment = 0
    label_by_asym_key: dict[tuple[str, int], str] = {}
    segment_count_by_chain: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    serial_to_original: dict[int, int] = {}
    conect: list[tuple[int, int, int]] = []
    for line in lines:
        record = line[:6].strip().upper()
        if record == "MODEL":
            model_id = line[10:14].strip() or str(len(rows))
            if first_model is None:
                first_model = model_id
                active = True
            else:
                active = model_id == first_model
            continue
        if record == "ENDMDL":
            if active:
                active = False
            continue
        if record == "TER" and active:
            segment += 1
            continue
        if record in {"ATOM", "HETATM"} and active:
            if len(line) < 54:
                raise DesignInputError(f"short PDB coordinate row: {line!r}")
            altloc = _normal_alt_id(line[16:17])
            try:
                serial = int(line[6:11])
                auth_seq = line[22:26].strip()
                int(auth_seq)
                coord = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError as exc:
                raise DesignInputError(f"invalid PDB atom row: {line!r}") from exc
            if serial in serial_to_original:
                raise DesignInputError(
                    f"duplicate PDB atom serial {serial} in the first model"
                )
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip().upper()
            chain = line[21:22].strip()
            if not chain:
                raise DesignInputError("blank PDB chain IDs are unsupported")
            asym_key = (chain, segment)
            if asym_key not in label_by_asym_key:
                segment_count_by_chain[chain] += 1
                ordinal = segment_count_by_chain[chain]
                label_by_asym_key[asym_key] = (
                    chain if ordinal == 1 else f"{chain}_{ordinal}"
                )
            element = _normal_element(line[76:78] if len(line) >= 78 else "", atom_name)
            row = {
                "chain_id": chain,
                "asym_key": asym_key,
                "entity_key": asym_key,
                "res_name": res_name,
                "atom_name": atom_name,
                "element": element,
                "formal_charge": _file_formal_charge(
                    line[78:80], where=f"PDB atom serial {serial}", pdb_style=True
                ),
                "coord": coord,
                "hetero": record == "HETATM",
                "mol_type": _modality(
                    res_name,
                    parent_sources=parent_sources,
                    declared_modality=(
                        None
                        if declared_polymer_chain_modalities is None
                        else declared_polymer_chain_modalities.get(chain)
                    ),
                ),
                "label_entity_id": chain,
                "label_asym_id": label_by_asym_key[asym_key],
                "label_seq_id": auth_seq,
                "label_seq_authoritative": False,
                "auth_asym_id": chain,
                "auth_seq_id": auth_seq,
                "auth_ins_code": line[26:27].strip(),
                "label_alt_id": altloc,
                "occupancy": _normal_occupancy(
                    line[54:60] if len(line) >= 60 else "",
                    where=f"PDB atom serial {serial}",
                ),
                "serial": serial,
            }
            serial_to_original[serial] = len(rows)
            rows.append(row)
        elif record == "CONECT":
            fields = line[6:].split()
            if len(fields) >= 2:
                try:
                    source = int(fields[0])
                    targets = [int(value) for value in fields[1:]]
                except ValueError as exc:
                    raise DesignInputError(f"invalid CONECT row: {line!r}") from exc
                for target in targets:
                    conect.append((source, target, int(struc.BondType.ANY)))
    rows, _ = _select_altloc_rows(rows)
    rows = _restore_source_atom_order(
        rows, key="serial", where="PDB first model"
    )
    atoms = _rows_to_atom_array(rows, use_label_seq=False)
    attach_source_sequences(atoms, pdb_lines=lines)
    kept_serial_to_row: dict[int, int] = {}
    kept_key_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    for row_index in range(len(atoms)):
        key = (
            atoms.label_asym_id[row_index],
            atoms.auth_asym_id[row_index],
            atoms.auth_seq_id[row_index],
            atoms.auth_ins_code[row_index],
            atoms.atom_name[row_index],
        )
        kept_key_counts[key] += 1
    for source in rows:
        key = (
            source["label_asym_id"],
            source["auth_asym_id"],
            source["auth_seq_id"],
            source["auth_ins_code"],
            source["atom_name"],
        )
        if (
            source["res_name"] in _WATER
            or source["element"] in _HYDROGEN_ELEMENTS
            or kept_key_counts[key] != 1
        ):
            continue
        matches = np.where(
            (atoms.label_asym_id == key[0])
            & (atoms.auth_asym_id == key[1])
            & (atoms.auth_seq_id == key[2])
            & (atoms.auth_ins_code == key[3])
            & (atoms.atom_name == key[4])
        )[0]
        if len(matches) == 1:
            kept_serial_to_row[int(source["serial"])] = int(matches[0])
    file_bonds: list[tuple[int, int, int]] = []
    for serial_i, serial_j, order in conect:
        if serial_i in kept_serial_to_row and serial_j in kept_serial_to_row:
            file_bonds.append((kept_serial_to_row[serial_i], kept_serial_to_row[serial_j], order))
    return atoms, file_bonds, kept_serial_to_row


def _column(category: Any, name: str, length: int, default: str = "") -> np.ndarray:
    if name not in category:
        return np.full(length, default, dtype=object)
    return category[name].as_array(str, default)


def _chem_comp_types(block: Any) -> dict[str, str]:
    if "chem_comp" not in block:
        return {}
    category = block["chem_comp"]
    ids = _column(category, "id", category.row_count)
    types = _column(category, "type", category.row_count)
    return {str(comp).upper(): str(kind) for comp, kind in zip(ids, types)}


def _chem_comp_atom_charges(block: Any) -> dict[tuple[str, str], int]:
    """Read only embedded, precisely named atom charge declarations."""
    if "chem_comp_atom" not in block or "charge" not in block["chem_comp_atom"]:
        return {}
    category = block["chem_comp_atom"]
    n = category.row_count
    result: dict[tuple[str, str], int] = {}
    for component, name, value in zip(
        _column(category, "comp_id", n),
        _column(category, "atom_id", n),
        _column(category, "charge", n),
        strict=True,
    ):
        charge = _file_formal_charge(value, where=f"chem_comp_atom {component}:{name}")
        if charge is None:
            continue
        key = (str(component).strip().upper(), str(name).strip())
        if any(part in {"", ".", "?"} for part in key):
            raise DesignInputError("chem_comp_atom charge lacks a component or atom ID")
        previous = result.get(key)
        if previous is not None and previous != charge:
            raise DesignInputError(f"conflicting chem_comp_atom charge for {key[0]}:{key[1]}")
        result[key] = charge
    return result


def _read_cif_rows(
    path: Path,
    *,
    parent_sources: Mapping[str, str | None] | None = None,
    declared_polymer_chain_modalities: Mapping[str, str] | None = None,
) -> tuple[
    struc.AtomArray,
    Any,
    list[dict[str, Any]],
    tuple[_AltlocResidueSelection, ...],
]:
    try:
        cif = pdbx.CIFFile.read(path)
    except Exception as exc:
        raise DesignInputError(f"cannot parse CIF {path}: {exc}") from exc
    if not cif:
        raise DesignInputError(f"CIF contains no data block: {path}")
    block = cif[next(iter(cif.keys()))]
    if "atom_site" not in block:
        raise DesignInputError(f"CIF contains no atom_site category: {path}")
    category = block["atom_site"]
    n = category.row_count
    model = _column(category, "pdbx_PDB_model_num", n, "1")
    first_model = str(model[0]) if n else "1"
    altloc = _column(category, "label_alt_id", n, ".")
    fields = {
        name: _column(category, name, n, default)
        for name, default in {
            "id": "", "group_PDB": "ATOM", "type_symbol": "", "label_atom_id": "",
            "label_comp_id": "", "label_asym_id": "", "label_entity_id": "",
            "label_seq_id": ".", "auth_asym_id": "", "auth_seq_id": ".",
            "pdbx_PDB_ins_code": "", "Cartn_x": "", "Cartn_y": "", "Cartn_z": "",
            "occupancy": "1.0",
            "pdbx_formal_charge": "",
        }.items()
    }
    chem_types = _chem_comp_types(block)
    component_charges = _chem_comp_atom_charges(block)
    entity_types = {}
    if "entity" in block:
        entities = block["entity"]
        entity_types = {
            str(entity).strip(): str(kind).strip().lower()
            for entity, kind in zip(
                _column(entities, "id", entities.row_count),
                _column(entities, "type", entities.row_count),
            )
        }
    rows: list[dict[str, Any]] = []
    for index in range(n):
        if str(model[index]) != first_model:
            continue
        asym = str(fields["label_asym_id"][index]).strip()
        if not asym or asym in {".", "?"}:
            raise DesignInputError("CIF atom_site requires label_asym_id")
        atom_name = str(fields["label_atom_id"][index]).strip()
        res_name = str(fields["label_comp_id"][index]).strip().upper()
        charge = _file_formal_charge(
            fields["pdbx_formal_charge"][index], where=f"CIF atom_site row {index}"
        )
        component_charge = component_charges.get((res_name, atom_name))
        if component_charge is not None:
            if charge is not None and charge != component_charge:
                raise DesignInputError(f"atom_site/chem_comp_atom charge conflict for {res_name}:{atom_name}")
            charge = component_charge
        try:
            coord = tuple(float(fields[axis][index]) for axis in ("Cartn_x", "Cartn_y", "Cartn_z"))
        except ValueError as exc:
            raise DesignInputError(f"non-numeric CIF coordinate at atom_site row {index}") from exc
        auth_ins = str(fields["pdbx_PDB_ins_code"][index]).strip()
        if auth_ins in {".", "?"}:
            auth_ins = ""
        entity = str(fields["label_entity_id"][index]).strip()
        if entity in {"", ".", "?"}:
            entity = asym
        auth_asym = str(fields["auth_asym_id"][index]).strip()
        if not _identifier_is_present(auth_asym):
            auth_asym = asym
        declared_modality = (
            None
            if declared_polymer_chain_modalities is None
            else declared_polymer_chain_modalities.get(asym)
        )
        # Entity membership distinguishes a free amino acid from a polymer
        # residue; an explicit caller declaration retains its precedence.
        if declared_modality is None and entity_types.get(entity) == "non-polymer":
            declared_modality = "ligand"
        rows.append(
            {
                "chain_id": asym,
                "asym_key": asym,
                "entity_key": entity,
                "res_name": res_name,
                "atom_name": atom_name,
                "element": _normal_element(fields["type_symbol"][index], atom_name),
                "formal_charge": charge,
                "coord": coord,
                "hetero": str(fields["group_PDB"][index]).strip().upper() != "ATOM",
                "mol_type": _modality(
                    res_name,
                    chem_types.get(res_name),
                    parent_sources=parent_sources,
                    declared_modality=declared_modality,
                ),
                "label_entity_id": entity,
                "label_asym_id": asym,
                "label_seq_id": str(fields["label_seq_id"][index]).strip(),
                "label_seq_authoritative": _identifier_is_present(
                    fields["label_seq_id"][index]
                ),
                "auth_asym_id": auth_asym,
                "auth_seq_id": str(fields["auth_seq_id"][index]).strip(),
                "auth_ins_code": auth_ins,
                "label_alt_id": _normal_alt_id(altloc[index]),
                "occupancy": _normal_occupancy(
                    fields["occupancy"][index], where=f"CIF atom_site row {index}"
                ),
                "source_atom_id": str(fields["id"][index]).strip(),
            }
        )
    rows, altloc_ledger = _select_altloc_rows(rows)
    rows = _restore_source_atom_order(
        rows, key="source_atom_id", where="CIF atom_site"
    )
    return _rows_to_atom_array(rows, use_label_seq=True), block, rows, altloc_ledger


def _chem_comp_atom_elements(block: Any) -> dict[str, dict[str, str]]:
    if "chem_comp_atom" not in block:
        return {}
    category = block["chem_comp_atom"]
    n = category.row_count
    components = _column(category, "comp_id", n)
    names = _column(category, "atom_id", n)
    elements = _column(category, "type_symbol", n)
    result: dict[str, dict[str, str]] = defaultdict(dict)
    for raw_component, raw_name, raw_element in zip(
        components, names, elements, strict=True
    ):
        component = str(raw_component).strip().upper()
        name = str(raw_name).strip()
        if component in {"", ".", "?"} or name in {"", ".", "?"}:
            raise DesignInputError("chem_comp_atom contains a missing component or atom ID")
        raw_element_text = str(raw_element).strip()
        element = (
            ""
            if raw_element_text in {"", ".", "?"}
            else _normal_element(raw_element_text)
        )
        previous = result[component].get(name)
        if previous is not None and previous != element:
            raise DesignInputError(
                f"chem_comp_atom element conflict for {component}:{name}"
            )
        result[component][name] = element
    return {component: dict(values) for component, values in result.items()}


def _cif_component_bond_type(order_name: str, aromatic_flag: str) -> int:
    """Decode both explicit CIF bond facts without discarding Kekule order."""
    flag = str(aromatic_flag).strip().upper()
    if flag not in {"", ".", "?", "Y", "N"}:
        raise DesignInputError(f"invalid chem_comp_bond aromatic flag: {aromatic_flag!r}")
    # An explicit aromatic fact does not require a specified Kekule order.
    # Missing markers are not contradictory declarations; malformed text is.
    if flag == "Y" and str(order_name).strip() in {"", ".", "?"}:
        return int(struc.BondType.AROMATIC)
    bond_type = _ORDER_TO_BIOTITE.get(
        str(order_name).strip().upper(), int(struc.BondType.ANY)
    )
    if flag == "Y":
        aromatic_types = {
            int(struc.BondType.SINGLE): int(struc.BondType.AROMATIC_SINGLE),
            int(struc.BondType.DOUBLE): int(struc.BondType.AROMATIC_DOUBLE),
            int(struc.BondType.TRIPLE): int(struc.BondType.AROMATIC_TRIPLE),
            int(struc.BondType.AROMATIC): int(struc.BondType.AROMATIC),
        }
        if bond_type not in aromatic_types:
            raise DesignInputError(
                f"chem_comp_bond aromatic flag Y is incompatible with order {order_name!r}"
            )
        return aromatic_types[bond_type]
    if flag == "N" and bond_type == int(struc.BondType.AROMATIC):
        raise DesignInputError("chem_comp_bond aromatic order contradicts aromatic flag N")
    return bond_type


def _cif_component_bonds(block: Any, atoms: struc.AtomArray) -> list[tuple[int, int, int]]:
    if "chem_comp_bond" not in block:
        return []
    category = block["chem_comp_bond"]
    n = category.row_count
    comp = _column(category, "comp_id", n)
    atom1 = _column(category, "atom_id_1", n)
    atom2 = _column(category, "atom_id_2", n)
    order = _column(category, "value_order", n, "")
    aromatic = _column(category, "pdbx_aromatic_flag", n, "")
    atom_elements = _chem_comp_atom_elements(block)
    standard_by_modality = {
        "protein": set(PROTEIN_RES_ATOMS),
        "rna": set(RNA_RES_ATOMS),
        "dna": set(DNA_RES_ATOMS),
    }
    result: list[tuple[int, int, int]] = []
    for comp_id, name1, name2, order_name, aromatic_flag in zip(comp, atom1, atom2, order, aromatic):
        component = str(comp_id).strip().upper()
        endpoint_names = (str(name1).strip(), str(name2).strip())
        comp_mask = atoms.res_name == component
        if not comp_mask.any():
            continue
        bond_type = _cif_component_bond_type(order_name, aromatic_flag)
        for modality in sorted(set(str(value) for value in atoms.mol_type[comp_mask])):
            modality_mask = comp_mask & (atoms.mol_type == modality)
            if modality == "ligand":
                scope_values = atoms.component_uid
            elif component not in standard_by_modality.get(modality, set()):
                scope_values = atoms.residue_uid
                if bond_type not in _EXACT_COVALENT_TYPES:
                    raise DesignInputError(
                        "modified-polymer chem_comp_bond requires an exact "
                        f"single/double/triple/aromatic order for {component}:"
                        f"{endpoint_names[0]}-{endpoint_names[1]}"
                    )
            else:
                continue

            for scope_uid in np.unique(scope_values[modality_mask]):
                indices = np.where(modality_mask & (scope_values == scope_uid))[0]
                resolved: list[int | None] = []
                for endpoint_name in endpoint_names:
                    matches = indices[atoms.atom_name[indices] == endpoint_name]
                    if len(matches) == 1:
                        resolved.append(int(matches[0]))
                        continue
                    if len(matches) > 1:
                        raise DesignInputError(
                            "chem_comp_bond endpoint is ambiguous for "
                            f"{component}:{endpoint_name}"
                        )
                    endpoint_element = atom_elements.get(component, {}).get(
                        endpoint_name
                    )
                    if endpoint_element in _HYDROGEN_ELEMENTS:
                        resolved.append(None)
                        continue
                    raise DesignInputError(
                        "chem_comp_bond heavy endpoint is absent for "
                        f"{component}:{endpoint_name}"
                    )
                if any(value is None for value in resolved):
                    continue
                result.append((int(resolved[0]), int(resolved[1]), int(bond_type)))
    return result


def _struct_conn_endpoint_locator(
    category: Any,
    prefix: str,
    row: int,
) -> tuple[str, str, str, str, str, str, str]:
    n = category.row_count
    asym = str(_column(category, f"{prefix}_label_asym_id", n)[row]).strip()
    seq = str(_column(category, f"{prefix}_label_seq_id", n, ".")[row]).strip()
    auth_asym = str(
        _column(category, f"{prefix}_auth_asym_id", n, asym)[row]
    ).strip()
    auth_seq = str(_column(category, f"{prefix}_auth_seq_id", n, ".")[row]).strip()
    comp = str(_column(category, f"{prefix}_label_comp_id", n)[row]).strip().upper()
    atom = str(_column(category, f"{prefix}_label_atom_id", n)[row]).strip()
    alt_id = _normal_alt_id(
        _column(category, f"pdbx_{prefix}_label_alt_id", n, ".")[row]
    )
    if asym in {"", ".", "?"} or comp in {"", ".", "?"} or atom in {
        "",
        ".",
        "?",
    }:
        raise DesignInputError(
            f"struct_conn endpoint {prefix} has a missing label locator"
        )
    if not _identifier_is_present(auth_asym):
        auth_asym = asym
    if not _identifier_is_present(auth_seq):
        auth_seq = "."
    return asym, seq, comp, atom, alt_id, auth_asym, auth_seq


def _struct_conn_alt_is_selected(
    locator: tuple[str, str, str, str, str, str, str],
    altloc_ledger: tuple[_AltlocResidueSelection, ...],
) -> bool:
    asym, seq, comp, atom, alt_id, auth_asym, auth_seq = locator
    candidates = [
        entry
        for entry in altloc_ledger
        if entry.label_asym_id == asym
        and (
            not _identifier_is_present(seq)
            or seq in entry.label_seq_ids
        )
    ]
    if not _identifier_is_present(seq) and _identifier_is_present(auth_seq):
        candidates = [
            entry
            for entry in candidates
            if (auth_asym, auth_seq) in entry.auth_locator_keys
        ]
    if not _identifier_is_present(seq) and not _identifier_is_present(auth_seq):
        candidates = [
            entry
            for entry in candidates
            if (comp, atom)
            in {
                (source_comp, source_atom)
                for source_comp, source_atom, _ in entry.source_atom_alt_ids
            }
            or (
                entry.chosen_res_name == comp
                and atom in entry.selected_atom_names
            )
        ]
    if len(candidates) != 1:
        raise DesignInputError(
            f"struct_conn endpoint {asym}/{seq}/{comp}/{atom} matched "
            f"{len(candidates)} physical source residue/components"
        )
    selection = candidates[0]
    source_atom_alt_ids = {
        (source_comp, source_atom): source_alt_ids
        for source_comp, source_atom, source_alt_ids
        in selection.source_atom_alt_ids
    }
    atom_alt_ids = source_atom_alt_ids.get((comp, atom))
    selected_identity_matches = (
        selection.chosen_res_name == comp
        and atom in selection.selected_atom_names
    )

    if alt_id:
        if alt_id not in selection.available_alt_ids:
            raise DesignInputError(
                f"struct_conn endpoint {asym}/{seq}/{comp}/{atom} names unknown "
                f"alternate location {alt_id!r}"
            )
        if alt_id != selection.chosen_alt_id:
            if atom_alt_ids is None or (
                "" not in atom_alt_ids and alt_id not in atom_alt_ids
            ):
                raise DesignInputError(
                    f"struct_conn endpoint {asym}/{seq}/{comp}/{atom} is not present "
                    f"in alternate location {alt_id!r}"
                )
            return False
        if not selected_identity_matches:
            if atom_alt_ids is None:
                raise DesignInputError(
                    f"struct_conn endpoint {asym}/{seq}/{comp}/{atom} is absent from "
                    "the source atom_site"
                )
            raise DesignInputError(
                f"struct_conn endpoint {asym}/{seq}/{comp}/{atom} is absent from "
                "the selected alternate location"
            )
        return True

    if selected_identity_matches:
        return True
    if atom_alt_ids is None:
        raise DesignInputError(
            f"struct_conn endpoint {asym}/{seq}/{comp}/{atom} is absent from "
            "the source atom_site"
        )
    raise DesignInputError(
        f"struct_conn endpoint {asym}/{seq}/{comp}/{atom} is absent from "
        "the selected alternate location"
    )


def _resolve_struct_conn_endpoint(
    atoms: struc.AtomArray,
    locator: tuple[str, str, str, str, str, str, str],
) -> int:
    asym, seq, comp, atom, _, auth_asym, auth_seq = locator
    mask = (atoms.label_asym_id == asym) & (atoms.res_name == comp) & (atoms.atom_name == atom)
    if seq not in {"", ".", "?"}:
        mask &= atoms.label_seq_id == seq
    elif _identifier_is_present(auth_seq):
        mask &= atoms.auth_asym_id == auth_asym
        mask &= atoms.auth_seq_id == auth_seq
    matches = np.where(mask)[0]
    if len(matches) != 1:
        raise DesignInputError(f"struct_conn endpoint {asym}/{seq}/{comp}/{atom} matched {len(matches)} atoms")
    return int(matches[0])


def _cif_struct_conn_bonds(
    block: Any,
    atoms: struc.AtomArray,
    altloc_ledger: tuple[_AltlocResidueSelection, ...],
) -> list[tuple[int, int, int]]:
    if "struct_conn" not in block:
        return []
    category = block["struct_conn"]
    connection_type = _column(category, "conn_type_id", category.row_count, "")
    order = _column(category, "pdbx_value_order", category.row_count, "")
    result = []
    for row in range(category.row_count):
        kind = str(connection_type[row]).strip().lower()
        if kind not in {"covale", "disulf", "metalc"}:
            continue
        locators = tuple(
            _struct_conn_endpoint_locator(category, prefix, row)
            for prefix in ("ptnr1", "ptnr2")
        )
        selected = tuple(
            _struct_conn_alt_is_selected(locator, altloc_ledger)
            for locator in locators
        )
        endpoints = tuple(
            _resolve_struct_conn_endpoint(atoms, locator) if keep else None
            for locator, keep in zip(locators, selected, strict=True)
        )
        if any(endpoint is None for endpoint in endpoints):
            continue
        bond_type = (
            int(struc.BondType.COORDINATION)
            if kind == "metalc"
            else _ORDER_TO_BIOTITE.get(
                str(order[row]).upper(), int(struc.BondType.ANY)
            )
        )
        result.append((int(endpoints[0]), int(endpoints[1]), bond_type))
    return result


def _ccd_cif_topologies(path: Path) -> dict[str, _CcdTopology]:
    try:
        cif = pdbx.CIFFile.read(path)
    except Exception as exc:
        raise DesignInputError(f"cannot parse CCD file {path}: {exc}") from exc
    atom_names: dict[str, set[str]] = defaultdict(set)
    atom_elements: dict[str, dict[str, str]] = defaultdict(dict)
    bonds: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    component_spellings: dict[str, str] = {}

    def normalized_component_id(value: object) -> str:
        spelling = str(value).strip()
        component = spelling.upper()
        if not component or component in {".", "?"}:
            raise DesignInputError("CCD component ID is missing")
        previous = component_spellings.get(component)
        if previous is not None and previous != spelling:
            raise DesignInputError(
                f"CCD component ID has case-colliding spellings {previous!r} and {spelling!r}"
            )
        component_spellings[component] = spelling
        return component

    for block_name in cif.keys():
        block = cif[block_name]
        if "chem_comp_atom" in block:
            category = block["chem_comp_atom"]
            n = category.row_count
            component_ids = _column(category, "comp_id", n, str(block_name).upper())
            names = _column(category, "atom_id", n)
            elements = _column(category, "type_symbol", n, "")
            for raw_component_id, atom_name, raw_element in zip(
                component_ids, names, elements, strict=True
            ):
                if str(atom_name) not in {"", ".", "?"}:
                    component = normalized_component_id(raw_component_id)
                    name = str(atom_name)
                    element = _normal_element(str(raw_element), name)
                    previous = atom_elements[component].get(name)
                    if previous is not None and previous != element:
                        raise DesignInputError(
                            f"CCD component {component!r} atom {name!r} has conflicting elements"
                        )
                    atom_names[component].add(name)
                    atom_elements[component][name] = element
        if "chem_comp_bond" not in block:
            continue
        category = block["chem_comp_bond"]
        n = category.row_count
        comp = _column(category, "comp_id", n, str(block_name).upper())
        atom1 = _column(category, "atom_id_1", n)
        atom2 = _column(category, "atom_id_2", n)
        order = _column(category, "value_order", n, "")
        aromatic = _column(category, "pdbx_aromatic_flag", n, "")
        for comp_id, name1, name2, order_name, aromatic_flag in zip(comp, atom1, atom2, order, aromatic):
            bonds[normalized_component_id(comp_id)].append(
                (str(name1), str(name2), _cif_component_bond_type(order_name, aromatic_flag))
            )
    return {
        component: _CcdTopology(
            atom_names=frozenset(
                atom_names[component]
                or {name for edge in bonds[component] for name in edge[:2]}
            ),
            bonds=tuple(bonds[component]),
            atom_elements=tuple(sorted(atom_elements[component].items())),
        )
        for component in atom_names.keys() | bonds.keys()
    }


@lru_cache(maxsize=4)
def _compact_ccd_topologies(path_text: str) -> dict[str, _CcdTopology]:
    path = Path(path_text)
    try:
        if path.stat().st_size > _MAX_COMPACT_CCD_COMPRESSED_BYTES:
            raise DesignInputError(
                f"compact CCD topology exceeds {_MAX_COMPACT_CCD_COMPRESSED_BYTES} compressed bytes"
            )
        with gzip.open(path, "rb") as stream:
            encoded = stream.read(_MAX_COMPACT_CCD_UNCOMPRESSED_BYTES + 1)
        if len(encoded) > _MAX_COMPACT_CCD_UNCOMPRESSED_BYTES:
            raise DesignInputError(
                f"compact CCD topology exceeds {_MAX_COMPACT_CCD_UNCOMPRESSED_BYTES} uncompressed bytes"
            )
        payload = json.loads(encoded.decode("utf-8"))
    except DesignInputError:
        raise
    except Exception as exc:
        raise DesignInputError(f"cannot parse compact CCD topology {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "components"}:
        raise DesignInputError("compact CCD topology must contain exactly schema_version and components")
    schema_version = payload["schema_version"]
    if schema_version not in {1, 2} or not isinstance(payload["components"], Mapping):
        raise DesignInputError("unsupported compact CCD topology schema")
    result: dict[str, _CcdTopology] = {}
    for raw_component, raw_entry in payload["components"].items():
        if not isinstance(raw_component, str):
            raise DesignInputError("compact CCD component IDs must be strings")
        component = raw_component.strip().upper()
        if component in result:
            raise DesignInputError(
                f"duplicate normalized compact CCD component ID {component!r}"
            )
        expected_keys = {"atoms", "bonds"} | (
            {"elements"} if schema_version == 2 else set()
        )
        if not component or not isinstance(raw_entry, Mapping) or set(raw_entry) != expected_keys:
            raise DesignInputError("invalid compact CCD component entry")
        raw_atoms = raw_entry["atoms"]
        raw_bonds = raw_entry["bonds"]
        raw_elements = raw_entry.get("elements")
        if (
            not isinstance(raw_atoms, list)
            or not all(isinstance(name, str) and name not in {"", ".", "?"} for name in raw_atoms)
            or len(raw_atoms) != len(set(raw_atoms))
            or not isinstance(raw_bonds, list)
        ):
            raise DesignInputError(f"invalid compact CCD component {component!r}")
        if schema_version == 2 and (
            not isinstance(raw_elements, list)
            or len(raw_elements) != len(raw_atoms)
            or not all(
                isinstance(element, str)
                and element == _normal_element(element, atom_name)
                for atom_name, element in zip(raw_atoms, raw_elements, strict=True)
            )
        ):
            raise DesignInputError(
                f"invalid compact CCD element signature for {component!r}"
            )
        bonds: list[tuple[str, str, int]] = []
        seen: set[tuple[str, str]] = set()
        for row in raw_bonds:
            if not isinstance(row, list) or len(row) != 3:
                raise DesignInputError(f"invalid compact CCD bond for {component!r}")
            left, right, order = row
            if (
                not isinstance(left, str)
                or not isinstance(right, str)
                or left not in raw_atoms
                or right not in raw_atoms
                or left == right
                or isinstance(order, bool)
                or not isinstance(order, int)
                or order not in set(_ORDER_TO_BIOTITE.values())
            ):
                raise DesignInputError(f"invalid compact CCD bond for {component!r}")
            pair = tuple(sorted((left, right)))
            if pair in seen:
                raise DesignInputError(f"duplicate compact CCD bond for {component!r}")
            seen.add(pair)
            bonds.append((left, right, order))
        result[component] = _CcdTopology(
            frozenset(raw_atoms),
            tuple(bonds),
            ()
            if raw_elements is None
            else tuple(zip(raw_atoms, raw_elements, strict=True)),
        )
    return result


@lru_cache(maxsize=4)
def _ccd_topologies(path: Path) -> ABCMapping[str, _CcdTopology]:
    if path.name.endswith(".json.gz"):
        try:
            component_ids = ccd_topology_ids(path)
            records = ccd_topology_components(path)
        except Exception as exc:
            if path.name == "ccd.json.gz":
                raise DesignInputError(f"cannot parse merged CCD topology {path}: {exc}") from exc
        else:
            return _LazyCcdTopologies(records, component_ids)
        return _compact_ccd_topologies(str(path))
    return _ccd_cif_topologies(path)


def load_ccd_standard_parents(path: Path) -> dict[str, str | None]:
    """Read explicit non-standard parents without importing training parsers."""

    if path.name.endswith(".json.gz"):
        return {}

    try:
        cif = pdbx.CIFFile.read(path)
    except Exception as exc:
        raise DesignInputError(f"cannot parse CCD file {path}: {exc}") from exc
    result: dict[str, str | None] = {}
    for block_name in cif.keys():
        block = cif[block_name]
        if "chem_comp" not in block:
            continue
        category = block["chem_comp"]
        n = category.row_count
        ids = _column(category, "id", n, str(block_name).upper())
        has_parent_column = "mon_nstd_parent_comp_id" in category
        parents = _column(category, "mon_nstd_parent_comp_id", n, "")
        for comp_id, raw_parent in zip(ids, parents):
            name = str(comp_id).strip().upper()
            parent = str(raw_parent).strip().upper()
            if not has_parent_column:
                # The source omitted the parent field altogether: allow a
                # lower-precedence source or COMMON_STANDARD_PARENT to fill
                # it, matching CCDComponent.parent_ids is None.
                continue
            if parent in {"", ".", "?"}:
                # The field is present and explicitly empty.  Preserve that
                # fact as a blocking marker instead of falling back by name.
                result[name] = None
                continue
            if any(separator in parent for separator in (",", ";", " ")):
                # An explicit multi-parent value is concrete but unusable for
                # proxy selection.  It must block a misleading canonical-name
                # fallback just like parent_ids with len != 1.
                result[name] = None
                continue
            previous = result.get(name)
            if name in result and previous != parent:
                raise DesignInputError(f"CCD component {name!r} has conflicting parents")
            result[name] = parent
    return result


def _apply_ligand_ccd_fallback(
    atoms: struc.AtomArray,
    file_bonds: list[tuple[int, int, int]],
    ccd_path: Path | None,
    *,
    strict_type_bond_chains: frozenset[str] = frozenset(),
) -> list[tuple[int, int, int]]:
    result = list(file_bonds)
    ccd: dict[str, _CcdTopology] | None = None

    def load_ccd() -> dict[str, _CcdTopology]:
        nonlocal ccd
        if ccd is None:
            if ccd_path is None:
                ccd = {}
            else:
                ccd = _ccd_topologies(ccd_path)
        return ccd

    def lookup_topology(component_id: str) -> _CcdTopology | None:
        table = load_ccd()
        # The merged lazy view is a dict-backed cache.  Use the built-in lookup
        # for warm hits and fall back to its lazy materializer on a miss.
        if isinstance(table, _LazyCcdTopologies):
            try:
                return dict.__getitem__(table, component_id)
            except KeyError:
                try:
                    return table[component_id]
                except KeyError:
                    return None
        try:
            return table[component_id]
        except KeyError:
            return None

    def component_edges(indices: np.ndarray) -> list[tuple[int, int, int]]:
        row_set = set(int(value) for value in indices)
        return [
            (int(atom_i), int(atom_j), int(order))
            for atom_i, atom_j, order in result
            if int(atom_i) in row_set and int(atom_j) in row_set
        ]

    def connected(indices: np.ndarray, edges: list[tuple[int, int, int]]) -> bool:
        if len(indices) <= 1:
            return True
        adjacency = {int(row): set() for row in indices}
        for atom_i, atom_j, _ in edges:
            adjacency[atom_i].add(atom_j)
            adjacency[atom_j].add(atom_i)
        reached = {int(indices[0])}
        pending = [int(indices[0])]
        while pending:
            row = pending.pop()
            for neighbor in adjacency[row]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        return len(reached) == len(indices)

    def normalized_component_id(indices: np.ndarray) -> str:
        names = {str(atoms.res_name[row]).strip().upper() for row in indices}
        if len(names) != 1 or next(iter(names)) in {"", ".", "?"}:
            raise DesignInputError("ligand component identity is missing or ambiguous")
        return next(iter(names))

    def source_name_rows(indices: np.ndarray) -> dict[str, int]:
        by_name: dict[str, int] = {}
        for row in indices:
            name = str(atoms.atom_name[int(row)])
            if name in by_name:
                raise DesignInputError(
                    f"ligand component has duplicate atom name {name!r}"
                )
            by_name[name] = int(row)
        return by_name

    def ccd_heavy_signature(
        component_id: str,
        topology: _CcdTopology,
    ) -> dict[str, str]:
        elements = dict(topology.atom_elements)
        if not elements:
            raise DesignInputError(
                f"CCD component {component_id!r} lacks an exact atom-element signature"
            )
        return {
            name: str(element).strip().upper()
            for name, element in elements.items()
            if str(element).strip().upper() not in {"H", "D", "T"}
        }

    def strict_ccd_failure(component_id: str, detail: str, *, partial: bool) -> None:
        message = f"type_bond CCD normalization for {component_id!r} failed: {detail}"
        if partial:
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        raise DesignInputError(message)

    strict_signatures: dict[
        str,
        tuple[
            tuple[tuple[str, str], ...],
            tuple[tuple[str, str, int], ...],
        ],
    ] = {}

    for component_uid in np.unique(atoms.component_uid[atoms.mol_type == "ligand"]):
        indices = np.where(atoms.component_uid == component_uid)[0]
        chains = {str(atoms.chain_id[row]) for row in indices}
        if len(chains) != 1:
            raise DesignInputError("one ligand component spans multiple chain IDs")
        strict = next(iter(chains)) in strict_type_bond_chains
        edges = component_edges(indices)

        if strict:
            component_id = normalized_component_id(indices)
            by_name = source_name_rows(indices)
            concrete = all(order in _EXACT_COVALENT_TYPES for _, _, order in edges)
            complete_source = concrete and connected(indices, edges)
            if not complete_source:
                partial = bool(edges)
                try:
                    topology = lookup_topology(component_id)
                except DesignInputError as exc:
                    strict_ccd_failure(component_id, str(exc), partial=partial)
                if topology is None:
                    strict_ccd_failure(
                        component_id,
                        "component ID is absent from the configured CCD",
                        partial=partial,
                    )
                try:
                    expected_elements = ccd_heavy_signature(component_id, topology)
                except DesignInputError as exc:
                    strict_ccd_failure(component_id, str(exc), partial=partial)
                observed_elements = {
                    name: str(atoms.element[row]).strip().upper()
                    for name, row in by_name.items()
                }
                if observed_elements != expected_elements:
                    strict_ccd_failure(
                        component_id,
                        "heavy-atom name/element signature mismatch",
                        partial=partial,
                    )

                ccd_edges: dict[tuple[str, str], int] = {}
                for name1, name2, raw_order in topology.bonds:
                    if name1 not in expected_elements or name2 not in expected_elements:
                        continue
                    pair = tuple(sorted((str(name1), str(name2))))
                    order = int(raw_order)
                    if order not in _EXACT_COVALENT_TYPES:
                        strict_ccd_failure(
                            component_id,
                            f"CCD edge {pair} has unsupported order {order}",
                            partial=partial,
                        )
                    previous = ccd_edges.get(pair)
                    if previous is not None and previous != order:
                        strict_ccd_failure(
                            component_id,
                            f"CCD edge {pair} has conflicting orders",
                            partial=partial,
                        )
                    ccd_edges[pair] = order
                ccd_row_edges = [
                    (by_name[name1], by_name[name2], order)
                    for (name1, name2), order in sorted(ccd_edges.items())
                ]
                if not connected(indices, ccd_row_edges):
                    strict_ccd_failure(
                        component_id,
                        "CCD heavy-atom graph is disconnected or empty",
                        partial=partial,
                    )

                for atom_i, atom_j, raw_order in edges:
                    pair = tuple(
                        sorted(
                            (
                                str(atoms.atom_name[atom_i]),
                                str(atoms.atom_name[atom_j]),
                            )
                        )
                    )
                    ccd_order = ccd_edges.get(pair)
                    if ccd_order is None:
                        strict_ccd_failure(
                            component_id,
                            f"source edge {pair} is absent from CCD",
                            partial=partial,
                        )
                    if raw_order in _EXACT_COVALENT_TYPES and raw_order != ccd_order:
                        strict_ccd_failure(
                            component_id,
                            f"source/CCD order conflict for edge {pair}",
                            partial=partial,
                        )

                row_set = set(int(value) for value in indices)
                result = [
                    edge
                    for edge in result
                    if not (int(edge[0]) in row_set and int(edge[1]) in row_set)
                ]
                result.extend(ccd_row_edges)
                edges = ccd_row_edges

            if any(order not in _EXACT_COVALENT_TYPES for _, _, order in edges):
                raise DesignInputError(
                    f"type_bond component {component_id!r} retains an unknown bond order"
                )
            signature = (
                tuple(
                    sorted(
                        (str(atoms.atom_name[row]), str(atoms.element[row]).upper())
                        for row in indices
                    )
                ),
                tuple(
                    sorted(
                        (
                            *sorted(
                                (
                                    str(atoms.atom_name[atom_i]),
                                    str(atoms.atom_name[atom_j]),
                                )
                            ),
                            int(order),
                        )
                        for atom_i, atom_j, order in edges
                    )
                ),
            )
            previous = strict_signatures.get(component_id)
            if previous is not None and previous != signature:
                raise DesignInputError(
                    f"type_bond component ID {component_id!r} is reused with a different complete graph"
                )
            strict_signatures[component_id] = signature
            continue

        # Legacy reader behavior for context and non-type_bond design modes:
        # patch only file-declared ANY orders, or recover a graph when none is
        # present.  Do not apply the new strict completion/conflict semantics.
        if edges:
            if any(order == int(struc.BondType.ANY) for _, _, order in edges):
                topology = lookup_topology(normalized_component_id(indices))
                by_edge = (
                    {}
                    if topology is None
                    else {
                        tuple(sorted((name1, name2))): int(order)
                        for name1, name2, order in topology.bonds
                    }
                )
                for row, (atom_i, atom_j, order) in enumerate(result):
                    if (
                        atom_i in indices
                        and atom_j in indices
                        and order == int(struc.BondType.ANY)
                    ):
                        pair = tuple(
                            sorted(
                                (
                                    str(atoms.atom_name[atom_i]),
                                    str(atoms.atom_name[atom_j]),
                                )
                            )
                        )
                        replacement = by_edge.get(pair)
                        if replacement is not None:
                            result[row] = (atom_i, atom_j, replacement)
            continue
        if len(indices) <= 1:
            continue
        component_id = normalized_component_id(indices)
        topology = lookup_topology(component_id)
        if topology is None or not topology.bonds:
            raise DesignInputError(
                f"multi-atom ligand {component_id!r} has no file graph and no usable CCD topology"
            )
        present_names = set(source_name_rows(indices))
        unknown_names = sorted(present_names.difference(topology.atom_names))
        if unknown_names:
            raise DesignInputError(
                f"CCD topology for {component_id!r} does not recognize input atom names "
                f"{unknown_names}"
            )
        recovered = 0
        for name1, name2, order in topology.bonds:
            match1 = indices[atoms.atom_name[indices] == name1]
            match2 = indices[atoms.atom_name[indices] == name2]
            if len(match1) == 0 or len(match2) == 0:
                continue
            if len(match1) != 1 or len(match2) != 1:
                raise DesignInputError(
                    f"CCD topology for {component_id!r} has ambiguous input atom names"
                )
            result.append((int(match1[0]), int(match2[0]), int(order)))
            recovered += 1
        if recovered == 0:
            raise DesignInputError(
                f"CCD topology for {component_id!r} has no edge between resolved input atoms"
            )
    return result


class StructureReader:
    def __init__(self, ccd_path: str | Path | None = None) -> None:
        self.ccd_path = (
            _DEFAULT_COMPACT_CCD_PATH
            if ccd_path is None
            else Path(ccd_path).expanduser().resolve()
        )
        if not self.ccd_path.is_file():
            origin = "packaged" if ccd_path is None else "configured"
            raise DesignInputError(f"{origin} CCD file does not exist: {self.ccd_path}")

    def read(
        self,
        path: str | Path,
        *,
        strict_type_bond_chains: frozenset[str] = frozenset(),
        polymer_parent_sources: Mapping[str, str | None] | None = None,
        declared_polymer_chain_modalities: Mapping[str, str] | None = None,
    ) -> struc.AtomArray:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise DesignInputError(f"input structure does not exist: {source}")
        suffix = source.suffix.lower()
        if suffix not in {".pdb", ".cif"}:
            raise DesignInputError("standalone reader accepts only .pdb and .cif")
        parent_sources = (
            dict(polymer_parent_sources)
            if polymer_parent_sources is not None
            else (
                {}
                if self.ccd_path.name.endswith(".json.gz")
                else load_ccd_standard_parents(self.ccd_path)
            )
        )
        if polymer_parent_sources is None and suffix == ".cif":
            parent_sources.update(load_ccd_standard_parents(source))
        try:
            if suffix == ".pdb":
                atoms, file_bonds, _ = _read_pdb(
                    source,
                    parent_sources=parent_sources,
                    declared_polymer_chain_modalities=declared_polymer_chain_modalities,
                )
            else:
                atoms, block, _, altloc_ledger = _read_cif_rows(
                    source,
                    parent_sources=parent_sources,
                    declared_polymer_chain_modalities=declared_polymer_chain_modalities,
                )
                file_bonds = _cif_component_bonds(
                    block, atoms
                ) + _cif_struct_conn_bonds(block, atoms, altloc_ledger)
                attach_source_sequences(atoms, block=block)
        except DesignInputError:
            raise
        except Exception as exc:
            format_name = "PDB" if suffix == ".pdb" else "CIF"
            raise DesignInputError(f"cannot parse {format_name} {source}: {exc}") from exc
        file_bonds = _apply_ligand_ccd_fallback(
            atoms,
            file_bonds,
            self.ccd_path,
            strict_type_bond_chains=frozenset(
                str(value) for value in strict_type_bond_chains
            ),
        )
        complete = False
        if suffix == ".cif" and "struct_conn_type" in block:
            types = block["struct_conn_type"]
            if "criteria" in types:
                complete = COMPLETE_BACKBONE_CRITERIA in types["criteria"].as_array(str).tolist()
        mark_source_breaks(atoms, file_bonds, complete=complete)
        atoms.bonds = _merge_template_and_declared_bonds(
            len(atoms),
            _polymer_template_bonds(atoms), file_bonds,
        )
        atoms.source_graph_complete[:] = complete or suffix == ".pdb"
        validate_atom_array(atoms, phase="identity")
        return atoms


def read_structure(
    path: Path,
    *,
    ccd_path: Path | None = None,
    strict_type_bond_chains: frozenset[str] = frozenset(),
    polymer_parent_sources: Mapping[str, str | None] | None = None,
    declared_polymer_chain_modalities: Mapping[str, str] | None = None,
) -> struc.AtomArray:
    """Read one standalone inference structure under the controlled ABI."""

    return StructureReader(ccd_path=ccd_path).read(
        path,
        strict_type_bond_chains=strict_type_bond_chains,
        polymer_parent_sources=polymer_parent_sources,
        declared_polymer_chain_modalities=declared_polymer_chain_modalities,
    )
