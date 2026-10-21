"""Lean, truthful structure serialization for standalone inference."""

from __future__ import annotations

import copy
from functools import lru_cache
import os
import re
import tempfile
import warnings
from pathlib import Path
from src.common.chemistry.ccd import polymer_type_pairs
from src.common.chemistry.charges import formal_charge_values
from typing import Iterable, Mapping

import numpy as np
from biotite import structure as struc
from biotite.structure.io import pdb, pdbx

from .atoms import DNA_RES_ATOMS, PROTEIN_RES_ATOMS, RNA_RES_ATOMS
from .ligand_identity import (
    ligand_identity_matches_component,
)
from ..core.records import ValidatedLigandIdentity
from .source_mapping import COMPLETE_BACKBONE_CRITERIA
from .visualization import condition_b_factors


_SAFE_BLOCK = re.compile(r"[^A-Za-z0-9_-]")
_PDB_CHAIN_ID = re.compile(r"[A-Za-z0-9]")
_PDB_RES_NAME = re.compile(r"[A-Z0-9]{1,3}")
_PDB_ATOM_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9']{0,3}")
_PDB_INS_CODE = re.compile(r"[A-Za-z0-9]")
_PDB_ELEMENT = re.compile(r"[A-Z]{1,2}")
_POLYMER_TYPES = frozenset({"protein", "rna", "dna"})
_STANDARD_COMPONENT_IDS = frozenset(
    set(PROTEIN_RES_ATOMS) | set(RNA_RES_ATOMS) | set(DNA_RES_ATOMS)
)
_STANDARD_COMPONENT_MODALITIES = {
    **dict.fromkeys(PROTEIN_RES_ATOMS, "protein"),
    **dict.fromkeys(RNA_RES_ATOMS, "rna"),
    **dict.fromkeys(DNA_RES_ATOMS, "dna"),
}
_CHEM_COMP_TYPES = {
    "protein": "peptide linking",
    "rna": "rna linking",
    "dna": "dna linking",
    "ligand": "non-polymer",
}
_COVALENT_TYPES = {
    int(struc.BondType.SINGLE): "sing",
    int(struc.BondType.DOUBLE): "doub",
    int(struc.BondType.TRIPLE): "trip",
    int(struc.BondType.AROMATIC): "arom",
    int(struc.BondType.AROMATIC_SINGLE): "sing",
    int(struc.BondType.AROMATIC_DOUBLE): "doub",
    int(struc.BondType.AROMATIC_TRIPLE): "trip",
}


class StructureWriteError(ValueError):
    """The authoritative structure cannot be serialized without loss."""


def _annotation(atom_array: struc.AtomArray, name: str, default: object) -> np.ndarray:
    if name in atom_array.get_annotation_categories():
        return np.asarray(atom_array.get_annotation(name))
    return np.full(len(atom_array), default)


def _partition_signature(values: np.ndarray) -> tuple[int, ...]:
    remap: dict[int, int] = {}
    result: list[int] = []
    for raw_value in np.asarray(values, dtype=np.int64):
        value = int(raw_value)
        if value not in remap:
            remap[value] = len(remap)
        result.append(remap[value])
    return tuple(result)


def _safe_block_name(value: str) -> str:
    result = _SAFE_BLOCK.sub("_", str(value)).strip("_")
    return result or "design"


def _component_ids(
    atom_array: struc.AtomArray,
    validated_ligand_identities: Mapping[int, ValidatedLigandIdentity] | None = None,
) -> tuple[np.ndarray, dict[int, str]]:
    component_uid = _annotation(atom_array, "component_uid", -1).astype(np.int64)
    mol_type = _annotation(atom_array, "mol_type", "").astype(str)
    is_design = _annotation(atom_array, "is_design", False).astype(bool)
    if np.any(component_uid < 0):
        raise StructureWriteError("component_uid must be finalized before writing")

    component_name: dict[int, str] = {}
    identities = {} if validated_ligand_identities is None else dict(
        validated_ligand_identities
    )
    ligand_records: list[tuple[int, ValidatedLigandIdentity | None]] = []
    bond_rows = _bond_rows(atom_array)
    for uid in dict.fromkeys(component_uid.tolist()):
        rows = np.flatnonzero(component_uid == uid)
        original = str(atom_array.res_name[rows[0]]).strip()
        is_ligand = bool(np.all(mol_type[rows] == "ligand"))
        if not is_ligand:
            component_name[int(uid)] = original or f"C{int(uid):03d}"
            continue
        identity = identities.get(int(uid))
        if identity is not None and not ligand_identity_matches_component(
            atom_array, rows, identity
        ):
            identity = None
        ligand_records.append((int(uid), identity))

    nonligand_marker = object()
    missing_marker = object()
    used: dict[str, object] = {
        name: nonligand_marker for name in component_name.values()
    }
    trusted_uids: set[int] = set()
    for uid, identity in ligand_records:
        if identity is None:
            continue
        base = identity.component_id
        signature = (
            identity.atom_signature,
            identity.edge_signature,
        )
        previous = used.get(base, missing_marker)
        if previous is nonligand_marker:
            # One mmCIF component ID cannot simultaneously declare polymer and
            # non-polymer chemistry.  The ligand remains valid, but preserving
            # this ID would make the file ambiguous, so use its local LIGxx ID.
            continue
        if previous is not missing_marker and previous != signature:
            raise StructureWriteError(
                f"ligand component ID {base!r} is reused with a different complete graph"
            )
        used[base] = signature
        component_name[uid] = base
        trusted_uids.add(uid)

    ligand_ordinal = 0
    for uid, _identity in ligand_records:
        if uid in trusted_uids:
            continue
        ligand_ordinal += 1
        name = f"LIG{ligand_ordinal:02d}"
        while name in used:
            ligand_ordinal += 1
            name = f"LIG{ligand_ordinal:02d}"
        used[name] = None
        component_name[uid] = name
    return component_uid, component_name


def _entity_ids(atom_array: struc.AtomArray) -> np.ndarray:
    entity_uid = _annotation(atom_array, "entity_uid", -1).astype(np.int64)
    if np.any(entity_uid < 0):
        raise StructureWriteError("entity_uid must be finalized before writing")
    return np.asarray([str(int(value) + 1) for value in entity_uid], dtype="U32")


def _label_seq(atom_array: struc.AtomArray, index: int, mol_type: str) -> str:
    if mol_type not in _POLYMER_TYPES:
        return "."
    value = int(atom_array.res_id[index])
    if value <= 0:
        raise StructureWriteError("polymer label_seq_id must be positive")
    return str(value)


def _author_seq_ids(
    atom_array: struc.AtomArray,
    *,
    label_asym: np.ndarray,
    mol_type: np.ndarray,
    component_uid: np.ndarray,
) -> np.ndarray:
    """Return unique author residue locators for every physical instance.

    Polymer rows deliberately use the emitted label_seq_id value.  A
    nonpolymer label_seq_id is "." by contract, so its author sequence
    value carries the physical component identity.  Preserve the normalized
    res_id when it is unique within an asym and allocate deterministic unused
    integer IDs only for colliding component instances.  Numeric locators are
    required because downstream CIF readers interpret auth_seq_id as an
    integer residue identifier.
    """

    result = np.empty(len(atom_array), dtype="<U64")
    polymer = np.isin(mol_type, tuple(_POLYMER_TYPES))
    result[polymer] = np.asarray(atom_array.res_id[polymer], dtype=np.int64).astype(str)
    source_format = _annotation(atom_array, "source_format", "").astype(str)
    source_auth = _annotation(atom_array, "auth_seq_id", "").astype(str)
    retained = (source_format != "") & ~np.isin(source_auth, ("", ".", "?"))
    result[retained] = source_auth[retained]

    nonpolymer = ~polymer
    for asym in dict.fromkeys(label_asym.tolist()):
        asym_rows = np.flatnonzero(nonpolymer & (label_asym == asym))
        if not len(asym_rows):
            continue
        instances: dict[int, np.ndarray] = {}
        for uid in dict.fromkeys(component_uid[asym_rows].tolist()):
            rows = asym_rows[component_uid[asym_rows] == uid]
            if len(set(int(value) for value in atom_array.res_id[rows])) != 1:
                raise StructureWriteError(
                    "nonpolymer component instance has multiple normalized residue IDs"
                )
            known = rows[retained[rows]]
            if len(known):
                # A retained scaffold and its newly completed atoms describe
                # the same physical component instance and source locator.
                result[rows] = source_auth[int(known[0])]
                continue
            instances[int(uid)] = rows
        by_base: dict[int, list[int]] = {}
        for uid, rows in instances.items():
            by_base.setdefault(int(atom_array.res_id[rows[0]]), []).append(uid)
        polymer_auth_ids = {
            int(result[row])
            for row in np.flatnonzero((label_asym == asym) & (polymer | retained))
            if re.fullmatch(r"[+-]?\d+", str(result[row]))
        }
        # Reserve every ID that can be retained for an unambiguous component,
        # including polymer author IDs in the same asymmetry.  Allocate the
        # smallest unused integer for each collision in UID order.  This
        # preserves all available source IDs and is stable across writes.
        reserved = polymer_auth_ids | {
            base
            for base, uids in by_base.items()
            if len(uids) == 1 and base not in polymer_auth_ids
        }
        used = set(reserved)
        for base, uids in sorted(by_base.items()):
            for uid in sorted(uids):
                if len(uids) == 1 and base not in polymer_auth_ids:
                    locator = base
                else:
                    locator = base
                    while locator in used:
                        locator += 1
                    used.add(locator)
                result[instances[uid]] = str(locator)
    if np.any(result == ""):
        raise StructureWriteError(
            "every atom needs a non-empty author sequence locator"
        )
    return result


def _bond_rows(atom_array: struc.AtomArray) -> np.ndarray:
    if atom_array.bonds is None:
        return np.empty((0, 3), dtype=np.int64)
    bonds = np.asarray(atom_array.bonds.as_array(), dtype=np.int64)
    return bonds.reshape((-1, 3))


def _canonical_polymer_bonds(
    atom_array: struc.AtomArray,
) -> dict[tuple[int, int], int]:
    """Return template-expected polymer edges in global authoring rows."""

    mol_type = _annotation(atom_array, "mol_type", "").astype(str)
    result: dict[tuple[int, int], int] = {}
    has_asym_uid = "asym_uid" in atom_array.get_annotation_categories()
    asym_uid = _annotation(atom_array, "asym_uid", -1).astype(np.int64)
    if has_asym_uid and np.any(asym_uid < 0):
        raise StructureWriteError("asym_uid must be finalized before writing")
    if not has_asym_uid:
        chain_to_uid: dict[str, int] = {}
        asym_uid = np.asarray(
            [
                chain_to_uid.setdefault(str(chain), len(chain_to_uid))
                for chain in atom_array.chain_id
            ],
            dtype=np.int64,
        )
    groups = list(
        dict.fromkeys(
            (int(asym_uid[row]), str(mol_type[row]))
            for row in range(len(atom_array))
            if mol_type[row] in _POLYMER_TYPES
        )
    )
    for group_asym_uid, modality in groups:
        rows = np.where(
            (asym_uid == group_asym_uid)
            & (mol_type == modality)
        )[0]
        try:
            local = struc.connect_via_residue_names(
                atom_array[rows], inter_residue=True
            )
        except Exception as exc:
            raise StructureWriteError(
                "cannot normalize canonical polymer topology for "
                f"internal asym {group_asym_uid}"
            ) from exc
        for local_i, local_j, raw_type in local.as_array():
            key = tuple(
                sorted((int(rows[int(local_i)]), int(rows[int(local_j)])))
            )
            bond_type = int(raw_type)
            previous = result.get(key)
            if previous is not None and previous != bond_type:
                raise StructureWriteError(
                    f"canonical polymer topology conflicts for rows {key}"
                )
            result[key] = bond_type
    return result


def _fast_canonical_polymer_bonds(
    atom_array: struc.AtomArray,
) -> dict[tuple[int, int], int] | None:
    """Classify emitted polymer rows without rebuilding the full CCD graph.

    The writer only needs to classify bonds already present in the authoritative
    ``AtomArray``. Intra-residue template edges come directly from Biotite's
    named component table, while ordinary adjacent peptide/nucleic links use
    their named endpoints. Non-standard boundaries are delegated to the same
    Biotite connector on the two residues only. Returning ``None`` keeps the
    complete connector as a fail-closed fallback for unusual inputs.

    This function uses residue/component names and existing bond rows only; it
    never infers topology from coordinates or distances.
    """

    categories = atom_array.get_annotation_categories()
    if "residue_uid" not in categories:
        return None
    if atom_array.bonds is None:
        return {}

    mol_type = _annotation(atom_array, "mol_type", "").astype(str)
    if not np.any(np.isin(mol_type, tuple(_POLYMER_TYPES))):
        return {}
    asym_uid = _annotation(atom_array, "asym_uid", -1).astype(np.int64)
    if np.any(asym_uid < 0):
        return None
    residue_uid = _annotation(atom_array, "residue_uid", -1).astype(np.int64)
    if np.any(residue_uid < 0):
        return None
    residue_names = np.asarray(atom_array.res_name).astype(str)
    atom_names = np.asarray(atom_array.atom_name).astype(str)
    raw_rows = _bond_rows(atom_array)
    raw_keys = {
        tuple(sorted((int(row_i), int(row_j))))
        for row_i, row_j, _raw_type in raw_rows
    }

    rows_by_residue: dict[int, list[int]] = {}
    for row, uid in enumerate(residue_uid):
        rows_by_residue.setdefault(int(uid), []).append(row)
    residue_info: dict[int, tuple[int, str, str, dict[str, int]]] = {}
    group_residues: dict[tuple[int, str], list[int]] = {}
    template_cache: dict[tuple[str, str], Mapping[tuple[str, str], int]] = {}
    result: dict[tuple[int, int], int] = {}

    for uid, rows in rows_by_residue.items():
        first = rows[0]
        modality = str(mol_type[first])
        if modality not in _POLYMER_TYPES:
            continue
        if rows[-1] - rows[0] + 1 != len(rows):
            return None
        component = str(residue_names[first])
        by_name: dict[str, int] = {}
        for row in rows:
            if str(mol_type[row]) != modality or str(residue_names[row]) != component:
                return None
            name = str(atom_names[row])
            if name in by_name:
                return None
            by_name[name] = row
        cache_key = (modality, component)
        try:
            template = template_cache.get(cache_key)
            if template is None:
                template = struc.info.bonds_in_residue(component)
                template_cache[cache_key] = template
        except Exception:
            return None
        residue_info[uid] = (
            int(asym_uid[first]),
            modality,
            component,
            by_name,
        )
        group_residues.setdefault((int(asym_uid[first]), modality), []).append(uid)
        for (atom_i, atom_j), raw_type in template.items():
            row_i = by_name.get(str(atom_i))
            row_j = by_name.get(str(atom_j))
            if row_i is None or row_j is None:
                continue
            key = tuple(sorted((row_i, row_j)))
            if key not in raw_keys:
                continue
            bond_type = int(raw_type)
            previous = result.get(key)
            if previous is not None and previous != bond_type:
                return None
            result[key] = bond_type

    for (_group_asym_uid, modality), residue_uids in group_residues.items():
        for left_uid, right_uid in zip(residue_uids, residue_uids[1:]):
            left = residue_info[left_uid]
            right = residue_info[right_uid]
            left_rows = rows_by_residue[left_uid]
            right_rows = rows_by_residue[right_uid]
            left_names = left[3]
            right_names = right[3]
            link_names = (
                ("C", "N")
                if modality == "protein"
                else ("O3'", "P")
            )
            candidate_i = left_names.get(link_names[0])
            candidate_j = right_names.get(link_names[1])
            candidate = (
                tuple(sorted((candidate_i, candidate_j)))
                if candidate_i is not None and candidate_j is not None
                else None
            )
            needs_connector = (
                left[2] not in _STANDARD_COMPONENT_IDS
                or right[2] not in _STANDARD_COMPONENT_IDS
                or int(atom_array.res_id[right_rows[0]])
                - int(atom_array.res_id[left_rows[0]])
                != 1
            )
            if not needs_connector:
                if candidate is not None and candidate in raw_keys:
                    result[candidate] = int(struc.BondType.SINGLE)
                continue
            if not any(
                key[0] in left_rows and key[1] in right_rows
                for key in raw_keys
            ):
                continue
            local_rows = np.asarray(left_rows + right_rows, dtype=np.int64)
            try:
                connected = struc.connect_via_residue_names(
                    atom_array[local_rows], inter_residue=True
                )
            except Exception:
                return None
            for local_i, local_j, raw_type in connected.as_array():
                key = tuple(
                    sorted(
                        (
                            int(local_rows[int(local_i)]),
                            int(local_rows[int(local_j)]),
                        )
                    )
                )
                if key not in raw_keys:
                    continue
                previous = result.get(key)
                bond_type = int(raw_type)
                if previous is not None and previous != bond_type:
                    return None
                result[key] = bond_type
    return result


def _bond_order(
    raw_type: int,
    *,
    inter_component: bool,
    touches_design: bool,
) -> str | None:
    if inter_component and int(raw_type) in (5, 6, 7):
        raise StructureWriteError(
            "struct_conn cannot preserve an explicit aromatic Kekule subtype"
        )
    if int(raw_type) == int(struc.BondType.ANY):
        # Unknown-positive bonds are still authoritative existence facts.  The
        # lean CIF contract serializes them as '?' so a consumer can preserve
        # the edge without inventing a concrete chemistry class.
        return "?"
    if int(raw_type) == int(struc.BondType.COORDINATION):
        return None
    order = _COVALENT_TYPES.get(int(raw_type))
    if order is None:
        raise StructureWriteError(f"unsupported covalent bond type {raw_type}")
    return order


def _internal_bond_descriptor(raw_type: int) -> tuple[str | None, str]:
    return (
        _bond_order(raw_type, inter_component=False, touches_design=False),
        # Plain CCD Kekule order does not independently assert non-aromaticity.
        "Y" if int(raw_type) in (5, 6, 7, 9) else "?",
    )


@lru_cache(maxsize=128)
def _template_order_matches(declared: int, template: int) -> bool:
    from .reader import _template_order_matches as matches
    # A positive unknown source order does not contradict trusted canonical
    # polymer chemistry.  The reader already refines this case from the same
    # template; preserve concrete-order conflicts and unknown special edges.
    if int(declared) == int(struc.BondType.ANY):
        return True
    return matches(declared, template)


def _shared_modified_polymer_component_edges(
    atom_array: struc.AtomArray,
    *,
    mol_type: np.ndarray,
    comp_id: np.ndarray,
) -> dict[str, frozenset[tuple[str, str, str, str]]]:
    """Return only component edges valid for every modified-residue instance.

    A CIF component definition applies to *all* residues with its component
    ID.  Modified residues can retain different leaving atoms, so an edge that
    exists in only one instance must be emitted as residue-local struct_conn
    rather than poisoning the shared chem_comp_bond definition.
    """

    residue_uid = _annotation(atom_array, "residue_uid", -1).astype(np.int64)
    if not np.any(
        np.isin(mol_type, tuple(_POLYMER_TYPES))
        & ~np.isin(comp_id, tuple(_STANDARD_COMPONENT_IDS))
    ):
        return {}
    instance_edges: dict[str, dict[int, set[tuple[str, str, str, str]]]] = {}
    component_orders: dict[tuple[str, str, str], tuple[str, str]] = {}
    for row, modality in enumerate(mol_type):
        component = str(comp_id[row])
        if modality not in _POLYMER_TYPES or component in _STANDARD_COMPONENT_IDS:
            continue
        uid = int(residue_uid[row])
        if uid < 0:
            raise StructureWriteError(
                "modified-polymer residue_uid must be finalized before writing"
            )
        instance_edges.setdefault(component, {}).setdefault(uid, set())

    for raw_i, raw_j, raw_type in _bond_rows(atom_array):
        row_i, row_j = int(raw_i), int(raw_j)
        if (
            int(residue_uid[row_i]) != int(residue_uid[row_j])
            or mol_type[row_i] != mol_type[row_j]
            or mol_type[row_i] not in _POLYMER_TYPES
            or str(comp_id[row_i]) != str(comp_id[row_j])
            or str(comp_id[row_i]) in _STANDARD_COMPONENT_IDS
            or int(raw_type) == int(struc.BondType.COORDINATION)
        ):
            continue
        order, aromatic_flag = _internal_bond_descriptor(int(raw_type))
        if order is None or order == "?":
            # Modified-polymer component definitions require concrete orders;
            # unknown-positive existence remains instance-local struct_conn.
            continue
        atom_i, atom_j = sorted(
            (str(atom_array.atom_name[row_i]), str(atom_array.atom_name[row_j]))
        )
        component = str(comp_id[row_i])
        order_key = (component, atom_i, atom_j)
        previous_order = component_orders.get(order_key)
        descriptor = (order, aromatic_flag)
        if previous_order is not None and previous_order != descriptor:
            raise StructureWriteError(
                f"component {component!r} reuses internal bond "
                f"{atom_i!r}-{atom_j!r} with conflicting orders"
            )
        component_orders[order_key] = descriptor
        instance_edges[component][int(residue_uid[row_i])].add(
            (atom_i, atom_j, order, aromatic_flag)
        )

    return {
        component: frozenset.intersection(
            *(frozenset(edges) for edges in instances.values())
        )
        for component, instances in instance_edges.items()
    }


def _category(data: dict[str, Iterable[object]]) -> pdbx.CIFCategory:
    return pdbx.CIFCategory({key: list(values) for key, values in data.items()})


def _cif_multiline(value: str) -> str:
    return "\n;" + value + "\n;\n"


def _cif_escape(value: object) -> str:
    value = str(value)
    if "\n" in value:
        return _cif_multiline(value)
    if "'" in value and '"' in value:
        return _cif_multiline(value)
    if len(value) == 0:
        return "''"
    if value[0] == "_":
        return "'" + value + "'"
    if "'" in value:
        return '"' + value + '"'
    if '"' in value:
        return "'" + value + "'"
    if " " in value or "\t" in value:
        return "'" + value + "'"
    return value


def _serialize_cif_category(
    name: str, category: pdbx.CIFCategory | Mapping[str, Iterable[object]]
) -> str:
    columns: list[tuple[str, list[str]]] = []
    row_count: int | None = None
    for column_name, column in category.items():
        source = column.as_array(str).tolist() if isinstance(category, pdbx.CIFCategory) else column
        values = [_cif_escape(value) for value in source]
        if row_count is None:
            row_count = len(values)
        elif len(values) != row_count:
            raise ValueError(
                f"All columns must have the same length, but '{column_name}' "
                f"has length {len(values)}, while the first column has "
                f"row_count {row_count}"
            )
        columns.append((column_name, values))
    if not columns:
        raise ValueError("At least one column is required")
    assert row_count is not None
    if row_count == 0:
        raise ValueError("At least one row is required")
    if row_count == 1:
        keys = ["_" + name + "." + column_name for column_name, _ in columns]
        required_length = max(len(key) for key in keys) + 3
        return "\n".join(
            (key.ljust(required_length) + values[0]).strip()
            for key, (_, values) in zip(keys, columns, strict=True)
        ) + "\n"
    key_lines = [
        "_" + name + "." + column_name + " "
        for column_name, _ in columns
    ]
    widths = [max(map(len, values)) + 1 for _, values in columns]
    # Pad each column once, then let zip/join assemble rows. This avoids the
    # per-cell generator/indexing cost, including for the backbone edge table.
    padded = [
        [value.ljust(width) for value in values]
        for (_, values), width in zip(columns, widths, strict=True)
    ]
    value_lines = ["".join(row).strip() for row in zip(*padded, strict=True)]
    return "\n".join(["loop_", *key_lines, *value_lines]) + "\n"


def _write_cif_fast(
    block_name: str,
    categories: Mapping[str, pdbx.CIFCategory | Mapping[str, Iterable[object]]],
    output_path: Path,
) -> None:
    """Write the same lean CIF layout without Biotite's per-cell serializer."""

    text_blocks = ["data_" + block_name + "\n#\n"]
    for category_name, category in categories.items():
        text_blocks.append(_serialize_cif_category(category_name, category))
        text_blocks.append("#\n")
    text_blocks.append("")
    with open(output_path, "w") as handle:
        handle.write("".join(text_blocks))


@lru_cache(maxsize=1)
def _polymer_component_types() -> Mapping[str, list[str]]:
    """Read only the cached compact CCD metadata view for CIF classification."""
    return polymer_type_pairs()


def _chem_comp_category(
    *,
    comp_id: np.ndarray,
    mol_type: np.ndarray,
) -> pdbx.CIFCategory:
    """Provide every atom-site component with the fields Mol* requires.

    ``_chem_comp`` is not merely ligand metadata: once present, Mol* resolves
    every ``label_comp_id`` through it.  Omitting standard polymer components
    therefore creates an undefined component row while loading generated
    proteins.  The entries below state only component class and standardness;
    they do not invent formulas, charges, sequences, or atom tables.
    """

    packaged = None
    rows: dict[str, tuple[str, str]] = {}
    for raw_component, raw_modality in dict.fromkeys(zip(comp_id.tolist(), mol_type.tolist())):
        component = str(raw_component)
        modality = str(raw_modality)
        component_type = _CHEM_COMP_TYPES.get(modality)
        known = None
        if modality in _POLYMER_TYPES:
            standard_modality = _STANDARD_COMPONENT_MODALITIES.get(component)
            if standard_modality is not None:
                if standard_modality != modality:
                    raise StructureWriteError(
                        f"component {component!r} standard modality conflicts with {modality!r}"
                    )
                if modality == "protein" and component != "GLY":
                    component_type = "l-peptide linking"
            else:
                if packaged is None:
                    packaged = _polymer_component_types()
                known = packaged.get(component)
        if known is not None:
            if known[0] != modality:
                raise StructureWriteError(
                    f"component {component!r} CCD modality conflicts with {modality!r}"
                )
            if known[1]:
                component_type = known[1].lower()
        if component_type is None:
            raise StructureWriteError(
                f"unsupported component modality {modality!r} for {component!r}"
            )
        standard_flag = (
            "yes"
            if modality in _POLYMER_TYPES and component in _STANDARD_COMPONENT_IDS
            else "no"
        )
        record = (component_type, standard_flag)
        previous = rows.get(component)
        if previous is not None and previous != record:
            raise StructureWriteError(
                f"component ID {component!r} is reused with incompatible component semantics"
            )
        rows[component] = record
    return _category(
        {
            "id": list(rows),
            "type": [record[0] for record in rows.values()],
            "mon_nstd_flag": [record[1] for record in rows.values()],
        }
    )


def write_cif(
    atom_array: struc.AtomArray,
    output_path: str | Path,
    *,
    entry_id: str,
    validated_ligand_identities: Mapping[int, ValidatedLigandIdentity] | None = None,
) -> None:
    """Write authoritative heavy atoms with condition categories in the B column.

    B_iso_or_equiv encodes authored condition categories, not confidence or
    experimental displacement parameters. An explicit preview b_factor
    annotation overrides the category calculation.
    """

    if len(atom_array) == 0:
        raise StructureWriteError("cannot write an empty structure")
    coord = np.asarray(atom_array.coord, dtype=np.float64)
    if coord.shape != (len(atom_array), 3) or not np.isfinite(coord).all():
        raise StructureWriteError("all output coordinates must be finite")
    component_uid, component_name = _component_ids(
        atom_array,
        validated_ligand_identities,
    )
    entity_id = _entity_ids(atom_array)
    mol_type = _annotation(atom_array, "mol_type", "").astype(str)
    label_asym = _annotation(atom_array, "label_asym_id", "").astype(str)
    label_asym = np.asarray(
        [value or str(atom_array.chain_id[i]) for i, value in enumerate(label_asym)]
    )
    if any(not value for value in label_asym):
        raise StructureWriteError("every atom needs a non-empty label_asym_id")
    asym_uid = _annotation(atom_array, "asym_uid", -1).astype(np.int64)
    if np.any(asym_uid < 0):
        raise StructureWriteError("asym_uid must be finalized before writing")
    for label in dict.fromkeys(label_asym.tolist()):
        if len(np.unique(asym_uid[label_asym == label])) != 1:
            raise StructureWriteError(
                f"label_asym_id {label!r} names multiple internal asym segments"
            )
    for uid in np.unique(asym_uid):
        if len(np.unique(label_asym[asym_uid == uid])) != 1:
            raise StructureWriteError(
                f"internal asym {int(uid)} has multiple label_asym_id values"
            )
    comp_id = np.asarray(
        [
            component_name[int(uid)]
            if mol_type[index] == "ligand"
            else str(atom_array.res_name[index])
            for index, uid in enumerate(component_uid)
        ]
    )
    # The standalone output uses label asymmetries as its stable public chain
    # identity.  Author locators are therefore emitted in that same namespace;
    # polymer author numbering is intentionally not copied from source input.
    auth_asym = label_asym.copy()
    source_format = _annotation(atom_array, "source_format", "").astype(str)
    source_auth_asym = _annotation(atom_array, "auth_asym_id", "").astype(str)
    retained = (source_format != "") & ~np.isin(source_auth_asym, ("", ".", "?"))
    auth_asym[retained] = source_auth_asym[retained]
    auth_seq = _author_seq_ids(
        atom_array,
        label_asym=label_asym,
        mol_type=mol_type,
        component_uid=component_uid,
    )
    try:
        charges = formal_charge_values(atom_array)
    except ValueError as exc:
        raise StructureWriteError(str(exc)) from exc

    atom_site = {
        "group_PDB": ["HETATM" if bool(v) else "ATOM" for v in atom_array.hetero],
        "id": [str(i + 1) for i in range(len(atom_array))],
        "type_symbol": [str(v).upper() for v in atom_array.element],
        "label_atom_id": [str(v) for v in atom_array.atom_name],
        "label_alt_id": ["."] * len(atom_array),
        "label_comp_id": comp_id.tolist(),
        "label_asym_id": label_asym.tolist(),
        "label_entity_id": entity_id.tolist(),
        "label_seq_id": [
            _label_seq(atom_array, i, str(mol_type[i])) for i in range(len(atom_array))
        ],
        "auth_asym_id": auth_asym.tolist(),
        "auth_seq_id": auth_seq.tolist(),
        "pdbx_PDB_ins_code": [str(v) or "." for v in atom_array.ins_code],
        "Cartn_x": [f"{v:.6f}" for v in coord[:, 0]],
        "Cartn_y": [f"{v:.6f}" for v in coord[:, 1]],
        "Cartn_z": [f"{v:.6f}" for v in coord[:, 2]],
        "B_iso_or_equiv": condition_b_factors(atom_array).astype(str).tolist(),
        "pdbx_PDB_model_num": ["1"] * len(atom_array),
    }
    if charges is not None and any(value is not None for value in charges):
        atom_site["pdbx_formal_charge"] = ["?" if value is None else str(value) for value in charges]

    # The fast serializer consumes these columns directly. Wrapping every
    # already formatted value in CIFColumn would only copy and convert it back.
    block: dict[str, pdbx.CIFCategory | Mapping[str, Iterable[object]]] = {"atom_site": atom_site}

    unique_entities: list[str] = list(dict.fromkeys(entity_id.tolist()))
    block["entity"] = dict(
        {
            "id": unique_entities,
            "type": [
                "polymer"
                if np.any(mol_type[entity_id == value] != "ligand")
                else "non-polymer"
                for value in unique_entities
            ],
        }
    )
    unique_asym: list[str] = list(dict.fromkeys(label_asym.tolist()))
    asym_entity: list[str] = []
    for asym in unique_asym:
        values = list(dict.fromkeys(entity_id[label_asym == asym].tolist()))
        if len(values) != 1:
            raise StructureWriteError(
                f"label_asym_id {asym!r} maps to multiple entities"
            )
        asym_entity.append(values[0])
    block["struct_asym"] = dict(
        {"id": unique_asym, "entity_id": asym_entity}
    )
    block["struct_conn_type"] = {"id": ["covale"], "criteria": [COMPLETE_BACKBONE_CRITERIA], "reference": ["."]}
    # Confirmed source sequence is optional. Inspect one source record per
    # entity; ordinary generated dumps take no sequence-validation pass.
    sequence_records = _annotation(atom_array, "source_polymer_sequence", None)
    sequence_positions = _annotation(atom_array, "source_polymer_position", 0).astype(np.int64)
    full_sequences = {}
    for entity in unique_entities:
        entity_rows = np.flatnonzero(entity_id == entity)
        if not len(entity_rows) or np.any(atom_array.is_design[entity_rows]):
            continue
        first = int(entity_rows[0])
        sequence = sequence_records[first]
        if sequence is not None and len(sequence):
            full_sequences[entity] = (tuple(sequence), str(mol_type[first]))
    if full_sequences:
        poly_sequence = {"entity_id": [], "num": [], "mon_id": [], "hetero": []}
        for entity, (sequence, _modality) in full_sequences.items():
            for position, component in enumerate(sequence, start=1):
                poly_sequence["entity_id"].append(entity)
                poly_sequence["num"].append(str(position))
                poly_sequence["mon_id"].append(str(component))
                poly_sequence["hetero"].append("n")
        block["entity_poly_seq"] = poly_sequence
        scheme = {name: [] for name in ("asym_id", "entity_id", "seq_id", "mon_id", "pdb_seq_num", "auth_seq_num", "pdb_mon_id", "auth_mon_id", "pdb_strand_id", "pdb_ins_code", "hetero")}
        for asym in unique_asym:
            rows = np.flatnonzero(label_asym == asym)
            if not len(rows) or entity_id[rows[0]] not in full_sequences:
                continue
            entity = str(entity_id[rows[0]])
            sequence, _modality = full_sequences[entity]
            by_position = {}
            for row in rows:
                by_position.setdefault(int(sequence_positions[row]), int(row))
            for position, component in enumerate(sequence, start=1):
                row = by_position.get(position)
                values = (asym, entity, str(position), str(component),
                          "?" if row is None else str(auth_seq[row]),
                          "?" if row is None else str(auth_seq[row]),
                          "?" if row is None else str(comp_id[row]),
                          "?" if row is None else str(comp_id[row]),
                          str(auth_asym[rows[0]]),
                          "." if row is None else (str(atom_array.ins_code[row]) or "."), "n")
                for name, value in zip(scheme, values, strict=True):
                    scheme[name].append(value)
        block["pdbx_poly_seq_scheme"] = scheme

    if full_sequences:
        extra = [(component, modality) for sequence, modality in full_sequences.values()
                 for component in dict.fromkeys(sequence)]
        block["chem_comp"] = _chem_comp_category(
            comp_id=np.concatenate((comp_id, np.asarray([item[0] for item in extra]))),
            mol_type=np.concatenate((mol_type, np.asarray([item[1] for item in extra]))),
        )
    else:
        block["chem_comp"] = _chem_comp_category(comp_id=comp_id, mol_type=mol_type)

    internal = {
        "comp_id": [],
        "atom_id_1": [],
        "atom_id_2": [],
        "value_order": [],
        "pdbx_aromatic_flag": [],
    }
    internal_order_by_pair: dict[tuple[str, str, str], tuple[str, str]] = {}
    external = {
        "id": [],
        "conn_type_id": [],
        "ptnr1_label_asym_id": [],
        "ptnr1_label_comp_id": [],
        "ptnr1_label_seq_id": [],
        "ptnr1_label_atom_id": [],
        "ptnr1_auth_asym_id": [],
        "ptnr1_auth_seq_id": [],
        "pdbx_ptnr1_PDB_ins_code": [],
        "ptnr2_label_asym_id": [],
        "ptnr2_label_comp_id": [],
        "ptnr2_label_seq_id": [],
        "ptnr2_label_atom_id": [],
        "ptnr2_auth_asym_id": [],
        "ptnr2_auth_seq_id": [],
        "pdbx_ptnr2_PDB_ins_code": [],
        "pdbx_value_order": [],
    }
    canonical_polymer = _fast_canonical_polymer_bonds(atom_array)
    if canonical_polymer is None:
        canonical_polymer = _canonical_polymer_bonds(atom_array)
    residue_uid = _annotation(atom_array, "residue_uid", -1).astype(np.int64)
    shared_modified_edges = _shared_modified_polymer_component_edges(
        atom_array,
        mol_type=mol_type,
        comp_id=comp_id,
    )
    for i, j, raw_type in _bond_rows(atom_array):
        i, j = int(i), int(j)
        is_coordination = int(raw_type) == int(struc.BondType.COORDINATION)
        same_component = int(component_uid[i]) == int(component_uid[j])
        both_ligand = mol_type[i] == mol_type[j] == "ligand"
        nonstandard_polymer_component = (
            same_component
            and int(residue_uid[i]) == int(residue_uid[j])
            and mol_type[i] == mol_type[j]
            and mol_type[i] in _POLYMER_TYPES
            and str(comp_id[i]) == str(comp_id[j])
            and str(comp_id[i]) not in _STANDARD_COMPONENT_IDS
        )
        touches_design = bool(atom_array.is_design[i] or atom_array.is_design[j])
        shared_modified_edge = False
        if nonstandard_polymer_component and not is_coordination:
            order, aromatic_flag = _internal_bond_descriptor(int(raw_type))
            atom_i, atom_j = sorted(
                (str(atom_array.atom_name[i]), str(atom_array.atom_name[j]))
            )
            shared_modified_edge = (
                atom_i,
                atom_j,
                order,
                aromatic_flag,
            ) in shared_modified_edges.get(str(comp_id[i]), frozenset())
        if (
            same_component
            and (both_ligand or shared_modified_edge)
            and not is_coordination
        ):
            order, aromatic_flag = _internal_bond_descriptor(int(raw_type))
            if order is None:
                continue
            resolved_comp_id = (
                component_name[int(component_uid[i])]
                if both_ligand
                else str(comp_id[i])
            )
            atom_i, atom_j = sorted(
                (str(atom_array.atom_name[i]), str(atom_array.atom_name[j]))
            )
            edge_key = (resolved_comp_id, atom_i, atom_j)
            previous_order = internal_order_by_pair.get(edge_key)
            descriptor = (order, aromatic_flag)
            if previous_order is not None and previous_order != descriptor:
                raise StructureWriteError(
                    f"component {resolved_comp_id!r} reuses internal bond "
                    f"{atom_i!r}-{atom_j!r} with conflicting orders"
                )
            if previous_order == descriptor:
                continue
            internal_order_by_pair[edge_key] = descriptor
            internal["comp_id"].append(resolved_comp_id)
            internal["atom_id_1"].append(atom_i)
            internal["atom_id_2"].append(atom_j)
            internal["value_order"].append(order)
            internal["pdbx_aromatic_flag"].append(aromatic_flag)
            continue
        key = tuple(sorted((i, j)))
        expected_polymer_order = canonical_polymer.get(key)
        same_residue = int(residue_uid[i]) == int(residue_uid[j])
        if expected_polymer_order is not None and not nonstandard_polymer_component:
            if not _template_order_matches(int(raw_type), expected_polymer_order):
                raise StructureWriteError(
                    f"polymer bond rows {key} conflict with canonical order"
                )
            if same_residue:
                continue
        if is_coordination:
            connection_type = "metalc"
            order = "?"
        else:
            connection_type = "covale"
            order = _bond_order(
                int(raw_type),
                inter_component=True,
                touches_design=touches_design,
            )
            if order is None:
                raise AssertionError("coordination routing invariant failed")
        external["id"].append(
            f"{connection_type}_{len(external['id']) + 1:06d}"
        )
        external["conn_type_id"].append(connection_type)
        for side, row in (("ptnr1", i), ("ptnr2", j)):
            external[f"{side}_label_asym_id"].append(str(label_asym[row]))
            external[f"{side}_label_comp_id"].append(str(comp_id[row]))
            external[f"{side}_label_seq_id"].append(
                _label_seq(atom_array, row, str(mol_type[row]))
            )
            external[f"{side}_label_atom_id"].append(str(atom_array.atom_name[row]))
            external[f"{side}_auth_asym_id"].append(str(auth_asym[row]))
            external[f"{side}_auth_seq_id"].append(str(auth_seq[row]))
            external[f"pdbx_{side}_PDB_ins_code"].append(str(atom_array.ins_code[row]) or ".")
        external["pdbx_value_order"].append(order)

    if internal["comp_id"]:
        block["chem_comp_bond"] = internal
    if external["id"]:
        block["struct_conn"] = external
        if "metalc" in external["conn_type_id"]:
            block["struct_conn_type"] = {"id": ["covale", "metalc"], "criteria": [COMPLETE_BACKBONE_CRITERIA, "Author supplied metal coordination."], "reference": [".", "."]}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_cif_fast(_safe_block_name(entry_id), block, output_path)


def assert_pdb_representable(
    atom_array: struc.AtomArray, *, topology_atoms: struc.AtomArray | None = None
) -> None:
    """Reject every known PDB information-loss case before sampling/writing."""

    mol_type = _annotation(atom_array, "mol_type", "").astype(str)
    if np.any(mol_type == "ligand"):
        raise StructureWriteError("PDB output does not support ligand content")
    if len(atom_array) > 99999:
        raise StructureWriteError("PDB atom serials exceed 99999")
    source_cif = _annotation(atom_array, "source_format", "").astype(str) == "cif"
    if source_cif.any():
        label_asym = _annotation(atom_array, "label_asym_id", "").astype(str)
        auth_asym = _annotation(atom_array, "auth_asym_id", "").astype(str)
        auth_seq = _annotation(atom_array, "auth_seq_id", "").astype(str)
        if (np.any(auth_asym[source_cif] != label_asym[source_cif])
                or np.any(auth_seq[source_cif] != atom_array.res_id[source_cif].astype(str))):
            raise StructureWriteError("PDB cannot preserve distinct CIF label/author namespaces")

    if "asym_uid" in atom_array.get_annotation_categories():
        asym_uid = np.asarray(atom_array.asym_uid, dtype=np.int64)
        if np.any(asym_uid < 0):
            raise StructureWriteError("asym_uid must be finalized before writing")
        chain_id = np.asarray(atom_array.chain_id).astype(str)
        for chain in dict.fromkeys(chain_id.tolist()):
            if len(np.unique(asym_uid[chain_id == chain])) != 1:
                raise StructureWriteError(
                    f"PDB author chain {chain!r} contains multiple TER segments"
                )

    coord = np.asarray(atom_array.coord, dtype=np.float64)
    if coord.shape != (len(atom_array), 3) or not np.isfinite(coord).all():
        raise StructureWriteError("PDB output coordinates must be finite")
    if any(len(f"{float(value):8.3f}") > 8 for value in coord.flat):
        raise StructureWriteError("PDB coordinates exceed the 8.3 field range")

    for value in np.asarray(atom_array.chain_id).astype(str):
        if not value.isascii() or _PDB_CHAIN_ID.fullmatch(value) is None:
            raise StructureWriteError(
                "PDB chain IDs must be exactly one character and one safe ASCII "
                "alphanumeric byte"
            )
    for value in np.asarray(atom_array.res_name).astype(str):
        if not value.isascii() or _PDB_RES_NAME.fullmatch(value) is None:
            raise StructureWriteError("PDB residue names are not safely representable")
    for value in np.asarray(atom_array.atom_name).astype(str):
        if not value.isascii() or _PDB_ATOM_NAME.fullmatch(value) is None:
            raise StructureWriteError("PDB atom names are not safely representable")
    for value in np.asarray(atom_array.ins_code).astype(str):
        if value and (
            not value.isascii() or _PDB_INS_CODE.fullmatch(value) is None
        ):
            raise StructureWriteError(
                "PDB insertion codes are not safely representable"
            )
    for value in np.asarray(atom_array.element).astype(str):
        if not value.isascii() or _PDB_ELEMENT.fullmatch(value) is None:
            raise StructureWriteError("PDB elements are not safely representable")

    residue_ids = np.asarray(atom_array.res_id, dtype=np.int64)
    if np.any((residue_ids < -999) | (residue_ids > 9999)):
        raise StructureWriteError("PDB residue numbers exceed -999..9999")
    if _explicit_cif_bonds(
        atom_array if topology_atoms is None else topology_atoms,
        include_backbone=False,
    ):
        raise StructureWriteError(
            "PDB output cannot preserve noncanonical/cross-chain polymer bonds"
        )


def write_pdb(atom_array: struc.AtomArray, output_path: str | Path) -> None:
    pdb_atoms = copy.deepcopy(atom_array)
    source_format = _annotation(pdb_atoms, "source_format", "").astype(str)
    retained = source_format == "pdb"
    if retained.any():
        try:
            pdb_atoms.res_id[retained] = _annotation(pdb_atoms, "auth_seq_id", "")[retained].astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise StructureWriteError("source PDB residue numbering is not integral") from exc
        pdb_atoms.ins_code[retained] = _annotation(pdb_atoms, "auth_ins_code", "")[retained]
        pdb_atoms.chain_id[retained] = _annotation(pdb_atoms, "auth_asym_id", "")[retained]
    assert_pdb_representable(pdb_atoms, topology_atoms=atom_array)
    try:
        charges = formal_charge_values(pdb_atoms)
    except ValueError as exc:
        raise StructureWriteError(str(exc)) from exc
    charge_fields = None
    if charges is not None and any(value is not None for value in charges):
        charge_fields = []
        for value in charges:
            if value is None:
                charge_fields.append("  ")
            elif not -9 <= value <= 9:
                raise StructureWriteError("PDB formal charges exceed -9..9")
            else:
                charge_fields.append(" 0" if value == 0 else f"{abs(value)}{'+' if value > 0 else '-'}")
    # Biotite maps an explicit zero to blank and cannot represent None.  Keep
    # those two states separate by serializing the final two columns directly.
    if "charge" in pdb_atoms.get_annotation_categories():
        pdb_atoms.del_annotation("charge")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_file = pdb.PDBFile()
    pdb_atoms.set_annotation(
        "atom_id", np.arange(1, len(pdb_atoms) + 1, dtype=np.int64)
    )
    pdb.set_structure(pdb_file, pdb_atoms, hybrid36=False)
    if charge_fields is not None:
        index = 0
        for line_index, line in enumerate(pdb_file.lines):
            if line.startswith(("ATOM  ", "HETATM")):
                pdb_file.lines[line_index] = line[:78].ljust(78) + charge_fields[index] + line[80:]
                index += 1
    if retained.any():
        sequences = _annotation(pdb_atoms, "source_polymer_sequence", None)
        seqres = []
        for chain in dict.fromkeys(pdb_atoms.chain_id[retained].tolist()):
            rows = np.flatnonzero(retained & (pdb_atoms.chain_id == chain))
            if not len(rows) or np.any(pdb_atoms.is_design[rows]):
                continue
            sequence = sequences[int(rows[0])]
            if sequence is None:
                continue
            if len(sequence) > 9999:
                raise StructureWriteError("confirmed polymer sequence exceeds PDB SEQRES limits")
            for start in range(0, len(sequence), 13):
                components = " ".join(str(value) for value in sequence[start:start + 13])
                seqres.append(f"SEQRES {start // 13 + 1:3d} {chain} {len(sequence):4d}  {components}")
        if seqres:
            pdb_file.lines = seqres + pdb_file.lines
    pdb_file.write(output_path)


def _semantic_bonds(atom_array: struc.AtomArray) -> list[tuple[int, int, int]]:
    return sorted(
        (min(int(i), int(j)), max(int(i), int(j)), int(order))
        for i, j, order in _bond_rows(atom_array)
    )


def _explicit_cif_bonds(
    atom_array: struc.AtomArray, *, include_backbone: bool = True
) -> list[tuple[int, int, int]]:
    """Bonds represented explicitly in chem_comp_bond/struct_conn.

    Canonical intra-residue edges are reconstructed from component templates;
    all inter-residue edges are emitted to preserve the observed source graph.
    PDB representability checks may exclude its implicit canonical backbone.
    """

    canonical = _canonical_polymer_bonds(atom_array)
    component_uid = _annotation(atom_array, "component_uid", -1).astype(np.int64)
    residue_uid = _annotation(atom_array, "residue_uid", -1).astype(np.int64)
    mol_type = _annotation(atom_array, "mol_type", "").astype(str)
    res_name = np.asarray(atom_array.res_name).astype(str)
    result: list[tuple[int, int, int]] = []
    for raw_i, raw_j, raw_order in _bond_rows(atom_array):
        row_i, row_j = int(raw_i), int(raw_j)
        key = tuple(sorted((row_i, row_j)))
        order = int(raw_order)
        nonstandard_polymer_component = (
            int(component_uid[row_i]) == int(component_uid[row_j])
            and int(residue_uid[row_i]) == int(residue_uid[row_j])
            and mol_type[row_i] == mol_type[row_j]
            and mol_type[row_i] in _POLYMER_TYPES
            and res_name[row_i] == res_name[row_j]
            and res_name[row_i] not in _STANDARD_COMPONENT_IDS
        )
        if nonstandard_polymer_component:
            result.append((key[0], key[1], order))
            continue
        expected = canonical.get(key)
        if expected is not None and (
            not include_backbone or residue_uid[row_i] == residue_uid[row_j]
        ):
            if not _template_order_matches(order, expected):
                raise StructureWriteError(
                    f"polymer bond rows {key} conflict with canonical order"
                )
            continue
        result.append((key[0], key[1], order))
    return sorted(result)


def _lean_cif_explicit_bonds(path: Path) -> list[tuple[int, int, int]]:
    """Read the exact lean chem_comp_bond/struct_conn topology directly.

    Biotite's generic structure converter requires optional dictionary columns
    that the public lean CIF contract deliberately omits.  This validator reads
    only the emitted standard endpoint/order columns and never falls back to CCD
    or distance perception.
    """

    cif = pdbx.CIFFile.read(path)
    if len(cif) != 1:
        raise StructureWriteError("written CIF must contain exactly one data block")
    block = next(iter(cif.values()))
    atom_site = block.get("atom_site")
    if atom_site is None:
        raise StructureWriteError("written CIF is missing atom_site")

    def values(category: pdbx.CIFCategory, name: str) -> list[str]:
        column = category.get(name)
        if column is None:
            raise StructureWriteError(f"written CIF is missing column {name}")
        return column.as_array(str).tolist()

    atom_asym = values(atom_site, "label_asym_id")
    atom_comp = values(atom_site, "label_comp_id")
    atom_seq = values(atom_site, "label_seq_id")
    atom_auth_asym = (
        values(atom_site, "auth_asym_id")
        if "auth_asym_id" in atom_site
        else list(atom_asym)
    )
    atom_auth_seq = (
        values(atom_site, "auth_seq_id")
        if "auth_seq_id" in atom_site
        else list(atom_seq)
    )
    atom_name = values(atom_site, "label_atom_id")
    atom_group = values(atom_site, "group_PDB")
    atom_count = len(atom_asym)
    if not all(
        len(column) == atom_count
        for column in (
            atom_comp,
            atom_seq,
            atom_auth_asym,
            atom_auth_seq,
            atom_name,
            atom_group,
        )
    ):
        raise StructureWriteError("written CIF atom_site columns have unequal lengths")

    order_to_type = {
        "sing": int(struc.BondType.SINGLE),
        "doub": int(struc.BondType.DOUBLE),
        "trip": int(struc.BondType.TRIPLE),
        "arom": int(struc.BondType.AROMATIC),
        "?": int(struc.BondType.ANY),
    }
    result: list[tuple[int, int, int]] = []

    internal = block.get("chem_comp_bond")
    if internal is not None:
        comp_ids = values(internal, "comp_id")
        atom_ids_1 = values(internal, "atom_id_1")
        atom_ids_2 = values(internal, "atom_id_2")
        orders = values(internal, "value_order")
        flags = (
            values(internal, "pdbx_aromatic_flag")
            if "pdbx_aromatic_flag" in internal
            else ["?"] * len(comp_ids)
        )
        if not all(
            len(column) == len(comp_ids)
            for column in (atom_ids_1, atom_ids_2, orders, flags)
        ):
            raise StructureWriteError(
                "written CIF chem_comp_bond columns have unequal lengths"
            )
        definitions: dict[str, list[tuple[str, str, int]]] = {}
        exact_definitions = {}
        for comp_id, atom_id_1, atom_id_2, raw_order, raw_flag in zip(
            comp_ids, atom_ids_1, atom_ids_2, orders, flags, strict=True
        ):
            order = order_to_type.get(raw_order.lower())
            if order is None:
                raise StructureWriteError(
                    f"written CIF has unsupported bond order {raw_order!r}"
                )
            flag = raw_flag.strip().upper()
            if flag not in ("", ".", "?", "N", "Y"):
                raise StructureWriteError("written CIF has invalid aromatic flag")
            if flag == "Y":
                if order not in (1, 2, 3, 9):
                    raise StructureWriteError("written CIF aromatic flag contradicts bond order")
                order = {1: 5, 2: 6, 3: 7, 9: 9}[order]
            elif flag == "N" and order == 9:
                raise StructureWriteError("written CIF nonaromatic flag contradicts AROM")
            pair = (comp_id, *sorted((atom_id_1, atom_id_2)))
            previous = exact_definitions.get(pair)
            if previous is not None and previous != order:
                raise StructureWriteError("written CIF has conflicting explicit component bonds")
            exact_definitions[pair] = order
            definitions.setdefault(comp_id, []).append(
                (atom_id_1, atom_id_2, order)
            )
        for comp_id, bonds in definitions.items():
            instance_keys = list(
                dict.fromkeys(
                    (
                        atom_asym[row],
                        atom_seq[row],
                        atom_auth_asym[row],
                        atom_auth_seq[row],
                        atom_comp[row],
                    )
                    for row in range(atom_count)
                    if atom_comp[row] == comp_id
                )
            )
            if not instance_keys:
                raise StructureWriteError(
                    f"written CIF bond component {comp_id!r} has no atom instance"
                )
            for asym_id, seq_id, auth_asym_id, auth_seq_id, _ in instance_keys:
                instance_rows = [
                    row
                    for row in range(atom_count)
                    if atom_asym[row] == asym_id
                    and atom_seq[row] == seq_id
                    and atom_auth_asym[row] == auth_asym_id
                    and atom_auth_seq[row] == auth_seq_id
                    and atom_comp[row] == comp_id
                ]
                by_name: dict[str, list[int]] = {}
                for row in instance_rows:
                    by_name.setdefault(atom_name[row], []).append(row)
                for atom_id_1, atom_id_2, order in bonds:
                    rows_1 = by_name.get(atom_id_1, [])
                    rows_2 = by_name.get(atom_id_2, [])
                    if len(rows_1) != 1 or len(rows_2) != 1:
                        raise StructureWriteError(
                            "written CIF chem_comp_bond endpoint is not unique"
                        )
                    i, j = sorted((rows_1[0], rows_2[0]))
                    result.append((i, j, order))

    external = block.get("struct_conn")
    if external is not None:
        connection_types = values(external, "conn_type_id")
        orders = values(external, "pdbx_value_order")
        endpoint_columns = {
            side: {
                field: values(external, f"{side}_label_{field}")
                for field in ("asym_id", "comp_id", "seq_id", "atom_id")
            }
            for side in ("ptnr1", "ptnr2")
        }
        for side in ("ptnr1", "ptnr2"):
            endpoint_columns[side]["auth_asym_id"] = (
                values(external, f"{side}_auth_asym_id")
                if f"{side}_auth_asym_id" in external
                else list(endpoint_columns[side]["asym_id"])
            )
            endpoint_columns[side]["auth_seq_id"] = (
                values(external, f"{side}_auth_seq_id")
                if f"{side}_auth_seq_id" in external
                else list(endpoint_columns[side]["seq_id"])
            )
        row_count = len(connection_types)
        if len(orders) != row_count or any(
            len(column) != row_count
            for side in endpoint_columns.values()
            for column in side.values()
        ):
            raise StructureWriteError("written CIF struct_conn columns have unequal lengths")
        for connection_index, connection_type in enumerate(connection_types):
            endpoint_rows: list[int] = []
            for side in ("ptnr1", "ptnr2"):
                columns = endpoint_columns[side]
                matches = [
                    row
                    for row in range(atom_count)
                    if atom_asym[row] == columns["asym_id"][connection_index]
                    and atom_comp[row] == columns["comp_id"][connection_index]
                    and atom_seq[row] == columns["seq_id"][connection_index]
                    and atom_name[row] == columns["atom_id"][connection_index]
                    and (
                        str(columns["auth_seq_id"][connection_index]).strip()
                        in {"", ".", "?"}
                        or (
                            atom_auth_asym[row]
                            == columns["auth_asym_id"][connection_index]
                            and atom_auth_seq[row]
                            == columns["auth_seq_id"][connection_index]
                        )
                    )
                ]
                if len(matches) != 1:
                    raise StructureWriteError(
                        "written CIF struct_conn endpoint is not unique"
                    )
                endpoint_rows.append(matches[0])
            if connection_type.lower() == "metalc":
                order = int(struc.BondType.COORDINATION)
            else:
                order = order_to_type.get(orders[connection_index].lower())
                if order is None:
                    raise StructureWriteError(
                        f"written CIF has unsupported bond order "
                        f"{orders[connection_index]!r}"
                    )
            i, j = sorted(endpoint_rows)
            result.append((i, j, order))
    return sorted(result)


def _assert_roundtrip_semantics(
    original: struc.AtomArray,
    reread: struc.AtomArray,
) -> None:
    if len(original) != len(reread):
        raise StructureWriteError("standalone readback changed the atom count")
    original_labels = (
        np.asarray(original.label_asym_id).astype(str)
        if "label_asym_id" in original.get_annotation_categories()
        else np.asarray(original.chain_id).astype(str)
    )
    reread_labels = (
        np.asarray(reread.label_asym_id).astype(str)
        if "label_asym_id" in reread.get_annotation_categories()
        else np.asarray(reread.chain_id).astype(str)
    )
    if not np.array_equal(original_labels, reread_labels):
        raise StructureWriteError("standalone readback changed label_asym_id")
    for name in ("atom_name", "element", "mol_type"):
        if not np.array_equal(
            np.asarray(original.get_annotation(name)).astype(str),
            np.asarray(reread.get_annotation(name)).astype(str),
        ):
            raise StructureWriteError(f"standalone readback changed {name}")
    if "asym_uid" in original.get_annotation_categories():
        if _partition_signature(original.asym_uid) != _partition_signature(
            reread.asym_uid
        ):
            raise StructureWriteError("standalone readback changed the asym partition")
    polymer = np.asarray(original.mol_type).astype(str) != "ligand"
    for name in ("res_name", "res_id", "ins_code"):
        if not np.array_equal(
            np.asarray(original.get_annotation(name))[polymer],
            np.asarray(reread.get_annotation(name))[polymer],
        ):
            raise StructureWriteError(f"standalone readback changed polymer {name}")
    if not np.allclose(
        np.asarray(original.coord, dtype=np.float64),
        np.asarray(reread.coord, dtype=np.float64),
        atol=1.0e-5,
        rtol=1.0e-6,
    ):
        raise StructureWriteError("standalone readback changed coordinates")
    if _canonical_polymer_bonds(original) != _canonical_polymer_bonds(reread):
        raise StructureWriteError("standalone readback changed canonical polymer topology")
    if _explicit_cif_bonds(original) != _explicit_cif_bonds(reread):
        raise StructureWriteError("standalone readback changed explicit topology")


def _validate_temporary_readback(
    original: struc.AtomArray,
    path: Path,
    validated_ligand_identities: Mapping[int, ValidatedLigandIdentity] | None = None,
) -> None:
    # Import lazily to keep the writer module acyclic at import time.
    from .reader import read_structure

    suffix = path.suffix.lower()
    if suffix == ".cif":
        with warnings.catch_warnings():
            for auth_name, label_name in (
                ("auth_asym_id", "label_asym_id"),
                ("auth_seq_id", "label_seq_id"),
                ("auth_comp_id", "label_comp_id"),
                ("auth_atom_id", "label_atom_id"),
            ):
                message = re.escape(
                    f"Attribute '{auth_name}' not found within 'atom_site' "
                    f"category. The fallback attribute '{label_name}' will be "
                    "used instead"
                )
                warnings.filterwarnings(
                    "ignore",
                    message=rf"^{message}$",
                    category=UserWarning,
                )
            reread = read_structure(path)
            independent = pdbx.get_structure(
                pdbx.CIFFile.read(path), model=1, include_bonds=False,
                use_author_fields=False,
            )
    else:
        reread = read_structure(path)
        independent_file = pdb.PDBFile.read(path)
        independent = pdb.get_structure(independent_file, model=1, extra_fields=[])
    _assert_roundtrip_semantics(original, reread)
    if suffix == ".cif":
        _original_uids = _annotation(original, "component_uid", -1).astype(np.int64)
        original_mol_type = _annotation(original, "mol_type", "").astype(str)
        expected_components = _component_ids(
            original,
            validated_ligand_identities,
        )[1]
        for original_uid, expected_name in expected_components.items():
            original_rows = np.flatnonzero(_original_uids == int(original_uid))
            if len(original_rows) == 0:
                raise StructureWriteError("temporary readback lost a source component")
            if not np.all(original_mol_type[original_rows] == "ligand"):
                continue
            if len(reread) != len(original) or not np.all(
                np.asarray(reread.res_name)[original_rows].astype(str) == expected_name
            ):
                raise StructureWriteError(
                    "temporary readback changed the format-visible component ID"
                )
    if len(independent) != len(original):
        raise StructureWriteError("independent readback changed the atom count")
    expected_chain_id = (
        np.asarray(original.label_asym_id).astype(str)
        if suffix == ".cif"
        and "label_asym_id" in original.get_annotation_categories()
        else np.asarray(original.chain_id).astype(str)
    )
    if not np.array_equal(
        expected_chain_id,
        np.asarray(independent.chain_id).astype(str),
    ):
        raise StructureWriteError(
            "independent readback changed the format-visible chain identity"
        )
    for name in ("atom_name", "element", "hetero"):
        if not np.array_equal(
            np.asarray(original.get_annotation(name)),
            np.asarray(independent.get_annotation(name)),
        ):
            raise StructureWriteError(
                f"independent readback changed atom identity/order field {name}"
            )
    polymer = np.asarray(original.mol_type).astype(str) != "ligand"
    for name in ("res_name", "res_id", "ins_code"):
        if not np.array_equal(
            np.asarray(original.get_annotation(name))[polymer],
            np.asarray(independent.get_annotation(name))[polymer],
        ):
            raise StructureWriteError(
                f"independent readback changed polymer identity/order field {name}"
            )
    if not np.allclose(
        np.asarray(independent.coord, dtype=np.float64),
        np.asarray(original.coord, dtype=np.float64),
        atol=1.0e-5,
        rtol=1.0e-6,
    ):
        raise StructureWriteError("independent readback changed coordinates")
    if suffix == ".cif":
        if _lean_cif_explicit_bonds(path) != _explicit_cif_bonds(original):
            raise StructureWriteError(
                "independent readback changed explicit topology"
            )


def write_structure_atomic(
    atom_array: struc.AtomArray,
    output_path: str | Path,
    *,
    entry_id: str,
    validated_ligand_identities: Mapping[int, ValidatedLigandIdentity] | None = None,
) -> Path:
    """Serialize beside the destination and atomically publish on success."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix not in {".cif", ".pdb"}:
        raise StructureWriteError(f"unsupported output extension {suffix!r}")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=f".tmp{suffix}",
        dir=output_path.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        if suffix == ".cif":
            write_cif(
                atom_array,
                temporary_path,
                entry_id=entry_id,
                validated_ligand_identities=validated_ligand_identities,
            )
        else:
            write_pdb(atom_array, temporary_path)
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


__all__ = [
    "StructureWriteError",
    "assert_pdb_representable",
    "write_cif",
    "write_pdb",
    "write_structure_atomic",
]
