"""CCD identity proofs for standalone ligand output naming.

This module is intentionally coordinate-free.  It compares the heavy-atom
inventory and complete internal concrete graph in the authoring AtomArray with
the compact CCD topology used by the standalone reader.  A failed comparison
is represented by ``None`` and is never repaired by guessing a component ID.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import biotite.structure as struc

from ..core.records import ValidatedLigandIdentity


_HYDROGEN_ELEMENTS = frozenset({"H", "D", "T"})
_ORDER_NAMES = {
    int(struc.BondType.SINGLE): "single",
    int(struc.BondType.DOUBLE): "double",
    int(struc.BondType.TRIPLE): "triple",
    int(struc.BondType.AROMATIC): "aromatic",
    int(struc.BondType.AROMATIC_SINGLE): "aromatic",
    int(struc.BondType.AROMATIC_DOUBLE): "aromatic",
    int(struc.BondType.AROMATIC_TRIPLE): "aromatic",
}
_CCD_ORDER_NAMES = {
    int(struc.BondType.SINGLE): "single",
    int(struc.BondType.DOUBLE): "double",
    int(struc.BondType.TRIPLE): "triple",
    int(struc.BondType.AROMATIC): "aromatic",
}


def _order_name(raw_order: object) -> str | None:
    try:
        return _ORDER_NAMES.get(int(raw_order))
    except (TypeError, ValueError):
        return None


def _ccd_order_name(raw_order: object) -> str | None:
    try:
        return _CCD_ORDER_NAMES.get(int(raw_order))
    except (TypeError, ValueError):
        return None


def _heavy_rows(atom_array: struc.AtomArray, rows: Sequence[int]) -> tuple[int, ...] | None:
    normalized = tuple(int(row) for row in rows)
    if not normalized or len(set(normalized)) != len(normalized):
        return None
    if any(row < 0 or row >= len(atom_array) for row in normalized):
        return None
    elements = np.asarray(atom_array.element).astype(str)
    return tuple(
        row for row in normalized if elements[row].strip().upper() not in _HYDROGEN_ELEMENTS
    )


def compute_ligand_component_signature(
    atom_array: struc.AtomArray,
    rows: Sequence[int],
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...]] | None:
    """Return a normalized heavy inventory and complete concrete internal graph.

    ``None`` means the source cannot prove a CCD identity (for example an
    ambiguous atom name, an ANY/unknown order, or a non-ligand component).
    Hydrogen rows and hydrogen-involving bonds are ignored because compact CCD
    entries retain optional hydrogens while standalone authoring arrays are
    heavy-atom-only.
    """

    normalized = tuple(int(row) for row in rows)
    if not normalized or len(set(normalized)) != len(normalized):
        return None
    categories = set(atom_array.get_annotation_categories())
    if "mol_type" not in categories or "res_name" not in categories:
        return None
    if set(np.asarray(atom_array.mol_type)[list(normalized)].astype(str)) != {"ligand"}:
        return None
    if "component_uid" in categories:
        component_values = {
            int(value) for value in np.asarray(atom_array.component_uid)[list(normalized)]
        }
        if len(component_values) != 1:
            return None
    if "identity_condition" not in categories:
        return None
    if not bool(np.asarray(atom_array.identity_condition)[list(normalized)].astype(bool).all()):
        return None
    residue_names = {
        str(value).strip().upper()
        for value in np.asarray(atom_array.res_name)[list(normalized)]
    }
    if len(residue_names) != 1 or not next(iter(residue_names)):
        return None
    heavy = _heavy_rows(atom_array, normalized)
    if not heavy:
        return None
    names = [str(atom_array.atom_name[row]).strip() for row in heavy]
    elements = [str(atom_array.element[row]).strip().upper() for row in heavy]
    if any(not name or name in {".", "?"} for name in names):
        return None
    if any(not element or element in {".", "?"} for element in elements):
        return None
    if len(names) != len(set(names)):
        return None
    atom_signature = tuple(sorted(zip(names, elements, strict=True)))
    row_to_name = {row: name for row, name in zip(heavy, names, strict=True)}
    edges: dict[tuple[str, str], str] = {}
    if atom_array.bonds is not None:
        heavy_rows = set(heavy)
        for raw_i, raw_j, raw_order in atom_array.bonds.as_array():
            row_i, row_j = int(raw_i), int(raw_j)
            if row_i not in heavy_rows or row_j not in heavy_rows:
                continue
            order = _order_name(raw_order)
            if order is None:
                return None
            left, right = sorted((row_to_name[row_i], row_to_name[row_j]))
            if left == right:
                return None
            key = (left, right)
            previous = edges.get(key)
            if previous is not None and previous != order:
                return None
            edges[key] = order
    edge_signature = tuple(
        (left, right, order)
        for (left, right), order in sorted(edges.items())
    )
    return atom_signature, edge_signature


def build_validated_ligand_identity(
    component_id: str,
    atom_signature: Sequence[Sequence[str]],
    edge_signature: Sequence[Sequence[str]],
) -> ValidatedLigandIdentity:
    """Build a sidecar from an already independently validated exact graph."""

    return ValidatedLigandIdentity(
        component_id=str(component_id),
        atom_signature=tuple(tuple(str(value) for value in row) for row in atom_signature),
        edge_signature=tuple(tuple(str(value) for value in row) for row in edge_signature),
    )


def ligand_identity_matches_component(
    atom_array: struc.AtomArray,
    rows: Sequence[int],
    identity: ValidatedLigandIdentity,
) -> bool:
    """Revalidate an immutable sidecar against the current AtomArray."""

    categories = set(atom_array.get_annotation_categories())
    if "component_uid" in categories:
        component_values = {
            int(value) for value in np.asarray(atom_array.component_uid)[list(rows)]
        }
        if len(component_values) != 1:
            return False
    residue_names = {
        str(value).strip().upper()
        for value in np.asarray(atom_array.res_name)[list(rows)]
    }
    if residue_names != {identity.component_id}:
        return False
    signature = compute_ligand_component_signature(atom_array, rows)
    return signature == (identity.atom_signature, identity.edge_signature)


def _topology_signature(topology: object) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...]] | None:
    atom_elements = tuple(getattr(topology, "atom_elements", ()))
    if not atom_elements:
        return None
    topology_atom_names = {
        str(value).strip() for value in getattr(topology, "atom_names", ())
    }
    elements: dict[str, str] = {}
    for raw_name, raw_element in atom_elements:
        name = str(raw_name).strip()
        element = str(raw_element).strip().upper()
        if not name or not element or name in elements and elements[name] != element:
            return None
        elements[name] = element
    heavy_elements = {
        name: element
        for name, element in elements.items()
        if element not in _HYDROGEN_ELEMENTS
    }
    if not heavy_elements:
        return None
    if topology_atom_names and not set(elements).issubset(topology_atom_names):
        return None
    edges: dict[tuple[str, str], str] = {}
    for raw_left, raw_right, raw_order in getattr(topology, "bonds", ()):
        left, right = str(raw_left), str(raw_right)
        if left not in heavy_elements or right not in heavy_elements:
            continue
        order = _ccd_order_name(raw_order)
        if order is None:
            return None
        pair = tuple(sorted((left, right)))
        previous = edges.get(pair)
        if previous is not None and previous != order:
            return None
        edges[pair] = order
    return (
        tuple(sorted(heavy_elements.items())),
        tuple((left, right, order) for (left, right), order in sorted(edges.items())),
    )


def _matches_specific_kekule_orders(
    atom_array: struc.AtomArray,
    rows: Sequence[int],
    topology: object,
) -> bool:
    """Allow exact known Kekule counterparts without changing witness encoding."""

    source = compute_ligand_component_signature(atom_array, rows)
    target = _topology_signature(topology)
    if source is None or target is None or source[0] != target[0]:
        return False
    source_edges = {(left, right): order for left, right, order in source[1]}
    target_edges = {(left, right): order for left, right, order in target[1]}
    if source_edges.keys() != target_edges.keys():
        return False
    heavy = _heavy_rows(atom_array, rows)
    if heavy is None:
        return False
    names = {row: str(atom_array.atom_name[row]).strip() for row in heavy}
    raw_orders: dict[tuple[str, str], int] = {}
    if atom_array.bonds is not None:
        for raw_i, raw_j, raw_order in atom_array.bonds.as_array():
            i, j = int(raw_i), int(raw_j)
            if i not in names or j not in names:
                continue
            pair = tuple(sorted((names[i], names[j])))
            order = int(raw_order)
            if pair in raw_orders and raw_orders[pair] != order:
                return False
            raw_orders[pair] = order
    counterparts = {
        int(struc.BondType.AROMATIC_SINGLE): "single",
        int(struc.BondType.AROMATIC_DOUBLE): "double",
        int(struc.BondType.AROMATIC_TRIPLE): "triple",
    }
    return all(
        order == target_edges[pair]
        or counterparts.get(raw_orders.get(pair)) == target_edges[pair]
        for pair, order in source_edges.items()
    )


def match_ligand_component_against_ccd(
    atom_array: struc.AtomArray,
    rows: Sequence[int],
    *,
    ccd_path: Path | None = None,
) -> ValidatedLigandIdentity | None:
    """Return one exact compact-CCD identity proof, or ``None``.

    The compact reader topology is the authority for both configured and
    packaged CCD paths.  Partial graphs, free/SMILES identities, unknown bond
    orders, and CCD files without element facts are intentionally non-matches.
    """

    signature = compute_ligand_component_signature(atom_array, rows)
    if signature is None:
        return None
    component_id = next(
        iter({str(value).strip().upper() for value in np.asarray(atom_array.res_name)[list(rows)]})
    )
    from .reader import _DEFAULT_COMPACT_CCD_PATH, _ccd_topologies

    path = _DEFAULT_COMPACT_CCD_PATH if ccd_path is None else Path(ccd_path)
    topologies = _ccd_topologies(path)
    topology = topologies.get(component_id)
    if topology is None:
        return None
    if _topology_signature(topology) != signature and not _matches_specific_kekule_orders(
        atom_array, rows, topology
    ):
        return None
    return ValidatedLigandIdentity(
        component_id=component_id,
        atom_signature=signature[0],
        edge_signature=signature[1],
    )


def validated_ligand_identities_from_ccd(
    atom_array: struc.AtomArray,
    *,
    ccd_path: Path | None = None,
) -> dict[int, ValidatedLigandIdentity]:
    """Find exact identities independently for every ligand component."""

    if "component_uid" not in atom_array.get_annotation_categories():
        return {}
    result: dict[int, ValidatedLigandIdentity] = {}
    component_uids = np.asarray(atom_array.component_uid, dtype=np.int64)
    for raw_uid in dict.fromkeys(component_uids.tolist()):
        uid = int(raw_uid)
        rows = np.flatnonzero(component_uids == uid)
        identity = match_ligand_component_against_ccd(
            atom_array,
            rows,
            ccd_path=ccd_path,
        )
        if identity is not None:
            result[uid] = identity
    return result


__all__ = [
    "build_validated_ligand_identity",
    "compute_ligand_component_signature",
    "ligand_identity_matches_component",
    "match_ligand_component_against_ccd",
    "validated_ligand_identities_from_ccd",
]
