"""Deterministic geometry repair for private hydrogens added by RDKit.

The public structure remains heavy-atom-only.  This helper is used only on a
private ``Chem.AddHs`` molecule before it is handed to a force-field backend.
It repairs the small class of added hydrogen coordinates that RDKit may leave
at an origin or as non-finite values, without changing the molecule graph or
any heavy-atom coordinates.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from rdkit import Chem


_MIN_BOND_DISTANCE_A = 0.4
_MAX_BOND_DISTANCE_A = 2.0
_MIN_DIRECTION_NORM = 1.0e-8
_MIN_CLEARANCE_A = 0.4


def _as_point(conformer: Any, index: int) -> np.ndarray:
    point = conformer.GetAtomPosition(int(index))
    return np.asarray((float(point.x), float(point.y), float(point.z)), dtype=float)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= _MIN_DIRECTION_NORM:
        return None
    return vector / norm


def _covalent_radius(atom: Any) -> float:
    atomic_number = int(atom.GetAtomicNum())
    if atomic_number <= 0:
        raise ValueError("hydrogen parent has no valid element")
    radius = float(Chem.GetPeriodicTable().GetRcovalent(atomic_number))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            f"no finite covalent radius for hydrogen parent element {atomic_number}"
        )
    return radius


def _candidate_directions(away: np.ndarray | None) -> list[np.ndarray]:
    """Return a fixed, frame-aware candidate set with no random state."""
    candidates: list[np.ndarray] = []
    if away is not None:
        # A deterministic frame around the local outward direction gives
        # methyl/methylene groups several separated choices while retaining the
        # local heavy-neighbour orientation.
        reference = np.eye(3, dtype=float)[
            int(np.argmin(np.abs(np.asarray(away, dtype=float))))
        ]
        side = _unit(np.cross(away, reference))
        if side is not None:
            other = _unit(np.cross(away, side))
            if other is not None:
                for polar in (1.0, 0.5, 0.0, -0.5, -1.0):
                    radial = math.sqrt(max(0.0, 1.0 - polar * polar))
                    for step in range(16):
                        theta = 2.0 * math.pi * step / 16.0
                        vector = (
                            polar * away
                            + radial * math.cos(theta) * side
                            + radial * math.sin(theta) * other
                        )
                        normalized = _unit(vector)
                        if normalized is not None:
                            candidates.append(normalized)
    # Canonical directions make the zero-sum frame deterministic as well.
    for vector in (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ):
        candidates.append(np.asarray(vector, dtype=float))
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            for z in (-1.0, 1.0):
                candidates.append(np.asarray((x, y, z), dtype=float) / math.sqrt(3.0))
    return candidates


def _repair_for_parent(
    molecule: Chem.Mol,
    conformer: Any,
    parent: Any,
    bad_hydrogens: list[int],
    valid_hydrogens: list[int],
) -> None:
    parent_index = int(parent.GetIdx())
    parent_point = _as_point(conformer, parent_index)
    if not np.isfinite(parent_point).all():
        raise ValueError(
            f"cannot repair hydrogen parent {parent_index}: coordinate is non-finite"
        )

    heavy_neighbours: list[np.ndarray] = []
    for neighbour in parent.GetNeighbors():
        neighbour_index = int(neighbour.GetIdx())
        if neighbour_index >= molecule.GetNumAtoms():
            continue
        if neighbour.GetAtomicNum() == 1:
            continue
        point = _as_point(conformer, neighbour_index)
        if not np.isfinite(point).all():
            continue
        direction = _unit(point - parent_point)
        if direction is not None:
            heavy_neighbours.append(direction)

    occupied_points: list[np.ndarray] = []
    occupied_directions: list[np.ndarray] = []
    for index in valid_hydrogens:
        point = _as_point(conformer, index)
        if not np.isfinite(point).all():
            continue
        direction = _unit(point - parent_point)
        if direction is None:
            continue
        occupied_points.append(point)
        occupied_directions.append(direction)

    # A parent with no usable heavy frame or already-positioned H frame cannot
    # receive a reproducibly oriented new H.  Fail explicitly rather than
    # manufacturing a molecule-wide embedding.
    if not heavy_neighbours and not occupied_directions:
        raise ValueError(
            f"cannot repair hydrogen parent {parent_index}: degenerate local heavy frame"
        )

    anchor_directions = heavy_neighbours + occupied_directions
    away = _unit(-np.sum(anchor_directions, axis=0))
    candidates = _candidate_directions(away)
    target_distance = _covalent_radius(parent) + float(
        Chem.GetPeriodicTable().GetRcovalent(1)
    )
    if not math.isfinite(target_distance) or target_distance <= 0.0:
        raise ValueError("hydrogen covalent radius is not finite")

    for hydrogen_index in sorted(bad_hydrogens):
        best: tuple[tuple[float, float, float, float], np.ndarray] | None = None
        for ordinal, direction in enumerate(candidates):
            point = parent_point + target_distance * direction
            if not np.isfinite(point).all():
                continue
            points = occupied_points + [
                _as_point(conformer, neighbour.GetIdx())
                for neighbour in parent.GetNeighbors()
                if neighbour.GetAtomicNum() != 1
                and np.isfinite(_as_point(conformer, int(neighbour.GetIdx()))).all()
            ]
            clearance = min(
                (float(np.linalg.norm(point - existing)) for existing in points),
                default=float("inf"),
            )
            safe = 1.0 if clearance >= _MIN_CLEARANCE_A else 0.0
            alignment = float(np.dot(direction, away)) if away is not None else 0.0
            # Safety first, then preserve the local outward frame, then use
            # geometric clearance and a stable ordinal tie-breaker.
            score = (safe, alignment, clearance, -float(ordinal))
            if best is None or score > best[0]:
                best = (score, point)
        if best is None or best[0][0] < 1.0:
            raise ValueError(
                f"cannot place hydrogen {hydrogen_index}: degenerate local heavy frame"
            )
        _, point = best
        conformer.SetAtomPosition(int(hydrogen_index), tuple(float(v) for v in point))
        occupied_points.append(point)
        direction = _unit(point - parent_point)
        if direction is not None:
            occupied_directions.append(direction)


def repair_added_hydrogen_geometry(molecule: Chem.Mol, heavy_count: int) -> Chem.Mol:
    """Repair invalid coordinates for ``Chem.AddHs`` hydrogens in place.

    ``heavy_count`` is the number of heavy atoms at the front of the molecule,
    as produced by ``Chem.AddHs``.  Only atoms at indices ``>= heavy_count``
    with element H are candidates.  A candidate is repaired when its coordinate
    is non-finite or its bond distance to its unique heavy parent is outside
    ``[0.4, 2.0]`` Angstrom.  The new distance is the sum of RDKit's covalent
    radii for H and the parent element.  Local heavy-neighbour and already-valid
    H directions choose a deterministic placement; no graph, atom order,
    charge, or heavy-atom coordinate is changed.

    The original ``Chem.Mol`` is returned after in-place conformer updates.
    A missing conformer, malformed added-H parent relationship, non-finite
    parent coordinate, or degenerate local frame raises ``ValueError``.
    """
    if not isinstance(molecule, Chem.Mol):
        raise TypeError("molecule must be an RDKit Chem.Mol")
    if isinstance(heavy_count, bool) or not isinstance(heavy_count, int):
        raise TypeError("heavy_count must be an integer")
    atom_count = molecule.GetNumAtoms()
    if heavy_count < 0 or heavy_count > atom_count:
        raise ValueError("heavy_count is outside the molecule atom range")
    if molecule.GetNumConformers() == 0:
        raise ValueError("private hydrogen geometry requires a conformer")
    conformer = molecule.GetConformer()

    for index in range(heavy_count):
        if molecule.GetAtomWithIdx(index).GetAtomicNum() == 1:
            raise ValueError("heavy_count does not split heavy atoms from added H")

    by_parent: dict[int, list[int]] = {}
    valid_by_parent: dict[int, list[int]] = {}
    for index in range(heavy_count, atom_count):
        atom = molecule.GetAtomWithIdx(index)
        if atom.GetAtomicNum() != 1:
            raise ValueError("atoms after heavy_count must be added hydrogens")
        neighbours = list(atom.GetNeighbors())
        parents = [neighbour for neighbour in neighbours if neighbour.GetAtomicNum() != 1]
        if len(neighbours) != 1 or len(parents) != 1 or int(parents[0].GetIdx()) >= heavy_count:
            raise ValueError(
                f"added hydrogen {index} is not singly attached to one heavy parent"
            )
        parent_index = int(parents[0].GetIdx())
        parent_point = _as_point(conformer, parent_index)
        hydrogen_point = _as_point(conformer, index)
        distance = float(np.linalg.norm(hydrogen_point - parent_point))
        valid = (
            np.isfinite(hydrogen_point).all()
            and np.isfinite(parent_point).all()
            and math.isfinite(distance)
            and _MIN_BOND_DISTANCE_A <= distance <= _MAX_BOND_DISTANCE_A
        )
        if valid:
            valid_by_parent.setdefault(parent_index, []).append(index)
        else:
            by_parent.setdefault(parent_index, []).append(index)

    for parent_index in sorted(by_parent):
        parent = molecule.GetAtomWithIdx(parent_index)
        _repair_for_parent(
            molecule,
            conformer,
            parent,
            by_parent[parent_index],
            valid_by_parent.get(parent_index, []),
        )
    return molecule


__all__ = ["repair_added_hydrogen_geometry"]
