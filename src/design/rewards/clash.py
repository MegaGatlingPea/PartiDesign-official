"""Continuous pairwise van-der-Waals overlap reward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from biotite import structure as struc
from rdkit import Chem


@dataclass(frozen=True, slots=True)
class ClashScoreBatch:
    candidate_ids: tuple[str, ...]
    total: torch.Tensor
    design_design: torch.Tensor
    design_context: torch.Tensor

    def __post_init__(self) -> None:
        size = len(self.candidate_ids)
        for name in ("total", "design_design", "design_context"):
            value = getattr(self, name)
            if value.dtype != torch.float32 or value.shape != (size,):
                raise ValueError(f"{name} must be float32[{size}]")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} contains a non-finite score")


def _atom_array(candidate: object) -> struc.AtomArray:
    for name in ("atoms", "structure", "atom_array"):
        value = getattr(candidate, name, None)
        if isinstance(value, struc.AtomArray):
            return value
    raise TypeError("search candidate does not contain an AtomArray")


def _candidate_id(candidate: object) -> str:
    value = getattr(candidate, "candidate_id", None)
    if not isinstance(value, str) or not value:
        raise TypeError("search candidate needs a non-empty candidate_id")
    return value


def _condition_edge(condition: object) -> tuple[int, int]:
    if hasattr(condition, "atom_uid_a") and hasattr(condition, "atom_uid_b"):
        return int(condition.atom_uid_a), int(condition.atom_uid_b)
    endpoints = getattr(condition, "atom_uids", None)
    if endpoints is not None and len(endpoints) == 2:
        return int(endpoints[0]), int(endpoints[1])
    raise TypeError("topology condition does not expose its two atom UIDs")


def _normalized_element(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("empty element in finalized search candidate")
    return text[0].upper() + text[1:].lower()


def _radii(elements: np.ndarray) -> np.ndarray:
    table = Chem.GetPeriodicTable()
    radii = np.empty(len(elements), dtype=np.float64)
    for index, raw in enumerate(elements):
        element = _normalized_element(raw)
        try:
            atomic_number = int(table.GetAtomicNumber(element))
        except RuntimeError as exc:
            raise ValueError(f"unsupported heavy-atom element {element!r}") from exc
        if atomic_number <= 1:
            raise ValueError(f"unsupported heavy-atom element {element!r}")
        try:
            radius = float(table.GetRvdw(atomic_number))
        except RuntimeError as exc:
            raise ValueError(f"missing VDW radius for element {element!r}") from exc
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(f"missing VDW radius for element {element!r}")
        radii[index] = radius
    return radii


def validate_vdw_elements(elements: Sequence[object]) -> None:
    """Reject elements that cannot participate in the continuous clash reward."""

    _radii(np.asarray(tuple(elements), dtype=object))


def _exclusion_pairs(candidate: object, atom_array: struc.AtomArray) -> set[tuple[int, int]]:
    neighbours: list[set[int]] = [set() for _ in range(len(atom_array))]
    if atom_array.bonds is not None:
        for atom_i, atom_j, _ in atom_array.bonds.as_array():
            i, j = int(atom_i), int(atom_j)
            neighbours[i].add(j)
            neighbours[j].add(i)

    uid_to_row: dict[int, int] = {}
    if "atom_uid" in atom_array.get_annotation_categories():
        uid_to_row = {
            int(uid): row
            for row, uid in enumerate(np.asarray(atom_array.atom_uid).tolist())
        }
    for condition in getattr(candidate, "topology_conditions", ()):
        uid_i, uid_j = _condition_edge(condition)
        if uid_i not in uid_to_row or uid_j not in uid_to_row:
            raise ValueError("topology-condition endpoint is absent from candidate")
        i, j = uid_to_row[uid_i], uid_to_row[uid_j]
        neighbours[i].add(j)
        neighbours[j].add(i)

    excluded: set[tuple[int, int]] = set()
    for atom_i, direct in enumerate(neighbours):
        for atom_j in direct:
            excluded.add((min(atom_i, atom_j), max(atom_i, atom_j)))
        for middle in direct:
            for atom_j in neighbours[middle]:
                if atom_i != atom_j:
                    excluded.add((min(atom_i, atom_j), max(atom_i, atom_j)))
    return excluded


def _score_one(candidate: object) -> tuple[float, float]:
    atom_array = _atom_array(candidate)
    coordinates = np.asarray(atom_array.coord, dtype=np.float64)
    if coordinates.shape != (len(atom_array), 3) or not np.isfinite(coordinates).all():
        raise ValueError("clash reward requires finite finalized coordinates")
    if "is_design" not in atom_array.get_annotation_categories():
        raise ValueError("clash reward requires the finalized is_design annotation")
    is_design = np.asarray(atom_array.is_design, dtype=bool)
    if not bool(is_design.any()):
        raise ValueError("clash reward requires at least one design atom")
    radii = _radii(np.asarray(atom_array.element))
    excluded = _exclusion_pairs(candidate, atom_array)

    design_rows = np.flatnonzero(is_design)
    context_rows = np.flatnonzero(~is_design)
    dd_penalty = 0.0
    dc_penalty = 0.0
    for position, atom_i in enumerate(design_rows):
        for atom_j in design_rows[position + 1 :]:
            pair = (min(int(atom_i), int(atom_j)), max(int(atom_i), int(atom_j)))
            if pair in excluded:
                continue
            distance = float(np.linalg.norm(coordinates[atom_i] - coordinates[atom_j]))
            rho = distance / float(radii[atom_i] + radii[atom_j])
            dd_penalty += max(0.0, 1.0 - rho) ** 2
        for atom_j in context_rows:
            pair = (min(int(atom_i), int(atom_j)), max(int(atom_i), int(atom_j)))
            if pair in excluded:
                continue
            distance = float(np.linalg.norm(coordinates[atom_i] - coordinates[atom_j]))
            rho = distance / float(radii[atom_i] + radii[atom_j])
            dc_penalty += max(0.0, 1.0 - rho) ** 2
    return -dd_penalty, -dc_penalty


class ClashReward:
    """Rank terminal candidates by continuous DD and DC VDW overlap."""

    def __init__(
        self,
        design_design_weight: float = 1.0,
        design_context_weight: float = 1.0,
    ) -> None:
        self.design_design_weight = float(design_design_weight)
        self.design_context_weight = float(design_context_weight)
        values = (self.design_design_weight, self.design_context_weight)
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("clash weights must be finite and nonnegative")
        if not any(value > 0 for value in values):
            raise ValueError("at least one clash weight must be positive")

    @torch.inference_mode()
    def score_batch(
        self,
        candidates: Sequence[object],
        *,
        device: torch.device | str = "cpu",
    ) -> ClashScoreBatch:
        candidate_ids = tuple(_candidate_id(candidate) for candidate in candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique within one reward batch")
        values = [_score_one(candidate) for candidate in candidates]
        dd = torch.tensor([value[0] for value in values], dtype=torch.float32, device=device)
        dc = torch.tensor([value[1] for value in values], dtype=torch.float32, device=device)
        total = self.design_design_weight * dd + self.design_context_weight * dc
        return ClashScoreBatch(candidate_ids, total, dd, dc)


__all__ = ["ClashReward", "ClashScoreBatch", "validate_vdw_elements"]
