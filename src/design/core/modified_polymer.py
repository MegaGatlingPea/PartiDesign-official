"""Coordinate planning for modified polymer context residues.

This module deliberately contains only standalone inference dependencies.  A
source residue can have atoms that are not representable by the standard
proxy residue decoded by the model.  Representable atoms use the generated
proxy coordinates directly; the remaining source atoms are recovered in the
source frame with a deterministic, proper-rotation Kabsch fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np

from ..utils.atoms import (
    DNA_ATOM27,
    DNA_RES_ATOMS,
    PROTEIN_ATOM37,
    PROTEIN_RES_ATOMS,
    RNA_ATOM27,
    RNA_RES_ATOMS,
)


# Internal acceptance threshold for source-frame proxy alignment, in Angstrom.
KABSCH_RMSD_TOLERANCE_ANGSTROM: Final[float] = 0.75

_INVENTORIES: Final[dict[str, Mapping[str, tuple[str, ...]]]] = {
    "protein": PROTEIN_RES_ATOMS,
    "rna": RNA_RES_ATOMS,
    "dna": DNA_RES_ATOMS,
}
_ATOM_ORDERS: Final[dict[str, tuple[str, ...]]] = {
    "protein": PROTEIN_ATOM37,
    "rna": RNA_ATOM27,
    "dna": DNA_ATOM27,
}
_UNKNOWN_RESIDUES: Final[dict[str, str]] = {
    "protein": "UNK",
    "rna": "N",
    "dna": "DN",
}
_PHOSPHATE_ALIASES: Final[dict[str, str]] = {
    "O1P": "OP1",
    "O2P": "OP2",
    "O3P": "OP3",
}


def _normalise_modality(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("modality must be a string")
    modality = value.strip().lower()
    if modality not in _INVENTORIES:
        raise ValueError(f"unsupported polymer modality {value!r}")
    return modality


def _normalise_residue_name(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    residue_name = value.strip().upper()
    if not residue_name:
        raise ValueError(f"{field} must be non-empty")
    return residue_name


def _normalise_atom_name(
    value: str,
    *,
    modality: str,
    residue_name: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError("source atom names must be strings")
    atom_name = value.strip().upper()
    if not atom_name:
        raise ValueError("source atom names must be non-empty")
    if modality in {"rna", "dna"}:
        atom_name = atom_name.replace("*", "'")
    atom_name = _PHOSPHATE_ALIASES.get(atom_name, atom_name)
    # CCD commonly spells these selenium atoms as ``SE`` while the standard
    # parent uses the sulfur slot of the corresponding residue.
    if residue_name == "MSE" and atom_name == "SE":
        atom_name = "SD"
    elif residue_name == "SEC" and atom_name == "SE":
        atom_name = "SG"
    return atom_name


def _literal_atom_name(value: str) -> str:
    """Normalize spelling without applying a source-to-proxy alias."""

    if not isinstance(value, str):
        raise ValueError("source atom names must be strings")
    atom_name = value.strip().upper()
    if not atom_name:
        raise ValueError("source atom names must be non-empty")
    return atom_name


def _common_inventory(modality: str) -> tuple[str, ...]:
    inventories = _INVENTORIES[modality]
    common = set.intersection(*(set(names) for names in inventories.values()))
    # Preserve the model's standalone atom order rather than set iteration.
    return tuple(atom for atom in _ATOM_ORDERS[modality] if atom in common)


def proxy_inventory(modality: str, proxy_residue_name: str) -> tuple[str, ...]:
    """Return the ordered atom inventory used by a standard proxy residue.

    A known standard parent uses its residue-specific inventory.  ``UNK`` for
    protein, ``N`` for RNA, and ``DN`` for DNA use the atoms common to every
    standard residue of that modality, which is the identity-safe inventory.
    """

    normalised_modality = _normalise_modality(modality)
    residue_name = _normalise_residue_name(
        proxy_residue_name, field="proxy_residue_name"
    )
    if residue_name == _UNKNOWN_RESIDUES[normalised_modality]:
        return _common_inventory(normalised_modality)
    try:
        return tuple(_INVENTORIES[normalised_modality][residue_name])
    except KeyError as exc:
        raise ValueError(
            f"unsupported {normalised_modality} proxy residue {proxy_residue_name!r}"
        ) from exc


def proxy_atom_name(
    *,
    modality: str,
    residue_name: str,
    proxy_residue_name: str,
    source_atom_name: str,
) -> str | None:
    """Alias one source atom into the proxy inventory, or return ``None``.

    The residue-specific aliases are applied before checking the selected
    proxy inventory.  Thus unsupported source atoms are represented by their
    normalized name by the planner but are not retained as proxy atoms.
    """

    normalised_modality = _normalise_modality(modality)
    source_residue_name = _normalise_residue_name(residue_name, field="residue_name")
    inventory = proxy_inventory(normalised_modality, proxy_residue_name)
    atom_name = _normalise_atom_name(
        source_atom_name,
        modality=normalised_modality,
        residue_name=source_residue_name,
    )
    if atom_name not in inventory:
        return None
    return atom_name


class ModifiedPolymerRecoveryWarning(UserWarning):
    """Warning payload for a modified-polymer coordinate fallback.

    The planner is intentionally pure and has no chain/residue context from
    which to emit this warning.  Callers can construct this warning from the
    returned ``fallback_reason`` while retaining structured location fields.
    """

    def __init__(
        self,
        chain_id: str,
        res_id: int,
        res_name: str,
        reason: str,
    ) -> None:
        self.chain_id = chain_id
        self.res_id = res_id
        self.res_name = res_name
        self.reason = reason
        super().__init__(
            f"modified polymer {res_name} fallback: {reason} "
            f"(chain_id={chain_id!r}, res_id={res_id!r})"
        )


def _coordinate_copy(value: object, *, field: str) -> np.ndarray:
    try:
        coordinate = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a numeric coordinate") from exc
    if coordinate.shape != (3,):
        raise ValueError(f"{field} must have shape [3]")
    coordinate = np.array(coordinate, dtype=np.float32, copy=True)
    if not np.isfinite(coordinate).all():
        raise ValueError(f"{field} must be finite")
    return coordinate


def _freeze_name_mapping(value: Mapping[int, str]) -> Mapping[int, str]:
    result: dict[int, str] = {}
    for raw_row, raw_name in value.items():
        if isinstance(raw_row, bool) or not isinstance(raw_row, (int, np.integer)):
            raise ValueError("coordinate-plan row keys must be integer source rows")
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("coordinate-plan proxy atom names must be non-empty strings")
        result[int(raw_row)] = raw_name
    return MappingProxyType(result)


def _freeze_coordinate_mapping(
    value: Mapping[int, object],
    *,
    field: str,
) -> Mapping[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for raw_row, raw_coordinate in value.items():
        if isinstance(raw_row, bool) or not isinstance(raw_row, (int, np.integer)):
            raise ValueError("coordinate-plan row keys must be integer source rows")
        result[int(raw_row)] = _coordinate_copy(
            raw_coordinate,
            field=f"{field}[{int(raw_row)}]",
        )
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class ModifiedPolymerCoordinatePlan:
    """Source-row coordinate plan for one modified polymer residue."""

    proxy_atom_name_by_row: Mapping[int, str]
    common_coordinates: Mapping[int, np.ndarray]
    extra_coordinates: Mapping[int, np.ndarray]
    alignment_rmsd: float | None = None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proxy_atom_name_by_row",
            _freeze_name_mapping(self.proxy_atom_name_by_row),
        )
        object.__setattr__(
            self,
            "common_coordinates",
            _freeze_coordinate_mapping(
                self.common_coordinates,
                field="common_coordinates",
            ),
        )
        object.__setattr__(
            self,
            "extra_coordinates",
            _freeze_coordinate_mapping(
                self.extra_coordinates,
                field="extra_coordinates",
            ),
        )
        if self.alignment_rmsd is not None:
            try:
                alignment_rmsd = float(self.alignment_rmsd)
            except (TypeError, ValueError) as exc:
                raise ValueError("alignment_rmsd must be finite or None") from exc
            if not np.isfinite(alignment_rmsd):
                raise ValueError("alignment_rmsd must be finite or None")
            object.__setattr__(self, "alignment_rmsd", alignment_rmsd)
        if self.fallback_reason is not None and not isinstance(
            self.fallback_reason, str
        ):
            raise ValueError("fallback_reason must be a string or None")


def _source_name_sequence(source_atom_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(source_atom_names, (str, bytes)):
        raise ValueError("source_atom_names must be a one-dimensional sequence")
    try:
        values = tuple(source_atom_names)
    except TypeError as exc:
        raise ValueError("source_atom_names must be a one-dimensional sequence") from exc
    if any(isinstance(value, (list, tuple, np.ndarray)) for value in values):
        raise ValueError("source_atom_names must be a one-dimensional sequence")
    return values


def _resolve_source_atom_names(
    *,
    modality: str,
    residue_name: str,
    raw_names: Sequence[str],
) -> tuple[list[str], set[int]]:
    source_names: list[str] = []
    literal_names: list[str] = []
    rows_by_proxy_name: dict[str, list[int]] = {}
    for row, raw_name in enumerate(raw_names):
        name = _normalise_atom_name(
            raw_name,
            modality=modality,
            residue_name=residue_name,
        )
        source_names.append(name)
        literal_names.append(_literal_atom_name(raw_name))
        rows_by_proxy_name.setdefault(name, []).append(row)

    winner_rows: set[int] = set()
    for name, rows in rows_by_proxy_name.items():
        if len(rows) == 1:
            winner_rows.add(rows[0])
            continue
        literal_rows = [row for row in rows if literal_names[row] == name]
        if len(literal_rows) != 1:
            raise ValueError(
                f"source atom names alias to ambiguous proxy slot {name!r}"
            )
        winner_rows.add(literal_rows[0])
    return source_names, winner_rows


def proxy_atom_rows(
    *,
    modality: str,
    residue_name: str,
    proxy_residue_name: str,
    source_atom_names: Sequence[str],
) -> Mapping[int, str]:
    """Return winning source-row to proxy-name mappings after alias resolution."""

    normalised_modality = _normalise_modality(modality)
    source_residue_name = _normalise_residue_name(residue_name, field="residue_name")
    inventory = set(proxy_inventory(normalised_modality, proxy_residue_name))
    raw_names = _source_name_sequence(source_atom_names)
    source_names, winner_rows = _resolve_source_atom_names(
        modality=normalised_modality,
        residue_name=source_residue_name,
        raw_names=raw_names,
    )
    return MappingProxyType(
        {
            row: name
            for row, name in enumerate(source_names)
            if row in winner_rows and name in inventory
        }
    )


def _source_coordinates_array(source_coordinates: object, count: int) -> np.ndarray:
    try:
        coordinates = np.asarray(source_coordinates, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_coordinates must be numeric [N,3]") from exc
    if coordinates.shape != (count, 3):
        raise ValueError("source_coordinates must have shape [N,3]")
    return coordinates


def _source_valid_array(source_coord_valid: object, count: int) -> np.ndarray:
    try:
        valid = np.asarray(source_coord_valid)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_coord_valid must be one-dimensional [N]") from exc
    if valid.shape != (count,):
        raise ValueError("source_coord_valid must have shape [N]")
    try:
        return valid.astype(bool, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_coord_valid must be boolean-compatible") from exc


def _source_preserve_array(
    preserve_source_coord_mask: object | None,
    count: int,
) -> np.ndarray:
    if preserve_source_coord_mask is None:
        return np.zeros(count, dtype=bool)
    try:
        preserve = np.asarray(preserve_source_coord_mask)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "preserve_source_coord_mask must be one-dimensional [N]"
        ) from exc
    if preserve.shape != (count,):
        raise ValueError("preserve_source_coord_mask must have shape [N]")
    try:
        return preserve.astype(bool, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "preserve_source_coord_mask must be boolean-compatible"
        ) from exc


def _generated_coordinates_by_name(
    generated_proxy_coordinates: object,
    inventory: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if isinstance(generated_proxy_coordinates, Mapping):
        normalised: dict[str, object] = {}
        for raw_name, coordinate in generated_proxy_coordinates.items():
            if not isinstance(raw_name, str):
                raise ValueError("generated proxy-coordinate mapping keys must be strings")
            name = raw_name.strip().upper().replace("*", "'")
            name = _PHOSPHATE_ALIASES.get(name, name)
            if name in normalised:
                raise ValueError("generated proxy-coordinate mapping has duplicate atom names")
            normalised[name] = coordinate
        missing = [name for name in inventory if name not in normalised]
        if missing:
            raise ValueError(
                "generated proxy coordinates are missing inventory atoms: "
                + ", ".join(missing)
            )
        return {
            name: _coordinate_copy(
                normalised[name],
                field=f"generated_proxy_coordinates[{name!r}]",
            )
            for name in inventory
        }

    try:
        coordinates = np.asarray(generated_proxy_coordinates, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("generated_proxy_coordinates must be numeric [M,3]") from exc
    if coordinates.shape != (len(inventory), 3):
        raise ValueError(
            "generated_proxy_coordinates must have shape "
            f"[{len(inventory)},3] for the selected proxy inventory"
        )
    result: dict[str, np.ndarray] = {}
    for index, name in enumerate(inventory):
        result[name] = _coordinate_copy(
            coordinates[index],
            field=f"generated_proxy_coordinates[{index}]",
        )
    return result


def _priority_groups(modality: str) -> tuple[tuple[str, ...], ...]:
    if modality == "protein":
        return (
            ("N", "CA", "C"),
            ("N", "CA", "C", "O"),
        )
    return (
        ("C1'", "C2'", "C3'", "C4'", "O4'"),
        ("C1'", "N9", "C4", "C8"),
        ("C1'", "N1", "C2", "C6"),
        ("P", "O5'", "C5'", "C4'"),
    )


def _rank_two(coordinates: np.ndarray) -> bool:
    if coordinates.shape[0] < 3 or not np.isfinite(coordinates).all():
        return False
    centered = coordinates - coordinates.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if not np.isfinite(singular_values).all() or singular_values[0] <= 0.0:
        return False
    return bool(singular_values[1] > singular_values[0] * 1.0e-8)


def _kabsch_row_vector(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        return None
    if source.shape[0] < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        return None
    if not _rank_two(source) or not _rank_two(target):
        return None
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = source_centered.T @ target_centered
    try:
        left, _, right_transpose = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(left).all() or not np.isfinite(right_transpose).all():
        return None
    unconstrained = left @ right_transpose
    determinant = float(np.linalg.det(unconstrained))
    if not np.isfinite(determinant):
        return None
    correction = np.eye(3, dtype=np.float64)
    if determinant < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_transpose
    rotation_determinant = float(np.linalg.det(rotation))
    if not np.isfinite(rotation_determinant) or rotation_determinant <= 0.0:
        return None
    translation = target_centroid - source_centroid @ rotation
    transformed = source @ rotation + translation
    residual = transformed - target
    if not np.isfinite(transformed).all() or not np.isfinite(residual).all():
        return None
    rmsd = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    if not np.isfinite(rmsd):
        return None
    return rotation, translation, rmsd


def _fallback_plan(
    proxy_names: Mapping[int, str],
    common_coordinates: Mapping[int, np.ndarray],
    *,
    reason: str,
    alignment_rmsd: float | None = None,
    preserved_extra_coordinates: Mapping[int, np.ndarray] | None = None,
) -> ModifiedPolymerCoordinatePlan:
    return ModifiedPolymerCoordinatePlan(
        proxy_atom_name_by_row=proxy_names,
        common_coordinates=common_coordinates,
        extra_coordinates=(
            {} if preserved_extra_coordinates is None else preserved_extra_coordinates
        ),
        alignment_rmsd=alignment_rmsd,
        fallback_reason=reason,
    )


def plan_modified_polymer_coordinates(
    *,
    modality: str,
    residue_name: str,
    proxy_residue_name: str,
    source_atom_names: Sequence[str],
    source_coordinates: object,
    source_coord_valid: object,
    generated_proxy_coordinates: object,
    preserve_source_coord_mask: object | None = None,
) -> ModifiedPolymerCoordinatePlan:
    """Plan generated/common and recovered/extra coordinates for one residue.

    ``generated_proxy_coordinates`` is indexed by :func:`proxy_inventory`.
    A name-keyed mapping is also accepted for callers that already carry atom
    labels.  Source rows are never reordered: every returned mapping uses the
    original source row index as its key.
    """

    normalised_modality = _normalise_modality(modality)
    source_residue_name = _normalise_residue_name(residue_name, field="residue_name")
    normalised_proxy_residue = _normalise_residue_name(
        proxy_residue_name,
        field="proxy_residue_name",
    )
    inventory = proxy_inventory(normalised_modality, normalised_proxy_residue)
    generated_by_name = _generated_coordinates_by_name(
        generated_proxy_coordinates,
        inventory,
    )

    raw_names = _source_name_sequence(source_atom_names)
    source_names, winner_rows = _resolve_source_atom_names(
        modality=normalised_modality,
        residue_name=source_residue_name,
        raw_names=raw_names,
    )

    count = len(source_names)
    source = _source_coordinates_array(source_coordinates, count)
    valid = _source_valid_array(source_coord_valid, count)
    preserve_source = _source_preserve_array(preserve_source_coord_mask, count)
    inventory_set = set(inventory)

    proxy_names = {index: name for index, name in enumerate(source_names)}
    common_coordinates = {
        index: np.array(generated_by_name[name], dtype=np.float32, copy=True)
        for index, name in enumerate(source_names)
        if name in inventory_set and index in winner_rows
    }
    extra_rows = [
        index
        for index, name in enumerate(source_names)
        if name not in inventory_set and index in winner_rows
    ]
    preserved_extra_rows = [index for index in extra_rows if bool(preserve_source[index])]
    recoverable_extra_rows = [
        index for index in extra_rows if not bool(preserve_source[index])
    ]
    if any(not bool(valid[index]) for index in preserved_extra_rows):
        raise ValueError("preserved source extra coordinates must be valid")
    if any(not bool(np.isfinite(source[index]).all()) for index in preserved_extra_rows):
        raise ValueError("preserved source extra coordinates must be finite")
    preserved_extra_coordinates = {
        index: np.asarray(source[index], dtype=np.float32)
        for index in preserved_extra_rows
    }
    if not extra_rows:
        # With no unsupported source row, no alignment is needed and no
        # fallback is reported; common rows still use exact generated proxy
        # coordinates.
        return ModifiedPolymerCoordinatePlan(
            proxy_atom_name_by_row=proxy_names,
            common_coordinates=common_coordinates,
            extra_coordinates={},
            alignment_rmsd=None,
            fallback_reason=None,
        )

    # Hard source extras are output obligations and remain exact.  Other extras
    # are recoverable only through Kabsch; an invalid coordinate or unusable
    # frame therefore falls back to the intersection plus any hard extras.
    if not recoverable_extra_rows:
        return ModifiedPolymerCoordinatePlan(
            proxy_atom_name_by_row=proxy_names,
            common_coordinates=common_coordinates,
            extra_coordinates=preserved_extra_coordinates,
            alignment_rmsd=None,
            fallback_reason=None,
        )

    if any(not bool(valid[index]) for index in recoverable_extra_rows):
        return _fallback_plan(
            proxy_names,
            common_coordinates,
            reason="unrecoverable source extra coordinates",
            preserved_extra_coordinates=preserved_extra_coordinates,
        )
    if any(
        not bool(np.isfinite(source[index]).all())
        for index in recoverable_extra_rows
    ):
        return _fallback_plan(
            proxy_names,
            common_coordinates,
            reason="nonfinite source extra coordinates",
            preserved_extra_coordinates=preserved_extra_coordinates,
        )

    valid_common_rows = [
        index
        for index in common_coordinates
        if bool(valid[index])
    ]
    if any(not bool(np.isfinite(source[index]).all()) for index in valid_common_rows):
        return _fallback_plan(
            proxy_names,
            common_coordinates,
            reason="nonfinite source common coordinates",
            preserved_extra_coordinates=preserved_extra_coordinates,
        )
    finite_common_rows = [
        index
        for index in valid_common_rows
        if bool(np.isfinite(source[index]).all())
    ]
    row_by_name = {
        name: index
        for index, name in enumerate(source_names)
        if name in inventory_set and index in finite_common_rows
    }
    if len(finite_common_rows) < 3:
        return _fallback_plan(
            proxy_names,
            common_coordinates,
            reason="insufficient finite common correspondences",
            preserved_extra_coordinates=preserved_extra_coordinates,
        )

    selected_rows: list[int] | None = None
    for group in (*_priority_groups(normalised_modality), inventory):
        candidate = [row_by_name[name] for name in group if name in row_by_name]
        if len(candidate) < 3:
            continue
        candidate_source = source[candidate]
        candidate_target = np.asarray(
            [generated_by_name[source_names[index]] for index in candidate],
            dtype=np.float64,
        )
        if _rank_two(candidate_source) and _rank_two(candidate_target):
            selected_rows = candidate
            break
    if selected_rows is None:
        return _fallback_plan(
            proxy_names,
            common_coordinates,
            reason="degenerate common correspondences",
            preserved_extra_coordinates=preserved_extra_coordinates,
        )

    # Fit the selected priority frame itself.  Additional finite common atoms
    # are evaluation-only points: they contribute to the RMSD gate but cannot
    # dilute the deterministic backbone/sugar/base frame used for recovery.
    fit_source = source[selected_rows]
    fit_target = np.asarray(
        [generated_by_name[source_names[index]] for index in selected_rows],
        dtype=np.float64,
    )
    fit = _kabsch_row_vector(fit_source, fit_target)
    if fit is None:
        return _fallback_plan(
            proxy_names,
            common_coordinates,
            reason="degenerate common correspondences",
            preserved_extra_coordinates=preserved_extra_coordinates,
        )
    rotation, translation, _ = fit
    evaluation_rows = sorted(finite_common_rows)
    evaluation_source = source[evaluation_rows]
    evaluation_target = np.asarray(
        [generated_by_name[source_names[index]] for index in evaluation_rows],
        dtype=np.float64,
    )
    evaluation_residual = (
        evaluation_source @ rotation + translation
    ) - evaluation_target
    if not np.isfinite(evaluation_residual).all():
        return _fallback_plan(
            proxy_names,
            common_coordinates,
            reason="nonfinite alignment RMSD",
            preserved_extra_coordinates=preserved_extra_coordinates,
        )
    alignment_rmsd = float(
        np.sqrt(np.mean(np.sum(evaluation_residual * evaluation_residual, axis=1)))
    )
    if not np.isfinite(alignment_rmsd):
        return _fallback_plan(
            proxy_names,
            common_coordinates,
            reason="nonfinite alignment RMSD",
            preserved_extra_coordinates=preserved_extra_coordinates,
        )
    if alignment_rmsd > KABSCH_RMSD_TOLERANCE_ANGSTROM:
        return _fallback_plan(
            proxy_names,
            common_coordinates,
            reason="alignment RMSD exceeds tolerance",
            alignment_rmsd=alignment_rmsd,
            preserved_extra_coordinates=preserved_extra_coordinates,
        )

    extra_source = source[recoverable_extra_rows]
    extra_values = extra_source @ rotation + translation
    try:
        extra_values_float32 = np.asarray(extra_values, dtype=np.float32)
    except (TypeError, ValueError, FloatingPointError):
        return _fallback_plan(
            proxy_names,
            common_coordinates,
            reason="nonfinite recovered extra coordinates",
            alignment_rmsd=alignment_rmsd,
            preserved_extra_coordinates=preserved_extra_coordinates,
        )
    if not np.isfinite(extra_values_float32).all():
        return _fallback_plan(
            proxy_names,
            common_coordinates,
            reason="nonfinite recovered extra coordinates",
            alignment_rmsd=alignment_rmsd,
            preserved_extra_coordinates=preserved_extra_coordinates,
        )
    extra_coordinates = dict(preserved_extra_coordinates)
    extra_coordinates.update(
        {
            index: np.array(coordinate, dtype=np.float32, copy=True)
            for index, coordinate in zip(
                recoverable_extra_rows, extra_values_float32
            )
        }
    )
    return ModifiedPolymerCoordinatePlan(
        proxy_atom_name_by_row=proxy_names,
        common_coordinates=common_coordinates,
        extra_coordinates=extra_coordinates,
        alignment_rmsd=alignment_rmsd,
        fallback_reason=None,
    )


__all__ = [
    "KABSCH_RMSD_TOLERANCE_ANGSTROM",
    "ModifiedPolymerCoordinatePlan",
    "ModifiedPolymerRecoveryWarning",
    "plan_modified_polymer_coordinates",
    "proxy_atom_name",
    "proxy_atom_rows",
    "proxy_inventory",
]
