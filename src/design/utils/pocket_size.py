"""Offline pocket-size geometry and ligand atom-count sampling helpers.

The atom-count distribution is copied from the pinned TargetDiff
utils/evaluation/atom_num_config.py CONFIG literal.  The source is recorded
in the bundled JSON asset and in this module for auditability; this helper does
not claim that the historical distribution is scientifically calibrated for
every standalone inference use.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Final

import numpy as np

from partidesign.paths import resource_root


__all__ = ["pocket_space_size", "sample_ligand_atom_count"]

_ASSET_PATH: Final[Path] = (
    resource_root()
    / "configs"
    / "resources"
    / "ligand"
    / "ligand_atom_count_distribution.json"
)
_SOURCE_URL: Final[str] = (
    "https://raw.githubusercontent.com/guanjq/targetdiff/"
    "142f1eb7178480d435fe0b8cb95a99beb48997c7/"
    "utils/evaluation/atom_num_config.py"
)
_SOURCE_COMMIT: Final[str] = "142f1eb7178480d435fe0b8cb95a99beb48997c7"
_PAIR_BLOCK_SIZE: int = 512
_TOP_K: Final[int] = 10


def _load_distribution() -> tuple[np.ndarray, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    """Load and validate the offline atom-count distribution asset."""

    try:
        payload = json.loads(_ASSET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load ligand atom-count asset {_ASSET_PATH}") from exc

    if not isinstance(payload, Mapping):
        raise RuntimeError("ligand atom-count asset must contain a JSON object")
    if payload.get("schema_version") != 1 or payload.get("version") != 1:
        raise RuntimeError("unsupported ligand atom-count asset version")

    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise RuntimeError("ligand atom-count asset is missing source provenance")
    if source.get("url") != _SOURCE_URL or source.get("raw_url") != _SOURCE_URL:
        raise RuntimeError("ligand atom-count asset source URL is not pinned")
    if source.get("commit") != _SOURCE_COMMIT:
        raise RuntimeError("ligand atom-count asset source commit is not pinned")

    raw_bounds = payload.get("bounds")
    if not isinstance(raw_bounds, list) or not raw_bounds:
        raise RuntimeError("ligand atom-count bounds must be a non-empty list")
    try:
        bounds = np.asarray(raw_bounds, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ligand atom-count bounds must be numeric") from exc
    if bounds.ndim != 1 or not np.isfinite(bounds).all():
        raise RuntimeError("ligand atom-count bounds must be finite and one-dimensional")
    if not np.all(bounds[1:] > bounds[:-1]):
        raise RuntimeError("ligand atom-count bounds must be strictly increasing")

    raw_bins = payload.get("bins")
    if not isinstance(raw_bins, list) or len(raw_bins) != len(bounds) + 1:
        raise RuntimeError("ligand atom-count bins must be one longer than bounds")

    bins: list[tuple[np.ndarray, np.ndarray]] = []
    for index, raw_bin in enumerate(raw_bins):
        if not isinstance(raw_bin, Mapping):
            raise RuntimeError(f"ligand atom-count bin {index} must be an object")
        raw_counts = raw_bin.get("counts")
        raw_probabilities = raw_bin.get("probabilities")
        if not isinstance(raw_counts, list) or not isinstance(raw_probabilities, list):
            raise RuntimeError(f"ligand atom-count bin {index} has invalid support fields")
        if not raw_counts or len(raw_counts) != len(raw_probabilities):
            raise RuntimeError(f"ligand atom-count bin {index} support/probability lengths differ")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0
            for value in raw_counts
        ):
            raise RuntimeError(f"ligand atom-count bin {index} has invalid atom counts")
        if len(set(int(value) for value in raw_counts)) != len(raw_counts):
            raise RuntimeError(f"ligand atom-count bin {index} has duplicate atom counts")
        try:
            counts = np.asarray(raw_counts, dtype=np.int64)
            probabilities = np.asarray(raw_probabilities, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"ligand atom-count bin {index} has non-numeric values") from exc
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
            raise RuntimeError(f"ligand atom-count bin {index} probabilities must be finite and non-negative")
        if not np.isclose(
            float(probabilities.sum(dtype=np.float64)),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"ligand atom-count bin {index} probabilities must sum to one")
        bins.append((counts, probabilities))

    return bounds, tuple(bins)


_BOUNDS, _BINS = _load_distribution()


def _coerce_space_size(space_size: object) -> float:
    """Validate one finite, non-negative scalar space size."""

    if isinstance(space_size, (bool, np.bool_)):
        raise TypeError("space_size must be a real numeric scalar")
    value = np.asarray(space_size)
    if value.ndim != 0 or value.dtype.kind not in "iuf":
        raise TypeError("space_size must be a real numeric scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("space_size must be finite")
    if result < 0.0:
        raise ValueError("space_size must be non-negative")
    return result


def _retain_largest(values: np.ndarray, retained: np.ndarray) -> np.ndarray:
    """Keep at most the largest _TOP_K values from two one-dimensional arrays."""

    if values.size == 0:
        return retained
    take = min(_TOP_K, values.size)
    if values.size > take:
        values = np.partition(values, values.size - take)[-take:]
    merged = np.concatenate((retained, values))
    if merged.size > _TOP_K:
        merged = np.partition(merged, merged.size - _TOP_K)[-_TOP_K:]
    return merged


def pocket_space_size(coords_angstrom: object) -> float:
    """Return the median of the ten largest pairwise Euclidean distances.

    Fewer than ten pairs contribute all available pairs.  Pair distances are
    evaluated one block at a time and only the largest ten values are retained,
    so the implementation does not allocate an N x N distance matrix for the
    complete receptor.
    """

    try:
        coords = np.asarray(coords_angstrom, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("coords_angstrom must be a finite Nx3 array") from exc
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 2:
        raise ValueError("coords_angstrom must have shape (N, 3) with N >= 2")
    if not np.isfinite(coords).all():
        raise ValueError("coords_angstrom must contain only finite values")

    retained = np.empty(0, dtype=np.float64)
    block_size = int(_PAIR_BLOCK_SIZE)
    if block_size <= 0:
        raise RuntimeError("pair-distance block size must be positive")

    point_count = coords.shape[0]
    for start in range(0, point_count, block_size):
        stop = min(start + block_size, point_count)
        left = coords[start:stop]
        for other_start in range(start, point_count, block_size):
            other_stop = min(other_start + block_size, point_count)
            right = coords[other_start:other_stop]
            with np.errstate(over="ignore", invalid="ignore"):
                delta = left[:, None, :] - right[None, :, :]
                distances = np.sqrt(
                    np.einsum("ijk,ijk->ij", delta, delta, dtype=np.float64)
                )
            if not np.isfinite(distances).all():
                raise ValueError("pairwise distances are not finite")
            if start == other_start:
                if left.shape[0] < 2:
                    continue
                distances = distances[np.triu_indices(left.shape[0], k=1)]
            else:
                distances = distances.reshape(-1)
            retained = _retain_largest(distances, retained)

    if retained.size == 0:
        raise RuntimeError("no pairwise distances were computed")
    result = float(np.median(retained))
    if not np.isfinite(result):
        raise RuntimeError("pocket space size is not finite")
    return result


def sample_ligand_atom_count(
    space_size: object,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample (heavy_atom_count, bin_index) for a non-negative pocket size.

    Bin selection follows TargetDiff's strict convention: a bound belongs to
    the bin above it (the first bound strictly greater than space_size is
    selected).  Sampling uses only the supplied numpy.random.Generator.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    value = _coerce_space_size(space_size)
    bin_index = int(np.searchsorted(_BOUNDS, value, side="right"))
    counts, probabilities = _BINS[bin_index]
    atom_count = int(rng.choice(counts, p=probabilities))
    return atom_count, bin_index
