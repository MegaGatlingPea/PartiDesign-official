"""Pure CPU ipSAE d0res scores for an ordered pair of PAE blocks.

Adapted from DunbrackLab/IPSAE ``ipsae.py``, version 4 (January 3, 2026):
https://github.com/DunbrackLab/IPSAE/blob/main/ipsae.py
Copyright (c) 2025 Lab of Dr. Roland Dunbrack. MIT License; see
``licenses/IPSAE_LICENSE.txt``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

IPSAE_METHODS = frozenset({"esmfold2-fast", "esmfold2-standard", "af3", "af3-la", "mol1"})
IPSAE_METRIC_NAMES = (
    "ipsae", "ipsae_design_to_context", "ipsae_context_to_design",
    "ipsae_pae_cutoff", "ipsae_pairs",
)


def _pae_array(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a two-dimensional real numeric array")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative PAE values")
    return array


def _positive_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or array.dtype.kind not in "iuf"
        or not np.isfinite(array)
        or array <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")
    return float(array)


def _directed_score(pae: np.ndarray, pae_cutoff: float, d0_min: float) -> float:
    selected = pae < pae_cutoff
    counts = selected.sum(axis=1)
    # Version 4 d0res uses calc_d0_array, including its n=27 behavior.
    lengths = np.maximum(counts, 26).astype(np.float64)
    d0 = np.maximum(d0_min, 1.24 * (lengths - 15) ** (1.0 / 3.0) - 1.8)
    weights = np.zeros_like(pae)
    np.divide(pae, d0[:, None], out=weights, where=selected)
    with np.errstate(over="ignore"):
        np.square(weights, out=weights)
    weights += 1.0
    np.reciprocal(weights, out=weights)
    row_sums = np.sum(weights, axis=1, where=selected)
    row_scores = np.zeros(pae.shape[0], dtype=np.float64)
    np.divide(row_sums, counts, out=row_scores, where=counts > 0)
    return float(row_scores.max(initial=0.0))


def ipsae_pair(
    pae_forward: ArrayLike,
    pae_reverse: ArrayLike,
    *,
    pae_cutoff: float,
    d0_min: float = 1.0,
) -> dict[str, float]:
    """Return forward, reverse, and maximum-direction ipSAE d0res scores.

    ``pae_forward`` has shape ``(N, M)`` and ``pae_reverse`` has shape
    ``(M, N)``. They contain genuinely directional PAE values in angstroms;
    the reverse block is not inferred by transposing the forward block.
    Only PAE values strictly below ``pae_cutoff`` contribute. Rows with no
    selected partners, and pairs with either empty partner, score zero.

    Use ``d0_min=1.0`` for protein pairs and ``d0_min=2.0`` when either
    partner is nucleic acid. Both scalar parameters must be finite and positive.
    The calculation uses float64 and does not mutate the input arrays.
    """
    cutoff = _positive_scalar(pae_cutoff, "pae_cutoff")
    minimum = _positive_scalar(d0_min, "d0_min")
    forward_pae = _pae_array(pae_forward, "pae_forward")
    reverse_pae = _pae_array(pae_reverse, "pae_reverse")
    if reverse_pae.shape != forward_pae.shape[::-1]:
        raise ValueError("pae_reverse must have the transposed shape of pae_forward")
    forward = _directed_score(forward_pae, cutoff, minimum)
    reverse = _directed_score(reverse_pae, cutoff, minimum)
    return {"forward": forward, "reverse": reverse, "symmetric": max(forward, reverse)}
