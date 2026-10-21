"""Opaque, row-bounded pair LayerNorm and bias projection.

The pair features are invariant across the FM attention stack, but their
``LayerNorm -> Linear`` projection can otherwise expose a Python loop over the
symbolic token dimension to ``torch.compile``.  This module keeps that loop in
one custom operator.  The operator owns no model state and is intended for
inference only; callers pass the already folded projection parameters for the
current layer group.
"""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import Tensor
from torch.nn import functional as F


_NAMESPACE: Final[str] = "partidesign"
_OP_NAME: Final[str] = f"{_NAMESPACE}::row_pair_bias"
_AUTOCAST_DTYPES: Final[frozenset[torch.dtype]] = frozenset(
    (torch.float16, torch.bfloat16)
)
_FLOAT_DTYPES: Final[frozenset[torch.dtype]] = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)


def _validate_inputs(
    pair: Tensor,
    folded_weight: Tensor,
    offset: Tensor,
    eps: float,
    projection_dtype: torch.dtype,
    requested_rows: int,
    norm_weight: Tensor | None,
    norm_bias: Tensor | None,
) -> tuple[int, int, int, int]:
    if pair.ndim != 4:
        raise ValueError("pair must have shape [B, N, N, D]")
    batch_size, n_rows, n_cols, pair_dim = pair.shape
    if n_rows != n_cols:
        raise ValueError("pair must have a square [N, N] token-pair plane")
    if not pair.is_floating_point():
        raise TypeError("pair must use a floating-point dtype")

    if folded_weight.ndim != 2:
        raise ValueError("folded_weight must have shape [LH, D]")
    if folded_weight.shape[1] != pair_dim:
        raise ValueError("folded_weight's D dimension must match pair")
    if offset.ndim != 1 or offset.shape[0] != folded_weight.shape[0]:
        raise ValueError("offset must have shape [LH]")
    if folded_weight.device != pair.device or offset.device != pair.device:
        raise ValueError("pair, folded_weight, and offset must share a device")
    if not folded_weight.is_floating_point() or not offset.is_floating_point():
        raise TypeError("folded_weight and offset must use floating-point dtypes")
    if (norm_weight is None) != (norm_bias is None):
        raise ValueError("norm_weight and norm_bias must be provided together")
    if norm_weight is not None and norm_bias is not None:
        if norm_weight.ndim != 1 or norm_weight.shape[0] != pair_dim:
            raise ValueError("norm_weight must have shape [D]")
        if norm_bias.ndim != 1 or norm_bias.shape[0] != pair_dim:
            raise ValueError("norm_bias must have shape [D]")
        if norm_weight.device != pair.device or norm_bias.device != pair.device:
            raise ValueError("norm_weight and norm_bias must share pair's device")
        if not norm_weight.is_floating_point() or not norm_bias.is_floating_point():
            raise TypeError("norm_weight and norm_bias must use floating-point dtypes")

    if not isinstance(eps, float) or not math.isfinite(eps) or eps < 0.0:
        raise ValueError("eps must be a finite non-negative float")
    if projection_dtype not in _FLOAT_DTYPES:
        raise ValueError(
            "projection_dtype must be one of float16, bfloat16, float32, or float64"
        )
    if type(requested_rows) is not int or requested_rows < 0:
        raise ValueError("requested_rows must be a non-negative int")
    return batch_size, n_rows, pair_dim, folded_weight.shape[0]


def _row_count(n: int, requested_rows: int) -> int:
    """Tile the complete pair plane; tiling never disables an input shape."""
    return max(1, min(n, requested_rows)) if requested_rows else max(1, n)


def _row_pair_bias_impl(
    pair: Tensor,
    folded_weight: Tensor,
    offset: Tensor,
    eps: float,
    projection_dtype: torch.dtype,
    requested_rows: int,
    norm_weight: Tensor | None,
    norm_bias: Tensor | None,
) -> Tensor:
    batch_size, n, pair_dim, n_outputs = _validate_inputs(
        pair,
        folded_weight,
        offset,
        eps,
        projection_dtype,
        requested_rows,
        norm_weight,
        norm_bias,
    )
    row_count = _row_count(n, requested_rows)

    output = torch.empty(
        (batch_size, n_outputs, n, n),
        device=pair.device,
        dtype=projection_dtype,
    )
    device_type = pair.device.type
    use_autocast = projection_dtype in _AUTOCAST_DTYPES and device_type in {
        "cpu",
        "cuda",
    }
    for row_start in range(0, n, row_count):
        row_stop = min(n, row_start + row_count)
        pair_rows = pair[:, row_start:row_stop]
        # A custom op implementation is allowed to retain the caller's
        # autocast state.  Set it explicitly so an outer autocast context and
        # an explicit projection_dtype have identical, deterministic semantics.
        with torch.autocast(
            device_type=device_type,
            dtype=projection_dtype if use_autocast else None,
            enabled=use_autocast,
        ):
            normalized = F.layer_norm(
                pair_rows,
                (pair_dim,),
                weight=norm_weight,
                bias=norm_bias,
                eps=eps,
            )
            if not use_autocast:
                normalized = normalized.to(dtype=projection_dtype)
                weight = folded_weight.to(dtype=projection_dtype)
                bias = offset.to(dtype=projection_dtype)
            else:
                # F.linear's autocast policy performs the same weight and
                # output promotion as a caller-side autocast context.  The
                # final cast below also covers CPU/device combinations whose
                # autocast kernel keeps the result in a wider accumulator.
                weight = folded_weight
                bias = offset
            projected = F.linear(normalized, weight, bias)
        projected = projected.to(dtype=projection_dtype)
        output[:, :, row_start:row_stop, :] = projected.movedim(-1, 1)
        del pair_rows, normalized, projected
    return output.contiguous()


@torch.library.custom_op(_OP_NAME, mutates_args=())
def _row_pair_bias_op(
    pair: Tensor,
    folded_weight: Tensor,
    offset: Tensor,
    eps: float,
    projection_dtype: torch.dtype,
    requested_rows: int,
    norm_weight: Tensor | None = None,
    norm_bias: Tensor | None = None,
) -> Tensor:
    """Opaque implementation for ``partidesign::row_pair_bias``."""

    return _row_pair_bias_impl(
        pair,
        folded_weight,
        offset,
        eps,
        projection_dtype,
        requested_rows,
        norm_weight,
        norm_bias,
    )


@_row_pair_bias_op.register_fake
def _row_pair_bias_fake(
    pair: Tensor,
    folded_weight: Tensor,
    offset: Tensor,
    eps: float,
    projection_dtype: torch.dtype,
    requested_rows: int,
    norm_weight: Tensor | None = None,
    norm_bias: Tensor | None = None,
) -> Tensor:
    del offset, eps, requested_rows, norm_weight, norm_bias
    return torch.empty(
        (pair.shape[0], folded_weight.shape[0], pair.shape[1], pair.shape[1]),
        device=pair.device,
        dtype=projection_dtype,
    )


def row_pair_bias(
    pair: Tensor,
    folded_weight: Tensor,
    offset: Tensor,
    eps: float,
    projection_dtype: torch.dtype,
    requested_rows: int,
    norm_weight: Tensor | None = None,
    norm_bias: Tensor | None = None,
) -> Tensor:
    """Project pair rows with an opaque, bounded custom operator.

    Args:
        pair: Pair representation with shape ``[B, N, N, D]``.
        folded_weight: LayerNorm-folded linear weight with shape ``[LH, D]``.
        offset: LayerNorm-folded linear bias with shape ``[LH]``.
        eps: Affine-free LayerNorm epsilon.
        projection_dtype: Required output/autocast projection dtype.
        requested_rows: Rows per internal iteration; zero processes the full
            pair plane. Every row is evaluated regardless of input length.
        norm_weight: Optional LayerNorm scale.  It must be paired with
            ``norm_bias``; both ``None`` selects affine-free fused behavior.
        norm_bias: Optional LayerNorm bias paired with ``norm_weight``.
    """

    if torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad
        for tensor in (pair, folded_weight, offset, norm_weight, norm_bias)
    ):
        raise RuntimeError("row_pair_bias is inference-only; call it with gradients disabled")

    return _row_pair_bias_op(
        pair,
        folded_weight,
        offset,
        eps,
        projection_dtype,
        requested_rows,
        norm_weight,
        norm_bias,
    )


__all__ = ["row_pair_bias"]
