"""Bounded per-forward pair-bias projection for the FM attention stack.

The bank deliberately owns no tensors.  It folds the pair LayerNorm affine
parameters into the per-layer bias projections for one layer group, computes
that group's bias in row chunks, and returns the temporary projected tensor to
the caller.  A new bank result is therefore made for every forward and every
group; nothing derived from model weights or pair features is persistent.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from .row_pair_bias import row_pair_bias


class PairBiasBank:
    """Compute a bounded group of already-projected pair attention biases."""

    @staticmethod
    def projection_dtype(pair_feats: Tensor) -> torch.dtype:
        device_type = pair_feats.device.type
        return (torch.get_autocast_dtype(device_type)
                if torch.is_autocast_enabled(device_type) else pair_feats.dtype)

    @staticmethod
    def resolve_group_size(
        pair_feats: Tensor,
        attention_modules: Sequence[torch.nn.Module],
        requested_group_size: int,
    ) -> int:
        """Group compatible layers independently of batch size and token count."""

        if not attention_modules:
            raise ValueError("pair-bias attention module sequence cannot be empty")
        if len({float(attention.pair_norm.eps) for attention in attention_modules}) != 1:
            return 0
        heads = {int(attention.heads) for attention in attention_modules}
        if len(heads) != 1:
            raise ValueError("pair-bias layers must use one head count")
        return max(1, min(int(requested_group_size), len(attention_modules)))

    @staticmethod
    def _fold_projection(
        attention_modules: Sequence[torch.nn.Module],
        pair_feats: Tensor,
    ) -> tuple[Tensor, Tensor, float]:
        """Return the current group's LN-folded projection and offset.

        ``pair_norm`` is applied affine-free below.  Its weight and bias are
        folded into each ``to_bias`` projection at call time, so state loading,
        device moves, dtype changes, and parameter edits are reflected on the
        next forward without an invalidation cache.
        """

        projection_weights = []
        projection_biases = []
        if len({float(attention.pair_norm.eps) for attention in attention_modules}) != 1:
            raise ValueError('grouped pair-bias projection requires identical LayerNorm epsilon')
        for attention in attention_modules:
            pair_norm = attention.pair_norm
            to_bias = attention.to_bias
            dtype = pair_feats.dtype
            device = pair_feats.device
            ln_weight = pair_norm.weight.to(device=device, dtype=dtype)
            ln_bias = pair_norm.bias.to(device=device, dtype=dtype)
            bias_weight = to_bias.weight.to(device=device, dtype=dtype)
            projection_weights.append(bias_weight * ln_weight.unsqueeze(0))
            projection_biases.append(torch.matmul(bias_weight, ln_bias))

        return (
            torch.cat(projection_weights, dim=0).contiguous(),
            torch.cat(projection_biases, dim=0).contiguous(),
            float(attention_modules[0].pair_norm.eps),
        )

    @classmethod
    def compute(
        cls,
        pair_feats: Tensor,
        attention_modules: Sequence[torch.nn.Module],
        layer_start: int,
        layer_stop: int,
        row_chunk_size: int = 0,
    ) -> Tensor:
        """Compute ``[B, (layer_stop-layer_start)*H, N, N]`` pair bias.

        ``attention_modules`` is the FM transformer's static attention-module
        tuple.  Only the selected group is projected, keeping the temporary
        head plane proportional to ``layer_group_size`` rather than the full
        number of FM layers.
        """

        if pair_feats.ndim != 4:
            raise ValueError("pair_feats must have shape [B, N, N, D]")
        if pair_feats.shape[1] != pair_feats.shape[2]:
            raise ValueError("pair_feats must have a square token-pair plane")
        if not 0 <= layer_start < layer_stop <= len(attention_modules):
            raise ValueError("invalid pair-bias layer group")

        group = attention_modules[layer_start:layer_stop]
        if not group:
            raise ValueError("pair-bias layer group cannot be empty")
        heads = {int(attention.heads) for attention in group}
        if len(heads) != 1:
            raise ValueError("pair-bias layer group must use one head count")
        n_heads = heads.pop()

        projection, offset, eps = cls._fold_projection(group, pair_feats)
        return row_pair_bias(pair_feats, projection, offset, eps,
                             cls.projection_dtype(pair_feats),
                             int(row_chunk_size))
