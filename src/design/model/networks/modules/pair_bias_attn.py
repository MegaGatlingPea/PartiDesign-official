# MIT License

# Copyright (c) 2022 MattMcPartlon

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional

import torch
from torch import Tensor, einsum, nn
from torch.nn import functional as F

from src.design.model.networks.modules.adaptive_ln_scale import (
    AdaptiveLayerNorm,
    AdaptiveOutputScale,
)


def exists(val) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x, y):
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


max_neg_value = lambda x: torch.finfo(x.dtype).min


class PairBiasAttention(nn.Module):
    """
    Scalar Feature masked attention with pair bias and gating.
    Code modified from
    https://github.com/MattMcPartlon/protein-docking/blob/main/protein_learning/network/modules/node_block.py
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: Optional[int] = None,
        **kawrgs,  # noqa
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads, self.scale = heads, dim_head**-0.5
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        # Plain configuration state only.  It is intentionally not a buffer,
        # parameter, or derived-weight cache, so the checkpoint ABI is stable.
        self._attention_use_sdpa = False
        self._bioir_backend = None
        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Optional[Tensor],
        mask: Optional[Tensor],
        pair_bias_valid_mask: Optional[Tensor] = None,
        projected_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """Multi-head scalar Attention Layer

        :param node_feats: scalar features of shape (b,n,d_s)
        :param pair_feats: pair features of shape (b,n,n,d_e)
        :param mask: boolean tensor of node adjacencies
        :return:
        """
        assert exists(self.to_bias) or (not exists(pair_feats) and projected_bias is None)
        if projected_bias is not None:
            assert exists(self.to_bias)
            assert projected_bias.ndim == 4
            assert projected_bias.shape[1] == self.heads
            assert projected_bias.shape[0] == node_feats.shape[0]
            assert projected_bias.shape[2:] == (node_feats.shape[1], node_feats.shape[1])
            if pair_feats is not None:
                assert projected_bias.shape[:1] + projected_bias.shape[2:] == pair_feats.shape[:-1]
        node_feats, h = self.node_norm(node_feats), self.heads
        if projected_bias is None:
            pair_feats = self.pair_norm(pair_feats) if exists(pair_feats) else None
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)
        # pair bias: [b, n, n, h] -> [b, h, n, n]
        b = (
            projected_bias
            if projected_bias is not None
            else (
                self.to_bias(pair_feats).movedim(-1, 1)
                if exists(pair_feats)
                else 0
            )
        )
        if pair_bias_valid_mask is not None:
            assert exists(pair_feats) or projected_bias is not None
            expected_pair_shape = (
                pair_feats.shape[:-1]
                if pair_feats is not None
                else projected_bias.shape[:1] + projected_bias.shape[2:]
            )
            assert pair_bias_valid_mask.shape == expected_pair_shape
            b = b * pair_bias_valid_mask[:, None].to(dtype=b.dtype)
        if projected_bias is not None and b.dtype != q.dtype:
            # The stock projection follows the active autocast dtype.  Keep
            # the optional pre-projected route on that same attention dtype;
            # this cast is ephemeral and does not alter checkpoint state.
            b = b.to(dtype=q.dtype)
        # q, k, v, g: [b, n, h*d] -> [b, h, n, d]
        q = q.unflatten(-1, (h, -1)).movedim(-2, 1)
        k = k.unflatten(-1, (h, -1)).movedim(-2, 1)
        v = v.unflatten(-1, (h, -1)).movedim(-2, 1)
        g = g.unflatten(-1, (h, -1)).movedim(-2, 1)
        attn_feats = self._attn(q, k, v, b, mask)
        # [b, h, n, d] -> [b, n, h*d]
        attn_feats = (torch.sigmoid(g) * attn_feats).movedim(1, -2).flatten(start_dim=-2)
        return self.to_out_node(attn_feats)

    def _attn(self, q, k, v, b, mask: Optional[Tensor]) -> Tensor:
        """Perform attention update"""
        if self._bioir_backend is not None and not self.training and not torch.is_grad_enabled():
            return self._bioir_backend.attention(q, k, v, b, mask)
        if self._attention_use_sdpa and not self.training and not torch.is_grad_enabled():
            attn_mask = None
            if torch.is_tensor(b):
                attn_mask = b.to(dtype=q.dtype)
            if mask is not None:
                if attn_mask is None:
                    attn_mask = q.new_zeros(
                        (q.shape[0], 1, q.shape[-2], k.shape[-2])
                    )
                mask = mask.unsqueeze(1)
                # Keep masked logits finite.  This retains the old padded-row
                # behavior (finite softmax, then the outer token mask zeros
                # padded outputs) and avoids all-masked-row NaNs.
                attn_mask = attn_mask + (
                    ~mask
                ).to(dtype=attn_mask.dtype) * torch.finfo(attn_mask.dtype).min
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                scale=self.scale,
            )
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if exists(mask):
            # [b, i, j] -> [b, 1, i, j]
            mask = mask.unsqueeze(1)
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1)
        return einsum("b h i j, b h j d -> b h i d", attn, v)


class MultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, dim_pair, nheads, dim_cond, use_qkln):
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = PairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
        )
        self.scale_output = AdaptiveOutputScale(dim=dim_token, dim_cond=dim_cond)

    def forward(
        self,
        x,
        pair_rep,
        cond,
        mask,
        pair_bias_valid_mask=None,
        projected_bias=None,
    ):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        x = self.adaln(x, cond, mask)
        x = self.mha(
            node_feats=x,
            pair_feats=pair_rep,
            mask=pair_mask,
            pair_bias_valid_mask=pair_bias_valid_mask,
            projected_bias=projected_bias,
        )
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]
