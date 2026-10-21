"""AF2 triangle multiplication adapter for the copied Pallas kernel.

The public entry point accepts the parameter layout created by the vendored
ColabDesign module (contiguous left/right projection columns).  The pinned
kernel uses interleaved projection columns, so conversion is kept here at the
adapter boundary.  CPU and unsupported dtype backends use the same native JAX
statement; GPU bf16 calls use the biased Pallas implementation.
"""

from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp


F32 = jnp.float32
BF16 = jnp.bfloat16
LN_EPS = 1e-5
_GPU_TILE = 16


def _ln(x, scale, offset, axis=-1):
    x32 = x.astype(F32)
    mean = jnp.mean(x32, axis=axis, keepdims=True)
    var = jnp.mean(x32 * x32, axis=axis, keepdims=True) - mean * mean
    return (x32 - mean) * jax.lax.rsqrt(var + LN_EPS) * scale.astype(F32) + offset.astype(F32)


def _linear(x, weights, bias):
    return jnp.einsum("...c,cd->...d", x, weights, precision=None) + bias


def _pack_projection(weights, input_channels, output_channels):
    del input_channels
    return weights.reshape(weights.shape[0], 2, output_channels).transpose(0, 2, 1).reshape(weights.shape[0], 2 * output_channels)


def _pack_bias(bias, output_channels):
    return bias.reshape(2, output_channels).transpose(1, 0).reshape(2 * output_channels)


def _kernel_params(p: Mapping[str, jax.Array], input_channels: int, intermediate_channels: int):
    return {
        "ln_in_scale": p["ln_in_scale"],
        "ln_in_offset": p["ln_in_offset"],
        "w_proj": _pack_projection(p["w_proj"], input_channels, intermediate_channels),
        "w_gate": _pack_projection(p["w_gate"], input_channels, intermediate_channels),
        "b_proj": _pack_bias(p["b_proj"], intermediate_channels),
        "b_gate": _pack_bias(p["b_gate"], intermediate_channels),
        "ln_c_scale": p["ln_c_scale"],
        "ln_c_offset": p["ln_c_offset"],
        "w_out": p["w_out"],
        "w_gl": p["w_gl"],
        "b_out": p["b_out"],
        "b_gl": p["b_gl"],
    }


def triangle_multiplication_reference(act, mask, p: Mapping[str, jax.Array], *, equation: str):
    """Native AF2 fused-projection statement with stock dtype rounding."""
    if equation not in ("ikc,jkc->ijc", "kjc,kic->ijc"):
        raise ValueError(equation)
    dtype = act.dtype
    channels = int(act.shape[-1])
    intermediate = int(p["w_proj"].shape[-1] // 2)
    x = _ln(act, p["ln_in_scale"], p["ln_in_offset"]).astype(dtype)
    projection = _linear(x, p["w_proj"].astype(dtype), p["b_proj"].astype(dtype))
    gate = _linear(x, p["w_gate"].astype(dtype), p["b_gate"].astype(dtype))
    projected = projection * jax.nn.sigmoid(gate)
    projected *= mask.astype(dtype)[..., None]
    left, right = jnp.split(projected, (intermediate,), axis=-1)
    if equation == "ikc,jkc->ijc":
        tri = jnp.einsum("ikc,jkc->ijc", left, right)
    else:
        tri = jnp.einsum("kjc,kic->ijc", left, right)
    center = _ln(tri, p["ln_c_scale"], p["ln_c_offset"]).astype(dtype)
    output = _linear(center, p["w_out"].astype(dtype), p["b_out"].astype(dtype))
    output_gate = jax.nn.sigmoid(_linear(x, p["w_gl"].astype(dtype), p["b_gl"].astype(dtype)))
    return (output * output_gate).astype(act.dtype)


def _pad_inputs(act, mask, tile):
    n = int(act.shape[0])
    padded = ((n + tile - 1) // tile) * tile
    if padded == n:
        return act, mask, n
    act_padded = jnp.pad(act, ((0, padded - n), (0, padded - n), (0, 0)))
    mask_padded = jnp.pad(mask, ((0, padded - n), (0, padded - n)))
    return act_padded, mask_padded, n


def _pallas_triangle(act, mask, p, *, equation):
    from . import fpf_pallas_bias

    act_padded, mask_padded, n = _pad_inputs(act, mask, _GPU_TILE)
    cfg = {
        "t1": _GPU_TILE,
        "t2": _GPU_TILE,
        "w1": 4,
        "w2": 4,
        "s1": 2,
        "s2": 2,
        "ein": "xla",
    }
    result = fpf_pallas_bias.triangle_multiplication_fused_bias(
        act_padded,
        mask_padded,
        _kernel_params(p, int(act.shape[-1]), int(p["w_proj"].shape[-1] // 2)),
        equation=equation,
        cfg=cfg,
    )
    return result[:n, :n, :]


def triangle_multiplication_fused(act, mask, p: Mapping[str, jax.Array], *, equation: str):
    """Run AF2 triangle multiplication with the GPU Pallas fast path."""
    if act.ndim != 3 or act.shape[0] != act.shape[1]:
        raise ValueError(f"expected square pair activations, got {act.shape}")
    if jax.default_backend() == "gpu" and act.dtype == BF16:
        return _pallas_triangle(act, mask, p, equation=equation)
    return triangle_multiplication_reference(act, mask, p, equation=equation)
