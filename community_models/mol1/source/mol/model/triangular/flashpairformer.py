# Copyright 2026 Molence Bio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional FlashPairformer TMK3 integration, without upstream model patches.

The adapter reuses parameters registered on the model without changing
checkpoint keys. Packed inference weights are invalidated when parameters
change, a checkpoint is loaded, or a module changes device.
"""

import importlib
import logging
import os
import sys
from collections import Counter

import torch

logger = logging.getLogger(__name__)
_REPORTED = set()


def _import_trimul():
    """Import ``opt_core.kernels.trimul``, honouring ``OPT_CORE_DIR``.

    ``opt_core`` is normally installed from the revision pinned in
    ``requirements.txt``. When it is only available as a source checkout,
    ``OPT_CORE_DIR`` points at that root. Reading the variable here means every
    caller gets the same resolution -- previously only ``inference_demo.sh``
    put it on ``PYTHONPATH``, so tests and library users had to repeat that by
    hand, and eleven kernel tests failed with "opt_core not installed" on a
    machine where the source tree was in fact present.
    """
    try:
        return importlib.import_module("opt_core.kernels.trimul")
    except ImportError:
        source_root = os.environ.get("OPT_CORE_DIR")
        if not source_root or not os.path.isdir(source_root):
            raise
        if source_root not in sys.path:
            sys.path.append(source_root)
        importlib.invalidate_caches()
        return importlib.import_module("opt_core.kernels.trimul")


def clear_flashpairformer_cache(model):
    """Release packed weights after external ``parameter.data`` modifications.

    Ordinary load_state_dict(), copy_(), and device/dtype moves are handled
    automatically. This helper also lets long-lived workers release the cache.
    """
    for module in model.modules():
        module.__dict__.pop("_flashpairformer_cache", None)
        module.__dict__.pop("_flashpairformer_signature", None)


def flashpairformer_stats(model):
    """Aggregate served/fallback counts; no tensors or checkpoint state."""
    counts = Counter()
    for module in model.modules():
        counts.update(getattr(module, "_flashpairformer_stats", {}))
    return dict(counts)


def _record(module, reason):
    counts = module.__dict__.setdefault("_flashpairformer_stats", Counter())
    counts[reason] += 1
    if reason not in _REPORTED:
        _REPORTED.add(reason)
        logger.info("FlashPairformer triangle multiplication: %s", reason)


def _weights(module):
    return {
        "ln_in_w": module.layer_norm_in.weight,
        "ln_in_b": module.layer_norm_in.bias,
        "w_ag": module.linear_a_g.weight,
        "w_ap": module.linear_a_p.weight,
        "w_bg": module.linear_b_g.weight,
        "w_bp": module.linear_b_p.weight,
        "ln_out_w": module.layer_norm_out.weight,
        "ln_out_b": module.layer_norm_out.bias,
        "w_o": module.linear_z.weight,
        "w_og": module.linear_g.weight,
    }


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:  # Parameters created inside inference_mode have no counter.
        return None


def _weight_cache(module, weights, compute_dtype):
    # Upstream's pack key omits version counters and several parameters. Check
    # every parameter here: model-merging workers reload tensors in place.
    signature = (
        compute_dtype,
        tuple(
            (
                name, id(t), t.data_ptr(), _version(t), tuple(t.shape),
                tuple(t.stride()), t.dtype, t.device,
            )
            for name, t in weights.items()
        ),
    )
    if module.__dict__.get("_flashpairformer_signature") != signature:
        module._flashpairformer_cache = {}
        module._flashpairformer_signature = signature
    return module._flashpairformer_cache


def triangle_multiplication(module, z, mask=None):
    """Return the BF16 update, or None to use Mol-1's stock backend.

    No residual is fused: the caller preserves stock residual addition and its
    dtype, including FP32-resident pairs under BF16 autocast. Grad-enabled or
    training calls use the differentiable stock path. Runtime kernel errors
    propagate; only explicit upstream support refusals fall back.
    """
    if module.training or torch.is_grad_enabled():
        clear_flashpairformer_cache(module)
        _record(module, "fallback:autograd_or_training")
        return None
    compute_dtype = (
        torch.get_autocast_dtype("cuda")
        if z.is_cuda and torch.is_autocast_enabled("cuda")
        else module.linear_a_p.weight.dtype
    )
    reason = None
    if not z.is_cuda:
        reason = "cpu"
    elif compute_dtype != torch.bfloat16 or z.dtype not in (torch.bfloat16, torch.float32):
        reason = "requires_bf16_compute"
    elif z.ndim != 3 or z.shape[-3] != z.shape[-2]:
        reason = "requires_single_square_pair"
    elif module.c_z != module.c_hidden or module.c_z % 64:
        reason = "unsupported_channels"
    elif mask is not None and (mask.shape != z.shape[:-1] or mask.device != z.device):
        reason = "unsupported_mask_layout"
    if reason is not None:
        _record(module, "fallback:" + reason)
        return None

    # cuEq uses a different Torch implementation for short sequences. TMK3
    # reproduces its fused implementation, so preserve this stock dispatch too.
    cueq = importlib.import_module("cuequivariance_ops_torch.triangle_multiplicative_update")
    threshold = max(100, getattr(cueq, "CUEQ_TRIMUL_FALLBACK_THRESHOLD", 100))
    if z.shape[-2] <= threshold:
        _record(module, "fallback:cueq_short_sequence")
        return None

    try:
        trimul = _import_trimul()
    except ImportError as exc:
        raise RuntimeError(
            "triangle_multiplicative=flashpairformer requires Anthropic opt_core. "
            "Install the pinned dependency from requirements.txt, or point "
            "OPT_CORE_DIR at a source checkout of it; see docs/kernels.md."
        ) from exc
    weights = _weights(module)
    cache = _weight_cache(module, weights, compute_dtype)
    try:
        update = trimul.triangle_multiplication(
            z,
            mask,
            direction="outgoing" if module._outgoing else "incoming",
            weights=weights,
            word="tmk3_exact",
            residual=False,
            cache=cache,
            eps=module.layer_norm_in.eps,
        )
    except trimul.Refusal as exc:
        _record(module, "fallback:upstream_refusal:" + str(exc))
        return None
    _record(module, "tmk3_exact")
    return update
