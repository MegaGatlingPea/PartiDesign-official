"""Optional BioNeMo primitives and graph regions for the existing FM network.

The checkpoint modules, conditioning and solver remain PartiDesign's. BioIR
is imported only when selected; its graph tracker owns capture and replay.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from functools import lru_cache
from importlib import metadata
import os
import re
from typing import Any
import weakref

import torch
from torch import Tensor


_DEPENDENCY_HELP = (
    "The BioIR backend requires the complete partidesign_cu13 environment "
    "(partidesign-bioir-kernels 0.2.x, PyTorch >=2.11, CUDA 13), or a compatible "
    "official bionemo-ir 0.2.x install. Install exactly one BioIR provider."
)
_CONFIG_KEYS = frozenset({"cuda_graph", "max_graphs", "max_tokens"})


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or set(config) != _CONFIG_KEYS:
        raise ValueError(f"model.bioir requires exactly {sorted(_CONFIG_KEYS)}")
    if type(config["cuda_graph"]) is not bool:
        raise TypeError("model.bioir.cuda_graph must be bool")
    for name in ("max_graphs", "max_tokens"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"model.bioir.{name} must be a positive integer")
    return dict(config)


@lru_cache(maxsize=1)
def _load_api() -> dict[str, Any]:
    """Import the supported API without creating a CUDA context or a kernel."""
    providers = {}
    for name in ("partidesign-bioir-kernels", "bionemo-ir"):
        try:
            providers[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    if len(providers) != 1:
        reason = "No BioIR provider is installed" if not providers else "Multiple BioIR providers own the same namespace"
        raise RuntimeError(f"{reason}. {_DEPENDENCY_HELP}")
    provider, release = next(iter(providers.items()))
    if not re.match(r"^0\.2\.", release):
        raise RuntimeError(f"Unsupported {provider} version {release!r}. {_DEPENDENCY_HELP}")
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if match is None or tuple(map(int, match.groups())) < (2, 11):
        raise RuntimeError(f"BioIR 0.2 requires PyTorch >=2.11, found {torch.__version__}. {_DEPENDENCY_HELP}")
    try:
        from bionemo_ir._torch.attention_backend.interface import AttentionType
        from bionemo_ir._torch.attention_backend.utils import auto_select_pairwise_attention_backend, create_attention
        from bionemo_ir._torch.custom_ops import get_adaln_layernorm_sigmoid_op
        from bionemo_ir._torch.graph_optimization.config import CUDAGraphOptimizationConfig, GraphOptimizationMode, InputKeyMethod
        from bionemo_ir._torch.graph_optimization.region import GraphRegion
        from bionemo_ir.dsl_kernels.triton.fused_swiglu import FusedSwiGLU
    except (ImportError, OSError, AttributeError) as error:
        raise RuntimeError(f"The installed BioIR lacks a required kernel API. {_DEPENDENCY_HELP}") from error
    return {
        "provider": provider, "version": release, "AttentionType": AttentionType,
        "auto_select_attention": auto_select_pairwise_attention_backend,
        "create_attention": create_attention, "get_adaln_op": get_adaln_layernorm_sigmoid_op,
        "FusedSwiGLU": FusedSwiGLU, "GraphRegion": GraphRegion,
        "CUDAGraphOptimizationConfig": CUDAGraphOptimizationConfig,
        "GraphOptimizationMode": GraphOptimizationMode, "InputKeyMethod": InputKeyMethod,
    }


def require_bioir() -> dict[str, str | None]:
    """Validate optional dependencies/API without initializing GPU execution."""
    api = _load_api()
    return {"provider": api["provider"], "bionemo_ir": api["version"], "torch": torch.__version__, "cuda_build": torch.version.cuda}


class BioIRStartupError(RuntimeError):
    """A controlled optional dependency or deterministic kernel admission failure."""


def _probe_device_indices() -> tuple[int, ...]:
    if not torch.cuda.is_available():
        raise BioIRStartupError("CUDA is unavailable")
    count = torch.cuda.device_count()
    if count < 1:
        raise BioIRStartupError("no visible CUDA devices")
    raw_rank = os.environ.get("LOCAL_RANK")
    if raw_rank is None:
        return tuple(range(count))
    try:
        rank = int(raw_rank)
    except ValueError as error:
        raise ValueError(f"invalid LOCAL_RANK={raw_rank!r}") from error
    if rank < 0 or rank >= count:
        raise ValueError(f"LOCAL_RANK={rank} is outside the visible CUDA device set")
    return (rank,)


def _probe_bioir_primitives(api: Mapping[str, Any], device: torch.device) -> None:
    """Exercise real kernels using constants, without consuming CPU/CUDA RNG."""
    backend = _BioIRBackend(api, {"cuda_graph": False, "max_graphs": 1, "max_tokens": 16})
    with torch.inference_mode(), torch.cuda.device(device):
        q = torch.zeros((1, 2, 16, 64), dtype=torch.bfloat16, device=device)
        attention = backend.attention(
            q, q, torch.ones_like(q), None,
            torch.ones((1, 16, 16), dtype=torch.bool, device=device),
        )
        x = torch.zeros((1, 16, 128), dtype=torch.float32, device=device)
        adaln = api["get_adaln_op"](torch.float32, N=128, rms_norm=False)(
            x, torch.zeros_like(x), torch.zeros_like(x), out=torch.empty_like(x), eps=1e-5,
        )
        swiglu = backend.swiglu(torch.zeros((1, 16, 256), dtype=torch.bfloat16, device=device))
        torch.cuda.synchronize(device)
        if (not torch.allclose(attention, torch.ones_like(attention), atol=5e-3, rtol=5e-3)
                or not bool(torch.isfinite(adaln).all()) or bool(torch.count_nonzero(adaln))
                or not bool(torch.isfinite(swiglu).all()) or bool(torch.count_nonzero(swiglu))):
            raise BioIRStartupError("deterministic BioIR primitive probe failed")


def probe_bioir_runtime() -> dict[str, Any]:
    """Admit the optional backend on actual visible devices before generation.

    Initial launchers check every visible index; externally launched children
    check LOCAL_RANK. Device contexts restore the caller's current device.
    Graph capture is left to GraphRegion and its existing eager fallback.
    """
    startup_errors = (ImportError, OSError, RuntimeError)
    try:
        indices = _probe_device_indices()
        from triton.compiler.errors import TritonError
        from cutlass.base_dsl.common import DSLBaseError
        startup_errors += (TritonError, DSLBaseError)
        api = _load_api()
        for index in indices:
            _probe_bioir_primitives(api, torch.device("cuda", index))
    except (torch.OutOfMemoryError, MemoryError):
        raise
    except startup_errors as error:
        # Optional wrappers can report allocation failure as a CUDA status.
        # OOM is never an optional-backend fallback signal.
        if "out of memory" in str(error).lower().replace("_", " "):
            raise
        if isinstance(error, BioIRStartupError):
            raise
        raise BioIRStartupError(f"{type(error).__name__}: {error}") from error
    return {"provider": api["provider"], "version": api["version"], "devices": indices}


def _network(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "nn", model)


def _tensor_version(tensor: Tensor) -> int | None:
    # Inference tensors have no mutation counter. Their mask values must be
    # checked on every call; object identity cannot certify unchanged padding.
    return None if tensor.is_inference() else tensor._version


def _validate_left_mask(mask: Tensor) -> None:
    if mask.ndim != 2 or mask.dtype != torch.bool or min(mask.shape) < 1:
        raise ValueError("BioIR token_mask must be a nonempty [B,N] bool tensor")
    lengths = mask.sum(dim=-1)
    prefix = torch.arange(mask.shape[-1], device=mask.device)[None, :] < lengths[:, None]
    if not bool(((lengths > 0).all() & (mask == prefix).all()).item()):
        raise ValueError("BioIR requires real tokens in every sample and left-aligned token padding")


class _BioIRBackend:
    """Plain state with no registered parameters, buffers, or copied weights."""

    def __init__(self, api: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self.api, self.config = dict(api), dict(config)
        self.attention_kernels: dict[tuple, Any] = {}
        self.adaln_kernels: dict[tuple, Any] = {}
        self.swiglu_kernels: dict[tuple, Any] = {}
        self.attention_backends: set[str] = set()
        self.adaln_backends: set[str] = set()
        self.seen_shapes: set[tuple[int, int]] = set()
        self.retired_graph_counts: Counter[str] = Counter()
        self.mask_validation_count = self.graph_resets = self.out_of_range_calls = 0
        self._trusted_forward_depth = 0
        self._mask_ref: weakref.ReferenceType | None = None
        self._mask_version: int | None = None
        self._active_kv_lengths: Tensor | None = None

    @staticmethod
    def _require_cuda(tensor: Tensor) -> None:
        if tensor.device.type != "cuda":
            raise RuntimeError("The explicitly selected BioIR backend requires CUDA inputs")

    @contextmanager
    def _trusted_forward(self):
        self._trusted_forward_depth += 1
        try:
            yield
        finally:
            self._trusted_forward_depth -= 1

    def _validate_mask_once(self, mask: Tensor) -> None:
        version = _tensor_version(mask)
        if (
            version is not None
            and self._mask_ref is not None
            and self._mask_ref() is mask
            and version == self._mask_version
        ):
            return
        _validate_left_mask(mask)
        self._mask_ref, self._mask_version = weakref.ref(mask), version
        self.mask_validation_count += 1

    def forward(self, network: torch.nn.Module, batch: Mapping[str, Any]) -> Any:
        mask = batch["token_mask"]
        self._require_cuda(mask)
        self._validate_mask_once(mask)
        self.seen_shapes.add(tuple(mask.shape))
        region = getattr(network, "_bioir_graph_region", None)
        with torch.cuda.device(mask.device), self._trusted_forward():
            if region is None:
                return network._forward_eager(batch)
            if mask.shape[-1] > self.config["max_tokens"]:
                self.out_of_range_calls += 1
                if region.tracker is not None:
                    self.retired_graph_counts.update(region.tracker.execution_counts)
                    region.reset()
                    self.graph_resets += 1
                return network._forward_eager(batch)
            # BioIR 0.2's capture/replay paths clone graph-owned output leaves.
            # The solver can therefore retain its preceding clean prediction.
            return region(batch)

    def prepare_forward(self, mask: Tensor) -> None:
        # This reduction belongs inside the captured network, so replay uses
        # the current mask values. All attention layers share its result.
        self._active_kv_lengths = mask.sum(dim=-1, dtype=torch.int32).contiguous()

    def attention(self, q: Tensor, k: Tensor, v: Tensor, bias: Any, mask: Tensor | None) -> Tensor:
        self._require_cuda(q)
        # Full-width Q/K LayerNorm may return FP32 under CUDA autocast. Native
        # SDPA casts its operands to the autocast compute dtype at this point;
        # apply that same conversion before choosing the BioIR implementation.
        if torch.is_autocast_enabled("cuda"):
            compute_dtype = torch.get_autocast_dtype("cuda")
            if compute_dtype in (torch.float16, torch.bfloat16):
                q, k, v = (tensor.to(dtype=compute_dtype) for tensor in (q, k, v))
        batch, heads, nq, dim = q.shape
        nk = k.shape[-2]
        with torch.cuda.device(q.device):
            name = self.api["auto_select_attention"](q.dtype)
            trusted_lengths = self._active_kv_lengths if self._trusted_forward_depth else None
            key_mask = None
            if trusted_lengths is None:
                key_mask = torch.ones((batch, nk), dtype=torch.bool, device=q.device) if mask is None else mask.any(dim=-2)
            if mask is not None and name == "CuTeDSL" and not self._trusted_forward_depth:
                # Direct layer calls may have general masks. Only a validated
                # outer-product prefix mask can be represented by KV counts.
                query_mask = mask.any(dim=-1)
                outer = query_mask[:, :, None] & key_mask[:, None, :]
                lengths = key_mask.sum(-1)
                prefix = torch.arange(nk, device=q.device)[None, :] < lengths[:, None]
                legal = (mask == outer).all() & (key_mask == prefix).all() & (lengths > 0).all()
                if not bool(legal.item()):
                    name = "SDPA"
            key = (q.device, q.dtype, heads, dim, name)
            kernel = self.attention_kernels.get(key)
            if kernel is None:
                kernel = self.api["create_attention"](
                    name, layer_idx=0, num_heads=heads, head_dim=dim,
                    attention_type=self.api["AttentionType"].PAIRWISE,
                )
                self.attention_kernels[key] = kernel
            self.attention_backends.add(name)
            projected = bias.to(q) if isinstance(bias, Tensor) else q.new_zeros((batch, heads, nq, nk))
            if name == "CuTeDSL":
                mask_bias = trusted_lengths if trusted_lengths is not None else key_mask.sum(dim=-1, dtype=torch.int32).contiguous()
            else:
                mask_bias = q.new_zeros((batch, 1, nq, nk))
                if mask is not None:
                    mask_bias = (~mask[:, None]).to(q.dtype) * torch.finfo(q.dtype).min
            q_flat, k_flat, v_flat = (
                tensor.transpose(1, 2).contiguous().flatten(start_dim=2) for tensor in (q, k, v)
            )
            return kernel.forward(q_flat, k_flat, v_flat, biases=[mask_bias, projected]).movedim(-2, 1)

    def adaln(self, module: torch.nn.Module, x: Tensor, condition: Tensor, mask: Tensor) -> Tensor:
        self._require_cuda(x)
        normalized_condition = module.norm_cond(condition)
        # Keep checkpoint projections separate, including their BF16 rounding.
        scale = module.to_gamma[0](normalized_condition)
        shift = module.to_beta(normalized_condition)
        dtype = x.dtype if x.dtype == scale.dtype == shift.dtype else torch.float32
        key = (x.device, dtype, x.shape[-1])
        with torch.cuda.device(x.device):
            operation = self.adaln_kernels.get(key)
            if operation is None:
                operation = self.api["get_adaln_op"](dtype, N=x.shape[-1], rms_norm=False)
                self.adaln_kernels[key] = operation
            self.adaln_backends.add(getattr(operation, "__name__", type(operation).__name__))
            # Both input and destination must satisfy the CuTe contiguous
            # layout. Never overwrite x: the caller may retain it as a residual.
            kernel_input = x.contiguous()
            out = operation(
                kernel_input, scale.contiguous(), shift.contiguous(),
                out=torch.empty_like(kernel_input), eps=module.norm.eps,
            )
        return out * mask[..., None]

    def swiglu(self, packed: Tensor) -> Tensor:
        self._require_cuda(packed)
        if packed.shape[-1] % 2:
            raise ValueError("BioIR SwiGLU requires an even packed [value|gate] width")
        width = packed.shape[-1] // 2
        key = (packed.device, packed.dtype, width)
        with torch.cuda.device(packed.device):
            operation = self.swiglu_kernels.get(key)
            if operation is None:
                operation = self.api["FusedSwiGLU"](d=width, three_way=False, dtype=packed.dtype)
                self.swiglu_kernels[key] = operation
            return operation(packed.contiguous())


def configure_bioir(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.nn.Module:
    """Enable BioIR after strict native loading, preserving state-dict paths."""
    from .networks.local_latents import LocalLatentsTransformer
    from .networks.modules.adaptive_ln_scale import AdaptiveLayerNorm
    from .networks.modules.pair_bias_attn import PairBiasAttention
    from .networks.modules.swiglu import SwiGLU

    options = _validate_config(config)
    network = _network(model)
    if not isinstance(network, LocalLatentsTransformer):
        raise TypeError("configure_bioir requires the uncompiled PartiDesign FM network")
    if getattr(model, "_denoiser_compiled", False):
        raise RuntimeError("BioIR and native torch.compile cannot wrap the same denoiser")
    if getattr(network, "_bioir_backend", None) is not None:
        raise RuntimeError("The FM network is already configured for BioIR")
    api = _load_api()
    backend = _BioIRBackend(api, options)
    region = None
    if options["cuda_graph"]:
        policy = api["CUDAGraphOptimizationConfig"](
            graph_optimization_mode=api["GraphOptimizationMode"].CUDA_GRAPH_VIA_TORCH,
            input_key_method=api["InputKeyMethod"].EXACT,
            input_routing_config=None,
            num_calls_for_kernel_compilation=1,
            num_calls_for_memory_allocator=1,
            capture_on_first_call=True,
            verify_capture=True,
            num_graphs_max_for_this_module=options["max_graphs"],
            graph_cache_budget_bytes=4 << 30,
        )
        region = api["GraphRegion"](network, "_forward_eager", policy)
        region.train(network.training)
    # Kernel state is a plain object. The graph region owns no model weights.
    for module in network.modules():
        if isinstance(module, (PairBiasAttention, AdaptiveLayerNorm, SwiGLU)):
            module._bioir_backend = backend
    network._bioir_backend = backend
    network._bioir_graph_region = region
    return model


def bioir_status(model: torch.nn.Module) -> dict[str, Any]:
    """Report actual kernel choices and graph activity without touching CUDA."""
    network = _network(model)
    backend = getattr(network, "_bioir_backend", None)
    if backend is None:
        return {"enabled": False}
    region = getattr(network, "_bioir_graph_region", None)
    tracker = None if region is None else region.tracker
    counts = Counter(backend.retired_graph_counts)
    if tracker is not None:
        counts.update(tracker.execution_counts)
    return {
        "enabled": True, "bionemo_ir": backend.api["version"],
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "attention_backends": sorted(backend.attention_backends),
        "adaln_backends": sorted(backend.adaln_backends),
        "swiglu": "FusedSwiGLU" if backend.swiglu_kernels else None,
        "mask_validations": backend.mask_validation_count,
        "seen_shapes": [list(shape) for shape in sorted(backend.seen_shapes)],
        "graph": {
            **backend.config, "verify_capture": True, "budget_bytes": 4 << 30,
            "execution_counts": dict(counts),
            "cached_keys": 0 if tracker is None else len(tracker.graph_state_by_key),
            "fallback_keys": 0 if tracker is None else sum(tracker.fallback_to_eager_by_key.values()),
            "resets_for_long_inputs": backend.graph_resets,
            "out_of_range_calls": backend.out_of_range_calls,
        },
    }
