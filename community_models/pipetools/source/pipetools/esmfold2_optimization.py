"""Batch-scoped ESMFold2 dispatch and auditable pinned-kit integration.

No import of torch or an optional accelerator occurs on the off path. A route is
frozen for the worker's lifetime: upstream drivers patch process-wide classes.
Validation profiles are explicit benchmark-only routes and never auto records.
"""
from __future__ import annotations

import gc
import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = 1
KIT = Path(__file__).with_name("_esmfold2_kit")
VALIDATION_ENV = "PARTIDESIGN_ESMFOLD2_VALIDATION_PROFILE"
PROFILES = {
    "host": {"sg", "swa", "pb"},
    "graphs": {"tg", "sg", "eg", "pb", "swa", "ls", "rg", "ax", "ro"},
    "exact": {"tg", "sg", "eg", "pb", "swa", "ls", "rg", "ax", "ro", "t3", "t5", "t6", "xte", "xln", "disto"},
    "fast": {"tg", "sg", "eg", "pb", "swa", "ls", "rg", "ax", "af", "ro", "kd", "dit", "t3", "t5", "t10", "t15", "tx", "xln", "disto"},
    "memory": {"sg", "swa", "pb", "ax", "ro", "t5", "x2b", "x3", "x6", "x7", "x8", "x10", "x4"},
    "mk": {"sg", "swa", "pb", "mk"},
}


DEFAULT_CONFIG = {"mode": "auto", "disable": []}
LEVERS = frozenset("tg sg eg pb t1 t3 t5 t6 t10 mk ls rg ax af t15 xte t16 tx disto xln ro kd dit x2b x3 x6 x7 x8 x10 x4 swa fc".split())
_UNREGISTERED_SIGNATURE = object()


def validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict) or set(config) - set(DEFAULT_CONFIG):
        raise ValueError("ESMFold2 optimization has unknown or malformed fields")
    result = {**DEFAULT_CONFIG, **config}
    if result["mode"] not in {"off", "exact", "fast", "auto"}:
        raise ValueError("ESMFold2 optimization mode is invalid")
    disabled = result["disable"]
    if (not isinstance(disabled, list) or any(not isinstance(v, str) or v not in LEVERS for v in disabled)
            or len(set(disabled)) != len(disabled)):
        raise ValueError("ESMFold2 optimization disable must contain unique supported levers")
    result["disable"] = list(disabled)
    return result



def _kit_module(name: str) -> Any:
    # The pinned upstream uses sibling absolute imports. Expose only this
    # vendored directory; refuse preloaded modules from any other source.
    previous = sys.modules.get(name)
    if previous is not None and Path(previous.__file__).resolve().parent != KIT.resolve():
        raise RuntimeError(f"Foreign ESMFold2 optimization module already loaded: {name}")
    if str(KIT) not in sys.path:
        sys.path.insert(0, str(KIT))
    return importlib.import_module(name)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def capability(torch: Any) -> dict[str, Any]:
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True, timeout=10).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = "unavailable"
    return {"compute_capability": [prop.major, prop.minor], "name": prop.name,
            "uuid": str(getattr(prop, "uuid", "unavailable")), "memory_bytes": prop.total_memory,
            "shared_memory_per_block": prop.shared_memory_per_block_optin,
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "triton": _package_version("triton"), "driver": driver,
            "structure_dtype": "float32", "esmc_dtype": "bfloat16"}



def select_route(
    config: dict[str, Any],
    caps: dict[str, Any],
    model_mode: str,
) -> tuple[str, list[str]]:
    mode = config.get("mode", "auto")
    if mode == "off":
        return "off", ["optimization explicitly disabled"]
    if model_mode != "fast":
        return "off", ["optimization is available only for the Fast checkpoint"]

    profile = os.environ.get(VALIDATION_ENV, "").strip()
    if profile:
        if profile not in PROFILES:
            raise ValueError(f"Unknown {VALIDATION_ENV}: {profile}")
        return profile, ["explicit optimization profile"]

    if mode == "exact":
        return "exact", ["exact optimization profile requested"]
    if mode == "auto":
        return "graphs", ["automatic route uses the tested graph composition"]
    if mode == "fast":
        return "fast", ["Fast optimization profile requested"]
    raise ValueError(f"unsupported optimization mode: {mode!r}")


class OptimizationSession:
    def __init__(
        self,
        model: Any,
        builder: Any,
        torch: Any,
        config: dict[str, Any],
        model_mode: str,
    ):
        self.model, self.torch, self.config = model, torch, config
        self.capability = capability(torch)
        self.route, self.reasons = select_route(config, self.capability, model_mode)
        self.modules: dict[str, Any] = {}
        self.installation: dict[str, Any] = {}
        self.disabled: dict[str, str] = {}
        self.levers: set[str] = set()
        self._last_signature = None
        self._generation_shapes: set[Any] = set()
        self._base_memory = int(torch.cuda.memory_allocated())
        self._request_count = 0
        self._allocation_retries = 0
        self._graph_capture_failed_signature = None
        self._attempt_signature_registered = False
        self._retrying = False
        if self.route != "off":
            self._install(builder)

    def _module(self, name: str) -> Any:
        result = _kit_module(name)
        self.modules[name] = result
        return result

    def _optional_provider(self, lever: str, callback: Any) -> Any:
        """Install one provider-backed lever without disabling the other fast paths."""
        try:
            return callback()
        except (ImportError, ModuleNotFoundError) as error:
            reason = (
                f"optional provider unavailable: {type(error).__name__}"
                + (f": {error.name}" if getattr(error, "name", None) else "")
            )
            self.disabled[lever] = reason
            return {"enabled": False, "reason": reason}

    def _install(self, builder: Any) -> None:
        fl = set(PROFILES[self.route]) - set(self.config.get("disable", ()))
        for flag in self.config.get("disable", ()):
            self.disabled[flag] = "explicit per-lever disable"
        # One owner for the Transition; the Hopper binary is never offered to
        # Ada/Blackwell. The provider still validates stack/shape at each call.
        if "t15" in fl and self.capability["compute_capability"] == [9, 0]:
            fl.remove("t15")
            if "t16" not in self.config.get("disable", ()):
                fl.add("t16")
        if self.capability["shared_memory_per_block"] < 160 * 1024:
            for flag in ("t3", "t6"):
                if flag in fl:
                    fl.remove(flag); self.disabled[flag] = "shared-memory capacity below upstream tile requirement"
            if self.route == "exact" and "t1" not in self.config.get("disable", ()):
                fl.add("t1")
        if "ro" not in fl:
            for flag in ("kd", "dit"):
                if flag in fl:
                    fl.remove(flag); self.disabled[flag] = "requires rollout lever ro"
        if "tg" not in fl:
            for flag in ("ls", "rg"):
                if flag in fl:
                    fl.remove(flag); self.disabled[flag] = "requires trunk graph lever tg"
        if "ls" not in fl:
            fl.discard("rg")
        # Do not inherit upstream engineering environment overrides silently.
        inherited = sorted(k for k in os.environ if k.startswith("EF2_"))
        if inherited:
            raise RuntimeError(f"Remove upstream EF2_* overrides; use the recorded PartiDesign selector: {inherited}")
        # No user token/VRAM caps. Capture is attempted on the actual runtime
        # device; allocation OOM is handled at the complete-request boundary.
        eo = self._module("ef2_opt")
        eo.CFG.graph_budget_tokens = 0
        eo.CFG.graph_budget_tokens_trunk = eo.CFG.graph_budget_tokens_encoder = eo.CFG.graph_budget_tokens_sampler = -1
        eo.CFG.recycle_graph_max_tokens = sys.maxsize
        eo.CFG.lru_trunk = eo.CFG.lru_generic = 1
        eo.CFG.lru_sampler = 8
        eo.CFG.lru_pair_bias = 1
        eo.CFG.graph_capture = self.route not in {"host", "memory", "mk"}
        # Keep baseline pair chunk=64 until its own attribution/gates pass.
        self.installation["pair_chunk_size"] = 64
        self.installation["graph_policy"] = "no_token_or_fixed_memory_limit"
        self.installation["graph_capture"] = eo.CFG.graph_capture
        self.installation["graph_capture_requested"] = eo.CFG.graph_capture
        if "mk" in fl:
            self.installation["mk"] = self._module("ef2_mk_sampler").enable(self.model, hoist="overlap")
        w4 = fl & {"t1", "t3", "t5", "t6", "t10"}
        if w4:
            self.installation["w4"] = self._module("ef2_w4").enable(self.model, transition_nolin="t1" in fl, tiles=True if "t3" in fl else None, weight_cache="t5" in fl, transition_rowblock="t6" in fl, transition_fused="t10" in fl)
        if "tx" in fl:
            self.installation["tx"] = self._optional_provider(
                "tx", lambda: self._module("ef2_w4").enable_tx("fast")
            )
        if fl & {"ax", "af"}:
            self.installation["atom"] = self._module("ef2_atom").install(self.model, hoist_adaln="ax" in fl, prefix_varlen="ax" in fl, rope_tables="ax" in fl, skip_copies="ax" in fl, fused_block="af" in fl, seg_mean="af" in fl, gemm="bf16" if "af" in fl else "fp32")
        self.installation["core"] = eo.install(self.model, trunk_graphs="tg" in fl, sampler_graphs="sg" in fl, encoder_graphs="eg" in fl, pair_bias_cache="pb" in fl, swa_mask_cache="swa" in fl, esmc_cache=False, builder=None, loop_static="ls" in fl, recycle_graph="rg" in fl)
        if "t15" in fl:
            self.installation["t15"] = self._optional_provider(
                "t15",
                lambda: self._module("ef2_pair_v2").install(
                    self.model, trunk=True, msa=False
                ),
            )
        if "t16" in fl:
            self.installation["t16"] = self._optional_provider(
                "t16",
                lambda: self._module("ef2_transition_cute").install(
                    self.model, trunk=True, msa=False
                ),
            )
        if "xte" in fl:
            self.installation["xte"] = self._module("ef2_xte").install(self.model)
        if "disto" in fl:
            self.installation["disto"] = self._module("ef2_hoist").install(self.model, disto=True)
        if "xln" in fl:
            self.installation["xln"] = self._module("ef2_xln").install(self.model)
        xl = fl & {"x2b", "x3", "x6", "x7", "x8", "x10", "x4"}
        if xl:
            self.installation["xl"] = self._module("ef2_xl").apply(self.model, loopfree="x2b" in fl, free="z,relpos,lmz" if "x2b" in fl else "", cond="lean" if "x3" in fl else "", lmpair="x6" in fl, relpos="x7" in fl, initlean="x8" in fl, distocpu="x10" in fl, esmc_offload="x4" in fl, esmc_min_tok=0, esmc_stream="pinned")
        if "ro" in fl:
            fast = "dit" in fl
            self.installation["dit"] = self._module("ef2_dit").install(self.model, rollout=True, graph=True, static_once=True, bias_f32=True, lru=1, kabsch="device" if "kd" in fl else "torch", dit=fast, gemm="bf16" if fast else "fp32", cond="bf16" if fast else "fp32", attn="sdpa", attn_precision="bf16" if fast else "ieee", fused_ew=fast)
        self.levers = fl
        self.torch.cuda.synchronize()
        self._base_memory = int(self.torch.cuda.memory_allocated())

    def before_request(self, features: dict[str, Any]) -> None:
        self._attempt_signature_registered = True
        if self.route == "off":
            return
        signature = tuple((name, tuple(value.shape), str(value.dtype), str(value.device)) for name, value in sorted(features.items()) if hasattr(value, "shape"))
        # A generation holds complete coupled graph/static buffers. Evict the
        # whole generation, never individual shared-pool entries.
        if signature not in self._generation_shapes and bool(self._generation_shapes):
            self.clear("owner_shape_change")
        if (not getattr(self, "_retrying", False) and signature != self._last_signature
                and getattr(self, "_graph_capture_failed_signature", None) is not None):
            self.modules["ef2_opt"].CFG.graph_capture = bool(self.installation["graph_capture_requested"])
            if "ef2_dit" in self.modules:
                self.modules["ef2_dit"]._CFG["graph"] = True
            self._graph_capture_failed_signature = None
            self.installation["graph_capture"] = self.installation["graph_capture_requested"]
        self._generation_shapes.add(signature)
        self._last_signature = signature

    def _is_oom(self, error: BaseException) -> bool:
        oom_type = getattr(getattr(self.torch, "cuda", None), "OutOfMemoryError", None)
        return oom_type is not None and isinstance(error, oom_type)

    def _disable_capture_for_retry(self) -> None:
        core = self.modules.get("ef2_opt")
        if core is not None:
            core.CFG.graph_capture = False
        if "ef2_dit" in self.modules:
            self.modules["ef2_dit"]._CFG["graph"] = False
        self.installation["graph_capture"] = False

    def _run_with_allocation_retry(self, call: Any) -> Any:
        """Run one request attempt and retry one CUDA allocation failure.

        The callback is deliberately larger than the model forward.  Input
        preparation and ESM-C can allocate on the owner GPU before the model
        installs a shape, so those stages must share the same bounded retry
        boundary and capture fallback.
        """

        core = self.modules.get("ef2_opt")
        capture_requested = bool(self.installation.get("graph_capture_requested", False))
        if core is None or not (getattr(core.CFG, "graph_capture", False) or capture_requested):
            return call()
        import random
        import numpy as np

        cpu_rng = self.torch.get_rng_state()
        cuda_rng = self.torch.cuda.get_rng_state()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        self._retrying = False
        self._attempt_signature_registered = False
        try:
            return call()
        except Exception as error:
            if not self._is_oom(error):
                raise
            if not getattr(core.CFG, "graph_capture", False):
                raise
            # The failed callback traceback can retain feature/LM tensors.
            # Drop every exception link before clearing graphs and retrying.
            error.__traceback__ = None
            error.__context__ = None
            error.__cause__ = None
            failed_signature = (
                self._last_signature
                if self._attempt_signature_registered
                else _UNREGISTERED_SIGNATURE
            )
            del error
        self.clear("owner_allocation_oom")
        gc.collect()
        empty_cache = getattr(self.torch.cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        self._disable_capture_for_retry()
        self._graph_capture_failed_signature = failed_signature
        self._allocation_retries += 1
        self.torch.set_rng_state(cpu_rng)
        self.torch.cuda.set_rng_state(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        self._retrying = True
        self._attempt_signature_registered = False
        try:
            return call()
        finally:
            self._retrying = False

    def request(self, call: Any) -> Any:
        """Run one logical request, with an optional bounded OOM retry."""

        self._request_count = getattr(self, "_request_count", 0) + 1
        return self._run_with_allocation_retry(call)

    def clear(self, reason: str) -> None:
        if "ef2_opt" in self.modules:
            self.torch.cuda.synchronize()
            self.modules["ef2_opt"].clear_graphs(self.model, reason=reason)
        self._generation_shapes.clear()

    def forward(self, call: Any) -> Any:
        """Backward-compatible model-only entry point for focused callers."""

        return self._run_with_allocation_retry(call)

    def after_request(self) -> None:
        # No estimated memory budget; preserve the current complete generation.
        # A shape change or actual allocation failure clears it atomically.
        return

    def error(self) -> None:
        if self.route != "off":
            # Synchronization errors propagate; a poisoned context never serves
            # the next candidate under the baseline/fallback label.
            self.clear("owner_candidate_error")

    def _lever_status(self, stats: dict[str, Any]) -> dict[str, Any]:
        status = {flag: {"state": "installed_not_observed"} for flag in self.levers}
        status.update({flag: {"state": "disabled", "reason": reason}
                       for flag, reason in self.disabled.items()})
        # Installation requests are not proof that a provider served the GPU.
        counters = {
            "tx": ("ef2_w4", ("tx_calls",), ("tx_refused", "tx_class_stock")),
            "t1": ("ef2_w4", ("t1_calls",), ()),
            "t3": ("ef2_w4", ("tiles_applied",), ()),
            "t5": ("ef2_w4", ("t5_trimul_calls", "t5_transition_calls"), ()),
            "t6": ("ef2_w4", ("t6_calls",), ()),
            "t10": ("ef2_w4", ("t10_calls",), ()),
            "t15": ("ef2_pair_v2", ("t15_calls",), ("t15_refused",)),
            "xln": ("ef2_xln", ("xln_served",), ("xln_fallback",)),
            "af": ("ef2_atom", ("a5_enc_calls", "a5_dec_calls"), ("a5_layout_fallbacks",)),
            "dit": ("ef2_dit", ("dit_steps",), ()),
            "kd": ("ef2_dit", ("kabsch_device",), ()),
            "ro": ("ef2_dit", ("rollout_folds",), ()),
            "tg": ("ef2_opt", ("trunk_graph_replays",), ()),
            "rg": ("ef2_opt", ("recycle_graph_replays",), ()),
            "swa": ("ef2_opt", ("swa_mask_hits",), ()),
        }
        for flag, (module, served_names, refused_names) in counters.items():
            values = stats.get(module, {})
            served = sum(int(values.get(name, 0)) for name in served_names)
            refused = sum(int(values.get(name, 0)) for name in refused_names)
            if flag in self.levers or served:
                state = "partial" if served and refused else "served" if served else "refused" if refused else "installed_not_observed"
                status[flag] = {"state": state, "observed_calls": served, "fallback_calls": refused}
        w4 = self.installation.get("w4", {})
        if "t10" in self.levers and not w4.get("transition_fused", False):
            reason = w4.get("device", {}).get("disabled", {}).get("T10", "provider did not enable fused Transition")
            status["t10"] = {"state": "refused", "reason": reason, "fallback": "t1" if w4.get("transition_nolin") else "stock"}
        xln = self.installation.get("xln", {})
        if "xln" in self.levers and not xln.get("levers", {}).get("xln", False):
            status["xln"] = {"state": "refused", "reason": xln.get("refusal", "no eligible sites")}
        if "af" in self.levers and "ef2_atom" in self.modules:
            atom = self.modules["ef2_atom"]
            if not getattr(atom, "STATE", {}).get("flash_attn", False):
                status["af"]["attention"] = "existing dense attention; optional FlashAttention is unavailable"
        return status

    def evidence(self) -> dict[str, Any]:
        stats = {name: mod.stats() for name, mod in self.modules.items() if hasattr(mod, "stats")}
        invariant = self.modules["ef2_opt"].invariant_report() if "ef2_opt" in self.modules else {"ok": True}
        return {"schema": SCHEMA, "requested": self.config,
                "route": self.route, "capability": self.capability, "reasons": self.reasons,
                "selected_levers": sorted(self.levers), "disabled_levers": self.disabled,
                "lever_status": self._lever_status(stats),
                "installation": self.installation, "counters": stats, "pool_invariant": invariant,
                "requests": self._request_count, "allocation_retries": self._allocation_retries, "cached_shape_generations": len(self._generation_shapes),
                "allocated_bytes": int(self.torch.cuda.memory_allocated()),
                "reserved_bytes": int(self.torch.cuda.memory_reserved())}
