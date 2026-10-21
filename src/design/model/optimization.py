"""Runtime switches for the standalone FM inference optimizations.

The optimized implementations are ordinary module paths.  They construct any
derived values inside forward from the current parameters, so dispatching only
needs to apply the user configuration and refresh the actual runtime
device/compile state.  Hardware, shape, batch, precision, and request-content
admission belongs to neither this module nor the production inference path.
"""

from collections.abc import Mapping
import logging
from typing import Any

import torch


DEFAULT_OPTIMIZATION = {
    "mode": "auto",
    "sdpa": True,
    "pair_bias": True,
    "pair_condition": True,
    "solver": True,
    "float_one_hot": True,
}

_MODES = frozenset(("off", "exact", "fast", "auto"))
_BOOLEAN_LEVERS = (
    "sdpa",
    "pair_bias",
    "pair_condition",
    "solver",
    "float_one_hot",
)


def validate_optimization_config(config: Mapping[str, Any]) -> None:
    """Validate the closed model.optimization configuration contract."""

    if not isinstance(config, Mapping) or set(config) != set(DEFAULT_OPTIMIZATION):
        raise ValueError(
            "model.optimization keys must be exactly "
            f"{sorted(DEFAULT_OPTIMIZATION)}"
        )
    if config["mode"] not in _MODES:
        raise ValueError("model.optimization.mode must be off, exact, fast, or auto")
    for key in _BOOLEAN_LEVERS:
        if type(config[key]) is not bool:
            raise TypeError(f"model.optimization.{key} must be boolean")


def _normalise_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


def _module_device(module: Any) -> torch.device:
    """Return the current module device without querying accelerator state."""

    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            return parameter.device
    buffers = getattr(module, "buffers", None)
    if callable(buffers):
        for buffer in buffers():
            return buffer.device
    return torch.device("cpu")


def select_optimization(
    config: Mapping[str, Any],
    *,
    capability: tuple[int, int] | None = None,
    head_dim: int | None = None,
    context_mode: str | None = None,
    compiled: bool = False,
    device: torch.device | str | None = None,
    # Kept as ignored compatibility keywords for callers from the pre-simplified
    # dispatcher.  They must never influence the production route.
    _validation_probe: bool = False,
    scope: Mapping[str, Any] | None = None,
    compile_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the configured route without a hardware or request whitelist.

    auto and fast select the same native optimized operations.  The distinction
    is retained in requested for diagnostics.  exact enables only the solver
    and floating one-hot path; attention and pair-time shortcuts remain
    disabled.  Module-level guards still make every optimized operation a no-op
    for training or autograd forwards.
    """

    del capability, head_dim, context_mode, _validation_probe, scope, compile_config
    validate_optimization_config(config)

    requested = str(config["mode"])
    selected = "fast" if requested in {"auto", "fast"} else requested
    is_compiled = bool(compiled)
    active = selected != "off"
    fast = selected == "fast"
    exact = selected == "exact"

    if not active:
        reason = "optimizations disabled by configuration"
    elif exact:
        reason = "exact route enables solver and floating one-hot only"
    else:
        reason = "native optimized operations enabled by configuration"

    return {
        "sdpa": bool(fast and config["sdpa"]),
        "pair_bias": bool(fast and config["pair_bias"]),
        "pair_condition": bool(fast and config["pair_condition"]),
        "solver": bool(active and config["solver"]),
        "float_one_hot": bool(active and config["float_one_hot"]),
        "requested": requested,
        "selected": selected,
        "compiled": is_compiled,
        "device": None if device is None else str(_normalise_device(device)),
        "reason": reason,
    }


def configure_optimization(
    model: Any,
    config: Mapping[str, Any],
    *,
    _actual_device: torch.device | str | None = None,
    # The old request-scope argument is accepted while callers transition, but
    # request contents are intentionally irrelevant to dispatch.
    scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply optimization flags to the FM network and solver.

    This function is called at startup and when the runtime device or compile
    marker changes.  It does not inspect a request batch or model parameter
    identities.
    """

    del scope
    validate_optimization_config(config)
    network = getattr(model, "nn", None)
    if network is None:
        raise AttributeError("inference model must expose an nn network")
    base_network = getattr(network, "_orig_mod", network)
    device = _normalise_device(
        _actual_device if _actual_device is not None else _module_device(base_network)
    )
    compiled = bool(getattr(model, "_denoiser_compiled", False))
    selected = select_optimization(
        config,
        compiled=compiled,
        device=device,
    )

    configure_attention = getattr(base_network, "configure_attention_optimization")
    configure_attention(
        sdpa=selected["sdpa"],
        fused_pair_bias=selected["pair_bias"],
        # Zero means no dispatcher-imposed row cap.  The bank reuses one pair
        # representation across layers and computes temporary projections from
        # the current LayerNorm/projection parameters on each forward.
        row_chunk_size=0,
        layer_group_size=max(1, int(getattr(base_network, "nlayers", 1))),
    )
    pair_builder = getattr(base_network, "pair_repr_builder", None)
    if pair_builder is not None and hasattr(pair_builder, "broadcast_time_condition"):
        pair_builder.broadcast_time_condition = selected["pair_condition"]
    modules = getattr(base_network, "modules", None)
    module_iter = modules() if callable(modules) else (base_network,)
    for module in module_iter:
        if hasattr(module, "float_one_hot"):
            module.float_one_hot = selected["float_one_hot"]
    solver = getattr(model, "solver", None)
    if solver is not None and hasattr(solver, "optimized"):
        solver.optimized = selected["solver"]

    selected["device"] = str(device)
    model._inference_optimization = selected
    model._inference_optimization_config = dict(config)
    model._inference_optimization_device = device
    return selected


def ensure_optimization_device(
    model: Any,
    device: torch.device | str,
    *,
    scope: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Refresh dispatch after a device move or lazy compile transition.

    The request seam calls this for every prepared sample, but the fast path is
    a pair of scalar state comparisons.  No batch values, parameters, buffers,
    or accelerator capabilities are scanned here.
    """

    del scope
    config = getattr(model, "_inference_optimization_config", None)
    if config is None:
        return None
    actual_device = _normalise_device(device)
    compiled = bool(getattr(model, "_denoiser_compiled", False))
    current = getattr(model, "_inference_optimization", None)
    needs_refresh = (
        not isinstance(current, Mapping)
        or current.get("device") != str(actual_device)
        or bool(current.get("compiled")) != compiled
    )
    if needs_refresh:
        decision = configure_optimization(
            model,
            config,
            _actual_device=actual_device,
        )
        logging.getLogger(__name__).info(
            "FM optimization on actual inference device: %s", decision
        )
        return decision
    return current


__all__ = [
    "DEFAULT_OPTIMIZATION",
    "configure_optimization",
    "ensure_optimization_device",
    "select_optimization",
    "validate_optimization_config",
]
