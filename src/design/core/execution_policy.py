"""Resolve the requested FM execution policy once, before generation outputs."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from omegaconf import open_dict

from .compile_policy import validate_compile_enabled
from .records import DesignConfigError


AUTO_COMPILE_SAMPLE_THRESHOLD = 100


@dataclass(frozen=True)
class ExecutionPolicy:
    requested_backend: str
    requested_compile: bool | str
    resolved_backend: str
    resolved_compile: bool
    reason: str
    fm_samples: int

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


def count_fm_samples(requests: Sequence[Any], search: Mapping[str, Any]) -> int:
    """Count selected, fully expanded requests and their Best-of-N replicas."""
    replicas = 1
    if search.get("algorithm") == "best_of_n":
        replicas = search["best_of_n"]["replicas"]
        if type(replicas) is not int or replicas < 1:
            raise DesignConfigError("search.best_of_n.replicas must be a positive integer")
    return len(requests) * replicas


def resolve_execution_policy(
    model: Mapping[str, Any], requests: Sequence[Any], search: Mapping[str, Any],
    *, bioir_probe: Callable[[], Any] | None = None,
) -> ExecutionPolicy:
    """Only a dedicated optional-backend startup error permits auto fallback.

    Inference errors, OOM, input errors, and unexpected startup failures propagate.
    """
    backend = model.get("execution_backend", "auto")
    enabled = validate_compile_enabled(model["compile"]["enabled"])
    samples = count_fm_samples(requests, search)
    buckets = model["compile"].get("token_buckets")
    if backend not in {"auto", "native", "bioir"}:
        raise DesignConfigError("model.execution_backend must be auto, native, or bioir")
    if backend == "bioir" and (enabled is True or buckets is not None):
        raise DesignConfigError("BioIR conflicts with native compilation or token buckets")

    def decision(resolved: str, compiled: bool, reason: str) -> ExecutionPolicy:
        return ExecutionPolicy(backend, enabled, resolved, compiled, reason, samples)

    if backend != "bioir" and enabled != "auto":
        return decision("native", enabled, "explicit native compile setting")
    if backend != "bioir" and buckets is not None:
        return decision("native", True, "explicit token bucket compile policy")
    if backend != "bioir" and samples >= AUTO_COMPILE_SAMPLE_THRESHOLD:
        return decision("native", True, "FM sample count >=100 selects native dynamic compilation")
    if backend == "native":
        return decision("native", False, "FM sample count <100 selects native eager execution")

    from ..model.bioir import BioIRStartupError, probe_bioir_runtime
    probe = probe_bioir_runtime if bioir_probe is None else bioir_probe
    try:
        probe()
    except BioIRStartupError as error:
        if backend == "bioir":
            raise DesignConfigError(f"explicit BioIR startup failed: {error}") from error
        return decision("native", False, f"BioIR startup unavailable: {error}")
    return decision("bioir", False, "explicit BioIR backend" if backend == "bioir" else "FM sample count <100 and BioIR startup probe passed")


def apply_execution_policy(config: Any, policy: ExecutionPolicy) -> None:
    """Attach actual settings and small runtime provenance to the saved config."""
    config.model.execution_backend = policy.resolved_backend
    config.model.compile.enabled = policy.resolved_compile
    with open_dict(config.model):
        config.model.execution_policy = policy.metadata()
