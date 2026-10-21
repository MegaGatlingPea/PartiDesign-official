"""Narrow configuration objects used by the offline reward adapters.

This vendored compatibility layer intentionally supports only the two pieces
PartiDesign calls: one-sequence MPNN and ESMFold2.  General PipeTools task,
Protenix, RF3, Ray, and MSA configuration do not belong in this release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SequenceConfig:
    model: str = "gt"
    count: int = 1
    cpus: float = 8.0
    gpus: float = 1.0
    options: dict[str, Any] = field(default_factory=dict)


def _default_esmfold2_optimization() -> dict[str, Any]:
    return {
        "mode": "auto",
        "disable": [],
    }


@dataclass(slots=True)
class ESMFold2Config:
    mode: str = "fast"
    samples: int = 1
    cpus: float = 4.0
    gpus: float = 1.0
    seed: int = 101
    loops: int = 20
    steps: int = 100
    lm_dropout: float = 0.3
    lm_mask_pct: float | None = None
    esmc_attention_backend: str = "sdpa"
    context_embedding_cache: bool | None = None
    timeout_seconds: int = 7200
    optimization: dict[str, Any] = field(default_factory=_default_esmfold2_optimization)


def effective_esmfold2_lm_mask_pct(config: ESMFold2Config) -> float:
    """Resolve the deterministic cache-safe masking default."""

    return 0.0 if config.lm_mask_pct is None else config.lm_mask_pct


__all__ = [
    "ESMFold2Config",
    "SequenceConfig",
    "effective_esmfold2_lm_mask_pct",
]
