"""Standalone structure-only PartiScore public API (lazy exports)."""
from importlib import import_module

from partidesign.cache import configure_runtime_caches as _configure_runtime_caches

_configure_runtime_caches()

__all__ = ["PartiScorer", "ScoreResult", "InterfaceScore", "PreparedStructure",
           "prepare_structure", "source_partners", "validate_checkpoint"]

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = "api" if name in {"PartiScorer", "validate_checkpoint"} else (
        "records" if name in {"ScoreResult", "InterfaceScore"} else "features")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
