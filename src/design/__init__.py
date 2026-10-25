"""Standalone PartiDesign inference package with lazy public model entry points."""
from partidesign.cache import configure_runtime_caches as _configure_runtime_caches

_configure_runtime_caches()

__all__ = ["DesignRunner", "DesignRunResult", "run_design"]

def __getattr__(name):
    if name == "DesignRunResult":
        from .core.records import DesignRunResult
        return DesignRunResult
    if name in {"DesignRunner", "run_design"}:
        from .core.api import DesignRunner, run_design
        return {"DesignRunner": DesignRunner, "run_design": run_design}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
