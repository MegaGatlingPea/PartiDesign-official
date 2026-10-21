"""Route process filesystem caches into the selected PartiDesign root.

This module is intentionally dependency-free and safe to import from package
initializers. It only configures filesystem cache locations. Model objects,
resident backend caches, and any disk reload policy remain owned by their
callers.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Final, Mapping


MODEL_ROOT_ENV: Final[str] = "PARTIDESIGN_MODEL_DIR"
CACHE_ENV_VARS: Final[tuple[str, ...]] = (
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TORCH_HOME",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "XDG_CACHE_HOME",
    "JAX_COMPILATION_CACHE_DIR",
    "NUMBA_CACHE_DIR",
    "PYTHONPYCACHEPREFIX",
)

# Keep each library's cache in a named child so that a cache produced by one
# runtime cannot be mistaken for another runtime's state. XDG_CACHE_HOME is
# the one exception: it is itself the common cache root and therefore points
# directly at ROOT/.cache.
DEFAULT_CACHE_SUBDIRS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "TRITON_CACHE_DIR": "triton",
        "TORCH_EXTENSIONS_DIR": "torch_extensions",
        "TORCH_HOME": "torch",
        "HF_HOME": "huggingface",
        "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
        "XDG_CACHE_HOME": ".",
        "JAX_COMPILATION_CACHE_DIR": "jax",
        "NUMBA_CACHE_DIR": "numba",
        "PYTHONPYCACHEPREFIX": "pycache",
    }
)


def _is_source_checkout(path: Path) -> bool:
    """Return whether path has the source layout used by this project."""

    return (
        (path / "pyproject.toml").is_file()
        and (path / "partidesign").is_dir()
        and (path / "src" / "design").is_dir()
    )


def _normalized_root(value: str | os.PathLike[str]) -> Path:
    """Resolve a caller-selected root without consulting HOME or CODEX_HOME."""

    return Path(value).expanduser().resolve()


def _default_root() -> Path:
    """Find a writable project root for source and installed execution.

    PARTIDESIGN_MODEL_DIR is the existing project-specific selection
    mechanism. A source checkout next to this module is preferred next; an
    installed wheel cannot safely write beside its package, so it falls back
    to the current working directory (or a checkout found above it).
    """

    selected = os.environ.get(MODEL_ROOT_ENV)
    if selected and _normalized_root(selected).is_dir():
        return _normalized_root(selected)

    module_root = Path(__file__).resolve().parents[1]
    if _is_source_checkout(module_root):
        return module_root

    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if _is_source_checkout(candidate):
            return candidate
    return current


def _cache_paths(root: Path) -> dict[str, Path]:
    cache_root = root / ".cache"
    return {
        name: cache_root / relative
        for name, relative in DEFAULT_CACHE_SUBDIRS.items()
    }


def configure_runtime_caches(root: str | os.PathLike[str] | None = None) -> Path:
    """Configure all supported filesystem caches below ROOT/.cache.

    The function is deliberately repeatable. A later explicit call can move
    the process from the source checkout to a selected model root before a
    model or worker subprocess is created. The returned path is the resolved
    common cache root; callers that need a particular location should use the
    corresponding environment variable after this function returns.

    The process environment is the inheritance boundary for child interpreters
    and distributed workers. sys.pycache_prefix is updated in the current
    interpreter as well because PYTHONPYCACHEPREFIX is only read by Python
    during interpreter startup.
    """

    selected_root = _normalized_root(root) if root is not None else _default_root()
    cache_root = (selected_root / ".cache").resolve()
    paths = _cache_paths(selected_root)

    # Creating the directories here gives every backend a usable destination
    # without importing any optional runtime (torch, JAX, transformers, etc.).
    for path in {cache_root, *paths.values()}:
        path.mkdir(parents=True, exist_ok=True)

    for name, path in paths.items():
        os.environ[name] = str(path.resolve())
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    return cache_root


__all__ = [
    "CACHE_ENV_VARS",
    "DEFAULT_CACHE_SUBDIRS",
    "MODEL_ROOT_ENV",
    "configure_runtime_caches",
]
