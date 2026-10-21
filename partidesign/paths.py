"""Local code, packaged resources, and explicitly selected model assets."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Iterator

ENV_MODEL_DIR = "PARTIDESIGN_MODEL_DIR"


def code_root() -> Path:
    """Root containing the installed src namespace or the source checkout."""
    return Path(__file__).resolve().parents[1]


def resource_root() -> Path:
    """Immutable configs/community source shipped with this distribution."""
    packaged = Path(__file__).resolve().parent / "_resources"
    return packaged if packaged.is_dir() else code_root()


def inference_config_dir() -> Path:
    return resource_root() / "configs"


def community_source(backend: str, repo_root: str | Path | None = None) -> Path:
    root = resource_root()
    if repo_root is not None and ENV_MODEL_DIR not in os.environ and root == code_root():
        root = Path(repo_root)
    return root / "community_models" / backend / "source"


def af3_source(repo_root: str | Path | None = None) -> Path:
    """Shared AlphaFold 3 SDK source for both AF3 model targets."""
    return community_source("af3", repo_root)


def af3_asset(target: str = "af3", repo_root: str | Path | None = None) -> Path:
    """Selected AF3 target asset root, preserving model-root overrides."""
    if target not in {"af3", "af3-la"}:
        raise ValueError(f"unknown AF3 target: {target}")
    return community_asset(target, repo_root)


def offline_site(repo_root: str | Path | None = None) -> Path:
    root = resource_root()
    if repo_root is not None and ENV_MODEL_DIR not in os.environ and root == code_root():
        root = Path(repo_root)
    return root / "community_models" / "offline_site"


def _is_checkout(root: Path) -> bool:
    return (
        (root / "pyproject.toml").is_file()
        and (root / "partidesign" / "cli.py").is_file()
        and (root / "src" / "design").is_dir()
    )


def model_dir(value: str | Path | None = None, *, invocation_dir: Path | None = None) -> Path:
    """CLI location: explicit argument, environment, then the source checkout."""
    base = (Path.cwd() if invocation_dir is None else invocation_dir).resolve()
    raw = value if value is not None else os.environ.get(ENV_MODEL_DIR)
    if raw is not None:
        path = Path(raw).expanduser()
        return (path if path.is_absolute() else base / path).resolve()
    for candidate in (base, *base.parents):
        if _is_checkout(candidate):
            return candidate
    installed_source = code_root()
    if _is_checkout(installed_source):
        return installed_source
    raise ValueError(
        "Cannot locate the PartiDesign repository root. Run from the repository "
        "or set --model-dir PATH / PARTIDESIGN_MODEL_DIR. "
        "Weights belong in ROOT/ckpts and ROOT/community_models."
    )


def asset_root(repo_root: str | Path | None = None) -> Path:
    """Native adapters preserve checkout assets unless a root is selected."""
    if os.environ.get(ENV_MODEL_DIR) is not None:
        return model_dir()
    return Path(repo_root).resolve() if repo_root is not None else code_root()


def community_asset(backend: str, repo_root: str | Path | None = None) -> Path:
    return asset_root(repo_root) / "community_models" / backend


def cache_dir(name: str, repo_root: str | Path | None = None) -> Path:
    return asset_root(repo_root) / ".cache" / name


@contextmanager
def using_model_dir(root: str | Path) -> Iterator[None]:
    """Propagate one normalized resource root to child processes and ranks."""
    previous = os.environ.get(ENV_MODEL_DIR)
    os.environ[ENV_MODEL_DIR] = str(Path(root).expanduser().resolve())
    from .cache import configure_runtime_caches
    # Read-only checks must not create an absent model root.
    if Path(root).expanduser().is_dir():
        configure_runtime_caches(root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ENV_MODEL_DIR, None)
        else:
            os.environ[ENV_MODEL_DIR] = previous
