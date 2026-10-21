from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .ccd_bonds import CCDBondCacheMetadata, validate_ccd_bond_cache
from .errors import AdapterError
from partidesign.cache import DEFAULT_CACHE_SUBDIRS
from partidesign.paths import (community_source, asset_root, cache_dir, offline_site, resource_root, code_root)


ESMFOLD2_CHECKPOINT_ASSETS = ("config.json", "model.safetensors")
ESMC_CHECKPOINT_ASSETS = ("config.json", "model.safetensors.index.json")
_CUDA_DEVICE_TOKEN = re.compile(
    r"(?:[0-9]+|GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}|"
    r"MIG-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12})"
)


def resolve_owner_cuda_visible_device(environment: Mapping[str, str]) -> str:
    """Resolve one physical CUDA owner for a standalone or DDP worker.

    ``CUDA_VISIBLE_DEVICES`` is an ordered physical-device namespace.  A
    multi-device parent selects by ``LOCAL_RANK``; a single-device parent has
    already scoped ownership and must keep that entry even when its local rank
    is non-zero.  When the variable is absent, CUDA's native physical ordering
    is used and local rank (or zero for standalone) is the selected device.
    """

    raw_rank = environment.get("LOCAL_RANK")
    if raw_rank is None:
        local_rank = 0
    else:
        try:
            local_rank = int(raw_rank)
        except ValueError as exc:
            raise AdapterError(f"Invalid LOCAL_RANK for owner GPU: {raw_rank!r}") from exc
        if local_rank < 0:
            raise AdapterError(f"Invalid LOCAL_RANK for owner GPU: {raw_rank!r}")

    raw_visible = environment.get("CUDA_VISIBLE_DEVICES")
    if raw_visible is None:
        return str(local_rank)
    if not raw_visible.strip():
        raise AdapterError("CUDA_VISIBLE_DEVICES explicitly hides every GPU")
    visible = [value.strip() for value in raw_visible.split(",")]
    if any(not value or _CUDA_DEVICE_TOKEN.fullmatch(value) is None for value in visible):
        raise AdapterError("CUDA_VISIBLE_DEVICES does not name a usable GPU set")
    if len(set(visible)) != len(visible):
        raise AdapterError("CUDA_VISIBLE_DEVICES contains duplicate GPU owners")
    if len(visible) == 1:
        return visible[0]
    if local_rank >= len(visible):
        raise AdapterError("LOCAL_RANK does not resolve one owner GPU")
    return visible[local_rank]


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    repo_root: Path
    model_root: Path
    cache_root: Path
    mpnn_python: Path
    proteinmpnn_root: Path
    ligandmpnn_root: Path
    esmfold2_root: Path | None = None
    esmfold2_source: Path | None = None
    esmfold2_transformers_source: Path | None = None
    esmfold2_python: Path | None = None
    esmfold2_fast_checkpoint: Path | None = None
    esmfold2_standard_checkpoint: Path | None = None
    esmfold2_esmc_checkpoint: Path | None = None
    esmfold2_ccd: Path | None = None
    esmfold2_ccd_bonds: Path | None = None
    esmfold2_common: Path | None = None

    def __post_init__(self) -> None:
        root = self.esmfold2_root or (self.model_root / "esmfold2")
        defaults = {
            "esmfold2_root": root,
            "esmfold2_source": (community_source("esmfold2", self.repo_root) if os.environ.get("PARTIDESIGN_MODEL_DIR") or resource_root() != code_root() else root / "source") / "esm",
            "esmfold2_transformers_source": (community_source("esmfold2", self.repo_root) if os.environ.get("PARTIDESIGN_MODEL_DIR") or resource_root() != code_root() else root / "source") / "transformers",
            "esmfold2_python": root / "env" / "bin" / "python",
            "esmfold2_fast_checkpoint": root / "checkpoint" / "fast",
            "esmfold2_standard_checkpoint": root / "checkpoint" / "standard",
            "esmfold2_esmc_checkpoint": root / "checkpoint" / "esmc-6b",
            "esmfold2_ccd": root / "common" / "ccd.pkl",
            "esmfold2_ccd_bonds": root / "common" / "ccd_bonds.sqlite3",
            "esmfold2_common": root / "common",
        }
        for name, value in defaults.items():
            if getattr(self, name) is None:
                normalized = (
                    value.expanduser().absolute()
                    if name == "esmfold2_python"
                    else value.resolve()
                )
                object.__setattr__(self, name, normalized)

    @classmethod
    def discover(cls, repo_root: Path) -> "RuntimePaths":
        model_root = Path(
            os.environ.get("PIPETOOLS_MODEL_ROOT", asset_root(repo_root) / "community_models")
        ).expanduser().resolve()
        cache_root = Path(
            os.environ.get("PIPETOOLS_CACHE_ROOT", cache_dir("pipetools", repo_root))
        ).expanduser().resolve()
        mpnn_python = _resolve_python(
            os.environ.get("PIPETOOLS_MPNN_PYTHON") or sys.executable,
            [model_root / "pipetools" / "env" / "bin" / "python"],
        )
        esmfold2_root = model_root / "esmfold2"
        esmfold2_python = _resolve_python(
            os.environ.get("PIPETOOLS_ESMFOLD2_PYTHON") or sys.executable,
            [esmfold2_root / "env" / "bin" / "python"],
        )
        return cls(
            repo_root=repo_root.resolve(),
            model_root=model_root,
            cache_root=cache_root,
            mpnn_python=mpnn_python,
            proteinmpnn_root=model_root / "proteinmpnn",
            ligandmpnn_root=model_root / "ligandmpnn",
            esmfold2_root=esmfold2_root,
            esmfold2_source=community_source("esmfold2", repo_root) / "esm",
            esmfold2_transformers_source=(
                community_source("esmfold2", repo_root) / "transformers"
            ),
            esmfold2_python=esmfold2_python,
            esmfold2_fast_checkpoint=esmfold2_root / "checkpoint" / "fast",
            esmfold2_standard_checkpoint=(
                esmfold2_root / "checkpoint" / "standard"
            ),
            esmfold2_esmc_checkpoint=esmfold2_root / "checkpoint" / "esmc-6b",
            esmfold2_ccd=esmfold2_root / "common" / "ccd.pkl",
            esmfold2_ccd_bonds=(
                esmfold2_root / "common" / "ccd_bonds.sqlite3"
            ),
            esmfold2_common=esmfold2_root / "common",
        )

    @property
    def proteinmpnn_source(self) -> Path:
        return community_source("proteinmpnn", self.repo_root)

    @property
    def ligandmpnn_source(self) -> Path:
        return community_source("ligandmpnn", self.repo_root)

    def subprocess_env(
        self,
        *,
        direct_network: bool = False,
        block_network: bool = False,
    ) -> dict[str, str]:
        if direct_network and block_network:
            raise AdapterError("subprocess network policy cannot be direct and blocked")
        env = os.environ.copy()
        package_src = self.repo_root / "src"
        python_paths = [
            str(offline_site(self.repo_root)),
            str(self.repo_root),
            str(package_src),
        ]
        # Runtime import ownership is explicit.  Inheriting an arbitrary host
        # PYTHONPATH would permit an unrelated loose module to shadow a
        # model or dependency while retaining the same resume key.
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for cache_name, relative in DEFAULT_CACHE_SUBDIRS.items():
            env[cache_name] = str((self.cache_root / relative).resolve())
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[key] = "1"
        if direct_network:
            for key in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "FTP_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
                "ftp_proxy",
                "MMSEQS_SERVICE_HOST_URL",
            ):
                env.pop(key, None)
            env["NO_PROXY"] = "*"
            env["no_proxy"] = "*"
        elif block_network:
            blackhole = "http://127.0.0.1:9"
            for key in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "FTP_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
                "ftp_proxy",
            ):
                env[key] = blackhole
            env["NO_PROXY"] = ""
            env["no_proxy"] = ""
            env["MMSEQS_SERVICE_HOST_URL"] = (
                blackhole + "/pipetools-fold-msa-search-forbidden"
            )
            env["PIPETOOLS_NETWORK_POLICY"] = "fold-offline"
        return env

    def esmfold2_checkpoint(self, mode: str) -> Path:
        if mode == "fast":
            return self.esmfold2_fast_checkpoint
        if mode == "standard":
            return self.esmfold2_standard_checkpoint
        raise AdapterError(f"Unsupported ESMFold2 mode: {mode!r}")

    def validate_esmfold2(self, mode: str) -> CCDBondCacheMetadata:
        if not self.esmfold2_python.is_file():
            raise AdapterError(
                "ESMFold2 Python is missing; set PIPETOOLS_ESMFOLD2_PYTHON"
            )
        if not (self.esmfold2_source / "esm").is_dir():
            raise AdapterError(
                f"Biohub ESM source is missing at {self.esmfold2_source}"
            )
        if not (
            self.esmfold2_transformers_source / "src" / "transformers"
        ).is_dir():
            raise AdapterError(
                "Biohub Transformers source is missing at "
                f"{self.esmfold2_transformers_source}"
            )
        checkpoint = self.esmfold2_checkpoint(mode)
        missing = [
            str(checkpoint / name)
            for name in ESMFOLD2_CHECKPOINT_ASSETS
            if not (checkpoint / name).is_file()
        ]
        missing.extend(
            str(self.esmfold2_esmc_checkpoint / name)
            for name in ESMC_CHECKPOINT_ASSETS
            if not (self.esmfold2_esmc_checkpoint / name).is_file()
        )
        index_path = self.esmfold2_esmc_checkpoint / "model.safetensors.index.json"
        if index_path.is_file():
            try:
                index = json.loads(index_path.read_text())
                weight_map = index["weight_map"]
                shard_names = sorted(set(weight_map.values()))
                if not isinstance(weight_map, dict) or not shard_names or not all(
                    isinstance(name, str) and Path(name).name == name
                    for name in shard_names
                ):
                    raise ValueError("invalid weight_map")
                missing.extend(
                    str(self.esmfold2_esmc_checkpoint / name)
                    for name in shard_names
                    if not (self.esmfold2_esmc_checkpoint / name).is_file()
                )
            except Exception as exc:
                raise AdapterError(
                    f"ESMC checkpoint index is invalid at {index_path}: {exc}"
                ) from exc
        if not self.esmfold2_ccd.is_file():
            missing.append(str(self.esmfold2_ccd))
        if not self.esmfold2_ccd_bonds.is_file():
            missing.append(str(self.esmfold2_ccd_bonds))
        components_cif = self.esmfold2_common / "components.cif"
        if not components_cif.is_file():
            missing.append(str(components_cif))
        if missing:
            raise AdapterError(
                "ESMFold2 runtime assets are incomplete: " + ", ".join(missing)
            )
        try:
            return validate_ccd_bond_cache(self.esmfold2_ccd_bonds)
        except Exception as exc:
            raise AdapterError(f"ESMFold2 CCD bond cache is invalid: {exc}") from exc

    def esmfold2_subprocess_env(self) -> dict[str, str]:
        """Build an offline, import-owned ESMFold2 worker environment."""

        env = self.subprocess_env(
            block_network=True,
        )
        env["PYTHONPATH"] = os.pathsep.join(
            (
                str(community_source("pipetools", self.repo_root)),
                str(offline_site(self.repo_root)),
                str(self.repo_root),
                str(self.repo_root / "src"),
                str(self.esmfold2_source),
                str(self.esmfold2_transformers_source / "src"),
            )
        )
        env["ESMCFOLD_CCD_PATH"] = str(self.esmfold2_ccd)
        env["HF_HOME"] = str(self.cache_root / "huggingface")
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["CUDA_VISIBLE_DEVICES"] = resolve_owner_cuda_visible_device(env)
        return env

def _resolve_python(explicit: str | None, candidates: list[Path]) -> Path:
    if explicit:
        # A venv's Python is normally a symlink.  Preserve the invocation path:
        # executing its resolved base interpreter can bypass pyvenv.cfg and the
        # model-owned site-packages entirely.
        return Path(explicit).expanduser().absolute()
    for candidate in candidates:
        if candidate.is_file():
            return candidate.expanduser().absolute()
    if not candidates:
        raise AdapterError("No model Python location is configured")
    # Keep the missing model-owned path so adapter preflight reports the exact
    # site setting to create or override.  Falling back to the Ray host Python
    # would silently break model-environment isolation.
    return candidates[0].expanduser().absolute()
