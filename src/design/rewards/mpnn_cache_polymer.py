"""Persistent NA-MPNN and CyclicMPNN backends.

The reward workers in :mod:`src.design.rewards` deliberately load one model
per subprocess invocation.  Validation may visit many structures in one run,
so this module keeps the *constructed model object* resident and reuses it for
subsequent request batches.  The native workers remain the authority for
candidate parsing and sampling semantics; this module only supplies the
resident-model lifecycle around them.

``run_batch`` keeps the small worker-shaped result protocol.  ``generate`` and
``validate_sequences_esm`` are convenience adapters for the validation
sequence planner and return ``(sequences, failures)`` mappings.  Request
values (seed, temperature, fixed positions, and bias inputs) are intentionally
excluded from the cache identity.
"""

from __future__ import annotations

import gc
import importlib
import json
import contextlib
import os
import random
import sys
import time
from pathlib import Path
from partidesign.paths import community_asset, community_source, ENV_MODEL_DIR
from typing import Any, Callable, Iterable, Mapping


class MPNNCacheError(RuntimeError):
    """Base error for the persistent MPNN adapters."""


class MPNNCacheConfigurationError(MPNNCacheError):
    """Raised when a backend configuration cannot be represented safely."""


class MPNNCacheSystemError(MPNNCacheError):
    """Raised when a resident model cannot be loaded or maintained."""


ModelLoader = Callable[..., Any]
Sampler = Callable[[Any, Any], str]
ModelFactory = Callable[[Any], Any]
BiasApplier = Callable[[Any, str, str], Any]


_REQUEST_ONLY_KEYS = frozenset(
    {
        "seed",
        "temperature",
        "count",
        "num",
        "batch",
        "omit_aas",
        "rm_aa",
        "bias_aa_json",
    }
)


def _reward_sibling(name: str) -> Any:
    """Import a sibling both as a package and from the JSONL worker process.

    ``mpnn_cache_worker.py`` is launched by path, so it first imports this
    module as top-level ``mpnn_cache_polymer``.  The native workers themselves
    use package-relative imports; make the repository package visible before
    importing them in that mode.
    """

    if __package__:
        return importlib.import_module(f".{name}", __package__)
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(f"src.design.rewards.{name}")


def _settings_value(value: Any) -> Any:
    """Return a hashable structural model-settings value.

    Cache admission follows the configured model path and settings directly.
    It deliberately does not inspect file contents or filesystem metadata.
    Request-local values such as seeds and temperatures remain excluded.
    """

    if isinstance(value, Mapping):
        return tuple(
            (str(key), _settings_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _REQUEST_ONLY_KEYS
        )
    if isinstance(value, (list, tuple)):
        return tuple(_settings_value(item) for item in value)
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _resolve_config_path(value: Any, root: Path | None) -> Any:
    if root is None or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(_resolve_config_path(item, root) for item in value)
    path = Path(str(value)).expanduser()
    return str((root / path).resolve()) if not path.is_absolute() else str(path.resolve())


def _cache_settings(
    family: str, device: str, config: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Build direct structural model settings for the resident cache."""

    return (str(family), str(device), _settings_value(config))


def _seed_everything(seed: Any) -> None:
    """Mirror the NA worker's seed contract for CyclicMPNN too."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class _ResidentModelCache:
    """Small per-backend resident-object cache with observable counters."""

    def __init__(self) -> None:
        self._models: dict[tuple[Any, ...], Any] = {}
        self.model_load_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def get_or_load(
        self, settings: tuple[Any, ...], loader: Callable[[], Any]
    ) -> tuple[Any, bool]:
        if settings in self._models:
            self.cache_hits += 1
            return self._models[settings], False
        self.cache_misses += 1
        try:
            model = loader()
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise MPNNCacheSystemError(
                "cannot load resident MPNN model for the configured settings: "
                f"{type(exc).__name__}: {detail}"
            ) from exc
        self._models[settings] = model
        self.model_load_count += 1
        return model, True

    def stats(self) -> dict[str, Any]:
        return {
            "model_load_count": self.model_load_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "resident_models": len(self._models),
        }

    def clear(self) -> None:
        models = tuple(self._models.values())
        self._models.clear()
        for model in models:
            # Do not require model implementations to expose a cleanup API.
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class _Backend:
    family = "mpnn"

    def __init__(self, *, device: str = "cuda:0") -> None:
        self.device = str(device)
        self._cache = _ResidentModelCache()

    @property
    def model_load_count(self) -> int:
        return self._cache.model_load_count

    @property
    def cache_hits(self) -> int:
        return self._cache.cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache.cache_misses

    def cache_stats(self) -> dict[str, Any]:
        return self._cache.stats()

    def close(self) -> None:
        self._cache.clear()

    @staticmethod
    def _result_maps(result: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
        sequences: dict[str, str] = {}
        failures: dict[str, str] = {}
        for item in result.get("results", ()):
            if not isinstance(item, Mapping):
                continue
            candidate_id = str(item.get("candidate_id", ""))
            if item.get("status") == "ok":
                sequences[candidate_id] = str(item["sequence"])
            else:
                failures[candidate_id] = str(item.get("error", "unknown MPNN failure"))
        return sequences, failures


class NAMPNNBackend(_Backend):
    """Persistent adapter around ``na_mpnn_worker``'s native sampler."""

    family = "na_mpnn"

    def __init__(
        self,
        repo_root: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        *,
        device: str = "cuda:0",
        loader: ModelLoader | None = None,
        sampler: Sampler | None = None,
    ) -> None:
        super().__init__(device=device)
        self.repo_root = Path(repo_root).expanduser().resolve() if repo_root else None
        self.config: dict[str, Any] = dict(config or {})
        self._loader = loader
        self._sampler = sampler

    def _settings(self, override: Mapping[str, Any] | None) -> dict[str, Any]:
        settings = dict(self.config)
        if override is not None:
            settings.update(dict(override))
        root = self.repo_root or Path(__file__).resolve().parents[3]
        asset_root = Path(
            settings.get("asset_root") or community_asset("na_mpnn", root)
        ).expanduser()
        settings.setdefault("_repo_root", str(root))
        settings.setdefault("asset_root", str(asset_root))
        settings.setdefault("checkpoint", str(asset_root / "checkpoint" / "s_19137.pt"))
        source_root = community_source("na_mpnn", root)
        settings.setdefault("source_path", str(source_root))
        settings.setdefault(
            "source_files",
            [
                str(source_root / "inference" / "data_utils.py"),
                str(source_root / "inference" / "model_utils.py"),
                str(source_root / "inference" / "run.py"),
            ],
        )
        for name in ("asset_root", "checkpoint", "source_path", "source_files"):
            if name in settings:
                settings[name] = _resolve_config_path(settings[name], Path.cwd() if ENV_MODEL_DIR in os.environ else self.repo_root)
        return settings

    def _load(self, settings: Mapping[str, Any]) -> Any:
        native = _reward_sibling("na_mpnn_worker")
        if self._loader is not None:
            loaded = self._loader(
                self.repo_root, device=self.device, config=dict(settings)
            )
            return loaded
        load_na_mpnn_model = _reward_sibling("na_mpnn_runtime").load_na_mpnn_model

        # The checked-in NA runtime owns the fixed, validated model inventory.
        # A custom checkpoint is accepted only through an explicit test or
        # application loader; silently ignoring it would make cache identity
        # incorrect.
        configured_checkpoint = settings.get("checkpoint")
        if configured_checkpoint is not None:
            asset_root = Path(
                settings.get("asset_root")
                or community_asset("na_mpnn", self.repo_root)
            )
            default_checkpoint = asset_root / "checkpoint" / "s_19137.pt"
            if Path(str(configured_checkpoint)).expanduser().resolve() != default_checkpoint.resolve():
                raise MPNNCacheConfigurationError(
                    "NA-MPNN native loader has a fixed validated checkpoint; "
                    "use an explicit loader for another checkpoint"
                )
        loaded = load_na_mpnn_model(self.repo_root, device=self.device)
        return loaded

    def _get_model(
        self, settings: Mapping[str, Any]
    ) -> tuple[Any, bool, tuple[Any, ...]]:
        cache_settings = _cache_settings(self.family, self.device, settings)
        model, loaded = self._cache.get_or_load(
            cache_settings, lambda: self._load(settings)
        )
        return model, loaded, cache_settings

    def run_batch(
        self,
        payload: Mapping[str, Any],
        *,
        config_override: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        native = _reward_sibling("na_mpnn_worker")

        requests = native._validate_batch(payload)
        settings = self._settings(config_override)
        loaded, newly_loaded, _cache_settings_used = self._get_model(settings)
        if self._sampler is None:
            results = native._run_loaded_batch(loaded, requests)
        else:
            results = []
            for raw_request in requests:
                try:
                    request = native._candidate_request(raw_request)
                    sequence = self._sampler(loaded, request)
                    results.append(
                        {
                            "candidate_id": request.candidate_id,
                            "status": "ok",
                            "sequence": str(sequence),
                        }
                    )
                except (
                    native.NAMPSequenceError,
                    ValueError,
                    TypeError,
                    OSError,
                ) as exc:
                    candidate_id = (
                        str(raw_request.get("candidate_id", ""))
                        if isinstance(raw_request, Mapping)
                        else ""
                    )
                    results.append(
                        {
                            "candidate_id": candidate_id,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        return {
            "schema_version": 1,
            "model_loads": int(newly_loaded),
            "model_load_count": self.model_load_count,
            "cache_hit": not newly_loaded,
            "results": results,
        }

    def generate(
        self,
        requests: Iterable[Mapping[str, Any]],
        *,
        config_override: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        result = self.run_batch(
            {"schema_version": 1, "requests": list(requests)},
            config_override=config_override,
        )
        return self._result_maps(result)

    def validate_sequences_esm(
        self,
        payload_or_requests: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        *,
        config_override: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Validation-facing alias returning ``(sequences, failures)``."""

        if isinstance(payload_or_requests, Mapping) and "requests" in payload_or_requests:
            result = self.run_batch(payload_or_requests, config_override=config_override)
            return self._result_maps(result)
        return self.generate(payload_or_requests, config_override=config_override)


class CyclicMPNNBackend(_Backend):
    """Persistent adapter around the native CyclicMPNN sampling operations."""

    family = "cyclicmpnn"

    def __init__(
        self,
        repo_root: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        *,
        device: str = "cuda:0",
        model_factory: ModelFactory | None = None,
        bias_applier: BiasApplier | None = None,
    ) -> None:
        super().__init__(device=device)
        self.repo_root = Path(repo_root).expanduser().resolve() if repo_root else None
        self.config: dict[str, Any] = dict(config or {})
        self._model_factory = model_factory
        self._bias_applier = bias_applier

    def _settings(self, override: Mapping[str, Any] | None) -> dict[str, Any]:
        settings = dict(self.config)
        if override is not None:
            settings.update(dict(override))
        for name in ("checkpoint",):
            if name in settings and settings[name]:
                settings[name] = _resolve_config_path(settings[name], Path.cwd() if ENV_MODEL_DIR in os.environ else self.repo_root)
        return settings

    @staticmethod
    def _model_config(settings: Mapping[str, Any]) -> dict[str, Any]:
        required = ("model_name", "base_model_name", "checkpoint", "backbone_noise")
        missing = [name for name in required if name not in settings]
        if missing:
            raise MPNNCacheConfigurationError(
                f"CyclicMPNN config is missing keys: {', '.join(missing)}"
            )
        return {
            "weights": "original",
            "model_name": str(settings["model_name"]),
            "base_model_name": str(settings["base_model_name"]),
            "model_weights_path": str(settings["checkpoint"]),
            "backbone_noise": float(settings["backbone_noise"]),
        }

    def _load(self, settings: Mapping[str, Any]) -> Any:
        model_config = self._model_config(settings)
        if self._model_factory is not None:
            return self._model_factory(model_config)
        try:
            from ml_collections import ConfigDict
            from pxdbench.tools.protmpnn.main_mpnn import _make_mpnn_model
        except Exception as exc:
            raise MPNNCacheSystemError(
                "CyclicMPNN runtime dependencies are unavailable"
            ) from exc
        return _make_mpnn_model(ConfigDict(model_config))

    def _get_model(
        self, settings: Mapping[str, Any]
    ) -> tuple[Any, bool, tuple[Any, ...]]:
        cache_settings = _cache_settings(self.family, self.device, settings)
        model, loaded = self._cache.get_or_load(
            cache_settings, lambda: self._load(settings)
        )
        return model, loaded, cache_settings

    def _apply_bias(self, model: Any, settings: Mapping[str, Any]) -> None:
        path = str(settings.get("bias_aa_json") or "")
        if self._bias_applier is not None:
            self._bias_applier(model, path, "")
            return
        from pxdbench.tools.protmpnn.main_mpnn import _apply_bias_inputs

        _apply_bias_inputs(model, path, "")

    def run_batch(
        self,
        payload: Mapping[str, Any],
        *,
        config_override: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("invalid CyclicMPNN batch protocol")
        raw_requests = payload.get("requests")
        if not isinstance(raw_requests, list) or not raw_requests:
            raise ValueError("invalid CyclicMPNN requests")
        payload_config = payload.get("config")
        if payload_config is not None and not isinstance(payload_config, Mapping):
            raise ValueError("invalid CyclicMPNN config")
        settings = self._settings(
            dict(payload_config or {}) | dict(config_override or {})
        )
        started = time.perf_counter()
        model, newly_loaded, _cache_settings_used = self._get_model(settings)
        load_seconds = time.perf_counter() - started if newly_loaded else 0.0
        native = _reward_sibling("cyclicmpnn_worker")
        results = native._sample_requests(
            model, raw_requests, settings,
            lambda current: self._apply_bias(current, settings),
            seed_everything=_seed_everything,
        )
        return {
            "schema_version": 1,
            "model_loads": int(newly_loaded),
            "model_load_count": self.model_load_count,
            "model_load_seconds": load_seconds,
            "cache_hit": not newly_loaded,
            "results": results,
        }

    def generate(
        self,
        requests: Iterable[Mapping[str, Any]],
        *,
        config_override: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        result = self.run_batch(
            {"config": self.config, "requests": list(requests)},
            config_override=config_override,
        )
        return self._result_maps(result)

    def validate_sequences_esm(
        self,
        payload_or_requests: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        *,
        config_override: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Validation-facing alias returning ``(sequences, failures)``."""

        if isinstance(payload_or_requests, Mapping) and "requests" in payload_or_requests:
            result = self.run_batch(payload_or_requests, config_override=config_override)
            return self._result_maps(result)
        return self.generate(payload_or_requests, config_override=config_override)


class _IPCPolymerRuntime:
    """Bridge the shared JSONL cache worker to the native file protocol.

    ``mpnn_cache_worker`` keeps one runtime object per backend and calls
    ``run(request)`` with the outer cache request.  The command still names the
    native worker's ``--request`` and ``--result`` files, so this bridge reads
    the request JSON, executes the resident adapter, and writes the exact
    worker-shaped result file expected by ``mpnn.py``.
    """

    def __init__(self, backend: str, script: Path | None = None) -> None:
        self.backend_name = str(backend)
        self.script = None if script is None else Path(script)
        self.backend: NAMPNNBackend | CyclicMPNNBackend | None = None

    @staticmethod
    def _command_option(command: list[str], name: str) -> str:
        try:
            index = command.index(name)
        except ValueError as exc:
            raise ValueError(f"cached {name} command option is missing") from exc
        if index + 1 >= len(command):
            raise ValueError(f"cached {name} command option has no value")
        return str(command[index + 1])

    @staticmethod
    def _path(value: str, base: Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else base / path

    @staticmethod
    def _device_from_request(request: Mapping[str, Any]) -> str:
        environment = request.get("env")
        if isinstance(environment, Mapping) and "CUDA_VISIBLE_DEVICES" in environment:
            visible = str(environment.get("CUDA_VISIBLE_DEVICES", "")).strip()
            if not visible or visible == "-1":
                return "cpu"
        try:
            import torch

            return "cuda:0" if bool(torch.cuda.is_available()) else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_backend(self, request: Mapping[str, Any], root: Path) -> None:
        if self.backend is not None:
            return
        device = self._device_from_request(request)
        if self.backend_name == "na_mpnn":
            self.backend = NAMPNNBackend(root, device=device)
        elif self.backend_name == "cyclicmpnn":
            self.backend = CyclicMPNNBackend(root, device=device)
        else:
            raise ValueError(f"unsupported polymer cache backend: {self.backend_name}")

    def run(self, request: Mapping[str, Any]) -> dict[str, Any]:
        command = [str(value) for value in request.get("command", ())]
        if len(command) < 2:
            raise ValueError("cached polymer command is too short")
        root = Path(str(request.get("cwd") or Path.cwd())).expanduser().resolve()
        request_path = self._path(self._command_option(command, "--request"), root)
        result_path = self._path(self._command_option(command, "--result"), root)
        try:
            payload = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MPNNCacheSystemError(
                f"cached {self.backend_name} request JSON is unreadable: {request_path}"
            ) from exc
        started = time.perf_counter()
        log_path_value = request.get("log_path")
        if log_path_value:
            log_path = self._path(str(log_path_value), root)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("a", encoding="utf-8")
        else:
            stream = open(os.devnull, "w", encoding="utf-8")
        with stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            self._ensure_backend(request, root)
            assert self.backend is not None
            raw = self.backend.run_batch(payload)
        try:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(raw, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise MPNNCacheSystemError(
                f"cached {self.backend_name} result cannot be written: {result_path}"
            ) from exc
        stats = self.backend.cache_stats()
        loads_this_request = int(raw.get("model_loads", 0))
        return {
            "status": "ok",
            "error": None,
            "backend": self.backend_name,
            "cache_hit": bool(raw.get("cache_hit", False)),
            "new_model_loads": loads_this_request,
            "model_loads_this_request": loads_this_request,
            "model_load_count": int(stats["model_load_count"]),
            "elapsed_seconds": time.perf_counter() - started,
        }

    @property
    def model_load_count(self) -> int:
        return 0 if self.backend is None else self.backend.model_load_count

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()


def make_backend(backend: str, script: Path | str | None = None) -> _IPCPolymerRuntime:
    """Factory consumed by ``mpnn_cache_worker._make_runtime``."""

    normalized = str(backend).lower().replace("-", "_")
    if normalized not in {"na_mpnn", "cyclicmpnn", "cyclic_mpnn"}:
        raise ValueError(f"unsupported cached polymer backend: {backend!r}")
    if normalized == "cyclic_mpnn":
        normalized = "cyclicmpnn"
    return _IPCPolymerRuntime(normalized, None if script is None else Path(script))


# Short aliases make family selection in the validation planner explicit.
NA_MPNNBackend = NAMPNNBackend
CyclicMPNNCacheBackend = CyclicMPNNBackend


def build_polymer_mpnn_backend(
    family: str,
    repo_root: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    *,
    device: str = "cuda:0",
    **kwargs: Any,
) -> NAMPNNBackend | CyclicMPNNBackend:
    """Construct the persistent backend for a supported non-protein family."""

    normalized = str(family).lower().replace("-", "_")
    if normalized in {"na_mpnn", "nampnn", "na"}:
        return NAMPNNBackend(
            repo_root,
            config,
            device=device,
            loader=kwargs.get("loader"),
            sampler=kwargs.get("sampler"),
        )
    if normalized in {"cyclicmpnn", "cyclic_mpnn", "cyclic"}:
        return CyclicMPNNBackend(
            repo_root,
            config,
            device=device,
            model_factory=kwargs.get("model_factory"),
            bias_applier=kwargs.get("bias_applier"),
        )
    raise ValueError(f"unsupported polymer MPNN family: {family!r}")


__all__ = [
    "CyclicMPNNBackend",
    "CyclicMPNNCacheBackend",
    "MPNNCacheConfigurationError",
    "MPNNCacheError",
    "MPNNCacheSystemError",
    "NAMPNNBackend",
    "NA_MPNNBackend",
    "build_polymer_mpnn_backend",
    "make_backend",
]
