"""Resident ProteinMPNN subprocess for the shared reward/validation cache.

The upstream ProteinMPNN entry point remains the numerical implementation.  This
worker keeps its loaded checkpoint/model objects resident, executes ordinary
Protein/Soluble/CA ``sample`` calls eagerly, and runs the same argv once per
request. Tied and unsupported calls retain the stock model methods.
"""
from __future__ import annotations

import contextlib
import gc
import importlib
import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from mpnn_cache_backends import LigandMPNNBackend
except ImportError:  # pragma: no cover - package import path in unit tests
    from .mpnn_cache_backends import LigandMPNNBackend


def _emit(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n")
    sys.stdout.flush()


def _option(argv: Sequence[str], name: str, default: str | None = None) -> str | None:
    try:
        index = list(argv).index(name)
    except ValueError:
        return default
    if index + 1 >= len(argv):
        raise ValueError(f"ProteinMPNN cache command lacks a value for {name}")
    return str(argv[index + 1])


def _flag(argv: Sequence[str], name: str) -> bool:
    return name in argv


def _settings_for_command(
    command: Sequence[str], torch: Any
) -> tuple[tuple[Any, ...], Path]:
    if len(command) < 2:
        raise ValueError("ProteinMPNN cache command is too short")
    script = Path(command[1]).expanduser().resolve()
    if script.name != "protein_mpnn_run.py":
        raise ValueError("ProteinMPNN cache only accepts protein_mpnn_run.py")
    source = script.parent
    weights_value = _option(command, "--path_to_model_weights")
    if weights_value:
        weights_dir = Path(weights_value).expanduser().resolve()
    else:
        if _flag(command, "--ca_only"):
            directory_name = "ca_model_weights"
        elif _flag(command, "--use_soluble_model"):
            directory_name = "soluble_model_weights"
        else:
            directory_name = "vanilla_model_weights"
        weights_dir = source / directory_name
    model_name = _option(command, "--model_name", "v_48_020") or "v_48_020"
    checkpoint = (weights_dir / f"{model_name}.pt").resolve()
    if _flag(command, "--ca_only"):
        family = "ca"
    elif _flag(command, "--use_soluble_model"):
        family = "soluble"
    else:
        family = "vanilla"
    device = "cuda:0" if bool(torch.cuda.is_available()) else "cpu"
    settings = (
        "proteinmpnn",
        family,
        str(checkpoint),
        bool(_flag(command, "--ca_only")),
        bool(_flag(command, "--use_soluble_model")),
        str(_option(command, "--backbone_noise", "0.0")),
        device,
    )
    return settings, checkpoint


class _ResidentProteinMPNN:
    def __init__(self, script: Path):
        source = script.parent
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        self.torch = importlib.import_module("torch")
        self.utils = importlib.import_module("protein_mpnn_utils")
        self._torch_load = self.torch.load
        self._model_type = self.utils.ProteinMPNN
        self._active: dict[str, Any] | None = None
        self._pending_model: Any = None
        self._pending_checkpoint: Any = None
        self._restores: list[tuple[Any, Any]] = []
        self._cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.model_load_count = 0
        self.constructor_count = 0
        self.state_dict_load_count = 0
        def cached_load(path: Any, *args: Any, **kwargs: Any) -> Any:
            active = self._active
            resolved = Path(str(path)).expanduser().resolve()
            if active is not None:
                entry = self._cache.get(active["settings"])
                if entry is not None and resolved == entry["checkpoint_path"]:
                    return entry["checkpoint"]
            value = self._torch_load(path, *args, **kwargs)
            if active is not None and resolved == active["checkpoint_path"]:
                self._pending_checkpoint = value
            return value

        runtime = self
        base_model_type = self._model_type

        class CachedProteinMPNN(base_model_type):
            """Subclass wrapper that preserves vendor super(ProteinMPNN, self)."""

            def __new__(cls, *args: Any, **kwargs: Any) -> Any:
                active = runtime._active
                if active is not None:
                    entry = runtime._cache.get(active["settings"])
                    if entry is not None:
                        active["cache_hit"] = True
                        return entry["model"]
                return super().__new__(cls)

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                if getattr(self, "_partidesign_cache_initialized", False):
                    active = runtime._active
                    if active is not None:
                        original = self.load_state_dict

                        def skip_load(*_args: Any, **_kwargs: Any) -> None:
                            return None

                        self.load_state_dict = skip_load
                        runtime._restores.append((self, original))
                    return
                # The vendor class uses ``super(ProteinMPNN, self)``;
                # temporarily restore its module-global class while invoking
                # the original constructor so the wrapper cannot recurse.
                previous_class = runtime.utils.ProteinMPNN
                runtime.utils.ProteinMPNN = base_model_type
                try:
                    # Vendor parameter initialization runs on CPU before the
                    # checkpoint is applied. It must not shift sampling RNG on
                    # a cold request relative to an otherwise identical hit.
                    with runtime.torch.random.fork_rng(devices=[]):
                        base_model_type.__init__(self, *args, **kwargs)
                finally:
                    runtime.utils.ProteinMPNN = previous_class
                self._partidesign_cache_initialized = True
                runtime.constructor_count += 1
                original = self.load_state_dict

                def counted_load(*load_args: Any, **load_kwargs: Any) -> Any:
                    runtime.state_dict_load_count += 1
                    return original(*load_args, **load_kwargs)

                self.load_state_dict = counted_load
                runtime._restores.append((self, original))
                runtime._pending_model = self
        self.torch.load = cached_load
        self.utils.ProteinMPNN = CachedProteinMPNN

    def _restore_model_methods(self) -> None:
        for model, original in self._restores:
            model.load_state_dict = original
        self._restores.clear()

    def run(self, request: Mapping[str, Any]) -> dict[str, Any]:
        command = [str(value) for value in request["command"]]
        settings, checkpoint = _settings_for_command(command, self.torch)
        constructors_before = self.constructor_count
        state_dict_before = self.state_dict_load_count
        self._active = {
            "settings": settings,
            "checkpoint_path": checkpoint,
            "cache_hit": False,
        }
        self._pending_model = None
        self._pending_checkpoint = None
        started = time.perf_counter()
        old_cwd = Path.cwd()
        old_env = os.environ.copy()
        status = "ok"
        error: str | None = None
        try:
            environment = request.get("env")
            if isinstance(environment, Mapping):
                os.environ.clear()
                os.environ.update({str(k): str(v) for k, v in environment.items()})
            script = Path(command[1]).expanduser().resolve()
            os.chdir(str(request["cwd"]))
            log_path = request.get("log_path")
            if log_path:
                log_file = Path(str(log_path)).expanduser()
                log_file.parent.mkdir(parents=True, exist_ok=True)
                stream = log_file.open("a", encoding="utf-8")
            else:
                stream = open(os.devnull, "w", encoding="utf-8")
            with stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                old_argv = sys.argv
                sys.argv = command[1:]
                try:
                    runpy.run_path(str(script), run_name="__main__")
                finally:
                    sys.argv = old_argv
        except SystemExit as exc:
            if exc.code not in (None, 0):
                status = "failed"
                error = f"ProteinMPNN exited with status {exc.code}"
        except BaseException as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=sys.stderr)
        finally:
            os.chdir(str(old_cwd))
            os.environ.clear()
            os.environ.update(old_env)
            self._restore_model_methods()
            active = self._active
            if (
                status == "ok"
                and active is not None
                and not active["cache_hit"]
                and self._pending_model is not None
                and self._pending_checkpoint is not None
            ):
                self._cache[active["settings"]] = {
                    "model": self._pending_model,
                    "checkpoint": self._pending_checkpoint,
                    "checkpoint_path": active["checkpoint_path"],
                }
                self.model_load_count += 1
            self._pending_model = None
            self._pending_checkpoint = None
            self._active = None
        cache_hit = bool(active and active["cache_hit"])
        loads_this_request = int(status == "ok" and not cache_hit)
        return {
            "status": status,
            "error": error,
            "backend": "proteinmpnn",
            "cache_hit": cache_hit,
            "new_model_loads": loads_this_request,
            "model_loads_this_request": loads_this_request,
            "constructors_this_request": int(self.constructor_count - constructors_before),
            "state_dict_loads_this_request": int(self.state_dict_load_count - state_dict_before),
            "constructor_count": int(self.constructor_count),
            "state_dict_load_count": int(self.state_dict_load_count),
            "model_load_count": int(self.model_load_count),
            "elapsed_seconds": time.perf_counter() - started,
        }

    def close(self) -> None:
        self._restore_model_methods()
        self._cache.clear()
        gc.collect()
        if bool(self.torch.cuda.is_available()):
            self.torch.cuda.empty_cache()


def _backend_name(request: Mapping[str, Any], command: Sequence[str]) -> str:
    value = request.get("backend")
    if value:
        return str(value)
    script = Path(str(command[1])).name if len(command) > 1 else ""
    if script == "protein_mpnn_run.py":
        return "proteinmpnn"
    if script == "run.py" and "ligandmpnn" in str(command[1]).replace("\\", "/"):
        return "ligandmpnn"
    return ""


def _make_runtime(backend: str, command: Sequence[str]) -> Any:
    script = Path(str(command[1])).expanduser().resolve()
    if backend == "proteinmpnn":
        return _ResidentProteinMPNN(script)
    if backend == "ligandmpnn":
        return LigandMPNNBackend(script)
    if backend in {"na_mpnn", "cyclicmpnn"}:
        try:
            try:
                from mpnn_cache_polymer import make_backend
            except ImportError:  # pragma: no cover - package import path
                from .mpnn_cache_polymer import make_backend
            return make_backend(backend, script)
        except ImportError as exc:
            raise ValueError(f"cached backend {backend!r} is not installed") from exc
    raise ValueError(f"unsupported cached MPNN backend {backend!r}")


def serve() -> None:
    runtimes: dict[str, Any] = {}
    _emit({"type": "ready", "schema_version": 1})
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if not isinstance(request, dict):
            raise ValueError("MPNN cache request must be an object")
        if request.get("type") == "close":
            break
        command = request.get("command")
        request_id = request.get("request_id")
        if not isinstance(command, list) or len(command) < 2:
            _emit({"type": "result", "request_id": request_id, "status": "failed", "error": "command is not a valid list"})
            continue
        backend = _backend_name(request, command)
        try:
            if not backend:
                raise ValueError("request must declare a cached MPNN backend")
            runtime = runtimes.get(backend)
            if runtime is None:
                runtime = _make_runtime(backend, command)
                runtimes[backend] = runtime
            result = runtime.run(request)
        except BaseException as exc:
            runtime = runtimes.get(backend)
            result = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "backend": backend,
                "cache_hit": False,
                "new_model_loads": 0,
                "model_loads_this_request": 0,
                "model_load_count": 0 if runtime is None else int(getattr(runtime, "model_load_count", 0)),
            }
        result["request_id"] = request_id
        _emit({"type": "result", **result})
    for runtime in runtimes.values():
        try:
            runtime.close()
        except BaseException:
            pass


if __name__ == "__main__":
    serve()
