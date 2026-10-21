"""Resident native backends used by the shared MPNN cache worker.

The vendor entry points remain the numerical implementation.  These wrappers
only keep their torch model and checkpoint objects alive between JSONL
requests; request seeds, paths, masks, and sample counts are untouched.
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


class MPNNBackendError(RuntimeError):
    """A cached native MPNN backend could not serve a request."""


def _option(argv: Sequence[str], name: str, default: str | None = None) -> str | None:
    try:
        index = list(argv).index(name)
    except ValueError:
        return default
    if index + 1 >= len(argv):
        raise ValueError(f"MPNN cache command lacks a value for {name}")
    return str(argv[index + 1])


def _flag(argv: Sequence[str], name: str) -> bool:
    return name in argv


def _resolve(value: str | None, *, cwd: Path, default: Path | None = None) -> Path | None:
    if value is None or value == "":
        return default.resolve() if default is not None else None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


class LigandMPNNBackend:
    """Resident wrapper for the official ``ligandmpnn/source/run.py``.

    ``run.py`` imports its model classes at module scope, so this backend uses
    temporary constructor wrappers in the isolated cache child.  The vendor
    module globals are restored while each original constructor runs; this is
    required by the vendor's explicit ``super(ProteinMPNN, self)`` calls.
    """

    name = "ligandmpnn"

    def __init__(self, script: Path):
        self.script = Path(script).expanduser().resolve()
        if self.script.name != "run.py":
            raise MPNNBackendError(f"LigandMPNN backend requires run.py, got {self.script}")
        self.source = self.script.parent
        if str(self.source) not in sys.path:
            sys.path.insert(0, str(self.source))
        self.torch = importlib.import_module("torch")
        self.model_utils = importlib.import_module("model_utils")
        self.sc_utils = importlib.import_module("sc_utils")
        self._torch_load = self.torch.load
        self._base_model = self.model_utils.ProteinMPNN
        self._base_packer = self.sc_utils.Packer
        self._active: dict[str, Any] | None = None
        self._pending_model: Any = None
        self._pending_packer: Any = None
        self._pending_checkpoints: dict[str, Any] = {}
        self._restores: list[tuple[Any, Any]] = []
        self._model_cache: dict[str, dict[str, Any]] = {}
        self._packer_cache: dict[str, dict[str, Any]] = {}
        self._checkpoint_cache: dict[str, Any] = {}
        self.model_load_count = 0
        self.constructor_count = 0
        self.state_dict_load_count = 0
        self._install_hooks()

    @staticmethod
    def _checkpoint_path(path: Path) -> str:
        """Use the configured checkpoint path directly for resident reuse."""
        return str(path.expanduser().resolve())

    def _install_hooks(self) -> None:
        runtime = self
        base_model = self._base_model
        base_packer = self._base_packer

        def cached_load(path: Any, *args: Any, **kwargs: Any) -> Any:
            resolved = Path(str(path)).expanduser().resolve()
            checkpoint_path = runtime._checkpoint_path(resolved)
            if checkpoint_path in runtime._checkpoint_cache:
                return runtime._checkpoint_cache[checkpoint_path]
            value = runtime._torch_load(path, *args, **kwargs)
            runtime._checkpoint_cache[checkpoint_path] = value
            if runtime._active is not None:
                runtime._pending_checkpoints[checkpoint_path] = value
            return value

        class CachedProteinMPNN(base_model):
            def __new__(cls, *args: Any, **kwargs: Any) -> Any:
                active = runtime._active
                if active is not None:
                    entry = runtime._model_cache.get(active["model_settings"])
                    if entry is not None:
                        active["model_cache_hit"] = True
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
                previous_class = runtime.model_utils.ProteinMPNN
                runtime.model_utils.ProteinMPNN = base_model
                try:
                    # CPU parameter initialization must not advance the
                    # request sampling stream only when the model is cold.
                    with runtime.torch.random.fork_rng(devices=[]):
                        base_model.__init__(self, *args, **kwargs)
                finally:
                    runtime.model_utils.ProteinMPNN = previous_class
                self._partidesign_cache_initialized = True
                runtime.constructor_count += 1
                original = self.load_state_dict

                def counted_load(*load_args: Any, **load_kwargs: Any) -> Any:
                    runtime.state_dict_load_count += 1
                    return original(*load_args, **load_kwargs)

                self.load_state_dict = counted_load
                runtime._restores.append((self, original))
                runtime._pending_model = self

        class CachedPacker(base_packer):
            def __new__(cls, *args: Any, **kwargs: Any) -> Any:
                active = runtime._active
                if active is not None:
                    entry = runtime._packer_cache.get(active["packer_settings"])
                    if entry is not None:
                        active["packer_cache_hit"] = True
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
                previous_class = runtime.sc_utils.Packer
                runtime.sc_utils.Packer = base_packer
                try:
                    # CPU parameter initialization must not advance the
                    # request sampling stream only when the model is cold.
                    with runtime.torch.random.fork_rng(devices=[]):
                        base_packer.__init__(self, *args, **kwargs)
                finally:
                    runtime.sc_utils.Packer = previous_class
                self._partidesign_cache_initialized = True
                runtime.constructor_count += 1
                original = self.load_state_dict

                def counted_load(*load_args: Any, **load_kwargs: Any) -> Any:
                    runtime.state_dict_load_count += 1
                    return original(*load_args, **load_kwargs)

                self.load_state_dict = counted_load
                runtime._restores.append((self, original))
                runtime._pending_packer = self

        self.torch.load = cached_load
        self.model_utils.ProteinMPNN = CachedProteinMPNN
        self.sc_utils.Packer = CachedPacker

    def _settings_for_command(
        self, command: Sequence[str], cwd: Path
    ) -> tuple[tuple[Any, ...], tuple[Any, ...] | None, Path, Path | None]:
        model_type = _option(command, "--model_type", "protein_mpnn") or "protein_mpnn"
        checkpoint_flag = f"--checkpoint_{model_type}"
        checkpoint = _resolve(
            _option(command, checkpoint_flag),
            cwd=cwd,
            default=self.source / "model_params" / f"{model_type}.pt",
        )
        if checkpoint is None:
            raise MPNNBackendError(f"missing checkpoint for model_type={model_type}")
        device = "cuda:0" if bool(self.torch.cuda.is_available()) else "cpu"
        model_settings = (
            self.name,
            model_type,
            str(checkpoint),
            device,
            str(_option(command, "--ligand_mpnn_use_atom_context", "1")),
            str(_option(command, "--ligand_mpnn_use_side_chain_context", "0")),
        )
        side_path = _resolve(_option(command, "--checkpoint_path_sc"), cwd=cwd)
        packer_settings = (
            (self.name, "packer", str(side_path), device)
            if _flag(command, "--pack_side_chains") and side_path is not None
            else None
        )
        return model_settings, packer_settings, checkpoint, side_path

    def _restore_model_methods(self) -> None:
        for model, original in self._restores:
            model.load_state_dict = original
        self._restores.clear()

    def run(self, request: Mapping[str, Any]) -> dict[str, Any]:
        command = [str(value) for value in request["command"]]
        cwd = Path(str(request["cwd"])).expanduser().resolve()
        model_settings, packer_settings, main_checkpoint, side_checkpoint = (
            self._settings_for_command(command, cwd)
        )
        main_checkpoint_path = self._checkpoint_path(main_checkpoint)
        side_checkpoint_path = (
            self._checkpoint_path(side_checkpoint) if side_checkpoint is not None else None
        )
        constructors_before = self.constructor_count
        state_dict_before = self.state_dict_load_count
        self._active = {
            "model_settings": model_settings,
            "packer_settings": packer_settings,
            "main_checkpoint": main_checkpoint_path,
            "side_checkpoint": side_checkpoint_path,
            "model_cache_hit": False,
            "packer_cache_hit": not bool(packer_settings),
        }
        self._pending_model = None
        self._pending_packer = None
        self._pending_checkpoints = {}
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
            os.chdir(str(cwd))
            log_path = request.get("log_path")
            if log_path:
                path = Path(str(log_path)).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.open("a", encoding="utf-8")
            else:
                stream = open(os.devnull, "w", encoding="utf-8")
            with stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                old_argv = sys.argv
                sys.argv = command[1:]
                try:
                    runpy.run_path(str(self.script), run_name="__main__")
                finally:
                    sys.argv = old_argv
        except SystemExit as exc:
            if exc.code not in (None, 0):
                status = "failed"
                error = f"LigandMPNN exited with status {exc.code}"
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
            if status == "ok" and active is not None:
                if self._pending_model is not None and not active["model_cache_hit"]:
                    self._model_cache[model_settings] = {
                        "model": self._pending_model,
                        "checkpoint": self._pending_checkpoints.get(main_checkpoint_path),
                    }
                    self.model_load_count += 1
                if (
                    self._pending_packer is not None
                    and packer_settings
                    and not active["packer_cache_hit"]
                ):
                    self._packer_cache[packer_settings] = {
                        "model": self._pending_packer,
                        "checkpoint": self._pending_checkpoints.get(side_checkpoint_path),
                    }
                    self.model_load_count += 1
            self._pending_model = None
            self._pending_packer = None
            self._pending_checkpoints = {}
            self._active = None
        active = active or {}
        main_hit = bool(active.get("model_cache_hit", False))
        packer_hit = bool(active.get("packer_cache_hit", False))
        return {
            "status": status,
            "error": error,
            "backend": self.name,
            "cache_hit": bool(main_hit and packer_hit),
            "main_cache_hit": main_hit,
            "packer_cache_hit": packer_hit,
            "new_model_loads": int(
                (not main_hit) + (bool(packer_settings) and not packer_hit)
                if status == "ok"
                else 0
            ),
            "model_loads_this_request": int(
                (not main_hit) + (bool(packer_settings) and not packer_hit)
                if status == "ok"
                else 0
            ),
            "constructors_this_request": int(self.constructor_count - constructors_before),
            "state_dict_loads_this_request": int(self.state_dict_load_count - state_dict_before),
            "constructor_count": int(self.constructor_count),
            "state_dict_load_count": int(self.state_dict_load_count),
            "model_load_count": int(self.model_load_count),
            "elapsed_seconds": time.perf_counter() - started,
        }

    def close(self) -> None:
        self._restore_model_methods()
        self.torch.load = self._torch_load
        self.model_utils.ProteinMPNN = self._base_model
        self.sc_utils.Packer = self._base_packer
        self._model_cache.clear()
        self._packer_cache.clear()
        self._checkpoint_cache.clear()
        gc.collect()
        if bool(self.torch.cuda.is_available()):
            self.torch.cuda.empty_cache()


__all__ = ["LigandMPNNBackend", "MPNNBackendError"]
