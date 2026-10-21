"""Shared resident cache for the native Protein/Ligand MPNN workers.

A pool of isolated children serves prepared requests. Model identity is derived
from backend/family, checkpoint file metadata, device and model-shape settings;
seed, temperature, output paths and sample count remain request-local.
"""
from __future__ import annotations

import atexit
import json
import os
import select
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


class MPNNCacheError(RuntimeError):
    """The resident ProteinMPNN cache protocol or worker failed."""


_STDERR_TAIL_LIMIT = 4_096


class _PersistentMPNNProcess:
    def __init__(
        self,
        *,
        python: str,
        worker: Path,
        env: Mapping[str, str],
        timeout: float,
    ):
        self.python = str(python)
        self.worker = Path(worker).expanduser().resolve()
        self.timeout = float(timeout)
        self.env = {str(key): str(value) for key, value in env.items()}
        self.process = subprocess.Popen(
            [self.python, str(self.worker)],
            cwd=str(self.worker.parent),
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._stdout_buffer = bytearray()
        self._stderr_buffer = bytearray()
        self._stderr_closed = False
        try:
            ready = self._read_line(self.timeout)
            if ready.get("type") != "ready" or ready.get("schema_version") != 1:
                raise MPNNCacheError("resident ProteinMPNN worker did not become ready")
        except BaseException:
            self.close(force=True)
            raise

    def _drain_stderr(self) -> None:
        process = self.process
        stream = None if process is None else process.stderr
        if stream is None or self._stderr_closed:
            return
        try:
            fd = stream.fileno()
        except (OSError, ValueError):
            self._stderr_closed = True
            return
        while True:
            try:
                ready, _, _ = select.select([fd], [], [], 0)
                if not ready:
                    return
                chunk = os.read(fd, 65536)
            except (OSError, ValueError):
                self._stderr_closed = True
                return
            if not chunk:
                self._stderr_closed = True
                return
            self._stderr_buffer.extend(chunk)
            if len(self._stderr_buffer) > _STDERR_TAIL_LIMIT:
                del self._stderr_buffer[:-_STDERR_TAIL_LIMIT]

    def _failure(self, message: str) -> MPNNCacheError:
        self._drain_stderr()
        tail = " ".join(
            bytes(self._stderr_buffer).decode("utf-8", errors="replace").split()
        )
        if tail:
            return MPNNCacheError(f"{message}; worker_stderr_tail={tail}")
        return MPNNCacheError(message)

    def _read_line(self, timeout: float) -> dict[str, Any]:
        process = self.process
        if process.stdout is None:
            raise MPNNCacheError("resident ProteinMPNN worker has no stdout")
        stdout_fd = process.stdout.fileno()
        deadline = time.monotonic() + max(1.0, float(timeout))
        while True:
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._stdout_buffer[:newline])
                del self._stdout_buffer[: newline + 1]
                try:
                    value = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise self._failure(
                        "resident ProteinMPNN worker emitted invalid JSON"
                    ) from exc
                if not isinstance(value, dict):
                    raise self._failure(
                        "resident ProteinMPNN worker emitted a non-object"
                    )
                return value
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._failure("resident ProteinMPNN worker timed out")
            fds = [stdout_fd]
            if process.stderr is not None and not self._stderr_closed:
                try:
                    fds.append(process.stderr.fileno())
                except (OSError, ValueError):
                    self._stderr_closed = True
            ready, _, _ = select.select(fds, [], [], remaining)
            if not ready:
                raise self._failure("resident ProteinMPNN worker timed out")
            if len(fds) > 1 and fds[1] in ready:
                self._drain_stderr()
            if stdout_fd not in ready:
                continue
            chunk = os.read(stdout_fd, 65536)
            if not chunk:
                raise self._failure(
                    f"resident ProteinMPNN worker exited with status {process.poll()}"
                )
            self._stdout_buffer.extend(chunk)

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path | None,
        timeout: float,
        backend: str,
    ) -> dict[str, Any]:
        process = self.process
        if process.stdin is None:
            raise MPNNCacheError("resident ProteinMPNN worker has no stdin")
        payload = {
            "request_id": uuid.uuid4().hex,
            "backend": str(backend),
            "command": [str(value) for value in command],
            "cwd": str(Path(cwd).expanduser().resolve()),
            "env": {str(key): str(value) for key, value in env.items()},
            "log_path": None if log_path is None else str(Path(log_path).expanduser().resolve()),
        }
        process.stdin.write((json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        process.stdin.flush()
        result = self._read_line(float(timeout))
        if result.get("type") != "result":
            raise MPNNCacheError("resident MPNN worker returned an invalid response")
        if result.get("request_id") != payload["request_id"]:
            raise MPNNCacheError("resident MPNN worker response id does not match request")
        if result.get("status") != "ok":
            error = str(result.get("error") or "resident MPNN worker failed")
            raise MPNNCacheError(f"{backend} request failed: {error}")
        return result

    def close(self, *, force: bool = False) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if force and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        try:
            if process.stdin is not None:
                process.stdin.write(b'{"type":"close"}\n')
                process.stdin.flush()
                process.stdin.close()
            process.wait(timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=30)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()


_ProcessIdentity = tuple[str, str, str, str, tuple[tuple[str, str], ...]]
_PROCESS_POOL: dict[_ProcessIdentity, _PersistentMPNNProcess] = {}
_LOCK = threading.RLock()
_OWNER_PID = os.getpid()


def _reset_after_fork() -> None:
    """Drop inherited pipe handles without closing the parent's workers."""
    global _PROCESS_POOL, _LOCK, _OWNER_PID
    current = os.getpid()
    if current != _OWNER_PID:
        _PROCESS_POOL = {}
        _LOCK = threading.RLock()
        _OWNER_PID = current


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _environment_identity(
    python: str, env: Mapping[str, str], *, backend: str, cwd: Path, worker: Path,
) -> _ProcessIdentity:
    # Device/runtime libraries read environment variables at import time.
    # Retain the complete canonical environment rather than an incomplete
    # CUDA allowlist; values stay private to the in-process pool identity.
    return (
        str(backend), str(Path(cwd).expanduser().resolve()), str(python),
        str(Path(worker).expanduser().resolve()),
        tuple(sorted((str(key), str(value)) for key, value in env.items())),
    )


def _infer_backend(command: Sequence[str]) -> str:
    if len(command) < 2:
        raise MPNNCacheError("cached MPNN command is too short")
    script = Path(str(command[1])).name
    if script == "protein_mpnn_run.py":
        return "proteinmpnn"
    if script == "run.py" and "ligandmpnn" in str(command[1]).replace("\\", "/"):
        return "ligandmpnn"
    raise MPNNCacheError(
        "cannot infer cached MPNN backend; pass backend explicitly for this native command"
    )


def run_mpnn_cached(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    log_path: Path | None = None,
    timeout: float = 7200,
    python: str | None = None,
    worker: Path | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Execute one native MPNN request through a resident backend."""
    values = [str(value) for value in command]
    selected_backend = str(backend or _infer_backend(values)).lower()
    selected_backend = {
        "protein_mpnn": "proteinmpnn",
        "ligand_mpnn": "ligandmpnn",
        "cyclic_mpnn": "cyclicmpnn",
        "nampnn": "na_mpnn",
    }.get(selected_backend, selected_backend)
    if selected_backend not in {"proteinmpnn", "ligandmpnn", "cyclicmpnn", "na_mpnn", "af2"}:
        raise MPNNCacheError(f"unsupported cached MPNN backend {selected_backend!r}")
    if len(values) < 2:
        raise MPNNCacheError("cached MPNN command is too short")
    selected_python = str(python or values[0] or sys.executable)
    selected_env = {
        str(key): str(value)
        for key, value in (os.environ if env is None else env).items()
    }
    if selected_backend in {"na_mpnn", "cyclicmpnn", "af2"}:
        # JAX defaults to reserving most of the GPU at import time.  A
        # resident polymer worker must coexist with AF2/ESM workers; preserve
        # an explicit caller value but make the safe default before child spawn.
        selected_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    selected_worker = Path(worker or Path(__file__).with_name("mpnn_cache_worker.py"))
    selected_cwd = Path(cwd).expanduser().resolve()
    identity = _environment_identity(
        selected_python,
        selected_env,
        backend=selected_backend,
        cwd=selected_cwd,
        worker=selected_worker,
    )
    _reset_after_fork()
    with _LOCK:
        process = _PROCESS_POOL.get(identity)
        if process is None:
            process = _PersistentMPNNProcess(
                python=selected_python,
                worker=selected_worker,
                env=selected_env,
                timeout=timeout,
            )
            _PROCESS_POOL[identity] = process
        try:
            return process.run(
                values,
                cwd=selected_cwd,
                env=selected_env,
                log_path=log_path,
                timeout=timeout,
                backend=selected_backend,
            )
        except BaseException:
            # An interrupted request may still be running in the child. It
            # must never be reused with an unread response from the old call.
            _PROCESS_POOL.pop(identity, None)
            process.close(force=True)
            raise


def run_cached(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    log_path: Path | None = None,
    timeout: float = 7200,
    python: str | None = None,
    worker: Path | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Canonical shared-cache entry point used by reward integrations."""
    return run_mpnn_cached(
        command,
        cwd=cwd,
        env=env,
        log_path=log_path,
        timeout=timeout,
        python=python,
        worker=worker,
        backend=backend,
    )


def run_protein_mpnn_cached(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    log_path: Path | None = None,
    timeout: float = 7200,
    python: str | None = None,
    worker: Path | None = None,
) -> dict[str, Any]:
    """Backward-compatible Protein/Soluble ProteinMPNN entry point."""
    return run_mpnn_cached(
        command,
        cwd=cwd,
        env=env,
        log_path=log_path,
        timeout=timeout,
        python=python,
        worker=worker,
        backend="proteinmpnn",
    )


def close_mpnn_cache() -> None:
    """Close every resident child and release all cached MPNN models."""
    _reset_after_fork()
    with _LOCK:
        processes = list(_PROCESS_POOL.values())
        _PROCESS_POOL.clear()
        for process in processes:
            process.close()


__all__ = [
    "MPNNCacheError",
    "run_mpnn_cached",
    "run_cached",
    "run_protein_mpnn_cached",
    "close_mpnn_cache",
    "close_cached_workers",
]



def close_cached_workers() -> None:
    """Stable lifecycle alias for callers serving all MPNN backend families."""
    close_mpnn_cache()


atexit.register(close_mpnn_cache)
