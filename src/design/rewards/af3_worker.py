"""Persistent native AlphaFold 3 model worker.

The worker owns one ``ModelRunner`` and one CCD cache for its entire lifetime.
The SDK is imported lazily so the design package remains usable in environments
where JAX/Haiku/Tokamax are intentionally absent.  Callers should create one
worker per process and reuse it for all candidates on that process.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np

from partidesign.paths import af3_source, code_root, offline_site

from .af3_input import AF3Input, AF3InputError


# Keep the planning ABI's native bucket set stable.  The upstream script has
# additional intermediate buckets, but exposing those here would change the
# requested plan (and shape/compile cache keys) for an otherwise identical
# request.
NATIVE_BUCKETS = (256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120)


class AF3WorkerError(RuntimeError):
  """Raised for a persistent AF3 worker/runtime failure."""

  def __init__(self, message: str, *, fatal: bool = True) -> None:
    super().__init__(message)
    self.fatal = bool(fatal)


class AF3OutputError(AF3WorkerError):
  """One prediction's invalid output; the resident model remains usable."""

  def __init__(self, message: str) -> None:
    super().__init__(message, fatal=False)



def _json_default(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  raise TypeError(f"AF3 service value is not JSON serializable: {type(value).__name__}")


def _jsonable(value: Any) -> Any:
  if isinstance(value, Mapping):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [_jsonable(item) for item in value]
  if isinstance(value, np.ndarray):
    return _jsonable(value.tolist())
  if isinstance(value, np.generic):
    return _jsonable(value.item())
  if hasattr(value, "tolist"):
    try:
      return _jsonable(value.tolist())
    except (TypeError, ValueError):
      pass
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, (str, int, bool)) or value is None:
    return value
  if hasattr(value, "__dict__"):
    return _jsonable(vars(value))
  return str(value)


def _configured_ccd_dir(repo_root: Path, config: Any) -> Path:
  configured = _get(
      config,
      "externalccd_dir",
      _get(config, "externalccd_pickle_path", _get(config, "ccd_pickle_path", None)),
  )
  if configured is None:
    return repo_root / "community_models" / "af3" / "ccd"
  raw = Path(str(configured)).expanduser()
  if not raw.is_absolute():
    raw = repo_root / raw
  return raw if raw.is_dir() else raw.parent


def _af3_pythonpath(*sources: Path) -> str:
  """Build the AF3 child path from package-owned roots only."""
  values: list[str] = []
  seen: set[str] = set()
  for path in (code_root(), offline_site(), *sources):
    value = str(Path(path).expanduser().resolve())
    if value and value not in seen:
      seen.add(value)
      values.append(value)
  return os.pathsep.join(values)


def _af3_worker_environment(repo_root: Path, config: Any) -> dict[str, str]:
  """Return the persistent child environment with a closed import path."""
  environment = dict(os.environ)
  environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  environment["PYTHONPATH"] = _af3_pythonpath(af3_source() / "src")
  ccd_dir = _configured_ccd_dir(repo_root, config).resolve()
  environment["PARTIDESIGN_AF3_CCD_DIR"] = str(ccd_dir)
  environment["LIBCIFPP_DATA_DIR"] = str(ccd_dir)
  return environment


@contextmanager
def _redirect_native_output(log_dir: Path | None):
  """Keep Python and native SDK stdout away from the JSON-lines protocol."""
  handle = None
  stream = sys.stderr
  old_stdout_fd = os.dup(1)
  old_stderr_fd = os.dup(2)
  if log_dir is not None:
    path = Path(log_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    handle = (path / f"worker_{os.getpid()}.log").open("a", encoding="utf-8")
    stream = handle
  try:
    os.dup2(stream.fileno(), 1)
    os.dup2(stream.fileno(), 2)
    with redirect_stdout(stream), redirect_stderr(stream):
      yield
  finally:
    try:
      stream.flush()
    except (AttributeError, OSError, ValueError):
      pass
    os.dup2(old_stdout_fd, 1)
    os.dup2(old_stderr_fd, 2)
    os.close(old_stdout_fd)
    os.close(old_stderr_fd)
    if handle is not None:
      handle.close()


class _PersistentAF3Service:
  """Parent-side JSON-lines client for one resident native AF3 child."""

  def __init__(
      self,
      repo_root: Path,
      config: Any,
      method: str,
      device: str | int | None,
      log_dir: Path | None,
  ) -> None:
    self.repo_root = repo_root
    self.config = config
    self.method = method
    self.device = device
    self._log_dir = None if log_dir is None else Path(log_dir).expanduser().resolve()
    self._process: subprocess.Popen[bytes] | None = None
    self._stdout_buffer = bytearray()
    self._stderr_thread: threading.Thread | None = None
    self._stderr_tail = bytearray()
    self._stderr_lock = threading.RLock()
    self._lock = threading.RLock()
    self._fatal: str | None = None
    self._fatal_error: AF3WorkerError | None = None
    self._closed = False

  @property
  def fatal(self) -> str | None:
    return self._fatal

  @property
  def usable(self) -> bool:
    if self._closed or self._fatal is not None:
      return False
    if self._process is not None and self._process.poll() is not None:
      self._record_failure(self._failure(
          f"persistent AF3 child exited with {self._process.returncode}"
      ))
      return False
    return True

  def _record_failure(self, error: Exception) -> AF3WorkerError:
    # EOF/log updates during cleanup must never replace the business cause.
    if self._fatal_error is None:
      failure = (AF3WorkerError(self._fatal) if self._fatal is not None
                 else error if isinstance(error, AF3WorkerError) else AF3WorkerError(str(error)))
      if failure is not error:
        failure.__cause__ = error
      self._fatal_error = failure
      self._fatal = str(failure)
    return self._fatal_error

  def failure(self) -> AF3WorkerError:
    return self._record_failure(self._failure("persistent AF3 child is unavailable"))

  def _timeout(self) -> float:
    value = _get(self.config, "timeout_seconds", 1800.0)
    try:
      timeout = float(value)
    except (TypeError, ValueError) as exc:
      raise AF3WorkerError(
          "AF3 timeout_seconds must be a finite positive number"
      ) from exc
    if timeout <= 0 or not np.isfinite(timeout):
      raise AF3WorkerError("AF3 timeout_seconds must be a finite positive number")
    return timeout

  def _log_path(self) -> Path | None:
    if self._log_dir is None:
      return None
    self._log_dir.mkdir(parents=True, exist_ok=True)
    return self._log_dir / f"worker_{os.getpid()}.log"

  def _drain_stderr(self, stream: Any) -> None:
    try:
      while True:
        line = stream.readline()
        if not line:
          break
        with self._stderr_lock:
          self._stderr_tail.extend(line)
          if len(self._stderr_tail) > 8192:
            del self._stderr_tail[:-8192]
          path = self._log_path()
          if path is not None:
            with path.open("ab") as output:
              output.write(line)
    except (OSError, ValueError):
      return

  def _tail(self) -> str:
    with self._stderr_lock:
      return bytes(self._stderr_tail).decode("utf-8", errors="replace")[-4000:]

  def _failure(self, message: str) -> AF3WorkerError:
    if self._fatal_error is not None:
      return self._fatal_error
    tail = self._tail()
    suffix = f"; stderr tail: {tail}" if tail else ""
    path = self._log_path()
    location = f"; log: {path}" if path is not None else ""
    return AF3WorkerError(f"{message}{location}{suffix}")

  def _terminate(self) -> None:
    process, self._process = self._process, None
    if process is None:
      return
    try:
      if process.poll() is None:
        process.terminate()
        try:
          process.wait(timeout=5)
        except subprocess.TimeoutExpired:
          process.kill()
          process.wait(timeout=5)
      else:
        process.wait(timeout=1)
    except (OSError, subprocess.TimeoutExpired):
      pass
    for stream in (process.stdin, process.stdout, process.stderr):
      try:
        if stream is not None:
          stream.close()
      except (OSError, ValueError):
        pass
    thread, self._stderr_thread = self._stderr_thread, None
    if thread is not None:
      thread.join(timeout=5)

  def _read_message(self, timeout: float) -> dict[str, Any]:
    process = self._process
    if process is None or process.stdout is None:
      raise self._failure("persistent AF3 child is unavailable")
    deadline = time.monotonic() + max(0.001, timeout)
    while True:
      newline = self._stdout_buffer.find(b"\n")
      if newline >= 0:
        raw_line = bytes(self._stdout_buffer[:newline])
        del self._stdout_buffer[: newline + 1]
        try:
          value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
          continue
        if isinstance(value, dict):
          return value
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        raise self._failure("persistent AF3 child timed out")
      try:
        ready, _, _ = select.select([process.stdout.fileno()], [], [], remaining)
      except (OSError, ValueError) as exc:
        raise self._failure("persistent AF3 child protocol read failed") from exc
      if not ready:
        raise self._failure("persistent AF3 child timed out")
      try:
        chunk = os.read(process.stdout.fileno(), 65536)
      except OSError as exc:
        raise self._failure("persistent AF3 child protocol read failed") from exc
      if not chunk:
        raise self._failure(
            f"persistent AF3 child exited with {process.poll()}"
        )
      self._stdout_buffer.extend(chunk)

  def _start(self) -> None:
    if self._process is not None:
      return
    if self._fatal is not None:
      raise self.failure()
    config_json = json.dumps(_jsonable(self.config), allow_nan=False, sort_keys=True)
    command = [
        sys.executable,
        "-m",
        "src.design.rewards.af3_worker",
        "--af3-service-worker",
        "--repo-root",
        str(self.repo_root),
        "--method",
        self.method,
        "--config-json",
        config_json,
    ]
    if self.device is not None:
      command.extend(["--device", str(self.device)])
    if self._log_dir is not None:
      command.extend(["--log-dir", str(self._log_dir)])
    env = _af3_worker_environment(self.repo_root, self.config)
    try:
      process = subprocess.Popen(
          command,
          cwd=str(code_root()),
          env=env,
          stdin=subprocess.PIPE,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          bufsize=0,
      )
    except OSError as exc:
      failure = self._record_failure(AF3WorkerError(f"failed to start persistent AF3 child: {exc}"))
      raise failure from exc
    self._process = process
    if process.stderr is not None:
      self._stderr_thread = threading.Thread(
          target=self._drain_stderr,
          args=(process.stderr,),
          name=f"af3-stderr-{os.getpid()}",
          daemon=True,
      )
      self._stderr_thread.start()
    try:
      ready = self._read_message(self._timeout())
    except Exception as exc:
      failure = self._record_failure(exc)
      self._terminate()
      if failure is exc:
        raise
      raise failure from exc
    if ready.get("type") != "ready" or ready.get("schema_version") != 1:
      message = str(
          ready.get("message", "persistent AF3 child did not become ready")
      )
      failure = self._record_failure(AF3WorkerError(message))
      self._terminate()
      raise failure

  def set_log_dir(self, log_dir: Path | None) -> None:
    with self._lock:
      self._log_dir = None if log_dir is None else Path(log_dir).expanduser().resolve()
      if self._process is None or not self.usable:
        return
      try:
        self._send({
            "type": "set_log_dir",
            "path": None if self._log_dir is None else str(self._log_dir),
        })
        response = self._read_message(self._timeout())
      except Exception as exc:
        failure = self._record_failure(exc)
        self._terminate()
        if failure is exc:
          raise
        raise failure from exc
      if response.get("type") != "log_dir_set":
        failure = self._record_failure(AF3WorkerError(str(
            response.get("message", "AF3 child log setup failed")
        )))
        self._terminate()
        raise failure

  def _send(self, message: Mapping[str, Any]) -> None:
    process = self._process
    if process is None or process.stdin is None:
      raise self._failure("persistent AF3 child is unavailable")
    data = (json.dumps(_jsonable(message), default=_json_default, allow_nan=False) + "\n").encode("utf-8")
    try:
      process.stdin.write(data)
      process.stdin.flush()
    except (BrokenPipeError, OSError) as exc:
      raise self._failure("persistent AF3 child protocol write failed") from exc

  def submit(self, payload: Mapping[str, Any], workdir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with self._lock:
      if self._closed:
        raise AF3WorkerError("persistent AF3 service is closed")
      if self._fatal is not None:
        raise self.failure()
      self._start()
      try:
        self._send({"type": "predict", "payload": dict(payload), "workdir": str(workdir)})
        response = self._read_message(self._timeout())
      except Exception as exc:
        # Only the owner may replace a broken service, with a bounded retry.
        failure = self._record_failure(exc)
        self._terminate()
        if failure is exc:
          raise
        raise failure from exc
      if response.get("type") == "error":
        message = str(response.get("message", "AF3 child request failed"))
        if bool(response.get("fatal")):
          failure = self._record_failure(AF3WorkerError(message))
          self._terminate()
          raise failure
        if response.get("error_type") == "AF3InputError":
          raise AF3InputError(message)
        raise AF3OutputError(message)
      if response.get("type") != "result" or not isinstance(response.get("predictions"), list):
        failure = self._record_failure(AF3WorkerError(
            "persistent AF3 child returned an invalid prediction response"
        ))
        self._terminate()
        raise failure
      if not all(isinstance(item, Mapping) for item in response["predictions"]):
        failure = self._record_failure(AF3WorkerError(
            "persistent AF3 child returned a non-mapping prediction"
        ))
        self._terminate()
        raise failure
      return [dict(item) for item in response["predictions"]]

  def close(self) -> None:
    close_error: AF3WorkerError | None = None
    with self._lock:
      if self._closed:
        return
      self._closed = True
      process = self._process
      try:
        if process is not None and self._fatal is None:
          self._send({"type": "close"})
          response = self._read_message(min(30.0, self._timeout()))
          if response.get("type") != "closed":
            message = (
                "persistent AF3 child did not acknowledge close: "
                f"{response.get('type', 'missing')!r}"
            )
            close_error = self._record_failure(self._failure(message))
      except Exception as exc:
        if isinstance(exc, AF3WorkerError):
          close_error = exc
        else:
          close_error = self._failure("persistent AF3 child close failed")
        close_error = self._record_failure(close_error)
      finally:
        # Reap the child even when the close acknowledgement is malformed or
        # the pipe fails.  Never report a false successful close.
        self._terminate()
    if close_error is not None:
      raise close_error


def _service_child_main(args: argparse.Namespace) -> None:
  protocol_stdout = sys.stdout
  try:
    config = json.loads(args.config_json)
    if not isinstance(config, Mapping):
      raise ValueError("AF3 child config must be a mapping")
    device = args.device if args.device else None
    worker = AF3Worker(
        args.repo_root,
        config,
        method=args.method,
        device=device,
        _child=True,
    )
  except Exception as exc:
    protocol_stdout.write(json.dumps({
        "type": "startup_error",
        "schema_version": 1,
        "message": f"{type(exc).__name__}: {exc}",
    }) + "\n")
    protocol_stdout.flush()
    return

  child_log_dir = None if args.log_dir is None else Path(args.log_dir)

  def send(message: Mapping[str, Any]) -> None:
    protocol_stdout.write(
        json.dumps(_jsonable(message), default=_json_default, allow_nan=False) + "\n"
    )
    protocol_stdout.flush()

  send({"type": "ready", "schema_version": 1})
  for line in sys.stdin:
    if not line.strip():
      continue
    try:
      message = json.loads(line)
      if not isinstance(message, Mapping):
        raise ValueError("AF3 child request must be a mapping")
      request_type = str(message.get("type", ""))
      if request_type == "close":
        send({"type": "closed", "schema_version": 1})
        break
      if request_type == "set_log_dir":
        path = message.get("path")
        child_log_dir = None if path is None else Path(str(path)).expanduser().resolve()
        send({"type": "log_dir_set", "schema_version": 1})
        continue
      if request_type != "predict":
        raise ValueError(f"unknown AF3 child request type {request_type!r}")
      payload = message.get("payload")
      if not isinstance(payload, Mapping):
        raise AF3InputError("AF3 child payload must be a mapping")
      with _redirect_native_output(child_log_dir):
        predictions = worker._predict_direct(
            payload,
            str(message.get("workdir", ".")),
        )
      send({"type": "result", "schema_version": 1, "predictions": predictions})
    except Exception as exc:
      fatal = not isinstance(exc, (AF3InputError, AF3OutputError))
      send({
          "type": "error",
          "schema_version": 1,
          "fatal": fatal,
          "error_type": type(exc).__name__,
          "message": f"{type(exc).__name__}: {exc}",
      })
      if fatal:
        break

def _get(config: Any, key: str, default: Any = None) -> Any:
  if isinstance(config, Mapping):
    if key in config:
      return config[key]
    for section_name in ("af3",):
      section = config.get(section_name)
      if isinstance(section, Mapping) and key in section:
        return section[key]
    return default
  value = getattr(config, key, None)
  if value is not None:
    return value
  for section_name in ("af3",):
    section = getattr(config, section_name, None)
    if isinstance(section, Mapping) and key in section:
      return section[key]
    if section is not None and hasattr(section, key):
      return getattr(section, key)
  return default


def _sdk_source(repo_root: Path, method: str) -> Path:
  del repo_root, method
  source = af3_source()
  candidate = source / "src"
  if (candidate / "alphafold3").is_dir():
    return candidate.resolve()
  raise AF3WorkerError(
      f"AlphaFold 3 SDK source is missing: {source}; install the packaged community source"
  )


def _prepare_absl_flags() -> None:
  """Parse an empty Abseil argv exactly once after SDK flag definitions.

  ``run_alphafold`` defines global flags at import time.  A design CLI may use
  argparse or Hydra before reaching this worker, so forwarding its argv would
  make Abseil consume unrelated options.  An already parsed flag registry is
  left untouched; otherwise it is initialized with only the worker program
  name.
  """
  try:
    from absl import flags
  except Exception:
    return
  if flags.FLAGS.is_parsed():
    return
  original_argv = sys.argv
  try:
    sys.argv = ["af3-native-worker"]
    flags.FLAGS(["af3-native-worker"])
  finally:
    sys.argv = original_argv


def _normalise_method(method: str) -> str:
  value = str(method).strip().lower()
  if value in {"af3", "stock", "af3-stock"}:
    return "af3"
  if value in {"af3-la", "la", "leaving-atom", "leaving_atom"}:
    return "af3-la"
  raise AF3WorkerError(f"AF3 method must be af3 or af3-la, got {method!r}")


def _configure_ccd_environment(repo_root: Path, config: Any) -> None:
  """Point SDK/resource imports at the shared CCD before they are imported."""
  configured = _get(
      config,
      "externalccd_dir",
      _get(config, "externalccd_pickle_path", _get(config, "ccd_pickle_path", None)),
  )
  if configured is None:
    from partidesign.paths import af3_asset
    ccd_dir = af3_asset("af3", repo_root) / "ccd"
  else:
    raw = Path(str(configured)).expanduser()
    if not raw.is_absolute():
      raw = repo_root / raw
    ccd_dir = raw if raw.is_dir() else raw.parent
  # Resolve this worker owner's asset root before SDK imports. Do not inherit
  # an automatically configured CCD directory from a different model root.
  os.environ["PARTIDESIGN_AF3_CCD_DIR"] = str(ccd_dir.resolve())
  os.environ["LIBCIFPP_DATA_DIR"] = str(ccd_dir.resolve())


def _as_path(value: Any) -> Path | None:
  if value is None:
    return None
  return Path(str(value)).expanduser().resolve()


def _finite_metric(value: Any) -> float | None:
  if value is None:
    return None
  try:
    scalar = float(np.asarray(value).reshape(-1)[0])
  except (TypeError, ValueError, IndexError):
    return None
  return scalar if np.isfinite(scalar) else None


def _ordered_chain_ids(value: Any) -> list[str]:
  if value is None:
    return []
  if isinstance(value, Mapping):
    values = list(value)
  elif isinstance(value, np.ndarray):
    values = value.reshape(-1).tolist()
  elif isinstance(value, (str, bytes, bytearray)):
    values = [value]
  else:
    try:
      values = list(value)
    except TypeError:
      values = [value]
  return [str(item) for item in values]


def _unique_chain_ids(value: Any) -> list[str]:
  result: list[str] = []
  for name in _ordered_chain_ids(value):
    if name not in result:
      result.append(name)
  return result


def _mapping_value(mapping: Mapping[Any, Any], key: str) -> Any:
  if key in mapping:
    return mapping[key]
  for candidate, value in mapping.items():
    if str(candidate) == str(key):
      return value
  return None


def _confidence_array(
    raw: Any, field: str, shape: tuple[int, ...], *,
    nan_mask: Any = None, upper_bound: float | None = None,
) -> np.ndarray:
  try:
    values = np.asarray(raw)
    if values.dtype.kind not in "iuf":
      raise TypeError("confidence must contain numbers")
    values = values.astype(np.float64, copy=False)
  except (TypeError, ValueError) as exc:
    raise AF3OutputError(f"native {field} is not numeric") from exc
  if values.shape != shape:
    raise AF3OutputError(f"native {field} shape {values.shape} does not match {shape}")
  if nan_mask is None:
    if not np.isfinite(values).all():
      raise AF3OutputError(f"native {field} contains invalid values")
  else:
    allowed = np.asarray(nan_mask, dtype=bool)
    if allowed.shape != shape:
      raise AF3OutputError(f"native {field} undefined mask has invalid shape")
    if np.isinf(values).any() or np.any(np.isnan(values) & ~allowed):
      raise AF3OutputError(f"native {field} contains invalid values")
  # Only permitted NaNs remain; their comparisons are false. Avoid materializing
  # a second full matrix of finite entries just to check the bounds.
  if np.any(values < 0.0) or (upper_bound is not None and np.any(values > upper_bound)):
    bounds = "nonnegative" if upper_bound is None else f"in [0, {upper_bound:g}]"
    raise AF3OutputError(f"native {field} values must be {bounds}")
  return values


def _chain_metric(
    metadata: Mapping[str, Any], aliases: tuple[str, ...],
    chain_ids: Sequence[str], field: str, *, nan_mask: Any = None,
    upper_bound: float | None = 1.0,
    native_arrays: dict[str, np.ndarray] | None = None,
) -> list[float | None] | None:
  result = None
  for name in aliases:
    if name not in metadata:
      continue
    raw = metadata[name]
    if isinstance(raw, Mapping):
      raw = [_mapping_value(raw, chain_id) for chain_id in chain_ids]
    values = _confidence_array(raw, name, (len(chain_ids),),
                               nan_mask=nan_mask, upper_bound=upper_bound)
    if native_arrays is not None:
      native_arrays[name] = values
    if result is None:
      result = _jsonable(values)
  return result


def _chain_pair_metric(
    metadata: Mapping[str, Any], aliases: tuple[str, ...],
    chain_ids: Sequence[str], field: str, *, nan_mask: Any = None,
    upper_bound: float | None = 1.0,
    native_arrays: dict[str, np.ndarray] | None = None,
) -> list[list[float | None]] | None:
  result = None
  for name in aliases:
    if name not in metadata:
      continue
    raw = metadata[name]
    if isinstance(raw, Mapping):
      rows = []
      for left in chain_ids:
        row = _mapping_value(raw, left)
        if not isinstance(row, Mapping):
          raise AF3OutputError(f"native {field} row {left!r} is not a mapping")
        rows.append([_mapping_value(row, right) for right in chain_ids])
      raw = rows
    values = _confidence_array(raw, name, (len(chain_ids), len(chain_ids)),
                               nan_mask=nan_mask, upper_bound=upper_bound)
    if native_arrays is not None:
      native_arrays[name] = values
    if result is None:
      result = _jsonable(values)
  return result


# Single-atom nonpolymer ions may have no native backbone frame. This guard
# deliberately excludes ordinary one/two-atom molecular ligands.
_ION_ELEMENTS = frozenset(
    "LI BE F NA MG AL CL K CA SC TI V CR MN FE CO NI CU ZN GA BR RB SR Y ZR "
    "NB MO TC RU RH PD AG CD IN SN I CS BA LA CE PR ND PM SM EU GD TB DY HO "
    "ER TM YB LU HF TA W RE OS IR PT AU HG TL PB BI FR RA AC TH PA U NP PU "
    "AM CM BK CF ES FM MD NO LR".split()
)


def _native_confidence_masks(
    payload: Mapping[str, Any], native_atoms: Sequence[Mapping[str, Any]],
    token_chain_ids: Sequence[str], chain_ids: Sequence[str], frame_mask: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  count = len(token_chain_ids)
  if frame_mask is None:
    # Test adapters without CPU features cannot claim an undefined ion frame.
    return np.zeros(len(chain_ids), dtype=bool), np.zeros(len(chain_ids), dtype=bool), None
  frames = np.asarray(frame_mask)
  if frames.dtype.kind != "b" or frames.shape != (count,):
    raise AF3OutputError("native CPU frame mask differs from confidence tokens")
  entries = {str(entry["id"]): entry for entry in payload.get("chains", ())}
  no_frames, ions = [], []
  for chain in chain_ids:
    token_mask = np.asarray([value == chain for value in token_chain_ids], dtype=bool)
    if not token_mask.any():
      raise AF3OutputError(f"native confidence chain {chain!r} has no tokens")
    atoms = [atom for atom in native_atoms if str(atom["chain_id"]) == chain]
    source = payload["native_to_source_chain"][chain]
    source_atoms = [atom for atom in payload["source_atom_ledger"]
                    if str(atom["chain_id"]) == str(source)]
    element = str(atoms[0].get("atom_element", atoms[0].get("element", ""))).upper() if len(atoms) == 1 else ""
    ion = (entries.get(chain, {}).get("modality") == "ligand"
           and int(token_mask.sum()) == 1 and len(atoms) == 1 and len(source_atoms) == 1
           and str(source_atoms[0].get("mol_type", "")) == "ligand"
           and element in _ION_ELEMENTS
           and str(source_atoms[0].get("element", "")).upper() == element)
    ions.append(ion)
    no_frames.append(not frames[token_mask].any() and ion)
  return np.asarray(no_frames, dtype=bool), np.asarray(ions, dtype=bool), frames.copy()


def _validate_native_metadata(
    metadata: Mapping[str, Any], chain_ids: Sequence[str], no_frames: np.ndarray,
) -> dict[str, Any]:
  """Validate SDK arrays before JSON conversion can erase NaN versus infinity."""
  n = len(chain_ids)
  pair_tm_nan = no_frames[:, None] & no_frames[None, :]
  pair_pae_nan = np.broadcast_to(no_frames[:, None], (n, n))
  cross_nan = np.full(n, n == 1 or bool(no_frames.all()), dtype=bool)
  masks: dict[str, Any] = {}
  native_arrays: dict[str, np.ndarray] = {}
  checked: set[str] = set()

  def chain(aliases, mask=None, upper=1.0):
    value = _chain_metric(metadata, aliases, chain_ids, aliases[0],
                          nan_mask=mask, upper_bound=upper, native_arrays=native_arrays)
    for name in aliases:
      if name in metadata:
        checked.add(name)
        masks[name] = mask
    return value

  def pair(aliases, mask=None, upper=1.0):
    value = _chain_pair_metric(metadata, aliases, chain_ids, aliases[0],
                               nan_mask=mask, upper_bound=upper, native_arrays=native_arrays)
    for name in aliases:
      if name in metadata:
        checked.add(name)
        masks[name] = mask
    return value

  ptm = chain(("chain_ptm", "iptm_ichain"), no_frames)
  iptm = chain(("chain_iptm", "iptm_xchain"), cross_nan)
  pair_iptm = pair(("chain_pair_iptm", "pair_iptm"), pair_tm_nan)
  pair(("chain_pair_pae_min",), pair_pae_nan, None)
  pair(("chain_pair_pae_mean",), pair_pae_nan, None)
  pair(("chain_pair_pde_min",), upper=None)
  pair(("chain_pair_pde_mean",), upper=None)
  chain(("pae_ichain",), no_frames, None)
  chain(("pae_xchain",), cross_nan, None)
  chain(("intra_chain_single_pde",), upper=None)
  chain(("cross_chain_single_pde",), upper=None)

  scalars: dict[str, float | None] = {}
  for aliases, allowed in (
      (("ptm", "predicted_tm_score"), bool(no_frames.all())),
      (("iptm", "interface_predicted_tm_score"), n == 1 or bool(no_frames.all())),
      (("ptm_iptm_average",), n == 1 or bool(no_frames.all())),
      (("ranking_confidence",), bool(no_frames.all())),
      (("fraction_disordered",), False),
  ):
    for name in aliases:
      if name in metadata:
        values = _confidence_array(metadata[name], name, (), nan_mask=np.asarray(allowed), upper_bound=1.0)
        checked.add(name)
        native_arrays[name] = values
        masks[name] = np.asarray(allowed)
        if aliases[0] not in scalars:
          scalars[aliases[0]] = None if np.isnan(values) else float(values)

  def finite_rest(value, field):
    if isinstance(value, Mapping):
      for key, item in value.items():
        finite_rest(item, f"{field}.{key}")
    elif isinstance(value, np.ndarray) and value.ndim == 0:
      finite_rest(value.item(), field)
    elif isinstance(value, (list, tuple, np.ndarray)):
      for index, item in enumerate(value):
        finite_rest(item, f"{field}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
      raise AF3OutputError(f"native {field} contains non-finite values")

  for name, value in metadata.items():
    if name not in checked:
      finite_rest(value, name)
  return {"chain_ptm": ptm, "chain_iptm": iptm, "chain_pair_iptm": pair_iptm,
          "ptm": scalars.get("ptm"), "iptm": scalars.get("iptm"),
          "undefined_masks": masks, "native_arrays": native_arrays,
          "nan_masks": {name: np.isnan(value) for name, value in native_arrays.items()}}


def _flatten_cif_file(cif_file: Any) -> dict[str, list[Any]]:
  """Flatten a biotite CIFFile into the SDK metadata ABI."""
  block = next(iter(cif_file.values()))
  result: dict[str, list[Any]] = {}
  for category_name, category in block.items():
    for column_name, column in category.items():
      result[f"_{category_name}.{column_name}"] = column.as_array().tolist()
  return result



def _ccd_values(entry: Any, field: str) -> list[Any]:
  """Return one CCD column without treating a scalar as a row sequence."""
  if entry is None or not hasattr(entry, "get"):
    return []
  value = entry.get(field, ())
  if value is None:
    return []
  if isinstance(value, (str, bytes)):
    return [value]
  try:
    return list(value)
  except TypeError:
    return []


def _ccd_marker(value: Any) -> str:
  token = str(value).strip()
  return "?" if token in {"", ".", "?"} else token


def _component_graph(entry: Any, component_id: str) -> dict[str, Any] | None:
  """Extract the authoritative heavy graph and chemical annotations from CCD."""
  names = _ccd_values(entry, "_chem_comp_atom.atom_id")
  elements = _ccd_values(entry, "_chem_comp_atom.type_symbol")
  if not names and not elements:
    return None
  if not names or len(names) != len(elements):
    raise AF3InputError(
        f"CCD component {component_id!r} has an incomplete atom loop"
    )
  charges = _ccd_values(entry, "_chem_comp_atom.charge")
  if not charges:
    charges = _ccd_values(entry, "_chem_comp_atom.pdbx_formal_charge")
  atom_aromatic = _ccd_values(entry, "_chem_comp_atom.pdbx_aromatic_flag")
  atom_stereo = _ccd_values(entry, "_chem_comp_atom.pdbx_stereo_config")
  for column_name, values in (
      ("charge", charges),
      ("pdbx_aromatic_flag", atom_aromatic),
      ("pdbx_stereo_config", atom_stereo),
  ):
    if values and len(values) != len(names):
      raise AF3InputError(
          f"CCD component {component_id!r} has an incomplete atom {column_name} loop"
      )
  atoms: list[dict[str, str]] = []
  by_name: dict[str, int] = {}
  heavy_indexes: list[int] = []
  for index, (raw_name, raw_element) in enumerate(zip(names, elements, strict=True)):
    name = str(raw_name).strip()
    element = str(raw_element).strip().upper()
    if not name or name in {".", "?"} or any(char.isspace() for char in name):
      raise AF3InputError(
          f"CCD component {component_id!r} has an invalid atom name"
      )
    if not element or element in {".", "?"}:
      raise AF3InputError(
          f"CCD component {component_id!r} has an invalid atom element"
      )
    if name in by_name:
      raise AF3InputError(
          f"CCD component {component_id!r} repeats atom {name!r}"
      )
    by_name[name] = index
    if element not in {"H", "D", "T"}:
      heavy_indexes.append(index)
    atoms.append({
        "name": name,
        "element": element,
        "charge": _ccd_marker(charges[index]) if charges else "?",
        "aromatic": _ccd_marker(atom_aromatic[index]) if atom_aromatic else "?",
        "stereo": _ccd_marker(atom_stereo[index]) if atom_stereo else "?",
    })
  if not heavy_indexes:
    raise AF3InputError(f"CCD component {component_id!r} has no heavy atoms")

  atom1 = _ccd_values(entry, "_chem_comp_bond.atom_id_1")
  atom2 = _ccd_values(entry, "_chem_comp_bond.atom_id_2")
  orders = _ccd_values(entry, "_chem_comp_bond.value_order")
  aromatic = _ccd_values(entry, "_chem_comp_bond.pdbx_aromatic_flag")
  stereo = _ccd_values(entry, "_chem_comp_bond.pdbx_stereo_config")
  bond_count = len(atom1)
  if len(atom2) != bond_count or len(orders) != bond_count:
    if bond_count or atom2 or orders:
      raise AF3InputError(
          f"CCD component {component_id!r} has an incomplete bond loop"
      )
  for column_name, values in (
      ("pdbx_aromatic_flag", aromatic),
      ("pdbx_stereo_config", stereo),
  ):
    if values and len(values) != bond_count:
      raise AF3InputError(
          f"CCD component {component_id!r} has an incomplete bond {column_name} loop"
      )
  heavy_names = {atoms[index]["name"] for index in heavy_indexes}
  bonds: list[dict[str, str]] = []
  seen_bonds: dict[tuple[str, str], tuple[str, str, str]] = {}
  allowed_orders = {"SING", "DOUB", "TRIP", "AROM"}
  for index, (raw_left, raw_right, raw_order) in enumerate(
      zip(atom1, atom2, orders, strict=True)
  ):
    left, right = str(raw_left).strip(), str(raw_right).strip()
    if left not in by_name or right not in by_name or left == right:
      raise AF3InputError(
          f"CCD component {component_id!r} has an invalid bond endpoint"
      )
    if left not in heavy_names or right not in heavy_names:
      # Hydrogens remain private to the CCD; the published atom table is
      # intentionally heavy-atom-only.
      continue
    order = _ccd_marker(raw_order).upper()
    if order not in allowed_orders:
      raise AF3InputError(
          f"CCD component {component_id!r} has unsupported bond order {order!r}"
      )
    aromatic_flag = _ccd_marker(aromatic[index]).upper() if aromatic else "?"
    stereo_config = _ccd_marker(stereo[index]) if stereo else "?"
    if aromatic_flag not in {"?", "Y", "N"}:
      raise AF3InputError(
          f"CCD component {component_id!r} has invalid aromatic flag"
      )
    key = tuple(sorted((left, right)))
    signature = (order, aromatic_flag, stereo_config)
    previous = seen_bonds.get(key)
    if previous is not None and previous != signature:
      raise AF3InputError(
          f"CCD component {component_id!r} repeats a bond with conflicting facts"
      )
    seen_bonds[key] = signature
  bonds = [
      {
          "atom_id_1": pair[0],
          "atom_id_2": pair[1],
          "value_order": facts[0],
          "aromatic": facts[1],
          "stereo": facts[2],
      }
      for pair, facts in sorted(seen_bonds.items())
  ]
  return {
      "component_id": str(component_id).strip().upper(),
      "atoms": [atoms[index] for index in heavy_indexes],
      "bonds": bonds,
  }


def _component_graph_signature(graph: Mapping[str, Any]) -> tuple[Any, ...]:
  atoms = tuple(
      sorted(
          (
              str(atom["name"]),
              str(atom["element"]),
              str(atom.get("charge", "?")),
              str(atom.get("aromatic", "?")),
              str(atom.get("stereo", "?")),
          )
          for atom in graph["atoms"]
      )
  )
  bonds = tuple(
      sorted(
          (
              min(str(bond["atom_id_1"]), str(bond["atom_id_2"])),
              max(str(bond["atom_id_1"]), str(bond["atom_id_2"])),
              str(bond["value_order"]),
              str(bond.get("aromatic", "?")),
              str(bond.get("stereo", "?")),
          )
          for bond in graph["bonds"]
      )
  )
  return atoms, bonds


def _parse_user_component_graphs(user_ccd: str | None) -> dict[str, dict[str, Any]]:
  if not user_ccd:
    return {}
  try:
    from .af3_input import _parse_user_ccd
    entries = _parse_user_ccd(str(user_ccd))
  except AF3InputError:
    raise
  except Exception as exc:
    raise AF3InputError(
        f"native user CCD cannot be parsed: {type(exc).__name__}: {exc}"
    ) from exc
  result: dict[str, dict[str, Any]] = {}
  for raw_id, entry in entries.items():
    component_id = str(raw_id).strip().upper()
    graph = _component_graph(entry, component_id)
    if graph is None:
      raise AF3InputError(
          f"native user CCD component {component_id!r} has no atom graph"
      )
    result[component_id] = graph
  return result


def _cached_ccd_for_payload(worker: Any, payload: Mapping[str, Any]) -> Any | None:
  """Use the Ccd object already resident in the worker, without reloading it."""
  wanted = "" if payload.get("user_ccd") is None else str(payload.get("user_ccd"))
  cache = getattr(worker, "_ccd_cache", {})
  if not isinstance(cache, Mapping):
    return None
  for key, value in cache.items():
    try:
      if len(key) >= 2 and str(key[1]) == wanted:
        return value
    except (TypeError, IndexError):
      continue
  return None


def _category_rows(category: Any) -> tuple[list[str], list[dict[str, Any]]]:
  if category is None:
    return [], []
  columns = [str(name) for name in category.keys()]
  arrays = {name: category[name].as_array().tolist() for name in columns}
  rows = [
      {name: arrays[name][index] for name in columns}
      for index in range(category.row_count)
  ]
  return columns, rows


def _set_cif_category(block: Any, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
  from biotite.structure.io.pdbx import CIFCategory
  if not rows:
    return
  columns: list[str] = []
  for row in rows:
    for key in row:
      if key not in columns:
        columns.append(str(key))
  if not columns:
    return
  values = {
      column: [row.get(column, "?") for row in rows]
      for column in columns
  }
  block[name] = CIFCategory(values, name=name)


def _add_authoritative_component_graphs(
    block: Any,
    payload: Mapping[str, Any],
    restored_rows: Sequence[Mapping[str, Any]],
    *,
    ccd: Any | None,
) -> None:
  """Publish source-named CCD graphs so standalone readers need no native alias."""
  user_graphs = _parse_user_component_graphs(payload.get("user_ccd"))
  native_to_source = {
      str(native).strip().upper(): str(source).strip().upper()
      for source, native in dict(payload.get("native_component_map", {})).items()
  }
  source_to_native = {
      str(source).strip().upper(): str(native).strip().upper()
      for source, native in dict(payload.get("native_component_map", {})).items()
  }
  ccd_graphs: dict[str, dict[str, Any]] = {}
  from ..model.constants import DNA_RES_ATOMS, PROTEIN_RES_ATOMS, RNA_RES_ATOMS
  standard_components = set(PROTEIN_RES_ATOMS) | set(RNA_RES_ATOMS) | set(DNA_RES_ATOMS)

  def needs_graph(row: Mapping[str, Any]) -> bool:
    return str(row.get("mol_type", "")) == "ligand" or str(row.get("res_name", "")).strip().upper() not in standard_components

  if ccd is not None and hasattr(ccd, "get"):
    candidate_ids = set(native_to_source)
    candidate_ids.update(native_to_source.values())
    candidate_ids.update(
        str(row.get("native_res_name", "")).strip().upper()
        for row in restored_rows if needs_graph(row)
    )
    candidate_ids.update(
        str(row.get("res_name", "")).strip().upper()
        for row in restored_rows if needs_graph(row)
    )
    for component_id in candidate_ids:
      entry = ccd.get(component_id)
      if entry is None:
        continue
      graph = _component_graph(entry, component_id)
      if graph is not None:
        ccd_graphs[component_id] = graph

  def candidate_graph(source_component: str, native_components: set[str]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    mapped_native = source_to_native.get(source_component)
    candidate_ids = (
        (mapped_native, *sorted(native_components))
        if mapped_native is not None
        else (source_component, *sorted(native_components))
    )
    for component_id in candidate_ids:
      if component_id is None:
        continue
      normalized = str(component_id).strip().upper()
      if normalized in seen_ids:
        continue
      seen_ids.add(normalized)
      user_graph = user_graphs.get(normalized)
      native_graph = ccd_graphs.get(normalized)
      if user_graph is not None and native_graph is not None:
        if _component_graph_signature(user_graph) != _component_graph_signature(native_graph):
          raise AF3InputError(
              f"published component {source_component!r} has conflicting "
              f"authoritative graphs for {normalized!r}"
          )
      graph = user_graph or native_graph
      if graph is not None:
        candidates.append(graph)
    if not candidates:
      return None
    signature = _component_graph_signature(candidates[0])
    if any(_component_graph_signature(item) != signature for item in candidates[1:]):
      raise AF3InputError(
          f"published component {source_component!r} has conflicting authoritative graphs"
      )
    return candidates[0]

  instances: dict[str, list[list[Mapping[str, Any]]]] = {}
  for row in restored_rows:
    if not needs_graph(row):
      continue
    source_component = str(row.get("res_name", "")).strip().upper()
    key = (
        str(row.get("chain_id", "")),
        int(row.get("res_id", 0)),
        str(row.get("ins_code", "")),
        source_component,
    )
    bucket = next(
        (
            value for value in instances.setdefault(source_component, [])
            if value and (
                str(value[0].get("chain_id", "")),
                int(value[0].get("res_id", 0)),
                str(value[0].get("ins_code", "")),
                str(value[0].get("res_name", "")).strip().upper(),
            ) == key
        ),
        None,
    )
    if bucket is None:
      bucket = []
      instances[source_component].append(bucket)
    bucket.append(row)

  graphs: dict[str, dict[str, Any]] = {}
  for source_component, component_instances in instances.items():
    native_components = {
        str(row.get("native_res_name", "")).strip().upper()
        for instance in component_instances
        for row in instance
    }
    graph = candidate_graph(source_component, native_components)
    if graph is None:
      if any(len(instance) > 1 for instance in component_instances):
        raise AF3InputError(
            f"published ligand component {source_component!r} has no "
            "authoritative atom graph"
        )
      # A one-atom ligand has no internal edge for the standalone reader to
      # recover; leave its component metadata untouched.
      continue
    published_graph: dict[str, Any] | None = None
    for instance in component_instances:
      mapped_names: dict[str, str] = {}
      rows_by_name: dict[str, list[Mapping[str, Any]]] = {}
      rows_by_native_name: dict[str, list[Mapping[str, Any]]] = {}
      for row in instance:
        rows_by_name.setdefault(str(row.get("atom_name", "")).strip(), []).append(row)
        rows_by_native_name.setdefault(
            str(row.get("native_atom_name", "")).strip(), []
        ).append(row)
      for atom in graph["atoms"]:
        graph_name = str(atom["name"])
        candidates = rows_by_name.get(graph_name, [])
        if len(candidates) != 1:
          candidates = rows_by_native_name.get(graph_name, [])
        if len(candidates) != 1:
          raise AF3InputError(
              f"published component {source_component!r} cannot map CCD atom "
              f"{graph_name!r} to one output atom"
          )
        row = candidates[0]
        if str(row.get("element", "")).strip().upper() != str(atom["element"]).upper():
          raise AF3InputError(
              f"published component {source_component!r} changed element for "
              f"atom {graph_name!r}"
          )
        mapped_names[graph_name] = str(row["atom_name"])
      output_names = {str(row.get("atom_name", "")).strip() for row in instance}
      if output_names != set(mapped_names.values()):
        raise AF3InputError(
            f"published component {source_component!r} has an atom inventory "
            "different from its authoritative CCD graph"
        )
      renamed = {
          "component_id": source_component,
          "atoms": [
              {**atom, "name": mapped_names[str(atom["name"])]}
              for atom in graph["atoms"]
          ],
          "bonds": [
              {
                  **bond,
                  "atom_id_1": mapped_names[str(bond["atom_id_1"])],
                  "atom_id_2": mapped_names[str(bond["atom_id_2"])],
              }
              for bond in graph["bonds"]
          ],
      }
      if published_graph is None:
        published_graph = renamed
      elif _component_graph_signature(published_graph) != _component_graph_signature(renamed):
        raise AF3InputError(
            f"published component {source_component!r} has repeated conflicting graphs"
        )
    if published_graph is not None:
      previous = graphs.get(source_component)
      if previous is not None and (
          _component_graph_signature(previous)
          != _component_graph_signature(published_graph)
      ):
        raise AF3InputError(
            f"published component {source_component!r} is reused with a conflicting graph"
        )
      graphs[source_component] = published_graph

  if not graphs:
    return
  graph_ids = set(graphs)
  reverse = {
      str(native).strip().upper(): str(source).strip().upper()
      for native, source in native_to_source.items()
  }

  comp_columns, comp_rows = _category_rows(block.get("chem_comp"))
  normalized_comp_rows: list[dict[str, Any]] = []
  seen_comp_ids: set[str] = set()
  for row in comp_rows:
    raw = str(row.get("id", "")).strip().upper()
    published = reverse.get(raw, raw)
    if not published or published in {".", "?"}:
      continue
    row = dict(row)
    row["id"] = published
    if published in seen_comp_ids:
      continue
    seen_comp_ids.add(published)
    normalized_comp_rows.append(row)
  for component_id in sorted(graph_ids):
    if component_id not in seen_comp_ids:
      normalized_comp_rows.append({
          "id": component_id,
          "type": "NON-POLYMER",
          "mon_nstd_flag": "y",
      })
      seen_comp_ids.add(component_id)
  _set_cif_category(block, "chem_comp", normalized_comp_rows)

  _atom_columns, atom_rows = _category_rows(block.get("chem_comp_atom"))
  retained_atoms: list[dict[str, Any]] = []
  for row in atom_rows:
    raw = str(row.get("comp_id", "")).strip().upper()
    published = reverse.get(raw, raw)
    if published in graph_ids:
      continue
    row = dict(row)
    row["comp_id"] = published
    retained_atoms.append(row)
  generated_atoms: list[dict[str, Any]] = []
  for component_id in sorted(graph_ids):
    graph = graphs[component_id]
    for ordinal, atom in enumerate(graph["atoms"], start=1):
      generated_atoms.append({
          "comp_id": component_id,
          "atom_id": atom["name"],
          "type_symbol": atom["element"],
          "charge": atom.get("charge", "?"),
          "pdbx_formal_charge": atom.get("charge", "?"),
          "pdbx_aromatic_flag": atom.get("aromatic", "?"),
          "pdbx_stereo_config": atom.get("stereo", "?"),
          "pdbx_ordinal": str(ordinal),
      })
  _set_cif_category(block, "chem_comp_atom", [*retained_atoms, *generated_atoms])

  _bond_columns, bond_rows = _category_rows(block.get("chem_comp_bond"))
  retained_bonds: list[dict[str, Any]] = []
  for row in bond_rows:
    raw = str(row.get("comp_id", "")).strip().upper()
    published = reverse.get(raw, raw)
    if published in graph_ids:
      continue
    row = dict(row)
    row["comp_id"] = published
    retained_bonds.append(row)
  generated_bonds: list[dict[str, Any]] = []
  for component_id in sorted(graph_ids):
    graph = graphs[component_id]
    for ordinal, bond in enumerate(graph["bonds"], start=1):
      generated_bonds.append({
          "comp_id": component_id,
          "atom_id_1": bond["atom_id_1"],
          "atom_id_2": bond["atom_id_2"],
          "value_order": bond["value_order"],
          "pdbx_aromatic_flag": bond.get("aromatic", "?"),
          "pdbx_stereo_config": bond.get("stereo", "?"),
          "pdbx_ordinal": str(ordinal),
      })
  _set_cif_category(block, "chem_comp_bond", [*retained_bonds, *generated_bonds])

  # Keep component scheme tables consistent for native readers and metadata tools.
  for category_name in ("pdbx_nonpoly_scheme", "pdbx_poly_seq_scheme"):
    category = block.get(category_name)
    if category is None or "mon_id" not in category:
      continue
    values = [
        reverse.get(str(value).strip().upper(), str(value).strip())
        for value in category["mon_id"].as_array()
    ]
    category["mon_id"] = values


def _restore_authoritative_cif(
    structure: Any,
    payload: Mapping[str, Any],
    native_cif: str,
    *,
    ccd: Any | None = None,
) -> tuple[str, dict[str, list[Any]], list[dict[str, Any]]]:
  """Restore source locators in the authoritative output CIF.

  AF3's structure object intentionally uses its own chain and residue labels.
  The public output must retain source chain IDs, residue IDs/insertion codes,
  and source component names while keeping native atom inventories for sequence
  mutations and generated atoms.  Every native atom and residue is checked
  against the source ledgers before a file is published.
  """
  try:
    from biotite.structure.io import pdbx
  except Exception as exc:  # pragma: no cover - package dependency
    raise AF3WorkerError("biotite CIF support is required for AF3 output") from exc

  try:
    cif_file = pdbx.CIFFile.deserialize(native_cif)
    block = next(iter(cif_file.values()))
    atom_site = block["atom_site"]
    atom_names = [str(value).strip() for value in atom_site["label_atom_id"].as_array()]
    atom_elements = (
        [str(value).strip().upper() for value in atom_site["type_symbol"].as_array()]
        if "type_symbol" in atom_site else []
    )
    atoms = list(structure.iter_atoms())
  except (KeyError, TypeError, ValueError, AttributeError) as exc:
    raise AF3InputError("native AF3 CIF atom table is malformed") from exc
  if len(atoms) != len(atom_names):
    raise AF3InputError(
        f"native AF3 CIF atom count {len(atom_names)} differs from structure "
        f"iterator count {len(atoms)}"
    )
  if not atoms:
    raise AF3InputError("native AF3 structure has no atoms")

  native_to_source = {
      str(native): str(source)
      for native, source in dict(payload.get("native_to_source_chain", {})).items()
  }
  from .af3_mapping import validate_source_mapping
  try:
    validate_source_mapping(payload)
  except (TypeError, KeyError, ValueError) as exc:
    raise AF3InputError(str(exc)) from exc

  source_residue_map = {
      str(chain): list(rows)
      for chain, rows in dict(payload.get("source_residue_map", {})).items()
  }
  source_ledger = list(payload.get("source_atom_ledger", ()))
  source_rows_by_residue: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
  source_atoms_by_key: dict[tuple[str, int, str, str], Mapping[str, Any]] = {}
  for row in source_ledger:
    source_chain = str(row.get("chain_id", ""))
    source_res_id = int(row.get("res_id", 0))
    source_ins_code = str(row.get("ins_code", ""))
    key = (source_chain, source_res_id, source_ins_code)
    source_rows_by_residue.setdefault(key, []).append(row)
    atom_key = (*key, str(row.get("atom_name", "")).strip())
    if atom_key in source_atoms_by_key:
      raise AF3InputError(f"duplicate source atom ledger key {atom_key!r}")
    source_atoms_by_key[atom_key] = row

  source_atoms_by_uid: dict[int, Mapping[str, Any]] = {}
  for row in source_ledger:
    raw_uid = row.get("atom_uid")
    if raw_uid is None:
      continue
    try:
      atom_uid = int(raw_uid)
    except (TypeError, ValueError) as exc:
      raise AF3InputError("source atom ledger contains a non-integer atom UID") from exc
    previous = source_atoms_by_uid.get(atom_uid)
    if previous is not None and dict(previous) != dict(row):
      raise AF3InputError(f"duplicate source atom UID {atom_uid}")
    source_atoms_by_uid[atom_uid] = row

  native_component_map = {
      str(source): str(native)
      for source, native in dict(payload.get("native_component_map", {})).items()
  }
  native_component_to_source = {
      native: source for source, native in native_component_map.items()
  }
  if len(native_component_to_source) != len(native_component_map):
    raise AF3InputError("AF3 custom component map is non-bijective")

  chain_entry_data = {
      str(entry.get("id", "")): dict(entry)
      for entry in payload.get("chains", ())
      if isinstance(entry, Mapping)
  }
  chain_entries = {
      chain_id: str(entry.get("modality", ""))
      for chain_id, entry in chain_entry_data.items()
  }
  native_atom_source_uids: dict[tuple[str, str], int] = {}
  for native_chain, entry in chain_entry_data.items():
    for map_name in ("smiles_atom_map", "molecule_atom_map"):
      mappings = entry.get(map_name, ())
      if mappings is None:
        continue
      if isinstance(mappings, Mapping) or isinstance(mappings, (str, bytes)):
        raise AF3InputError(f"native {map_name} for chain {native_chain!r} is malformed")
      try:
        mapping_rows = list(mappings)
      except TypeError as exc:
        raise AF3InputError(f"native {map_name} for chain {native_chain!r} is malformed") from exc
      for mapping in mapping_rows:
        if not isinstance(mapping, Mapping):
          raise AF3InputError(f"native {map_name} for chain {native_chain!r} is malformed")
        native_name = str(
            mapping.get("native_atom_name", mapping.get("compiler_atom_name", ""))
        ).strip()
        if not native_name:
          raise AF3InputError(f"native {map_name} for chain {native_chain!r} has no atom name")
        try:
          atom_uid = int(mapping["source_atom_uid"])
        except (KeyError, TypeError, ValueError) as exc:
          raise AF3InputError(
              f"native {map_name} for chain {native_chain!r} has no source atom UID"
          ) from exc
        key = (native_chain, native_name)
        previous = native_atom_source_uids.get(key)
        if previous is not None and previous != atom_uid:
          raise AF3InputError(
              f"native {map_name} maps {native_chain!r}/{native_name!r} to "
              "conflicting source atom UIDs"
          )
        native_atom_source_uids[key] = atom_uid
  # The source ledger is authoritative for supplied atoms.  Canonical polymer
  # inventories are used only to admit atoms that the native refold completed
  # from a standard parent residue (for example a missing side chain).
  try:
    from ..model.constants import DNA_RES_ATOMS, PROTEIN_RES_ATOMS, RNA_RES_ATOMS
  except Exception:  # pragma: no cover - design constants are present in normal runs
    DNA_RES_ATOMS = {}
    PROTEIN_RES_ATOMS = {}
    RNA_RES_ATOMS = {}
  canonical_atoms = {
      "protein": PROTEIN_RES_ATOMS,
      "rna": RNA_RES_ATOMS,
      "dna": DNA_RES_ATOMS,
  }
  parent_names = {
      "protein": {
          "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
          "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
          "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
          "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
          "X": "UNK",
      },
      "rna": {"A": "A", "G": "G", "C": "C", "U": "U", "N": "N"},
      "dna": {"A": "DA", "G": "DG", "C": "DC", "T": "DT", "N": "DN"},
  }

  def _element(value: Any, atom_name: str) -> str:
    token = str(value).strip().upper()
    if token in {"", ".", "?"}:
      letters = "".join(character for character in str(atom_name) if character.isalpha())
      token = letters[:1].upper() if letters else ""
    return token

  def _canonical_element(atom_name: str) -> str:
    name = str(atom_name).strip().upper()
    if name in {"SE"}:
      return "SE"
    if name in {"CL", "BR", "NA", "MG", "CA", "ZN", "FE"}:
      # Protein/nucleic-acid canonical CA is alpha carbon; atom-name context
      # below overrides this generic ion spelling where it is in an inventory.
      if name == "CA":
        return "C"
      return name
    letters = "".join(character for character in name if character.isalpha())
    return letters[:1] if letters else ""

  def _parent_name(
      native_chain: str,
      native_res_id: int,
      source_component: str,
      modality: str,
      *,
      prefer_sequence: bool = False,
  ) -> str | None:
    inventory = canonical_atoms.get(modality, {})
    entry = chain_entry_data.get(native_chain, {})
    sequence = str(entry.get("sequence", ""))
    sequence_parent = None
    if 1 <= native_res_id <= len(sequence):
      candidate = parent_names.get(modality, {}).get(sequence[native_res_id - 1].upper())
      if candidate in inventory:
        sequence_parent = candidate
    if prefer_sequence and sequence_parent is not None:
      return sequence_parent
    if source_component in inventory:
      return source_component
    return sequence_parent

  def _source_identity_atom_name(native_chain: str, native_atom_name: str) -> str:
    mapped_uid = native_atom_source_uids.get((native_chain, native_atom_name))
    if mapped_uid is None:
      return native_atom_name
    source_atom = source_atoms_by_uid.get(mapped_uid)
    if source_atom is None:
      raise AF3InputError(
          f"native atom {native_chain!r}/{native_atom_name!r} maps to "
          f"unknown source atom UID {mapped_uid}"
      )
    return str(source_atom.get("atom_name", native_atom_name)).strip()

  restored_rows: list[dict[str, Any]] = []
  native_lookup: dict[tuple[str, int, str, str], dict[str, Any]] = {}
  native_atoms_by_residue: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
  seen_residue_ids: dict[str, set[int]] = {}
  seen_native_chains: list[str] = []
  for atom_index, atom in enumerate(atoms):
    native_chain = str(atom.get("chain_id", ""))
    if native_chain not in seen_native_chains:
      seen_native_chains.append(native_chain)
    source_chain = native_to_source.get(native_chain)
    if source_chain is None:
      raise AF3InputError(f"native chain {native_chain!r} has no source mapping")
    rows = source_residue_map.get(native_chain)
    if rows is None:
      raise AF3InputError(f"native chain {native_chain!r} has no residue ledger")
    native_res_id = int(atom.get("res_id", 0))
    native_atoms_by_residue.setdefault((native_chain, native_res_id), []).append(atom)
    if native_res_id < 1 or native_res_id > len(rows):
      raise AF3InputError(
          f"native chain {native_chain!r} residue {native_res_id} is outside "
          f"the source sequence length {len(rows)}"
      )
    seen_residue_ids.setdefault(native_chain, set()).add(native_res_id)
    native_atom_name = str(atom.get("atom_name", "")).strip()
    if not native_atom_name or native_atom_name != atom_names[atom_index]:
      raise AF3InputError(f"native AF3 CIF atom {atom_index} disagrees with structure iterator")
    if atom_elements:
      iterator_element = str(atom.get("atom_element", atom.get("element", ""))).strip().upper()
      if iterator_element and atom_elements[atom_index] and iterator_element != atom_elements[atom_index]:
        raise AF3InputError(f"native AF3 CIF element {atom_index} disagrees with structure iterator")
    residue = rows[native_res_id - 1]
    if residue is None:
      # Confirmed full-sequence positions have native predictions but no source
      # occurrences. They stay in the private native file and are not restored.
      continue
    residue_source_chain = str(residue.get("source_chain_id", ""))
    if residue_source_chain != source_chain:
      raise AF3InputError(
          f"native chain {native_chain!r} residue {native_res_id} maps to "
          f"source chain {residue_source_chain!r}, expected {source_chain!r}"
      )
    source_res_id = int(residue.get("source_res_id", 0))
    source_ins_code = str(residue.get("source_ins_code", ""))
    if not source_chain or source_res_id <= 0:
      raise AF3InputError("source residue ledger has an invalid locator")
    source_key = (source_chain, source_res_id, source_ins_code)
    source_residue_rows = source_rows_by_residue.get(source_key, ())
    if not source_residue_rows:
      raise AF3InputError(f"source residue ledger has no rows for {source_key!r}")
    modality = chain_entries.get(native_chain, "")
    if modality not in {"protein", "rna", "dna", "ligand"}:
      modality_values = {str(row.get("mol_type", "")) for row in source_residue_rows}
      if len(modality_values) != 1 or next(iter(modality_values)) not in {
          "protein", "rna", "dna", "ligand"
      }:
        raise AF3InputError(f"native chain {native_chain!r} has no valid modality")
      modality = next(iter(modality_values))
    native_res_name = str(atom.get("res_name", ""))
    source_atom = source_atoms_by_key.get((*source_key, native_atom_name))
    if source_atom is None:
      mapped_uid = native_atom_source_uids.get((native_chain, native_atom_name))
      if mapped_uid is not None:
        candidate = source_atoms_by_uid.get(mapped_uid)
        if candidate is None:
          raise AF3InputError(
              f"native atom {native_chain!r}/{native_atom_name!r} maps to "
              f"unknown source atom UID {mapped_uid}"
          )
        if (
            str(candidate.get("chain_id", "")) != source_chain
            or int(candidate.get("res_id", 0)) != source_res_id
            or str(candidate.get("ins_code", "")) != source_ins_code
        ):
          raise AF3InputError(
              f"native atom {native_chain!r}/{native_atom_name!r} maps outside "
              "its source residue"
          )
        candidate_element = str(candidate.get("element", "")).strip().upper()
        if candidate_element and atom_elements and candidate_element != atom_elements[atom_index]:
          raise AF3InputError(
              f"native atom {native_chain!r}/{native_atom_name!r} changed source element"
          )
        source_atom = candidate
    component_names = {
        str(row.get("res_name", "")).strip()
        for row in source_residue_rows
        if str(row.get("res_name", "")).strip()
    }
    if len(component_names) != 1:
      raise AF3InputError(
          f"source residue {source_key!r} has ambiguous component names"
      )
    source_component = next(iter(component_names))
    if modality == "ligand":
      restored_res_name = source_component
      mapped_source_component = native_component_to_source.get(native_res_name)
      if mapped_source_component is not None and mapped_source_component != restored_res_name:
        raise AF3InputError(
            f"native component {native_res_name!r} disagrees with source "
            f"component {restored_res_name!r}"
        )
    else:
      # Context and fixed-modified residues keep their authoritative source
      # component.  Design residues keep the native component/inventory because
      # the requested sequence may have changed their identity or atom set.
      is_design = any(bool(row.get("is_design", False)) for row in source_residue_rows)
      restored_res_name = native_res_name if is_design else source_component
    restored_atom_name = (
        str(source_atom.get("atom_name", native_atom_name))
        if source_atom is not None else native_atom_name
    )
    row = {
        **{key: residue[key] for key in ("label_asym_id", "label_seq_id", "auth_asym_id", "auth_seq_id", "auth_ins_code", "source_format")},
        "native_atom_index": atom_index,
        "source_observed_index": int(residue["source_observed_index"]),
        "source_token_uid": int(residue["source_token_uid"]),
        "source_atom_row": None if source_atom is None else int(source_atom["row"]),
        "native_chain_id": native_chain,
        "native_res_id": native_res_id,
        "native_res_name": native_res_name,
        "native_atom_name": native_atom_name,
        "chain_id": source_chain,
        "res_id": source_res_id,
        "ins_code": source_ins_code,
        "res_name": restored_res_name,
        "atom_name": restored_atom_name,
        "element": _element(atom.get("atom_element", atom.get("element", "")), native_atom_name),
        "mol_type": modality,
        "source_atom_uid": None if source_atom is None else source_atom.get("atom_uid"),
        "is_design": bool(False if source_atom is None else source_atom.get("is_design", False)),
    }
    native_key = (native_chain, native_res_id, native_res_name, native_atom_name)
    if native_key in native_lookup:
      raise AF3InputError(f"duplicate native atom identity {native_key!r}")
    native_lookup[native_key] = row
    restored_rows.append(row)

  # Validate each native residue against the source identity ledger before
  # rewriting any CIF labels.  Context and ligand inventories are fail-closed;
  # only explicit design polymer residues may change inventory, and their
  # native additions must still belong to the requested canonical parent.
  for (native_chain, native_res_id), native_residue_atoms in native_atoms_by_residue.items():
    rows = source_residue_map.get(native_chain)
    if rows is None or native_res_id < 1 or native_res_id > len(rows):
      raise AF3InputError(
          f"native residue {(native_chain, native_res_id)!r} has no source mapping"
      )
    residue = rows[native_res_id - 1]
    if residue is None:
      entry = chain_entry_data[native_chain]
      modality = str(entry["modality"])
      parent = _parent_name(native_chain, native_res_id, "", modality, prefer_sequence=True)
      modification = dict((int(position), str(component)) for component, position in entry.get("modifications", ())).get(native_res_id)
      expected_component = modification or parent
      names = [str(atom.get("atom_name", "")).strip() for atom in native_residue_atoms]
      components = {str(atom.get("res_name", "")).strip() for atom in native_residue_atoms}
      if components != {expected_component} or len(set(names)) != len(names):
        raise AF3InputError(f"native unobserved residue {(native_chain, native_res_id)!r} changed its requested component/inventory")
      if modification is not None:
        if ccd is None or not hasattr(ccd, "get"):
          raise AF3InputError("native unobserved modification requires the resident authoritative CCD")
        graph = _component_graph(ccd.get(modification), modification)
        if graph is None:
          raise AF3InputError("native unobserved modification has no authoritative CCD graph")
        expected = {str(atom["name"]): str(atom["element"]).upper() for atom in graph["atoms"]}
      else:
        expected = {name: _canonical_element(name) for name in canonical_atoms.get(modality, {}).get(parent or "", ())}
      if modality == "protein" and native_res_id == len(rows):
        expected["OXT"] = "O"
      if modality in {"rna", "dna"} and native_res_id == 1:
        expected["OP3"] = "O"
      for atom in native_residue_atoms:
        name = str(atom["atom_name"]).strip()
        if expected.get(name) != _element(atom.get("atom_element", atom.get("element", "")), name):
          raise AF3InputError(f"native unobserved residue {(native_chain, native_res_id)!r} has unknown or element-changed atom {name!r}")
      continue
    source_chain = str(residue.get("source_chain_id", ""))
    source_res_id = int(residue.get("source_res_id", 0))
    source_ins_code = str(residue.get("source_ins_code", ""))
    source_key = (source_chain, source_res_id, source_ins_code)
    source_residue_rows = source_rows_by_residue.get(source_key, ())
    if not source_residue_rows:
      raise AF3InputError(f"source residue ledger has no rows for {source_key!r}")
    modality = chain_entries.get(native_chain, "")
    if modality not in {"protein", "rna", "dna", "ligand"}:
      modality_values = {str(row.get("mol_type", "")) for row in source_residue_rows}
      if len(modality_values) != 1:
        raise AF3InputError(f"source residue {source_key!r} has mixed modalities")
      modality = next(iter(modality_values))
    design_values = {bool(row.get("is_design", False)) for row in source_residue_rows}
    if len(design_values) != 1:
      raise AF3InputError(f"source residue {source_key!r} mixes design and context atoms")
    is_design = next(iter(design_values))

    source_heavy: dict[str, str] = {}
    source_component_names = {
        str(row.get("res_name", "")).strip()
        for row in source_residue_rows
        if str(row.get("res_name", "")).strip()
    }
    if len(source_component_names) != 1:
      raise AF3InputError(f"source residue {source_key!r} has ambiguous component names")
    source_component = next(iter(source_component_names))
    for row in source_residue_rows:
      atom_name = str(row.get("atom_name", "")).strip()
      element = _element(row.get("element", ""), atom_name)
      if not atom_name:
        raise AF3InputError(f"source residue {source_key!r} has an unnamed atom")
      if element in {"H", "D", "T"}:
        continue
      if atom_name in source_heavy:
        raise AF3InputError(
            f"source residue {source_key!r} has duplicate atom {atom_name!r}"
        )
      source_heavy[atom_name] = element

    native_heavy: dict[str, str] = {}
    native_components: set[str] = set()
    for atom in native_residue_atoms:
      atom_name = _source_identity_atom_name(
          native_chain,
          str(atom.get("atom_name", "")).strip(),
      )
      element = _element(atom.get("atom_element", atom.get("element", "")), atom_name)
      native_components.add(str(atom.get("res_name", "")).strip())
      if element in {"H", "D", "T"}:
        continue
      if atom_name in native_heavy:
        raise AF3InputError(
            f"native residue {(native_chain, native_res_id)!r} has duplicate atom {atom_name!r}"
        )
      native_heavy[atom_name] = element
    if len(native_components) != 1 or not next(iter(native_components), ""):
      raise AF3InputError(
          f"native residue {(native_chain, native_res_id)!r} has ambiguous component names"
      )
    native_component = next(iter(native_components))

    # Explicit design polymer residues are allowed to change component and
    # inventory for the requested variant.  Every ligand and every all-context
    # residue remains source-identity strict; standard parent completion is
    # handled only in that strict branch.
    strict_identity = modality == "ligand" or not is_design
    if strict_identity:
      parent = _parent_name(
          native_chain, native_res_id, source_component, modality, prefer_sequence=False
      )
      allowed_components = {source_component}
      modification = dict((int(position), str(component)) for component, position in chain_entry_data[native_chain].get("modifications", ())).get(native_res_id)
      mapped_component = native_component_map.get(source_component)
      if mapped_component is not None:
        allowed_components.add(str(mapped_component))
      if parent is not None and modification is None:
        allowed_components.add(parent)
      if native_component not in allowed_components:
        raise AF3InputError(
            f"native residue {(native_chain, native_res_id)!r} component "
            f"{native_component!r} is not an allowed source/parent component"
        )

      shared_names = set(source_heavy) & set(native_heavy)
      element_drift = sorted(
          name for name in shared_names if source_heavy[name] != native_heavy[name]
      )
      if element_drift:
        raise AF3InputError(
            f"native residue {(native_chain, native_res_id)!r} changed source "
            f"element for atoms {element_drift!r}"
        )

      terminal = modality == "protein" and native_res_id == len(rows)
      missing = set(source_heavy) - set(native_heavy)
      optional_missing = {"OXT"} if terminal else set()
      missing_required = missing - optional_missing
      if missing_required:
        raise AF3InputError(
            f"native residue {(native_chain, native_res_id)!r} is missing source "
            f"atoms {sorted(missing_required)!r}"
        )

      expected_elements = {}
      if modification is not None and ccd is not None and hasattr(ccd, "get"):
        graph = _component_graph(ccd.get(modification), modification)
        if graph is not None:
          expected_elements = {str(atom["name"]): str(atom["element"]).upper() for atom in graph["atoms"]}
      expected = set(expected_elements) or set(canonical_atoms.get(modality, {}).get(parent or "", ()))
      if terminal and modality == "protein":
        expected.add("OXT")
      # Native AF3 completes the free 5-prime phosphate oxygen. It is
      # valid only at the first RNA/DNA residue and must remain oxygen.
      if modality in {"rna", "dna"} and native_res_id == 1:
        expected.add("OP3")
      extra = set(native_heavy) - set(source_heavy)
      unknown_extra = extra - expected
      if unknown_extra:
        raise AF3InputError(
            f"native residue {(native_chain, native_res_id)!r} has unknown extra "
            f"atoms {sorted(unknown_extra)!r}"
        )
      canonical_drift = sorted(
          name for name in extra & expected
          if expected_elements.get(name, _canonical_element(name)) != native_heavy[name]
      )
      if canonical_drift:
        raise AF3InputError(
            f"native residue {(native_chain, native_res_id)!r} has invalid canonical "
            f"element for atoms {canonical_drift!r}"
        )

  expected_native_chains = set(source_residue_map)
  if set(seen_native_chains) != expected_native_chains:
    raise AF3InputError(
        f"native/source chain count mismatch: native={seen_native_chains!r}, "
        f"expected={sorted(expected_native_chains)!r}"
    )
  for native_chain, rows in source_residue_map.items():
    expected_ids = set(range(1, len(rows) + 1))
    if seen_residue_ids.get(native_chain, set()) != expected_ids:
      raise AF3InputError(
          f"native/source residue count mismatch for chain {native_chain!r}"
      )

  from .af3_restore import rebuild_source_categories
  rebuild_source_categories(block, payload, restored_rows, native_lookup)

  _add_authoritative_component_graphs(
      block,
      payload,
      restored_rows,
      ccd=ccd,
  )
  return cif_file.serialize(), _flatten_cif_file(cif_file), restored_rows


def _native_prediction(
    inference_result: Any,
    *,
    worker: "AF3Worker",
    payload: Mapping[str, Any],
    protocol: str,
    sample: int,
    directory: Path,
    native_frame_mask: Any = None,
) -> dict[str, Any]:
  """Convert one native SDK result to the validation/reward prediction ABI."""
  structure = inference_result.predicted_structure
  directory.mkdir(parents=True, exist_ok=True)
  stem = f"{payload.get('name', 'af3')}_{protocol}_sample{int(sample)}"
  cif_path = directory / f"{stem}.cif"
  native_cif = structure.to_mmcif()
  native_cif_path = directory / f"{stem}.native.cif"
  native_cif_path.write_text(native_cif, encoding="utf-8")
  cached_ccd = _cached_ccd_for_payload(worker, payload)
  restored_cif, restored_flat_cif, restored_rows = _restore_authoritative_cif(
      structure,
      payload,
      native_cif,
      ccd=cached_ccd,
  )
  cif_string = restored_cif
  mmcif_metadata = getattr(worker, "_sdk", {}).get("mmcif_metadata")
  if mmcif_metadata is not None:
    # The SDK metadata helper requires its immutable CifDict (with
    # copy_and_update), whereas CIF restoration uses biotite for authoritative
    # source labels.  Reparse the restored text into the SDK representation
    # before enrichment; the flat map remains the CPU-test fallback.
    try:
      from alphafold3.structure import mmcif as sdk_mmcif
      metadata_input = sdk_mmcif.from_string(restored_cif)
    except (ImportError, ModuleNotFoundError):
      metadata_input = restored_flat_cif
    try:
      enriched = mmcif_metadata.add_metadata_to_mmcif(
          old_cif=metadata_input,
          version=f"native-{worker.method}",
          model_id=worker.method.encode("ascii"),
      )
      cif_string = enriched.to_string()
    except ModuleNotFoundError:
      # CPU contract tests may inject a metadata double without the optional
      # native C++ extension.  Native workers have the module available.
      cif_string = restored_cif

  # The SDK metadata writer currently drops ligand CCD loops.  Reapply the
  # authoritative source graph after enrichment, while retaining all metadata
  # categories and native atom coordinates.
  try:
    from biotite.structure.io import pdbx
    enriched_file = pdbx.CIFFile.deserialize(cif_string)
    enriched_block = next(iter(enriched_file.values()))
    if "atom_site" in enriched_block:
      authoritative_block = next(iter(pdbx.CIFFile.deserialize(restored_cif).values()))
      for name in ("atom_site", "struct_asym", "entity", "entity_poly", "entity_poly_seq",
                   "pdbx_poly_seq_scheme", "pdbx_nonpoly_scheme", "struct_conn", "struct_conn_type"):
        if name in authoritative_block:
          enriched_block[name] = authoritative_block[name]
        elif name in enriched_block:
          del enriched_block[name]
      _add_authoritative_component_graphs(
          enriched_block,
          payload,
          restored_rows,
          ccd=cached_ccd,
      )
      cif_string = enriched_file.serialize()
  except AF3InputError:
    raise
  except (KeyError, TypeError, ValueError, AttributeError) as exc:
    raise AF3OutputError(
        "enriched AF3 CIF cannot retain authoritative ligand chemistry"
    ) from exc
  cif_path.write_text(cif_string, encoding="utf-8")

  native_metadata = dict(getattr(inference_result, "metadata", {}) or {})
  metadata = native_metadata
  numerical = dict(getattr(inference_result, "numerical_data", {}) or {})
  native_atoms = list(structure.iter_atoms())
  atom_plddt = _confidence_array(
      getattr(structure, "atom_b_factor", ()), "pLDDT", (len(native_atoms),), upper_bound=100.0,
  )
  mean_plddt = float(np.mean(atom_plddt)) if atom_plddt.size else 0.0
  token_chain_ids = [str(value) for value in metadata.get("token_chain_ids", ())]
  token_res_ids = [int(value) for value in metadata.get("token_res_ids", ())]
  full_pae = numerical.get("full_pae", numerical.get("pae"))
  full_pde = numerical.get("full_pde", numerical.get("pde"))
  if full_pae is None or full_pde is None:
    raise AF3OutputError("native AF3 confidence must expose separate full PAE and PDE")
  for name, value in (("full PAE", full_pae), ("full PDE", full_pde)):
    _confidence_array(value, name, (len(token_chain_ids), len(token_chain_ids)))
  from .af3_restore import project_source_confidence
  projection = project_source_confidence(payload, token_chain_ids, token_res_ids, full_pae, full_pde)
  native_to_source = {str(native): str(source) for native, source in payload["native_to_source_chain"].items()}
  native_chain_ids = []
  for field in ("chain_ids", "native_chain_ids"):
    candidate = _ordered_chain_ids(metadata.get(field))
    if candidate:
      native_chain_ids = candidate
      break
  if not native_chain_ids:
    native_chain_ids = _unique_chain_ids(token_chain_ids)
  if len(native_chain_ids) != len(set(native_chain_ids)):
    raise AF3InputError("native confidence contains duplicate chain IDs")
  if set(native_chain_ids) != set(native_to_source):
    raise AF3InputError("native confidence chain IDs do not uniquely cover the prepared native entities")
  no_frames, ion_chains, native_frames = _native_confidence_masks(
      payload, native_atoms, token_chain_ids, native_chain_ids, native_frame_mask,
  )
  validated = _validate_native_metadata(native_metadata, native_chain_ids, no_frames)
  chain_ptm, chain_iptm = validated["chain_ptm"], validated["chain_iptm"]
  pair_iptm = validated["chain_pair_iptm"]
  metadata = _jsonable(native_metadata)
  source_chain_ids = [native_to_source[chain] for chain in native_chain_ids]
  source_confidence_chain_map = {chain: chain for chain in payload["source_to_native_chain"]}
  chain_plddt_values, native_chain_plddt_values = {}, {}
  product_ledger = []
  restored_by_native_index = {int(row["native_atom_index"]): row for row in restored_rows}
  native_product_ledger = []
  for index, atom in enumerate(native_atoms):
    chain = str(atom["chain_id"])
    native_chain_plddt_values.setdefault(chain, []).append(float(atom_plddt[index]))
    native_record = {"row": index, "chain_id": chain, "res_id": int(atom["res_id"]),
                     "res_name": str(atom["res_name"]), "atom_name": str(atom["atom_name"]),
                     "element": str(atom.get("atom_element", atom.get("element", ""))),
                     "source_atom_uid": None, "source_residue_mapped": index in restored_by_native_index}
    source = restored_by_native_index.get(index)
    if source is not None:
      native_record.update(source_chain_id=source["chain_id"], source_res_id=source["res_id"],
                           source_ins_code=source["ins_code"], source_atom_uid=source["source_atom_uid"])
    native_product_ledger.append(native_record)
  for index, row in enumerate(restored_rows):
    chain_plddt_values.setdefault(row["chain_id"], []).append(float(atom_plddt[row["native_atom_index"]]))
    product_ledger.append({"row": index, **{key: value for key, value in row.items() if key != "native_atom_index"}})
  source_atom_counts = {chain: len(values) for chain, values in chain_plddt_values.items()}
  native_atom_counts = {chain: len(values) for chain, values in native_chain_plddt_values.items()}
  source_per_chain_plddt = {chain: float(np.mean(values)) for chain, values in chain_plddt_values.items()}
  native_per_chain_plddt = {chain: float(np.mean(values)) for chain, values in native_chain_plddt_values.items()}
  per_chain_native_token_count = {chain: token_chain_ids.count(chain) for chain in native_chain_ids}
  native_source_token_count = {chain: sum(native == chain and mapped for native, mapped in zip(token_chain_ids, projection["native_token_source_mask"], strict=True)) for chain in native_chain_ids}
  per_chain_token_count = {chain: projection["token_chain_ids"].count(chain) for chain in payload["source_to_native_chain"]}
  ptm, iptm = validated["ptm"], validated["iptm"]
  prediction_key = f"{protocol}_sample{int(sample)}"
  metadata.update(key=prediction_key, protocol=protocol, sample=int(sample))
  confidence = {
      "mean_plddt": mean_plddt, "native_plddt": mean_plddt, "ptm": ptm, "iptm": iptm,
      "chain_ids": native_chain_ids, "native_chain_ids": native_chain_ids, "source_chain_ids": source_chain_ids,
      "chain_ptm": chain_ptm, "chain_iptm": chain_iptm, "chain_pair_iptm": pair_iptm, "pair_iptm": pair_iptm,
      "token_chain_label_map": native_to_source,
      "confidence_chain_map": source_confidence_chain_map,
      "native_to_source_chain": native_to_source, "source_to_native_chain": dict(payload["source_to_native_chain"]),
      "per_chain_plddt": source_per_chain_plddt, "native_per_chain_plddt": native_per_chain_plddt,
      "per_chain_atom_count": source_atom_counts, "native_per_chain_atom_count": native_atom_counts,
      "per_chain_token_count": per_chain_token_count, "native_per_chain_token_count": per_chain_native_token_count,
      "native_per_chain_source_token_count": native_source_token_count,
      "native_frame_mask": native_frames,
      "native_ion_chain_mask": ion_chains,
      "native_undefined_confidence_masks": validated["undefined_masks"],
      **projection,
  }
  confidence.update(pae_matrix=confidence["full_pae"], pde_matrix=confidence["full_pde"])
  raw_confidence_path = directory / f"{stem}.confidence.json"
  raw_confidence = {key: value for key, value in confidence.items()
                    if key not in {"full_pae", "full_pde", "pae_matrix", "pde_matrix"}}
  raw_confidence["plddt"] = atom_plddt.tolist()
  raw_confidence_path.write_text(json.dumps(_jsonable(raw_confidence), sort_keys=True, allow_nan=False), encoding="utf-8")
  confidence["plddt"] = mean_plddt
  return {
      **confidence,
      "model": worker.method, "protocol": protocol, "sample": int(sample),
      "key": prediction_key, "sample_id": prediction_key,
      "prediction_id": f"{worker.method}_{prediction_key}", "status": "complete",
      "cif_path": str(cif_path), "structure_path": str(cif_path), "pdb_path": str(cif_path),
      "native_cif_path": str(native_cif_path),
      "plddt": mean_plddt / 100.0, "plddt_fraction": mean_plddt / 100.0,
      "native_iptm": iptm, "pae": projection["full_pae"], "pde": projection["full_pde"],
      "confidence": confidence, "raw_confidence_path": str(raw_confidence_path),
      "_confidence_chain_map": source_confidence_chain_map, "metadata": metadata,
      "source_atom_ledger": list(payload["source_atom_ledger"]),
      "product_atom_ledger": product_ledger, "restored_product_atom_ledger": product_ledger,
      "native_product_atom_ledger": native_product_ledger,
      "source_residue_map": {chain: list(rows) for chain, rows in payload["source_residue_map"].items()},
      "native_component_map": dict(payload.get("native_component_map", {})),
      "source_component_map": {str(native): str(source) for source, native in dict(payload.get("native_component_map", {})).items()},
      "_native_confidence": {"metadata": validated["native_arrays"],
                             "frames_mask": native_frames,
                             "ion_chain_mask": ion_chains,
                             "undefined_masks": validated["undefined_masks"],
                             "nan_masks": validated["nan_masks"]},
  }


class AF3Worker:
  """Persistent service around one native AF3 ``ModelRunner`` and CCD cache."""

  def __init__(
      self,
      repo_root: str | os.PathLike[str],
      config: Any,
      method: str = "af3-la",
      device: str | int | None = None,
      *,
      runner: Any | None = None,
      _child: bool = False,
  ) -> None:
    self.repo_root = Path(repo_root).expanduser().resolve()
    self.config = config
    self.method = _normalise_method(method)
    self.device = device if device is not None else _get(config, "device", None)
    self._runner = runner
    self._child_mode = bool(_child)
    self._sdk_loaded = False
    self._sdk: dict[str, Any] = {}
    self._ccd_cache: dict[tuple[str, str], Any] = {}
    self._service: _PersistentAF3Service | None = None
    self._fatal: str | None = None
    self._fatal_error: AF3WorkerError | None = None
    self._log_dir: Path | None = None
    self._log_dir_explicit = False
    self._child_log_dir: Path | None = None
    self._closed = False
    self._lock = threading.RLock()

  @property
  def model_runner(self) -> Any | None:
    return self._runner if self._child_mode or self._service is None else None

  @property
  def ccd_cache_size(self) -> int:
    return len(self._ccd_cache)

  def set_log_dir(self, log_dir: str | os.PathLike[str] | None) -> None:
    """Set the child native log directory without reloading the model."""
    self._log_dir_explicit = log_dir is not None
    self._log_dir = None if log_dir is None else Path(log_dir).expanduser().resolve()
    if self._child_mode:
      self._child_log_dir = self._log_dir
      return
    if self._service is not None:
      self._service.set_log_dir(self._log_dir)

  def _load_runtime(self) -> None:
    if self._sdk_loaded:
      return
    _configure_ccd_environment(self.repo_root, self.config)
    source = _sdk_source(self.repo_root, self.method)
    if str(source) not in sys.path:
      sys.path.insert(0, str(source))
    sdk_root = source.parent
    if (sdk_root / "run_alphafold.py").is_file() and str(sdk_root) not in sys.path:
      sys.path.insert(0, str(sdk_root))
    try:
      import jax
      from alphafold3.constants import chemical_components
      from alphafold3.model import feat_batch, features, mmcif_metadata
      from alphafold3.model.pipeline import pipeline
    except Exception as exc:  # pragma: no cover - depends on SDK env
      raise AF3WorkerError("failed to import the AlphaFold 3 SDK") from exc
    # ``run_alphafold.py`` is a sibling of the SDK's ``src`` tree in the
    # upstream checkout.  Some distributions copy it into ``source`` instead;
    # support both layouts without importing it at module import time.
    run_alphafold = None
    try:
      import run_alphafold as _run_alphafold  # type: ignore
      run_alphafold = _run_alphafold
    except Exception:
      try:
        from alphafold3 import run_alphafold as _run_alphafold  # type: ignore
        run_alphafold = _run_alphafold
      except Exception as exc:  # pragma: no cover - depends on SDK layout
        raise AF3WorkerError("AlphaFold 3 run_alphafold module is missing") from exc
    # Parse only after the SDK has registered its flags, while still hiding the
    # design CLI's argparse/Hydra arguments from Abseil.
    _prepare_absl_flags()
    self._sdk = {
        "jax": jax,
        "chemical_components": chemical_components,
        "mmcif_metadata": mmcif_metadata,
        "feat_batch": feat_batch,
        "features": features,
        "pipeline": pipeline,
        "run_alphafold": run_alphafold,
    }
    if self._runner is None:
      checkpoint = self._checkpoint_dir()
      config = run_alphafold.make_model_config(
          flash_attention_implementation=str(
              _get(self.config, "attention_backend", _get(self.config, "flash_attention_implementation", "triton"))
          ),
          num_diffusion_samples=int(_get(self.config, "samples", 1)),
          num_recycles=int(_get(self.config, "recycles", 10)),
      )
      requested_text = None if self.device is None else str(self.device)
      requested_gpu = bool(
          requested_text is not None
          and (requested_text.startswith("cuda:") or requested_text.startswith("gpu:"))
      )
      try:
        gpu_devices = list(jax.local_devices(backend="gpu"))
      except Exception:
        gpu_devices = []
      if requested_gpu and not gpu_devices:
        raise AF3WorkerError(
            f"requested AF3 device {self.device!r} is unavailable; no visible GPU devices"
        )
      devices = gpu_devices if gpu_devices else list(jax.devices())
      if not devices:
        raise AF3WorkerError("AlphaFold 3 found no JAX devices")
      selected_device = None
      if self.device is not None:
        requested = requested_text
        if requested.startswith("cuda:") or requested.startswith("gpu:"):
          requested = requested.split(":", 1)[1]
        try:
          ordinal = int(requested)
        except ValueError:
          ordinal = -1
        if 0 <= ordinal < len(devices):
          selected_device = devices[ordinal]
        else:
          selected_device = next(
              (item for item in devices if str(item) == str(self.device)), None
          )
        if selected_device is None:
          raise AF3WorkerError(
              f"requested AF3 device {self.device!r} is unavailable; "
              f"visible devices={devices!r}"
          )
      else:
        selected_device = devices[0]
      self._runner = run_alphafold.ModelRunner(
          config=config,
          device=selected_device,
          model_dir=checkpoint,
      )
    self._sdk_loaded = True

  def _checkpoint_dir(self) -> Path:
    configured_value = _get(self.config, "checkpoint", None)
    configured = _as_path(configured_value)
    if configured is not None and configured_value is not None:
      raw = Path(str(configured_value)).expanduser()
      if not raw.is_absolute():
        configured = (self.repo_root / raw).resolve()
    if configured is None:
      from partidesign.paths import af3_asset
      configured = (
          af3_asset(self.method, self.repo_root) / "checkpoints"
          / ("af3_leaving_atom.bin.zst" if self.method == "af3-la" else "af3.bin.zst")
      )
    if configured.is_file():
      return configured.parent
    return configured

  def _ccd(self, user_ccd: str | None) -> Any:
    self._load_runtime()
    configured = _get(
        self.config,
        "externalccd_pickle_path",
        _get(self.config, "ccd_pickle_path", None),
    )
    ccd_path = _as_path(configured)
    if ccd_path is not None and configured is not None:
      raw = Path(str(configured)).expanduser()
      if not raw.is_absolute():
        ccd_path = (self.repo_root / raw).resolve()
    if ccd_path is not None and ccd_path.is_dir():
      for candidate in ("ccd.pkl", "ccd.pickle", "chemical_components.pkl"):
        if (ccd_path / candidate).is_file():
          ccd_path = ccd_path / candidate
          break
    cache_prefix = "__base__" if ccd_path is None else str(ccd_path)
    key = (cache_prefix, "" if user_ccd is None else str(user_ccd))
    if key not in self._ccd_cache:
      ccd_cls = self._sdk["chemical_components"].Ccd
      kwargs = {"user_ccd": user_ccd}
      if ccd_path is not None:
        kwargs["ccd_pickle_path"] = ccd_path
      self._ccd_cache[key] = ccd_cls(**kwargs)
    return self._ccd_cache[key]

  def _buckets(self) -> tuple[int, ...] | None:
    value = _get(self.config, "buckets", NATIVE_BUCKETS)
    if value is None or value == "native":
      return NATIVE_BUCKETS
    if isinstance(value, str):
      if value.strip().lower() == "native":
        return NATIVE_BUCKETS
      value = [part.strip() for part in value.split(",") if part.strip()]
    try:
      result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
      raise AF3WorkerError("AF3 buckets must be a sequence of integers") from exc
    if not result or tuple(sorted(set(result))) != result:
      raise AF3WorkerError("AF3 buckets must be strictly increasing")
    return result

  def _cyclic_features(self, payload: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
    token_count = int(np.asarray(batch["seq_length"]).reshape(-1)[0])
    cyclic_chains = {str(value) for value in payload.get("cyclic_chains", ())}
    if not cyclic_chains:
      return {}
    # TokenFeatures assigns asym IDs in input chain order, starting at 1.  The
    # native pipeline is configured with one token per modified residue, so
    # residue_index remains a stable chain-local position here.
    asym = np.asarray(batch["asym_id"], dtype=np.int32)
    residue_index = np.asarray(batch["residue_index"], dtype=np.int32)
    offset = np.full(asym.shape, -1, dtype=np.int32)
    length = np.zeros(asym.shape, dtype=np.int32)
    chain_entries = list(payload.get("chains", ()))
    for chain_index, entry in enumerate(chain_entries, start=1):
      native_id = str(entry.get("id", ""))
      if native_id not in cyclic_chains:
        continue
      chain_mask = (asym == chain_index) & (np.arange(len(asym)) < token_count)
      chain_length = int(payload.get("cyclic_lengths", {}).get(native_id, int(chain_mask.sum())))
      if chain_length <= 0:
        continue
      offset[chain_mask] = residue_index[chain_mask] - 1
      length[chain_mask] = chain_length
    return {"cyclic_offset": offset, "cyclic_length": length}

  def _featurise(self, native_input: AF3Input, fold_input: Any) -> Mapping[str, Any]:
    self._load_runtime()
    pipeline_cls = self._sdk["pipeline"].WholePdbPipeline
    config = pipeline_cls.Config(
        buckets=list(self._buckets() or ()),
        max_templates=int(getattr(native_input, "msa_context", {}).get("max_templates", 0)),
        flatten_non_standard_residues=False,
        resolve_msa_overlaps=bool(getattr(native_input, "msa_context", {}).get("resolve_msa_overlaps", False)),
        fix_standalone_glycans=bool(_get(self.config, "fix_standalone_glycans", self.method == "af3-la")),
    )
    # Inputs already contain cached alignments/templates or explicit empty
    # fields. Model featurization never performs database searches here.
    pipeline = pipeline_cls(config=config)
    try:
      ccd = self._ccd(native_input.user_ccd)
      batch = pipeline.process_item(
          fold_input=fold_input,
          ccd=ccd,
          random_state=np.random.RandomState(int(native_input.rng_seeds[0])),
          random_seed=int(native_input.rng_seeds[0]),
      )
    except AF3InputError:
      raise
    except (TypeError, ValueError, KeyError, IndexError) as exc:
      raise AF3InputError(
          f"AF3 native CPU featurization failed: {type(exc).__name__}: {exc}"
      ) from exc
    cyclic = self._cyclic_features(native_input.payload(), batch)
    if cyclic:
      batch = dict(batch)
      batch.update(cyclic)
    return batch

  def _predict_direct(
      self,
      payload: AF3Input | Mapping[str, Any],
      workdir: str | os.PathLike[str],
  ) -> list[dict[str, Any]]:
    """Execute one request inside the resident child process."""
    with self._lock:
      if self._closed:
        raise AF3WorkerError("AF3 worker is closed")
      self._load_runtime()
      if isinstance(payload, AF3Input):
        native_input = payload
        payload_map = native_input.payload()
      else:
        try:
          payload_map = dict(payload)
          native_input = AF3Input.from_payload(payload_map)
        except AF3InputError:
          raise
        except (TypeError, ValueError, KeyError, IndexError) as exc:
          raise AF3InputError(
              f"AF3 native request construction failed: "
              f"{type(exc).__name__}: {exc}"
          ) from exc
      try:
        fold_input = native_input.to_folding_input()
      except AF3InputError:
        raise
      except (TypeError, ValueError, KeyError, IndexError) as exc:
        raise AF3InputError(
            f"AF3 native folding input construction failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
      batch = self._featurise(native_input, fold_input)
      try:
        token_count = int(np.asarray(batch["seq_length"]).item())
        frame_mask = np.asarray(batch["frames_mask"])
        if frame_mask.dtype.kind != "b" or frame_mask.ndim != 1 or len(frame_mask) < token_count:
          raise ValueError("invalid frames_mask")
        native_frame_mask = frame_mask[:token_count].copy()
      except (KeyError, TypeError, ValueError) as exc:
        raise AF3WorkerError("AF3 CPU features have no valid native frame mask") from exc
      jax = self._sdk["jax"]
      result = self._runner.run_inference(
          batch,
          jax.random.PRNGKey(int(native_input.rng_seeds[0])),
      )
      inference_results = list(self._runner.extract_inference_results(
          batch=batch,
          result=result,
          target_name=native_input.name,
      ))
      output_dir = Path(workdir).expanduser().resolve()
      try:
        output_dir.mkdir(parents=True, exist_ok=True)
        return [
            _native_prediction(
                inference_result,
                worker=self,
                payload=payload_map,
                protocol=native_input.protocol,
                sample=index,
                directory=output_dir,
                native_frame_mask=native_frame_mask,
            )
            for index, inference_result in enumerate(inference_results)
        ]
      except (AF3InputError, AF3WorkerError):
        raise
      except (OSError, TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:
        raise AF3OutputError(f"AF3 prediction output failed: {type(exc).__name__}: {exc}") from exc

  def _ensure_service(self) -> _PersistentAF3Service:
    if self._fatal is not None:
      raise self._fatal_error or AF3WorkerError(self._fatal)
    if self._service is not None and not self._service.usable:
      failure = self._service.failure()
      self._discard_service()
      raise failure
    if self._service is None:
      self._service = _PersistentAF3Service(
          self.repo_root,
          self.config,
          self.method,
          self.device,
          self._log_dir,
      )
    return self._service

  def _discard_service(self) -> None:
    service, self._service = self._service, None
    if service is not None:
      try:
        service.close()
      except AF3WorkerError:
        # The earliest failure is retained by the caller, not close-time IPC.
        pass

  def predict(
      self,
      payload: AF3Input | Mapping[str, Any],
      workdir: str | os.PathLike[str],
  ) -> list[dict[str, Any]]:
    """Submit one request to the resident AF3 child service.

    A caller-provided ``runner`` is an in-process test hook.  Production
    workers use the persistent child so the parent can enforce an IPC deadline.
    A broken child gets one replacement/retry; healthy children stay resident.
    """
    if self._closed:
      raise AF3WorkerError("AF3 worker is closed")
    if self._child_mode or self._runner is not None:
      return self._predict_direct(payload, workdir)
    payload_map = payload.payload() if isinstance(payload, AF3Input) else dict(payload)
    with self._lock:
      first_failure: AF3WorkerError | None = None
      for attempt in range(2):
        try:
          return self._ensure_service().submit(payload_map, workdir)
        except AF3WorkerError as exc:
          if not exc.fatal:
            raise
          if self._fatal_error is not None:
            raise self._fatal_error
          if first_failure is None:
            first_failure = exc
          self._discard_service()
          if attempt:
            failure = AF3WorkerError(
                f"persistent AF3 recovery failed; first failure: {first_failure}; retry: {exc}"
            )
            self._fatal = str(failure)
            self._fatal_error = failure
            raise failure from first_failure
      raise AssertionError("AF3 retry accounting failed")

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    if self._service is not None:
      self._service.close()
      self._service = None
    # The child SDK runner has no unload API.  Drop references only after the
    # child service has exited; no request recreates a closed worker.
    self._runner = None
    self._ccd_cache.clear()
    self._sdk.clear()
    self._sdk_loaded = False

  def __enter__(self) -> "AF3Worker":
    return self

  def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
    self.close()


def _main() -> None:
  parser = argparse.ArgumentParser(description="Persistent native AF3 service")
  parser.add_argument("--af3-service-worker", action="store_true")
  parser.add_argument("--repo-root", required=True)
  parser.add_argument("--method", default="af3-la")
  parser.add_argument("--config-json", required=True)
  parser.add_argument("--device")
  parser.add_argument("--log-dir")
  args = parser.parse_args()
  if not args.af3_service_worker:
    parser.error("--af3-service-worker is required")
  _service_child_main(args)


if __name__ == "__main__":
  _main()


__all__ = [
    "AF3Worker",
    "AF3WorkerError",
    "NATIVE_BUCKETS",
]
