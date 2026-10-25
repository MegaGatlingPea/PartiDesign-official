"""Isolated CPU execution for every native candidate; the driver is SDK-free."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
from .schema import Candidate

_REQUIRED = (
    "bc1_binder_energy_score", "bc1_surface_hydrophobicity", "bc1_shape_complementarity",
    "bc1_dg", "bc1_dsasa", "bc1_interface_residues", "bc1_interface_hbonds",
    "bc1_interface_unsat_hbonds",
)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, allow_nan=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict:
    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON value {value}")
    with path.open() as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("native JSON result must be an object")
    return value


def _child_command(job: Path, result: Path) -> list[str]:
    return [sys.executable, "-m", "src.postfilter.worker", "--job", str(job), "--result", str(result)]


def _failure(reason: str, *, relaxed_path: str | None = None, details: dict | None = None) -> dict:
    return {"status": "failed", "geometry_status": "complete" if relaxed_path else "failed",
            "relaxed_path": relaxed_path, "private_state_path": None, "metrics": {},
            "errors": {metric: reason for metric in _REQUIRED}, "error": reason,
            "details": details or {}}


def _validate_result(value: dict, expected_path: Path) -> dict:
    if value.get("status") not in {"complete", "failed"}:
        raise ValueError("native result has no valid status")
    if value.get("relaxed_path") not in (None, str(expected_path)):
        raise ValueError("native result names an unexpected relaxed path")
    metrics, errors = value.get("metrics"), value.get("errors")
    if not isinstance(metrics, dict) or not isinstance(errors, dict):
        raise ValueError("native result lacks metric and error objects")
    for key, number in metrics.items():
        if (not isinstance(key, str) or not key.startswith("bc1_") or isinstance(number, bool)
                or not isinstance(number, (int, float)) or not math.isfinite(number)):
            raise ValueError("native result contains an invalid metric")
    if any(not isinstance(key, str) or not isinstance(reason, str) for key, reason in errors.items()):
        raise ValueError("native result contains invalid metric errors")
    if set(metrics) & set(errors):
        raise ValueError("native result both succeeds and fails the same metric")
    if not isinstance(value.get("details", {}), dict) or not isinstance(value.get("error", ""), (str, type(None))):
        raise ValueError("native result has invalid diagnostics")
    if value["status"] == "complete" and any(metric not in metrics for metric in _REQUIRED):
        raise ValueError("completed native result lacks required metrics")
    return value


def _recover_geometry(candidate: Candidate, job: dict, checkpoint_path: Path,
                      output_path: Path, private_output_path: Path) -> tuple[str | None, str | None]:
    from .pose_mapping import (_category_values, read_source, validate_relaxed_file,
                               validate_native_state_document)
    if not checkpoint_path.is_file():
        return None, None
    checkpoint = _read_json(checkpoint_path)
    stage_path = Path(job["output_path"])
    state_path = Path(job["private_state_path"])
    geometry_read_path = stage_path if stage_path.exists() else output_path
    state_read_path = state_path if state_path.exists() else private_output_path
    if (checkpoint.get("geometry_complete") is not True
            or checkpoint.get("relaxed_path") != str(stage_path)
            or checkpoint.get("private_state_path") != str(state_path)
            or not geometry_read_path.is_file() or geometry_read_path.is_symlink()
            or not state_read_path.is_file() or state_read_path.is_symlink()):
        raise ValueError("native geometry checkpoint is malformed or private state is absent")
    source = read_source(candidate.refold_path, candidate.binder_chains, candidate.target_chains)
    if checkpoint.get("source_document") != _category_values(source.cif, exclude_coordinates=False):
        raise ValueError("source structure changed before native publication")
    relaxed = validate_relaxed_file(source, geometry_read_path)
    state = _read_json(state_read_path)
    validate_native_state_document(relaxed, state)
    expected_state = checkpoint.get("completed_native_state")
    if expected_state is not None and state != expected_state:
        raise ValueError("delivered native state differs from the completed geometry checkpoint")
    if (geometry_read_path != stage_path or state_read_path != state_path) and expected_state is None:
        raise ValueError("delivered geometry lacks a complete native state checkpoint")
    private_output_path.parent.mkdir(parents=True, exist_ok=True)
    if state_read_path != private_output_path:
        _atomic_copy(state_read_path, private_output_path)
    if geometry_read_path != output_path:
        _atomic_copy(geometry_read_path, output_path)
    delivered = validate_relaxed_file(source, output_path)
    delivered_state = _read_json(private_output_path)
    validate_native_state_document(delivered, delivered_state)
    if delivered_state != state:
        raise ValueError("delivered private native state differs after publication")
    return str(output_path), str(private_output_path)


def execute_native(candidate: Candidate, output_path: Path, config: dict) -> dict:
    """Run a candidate in an isolated CPU child and report all ordinary failures."""
    try:
        return _execute_native(candidate, output_path, config)
    except Exception as exc:
        return _failure(f"native execution failed: {type(exc).__name__}: {exc}")


def _run_job(job: dict) -> dict:
    # Reached only by the disposable subprocess entry point.
    from .pose_mapping import (_category_values, load_pose_mapping, read_source,
        validate_relaxed_file, write_relaxed_cif, native_state_document,
        validate_native_state_document, restore_native_state)
    from . import rosetta
    candidate = Candidate.from_dict(job["candidate"])
    config = rosetta.validate_config(job["config"])
    output_path, state_path = Path(job["output_path"]), Path(job["private_state_path"])
    source = read_source(candidate.refold_path, candidate.binder_chains, candidate.target_chains)
    if source.sequences != candidate.sequences:
        raise ValueError("current source sequences differ from the selected prediction")
    source_document = _category_values(source.cif, exclude_coordinates=False)
    mode = config.get("mode", "relax")
    scoring_source = validate_relaxed_file(source, output_path) if mode == "metrics_only" else source
    state = None
    checkpoint = {"geometry_complete": True, "relaxed_path": str(output_path),
                  "private_state_path": str(state_path), "source_document": source_document}
    if mode == "metrics_only":
        state = _read_json(state_path)
        validate_native_state_document(scoring_source, state)
        checkpoint["completed_native_state"] = state
        _atomic_json(Path(job["checkpoint_path"]), checkpoint)
    pr = rosetta.initialize(config)
    pose, mapping = load_pose_mapping(scoring_source, pr, Path(job["work_dir"]))
    if mode == "metrics_only":
        restore_native_state(scoring_source, pose, mapping, state, pr)
    rosetta.configure_partner_foldtree(scoring_source, pose, mapping, pr)
    if mode == "relax":
        rosetta.relax_pose(source, pose, mapping, pr, config)
        write_relaxed_cif(source, pose, mapping, output_path)
        state = native_state_document(source, pose, mapping)
        _atomic_json(state_path, state)
        validate_native_state_document(validate_relaxed_file(source, output_path), _read_json(state_path))
    checkpoint["completed_native_state"] = state
    _atomic_json(Path(job["checkpoint_path"]), checkpoint)
    rosetta.seed_metrics(pr, config["seed"])
    result = rosetta.calculate_metrics(scoring_source, pose, mapping, pr, config)
    missing = [metric for metric in _REQUIRED if metric not in result["metrics"]]
    for metric in missing:
        result["errors"].setdefault(metric, "native metric was not computed")
    result.update({"status": "failed" if missing else "complete", "geometry_status": "complete",
        "relaxed_path": str(output_path), "private_state_path": str(state_path),
        "error": "required native metrics unavailable: " + ", ".join(missing) if missing else None})
    result["details"].update({"mode": mode, "scorefunction": config["scorefunction"],
        "seed": config["seed"], "metric_phase_seed": config["seed"],
        "threads": config["threads"], "relax": config["relax"]})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private isolated native postfilter worker")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = _run_job(_read_json(args.job))
    except Exception as exc:
        result = _failure(f"{type(exc).__name__}: {exc}")
    _atomic_json(args.result, result)
    return 0


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as target, source.open("rb") as origin:
            shutil.copyfileobj(origin, target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _job_workspace(output: Path, config: dict):
    path = config.get("job_directory")
    if path is None:
        with tempfile.TemporaryDirectory(prefix=f".{output.stem}.native-", dir=output.parent) as directory:
            yield Path(directory)
        return
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "driver.lock").open("a+") as lock:
        deadline = time.monotonic() + float(config.get("timeout", 1800))
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("another driver still owns this persistent native job")
                time.sleep(.1)
        try:
            yield directory
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


_SUPERVISOR = r'''
import json, os, pathlib, subprocess, sys, time
command = json.loads(sys.argv[1])
exit_path = pathlib.Path(sys.argv[2])
try:
    process = subprocess.Popen(command)
    code = process.wait()
    result = {'command': command, 'returncode': code, 'completed_at': time.time()}
except Exception as exc:
    result = {'command': command, 'returncode': 127, 'error': str(exc), 'completed_at': time.time()}
temporary = exit_path.with_suffix('.tmp')
with temporary.open('w') as handle:
    json.dump(result, handle, allow_nan=False)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, exit_path)
'''


def _process_identity(pid: int) -> dict | None:
    try:
        path = Path("/proc") / str(pid)
        if path.stat().st_uid != os.getuid():
            return None
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        command = (path / "cmdline").read_bytes().split(b"\0")
        birth = time.time() - time.clock_gettime(time.CLOCK_BOOTTIME) + int(fields[19]) / os.sysconf("SC_CLK_TCK")
        return {"pid": pid, "start_ticks": int(fields[19]), "started_at": birth,
                "command": [value.decode() for value in command if value]}
    except (OSError, ValueError, IndexError):
        return None


def _find_job_process(command: list[str]) -> list[dict]:
    result = []
    if not Path("/proc").is_dir():
        raise RuntimeError("persistent native recovery requires Linux process identity")
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            identity = _process_identity(int(path.name))
            if identity and identity["command"] == command:
                result.append(identity)
    return result


def _same_process(current: dict | None, saved: dict) -> bool:
    return current is not None and all(current.get(key) == saved.get(key)
                                       for key in ("pid", "start_ticks", "command"))


def _wait_job_process(identity: dict, timeout: float, exit_path: Path,
                      native_command: list[str], direct_process=None) -> tuple[int | None, str | None]:
    while True:
        if exit_path.is_file():
            value = _read_json(exit_path)
            if value.get("command") != native_command or type(value.get("returncode")) is not int:
                return None, "native supervisor exit record is malformed"
            if direct_process is not None:
                direct_process.wait()
            code = value["returncode"]
            return code, None if code == 0 else f"native child exited with code {code}"
        current = _process_identity(identity["pid"])
        if current is None:
            if exit_path.is_file():
                continue
            if direct_process is not None:
                direct_process.wait()
            return None, "native supervisor stopped without an exit record"
        if not _same_process(current, identity):
            return None, "saved native process identity changed; process was not touched"
        if time.time() >= identity["started_at"] + timeout:
            current = _process_identity(identity["pid"])
            if not _same_process(current, identity) or os.getpgid(identity["pid"]) != identity["pid"]:
                return None, "native timeout owner could not be verified; process was not touched"
            try:
                os.killpg(identity["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
            if direct_process is not None:
                direct_process.wait()
            return None, f"native child timed out after {timeout:g} seconds"
        time.sleep(.05)


def _run_or_recover_process(stage: Path, command: list[str], env: dict,
                            timeout: float, checkpoint_path: Path) -> tuple[int | None, str | None]:
    exit_path, process_path = stage / "exit.json", stage / "process.json"
    supervised = [sys.executable, "-c", _SUPERVISOR, json.dumps(command), str(exit_path)]
    direct_process = None
    identity = None
    if process_path.is_file():
        identity = _read_json(process_path)
        if identity.get("command") != supervised:
            return None, "saved native process command differs from the job"
    elif not exit_path.is_file():
        existing = _find_job_process(supervised)
        if len(existing) > 1:
            return None, "multiple native processes claim this persistent job"
        if existing:
            identity = dict(existing[0])
            _atomic_json(process_path, identity)
    if identity is None and not exit_path.is_file() and not checkpoint_path.is_file():
        started_at = time.time()
        with (stage / "native.log").open("ab") as log:
            direct_process = subprocess.Popen(supervised, env=env, stdout=log, cwd=stage,
                stderr=subprocess.STDOUT, start_new_session=True)
        identity = _process_identity(direct_process.pid)
        identity_deadline = time.monotonic() + 5
        while ((identity is None or identity["command"] != supervised)
               and direct_process.poll() is None and time.monotonic() < identity_deadline):
            time.sleep(.005)
            identity = _process_identity(direct_process.pid)
        if identity is not None and identity["command"] != supervised:
            return None, "native supervisor command could not be verified after startup"
        if identity is None:
            direct_process.wait()
            if not exit_path.is_file():
                return None, "native supervisor exited before its identity was recorded"
        else:
            identity["started_at"] = started_at
            _atomic_json(process_path, identity)
    if exit_path.is_file():
        value = _read_json(exit_path)
        if value.get("command") != command or type(value.get("returncode")) is not int:
            return None, "native supervisor exit record is malformed"
        if direct_process is not None:
            direct_process.wait()
        code = value["returncode"]
        return code, None if code == 0 else f"native child exited with code {code}"
    if identity is None:
        return None, "completed geometry recovered without native metric output"
    return _wait_job_process(identity, timeout, exit_path, command, direct_process)


def _execute_native(candidate: Candidate, output_path: Path, config: dict) -> dict:
    output_path = Path(output_path).expanduser().resolve()
    input_path = Path(candidate.refold_path).expanduser().resolve()
    if output_path == input_path:
        return _failure("relaxed output cannot overwrite the source structure")
    if output_path.suffix.lower() != ".cif":
        return _failure("native relaxed output must use the .cif suffix")
    timeout, threads = config.get("timeout", 1800), config.get("threads", 1)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        return _failure("native timeout must be a positive finite number")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        return _failure("native threads must be a positive integer")
    private_output_path = Path(config.get("private_state_path", output_path.with_suffix(".native.json"))).expanduser().resolve()
    if private_output_path in {input_path, output_path}:
        return _failure("private native state cannot overwrite a source or public structure")
    mode = config.get("mode", "relax")
    if mode not in {"relax", "metrics_only"}:
        return _failure("native mode must be relax or metrics_only")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _job_workspace(output_path, config) as stage:
        job_path, result_path = stage / "job.json", stage / "result.json"
        checkpoint_path = stage / "geometry.json"
        geometry_path = output_path if mode == "metrics_only" else stage / "relaxed.cif"
        if mode == "metrics_only" and Path(config.get("relaxed_path", output_path)).expanduser().resolve() != output_path:
            return _failure("metrics_only relaxed_path must equal output_path")
        private_stage_path = private_output_path if mode == "metrics_only" else stage / "native.json"
        job = {"candidate": candidate.to_dict(), "output_path": str(geometry_path),
            "private_state_path": str(private_stage_path), "published_output_path": str(output_path),
            "published_private_state_path": str(private_output_path), "checkpoint_path": str(checkpoint_path),
            "work_dir": str(stage), "config": dict(config)}
        job = json.loads(json.dumps(job, allow_nan=False))
        if job_path.exists():
            if _read_json(job_path) != job:
                return _failure("saved native job does not match candidate, paths or configuration")
        else:
            _atomic_json(job_path, job)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = str(threads)
        try:
            returncode, failure = _run_or_recover_process(stage, _child_command(job_path, result_path), env, timeout, checkpoint_path)
        except Exception as exc:
            returncode, failure = None, f"native process recovery failed: {type(exc).__name__}: {exc}"
        geometry, private_state = None, None
        try:
            geometry, private_state = _recover_geometry(candidate, job, checkpoint_path, output_path, private_output_path)
        except Exception as exc:
            failure = f"native geometry validation failed: {type(exc).__name__}: {exc}"
        details: dict[str, Any] = {"child_exitcode": returncode, "geometry_complete": bool(geometry)}
        if config.get("job_directory"):
            details["job_directory"] = str(stage)
        log_path = stage / "native.log"
        if log_path.is_file():
            with log_path.open("rb") as log:
                log.seek(max(0, log_path.stat().st_size - 4000))
                details["native_log_tail"] = log.read().decode(errors="replace")
        if failure is not None:
            value = _failure(failure, relaxed_path=geometry, details=details)
        else:
            try:
                value = _validate_result(_read_json(result_path), geometry_path)
            except Exception as exc:
                value = _failure(f"malformed or missing native output: {type(exc).__name__}: {exc}",
                                 relaxed_path=geometry, details=details)
        if value["status"] == "complete" and not geometry:
            return _failure("native child did not complete relaxed geometry", details=details)
        value.update({"relaxed_path": geometry, "private_state_path": private_state,
                      "geometry_status": "complete" if geometry else "failed"})
        value.setdefault("details", {}).update(details)
        return value


if __name__ == "__main__":
    raise SystemExit(main())
