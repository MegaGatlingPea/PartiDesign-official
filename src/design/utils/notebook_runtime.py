"""Detached native CLI jobs and result discovery for the interactive notebook.

Importing this module uses only the standard library. Job metadata lives in
tmux session options; the ignored cache contains only launch scripts and logs.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import csv
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time


_OWNER = "partidesign-notebook"
_OPTION = "@partidesign_notebook"
_COUNTER = itertools.count(1)
_FOLD_METHODS = {"af2_ig", "esmfold2-fast", "esmfold2-standard", "af3", "af3-la"}


def _repo(value) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Repository directory does not exist: {root}")
    return root


def _path(root: Path, value) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _integer(value, name: str, minimum=1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _affinity() -> list[int]:
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))


def resources() -> dict:
    """Read hardware resources without importing a GPU/model framework."""
    cpus = _affinity()
    result = {"gpus": [], "cpu_count": len(cpus), "cpu_ids": cpus,
              "tmux": shutil.which("tmux") is not None}
    if not shutil.which("nvidia-smi"):
        return result
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
        )
        if probe.returncode:
            result["gpu_error"] = probe.stderr.strip()
            return result
        for row in csv.reader(probe.stdout.splitlines()):
            index, name, used, total, utilization = (part.strip() for part in row)
            result["gpus"].append({"id": index, "index": int(index), "name": name,
                                   "memory_used_mb": int(used), "memory_total_mb": int(total),
                                   "utilization": int(utilization)})
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        result["gpu_error"] = str(exc)
    return result


def freeze_task_inputs(repo_root, task_file, destination) -> Path:
    """Copy one native task and its mutable structure dependencies.

    ``destination`` must be a new or empty directory. Task paths follow the
    native invocation-directory rules; a custom scaffold's one CIF path
    follows the native scaffold-YAML-directory rule. Static scaffold IDs are
    retained. The returned ``task.yaml`` is published after native validation.
    """
    import yaml
    from . import notebook as authoring

    root = _repo(repo_root)
    source_task = _path(root, task_file)
    target = _path(root, destination)
    if source_task.suffix.lower() != ".yaml":
        raise ValueError("Task snapshot source must end in .yaml")
    document = yaml.safe_load(source_task.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or len(document) != 1:
        raise ValueError("A task snapshot requires exactly one named native task")
    name, payload = next(iter(document.items()))
    task, normalized = authoring._task_document(name, payload, root)
    payload = yaml.safe_load(normalized)[name]
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Task snapshot destination must be new or empty")
    target.mkdir(parents=True, exist_ok=True)
    if task.input_path is not None:
        copied = target / f"input{task.input_path.suffix}"
        shutil.copyfile(task.input_path, copied)
        fields = payload["antigen"] if task.format == "antibody" else payload
        fields["input"] = str(copied)
    custom = (payload.get("antibody") or {}).get("scaffold")
    if task.antibody is not None and isinstance(custom, Mapping):
        from ..antibody.scaffolds import load_scaffold

        scaffold = load_scaffold(task.antibody.scaffold_path, task.antibody.source_roles)
        saved_yaml = target / "scaffold" / scaffold.yaml_path.name
        saved_cif = saved_yaml.parent / "structure" / scaffold.cif_path.name
        saved_cif.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scaffold.cif_path, saved_cif)
        scaffold_document = yaml.safe_load(scaffold.yaml_path.read_text(encoding="utf-8"))
        scaffold_document["path"] = str(saved_cif.relative_to(saved_yaml.parent))
        saved_yaml.write_text(yaml.safe_dump(scaffold_document, sort_keys=False, allow_unicode=True), encoding="utf-8")
        custom["path"] = str(saved_yaml)
    _, frozen = authoring._task_document(name, payload, root)
    descriptor, temporary = tempfile.mkstemp(prefix=".task.", suffix=".yaml", dir=target)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(frozen)
        result = target / "task.yaml"
        os.replace(temporary, result)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return result


def build_design_command(repo_root, task_file, task_name, output_dir,
                         compile_enabled=False, checkpoint=None, batch_size=None,
                         num_workers=None) -> list[str]:
    """Build native design argv; --out is the container above TASK_NAME."""
    root = _repo(repo_root)
    if type(compile_enabled) is not bool:
        raise ValueError("compile_enabled must be boolean")
    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError("task_name must be a nonempty native task name")
    command = [sys.executable, "-m", "partidesign", "design", str(_path(root, task_file)),
               "--task", task_name, "--out", str(_path(root, output_dir)),
               "--compile", str(compile_enabled).lower()]
    if checkpoint is not None:
        command += ["--ckpt", str(_path(root, checkpoint))]
    for flag, value, minimum in (("--batch-size", batch_size, 1), ("--num-workers", num_workers, 0)):
        if value is not None:
            command += [flag, str(_integer(value, flag, minimum))]
    return command


def build_validate_command(repo_root, input_dir, config_file) -> list[str]:
    root = _repo(repo_root)
    return [sys.executable, "-m", "partidesign", "validate", str(_path(root, input_dir)),
            "--config", str(_path(root, config_file))]


def write_validate_config(path, methods, sources, criteria=(), worker_gpus=.5,
                          worker_cpus=1, method_options=None) -> Path:
    """Write and native-parse a complete validation config, retaining GT/self1.

    ``sources`` contains native source/count/label/settings mappings. Backend
    options use the native root groups (af2, af3, esmfold2, sequence, scorer,
    vina, structure_stage); unknown fields are rejected by the native parser.
    """
    from copy import deepcopy
    import yaml
    from src.validate.config import DEFAULTS, merge, normalize_sources, parse_args

    destination = Path(path).expanduser().resolve()
    if destination.suffix != ".yaml":
        raise ValueError("Validation configuration must end in .yaml")
    if isinstance(methods, str) or isinstance(criteria, str):
        raise ValueError("methods and criteria must be sequences of names")
    options = {} if method_options is None else dict(method_options)
    reserved = {"methods", "criteria", "worker_gpus", "worker_cpus", "unfiltered_success"}
    if reserved.intersection(options) or "sources" in options.get("sequence", {}):
        raise ValueError("method_options cannot override notebook method/source/resource choices")
    config = merge(deepcopy(DEFAULTS), options)
    config.update(methods=list(methods), criteria=list(criteria), worker_gpus=worker_gpus,
                  worker_cpus=worker_cpus, unfiltered_success=not criteria)
    config["sequence"]["sources"] = normalize_sources(list(sources) or [{"source": "gt", "count": 1}])
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".yaml", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        parse_args([str(destination.parent), "--config", temporary, "--plan"])
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def build_cluster_command(repo_root, input_dir, method=None, workers=16, threshold=None) -> list[str]:
    root = _repo(repo_root)
    _integer(workers, "workers")
    if workers > 16:
        raise ValueError("Cluster workers must be in 1..16")
    command = [sys.executable, "-m", "partidesign", "cluster", str(_path(root, input_dir)),
               "--workers", str(workers)]
    if method is not None:
        if not isinstance(method, str) or not method.strip():
            raise ValueError("method must be a nonempty native validation method")
        command += ["--method", method]
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("TM threshold must be finite and in (0, 1]")
        command += ["--tm-threshold", str(threshold)]
    return command


def _tmux(*arguments, check=True):
    if not shutil.which("tmux"):
        raise RuntimeError("tmux is required for detached notebook jobs")
    process = subprocess.run(["tmux", *map(str, arguments)], capture_output=True, text=True, timeout=15)
    if check and process.returncode:
        raise RuntimeError(process.stderr.strip() or "tmux command failed")
    return process


def _session_rows() -> list[tuple[str, str]]:
    probe = _tmux("list-sessions", "-F", "#{session_id}\t#{session_name}", check=False)
    if probe.returncode:
        if "no server running" in probe.stderr or "no sessions" in probe.stderr or "error connecting" in probe.stderr:
            return []
        raise RuntimeError(probe.stderr.strip())
    return [tuple(line.split("\t", 1)) for line in probe.stdout.splitlines() if "\t" in line]


def _metadata(session_id: str) -> dict | None:
    probe = _tmux("show-options", "-qv", "-t", session_id, _OPTION, check=False)
    try:
        value = json.loads(probe.stdout)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _owned(root: Path, name: str, metadata: dict | None) -> bool:
    if not metadata or metadata.get("owner") != _OWNER or metadata.get("repo_root") != str(root) or metadata.get("session") != name:
        return False
    cache = root / ".cache" / "notebook_runtime"
    for field, suffix in (("script_path", ".py"), ("log_path", ".log"), ("exitcode_path", ".exitcode")):
        path = cache / f"{name}{suffix}"
        if metadata.get(field) != str(path) or not path.resolve().is_relative_to(cache.resolve()):
            return False
    return True


def _job(root: Path, session: str) -> tuple[str, dict]:
    if not isinstance(session, str):
        raise ValueError("session must be an owned notebook tmux session name")
    for session_id, name in _session_rows():
        if name == session:
            metadata = _metadata(session_id)
            if _owned(root, name, metadata):
                return session_id, metadata
            break
    raise ValueError(f"Session is not owned by this notebook repository: {session}")


def _process(pid: int):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "ppid": int(stat[1]), "group": int(stat[2]),
                "start": stat[19], "state": stat[0]}
    except (OSError, ValueError, IndexError):
        return None


def _descendants(pid: int) -> list[dict]:
    records = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        record = _process(int(path.parent.name))
        if record:
            records.append(record)
    owned, result = {pid}, []
    while True:
        children = [record for record in records if record["ppid"] in owned and record["pid"] not in owned]
        if not children:
            return result
        result.extend(children)
        owned.update(record["pid"] for record in children)


def _signal_processes(records, signum):
    for record in reversed(records):
        current = _process(record["pid"])
        if current and current["start"] == record["start"] and current["state"] != "Z":
            try:
                os.kill(record["pid"], signum)
            except ProcessLookupError:
                pass


# The cached runner has no project imports, so reconnecting/reloading the
# notebook cannot change an already launched command or its stop handling.
_RUNNER = r'''
import os, pathlib, shlex, signal, subprocess, time, traceback
JOB = __JOB__
__PROCESS_HELPERS__
child = None
requested = None
owned = []
def forward(signum, frame):
    global requested, owned
    requested = signum
    owned = list({(p['pid'], p['start']): p for p in owned + _descendants(os.getpid())}.values())
    group = None
    if child is not None and child.poll() is None:
        group = child.pid
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass
    _signal_processes([p for p in owned if p['group'] != group], signum)
signal.signal(signal.SIGINT, forward)
signal.signal(signal.SIGTERM, forward)
code = 1
with open(JOB['log_path'], 'a', buffering=1) as log:
    try:
        if JOB['cpu_ids'] is not None:
            os.sched_setaffinity(0, JOB['cpu_ids'])
        env = os.environ.copy()
        env.update(JOB['environment'])
        log.write('$ ' + shlex.join(JOB['argv']) + '\n')
        child = subprocess.Popen(JOB['argv'], cwd=JOB['repo_root'], env=env,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        if requested is not None:
            forward(requested, None)
        code = child.wait()
    except BaseException:
        traceback.print_exc(file=log)
    finally:
        if requested is not None:
            _signal_processes(owned, signal.SIGTERM)
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                live = [p for p in owned if (q := _process(p['pid'])) and q['start'] == p['start'] and q['state'] != 'Z']
                if not live:
                    break
                time.sleep(.05)
            _signal_processes(owned, signal.SIGKILL)
        code = 128 - code if code < 0 else code
        temporary = pathlib.Path(JOB['exitcode_path'] + '.tmp')
        temporary.write_text(str(code) + '\n')
        temporary.replace(JOB['exitcode_path'])
        log.write('[exit code ' + str(code) + ']\n')
raise SystemExit(code)
'''


def launch_job(repo_root, kind, argv, *, gpu_ids=None, cpu_budget=None,
               output_dir=None, task_name=None) -> dict:
    """Detach one native command in a retained, repository-owned tmux session."""
    import inspect

    root = _repo(repo_root)
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", kind):
        raise ValueError("kind must contain lowercase letters, digits, '_' or '-'")
    if isinstance(argv, (str, bytes)) or not argv or any(not isinstance(item, str) or "\0" in item for item in argv):
        raise ValueError("argv must be a nonempty sequence of strings")
    available_cpus = _affinity()
    cpu_ids = None
    environment = {}
    if cpu_budget is not None:
        _integer(cpu_budget, "cpu_budget")
        if cpu_budget > len(available_cpus):
            raise ValueError(f"CPU budget exceeds the {len(available_cpus)} CPUs in caller affinity")
        cpu_ids = available_cpus[:cpu_budget]
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            environment[key] = str(cpu_budget)
    if gpu_ids is not None:
        if isinstance(gpu_ids, (str, bytes)):
            raise ValueError("gpu_ids must be a sequence of physical GPU IDs")
        gpu_ids = [str(value) for value in gpu_ids]
        if len(set(gpu_ids)) != len(gpu_ids) or any(not re.fullmatch(r"[A-Za-z0-9_./-]+", value) for value in gpu_ids):
            raise ValueError("GPU IDs must be unique nonempty IDs")
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    started = datetime.now(timezone.utc)
    name = f"pdnotebook-{kind.replace('_', '-')}-{started:%Y%m%d-%H%M%S}-{os.getpid()}-{next(_COUNTER)}"
    cache = root / ".cache" / "notebook_runtime"
    cache.mkdir(parents=True, exist_ok=True)
    if not cache.resolve().is_relative_to(root):
        raise ValueError("Notebook runtime cache must remain inside its repository")
    metadata = {"owner": _OWNER, "repo_root": str(root), "session": name, "kind": kind,
                "argv": list(argv), "python": sys.executable, "started_at": started.isoformat(),
                "script_path": str(cache / f"{name}.py"), "log_path": str(cache / f"{name}.log"),
                "exitcode_path": str(cache / f"{name}.exitcode"),
                "output_dir": str(_path(root, output_dir)) if output_dir is not None else None,
                "task_name": task_name, "gpu_ids": gpu_ids, "cpu_budget": cpu_budget, "cpu_ids": cpu_ids}
    helpers = "from pathlib import Path\n" + "\n".join(inspect.getsource(helper) for helper in (_process, _descendants, _signal_processes))
    script = _RUNNER.replace("__JOB__", repr({**metadata, "environment": environment})).replace("__PROCESS_HELPERS__", helpers)
    with Path(metadata["script_path"]).open("x", encoding="utf-8") as stream:
        stream.write(script)
    Path(metadata["log_path"]).touch(exist_ok=False)
    session_id = None
    try:
        # A live placeholder prevents a fast command finishing before the
        # remain-on-exit option and metadata have been installed.
        env_args = [item for key, value in os.environ.items() for item in ("-e", f"{key}={value}")]
        session_id = _tmux("new-session", "-d", "-P", "-F", "#{session_id}", "-s", name,
                           "-c", root, *env_args,
                           shlex.join([sys.executable, "-c", "import time; time.sleep(31536000)"])).stdout.strip()
        global_environment = _tmux("show-environment", "-g").stdout.splitlines()
        for entry in global_environment:
            key = entry.lstrip("-").split("=", 1)[0]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) and key not in os.environ:
                _tmux("set-environment", "-r", "-t", session_id, key)
        _tmux("set-window-option", "-t", f"{session_id}:0", "remain-on-exit", "on")
        _tmux("set-option", "-t", session_id, _OPTION, json.dumps(metadata, separators=(",", ":")))
        _tmux("respawn-pane", "-k", "-t", f"{session_id}:0.0",
              "exec " + shlex.join([sys.executable, metadata["script_path"]]))
    except BaseException:
        if session_id:
            _tmux("kill-session", "-t", session_id, check=False)
        raise
    return job_status(root, name)


def job_status(repo_root, session) -> dict:
    root = _repo(repo_root)
    session_id, metadata = _job(root, session)
    pane = _tmux("list-panes", "-t", f"{session_id}:0", "-F", "#{pane_pid}\t#{pane_dead}\t#{pane_dead_status}").stdout.strip().split("\t")
    code = None
    try:
        code = int(Path(metadata["exitcode_path"]).read_text().strip())
    except (OSError, ValueError):
        pass
    dead = len(pane) > 1 and pane[1] == "1"
    if code is None and dead and len(pane) > 2 and pane[2]:
        code = int(pane[2])
    stopping = _tmux("show-options", "-qv", "-t", session_id, "@partidesign_stop_requested").stdout.strip() == "1"
    state = "stopped" if code is not None and stopping else "completed" if code == 0 else "failed" if code is not None else "unknown" if dead else "running"
    return {**metadata, "status": state, "state": state, "exit_code": code,
            "pane_pid": int(pane[0]) if pane and pane[0].isdigit() else None}


def list_jobs(repo_root) -> list[dict]:
    root = _repo(repo_root)
    result = []
    for session_id, name in _session_rows():
        if _owned(root, name, _metadata(session_id)):
            try:
                result.append(job_status(root, name))
            except ValueError:
                continue
    return sorted(result, key=lambda job: job["started_at"], reverse=True)


def _runner_matches(job: dict) -> bool:
    pid = job.get("pane_pid")
    if not pid:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return len(command) >= 2 and os.fsdecode(command[0]) == job["python"] and os.fsdecode(command[1]) == job["script_path"]


def stop_job(repo_root, session) -> dict:
    """Stop only the verified runner's descendants, including new sessions."""
    root = _repo(repo_root)
    session_id, _ = _job(root, session)
    job = job_status(root, session)
    if job["status"] != "running":
        return job
    if not _runner_matches(job):
        latest = job_status(root, session)
        if latest["status"] != "running":
            return latest
        raise ValueError("Owned tmux pane is not running its recorded notebook script")
    _tmux("set-option", "-t", session_id, "@partidesign_stop_requested", "1")
    for signum, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        if not _runner_matches(job):
            break
        if signum == signal.SIGKILL:
            _signal_processes(_descendants(job["pane_pid"]), signum)
        else:
            try:
                os.kill(job["pane_pid"], signum)
            except ProcessLookupError:
                return job_status(root, session)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = job_status(root, session)
            if status["status"] != "running":
                return status
            time.sleep(.1)
    return job_status(root, session)


def tail_log(job, lines=80) -> str:
    _integer(lines, "lines")
    path = Path(job["log_path"])
    if not path.is_file():
        return ""
    with path.open(encoding="utf-8", errors="replace") as stream:
        return "".join(deque(stream, maxlen=lines))


def discover_results(input_dir) -> list[dict]:
    """Map actual result.csv filenames to native successful refold slots.

    Each raw row is included, even before validation. Refolds come only from
    native success.csv passing cohorts and explicit ``*.structure`` columns;
    filenames are never reconstructed from sample or sequence labels.
    """
    from src.validate.planning import inventory
    from src.cluster.inputs import _passing

    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Result directory does not exist: {root}")
    tasks = [root] if (root / "result.csv").is_file() else [path for path in sorted(root.iterdir()) if path.is_dir() and not path.name.startswith(".") and path.name not in {"validate", "original"} and (path / "result.csv").is_file()]
    result = []
    for task in tasks:
        samples = inventory(task)
        for label, sample in samples.items():
            result.append({"task": task.name, "task_dir": str(task), "sample_label": label,
                           "raw_path": sample["path"] if sample["row"].get("filename", "").strip() else None,
                           "method": None, "source": None, "sequence_label": None,
                           "refold_path": None, "success": sample["error"] is None,
                           "kind": "raw", "error": sample["error"]})
        validate = task / "validate"
        if not validate.is_dir():
            continue
        for method_root in sorted(validate.iterdir()):
            table = method_root / "success.csv"
            if method_root.name not in _FOLD_METHODS or not table.is_file():
                continue
            with table.open(newline="", encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    label = row.get("sample", "")
                    sample = samples.get(label)
                    if sample is None or sample["error"] is not None or row.get("status") != "complete" or row.get("method") != method_root.name:
                        continue
                    if row.get("input_file") and Path(row["input_file"]).name != Path(sample["path"]).name:
                        continue
                    try:
                        passing = _passing(row, context=str(table))
                        source_index = int(row["source_index"])
                        seq_idx = int(row["seq_idx"])
                    except (ValueError, TypeError, KeyError):
                        continue
                    if not passing or source_index < 0 or seq_idx < 0:
                        continue
                    source = row.get("sequence_source")
                    branch = row.get("sequence_branch") or ("self" if source == "gt" else source)
                    if not source or not branch:
                        continue
                    for field, saved in row.items():
                        if not field.endswith(".structure") or not saved:
                            continue
                        prediction = field.removesuffix(".structure")
                        if row.get(f"{prediction}.status") != "complete":
                            continue
                        path = Path(saved).expanduser()
                        path = (path if path.is_absolute() else method_root / path).resolve()
                        if path.suffix.lower() not in {".cif", ".pdb"} or not path.is_file() or not path.is_relative_to((method_root / "refolds").resolve()):
                            continue
                        result.append({"task": task.name, "task_dir": str(task), "sample_label": label,
                                       "raw_path": sample["path"], "method": method_root.name, "source": source,
                                       "sequence_label": f"{branch}{source_index + 1}", "refold_path": str(path),
                                       "success": True, "kind": "refold", "filter_passed": True,
                                       "seq_idx": seq_idx, "source_index": source_index,
                                       "sequence_branch": branch, "prediction_key": prediction,
                                       "passing_criteria": list(passing)})
    return result
