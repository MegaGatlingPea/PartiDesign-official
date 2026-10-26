"""Select metadata, dispatch a shared Ray pool, and report final records."""
from __future__ import annotations

from collections import Counter, deque
import copy
from dataclasses import dataclass
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from .config import dependency_status, load_config
from .criteria import evaluate, finite_number, load_presets, profile_for_kind
from .schema import Candidate
from .storage import atomic_json, read_json
from .worker import (PostfilterWorker, candidate_paths, computation_settings, failed_record,
                     required_tools, thread_environment)


@dataclass(frozen=True)
class _Job:
    candidate: Candidate
    output: Path


def cpu_budget() -> int:
    limits = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        try:
            limits.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        match = re.fullmatch(r"\s*(\d+)(?:\(x\d+\))?(?:,.*)?\s*", os.environ.get(name, ""))
        if match and int(match[1]) > 0:
            limits.append(int(match[1]))
    return max(1, min(limits))


def worker_count(config: dict, jobs: int, budget: int | None = None) -> int:
    budget = cpu_budget() if budget is None else budget
    cap = budget // config["threads"]
    if cap < 1:
        raise ValueError("postfilter threads exceeds the allocated CPU budget")
    requested = config["workers"]
    return min(jobs, cap if requested is None else min(requested, cap))


def _candidate_matches(saved: dict, candidate: Candidate) -> bool:
    try:
        previous = Candidate.from_dict(saved).to_dict()
    except (KeyError, TypeError, ValueError):
        return False
    current = candidate.to_dict()
    previous.pop("passing_criteria", None)
    current.pop("passing_criteria", None)
    return previous == current


def _completed_record(job: _Job, config: dict, bundle: dict) -> dict | None:
    destination, path = candidate_paths(job.candidate.identity, job.output)
    record = read_json(path)
    if not record:
        return None
    if record.get("version") != 2:
        raise ValueError(f"legacy postfilter record in {job.output}; request a fresh --out directory")
    if (record.get("status") != "complete" or record.get("geometry_status") != "complete"
            or record.get("identity") != job.candidate.identity
            or not _candidate_matches(record.get("candidate", {}), job.candidate)
            or record.get("relaxed_path") != str(destination) or not destination.is_file()):
        return None
    details = record.get("details", {})
    if not isinstance(details, dict):
        return None
    kind = details.get("binder_kind", job.candidate.binder_kind)
    try:
        names, required = profile_for_kind(kind, bundle)
    except ValueError:
        return None
    if record.get("settings") != computation_settings(config, bundle, required):
        return None
    metrics = record.get("metrics", {})
    if not isinstance(metrics, dict) or any(not finite_number(metrics.get(name)) for name in required):
        return None
    record["candidate"] = job.candidate.to_dict()
    record["decisions"] = {name: evaluate(metrics, name, bundle["presets"][name]) for name in names}
    return record


def _scientific_settings(config: dict, bundle: dict) -> dict:
    stable_bundle = copy.deepcopy(bundle)
    for preset in stable_bundle["presets"].values():
        preset.pop("description", None)
        for rule in preset["rules"]:
            rule.pop("value", None)
    profiles = {}
    for kind in ("protein", "peptide"):
        names, required = profile_for_kind(kind, bundle)
        profiles[kind] = {"names": names, "required": sorted(required)}
    return {
        "config": {name: config[name] for name in ("seed", "threads", "dalphaball", "dssp")},
        "bundle": stable_bundle, "profiles": profiles,
    }


def _check_output(output: Path, task: str, config: dict, bundle: dict, rerun: bool = False) -> None:
    saved = read_json(output / "resolved_config.json")
    if saved is None:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"unrecognized or legacy postfilter output {output}; request a fresh --out directory")
        return
    if saved.get("version") != 2 or saved.get("task_dir") != task:
        raise ValueError(f"legacy or different-task postfilter output {output}; request a fresh --out directory")
    try:
        matches = _scientific_settings(saved.get("config", {}), saved.get("bundle", {})) == _scientific_settings(config, bundle)
    except (KeyError, TypeError, ValueError):
        matches = False
    if not matches and not rerun:
        raise ValueError(f"postfilter scientific settings changed in {output}; use --rerun for selected samples")


def _dispatch_jobs(jobs: list[_Job], config: dict, bundle: dict, *, ray_module=None) -> tuple[list[dict], list[dict]]:
    """Run each sample once, replacing lost actors only for subsequent samples."""
    if not jobs:
        return [], []
    from tqdm import tqdm

    ray = importlib.import_module("ray") if ray_module is None else ray_module
    budget = cpu_budget()
    owned_runtime = not ray.is_initialized()
    actors = {}
    ready = {}
    active = {}
    waiting = deque(enumerate(jobs))
    results = [None] * len(jobs)
    audit = []
    progress = tqdm(total=len(jobs), desc="Postfilter", unit="sample", file=sys.stderr)
    try:
        count = worker_count(config, len(jobs), budget)
        if owned_runtime:
            ray.init(address="local", num_cpus=count * config["threads"], num_gpus=0,
                     include_dashboard=False, log_to_driver=False)
        resources = ray.cluster_resources() if owned_runtime else ray.available_resources()
        if resources.get("CPU", 0) < config["threads"]:
            raise ValueError("Ray has insufficient CPU resources for one postfilter worker")
        budget = min(budget, math.floor(resources["CPU"]))
        count = worker_count(config, len(jobs), budget)
        remote_worker = ray.remote(
            num_cpus=config["threads"], num_gpus=0, max_restarts=0, max_task_retries=0,
            runtime_env={"env_vars": thread_environment(config["threads"])},
        )(PostfilterWorker)

        def start_actor():
            actor = remote_worker.remote(config, bundle)
            actors[actor] = {"submitted": 0}
            ready[actor.ready.remote()] = (actor, time.monotonic())

        def submit(actor):
            index, job = waiting.popleft()
            # Invalidate prior owned output before a new sample can publish.
            destination, completion = candidate_paths(job.candidate.identity, job.output)
            if destination.resolve() == Path(job.candidate.refold_path).expanduser().resolve():
                raise ValueError("relaxed output cannot overwrite the source structure")
            completion.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            reference = actor.process.remote(job.candidate.to_dict(), str(job.output))
            actors[actor]["submitted"] += 1
            active[reference] = (actor, index, job, time.monotonic())

        def lost_sample(reference, actor, index, job, started, reason, *, terminal=True):
            info = actors.pop(actor)
            ray.kill(actor, no_restart=True)
            if not terminal:
                # A timed-out process may still be publishing. Ray must confirm
                # this call is terminal before its final completion is inspected.
                terminated, _ = ray.wait([reference], num_returns=1, timeout=config["timeout"])
                if not terminated:
                    raise RuntimeError("timed-out postfilter worker did not terminate")
                try:
                    ray.get(reference)
                except Exception:
                    pass
            audit.append({**info, "retired_reason": reason})
            # This is the only exceptional-path completion read.
            try:
                record = _completed_record(job, config, bundle)
            except (OSError, TypeError, ValueError):
                record = None
            if record is None:
                record = failed_record(job.candidate, reason, time.monotonic() - started)
                record["geometry_status"] = "failed"
                names, required = profile_for_kind(job.candidate.binder_kind, bundle)
                record["settings"] = computation_settings(config, bundle, required)
                record["errors"] = {name: reason for name in sorted(required)}
                record["decisions"] = {
                    name: {"criteria": name, "status": "unavailable", "reasons": [reason]} for name in names
                }
                atomic_json(candidate_paths(job.candidate.identity, job.output)[1], record)
            results[index] = record
            progress.update(1)
            if waiting:
                start_actor()

        def completed_reference(reference):
            if reference in ready:
                actor, _ = ready.pop(reference)
                try:
                    info = ray.get(reference)
                except Exception as exc:
                    raise RuntimeError(f"postfilter worker startup failed: {exc}") from exc
                if not isinstance(info, dict) or info.get("initializations") != 1:
                    raise RuntimeError("postfilter worker startup returned invalid initialization counters")
                actors[actor].update(info)
                if waiting:
                    submit(actor)
                return
            actor, index, job, started = active.pop(reference)
            try:
                record = ray.get(reference)
            except Exception as exc:
                lost_sample(reference, actor, index, job, started, f"worker failed: {type(exc).__name__}: {exc}")
            else:
                if (not isinstance(record, dict) or record.get("version") != 2
                        or record.get("identity") != job.candidate.identity
                        or record.get("status") not in {"complete", "failed"}):
                    lost_sample(reference, actor, index, job, started, "worker returned an invalid sample record")
                else:
                    results[index] = record
                    progress.update(1)
                    if waiting:
                        submit(actor)

        for _ in range(count):
            start_actor()
        while ready or active:
            deadlines = [item[3] + config["timeout"] for item in active.values()]
            deadlines.extend(started + config["timeout"] for _, started in ready.values())
            deadline = min(deadlines, default=None)
            timeout = max(0.0, deadline - time.monotonic()) if deadline is not None else None
            completed, _ = ray.wait([*ready, *active], num_returns=1, timeout=timeout)
            for reference in completed:
                completed_reference(reference)
            while True:
                now = time.monotonic()
                expired_ready = [reference for reference, (_, started) in ready.items()
                                 if now - started >= config["timeout"]]
                expired_active = [reference for reference, item in active.items()
                                  if now - item[3] >= config["timeout"]]
                expired = [*expired_ready, *expired_active]
                if not expired:
                    break
                # Driver handling may cross a deadline after other calls have
                # already finished. Observe those futures before declaring a timeout.
                completed, _ = ray.wait(expired, num_returns=len(expired), timeout=0)
                if completed:
                    for reference in completed:
                        completed_reference(reference)
                    continue
                if expired_ready:
                    raise RuntimeError(f"postfilter worker startup timeout after {config['timeout']} seconds")
                reference = expired_active[0]
                actor, index, job, started = active.pop(reference)
                lost_sample(reference, actor, index, job, started,
                            f"worker timeout after {config['timeout']} seconds", terminal=False)
        for actor, info in actors.items():
            try:
                stats = ray.get(actor.stats.remote())
            except Exception as exc:
                stats = {"stats_error": f"{type(exc).__name__}: {exc}"}
            audit.append({**info, **stats})
        return results, audit
    finally:
        try:
            for actor in actors:
                ray.kill(actor, no_restart=True)
            if owned_runtime:
                ray.shutdown()
        finally:
            progress.close()


def run(args):
    from partidesign.paths import inference_config_dir
    from .inputs import select_refolds
    from .reporting import write_reports

    if not args.inputs:
        raise ValueError("postfilter requires RUN_OR_TASK_OR_SUCCESS")
    config = load_config(args.config)
    bundle = load_presets(inference_config_dir() / "postfilter_presets.yaml")
    selected = select_refolds(args.inputs, task_names=args.task)
    if not selected.task_dirs:
        reasons = "; ".join(item.get("reason", item.get("code", "selection error")) for item in selected.diagnostics)
        raise ValueError(reasons or "No validation task matched the selected inputs")
    task_names = [Path(task).name for task in selected.task_dirs]
    duplicates = sorted(name for name, count in Counter(task_names).items() if count > 1)
    if duplicates:
        raise ValueError("duplicate selected task names: " + ", ".join(duplicates))
    nonerrors = {"not_ppi", "no_quality_success", "not_complex"}
    selection_errors = [item for item in selected.diagnostics if item.get("code") not in nonerrors]
    tools = set()
    for kind in {candidate.binder_kind for candidate in selected.candidates}:
        tools.update(required_tools(profile_for_kind(kind, bundle)[1]))
        if kind == "protein":
            tools.update(required_tools(profile_for_kind("peptide", bundle)[1]))
    dependencies = dependency_status(config, required_tools=tools)
    if args.plan:
        budget = cpu_budget()
        print(json.dumps({
            "selected": len(selected.candidates), "tasks": selected.task_dirs,
            "dependencies": dependencies, "selection_counts": selected.counts,
            "defaults": bundle["defaults"], "protocol": bundle["protocol"],
            "cpu_budget": budget, "workers": worker_count(config, len(selected.candidates), budget),
            "threads": config["threads"],
            "candidates": [{"identity": candidate.identity, "refold_path": candidate.refold_path,
                            "binder_kind": candidate.binder_kind} for candidate in selected.candidates],
            "diagnostics": selected.diagnostics,
        }, indent=2))
        return 2 if selection_errors else 0

    outputs = {}
    for task, name in zip(selected.task_dirs, task_names):
        output = (Path(args.out).expanduser().absolute() / name if len(task_names) > 1 else Path(args.out).expanduser().absolute()) if args.out else Path(task) / "postfilter"
        output = output.resolve()
        if output == Path(task).resolve() or output.is_relative_to((Path(task) / "validate").resolve()):
            raise ValueError("postfilter output must be independent of upstream validation")
        _check_output(output, task, config, bundle, args.rerun)
        outputs[task] = output
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("selected tasks resolve to the same postfilter output")

    records = {task: [] for task in selected.task_dirs}
    jobs = []
    positions = []
    resumed = Counter()
    resumed_records = []
    for candidate in selected.candidates:
        job = _Job(candidate, outputs[candidate.task_dir])
        previous = None if args.rerun else _completed_record(job, config, bundle)
        index = len(records[candidate.task_dir])
        records[candidate.task_dir].append(previous)
        if previous is None:
            jobs.append(job)
            positions.append((candidate.task_dir, index))
        else:
            resumed[candidate.task_dir] += 1
            resumed_records.append((job, previous))
    if jobs and any(value != "available" for value in dependencies.values()):
        raise ValueError(f"postfilter dependency gap: {dependencies}; see docs/postfilter.md")
    if jobs:
        worker_count(config, len(jobs))
    for task, output in outputs.items():
        atomic_json(output / "resolved_config.json", {
            "version": 2, "task_dir": task, "config": config, "bundle": bundle,
        })
    dispatched, audit = _dispatch_jobs(jobs, config, bundle)
    for (task, index), record in zip(positions, dispatched):
        records[task][index] = record
    # Rejudging thresholds only touches the existing small completed records.
    for job, record in resumed_records:
        atomic_json(candidate_paths(job.candidate.identity, job.output)[1], record)
    if audit:
        print(json.dumps({"worker_stats": audit}))
    failures = bool(selection_errors)
    for task, output in outputs.items():
        diagnostics = [item for item in selected.diagnostics if item.get("task_dir") in (None, task)]
        counts = {
            **selected.counts.get(task, {}), "selected": len(records[task]),
            "resumed": resumed[task], "executed": len(records[task]) - resumed[task],
            "metadata_error_events": sum(item.get("code") not in nonerrors for item in diagnostics),
        }
        summary = write_reports(output, records[task], diagnostics, counts)
        failures |= any(record["status"] != "complete" for record in records[task])
        print(json.dumps({"task": task, "output": str(output), **summary}))
    return 2 if failures else 0
