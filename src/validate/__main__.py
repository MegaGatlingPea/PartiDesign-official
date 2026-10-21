"""python -m src.validate RUN [RUN ...]"""
from __future__ import annotations

from contextlib import ExitStack
import json
import logging
import math
import os
from pathlib import Path
from partidesign.paths import inference_config_dir
import sys
import time

from .config import parse_args
from .progress import emit
from .filters import load_presets
from .planning import make_plan, resolve_tasks
from .reporting import rebuild
from .storage import atomic_json, locked, read_json


def _worker_resources(cfg, method):
    raw_gpus = cfg.get("worker_gpus")
    default_gpus = {"af2_ig": 0.25, "af3": 0.5, "af3-la": 0.5}.get(method, 1.0)
    worker_gpus = default_gpus if raw_gpus is None else raw_gpus
    worker_cpus = cfg.get("worker_cpus", 1.0)
    return float(worker_gpus), float(worker_cpus)


_LOG = logging.getLogger(__name__)
_ACTOR_CLOSE_TIMEOUT_SECONDS = 30.0


def _close_validation_actors(ray, actors, *, timeout=None):
    """Close resident validation backends before tearing down their actors."""
    if timeout is None:
        timeout = _ACTOR_CLOSE_TIMEOUT_SECONDS
    timeout = max(0.0, float(timeout))
    pending = {}
    for actor_index, actor in enumerate(actors):
        try:
            close_ref = actor.close.remote()
        except Exception as exc:
            _LOG.warning(
                "validation actor close request failed; proceeding to kill "
                "(actor=%s): %s",
                actor_index,
                exc,
            )
            continue
        if close_ref is None:
            _LOG.warning(
                "validation actor close request returned no acknowledgement "
                "(actor=%s); proceeding to kill",
                actor_index,
            )
            continue
        pending[close_ref] = actor_index

    deadline = time.monotonic() + timeout
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            for actor_index in pending.values():
                _LOG.warning(
                    "validation actor close acknowledgement timed out; "
                    "proceeding to kill (actor=%s, timeout=%.1fs)",
                    actor_index,
                    timeout,
                )
            break
        try:
            ready, _ = ray.wait(
                list(pending),
                num_returns=1,
                timeout=remaining,
            )
        except Exception as exc:
            for actor_index in pending.values():
                _LOG.warning(
                    "validation actor close wait failed; proceeding to kill "
                    "(actor=%s): %s",
                    actor_index,
                    exc,
                )
            break
        if not ready:
            for actor_index in pending.values():
                _LOG.warning(
                    "validation actor close acknowledgement timed out; "
                    "proceeding to kill (actor=%s, timeout=%.1fs)",
                    actor_index,
                    timeout,
                )
            break
        for close_ref in ready:
            actor_index = pending.pop(close_ref, None)
            if actor_index is None:
                continue
            try:
                ray.get(close_ref)
            except Exception as exc:
                _LOG.warning(
                    "validation actor close failed; proceeding to kill "
                    "(actor=%s): %s",
                    actor_index,
                    exc,
                )


def run_gpu_jobs(repo, plans, cfg, criteria, presets, on_task_finished=None):
    """Dynamic sample queue; each actor owns one resident backend."""
    from .runner import ValidationWorker

    if plans and plans[0]["method"] == "vina_dock":
        # CPU docking deliberately needs neither CUDA nor a Ray GPU actor.
        worker = ValidationWorker(repo, "vina_dock", cfg, criteria, presets, device="cpu")
        try:
            for plan in plans:
                for sample in plan["samples"]:
                    worker.process(plan, sample)
                if on_task_finished is not None:
                    on_task_finished(plan)
        finally:
            worker.close()
        return
    remaining = {id(plan): len(plan["samples"]) for plan in plans}

    def finished(plan):
        remaining[id(plan)] -= 1
        if remaining[id(plan)] == 0 and on_task_finished is not None:
            on_task_finished(plan)

    jobs = []
    for plan in plans:
        if not plan["samples"] and on_task_finished is not None:
            on_task_finished(plan)
        for sample, entry in plan["samples"].items():
            if entry.get("capability", {}).get("status") in {"invalid", "not_applicable"}:
                worker = ValidationWorker(repo, plan["method"], cfg, criteria, presets, device="cpu")
                worker.process(plan, sample)
                worker.close()
                finished(plan)
            else:
                jobs.append((plan, sample))
    if not jobs:
        return
    import ray
    import torch

    count = torch.cuda.device_count()
    requested_gpus = count
    worker_gpus, worker_cpus = _worker_resources(cfg, plans[0]["method"])
    if count < 1:
        raise ValueError("No CUDA GPUs are visible; set CUDA_VISIBLE_DEVICES to the desired GPUs")
    if ray.is_initialized():
        raise RuntimeError("Run standalone validation in its own process; Ray is already initialized")
    actor_capacity = math.floor(1.0 / worker_gpus) * requested_gpus
    actor_count = min(actor_capacity, len(jobs))
    from .config import config_for_method
    method_cfg = config_for_method(cfg, plans[0]["method"])
    msa_enabled = plans[0]["method"] in {"af3", "af3-la"} and method_cfg.get("af3", {}).get("msa", {}).get("mode", "off") != "off"
    resources = None
    num_cpus = math.ceil(actor_count * worker_cpus)
    if msa_enabled:
        from .af3_msa_runtime import search_resources
        resources = search_resources(method_cfg["af3"]["msa"], actor_count, worker_cpus)
        num_cpus = resources["ray_cpus"]
        emit({"phase": "msa_cpu_resources", **resources})
    ray.init(address="local", num_gpus=requested_gpus, num_cpus=num_cpus,
             include_dashboard=False, log_to_driver=True)
    actors = []
    try:
        remote_kwargs = {"num_gpus": worker_gpus, "num_cpus": worker_cpus, "max_restarts": 0}
        if plans[0]["method"] in {"af2_ig", "af3", "af3-la"}:
            remote_kwargs["runtime_env"] = {
                "env_vars": {
                    "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
                        "XLA_PYTHON_CLIENT_PREALLOCATE", "false"
                    )
                }
            }
        remote = ray.remote(**remote_kwargs)(ValidationWorker)
        emit({"phase": "workers_started", "workers": actor_count,
              "worker_gpus": worker_gpus, "worker_cpus": worker_cpus, "gpus": requested_gpus})
        for _ in range(actor_count):
            actors.append(remote.remote(str(repo), plans[0]["method"], cfg, criteria, presets))
        if msa_enabled:
            from .af3_msa_runtime import run_staged_jobs
            run_staged_jobs(ray, actors, jobs, repo, method_cfg, resources, finished)
            return
        pending, position = {}, 0
        for actor in actors:
            plan, sample = jobs[position]
            position += 1
            pending[actor.process.remote(plan, sample)] = (actor, plan)
        while pending:
            ready, _ = ray.wait(list(pending), num_returns=1, timeout=30)
            if not ready:
                emit({"phase": "running", "pending_samples": len(pending)})
                continue
            for handle in ready:
                actor, completed_plan = pending.pop(handle)
                result = ray.get(handle)
                emit({"sample_finished": result}, Path(completed_plan["task_dir"]) / "validate" / completed_plan["method"] / "logs" / "validate.log")
                finished(completed_plan)
                if position < len(jobs):
                    plan, sample = jobs[position]
                    position += 1
                    pending[actor.process.remote(plan, sample)] = (actor, plan)
    finally:
        _close_validation_actors(ray, actors)
        for actor_index, actor in enumerate(actors):
            try:
                ray.kill(actor, no_restart=True)
            except Exception as exc:
                _LOG.warning(
                    "validation actor kill failed after close handling "
                    "(actor=%s): %s",
                    actor_index,
                    exc,
                )
        ray.shutdown()


def main(argv=None):
    args, cfg = parse_args(argv)
    if sum((args.refilter, args.recompute_metrics, args.plan)) > 1:
        raise ValueError("--refilter, --recompute-metrics, and --plan are mutually exclusive")
    if args.recompute_metrics and "vina_dock" in cfg["methods"]:
        raise ValueError("Vina energies cannot be recomputed without docking; use --refilter or --rerun")
    if args.recompute_metrics and (args.rerun or args.score):
        raise ValueError("--recompute-metrics cannot regenerate sequences or run PartiScore")
    repo = Path(__file__).resolve().parents[2]
    presets = load_presets(args.presets or inference_config_dir() / "criteria.yaml")
    unknown = set(cfg["criteria"]) - presets.keys()
    if unknown:
        raise ValueError(f"Unknown criteria: {sorted(unknown)}")
    tasks = resolve_tasks(args.runs, repo, args.task)
    if args.refilter:
        for task in tasks:
            for method in cfg["methods"]:
                root = task / "validate" / method
                plan = read_json(root / ".state/plan.json")
                if not plan:
                    raise ValueError(f"No previous validation to refilter: {root}")
                with locked(root / ".state/run.lock", blocking=False):
                    with ExitStack() as workers:
                        for sample in plan.get("samples", {}):
                            if Path(sample).name != sample or sample in {".", ".."}:
                                raise ValueError("Invalid sample in saved plan")
                            workers.enter_context(locked(root / ".state" / f"{sample}.worker.lock", blocking=False))
                        summary = rebuild(root, cfg["criteria"], presets, persist=True)
                emit({"task": str(task), "method": method, **summary}, root / "logs" / "validate.log")
        return 0
    if args.recompute_metrics:
        from .runner import recompute_persisted_metrics

        for task in tasks:
            for method in cfg["methods"]:
                root = task / "validate" / method
                plan = read_json(root / ".state/plan.json")
                if not plan:
                    raise ValueError(f"No previous validation to recompute: {root}")
                if plan.get("method") != method:
                    raise ValueError(f"Saved validation plan has a different method: {root}")
                with locked(root / ".state/run.lock", blocking=False):
                    with ExitStack() as workers:
                        for sample in plan.get("samples", {}):
                            if Path(sample).name != sample or sample in {".", ".."}:
                                raise ValueError("Invalid sample in saved plan")
                            workers.enter_context(locked(root / ".state" / f"{sample}.worker.lock", blocking=False))
                        summary = recompute_persisted_metrics(root, plan)
                        report = rebuild(root, cfg["criteria"], presets, persist=True)
                emit({"task": str(task), "method": method, **summary, **report}, root / "logs" / "validate.log")
        return 0
    plans = [make_plan(task, method, cfg, repo, args.limit)
             for method in cfg["methods"] for task in tasks]
    if args.plan:
        msa = cfg.get("af3", {}).get("msa", {})
        print(json.dumps({"plans": plans, "af3_msa": {
            "mode": msa.get("mode", "off"), "search_workers": msa.get("search_workers", 2),
            "cache_dir": msa.get("cache_dir", ".cache/af3_msa"),
            "context_queries": "resolved after sequence/input preparation; future MPNN queries unknown",
            "search_started": False,
        }}, indent=2))
        return 0
    emit({"phase": "planning", "tasks": len(tasks), "methods": cfg["methods"]})
    from .capabilities import classify_inputs

    classify_inputs(plans, cfg)
    failures = 0
    for method in cfg["methods"]:
        selected = [p for p in plans if p["method"] == method]
        with ExitStack() as stack:
            for plan in sorted(selected, key=lambda p: p["task_dir"]):
                root = Path(plan["task_dir"]) / "validate" / method
                stack.enter_context(locked(root / ".state/run.lock", blocking=False))
                atomic_json(root / ".state/plan.json", plan)
            emit({"phase": "running", "method": method, "criteria": cfg["criteria"],
                  "tasks": len(selected),
                  "samples": sum(len(p["samples"]) for p in selected)})
            summaries = []

            def publish_task(plan):
                root = Path(plan["task_dir"]) / "validate" / method
                summary = rebuild(root, cfg["criteria"], presets, persist=True)
                emit({"task": plan["task_dir"], "method": method, **summary}, root / "logs" / "validate.log")
                summaries.append(summary)
            run_gpu_jobs(repo, selected, cfg, cfg["criteria"], presets, on_task_finished=publish_task)
        failures += sum(summary.get("incomplete_sequences", 0) for summary in summaries)
    return 2 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        emit({"phase": "failed", "error": str(exc)}, error=True)
        sys.exit(2)
