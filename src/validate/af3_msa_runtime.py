"""Bounded CPU search scheduling outside resident AF3 validation actors."""
from __future__ import annotations

import math
import os
from pathlib import Path

from .progress import emit


def available_cpus():
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.is_file():
        fields = quota.read_text().split()
        if fields[0] != "max":
            count = min(count, max(1, math.floor(int(fields[0]) / int(fields[1]))))
    return count


def search_resources(settings, actor_count, worker_cpus):
    """Reserve actor CPUs, host headroom, and tool costs before Ray starts."""
    capacity = available_cpus()
    actor_cpus = math.ceil(actor_count * worker_cpus)
    headroom = max(2, min(16, capacity // 6))
    maximum = max(0, capacity - actor_cpus - headroom)
    raw = settings["cpu_budget"]
    budget = min(72, maximum) if raw == "auto" else int(raw)
    if budget > maximum:
        raise ValueError(f"MSA CPU budget {budget} plus {actor_cpus} actor CPUs and {headroom} headroom exceeds available {capacity}")
    return {"available": capacity, "actor_cpus": actor_cpus, "headroom": headroom,
            "search_cpus": budget, "ray_cpus": actor_cpus + budget}


def query_cost(settings, modality):
    if settings["mode"] == "cache-only":
        return 0
    return (max(4 * (settings["jackhmmer_n_cpu"] + 1), 9 if settings.get("templates") else 0) if modality == "protein"
            else 3 * (settings["nhmmer_n_cpu"] + 1))


def search_context(repo, settings, query):
    """A CPU-only Ray task: no backend/MPNN/JAX model allocation."""
    from .af3_msa import AF3MSACache
    try:
        reference = AF3MSACache(repo, settings).ensure(query["modality"], query["sequence"])
        return {"status": "ready", "reference": reference}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def run_staged_jobs(ray, actors, jobs, repo, cfg, resources, finished):
    """Prepare ahead at most two samples per actor, refold ready tickets first."""
    from .af3_msa import AF3MSACache
    settings = cfg["af3"]["msa"]
    search_remote = ray.remote(search_context)
    pending_gpu, pending_search = {}, {}
    tickets = set()
    samples, queries, results = {}, {}, {}
    idle = list(actors)
    cache = None
    position = 0
    reserved = 0
    helpers = max(2, settings["search_workers"])

    def key(query):
        return query["modality"], query["sequence"]

    def register(query):
        nonlocal cache
        identity = key(query)
        if identity in queries:
            return identity
        queries[identity] = query
        if cache is None:
            cache = AF3MSACache(repo, settings, readonly=True)
        reference = None if settings["refresh"] else cache.lookup(*identity)
        if reference is not None:
            results[identity] = {"status": "ready", "reference": reference}
            emit({"phase": "msa_cache_hit", "modality": identity[0], "length": len(identity[1])})
        elif settings["mode"] == "cache-only":
            results[identity] = {"status": "failed", "error": "No matching ready AF3 MSA cache entry (cache-only)"}
        elif query_cost(settings, identity[0]) + helpers > resources["search_cpus"]:
            raise ValueError(f"MSA {identity[0]} query needs {query_cost(settings, identity[0]) + helpers} CPU slots including helpers; search budget is {resources['search_cpus']}. Increase --msa-cpus or lower tool threads")
        return identity

    try:
        while pending_gpu or pending_search or samples or position < len(jobs):
            active_queries = {item[0] for item in pending_search.values()}
            for identity, query in queries.items():
                if identity in results or identity in active_queries:
                    continue
                cost = query_cost(settings, identity[0])
                if len(pending_search) >= settings["search_workers"] or reserved + cost + helpers > resources["search_cpus"]:
                    continue
                handle = search_remote.options(num_cpus=cost, num_gpus=0,
                    runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}}).remote(str(repo), settings, query)
                pending_search[handle] = (identity, cost)
                active_queries.add(identity)
                reserved += cost
                emit({"phase": "msa_search_started", "modality": identity[0], "length": len(identity[1]), "active_queries": len(pending_search), "tool_cpu_slots": reserved})
            for ticket_path, item in list(samples.items()):
                if not idle or not all(identity in results for identity in item["keys"]):
                    continue
                actor = idle.pop(0)
                failures = [{"modality": identity[0], "sequence": identity[1], "error": results[identity]["error"]}
                            for identity in item["keys"] if results[identity]["status"] != "ready"]
                refs = [results[identity]["reference"] for identity in item["keys"] if results[identity]["status"] == "ready"]
                pending_gpu[actor.refold_prepared.remote(item["plan"], item["sample"], ticket_path, refs,
                    failures or None)] = (actor, "refold", item["plan"], item["sample"])
                del samples[ticket_path]
            preparing = sum(kind == "prepare" for _, kind, _, _ in pending_gpu.values())
            while idle and position < len(jobs) and len(samples) + preparing < 2 * len(actors):
                actor = idle.pop(0)
                plan, sample = jobs[position]
                position += 1
                pending_gpu[actor.prepare_sample.remote(plan, sample)] = (actor, "prepare", plan, sample)
                preparing += 1
            handles = list(pending_gpu) + list(pending_search)
            if not handles:
                if samples:
                    raise RuntimeError("AF3 staged queue cannot progress with current CPU resources")
                continue
            ready, _ = ray.wait(handles, num_returns=1, timeout=30)
            if not ready:
                emit({"phase": "af3_staged_running", "active_searches": len(pending_search),
                      "prepared_samples": len(samples), "gpu_requests": len(pending_gpu)})
            for handle in ready:
                if handle in pending_search:
                    identity, cost = pending_search.pop(handle)
                    reserved -= cost
                    results[identity] = ray.get(handle)
                    emit({"phase": "msa_search_finished", "modality": identity[0], "length": len(identity[1]), "status": results[identity]["status"], "error": results[identity].get("error")})
                else:
                    actor, kind, plan, sample = pending_gpu.pop(handle)
                    result = ray.get(handle)
                    idle.append(actor)
                    if kind == "prepare":
                        tickets.add(result["ticket"])
                        identities = [register(query) for query in result["queries"]]
                        samples[result["ticket"]] = {"plan": plan, "sample": sample, "keys": identities}
                    else:
                        emit({"sample_finished": result}, Path(plan["task_dir"]) / "validate" / plan["method"] / "logs" / "validate.log")
                        finished(plan)
    finally:
        for handle in list(pending_search) + list(pending_gpu):
            ray.cancel(handle, force=False)
        for ticket_path in tickets:
            Path(ticket_path).unlink(missing_ok=True)
