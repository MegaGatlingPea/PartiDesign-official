"""One single-core worker per validation success directory; no model imports."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
import csv
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from .inputs import ClusterJob, load_job, discover_jobs


MAX_WORKERS = 16
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
_POLYMERS = frozenset({"protein", "rna", "dna"})
_DETAIL_FIELDS = (
    "run", "task", "method", "criterion", "design_signature", "sample", "input_file",
    "design_chains", "passing_branches", "cluster_id", "cluster_size", "representative",
    "representative_file", "is_representative", "tm_to_representative", "tm_threshold",
    "score_normalization", "algorithm", "structure_source", "usalign", "usalign_version",
)
_SUMMARY_FIELDS = (
    "run", "task", "method", "criterion", "design_signature", "n_success_backbones",
    "n_clusters", "n_singletons", "largest_cluster", "tm_threshold", "score_normalization",
    "algorithm", "structure_source", "usalign", "usalign_version",
)


@dataclass(frozen=True)
class WorkerOptions:
    criteria: tuple[str, ...]
    threshold: float
    executable: str
    version: str
    scratch_root: Path


def _single_core() -> None:
    for name in _THREAD_VARS:
        os.environ[name] = "1"


def _sample_key(name: str):
    return tuple((1, int(piece)) if piece.isdigit() else (0, piece)
                 for piece in re.split(r"(\d+)", name))


def _polymer_chains(sample) -> tuple[dict, ...]:
    chains = tuple(chain for chain in sample.design_chains
                   if str(chain.get("modality", "")).lower() in _POLYMERS)
    if not chains:
        raise ValueError(f"{sample.structure}: no design protein/RNA/DNA chains; ligand clustering is not supported")
    return chains


def _signature(sample) -> tuple[str, ...]:
    return tuple(sorted(str(chain["modality"]).lower() for chain in _polymer_chains(sample)))


def _partitions(data):
    partitions: dict[tuple[str, ...], list[str]] = {}
    for name in sorted(data.samples, key=_sample_key):
        partitions.setdefault(_signature(data.samples[name]), []).append(name)
    return partitions


def _membership_masks(data) -> dict[str, int]:
    masks = {name: 0 for name in data.samples}
    for index, names in enumerate(data.groups.values()):
        for name in names:
            masks[name] |= 1 << index
    return masks


def _comparison_count(data) -> int:
    masks = _membership_masks(data)
    return sum(bool(masks[first] & masks[second])
               for names in _partitions(data).values()
               for first, second in combinations(names, 2))


def resolve_executable(path: Path | str | None) -> str:
    if path is not None:
        value = Path(path).expanduser().resolve()
        if not value.is_file() or not os.access(value, os.X_OK):
            raise ValueError(f"USalign is not executable: {value}")
        return str(value)
    local = Path(sys.executable).parent / "USalign"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    found = shutil.which("USalign")
    if found:
        return str(Path(found).resolve())
    raise ValueError("USalign was not found; install bioconda::usalign in the active environment or pass --usalign /path/to/USalign")


def executable_version(executable: str) -> str:
    result = subprocess.run([executable, "-v"], capture_output=True, text=True,
                            timeout=15, check=False)
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        if "US-align" in line or "USalign" in line:
            return line.strip().strip("* ")
    raise RuntimeError(f"Cannot identify USalign executable: {executable}")


def _write_reports(method_dir: Path, details: list[dict], summaries: list[dict],
                   scratch_root: Path) -> None:
    # Stage and back up reports under the ignored cache. Restore an earlier
    # report if publishing the second CSV fails, so a partial pair is not left
    # behind after an ordinary filesystem error.
    staging_root = scratch_root
    if scratch_root.stat().st_dev != method_dir.stat().st_dev:
        # Atomic replacement requires a cache on the output filesystem.
        staging_root = method_dir / ".cache" / "cluster"
        staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reports-", dir=staging_root) as temporary:
        temporary = Path(temporary)
        staged = []
        published = []
        for filename, fields, rows in (("clusters.csv", _DETAIL_FIELDS, details),
                                        ("cluster_summary.csv", _SUMMARY_FIELDS, summaries)):
            destination = method_dir / filename
            new = temporary / (filename + ".new")
            backup = temporary / (filename + ".old") if destination.exists() else None
            if backup is not None:
                shutil.copyfile(destination, backup)
            with new.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            staged.append((new, destination, backup))
        try:
            for new, destination, backup in staged:
                os.replace(new, destination)
                published.append((destination, backup))
        except OSError as error:
            rollback_errors = []
            for destination, backup in reversed(published):
                try:
                    if backup is None:
                        destination.unlink(missing_ok=True)
                    else:
                        os.replace(backup, destination)
                except OSError as rollback_error:
                    rollback_errors.append(str(rollback_error))
            if rollback_errors:
                raise RuntimeError(
                    "Failed to publish and restore cluster reports; inspect "
                    + str(method_dir) + ": " + "; ".join(rollback_errors)
                ) from error
            raise

def process_job(job: ClusterJob, options: WorkerOptions) -> dict:
    _single_core()
    import numpy as np
    from .alignment import prepare_structure, score_pair, greedy_clusters

    started = time.monotonic()
    data = load_job(job, criteria=options.criteria)
    details: list[dict] = []
    summaries: list[dict] = []
    comparisons = 0
    masks = _membership_masks(data)
    base = {"run": job.task_dir.parent.name, "task": job.task_dir.name,
            "method": job.method_dir.name, "tm_threshold": options.threshold,
            "score_normalization": "mean(TM_L1,TM_L2)", "algorithm": "px_greedy",
            "structure_source": "original", "usalign": options.executable,
            "usalign_version": options.version}
    options.scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="success-", dir=options.scratch_root) as scratch:
        scratch = Path(scratch)
        for signature, names in sorted(_partitions(data).items()):
            prepared = []
            for index, name in enumerate(names):
                sample = data.samples[name]
                output = scratch / f"design{index}.cif"
                actual = prepare_structure(sample.structure, _polymer_chains(sample), output)
                if actual != signature:
                    raise ValueError(f"{sample.structure}: extracted design composition {actual} != {signature}")
                prepared.append(output)
            scores = np.full((len(names), len(names)), np.nan, dtype=np.float64)
            np.fill_diagonal(scores, 1.0)
            multimer = len(signature) > 1
            molecule_type = "auto" if len(set(signature)) > 1 else signature[0]
            for first, second in combinations(range(len(names)), 2):
                if not masks[names[first]] & masks[names[second]]:
                    continue
                value = score_pair(prepared[first], prepared[second],
                                   executable=options.executable, multimer=multimer,
                                   molecule_type=molecule_type)
                scores[first, second] = scores[second, first] = value
                comparisons += 1
            for criterion, selected_names in sorted(data.groups.items()):
                selected = set(selected_names)
                indices = [index for index, name in enumerate(names) if name in selected]
                if not indices:
                    continue
                cohort_names = [names[index] for index in indices]
                cohort_scores = scores[np.ix_(indices, indices)]
                clusters = greedy_clusters(cohort_names, cohort_scores, options.threshold)
                summary = dict(base, criterion=criterion, design_signature="+".join(signature),
                               n_success_backbones=len(indices), n_clusters=len(clusters),
                               n_singletons=sum(len(cluster) == 1 for cluster in clusters),
                               largest_cluster=max(map(len, clusters), default=0))
                summaries.append(summary)
                for cluster_id, cluster in enumerate(clusters, 1):
                    representative = cohort_names[cluster[0]]
                    for index in cluster:
                        name = cohort_names[index]
                        sample = data.samples[name]
                        details.append(dict(
                            base, criterion=criterion, design_signature="+".join(signature),
                            sample=name, input_file=str(sample.structure),
                            design_chains=json.dumps([chain["chain"] for chain in _polymer_chains(sample)]),
                            passing_branches=json.dumps(data.sources.get(criterion, {}).get(name, ())),
                            cluster_id=cluster_id, cluster_size=len(cluster), representative=representative,
                            representative_file=str(data.samples[representative].structure),
                            is_representative=int(index == cluster[0]),
                            tm_to_representative=f"{cohort_scores[cluster[0], index]:.8f}",
                        ))
    for criterion, names in sorted(data.groups.items()):
        if not names:
            summaries.append(dict(base, criterion=criterion, design_signature="",
                                  n_success_backbones=0, n_clusters=0,
                                  n_singletons=0, largest_cluster=0))
    _write_reports(job.method_dir, details, summaries, options.scratch_root)
    return {"task": str(job.task_dir), "method": job.method_dir.name,
            "backbones": len(data.samples), "groups": len(summaries),
            "clusters": sum(row["n_clusters"] for row in summaries),
            "comparisons": comparisons, "seconds": round(time.monotonic() - started, 3),
            "pid": os.getpid(), "thread_limit": 1,
            "report": str(job.method_dir / "clusters.csv")}


def run(paths, *, tasks=(), methods=(), criteria=(), threshold=0.5,
        workers=16, executable=None, plan=False) -> int:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("--tm-threshold must be finite and in (0,1]")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise ValueError("--workers must be an integer in [1,16]")
    jobs = discover_jobs(paths, tasks=tasks, methods=methods)
    if not jobs:
        raise ValueError("No validation success.csv tables matched; pass a task, direct-child run, method, or success directory")
    # Spawned workers inherit this limit before NumPy/Biotite imports. Each job
    # evaluates pairs sequentially; no nested process or thread pools are used.
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    active = min(workers, available, len(jobs))
    print(f"Cluster directories: {len(jobs)}; workers: {active}; CPU threads per worker: 1", flush=True)
    if plan:
        for job in jobs:
            data = load_job(job, criteria=criteria)
            print(f"PLAN {job.task_dir} method={job.method_dir.name} "
                  f"successful_backbones={len(data.samples)} criteria={','.join(sorted(data.groups)) or '(none)'} "
                  f"pairs={_comparison_count(data)}", flush=True)
        return 0
    resolved = resolve_executable(executable)
    options = WorkerOptions(tuple(criteria), threshold, resolved,
                            executable_version(resolved), Path.cwd() / ".cache" / "cluster")
    saved_environment = {name: os.environ.get(name) for name in _THREAD_VARS}
    _single_core()
    failures = 0
    try:
        with ProcessPoolExecutor(max_workers=active,
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_single_core) as pool:
            pending = {pool.submit(process_job, job, options): job for job in jobs}
            for future in as_completed(pending):
                job = pending[future]
                try:
                    result = future.result()
                except Exception as error:
                    failures += 1
                    print(f"FAILED {job.task_dir} method={job.method_dir.name}: "
                          f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
                    continue
                print("DONE " + json.dumps(result, sort_keys=True), flush=True)
    finally:
        for name, value in saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    print(f"Completed directories: {len(jobs) - failures}/{len(jobs)}", flush=True)
    return 2 if failures else 0
