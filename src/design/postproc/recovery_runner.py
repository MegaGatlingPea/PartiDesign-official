"""Bounded per-sample execution for ``partidesign relax`` recovery.

Recovery differs from the normal inference path because the public CIF is the
only input available to the driver.  Keep the driver light: candidate recovery
and private chemistry preparation happen in the same worker invocation as the
OpenMM refinement.  The driver receives one compact prepared payload at a time
for its ownership checks and then drops it before retaining result accounting.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .context import clear_context_chemistry_cache, preload_context_chemistry
from .recovery import candidate_from_existing
from .runner import (
    PostprocessJob,
    PostprocessPreparationError,
    PostprocessResult,
    SampleParameterizationPayload,
    _RAY_GPU_SLOTS_PER_JOB,
    _RAY_JOB_NUM_CPUS,
    _RAY_JOB_NUM_GPUS,
    _RAY_WORKER_MAX_CALLS,
    _RAY_INFRA_MAX_RETRIES,
    _run_one,
    build_parameterization_payload,
)


_LOG = logging.getLogger(__name__)
_RECOVERY_PRELOADED: set[tuple[Any, ...]] = set()


@dataclass(frozen=True, slots=True)
class RecoveryJob:
    """Small driver-owned descriptor for one existing output sample."""

    task_name: str
    sample_id: str
    base_cif: Path
    refined_cif: Path
    task: Any
    row: Mapping[str, Any]
    ccd_path: Path | None = None
    rerun: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return self.task_name, self.sample_id


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """One recovery outcome, with transient worker preparation metadata."""

    task_name: str
    sample_id: str
    succeeded: bool
    message: str = ""
    initial_energy_kj_mol: float | None = None
    final_energy_kj_mol: float | None = None
    rms_force_kj_mol_nm: float | None = None
    prepared_payload: SampleParameterizationPayload | None = None
    antibody_context: Mapping[str, Any] | None = None
    staged: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return self.task_name, self.sample_id


def _with_preparation(
    result: PostprocessResult,
    *,
    payload: SampleParameterizationPayload | None = None,
    antibody_context: Mapping[str, Any] | None = None,
) -> RecoveryResult:
    return RecoveryResult(
        task_name=result.task_name,
        sample_id=result.sample_id,
        succeeded=result.succeeded,
        message=result.message,
        initial_energy_kj_mol=result.initial_energy_kj_mol,
        final_energy_kj_mol=result.final_energy_kj_mol,
        rms_force_kj_mol_nm=result.rms_force_kj_mol_nm,
        prepared_payload=payload,
        antibody_context=antibody_context,
    )


def _run_recovery_one(job: RecoveryJob, config: Mapping[str, Any]) -> RecoveryResult:
    """Prepare and refine one existing sample inside one Ray task."""

    payload: SampleParameterizationPayload | None = None
    antibody_context: Mapping[str, Any] | None = None
    try:
        # Keep source/CCD preparation cached for this worker's bounded
        # max_calls window.  The key is explicit; no digest or manifest is used.
        input_path = getattr(job.task, "input_path", None)
        design_chains = tuple(sorted(
            str(chain.chain_id)
            for chain in getattr(job.task, "design_chains", ())
        ))
        preload_key = (
            None if input_path is None else str(Path(input_path).resolve()),
            design_chains,
            None if job.ccd_path is None else str(Path(job.ccd_path).resolve()),
        )
        if preload_key not in _RECOVERY_PRELOADED:
            preload_context_chemistry((job.task,), ccd_path=job.ccd_path)
            _RECOVERY_PRELOADED.add(preload_key)
        candidate = candidate_from_existing(
            job.base_cif,
            job.row,
            job.task,
            ccd_path=job.ccd_path,
        )
        payload = build_parameterization_payload(
            candidate,
            task_name=job.task_name,
            sample_id=job.sample_id,
        )
        if getattr(job.task, "format", "normal") == "antibody":
            from ..antibody.results import antibody_result_context

            antibody_context = antibody_result_context(candidate)

        if job.refined_cif.is_symlink():
            raise ValueError("refined artifact may not be a symlink")
        if job.refined_cif.exists():
            if not job.rerun:
                # Driver-side validate_refined_artifact remains authoritative;
                # this result only tells it that an owned staging file exists.
                return RecoveryResult(
                    job.task_name,
                    job.sample_id,
                    True,
                    prepared_payload=payload,
                    antibody_context=antibody_context,
                    staged=True,
                )
            # --rerun explicitly authorizes replacing this one owned staging
            # file.  The base CIF is never touched by worker preparation.
            job.refined_cif.unlink()

        prepared = PostprocessJob(
            task_name=job.task_name,
            sample_id=job.sample_id,
            base_cif=job.base_cif,
            refined_cif=job.refined_cif,
            payload=payload,
        )
        result = _run_one(prepared, config)
        return _with_preparation(
            result,
            payload=payload if result.succeeded else None,
            antibody_context=antibody_context if result.succeeded else None,
        )
    except PostprocessPreparationError as error:
        return RecoveryResult(
            job.task_name,
            job.sample_id,
            False,
            f"{type(error).__name__}: {error}",
        )


def _validate_jobs(jobs: tuple[RecoveryJob, ...]) -> list[tuple[str, str]]:
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("recovery job keys must be globally unique")
    return keys


def _without_worker_state(result: RecoveryResult) -> RecoveryResult:
    """Drop candidate chemistry before retaining completed result accounting."""

    return replace(result, prepared_payload=None, antibody_context=None)


def run_recovery_postprocessing(
    jobs: tuple[RecoveryJob, ...] | list[RecoveryJob],
    *,
    config: Mapping[str, Any],
    on_result: Any = None,
) -> tuple[RecoveryResult, ...]:
    """Run recovery jobs with bounded local-Ray replenishment."""

    jobs = tuple(jobs)
    keys = _validate_jobs(jobs)
    if not jobs:
        return ()
    use_ray = bool(config.get("use_ray", True))
    platform = str(config.get("platform", "CUDA")).upper()
    if not use_ray:
        if platform != "CPU":
            raise ValueError("sequential recovery postprocessing is a CPU diagnostic only")
        completed: list[RecoveryResult] = []
        try:
            for job in jobs:
                result = _run_recovery_one(job, config)
                if result.key != job.key:
                    raise RuntimeError("recovery result violated sample ownership")
                if on_result is not None:
                    on_result(job, result)
                completed.append(_without_worker_state(result))
        finally:
            clear_context_chemistry_cache()
            _RECOVERY_PRELOADED.clear()
        return tuple(completed)
    if platform != "CUDA":
        raise ValueError("Ray recovery postprocessing requires the CUDA platform")

    import ray

    started_here = not ray.is_initialized()
    if not started_here:
        raise RuntimeError("recovery does not attach to an existing Ray runtime")
    results: dict[tuple[str, str], RecoveryResult] = {}
    try:
        ray.init(address="local", include_dashboard=False)
        resources = ray.cluster_resources()
        cpu_capacity = math.floor(float(resources.get("CPU", 0.0)))
        gpu_capacity = math.floor(
            float(resources.get("GPU", 0.0)) * _RAY_GPU_SLOTS_PER_JOB
        )
        max_inflight = min(cpu_capacity, gpu_capacity)
        if max_inflight < 1:
            raise RuntimeError(
                "runner-owned local Ray runtime exposes no feasible resources "
                f"(CPU capacity={cpu_capacity}, GPU capacity={gpu_capacity})"
            )
        _LOG.info(
            "recovery Ray resources: per_task_cpu=%s per_task_gpu=%s "
            "max_inflight=%s selected_samples=%s",
            _RAY_JOB_NUM_CPUS, _RAY_JOB_NUM_GPUS, max_inflight, len(jobs),
        )
        remote_run = ray.remote(
            num_cpus=_RAY_JOB_NUM_CPUS,
            num_gpus=_RAY_JOB_NUM_GPUS,
            max_retries=_RAY_INFRA_MAX_RETRIES,
            retry_exceptions=False,
            max_calls=_RAY_WORKER_MAX_CALLS,
        )(_run_recovery_one)
        jobs_by_key = {job.key: job for job in jobs}
        pending: dict[Any, tuple[str, str]] = {}
        iterator = iter(jobs)
        exhausted = False
        while pending or not exhausted:
            while len(pending) < max_inflight and not exhausted:
                try:
                    job = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                reference = remote_run.remote(job, dict(config))
                pending[reference] = job.key
            if not pending:
                continue
            ready, _ = ray.wait(list(pending), num_returns=1)
            reference = ready[0]
            expected_key = pending.pop(reference)
            try:
                result = ray.get(reference)
            except Exception as error:
                from .openmm.refine import OpenMMSystemError
                raise OpenMMSystemError(
                    f"OpenMM recovery task={expected_key[0]} sample={expected_key[1]} failed ("
                    f"infrastructure retry limit={_RAY_INFRA_MAX_RETRIES}): "
                    f"{type(error).__name__}: {error}"
                ) from error
            if result.key != expected_key or result.key in results:
                raise RuntimeError("Ray recovery result violated sample ownership")
            if on_result is not None:
                on_result(jobs_by_key[expected_key], result)
            results[result.key] = _without_worker_state(result)
    finally:
        if started_here and ray.is_initialized():
            ray.shutdown()
    if set(results) != set(keys):
        raise RuntimeError("Ray recovery result accounting is incomplete")
    return tuple(results[key] for key in keys)


__all__ = ["RecoveryJob", "RecoveryResult", "run_recovery_postprocessing"]
