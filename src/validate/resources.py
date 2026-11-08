"""Shared reservation rules for resident prediction workers."""
import math

def worker_resources(config, method):
    from .methods import VALIDATION_REFOLD_METHODS
    raw = config.get("worker_gpus")
    spec = VALIDATION_REFOLD_METHODS.get(method)
    gpu = (spec.worker_gpus if spec else 1.0) if raw is None else raw
    cpu = config.get("worker_cpus", 1.0)
    if isinstance(gpu, bool) or not isinstance(gpu, (int, float)) or not math.isfinite(gpu) or not 0 < gpu <= 1:
        raise ValueError("worker_gpus must be finite and in (0, 1]")
    if isinstance(cpu, bool) or not isinstance(cpu, (int, float)) or not math.isfinite(cpu) or cpu <= 0:
        raise ValueError("worker_cpus must be a positive finite number")
    return float(gpu), float(cpu)

def worker_capacity(resources, gpu, cpu, *, limit=None):
    capacity = int(float(resources.get("CPU", 0)) // cpu)
    if gpu:
        capacity = min(capacity, math.floor(1.0 / gpu) * int(resources.get("GPU", 0)))
    return min(capacity, limit) if limit is not None else capacity
