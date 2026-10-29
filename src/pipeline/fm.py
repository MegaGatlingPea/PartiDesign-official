"""Resident native design execution for selected partial-diffusion references."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from .state import write_csv


class PartialOwner:
    """Reuse native FM/VAE weights inside the existing GPU resource owner."""

    def __init__(self):
        self.models = {}
        self.loads = 0

    def _load(self, checkpoint, **options):
        from src.design.model.checkpoint import load_combined_checkpoint
        key = json.dumps([str(checkpoint), options], sort_keys=True)
        if key not in self.models:
            self.models[key] = load_combined_checkpoint(checkpoint, **options)
            self.loads += 1
        return self.models[key]

    @staticmethod
    def _compile(model, settings):
        from src.design.model.checkpoint import compile_denoiser
        previous = getattr(model, "_inference_compile_config", None)
        if previous == settings:
            return model
        if getattr(model, "_denoiser_compiled", False):
            model.nn = model.nn._orig_mod
            model._denoiser_compiled = False
        return compile_denoiser(model, settings)

    def run(self, references, *, authored, effective_generation, settings, output, seed,
            public_output=None):
        """Batch references through the same design runtime and output assembler."""
        import torch
        from src.design.core.api import DesignRunner
        from src.design.core.runtime import run_runtime
        from src.validate.inputs import discover_inputs

        output = Path(output)
        payloads = {}
        for reference in references:
            payload = copy.deepcopy(authored)
            payload.update(sample_num=settings["replicas"], score=False, postprocess=False)
            payload["partial"] = {
                "reference": reference["path"],
                "result_csv": reference["result_csv"],
                "fraction": settings["fraction"],
            }
            payloads[reference["candidate_id"]] = payload
        native = copy.deepcopy(effective_generation)
        native.update(tasks=payloads, tasks_file=None, task_names=list(payloads),
                      output_dir=str(output), run_name="partial", seed=int(seed))
        # Recompute the derived execution policy for the reference batch.
        native["model"].pop("execution_policy", None)
        native["search"]["algorithm"] = None
        native["reward"] = {"_target_": None}
        runner = DesignRunner(native, invocation_dir=Path.cwd())
        before = self.loads
        try:
            result = run_runtime(runner.loaded, runner.tasks, runner.requests,
                                 model_loader=self._load, model_compiler=self._compile,
                                 compact_task_snapshots=True)
            entries = list(discover_inputs(output, list(payloads)))
            destinations = {}
            if public_output is not None:
                public_output = Path(public_output)
                public_output.mkdir(parents=True, exist_ok=True)
                for parent_id in payloads:
                    source = output / parent_id
                    destination = public_output / parent_id
                    private = source / ".work"
                    private.mkdir(exist_ok=True)
                    # The native result table and structures remain direct
                    # children; helper products stay with their parent hidden.
                    for name in ("original", "logs"):
                        path = source / name
                        if path.exists():
                            os.replace(path, private / name)
                    snapshot = source / "resolved_config.yaml"
                    if snapshot.is_file():
                        metadata = private / parent_id
                        metadata.mkdir()
                        os.replace(snapshot, metadata / snapshot.name)
                    native_rows = [_relocated(dict(entry.row), source, destination)
                                   for entry in entries if entry.task_dir.name == parent_id]
                    write_csv(source / "result.csv", native_rows)
                    os.replace(source, destination)
                    destinations[parent_id] = (source, destination)
            rows = []
            for entry in entries:
                if entry.row.get("error"):
                    continue
                if entry.error:
                    raise RuntimeError(entry.error)
                parent_id = entry.task_dir.name
                path, row = str(entry.path), dict(entry.row)
                if parent_id in destinations:
                    source, destination = destinations[parent_id]
                    path = _relocated(path, source, destination)
                    row = _relocated(row, source, destination)
                rows.append({"parent_id": parent_id, "path": path, "row": row})
            expected = len(references) * settings["replicas"]
            if len(rows) != result.succeeded or result.succeeded + result.failed != expected:
                raise RuntimeError(f"Partial diffusion accounting differs from {expected} replicas")
            return {"candidates": rows, "failed": result.failed, "model_loads": self.loads - before}
        finally:
            # Refold workers retain their reservation. FM weights stay cached on
            # CPU between phases; no second Ray reservation or checkpoint reload.
            for model in self.models.values():
                model.to(torch.device("cpu"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def close(self):
        self.models.clear()


def _relocated(value, source, destination):
    """Rewrite only paths owned by the atomically published parent folder."""
    if isinstance(value, dict):
        return {key: _relocated(item, source, destination) for key, item in value.items()}
    if isinstance(value, str):
        return value.replace(str(source) + "/", str(destination) + "/")
    return value
