"""Resident native design execution for selected partial-diffusion references."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from .state import failed_row, refreshed_row, write_csv


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
            payload = copy.deepcopy(reference.get("authored", authored))
            effective = reference.get("settings", settings)
            payload.update(sample_num=effective["replicas"], score=False, postprocess=False)
            partial = copy.deepcopy(payload.get("partial") or {})
            partial.update({
                "reference": reference["path"],
                "fraction": effective["fraction"],
            })
            partial.pop("result_csv", None)
            if reference.get("result_csv"):
                partial["result_csv"] = reference["result_csv"]
            payload["partial"] = partial
            payloads[reference["candidate_id"]] = payload
        # Replicas are expanded by DesignRunner before native batching. Keep
        # the configured batch size even when there is only one parent.
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
            from .memory_trace import generation_memory_trace
            with generation_memory_trace():
                result = run_runtime(runner.loaded, runner.tasks, runner.requests,
                                     model_loader=self._load, model_compiler=self._compile,
                                     compact_task_snapshots=True)
            entries = list(discover_inputs(output, list(payloads)))
            resolved_rows = {}
            for entry in entries:
                row = dict(entry.row)
                if not failed_row(row) and not entry.error and '"ligand"' in str(row.get("design_chains","")):
                    from src.validate.inputs import load_input
                    record=load_input(entry.path,row,entry.task_dir)
                    row=refreshed_row(record,entry.path,atoms=record.atoms)
                if row.get("heavy_role") and not entry.error and not row.get("error"):
                    from src.validate.inputs import load_input
                    row = dict(load_input(entry.path,row,entry.task_dir).row)
                resolved_rows[str(entry.path)] = row
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
                        for entry in entries:
                            if entry.task_dir.name==parent_id:
                                resolved_rows[str(entry.path)]["_pipeline_metadata"]=str(metadata)
                    native_rows = [_relocated(resolved_rows[str(entry.path)], source, destination)
                                   for entry in entries if entry.task_dir.name == parent_id]
                    write_csv(source / "result.csv", native_rows)
                    os.replace(source, destination)
                    destinations[parent_id] = (source, destination)
            rows = []
            for entry in entries:
                if failed_row(entry.row):
                    continue
                if entry.error:
                    raise RuntimeError(entry.error)
                parent_id = entry.task_dir.name
                path, row = str(entry.path), resolved_rows[str(entry.path)]
                if parent_id in destinations:
                    source, destination = destinations[parent_id]
                    path = _relocated(path, source, destination)
                    row = _relocated(row, source, destination)
                rows.append({"parent_id": parent_id, "path": path, "row": row})
            expected = sum(reference.get("settings", settings)["replicas"] for reference in references)
            if len(rows) != result.succeeded or result.succeeded + result.failed != expected:
                raise RuntimeError(f"Partial diffusion accounting differs from {expected} replicas")
            return {"candidates": rows, "failed": result.failed, "model_loads": self.loads - before}
        finally:
            # Release CUDA weight storage before the pipeline closes the phase
            # owner and releases its CUDA context.
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
