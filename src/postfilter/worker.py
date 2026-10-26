"""Persistent CPU workers that own a sample through final publication."""
from __future__ import annotations

import copy
import math
import numbers
import os
from pathlib import Path
import time

from .criteria import evaluate, profile_for_kind
from .schema import Candidate
from .storage import atomic_json


def candidate_paths(identity: str, output_root: str | Path) -> tuple[Path, Path]:
    relative = Path(identity)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in identity.split("/")):
        raise ValueError("invalid candidate identity")
    root = Path(output_root).expanduser().absolute()
    return root / "structures" / (identity + ".cif"), root / "records" / (identity + ".json")


def failed_record(candidate: Candidate, error: str, elapsed: float = 0.0) -> dict:
    return {
        "version": 2, "identity": candidate.identity, "candidate": candidate.to_dict(),
        "status": "failed", "geometry_status": "not_run", "relaxed_path": None,
        "metrics": {}, "errors": {}, "decisions": {}, "error": error,
        "timings": {"total": elapsed},
    }


def thread_environment(threads: int) -> dict[str, str]:
    return {name: str(threads) for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    )}


def required_tools(required: set[str]) -> set[str]:
    tools = set()
    if "bc1_interface_unsat_hbonds" in required:
        tools.add("dalphaball")
    if required & {"bc1_binder_helix_percent", "bc1_binder_sheet_percent", "bc1_binder_loop_percent"}:
        tools.add("dssp")
    return tools


def computation_settings(config: dict, bundle: dict, required: set[str]) -> dict:
    return {
        "seed": config["seed"], "threads": config["threads"],
        "scorefunction": bundle["protocol"]["scorefunction"],
        "relax": copy.deepcopy(bundle["protocol"]["relax"]),
        **{name: config[name] for name in sorted(required_tools(required))},
    }


def _scalar_details(value: dict) -> dict:
    return {
        str(key): item for key, item in value.items()
        if item is None or type(item) in (str, bool, int)
        or (type(item) is float and math.isfinite(item))
    }


class PostfilterWorker:
    """One initialized SDK and its original in-memory random stream per actor."""

    def __init__(self, config: dict, bundle: dict):
        threads = config["threads"]
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        os.environ.update(thread_environment(threads))
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.config, self.bundle = copy.deepcopy(config), copy.deepcopy(bundle)
        self.native_config = {**self.config, **copy.deepcopy(bundle["protocol"])}
        from . import rosetta

        self.pr = rosetta.initialize(self.native_config)
        self._rng = self.pr.rosetta.numeric.random.rg()
        stream = self.pr.rosetta.std.ostringstream()
        self._rng.saveState(stream)
        self._initial_rng_state = stream.str()
        if not isinstance(self._initial_rng_state, str) or not self._initial_rng_state:
            raise RuntimeError("native SDK did not expose an in-memory RNG state")
        # Validate the restore binding during startup, before accepting samples.
        self._reset_rng()
        self._processed = self._completed = self._failed = 0

    def _reset_rng(self) -> None:
        stream = self.pr.rosetta.std.istringstream(self._initial_rng_state)
        self._rng.restoreState(stream)

    def ready(self) -> dict:
        return self.stats()

    def stats(self) -> dict:
        return {
            "pid": os.getpid(), "initializations": 1, "processed": self._processed,
            "completed": self._completed, "failed": self._failed,
        }

    def process(self, candidate_dict: dict, output_root: str) -> dict:
        from . import metrics_bc2, pose_mapping, rosetta

        started = time.monotonic()
        candidate = Candidate.from_dict(candidate_dict)
        destination, record_path = candidate_paths(candidate.identity, output_root)
        if destination.resolve() == Path(candidate.refold_path).expanduser().resolve():
            raise ValueError("relaxed output cannot overwrite the source structure")
        names, required = profile_for_kind(candidate.binder_kind, self.bundle)
        record = failed_record(candidate, "")
        record["settings"] = computation_settings(self.config, self.bundle, required)
        timings = record["timings"]
        self._processed += 1
        try:
            self._reset_rng()
            step = time.monotonic()
            source = pose_mapping.read_candidate_source(candidate)
            timings["parse"] = time.monotonic() - step
            names, required = profile_for_kind(source.binder_kind, self.bundle)
            record["details"] = {"binder_kind": source.binder_kind}
            record["settings"] = computation_settings(self.config, self.bundle, required)

            step = time.monotonic()
            pose, mapping = pose_mapping.load_pose_mapping(source, self.pr)
            rosetta.configure_partner_foldtree(source, pose, mapping, self.pr)
            timings["pose"] = time.monotonic() - step
            step = time.monotonic()
            pose = rosetta.relax_pose(source, pose, mapping, self.pr, self.native_config)
            timings["relax"] = time.monotonic() - step

            step = time.monotonic()
            rosetta.seed_metrics(self.pr, self.native_config["seed"])
            native = rosetta.calculate_metrics(
                source, pose, mapping, self.pr, self.native_config, required_metrics=required,
            )
            timings["rosetta_metrics"] = time.monotonic() - step
            step = time.monotonic()
            inputs = metrics_bc2.compute_input_metrics_from_source(source, required, self.native_config)
            timings["input_metrics"] = time.monotonic() - step
            measured = {**native.get("metrics", {}), **inputs.get("metrics", {})}
            errors = {**native.get("errors", {}), **inputs.get("errors", {})}
            for name in sorted(required):
                value = measured.get(name)
                if isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value):
                    record["metrics"][name] = int(value) if isinstance(value, numbers.Integral) else float(value)
                else:
                    record["errors"][name] = str(errors.get(name) or "required metric unavailable")
            for prefix, values in (("native", native), ("input", inputs)):
                record["details"].update({
                    prefix + "_" + key: value
                    for key, value in _scalar_details(values.get("details", {})).items()
                })
            record["decisions"] = {
                name: evaluate(record["metrics"], name, self.bundle["presets"][name]) for name in names
            }
            step = time.monotonic()
            destination.parent.mkdir(parents=True, exist_ok=True)
            pose_mapping.write_relaxed_cif(source, pose, mapping, destination)
            timings["write"] = time.monotonic() - step
            record.update(geometry_status="complete", relaxed_path=str(destination))
            if record["errors"]:
                record["error"] = "; ".join(f"{key}: {value}" for key, value in record["errors"].items())
            else:
                record["status"] = "complete"
        except Exception as exc:
            record["status"], record["error"] = "failed", f"{type(exc).__name__}: {exc}"
            if record["geometry_status"] != "complete":
                record["geometry_status"] = "failed"
            for name in required:
                if name not in record["metrics"]:
                    record["errors"].setdefault(name, record["error"])
            record["decisions"] = {
                name: {"criteria": name, "status": "unavailable", "reasons": [record["error"]]}
                for name in names
            }
        timings["total"] = time.monotonic() - started
        # The small record is the final publication; no successful-path readback.
        atomic_json(record_path, record)
        if record["status"] == "complete":
            self._completed += 1
        else:
            self._failed += 1
        return record
