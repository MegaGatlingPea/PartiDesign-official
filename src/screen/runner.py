"""Screen orchestration over the same resident backends used by pipeline."""
from __future__ import annotations
from collections import Counter, defaultdict
import copy
import csv
import json
import math
import os
from pathlib import Path
import tempfile

from .config import affinity_eligible, ranking_for, request_for
from src.validate.resources import worker_capacity, worker_resources

_GROUPS = {"esmfold2-fast": "esmfold2", "esmfold2-standard": "esmfold2",
           "af3": "af3", "af3-la": "af3", "mol1": "mol1", "mol1-constraint": "mol1_constraint", "boltz2": "boltz2"}
_AFFINITY_KEYS = ("affinity", "affinity_samples", "affinity_steps")


def _write_json(path, value):
    from src.pipeline.backends import _encode_evidence
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(_encode_evidence(value), stream, allow_nan=False, sort_keys=True)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path):
    from src.pipeline.backends import _decode
    try:
        return _decode(json.loads(Path(path).read_text()))
    except FileNotFoundError:
        return None


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def method_config(task, method, args=None):
    from src.validate.config import DEFAULTS, config_for_method
    result = copy.deepcopy(DEFAULTS)
    result["methods"] = [method]
    result[_GROUPS[method]]["samples"] = task["samples"]
    if args is not None:
        for field in ("worker_gpus", "worker_cpus"):
            value = getattr(args, field, None)
            if value is not None:
                result[field] = value
    return config_for_method(result, method)


def _consumer_settings(chain, method):
    settings = copy.deepcopy(chain["msa"])
    if method == "esmfold2-fast":
        settings.update(mode="off", templates=False, refresh=False, cache_profile=None)
    elif method == "esmfold2-standard":
        settings["templates"] = False
    return settings


def prepare_contexts(task, method, repo, caches, references, usage):
    """Resolve each shared query before reserving a prediction GPU worker."""
    from src.validate.af3_msa import AF3MSACache, msa_query_enabled
    from src.validate.msa_options import query_identity
    contexts = copy.deepcopy(task["contexts"])
    for chain in contexts:
        settings = _consumer_settings(chain, method)
        chain["msa"] = settings
        eligible = msa_query_enabled(settings, chain["modality"])
        if method in {"esmfold2-fast", "esmfold2-standard", "boltz2"}:
            eligible = eligible and chain["modality"] == "protein"
        if not eligible:
            continue
        key = query_identity(settings, chain["modality"], chain["sequence"])
        if key not in references:
            profile = json.dumps(settings, sort_keys=True)
            if profile not in caches:
                caches[profile] = AF3MSACache(repo, settings)
            references[key] = caches[profile].ensure(chain["modality"], chain["sequence"])
        reference = copy.deepcopy(references[key])
        chain["msa_reference"] = reference
        usage.append({"task": task["name"], "method": method, "chain": chain["id"],
                      **{k: reference.get(k) for k in ("entry_id", "profile_id", "generation", "source", "msa_depth", "template_count", "unpaired_msa_path")}})
    return contexts


def make_plan(tasks, args):
    jobs = []
    for task in tasks.values():
        good = sum(not row.get("error") for row in task["rows"])
        for method in task["methods"]:
            config = method_config(task, method, args)
            gpu, cpu = worker_resources(config, method)
            requested = {c["id"]: c["pocket"] for c in task["contexts"] if c.get("pocket")}
            jobs.append(dict(task=task["name"], method=method, candidates=len(task["rows"]),
                             valid_candidates=good, invalid_candidates=len(task["rows"])-good,
                             samples=task["samples"], planned_structures=good*task["samples"],
                             worker_gpus=gpu, worker_cpus=cpu, pocket=requested,
                             pocket_will_apply=bool(requested) and method in {"boltz2", "mol1-constraint"},
                             msa=[dict(chain=c["id"], modality=c["modality"],
                                       **{k:_consumer_settings(c,method)[k] for k in ("mode","source","templates","max_template_date")})
                                  for c in task["contexts"]]))
    return {"command": "screen", "tasks": len(tasks), "jobs": jobs,
            "representative": "native confidence before candidate ranking", "pocket_filter": False}


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _prediction_complete(prediction):
    if not isinstance(prediction, dict) or not _finite(prediction.get("native_ranking_score")):
        return False
    for field in ("structure_path", "raw_confidence_path", "raw_confidence_npz_path"):
        path = prediction.get(field)
        if field == "structure_path" and not path:
            return False
        if path and not Path(path).is_file():
            return False
    return True

def _affinity_complete(prediction):
    if not prediction or not all(_finite(prediction.get("confidence", {}).get("boltz2_"+name)) for name in (
            "affinity_pred_value", "affinity_probability_binary", "affinity_pred_value1",
            "affinity_probability_binary1", "affinity_pred_value2", "affinity_probability_binary2")):
        return False
    metadata = prediction.get("metadata", {})
    for name in ("boltz2_affinity_samples_path", "boltz2_affinity_scored_pose_path"):
        path = metadata.get(name)
        if not path or not Path(path).is_file() or not Path(path).stat().st_size:
            return False
    for path in metadata.get("affinity_required_artifacts", []):
        if not Path(path).is_file():
            return False
    return True


def representative(predictions):
    complete = [p for p in predictions if _prediction_complete(p)]
    if not complete:
        raise ValueError("No complete prediction has a finite native ranking score")
    return max(complete, key=lambda p: (p["native_ranking_score"], -int(p["sample"])))


class ScreenWorker:
    """One resident shared backend inside one CPU/GPU-reserved Ray actor."""
    def __init__(self, repo, method, config, model_root):
        self.repo, self.method = Path(repo), method
        self.config = copy.deepcopy(config)
        self.backend = None
        from partidesign.paths import ENV_MODEL_DIR
        os.environ[ENV_MODEL_DIR] = str(model_root)

    def _models(self):
        if self.backend is None:
            from src.validate.backend import create_backend
            self.backend = create_backend(self.repo, self.method, self.config, "cuda:0")
        return self.backend

    def process(self, job):
        path = Path(job["state_path"])
        previous = _read_json(path)
        if previous and previous["identity"] != job["identity"]:
            raise ValueError("Screen request changed; use a new output directory")
        record = previous or {"identity": job["identity"], "request": job["request"], "status": "pending", "predictions": []}
        predictions = [p for p in record["predictions"] if _prediction_complete(p)]
        record["predictions"] = predictions
        requested = int(self.config[_GROUPS[self.method]]["samples"])
        try:
            if len(predictions) < requested:
                backend = self._models()
                completed = tuple(f"complex_sample{p['sample']}" for p in predictions)
                for prediction in backend.predict_sequences(job["request"], job["output"], completed_keys=completed):
                    if not _prediction_complete(prediction):
                        raise ValueError("Native sequence prediction artifacts/ranking are incomplete")
                    index = int(prediction["sample"])
                    predictions = [p for p in predictions if int(p["sample"]) != index] + [prediction]
                    record.update(predictions=predictions, status="partial")
                    _write_json(path, record)
            if {int(p["sample"]) for p in predictions} != set(range(requested)):
                raise ValueError("Native backend did not return all requested structure samples")
            selected = representative(predictions)
            if self.method == "boltz2" and affinity_eligible(job["request"]):
                if not _affinity_complete(selected):
                    selected = self._models().add_affinity(selected)
                    predictions = [selected if int(p["sample"]) == int(selected["sample"]) else p for p in predictions]
                    if not _affinity_complete(selected):
                        raise ValueError("Native affinity output or internally scored pose evidence is incomplete")
            record.update(status="complete", predictions=predictions,
                          representative_sample=int(selected["sample"]), error=None)
        except Exception as exc:
            record.update(status="partial" if predictions else "failed", predictions=predictions,
                          error=f"{type(exc).__name__}: {exc}")
        _write_json(path, record)
        return {"state_path": str(path), "status": record["status"], "error": record.get("error")}

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None


def _signature(value):
    """Keep scientific inputs; CPU reservation/search access policy is not one."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"worker_gpus", "worker_cpus", "refresh", "search_workers", "cpu_budget",
                       "jackhmmer_n_cpu", "nhmmer_n_cpu", "timeout_seconds"}:
                continue
            if key == "mode" and item in {"auto", "cache-only"}:
                item = "enabled"
            result[key] = _signature(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_signature(item) for item in value]
    return value


def _record_complete(record, samples, affinity=False):
    if not record or record.get("status") != "complete":
        return False
    predictions = record.get("predictions", [])
    if {int(p["sample"]) for p in predictions} != set(range(samples)) or not all(map(_prediction_complete, predictions)):
        return False
    if affinity:
        selected = next((p for p in predictions if int(p["sample"]) == record.get("representative_sample")), None)
        if not _affinity_complete(selected):
            return False
    return True


def _metric_values(prediction):
    values = dict(prediction.get("metrics", {}))
    for key, value in prediction.get("confidence", {}).items():
        if _finite(value) and (key.startswith("boltz2_") or key in {"ptm", "iptm", "ranking_score"}):
            values[key] = value
    return {key: value for key, value in values.items() if value is None or _finite(value)}


def _rank(rows, ranking):
    valid = [row for row in rows if row.get("status") == "complete" and _finite(row.get(ranking["metric"]))]
    reverse = ranking["order"] == "desc"
    valid.sort(key=lambda row: ((-1 if reverse else 1)*row[ranking["metric"]], row["id"]))
    numbered = {row["id"]: rank for rank, row in enumerate(valid, 1)}
    remainder = sorted((row for row in rows if row["id"] not in numbered), key=lambda row: row["id"])
    return [dict(row, rank=numbered.get(row["id"], ""), ranking_metric=ranking["metric"],
                 ranking_value=row.get(ranking["metric"])) for row in [*valid, *remainder]]


def rebuild_reports(task, folder):
    rows, sample_rows = [], []
    for row in task["rows"]:
        for method in task["methods"]:
            state = _read_json(folder/".state"/f"row_{row['row_index']:06d}"/f"{method}.json")
            report = {"task": task["name"], "id": row["id"], "modality": row["modality"],
                      "method": method, "status": state.get("status", "pending") if state else "pending",
                      "representative_sample": "", "structure_path": "", "pocket_applied": False,
                      "reason": (state or {}).get("error") or row.get("error") or ""}
            if state:
                predictions = state.get("predictions", [])
                for prediction in predictions:
                    sample_rows.append(dict(task=task["name"], id=row["id"], modality=row["modality"], method=method,
                                            sample=prediction["sample"], native_ranking_score=prediction.get("native_ranking_score"),
                                            structure_path=prediction.get("structure_path"), **_metric_values(prediction)))
                selected = next((p for p in predictions if int(p["sample"]) == state.get("representative_sample")), None)
                if selected:
                    report.update(_metric_values(selected))
                    report.update(representative_sample=selected["sample"], structure_path=selected["structure_path"],
                                  native_ranking_score=selected["native_ranking_score"])
                    metadata = selected.get("metadata", {})
                    pocket = metadata.get("pocket", metadata.get("pocket_state", metadata.get("pocket_conditioning", {})))
                    report["pocket_applied"] = bool(pocket.get("applied", False)) if isinstance(pocket, dict) else False
                    if isinstance(pocket, dict) and pocket.get("requested") and not report["pocket_applied"]:
                        report["reason"] = report["reason"] or pocket.get("reason") or "Pocket conditioning is unsupported by this method"
            rows.append(report)
    _write_csv(folder/"scores.csv", rows)
    _write_csv(folder/"samples.csv", sample_rows)
    modalities = list(dict.fromkeys(row["modality"] for row in task["rows"]))
    for method in task["methods"]:
        for modality in modalities:
            selected = [row for row in rows if row["method"] == method and row["modality"] == modality]
            _write_csv(folder/"rankings"/method/f"{modality}.csv", _rank(selected, ranking_for(task, modality, method)))
    if "boltz2" in task["methods"]:
        selected = [row for row in rows if row["method"] == "boltz2" and row["modality"] == "ligand"]
        if selected:
            _write_csv(folder/"ligand_ranking.csv", _rank(selected, ranking_for(task, "ligand", "boltz2")))
    summary = dict(task=task["name"], candidates=len(task["rows"]), requested_jobs=len(rows),
                   requested_structures=len(task["rows"])*len(task["methods"])*task["samples"],
                   statuses=dict(Counter(row["status"] for row in rows)), pocket_filter=False,
                   metric_units={"design_plddt": "0..1", "design_ipsae": "0..1", "binder_iptm": "0..1",
                                 "ligand_iptm": "0..1", "boltz2_affinity_pred_value": "log10(IC50 in micromolar)"})
    _write_json(folder/"summary.json", summary)
    return summary


def run(tasks, args, repo, model_root):
    """Prepare shared context inputs, then schedule resident native workers."""
    from src.validate.storage import locked
    root = Path(args.out).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ValueError("Screen output already exists; use --resume or a new --out")
    root.mkdir(parents=True, exist_ok=True)
    groups, work, usage = {}, [], []
    caches, references = {}, {}
    with locked(root/".screen.lock", blocking=False):
        for task in tasks.values():
            folder = root/task["name"]
            manifest = folder/".screen.json"
            identity = _signature(task)
            previous = _read_json(manifest)
            if previous and previous.get("identity") != identity:
                raise ValueError(f"Screen task {task['name']} changed; use a new output directory")
            _write_json(manifest, {"kind": "partidesign.screen.v1", "identity": identity, "task": task})
            _write_csv(folder/"candidates.csv", [{k:row[k] for k in ("id","modality","seq","error")} for row in task["rows"]])
            # A template-enabled search can also serve an MSA-only consumer.
            # Resolve that shared superset first rather than searching twice.
            methods = sorted(task["methods"], key=lambda method: not any(
                c["modality"] == "protein" and _consumer_settings(c, method)["mode"] != "off"
                and _consumer_settings(c, method)["templates"] for c in task["contexts"]))
            for method in methods:
                config = method_config(task, method, args)
                gpu, cpu = worker_resources(config, method)
                group_key = json.dumps({"method": method, "config": config}, sort_keys=True)
                contexts, preparation_error = None, None
                for row in task["rows"]:
                    state_path = folder/".state"/f"row_{row['row_index']:06d}"/f"{method}.json"
                    output = folder/"predictions"/f"candidate_{row['row_index']:06d}"/method
                    if row.get("error"):
                        _write_json(state_path, dict(identity={"row":row}, status="failed", predictions=[], error=row["error"]))
                        continue
                    logical = request_for(task, row)
                    request_identity = _signature({"request": logical, "config": config})
                    old = _read_json(state_path)
                    if old and old.get("identity") != request_identity:
                        raise ValueError(f"Screen request changed for {task['name']}/{row['id']}/{method}")
                    if _record_complete(old, task["samples"], method=="boltz2" and affinity_eligible(logical)):
                        continue
                    if old and old.get("request"):
                        request = old["request"]
                    else:
                        if contexts is None and preparation_error is None:
                            try:
                                contexts = prepare_contexts(task, method, repo, caches, references, usage)
                            except Exception as exc:
                                preparation_error = f"context preparation failed: {type(exc).__name__}: {exc}"
                        if preparation_error:
                            _write_json(state_path, dict(identity=request_identity, status="failed", predictions=[], error=preparation_error))
                            continue
                        request = copy.deepcopy(logical)
                        request["chains"] = [*copy.deepcopy(contexts), request["chains"][-1]]
                    job = dict(identity=request_identity, request=request, state_path=str(state_path), output=str(output))
                    group = groups.setdefault(group_key, dict(method=method, config=config, gpu=gpu, cpu=cpu, jobs=[]))
                    group["jobs"].append(job)
            rebuild_reports(task, folder)
            work.append((task,folder))
        if usage:
            for task, folder in work:
                selected = [row for row in usage if row["task"] == task["name"]]
                if selected:
                    _write_csv(folder/"msa.csv", selected)
        if groups:
            import ray
            import torch
            if ray.is_initialized():
                raise RuntimeError("screen owns one Ray runtime; another runtime is already active")
            visible = torch.cuda.device_count()
            if not visible:
                raise RuntimeError("No CUDA GPUs are visible for screen")
            available_cpus = len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count() or 1
            ray.init(address="local", num_gpus=visible, num_cpus=available_cpus, include_dashboard=False, log_to_driver=True)
            try:
                for group in groups.values():
                    count = worker_capacity(ray.cluster_resources(), group["gpu"], group["cpu"], limit=len(group["jobs"]))
                    if count < 1:
                        raise RuntimeError("Insufficient CPU/GPU resources for a screen worker")
                    threads = str(max(1,int(group["cpu"])))
                    environment = {"PARTIDESIGN_MODEL_DIR":str(model_root), "XLA_PYTHON_CLIENT_PREALLOCATE":"false",
                                   "OMP_NUM_THREADS":threads,"MKL_NUM_THREADS":threads,"OPENBLAS_NUM_THREADS":threads}
                    remote = ray.remote(num_gpus=group["gpu"], num_cpus=group["cpu"], max_restarts=0,
                                        runtime_env={"env_vars":environment})(ScreenWorker)
                    actors = [remote.remote(str(repo),group["method"],group["config"],str(model_root)) for _ in range(count)]
                    print(f"Screen {group['method']}: {count} resident workers, {group['gpu']:g} GPU / {group['cpu']:g} CPU each",flush=True)
                    queue = iter(group["jobs"])
                    pending = {}
                    def dispatch(actor):
                        job = next(queue,None)
                        if job is not None:
                            pending[actor.process.remote(job)] = (actor,job)
                    try:
                        for actor in actors:
                            dispatch(actor)
                        while pending:
                            ready,_ = ray.wait(list(pending),num_returns=1)
                            for reference in ready:
                                actor,job = pending.pop(reference)
                                try:
                                    result = ray.get(reference)
                                    print(f"  {group['method']}: {result['status']} ({Path(job['output']).parent.name})",flush=True)
                                except Exception as exc:
                                    record = _read_json(job["state_path"]) or dict(identity=job["identity"],request=job["request"],predictions=[])
                                    record.update(status="failed",error=f"worker failed: {type(exc).__name__}: {exc}")
                                    _write_json(job["state_path"],record)
                                dispatch(actor)
                    finally:
                        from src.validate.__main__ import _close_validation_actors
                        _close_validation_actors(ray,actors)
                        for actor in actors:
                            ray.kill(actor,no_restart=True)
            finally:
                ray.shutdown()
                for task,folder in work:
                    rebuild_reports(task,folder)
        summaries = [rebuild_reports(task,folder) for task,folder in work]
        complete = sum(summary["statuses"].get("complete",0) for summary in summaries)
        total = sum(summary["requested_jobs"] for summary in summaries)
        print(f"Screen complete: {complete}/{total} candidate-method jobs; results: {root}",flush=True)
        return 0 if complete == total else 2
