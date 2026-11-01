"""Multi-method final filtering: union selection and method-labelled best CIFs."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from .layout import stage_folder, stage_state
from .state import atomic_json, write_csv


def _selected_keys(record):
    selection = record["prediction_selection"]
    models = selection.get("selected_models")
    if models:
        return [value["selected_prediction_key"] for value in models.values()]
    key = selection.get("selected_prediction_key")
    if key is not None:
        return [key]
    # Explicit `all` still filters every conformer, but publication has one
    # ranked complex per method (and one per configured AF2 model).
    from .runner import final_prediction_view
    _, ranked = final_prediction_view(record)
    return _selected_keys(dict(record, prediction_selection=ranked)) if ranked.get("selected_prediction_key") else []


def _evaluation_keys(record):
    if record["prediction_selection"]["mode"] == "all":
        return record["expected_predictions"]
    selected = _selected_keys(record)
    if record["method"] == "af2_ig":
        return [item for key in selected for item in
                (key, "monomer_" + key.removeprefix("complex_"))]
    return selected


def threshold_rows(parent, record, definitions, *, method_passed, union_passed):
    """Expose each selected prediction's rule values, including failed methods."""
    from src.validate.filters import OPS
    selected = _evaluation_keys(record)
    predictions = record["predictions"]
    for name, decision in record["decisions"].items():
        preset = definitions[name]
        for rule in preset["rules"]:
            scope = rule.get("scope", "complex")
            if scope == "paired":
                values = [(key.removeprefix("complex_"), record.get("paired_metrics", {}).get(
                    key.removeprefix("complex_"), {}))
                    for key in selected if key.startswith("complex_")]
            else:
                values = [(key, predictions[key].get("metrics", {})) for key in selected
                          if key in predictions and (scope == "prediction" or predictions[key]["protocol"] == scope)]
            if preset["reduction"] == "best_ranked" and values:
                ranking = [metrics.get(name) for _, metrics in values for name in ("iptm", "ptm")]
                if all(isinstance(v, (float, int)) and math.isfinite(v) for v in ranking):
                    values = sorted(values, key=lambda pair: (
                        -0.8 * pair[1]["iptm"] - 0.2 * pair[1]["ptm"], pair[0]))[:1]
            if preset["reduction"] == "all_and_mean" and values:
                numbers = [metrics.get(rule["metric"]) for _, metrics in values]
                valid = all(isinstance(v, (float, int)) and math.isfinite(v) for v in numbers)
                values.append(("mean", {rule["metric"]: sum(numbers) / len(numbers) if valid else None}))
            if not values:
                values = [("", {})]
            for key, metrics in values:
                actual = metrics.get(rule["metric"])
                valid = isinstance(actual, (float, int)) and math.isfinite(actual)
                yield {"candidate_id": parent.candidate_id, "method": record["method"],
                       "prediction_key": key, "criteria": name, "scope": scope,
                       "metric": rule["metric"], "operator": rule["op"], "threshold": rule["value"],
                       "value": actual if valid else "", "passed": valid and OPS[rule["op"]](actual, rule["value"]),
                       "criteria_status": decision["status"], "method_passed": method_passed,
                       "union_passed": union_passed}


def _saved_final(cfg, task_name, root):
    from .runner import _load_stage
    state = stage_state(cfg, root)
    saved = _load_stage(state, folder=root, task=task_name)
    if saved is not None:
        published = json.loads(state.read_text()).get("published_cifs")
        if (published is None or any(not Path(path).is_file() for path in published) or
                not (root / "filter_results.csv").is_file() or
                (published and not (root / "success" / "result.csv").is_file())):
            saved = None
    return None if saved is None else (saved[0], dict(saved[1], _resumed=True))


def prepare_finals(owner, cfg, jobs):
    """Run final predictions by method across tasks, then publish each result.

    Each job provides task_name, stage, parents, previous and optional force.
    The returned task mapping contains the usual (candidates, summary) pairs.
    """
    from .runner import run_stage, _stage_inputs
    results, pending, groups = {}, [], {}
    for job in jobs:
        task_name, stage = job["task_name"], job["stage"]
        parents, previous = job["parents"], job["previous"]
        force = job.get("force", False)
        validators = stage.get("validators")
        if validators:
            root = stage_folder(cfg, task_name, cfg["tasks"][task_name]["recycle"], final=True)
            saved = None if force else _saved_final(cfg, task_name, root)
            if saved is not None:
                results[task_name] = saved
                continue
            root.mkdir(parents=True, exist_ok=True)
            atomic_json(stage_state(cfg, root), {"status": "running"})
            parents = _stage_inputs(parents, root, previous, task=task_name, layout=cfg.get("_layout"))
            previous = root
            pending.append((task_name, stage, parents, previous, force))
        for validator in validators or [stage]:
            groups.setdefault(validator["method"], []).append(
                (task_name, validator, parents, previous, force, bool(validators)))
    for group in groups.values():
        for task_name, stage, parents, previous, force, multi in group:
            options = {}
            if multi:
                options = {"folder_override": previous / ".work" / stage["method"],
                           "predictions_dir": previous / "predictions" / stage["method"],
                           "publish_success": False}
            result = run_stage(owner, cfg, task_name, stage, parents, previous,
                               final=True, force=force, **options)
            if not multi:
                results[task_name] = result
    for task_name, stage, parents, previous, force in pending:
        results[task_name] = run_final(owner, cfg, task_name, stage, parents, previous,
                                      force=force, prepared=True)
    return results


def run_final(owner, cfg, task_name, stage, parents, previous, *, force=False, prepared=False):
    """Evaluate the same pool with every method, then publish the union once."""
    from .runner import run_stage, _load_stage, _publish, _stage_inputs
    validators = stage.get("validators")
    if not validators:
        return run_stage(owner, cfg, task_name, stage, parents, previous, final=True, force=force)
    root = stage_folder(cfg, task_name, cfg["tasks"][task_name]["recycle"], final=True)
    state = stage_state(cfg, root)
    saved = None if force or prepared else _saved_final(cfg, task_name, root)
    if saved is not None:
        return saved
    if not prepared:
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(state, {"status": "running"})
        parents = _stage_inputs(parents, root, previous, task=task_name, layout=cfg.get("_layout"))
        previous = root
    methods = {}
    # Stage by method, keeping the native resource owner hot across candidates.
    for validator in validators:
        method = validator["method"]
        folder = root / ".work" / method
        if prepared:
            result = _load_stage(stage_state(cfg, folder), folder=folder, task=task_name)
            if result is None:
                raise RuntimeError(f"Prepared final stage is incomplete: {task_name}/{method}")
            survivors, summary = result
        else:
            survivors, summary = run_stage(
                owner, cfg, task_name, validator, parents, previous, final=True, force=force,
                folder_override=folder, predictions_dir=root / "predictions" / method,
                publish_success=False)
        methods[method] = {"passed": {c.parent_id for c in survivors}, "summary": summary,
                           "stage": validator}
    all_rows, success_rows, filters, output = [], [], [], []
    success = root / "success"
    # Resume can complete a previously interrupted union publication.
    if success.is_dir():
        import shutil
        shutil.rmtree(success)
    for ordinal, parent in enumerate(parents, 1):
        passed = [method for method, data in methods.items() if parent.candidate_id in data["passed"]]
        row = dict(parent.row, candidate_id=parent.candidate_id, parent_id=parent.parent_id,
                   status="passed" if passed else "filtered", passed_methods=passed)
        paths = {}
        for method, data in methods.items():
            record_path = root / "predictions" / method / f"proposal{ordinal:06d}.json"
            record = json.loads(record_path.read_text())
            method_passed = method in passed
            row[method + "_passed"] = method_passed
            filters.extend(threshold_rows(parent, record, cfg["tasks"][task_name]["presets"],
                                          method_passed=method_passed, union_passed=bool(passed)))
            best_metrics = {}
            for key in _selected_keys(record):
                prediction = record["predictions"][key]
                best_metrics[key] = prediction.get("metrics", {})
                if not passed:
                    continue
                target = success / f"{parent.candidate_id}__{method}__{key}.cif"
                target.parent.mkdir(parents=True, exist_ok=True)
                source = Path(prediction["structure_path"])
                if source.suffix.lower() == ".cif":
                    _publish(source, target)
                else:
                    from src.design.utils.reader import read_structure
                    from src.design.utils.writer import write_structure_atomic
                    write_structure_atomic(read_structure(source), target, entry_id=parent.candidate_id)
                paths.setdefault(method, []).append(str(target))
                success_rows.append({**parent.row, "filename": target.name,
                                     "candidate_id": parent.candidate_id, "method": method,
                                     "prediction_key": key, "method_passed": method_passed,
                                     "union_passed": True, "passed_methods": passed,
                                     "passed_criteria": [name for name, d in record["decisions"].items() if d["status"] == "passed"],
                                     **prediction.get("metrics", {})})
            row[method + "_metrics"] = best_metrics
        row["best_cifs"] = paths
        if passed:
            # Internal candidate accounting needs one path; every method's best
            # is published above and all decisions remain in the result tables.
            path = paths[passed[0]][0]
            row["filename"] = path
            output.append(replace(parent, path=path, row=dict(parent.row, filename=path)))
        else:
            row["filename"] = ""
        all_rows.append(row)
    write_csv(root / "result.csv", all_rows)
    write_csv(root / "selection.csv", [
        {"candidate_id": row["candidate_id"], "status": row["status"],
         "passed_methods": row["passed_methods"], "best_cifs": row["best_cifs"]}
        for row in all_rows])
    write_csv(root / "filter_results.csv", filters)
    if success_rows:
        write_csv(success / "result.csv", success_rows)
    summaries = [data["summary"] for data in methods.values()]
    count = {"task": task_name, "stage": "final", "method": "+".join(methods),
             "parent_count": len(parents), "proposal_count": len(parents),
             "unique_sequence_count": len(parents), "duplicate_count": 0,
             "family_limited_count": 0, "relax_success_count": 0, "relax_failed_count": 0,
             "partial_success_count": 0, "partial_failed_count": 0, "partial_model_loads": 0,
             "passing_count": len(output), "selected_count": len(output),
             "combination": "any", "methods": list(methods),
             "method_passing_counts": {m: len(d["passed"]) for m, d in methods.items()}}
    for field in ("actual_prediction_count", "cache_hits", "unique_request_count"):
        count[field] = sum(summary[field] for summary in summaries)
    atomic_json(state, {"status": "complete", "candidates": [c.saved() for c in output],
                        "archive": [], "summary": count,
                        "published_cifs": [str(success / row["filename"]) for row in success_rows]})
    return output, dict(count, _resumed=False, _archive=[])
