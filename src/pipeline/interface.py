"""Readable public operations lowered into the existing workflow schema."""
from __future__ import annotations
import copy
from src.validate.config import DEFAULTS as VALIDATE_DEFAULTS
_VALIDATION_FIELDS = set(VALIDATE_DEFAULTS) - {"methods", "criteria", "unfiltered_success", "rerun", "rerun_refold"}


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return copy.deepcopy(value)


def _put(target, key, value, label):
    if not isinstance(target, dict) or key in target:
        raise ValueError(f"{label}: configure {key} only once")
    target[key] = value


def _validation(value, label):
    result = _object(value, label)
    direct = {key: result.pop(key) for key in list(result) if key in _VALIDATION_FIELDS}
    if direct:
        validation = result.setdefault("validation", {})
        if not isinstance(validation, dict) or set(validation) & direct.keys():
            raise ValueError(f"{label}: duplicate validation settings")
        validation.update(direct)
    return result


def iteration_overlay(value, label="iteration"):
    result = _object(value, label)
    if "mpnn" in result:
        mpnn = result.pop("mpnn")
        mpnn = {"count": mpnn} if type(mpnn) is int else _object(mpnn, f"{label}.mpnn")
        _put(result.setdefault("sequence", {}), "mpnn", mpnn, label)
    if "include_gt" in result:
        _put(result.setdefault("sequence", {}), "include_gt", result.pop("include_gt"), label)
    for key in ("partial", "relax"):
        if key in result:
            setting = result.pop(key)
            setting = {"enabled": setting} if type(setting) is bool else _object(setting, f"{label}.{key}")
            _put(result.setdefault("transition", {}), key, setting, label)
    if "refold" in result:
        refold = result["refold"]
        result["refold"] = {"method": refold} if isinstance(refold, str) else _validation(refold, f"{label}.refold")
    if "filter" in result:
        filtering = result["filter"]
        if isinstance(filtering, str): result["filter"] = {"criteria": [filtering]}
        elif isinstance(filtering, list): result["filter"] = {"criteria": filtering}
    return result


def workflow_overlay(value, label="pipeline"):
    result = _object(value, label)
    if "iteration" in result:
        result["iteration"] = iteration_overlay(result["iteration"], f"{label}.iteration")
    if "rounds" in result:
        result["rounds"] = {key: iteration_overlay(row, f"{label}.rounds.{key}")
                            for key, row in _object(result["rounds"], f"{label}.rounds").items()}
    if "final" in result:
        final = _validation(result["final"], f"{label}.final")
        if "max_pass" in final:
            _put(final.setdefault("pool", {}), "max_pass", final.pop("max_pass"), label)
        result["final"] = final
    if "tasks" in result:
        result["tasks"] = {name: workflow_overlay(row, f"{label}.tasks.{name}")
                           for name, row in _object(result["tasks"], f"{label}.tasks").items()}
    return result


def unpack_document(document):
    """A plain task file remains valid; a combined file has pipeline + tasks."""
    if "pipeline" in document and "tasks" in document:
        unknown = set(document) - {"pipeline", "tasks"}
        if unknown:
            raise ValueError(f"Unknown pipeline document sections: {sorted(unknown)}")
        tasks = _object(document["tasks"], "tasks")
        overlay = workflow_overlay(document["pipeline"])
    else:
        tasks, overlay = copy.deepcopy(document), {}
    for name, task in tasks.items():
        if not isinstance(task, dict):
            raise ValueError(f"task {name} must be a mapping")
        if "pipeline" in task:
            configured = overlay.setdefault("tasks", {})
            if name in configured:
                raise ValueError(f"task {name}: configure pipeline overrides in one place")
            configured[name] = workflow_overlay(task.pop("pipeline"), f"task {name}.pipeline")
    return tasks, overlay
