"""Locate native chemistry/motif metadata in ordinary and pipeline task leaves."""
from pathlib import Path
from os import PathLike
from collections.abc import Mapping


def snapshot_path(task):
    task = Path(task)
    direct = task / "resolved_config.yaml"
    if direct.is_file():
        return direct
    # Pipeline stages keep their one-task native snapshot beside private inputs.
    for name in dict.fromkeys((task.parent.name, task.name)):
        path = task / ".work" / name / "resolved_config.yaml"
        if path.is_file():
            return path
    return None


def task_config(config, task):
    tasks = config.get("tasks", {})
    if not isinstance(tasks, Mapping):
        return None
    result = tasks.get(Path(task).name)
    if result is None and len(tasks) == 1:
        result = next(iter(tasks.values()))
    return result if isinstance(result, Mapping) else None


def authoring_dir(input_record):
    """Locate immutable task authoring without changing input/output ownership.

    Pipeline rows may refer to a seed-specific snapshot while several seeds
    share one structural task directory. Paths remain accepted by chemistry
    readers that already received their metadata directory explicitly.
    """
    if isinstance(input_record, (str, PathLike)):
        return Path(input_record)
    if isinstance(input_record, Mapping):
        row, task = input_record.get("row", {}), input_record.get("task_dir")
    else:
        row, task = getattr(input_record, "row", {}), getattr(input_record, "task_dir", None)
    pointer = row.get("_pipeline_metadata") if isinstance(row, Mapping) else None
    return Path(pointer or task) if pointer or task is not None else None
