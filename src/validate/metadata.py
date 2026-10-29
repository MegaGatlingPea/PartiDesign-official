"""Locate native chemistry/motif metadata in ordinary and pipeline task leaves."""
from pathlib import Path
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
