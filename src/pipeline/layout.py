"""Paths for task-first runs, with unchanged paths for existing runs."""
from __future__ import annotations

from pathlib import Path


TASK_LAYOUT = "task"


def task_layout(config):
    # Direct stage callers and runs saved before this layout retain their paths.
    return config.get("_layout") == TASK_LAYOUT


def stage_folder(config, task, number, *, final=False):
    root = Path(config["output_dir"])
    if task_layout(config):
        return root / task / ("final" if final else f"round_{number}")
    return root / ("final" if final else f"recycle_{number:03d}") / task


def design_folder(config, task):
    root = Path(config["output_dir"])
    return root / task / "design" if task_layout(config) else root / "generation" / task


def run_work(config):
    root = Path(config["output_dir"])
    return root / ".work" if task_layout(config) else root


def metadata_folder(folder, task):
    """Locate native metadata without changing the structural input boundary."""
    folder = Path(folder)
    hidden = folder / ".work" / task
    return hidden if (hidden / "resolved_config.yaml").is_file() else folder


def stage_state(config, folder):
    folder = Path(folder)
    return folder / ".work" / "state.json" if task_layout(config) else folder / ".state.json"


def refold_folder(config, folder, *, final=False):
    folder = Path(folder)
    if task_layout(config):
        return folder / ".work" / "refolds" if final else folder / "refold"
    return folder / "inputs" / "refolds"
