"""Task-wide final-score ordering and publication of owned structure filenames.

Callers supply scores for the actual retained structures and own publication of
their result tables.  This module never loads or calls a scoring model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


_SCORE_SUFFIX = re.compile(r"(?:_score_rank\d+)+$")
_SAMPLE_PREFIX = re.compile(r"^sample(\d+)(?:_|\.|$)")


def strip_score_rank(filename: str) -> str:
    """Remove only the final-score suffix, preserving sample/search identity."""
    if not filename:
        return ""
    path = Path(str(filename))
    return str(path.with_name(_SCORE_SUFFIX.sub("", path.stem) + path.suffix))


def _finite_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _number_key(value: Any) -> tuple[int, int | str]:
    text = "" if value is None else str(value).strip()
    if not isinstance(value, bool) and re.fullmatch(r"[+-]?\d+", text):
        return 0, int(text)
    if not isinstance(value, bool):
        try:
            number = float(value)
            if math.isfinite(number) and number.is_integer():
                return 0, int(number)
        except (TypeError, ValueError):
            pass
    return 1, text


def _identity_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    filename = strip_score_rank(str(row.get("filename") or ""))
    sample = row.get("base_sample_id", row.get("sample_id"))
    if sample is None or str(sample).strip() == "":
        match = _SAMPLE_PREFIX.match(Path(filename).name)
        sample = match.group(1) if match else ""
    reward_rank = row.get("global_rank", row.get("reward_rank", row.get("rank", "")))
    return _number_key(sample), _number_key(reward_rank), filename


def rank_score_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return final-score ranks; ties keep sample, reward rank, then filename.

    Missing or non-finite scores stay blank and follow all ranked rows.  Existing
    sample/reward fields are retained, and input rows are never mutated.
    """
    prepared = []
    for row in rows:
        item = dict(row)
        score = _finite_score(item.get("final_partiscore"))
        item["final_partiscore"] = "" if score is None else score
        item["score_rank"] = ""
        prepared.append(item)
    prepared.sort(key=lambda row: (
        row["final_partiscore"] == "",
        -row["final_partiscore"] if row["final_partiscore"] != "" else 0,
        *_identity_key(row),
    ))
    for rank, row in enumerate(prepared, 1):
        if row["final_partiscore"] != "":
            row["score_rank"] = rank
    return prepared


def _update_references(row: Mapping[str, Any], rename_map: Mapping[str, str]) -> dict[str, Any]:
    item = dict(row)
    for field in ("filename", "refined_filename"):
        value = item.get(field)
        if value is not None and str(value):
            path = Path(str(value))
            if path.name in rename_map:
                updated = path.with_name(rename_map[path.name])
                item[field] = updated if isinstance(value, Path) else str(updated)
    return item


def _rename_owned_files(task_dir: Path, rename_map: Mapping[str, str]) -> None:
    """Publish a rename batch without overwriting files, rolling back failures."""
    if not rename_map:
        return
    task_dir = Path(task_dir)
    if task_dir.is_symlink() or not task_dir.is_dir():
        raise ValueError(f"score ranking requires a real task directory: {task_dir}")
    sources = set(rename_map)
    if len(set(rename_map.values())) != len(rename_map):
        raise ValueError("score ranking has duplicate destination filenames")
    for source, destination in rename_map.items():
        for filename in (source, destination):
            if Path(filename).name != filename or Path(filename).suffix.lower() not in {".cif", ".pdb"}:
                raise ValueError(f"invalid score-ranking filename: {filename!r}")
        path = task_dir / source
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"score-ranking source is not a regular file: {path}")
        target = task_dir / destination
        if (target.exists() or target.is_symlink()) and destination not in sources:
            raise FileExistsError(f"score-ranking destination already exists: {target}")

    staged: dict[str, Path] = {}
    published: list[str] = []
    temporaries: list[Path] = []
    safe_cleanup = False
    try:
        for source in rename_map:
            descriptor, temporary = tempfile.mkstemp(prefix=".score-rank-", dir=task_dir)
            os.close(descriptor)
            temporary_path = Path(temporary)
            temporaries.append(temporary_path)
            os.replace(task_dir / source, temporary_path)
            staged[source] = temporary_path
        for source, temporary in staged.items():
            # An exclusive link also protects against a destination appearing
            # after preflight. All files are on the same filesystem.
            os.link(temporary, task_dir / rename_map[source])
            published.append(source)
        for temporary in temporaries:
            temporary.unlink()
        safe_cleanup = True
    except BaseException as error:
        try:
            # A cleanup failure can occur after a few temporary links vanished.
            for source in published:
                temporary = staged[source]
                if not temporary.exists():
                    os.link(task_dir / rename_map[source], temporary)
            for source in published:
                (task_dir / rename_map[source]).unlink()
            for source, temporary in staged.items():
                os.link(temporary, task_dir / source)
                temporary.unlink()
            safe_cleanup = True
        except BaseException as rollback_error:
            raise RuntimeError(
                f"score-ranking publication failed ({error}); "
                f"rollback failed ({rollback_error}); retained staging: {temporaries}"
            ) from error
        raise
    finally:
        if safe_cleanup:
            for temporary in temporaries:
                temporary.unlink(missing_ok=True)


def finalize_score_ranking(
    task_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    interfaces: Sequence[Mapping[str, Any]],
    *,
    rename: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Rank one complete task and update its final structure references.

    ``rename=False`` returns the same publication plan without filesystem work.
    Recovery callers decide whether an output is managed before invoking this
    function; legacy tasks are not discovered or renamed automatically.
    """
    ranked = rank_score_rows(rows)
    rename_map: dict[str, str] = {}
    filenames: set[str] = set()
    for row in ranked:
        filename = str(row.get("filename") or "")
        if not filename:
            if row["score_rank"] != "":
                raise ValueError("a ranked score row has no structure filename")
            continue
        if filename in filenames:
            raise ValueError(f"duplicate score-ranking filename: {filename!r}")
        filenames.add(filename)
        original = strip_score_rank(filename)
        path = Path(original)
        destination = (
            f"{path.stem}_score_rank{row['score_rank']}{path.suffix}"
            if row["score_rank"] != "" else original
        )
        if filename != destination:
            rename_map[filename] = destination
    updated_rows = [_update_references(row, rename_map) for row in ranked]
    updated_interfaces = [_update_references(row, rename_map) for row in interfaces]
    if rename:
        _rename_owned_files(task_dir, rename_map)
    return updated_rows, updated_interfaces, rename_map


__all__ = ["finalize_score_ranking", "rank_score_rows", "strip_score_rank"]
