"""Directory seeds expressed with the native task authoring grammar."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

_METADATA = {"filename", "task", "sample_id", "source_status"}


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"invalid JSON constant {value}")


def _cell(text):
    value = text.strip()
    try:
        return json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant)
    except json.JSONDecodeError:
        if value[:1] in {'{', '[', '"'}:
            raise
        return value


def _path(value, base, field):
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError(f"{field} must be a local path")
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def _paths(authored, base):
    if authored.get("input") is not None:
        authored["input"] = _path(authored["input"], base, "input")
    partial = authored.get("partial")
    if isinstance(partial, dict):
        for key in ("reference", "result_csv"):
            if key in partial:
                partial[key] = _path(partial[key], base, f"partial.{key}")
    antigen = authored.get("antigen")
    if isinstance(antigen, dict) and antigen.get("input") is not None:
        antigen["input"] = _path(antigen["input"], base, "antigen.input")
    antibody = authored.get("antibody")
    scaffold = antibody.get("scaffold") if isinstance(antibody, dict) else None
    if isinstance(scaffold, dict) and "path" in scaffold:
        scaffold["path"] = _path(scaffold["path"], base, "antibody.scaffold.path")


def validation_payload(row):
    """Complete only input-stage controls on a copy for the native parser."""
    payload = copy.deepcopy(row["authored"])
    payload.setdefault("sample_num", 1)
    partial = payload.setdefault("partial", {})
    if isinstance(partial, dict):
        # PDB references are converted to CIF during formal import, not preview.
        partial["reference"] = str(Path(row["path"]).with_suffix(".cif"))
        partial.setdefault("fraction", 0.625)
    return payload


def read_annotation(directory, *, invocation_dir=None):
    """Read listed seeds without scanning structures or creating run output.

    Metadata is separate from complete native authoring payloads. Each row is
    validated by the native parser; geometry and resolved ledgers belong to the
    formal input stage, where PDB sources are converted exactly once.
    """
    base = Path(invocation_dir or Path.cwd())
    directory = Path(directory).expanduser()
    directory = (directory if directory.is_absolute() else base / directory).resolve()
    annotation = directory / "annotation.csv"
    rows, seen = [], set()
    from src.design.core.tasks import parse_task
    with annotation.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        try:
            fields = reader.fieldnames
        except csv.Error as exc:
            raise ValueError(f"{annotation}: invalid CSV header: {exc}") from exc
        if not fields or "filename" not in fields:
            raise ValueError(f"{annotation}: filename column is required")
        if any(not key or key != key.strip() for key in fields) or len(fields) != len(set(fields)):
            raise ValueError(f"{annotation}: column names must be nonempty, unique and unpadded")
        while True:
            try:
                raw = next(reader)
            except StopIteration:
                break
            except csv.Error as exc:
                raise ValueError(f"{annotation}: row {reader.line_num}: invalid CSV: {exc}") from exc
            number = reader.line_num
            context = f"{annotation}: row {number}"
            try:
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError("column count differs from header")
                if all(not value.strip() for value in raw.values()):
                    continue
                path = _path(raw["filename"].strip(), directory, "filename")
                if Path(path).suffix.lower() not in {".cif", ".pdb"}:
                    raise ValueError("filename must name a CIF or PDB structure")
                if not Path(path).is_file():
                    raise ValueError(f"structure does not exist: {path}")
                task = raw.get("task", "").strip() or "optimize"
                sample_id = raw.get("sample_id", "").strip() or Path(path).stem
                context += f" ({task}/{sample_id})"
                if task in {".", ".."} or "/" in task or "\\" in task:
                    raise ValueError("invalid task name: must be one directory name")
                if sample_id in {".", ".."} or "/" in sample_id or "\\" in sample_id:
                    raise ValueError("sample_id must be one nonempty filename component")
                if (task, sample_id) in seen:
                    raise ValueError("duplicate sample_id within task")
                authored = {key: _cell(value) for key, value in raw.items()
                            if key not in _METADATA and value.strip()}
                _paths(authored, directory)
                if authored.get("format", "normal") == "normal":
                    authored.setdefault("input", path)
                partial = authored.get("partial")
                if isinstance(partial, dict) and "reference" in partial and partial["reference"] != path:
                    raise ValueError("partial.reference must name this row filename")
                row = {"task": task, "sample_id": sample_id, "path": path,
                       "authored": authored, "row_number": number,
                       "source_status": raw.get("source_status", "").strip()}
                parse_task(task, validation_payload(row), invocation_dir=directory)
            except (TypeError, ValueError, OSError) as exc:
                raise ValueError(f"{context}: {exc}") from exc
            seen.add((task, sample_id))
            rows.append(row)
    if not rows:
        raise ValueError(f"{annotation}: at least one seed row is required")
    return {"directory": str(directory), "annotation": str(annotation), "rows": rows}


def source_tasks(seed_input):
    """Native per-seed tasks for shared source/MSA preparation."""
    return {f"seed_{index:06d}": validation_payload(row)
            for index, row in enumerate(seed_input["rows"])}
