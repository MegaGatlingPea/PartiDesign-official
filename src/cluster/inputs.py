"""Discover validation jobs and recover their successful generated backbones.

Validation rows identify sequence slots.  Clustering uses the final filename stem
in the task's current result.csv, so copies of a task do not depend on saved
absolute paths or on the validation success/refold directory contents.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ClusterJob:
    task_dir: Path
    method_dir: Path

    @property
    def method(self) -> str:
        return self.method_dir.name


@dataclass(frozen=True, slots=True)
class ClusterSample:
    sample: str
    structure: Path
    design_chains: tuple[dict, ...]


@dataclass(frozen=True, slots=True)
class ClusterInput:
    job: ClusterJob
    samples: dict[str, ClusterSample]
    groups: dict[str, tuple[str, ...]]
    sources: dict[str, dict[str, tuple[str, ...]]]


def _names(values: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Task, method and criterion names must be nonempty strings")
        if value != value.strip():
            raise ValueError(f"Name contains surrounding whitespace: {value!r}")
        if value not in result:
            result.append(value)
    return tuple(result)


def _method_job(method_dir: Path) -> ClusterJob:
    if method_dir.parent.name != "validate":
        raise ValueError(f"Validation method must be TASK/validate/METHOD: {method_dir}")
    return ClusterJob(method_dir.parent.parent, method_dir)


def _task_jobs(task_dir: Path) -> list[ClusterJob]:
    validate = task_dir / "validate"
    if not validate.is_dir():
        return []
    return [
        ClusterJob(task_dir, child)
        for child in sorted(validate.iterdir())
        if child.is_dir() and not child.name.startswith(".")
        and (child / "success.csv").is_file()
    ]


def discover_jobs(
    paths: Path | str | Iterable[Path | str],
    *,
    methods: Iterable[str] = (),
    tasks: Iterable[str] = (),
) -> list[ClusterJob]:
    """Find jobs from success, method, task, or direct-child run directories.

    Every returned job belongs to one task and one validation method.  Discovery
    visits only the documented directory levels and never walks structure files.
    Symlink aliases share one resolved job; selectors use actual directory names.
    """
    roots = (paths,) if isinstance(paths, (str, Path)) else paths
    selected_methods, selected_tasks = set(_names(methods)), set(_names(tasks))
    result = []
    seen: set[tuple[Path, Path]] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Cluster input directory does not exist: {root}")
        if root.name == "success" and (root.parent / "success.csv").is_file():
            jobs = [_method_job(root.parent)]
        elif (root / "success.csv").is_file():
            jobs = [_method_job(root)]
        elif (root / "result.csv").is_file():
            jobs = _task_jobs(root)
        else:
            jobs = [
                job
                for task_dir in sorted(root.iterdir())
                if task_dir.is_dir() and not task_dir.name.startswith(".")
                and (task_dir / "result.csv").is_file()
                for job in _task_jobs(task_dir)
            ]
        for job in jobs:
            job = ClusterJob(job.task_dir.resolve(), job.method_dir.resolve())
            if job.method_dir.parent != job.task_dir / "validate":
                raise ValueError(f"Job method directory is not TASK/validate/METHOD: {job.method_dir}")
            if selected_methods and job.method not in selected_methods:
                continue
            if selected_tasks and job.task_dir.name not in selected_tasks:
                continue
            identity = (job.task_dir, job.method_dir)
            if identity not in seen:
                seen.add(identity)
                result.append(job)
    return result


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if not path.is_file():
        raise ValueError(f"Required CSV file is missing: {path}")
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = tuple(reader.fieldnames or ())
            if len(fields) != len(set(fields)):
                raise ValueError(f"Duplicate column names in {path}")
            rows = []
            for index, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"{path}:{index}: more cells than column names")
                rows.append({key: (value or "").strip() for key, value in row.items()})
            return fields, rows
    except (OSError, csv.Error, UnicodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def _json(value: str, *, context: str, field: str):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: {field} is not valid JSON") from exc


def _criterion_columns(fields: tuple[str, ...]) -> set[str]:
    # Refold prediction columns also end in .status.  Filter decisions have a
    # .success column, or a paired .status/.reason when only status is stored.
    return {
        name.rsplit(".", 1)[0]
        for name in fields
        if (name.endswith(".success") or (
            name.endswith(".status") and f"{name[:-7]}.reason" in fields
        )) and name.rsplit(".", 1)[0]
    }


def _passing(row: dict[str, str], *, context: str) -> tuple[str, ...]:
    explicit = "passing_criteria" in row
    value = _json(row["passing_criteria"] or "[]", context=context, field="passing_criteria") if explicit else []
    if not isinstance(value, list) or any(
        not isinstance(name, str) or not name or name != name.strip()
        for name in value
    ):
        raise ValueError(f"{context}: passing_criteria must be a JSON list of nonempty names")
    passed = set(value)
    names = _criterion_columns(tuple(row))
    names.update(passed)
    for name in names:
        raw_success = row.get(f"{name}.success", "").lower()
        if raw_success not in {"", "0", "1", "false", "true"}:
            raise ValueError(f"{context}: invalid {name}.success value {raw_success!r}")
        flag = None if not raw_success else raw_success in {"1", "true"}
        status = row.get(f"{name}.status", "")
        if status not in {"", "passed", "failed", "unavailable", "not_applicable"}:
            raise ValueError(f"{context}: invalid {name}.status value {status!r}")
        if flag is not None and status and flag != (status == "passed"):
            raise ValueError(f"{context}: conflicting {name}.success and {name}.status")
        evidence = flag is True or status == "passed"
        if explicit:
            if (evidence and name not in passed) or (
                name in passed and (flag is False or status in {"failed", "unavailable", "not_applicable"})
            ):
                raise ValueError(f"{context}: passing_criteria conflicts with {name} decision")
        elif evidence:
            passed.add(name)
    return tuple(sorted(passed))


def _source_label(row: dict[str, str], *, context: str) -> str:
    source = row.get("sequence_source", "").strip().lower()
    if not source:
        raise ValueError(f"{context}: successful sequence has no sequence_source")
    settings = _json(row.get("sequence_settings") or "{}", context=context, field="sequence_settings")
    if not isinstance(settings, dict):
        raise ValueError(f"{context}: sequence_settings must be a JSON object")
    fixed = settings.get("fixed_interface")
    if fixed is not None and not isinstance(fixed, dict):
        raise ValueError(f"{context}: sequence_settings.fixed_interface must be an object or null")
    enabled = fixed.get("enabled", False) if fixed is not None else False
    if type(enabled) is not bool:
        raise ValueError(f"{context}: sequence_settings.fixed_interface.enabled must be a boolean")
    if source == "gt":
        if enabled:
            raise ValueError(f"{context}: GT cannot enable fixed_interface")
        return "self"
    branch = row.get("sequence_branch", "") or source
    if branch == "self":
        raise ValueError(f"{context}: 'self' sequence branch is reserved for GT")
    canonical = f"{source}_fi" if enabled else source
    # The source settings determine FI.  A custom branch remains readable but
    # cannot turn an ordinary MPNN source into an FI source by its name alone.
    if branch in {source, f"{source}_fi"}:
        return canonical
    return f"{branch} ({canonical})"


def _sample_evidence(
    row: dict[str, str], sample: str, *, context: str, previous: dict
) -> None:
    """Reject conflicting original identities across one sample's slots."""
    saved = row.get("input_file", "")
    if saved:
        filename = Path(saved).name
        if Path(filename).stem != sample:
            raise ValueError(f"{context}: input_file conflicts with sample {sample!r}")
        if "input_file" in previous and previous["input_file"] != filename:
            raise ValueError(f"{context}: conflicting input_file values for sample {sample!r}")
        previous["input_file"] = filename
    for field in ("design_chains", "design_modality", "design_sequence", "token_length"):
        value = row.get(field, "")
        if not value:
            continue
        if field == "design_chains":
            value = _json(value, context=context, field=field)
        if field in previous and previous[field] != value:
            raise ValueError(f"{context}: conflicting {field} values for sample {sample!r}")
        previous[field] = value


def _design_chains(row: dict[str, str], *, context: str) -> tuple[dict, ...]:
    raw = row.get("design_chains", "")
    if not raw:
        raise ValueError(f"{context}: missing design_chains metadata with final label_asym_id values")
    chains = _json(raw, context=context, field="design_chains")
    if not isinstance(chains, list) or not chains:
        raise ValueError(f"{context}: design_chains must be a nonempty JSON list")
    seen = set()
    for index, item in enumerate(chains):
        prefix = f"{context}: design_chains[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix} must be an object")
        missing = {"chain", "modality", "sequence", "residue_count"}.difference(item)
        if missing:
            raise ValueError(f"{prefix} is missing {', '.join(sorted(missing))}")
        chain, modality = item["chain"], item["modality"]
        if not isinstance(chain, str) or not chain or chain != chain.strip():
            raise ValueError(f"{prefix}.chain must be a nonempty final label_asym_id")
        if chain in seen:
            raise ValueError(f"{context}: duplicate design chain {chain!r}")
        seen.add(chain)
        if not isinstance(modality, str) or modality not in {"protein", "rna", "dna", "ligand"}:
            raise ValueError(f"{prefix}.modality is invalid")
        if not isinstance(item["sequence"], str) or not item["sequence"]:
            raise ValueError(f"{prefix}.sequence must be a nonempty string")
        count = item["residue_count"]
        if type(count) is not int or count < 1:
            raise ValueError(f"{prefix}.residue_count must be a positive integer")
        if "sub_modality" in item and (modality != "protein" or item["sub_modality"] != "peptide"):
            raise ValueError(f"{prefix}.sub_modality is invalid")
        if "cyclic_offset" in item and (
            modality != "protein" or type(item["cyclic_offset"]) is not bool
            or (item["cyclic_offset"] and count < 3)
        ):
            raise ValueError(f"{prefix}.cyclic_offset is invalid")
    return tuple(chains)


def _samples(task_dir: Path, selected: set[str]) -> dict[str, ClusterSample]:
    result_path = task_dir / "result.csv"
    _, rows = _read_csv(result_path)
    matched: dict[str, tuple[dict[str, str], int]] = {}
    for index, row in enumerate(rows, start=2):
        filename = row.get("filename", "")
        sample = Path(filename).stem
        if not filename or sample not in selected:
            continue
        if sample in matched and row != matched[sample][0]:
            raise ValueError(f"{result_path}:{index}: conflicting result.csv rows for sample {sample!r}")
        matched[sample] = (row, index)
    missing = selected.difference(matched)
    if missing:
        raise ValueError(f"{result_path}: no current filename matching successful sample(s): {', '.join(sorted(missing))}")
    samples = {}
    for sample in sorted(selected):
        row, index = matched[sample]
        context = f"{result_path}:{index} ({sample})"
        relative = Path(row["filename"])
        if relative.is_absolute():
            raise ValueError(f"{context}: result filename must be relative to its task")
        structure = (task_dir / relative).resolve()
        if not structure.is_relative_to(task_dir) or any(
            part in {"validate", "original", "refolds", "success"} for part in relative.parts
        ):
            raise ValueError(f"{context}: result filename does not identify an original task structure")
        if not structure.is_file():
            raise ValueError(f"{context}: generated structure is missing: {structure}")
        samples[sample] = ClusterSample(sample, structure, _design_chains(row, context=context))
    return samples


def load_job(job: ClusterJob, *, criteria: Iterable[str] = ()) -> ClusterInput:
    """Load each selected successful backbone once, grouped by passed criterion.

    ``sources[criterion][sample]`` records its passing sequence branches.  Only
    selected samples' current result rows and structure paths are inspected;
    saved ``input_file`` and all refold files are deliberately non-authoritative.
    For an empty report without criterion columns, explicit criteria create empty
    groups; unknown names are rejected when rows or criterion columns exist.
    """
    job = ClusterJob(Path(job.task_dir).expanduser().resolve(), Path(job.method_dir).expanduser().resolve())
    if job.method_dir.parent != job.task_dir / "validate":
        raise ValueError(f"Job method directory is not TASK/validate/METHOD: {job.method_dir}")
    success_path = job.method_dir / "success.csv"
    fields, rows = _read_csv(success_path)
    requested = _names(criteria)
    known = _criterion_columns(fields)
    passing_rows = []
    for index, row in enumerate(rows, start=2):
        context = f"{success_path}:{index}"
        passed = _passing(row, context=context)
        known.update(passed)
        passing_rows.append((row, passed, context))
    unknown = set(requested).difference(known)
    if unknown and (rows or known):
        raise ValueError(f"{success_path}: unknown criterion/criteria: {', '.join(sorted(unknown))}")
    # The validator writes only its base columns when there are no successes.
    # With neither rows nor criterion columns there is no contrary evidence:
    # an explicitly requested criterion denotes an empty cohort, not a failure.
    names = requested or tuple(sorted({name for _, passed, _ in passing_rows for name in passed}))
    grouped: dict[str, dict[str, set[str]]] = {name: {} for name in names}
    evidence: dict[str, dict] = {}
    for row, passed, context in passing_rows:
        chosen = set(passed).intersection(names)
        if not chosen:
            continue
        sample = row.get("sample", "")
        if not sample or sample in {".", ".."} or Path(sample).name != sample:
            raise ValueError(f"{context}: invalid or missing sample filename stem")
        _sample_evidence(row, sample, context=context, previous=evidence.setdefault(sample, {}))
        label = _source_label(row, context=context)
        for name in chosen:
            grouped[name].setdefault(sample, set()).add(label)
    selected = {sample for values in grouped.values() for sample in values}
    samples = _samples(job.task_dir, selected) if selected else {}
    groups = {name: tuple(sorted(values)) for name, values in grouped.items()}
    sources = {name: {sample: tuple(sorted(labels)) for sample, labels in sorted(values.items())}
               for name, values in grouped.items()}
    return ClusterInput(job, samples, groups, sources)
