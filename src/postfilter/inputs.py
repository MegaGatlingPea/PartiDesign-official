"""Select published, named-quality-passed complex refolds using metadata only."""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import re
import stat

from .schema import Candidate, Selection
from .storage import read_json

METHODS = ("af2_ig", "esmfold2-fast", "esmfold2-standard", "af3", "af3-la")
_PEPTIDE = {"peptide", "cyclic-peptide", "cyclic_peptide"}


class _Rejected(ValueError):
    def __init__(self, code, reason):
        super().__init__(reason)
        self.code = code


def _reject(code, reason):
    raise _Rejected(code, reason)


def _diagnostic(selection, code, reason, **context):
    selection.diagnostics.append({"stage": "source-selection", "code": code, "reason": str(reason), **context})


def _names(values):
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError("Task names must be nonempty strings")
    return set(values)


def _identifier(value):
    return (isinstance(value, str) and bool(value) and value == value.strip()
            and value not in {".", ".."} and "/" not in value and "\\" not in value)


def _method_roots(root):
    if root.name == "success":
        yield root.parent
    elif root.parent.name == "validate":
        yield root
    elif root.name == "validate":
        for method in METHODS:
            path = root / method
            if path.is_dir():
                yield path
    elif (root / "validate").is_dir():
        yield from _method_roots(root / "validate")
    else:
        with os.scandir(root) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                if entry.name.startswith(".") or entry.name in {"original", "postfilter"}:
                    continue
                path = root / entry.name
                if entry.is_dir(follow_symlinks=False) and (path / "validate").is_dir():
                    yield from _method_roots(path / "validate")


def _chain_labels(value, description, *, empty=False):
    if isinstance(value, str):
        value = value.split(",") if value else []
    if not isinstance(value, (list, tuple)) or any(not _identifier(chain) for chain in value):
        _reject("metadata_incomplete", f"{description} must contain explicit chain labels")
    labels = tuple(value)
    if not empty and not labels:
        _reject("metadata_incomplete", f"{description} is empty")
    if len(set(labels)) != len(labels):
        _reject("ambiguous_chain_roles", f"{description} contains duplicate chain labels")
    return labels


def _design_metadata(value):
    """Read saved output labels or authored chain declarations, without chemistry."""
    if value is None:
        return None, set()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            _reject("metadata_incomplete", "Saved design_chains is not valid JSON")
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        _reject("metadata_incomplete", "Saved design_chains must list explicit binder roles")
    labels, kinds = [], set()
    for entry in value:
        if isinstance(entry, str):
            labels.append(entry)
            continue
        if not isinstance(entry, dict) or not entry:
            _reject("metadata_incomplete", "Saved design-chain role is malformed")
        if "chain" in entry:
            labels.append(entry["chain"])
            modality = entry.get("modality")
            subtype = entry.get("sub_modality")
            if modality is not None:
                if modality not in {"protein", *_PEPTIDE}:
                    _reject("not_ppi", "Designed partners must be protein or peptide")
                kinds.add("peptide" if modality in _PEPTIDE or subtype in _PEPTIDE else "protein")
            elif subtype in _PEPTIDE:
                kinds.add("peptide")
        else:
            for chain, modality in entry.items():
                labels.append(chain)
                if modality not in {"protein", *_PEPTIDE}:
                    _reject("not_ppi", "Designed partners must be protein or peptide")
                kinds.add("peptide" if modality in _PEPTIDE else "protein")
    if len(kinds) > 1:
        _reject("not_ppi", "Mixed protein and peptide binder declarations are unsupported")
    return _chain_labels(labels, "Saved binder chain roles"), kinds


def _kind_hint(metadata):
    kinds = set()
    for key in ("binder_kind", "sub_modality", "modality", "design_modality"):
        value = metadata.get(key)
        if value in _PEPTIDE:
            kinds.add("peptide")
        elif key == "binder_kind" and value == "protein":
            kinds.add("protein")
        elif value and value not in {"protein", "joint", "antibody"}:
            _reject("not_ppi", f"Saved {key} is not protein or peptide")
    return kinds


def _row_roles(row):
    if row.get("design_chains") not in (None, ""):
        return _design_metadata(row["design_chains"])
    if any(row.get(key) for key in ("heavy_role", "heavy_chain", "light_chain")):
        if row.get("heavy_role") not in {"VH", "VHH"} or not row.get("heavy_chain"):
            _reject("metadata_incomplete", "Antibody row requires an explicit heavy chain and VH/VHH role")
        light = row.get("light_chain")
        if row["heavy_role"] == "VHH" and light:
            _reject("ambiguous_chain_roles", "VHH row also declares a light chain")
        labels = [row["heavy_chain"]] + ([light] if light else [])
        return _chain_labels(labels, "Saved antibody binder roles"), {"protein"}
    if row.get("design_chain_id"):
        return _chain_labels([row["design_chain_id"]], "Saved binder chain role"), set()
    return None, set()


class _OptionalMetadata:
    """Read task/subtype supplements at most once; no source-slot reconstruction."""
    def __init__(self, root, task_definitions):
        self.root = root
        self.task_dir = root.parent.parent
        self.task_definitions = task_definitions
        self.plan = None

    def task(self):
        key = str(self.task_dir)
        if key not in self.task_definitions:
            task = {}
            path = self.task_dir / "resolved_config.yaml"
            try:
                with path.open(encoding="utf-8") as stream:
                    import yaml
                    saved = yaml.safe_load(stream)
                if isinstance(saved, dict) and isinstance(saved.get("tasks"), dict):
                    value = saved["tasks"].get(self.task_dir.name)
                    if isinstance(value, dict):
                        task = value
            except (OSError, UnicodeError, ValueError, ImportError):
                pass
            except yaml.YAMLError:
                pass
            self.task_definitions[key] = task
        return self.task_definitions[key]

    def row(self, sample):
        if self.plan is None:
            try:
                self.plan = read_json(self.root / ".state" / "plan.json") or {}
            except (OSError, ValueError):
                self.plan = {}
        samples = self.plan.get("samples")
        entry = samples.get(sample) if isinstance(samples, dict) else None
        row = entry.get("row") if isinstance(entry, dict) else None
        return row if isinstance(row, dict) else {}


def _assembly(record, sequences, supplement):
    metadata = record.get("input_metadata", {})
    if not isinstance(metadata, dict):
        _reject("metadata_incomplete", "Saved input_metadata must be an object")
    row = metadata.get("row", {})
    if not isinstance(row, dict):
        _reject("metadata_incomplete", "Saved input row metadata must be an object")
    binder, kinds = _design_metadata(metadata.get("design_chains"))
    kinds |= _kind_hint(metadata) | _kind_hint(row)
    if binder is None:
        binder, row_kinds = _row_roles(row)
        kinds |= row_kinds
    if binder is None:
        row = supplement.row(record["sample"])
        binder, row_kinds = _row_roles(row)
        kinds |= row_kinds | _kind_hint(row)
    if binder is None:
        binder, task_kinds = _design_metadata(supplement.task().get("design_chains"))
        kinds |= task_kinds
    if binder is None:
        _reject("missing_chain_roles", "Saved binder roles are absent from input_metadata and optional task/row metadata")
    target_value = metadata.get("target_chains", row.get("antigen_chains", row.get("target_chains")))
    target = (_chain_labels(target_value, "Saved target chain roles") if target_value is not None
              else tuple(chain for chain in sequences if chain not in binder))
    if not target:
        _reject("not_ppi", "A protein complex requires both binder and target chain assemblies")
    if set(binder).intersection(target) or set(binder + target) != set(sequences):
        _reject("ambiguous_chain_roles", "Saved binder/target roles overlap or do not cover the saved sequence map")
    cycles = _chain_labels(metadata.get("cyclic_chains", []), "Saved physical cyclic chains", empty=True)
    if set(cycles).difference(sequences):
        _reject("metadata_mismatch", "Saved physical cyclic chains contain an unknown saved chain")
    if set(binder).issubset(cycles):
        kinds.add("peptide")
    if "peptide" not in kinds and metadata.get("binder_kind") != "protein":
        definition = supplement.task()
        _, task_kinds = _design_metadata(definition.get("design_chains"))
        kinds |= task_kinds | _kind_hint(definition)
        if not task_kinds and definition.get("format") != "antibody":
            row = row or supplement.row(record["sample"])
            _, row_kinds = _row_roles(row)
            kinds |= row_kinds | _kind_hint(row)
    return binder, target, "peptide" if "peptide" in kinds else "protein", cycles


def _checked_record(record, root, record_path):
    if not isinstance(record, dict):
        _reject("metadata_incomplete", "Published sequence JSON is missing or is not an object")
    if record.get("status") != "complete":
        _reject("incomplete_record", "Published sequence record is not complete")
    sample = record.get("sample")
    if not _identifier(sample) or record_path.parent.name != sample:
        _reject("metadata_mismatch", "Published sample directory disagrees with its saved identity")
    if record.get("task") != root.parent.parent.name or record.get("method") != root.name:
        _reject("metadata_mismatch", "Published record task/method disagrees with its current directory")
    sequence = record.get("sequence")
    if not isinstance(sequence, dict):
        _reject("metadata_incomplete", "Saved sequence metadata is missing")
    index = sequence.get("seq_index")
    if type(index) is not int or index < 0 or record_path.name != f"seq{index}.json":
        _reject("metadata_mismatch", "Published sequence filename disagrees with its saved index")
    if not _identifier(sequence.get("source")) or type(sequence.get("source_index")) is not int or sequence["source_index"] < 0:
        _reject("metadata_incomplete", "Saved sequence source and source index are required")
    sequences = sequence.get("sequences")
    if not isinstance(sequences, dict) or not sequences or any(
        not _identifier(chain) or not isinstance(value, str) or not value
        for chain, value in sequences.items()
    ):
        _reject("metadata_incomplete", "Actual saved sequence map is missing or malformed")
    return sequence


def _passing_keys(record):
    decisions = record.get("decisions")
    if not isinstance(decisions, dict):
        _reject("metadata_incomplete", "Saved named quality decisions are missing")
    selected = defaultdict(set)
    for name, decision in decisions.items():
        if (not isinstance(name, str) or not name.strip() or name != name.strip()
                or name.lower() == "unfiltered"):
            continue
        if not isinstance(decision, dict) or decision.get("status") != "passed":
            continue
        if decision.get("criteria", name) != name:
            _reject("metadata_mismatch", f"Saved decision name disagrees with criterion {name}")
        keys = decision.get("passing_prediction_keys")
        if not isinstance(keys, list) or not keys or any(not _identifier(key) for key in keys):
            _reject("metadata_incomplete", f"Passed criterion {name} has no concrete passing prediction keys")
        for key in keys:
            selected[key].add(name)
    return selected


def _published_prediction(prediction, key, record_path):
    if prediction.get("key", key) != key:
        _reject("metadata_mismatch", "Passing prediction key disagrees with its saved identity")
    if prediction.get("status") != "complete" or prediction.get("metrics_status") != "complete":
        _reject("incomplete_prediction", "Passing complex prediction or its metric state is incomplete")
    saved = prediction.get("structure_path")
    if not isinstance(saved, str) or not saved:
        _reject("metadata_incomplete", "Passing complex prediction has no saved structure filename")
    published = record_path.parent / Path(saved).name
    if published.suffix.lower() not in {".cif", ".pdb"}:
        _reject("invalid_refold", "Passing complex prediction is not a CIF or PDB structure")
    try:
        path = published.resolve(strict=True)
        mode = path.stat().st_mode
    except FileNotFoundError:
        _reject("missing_refold", f"Published complex sibling is missing: {published}")
    if not stat.S_ISREG(mode):
        _reject("invalid_refold", f"Published complex sibling is not a regular file: {published}")
    return path


def select_refolds(inputs, *, task_names=None) -> Selection:
    """Enumerate published success JSONs once and retain actual passing complexes."""
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]
    tasks = _names(task_names)
    selection = Selection()
    roots, seen_roots, task_definitions = [], set(), {}
    for value in inputs:
        raw = Path(value).expanduser()
        context = {"input": str(raw)}
        try:
            root = raw.resolve(strict=True)
            if not root.is_dir():
                _reject("invalid_input", "Select a run, task, validation method or success directory")
            discovered = list(_method_roots(root))
            if not discovered:
                _diagnostic(selection, "no_success_publication", "No supported validation method was found", **context)
            for method_root in discovered:
                if method_root not in seen_roots:
                    seen_roots.add(method_root)
                    roots.append(method_root)
        except (OSError, ValueError, RuntimeError) as exc:
            _diagnostic(selection, getattr(exc, "code", "invalid_input"), exc, **context)
    seen = set()
    for root in roots:
        task_dir = root.parent.parent
        context = {"task_dir": str(task_dir), "method_dir": str(root), "task": task_dir.name, "method": root.name}
        if tasks is not None and task_dir.name not in tasks:
            continue
        if root.name not in METHODS or root.parent.name != "validate":
            _diagnostic(selection, "unsupported_method", f"Unsupported validation method directory: {root}", **context)
            continue
        if str(task_dir) not in selection.task_dirs:
            selection.task_dirs.append(str(task_dir))
        counts = selection.counts.setdefault(str(task_dir), {
            "published_records": 0, "nominated": 0, "deduplicated": 0, "non_ppi": 0, "selected": 0,
        })
        supplement = _OptionalMetadata(root, task_definitions)
        for record_path in sorted((root / "success").rglob("seq*.json")):
            if not re.fullmatch(r"seq\d+\.json", record_path.name):
                continue
            counts["published_records"] += 1
            record_context = {**context, "record_path": str(record_path)}
            try:
                record = read_json(record_path)
                sequence = _checked_record(record, root, record_path)
                record_context.update(sample=record["sample"], seq_index=sequence["seq_index"])
                nominations = _passing_keys(record)
                predictions = record.get("predictions")
                if not isinstance(predictions, dict):
                    _reject("metadata_incomplete", "Saved prediction metadata is missing")
                complexes = {}
                for key, names in nominations.items():
                    prediction = predictions.get(key)
                    if not isinstance(prediction, dict):
                        _diagnostic(selection, "metadata_incomplete", "Named passing key has no saved prediction",
                                    **record_context, prediction_key=key)
                        continue
                    if prediction.get("protocol") != "complex":
                        continue
                    identity = (str(root), record["sample"], sequence["seq_index"], key)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    counts["nominated"] += len(names)
                    counts["deduplicated"] += 1
                    complexes[key] = names
                if not complexes:
                    continue
                try:
                    binder, target, kind, cycles = _assembly(record, sequence["sequences"], supplement)
                except _Rejected as exc:
                    if exc.code == "not_ppi":
                        counts["non_ppi"] += len(complexes)
                    raise
            except (OSError, UnicodeError, TypeError, ValueError) as exc:
                _diagnostic(selection, getattr(exc, "code", "metadata_incomplete"), exc, **record_context)
                continue
            for key, names in complexes.items():
                try:
                    path = _published_prediction(predictions[key], key, record_path)
                    selection.candidates.append(Candidate(
                        task_dir=str(task_dir), method_dir=str(root), task=task_dir.name, method=root.name,
                        sample=record["sample"], seq_index=sequence["seq_index"],
                        source=sequence["source"], source_index=sequence["source_index"],
                        prediction_key=key, refold_path=str(path), record_path=str(record_path),
                        binder_chains=binder, target_chains=target, sequences=dict(sequence["sequences"]),
                        passing_criteria=tuple(sorted(names)), binder_kind=kind, declared_cyclic_chains=cycles,
                    ))
                    counts["selected"] += 1
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    _diagnostic(selection, getattr(exc, "code", "invalid_refold"), exc,
                                **record_context, prediction_key=key)
    return selection
