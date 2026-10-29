"""Adapt published pipeline final representatives to native postfilter inputs.

Only final state/CSV publications nominate structures. Raw predictions supply
sequence, ranking and named-quality evidence; they never nominate alternatives.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import stat

from .inputs import _assembly, _design_metadata, _diagnostic, _identifier, _passing_keys, _reject
from .schema import Candidate
from .storage import read_json


def _rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            identity = row.get("candidate_id")
            if not _identifier(identity) or identity in rows:
                _reject("metadata_mismatch", f"Missing or duplicate candidate identity in {path}")
            rows[identity] = row
    return rows


def _cell(value):
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        if value[:1] in {"[", "{"}:
            return json.loads(value)
        return value
    if value is None:
        return ""
    return str(value)


def _sequence(record):
    if record.get("status") != "complete":
        _reject("incomplete_record", "Published final prediction record is not complete")
    sequence = record.get("sequence")
    if not isinstance(sequence, dict):
        _reject("metadata_incomplete", "Published final sequence metadata is missing")
    if (type(sequence.get("seq_index")) is not int or sequence["seq_index"] < 0
            or not _identifier(sequence.get("source"))
            or type(sequence.get("source_index")) is not int or sequence["source_index"] < 0):
        _reject("metadata_incomplete", "Published final sequence provenance is incomplete")
    sequences = sequence.get("sequences")
    if not isinstance(sequences, dict) or not sequences or any(
        not _identifier(chain) or not isinstance(value, str) or not value
        for chain, value in sequences.items()
    ):
        _reject("metadata_incomplete", "Published final sequence map is missing or malformed")
    return sequence


def _partners(record, row, sequences):
    # The final row describes the current geometry. Never infer binder labels
    # from chain order, a prediction's private labels, or a starting structure.
    authored = _cell(row.get("design_chains", ""))
    binder, kinds = _design_metadata(authored)
    if not binder or not isinstance(authored, list) or any(not isinstance(x, dict) for x in authored):
        _reject("missing_chain_roles", "Final row requires explicit design-chain metadata")
    for chain in authored:
        if chain.get("sequence") != sequences.get(chain.get("chain")):
            _reject("metadata_mismatch", "Final row design sequence differs from its published prediction")
    for field in ("fixed_restypes", "bond_constraints"):
        if field not in row or not isinstance(_cell(row[field]), list):
            _reject("metadata_incomplete", f"Final row has no saved {field} chemistry metadata")
    metadata = record.get("input_metadata", {})
    if not isinstance(metadata, dict):
        _reject("metadata_incomplete", "Final input_metadata must be an object")
    metadata = dict(metadata, row=row, design_chains=authored,
                    binder_kind="peptide" if "peptide" in kinds else "protein")
    return _assembly({"sample": row["candidate_id"], "input_metadata": metadata}, sequences, None)


def _model(key):
    match = re.fullmatch(r"complex_model(\d+)_sample\d+", key)
    return int(match[1]) if match else None


def _ranked(record, key):
    method = record.get("method")
    if method not in {"af2_ig", "af3"}:
        _reject("unsupported_method", f"Unsupported final pipeline method: {method}")
    predictions = record.get("predictions")
    expected = record.get("expected_predictions")
    if (not isinstance(predictions, dict) or not isinstance(expected, list) or not expected
            or any(not _identifier(value) for value in expected)
            or len(expected) != len(set(expected))
            or any(not isinstance(predictions.get(value), dict) for value in expected)):
        _reject("metadata_incomplete", "Final prediction identities are missing or malformed")
    saved = record.get("prediction_selection")
    if not isinstance(saved, dict) or saved.get("mode") not in {"best_ranked", "all"}:
        _reject("metadata_incomplete", "Saved final prediction selection is missing")
    models = {_model(value) for value in expected if value.startswith("complex_")} if method == "af2_ig" else None
    if models is not None and (not models or None in models):
        _reject("metadata_incomplete", "Configured final AF2 model identities are missing")
    nominations = _passing_keys(record)
    required = [key]
    view = record
    if saved["mode"] == "best_ranked":
        from src.pipeline.runner import final_prediction_view
        view, ranked = final_prediction_view(record, models=sorted(models) if models is not None else None)
        if ranked.get("selected_prediction_key") != key:
            _reject("metadata_mismatch", "Published final representative is not its saved-policy native-ranked winner")
        for field in ("selected_prediction_key", "paired_prediction_key", "selected_models"):
            if saved.get(field) != ranked.get(field):
                _reject("metadata_mismatch", f"Saved final {field} disagrees with native ranking")
        if method == "af2_ig":
            required = [value["selected_prediction_key"] for value in ranked["selected_models"].values()]
    elif method == "af2_ig":
        passing_models = {_model(value) for value in nominations if value in expected
                          and predictions.get(value, {}).get("protocol") == "complex"}
        if not models.issubset(passing_models):
            _reject("no_quality_success", "Not every configured final AF2 model passed named quality criteria")
    if method == "af2_ig":
        if saved.get("status") != "passed":
            _reject("no_quality_success", "Final AF2 model gate did not pass")
        paired = record.get("paired_metrics")
        for winner in required:
            suffix = winner.removeprefix("complex_")
            monomer = predictions.get("monomer_" + suffix)
            if (not isinstance(monomer, dict) or monomer.get("protocol") != "monomer"
                    or monomer.get("key", "monomer_" + suffix) != "monomer_" + suffix
                    or monomer.get("status") != "complete" or monomer.get("metrics_status") != "complete"
                    or not isinstance(paired, dict) or not isinstance(paired.get(suffix), dict)):
                _reject("metadata_incomplete", "Final AF2 winner lacks its same-model/sample monomer comparison")
    names = set()
    for winner in required:
        prediction = view["predictions"].get(winner)
        if (winner not in expected or not isinstance(prediction, dict) or prediction.get("protocol") != "complex"
                or prediction.get("key", winner) != winner
                or prediction.get("status") != "complete" or prediction.get("metrics_status") != "complete"):
            _reject("incomplete_prediction", "Final selected complex or metrics are incomplete")
        if winner not in nominations:
            _reject("no_quality_success", "Not every final selected model passed named quality criteria")
        names.update(nominations[winner])
    return tuple(sorted(names))


def _accepted(candidate, row, decision, root):
    identity = candidate.get("candidate_id")
    if not _identifier(identity) or candidate.get("task") != root.parent.name:
        _reject("metadata_mismatch", "Final state identity disagrees with its task directory")
    if (not row or row.get("status") != "passed" or row.get("error")
            or not decision or str(decision.get("selected", "")).lower() != "true"
            or decision.get("representative_id") != identity):
        _reject("metadata_mismatch", "Final state candidate is not an accepted CSV representative")
    saved_row = candidate.get("row")
    if not isinstance(saved_row, dict) or not saved_row:
        _reject("metadata_incomplete", "Final state has no current row metadata")
    for field in ("filename", "design_chains", "design_modality", "design_chain_id",
                  "contig_string", "tip_atoms", "fixed_restypes", "bond_constraints"):
        if field in saved_row and (field not in row or _cell(row[field]) != _cell(saved_row[field])):
            _reject("metadata_mismatch", f"Final state and result row disagree on {field}")
    for field in ("parent_id", "origin_id", "source", "prediction_key"):
        if not candidate.get(field) or candidate[field] != row.get(field):
            _reject("metadata_mismatch", f"Final state and result row disagree on {field}")
    saved = candidate.get("path")
    if (not isinstance(saved, str) or saved != row.get("filename")
            or Path(saved).name != identity + ".cif" or Path(saved).parent.name != "success"):
        _reject("metadata_mismatch", "Final state/result filename does not name its published success CIF")
    path = root / "success" / Path(saved).name
    try:
        path = path.resolve(strict=True)
        mode = path.stat().st_mode
    except FileNotFoundError:
        _reject("missing_refold", f"Published final representative is missing: {path}")
    if not stat.S_ISREG(mode):
        _reject("invalid_refold", f"Published final representative is not a regular file: {path}")
    return path


def select_final(root, selection, *, task_names=None):
    """Read each published final metadata file once, without parsing structures."""
    task_dir = root.parent
    if task_names is not None and task_dir.name not in task_names:
        return
    context = {"task_dir": str(task_dir), "method_dir": str(root), "task": task_dir.name}
    if str(task_dir) not in selection.task_dirs:
        selection.task_dirs.append(str(task_dir))
    counts = selection.counts.setdefault(str(task_dir), {
        "published_records": 0, "nominated": 0, "deduplicated": 0, "non_ppi": 0, "selected": 0,
    })
    try:
        state = read_json(root / ".work" / "state.json")
        if not isinstance(state, dict) or state.get("status") != "complete":
            _reject("incomplete_record", "Pipeline final state is missing or is not complete")
        candidates = state.get("candidates")
        if not isinstance(candidates, list) or any(not isinstance(value, dict) for value in candidates):
            _reject("metadata_incomplete", "Pipeline final accepted candidate state is malformed")
        rows, decisions = _rows(root / "result.csv"), _rows(root / "selection.csv")
        records = {}
        for path in sorted((root / "predictions").glob("proposal*.json")):
            if not re.fullmatch(r"proposal\d+\.json", path.name):
                continue
            counts["published_records"] += 1
            record = read_json(path)
            if not isinstance(record, dict):
                _reject("metadata_incomplete", f"Published final record is missing: {path}")
            identity = (record.get("request_key"), record.get("parent_id"))
            records.setdefault(identity, []).append((path, record))
        seen = set()
        for candidate in candidates:
            identity = candidate.get("candidate_id")
            candidate_context = dict(context, sample=identity)
            try:
                if not _identifier(identity) or identity in seen:
                    _reject("metadata_mismatch", "Pipeline final state has a missing or duplicate candidate identity")
                seen.add(identity)
                row = rows.get(identity)
                path = _accepted(candidate, row, decisions.get(identity), root)
                request, separator, key = candidate["prediction_key"].rpartition(":")
                matches = records.get((request, candidate["parent_id"]), [])
                if not separator or not _identifier(key) or len(matches) != 1:
                    _reject("metadata_incomplete", "Accepted final representative has no unique published prediction record")
                record_path, record = matches[0]
                candidate_context.update(record_path=str(record_path), prediction_key=key,
                                         method=record.get("method"))
                sequence = _sequence(record)
                if (record.get("origin_id") != candidate["origin_id"]
                        or sequence["source"] != candidate["source"]):
                    _reject("metadata_mismatch", "Final representative sequence provenance disagrees with its publication")
                names = _ranked(record, key)
                counts["nominated"] += len(names)
                counts["deduplicated"] += 1
                binder, target, kind, cycles = _partners(record, row, sequence["sequences"])
                selection.candidates.append(Candidate(
                    task_dir=str(task_dir), method_dir=str(root), task=task_dir.name, method=record["method"],
                    sample=identity, seq_index=sequence["seq_index"], source=sequence["source"],
                    source_index=sequence["source_index"], prediction_key=key, refold_path=str(path),
                    record_path=str(record_path), binder_chains=binder, target_chains=target,
                    sequences=dict(sequence["sequences"]), passing_criteria=names,
                    binder_kind=kind, declared_cyclic_chains=cycles,
                ))
                counts["selected"] += 1
            except (OSError, UnicodeError, KeyError, TypeError, ValueError) as exc:
                if getattr(exc, "code", None) == "not_ppi":
                    counts["non_ppi"] += 1
                _diagnostic(selection, getattr(exc, "code", "metadata_incomplete"), exc, **candidate_context)
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as exc:
        _diagnostic(selection, getattr(exc, "code", "metadata_incomplete"), exc, **context)
