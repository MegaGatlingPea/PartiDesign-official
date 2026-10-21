"""Compact CSV/success reports rebuilt from per-sequence JSON files."""
from __future__ import annotations

import math
import os
from pathlib import Path

from .filters import apply_presets
from .planning import (
    expected_keys,
    prediction_metadata_valid,
    sequence_slot_count,
    sequence_sources_by_sample,
)
from .storage import (
    atomic_csv,
    atomic_json,
    ensure_directory,
    locked,
    nested_record_valid,
    read_json,
    required_scores_valid,
)


def _scalar(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def summarize_record(record: dict, result_row: dict | None = None) -> dict:
    sequence = record.get("sequence", {})
    row = {
        "run": record.get("run"),
        "task": record.get("task"),
        "sample": record.get("sample"),
        "input_file": record.get("input_file"),
        "method": record.get("method"),
        "seq_idx": sequence.get("seq_index"),
        "sequence_source": sequence.get("source"),
        "sequence_branch": sequence.get("sequence_branch", "self" if sequence.get("source") == "gt" else sequence.get("source")),
        "structure_stage": record.get("structure_stage", "base"),
        "interface_positions": sequence.get("interface_positions", sequence.get("settings", {}).get("interface_positions")),
        "interface_target_chains": sequence.get("interface_target_chains", sequence.get("settings", {}).get("interface_target_chains")),
        "interface_applicable": sequence.get("interface_applicable", sequence.get("settings", {}).get("interface_applicable")),
        "source_index": sequence.get("source_index"),
        "sequence_settings": sequence.get("settings", {}),
        "sequences": sequence.get("sequences", {}),
        "status": record.get("status"),
        "error": record.get("error"),
        "failure_stage": record.get("failure_stage"),
        "source_error": record.get("source_error"),
    }
    metadata = record.get("input_metadata", {})
    row_metadata = result_row if isinstance(result_row, dict) else metadata.get("row", {})
    for key in ("design_modality", "design_sequence", "contig_string", "tip_atoms",
                "fixed_restypes", "bond_constraints"):
        row[key] = row_metadata.get(key)
    row["metadata_states"] = metadata.get("metadata_states")
    if isinstance(row_metadata, dict) and "design_chains" in row_metadata:
        row["design_chains"] = row_metadata.get("design_chains")
    else:
        row["design_chains"] = metadata.get("design_chains")
    row["cyclic_chains"] = metadata.get("cyclic_chains")
    row["cyclic_offset_chains"] = metadata.get("cyclic_offset_chains", [])
    row["token_length"] = row_metadata.get("token_length")
    import json
    try:
        declared = json.loads(row_metadata.get("design_chains") or "[]")
        row["design_lengths"] = {entry["chain"]: entry["residue_count"] for entry in declared}
        modes = {entry["modality"] for entry in declared}
        row["design_mode"] = "joint" if "protein" in modes and "ligand" in modes else "single"
        bonds = json.loads(row_metadata.get("bond_constraints") or "[]")
        disulfide = any(b.get("atom1", {}).get("atom") == "SG" and b.get("atom2", {}).get("atom") == "SG" for b in bonds)
        row["topology"] = "head_tail_cyclic" if row["cyclic_chains"] or row["cyclic_offset_chains"] else "disulfide" if disulfide else "linear_or_other"
    except (TypeError, ValueError, KeyError):
        row["design_lengths"] = {}
        row["design_mode"] = "legacy_unknown"
    values = {}
    for key, prediction in sorted(record.get("predictions", {}).items()):
        row[f"{key}.status"] = prediction.get("status")
        row[f"{key}.structure"] = prediction.get("structure_path")
        row[f"{key}.error_stage"] = prediction.get("error_stage")
        row[f"{key}.provenance"] = prediction.get("metric_provenance", {})
        if prediction.get("status") != "complete" or prediction.get("metrics_status") != "complete":
            row[f"{key}.error"] = prediction.get("error")
            if prediction.get("model") != "vina_dock":
                continue
        for name, value in prediction.get("metrics", {}).items():
            row[f"{key}.{name}"] = value
            if _scalar(value):
                values.setdefault(f"{prediction.get('protocol')}.{name}", []).append(value)
        for name, reason in prediction.get("reasons", {}).items():
            row[f"{key}.{name}_reason"] = reason
    for key, metrics in sorted(record.get("paired_metrics", {}).items()):
        for name, value in metrics.items():
            row[f"paired.{key}.{name}"] = value
            if _scalar(value):
                values.setdefault(f"paired.{name}", []).append(value)
    for name, group in values.items():
        row[f"mean.{name}"] = sum(group) / len(group)
    for key, score in record.get("scores", {}).items():
        if isinstance(score, dict):
            for metric, value in score.items():
                row[f"partiscore.{key}.{metric}"] = value
        else:
            row[f"partiscore.{key}"] = score
    for pair, reasons in record.get("paired_reasons", {}).items():
        for name, reason in reasons.items():
            row[f"paired.{pair}.{name}_reason"] = reason
    passing = []
    for name, decision in record.get("decisions", {}).items():
        status = decision["status"]
        row[f"{name}.status"] = status
        row[f"{name}.success"] = 1 if status == "passed" else 0 if status == "failed" else None
        row[f"{name}.reason"] = "; ".join(decision.get("reasons", []))
        if status == "passed":
            passing.append(name)
    row["passing_criteria"] = passing
    return row


def _current_vina_inventory(task_dir: Path) -> tuple[dict, str | None]:
    """Read the live result.csv inventory used by Vina refiltering.

    A saved validation plan is a historical snapshot.  Vina's postprocess gate
    must therefore resolve the row and path again from the task's current
    result.csv before cached docking metrics are considered.
    """

    from .planning import inventory

    try:
        return inventory(task_dir), None
    except (OSError, ValueError) as exc:
        return {}, (
            "postprocess: current result.csv inventory unavailable: "
            f"{type(exc).__name__}: {exc}"
        )


def _resolve_inventory_path(value, task_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = task_dir / path
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _cached_vina_path_error(record: dict, current_path: str, task_dir: Path) -> str | None:
    """Reject cached Vina metrics bound to a different result-row structure."""

    current = _resolve_inventory_path(current_path, task_dir)
    if current is None:
        return "postprocess: current result.csv row has no resolved structure path"
    candidates = []
    saved_input = _resolve_inventory_path(record.get("input_file"), task_dir)
    if saved_input is not None:
        candidates.append(("record.input_file", saved_input))
    for key, prediction in record.get("predictions", {}).items():
        if not isinstance(prediction, dict):
            continue
        saved_structure = _resolve_inventory_path(
            prediction.get("structure_path"), task_dir
        )
        if saved_structure is not None:
            candidates.append((f"prediction.{key}.structure_path", saved_structure))
    for label, saved in candidates:
        if saved != current:
            return (
                "postprocess: cached Vina structure path differs from current "
                f"result.csv row ({label}: {saved}; current: {current})"
            )
    return None


def _invalidate_vina_postprocess(record: dict, message: str) -> None:
    """Keep cached metrics while making a stale Vina success non-publishable."""

    record["status"] = "partial" if record.get("predictions") else "failed"
    previous = record.get("error")
    record["error"] = (
        previous if previous and message in str(previous)
        else f"{message}; cached Vina result invalidated during refilter"
        if not previous else f"{message}; previous error: {previous}"
    )
    record["failure_stage"] = "postprocess"
    for prediction in record.get("predictions", {}).values():
        if not isinstance(prediction, dict):
            continue
        prediction["error_stage"] = "postprocess"
        prior = prediction.get("error")
        prediction["error"] = (
            prior if prior and message in str(prior)
            else message if not prior else f"{message}; previous error: {prior}"
        )
    for decision in record.get("decisions", {}).values():
        if not isinstance(decision, dict):
            continue
        if decision.get("status") != "passed":
            continue
        decision["status"] = "failed"
        reasons = list(decision.get("reasons", []))
        reasons.append(message)
        decision["reasons"] = reasons


def _vina_postprocess_error(
    record: dict,
    current_entry: dict | None,
    inventory_error: str | None,
    *,
    sample: str,
    task_dir: Path,
) -> str | None:
    if current_entry is None:
        return inventory_error or (
            "postprocess: current result.csv row missing for sample "
            f"{sample!r}"
        )
    if current_entry.get("error"):
        return (
            "postprocess: current result.csv row/path is unavailable for "
            f"sample {sample!r}: {current_entry['error']}"
        )
    row = current_entry.get("row")
    if not isinstance(row, dict):
        return (
            "postprocess: current result.csv row missing for sample "
            f"{sample!r}"
        )
    status = row.get("postprocess_status")
    if status != "succeeded":
        if status in (None, ""):
            return (
                "postprocess: current result.csv row lacks explicit "
                "postprocess_status='succeeded'"
            )
        return (
            "postprocess: current result.csv row has "
            f"postprocess_status={status!r}; expected 'succeeded'"
        )
    current_path = current_entry.get("path")
    if not isinstance(current_path, str) or not current_path:
        return (
            "postprocess: current result.csv row has no structure path "
            f"for sample {sample!r}"
        )
    from types import SimpleNamespace
    from .ligand import require_structure_stage
    try:
        require_structure_stage(SimpleNamespace(
            path=Path(current_path), task_dir=task_dir, row=row
        ), "postprocessed")
    except (OSError, ValueError) as exc:
        return f"postprocess: {exc}"
    return _cached_vina_path_error(record, current_path, task_dir)


def records_for_plan(root: Path, plan: dict, *, strict=True):
    """Yield saved records, rechecking Vina against the live result.csv row."""

    method = plan.get("method", root.name)
    expected = expected_keys(method, plan["config"]) if plan.get("config") else None
    current_inventory: dict[str, dict] = {}
    inventory_error = None
    task_dir = Path(plan.get("task_dir", root.parent.parent))
    source_map = sequence_sources_by_sample(plan)
    if method == "vina_dock":
        current_task_dir = root.parent.parent.resolve()
        if task_dir.resolve() != current_task_dir:
            raise ValueError(
                "Vina validation task directory differs from its saved plan; "
                "rerun validation in the current task directory"
            )
        task_dir = current_task_dir
        current_inventory, inventory_error = _current_vina_inventory(task_dir)
    for sample in list(plan.get("samples", {})):
        if Path(sample).name != sample or sample in {".", ".."}:
            raise ValueError("Unsafe sample name in validation plan")
        saved_entry = plan.get("samples", {}).get(sample)
        if not isinstance(saved_entry, dict):
            saved_entry = {}
        current_entry = (
            current_inventory.get(sample) if method == "vina_dock" else saved_entry
        )
        if method == "vina_dock":
            # Keep later summarization/grouping on the live row as well.  A
            # missing row deliberately replaces the saved row with no metadata.
            if current_entry is None:
                current_entry = {
                    "path": saved_entry.get("path"),
                    "row": {},
                    "error": inventory_error or (
                        "postprocess: current result.csv row missing for sample "
                        f"{sample!r}"
                    ),
                }
            plan["samples"][sample] = current_entry
        for seq_index in range(sequence_slot_count(plan, sample, source_map)):
            path = root / "refolds" / sample / f"seq{seq_index}.json"
            try:
                record = read_json(path)
            except ValueError:
                if strict:
                    raise
                continue
            if record is None:
                continue
            if not nested_record_valid(record):
                if strict:
                    raise ValueError(f"Malformed sequence state: {path}")
                continue
            sequence = record.get("sequence", {})
            if sequence.get("seq_index") != seq_index:
                if strict:
                    raise ValueError(f"Sequence index differs from path: {path}")
                continue
            if expected is not None:
                predictions = record.get("predictions", {})
                if not isinstance(predictions, dict) or set(predictions) - set(expected):
                    if strict:
                        raise ValueError(f"Unexpected prediction in sequence state: {path}")
                    continue
                if (record.get("expected_predictions") != list(expected)
                        or (record.get("status") == "complete" and set(predictions) != set(expected))):
                    record["status"] = "partial"
                    record["error"] = "requested predictions are incomplete; rerun validate"
                record["expected_predictions"] = list(expected)
                if any(not prediction_metadata_valid(key, value, method)
                       for key, value in predictions.items()):
                    if strict:
                        raise ValueError(f"Malformed prediction metadata: {path}")
                    continue
            if record.get("status") == "complete" and not required_scores_valid(
                record, plan.get("config", {}).get("scorer", {}), method
            ):
                record["status"] = "partial"
                record["error"] = "requested PartiScore stage is missing or failed; rerun validate"
            if method == "vina_dock":
                if current_entry and current_entry.get("path"):
                    record["input_file"] = current_entry["path"]
                gate_error = _vina_postprocess_error(
                    record,
                    current_entry,
                    inventory_error,
                    sample=sample,
                    task_dir=task_dir,
                )
                if gate_error and record.get("status") != "not_applicable":
                    _invalidate_vina_postprocess(record, gate_error)
            yield path, record


def _publish_success(root: Path, records: list[tuple[Path, dict]]) -> dict[str, str]:
    """Publish only successful sequence JSON and refold structures."""
    success_root = root / "success"
    refolds = (root / "refolds").resolve()
    desired: dict[str, Path] = {}
    for path, record in records:
        if not any(d.get("status") == "passed" for d in record.get("decisions", {}).values()):
            continue
        sample = str(record.get("sample"))
        desired[str(Path("success") / sample / path.name)] = path
        for prediction in record.get("predictions", {}).values():
            target = prediction.get("structure_path")
            if not target:
                continue
            target = Path(target).resolve()
            if target.is_relative_to(refolds) and target.is_file():
                desired[str(Path("success") / sample / target.name)] = target
    if success_root.is_dir():
        for path in sorted(success_root.rglob("*"), reverse=True):
            if path.is_symlink():
                try:
                    if path.resolve().is_relative_to(refolds):
                        path.unlink()
                except OSError:
                    path.unlink(missing_ok=True)
    for relative, target in desired.items():
        path = root / relative
        ensure_directory(path.parent)
        if path.exists() or path.is_symlink():
            if path.is_symlink() and path.resolve() == target:
                continue
            if path.is_symlink():
                path.unlink()
            else:
                raise ValueError(f"Refusing to replace unrelated success artifact: {path}")
        path.symlink_to(os.path.relpath(target, path.parent))
    return {str(target): relative for relative, target in desired.items()}


def rebuild(root: Path, criteria: list[str], presets: dict, *, validate_artifacts=False, persist=False) -> dict:
    """Rebuild reports from saved metrics; ``validate_artifacts`` is obsolete."""
    del validate_artifacts
    with locked(root / ".state" / "reports.lock"):
        plan = read_json(root / ".state" / "plan.json")
        if not plan:
            raise ValueError(f"No saved validation plan: {root}")
        unfiltered = plan.get("config", {}).get("unfiltered_success") is True and not criteria
        cohorts = ["unfiltered"] if unfiltered else criteria
        records = []
        for path, record in records_for_plan(root, plan, strict=persist):
            apply_presets(record, criteria, presets, unfiltered=unfiltered)
            if persist:
                for decision in record.get("decisions", {}).values():
                    if isinstance(decision, dict):
                        decision.pop("definition", None)
                metadata = record.get("input_metadata")
                if isinstance(metadata, dict):
                    metadata.pop("row", None)
                    metadata.pop("metadata_states", None)
                    metadata.pop("chemistry_file", None)
                atomic_json(path, record)
            records.append((path, record))
        rows = [summarize_record(record, plan.get("samples", {}).get(record.get("sample"), {}).get("row"))
                for _, record in records]
        rows.sort(key=lambda row: (str(row.get("sample")), int(row.get("seq_idx") or 0)))
        base = ["run", "task", "sample", "input_file", "method", "seq_idx", "sequence_source",
                "source_index", "sequences", "status", "error"]
        atomic_csv(root / "metrics.csv", rows, base)
        links = _publish_success(root, records)
        success_rows = [{key: links.get(value, value) if isinstance(value, str) else value
                         for key, value in row.items()} for row in rows if row["passing_criteria"]]
        atomic_csv(root / "success.csv", success_rows, base)
        counts = {}
        for name in cohorts:
            decisions = [(r.get("sample"), r.get("decisions", {}).get(name, {}).get("status"))
                         for _, r in records]
            counts[name] = {
                "passed_sequences": sum(s == "passed" for _, s in decisions),
                "passed_structures": len({sample for sample, s in decisions if s == "passed"}),
                "evaluated_sequences": sum(s in {"passed", "failed"} for _, s in decisions),
                "unavailable_sequences": sum(s == "unavailable" for _, s in decisions),
            }
        complete = sum(r.get("status") == "complete" for _, r in records)
        planned_structures = plan.get("generation_planned", len(plan["samples"]))
        source_map = sequence_sources_by_sample(plan)
        recorded_slots = sum(
            sequence_slot_count(plan, sample, source_map)
            for sample in plan.get("samples", {})
        )
        planned = recorded_slots + max(
            0, planned_structures - len(plan.get("samples", {}))
        ) * plan.get("sequence_count", 0)
        summary = {
            "planned_structures": planned_structures,
            "planned_sequences": planned,
            "recorded_sequences": len(rows),
            "not_applicable_sequences": sum(r.get("status") == "not_applicable" for _, r in records),
            "complete_sequences": complete,
            "incomplete_sequences": planned - complete - sum(
                r.get("status") == "not_applicable" for _, r in records
            ),
            "criteria": counts,
        }
        summary.update(benchmark_summary(plan, records, cohorts, planned_structures))
        summary.update(grouped_summary(plan, records, cohorts, planned_structures))
        if plan.get("method") == "vina_dock":
            valid = sum(
                record.get("status") == "complete"
                and record.get("predictions", {}).get("original", {}).get("metrics", {}).get("rdkit_valid") is True
                and record.get("predictions", {}).get("original", {}).get("metrics", {}).get("docking_success") is True
                for _, record in records)
            passed = sum(record.get("decisions", {}).get("odesign_denovo_ligand", {}).get("status") == "passed"
                         for _, record in records)
            summary["denovo_ligand"] = {
                "planned_molecules": planned_structures, "valid_molecules": valid,
                "valid_rate_all_planned": valid / planned_structures if planned_structures else None,
                "successful_molecules": passed,
                "success_rate_all_planned": passed / planned_structures if planned_structures else None,
                "success_rate_valid_conditional": passed / valid if valid else None,
            }
        atomic_json(root / "summary.json", summary)
        return summary


def benchmark_summary(plan, records, criteria, planned_structures):
    """All-planned rates; completed-only rates are labeled conditional."""
    from collections import Counter
    source_map = sequence_sources_by_sample(plan)
    branch_specs = {}
    branch_structures = Counter()
    if source_map is None:
        for spec in plan.get("config", {}).get("sequence", {}).get("sources", []):
            label = spec.get("label", "self" if spec["source"] == "gt" else spec["source"])
            branch_specs[label] = spec["count"]
        branch_structures.update({label: planned_structures for label in branch_specs})
    else:
        for sample, specs in source_map.items():
            labels_seen = set()
            for spec in specs:
                label = spec.get("label", "self" if spec["source"] == "gt" else spec["source"])
                branch_specs[label] = branch_specs.get(label, 0) + spec["count"]
                labels_seen.add(label)
            for label in labels_seen:
                branch_structures[label] += 1
        # A uniform implicit source plan can account for unrecorded generated
        # rows without guessing a modality from completed records. Mixed
        # modality plans leave those rows unassigned until their metadata
        # exists, while preserving their overall denominator.
        missing = max(0, planned_structures - len(plan.get("samples", {})))
        signatures = {
            tuple(sorted((spec.get("label", "self" if spec["source"] == "gt" else spec["source"]), spec["count"])
                         for spec in specs))
            for specs in source_map.values()
        }
        if missing and len(signatures) == 1:
            for label, count in signatures.pop():
                branch_specs[label] = branch_specs.get(label, 0) + missing * count
                branch_structures[label] += missing
    if not branch_specs:
        branch_specs = {"legacy": plan["sequence_count"]}
        branch_structures["legacy"] = planned_structures
    def branch(record):
        sequence = record.get("sequence", {})
        return sequence.get("sequence_branch", "self" if sequence.get("source") == "gt" else sequence.get("source"))
    branches = {}
    for label, budget in branch_specs.items():
        chosen = [r for _, r in records if branch(r) == label]
        planned = budget if source_map is not None else planned_structures * budget
        entry = {"planned_sequences": planned, "recorded_sequences": len(chosen),
                 "complete_sequences": sum(r.get("status") == "complete" for r in chosen), "criteria": {}}
        planned_backbones = branch_structures.get(label, planned_structures)
        for name in criteria:
            passed = [r for r in chosen if r.get("decisions", {}).get(name, {}).get("status") == "passed"]
            evaluated = sum(r.get("decisions", {}).get(name, {}).get("status") in {"passed", "failed"} for r in chosen)
            entry["criteria"][name] = {"passed_sequences": len(passed),
                "rate_all_planned_sequences": len(passed) / planned if planned else None,
                "passed_backbones": len({r["sample"] for r in passed}),
                "rate_all_planned_backbones": len({r["sample"] for r in passed}) / planned_backbones if planned_backbones else None,
                "rate_evaluated_sequences_conditional": len(passed) / evaluated if evaluated else None}
        branches[label] = entry
    combined = {}
    for name in criteria:
        passed = [r for _, r in records if r.get("decisions", {}).get(name, {}).get("status") == "passed"]
        combined[name] = {"passed_backbones": len({r["sample"] for r in passed}),
            "rate_all_planned_backbones": len({r["sample"] for r in passed}) / planned_structures if planned_structures else None}
    stages = Counter()
    stages["generation_missing_rows"] = max(0, planned_structures - len(plan["samples"]))
    planned_slots = sum(
        sequence_slot_count(plan, sample, source_map)
        for sample in plan.get("samples", {})
    ) + max(0, planned_structures - len(plan.get("samples", {}))) * plan.get("sequence_count", 0)
    stages["unrecorded_sequence_slots"] = max(0, planned_slots - len(records))
    for _, record in records:
        if record.get("status") == "complete":
            continue
        if record.get("status") in {"pending", "running"} and not record.get("error"):
            stages[record["status"] + "_sequences"] += 1
            continue
        stage = record.get("failure_stage")
        if not stage:
            predictions = list(record.get("predictions", {}).values())
            stage = next((p.get("error_stage") for p in predictions if p.get("error_stage")), None)
        if not stage:
            error = str(record.get("error", ""))
            if "postprocess:" in error or "postprocessed structure" in error:
                stage = "postprocess"
            elif "AF3 MSA search failed" in error:
                stage = "msa_search"
            elif record.get("sequence", {}).get("sequences") and any(
                name in error for name in (
                    "AF3BackendError", "AF3WorkerError", "AF3ValidationError",
                    "AF3InputError", "AF3OutputError",
                )
            ):
                # Legacy rows saved the sequence before the native backend
                # failed, but did not persist the refold producer stage.
                stage = "refold"
            elif not record.get("predictions"):
                stage = "mpnn" if record.get("sequence", {}).get("source") != "gt" else "generation_or_input"
            elif any(prediction.get("status") != "complete"
                     for prediction in record["predictions"].values()):
                stage = "refold"
            elif any(prediction.get("metrics_status") != "complete"
                     for prediction in record["predictions"].values()):
                stage = "metrics"
            elif not required_scores_valid(record, plan.get("config", {}).get("scorer", {}), plan.get("method")):
                stage = "partiscore"
            else:
                stage = "refold"
        stages[stage + "_sequences"] += 1
    hist = {"0": 0, "1-2": 0, "3-9": 0, ">=10": 0}
    normalized, clashes = [], []
    for _, record in records:
        for prediction in record.get("predictions", {}).values():
            value = prediction.get("metrics", {}).get("loose_clash_count")
            if isinstance(value, (int, float)) and math.isfinite(value):
                clashes.append(value)
                key = "0" if value == 0 else "1-2" if value <= 2 else "3-9" if value < 10 else ">=10"
                hist[key] += 1
                norm = prediction["metrics"].get("loose_clash_per_1000_design_heavy")
                if _scalar(norm):
                    normalized.append(norm)
    ame = {}
    for name in criteria:
        if name == "af3_ame":
            ordinary = [r for _, r in records if r.get("sequence", {}).get("source") != "gt"
                        and branch(r) != "self_fi"
                        and not (r.get("sequence", {}).get("settings", {}).get("fixed_interface") or {}).get("enabled")]
            for pool, chosen in (("first1", [r for r in ordinary if r["sequence"].get("source_index") == 0]), ("best8", ordinary)):
                passed = {r["sample"] for r in chosen if r.get("decisions", {}).get(name, {}).get("status") == "passed"}
                ame[pool] = {"passed_backbones": len(passed),
                            "rate_all_planned_backbones": len(passed) / planned_structures if planned_structures else None}
    return {"sequence_branches": branches, "combined_budget": combined, "failure_stages": dict(stages),
            "ame_ordinary_pool": ame,
            "loose_clashes": {"evaluated_predictions": len(clashes), "histogram": hist,
                             "le2_rate_evaluated_predictions_conditional": sum(c <= 2 for c in clashes) / len(clashes) if clashes else None,
                             "mean_per_1000_design_heavy_atoms": sum(normalized) / len(normalized) if normalized else None}}


def grouped_summary(plan, records, criteria, planned_structures):
    from collections import Counter
    from .planning import row_length_group
    budgets = plan.get("length_budgets")
    allocation = plan.get("length_budget_basis", "recorded_lengths_with_unallocated_failures")
    if not budgets:
        budgets = dict(Counter(row_length_group(e.get("row", {})) for e in plan["samples"].values()))
        budgets["unknown"] = budgets.get("unknown", 0) + max(0, planned_structures - len(plan["samples"]))
    groups = {}
    for length, planned in budgets.items():
        samples = {sample for sample, entry in plan["samples"].items() if row_length_group(entry.get("row", {})) == length}
        chosen = [record for _, record in records if record.get("sample") in samples]
        outcomes = {}
        for name in criteria:
            passed = {record["sample"] for record in chosen if record.get("decisions", {}).get(name, {}).get("status") == "passed"}
            outcomes[name] = {"passed_backbones": len(passed), "success_rate": len(passed) / planned if planned else None}
        groups[length] = {"planned_backbones": planned, "recorded_backbones": len(samples),
                          "recorded_sequences": len(chosen), "criteria": outcomes}
    return {"length_budget_basis": allocation, "length_groups": groups}
