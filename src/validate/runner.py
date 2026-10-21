"""Streaming validation worker with simple per-sequence recovery."""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile
from time import perf_counter
from typing import Any, Mapping

from .config import config_for_method
from .progress import emit

from .filters import apply_presets
from .planning import (
    expected_keys, prediction_metadata_valid, sequence_slot_count, sequence_sources_by_sample,
)
from .storage import (
    _msa_context_identity,
    atomic_json,
    begin_sequence,
    ensure_directory,
    locked,
    metrics_valid,
    prediction_valid,
    read_json,
    required_scores_valid,
    score_status_valid,
)


_UNRECOMPUTABLE_CONFIDENCE_METRICS = (
    "pae_interaction",
    "antibody_antigen_pae_angstrom",
    "ipae_angstrom",
    "ipae_normalized",
    "min_interchain_pae_angstrom",
    "min_design_target_pae_angstrom",
    "min_target_design_pae_angstrom",
    "design_target_pae_pairs",
)


class AF3MSAResumeError(ValueError):
    """An explicit refold-only rerun is required before changing MSA identity."""


def _fatal_infrastructure_error(exc: BaseException) -> bool:
    """Preserve a backend's fatal marker through exception wrappers."""
    pending = [exc]
    seen = set()
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        fatal = getattr(error, "fatal", None)
        if fatal is not None:
            if fatal:
                return True
            # A recovered request/output error is explicitly nonfatal, even
            # when Python retains the old service failure in its context.
            continue
        if error.__cause__ is not None:
            pending.append(error.__cause__)
        elif not error.__suppress_context__ and error.__context__ is not None:
            pending.append(error.__context__)
    return False


def _clear_failure_fields(value: dict) -> bool:
    fields = ("error", "error_stage", "failure_stage")
    changed = any(field in value for field in fields)
    for field in fields:
        value.pop(field, None)
    return changed


def _clear_completed_failure_fields(record: dict) -> bool:
    """Clear failure metadata after every requested stage has recovered."""
    changed = _clear_failure_fields(record)
    for prediction in record.get("predictions", {}).values():
        changed = _clear_failure_fields(prediction) or changed
        score = prediction.get("partiscore")
        if score_status_valid(score):
            changed = _clear_failure_fields(score) or changed
    for score in record.get("scores", {}).values():
        if score_status_valid(score):
            changed = _clear_failure_fields(score) or changed
    return changed


def sequence_slots(sources):
    index = 0
    for spec in sources:
        for source_index in range(spec["count"]):
            yield {"seq_index": index, "source": spec["source"], "source_index": source_index,
                   "sequence_branch": spec.get("label", "self" if spec["source"] == "gt" else spec["source"]),
                   "sequences": {}, "settings": spec.get("settings", {})}
            index += 1


def _compact_confidence(value):
    """Keep scalar confidence evidence and discard arrays from public JSON."""
    if isinstance(value, dict):
        return {str(key): compact for key, item in value.items()
                if (compact := _compact_confidence(item)) is not None}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "ndim") and getattr(value, "ndim", 1) == 0:
        try:
            return value.item()
        except (TypeError, ValueError):
            return None
    return None


def _clean_prediction(prediction: dict) -> None:
    """Remove worker-only arrays and legacy artifact fields in place."""
    prediction.pop("_confidence_arrays", None)
    prediction.pop("_confidence_chain_map", None)
    native_af3 = prediction.get("model") in {"af3", "af3-la"}
    for field in ("confidence_path", "artifacts"):
        prediction.pop(field, None)
    if not native_af3:
        prediction.pop("raw_confidence_path", None)
    if native_af3:
        for field in (
            "full_pae", "full_pde", "pae", "pde", "token_chain_ids", "native_token_chain_ids",
            "source_token_chain_ids", "token_res_ids", "source_atom_ledger", "product_atom_ledger",
            "restored_product_atom_ledger", "native_product_atom_ledger", "source_residue_map",
            "native_component_map", "source_component_map", "cif", "pdb", "cif_string", "pdb_string",
            "source_to_native_chain", "native_to_source_chain", "chain_ids", "native_chain_ids",
            "source_chain_ids", "chain_ptm", "chain_iptm", "chain_pair_iptm", "pair_iptm",
            "per_chain_plddt", "native_per_chain_plddt", "per_chain_token_count",
            "native_per_chain_token_count", "per_chain_atom_count", "native_per_chain_atom_count",
            "native_full_pae", "native_full_pde", "native_token_res_ids", "native_token_source_mask",
            "source_token_indices", "source_token_residue_map", "token_ins_codes",
            "native_per_chain_source_token_count", "pae_matrix", "pde_matrix", "confidence_chain_map",
        ):
            prediction.pop(field, None)
        prediction["metadata"] = {
            name: prediction.get("metadata", {}).get(name)
            for name in ("key", "protocol", "sample")
        }
    for field in ("source", "source_index", "cyclic"):
        prediction.pop(field, None)
    if isinstance(prediction.get("confidence"), dict):
        for field in ("native_metric_definition", "context_embedding_cache"):
            prediction["confidence"].pop(field, None)
        prediction["confidence"] = _compact_confidence(prediction["confidence"])


def _clean_record(record: dict) -> None:
    for prediction in record.get("predictions", {}).values():
        if isinstance(prediction, dict):
            _clean_prediction(prediction)
    for decision in record.get("decisions", {}).values():
        if isinstance(decision, dict):
            decision.pop("definition", None)
    if isinstance(record.get("sequence"), dict):
        record["sequence"] = _compact_sequence(record["sequence"])
    metadata = record.get("input_metadata")
    if isinstance(metadata, dict):
        metadata.pop("row", None)
        metadata.pop("metadata_states", None)
        metadata.pop("chemistry_file", None)


def _compact_sequence(variant: dict) -> dict:
    """Persist sequence provenance without backend configs or path ledgers."""
    sequence = {
        key: variant.get(key)
        for key in ("seq_index", "source", "source_index", "sequences", "sequence_branch", "label",
                    "interface_positions", "interface_target_chains", "interface_applicable")
        if variant.get(key) is not None
    }
    settings = variant.get("settings", {})
    if isinstance(settings, dict):
        keep = {"seed", "batch_seed", "batch_size", "temperature", "temp", "fixed_positions", "fixed_restypes",
                "bond_constraints", "metadata_states", "backend", "backend_source",
                "fixed_interface", "interface_positions", "interface_target_chains", "interface_applicable"}
        compact = {key: settings[key] for key in keep if key in settings}
        if compact:
            sequence["settings"] = compact
    fixed_interface = settings.get("fixed_interface") if isinstance(settings, dict) else None
    if isinstance(fixed_interface, dict) and fixed_interface.get("enabled"):
        targets = variant.get("interface_target_chains") or settings.get("interface_target_chains")
        known = variant.get("interface_target_chains") is not None or settings.get("interface_target_chains") is not None
        sequence["interface_applicable"] = bool(targets) if known else None
    return sequence


def _remove_temporary(output: Path, work: Path | None = None) -> None:
    """Delete only this sequence's worker scratch data."""
    paths = [output / name for name in (".inputs", "raw", "chemistry.json")]
    # A killed worker can leave its current refold scratch directory behind.
    # The sample lock ensures that no other worker is using these directories.
    paths.extend(path for path in output.glob(".seq*_*")
                 if path.name.split("_", 1)[0][4:].isdigit() and path.is_dir())
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
    if work is not None and work.exists() and not work.is_symlink():
        shutil.rmtree(work)


def _restore_persisted_only_metrics(
    previous_metrics: Mapping[str, Any],
    metrics: dict[str, Any],
    reasons: dict[str, str],
) -> None:
    """Retain values whose raw confidence arrays were intentionally discarded."""
    for name in _UNRECOMPUTABLE_CONFIDENCE_METRICS:
        if metrics.get(name) is None and previous_metrics.get(name) is not None:
            metrics[name] = previous_metrics[name]
            reasons.pop(name, None)
    if "partiscore" in previous_metrics:
        metrics["partiscore"] = previous_metrics["partiscore"]


def _completed_prediction(prediction: Mapping[str, Any]) -> bool:
    if prediction.get("model") in {"af3", "af3-la"}:
        structure = prediction.get("structure_path")
        return bool(structure and prediction_valid(dict(prediction), Path(str(structure)).parent))
    structure_path = prediction.get("structure_path")
    return (
        str(prediction.get("status", "")).lower() == "complete"
        and isinstance(structure_path, str)
        and Path(structure_path).is_file()
    )


def recompute_persisted_metrics(root: Path, plan: Mapping[str, Any]) -> dict[str, int]:
    """Re-evaluate saved structures without starting validation backends."""
    from .inputs import load_input
    from .metrics import evaluate_pair, evaluate_prediction

    method = plan.get("method")
    config = plan.get("config")
    samples = plan.get("samples")
    sequence_count = plan.get("sequence_count")
    task_dir = plan.get("task_dir")
    if (not isinstance(method, str) or not isinstance(config, Mapping)
            or not isinstance(samples, Mapping) or type(sequence_count) is not int
            or sequence_count < 0 or not isinstance(task_dir, str)):
        raise ValueError(f"Malformed saved validation plan: {root}")
    expected = expected_keys(method, config)
    source_map = sequence_sources_by_sample(plan)
    recomputed_predictions = skipped_predictions = recomputed_pairs = updated_records = 0
    for sample, entry in samples.items():
        if not isinstance(sample, str) or Path(sample).name != sample or sample in {".", ".."}:
            raise ValueError("Invalid sample in saved plan")
        if not isinstance(entry, Mapping):
            raise ValueError(f"Malformed saved plan sample: {sample}")
        path_value, row = entry.get("path"), entry.get("row")
        if not isinstance(path_value, str) or not isinstance(row, Mapping):
            raise ValueError(f"Malformed saved plan input: {sample}")
        input_data = load_input(Path(path_value), dict(row), Path(task_dir))
        if input_data.reasons:
            raise ValueError("Unresolved input metadata: " + "; ".join(input_data.reasons))
        if input_data.sample_id != sample:
            raise ValueError("Loaded input sample differs from saved validation plan")
        for seq_index in range(sequence_slot_count(plan, sample, source_map)):
            record_path = root / "refolds" / sample / f"seq{seq_index}.json"
            record = read_json(record_path)
            if record is None:
                continue
            predictions = record.get("predictions")
            sequence = record.get("sequence")
            if not isinstance(predictions, dict) or not isinstance(sequence, Mapping):
                raise ValueError(f"Malformed sequence state: {record_path}")
            if set(predictions) - set(expected):
                raise ValueError(f"Unexpected prediction in sequence state: {record_path}")
            changed = False
            for key in expected:
                prediction = predictions.get(key)
                if not isinstance(prediction, dict):
                    if prediction is not None:
                        raise ValueError(f"Malformed prediction in sequence state: {record_path}")
                    continue
                if not prediction_metadata_valid(key, prediction, method):
                    raise ValueError(f"Malformed prediction metadata: {record_path}")
                if not _completed_prediction(prediction):
                    skipped_predictions += 1
                    continue
                previous_metrics = prediction.get("metrics", {})
                if not isinstance(previous_metrics, Mapping):
                    raise ValueError(f"Malformed saved metrics: {record_path}")
                if method in {"af3", "af3-la"}:
                    from .af3 import normalize_af3_confidence
                    prediction.update(normalize_af3_confidence(
                        prediction, input_data.design_chains,
                        antibody_chains=tuple(filter(None, (input_data.row.get("heavy_chain"), input_data.row.get("light_chain")))),
                    ))
                evaluated = evaluate_prediction(input_data, sequence, prediction)
                metrics = dict(evaluated["metrics"])
                reasons = dict(evaluated.get("reasons", {}))
                _restore_persisted_only_metrics(previous_metrics, metrics, reasons)
                prediction.update(
                    metrics=metrics,
                    reasons=reasons,
                    metric_provenance=dict(evaluated.get("provenance", {})),
                    metrics_status="complete",
                )
                recomputed_predictions += 1
                changed = True
            if method == "af2_ig":
                paired_metrics = record.setdefault("paired_metrics", {})
                paired_reasons = record.setdefault("paired_reasons", {})
                if not isinstance(paired_metrics, dict) or not isinstance(paired_reasons, dict):
                    raise ValueError(f"Malformed paired metrics in sequence state: {record_path}")
                for model in config.get("af2", {}).get("models", []):
                    for sample_index in range(config.get("af2", {}).get("samples", 0)):
                        suffix = f"model{model}_sample{sample_index}"
                        complex_prediction = predictions.get(f"complex_{suffix}")
                        monomer_prediction = predictions.get(f"monomer_{suffix}")
                        if not (isinstance(complex_prediction, Mapping) and isinstance(monomer_prediction, Mapping)
                                and _completed_prediction(complex_prediction)
                                and _completed_prediction(monomer_prediction)):
                            continue
                        paired = evaluate_pair(input_data, sequence, complex_prediction, monomer_prediction)
                        paired_metrics[suffix] = dict(paired["metrics"])
                        paired_reasons[suffix] = dict(paired.get("reasons", {}))
                        recomputed_pairs += 1
                        changed = True
            if changed:
                _clean_record(record)
                atomic_json(record_path, record)
                updated_records += 1
    return {
        "recomputed_predictions": recomputed_predictions,
        "skipped_predictions": skipped_predictions,
        "recomputed_pairs": recomputed_pairs,
        "updated_records": updated_records,
    }


class ValidationWorker:
    """Ray reserves configured resources for one resident backend per worker."""

    def __init__(self, repo_root, method, config, criteria, presets, device="cuda:0"):
        self.repo = Path(repo_root)
        self.method, self.cfg = method, config_for_method(config, method)
        self.criteria, self.presets, self.device = criteria, presets, device
        self.backend = self.generator = self.scorer = None
        self.original_scores = {}
        self._prepared_tickets = set()
        self.log_dir = None
        self.log_context = {}
        self._msa_context = None

    def _models(self):
        if self.backend is None:
            if self.method == "af2_ig":
                from .af2 import AF2Backend
                self.backend = AF2Backend(self.repo, self.cfg["af2"], self.device)
            elif self.method in {"esmfold2-fast", "esmfold2-standard"}:
                from .esmfold2 import ESMFold2Backend
                self.backend = ESMFold2Backend(self.repo, self.cfg["esmfold2"], self.device)
            elif self.method in {"af3", "af3-la"}:
                from .af3 import AF3Backend, AF3Validator
                self.backend = AF3Validator(
                    AF3Backend(self.repo, self.cfg["af3"], method=self.method, device=self.device),
                    require_structure=True,
                )
        if self.backend is not None and self.log_dir is not None:
            setter = getattr(self.backend, "set_log_dir", None)
            if setter is not None:
                setter(self.log_dir)
        return self.backend

    def _log(self, phase, **fields):
        path = self.log_dir / f"worker_{os.getpid()}.log" if self.log_dir else None
        emit({**self.log_context, "phase": phase, **fields}, path)

    def _score(self, path, input):
        if self.scorer is None:
            from .scoring import ScoreBackend
            self.scorer = ScoreBackend(self.repo, self.cfg["scorer"], self.device)
        return self.scorer.score(path, input.design_chains)

    def _sources_for_input(self, input=None):
        """Resolve validation sources without mutating the shared config.

        The parser marks an authored source list explicitly.  A few callers
        construct a worker config directly, so an unmarked list that differs
        from the parser default is treated as explicit as well.
        """
        from .sequences import default_validation_sources, normalize_sequence_sources

        configured = self.cfg["sequence"]["sources"]
        if input is None:
            return normalize_sequence_sources(configured, default_gt=False)
        marker = self.cfg.get("_sequence_sources_explicit")
        if not marker:
            from .config import DEFAULTS

            default_sources = normalize_sequence_sources(
                DEFAULTS["sequence"]["sources"], default_gt=False
            )
            explicit = normalize_sequence_sources(configured, default_gt=False) != default_sources
        else:
            explicit = True
        return default_validation_sources(
            input,
            method=self.method,
            configured_sources=configured,
            explicit=explicit,
        )

    @staticmethod
    def _saved_variant(path: Path, slot: dict) -> dict | None:
        try:
            record = read_json(path)
        except ValueError:
            return None
        sequence = record.get("sequence") if record else None
        if not isinstance(sequence, dict):
            return None
        if sequence.get("seq_index") != slot["seq_index"]:
            return None
        if (sequence.get("source"), sequence.get("source_index")) != (slot["source"], slot["source_index"]):
            return None
        if sequence.get("sequence_branch", "self" if sequence.get("source") == "gt" else sequence.get("source")) != slot.get("sequence_branch"):
            return None
        if not isinstance(sequence.get("sequences"), dict) or not sequence["sequences"]:
            return None
        return dict(sequence, status="complete")

    def _variants(self, input, plan, sample, root, sources=None):
        from .sequences import SequenceGenerator

        if self.method == "vina_dock":
            yield {"seq_index": 0, "source": "gt", "source_index": 0,
                   "sequence_branch": "self", "status": "complete",
                   "sequences": {chain: entry["smiles"] for chain, entry in input.ligand_entries.items()}}
            return
        sample_root = root / "refolds" / sample
        sources = self._sources_for_input(input) if sources is None else sources
        slots = list(sequence_slots(sources))
        existing = []
        for slot in (() if self.cfg.get("rerun") else slots):
            saved = self._saved_variant(sample_root / f"seq{slot['seq_index']}.json", slot)
            if saved is not None:
                existing.append(saved)
        known = {int(v["seq_index"]): v for v in existing}
        if self.cfg.get("rerun_refold") and len(known) != len(slots):
            raise ValueError("--rerun-refold requires all requested sequence slots to exist; it never generates new MPNN sequences")
        for variant in existing:
            yield variant
        work = root / ".state" / "mpnn" / sample
        _remove_temporary(sample_root, work)
        if len(known) == len(slots):
            return
        self._log("sequence_generation", sample=sample)
        if self.generator is None:
            self.generator = SequenceGenerator(self.repo, self.cfg, self.device)
        generator = self.generator.generate_batches(
            input, sources, work, existing=existing
        )
        try:
            generation_started = perf_counter()
            for batch in generator:
                self._log("sequence_generation_finished", sample=sample,
                          count=len(batch), seconds=round(perf_counter() - generation_started, 6))
                pending = []
                for variant in batch:
                    index = int(variant["seq_index"])
                    if index in known:
                        continue
                    known[index] = variant
                    path = sample_root / f"seq{index}.json"
                    record = begin_sequence(path, self._context(plan, sample, plan["samples"][sample]),
                                            variant, expected_keys(self.method, self.cfg), force=True)
                    self._save(path, record)
                    pending.append(variant)
                # Every draw is durable before refolding the first member.
                _remove_temporary(sample_root, work)
                yield from pending
                generation_started = perf_counter()
        finally:
            close = getattr(generator, "close", None)
            if close is not None:
                close()
            _remove_temporary(root / "refolds" / sample, work)

    def _context(self, plan, sample, entry):
        task = Path(plan["task_dir"])
        context = {
            "run": task.parent.name,
            "task": task.name,
            "sample": sample,
            "method": self.method,
            "input_file": entry["path"],
            "structure_stage": self.cfg["vina"]["structure_stage"] if self.method == "vina_dock" else self.cfg.get("structure_stage", "base"),
        }
        # Recovery can replace a structure without changing its final rank.
        # Reuse the existing stat-based preparation contract to invalidate its
        # saved predictions; no content hash or extra output file is needed.
        try:
            stat = Path(entry["path"]).stat()
        except OSError:
            pass  # Missing inputs are reported by the normal input loader.
        else:
            context.update(input_size=stat.st_size, input_mtime_ns=stat.st_mtime_ns)
        if self.method in {"af3", "af3-la"} and self._msa_context is not None:
            context["msa_context"] = self._msa_context
        return context

    def _save(self, path, record):
        _clean_record(record)
        apply_presets(record, self.criteria, self.presets,
                      unfiltered=self.cfg.get("unfiltered_success", False))
        _clean_record(record)
        atomic_json(path, record)

    def _metrics(self, input, variant, prediction):
        from .metrics import evaluate_prediction

        started = perf_counter()
        try:
            if self.method in {"af3", "af3-la"}:
                from .af3 import normalize_af3_confidence
                prediction.update(normalize_af3_confidence(
                    prediction, input.design_chains,
                    antibody_chains=tuple(filter(None, (input.row.get("heavy_chain"), input.row.get("light_chain")))),
                ))
            evaluated = evaluate_prediction(input, variant, prediction)
            prediction.update(metrics=evaluated["metrics"], reasons=evaluated.get("reasons", {}),
                              metric_provenance=evaluated.get("provenance", {}),
                              metrics_status="complete")
            _clear_failure_fields(prediction)
        except Exception as exc:
            if _fatal_infrastructure_error(exc):
                raise
            error = f"{type(exc).__name__}: {exc}"
            prediction.update(metrics={}, reasons={"evaluation": error},
                              metrics_status="failed", error=error, error_stage="metrics")
        self._log("metrics_finished", seq_index=variant["seq_index"],
                  seconds=round(perf_counter() - started, 6), status=prediction["metrics_status"])
        score_required = self.method == "partiscore" or (
            self.cfg["scorer"]["enabled"] and self.cfg["scorer"]["refold"]
        )
        if score_required and not score_status_valid(prediction.get("partiscore")):
            try:
                score = self._score(prediction["structure_path"], input)
                prediction["partiscore"] = score
            except Exception as exc:
                if _fatal_infrastructure_error(exc):
                    raise
                prediction["partiscore"] = {"status": "failed", "error": str(exc),
                                           "error_stage": "partiscore"}
        if score_required:
            score = prediction.get("partiscore")
            if score_status_valid(score):
                _clear_failure_fields(score)
            else:
                error = (str(score.get("error") or "PartiScore failed")
                         if isinstance(score, dict) else "PartiScore failed")
                if isinstance(score, dict):
                    score.update(error=error, error_stage="partiscore")
                if metrics_valid(prediction):
                    prediction.update(error=error, error_stage="partiscore")
        if score_required and isinstance(prediction.get("metrics"), dict):
            score = prediction.get("partiscore")
            prediction["metrics"]["partiscore"] = score.get("score") if score_status_valid(score) else None
        _clean_prediction(prediction)

    def _original_score(self, input):
        cache_key = str(input.path)
        cached = self.original_scores.get(cache_key)
        if score_status_valid(cached):
            _clear_failure_fields(cached)
            return cached
        score = self._score(input.path, input)
        if isinstance(score, dict):
            score = {**score, "structure_path": str(input.path), "sequence_source": "gt",
                     "description": "original input structure before sequence redesign"}
        if score_status_valid(score):
            _clear_failure_fields(score)
            self.original_scores[cache_key] = score
        elif isinstance(score, dict):
            score.update(error=score.get("error") or "Original PartiScore failed",
                         error_stage="partiscore")
        return score

    def _run_variant(self, input, variant, plan, sample, entry, root, prepared_inputs=None):
        output = ensure_directory(root / "refolds" / sample)
        _remove_temporary(output, root / ".state" / "mpnn" / sample)
        path = output / f"seq{variant['seq_index']}.json"
        expected = expected_keys(self.method, self.cfg)
        sequence = _compact_sequence(variant)
        context = self._context(plan, sample, entry)
        previous = read_json(path)
        if self.method in {"af3", "af3-la"} and previous and previous.get("predictions"):
            previous_msa = previous.get("msa_context") or {"mode": "off"}
            current_msa = context.get("msa_context", {"mode": "off"})
            if _msa_context_identity(previous_msa) != _msa_context_identity(current_msa) and not (self.cfg.get("rerun") or self.cfg.get("rerun_refold")):
                raise AF3MSAResumeError("Saved AF3 predictions use a different MSA cache/profile/generation. Use --rerun-refold to keep sequences and replace refolds, or --rerun to regenerate everything")
        record = begin_sequence(path, context, sequence, expected,
                                force=bool(self.cfg.get("rerun") or self.cfg.get("rerun_refold")))
        if record.get("status") == "complete":
            if required_scores_valid(record, self.cfg["scorer"], self.method):
                if _clear_completed_failure_fields(record):
                    self._save(path, record)
            self._log("sequence_skipped", sample=sample, seq_index=variant["seq_index"])
            return
        record["source_error"] = input.row.get("error") or None
        record["input_metadata"] = {"design_chains": list(input.design_chains),
                                    "cyclic_chains": list(input.cyclic_chains),
                                    "cyclic_offset_chains": list(getattr(input, "cyclic_offset_chains", ()))}
        if variant.get("status") == "failed":
            record.update(status="failed", error=variant.get("error", "sequence generation failed"),
                          failure_stage=variant.get("failure_stage") or variant.get("error_stage")
                          or ("generation_or_input" if variant.get("source") == "gt" else "mpnn"))
            self._save(path, record)
            _remove_temporary(output, root / ".state" / "mpnn" / sample)
            return
        record["status"] = "running"
        self._log("sequence_started", sample=sample, seq_index=variant["seq_index"], source=variant["source"])
        failure_stage = "refold"
        try:
            scorer = self.cfg["scorer"]
            if scorer["enabled"] and scorer["original"]:
                failure_stage = "partiscore"
                try:
                    record["scores"] = {"original": self._original_score(input)}
                except Exception as exc:
                    if _fatal_infrastructure_error(exc):
                        raise
                    record["scores"] = {"original": {"status": "failed", "error": str(exc),
                                                       "error_stage": "partiscore"}}
            failure_stage = "refold"
            if self.method == "vina_dock":
                from .ligand import evaluate_ligand_docking
                evaluated = evaluate_ligand_docking(input, config=self.cfg["vina"],
                    work_dir=root / ".state" / "vina" / sample)
                ok = evaluated["metrics"]["docking_success"]
                prediction = {"key": "original", "protocol": "complex", "model": "vina_dock",
                    "sample": 0, "status": "complete" if ok else "failed",
                    "structure_path": str(input.path), "confidence": {},
                    "metrics": evaluated["metrics"], "metrics_status": "complete",
                    "reasons": evaluated["reasons"], "metric_provenance": evaluated["provenance"]}
                if not ok:
                    prediction.update(error_stage=evaluated.get("error_stage"),
                        error="; ".join(evaluated["reasons"].values()))
                    record["failure_stage"] = evaluated.get("error_stage")
                record["predictions"]["original"] = prediction
            elif self.method == "partiscore":
                structure = variant.get("structure_path") or str(input.path)
                prediction = {
                    "key": "original", "protocol": "complex", "model": "partiscore", "sample": 0,
                    "status": "complete", "structure_path": str(structure), "confidence": {},
                    "metrics": {}, "partiscore": {"status": "scoring_pending"},
                }
                record["predictions"]["original"] = prediction
                failure_stage = "metrics"
                self._metrics(input, variant, prediction)
            else:
                refold_started = perf_counter()
                completed = set()
                for key, prediction in record["predictions"].items():
                    if (key in expected and prediction_metadata_valid(key, prediction, self.method)
                            and _completed_prediction(prediction)):
                        if not metrics_valid(prediction) or (scorer["enabled"] and scorer["refold"]
                                and not score_status_valid(prediction.get("partiscore"))):
                            failure_stage = "metrics"
                            self._metrics(input, variant, prediction)
                        completed.add(key)
                failure_stage = "refold"
                predictions = ([] if set(expected) <= completed else
                               self._models().predict(input, variant, output, completed,
                                   **({"prepared_inputs": prepared_inputs} if prepared_inputs is not None else {})))
                try:
                    for prediction in predictions:
                        self._log("refold_result", seq_index=variant["seq_index"],
                                  seconds=round(perf_counter() - refold_started, 6))
                        key = prediction["key"]
                        if key not in expected or not prediction_metadata_valid(key, prediction, self.method):
                            raise ValueError(f"Backend returned unexpected prediction metadata for {key}")
                        record["predictions"][key] = prediction
                        if prediction.get("status") == "complete":
                            failure_stage = "metrics"
                            self._metrics(input, variant, prediction)
                        else:
                            prediction["error_stage"] = prediction.get("error_stage") or "refold"
                            _clean_prediction(prediction)
                        failure_stage = "refold"
                        refold_started = perf_counter()
                finally:
                    close = getattr(predictions, "close", None)
                    if close is not None:
                        close()
            if self.method == "af2_ig":
                from .metrics import evaluate_pair
                failure_stage = "metrics"
                for model in self.cfg["af2"]["models"]:
                    for sample_index in range(self.cfg["af2"]["samples"]):
                        suffix = f"model{model}_sample{sample_index}"
                        complex_p = record["predictions"].get(f"complex_{suffix}", {})
                        monomer_p = record["predictions"].get(f"monomer_{suffix}", {})
                        if complex_p.get("status") == monomer_p.get("status") == "complete":
                            pair = evaluate_pair(input, variant, complex_p, monomer_p)
                            record["paired_metrics"][suffix] = pair["metrics"]
                            record["paired_reasons"][suffix] = pair.get("reasons", {})
            record["status"] = "complete" if all(
                record["predictions"].get(key, {}).get("status") == "complete"
                and metrics_valid(record["predictions"][key]) for key in expected
            ) else "partial"
            if not required_scores_valid(record, scorer, self.method):
                record["status"] = "partial"
            if record["status"] == "complete":
                _clear_completed_failure_fields(record)
            else:
                for key in expected:
                    prediction = record["predictions"].get(key, {})
                    if prediction.get("status") != "complete" or not metrics_valid(prediction):
                        record.update(
                            failure_stage=prediction.get("error_stage") or (
                                "metrics" if prediction.get("status") == "complete" else "refold"),
                            error=prediction.get("error") or f"Incomplete prediction: {key}",
                        )
                        break
                else:
                    record.update(failure_stage="partiscore", error="Required PartiScore failed")
        except Exception as exc:
            record.update(status="partial" if record["predictions"] else "failed",
                          error=f"{type(exc).__name__}: {exc}", failure_stage=failure_stage)
            for prediction in record["predictions"].values():
                if isinstance(prediction, dict):
                    _clean_prediction(prediction)
            if _fatal_infrastructure_error(exc):
                raise
        finally:
            save_started = perf_counter()
            _remove_temporary(output, root / ".state" / "mpnn" / sample)
            self._save(path, record)
            self._log("save_finished", seq_index=variant["seq_index"],
                      seconds=round(perf_counter() - save_started, 6))
            self._log("sequence_finished", sample=sample, seq_index=variant["seq_index"], status=record["status"], error=record.get("error"))

    def prepare_sample(self, plan: dict, sample: str):
        """Persist exact requests and release the GPU actor while CPUs search."""
        from .inputs import load_input
        from src.design.rewards.af3_backend import prepare_af3_inputs
        from src.design.rewards.af3_input import msa_queries

        root = Path(plan["task_dir"]) / "validate" / self.method
        entry = plan["samples"][sample]
        self.log_dir = root / "logs"
        self.log_context = {"task": str(plan["task_dir"]), "method": self.method, "sample": sample}
        self._msa_context = None
        with locked(root / ".state" / f"{sample}.worker.lock"):
            if entry.get("error"):
                raise ValueError(entry["error"])
            input = load_input(Path(entry["path"]), entry["row"], Path(plan["task_dir"]))
            if input.reasons or input.sample_id != sample:
                raise ValueError("Invalid prepared validation input: " + "; ".join(input.reasons))
            if self.cfg.get("structure_stage") == "postprocessed":
                from .ligand import require_structure_stage
                require_structure_stage(input, "postprocessed")
            sources = self._sources_for_input(input)
            slots = list(sequence_slots(sources))
            variants, queries = [], {}
            for variant in self._variants(input, plan, sample, root, sources):
                index = variant["seq_index"]
                if type(index) is not int or not 0 <= index < len(slots) or any(v["variant"]["seq_index"] == index for v in variants):
                    raise ValueError("Invalid or duplicate prepared sequence index")
                if any(variant.get(key) != slots[index][key] for key in ("source", "source_index")):
                    raise ValueError("Prepared sequence differs from requested source slot")
                payloads = {} if variant.get("status") == "failed" else prepare_af3_inputs(input, variant, self.cfg["af3"], self.repo)
                for payload in payloads.values():
                    for query in msa_queries(payload, input.design_chains):
                        queries[(query["modality"], query["sequence"])] = {key: query[key] for key in ("modality", "sequence")}
                variants.append({"variant": variant, "payloads": payloads})
            if len(variants) != len(slots):
                raise ValueError("Missing prepared sequence variants")
            from .af3_msa import resolve_msa_settings
            cache_root = Path(resolve_msa_settings(self.cfg["af3"]["msa"], self.repo)["cache_dir"])
            prepared_dir = ensure_directory(cache_root / "prepared")
            fd, filename = tempfile.mkstemp(prefix="sample-", suffix=".json", dir=prepared_dir)
            os.close(fd)
            stat = input.path.stat()
            ticket = {"version": 1, "method": self.method, "sample": sample,
                      "task_dir": str(input.task_dir), "input_path": str(input.path),
                      "input_size": stat.st_size, "input_mtime_ns": stat.st_mtime_ns,
                      "row": entry["row"], "design_chains": list(input.design_chains),
                      "variants": variants}
            self._prepared_tickets.add(filename)
            try:
                atomic_json(Path(filename), ticket)
            except BaseException:
                Path(filename).unlink(missing_ok=True)
                self._prepared_tickets.discard(filename)
                raise
            self._log("af3_input_prepared", variants=len(variants), unique_queries=len(queries))
            return {"ticket": filename, "queries": list(queries.values())}

    def refold_prepared(self, plan: dict, sample: str, ticket_path: str, references: list[dict], search_error=None):
        try:
            return self._refold_ticket(plan, sample, ticket_path, references, search_error)
        finally:
            Path(ticket_path).unlink(missing_ok=True)
            self._prepared_tickets.discard(ticket_path)

    def _refold_ticket(self, plan: dict, sample: str, ticket_path: str, references: list[dict], search_error=None):
        """Resume the same hot backend using the exact persisted chemistry input."""
        from .inputs import load_input
        from src.design.rewards.af3_input import attach_msa_references, msa_queries

        root = Path(plan["task_dir"]) / "validate" / self.method
        entry = plan["samples"][sample]
        self.log_dir = root / "logs"
        self.log_context = {"task": str(plan["task_dir"]), "method": self.method, "sample": sample}
        path = Path(ticket_path)
        ticket = read_json(path)
        required_fields = {"version", "method", "sample", "task_dir", "input_path", "row",
                           "input_size", "input_mtime_ns", "design_chains", "variants"}
        if (not isinstance(ticket, Mapping) or not required_fields <= ticket.keys()
                or type(ticket["version"]) is not int or ticket["version"] != 1
                or not isinstance(ticket["row"], Mapping)
                or not isinstance(ticket["design_chains"], list)
                or any(not isinstance(chain, str) for chain in ticket["design_chains"])
                or not isinstance(ticket["variants"], list)
                or any(not isinstance(item, Mapping) or not isinstance(item.get("variant"), Mapping)
                       or not isinstance(item.get("payloads"), Mapping) for item in ticket["variants"])):
            raise ValueError("Invalid prepared AF3 ticket schema/version")
        if ticket["method"] != self.method or ticket["sample"] != sample or ticket["task_dir"] != plan["task_dir"]:
            raise ValueError("Prepared AF3 ticket identity differs from requested sample")
        input_path = Path(entry["path"])
        stat = input_path.stat()
        if str(input_path) != ticket["input_path"] or entry["row"] != ticket["row"] or (stat.st_size, stat.st_mtime_ns) != (ticket["input_size"], ticket["input_mtime_ns"]):
            raise ValueError("Validation source changed during MSA preparation")
        with locked(root / ".state" / f"{sample}.worker.lock"):
            input = load_input(input_path, entry["row"], Path(plan["task_dir"]))
            processed = 0
            for item in ticket["variants"]:
                variant = item["variant"]
                required = {(query["modality"], query["sequence"])
                            for payload in item["payloads"].values()
                            for query in msa_queries(payload, ticket["design_chains"])}
                failures = ([search_error] if isinstance(search_error, str) else
                            [failure["error"] for failure in (search_error or [])
                             if (failure["modality"], failure["sequence"]) in required])
                if failures:
                    sequence_path = root / "refolds" / sample / f"seq{variant['seq_index']}.json"
                    record = read_json(sequence_path)
                    # A failed search cannot invalidate previously published refolds,
                    # including during an explicit refold-only rerun.
                    if record and record.get("predictions"):
                        self._log("msa_search_failed_preserved", seq_index=variant["seq_index"], error="; ".join(failures))
                    else:
                        record = record or begin_sequence(sequence_path, self._context(plan, sample, entry), _compact_sequence(variant), expected_keys(self.method, self.cfg))
                        record.update(status="failed", error="AF3 MSA search failed: " + "; ".join(failures), failure_stage="msa_search")
                        self._save(sequence_path, record)
                else:
                    payloads = {protocol: attach_msa_references(payload, ticket["design_chains"], references)
                                for protocol, payload in item["payloads"].items()}
                    contexts = {protocol: payload["msa_context"] for protocol, payload in payloads.items()}
                    self._msa_context = {"protocols": contexts} if any(context["chains"] for context in contexts.values()) else None
                    self._run_variant(input, variant, plan, sample, entry, root, prepared_inputs=payloads)
                processed += 1
            self._log("sample_finished", processed_sequences=processed)
            return {"task": plan["task_dir"], "sample": sample, "processed_sequences": processed,
                    "msa_error": search_error}

    def process(self, plan: dict, sample: str):
        root = Path(plan["task_dir"]) / "validate" / self.method
        self.log_dir = root / "logs"
        self.log_context = {"task": str(plan["task_dir"]), "method": self.method, "sample": sample}
        self._msa_context = None
        with locked(root / ".state" / f"{sample}.worker.lock"):
            self._log("sample_started")
            result = self._process_sample(plan, sample)
            self._log("sample_finished", result=result)
            return result

    def _process_sample(self, plan: dict, sample: str):
        from .inputs import load_input

        root = Path(plan["task_dir"]) / "validate" / self.method
        entry = plan["samples"][sample]
        # Clean an interrupted attempt even if input parsing now fails.
        _remove_temporary(root / "refolds" / sample, root / ".state" / "mpnn" / sample)
        processed = set()
        source_map = sequence_sources_by_sample(plan)
        sources = (source_map[sample] if source_map is not None and sample in source_map
                   else self._sources_for_input(entry.get("row", {})))
        capability = entry.get("capability", {})
        if capability.get("status") in {"invalid", "not_applicable"}:
            for variant in sequence_slots(sources):
                path = root / "refolds" / sample / f"seq{variant['seq_index']}.json"
                record = begin_sequence(path, self._context(plan, sample, entry), variant,
                                        expected_keys(self.method, self.cfg))
                record.update(status="not_applicable" if capability["status"] == "not_applicable" else "failed",
                              error=capability["reason"], capability=capability,
                              failure_stage="generation_or_input",
                              source_error=entry["row"].get("error") or None,
                              input_metadata={})
                self._save(path, record)
            return {"task": plan["task_dir"], "sample": sample, "capability": capability}
        failure_stage = "generation_or_input"
        try:
            if entry.get("error"):
                raise ValueError(entry["error"])
            load_started = perf_counter()
            input = load_input(Path(entry["path"]), entry["row"], Path(plan["task_dir"]))
            stage = self.cfg["vina"]["structure_stage"] if self.method == "vina_dock" else self.cfg.get("structure_stage")
            if stage == "postprocessed":
                from .ligand import require_structure_stage
                # Check the CURRENT row before reusing any completed state.
                failure_stage = "postprocess"
                require_structure_stage(input, "postprocessed")
            failure_stage = "generation_or_input"
            self._log("input_loaded", seconds=round(perf_counter() - load_started, 6))
            if input.reasons:
                raise ValueError("Unresolved input metadata: " + "; ".join(input.reasons))
            if input.sample_id != sample:
                raise ValueError("Loaded input sample differs from result.csv")
            sources = self._sources_for_input(input)
            if self.method == "af2_ig":
                from .af2 import AF2CandidateError
                try:
                    backend = self._models()
                    validator = getattr(backend, "validate_input", backend.validate_capability)
                    validator(input)
                except AF2CandidateError as exc:
                    capability = {"status": "not_applicable", "reason": str(exc)}
                    for variant in sequence_slots(sources):
                        path = root / "refolds" / sample / f"seq{variant['seq_index']}.json"
                        record = begin_sequence(path, self._context(plan, sample, entry), variant,
                                                expected_keys(self.method, self.cfg))
                        record.update(status="not_applicable", error=str(exc), capability=capability,
                                      failure_stage="generation_or_input",
                                      input_metadata={})
                        self._save(path, record)
                    return {"task": plan["task_dir"], "sample": sample, "capability": capability}
            slots = list(sequence_slots(sources))
            failure_stage = "mpnn"
            for variant in self._variants(input, plan, sample, root, sources):
                index = variant.get("seq_index")
                if type(index) is not int or not 0 <= index < len(slots) or index in processed:
                    raise ValueError("Sequence generator returned duplicate or invalid sequence index")
                if any(variant.get(key) != slots[index][key] for key in ("source", "source_index")):
                    raise ValueError("Sequence generator provenance differs from the explicit request")
                failure_stage = "refold"
                self._run_variant(input, variant, plan, sample, entry, root)
                processed.add(index)
                failure_stage = "mpnn"
            if len(processed) != len(slots):
                raise ValueError("Sequence generator returned fewer variants than requested")
        except AF3MSAResumeError:
            raise
        except Exception as exc:
            if _fatal_infrastructure_error(exc):
                raise
            error = f"{type(exc).__name__}: {exc}"
            for variant in sequence_slots(sources):
                if variant["seq_index"] in processed:
                    continue
                path = root / "refolds" / sample / f"seq{variant['seq_index']}.json"
                saved = self._saved_variant(path, variant)
                record = begin_sequence(path, self._context(plan, sample, entry), saved or variant,
                                        expected_keys(self.method, self.cfg))
                record.update(status="failed", error=error, source_error=entry["row"].get("error") or None,
                              failure_stage=failure_stage,
                              input_metadata={})
                self._save(path, record)
        return {"task": plan["task_dir"], "sample": sample, "processed_sequences": len(processed)}

    def close(self):
        tickets = getattr(self, "_prepared_tickets", set())
        for ticket_path in tickets:
            Path(ticket_path).unlink(missing_ok=True)
        tickets.clear()
        first_error = None
        for model in (self.scorer, self.backend, self.generator):
            if model is not None and hasattr(model, "close"):
                try:
                    model.close()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
        if first_error is not None:
            raise first_error
