"""Retry OpenMM for existing result.csv outputs without loading an FM model."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import fcntl
import io
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from .context import preload_context_chemistry, clear_context_chemistry_cache
from .recovery_runner import RecoveryJob, run_recovery_postprocessing
from .runner import (PostprocessJob, PostprocessPreparationError,
                     build_parameterization_payload, preflight_postprocessing,
                     run_postprocessing)
from ..utils.reader import read_structure

_LOG = logging.getLogger(__name__)

def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical(value: Any) -> str:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _ensure_directory(path: Path) -> Path:
    path = Path(path)
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError(f"Managed output has symlink component: {item}")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def locked(path: Path, *, blocking=True):
    _ensure_directory(path.parent)
    if path.is_symlink():
        raise ValueError(f"Lock may not be a symlink: {path}")
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"Another validation owns {path.parent}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _atomic_text(path: Path, content: str) -> None:
    _ensure_directory(path.parent)
    if path.is_symlink():
        raise ValueError(f"Refusing to replace symlink: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_csv(path: Path, rows: list[dict], base_fields: list[str]) -> None:
    fields = list(
        dict.fromkeys(base_fields + sorted({key for row in rows for key in row}))
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: ""
                if value is None
                else _canonical(value)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
        )
    _atomic_text(path, buffer.getvalue())



def validate_refined_artifact(base, refined, payload=None):
    """Compare public identities, graph, and any explicitly fixed coordinates."""
    from .openmm.refine import atom_key
    before, after = read_structure(base), read_structure(refined)
    def inventory(atoms):
        values = {atom_key(atoms, row): str(atoms.element[row]).upper()
                  for row in range(len(atoms))}
        if len(values) != len(atoms):
            raise ValueError("refined artifact has duplicate atom identities")
        return values
    if inventory(before) != inventory(after):
        raise ValueError("refined artifact changed public atom identity or elements")
    def edges(atoms):
        if atoms.bonds is None:
            raise ValueError("refined artifact lacks a bond graph")
        return {(*sorted((atom_key(atoms, int(i)), atom_key(atoms, int(j)))), int(order))
                for i, j, order in atoms.bonds.as_array()}
    if edges(before) != edges(after):
        raise ValueError("refined artifact changed public bond endpoints or orders")
    if not np.isfinite(after.coord).all():
        raise ValueError("refined artifact has non-finite coordinates")
    if payload is not None:
        # Private ligand names can be rewritten to the public CIF component
        # names by the writer. Resolve those aliases against the emitted
        # structure before comparing motion ownership.
        if getattr(payload, "output_atom_aliases", ()):
            from .runner import payload_for_output
            payload = payload_for_output(payload, after)
        public_keys = set(inventory(before))
        moving = set()
        for name in ("native_moving_keys", "local_moving_keys"):
            values = getattr(payload, name, ()) or ()
            moving.update(values)
        peptide_keys = {
            key
            for record in (getattr(payload, "peptide_records", ()) or ())
            for key in (getattr(record, "heavy_keys", ()) or ())
        }
        moving.update(peptide_keys)
        cobalt_fixed = {
            key
            for group in (getattr(payload, "cobalt_hexammine_groups", ()) or ())
            for key in group
        }
        explicit_motion = bool(moving)
        if moving.difference(public_keys):
            raise ValueError("payload moving ownership is absent from the public structure")
        if cobalt_fixed.difference(public_keys):
            raise ValueError("payload cobalt ownership is absent from the public structure")
        fixed_keys = (public_keys - moving if explicit_motion else set()) | cobalt_fixed
        if fixed_keys:
            before_coords = {
                atom_key(before, row): np.asarray(before.coord[row], dtype=float)
                for row in range(len(before))
            }
            after_coords = {
                atom_key(after, row): np.asarray(after.coord[row], dtype=float)
                for row in range(len(after))
            }
            for key in sorted(fixed_keys):
                if not np.allclose(
                    before_coords[key], after_coords[key], rtol=0, atol=1.0e-4
                ):
                    raise ValueError(
                        f"refined artifact moved fixed-context atom {key}; "
                        "use --rerun to replace this staging file"
                    )
    return after


def _owned_file(directory, filename):
    if not filename or Path(filename).name != filename or Path(filename).suffix != ".cif":
        raise ValueError(f"invalid result filename: {filename!r}")
    path = directory / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"result structure is not a regular file: {path}")
    return path


def _clear_geometry_scores(row, antibody):
    for field in tuple(row):
        if (field.startswith("refined_partiscore") or field == "refined_filename"
                or field in {"final_partiscore", "score_rank"}
                or (antibody and field.startswith("partiscore"))):
            row[field] = ""
    if antibody:
        for field in ("framework_rmsd", "antigen_contacts", "antigen_clashes"):
            row[field] = ""


def _legacy_refined_filenames(path):
    """Return refined identities already represented by the legacy score CSV."""
    if path.is_symlink():
        raise ValueError(f"PartiScore CSV may not be a symlink: {path}")
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            row.get("filename", "")
            for row in csv.DictReader(stream)
            if row.get("filename") and row.get("source") == "refined"
        }


def _has_row_value(row, field):
    value = row.get(field, "")
    return value is not None and str(value).strip() != ""


def _has_final_score(row):
    value = row.get("final_partiscore")
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


_INTERFACE_FIELDS = ["filename", "source", "partner_a", "partner_b", "score",
                     "mean_log_density", "density_sum", "pair_count", "error"]


def _read_interfaces(directory):
    path = directory / "partiscore.csv"
    if path.is_symlink():
        raise ValueError("PartiScore CSV may not be a symlink")
    if not path.exists():
        return [], list(_INTERFACE_FIELDS)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    return rows, list(dict.fromkeys(_INTERFACE_FIELDS + fields))


def _publish_score_ranking(directory, rows, interfaces, fields, interface_fields,
                           *, clear=False):
    """Publish one complete task, rolling filenames back if a table fails."""
    from ..utils.score_ranking import finalize_score_ranking, _rename_owned_files

    values = [dict(row) for row in rows]
    if clear:
        for index, row in enumerate(values):
            row["_row_index"] = index
            row["final_partiscore"] = ""
    paths = [directory / "result.csv"]
    if interface_fields is not None:
        paths.append(directory / "partiscore.csv")
    previous = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"score-ranking table may not be a symlink: {path}")
        previous[path] = path.read_bytes().decode("utf-8") if path.exists() else None
    rename_map = {}
    try:
        ranked, updated_interfaces, rename_map = finalize_score_ranking(
            directory, values, interfaces)
        if clear:
            ranked.sort(key=lambda row: row["_row_index"])
            for row in ranked:
                original = rows[row.pop("_row_index")]
                row["final_partiscore"] = original.get("final_partiscore", "")
        atomic_csv(directory / "result.csv", ranked, fields)
        if interface_fields is not None:
            atomic_csv(directory / "partiscore.csv", updated_interfaces, interface_fields)
    except BaseException as error:
        try:
            if rename_map:
                _rename_owned_files(directory, {value: key for key, value in rename_map.items()})
            for path, content in previous.items():
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    _atomic_text(path, content)
        except BaseException as rollback_error:
            raise RuntimeError(
                f"recovery score-ranking publication failed ({error}); "
                f"rollback failed ({rollback_error})") from error
        raise
    return ranked, updated_interfaces


def _needs_task_ranking(directory, rows):
    from ..utils.score_ranking import finalize_score_ranking

    ranked, _interfaces, _renames = finalize_score_ranking(directory, rows, (), rename=False)
    identity = lambda row: (row.get("filename", ""), str(row.get("score_rank", "")))
    return [identity(row) for row in rows] != [identity(row) for row in ranked]


def _needs_refined_score(task, row, legacy_refined):
    """Identify a successful refined row whose scalar score is still missing."""
    if row.get("postprocess_status") != "succeeded":
        return False
    if _has_final_score(row):
        return False
    if getattr(task, "format", "normal") == "antibody":
        # Antibody scoring is stored in the public antigen/internal fields.
        # Status columns may be present in newer rows; score/pair-count/error
        # values cover the compact legacy antibody schema.
        for prefix in ("antigen", "internal"):
            evidence = (
                f"partiscore_{prefix}_status",
                f"partiscore_{prefix}",
                f"partiscore_{prefix}_pair_count",
                f"partiscore_{prefix}_error",
            )
            if not any(_has_row_value(row, field) for field in evidence):
                return True
        return False
    if _has_row_value(row, "refined_partiscore_score"):
        return False
    # A non-empty status, including no_contact/no_interface/error, is already
    # a durable scoring outcome and must not be retried as a missing score.
    if _has_row_value(row, "refined_partiscore_status"):
        return False
    return bool(row.get("filename")) and row["filename"] not in legacy_refined


def _row_design_chains(task, row):
    """Use persisted final design labels, with the task as a legacy fallback."""
    if getattr(task, "format", "normal") == "antibody":
        return tuple(
            row[field]
            for field in ("heavy_chain", "light_chain")
            if row.get(field)
        )
    try:
        values = json.loads(row.get("design_chains", "") or "[]")
    except (TypeError, ValueError):
        values = ()
    labels = tuple(
        str(item["chain"])
        for item in values
        if isinstance(item, dict) and item.get("chain")
    )
    if labels:
        return tuple(dict.fromkeys(labels))
    return tuple(str(chain.chain_id) for chain in getattr(task, "design_chains", ()))


def _select_score_only(runner, tables, root, selected):
    """Select score repair rows without adding them to the OpenMM job set."""
    selected_keys = {(task.name, row["filename"]) for task, row, _ in selected}
    score_only = []
    for task in runner.tasks:
        if not task.postprocess or not task.override or not task.score:
            continue
        rows = tables.get(task.name, ())
        if not rows:
            continue
        directory = root / task.name
        legacy = _legacy_refined_filenames(directory / "partiscore.csv")
        for row in rows:
            filename = row.get("filename", "")
            key = (task.name, filename)
            if key in selected_keys or not _needs_refined_score(task, row, legacy):
                continue
            base = _owned_file(directory, filename)
            score_only.append((task, row, base))
    return score_only


def recover_postprocessing(runner, *, limit=None, rerun=False, plan=False,
                           platform=None, charge_method=None):
    """Recover failed/pending existing samples, honoring task score and override."""
    from omegaconf import OmegaConf
    from .recovery import candidate_from_existing
    from ..core.records import DesignRunResult

    root = Path(str(runner.config.output_dir)).absolute()
    if any(item.is_symlink() for item in (root, *root.parents)) or not root.is_dir():
        raise ValueError("relax --out must be an existing regular inference directory")
    if limit is not None and limit < 1:
        raise ValueError("relax --limit must be positive")
    config = OmegaConf.to_container(runner.config.postprocess, resolve=True)
    if platform is not None:
        config["platform"] = platform.upper()
        config["use_ray"] = platform.upper() == "CUDA"
    if charge_method is not None:
        config["charge_method"] = charge_method
    selected = []
    tables = {}
    fields = {}
    for task in runner.tasks:
        if not task.postprocess or not task.override:
            continue
        directory = root / task.name
        path = directory / "result.csv"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"existing task result.csv is missing: {path}")
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields[task.name] = list(reader.fieldnames or ())
        if not {"filename", "postprocess_status"}.issubset(fields[task.name]):
            raise ValueError(f"result.csv lacks postprocess identity/status: {path}")
        names = [row["filename"] for row in rows if row["filename"]]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate sample identity in {path}")
        tables[task.name] = rows
        count = 0
        for row in rows:
            if not row["filename"] or (not rerun and row["postprocess_status"] == "succeeded"):
                continue
            base = _owned_file(directory, row["filename"])
            if limit is not None and count >= limit:
                break
            selected.append((task, row, base))
            count += 1
    if plan:
        return {"tasks": len(tables), "samples": len(selected),
                "platform": config["platform"], "output_dir": str(root)}
    managed = {task.name for task in runner.tasks if task.score
               and {"final_partiscore", "score_rank"}.issubset(fields.get(task.name, ()))}
    reused_final = set()
    for task in runner.tasks:
        if task.name not in managed:
            continue
        for row in tables[task.name]:
            if _has_final_score(row):
                continue
            value = (row.get("partiscore_antigen") if task.format == "antibody"
                     else row.get("refined_partiscore_score")
                     if row.get("postprocess_status") == "succeeded" else None)
            status = row.get("partiscore_antigen_status" if task.format == "antibody"
                             else "refined_partiscore_status")
            if status not in (None, "", "ok"):
                continue
            if _has_final_score({"final_partiscore": value}):
                row["final_partiscore"] = value
                reused_final.add(task.name)
    score_only = _select_score_only(runner, tables, root, selected)
    ranking_only = reused_final | {name for name in managed
                                  if _needs_task_ranking(root / name, tables[name])}
    if not selected and not score_only and not ranking_only:
        return DesignRunResult(root, 0, 0)
    recovery_cuda = (
        bool(config.get("use_ray", True))
        and str(config.get("platform", "CUDA")).upper() == "CUDA"
    )
    if selected:
        preflight_postprocessing(config)
    ccd_path = None if runner.config.ccd_path is None else Path(str(runner.config.ccd_path))
    dirty = set()
    changes = 0
    last_flush = time.monotonic()
    failed = 0
    completed = 0
    candidates = {}
    final_paths = {}
    jobs = []
    by_key = {}
    changed_geometry = set()
    interface_tables = {}
    interface_fields = {}
    interface_dirty = set()

    def flush(force=False):
        nonlocal changes, last_flush
        if not dirty or (not force and changes < 32 and time.monotonic() - last_flush < 2):
            return
        for name in sorted(dirty):
            atomic_csv(root / name / "result.csv", tables[name], fields[name])
            if name in interface_dirty:
                atomic_csv(root / name / "partiscore.csv", interface_tables[name], interface_fields[name])
        dirty.clear()
        interface_dirty.clear()
        changes = 0
        last_flush = time.monotonic()

    def record(task, row, status, message=""):
        nonlocal changes
        row["postprocess_status"] = status
        if task.format == "antibody":
            # The generated structure remains usable after a chemistry failure.
            row["status"] = "partial" if message else "ok"
            row["error_stage"] = "postprocess" if message else ""
            row["error"] = message
        else:
            # Replace only this stage's diagnostic and preserve other stages.
            errors = [part for part in row.get("error", "").split(" | ")
                      if part and not part.startswith("postprocess:")]
            if message:
                errors.append("postprocess: " + message)
            row["error"] = " | ".join(errors)
        dirty.add(task.name)
        changes += 1
        flush()

    def finish_geometry(task, row, path, candidate=None, antibody_context=None):
        _clear_geometry_scores(row, task.format == "antibody")
        dirty.add(task.name)
        changed_geometry.add((task.name, row["filename"]))
        if task.name in interface_tables:
            interface_tables[task.name] = [item for item in interface_tables[task.name]
                                          if item.get("filename") != row["filename"]]
            interface_dirty.add(task.name)
        if task.format == "antibody":
            from ..antibody.results import antibody_result_context, refresh_antibody_result
            context = antibody_context
            if context is None and candidate is not None:
                context = antibody_result_context(candidate)
            row.update(refresh_antibody_result(row, read_structure(path), context or {}))
        final_paths[(task.name, row["filename"])] = path

    def on_result(job, result):
        nonlocal completed, failed
        if result.key != job.key:
            raise RuntimeError("recovery callback violated sample ownership")
        task, row, base = by_key[job.key]
        path = base
        if result.succeeded:
            payload = getattr(result, "prepared_payload", None)
            if payload is None:
                payload = getattr(job, "payload", None)
            if payload is None:
                raise RuntimeError(
                    "successful recovery result omitted its prepared payload"
                )
            if getattr(payload, "key", None) != job.key:
                raise RuntimeError(
                    "successful recovery result prepared payload violated sample ownership"
                )
            try:
                validate_refined_artifact(base, job.refined_cif, payload)
            except (ValueError, RuntimeError) as error:
                failed += 1
                final_paths[job.key] = base
                record(
                    task,
                    row,
                    "failed",
                    f"{error}; use --rerun to replace this staging file",
                )
            else:
                os.replace(job.refined_cif, base)
                finish_geometry(
                    task, row, base, candidates.get(job.key),
                    getattr(result, "antibody_context", None),
                )
                completed += 1
                record(task, row, "succeeded", result.message)
        else:
            failed += 1
            final_paths[job.key] = base
            record(task, row, "failed", result.message)
        _LOG.info("%s/%s postprocess=%s initial_energy=%s final_energy=%s",
                  *job.key, row["postprocess_status"],
                  result.initial_energy_kj_mol, result.final_energy_kj_mol)

    # The lock prevents another recovery process from editing the same CSVs.
    # Existing generation runs are explicitly stopped before invoking recovery.
    with locked(root / ".relax.lock", blocking=False):
        try:
            ranking_tasks = ranking_only | {task.name for task, _row, _path in selected + score_only
                                            if task.name in managed}
            task_by_name = {task.name: task for task in runner.tasks}
            for name in sorted(ranking_tasks):
                directory = root / name
                task = task_by_name[name]
                if task.format != "antibody" or (directory / "partiscore.csv").exists():
                    interfaces, columns = _read_interfaces(directory)
                    interface_tables[name], interface_fields[name] = interfaces, columns
                cleared, interfaces = _publish_score_ranking(
                    directory, tables[name], interface_tables.get(name, ()), fields[name],
                    interface_fields.get(name), clear=True)
                # Keep selected row references live while changing their public names.
                for row, updated in zip(tables[name], cleared, strict=True):
                    row.clear()
                    row.update(updated)
                if name in interface_tables:
                    interface_tables[name] = interfaces
            selected = [(task, row, _owned_file(root / task.name, row["filename"])
                         if task.name in ranking_tasks else base)
                        for task, row, base in selected]
            score_only = [(task, row, _owned_file(root / task.name, row["filename"])
                           if task.name in ranking_tasks else base)
                          for task, row, base in score_only]
            submitted_jobs = 0
            recovery_jobs = []
            if selected and recovery_cuda:
                for task, row, base in selected:
                    key = (task.name, row["filename"])
                    by_key[key] = (task, row, base)
                    final_paths[key] = base
                    recovery_jobs.append(RecoveryJob(
                        task_name=task.name,
                        sample_id=row["filename"],
                        base_cif=base,
                        refined_cif=base.with_suffix(".refined.cif"),
                        task=task,
                        row=row,
                        ccd_path=ccd_path,
                        rerun=rerun,
                    ))
                    record(task, row, "pending")
                flush(force=True)
                results = run_recovery_postprocessing(
                    tuple(recovery_jobs), config=config, on_result=on_result
                )
                submitted_jobs = len(recovery_jobs)
            elif selected:
                preload_context_chemistry(runner.tasks, ccd_path=ccd_path)
                for task, row, base in selected:
                    key = (task.name, row["filename"])
                    by_key[key] = (task, row, base)
                    final_paths[key] = base
                    try:
                        candidate = candidate_from_existing(base, row, task, ccd_path=ccd_path)
                        candidates[key] = candidate
                        payload = build_parameterization_payload(candidate, task_name=task.name,
                                                                 sample_id=row["filename"])
                        refined = base.with_suffix(".refined.cif")
                        if refined.is_symlink():
                            raise ValueError("refined artifact may not be a symlink")
                        if refined.exists() and not rerun:
                            try:
                                validate_refined_artifact(base, refined, payload)
                            except (ValueError, RuntimeError) as error:
                                raise PostprocessPreparationError(
                                    "existing refined artifact is invalid; use --rerun "
                                    f"to replace this staging file ({error})") from error
                            os.replace(refined, base)
                            finish_geometry(task, row, base, candidate)
                            completed += 1
                            record(task, row, "succeeded")
                            continue
                        if refined.exists():
                            # --rerun is explicit authorization to replace this one
                            # owned stale staging artifact; base remains untouched.
                            refined.unlink()
                    except PostprocessPreparationError as error:
                        failed += 1
                        record(task, row, "failed", str(error))
                        continue
                    jobs.append(PostprocessJob(task.name, row["filename"], base, refined, payload))
                    record(task, row, "pending")
                clear_context_chemistry_cache()
                flush(force=True)
                results = run_postprocessing(tuple(jobs), config=config, on_result=on_result)
                submitted_jobs = len(jobs)
            else:
                results = ()
            if len(results) != submitted_jobs:
                raise RuntimeError("recovered postprocess result count differs from submitted jobs")
            for task, row, base in score_only:
                final_paths[(task.name, row["filename"])] = base
            flush(force=True)

            score_entries = [(task, row, final_paths[(task.name, row["filename"])])
                             for task, row, _base in selected if task.score
                             and ((task.name, row["filename"]) in changed_geometry
                                  or not _has_final_score(row))]
            score_entries.extend(score_only)
            score_failed = 0
            if score_entries:
                from src.scoring import PartiScorer
                options = OmegaConf.to_container(runner.config.scorer, resolve=True)
                checkpoint = options.pop("checkpoint")
                scorer = PartiScorer.from_checkpoint(checkpoint, **options)
                try:
                    score_results = scorer.score_files(
                        [path for _task, _row, path in score_entries],
                        identities=[f"{task.name}/{row['filename']}" for task, row, _path in score_entries],
                        design_chains=[
                            (
                                sorted(set(str(chain) for chain in
                                    candidates[(task.name, row["filename"])].atoms.chain_id[
                                        candidates[(task.name, row["filename"])].atoms.is_design]))
                                if task.format != "antibody" and (task.name, row["filename"]) in candidates
                                else _row_design_chains(task, row)
                            )
                            for task, row, _path in score_entries],
                        scope="design_groups")
                    interface_rows = {}
                    for (task, row, path), result in zip(score_entries, score_results, strict=True):
                        expected = f"{task.name}/{row['filename']}"
                        if result.identity != expected:
                            raise RuntimeError("recovery PartiScore result identity mismatch")
                        score_failed += result.status == "error"
                        if task.format == "antibody":
                            from ..antibody.scoring import antibody_score_record
                            values = antibody_score_record(result,
                                [row[field] for field in ("heavy_chain", "light_chain") if row.get(field)])
                            row.update({key: value for key, value in values.items() if key in row})
                            if result.status == "error":
                                row["status"] = "partial"
                                row["error_stage"] = ";".join(dict.fromkeys(
                                    [s for s in row.get("error_stage", "").split(";") if s] + ["score"]))
                                row["error"] = " | ".join(filter(None, (
                                    row.get("error", ""), "partiscore: " + (result.error or "failed"))))
                        else:
                            source = "refined" if row["postprocess_status"] == "succeeded" else "design"
                            prefix = "refined_partiscore" if source == "refined" else "partiscore"
                            score_values = {f"{prefix}_score": result.score,
                                            f"{prefix}_pair_count": result.pair_count,
                                            f"{prefix}_status": result.status,
                                            f"{prefix}_error": result.error,
                                            f"{prefix}_source": source,
                                            f"{prefix}_weights": result.weights,
                                            f"{prefix}_checkpoint": str(result.checkpoint)}
                            if source == "refined":
                                score_values["refined_filename"] = path.name
                            elif _has_row_value(row, "partiscore_score"):
                                # This public value can be the pre-relax search reward.
                                score_values.pop("partiscore_score")
                            row.update(score_values)
                            fields[task.name].extend(key for key in score_values
                                                     if key not in fields[task.name])
                            interface_rows.setdefault(task.name, []).extend(
                                {"filename": row["filename"], "source": source,
                                 **{k: v for k, v in item.as_dict().items() if k != "status"}}
                                for item in result.interfaces)
                        if "final_partiscore" in row:
                            row["final_partiscore"] = (result.score if result.status == "ok"
                                                       and _has_final_score({"final_partiscore": result.score})
                                                       else "")
                            if task.format == "antibody":
                                row["final_partiscore"] = (values.get("partiscore_antigen")
                                                           if values.get("partiscore_antigen_status") == "ok"
                                                           else "")
                        dirty.add(task.name)
                    for name, additions in interface_rows.items():
                        path = root / name / "partiscore.csv"
                        replaced = {row["filename"] for task, row, _ in score_entries if task.name == name}
                        previous, columns = ((interface_tables[name], interface_fields[name])
                                             if name in interface_tables else _read_interfaces(root / name))
                        previous = [item for item in previous if item["filename"] not in replaced]
                        if name in interface_tables:
                            interface_tables[name] = previous + additions
                            interface_dirty.add(name)
                        else:
                            atomic_csv(path, previous + additions, columns)
                finally:
                    scorer.close()
            flush(force=True)
            for name in sorted(ranking_tasks):
                ranked, interfaces = _publish_score_ranking(
                    root / name, tables[name], interface_tables.get(name, ()), fields[name],
                    interface_fields.get(name))
                tables[name] = ranked
                if name in interface_tables:
                    interface_tables[name] = interfaces
            return DesignRunResult(root, len(selected), 0, completed, failed,
                                   len(score_entries) - score_failed, score_failed)
        finally:
            clear_context_chemistry_cache()
            flush(force=True)
