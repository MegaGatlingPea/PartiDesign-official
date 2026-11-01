"""Small atomic helpers for validation outputs.

Validation state is deliberately plain: a sequence JSON stores the sequence,
its predictions and final metrics. Resume compares the recorded sequence,
request context, slot metadata and ordinary owned structure files. It does not
hash source contents or implementation metadata.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import csv
import fcntl
import json
import math
from numbers import Number
import os
from pathlib import Path
import tempfile
from typing import Any


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical(value: Any) -> str:
    """Stable text representation used only for CSV cells."""
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def ensure_directory(path: Path) -> Path:
    path = Path(path)
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError(f"Managed output has symlink component: {item}")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def locked(path: Path, *, blocking=True):
    ensure_directory(path.parent)
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


def atomic_text(path: Path, content: str) -> None:
    ensure_directory(path.parent)
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


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> dict | None:
    if path.is_symlink():
        raise ValueError(f"State may not be a symlink: {path}")
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        raise ValueError(f"Unreadable validation state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Validation state is not an object: {path}")
    return value


def owned_artifact(path: Path, root: Path) -> Path:
    """Return a managed regular file, used for deliberate local copies."""
    path, root = Path(path), Path(root)
    if not path.is_absolute():
        raise ValueError(f"Artifact path must be absolute: {path}")
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError(f"Artifact has symlink component: {item}")
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Artifact is not a regular file owned by {root}: {path}")
    return path


def score_status_valid(value: Any) -> bool:
    return isinstance(value, dict) and value.get("status") in {"ok", "no_contact", "no_interface"}


def nested_record_valid(record: dict) -> bool:
    """Check only JSON container shapes needed by filters/reporting."""
    predictions = record.get("predictions", {})
    if not isinstance(predictions, dict) or any(not isinstance(p, dict) for p in predictions.values()):
        return False
    for prediction in predictions.values():
        for field in ("metrics", "reasons", "partiscore", "confidence"):
            if field in prediction and not isinstance(prediction[field], dict):
                return False
    for field in ("paired_metrics", "paired_reasons", "scores", "decisions"):
        values = record.get(field, {})
        if not isinstance(values, dict) or any(not isinstance(value, dict) for value in values.values()):
            return False
    return True


def required_scores_valid(record: dict, settings: dict, method: str) -> bool:
    if settings.get("enabled") and settings.get("original"):
        if not score_status_valid(record.get("scores", {}).get("original")):
            return False
    if method == "partiscore" or (settings.get("enabled") and settings.get("refold")):
        if any(not score_status_valid(pred.get("partiscore")) for pred in record.get("predictions", {}).values()):
            return False
    return True


def metrics_valid(prediction: dict) -> bool:
    return (
        isinstance(prediction, dict)
        and isinstance(prediction.get("metrics", {}), dict)
        and isinstance(prediction.get("reasons", {}), dict)
        and ("partiscore" not in prediction or isinstance(prediction["partiscore"], dict))
        and prediction.get("metrics_status") == "complete"
    )


def _same_value(left: Any, right: Any) -> bool:
    """Compare JSON-like provenance without introducing content fingerprints."""
    if isinstance(left, os.PathLike):
        left = os.fspath(left)
    if isinstance(right, os.PathLike):
        right = os.fspath(right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if len(left) != len(right) or set(left) != set(right):
            return False
        return all(_same_value(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same_value(item, other) for item, other in zip(left, right)
        )
    if isinstance(left, Number) and isinstance(right, Number):
        if isinstance(left, bool) or isinstance(right, bool):
            return type(left) is type(right) and left == right
        try:
            return bool(left == right) and math.isfinite(float(left)) and math.isfinite(float(right))
        except (TypeError, ValueError, OverflowError):
            return False
    # Numpy arrays and similar scalar containers are not expected in persisted
    # state, but accepting their ordinary list/scalar representation makes
    # direct callers compare the same way as JSON-loaded records.
    for value, other in ((left, right), (right, left)):
        if not isinstance(value, (str, bytes, Mapping, list, tuple, Number)) and hasattr(value, "tolist"):
            try:
                return _same_value(value.tolist(), other)
            except (TypeError, ValueError):
                return False
    try:
        result = left == right
        return bool(result)
    except (TypeError, ValueError):
        return False


def _msa_context_identity(value):
    """Preserve AF3 data identity across relocation of an unchanged cache."""
    if isinstance(value, Mapping):
        return {key: _msa_context_identity(item) for key, item in value.items()
                if key != "cache_dir"}
    if isinstance(value, (list, tuple)):
        return [_msa_context_identity(item) for item in value]
    return value


def _context_matches(record: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    for key, value in context.items():
        if key == "msa_context" and context.get("method") in {"af3", "af3-la"}:
            if _msa_context_identity(record.get(key)) != _msa_context_identity(value):
                return False
            continue
        if value is None or key not in record or not _same_value(record[key], value):
            if value is not None:
                return False
    return True


def _sequence_matches(record: Mapping[str, Any], sequence: Mapping[str, Any]) -> bool:
    previous = record.get("sequence")
    return isinstance(previous, Mapping) and _same_value(previous, sequence)


def _structure_path(prediction: Mapping[str, Any]) -> Path | None:
    value = prediction.get("structure_path")
    if not value:
        value = prediction.get("cif_path") or prediction.get("pdb_path")
    if not value:
        return None
    try:
        return Path(value)
    except (TypeError, ValueError):
        return None


def _prediction_metadata_valid(
    key: str,
    prediction: Mapping[str, Any],
    method: str | None,
    sequence: Mapping[str, Any] | None = None,
) -> bool:
    """Validate the slot identity before allowing a structure to be reused."""
    if not isinstance(prediction, Mapping):
        return False
    if prediction.get("key") != key:
        return False
    if sequence is not None:
        for field in ("source", "source_index"):
            if field in prediction and field in sequence and not _same_value(
                prediction[field], sequence[field]
            ):
                return False
    if method in {"af2_ig", "esmfold2-fast", "esmfold2-standard", "af3", "af3-la", "partiscore", "vina_dock"}:
        # Import lazily so storage remains usable by the planning/reporting
        # utilities without creating an import cycle at module load time.
        from .planning import prediction_metadata_valid

        if not prediction_metadata_valid(key, dict(prediction), method):
            return False
    metadata = prediction.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            return False
        for field in ("key", "protocol", "model", "sample"):
            if field in metadata and field in prediction and not _same_value(
                metadata[field], prediction[field]
            ):
                return False
    return True


def _af3_matrix_shape(value: Any) -> tuple[int, int] | None:
    """Return a finite non-negative square JSON matrix shape."""
    if not isinstance(value, (list, tuple)) or not value:
        return None
    size = len(value)
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != size:
            return None
        for entry in row:
            if isinstance(entry, bool) or not isinstance(entry, Number):
                return None
            try:
                numeric = float(entry)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(numeric) or numeric < 0.0:
                return None
    return size, size


def _af3_confidence_valid(prediction: Mapping[str, Any], root: Path | None) -> bool:
    """Validate the private native AF3 confidence artifact needed for resume."""
    value = prediction.get("raw_confidence_path")
    if not value:
        return False
    try:
        confidence_path = Path(value)
        if root is not None:
            confidence_path = owned_artifact(confidence_path, root)
        elif (
            not confidence_path.is_absolute()
            or any(item.is_symlink() for item in (confidence_path, *confidence_path.parents))
            or not confidence_path.is_file()
        ):
            return False
        raw = json.loads(confidence_path.read_text())
    except (OSError, TypeError, ValueError, RuntimeError):
        return False
    if not isinstance(raw, Mapping):
        return False

    pae = raw.get("full_pae", raw.get("pae_matrix"))
    pde = raw.get("full_pde", raw.get("pde_matrix"))
    native_projection = pae is None and pde is None
    if native_projection:
        # Current workers keep the native matrices once and the source-token
        # projection, rather than serializing duplicate projected matrices.
        pae = raw.get("native_full_pae")
        pde = raw.get("native_full_pde")
    pae_shape = _af3_matrix_shape(pae)
    pde_shape = _af3_matrix_shape(pde)
    if pae_shape is None or pde_shape is None or pae_shape != pde_shape:
        return False

    labels = raw.get("token_chain_ids")
    label_size = pae_shape[0]
    if native_projection:
        indexes = raw.get("source_token_indices")
        native_labels = raw.get("native_token_chain_ids")
        source_labels = raw.get("source_token_chain_ids")
        source_mask = raw.get("native_token_source_mask")
        chain_map = raw.get("native_to_source_chain")
        if (
            not isinstance(indexes, (list, tuple)) or not indexes
            or any(type(index) is not int or not 0 <= index < label_size
                   for index in indexes)
            or len(set(indexes)) != len(indexes)
            or not isinstance(native_labels, (list, tuple))
            or len(native_labels) != label_size
            or any(not isinstance(label, str) or not label
                   for label in native_labels)
            or not isinstance(source_labels, (list, tuple))
            or len(source_labels) != len(indexes)
            or any(not isinstance(label, str) or not label
                   for label in source_labels)
            or not isinstance(source_mask, (list, tuple))
            or len(source_mask) != label_size
            or any(type(mapped) is not bool for mapped in source_mask)
            or not isinstance(chain_map, Mapping)
        ):
            return False
        for index, source_label in zip(indexes, source_labels):
            if (not source_mask[index]
                    or chain_map.get(native_labels[index]) != source_label):
                return False
        label_size = len(indexes)
        if labels != source_labels:
            return False
    if not isinstance(labels, (list, tuple)) or len(labels) != label_size:
        return False
    if any(not isinstance(label, str) or not label for label in labels):
        return False

    native_plddt = raw.get("native_plddt")
    if isinstance(native_plddt, bool) or not isinstance(native_plddt, Number):
        return False
    try:
        native_plddt = float(native_plddt)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(native_plddt) or not 0.0 <= native_plddt <= 100.0:
        return False
    mean_plddt = raw.get("mean_plddt")
    if mean_plddt is not None:
        if isinstance(mean_plddt, bool) or not isinstance(mean_plddt, Number):
            return False
        try:
            mean_plddt = float(mean_plddt)
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            not math.isfinite(mean_plddt)
            or not 0.0 <= mean_plddt <= 100.0
            or not math.isclose(mean_plddt, native_plddt)
        ):
            return False
    plddt = raw.get("plddt")
    if plddt is not None:
        if not isinstance(plddt, (list, tuple)):
            return False
        for entry in plddt:
            if isinstance(entry, bool) or not isinstance(entry, Number):
                return False
            try:
                entry = float(entry)
            except (TypeError, ValueError, OverflowError):
                return False
            if not math.isfinite(entry) or not 0.0 <= entry <= 100.0:
                return False
    return True


def prediction_valid(
    prediction: dict,
    root: Path | None = None,
    method: str | None = None,
) -> bool:
    """A completed prediction is reusable only when its artifacts remain valid."""
    if not isinstance(prediction, dict) or prediction.get("status") != "complete":
        return False
    path = _structure_path(prediction)
    if path is None:
        return False
    try:
        if any(item.is_symlink() for item in (path, *path.parents)):
            return False
    except OSError:
        return False
    if root is not None:
        try:
            path = owned_artifact(path, root)
        except (OSError, ValueError):
            return False
    try:
        if not path.is_file():
            return False
    except OSError:
        return False
    af3_method = method
    if af3_method is None and prediction.get("model") in {"af3", "af3-la"}:
        af3_method = str(prediction["model"])
    if af3_method in {"af3", "af3-la"}:
        confidence_root = root
        if confidence_root is None:
            # AF3 native artifacts are emitted beside the owned structure.  A
            # direct predicate caller still gets ownership enforcement.
            confidence_root = path.parent
        if not _af3_confidence_valid(prediction, confidence_root):
            return False
    return True


def record_complete(
    record: dict,
    expected: list[str] | tuple[str, ...],
    *,
    root: Path | None = None,
    method: str | None = None,
) -> bool:
    if record.get("status") != "complete" or record.get("expected_predictions") != list(expected):
        return False
    predictions = record.get("predictions", {})
    if not isinstance(predictions, dict) or set(predictions) != set(expected):
        return False
    return all(
        _prediction_metadata_valid(key, prediction, method)
        and prediction_valid(prediction, root, method)
        and metrics_valid(prediction)
        for key, prediction in predictions.items()
    )


_OWNED_PREDICTION_METHODS = {"af2_ig", "esmfold2-fast", "esmfold2-standard", "af3", "af3-la"}


def _retained_pairs(record: Mapping[str, Any], predictions: Mapping[str, dict]) -> tuple[dict, dict]:
    """Keep pair metrics only while both constituent predictions survive."""
    paired_metrics = record.get("paired_metrics", {})
    paired_reasons = record.get("paired_reasons", {})
    if not isinstance(paired_metrics, Mapping):
        paired_metrics = {}
    if not isinstance(paired_reasons, Mapping):
        paired_reasons = {}
    retained_metrics: dict[str, dict] = {}
    retained_reasons: dict[str, dict] = {}
    for pair_key, metrics in paired_metrics.items():
        if not isinstance(pair_key, str) or not isinstance(metrics, dict):
            continue
        suffix = pair_key.split("_", 1)[1] if pair_key.startswith(("complex_", "monomer_")) else pair_key
        if f"complex_{suffix}" not in predictions or f"monomer_{suffix}" not in predictions:
            continue
        retained_metrics[pair_key] = dict(metrics)
        reasons = paired_reasons.get(pair_key, {})
        if isinstance(reasons, dict):
            retained_reasons[pair_key] = dict(reasons)
    return retained_metrics, retained_reasons


def _retain_prediction(
    key: str,
    prediction: Any,
    *,
    method: str | None,
    sequence: Mapping[str, Any],
    root: Path | None,
) -> dict | None:
    if not isinstance(prediction, dict):
        return None
    if not _prediction_metadata_valid(key, prediction, method, sequence):
        return None
    if not prediction_valid(prediction, root, method):
        return None
    retained = dict(prediction)
    # A structure can be safely resumed even when its metrics need a fresh
    # evaluation.  Leave complete metrics in place; clear incomplete metric
    # state so it cannot make a pending record look scientifically complete.
    if (
        not metrics_valid(retained)
        or "metrics" not in retained
        or "reasons" not in retained
    ):
        for field in ("metrics", "reasons", "metric_provenance", "metrics_status"):
            retained.pop(field, None)
    if "partiscore" in retained and not score_status_valid(retained["partiscore"]):
        retained.pop("partiscore", None)
    return retained


def _valid_scores(record: Mapping[str, Any]) -> dict:
    scores = record.get("scores", {})
    if not isinstance(scores, Mapping):
        return {}
    return {
        str(key): dict(value)
        for key, value in scores.items()
        if score_status_valid(value)
    }


def begin_sequence(path: Path, context: dict, sequence: dict, expected: list[str], *, force=False) -> dict:
    """Open a sequence checkpoint and retain safe completed predictions.

    The sequence itself remains caller-owned and must match exactly.  A partial
    record may retain completed backend outputs so the runner can pass their
    keys back to the native backend; unfinished predictions are regenerated.
    """
    path = Path(path)
    expected = list(expected)
    try:
        old = read_json(path)
    except ValueError:
        if any(item.is_symlink() for item in (path, *path.parents)):
            raise
        old = None
    method = context.get("method") if isinstance(context, Mapping) else None
    root = path.parent if method in _OWNED_PREDICTION_METHODS else None
    compatible = bool(
        old
        and not force
        and old.get("status") in {"pending", "running", "partial", "complete", "failed"}
        and old.get("expected_predictions") == expected
        and _context_matches(old, context)
        and _sequence_matches(old, sequence)
    )
    if compatible and record_complete(old, expected, root=root, method=method):
        return old
    record = {
        **{key: value for key, value in context.items() if value is not None},
        "sequence": sequence,
        "expected_predictions": expected,
        "predictions": {},
        "paired_metrics": {},
        "paired_reasons": {},
        "decisions": {},
        "status": "pending",
    }
    if compatible:
        previous = old.get("predictions", {})
        if isinstance(previous, Mapping):
            for key in expected:
                prediction = _retain_prediction(
                    key,
                    previous.get(key),
                    method=method,
                    sequence=sequence,
                    root=root,
                )
                if prediction is not None:
                    record["predictions"][key] = prediction
        paired_metrics, paired_reasons = _retained_pairs(old, record["predictions"])
        record["paired_metrics"] = paired_metrics
        record["paired_reasons"] = paired_reasons
        scores = _valid_scores(old)
        if scores:
            record["scores"] = scores
    atomic_json(path, record)
    return record


def atomic_csv(path: Path, rows: list[dict], base_fields: list[str]) -> None:
    import io

    fields = list(dict.fromkeys(base_fields + sorted({k for row in rows for k in row})))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: "" if v is None else canonical(v) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    atomic_text(path, buffer.getvalue())
