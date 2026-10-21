from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..ccd_bonds import CCDBondCacheReader
from ..config import ESMFold2Config, effective_esmfold2_lm_mask_pct
from ..esmfold2_evidence import (
    atomic_number,
    decode_atom_name_rows,
)
from ..errors import AdapterError
from ..json_safety import normalize_nonfinite
from ..model import FoldResult
from ..runtime import RuntimePaths


_POLYMER_MOL_TYPE = {"protein": 0, "dna": 1, "rna": 2}
_PROTEIN_COMPONENT = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}
_PROTEIN_RES_TYPE = {
    letter: index
    for index, letter in enumerate(
        "ARNDCQEGHILKMFPSTWYVX",
        start=2,
    )
}
_PROTEIN_INPUT_ID = {
    "L": 4,
    "A": 5,
    "G": 6,
    "V": 7,
    "S": 8,
    "E": 9,
    "R": 10,
    "T": 11,
    "I": 12,
    "D": 13,
    "P": 14,
    "K": 15,
    "Q": 16,
    "N": 17,
    "F": 18,
    "Y": 19,
    "M": 20,
    "H": 21,
    "W": 22,
    "C": 23,
    # The current input builder maps canonical UNK/X through its backbone-only
    # fallback, which uses the shared non-protein input ID.
    "X": 24,
}
_NUCLEIC_RES_TYPE = {
    "rna": {"A": 23, "G": 24, "C": 25, "U": 26, "N": 27},
    "dna": {"A": 28, "G": 29, "C": 30, "T": 31, "N": 32},
}
_MODIFICATION_LEGACY_KEYS = frozenset({"position", "ccd"})
_MODIFICATION_STRICT_KEYS = frozenset(
    {"position", "ccd", "atom_names", "infer_standard_polymer_bonds"}
)
_PRIVATE_COVALENT_ORDER_TO_MMCIF = {
    "?": "?",
    "single": "sing",
    "double": "doub",
    "triple": "trip",
    "aromatic": "arom",
}
_PRIVATE_COVALENT_ORDERS = frozenset(_PRIVATE_COVALENT_ORDER_TO_MMCIF)


def _mmcif_covalent_order(raw: Any) -> str:
    if not isinstance(raw, str) or raw not in _PRIVATE_COVALENT_ORDERS:
        raise AdapterError(
            "ESMFold2 private covalent bond order must be one of ?, single, "
            "double, triple, aromatic"
        )
    return _PRIVATE_COVALENT_ORDER_TO_MMCIF[raw]


@dataclass(frozen=True, slots=True)
class ESMFold2BatchFoldRequest:
    request_key: str
    input_json: Path
    raw_dir: Path
    final_refold_dir: Path
    sample_id: str
    variant_id: str


@dataclass(slots=True)
class ESMFold2BatchFoldOutcome:
    request_key: str
    request_id: str
    folds: list[FoldResult]
    error: str | None
    attempt_dir: Path


def _validate_request_set(requests: list[ESMFold2BatchFoldRequest]) -> None:
    if not requests:
        raise AdapterError("ESMFold2 persistent batch must contain at least one request")
    if len({request.request_key for request in requests}) != len(requests):
        raise AdapterError("ESMFold2 batch contains duplicate request keys")
    if len({(request.sample_id, request.variant_id) for request in requests}) != len(
        requests
    ):
        raise AdapterError("ESMFold2 batch contains duplicate sample/variant pairs")


def _validated_new_batch_root(path: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or raw.exists():
        raise AdapterError(f"ESMFold2 batch root already exists or is unsafe: {raw}")
    return raw.resolve()


def _effective_model_contract(config: ESMFold2Config) -> dict[str, Any]:
    from ..esmfold2_optimization import validate_config

    optimization = validate_config(
        getattr(config, "optimization", {"mode": "auto", "disable": []})
    )
    return {
        "seed": config.seed,
        "loops": config.loops,
        "steps": config.steps,
        "samples": config.samples,
        "lm_dropout": config.lm_dropout,
        "lm_mask_pct": effective_esmfold2_lm_mask_pct(config),
        "esmc_attention_backend": config.esmc_attention_backend,
        "context_embedding_cache": config.context_embedding_cache,
        "optimization": optimization,
    }


def _validated_request_input(
    request: ESMFold2BatchFoldRequest,
) -> tuple[Path, dict[str, Any], Path]:
    input_path = _safe_input(request.input_json)
    try:
        input_payload = json.loads(input_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterError(f"Invalid ESMFold2 private input {input_path}: {exc}") from exc
    if (
        not isinstance(input_payload, dict)
        or input_payload.get("sample_id") != request.sample_id
        or input_payload.get("variant_id") != request.variant_id
        or input_payload.get("msa_policy") != "none"
    ):
        raise AdapterError(
            "ESMFold2 private input identity/MSA policy differs from its request"
        )
    raw_parent = Path(request.raw_dir).expanduser()
    if raw_parent.is_symlink() or (raw_parent.exists() and not raw_parent.is_dir()):
        raise AdapterError(f"ESMFold2 raw request root is unsafe: {raw_parent}")
    return input_path, input_payload, raw_parent.resolve()


def run_esmfold2_batch(
    runtime: RuntimePaths,
    requests: list[ESMFold2BatchFoldRequest],
    batch_root: Path,
    config: ESMFold2Config,
) -> list[ESMFold2BatchFoldOutcome]:
    """Run one persistent worker for the requested folds.

    Outputs are accepted from the current worker response. An interrupted
    process is simply rerun by the caller; no persisted state or model
    identity records are used to admit old output.
    """
    runtime.validate_esmfold2(config.mode)
    _validate_request_set(requests)
    return _run_esmfold2_batch_fresh(runtime, requests, batch_root, config)


def _run_esmfold2_batch_fresh(
    runtime: RuntimePaths,
    requests: list[ESMFold2BatchFoldRequest],
    batch_root: Path,
    config: ESMFold2Config,
) -> list[ESMFold2BatchFoldOutcome]:
    """Run one offline ESMFold2 process and publish its current results."""
    runtime.validate_esmfold2(config.mode)
    _validate_request_set(requests)
    batch_root = _validated_new_batch_root(batch_root)
    batch_root.mkdir(parents=True, exist_ok=False)
    batch_id = batch_root.name
    effective_config = _effective_model_contract(config)

    planned: list[tuple[ESMFold2BatchFoldRequest, Path, str, dict[str, Any]]] = []
    worker_requests: list[dict[str, Any]] = []
    for request in requests:
        input_path, input_payload, raw_parent = _validated_request_input(request)
        expected_prefix = f"{request.sample_id}__{request.variant_id}"
        raw_parent.mkdir(parents=True, exist_ok=True)
        attempt_dir = raw_parent / f"attempt_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        if attempt_dir.exists() or attempt_dir.is_symlink():
            raise AdapterError(f"ESMFold2 attempt target is unsafe: {attempt_dir}")
        request.final_refold_dir.mkdir(parents=True, exist_ok=True)
        request_id = uuid.uuid4().hex
        worker_requests.append(
            {
                "request_id": request_id,
                "input_json": str(input_path),
                "output_dir": str(attempt_dir),
                "expected_prefix": expected_prefix,
                "expected_samples": config.samples,
            }
        )
        planned.append((request, attempt_dir, request_id, input_payload))

    request_json = batch_root / "request.json"
    result_json = batch_root / "result.json"
    request_payload = {
        "schema_version": 1,
        "batch_id": batch_id,
        "result_json": str(result_json),
        "mode": config.mode,
        "paths": {
            "fold_checkpoint": str(runtime.esmfold2_checkpoint(config.mode)),
            "esmc_checkpoint": str(runtime.esmfold2_esmc_checkpoint),
            "ccd": str(runtime.esmfold2_ccd),
            "ccd_bonds": str(runtime.esmfold2_ccd_bonds),
            "components_cif": str(runtime.esmfold2_common / "components.cif"),
        },
        "config": effective_config,
        "requests": worker_requests,
    }
    _atomic_json(request_json, request_payload)
    command = (
        str(runtime.esmfold2_python.resolve()),
        "-m",
        "pipetools.workers.esmfold2",
        "batch",
        "--request-json",
        str(request_json),
    )
    environment = runtime.esmfold2_subprocess_env()
    if (
        environment.get("PIPETOOLS_NETWORK_POLICY") != "fold-offline"
        or environment.get("HF_HUB_OFFLINE") != "1"
        or environment.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise AdapterError("ESMFold2 worker environment is not strictly offline")
    try:
        timeout_seconds = config.timeout_seconds * len(planned)
        completed = subprocess.run(
            list(command),
            cwd=runtime.esmfold2_source,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
        )
        process_exit: int | str = completed.returncode
        stdout, stderr = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        process_exit = "timeout"
        stdout = _timeout_text(exc.stdout)
        stderr = _timeout_text(exc.stderr)
    _atomic_text(batch_root / "stdout.log", stdout)
    _atomic_text(batch_root / "stderr.log", stderr)

    result_payload: dict[str, Any] | None = None
    result_error: str | None = None
    if result_json.is_symlink() or not result_json.is_file():
        result_error = "ESMFold2 worker did not write result.json"
    else:
        try:
            loaded = json.loads(result_json.read_text())
            if not isinstance(loaded, dict):
                raise ValueError("result is not an object")
            result_payload = loaded
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            result_error = f"Invalid ESMFold2 worker result: {exc}"

    raw_results = result_payload.get("results") if result_payload is not None else None
    if (
        result_payload is None
        or result_payload.get("schema_version") != 1
        or result_payload.get("batch_id") != batch_id
        or result_payload.get("mode") != config.mode
        or result_payload.get("request_count") != len(planned)
        or not isinstance(raw_results, list)
        or len(raw_results) != len(planned)
    ):
        if result_error is None:
            result_error = "ESMFold2 worker result cardinality differs from requests"
        raw_results = []

    validation_ccd_reader = None
    if result_error is None:
        validation_ccd_reader = CCDBondCacheReader(
            runtime.esmfold2_ccd_bonds,
            source_cif=runtime.esmfold2_common / "components.cif",
        )
    outcomes: list[ESMFold2BatchFoldOutcome] = []
    for index, (request, attempt_dir, request_id, input_payload) in enumerate(planned):
        item = raw_results[index] if index < len(raw_results) else None
        error: str | None = result_error
        if error is None and (
            not isinstance(item, dict)
            or set(item) != {"request_id", "status", "error", "artifacts"}
            or item.get("request_id") != request_id
        ):
            error = "ESMFold2 worker result item differs from request"
        if error is None and item.get("status") != "ok":
            error = str(item.get("error") or "ESMFold2 request failed")
        folds: list[FoldResult] = []
        if error is None:
            try:
                artifacts = _validated_artifacts(
                    attempt_dir,
                    item,
                    request_id,
                    request.sample_id,
                    request.variant_id,
                    config.samples,
                    expected_input_payload=input_payload,
                    mode=config.mode,
                    effective_config=effective_config,
                    ccd_bond_cache=runtime.esmfold2_ccd_bonds,
                    ccd_source_cif=runtime.esmfold2_common / "components.cif",
                    ccd_reader=validation_ccd_reader,
                )
                folds = _publish_artifacts(request, attempt_dir, artifacts)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if not attempt_dir.exists():
            attempt_dir.mkdir(parents=True, exist_ok=False)
        _atomic_text(
            attempt_dir / "process.log",
            _format_process_log(
                config.mode,
                {"batch_id": batch_id, "request_id": request_id},
                command,
                process_exit,
                "ok" if error is None else "error",
                error or "",
                stdout,
                stderr,
            ),
        )
        outcomes.append(
            ESMFold2BatchFoldOutcome(
                request_key=request.request_key,
                request_id=request_id,
                folds=folds,
                error=error,
                attempt_dir=attempt_dir,
            )
        )
    if validation_ccd_reader is not None:
        validation_ccd_reader.close()
    if process_exit != 0:
        raise AdapterError(
            "ESMFold2 worker failed: "
            f"{process_exit}; stderr_tail={_sanitized_worker_text(stderr)}"
        )
    return outcomes




def _publish_artifacts(
    request: ESMFold2BatchFoldRequest,
    attempt_dir: Path,
    artifacts: list[dict[str, Any]],
) -> list[FoldResult]:
    """Publish validated structures and expose confidence through FoldResult."""
    request.final_refold_dir.mkdir(parents=True, exist_ok=True)
    destinations: list[Path] = []
    folds: list[FoldResult] = []
    try:
        for sample_index, artifact in enumerate(artifacts):
            source = Path(artifact["structure"])
            destination = request.final_refold_dir / (
                f"{request.variant_id}__s{sample_index:03d}.cif"
            )
            _atomic_copy(source, destination)
            destinations.append(destination)
            confidence_record = json.loads(Path(artifact["confidence"]).read_text())
            confidence, discovered = normalize_nonfinite(
                confidence_record["confidence"]
            )
            cache_evidence = confidence_record.get("context_embedding_cache")
            if isinstance(cache_evidence, dict):
                confidence["context_embedding_cache"] = dict(cache_evidence)
            recorded = confidence_record.get("confidence_unavailable", [])
            if not isinstance(recorded, list):
                recorded = []
            if discovered:
                recorded = [*recorded, *discovered]
            folds.append(
                FoldResult(
                    sample_id=request.sample_id,
                    variant_id=request.variant_id,
                    sample_index=sample_index,
                    structure_path=str(destination.resolve()),
                    confidence=confidence,
                    confidence_unavailable=recorded,
                    atom_omissions=list(artifact.get("atom_omissions", [])),
                    status="ok",
                    raw_dir=str(attempt_dir.resolve()),
                )
            )
    except Exception:
        for destination in destinations:
            if destination.is_file() and not destination.is_symlink():
                destination.unlink()
        raise
    return folds


def _optional_text(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        return ""
    try:
        return path.read_text()
    except (OSError, UnicodeError):
        return ""


def _valid_context_embedding_cache_evidence(
    evidence: object,
    effective_config: dict[str, object],
) -> bool:
    """Validate optional worker-local context LM cache telemetry."""
    if evidence is None:
        return True
    if not isinstance(evidence, dict) or set(evidence) != {
        "enabled",
        "eligible",
        "hits",
        "misses",
        "entries",
        "full_hits",
        "full_misses",
        "full_entries",
        "context_hits",
        "context_misses",
        "context_entries",
        "lm_embedding_prep",
    }:
        return False
    policy = effective_config.get("context_embedding_cache")
    if policy not in {None, True, False}:
        return False
    if evidence["enabled"] and not evidence["eligible"]:
        return False
    if policy is False and evidence["enabled"]:
        return False
    for field in (
        "hits",
        "misses",
        "entries",
        "full_hits",
        "full_misses",
        "full_entries",
        "context_hits",
        "context_misses",
        "context_entries",
    ):
        if (
            not isinstance(evidence[field], int)
            or isinstance(evidence[field], bool)
            or evidence[field] < 0
        ):
            return False
    if (
        evidence["hits"] != evidence["full_hits"] + evidence["context_hits"]
        or evidence["misses"]
        != evidence["full_misses"] + evidence["context_misses"]
        or evidence["entries"]
        != evidence["full_entries"] + evidence["context_entries"]
    ):
        return False
    value = evidence["lm_embedding_prep"]
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )

def _validated_artifacts(
    attempt_dir: Path,
    result: dict[str, Any],
    request_id: str,
    sample_id: str,
    variant_id: str,
    expected_count: int,
    *,
    expected_input_payload: dict[str, Any],
    mode: str,
    effective_config: dict[str, Any],
    ccd_bond_cache: Path,
    ccd_source_cif: Path,
    ccd_reader: CCDBondCacheReader | None = None,
) -> list[dict[str, Any]]:
    if (
        set(result) != {"request_id", "status", "error", "artifacts"}
        or result.get("request_id") != request_id
        or result.get("status") != "ok"
        or result.get("error") is not None
        or not isinstance(result.get("artifacts"), list)
        or len(result["artifacts"]) != expected_count
    ):
        raise AdapterError("ESMFold2 request result is incomplete")
    expected_prefix = f"{sample_id}__{variant_id}"
    resolved_attempt = attempt_dir.resolve()
    input_payload = expected_input_payload
    sequences = input_payload.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise AdapterError("ESMFold2 input has no sequence ledger")
    chain_ids = [entry.get("chain_id") for entry in sequences if isinstance(entry, dict)]
    if (
        len(chain_ids) != len(sequences)
        or any(not isinstance(chain_id, str) or not chain_id for chain_id in chain_ids)
        or len(set(chain_ids)) != len(chain_ids)
    ):
        raise AdapterError("ESMFold2 input chain ledger is malformed")
    validated: list[dict[str, Any]] = []
    for sample_index, artifact in enumerate(result["artifacts"]):
        if not isinstance(artifact, dict) or set(artifact) != {
            "sample_index",
            "structure",
            "confidence",
            "raw_confidence",
        }:
            raise AdapterError("ESMFold2 artifact record is malformed")
        if artifact["sample_index"] != sample_index:
            raise AdapterError("ESMFold2 artifact sample order differs")
        expected_names = {
            "structure": f"{expected_prefix}_sample_{sample_index}.cif",
            "confidence": f"{expected_prefix}_sample_{sample_index}.confidence.json",
            "raw_confidence": f"{expected_prefix}_sample_{sample_index}.confidence.npz",
        }
        for field, expected_name in expected_names.items():
            path = Path(str(artifact[field]))
            if (
                not path.is_absolute()
                or path.is_symlink()
                or not path.is_file()
                or not path.stat().st_size
                or path.resolve().parent != resolved_attempt
                or path.name != expected_name
            ):
                raise AdapterError(
                    f"ESMFold2 {field} artifact differs from its request evidence"
                )
        try:
            confidence = json.loads(Path(artifact["confidence"]).read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AdapterError(f"Cannot parse ESMFold2 confidence: {exc}") from exc
        if (
            not isinstance(confidence, dict)
            or not isinstance(confidence.get("confidence"), dict)
            or confidence["confidence"].get("model") != "esmfold2"
            or confidence["confidence"].get("mode") != mode
            or confidence.get("effective_config")
            != {
                **effective_config,
                "mode": mode,
                "external_msa_search": False,
                "msa_max_depth": 1,
                "msa_column_mask_rate": 0.1,
                "msa_subsample_at_inference": False,
            }
            or not isinstance(confidence.get("confidence_unavailable"), list)
            or not _valid_context_embedding_cache_evidence(
                confidence.get("context_embedding_cache"),
                effective_config,
            )
        ):
            raise AdapterError("ESMFold2 confidence record is incomplete")
        raw_confidence_path = Path(artifact["raw_confidence"])
        raw_confidence_arrays = _load_esmfold2_raw_confidence(raw_confidence_path)
        if ccd_reader is None:
            with CCDBondCacheReader(
                ccd_bond_cache,
                source_cif=ccd_source_cif,
            ) as ephemeral_ccd_reader:
                atom_omissions = validate_esmfold2_raw_confidence(
                    raw_confidence_path,
                    confidence,
                    input_payload,
                    ccd_reader=ephemeral_ccd_reader,
                    raw_arrays=raw_confidence_arrays,
                )
                validate_esmfold2_output_identity(
                    Path(artifact["structure"]),
                    input_payload,
                    ccd_reader=ephemeral_ccd_reader,
                )
        else:
            atom_omissions = validate_esmfold2_raw_confidence(
                raw_confidence_path,
                confidence,
                input_payload,
                ccd_reader=ccd_reader,
                raw_arrays=raw_confidence_arrays,
            )
            validate_esmfold2_output_identity(
                Path(artifact["structure"]),
                input_payload,
                ccd_reader=ccd_reader,
            )
        validated.append({**artifact, "atom_omissions": atom_omissions})
    return validated


def _load_esmfold2_raw_confidence(path: Path) -> dict[str, np.ndarray]:
    raw_path = Path(path)
    if raw_path.is_symlink() or not raw_path.is_file() or not raw_path.stat().st_size:
        raise AdapterError("ESMFold2 confidence arrays are missing or unsafe")
    expected = {"plddt", "plddt_per_atom", "plddt_ca", "pae", "pde", "ptm",
                "iptm", "complex_plddt", "complex_iplddt", "pair_chains_iptm",
                "token_index", "residue_index", "asym_id", "entity_id",
                "sym_id", "token_attention_mask"}
    with np.load(raw_path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise AdapterError("ESMFold2 confidence array keys differ")
        return {name: np.asarray(archive[name]) for name in expected}


def validate_esmfold2_raw_confidence(
    path: Path,
    confidence_record: dict[str, Any],
    expected_input: dict[str, Any] | list[dict[str, Any]],
    *,
    ccd_reader: CCDBondCacheReader | None = None,
    raw_arrays: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Check confidence outputs and the small token-label contract."""
    if isinstance(expected_input, list):
        expected_input = {"sequences": expected_input, "covalent_bonds": []}
    if not isinstance(expected_input, dict):
        raise AdapterError("ESMFold2 expected input ledger is malformed")
    sequences = expected_input.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise AdapterError("ESMFold2 expected sequence ledger is malformed")
    arrays = _load_esmfold2_raw_confidence(path) if raw_arrays is None else raw_arrays
    for name, array in arrays.items():
        if (array.dtype == np.dtype("O")
                or np.issubdtype(array.dtype, np.complexfloating)
                or not (np.issubdtype(array.dtype, np.number)
                        or np.issubdtype(array.dtype, np.bool_))
                or not np.all(np.isfinite(array))):
            raise AdapterError(f"ESMFold2 confidence array {name} is invalid")
    token_mask = np.asarray(arrays["token_attention_mask"]).astype(bool, copy=False)
    if token_mask.ndim != 1 or not token_mask.any():
        raise AdapterError("ESMFold2 token labels are malformed")
    token_count = token_mask.size
    for name in ("token_index", "residue_index", "asym_id", "entity_id", "sym_id"):
        if np.asarray(arrays[name]).shape != (token_count,):
            raise AdapterError(f"ESMFold2 token label {name} shape differs")
    if np.asarray(arrays["plddt"]).ndim != 1 or np.asarray(arrays["plddt_ca"]).ndim != 1:
        raise AdapterError("ESMFold2 token confidence shape differs")
    if np.asarray(arrays["plddt_per_atom"]).ndim != 1:
        raise AdapterError("ESMFold2 per-atom confidence shape differs")
    for name in ("pae", "pde"):
        if np.asarray(arrays[name]).shape != (token_count, token_count):
            raise AdapterError(f"ESMFold2 confidence {name} shape differs")
    asym = np.asarray(arrays["asym_id"]).astype(int, copy=False)
    valid_asym = asym[token_mask]
    if valid_asym.size == 0 or np.any(valid_asym < 0):
        raise AdapterError("ESMFold2 token chain labels are invalid")
    chain_count = int(valid_asym.max()) + 1
    if set(valid_asym.tolist()) != set(range(chain_count)) or len(sequences) != chain_count:
        raise AdapterError("ESMFold2 token chain labels differ from input")
    if np.asarray(arrays["pair_chains_iptm"]).shape != (chain_count, chain_count):
        raise AdapterError("ESMFold2 chain-pair confidence shape differs")
    for name in ("ptm", "iptm", "complex_plddt", "complex_iplddt"):
        if np.asarray(arrays[name]).size != 1:
            raise AdapterError(f"ESMFold2 confidence {name} is not scalar")
    confidence = confidence_record.get("confidence")
    if not isinstance(confidence, dict):
        raise AdapterError("ESMFold2 confidence JSON summary is missing")
    def json_number(name: str) -> float:
        value = confidence.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AdapterError(f"ESMFold2 confidence JSON {name} is not numeric")
        value = float(value)
        if not np.isfinite(value):
            raise AdapterError(f"ESMFold2 confidence JSON {name} is non-finite")
        return value
    def raw_scalar(name: str) -> float:
        return float(np.asarray(arrays[name]).reshape(-1)[0])
    comparisons = {
        "plddt": raw_scalar("complex_plddt") * 100.0,
        "ptm": raw_scalar("ptm"),
        "iptm": raw_scalar("iptm"),
        "complex_iplddt": raw_scalar("complex_iplddt") * 100.0,
    }
    valid = np.flatnonzero(token_mask)
    pair_index = np.ix_(valid, valid)
    comparisons["pae_mean"] = float(np.asarray(arrays["pae"])[pair_index].mean())
    comparisons["pde_mean"] = float(np.asarray(arrays["pde"])[pair_index].mean())
    for name, expected_value in comparisons.items():
        if not np.isclose(json_number(name), expected_value, rtol=1e-5, atol=1e-4):
            raise AdapterError(f"ESMFold2 confidence JSON {name} differs from arrays")
    per_chain = confidence.get("per_chain_plddt")
    expected_chain_ids = [entry.get("chain_id") for entry in sequences if isinstance(entry, dict)]
    if not isinstance(per_chain, dict) or set(per_chain) != set(expected_chain_ids):
        raise AdapterError("ESMFold2 per-chain confidence ledger differs")
    for chain_index, chain_id in enumerate(expected_chain_ids):
        chain_mask = token_mask & (asym == chain_index)
        value = per_chain.get(chain_id)
        expected_value = float(np.asarray(arrays["plddt"])[chain_mask].mean() * 100.0)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isclose(float(value), expected_value, rtol=1e-5, atol=1e-4)):
            raise AdapterError(f"ESMFold2 per-chain confidence for {chain_id} differs")
    return []


def _validated_leaving_atom_omissions(
    sequences: list[dict[str, Any]],
    bonds: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    ccd_reader: CCDBondCacheReader | None,
) -> list[dict[str, Any]]:
    bonded_chains: set[str] = set()
    for bond in bonds:
        if not isinstance(bond, dict):
            raise AdapterError("ESMFold2 private covalent bond is malformed")
        for side in (1, 2):
            endpoint = bond.get(f"endpoint{side}")
            chain_id = endpoint.get("chain_id") if isinstance(endpoint, dict) else None
            if not isinstance(chain_id, str) or not chain_id:
                raise AdapterError("ESMFold2 private covalent endpoint is malformed")
            bonded_chains.add(chain_id)

    atom_mask = arrays["atom_attention_mask"].astype(bool, copy=False)
    token_mask = arrays["token_attention_mask"].astype(bool, copy=False)
    atom_indices = np.flatnonzero(atom_mask)
    atom_to_token = arrays["atom_to_token"].astype(int, copy=False)[atom_indices]
    if (
        np.any(atom_to_token < 0)
        or np.any(atom_to_token >= token_mask.size)
        or not np.all(token_mask[atom_to_token])
    ):
        raise AdapterError("ESMFold2 omission atom-to-token ledger is invalid")
    try:
        atom_names = decode_atom_name_rows(
            arrays["ref_atom_name_chars"][atom_indices]
        )
        omission_names = decode_atom_name_rows(
            arrays["omission_atom_name_chars"]
        )
    except ValueError as exc:
        raise AdapterError(str(exc)) from exc
    asym_id = arrays["asym_id"].astype(int, copy=False)
    residue_index = arrays["residue_index"].astype(int, copy=False)
    prepared = {
        (
            int(asym_id[token]),
            int(residue_index[token]),
            name,
        )
        for token, name in zip(atom_to_token, atom_names, strict=True)
    }

    chain_indices = arrays["omission_chain_index"].astype(int, copy=False)
    residue_indices = arrays["omission_residue_index"].astype(int, copy=False)
    atomic_numbers = arrays["omission_atomic_number"].astype(int, copy=False)
    observed = list(
        zip(
            chain_indices.tolist(),
            residue_indices.tolist(),
            omission_names,
            atomic_numbers.tolist(),
            strict=True,
        )
    )
    if len(observed) != len(set(observed)):
        raise AdapterError("ESMFold2 omission ledger contains duplicates")
    if observed and ccd_reader is None:
        raise AdapterError(
            "ESMFold2 omission ledger cannot be verified without CCD authority"
        )

    expected: list[tuple[int, int, str, int]] = []
    if ccd_reader is not None:
        for chain_index, entry in enumerate(sequences):
            if not isinstance(entry, dict):
                raise AdapterError("ESMFold2 private sequence entry is malformed")
            chain_id = entry.get("chain_id")
            components = entry.get("ccd")
            if (
                entry.get("modality") != "ligand"
                or chain_id not in bonded_chains
                or not isinstance(components, list)
            ):
                continue
            for position, component in enumerate(components):
                if not isinstance(component, str) or not component:
                    raise AdapterError("ESMFold2 private ligand CCD ledger is malformed")
                for atom in ccd_reader.query_leaving_atoms(component):
                    try:
                        number = atomic_number(atom.type_symbol)
                    except ValueError as exc:
                        raise AdapterError(str(exc)) from exc
                    if number == 1:
                        continue
                    if (chain_index, position, atom.atom_id) not in prepared:
                        expected.append(
                            (chain_index, position, atom.atom_id, number)
                        )
    expected.sort(key=lambda value: (value[0], value[1], value[2]))
    if observed != expected:
        raise AdapterError(
            "ESMFold2 omission ledger differs from CCD leaving-atom authority"
        )

    result: list[dict[str, Any]] = []
    for chain_index, position, atom_name, number in expected:
        if not 0 <= chain_index < len(sequences):
            raise AdapterError("ESMFold2 omission chain index is invalid")
        entry = sequences[chain_index]
        components = entry.get("ccd")
        if not isinstance(components, list) or not 0 <= position < len(components):
            raise AdapterError("ESMFold2 omission residue index is invalid")
        result.append(
            {
                "chain_id": entry["chain_id"],
                "residue_index": position,
                "component": components[position],
                "atom_name": atom_name,
                "atomic_number": number,
                "reason": "ccd_leaving_atom",
            }
        )
    return result


def validate_esmfold2_output_identity(
    path: Path,
    input_payload: dict[str, Any],
    *,
    raw_confidence_path: Path | None = None,
    require_exact_entity_ids: bool = True,
    require_entity_categories: bool = True,
    validate_covalent_topology: bool = True,
    expected_polymer_components_by_chain: dict[str, list[str]] | None = None,
    ccd_reader: CCDBondCacheReader | None = None,
    raw_confidence_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    """Validate post-serialization canonical chain/entity/token identity.

    ESMFold2's in-memory chain objects are validated before inference, but the
    upstream mmCIF writer is a separate serialization boundary.  This check is
    intentionally based on ``label_asym_id`` (never ``auth_asym_id``) and
    requires the output entity equivalence relation, polymer residue coverage,
    ligand CCD identity, modifications, and explicit covalent endpoints to
    remain faithful to the private canonical input.
    """

    cif_path = Path(path)
    if cif_path.is_symlink() or not cif_path.is_file() or not cif_path.stat().st_size:
        raise AdapterError("ESMFold2 output CIF is missing or unsafe")
    sequences = input_payload.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise AdapterError("ESMFold2 output identity input has no sequences")
    private_smiles_entries = [
        entry
        for entry in sequences
        if isinstance(entry, dict)
        and isinstance(entry.get("smiles"), str)
        and bool(entry["smiles"])
    ]
    ccd_ligand_uses_lig = any(
        isinstance(entry, dict)
        and isinstance(entry.get("ccd"), list)
        and "LIG" in entry["ccd"]
        for entry in sequences
    )
    if len(private_smiles_entries) > 1 or (
        private_smiles_entries and ccd_ligand_uses_lig
    ):
        # The upstream decoder currently uses the placeholder component LIG
        # for every private SMILES chain.  Accepting multiple such chains, or
        # a CCD component already named LIG, would merge unrelated graphs
        # under one component identity.
        raise AdapterError(
            "ESMFold2 cannot validate multiple private graphs under one LIG "
            "component"
        )
    polymer_chain_ids = [
        entry.get("chain_id")
        for entry in sequences
        if isinstance(entry, dict)
        and entry.get("modality") in {"protein", "rna", "dna"}
    ]
    normalized_component_overrides: dict[str, list[str]] | None = None
    if expected_polymer_components_by_chain is not None:
        if (
            not isinstance(expected_polymer_components_by_chain, dict)
            or any(
                not isinstance(chain_id, str) or not chain_id
                for chain_id in polymer_chain_ids
            )
            or set(expected_polymer_components_by_chain)
            != set(polymer_chain_ids)
        ):
            raise AdapterError(
                "Output identity polymer-component override ledger differs"
            )
        normalized_component_overrides = {}
        for chain_id in polymer_chain_ids:
            components = expected_polymer_components_by_chain.get(chain_id)
            if (
                not isinstance(components, list)
                or not components
                or any(
                    not isinstance(component, str) or not component
                    for component in components
                )
            ):
                raise AdapterError(
                    "Output identity polymer-component override is malformed"
                )
            normalized_component_overrides[chain_id] = list(components)

    def expected_polymer_components(entry: dict[str, Any]) -> list[str]:
        if normalized_component_overrides is None:
            return _expected_polymer_components(entry)
        chain_id = entry.get("chain_id")
        components = normalized_component_overrides.get(chain_id)
        sequence = entry.get("sequence")
        if (
            components is None
            or not isinstance(sequence, str)
            or len(components) != len(sequence)
        ):
            raise AdapterError(
                "Output identity polymer-component override length differs"
            )
        return components

    try:
        from biotite.structure.io import pdbx

        cif_file = pdbx.CIFFile.read(cif_path)
        blocks = list(cif_file.values())
        if len(blocks) != 1:
            raise ValueError(f"expected one data block, got {len(blocks)}")
        atom_site = blocks[0]["atom_site"]
        arrays = {
            name: atom_site[name].as_array(str)
            for name in (
                "label_asym_id",
                "label_entity_id",
                "label_seq_id",
                "label_comp_id",
                "label_atom_id",
                "type_symbol",
            )
        }
    except Exception as exc:
        raise AdapterError(f"Cannot parse ESMFold2 output identity: {exc}") from exc
    lengths = {len(values) for values in arrays.values()}
    if lengths != {len(arrays["label_asym_id"])} or not next(iter(lengths), 0):
        raise AdapterError("ESMFold2 output atom_site identity columns are malformed")

    expected_chain_ids: list[str] = []
    expected_entities: list[int] = []
    expected_output_entities: list[str] = []
    for entry in sequences:
        if not isinstance(entry, dict):
            raise AdapterError("ESMFold2 private sequence ledger is malformed")
        chain_id = entry.get("chain_id")
        entity_index = entry.get("model_entity_index")
        output_entity_id = entry.get("output_entity_id")
        if (
            not isinstance(chain_id, str)
            or not chain_id
            or not isinstance(entity_index, int)
            or isinstance(entity_index, bool)
            or entity_index < 1
            or not isinstance(output_entity_id, str)
            or not output_entity_id
        ):
            raise AdapterError("ESMFold2 private chain/entity identity is malformed")
        expected_chain_ids.append(chain_id)
        expected_entities.append(entity_index)
        expected_output_entities.append(output_entity_id)
    if len(set(expected_chain_ids)) != len(expected_chain_ids):
        raise AdapterError("ESMFold2 private label_asym_id values are duplicate")

    observed_chain_ids = list(dict.fromkeys(map(str, arrays["label_asym_id"])))
    if observed_chain_ids != expected_chain_ids:
        raise AdapterError("ESMFold2 output changed label_asym_id chain order/identity")
    rows_by_chain: dict[str, list[int]] = {
        chain_id: [
            index
            for index, observed in enumerate(arrays["label_asym_id"])
            if str(observed) == chain_id
        ]
        for chain_id in expected_chain_ids
    }
    observed_entities: list[str] = []
    for chain_id in expected_chain_ids:
        values = {
            str(arrays["label_entity_id"][index])
            for index in rows_by_chain[chain_id]
        }
        if len(values) != 1 or next(iter(values)) in {"", ".", "?"}:
            raise AdapterError(
                f"ESMFold2 output chain {chain_id} has ambiguous label_entity_id"
            )
        observed_entities.append(next(iter(values)))
    if require_exact_entity_ids and observed_entities != expected_output_entities:
        raise AdapterError("ESMFold2 output changed exact variant label_entity_id values")
    for left in range(len(expected_chain_ids)):
        for right in range(len(expected_chain_ids)):
            if (expected_entities[left] == expected_entities[right]) != (
                observed_entities[left] == observed_entities[right]
            ):
                raise AdapterError(
                    "ESMFold2 output collapsed distinct entities or split entity copies"
                )
    validated_entity_ids = (
        expected_output_entities if require_exact_entity_ids else observed_entities
    )

    if require_entity_categories:
        try:
            struct_asym = blocks[0]["struct_asym"]
            struct_chain_ids = struct_asym["id"].as_array(str).tolist()
            struct_entity_ids = struct_asym["entity_id"].as_array(str).tolist()
            entity_ids = blocks[0]["entity"]["id"].as_array(str).tolist()
        except Exception as exc:
            raise AdapterError(
                f"ESMFold2 output entity categories are incomplete: {exc}"
            ) from exc
        unique_output_entities = list(dict.fromkeys(validated_entity_ids))
        if (
            struct_chain_ids != expected_chain_ids
            or struct_entity_ids != validated_entity_ids
            or entity_ids != unique_output_entities
        ):
            raise AdapterError(
                "ESMFold2 output entity categories disagree with atom_site"
            )
    polymer_entities = list(
        dict.fromkeys(
            output_entity
            for output_entity, entry in zip(
                validated_entity_ids, sequences, strict=True
            )
            if entry.get("modality") in {"protein", "rna", "dna"}
        )
    )
    if polymer_entities and require_entity_categories:
        try:
            entity_poly_ids = (
                blocks[0]["entity_poly"]["entity_id"].as_array(str).tolist()
            )
            entity_poly_seq = blocks[0]["entity_poly_seq"]
            entity_poly_seq_columns = {
                name: entity_poly_seq[name].as_array(str).tolist()
                for name in ("entity_id", "num", "mon_id")
            }
        except Exception as exc:
            raise AdapterError(
                f"ESMFold2 output polymer entity categories are incomplete: {exc}"
            ) from exc
        expected_entity_components: dict[str, list[str]] = {}
        for output_entity, entry in zip(
            validated_entity_ids, sequences, strict=True
        ):
            if entry.get("modality") not in {"protein", "rna", "dna"}:
                continue
            components = expected_polymer_components(entry)
            previous = expected_entity_components.setdefault(
                output_entity, components
            )
            if previous != components:
                raise AdapterError(
                    "ESMFold2 one output polymer entity has multiple chemistries"
                )
        if entity_poly_ids != polymer_entities:
            raise AdapterError("ESMFold2 output polymer entity categories disagree")
        observed_entity_components: dict[str, dict[int, str]] = {}
        row_count = len(entity_poly_seq_columns["entity_id"])
        if any(len(values) != row_count for values in entity_poly_seq_columns.values()):
            raise AdapterError("ESMFold2 entity_poly_seq columns are malformed")
        for entity_id, raw_num, mon_id in zip(
            entity_poly_seq_columns["entity_id"],
            entity_poly_seq_columns["num"],
            entity_poly_seq_columns["mon_id"],
            strict=True,
        ):
            try:
                position = int(raw_num)
            except ValueError as exc:
                raise AdapterError("ESMFold2 entity_poly_seq num is invalid") from exc
            entity_rows = observed_entity_components.setdefault(entity_id, {})
            if position < 1 or position in entity_rows:
                raise AdapterError("ESMFold2 entity_poly_seq positions are invalid")
            entity_rows[position] = mon_id
        if set(observed_entity_components) != set(expected_entity_components):
            raise AdapterError("ESMFold2 entity_poly_seq entity ledger differs")
        for entity_id, expected_components in expected_entity_components.items():
            observed = observed_entity_components[entity_id]
            if observed != {
                index: component
                for index, component in enumerate(expected_components, start=1)
            }:
                raise AdapterError(
                    f"ESMFold2 entity_poly_seq chemistry differs for entity {entity_id}"
                )

    for entry, chain_id in zip(sequences, expected_chain_ids, strict=True):
        rows = rows_by_chain[chain_id]
        modality = entry.get("modality")
        if modality == "ligand":
            expected_ccd = entry.get("ccd")
            expected_smiles = entry.get("smiles")
            component_by_position: dict[int, set[str]] = {}
            for row in rows:
                raw_position = str(arrays["label_seq_id"][row])
                try:
                    position = int(raw_position)
                except ValueError as exc:
                    raise AdapterError(
                        f"ESMFold2 output ligand {chain_id} has invalid label_seq_id"
                    ) from exc
                if position < 1 or str(position) != raw_position:
                    raise AdapterError(
                        f"ESMFold2 output ligand {chain_id} has invalid label_seq_id"
                    )
                component_by_position.setdefault(position, set()).add(
                    str(arrays["label_comp_id"][row])
                )
            if isinstance(expected_smiles, str) and expected_smiles:
                expected_components = {1: {"LIG"}}
            elif isinstance(expected_ccd, list) and expected_ccd:
                expected_components = {
                    index: {component}
                    for index, component in enumerate(expected_ccd, start=1)
                }
            else:
                expected_components = {}
            if component_by_position != expected_components:
                raise AdapterError(
                    f"ESMFold2 output ligand {chain_id} changed CCD identity"
                )
            continue
        sequence = entry.get("sequence")
        if modality not in {"protein", "rna", "dna"} or not isinstance(
            sequence, str
        ) or not sequence:
            raise AdapterError(
                f"ESMFold2 private polymer {chain_id} identity is malformed"
            )
        observed_positions: set[int] = set()
        component_by_position: dict[int, set[str]] = {}
        for row in rows:
            raw_position = str(arrays["label_seq_id"][row])
            try:
                position = int(raw_position)
            except ValueError as exc:
                raise AdapterError(
                    f"ESMFold2 output polymer {chain_id} has invalid label_seq_id"
                ) from exc
            if position < 1 or str(position) != raw_position:
                raise AdapterError(
                    f"ESMFold2 output polymer {chain_id} has invalid label_seq_id"
                )
            observed_positions.add(position)
            component_by_position.setdefault(position, set()).add(
                str(arrays["label_comp_id"][row])
            )
        if observed_positions != set(range(1, len(sequence) + 1)):
            raise AdapterError(
                f"ESMFold2 output polymer {chain_id} residue coverage differs"
            )
        expected_components = expected_polymer_components(entry)
        for position, component in enumerate(expected_components, start=1):
            if component_by_position.get(position) != {component}:
                raise AdapterError(
                    f"ESMFold2 output polymer {chain_id} changed residue chemistry"
                )

    _validate_output_modified_atom_allowlists(arrays, sequences)

    if raw_confidence_path is not None or raw_confidence_arrays is not None:
        _validate_output_atom_inventory(
            arrays,
            sequences,
            raw_confidence_path,
            raw_confidence_arrays=raw_confidence_arrays,
        )

    bonds = input_payload.get("covalent_bonds", [])
    if not isinstance(bonds, list):
        raise AdapterError("ESMFold2 private covalent-bond ledger is malformed")
    if bonds and validate_covalent_topology:
        _validate_output_covalent_topology(blocks[0], sequences, bonds)

    if ccd_reader is not None:
        _validate_output_ccd_bonds(
            blocks[0],
            sequences,
            ccd_reader,
        )


def _validate_output_modified_atom_allowlists(
    output_columns: dict[str, np.ndarray],
    sequences: list[dict[str, Any]],
) -> None:
    """Check strict modified occurrences at the serialized atom-site boundary."""

    chain_values = output_columns["label_asym_id"]
    position_values = output_columns["label_seq_id"]
    component_values = output_columns["label_comp_id"]
    atom_values = output_columns["label_atom_id"]
    for entry in sequences:
        if not isinstance(entry, dict) or entry.get("modality") not in _POLYMER_MOL_TYPE:
            continue
        modifications = _validated_private_modifications(entry)
        for modification in modifications:
            if set(modification) == _MODIFICATION_LEGACY_KEYS:
                continue
            chain_id = entry["chain_id"]
            position = modification["position"]
            rows = [
                index
                for index, (observed_chain, observed_position) in enumerate(
                    zip(chain_values, position_values, strict=True)
                )
                if str(observed_chain) == chain_id
                and str(observed_position) == str(position + 1)
            ]
            observed_names = [str(atom_values[index]) for index in rows]
            expected_names = modification["atom_names"]
            if (
                len(observed_names) != len(expected_names)
                or set(observed_names) != set(expected_names)
            ):
                missing = sorted(set(expected_names).difference(observed_names))
                extra = sorted(set(observed_names).difference(expected_names))
                raise AdapterError(
                    "ESMFold2 output strict modified-polymer atom allowlist differs "
                    f"for {chain_id}:{position + 1}: missing={missing}, "
                    f"extra={extra}"
                )
            observed_components = {str(component_values[index]) for index in rows}
            if observed_components != {modification["ccd"]}:
                raise AdapterError(
                    "ESMFold2 output strict modified-polymer CCD identity differs "
                    f"for {chain_id}:{position + 1}"
                )


def _output_covalent_endpoint(
    endpoint: Any,
    by_chain: dict[str, dict[str, Any]],
) -> tuple[str, str, str, str]:
    if not isinstance(endpoint, dict) or set(endpoint) != {
        "chain_id",
        "residue_index",
        "atom_name",
    }:
        raise AdapterError("ESMFold2 private covalent endpoint is malformed")
    chain_id = endpoint.get("chain_id")
    residue_index = endpoint.get("residue_index")
    atom_name = endpoint.get("atom_name")
    chain = by_chain.get(chain_id)
    if (
        chain is None
        or not isinstance(residue_index, int)
        or isinstance(residue_index, bool)
        or residue_index < 0
        or not isinstance(atom_name, str)
        or not atom_name
        or atom_name.strip() != atom_name
    ):
        raise AdapterError("ESMFold2 private covalent endpoint is malformed")
    component = _component_at_position(chain, residue_index)
    return chain_id, component, str(residue_index + 1), atom_name


def _validate_output_covalent_topology(
    block: Any,
    sequences: list[dict[str, Any]],
    bonds: list[dict[str, Any]],
) -> None:
    """Validate expected links while allowing unrelated authoritative rows."""

    try:
        category = block["struct_conn"]
        names = {
            field: category[field].as_array(str)
            for field in (
                "conn_type_id",
                "pdbx_value_order",
                "ptnr1_label_asym_id",
                "ptnr2_label_asym_id",
                "ptnr1_label_comp_id",
                "ptnr2_label_comp_id",
                "ptnr1_label_seq_id",
                "ptnr2_label_seq_id",
                "ptnr1_label_atom_id",
                "ptnr2_label_atom_id",
            )
        }
    except Exception as exc:
        raise AdapterError(
            f"ESMFold2 output lacks explicit covalent topology: {exc}"
        ) from exc
    row_count = len(next(iter(names.values())))
    if any(len(values) != row_count for values in names.values()):
        raise AdapterError("ESMFold2 output explicit covalent topology is ragged")

    observed: dict[
        tuple[tuple[str, str, str, str], tuple[str, str, str, str]],
        list[tuple[str, str]],
    ] = {}
    for row in range(row_count):
        left = tuple(
            str(names[f"ptnr1_{field}"][row])
            for field in ("label_asym_id", "label_comp_id", "label_seq_id", "label_atom_id")
        )
        right = tuple(
            str(names[f"ptnr2_{field}"][row])
            for field in ("label_asym_id", "label_comp_id", "label_seq_id", "label_atom_id")
        )
        if any(value in {"", ".", "?"} for value in (*left, *right)):
            raise AdapterError("ESMFold2 output covalent endpoint is empty")
        key = tuple(sorted((left, right)))  # type: ignore[assignment]
        observed.setdefault(key, []).append(
            (str(names["conn_type_id"][row]), str(names["pdbx_value_order"][row]))
        )

    by_chain = {entry.get("chain_id"): entry for entry in sequences}
    expected: dict[
        tuple[tuple[str, str, str, str], tuple[str, str, str, str]], str
    ] = {}
    for bond in bonds:
        if not isinstance(bond, dict) or set(bond) != {
            "endpoint1",
            "endpoint2",
            "bond_order",
        }:
            raise AdapterError("ESMFold2 private covalent bond is malformed")
        order = _mmcif_covalent_order(bond.get("bond_order"))
        left = _output_covalent_endpoint(bond["endpoint1"], by_chain)
        right = _output_covalent_endpoint(bond["endpoint2"], by_chain)
        key = tuple(sorted((left, right)))  # type: ignore[assignment]
        previous = expected.get(key)
        if previous is not None and previous != order:
            raise AdapterError("ESMFold2 private covalent bond order conflicts")
        expected[key] = order

    for key, order in expected.items():
        rows = observed.get(key)
        if rows is None:
            raise AdapterError("ESMFold2 output lacks explicit covalent topology")
        if len(rows) != 1:
            raise AdapterError("ESMFold2 output contains duplicate covalent topology")
        conn_type, observed_order = rows[0]
        if conn_type != "covale" or observed_order != order:
            raise AdapterError("ESMFold2 output covalent bond order/type differs")


def _output_component_atom_intersection(block: Any) -> dict[str, frozenset[str]]:
    try:
        atom_site = block["atom_site"]
        values = tuple(
            zip(
                atom_site["label_comp_id"].as_array(str),
                atom_site["label_asym_id"].as_array(str),
                atom_site["label_seq_id"].as_array(str),
                atom_site["label_atom_id"].as_array(str),
                strict=True,
            )
        )
    except Exception as exc:
        raise AdapterError(
            f"ESMFold2 output component atom inventory is unavailable: {exc}"
        ) from exc
    instances: dict[tuple[str, str, str], set[str]] = {}
    for component, asym, sequence, atom in values:
        key = (str(component), str(asym), str(sequence))
        instances.setdefault(key, set()).add(str(atom))
    by_component: dict[str, list[set[str]]] = {}
    for (component, _asym, _sequence), names in instances.items():
        by_component.setdefault(component, []).append(names)
    return {
        component: frozenset(set.intersection(*names))
        for component, names in by_component.items()
    }


def _validate_output_ccd_bonds(
    block: Any,
    sequences: list[dict[str, Any]],
    ccd_reader: CCDBondCacheReader,
) -> None:
    """Verify CCD internal orders on the atoms present in every occurrence."""

    component_ids: list[str] = []
    for entry in sequences:
        if not isinstance(entry, dict):
            raise AdapterError("ESMFold2 private sequence ledger is malformed")
        if entry.get("modality") == "ligand":
            components = entry.get("ccd")
            if isinstance(components, list):
                component_ids.extend(
                    component for component in components if isinstance(component, str)
                )
        elif entry.get("modality") in _POLYMER_MOL_TYPE:
            component_ids.extend(
                modification["ccd"]
                for modification in _validated_private_modifications(entry)
            )
    component_ids = list(dict.fromkeys(component_ids))
    if not component_ids:
        return
    present = _output_component_atom_intersection(block)
    try:
        category = block.get("chem_comp_bond")
    except AttributeError:
        category = None
    observed_by_component: dict[
        str, dict[tuple[str, str], tuple[str, str, str]]
    ] = {}
    if category is not None:
        required = (
            "comp_id",
            "atom_id_1",
            "atom_id_2",
            "value_order",
            "pdbx_aromatic_flag",
            "pdbx_stereo_config",
        )
        try:
            arrays = {name: category[name].as_array(str) for name in required}
        except Exception as exc:
            raise AdapterError(
                f"ESMFold2 output chem_comp_bond category is incomplete: {exc}"
            ) from exc
        row_count = len(next(iter(arrays.values())))
        if any(len(values) != row_count for values in arrays.values()):
            raise AdapterError("ESMFold2 output chem_comp_bond category is ragged")
        for row in range(row_count):
            component = str(arrays["comp_id"][row])
            if component not in component_ids:
                continue
            atom_pair = tuple(
                sorted(
                    (
                        str(arrays["atom_id_1"][row]),
                        str(arrays["atom_id_2"][row]),
                    )
                )
            )
            if (
                atom_pair[0] in {"", ".", "?"}
                or atom_pair[1] in {"", ".", "?"}
                or atom_pair[0] == atom_pair[1]
                or component not in present
                or not set(atom_pair).issubset(present[component])
            ):
                raise AdapterError(
                    f"ESMFold2 output chem_comp_bond {component!r} uses an absent atom"
                )
            signature = (
                str(arrays["value_order"][row]),
                str(arrays["pdbx_aromatic_flag"][row]),
                str(arrays["pdbx_stereo_config"][row]),
            )
            component_rows = observed_by_component.setdefault(component, {})
            previous = component_rows.get(atom_pair)
            if previous is not None:
                if previous != signature:
                    raise AdapterError(
                        "ESMFold2 output chem_comp_bond has a hard order conflict"
                    )
                raise AdapterError("ESMFold2 output chem_comp_bond has duplicate rows")
            component_rows[atom_pair] = signature

    for component in component_ids:
        expected_rows = {}
        try:
            ccd_bonds = ccd_reader.query(component)
        except Exception as exc:
            raise AdapterError(
                f"ESMFold2 CCD bond authority rejected component {component!r}: {exc}"
            ) from exc
        for bond in ccd_bonds:
            atom_pair = tuple(sorted((bond.atom_id_1, bond.atom_id_2)))
            atoms = present.get(component, frozenset())
            if not set(atom_pair).issubset(atoms):
                continue
            expected_rows[atom_pair] = (
                bond.value_order,
                bond.aromatic_flag,
                bond.stereo_config,
            )
        observed_rows = observed_by_component.get(component, {})
        if observed_rows != expected_rows:
            raise AdapterError(
                f"ESMFold2 output CCD bond graph/order differs for {component!r}"
            )


def _validate_output_atom_inventory(
    output_columns: dict[str, np.ndarray],
    sequences: list[dict[str, Any]],
    raw_confidence_path: Path | None,
    *,
    raw_confidence_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    required = {
        "token_attention_mask",
        "atom_attention_mask",
        "asym_id",
        "residue_index",
        "atom_to_token",
        "ref_element",
        "ref_atom_name_chars",
    }
    if raw_confidence_arrays is not None:
        if not required.issubset(raw_confidence_arrays):
            raise AdapterError("ESMFold2 raw evidence lacks prepared atom inventory")
        raw = {
            name: np.asarray(raw_confidence_arrays[name])
            for name in required
        }
    else:
        if raw_confidence_path is None:
            raise AdapterError("ESMFold2 prepared atom inventory path is missing")
        try:
            with np.load(raw_confidence_path, allow_pickle=False) as archive:
                if not required.issubset(archive.files):
                    raise AdapterError(
                        "ESMFold2 raw evidence lacks prepared atom inventory"
                    )
                raw = {name: np.asarray(archive[name]) for name in required}
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(
                f"Cannot read ESMFold2 prepared atom inventory: {exc}"
            ) from exc

    token_mask = raw["token_attention_mask"].astype(bool, copy=False)
    atom_mask = raw["atom_attention_mask"].astype(bool, copy=False)
    if token_mask.ndim != 1 or atom_mask.ndim != 1:
        raise AdapterError("ESMFold2 prepared atom inventory masks are malformed")
    atom_indices = np.flatnonzero(atom_mask)
    atom_to_token = raw["atom_to_token"].astype(int, copy=False)
    if atom_to_token.shape != atom_mask.shape:
        raise AdapterError("ESMFold2 prepared atom-to-token shape differs")
    token_indices = atom_to_token[atom_indices]
    if (
        np.any(token_indices < 0)
        or np.any(token_indices >= token_mask.size)
        or not np.all(token_mask[token_indices])
    ):
        raise AdapterError("ESMFold2 prepared atom-to-token values are invalid")
    try:
        names = decode_atom_name_rows(
            raw["ref_atom_name_chars"][atom_indices]
        )
    except ValueError as exc:
        raise AdapterError(str(exc)) from exc
    asym_id = raw["asym_id"].astype(int, copy=False)
    residue_index = raw["residue_index"].astype(int, copy=False)
    elements = raw["ref_element"].astype(int, copy=False)[atom_indices]
    expected: Counter[tuple[str, int, str, str, int]] = Counter()
    for token, atom_name, element in zip(
        token_indices, names, elements.tolist(), strict=True
    ):
        chain_index = int(asym_id[token])
        position = int(residue_index[token])
        if not 0 <= chain_index < len(sequences):
            raise AdapterError("ESMFold2 prepared atom chain index is invalid")
        entry = sequences[chain_index]
        component = _component_at_position(entry, position)
        if not 1 <= int(element) <= 118:
            raise AdapterError("ESMFold2 prepared atom element is invalid")
        expected[
            (entry["chain_id"], position + 1, component, atom_name, int(element))
        ] += 1

    observed: Counter[tuple[str, int, str, str, int]] = Counter()
    row_count = len(output_columns["label_asym_id"])
    for row in range(row_count):
        raw_position = str(output_columns["label_seq_id"][row])
        try:
            position = int(raw_position)
        except ValueError as exc:
            raise AdapterError(
                "ESMFold2 output atom inventory has invalid label_seq_id"
            ) from exc
        if position < 1 or str(position) != raw_position:
            raise AdapterError(
                "ESMFold2 output atom inventory has invalid label_seq_id"
            )
        try:
            element = atomic_number(str(output_columns["type_symbol"][row]))
        except ValueError as exc:
            raise AdapterError(str(exc)) from exc
        key = (
            str(output_columns["label_asym_id"][row]),
            position,
            str(output_columns["label_comp_id"][row]),
            str(output_columns["label_atom_id"][row]),
            element,
        )
        if any(value in {"", ".", "?"} for value in key[:1] + key[2:4]):
            raise AdapterError("ESMFold2 output atom inventory has an empty key")
        observed[key] += 1
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        raise AdapterError(
            "ESMFold2 output atom inventory differs from prepared model input: "
            f"missing={sum(missing.values())}, extra={sum(extra.values())}"
        )


def _component_at_position(entry: dict[str, Any], position: int) -> str:
    if not isinstance(position, int) or isinstance(position, bool) or position < 0:
        raise AdapterError("ESMFold2 prepared residue index is invalid")
    if entry.get("modality") == "ligand":
        if isinstance(entry.get("smiles"), str) and entry["smiles"]:
            components = ["LIG"]
        else:
            components = entry.get("ccd")
    else:
        components = _expected_polymer_components(entry)
    if not isinstance(components, list) or not 0 <= position < len(components):
        raise AdapterError("ESMFold2 prepared residue/component ledger differs")
    component = components[position]
    if not isinstance(component, str) or not component:
        raise AdapterError("ESMFold2 prepared component identity is malformed")
    return component


def _expected_polymer_components(entry: dict[str, Any]) -> list[str]:
    modality = entry.get("modality")
    sequence = entry.get("sequence")
    if modality not in _POLYMER_MOL_TYPE or not isinstance(sequence, str) or not sequence:
        raise AdapterError("ESMFold2 private polymer chemistry is malformed")
    modifications = _validated_private_modifications(entry)
    modification_by_position: dict[int, str] = {}
    for modification in modifications:
        position = modification["position"]
        ccd = modification["ccd"]
        modification_by_position[position] = ccd

    components: list[str] = []
    for position, symbol in enumerate(sequence):
        if position in modification_by_position:
            components.append(modification_by_position[position])
            continue
        if modality == "protein":
            component = _PROTEIN_COMPONENT.get(symbol)
        elif modality == "dna":
            component = {
                "A": "DA",
                "C": "DC",
                "G": "DG",
                "T": "DT",
                "N": "UNK",
            }.get(symbol)
        else:
            component = (
                symbol
                if symbol in {"A", "C", "G", "U"}
                else "UNK" if symbol == "N" else None
            )
        if component is None:
            raise AdapterError(
                f"ESMFold2 private {modality} sequence contains unsupported symbol "
                f"{symbol!r} at position {position + 1}"
            )
        components.append(component)
    return components


def _validated_private_modifications(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate legacy and strict modified-polymer occurrence ledgers."""

    sequence = entry.get("sequence")
    if not isinstance(sequence, str) or not sequence:
        raise AdapterError("ESMFold2 private polymer sequence is malformed")
    modifications = entry.get("modifications", [])
    if modifications is None:
        modifications = []
    if not isinstance(modifications, list):
        raise AdapterError("ESMFold2 private modification ledger is malformed")
    result: list[dict[str, Any]] = []
    positions: set[int] = set()
    for modification in modifications:
        if not isinstance(modification, dict):
            raise AdapterError("ESMFold2 private modification entry is malformed")
        keys = frozenset(modification)
        if keys not in {_MODIFICATION_LEGACY_KEYS, _MODIFICATION_STRICT_KEYS}:
            raise AdapterError(
                "ESMFold2 private modification entry has unsupported fields"
            )
        position = modification.get("position")
        ccd = modification.get("ccd")
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or not 0 <= position < len(sequence)
            or position in positions
            or not isinstance(ccd, str)
            or not ccd
            or ccd.strip() != ccd
        ):
            raise AdapterError("ESMFold2 private modification entry is invalid")
        positions.add(position)
        if keys == _MODIFICATION_STRICT_KEYS:
            atom_names = modification.get("atom_names")
            if (
                not isinstance(atom_names, list)
                or not atom_names
                or len(atom_names) != len(set(atom_names))
                or any(
                    not isinstance(name, str)
                    or not name
                    or name.strip() != name
                    or len(name) > 4
                    for name in atom_names
                )
                or not isinstance(
                    modification.get("infer_standard_polymer_bonds"), bool
                )
                or modification["infer_standard_polymer_bonds"]
            ):
                raise AdapterError(
                    "ESMFold2 private strict modification atom ledger is invalid"
                )
            result.append(
                {
                    "position": position,
                    "ccd": ccd,
                    "atom_names": list(atom_names),
                    "infer_standard_polymer_bonds": False,
                }
            )
        else:
            result.append({"position": position, "ccd": ccd})
    return result


def _safe_input(path: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise AdapterError(f"ESMFold2 private input is missing or unsafe: {raw}")
    return raw.resolve()


def _format_process_log(
    mode: str,
    batch_record: dict[str, Any],
    argv: tuple[str, ...],
    exit_code: int | str,
    request_status: str,
    request_error: str,
    stdout: str,
    stderr: str,
) -> str:
    return (
        "execution_mode=persistent-batch\n"
        "network_policy=fold-offline\n"
        "model=esmfold2\n"
        f"mode={mode}\n"
        f"batch_record={json.dumps(batch_record, sort_keys=True, allow_nan=False)}\n"
        f"argv={json.dumps(list(argv), allow_nan=False)}\n"
        f"exit_code={exit_code}\n"
        f"request_status={request_status}\n"
        f"request_error={json.dumps(request_error, allow_nan=False)}\n"
        "--- stdout ---\n"
        f"{stdout}\n"
        "--- stderr ---\n"
        f"{stderr}"
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _timeout_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def _sanitized_worker_text(value: str, *, limit: int = 2_000) -> str:
    """Return a bounded printable worker diagnostic for a run-fatal error."""

    sanitized = re.sub(
        r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@",
        r"\1<redacted>@",
        value,
    )
    sanitized = re.sub(
        r"(?i)\b(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+",
        r"\1<redacted>",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\bbearer\s+[a-z0-9._~+/=-]+",
        "Bearer <redacted>",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\b(access[_-]?token|refresh[_-]?token|token|password|passwd|"
        r"api[_-]?key|secret)\b(\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
        r"\1\2<redacted>",
        sanitized,
    )
    sanitized = "".join(
        character if character.isprintable() else " " for character in sanitized
    )
    return sanitized[-limit:].strip() or "unavailable"
