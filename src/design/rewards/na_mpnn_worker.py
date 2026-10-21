"""One-process, one-model-load NA-MPNN JSON batch worker.

The protocol is intentionally private and small.  It carries a temporary PDB
text/path plus one private chain alias; public PartiDesign selectors never
cross this boundary.  Candidate input/result errors are isolated, while asset
or model-load errors abort the batch.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .na_mpnn_runtime import (
    NAMPSequenceError,
    NAMPSystemError,
    NAMPNNError,
    LoadedNAMPNN,
    load_na_mpnn_model,
)
from .aa_omission import resolve_rm_aa


_TOP_LEVEL_KEYS = frozenset({"schema_version", "requests", "rm_aa"})
_CANDIDATE_KEYS = frozenset(
    {
        "candidate_id",
        "pdb_text",
        "pdb_path",
        "design_chain",
        "designable_positions",
        "fixed_positions",
        "seed",
        "temperature",
    }
)


@dataclass(frozen=True)
class CandidateRequest:
    candidate_id: str
    pdb_text: str | None
    pdb_path: Path | None
    design_chain: str
    designable_positions: tuple[int, ...]
    fixed_positions: tuple[int, ...]
    seed: int
    temperature: float


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    return value


def _parse_positions(value: Any, *, alias: str, where: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list")
    positions: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{where} must contain positive integer positions")
        if isinstance(item, int):
            position = item
        elif isinstance(item, str) and item.startswith(alias):
            suffix = item[len(alias) :]
            if not suffix.isdigit():
                raise ValueError(f"{where} contains an invalid selector: {item!r}")
            position = int(suffix)
        else:
            raise ValueError(f"{where} must contain integers or {alias}<position>")
        if position < 1 or position in positions:
            raise ValueError(f"{where} contains a duplicate or non-positive position")
        positions.append(position)
    return tuple(sorted(positions))


def _candidate_request(value: Any) -> CandidateRequest:
    raw = _require_mapping(value, "candidate request")
    unknown = set(raw) - _CANDIDATE_KEYS
    if unknown:
        raise ValueError(f"candidate request has unknown keys: {sorted(unknown)}")
    required = {"candidate_id", "design_chain", "seed", "temperature"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"candidate request is missing keys: {sorted(missing)}")
    candidate_id = raw["candidate_id"]
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a non-empty string")
    design_chain = raw["design_chain"]
    if (
        not isinstance(design_chain, str)
        or len(design_chain) != 1
        or design_chain.isspace()
    ):
        raise ValueError("design_chain must be one non-whitespace private alias")
    has_text = "pdb_text" in raw
    has_path = "pdb_path" in raw
    if has_text == has_path:
        raise ValueError("provide exactly one of pdb_text or pdb_path")
    pdb_text: str | None = None
    pdb_path: Path | None = None
    if has_text:
        pdb_text = raw["pdb_text"]
        if not isinstance(pdb_text, str) or not pdb_text.strip():
            raise ValueError("pdb_text must be non-empty text")
    else:
        value_path = raw["pdb_path"]
        if not isinstance(value_path, str) or not value_path.strip():
            raise ValueError("pdb_path must be a non-empty string")
        pdb_path = Path(value_path).expanduser()
        try:
            if pdb_path.is_symlink() or not pdb_path.is_file():
                raise ValueError("pdb_path must name a regular non-symlink file")
        except OSError as exc:
            raise ValueError("pdb_path is unavailable") from exc
    designable = _parse_positions(
        raw.get("designable_positions"), alias=design_chain, where="designable_positions"
    )
    fixed = _parse_positions(
        raw.get("fixed_positions"), alias=design_chain, where="fixed_positions"
    )
    if set(designable) & set(fixed):
        raise ValueError("designable_positions and fixed_positions overlap")
    if not designable and not fixed:
        raise ValueError("at least one designable or fixed position is required")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    temperature = raw["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be numeric")
    temperature = float(temperature)
    if not math.isfinite(temperature) or not 0.0 < temperature <= 10.0:
        raise ValueError("temperature must be finite and in (0, 10]")
    return CandidateRequest(
        candidate_id=candidate_id,
        pdb_text=pdb_text,
        pdb_path=pdb_path,
        design_chain=design_chain,
        designable_positions=designable,
        fixed_positions=fixed,
        seed=seed,
        temperature=temperature,
    )


def _validate_batch(payload: Any) -> list[Any]:
    raw = _require_mapping(payload, "NA-MPNN batch request")
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"NA-MPNN batch request has unknown keys: {sorted(unknown)}")
    if raw.get("schema_version") != 1:
        raise ValueError("NA-MPNN batch schema_version must be 1")
    # ``rm_aa`` is a protein-only omission policy.  Validate and carry it at
    # the protocol boundary, but deliberately do not translate it into the
    # RNA/DNA vocabulary used by this worker.
    if "rm_aa" in raw:
        resolve_rm_aa(raw)
    requests = raw.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("NA-MPNN batch requests must be a non-empty list")
    candidate_ids: set[str] = set()
    for request in requests:
        if not isinstance(request, Mapping):
            # The malformed request is isolated by ``run_batch``.  Batch-level
            # validation owns only the envelope and duplicate IDs that can be
            # checked without interpreting a candidate's chemistry fields.
            continue
        candidate_id = request.get("candidate_id")
        if isinstance(candidate_id, str):
            if candidate_id in candidate_ids:
                raise ValueError(f"duplicate candidate_id: {candidate_id}")
            candidate_ids.add(candidate_id)
    return requests


def _seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_pdb(request: CandidateRequest, directory: Path) -> Path:
    if request.pdb_text is not None:
        path = directory / "input.pdb"
        path.write_text(request.pdb_text, encoding="utf-8")
        return path
    assert request.pdb_path is not None
    return request.pdb_path


def _canonical_base(token: int, *, is_rna: bool, loaded: LoadedNAMPNN) -> str:
    try:
        value = loaded.vocab.restype_int_to_str[int(token)]
    except (KeyError, TypeError, ValueError) as exc:
        raise NAMPSequenceError(f"NA-MPNN emitted an unknown token: {token!r}") from exc
    if is_rna:
        mapping = {"a": "A", "c": "C", "g": "G", "t": "U"}
    else:
        mapping = {"a": "A", "c": "C", "g": "G", "t": "T"}
    try:
        return mapping[value.lower()]
    except KeyError as exc:
        raise NAMPSequenceError(
            f"NA-MPNN emitted non-{('RNA' if is_rna else 'DNA')} token {value!r}"
        ) from exc


@dataclass
class _PreparedCandidate:
    request: CandidateRequest
    feature: dict[str, Any]
    rows: tuple[int, ...]
    is_rna: bool
    native_sequence: str
    fixed_positions: frozenset[int]


def _prepare_candidate(
    loaded: LoadedNAMPNN, request: CandidateRequest
) -> _PreparedCandidate:
    import torch

    _seed(request.seed)
    with tempfile.TemporaryDirectory(prefix="partidesign-na-mpnn-") as temporary:
        pdb_path = _read_pdb(request, Path(temporary))
        try:
            parsed, _backbone, _other_atoms, _icodes, _water = loaded.data_utils.parse_PDB(
                str(pdb_path),
                device=loaded.device,
                chains=[],
                model_type="na_mpnn",
                parse_na_only=False,
                na_shared_tokens=True,
                load_residues_with_missing_atoms=1,
            )
        except Exception as exc:
            raise NAMPSequenceError(f"cannot parse candidate PDB: {exc}") from exc
        chain_letters = [str(value) for value in parsed["chain_letters"]]
        rows = tuple(
            index for index, value in enumerate(chain_letters)
            if value == request.design_chain
        )
        if not rows:
            raise NAMPSequenceError(f"design chain alias is absent: {request.design_chain}")
        dna_mask = parsed["dna_mask"].detach().cpu().numpy().astype(bool)
        rna_mask = parsed["rna_mask"].detach().cpu().numpy().astype(bool)
        dna_rows = bool(np.all(dna_mask[list(rows)]))
        rna_rows = bool(np.all(rna_mask[list(rows)]))
        if dna_rows == rna_rows:
            raise NAMPSequenceError(
                "design chain must be exactly one standard DNA or RNA chain"
            )
        length = len(rows)
        designable = set(request.designable_positions)
        fixed = set(request.fixed_positions)
        for position in (*designable, *fixed):
            if position > length:
                raise NAMPSequenceError(
                    f"position {request.design_chain}{position} exceeds "
                    f"design-chain length {length}"
                )
        if not designable:
            designable = set(range(1, length + 1)) - fixed
        if not designable:
            raise NAMPSequenceError(
                "no designable positions remain after fixed constraints"
            )
        fixed.update(set(range(1, length + 1)) - designable)
        if designable & fixed:
            raise NAMPSequenceError("designable/fixed position overlap")

        native = parsed["S"].detach().cpu().numpy()
        native_sequence = "".join(
            _canonical_base(native[row], is_rna=rna_rows, loaded=loaded)
            for row in rows
        )
        chain_mask = torch.zeros_like(parsed["mask"], dtype=torch.int32)
        for offset, row in enumerate(rows, 1):
            if offset in designable:
                chain_mask[row] = 1
        parsed["chain_mask"] = chain_mask
        feature = loaded.data_utils.featurize(parsed)
        feature["chain_mask"] = chain_mask[None, :]
        feature["batch_size"] = 1
        feature["temperature"] = request.temperature
        feature["bias"] = torch.zeros(
            (1, feature["S"].shape[1], loaded.vocab.num_letters),
            dtype=torch.float32,
            device=loaded.device,
        )
        feature["symmetry_residues"] = [[]]
        feature["symmetry_weights"] = [[]]
        feature["randn"] = torch.randn(
            (1, feature["mask"].shape[1]), device=loaded.device
        )
        return _PreparedCandidate(
            request=request,
            feature=feature,
            rows=rows,
            is_rna=rna_rows,
            native_sequence=native_sequence,
            fixed_positions=frozenset(fixed),
        )


def _decode_sampled_sequence(
    loaded: LoadedNAMPNN,
    prepared: _PreparedCandidate,
    sampled: Mapping[str, Any],
    *,
    sample_index: int,
) -> str:
    import torch

    if not isinstance(sampled, Mapping) or "S" not in sampled:
        raise NAMPSequenceError("NA-MPNN result lacks the sampled S tensor")
    values = sampled["S"]
    if (
        not isinstance(values, torch.Tensor)
        or values.ndim != 2
        or sample_index < 0
        or sample_index >= values.shape[0]
    ):
        raise NAMPSequenceError("NA-MPNN result S must have shape [batch, length]")
    if values.shape[1] != prepared.feature["S"].shape[1]:
        raise NAMPSequenceError("NA-MPNN result length does not match the input")
    generated = values[sample_index].detach().cpu().numpy()
    output_sequence = "".join(
        _canonical_base(generated[row], is_rna=prepared.is_rna, loaded=loaded)
        for row in prepared.rows
    )
    for position in sorted(prepared.fixed_positions):
        if output_sequence[position - 1] != prepared.native_sequence[position - 1]:
            raise NAMPSequenceError(
                f"NA-MPNN changed fixed residue "
                f"{prepared.request.design_chain}{position}"
            )
    return output_sequence


def _sample_candidate(loaded: LoadedNAMPNN, request: CandidateRequest) -> str:
    import torch

    prepared = _prepare_candidate(loaded, request)
    try:
        with torch.no_grad():
            sampled = loaded.model.sample(prepared.feature)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise NAMPSequenceError(
            f"NA-MPNN candidate sampling failed: {exc}"
        ) from exc
    return _decode_sampled_sequence(
        loaded, prepared, sampled, sample_index=0
    )


def _stack_batch_features(
    prepared: list[_PreparedCandidate],
) -> dict[str, Any]:
    import torch

    if not prepared:
        raise ValueError("cannot stack an empty NA-MPNN batch")
    keys = tuple(prepared[0].feature)
    key_set = set(keys)
    if any(set(item.feature) != key_set for item in prepared[1:]):
        raise NAMPSequenceError("NA-MPNN candidates produced different feature keys")
    feature: dict[str, Any] = {}
    for key in keys:
        values = [item.feature[key] for item in prepared]
        first = values[0]
        if isinstance(first, torch.Tensor):
            if any(
                not isinstance(value, torch.Tensor)
                or value.ndim == 0
                or value.shape[0] != 1
                for value in values
            ):
                raise NAMPSequenceError(
                    f"NA-MPNN feature {key!r} is not stackable"
                )
            try:
                feature[key] = torch.cat(values, dim=0)
            except (RuntimeError, TypeError) as exc:
                raise NAMPSequenceError(
                    f"NA-MPNN feature {key!r} has incompatible shapes"
                ) from exc
        elif key in {"batch_size", "temperature", "symmetry_residues", "symmetry_weights"}:
            continue
        elif any(value != first for value in values):
            raise NAMPSequenceError(
                f"NA-MPNN control feature {key!r} differs across candidates"
            )
        else:
            feature[key] = first
    device = feature["S"].device
    feature["batch_size"] = len(prepared)
    feature["temperature"] = torch.tensor(
        [[item.request.temperature] for item in prepared],
        dtype=torch.float32,
        device=device,
    )
    feature["symmetry_residues"] = [[]]
    feature["symmetry_weights"] = [[]]
    return feature


def _sample_candidates(
    loaded: LoadedNAMPNN, requests: list[CandidateRequest]
) -> list[dict[str, Any]]:
    """Sample parsed requests in true batches grouped by input length and shared seed."""

    prepared_groups: dict[tuple[int, int], list[_PreparedCandidate]] = {}
    results_by_id: dict[str, dict[str, Any]] = {}
    for request in requests:
        try:
            prepared = _prepare_candidate(loaded, request)
        except (NAMPSequenceError, ValueError, TypeError, OSError) as exc:
            results_by_id[request.candidate_id] = {
                "candidate_id": request.candidate_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        length = int(prepared.feature["S"].shape[1])
        prepared_groups.setdefault((length, request.seed), []).append(prepared)

    import torch

    for group in prepared_groups.values():
        feature = _stack_batch_features(group)
        _seed(group[0].request.seed)
        try:
            # Preparation seeds each request separately, so concatenating its
            # noise repeats the same decoding order for equal-length slots.
            # Draw the real batch with a separate generator to leave the
            # categorical RNG stream and single-slot behavior unchanged.
            random_order = feature["randn"]
            order_generator = torch.Generator(device=random_order.device)
            order_generator.manual_seed(group[0].request.seed)
            feature["randn"] = torch.randn(
                random_order.shape,
                dtype=random_order.dtype,
                device=random_order.device,
                generator=order_generator,
            )
            with torch.no_grad():
                sampled = loaded.model.sample_batch(feature)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            for prepared in group:
                results_by_id[prepared.request.candidate_id] = {
                    "candidate_id": prepared.request.candidate_id,
                    "status": "failed",
                    "error": error,
                }
            continue
        for index, prepared in enumerate(group):
            try:
                sequence = _decode_sampled_sequence(
                    loaded, prepared, sampled, sample_index=index
                )
            except (NAMPSequenceError, ValueError, TypeError, OSError) as exc:
                results_by_id[prepared.request.candidate_id] = {
                    "candidate_id": prepared.request.candidate_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            else:
                results_by_id[prepared.request.candidate_id] = {
                    "candidate_id": prepared.request.candidate_id,
                    "status": "ok",
                    "sequence": sequence,
                }
    return [results_by_id[request.candidate_id] for request in requests]


def _run_loaded_batch(
    loaded: LoadedNAMPNN, raw_requests: list[Any]
) -> list[dict[str, Any]]:
    parsed_entries: list[tuple[int, CandidateRequest]] = []
    results_by_index: dict[int, dict[str, Any]] = {}
    for index, raw_request in enumerate(raw_requests):
        try:
            parsed_entries.append((index, _candidate_request(raw_request)))
        except (NAMPSequenceError, ValueError, TypeError, OSError) as exc:
            candidate_id = (
                str(raw_request.get("candidate_id", "<unknown>"))
                if isinstance(raw_request, Mapping)
                else "<unknown>"
            )
            results_by_index[index] = {
                "candidate_id": candidate_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    requests = [request for _index, request in parsed_entries]
    if callable(getattr(loaded.model, "sample_batch", None)):
        sampled_results = _sample_candidates(loaded, requests)
    else:
        sampled_results = []
        for request in requests:
            try:
                sampled_results.append(
                    {
                        "candidate_id": request.candidate_id,
                        "status": "ok",
                        "sequence": _sample_candidate(loaded, request),
                    }
                )
            except (NAMPSequenceError, ValueError, TypeError, OSError) as exc:
                sampled_results.append(
                    {
                        "candidate_id": request.candidate_id,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    for (index, _request), result in zip(parsed_entries, sampled_results):
        results_by_index[index] = result
    return [results_by_index[index] for index in range(len(raw_requests))]


def run_batch(
    payload: Any,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Run one closed batch; model loading is performed exactly once."""

    requests = _validate_batch(payload)
    loaded = load_na_mpnn_model(repo_root)
    return {
        "schema_version": 1,
        "model_loads": 1,
        "results": _run_loaded_batch(loaded, requests),
    }



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    result = run_batch(payload)
    Path(args.result).write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
