"""Owner-local offline MPNN + ESMFold2 reward execution."""

from __future__ import annotations

import uuid
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from partidesign.paths import (community_source, community_asset, asset_root, cache_dir, offline_site, code_root)
from typing import Any, Mapping, Sequence

import numpy as np
from biotite.structure import AtomArray

from ..core.records import SearchCandidate
from ..utils.reader import read_structure
from .esmfold2 import ESMFold2Confidence, score_confidences
from .sequence import plan_reward_sequence_source, normalize_reward_sequence_config
from .esmfold2_input import (
    build_esmfold2_input,
    candidate_chain_modalities,
)
from .mpnn import MPNNSequenceError, MPNNSystemError, generate_mpnn_sequences
from .esmfold2_mapping import SOURCE_MAPPING, mapped_confidence_summary, restore_source_output

def plan_sequence_source(*args: Any, **kwargs: Any) -> Any:
    """Compatibility seam forwarding runner planning to the reward helper."""
    return plan_reward_sequence_source(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class FoldRewardResult:
    candidate_id: str
    score: float
    confidence: ESMFold2Confidence
    fold_cif: Path
    sequence_source: str
    cache: Mapping[str, Any]
    design_sequence: str | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FoldRewardFailure:
    candidate_id: str
    message: str


def _load_pipetools(repo_root: Path) -> tuple[Any, Any, Any, Any]:
    source = community_source("pipetools", repo_root)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from pipetools.adapters.esmfold2 import (  # type: ignore[import-not-found]
        ESMFold2BatchFoldRequest,
        run_esmfold2_batch,
    )
    from pipetools.config import ESMFold2Config  # type: ignore[import-not-found]
    from pipetools.runtime import RuntimePaths  # type: ignore[import-not-found]
    return ESMFold2BatchFoldRequest, run_esmfold2_batch, ESMFold2Config, RuntimePaths


def _runtime_paths(repo_root: Path, RuntimePaths: Any) -> Any:
    models = asset_root(repo_root) / "community_models"
    return RuntimePaths(
        repo_root=repo_root,
        model_root=models,
        cache_root=cache_dir("partidesign_reward", repo_root),
        mpnn_python=Path(sys.executable),
        proteinmpnn_root=models / "proteinmpnn",
        ligandmpnn_root=models / "ligandmpnn",
        esmfold2_root=models / "esmfold2",
        esmfold2_python=Path(sys.executable),
        esmfold2_common=models / "esmfold2" / "common",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _reported_polymer_sequences(
    candidate: SearchCandidate,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    """Render fixed modified identities, but not atom-inventory CCD overlays."""

    result: dict[str, str] = {}
    atoms = candidate.atoms
    chain_values = np.asarray(atoms.chain_id).astype(str)
    logical_payload = payload.get(SOURCE_MAPPING, {})
    entries = logical_payload.get("logical_sequences", payload.get("sequences", ()))
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("ESMFold2 sequence entry is malformed")
        chain_id = raw_entry.get("chain_id")
        sequence = raw_entry.get("sequence")
        if not isinstance(chain_id, str) or not isinstance(sequence, str):
            continue
        residues = list(sequence)
        rows = np.flatnonzero(chain_values == chain_id)
        token_uids = sorted(
            {int(value) for value in np.asarray(atoms.token_uid[rows], dtype=np.int64)}
        )
        if len(token_uids) != len(residues):
            raise ValueError("ESMFold2 reporting token ledger changed sequence length")
        for position, token_uid in enumerate(token_uids):
            token_rows = np.flatnonzero(
                np.asarray(atoms.token_uid, dtype=np.int64) == token_uid
            )
            residue_uids = {
                int(value)
                for value in np.asarray(
                    atoms.residue_uid[token_rows], dtype=np.int64
                )
            }
            if len(residue_uids) != 1:
                raise ValueError("ESMFold2 reporting token spans multiple residues")
            component_id = candidate.fixed_modified_residue_names.get(
                next(iter(residue_uids))
            )
            if component_id is not None:
                residues[position] = f"[{component_id}]"
        result[chain_id] = "".join(residues)
    return result


def _confidence(
    candidate_id: str,
    design_chains: Sequence[str],
    raw: Mapping[str, Any],
    motif_rmsd_angstrom: float | None = None,
    *, antibody_antigen_only: bool = False,
) -> ESMFold2Confidence:
    per_chain = raw.get("per_chain_plddt")
    per_chain_count = raw.get("per_chain_token_count")
    pair = raw.get("chain_pair_iptm")
    if (
        not isinstance(per_chain, Mapping)
        or not isinstance(per_chain_count, Mapping)
        or not isinstance(pair, Mapping)
    ):
        raise ValueError("ESMFold2 confidence lacks per-chain values")
    design_values = [float(per_chain[chain]) for chain in design_chains]
    design_counts = [int(per_chain_count[chain]) for chain in design_chains]
    if (
        not design_values
        or not np.isfinite(design_values).all()
        or any(count < 1 for count in design_counts)
    ):
        raise ValueError("ESMFold2 design-chain pLDDT is incomplete")
    chains = list(per_chain)
    interfaces: list[float] = []
    design = set(design_chains)
    for left_index, left in enumerate(chains):
        for right in chains[left_index + 1 :]:
            if left not in design and right not in design:
                continue
            if antibody_antigen_only and left in design and right in design:
                continue
            directed = np.asarray(
                [float(pair[left][right]), float(pair[right][left])],
                dtype=np.float64,
            )
            if not np.isfinite(directed).all():
                raise ValueError("ESMFold2 pair ipTM is non-finite")
            # pair_chains_iptm is directional.  An interface in the public
            # reward is unordered, hence average both directions before the
            # equal-weight mean.  This makes scoring invariant to entity order.
            interfaces.append(float(directed.mean()))
    ptm = float(raw.get("ptm", float("nan")))
    if not np.isfinite(ptm) or not 0 <= ptm <= 1:
        raise ValueError("ESMFold2 pTM is missing, non-finite, or outside [0,1]")
    interface_iplddt: float | None = None
    if interfaces:
        interface_iplddt = float(raw.get("complex_iplddt", float("nan")))
        if (
            not np.isfinite(interface_iplddt)
            or not 0 <= interface_iplddt <= 100
        ):
            raise ValueError(
                "ESMFold2 complex ipLDDT is missing, non-finite, or outside [0,100]"
            )
    return ESMFold2Confidence(
        candidate_id=candidate_id,
        design_plddt=float(
            np.average(
                np.asarray(design_values, dtype=np.float64),
                weights=np.asarray(design_counts, dtype=np.float64),
            )
        ),
        interface_iptm=(
            float(np.mean(np.asarray(interfaces, dtype=np.float64)))
            if interfaces
            else None
        ),
        ptm=ptm,
        interface_iplddt=interface_iplddt,
        motif_rmsd_angstrom=motif_rmsd_angstrom,
    )


def _atom_position_keys(atoms: AtomArray) -> dict[int, tuple[str, int, str]]:
    from ..utils.source_mapping import canonical_chain_ids, polymer_residue_groups
    from .esmfold2_mapping import _ligand_groups

    result: dict[int, tuple[str, int, str]] = {}
    for chain in dict.fromkeys(canonical_chain_ids(atoms).tolist()):
        polymer = [group["row_indices"] for group in polymer_residue_groups(atoms, chain)]
        for position, rows in enumerate(polymer + _ligand_groups(atoms, chain), start=1):
            for row in rows:
                result[int(row)] = (chain, position, str(atoms.atom_name[row]))
    return result


def motif_rmsd_angstrom(
    candidate: SearchCandidate,
    folded: AtomArray,
) -> float | None:
    """Return aligned motif RMSD, or ``None`` for an insufficient tip motif."""

    source = candidate.atoms
    source_mask = np.asarray(source.motif_coord_condition, dtype=bool) & np.asarray(
        source.coord_valid, dtype=bool
    )
    source_rows = np.flatnonzero(source_mask)
    if len(source_rows) < 3:
        return None
    source_keys = _atom_position_keys(source)
    folded_keys = _atom_position_keys(folded)
    folded_by_key: dict[tuple[str, int, str], list[int]] = {}
    for row, key in folded_keys.items():
        folded_by_key.setdefault(key, []).append(row)
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for row in source_rows:
        matches = folded_by_key.get(source_keys[int(row)], [])
        if len(matches) != 1:
            return None
        left.append(np.asarray(source.coord[row], dtype=np.float64))
        right.append(np.asarray(folded.coord[matches[0]], dtype=np.float64))
    source_xyz = np.asarray(left, dtype=np.float64)
    folded_xyz = np.asarray(right, dtype=np.float64)
    if (
        np.linalg.matrix_rank(source_xyz - source_xyz.mean(axis=0)) < 2
        or np.linalg.matrix_rank(folded_xyz - folded_xyz.mean(axis=0)) < 2
    ):
        return None
    source_centered = source_xyz - source_xyz.mean(axis=0)
    folded_centered = folded_xyz - folded_xyz.mean(axis=0)
    u, _singular, vt = np.linalg.svd(folded_centered.T @ source_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = folded_centered @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - source_centered) ** 2, axis=1))))


def run_fold_reward_batch(
    candidates: Sequence[SearchCandidate],
    *,
    reward_config: Mapping[str, Any],
    repo_root: Path,
    owner_work_root: Path,
    stage_output_root: Path | None = None,
) -> tuple[tuple[FoldRewardResult, ...], tuple[FoldRewardFailure, ...]]:
    """Run MPNN candidate-locally, then one persistent ESMFold2 batch."""

    if not candidates:
        return (), ()
    if owner_work_root.exists() or owner_work_root.is_symlink():
        raise RuntimeError(f"fold reward work root already exists: {owner_work_root}")
    owner_work_root.mkdir(parents=True)
    Request, run_batch, Config, RuntimePaths = _load_pipetools(repo_root)
    requests: list[Any] = []
    ledgers: dict[
        str,
        tuple[SearchCandidate, Any, list[str], dict[str, str], dict[str, Any]],
    ] = {}
    failures: list[FoldRewardFailure] = []
    sequence_config = reward_config["sequence"]
    planned: list[tuple[SearchCandidate, Any]] = []
    for candidate in candidates:
        try:
            plan = plan_sequence_source(
                candidate,
                requested=str(sequence_config["source"]),
                user_fixed_positions=tuple(sequence_config["fixed_positions"]),
                fixed_interface=sequence_config.get("fixed_interface"),
                reward_method="esmfold2",
            )
            planned.append((candidate, plan))
        except ValueError as exc:
            failures.append(FoldRewardFailure(candidate.candidate_id, str(exc)))
    antibody_plans = {}
    if any(candidate.base_request.task.format == "antibody" for candidate, _plan in planned):
        from .mpnn import build_antibody_mpnn_plan
        accepted = []
        for candidate, plan in planned:
            try:
                if candidate.base_request.task.format == "antibody":
                    antibody_plans[candidate.candidate_id] = build_antibody_mpnn_plan(
                        candidate, sequence_plan=plan,
                    )
                accepted.append((candidate, plan))
            except MPNNSequenceError as exc:
                failures.append(FoldRewardFailure(candidate.candidate_id, str(exc)))
        planned = accepted
    effective_reward_config = reward_config
    if (
        str(sequence_config["source"]) == "auto"
        and any(plan.source == "proteinmpnn" for _candidate, plan in planned)
    ):
        try:
            effective_reward_config = normalize_reward_sequence_config(
                reward_config,
                requested="auto",
                require_soluble=True,
            )
        except ValueError as exc:
            raise MPNNSystemError(str(exc)) from exc
    try:
        sequence_plans, mpnn_failures = generate_mpnn_sequences(
            planned,
            reward_config=effective_reward_config,
            repo_root=repo_root,
            work_dir=owner_work_root / "mpnn-batches",
            **({"antibody_plans": antibody_plans} if antibody_plans else {}),
        )
    except MPNNSystemError:
        raise
    except MPNNSequenceError as exc:
        # Conversion and fixed-mask errors are candidate-declared failures, but
        # the batched model protocol itself is systemic.  Preparation above is
        # candidate-local; reaching this boundary means the whole owner batch
        # cannot be trusted.
        raise MPNNSystemError(f"batched MPNN preparation failed: {exc}") from exc
    failures.extend(
        FoldRewardFailure(candidate_id, message)
        for candidate_id, message in sorted(mpnn_failures.items())
    )
    for candidate_index, (candidate, _initial_plan) in enumerate(planned):
        if candidate.candidate_id in mpnn_failures:
            continue
        safe = f"candidate_{candidate_index}"
        candidate_root = owner_work_root / safe
        try:
            plan = sequence_plans[candidate.candidate_id]
            payload = build_esmfold2_input(candidate, plan)
            payload["sample_id"] = safe
            input_path = candidate_root / "input.json"
            _write_json(input_path, payload)
            if candidate.base_request.task.format == "antibody":
                import torch
                stage = {"candidate": candidate, "plan": plan,
                         "payload": payload, "reward_config": dict(effective_reward_config)}
                torch.save(stage, candidate_root / "finalized_antibody.pt")
                if stage_output_root is not None:
                    import os
                    destination = (stage_output_root / candidate.base_request.task.name / "antibody_stages"
                                   / f"{candidate.base_request.sample_id}_candidate_{candidate.candidate_ordinal}.pt")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(".tmp")
                    torch.save(stage, temporary)
                    os.replace(temporary, destination)
            requests.append(
                Request(
                    request_key=candidate.candidate_id,
                    input_json=input_path,
                    raw_dir=candidate_root / "raw",
                    final_refold_dir=candidate_root / "fold",
                    sample_id=safe,
                    variant_id="sequence_000",
                )
            )
            ledgers[candidate.candidate_id] = (
                candidate,
                plan,
                list(payload["design_chains"]),
                _reported_polymer_sequences(candidate, payload),
                payload,
            )
        except (KeyError, ValueError, OSError) as exc:
            failures.append(FoldRewardFailure(candidate.candidate_id, str(exc)))
    if not requests:
        return (), tuple(failures)

    esm = reward_config["esmfold2"]
    config_kwargs = {
        "mode": str(esm["mode"]),
        "samples": 1,
        "cpus": 4.0,
        "gpus": 1.0,
        "seed": 101,
        "loops": int(esm["loops"]),
        "steps": int(esm["steps"]),
        "lm_dropout": float(esm["lm_dropout"]),
        "lm_mask_pct": (
            None if esm["lm_mask_pct"] is None else float(esm["lm_mask_pct"])
        ),
        "esmc_attention_backend": str(esm["attention_backend"]),
        "context_embedding_cache": esm["context_embedding_cache"],
        "timeout_seconds": int(esm["timeout_seconds"]),
    }
    config_kwargs["optimization"] = dict(esm.get("optimization", {"mode": "auto", "disable": []}))
    config = Config(**config_kwargs)
    batch_id = uuid.uuid4().hex
    runtime = _runtime_paths(repo_root, RuntimePaths)
    outcomes = run_batch(
        runtime,
        requests,
        owner_work_root / "fold_batches" / batch_id,
        config,
    )
    result: list[FoldRewardResult] = []
    weights = reward_config["score"]
    for outcome in outcomes:
        if outcome.error is not None or len(outcome.folds) != 1:
            failures.append(
                FoldRewardFailure(
                    outcome.request_key,
                    outcome.error or "ESMFold2 did not return exactly one fold",
                )
            )
            continue
        fold = outcome.folds[0]
        if fold.status != "ok" or fold.structure_path is None:
            failures.append(
                FoldRewardFailure(outcome.request_key, fold.error or "ESMFold2 fold failed")
            )
            continue
        if outcome.request_key not in ledgers:
            raise RuntimeError(
                f"ESMFold2 worker returned unknown candidate {outcome.request_key!r}"
            )
        _candidate, _plan, design_chains, reported_sequences, payload = ledgers[
            outcome.request_key
        ]
        try:
            restore_source_output(Path(fold.structure_path), payload)
            mapped_confidence = mapped_confidence_summary(fold.confidence)
            folded_atoms = read_structure(
                Path(fold.structure_path),
                declared_polymer_chain_modalities=(
                    candidate_chain_modalities(_candidate)
                ),
            )
            confidence = _confidence(
                outcome.request_key,
                design_chains,
                mapped_confidence,
                motif_rmsd_angstrom(_candidate, folded_atoms),
                antibody_antigen_only=_candidate.base_request.task.format == "antibody",
            )
            score = score_confidences(
                [confidence],
                monomer_plddt_weight=float(weights["monomer_plddt_weight"]),
                complex_plddt_weight=float(weights["complex_plddt_weight"]),
                complex_iptm_weight=float(weights["complex_iptm_weight"]),
            )
        except (KeyError, TypeError, ValueError, IndexError, OSError) as error:
            failures.append(
                FoldRewardFailure(
                    outcome.request_key,
                    "ESMFold2 candidate result is invalid: "
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        result.append(
            FoldRewardResult(
                candidate_id=outcome.request_key,
                score=float(score.total[0].item()),
                confidence=confidence,
                fold_cif=Path(fold.structure_path),
                sequence_source=_plan.source,
                cache=dict(fold.confidence.get("context_embedding_cache", {})),
                design_sequence=(
                    reported_sequences.get(design_chains[0])
                    if len(design_chains) == 1
                    else None
                ),
                metrics={
                    **{
                        "esmfold2_score": float(score.total[0].item()),
                        "esmfold2_plddt": float(confidence.design_plddt),
                        "esmfold2_ptm": float(confidence.ptm),
                    },
                    **(
                        {
                            "esmfold2_iplddt": float(confidence.interface_iplddt),
                            "esmfold2_iptm": float(confidence.interface_iptm),
                        }
                        if confidence.interface_iplddt is not None
                        and confidence.interface_iptm is not None
                        else {}
                    ),
                },
            )
        )
    return tuple(result), tuple(failures)


__all__ = [
    "FoldRewardFailure",
    "FoldRewardResult",
    "run_fold_reward_batch",
    "motif_rmsd_angstrom",
]
