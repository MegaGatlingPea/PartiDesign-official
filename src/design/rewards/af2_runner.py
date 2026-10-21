"""Owner-local offline MPNN + ColabDesign AF2 reward execution."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from partidesign.paths import (community_source, community_asset, asset_root, cache_dir, offline_site, code_root)
from typing import Any, Mapping, Sequence

import numpy as np
from biotite import structure as struc

from ..core.records import SearchCandidate
from ..utils.reader import read_structure
from ..utils.writer import write_structure_atomic
from .af2 import AF2Confidence, af2_design_chain, score_af2_confidences
from .af2_mapping import (
    af2_source_ledger,
    af2_preparation_ledger,
    private_chain_id,
    restore_af2_residue_identity,
    write_private_af2_pdb,
)
from .esmfold2 import (
    has_cyclic_offset,
    has_required_head_to_tail_single_bond,
)
from .sequence import plan_reward_sequence_source, normalize_reward_sequence_config
from .esmfold2_input import _explicit_covalent_edges
from .esmfold2_runner import FoldRewardFailure, FoldRewardResult
from .mpnn_cache import MPNNCacheError, run_cached
from .mpnn import (
    MPNNSequenceError,
    MPNNSystemError,
    _offline_environment as _community_offline_environment,
    _release_pythonpath,
    generate_mpnn_sequences,
)


_ORDER_TO_BIOTITE = {
    "?": int(struc.BondType.ANY),
    "single": int(struc.BondType.SINGLE),
    "double": int(struc.BondType.DOUBLE),
    "triple": int(struc.BondType.TRIPLE),
    "aromatic": int(struc.BondType.AROMATIC),
}

def plan_sequence_source(*args: Any, **kwargs: Any) -> Any:
    """Compatibility seam forwarding runner planning to the reward helper."""
    return plan_reward_sequence_source(*args, **kwargs)


def _offline_environment(repo_root: Path) -> dict[str, str]:
    """Return the shared offline worker environment with vendored AF2 first."""

    environment = _community_offline_environment(repo_root)
    source = community_source("colabdesign")
    package = source / "colabdesign" / "__init__.py"
    if (
        not source.is_dir()
        or source.is_symlink()
        or not package.is_file()
        or package.is_symlink()
    ):
        raise MPNNSystemError("repo-owned ColabDesign source is missing or unsafe")
    environment["PYTHONPATH"] = _release_pythonpath(source)
    return environment


def _position_keys(atoms: Any) -> dict[tuple[str, int, str], int]:
    next_position: dict[str, int] = {}
    residue_position: dict[tuple[str, int, str, str], int] = {}
    result: dict[tuple[str, int, str], int] = {}
    for row in range(len(atoms)):
        chain = str(atoms.chain_id[row])
        residue = (
            chain,
            int(atoms.res_id[row]),
            str(atoms.ins_code[row]),
            str(atoms.res_name[row]),
        )
        if residue not in residue_position:
            next_position[chain] = next_position.get(chain, 0) + 1
            residue_position[residue] = next_position[chain]
        key = (chain, residue_position[residue], str(atoms.atom_name[row]))
        if key in result:
            raise ValueError(f"duplicate AF2 atom key {key}")
        result[key] = row
    return result


def _write_private_af2_pdb(
    atoms: Any, output_path: Path, *, design_chain: str | None = None,
    chain_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Prepare native occurrence IDs and retain the reversible source ledger."""

    if design_chain is None:
        design = list(dict.fromkeys(
            str(chain) for chain, selected in zip(atoms.chain_id, atoms.is_design)
            if bool(selected)
        ))
        if len(design) != 1:
            raise ValueError("AF2 private template requires one design chain")
        design_chain = design[0]
    if chain_order is None:
        chain_order = _ordered_chain_ids(atoms)
    ledger = af2_source_ledger(atoms, chain_order, design_chain)
    write_private_af2_pdb(atoms, output_path, ledger=ledger)
    return ledger


def _ordered_chain_ids(atoms: Any) -> list[str]:
    result: list[str] = []
    for value in np.asarray(atoms.chain_id).astype(str):
        if value not in result:
            result.append(value)
    return result


def _af2_cyclic_offset(candidate: SearchCandidate, design_chain: str) -> bool:
    """Derive AF2 offset state from task intent and authoritative topology."""

    return (
        has_required_head_to_tail_single_bond(candidate, design_chain)
        or has_cyclic_offset(candidate, design_chain)
    )


def _restore_af2_chain_identity(
    candidate: SearchCandidate,
    folded: Any,
    expected_chain_ids: Sequence[str],
    *, source_ledger: Mapping[str, Any] | None = None,
) -> Any:
    """Restore exact source residue occurrences after native serialization."""

    expected = [str(value) for value in expected_chain_ids]
    if source_ledger is None:
        design = list(dict.fromkeys(
            str(chain) for chain, selected in zip(candidate.atoms.chain_id, candidate.atoms.is_design)
            if bool(selected)
        ))
        if len(design) != 1:
            raise ValueError("AF2 source identity requires one design chain")
        source_ledger = af2_source_ledger(candidate.atoms, expected, design[0])
    return restore_af2_residue_identity(folded, source_ledger, expected)


def _overlay_special_topology(candidate: SearchCandidate, folded: Any) -> Any:
    edges = _explicit_covalent_edges(candidate, include_topology_conditions=False)
    if not edges:
        return folded
    source = candidate.atoms
    source_keys = _position_keys(source)
    source_key_by_row = {row: key for key, row in source_keys.items()}
    folded_keys = _position_keys(folded)
    uid_to_row = {int(uid): row for row, uid in enumerate(source.atom_uid)}
    existing = [] if folded.bonds is None else [tuple(map(int, row)) for row in folded.bonds.as_array()]
    by_pair = {
        tuple(sorted((left, right))): order for left, right, order in existing
    }
    # AF2 never consumes atom-level topology conditions.  Restore only the
    # authoritative non-template edges already carried by the candidate's
    # BondList; a condition-only edge must not become output topology here.
    for uid_a, uid_b, order_name in edges:
        source_a, source_b = uid_to_row[uid_a], uid_to_row[uid_b]
        source_key_a = source_key_by_row[source_a]
        source_key_b = source_key_by_row[source_b]
        if source_key_a not in folded_keys or source_key_b not in folded_keys:
            raise ValueError("AF2 fold omitted an authoritative covalent endpoint")
        pair = tuple(sorted((folded_keys[source_key_a], folded_keys[source_key_b])))
        order = _ORDER_TO_BIOTITE[order_name]
        previous = by_pair.get(pair)
        if previous is not None and order == int(struc.BondType.ANY):
            # An existence-only source fact must not weaken a concrete edge
            # predicted by the fold/parser.  Concrete source facts below are
            # authoritative and intentionally overwrite any folded order.
            continue
        by_pair[pair] = order
    folded.bonds = struc.BondList(
        len(folded),
        np.asarray(
            [(left, right, order) for (left, right), order in sorted(by_pair.items())],
            dtype=np.uint32,
        ),
    )
    return folded


def run_af2_reward_batch(
    candidates: Sequence[SearchCandidate],
    *,
    reward_config: Mapping[str, Any],
    repo_root: Path,
    owner_work_root: Path,
) -> tuple[tuple[FoldRewardResult, ...], tuple[FoldRewardFailure, ...]]:
    if not candidates:
        return (), ()
    if owner_work_root.exists() or owner_work_root.is_symlink():
        raise RuntimeError(f"AF2 work root already exists: {owner_work_root}")
    owner_work_root.mkdir(parents=True)
    sequence_config = reward_config["sequence"]
    af2_config = reward_config["af2"]
    model = str(af2_config["model"])
    requests: list[dict[str, Any]] = []
    ledgers: dict[str, tuple[SearchCandidate, Any, tuple[str, ...], Mapping[str, Any]]] = {}
    failures: list[FoldRewardFailure] = []
    planned: list[tuple[SearchCandidate, Any]] = []
    design_chain_by_id: dict[str, str] = {}
    for candidate in candidates:
        try:
            design_chain = af2_design_chain(candidate, model)
            plan = plan_sequence_source(
                candidate,
                requested=str(sequence_config["source"]),
                user_fixed_positions=tuple(sequence_config["fixed_positions"]),
                fixed_interface=sequence_config.get("fixed_interface"),
                reward_method="esmfold2",
            )
            planned.append((candidate, plan))
            design_chain_by_id[candidate.candidate_id] = design_chain
        except (MPNNSequenceError, ValueError, OSError) as error:
            failures.append(FoldRewardFailure(candidate.candidate_id, str(error)))
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
        )
    except MPNNSystemError:
        raise
    except MPNNSequenceError as error:
        raise MPNNSystemError(f"batched MPNN preparation failed: {error}") from error
    failures.extend(
        FoldRewardFailure(candidate_id, message)
        for candidate_id, message in sorted(mpnn_failures.items())
    )
    for candidate, _initial_plan in planned:
        if candidate.candidate_id in mpnn_failures:
            continue
        root = owner_work_root / ("candidate_" + candidate.candidate_id.replace("/", "_"))
        try:
            plan = sequence_plans[candidate.candidate_id]
            design_chain = design_chain_by_id[candidate.candidate_id]
            if plan.design_chain != design_chain:
                raise ValueError("AF2 and MPNN design-chain identity differs")
            input_pdb = root / "input.pdb"
            context_chains = sorted(
                {
                    str(chain)
                    for chain, is_design in zip(
                        np.asarray(candidate.atoms.chain_id).astype(str),
                        np.asarray(candidate.atoms.is_design, dtype=bool),
                    )
                    if not bool(is_design)
                }
            )
            source_ledger = _write_private_af2_pdb(
                candidate.atoms, input_pdb, design_chain=design_chain,
                chain_order=[*context_chains, design_chain],
            )
            requests.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "input_pdb": str(input_pdb),
                    "output_pdb": str(root / "fold.pdb"),
                    "design_chain": private_chain_id(source_ledger, design_chain),
                    "context_chains": [private_chain_id(source_ledger, chain) for chain in context_chains],
                    "source_design_chain": design_chain,
                    "source_context_chains": context_chains,
                    "source_ledger": af2_preparation_ledger(source_ledger),
                    "design_sequence": plan.chain_sequences[design_chain],
                    "cyclic_offset": _af2_cyclic_offset(candidate, design_chain),
                }
            )
            ledgers[candidate.candidate_id] = (
                candidate,
                plan,
                tuple([*context_chains, design_chain]),
                source_ledger,
            )
        except (KeyError, ValueError, OSError) as error:
            failures.append(FoldRewardFailure(candidate.candidate_id, str(error)))
    if not requests:
        return (), tuple(failures)

    request_path = owner_work_root / "af2-request.json"
    result_path = owner_work_root / "af2-result.json"
    payload = {
        "schema_version": 1,
        "config": {
            **dict(af2_config),
            "params_dir": str(
                community_asset("colabdesign", repo_root) / "checkpoint" / "af2"
            ),
        },
        "requests": requests,
    }
    request_path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    command = [
        sys.executable,
        "-m",
        "src.design.rewards.af2_worker",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    try:
        run_cached(
            command,
            backend="af2",
            worker=Path(__file__).with_name("af2_worker.py"),
            cwd=repo_root,
            env=_offline_environment(repo_root),
            log_path=owner_work_root / "af2-worker.log",
            timeout=int(af2_config["timeout_seconds"]),
        )
    except (MPNNCacheError, OSError) as error:
        raise MPNNSystemError(f"AF2 worker failed: {error}") from error
    raw = json.loads(result_path.read_text())
    if not isinstance(raw, Mapping) or not isinstance(raw.get("results"), list):
        raise RuntimeError("AF2 worker returned an invalid protocol payload")
    results: list[FoldRewardResult] = []
    weights = reward_config["score"]
    for item in raw["results"]:
        candidate_id = str(item.get("candidate_id"))
        if candidate_id not in ledgers:
            raise RuntimeError(
                f"AF2 worker returned unknown candidate {candidate_id!r}"
            )
        if item.get("status") != "ok":
            failures.append(FoldRewardFailure(candidate_id, str(item.get("error"))))
            continue
        candidate, plan, expected_chain_ids, source_ledger = ledgers[candidate_id]
        try:
            confidence = AF2Confidence(
                candidate_id=candidate_id,
                model=model,
                design_plddt=float(item["design_plddt"]),
                ptm=float(item["ptm"]),
                pae=float(item["pae"]),
                interface_iptm=(
                    None if model == "ptm" else float(item["interface_iptm"])
                ),
                interface_ipae=(
                    None if model == "ptm" else float(item["interface_ipae"])
                ),
            )
            score = score_af2_confidences(
                [confidence],
                monomer_plddt_weight=float(weights["monomer_plddt_weight"]),
                complex_plddt_weight=float(weights["complex_plddt_weight"]),
                complex_iptm_weight=float(weights["complex_iptm_weight"]),
            )
            folded = _restore_af2_chain_identity(
                candidate,
                read_structure(Path(str(item["fold_pdb"]))),
                expected_chain_ids,
                source_ledger=source_ledger,
            )
            folded = _overlay_special_topology(candidate, folded)
            fold_cif = Path(str(item["fold_pdb"])).with_suffix(".cif")
            write_structure_atomic(folded, fold_cif, entry_id=candidate_id)
        except (KeyError, TypeError, ValueError, IndexError, OSError) as error:
            failures.append(
                FoldRewardFailure(
                    candidate_id,
                    f"AF2 candidate result is invalid: {type(error).__name__}: {error}",
                )
            )
            continue
        results.append(
            FoldRewardResult(
                candidate_id=candidate_id,
                score=float(score.total[0]),
                confidence=confidence,  # type: ignore[arg-type]
                fold_cif=fold_cif,
                sequence_source=plan.source,
                cache={
                    "model_cache_hits": int(raw.get("model_cache_hits", 0)),
                    "model_cache_misses": int(raw.get("model_cache_misses", 0)),
                },
                design_sequence=plan.chain_sequences[plan.design_chain],
                metrics=(
                    {
                        "af2ptm_score": float(score.total[0]),
                        "af2ptm_plddt": confidence.design_plddt,
                        "af2ptm_ptm": confidence.ptm,
                        "af2ptm_pae": confidence.pae,
                    }
                    if model == "ptm"
                    else {
                        "af2multimer_score": float(score.total[0]),
                        "af2multimer_plddt": confidence.design_plddt,
                        "af2multimer_ptm": confidence.ptm,
                        "af2multimer_iptm": float(confidence.interface_iptm),
                        "af2multimer_ipae": float(confidence.interface_ipae),
                    }
                ),
            )
        )
    return tuple(results), tuple(failures)


__all__ = ["run_af2_reward_batch"]
