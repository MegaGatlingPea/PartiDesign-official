"""Validation-owned MPNN adapters.

The reward adapter historically fixed its native worker seed for ranking
reproducibility. Validation passes one explicit command seed per native batch;
the singleton ProteinMPNN compatibility adapter below reuses the canonical
constraint helpers while substituting that seed in its command.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from partidesign.paths import community_source
from typing import Any, Mapping, Sequence


def generate_protein_mpnn_sequences(
    items: Sequence[tuple[Any, Any]],
    *,
    reward_config: Mapping[str, Any],
    repo_root: Path,
    work_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run the pinned ProteinMPNN worker with a validation-owned seed.

    ``items`` follows ``generate_mpnn_sequences`` and normally contains one
    candidate.  Keeping the batch shape allows the existing private MPNN
    canonical sample/constraint logic to remain authoritative while this
    adapter changes only the native command's seed argument.
    """
    from src.design.rewards import mpnn as native

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise native.MPNNSequenceError("validation ProteinMPNN seed must be a non-negative integer")
    if not items:
        return {}, {}
    backend = reward_config["sequence"]["proteinmpnn"]
    native._backend_assets(backend, "proteinmpnn")
    _family, ca_only, _use_soluble = native._protein_model_contract(backend)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=False)
    api, _sequence_config, model_types, _runtime_paths, _source = native._load_pipetools(repo_root)
    prepare_mpnn_input = api[0]
    parsed_rows: list[dict[str, Any]] = []
    chain_ids: dict[str, Any] = {}
    fixed_positions: dict[str, Any] = {}
    pssm_values: dict[str, Any] = {}
    tied_values: dict[str, Any] = {}
    omit_values: dict[str, Any] = {}
    plan_by_name: dict[str, Any] = {}
    safe_by_candidate: dict[str, str] = {}
    failures: dict[str, str] = {}
    parser = community_source("proteinmpnn", repo_root) / "protein_mpnn_utils.py"
    if str(parser.parent) not in sys.path:
        sys.path.insert(0, str(parser.parent))
    from protein_mpnn_utils import parse_PDB  # type: ignore[import-not-found]

    for candidate_index, (candidate, plan) in enumerate(items):
        try:
            safe = f"candidate_{candidate_index}"
            candidate_root = work_dir / "prepared" / safe
            candidate_root.mkdir(parents=True)
            sample = native._canonical_sample(candidate, plan, candidate_root, model_types)
            prepared = prepare_mpnn_input(
                sample,
                candidate_root / "native-input",
                ca_only=ca_only,
                materialize=True,
            )
            parsed = parse_PDB(str(prepared.pdb_path), ca_only=ca_only)
            if not isinstance(parsed, list) or len(parsed) != 1:
                raise native.MPNNSystemError("ProteinMPNN parser did not return one structure")
            parsed[0]["name"] = safe
            if plan.design_chain is None:
                raise native.MPNNSequenceError("ProteinMPNN plan lacks its design chain")
            design_alias = prepared.alias_by_chain[plan.design_chain]
            all_aliases = sorted(
                prepared.alias_by_chain[record.chain_id]
                for record in sample.chains
                if record.modality == "protein"
            )
            fixed, pssm, tied = native._proteinmpnn_constraint_payloads(
                sample, prepared, plan, backend
            )
            omit = native._protein_omit_payload(
                candidate,
                plan,
                name=safe,
                design_alias=design_alias,
                backend=backend,
                chain_aliases=all_aliases,
            )
        except native.MPNNSequenceError as exc:
            failures[candidate.candidate_id] = str(exc)
            continue
        parsed_rows.append(parsed[0])
        chain_ids[safe] = (
            [design_alias],
            [value for value in all_aliases if value != design_alias],
        )
        fixed_positions[safe] = fixed[prepared.pdb_path.stem]
        if pssm is not None:
            pssm_values[safe] = pssm[prepared.pdb_path.stem]
        if tied is not None:
            tied_values[safe] = tied[prepared.pdb_path.stem]
        if omit is not None:
            omit_values.update(omit)
        plan_by_name[safe] = plan
        safe_by_candidate[candidate.candidate_id] = safe

    if not parsed_rows:
        return {}, failures

    jsonl_path = work_dir / "parsed.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=True) + "\n" for row in parsed_rows),
        encoding="utf-8",
    )

    def jsonl(name: str, payload: Any) -> Path:
        path = work_dir / name
        path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return path

    chain_path = jsonl("chain_id.jsonl", chain_ids)
    fixed_path = jsonl("fixed_positions.jsonl", fixed_positions)
    output = work_dir / "output"
    bias = backend.get("bias_aa") or {}
    command = native._proteinmpnn_command(
        repo_root=repo_root,
        backend=backend,
        jsonl_path=jsonl_path,
        chain_path=chain_path,
        fixed_path=fixed_path,
        output=output,
        bias_path=jsonl("bias_aa.jsonl", bias) if bias else None,
        pssm_path=jsonl("pssm.jsonl", pssm_values) if pssm_values else None,
        tied_path=jsonl("tied_positions.jsonl", tied_values) if tied_values else None,
        omit_path=jsonl("omit_AA.jsonl", omit_values) if omit_values else None,
    )
    seed_positions = [index for index, value in enumerate(command) if value == "--seed"]
    if len(seed_positions) != 1:
        raise native.MPNNSystemError("ProteinMPNN command has an unexpected seed contract")
    command[seed_positions[0] + 1] = str(seed)
    from src.design.rewards.mpnn_cache import MPNNCacheError, run_mpnn_cached

    try:
        run_mpnn_cached(
            command,
            cwd=community_source("proteinmpnn", repo_root),
            env=native._offline_environment(repo_root),
            log_path=work_dir / "process.log",
            timeout=7200,
            backend="proteinmpnn",
        )
    except (MPNNCacheError, OSError) as exc:
        raise native.MPNNSystemError(
            f"validation proteinmpnn worker failed: {type(exc).__name__}: {exc}"
        ) from exc
    result: dict[str, Any] = {}
    for candidate, _plan in items:
        if candidate.candidate_id in failures:
            continue
        safe = safe_by_candidate[candidate.candidate_id]
        try:
            sequence = native._read_proteinmpnn_design_sequence(
                output / "seqs" / f"{safe}.fa"
            )
            result[candidate.candidate_id] = native._validate_sequence(plan_by_name[safe], sequence)
        except native.MPNNSequenceError as exc:
            failures[candidate.candidate_id] = str(exc)
    return result, failures


def generate_native_mpnn_sequences(
    items: Sequence[tuple[Any, Any]],
    *,
    reward_config: Mapping[str, Any],
    repo_root: Path,
    work_dir: Path,
    seed: int,
    antibody_plans: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Call the shared native MPNN worker for one validation batch.

    Validation assigns one command seed to the batch and keeps each returned
    slot's source index separately. The reward worker keeps its historical
    default when called without this keyword, while this adapter requires the
    native worker to accept an explicit batch seed so provenance matches the
    command that actually ran.
    """
    from src.design.rewards import mpnn as native

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise native.MPNNSequenceError(
            "validation MPNN seed must be a non-negative integer"
        )
    try:
        return native.generate_mpnn_sequences(
            items,
            reward_config=reward_config,
            repo_root=repo_root,
            work_dir=work_dir,
            seed=seed,
            **({"antibody_plans": antibody_plans} if antibody_plans else {}),
        )
    except TypeError as exc:
        # A pre-change reward checkout silently cannot honour validation's
        # per-slot seed. Make that incompatibility actionable instead of
        # emitting a sequence with a false seed in its JSON provenance.
        if "seed" in str(exc) and "keyword" in str(exc):
            raise native.MPNNSystemError(
                "native MPNN worker lacks the validation seed override API"
            ) from exc
        raise


__all__ = ["generate_native_mpnn_sequences", "generate_protein_mpnn_sequences"]
