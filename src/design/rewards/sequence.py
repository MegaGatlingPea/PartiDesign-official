"""Reward-aware sequence-source planning.

The ESMFold2 planner remains the source of truth for sequence eligibility and
fixed-position handling.  Reward ``auto`` keeps nucleic designs in their
finalized ``design`` form and maps eligible protein/cyclic routes to the
existing ``proteinmpnn`` backend key, whose caller-selected configuration
identifies the SolubleMPNN variant.  PartiScore additionally rejects an
explicit ``na_mpnn`` request before any backend validation or asset lookup.
"""

from __future__ import annotations

import copy
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from ..core.records import SearchCandidate
from .esmfold2 import (
    SequenceSourcePlan,
    _na_design_is_present,
    plan_sequence_source,
    resolve_sequence_source,
)


_PARTISCORE_SEQUENCE_SOURCES = frozenset(
    {"auto", "design", "proteinmpnn", "ligandmpnn", "cyclicmpnn"}
)
_AUTO_SOLUBLE_BACKEND_KEYS = (
    "solublempnn",
    "soluble_mpnn",
    "solublempnn_defaults",
)


def _auto_soluble_backend(
    sequence: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Return the named SolubleMPNN options carried by a reward config.

    Reward sources intentionally keep their public namespace small.  The
    selected backend therefore lives under an internal metadata key until it
    is copied to the native proteinmpnn slot.  solublempnn_defaults is accepted
    for the shared sequence policy file; a caller-specific solublempnn block
    takes precedence over those stock defaults.
    """

    for key in _AUTO_SOLUBLE_BACKEND_KEYS:
        value = sequence.get(key)
        if isinstance(value, Mapping):
            return value, key
    auto = sequence.get("auto")
    if isinstance(auto, Mapping):
        for key in ("protein", "cyclic"):
            value = auto.get(key)
            if isinstance(value, Mapping):
                return value, f"auto.{key}"
    return None, None


def _looks_like_stock_vanilla_backend(backend: Mapping[str, Any]) -> bool:
    """Recognize only the shipped vanilla checkpoint/temperature pair.

    This check is deliberately conservative.  It is used to identify an
    inherited stock block when a separate SolubleMPNN metadata block is
    present; custom checkpoint or temperature values must not be silently
    replaced by the automatic route.
    """

    checkpoint = str(backend.get("checkpoint", "")).replace("\\", "/")
    checkpoint_name = PurePosixPath(checkpoint).name
    try:
        temperature = float(backend.get("temperature", float("nan")))
    except (TypeError, ValueError):
        temperature = float("nan")
    return (
        str(backend.get("model_type", "")) == "protein_mpnn"
        and checkpoint_name == "v_48_020.pt"
        and "/vanilla/" in f"/{checkpoint.strip('/')}/"
        and temperature == 0.0001
    )


def normalize_reward_sequence_config(
    reward_config: Mapping[str, Any],
    *,
    requested: str,
    require_soluble: bool = False,
) -> Mapping[str, Any]:
    """Resolve the internal SolubleMPNN backend for reward auto.

    proteinmpnn remains the native generation key, so this function only
    changes the effective backend options.  Explicit proteinmpnn requests
    return the original mapping untouched.  A separate solublempnn block
    (or shared solublempnn_defaults block) is required before an automatic
    protein/cyclic route can safely replace the vanilla block.  When callers
    provide a non-stock vanilla checkpoint or temperature while relying only
    on stock metadata, the choice is ambiguous and an actionable error is
    raised instead of overwriting the user value.

    The returned mapping is a deep copy only when a Soluble backend is
    selected; callers can therefore pass the original config through for
    explicit sources without changing ownership or cache behavior.
    """

    if requested != "auto":
        return reward_config
    sequence = reward_config.get("sequence")
    if not isinstance(sequence, Mapping):
        if require_soluble:
            raise ValueError("reward.sequence is required for automatic MPNN routing")
        return reward_config
    automatic = sequence.get("auto_proteinmpnn")
    if isinstance(automatic, Mapping):
        effective = copy.deepcopy(dict(reward_config))
        effective["sequence"]["proteinmpnn"] = copy.deepcopy(dict(automatic))
        return effective
    protein = sequence.get("proteinmpnn")
    if not isinstance(protein, Mapping):
        if require_soluble:
            raise ValueError(
                "reward.sequence.source=auto requires a proteinmpnn backend block"
            )
        return reward_config
    # A caller may already have supplied the Soluble contract in the native
    # slot.  Preserve it and, crucially, do not reinterpret its checkpoint or
    # temperature as vanilla options.
    if (
        str(protein.get("model_type", "")) == "soluble_mpnn"
        and bool(protein.get("use_soluble_model", True))
    ):
        return reward_config
    soluble, metadata_key = _auto_soluble_backend(sequence)
    if soluble is None:
        if require_soluble:
            raise ValueError(
                "reward.sequence.source=auto selected a protein/cyclic route, "
                "but no SolubleMPNN options were provided; add "
                "reward.sequence.solublempnn (or solublempnn_defaults)"
            )
        return reward_config
    model_type = str(soluble.get("model_type", ""))
    if model_type != "soluble_mpnn" or soluble.get("use_soluble_model") is False:
        raise ValueError(
            f"automatic SolubleMPNN metadata {metadata_key!r} must use "
            "model_type=soluble_mpnn and use_soluble_model=true"
        )
    if (
        metadata_key == "solublempnn_defaults"
        and not _looks_like_stock_vanilla_backend(protein)
    ):
        raise ValueError(
            "reward.sequence.source=auto has explicit proteinmpnn checkpoint or "
            "temperature values but only stock SolubleMPNN metadata is available; "
            "put the intended values in reward.sequence.solublempnn"
        )
    effective_sequence = copy.deepcopy(dict(sequence))
    effective_sequence["proteinmpnn"] = copy.deepcopy(dict(soluble))
    effective_config = copy.deepcopy(dict(reward_config))
    effective_config["sequence"] = effective_sequence
    return effective_config


def _resolve_auto_reward_source(candidate: SearchCandidate) -> str:
    """Resolve reward ``auto`` without routing nucleic designs to NA-MPNN.

    The public reward source remains ``auto``.  Existing generation and
    packing code use ``proteinmpnn`` as the native key for both ProteinMPNN
    and SolubleMPNN, so the selected Soluble backend is configured by the
    caller after this modality-aware route is chosen.
    """

    if _na_design_is_present(candidate.atoms):
        return "design"
    source = resolve_sequence_source(candidate, "auto")
    if source in {"proteinmpnn", "cyclicmpnn"}:
        return "proteinmpnn"
    return source


def resolve_reward_sequence_source(
    candidate: SearchCandidate,
    requested: str,
    *,
    reward_method: str = "partiscore",
) -> str:
    """Resolve a reward's requested sequence source for one candidate.

    Explicit ESMFold2 sources delegate to the existing resolver unchanged;
    only ``auto`` is normalized to keep nucleic designs in ``design`` and to
    route eligible protein/cyclic designs through the native protein backend
    key.  PartiScore shares the eligible protein routes but has no NA-MPNN
    backend, so an explicit ``na_mpnn`` request fails before asset lookup.
    """

    if reward_method == "esmfold2":
        if requested == "auto":
            return _resolve_auto_reward_source(candidate)
        return resolve_sequence_source(candidate, requested)
    if reward_method != "partiscore":
        raise ValueError(f"unsupported reward method {reward_method!r}")
    if not isinstance(requested, str) or requested == "na_mpnn":
        raise ValueError("PartiScore reward does not support na_mpnn")
    if requested not in _PARTISCORE_SEQUENCE_SOURCES:
        allowed = ", ".join(sorted(_PARTISCORE_SEQUENCE_SOURCES))
        raise ValueError(
            f"unsupported PartiScore sequence source {requested!r}; expected {allowed}"
        )
    if requested == "auto":
        return _resolve_auto_reward_source(candidate)
    return resolve_sequence_source(candidate, requested)


def plan_reward_sequence_source(
    candidate: SearchCandidate,
    *,
    requested: str,
    user_fixed_positions: Sequence[str] = (),
    fixed_interface: Mapping[str, Any] | None = None,
    reward_method: str = "partiscore",
) -> SequenceSourcePlan:
    """Build a reward-aware sequence plan compatible with ``SequenceSourcePlan``."""

    if reward_method == "esmfold2":
        source = _resolve_auto_reward_source(candidate) if requested == "auto" else requested
        return plan_sequence_source(
            candidate,
            requested=source,
            user_fixed_positions=user_fixed_positions,
            fixed_interface=fixed_interface,
        )
    source = resolve_reward_sequence_source(
        candidate,
        requested,
        reward_method=reward_method,
    )
    return plan_sequence_source(
        candidate,
        requested=source,
        user_fixed_positions=user_fixed_positions,
        fixed_interface=fixed_interface,
    )


__all__ = [
    "normalize_reward_sequence_config",
    "plan_reward_sequence_source",
    "resolve_reward_sequence_source",
]
