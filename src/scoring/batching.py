"""Bounded canonical-pose padding and deterministic PartiScore batch planning."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import torch

_DEFAULT_INFERENCE_BATCH_SIZE = 8
_DEFAULT_MAX_PAIR_ELEMENTS = _DEFAULT_INFERENCE_BATCH_SIZE * 512 * 512
_PARTISCORE_PAIR_FIELDS = frozenset((
    "token_bond_type",
    "token_bond_pair_mask",
    "_partiscore_covalent_bond_pair_mask",
))


def _config_value(config: Any, path: Sequence[str], default: Any) -> Any:
    """Read a possibly absent nested config value without requiring OmegaConf."""

    current = config
    for name in path:
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(name)
        else:
            current = getattr(current, name, None)
    return default if current is None else current


def _inference_score_limits(
    module: Any,
    *,
    inference_batch_size: int | None = None,
    max_pair_elements: int | None = None,
) -> tuple[int, int]:
    """Return the bounded batch size and pair-element budget for pose scoring."""

    config = getattr(module, "config", None)
    if inference_batch_size is None:
        inference_batch_size = _config_value(
            config,
            ("scoring", "inference_batch_size"),
            _config_value(config, ("inference_batch_size",), _DEFAULT_INFERENCE_BATCH_SIZE),
        )
    if max_pair_elements is None:
        max_pair_elements = _config_value(
            config,
            ("scoring", "max_pair_elements"),
            _config_value(config, ("max_pair_elements",), _DEFAULT_MAX_PAIR_ELEMENTS),
        )
    if (
        isinstance(inference_batch_size, bool)
        or isinstance(max_pair_elements, bool)
        or not isinstance(inference_batch_size, int)
        or not isinstance(max_pair_elements, int)
    ):
        raise ValueError("scoring batch limits must be positive integers")
    batch_size = int(inference_batch_size)
    max_pair_elements = int(max_pair_elements)
    if batch_size <= 0 or max_pair_elements <= 0:
        raise ValueError("scoring batch limits must be positive integers")
    return batch_size, max_pair_elements


def _single_pose_token_count(batch: Mapping[str, Any]) -> int:
    """Return a single pose's token count, accepting ``[N]`` or ``[1, N]``."""

    token_mask = batch.get("token_mask")
    if not torch.is_tensor(token_mask):
        raise ValueError("PartiScore pose requires tensor token_mask")
    if token_mask.ndim == 1:
        return int(token_mask.shape[0])
    if token_mask.ndim == 2 and token_mask.shape[0] == 1:
        return int(token_mask.shape[1])
    raise ValueError(
        "score_poses expects single-pose token_mask with shape [N] or [1, N], "
        f"got {tuple(token_mask.shape)}"
    )


def _single_pose_field(
    value: Any,
    key: str,
    token_count: int,
) -> torch.Tensor:
    """Normalize one required tensor field to its unbatched single-pose form."""

    if not torch.is_tensor(value):
        raise ValueError(f"PartiScore pose field {key!r} must be a tensor")
    result = value
    if key in _PARTISCORE_PAIR_FIELDS:
        if result.ndim == 3 and result.shape[0] == 1:
            if result.shape[1] != token_count or result.shape[2] != token_count:
                raise ValueError(
                    f"PartiScore pose field {key!r} has incompatible shape "
                    f"{tuple(result.shape)} for token count {token_count}"
                )
            result = result[0]
        if result.ndim < 2 or tuple(result.shape[:2]) != (token_count, token_count):
            raise ValueError(
                f"PartiScore pose field {key!r} has incompatible shape "
                f"{tuple(result.shape)} for token count {token_count}"
            )
    else:
        if result.ndim >= 2 and result.shape[0] == 1 and result.shape[1] == token_count:
            result = result[0]
        if result.ndim < 1 or result.shape[0] != token_count:
            raise ValueError(
                f"PartiScore pose field {key!r} has incompatible shape "
                f"{tuple(result.shape)} for token count {token_count}"
            )
    return result


def _pad_single_pose_field(value: torch.Tensor, key: str, token_count: int, nmax: int) -> torch.Tensor:
    """Pad a required single-pose field without truncating any real token."""

    if key in _PARTISCORE_PAIR_FIELDS:
        if tuple(value.shape[:2]) != (token_count, token_count):
            raise ValueError(f"invalid pair field shape for {key!r}: {tuple(value.shape)}")
        fill = False if value.dtype == torch.bool else 0
        result = value.new_full((nmax, nmax, *value.shape[2:]), fill)
        result[:token_count, :token_count] = value
        return result
    if value.shape[0] != token_count:
        raise ValueError(f"invalid token field shape for {key!r}: {tuple(value.shape)}")
    if key == "source_mol_id":
        fill = -1
    elif value.dtype == torch.bool:
        fill = False
    else:
        fill = 0
    result = value.new_full((nmax, *value.shape[1:]), fill)
    result[:token_count] = value
    return result


def _batch_canonical_poses(
    poses: Sequence[Mapping[str, Any]],
    identities: Sequence[str],
    *,
    fields: Sequence[str],
    device: torch.device | str,
) -> dict[str, Any]:
    """Pad prepared single-pose canonical inputs to one scorer batch.

    Only explicitly requested fields are carried into the score path.
    This keeps metadata/cache paths out of the model batch and gives all
    padding a single, auditable policy.  In particular, padded source molecule
    IDs are ``-1`` and can never create an inter-sample edge.
    """

    if len(poses) != len(identities):
        raise ValueError("pose and identity counts must match")
    if not poses:
        return {"partiscore_id": []}
    fields = tuple(fields)
    lengths = [_single_pose_token_count(pose) for pose in poses]
    nmax = max(lengths)
    result: dict[str, Any] = {"partiscore_id": list(identities)}
    for key in fields:
        values: list[torch.Tensor] = []
        for pose, token_count in zip(poses, lengths):
            if key not in pose:
                raise ValueError(f"PartiScore pose is missing required field {key!r}")
            value = _single_pose_field(pose[key], key, token_count)
            values.append(value.to(device))
        result[key] = torch.stack(
            [
                _pad_single_pose_field(value, key, token_count, nmax)
                for value, token_count in zip(values, lengths)
            ],
            dim=0,
        )
    return result


def _batch_single_pose_inputs(module, poses, identities):
    """Compatibility wrapper for the training module cache-field policy."""
    return _batch_canonical_poses(
        poses, identities, fields=module._cache_fields(), device=module.device
    )


def _plan_score_batches(
    module: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    inference_batch_size: int | None = None,
    max_pair_elements: int | None = None,
) -> list[list[Mapping[str, Any]]]:
    """Plan deterministic same-target batches under ``B * Nmax**2`` budget."""

    batch_limit, pair_budget = _inference_score_limits(
        module,
        inference_batch_size=inference_batch_size,
        max_pair_elements=max_pair_elements,
    )
    by_target: dict[Any, list[Mapping[str, Any]]] = {}
    target_order: list[Any] = []
    for record in records:
        target = record.get("target_id")
        if target not in by_target:
            by_target[target] = []
            target_order.append(target)
        by_target[target].append(record)
    planned: list[list[Mapping[str, Any]]] = []
    for target in target_order:
        group = by_target[target]
        lengths = {
            id(record): _single_pose_token_count(record["batch"])
            for record in group
        }
        # Length sorting reduces Nmax padding.  Original ordinal/index remains
        # the deterministic tie-breaker; output rows are restored separately.
        ordered = sorted(
            group,
            key=lambda record: (
                lengths[id(record)],
                int(record.get("ordinal", record.get("index", 0))),
                int(record.get("index", 0)),
            ),
        )
        current: list[Mapping[str, Any]] = []
        current_max = 0
        for record in ordered:
            length = lengths[id(record)]
            candidate_max = max(current_max, length)
            within = (
                len(current) < batch_limit
                and (len(current) + 1) * candidate_max * candidate_max <= pair_budget
            )
            if current and not within:
                planned.append(current)
                current = []
                current_max = 0
                candidate_max = length
            current.append(record)
            current_max = candidate_max if current_max else length
            # A single oversized pose is intentionally emitted as a singleton;
            # callers score it at full length rather than crop or discard it.
            if len(current) == 1 and length * length > pair_budget:
                planned.append(current)
                current = []
                current_max = 0
        if current:
            planned.append(current)
    return planned
