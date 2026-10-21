"""Fixed Best-of-N/FK search mechanics and stable winner selection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Mapping, Sequence, TypeVar

import torch

from ..rewards.clash import ClashScoreBatch


StateT = TypeVar("StateT")
CandidateT = TypeVar("CandidateT")


class SearchError(ValueError):
    pass


class FKBranchRejected(Exception):
    """One FK branch produced no valid terminal candidate."""


@dataclass(frozen=True, slots=True)
class WinnerAssignment:
    output_key: object
    candidate: object
    score: float


def _candidate_id(candidate: object) -> str:
    value = getattr(candidate, "candidate_id", None)
    if not isinstance(value, str) or not value:
        raise SearchError("candidate_id must be non-empty")
    return value


def _candidate_ordinal(candidate: object) -> int:
    value = getattr(candidate, "candidate_ordinal", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SearchError("candidate_ordinal must be a nonnegative integer")
    return value


def _candidate_group(candidate: object) -> Hashable:
    explicit = getattr(candidate, "group_key", None)
    if explicit is not None:
        return explicit
    task_name = getattr(candidate, "task_name", None)
    if task_name is None:
        base_request = getattr(candidate, "base_request", None)
        task = getattr(base_request, "task", None)
        task_name = getattr(task, "name", None)
    lengths = getattr(candidate, "actual_joint_length", None)
    if not isinstance(task_name, str) or lengths is None:
        raise SearchError("candidate must expose task_name and actual_joint_length")
    return task_name, tuple(int(value) for value in lengths)


def stable_candidate_ordinal(
    base_global_ordinal: int,
    terminal_index: int,
    terminals_per_base: int,
) -> int:
    values = (base_global_ordinal, terminal_index, terminals_per_base)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise SearchError("candidate ordinal inputs must be integers")
    if base_global_ordinal < 0 or terminal_index < 0 or terminals_per_base <= 0:
        raise SearchError("candidate ordinal inputs are out of range")
    if terminal_index >= terminals_per_base:
        raise SearchError("terminal_index exceeds the declared per-base stride")
    return base_global_ordinal * terminals_per_base + terminal_index


def rank_candidates(
    candidates: Sequence[object],
    scores: ClashScoreBatch,
) -> list[tuple[object, float]]:
    candidate_by_id = {_candidate_id(candidate): candidate for candidate in candidates}
    if len(candidate_by_id) != len(candidates):
        raise SearchError("candidate IDs are not unique")
    if set(candidate_by_id) != set(scores.candidate_ids):
        raise SearchError("reward result candidate IDs do not match the candidate set")
    score_by_id = {
        candidate_id: float(value)
        for candidate_id, value in zip(scores.candidate_ids, scores.total.tolist())
    }
    return sorted(
        ((candidate, score_by_id[_candidate_id(candidate)]) for candidate in candidates),
        key=lambda pair: (-pair[1], _candidate_ordinal(pair[0])),
    )


def select_group_winners(
    candidates: Sequence[object],
    scores: ClashScoreBatch,
    output_slots_by_group: Mapping[Hashable, Sequence[object]],
) -> tuple[list[WinnerAssignment], dict[Hashable, int]]:
    """Assign ranked candidates to pre-enumerated slots; never invent/backfill slots."""

    ranked_by_group: dict[Hashable, list[tuple[object, float]]] = defaultdict(list)
    for candidate, score in rank_candidates(candidates, scores):
        ranked_by_group[_candidate_group(candidate)].append((candidate, score))

    assignments: list[WinnerAssignment] = []
    missing: dict[Hashable, int] = {}
    for group, slots in output_slots_by_group.items():
        winners = ranked_by_group.get(group, ())
        count = min(len(slots), len(winners))
        for slot, (candidate, score) in zip(slots[:count], winners[:count]):
            assignments.append(WinnerAssignment(slot, candidate, score))
        if count < len(slots):
            missing[group] = len(slots) - count
    unexpected = set(ranked_by_group) - set(output_slots_by_group)
    if unexpected:
        raise SearchError(f"candidate groups have no requested output slots: {unexpected}")
    return assignments, missing


def best_of_n_specs(base_requests: Sequence[object], replicas: int) -> list[tuple[object, int, int]]:
    """Return ``(request, replica_index, global_candidate_ordinal)`` specs."""

    if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas <= 0:
        raise SearchError("Best-of-N replicas must be a positive integer")
    result: list[tuple[object, int, int]] = []
    for request in base_requests:
        global_ordinal = getattr(request, "global_ordinal", None)
        if isinstance(global_ordinal, bool) or not isinstance(global_ordinal, int):
            raise SearchError("base request lacks a global_ordinal")
        for replica in range(replicas):
            result.append(
                (
                    request,
                    replica,
                    stable_candidate_ordinal(global_ordinal, replica, replicas),
                )
            )
    return result


def _score_vector(scores: ClashScoreBatch, candidate_ids: Sequence[str]) -> torch.Tensor:
    score_by_id = {
        candidate_id: value for candidate_id, value in zip(scores.candidate_ids, scores.total)
    }
    if set(score_by_id) != set(candidate_ids):
        raise SearchError("FK reward result IDs do not match lookahead candidates")
    result = torch.stack([score_by_id[candidate_id] for candidate_id in candidate_ids])
    if not bool(torch.isfinite(result).all().item()):
        raise SearchError("FK reward returned non-finite values")
    return result.float()


@torch.inference_mode()
def run_fk_request(
    initial_states: Sequence[StateT],
    *,
    step_checkpoints: Sequence[int],
    n_branch: int,
    temperature: float,
    advance_segment: Callable[[StateT, int, int, int], StateT],
    terminal_lookahead: Callable[[StateT, int, int, int], CandidateT],
    score_lookahead: Callable[[Sequence[CandidateT]], ClashScoreBatch],
    generator: torch.Generator | None = None,
) -> list[CandidateT]:
    """Run FK steering for one resolved base request entirely in memory."""

    if not initial_states:
        raise SearchError("FK needs at least one initial beam state")
    if isinstance(n_branch, bool) or not isinstance(n_branch, int) or n_branch <= 0:
        raise SearchError("FK n_branch must be positive")
    if not torch.isfinite(torch.tensor(float(temperature))) or temperature <= 0:
        raise SearchError("FK temperature must be finite and positive")
    checkpoints = [int(value) for value in step_checkpoints]
    if len(checkpoints) < 2 or checkpoints[0] != 0:
        raise SearchError("FK checkpoints must start at zero and contain an interval")
    if any(right <= left for left, right in zip(checkpoints, checkpoints[1:])):
        raise SearchError("FK checkpoints must be strictly increasing")

    beam_width = len(initial_states)
    states = list(initial_states)
    terminal_step = checkpoints[-1]
    final_candidates: list[CandidateT] = []
    for round_index, (start, end) in enumerate(zip(checkpoints, checkpoints[1:])):
        branches: list[StateT] = []
        lookahead: list[CandidateT] = []
        for parent_slot, parent in enumerate(states):
            for branch_index in range(n_branch):
                attempt_index = (
                    round_index * beam_width * n_branch
                    + parent_slot * n_branch
                    + branch_index
                )
                try:
                    branch = advance_segment(parent, start, end, branch_index)
                    candidate = terminal_lookahead(
                        branch, end, terminal_step, attempt_index
                    )
                except FKBranchRejected:
                    continue
                branches.append(branch)
                lookahead.append(candidate)
        if not lookahead:
            return []
        ids = [_candidate_id(candidate) for candidate in lookahead]
        values = _score_vector(score_lookahead(lookahead), ids)
        probabilities = torch.softmax(values / float(temperature), dim=0)
        if not bool(torch.isfinite(probabilities).all().item()):
            raise SearchError("FK resampling probabilities are non-finite")
        selected = torch.multinomial(
            probabilities,
            num_samples=beam_width,
            replacement=True,
            generator=generator,
        )
        if end == terminal_step:
            final_candidates = [lookahead[int(index)] for index in selected.tolist()]
            break
        states = [branches[int(index)] for index in selected.tolist()]
    if not final_candidates:
        raise SearchError("FK did not reach its terminal checkpoint")
    return final_candidates


__all__ = [
    "FKBranchRejected",
    "SearchError",
    "WinnerAssignment",
    "best_of_n_specs",
    "rank_candidates",
    "run_fk_request",
    "select_group_winners",
    "stable_candidate_ordinal",
]
