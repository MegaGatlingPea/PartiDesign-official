"""Design-facing PartiScore adapter; the shared model is loaded only on use."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import torch


def ranking_score(result: Any) -> float:
    """Translate only the declared no-contact outcome to a ranking sentinel."""
    if result.status == "no_contact":
        if result.score is not None or result.pair_count != 0:
            raise RuntimeError("PartiScore no_contact must have null score and zero pair_count")
        return -math.inf
    if result.status != "ok":
        raise ValueError(result.error or f"PartiScore reward is inapplicable: {result.status}")
    if isinstance(result.score, bool) or not isinstance(result.score, (int, float)):
        raise RuntimeError("PartiScore returned a nonnumeric successful score")
    score = float(result.score)
    if not math.isfinite(score):
        raise RuntimeError("PartiScore returned a non-finite successful score")
    return score


def reward_record(result: Any, *, source: str = "design") -> dict[str, Any]:
    return {
        "reward_score": result.score,
        "reward_status": result.status,
        "reward_error": result.error,
        "reward_structure_source": source,
        "reward_weights": result.weights,
        "reward_checkpoint": str(result.checkpoint),
    }


@dataclass(frozen=True)
class PartiScoreBatch:
    candidate_ids: tuple[str, ...]
    total: torch.Tensor
    failures: dict[str, str]
    records: dict[str, dict[str, Any]]


class PartiScoreReward:
    def __init__(self, scorer: Callable[[], Any]) -> None:
        self._scorer = scorer

    def score_batch(self, candidates: Sequence[Any], *, device: Any = "cpu") -> PartiScoreBatch:
        ids = [candidate.candidate_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("PartiScore candidate IDs must be unique")
        if not candidates:
            return PartiScoreBatch((), torch.empty(0, dtype=torch.float64, device=device), {}, {})
        results = self._scorer().score_structures(
            [candidate.atoms for candidate in candidates],
            identities=ids,
            design_chains=[
                [chain.chain_id for chain in candidate.base_request.task.design_chains]
                for candidate in candidates
            ],
            scope="design_groups",
        )
        if [result.identity for result in results] != ids:
            raise RuntimeError("PartiScore results differ from requested candidate IDs")
        kept: list[str] = []
        scores: list[float] = []
        failures: dict[str, str] = {}
        records: dict[str, dict[str, Any]] = {}
        for candidate, result in zip(candidates, results, strict=True):
            antibody_metrics = {}
            if getattr(candidate.base_request.task, "format", "normal") == "antibody":
                from ..antibody.scoring import (antibody_binder_chains,
                                               antibody_score_components, antibody_score_record)
                binder = antibody_binder_chains(candidate)
                antibody_metrics = antibody_score_record(result, binder)
                result, _internal = antibody_score_components(result, binder)
            if result.status in {"error", "no_interface"}:
                failures[result.identity] = result.error or f"PartiScore: {result.status}"
                continue
            scores.append(ranking_score(result))
            kept.append(result.identity)
            records[result.identity] = {**reward_record(result), **antibody_metrics}
        return PartiScoreBatch(
            tuple(kept), torch.tensor(scores, dtype=torch.float64, device=device), failures, records
        )
