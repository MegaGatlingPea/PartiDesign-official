"""Pure AlphaFold2 reward eligibility and score reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from ..core.records import SearchCandidate
from .esmfold2 import _AA3_TO_1, protein_tokens


@dataclass(frozen=True, slots=True)
class AF2Confidence:
    candidate_id: str
    model: str
    design_plddt: float
    ptm: float
    pae: float
    interface_iptm: float | None = None
    interface_ipae: float | None = None


@dataclass(frozen=True, slots=True)
class AF2ScoreBatch:
    candidate_ids: tuple[str, ...]
    total: torch.Tensor

    def __post_init__(self) -> None:
        if self.total.dtype != torch.float32 or self.total.shape != (
            len(self.candidate_ids),
        ):
            raise ValueError("AF2 total score has invalid tensor ABI")
        if not bool(torch.isfinite(self.total).all().item()):
            raise ValueError("AF2 total score contains non-finite values")


def af2_design_chain(candidate: SearchCandidate, model: str) -> str:
    atoms = candidate.atoms
    modalities = set(np.asarray(atoms.mol_type).astype(str).tolist())
    if modalities != {"protein"}:
        raise ValueError("AF2 reward accepts only all-protein candidates")
    if any(str(value) not in _AA3_TO_1 for value in atoms.res_name):
        raise ValueError("AF2 reward requires standard protein residues")
    tokens = protein_tokens(atoms)
    design = sorted({token.chain_id for token in tokens if token.is_design})
    context = sorted({token.chain_id for token in tokens if not token.is_design})
    if len(design) != 1:
        raise ValueError("AF2 reward requires exactly one protein design chain")
    if model == "ptm" and context:
        raise ValueError("AF2 pTM accepts a protein monomer without context chains")
    if model == "multimer" and not context:
        raise ValueError("AF2 Multimer requires at least one protein context chain")
    if model not in {"ptm", "multimer"}:
        raise ValueError("AF2 model must be ptm or multimer")
    return design[0]


def score_af2_confidences(
    values: Sequence[AF2Confidence],
    *,
    monomer_plddt_weight: float,
    complex_plddt_weight: float,
    complex_iptm_weight: float,
    device: torch.device | str = "cpu",
) -> AF2ScoreBatch:
    ids = tuple(value.candidate_id for value in values)
    if len(ids) != len(set(ids)):
        raise ValueError("AF2 confidence candidate IDs must be unique")
    totals: list[float] = []
    for value in values:
        plddt = float(value.design_plddt)
        ptm = float(value.ptm)
        pae = float(value.pae)
        if not np.isfinite(plddt) or not 0 <= plddt <= 100:
            raise ValueError("AF2 design pLDDT must be finite in [0,100]")
        if not np.isfinite(ptm) or not 0 <= ptm <= 1:
            raise ValueError("AF2 pTM must be finite in [0,1]")
        if not np.isfinite(pae) or pae < 0:
            raise ValueError("AF2 PAE must be finite and nonnegative")
        normalized = plddt / 100.0
        if value.model == "ptm":
            if value.interface_iptm is not None or value.interface_ipae is not None:
                raise ValueError("AF2 pTM result cannot carry interface metrics")
            totals.append(float(monomer_plddt_weight) * normalized)
            continue
        iptm = float(value.interface_iptm) if value.interface_iptm is not None else np.nan
        ipae = float(value.interface_ipae) if value.interface_ipae is not None else np.nan
        if not np.isfinite(iptm) or not 0 <= iptm <= 1:
            raise ValueError("AF2 Multimer ipTM must be finite in [0,1]")
        if not np.isfinite(ipae) or ipae < 0:
            raise ValueError("AF2 Multimer iPAE must be finite and nonnegative")
        totals.append(
            float(complex_plddt_weight) * normalized
            + float(complex_iptm_weight) * iptm
        )
    return AF2ScoreBatch(ids, torch.tensor(totals, dtype=torch.float32, device=device))


class AF2Reward:
    """Configuration marker; JAX work is deferred until FM offload."""

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)


__all__ = [
    "AF2Confidence",
    "AF2Reward",
    "AF2ScoreBatch",
    "af2_design_chain",
    "score_af2_confidences",
]
