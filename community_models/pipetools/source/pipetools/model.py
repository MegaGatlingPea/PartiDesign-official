from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .json_safety import normalize_nonfinite


@dataclass(frozen=True, slots=True)
class ChainRecord:
    chain_id: str
    entity_id: str
    modality: str
    sequence: str
    entity_key: str
    source_chain_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResidueRecord:
    """Observed source-to-canonical residue/atom identity.

    ``position`` is the canonical label sequence position used by private inputs.
    Source identifiers retain the author numbering exposed by the input file.
    Missing polymer residues have no atom record and are intentionally absent.
    """

    chain_id: str
    position: int
    ins_code: str
    source_chain_id: str
    source_res_id: int
    source_ins_code: str
    res_name: str
    atom_names: tuple[str, ...]
    atom_elements: tuple[str, ...] = ()


@dataclass(slots=True)
class CanonicalSample:
    sample_id: str
    source_path: Path
    canonical_cif: Path
    canonical_json: Path
    inference_json: Path
    chains: list[ChainRecord]
    design_chain: ChainRecord
    context_chains: list[ChainRecord]
    residues: list[ResidueRecord] = field(default_factory=list)
    motif_residues: list[tuple[str, int, str]] = field(default_factory=list)
    motif_atoms: list[tuple[str, int, str, str]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("source_path", "canonical_cif", "canonical_json", "inference_json"):
            payload[key] = str(payload[key])
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CanonicalSample":
        data = dict(payload)
        for key in ("source_path", "canonical_cif", "canonical_json", "inference_json"):
            data[key] = Path(data[key])
        data["chains"] = [ChainRecord(**item) for item in data["chains"]]
        data["design_chain"] = ChainRecord(**data["design_chain"])
        data["context_chains"] = [ChainRecord(**item) for item in data["context_chains"]]
        data["residues"] = [
            ResidueRecord(
                **{
                    **item,
                    "atom_names": tuple(item.get("atom_names", ())),
                    "atom_elements": tuple(item.get("atom_elements", ())),
                }
            )
            for item in data.get("residues", [])
        ]
        data["motif_residues"] = [tuple(item) for item in data.get("motif_residues", [])]
        data["motif_atoms"] = [tuple(item) for item in data.get("motif_atoms", [])]
        return cls(**data)


@dataclass(slots=True)
class SequenceVariant:
    variant_id: str
    source: str
    design_chain: str
    design_sequence: str
    chain_sequences: dict[str, str]
    scores: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SequenceVariant":
        data = dict(payload)
        return cls(**data)


@dataclass(slots=True)
class FoldResult:
    sample_id: str
    variant_id: str
    sample_index: int
    structure_path: str | None
    confidence: dict[str, Any]
    status: str
    error: str | None = None
    raw_dir: str | None = None
    confidence_unavailable: list[dict[str, str]] = field(default_factory=list)
    atom_omissions: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FoldResult":
        data = dict(payload)
        confidence, unavailable = normalize_nonfinite(data.get("confidence", {}))
        recorded = data.get("confidence_unavailable", [])
        if not isinstance(recorded, list):
            recorded = []
        data["confidence"] = confidence
        data["confidence_unavailable"] = _merge_unavailable(recorded, unavailable)
        return cls(**data)


def _merge_unavailable(
    recorded: list[Any],
    discovered: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in [*recorded, *discovered]:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        reason = item.get("reason")
        if not isinstance(path, str) or not isinstance(reason, str):
            continue
        identity = (path, reason)
        if identity not in seen:
            seen.add(identity)
            merged.append({"path": path, "reason": reason})
    return merged
