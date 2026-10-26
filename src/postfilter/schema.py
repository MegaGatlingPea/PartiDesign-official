"""Small metadata records passed between the driver and Ray workers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Candidate:
    task_dir: str
    method_dir: str
    task: str
    method: str
    sample: str
    seq_index: int
    source: str
    source_index: int
    prediction_key: str
    refold_path: str
    record_path: str
    binder_chains: tuple[str, ...]
    target_chains: tuple[str, ...]
    sequences: dict[str, str]
    passing_criteria: tuple[str, ...]
    binder_kind: str = "protein"
    declared_cyclic_chains: tuple[str, ...] = ()

    @property
    def identity(self) -> str:
        return f"{self.method}/{self.sample}/seq{self.seq_index}/{self.prediction_key}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Candidate":
        data = dict(value)
        for key in ("binder_chains", "target_chains", "passing_criteria", "declared_cyclic_chains"):
            data[key] = tuple(data.get(key, ()))
        return cls(**data)


@dataclass
class Selection:
    candidates: list[Candidate] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)
    task_dirs: list[str] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
