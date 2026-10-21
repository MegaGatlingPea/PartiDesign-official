"""JSON-safe public structure scoring records."""
from dataclasses import asdict, dataclass, field


class ScoreInputError(ValueError):
    """Declared structure-input failure, safe to report for one candidate."""

@dataclass(frozen=True)
class InterfaceScore:
    partner_a: str
    partner_b: str
    score: float | None
    mean_log_density: float | None
    density_sum: float
    pair_count: int
    status: str
    error: str | None = None

    def as_dict(self):
        return asdict(self)

@dataclass(frozen=True)
class ScoreResult:
    identity: str
    score: float | None = None
    mean_log_density: float | None = None
    density_sum: float = 0.0
    pair_count: int = 0
    status: str = "error"
    error: str | None = None
    interfaces: list[InterfaceScore] = field(default_factory=list)
    checkpoint: str = ""
    weights: str = "raw"
    scope: str = "all"

    def as_dict(self):
        return asdict(self)
