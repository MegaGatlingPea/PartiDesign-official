"""Deterministic, CPU-only sequence selection for hallucination rounds.

A descriptor represents public residue identities, not MPNN parent proxies.
Cyclic equivalence is forward rotation only. Constraint labels supplied in
anchors rotate with their residue and participate in every comparison.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil, isfinite
from numbers import Real
from typing import Iterable, Sequence

from rapidfuzz.distance.Levenshtein import distance as _edit_distance
from rapidfuzz.process import cdist as _edit_cdist

_POLYMER_MODALITIES = frozenset({"protein", "rna", "dna"})


def _nonempty_string(value, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    return value


def _integer(value, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class ChainSequence:
    chain_id: str
    modality: str
    tokens: tuple[str, ...]
    cyclic: bool = False
    anchors: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self):
        _nonempty_string(self.chain_id, "chain_id")
        if self.modality not in _POLYMER_MODALITIES:
            raise ValueError("sequence selection supports protein, rna, or dna design chains; ligand design is unsupported")
        if type(self.cyclic) is not bool:
            raise ValueError("cyclic must be boolean")
        if not isinstance(self.tokens, tuple) or not self.tokens:
            raise ValueError("tokens must be a nonempty tuple of residue identities")
        for token in self.tokens:
            _nonempty_string(token, "residue token")
        if not isinstance(self.anchors, tuple):
            raise ValueError("anchors must be a tuple")
        anchors = self.anchors or tuple(() for _ in self.tokens)
        if len(anchors) != len(self.tokens):
            raise ValueError("anchors must have one entry per residue token")
        normalized = []
        for values in anchors:
            if not isinstance(values, tuple):
                raise ValueError("each residue's anchors must be a tuple")
            for value in values:
                _nonempty_string(value, "anchor")
            normalized.append(tuple(sorted(set(values))))
        object.__setattr__(self, "anchors", tuple(normalized))


def _chains(chains: Sequence[ChainSequence]) -> tuple[ChainSequence, ...]:
    result = tuple(chains)
    if not result or any(not isinstance(chain, ChainSequence) for chain in result):
        raise ValueError("at least one polymer ChainSequence is required")
    ids = [chain.chain_id for chain in result]
    if len(ids) != len(set(ids)):
        raise ValueError("design chain IDs must be unique")
    return result


@dataclass(frozen=True)
class SelectionCandidate:
    candidate_id: str
    parent_id: str
    score: float
    chains: tuple[ChainSequence, ...]
    family_id: str = ""

    def __post_init__(self):
        _nonempty_string(self.candidate_id, "candidate_id")
        _nonempty_string(self.parent_id, "parent_id")
        if self.family_id != "":
            _nonempty_string(self.family_id, "family_id")
        if isinstance(self.score, bool) or not isinstance(self.score, Real) or not isfinite(self.score):
            raise ValueError("candidate score must be finite and numeric")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "chains", _chains(self.chains))


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[SelectionCandidate, ...]
    decisions: dict[str, dict]
    counts: dict[str, int]


def _symbols(chain: ChainSequence) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(zip(chain.tokens, chain.anchors))


def _rotations(values: tuple | str, cyclic: bool) -> tuple:
    if not cyclic:
        return (values,)
    return tuple(dict.fromkeys(values[offset:] + values[:offset] for offset in range(len(values))))


def sequence_key(chains: Sequence[ChainSequence]) -> tuple:
    """Exact sequence/constraint key preserving chain roles and forward cycles.

    This is not a prediction-cache key: identical design sequences can still
    have different contexts, templates, parameters or initial guesses.
    """
    return tuple(
        (chain.chain_id, chain.modality, chain.cyclic,
         min(_rotations(_symbols(chain), chain.cyclic)))
        for chain in _chains(chains)
    )


@dataclass(frozen=True)
class _PreparedChain:
    role: tuple[str, str, bool]
    canonical: str
    strings: tuple[str, ...]
    counts: Counter | None


def _prepare_sequences(sequences):
    """Encode symbols/rotations once for a whole exact-distance computation."""
    alphabet = {}
    prepared = []
    for chains in sequences:
        result = []
        for chain in chains:
            encoded = "".join(chr(alphabet.setdefault(symbol, len(alphabet))) for symbol in _symbols(chain))
            rotations = _rotations(encoded, chain.cyclic)
            result.append(_PreparedChain(
                (chain.chain_id, chain.modality, chain.cyclic), min(rotations), rotations,
                Counter(encoded) if chain.cyclic else None))
        prepared.append(tuple(result))
    return prepared


def _prepared_chain_distance(left: _PreparedChain, right: _PreparedChain) -> float:
    if left.canonical == right.canonical:
        return 0.0
    length = max(len(left.canonical), len(right.canonical))
    best = _edit_distance(left.canonical, right.canonical)
    if not left.role[2] or len(left.strings) == len(right.strings) == 1:
        return best / length
    # Histogram lower bound applies to every pair of forward origins. Only
    # stop when this exact bound is reached; both rotation sets stay complete.
    a_counts, b_counts = left.counts, right.counts
    discrepancy = (sum(abs(count - b_counts.get(symbol, 0)) for symbol, count in a_counts.items())
                   + sum(count for symbol, count in b_counts.items() if symbol not in a_counts))
    lower = max(1, abs(len(left.canonical) - len(right.canonical)), (discrepancy + 1) // 2)
    if best == lower:
        return best / length
    # Compare both complete rotation sets in C++. Clipping at an already known
    # upper bound preserves the exact minimum and avoids Python's inner loop.
    return int(_edit_cdist(left.strings, right.strings, scorer=_edit_distance,
                           score_cutoff=best - 1).min()) / length


def _prepared_distance(left, right) -> float:
    if len(left) != len(right):
        return 1.0
    distance = 0.0
    for a, b in zip(left, right):
        if a.role != b.role:
            return 1.0
        distance = max(distance, _prepared_chain_distance(a, b))
        if distance == 1.0:
            break
    return distance


def sequence_distance(left_chains: Sequence[ChainSequence],
                      right_chains: Sequence[ChainSequence]) -> float:
    """Normalized unit edit distance; maximum over corresponding design chains.

    Cyclic chains consider all forward orientations of both operands. Chain
    role, modality or cyclic-intent mismatches have distance one.
    """
    left, right = _prepare_sequences((_chains(left_chains), _chains(right_chains)))
    return _prepared_distance(left, right)


def select_candidates(
    candidates: Iterable[SelectionCandidate],
    *,
    mode: str = "top",
    num_clusters: int = 200,
    max_pass: int | None = 200,
    max_pass_ratio: float | None = None,
    per_parent: int | None = None,
    per_family: int | None = None,
    unique_count: int | None = None,
) -> SelectionResult:
    """Select qualified candidates without model calls or file writes.

    Parent limits precede task-wide deduplication; family limits follow it.
    A family joins one retained refold with its partial children. No backfill.
    unique_count is the pre-screen unique sequence count, including cache
    hits. In cluster mode max_pass is unused: fixed K is the count policy.
    """
    if mode not in {"top", "cluster"}:
        raise ValueError("selection mode must be top or cluster")
    _integer(num_clusters, "num_clusters")
    if max_pass is not None:
        _integer(max_pass, "max_pass")
    if per_parent is not None:
        _integer(per_parent, "per_parent")
    if per_family is not None:
        _integer(per_family, "per_family")
    if unique_count is not None:
        _integer(unique_count, "unique_count", minimum=0)
    if max_pass_ratio is not None:
        if (isinstance(max_pass_ratio, bool) or not isinstance(max_pass_ratio, Real)
                or not isfinite(max_pass_ratio) or not 0 < max_pass_ratio <= 1):
            raise ValueError("max_pass_ratio must be finite in (0,1]")
        if mode != "top":
            raise ValueError("max_pass_ratio cannot be combined with fixed-K clustering")
    values = list(candidates)
    if any(not isinstance(value, SelectionCandidate) for value in values):
        raise ValueError("candidates must contain SelectionCandidate values")
    ids = [value.candidate_id for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique within a selection")
    ranked = sorted(values, key=lambda value: (-value.score, value.candidate_id))
    keys = {value.candidate_id: sequence_key(value.chains) for value in ranked}
    input_unique = len(set(keys.values()))
    denominator = input_unique if unique_count is None else unique_count
    if max_pass_ratio is not None and denominator < input_unique:
        raise ValueError("unique_count cannot be smaller than the qualified unique sequence count")
    decisions = {
        value.candidate_id: {"selected": False, "reason": "not_selected",
                             "duplicate_of": None, "cluster_id": None,
                             "representative_id": None}
        for value in ranked
    }
    parent_counts: dict[str, int] = {}
    parent_retained = []
    for value in ranked:
        count = parent_counts.get(value.parent_id, 0)
        if per_parent is not None and count >= per_parent:
            decisions[value.candidate_id]["reason"] = "parent_limit"
            continue
        parent_counts[value.parent_id] = count + 1
        parent_retained.append(value)
    seen: dict[tuple, SelectionCandidate] = {}
    unique = []
    for value in parent_retained:
        key = keys[value.candidate_id]
        if key in seen:
            decisions[value.candidate_id].update(
                reason="sequence_duplicate", duplicate_of=seen[key].candidate_id)
        else:
            seen[key] = value
            unique.append(value)

    family_retained = unique
    if per_family is not None:
        family_counts: dict[str, int] = {}
        family_retained = []
        for value in unique:
            family = value.family_id or value.parent_id
            count = family_counts.get(family, 0)
            if count >= per_family:
                decisions[value.candidate_id]["reason"] = "family_limit"
                continue
            family_counts[family] = count + 1
            family_retained.append(value)

    selected: list[SelectionCandidate] = []
    cluster_count = 0
    if mode == "cluster":
        cap = min(num_clusters, len(family_retained))
        if family_retained:
            if len(family_retained) <= num_clusters:
                labels = list(range(len(family_retained)))
            else:
                import numpy as np
                from scipy.cluster.hierarchy import cut_tree, linkage

                size = len(family_retained)
                condensed = np.empty(size * (size - 1) // 2, dtype=np.float64)
                prepared = _prepare_sequences(value.chains for value in family_retained)
                index = 0
                for left in range(size):
                    for right in range(left + 1, size):
                        condensed[index] = _prepared_distance(prepared[left], prepared[right])
                        index += 1
                tree = linkage(condensed, method="average")
                labels = cut_tree(tree, n_clusters=[num_clusters])[:, 0].tolist()
                if len(set(labels)) != num_clusters:
                    raise RuntimeError("fixed-K clustering did not return the requested number of nonempty clusters")
            groups: dict[int, list[SelectionCandidate]] = {}
            for value, label in zip(family_retained, labels):
                groups.setdefault(int(label), []).append(value)
            ordered = sorted(groups.values(), key=lambda group: (-group[0].score, group[0].candidate_id))
            cluster_count = len(ordered)
            for cluster_id, group in enumerate(ordered, start=1):
                representative = group[0]
                selected.append(representative)
                for value in group:
                    decisions[value.candidate_id].update(
                        selected=value is representative,
                        reason="selected" if value is representative else "cluster_member",
                        cluster_id=cluster_id, cluster_size=len(group),
                        representative_id=representative.candidate_id)
    else:
        cap = len(family_retained) if max_pass is None else min(len(family_retained), max_pass)
        if max_pass_ratio is not None:
            cap = min(cap, ceil(float(max_pass_ratio) * denominator))
        selected = family_retained[:cap]
        selected_ids = {value.candidate_id for value in selected}
        for value in family_retained:
            chosen = value.candidate_id in selected_ids
            decisions[value.candidate_id].update(
                selected=chosen, reason="selected" if chosen else "max_pass",
                representative_id=value.candidate_id if chosen else None)

    return SelectionResult(
        selected=tuple(selected), decisions=decisions,
        counts={"input_count": len(values),
                "parent_retained_count": len(parent_retained),
                "unique_count": len(unique), "selected_count": len(selected),
                "duplicate_count": len(parent_retained) - len(unique),
                "parent_limited_count": len(values) - len(parent_retained),
                "family_retained_count": len(family_retained),
                "family_limited_count": len(unique) - len(family_retained),
                "cluster_count": cluster_count, "selection_cap": cap,
                "denominator_count": denominator})


__all__ = ["ChainSequence", "SelectionCandidate", "SelectionResult",
           "sequence_key", "sequence_distance", "select_candidates"]
