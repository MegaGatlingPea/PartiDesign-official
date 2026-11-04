"""Exact shared context-MSA preparation for the native ESMFold2 schema."""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .af3_msa import AF3MSACache, validate_msa_settings


def msa_settings(section: Mapping[str, Any]) -> dict[str, Any]:
    """Project shared settings to ESMFold2's protein-alignment consumer."""
    raw = section.get("msa", {"mode": "off"})
    if isinstance(raw, Mapping):
        raw = dict(raw, templates=False)
        if section.get("mode", "fast") == "fast":
            raw.update(mode="off", refresh=False, cache_profile=None)
    return validate_msa_settings(raw)


def _has_msa(entry: Mapping[str, Any]) -> bool:
    msa = entry.get("msa")
    return any(bool(value) for value in msa.values()) if isinstance(msa, Mapping) else bool(msa)


def msa_queries(payload: Mapping[str, Any], *, settings: Mapping[str, Any]) -> list[dict[str, str]]:
    """Query final mapped protein context entries without supplied alignments."""
    if settings.get("mode", "off") == "off":
        return []
    queries = []
    for entry in payload.get("sequences", ()):
        if (entry.get("modality") != "protein" or entry.get("role") != "context"
                or _has_msa(entry)):
            continue
        sequence = entry.get("sequence")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("ESMFold2 context MSA requires a final nonempty protein sequence")
        queries.append({"modality": "protein", "sequence": sequence})
    return queries


def attach_msa_references(
    payload: Mapping[str, Any], references: Sequence[Mapping[str, Any]], *,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach immutable shared paths while retaining explicitly supplied MSA."""
    result = copy.deepcopy(dict(payload))
    required = {(query["modality"], query["sequence"])
                for query in msa_queries(result, settings=settings)}
    previous = result.get("msa_context")
    context = dict(previous) if isinstance(previous, Mapping) else {}
    old_chains = {chain.get("chain_id"): chain for chain in context.get("chains", ())
                  if isinstance(chain, Mapping)}
    chains = []
    for entry in result.get("sequences", ()):
        identity = (entry.get("modality"), entry.get("sequence"))
        if (entry.get("role") == "context" and identity in required
                and not _has_msa(entry)):
            matching = [ref for ref in references
                        if (ref.get("modality"), ref.get("sequence")) == identity]
            if len(matching) != 1:
                raise ValueError(f"Expected one ready MSA reference for ESMFold2 chain {entry.get('chain_id')}")
            reference = matching[0]
            raw_path = reference.get("unpaired_msa_path")
            if not isinstance(raw_path, (str, Path)) or not str(raw_path):
                raise ValueError("ESMFold2 MSA reference has no unpaired alignment path")
            entry["msa"] = {"unpaired_msa_path": str(Path(raw_path).expanduser().resolve())}
            paired = reference.get("paired_msa_path")
            if paired:
                entry["msa"]["paired_msa_path"] = str(Path(paired).expanduser().resolve())
            evidence = {key: copy.deepcopy(reference.get(key)) for key in (
                "entry_id", "profile_id", "profile_text", "cache_dir", "generation", "generation_id",
                "provenance", "source", "server_url", "msa_depth", "paired_msa_depth")}
        elif _has_msa(entry):
            old = old_chains.get(entry.get("chain_id"), {})
            evidence = (dict(old) if old.get("modality") == identity[0]
                        and old.get("sequence") == identity[1] else {"msa": copy.deepcopy(entry["msa"])})
        else:
            continue
        chains.append({**evidence, "chain_id": entry.get("chain_id"),
                       "modality": identity[0], "sequence": identity[1]})
    result["msa_policy"] = "context" if chains else "none"
    context.update(mode="context" if chains else "off", chains=chains,
                   policy={key: settings.get(key) for key in (
                       "mode", "source", "server_url", "max_template_date", "templates")})
    result["msa_context"] = context
    return result


def prepare_esmfold2_input(
    input_record: Any, variant: Mapping[str, Any], section: Mapping[str, Any], repo_root: str | Path,
    *, references: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prepare the final native request before any model or service allocation."""
    from .esmfold2 import _build_payload

    payload = _build_payload(input_record, variant)
    previous = payload.get("msa_context")
    payload["msa_context"] = dict(previous) if isinstance(previous, Mapping) else {}
    payload["msa_context"]["native"] = {
        "max_depth": section.get("msa_max_depth", 1024),
        "column_mask_rate": section.get("msa_column_mask_rate", 0.1),
        "subsample_at_inference": section.get("msa_subsample_at_inference", True),
    }
    settings = msa_settings(section)
    if references is None:
        queries = {(query["modality"], query["sequence"])
                   for query in msa_queries(payload, settings=settings)}
        references = []
        if queries:
            cache = AF3MSACache(repo_root, settings)
            references = [cache.ensure(modality, sequence) for modality, sequence in sorted(queries)]
    return attach_msa_references(payload, references, settings=settings)
