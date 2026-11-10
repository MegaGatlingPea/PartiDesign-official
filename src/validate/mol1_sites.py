"""Project declared protein-context hotspots into Mol-1 native SITE inputs.

Only original source membership is cached. Native chain/sequence positions are
bound afresh for each candidate; partners include all designed chains.
"""
from __future__ import annotations

from collections.abc import Mapping

from .pocket import (
    PocketResolutionError, SourcePocketCache, _context_position, resolve_source_hotspot,
)


class SourceSiteCache(SourcePocketCache):
    """Owner-local source membership cache shared with the pocket resolver."""


def _without_site(payload):
    if not isinstance(payload, Mapping) or ("mol1_site" not in payload and "mol1_site_evidence" not in payload):
        return payload
    return {key: value for key, value in payload.items()
            if key not in {"mol1_site", "mol1_site_evidence"}}


def _protein_position(payload, native_id, position, source_id):
    forward = payload.get("source_to_native_chain", {}).get(source_id)
    if not isinstance(forward, (list, tuple)):
        raise PocketResolutionError(f"Mol-1 SITE chain {native_id!r} has an incorrect source binding")
    chain = _context_position(payload, native_id, position, source_id)
    if chain["modality"] != "protein":
        raise PocketResolutionError("Mol-1 SITE hotspots require protein context residues")
    return chain


def _site_payload(payload, contacts, evidence):
    ids, partners = set(), []
    for chain in payload["chains"]:
        native_id = chain.get("id")
        if not isinstance(native_id, str) or not native_id or native_id in ids:
            raise PocketResolutionError("Mol-1 SITE native chain IDs must be unique and nonempty")
        ids.add(native_id)
        if chain.get("role") == "design":
            partners.append(native_id)
    if not partners:
        raise PocketResolutionError("Mol-1 SITE requires at least one designed partner chain")
    residues = sorted({(chain, position) for chain, position in contacts})
    sources = sorted({chain for chain, _ in residues})
    if set(sources) & set(partners):
        raise PocketResolutionError("Mol-1 SITE source and partner chains must be disjoint")
    site = {"source_chains": sources, "partner_chains": sorted(partners),
            "residue_ids": [[chain, position] for chain, position in residues]}
    return dict(payload, mol1_site=site, mol1_site_evidence=evidence)


def bind_mol1_source_site(input_record, payload, *, repo_root=None, cache=None):
    """Render exact retained source hotspots as protein-source Mol-1 SITE."""
    resolved = resolve_source_hotspot(
        input_record, payload, repo_root=repo_root,
        cache=cache if cache is not None else SourceSiteCache(),
    )
    if resolved is None:
        return _without_site(payload)
    contacts = resolved["contacts"]
    for record in resolved["source_occurrences"]:
        _protein_position(payload, record["native_chain_id"],
                          record["native_residue_index"], record["source_chain_id"])
    evidence = {key: value for key, value in resolved.items() if key != "contacts"}
    return _site_payload(payload, contacts, evidence)


def bind_mol1_sequence_site(request, payload):
    """Bind explicit context sequence positions; no source atoms are invented."""
    contacts, records = [], []
    for authored in request["chains"]:
        positions = authored.get("pocket")
        if positions is None:
            continue
        if not isinstance(positions, (list, tuple)):
            raise PocketResolutionError("Mol-1 SITE pocket must be a list of 1-based sequence positions")
        if not positions:
            continue
        if authored.get("role") != "context" or authored.get("modality") != "protein":
            raise PocketResolutionError("Mol-1 SITE hotspots require protein context residues")
        source_id = authored["id"]
        native_ids = payload.get("source_to_native_chain", {}).get(source_id)
        if (not isinstance(native_ids, (list, tuple)) or len(native_ids) != 1
                or not isinstance(native_ids[0], str)):
            raise PocketResolutionError(f"Mol-1 SITE sequence chain {source_id!r} has no unique native mapping")
        native_id = native_ids[0]
        for position in positions:
            native = _protein_position(payload, native_id, position, source_id)
            if native["sequence"] != authored.get("sequence"):
                raise PocketResolutionError(f"Mol-1 SITE sequence chain {source_id!r} differs from its native sequence binding")
            contacts.append((native_id, position))
            records.append({"source_chain_id": source_id, "sequence_position": position,
                            "native_chain_id": native_id, "native_residue_index": position})
    if not contacts:
        return _without_site(payload)
    return _site_payload(payload, contacts, {"source": "sequence", "source_occurrences": records})


__all__ = ["SourceSiteCache", "bind_mol1_source_site", "bind_mol1_sequence_site"]
