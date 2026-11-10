"""Coordinate-free requests for the existing resident cofold backends.

Only native sequence and SMILES parsers create atoms. There is no source
AtomArray, source residue ledger, or reference structure in this interface.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import re

_POLYMERS = frozenset({"protein", "rna", "dna"})
_ALPHABET = {"protein": frozenset("ACDEFGHIKLMNPQRSTVWY"),
             "rna": frozenset("ACGU"), "dna": frozenset("ACGT")}
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def validate_sequence_request(request):
    """Validate the normalized v1 request without constructing coordinates."""
    if not isinstance(request, Mapping) or type(request.get("version")) is not int or request["version"] != 1:
        raise ValueError("Sequence request version must be 1")
    name = request.get("name")
    if (not isinstance(name, str) or name in {"", ".", ".."}
            or Path(name).name != name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)):
        raise ValueError("Sequence request name must be a safe file basename")
    chains = request.get("chains")
    if not isinstance(chains, list) or not chains:
        raise ValueError("Sequence request chains must be a nonempty list")
    normalized, seen = [], set()
    for raw in chains:
        if not isinstance(raw, Mapping):
            raise ValueError("Sequence request chain must be a mapping")
        chain = deepcopy(dict(raw))
        cid, role, modality = chain.get("id"), chain.get("role"), chain.get("modality")
        if not isinstance(cid, str) or not _ID.fullmatch(cid) or cid in seen:
            raise ValueError("Sequence chain IDs must be unique letter-prefixed native IDs")
        seen.add(cid)
        if role not in {"context", "design"} or modality not in _POLYMERS | {"ligand"}:
            raise ValueError(f"Sequence chain {cid!r} has an invalid role/modality")
        if modality in _POLYMERS:
            sequence = chain.get("sequence")
            if (not isinstance(sequence, str) or not sequence or set(sequence) - _ALPHABET[modality]
                    or "smiles" in chain):
                raise ValueError(f"Sequence chain {cid!r} requires an exact canonical {modality} sequence")
        else:
            smiles = chain.get("smiles")
            if not isinstance(smiles, str) or not smiles or "sequence" in chain:
                raise ValueError(f"Ligand chain {cid!r} requires SMILES")
            from rdkit import Chem
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None or not any(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()):
                raise ValueError(f"Ligand chain {cid!r} has invalid or empty heavy-atom SMILES")
            chain["canonical_isomeric_smiles"] = Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True)
        if chain.get("msa_reference") is not None and role != "context":
            raise ValueError("Designed chains must remain query-only")
        if chain.get("msa") is not None:
            if role != "context":
                raise ValueError("Designed chains must remain query-only")
            from .af3_msa import validate_msa_settings
            chain["msa"] = validate_msa_settings(chain["msa"])
        pocket = chain.get("pocket")
        if pocket is not None:
            if (role != "context" or modality not in _POLYMERS or not isinstance(pocket, list)
                    or any(type(pos) is not int or not 1 <= pos <= len(chain["sequence"]) for pos in pocket)
                    or len(set(pocket)) != len(pocket)):
                raise ValueError(f"Pocket for {cid!r} must contain unique 1-based context polymer positions")
        normalized.append(chain)
    design = request.get("design_chains")
    if (not isinstance(design, list) or not design or any(not isinstance(cid, str) for cid in design)
            or len(set(design)) != len(design)
            or set(design) != {c["id"] for c in normalized if c["role"] == "design"}):
        raise ValueError("Sequence request design_chains must exactly match design roles")
    return {**deepcopy(dict(request)), "chains": normalized, "design_chains": list(design)}


def _settings(chain, method, config):
    from .af3_msa import validate_msa_settings
    settings = validate_msa_settings(chain.get("msa", config.get("msa", {"mode": "off"})))
    if method == "esmfold2-fast":
        settings.update(mode="off", templates=False, refresh=False, cache_profile=None)
    elif method == "esmfold2-standard":
        settings["templates"] = False
    return settings


def prepare_sequence_payload(request, repo, method, config, cache=None):
    """Attach per-context exact shared cache references; candidates stay query-only."""
    from .af3_msa import AF3MSACache, msa_query_enabled
    from src.design.rewards.af3_input import _validate_msa_reference
    request = validate_sequence_request(request)
    caches = cache if isinstance(cache, dict) else {}
    chains, contexts, policies = [], [], {}
    for original in request["chains"]:
        entry = {key: deepcopy(value) for key, value in original.items()
                 if key not in {"msa", "msa_reference", "pocket", "canonical_isomeric_smiles"}}
        entry["source_id"] = entry["id"]
        entry["input_semantics"] = "sequence"
        if entry["role"] == "context":
            settings = _settings(original, method, config)
            policies[entry["id"]] = settings
            eligible = msa_query_enabled(settings, entry["modality"])
            if method in {"boltz2", "esmfold2-standard", "esmfold2-fast"}:
                eligible = eligible and entry["modality"] == "protein"
            if eligible:
                reference = original.get("msa_reference")
                if reference is None:
                    key = json.dumps(settings, sort_keys=True)
                    if isinstance(cache, AF3MSACache):
                        current_cache = cache
                    else:
                        current_cache = caches.get(key)
                        if current_cache is None:
                            current_cache = caches[key] = AF3MSACache(repo, settings)
                    reference = current_cache.ensure(entry["modality"], entry["sequence"])
                reference = deepcopy(dict(reference))
                if not settings["templates"]:
                    reference["templates"] = []
                    reference["template_count"] = 0
                _validate_msa_reference(entry, reference)
                entry["msa_reference"] = reference
                contexts.append({"native_id": entry["id"], "source_id": entry["id"],
                                 **{key: reference.get(key) for key in (
                                     "modality", "sequence", "entry_id", "profile_id", "cache_dir", "generation",
                                     "source", "msa_depth", "paired_msa_depth", "template_count")}})
        chains.append(entry)
    requested_pocket = {c["id"]: list(c["pocket"]) for c in request["chains"] if c.get("pocket")}
    payload = {"name": request["name"], "protocol": "complex", "rng_seeds": [int(config.get("seed", 101))],
               "chains": chains, "design_chains": request["design_chains"], "sequence_request": request,
               "source_to_native_chain": {c["id"]: [c["id"]] for c in chains},
               "native_to_source_chain": {c["id"]: c["id"] for c in chains},
               "bonded_atom_pairs": [], "user_ccd": None, "cyclic_chains": [],
               "msa_context": {"chains": contexts, "per_chain_policy": policies,
                               "resolve_msa_overlaps": bool(contexts),
                               "max_templates": 4 if any(c.get("msa_reference", {}).get("templates") for c in chains) else 0,
                               "policy": {"mode": "cache-only" if contexts else "off",
                                          "templates": any(c.get("msa_reference", {}).get("templates") for c in chains)}},
               "pocket_state": {"requested": requested_pocket, "applied": False,
                                "reason": "model does not support pocket steering" if requested_pocket else None}}
    if method == "boltz2":
        from .pocket import bind_sequence_pocket
        payload = bind_sequence_pocket(request, payload)
        if payload.get("boltz2_pocket"):
            payload["pocket_state"].update(applied=True, reason=None)
    elif method == "mol1-constraint":
        from .mol1_sites import bind_mol1_sequence_site
        payload = bind_mol1_sequence_site(request, payload)
        if payload.get("mol1_site"):
            payload["pocket_state"].update(applied=True, reason=None)
    return payload


def af3_sequence_input(payload):
    """Construct native AF3 input directly, without the source restoration ABI."""
    from src.design.rewards.af3_input import AF3Input
    return AF3Input(name=payload["name"], chains=tuple(payload["chains"]),
                    rng_seeds=tuple(payload["rng_seeds"]), bonded_atom_pairs=(), user_ccd=None,
                    source_to_native_chain={}, native_to_source_chain={}, protocol="complex",
                    msa_context=deepcopy(payload.get("msa_context", {})))


def esmfold2_sequence_payload(payload):
    """Render PipeTools' native sequence input schema, with no source segments."""
    sequences, context, entities, copies = [], [], {}, {}
    for chain in payload["chains"]:
        entry = {key: deepcopy(chain[key]) for key in ("modality", "role", "sequence", "smiles") if key in chain}
        entry["chain_id"] = chain["id"]
        identity = (chain["modality"], chain.get("sequence"), chain.get("smiles"))
        entity = entities.setdefault(identity, len(entities) + 1)
        copies[entity] = copies.get(entity, 0) + 1
        entry.update(model_entity_index=entity, model_entity_copy=copies[entity], output_entity_id=str(entity), modifications=[])
        reference = chain.get("msa_reference")
        if reference is not None:
            entry["msa"] = {key: reference[key] for key in ("unpaired_msa_path", "paired_msa_path") if reference.get(key)}
            context.append({"chain_id": chain["id"], "modality": chain["modality"], "sequence": chain["sequence"],
                            **{key: reference.get(key) for key in ("entry_id", "profile_id", "msa_depth", "source")}})
        sequences.append(entry)
    return {"schema_version": 1, "sample_id": payload["name"], "variant_id": "seq0",
            "sequences": sequences, "design_chains": list(payload["design_chains"]),
            "design_chain": payload["design_chains"][0], "covalent_bonds": [],
            "msa_policy": "context" if context else "none",
            "msa_context": {**deepcopy(payload["msa_context"]), "chains": context},
            "partidesign_sequence_request": deepcopy(payload["sequence_request"])}


__all__ = ["validate_sequence_request", "prepare_sequence_payload", "af3_sequence_input", "esmfold2_sequence_payload"]
