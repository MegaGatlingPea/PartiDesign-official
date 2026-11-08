"""Boltz-2 inputs backed by the existing shared context MSA/template cache.

The native YAML contains no template entries: exact shared residue mappings are
applied to native template tensors after ordinary native featurization.  This
avoids Boltz's offset-only template alignment without changing model code.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from functools import lru_cache
from typing import Any


def _msa_design_chains(payload, design_chains):
    # Boltz consumes protein MSAs only. RNA/DNA never become online queries here.
    return [*map(str, design_chains), *[str(c["source_id"]) for c in payload["chains"]
                                     if c["modality"] != "protein"]]


def boltz2_msa_queries(payload, design_chains, settings):
    from src.design.rewards.af3_input import msa_queries
    return msa_queries(payload, _msa_design_chains(payload, design_chains), settings)


def attach_boltz2_references(payload, design_chains, references, *, settings):
    from src.design.rewards.af3_input import attach_msa_references
    return attach_msa_references(payload, _msa_design_chains(payload, design_chains),
                                 references, settings=settings)

def _private_ligand_view(input_record, native_variant):
    """Repair representational incompatibilities on one private free-ligand view.

    Original coordinates, chemistry arrays, author provenance and files are not
    modified. The shared matcher checks complete chemical identity. Mixed
    author locators are made coherent only for this logical ligand residue.
    """
    import copy
    from dataclasses import is_dataclass, replace
    import numpy as np
    from src.design.utils.source_mapping import canonical_chain_ids, source_locator
    from src.design.rewards.af3_chemistry import ChemistryError, source_graph, trusted_smiles, map_source_to_molecule
    atoms = getattr(input_record, "atoms", None)
    authored = native_variant.get("sequences", {}) if isinstance(native_variant, Mapping) else {}
    if atoms is None:
        return input_record, {"applied": False}
    chain_ids = canonical_chain_ids(atoms)
    modalities = np.asarray(atoms.mol_type).astype(str)
    designs = set(map(str, input_record.design_chains))
    candidates = [cid for cid in designs if np.any((chain_ids == cid) & (modalities == "ligand"))]
    if len(candidates) != 1:
        return input_record, {"applied": False}
    cid = candidates[0]
    declaration = authored.get(cid)
    if not isinstance(declaration, Mapping) or not declaration.get("smiles"):
        return input_record, {"applied": False}
    rows = np.flatnonzero((chain_ids == cid) & (modalities == "ligand"))
    logical = {(int(atoms.res_id[i]), str(atoms.ins_code[i])) for i in rows}
    if len(logical) != 1 or (hasattr(atoms, "residue_uid") and len(set(atoms.residue_uid[rows])) != 1):
        raise ValueError("Boltz-2 private ligand view requires one logical ligand residue")
    bonds = atoms.bonds.as_array().copy() if atoms.bonds is not None else np.empty((0, 3), dtype=int)
    selected = set(map(int, rows))
    if any((int(a) in selected) != (int(b) in selected) for a, b, _ in bonds):
        raise ValueError("Boltz-2 private ligand view requires a free noncovalent ligand")
    elements = np.char.upper(np.asarray(atoms.element).astype(str))
    heavy_rows = [int(i) for i in rows if elements[i] not in {"H", "D", "T"}]
    graph = source_graph(atoms, heavy_rows, preserve_aromatic_orders=True)
    trusted = trusted_smiles(str(declaration["smiles"]))
    from rdkit import Chem
    canonical = Chem.MolToSmiles(trusted, isomericSmiles=True)
    original_entry = (getattr(input_record, "ligand_entries", {}) or {}).get(cid, {})
    if original_entry.get("smiles"):
        original = trusted_smiles(str(original_entry["smiles"]))
        if Chem.MolToSmiles(original, isomericSmiles=True) != canonical:
            raise ValueError("Boltz-2 private ligand view changes authoritative SMILES identity")
    from rdkit.Chem import rdMolDescriptors
    from .ligands import is_free_design_ligand
    mapping_kind = "strict"
    try:
        map_source_to_molecule(graph, trusted)
    except ChemistryError as exc:
        if "ambiguous" not in str(exc) or not is_free_design_ligand(input_record, cid):
            raise
        map_source_to_molecule(graph, trusted, allow_symmetric=True)
        mapping_kind = "free_design_symmetric"
    chemistry = {"mapping": mapping_kind, "canonical_isomeric_smiles": canonical,
                 "formula": rdMolDescriptors.CalcMolFormula(trusted),
                 "complete_heavy_graph_and_explicit_charges_match": True}
    original_locators = [{"row": int(i), "atom_name": str(atoms.atom_name[i]), **source_locator(atoms, int(i))} for i in rows]
    logical_residue, logical_insertion = next(iter(logical))
    replacements = {"auth_asym_id": cid, "auth_seq_id": str(logical_residue), "auth_ins_code": logical_insertion}
    mixed = [field for field in replacements if len({item[field] for item in original_locators}) > 1]
    if not mixed:
        return input_record, {"applied": False, "chemistry": chemistry}
    private = atoms.copy()
    for field in mixed:
        previous = np.asarray(getattr(atoms, field)).astype(str)
        values = np.asarray([replacements[field] if i in selected else value for i, value in enumerate(previous)], dtype=str)
        private.set_annotation(field, values)
    if is_dataclass(input_record):
        conditioned = replace(input_record, atoms=private)
    elif isinstance(input_record, Mapping):
        conditioned = dict(input_record, atoms=private)
    else:
        conditioned = copy.copy(input_record)
        conditioned.atoms = private
    evidence = {"applied": True, "source_chain_id": cid, "source_rows": list(map(int, rows)),
                "chemistry": chemistry, "aromatic_bonds_normalized": [],
                "author_locator_fields_normalized": mixed,
                "original_per_atom_locators": original_locators,
                "original_coordinates_and_arrays_preserved": True}
    return conditioned, evidence


@lru_cache(maxsize=8)
def _canonical_boltz_ccd(root_text):
    from partidesign.boltz_assets import ensure_boltz_mols
    from boltz.data.mol import load_canonicals
    from .boltz2 import _BoltzCCD
    root = Path(root_text)
    moldir = ensure_boltz_mols(root)
    molecules = load_canonicals(moldir)
    provider = _BoltzCCD(molecules)
    inventory = {name: provider.get(name) for name in molecules}
    return inventory, {"source": "boltz", "root": str(root), "moldir": str(moldir),
                       "canonical_components": sorted(inventory), "AF3_asset_fallback": False}


def _boltz_ccd_inventory(repo_root):
    """Resolve authoritative canonical chemistry from the selected Boltz assets."""
    from partidesign.paths import community_asset
    from .boltz2 import _source_imports
    _source_imports(repo_root)
    return _canonical_boltz_ccd(str(community_asset("boltz", repo_root).resolve()))


def _canonical_payload(input_record, variant, config, repo_root):
    from src.design.rewards.af3_backend import _variant_with_ligand_smiles
    from src.design.rewards.af3_input import build_af3_input
    native_variant = _variant_with_ligand_smiles(input_record, variant)
    conditioned, compatibility = _private_ligand_view(input_record, native_variant)
    ccd, inventory_evidence = _boltz_ccd_inventory(repo_root)
    payload = build_af3_input(conditioned, native_variant, seed=config["seed"],
                              protocol="complex", ccd=ccd).payload()
    payload["boltz2_private_input_compatibility"] = compatibility
    payload["boltz2_ccd_inventory"] = inventory_evidence
    # AF3 materializes explicit SMILES as custom CCDs. Retain their original
    # chemistry for Boltz's SMILES parser; keep AF3 ledgers for common restoration.
    authored = native_variant.get("sequences", {}) if isinstance(native_variant, Mapping) else {}
    for chain in payload["chains"]:
        value = authored.get(str(chain["source_id"]))
        if chain["modality"] == "ligand" and isinstance(value, Mapping) and value.get("smiles"):
            chain["smiles"] = str(value["smiles"])
    payload["name"] = "boltz2" + str(payload["name"])[3:] if str(payload["name"]).startswith("af3") else str(payload["name"])
    return payload


def prepare_boltz2_inputs(input_record, variant, config, repo_root, *, cache=None, pocket_cache=None):
    """Prepare one complex through native chemistry and shared CPU-only searches."""
    if config.get("mode", "complex") != "complex":
        raise ValueError("Boltz-2 validation currently supports complex mode only")
    payload = _canonical_payload(input_record, variant, config, repo_root)
    from .pocket import bind_source_pocket
    payload = bind_source_pocket(input_record, payload, repo_root=repo_root, cache=pocket_cache)
    settings = config["msa"]
    queries = {(q["modality"], q["sequence"])
               for q in boltz2_msa_queries(payload, input_record.design_chains, settings)}
    references = []
    if queries:
        from .af3_msa import AF3MSACache
        cache = cache or AF3MSACache(repo_root, settings)
        references = [cache.ensure(modality, sequence) for modality, sequence in sorted(queries)]
    prepared = attach_boltz2_references(payload, input_record.design_chains, references, settings=settings)
    return {"complex": prepared}


def _connected_smiles(smiles):
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("Boltz-2 requires an authoritative connected ligand SMILES")


def _reject_ligand_links(payload, ligand_ids):
    source_ids = {str(c["source_id"]) for c in payload["chains"] if str(c["id"]) in ligand_ids}
    for left, right in payload.get("bonded_atom_pairs", ()):
        if str(left[0]) in ligand_ids or str(right[0]) in ligand_ids:
            raise ValueError("Boltz-2 affinity does not support covalently attached ligands")
    ledger = {int(row["row"]): row for row in payload.get("source_atom_ledger", ())}
    for a, b, _ in payload.get("source_inter_residue_bonds", ()):
        left, right = ledger.get(int(a)), ledger.get(int(b))
        if left is None or right is None:
            raise ValueError("Boltz-2 source bond has no authoritative atom ledger")
        lhs = str(left["chain_id"]) in source_ids
        rhs = str(right["chain_id"]) in source_ids
        if lhs != rhs:
            raise ValueError("Boltz-2 affinity does not support covalently attached ligands")



def _reject_unsupported_links(payload):
    if payload.get("cyclic_chains"):
        raise ValueError("Boltz-2 validation does not yet support cyclic polymers")
    if payload.get("bonded_atom_pairs"):
        raise ValueError("Boltz-2 validation does not yet support explicit covalent links")
    ledger = {int(row["row"]): row for row in payload.get("source_atom_ledger", ())}
    positions = {}
    for native_id, residues in payload.get("source_residue_map", {}).items():
        for pos, residue in enumerate(residues):
            if residue is not None:
                key = (str(residue["source_chain_id"]), int(residue["source_res_id"]),
                       str(residue.get("source_ins_code", "")))
                positions[key] = (str(native_id), pos)
    for a, b, order in payload.get("source_inter_residue_bonds", ()):
        left, right = ledger.get(int(a)), ledger.get(int(b))
        if left is None or right is None:
            raise ValueError("Boltz-2 source bond has no authoritative atom ledger")
        key = lambda row: (str(row["chain_id"]), int(row["res_id"]), str(row.get("ins_code", "")))
        lk, rk = key(left), key(right)
        if lk == rk:
            continue  # Complete residue-internal chemistry belongs to CCD/SMILES.
        lp, rp = positions.get(lk), positions.get(rk)
        normal = (left.get("mol_type") == right.get("mol_type") == "protein"
                  and lp is not None and rp is not None and lp[0] == rp[0]
                  and int(order) in {0, 1}
                  and ((left["atom_name"], right["atom_name"], rp[1] - lp[1]) == ("C", "N", 1)
                       or (right["atom_name"], left["atom_name"], lp[1] - rp[1]) == ("C", "N", 1)))
        if not normal:
            raise ValueError("Boltz-2 validation does not yet support nonstandard polymer or component links")


def boltz2_job(payload, directory, *, affinity=True):
    """Render explicit local MSA inputs; never request native Boltz searches.

    ``directory`` is accepted for the common adapter interface. No input pose or
    templates are written: cached template features are attached separately.
    """
    del directory
    from src.design.rewards.af3_input import _validate_msa_reference
    sequences, seen, protein_msas = [], set(), {}
    chains = payload["chains"]
    design_ligands = [str(c["id"]) for c in chains
                      if c["modality"] == "ligand" and c.get("role") == "design"]
    if affinity:
        if len(design_ligands) != 1:
            raise ValueError("Boltz-2 affinity requires one unambiguous designed ligand")
        if not any(c["modality"] == "protein" and c.get("role") != "design" for c in chains):
            raise ValueError("Boltz-2 ligand affinity requires protein context")
        if any(c["modality"] in {"rna", "dna"} for c in chains):
            raise ValueError("Boltz-2 ligand affinity currently supports protein context only")
        _reject_ligand_links(payload, set(design_ligands))
    policy = payload.get("msa_context", {}).get("policy", {})
    for chain in chains:
        cid, modality = str(chain["id"]), str(chain["modality"])
        if not cid or cid in seen:
            raise ValueError("Boltz-2 native chain IDs must be unique and nonempty")
        seen.add(cid)
        entity = {"id": cid}
        if modality in {"protein", "rna", "dna"}:
            if chain.get("modifications"):
                raise ValueError("Boltz-2 validation does not yet support modified polymers")
            entity["sequence"] = str(chain["sequence"])
            if modality == "protein":
                reference = chain.get("msa_reference")
                if reference is not None:
                    if chain.get("role") == "design":
                        raise ValueError("Boltz-2 designed proteins must remain query-only")
                    _validate_msa_reference(chain, reference)
                    entity["msa"] = str(Path(reference["unpaired_msa_path"]).resolve())
                else:
                    if policy.get("mode") in {"auto", "cache-only"} and chain.get("role") != "design":
                        raise ValueError(f"Missing shared MSA reference for context chain {cid}")
                    entity["msa"] = "empty"
                previous = protein_msas.setdefault(entity["sequence"], entity["msa"])
                if previous != entity["msa"]:
                    raise ValueError("Boltz-2 cannot give identical protein sequences different MSAs; context/design query-only policy conflicts")
        elif modality == "ligand":
            if chain.get("smiles"):
                _connected_smiles(str(chain["smiles"]))
                entity["smiles"] = str(chain["smiles"])
            elif chain.get("ccd_ids"):
                ids = list(chain["ccd_ids"])
                if affinity and cid in design_ligands and len(ids) != 1:
                    raise ValueError("Boltz-2 affinity requires one connected ligand component")
                entity["ccd"] = ids[0] if len(ids) == 1 else ids
            else:
                raise ValueError(f"Boltz-2 ligand {cid} has no authoritative chemistry")
        else:
            raise ValueError(f"Unsupported Boltz-2 chain modality: {modality}")
        sequences.append({modality: entity})
    _reject_unsupported_links(payload)
    job = {"version": 1, "sequences": sequences}
    from .pocket import native_pocket_constraints
    constraints = native_pocket_constraints(payload)
    if constraints:
        job["constraints"] = constraints
    if affinity:
        job["properties"] = [{"affinity": {"binder": design_ligands[0]}}]
    return job


def _mapping(template, query_length):
    from src.design.rewards.af3_input import _validate_template_reference
    _validate_template_reference(template, query_length)
    result = {}
    for q, t in template["query_to_template_map"].items():
        if isinstance(q, bool) or isinstance(t, bool):
            raise ValueError("Boltz-2 template indices must be integers")
        qi, ti = int(q), int(t)
        if str(qi) != str(q) or str(ti) != str(t):
            raise ValueError("Boltz-2 template indices must be canonical integers")
        result[qi] = ti
    if not result or len(set(result.values())) != len(result):
        raise ValueError("Boltz-2 template mapping must be nonempty and one-to-one")
    return result


def _template_rows(tokenized, payload, ccd, moldir):
    from boltz.data import const
    from boltz.data.parse.mmcif import parse_mmcif
    from boltz.data.tokenize.boltz2 import tokenize_structure
    policy = payload.get("msa_context", {}).get("policy", {})
    if policy.get("templates") is False or policy.get("mode") == "off":
        return []
    chain_to_asym = {str(c["name"]): int(c["asym_id"]) for c in tokenized.structure.chains}
    parsed_files, rows = {}, []
    for chain in payload["chains"]:
        templates = chain.get("msa_reference", {}).get("templates", ())
        if not templates:
            continue
        if chain["modality"] != "protein" or chain.get("role") == "design":
            raise ValueError("Boltz-2 shared templates require fixed protein context")
        cid = str(chain["id"])
        if cid not in chain_to_asym:
            raise ValueError(f"Boltz-2 template query chain is absent: {cid}")
        # Local array positions are essential after AffinityCropper: token_idx
        # retains the original index, while native tensor axes are shortened.
        q_indices = {int(t["res_idx"]): i for i, t in enumerate(tokenized.tokens)
                     if int(t["asym_id"]) == chain_to_asym[cid]}
        for template in templates:
            mapping = _mapping(template, len(chain["sequence"]))
            path = str(Path(template["mmcif_path"]).resolve())
            if path not in parsed_files:
                parsed = parse_mmcif(path, mols=ccd, moldir=moldir,
                                     use_assembly=False, compute_interfaces=False)
                proteins = [c for c in parsed.data.chains
                            if int(c["mol_type"]) == const.chain_type_ids["PROTEIN"]]
                if len(proteins) != 1:
                    raise ValueError("Shared Boltz-2 template must contain one protein chain")
                protein = proteins[0]
                template_chain = str(protein["name"])
                sequence = parsed.sequences[template_chain]
                if max(mapping.values()) >= len(sequence):
                    raise ValueError("Shared template index exceeds native full SEQRES")
                tokens, _ = tokenize_structure(parsed.data)
                by_residue = {int(t["res_idx"]): t for t in tokens
                              if int(t["asym_id"]) == int(protein["asym_id"])}
                if len(by_residue) != len(sequence):
                    raise ValueError("Native Boltz-2 template lost the full SEQRES axis")
                parsed_files[path] = by_residue
            by_residue = parsed_files[path]
            native_rows = []
            for q, t in sorted(mapping.items()):
                if t not in by_residue:
                    raise ValueError("Shared template index is absent from native residue axis")
                if q in q_indices:
                    native_rows.append({"token": by_residue[t], "q_idx": q_indices[q], "pdb_id": len(rows)})
            if native_rows:
                # A cached template is a single-chain structure; separate rows
                # avoid inventing inter-chain template geometry for its copies.
                rows.append(native_rows)
    return rows


def shared_template_features(tokenized, payload, ccd, moldir, max_tokens=None):
    """Convert exact shared maps to native template features, without re-alignment."""
    import torch
    from boltz.data.feature.featurizerv2 import compute_template_features
    count = len(tokenized.tokens) if max_tokens is None else int(max_tokens)
    if count < len(tokenized.tokens):
        raise ValueError("Boltz-2 template tensor length is shorter than tokenized input")
    rows = _template_rows(tokenized, payload, ccd, moldir)
    if not rows:
        return None
    results = []
    for row in rows:
        features = compute_template_features(tokenized, row, count)
        features["template_force"] = torch.tensor(False)
        features["template_force_threshold"] = torch.tensor(float("inf"), dtype=torch.float32)
        results.append(features)
    return {key: torch.stack([result[key] for result in results]) for key in results[0]}


def input_evidence(tokenized, features, payload):
    """Report actual native MSA/template axes independently from cache presence."""
    from .pocket import pocket_feature_evidence
    import numpy as np
    from boltz.data import const
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    by_chain = {str(c["name"]): int(c["asym_id"]) for c in tokenized.structure.chains}
    msa_features = array(features["msa"]) if "msa" in features else None
    template_masks = {name: array(features[name]) for name in
                      ("template_mask", "template_mask_cb", "template_mask_frame") if name in features}
    chains = []
    for chain in payload["chains"]:
        cid = str(chain["id"])
        asym = by_chain[cid]
        indices = [i for i, t in enumerate(tokenized.tokens) if int(t["asym_id"]) == asym]
        msa = tokenized.msa.get(asym, tokenized.msa.get(str(asym)))
        reference = chain.get("msa_reference")
        entry = {"native_id": cid, "source_id": str(chain["source_id"]),
                 "role": chain.get("role"), "modality": chain["modality"],
                 "native_msa_depth": len(msa.sequences) if msa is not None else 1,
                 "query_only": msa is None or len(msa.sequences) <= 1,
                 "tokens": len(indices), "shared_reference": reference}
        if msa_features is not None:
            entry["feature_nonquery_residues"] = int(np.count_nonzero(msa_features[1:, indices] != const.token_ids["-"]))
        entry["template_counts"] = {name: int(np.count_nonzero(value[..., indices]))
                                     for name, value in template_masks.items()}
        chains.append(entry)
    return {"chains": chains, "msa_feature_shape": list(msa_features.shape) if msa_features is not None else None,
            "shared_paired_msa_file_used": False,
            "template_feature_shapes": {name: list(value.shape) for name, value in template_masks.items()},
            "msa_context": payload.get("msa_context", {}),
            "private_input_compatibility": payload.get("boltz2_private_input_compatibility", {"applied": False}),
            "chemistry_inventory": payload.get("boltz2_ccd_inventory"),
            "pocket_conditioning": pocket_feature_evidence(tokenized, features, payload)}
