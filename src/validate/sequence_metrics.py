"""Native sequence records and metrics with no reference geometry."""
from __future__ import annotations
from collections.abc import Mapping
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .ipsae import ipsae_pair

_POLYMERS = frozenset({"protein", "rna", "dna"})
METRIC_UNITS = {"design_plddt": "0..1", "design_ipsae": "0..1", "binder_iptm": "0..1", "ligand_iptm": "0..1"}


def plain(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return None
        raise ValueError("Native confidence contains infinity")
    return value


def _scalar(value, *, unit=False):
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "iuf":
        return None
    result = float(array.reshape(-1)[0])
    return result if np.isfinite(result) and (not unit or 0 <= result <= 1) else None


def structure_from_cif(path):
    """Read real native output atom rows without constructing source atoms."""
    from biotite.structure.io import pdbx
    text = Path(path).read_text()
    atom = next(iter(pdbx.CIFFile.deserialize(text).values()))["atom_site"]
    def column(name):
        return atom[name].as_array().tolist()
    records = [dict(chain_id=str(chain), res_id=int(residue), res_name=str(component),
                    atom_name=str(name), atom_element=str(element))
               for name, chain, residue, component, element in zip(
                   *(column(key) for key in ("label_atom_id", "label_asym_id", "auth_seq_id", "label_comp_id", "type_symbol")), strict=True)]
    return SimpleNamespace(to_mmcif=lambda: text, iter_atoms=lambda: iter(records),
                           atom_b_factor=np.asarray(column("B_iso_or_equiv"), dtype=float))


def _pair_confidence(confidence, left, right):
    """A missing scoped field stays missing, including two-chain requests."""
    pair = confidence.get("binder_pair_iptm", confidence.get("chain_pair_iptm", confidence.get("pair_iptm")))
    if isinstance(pair, Mapping):
        row = pair.get(left)
        return _scalar(row.get(right), unit=True) if isinstance(row, Mapping) else None
    if pair is not None:
        order = list(map(str, confidence.get("chain_ids", ())))
        try:
            return _scalar(np.asarray(pair)[order.index(left), order.index(right)], unit=True)
        except (ValueError, IndexError, TypeError):
            pass
    return None


def sequence_metrics(prediction, request, pae_cutoff=10.0):
    """Calculate metrics from exact native atom/token and role evidence."""
    from .sequence_inputs import validate_sequence_request
    request = validate_sequence_request(request)
    prediction = dict(prediction)
    raw = {}
    if prediction.get("raw_confidence_path") and ("full_pae" not in prediction
            or not prediction.get("metadata", {}).get("token_ledger")):
        try:
            raw = json.loads(Path(prediction["raw_confidence_path"]).read_text())
            prediction["metadata"] = {**raw.get("metadata", {}), **prediction.get("metadata", {})}
            prediction["confidence"] = {**raw.get("confidence", {}), **prediction.get("confidence", {})}
            path = prediction.get("raw_confidence_npz_path", raw.get("arrays_path"))
            if "full_pae" not in prediction and path:
                with np.load(path, allow_pickle=False) as archive:
                    prediction["full_pae"] = archive["full_pae"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if not np.isfinite(pae_cutoff) or pae_cutoff <= 0:
        raise ValueError("ipSAE PAE cutoff must be finite and positive")
    metrics = dict.fromkeys(METRIC_UNITS)
    reasons, pairs = {}, []
    entries = {c["id"]: c for c in request["chains"]}
    design = list(request["design_chains"])
    context = [c["id"] for c in request["chains"] if c["role"] == "context"]
    confidence = prediction.get("confidence") or {}
    metadata = prediction.get("metadata") or {}
    atoms = metadata.get("atom_ledger", ())
    atom_plddt = prediction.get("_sequence_atom_plddt")
    if atom_plddt is not None:
        values = np.asarray(atom_plddt, dtype=float)
        selected = [i for i, atom in enumerate(atoms) if atom["chain_id"] in design]
        if (values.shape == (len(atoms),) and selected and np.isfinite(values[selected]).all()
                and np.all((values[selected] >= 0) & (values[selected] <= 100))):
            metrics["design_plddt"] = float(values[selected].mean()) / 100.0
    elif isinstance(confidence.get("per_chain_plddt"), Mapping):
        values, weights = [], []
        for cid in design:
            value = _scalar(confidence["per_chain_plddt"].get(cid))
            weight = confidence.get("per_chain_atom_count", {}).get(cid)
            if value is None or not 0 <= value <= 100 or not isinstance(weight, (int, float)) or weight <= 0:
                break
            values.append(value); weights.append(weight)
        else:
            if values:
                metrics["design_plddt"] = float(np.average(values, weights=weights)) / 100.0
    if metrics["design_plddt"] is None:
        reasons["design_plddt"] = "Native design atom pLDDT mapping is unavailable"
    values = [_pair_confidence(confidence, left, right) for left in design for right in context]
    if values and all(value is not None for value in values):
        metrics["binder_iptm"] = float(np.mean(values))
    else:
        reasons["binder_iptm"] = "Native scoped design/context chain-pair iPTM is unavailable"
    ligands = [c["id"] for c in request["chains"] if c["modality"] == "ligand"]
    ligand_context = [cid for cid in context if entries[cid]["modality"] in _POLYMERS]
    if len(ligands) == 1 and ligands[0] in design and ligand_context:
        values = [_pair_confidence(confidence, ligands[0], right) for right in ligand_context]
        if all(value is not None for value in values):
            metrics["ligand_iptm"] = float(np.mean(values))
    if metrics["ligand_iptm"] is None:
        reasons["ligand_iptm"] = "Native scoped single designed ligand/polymer chain-pair iPTM is unavailable"
    eligible = [(left, right) for left in design for right in context
                if entries[left]["modality"] in _POLYMERS and entries[right]["modality"] in _POLYMERS]
    matrix = prediction.get("full_pae", confidence.get("full_pae", confidence.get("pae_matrix")))
    tokens = metadata.get("token_ledger", ())
    selected, failures = {}, {}
    try:
        pae = np.asarray(matrix)
        if (pae.dtype.kind not in "iuf" or pae.ndim != 2 or pae.shape != (len(tokens), len(tokens))
                or not np.isfinite(pae).all() or np.any(pae < 0)):
            raise ValueError("Raw native PAE/token ledger is unavailable or malformed")
        for cid in {cid for pair in eligible for cid in pair}:
            by_position = {}
            for row, token in enumerate(tokens):
                if token["chain_id"] != cid:
                    continue
                position = token.get("residue_index")
                if type(position) is not int or not 1 <= position <= len(entries[cid]["sequence"]):
                    failures[cid] = f"Native polymer token position is invalid for chain {cid!r}"
                    continue
                by_position.setdefault(position, []).append(row)
            if cid in failures:
                continue
            representatives = []
            anchor = "CA" if entries[cid]["modality"] == "protein" else "C1'"
            for pos in range(1, len(entries[cid]["sequence"]) + 1):
                owned = by_position.get(pos, [])
                if not owned:
                    failures[cid] = f"Chain {cid!r} lacks native PAE token for residue {pos}"
                    break
                labels = [tokens[row].get("residue_atom_name") for row in owned]
                if all(label is None for label in labels) and len(owned) == 1:
                    representatives.extend(owned)
                else:
                    allowed = {anchor, "C1*"} if anchor == "C1'" else {anchor}
                    anchored = [row for row in owned if tokens[row].get("residue_atom_name") in allowed]
                    if len(anchored) != 1:
                        failures[cid] = f"Chain {cid!r} residue {pos} needs exactly one {anchor} token"
                        break
                    representatives.extend(anchored)
            if cid not in failures:
                selected[cid] = np.asarray(representatives, dtype=int)
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        pae = None
        failures.update({cid: str(exc) for pair in eligible for cid in pair})
    for left, right in eligible:
        row = {"design_chain": left, "context_chain": right, "ipsae": None,
               "design_to_context": None, "context_to_design": None, "status": "unavailable"}
        reason = failures.get(left) or failures.get(right)
        if reason:
            row["reason"] = reason
        else:
            lrows, rrows = selected[left], selected[right]
            score = ipsae_pair(pae[np.ix_(lrows, rrows)], pae[np.ix_(rrows, lrows)],
                               pae_cutoff=pae_cutoff,
                               d0_min=2.0 if any(entries[cid]["modality"] in {"rna", "dna"} for cid in (left, right)) else 1.0)
            row.update(ipsae=score["symmetric"], design_to_context=score["forward"],
                       context_to_design=score["reverse"], status="complete")
        pairs.append(row)
    if pairs and all(row["status"] == "complete" for row in pairs):
        metrics["design_ipsae"] = max(row["ipsae"] for row in pairs)
    else:
        reasons["design_ipsae"] = (next((row.get("reason") for row in pairs if row["status"] != "complete"), None)
                                   or "ipSAE has no eligible design/context polymer chain pair")
    return {"metrics": metrics, "metric_reasons": reasons, "metric_units": dict(METRIC_UNITS),
            "ipsae_pairs": pairs, "ipsae_pae_cutoff": float(pae_cutoff)}


def native_sequence_prediction(inference_result, *, payload, model, sample, directory,
                               atom_to_token=None, native_token_atoms_layout=None,
                               token_plddt=None, native_frame_mask=None, ranking_score=None):
    """Publish a native output directly, before source restoration or cleaning."""
    from .sequence_inputs import validate_sequence_request
    request = validate_sequence_request(payload["sequence_request"])
    entries = {chain["id"]: chain for chain in request["chains"]}
    structure = inference_result.predicted_structure
    atoms = list(structure.iter_atoms())
    metadata = dict(getattr(inference_result, "metadata", {}) or {})
    numerical = dict(getattr(inference_result, "numerical_data", {}) or {})
    chains = list(map(str, metadata.get("token_chain_ids", ())))
    residues = [int(value) for value in metadata.get("token_res_ids", ())]
    pae, pde = np.asarray(numerical.get("full_pae")), np.asarray(numerical.get("full_pde"))
    size = len(chains)
    if (not size or len(residues) != size or set(chains) != set(entries)
            or pae.shape != (size, size) or pde.shape != (size, size)
            or not np.isfinite(pae).all() or not np.isfinite(pde).all()
            or np.any(pae < 0) or np.any(pde < 0)):
        raise ValueError("Native sequence confidence differs from actual chain/token ledger")
    if set(str(atom["chain_id"]) for atom in atoms) != set(entries):
        raise ValueError("Native sequence output changed request chain identity")
    if model in {"af3", "af3-la"}:
        # AF3's matrix axes use sorted native asym IDs: structure.chains[asym-1].
        # Request order is not an authoritative mapping for this matrix.
        axis = getattr(structure, "chains", None)
        if axis is None:
            raise ValueError("Native AF3 chain-pair confidence has no chain axis")
        order = [str(chain) for chain in axis]
        pair = np.asarray(metadata.get("chain_pair_iptm"))
        if len(set(order)) != len(order) or set(order) != set(entries) or pair.shape != (len(order), len(order)):
            raise ValueError("Native AF3 chain-pair confidence differs from its chain axis")
        metadata["chain_ids"] = order
    atom_plddt = np.asarray(structure.atom_b_factor, dtype=float)
    if atom_plddt.shape != (len(atoms),) or not np.isfinite(atom_plddt).all() or np.any((atom_plddt < 0) | (atom_plddt > 100)):
        raise ValueError("Native sequence atom pLDDT is malformed")
    anchors = metadata.get("token_residue_atom_names")
    if native_token_atoms_layout is not None:
        from src.design.rewards.af3_worker import _token_residue_atom_names
        anchors = _token_residue_atom_names(native_token_atoms_layout, size)
    if anchors is not None and len(anchors) != size:
        raise ValueError("Native token residue anchors differ from PAE axis")
    if atom_to_token is not None:
        atom_to_token = np.asarray(atom_to_token, dtype=int)
        if atom_to_token.shape != (len(atoms),) or np.any(atom_to_token < 0) or np.any(atom_to_token >= size):
            raise ValueError("Native atom-to-token mapping differs from output atoms")
    if native_frame_mask is not None and np.asarray(native_frame_mask).shape != (size,):
        raise ValueError("Native sequence frame mask differs from confidence token axis")
    token_ledger = [{"token_index": i, "chain_id": cid, "role": entries[cid]["role"],
                     "modality": entries[cid]["modality"], "residue_index": residues[i],
                     "residue_atom_name": (None if anchors is None else str(anchors[i]))}
                    for i, cid in enumerate(chains)]
    atom_ledger = []
    for i, atom in enumerate(atoms):
        cid = str(atom["chain_id"])
        token = None if atom_to_token is None else int(atom_to_token[i])
        if token is None:
            candidates = [j for j, (chain, residue) in enumerate(zip(chains, residues, strict=True))
                          if cid == chain and int(atom["res_id"]) == residue]
            if native_token_atoms_layout is not None and len(candidates) > 1:
                candidates = [j for j in candidates if str(atom["atom_name"]) in np.asarray(native_token_atoms_layout[j]).tolist()]
            if len(candidates) == 1:
                token = candidates[0]
        if token is not None and chains[token] != cid:
            raise ValueError("Native atom/token chain ownership differs")
        atom_ledger.append({"atom_index": i, "chain_id": cid, "residue_index": int(atom["res_id"]),
                            "res_name": str(atom["res_name"]), "atom_name": str(atom["atom_name"]),
                            "element": str(atom.get("atom_element", atom.get("element", ""))),
                            "token_index": token, "role": entries[cid]["role"], "modality": entries[cid]["modality"]})
    confidence = plain(metadata)
    confidence.update({"plddt": float(atom_plddt.mean()), "native_plddt": float(atom_plddt.mean()),
                       "token_chain_ids": chains, "token_res_ids": residues,
                       "per_chain_plddt": {cid: float(np.mean([atom_plddt[i] for i, atom in enumerate(atom_ledger) if atom["chain_id"] == cid])) for cid in entries},
                       "per_chain_atom_count": {cid: sum(atom["chain_id"] == cid for atom in atom_ledger) for cid in entries},
                       "per_chain_token_count": {cid: chains.count(cid) for cid in entries}})
    if ranking_score is None:
        if model in {"af3", "af3-la", "mol1", "mol1-constraint"}:
            ranking_score = _scalar(metadata.get("ranking_score"))
        elif model == "boltz2":
            ranking_score = _scalar(metadata.get("confidence_score", metadata.get("boltz2_native_confidence_score")))
        else:
            iptm, ptm = _scalar(metadata.get("iptm")), _scalar(metadata.get("ptm"))
            ranking_score = None if iptm is None or ptm is None else 0.8 * iptm + 0.2 * ptm
    ranking_score = _scalar(ranking_score)
    if ranking_score is None:
        raise ValueError(f"Native {model} representative ranking score is unavailable")
    key = f"complex_sample{int(sample)}"
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{request['name']}_{key}"
    structure_path = directory / f"{stem}.cif"
    structure_path.write_text(structure.to_mmcif())
    arrays_path = directory / f"{stem}.confidence.npz"
    arrays = {"full_pae": pae, "full_pde": pde, "atom_plddt": atom_plddt,
              "token_chain_ids": np.asarray(chains), "token_res_ids": np.asarray(residues)}
    if token_plddt is not None:
        arrays["token_plddt"] = np.asarray(token_plddt)
    if atom_to_token is not None:
        arrays["atom_to_token"] = atom_to_token
    if native_frame_mask is not None:
        arrays["native_frame_mask"] = np.asarray(native_frame_mask, dtype=bool)
    if anchors is not None:
        arrays["token_residue_atom_names"] = np.asarray(anchors)
    np.savez_compressed(arrays_path, **arrays)
    metadata = plain(metadata)
    metadata.update(key=key, sample=int(sample), protocol="complex", input_kind="sequences",
                    token_ledger=token_ledger, atom_ledger=atom_ledger,
                    chain_ledger=[{"chain_id": cid, "role": entry["role"], "modality": entry["modality"]} for cid, entry in entries.items()],
                    pocket=plain(payload.get("pocket_state", {})), native_ranking_score=ranking_score)
    result = {"key": key, "model": model, "sample": int(sample), "protocol": "complex", "status": "complete",
              "structure_path": str(structure_path), "cif_path": str(structure_path), "native_cif_path": str(structure_path),
              "confidence": confidence, "metadata": metadata, "native_ranking_score": ranking_score,
              "native_plddt": confidence["plddt"], "plddt": confidence["plddt"] / 100.0,
              "ptm": confidence.get("ptm"), "iptm": confidence.get("iptm"),
              "token_chain_ids": chains, "token_res_ids": residues,
              "full_pae": pae, "full_pde": pde, "raw_confidence_npz_path": str(arrays_path),
              "msa_context": plain(payload.get("msa_context", {})), "_sequence_atom_plddt": atom_plddt}
    result.update(sequence_metrics(result, request))
    result.pop("_sequence_atom_plddt")
    metadata.update(ipsae_pairs=result["ipsae_pairs"], ipsae_pae_cutoff=result["ipsae_pae_cutoff"],
                    metric_reasons=result["metric_reasons"], metric_units=result["metric_units"])
    raw_path = directory / f"{stem}.confidence.json"
    raw_path.write_text(json.dumps({"confidence": confidence, "metadata": metadata,
                                    "arrays_path": str(arrays_path), "metrics": result["metrics"],
                                    "metric_reasons": result["metric_reasons"], "metric_units": result["metric_units"]},
                                   indent=2, allow_nan=False) + "\n")
    result["raw_confidence_path"] = str(raw_path)
    return result


__all__ = ["METRIC_UNITS", "sequence_metrics", "native_sequence_prediction", "structure_from_cif"]
