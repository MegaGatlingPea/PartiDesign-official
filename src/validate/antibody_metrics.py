"""Prediction-only antibody contacts with explicit partner and paratope identity."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping

import numpy as np

from .benchmark_metrics import _antibody_chain_groups


ANTIBODY_CONTACT_METRIC_NAMES = (
    "antibody_interface_residues",
    "antibody_off_paratope_contact_fraction",
)
_PIECES = ("fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4")


def _positions(text, length, label):
    if not isinstance(text, str):
        raise ValueError(f"{label} must be saved dense label_seq_id ranges")
    selected = set()
    for interval in text.split(",") if text else ():
        match = re.fullmatch(r"([1-9][0-9]*)(?:-([1-9][0-9]*))?", interval)
        if match is None:
            raise ValueError(f"invalid {label}")
        start, end = int(match[1]), int(match[2] or match[1])
        if end < start or end > length:
            raise ValueError(f"invalid {label}")
        values = set(range(start, end + 1))
        if selected.intersection(values):
            raise ValueError(f"invalid {label}")
        selected.update(values)
    return selected


def _entry_masks(row, residues, roles):
    raw = row.get("_antibody_context")
    if raw in (None, ""):
        return None, None
    try:
        context = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        raise ValueError("saved antibody context is invalid") from None
    if not isinstance(context, Mapping):
        raise ValueError("saved antibody context must be an object")
    entries = context.get("entries", ())
    if not isinstance(entries, (list, tuple)):
        raise ValueError("saved antibody entries are unavailable")
    if not entries:
        return None, None
    indexed = {chain: {} for chain in residues}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("saved antibody entry is invalid")
        chain = str(entry.get("output_chain_id") or entry.get("chain_id") or "")
        if chain not in indexed:
            if entry.get("role") in {"VH", "VHH", "VL"}:
                raise ValueError("saved antibody entry chain does not match the result row")
            continue
        if entry.get("role") != roles[chain]:
            raise ValueError(f"saved antibody entry role differs for {chain!r}")
        position = entry.get("position")
        if isinstance(position, bool) or not re.fullmatch(r"[1-9][0-9]*", str(position)):
            raise ValueError(f"saved antibody entry position is invalid for {chain!r}")
        position = int(position)
        if position in indexed[chain]:
            raise ValueError(f"saved antibody entries overlap for {chain!r}")
        indexed[chain][position] = entry
    for chain, groups in residues.items():
        if set(indexed[chain]) != set(range(1, len(groups) + 1)):
            raise ValueError(f"saved antibody residue mapping is incomplete for {chain!r}")
    explicit = any("paratope" in entry for chain in indexed.values() for entry in chain.values())
    masks = {}
    for chain, entries_by_position in indexed.items():
        allowed = set()
        for position, entry in entries_by_position.items():
            if explicit:
                value = entry.get("paratope")
                if not isinstance(value, (bool, np.bool_)):
                    raise ValueError(f"saved antibody paratope mask is incomplete for {chain!r}")
            else:
                # Recover the old default only from a complete saved layout.
                region = entry.get("region_id")
                if (isinstance(region, bool) or not isinstance(region, (int, np.integer))
                        or region not in (1, 2, 3, 4)):
                    raise ValueError(f"saved antibody region layout is incomplete for {chain!r}")
                value = region in (2, 3, 4)
            if value:
                allowed.add(position)
        masks[chain] = allowed
    return masks, "entry.paratope" if explicit else "legacy_entry.region_id"


def _paratope_mask(row, reference, residues):
    from .metrics import _chain_sequence

    heavy = str(row["heavy_chain"]).strip()
    light = str(row.get("light_chain") or "").strip()
    roles = {heavy: str(row["heavy_role"]).strip()}
    if light:
        roles[light] = "VL"
    sides = [("heavy", heavy)] + ([("light", light)] if light else [])
    for chain, groups in residues.items():
        for position, residue in enumerate(groups, 1):
            atom = next(iter(residue["rows"].values()))
            labels = getattr(reference, "label_seq_id", None)
            if (labels is None or str(labels[atom]) != str(position)
                    or residue["ins_code"]):
                raise ValueError(f"antibody dense residue mapping is unavailable for {chain!r}")

    entry_masks, entry_source = _entry_masks(row, residues, roles)
    public = any(f"{side}_paratope_positions" in row for side, _chain in sides)
    if public:
        masks = {}
        for side, chain in sides:
            key = f"{side}_paratope_positions"
            if key not in row:
                raise ValueError(f"saved {key} is unavailable")
            masks[chain] = _positions(row[key], len(residues[chain]), key)
        if entry_source == "entry.paratope" and masks != entry_masks:
            raise ValueError("saved public and entry antibody paratope masks disagree")
        source = ("entry.paratope+result.paratope_positions"
                  if entry_source == "entry.paratope" else "result.paratope_positions")
        return masks, source
    if entry_masks is not None:
        return entry_masks, entry_source

    # All seven saved segments and ranges must reconstruct the verified input
    # sequence and dense layout. Neither mutable nor source numbering is used.
    masks = {}
    for side, chain in sides:
        required = [f"{side}_{piece}{suffix}" for piece in _PIECES for suffix in ("", "_range")]
        if any(key not in row for key in required):
            raise ValueError(f"antibody paratope mask and saved {side} region layout are unavailable")
        sequence = "".join(str(row[key]) for key in required[::2])
        if sequence != _chain_sequence(reference, chain):
            raise ValueError(f"saved {side} region layout differs from the input sequence")
        cursor, allowed = 1, set()
        for piece in _PIECES:
            key = f"{side}_{piece}"
            part = row[key]
            if not isinstance(part, str):
                raise ValueError(f"saved {side} region layout is invalid")
            end = cursor + len(part) - 1
            expected = (str(cursor) if cursor == end else f"{cursor}-{end}") if part else ""
            if row[key + "_range"] != expected:
                raise ValueError(f"saved {side} region ranges are invalid")
            if piece.startswith("cdr"):
                allowed.update(range(cursor, end + 1))
            cursor = end + 1
        masks[chain] = allowed
    return masks, "legacy_result.cdr_ranges"


def contact_metrics(input_data, predicted, chain_map, ligand_chain_map, ligand_atom_maps,
                    *, metric_provenance=None):
    """Compute both metrics from one union of predicted heavy-atom contacts.

    ``chain_map`` and ligand maps are the validator's already verified identity
    maps. Polymer residue order is used only after that complete sequence proof.
    No prediction numbering, mutable mask, or inferred partner is used.
    """
    from .metrics import _atom_element, _chain_mol_type, _chain_values, _residue_rows

    interface, off_paratope = ANTIBODY_CONTACT_METRIC_NAMES
    values, reasons = dict.fromkeys(ANTIBODY_CONTACT_METRIC_NAMES), {}
    provenance = {"stage": "refold", "distance_angstrom_lte": 4.0,
                  "atom_selection": "valid_heavy", "counting": "unique_binder_residues"}
    reference = input_data.atoms
    try:
        antibody, antigen = _antibody_chain_groups(input_data.row)
        provenance.update(antibody_chains=list(antibody), antigen_chains=list(antigen))
        mapped = {}
        reference_residues, binder_keys = {}, []
        for chain in antibody + antigen:
            modality = _chain_mol_type(reference, chain)
            if chain in antibody and modality != "protein":
                raise ValueError(f"antibody chain {chain!r} is not protein")
            if modality in {"protein", "rna", "dna"}:
                target = chain_map.get(chain)
                if target is None or _chain_mol_type(predicted, target) != modality:
                    raise ValueError(f"verified chain mapping unavailable for {chain!r}")
                ref_groups = _residue_rows(reference, chain, polymer_only=True)
                pred_groups = _residue_rows(predicted, target, polymer_only=True)
                if not ref_groups or len(ref_groups) != len(pred_groups):
                    raise ValueError(f"verified residue mapping unavailable for {chain!r}")
                if chain in antibody:
                    reference_residues[chain] = ref_groups
                    binder_keys.extend((target, residue["uid"], chain, position)
                                       for position, residue in enumerate(pred_groups, 1))
            elif modality == "ligand" and chain in antigen:
                target = ligand_chain_map.get(chain)
                mapping = ligand_atom_maps.get((chain, target))
                if (target is None or _chain_mol_type(predicted, target) != "ligand"
                        or not isinstance(mapping, Mapping) or not mapping):
                    raise ValueError(f"antigen ligand {chain!r} lacks an authoritative atom mapping")
            else:
                raise ValueError(f"antigen chain {chain!r} has unsupported modality")
            mapped[chain] = target
        if len(set(mapped.values())) != len(mapped):
            raise ValueError("antibody/antigen prediction chain mappings overlap")
        provenance["prediction_chain_map"] = mapped

        labels = _chain_values(predicted)
        coordinates = np.asarray(predicted.coord, dtype=np.float64)
        valid = np.isfinite(coordinates).all(axis=1)
        declared_valid = getattr(predicted, "coord_valid", None)
        if declared_valid is not None:
            valid &= np.asarray(declared_valid, dtype=bool)
        valid &= np.fromiter((_atom_element(predicted, row) not in {"H", "D", "T"}
                              for row in range(len(predicted))), dtype=bool, count=len(predicted))
        identities = {(chain, uid): index for index, (chain, uid, _source, _position)
                      in enumerate(binder_keys)}
        binder_index = np.fromiter((identities.get((chain, int(uid)), -1)
                                   for chain, uid in zip(labels, predicted.residue_uid, strict=True)),
                                  dtype=np.int64, count=len(predicted))
        binder_atoms = np.flatnonzero(valid & (binder_index >= 0))
        antigen_coords = coordinates[valid & np.isin(labels, [mapped[chain] for chain in antigen])]
        contacting = np.zeros(len(binder_keys), dtype=bool)
        # Both outputs reuse this physical-residue contact mask and one union
        # of antigen atoms, with bounded vectorized distance blocks.
        for start in range(0, len(binder_atoms), 256):
            rows = binder_atoms[start:start + 256]
            delta = coordinates[rows, None, :] - antigen_coords[None, :, :]
            hits = np.any(np.einsum("ijk,ijk->ij", delta, delta) <= 16.0, axis=1)
            contacting[binder_index[rows[hits]]] = True
        count = int(np.count_nonzero(contacting))
        values[interface] = count
        provenance["contacting_binder_residue_count"] = count
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as exc:
        reasons.update({name: str(exc) for name in ANTIBODY_CONTACT_METRIC_NAMES})
        provenance["unavailable_reason"] = str(exc)
    else:
        try:
            masks, mask_source = _paratope_mask(input_data.row, reference, reference_residues)
            allowed = np.asarray([position in masks[chain] for _target, _uid, chain, position in binder_keys])
            values[off_paratope] = float(np.count_nonzero(contacting & ~allowed) / count) if count else 0.0
            provenance["paratope_source"] = mask_source
        except (AttributeError, KeyError, TypeError, ValueError, IndexError) as exc:
            reasons[off_paratope] = str(exc)
            provenance["paratope_unavailable_reason"] = str(exc)
    if metric_provenance is not None:
        for name in ANTIBODY_CONTACT_METRIC_NAMES:
            metric_provenance[name] = dict(provenance)
    return values, reasons
