"""Antibody layout and soft framework geometry, without hard motif conditions."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..core.records import CompiledSample, DesignSampleError, ExpandedRequest, LengthPlan
from ..utils.atoms import PROTEIN_ATOM37
from .scaffolds import load_scaffold


def compile_antibody_request(
    request: ExpandedRequest, *, compile_normal: Callable[..., CompiledSample],
    ccd_path: Path | None = None, trusted_ccd_components: Any = None, rng: Any = None,
) -> CompiledSample:
    spec = request.task.antibody
    if spec is None:
        raise DesignSampleError("antibody task has no parsed antibody specification")
    if rng is None:
        rng = np.random
    scaffold = load_scaffold(spec.scaffold_path, spec.source_roles)
    source_by_role = {chain.role: chain for chain in scaffold.chains}
    entries: list[dict[str, Any]] = []
    sequence_conditions: dict[str, list[dict[str, Any]]] = {}
    actual_lengths: dict[str, int] = {}
    for design in spec.chains:
        source = source_by_role[design.role]
        index_by_source = {n: i for i, n in enumerate(source.source_positions)}
        source_region = {n: cdr for cdr, window in enumerate(source.cdr_positions, 1) for n in window}
        layout: list[tuple[int | None, int, bool]] = []
        selected = set(design.cdrs)
        chosen_fr = set(design.framework_positions)
        i = 0
        while i < len(source.source_positions):
            n = source.source_positions[i]
            cdr = source_region.get(n, 0)
            if cdr in selected:
                window = source.cdr_positions[cdr-1]
                lo, hi = design.lengths[cdr-1]
                if hasattr(rng, "integers"):
                    count = int(rng.integers(lo, hi + 1))
                else:
                    count = int(rng.randint(lo, hi + 1))
                if design.official_edits[cdr-1]:
                    excluded = set(source.excluded_positions)
                    anchors = [n for n, a, b in source.insertions if n in window]
                    if len(anchors) != 1:
                        raise DesignSampleError("official CDR layout requires one audited insertion anchor")
                    kept = [n for n in window if n not in excluded]
                    insertion_count = count - len(kept)
                    if insertion_count < 0:
                        raise DesignSampleError("official CDR final length is smaller than retained source positions")
                    for source_n in window:
                        if source_n == anchors[0]:
                            layout.extend((None, cdr + 1, True) for _ in range(insertion_count))
                        if source_n not in excluded:
                            layout.append((source_n, cdr + 1, True))
                elif count == len(window):
                    layout.extend((source_n, cdr + 1, True) for source_n in window)
                else:
                    # A custom final-length replacement has no asserted source
                    # residue correspondence. It does not invent IMGT numbering.
                    layout.extend((None, cdr + 1, True) for _ in range(count))
                i += len(window)
            else:
                layout.append((n, cdr + 1 if cdr else 1, n in chosen_fr))
                i += 1
        fixed = []
        for position, (source_position, region, mutable) in enumerate(layout, 1):
            source_index = index_by_source.get(source_position)
            amino_acid = source.sequence[source_index] if source_index is not None else None
            if not mutable:
                fixed.append({"resid": position, "residue": amino_acid})
            visible = spec.template and source_index is not None and region == 1
            mask = source.atom_mask[source_index].copy() if visible else np.zeros(37, dtype=bool)
            coords = source.coords_nm[source_index].copy() if visible else np.zeros((37, 3), dtype=np.float32)
            entries.append({
                "chain_id": design.chain_id, "role": design.role,
                "position": position, "region_id": region, "mutable": mutable,
                "source_chain": source.source_chain, "source_label_seq_id": source_position,
                "source_auth_seq_id": source.author_numbers[source_index] if source_index is not None else None,
                "source_ins_code": source.insertion_codes[source_index] if source_index is not None else None,
                "source_amino_acid": amino_acid, "template_coords_nm": coords, "template_mask": mask,
                "source_present": source.source_present[source_index] if source_index is not None else None,
            })
        if fixed:
            sequence_conditions[design.chain_id] = fixed
        actual_lengths[design.chain_id] = len(layout)
    lowered_task = replace(request.task, format="normal", antibody=spec,
        condition={"sequence": sequence_conditions})
    lowered = replace(request, task=lowered_task, plan=LengthPlan(request.plan.index, actual_lengths))
    compiled = compile_normal(lowered, ccd_path=ccd_path, trusted_ccd_components=trusted_ccd_components, rng=rng)
    compiled.request = request
    compiled.antibody_metadata = {"scaffold": scaffold.name, "source_numbering": "label_seq_id", "entries": entries,
        "boundary_unresolved": {c.role: list(c.boundary_unresolved) for c in scaffold.chains}}
    attach_candidate_annotations(compiled, compiled.atoms)
    return compiled


def attach_candidate_annotations(compiled: CompiledSample, atoms: Any) -> None:
    if not compiled.antibody_metadata:
        return
    role = np.zeros(len(atoms), dtype=np.int64)
    region = np.zeros(len(atoms), dtype=np.int64)
    mutable = np.zeros(len(atoms), dtype=bool)
    source_label = np.full(len(atoms), -1, dtype=np.int64)
    for entry in compiled.antibody_metadata["entries"]:
        rows = (atoms.chain_id == entry["chain_id"]) & (atoms.res_id == entry["position"])
        if not rows.any() or not np.all(atoms.is_design[rows]) or not np.all(atoms.mol_type[rows] == "protein"):
            raise DesignSampleError("antibody residue ledger does not resolve to one generated protein residue")
        if len(np.unique(atoms.residue_uid[rows])) != 1:
            raise DesignSampleError("antibody residue locator resolves ambiguously")
        role[rows] = 2 if entry["role"] == "VL" else 1
        region[rows] = entry["region_id"]
        mutable[rows] = entry["mutable"]
        source_label[rows] = entry["source_label_seq_id"] or -1
    for name, values in (("antibody_chain_role_id", role), ("antibody_region_id", region),
                         ("antibody_sequence_mutable", mutable), ("antibody_source_label_seq_id", source_label)):
        atoms.set_annotation(name, values)


def template_canonical_fields(compiled: CompiledSample, layout: Any, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    n = layout.num_model_tokens
    fields = {
        "design_entity_id": torch.where(sample["design_token_mask"], 0, -1).long(),
        "template_group_id": torch.zeros(n, dtype=torch.long),
        "template_protein_atom37_coords_nm": torch.zeros((n, 37, 3), dtype=torch.float32),
        "template_protein_atom37_mask": torch.zeros((n, 37), dtype=torch.bool),
        "antibody_chain_role_id": torch.zeros(n, dtype=torch.long),
        "antibody_region_id": torch.zeros(n, dtype=torch.long),
    }
    entries = {(e["chain_id"], e["position"]): e for e in compiled.antibody_metadata["entries"]}
    for token, center in enumerate(layout.model_token_to_row):
        key = str(compiled.atoms.chain_id[center]), int(compiled.atoms.res_id[center])
        entry = entries.get(key)
        if entry is None:
            continue
        fields["antibody_chain_role_id"][token] = 2 if entry["role"] == "VL" else 1
        fields["antibody_region_id"][token] = entry["region_id"]
        fields["template_protein_atom37_coords_nm"][token] = torch.from_numpy(entry["template_coords_nm"])
        fields["template_protein_atom37_mask"][token] = torch.from_numpy(entry["template_mask"])
    mask = fields["template_protein_atom37_mask"]
    coords = fields["template_protein_atom37_coords_nm"]
    fields["template_group_id"][mask.any(-1)] = 2
    # Match training's independent visible-framework frame. No hidden CDR or
    # antigen coordinate participates in its translation or rotation.
    if bool(mask.any()):
        center = coords[mask].mean(0)
        rotation = torch.eye(3)
        ni, ai, ci = [PROTEIN_ATOM37.index(a) for a in ("N", "CA", "C")]
        for token in torch.where(mask[:, ni] & mask[:, ai] & mask[:, ci])[0].tolist():
            x, y = coords[token, ci] - coords[token, ai], coords[token, ni] - coords[token, ai]
            if x.norm() < 1e-6:
                continue
            x = x / x.norm()
            y = y - (y*x).sum()*x
            if y.norm() < 1e-6:
                continue
            y = y / y.norm()
            rotation = torch.stack((x, y, torch.linalg.cross(x, y)), dim=-1)
            break
        coords[mask] = (coords[mask] - center) @ rotation
    return fields
