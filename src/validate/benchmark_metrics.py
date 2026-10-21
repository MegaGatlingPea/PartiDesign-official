"""Benchmark-specific geometry and auxiliary clash measurements (Angstrom)."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def design_target_chain_pairs(reference, design_chains):
    """Select the molecular partners of the redesigned polymer, including joint ligand."""
    from .metrics import _ordered_chains, _chain_mol_type
    polymers = [c for c in design_chains if _chain_mol_type(reference, c) in {"protein", "rna", "dna"}]
    outside = [c for c in _ordered_chains(reference) if c not in polymers
               and _chain_mol_type(reference, c) in {"protein", "rna", "dna", "ligand"}]
    # Polymer-ligand tasks may redesign both partners. Context-context pairs
    # are never candidates, and the redesigned polymer is never its own target.
    kinds = {_chain_mol_type(reference, c) for c in outside}
    preferred = None
    if "ligand" in kinds:
        if kinds & {"rna", "dna"}:
            raise ValueError("mixed NA/ligand targets require explicit benchmark partner selection")
        preferred = "ligand"
    if preferred:
        outside = [c for c in outside if _chain_mol_type(reference, c) == preferred]
    return [(left, right) for left in polymers for right in outside if left != right]


def scoped_pae(matrix, positions, reference, design_chains, *, direction="bidirectional"):
    """Take a minimum over verified partner blocks in the requested direction."""
    if direction not in {"bidirectional", "target_to_design"}:
        raise ValueError("PAE direction must be bidirectional or target_to_design")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("PAE is not a square matrix")
    pairs = design_target_chain_pairs(reference, design_chains)
    if not pairs:
        raise ValueError("no designed polymer/target molecular chain pair")
    required = list(dict.fromkeys(chain for pair in pairs for chain in pair))
    indices = {}
    for chain in required:
        values = np.asarray(positions.get(chain, ()))
        if values.size == 0:
            raise ValueError("required design/target chain has no PAE token mapping")
        if (values.ndim != 1 or values.dtype.kind not in "iu"
                or np.any(values < 0) or np.any(values >= matrix.shape[0])
                or len(np.unique(values)) != len(values)):
            raise ValueError("design/target PAE token mapping is invalid")
        indices[chain] = values.astype(np.int64, copy=False)
    mapped = np.concatenate(list(indices.values()))
    if len(np.unique(mapped)) != len(mapped):
        raise ValueError("design/target PAE token mappings overlap")
    ordered_pairs = [(right, left) for left, right in pairs]
    if direction == "bidirectional":
        ordered_pairs = [pair for left, right in pairs
                         for pair in ((left, right), (right, left))]
    blocks = [matrix[np.ix_(indices[left], indices[right])].ravel()
              for left, right in ordered_pairs]
    values = np.concatenate(blocks)
    if not np.isfinite(values).all():
        raise ValueError("design-target PAE contains nonfinite values")
    return float(np.min(values)), pairs


def _antibody_chain_groups(row: Mapping[str, object]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return validated antibody and antigen chain groups from an accepted row."""

    if not isinstance(row, Mapping) or "heavy_role" not in row:
        raise ValueError("antibody metadata is unavailable")
    role = str(row.get("heavy_role") or "").strip()
    if role not in {"VH", "VHH"}:
        raise ValueError("invalid antibody heavy_role")
    heavy = str(row.get("heavy_chain") or "").strip()
    light = str(row.get("light_chain") or "").strip()
    if not heavy:
        raise ValueError("antibody heavy chain is unavailable")
    if role == "VHH" and light:
        raise ValueError("VHH antibody cannot contain a light chain")
    if light and light == heavy:
        raise ValueError("antibody heavy and light chains overlap")
    antibody = tuple(dict.fromkeys(chain for chain in (heavy, light) if chain))
    raw_antigens = row.get("antigen_chains")
    if isinstance(raw_antigens, (list, tuple)):
        antigen = tuple(str(value).strip() for value in raw_antigens if str(value).strip())
    else:
        text = str(raw_antigens or "")
        antigen = tuple(
            part.strip() for part in text.replace(";", ",").replace("|", ",").split(",")
            if part.strip()
        )
    antigen = tuple(dict.fromkeys(antigen))
    if not antigen:
        raise ValueError("antigen chain metadata is unavailable")
    if set(antibody) & set(antigen):
        raise ValueError("antibody and antigen chain groups overlap")
    return antibody, antigen


def antibody_antigen_pae(
    matrix: np.ndarray,
    positions: Mapping[str, np.ndarray],
    reference,
    row: Mapping[str, object],
) -> float:
    """Mean PAE over every antibody/antigen residue pair in both directions.

    The antibody and antigen groups come from the validated antibody result row.
    Every residue-token block is included, so a single low pair cannot substitute
    for the full interface mean and VH:VL is never included.
    """

    from .metrics import _chain_mol_type

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("PAE is not a square matrix")
    if not isinstance(positions, Mapping):
        raise ValueError("PAE token mapping is unavailable")
    antibody, antigen = _antibody_chain_groups(row)
    for chain in antibody:
        if _chain_mol_type(reference, chain) != "protein":
            raise ValueError(f"antibody chain {chain!r} is not protein")
    supported_antigen_modalities = {"protein", "rna", "dna", "ligand"}
    for chain in antigen:
        if _chain_mol_type(reference, chain) not in supported_antigen_modalities:
            raise ValueError(f"antigen chain {chain!r} has unsupported modality")

    validated_positions: dict[str, np.ndarray] = {}
    used_tokens: set[int] = set()
    for chain in antibody + antigen:
        raw_indices = positions.get(chain)
        if raw_indices is None:
            raise ValueError(f"antibody/antigen chain {chain!r} lacks PAE token mapping")
        try:
            raw_array = np.asarray(raw_indices)
            if raw_array.ndim != 1 or not len(raw_array) or raw_array.dtype.kind == "b":
                raise ValueError
            numeric = np.asarray(raw_array, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"antibody/antigen chain {chain!r} has invalid PAE token mapping") from None
        if (
            not np.isfinite(numeric).all()
            or not np.equal(numeric, np.floor(numeric)).all()
        ):
            raise ValueError(f"antibody/antigen chain {chain!r} has invalid PAE token mapping")
        indices = numeric.astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= matrix.shape[0]):
            raise ValueError(f"antibody/antigen chain {chain!r} has invalid PAE token mapping")
        if len(np.unique(indices)) != len(indices) or used_tokens.intersection(indices.tolist()):
            raise ValueError("antibody/antigen PAE token mappings overlap")
        used_tokens.update(indices.tolist())
        validated_positions[chain] = indices

    blocks: list[np.ndarray] = []
    for antibody_chain in antibody:
        for antigen_chain in antigen:
            left = validated_positions[antibody_chain]
            right = validated_positions[antigen_chain]
            blocks.append(matrix[np.ix_(left, right)].ravel())
            blocks.append(matrix[np.ix_(right, left)].ravel())
    values = np.concatenate(blocks) if blocks else np.asarray([], dtype=np.float64)
    if not len(values):
        raise ValueError("antibody/antigen PAE has no residue pairs")
    if not np.isfinite(values).all():
        raise ValueError("antibody/antigen PAE contains nonfinite values")
    if np.any(values < 0.0):
        raise ValueError("antibody/antigen PAE contains negative values")
    return float(np.mean(values))


def _require_rigid_fit_geometry(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    label: str,
) -> None:
    """Reject fits that cannot determine a three-dimensional rigid transform."""

    from . import metrics as m

    if (
        reference.shape != predicted.shape
        or reference.ndim != 2
        or reference.shape[1:] != (3,)
        or len(reference) < 3
    ):
        raise m.MetricUnavailable(f"{label} fit is underconstrained")
    try:
        reference_rank = int(np.linalg.matrix_rank(reference - reference.mean(axis=0)))
        predicted_rank = int(np.linalg.matrix_rank(predicted - predicted.mean(axis=0)))
    except (FloatingPointError, np.linalg.LinAlgError, ValueError, TypeError) as exc:
        raise m.MetricUnavailable(f"{label} fit is numerically underconstrained") from exc
    if reference_rank < 2 or predicted_rank < 2:
        raise m.MetricUnavailable(f"{label} fit requires three noncollinear anchors")


def antigen_aligned_antibody_ca_rmsd(
    input_data,
    predicted,
    chain_map: Mapping[str, str],
    ligand_chain_map: Mapping[str, str] | None = None,
    ligand_atom_maps: Mapping[tuple[str, str], Mapping[int, int]] | None = None,
) -> float:
    """Align all declared antigen anchors, then measure antibody Cα atoms.

    Protein antigens use Cα anchors, nucleic-acid antigens use their C1' token
    centers, and ligand antigens require the authoritative heavy-atom map
    already resolved by the validation pipeline.  The optional ligand maps
    preserve the original three-argument protein-only caller contract.
    """

    from . import metrics as m

    antibody, antigen = _antibody_chain_groups(input_data.row)
    reference = input_data.atoms
    ligand_chain_map = ligand_chain_map if isinstance(ligand_chain_map, Mapping) else {}
    ligand_atom_maps = ligand_atom_maps if isinstance(ligand_atom_maps, Mapping) else {}
    for chain in antibody:
        predicted_chain = chain_map.get(chain)
        if m._chain_mol_type(reference, chain) != "protein":
            raise m.MetricUnavailable(f"antibody chain {chain!r} is not protein")
        if predicted_chain is None:
            raise m.MetricUnavailable(f"chain mapping unavailable for {chain!r}")
        if m._chain_mol_type(predicted, predicted_chain) != "protein":
            raise m.MetricUnavailable(f"predicted antibody chain mapping for {chain!r} is not protein")

    fit_refs: list[np.ndarray] = []
    fit_preds: list[np.ndarray] = []
    for chain in antigen:
        modality = m._chain_mol_type(reference, chain)
        predicted_chain = chain_map.get(chain)
        if modality in {"protein", "rna", "dna"}:
            if predicted_chain is None:
                raise m.MetricUnavailable(f"chain mapping unavailable for antigen {chain!r}")
            if m._chain_mol_type(predicted, predicted_chain) != modality:
                raise m.MetricUnavailable(f"predicted antigen chain mapping for {chain!r} has wrong modality")
            atom_names = ("CA",) if modality == "protein" else ("C1'",)
            fit_ref, fit_pred, fit_matched, fit_expected = m._paired_points(
                reference, predicted, chain_map, (chain,), atom_names
            )
        elif modality == "ligand":
            predicted_chain = ligand_chain_map.get(chain)
            mapping = ligand_atom_maps.get((chain, predicted_chain)) if predicted_chain is not None else None
            if predicted_chain is None or not isinstance(mapping, Mapping):
                raise m.MetricUnavailable(
                    f"ligand {chain!r} lacks a unique authoritative atom mapping"
                )
            if m._chain_mol_type(predicted, predicted_chain) != "ligand":
                raise m.MetricUnavailable(f"predicted antigen chain mapping for {chain!r} is not ligand")
            fit_ref, fit_pred, fit_matched, fit_expected = m._mapped_ligand_points(
                reference, predicted, chain, predicted_chain, mapping,
                allow_predicted_extra=True,
            )
        else:
            raise m.MetricUnavailable(f"antigen chain {chain!r} has unsupported modality")
        reason = m._coverage_reason(
            f"antigen {chain}", fit_ref, fit_pred, fit_matched, fit_expected
        )
        if reason:
            raise m.MetricUnavailable(reason)
        fit_refs.append(fit_ref)
        fit_preds.append(fit_pred)
    if not fit_refs:
        raise m.MetricUnavailable("antigen has no alignment anchors")
    fit_ref = np.concatenate(fit_refs, axis=0)
    fit_pred = np.concatenate(fit_preds, axis=0)
    _require_rigid_fit_geometry(fit_ref, fit_pred, label="antigen")

    measure_ref, measure_pred, measure_matched, measure_expected = m._paired_points(
        reference, predicted, chain_map, antibody, ("CA",)
    )
    reason = m._coverage_reason(
        "antibody", measure_ref, measure_pred, measure_matched, measure_expected
    )
    if reason:
        raise m.MetricUnavailable(reason)
    try:
        return m._fit_then_measure(fit_ref, fit_pred, measure_ref, measure_pred)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
        raise m.MetricUnavailable("antigen rigid fit is numerically unavailable") from exc


def complex_ca_p_rmsd(input_data, predicted, chain_map: Mapping[str, str]) -> float:
    """Jointly fit all protein CA and observed nucleic-acid P atoms.

    The reference is the actual generated input structure. ``chain_map`` is
    the evaluator's exact expected-sequence mapping, including requested
    sequence variants. Complete polymer and selected-atom coverage are
    required before one fit over the concatenated points.
    """

    from . import metrics as m

    reference = input_data.atoms
    polymer_types = {"protein", "rna", "dna"}
    protein_chains = []
    na_chains = []
    for chain in m._ordered_chains(reference):
        modality = m._chain_mol_type(reference, chain)
        if modality == "protein":
            protein_chains.append(chain)
        elif modality in {"rna", "dna"}:
            na_chains.append(chain)
    if not protein_chains or not na_chains:
        raise m.MetricUnavailable(
            "full complex CA/P metric requires both protein and nucleic-acid polymers"
        )

    polymer_chains = protein_chains + na_chains
    mapped_chains = []
    for chain in polymer_chains:
        target = chain_map.get(chain)
        if target is None:
            raise m.MetricUnavailable(f"full complex CA/P chain mapping unavailable for {chain!r}")
        if m._chain_mol_type(predicted, target) != m._chain_mol_type(reference, chain):
            raise m.MetricUnavailable(f"full complex CA/P chain {chain!r} has wrong predicted modality")
        mapped_chains.append(target)
    predicted_polymers = {
        chain for chain in m._ordered_chains(predicted)
        if m._chain_mol_type(predicted, chain) in polymer_types
    }
    if len(set(mapped_chains)) != len(mapped_chains) or set(mapped_chains) != predicted_polymers:
        raise m.MetricUnavailable("full complex CA/P polymer chain mapping is incomplete")

    refs, preds = [], []
    for chains, names in ((protein_chains, ("CA",)), (na_chains, ("P",))):
        ref, pred, matched, expected = m._paired_points(
            reference, predicted, chain_map, chains, names,
            reference_valid_only=names == ("P",),
        )
        reason = m._coverage_reason(f"full complex {names[0]}", ref, pred, matched, expected)
        if reason:
            raise m.MetricUnavailable(reason)
        refs.append(ref)
        preds.append(pred)
    return m._rmsd_with_fit(np.concatenate(refs), np.concatenate(preds))


def loose_clashes(atoms, design_chains, cutoff=1.5):
    """Count unordered design-involving heavy pairs, excluding graph distances 1 and 2."""
    from scipy.spatial import cKDTree
    from .metrics import _chain_values, _atom_element, _valid_coord
    graph = getattr(atoms, "bonds", None)
    if graph is None:
        raise ValueError("loose clash measurement requires authoritative bond graph")
    chains = _chain_values(atoms)
    heavy = np.asarray([i for i in range(len(chains)) if _atom_element(atoms, i) not in {"H", "D", "T"}], int)
    if any(not _valid_coord(atoms, int(i)) for i in heavy):
        raise ValueError("heavy-atom coordinates are incomplete")
    design = np.isin(chains, list(design_chains))
    design_count = int(design[heavy].sum())
    if design_count == 0:
        raise ValueError("no designed heavy atoms")
    neighbors = {i: set() for i in range(len(chains))}
    for a, b, _ in graph.as_array():
        neighbors[int(a)].add(int(b))
        neighbors[int(b)].add(int(a))
    excluded = set()
    for a, ns in neighbors.items():
        for b in ns | set().union(*(neighbors[n] for n in ns)):
            if a != b:
                excluded.add((min(a, b), max(a, b)))
    count = 0
    for ha, hb in cKDTree(np.asarray(atoms.coord)[heavy]).query_pairs(cutoff):
        a, b = sorted((int(heavy[ha]), int(heavy[hb])))
        if not (design[a] or design[b]) or (a, b) in excluded:
            continue
        if np.linalg.norm(atoms.coord[a] - atoms.coord[b]) < cutoff:
            count += 1
    return {"loose_clash_count": count, "loose_clash_per_1000_design_heavy": count * 1000.0 / design_count,
            "loose_clash_le2": int(count <= 2), "design_heavy_atom_count": design_count}


def extra_geometry(input_data, predicted, chain_map, ligand_chain_map, ligand_atom_maps, source_reference):
    from . import metrics as m
    values, reasons = {}, {}

    def calculate(name, callback):
        try:
            values[name] = float(callback())
        except (m.MetricUnavailable, ValueError, KeyError) as exc:
            values[name], reasons[name] = None, str(exc)

    def points(chains, names):
        ref, pred, matched, expected = m._paired_points(
            input_data.atoms, predicted, chain_map, chains, names,
            reference_valid_only=tuple(names) == ("P",))
        why = m._coverage_reason(str(names), ref, pred, matched, expected)
        if why:
            raise m.MetricUnavailable(why)
        return ref, pred

    calculate("complex_ca_p_rmsd", lambda: complex_ca_p_rmsd(input_data, predicted, chain_map))

    na_chains = [c for c in input_data.design_chains if m._chain_mol_type(input_data.atoms, c) in {"rna", "dna"}]
    if na_chains:
        calculate("na_p_self_rmsd", lambda: m._rmsd_with_fit(*points(na_chains, ("P",))))
        targets = [c for c in m._ordered_chains(input_data.atoms)
                   if c not in na_chains and m._chain_mol_type(input_data.atoms, c) == "ligand"]
        if targets:
            def ligand_placement():
                fit_ref, fit_pred = points(na_chains, ("P",))
                reference_points, predicted_points = [], []
                for chain in targets:
                    target = ligand_chain_map.get(chain)
                    mapping = ligand_atom_maps.get((chain, target))
                    if target is None or mapping is None:
                        raise m.MetricUnavailable(f"ligand {chain} lacks unique authoritative atom mapping")
                    ref, pred, matched, expected = m._mapped_ligand_points(
                        input_data.atoms, predicted, chain, target, mapping, allow_predicted_extra=True)
                    why = m._coverage_reason("ligand heavy atoms", ref, pred, matched, expected)
                    if why:
                        raise m.MetricUnavailable(why)
                    reference_points.append(ref)
                    predicted_points.append(pred)
                return m._fit_then_measure(fit_ref, fit_pred, np.concatenate(reference_points), np.concatenate(predicted_points))
            calculate("na_aligned_ligand_rmsd", ligand_placement)

    if isinstance(getattr(input_data, "row", None), Mapping) and "heavy_role" in input_data.row:
        calculate(
            "antigen_aligned_antibody_ca_rmsd",
            lambda: antigen_aligned_antibody_ca_rmsd(
                input_data,
                predicted,
                chain_map,
                ligand_chain_map=ligand_chain_map,
                ligand_atom_maps=ligand_atom_maps,
            ),
        )

    if input_data.motif_positions and source_reference is not None:
        def motif_backbone():
            ref, pred, _, missing = m._motif_pairs(input_data, predicted, chain_map,
                reference=source_reference, atom_names_override=("N", "CA", "C"))
            if missing or not len(ref):
                raise m.MetricUnavailable("incomplete source motif N/CA/C correspondence")
            return m._motif_rmsd_with_fit(ref, pred)
        calculate("motif_rmsd_reference_bb3", motif_backbone)
        def scaffold():
            refs, preds = [], []
            for chain in input_data.design_chains:
                if m._chain_mol_type(input_data.atoms, chain) != "protein":
                    continue
                target = chain_map.get(chain)
                if target is None:
                    raise m.MetricUnavailable("scaffold chain mapping unavailable")
                original = m._residue_rows(input_data.atoms, chain)
                folded = m._residue_rows(predicted, target)
                if len(original) != len(folded):
                    raise m.MetricUnavailable("scaffold residue count differs")
                motif = set(input_data.motif_positions.get(chain, ()))
                for ref_res, pred_res in zip(original, folded, strict=True):
                    if (ref_res["label_seq"], ref_res["ins_code"]) in motif:
                        continue
                    a, b = ref_res["rows"].get("CA"), pred_res["rows"].get("CA")
                    if a is None or b is None or not m._valid_coord(input_data.atoms, a) or not m._valid_coord(predicted, b):
                        raise m.MetricUnavailable("scaffold CA missing")
                    refs.append(input_data.atoms.coord[a])
                    preds.append(predicted.coord[b])
            return m._rmsd_with_fit(np.asarray(refs), np.asarray(preds))
        calculate("scaffold_ca_rmsd", scaffold)
    try:
        if any(c not in chain_map for c in input_data.design_chains):
            raise ValueError("design chain mapping incomplete for loose clashes")
        values.update(loose_clashes(predicted, tuple(chain_map[c] for c in input_data.design_chains)))
    except (ValueError, AttributeError) as exc:
        values["loose_clash_count"] = None
        reasons["loose_clash_count"] = str(exc)
    return values, reasons
