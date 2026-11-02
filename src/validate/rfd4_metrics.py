"""Pinned RFD4 metric selections without RFD4/AtomWorks dependencies.

Primary source: RFD4-Proteina ddd934f7e6b86f7582ce487d5b19f6c526c644b5,
metrics/{oracles/oracle_metric.py,gt_structure_comparison.py,design_metrics.py}
and configs/metrics/refolding_metrics/all_refolding_metrics.yaml. By explicit
PartiDesign policy the clash selection measures the REFOLD structure.
For authorized canonical sequence changes, target-aligned RMSDs use a
PartiDesign comparable-atoms extension: unchanged residues retain observed
atoms, changed proteins use backbone atoms and changed NA uses its framework.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

RFD4_METRIC_NAMES = (
    "rfd4_min_ipae_angstrom", "rfd4_backbone_rmsd",
    "rfd4_target_aligned_binder_rmsd", "rfd4_target_backbone_aligned_binder_rmsd",
    "rfd4_backbone_aligned_ligand_rmsd", "rfd4_motif_rmsd_reference",
    "rfd4_has_ligand_clashes",
)
_BACKBONE = frozenset(("N", "CA", "C", "O", "OXT"))
_NA_FRAMEWORK = frozenset(("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'",
                           "C3'", "O3'", "C2'", "O2'", "C1'"))
_POLYMER = frozenset(("protein", "rna", "dna"))


def _partners(reference, design_chains, row):
    from . import metrics as m
    from .benchmark_metrics import _antibody_chain_groups, design_target_chain_pairs
    if "heavy_role" in row:
        binder, target = _antibody_chain_groups(row)
        if any(m._chain_mol_type(reference, c) != "protein" for c in binder):
            raise ValueError("RFD4 antibody chains must be protein")
        if any(m._chain_mol_type(reference, c) not in _POLYMER | {"ligand"} for c in target):
            raise ValueError("RFD4 antigen chain has unsupported modality")
        return [(a, b) for a in binder for b in target]
    design = set(design_chains)
    pairs = [(a, b) for a, b in design_target_chain_pairs(reference, design_chains)
             if a in design and b not in design]
    if not pairs:
        raise ValueError("RFD4 PAE has no designed polymer/target chain pair")
    return pairs


def _bool_mask(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind != "b":
        raise ValueError(f"RFD4 {name} must be a boolean mask on the actual token axis")
    return array


def _metadata(prediction, payload, *names):
    for name in names:
        if prediction.get(name) is not None:
            return prediction[name]
        if payload.get(name) is not None:
            return payload[name]
    return None


def _native_summary_pae(prediction, payload, pairs):
    """Only genuine native minima, never a saved validation scalar."""
    native = prediction.get("_native_confidence")
    summary = native.get("metadata") if isinstance(native, Mapping) else None
    if not isinstance(summary, Mapping) or "chain_pair_pae_min" not in summary:
        raise ValueError("RFD4 PAE has no full matrix or authoritative native chain-pair minima")
    chains = _metadata(prediction, payload, "native_chain_ids")
    mapping = _metadata(prediction, payload, "native_to_source_chain")
    if chains is None or not isinstance(mapping, Mapping):
        raise ValueError("RFD4 native minima lack exact source-chain mapping")
    chains = tuple(str(c) for c in chains)
    if not chains or len(set(chains)) != len(chains):
        raise ValueError("RFD4 native summary chain order is invalid")
    matrix = np.asarray(summary["chain_pair_pae_min"], dtype=object)
    if matrix.shape != (len(chains), len(chains)):
        raise ValueError("RFD4 native summary matrix dimensions differ")
    labels = [str(mapping.get(c, "")) for c in chains]
    if any(not c for c in labels):
        raise ValueError("RFD4 native summary chain mapping is incomplete")
    values = []
    for a, b in pairs:
        directions = []
        for left, right in ((a, b), (b, a)):
            rows = [i for i, c in enumerate(labels) if c == left]
            cols = [i for i, c in enumerate(labels) if c == right]
            if not rows or not cols:
                raise ValueError("RFD4 native summary lacks a required partner chain")
            observed = []
            for value in matrix[np.ix_(rows, cols)].ravel():
                if value is None:
                    continue
                number = float(value)
                if math.isnan(number):
                    continue
                if not math.isfinite(number) or number < 0:
                    raise ValueError("RFD4 native minima contain invalid values")
                observed.append(number)
            if observed:
                directions.append(min(observed))
        if directions:
            values.append(float(np.mean(directions)))
    if not values:
        raise ValueError("RFD4 native minima have no valid frame direction")
    return min(values)


def min_ipae(reference, predicted, chain_map, design_chains, row, prediction, payload):
    """Mean valid directional minima per partner pair, then min across pairs.

    The first PAE axis supplies the alignment frame. Frame validity restricts
    rows; token validity restricts both axes. All actual tokens are retained.
    """
    from . import metrics as m
    pairs = _partners(reference, design_chains, row)
    native = _metadata(prediction, payload, "native_full_pae", "native_pae_matrix")
    if native is not None:
        matrix = np.asarray(native, dtype=np.float64)
        labels = _metadata(prediction, payload, "native_token_chain_ids")
        if labels is None:
            raise ValueError("RFD4 full native PAE has no exact logical token chain labels")
        labels = np.asarray(labels).astype(str)
        mapping = _metadata(prediction, payload, "native_to_source_chain")
        if isinstance(mapping, Mapping):
            labels = np.asarray([str(mapping.get(c, "")) for c in labels], dtype=str)
        frame = _metadata(prediction, payload, "native_frame_mask", "native_token_has_frame", "native_frames_mask")
        if frame is None:
            private = prediction.get("_native_confidence")
            if isinstance(private, Mapping):
                frame = private.get("frames_mask")
        if frame is None and _metadata(prediction, payload, "native_full_pae") is not None:
            value = _native_summary_pae(prediction, payload, pairs)
            return value, {"source": "native_chain_pair_pae_min", "pairs": [list(p) for p in pairs]}
        attention = _metadata(prediction, payload, "native_token_attention_mask", "native_token_mask")
        pair_mask = _metadata(prediction, payload, "native_pae_mask", "native_pair_mask")
        source = "full_native_token_axis"
    else:
        raw = m._lookup(payload, "pae_matrix", "predicted_aligned_error", "predicted_aligned_error_matrix", "pae")
        if raw is None or np.asarray(raw).ndim != 2:
            value = _native_summary_pae(prediction, payload, pairs)
            return value, {"source": "native_chain_pair_pae_min", "pairs": [list(p) for p in pairs]}
        matrix = np.asarray(raw, dtype=np.float64)
        labels, reason = m._resolve_token_chain_labels(payload, prediction, predicted, matrix.shape[0])
        if labels is None:
            # Only AF2 guarantees that PDB residue order is its actual PAE axis.
            method = str(prediction.get("method", prediction.get("model", ""))).lower()
            if "af2" not in method:
                raise ValueError(reason or "RFD4 PAE lacks exact token chain labels")
            labels = np.asarray([c for c in m._ordered_chains(predicted)
                                 for _ in m._residue_rows(predicted, c)], dtype=str)
        inverse = {str(p): str(r) for r, p in chain_map.items()}
        declared = prediction.get("_confidence_chain_map", {})
        if isinstance(declared, Mapping):
            for r, p in declared.items():
                if str(r) in m._ordered_chains(reference) and str(p) not in inverse:
                    inverse[str(p)] = str(r)
        labels = np.asarray([inverse.get(str(c), "") for c in labels], dtype=str)
        frame = _metadata(prediction, payload, "token_has_frame", "token_frame_mask", "pae_frame_mask", "frames_mask")
        attention = _metadata(prediction, payload, "token_attention_mask", "token_mask")
        pair_mask = _metadata(prediction, payload, "pae_mask", "pair_mask")
        source = "actual_token_axis"
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("RFD4 PAE must be a nonempty square matrix")
    size = matrix.shape[0]
    if labels.shape != (size,):
        raise ValueError("RFD4 PAE labels do not cover its actual token axis")
    valid = np.ones(size, dtype=bool) if attention is None else _bool_mask(attention, (size,), "token validity")
    if np.any((labels == "") & valid):
        raise ValueError("RFD4 PAE has an unmapped valid native token")
    frames = np.ones(size, dtype=bool) if frame is None else _bool_mask(frame, (size,), "frame validity")
    eligible = None if pair_mask is None else _bool_mask(pair_mask, matrix.shape, "pair validity")
    values, details = [], []
    for a, b in pairs:
        if not np.any((labels == a) & valid) or not np.any((labels == b) & valid):
            raise ValueError("RFD4 PAE lacks a required partner's valid tokens")
        directions = []
        for left, right in ((a, b), (b, a)):
            rows = np.flatnonzero((labels == left) & valid & frames)
            cols = np.flatnonzero((labels == right) & valid)
            block = matrix[np.ix_(rows, cols)]
            selected = block.ravel() if eligible is None else block[eligible[np.ix_(rows, cols)]]
            if not len(selected):
                continue
            if not np.isfinite(selected).all() or np.any(selected < 0):
                raise ValueError("RFD4 PAE has nonfinite or negative values in a valid partner block")
            directions.append(float(np.min(selected)))
        if directions:
            value = float(np.mean(directions))
            values.append(value)
            details.append({"pair": [a, b], "directional_minima": directions, "mean": value})
    if not values:
        raise ValueError("RFD4 PAE has no valid partner/frame direction")
    return min(values), {"source": source, "frame_mask_supplied": frame is not None, "pairs": details}


def confidence_metrics(metrics, reasons, provenance, reference, predicted, chain_map,
                       design_chains, row, prediction, payload):
    names = ("rfd4_min_ipae_angstrom", "min_interchain_pae_angstrom")
    try:
        value, evidence = min_ipae(reference, predicted, chain_map, design_chains, row, prediction, payload)
    except (ValueError, TypeError, IndexError, KeyError, OverflowError) as exc:
        for name in names:
            metrics[name], reasons[name] = None, str(exc)
    else:
        for name in names:
            metrics[name] = value
            reasons.pop(name, None)
            provenance[name] = evidence


def _mutable_design_residues(input_data):
    """Recover identity permission from the public input, without fixing coordinates.

    Output structures do not restore the compiler's identity_condition mask.
    Fixed/motif locators and chemical anchors therefore remain authoritative
    even when the refold's full sequence matches the requested variant.
    """
    from . import metrics as m
    from .inputs import _locator_rows
    reference = input_data.atoms
    labels = m._chain_values(reference)
    residue_uids = np.asarray(reference.residue_uid, dtype=np.int64)
    mutable = set(residue_uids[np.isin(labels, input_data.design_chains)].tolist())

    def protect(locator):
        rows = _locator_rows(reference, locator)
        if not rows:
            raise m.MetricUnavailable("RFD4 fixed identity locator is unavailable")
        mutable.difference_update(residue_uids[rows].tolist())

    for locator in input_data.fixed_restypes:
        protect(locator)
    for chain, locators in input_data.fixed_positions.items():
        for locator in locators:
            protect({**locator, "chain": chain})
    for chain, positions in input_data.motif_positions.items():
        residues = m._residue_rows(reference, chain)
        for position, insertion in positions:
            residue = m._residue_at_position(residues, position, insertion)
            if residue is None:
                raise m.MetricUnavailable("RFD4 motif identity locator is unavailable")
            mutable.discard(int(residue["uid"]))
    if "identity_condition" in reference.get_annotation_categories():
        identity = _bool_mask(reference.identity_condition, (len(reference),), "identity condition")
        mutable.difference_update(residue_uids[identity].tolist())

    def protect_anchor(row):
        kind, name = str(reference.mol_type[row]), str(reference.atom_name[row])
        # Match sequence planning: protein backbone links do not force an
        # identity; every declared nucleic-acid topology endpoint does.
        if kind in {"rna", "dna"} or (kind == "protein" and name not in _BACKBONE):
            mutable.discard(int(residue_uids[row]))

    for bond in input_data.bond_constraints:
        for endpoint in ("atom1", "atom2"):
            rows = _locator_rows(reference, bond[endpoint])
            if len(rows) != 1:
                raise m.MetricUnavailable("RFD4 chemical anchor locator is unavailable")
            protect_anchor(rows[0])
    bonds = getattr(reference, "bonds", None)
    if bonds is not None:
        ordinals = {int(item["uid"]): ordinal
                    for chain in m._ordered_chains(reference)
                    for ordinal, item in enumerate(m._residue_rows(reference, chain))}
        for left, right, _ in bonds.as_array().tolist():
            left, right = int(left), int(right)
            if residue_uids[left] == residue_uids[right]:
                continue  # Template atoms inside one residue are not anchors.
            if (labels[left] == labels[right]
                    and str(reference.mol_type[left]) == str(reference.mol_type[right])
                    and str(reference.mol_type[left]) in {"rna", "dna"}
                    and abs(ordinals[int(residue_uids[left])] - ordinals[int(residue_uids[right])]) == 1
                    and {str(reference.atom_name[left]), str(reference.atom_name[right])} == {"O3'", "P"}):
                continue  # The ordinary polymer connection is not a topology condition.
            protect_anchor(left)
            protect_anchor(right)
    return mutable


def _polymer_rows(reference, predicted, chain_map, chains, *, backbone=False,
                  mutable_residues=None, permission_error=None, optional_terminal_oxt=()):
    from . import metrics as m
    from ..design.utils.atoms import PROTEIN_RES_ATOMS, RNA_RES_ATOMS, DNA_RES_ATOMS
    standard_atoms = {"protein": PROTEIN_RES_ATOMS, "rna": RNA_RES_ATOMS, "dna": DNA_RES_ATOMS}
    rows = []
    for chain in chains:
        target = chain_map.get(chain)
        if target is None:
            raise m.MetricUnavailable(f"RFD4 chain {chain!r} lacks verified correspondence")
        original = m._residue_rows(reference, chain, polymer_only=True)
        refolded = m._residue_rows(predicted, target, polymer_only=True)
        if len(original) != len(refolded) or not original:
            raise m.MetricUnavailable("RFD4 polymer residue correspondence is incomplete")
        kind = m._chain_mol_type(reference, chain)
        for a, b in zip(original, refolded, strict=True):
            changed = str(a["res_name"]) != str(b["res_name"])
            conserved = _BACKBONE if kind == "protein" else _NA_FRAMEWORK
            if changed and (not backbone or mutable_residues is not None):
                if permission_error is not None:
                    raise m.MetricUnavailable(permission_error)
                canonical = standard_atoms.get(kind, {})
                if (str(a["res_name"]) not in canonical or str(b["res_name"]) not in canonical
                        or str(a["res_name"]) in {"UNK", "N", "DN"}
                        or str(b["res_name"]) in {"UNK", "N", "DN"}
                        or m._chain_mol_type(predicted, target) != kind):
                    raise m.MetricUnavailable("RFD4 comparable atoms require canonical residues of the same polymer modality")
                if mutable_residues is None or int(a["uid"]) not in mutable_residues:
                    raise m.MetricUnavailable("RFD4 residue identity changed outside mutable design positions")
            selected = {name: i for name, i in a["rows"].items()
                        if (not backbone or name in _BACKBONE)
                        and (backbone or not changed or name in conserved)
                        and m._valid_coord(reference, i)}
            # Native refolders differ on ordinary terminal OXT completion.
            # Normalize only this optional atom; named motif/chemical OXT
            # and every other observed source atom remain required.
            if (selected.get("OXT") in optional_terminal_oxt
                    and str(b["res_name"]) in PROTEIN_RES_ATOMS
                    and str(b["res_name"]) != "UNK"):
                other = b["rows"].get("OXT")
                if other is None or not m._valid_coord(predicted, other):
                    selected.pop("OXT")
            if changed and not backbone and not selected:
                raise m.MetricUnavailable("RFD4 changed residue has no observed conserved atoms")
            if not backbone:
                present = {name for name, i in b["rows"].items() if m._valid_coord(predicted, i)}
                # The pinned AF3Oracle normalization removes model-completed
                # standard atoms, including added terminal OXT/phosphate, when
                # their coordinates were absent from the observed input.
                extras = present - set(selected)
                canonical = standard_atoms.get(kind, {}).get(str(b["res_name"] if changed else a["res_name"]))
                allowed = set(canonical or ())
                if canonical is not None and kind == "protein":
                    allowed.add("OXT")
                if extras and (canonical is None or not extras <= allowed):
                    raise m.MetricUnavailable("RFD4 all-atom inventory differs between design and refold")
            for name, i in selected.items():
                j = b["rows"].get(name)
                if j is None or not m._valid_coord(predicted, j):
                    raise m.MetricUnavailable(f"RFD4 refold lacks observed {name} atom")
                if m._atom_element(reference, i) != m._atom_element(predicted, j):
                    raise m.MetricUnavailable("RFD4 named atom elements differ")
                rows.append((int(i), int(j)))
    if not rows:
        raise m.MetricUnavailable("RFD4 selection has no observed atoms")
    return rows


def _optional_terminal_oxt(input_data):
    """Ordinary canonical terminal OXT is optional unless explicitly selected."""
    from . import metrics as m
    from .inputs import _locator_rows
    from ..design.utils.atoms import PROTEIN_RES_ATOMS
    reference = input_data.atoms
    optional = set()
    for chain in m._ordered_chains(reference):
        if m._chain_mol_type(reference, chain) != "protein":
            continue
        residues = m._residue_rows(reference, chain, polymer_only=True)
        terminal = residues[-1]
        if str(terminal["res_name"]) not in PROTEIN_RES_ATOMS or str(terminal["res_name"]) == "UNK":
            continue
        row = terminal["rows"].get("OXT")
        if row is not None:
            optional.add(int(row))
    for chain, positions in input_data.motif_positions.items():
        residues = m._residue_rows(reference, chain)
        for position, insertion in positions:
            names = input_data.tip_positions.get(chain, {}).get((position, insertion))
            if names is None or "OXT" in names:
                residue = m._residue_at_position(residues, position, insertion)
                if residue is not None:
                    optional.discard(residue["rows"].get("OXT"))
    for chain, tips in input_data.tip_positions.items():
        residues = m._residue_rows(reference, chain)
        for (position, insertion), names in tips.items():
            if "OXT" in names:
                residue = m._residue_at_position(residues, position, insertion)
                if residue is not None:
                    optional.discard(residue["rows"].get("OXT"))
    for bond in input_data.bond_constraints:
        if bond.get("required_output", True):
            for endpoint in ("atom1", "atom2"):
                optional.difference_update(_locator_rows(reference, bond[endpoint]))
    bonds = getattr(reference, "bonds", None)
    if bonds is not None:
        for left, right, _ in bonds.as_array().tolist():
            if int(reference.residue_uid[int(left)]) != int(reference.residue_uid[int(right)]):
                optional.difference_update((int(left), int(right)))
    return optional


def _ligand_rows(reference, predicted, ligand_chain_map, ligand_atom_maps):
    from . import metrics as m
    from .ligand_metrics import ligand_chains
    rows = []
    for chain in ligand_chains(reference):
        target = ligand_chain_map.get(chain)
        mapping = ligand_atom_maps.get((chain, target))
        if target is None or mapping is None:
            raise m.MetricUnavailable("RFD4 ligand lacks verified graph atom correspondence")
        ref, pred, matched, expected = m._mapped_ligand_points(
            reference, predicted, chain, target, mapping, allow_predicted_extra=True)
        why = m._coverage_reason("RFD4 ligand", ref, pred, matched, expected)
        if why:
            raise m.MetricUnavailable(why)
        rows.extend((int(a), int(mapping[a])) for a in sorted(mapping))
    if not rows:
        raise m.MetricUnavailable("RFD4 selection has no mapped ligand atoms")
    return rows


def _atom_permutations(reference, rows, score, cache):
    """Graph automorphisms respect chemistry, attachments, and score selection."""
    from . import metrics as m
    from .ligand_metrics import _atom_facts
    bonds = getattr(reference, "bonds", None)
    if bonds is None:
        return []  # Upstream also leaves atom names intact without a graph.
    if "neighbors" not in cache:
        neighbors = defaultdict(list)
        for left, right, order in bonds.as_array().tolist():
            neighbors[int(left)].append((int(right), int(order)))
            neighbors[int(right)].append((int(left), int(order)))
        cache["neighbors"] = neighbors
        if "facts" not in cache:
            cache["facts"] = _atom_facts(reference, np.arange(len(reference), dtype=np.int64))
    neighbors, facts = cache["neighbors"], cache["facts"]
    selected = {int(row[0]): i for i, row in enumerate(rows)}
    groups = defaultdict(list)
    modalities = np.asarray(reference.mol_type).astype(str)
    molecules = getattr(reference, "molecule_uid", reference.residue_uid)
    labels = m._chain_values(reference)
    for local, (row, _) in enumerate(rows):
        if score[local]:
            kind = "ligand" if modalities[row] not in _POLYMER else "residue"
            groups[(kind, int(molecules[row] if kind == "ligand" else reference.residue_uid[row]))].append(row)
    result = []
    for group in groups.values():
        if len(group) < 2:
            continue
        ordered = tuple(sorted(group))
        if ordered not in cache:
            # Keep the complete chemical unit in the graph even when a motif
            # selects disconnected tips. Atoms outside this score mask stay fixed.
            first = group[0]
            unit_ids = molecules if modalities[first] not in _POLYMER else reference.residue_uid
            unit_mask = np.asarray(unit_ids) == int(unit_ids[first])
            if modalities[first] not in _POLYMER:
                unit_mask &= ~np.isin(modalities, list(_POLYMER))
            group_set = set(np.flatnonzero(unit_mask).tolist())
            scored = set(group)
            graph = nx.Graph()
            for row in group_set:
                external = tuple(sorted((m._atom_element(reference, other), str(reference.atom_name[other]),
                                         str(labels[other]), int(reference.residue_uid[other]), order)
                                        for other, order in neighbors[row] if other not in group_set))
                barrier = None if row in scored and m._valid_coord(reference, row) else row
                graph.add_node(row, fact=(m._atom_element(reference, row), facts[row], external, barrier))
                for other, order in neighbors[row]:
                    if other in group_set:
                        graph.add_edge(row, other, order=order)
            candidates = [ordered]
            if nx.is_connected(graph):
                matcher = nx.algorithms.isomorphism.GraphMatcher(
                    graph, graph, node_match=nx.algorithms.isomorphism.categorical_node_match("fact", None),
                    edge_match=nx.algorithms.isomorphism.categorical_edge_match("order", None))
                for candidate in matcher.isomorphisms_iter():
                    candidates.append(tuple(candidate[a] for a in ordered))
                    if len(candidates) > 721:
                        raise m.MetricUnavailable("RFD4 atom symmetry exceeds 720 permutations")
            cache[ordered] = [ordered] + sorted(set(candidates) - {ordered})
        candidates = cache[ordered]
        if len(candidates) > 1:
            result.append((np.asarray([selected[a] for a in ordered]),
                           np.asarray([[selected[a] for a in c] for c in candidates])))
    return result


def _subunit_groups(reference, predicted, rows, fit, score, cache):
    """Prove equivalent inventories and preserve fit/score and bond boundaries."""
    from . import metrics as m
    from .ligand_metrics import _atom_facts
    labels, pred_labels = m._chain_values(reference), m._chain_values(predicted)
    if "facts" not in cache:
        cache["facts"] = _atom_facts(reference, np.arange(len(reference), dtype=np.int64))
    facts = cache["facts"]
    by_chain, groups = defaultdict(list), defaultdict(list)
    for i, (row, _) in enumerate(rows):
        by_chain[str(labels[row])].append(i)
    edges = [] if getattr(reference, "bonds", None) is None else reference.bonds.as_array().tolist()
    linked = {str(labels[int(a)]) for a, b, _ in edges if labels[int(a)] != labels[int(b)]}
    linked |= {str(labels[int(b)]) for a, b, _ in edges if labels[int(a)] != labels[int(b)]}
    for chain, indices in by_chain.items():
        if chain in linked:
            continue
        inventory = np.flatnonzero(labels == chain).tolist()
        residue_order = {uid: i for i, uid in enumerate(dict.fromkeys(
            int(reference.residue_uid[row]) for row in inventory))}
        inventory.sort(key=lambda row: (residue_order[int(reference.residue_uid[row])],
                                        str(reference.atom_name[row]), m._atom_element(reference, row)))
        local = {row: i for i, row in enumerate(inventory)}
        indices = sorted(indices, key=lambda i: local[rows[i][0]])
        atoms = []
        for row in inventory:
            uid = int(reference.residue_uid[row])
            ordinal = residue_order.setdefault(uid, len(residue_order))
            atoms.append((ordinal, str(reference.res_name[row]), str(reference.atom_name[row]),
                          m._atom_element(reference, row), facts[row]))
        graph = tuple(sorted((min(local[int(a)], local[int(b)]), max(local[int(a)], local[int(b)]), int(order))
                             for a, b, order in edges if int(a) in local and int(b) in local))
        selection = tuple((local[rows[i][0]], bool(fit[i]), bool(score[i])) for i in indices)
        # A sequence proposal can break an input homomer's equivalence.
        # Preserve the verified chain mapping unless copies are equivalent in
        # both structures; backbone scoring must not swap distinct variants.
        pred_chain = str(pred_labels[rows[indices[0]][1]])
        pred_signature = tuple((str(item["res_name"]),) for item in m._residue_rows(predicted, pred_chain))
        groups[(tuple(atoms), graph, selection, pred_signature)].append(np.asarray(indices, dtype=int))
    return [group for group in groups.values() if len(group) > 1]


def _resolve_subunits(reference, predicted, ref, pred, rows, fit, score, minimum, cache):
    """Fit-frame Hungarian matching, without factorial copy enumeration.

    Identity is the initial authoritative mapping. A linear number of starts
    aligns one repeated fit subunit to its equivalent copies; each start then
    alternates rigid fitting and whole-subunit assignment to convergence.
    """
    from . import metrics as m
    groups = _subunit_groups(reference, predicted, rows, fit, score, cache)
    if not groups:
        return ref.copy()
    initial, _ = m._kabsch(ref[fit], pred[fit], minimum_points=minimum)
    starts = [initial]
    for group in groups:
        anchor = group[0][fit[group[0]]]
        if len(anchor) < 3 or np.linalg.matrix_rank(pred[anchor] - pred[anchor].mean(axis=0)) < 2:
            continue
        for copy in group[1:]:
            target = copy[fit[copy]]
            transform, _ = m._kabsch(ref[target], pred[anchor], minimum_points=3)
            starts.append(transform)
        break
    identity = np.arange(len(rows))
    best, best_error = ref.copy(), math.inf
    for transform in starts:
        permutation, seen = identity.copy(), set()
        while True:
            moved = m._apply_transform(pred, transform)
            candidate = identity.copy()
            for group in groups:
                measured = score[group[0]]
                if not measured.any():
                    measured = fit[group[0]]
                costs = np.asarray([[float(np.sum((moved[destination[measured]] - ref[source[measured]]) ** 2))
                                     for source in group] for destination in group])
                destinations, sources = linear_sum_assignment(costs)
                for destination, source in zip(destinations, sources, strict=True):
                    candidate[group[int(destination)]] = group[int(source)]
            key = tuple(candidate.tolist())
            aligned = ref[candidate]
            transform, _ = m._kabsch(aligned[fit], pred[fit], minimum_points=minimum)
            moved = m._apply_transform(pred[score], transform)
            error = float(np.sum((aligned[score] - moved) ** 2))
            if error < best_error:
                best, best_error = aligned.copy(), error
            if np.array_equal(candidate, permutation) or key in seen:
                break
            seen.add(key)
            permutation = candidate
    return best


def _compare(reference, predicted, fit_rows, score_rows, *, motif=False, symmetry_cache=None):
    from . import metrics as m
    rows = list(dict.fromkeys(fit_rows + score_rows))
    if not fit_rows or not score_rows:
        raise m.MetricUnavailable("RFD4 fit/score selection is empty")
    index = {pair: i for i, pair in enumerate(rows)}
    fit = np.zeros(len(rows), dtype=bool)
    score = fit.copy()
    fit[[index[p] for p in fit_rows]] = True
    score[[index[p] for p in score_rows]] = True
    ref = np.asarray([reference.coord[a] for a, _ in rows], dtype=np.float64)
    pred = np.asarray([predicted.coord[b] for _, b in rows], dtype=np.float64)
    if not np.isfinite(ref).all() or not np.isfinite(pred).all():
        raise m.MetricUnavailable("RFD4 coordinate selection is incomplete")
    if not motif:
        from .benchmark_metrics import _require_rigid_fit_geometry
        _require_rigid_fit_geometry(ref[fit], pred[fit], label="RFD4")
    minimum = 1 if motif else 3
    cache = symmetry_cache if symmetry_cache is not None else {}
    ref = _resolve_subunits(reference, predicted, ref, pred, rows, fit, score, minimum, cache)
    transform, _ = m._kabsch(ref[fit], pred[fit], minimum_points=minimum)
    moved = m._apply_transform(pred, transform)
    for destination, candidates in _atom_permutations(reference, rows, score, cache):
        active = np.any(candidates != candidates[:1], axis=0)
        errors = np.sum((ref[candidates[:, active]] - moved[destination[active]][None]) ** 2, axis=(1, 2))
        ref[destination] = ref[candidates[int(np.argmin(errors))]]
    transform, _ = m._kabsch(ref[fit], pred[fit], minimum_points=minimum)
    moved = m._apply_transform(pred[score], transform)
    return float(np.sqrt(np.mean(np.sum((ref[score] - moved) ** 2, axis=1))))


def motif_rmsd(input_data, predicted, chain_map, source, reference_chain_map=None):
    from . import metrics as m
    if source is None:
        raise m.MetricUnavailable("RFD4 motif RMSD requires original source/reference coordinates")
    if not input_data.motif_positions:
        raise m.MetricUnavailable("RFD4 motif RMSD has no coordinate motif atoms")
    groups = defaultdict(list)
    source_categories = set(source.get_annotation_categories())
    input_categories = set(input_data.atoms.get_annotation_categories())
    group_name = "condition_CoordinateGroup_1_atom"
    for chain, positions in input_data.motif_positions.items():
        source_chain = m._motif_reference_chain(input_data, source, chain, explicit_map=reference_chain_map)
        if source_chain is None:
            raise m.MetricUnavailable("RFD4 original motif chain correspondence is unavailable")
        residues = m._residue_rows(source, source_chain)
        originals = m._residue_rows(input_data.atoms, chain)
        for position, insertion in positions:
            original = m._residue_at_position(originals, position, insertion)
            a = m._residue_at_position(residues, position, insertion)
            b = m._predicted_motif_residue(input_data, predicted, chain_map, chain, position, insertion)
            if a is None or b is None or original is None:
                raise m.MetricUnavailable("RFD4 original motif residue correspondence is incomplete")
            names = input_data.tip_positions.get(chain, {}).get((position, insertion))
            if names is None:
                names = tuple(name for name, row in a["rows"].items() if m._valid_coord(source, row))
            for name in names:
                i, j = a["rows"].get(name), b["rows"].get(name)
                if (i is None or j is None or not m._valid_coord(source, i)
                        or not m._valid_coord(predicted, j)):
                    raise m.MetricUnavailable("RFD4 original motif atom correspondence is incomplete")
                if str(source.mol_type[i]) != "protein":
                    continue
                if m._atom_element(source, i) != m._atom_element(predicted, j):
                    raise m.MetricUnavailable("RFD4 original motif atom elements differ")
                if group_name in source_categories:
                    group = source.get_annotation(group_name)[i]
                elif group_name in input_categories and name in original["rows"]:
                    group = input_data.atoms.get_annotation(group_name)[original["rows"][name]]
                else:
                    group = 0  # Absence never invents independent groups.
                if isinstance(group, np.generic):
                    group = group.item()
                if group is None or (isinstance(group, float) and not math.isfinite(group)):
                    raise m.MetricUnavailable("RFD4 motif coordinate group is invalid")
                groups[group].append((int(i), int(j)))
    if not groups:
        raise m.MetricUnavailable("RFD4 source motif has no selected protein atoms")
    squared, count, cache = 0.0, 0, {}
    for rows in groups.values():
        value = _compare(source, predicted, rows, rows, motif=True, symmetry_cache=cache)
        squared += len(rows) * value * value
        count += len(rows)
    return math.sqrt(squared / count)


def _fixed_atom_states(input_data, predicted, chain_map, ligand_chain_map, ligand_atom_maps):
    """Project coordinate conditions, never identity-only fixed_restypes."""
    from . import metrics as m
    reference = input_data.atoms
    ref_chains = m._chain_values(reference)
    # The native compiler imports context with motif_coord_condition equal to
    # coord_valid. Public output files do not serialize these private flags.
    observed = np.asarray([m._valid_coord(reference, i) for i in range(len(reference))])
    fixed = ~np.isin(ref_chains, list(input_data.design_chains)) & observed
    categories = set(reference.get_annotation_categories())
    unsupported = [name for name in categories if name.startswith("condition_")
                   and "Coordinate" in name and name != "condition_CoordinateGroup_1_atom"]
    if unsupported:
        raise m.MetricUnavailable("RFD4 coordinate-conditioning representation is unavailable: " + ", ".join(sorted(unsupported)))
    explicit = False
    for name in ("C_CRD", "is_coordinate_conditioned"):
        if name in categories:
            fixed = _bool_mask(reference.get_annotation(name), (len(reference),), name).copy()
            if np.any(fixed & ~observed):
                raise m.MetricUnavailable("RFD4 fixed coordinate has no observed source atom")
            explicit = True
            break
    if not explicit:
        if "motif_coord_condition" in categories:
            fixed |= _bool_mask(reference.motif_coord_condition, (len(reference),), "motif coordinate condition") & observed
        for chain, positions in input_data.motif_positions.items():
            residues = m._residue_rows(reference, chain)
            for position, insertion in positions:
                residue = m._residue_at_position(residues, position, insertion)
                if residue is None:
                    raise m.MetricUnavailable("RFD4 fixed motif locator is unavailable")
                default_names = tuple(name for name, row in residue["rows"].items() if observed[row])
                names = input_data.tip_positions.get(chain, {}).get((position, insertion), default_names)
                for name in names:
                    row = residue["rows"].get(name)
                    if row is None or not observed[row]:
                        raise m.MetricUnavailable("RFD4 fixed motif atom is unavailable")
                    fixed[row] = True
    # Internal-coordinate-only permission is not a Cartesian fixed condition.
    internal = getattr(reference, "C_INTCRD", None)
    if internal is not None and not explicit:
        fixed[_bool_mask(internal, (len(reference),), "internal coordinate condition")] = False
    states = np.full(len(predicted), -1, dtype=np.int8)
    pred_labels = m._chain_values(predicted)
    for chain in input_data.design_chains:
        target = chain_map.get(chain, ligand_chain_map.get(chain))
        if target is not None:
            states[pred_labels == target] = 0
    for chain, target in chain_map.items():
        if not m._polymer_chain(reference, chain):
            continue
        # A model-completed atom without source coordinates was not C_CRD.
        states[(pred_labels == target) & np.isin(np.asarray(predicted.mol_type).astype(str), list(_POLYMER))] = 0
        original = m._residue_rows(reference, chain, polymer_only=True)
        refolded = m._residue_rows(predicted, target, polymer_only=True)
        if len(original) != len(refolded):
            raise m.MetricUnavailable("RFD4 fixed-coordinate residue mapping is incomplete")
        for a, b in zip(original, refolded, strict=True):
            for name, row in a["rows"].items():
                other = b["rows"].get(name)
                if other is not None:
                    states[other] = int(fixed[row])
    for chain, target in ligand_chain_map.items():
        states[(pred_labels == target) & (np.asarray(predicted.mol_type).astype(str) == "ligand")] = 0
        for row, other in ligand_atom_maps.get((chain, target), {}).items():
            states[int(other)] = int(fixed[int(row)])
    return states


def has_ligand_clashes(input_data, predicted, chain_map, ligand_chain_map, ligand_atom_maps):
    """RFD4 polymer/non-polymer scope, applied to the requested refold."""
    from . import metrics as m
    modalities = np.asarray(predicted.mol_type).astype(str)
    polymer = np.isin(modalities, list(_POLYMER))
    if not polymer.any() or polymer.all():
        raise m.MetricUnavailable("RFD4 clash check has no polymer/non-polymer atom pair")
    if getattr(predicted, "bonds", None) is None:
        raise m.MetricUnavailable("RFD4 clash check requires an authoritative refold bond graph")
    fixed = _fixed_atom_states(input_data, predicted, chain_map, ligand_chain_map, ligand_atom_maps)
    valid = np.asarray([m._valid_coord(predicted, i) for i in range(len(predicted))])
    usable = np.flatnonzero(valid)
    bonded = {tuple(sorted((int(a), int(b)))) for a, b, _ in predicted.bonds.as_array()}
    chains = (np.asarray(predicted.chain_iid).astype(str)
              if "chain_iid" in predicted.get_annotation_categories() else m._chain_values(predicted))
    uncertain = False
    coords = np.asarray(predicted.coord, dtype=np.float64)
    for i, j in cKDTree(coords[usable]).query_pairs(1.5):
        a, b = sorted((int(usable[i]), int(usable[j])))
        if (polymer[a] == polymer[b] or (a, b) in bonded
                or (chains[a] == chains[b] and predicted.residue_uid[a] == predicted.residue_uid[b])
                or np.linalg.norm(coords[a] - coords[b]) >= 1.5):
            continue
        if fixed[a] == 1 and fixed[b] == 1:
            continue
        if fixed[a] == 0 or fixed[b] == 0:
            return True
        uncertain = True
    if uncertain:
        raise m.MetricUnavailable("RFD4 clash candidate lacks coordinate-conditioning correspondence")
    if not valid.all():
        raise m.MetricUnavailable("RFD4 clash coordinates are incomplete")
    return False


def geometry_metrics(input_data, predicted, chain_map, ligand_chain_map, ligand_atom_maps,
                     source_reference, *, motif_geometry=True, reference_chain_map=None,
                     metric_provenance=None):
    from . import metrics as m
    from .benchmark_metrics import _antibody_chain_groups
    values, reasons, cache = {}, {}, {}
    reference = input_data.atoms

    def calculate(name, callback):
        try:
            values[name] = callback()
        except (m.MetricUnavailable, ValueError, TypeError, IndexError, KeyError, np.linalg.LinAlgError) as exc:
            values[name], reasons[name] = None, str(exc)

    def compare(name, fit, score, *, fit_selection, score_selection, comparable=False):
        value = _compare(reference, predicted, fit, score, symmetry_cache=cache)
        if metric_provenance is not None:
            changed = {int(reference.residue_uid[a]) for a, b in fit + score
                       if str(reference.res_name[a]) != str(predicted.res_name[b])}
            metric_provenance[name] = {
                "selection": "comparable_atoms_extension" if comparable and changed else "rfd4_observed_atoms",
                "fit_selection": fit_selection, "score_selection": score_selection,
                "changed_residue_count": len(changed),
                "fit_atom_count": len(fit), "score_atom_count": len(score),
            }
        return value

    mutable, permission_error = set(), None
    try:
        mutable = _mutable_design_residues(input_data)
    except (m.MetricUnavailable, ValueError, TypeError, IndexError, KeyError) as exc:
        permission_error = str(exc)
    optional_oxt = _optional_terminal_oxt(input_data)

    def comparable_rows(chains, *, backbone=False):
        return _polymer_rows(reference, predicted, chain_map, chains,
                             backbone=backbone, mutable_residues=mutable,
                             permission_error=permission_error, optional_terminal_oxt=optional_oxt)

    proteins = [c for c in m._ordered_chains(reference) if m._chain_mol_type(reference, c) == "protein"]
    backbone, backbone_error = None, "RFD4 selection has no protein backbone atoms"
    try:
        backbone = _polymer_rows(reference, predicted, chain_map, proteins, backbone=True,
                                 optional_terminal_oxt=optional_oxt)
    except m.MetricUnavailable as exc:
        backbone_error = str(exc)

    def require_backbone():
        if backbone is None:
            raise m.MetricUnavailable(backbone_error)
        return backbone

    name = "rfd4_backbone_rmsd"
    calculate(name, lambda: compare(name, require_backbone(), require_backbone(),
        fit_selection="all_protein_backbone", score_selection="all_protein_backbone"))
    name = "rfd4_backbone_aligned_ligand_rmsd"
    calculate(name, lambda: compare(name,
        require_backbone(), _ligand_rows(reference, predicted, ligand_chain_map, ligand_atom_maps),
        fit_selection="all_protein_backbone", score_selection="all_mapped_ligand_atoms"))
    binder = list(input_data.design_chains)
    targets = [c for c in m._ordered_chains(reference) if c not in binder and m._chain_mol_type(reference, c) in _POLYMER]
    if "heavy_role" in input_data.row:
        try:
            binder, targets = map(list, _antibody_chain_groups(input_data.row))
        except ValueError:
            targets = []
    binder = [c for c in binder if m._chain_mol_type(reference, c) in _POLYMER]

    def target_aligned(backbone_only):
        if not binder or not targets:
            raise m.MetricUnavailable("RFD4 target-aligned RMSD requires designed binder and polymer target")
        kinds = {m._chain_mol_type(reference, c) for c in binder}
        target_kinds = {m._chain_mol_type(reference, c) for c in targets}
        if backbone_only and (kinds != {"protein"} or target_kinds != {"protein"}):
            raise m.MetricUnavailable("RFD4 target-backbone alignment requires protein binder/target")
        if kinds != {"protein"} and not (kinds <= {"rna", "dna"} and target_kinds == {"protein"}):
            raise m.MetricUnavailable("RFD4 target-aligned binder modalities are unsupported")
        fit = comparable_rows(targets, backbone=backbone_only)
        fixed = _fixed_atom_states(input_data, predicted, chain_map, ligand_chain_map, ligand_atom_maps)
        # The pinned all-target mask also includes generated DNA context.
        fit = [(a, b) for a, b in fit if fixed[b] == 1 or (not backbone_only and str(reference.mol_type[a]) == "dna")]
        score = comparable_rows(binder)
        name = ("rfd4_target_backbone_aligned_binder_rmsd" if backbone_only
                else "rfd4_target_aligned_binder_rmsd")
        return compare(name, fit, score,
            fit_selection="target_protein_backbone" if backbone_only else "target_observed_atoms",
            score_selection="unchanged_all_observed_atoms_mutated_protein_backbone_or_na_framework",
            comparable=True)

    calculate("rfd4_target_aligned_binder_rmsd", lambda: target_aligned(False))
    calculate("rfd4_target_backbone_aligned_binder_rmsd", lambda: target_aligned(True))
    if motif_geometry:
        calculate("rfd4_motif_rmsd_reference", lambda: motif_rmsd(
            input_data, predicted, chain_map, source_reference, reference_chain_map))
    else:
        values["rfd4_motif_rmsd_reference"] = None
        reasons["rfd4_motif_rmsd_reference"] = "motif geometry skipped for this pipeline evaluation"
    calculate("rfd4_has_ligand_clashes", lambda: has_ligand_clashes(
        input_data, predicted, chain_map, ligand_chain_map, ligand_atom_maps))
    return values, reasons
