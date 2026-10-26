"""Actor-local BC1 standard-protein relaxation and native interface metrics.

Protocol adapted from BindCraft, Martin Pacesa and contributors (MIT), commit
7713aa0d0d351e4117a8befeb8541f3a8ebd3368, functions/pyrosetta_utils.py.
Explicit assemblies, exact CIF mapping and per-metric errors are PD changes.
"""
from __future__ import annotations
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable
import numpy as np
from .pose_mapping import (MappingError, PoseMapping, SourceStructure,
                           pose_coordinates, verify_pose, verify_required_connections)

REQUIRED_METRICS = (
    "bc1_binder_energy_score", "bc1_surface_hydrophobicity", "bc1_shape_complementarity",
    "bc1_dg", "bc1_dsasa", "bc1_interface_residues", "bc1_interface_hbonds",
    "bc1_interface_unsat_hbonds",
)
_IAM_METRICS = ("bc1_shape_complementarity", "bc1_dg", "bc1_dsasa",
                "bc1_interface_hbonds", "bc1_packstat", "bc1_dg_dsasa")


def validate_config(config: dict) -> dict:
    result = dict(config)
    if result.get("scorefunction", "beta_nov16") != "beta_nov16":
        raise ValueError("native postfilter scorefunction must be beta_nov16")
    for name, default in (("seed", 2026), ("threads", 1)):
        value = result.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "seed" else 1):
            raise ValueError(f"native {name} must be an integer in range")
        result[name] = value
    relax = dict(result.get("relax", {}))
    for name, expected in (("max_iter", 200), ("repeats", 1)):
        value = relax.get(name, expected)
        if type(value) is not int or value != expected:
            raise ValueError(f"native relax.{name} must be {expected}")
        relax[name] = expected
    if set(relax).difference({"max_iter", "repeats"}):
        raise ValueError("native relax supports only max_iter and repeats")
    result["relax"] = relax
    result["scorefunction"] = "beta_nov16"
    return result


def initialize(config: dict) -> Any:
    """Import and initialize the licensed SDK once inside a persistent actor."""
    config = validate_config(config)
    import pyrosetta as pr
    options = ["-mute", "all", "-corrections::beta_nov16", "true",
        "-relax:default_repeats", str(config["relax"]["repeats"]),
        "-constant_seed", "-jran", str(config["seed"]),
        "-multithreading:total_threads", str(config["threads"]), "-in:detect_disulf", "false"]
    if config.get("dalphaball"):
        path = Path(config["dalphaball"]).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            options.extend(("-holes:dalphaball", str(path)))
    pr.init(options=" ".join(options))
    return pr


def seed_metrics(pr: Any, seed: int) -> None:
    """Reset native metric RNG to the actor protocol's metric-phase seed."""
    pr.rosetta.numeric.random.rg().set_seed(seed)


def configure_partner_foldtree(source: SourceStructure, pose: Any, mapping: PoseMapping,
                               pr: Any) -> int:
    """Build and check one separation jump between the explicit assemblies."""
    tree = pr.rosetta.core.kinematics.FoldTree()
    segments: dict[int, list[int]] = {}
    segment_chain: dict[int, str] = {}
    for occurrence in source.residues:
        segments.setdefault(occurrence.segment, []).append(mapping.pose_residues[occurrence.ordinal - 1])
        segment_chain[occurrence.segment] = occurrence.chain
    for indices in segments.values():
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise MappingError("Pose reordered connected source segments")
        if len(indices) > 1:
            tree.add_edge(indices[0], indices[-1], -1)
    roots: list[int] = []
    jump = 0
    for chain_set in (set(source.target_chains), set(source.binder_chains)):
        starts = [indices[0] for segment, indices in segments.items() if segment_chain[segment] in chain_set]
        roots.append(starts[0])
        for start in starts[1:]:
            jump += 1
            tree.add_edge(starts[0], start, jump)
    jump += 1
    tree.add_edge(roots[0], roots[1], jump)
    if not tree.reorder(1) or not tree.check_fold_tree():
        raise MappingError("cannot construct an assembly-preserving FoldTree")
    partition = tree.partition_by_jump(jump)
    selected = {i for i in range(1, pose.total_residue() + 1) if partition[i]}
    binder = {mapping.pose_residues[i - 1] for i in source.binder_indices}
    target = {mapping.pose_residues[i - 1] for i in source.target_indices}
    if selected != binder and selected != target:
        raise MappingError("separation jump does not partition the declared assemblies")
    before = pose_coordinates(pose, mapping)
    pose.fold_tree(tree)
    if not np.allclose(before, pose_coordinates(pose, mapping), rtol=0, atol=1e-10):
        raise MappingError("assembly FoldTree changed source coordinates")
    mapping.separation_jump = jump
    mapping.binder_pose_residues = tuple(sorted(binder))
    mapping.target_pose_residues = tuple(sorted(target))
    mapping.details.update({"partners": source.partners, "separation_jump": jump,
        "binder_residues": len(binder), "target_residues": len(target)})
    verify_pose(source, pose, mapping)
    return jump


def _head_tail_constraints(source: SourceStructure, pose: Any, mapping: PoseMapping,
                           pr: Any) -> list[Any]:
    """Native overlap atoms constrain the closed bond's length, angles and omega."""
    constraints = []
    atom_id = pr.rosetta.core.id.AtomID
    constraint_type = pr.rosetta.core.scoring.constraints.AtomPairConstraint
    harmonic = pr.rosetta.core.scoring.func.HarmonicFunc
    for first, last in getattr(source, "head_tail_pairs", ()):
        first_index, last_index = mapping.pose_residues[first - 1], mapping.pose_residues[last - 1]
        for left_index, left_atom, right_index, right_atom in (
                (first_index, "N", last_index, "OVL1"),
                (first_index, "CA", last_index, "OVL2"),
                (last_index, "C", first_index, "OVU1")):
            constraint = constraint_type(
                atom_id(pose.residue(left_index).atom_index(left_atom), left_index),
                atom_id(pose.residue(right_index).atom_index(right_atom), right_index),
                harmonic(0.0, 0.1))
            pose.add_constraint(constraint)
            constraints.append(constraint)
    return constraints


def validate_head_tail_geometry(source: SourceStructure, pose: Any, mapping: PoseMapping) -> list[dict]:
    """Check the real terminal bond geometry after relaxation."""
    def xyz(residue, atom):
        return np.asarray([float(v) for v in pose.residue(residue).xyz(atom)], dtype=np.float64)
    def angle(a, b, c):
        u, v = a - b, c - b
        norm = float(np.linalg.norm(u) * np.linalg.norm(v))
        if not math.isfinite(norm) or norm <= 0:
            raise MappingError("degenerate head-to-tail peptide bond geometry")
        return math.degrees(math.acos(float(np.clip(np.dot(u, v) / norm, -1.0, 1.0))))
    result = []
    for first, last in getattr(source, "head_tail_pairs", ()):
        first_index, last_index = mapping.pose_residues[first - 1], mapping.pose_residues[last - 1]
        ca_last, carbon, nitrogen, ca_first = (xyz(last_index, "CA"), xyz(last_index, "C"),
                                               xyz(first_index, "N"), xyz(first_index, "CA"))
        distance = float(np.linalg.norm(carbon - nitrogen))
        carbon_angle, nitrogen_angle = angle(ca_last, carbon, nitrogen), angle(carbon, nitrogen, ca_first)
        axis = nitrogen - carbon
        axis /= np.linalg.norm(axis)
        v = ca_last - carbon
        w = ca_first - nitrogen
        v -= np.dot(v, axis) * axis
        w -= np.dot(w, axis) * axis
        if min(float(np.linalg.norm(v)), float(np.linalg.norm(w))) <= 0:
            raise MappingError("degenerate head-to-tail peptide omega geometry")
        omega = math.degrees(math.atan2(float(np.dot(np.cross(axis, v), w)), float(np.dot(v, w))))
        planarity = min(abs(omega), abs(180.0 - abs(omega)))
        if (not all(math.isfinite(value) for value in (distance, carbon_angle, nitrogen_angle, omega))
                or not 1.1 <= distance <= 1.7
                or not 85.0 <= carbon_angle <= 155.0
                or not 85.0 <= nitrogen_angle <= 155.0
                or planarity > 45.0):
            raise MappingError("relaxed head-to-tail peptide geometry is invalid: "
                f"C-N={distance:.3f} A, angles={carbon_angle:.2f}/{nitrogen_angle:.2f}, omega={omega:.2f}")
        result.append({"chain": source.residues[first - 1].chain,
            "first_occurrence": first, "last_occurrence": last,
            "distance": distance, "carbon_angle": carbon_angle,
            "nitrogen_angle": nitrogen_angle, "omega": omega})
    if result:
        mapping.details["head_tail_closures"] = len(result)
        mapping.details["head_tail_max_distance"] = max(item["distance"] for item in result)
    return result


def relax_pose(source: SourceStructure, pose: Any, mapping: PoseMapping,
               pr: Any, config: dict) -> Any:
    config = validate_config(config)
    start_pose = pose.clone()
    movemap = pr.rosetta.core.kinematics.MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)
    movemap.set_jump(False)
    scorefxn = pr.get_fa_scorefxn()
    cycles = getattr(source, "head_tail_pairs", ())
    original_constraints = pose.constraint_set().clone() if cycles else None
    if cycles:
        _head_tail_constraints(source, pose, mapping, pr)
        # Only the relax scorefunction sees closure penalties. Public BC1
        # energies below use the unchanged native beta_nov16 scorefunction.
        scorefxn.set_weight(pr.rosetta.core.scoring.atom_pair_constraint, 1.0)
    relax = pr.rosetta.protocols.relax.FastRelax(scorefxn, config["relax"]["repeats"])
    relax.set_movemap(movemap)
    relax.max_iter(config["relax"]["max_iter"])
    relax.min_type("lbfgs_armijo_nonmonotone")
    # Keep the baseline setter state; the SDK's separate constrain_coords
    # gate stays disabled, so no starting-coordinate restraints are created.
    relax.constrain_relax_to_start_coords(True)
    task_factory = pr.rosetta.core.pack.task.TaskFactory()
    task_factory.push_back(pr.rosetta.core.pack.task.operation.RestrictToRepacking())
    relax.set_task_factory(task_factory)
    mapping.details["relax_effective"] = {
        "max_iter": int(relax.max_iter()), "repeats": int(relax.default_repeats()),
        "min_type": str(relax.min_type()),
        "constrain_start_flag": bool(relax.constrain_relax_to_start_coords()),
        "coordinate_constraints": bool(relax.constrain_coords()),
    }
    try:
        relax.apply(pose)
    finally:
        if original_constraints is not None:
            pose.constraint_set(original_constraints)
    align = pr.rosetta.protocols.simple_moves.AlignChainMover()
    align.source_chain(0)
    align.target_chain(0)
    align.pose(start_pose)
    align.apply(pose)
    if not np.isfinite(pose_coordinates(pose, mapping)).all():
        raise MappingError("Rosetta returned nonfinite coordinates")
    verify_required_connections(source, pose, mapping)
    validate_head_tail_geometry(source, pose, mapping)
    return pose


def _interface_contacts(source: SourceStructure, pose: Any, mapping: PoseMapping,
                        *, include_composition: bool = True) -> tuple[list[int], dict[str, int]]:
    # BC1's Bio.PDB hotspot calculation unfolds every atom in Rosetta's PDB,
    # including native hydrogens and terminal preparation atoms.
    binder_coordinates, binder_occurrences, target_coordinates = [], [], []
    for occurrence in source.residues:
        residue = pose.residue(mapping.pose_residues[occurrence.ordinal - 1])
        coordinates = [[float(value) for value in residue.xyz(i)] for i in range(1, residue.natoms() + 1)
                       if not residue.atom_type(i).is_virtual()]
        if occurrence.chain in source.binder_chains:
            binder_coordinates.extend(coordinates)
            binder_occurrences.extend([occurrence.ordinal] * len(coordinates))
        else:
            target_coordinates.extend(coordinates)
    binder_coordinates = np.asarray(binder_coordinates, dtype=np.float64)
    target_coordinates = np.asarray(target_coordinates, dtype=np.float64)
    contacts: set[int] = set()
    for offset in range(0, len(binder_coordinates), 128):
        distance2 = np.sum((binder_coordinates[offset:offset + 128, None, :]
                           - target_coordinates[None, :, :]) ** 2, axis=-1)
        for local_index in np.flatnonzero(np.any(distance2 <= 16.0, axis=1)):
            contacts.add(binder_occurrences[offset + int(local_index)])
    composition = {}
    if include_composition:
        composition = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
        from .pose_mapping import _AA
        for ordinal in contacts:
            composition[_AA[source.residues[ordinal - 1].name]] += 1
    return sorted(contacts), composition


def _selector(pr: Any, indices: list[int]) -> Any:
    return pr.rosetta.core.select.residue_selector.ResidueIndexSelector(",".join(map(str, indices)))


def _binder_surface(source: SourceStructure, pose: Any, mapping: PoseMapping, pr: Any) -> float:
    indices = pr.rosetta.utility.vector1_unsigned_long()
    for ordinal in source.binder_indices:
        indices.append(mapping.pose_residues[ordinal - 1])
    binder_pose = pr.Pose()
    pr.rosetta.core.io.pose_from_pose(binder_pose, pose, indices)
    selector = pr.rosetta.core.select.residue_selector.LayerSelector()
    selector.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
    selected = selector.apply(binder_pose)
    total, hydrophobic = 0, 0
    for index in range(1, binder_pose.total_residue() + 1):
        if selected[index]:
            residue = binder_pose.residue(index)
            total += 1
            hydrophobic += int(residue.is_apolar() or residue.name3() in {"PHE", "TRP", "TYR"})
    if not total:
        raise ValueError("LayerSelector selected no binder surface residues")
    return hydrophobic / total


def _iam_metrics(source: SourceStructure, pose: Any, pr: Any, scorefxn: Any,
                 required_metrics=None) -> dict[str, float]:
    wanted = set(_IAM_METRICS if required_metrics is None else required_metrics)
    mover = pr.rosetta.protocols.analysis.InterfaceAnalyzerMover()
    partners: Any = source.partners
    docking_type = getattr(pr.rosetta.core.pose, "DockingPartners", None)
    if docking_type is not None:
        partners = docking_type.docking_partners_from_string(partners)
    mover.set_interface(partners)
    mover.set_scorefunction(scorefxn)
    mover.set_compute_packstat("bc1_packstat" in wanted)
    mover.set_compute_interface_energy(bool(wanted & {"bc1_dg", "bc1_dg_dsasa", "bc1_interface_hbonds"}))
    mover.set_calc_dSASA(bool(wanted & {"bc1_dsasa", "bc1_dg_dsasa", "bc1_interface_hbonds"}))
    mover.set_calc_hbond_sasaE("bc1_interface_hbonds" in wanted)
    mover.set_compute_interface_sc("bc1_shape_complementarity" in wanted)
    mover.set_pack_input(False)
    mover.set_pack_separated(True)
    mover.apply(pose.clone())
    def read(action):
        try:
            return action()
        except Exception as exc:
            return exc
    data = read(mover.get_all_data) if wanted & {
        "bc1_shape_complementarity", "bc1_interface_hbonds", "bc1_dg_dsasa"} else None
    def field(name):
        if isinstance(data, Exception):
            raise data
        return getattr(data, name)
    actions = {
        "bc1_shape_complementarity": lambda: field("sc_value"),
        "bc1_interface_hbonds": lambda: field("interface_hbonds"),
        "bc1_dg": mover.get_interface_dG,
        "bc1_dsasa": mover.get_interface_delta_sasa,
        "bc1_packstat": mover.get_interface_packstat,
        "bc1_dg_dsasa": lambda: field("dG_dSASA_ratio") * 100,
    }
    return {name: read(actions[name]) for name in _IAM_METRICS if name in wanted}


def _bun_parameters(pr: Any) -> dict:
    """Read the same defaults used by the pinned XML filter."""
    options = pr.rosetta.basic.options
    return {"bound_burial_cutoff": float(options.get_real_option("bunsat_calc2:sasa_burial_cutoff")),
            "apo_burial_cutoff": 0.2, "probe_radius": 1.1, "residue_surface_cutoff": 20.0,
            "max_hbond_energy": float(options.get_real_option("score:hb_max_energy")),
            "generous_hbonds": True, "legacy_counting": False, "use_vsasa": True,
            "use_sc_neighbors": False, "ignore_surface_res": False, "include_donors": True,
            "include_acceptor_hydrogens": False, "cumulative_hbonds": True,
            "ignore_bb_heavy_unsats": False, "atomic_depth_apo_surface": -1.0}


def _bun_atom_mask(pr: Any, pose: Any, selected: tuple[int, int] | None = None) -> Any:
    mask = pr.rosetta.core.id.AtomID_Map_bool_t(pose.total_residue(), False)
    for residue in range(1, pose.total_residue() + 1):
        mask.resize(residue, pose.residue(residue).natoms(), selected is None)
    if selected is not None:
        residue, atom = selected
        mask.set(pr.rosetta.core.id.AtomID(atom, residue), True)
    return mask


def _bun_integer(value: str, *, maximum: int | None = None) -> int:
    result = json.loads(value)
    if type(result) is not int or result < 0 or (maximum is not None and result > maximum):
        raise ValueError("native BUN counter returned an invalid integer")
    return result


def _bun_native_counts(calculator: Any, pose: Any) -> tuple[int, list[int]]:
    total = _bun_integer(calculator.get("all_heavy_unsats", pose))
    residues = json.loads(calculator.get("residue_bur_unsat_polars", pose))
    if (not isinstance(residues, list) or len(residues) != pose.total_residue()
            or any(type(value) is not int or value < 0 for value in residues)
            or sum(residues) != total):
        raise ValueError("native BUN residue counts do not match the heavy-atom total")
    return total, residues


def _bun_native_flag(calculator: Any, pose: Any, pr: Any, residue: int, atom: int) -> int:
    # This is a counting-region mask. Native SASA and H-bonds still use ALL atoms.
    calculator.set_special_region(_bun_atom_mask(pr, pose, (residue, atom)))
    return _bun_integer(calculator.get("all_heavy_unsats", pose), maximum=1)


def _bun_calculator_metric(pose: Any, mapping: PoseMapping, pr: Any, parameters: dict) -> tuple[int, dict]:
    """Native atomwise ddG reduction without the SDK wrapper's chain-count guard.

    Follows Rosetta commit 49e05e797ca4383c6038f090e5db6c2593cb4963,
    protocols/simple_filters/BuriedUnsatHbondFilter.cc:492-625. Each boolean
    comes from its unchanged BuriedUnsatisfiedPolarsCalculator. This wheel
    exposes scalar counters and selection masks, but not its typed atom-map
    MetricValue; single-atom masks retrieve exactly the same heavy-atom flags.
    """
    module = getattr(pr.rosetta.protocols, "simple_pose_metric_calculators", None)
    calculator_type = getattr(module, "BuriedUnsatisfiedPolarsCalculator", None)
    if calculator_type is None:
        raise RuntimeError("native BuriedUnsatisfiedPolarsCalculator is unavailable")
    if pr.rosetta.core.pose.symmetry.is_symmetric(pose):
        raise MappingError("the multi-chain BUN adapter requires an asymmetric Pose")
    jump = mapping.separation_jump
    if type(jump) is not int or not 1 <= jump <= pose.num_jump():
        raise MappingError("an explicit valid assembly separation jump is required for BC1 BUNs")
    binder = set(mapping.binder_pose_residues)
    target = set(mapping.target_pose_residues)
    if not binder or not target or binder & target or binder | target != set(range(1, pose.total_residue() + 1)):
        raise MappingError("BUN partner residue sets must partition the whole Pose")
    partition = pose.fold_tree().partition_by_jump(jump)
    selected = {r for r in range(1, pose.total_residue() + 1) if partition[r]}
    if selected != binder and selected != target:
        raise MappingError("BUN separation jump does not match the explicit assemblies")
    started = time.perf_counter()
    bound, apo = pose.clone(), pose.clone()
    scorefxn = pr.get_fa_scorefxn()
    scorefxn(bound)
    mover = pr.rosetta.protocols.rigid.RigidBodyTransMover(apo, jump)
    mover.trans_axis(mover.trans_axis())
    mover.step_size(1000.0)
    mover.apply(apo)
    apo.update_residue_neighbors()
    scorefxn(apo)
    if (bound.total_residue() != apo.total_residue() or bound.num_chains() != apo.num_chains()
            or any(bound.residue(r).natoms() != apo.residue(r).natoms()
                   for r in range(1, bound.total_residue() + 1))):
        raise MappingError("native BUN separation changed the Pose atom inventory")
    def make_calculator(current_pose, cutoff):
        return calculator_type("dalphaball", "default", _bun_atom_mask(pr, current_pose),
            cutoff, parameters["probe_radius"], parameters["residue_surface_cutoff"],
            parameters["max_hbond_energy"], parameters["generous_hbonds"], parameters["legacy_counting"],
            parameters["use_vsasa"], parameters["use_sc_neighbors"], parameters["ignore_surface_res"],
            parameters["include_donors"], parameters["include_acceptor_hydrogens"], parameters["cumulative_hbonds"])
    bound_calculator = make_calculator(bound, parameters["bound_burial_cutoff"])
    apo_calculator = make_calculator(apo, parameters["apo_burial_cutoff"])
    bound_total, bound_counts = _bun_native_counts(bound_calculator, bound)
    apo_total, apo_counts = _bun_native_counts(apo_calculator, apo)
    atoms, query_count = [], 0
    for r, expected in enumerate(bound_counts, 1):
        if not expected:
            continue
        observed = 0
        residue = bound.residue(r)
        for atom in range(1, residue.nheavyatoms() + 1):
            # The pinned native calculator can set a heavy flag only inside
            # its donor/acceptor branch. Other native atom types are exactly
            # zero; full residue/total checks below reject any disagreement.
            atom_type = residue.atom_type(atom)
            if not (atom_type.is_donor() or atom_type.is_acceptor()):
                continue
            flag = _bun_native_flag(bound_calculator, bound, pr, r, atom)
            query_count += 1
            if flag:
                observed += 1
                atoms.append((r, atom))
        if observed != expected:
            raise ValueError("native BUN atom flags do not match the full residue count")
    if len(atoms) != bound_total:
        raise ValueError("native BUN atom flags do not match the full heavy-atom count")
    count = 0
    for r, atom in atoms:
        apo_flag = _bun_native_flag(apo_calculator, apo, pr, r, atom)
        query_count += 1
        count += int(not apo_flag)
    return count, {"bound_heavy_unsats": bound_total, "apo_heavy_unsats": apo_total,
        "atom_mask_queries": query_count, "atom_query_selection": "native_donor_or_acceptor",
        "separation_distance": 1000.0, "elapsed_seconds": time.perf_counter() - started}


def _bc1_bun_metric(pose: Any, mapping: PoseMapping, pr: Any, config: dict,
                    *, force_calculator: bool = False) -> tuple[float, dict]:
    """Use the pinned XML path where supported, and its exact reduction otherwise."""
    tool = config.get("dalphaball")
    if not tool or not Path(tool).is_file() or not os.access(tool, os.X_OK):
        raise ValueError("DAlphaBall is required for BC1 buried unsatisfied hydrogen bonds")
    if mapping.separation_jump is None:
        raise MappingError("an assembly separation jump is required for BC1 BUNs")
    parameters = _bun_parameters(pr)
    details = {"parameters": parameters, "separation_jump": mapping.separation_jump,
               "pose_chains": int(pose.num_chains())}
    if pose.num_chains() < 4 and not force_calculator:
        xml = (f'<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" '
               f'ignore_surface_res="false" use_ddG_style="true" jump_number="{mapping.separation_jump}" '
               'dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />')
        native_filter = pr.rosetta.protocols.rosetta_scripts.XmlObjects.static_get_filter(xml)
        details["method"] = "xml_ddg"
        return native_filter.report_sm(pose.clone()), details
    count, calculation = _bun_calculator_metric(pose, mapping, pr, parameters)
    details.update(calculation)
    details["method"] = "native_atom_masks_ddg"
    return float(count), details


def calculate_metrics(source: SourceStructure, pose: Any, mapping: PoseMapping,
                      pr: Any, config: dict, required_metrics=None) -> dict:
    """Calculate requested native BC1 values and only their actual dependencies."""
    supported = set(REQUIRED_METRICS) | set(_IAM_METRICS) | {
        "bc1_interface_hydrophobicity", "bc1_interface_fraction",
        "bc1_interface_hbond_percentage", "bc1_interface_unsat_hbond_percentage"}
    wanted = set(REQUIRED_METRICS if required_metrics is None else required_metrics) & supported
    needed = set(wanted)
    if "bc1_interface_fraction" in wanted:
        needed.add("bc1_dsasa")
    if "bc1_interface_hbond_percentage" in wanted:
        needed.add("bc1_interface_hbonds")
    if "bc1_interface_unsat_hbond_percentage" in wanted:
        needed.add("bc1_interface_unsat_hbonds")
    metrics: dict[str, int | float] = {}
    errors: dict[str, str] = {}
    raw: dict[str, float] = {}
    failures: dict[str, str] = {}
    details = dict(mapping.details)
    def capture(names: tuple[str, ...], action: Callable[[], Any]) -> None:
        try:
            values = action()
            if len(names) == 1 and not isinstance(values, dict):
                values = {names[0]: values}
            for name in names:
                try:
                    value = values[name]
                    if isinstance(value, Exception):
                        raise value
                    if isinstance(value, bool):
                        raise ValueError("boolean value is not a native metric")
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError("nonfinite native metric")
                    raw[name] = number
                    if name in wanted:
                        metrics[name] = round(number, 2)
                except Exception as exc:
                    failures[name] = f"{type(exc).__name__}: {exc}"
                    if name in wanted:
                        errors[name] = failures[name]
        except Exception as exc:
            for name in names:
                failures[name] = f"{type(exc).__name__}: {exc}"
                if name in wanted:
                    errors[name] = failures[name]
    scorefxn = pr.get_fa_scorefxn() if needed & (set(_IAM_METRICS) | {"bc1_binder_energy_score"}) else None
    selector = None
    if wanted & {"bc1_binder_energy_score", "bc1_interface_fraction"}:
        indices = [mapping.pose_residues[i - 1] for i in source.binder_indices]
        selector = _selector(pr, indices)
    if "bc1_binder_energy_score" in wanted:
        def binder_energy() -> float:
            metric = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
            metric.set_scorefunction(scorefxn)
            metric.set_residue_selector(selector)
            return metric.calculate(pose.clone())
        capture(("bc1_binder_energy_score",), binder_energy)
    if "bc1_surface_hydrophobicity" in wanted:
        capture(("bc1_surface_hydrophobicity",), lambda: _binder_surface(source, pose.clone(), mapping, pr))
    iam_names = tuple(name for name in _IAM_METRICS if name in needed)
    if iam_names:
        capture(iam_names, lambda: _iam_metrics(source, pose, pr, scorefxn, required_metrics=iam_names))
    contact_names = {"bc1_interface_residues", "bc1_interface_hydrophobicity",
                     "bc1_interface_hbond_percentage", "bc1_interface_unsat_hbond_percentage"}
    contacts = None
    if wanted & contact_names:
        try:
            contacts, composition = _interface_contacts(source, pose, mapping,
                include_composition="bc1_interface_hydrophobicity" in wanted)
            if "bc1_interface_residues" in wanted:
                metrics["bc1_interface_residues"] = len(contacts)
            if "bc1_interface_hydrophobicity" in wanted:
                metrics["bc1_interface_hydrophobicity"] = round(
                    100 * sum(composition[aa] for aa in "ACFILMPVWY") / len(contacts), 2) if contacts else 0.0
        except Exception as exc:
            for name in wanted & contact_names:
                errors[name] = f"{type(exc).__name__}: {exc}"
    if "bc1_interface_unsat_hbonds" in needed:
        def buns() -> float:
            value, calculation = _bc1_bun_metric(pose, mapping, pr, config)
            details["bun_calculation"] = calculation
            return value
        capture(("bc1_interface_unsat_hbonds",), buns)
    if "bc1_interface_fraction" in wanted:
        def binder_sasa_fraction() -> float:
            if "bc1_dsasa" not in raw:
                raise ValueError(failures.get("bc1_dsasa", "interface dSASA is unavailable"))
            metric = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
            metric.set_residue_selector(selector)
            sasa = float(metric.calculate(pose.clone()))
            return 100 * raw["bc1_dsasa"] / sasa if sasa > 0 else 0.0
        capture(("bc1_interface_fraction",), binder_sasa_fraction)
    for source_name, output_name in (
            ("bc1_interface_hbonds", "bc1_interface_hbond_percentage"),
            ("bc1_interface_unsat_hbonds", "bc1_interface_unsat_hbond_percentage")):
        if output_name not in wanted or output_name in errors:
            continue
        if not contacts:
            errors[output_name] = "no binder-target contacts"
        elif source_name not in raw:
            errors[output_name] = failures.get(source_name, "native dependency is unavailable")
        else:
            metrics[output_name] = round(100 * raw[source_name] / len(contacts), 2)
    details.update({"comparison_decimals": 2, "coordinate_stage": "relaxed_refold"})
    return {"metrics": metrics, "errors": errors, "details": details}
