"""Minimize a fully assembled hybrid with explicit context motion control."""
import numpy as np


def prepare_local_products(candidate):
    """Freeze graph-connected product chemistry when design touches context."""
    from .product_graph import build_product_records
    from .regions import plan_covalent_region
    from .refine import atom_key
    from biotite.structure import BondType
    from ..runner import PostprocessPreparationError

    atoms = candidate.atoms
    moving = set(map(int, np.flatnonzero(atoms.is_design)))
    edges = () if atoms.bonds is None else atoms.bonds.as_array()
    if any(int(order) == int(BondType.COORDINATION) and (int(i) in moving or int(j) in moving)
           for i, j, order in edges):
        raise PostprocessPreparationError("local postprocess does not parameterize metal coordination")
    if not any((int(i) in moving) != (int(j) in moving) for i, j, _ in edges):
        return (), ()
    plan_covalent_region(atoms, moving)  # Validate attachment endpoints/ownership.
    graph = [set() for _ in atoms]
    for i, j, _ in edges:
        graph[int(i)].add(int(j))
        graph[int(j)].add(int(i))
    seen = set()
    groups = []
    for row in sorted(moving):
        if row in seen:
            continue
        group, queue = {row}, [row]
        while queue:
            for neighbor in graph[queue.pop()]:
                if neighbor not in group:
                    group.add(neighbor)
                    queue.append(neighbor)
        seen.update(group)
        groups.append(sorted(group))
    return build_product_records(candidate, groups), tuple(atom_key(atoms, r) for r in sorted(moving))


def minimize_local_products(atoms, *, payload, platform_name, charge_method, move_context=False):
    """Parameterize each local product fragment, assemble once, minimize once."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from rdkit import Chem
    from .refine import (atom_key, OpenMMPostprocessError, _sample_postprocess_stage,
                         _check_context_motion_support)
    _check_context_motion_support(payload, move_context=move_context)
    from .regions import plan_covalent_region
    from .hybrid import cap_product_record, parameterize_product_record
    from .mapping import product_atom_keys
    from .scaffold import complete_reference_termini, parameterize_scaffold
    from .assembly import assemble_product
    from .reference_chemistry import match_nucleic_reference

    atoms = atoms.copy()
    row_by_key = {atom_key(atoms, row): row for row in range(len(atoms))}
    moving = set(payload.local_moving_keys)
    if not moving or not moving.issubset(row_by_key):
        raise OpenMMPostprocessError("local payload moving atoms differ from final structure")
    atoms.is_design[:] = False
    moving_rows = [row_by_key[key] for key in moving]
    atoms.is_design[moving_rows] = True
    region = plan_covalent_region(atoms, moving_rows)
    retained = {atom_key(atoms, row) for row in (*region.product_rows, *region.halo_rows)}
    reference = complete_reference_termini(atoms[~atoms.is_design])
    scaffold, topology = parameterize_scaffold(reference)
    records = match_nucleic_reference(
        payload.local_product_records, atoms,
        {atom_key(atoms, row) for row in region.product_rows}, reference, scaffold, topology,
    )
    product, molecule, keys = None, None, []
    with TemporaryDirectory(prefix="partidesign-local-product-") as directory:
        for ordinal, record in enumerate(records):
            selection = retained.intersection(record.heavy_keys)
            if not selection:
                continue
            local, _ = cap_product_record(record, selection)
            work = Path(directory) / str(ordinal)
            work.mkdir()
            with _sample_postprocess_stage("local product parameterization"):
                fragment, fragment_mol = parameterize_product_record(local, atoms, work, charge_method=charge_method)
            keys.extend(product_atom_keys(fragment_mol, local, record))
            product = fragment if product is None else product + fragment
            molecule = fragment_mol if molecule is None else Chem.CombineMols(molecule, fragment_mol)
        if product is None:
            raise OpenMMPostprocessError("local payload contains no selected product")
        assembled, final_keys, _ = assemble_product(
            atoms, region, reference_atoms=reference, scaffold=scaffold, scaffold_topology=topology,
            product=product, molecule=molecule, product_keys=keys, full_records=records,
        )
        return minimize_assembled_product(atoms, assembled, final_keys, moving_rows,
                                          platform_name=platform_name, move_context=move_context)


def minimize_assembled_product(atoms, structure, final_keys, moving_rows, *, platform_name, move_context=False):
    from openmm import CustomExternalForce, Platform, VerletIntegrator, unit
    from openmm.app import NoCutoff, Simulation
    from .mapping import heavy_parent
    from .scaffold import rebuild_amber_exceptions, audit_amber_exceptions
    from .refine import atom_key, OpenMMPostprocessError, _sample_postprocess_stage, _run_openmm_minimizer

    public = {atom_key(atoms, row): row for row in range(len(atoms))}
    moving_rows = set(map(int, moving_rows))
    if not moving_rows or any(row < 0 or row >= len(atoms) or not atoms.is_design[row] for row in moving_rows):
        raise OpenMMPostprocessError("hybrid motion selection includes invalid/context atoms")
    moving = set(public) if move_context else {atom_key(atoms, row) for row in moving_rows}
    if len(final_keys) != len(structure.atoms) or len(final_keys) != len(set(final_keys)) or not set(public).issubset(final_keys):
        raise OpenMMPostprocessError("hybrid minimization atom mapping is invalid")
    positions = np.asarray(structure.coordinates, dtype=float).copy() / 10
    for particle, key in enumerate(final_keys):
        if key in public:
            positions[particle] = atoms.coord[public[key]] / 10
    topology = structure.topology
    with _sample_postprocess_stage("local product force field"):
        system = structure.createSystem(nonbondedMethod=NoCutoff, constraints=None, rigidWater=False)
        rebuild_amber_exceptions(system, topology)
        audit_amber_exceptions(system, topology)
    restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", 10 * unit.kilocalories_per_mole / unit.angstrom**2)
    for name in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    fixed = []
    mobile = []
    for particle, key in enumerate(final_keys):
        if not move_context and heavy_parent(key) not in moving:
            system.setParticleMass(particle, 0)
            fixed.append(particle)
        else:
            mobile.append(particle)
        if key in moving:
            restraint.addParticle(particle, positions[particle])
    system.addForce(restraint)
    from .stability import guard_zero_lj_exceptions
    guard_zero_lj_exceptions(system, topology)
    properties = {"DeviceIndex": "0", "Precision": "mixed", "UseBlockingSync": "true"} if platform_name == "CUDA" else {"Threads": "1"}
    with _sample_postprocess_stage("local product OpenMM context"):
        simulation = Simulation(topology, system, VerletIntegrator(0.001 * unit.picoseconds),
                                Platform.getPlatformByName(platform_name), properties)
        simulation.context.setPositions(positions * unit.nanometer)
        initial = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not np.isfinite(initial):
        raise OpenMMPostprocessError("local product initial energy is non-finite before minimization")
    _run_openmm_minimizer(simulation, unit)
    with _sample_postprocess_stage("local product final state"):
        state = simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        final = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        xyz = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    if not np.isfinite(xyz).all() or not np.isfinite(forces).all() or not np.isfinite([initial, final]).all():
        raise OpenMMPostprocessError("local product minimization returned non-finite numerical output")
    if fixed and not np.allclose(xyz[fixed], positions[fixed], rtol=0, atol=1e-6):
        raise OpenMMPostprocessError("local product minimization moved fixed context")
    result = atoms.copy()
    for particle, key in enumerate(final_keys):
        if key in moving:
            result.coord[public[key]] = xyz[particle] * 10
    return result, float(initial), float(final), float(np.sqrt(np.mean(np.sum(forces[mobile]**2, axis=-1))))
