"""Private whole-peptide chemistry; never alter the public residue representation."""
from __future__ import annotations

import numpy as np
from rdkit import Chem

from ...utils.atoms import PROTEIN_RES_ATOMS
from ...utils.ccd import load_packaged_polymer_ccd


def peptide_records(candidate):
    """Select non-template design peptide graphs and freeze their exact chemistry."""
    from ..runner import (
        PostprocessPreparationError, _atom_key, gaff_record_from_rdkit,
        _validate_gaff_record_against_structure,
    )

    atoms = candidate.atoms
    chains = {c.chain_id for c in candidate.base_request.task.design_chains if c.modality == "protein"}
    if not chains:
        return ()
    edges = [] if atoms.bonds is None else atoms.bonds.as_array()
    adjacency = [set() for _ in atoms]
    for i, j, _ in edges:
        adjacency[int(i)].add(int(j))
        adjacency[int(j)].add(int(i))
    special = set()
    standard = set(PROTEIN_RES_ATOMS) - {"UNK"}
    for chain in chains:
        rows = np.flatnonzero(atoms.chain_id == chain)
        if any(str(atoms.res_name[r]) not in standard for r in rows):
            special.add(chain)
        for i, j, _ in edges:
            i, j = int(i), int(j)
            if str(atoms.chain_id[i]) != chain and str(atoms.chain_id[j]) != chain:
                continue
            same_residue = (atoms.chain_id[i], atoms.res_id[i], atoms.ins_code[i]) == (atoms.chain_id[j], atoms.res_id[j], atoms.ins_code[j])
            if same_residue:
                continue
            if atoms.chain_id[i] == atoms.chain_id[j]:
                lo, hi = sorted((i, j), key=lambda r: int(atoms.res_id[r]))
                if int(atoms.res_id[hi]) == int(atoms.res_id[lo]) + 1 and atoms.atom_name[lo] == "C" and atoms.atom_name[hi] == "N":
                    continue
            # Canonical peptide closure and cystine bonds have native Amber
            # parameters. Only non-native links require a whole GAFF product.
            pair = {str(atoms.atom_name[i]), str(atoms.atom_name[j])}
            if pair == {"C", "N"} and all(str(atoms.res_name[r]) in standard for r in (i, j)):
                continue
            if pair == {"SG"} and atoms.res_name[i] == atoms.res_name[j] == "CYS":
                continue
            special.add(chain)
    if not special:
        return ()
    selected = set(map(int, np.flatnonzero(np.isin(atoms.chain_id, list(special)))))
    queue = list(selected)
    while queue:
        for neighbor in adjacency[queue.pop()]:
            if neighbor not in selected:
                selected.add(neighbor)
                queue.append(neighbor)
    if any(not bool(atoms.is_design[r]) for r in selected):
        raise PostprocessPreparationError("whole-peptide optimization does not support covalent receptor attachment")
    if any(str(atoms.mol_type[r]) not in {"protein", "ligand"} for r in selected):
        raise PostprocessPreparationError("whole-peptide optimization supports protein and attached ligand bridges only")
    private_errors = getattr(candidate, "private_ligand_errors", {})
    for uid in {int(atoms.molecule_uid[r]) for r in selected if atoms.mol_type[r] == "ligand"}:
        if uid in private_errors:
            raise PostprocessPreparationError(
                f"selected peptide bridge {uid} has invalid private chemistry: {private_errors[uid]}"
            )
    # Disconnected molecules are independent GAFF records, never joined by proximity.
    groups = []
    remaining = set(selected)
    while remaining:
        group = {min(remaining)}
        queue = list(group)
        while queue:
            for neighbor in adjacency[queue.pop()]:
                if neighbor not in group:
                    group.add(neighbor)
                    queue.append(neighbor)
        remaining.difference_update(group)
        groups.append(sorted(group))
    ccd = load_packaged_polymer_ccd()
    private_charges = {}
    private_kekule = {}
    for uid, molecule in candidate.private_ligand_molecules.items():
        rows = np.flatnonzero((atoms.molecule_uid == int(uid)) & (atoms.mol_type == "ligand")).tolist()
        if not rows or not set(rows).intersection(selected):
            continue
        record = gaff_record_from_rdkit(molecule, record_id=f"bridge-{uid}", molecule_uid=int(uid), component_uid=int(atoms.component_uid[rows[0]]), heavy_keys=tuple(_atom_key(atoms, r) for r in rows))
        _validate_gaff_record_against_structure(record, atoms, rows)
        private_charges.update(zip(rows, record.formal_charges, strict=True))
        reference = Chem.Mol(molecule)
        Chem.Kekulize(reference, clearAromaticFlags=True)
        heavy_rows = dict(zip(
            (atom.GetIdx() for atom in reference.GetAtoms() if atom.GetAtomicNum() > 1),
            rows, strict=True,
        ))
        for bond in reference.GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if left in heavy_rows and right in heavy_rows:
                private_kekule[tuple(sorted((heavy_rows[left], heavy_rows[right])))] = bond.GetBondType()
    order_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 5: Chem.BondType.SINGLE, 6: Chem.BondType.DOUBLE, 7: Chem.BondType.TRIPLE, 9: Chem.BondType.AROMATIC}
    records = []
    for ordinal, rows in enumerate(groups):
        local = {r: i for i, r in enumerate(rows)}
        rw = Chem.RWMol()
        for r in rows:
            component = ccd.get(str(atoms.res_name[r]))
            fact = component.atom_by_id(str(atoms.atom_name[r])) if component else None
            charge = private_charges.get(r)
            if charge is None:
                if fact is None or fact.formal_charge is None or str(fact.element).upper() != str(atoms.element[r]).upper():
                    raise PostprocessPreparationError(f"no authoritative atom charge for {_atom_key(atoms, r)}")
                charge = fact.formal_charge
            atom = Chem.Atom(str(atoms.element[r]).capitalize())
            atom.SetFormalCharge(int(charge))
            rw.AddAtom(atom)
        aromatic_pairs = []
        for i, j, order in edges:
            i, j, order = int(i), int(j), int(order)
            if i in local and j in local:
                if order not in order_map:
                    raise PostprocessPreparationError("whole-peptide graph needs concrete covalent bond orders")
                bond_type = order_map[order]
                if order in {5, 6, 7, 9}:
                    aromatic_pairs.append((local[i], local[j]))
                if order == 9:
                    # An aromatic-only CIF may omit Kekule orders. Recover
                    # them from the validated chemistry, including [nH] state,
                    # rather than inventing a nitrogen tautomer.
                    bond_type = private_kekule.get(tuple(sorted((i, j))), bond_type)
                    if bond_type == Chem.BondType.AROMATIC and _atom_key(atoms, i)[:4] == _atom_key(atoms, j)[:4]:
                        component = ccd.get(str(atoms.res_name[i]))
                        names = {str(atoms.atom_name[i]), str(atoms.atom_name[j])}
                        for fact in (() if component is None else component.bonds):
                            if {fact.atom_id_1, fact.atom_id_2} == names:
                                bond_type = {"single": Chem.BondType.SINGLE, "double": Chem.BondType.DOUBLE, "triple": Chem.BondType.TRIPLE}.get(fact.order, bond_type)
                                break
                rw.AddBond(local[i], local[j], bond_type)
                if bond_type == Chem.BondType.AROMATIC:
                    rw.GetAtomWithIdx(local[i]).SetIsAromatic(True)
                    rw.GetAtomWithIdx(local[j]).SetIsAromatic(True)
        molecule = rw.GetMol()
        # Explicit product-state convention: neutral amides, ionized free carboxyl
        # and standard backbone N termini. This is not a pKa predictor.
        for i, r in enumerate(rows):
            atom = molecule.GetAtomWithIdx(i)
            neighbors = list(atom.GetNeighbors())
            def carbonyl(carbon):
                return carbon.GetSymbol() == "C" and any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(carbon).GetSymbol() == "O" for b in carbon.GetBonds())
            if atom.GetSymbol() == "N":
                if any(carbonyl(n) for n in neighbors):
                    atom.SetFormalCharge(0)
                elif str(atoms.mol_type[r]) == "protein" and atoms.atom_name[r] == "N" and (len(neighbors) == 1 or (atoms.res_name[r] == "PRO" and len(neighbors) == 2)):
                    atom.SetFormalCharge(1)
            if atom.GetSymbol() == "O" and len(neighbors) == 1 and carbonyl(neighbors[0]) and molecule.GetBondBetweenAtoms(i, neighbors[0].GetIdx()).GetBondTypeAsDouble() == 1:
                atom.SetFormalCharge(-1)
        try:
            Chem.SanitizeMol(molecule)
            if any(not molecule.GetBondBetweenAtoms(i, j).GetIsAromatic() for i, j in aromatic_pairs):
                raise PostprocessPreparationError("whole-peptide graph contradicts authoritative aromatic bond facts")
            conformer = Chem.Conformer(len(rows))
            for i, r in enumerate(rows):
                conformer.SetAtomPosition(i, atoms.coord[r].astype(float).tolist())
            molecule.AddConformer(conformer)
            Chem.AssignStereochemistryFrom3D(molecule)
            molecule = Chem.AddHs(molecule, addCoords=True)
            Chem.Kekulize(molecule, clearAromaticFlags=True)
            records.append(gaff_record_from_rdkit(molecule, record_id=f"peptide-{ordinal}", molecule_uid=ordinal, component_uid=ordinal, heavy_keys=tuple(_atom_key(atoms, r) for r in rows)))
        except PostprocessPreparationError:
            raise
        except (ValueError, RuntimeError) as error:
            raise PostprocessPreparationError("whole-peptide product chemistry does not sanitize") from error
    return tuple(records)


def minimize_peptides(atoms, *, payload, platform_name, charge_method, fixed_product_record_ids=()):
    import copy
    from dataclasses import replace
    from biotite import structure as struc
    from openmm import CustomExternalForce, Platform, VerletIntegrator, unit
    from openmm.app import ForceField, Modeller, NoCutoff, Simulation
    from openmmforcefields.generators import GAFFTemplateGenerator
    from ..runner import SampleParameterizationPayload, _ambertools_on_path
    from .refine import (
        OpenMMPostprocessError, atom_key, _build_openmm_topology,
        _create_ordinary_system, _run_openmm_minimizer,
        _sample_postprocess_stage,
    )

    row_by_key = {atom_key(atoms, r): r for r in range(len(atoms))}
    if len(row_by_key) != len(atoms):
        raise OpenMMPostprocessError("peptide source heavy keys are not unique")
    private = copy.deepcopy(atoms)
    selected = set()
    selected_moving = set()
    fixed_ids = set(fixed_product_record_ids)
    if not fixed_ids.issubset({r.record_id for r in payload.peptide_records}):
        raise OpenMMPostprocessError("fixed product ID is absent from the private payload")
    private_records = []
    private_chains = set()
    for ordinal, record in enumerate(payload.peptide_records):
        if any(k not in row_by_key for k in record.heavy_keys):
            raise OpenMMPostprocessError("peptide payload atom absent from final structure")
        rows = [row_by_key[k] for k in record.heavy_keys]
        if selected.intersection(rows):
            raise OpenMMPostprocessError("peptide payload duplicates an atom")
        selected.update(rows)
        if record.record_id not in fixed_ids:
            selected_moving.update(rows)
        chain = f"PG{ordinal}"
        while chain in set(map(str, private.chain_id)):
            chain = "X" + chain
        private_chains.add(chain)
        private.chain_id[rows] = chain
        private.res_id[rows] = 1
        private.res_name[rows] = "PEP"
        private.ins_code[rows] = ""
        for i, r in enumerate(rows):
            private.atom_name[r] = f"X{i}"
        private_records.append(replace(record, heavy_keys=tuple(atom_key(private, r) for r in rows)))
    # Add only a private terminal carboxyl OXT required by the receptor template.
    # Its coordinates are fixed and it is never emitted in the public CIF.
    for r in range(len(atoms)):
        if r in selected or atoms.mol_type[r] != "protein" or atoms.atom_name[r] != "C":
            continue
        same = (atoms.chain_id == atoms.chain_id[r]) & (atoms.res_id == atoms.res_id[r]) & (atoms.ins_code == atoms.ins_code[r])
        neighbors, _ = atoms.bonds.get_bonds(r)
        if any(atoms.atom_name[n] == "N" and not same[n] for n in neighbors) or np.any(same & (atoms.atom_name == "OXT")):
            continue
        ca, ox = np.flatnonzero(same & (atoms.atom_name == "CA")), np.flatnonzero(same & (atoms.atom_name == "O"))
        if len(ca) != 1 or len(ox) != 1:
            raise OpenMMPostprocessError("cannot complete private receptor carboxyl terminus")
        v, w = atoms.coord[ca[0]] - atoms.coord[r], atoms.coord[ox[0]] - atoms.coord[r]
        if min(np.linalg.norm(v), np.linalg.norm(w)) < 1e-6:
            raise OpenMMPostprocessError("degenerate receptor terminal frame")
        direction = -(v / np.linalg.norm(v) + w / np.linalg.norm(w))
        if np.linalg.norm(direction) < 1e-6:
            raise OpenMMPostprocessError("degenerate receptor terminal frame")
        extra = atoms[[r]].copy()
        extra.atom_name[0], extra.element[0] = "OXT", "O"
        extra.coord[0] = atoms.coord[r] + 1.25 * direction / np.linalg.norm(direction)
        n = len(private)
        private = private + extra
        private.bonds.add_bond(r, n, struc.BondType.SINGLE)
    private_payload = SampleParameterizationPayload(
        payload.task_name, payload.sample_id,
        tuple(private_records) + tuple(payload.gaff_ligands),
        fixed_coordination_records=payload.fixed_coordination_records)
    with _ambertools_on_path(), _sample_postprocess_stage("peptide topology"):
        topology, positions, molecules, keys = _build_openmm_topology(private, private_payload)
    from .scaffold import split_private_backbone_breaks
    from .preparation import repair_standard_heavy_atoms, add_polymer_hydrogens
    from .native import prepare_native_nucleic, native_hydrogen_variants
    with _sample_postprocess_stage("peptide native context repair"):
        topology = split_private_backbone_breaks(topology)
        topology, positions = repair_standard_heavy_atoms(topology, positions)
    forcefield = ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    with _sample_postprocess_stage("peptide nucleic endpoints"):
        topology, positions = prepare_native_nucleic(topology, positions, forcefield)
    if payload.cobalt_hexammine_groups:
        from .cobalt import prepare_cobalt_hexammine
        with _sample_postprocess_stage("cobalt hexammine parameterization"):
            topology, positions = prepare_cobalt_hexammine(
                topology, positions, forcefield,
                heavy_groups=payload.cobalt_hexammine_groups)
    if payload.fixed_coordination_records:
        from .coordination import prepare_fixed_coordination
        with _sample_postprocess_stage("fixed coordination parameterization"):
            topology, positions, _ = prepare_fixed_coordination(
                topology, positions, forcefield, payload.fixed_coordination_records)
    modeller = Modeller(topology, positions)
    with _sample_postprocess_stage("peptide hydrogenation"):
        add_polymer_hydrogens(
            modeller, forcefield=None, platform=Platform.getPlatformByName("CPU"),
            variants=native_hydrogen_variants(topology, forcefield))
    with _ambertools_on_path():
        system = _create_ordinary_system(
            forcefield, modeller, molecules, GAFFTemplateGenerator,
            charge_method=charge_method, nonbonded_method=NoCutoff, constraints=None)
    restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", 10 * unit.kilocalories_per_mole / unit.angstrom**2)
    for name in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    moving = selected_moving | {row_by_key[key] for key in payload.native_moving_keys}
    moving_residues = {atom_key(private, row)[:4] for row in moving}
    private_rows = {k: r for r, k in enumerate(keys)}
    mapping = {}
    for atom, position in zip(modeller.topology.atoms(), modeller.positions):
        key = (atom.residue.chain.id, int(atom.residue.id), atom.residue.insertionCode, atom.residue.name, atom.name)
        if key[:4] not in moving_residues:
            system.setParticleMass(atom.index, 0)
        if key in private_rows and private_rows[key] in moving:
            mapping[atom.index] = private_rows[key]
            restraint.addParticle(atom.index, position.value_in_unit(unit.nanometer))
    if set(mapping.values()) != moving or len(mapping) != len(moving):
        raise OpenMMPostprocessError("peptide hydrogenation changed original heavy atom mapping")
    mobile = [i for i in range(system.getNumParticles())
              if system.getParticleMass(i).value_in_unit(unit.dalton) > 0]
    if not mobile:
        raise OpenMMPostprocessError("private product minimization has no movable design atoms")
    system.addForce(restraint)
    from .stability import guard_zero_lj_exceptions
    guard_zero_lj_exceptions(system, modeller.topology)
    properties = {"DeviceIndex": "0", "Precision": "mixed", "UseBlockingSync": "true"} if platform_name == "CUDA" else {"Threads": "1"}
    with _sample_postprocess_stage("peptide CUDA/CPU context"):
        simulation = Simulation(modeller.topology, system, VerletIntegrator(0.001 * unit.picoseconds), Platform.getPlatformByName(platform_name), properties)
        simulation.context.setPositions(modeller.positions)
        initial = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not np.isfinite(initial):
        raise OpenMMPostprocessError("whole-peptide initial energy is non-finite before minimization")
    _run_openmm_minimizer(simulation, unit)
    with _sample_postprocess_stage("peptide final state"):
        state = simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        final = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        xyz = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    if not np.isfinite(xyz).all() or not np.isfinite(forces).all() or not np.isfinite([initial, final]).all():
        raise OpenMMPostprocessError("whole-peptide minimization returned non-finite numerical output")
    refined = copy.deepcopy(atoms)
    for particle, row in mapping.items():
        refined.coord[row] = xyz[particle] * 10
    return refined, float(initial), float(final), float(np.sqrt(np.mean(np.sum(forces[mobile]**2, axis=-1))))


def write_peptide_coordinates(base_cif, refined_cif, original, refined):
    """Preserve every public chemical category; replace only verified atom coordinates."""
    import os
    import tempfile
    from pathlib import Path
    from biotite.structure.io.pdbx import CIFFile, CIFColumn
    from ...utils.reader import read_structure
    from .refine import OpenMMPostprocessError, atom_key

    destination = Path(refined_cif)
    if destination.exists() or destination.is_symlink():
        raise OpenMMPostprocessError("refined destination already exists")
    cif = CIFFile.read(base_cif)
    site = cif.block["atom_site"]
    coords = np.column_stack([site["Cartn_" + d].as_array(float) for d in "xyz"])
    if coords.shape != original.coord.shape or not np.allclose(coords, original.coord, rtol=0, atol=1e-4):
        raise OpenMMPostprocessError("base CIF atom row mapping is ambiguous")
    for axis, d in enumerate("xyz"):
        site["Cartn_" + d] = CIFColumn([f"{value:.6f}" for value in refined.coord[:, axis]])
    fd, temporary = tempfile.mkstemp(prefix=".peptide-", suffix=".cif", dir=destination.parent)
    os.close(fd)
    try:
        cif.write(temporary)
        check = read_structure(temporary, ccd_path=None)
        if [atom_key(check, r) for r in range(len(check))] != [atom_key(original, r) for r in range(len(original))]:
            raise OpenMMPostprocessError("peptide refined readback changed identities")
        if not np.array_equal(check.bonds.as_array(), original.bonds.as_array()) or not np.allclose(check.coord, refined.coord, rtol=0, atol=1e-4):
            raise OpenMMPostprocessError("peptide refined readback changed graph or coordinates")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
