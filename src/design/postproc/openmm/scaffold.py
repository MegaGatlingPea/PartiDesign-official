"""Unlinked Amber receptor reference used outside a local product region."""
from __future__ import annotations

from collections import deque
import math
import numpy as np


def complete_reference_termini(atoms):
    """Add private protein carboxyl oxygen for Amber template matching only."""
    from biotite.structure import BondType
    from .refine import OpenMMPostprocessError

    result = atoms.copy()
    for r in range(len(atoms)):
        if atoms.mol_type[r] != "protein" or atoms.atom_name[r] != "C":
            continue
        same = ((atoms.chain_id == atoms.chain_id[r]) & (atoms.res_id == atoms.res_id[r])
                & (atoms.ins_code == atoms.ins_code[r]))
        neighbors, orders = atoms.bonds.get_bonds(r)
        if (any(not same[n] and int(order) == int(BondType.SINGLE)
                and str(atoms.element[n]).upper() in {"C", "N", "O", "S"}
                for n, order in zip(neighbors, orders))
                or np.any(same & (atoms.atom_name == "OXT"))):
            continue
        ca, oxygen = np.flatnonzero(same & (atoms.atom_name == "CA")), np.flatnonzero(same & (atoms.atom_name == "O"))
        if len(ca) != 1 or len(oxygen) != 1:
            raise OpenMMPostprocessError("private receptor terminus lacks an unambiguous backbone frame")
        v, w = atoms.coord[ca[0]] - atoms.coord[r], atoms.coord[oxygen[0]] - atoms.coord[r]
        if min(np.linalg.norm(v), np.linalg.norm(w)) < 1e-6:
            raise OpenMMPostprocessError("private receptor terminus frame is degenerate")
        direction = -(v / np.linalg.norm(v) + w / np.linalg.norm(w))
        if np.linalg.norm(direction) < 1e-6:
            raise OpenMMPostprocessError("private receptor terminus frame is degenerate")
        extra = atoms[[r]].copy()
        extra.atom_name[0], extra.element[0] = "OXT", "O"
        extra.coord[0] = atoms.coord[r] + 1.25 * direction / np.linalg.norm(direction)
        n = len(result)
        result = result + extra
        result.bonds.add_bond(r, n, BondType.SINGLE)
    return result


def _exception_distances(topology, count):
    from .refine import OpenMMPostprocessError
    if topology.getNumAtoms() != count:
        raise OpenMMPostprocessError("Amber topology and particle inventory disagree")
    graph = [set() for _ in range(count)]
    for i, j in topology.bonds():
        graph[i.index].add(j.index)
        graph[j.index].add(i.index)
    expected = {}
    for start in range(len(graph)):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            row = queue.popleft()
            if distances[row] == 3:
                continue
            for neighbor in graph[row]:
                if neighbor not in distances:
                    distances[neighbor] = distances[row] + 1
                    queue.append(neighbor)
        expected.update({(start, end): distance for end, distance in distances.items() if end > start})
    return expected


def rebuild_amber_exceptions(system, topology):
    """Use shortest FINAL graph distance, not individual torsion paths.

    A ring can have a three-bond torsion path between atoms also separated by
    two bonds. Some ParmEd conversions turn that pair into a 1-4 interaction;
    Amber requires the shorter path's complete exclusion instead.
    """
    from openmm import NonbondedForce, unit
    from .refine import OpenMMPostprocessError

    forces = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
    if len(forces) != 1:
        raise OpenMMPostprocessError("hybrid Amber system requires one nonbonded force")
    nb = forces[0]
    expected = _exception_distances(topology, nb.getNumParticles())
    existing = {}
    for index in range(nb.getNumExceptions()):
        i, j, *_ = nb.getExceptionParameters(index)
        key = tuple(sorted((int(i), int(j))))
        if key in existing or key not in expected:
            raise OpenMMPostprocessError("unexpected hybrid exception outside final bond graph")
        existing[key] = index
    for (i, j), distance in expected.items():
        qi, si, ei = nb.getParticleParameters(i)
        qj, sj, ej = nb.getParticleParameters(j)
        sigma = (si + sj) / 2
        if distance < 3:
            q, epsilon = 0 * unit.elementary_charge**2, 0 * unit.kilojoule_per_mole
        else:
            q = qi * qj / 1.2
            epsilon = math.sqrt(ei.value_in_unit(unit.kilojoule_per_mole) * ej.value_in_unit(unit.kilojoule_per_mole)) / 2 * unit.kilojoule_per_mole
        if (i, j) in existing:
            nb.setExceptionParameters(existing[i, j], i, j, q, sigma, epsilon)
        else:
            nb.addException(i, j, q, sigma, epsilon)


def audit_amber_exceptions(system, topology):
    """Prove native Amber 1-2/1-3 exclusions and 1-4 scaling before rebuilding."""
    from openmm import NonbondedForce, unit
    from .refine import OpenMMPostprocessError

    forces = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
    if len(forces) != 1:
        raise OpenMMPostprocessError("Amber scaffold requires one nonbonded force")
    nb = forces[0]
    expected = _exception_distances(topology, nb.getNumParticles())
    observed = {}
    for index in range(nb.getNumExceptions()):
        i, j, q, sigma, epsilon = nb.getExceptionParameters(index)
        key = tuple(sorted((int(i), int(j))))
        if key in observed or key not in expected:
            raise OpenMMPostprocessError("Amber scaffold has duplicate or nonstandard exclusions")
        observed[key] = True
        qv = q.value_in_unit(unit.elementary_charge**2)
        ev = epsilon.value_in_unit(unit.kilojoule_per_mole)
        if expected[key] < 3:
            valid = abs(qv) < 1e-9 and abs(ev) < 1e-9
        else:
            qi, si, ei = nb.getParticleParameters(i)
            qj, sj, ej = nb.getParticleParameters(j)
            wanted_q = qi.value_in_unit(unit.elementary_charge) * qj.value_in_unit(unit.elementary_charge) / 1.2
            wanted_e = math.sqrt(ei.value_in_unit(unit.kilojoule_per_mole) * ej.value_in_unit(unit.kilojoule_per_mole)) / 2
            wanted_s = (si.value_in_unit(unit.nanometer) + sj.value_in_unit(unit.nanometer)) / 2
            valid = (math.isclose(qv, wanted_q, abs_tol=1e-8, rel_tol=1e-6)
                     and math.isclose(ev, wanted_e, abs_tol=1e-8, rel_tol=1e-6)
                     and math.isclose(sigma.value_in_unit(unit.nanometer), wanted_s, abs_tol=1e-8, rel_tol=1e-6))
        if not valid:
            raise OpenMMPostprocessError(f"Amber scaffold uses nonstandard exclusion/scaling at {key}: distance={expected[key]}, q={qv}, epsilon={ev}")
    if set(observed) != set(expected):
        raise OpenMMPostprocessError("Amber scaffold exclusion graph is incomplete")


def split_private_backbone_breaks(topology):
    """Expose genuine polymer breaks to Modeller's terminal H rules.

    Chain numbering is ignored.  Protein C--N and nucleic O3'--P continuity
    are accepted only when the graph edge is present.  Nonpolymer residues
    and ions are isolated into private chain segments.  Atom order, IDs,
    residue metadata, and every original bond are preserved.
    """
    from openmm.app import Topology

    protein_names = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL",
    }
    nucleic_names = {
        "A", "C", "G", "U", "I", "DA", "DC", "DG", "DT", "DI", "DU",
        "RA", "RC", "RG", "RU",
    }

    def residue_kind(residue):
        name = str(residue.name).upper()
        if name in protein_names:
            return "protein"
        if name in nucleic_names:
            return "nucleic"
        return "nonpolymer"

    def named(residue, name):
        return next((atom for atom in residue.atoms() if atom.name == name), None)

    result = Topology()
    bonds = list(topology.bonds())
    edges = {frozenset((a.index, b.index)) for a, b in bonds}
    mapped = {}
    for chain in topology.chains():
        previous = None
        previous_kind = None
        target_chain = None
        for residue in chain.residues():
            kind = residue_kind(residue)
            split = (
                previous is None
                or kind == "nonpolymer"
                or previous_kind == "nonpolymer"
                or kind != previous_kind
            )
            if not split and kind == "protein":
                left = named(previous, "C")
                right = named(residue, "N")
                split = (
                    left is None
                    or right is None
                    or frozenset((left.index, right.index)) not in edges
                )
            elif not split and kind == "nucleic":
                split = True
                for left_name, right_name in (
                    ("O3'", "P"),
                    ("O3*", "P"),
                    ("C3'", "P"),
                    ("C3*", "P"),
                ):
                    left = named(previous, left_name)
                    right = named(residue, right_name)
                    if (
                        left is not None
                        and right is not None
                        and frozenset((left.index, right.index)) in edges
                    ):
                        split = False
                        break
            if split:
                target_chain = result.addChain(chain.id)
            target_residue = result.addResidue(
                residue.name,
                target_chain,
                residue.id,
                residue.insertionCode,
            )
            for atom in residue.atoms():
                mapped[atom.index] = result.addAtom(
                    atom.name,
                    atom.element,
                    target_residue,
                    atom.id,
                    atom.formalCharge,
                )
            previous = residue
            previous_kind = kind
    for bond in bonds:
        result.addBond(
            mapped[bond[0].index],
            mapped[bond[1].index],
            bond.type,
            bond.order,
        )
    result.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    return result

def parameterize_scaffold(atoms):
    """Parameterize supplied receptor alone; do not change public coordinates.

    The reference may represent the unlinked site because every term touching
    that site will be replaced by product parameters. No unknown residue is
    silently converted to a standard residue.
    """
    import parmed as pmd
    from openmm import Platform
    from openmm.app import ForceField, Modeller, NoCutoff
    from ..runner import SampleParameterizationPayload
    from .native import native_hydrogen_variants, prepare_native_nucleic
    from .preparation import add_polymer_hydrogens, repair_standard_heavy_atoms
    from .refine import _build_openmm_topology, _sample_postprocess_stage

    from .protein import prepare_native_protein
    with _sample_postprocess_stage("Amber receptor topology"):
        top, xyz, _, _ = _build_openmm_topology(atoms, SampleParameterizationPayload("private", "scaffold"))
        top = split_private_backbone_breaks(top)
        top, xyz = repair_standard_heavy_atoms(top, xyz)
        forcefield = ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
        top, xyz = prepare_native_protein(top, xyz, forcefield)
        top, xyz = prepare_native_nucleic(top, xyz, forcefield)
        variants = native_hydrogen_variants(top, forcefield)
    modeller = Modeller(top, xyz)
    cpu = Platform.getPlatformByName("CPU")
    previous_threads = cpu.getPropertyDefaultValue("Threads")
    with _sample_postprocess_stage("Amber receptor hydrogenation"):
        try:
            cpu.setPropertyDefaultValue("Threads", "1")
            add_polymer_hydrogens(
                modeller,
                forcefield=forcefield,
                platform=cpu,
                variants=variants,
            )
        finally:
            cpu.setPropertyDefaultValue("Threads", previous_threads)
    with _sample_postprocess_stage("Amber receptor parameterization"):
        system = forcefield.createSystem(
            modeller.topology, nonbondedMethod=NoCutoff, constraints=None, rigidWater=False,
        )
    audit_amber_exceptions(system, modeller.topology)
    parm = pmd.openmm.load_topology(modeller.topology, system, xyz=modeller.positions)
    # load_topology expresses Amber's 1-4s as explicit adjustments. They have
    # just been independently checked; regenerate them from the FINAL graph,
    # so newly created cross-region paths also receive the same scaling.
    parm.adjusts.clear()
    parm.adjust_types.clear()
    for torsion in parm.dihedrals:
        torsion.type.scee = 1.2
        torsion.type.scnb = 2.0
    return parm, modeller.topology
