"""Native Amber nucleic-acid endpoints in the private parameterization view.

5'-monophosphate parameters follow AmberTools terminal_monophosphate.lib:
OP3/OH -0.621, HOP3/HO +0.3129, and terminal DNA P 1.1661.
This is the explicit mono-protonated terminal-phosphate convention, not a
pKa predictor. Original heavy atoms, positions, names and bonds are preserved.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

_NUCLEIC = frozenset(("A", "C", "G", "U", "DA", "DC", "DG", "DT"))


def _external_names(residue, bonds):
    result = set()
    for a, b in bonds:
        if a.residue is residue and b.residue is not residue:
            result.add(a.name)
        if b.residue is residue and a.residue is not residue:
            result.add(b.name)
    return result


def _template_name(residue, bonds):
    names = {a.name for a in residue.atoms() if a.element is not None and a.element.symbol != "H"}
    external = _external_names(residue, bonds)
    outgoing = "O3'" in external
    incoming = "P" in external
    if "P" in names:
        if incoming:
            return residue.name if outgoing else residue.name + "3"
        return "_PD_" + residue.name + ("5P" if outgoing else "5P3")
    return residue.name + ("5" if outgoing else "N")


def _register_phosphate_template(forcefield, residue_name, outgoing):
    from openmm.app import ForceField, element
    name = "_PD_" + residue_name + ("5P" if outgoing else "5P3")
    if name in forcefield._templates:
        return
    base = residue_name if outgoing else residue_name + "3"
    template = deepcopy(forcefield._templates[base])
    template.name = name
    p_index = template.atomIndices["P"]
    template.externalBonds = [i for i in template.externalBonds if i != p_index]
    template.atoms[p_index].externalBonds = 0
    prefix = "DNA" if residue_name.startswith("D") else "RNA"
    if prefix == "DNA":
        template.atoms[p_index].parameters["charge"] = 1.1661
    template.addAtom(ForceField._TemplateAtomData(
        "OP3", prefix + "-OH", element.oxygen, {"charge": -0.621}))
    template.addAtom(ForceField._TemplateAtomData(
        "HOP3", prefix + "-HO", element.hydrogen, {"charge": 0.3129}))
    template.addBondByName("P", "OP3")
    template.addBondByName("OP3", "HOP3")
    forcefield.registerResidueTemplate(template)


def prepare_native_nucleic(topology, positions, forcefield):
    """Complete only recognizable canonical terminal phosphate in a private copy."""
    from openmm import unit
    from openmm.app import Topology, element
    from .refine import OpenMMPostprocessError
    bonds = list(topology.bonds())
    adjacency = {a.index: [] for a in topology.atoms()}
    for a, b in bonds:
        adjacency[a.index].append(b)
        adjacency[b.index].append(a)
    xyz = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
    additions = {}
    for residue in topology.residues():
        if residue.name not in _NUCLEIC:
            continue
        atoms = {a.name: a for a in residue.atoms()}
        if "P" not in atoms or "P" in _external_names(residue, bonds):
            continue
        p = atoms["P"]
        expected = {"OP1", "OP2", "O5'"}
        neighbors = adjacency[p.index]
        heavy = {a.name for a in neighbors if a.element != element.hydrogen}
        if any(a.element == element.hydrogen for a in neighbors):
            raise OpenMMPostprocessError("canonical terminal phosphate cannot contain a P-H bond")
        if heavy not in (expected, expected | {"OP3"}):
            raise OpenMMPostprocessError("terminal phosphate has unsupported heavy connectivity")
        outgoing = "O3'" in _external_names(residue, bonds)
        _register_phosphate_template(forcefield, residue.name, outgoing)
        if "OP3" not in atoms:
            vectors = np.array([xyz[atoms[n].index] - xyz[p.index] for n in sorted(expected)])
            lengths = np.linalg.norm(vectors, axis=1)
            if np.any(lengths < 1e-8):
                raise OpenMMPostprocessError("terminal phosphate oxygen frame is degenerate")
            direction = -(vectors / lengths[:, None]).sum(axis=0)
            if np.linalg.norm(direction) < 1e-8:
                direction = np.cross(vectors[0], vectors[1])
            if np.linalg.norm(direction) < 1e-8:
                raise OpenMMPostprocessError("terminal phosphate oxygen frame is degenerate")
            additions[residue.index] = xyz[p.index] + 0.16 * direction / np.linalg.norm(direction)
    if not additions:
        return topology, positions
    rebuilt = Topology()
    mapped = {}
    coordinates = []
    extra = {}
    for chain in topology.chains():
        new_chain = rebuilt.addChain(chain.id)
        for residue in chain.residues():
            new_residue = rebuilt.addResidue(residue.name, new_chain, residue.id, residue.insertionCode)
            for atom in residue.atoms():
                mapped[atom.index] = rebuilt.addAtom(atom.name, atom.element, new_residue, atom.id)
                coordinates.append(xyz[atom.index])
            if residue.index in additions:
                extra[residue.index] = rebuilt.addAtom(
                    "OP3", element.oxygen, new_residue, id="private:terminal-phosphate")
                coordinates.append(additions[residue.index])
    for bond in bonds:
        rebuilt.addBond(mapped[bond[0].index], mapped[bond[1].index], bond.type, bond.order)
    for residue in topology.residues():
        if residue.index in extra:
            p = next(a for a in residue.atoms() if a.name == "P")
            rebuilt.addBond(mapped[p.index], extra[residue.index], order=1)
    rebuilt.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    return rebuilt, unit.Quantity(np.array(coordinates), unit.nanometer)


def native_hydrogen_variants(topology, forcefield):
    """Use exact native H inventories, independent of author-chain position."""
    from openmm.app import element
    from .refine import OpenMMPostprocessError
    bonds = list(topology.bonds())
    variants = [None] * topology.getNumResidues()
    for residue in topology.residues():
        if residue.name not in _NUCLEIC:
            continue
        name = _template_name(residue, bonds)
        if name not in forcefield._templates:
            raise OpenMMPostprocessError("native nucleic endpoint template is unavailable")
        template = forcefield._templates[name]
        actual = {a.name for a in residue.atoms() if a.element != element.hydrogen}
        expected = {a.name for a in template.atoms if a.element != element.hydrogen}
        if actual != expected:
            raise OpenMMPostprocessError(
                f"native nucleic heavy inventory mismatch for {residue.name}: "
                f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
        hydrogen_pairs = []
        for i, j in template.bonds:
            a, b = template.atoms[i], template.atoms[j]
            if a.element == element.hydrogen:
                hydrogen_pairs.append((a.name, b.name))
            elif b.element == element.hydrogen:
                hydrogen_pairs.append((b.name, a.name))
        variants[residue.index] = hydrogen_pairs
    return variants
