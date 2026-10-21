"""Graph-defined protein ends after private heavy-atom completion.

Isolated standard residues use a generic combination of Amber N/C terminal
charge increments.  This is an approximate zwitterion model, not a fitted free
amino-acid force field.  Ordinary peptide templates are unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from .preparation import _PROTEIN


def _register_isolated_template(forcefield, base):
    from openmm.app import ForceField, element

    name = "_PD_NC_" + base
    if name in forcefield._templates:
        return
    internal, nterm, cterm = (forcefield._templates[key] for key in (base, "N" + base, "C" + base))
    inventories = [{atom.name: atom for atom in template.atoms} for template in (internal, nterm, cterm)]
    ia, na, ca = inventories
    if set(ca) - set(na) != {"OXT"} | (set(ia) - set(na)):
        raise ValueError("incompatible Amber terminal atom inventories")
    missing = set(ia) - set(na)
    n_hydrogens = [nterm.atoms[j].name for i, j in nterm.bonds
                   if nterm.atoms[i].name == "N" and nterm.atoms[j].element == element.hydrogen]
    n_hydrogens += [nterm.atoms[i].name for i, j in nterm.bonds
                    if nterm.atoms[j].name == "N" and nterm.atoms[i].element == element.hydrogen]
    correction = sum(float(ca[key].parameters["charge"]) - float(ia[key].parameters["charge"])
                     for key in missing)
    if missing and (not n_hydrogens or any(ia[key].element != element.hydrogen for key in missing)):
        raise ValueError("unsupported Amber N-terminal inventory change")
    template = ForceField._TemplateData(name)
    for atom in nterm.atoms:
        parameters = deepcopy(atom.parameters)
        if atom.name in ia and atom.name in ca:
            parameters["charge"] += ca[atom.name].parameters["charge"] - ia[atom.name].parameters["charge"]
        if atom.name in n_hydrogens:
            parameters["charge"] += correction / len(n_hydrogens)
        atom_type = ca[atom.name].type if atom.name in {"C", "O"} else atom.type
        template.addAtom(ForceField._TemplateAtomData(atom.name, atom_type, atom.element, parameters))
    oxygen = ca["OXT"]
    template.addAtom(ForceField._TemplateAtomData("OXT", oxygen.type, oxygen.element, deepcopy(oxygen.parameters)))
    for i, j in nterm.bonds:
        template.addBondByName(nterm.atoms[i].name, nterm.atoms[j].name)
    template.addBondByName("C", "OXT")
    # Equivalent terminal oxygen sites must have identical charges regardless
    # of the arbitrary O/OXT graph match. Preserve their total charge.
    oxygen_atoms = [template.atoms[template.atomIndices[key]] for key in ("O", "OXT")]
    oxygen_charge = sum(float(atom.parameters["charge"]) for atom in oxygen_atoms) / 2
    for atom in oxygen_atoms:
        atom.parameters["charge"] = oxygen_charge
    for i in nterm.externalBonds:
        if nterm.atoms[i].name != "C":
            template.addExternalBondByName(nterm.atoms[i].name)
    charge = sum(float(atom.parameters["charge"]) for atom in template.atoms)
    expected = sum(float(atom.parameters["charge"]) for atom in nterm.atoms) + sum(float(atom.parameters["charge"]) for atom in cterm.atoms) - sum(float(atom.parameters["charge"]) for atom in internal.atoms)
    if not np.isfinite(charge) or not np.isclose(charge, expected, atol=1e-6, rtol=0):
        raise ValueError("isolated Amber residue charge is not conserved")
    forcefield.registerResidueTemplate(template)


def prepare_native_protein(topology, positions, forcefield):
    """Complete private OXT and recognize residues with both backbone ends open.

    Must run AFTER missing-heavy repair.  Existing atom IDs, coordinates and
    edges survive.  Terminal states follow actual external bonds, never author
    numbering, geometric proximity, or an assumed chain end.
    """
    from openmm import unit
    from openmm.app import Topology, element

    bonds = list(topology.bonds())
    external = {residue.index: set() for residue in topology.residues()}
    for a, b in bonds:
        if a.residue is not b.residue:
            external[a.residue.index].add(a.name)
            external[b.residue.index].add(b.name)
    xyz = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
    additions = {}
    isolated_heavy = set()
    for residue in topology.residues():
        if residue.name not in _PROTEIN:
            continue
        atoms = {atom.name: atom for atom in residue.atoms()}
        if "C" in external[residue.index]:
            continue
        if not {"N", "CA", "C", "O"}.issubset(atoms):
            raise ValueError("repaired protein end lacks a complete backbone frame")
        if "OXT" not in atoms:
            center = xyz[atoms["C"].index]
            vectors = np.asarray([xyz[atoms[key].index] - center for key in ("CA", "O")])
            lengths = np.linalg.norm(vectors, axis=1)
            if np.any(lengths < 1e-8):
                raise ValueError("protein terminal frame is degenerate")
            direction = -(vectors / lengths[:, None]).sum(axis=0)
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                raise ValueError("protein terminal frame is degenerate")
            additions[residue.index] = center + .125 * direction / norm
        if "N" not in external[residue.index]:
            isolated_heavy.add(frozenset({atom.name for atom in residue.atoms()
                                          if atom.element != element.hydrogen} - {"OXT"}))
    if isolated_heavy:
        for base, template in list(forcefield._templates.items()):
            if "N" + base not in forcefield._templates or "C" + base not in forcefield._templates:
                continue
            heavy = frozenset(atom.name for atom in template.atoms if atom.element != element.hydrogen)
            if heavy in isolated_heavy:
                _register_isolated_template(forcefield, base)
    if not additions:
        return topology, positions
    rebuilt = Topology()
    mapped, coordinates, extra = {}, [], {}
    for chain in topology.chains():
        target_chain = rebuilt.addChain(chain.id)
        for residue in chain.residues():
            target = rebuilt.addResidue(residue.name, target_chain, residue.id, residue.insertionCode)
            for atom in residue.atoms():
                mapped[atom.index] = rebuilt.addAtom(atom.name, atom.element, target, atom.id, atom.formalCharge)
                coordinates.append(xyz[atom.index])
            if residue.index in additions:
                extra[residue.index] = rebuilt.addAtom("OXT", element.oxygen, target, id="private:protein-terminus")
                coordinates.append(additions[residue.index])
    for bond in bonds:
        rebuilt.addBond(mapped[bond[0].index], mapped[bond[1].index], bond.type, bond.order)
    for residue in topology.residues():
        if residue.index in extra:
            carbon = next(atom for atom in residue.atoms() if atom.name == "C")
            rebuilt.addBond(mapped[carbon.index], extra[residue.index], order=1)
    rebuilt.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    return rebuilt, unit.Quantity(np.asarray(coordinates), unit.nanometer)
