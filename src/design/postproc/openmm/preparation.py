"""Native-heavy repair and graph-aware polymer hydrogen preparation."""

from __future__ import annotations

from collections import defaultdict
from io import StringIO
from typing import Any

import numpy as np


_PROTEIN = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)
_NUCLEIC = frozenset(
    "A C G U I DA DC DG DT DI DU RA RC RG RU".split()
)
_STANDARD = _PROTEIN | _NUCLEIC
_FORBIDDEN_TERMINAL = frozenset(("OXT", "OP3"))
_CCD_ORDER_VALUES = {
    "single": 1.0,
    "double": 2.0,
    "triple": 3.0,
    "aromatic": 1.5,
}


def _position_data(positions: Any, count: int):
    from openmm import unit

    if hasattr(positions, "value_in_unit") and hasattr(positions, "unit"):
        output_unit = positions.unit
        output = np.asarray(positions.value_in_unit(output_unit), dtype=float)
        nanometer = unit.Quantity(
            np.asarray(positions.value_in_unit(unit.nanometer), dtype=float),
            unit.nanometer,
        )
    else:
        output_unit = unit.nanometer
        output = np.asarray([tuple(value) for value in positions], dtype=float)
        nanometer = unit.Quantity(output, unit.nanometer)
    if output.shape != (count, 3) or not np.isfinite(output).all():
        raise ValueError("positions must be finite coordinates with shape (N, 3)")
    return output_unit, output, nanometer


def _serial_copy(topology):
    from openmm.app import Topology

    result = Topology()
    result.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    mapped = {}
    for ci, chain in enumerate(topology.chains()):
        target_chain = result.addChain(f"chain{ci}")
        for ri, residue in enumerate(chain.residues()):
            target_residue = result.addResidue(
                residue.name, target_chain, id=str(ri + 1), insertionCode=""
            )
            for atom in residue.atoms():
                mapped[atom] = result.addAtom(
                    atom.name,
                    atom.element,
                    target_residue,
                    id=f"src:{atom.index}",
                    formalCharge=atom.formalCharge,
                )
    for bond in topology.bonds():
        result.addBond(mapped[bond[0]], mapped[bond[1]], bond.type, bond.order)
    return result


def _slots(topology):
    slots = {}
    for ci, chain in enumerate(topology.chains()):
        for ri, residue in enumerate(chain.residues()):
            for atom in residue.atoms():
                slot = (ci, ri, atom.name)
                if slot in slots:
                    raise ValueError("duplicate atom name in an OpenMM residue")
                slots[slot] = atom
    return slots


def _residue_slot(residue):
    return residue.chain.index, list(residue.chain.residues()).index(residue)


def _approved_edge(a, b, fixed):
    if a[:2] == b[:2]:
        return True
    names = {a[2], b[2]}
    left = fixed[a].residue.name
    right = fixed[b].residue.name
    if left in _PROTEIN and right in _PROTEIN and names == {"C", "N"}:
        return True
    return (
        left in _NUCLEIC
        and right in _NUCLEIC
        and names
        in ({"O3'", "P"}, {"O3*", "P"})
    )


def _template_external_edge(left, right):
    """Return whether an inter-residue edge is part of a native template.

    PDBFixer knows only standard peptide and nucleic connectivity.  A fixed
    product can also contain a side-chain or ester edge which is authoritative
    in the input graph.  Such an edge must be allowed to consume a template
    leaving atom, rather than causing PDBFixer to restore that atom first.
    """

    if left.residue is right.residue:
        return True
    left_name = str(left.residue.name).upper()
    right_name = str(right.residue.name).upper()
    names = {str(left.name), str(right.name)}
    if left_name in _PROTEIN and right_name in _PROTEIN:
        if names == {"C", "N"}:
            return True
        if (
            names == {"SG"}
            and left_name == right_name == "CYS"
        ):
            # Preserve the existing disulfide route.  It is a native CYS
            # relation even though it is not a peptide backbone edge.
            return True
    if left_name in _NUCLEIC and right_name in _NUCLEIC:
        if names in (
            {"O3'", "P"},
            {"O3*", "P"},
            {"C3'", "P"},
            {"C3*", "P"},
        ):
            return True
    return False


def _non_template_residue_slots(topology):
    """Return residue slots touched by authoritative non-template edges."""

    protected = set()
    for left, right in topology.bonds():
        if left.residue is right.residue or _template_external_edge(left, right):
            continue
        protected.add(_residue_slot(left.residue))
        protected.add(_residue_slot(right.residue))
    return frozenset(protected)


def _ccd_bond_order(residue_name, left_name, right_name):
    from ...utils.ccd import load_packaged_polymer_ccd

    component = load_packaged_polymer_ccd().get(str(residue_name).upper())
    if component is None:
        raise ValueError(
            f"non-template-linked residue {residue_name!r} has no packaged CCD component"
        )
    pair = {str(left_name), str(right_name)}
    facts = [
        bond
        for bond in component.bonds
        if {str(bond.atom_id_1), str(bond.atom_id_2)} == pair
    ]
    if len(facts) != 1 or facts[0].order not in _CCD_ORDER_VALUES:
        raise ValueError(
            f"non-template-linked residue {residue_name!r} has an ambiguous CCD bond "
            f"for {left_name!r}-{right_name!r}"
        )
    return _CCD_ORDER_VALUES[facts[0].order]


def _ccd_atom_valence(residue_name, atom_name):
    from ...utils.ccd import load_packaged_polymer_ccd

    component = load_packaged_polymer_ccd().get(str(residue_name).upper())
    if component is None:
        raise ValueError(
            f"non-template-linked residue {residue_name!r} has no packaged CCD component"
        )
    facts = [
        bond
        for bond in component.bonds
        if str(atom_name) in {str(bond.atom_id_1), str(bond.atom_id_2)}
    ]
    if not facts:
        raise ValueError(
            f"non-template-linked residue {residue_name!r} has no CCD bonds for {atom_name!r}"
        )
    return sum(
        _ccd_bond_order(residue_name, bond.atom_id_1, bond.atom_id_2)
        for bond in facts
    )


def _openmm_bond_order(bond):
    raw = getattr(bond, "order", None)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def _template_neighbors(template, atom):
    neighbors = []
    for bond in template.topology.bonds():
        if bond[0] is atom:
            neighbors.append(bond[1])
        elif bond[1] is atom:
            neighbors.append(bond[0])
    return neighbors


def _validate_consumed_missing_atoms(topology, fixer, residue, missing):
    """Prove that each protected missing atom is consumed by a product edge.

    The proof uses the packaged standard-residue graph and the authoritative
    source graph.  It never creates an inferred atom or bond.  A protected
    residue whose missing inventory cannot be explained exactly by concrete
    non-template edges fails closed.
    """

    ci, ri = _residue_slot(residue)
    source_chains = list(topology.chains())
    source_residue = list(source_chains[ci].residues())[ri]
    source_atoms = _slots(topology)
    template = fixer.templates.get(str(residue.name))
    if template is None:
        raise ValueError(
            f"non-template-linked residue {residue.name!r} has no PDBFixer template"
        )

    missing_by_parent = defaultdict(float)
    missing_names = []
    for atom in missing:
        neighbors = _template_neighbors(template, atom)
        if len(neighbors) != 1:
            raise ValueError(
                f"non-template-linked residue {residue.name!r} missing heavy atom "
                f"{atom.name!r} has ambiguous template connectivity"
            )
        parent = neighbors[0]
        parent_slot = (ci, ri, str(parent.name))
        if parent_slot not in source_atoms:
            raise ValueError(
                f"non-template-linked residue {residue.name!r} missing heavy atom "
                f"{atom.name!r} has no authoritative parent"
            )
        try:
            order = _ccd_bond_order(residue.name, parent.name, atom.name)
        except ValueError as error:
            raise ValueError(
                f"non-template-linked residue {residue.name!r} missing heavy atom "
                f"{atom.name!r} cannot be resolved from CCD"
            ) from error
        missing_by_parent[str(parent.name)] += order
        missing_names.append(str(atom.name))

    for parent_name, missing_order in missing_by_parent.items():
        parent = source_atoms[(ci, ri, parent_name)]
        internal_order = 0.0
        external_order = 0.0
        for bond in topology.bonds():
            if bond[0] is parent:
                other = bond[1]
            elif bond[1] is parent:
                other = bond[0]
            else:
                continue
            order = _openmm_bond_order(bond)
            if order is None:
                raise ValueError(
                    f"non-template-linked residue {residue.name!r} has an ambiguous "
                    f"bond order at {parent_name!r}"
                )
            if other.residue is source_residue:
                internal_order += order
            elif not _template_external_edge(parent, other):
                external_order += order
        try:
            expected_order = _ccd_atom_valence(residue.name, parent_name)
        except ValueError as error:
            raise ValueError(
                f"non-template-linked residue {residue.name!r} has no concrete "
                f"valence proof at {parent_name!r}"
            ) from error
        if not np.isclose(
            internal_order + missing_order,
            expected_order,
            rtol=0,
            atol=1e-6,
        ) or not np.isclose(
            external_order,
            missing_order,
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError(
                f"non-template-linked residue {residue.name!r} has incomplete or "
                f"ambiguous missing heavy atoms: {', '.join(missing_names)}"
            )


def repair_standard_heavy_atoms(topology, positions):
    """Add only missing standard protein/RNA/DNA heavy atoms.

    PDBFixer supplies template coordinates.  The returned topology is rebuilt
    from the original, so original IDs, order, elements, coordinates, bond
    types/orders, and old-old edges remain authoritative.  Missing residues,
    nonstandard replacement, heterogen removal, OXT, OP3, and terminal
    phosphate inference are excluded.
    """

    from openmm import Platform
    from openmm.app import PDBxFile

    count = topology.getNumAtoms()
    if count == 0:
        return topology, positions
    output_unit, original_positions, serialized_positions = _position_data(
        positions, count
    )
    original_slots = _slots(topology)
    protected_slots = _non_template_residue_slots(topology)
    buffer = StringIO()
    PDBxFile.writeFile(
        _serial_copy(topology), serialized_positions, buffer, keepIds=True
    )
    buffer.seek(0)
    try:
        from pdbfixer import PDBFixer
    except ImportError as error:
        raise ImportError(
            "repair_standard_heavy_atoms requires PDBFixer"
        ) from error

    cpu = Platform.getPlatformByName("CPU")
    previous_threads = cpu.getPropertyDefaultValue("Threads")
    try:
        cpu.setPropertyDefaultValue("Threads", "1")
        fixer = PDBFixer(pdbxfile=buffer, platform=cpu)
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()

        approved = {}
        filtered = {}
        for residue, atoms in fixer.missingAtoms.items():
            if residue.name not in _STANDARD:
                continue
            ci, ri = _residue_slot(residue)
            if (ci, ri) in protected_slots:
                _validate_consumed_missing_atoms(topology, fixer, residue, atoms)
                continue
            kept = []
            for atom in atoms:
                if (
                    atom.element is not None
                    and atom.element.atomic_number > 1
                    and atom.name not in _FORBIDDEN_TERMINAL
                ):
                    approved[(ci, ri, atom.name)] = atom.element
                    kept.append(atom)
            if kept:
                filtered[residue] = kept
        fixer.missingAtoms = filtered
        fixer.missingTerminals = {}
        if not approved:
            return topology, positions

        fixer.addMissingAtoms(seed=0)
    finally:
        cpu.setPropertyDefaultValue("Threads", previous_threads)

    fixed = _slots(fixer.topology)
    original_chains = list(topology.chains())
    fixed_chains = list(fixer.topology.chains())
    if len(original_chains) != len(fixed_chains):
        raise ValueError("PDBFixer changed the original chain count")
    for ci, chain in enumerate(original_chains):
        original_residues = list(chain.residues())
        fixed_residues = list(fixed_chains[ci].residues())
        if len(original_residues) != len(fixed_residues):
            raise ValueError("PDBFixer inserted an unexpected residue")
        for ri, residue in enumerate(original_residues):
            fixed_residue = fixed_residues[ri]
            if residue.name != fixed_residue.name:
                raise ValueError("PDBFixer changed an original residue")
            for atom in residue.atoms():
                fixed_atom = fixed.get((ci, ri, atom.name))
                if fixed_atom is None or fixed_atom.element != atom.element:
                    raise ValueError("PDBFixer changed an original atom")

    new_slots = set(fixed).difference(original_slots)
    if new_slots != set(approved):
        raise ValueError("PDBFixer returned an unapproved heavy-atom inventory")
    for slot in new_slots:
        atom = fixed[slot]
        if (
            atom.element is None
            or atom.element.atomic_number <= 1
            or atom.element != approved[slot]
        ):
            raise ValueError("PDBFixer returned a non-heavy or wrong-element atom")

    from openmm import unit
    from openmm.app import Topology

    result = Topology()
    result.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    output_atoms = {}
    output_slots = {}
    result_positions = []
    fixed_positions = np.asarray(
        fixer.positions.value_in_unit(output_unit), dtype=float
    )
    for ci, chain in enumerate(original_chains):
        target_chain = result.addChain(chain.id)
        for ri, residue in enumerate(chain.residues()):
            target_residue = result.addResidue(
                residue.name,
                target_chain,
                id=residue.id,
                insertionCode=residue.insertionCode,
            )
            for atom in residue.atoms():
                target_atom = result.addAtom(
                    atom.name,
                    atom.element,
                    target_residue,
                    id=atom.id,
                    formalCharge=atom.formalCharge,
                )
                output_atoms[atom] = target_atom
                output_slots[(ci, ri, atom.name)] = target_atom
                result_positions.append(original_positions[atom.index])
            for slot in sorted(new_slots, key=lambda value: value[2]):
                if slot[:2] != (ci, ri):
                    continue
                fixed_atom = fixed[slot]
                output_slots[slot] = result.addAtom(
                    fixed_atom.name,
                    fixed_atom.element,
                    target_residue,
                    id=f"private:missing-heavy:{ci}:{ri}:{fixed_atom.name}",
                    formalCharge=fixed_atom.formalCharge,
                )
                result_positions.append(fixed_positions[fixed_atom.index])

    old_edges = set()
    for bond in topology.bonds():
        left, right = output_atoms[bond[0]], output_atoms[bond[1]]
        result.addBond(left, right, bond.type, bond.order)
        old_edges.add(frozenset((left.index, right.index)))

    fixed_atom_slots = {atom: slot for slot, atom in fixed.items()}
    for bond in fixer.topology.bonds():
        left_slot = fixed_atom_slots[bond[0]]
        right_slot = fixed_atom_slots[bond[1]]
        if left_slot in original_slots and right_slot in original_slots:
            continue
        if not _approved_edge(left_slot, right_slot, fixed):
            continue
        if left_slot[:2] in protected_slots or right_slot[:2] in protected_slots:
            continue
        left, right = output_slots[left_slot], output_slots[right_slot]
        edge = frozenset((left.index, right.index))
        if edge not in old_edges:
            result.addBond(left, right, bond.type, bond.order)
            old_edges.add(edge)

    degree = defaultdict(int)
    for bond in result.bonds():
        degree[bond[0].index] += 1
        degree[bond[1].index] += 1
    if any(degree[output_slots[slot].index] == 0 for slot in new_slots):
        raise ValueError("PDBFixer returned an unconnected heavy atom")
    return result, unit.Quantity(np.asarray(result_positions), output_unit)


def _kind(residue):
    name = str(residue.name).upper()
    if name in _PROTEIN:
        return "protein"
    if name in _NUCLEIC:
        return "nucleic"
    return "nonpolymer"


def _named(residue, name):
    return next((atom for atom in residue.atoms() if atom.name == name), None)


def _edges(topology):
    return {frozenset((a.index, b.index)) for a, b in topology.bonds()}


def _closed_chains(topology):
    edges = _edges(topology)
    closures = []
    for ci, chain in enumerate(topology.chains()):
        residues = list(chain.residues())
        if len(residues) < 2:
            continue
        kinds = {_kind(residue) for residue in residues}
        if len(kinds) != 1 or "nonpolymer" in kinds:
            continue
        kind = next(iter(kinds))
        if kind == "protein":
            left = _named(residues[-1], "C")
            right = _named(residues[0], "N")
            closed = (
                left is not None
                and right is not None
                and frozenset((left.index, right.index)) in edges
            )
        else:
            closed = False
            for left_name, right_name in (
                ("O3'", "P"),
                ("O3*", "P"),
                ("C3'", "P"),
                ("C3*", "P"),
            ):
                left = _named(residues[-1], left_name)
                right = _named(residues[0], right_name)
                if (
                    left is not None
                    and right is not None
                    and frozenset((left.index, right.index)) in edges
                ):
                    closed = True
                    break
        if closed:
            closures.append((ci, 0, len(residues) - 1, kind))
    return closures


def _neighbors(topology):
    result = defaultdict(list)
    for a, b in topology.bonds():
        result[a].append(b)
        result[b].append(a)
    return result


def add_polymer_hydrogens(
    modeller,
    *,
    forcefield=None,
    platform=None,
    variants=None,
):
    """Mutate a Modeller with graph-aware polymer hydrogens.

    Non-None variant entries, including explicit hydrogen lists supplied by
    native nucleic/end templates, are passed unchanged.  Closed peptide chains
    and explicit cross-chain peptide links lose only H2/H3 at a graph-connected N end; closed nucleic chains lose only
    HO5'/HO3'.  Automatic disulfide cysteines lose any sulfur hydrogen.
    """

    from openmm.app import Modeller

    original_residues = list(modeller.topology.residues())
    variant_list = (
        [None] * len(original_residues)
        if variants is None
        else list(variants)
    )
    if len(variant_list) != len(original_residues):
        raise ValueError("variants must match the residue count")

    prepared = Modeller(modeller.topology, modeller.positions)
    prepared.addHydrogens(
        forcefield=forcefield,
        variants=variant_list,
        platform=platform,
    )
    prepared_chains = list(prepared.topology.chains())
    prepared_neighbors = _neighbors(prepared.topology)
    delete = []

    def remove_hydrogens(residue, parent_name, names):
        parent = _named(residue, parent_name)
        if parent is None:
            return
        delete.extend(
            atom
            for atom in residue.atoms()
            if atom.element is not None
            and atom.element.atomic_number == 1
            and atom.name in names
            and parent in prepared_neighbors[atom]
        )

    # Author chain boundaries need not be chemical termini. In particular,
    # explicit C--N links can join distinct chains. Follow the actual graph
    # when removing terminal amine hydrogens, retaining custom variants.
    original_neighbors = _neighbors(modeller.topology)
    prepared_residues = list(prepared.topology.residues())
    for source in original_residues:
        if _kind(source) != "protein" or variant_list[source.index] is not None:
            continue
        nitrogen = _named(source, "N")
        if nitrogen is not None and any(
            neighbor.residue is not source and neighbor.name == "C"
            and _kind(neighbor.residue) == "protein"
            for neighbor in original_neighbors[nitrogen]
        ):
            remove_hydrogens(
                prepared_residues[source.index],
                "N", frozenset(("H2", "H3")),
            )

    for ci, first, last, kind in _closed_chains(modeller.topology):
        source_residues = list(
            list(modeller.topology.chains())[ci].residues()
        )
        target_residues = list(prepared_chains[ci].residues())
        first_custom = variant_list[source_residues[first].index] is not None
        if not first_custom:
            if kind == "protein":
                remove_hydrogens(
                    target_residues[first], "N", frozenset(("H2", "H3"))
                )
            else:
                remove_hydrogens(
                    target_residues[first],
                    "O5'",
                    frozenset(("HO5'", "HO5*")),
                )
        if kind == "nucleic":
            last_custom = variant_list[source_residues[last].index] is not None
            if not last_custom:
                remove_hydrogens(
                    target_residues[last],
                    "O3'",
                    frozenset(("HO3'", "HO3*")),
                )

    disulfides = set()
    for a, b in modeller.topology.bonds():
        if (
            a.name == b.name == "SG"
            and a.residue.name == b.residue.name == "CYS"
        ):
            disulfides.update((a.residue.index, b.residue.index))
    target_residues = list(prepared.topology.residues())
    for residue_index in disulfides:
        if variant_list[residue_index] is not None:
            continue
        residue = target_residues[residue_index]
        sulfur = _named(residue, "SG")
        if sulfur is not None:
            delete.extend(
                atom
                for atom in residue.atoms()
                if atom.element is not None
                and atom.element.atomic_number == 1
                and sulfur in prepared_neighbors[atom]
            )

    if delete:
        prepared.delete(delete)
    modeller.topology = prepared.topology
    modeller.positions = prepared.positions
    return modeller


__all__ = ["repair_standard_heavy_atoms", "add_polymer_hydrogens"]
