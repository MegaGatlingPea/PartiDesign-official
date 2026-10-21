"""Explicit parameter ownership for a local GAFF product in an Amber scaffold.

Input structures are already parameterized. This layer never infers parameters:
local-product atoms/terms own the changed region, Amber owns its complement.
Private caps may disappear only outside every term touching the changed region.
"""
from __future__ import annotations

from copy import copy
from collections.abc import Hashable, Sequence
from itertools import combinations
from dataclasses import replace

import numpy as np


def audit_mol2_orders(molecule, bonds, *, atom_types=None):
    """Allow aromatic/Kekule and narrowly typed GAFF phosphodiester resonance.

    GAFF2 uses p5--o for both nonbridging oxygens of an anionic phosphate.
    This is a private parameter representation, never an output graph change.
    """
    from rdkit import Chem
    from .refine import OpenMMPostprocessError

    rw = Chem.RWMol()
    for original in molecule.GetAtoms():
        atom = Chem.Atom(original)
        atom.SetIsAromatic(False)
        rw.AddAtom(atom)
    types = {"1": Chem.BondType.SINGLE, "2": Chem.BondType.DOUBLE,
             "3": Chem.BondType.TRIPLE, "ar": Chem.BondType.AROMATIC,
             "am": Chem.BondType.SINGLE}
    try:
        for i, j, order in bonds:
            if order not in types:
                raise OpenMMPostprocessError("local MOL2 has an unsupported bond order")
            rw.AddBond(i, j, types[order])
        observed = rw.GetMol()
        expected = Chem.Mol(molecule)
        if atom_types is not None:
            if len(atom_types) != expected.GetNumAtoms():
                raise OpenMMPostprocessError("local MOL2 atom types do not match product")
            for p in expected.GetAtoms():
                if p.GetAtomicNum() != 15 or p.GetFormalCharge() != 0 or p.GetDegree() != 4:
                    continue
                neighbors = list(p.GetNeighbors())
                if any(a.GetAtomicNum() != 8 for a in neighbors):
                    continue
                terminal = [a for a in neighbors if a.GetDegree() == 1]
                bridges = [a for a in neighbors if a.GetDegree() == 2 and a.GetFormalCharge() == 0
                           and all(n.GetAtomicNum() in (6, 15) for n in a.GetNeighbors())]
                if len(terminal) != 2 or len(bridges) != 2:
                    continue
                double = [a for a in terminal if a.GetFormalCharge() == 0
                          and expected.GetBondBetweenAtoms(p.GetIdx(), a.GetIdx()).GetBondType() == Chem.BondType.DOUBLE]
                anion = [a for a in terminal if a.GetFormalCharge() == -1
                         and expected.GetBondBetweenAtoms(p.GetIdx(), a.GetIdx()).GetBondType() == Chem.BondType.SINGLE]
                if len(double) != 1 or len(anion) != 1:
                    continue
                if (atom_types[p.GetIdx()] != 'p5'
                    or any(atom_types[a.GetIdx()] != 'o' for a in terminal)
                    or any(atom_types[a.GetIdx()] != 'os' for a in bridges)):
                    continue
                if any(expected.GetBondBetweenAtoms(p.GetIdx(), a.GetIdx()).GetBondType() != Chem.BondType.SINGLE
                       for a in bridges):
                    continue
                if any(observed.GetBondBetweenAtoms(p.GetIdx(), a.GetIdx()) is None
                       or observed.GetBondBetweenAtoms(p.GetIdx(), a.GetIdx()).GetBondType() != Chem.BondType.SINGLE
                       for a in neighbors):
                    continue
                # Canonicalize *private copies* to P(+)-O(-). Atom and total
                # charge are not read from partial charges; this exact resonance
                # transform conserves the supplied explicit product charge.
                oxygen = double[0].GetIdx()
                expected.GetBondBetweenAtoms(p.GetIdx(), oxygen).SetBondType(Chem.BondType.SINGLE)
                for mol in (expected, observed):
                    mol.GetAtomWithIdx(p.GetIdx()).SetFormalCharge(1)
                    mol.GetAtomWithIdx(oxygen).SetFormalCharge(-1)
        Chem.SanitizeMol(observed)
        Chem.SanitizeMol(expected)
    except (ValueError, RuntimeError) as error:
        raise OpenMMPostprocessError("local MOL2 chemistry cannot be sanitized") from error
    def signature(mol):
        return sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                       str(b.GetBondType())) for b in mol.GetBonds())
    if signature(observed) != signature(expected):
        raise OpenMMPostprocessError("local MOL2 changed explicit product bond orders")


def constrain_core_charge(structure, final_keys, core_keys, formal_charge, *, target_charge=None):
    """Apply a uniform local charge-fit constraint; leave Amber complement exact.

    The caller supplies the integer charge of whole product residues/design
    components from their explicit chemistry, never by rounding partial charges.
    This is a local-refinement approximation, not a binding-energy model.
    """
    from .refine import OpenMMPostprocessError

    if isinstance(formal_charge, bool) or int(formal_charge) != formal_charge:
        raise OpenMMPostprocessError("local product formal charge must be explicit integer")
    if len(final_keys) != len(structure.atoms) or not core_keys.issubset(final_keys):
        raise OpenMMPostprocessError("local charge constraint atom mapping is invalid")
    indices = [i for i, key in enumerate(final_keys) if key in core_keys]
    if not indices:
        raise OpenMMPostprocessError("local charge constraint has empty product core")
    original = np.asarray([a.charge for a in structure.atoms], dtype=float)
    if not np.isfinite(original).all():
        raise OpenMMPostprocessError("local charge constraint has non-finite parameters")
    target = float(formal_charge) if target_charge is None else float(target_charge)
    if not np.isfinite(target):
        raise OpenMMPostprocessError("local product charge target is non-finite")
    correction = (target - original[indices].sum()) / len(indices)
    for i in indices:
        structure.atoms[i].charge += correction
    return dict(core_formal_charge=int(formal_charge), core_target_partial_charge=target,
                core_partial_charge_before=float(original[indices].sum()),
                per_atom_correction=float(correction), corrected_atoms=len(indices))


def cap_product_record(record, retained_keys: set[Hashable]):
    """Extract a local product, saturating only private single-bond cuts with H.

    Formal charges and real-product H counts come from the complete product,
    never from treating an artificial boundary as a new polymer terminus.
    Returned caps are private parameterization atoms, not output repair atoms.
    """
    from .refine import OpenMMPostprocessError

    if not retained_keys or not retained_keys.issubset(record.heavy_keys):
        raise OpenMMPostprocessError("local cap selection is absent from the complete product")
    indices = [i for i, key in enumerate(record.heavy_keys) if key in retained_keys]
    local = {old: new for new, old in enumerate(indices)}
    hydrogens = [record.hydrogen_counts[i] for i in indices]
    bonds = []
    cut_counts = {}
    for i, j, order in record.bonds:
        if i in local and j in local:
            bonds.append((local[i], local[j], order))
        elif (i in local) != (j in local):
            if order != "single":
                raise OpenMMPostprocessError("local fragment cannot cap a non-single boundary bond")
            old = i if i in local else j
            hydrogens[local[old]] += 1
            cut_counts[record.heavy_keys[old]] = cut_counts.get(record.heavy_keys[old], 0) + 1
    return replace(
        record, record_id=f"local-{record.record_id}",
        heavy_keys=tuple(record.heavy_keys[i] for i in indices),
        elements=tuple(record.elements[i] for i in indices),
        formal_charges=tuple(record.formal_charges[i] for i in indices),
        bonds=tuple(bonds), hydrogen_counts=tuple(hydrogens),
    ), cut_counts


def parameterize_product_record(record, atoms, work, *, charge_method):
    """Generate one audited local GAFF2 structure using installed AmberTools."""
    from rdkit import Chem
    import parmed as pmd
    from .refine import _build_private_ligand, atom_key, OpenMMPostprocessError
    from .covalent import (
        _run, _antechamber_args, _validate_antechamber_sqm,
        _validated_frcmod, _parse_mol2,
    )

    row_by_key = {atom_key(atoms, r): r for r in range(len(atoms))}
    molecule = _build_private_ligand(record, atoms, row_by_key)
    total_charge = sum(record.formal_charges)
    writer = Chem.SDWriter(str(work / "product.sdf"))
    writer.write(molecule)
    writer.close()
    _run("antechamber", _antechamber_args("product.sdf", total_charge, charge_method), work)
    if charge_method == "am1bcc":
        _validate_antechamber_sqm(work)
    elif any((work / name).exists() for name in ("sqm.in", "sqm.out")):
        raise OpenMMPostprocessError("local Gasteiger unexpectedly produced SQM output")
    mol2_atoms, mol2_bonds = _parse_mol2(work / "product.mol2", [a.GetSymbol() for a in molecule.GetAtoms()])
    charges = np.asarray([atom[3] for atom in mol2_atoms])
    if not np.isfinite(charges).all() or not np.isclose(charges.sum(), total_charge, atol=5e-4, rtol=0):
        raise OpenMMPostprocessError("local product partial charges do not conserve explicit charge")
    expected_graph = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in molecule.GetBonds()}
    if {frozenset((i, j)) for i, j, _ in mol2_bonds} != expected_graph:
        raise OpenMMPostprocessError("local product MOL2 changed the explicit graph")
    audit_mol2_orders(molecule, mol2_bonds, atom_types=[atom[2] for atom in mol2_atoms])
    observed_xyz = np.asarray([a[1] for a in mol2_atoms])
    if not np.allclose(observed_xyz, molecule.GetConformer().GetPositions(), atol=2e-3, rtol=0):
        raise OpenMMPostprocessError("local parameterization changed atom coordinate mapping")
    _run("parmchk2", ["-i", "product.mol2", "-f", "mol2", "-o", "product.frcmod", "-s", "2", "-a", "Y", "-w", "Y"], work)
    _validated_frcmod((work / "product.frcmod").read_text(), label="local product")
    (work / "product.leap").write_text(
        "source leaprc.gaff2\nloadAmberParams product.frcmod\n"
        "product = loadMol2 product.mol2\nsaveAmberParm product product.prmtop product.inpcrd\nquit\n"
    )
    _run("tleap", ["-f", "product.leap"], work)
    parm = pmd.load_file(str(work / "product.prmtop"), str(work / "product.inpcrd"))
    if len(parm.atoms) != molecule.GetNumAtoms() or any(
        a.atomic_number != molecule.GetAtomWithIdx(i).GetAtomicNum() for i, a in enumerate(parm.atoms)
    ):
        raise OpenMMPostprocessError("local PRMTOP changed atom inventory/order")
    if {frozenset((b.atom1.idx, b.atom2.idx)) for b in parm.bonds} != expected_graph:
        raise OpenMMPostprocessError("local PRMTOP changed explicit product topology")
    if not np.allclose(parm.coordinates, observed_xyz, atol=2e-3, rtol=0):
        raise OpenMMPostprocessError("local PRMTOP changed coordinate mapping")
    return parm, molecule


def merge_parameterized_region(
    scaffold, product, *, scaffold_keys: Sequence[Hashable | None],
    product_keys: Sequence[Hashable | None], core_keys: set[Hashable],
    final_keys: Sequence[Hashable], expected_bonds: set[frozenset],
):
    """Return a fully parameterized ParmEd Structure with exact final ownership.

    ``None`` marks private reference/cap atoms, never public atoms. ``final_keys``
    includes explicit private hydrogens selected for the actual product state.
    Coordinates come from each atom's owning structure; callers must put original
    heavy coordinates there before minimization.
    """
    import parmed as pmd
    from .refine import OpenMMPostprocessError

    def fail(message):
        raise OpenMMPostprocessError(message)

    final = tuple(final_keys)
    if len(set(final)) != len(final) or not core_keys or not core_keys.issubset(final):
        fail("hybrid final atom ownership is invalid")
    sources = ((scaffold, tuple(scaffold_keys), False), (product, tuple(product_keys), True))
    owners = {}
    for source, keys, local in sources:
        if len(keys) != len(source.atoms):
            fail("hybrid parameter atom mapping length disagrees")
        real = [k for k in keys if k is not None]
        if len(set(real)) != len(real):
            fail("hybrid parameter atom mapping is not unique")
        unsupported = ("adjusts", "urey_bradleys", "rb_torsions", "trigonal_angles",
                       "out_of_plane_bends", "pi_torsions", "stretch_bends",
                       "torsion_torsions")
        if any(getattr(source, name, ()) for name in unsupported):
            fail("hybrid source uses unsupported explicit exceptions or force families")
        for atom, key in zip(source.atoms, keys, strict=True):
            if key in final and (key in core_keys) == local:
                owners[key] = atom
    if set(owners) != set(final):
        fail("hybrid final atoms lack an authoritative parameter owner")
    out = pmd.Structure()
    mapped = {}
    positions = []
    for key in final:
        source_atom = owners[key]
        atom = copy(source_atom)
        residue = source_atom.residue
        out.add_atom(atom, residue.name, residue.number, chain=residue.chain, inscode=residue.insertion_code)
        mapped[key] = atom
        positions.append((source_atom.xx, source_atom.xy, source_atom.xz))
    out.coordinates = np.asarray(positions, dtype=float)
    if not np.isfinite(out.coordinates).all():
        fail("hybrid source positions are non-finite")

    families = (("bonds", "bond_types", 2, pmd.Bond),
                ("angles", "angle_types", 3, pmd.Angle),
                ("dihedrals", "dihedral_types", 4, pmd.Dihedral),
                ("impropers", "improper_types", 4, pmd.Improper),
                ("cmaps", "cmap_types", 5, pmd.Cmap))
    for name, type_name, size, constructor in families:
        type_cache = {}
        for source, keys, local in sources:
            for term in getattr(source, name):
                ids = tuple(keys[getattr(term, f"atom{i}").idx] for i in range(1, size + 1))
                touches_core = any(key in core_keys for key in ids)
                if touches_core != local:
                    continue
                if any(key not in mapped for key in ids):
                    if local:
                        fail("local product cap intersects a required hybrid boundary term")
                    fail("Amber environment term lost an atom outside the local product")
                if term.type is None:
                    fail("hybrid source contains an unparameterized term")
                cache_key = id(term.type)
                if cache_key not in type_cache:
                    type_cache[cache_key] = copy(term.type)
                    getattr(out, type_name).append(type_cache[cache_key])
                kwargs = {"type": type_cache[cache_key]}
                if name == "bonds":
                    kwargs["order"] = term.order
                if name == "dihedrals":
                    kwargs.update(improper=term.improper, ignore_end=term.ignore_end)
                getattr(out, name).append(constructor(*(mapped[key] for key in ids), **kwargs))
        getattr(out, type_name).claim()
    actual = [frozenset((final[b.atom1.idx], final[b.atom2.idx])) for b in out.bonds]
    if len(actual) != len(set(actual)) or set(actual) != set(expected_bonds):
        fail("hybrid parameter graph differs from authoritative product topology")
    graph = {i: set() for i in range(len(final))}
    for bond in out.bonds:
        i, j = bond.atom1.idx, bond.atom2.idx
        graph[i].add(j)
        graph[j].add(i)
    core_indices = {mapped[key].idx for key in core_keys}
    def canonical(ids):
        return min(tuple(ids), tuple(reversed(ids)))
    expected_angles = {
        canonical((i, center, j)) for center, neighbors in graph.items()
        for i, j in combinations(sorted(neighbors), 2)
        if core_indices.intersection((i, center, j))
    }
    actual_angles = {canonical((a.atom1.idx, a.atom2.idx, a.atom3.idx)) for a in out.angles}
    expected_propers = {
        canonical((i, j, k, l)) for j in graph for k in graph[j]
        for i in graph[j] - {k} for l in graph[k] - {j}
        if len({i, j, k, l}) == 4 and core_indices.intersection((i, j, k, l))
    }
    actual_propers = {
        canonical((d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx))
        for d in out.dihedrals if not d.improper
    }
    if not expected_angles.issubset(actual_angles) or not expected_propers.issubset(actual_propers):
        fail("hybrid local angle/proper coverage is incomplete")
    out.unchange()
    return out
