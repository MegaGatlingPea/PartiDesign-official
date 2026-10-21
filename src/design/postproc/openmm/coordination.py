"""Validated, rigid metal-coordination contexts for private OpenMM views.

The public structure remains a heavy-atom graph.  A fixed component is accepted
only when its exact named CCD graph is present, all of its atoms are context
owned, and the public graph is closed.  CCD DATIVE metal edges may be encoded by
legacy PDB readers as SINGLE or ANY; that compatibility is deliberately narrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers, rdPartialCharges

from partidesign.paths import resource_root


AtomKey = tuple[str, int, str, str, str]

_PUBLIC_BONDS = {
    0: "any", 1: "single", 2: "double", 3: "triple", 4: "quadruple",
    5: "aromatic_single", 6: "aromatic_double", 7: "aromatic_triple",
    8: "coordination", 9: "aromatic",
}
_METALS = frozenset(
    "Li Be Na Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv".split()
)
_UFF_A_TO_NM = 0.1
_KCAL_TO_KJ = 4.184
_SIGMA_FACTOR = 2.0 ** (-1.0 / 6.0)
_UFF_FORCED_SP3 = frozenset({12, 13, 14, 15, 50, 51, 52, 81, 82, 83, 84})


def _err(message: str) -> ValueError:
    return ValueError(message)


def _symbol(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise _err("fixed coordination atom has an empty element")
    return text[:1].upper() + text[1:].lower()


def _atom_key(atoms: Any, row: int) -> AtomKey:
    try:
        return (
            str(atoms.chain_id[row]), int(atoms.res_id[row]), str(atoms.ins_code[row]),
            str(atoms.res_name[row]), str(atoms.atom_name[row]),
        )
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        raise _err("fixed coordination rows lack stable atom identities") from exc


def _atom_name(atom: Any) -> str:
    if not atom.HasProp("name") or not atom.GetProp("name").strip():
        raise _err("trusted CCD atom lacks a stable name")
    return atom.GetProp("name").strip()


def _ccd_bond_name(bond: Any) -> str:
    mapping = {
        Chem.BondType.SINGLE: "single", Chem.BondType.DOUBLE: "double",
        Chem.BondType.TRIPLE: "triple", Chem.BondType.AROMATIC: "aromatic",
    }
    if bond.GetBondType() == Chem.BondType.DATIVE:
        return "dative"
    try:
        return mapping[bond.GetBondType()]
    except KeyError as exc:
        raise _err(f"trusted CCD has unsupported bond type {bond.GetBondType()}") from exc


def _ccd_from_context(name: str) -> Chem.Mol:
    from ..context import _CONTEXT_COMPONENTS, _load_components
    normalized = str(name).strip().upper()
    if not normalized:
        raise _err("fixed coordination CCD identity is empty")
    _load_components({normalized})
    carrier = _CONTEXT_COMPONENTS.get(normalized)
    if carrier is None:
        raise _err(f"trusted CCD has no named component {normalized!r}")
    try:
        return Chem.Mol(carrier.rdkit_binary)
    except Exception as exc:
        raise _err("trusted CCD binary cannot be decoded") from exc


def _component(name: str, value: Any | None) -> Chem.Mol:
    if value is None:
        return _ccd_from_context(name)
    if isinstance(value, Chem.Mol):
        return Chem.Mol(value)
    carrier_id = getattr(value, "component_id", None)
    if carrier_id is not None and str(carrier_id).strip().upper() != name:
        raise _err("trusted CCD component carrier has the wrong identity")
    binary = getattr(value, "rdkit_binary", None)
    if isinstance(binary, (bytes, bytearray)):
        try:
            return Chem.Mol(bytes(binary))
        except Exception as exc:
            raise _err("trusted CCD component binary cannot be decoded") from exc
    raise _err("fixed coordination component must be a trusted CCD carrier or Mol")


def _ordered_molecules(source: Chem.Mol, names: Sequence[str]):
    """Return public-heavy-order heavy/full molecules and trusted H names."""
    try:
        raw = Chem.Mol(source)
        heavy = Chem.RemoveHs(raw, sanitize=True)
    except Exception as exc:
        raise _err("trusted CCD molecule cannot be reduced to heavy atoms") from exc
    if heavy.GetNumAtoms() != len(names):
        raise _err("trusted CCD heavy-atom count does not match fixed component")
    by_name = {}
    for index, atom in enumerate(heavy.GetAtoms()):
        name = _atom_name(atom)
        if name in by_name:
            raise _err("trusted CCD heavy atom names are ambiguous")
        by_name[name] = index
    if set(by_name) != set(names):
        raise _err("trusted CCD heavy atom names do not match fixed component")
    heavy_order = [by_name[name] for name in names]
    heavy = Chem.RenumberAtoms(heavy, heavy_order)

    raw_heavy = {_atom_name(a): i for i, a in enumerate(raw.GetAtoms()) if a.GetAtomicNum() > 1}
    if set(raw_heavy) != set(names):
        raise _err("trusted CCD full molecule has an unexpected heavy inventory")
    hydrogens = [i for i, a in enumerate(raw.GetAtoms()) if a.GetAtomicNum() == 1]
    hydrogens.sort(key=lambda i: (
        raw.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx()
        if raw.GetAtomWithIdx(i).GetNeighbors() else len(raw.GetAtoms()), i,
    ))
    full = Chem.RenumberAtoms(raw, [raw_heavy[name] for name in names] + hydrogens)
    full.RemoveAllConformers()
    h_names = [[] for _ in names]
    for index in range(len(names), full.GetNumAtoms()):
        atom = full.GetAtomWithIdx(index)
        neighbours = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbours) != 1 or neighbours[0] >= len(names):
            raise _err("trusted CCD private hydrogen is not singly attached to a heavy atom")
        h_names[neighbours[0]].append(_atom_name(atom))
    return heavy, full, tuple(tuple(v) for v in h_names)


def _private_binary(molecule: Chem.Mol) -> bytes:
    copy = Chem.Mol(molecule)
    copy.RemoveAllConformers()
    return bytes(copy.ToBinary(Chem.PropertyPickleOptions.AllProps))


def _donor_charges(molecule: Chem.Mol, donors: Sequence[int], metal: int,
                   h_names: Sequence[Sequence[str]]) -> tuple[float, ...]:
    """Assign Gasteiger charges independently to donor-side fragments.

    CCD coordination bonds are deliberately removed before constructing each
    fragment.  This keeps the calculation independent of the metal's formal
    valence while still covering donor groups larger than a single atom+H set.
    """
    count = molecule.GetNumAtoms()
    charges = [math.nan] * count
    if metal < 0 or metal >= len(h_names):
        raise _err("fixed coordination metal index is invalid")
    donor_set = set(int(v) for v in donors)
    heavy_count = len(h_names)
    if not donor_set or any(v == metal or v < 0 or v >= heavy_count for v in donor_set):
        raise _err("fixed coordination donor index is invalid")
    charges[metal] = float(molecule.GetAtomWithIdx(metal).GetFormalCharge())

    adjacency = [set() for _ in range(count)]
    for bond in molecule.GetBonds():
        if bond.GetBondType() == Chem.BondType.DATIVE:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i != metal and j != metal:
            adjacency[i].add(j)
            adjacency[j].add(i)
    components = []
    remaining = set(range(count)) - {metal}
    while remaining:
        root = min(remaining)
        component = {root}
        queue = [root]
        remaining.remove(root)
        while queue:
            current = queue.pop()
            for neighbour in adjacency[current]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        components.append(tuple(sorted(component)))

    for indices in components:
        rw = Chem.RWMol()
        mapping = {}
        for index in indices:
            atom = Chem.Atom(molecule.GetAtomWithIdx(index))
            # All hydrogens are explicit in the trusted binary.  Suppress any
            # inherited implicit-H count before adding explicit H atoms.
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
            mapping[index] = rw.AddAtom(atom)
        for bond in molecule.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in mapping and j in mapping and bond.GetBondType() != Chem.BondType.DATIVE:
                rw.AddBond(mapping[i], mapping[j], bond.GetBondType())
        fragment = rw.GetMol()
        try:
            Chem.SanitizeMol(fragment)
            rdPartialCharges.ComputeGasteigerCharges(fragment)
        except Exception as exc:
            raise _err("donor fragment Gasteiger charge assignment failed") from exc
        values = []
        for atom in fragment.GetAtoms():
            if not atom.HasProp("_GasteigerCharge"):
                raise _err("donor fragment Gasteiger charge is absent")
            try:
                value = float(atom.GetProp("_GasteigerCharge"))
            except (TypeError, ValueError) as exc:
                raise _err("donor fragment Gasteiger charge is not numeric") from exc
            if not math.isfinite(value):
                raise _err("donor fragment Gasteiger charge is non-finite")
            values.append(value)
        target = sum(int(molecule.GetAtomWithIdx(index).GetFormalCharge()) for index in indices)
        if not math.isclose(sum(values), target, rel_tol=0, abs_tol=5e-4):
            raise _err("donor fragment Gasteiger charge does not conserve integer charge")
        for index, value in zip(indices, values):
            charges[index] = value

    donor_components = {index for indices in components for index in indices if index in donor_set}
    if donor_components != donor_set:
        raise _err("fixed coordination donor fragment is missing")
    if any(not math.isfinite(float(v)) for v in charges):
        raise _err("fixed coordination private atoms lack finite partial charges")
    formal = sum(int(a.GetFormalCharge()) for a in molecule.GetAtoms())
    if not math.isclose(sum(charges), formal, rel_tol=0, abs_tol=5e-4):
        raise _err("fixed coordination partial charges do not conserve integer charge")
    return tuple(float(v) for v in charges)


@dataclass(frozen=True, slots=True)
class FixedCoordinationRecord:
    """Coordinate-free trusted chemistry for one fixed context component."""

    record_id: str
    molecule_uid: int
    component_uid: int
    heavy_keys: tuple[AtomKey, ...]
    elements: tuple[str, ...]
    formal_charges: tuple[int, ...]
    bonds: tuple[tuple[int, int, str], ...]
    hydrogen_counts: tuple[int, ...]
    component_id: str
    trusted_molecule_binary: bytes
    metal_index: int
    donor_indices: tuple[int, ...]
    hydrogen_names: tuple[tuple[str, ...], ...] = ()
    private_partial_charges: tuple[float, ...] = ()

    def __post_init__(self):
        size = len(self.heavy_keys)
        if not self.record_id or not self.component_id or not size:
            raise ValueError("fixed coordination record needs an identity and heavy atoms")
        if len(set(self.heavy_keys)) != size:
            raise ValueError("fixed coordination heavy keys must be unique")
        if not (len(self.elements) == len(self.formal_charges) ==
                len(self.hydrogen_counts) == size):
            raise ValueError("fixed coordination heavy arrays have inconsistent lengths")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in self.formal_charges):
            raise ValueError("fixed coordination formal charges must be integers")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in self.hydrogen_counts):
            raise ValueError("fixed coordination hydrogen counts must be nonnegative integers")
        if not 0 <= self.metal_index < size or not self.donor_indices:
            raise ValueError("fixed coordination metal/donor indices are invalid")
        if self.metal_index in self.donor_indices or len(set(self.donor_indices)) != len(self.donor_indices):
            raise ValueError("fixed coordination donors are duplicated or include the metal")
        for i, j, order in self.bonds:
            if not 0 <= i < size or not 0 <= j < size or i == j:
                raise ValueError("fixed coordination bond endpoint is invalid")
            if order not in {"single", "double", "triple", "aromatic", "dative"}:
                raise ValueError("fixed coordination bond order is invalid")
        if not isinstance(self.trusted_molecule_binary, bytes) or not self.trusted_molecule_binary:
            raise ValueError("fixed coordination trusted molecule binary is empty")
        if self.hydrogen_names and len(self.hydrogen_names) != size:
            raise ValueError("fixed coordination hydrogen names are not row aligned")
        if self.private_partial_charges and any(not math.isfinite(float(v)) for v in self.private_partial_charges):
            raise ValueError("fixed coordination partial charges must be finite")

    @property
    def ccd_id(self):
        return self.component_id

    @property
    def trusted_binary(self):
        return self.trusted_molecule_binary

    @property
    def residue_key(self):
        return self.heavy_keys[0][:4]


def build_fixed_coordination_record(
    atoms: Any, rows: Sequence[int], *, component: Any | None = None,
    component_id: str | None = None, record_id: str | None = None,
    expected_heavy_keys: Sequence[AtomKey] | None = None,
) -> FixedCoordinationRecord:
    """Validate one closed, non-design fixed coordination component."""
    try:
        selected = tuple(int(row) for row in rows)
    except (TypeError, ValueError) as exc:
        raise _err("fixed coordination rows must be integer indices") from exc
    if (not selected or len(set(selected)) != len(selected)
            or any(row < 0 or row >= len(atoms) for row in selected)):
        raise _err("fixed coordination rows are empty, duplicated, or out of range")
    if getattr(atoms, "bonds", None) is None:
        raise _err("fixed coordination requires an authoritative BondList")
    if not hasattr(atoms, "is_design") or any(bool(v) for v in np.asarray(atoms.is_design)[list(selected)]):
        raise _err("fixed coordination component contains design-owned atoms")
    try:
        molecules = np.asarray(atoms.molecule_uid)
        components = np.asarray(atoms.component_uid)
    except AttributeError as exc:
        raise _err("fixed coordination requires molecule_uid and component_uid") from exc
    uid, cuid = int(molecules[selected[0]]), int(components[selected[0]])
    if any(int(molecules[r]) != uid or int(components[r]) != cuid for r in selected):
        raise _err("fixed coordination rows span multiple components")
    all_rows = tuple(int(r) for r in np.flatnonzero((molecules == uid) & (components == cuid)))
    if set(all_rows) != set(selected):
        raise _err("fixed coordination component is incomplete")
    heavy_keys = tuple(_atom_key(atoms, r) for r in selected)
    if expected_heavy_keys is not None and tuple(expected_heavy_keys) != heavy_keys:
        raise _err("fixed coordination heavy keys disagree with the accepted record")
    if len({key[:4] for key in heavy_keys}) != 1:
        raise _err("fixed coordination component must occupy one residue")
    public_name = heavy_keys[0][3].strip().upper()
    name = str(component_id or public_name).strip().upper()
    if not name:
        raise _err("fixed coordination CCD identity is empty")
    # A public residue name is normally the exact CCD ID.  A normalized source
    # may expose an alias (for example LIG01) after its trusted CCD component
    # has been frozen; accepting that alias requires the explicit CCD identity.
    if name != public_name and (component_id is None or component is None):
        raise _err("fixed coordination requires an exact named CCD identity")

    source = _component(name, component)
    if len(Chem.GetMolFrags(source)) != 1:
        raise _err("trusted CCD coordination component is disconnected")
    names = tuple(key[4] for key in heavy_keys)
    heavy, full, h_names = _ordered_molecules(source, names)
    observed = tuple(_symbol(atoms.element[r]) for r in selected)
    if observed != tuple(atom.GetSymbol() for atom in heavy.GetAtoms()):
        raise _err("trusted CCD elements do not match fixed public atoms")

    selected_set = set(selected)
    public_edges = {}
    for raw_i, raw_j, raw_type in atoms.bonds.as_array():
        i, j = int(raw_i), int(raw_j)
        inside_i, inside_j = i in selected_set, j in selected_set
        if inside_i != inside_j:
            raise _err("fixed coordination component has an external heavy edge")
        if not inside_i:
            continue
        local = tuple(sorted((selected.index(i), selected.index(j))))
        order = _PUBLIC_BONDS.get(int(raw_type))
        if order is None:
            raise _err(f"unsupported public coordination bond type: {raw_type!r}")
        if local in public_edges and public_edges[local] != order:
            raise _err("fixed coordination public graph has conflicting bond orders")
        public_edges[local] = order
    ccd_edges = {}
    for bond in heavy.GetBonds():
        local = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        order = _ccd_bond_name(bond)
        if local in ccd_edges and ccd_edges[local] != order:
            raise _err("trusted CCD graph has conflicting bond orders")
        ccd_edges[local] = order
    if set(public_edges) != set(ccd_edges):
        raise _err("fixed coordination public graph is incomplete or has extra edges")
    for edge, trusted_order in ccd_edges.items():
        public_order = public_edges[edge]
        if trusted_order == "dative":
            if public_order not in {"single", "any", "coordination"}:
                raise _err("legacy public metal edge conflicts with trusted CCD DATIVE")
        elif public_order != trusted_order:
            raise _err("fixed coordination public graph conflicts with trusted CCD chemistry")

    dative = [edge for edge, order in ccd_edges.items() if order == "dative"]
    if not dative:
        raise _err("trusted CCD component has no DATIVE metal coordination")
    endpoints = {i for edge in dative for i in edge}
    metals = [i for i in endpoints if _symbol(heavy.GetAtomWithIdx(i).GetSymbol()) in _METALS]
    if len(metals) != 1:
        raise _err("trusted CCD DATIVE graph does not identify one metal center")
    metal = metals[0]
    donors = tuple(sorted(
        right if left == metal else left
        for left, right in dative
    ))
    if len(set(donors)) != len(donors) or any(
        _symbol(heavy.GetAtomWithIdx(i).GetSymbol()) in _METALS for i in donors
    ):
        raise _err("trusted CCD DATIVE graph has an unsupported donor arrangement")
    if any(int(atom.GetTotalNumHs()) != len(h_names[i]) for i, atom in enumerate(heavy.GetAtoms())):
        raise _err("trusted CCD hydrogen counts are inconsistent")
    charges = tuple(int(atom.GetFormalCharge()) for atom in heavy.GetAtoms())
    partial = _donor_charges(full, donors, metal, h_names)
    bonds = tuple((i, j, order) for (i, j), order in sorted(ccd_edges.items()))
    return FixedCoordinationRecord(
        record_id=str(record_id or f"fixed-{name}-{uid}-{cuid}"),
        molecule_uid=uid, component_uid=cuid, heavy_keys=heavy_keys,
        elements=tuple(atom.GetSymbol().upper() for atom in heavy.GetAtoms()),
        formal_charges=charges, bonds=bonds,
        hydrogen_counts=tuple(int(atom.GetTotalNumHs()) for atom in heavy.GetAtoms()),
        component_id=name, trusted_molecule_binary=_private_binary(full),
        metal_index=metal, donor_indices=donors, hydrogen_names=h_names,
        private_partial_charges=partial,
    )


def validate_fixed_coordination(*args: Any, **kwargs: Any):
    return build_fixed_coordination_record(*args, **kwargs)


def _decode(record: FixedCoordinationRecord) -> Chem.Mol:
    try:
        molecule = Chem.Mol(record.trusted_molecule_binary)
        Chem.SanitizeMol(molecule)
        return molecule
    except Exception as exc:
        raise _err("fixed coordination trusted molecule binary is invalid") from exc


def _private_with_coordinates(record: FixedCoordinationRecord, coords: np.ndarray):
    n = len(record.heavy_keys)
    molecule = _decode(record)
    heavy = Chem.RemoveHs(molecule, sanitize=True)
    if heavy.GetNumAtoms() != n:
        raise _err("fixed coordination trusted molecule heavy count changed")
    if tuple(_atom_name(a) for a in heavy.GetAtoms()) != tuple(key[4] for key in record.heavy_keys):
        raise _err("fixed coordination private atom order disagrees with heavy keys")
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape != (n, 3) or not np.isfinite(coords).all():
        raise _err("fixed coordination accepted heavy coordinates are invalid")
    conformer = Chem.Conformer(n)
    for i, xyz in enumerate(coords):
        conformer.SetAtomPosition(i, tuple(float(x) for x in xyz))
    heavy.RemoveAllConformers()
    heavy.AddConformer(conformer, assignId=True)
    try:
        expanded = Chem.AddHs(heavy, addCoords=True)
        Chem.SanitizeMol(expanded)
    except Exception as exc:
        raise _err("fixed coordination private hydrogen expansion failed") from exc
    observed = tuple(sum(nbr.GetAtomicNum() == 1 for nbr in expanded.GetAtomWithIdx(i).GetNeighbors())
                     for i in range(n))
    if observed != tuple(record.hydrogen_counts):
        raise _err("fixed coordination private hydrogen inventory changed")
    names = record.hydrogen_names or tuple(tuple() for _ in range(n))
    generated = [[] for _ in range(n)]
    for i in range(n, expanded.GetNumAtoms()):
        parent = next((nbr.GetIdx() for nbr in expanded.GetAtomWithIdx(i).GetNeighbors()
                       if nbr.GetAtomicNum() > 1), None)
        if parent is None:
            raise _err("fixed coordination private hydrogen has no heavy parent")
        generated[parent].append(i)
    if any(len(generated[i]) != len(names[i]) for i in range(n)):
        raise _err("fixed coordination trusted H names do not match expansion")
    for parent in range(n):
        for position, index in enumerate(generated[parent]):
            expanded.GetAtomWithIdx(index).SetProp("_PartiDesignName", names[parent][position])
    return expanded


def _topology_key(atom: Any) -> AtomKey:
    residue = atom.residue
    try:
        res_id = int(residue.id)
    except (TypeError, ValueError) as exc:
        raise _err("fixed coordination topology residue ID is not integer-like") from exc
    return (str(residue.chain.id), res_id, str(residue.insertionCode),
            str(residue.name), str(atom.name))


def fixed_coordination_keys(records: Iterable[FixedCoordinationRecord],
                            *, include_hydrogens: bool = True) -> tuple[AtomKey, ...]:
    output, seen = [], set()
    for record in records:
        for key in record.heavy_keys:
            if key in seen:
                raise ValueError("fixed coordination records overlap heavy atoms")
            seen.add(key)
            output.append(key)
        if include_hydrogens:
            for parent, names in zip(record.heavy_keys, record.hydrogen_names):
                for name in names:
                    key = (*parent[:4], str(name))
                    if key in seen:
                        raise ValueError("fixed coordination records overlap private H")
                    seen.add(key)
                    output.append(key)
    return tuple(output)


def _positions(positions: Any, count: int):
    from openmm import unit
    if hasattr(positions, "value_in_unit"):
        xyz = np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    else:
        xyz = np.asarray([tuple(v) for v in positions], dtype=np.float64)
    if xyz.shape != (count, 3) or not np.isfinite(xyz).all():
        raise _err("OpenMM positions are not finite with shape (N, 3)")
    return xyz


def expand_fixed_coordination(topology: Any, positions: Any,
                              records: Sequence[FixedCoordinationRecord]):
    """Expand private H while keeping every residue's atoms contiguous.

    OpenMM requires all atoms belonging to one residue to be adjacent in the
    topology atom order.  Each trusted molecule is therefore expanded before
    rebuilding, then its private H are appended immediately after that
    residue's original atoms.  Original public atoms, IDs, and coordinates
    remain keyed and unchanged; bonds are copied after all atoms exist.
    """
    from openmm import unit
    from openmm.app import Topology, element

    source_atoms = list(topology.atoms())
    xyz = _positions(positions, len(source_atoms))
    by_key = {_topology_key(atom): atom for atom in source_atoms}
    if len(by_key) != len(source_atoms):
        raise _err("fixed coordination topology heavy keys are not unique")

    # Precompute every accepted conformer before rebuilding.  This keeps all
    # coordinate validation independent of the output atom ordering.
    plans = []
    fixed = []
    records_by_residue = {}
    for record in records:
        heavy_atoms = []
        for key in record.heavy_keys:
            if key not in by_key:
                raise _err(f"fixed coordination heavy key is absent: {key}")
            heavy_atoms.append(by_key[key])
            fixed.append(key)
        if not heavy_atoms:
            raise _err("fixed coordination record has no heavy atoms")
        residue = heavy_atoms[0].residue
        if any(atom.residue is not residue for atom in heavy_atoms):
            raise _err("fixed coordination record spans multiple residues")
        expanded = _private_with_coordinates(
            record, xyz[[atom.index for atom in heavy_atoms]] * 10.0,
        )
        plan = {
            "record": record,
            "heavy_atoms": tuple(heavy_atoms),
            "residue": residue,
            "expanded": expanded,
            "generated": {},
        }
        plans.append(plan)
        records_by_residue.setdefault(id(residue), []).append(plan)
        for parent, names in zip(record.heavy_keys, record.hydrogen_names):
            for name in names:
                fixed.append((*parent[:4], str(name)))

    rebuilt = Topology()
    if topology.getPeriodicBoxVectors() is not None:
        rebuilt.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    mapped, out_xyz = {}, []
    for chain in topology.chains():
        new_chain = rebuilt.addChain(chain.id)
        for residue in chain.residues():
            new_residue = rebuilt.addResidue(
                residue.name, new_chain, residue.id, residue.insertionCode,
            )
            # Copy all original atoms first, preserving their source order and
            # coordinates.  Private H are appended before moving to the next
            # residue, which is required by OpenMM's topology invariant.
            for atom in residue.atoms():
                try:
                    new_atom = rebuilt.addAtom(
                        atom.name, atom.element, new_residue,
                        id=atom.id, formalCharge=atom.formalCharge,
                    )
                except (AttributeError, TypeError):
                    new_atom = rebuilt.addAtom(atom.name, atom.element, new_residue, id=atom.id)
                mapped[atom] = new_atom
                out_xyz.append(xyz[atom.index])
            for plan in records_by_residue.get(id(residue), ()):
                record = plan["record"]
                expanded = plan["expanded"]
                generated = {}
                for index in range(len(record.heavy_keys), expanded.GetNumAtoms()):
                    name = expanded.GetAtomWithIdx(index).GetProp("_PartiDesignName")
                    if name in generated:
                        raise _err("trusted private hydrogen names are duplicated")
                    try:
                        hydrogen = rebuilt.addAtom(
                            name, element.hydrogen, new_residue,
                            id=f"private:fixed:{record.record_id}:{name}",
                        )
                    except (AttributeError, TypeError):
                        hydrogen = rebuilt.addAtom(name, element.hydrogen, new_residue)
                    generated[name] = hydrogen
                    point = expanded.GetConformer().GetAtomPosition(index)
                    out_xyz.append(np.asarray((point.x, point.y, point.z), dtype=np.float64) * 0.1)
                plan["generated"] = generated

    if len(mapped) != len(source_atoms):
        raise _err("fixed coordination topology atom mapping is incomplete")
    for bond in topology.bonds():
        rebuilt.addBond(mapped[bond[0]], mapped[bond[1]], bond.type, bond.order)

    existing = {frozenset((a.index, b.index)) for a, b in rebuilt.bonds()}
    for plan in plans:
        record = plan["record"]
        heavy_atoms = plan["heavy_atoms"]
        expanded = plan["expanded"]
        generated = plan["generated"]
        for i, j, order in record.bonds:
            if order != "dative":
                continue
            pair = frozenset((mapped[heavy_atoms[i]].index, mapped[heavy_atoms[j]].index))
            if pair not in existing:
                rebuilt.addBond(mapped[heavy_atoms[i]], mapped[heavy_atoms[j]], order=1.0)
                existing.add(pair)
        for bond in expanded.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < len(record.heavy_keys) and j < len(record.heavy_keys):
                continue
            if i < len(record.heavy_keys):
                hname = expanded.GetAtomWithIdx(j).GetProp("_PartiDesignName")
                heavy_atom = mapped[heavy_atoms[i]]
            elif j < len(record.heavy_keys):
                hname = expanded.GetAtomWithIdx(i).GetProp("_PartiDesignName")
                heavy_atom = mapped[heavy_atoms[j]]
            else:
                raise _err("private coordination hydrogen-hydrogen bond is unsupported")
            if hname not in generated:
                raise _err("private coordination hydrogen mapping is incomplete")
            pair = frozenset((heavy_atom.index, generated[hname].index))
            if pair not in existing:
                rebuilt.addBond(heavy_atom, generated[hname], order=1.0)
                existing.add(pair)
    return rebuilt, unit.Quantity(np.asarray(out_xyz), unit.nanometer), tuple(fixed)


@lru_cache(maxsize=1)
def _official_uff_table() -> dict[str, tuple[float, float]]:
    """Load the version-pinned RDKit UFF x1/D1 subset shipped with PartiDesign.

    ``GetUFFVdWParams`` remains the authoritative fast path.  Some RDKit
    releases do not emit a charge flag for every metal that is present in
    their own parameter table (notably ``Ir6+3``), so this static copy is used
    only after the public helper returns ``None``.  Values are x1 in Angstrom
    and D1 in kcal/mol, exactly as represented in RDKit's ``Params.cpp``.
    """
    path = resource_root() / "configs" / "resources" / "openmm" / "uff_nonbonded.tsv"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise _err(f"fixed coordination UFF table is unavailable: {path}") from exc
    table: dict[str, tuple[float, float]] = {}
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("label\t"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise _err(f"fixed coordination UFF table row {line_number} is malformed")
        label = fields[0].strip()
        if not label:
            raise _err(f"fixed coordination UFF table row {line_number} has no label")
        try:
            x1, d1 = float(fields[1]), float(fields[2])
        except ValueError as exc:
            raise _err(f"fixed coordination UFF table row {line_number} is not numeric") from exc
        if not (math.isfinite(x1) and math.isfinite(d1) and x1 > 0 and d1 >= 0):
            raise _err(f"fixed coordination UFF table row {line_number} has invalid values")
        if label in table:
            raise _err(f"fixed coordination UFF table repeats label {label!r}")
        table[label] = (x1, d1)
    if not table:
        raise _err("fixed coordination UFF table is empty")
    return table


def _fallback_uff_label(atom: Any, table: dict[str, tuple[float, float]]) -> str | None:
    """Build the exact RDKit UFF label for a failed public lookup.

    This mirrors RDKit's geometry suffix rules closely enough to recover
    labels whose charge flag is omitted by a release's ``AtomTyper``.  The
    lookup is table-exact: if neither the formal-charge label nor the
    uncharged label exists, the atom is unsupported and must fail explicitly.
    """
    symbol = str(atom.GetSymbol())
    base = symbol if len(symbol) > 1 else f"{symbol}_"
    suffix = ""
    atomic_number = int(atom.GetAtomicNum())
    if atomic_number:
        periodic = Chem.GetPeriodicTable()
        if (periodic.GetDefaultValence(atomic_number) == -1 or
                (periodic.GetNOuterElecs(atomic_number) != 1 and
                 periodic.GetNOuterElecs(atomic_number) != 7)):
            if atomic_number in _UFF_FORCED_SP3:
                suffix = "3"
            elif atomic_number == 80:  # Hg has the fixed UFF ``Hg1`` geometry.
                suffix = "1"
            else:
                hybridization = str(atom.GetHybridization())
                suffix = {
                    "S": "", "SP": "1", "SP3": "3", "SP2D": "4",
                    "SP3D": "5", "SP3D2": "6",
                }.get(hybridization, "")
                if hybridization == "SP2":
                    conjugated = any(bool(bond.GetIsConjugated())
                                     for bond in atom.GetBonds())
                    if (bool(atom.GetIsAromatic()) or conjugated) and atomic_number in {6, 7, 8, 16}:
                        suffix = "R"
                    else:
                        suffix = "2"
    root = base + suffix
    formal_charge = int(atom.GetFormalCharge())
    candidates = [f"{root}+{formal_charge}"] if formal_charge > 0 else []
    candidates.append(root)
    for label in candidates:
        if label in table:
            return label
    return None


def _uff(molecule: Chem.Mol):
    table = None
    result = []
    for i, atom in enumerate(molecule.GetAtoms()):
        pair = rdForceFieldHelpers.GetUFFVdWParams(molecule, i, i)
        if pair is None:
            if table is None:
                table = _official_uff_table()
            label = _fallback_uff_label(atom, table)
            if label is None:
                raise _err(
                    "RDKit UFF has no exact van der Waals parameters for "
                    f"{atom.GetSymbol()} formal charge {atom.GetFormalCharge()} "
                    f"hybridization {atom.GetHybridization()}"
                )
            pair = table[label]
        try:
            rmin, epsilon = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise _err("RDKit UFF returned malformed van der Waals parameters") from exc
        if not (math.isfinite(rmin) and math.isfinite(epsilon) and rmin > 0 and epsilon >= 0):
            raise _err("RDKit UFF returned non-finite van der Waals parameters")
        # RDKit exposes x1 in Angstrom and D1 in kcal/mol.  UFF defines
        # sigma = x1 * 2**(-1/6); OpenMM expects sigma in nm and epsilon in kJ/mol.
        result.append((rmin * _UFF_A_TO_NM * _SIGMA_FACTOR, epsilon * _KCAL_TO_KJ))
    return tuple(result)


def _fixed_template_signature(
    molecule: Chem.Mol,
    values: Sequence[tuple[float, float]],
    charges: Sequence[float],
) -> tuple[Any, ...]:
    """Return exact residue-template semantics for safe deduplication.

    The tuple contains every field OpenMM uses for matching/parameterization:
    atom names, elements, formal/partial charges, converted UFF parameters,
    and the trusted bond graph including bond types.  Equality is exact; no
    digest or neighboring-element proxy is used.
    """
    atoms = []
    for index, atom in enumerate(molecule.GetAtoms()):
        atom_name = atom.GetProp("name") if atom.HasProp("name") else f"A{index}"
        atoms.append((
            str(atom_name), _symbol(atom.GetSymbol()), int(atom.GetFormalCharge()),
            float(charges[index]), float(values[index][0]), float(values[index][1]),
        ))
    bonds = tuple(sorted(
        (
            min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            str(bond.GetBondType()),
        )
        for bond in molecule.GetBonds()
    ))
    return tuple(atoms), bonds


def register_fixed_coordination_template(forcefield: Any, record: FixedCoordinationRecord):
    """Register or safely reuse one exact fixed-coordination template."""
    from openmm import unit
    from openmm.app import ForceField, element
    from openmm.app.forcefield import NonbondedGenerator

    marker = "_partidesign_fixed_coordination"
    signature_marker = f"{marker}_signatures"
    registered = dict(getattr(forcefield, marker, {}))
    signatures = dict(getattr(forcefield, signature_marker, {}))
    if record.record_id in registered:
        template_name = registered[record.record_id]
        try:
            return forcefield._templates[template_name]
        except KeyError as exc:
            raise _err("fixed coordination template registry is stale") from exc

    molecule = _decode(record)
    values = _uff(molecule)
    charges = tuple(record.private_partial_charges)
    if len(charges) != molecule.GetNumAtoms() or any(not math.isfinite(float(v)) for v in charges):
        raise _err("fixed coordination partial charges are incomplete")
    formal = sum(int(atom.GetFormalCharge()) for atom in molecule.GetAtoms())
    if not math.isclose(sum(charges), formal, rel_tol=0, abs_tol=5e-4):
        raise _err("fixed coordination partial charges do not conserve integer charge")
    signature = _fixed_template_signature(molecule, values, charges)
    if signature in signatures:
        template_name = signatures[signature]
        if template_name not in forcefield._templates:
            raise _err("fixed coordination template signature registry is stale")
        registered[record.record_id] = template_name
        setattr(forcefield, marker, registered)
        setattr(forcefield, signature_marker, signatures)
        return forcefield._templates[template_name]

    generators = [g for g in forcefield._forces if isinstance(g, NonbondedGenerator)]
    if len(generators) > 1:
        raise _err("forcefield has conflicting NonbondedForce generators")
    generator = generators[0] if generators else NonbondedGenerator(forcefield, 1.0, 1.0, True)
    if not generators:
        forcefield.registerGenerator(generator)

    # Use a monotonic local serial rather than record IDs: identical records
    # share this template, while distinct signatures always receive distinct
    # atom types and cannot be accidentally matched as one chemistry.
    serial = len(signatures)
    template_name = f"_PD_FIXED_TEMPLATE_{serial}"
    while template_name in forcefield._templates:
        serial += 1
        template_name = f"_PD_FIXED_TEMPLATE_{serial}"
    types = []
    for i, atom in enumerate(molecule.GetAtoms()):
        name = f"PD_FC_TEMPLATE_{serial}_{i}"
        types.append(name)
        elem = element.get_by_symbol(_symbol(atom.GetSymbol()))
        # ForceField accepts an OpenMM mass quantity (and converts it to amu);
        # ``float(Element.mass)`` is invalid because the value is unit-bearing.
        forcefield.registerAtomType({"name": name, "class": name, "element": elem, "mass": elem.mass})
        sigma, epsilon = values[i]
        generator.registerAtom({"type": name, "charge": charges[i] * unit.elementary_charge,
                                "sigma": sigma * unit.nanometer, "epsilon": epsilon * unit.kilojoule_per_mole})
    template = ForceField._TemplateData(template_name)
    for i, atom in enumerate(molecule.GetAtoms()):
        atom_name = atom.GetProp("name") if atom.HasProp("name") else (
            record.heavy_keys[i][4] if i < len(record.heavy_keys) else f"H{i}"
        )
        sigma, epsilon = values[i]
        template.addAtom(ForceField._TemplateAtomData(
            atom_name, types[i], element.get_by_symbol(_symbol(atom.GetSymbol())),
            {"charge": charges[i] * unit.elementary_charge,
             "sigma": sigma * unit.nanometer, "epsilon": epsilon * unit.kilojoule_per_mole},
        ))
    for bond in molecule.GetBonds():
        template.addBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    forcefield.registerResidueTemplate(template)
    signatures[signature] = template_name
    registered[record.record_id] = template_name
    setattr(forcefield, marker, registered)
    setattr(forcefield, signature_marker, signatures)
    return template


def prepare_fixed_coordination(topology: Any, positions: Any, forcefield: Any,
                               records: Sequence[FixedCoordinationRecord]):
    """Expand private H and register templates for an ordinary OpenMM route."""
    topology, positions, keys = expand_fixed_coordination(topology, positions, records)
    for record in records:
        register_fixed_coordination_template(forcefield, record)
    return topology, positions, keys


def register_nonbonded_template(forcefield: Any, record: FixedCoordinationRecord):
    return register_fixed_coordination_template(forcefield, record)


def expand_coordination_topology(*args: Any, **kwargs: Any):
    return expand_fixed_coordination(*args, **kwargs)


__all__ = [
    "AtomKey", "FixedCoordinationRecord", "build_fixed_coordination_record",
    "validate_fixed_coordination", "fixed_coordination_keys",
    "expand_fixed_coordination", "expand_coordination_topology",
    "register_fixed_coordination_template", "register_nonbonded_template",
    "prepare_fixed_coordination",
]
