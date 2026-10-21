"""Private completion of explicitly identified, incomplete fixed molecular context.

Only missing atoms of the source's exact trusted CCD component are appended.
The accepted public heavy inventory and graph are never changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from biotite import structure as struc
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from ..context import (
    _CONTEXT_COMPONENTS, _load_components, _apply_source_charges,
    context_component_name,
)
from ..runner import AtomKey, PostprocessPreparationError, _atom_key

_ORDER = {
    Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 9,
}


@dataclass(frozen=True, slots=True)
class FixedLigandCompletion:
    heavy_keys: tuple[AtomKey, ...]
    component_name: str
    rdkit_binary: bytes


def fixed_ligand_completions(candidate, *, excluded_keys=()):
    atoms = candidate.atoms
    excluded = set(excluded_keys)
    result = []
    for uid in np.unique(atoms.component_uid):
        rows = tuple(int(i) for i in np.flatnonzero(atoms.component_uid == uid))
        if (len(rows) <= 1 or any(atoms.is_design[i] for i in rows)
                or not all(atoms.mol_type[i] == "ligand" for i in rows)
                or any(_atom_key(atoms, i) in excluded for i in rows)):
            continue
        group = set(rows)
        if any((int(i) in group) != (int(j) in group)
               for i, j, _ in atoms.bonds.as_array()):
            continue
        name = context_component_name(atoms, rows, source_path=candidate.base_request.task.input_path)
        _load_components({name})
        component = _CONTEXT_COMPONENTS.get(name)
        if component is None:
            continue
        molecule = Chem.RemoveHs(Chem.Mol(component.rdkit_binary), sanitize=True)
        if molecule.GetNumAtoms() <= len(rows):
            continue
        if any(b.GetBondType() not in _ORDER for b in molecule.GetBonds()):
            continue
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise PostprocessPreparationError("fixed CCD completion requires one connected molecule")
        _match_retained(atoms, rows, molecule)
        molecule.RemoveAllConformers()
        result.append(FixedLigandCompletion(
            tuple(_atom_key(atoms, i) for i in rows), name,
            bytes(molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)),
        ))
    return tuple(result)


def _match_retained(atoms, rows, molecule):
    names = {}
    for i, atom in enumerate(molecule.GetAtoms()):
        if not atom.HasProp("name") or not atom.GetProp("name").strip():
            raise PostprocessPreparationError("fixed CCD completion requires named atoms")
        name = atom.GetProp("name").strip()
        if name in names:
            raise PostprocessPreparationError("fixed CCD atom names are ambiguous")
        names[name] = i
    local = {}
    for row in rows:
        name = str(atoms.atom_name[row])
        index = names.get(name)
        if index is None or index in local.values():
            raise PostprocessPreparationError("retained atom cannot map uniquely to the fixed CCD")
        if molecule.GetAtomWithIdx(index).GetSymbol().upper() != str(atoms.element[row]).upper():
            raise PostprocessPreparationError("retained fixed CCD atom has a different element")
        local[int(row)] = index
    expected = {}
    selected = set(local.values())
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in selected and j in selected:
            expected[tuple(sorted((i, j)))] = _ORDER[bond.GetBondType()]
    observed = {}
    for i, j, order in atoms.bonds.as_array():
        if int(i) in local and int(j) in local:
            observed[tuple(sorted((local[int(i)], local[int(j)])))] = int(order)
    if set(expected) != set(observed):
        raise PostprocessPreparationError("retained fixed CCD edge inventory is inconsistent")
    for pair, order in expected.items():
        raw = observed[pair]
        if raw != order and not (order == 9 and raw in {5, 6, 7, 9}):
            raise PostprocessPreparationError("retained fixed CCD bond order is inconsistent")
    return local


def _completion_coordinates(molecule, known):
    molecule = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    params.useRandomCoords = True
    params.maxIterations = 20
    params.SetCoordMap({int(i): Point3D(*map(float, xyz)) for i, xyz in known.items()})
    if AllChem.EmbedMolecule(molecule, params) < 0:
        # Accepted flex coordinates may not satisfy RDKit distance bounds.
        # Build only the missing region from a trusted stereochemical template,
        # align its attachment frame, and optimize with every observed atom fixed.
        molecule.RemoveAllConformers()
        params.SetCoordMap({})
        if AllChem.EmbedMolecule(molecule, params) < 0:
            raise PostprocessPreparationError("fixed CCD missing-atom geometry could not be completed")
        missing = {a.GetIdx() for a in molecule.GetAtoms()
                   if a.GetAtomicNum() > 1 and a.GetIdx() not in known}
        frontier = {n.GetIdx() for i in missing
                    for n in molecule.GetAtomWithIdx(i).GetNeighbors()
                    if n.GetIdx() in known}
        anchors, visited = [], set()
        while frontier:
            current = sorted(frontier)
            anchors.extend(current)
            visited.update(current)
            if len(anchors) >= 3:
                break
            frontier = {n.GetIdx() for i in current
                        for n in molecule.GetAtomWithIdx(i).GetNeighbors()
                        if n.GetIdx() in known and n.GetIdx() not in visited}
        if len(anchors) < 3:
            raise PostprocessPreparationError("fixed CCD completion has no observed attachment frame")
        conf = molecule.GetConformer()
        source = np.asarray([conf.GetAtomPosition(i) for i in anchors], dtype=float)
        target = np.asarray([known[i] for i in anchors], dtype=float)
        left = source - source.mean(axis=0)
        right = target - target.mean(axis=0)
        if np.linalg.matrix_rank(left) < 2 or np.linalg.matrix_rank(right) < 2:
            raise PostprocessPreparationError("fixed CCD completion attachment frame is degenerate")
        u, _, vt = np.linalg.svd(left.T @ right)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vt)
        rotation = u @ correction @ vt
        coordinates = (np.asarray(conf.GetPositions()) - source.mean(axis=0)) @ rotation + target.mean(axis=0)
        for i, xyz in enumerate(coordinates):
            conf.SetAtomPosition(i, tuple(map(float, xyz)))
    conf = molecule.GetConformer()
    for i, xyz in known.items():
        conf.SetAtomPosition(int(i), tuple(map(float, xyz)))
    forcefield = AllChem.UFFGetMoleculeForceField(molecule)
    if forcefield is None:
        raise PostprocessPreparationError("fixed CCD completion lacks UFF geometry parameters")
    for i in known:
        forcefield.AddFixedPoint(int(i))
    forcefield.Initialize()
    forcefield.Minimize(maxIts=200)
    for i, xyz in known.items():
        if not np.allclose(np.asarray(conf.GetAtomPosition(int(i))), xyz, rtol=0, atol=1e-6):
            raise PostprocessPreparationError("fixed CCD completion moved an observed atom")
    xyz = np.asarray(conf.GetPositions(), dtype=float)
    if not np.isfinite(xyz).all() or not np.isfinite(forcefield.CalcEnergy()):
        raise PostprocessPreparationError("fixed CCD completion returned non-finite geometry")
    return Chem.RemoveHs(molecule, sanitize=True)


def expand_fixed_ligands(atoms, specs: Sequence[FixedLigandCompletion], *, source_path=None, formal_charge_facts=()):
    """Append only private missing atoms and return full GAFF records."""
    from ..runner import gaff_record_from_rdkit
    public_length = len(atoms)
    view = atoms.copy()
    records = []
    next_uid = int(np.max(atoms.atom_uid)) + 1
    for ordinal, spec in enumerate(specs):
        by_key = {_atom_key(view, i): i for i in range(len(view))}
        if not set(spec.heavy_keys).issubset(by_key):
            raise PostprocessPreparationError("fixed CCD completion references absent atoms")
        rows = tuple(by_key[k] for k in spec.heavy_keys)
        if any(view.is_design[i] for i in rows):
            raise PostprocessPreparationError("fixed CCD completion cannot include design atoms")
        reference = Chem.Mol(spec.rdkit_binary)
        mapping = _match_retained(view, rows, reference)
        known = {i: view.coord[row].astype(float) for row, i in mapping.items()}
        completed = _completion_coordinates(reference, known)
        reverse = {i: row for row, i in mapping.items()}
        extras = []
        for i, atom in enumerate(completed.GetAtoms()):
            if i in reverse:
                continue
            extra = view[[rows[0]]].copy()
            extra.atom_name[0] = atom.GetProp("name")
            extra.element[0] = atom.GetSymbol().upper()
            extra.coord[0] = completed.GetConformer().GetAtomPosition(i)
            extra.atom_uid[0] = next_uid
            next_uid += 1
            extra.is_design[0] = False
            for annotation in ("centre_atom_mask", "identity_condition", "motif_coord_condition",
                               "token_center_condition", "hotspot_condition"):
                if annotation in extra.get_annotation_categories():
                    extra.get_annotation(annotation)[0] = False
            if "coord_valid" in extra.get_annotation_categories():
                extra.coord_valid[0] = True
            reverse[i] = len(view) + len(extras)
            extras.append(extra)
        original_edges = list(view.bonds.as_array())
        if extras:
            view = struc.concatenate([view, *extras])
        bonds = struc.BondList(len(view), np.asarray(original_edges, dtype=np.uint32).reshape(-1, 3))
        for bond in completed.GetBonds():
            i, j = reverse[bond.GetBeginAtomIdx()], reverse[bond.GetEndAtomIdx()]
            if i >= public_length or j >= public_length:
                bonds.add_bond(i, j, _ORDER[bond.GetBondType()])
        view.bonds = bonds
        all_rows = tuple(reverse[i] for i in range(completed.GetNumAtoms()))
        completed = _apply_source_charges(
            completed, view, all_rows, source_path, component_name=spec.component_name,
            frozen_facts=formal_charge_facts)
        # Preserve accepted observed heavy positions through the explicit-H conversion.
        conf = completed.GetConformer()
        for i, row in enumerate(all_rows):
            conf.SetAtomPosition(i, tuple(map(float, view.coord[row])))
        records.append(gaff_record_from_rdkit(
            completed, record_id=f"fixed-ccd-completion-{ordinal}",
            molecule_uid=int(view.molecule_uid[rows[0]]),
            component_uid=int(view.component_uid[rows[0]]),
            heavy_keys=tuple(_atom_key(view, i) for i in all_rows),
        ))
    if not np.array_equal(view.coord[:public_length], atoms.coord):
        raise PostprocessPreparationError("fixed CCD completion changed accepted coordinates")
    return view, tuple(records)
