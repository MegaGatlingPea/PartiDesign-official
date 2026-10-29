"""Ligand graph construction and hard-graph reconciliation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import os
from pathlib import Path
import pickle
from typing import Iterable, Sequence

import numpy as np
from biotite import structure as struc
from rdkit import Chem
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, slots=True)
class LigandGraph:
    """A concrete heavy-atom graph in authoring-row order."""

    bonds: tuple[tuple[int, int, int], ...]
    aromatic_atoms: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class TrustedCCDComponent:
    """Owner-local CCD source chemistry or an explicitly projected product."""

    component_id: str
    rdkit_binary: bytes

    def __post_init__(self) -> None:
        if not self.component_id or self.component_id != self.component_id.upper():
            raise ValueError("trusted CCD component ID must be non-empty uppercase text")
        if not isinstance(self.rdkit_binary, bytes) or not self.rdkit_binary:
            raise ValueError("trusted CCD component binary must be non-empty bytes")


_RD_TO_BIOTITE = {
    Chem.BondType.SINGLE: int(struc.BondType.SINGLE),
    Chem.BondType.DOUBLE: int(struc.BondType.DOUBLE),
    Chem.BondType.TRIPLE: int(struc.BondType.TRIPLE),
    Chem.BondType.AROMATIC: int(struc.BondType.AROMATIC),
}

def load_trusted_ccd_components(
    ccd_pickle_path: Path,
    component_ids: Iterable[str],
) -> dict[str, TrustedCCDComponent]:
    """Read requested molecules from the configured local CCD asset."""

    requested = tuple(sorted({str(value).strip().upper() for value in component_ids}))
    if not requested or any(not value for value in requested):
        raise ValueError("trusted CCD component request must be non-empty")
    try:
        with Path(ccd_pickle_path).open("rb") as stream:
            mapping = pickle.load(stream)
    except (EOFError, pickle.PickleError) as exc:
        raise ValueError("CCD asset cannot be decoded") from exc
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("trusted CCD asset is not a non-empty component mapping")
    result: dict[str, TrustedCCDComponent] = {}
    for component_id in requested:
        molecule = mapping.get(component_id)
        if not isinstance(molecule, Chem.Mol):
            raise ValueError(f"trusted CCD has no unique molecule for {component_id!r}")
        binary = bytes(
            molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)
        )
        result[component_id] = TrustedCCDComponent(component_id, binary)
    del mapping
    gc.collect()
    return result


def project_trusted_ccd_component(
    component: TrustedCCDComponent,
    omitted_atom_names: Sequence[str],
) -> TrustedCCDComponent:
    """Project an explicitly declared product from a validated CCD precursor.

    The compiler separately proves the complete precursor and the retained
    named graph. This private fragment is not a proof of the original CCD ID.
    """
    names = tuple(omitted_atom_names)
    if not names or len(set(names)) != len(names) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("CCD product omissions require unique non-empty atom names")
    molecule = Chem.RemoveHs(Chem.Mol(component.rdkit_binary), sanitize=True)
    by_name = {
        atom.GetProp("name"): atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.HasProp("name")
    }
    if len(by_name) != molecule.GetNumAtoms() or any(
        name not in by_name for name in names
    ):
        raise ValueError("CCD product omission does not match named precursor atoms")
    omitted_indices = {by_name[name] for name in names}
    affected_names = {
        neighbor.GetProp("name")
        for index in omitted_indices
        for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
        if neighbor.GetIdx() not in omitted_indices
    }
    editable = Chem.RWMol(molecule)
    for index in sorted(omitted_indices, reverse=True):
        editable.RemoveAtom(index)
    product = editable.GetMol()
    for atom in product.GetAtoms():
        if atom.GetProp("name") in affected_names:
            # Cap the private isolated fragment by normal implicit valence.
            # Whole-product preparation later consumes H at external links.
            atom.SetNoImplicit(False)
    if not product.GetNumAtoms():
        raise ValueError("CCD product omission removed the entire component")
    Chem.SanitizeMol(product)
    if len(Chem.GetMolFrags(product)) != 1:
        raise ValueError("CCD product omission disconnected the retained component")
    return TrustedCCDComponent(
        component.component_id,
        bytes(product.ToBinary(Chem.PropertyPickleOptions.AllProps)),
    )


def _bond_order_key(bond_type: object) -> int:
    value = _RD_TO_BIOTITE.get(bond_type)
    if value is None:
        raise ValueError(f"trusted CCD has unsupported bond order {bond_type}")
    return value


def trusted_ccd_private_molecule(
    atom_array: struc.AtomArray,
    component_rows: Sequence[int],
    component_id: str,
    component: TrustedCCDComponent,
) -> Chem.Mol:
    """Resolve a hard named graph against trusted CCD source/product chemistry.

    The match is deliberately exact: atom names, elements, internal endpoints and
    orders must form one bijection.  Formal charges and stereochemistry then come
    from that matched CCD molecule; neither is guessed from the public graph.
    """

    normalized_id = str(component_id).strip().upper()
    if component.component_id != normalized_id:
        raise ValueError("trusted CCD component carrier has the wrong identity")
    try:
        raw = Chem.Mol(component.rdkit_binary)
    except Exception as exc:
        raise ValueError("trusted CCD component binary cannot be decoded") from exc
    source = Chem.RemoveHs(raw, sanitize=True)
    if source.GetNumAtoms() == 0 or len(Chem.GetMolFrags(source)) != 1:
        raise ValueError(f"trusted CCD molecule {component_id!r} is not connected")

    rows = np.asarray(component_rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0 or len(set(rows.tolist())) != len(rows):
        raise ValueError("hard ligand rows must be a non-empty unique sequence")
    if source.GetNumAtoms() != len(rows):
        raise ValueError(
            f"trusted CCD atom count for {component_id!r} does not match hard graph"
        )
    names = [str(atom_array.atom_name[int(row)]).strip() for row in rows]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("hard ligand atom names are missing or ambiguous")
    source_by_name: dict[str, int] = {}
    for index, atom in enumerate(source.GetAtoms()):
        if not atom.HasProp("name"):
            raise ValueError(f"trusted CCD atom for {component_id!r} lacks a name")
        name = atom.GetProp("name").strip()
        if not name or name in source_by_name:
            raise ValueError(f"trusted CCD atom names for {component_id!r} are ambiguous")
        source_by_name[name] = index
    if set(source_by_name) != set(names):
        raise ValueError(
            f"trusted CCD atom names for {component_id!r} do not match hard graph"
        )

    source_order = [source_by_name[name] for name in names]
    ordered = Chem.RenumberAtoms(source, source_order)
    expected_elements = [str(atom_array.element[int(row)]).strip().upper() for row in rows]
    observed_elements = [atom.GetSymbol().upper() for atom in ordered.GetAtoms()]
    if observed_elements != expected_elements:
        raise ValueError(
            f"trusted CCD elements for {component_id!r} do not match hard atom types"
        )

    row_to_local = {int(row): index for index, row in enumerate(rows.tolist())}
    hard_edges: dict[tuple[int, int], int] = {}
    if atom_array.bonds is not None:
        for raw_i, raw_j, raw_type in atom_array.bonds.as_array():
            row_i, row_j = int(raw_i), int(raw_j)
            if row_i not in row_to_local or row_j not in row_to_local:
                continue
            key = tuple(sorted((row_to_local[row_i], row_to_local[row_j])))
            order = int(raw_type)
            previous = hard_edges.get(key)
            if previous is not None and previous != order:
                raise ValueError("hard ligand graph contains conflicting bond orders")
            hard_edges[key] = order
    ccd_edges = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))): _bond_order_key(
            bond.GetBondType()
        )
        for bond in ordered.GetBonds()
    }
    same_edges = hard_edges.keys() == ccd_edges.keys()
    same_orders = same_edges and all(
        hard_edges[key] == ccd_order
        or (
            ccd_order == int(struc.BondType.AROMATIC)
            and hard_edges[key]
            in {
                int(struc.BondType.SINGLE),
                int(struc.BondType.DOUBLE),
                int(struc.BondType.AROMATIC_SINGLE),
                int(struc.BondType.AROMATIC_DOUBLE),
            }
        )
        for key, ccd_order in ccd_edges.items()
    )
    if not same_orders:
        raise ValueError(
            f"trusted CCD bond endpoints/orders for {component_id!r} do not match hard graph"
        )

    charges = [atom.GetFormalCharge() for atom in ordered.GetAtoms()]
    rebuilt = private_molecule_from_authoring_graph(
        atom_array,
        rows,
        charges,
        trusted_hydrogen_counts=[atom.GetTotalNumHs() for atom in ordered.GetAtoms()],
    )
    if Chem.MolToSmiles(rebuilt, isomericSmiles=False) != Chem.MolToSmiles(
        ordered, isomericSmiles=False
    ):
        raise ValueError(
            f"trusted CCD chemistry for {component_id!r} is not equivalent to hard graph"
        )
    return ordered


def private_molecule_from_trusted_graph(
    atom_array: struc.AtomArray,
    component_rows: Sequence[int],
    trusted: Chem.Mol,
) -> Chem.Mol:
    """Apply final coordinates to a validated row-ordered trusted molecule."""

    rows = np.asarray(component_rows, dtype=np.int64)
    source = Chem.Mol(trusted)
    charges = [atom.GetFormalCharge() for atom in source.GetAtoms()]
    rebuilt = private_molecule_from_authoring_graph(
        atom_array,
        rows,
        charges,
        trusted_hydrogen_counts=[atom.GetTotalNumHs() for atom in source.GetAtoms()],
    )
    if Chem.MolToSmiles(rebuilt, isomericSmiles=False) != Chem.MolToSmiles(
        source, isomericSmiles=False
    ):
        raise ValueError("final hard ligand graph drifted from trusted chemistry")
    source.RemoveAllConformers()
    conformer = Chem.Conformer(len(rows))
    for local, row in enumerate(rows):
        conformer.SetAtomPosition(
            local, tuple(float(value) for value in atom_array.coord[int(row)])
        )
    source.AddConformer(conformer, assignId=True)
    Chem.SanitizeMol(source)
    return source


def _atomic_numbers(elements: Sequence[object]) -> list[int]:
    table = Chem.GetPeriodicTable()
    result: list[int] = []
    for raw in elements:
        text = str(raw).strip()
        if not text:
            raise ValueError("empty ligand element")
        symbol = text[0].upper() + text[1:].lower()
        number = int(table.GetAtomicNumber(symbol))
        if number <= 1:
            raise ValueError(f"unsupported generated heavy element {symbol!r}")
        result.append(number)
    return result


def _rdkit_to_authoring_rows(
    molecule: Chem.Mol,
    coordinates: np.ndarray,
    atomic_numbers: Sequence[int],
) -> np.ndarray:
    if molecule.GetNumAtoms() != len(atomic_numbers):
        raise ValueError("reconstructed graph changed the heavy-atom count")
    conformer = molecule.GetConformer()
    reconstructed = np.asarray(
        [list(conformer.GetAtomPosition(i)) for i in range(molecule.GetNumAtoms())],
        dtype=np.float64,
    )
    rd_numbers = np.asarray([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
    source_numbers = np.asarray(atomic_numbers)
    cost = np.linalg.norm(reconstructed[:, None, :] - coordinates[None, :, :], axis=-1)
    cost[rd_numbers[:, None] != source_numbers[None, :]] = np.inf
    rd_rows, source_rows = linear_sum_assignment(cost)
    if len(rd_rows) != len(atomic_numbers) or not np.isfinite(cost[rd_rows, source_rows]).all():
        raise ValueError("reconstructed atoms cannot be matched by element and coordinate")
    if float(cost[rd_rows, source_rows].max(initial=0.0)) > 1e-5:
        raise ValueError("OpenBabel reconstruction moved or replaced a heavy atom")
    mapping = np.empty(len(atomic_numbers), dtype=np.int64)
    mapping[rd_rows] = source_rows
    if len(set(mapping.tolist())) != len(mapping):
        raise ValueError("reconstructed atom mapping is not bijective")
    return mapping


def _obabel_molecule(
    coordinates_angstrom: np.ndarray,
    elements: Sequence[object],
) -> tuple[Chem.Mol, np.ndarray]:
    """Return the official reconstructed molecule and its private row mapping."""

    coordinates = np.asarray(coordinates_angstrom, dtype=np.float64)
    if coordinates.shape != (len(elements), 3) or not np.isfinite(coordinates).all():
        raise ValueError("OpenBabel requires finite heavy-atom coordinates")
    atomic_numbers = _atomic_numbers(elements)
    # Importing OpenBabel is intentionally delayed until this explicit method is used.
    from .ligand_reconstruction import MolReconsError, reconstruct_from_generated

    try:
        molecule = reconstruct_from_generated(
            coordinates,
            atomic_numbers,
            aromatic=None,
            basic_mode=True,
        )
    except MolReconsError as error:
        raise ValueError(
            "OpenBabel reconstruction could not produce a valid molecule"
        ) from error
    if molecule is None:
        raise ValueError("OpenBabel reconstruction returned no molecule")
    Chem.SanitizeMol(molecule)
    rd_to_source = _rdkit_to_authoring_rows(molecule, coordinates, atomic_numbers)
    return molecule, rd_to_source


def ligand_graph_from_rdkit(
    molecule: Chem.Mol,
    rd_to_source: Sequence[int] | None = None,
) -> LigandGraph:
    """Convert one sanitized RDKit molecule into an authoring-row graph."""

    if rd_to_source is None:
        rd_to_source = tuple(range(molecule.GetNumAtoms()))
    if len(rd_to_source) != molecule.GetNumAtoms():
        raise ValueError("RDKit-to-authoring map has the wrong length")
    bonds: list[tuple[int, int, int]] = []
    for bond in molecule.GetBonds():
        bond_type = _RD_TO_BIOTITE.get(bond.GetBondType())
        if bond_type is None:
            raise ValueError(f"unsupported reconstructed bond type {bond.GetBondType()}")
        atom_i = int(rd_to_source[bond.GetBeginAtomIdx()])
        atom_j = int(rd_to_source[bond.GetEndAtomIdx()])
        bonds.append((min(atom_i, atom_j), max(atom_i, atom_j), bond_type))
    bonds.sort()
    aromatic = [False] * len(rd_to_source)
    for rd_index, atom in enumerate(molecule.GetAtoms()):
        aromatic[int(rd_to_source[rd_index])] = bool(atom.GetIsAromatic())
    return LigandGraph(tuple(bonds), tuple(aromatic))


def reconstruct_obabel(
    coordinates_angstrom: np.ndarray,
    elements: Sequence[object],
) -> LigandGraph:
    """Run the selected obabel/liGAN route pinned to its attributed sources."""

    molecule, rd_to_source = _obabel_molecule(coordinates_angstrom, elements)
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("OpenBabel reconstruction produced a disconnected ligand")
    return ligand_graph_from_rdkit(molecule, rd_to_source)

def reconstruct_obabel_with_molecule(
    coordinates_angstrom: np.ndarray,
    elements: Sequence[object],
) -> tuple[LigandGraph, Chem.Mol]:
    """Return the public heavy graph and private sanitized chemistry object.

    The RDKit molecule is deliberately a private sidecar: its atom order and
    charge/H-perception state are not part of the authoring or writer ABI.
    """

    molecule, rd_to_source = _obabel_molecule(coordinates_angstrom, elements)
    graph = ligand_graph_from_rdkit(molecule, rd_to_source)
    # Private payload atom order follows authoring rows even though public
    # chemistry acceptance never compares atom order.
    source_to_rd = np.empty(len(rd_to_source), dtype=np.int64)
    for rd_index, source_index in enumerate(rd_to_source):
        source_to_rd[int(source_index)] = int(rd_index)
    ordered = Chem.RenumberAtoms(molecule, source_to_rd.tolist())
    return graph, ordered


def private_molecule_from_authoring_graph(
    atom_array: struc.AtomArray,
    component_rows: Sequence[int],
    formal_charges: Sequence[int],
    *,
    trusted_hydrogen_counts: Sequence[int] | None = None,
) -> Chem.Mol:
    """Build a sanitized private RDKit payload in authoring-row order.

    This helper is intentionally not a public structure contract. Formal
    charges come from a private decoder/SMILES/source-chemistry path and never
    become AtomArray annotations or CIF fields.

    Trusted wrappers may additionally supply exact row-ordered total H counts.
    Only that authority enables strict aromatic-subtype Kekule validation;
    no-reference callers retain their existing aromatic normalization semantics.
    """

    rows = np.asarray(component_rows, dtype=np.int64)
    charges = [int(value) for value in formal_charges]
    if rows.ndim != 1 or len(rows) == 0 or len(set(rows.tolist())) != len(rows):
        raise ValueError("private ligand rows must be a non-empty unique sequence")
    if len(charges) != len(rows):
        raise ValueError("private ligand charge count does not match authoring rows")
    hydrogen_counts = None
    if trusted_hydrogen_counts is not None:
        hydrogen_counts = list(trusted_hydrogen_counts)
        if len(hydrogen_counts) != len(rows) or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
            for value in hydrogen_counts
        ):
            raise ValueError(
                "trusted hydrogen counts must be row-aligned nonnegative integers"
            )
    row_to_local = {int(row): index for index, row in enumerate(rows.tolist())}
    subtype_orders = {
        int(struc.BondType.AROMATIC_SINGLE),
        int(struc.BondType.AROMATIC_DOUBLE),
        int(struc.BondType.AROMATIC_TRIPLE),
    }
    strict_aromatic = (
        hydrogen_counts is not None
        and atom_array.bonds is not None
        and any(
            int(i) in row_to_local
            and int(j) in row_to_local
            and int(order) in subtype_orders
            for i, j, order in atom_array.bonds.as_array()
        )
    )
    table = Chem.GetPeriodicTable()
    editable = Chem.RWMol()
    for row, charge in zip(rows, charges):
        symbol = str(atom_array.element[int(row)]).strip()
        normalized = symbol[:1].upper() + symbol[1:].lower()
        atomic_number = int(table.GetAtomicNumber(normalized))
        if atomic_number <= 1:
            raise ValueError(f"private ligand has unsupported heavy element {symbol!r}")
        atom = Chem.Atom(atomic_number)
        atom.SetFormalCharge(charge)
        if hydrogen_counts is not None:
            # Preserve trusted [nH]/charge chemistry without imposing Kekule
            # subtype validation on a generic aromatic graph.
            atom.SetNumExplicitHs(int(hydrogen_counts[row_to_local[int(row)]]))
            atom.SetNoImplicit(True)
        editable.AddAtom(atom)
    biotite_to_rdkit = {
        int(struc.BondType.SINGLE): Chem.BondType.SINGLE,
        int(struc.BondType.DOUBLE): Chem.BondType.DOUBLE,
        int(struc.BondType.TRIPLE): Chem.BondType.TRIPLE,
        int(struc.BondType.AROMATIC): Chem.BondType.AROMATIC,
        int(struc.BondType.AROMATIC_SINGLE): (
            Chem.BondType.SINGLE if strict_aromatic else Chem.BondType.AROMATIC
        ),
        int(struc.BondType.AROMATIC_DOUBLE): (
            Chem.BondType.DOUBLE if strict_aromatic else Chem.BondType.AROMATIC
        ),
        int(struc.BondType.AROMATIC_TRIPLE): (
            Chem.BondType.TRIPLE if strict_aromatic else Chem.BondType.AROMATIC
        ),
    }
    aromatic_constraints: list[tuple[int, int, Chem.BondType]] = []
    known_constraints: list[tuple[int, int, Chem.BondType]] = []
    if atom_array.bonds is not None:
        for raw_i, raw_j, raw_type in atom_array.bonds.as_array():
            row_i, row_j = int(raw_i), int(raw_j)
            if row_i not in row_to_local or row_j not in row_to_local:
                continue
            rd_type = biotite_to_rdkit.get(int(raw_type))
            if rd_type is None:
                raise ValueError("private ligand graph requires concrete covalent orders")
            local_i, local_j = row_to_local[row_i], row_to_local[row_j]
            editable.AddBond(local_i, local_j, rd_type)
            if strict_aromatic and int(raw_type) != int(struc.BondType.AROMATIC):
                known_constraints.append((local_i, local_j, rd_type))
            if strict_aromatic and int(raw_type) in subtype_orders:
                aromatic_constraints.append((local_i, local_j, rd_type))
            if rd_type == Chem.BondType.AROMATIC:
                editable.GetAtomWithIdx(local_i).SetIsAromatic(True)
                editable.GetAtomWithIdx(local_j).SetIsAromatic(True)
    molecule = editable.GetMol()
    conformer = Chem.Conformer(len(rows))
    for local, row in enumerate(rows):
        x, y, z = (float(value) for value in atom_array.coord[int(row)])
        conformer.SetAtomPosition(local, (x, y, z))
    molecule.AddConformer(conformer, assignId=True)
    if aromatic_constraints:
        # Trusted H fixes double-bond capacity when generic aromatic edges mix
        # with known S/D/T. Do not mark known bonds aromatic before Kekulization:
        # RDKit would then reset their explicit orders. No search or H guessing.
        Chem.SanitizeMol(
            molecule,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
        )
        for i, j, expected in known_constraints:
            if molecule.GetBondBetweenAtoms(i, j).GetBondType() != expected:
                raise ValueError("private aromatic graph changed a known Kekule order")
    Chem.SanitizeMol(molecule)
    for i, j, _ in aromatic_constraints:
        if not molecule.GetBondBetweenAtoms(i, j).GetIsAromatic():
            raise ValueError("private aromatic subtype is not chemically aromatic")
    if strict_aromatic and (
        [atom.GetTotalNumHs() for atom in molecule.GetAtoms()] != hydrogen_counts
    ):
        raise ValueError("private aromatic graph changed trusted hydrogen counts")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("private ligand graph is disconnected")
    return molecule


def reconstruct_obabel_molecule(
    coordinates_angstrom: np.ndarray,
    elements: Sequence[object],
) -> Chem.Mol:
    """Return the private sanitized molecule for optional parameterization."""

    molecule, _ = _obabel_molecule(coordinates_angstrom, elements)
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("OpenBabel reconstruction produced a disconnected ligand")
    return molecule


def replace_component_bonds(
    atom_array: struc.AtomArray,
    component_rows: Sequence[int],
    graph: LigandGraph,
) -> None:
    """Replace one free/partial ligand's internal graph without touching other edges."""

    rows = np.asarray(component_rows, dtype=np.int64)
    if rows.ndim != 1 or len(set(rows.tolist())) != len(rows):
        raise ValueError("component rows must be unique")
    if atom_array.bonds is None:
        existing = np.empty((0, 3), dtype=np.int64)
    else:
        existing = np.asarray(atom_array.bonds.as_array(), dtype=np.int64)
    component_set = set(rows.tolist())
    kept = [
        row.tolist()
        for row in existing
        if not (int(row[0]) in component_set and int(row[1]) in component_set)
    ]
    for local_i, local_j, bond_type in graph.bonds:
        if not 0 <= local_i < len(rows) or not 0 <= local_j < len(rows):
            raise ValueError("ligand graph endpoint exceeds the planned atom inventory")
        kept.append([int(rows[local_i]), int(rows[local_j]), int(bond_type)])
    atom_array.bonds = struc.BondList(
        len(atom_array), np.asarray(kept, dtype=np.uint32).reshape((-1, 3))
    )


def reconcile_required_edges(
    atom_array: struc.AtomArray,
    required_edges: Iterable[tuple[int, int, int]],
    *,
    complete_components: Iterable[tuple[Sequence[int], Iterable[tuple[int, int]]]] = (),
    forbidden_edges: Iterable[tuple[int, int]] = (),
) -> None:
    """Overlay required edges and remove scoped hard non-edges by stable atom UID."""

    if "atom_uid" not in atom_array.get_annotation_categories():
        raise ValueError("hard reconciliation requires finalized atom_uid")
    uid_to_row = {
        int(uid): row for row, uid in enumerate(np.asarray(atom_array.atom_uid).tolist())
    }
    required_edges = tuple(
        (int(uid_i), int(uid_j), int(bond_type))
        for uid_i, uid_j, bond_type in required_edges
    )
    forbidden = {tuple(sorted((int(uid_i), int(uid_j)))) for uid_i, uid_j in forbidden_edges}
    if any(uid_i == uid_j for uid_i, uid_j in forbidden):
        raise ValueError("forbidden edges cannot contain self-edges")
    conflict = forbidden.intersection(
        tuple(sorted((uid_i, uid_j))) for uid_i, uid_j, _ in required_edges
    )
    if conflict:
        raise ValueError(f"hard graph has positive/negative conflicts: {sorted(conflict)}")
    edges: dict[tuple[int, int], int] = {}
    if atom_array.bonds is not None:
        for row_i, row_j, bond_type in atom_array.bonds.as_array():
            key = (min(int(row_i), int(row_j)), max(int(row_i), int(row_j)))
            edges[key] = int(bond_type)

    for uid_i, uid_j in forbidden:
        # An unprotected free atom may have been removed by legacy fragment
        # salvage. A missing negative endpoint has no surviving edge to remove.
        if uid_i not in uid_to_row or uid_j not in uid_to_row:
            continue
        edges.pop(tuple(sorted((uid_to_row[uid_i], uid_to_row[uid_j]))), None)

    for raw_uids, raw_positive in complete_components:
        rows = [uid_to_row[int(uid)] for uid in raw_uids]
        positive = {
            tuple(sorted((uid_to_row[int(a)], uid_to_row[int(b)])))
            for a, b in raw_positive
        }
        for position, row_i in enumerate(rows):
            for row_j in rows[position + 1 :]:
                key = tuple(sorted((row_i, row_j)))
                if key not in positive:
                    edges.pop(key, None)

    refinable_orders = {
        int(struc.BondType.SINGLE),
        int(struc.BondType.DOUBLE),
        int(struc.BondType.TRIPLE),
        int(struc.BondType.AROMATIC),
        int(struc.BondType.AROMATIC_SINGLE),
        int(struc.BondType.AROMATIC_DOUBLE),
        int(struc.BondType.AROMATIC_TRIPLE),
    }
    for uid_i, uid_j, bond_type in required_edges:
        if int(uid_i) not in uid_to_row or int(uid_j) not in uid_to_row:
            raise ValueError("required bond endpoint is absent after decode")
        key = tuple(sorted((uid_to_row[int(uid_i)], uid_to_row[int(uid_j)])))
        if int(bond_type) == int(struc.BondType.ANY):
            proposed = edges.get(key)
            if proposed is not None and int(proposed) != int(struc.BondType.ANY):
                if int(proposed) not in refinable_orders:
                    raise ValueError(
                        "unknown-positive hard edge has an unsupported proposed order"
                    )
                # Hard authority fixes existence only.  A concrete model/CCD
                # proposal remains the next authority for order.
                continue
        # The compiler has already rejected hard-vs-hard order conflicts.
        # Decoder output is only a proposal, so an exact required LL/PL edge
        # overwrites a different proposed order instead of turning an ordinary
        # model mistake into a sample failure.
        edges[key] = int(bond_type)
    atom_array.bonds = struc.BondList(
        len(atom_array),
        np.asarray([[i, j, t] for (i, j), t in sorted(edges.items())], dtype=np.uint32).reshape((-1, 3)),
    )


__all__ = [
    "LigandGraph",
    "TrustedCCDComponent",
    "ligand_graph_from_rdkit",
    "load_trusted_ccd_components",
    "project_trusted_ccd_component",
    "private_molecule_from_authoring_graph",
    "private_molecule_from_trusted_graph",
    "reconcile_required_edges",
    "reconstruct_obabel",
    "reconstruct_obabel_with_molecule",
    "reconstruct_obabel_molecule",
    "replace_component_bonds",
    "trusted_ccd_private_molecule",
]
