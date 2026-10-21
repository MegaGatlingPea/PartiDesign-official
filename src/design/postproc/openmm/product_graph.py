"""Build private product graphs from explicitly selected heavy-atom groups.

The caller owns selection, cross-chain capping, and any valence-changing product
edits.  This module only materializes closed, connected components already present
in the accepted :class:`biotite.structure.AtomArray`.  It never infers missing
edges, bond orders, charges, or protonation.

Protein product records retain the historical whole-peptide convention: amide
nitrogens are neutral, free carboxyl oxygens are ionized, and standard protein
backbone N termini are protonated.  Those are protein-only rules.  RNA and DNA
atoms retain their CCD atom charges and are accepted only when the complete final
graph sanitizes and survives the explicit-hydrogen round trip.

If a product link changes valence, the caller must pass the final product graph,
including its concrete bond order and authoritative post-link atom charges.  The
builder does not remove link hydrogens or guess a new protonation state.  An
ambiguous or unsupported graph fails with ``PostprocessPreparationError``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from rdkit import Chem

from ...utils.ccd import load_packaged_polymer_ccd

if TYPE_CHECKING:
    from ..runner import GaffLigandRecord


_ORDER_TO_RDKIT = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    # Biotite preserves an explicit Kekule subtype for aromatic CIF bonds.
    # Keep the same interpretation as the existing peptide builder; the
    # aromatic consistency check below prevents an aromatic fact from being
    # silently discarded.
    5: Chem.BondType.SINGLE,
    6: Chem.BondType.DOUBLE,
    7: Chem.BondType.TRIPLE,
    9: Chem.BondType.AROMATIC,
}
_SUPPORTED_MODALITIES = frozenset(("protein", "rna", "dna", "ligand"))
_POLYMER_MODALITIES = frozenset(("protein", "rna", "dna"))


def _atom_key(atom_array: Any, row: int) -> tuple[str, int, str, str, str]:
    return (
        str(atom_array.chain_id[row]),
        int(atom_array.res_id[row]),
        str(atom_array.ins_code[row]),
        str(atom_array.res_name[row]),
        str(atom_array.atom_name[row]),
    )


def _component_override(
    overrides: Mapping[Any, Any] | None,
    atoms: Any,
    row: int,
) -> Any | None:
    """Return the most specific private CCD component for one accepted row.

    linked_context_view publishes both a component-ID key and a residue
    key. The latter permits two source residues with the same resolved CCD
    identity to carry different frozen charge facts without mutating the
    packaged component. An atom key is accepted as a narrow extension for
    callers that need one-row lookup.
    """

    if not overrides:
        return None
    key = _atom_key(atoms, row)
    residue_key = (key[0], key[1], key[2], key[3])
    for candidate in (key, residue_key, key[3]):
        try:
            component = overrides.get(candidate)
        except (AttributeError, TypeError):
            return None
        if component is not None:
            return component
    return None


def _component_atom_fact(component: Any | None, atom_name: str) -> Any | None:
    """Resolve a literal CCD atom ID, then one authoritative alias."""

    if component is None:
        return None
    fact = component.atom_by_id(str(atom_name))
    if fact is not None:
        return fact
    resolver = getattr(component, "resolve_atom_id", None)
    if resolver is None:
        return None
    resolved = resolver(str(atom_name))
    return None if resolved is None else component.atom_by_id(resolved)


def _normalise_groups(
    atoms: Any,
    groups: Sequence[Sequence[int]],
    *,
    error_type: type[ValueError],
) -> tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]:
    """Validate explicit closed groups and return normalized rows/edges.

    A selected atom may not have an authoritative edge to an unselected atom or
    to another group.  This is deliberate: callers must finish capping and all
    valence-changing edits before asking this helper to construct a product.
    """

    if isinstance(groups, (str, bytes)):
        raise error_type("product groups must be sequences of row sequences")
    n_atoms = len(atoms)
    normalized: list[tuple[int, ...]] = []
    row_to_group: dict[int, int] = {}
    for group_index, raw_group in enumerate(groups):
        try:
            raw_rows = tuple(raw_group)
        except TypeError as exc:
            raise error_type("product groups must be sequences of row sequences") from exc
        if not raw_rows:
            raise error_type("product groups cannot contain an empty component")
        rows: list[int] = []
        for raw_row in raw_rows:
            if isinstance(raw_row, bool):
                raise error_type("product group rows must be integer indices")
            try:
                row = int(raw_row)
            except (TypeError, ValueError) as exc:
                raise error_type("product group rows must be integer indices") from exc
            if row != raw_row:
                raise error_type("product group rows must be integer indices")
            if row < 0 or row >= n_atoms:
                raise error_type("product group row is outside the accepted structure")
            if row in row_to_group:
                raise error_type("product groups overlap heavy atoms")
            row_to_group[row] = group_index
            rows.append(row)
        if len(rows) != len(set(rows)):
            raise error_type("product group contains a duplicate heavy atom")
        normalized.append(tuple(sorted(rows)))

    raw_edges = () if atoms.bonds is None else atoms.bonds.as_array()
    internal_edges: list[tuple[int, int, int]] = []
    seen_edges: dict[tuple[int, int], int] = {}
    for raw_i, raw_j, raw_order in raw_edges:
        i, j, order = int(raw_i), int(raw_j), int(raw_order)
        if i == j:
            raise error_type("product graph contains a self-bond")
        if i < 0 or i >= n_atoms or j < 0 or j >= n_atoms:
            raise error_type("product graph contains an out-of-range bond endpoint")
        group_i = row_to_group.get(i)
        group_j = row_to_group.get(j)
        if group_i is None and group_j is None:
            # The explicit product scope may be a local subset of a larger
            # accepted structure.  An unrelated context edge is not part of
            # this product and must not make it fail merely because its source
            # graph carries an unresolved order.
            continue
        if order not in _ORDER_TO_RDKIT:
            raise error_type("product graph needs concrete covalent bond orders")
        if group_i is None or group_j is None:
            raise error_type("product group is not closed over the authoritative graph")
        if group_i != group_j:
            raise error_type("product groups are not closed over the authoritative graph")
        pair = tuple(sorted((i, j)))
        previous = seen_edges.get(pair)
        if previous is not None:
            if previous != order:
                raise error_type("product graph contains conflicting bond orders")
            raise error_type("product graph contains duplicate bonds")
        seen_edges[pair] = order
        internal_edges.append((i, j, order))

    # Explicit groups must be connected components, not arbitrary collections
    # of atoms.  A one-atom component is valid without a bond list.
    adjacency: dict[int, set[int]] = {row: set() for row in row_to_group}
    for i, j, _ in internal_edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    for rows in normalized:
        if len(rows) <= 1:
            continue
        visited = {rows[0]}
        queue = [rows[0]]
        while queue:
            current = queue.pop()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(visited) != len(rows):
            raise error_type("product group is not connected by authoritative bonds")

    return tuple(normalized), tuple(internal_edges)


def _private_chemistry(
    candidate: Any,
    atoms: Any,
    selected_rows: set[int],
    *,
    error_type: type[ValueError],
    gaff_record_from_rdkit: Any,
    validate_gaff_record: Any,
) -> tuple[dict[int, int], dict[tuple[int, int], Chem.BondType]]:
    """Collect validated private ligand charges and Kekule bond facts."""

    ligand_rows = {
        row
        for row in selected_rows
        if str(atoms.mol_type[row]) == "ligand"
    }
    private_errors = {
        int(key): str(value)
        for key, value in getattr(candidate, "private_ligand_errors", {}).items()
    }
    selected_uids = {int(atoms.molecule_uid[row]) for row in ligand_rows}
    for molecule_uid in sorted(selected_uids):
        if molecule_uid in private_errors:
            raise error_type(
                f"selected product ligand {molecule_uid} has invalid private chemistry: "
                f"{private_errors[molecule_uid]}"
            )

    private_charges: dict[int, int] = {}
    private_kekule: dict[tuple[int, int], Chem.BondType] = {}
    private_molecules = getattr(candidate, "private_ligand_molecules", {})
    for raw_uid, molecule in private_molecules.items():
        molecule_uid = int(raw_uid)
        rows = np.flatnonzero(
            (np.asarray(atoms.molecule_uid, dtype=np.int64) == molecule_uid)
            & (np.asarray(atoms.mol_type).astype(str) == "ligand")
        ).tolist()
        if not rows or not set(rows).intersection(ligand_rows):
            continue
        if not set(rows).issubset(selected_rows):
            raise error_type(
                f"private ligand molecule {molecule_uid} is not a closed product component"
            )
        heavy_keys = tuple(_atom_key(atoms, row) for row in rows)
        try:
            record = gaff_record_from_rdkit(
                molecule,
                record_id=f"bridge-{molecule_uid}",
                molecule_uid=molecule_uid,
                component_uid=int(atoms.component_uid[rows[0]]),
                heavy_keys=heavy_keys,
            )
            validate_gaff_record(record, atoms, rows)
        except error_type:
            raise
        except Exception as exc:
            raise error_type(
                f"private chemistry for ligand molecule {molecule_uid} is invalid"
            ) from exc
        private_charges.update(zip(rows, record.formal_charges, strict=True))
        reference = Chem.Mol(molecule)
        try:
            Chem.Kekulize(reference, clearAromaticFlags=True)
        except Exception as exc:
            raise error_type(
                f"private chemistry for ligand molecule {molecule_uid} cannot be Kekulized"
            ) from exc
        heavy_rows = dict(
            zip(
                (
                    atom.GetIdx()
                    for atom in reference.GetAtoms()
                    if atom.GetAtomicNum() > 1
                ),
                rows,
                strict=True,
            )
        )
        for bond in reference.GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if left in heavy_rows and right in heavy_rows:
                private_kekule[tuple(sorted((heavy_rows[left], heavy_rows[right])))] = (
                    bond.GetBondType()
                )
    return private_charges, private_kekule


def _apply_protein_product_charges(molecule: Chem.Mol, atoms: Any, rows: Sequence[int]) -> None:
    """Apply the historical product-state convention to protein rows only."""

    for local_index, row in enumerate(rows):
        if str(atoms.mol_type[row]) != "protein":
            continue
        atom = molecule.GetAtomWithIdx(local_index)
        neighbors = list(atom.GetNeighbors())

        def carbonyl(carbon: Chem.Atom) -> bool:
            return carbon.GetSymbol() == "C" and any(
                bond.GetBondTypeAsDouble() == 2
                and bond.GetOtherAtom(carbon).GetSymbol() == "O"
                for bond in carbon.GetBonds()
            )

        if atom.GetSymbol() == "N":
            if any(carbonyl(neighbor) for neighbor in neighbors):
                atom.SetFormalCharge(0)
            elif str(atoms.atom_name[row]) == "N" and (
                len(neighbors) == 1
                or (str(atoms.res_name[row]) == "PRO" and len(neighbors) == 2)
            ):
                atom.SetFormalCharge(1)
        if (
            atom.GetSymbol() == "O"
            and len(neighbors) == 1
            and carbonyl(neighbors[0])
            and molecule.GetBondBetweenAtoms(
                local_index, neighbors[0].GetIdx()
            ).GetBondTypeAsDouble()
            == 1
        ):
            atom.SetFormalCharge(-1)


def build_product_records(
    candidate: Any,
    groups: Sequence[Sequence[int]],
    *,
    formal_charge_overrides: Mapping[tuple[str, int, str, str, str], int] | None = None,
    component_overrides: Mapping[Any, Any] | None = None,
) -> tuple[GaffLigandRecord, ...]:
    """Build one :class:`GaffLigandRecord` per explicit closed product group.

    ``groups`` contains accepted-structure row indices.  Every group must be a
    non-empty connected component, groups must not overlap, and no authoritative
    bond may leave a group.  The returned record IDs and ordinal UIDs intentionally
    match the historical ``peptide_records`` convention for parity.
    Optional charge overrides must refer to selected atoms and describe the
    caller's verified final product state; this helper never derives them.
    component_overrides contains owner-local CCDComponent values keyed by
    resolved component ID, residue identity, or atom key. It is consulted
    before the packaged CCD and never mutates that global mapping.
    """

    from ..runner import (
        PostprocessPreparationError,
        _validate_gaff_record_against_structure,
        gaff_record_from_rdkit,
    )

    atoms = candidate.atoms
    normalized_groups, edges = _normalise_groups(
        atoms,
        groups,
        error_type=PostprocessPreparationError,
    )
    if not normalized_groups:
        return ()
    selected_rows = {row for group in normalized_groups for row in group}
    overrides = dict(formal_charge_overrides or {})
    selected_keys = {_atom_key(atoms, row) for row in selected_rows}
    if set(overrides).difference(selected_keys):
        raise PostprocessPreparationError("product charge override references an unselected atom")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
           for value in overrides.values()):
        raise PostprocessPreparationError("product formal charges must be integers")
    private_charges, private_kekule = _private_chemistry(
        candidate,
        atoms,
        selected_rows,
        error_type=PostprocessPreparationError,
        gaff_record_from_rdkit=gaff_record_from_rdkit,
        validate_gaff_record=_validate_gaff_record_against_structure,
    )
    ccd = load_packaged_polymer_ccd()
    records: list[Any] = []
    for ordinal, rows in enumerate(normalized_groups):
        local = {row: index for index, row in enumerate(rows)}
        modalities = {str(atoms.mol_type[row]) for row in rows}
        unsupported = modalities.difference(_SUPPORTED_MODALITIES)
        if unsupported:
            raise PostprocessPreparationError(
                f"product graph contains unsupported modalities: {sorted(unsupported)}"
            )
        rw = Chem.RWMol()
        for row in rows:
            component = _component_override(component_overrides, atoms, row)
            if component is None:
                component = ccd.get(str(atoms.res_name[row]))
            fact = _component_atom_fact(component, str(atoms.atom_name[row]))
            modality = str(atoms.mol_type[row])
            if modality in _POLYMER_MODALITIES and (
                component is None or component.polymer_modality != modality
            ):
                raise PostprocessPreparationError(
                    f"no authoritative CCD component for {_atom_key(atoms, row)}"
                )
            charge = overrides.get(_atom_key(atoms, row), private_charges.get(row))
            if charge is None:
                if (
                    fact is None
                    or fact.formal_charge is None
                    or str(fact.element).upper() != str(atoms.element[row]).upper()
                ):
                    raise PostprocessPreparationError(
                        f"no authoritative atom charge for {_atom_key(atoms, row)}"
                    )
                charge = int(fact.formal_charge)
            atom = Chem.Atom(str(atoms.element[row]).capitalize())
            atom.SetFormalCharge(int(charge))
            rw.AddAtom(atom)

        aromatic_pairs: list[tuple[int, int]] = []
        for raw_i, raw_j, raw_order in edges:
            if raw_i not in local or raw_j not in local:
                continue
            bond_type = _ORDER_TO_RDKIT[raw_order]
            if raw_order in {5, 6, 7, 9}:
                aromatic_pairs.append((local[raw_i], local[raw_j]))
            if raw_order == 9:
                # An aromatic-only accepted graph may omit Kekule orders.  A
                # validated private molecule or exact same-residue CCD bond is
                # the only allowed source of a concrete replacement.
                bond_type = private_kekule.get(
                    tuple(sorted((raw_i, raw_j))), bond_type
                )
                if (
                    bond_type == Chem.BondType.AROMATIC
                    and _atom_key(atoms, raw_i)[:4] == _atom_key(atoms, raw_j)[:4]
                ):
                    component = _component_override(component_overrides, atoms, raw_i)
                    if component is None:
                        component = ccd.get(str(atoms.res_name[raw_i]))
                    names = {
                        str(atoms.atom_name[raw_i]),
                        str(atoms.atom_name[raw_j]),
                    }
                    for fact in (() if component is None else component.bonds):
                        if {fact.atom_id_1, fact.atom_id_2} == names:
                            bond_type = {
                                "single": Chem.BondType.SINGLE,
                                "double": Chem.BondType.DOUBLE,
                                "triple": Chem.BondType.TRIPLE,
                            }.get(fact.order, bond_type)
                            break
            rw.AddBond(local[raw_i], local[raw_j], bond_type)
            if bond_type == Chem.BondType.AROMATIC:
                rw.GetAtomWithIdx(local[raw_i]).SetIsAromatic(True)
                rw.GetAtomWithIdx(local[raw_j]).SetIsAromatic(True)

        molecule = rw.GetMol()
        _apply_protein_product_charges(molecule, atoms, rows)
        # The caller owns verified final product-state edits. Explicit atom
        # charges take precedence over the historical protein convention.
        for local_index, row in enumerate(rows):
            key = _atom_key(atoms, row)
            if key in overrides:
                molecule.GetAtomWithIdx(local_index).SetFormalCharge(int(overrides[key]))
        try:
            Chem.SanitizeMol(molecule)
            if any(
                not molecule.GetBondBetweenAtoms(i, j).GetIsAromatic()
                for i, j in aromatic_pairs
            ):
                raise PostprocessPreparationError(
                    "product graph contradicts authoritative aromatic bond facts"
                )
            conformer = Chem.Conformer(len(rows))
            for local_index, row in enumerate(rows):
                conformer.SetAtomPosition(
                    local_index, atoms.coord[row].astype(float).tolist()
                )
            molecule.AddConformer(conformer)
            Chem.AssignStereochemistryFrom3D(molecule)
            molecule = Chem.AddHs(molecule, addCoords=True)
            Chem.Kekulize(molecule, clearAromaticFlags=True)
            records.append(
                gaff_record_from_rdkit(
                    molecule,
                    record_id=f"peptide-{ordinal}",
                    molecule_uid=ordinal,
                    component_uid=ordinal,
                    heavy_keys=tuple(_atom_key(atoms, row) for row in rows),
                )
            )
        except PostprocessPreparationError:
            raise
        except (ValueError, RuntimeError) as exc:
            raise PostprocessPreparationError(
                "product chemistry does not sanitize"
            ) from exc
    return tuple(records)


__all__ = ["build_product_records"]
