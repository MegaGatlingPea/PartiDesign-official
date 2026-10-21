
"""Bounded identity restoration for fixed ligand-classified polymer residues.

A source ligand is promoted to polymer chemistry only when its complete observed
heavy graph matches one authoritative CCD component and an explicit same-chain
peptide or nucleic polymer link is present. Coordinates never participate in the
identity decision. The returned private view is an owner-local copy; the source
AtomArray and packaged CCD are untouched.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from biotite import structure as struc
from rdkit import Chem

from ...utils.ccd import (
    CCDAtom,
    CCDBond,
    CCDComponent,
    CCDProvenance,
    load_packaged_polymer_ccd,
)
from ...utils.chemistry import TrustedCCDComponent

try:
    from .. import context as _context
except Exception:
    _context = None


AtomKey = tuple[str, int, str, str, str]
ResidueKey = tuple[str, int, str]

_ANY_ORDER = int(struc.BondType.ANY)
_SINGLE_ORDER = int(struc.BondType.SINGLE)
_DOUBLE_ORDER = int(struc.BondType.DOUBLE)
_TRIPLE_ORDER = int(struc.BondType.TRIPLE)
_AROMATIC_ORDER = int(struc.BondType.AROMATIC)
_AROMATIC_SINGLE_ORDER = int(struc.BondType.AROMATIC_SINGLE)
_AROMATIC_DOUBLE_ORDER = int(struc.BondType.AROMATIC_DOUBLE)
_AROMATIC_TRIPLE_ORDER = int(struc.BondType.AROMATIC_TRIPLE)
_POLYMER_MODALITIES = frozenset(("protein", "rna", "dna"))
_POLYMER_FAMILY_ALIASES = frozenset(("GTP", "GDP", "G", "GMP", "CPG"))
_MONATOMIC_METAL_ELEMENTS = frozenset(
    (
        "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA",
        "ZN", "CD", "HG", "AL", "FE", "CO", "NI", "CU", "MN",
    )
)
_SUPPORTED_INTERNAL_ORDERS = frozenset(
    (
        _SINGLE_ORDER,
        _DOUBLE_ORDER,
        _TRIPLE_ORDER,
        _AROMATIC_ORDER,
        _AROMATIC_SINGLE_ORDER,
        _AROMATIC_DOUBLE_ORDER,
        _AROMATIC_TRIPLE_ORDER,
    )
)


class LinkedContextError(ValueError):
    """Raised when a linked fixed residue cannot be resolved without guessing."""


@dataclass(frozen=True, slots=True)
class LinkedContextSpec(Mapping[str, Any]):
    """Coordinate-free, serializable proof for one linked fixed residue."""

    residue_key: ResidueKey
    source_component_id: str
    component_id: str
    modality: str
    linker_kind: str
    heavy_keys: tuple[AtomKey, ...]
    linker_keys: tuple[tuple[AtomKey, AtomKey], ...]
    atom_names: tuple[str, ...]
    atom_elements: tuple[str, ...]
    component_atoms: tuple[
        tuple[str, tuple[str, ...], str | None, int | None, bool | None], ...
    ]
    component_bonds: tuple[tuple[str, str, str], ...]
    raw_chem_type: str | None
    parent_ids: tuple[str, ...]
    component_formal_charge: int | None
    formal_charge_by_atom_name: tuple[tuple[str, int | None], ...]
    trusted_molecule_binary: bytes | None = None
    residue_uid: int | None = None

    def __post_init__(self) -> None:
        if len(self.heavy_keys) != len(set(self.heavy_keys)):
            raise ValueError("linked-context heavy atom keys must be unique")
        if len(self.atom_names) != len(self.heavy_keys):
            raise ValueError("linked-context atom inventory is incomplete")
        if len(self.atom_elements) != len(self.heavy_keys):
            raise ValueError("linked-context element inventory is incomplete")
        if self.modality not in _POLYMER_MODALITIES:
            raise ValueError("linked-context modality must be a polymer modality")
        if self.linker_kind not in {"peptide", "nucleic"}:
            raise ValueError("linked-context linker kind is unsupported")
        if self.component_id != self.component_id.strip().upper():
            raise ValueError("linked-context component ID must be uppercase")
        if any(
            isinstance(charge, bool)
            or (charge is not None and not isinstance(charge, int))
            for _, charge in self.formal_charge_by_atom_name
        ):
            raise ValueError("linked-context formal charges must be integers")

    @property
    def source_name(self) -> str:
        return self.source_component_id

    @property
    def resolved_component_id(self) -> str:
        return self.component_id

    @property
    def resolved_modality(self) -> str:
        return self.modality

    @property
    def link_type(self) -> str:
        return self.linker_kind

    @property
    def link_keys(self) -> tuple[tuple[AtomKey, AtomKey], ...]:
        return self.linker_keys

    @property
    def linker_edges(self) -> tuple[tuple[AtomKey, AtomKey], ...]:
        return self.linker_keys

    @property
    def atom_keys(self) -> tuple[AtomKey, ...]:
        return self.heavy_keys

    @property
    def original_residue_key(self) -> tuple[str, int, str, str]:
        return (*self.residue_key, self.source_component_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "residue_key": self.residue_key,
            "source_component_id": self.source_component_id,
            "component_id": self.component_id,
            "modality": self.modality,
            "linker_kind": self.linker_kind,
            "heavy_keys": self.heavy_keys,
            "linker_keys": self.linker_keys,
            "atom_names": self.atom_names,
            "atom_elements": self.atom_elements,
            "component_atoms": self.component_atoms,
            "component_bonds": self.component_bonds,
            "raw_chem_type": self.raw_chem_type,
            "parent_ids": self.parent_ids,
            "component_formal_charge": self.component_formal_charge,
            "formal_charge_by_atom_name": self.formal_charge_by_atom_name,
            "trusted_molecule_binary": self.trusted_molecule_binary,
            "residue_uid": self.residue_uid,
        }

    to_dict = as_dict

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())


def _atom_key(atoms: Any, row: int) -> AtomKey:
    return (
        str(atoms.chain_id[row]),
        int(atoms.res_id[row]),
        str(atoms.ins_code[row]),
        str(atoms.res_name[row]),
        str(atoms.atom_name[row]),
    )


def _residue_key(atoms: Any, row: int) -> ResidueKey:
    return (str(atoms.chain_id[row]), int(atoms.res_id[row]), str(atoms.ins_code[row]))


def _source_path(candidate: Any) -> Any:
    request = getattr(candidate, "base_request", None)
    task = getattr(request, "task", None)
    return getattr(task, "input_path", None)


def _residue_groups(atoms: Any) -> dict[ResidueKey, tuple[int, ...]]:
    grouped: dict[ResidueKey, list[int]] = {}
    for row in range(len(atoms)):
        grouped.setdefault(_residue_key(atoms, row), []).append(row)
    return {key: tuple(rows) for key, rows in grouped.items()}


def _is_heavy(element: object) -> bool:
    return str(element).strip().upper() not in {"H", "D", "T"}


def _rdkit_bond_order(bond: Any) -> str:
    kind = bond.GetBondType()
    if kind == Chem.BondType.SINGLE:
        return "single"
    if kind == Chem.BondType.DOUBLE:
        return "double"
    if kind == Chem.BondType.TRIPLE:
        return "triple"
    if kind == Chem.BondType.AROMATIC:
        return "aromatic"
    raise LinkedContextError("trusted CCD contains an unsupported bond order")


def _trusted_component(
    trusted: TrustedCCDComponent,
    *,
    raw_chem_type: str | None = None,
    modality: str | None = None,
    parent_ids: tuple[str, ...] | None = None,
) -> CCDComponent:
    """Convert a cached private molecule into coordinate-free CCD facts."""

    raw = Chem.Mol(trusted.rdkit_binary)
    raw.RemoveAllConformers()
    source = Chem.RemoveHs(raw, sanitize=False)
    heavy_indices = [
        atom.GetIdx() for atom in source.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    if not heavy_indices:
        raise LinkedContextError(
            f"trusted CCD component {trusted.component_id!r} has no heavy atoms"
        )
    index_set = set(heavy_indices)
    atoms: list[CCDAtom] = []
    for index in heavy_indices:
        atom = source.GetAtomWithIdx(index)
        if not atom.HasProp("name"):
            raise LinkedContextError(
                f"trusted CCD component {trusted.component_id!r} lacks atom names"
            )
        name = atom.GetProp("name").strip()
        aliases: list[str] = [name]
        if atom.HasProp("alt_name"):
            alias = atom.GetProp("alt_name").strip()
            if alias and alias not in aliases:
                aliases.append(alias)
        leaving: bool | None = None
        if atom.HasProp("leaving_atom"):
            try:
                leaving = bool(atom.GetBoolProp("leaving_atom"))
            except Exception:
                leaving = atom.GetProp("leaving_atom").strip().lower() in {
                    "1",
                    "true",
                    "y",
                    "yes",
                }
        atoms.append(
            CCDAtom(
                atom_id=name,
                aliases=tuple(aliases),
                element=atom.GetSymbol().upper(),
                formal_charge=int(atom.GetFormalCharge()),
                leaving=leaving,
                ideal_coord=None,
            )
        )
    local_index = {index: offset for offset, index in enumerate(heavy_indices)}
    bonds: list[CCDBond] = []
    for bond in source.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left not in index_set or right not in index_set:
            continue
        bonds.append(
            CCDBond(
                atoms[local_index[left]].atom_id,
                atoms[local_index[right]].atom_id,
                _rdkit_bond_order(bond),
            )
        )
    identifier = trusted.component_id.strip().upper()
    if raw_chem_type is None:
        raw_chem_type = "D-PEPTIDE LINKING" if identifier == "DVA" else "RNA LINKING"
    if modality is None:
        modality = "protein" if identifier == "DVA" else "rna"
    if parent_ids is None:
        parent_ids = () if identifier == "DVA" else ("G",)
    if modality not in _POLYMER_MODALITIES:
        raise LinkedContextError("trusted linked component is not a polymer")
    provenance = (
        CCDProvenance(
            kind="embedded",
            location="linked_context_private",
            source_name="ccd.pkl",
            source_size=len(trusted.rdkit_binary),
        ),
    )
    return CCDComponent(
        component_id=identifier,
        raw_chem_type=raw_chem_type,
        polymer_modality=modality,
        parent_ids=tuple(parent_ids),
        component_formal_charge=sum(
            atom.formal_charge or 0 for atom in atoms
        ),
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        provenance=provenance,
    )


def _component_atom_records(
    component: CCDComponent,
) -> tuple[tuple[str, tuple[str, ...], str | None, int | None, bool | None], ...]:
    return tuple(
        (
            atom.atom_id,
            tuple(atom.aliases),
            None if atom.element is None else str(atom.element),
            None if atom.formal_charge is None else int(atom.formal_charge),
            None if atom.leaving is None else bool(atom.leaving),
        )
        for atom in component.atoms
        if _is_heavy(atom.element)
    )


def _component_bond_records(
    component: CCDComponent,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (bond.atom_id_1, bond.atom_id_2, str(bond.order))
        for bond in component.bonds
    )


def _component_ids_for_source(source_name: str, packaged: Mapping[str, CCDComponent]) -> tuple[str, ...]:
    name = str(source_name).strip().upper()
    if not name or name.startswith("LIG"):
        return ()
    if name in _POLYMER_FAMILY_ALIASES:
        # Exact graph matching below determines whether this is GDP/GMP/GTP;
        # no source-file or PDB-specific lookup table is used.
        return tuple(packaged)
    if name in packaged:
        return (name,)
    private = getattr(_context, "_CONTEXT_COMPONENTS", {}) if _context is not None else {}
    if name in private:
        return (name,)
    return ()


def _cached_private_components() -> dict[str, CCDComponent]:
    private = getattr(_context, "_CONTEXT_COMPONENTS", {}) if _context is not None else {}
    result: dict[str, CCDComponent] = {}
    for name, trusted in tuple(private.items()):
        if not isinstance(trusted, TrustedCCDComponent):
            continue
        try:
            result[str(name).upper()] = _trusted_component(trusted)
        except Exception:
            continue
    return result


def _observed_internal_edges(
    atoms: Any,
    rows: Sequence[int],
) -> dict[tuple[str, str], int] | None:
    row_set = set(int(row) for row in rows)
    names = {int(row): str(atoms.atom_name[int(row)]).strip() for row in rows}
    result: dict[tuple[str, str], int] = {}
    if atoms.bonds is None:
        return {} if len(rows) <= 1 else None
    for raw_i, raw_j, raw_order in atoms.bonds.as_array():
        left, right, order = int(raw_i), int(raw_j), int(raw_order)
        if left not in row_set or right not in row_set:
            continue
        if left == right:
            return None
        pair = tuple(sorted((names[left], names[right])))
        if pair in result and result[pair] != order:
            return None
        if pair in result:
            return None
        result[pair] = order
    return result


def _bond_order_matches(observed: int, expected: str) -> bool:
    if observed == _ANY_ORDER:
        return False
    if expected == "single":
        return observed in {_SINGLE_ORDER, _AROMATIC_SINGLE_ORDER}
    if expected == "double":
        return observed in {_DOUBLE_ORDER, _AROMATIC_DOUBLE_ORDER}
    if expected == "triple":
        return observed in {_TRIPLE_ORDER, _AROMATIC_TRIPLE_ORDER}
    if expected == "aromatic":
        return observed in {
            _AROMATIC_ORDER, _AROMATIC_SINGLE_ORDER,
            _AROMATIC_DOUBLE_ORDER, _AROMATIC_TRIPLE_ORDER,
        }
    return False


def _match_component(
    component: CCDComponent,
    atoms: Any,
    rows: Sequence[int],
) -> dict[int, str] | None:
    """Return source-row to canonical atom mapping on an exact graph match."""

    rows = tuple(int(row) for row in rows)
    observed = _observed_internal_edges(atoms, rows)
    if observed is None:
        return None
    by_canonical = {
        atom.atom_id: atom for atom in component.atoms if _is_heavy(atom.element)
    }
    by_row: dict[int, str] = {}
    used: set[str] = set()
    for row in rows:
        source_name = str(atoms.atom_name[row]).strip()
        if not source_name:
            return None
        try:
            canonical = component.resolve_atom_id(source_name)
        except Exception:
            return None
        if canonical is None or canonical in used:
            return None
        fact = by_canonical.get(canonical)
        if fact is None:
            return None
        observed_element = str(atoms.element[row]).strip().upper()
        if fact.element is None or str(fact.element).upper() != observed_element:
            return None
        by_row[row] = canonical
        used.add(canonical)
    omitted = set(by_canonical).difference(used)
    if any(not bool(by_canonical[name].leaving) for name in omitted):
        return None
    expected: dict[tuple[str, str], str] = {}
    for bond in component.bonds:
        if bond.atom_id_1 not in used or bond.atom_id_2 not in used:
            continue
        pair = tuple(sorted((bond.atom_id_1, bond.atom_id_2)))
        if pair in expected:
            return None
        expected[pair] = str(bond.order)
    canonical_by_name = {str(atoms.atom_name[row]).strip(): name
                         for row, name in by_row.items()}
    translated = {
        tuple(sorted((canonical_by_name[left], canonical_by_name[right]))): order
        for (left, right), order in observed.items()
    }
    if set(translated) != set(expected):
        return None
    for pair, expected_order in expected.items():
        observed_order = translated.get(pair)
        if observed_order is None or not _bond_order_matches(observed_order, expected_order):
            return None
    return by_row


def _g_family_component(component: CCDComponent) -> bool:
    raw = " ".join(str(component.raw_chem_type or "").upper().split())
    return (
        raw == "RNA LINKING"
        and tuple(component.parent_ids or ()) in {(), ("G",)}
    )


def _candidate_components(
    source_name: str,
    packaged: Mapping[str, CCDComponent],
) -> tuple[CCDComponent, ...]:
    ids = _component_ids_for_source(source_name, packaged)
    private = _cached_private_components()
    normalized = str(source_name).strip().upper()
    result: dict[str, CCDComponent] = {}
    if normalized == "GTP":
        # GTP is absent from the compact release. The canonical exact
        # alternatives are GDP and the GMP-like G parent; modified guanosine
        # analogues are deliberately excluded even when their graph is equal.
        for name in ("GDP", "G"):
            component = packaged.get(name)
            if component is not None:
                result[name] = component
        for name in ("GTP", "GDP", "G", "CPG"):
            component = private.get(name)
            if component is not None:
                result[name] = component
        return tuple(result.values())
    for name in ids:
        if name in private:
            result[name] = private[name]
        elif name in packaged:
            result[name] = packaged[name]
    # A directly requested private component can be present without a packaged
    # counterpart, for example full GTP or a private GMP-like component.
    if normalized in private:
        result[normalized] = private[normalized]
    return tuple(result.values())


def _same_polymer_residue(atoms: Any, row: int, groups: Mapping[ResidueKey, tuple[int, ...]]) -> bool:
    key = _residue_key(atoms, row)
    residue_rows = groups.get(key, ())
    if not residue_rows:
        return False
    return (
        all(not bool(atoms.is_design[index]) for index in residue_rows)
        and all(str(atoms.mol_type[index]) in _POLYMER_MODALITIES for index in residue_rows)
    )


def _ignored_monatomic_metal(
    atoms: Any,
    row: int,
    groups: Mapping[ResidueKey, tuple[int, ...]],
) -> bool:
    residue_rows = groups.get(_residue_key(atoms, row), ())
    return (
        len(residue_rows) == 1
        and str(atoms.mol_type[row]) == "ligand"
        and str(atoms.element[row]).strip().upper() in _MONATOMIC_METAL_ELEMENTS
    )


def _direct_links(
    atoms: Any,
    rows: Sequence[int],
    component: CCDComponent,
    groups: Mapping[ResidueKey, tuple[int, ...]],
) -> tuple[str, tuple[tuple[AtomKey, AtomKey], ...]]:
    """Validate every external edge and return one linker modality."""

    row_set = set(int(row) for row in rows)
    residue = _residue_key(atoms, rows[0])
    links: list[tuple[str, AtomKey, AtomKey]] = []
    unsupported: list[tuple[int, int, int]] = []
    if atoms.bonds is None:
        return "", ()
    for raw_i, raw_j, raw_order in atoms.bonds.as_array():
        left, right, order = int(raw_i), int(raw_j), int(raw_order)
        inside_left, inside_right = left in row_set, right in row_set
        if inside_left == inside_right:
            continue
        ligand_row, polymer_row = (left, right) if inside_left else (right, left)
        if _ignored_monatomic_metal(atoms, polymer_row, groups):
            # Metal coordination is noncovalent context and must not turn a
            # fixed polymer-linked residue into an ambiguous covalent product.
            continue
        if not _same_polymer_residue(atoms, polymer_row, groups):
            unsupported.append((left, right, order))
            continue
        if str(atoms.chain_id[polymer_row]) != str(atoms.chain_id[rows[0]]):
            unsupported.append((left, right, order))
            continue
        if _residue_key(atoms, polymer_row) == residue:
            unsupported.append((left, right, order))
            continue
        ligand_name = str(atoms.atom_name[ligand_row]).strip()
        polymer_name = str(atoms.atom_name[polymer_row]).strip()
        if order not in {_ANY_ORDER, _SINGLE_ORDER}:
            unsupported.append((left, right, order))
            continue
        modality = str(component.polymer_modality)
        if modality == "protein" and (
            (ligand_name == "N" and polymer_name == "C")
            or (ligand_name == "C" and polymer_name == "N")
        ):
            kind = "peptide"
        elif modality in {"rna", "dna"} and (
            (ligand_name == "O3'" and polymer_name == "P")
            or (ligand_name == "P" and polymer_name == "O3'")
        ):
            # BondList is undirected; source readers may expose either endpoint
            # first. Both spellings are the same authoritative O3'-P linker.
            kind = "nucleic"
        else:
            unsupported.append((left, right, order))
            continue
        links.append(
            (
                kind,
                _atom_key(atoms, ligand_row),
                _atom_key(atoms, polymer_row),
            )
        )
    if unsupported:
        raise LinkedContextError(
            "linked fixed residue has an unsupported external polymer or ligand edge"
        )
    if not links:
        return "", ()
    kinds = {kind for kind, _, _ in links}
    if len(kinds) != 1:
        raise LinkedContextError("linked fixed residue has ambiguous linker modality")
    by_ligand: set[AtomKey] = set()
    for _, ligand_key, _ in links:
        if ligand_key in by_ligand:
            raise LinkedContextError(
                "linked fixed residue has multiple polymer links on one atom"
            )
        by_ligand.add(ligand_key)
    return next(iter(kinds)), tuple((ligand, polymer) for _, ligand, polymer in links)


def _spec_for_residue(
    candidate: Any,
    atoms: Any,
    rows: Sequence[int],
    *,
    packaged: Mapping[str, CCDComponent],
    groups: Mapping[ResidueKey, tuple[int, ...]],
) -> LinkedContextSpec | None:
    if not rows or any(str(atoms.mol_type[row]) != "ligand" for row in rows):
        return None
    if any(bool(atoms.is_design[row]) for row in rows):
        return None
    source_path = _source_path(candidate)
    if _context is None:
        return None
    try:
        source_name = str(
            _context.context_component_name(atoms, rows, source_path=source_path)
        ).strip().upper()
    except Exception:
        return None
    components = _candidate_components(source_name, packaged)
    matches: list[tuple[CCDComponent, dict[int, str]]] = []
    for component in components:
        match = _match_component(component, atoms, rows)
        if match is not None:
            matches.append((component, match))
    if not matches:
        return None
    # Duplicate private/package representations of one component are equivalent;
    # different component IDs are an identity ambiguity and fail closed.
    by_id: dict[str, tuple[CCDComponent, dict[int, str]]] = {}
    for component, match in matches:
        by_id.setdefault(component.component_id, (component, match))
    if len(by_id) != 1:
        raise LinkedContextError(
            "linked fixed residue matches multiple authoritative CCD identities: "
            + ", ".join(sorted(by_id))
        )
    component, match = next(iter(by_id.values()))
    kind, links = _direct_links(atoms, rows, component, groups)
    if not kind:
        return None
    if component.polymer_modality not in _POLYMER_MODALITIES:
        return None
    source_facts: dict[str, int] = {}
    if source_path is not None:
        try:
            facts = _context.source_formal_charge_facts(
                atoms,
                source_path,
                rows=rows,
            )
        except Exception as exc:
            raise LinkedContextError("source formal charge facts are ambiguous") from exc
        canonical_by_name = {str(atoms.atom_name[row]).strip(): name
                             for row, name in match.items()}
        source_facts = {
            canonical_by_name.get(str(key[4]).strip(), str(key[4]).strip()): int(value)
            for key, value in facts
            if _residue_key(atoms, rows[0]) == tuple(key[:3])
        }
    charge_facts: list[tuple[str, int | None]] = []
    for atom in component.atoms:
        charge = source_facts.get(atom.atom_id, atom.formal_charge)
        charge_facts.append((atom.atom_id, charge))
    trusted_binary = None
    private = getattr(_context, "_CONTEXT_COMPONENTS", {})
    trusted = private.get(component.component_id)
    if isinstance(trusted, TrustedCCDComponent):
        raw = Chem.Mol(trusted.rdkit_binary)
        raw.RemoveAllConformers()
        trusted_binary = bytes(raw.ToBinary(Chem.PropertyPickleOptions.AllProps))
    residue_uid = None
    if "residue_uid" in atoms.get_annotation_categories():
        residue_uid = int(atoms.residue_uid[rows[0]])
    return LinkedContextSpec(
        residue_key=_residue_key(atoms, rows[0]),
        source_component_id=source_name,
        component_id=component.component_id,
        modality=str(component.polymer_modality),
        linker_kind=kind,
        heavy_keys=tuple(_atom_key(atoms, row) for row in rows),
        linker_keys=links,
        atom_names=tuple(str(atoms.atom_name[row]).strip() for row in rows),
        atom_elements=tuple(str(atoms.element[row]).strip().upper() for row in rows),
        component_atoms=_component_atom_records(component),
        component_bonds=_component_bond_records(component),
        raw_chem_type=component.raw_chem_type,
        parent_ids=tuple(component.parent_ids or ()),
        component_formal_charge=component.component_formal_charge,
        formal_charge_by_atom_name=tuple(charge_facts),
        trusted_molecule_binary=trusted_binary,
        residue_uid=residue_uid,
    )


def linked_context_specs(candidate: Any) -> tuple[LinkedContextSpec, ...]:
    """Return authoritative identities for fixed ligand-classified linked residues."""

    atoms = candidate.atoms
    if atoms.bonds is None:
        return ()
    groups = _residue_groups(atoms)
    packaged = load_packaged_polymer_ccd()
    specs: list[LinkedContextSpec] = []
    for residue, rows in sorted(groups.items()):
        spec = _spec_for_residue(
            candidate,
            atoms,
            rows,
            packaged=packaged,
            groups=groups,
        )
        if spec is not None:
            specs.append(spec)
    return tuple(specs)


def _component_from_spec(spec: LinkedContextSpec) -> CCDComponent:
    frozen_charges = dict(spec.formal_charge_by_atom_name)
    atoms = tuple(
        CCDAtom(
            atom_id=name,
            aliases=tuple(aliases),
            element=element,
            formal_charge=frozen_charges.get(name, charge),
            leaving=leaving,
            ideal_coord=None,
        )
        for name, aliases, element, charge, leaving in spec.component_atoms
    )
    bonds = tuple(
        CCDBond(atom_id_1=left, atom_id_2=right, order=order)
        for left, right, order in spec.component_bonds
    )
    return CCDComponent(
        component_id=spec.component_id,
        raw_chem_type=spec.raw_chem_type,
        polymer_modality=spec.modality,
        parent_ids=spec.parent_ids,
        component_formal_charge=spec.component_formal_charge,
        atoms=atoms,
        bonds=bonds,
        provenance=(
            CCDProvenance(
                kind="embedded",
                location="linked_context_spec",
                source_name="linked_context",
                source_size=len(spec.trusted_molecule_binary or b""),
            ),
        ),
    )


def linked_context_view(
    atoms: Any,
    specs: Sequence[LinkedContextSpec],
) -> tuple[Any, dict[AtomKey, AtomKey], dict[Any, CCDComponent]]:
    """Apply only proven fixed residue identities to an owner-local AtomArray copy."""

    if specs is None:
        specs = ()
    private = atoms.copy()
    original_to_private: dict[AtomKey, AtomKey] = {}
    row_by_key: dict[AtomKey, int] = {}
    for row in range(len(atoms)):
        key = _atom_key(atoms, row)
        if key in row_by_key:
            raise LinkedContextError("accepted AtomArray has duplicate atom keys")
        row_by_key[key] = row
        original_to_private[key] = key
    overrides: dict[Any, CCDComponent] = {}
    edge_array = (
        np.asarray(atoms.bonds.as_array(), dtype=np.int64).copy()
        if atoms.bonds is not None
        else np.empty((0, 3), dtype=np.int64)
    )
    edge_lookup: dict[tuple[int, int], list[int]] = {}
    for index, (raw_i, raw_j, _order) in enumerate(edge_array):
        pair = tuple(sorted((int(raw_i), int(raw_j))))
        edge_lookup.setdefault(pair, []).append(index)
    seen_rows: set[int] = set()
    for raw_spec in specs:
        if not isinstance(raw_spec, LinkedContextSpec):
            try:
                spec = LinkedContextSpec(**dict(raw_spec))
            except Exception as exc:
                raise LinkedContextError("linked context spec is not valid") from exc
        else:
            spec = raw_spec
        rows: list[int] = []
        for key in spec.heavy_keys:
            row = row_by_key.get(key)
            if row is None:
                raise LinkedContextError(
                    f"linked context atom key is absent from the accepted view: {key!r}"
                )
            if row in seen_rows:
                raise LinkedContextError("linked context residues overlap")
            seen_rows.add(row)
            rows.append(row)
        if tuple(_residue_key(atoms, row) for row in rows) != (spec.residue_key,) * len(rows):
            raise LinkedContextError("linked context residue keys are inconsistent")
        public_names = {
            str(key[3]).strip().upper()
            for key in spec.heavy_keys
        }
        if len(public_names) != 1:
            raise LinkedContextError("linked context public source identity is inconsistent")
        old_name = str(atoms.res_name[rows[0]])
        if old_name.strip().upper() != next(iter(public_names)):
            raise LinkedContextError("linked context source identity changed before view")
        component = _component_from_spec(spec)
        if _match_component(component, atoms, rows) is None:
            raise LinkedContextError("linked context graph changed after chemistry was frozen")
        private.res_name[rows] = spec.component_id
        private.mol_type[rows] = spec.modality
        for ligand_key, polymer_key in spec.linker_keys:
            left = row_by_key.get(ligand_key)
            right = row_by_key.get(polymer_key)
            if left is None or right is None:
                raise LinkedContextError("linked context linker endpoint is absent")
            indices = edge_lookup.get(tuple(sorted((left, right))), ())
            if len(indices) != 1:
                raise LinkedContextError(
                    "linked context linker endpoint is not one authoritative edge"
                )
            edge_index = indices[0]
            order = int(edge_array[edge_index, 2])
            if order == _ANY_ORDER:
                edge_array[edge_index, 2] = _SINGLE_ORDER
            elif order != _SINGLE_ORDER:
                raise LinkedContextError(
                    "linked context linker order is neither ANY nor single"
                )
        residue_component_key = (*spec.residue_key, spec.component_id)
        overrides[residue_component_key] = component
        previous = overrides.get(spec.component_id)
        if previous is not None and previous != component:
            # Keep residue-specific entries authoritative when source charge facts
            # differ; a shared ID lookup is only a convenience.
            overrides.pop(spec.component_id, None)
        else:
            overrides[spec.component_id] = component
        for row in rows:
            old_key = _atom_key(atoms, row)
            original_to_private[old_key] = (
                str(atoms.chain_id[row]),
                int(atoms.res_id[row]),
                str(atoms.ins_code[row]),
                spec.component_id,
                str(atoms.atom_name[row]),
            )
    if atoms.bonds is not None:
        private.bonds = struc.BondList(len(private), edge_array)
    return private, original_to_private, overrides


__all__ = [
    "AtomKey",
    "LinkedContextError",
    "LinkedContextSpec",
    "linked_context_specs",
    "linked_context_view",
]
