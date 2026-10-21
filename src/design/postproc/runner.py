"""Private payloads and bounded scheduling for optional OpenMM postprocessing."""

from __future__ import annotations

import importlib
import logging
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import biotite.structure as struc
import numpy as np
from rdkit import Chem

from ..core.records import (
    DesignError,
    SearchCandidate,
    ValidatedLigandIdentity,
)


AtomKey = tuple[str, int, str, str, str]


class PostprocessPreparationError(ValueError):
    """One accepted base sample is not eligible for the requested postprocess."""


class OpenMMPreflightError(DesignError):
    """The requested optional OpenMM runtime is not usable by this process."""


_BIOTITE_ORDER_NAME = {
    int(struc.BondType.SINGLE): "single",
    int(struc.BondType.DOUBLE): "double",
    int(struc.BondType.TRIPLE): "triple",
    int(struc.BondType.AROMATIC): "aromatic",
    int(struc.BondType.AROMATIC_SINGLE): "aromatic",
    int(struc.BondType.AROMATIC_DOUBLE): "aromatic",
    int(struc.BondType.AROMATIC_TRIPLE): "aromatic",
}
_AMBERTOOLS_EXECUTABLES = (
    "antechamber",
    "parmchk2",
    "tleap",
)
_CHARGE_METHODS = frozenset(("gasteiger", "am1bcc"))
_RAY_GPU_SLOTS_PER_JOB = 8
_RAY_JOB_NUM_CPUS = 1
_RAY_JOB_NUM_GPUS = 1 / _RAY_GPU_SLOTS_PER_JOB
_RAY_WORKER_MAX_CALLS = 16
_RAY_INFRA_MAX_RETRIES = 1
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GaffLigandRecord:
    record_id: str
    molecule_uid: int
    component_uid: int
    heavy_keys: tuple[AtomKey, ...]
    elements: tuple[str, ...]
    formal_charges: tuple[int, ...]
    bonds: tuple[tuple[int, int, str], ...]
    hydrogen_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        size = len(self.heavy_keys)
        if not self.record_id or size == 0:
            raise ValueError("GAFF record needs an identity and heavy atoms")
        if not (
            len(self.elements)
            == len(self.formal_charges)
            == len(self.hydrogen_counts)
            == size
        ):
            raise ValueError("GAFF record arrays do not share one heavy-atom length")
        if len(set(self.heavy_keys)) != size:
            raise ValueError("GAFF heavy-atom keys must be unique")
        for atom_i, atom_j, order in self.bonds:
            if not 0 <= atom_i < size or not 0 <= atom_j < size or atom_i == atom_j:
                raise ValueError("GAFF bond endpoint is invalid")
            if order not in {"single", "double", "triple", "aromatic"}:
                raise ValueError("GAFF bond order is invalid")


@dataclass(frozen=True, slots=True)
class CovalentLinkRecord:
    polymer_key: AtomKey
    ligand_key: AtomKey
    ligand_record_id: str
    order: str
    product_total_charge: int
    remove_polymer_hydrogen: str
    remove_ligand_hydrogen_count: int
    chemistry_signature: tuple[Any, ...]

    def __post_init__(self) -> None:
        if self.order != "single":
            raise ValueError("the initial covalent route accepts one single bond")
        if not self.ligand_record_id or not self.chemistry_signature:
            raise ValueError("covalent link record is incomplete")
        if self.remove_ligand_hydrogen_count < 0:
            raise ValueError("removed ligand hydrogen count cannot be negative")


@dataclass(frozen=True, slots=True)
class SampleParameterizationPayload:
    task_name: str
    sample_id: str
    gaff_ligands: tuple[GaffLigandRecord, ...] = ()
    covalent_link: CovalentLinkRecord | None = None
    validated_ligand_identities: Mapping[int, ValidatedLigandIdentity] = field(
        default_factory=dict,
    )
    peptide_records: tuple[GaffLigandRecord, ...] = ()
    local_product_records: tuple[GaffLigandRecord, ...] = ()
    local_moving_keys: tuple[AtomKey, ...] = ()
    native_moving_keys: tuple[AtomKey, ...] = ()
    output_atom_aliases: tuple[tuple[AtomKey, AtomKey], ...] = ()
    context_formal_charges: tuple[tuple[AtomKey, int], ...] = ()
    cobalt_hexammine_groups: tuple[tuple[AtomKey, ...], ...] = ()
    linked_context_residues: tuple[Any, ...] = ()
    fixed_ligand_completions: tuple[Any, ...] = ()
    fixed_coordination_records: tuple[Any, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return self.task_name, self.sample_id

    def __post_init__(self) -> None:
        if not self.task_name or not self.sample_id:
            raise ValueError("parameterization payload key cannot be empty")
        cobalt_keys = [key for group in self.cobalt_hexammine_groups for key in group]
        if (any(len(group) != 7 for group in self.cobalt_hexammine_groups)
                or len(cobalt_keys) != len(set(cobalt_keys))):
            raise ValueError("cobalt hexammine atom ownership is invalid")
        gaff_keys = {key for record in (*self.gaff_ligands, *self.peptide_records,
                                       *self.local_product_records) for key in record.heavy_keys}
        if gaff_keys.intersection(cobalt_keys):
            raise ValueError("cobalt hexammine and GAFF atom ownership overlaps")
        private_keys = [
            key for record in (*self.linked_context_residues,
                                *self.fixed_ligand_completions,
                                *self.fixed_coordination_records)
            for key in record.heavy_keys
        ]
        if len(private_keys) != len(set(private_keys)):
            raise ValueError("fixed context chemistry atom ownership overlaps")
        if set(private_keys).intersection(gaff_keys | set(cobalt_keys)):
            raise ValueError("fixed context and GAFF atom ownership overlaps")
        charge_keys = [key for key, _charge in self.context_formal_charges]
        if len(charge_keys) != len(set(charge_keys)):
            raise ValueError("context atom charge facts must be unique")
        if any(isinstance(charge, bool) or not isinstance(charge, int)
               for _key, charge in self.context_formal_charges):
            raise ValueError("context formal charges must be integers")
        sources = [pair[0] for pair in self.output_atom_aliases]
        targets = [pair[1] for pair in self.output_atom_aliases]
        if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
            raise ValueError("output atom aliases must be one-to-one")
        if any(source[:3] + source[4:] != target[:3] + target[4:] for source, target in self.output_atom_aliases):
            raise ValueError("output atom aliases may only change file-local component ID")
        if bool(self.local_product_records) != bool(self.local_moving_keys):
            raise ValueError("local product payload needs both chemistry and motion ownership")
        if self.local_product_records:
            if self.gaff_ligands or self.covalent_link is not None or self.peptide_records:
                raise ValueError("local product and other payload routes are mutually exclusive")
            keys = [key for record in self.local_product_records for key in record.heavy_keys]
            if len(keys) != len(set(keys)) or not set(self.local_moving_keys).issubset(keys):
                raise ValueError("local product atom ownership is invalid")
            if len(self.local_moving_keys) != len(set(self.local_moving_keys)):
                raise ValueError("local product motion ownership is duplicated")
        if self.peptide_records:
            if self.covalent_link is not None:
                raise ValueError("whole-peptide and separate covalent-link routes are mutually exclusive")
            peptide_ids = [record.record_id for record in self.peptide_records]
            if len(peptide_ids) != len(set(peptide_ids)):
                raise ValueError("peptide record IDs must be unique within one sample")
            peptide_keys = [key for record in self.peptide_records for key in record.heavy_keys]
            if len(peptide_keys) != len(set(peptide_keys)):
                raise ValueError("peptide records must not overlap heavy atoms")
            ordinary_keys = {key for record in self.gaff_ligands for key in record.heavy_keys}
            if ordinary_keys.intersection(peptide_keys):
                raise ValueError("whole-peptide and ordinary ligand atom ownership overlaps")
        ids = [record.record_id for record in self.gaff_ligands]
        if len(ids) != len(set(ids)):
            raise ValueError("GAFF record IDs must be unique within one sample")
        if self.covalent_link is not None and self.covalent_link.ligand_record_id not in ids:
            raise ValueError("covalent link does not reference a GAFF ligand record")
        if not isinstance(self.validated_ligand_identities, Mapping):
            raise TypeError("validated_ligand_identities must be a mapping")
        normalized: dict[int, ValidatedLigandIdentity] = {}
        for raw_uid, identity in self.validated_ligand_identities.items():
            if isinstance(raw_uid, bool) or not isinstance(raw_uid, int) or raw_uid < 0:
                raise TypeError("validated ligand identity keys must be nonnegative integers")
            if not isinstance(identity, ValidatedLigandIdentity):
                raise TypeError(
                    "validated_ligand_identities values must be ValidatedLigandIdentity"
                )
            normalized[int(raw_uid)] = identity
        object.__setattr__(self, "validated_ligand_identities", dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class PostprocessJob:
    task_name: str
    sample_id: str
    base_cif: Path
    refined_cif: Path
    payload: SampleParameterizationPayload

    @property
    def key(self) -> tuple[str, str]:
        return self.task_name, self.sample_id


@dataclass(frozen=True, slots=True)
class PostprocessResult:
    task_name: str
    sample_id: str
    succeeded: bool
    message: str = ""
    initial_energy_kj_mol: float | None = None
    final_energy_kj_mol: float | None = None
    rms_force_kj_mol_nm: float | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.task_name, self.sample_id


def gaff_record_from_rdkit(
    molecule: Chem.Mol,
    *,
    record_id: str,
    molecule_uid: int,
    component_uid: int,
    heavy_keys: Sequence[AtomKey],
) -> GaffLigandRecord:
    """Freeze private GAFF chemistry without exposing coordinates or atom maps."""

    source = Chem.Mol(molecule)
    Chem.SanitizeMol(source)
    heavy_indices = [
        atom.GetIdx() for atom in source.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    if len(heavy_indices) != len(heavy_keys):
        raise ValueError("private molecule heavy atoms do not match accepted structure")
    source_to_heavy = {source_index: i for i, source_index in enumerate(heavy_indices)}
    elements: list[str] = []
    charges: list[int] = []
    hydrogens: list[int] = []
    for source_index in heavy_indices:
        atom = source.GetAtomWithIdx(source_index)
        elements.append(atom.GetSymbol().upper())
        charges.append(int(atom.GetFormalCharge()))
        # Include bracket hydrogens (e.g. aromatic [nH]) as well as actual H
        # neighbours. They are chemistry facts even before Chem.AddHs().
        hydrogens.append(int(atom.GetTotalNumHs(includeNeighbors=True)))
    bonds: list[tuple[int, int, str]] = []
    order_name = {
        Chem.BondType.SINGLE: "single",
        Chem.BondType.DOUBLE: "double",
        Chem.BondType.TRIPLE: "triple",
        Chem.BondType.AROMATIC: "aromatic",
    }
    for bond in source.GetBonds():
        source_i, source_j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if source_i not in source_to_heavy or source_j not in source_to_heavy:
            continue
        order = order_name.get(bond.GetBondType())
        if order is None:
            raise ValueError(f"unsupported private ligand bond {bond.GetBondType()}")
        bonds.append((source_to_heavy[source_i], source_to_heavy[source_j], order))
    return GaffLigandRecord(
        record_id=record_id,
        molecule_uid=int(molecule_uid),
        component_uid=int(component_uid),
        heavy_keys=tuple(heavy_keys),
        elements=tuple(elements),
        formal_charges=tuple(charges),
        bonds=tuple(bonds),
        hydrogen_counts=tuple(hydrogens),
    )


def _atom_key(atom_array: struc.AtomArray, row: int) -> AtomKey:
    return (
        str(atom_array.chain_id[row]),
        int(atom_array.res_id[row]),
        str(atom_array.ins_code[row]),
        str(atom_array.res_name[row]),
        str(atom_array.atom_name[row]),
    )


def _normalized_element(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise PostprocessPreparationError("accepted ligand has an empty element")
    return text[:1].upper() + text[1:].lower()


def _authoritative_ligand_bonds(
    atom_array: struc.AtomArray,
    rows: Sequence[int],
) -> tuple[tuple[int, int, str], ...]:
    if atom_array.bonds is None:
        raise PostprocessPreparationError(
            "accepted molecular ligand lacks an authoritative BondList"
        )
    row_to_local = {int(row): index for index, row in enumerate(rows)}
    result: dict[tuple[int, int], str] = {}
    for raw_i, raw_j, raw_type in atom_array.bonds.as_array():
        row_i, row_j = int(raw_i), int(raw_j)
        if row_i not in row_to_local or row_j not in row_to_local:
            continue
        order = _BIOTITE_ORDER_NAME.get(int(raw_type))
        if order is None:
            raise PostprocessPreparationError(
                "accepted ligand graph requires concrete ordinary bond orders"
            )
        key = tuple(sorted((row_to_local[row_i], row_to_local[row_j])))
        previous = result.get(key)
        if previous is not None and previous != order:
            raise PostprocessPreparationError(
                "accepted ligand graph contains conflicting bond orders"
            )
        result[key] = order
    if len(rows) > 1 and not result:
        raise PostprocessPreparationError(
            "accepted molecular ligand has no authoritative internal graph"
        )
    return tuple((i, j, order) for (i, j), order in sorted(result.items()))


def _design_ligand_mode(candidate: SearchCandidate, rows: Sequence[int]) -> str | None:
    atom_array = candidate.atoms
    if not bool(np.asarray(atom_array.is_design, dtype=bool)[list(rows)].all()):
        return None
    chains = {str(atom_array.chain_id[row]) for row in rows}
    if len(chains) != 1:
        raise PostprocessPreparationError(
            "one accepted ligand molecule spans multiple design chains"
        )
    chain = next(iter(chains))
    condition = candidate.base_request.task.condition
    ligand = condition.get("ligand", {}) if isinstance(condition, Mapping) else {}
    if not isinstance(ligand, Mapping):
        raise PostprocessPreparationError("compiled ligand condition is not a mapping")
    entry = ligand.get(chain)
    if entry is None:
        backbone = condition.get("backbone", ()) if isinstance(condition, Mapping) else ()
        return "backbone" if chain in (backbone or ()) else "free"
    if not isinstance(entry, Mapping):
        raise PostprocessPreparationError("compiled ligand mode is not a mapping")
    return str(entry.get("mode"))


def _context_fallback_allowed(candidate, rows) -> bool:
    return not bool(np.asarray(candidate.atoms.is_design, dtype=bool)[list(rows)].any())


def _context_molecule_from_authoritative_graph(atom_array, rows, source_path=None) -> Chem.Mol:
    from .context import context_private_molecule
    return context_private_molecule(atom_array, rows, source_path=source_path)


def _is_skippable_monatomic_ion(
    candidate: SearchCandidate,
    rows: Sequence[int],
) -> bool:
    if len(rows) != 1:
        return False
    row = int(rows[0])
    if not bool(candidate.atoms.is_design[row]):
        return True
    # A complete one-atom CIF component is an ion/template case.  A free
    # generated one-atom ligand still needs private molecular chemistry and is
    # not silently relabelled as an ion.
    return _design_ligand_mode(candidate, rows) == "type_bond"


def _validate_gaff_record_against_structure(
    record: GaffLigandRecord,
    atom_array: struc.AtomArray,
    rows: Sequence[int],
) -> None:
    expected_elements = tuple(_normalized_element(atom_array.element[row]) for row in rows)
    observed_elements = tuple(_normalized_element(value) for value in record.elements)
    if observed_elements != expected_elements:
        raise PostprocessPreparationError(
            "private ligand atom order/elements differ from the accepted structure"
        )
    expected_bonds = _authoritative_ligand_bonds(atom_array, rows)
    observed_bonds = tuple(
        (min(int(i), int(j)), max(int(i), int(j)), str(order))
        for i, j, order in record.bonds
    )
    if tuple(sorted(observed_bonds)) != tuple(sorted(expected_bonds)):
        # A CIF may retain a Kekule ring whereas its trusted private RDKit
        # molecule stores aromatic bonds. Compare sanitized chemistry using
        # exactly the same trusted atom charges/H inventory on BOTH graphs.
        # Never alter the public graph or accept a different edge inventory.
        def normalized(bonds):
            rw = Chem.RWMol()
            for element, charge, hydrogens in zip(record.elements, record.formal_charges, record.hydrogen_counts, strict=True):
                atom = Chem.Atom(_normalized_element(element))
                atom.SetFormalCharge(int(charge))
                atom.SetNoImplicit(True)
                atom.SetNumExplicitHs(int(hydrogens))
                rw.AddAtom(atom)
            types = {"single": Chem.BondType.SINGLE, "double": Chem.BondType.DOUBLE,
                     "triple": Chem.BondType.TRIPLE, "aromatic": Chem.BondType.AROMATIC}
            for i, j, order in bonds:
                rw.AddBond(int(i), int(j), types[order])
            molecule = rw.GetMol()
            Chem.SanitizeMol(molecule)
            return tuple(sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                                 max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), str(b.GetBondType()))
                                for b in molecule.GetBonds()))
        try:
            equivalent = normalized(expected_bonds) == normalized(observed_bonds)
        except (ValueError, RuntimeError):
            equivalent = False
        if equivalent:
            return
        raise PostprocessPreparationError(
            "private ligand graph differs from the authoritative BondList"
        )


def _internal_cys_sg(
    atom_array: struc.AtomArray,
    row: int,
) -> bool:
    if (
        str(atom_array.mol_type[row]) != "protein"
        or str(atom_array.res_name[row]).upper() != "CYS"
        or str(atom_array.atom_name[row]).upper() != "SG"
        or _normalized_element(atom_array.element[row]) != "S"
    ):
        return False
    residue = (
        str(atom_array.chain_id[row]),
        int(atom_array.res_id[row]),
        str(atom_array.ins_code[row]),
    )
    residue_rows = [
        index
        for index in range(len(atom_array))
        if (
            str(atom_array.chain_id[index]),
            int(atom_array.res_id[index]),
            str(atom_array.ins_code[index]),
        )
        == residue
    ]
    names = {str(atom_array.atom_name[index]).upper(): index for index in residue_rows}
    if set(names) != {"N", "CA", "C", "O", "CB", "SG"}:
        return False
    if atom_array.bonds is None:
        return False
    residue_set = set(residue_rows)
    previous_c = False
    next_n = False
    for raw_i, raw_j, _ in atom_array.bonds.as_array():
        i, j = int(raw_i), int(raw_j)
        if i == names["N"] and j not in residue_set:
            previous_c |= str(atom_array.atom_name[j]).upper() == "C"
        elif j == names["N"] and i not in residue_set:
            previous_c |= str(atom_array.atom_name[i]).upper() == "C"
        if i == names["C"] and j not in residue_set:
            next_n |= str(atom_array.atom_name[j]).upper() == "N"
        elif j == names["C"] and i not in residue_set:
            next_n |= str(atom_array.atom_name[i]).upper() == "N"
    return previous_c and next_n


def _nonbonded_metal_relation(atoms, i, j, raw_type):
    """Interpret explicit/legacy metal coordination as a nonbonded relation.

    These edges remain in the public CIF. The native ion force field supplies
    nonbonded interactions; they are not GAFF covalent bonds or peptide links.
    """
    metals = {"MG", "CA", "NA", "K", "BA", "SR", "CS", "ZN", "MN", "FE", "CO", "NI", "CU"}
    if raw_type not in {int(struc.BondType.ANY), int(struc.BondType.COORDINATION)}:
        return False
    for row in (i, j):
        uid = int(atoms.component_uid[row])
        if str(atoms.element[row]).upper() in metals and np.count_nonzero(atoms.component_uid == uid) == 1:
            return True
    return False


def _native_polymer_link(atoms, i, j, raw_type):
    from ..utils.atoms import PROTEIN_RES_ATOMS
    if raw_type != int(struc.BondType.SINGLE):
        return False
    standard = set(PROTEIN_RES_ATOMS) - {"UNK"}
    if not all(str(atoms.mol_type[r]) == "protein" and str(atoms.res_name[r]) in standard for r in (i, j)):
        return False
    names = {str(atoms.atom_name[i]), str(atoms.atom_name[j])}
    return names == {"C", "N"} or (names == {"SG"} and atoms.res_name[i] == atoms.res_name[j] == "CYS")


def _covalent_link_record(
    candidate: SearchCandidate,
    records: Sequence[GaffLigandRecord],
    *, excluded_keys: set[AtomKey] | None = None,
) -> CovalentLinkRecord | None:
    atom_array = candidate.atoms
    if atom_array.bonds is None:
        raise PostprocessPreparationError(
            "accepted structure lacks an authoritative BondList"
        )
    component_uid = np.asarray(atom_array.component_uid, dtype=np.int64)
    cross_component = [
        (int(i), int(j), int(raw_type))
        for i, j, raw_type in atom_array.bonds.as_array()
        if component_uid[int(i)] != component_uid[int(j)]
        and not (excluded_keys and _atom_key(atom_array, int(i)) in excluded_keys
                 and _atom_key(atom_array, int(j)) in excluded_keys)
        and not _native_polymer_link(atom_array, int(i), int(j), int(raw_type))
        and not _nonbonded_metal_relation(atom_array, int(i), int(j), int(raw_type))
    ]
    if not cross_component:
        return None
    if len(cross_component) != 1:
        raise PostprocessPreparationError(
            "initial OpenMM covalent scope requires exactly one cross-component bond"
        )
    row_i, row_j, raw_type = cross_component[0]
    if raw_type == int(struc.BondType.COORDINATION):
        raise PostprocessPreparationError(
            "metal coordination is outside the supported OpenMM routes"
        )
    if _BIOTITE_ORDER_NAME.get(raw_type) != "single":
        raise PostprocessPreparationError(
            "initial OpenMM covalent scope requires one concrete single bond"
        )
    modalities = (str(atom_array.mol_type[row_i]), str(atom_array.mol_type[row_j]))
    if set(modalities) != {"protein", "ligand"}:
        raise PostprocessPreparationError(
            "initial OpenMM covalent scope is protein CYS-SG to CIF ligand only"
        )
    protein_row = row_i if modalities[0] == "protein" else row_j
    ligand_row = row_j if protein_row == row_i else row_i
    if not _internal_cys_sg(atom_array, protein_row):
        raise PostprocessPreparationError(
            "polymer covalent endpoint is not one standard internal CYS SG"
        )
    input_path = candidate.base_request.task.input_path
    if input_path is None or input_path.suffix.lower() != ".cif":
        raise PostprocessPreparationError(
            "the supported covalent ligand must come from topology-rich CIF input"
        )
    if len(records) != 1:
        raise PostprocessPreparationError(
            "initial covalent compiler supports exactly one molecular ligand"
        )
    record = records[0]
    ligand_key = _atom_key(atom_array, ligand_row)
    if ligand_key not in record.heavy_keys:
        raise PostprocessPreparationError(
            "covalent ligand endpoint is absent from its private chemistry record"
        )
    linked_local = record.heavy_keys.index(ligand_key)
    if int(record.hydrogen_counts[linked_local]) < 1:
        raise PostprocessPreparationError(
            "supported CYS-ligand product requires one removable ligand hydrogen"
        )
    product_charge = int(sum(int(value) for value in record.formal_charges))
    signature = (
        "cys-sg-cif-ligand",
        record.elements,
        record.formal_charges,
        record.bonds,
        record.hydrogen_counts,
        linked_local,
        product_charge,
    )
    return CovalentLinkRecord(
        polymer_key=_atom_key(atom_array, protein_row),
        ligand_key=ligand_key,
        ligand_record_id=record.record_id,
        order="single",
        product_total_charge=product_charge,
        remove_polymer_hydrogen="HG",
        remove_ligand_hydrogen_count=1,
        chemistry_signature=signature,
    )


def build_parameterization_payload(candidate, *, task_name: str, sample_id: str) -> SampleParameterizationPayload:
    """Freeze chemistry and the same file-local ligand names used by the writer."""
    from dataclasses import replace
    from ..utils.writer import _component_ids

    payload = _build_parameterization_payload(candidate, task_name=task_name, sample_id=sample_id)
    atoms = candidate.atoms
    uids, names = _component_ids(atoms, candidate.validated_ligand_identities)
    aliases = []
    for row in np.flatnonzero(np.asarray(atoms.mol_type).astype(str) == "ligand"):
        source = _atom_key(atoms, int(row))
        target = (*source[:3], names[int(uids[row])], source[4])
        if source != target:
            aliases.append((source, target))
    fixed_na = ~np.asarray(atoms.is_design, dtype=bool) & np.isin(atoms.mol_type, ("rna", "dna"))
    context_charges = ()
    if (payload.linked_context_residues or payload.fixed_ligand_completions
            or payload.fixed_coordination_records):
        from .context import source_formal_charge_facts
        context_charges = source_formal_charge_facts(
            atoms, candidate.base_request.task.input_path,
            rows=np.flatnonzero(~np.asarray(atoms.is_design, dtype=bool)))
    elif np.any(fixed_na & ~np.isin(atoms.res_name, ("A", "C", "G", "U", "DA", "DC", "DG", "DT"))):
        from .context import source_formal_charge_facts
        context_charges = source_formal_charge_facts(
            atoms, candidate.base_request.task.input_path, rows=np.flatnonzero(fixed_na))
    return replace(payload, output_atom_aliases=tuple(aliases),
                   context_formal_charges=context_charges)


def payload_for_output(payload: SampleParameterizationPayload, atoms) -> SampleParameterizationPayload:
    """Bind private keys to the emitted CIF without changing public identity."""
    from dataclasses import replace
    from .openmm.refine import OpenMMPostprocessError

    keys = {_atom_key(atoms, row): row for row in range(len(atoms))}
    if len(keys) != len(atoms):
        raise OpenMMPostprocessError("output atom identity is not unique")
    aliases = dict(payload.output_atom_aliases)
    def rekey(key):
        if key in keys:
            return key
        target = aliases.get(key)
        if target not in keys or str(atoms.mol_type[keys[target]]) != "ligand":
            raise OpenMMPostprocessError(f"private atom key is absent from emitted CIF: {key}")
        return target
    def records(values):
        result = []
        for record in values:
            updated = replace(record, heavy_keys=tuple(rekey(key) for key in record.heavy_keys))
            try:
                _validate_gaff_record_against_structure(updated, atoms, [keys[key] for key in updated.heavy_keys])
            except PostprocessPreparationError as error:
                raise OpenMMPostprocessError("emitted CIF differs from private product chemistry") from error
            result.append(updated)
        return tuple(result)
    link = payload.covalent_link
    if link is not None:
        link = replace(link, polymer_key=rekey(link.polymer_key), ligand_key=rekey(link.ligand_key))
    return replace(payload, gaff_ligands=records(payload.gaff_ligands),
                   peptide_records=records(payload.peptide_records),
                   local_product_records=records(payload.local_product_records),
                   local_moving_keys=tuple(rekey(key) for key in payload.local_moving_keys),
                   native_moving_keys=tuple(rekey(key) for key in payload.native_moving_keys),
                   context_formal_charges=tuple((rekey(key), charge)
                                               for key, charge in payload.context_formal_charges),
                   cobalt_hexammine_groups=tuple(tuple(rekey(key) for key in group)
                                                  for group in payload.cobalt_hexammine_groups),
                   linked_context_residues=tuple(
                       replace(spec, heavy_keys=tuple(rekey(key) for key in spec.heavy_keys),
                               linker_keys=tuple((rekey(left), rekey(right))
                                                 for left, right in spec.linker_keys))
                       for spec in payload.linked_context_residues),
                   fixed_ligand_completions=tuple(
                       replace(spec, heavy_keys=tuple(rekey(key) for key in spec.heavy_keys))
                       for spec in payload.fixed_ligand_completions),
                   fixed_coordination_records=tuple(
                       replace(record, heavy_keys=tuple(rekey(key) for key in record.heavy_keys))
                       for record in payload.fixed_coordination_records),
                   covalent_link=link)


def _native_design_scope(candidate):
    """Select canonical whole-NA and native-compatible non-linear peptides."""
    from ..utils.atoms import PROTEIN_RES_ATOMS
    atoms = candidate.atoms
    design = np.asarray(atoms.is_design, dtype=bool)
    nucleic = design & np.isin(atoms.mol_type, ("rna", "dna"))
    native_na = bool(nucleic.any()) and all(
        str(atoms.res_name[r]) in {"A", "C", "G", "U", "DA", "DC", "DG", "DT"}
        for r in np.flatnonzero(nucleic))
    edges = () if atoms.bonds is None else atoms.bonds.as_array()
    special_native = False
    standard = set(PROTEIN_RES_ATOMS) - {"UNK"}
    protein = design & (atoms.mol_type == "protein")
    native_protein = bool(protein.any()) and all(
        str(atoms.res_name[r]) in standard for r in np.flatnonzero(protein))
    for raw_i, raw_j, _ in edges:
        i, j = int(raw_i), int(raw_j)
        if bool(nucleic[i]) != bool(nucleic[j]) and not _nonbonded_metal_relation(atoms, i, j, int(_)):
            native_na = False
        if not (protein[i] and protein[j]):
            continue
        same = (atoms.chain_id[i], atoms.res_id[i], atoms.ins_code[i]) == (
            atoms.chain_id[j], atoms.res_id[j], atoms.ins_code[j])
        if same:
            continue
        pair = {str(atoms.atom_name[i]), str(atoms.atom_name[j])}
        if pair == {"C", "N"}:
            c, n = (i, j) if atoms.atom_name[i] == "C" else (j, i)
            if atoms.chain_id[c] != atoms.chain_id[n] or int(atoms.res_id[n]) != int(atoms.res_id[c]) + 1:
                special_native = True
        elif pair == {"SG"} and atoms.res_name[i] == atoms.res_name[j] == "CYS":
            special_native = True
        else:
            native_protein = False
    moving = tuple(_atom_key(atoms, int(r)) for r in np.flatnonzero(design)) if (
        native_na or (native_protein and special_native)) else ()
    return native_na, moving


def _build_parameterization_payload(
    candidate: SearchCandidate,
    *,
    task_name: str,
    sample_id: str,
) -> SampleParameterizationPayload:
    """Freeze winner-local chemistry into a serializable, coordinate-free payload.

    The payload is private postprocess input.  It contains neither ``AtomArray`` nor
    public charge/atom-map facts and is safe to gather only after final selection.
    """

    atom_array = candidate.atoms
    required = {"mol_type", "molecule_uid", "component_uid", "is_design"}
    missing = required.difference(atom_array.get_annotation_categories())
    if missing:
        raise RuntimeError(
            f"accepted candidate lacks postprocess annotations: {sorted(missing)}"
        )
    if not np.isfinite(np.asarray(atom_array.coord, dtype=np.float64)).all():
        raise PostprocessPreparationError(
            "accepted base structure has non-finite coordinates"
        )
    if atom_array.bonds is None:
        raise PostprocessPreparationError(
            "postprocess requires an explicit bond graph (an empty BondList is valid)"
        )
    from .openmm.local_refine import prepare_local_products
    local_products, local_moving = prepare_local_products(candidate)
    if local_products:
        return SampleParameterizationPayload(
            task_name=str(task_name), sample_id=str(sample_id),
            validated_ligand_identities=dict(candidate.validated_ligand_identities),
            local_product_records=local_products, local_moving_keys=local_moving,
        )
    from .openmm.peptide import peptide_records
    from .openmm.nucleic import whole_nucleic_records

    from .openmm.linked_context import linked_context_specs, linked_context_view
    from .openmm.fixed_ligand import fixed_ligand_completions
    try:
        linked_specs = linked_context_specs(candidate)
        linked_keys = {key for spec in linked_specs for key in spec.heavy_keys}
        completions = fixed_ligand_completions(candidate, excluded_keys=linked_keys)
    except ValueError as error:
        raise PostprocessPreparationError(str(error)) from error
    completion_keys = {key for spec in completions for key in spec.heavy_keys}
    fixed_polymer_keys = set()
    if linked_specs:
        # Ignore only a proven closed, fixed polymer product in the separate
        # covalent-link route. A link into design or another modality is rejected.
        view, _, _ = linked_context_view(atom_array, linked_specs)
        view.bonds = struc.BondList(len(view), np.asarray([
            (i, j, order) for i, j, order in view.bonds.as_array()
            if not _nonbonded_metal_relation(view, int(i), int(j), int(order))
        ], dtype=np.uint32).reshape(-1, 3))
        from .openmm.context_polymer import _components
        for group in _components(view):
            if not any(_atom_key(atom_array, row) in linked_keys for row in group):
                continue
            if any(view.is_design[row] or view.mol_type[row] not in ("protein", "rna", "dna")
                   for row in group):
                raise PostprocessPreparationError("linked context must be a closed fixed polymer component")
            fixed_polymer_keys.update(_atom_key(atom_array, row) for row in group)
    native_na, native_moving = _native_design_scope(candidate)
    whole_peptides = () if native_na else whole_nucleic_records(candidate)
    if whole_peptides:
        # Whole-NA selection may leave disjoint design units. Include them
        # rather than silently freezing another design chain in this route.
        from dataclasses import replace
        from .openmm.product_graph import build_product_records
        selected = {key for record in whole_peptides for key in record.heavy_keys}
        remaining = {int(r) for r in np.flatnonzero(atom_array.is_design)
                     if _atom_key(atom_array, int(r)) not in selected}
        adjacency = [set() for _ in atom_array]
        for i, j, _ in atom_array.bonds.as_array():
            adjacency[int(i)].add(int(j))
            adjacency[int(j)].add(int(i))
        groups = []
        while remaining:
            group, queue = {min(remaining)}, [min(remaining)]
            while queue:
                for neighbor in adjacency[queue.pop()]:
                    if neighbor not in group:
                        if neighbor not in remaining:
                            raise PostprocessPreparationError("whole-polymer design component crosses context")
                        group.add(neighbor)
                        queue.append(neighbor)
            remaining.difference_update(group)
            groups.append(sorted(group))
        whole_peptides += tuple(replace(record, record_id=f"nucleic-extra-{ordinal}")
                                for ordinal, record in enumerate(build_product_records(candidate, groups)))
    else:
        whole_peptides = peptide_records(candidate)
    selected_keys = {key for record in whole_peptides for key in record.heavy_keys}
    from ..utils.atoms import PROTEIN_RES_ATOMS
    fixed = ~np.asarray(atom_array.is_design, dtype=bool)
    modified_fixed = fixed & (
        ((atom_array.mol_type == "protein")
         & ~np.isin(atom_array.res_name, tuple(set(PROTEIN_RES_ATOMS) - {"UNK"})))
        | (np.isin(atom_array.mol_type, ("rna", "dna"))
           & ~np.isin(atom_array.res_name, ("A", "C", "G", "U", "DA", "DC", "DG", "DT")))
    )
    if modified_fixed.any():
        # A complete fixed modified-polymer product belongs to the existing
        # whole-context parameterizer, including links across source chains.
        # Keep its graph intact; only exclude it from the separate PL-link
        # classifier. Any component touching design or another modality stays
        # subject to that classifier's normal restrictions.
        from .openmm.context_polymer import _components
        for group in _components(atom_array):
            group = np.asarray(group, dtype=np.int64)
            if (np.any(modified_fixed[group]) and np.all(fixed[group])
                    and len(set(np.asarray(atom_array.mol_type)[group])) == 1
                    and str(atom_array.mol_type[group[0]]) in ("protein", "rna", "dna")):
                fixed_polymer_keys.update(_atom_key(atom_array, row) for row in group)
    if whole_peptides or modified_fixed.any() or linked_specs or completions:
        # Public CIF does not carry design ownership. Preserve it explicitly
        # for the private fixed-context route, including protein/ligand designs.
        native_moving = tuple(_atom_key(atom_array, int(row))
                              for row in np.flatnonzero(atom_array.is_design))
    ligand_rows = np.asarray([
        int(row) for row in np.flatnonzero(np.asarray(atom_array.mol_type).astype(str) == "ligand")
        if _atom_key(atom_array, int(row)) not in
        (selected_keys | linked_keys | completion_keys)], dtype=np.int64)
    molecule_map = {int(key): value for key, value in candidate.private_ligand_molecules.items()}
    error_map = {int(key): str(value) for key, value in candidate.private_ligand_errors.items()}
    records: list[GaffLigandRecord] = []
    cobalt_groups = []
    coordination_records = []
    seen_molecule_uids = {int(atom_array.molecule_uid[row]) for row in range(len(atom_array))
                          if _atom_key(atom_array, row) in linked_keys | completion_keys}
    seen_molecule_uids.update({
        int(atom_array.molecule_uid[row]) for row in range(len(atom_array))
        if _atom_key(atom_array, row) in selected_keys and str(atom_array.mol_type[row]) == "ligand"})
    groups = {}
    for row in ligand_rows:
        identity = (int(atom_array.molecule_uid[row]), int(atom_array.component_uid[row]))
        groups.setdefault(identity, []).append(int(row))
    for (molecule_uid, component_uid), rows in groups.items():
        if molecule_uid < 0 or component_uid < 0:
            raise RuntimeError("accepted ligand has an invalid molecule/component UID")
        seen_molecule_uids.add(molecule_uid)
        if _is_skippable_monatomic_ion(candidate, rows):
            continue
        if not np.any(atom_array.is_design[rows]):
            from .context import context_component_name
            if context_component_name(atom_array, rows, source_path=candidate.base_request.task.input_path) == "NCO":
                from .openmm.cobalt import validate_cobalt_heavy_group
                cobalt_groups.append(validate_cobalt_heavy_group(atom_array, rows))
                continue
        if not np.any(atom_array.is_design[rows]):
            from .context import context_component_name, _load_components, _CONTEXT_COMPONENTS
            name = context_component_name(atom_array, rows, source_path=candidate.base_request.task.input_path)
            _load_components({name})
            component = _CONTEXT_COMPONENTS.get(name)
            trusted = Chem.Mol(component.rdkit_binary) if component is not None else None
            if trusted is not None and any(
                    bond.GetBondType() == Chem.BondType.DATIVE for bond in trusted.GetBonds()):
                from .openmm.coordination import build_fixed_coordination_record
                try:
                    coordination_records.append(build_fixed_coordination_record(
                        atom_array, rows, component=component, component_id=name))
                except ValueError as error:
                    raise PostprocessPreparationError(str(error)) from error
                continue
        components = {component_uid}
        molecule = molecule_map.get(molecule_uid)
        source_error = error_map.get(molecule_uid)
        if molecule is not None and source_error is not None:
            raise RuntimeError(
                "one accepted ligand has both private chemistry and a private error"
            )
        if molecule is None:
            if not _context_fallback_allowed(candidate, rows):
                detail = source_error or "no private ligand chemistry is available"
                raise PostprocessPreparationError(
                    f"ligand molecule {molecule_uid} is not parameterizable: {detail}"
                )
            try:
                molecule = _context_molecule_from_authoritative_graph(atom_array, rows, candidate.base_request.task.input_path)
            except Exception as error:
                raise PostprocessPreparationError(
                    f"retained context component {component_uid} has no matching authoritative chemistry: {error}"
                ) from error
        heavy_keys = tuple(_atom_key(atom_array, row) for row in rows)
        if len(heavy_keys) != len(set(heavy_keys)):
            raise PostprocessPreparationError(
                "accepted ligand heavy-atom keys are not unique"
            )
        try:
            record = gaff_record_from_rdkit(
                molecule,
                record_id=f"ligand-{molecule_uid}-{component_uid}",
                molecule_uid=molecule_uid,
                component_uid=next(iter(components)),
                heavy_keys=heavy_keys,
            )
            _validate_gaff_record_against_structure(record, atom_array, rows)
        except PostprocessPreparationError:
            raise
        except Exception as error:
            raise PostprocessPreparationError(
                f"private chemistry for ligand molecule {molecule_uid} is invalid"
            ) from error
        records.append(record)
    unexpected_private = set(molecule_map).difference(seen_molecule_uids)
    if unexpected_private:
        raise RuntimeError(
            f"private chemistry references absent ligand molecules: {sorted(unexpected_private)}"
        )
    if coordination_records:
        native_moving = tuple(_atom_key(atom_array, int(row))
                              for row in np.flatnonzero(atom_array.is_design))
    link = _covalent_link_record(candidate, records,
                                 excluded_keys=selected_keys | fixed_polymer_keys)
    return SampleParameterizationPayload(
        task_name=str(task_name),
        sample_id=str(sample_id),
        gaff_ligands=tuple(records),
        peptide_records=whole_peptides,
        covalent_link=link,
        validated_ligand_identities=dict(candidate.validated_ligand_identities),
        native_moving_keys=native_moving,
        cobalt_hexammine_groups=tuple(cobalt_groups),
        linked_context_residues=linked_specs,
        fixed_ligand_completions=completions,
        fixed_coordination_records=tuple(coordination_records),
    )


def _ambertools_executable(name: str) -> Path:
    path = Path(sys.executable).resolve().parent / name
    if not path.is_file() or not os.access(path, os.X_OK):
        raise OpenMMPreflightError(
            f"required AmberTools executable is unavailable: {path}"
        )
    return path


@contextmanager
def _ambertools_on_path() -> Iterator[None]:
    """Expose tools beside the active absolute Python only for one operation."""

    environment_bin = str(Path(sys.executable).resolve().parent)
    original_path = os.environ.get("PATH")
    os.environ["PATH"] = os.pathsep.join(
        (environment_bin, "" if original_path is None else original_path)
    )
    try:
        yield
    finally:
        if original_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original_path


def _probe_openmm_context(openmm: Any, platform: Any, *, platform_name: str) -> None:
    """Create and release a real minimal Context on the requested platform."""

    system = openmm.System()
    system.addParticle(1.0)
    integrator = openmm.VerletIntegrator(0.001)
    properties = (
        {"DeviceIndex": str(int(os.environ.get("LOCAL_RANK", "0")))}
        if platform_name == "CUDA"
        else {}
    )
    context = openmm.Context(system, integrator, platform, properties)
    del context
    del integrator


def preflight_postprocessing(config: Mapping[str, Any]) -> None:
    """Validate OpenMM resources; the caller gates this on selected task switches."""

    if "method" in config:
        raise OpenMMPreflightError(
            "postprocess.method was removed; use task postprocess: true or false"
        )
    charge_method = config.get("charge_method", "gasteiger")
    if charge_method not in _CHARGE_METHODS:
        raise OpenMMPreflightError(
            "OpenMM charge_method must be gasteiger or am1bcc"
        )
    platform_name = str(config.get("platform", "CUDA")).upper()
    use_ray = bool(config.get("use_ray", True))
    if str(config.get("force_field")) != "amber19sb":
        raise OpenMMPreflightError("OpenMM postprocess requires amber19sb")
    if (platform_name, use_ray) not in {("CUDA", True), ("CPU", False)}:
        raise OpenMMPreflightError(
            "OpenMM accepts only CUDA+Ray or the CPU/no-Ray diagnostic"
        )
    try:
        openmm = importlib.import_module("openmm")
        openmm_app = importlib.import_module("openmm.app")
        generators = importlib.import_module("openmmforcefields.generators")
        toolkits = importlib.import_module("openff.toolkit.utils.toolkits")
        importlib.import_module("parmed")
        importlib.import_module("pdbfixer")
        if use_ray:
            importlib.import_module("ray")
    except Exception as error:
        raise OpenMMPreflightError(
            f"optional OpenMM dependency is unavailable: {type(error).__name__}: {error}"
        ) from error
    required_executables = _AMBERTOOLS_EXECUTABLES + (
        (("sqm",) if charge_method == "am1bcc" else ())
    )
    for executable in required_executables:
        _ambertools_executable(executable)
    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
        if platform_name == "CUDA":
            _probe_openmm_context(
                openmm,
                platform,
                platform_name=platform_name,
            )
        openmm_app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
        gaff = generators.GAFFTemplateGenerator
        if "gaff-2.2.20" not in tuple(gaff.INSTALLED_FORCEFIELDS):
            raise RuntimeError("GAFF 2.2.20 resource is not installed")
        with _ambertools_on_path():
            wrapper_type = toolkits.AmberToolsToolkitWrapper
            if not wrapper_type.is_available():
                raise RuntimeError("OpenFF cannot select the AmberTools toolkit")
            wrapper = wrapper_type()
            registry = toolkits.ToolkitRegistry([wrapper])
            registered = tuple(registry.registered_toolkits)
            if len(registered) != 1 or registered[0].toolkit_name != "AmberTools":
                raise RuntimeError("OpenFF registry did not select AmberTools")
            supported_methods = getattr(wrapper, "_supported_charge_methods", {})
            if charge_method not in supported_methods:
                raise RuntimeError(
                    f"AmberTools does not support charge method {charge_method}"
                )
    except OpenMMPreflightError:
        raise
    except Exception as error:
        raise OpenMMPreflightError(
            f"optional OpenMM resource/platform preflight failed: {error}"
        ) from error


def _run_one(job: PostprocessJob, config: Mapping[str, Any]) -> PostprocessResult:
    from .openmm.refine import OpenMMPostprocessError, refine_one

    _LOG.info("OpenMM task=%s sample=%s worker_pid=%s", job.task_name, job.sample_id, os.getpid())
    if job.refined_cif.is_symlink():
        return PostprocessResult(job.task_name, job.sample_id, False,
                                 "OpenMMPostprocessError: refined destination is a symlink")
    if job.refined_cif.exists():
        # A worker may finish its atomic staging write and die before Ray gets
        # its return value. Validate and reuse that stage; never overwrite it.
        from .resume import validate_refined_artifact
        try:
            validate_refined_artifact(job.base_cif, job.refined_cif, job.payload)
        except (ValueError, RuntimeError, OSError) as error:
            return PostprocessResult(
                job.task_name, job.sample_id, False,
                f"OpenMMPostprocessError: refined destination already exists and is invalid ({error})",
            )
        _LOG.info("OpenMM reused verified stage task=%s sample=%s", job.task_name, job.sample_id)
        return PostprocessResult(job.task_name, job.sample_id, True)
    try:
        initial_energy, final_energy, rms_force = refine_one(
            job.base_cif,
            job.refined_cif,
            config=config,
            payload=job.payload,
        )
        return PostprocessResult(
            job.task_name,
            job.sample_id,
            True,
            initial_energy_kj_mol=initial_energy,
            final_energy_kj_mol=final_energy,
            rms_force_kj_mol_nm=rms_force,
        )
    except OpenMMPostprocessError as error:
        job.refined_cif.unlink(missing_ok=True)
        return PostprocessResult(
            job.task_name,
            job.sample_id,
            False,
            f"{type(error).__name__}: {error}",
        )
    finally:
        import gc
        gc.collect()


def run_postprocessing(
    jobs: Sequence[PostprocessJob],
    *,
    config: Mapping[str, Any],
    on_result: Callable[[PostprocessJob, PostprocessResult], None] | None = None,
) -> tuple[PostprocessResult, ...]:
    """Run accepted samples and publish each completed result on the driver."""

    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("postprocess job keys must be globally unique")
    if any(job.payload.key != job.key for job in jobs):
        raise ValueError("postprocess payload key does not match its accepted sample")
    if not jobs:
        return ()
    use_ray = bool(config.get("use_ray", True))
    platform = str(config.get("platform", "CUDA")).upper()
    if not use_ray:
        if platform != "CPU":
            raise ValueError("sequential postprocessing is a CPU diagnostic only")
        completed = []
        for job in jobs:
            result = _run_one(job, config)
            if result.key != job.key:
                raise RuntimeError("postprocess result violated sample ownership")
            if on_result is not None:
                on_result(job, result)
            completed.append(result)
        return tuple(completed)
    if platform != "CUDA":
        raise ValueError("Ray postprocessing requires the CUDA platform")

    import ray

    started_here = not ray.is_initialized()
    if not started_here:
        raise RuntimeError("standalone inference does not attach to an existing Ray runtime")
    results: dict[tuple[str, str], PostprocessResult] = {}
    try:
        ray.init(address="local", include_dashboard=False)
        from .openmm.refine import OpenMMSystemError

        resources = ray.cluster_resources()
        cpu_capacity = math.floor(float(resources.get("CPU", 0.0)))
        gpu_capacity = math.floor(
            float(resources.get("GPU", 0.0)) * _RAY_GPU_SLOTS_PER_JOB
        )
        max_inflight = min(cpu_capacity, gpu_capacity)
        if max_inflight < 1:
            raise OpenMMSystemError(
                "runner-owned local Ray runtime exposes no feasible resources "
                f"(CPU capacity={cpu_capacity}, GPU capacity={gpu_capacity})"
            )
        remote_run = ray.remote(
            num_cpus=_RAY_JOB_NUM_CPUS,
            num_gpus=_RAY_JOB_NUM_GPUS,
            max_retries=_RAY_INFRA_MAX_RETRIES,
            retry_exceptions=False,
            max_calls=_RAY_WORKER_MAX_CALLS,
        )(_run_one)
        job_by_key = {job.key: job for job in jobs}
        pending: dict[Any, tuple[str, str]] = {}
        iterator = iter(jobs)
        exhausted = False
        while pending or not exhausted:
            while len(pending) < max_inflight and not exhausted:
                try:
                    job = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                reference = remote_run.remote(job, dict(config))
                pending[reference] = job.key
            if not pending:
                continue
            ready, _ = ray.wait(list(pending), num_returns=1)
            reference = ready[0]
            expected_key = pending.pop(reference)
            try:
                result = ray.get(reference)
            except Exception as error:
                raise OpenMMSystemError(
                    f"OpenMM Ray task={expected_key[0]} sample={expected_key[1]} failed ("
                    f"infrastructure retry limit={_RAY_INFRA_MAX_RETRIES}): "
                    f"{type(error).__name__}: {error}"
                ) from error
            if result.key != expected_key or result.key in results:
                raise RuntimeError("Ray postprocess result violated sample ownership")
            if on_result is not None:
                on_result(job_by_key[expected_key], result)
            results[result.key] = result
    finally:
        if started_here and ray.is_initialized():
            ray.shutdown()
    if set(results) != set(keys):
        raise RuntimeError("Ray postprocess result accounting is incomplete")
    return tuple(results[key] for key in keys)


__all__ = [
    "AtomKey",
    "CovalentLinkRecord",
    "GaffLigandRecord",
    "OpenMMPreflightError",
    "PostprocessJob",
    "PostprocessPreparationError",
    "PostprocessResult",
    "SampleParameterizationPayload",
    "build_parameterization_payload",
    "preflight_postprocessing",
    "run_postprocessing",
    "gaff_record_from_rdkit",
]
