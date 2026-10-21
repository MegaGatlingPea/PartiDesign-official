from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from typing import Any

import numpy as np
import biotite.structure as struc

from ..core.records import DesignInputError


MOL_TYPE_TO_ID = {"protein": 0, "rna": 1, "dna": 2, "ligand": 3}
TOKEN_TYPE32 = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4, "GLN": 5,
    "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9, "LEU": 10, "LYS": 11,
    "MET": 12, "PHE": 13, "PRO": 14, "SER": 15, "THR": 16, "TRP": 17,
    "TYR": 18, "VAL": 19, "UNK": 20, "A": 21, "G": 22, "C": 23,
    "U": 24, "N": 25, "DA": 26, "DG": 27, "DC": 28, "DT": 29,
    "DN": 30, "-": 31,
}
TOKEN_TYPE_IGNORE_INDEX = -100

PROTEIN_ATOM37 = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
    "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
    "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
)
PROTEIN_RES_ATOMS = {
    "ALA": ("N", "CA", "C", "O", "CB"),
    "ARG": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
    "GLY": ("N", "CA", "C", "O"),
    "HIS": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
    "PHE": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD"),
    "SER": ("N", "CA", "C", "O", "CB", "OG"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    "TRP": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
    "UNK": ("N", "CA", "C", "O"),
}
RNA_ATOM27 = (
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'",
    "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "O6", "N1", "C2",
    "N2", "O2", "N3", "C4", "N4", "O4",
)
DNA_ATOM27 = (
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'",
    "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "O6", "N1", "C2", "N2",
    "O2", "N3", "C4", "N4", "O4", "C7",
)
RNA_RES_ATOMS = {
    "A": ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
    "G": ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
    "C": ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "U": ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
    "N": RNA_ATOM27[:12],
}
DNA_RES_ATOMS = {
    "DA": ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
    "DG": ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
    "DC": ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "DT": ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"),
    "DN": DNA_ATOM27[:11],
}
SIDECHAIN_TIP_ATOMS = {
    "ALA": ("CA", "CB"), "ARG": ("CD", "CZ", "NE", "NH1", "NH2"),
    "ASP": ("CB", "CG", "OD1", "OD2"), "ASN": ("CB", "CG", "ND2", "OD1"),
    "CYS": ("CA", "CB", "SG"), "GLU": ("CG", "CD", "OE1", "OE2"),
    "GLN": ("CG", "CD", "NE2", "OE1"), "GLY": (),
    "HIS": ("CB", "CG", "CD2", "CE1", "ND1", "NE2"),
    "ILE": ("CB", "CG1", "CG2", "CD1"), "LEU": ("CB", "CG", "CD1", "CD2"),
    "LYS": ("CE", "NZ"), "MET": ("CG", "CE", "SD"),
    "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("CA", "CB", "CG", "CD", "N"), "SER": ("CA", "CB", "OG"),
    "THR": ("CA", "CB", "CG2", "OG1"),
    "TRP": ("CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "NE1"),
    "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("CB", "CG1", "CG2"),
}

ELEMENTS_128 = (
    "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG", "AL",
    "SI", "P", "S", "CL", "AR", "K", "CA", "SC", "TI", "V", "CR", "MN", "FE",
    "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y",
    "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE",
    "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB",
    "DY", "HO", "ER", "TM", "YB", "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT",
    "AU", "HG", "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH", "PA",
    "U", "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM", "MD", "NO", "LR", "RF",
    "DB", "SG", "BH", "HS", "MT", "DS", "RG", "CN", "NH", "FL", "MC", "LV", "TS",
    "OG", "UNK_ELEM_119", "UNK_ELEM_120", "UNK_ELEM_121", "UNK_ELEM_122",
    "UNK_ELEM_123", "UNK_ELEM_124", "UNK_ELEM_125", "UNK_ELEM_126", "UNK_ELEM_127",
    "UNK_ELEM_128",
)
ELEMENT_TO_ID = {element: index for index, element in enumerate(ELEMENTS_128)}
UNKNOWN_ELEMENT_ID = ELEMENT_TO_ID["UNK_ELEM_128"]
LIGAND_BOND_TYPES = {"none": 0, "single": 1, "double": 2, "triple": 3, "aromatic": 4}

# Small, release-owned standard-parent kernel.  External CCD data extends this
# table for uncommon modified components; ambiguous/non-standard parents still
# fail rather than being guessed from atom names.
COMMON_STANDARD_PARENT = {
    "MSE": "MET", "FME": "MET", "SEP": "SER", "TPO": "THR",
    "PTR": "TYR", "SEC": "CYS", "CSO": "CYS", "CME": "CYS", "SCH": "CYS",
    "PYL": "LYS", "KCX": "LYS", "LLP": "LYS", "MLZ": "LYS",
    "HYP": "PRO",
    "PSU": "U", "5MC": "C", "OMU": "U", "1MA": "A",
    "2MG": "G", "M2G": "G", "7MG": "G", "OMG": "G",
    "H2U": "U", "5MU": "U",
    "5CM": "DC", "5HM": "DC", "5FC": "DC", "5HC": "DC",
}


_SCHEMA_DTYPES: dict[str, np.dtype[Any]] = {
    "chain_id": np.dtype("<U32"), "res_id": np.dtype(np.int64),
    "ins_code": np.dtype("<U8"), "res_name": np.dtype("<U16"),
    "atom_name": np.dtype("<U32"), "element": np.dtype("<U3"),
    "formal_charge": np.dtype(object),
    "hetero": np.dtype(np.bool_), "mol_type": np.dtype("<U8"),
    "is_design": np.dtype(np.bool_), "coord_valid": np.dtype(np.bool_),
    "centre_atom_mask": np.dtype(np.bool_), "identity_condition": np.dtype(np.bool_),
    "motif_coord_condition": np.dtype(np.bool_), "token_center_condition": np.dtype(np.bool_),
    "hotspot_condition": np.dtype(np.bool_), "entity_uid": np.dtype(np.int64),
    "asym_uid": np.dtype(np.int64), "component_uid": np.dtype(np.int64),
    "molecule_uid": np.dtype(np.int64), "residue_uid": np.dtype(np.int64),
    "token_uid": np.dtype(np.int64), "atom_uid": np.dtype(np.int64),
    "label_entity_id": np.dtype("<U128"), "label_asym_id": np.dtype("<U128"),
    "label_seq_id": np.dtype("<U64"), "auth_asym_id": np.dtype("<U128"),
    "auth_seq_id": np.dtype("<U64"), "auth_ins_code": np.dtype("<U8"),
    "label_alt_id": np.dtype("<U8"),
    "source_format": np.dtype("<U4"),
    "source_graph_complete": np.dtype(np.bool_),
    "source_break_before": np.dtype(np.bool_),
    "source_polymer_sequence": np.dtype(object),
    "source_polymer_position": np.dtype(np.int64),
}
ATOM_ANNOTATIONS = tuple(_SCHEMA_DTYPES)
UID_FIELDS = (
    "entity_uid", "asym_uid", "component_uid", "molecule_uid",
    "residue_uid", "token_uid", "atom_uid",
)


def create_atom_array(length: int) -> struc.AtomArray:
    n = int(length)
    if n < 0:
        raise ValueError("AtomArray length must be nonnegative")
    atoms = struc.AtomArray(n)
    atoms.coord = np.zeros((n, 3), dtype=np.float32)
    for name, dtype in _SCHEMA_DTYPES.items():
        default: Any
        if dtype.kind in {"i", "u"}:
            default = -1 if name in UID_FIELDS else 0
        elif dtype.kind == "b":
            default = False
        elif dtype.kind == "O":
            default = None
        else:
            default = ""
        atoms.set_annotation(name, np.full(n, default, dtype=dtype))
    atoms.bonds = struc.BondList(n)
    validate_atom_array(atoms, phase="allocated")
    return atoms


def _validate_string_width(values: np.ndarray, width: int, name: str) -> None:
    too_long = [str(value) for value in values if len(str(value)) > width]
    if too_long:
        raise DesignInputError(f"{name} exceeds width {width}: {too_long[:3]}")


def validate_atom_array(atoms: struc.AtomArray, *, phase: str = "allocated") -> None:
    if phase not in {"allocated", "identity", "tokenized"}:
        raise ValueError(f"unknown AtomArray validation phase: {phase}")
    if atoms.coord.shape != (len(atoms), 3) or atoms.coord.dtype != np.float32:
        raise TypeError("AtomArray coordinates must be float32[A,3]")
    categories = set(atoms.get_annotation_categories())
    optional_antibody = {
        "antibody_chain_role_id": np.dtype(np.int64),
        "antibody_region_id": np.dtype(np.int64),
        "antibody_sequence_mutable": np.dtype(np.bool_),
        "antibody_source_label_seq_id": np.dtype(np.int64),
    }
    present = categories & set(optional_antibody)
    if present and present != set(optional_antibody):
        raise TypeError("antibody annotations must be supplied as one complete group")
    expected = set(ATOM_ANNOTATIONS) | present
    for name in present:
        value = atoms.get_annotation(name)
        if value.dtype != optional_antibody[name] or value.shape != (len(atoms),):
            raise TypeError(f"invalid optional antibody annotation {name}")
    if present:
        role, region = atoms.antibody_chain_role_id, atoms.antibody_region_id
        if not np.isin(role, (0, 1, 2)).all() or not np.isin(region, (0, 1, 2, 3, 4)).all():
            raise ValueError("antibody role/region outside the trained ABI")
        if np.any(atoms.antibody_sequence_mutable & ((role == 0) | ~atoms.is_design)):
            raise ValueError("antibody mutable annotation cannot select context")
    if categories != expected:
        raise TypeError(
            f"AtomArray annotation schema mismatch: missing={sorted(expected-categories)}, "
            f"extra={sorted(categories-expected)}"
        )
    for name, dtype in _SCHEMA_DTYPES.items():
        value = atoms.get_annotation(name)
        if value.dtype != dtype or value.shape != (len(atoms),):
            raise TypeError(f"AtomArray annotation {name} has {value.dtype}/{value.shape}, expected {dtype}")
        if dtype.kind == "U":
            _validate_string_width(value, dtype.itemsize // 4, name)
    if atoms.bonds is not None:
        edges = atoms.bonds.as_array()
        if len(edges) and ((edges[:, :2] < 0).any() or (edges[:, :2] >= len(atoms)).any()):
            raise ValueError("BondList contains an out-of-range endpoint")
    valid = atoms.coord_valid
    if valid.any() and not np.isfinite(atoms.coord[valid]).all():
        raise ValueError("coord_valid rows must have finite coordinates")
    if (~valid).any() and not np.array_equal(atoms.coord[~valid], np.zeros_like(atoms.coord[~valid])):
        raise ValueError("coordinate-invalid rows must store zero coordinates")
    if phase in {"identity", "tokenized"} and len(atoms):
        for name in UID_FIELDS:
            if name == "token_uid" and phase == "identity":
                if not (atoms.token_uid == -1).all():
                    raise ValueError("token_uid must remain -1 before tokenization")
                continue
            if (atoms.get_annotation(name) < 0).any():
                raise ValueError(f"{name} must be finalized")
        if len(np.unique(atoms.atom_uid)) != len(atoms):
            raise ValueError("atom_uid must be unique")
        if (atoms.res_id <= 0).any():
            raise ValueError("normalized residue positions must be positive")
        if not set(atoms.mol_type) <= set(MOL_TYPE_TO_ID):
            raise ValueError("every atom must have a supported modality")
    if phase == "tokenized" and len(atoms):
        token_ids = np.unique(atoms.token_uid)
        if not np.array_equal(token_ids, np.arange(len(token_ids), dtype=np.int64)):
            raise ValueError("token_uid values must be contiguous from zero")


def subset_atom_array(atoms: struc.AtomArray, selector: Any) -> struc.AtomArray:
    validate_atom_array(atoms)
    selected = atoms[selector]
    if isinstance(selected, struc.Atom):
        selected = atoms[np.asarray([int(selector)])]
    selected.coord = np.asarray(selected.coord, dtype=np.float32)
    validate_atom_array(selected)
    return selected


def concatenate_atom_arrays(arrays: Sequence[struc.AtomArray]) -> struc.AtomArray:
    if not arrays:
        return create_atom_array(0)
    for atoms in arrays:
        validate_atom_array(atoms)
    if len(arrays) == 1:
        result = arrays[0].copy()
    else:
        result = struc.concatenate(list(arrays))
    result.coord = np.asarray(result.coord, dtype=np.float32)
    validate_atom_array(result)
    return result


def _factorize(keys: Iterable[Any]) -> np.ndarray:
    mapping: dict[Any, int] = {}
    result: list[int] = []
    for key in keys:
        if key not in mapping:
            mapping[key] = len(mapping)
        result.append(mapping[key])
    return np.asarray(result, dtype=np.int64)


def assign_identity_uids(
    atoms: struc.AtomArray,
    *,
    entity_keys: Sequence[Any] | None = None,
    asym_keys: Sequence[Any] | None = None,
    component_keys: Sequence[Any] | None = None,
    molecule_keys: Sequence[Any] | None = None,
    residue_keys: Sequence[Any] | None = None,
) -> None:
    n = len(atoms)
    if asym_keys is None:
        asym_keys = list(atoms.chain_id)
    if residue_keys is None:
        residue_keys = list(zip(asym_keys, atoms.res_id.tolist(), atoms.ins_code.tolist()))
    if entity_keys is None:
        entity_keys = [value if value else asym for value, asym in zip(atoms.label_entity_id, asym_keys)]
    if component_keys is None:
        component_keys = [
            (asym, "polymer") if mol != "ligand" else (residue, "ligand")
            for asym, residue, mol in zip(asym_keys, residue_keys, atoms.mol_type)
        ]
    if molecule_keys is None:
        molecule_keys = list(component_keys)
    for name, keys in (
        ("entity_uid", entity_keys), ("asym_uid", asym_keys),
        ("component_uid", component_keys), ("molecule_uid", molecule_keys),
        ("residue_uid", residue_keys),
    ):
        if len(keys) != n:
            raise ValueError(f"{name} key count does not match AtomArray")
        atoms.set_annotation(name, _factorize(keys))
    atoms.atom_uid = np.arange(n, dtype=np.int64)
    atoms.token_uid.fill(-1)


def tokenize_atom_array(
    atoms: struc.AtomArray,
    *,
    carrier_residue_uids: Collection[int] = (),
) -> None:
    """Assign authoring tokens and one representative row per token.

    Ordinary polymer tokens retain the strict CA/C1' representative contract.
    A source-backed modified-polymer carrier may lack that conventional atom;
    only for an explicitly named carrier residue do we then choose the row with
    the lowest finalized AtomUID.  The authoring token remains residue-level --
    atomization belongs to :mod:`src.design.core.model_tokens`.
    """

    validate_atom_array(atoms, phase="identity")
    normalized_carriers = {int(value) for value in carrier_residue_uids}
    token_keys: list[Any] = []
    for index in range(len(atoms)):
        if atoms.mol_type[index] == "ligand":
            token_keys.append(("ligand", int(atoms.atom_uid[index])))
        else:
            token_keys.append(("polymer", int(atoms.residue_uid[index])))
    atoms.token_uid = _factorize(token_keys)
    atoms.centre_atom_mask.fill(False)
    for token_uid in range(int(atoms.token_uid.max()) + 1 if len(atoms) else 0):
        indices = np.where(atoms.token_uid == token_uid)[0]
        modality = str(atoms.mol_type[indices[0]])
        preferred = "CA" if modality == "protein" else "C1'" if modality in {"rna", "dna"} else None
        residue_uids = {int(atoms.residue_uid[index]) for index in indices}
        is_carrier = bool(residue_uids.intersection(normalized_carriers))
        if is_carrier and (
            len(residue_uids) != 1 or not residue_uids <= normalized_carriers
        ):
            raise DesignInputError(
                f"token {token_uid} mixes a carrier residue with other rows"
            )
        if preferred is None:
            center = int(indices[0])
        else:
            matches = indices[atoms.atom_name[indices] == preferred]
            if len(matches) == 1:
                center = int(matches[0])
            elif len(matches) == 0 and is_carrier:
                ordered = sorted(
                    (int(atoms.atom_uid[index]), int(index)) for index in indices
                )
                center = ordered[0][1]
            else:
                raise DesignInputError(
                    f"token {token_uid} requires exactly one representative atom {preferred}"
                )
        atoms.centre_atom_mask[center] = True
    validate_atom_array(atoms, phase="tokenized")


def atom_uid_bonds(atoms: struc.AtomArray) -> set[tuple[int, int, int]]:
    if atoms.bonds is None:
        return set()
    result: set[tuple[int, int, int]] = set()
    for row_i, row_j, bond_type in atoms.bonds.as_array():
        uid_i = int(atoms.atom_uid[int(row_i)])
        uid_j = int(atoms.atom_uid[int(row_j)])
        result.add((min(uid_i, uid_j), max(uid_i, uid_j), int(bond_type)))
    return result
