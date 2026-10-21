import torch
from typing import Final

MOL_TYPE_TO_ID: Final[dict[str, int]] = {
    "protein": 0,
    "rna": 1,
    "dna": 2,
    "ligand": 3,
}
ID_TO_MOL_TYPE: Final[dict[int, str]] = {v: k for k, v in MOL_TYPE_TO_ID.items()}


# Protenix STD_RESIDUES_WITH_GAP. Keep the ids byte-for-byte compatible.
TOKEN_TYPE32: Final[dict[str, int]] = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "UNK": 20,
    "A": 21,
    "G": 22,
    "C": 23,
    "U": 24,
    "N": 25,
    "DA": 26,
    "DG": 27,
    "DC": 28,
    "DT": 29,
    "DN": 30,
    "-": 31,
}
ID_TO_TOKEN_TYPE32: Final[dict[int, str]] = {v: k for k, v in TOKEN_TYPE32.items()}
TOKEN_TYPE_IGNORE_INDEX: Final[int] = -100
GAP_TOKEN_ID: Final[int] = TOKEN_TYPE32["-"]


PROTEIN_RES_NAMES: Final[tuple[str, ...]] = tuple(
    res
    for res, idx in sorted(TOKEN_TYPE32.items(), key=lambda item: item[1])
    if idx <= 20
)
RNA_RES_NAMES: Final[tuple[str, ...]] = ("A", "G", "C", "U", "N")
DNA_RES_NAMES: Final[tuple[str, ...]] = ("DA", "DG", "DC", "DT", "DN")
POLYMER_RES_NAMES: Final[frozenset[str]] = frozenset(
    PROTEIN_RES_NAMES + RNA_RES_NAMES + DNA_RES_NAMES
)


# Protein atom37 order used by the current PartiDesign VAE.
PROTEIN_ATOM37: Final[tuple[str, ...]] = (
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
)


PROTEIN_RES_ATOMS: Final[dict[str, tuple[str, ...]]] = {
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
    "TRP": (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ),
    "TYR": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
    "UNK": ("N", "CA", "C", "O"),
}


# Atom91 polymer decoder schema.
#
# RNA/DNA deliberately use expanded true atom names instead of the RFD3NA
# slash/virtual slots.  Each slot corresponds to exactly one atom name, which
# keeps the coordinate target and atom mask unambiguous.
RNA_ATOM27: Final[tuple[str, ...]] = (
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
    "N9",
    "C8",
    "N7",
    "C5",
    "C6",
    "N6",
    "O6",
    "N1",
    "C2",
    "N2",
    "O2",
    "N3",
    "C4",
    "N4",
    "O4",
)
DNA_ATOM27: Final[tuple[str, ...]] = (
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "C1'",
    "N9",
    "C8",
    "N7",
    "C5",
    "C6",
    "N6",
    "O6",
    "N1",
    "C2",
    "N2",
    "O2",
    "N3",
    "C4",
    "N4",
    "O4",
    "C7",
)

RNA_RES_ATOMS: Final[dict[str, tuple[str, ...]]] = {
    "A": (
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
    ),
    "G": (
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "O6",
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
    ),
    "C": (
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "N4",
        "C5",
        "C6",
    ),
    "U": (
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C6",
    ),
    "N": RNA_ATOM27[:12],
}

DNA_RES_ATOMS: Final[dict[str, tuple[str, ...]]] = {
    "DA": (
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
    ),
    "DG": (
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "O6",
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
    ),
    "DC": (
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "N4",
        "C5",
        "C6",
    ),
    "DT": (
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C7",
        "C6",
    ),
    "DN": DNA_ATOM27[:11],
}


ATOM91_SIZE: Final[int] = 91
CONDITION_TOKEN_BOND_FLAG_DIM: Final[int] = 1
CONDITION_TOKEN_BOND_ENDPOINT_LIGAND_SLOT: Final[int] = ATOM91_SIZE
CONDITION_TOKEN_BOND_ENDPOINT_VOCAB_SIZE: Final[int] = ATOM91_SIZE + 1
CONDITION_TOKEN_BOND_PAIR_DIM: Final[int] = (
    CONDITION_TOKEN_BOND_FLAG_DIM + 2 * CONDITION_TOKEN_BOND_ENDPOINT_VOCAB_SIZE
)
PROTEIN_ATOM91_OFFSET: Final[int] = 0
RNA_ATOM91_OFFSET: Final[int] = 37
DNA_ATOM91_OFFSET: Final[int] = 64

RNA_ATOM27_TO_SLOT: Final[dict[str, int]] = {
    name: idx for idx, name in enumerate(RNA_ATOM27)
}
DNA_ATOM27_TO_SLOT: Final[dict[str, int]] = {
    name: idx for idx, name in enumerate(DNA_ATOM27)
}
PROTEIN_ATOM37_TO_SLOT: Final[dict[str, int]] = {
    name: idx for idx, name in enumerate(PROTEIN_ATOM37)
}

PROTEIN_RES_ATOM37_MASK: Final[dict[str, tuple[bool, ...]]] = {
    res: tuple(atom in atoms for atom in PROTEIN_ATOM37)
    for res, atoms in PROTEIN_RES_ATOMS.items()
}
RNA_RES_ATOM27_MASK: Final[dict[str, tuple[bool, ...]]] = {
    res: tuple(atom in atoms for atom in RNA_ATOM27)
    for res, atoms in RNA_RES_ATOMS.items()
}
DNA_RES_ATOM27_MASK: Final[dict[str, tuple[bool, ...]]] = {
    res: tuple(atom in atoms for atom in DNA_ATOM27)
    for res, atoms in DNA_RES_ATOMS.items()
}

NA_BACKBONE_ATOMS: Final[frozenset[str]] = frozenset(
    ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'")
)
NA_SUGAR_ATOMS: Final[frozenset[str]] = frozenset(("O4'", "C1'", "C2'", "O2'"))
NA_BASE_ATOMS: Final[frozenset[str]] = frozenset(
    (
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "O6",
        "N1",
        "C2",
        "N2",
        "O2",
        "N3",
        "C4",
        "N4",
        "O4",
        "C7",
    )
)


ELEMENTS_128: Final[tuple[str, ...]] = (
    "H",
    "HE",
    "LI",
    "BE",
    "B",
    "C",
    "N",
    "O",
    "F",
    "NE",
    "NA",
    "MG",
    "AL",
    "SI",
    "P",
    "S",
    "CL",
    "AR",
    "K",
    "CA",
    "SC",
    "TI",
    "V",
    "CR",
    "MN",
    "FE",
    "CO",
    "NI",
    "CU",
    "ZN",
    "GA",
    "GE",
    "AS",
    "SE",
    "BR",
    "KR",
    "RB",
    "SR",
    "Y",
    "ZR",
    "NB",
    "MO",
    "TC",
    "RU",
    "RH",
    "PD",
    "AG",
    "CD",
    "IN",
    "SN",
    "SB",
    "TE",
    "I",
    "XE",
    "CS",
    "BA",
    "LA",
    "CE",
    "PR",
    "ND",
    "PM",
    "SM",
    "EU",
    "GD",
    "TB",
    "DY",
    "HO",
    "ER",
    "TM",
    "YB",
    "LU",
    "HF",
    "TA",
    "W",
    "RE",
    "OS",
    "IR",
    "PT",
    "AU",
    "HG",
    "TL",
    "PB",
    "BI",
    "PO",
    "AT",
    "RN",
    "FR",
    "RA",
    "AC",
    "TH",
    "PA",
    "U",
    "NP",
    "PU",
    "AM",
    "CM",
    "BK",
    "CF",
    "ES",
    "FM",
    "MD",
    "NO",
    "LR",
    "RF",
    "DB",
    "SG",
    "BH",
    "HS",
    "MT",
    "DS",
    "RG",
    "CN",
    "NH",
    "FL",
    "MC",
    "LV",
    "TS",
    "OG",
    "UNK_ELEM_119",
    "UNK_ELEM_120",
    "UNK_ELEM_121",
    "UNK_ELEM_122",
    "UNK_ELEM_123",
    "UNK_ELEM_124",
    "UNK_ELEM_125",
    "UNK_ELEM_126",
    "UNK_ELEM_127",
    "UNK_ELEM_128",
)
ELEMENT_TO_ID: Final[dict[str, int]] = {elem: i for i, elem in enumerate(ELEMENTS_128)}
UNKNOWN_ELEMENT_ID: Final[int] = ELEMENT_TO_ID["UNK_ELEM_128"]

# Predicted design ligands are restricted to classical covalent heavy elements.
# The list covers the non-metal chemistry observed in a full resolved
# weightedPDB ligand-endpoint scan; target/context atoms remain unrestricted.
DESIGN_LIGAND_ELEMENTS: Final[tuple[str, ...]] = (
    "B",
    "C",
    "N",
    "O",
    "F",
    "SI",
    "P",
    "S",
    "CL",
    "AS",
    "SE",
    "BR",
    "I",
)
DESIGN_LIGAND_ELEMENT_IDS: Final[tuple[int, ...]] = tuple(
    ELEMENT_TO_ID[element] for element in DESIGN_LIGAND_ELEMENTS
)


LIGAND_BOND_TYPES: Final[dict[str, int]] = {
    "none": 0,
    "single": 1,
    "double": 2,
    "triple": 3,
    "aromatic": 4,
}

# Exhaustive MiDi GEOM-Drugs train/val/test scan (2026-07-29).  Keep the
# physical formal-charge value separate from its categorical decoder index.
LIGAND_FORMAL_CHARGES: Final[tuple[int, ...]] = (-2, -1, 0, 1, 2, 3)
LIGAND_FORMAL_CHARGE_TO_CLASS: Final[dict[int, int]] = {
    value: index for index, value in enumerate(LIGAND_FORMAL_CHARGES)
}
LIGAND_FORMAL_CHARGE_CLASS_TO_VALUE: Final[dict[int, int]] = {
    index: value for index, value in enumerate(LIGAND_FORMAL_CHARGES)
}
assert len(LIGAND_FORMAL_CHARGE_TO_CLASS) == len(LIGAND_FORMAL_CHARGES)


def normalise_atom_name(atom_name: str) -> str:
    atom = str(atom_name).strip().upper().replace("*", "'")
    aliases = {
        "O1P": "OP1",
        "O2P": "OP2",
        "O3P": "OP3",
    }
    return aliases.get(atom, atom)


def normalise_res_name(res_name: str) -> str:
    return str(res_name).strip().upper()


def element_id(element: str) -> int:
    elem = str(element).strip().upper()
    return ELEMENT_TO_ID.get(elem, UNKNOWN_ELEMENT_ID)


def atom_slot_index(mol_type: str, res_name: str, atom_name: str) -> int | None:
    mol = str(mol_type).strip().lower()
    res = normalise_res_name(res_name)
    atom = normalise_atom_name(atom_name)
    if mol == "protein":
        return PROTEIN_ATOM37_TO_SLOT.get(atom)
    if mol == "rna":
        names = RNA_RES_ATOMS.get(res)
        if names is None:
            names = RNA_RES_ATOMS["N"]
        if atom not in names:
            return None
        return RNA_ATOM27_TO_SLOT.get(atom)
    if mol == "dna":
        names = DNA_RES_ATOMS.get(res)
        if names is None:
            names = DNA_RES_ATOMS["DN"]
        if atom not in names:
            return None
        return DNA_ATOM27_TO_SLOT.get(atom)
    return None


def polymer_slot_to_atom91(mol_type: str, slot_idx: int) -> int:
    mol = str(mol_type).strip().lower()
    idx = int(slot_idx)
    if mol == "protein":
        assert 0 <= idx < len(PROTEIN_ATOM37)
        return PROTEIN_ATOM91_OFFSET + idx
    if mol == "rna":
        assert 0 <= idx < len(RNA_ATOM27)
        return RNA_ATOM91_OFFSET + idx
    if mol == "dna":
        assert 0 <= idx < len(DNA_ATOM27)
        return DNA_ATOM91_OFFSET + idx
    raise ValueError(f"atom91 has no polymer slot for mol_type={mol_type!r}")


assert len(TOKEN_TYPE32) == 32
assert len(PROTEIN_ATOM37) == 37
assert len(set(PROTEIN_ATOM37)) == len(PROTEIN_ATOM37)
assert len(PROTEIN_RES_ATOM37_MASK) == len(PROTEIN_RES_NAMES)
assert all(
    len(mask) == len(PROTEIN_ATOM37) for mask in PROTEIN_RES_ATOM37_MASK.values()
)
assert all(
    atom in PROTEIN_ATOM37 for atoms in PROTEIN_RES_ATOMS.values() for atom in atoms
)
assert len(RNA_ATOM27) == 27
assert len(DNA_ATOM27) == 27
assert len(set(RNA_ATOM27)) == len(RNA_ATOM27)
assert len(set(DNA_ATOM27)) == len(DNA_ATOM27)
assert all("/" not in atom for atom in RNA_ATOM27)
assert all("/" not in atom for atom in DNA_ATOM27)
assert PROTEIN_ATOM91_OFFSET == 0
assert RNA_ATOM91_OFFSET == PROTEIN_ATOM91_OFFSET + len(PROTEIN_ATOM37)
assert DNA_ATOM91_OFFSET == RNA_ATOM91_OFFSET + len(RNA_ATOM27)
assert DNA_ATOM91_OFFSET + len(DNA_ATOM27) == ATOM91_SIZE
assert CONDITION_TOKEN_BOND_FLAG_DIM == 1
assert CONDITION_TOKEN_BOND_ENDPOINT_LIGAND_SLOT == ATOM91_SIZE == 91
assert CONDITION_TOKEN_BOND_ENDPOINT_VOCAB_SIZE == ATOM91_SIZE + 1 == 92
assert CONDITION_TOKEN_BOND_PAIR_DIM == 1 + 92 + 92 == 185
assert all(len(mask) == len(RNA_ATOM27) for mask in RNA_RES_ATOM27_MASK.values())
assert all(len(mask) == len(DNA_ATOM27) for mask in DNA_RES_ATOM27_MASK.values())
assert all(atom in RNA_ATOM27 for atoms in RNA_RES_ATOMS.values() for atom in atoms)
assert all(atom in DNA_ATOM27 for atoms in DNA_RES_ATOMS.values() for atom in atoms)
assert atom_slot_index("rna", "G", "O6") == RNA_ATOM27_TO_SLOT["O6"]
assert atom_slot_index("rna", "A", "O6") is None
assert atom_slot_index("dna", "DT", "C7") == DNA_ATOM27_TO_SLOT["C7"]
assert len(ELEMENTS_128) == 128
assert len(DESIGN_LIGAND_ELEMENTS) == len(set(DESIGN_LIGAND_ELEMENTS))
assert all(element in ELEMENT_TO_ID for element in DESIGN_LIGAND_ELEMENTS)


# Old PartiFlow Stuff
# PDB and OpenFold have different atom ordering, these utils convert between the two
# PDB ordering: https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_1992.pdf
# Protein atom37 ordering used by this project.

ATOM_NUMBERING = {
    "N": 0,
    "CA": 1,
    "C": 2,
    "O": 3,
    "CB": 4,
    "OG": 5,
    "CG": 6,
    "CD1": 7,
    "CD2": 8,
    "CE1": 9,
    "CE2": 10,
    "CZ": 11,
    "OD1": 12,
    "ND2": 13,
    "CG1": 14,
    "CG2": 15,
    "CD": 16,
    "CE": 17,
    "NZ": 18,
    "OD2": 19,
    "OE1": 20,
    "NE2": 21,
    "OE2": 22,
    "OH": 23,
    "NE": 24,
    "NH1": 25,
    "NH2": 26,
    "OG1": 27,
    "SD": 28,
    "ND1": 29,
    "SG": 30,
    "NE1": 31,
    "CE3": 32,
    "CZ2": 33,
    "CZ3": 34,
    "CH2": 35,
    "OXT": 36,
}

PDB_TO_OPENFOLD_INDEX_TENSOR = torch.tensor(
    [ATOM_NUMBERING[atom] for atom in PROTEIN_ATOM37]
)

OPENFOLD_TO_PDB_INDEX_TENSOR = torch.tensor(
    [PROTEIN_ATOM37.index(atom) for atom in ATOM_NUMBERING]
)

AA_CHARACTER_PROTORP = {
    "ALA": "A",
    "CYS": "P",
    "GLU": "C",
    "ASP": "C",
    "GLY": "A",
    "PHE": "A",
    "ILE": "A",
    "HIS": "P",
    "LYS": "C",
    "MET": "A",
    "LEU": "A",
    "ASN": "P",
    "GLN": "P",
    "PRO": "A",
    "SER": "P",
    "ARG": "C",
    "THR": "P",
    "TRP": "P",
    "VAL": "A",
    "TYR": "P",
}

SIDECHAIN_TIP_ATOMS = {
    "ALA": ["CA", "CB"],
    "ARG": ["CD", "CZ", "NE", "NH1", "NH2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "ASN": ["CB", "CG", "ND2", "OD1"],
    "CYS": ["CA", "CB", "SG"],
    "GLU": ["CG", "CD", "OE1", "OE2"],
    "GLN": ["CG", "CD", "NE2", "OE1"],
    "GLY": [],
    "HIS": ["CB", "CG", "CD2", "CE1", "ND1", "NE2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CE", "NZ"],
    "MET": ["CG", "CE", "SD"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["CA", "CB", "CG", "CD", "N"],
    "SER": ["CA", "CB", "OG"],
    "THR": ["CA", "CB", "CG2", "OG1"],
    "TRP": ["CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "NE1"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["CB", "CG1", "CG2"],
}
