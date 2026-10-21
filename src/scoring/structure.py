"""Inference-only structure annotations and tokenization.

This module provides the narrow structure identity/token surface consumed by
PartiScore. It is an independent implementation of the AlphaFold 3 /
Protenix parser and tokenizer behavior used by the inference path; the
training parser, CIF readers, assembly logic, and data samplers remain
outside this module.

The residue and element inventories are sourced from the native inference
constants in src.common.constants. The behavior of the small
annotation/token operations follows the corresponding AlphaFold 3 parser
contract, without importing or vendoring Protenix.

Behavioral attribution: this bounded subset follows the Apache-2.0
AlphaFold 3 / Protenix parser.py and tokenizer.py implementation in the
read-only reference checkout at protenix/data/{core/parser.py,tokenizer.py};
only the inference operations listed above are reimplemented here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import logging
from typing import Iterable, Optional

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.molecules import get_molecule_indices

from src.common.constants import (
    DNA_RES_ATOMS,
    ELEMENTS_128,
    PROTEIN_RES_ATOMS,
    RNA_RES_ATOMS,
    TOKEN_TYPE32,
)


# Protenix's standard residue table excludes the gap token. Keeping this
# derived table local avoids importing the training/data namespace.
STD_RESIDUES = {
    name: int(token_id)
    for name, token_id in TOKEN_TYPE32.items()
    if name != "-"
}

# Atom tokens follow the residue token range. ELEMENTS_128 uses the same
# periodic-table ordering as AlphaFold 3's tokenizer inventory.
ELEMS = {
    element: len(STD_RESIDUES) + index
    for index, element in enumerate(ELEMENTS_128)
}


def _indexed_atom_names(
    atom_names: Iterable[str], *, prefix: Iterable[str] = ()
) -> dict[str, int]:
    """Build the residue-local atom-name-to-slot map used by tokatom_idx."""

    names = tuple(prefix) + tuple(atom_names)
    return {name: index for index, name in enumerate(names)}


def _build_res_atoms_dict() -> dict[str, dict[str, int]]:
    """Reconstruct the small native atom-slot inventory.

    The native inference atom inventories intentionally omit terminal OXT and
    nucleotide OP3 slots because those are optional source atoms. The
    original parser's token-atom annotation includes them, with OXT appended
    to protein residues and OP3 prepended to nucleotide residues.
    """

    result: dict[str, dict[str, int]] = {}
    for residue, atom_names in PROTEIN_RES_ATOMS.items():
        # The legacy standard UNK token has two additional canonical slots
        # even though the native atom37 inventory does not materialize them.
        if residue == "UNK":
            atom_names = tuple(atom_names) + ("CB", "CG")
        result[residue] = _indexed_atom_names(atom_names, prefix=())
        result[residue]["OXT"] = len(atom_names)
    for inventory in (RNA_RES_ATOMS, DNA_RES_ATOMS):
        for residue, atom_names in inventory.items():
            result[residue] = _indexed_atom_names(atom_names, prefix=("OP3",))
    return result


RES_ATOMS_DICT = _build_res_atoms_dict()


class Token:
    """One residue or atom token with lightweight arbitrary annotations."""

    def __init__(self, value, **kwargs):
        self.value = value
        self._annot = {}
        for name, annotation in kwargs.items():
            self._annot[name] = annotation

    def __getattr__(self, attr):
        if attr in super().__getattribute__("_annot"):
            return self._annot[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __repr__(self):
        annot_lst = []
        for key, value in self._annot.items():
            annot_lst.append(f"{key}={value}")
        return f"Token({self.value}, {','.join(annot_lst)})"

    def __setattr__(self, attr, value):
        if attr == "_annot":
            super().__setattr__(attr, value)
        elif attr == "value":
            super().__setattr__(attr, value)
        else:
            self._annot[attr] = value


class TokenArray:
    """A list-like collection of Token objects."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens

    def __repr__(self):
        repr_str = "TokenArray(\n"
        for token in self.tokens:
            repr_str += f"\t{token}\n"
        repr_str += ")"
        return repr_str

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        yield from self.tokens

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.tokens[index]
        return TokenArray([self.tokens[i] for i in index])

    def get_annotation(self, category):
        return [token._annot[category] for token in self.tokens]

    def set_annotation(self, category, values):
        assert len(values) == len(
            self.tokens
        ), "Length of values must match the number of tokens"
        for token, value in zip(self.tokens, values):
            token._annot[category] = value

    def get_values(self):
        return [token.value for token in self.tokens]


class AtomArrayTokenizer:
    """Tokenize a Biotite AtomArray into residue and atom tokens."""

    def __init__(self, atom_array: AtomArray):
        self.atom_array = atom_array

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        total_atom_num = 0
        for residue in struc.residue_iter(self.atom_array):
            atom_num = len(residue)
            first_atom = residue[0]
            residue_token = STD_RESIDUES.get(first_atom.res_name, None)
            if residue_token is not None and first_atom.mol_type != "ligand":
                token = Token(residue_token)
                atom_indices = list(
                    range(total_atom_num, total_atom_num + atom_num)
                )
                atom_names = [
                    self.atom_array[index].atom_name for index in atom_indices
                ]
                token.atom_indices = atom_indices
                token.atom_names = atom_names
                tokens.append(token)
                total_atom_num += atom_num
            else:
                for atom in residue:
                    atom_token = ELEMS.get(atom.element, None)
                    if atom_token is None:
                        raise ValueError(f"Unknown atom element: {atom.element}")
                    token = Token(atom_token)
                    token.atom_indices = [total_atom_num]
                    token.atom_names = [
                        self.atom_array[token.atom_indices[0]].atom_name
                    ]
                    tokens.append(token)
                    total_atom_num += 1

        assert total_atom_num == len(self.atom_array)
        return tokens

    def _set_token_annotations(self, token_array: TokenArray) -> TokenArray:
        centre_atom_indices = np.where(
            self.atom_array.centre_atom_mask == 1
        )[0]
        token_array.set_annotation("centre_atom_index", centre_atom_indices)
        assert len(token_array) == len(centre_atom_indices)
        return token_array

    def get_token_array(self) -> TokenArray:
        token_array = TokenArray(tokens=self.tokenize())
        return self._set_token_annotations(token_array)


class AddAtomArrayAnnot:
    """Add the inference structure annotations required by PartiScore."""

    @staticmethod
    def add_modified_res_mask(atom_array: AtomArray) -> AtomArray:
        modified_res_mask = []
        starts = struc.get_residue_starts(
            atom_array, add_exclusive_stop=True
        )
        for start, stop in zip(starts[:-1], starts[1:]):
            res_name = atom_array.res_name[start]
            mol_type = atom_array.mol_type[start]
            if res_name not in STD_RESIDUES and mol_type != "ligand":
                modified_res_mask.extend([1] * (stop - start))
            else:
                modified_res_mask.extend([0] * (stop - start))
        atom_array.set_annotation("modified_res_mask", modified_res_mask)
        return atom_array

    @staticmethod
    def add_centre_atom_mask(atom_array: AtomArray) -> AtomArray:
        res_name = list(STD_RESIDUES.keys())
        std_res = np.isin(atom_array.res_name, res_name) & (
            atom_array.mol_type != "ligand"
        )
        prot_res = np.char.str_len(atom_array.res_name) == 3
        prot_centre_atom = prot_res & (atom_array.atom_name == "CA")
        nuc_centre_atom = (~prot_res) & (atom_array.atom_name == "C1'")
        not_std_res = ~std_res
        centre_atom_mask = (
            std_res & (prot_centre_atom | nuc_centre_atom)
        ) | not_std_res
        atom_array.set_annotation(
            "centre_atom_mask", centre_atom_mask.astype(int)
        )
        return atom_array

    @staticmethod
    def add_ref_space_uid(atom_array: AtomArray) -> AtomArray:
        chain_res_id = np.vstack(
            (atom_array.asym_id_int, atom_array.res_id)
        ).T
        unique_id = np.unique(chain_res_id, axis=0)
        mapping_dict = {}
        for index, pair in enumerate(unique_id):
            asym_id_int, res_id = pair
            mapping_dict[(asym_id_int, res_id)] = index
        ref_space_uid = [
            mapping_dict[(asym_id_int, res_id)]
            for asym_id_int, res_id in chain_res_id
        ]
        atom_array.set_annotation("ref_space_uid", ref_space_uid)
        return atom_array

    @staticmethod
    def add_tokatom_idx(atom_array: AtomArray) -> AtomArray:
        tokatom_idx_list = []
        for atom in atom_array:
            atom_name_position = RES_ATOMS_DICT.get(atom.res_name, None)
            if atom.mol_type == "ligand" or atom_name_position is None:
                tokatom_idx = 0
            else:
                tokatom_idx = atom_name_position[atom.atom_name]
            tokatom_idx_list.append(tokatom_idx)
        atom_array.set_annotation("tokatom_idx", tokatom_idx_list)
        return atom_array

    @staticmethod
    def unique_chain_and_add_ids(atom_array: AtomArray) -> AtomArray:
        chain_ids = np.zeros(len(atom_array), dtype="<U16")
        chain_starts = struc.get_chain_starts(
            atom_array, add_exclusive_stop=True
        )

        chain_counter = Counter()
        for start, stop in zip(chain_starts[:-1], chain_starts[1:]):
            original_chain_id = atom_array.chain_id[start]
            count = chain_counter[original_chain_id]
            if count == 0:
                new_chain_id = original_chain_id
            else:
                new_chain_id = f"{original_chain_id}.{count}"
            chain_ids[start:stop] = new_chain_id
            chain_counter[original_chain_id] += 1

        assert "" not in chain_ids
        atom_array.del_annotation("chain_id")
        atom_array.set_annotation("chain_id", chain_ids)

        entity_id_unique = np.sort(np.unique(atom_array.label_entity_id))
        entity_id_dict = {
            entity: index for index, entity in enumerate(entity_id_unique)
        }
        asym_ids = np.zeros(len(atom_array), dtype=int)
        entity_ids = np.zeros(len(atom_array), dtype=int)
        sym_ids = np.zeros(len(atom_array), dtype=int)
        entity_counter = Counter()
        start_indices = struc.get_chain_starts(
            atom_array, add_exclusive_stop=True
        )
        for index in range(len(start_indices) - 1):
            start = start_indices[index]
            stop = start_indices[index + 1]
            asym_ids[start:stop] = index
            entity_id = atom_array.label_entity_id[start]
            entity_ids[start:stop] = entity_id_dict[entity_id]
            sym_ids[start:stop] = entity_counter[entity_id]
            entity_counter[entity_id] += 1

        atom_array.set_annotation("asym_id_int", asym_ids)
        atom_array.set_annotation("entity_id_int", entity_ids)
        atom_array.set_annotation("sym_id_int", sym_ids)
        return atom_array

    @staticmethod
    def remove_bonds_between_polymer_chains(
        atom_array: AtomArray, entity_poly_type: dict[str, str]
    ) -> struc.BondList:
        """Return a bond copy without inter-chain polymer bonds."""

        bonds = atom_array.bonds.copy()
        polymer_mask = np.isin(
            atom_array.label_entity_id, list(entity_poly_type.keys())
        )
        bond_array = bonds._bonds
        atom_i = bond_array[:, 0]
        atom_j = bond_array[:, 1]
        polymer_polymer = polymer_mask[atom_i] & polymer_mask[atom_j]
        different_chain = atom_array.chain_id[atom_i] != atom_array.chain_id[atom_j]
        remove = polymer_polymer & different_chain
        bonds._bonds = bond_array[~remove]
        bonds._remove_redundant_bonds()
        bonds._max_bonds_per_atom = bonds._get_max_bonds_per_atom()
        return bonds

    @staticmethod
    def find_equiv_mol_and_assign_ids(
        atom_array: AtomArray,
        entity_poly_type: Optional[dict[str, str]] = None,
        pdb_id: Optional[str] = None,
    ) -> AtomArray:
        if hasattr(atom_array, "is_resolved"):
            valid_mask = atom_array.is_resolved
        else:
            valid_mask = np.ones(len(atom_array), dtype=bool)

        if entity_poly_type is None:
            molecule_indices = get_molecule_indices(atom_array[valid_mask])
        else:
            bonds_filtered = AddAtomArrayAnnot.remove_bonds_between_polymer_chains(
                atom_array[valid_mask], entity_poly_type
            )
            molecule_indices = get_molecule_indices(bonds_filtered)

        mol_ids = np.array([-1] * len(atom_array), dtype=int)
        chain_names = np.unique(atom_array.chain_id)
        parent = {chain: chain for chain in chain_names}

        def find(chain):
            root = chain
            while parent[root] != root:
                parent[root] = parent[parent[root]]
                root = parent[root]
            return root

        def union(left, right):
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        valid_chain_ids = atom_array.chain_id[valid_mask]
        for atom_indices in molecule_indices:
            chain_ids = np.unique(valid_chain_ids[atom_indices])
            for left, right in zip(chain_ids[:-1], chain_ids[1:]):
                union(left, right)

        component_ids = {}
        for chain in chain_names:
            root = find(chain)
            if root not in component_ids:
                component_ids[root] = len(component_ids)
            mol_ids[atom_array.chain_id == chain] = component_ids[root]

        atom_array.set_annotation("mol_id", mol_ids)
        assert ~np.isin(-1, atom_array.mol_id), "Some mol_id is not assigned."

        mol_id_to_atom_name = {}
        entity_mol_dict = defaultdict(list)
        for mol_id in np.unique(atom_array.mol_id):
            mol_mask = atom_array.mol_id == mol_id
            _, chain_starts = np.unique(
                atom_array.chain_id[mol_mask], return_index=True
            )
            entity_ids = atom_array.label_entity_id[mol_mask][
                chain_starts
            ].tolist()
            entity_mol_dict[tuple(sorted(entity_ids))].append(mol_id)
            mol_id_to_atom_name[mol_id] = atom_array.atom_name[mol_mask]

        entity_mol_id_num = 0
        mol_id_to_entity_mol_ids = {}
        for molecule_ids in entity_mol_dict.values():
            checked_mol_id = []
            for mol_id in molecule_ids:
                molecule_atom_names = mol_id_to_atom_name[mol_id]
                if checked_mol_id:
                    for reference_mol_id in checked_mol_id:
                        reference_atom_names = mol_id_to_atom_name[
                            reference_mol_id
                        ]
                        if len(molecule_atom_names) == len(
                            reference_atom_names
                        ):
                            if np.all(
                                reference_atom_names == molecule_atom_names
                            ):
                                mol_id_to_entity_mol_ids[mol_id] = (
                                    mol_id_to_entity_mol_ids[
                                        reference_mol_id
                                    ]
                                )
                                break
                            logging.warning(
                                "Two mols have same entity_ids, but diff atom name: "
                                "ref_atom_name=%s atom_name=%s",
                                reference_atom_names[:5],
                                molecule_atom_names[:5],
                            )
                            continue
                        logging.warning(
                            "Two mols have same entity_ids, but diff atom num: "
                            "ref_atom_num=%s atom_num=%s",
                            len(reference_atom_names),
                            len(molecule_atom_names),
                        )
                    else:
                        entity_mol_id_num += 1
                        mol_id_to_entity_mol_ids[mol_id] = entity_mol_id_num
                else:
                    mol_id_to_entity_mol_ids[mol_id] = entity_mol_id_num
                checked_mol_id.append(mol_id)
            entity_mol_id_num += 1

        entity_mol_ids = np.array(
            [
                mol_id_to_entity_mol_ids[mol_id]
                for mol_id in atom_array.mol_id
            ],
            dtype=np.int32,
        )
        atom_array.set_annotation("entity_mol_id", entity_mol_ids)

        unique, inverse = np.unique(
            atom_array.mol_id, return_inverse=True
        )
        counts = np.bincount(inverse)
        mol_atom_index = np.zeros_like(atom_array.mol_id)
        for index in range(len(unique)):
            mol_atom_index[inverse == index] = np.arange(counts[index])
        atom_array.set_annotation("mol_atom_index", mol_atom_index)
        return atom_array


# Compatibility name used by the narrow scoring feature builder. This is a
# local alias, not a Protenix namespace alias.
Annot = AddAtomArrayAnnot


__all__ = [
    "Annot",
    "AddAtomArrayAnnot",
    "AtomArrayTokenizer",
    "ELEMS",
    "RES_ATOMS_DICT",
    "STD_RESIDUES",
    "Token",
    "TokenArray",
]
