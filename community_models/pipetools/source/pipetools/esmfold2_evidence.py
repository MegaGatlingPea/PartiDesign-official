from __future__ import annotations

from typing import Any, Mapping

import numpy as np


# Every tensor produced by the released ESMFold2 input builder is retained.
# Some fields currently pass through ``**kwargs`` rather than changing the
# forward result, but keeping the complete prepared input closes that ambiguity
# when upstream code evolves and makes the evidence independently replayable.
PREPARED_FEATURE_NAMES = (
    "token_index",
    "residue_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "mol_type",
    "res_type",
    "input_ids",
    "token_bonds",
    "token_attention_mask",
    "pocket_feature",
    "ref_pos",
    "ref_element",
    "ref_charge",
    "ref_atom_name_chars",
    "ref_space_uid",
    "gt_coords",
    "atom_attention_mask",
    "atom_to_token",
    "is_resolved",
    "distogram_atom_idx",
    "frames_idx",
    "disto_cond",
    "disto_cond_mask",
    "msa",
    "deletion_value",
    "has_deletion",
    "deletion_mean",
    "msa_attention_mask",
)

# These records are not model inputs.  They bind the only model-specific atom
# inventory exception to explicit CCD leaving-atom flags.  Indices are
# zero-based and atom names use the same four-character vocabulary as the
# prepared ESMFold2 features.
OMISSION_FEATURE_NAMES = (
    "omission_chain_index",
    "omission_residue_index",
    "omission_atom_name_chars",
    "omission_atomic_number",
)

EVIDENCE_FEATURE_NAMES = PREPARED_FEATURE_NAMES + OMISSION_FEATURE_NAMES

_ELEMENT_SYMBOLS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
    "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
_ATOMIC_NUMBER = {symbol.upper(): index for index, symbol in enumerate(_ELEMENT_SYMBOLS, 1)}


def decode_atom_name_rows(values: Any) -> tuple[str, ...]:
    """Decode the upstream 0-padded, ASCII-minus-32 atom-name encoding."""

    rows = np.asarray(values)
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError("ESMFold2 atom-name feature must have shape [N, 4]")
    result: list[str] = []
    for row in rows:
        codes = [int(value) for value in row]
        if any(code < 0 or code > 94 for code in codes):
            raise ValueError("ESMFold2 atom-name feature has an invalid code")
        seen_padding = False
        characters: list[str] = []
        for code in codes:
            if code == 0:
                seen_padding = True
                continue
            if seen_padding:
                raise ValueError("ESMFold2 atom-name padding is non-terminal")
            characters.append(chr(code + 32))
        name = "".join(characters)
        if not name:
            raise ValueError("ESMFold2 valid atom has an empty name")
        result.append(name)
    return tuple(result)


def encode_atom_name_rows(names: list[str]) -> np.ndarray:
    """Encode omission-ledger atom names with the upstream four-char codec."""

    result = np.zeros((len(names), 4), dtype=np.int64)
    for row, name in enumerate(names):
        if not isinstance(name, str) or not name or len(name) > 4:
            raise ValueError(f"CCD leaving atom name is not encodable: {name!r}")
        codes = [ord(character) - 32 for character in name]
        if any(code < 1 or code > 94 for code in codes):
            raise ValueError(f"CCD leaving atom name is not ASCII-safe: {name!r}")
        result[row, : len(codes)] = codes
    return result


def atomic_number(type_symbol: str) -> int:
    if not isinstance(type_symbol, str):
        raise ValueError("CCD atom type symbol is malformed")
    value = _ATOMIC_NUMBER.get(type_symbol.strip().upper())
    if value is None:
        raise ValueError(f"CCD atom type symbol is unknown: {type_symbol!r}")
    return value


__all__ = [
    "EVIDENCE_FEATURE_NAMES",
    "OMISSION_FEATURE_NAMES",
    "PREPARED_FEATURE_NAMES",
    "atomic_number",
    "decode_atom_name_rows",
    "encode_atom_name_rows",
]
