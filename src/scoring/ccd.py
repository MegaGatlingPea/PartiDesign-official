"""Small offline CCD lookup used by inference-only PartiScore preparation."""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

# Canonical inference residue alphabets; independent of external parsers.
PROT_STD_RESIDUES_ONE_TO_THREE = dict(zip(
    "ARNDCQEGHILKMFPSTWYVX",
    ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
     "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK"),
    strict=True))
RNA_STD_RESIDUES = frozenset(("A", "G", "C", "U", "N"))
DNA_STD_RESIDUES = frozenset(("DA", "DG", "DC", "DT", "DN"))
from partidesign.paths import resource_root
from src.common.chemistry.ccd import CCDResourceError, ccd_one_letter_codes


_COMPACT_ONE_LETTER_PATH = resource_root() / "configs" / "resources" / "chemistry" / "ccd.json.gz"


@lru_cache(maxsize=1)
def compact_one_letter_codes() -> Mapping[str, str | None]:
    """Load one-letter metadata from the unified resource.

    An explicitly selected legacy schema-1 file remains readable for transition
    tooling; the packaged default is always the merged schema-4 resource.
    """
    path = _COMPACT_ONE_LETTER_PATH
    if not path.is_file():
        raise RuntimeError(f"compact CCD one-letter asset is missing: {path}")
    try:
        if path.name == "ccd.json.gz":
            values = ccd_one_letter_codes(path)
        else:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise ValueError("schema")
            values = payload.get("one_letter_codes")
            if not isinstance(values, dict):
                raise ValueError("table")
    except (CCDResourceError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid compact CCD one-letter schema: {path}") from exc
    result: dict[str, str | None] = {}
    for raw_name, raw_code in values.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise RuntimeError(f"invalid compact CCD component code: {path}")
        if raw_code is not None and not isinstance(raw_code, str):
            raise RuntimeError(f"invalid compact CCD one-letter value: {path}")
        result[raw_name] = raw_code
    return MappingProxyType(result)


def canonical_residue_name(res_name: object, mol_type: object) -> str:
    """Match Protenix's canonical residue projection without fullcomponents.cif."""

    name = str(res_name)
    modality = str(mol_type)
    one_letter = compact_one_letter_codes().get(name)
    if one_letter == "?":
        one_letter = None
    if one_letter is None or len(one_letter) != 1:
        one_letter = "X" if modality == "protein" else "N"

    if modality == "protein":
        return PROT_STD_RESIDUES_ONE_TO_THREE.get(one_letter, "UNK")
    if modality == "dna":
        result = "D" + one_letter
        return result if result in DNA_STD_RESIDUES else "DN"
    if modality == "rna":
        return one_letter if one_letter in RNA_STD_RESIDUES else "N"
    return "UNK"


def canonical_residue_names(atom_array) -> list[str]:
    """Return one canonical name per atom, preserving residue boundaries."""

    import biotite.structure as struc

    result = [""] * len(atom_array)
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(starts[:-1], starts[1:], strict=True):
        name = canonical_residue_name(
            atom_array.res_name[start], atom_array.mol_type[start]
        )
        result[int(start) : int(stop)] = [name] * (int(stop) - int(start))
    return result


__all__ = [
    "canonical_residue_name",
    "canonical_residue_names",
    "compact_one_letter_codes",
]
