"""Small shared metadata for the existing native refold implementations."""
from __future__ import annotations

from typing import NamedTuple


class RefoldMethod(NamedTuple):
    group: str
    family: str
    msa: bool
    bond_policy: str
    worker_gpus: float


REFOLD_METHODS = {
    "esmfold2-fast": RefoldMethod("esmfold2", "esmfold2", False, "explicit", 1.0),
    "esmfold2-standard": RefoldMethod("esmfold2", "esmfold2", True, "explicit", 1.0),
    "af2_ig": RefoldMethod("af2", "af2", False, "native_af2", 0.25),
    "af3": RefoldMethod("af3", "af3", True, "omit_polymer_polymer", 0.5),
    "af3-la": RefoldMethod("af3", "af3", True, "omit_polymer_polymer", 0.5),
    "mol1": RefoldMethod("mol1", "mol1", True, "native_mol1", 1.0),
    "mol1-constraint": RefoldMethod("mol1_constraint", "mol1", True, "native_mol1", 1.0),
    "boltz2": RefoldMethod("boltz2", "boltz2", True, "native_boltz2", 1.0),
}

VALIDATION_REFOLD_METHODS = dict(REFOLD_METHODS)
