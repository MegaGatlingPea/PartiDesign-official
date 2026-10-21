"""Atom-level condition categories for structure coloring.

CIF B_iso_or_equiv values written by PartiDesign represent condition categories,
not experimental displacement parameters or model confidence.
"""

from __future__ import annotations

import numpy as np
from biotite import structure as struc


CONDITION_B_FACTOR_VALUES = {
    "context": 0.0,
    "free": 20.0,
    "identity": 40.0,
    "token_center": 60.0,
    "motif": 80.0,
    "hotspot": 100.0,
}


def condition_b_factors(atom_array: struc.AtomArray) -> np.ndarray:
    """Return one condition-category scalar per atom without changing the array.

    Design atoms use the precedence free < identity < token center < atom
    motif. Context atoms remain context even when their generic condition flags
    are set; hotspots override every other category on either role. Masks stay
    atom-level, so a partial motif does not color its unselected residue atoms.

    An explicitly attached b_factor annotation overrides this calculation.
    Preview callers may attach it after canonical validation; the strict atom
    schema and structure reader deliberately do not carry source B factors.
    """

    categories = set(atom_array.get_annotation_categories())
    if "b_factor" in categories:
        values = np.asarray(atom_array.get_annotation("b_factor"), dtype=np.float64)
        if values.shape != (len(atom_array),) or not np.isfinite(values).all():
            raise ValueError("b_factor annotation must be finite float[A]")
        return values.copy()

    def flag(name: str) -> np.ndarray:
        if name in categories:
            return np.asarray(atom_array.get_annotation(name), dtype=bool)
        return np.zeros(len(atom_array), dtype=bool)

    design = flag("is_design")
    values = np.full(len(atom_array), CONDITION_B_FACTOR_VALUES["context"])
    values[design] = CONDITION_B_FACTOR_VALUES["free"]
    for annotation, category in (
        ("identity_condition", "identity"),
        ("token_center_condition", "token_center"),
        ("motif_coord_condition", "motif"),
    ):
        values[design & flag(annotation)] = CONDITION_B_FACTOR_VALUES[category]
    values[flag("hotspot_condition")] = CONDITION_B_FACTOR_VALUES["hotspot"]
    return values


__all__ = ["CONDITION_B_FACTOR_VALUES", "condition_b_factors"]
