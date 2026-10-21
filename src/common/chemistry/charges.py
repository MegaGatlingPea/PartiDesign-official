"""Explicit per-atom formal charge facts, with unknown kept distinct from zero."""

from __future__ import annotations

import re

import numpy as np


_ALIASES = ("formal_charge", "charge", "pdbx_formal_charge", "atom_charge")
_MISSING = frozenset({"", ".", "?", "none", "nan", "unknown", "unspecified"})
_INTEGER = re.compile(r"[+-]?[0-9]+")
_PDB_INTEGER = re.compile(r"([0-9]+)([+-])")


def parse_formal_charge(value: object, where: str = "") -> int | None:
    """Parse an exact charge, accepting numeric runtime missing-value conventions."""
    if value is None:
        return None
    location = f" at {where}" if where else ""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"invalid formal charge{location}: {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        if np.isfinite(value) and value == np.trunc(value):
            return int(value)
        raise ValueError(f"non-integer formal charge{location}: {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _MISSING:
            return None
        if _INTEGER.fullmatch(text):
            return int(text)
        match = _PDB_INTEGER.fullmatch(text)
        if match:
            return int(match[1]) * (1 if match[2] == "+" else -1)
    raise ValueError(f"invalid formal charge{location}: {value!r}")


def formal_charge_values(atoms: object, rows=None) -> np.ndarray | None:
    """Merge present charge aliases on selected rows without inventing charges.

    Unknown values in one column never hide an explicit value in another.
    Incompatible explicit aliases are input errors.  ``None`` as the return
    value means that the structure has no charge annotation at all.
    """
    category_getter = getattr(atoms, "get_annotation_categories", None)
    categories = set() if category_getter is None else set(category_getter())
    aliases = [name for name in _ALIASES if name in categories or hasattr(atoms, name)]
    if not aliases:
        return None
    try:
        length = len(atoms)
    except TypeError:
        reference = getattr(atoms, "element", None)
        if reference is None:
            reference = getattr(atoms, "coord", None)
        length = None if reference is None else len(reference)
    getter = getattr(atoms, "get_annotation", None)
    result = None
    for alias in aliases:
        raw = np.asarray(getter(alias) if getter and alias in categories else getattr(atoms, alias))
        if length is None:
            length = len(raw) if raw.ndim else 0
        if raw.shape != (length,):
            raise ValueError(f"formal charge annotation {alias} has shape {raw.shape}, expected {(length,)}")
        selected = np.atleast_1d(raw if rows is None else raw[rows])
        if selected.ndim != 1:
            raise ValueError("formal charge rows must select a one-dimensional atom list")
        if result is None:
            result = np.full(len(selected), None, dtype=object)
        elif len(selected) != len(result):
            raise ValueError("formal charge aliases select inconsistent atom counts")
        for index, value in enumerate(selected):
            if value is None:
                continue
            try:
                charge = parse_formal_charge(value)
            except ValueError as exc:
                raise ValueError(f"{exc} at {alias} row {index}") from exc
            if charge is None:
                continue
            previous = result[index]
            if previous is not None and previous != charge:
                raise ValueError(f"conflicting formal charge aliases at row {index}: {previous} versus {charge}")
            result[index] = charge
    return result


__all__ = ["parse_formal_charge", "formal_charge_values"]
