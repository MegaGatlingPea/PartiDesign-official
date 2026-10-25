"""Direct structure state used by selection and later resume checks."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any

from .storage import _plain_path


def _primitive(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_primitive(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Structure state contains a non-finite value")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"Unsupported structure state value: {type(value).__name__}")



def _pdb_coordinates(path, atoms):
    rows = defaultdict(list)
    first_model, active = None, True
    for line in path.read_text(encoding="utf-8").splitlines():
        record = line[:6].strip().upper()
        if record == "MODEL":
            model = line[10:14].strip() or "1"
            if first_model is None:
                first_model = model
            active = model == first_model
        elif record == "ENDMDL":
            active = False
        elif record in {"ATOM", "HETATM"} and active:
            alt = line[16:17].strip()
            key = (line[21:22].strip(), line[22:26].strip(), line[26:27].strip(),
                   line[17:20].strip().upper(), line[12:16].strip(), "" if alt in {".", "?"} else alt)
            rows[key].append((int(line[6:11]), [float(line[30:38]), float(line[38:46]), float(line[46:54])]))
    for values in rows.values():
        values.sort(key=lambda item: item[0])
    fields = ("auth_asym_id", "auth_seq_id", "auth_ins_code", "res_name", "atom_name", "label_alt_id")
    coordinates = []
    for index in range(len(atoms)):
        key = tuple(str(atoms.get_annotation(field)[index]) for field in fields)
        if not rows.get(key):
            raise ValueError("Source PDB atom rows disagree with the controlled reader")
        coordinates.append(rows[key].pop(0)[1])
    return coordinates


def snapshot_refold(path: str | Path) -> dict:
    """Save atom annotations, source coordinates and the complete reader graph."""
    from src.design.utils.reader import (
        _read_cif_rows, load_ccd_standard_parents, read_structure,
    )
    import numpy as np

    path = _plain_path(path)
    if path.suffix.lower() not in {".cif", ".pdb"}:
        raise ValueError(f"Refold must be a CIF or PDB structure: {path}")
    atoms = read_structure(path, canonical_polymer_order=True)
    coordinates = np.asarray(atoms.coord, dtype=np.float64)
    if path.suffix.lower() == ".cif":
        _, _, rows, _ = _read_cif_rows(
            path, parent_sources=load_ccd_standard_parents(path), canonical_polymer_order=True,
        )
        kept = [row for row in rows if row["res_name"].upper() not in {"HOH", "WAT", "DOD", "H2O"}
                and row["element"].upper() not in {"H", "D", "T"}]
        exact = np.asarray([row["coord"] for row in kept], dtype=np.float64)
        if exact.shape != coordinates.shape or not np.array_equal(exact.astype(np.float32), atoms.coord):
            raise ValueError("Source coordinate rows disagree with the controlled reader")
        coordinates = exact
    else:
        coordinates = np.asarray(_pdb_coordinates(path, atoms), dtype=np.float64)
        if not np.array_equal(coordinates.astype(np.float32), atoms.coord):
            raise ValueError("Source PDB coordinates disagree with the controlled reader")
    if coordinates.shape != (len(atoms), 3) or not np.isfinite(coordinates).all():
        raise ValueError("Refold coordinates are empty, malformed or non-finite")
    annotations = {
        name: _primitive(atoms.get_annotation(name))
        for name in sorted(atoms.get_annotation_categories())
    }
    graph = [] if atoms.bonds is None else [
        [min(int(left), int(right)), max(int(left), int(right)), int(order)]
        for left, right, order in atoms.bonds.as_array().tolist()
    ]
    return {"schema_version": 1, "annotations": annotations,
            "coordinates": coordinates.tolist(), "bonds": sorted(graph)}
