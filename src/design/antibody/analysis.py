"""Pure antibody-candidate analysis and offline sequence filtering.

This module deliberately has no model, checkpoint, or runtime imports.  It
operates on an assembled :class:`biotite.structure.AtomArray` (or an object
with an ``atoms`` attribute) and on the coordinate-free antibody metadata
emitted by the compiler.  Coordinates in the standalone authoring IR are
Angstroms; compiler template coordinates are the canonical ``*_nm`` values and
are converted only while calculating template RMSD.

The public entry point is :func:`analyze_antibody`.  The small filtering and
diversity helpers are used by the offline CLI and are intentionally usable by
the runtime without importing that CLI.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

try:  # biotite is part of the standalone inference environment.
    import biotite.structure as struc
except ImportError:  # pragma: no cover - lets ``--help`` work in a minimal env.
    struc = None  # type: ignore[assignment]


# The compiler uses the following stable IDs.  Role 1 is shared by a heavy
# chain and a VHH; the label is preserved separately when the compiler gives us
# the textual role in metadata or in the task spec.
ROLE_CONTEXT = 0
ROLE_HEAVY = 1
ROLE_VHH = 1
ROLE_LIGHT = 2

REGION_CONTEXT = 0
REGION_FR1 = 1
REGION_CDR1 = 2
REGION_CDR2 = 3
REGION_CDR3 = 4

ROLE_NAMES = {ROLE_CONTEXT: "context", ROLE_HEAVY: "H", ROLE_LIGHT: "L"}
REGION_NAMES = {
    REGION_CONTEXT: "context",
    REGION_FR1: "FR1",
    REGION_CDR1: "CDR1",
    REGION_CDR2: "CDR2",
    REGION_CDR3: "CDR3",
}
CDR_IDS = (REGION_CDR1, REGION_CDR2, REGION_CDR3)
BACKBONE_ATOMS = frozenset(("N", "CA", "C", "O"))

# A fixed 21-symbol catalog: the twenty standard amino acids plus unknown.
# It is deliberately independent of a model vocabulary and never invents an
# IMGT number or a residue identity from a missing source label.
AMINO_ACID_CATALOG = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "UNK": "X",
}
AMINO_ACID_ONE_TO_THREE = {value: key for key, value in AMINO_ACID_CATALOG.items()}
_ONE_LETTER = frozenset(AMINO_ACID_ONE_TO_THREE)

_ANNOTATION_NAMES = (
    "antibody_chain_role_id",
    "antibody_region_id",
    "antibody_sequence_mutable",
    "antibody_source_label_seq_id",
)

_VDW_RADII_ANGSTROM = {
    "H": 1.20,
    "B": 1.92,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "FE": 1.80,
    "MG": 1.73,
    "CA": 2.31,
    "ZN": 1.39,
}


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "" or value == "." or value == "?":
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "" or value == "." or value == "?":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0", ""}:
            return False
    return bool(value)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _json_safe(value: Any) -> Any:
    """Convert a report tree to strict JSON-compatible Python values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _is_atom_array(value: Any) -> bool:
    return struc is not None and isinstance(value, struc.AtomArray)


def _extract_atoms(candidate: Any) -> Any:
    if _is_atom_array(candidate):
        return candidate
    for key in ("atoms", "structure", "atom_array"):
        value = _value(candidate, key)
        if _is_atom_array(value):
            return value
    raise TypeError("antibody analysis requires an AtomArray or candidate.atoms")


def _annotation_names(atoms: Any) -> set[str]:
    getter = getattr(atoms, "get_annotation_categories", None)
    if getter is None:
        return set()
    return set(getter())


def _annotation(atoms: Any, name: str, default: Any) -> np.ndarray:
    names = _annotation_names(atoms)
    if name in names:
        return np.asarray(atoms.get_annotation(name))
    return np.full(len(atoms), default)


def _validate_antibody_annotations(atoms: Any) -> bool:
    present = _annotation_names(atoms).intersection(_ANNOTATION_NAMES)
    if not present:
        return False
    missing = sorted(set(_ANNOTATION_NAMES).difference(present))
    if missing:
        raise ValueError(
            "antibody annotations are all-or-none; missing " + ", ".join(missing)
        )
    for name, allowed in (("antibody_chain_role_id", (0, 1, 2)),
                          ("antibody_region_id", (0, 1, 2, 3, 4))):
        values = np.asarray(atoms.get_annotation(name))
        if values.shape != (len(atoms),) or not np.isin(values, allowed).all():
            raise ValueError(f"invalid {name}; expected IDs in {allowed}")
    return True


def _normal_role(value: Any, chain_id: str = "") -> tuple[int, str]:
    text = _as_text(value).strip().upper()
    if text in {"H", "VH", "HEAVY", "HEAVY_CHAIN", "VARIABLE_HEAVY"}:
        return ROLE_HEAVY, "H"
    if text in {"VHH", "NANOBODY", "HEAVY_ONLY"}:
        return ROLE_VHH, "VHH"
    if text in {"L", "VL", "LIGHT", "LIGHT_CHAIN", "VARIABLE_LIGHT"}:
        return ROLE_LIGHT, "L"
    if text in {"CONTEXT", "ANTIGEN", "TARGET", ""}:
        return ROLE_CONTEXT, "context"
    role_id = _as_int(value, ROLE_CONTEXT)
    if role_id == ROLE_LIGHT:
        return role_id, "L"
    if role_id == ROLE_HEAVY:
        # Chain metadata can disambiguate H from VHH.  Do not guess from an
        # IMGT numbering scheme; only a literal VHH chain cue is accepted.
        return role_id, "VHH" if "VHH" in chain_id.upper() else "H"
    return ROLE_CONTEXT, "context"


def _normal_region(value: Any) -> tuple[int, str]:
    text = _as_text(value).strip().upper()
    aliases = {
        "FR": REGION_FR1,
        "FRAMEWORK": REGION_FR1,
        "FR1": REGION_FR1,
        "CDR1": REGION_CDR1,
        "CDR2": REGION_CDR2,
        "CDR3": REGION_CDR3,
        "CONTEXT": REGION_CONTEXT,
        "": REGION_CONTEXT,
    }
    region_id = aliases.get(text, _as_int(value, REGION_CONTEXT))
    return region_id, REGION_NAMES.get(region_id, "context")


def _one_letter(res_name: Any) -> str:
    text = _as_text(res_name).strip().upper()
    if text in _ONE_LETTER:
        return text
    return AMINO_ACID_CATALOG.get(text, "X")


def _source_entries_from(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (str, bytes, Mapping)) and hasattr(value, "chain_id"):
        try:
            return [vars(value)]
        except TypeError:
            return []
    if isinstance(value, Mapping):
        for key in ("entries", "antibody_metadata", "source_ledger"):
            nested = value.get(key)
            if nested is not None:
                return _source_entries_from(nested)
        if "chain_id" in value and any(
            key in value for key in ("position", "res_id", "label_seq_id")
        ):
            return [value]
        # A mapping keyed by chain/position is accepted as a convenience.
        if all(isinstance(item, Mapping) for item in value.values()):
            return [item for item in value.values()]
        return []
    if isinstance(value, (str, bytes)):
        return []
    try:
        result: list[Mapping[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                result.append(item)
            elif hasattr(item, "chain_id"):
                try:
                    result.append(vars(item))
                except TypeError:
                    continue
        return result
    except TypeError:
        return []


def _find_metadata(candidate: Any, explicit: Any) -> list[Mapping[str, Any]]:
    direct = _source_entries_from(explicit)
    if direct:
        return direct
    seen: set[int] = set()
    queue = [candidate]
    entries: list[Mapping[str, Any]] = []
    while queue:
        obj = queue.pop(0)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        for key in ("antibody_metadata", "compiled", "base_request", "task", "antibody"):
            value = _value(obj, key)
            if value is None:
                continue
            found = _source_entries_from(value)
            if found:
                entries.extend(found)
            elif not isinstance(value, (str, bytes, int, float, bool)):
                queue.append(value)
        direct_value = _value(obj, "antibody_metadata")
        if direct_value is not None:
            entries.extend(_source_entries_from(direct_value))
    # Stable deduplication preserves compiler order without relying on dict
    # insertion order from an external report.
    result: list[Mapping[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    for entry in entries:
        key = (
            _as_text(entry.get("chain_id")),
            _as_int(entry.get("position", entry.get("res_id", 0))),
            _as_text(entry.get("source_label_seq_id")),
            _as_text(entry.get("region_id")),
        )
        if key not in seen_keys:
            seen_keys.add(key)
            result.append(entry)
    return result


def _metadata_index(entries: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    index: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in entries:
        chain = _as_text(entry.get("chain_id", entry.get("output_chain_id", "")))
        raw_position = entry.get("position", entry.get("res_id", entry.get("label_seq_id")))
        position = _as_optional_int(raw_position)
        if not chain or position is None:
            continue
        index.setdefault((chain, position), entry)
    return index


def _task_chain_roles(candidate: Any) -> dict[str, tuple[int, str]]:
    """Extract role/output IDs from the parsed antibody request if present."""

    result: dict[str, tuple[int, str]] = {}
    queue = [candidate]
    seen: set[int] = set()
    while queue:
        obj = queue.pop(0)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        antibody = _value(obj, "antibody")
        if antibody is not None:
            chains = _value(antibody, "chains")
            if isinstance(chains, Mapping):
                chain_values = list(chains.values())
                chain_items = list(chains.items())
            elif isinstance(chains, Sequence) and not isinstance(chains, (str, bytes)):
                chain_values = list(chains)
                chain_items = [(None, value) for value in chain_values]
            else:
                chain_values = []
                chain_items = []
            for key, item in chain_items:
                if isinstance(item, str):
                    raw_role = item
                    item = {"role": raw_role}
                if isinstance(item, Mapping):
                    raw_role = item.get("role", item.get("chain_role", item.get("role_id", "context")))
                    get_id = item.get
                else:
                    raw_role = _value(item, "role", _value(item, "chain_role", _value(item, "role_id", "context")))
                    get_id = lambda name, default=None: _value(item, name, default)
                role_id, role_name = _normal_role(raw_role, _as_text(key))
                IDs = (
                    get_id("output_id"),
                    get_id("output_chain_id"),
                    get_id("chain_id"),
                    get_id("id"),
                    key,
                )
                for chain_id in IDs:
                    if chain_id is not None and _as_text(chain_id):
                        result[_as_text(chain_id)] = (role_id, role_name)
        for key in ("base_request", "task", "request"):
            nested = _value(obj, key)
            if nested is not None and not isinstance(nested, (str, bytes, int, float, bool)):
                queue.append(nested)
    return result


def _coordinate_valid_mask(atoms: Any) -> np.ndarray:
    coords = np.asarray(atoms.coord, dtype=np.float64)
    if coords.shape != (len(atoms), 3):
        raise ValueError("antibody coordinates must have shape [atom,3]")
    finite = np.isfinite(coords).all(axis=1)
    if "coord_valid" in _annotation_names(atoms):
        finite &= np.asarray(atoms.coord_valid, dtype=bool)
    return finite


@dataclass
class _Residue:
    chain_id: str
    position: int
    ins_code: str
    rows: tuple[int, ...]
    role_id: int
    role: str
    region_id: int
    region: str
    mutable: bool
    amino_acid: str
    source_chain: str
    source_label_seq_id: int
    source_auth_seq_id: str
    source_ins_code: str
    source_amino_acid: str
    source_present: bool
    template_coords_nm: np.ndarray | None
    template_mask: np.ndarray | None


def _residue_rows(atoms: Any) -> list[tuple[str, int, str, tuple[int, ...]]]:
    chains = _annotation(atoms, "chain_id", "")
    positions = _annotation(atoms, "res_id", 0)
    ins_codes = _annotation(atoms, "ins_code", "")
    groups: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for row in range(len(atoms)):
        key = (_as_text(chains[row]), _as_int(positions[row]), _as_text(ins_codes[row]))
        groups[key].append(row)
    return [(*key, tuple(rows)) for key, rows in sorted(groups.items(), key=lambda item: item[0])]


def _template_array(entry: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray | None:
    value = entry.get(key)
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != shape:
        return None
    return array


def _build_residues(
    atoms: Any,
    entries: Sequence[Mapping[str, Any]],
    candidate: Any,
) -> list[_Residue]:
    has_annotations = _validate_antibody_annotations(atoms)
    annotation_chains = _annotation(atoms, "chain_id", "")
    role_values = _annotation(atoms, "antibody_chain_role_id", 0)
    region_values = _annotation(atoms, "antibody_region_id", 0)
    mutable_values = _annotation(atoms, "antibody_sequence_mutable", False)
    source_label_values = _annotation(atoms, "antibody_source_label_seq_id", -1)
    res_names = _annotation(atoms, "res_name", "UNK")
    metadata = _metadata_index(entries)
    task_roles = _task_chain_roles(candidate)
    design_values = _annotation(atoms, "is_design", False)
    residues: list[_Residue] = []
    for chain_id, position, ins_code, rows in _residue_rows(atoms):
        entry = metadata.get((chain_id, position), {})
        if has_annotations:
            role_id, role_name = _normal_role(int(role_values[rows[0]]), chain_id)
            task_role = task_roles.get(chain_id)
            if task_role is not None and task_role[0] == role_id:
                role_name = task_role[1]
            region_id, region_name = _normal_region(int(region_values[rows[0]]))
            mutable = bool(np.asarray(mutable_values[list(rows)], dtype=bool).any())
            source_label = _as_int(source_label_values[rows[0]], -1)
        else:
            role_value = entry.get("role", entry.get("chain_role", entry.get("role_id")))
            if role_value is None:
                role_id, role_name = task_roles.get(chain_id, (ROLE_CONTEXT, "context"))
                if chain_id not in task_roles:
                    chain_hint = chain_id.strip().upper()
                    if chain_hint in {"H", "HEAVY", "HEAVY_CHAIN"}:
                        role_id, role_name = ROLE_HEAVY, "H"
                    elif chain_hint in {"L", "LIGHT", "LIGHT_CHAIN"}:
                        role_id, role_name = ROLE_LIGHT, "L"
                    elif "VHH" in chain_hint:
                        role_id, role_name = ROLE_VHH, "VHH"
                    elif bool(np.asarray(design_values[list(rows)], dtype=bool).any()):
                        # A bare explicit AtomArray can still describe one
                        # designed protein chain.  With no role metadata its
                        # heavy/light distinction is necessarily unavailable.
                        role_id, role_name = ROLE_HEAVY, "H"
            else:
                role_id, role_name = _normal_role(role_value, chain_id)
            region_id, region_name = _normal_region(entry.get("region", entry.get("region_id", 0)))
            mutable = _as_bool(entry.get("mutable", entry.get("sequence_mutable", False)))
            source_label = _as_int(entry.get("source_label_seq_id", -1), -1)
        if chain_id in task_roles and not has_annotations and not entry:
            role_id, role_name = task_roles[chain_id]
        source_chain = _as_text(entry.get("source_chain", chain_id))
        source_auth = _as_text(entry.get("source_auth_seq_id", entry.get("auth_seq_id", "")))
        source_ins = _as_text(entry.get("source_ins_code", entry.get("auth_ins_code", "")))
        raw_source_aa = entry.get("source_amino_acid") if entry else None
        source_aa = (
            _one_letter(raw_source_aa)
            if raw_source_aa not in (None, "", ".", "?")
            else ""
        )
        source_present = _as_bool(
            entry.get("source_present"),
            default=(source_label >= 0 and bool(entry)),
        )
        if not entry and has_annotations:
            # A source label alone is not enough to compare sequence identity;
            # the compiler metadata must also provide the source amino acid.
            source_present = False
        template_coords = _template_array(entry, "template_coords_nm", (37, 3))
        template_mask = _template_array(entry, "template_mask", (37,))
        if template_mask is not None:
            template_mask = template_mask.astype(bool)
        amino_acid = _one_letter(res_names[rows[0]])
        residues.append(
            _Residue(
                chain_id=chain_id,
                position=position,
                ins_code=ins_code,
                rows=rows,
                role_id=role_id,
                role=role_name,
                region_id=region_id,
                region=region_name,
                mutable=mutable,
                amino_acid=amino_acid,
                source_chain=source_chain,
                source_label_seq_id=source_label,
                source_auth_seq_id=source_auth,
                source_ins_code=source_ins,
                source_amino_acid=source_aa,
                source_present=source_present,
                template_coords_nm=template_coords,
                template_mask=template_mask,
            )
        )
    return residues


def _candidate_id(candidate: Any) -> str:
    value = _value(candidate, "candidate_id")
    if value:
        return _as_text(value)
    request = _value(candidate, "base_request")
    value = _value(request, "sample_id") if request is not None else None
    return _as_text(value, "candidate")


def _task_name(candidate: Any) -> str | None:
    request = _value(candidate, "base_request")
    task = _value(request, "task") if request is not None else _value(candidate, "task")
    value = _value(task, "name") if task is not None else _value(candidate, "task_name")
    return _as_text(value) if value else None


def _sample_id(candidate: Any) -> str | None:
    request = _value(candidate, "base_request")
    value = _value(request, "sample_id") if request is not None else _value(candidate, "sample_id")
    return _as_text(value) if value else None


def _atom_key(atoms: Any, row: int) -> tuple[str, int, str, str]:
    return (
        _as_text(_annotation(atoms, "chain_id", "")[row]),
        _as_int(_annotation(atoms, "res_id", 0)[row]),
        _as_text(_annotation(atoms, "ins_code", "")[row]),
        _as_text(_annotation(atoms, "atom_name", "")[row]).strip().upper(),
    )


def _partner_index(atoms: Any) -> dict[tuple[str, int, str, str], int]:
    result: dict[tuple[str, int, str, str], int] = {}
    for row in range(len(atoms)):
        result.setdefault(_atom_key(atoms, row), row)
    return result


def _fit_transform(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return row-vector rotation and translation mapping mobile to target."""

    mobile_centroid = np.mean(mobile, axis=0)
    target_centroid = np.mean(target, axis=0)
    mobile_centered = mobile - mobile_centroid
    target_centered = target - target_centroid
    covariance = mobile_centered.T @ target_centered
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    translation = target_centroid - mobile_centroid @ rotation
    return rotation, translation


def _aligned_rmsd(mobile: np.ndarray, target: np.ndarray) -> tuple[float | None, np.ndarray | None]:
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3 or not len(mobile):
        return None, None
    if not np.isfinite(mobile).all() or not np.isfinite(target).all():
        return None, None
    rotation, translation = _fit_transform(mobile, target)
    transformed = mobile @ rotation + translation
    residual = transformed - target
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))), (rotation, translation)


def _backbone_matches(
    candidate_atoms: Any,
    partner_atoms: Any,
    residues: Sequence[_Residue],
    *,
    region_ids: set[int] | None = None,
    binder_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    candidate_coords = np.asarray(candidate_atoms.coord, dtype=np.float64)
    partner_coords = np.asarray(partner_atoms.coord, dtype=np.float64)
    candidate_valid = _coordinate_valid_mask(candidate_atoms)
    partner_valid = _coordinate_valid_mask(partner_atoms)
    partner_rows = _partner_index(partner_atoms)
    mobile: list[np.ndarray] = []
    target: list[np.ndarray] = []
    chains: list[str] = []
    for residue in residues:
        if region_ids is not None and residue.region_id not in region_ids:
            continue
        if binder_only and residue.role_id not in {ROLE_HEAVY, ROLE_LIGHT}:
            continue
        for row in residue.rows:
            atom_name = _as_text(_annotation(candidate_atoms, "atom_name", "")[row]).strip().upper()
            if atom_name not in BACKBONE_ATOMS or not candidate_valid[row]:
                continue
            partner_row = partner_rows.get(_atom_key(candidate_atoms, row))
            if partner_row is None or not partner_valid[partner_row]:
                continue
            mobile.append(candidate_coords[row])
            target.append(partner_coords[partner_row])
            chains.append(residue.chain_id)
    if not mobile:
        return np.empty((0, 3)), np.empty((0, 3)), []
    return np.asarray(mobile), np.asarray(target), chains


def _per_chain_rmsd(mobile: np.ndarray, target: np.ndarray, chains: Sequence[str]) -> tuple[float | None, int]:
    if len(mobile) == 0:
        return None, 0
    squared: list[float] = []
    for chain in sorted(set(chains)):
        indices = np.asarray([index for index, value in enumerate(chains) if value == chain], dtype=np.int64)
        value, transform = _aligned_rmsd(mobile[indices], target[indices])
        if value is None or transform is None:
            continue
        rotation, translation = transform
        transformed = mobile[indices] @ rotation + translation
        squared.extend(np.sum((transformed - target[indices]) ** 2, axis=1).tolist())
    if not squared:
        return None, 0
    return float(np.sqrt(np.mean(squared))), len(squared)


def _template_fr_matches(
    atoms: Any,
    residues: Sequence[_Residue],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    try:
        from ..utils.atoms import PROTEIN_ATOM37
    except ImportError:  # pragma: no cover - package dependency guard.
        PROTEIN_ATOM37 = tuple()
    atom_names = _annotation(atoms, "atom_name", "")
    coords = np.asarray(atoms.coord, dtype=np.float64)
    valid = _coordinate_valid_mask(atoms)
    mobile: list[np.ndarray] = []
    target: list[np.ndarray] = []
    chains: list[str] = []
    for residue in residues:
        if residue.role_id not in {ROLE_HEAVY, ROLE_LIGHT} or residue.region_id != REGION_FR1:
            continue
        template = residue.template_coords_nm
        mask = residue.template_mask
        if template is None or mask is None:
            continue
        rows_by_name = {
            _as_text(atom_names[row]).strip().upper(): row for row in residue.rows
        }
        for slot, name in enumerate(PROTEIN_ATOM37):
            if name not in BACKBONE_ATOMS or slot >= len(mask) or not bool(mask[slot]):
                continue
            row = rows_by_name.get(name)
            if row is None or not valid[row] or not np.isfinite(template[slot]).all():
                continue
            # Compiler template coordinates are explicitly nanometres.
            mobile.append(coords[row])
            target.append(template[slot] * 10.0)
            chains.append(residue.chain_id)
    if not mobile:
        return np.empty((0, 3)), np.empty((0, 3)), []
    return np.asarray(mobile), np.asarray(target), chains


def _reference_metrics(
    atoms: Any,
    partner: Any | None,
    residues: Sequence[_Residue],
) -> dict[str, Any]:
    result = {
        "cdr_aligned_rmsd_angstrom": None,
        "cdr_aligned_atom_count": 0,
        "global_aligned_rmsd_angstrom": None,
        "global_aligned_atom_count": 0,
    }
    if partner is None:
        return result
    cdr_mobile, cdr_target, cdr_chains = _backbone_matches(
        atoms, partner, residues, region_ids=set(CDR_IDS)
    )
    global_mobile, global_target, global_chains = _backbone_matches(atoms, partner, residues)
    result["cdr_aligned_rmsd_angstrom"], result["cdr_aligned_atom_count"] = _per_chain_rmsd(
        cdr_mobile, cdr_target, cdr_chains
    )
    result["global_aligned_rmsd_angstrom"], result["global_aligned_atom_count"] = _per_chain_rmsd(
        global_mobile, global_target, global_chains
    )
    return result


def _target_aligned_rmsd(atoms: Any, folded: Any, residues: Sequence[_Residue]) -> tuple[float | None, int, str]:
    if folded is None:
        return None, 0, "missing_folded_atoms"
    candidate_valid = _coordinate_valid_mask(atoms)
    folded_valid = _coordinate_valid_mask(folded)
    candidate_coords = np.asarray(atoms.coord, dtype=np.float64)
    folded_coords = np.asarray(folded.coord, dtype=np.float64)
    folded_index = _partner_index(folded)
    binder_rows = [row for residue in residues if residue.role_id in {ROLE_HEAVY, ROLE_LIGHT} for row in residue.rows]
    context_rows = [row for residue in residues if residue.role_id == ROLE_CONTEXT for row in residue.rows]
    context_mobile: list[np.ndarray] = []
    context_target: list[np.ndarray] = []
    for row in context_rows:
        if not candidate_valid[row]:
            continue
        partner_row = folded_index.get(_atom_key(atoms, row))
        if partner_row is None or not folded_valid[partner_row]:
            continue
        context_mobile.append(candidate_coords[row])
        context_target.append(folded_coords[partner_row])
    binder_mobile: list[np.ndarray] = []
    binder_target: list[np.ndarray] = []
    for row in binder_rows:
        if not candidate_valid[row]:
            continue
        partner_row = folded_index.get(_atom_key(atoms, row))
        if partner_row is None or not folded_valid[partner_row]:
            continue
        binder_mobile.append(candidate_coords[row])
        binder_target.append(folded_coords[partner_row])
    if not binder_mobile:
        return None, 0, "no_matching_binder_atoms"
    if len(context_mobile) >= 3:
        rotation, translation = _fit_transform(np.asarray(context_mobile), np.asarray(context_target))
        transformed = np.asarray(binder_mobile) @ rotation + translation
        residual = transformed - np.asarray(binder_target)
        return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))), len(binder_mobile), "context"
    value, _ = _aligned_rmsd(np.asarray(binder_mobile), np.asarray(binder_target))
    return value, len(binder_mobile), "binder_fallback"


def _task_role_rows(atoms: Any, residues: Sequence[_Residue]) -> tuple[np.ndarray, np.ndarray, bool]:
    role_available = bool(_annotation_names(atoms).intersection(_ANNOTATION_NAMES))
    binder = np.zeros(len(atoms), dtype=bool)
    for residue in residues:
        if residue.role_id in {ROLE_HEAVY, ROLE_LIGHT}:
            binder[list(residue.rows)] = True
    if not binder.any():
        if "is_design" in _annotation_names(atoms):
            binder = np.asarray(atoms.is_design, dtype=bool) & (
                np.asarray(_annotation(atoms, "mol_type", "")).astype(str) == "protein"
            )
        elif "mol_type" in _annotation_names(atoms):
            # A metadata-free atom array is still analyzable when it contains
            # only one protein chain.  This path is intentionally conservative.
            mol_type = np.asarray(atoms.mol_type).astype(str)
            chain = np.asarray(_annotation(atoms, "chain_id", "")).astype(str)
            proteins = sorted(set(chain[mol_type == "protein"]))
            if len(proteins) == 1:
                binder = (chain == proteins[0]) & (mol_type == "protein")
    antigen = ~binder
    return np.flatnonzero(binder), np.flatnonzero(antigen), role_available


def _bond_exclusions(candidate: Any, atoms: Any) -> set[tuple[int, int]]:
    neighbours: list[set[int]] = [set() for _ in range(len(atoms))]
    bonds = getattr(atoms, "bonds", None)
    if bonds is not None:
        try:
            for raw_i, raw_j, _ in bonds.as_array():
                i, j = int(raw_i), int(raw_j)
                neighbours[i].add(j)
                neighbours[j].add(i)
        except (AttributeError, TypeError, ValueError):
            pass
    # SearchCandidate topology conditions are useful for generated covalent
    # edges that have not yet been materialized in the AtomArray bond list.
    uid_values = _annotation(atoms, "atom_uid", -1)
    uid_to_row = {_as_int(uid): row for row, uid in enumerate(uid_values)}
    for condition in (_value(candidate, "topology_conditions", ()) or ()):
        uid_i = _value(condition, "atom_uid_a")
        uid_j = _value(condition, "atom_uid_b")
        if uid_i is None or uid_j is None:
            endpoints = _value(condition, "atom_uids")
            if endpoints is not None and len(endpoints) == 2:
                uid_i, uid_j = endpoints
        if uid_i is None or uid_j is None:
            continue
        i, j = uid_to_row.get(_as_int(uid_i), -1), uid_to_row.get(_as_int(uid_j), -1)
        if i >= 0 and j >= 0 and i != j:
            neighbours[i].add(j)
            neighbours[j].add(i)
    excluded: set[tuple[int, int]] = set()
    for start, direct in enumerate(neighbours):
        for first in direct:
            excluded.add((min(start, first), max(start, first)))
            for second in neighbours[first]:
                if second != start:
                    excluded.add((min(start, second), max(start, second)))
    return excluded


def _element(atoms: Any, row: int) -> str:
    raw = _as_text(_annotation(atoms, "element", "")[row]).strip().upper()
    if not raw:
        atom_name = _as_text(_annotation(atoms, "atom_name", "")[row]).strip().upper()
        raw = atom_name[:2] if atom_name[:2] in _VDW_RADII_ANGSTROM else atom_name[:1]
    return raw


def _radius(element: str) -> float:
    if element in _VDW_RADII_ANGSTROM:
        return _VDW_RADII_ANGSTROM[element]
    return _VDW_RADII_ANGSTROM.get(element[:1], 1.70)


_PAIR_DISTANCE_BLOCK_SIZE = 4096
_PAIR_DISTANCE_TILE_SIZE = 65536


def _atom_radii(atoms: Any) -> np.ndarray:
    elements = _annotation(atoms, "element", "")
    atom_names = _annotation(atoms, "atom_name", "")
    radii = np.empty(len(atoms), dtype=np.float64)
    for row in range(len(atoms)):
        element = _as_text(elements[row]).strip().upper()
        if not element:
            atom_name = _as_text(atom_names[row]).strip().upper()
            element = atom_name[:2] if atom_name[:2] in _VDW_RADII_ANGSTROM else atom_name[:1]
        radii[row] = _radius(element)
    return radii


def _atom_pair_geometry_blocks(
    coords: np.ndarray,
    radii: np.ndarray,
    first_rows: np.ndarray,
    second_rows: np.ndarray,
    excluded_by_row: Mapping[int, np.ndarray],
    *,
    upper_triangle: bool = False,
    contact_threshold_angstrom: float | None,
    clash_overlap_threshold_angstrom: float,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not len(first_rows) or not len(second_rows):
        return
    # A tile spans the full second axis so flattening keeps first/second pair
    # order.  For a longer second axis, process one first row at a time.
    first_block_size = max(1, _PAIR_DISTANCE_TILE_SIZE // len(second_rows)) if len(second_rows) <= _PAIR_DISTANCE_BLOCK_SIZE else 1
    first_coords, second_coords = coords[first_rows], coords[second_rows]
    first_radii, second_radii = radii[first_rows], radii[second_rows]

    # Map sparse exclusions to second-axis positions once.  Position lists
    # preserve duplicate CDR rows; a row-to-single-position map would not.
    second_positions: dict[int, list[int]] = defaultdict(list)
    for position, row in enumerate(second_rows):
        second_positions[int(row)].append(position)
    excluded_columns_by_row: dict[int, list[int]] = {}
    excluded_columns: list[int] = []
    excluded_offsets = [0]
    for raw_first in first_rows:
        first = int(raw_first)
        if first not in excluded_columns_by_row:
            excluded_columns_by_row[first] = [
                position
                for second in excluded_by_row.get(first, ())
                for position in second_positions.get(int(second), ())
            ]
        excluded_columns.extend(excluded_columns_by_row[first])
        excluded_offsets.append(len(excluded_columns))
    excluded_columns_array = np.asarray(excluded_columns, dtype=np.intp)
    excluded_first_positions = np.repeat(np.arange(len(first_rows)), np.diff(excluded_offsets))

    roundoff = 8.0 * np.finfo(np.float64).eps
    for first_start in range(0, len(first_rows), first_block_size):
        first_stop = min(first_start + first_block_size, len(first_rows))
        excluded_start, excluded_stop = excluded_offsets[first_start], excluded_offsets[first_stop]
        tile_excluded_first = excluded_first_positions[excluded_start:excluded_stop] - first_start
        tile_excluded_second = excluded_columns_array[excluded_start:excluded_stop]
        start = first_start + 1 if upper_triangle else 0
        for offset in range(start, len(second_rows), _PAIR_DISTANCE_BLOCK_SIZE):
            stop = min(offset + _PAIR_DISTANCE_BLOCK_SIZE, len(second_rows))
            in_block = (tile_excluded_second >= offset) & (tile_excluded_second < stop)
            if upper_triangle or in_block.any():
                if upper_triangle:
                    allowed = np.arange(offset, stop)[None, :] > np.arange(first_start, first_stop)[:, None]
                else:
                    allowed = np.ones((first_stop - first_start, stop - offset), dtype=bool)
                allowed[tile_excluded_first[in_block], tile_excluded_second[in_block] - offset] = False
                first_positions, positions = np.nonzero(allowed)
                if not len(first_positions):
                    continue
                first_positions += first_start
                positions += offset
                firsts, seconds = first_rows[first_positions], second_rows[positions]
                delta = first_coords[first_positions] - second_coords[positions]
                distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
                radius_sums = first_radii[first_positions] + second_radii[positions]
            else:
                delta = first_coords[first_start:first_stop, None, :] - second_coords[None, offset:stop, :]
                distances = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta)).reshape(-1)
                delta = delta.reshape(-1, 3)
                radius_sums = (first_radii[first_start:first_stop, None] + second_radii[None, offset:stop]).reshape(-1)
                firsts = np.repeat(first_rows[first_start:first_stop], stop - offset)
                seconds = np.tile(second_rows[offset:stop], first_stop - first_start)
            overlaps = radius_sums - distances

            # The scalar norm uses a dot product, while the axis norm reduces
            # squared components.  Recheck only threshold-adjacent pairs with
            # that original scalar operation so its last-bit rounding cannot
            # change an inclusive contact or a strict clash classification.
            scale = np.maximum(1.0, np.maximum(radius_sums, distances))
            near_threshold = np.abs(overlaps - clash_overlap_threshold_angstrom) <= roundoff * np.maximum(
                scale, abs(clash_overlap_threshold_angstrom)
            )
            if contact_threshold_angstrom is not None:
                near_threshold |= np.abs(distances - contact_threshold_angstrom) <= roundoff * np.maximum(
                    scale, abs(contact_threshold_angstrom)
                )
            for index in np.flatnonzero(near_threshold):
                distances[index] = float(np.linalg.norm(delta[index]))
            overlaps = radius_sums - distances
            yield firsts, seconds, distances, overlaps


def _interface_metrics(
    candidate: Any,
    atoms: Any,
    residues: Sequence[_Residue],
    *,
    contact_threshold_angstrom: float,
    clash_overlap_threshold_angstrom: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binder_rows, antigen_rows, role_available = _task_role_rows(atoms, residues)
    valid = _coordinate_valid_mask(atoms)
    coords = np.asarray(atoms.coord, dtype=np.float64)
    exclusions = _bond_exclusions(candidate, atoms)
    excluded_neighbours: dict[int, list[int]] = defaultdict(list)
    for first, second in exclusions:
        excluded_neighbours[first].append(second)
        excluded_neighbours[second].append(first)
    excluded_by_row = {row: np.asarray(neighbours, dtype=np.intp) for row, neighbours in excluded_neighbours.items()}
    radii = _atom_radii(atoms)
    contact_pairs: list[tuple[int, int]] = []
    clash_pairs: list[tuple[int, int]] = []
    clash_penalty = 0.0
    for firsts, seconds, distances, overlaps in _atom_pair_geometry_blocks(
        coords,
        radii,
        binder_rows[valid[binder_rows]],
        antigen_rows[valid[antigen_rows]],
        excluded_by_row,
        contact_threshold_angstrom=contact_threshold_angstrom,
        clash_overlap_threshold_angstrom=clash_overlap_threshold_angstrom,
    ):
        contacting = distances <= contact_threshold_angstrom
        contact_firsts, contact_seconds = firsts[contacting].tolist(), seconds[contacting].tolist()
        contact_pairs.extend((min(first, second), max(first, second)) for first, second in zip(contact_firsts, contact_seconds))
        clashing = overlaps > clash_overlap_threshold_angstrom
        clash_firsts, clash_seconds = firsts[clashing].tolist(), seconds[clashing].tolist()
        clash_pairs.extend(
            (min(first, second), max(first, second))
            for first, second in zip(clash_firsts, clash_seconds)
        )
        clash_penalty += float(np.sum((overlaps[clashing] - clash_overlap_threshold_angstrom) ** 2))

    # CDR clashes are intentionally a separate diagnostic.  They include H/L
    # pairs within CDRs, whereas the interface metrics above never do.
    cdr_rows = np.asarray(
        [row for residue in residues if residue.role_id in {ROLE_HEAVY, ROLE_LIGHT} and residue.region_id in CDR_IDS for row in residue.rows],
        dtype=np.intp,
    )
    cdr_rows = cdr_rows[valid[cdr_rows]]
    cdr_clash_pairs: list[tuple[int, int]] = []
    cdr_clash_penalty = 0.0
    for firsts, seconds, _, overlaps in _atom_pair_geometry_blocks(
        coords,
        radii,
        cdr_rows,
        cdr_rows,
        excluded_by_row,
        upper_triangle=True,
        contact_threshold_angstrom=None,
        clash_overlap_threshold_angstrom=clash_overlap_threshold_angstrom,
    ):
        clashing = overlaps > clash_overlap_threshold_angstrom
        clash_firsts, clash_seconds = firsts[clashing].tolist(), seconds[clashing].tolist()
        cdr_clash_pairs.extend(
            (min(first, second), max(first, second))
            for first, second in zip(clash_firsts, clash_seconds)
        )
        cdr_clash_penalty += float(np.sum((overlaps[clashing] - clash_overlap_threshold_angstrom) ** 2))

    chain_values = _annotation(atoms, "chain_id", "")
    res_values = _annotation(atoms, "res_id", 0)
    mol_types = _annotation(atoms, "mol_type", "")
    residue_contact_pairs = sorted(
        {
            (
                _as_text(chain_values[first]),
                _as_int(res_values[first]),
                _as_text(chain_values[second]),
                _as_int(res_values[second]),
            )
            for first, second in contact_pairs
        }
    )
    metrics = {
        "binder_antigen_contact_count": int(len(contact_pairs)),
        "binder_antigen_contact_residue_count": int(len(residue_contact_pairs)),
        "binder_antigen_clash_count": int(len(clash_pairs)),
        "binder_antigen_clash_penalty": float(clash_penalty),
        "binder_cdr_clash_count": int(len(cdr_clash_pairs)),
        "binder_cdr_clash_penalty": float(cdr_clash_penalty),
        # Short aliases make offline threshold files readable while preserving
        # the explicit binder/antigen names above.
        "contact_count": int(len(contact_pairs)),
        "clash_count": int(len(clash_pairs)),
        "cdr_clash_count": int(len(cdr_clash_pairs)),
    }
    details = {
        "binder_chain_ids": sorted({_as_text(chain_values[row]) for row in binder_rows}),
        "antigen_chain_ids": sorted({_as_text(chain_values[row]) for row in antigen_rows}),
        "antigen_modalities": sorted({_as_text(mol_types[row]) for row in antigen_rows if _as_text(mol_types[row])}),
        "role_annotations_used": bool(role_available),
        "contact_residue_pairs": [list(pair) for pair in residue_contact_pairs],
    }
    return metrics, details


def _residue_report(residues: Sequence[_Residue]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    chains: dict[str, Any] = {}
    cdr_sequences: dict[str, dict[str, str]] = {}
    lengths: dict[str, int] = {}
    mutable_positions: dict[str, list[int]] = {}
    ledger: list[dict[str, Any]] = []
    fixed_audit: dict[str, Any] = {
        "checked": 0,
        "preserved": 0,
        "violations": [],
        "inserted_or_missing_source": 0,
        "source_coordinate_missing": 0,
        "source_identity_unavailable": 0,
        "by_region": {
            name: {"checked": 0, "preserved": 0, "violations": []}
            for name in ("FR1", "CDR1", "CDR2", "CDR3")
        },
    }
    for chain_id in sorted({residue.chain_id for residue in residues}):
        chain_residues = [residue for residue in residues if residue.chain_id == chain_id]
        chain_residues.sort(key=lambda residue: (residue.position, residue.ins_code))
        role_id = chain_residues[0].role_id if chain_residues else ROLE_CONTEXT
        role = chain_residues[0].role if chain_residues else "context"
        sequence = "".join(residue.amino_acid for residue in chain_residues)
        region_sequences: dict[str, str] = {}
        for region_id, region_name in REGION_NAMES.items():
            if region_id == REGION_CONTEXT:
                continue
            region_sequences[region_name] = "".join(
                residue.amino_acid for residue in chain_residues if residue.region_id == region_id
            )
        selected_cdrs = {name: region_sequences[name] for name in ("CDR1", "CDR2", "CDR3") if region_sequences[name]}
        cdr_sequences[chain_id] = selected_cdrs
        mutable = [residue.position for residue in chain_residues if residue.mutable]
        mutable_positions[chain_id] = mutable
        lengths[chain_id] = len(chain_residues)
        chains[chain_id] = {
            "role": role,
            "role_id": int(role_id),
            "length": len(chain_residues),
            "sequence": sequence,
            "region_sequences": region_sequences,
            "cdr_sequences": selected_cdrs,
            "mutable_positions": mutable,
            "positions": [int(residue.position) for residue in chain_residues],
            "source_label_seq_ids": [int(residue.source_label_seq_id) for residue in chain_residues],
        }
        for residue in chain_residues:
            source_identity_present = (
                residue.source_label_seq_id >= 0
                and residue.source_amino_acid in _ONE_LETTER
                and residue.source_amino_acid != "X"
            )
            source_aa = residue.source_amino_acid if source_identity_present else None
            row = {
                "chain_id": residue.chain_id,
                "role": residue.role,
                "role_id": int(residue.role_id),
                "position": int(residue.position),
                "ins_code": residue.ins_code,
                "region": residue.region,
                "region_id": int(residue.region_id),
                "mutable": bool(residue.mutable),
                "candidate_amino_acid": residue.amino_acid,
                "source_chain": residue.source_chain,
                "source_label_seq_id": int(residue.source_label_seq_id),
                "source_auth_seq_id": residue.source_auth_seq_id,
                "source_ins_code": residue.source_ins_code,
                "source_amino_acid": source_aa,
                "source_present": bool(residue.source_present),
                "source_coordinate_present": bool(residue.source_present),
                "source_identity_present": bool(source_identity_present),
            }
            ledger.append(row)
            if residue.role_id == ROLE_CONTEXT:
                continue
            if residue.mutable or not source_identity_present:
                if residue.source_label_seq_id < 0:
                    fixed_audit["inserted_or_missing_source"] += 1
                elif not source_identity_present:
                    fixed_audit["source_identity_unavailable"] += 1
                continue
            if not residue.source_present:
                fixed_audit["source_coordinate_missing"] += 1
            fixed_audit["checked"] += 1
            region_audit = fixed_audit["by_region"].setdefault(
                residue.region,
                {"checked": 0, "preserved": 0, "violations": []},
            )
            region_audit["checked"] += 1
            if source_aa == residue.amino_acid:
                fixed_audit["preserved"] += 1
                region_audit["preserved"] += 1
            else:
                violation = {
                    "chain_id": residue.chain_id,
                    "position": int(residue.position),
                    "source_amino_acid": source_aa,
                    "candidate_amino_acid": residue.amino_acid,
                }
                fixed_audit["violations"].append(violation)
                region_audit["violations"].append(violation)
    return chains, cdr_sequences, lengths, mutable_positions, ledger, fixed_audit


def _cdr_liabilities(atoms: Any, residues: Sequence[_Residue]) -> dict[str, Any]:
    bonds = set()
    if getattr(atoms, "bonds", None) is not None:
        try:
            bonds = {
                (min(int(i), int(j)), max(int(i), int(j)))
                for i, j, _ in atoms.bonds.as_array()
            }
        except (AttributeError, TypeError, ValueError):
            bonds = set()
    result: dict[str, Any] = {}
    hydrophobic = frozenset("AVILMFWY")
    for chain_id in sorted({residue.chain_id for residue in residues}):
        chain = [residue for residue in residues if residue.chain_id == chain_id]
        cdr_result: dict[str, Any] = {}
        for region_id in CDR_IDS:
            selected = [residue for residue in chain if residue.region_id == region_id]
            if not selected:
                continue
            sequence = "".join(residue.amino_acid for residue in selected)
            motif_positions = [
                int(selected[index].position)
                for index in range(max(0, len(sequence) - 2))
                if sequence[index] == "N" and sequence[index + 1] != "P" and sequence[index + 2] in {"S", "T"}
            ]
            cys_positions: list[int] = []
            paired_positions: set[int] = set()
            for residue in selected:
                if residue.amino_acid != "C":
                    continue
                cys_positions.append(int(residue.position))
                sg_rows = [
                    row for row in residue.rows if _as_text(_annotation(atoms, "atom_name", "")[row]).strip().upper() == "SG"
                ]
                for other in selected:
                    if other is residue or other.amino_acid != "C":
                        continue
                    other_sg = [
                        row for row in other.rows if _as_text(_annotation(atoms, "atom_name", "")[row]).strip().upper() == "SG"
                    ]
                    if any((min(first, second), max(first, second)) in bonds for first in sg_rows for second in other_sg):
                        paired_positions.update((int(residue.position), int(other.position)))
            clusters: list[dict[str, Any]] = []
            start = None
            for index, amino_acid in enumerate(sequence):
                if amino_acid in hydrophobic:
                    if start is None:
                        start = index
                elif start is not None:
                    if index - start >= 3:
                        clusters.append({"start_index": start, "end_index": index - 1, "sequence": sequence[start:index]})
                    start = None
            if start is not None and len(sequence) - start >= 3:
                clusters.append({"start_index": start, "end_index": len(sequence) - 1, "sequence": sequence[start:]})
            cdr_name = REGION_NAMES[region_id]
            cdr_result[cdr_name] = {
                "sequence": sequence,
                "selected": any(residue.mutable for residue in selected),
                "n_x_s_t_positions": motif_positions,
                "cysteine_count": len(cys_positions),
                "cysteine_positions": cys_positions,
                "unpaired_cysteine_positions": sorted(set(cys_positions).difference(paired_positions)),
                "cysteine_pairing_basis": "explicit SG-SG bond only; no experimental proof",
                "hydrophobic_clusters": clusters,
            }
        if cdr_result:
            result[chain_id] = cdr_result
    return result


def analyze_antibody(
    candidate: Any,
    *,
    scaffold_entries: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    reference_atoms: Any | None = None,
    folded_atoms: Any | None = None,
    contact_threshold_angstrom: float = 4.5,
    clash_overlap_threshold_angstrom: float = 0.4,
) -> dict[str, Any]:
    """Analyze one assembled antibody candidate without loading a model.

    ``candidate`` can be ``SearchCandidate``-like (``candidate.atoms``) or an
    ``AtomArray``.  ``scaffold_entries`` should be the compiler's metadata list;
    when omitted it is discovered from ``candidate.compiled.antibody_metadata``
    and the parsed antibody request.  Template coordinates are used only when
    both their mask and candidate coordinates are visible and valid.
    """

    if not np.isfinite(float(contact_threshold_angstrom)) or contact_threshold_angstrom < 0:
        raise ValueError("contact_threshold_angstrom must be finite and nonnegative")
    if not np.isfinite(float(clash_overlap_threshold_angstrom)) or clash_overlap_threshold_angstrom < 0:
        raise ValueError("clash_overlap_threshold_angstrom must be finite and nonnegative")
    atoms = _extract_atoms(candidate)
    if not hasattr(atoms, "coord"):
        raise TypeError("antibody AtomArray has no coordinates")
    entries = _find_metadata(candidate, scaffold_entries)
    residues = _build_residues(atoms, entries, candidate)
    chains, cdr_sequences, lengths, mutable_positions, ledger, fixed_audit = _residue_report(residues)
    metrics: dict[str, Any] = {
        "fr_aligned_backbone_rmsd_angstrom": None,
        "fr_aligned_backbone_atom_count": 0,
        "fr_per_chain_aligned_backbone_rmsd_angstrom": None,
        "cdr_aligned_rmsd_angstrom": None,
        "cdr_aligned_atom_count": 0,
        "global_aligned_rmsd_angstrom": None,
        "global_aligned_atom_count": 0,
        "target_aligned_rmsd_angstrom": None,
        "target_aligned_atom_count": 0,
        "binder_antigen_contact_count": 0,
        "binder_antigen_contact_residue_count": 0,
        "binder_antigen_clash_count": 0,
        "binder_antigen_clash_penalty": 0.0,
        "binder_cdr_clash_count": 0,
        "binder_cdr_clash_penalty": 0.0,
        "contact_count": 0,
        "clash_count": 0,
        "cdr_clash_count": 0,
    }
    template_mobile, template_target, template_chains = _template_fr_matches(atoms, residues)
    # The canonical metric aligns the complete paired VH/VL framework as one
    # geometry group, so a bad relative VH/VL placement is visible.  The
    # per-chain value remains a diagnostic for locating that discrepancy.
    metrics["fr_aligned_backbone_rmsd_angstrom"], metrics["fr_aligned_backbone_atom_count"] = _aligned_rmsd(
        template_mobile, template_target
    )[0], len(template_mobile)
    metrics["fr_per_chain_aligned_backbone_rmsd_angstrom"], _ = _per_chain_rmsd(
        template_mobile, template_target, template_chains
    )
    partner = reference_atoms if reference_atoms is not None else folded_atoms
    if partner is not None:
        if not _is_atom_array(partner):
            partner = _extract_atoms(partner)
        metrics.update(_reference_metrics(atoms, partner, residues))
    target_rmsd, target_count, target_basis = _target_aligned_rmsd(atoms, folded_atoms, residues)
    metrics["target_aligned_rmsd_angstrom"] = target_rmsd
    metrics["target_aligned_atom_count"] = int(target_count)
    interface_metrics, interface_details = _interface_metrics(
        candidate,
        atoms,
        residues,
        contact_threshold_angstrom=float(contact_threshold_angstrom),
        clash_overlap_threshold_angstrom=float(clash_overlap_threshold_angstrom),
    )
    metrics.update(interface_metrics)
    # Readable aliases keep the report convenient for downstream notebooks;
    # the explicit ``*_angstrom`` names remain canonical for filtering.
    metrics.update(
        {
            "fr_aligned_backbone_rmsd": metrics["fr_aligned_backbone_rmsd_angstrom"],
            "fr_per_chain_aligned_backbone_rmsd": metrics["fr_per_chain_aligned_backbone_rmsd_angstrom"],
            "cdr_aligned_rmsd": metrics["cdr_aligned_rmsd_angstrom"],
            "global_aligned_rmsd": metrics["global_aligned_rmsd_angstrom"],
            "target_aligned_rmsd": metrics["target_aligned_rmsd_angstrom"],
        }
    )
    report = {
        "schema": "antibody-analysis-v1",
        "status": "ok",
        "candidate_id": _candidate_id(candidate),
        "task_name": _task_name(candidate),
        "sample_id": _sample_id(candidate),
        "metrics": metrics,
        "thresholds": {
            "contact_threshold_angstrom": float(contact_threshold_angstrom),
            "clash_overlap_threshold_angstrom": float(clash_overlap_threshold_angstrom),
        },
        "geometry": {
            "reference_atoms_supplied": reference_atoms is not None,
            "folded_atoms_supplied": folded_atoms is not None,
            "reference_atoms_semantics": (
                "caller_supplied_reference"
                if reference_atoms is not None
                else None
            ),
            "folded_atoms_semantics": (
                "caller_supplied_folded_or_original_generated; not experimental reference"
                if folded_atoms is not None
                else None
            ),
            "target_alignment_basis": target_basis,
        },
        "chains": chains,
        "cdr_sequences": cdr_sequences,
        "lengths": lengths,
        "mutable_positions": mutable_positions,
        "source_ledger": ledger,
        "sequence_preservation": fixed_audit,
        "liabilities": {"cdrs": _cdr_liabilities(atoms, residues), "experimental_proof": False},
        "interface": interface_details,
        "analysis_errors": [],
    }
    return _json_safe(report)


def _report_id(report: Mapping[str, Any]) -> str:
    for key in ("candidate_id", "sample_id", "id"):
        value = report.get(key)
        if value:
            return str(value)
    return "candidate"


def _report_cdr_pair(report: Mapping[str, Any]) -> tuple[str, ...]:
    cdrs = report.get("cdr_sequences", {})
    chains = report.get("chains", {})
    if not isinstance(cdrs, Mapping):
        cdrs = {}
    role_to_chain: dict[str, str] = {}
    for chain_id, chain in chains.items() if isinstance(chains, Mapping) else ():
        if isinstance(chain, Mapping):
            raw_role = chain.get("role", chain.get("role_id", ""))
            role = _as_text(raw_role).upper()
            if role == str(ROLE_HEAVY):
                role = "H"
            elif role == str(ROLE_LIGHT):
                role = "L"
            if role in {"H", "VHH", "L"}:
                role_to_chain[role] = str(chain_id)
    ordered: list[str] = []
    for role in ("H", "VHH", "L"):
        chain_id = role_to_chain.get(role)
        if chain_id is None:
            continue
        chain_cdr = cdrs.get(chain_id, {}) if isinstance(cdrs, Mapping) else {}
        if not isinstance(chain_cdr, Mapping):
            chain_cdr = {}
        ordered.extend(f"{role}:{name}:{_as_text(chain_cdr.get(name, ''))}" for name in ("CDR1", "CDR2", "CDR3"))
    if not ordered and isinstance(cdrs, Mapping):
        for chain_id in sorted(cdrs):
            chain_cdr = cdrs[chain_id]
            if isinstance(chain_cdr, Mapping):
                ordered.extend(f"{chain_id}:{name}:{_as_text(chain_cdr.get(name, ''))}" for name in ("CDR1", "CDR2", "CDR3"))
    return tuple(ordered)


def _pair_hamming(left: Sequence[str], right: Sequence[str]) -> int:
    total = 0
    for a, b in zip(left, right):
        total += sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
    return total + abs(len(left) - len(right))


def select_paired_sequence_diversity(
    reports: Sequence[Mapping[str, Any]],
    *,
    max_count: int,
    min_pairwise_hamming: int = 0,
) -> list[dict[str, Any]]:
    """Select reports greedily by paired CDR diversity, preserving ownership.

    A selected item is always an intact report, so a heavy and light chain can
    never be recombined from different candidates.  Ties are resolved by the
    stable candidate/sample ID and then input order.
    """

    if isinstance(max_count, bool) or int(max_count) < 0:
        raise ValueError("max_count must be a nonnegative integer")
    if isinstance(min_pairwise_hamming, bool) or int(min_pairwise_hamming) < 0:
        raise ValueError("min_pairwise_hamming must be a nonnegative integer")
    ordered = sorted(
        enumerate(reports),
        key=lambda item: (_report_id(item[1]), _as_text(item[1].get("task_name")), item[0]),
    )
    selected: list[dict[str, Any]] = []
    selected_keys: list[tuple[str, ...]] = []
    for _, report in ordered:
        if len(selected) >= int(max_count):
            break
        key = _report_cdr_pair(report)
        if key in selected_keys or any(
            _pair_hamming(key, previous) < int(min_pairwise_hamming)
            for previous in selected_keys
        ):
            continue
        selected.append(dict(report))
        selected_keys.append(key)
    return selected


def refilter_antibody_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    min_contacts: int | None = None,
    max_clashes: int | None = None,
    max_cdr_clashes: int | None = None,
    max_fr_rmsd_angstrom: float | None = None,
    max_cdr_rmsd_angstrom: float | None = None,
    max_global_rmsd_angstrom: float | None = None,
    max_liability_count: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply explicit offline thresholds and return ``(kept, rejected)``."""

    checks: list[tuple[str, str, str, float | int]] = []
    if min_contacts is not None:
        checks.append(("min_contacts", "binder_antigen_contact_count", "min", int(min_contacts)))
    if max_clashes is not None:
        checks.append(("max_clashes", "binder_antigen_clash_count", "max", int(max_clashes)))
    if max_cdr_clashes is not None:
        checks.append(("max_cdr_clashes", "binder_cdr_clash_count", "max", int(max_cdr_clashes)))
    if max_fr_rmsd_angstrom is not None:
        checks.append(("max_fr_rmsd_angstrom", "fr_aligned_backbone_rmsd_angstrom", "max", float(max_fr_rmsd_angstrom)))
    if max_cdr_rmsd_angstrom is not None:
        checks.append(("max_cdr_rmsd_angstrom", "cdr_aligned_rmsd_angstrom", "max", float(max_cdr_rmsd_angstrom)))
    if max_global_rmsd_angstrom is not None:
        checks.append(("max_global_rmsd_angstrom", "global_aligned_rmsd_angstrom", "max", float(max_global_rmsd_angstrom)))
    for name, _, _, threshold in checks:
        if not np.isfinite(float(threshold)) or (name.startswith("min_") and float(threshold) < 0) or (name.startswith("max_") and float(threshold) < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for report in sorted(reports, key=lambda item: _report_id(item)):
        reasons: list[str] = []
        status = _as_text(report.get("status", "ok")).lower()
        if status not in {"ok", "success", "succeeded"}:
            reasons.append(f"status:{status or 'failed'}")
        metrics = report.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        for name, key, mode, threshold in checks:
            raw = metrics.get(key)
            value = _finite_float(raw)
            if value is None:
                reasons.append(f"{name}:missing")
            elif mode == "min" and value < float(threshold):
                reasons.append(f"{name}:{value:g}<{float(threshold):g}")
            elif mode == "max" and value > float(threshold):
                reasons.append(f"{name}:{value:g}>{float(threshold):g}")
        if max_liability_count is not None:
            if isinstance(max_liability_count, bool) or int(max_liability_count) < 0:
                raise ValueError("max_liability_count must be a nonnegative integer")
            liabilities = report.get("liabilities", {})
            total = 0
            if isinstance(liabilities, Mapping):
                cdrs = liabilities.get("cdrs", {})
                if isinstance(cdrs, Mapping):
                    for chain in cdrs.values():
                        if not isinstance(chain, Mapping):
                            continue
                        for item in chain.values():
                            if isinstance(item, Mapping):
                                total += len(item.get("n_x_s_t_positions", ()))
                                total += len(item.get("unpaired_cysteine_positions", ()))
                                total += len(item.get("hydrophobic_clusters", ()))
            if total > int(max_liability_count):
                reasons.append(f"max_liability_count:{total}>{int(max_liability_count)}")
        if reasons:
            item = dict(report)
            item["filter_reasons"] = reasons
            rejected.append(item)
        else:
            kept.append(dict(report))
    return kept, rejected


def summarize_antibody_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    failures: Sequence[Mapping[str, Any]] = (),
    diversity_count: int | None = None,
    diversity_min_hamming: int = 0,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic task-local refilter and diversity summary."""

    threshold_values = dict(thresholds or {})
    kept, rejected = refilter_antibody_reports(reports, **threshold_values)
    selected = kept
    if diversity_count is not None:
        selected = select_paired_sequence_diversity(
            kept, max_count=int(diversity_count), min_pairwise_hamming=int(diversity_min_hamming)
        )
    failed_reports = [report for report in reports if _as_text(report.get("status", "ok")).lower() not in {"ok", "success", "succeeded"}]
    failure_rows = [dict(item) for item in failures]
    failure_by_stage: dict[str, int] = defaultdict(int)
    for item in [*failed_reports, *failure_rows]:
        stage = _as_text(item.get("stage", item.get("error_stage", "unknown"))) or "unknown"
        failure_by_stage[stage] += 1
    return _json_safe(
        {
            "schema": "antibody-refilter-v1",
            "reports_total": len(reports),
            "reports_successful": len(reports) - len(failed_reports),
            "reports_failed": len(failed_reports),
            "source_csv_failed_rows": len(failure_rows),
            "candidate_failures_total": len(failed_reports) + len(failure_rows),
            "failure_by_stage": dict(sorted(failure_by_stage.items())),
            "thresholds": threshold_values,
            "kept_count": len(kept),
            "rejected_count": len(rejected),
            "rejected": rejected,
            "selected_count": len(selected),
            "selected_candidate_ids": [_report_id(item) for item in selected],
            "selected": selected,
            "fixed_sequence_audits": {
                _report_id(item): item.get("sequence_preservation", {})
                for item in sorted(reports, key=_report_id)
            },
            "failed_source_rows": failure_rows,
        }
    )


def load_report_json(path: str | Path) -> list[dict[str, Any]]:
    """Read a report JSON file in one of the supported list/container forms."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    if isinstance(raw, Mapping):
        for key in ("reports", "results", "candidates"):
            value = raw.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        if all(isinstance(value, Mapping) for value in raw.values()):
            result = []
            for key, value in raw.items():
                item = dict(value)
                item.setdefault("candidate_id", str(key))
                result.append(item)
            return result
    raise ValueError("reports JSON must contain a list or reports/results/candidates list")


__all__ = [
    "AMINO_ACID_CATALOG",
    "ROLE_CONTEXT",
    "ROLE_HEAVY",
    "ROLE_VHH",
    "ROLE_LIGHT",
    "REGION_CONTEXT",
    "REGION_FR1",
    "REGION_CDR1",
    "REGION_CDR2",
    "REGION_CDR3",
    "analyze_antibody",
    "load_report_json",
    "refilter_antibody_reports",
    "select_paired_sequence_diversity",
    "summarize_antibody_reports",
]
