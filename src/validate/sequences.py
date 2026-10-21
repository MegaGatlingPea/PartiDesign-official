"""Sequence planning for standalone validation.

This module adapts result-row/AtomArray validation inputs to the existing
offline MPNN workers. The public source list is explicit; validation never
uses the reward auto selector.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass, replace as dataclass_replace
from uuid import uuid4
from pathlib import Path
import os
from partidesign.paths import community_asset, ENV_MODEL_DIR
from src.design.rewards.aa_omission import resolve_rm_aa
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence


class SequenceGenerationError(RuntimeError):
    """A requested validation sequence source cannot be executed."""


_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common parent-preserving modified residues.  Unrecognised components
    # remain X and are accepted only when the explicit fixed-restype ledger
    # supplies a safe parent through the native candidate adapter.
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y", "HYP": "P",
    "CSO": "C", "CSD": "C", "DAL": "A", "DLE": "L", "DVA": "V",
    "DLY": "K", "DPN": "F", "DTH": "T", "DTR": "W", "DSG": "N",
}
_RNA3_TO_1 = {"A": "A", "G": "G", "C": "C", "U": "U"}
_DNA3_TO_1 = {"DA": "A", "DG": "G", "DC": "C", "DT": "T"}
_PUBLIC_MPNN = {
    "protein_mpnn", "soluble_mpnn", "ligand_mpnn", "cyclic_mpnn", "na_mpnn",
}
_SOURCE_ALIASES = {
    "gt": "gt",
    "protein_mpnn": "protein_mpnn", "proteinmpnn": "protein_mpnn",
    "protein_mpnn-fi": "protein_mpnn", "proteinmpnn-fi": "protein_mpnn",
    "soluble_mpnn": "soluble_mpnn", "solublempnn": "soluble_mpnn",
    "soluble_mpnn-fi": "soluble_mpnn", "solublempnn-fi": "soluble_mpnn",
    "ligand_mpnn": "ligand_mpnn", "ligandmpnn": "ligand_mpnn",
    "ligand_mpnn-fi": "ligand_mpnn", "ligandmpnn-fi": "ligand_mpnn",
    "cyclic_mpnn": "cyclic_mpnn", "cyclicmpnn": "cyclic_mpnn",
    "cyclic_mpnn-fi": "cyclic_mpnn", "cyclicmpnn-fi": "cyclic_mpnn",
    "na_mpnn": "na_mpnn", "nampnn": "na_mpnn",
    "na_mpnn-fi": "na_mpnn", "nampnn-fi": "na_mpnn",
}
_FI_SOURCE_ALIASES = frozenset({
    "protein_mpnn-fi", "proteinmpnn-fi",
    "soluble_mpnn-fi", "solublempnn-fi",
    "ligand_mpnn-fi", "ligandmpnn-fi",
    "cyclic_mpnn-fi", "cyclicmpnn-fi",
    "na_mpnn-fi", "nampnn-fi",
})
_INTERNAL_SOURCES = {
    "protein_mpnn": "proteinmpnn", "soluble_mpnn": "proteinmpnn",
    "ligand_mpnn": "ligandmpnn", "cyclic_mpnn": "cyclicmpnn",
    "na_mpnn": "na_mpnn",
}

_SOURCE_SPEC_FIELDS = frozenset({"source", "count", "settings", "label"})
_SAFE_BRANCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _normalize_fixed_interface(value: Any) -> dict[str, Any] | None:
    """Use the shared reward/validation FI schema without model imports."""
    from src.design.rewards.interface import normalize_fixed_interface

    return normalize_fixed_interface(value)


def _source_branch(source: str, fixed_interface: Mapping[str, Any] | None) -> str:
    suffix = "_fi" if fixed_interface is not None and bool(
        fixed_interface.get("enabled")
    ) else ""
    return f"{source}{suffix}"


def _validate_branch_label(label: Any) -> str:
    if not isinstance(label, str) or not label or not _SAFE_BRANCH.fullmatch(label):
        raise ValueError(
            "sequence source label must be a nonempty safe branch string "
            "using letters, digits, '.', '_' or '-'"
        )
    if label in {".", ".."} or ".." in label:
        raise ValueError("sequence source label cannot contain '..'")
    return label


def normalize_sequence_sources(
    sources: Any,
    *,
    default_gt: bool = True,
) -> list[dict[str, Any]]:
    """Normalize source branches shared by config and generation entrypoints."""

    if sources is None or (default_gt and not sources):
        sources = ({"source": "gt", "count": 1},)
    if not isinstance(sources, (list, tuple)) or not sources:
        raise ValueError("sequence.sources must be a nonempty ordered list")
    result: list[dict[str, Any]] = []
    labels: set[str] = set()
    seen_gt = False
    for item in sources:
        if not isinstance(item, Mapping) or set(item).difference(_SOURCE_SPEC_FIELDS):
            raise ValueError(
                "Each source has source, count, optional label, and optional settings"
            )
        raw_name = item.get("source")
        source_name = str(raw_name).strip().lower()
        source = _SOURCE_ALIASES.get(source_name, source_name)
        fixed_interface_alias = source_name in _FI_SOURCE_ALIASES
        if source != "gt" and source not in _PUBLIC_MPNN:
            raise ValueError(
                f"Unknown sequence source {raw_name!r}; choose explicitly from "
                "gt, protein_mpnn, protein_mpnn-fi, proteinmpnn, proteinmpnn-fi, "
                "soluble_mpnn, soluble_mpnn-fi, solublempnn, solublempnn-fi, "
                "ligand_mpnn, ligand_mpnn-fi, ligandmpnn, ligandmpnn-fi, "
                "cyclic_mpnn, cyclic_mpnn-fi, cyclicmpnn, cyclicmpnn-fi, "
                "na_mpnn, na_mpnn-fi, nampnn, or nampnn-fi"
            )
        count = item.get("count", 1)
        if type(count) is not int or count < 1:
            raise ValueError("Sequence source count must be a positive integer")
        if source == "gt":
            if count != 1:
                raise ValueError("GT has exactly one original sequence")
            if seen_gt:
                raise ValueError("Duplicate GT source; self must be declared once")
            seen_gt = True
        raw_settings = item.get("settings", {})
        if not isinstance(raw_settings, Mapping):
            raise ValueError("source settings must be a mapping")
        settings = copy.deepcopy(dict(raw_settings))
        if source == "gt" and settings:
            raise ValueError("GT source does not accept redesign settings")
        fixed_interface = None
        if fixed_interface_alias and "fixed_interface" not in settings:
            fixed_interface = _normalize_fixed_interface({"enabled": True})
            settings["fixed_interface"] = copy.deepcopy(fixed_interface)
        elif "fixed_interface" in settings:
            fixed_interface = _normalize_fixed_interface(settings["fixed_interface"])
            if fixed_interface_alias and (
                fixed_interface is None or not fixed_interface["enabled"]
            ):
                raise ValueError(
                    f"Source alias {raw_name!r} requires fixed_interface.enabled=true; "
                    "an explicit disabled fixed interface conflicts with the -fi alias"
                )
            settings["fixed_interface"] = copy.deepcopy(fixed_interface)
        raw_label = item.get("label")
        label = (
            "self" if source == "gt" and raw_label is None
            else _source_branch(source, fixed_interface) if raw_label is None
            else raw_label
        )
        label = _validate_branch_label(label)
        if source == "gt" and label != "self":
            raise ValueError("GT source label must be exactly 'self'")
        if source != "gt" and label == "self":
            raise ValueError("'self' label is reserved for the original GT source")
        if label in labels:
            raise ValueError(f"Duplicate sequence source label {label!r}")
        labels.add(label)
        result.append({
            "source": source,
            "count": count,
            "label": label,
            "settings": settings,
        })
    if not seen_gt:
        result.insert(0, {
            "source": "gt", "count": 1, "label": "self", "settings": {},
        })
    return result


_GT_SOURCE = {"source": "gt", "count": 1, "label": "self", "settings": {}}
_DEFAULT_FOLD_SOURCES = (
    _GT_SOURCE,
    {"source": "soluble_mpnn", "count": 1, "label": "soluble_mpnn", "settings": {}},
)
_DEFAULT_NA_SOURCES = (
    _GT_SOURCE,
    {"source": "na_mpnn", "count": 1, "label": "na_mpnn", "settings": {}},
)
_GT_ONLY_METHODS = frozenset({"partiscore", "vina_dock"})


def infer_design_modality(input_record: Any) -> str | None:
    """Return the modality of the row-declared design, excluding context.

    Validation inputs carry the authoritative design modality in their CSV
    metadata when available.  Programmatic callers and older rows may lack
    that field, so fall back to the ``is_design`` atom annotation (or the
    declared design chains) and inspect only those rows.  This prevents a
    nucleic-acid or ligand context chain from changing a protein default.
    """

    raw = _field(input_record, "design_modality")
    if raw in (None, ""):
        raw = _row_field(input_record, "design_modality")
    if isinstance(raw, str):
        normalized = raw.strip().lower().replace("-", "_")
        aliases = {
            "protein": "protein",
            "peptide": "protein",
            "cyclic": "protein",
            "cyclic_peptide": "protein",
            "rna": "rna",
            "dna": "dna",
            "nucleic": "rna",
            "nucleic_acid": "rna",
            "ligand": "ligand",
        }
        if normalized in aliases:
            return aliases[normalized]

    atoms = _field(input_record, "atoms")
    if atoms is None or not hasattr(atoms, "mol_type"):
        return None
    import numpy as np

    modalities = np.asarray(atoms.mol_type).astype(str)
    design = getattr(atoms, "is_design", None)
    mask = None if design is None else np.asarray(design, dtype=bool)
    if mask is None or mask.shape != modalities.shape or not bool(mask.any()):
        declared = _field(input_record, "design_chains", ()) or ()
        if isinstance(declared, str):
            declared = (declared,)
        declared = {str(value).strip() for value in declared if str(value).strip()}
        if declared and hasattr(atoms, "chain_id"):
            mask = np.isin(np.asarray(atoms.chain_id).astype(str), sorted(declared))
    if mask is None or not bool(mask.any()):
        return None
    values = {str(value).strip().lower() for value in modalities[mask]}
    if len(values) != 1:
        return None
    value = next(iter(values))
    return value if value in {"protein", "rna", "dna", "ligand"} else None


def default_validation_sources(
    input_record: Any,
    *,
    method: str | None = None,
    configured_sources: Sequence[Mapping[str, Any]] | None = None,
    explicit: bool = False,
) -> list[dict[str, Any]]:
    """Resolve the per-input validation sequence policy.

    ``configured_sources`` is authoritative when ``explicit`` is true.  The
    default policy never exposes a validation ``auto`` source: protein and
    cyclic protein designs use SolubleMPNN, RNA/DNA designs use NA-MPNN, and
    ligand-only or GT-only methods retain only the original structure.
    """

    if explicit:
        if configured_sources is None:
            raise ValueError("explicit validation sources are missing")
        return normalize_sequence_sources(configured_sources, default_gt=False)
    if method in _GT_ONLY_METHODS:
        return [copy.deepcopy(_GT_SOURCE)]
    modality = infer_design_modality(input_record)
    if modality in {"rna", "dna"}:
        return [copy.deepcopy(item) for item in _DEFAULT_NA_SOURCES]
    if modality == "protein":
        return [copy.deepcopy(item) for item in _DEFAULT_FOLD_SOURCES]
    return [copy.deepcopy(_GT_SOURCE)]


# Keep a descriptive resolver alias for callers that prefer the policy name.
resolve_default_sequence_sources = default_validation_sources


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _row_field(value: Any, name: str, default: Any = None) -> Any:
    row = _field(value, "row", None)
    return _field(row, name, default) if row is not None else default


def _decode_jsonish(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _array(atoms: Any, name: str) -> Any:
    try:
        return getattr(atoms, name)
    except AttributeError as exc:
        raise SequenceGenerationError(
            f"validation input atoms lack required annotation {name!r}"
        ) from exc


def _chain_order(atoms: Any) -> tuple[str, ...]:
    result: list[str] = []
    for value in (str(item) for item in _array(atoms, "chain_id")):
        if value not in result:
            result.append(value)
    return tuple(result)


def _token_groups(atoms: Any, rows: Sequence[int]) -> list[tuple[Any, list[int]]]:
    import numpy as np

    token_values = getattr(atoms, "token_uid", None)
    if token_values is not None:
        values = np.asarray(token_values, dtype=np.int64)
        if bool((values < 0).any()):
            raise SequenceGenerationError(
                "validation input atoms lack a finalized token_uid ledger"
            )
        keys = sorted({int(values[int(row)]) for row in rows})
        return [
            (key, [int(row) for row in rows if int(values[int(row)]) == key])
            for key in keys
        ]
    residue_values = getattr(atoms, "res_id", None)
    if residue_values is None:
        residue_values = getattr(atoms, "label_seq_id", None)
    if residue_values is None:
        raise SequenceGenerationError(
            "validation input atoms lack token_uid and residue numbering"
        )
    insertion_values = getattr(atoms, "ins_code", None)
    groups: list[tuple[Any, list[int]]] = []
    seen: list[tuple[str, str]] = []
    for row in rows:
        key = (
            str(residue_values[int(row)]),
            "" if insertion_values is None else str(insertion_values[int(row)]),
        )
        if key not in seen:
            seen.append(key)
            groups.append((key, []))
        groups[seen.index(key)][1].append(int(row))
    return groups


def _initial_sequences(input_record: Any) -> dict[str, str]:
    import numpy as np

    atoms = _field(input_record, "atoms")
    if atoms is None:
        raise SequenceGenerationError("validation input has no parsed atoms")
    chains = np.asarray(_array(atoms, "chain_id")).astype(str)
    residues = np.asarray(_array(atoms, "res_name")).astype(str)
    modalities = np.asarray(_array(atoms, "mol_type")).astype(str)
    _, fixed_parents = _fixed_modified_map(input_record)
    residue_uids = getattr(atoms, "residue_uid", None)
    residue_uids = None if residue_uids is None else np.asarray(residue_uids, dtype=np.int64)
    maps = {"protein": _AA3_TO_1, "rna": _RNA3_TO_1, "dna": _DNA3_TO_1}
    result: dict[str, str] = {}
    for chain in _chain_order(atoms):
        rows = [int(row) for row in np.flatnonzero(chains == chain)]
        polymer_rows = [
            row for row in rows
            if str(modalities[row]).strip().lower() in maps
        ]
        kinds = {
            str(modalities[row]).strip().lower()
            for row in polymer_rows
        }
        if len(kinds) > 1:
            raise SequenceGenerationError(
                f"chain {chain!r} contains mixed modalities: {sorted(kinds)}"
            )
        if not polymer_rows:
            continue
        modality = next(iter(kinds))
        letters: list[str] = []
        for _, token_rows in _token_groups(atoms, polymer_rows):
            names = {str(residues[row]).strip().upper() for row in token_rows}
            if len(names) != 1:
                raise SequenceGenerationError(
                    f"chain {chain!r} has inconsistent residue identity"
                )
            name = next(iter(names))
            letter = maps[modality].get(name)
            if letter is None and residue_uids is not None:
                token_row = int(token_rows[0])
                parent = fixed_parents.get(int(residue_uids[token_row]))
                letter = maps[modality].get(parent)
            letters.append(letter if letter is not None else ("X" if modality == "protein" else "N"))
        result[chain] = "".join(letters)
    design = tuple(str(value) for value in (_field(input_record, "design_chains", ()) or ()))
    if not design:
        design_mask = np.asarray(getattr(atoms, "is_design"), dtype=bool)
        design = tuple(
            chain for chain in _chain_order(atoms)
            if bool(np.any(design_mask & (chains == chain)))
        )
    if len(design) == 1:
        row_sequence = _row_field(input_record, "design_sequence")
        if isinstance(row_sequence, str) and row_sequence.strip():
            sequence = row_sequence.strip().upper()
            if design[0] in result and len(sequence) == len(result[design[0]]):
                if result[design[0]] != sequence:
                    raise SequenceGenerationError(
                        f"result.csv design_sequence disagrees with parsed GT structure for chain {design[0]!r}"
                    )
    return result


def _normalize_fixed_positions(input_record: Any) -> tuple[str, ...]:
    value = _decode_jsonish(_field(input_record, "fixed_positions", ()), default=())
    if isinstance(value, Mapping):
        if "positions" in value:
            value = value["positions"]
        else:
            flattened: list[dict[str, Any]] = []
            for chain, values in value.items():
                if isinstance(values, Mapping):
                    values = [values]
                if not isinstance(values, (list, tuple, set)):
                    values = [values]
                for item in values:
                    if isinstance(item, Mapping):
                        flattened.append({"chain": chain, **dict(item)})
                    else:
                        flattened.append({"chain": chain, "position": item})
            value = flattened
    if value is None:
        return ()
    if not isinstance(value, (list, tuple, set)):
        value = (value,)
    result: list[str] = []
    for raw in value:
        if isinstance(raw, Mapping):
            chain = raw.get("chain", raw.get("chain_id"))
            position = raw.get(
                "position", raw.get("residue_index", raw.get("res_id"))
            )
            if chain is not None and position is not None:
                if "residue_index" in raw and "position" not in raw:
                    position = int(position) + 1
                result.append(f"{chain}:{int(position)}")
        elif str(raw).strip():
            result.append(str(raw).strip())
    return tuple(result)


def _bond_atom_uid(value: Any, atoms: Any = None) -> int | None:
    if isinstance(value, Mapping):
        for key in ("atom_uid", "uid"):
            if key in value:
                try:
                    return int(value[key])
                except (TypeError, ValueError):
                    return None
        if atoms is not None:
            import numpy as np
            if value.get("atom_id") not in (None, "", ".", "?"):
                try:
                    row = int(value["atom_id"]) - 1
                    if 0 <= row < len(atoms.atom_uid):
                        return int(atoms.atom_uid[row])
                except (TypeError, ValueError):
                    return None
            chains = np.asarray(
                getattr(atoms, "label_asym_id", getattr(atoms, "chain_id"))
            ).astype(str)
            mask = chains == str(value.get("chain", value.get("chain_id", "")))
            if value.get("res_id") not in (None, "", ".", "?"):
                mask &= np.asarray(atoms.res_id, dtype=np.int64) == int(value["res_id"])
            if value.get("ins_code") not in (None, "", ".", "?"):
                mask &= np.asarray(atoms.ins_code).astype(str) == str(value["ins_code"])
            if value.get("atom") not in (None, "", ".", "?"):
                mask &= np.asarray(atoms.atom_name).astype(str) == str(value["atom"])
            rows = np.flatnonzero(mask)
            if len(rows) == 1:
                return int(atoms.atom_uid[int(rows[0])])
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_MODIFIED_PARENTS = {
    "MSE": "MET", "SEP": "SER", "TPO": "THR", "PTR": "TYR", "HYP": "PRO",
    "CSO": "CYS", "CSD": "CYS", "DAL": "ALA", "DLE": "LEU", "DVA": "VAL",
    "DLY": "LYS", "DPN": "PHE", "DTH": "THR", "DTR": "TRP", "DSG": "ASN",
}
_STANDARD_COMPONENTS = (
    frozenset(name for name in _AA3_TO_1 if name not in _MODIFIED_PARENTS)
    | frozenset(_RNA3_TO_1)
    | frozenset(_DNA3_TO_1)
)
_PARENT_ALIASES = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _fixed_residue_modality(input_record: Any, residue_uid: int) -> str:
    import numpy as np

    atoms = _field(input_record, "atoms")
    if atoms is None or not hasattr(atoms, "residue_uid") or not hasattr(atoms, "mol_type"):
        return "protein"
    rows = np.flatnonzero(
        np.asarray(atoms.residue_uid, dtype=np.int64) == int(residue_uid)
    )
    if not len(rows):
        return "protein"
    modalities = {str(value).strip().lower() for value in np.asarray(atoms.mol_type)[rows]}
    if len(modalities) != 1 or next(iter(modalities)) not in {"protein", "rna", "dna"}:
        raise SequenceGenerationError(
            f"fixed modified residue UID {residue_uid} has an unsupported or ambiguous modality"
        )
    return next(iter(modalities))


def _authoritative_fixed_parent(
    input_record: Any,
    component: str,
    modality: str,
) -> str | None:
    """Resolve a fixed CCD parent through the compiler's CCD precedence."""

    from src.design.core.compiler import _configured_polymer_ccd, _resolve_proxy_parent
    from src.design.utils.ccd import load_embedded_ccd, load_packaged_polymer_ccd

    embedded: Mapping[str, Any] = {}
    source_value = _field(input_record, "path", _row_field(input_record, "path", None))
    if source_value not in (None, ""):
        source = Path(str(source_value)).expanduser()
        if source.is_file() and source.suffix.lower() in {".cif", ".mmcif"}:
            embedded = load_embedded_ccd(source)

    configured: Mapping[str, Any] = {}
    configured_value = _field(
        input_record,
        "ccd_path",
        _field(input_record, "polymer_ccd_path", _row_field(input_record, "ccd_path", None)),
    )
    if configured_value not in (None, ""):
        configured_path = Path(str(configured_value)).expanduser()
        if not configured_path.is_absolute():
            task_dir = _field(input_record, "task_dir", None)
            if task_dir not in (None, ""):
                configured_path = Path(str(task_dir)).expanduser() / configured_path
        if configured_path.is_file():
            configured = _configured_polymer_ccd(str(configured_path.resolve()))

    packaged = load_packaged_polymer_ccd()
    parent = _resolve_proxy_parent(
        component,
        modality,
        embedded=embedded,
        configured=configured,
        packaged=packaged,
    )
    unknown = {"protein": "UNK", "rna": "N", "dna": "DN"}[modality]
    return None if parent == unknown else str(parent).upper()


def _fixed_restype_records(input_record: Any) -> tuple[tuple[int, str, str | None], ...]:
    """Resolve row fixed-restype locators to the finalized residue ledger."""

    import numpy as np

    raw = _decode_jsonish(_field(input_record, "fixed_restypes", ()), default=())
    if isinstance(raw, Mapping):
        raw = raw.get("residues", raw)
    if isinstance(raw, Mapping):
        raw = (raw,)
    if not isinstance(raw, (list, tuple)):
        raw = ()
    atoms = _field(input_record, "atoms")
    design_chains = set(_field(input_record, "design_chains", ()) or ())
    if raw and not design_chains and atoms is not None and hasattr(atoms, "is_design"):
        chain_values = np.asarray(getattr(atoms, "label_asym_id", atoms.chain_id))
        design_chains = {
            str(chain) for chain in chain_values[np.asarray(atoms.is_design, dtype=bool)]
        }
    result: list[tuple[int, str, str | None]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        component = item.get("component_id", item.get("res_name", item.get("restype")))
        if component in (None, "", ".", "?"):
            component = None
        else:
            component = str(component).strip().upper()
        raw_uid = item.get("residue_uid", item.get("res_uid", item.get("uid")))
        residue_uid: int | None = None
        if raw_uid not in (None, "", ".", "?"):
            try:
                residue_uid = int(raw_uid)
            except (TypeError, ValueError) as exc:
                raise SequenceGenerationError(
                    f"fixed_restypes[{index}] has an invalid residue_uid"
                ) from exc
        if residue_uid is None:
            if atoms is None or not hasattr(atoms, "residue_uid"):
                raise SequenceGenerationError(
                    f"fixed_restypes[{index}] needs a residue_uid or a resolvable structure locator"
                )
            chain = str(item.get("chain", item.get("label_asym_id", ""))).strip()
            if not chain or item.get("res_id") in (None, "", ".", "?"):
                raise SequenceGenerationError(
                    f"fixed_restypes[{index}] lacks chain/res_id for residue resolution"
                )
            try:
                res_id = int(item["res_id"])
            except (TypeError, ValueError) as exc:
                raise SequenceGenerationError(
                    f"fixed_restypes[{index}] has a non-integer res_id"
                ) from exc
            chains = np.asarray(
                getattr(atoms, "label_asym_id", getattr(atoms, "chain_id"))
            ).astype(str)
            mask = chains == chain
            mask &= np.asarray(atoms.res_id, dtype=np.int64) == res_id
            if item.get("ins_code") not in (None, "", ".", "?"):
                mask &= np.asarray(atoms.ins_code).astype(str) == str(item["ins_code"])
            rows = np.flatnonzero(mask)
            uids = {int(value) for value in np.asarray(atoms.residue_uid)[rows].tolist()}
            if len(uids) != 1:
                raise SequenceGenerationError(
                    f"fixed_restypes[{index}] locator resolves to {len(uids)} residues"
                )
            residue_uid = next(iter(uids))
        if atoms is not None and hasattr(atoms, "residue_uid"):
            rows = np.flatnonzero(
                np.asarray(atoms.residue_uid, dtype=np.int64) == residue_uid
            )
            if not len(rows):
                raise SequenceGenerationError(
                    f"fixed_restypes[{index}] residue_uid {residue_uid} is absent from structure"
                )
            # Legacy rows may include receptor entries. fixed_restypes only
            # constrains design residues; exclude context before CCD lookup.
            if design_chains:
                chain_values = getattr(atoms, "label_asym_id", atoms.chain_id)
                if not any(str(chain_values[row]) in design_chains for row in rows):
                    continue
            actual = {str(value).strip().upper() for value in np.asarray(atoms.res_name)[rows]}
            if len(actual) != 1:
                raise SequenceGenerationError(
                    f"fixed_restypes[{index}] residue_uid {residue_uid} has ambiguous residue identity"
                )
            actual_name = next(iter(actual))
            if component is None:
                component = actual_name
            elif component != actual_name:
                raise SequenceGenerationError(
                    f"fixed_restypes[{index}] restype {component!r} disagrees with structure residue {actual_name!r}"
                )
        if component is None:
            raise SequenceGenerationError(
                f"fixed_restypes[{index}] has no component/restype identity"
            )
        parent = item.get("parent", item.get("standard_parent"))
        if parent in (None, "", ".", "?"):
            parent = _MODIFIED_PARENTS.get(component)
        elif isinstance(parent, str):
            parent = _PARENT_ALIASES.get(parent.strip().upper(), parent.strip().upper())
        else:
            parent = str(parent).strip().upper()
        result.append((int(residue_uid), component, parent))
    return tuple(result)


def _fixed_modified_map(input_record: Any) -> tuple[dict[int, str], dict[int, str]]:
    names: dict[int, str] = {}
    parents: dict[int, str] = {}
    for residue_uid, component, parent in _fixed_restype_records(input_record):
        if component in _STANDARD_COMPONENTS:
            continue
        modality = _fixed_residue_modality(input_record, residue_uid)
        resolved_parent = _authoritative_fixed_parent(input_record, component, modality)
        if resolved_parent is not None:
            if parent is not None and parent != resolved_parent:
                raise SequenceGenerationError(
                    f"fixed modified residue {component!r} parent {parent!r} "
                    f"disagrees with CCD parent {resolved_parent!r}"
                )
            parent = resolved_parent
        parent_map = {
            "protein": _AA3_TO_1,
            "rna": _RNA3_TO_1,
            "dna": _DNA3_TO_1,
        }[modality]
        if parent is None or parent not in parent_map:
            raise SequenceGenerationError(
                f"fixed modified residue {component!r} lacks a canonical {modality} parent"
            )
        names[residue_uid] = component
        parents[residue_uid] = parent
    return names, parents


def _bond_order_name(value: Any) -> str:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None:
        return {1: "single", 2: "double", 3: "triple",
                4: "aromatic", 5: "aromatic", 6: "aromatic",
                7: "aromatic", 9: "aromatic"}.get(numeric, "?")
    order = str(value).lower()
    order = {"sing": "single", "doub": "double", "trip": "triple",
             "arom": "aromatic", "any": "?", "unknown": "?"}.get(order, order)
    return order if order in {"?", "single", "double", "triple", "aromatic"} else "?"


def _graph_edge_is_template(input_record: Any, left: int, right: int,
                            modified_uids: set[int]) -> bool:
    import numpy as np

    atoms = _field(input_record, "atoms")
    tokens = np.asarray(getattr(atoms, "token_uid"), dtype=np.int64)
    if int(tokens[left]) == int(tokens[right]):
        return True
    modalities = np.asarray(atoms.mol_type).astype(str)
    chains = np.asarray(
        getattr(atoms, "label_asym_id", getattr(atoms, "chain_id"))
    ).astype(str)
    if modalities[left] == modalities[right] == "ligand":
        molecules = getattr(atoms, "molecule_uid", None)
        components = getattr(atoms, "component_uid", None)
        if molecules is not None and int(molecules[left]) == int(molecules[right]):
            same_component = (
                components is None or int(components[left]) == int(components[right])
            )
            ligands = _field(input_record, "ligand_entries", {}) or {}
            row_sequence = _row_field(input_record, "design_sequence", "")
            has_smiles = (
                isinstance(ligands, Mapping)
                and any(isinstance(value, Mapping) and value.get("smiles")
                        for value in ligands.values())
            ) or bool(isinstance(row_sequence, str) and row_sequence.strip())
            if same_component or has_smiles:
                return True
    if modalities[left] != modalities[right] or "ligand" in {
        modalities[left], modalities[right]
    }:
        return False
    if chains[left] != chains[right] or abs(int(tokens[left]) - int(tokens[right])) != 1:
        return False
    residue_uids = getattr(atoms, "residue_uid", tokens)
    if int(residue_uids[left]) in modified_uids or int(residue_uids[right]) in modified_uids:
        return False
    atom_names = {str(atoms.atom_name[left]), str(atoms.atom_name[right])}
    if modalities[left] == "protein":
        return atom_names == {"C", "N"}
    if modalities[left] in {"rna", "dna"}:
        return atom_names == {"O3'", "P"}
    return False


def _graph_topology_conditions(input_record: Any, TopologyCondition: Any) -> tuple[Any, ...]:
    atoms = _field(input_record, "atoms")
    bonds = getattr(atoms, "bonds", None)
    if bonds is None:
        return ()
    if not hasattr(atoms, "atom_uid"):
        raise SequenceGenerationError("source bond graph lacks finalized atom_uid identities")
    modified_uids = set(_fixed_modified_map(input_record)[0])
    result: list[Any] = []
    for left, right, order in bonds.as_array().tolist():
        left, right = int(left), int(right)
        if _graph_edge_is_template(input_record, left, right, modified_uids):
            continue
        result.append(TopologyCondition(
            atom_uid_a=int(atoms.atom_uid[left]),
            atom_uid_b=int(atoms.atom_uid[right]),
            order=_bond_order_name(order),
            required_output=True,
        ))
    return tuple(result)


def _topology_conditions(input_record: Any) -> tuple[Any, ...]:
    try:
        from src.design.core.records import TopologyCondition
    except (ImportError, ModuleNotFoundError):
        return ()
    by_pair: dict[tuple[int, int], Any] = {}

    def add(condition: Any) -> None:
        pair = condition.canonical_uids()
        previous = by_pair.get(pair)
        if previous is None:
            by_pair[pair] = condition
            return
        previous_order, current_order = str(previous.order), str(condition.order)
        if previous_order != "?" and current_order != "?" and previous_order != current_order:
            raise SequenceGenerationError(
                f"conflicting bond orders for atom pair {pair}: {previous_order!r} versus {current_order!r}"
            )
        order = previous_order if previous_order != "?" else current_order
        by_pair[pair] = TopologyCondition(
            atom_uid_a=pair[0], atom_uid_b=pair[1], order=order,
            required_output=bool(previous.required_output or condition.required_output),
        )

    supplied = _field(input_record, "topology_conditions")
    for condition in supplied or ():
        if isinstance(condition, TopologyCondition):
            add(condition)
    raw = _decode_jsonish(_field(input_record, "bond_constraints", ()), default=())
    if isinstance(raw, Mapping):
        raw = raw.get("bonds", raw.get("constraints", ()))
    atoms = _field(input_record, "atoms")
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            left = item.get("atom_uid_a", item.get("endpoint1", item.get("atom1")))
            right = item.get("atom_uid_b", item.get("endpoint2", item.get("atom2")))
            atom_uid_a, atom_uid_b = _bond_atom_uid(left, atoms), _bond_atom_uid(right, atoms)
            if atom_uid_a is None or atom_uid_b is None or atom_uid_a == atom_uid_b:
                continue
            add(TopologyCondition(
                atom_uid_a=atom_uid_a, atom_uid_b=atom_uid_b,
                order=_bond_order_name(item.get("order", item.get("bond_order", item.get("type", "?")))),
                required_output=bool(item.get("required_output", True)),
            ))
    for condition in _graph_topology_conditions(input_record, TopologyCondition):
        add(condition)
    return tuple(by_pair.values())



def _context_modified_map(
    input_record: Any,
    atoms: Any,
) -> tuple[dict[int, str], dict[int, str]]:
    """Build private canonical proxies for nonstandard receptor polymers.

    fixed_restypes is a design-peptide contract. Receptor residues are
    immutable context and use owner-local ALA/GLY placeholders without CCD
    parent lookup; the public row and result metadata are never changed.
    """
    import numpy as np

    if atoms is None or not hasattr(atoms, "residue_uid"):
        return {}, {}
    residue_uids = np.asarray(atoms.residue_uid, dtype=np.int64)
    residue_names = np.asarray(atoms.res_name).astype(str)
    modalities = np.asarray(atoms.mol_type).astype(str)
    design = np.asarray(
        getattr(atoms, "is_design", np.zeros(len(atoms), dtype=bool)),
        dtype=bool,
    )
    names: dict[int, str] = {}
    parents: dict[int, str] = {}
    for residue_uid in sorted({int(value) for value in residue_uids.tolist()}):
        rows = np.flatnonzero(residue_uids == residue_uid)
        if not len(rows) or bool(design[rows].any()):
            continue
        kinds = {str(value).strip().lower() for value in modalities[rows]}
        if kinds != {"protein"}:
            continue
        components = {str(value).strip().upper() for value in residue_names[rows]}
        if len(components) != 1:
            raise SequenceGenerationError(
                f"context residue UID {residue_uid} has ambiguous component identity"
            )
        component = next(iter(components))
        if component in _STANDARD_COMPONENTS:
            continue
        # PipeTools MPNN requires a canonical context parent. Use a private
        # ALA/GLY placeholder without scanning CCD parent tables.
        names[residue_uid] = component
        parents[residue_uid] = "ALA" if "CB" in set(np.asarray(atoms.atom_name)[rows]) else "GLY"
    return names, parents


def _candidate_adapter(
    input_record: Any,
    candidate_id: str,
    candidate_ordinal: int = 0,
) -> Any:
    cyclic_offset_chains = _input_cyclic_offset_chains(input_record)
    existing = _field(input_record, "candidate")
    if existing is not None:
        # A loaded validation row normally has no SearchCandidate yet.  When
        # a caller supplies one, make a shallow owner-local copy so batched
        # slots still carry distinct IDs/ordinals into native workers.
        try:
            result = copy.copy(existing)
            result.candidate_id = candidate_id
            result.candidate_ordinal = int(candidate_ordinal)
            return _propagate_cyclic_offset_intent(result, cyclic_offset_chains)
        except (AttributeError, TypeError):
            return existing
    import numpy as np

    names, parents = _fixed_modified_map(input_record)
    atoms = _mpnn_atoms(input_record)
    design_uids = {
        int(value)
        for value in np.asarray(atoms.residue_uid, dtype=np.int64)[
            np.asarray(atoms.is_design, dtype=bool)
        ].tolist()
    } if atoms is not None and hasattr(atoms, "residue_uid") else set()
    # Keep fixed_restypes scoped to design polymer residues. Receptor
    # nonstandard identities are added only to this candidate-local proxy map.
    names = {uid: value for uid, value in names.items() if uid in design_uids}
    parents = {uid: value for uid, value in parents.items() if uid in design_uids}
    context_names, context_parents = _context_modified_map(input_record, atoms)
    names.update(context_names)
    parents.update(context_parents)
    if context_names and atoms is not None and hasattr(atoms, "identity_condition"):
        identity = np.asarray(atoms.identity_condition, dtype=bool).copy()
        residue_uids = np.asarray(atoms.residue_uid, dtype=np.int64)
        identity |= np.isin(residue_uids, list(context_names))
        atoms.identity_condition = identity
    return SimpleNamespace(
        candidate_id=candidate_id, candidate_ordinal=int(candidate_ordinal),
        base_request=SimpleNamespace(task=SimpleNamespace(
            condition={}, format=(
                "antibody" if _row_field(input_record, "heavy_role", "") else "normal"
            ),
            cyclic_offset=bool(cyclic_offset_chains),
            cyclic_offset_chains=cyclic_offset_chains,
        )),
        actual_joint_length=(), atoms=atoms,
        topology_conditions=_topology_conditions(input_record),
        private_ligand_molecules={}, private_ligand_errors={},
        ligand_fragment_metadata={}, motif_output_fields={},
        model_parent_residue={}, fixed_modified_residue_names=names,
        fixed_modified_parent_residue=parents, validated_ligand_identities={},
    )


def _restore_antibody_mpnn_annotations(input_record: Any, atoms: Any) -> None:
    """Restore final-CSV antibody roles and explicit mutable positions privately."""
    import numpy as np
    row = _field(input_record, "row", {})
    if not isinstance(row, Mapping) or "heavy_role" not in row:
        return
    from src.validate.inputs import _antibody_chain_metadata

    # Use the same boundary validation as load_input for programmatic callers.
    _antibody_chain_metadata(dict(row), atoms, "antibody MPNN input")
    labels = np.asarray(getattr(atoms, "label_asym_id", atoms.chain_id)).astype(str)
    positions = np.asarray(atoms.res_id, dtype=np.int64)
    roles = np.zeros(len(atoms), dtype=np.int64)
    mutable = np.zeros(len(atoms), dtype=bool)
    for prefix, role in (("heavy", 1), ("light", 2)):
        chain = row.get(f"{prefix}_chain", "")
        if not chain:
            continue
        rows = labels == str(chain)
        roles[rows] = role
        selected: set[int] = set()
        for interval in row.get(f"{prefix}_design_positions", "").split(","):
            if interval:
                bounds = interval.split("-")
                start, end = int(bounds[0]), int(bounds[-1])
                selected.update(range(start, end + 1))
        mutable |= rows & np.isin(positions, list(selected))
    atoms.set_annotation("antibody_chain_role_id", roles)
    atoms.set_annotation("antibody_sequence_mutable", mutable)


def _validation_antibody_plans(items: Sequence[tuple[Any, Any]]) -> dict[str, Any]:
    """Build explicit physical-chain plans, retaining FI/user-fixed residues."""
    from src.design.rewards.mpnn import build_antibody_mpnn_plan

    plans = {}
    for candidate, sequence_plan in items:
        task = _field(_field(candidate, "base_request"), "task")
        if _field(task, "format", "normal") != "antibody":
            continue
        plans[candidate.candidate_id] = build_antibody_mpnn_plan(
            candidate, sequence_plan=sequence_plan,
        )
    return plans


def _mpnn_atoms(input_record: Any) -> Any:
    """Copy parsed atoms and restore the row-declared design-chain mask.

    ``read_structure`` deliberately returns a source-faithful AtomArray with
    no inference-time design annotations.  The reward MPNN adapter consumes
    ``is_design`` to distinguish the sequence to redesign from context, so
    validation must derive that mask from ``result.csv`` rather than guessing
    a chain from atom order.  The copy keeps the input's coordinates and
    explicit bond graph intact for the caller and for topology conditions.
    """
    import numpy as np

    atoms = _field(input_record, "atoms")
    if atoms is None:
        return None
    try:
        result = atoms.copy()
    except AttributeError:
        result = copy.deepcopy(atoms)
    declared = {
        str(value).strip()
        for value in (_field(input_record, "design_chains", ()) or ())
        if str(value).strip()
    }
    categories = (
        set(result.get_annotation_categories())
        if hasattr(result, "get_annotation_categories")
        else {
            name for name in ("is_design", "identity_condition")
            if hasattr(result, name)
        }
    )
    if declared:
        chain_ids = np.asarray(getattr(result, "chain_id"), dtype=str)
        mask = np.isin(chain_ids, list(declared))
        label_ids = getattr(result, "label_asym_id", None)
        if label_ids is not None:
            mask |= np.isin(np.asarray(label_ids, dtype=str), list(declared))
        if "is_design" in categories:
            result.is_design = np.asarray(mask, dtype=bool)
        else:
            if hasattr(result, "set_annotation"):
                result.set_annotation("is_design", np.asarray(mask, dtype=bool))
            else:
                result.is_design = np.asarray(mask, dtype=bool)
    fixed_uids = {uid for uid, _component, _parent in _fixed_restype_records(input_record)}
    if fixed_uids and hasattr(result, "residue_uid") and hasattr(result, "is_design"):
        residue_uids = np.asarray(result.residue_uid, dtype=np.int64)
        design_mask = np.asarray(result.is_design, dtype=bool)
        fixed_uids = {
            uid for uid in fixed_uids
            if bool(np.any(design_mask & (residue_uids == int(uid))))
        }
    if fixed_uids and hasattr(result, "residue_uid"):
        identity = (
            np.asarray(result.identity_condition, dtype=bool).copy()
            if "identity_condition" in categories
            else np.zeros(len(result), dtype=bool)
        )
        identity |= np.isin(
            np.asarray(result.residue_uid, dtype=np.int64),
            list(fixed_uids),
        )
        if "identity_condition" in categories:
            result.identity_condition = identity
        else:
            if hasattr(result, "set_annotation"):
                result.set_annotation("identity_condition", identity)
            else:
                result.identity_condition = identity
    _restore_antibody_mpnn_annotations(input_record, result)
    return result



def _restore_source_chain_sequences(
    input_record: Any,
    sequences: Mapping[str, str],
) -> dict[str, str]:
    """Collapse owner-local receptor break aliases back to source chain IDs."""
    import re

    result = {str(key): str(value) for key, value in sequences.items()}
    atoms = _field(input_record, "atoms")
    if atoms is None:
        return result
    source_chains = _chain_order(atoms)
    for chain in source_chains:
        prefix = f"{chain}__break"
        keys = [key for key in result if key == chain or key.startswith(prefix)]
        if len(keys) <= 1:
            continue
        keys.sort(key=lambda key: 0 if key == chain else int(re.search(r"__break(\d+)$", key).group(1)))
        result[chain] = "".join(result.pop(key) for key in keys)
    return result


def _input_cyclic_offset_chains(input_record: Any) -> tuple[str, ...]:
    raw = _field(input_record, "cyclic_offset_chains", ())
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = (raw,)
    else:
        try:
            raw = tuple(raw)
        except TypeError:
            return ()
    return tuple(dict.fromkeys(
        str(value).strip() for value in raw if str(value).strip()
    ))


def _propagate_cyclic_offset_intent(
    candidate: Any,
    cyclic_offset_chains: tuple[str, ...],
) -> Any:
    if not cyclic_offset_chains:
        return candidate
    request = getattr(candidate, "base_request", None)
    task = getattr(request, "task", None)
    if request is None or task is None:
        return candidate

    if is_dataclass(task) and {
        field.name for field in dataclass_fields(task)
    } >= {"cyclic_offset", "cyclic_offset_chains"}:
        task = dataclass_replace(
            task,
            cyclic_offset=True,
            cyclic_offset_chains=cyclic_offset_chains,
        )
    else:
        task = copy.copy(task)
        task.cyclic_offset = True
        task.cyclic_offset_chains = cyclic_offset_chains
    if is_dataclass(request):
        request = dataclass_replace(request, task=task)
    else:
        request = copy.copy(request)
        request.task = task
    candidate.base_request = request
    return candidate


def _backend_defaults(repo_root: Path, public_source: str) -> dict[str, Any]:
    if public_source in {"protein_mpnn", "soluble_mpnn"}:
        model_type = "protein_mpnn" if public_source == "protein_mpnn" else "soluble_mpnn"
        family = "vanilla" if model_type == "protein_mpnn" else "soluble"
        return {
            "model_type": model_type,
            "checkpoint": str(community_asset("proteinmpnn", repo_root) / "checkpoint" / family / "v_48_020.pt"),
            "temperature": 0.0001 if public_source == "protein_mpnn" else 0.1,
            "backbone_noise": 0.0, "omit_aas": "", "rm_aa": "C", "bias_aa": {},
            "pssm_path": None, "pssm_multi": 0.0, "pssm_threshold": 0.0,
            "pssm_log_odds": False, "pssm_bias": False, "tied_positions_path": None,
            "ca_only": False, "use_soluble_model": model_type == "soluble_mpnn",
            "seed": 101,
        }
    if public_source == "ligand_mpnn":
        return {
            "model_type": "ligand_mpnn",
            "checkpoint": str(community_asset("ligandmpnn", repo_root) / "checkpoint" / "ligandmpnn_v_32_010_25.pt"),
            "temperature": 0.1, "backbone_noise": 0.0, "omit_aas": "", "rm_aa": "C", "bias_aa": {},
            "use_ligand_atoms": True, "use_sidechain_context": True,
            "pack_side_chains": False, "repack_everything": False,
            "homo_oligomer": False, "symmetry_residues_path": None,
            "symmetry_weights_path": None, "seed": 101,
        }
    if public_source == "cyclic_mpnn":
        return {
            "model_type": "cyclicmpnn",
            "checkpoint": str(community_asset("cyclicmpnn", repo_root) / "checkpoint" / "cyclicmpnn_48_010.pkl"),
            "base_model": "v_48_020", "temperature": 0.0001, "backbone_noise": 0.0,
            "omit_aas": "", "rm_aa": "C", "bias_aa": {}, "seed": 101,
        }
    if public_source == "na_mpnn":
        return {
            "model_type": "na_mpnn",
            "checkpoint": str(community_asset("na_mpnn", repo_root) / "checkpoint" / "s_19137.pt"),
            "temperature": 0.1, "rm_aa": "C", "seed": 101,
        }
    raise ValueError(f"unsupported validation sequence source {public_source!r}")


class SequenceGenerator:
    """Generate declared validation sequence variants in stable order."""

    def __init__(self, repo_root: Path, config: Mapping[str, Any] | None = None, device: Any = None):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.config = dict(config or {})
        self.device = device

    def close(self) -> None:
        """Release resident MPNN workers owned by this validation generator."""
        from src.design.rewards.mpnn_cache import close_cached_workers

        close_cached_workers()

    @staticmethod
    def _normalize_sources(sources: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
        return normalize_sequence_sources(sources)

    def _sequence_branch(
        self,
        public_source: str,
        source_settings: Mapping[str, Any] | None,
    ) -> str:
        current = getattr(self, "_sequence_branch_context", None)
        if current is not None:
            return str(current)
        fixed_interface = None
        if isinstance(source_settings, Mapping):
            fixed_interface = source_settings.get("fixed_interface")
        return _source_branch(public_source, fixed_interface)

    @staticmethod
    def _plan_source(
        planner: Any,
        candidate: Any,
        internal_source: str,
        fixed_positions: Sequence[str],
        fixed_interface: Mapping[str, Any] | None,
    ) -> Any:
        """Forward FI to the current shared planner; unsupported APIs fail."""
        return planner(
            candidate, requested=internal_source,
            user_fixed_positions=fixed_positions,
            fixed_interface=fixed_interface,
        )

    def _mpnn_seed(self, public_source, source_settings):
        sequence = self.config.get("sequence", {})
        internal = _INTERNAL_SOURCES[public_source]
        backend = sequence.get(public_source, sequence.get(internal))
        if backend is None:
            backend = self.config.get(public_source, self.config.get(internal, {}))
        backend = backend if isinstance(backend, Mapping) else {}
        return source_settings.get("seed", backend.get("seed", sequence.get("seed", 101)))

    def _mpnn_config(
        self,
        public_source: str,
        source_settings: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        path_base = Path.cwd() if ENV_MODEL_DIR in os.environ else self.repo_root
        sequence_config = dict(self.config.get("sequence", {}))
        internal_source = _INTERNAL_SOURCES[public_source]
        raw = sequence_config.get(public_source, sequence_config.get(internal_source))
        if raw is None:
            raw = self.config.get(public_source, self.config.get(internal_source))
        backend = _backend_defaults(self.repo_root, public_source)
        if isinstance(raw, Mapping):
            backend.update({
                key: copy.deepcopy(value)
                for key, value in raw.items()
                if key != "fixed_interface"
            })
            # A backend-local legacy omit_aas must override the default Cys
            # alias unless that same layer supplies rm_aa explicitly.
            if "omit_aas" in raw and "rm_aa" not in raw:
                backend["rm_aa"] = None
        if isinstance(source_settings, Mapping):
            allowed = {
                "temperature", "backbone_noise", "omit_aas", "rm_aa", "bias_aa",
                "pssm_multi", "pssm_threshold", "pssm_log_odds", "pssm_bias",
                "seed", "checkpoint", "pssm_path", "tied_positions_path",
                "model_type", "base_model", "base_model_name", "use_soluble_model",
                "ca_only", "use_ligand_atoms", "use_sidechain_context",
                "pack_side_chains", "repack_everything", "homo_oligomer",
                "symmetry_residues_path", "symmetry_weights_path", "asset_root",
            }
            native_settings = {
                key: value for key, value in source_settings.items()
                if key != "fixed_interface"
            }
            unknown = sorted(set(native_settings).difference(allowed))
            if unknown:
                raise ValueError(
                    f"unsupported {public_source} settings: {', '.join(map(str, unknown))}"
                )
            backend.update({
                key: copy.deepcopy(value)
                for key, value in native_settings.items()
                if key in allowed
            })
            # A source-local legacy omit_aas is itself an explicit local
            # policy.  Do not let an inherited backend rm_aa silently mask it;
            # a source-local rm_aa (including "" or null) remains canonical
            # whenever it is present.
            if "omit_aas" in native_settings and "rm_aa" not in native_settings:
                backend["rm_aa"] = None
        if internal_source == "proteinmpnn":
            backend["model_type"] = (
                "protein_mpnn" if public_source == "protein_mpnn" else "soluble_mpnn"
            )
            checkpoint = Path(str(backend["checkpoint"])).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = path_base / checkpoint
            backend["checkpoint"] = str(checkpoint)
            for key in ("pssm_path", "tied_positions_path"):
                value = backend.get(key)
                if value:
                    path = Path(str(value)).expanduser()
                    backend[key] = str(path if path.is_absolute() else path_base / path)
            if backend["model_type"] == "soluble_mpnn":
                backend["use_soluble_model"], backend["ca_only"] = True, False
        else:
            backend["model_type"] = _backend_defaults(self.repo_root, public_source)["model_type"]
            if public_source == "cyclic_mpnn":
                backend.setdefault("base_model", "v_48_020")
            for key in ("checkpoint", "pssm_path", "tied_positions_path",
                        "symmetry_residues_path", "symmetry_weights_path", "asset_root"):
                value = backend.get(key)
                if value:
                    path = Path(str(value)).expanduser()
                    backend[key] = str(path if path.is_absolute() else path_base / path)
        if "checkpoint" in backend:
            checkpoint = Path(str(backend["checkpoint"])).expanduser()
            backend["checkpoint"] = str(
                checkpoint if checkpoint.is_absolute() else path_base / checkpoint
            )
        # Resolve the unified setting only in this private copy.  Keeping the
        # source/backend mappings untouched preserves caller configuration and
        # lets an explicit source-local empty rm_aa disable legacy omit_aas.
        backend["omit_aas"] = resolve_rm_aa(backend)
        return {"sequence": {**sequence_config, internal_source: backend}}

    @staticmethod
    def _fi_covers_all_design_positions(candidate: Any, plan: Any, fixed_interface: Any) -> bool:
        """An all-fixed interface request has no sequence sampling degrees of freedom."""
        if not isinstance(fixed_interface, Mapping) or not fixed_interface.get("enabled", False):
            return False
        from src.design.rewards.esmfold2 import polymer_tokens, protein_tokens

        tokenizer = polymer_tokens if plan.source == "na_mpnn" else protein_tokens
        designed = {(token.chain_id, int(token.position))
                    for token in tokenizer(candidate.atoms) if token.is_design}
        fixed = {(str(chain), int(position)) for chain, position in plan.fixed_positions}
        return bool(designed) and designed.issubset(fixed)

    @staticmethod
    def _fi_passthrough(input_record: Any, plan: Any, public_source: str,
                        sequence_branch: str, fixed_interface: Any,
                        raw_seed: int, effective_backend: Any):
        sequences = _restore_source_chain_sequences(input_record, plan.chain_sequences)
        return sequences, {
            "backend": "identity_passthrough", "backend_source": "gt",
            "requested_backend": public_source, "sequence_source": public_source,
            "sequence_branch": sequence_branch, "sampled": False,
            "reason": "fixed_interface_covers_all_design_positions",
            "fixed_interface": fixed_interface,
            "fixed_positions": list(plan.fixed_positions),
            "interface_positions": _jsonable(getattr(plan, "interface_positions", None)),
            "interface_target_chains": _jsonable(getattr(plan, "interface_target_chains", None)),
            "effective_settings": copy.deepcopy(effective_backend),
            "seed": raw_seed, "sequence_seed": raw_seed,
        }, None

    def _generate_mpnn(
        self, input_record: Any, public_source: str, source_index: int,
        work_dir: Path,
        fixed_positions: Sequence[str],
        source_settings: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, str] | None, dict[str, Any], str | None]:
        from src.design.rewards.esmfold2 import plan_sequence_source
        from src.design.rewards.mpnn import MPNNSequenceError, MPNNSystemError
        from src.validate.mpnn import (
            generate_native_mpnn_sequences,
            generate_protein_mpnn_sequences,
        )

        sample_id = _field(input_record, "sample_id", _row_field(input_record, "sample_id", "sample"))
        effective_settings = dict(source_settings or {})
        sequence_branch = self._sequence_branch(public_source, effective_settings)
        fixed_interface = _normalize_fixed_interface(
            effective_settings.get("fixed_interface")
        ) if "fixed_interface" in effective_settings else None
        raw_seed = self._mpnn_seed(public_source, effective_settings)
        candidate_id = f"{sample_id}__{sequence_branch}__{public_source}__{source_index}"
        candidate = _candidate_adapter(input_record, candidate_id, source_index)
        if candidate.atoms is None:
            return None, {
                "backend": public_source,
                "sequence_branch": sequence_branch,
                "fixed_interface": fixed_interface,
                "sequence_seed": raw_seed,
            }, "MPNN input has no AtomArray"
        internal_source = _INTERNAL_SOURCES[public_source]
        backend = self._mpnn_config(public_source, effective_settings)["sequence"][internal_source]
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            return None, {
                "backend": public_source,
                "sequence_branch": sequence_branch,
                "fixed_interface": fixed_interface,
                "sequence_seed": raw_seed,
            }, "MPNN seed must be a non-negative integer"
        if raw_seed < 0:
            return None, {
                "backend": public_source,
                "sequence_branch": sequence_branch,
                "fixed_interface": fixed_interface,
                "sequence_seed": raw_seed,
            }, "MPNN seed must be a non-negative integer"
        # Every MPNN source draws a batch from one shared seed.
        effective_settings["seed"] = raw_seed
        try:
            plan = self._plan_source(
                plan_sequence_source,
                candidate,
                internal_source,
                fixed_positions,
                fixed_interface,
            )
            reward_config = self._mpnn_config(public_source, effective_settings)
            effective_backend = reward_config["sequence"][internal_source]
            if self._fi_covers_all_design_positions(candidate, plan, fixed_interface):
                return self._fi_passthrough(
                    input_record, plan, public_source, sequence_branch,
                    fixed_interface, raw_seed, effective_backend,
                )
            antibody_plans = _validation_antibody_plans([(candidate, plan)])
            if internal_source == "proteinmpnn" and not antibody_plans:
                generated, failures = generate_protein_mpnn_sequences(
                    [(candidate, plan)], reward_config=reward_config,
                    repo_root=self.repo_root, work_dir=work_dir,
                    seed=raw_seed,
                )
            else:
                generated, failures = generate_native_mpnn_sequences(
                    [(candidate, plan)], reward_config=reward_config,
                    repo_root=self.repo_root, work_dir=work_dir,
                    seed=raw_seed,
                    **({"antibody_plans": antibody_plans} if antibody_plans else {}),
                )
        except (MPNNSequenceError, SequenceGenerationError, ValueError, KeyError, OSError) as exc:
            return None, {
                "backend": public_source,
                "sequence_branch": sequence_branch,
                "fixed_interface": fixed_interface,
                "sequence_seed": raw_seed,
            }, str(exc)
        except MPNNSystemError:
            raise
        if candidate_id in failures:
            return None, {
                "backend": public_source,
                "sequence_source": public_source,
                "sequence_branch": sequence_branch,
                "fixed_interface": fixed_interface,
                "seed": raw_seed,
                "sequence_seed": raw_seed,
            }, str(failures[candidate_id])
        generated_plan = generated.get(candidate_id)
        if generated_plan is None:
            return None, {
                "backend": public_source,
                "sequence_source": public_source,
                "sequence_branch": sequence_branch,
                "fixed_interface": fixed_interface,
                "seed": raw_seed,
                "sequence_seed": raw_seed,
            }, "MPNN worker returned no sequence"
        sequences = _restore_source_chain_sequences(
            input_record,
            generated_plan.chain_sequences,
        )
        return sequences, {
            "backend": public_source, "backend_source": generated_plan.source,
            "sequence_branch": sequence_branch,
            "sequence_source": public_source,
            "fixed_interface": fixed_interface,
            "interface_positions": _jsonable(
                getattr(generated_plan, "interface_positions", None)
            ),
            "interface_target_chains": _jsonable(
                getattr(generated_plan, "interface_target_chains", None)
            ),
            "effective_settings": copy.deepcopy(effective_backend),
            "fixed_positions": list(generated_plan.fixed_positions),
            "batch_seed": raw_seed,
            "seed": raw_seed,
            "sequence_seed": raw_seed,
            "batch_size": 1,
            "device": None if self.device is None else str(self.device),
        }, None

    def _generate_mpnn_batch(
        self,
        input_record: Any,
        public_source: str,
        source_indices: Sequence[int],
        work_dir: Path,
        fixed_positions: Sequence[str],
        source_settings: Mapping[str, Any] | None = None,
    ) -> dict[int, tuple[dict[str, str] | None, dict[str, Any], str | None]]:
        """Generate all pending slots for one source/settings batch.

        Duplicate candidate entries share one input structure while carrying
        distinct IDs and ordinals. The native owner receives the complete list
        in one call and chooses its actual batch dimension.
        """
        indices = [int(value) for value in source_indices]
        if not indices:
            return {}
        if len(indices) == 1:
            index = indices[0]
            sequence_map, settings, error = self._generate_mpnn(
                input_record, public_source, index, work_dir,
                fixed_positions, source_settings,
            )
            return {index: (sequence_map, settings, error)}

        from src.design.rewards.esmfold2 import plan_sequence_source
        from src.design.rewards.mpnn import MPNNSequenceError, MPNNSystemError
        from src.validate.mpnn import generate_native_mpnn_sequences

        sample_id = _field(
            input_record,
            "sample_id",
            _row_field(input_record, "sample_id", "sample"),
        )
        effective_settings = dict(source_settings or {})
        sequence_branch = self._sequence_branch(public_source, effective_settings)
        fixed_interface = _normalize_fixed_interface(
            effective_settings.get("fixed_interface")
        ) if "fixed_interface" in effective_settings else None
        internal_source = _INTERNAL_SOURCES[public_source]
        backend = self._mpnn_config(public_source, effective_settings)["sequence"][internal_source]
        raw_seed = self._mpnn_seed(public_source, effective_settings)
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int) or raw_seed < 0:
            error = "MPNN seed must be a non-negative integer"
            return {
                index: (
                    None,
                    {
                        "backend": public_source,
                        "sequence_branch": sequence_branch,
                        "fixed_interface": fixed_interface,
                        "seed": raw_seed,
                        "sequence_seed": raw_seed,
                    },
                    error,
                )
                for index in indices
            }
        effective_settings["seed"] = raw_seed
        reward_config = self._mpnn_config(public_source, effective_settings)
        effective_backend = reward_config["sequence"][internal_source]

        items: list[tuple[Any, Any]] = []
        candidate_by_index: dict[int, Any] = {}
        results: dict[int, tuple[dict[str, str] | None, dict[str, Any], str | None]] = {}
        for source_index in indices:
            candidate_id = (
                f"{sample_id}__{sequence_branch}__{public_source}__{source_index}"
            )
            candidate = _candidate_adapter(input_record, candidate_id, source_index)
            if candidate.atoms is None:
                results[source_index] = (
                    None,
                    {
                        "backend": public_source,
                        "sequence_branch": sequence_branch,
                        "fixed_interface": fixed_interface,
                        "seed": raw_seed,
                        "sequence_seed": raw_seed,
                    },
                    "MPNN input has no AtomArray",
                )
                continue
            try:
                plan = self._plan_source(
                    plan_sequence_source,
                    candidate,
                    internal_source,
                    fixed_positions,
                    fixed_interface,
                )
            except (MPNNSequenceError, SequenceGenerationError, ValueError, KeyError, OSError) as exc:
                results[source_index] = (
                    None,
                    {
                        "backend": public_source,
                        "sequence_branch": sequence_branch,
                        "fixed_interface": fixed_interface,
                        "seed": raw_seed,
                        "sequence_seed": raw_seed,
                    },
                    str(exc),
                )
                continue
            if self._fi_covers_all_design_positions(candidate, plan, fixed_interface):
                results[source_index] = self._fi_passthrough(
                    input_record, plan, public_source, sequence_branch,
                    fixed_interface, raw_seed, effective_backend,
                )
                continue
            candidate_by_index[source_index] = candidate
            items.append((candidate, plan))

        if not items:
            return results
        # The parent removes each scratch tree after persistence; recreate only
        # its parent and leave the leaf to the native adapter.
        Path(work_dir).parent.mkdir(parents=True, exist_ok=True)
        try:
            antibody_plans = _validation_antibody_plans(items)
            generated, failures = generate_native_mpnn_sequences(
                items,
                reward_config=reward_config,
                repo_root=self.repo_root,
                work_dir=work_dir,
                seed=raw_seed,
                **({"antibody_plans": antibody_plans} if antibody_plans else {}),
            )
        except (MPNNSequenceError, SequenceGenerationError, ValueError, KeyError, OSError) as exc:
            for source_index in candidate_by_index:
                results[source_index] = (
                    None,
                    {
                        "backend": public_source,
                        "sequence_source": public_source,
                        "sequence_branch": sequence_branch,
                        "fixed_interface": fixed_interface,
                        "seed": raw_seed,
                        "sequence_seed": raw_seed,
                        "batch_size": len(items),
                    },
                    str(exc),
                )
            return results
        except MPNNSystemError:
            raise

        index_by_candidate = {
            candidate.candidate_id: source_index
            for source_index, candidate in candidate_by_index.items()
        }
        for candidate_id, source_index in index_by_candidate.items():
            base_settings = {
                "backend": public_source,
                "sequence_source": public_source,
                "sequence_branch": sequence_branch,
                "fixed_interface": fixed_interface,
                "batch_seed": raw_seed,
                "seed": raw_seed,
                "sequence_seed": raw_seed,
                "batch_size": len(items),
                "device": None if self.device is None else str(self.device),
            }
            if candidate_id in failures:
                results[source_index] = (None, base_settings, str(failures[candidate_id]))
                continue
            generated_plan = generated.get(candidate_id)
            if generated_plan is None:
                results[source_index] = (
                    None, base_settings, "MPNN worker returned no sequence"
                )
                continue
            sequences = _restore_source_chain_sequences(
                input_record,
                generated_plan.chain_sequences,
            )
            base_settings.update(
                backend_source=generated_plan.source,
                effective_settings=copy.deepcopy(effective_backend),
                fixed_positions=list(generated_plan.fixed_positions),
                interface_positions=_jsonable(
                    getattr(generated_plan, "interface_positions", None)
                ),
                interface_target_chains=_jsonable(
                    getattr(generated_plan, "interface_target_chains", None)
                ),
            )
            results[source_index] = (sequences, base_settings, None)
        return results

    def generate_batches(
        self, input: Any, sources: Sequence[Mapping[str, Any]] | None,
        work_dir: Path, existing: Sequence[Mapping[str, Any]] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield durable variant batches, one native MPNN call per source batch."""
        normalized = self._normalize_sources(sources)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        existing_indices: set[int] = set()
        for record in existing or ():
            if isinstance(record, Mapping) and isinstance(record.get("seq_index"), int):
                if str(record.get("status", "complete")) in {"complete", "ok"}:
                    existing_indices.add(int(record["seq_index"]))
        initial = _initial_sequences(input)
        fixed_positions = _normalize_fixed_positions(input)
        input_path = _field(input, "path", _row_field(input, "path", ""))
        metadata_states = _field(
            input,
            "metadata_states",
            {
                field: (
                    "known" if bool(_field(input, field, None)) else "unknown"
                )
                for field in ("fixed_restypes", "bond_constraints")
            },
        )
        seq_index = 0
        for spec in normalized:
            source, count = str(spec["source"]), int(spec["count"])
            sequence_branch = str(spec["label"])
            source_settings = spec.get("settings")
            fixed_interface = (
                source_settings.get("fixed_interface")
                if isinstance(source_settings, Mapping)
                else None
            )
            pending: list[tuple[int, int]] = []
            for source_index in range(count):
                current = seq_index
                seq_index += 1
                if current not in existing_indices:
                    pending.append((current, source_index))
            if not pending:
                continue

            def make_variant(
                current: int,
                source_index: int,
                sequence_map: Mapping[str, str] | None,
                backend_settings: Mapping[str, Any],
                error: str | None,
            ) -> dict[str, Any]:
                settings: dict[str, Any] = {
                    "source": source,
                    "sequence_source": source,
                    "sequence_branch": sequence_branch,
                    "label": sequence_branch,
                    "source_index": source_index,
                    "count": count,
                    "requested": _jsonable(spec),
                    "fixed_interface": _jsonable(fixed_interface),
                    "fixed_positions": list(fixed_positions),
                    "fixed_restypes": _jsonable(
                        _decode_jsonish(
                            _field(input, "fixed_restypes", ()), default=()
                        )
                    ),
                    "bond_constraints": _jsonable(
                        _decode_jsonish(
                            _field(input, "bond_constraints", ()), default=()
                        )
                    ),
                    "metadata_states": _jsonable(metadata_states),
                }
                settings.update(_jsonable(dict(backend_settings)))
                result: dict[str, Any] = {
                    "seq_index": current,
                    "source": source,
                    "sequence_source": source,
                    "sequence_branch": sequence_branch,
                    "source_index": source_index,
                    "sequence_seed": backend_settings.get("sequence_seed"),
                    "fixed_interface": _jsonable(
                        backend_settings.get("fixed_interface", fixed_interface)
                    ),
                    "interface_positions": _jsonable(
                        backend_settings.get("interface_positions")
                    ),
                    "interface_target_chains": _jsonable(
                        backend_settings.get("interface_target_chains")
                    ),
                    "sequences": {} if sequence_map is None else dict(sequence_map),
                    "structure_path": (
                        str(input_path)
                        if source == "gt" and input_path is not None
                        else None
                    ),
                    "settings": _jsonable(settings),
                    "status": "failed" if error is not None else "complete",
                }
                if error is not None:
                    result["error"] = error
                return result

            if source == "gt":
                yield [
                    make_variant(
                        current,
                        source_index,
                        dict(initial),
                        {
                            "backend": "ground_truth",
                            "sequence_source": "gt",
                            "sequence_branch": sequence_branch,
                            "sequence_seed": None,
                            "fixed_interface": None,
                        },
                        None,
                    )
                    for current, source_index in pending
                ]
                continue

            first_current = pending[0][0]
            batch_work = (
                work_dir
                / f"attempt-{uuid4().hex}"
                / f"batch-{first_current}_{sequence_branch}_{source}"
            )
            previous_branch = getattr(self, "_sequence_branch_context", None)
            self._sequence_branch_context = sequence_branch
            try:
                generation_settings = spec.get("settings")
                if generation_settings == {}:
                    generation_settings = None
                generated = self._generate_mpnn_batch(
                    input,
                    source,
                    # Replay the original native batch on partial resume so
                    # slot i still receives draw i; persist only pending slots.
                    list(range(count)),
                    batch_work,
                    fixed_positions,
                    generation_settings,
                )
            finally:
                if previous_branch is None:
                    self.__dict__.pop("_sequence_branch_context", None)
                else:
                    self._sequence_branch_context = previous_branch
            batch: list[dict[str, Any]] = []
            for current, source_index in pending:
                sequence_map, backend_settings, error = generated.get(
                    source_index,
                    (
                        None,
                        {"backend": source},
                        "MPNN worker returned no sequence",
                    ),
                )
                batch.append(
                    make_variant(
                        current,
                        source_index,
                        sequence_map,
                        backend_settings,
                        error,
                    )
                )
            yield batch

    def generate(
        self, input: Any, sources: Sequence[Mapping[str, Any]] | None,
        work_dir: Path, existing: Sequence[Mapping[str, Any]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Flatten :meth:`generate_batches` for the legacy public API."""
        for batch in self.generate_batches(input, sources, work_dir, existing):
            yield from batch


__all__ = [
    "SequenceGenerationError",
    "SequenceGenerator",
    "normalize_sequence_sources",
]
