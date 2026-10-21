"""Compact primitive result rows for standalone antibody output.

The runtime keeps antibody rows and their coordinate-free context on an
in-memory/distributed metadata boundary.  This module intentionally returns
only primitive values for the public row; the context helper is the private
wire payload used to re-analyse a final CIF after folding or postprocessing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .analysis import analyze_antibody


_PIECES = ("fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4")

ANTIBODY_RESULT_FIELDS = (
    "filename", "scaffold", "template", "heavy_role", "heavy_chain", "light_chain",
    "antigen_chains", "heavy_sequence", "light_sequence",
    "heavy_fr1", "heavy_fr1_range", "heavy_cdr1", "heavy_cdr1_range",
    "heavy_fr2", "heavy_fr2_range", "heavy_cdr2", "heavy_cdr2_range",
    "heavy_fr3", "heavy_fr3_range", "heavy_cdr3", "heavy_cdr3_range",
    "heavy_fr4", "heavy_fr4_range", "light_fr1", "light_fr1_range",
    "light_cdr1", "light_cdr1_range", "light_fr2", "light_fr2_range",
    "light_cdr2", "light_cdr2_range", "light_fr3", "light_fr3_range",
    "light_cdr3", "light_cdr3_range", "light_fr4", "light_fr4_range",
    "heavy_design_positions", "light_design_positions", "sequence_source",
    "structure_source", "reward_method", "reward_score", "rank",
    "final_partiscore", "score_rank",
    "partiscore_antigen", "partiscore_antigen_pair_count", "partiscore_internal",
    "partiscore_internal_pair_count", "refold_plddt", "refold_iptm",
    "fixed_sequence_ok", "framework_rmsd", "antigen_contacts", "antigen_clashes",
    "status", "postprocess_status", "error_stage", "error",
)

ANTIBODY_SCORE_FIELDS = ("final_partiscore", "score_rank")


def antibody_result_fields(*, score: bool = False) -> tuple[str, ...]:
    """Keep the legacy schema for tasks without final-file scoring."""
    return tuple(field for field in ANTIBODY_RESULT_FIELDS
                 if score or field not in ANTIBODY_SCORE_FIELDS)

_PIECE_FIELDS = tuple(f"{side}_{piece}" for side in ("heavy", "light") for piece in _PIECES)
_AA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
_AA_ONE = frozenset(_AA.values())


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _text(value: Any, default: str = "") -> str:
    return default if value is None else str(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "" or value == "." or value == "?":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", ".", "?"}:
            return bool(default)
        if lowered in {"0", "false", "no", "n"}:
            return False
        if lowered in {"1", "true", "yes", "y"}:
            return True
    return bool(value)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _primitive(value: Any) -> Any:
    """Return a JSON-safe primitive tree without serializing arbitrary owners."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _primitive(value.item())
    if isinstance(value, np.ndarray):
        return [_primitive(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_primitive(item) for item in value]
    return str(value)


def _metadata(candidate: Any) -> Mapping[str, Any]:
    """Find compiled antibody metadata on current and legacy candidates."""
    queue = [candidate]
    seen: set[int] = set()
    while queue:
        obj = queue.pop(0)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        direct = _value(obj, "antibody_metadata")
        if isinstance(direct, Mapping) and (
            "entries" in direct or "scaffold" in direct or "template" in direct
        ):
            return direct
        for key in ("compiled", "base_request", "task", "request"):
            nested = _value(obj, key)
            if nested is not None and not isinstance(nested, (str, bytes, int, float, bool)):
                queue.append(nested)
    return {}


def _metadata_entries(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = metadata.get("entries", metadata.get("source_ledger", ()))
    if isinstance(raw, Mapping):
        raw = raw.get("entries", raw.get("source_ledger", ()))
    if raw is None or isinstance(raw, (str, bytes)):
        return []
    result: list[Mapping[str, Any]] = []
    try:
        for item in raw:
            if isinstance(item, Mapping):
                result.append(item)
    except TypeError:
        pass
    return result


def antibody_result_context(candidate: Any) -> dict[str, Any]:
    """Serialize compiled metadata for final-CIF re-analysis.

    The tree contains no candidate, AtomArray, numpy array, or dataclass
    object. Template coordinates and masks remain in the entries so framework
    RMSD is recomputable after a distributed handoff or CIF readback.
    """
    metadata = _metadata(candidate)
    entries = _metadata_entries(metadata)
    template_from_entries = False
    for entry in entries:
        mask = entry.get("template_mask")
        if mask is not None and bool(np.asarray(mask, dtype=bool).any()):
            template_from_entries = True
            break
    requested_antibody = _value(_value(_value(candidate, "base_request"), "task"), "antibody")
    requested_template = _value(requested_antibody, "template")
    template_value = metadata.get("template")
    if requested_template is not None:
        template = _bool(requested_template)
    elif template_value is not None:
        template = _bool(template_value)
    else:
        # Reduced saved fixtures may not retain the parsed task spec; visible
        # template masks are the only reliable fallback in that case.
        template = template_from_entries
    conditions = _value(candidate, "topology_conditions", ()) or ()
    return {
        "scaffold": _primitive(metadata.get("scaffold", "")),
        "source_numbering": _primitive(metadata.get("source_numbering", "label_seq_id")),
        "template": bool(template),
        "boundary_unresolved": _primitive(metadata.get("boundary_unresolved", {})),
        "entries": _primitive(entries),
        "topology_conditions": _primitive(conditions),
    }


def _context_entries(context: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = context.get("entries", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _annotation(atoms: Any, name: str, default: Any) -> np.ndarray:
    try:
        categories = set(atoms.get_annotation_categories())
    except AttributeError:
        categories = set()
    if name in categories:
        return np.asarray(atoms.get_annotation(name))
    if hasattr(atoms, name):
        return np.asarray(getattr(atoms, name))
    return np.full(len(atoms), default)


def _aa(value: Any) -> str:
    text = _text(value).strip().upper()
    if len(text) == 1 and text in _AA_ONE:
        return text
    return _AA.get(text, "X")


def _role(value: Any, chain: str = "") -> str:
    text = _text(value).strip().upper()
    if text in {"", "0", "CONTEXT", "ANTIGEN", "TARGET"}:
        return ""
    if text in {"VH", "H", "HEAVY", "HEAVY_CHAIN", "VARIABLE_HEAVY"}:
        return "VH"
    if text in {"VHH", "NANOBODY", "HEAVY_ONLY"}:
        return "VHH"
    if text in {"VL", "L", "LIGHT", "LIGHT_CHAIN", "VARIABLE_LIGHT"}:
        return "VL"
    number = _int_or_none(value)
    if number == 0:
        return ""
    if number == 1:
        return "VHH" if "VHH" in chain.upper() else "VH"
    if number == 2:
        return "VL"
    if "VHH" in chain.upper():
        return "VHH"
    return ""


def _entry_index(entries: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in entries:
        chain = _text(entry.get("chain_id", entry.get("output_chain_id", "")))
        position = _int_or_none(entry.get("position", entry.get("res_id", entry.get("label_seq_id"))))
        if chain and position is not None:
            result.setdefault((chain, position), entry)
    return result


def _residue_records(atoms: Any, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    chains = _annotation(atoms, "chain_id", "")
    positions = _annotation(atoms, "res_id", 0)
    ins_codes = _annotation(atoms, "ins_code", "")
    labels = _annotation(atoms, "label_seq_id", "")
    res_names = _annotation(atoms, "res_name", "UNK")
    roles = _annotation(atoms, "antibody_chain_role_id", 0)
    regions = _annotation(atoms, "antibody_region_id", 0)
    mutable = _annotation(atoms, "antibody_sequence_mutable", False)
    source_labels = _annotation(atoms, "antibody_source_label_seq_id", -1)
    entries = _entry_index(_context_entries(context))
    groups: dict[tuple[str, int, str], list[int]] = {}
    for row in range(len(atoms)):
        key = (_text(chains[row]), int(positions[row]), _text(ins_codes[row]))
        groups.setdefault(key, []).append(row)

    records: list[dict[str, Any]] = []
    for (chain, position, ins_code), rows in groups.items():
        row = rows[0]
        entry = entries.get((chain, position), {})
        raw_role = entry.get("role", entry.get("chain_role", entry.get("role_id")))
        role = _role(raw_role, chain) if raw_role is not None else _role(roles[row], chain)
        raw_region = entry.get("region_id", entry.get("region"))
        region = _int_or_none(raw_region) if raw_region is not None else _int_or_none(regions[row])
        region = 0 if region is None else int(region)
        label = _int_or_none(labels[row]) or int(position)
        source_label = entry.get("source_label_seq_id")
        source_label = _int_or_none(source_label) if source_label is not None else _int_or_none(source_labels[row])
        raw_source_aa = entry.get("source_amino_acid")
        source_aa = _aa(raw_source_aa) if raw_source_aa not in (None, "", ".", "?") else ""
        if entry:
            mutable_value = _bool(entry.get("mutable"), _bool(mutable[row]))
            source_present = _bool(entry.get("source_present"), source_label is not None and source_label >= 0)
        else:
            mutable_value = _bool(mutable[row])
            source_present = source_label is not None and source_label >= 0
        records.append({
            "chain": chain, "position": position, "ins_code": ins_code, "label": label,
            "aa": _aa(res_names[row]), "role": role, "region": region,
            "mutable": mutable_value, "source_label": source_label,
            "source_aa": source_aa, "source_present": source_present,
        })
    records.sort(key=lambda item: (item["chain"], item["label"], item["position"], item["ins_code"]))
    return records


def _compact_ranges(values: Sequence[int]) -> str:
    if not values:
        return ""
    ordered = [int(value) for value in values]
    chunks: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
        else:
            chunks.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = value
    chunks.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(chunks)


def _piece_records(chain_records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    pieces = {name: [] for name in _PIECES}
    last_cdr_number = 0
    for record in chain_records:
        region = int(record["region"])
        if region == 1:
            pieces[f"fr{last_cdr_number + 1}"].append(record)
        elif region in (2, 3, 4):
            cdr_number = region - 1
            if cdr_number < last_cdr_number or cdr_number > last_cdr_number + 1:
                raise ValueError("antibody compiled regions are out of biological order")
            pieces[f"cdr{cdr_number}"].append(record)
            last_cdr_number = cdr_number
        else:
            raise ValueError(f"antibody variable residue has unsupported region_id {region}")
    return pieces


def _chain_sequence_and_pieces(chain_records: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, tuple[str, str]], list[int]]:
    pieces = _piece_records(chain_records)
    sequence = "".join(_text(record["aa"]) for record in chain_records)
    rendered: dict[str, tuple[str, str]] = {}
    labels: list[int] = []
    for name in _PIECES:
        selected = pieces[name]
        piece_sequence = "".join(_text(record["aa"]) for record in selected)
        piece_labels = [int(record["label"]) for record in selected]
        rendered[name] = (piece_sequence, _compact_ranges(piece_labels))
        labels.extend(piece_labels)
    if "".join(rendered[name][0] for name in _PIECES) != sequence:
        raise ValueError("antibody region pieces do not concatenate to the final sequence")
    expected_labels = [int(record["label"]) for record in chain_records]
    if labels != expected_labels:
        raise ValueError("antibody region ranges do not correspond to final CIF label_seq_id")
    return sequence, rendered, labels


def _structure_fields(atoms: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    records = _residue_records(atoms, context)
    by_chain: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_chain.setdefault(record["chain"], []).append(record)
    binder = {chain: values for chain, values in by_chain.items() if values and values[0]["role"] in {"VH", "VHH", "VL"}}
    heavy = [(chain, values) for chain, values in binder.items() if values[0]["role"] in {"VH", "VHH"}]
    light = [(chain, values) for chain, values in binder.items() if values[0]["role"] == "VL"]
    if len(heavy) != 1:
        raise ValueError(f"antibody output must contain exactly one heavy/VHH chain, found {len(heavy)}")
    if len(light) > 1:
        raise ValueError("antibody output must contain at most one light chain")
    heavy_chain, heavy_records = heavy[0]
    if light:
        light_chain, light_records = light[0]
    else:
        light_chain, light_records = "", []
    heavy_sequence, heavy_pieces, heavy_labels = _chain_sequence_and_pieces(heavy_records)
    light_sequence, light_pieces, light_labels = ("", {}, [])
    if light_records:
        light_sequence, light_pieces, light_labels = _chain_sequence_and_pieces(light_records)
    binder_chain_ids = set(binder)
    antigen_chains = [chain for chain in by_chain if chain not in binder_chain_ids]

    fixed_ok = True
    for record in (*heavy_records, *light_records):
        if record["mutable"]:
            continue
        if record["source_aa"] in _AA_ONE and record["source_aa"] != record["aa"]:
            fixed_ok = False

    # ``analyze_antibody`` has a legacy chain-name fallback when it cannot
    # find a compiled entry for a residue.  Final CIFs do not carry the
    # optional antibody annotations, so make every residue outside the
    # compiled binder entries explicit context before handing the structure
    # to the geometry analyzer.  This is especially important for antigen
    # chains whose literal name happens to be ``VHH``/``VH``/``VL``.
    analysis_entries = list(_context_entries(context))
    known_positions = {
        (
            _text(entry.get("chain_id", entry.get("output_chain_id", ""))),
            _int_or_none(entry.get("position", entry.get("res_id", entry.get("label_seq_id")))),
        )
        for entry in analysis_entries
    }
    for record in records:
        key = (record["chain"], int(record["position"]))
        if record["role"] not in {"VH", "VHH", "VL"} and key not in known_positions:
            analysis_entries.append({
                "chain_id": record["chain"],
                "position": int(record["position"]),
                "role": "context",
                "region_id": 0,
                "mutable": False,
                "source_present": False,
            })
            known_positions.add(key)

    report_candidate = SimpleNamespace(
        atoms=atoms,
        antibody_metadata={"entries": analysis_entries, "scaffold": context.get("scaffold", "")},
        topology_conditions=tuple(context.get("topology_conditions", ()) or ()),
    )
    report = analyze_antibody(report_candidate, scaffold_entries=analysis_entries)
    metrics = report.get("metrics", {})
    has_antigen = bool(antigen_chains)
    return {
        "heavy_role": heavy_records[0]["role"], "heavy_chain": heavy_chain,
        "light_chain": light_chain, "antigen_chains": ",".join(antigen_chains),
        "heavy_sequence": heavy_sequence, "light_sequence": light_sequence,
        "heavy_design_positions": _compact_ranges([int(record["label"]) for record in heavy_records if record["mutable"]]),
        "light_design_positions": _compact_ranges([int(record["label"]) for record in light_records if record["mutable"]]),
        "fixed_sequence_ok": bool(fixed_ok),
        "framework_rmsd": _finite(metrics.get("fr_aligned_backbone_rmsd_angstrom")),
        "antigen_contacts": int(metrics.get("binder_antigen_contact_count", 0)) if has_antigen else "",
        "antigen_clashes": int(metrics.get("binder_antigen_clash_count", 0)) if has_antigen else "",
        "_heavy_pieces": heavy_pieces, "_light_pieces": light_pieces,
        "_heavy_labels": heavy_labels, "_light_labels": light_labels,
    }


def _apply_structure_fields(row: dict[str, Any], atoms: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    derived = _structure_fields(atoms, context)
    for field in ("heavy_role", "heavy_chain", "light_chain", "antigen_chains", "heavy_sequence", "light_sequence", "heavy_design_positions", "light_design_positions", "fixed_sequence_ok", "framework_rmsd", "antigen_contacts", "antigen_clashes"):
        row[field] = derived[field]
    for prefix, pieces in (("heavy", derived["_heavy_pieces"]), ("light", derived["_light_pieces"])):
        for name in _PIECES:
            sequence, ranges = pieces.get(name, ("", ""))
            row[f"{prefix}_{name}"] = sequence
            row[f"{prefix}_{name}_range"] = ranges
    return row


def refresh_antibody_result(row: Mapping[str, Any], atoms: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute structure-derived cells from final atoms and primitive context."""
    if not isinstance(row, Mapping):
        raise TypeError("antibody result row must be a mapping")
    if not isinstance(context, Mapping):
        raise TypeError("antibody result context must be a mapping")
    score = any(field in row for field in ANTIBODY_SCORE_FIELDS)
    updated = {field: row.get(field, "") for field in antibody_result_fields(score=score)}
    return _apply_structure_fields(updated, atoms, context)


def _fold_value(fold: Any, names: Sequence[str]) -> float | None:
    if fold is None:
        return None
    metrics = _value(fold, "metrics", {})
    confidence = _value(fold, "confidence")
    for source in (metrics, confidence):
        for name in names:
            result = _finite(_value(source, name))
            if result is not None:
                return result
    return None


def build_antibody_result(candidate: Any, output_atoms: Any, output_path: Path, *, fold: Any = None) -> dict[str, Any]:
    """Build one compact antibody CSV row without writing sidecar files."""
    context = antibody_result_context(candidate)
    row = {field: "" for field in ANTIBODY_RESULT_FIELDS}
    row.update({
        "filename": Path(output_path).name,
        "scaffold": context.get("scaffold", ""),
        "template": bool(context.get("template", False)),
        "sequence_source": _text(_value(fold, "sequence_source"), "fold" if fold is not None else "design"),
        "structure_source": "fold" if fold is not None else "design",
        "status": "ok",
        "refold_plddt": _fold_value(fold, ("refold_plddt", "esmfold2_plddt", "design_plddt")),
        "refold_iptm": _fold_value(fold, ("refold_iptm", "esmfold2_iptm", "interface_iptm")),
    })
    return _apply_structure_fields(row, output_atoms, context)


def publish_antibody_outputs(candidate: Any, output_atoms: Any, output_path: Path, *, fold: Any = None) -> dict[str, Any] | None:
    """Return an antibody result row; retain the no-op behavior for other tasks."""
    task = _value(_value(candidate, "base_request"), "task")
    if task is not None and _text(_value(task, "format"), "normal") != "antibody":
        return None
    return build_antibody_result(candidate, output_atoms, output_path, fold=fold)


__all__ = ["ANTIBODY_RESULT_FIELDS", "ANTIBODY_SCORE_FIELDS", "antibody_result_fields", "antibody_result_context", "build_antibody_result", "publish_antibody_outputs", "refresh_antibody_result"]
