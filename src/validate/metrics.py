"""Model-neutral structural and confidence metrics for validation rows.

This module intentionally consumes an already completed prediction record.  It
does not run a folding/scoring backend and does not decide success criteria.
Every unavailable value is represented by ``None`` and a reason, allowing the
caller to distinguish an inapplicable metric from a failed calculation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..design.utils.reader import read_structure
from ..design.utils.atoms import COMMON_STANDARD_PARENT
from .inputs import ValidationInput


_BACKBONE3 = ("N", "CA", "C")
_BACKBONE4 = ("N", "CA", "C", "O")
_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}
_METRIC_NAMES = (
    "plddt", "plddt_total", "ptm", "iptm", "binder_ptm", "binder_iptm",
    "pae_interaction",
    "ipae_normalized", "ipae_angstrom", "min_interchain_pae_angstrom",
    "complex_design_rmsd", "complex_ca_p_rmsd", "design_rmsd", "binder_design_rmsd",
    "context_aligned_design_rmsd", "ligand_rmsd", "context_aligned_ligand_rmsd",
    "receptor_aligned_ligand_rmsd", "complexa_aligned_ligand_rmsd",
    "min_design_target_pae_angstrom", "min_target_design_pae_angstrom",
    "antibody_antigen_pae_angstrom", "antigen_aligned_antibody_ca_rmsd",
    "binder_scRMSD_bb3", "motif_rmsd_reference", "motif_rmsd_design",
    "correct_motif_sequence", "motif_sequence_recovery", "sequence_recovery",
    "has_ligand_clashes",
    "na_token_center_rmsd", "na_aligned_protein_ca_rmsd",
    "protein_aligned_na_p_rmsd",
)


class MetricUnavailable(ValueError):
    """A requested metric cannot be computed from the available identity."""


def _missing(metrics: dict[str, Any], reasons: dict[str, str], name: str, reason: str) -> None:
    metrics.setdefault(name, None)
    reasons.setdefault(name, reason)


def _chain_values(atoms: Any) -> np.ndarray:
    if "label_asym_id" in atoms.get_annotation_categories():
        return np.asarray(atoms.label_asym_id).astype(str)
    return np.asarray(atoms.chain_id).astype(str)


def _ordered_chains(atoms: Any) -> list[str]:
    return list(dict.fromkeys(_chain_values(atoms).tolist()))


def _residue_rows(atoms: Any, chain: str, *, polymer_only: bool = False) -> list[dict[str, Any]]:
    values = _chain_values(atoms)
    selected = values == str(chain)
    if polymer_only:
        selected &= np.isin(np.asarray(atoms.mol_type).astype(str), ("protein", "rna", "dna"))
    rows = np.flatnonzero(selected)
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw_row in rows.tolist():
        row = int(raw_row)
        uid = int(atoms.residue_uid[row])
        if uid in seen:
            continue
        seen.add(uid)
        residue_rows = rows[np.asarray(atoms.residue_uid[rows] == uid)]
        atom_rows = {str(atoms.atom_name[index]): int(index) for index in residue_rows.tolist()}
        label_seq = str(atoms.label_seq_id[residue_rows[0]])
        try:
            label_number = int(label_seq)
        except ValueError:
            label_number = int(atoms.res_id[residue_rows[0]])
        result.append({
            "uid": uid,
            "res_id": int(atoms.res_id[residue_rows[0]]),
            "label_seq": label_number,
            "ins_code": str(atoms.ins_code[residue_rows[0]]),
            "res_name": str(atoms.res_name[residue_rows[0]]),
            "rows": atom_rows,
        })
    return result


def _polymer_chain(atoms: Any, chain: str) -> bool:
    return _chain_mol_type(atoms, chain) in {"protein", "rna", "dna"}


def _chain_mol_type(atoms: Any, chain: str) -> str | None:
    rows = np.flatnonzero(_chain_values(atoms) == str(chain))
    if not len(rows):
        return None
    kinds = set(np.asarray(atoms.mol_type).astype(str)[rows].tolist())
    polymer = kinds.intersection({"protein", "rna", "dna"})
    if len(polymer) == 1 and not kinds.difference(polymer | {"ligand"}):
        return next(iter(polymer))
    return next(iter(kinds)) if len(kinds) == 1 else None


def _is_ligand_chain(atoms: Any, chain: str) -> bool:
    return _chain_mol_type(atoms, chain) == "ligand"


def _chain_sequence(atoms: Any, chain: str) -> str:
    modality = _chain_mol_type(atoms, chain)
    sequence: list[str] = []
    for item in _residue_rows(atoms, chain, polymer_only=modality in {"protein", "rna", "dna"}):
        value = str(item["res_name"]).strip().upper()
        if modality == "rna":
            value = COMMON_STANDARD_PARENT.get(value, value)
            sequence.append(value if value in {"A", "C", "G", "U"} else "X")
        elif modality == "dna":
            value = COMMON_STANDARD_PARENT.get(value, value)
            sequence.append({"DA": "A", "DC": "C", "DG": "G", "DT": "T"}.get(value, "X"))
        elif len(value) == 1 and value in set(_AA3_TO_1.values()):
            sequence.append(value)
        else:
            sequence.append(_AA3_TO_1.get(value, "X"))
    return "".join(sequence)


def _valid_coord(atoms: Any, row: int) -> bool:
    """Return whether one atom row has a usable coordinate."""

    declared = getattr(atoms, "coord_valid", None)
    if declared is not None and not bool(declared[row]):
        return False
    try:
        return bool(np.isfinite(np.asarray(atoms.coord[row], dtype=np.float64)).all())
    except (TypeError, ValueError, IndexError):
        return False


def _atom_element(atoms: Any, row: int) -> str:
    values = getattr(atoms, "element", None)
    if values is not None:
        value = str(values[row]).strip().upper()
        if value not in {"", ".", "?"}:
            return value
    atom_name = str(atoms.atom_name[row]).strip().upper()
    return next((char for char in atom_name if char.isalpha()), "?")


def _heavy_atom_names(atoms: Any, residue: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, row in residue["rows"].items()
            if _atom_element(atoms, int(row)) not in {"H", "D"}
        )
    )


def _variant_sequences(variant: Mapping[str, Any] | None) -> dict[str, str]:
    if not variant:
        return {}
    raw = variant.get("sequences", variant.get("chain_sequences", variant))
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, str] = {}
    for chain, sequence in raw.items():
        if isinstance(sequence, str) and sequence:
            result[str(chain)] = sequence.upper()
    return result


def _expected_chain_sequences(
    reference: Any,
    input_data: ValidationInput,
    variant: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Build the sequence contract used to identify prediction chains.

    Design chains are compared to the requested variant sequence when one is
    supplied (MPNN/refold output may intentionally differ from the generated
    input). Context chains always use the loaded input structure as their
    ground truth. This makes chain labels a useful hint without accepting a
    positional or equal-length guess.
    """

    variant_sequences = _variant_sequences(variant)
    expected: dict[str, str] = {}
    for chain in _ordered_chains(reference):
        if chain in input_data.design_chains and chain in variant_sequences:
            expected[chain] = variant_sequences[chain]
        else:
            expected[chain] = _chain_sequence(reference, chain)
    return expected


def _sequence_matches(atoms: Any, chain: str, expected: str) -> bool:
    actual = _chain_sequence(atoms, chain)
    return bool(expected) and actual == expected


def _ligand_graph_matches(
    reference: Any,
    predicted: Any,
    reference_chain: str,
    predicted_chain: str,
) -> bool:
    if not _is_ligand_chain(reference, reference_chain):
        return True
    from .ligand_metrics import LigandMappingError, atom_mapping

    try:
        atom_mapping(reference, predicted, reference_chain, predicted_chain)
    except LigandMappingError:
        return False
    return True


def _chain_map(
    reference: Any,
    predicted: Any,
    input_data: ValidationInput,
    prediction: Mapping[str, Any],
    variant: Mapping[str, Any] | None,
) -> dict[str, str]:
    ref_chains = _ordered_chains(reference)
    pred_chains = _ordered_chains(predicted)
    expected_sequences = _expected_chain_sequences(reference, input_data, variant)
    available = set(pred_chains)
    result: dict[str, str] = {}

    def identity_matches(ref_chain: str, candidate: str) -> bool:
        # Ligand residue names are backend-specific (and may be one SMILES
        # string in the input ledger), so the explicit heavy-atom graph is
        # their identity contract.  Protein/polymer chains still require an
        # exact sequence match in addition to any topology proof.
        if not _is_ligand_chain(reference, ref_chain):
            expected = expected_sequences.get(ref_chain, "")
            if not _sequence_matches(predicted, candidate, expected):
                return False
        return _ligand_graph_matches(reference, predicted, ref_chain, candidate)

    explicit = prediction.get("chain_map")
    if isinstance(explicit, Mapping):
        # An explicit map is still an identity assertion, not permission to
        # compare unrelated chains. Reject duplicate, absent, or
        # sequence-inconsistent targets and leave those chains unavailable.
        for ref_chain in ref_chains:
            candidate = explicit.get(ref_chain)
            if candidate is None:
                continue
            candidate = str(candidate)
            if (
                candidate in available
                and identity_matches(ref_chain, candidate)
            ):
                result[ref_chain] = candidate
                available.remove(candidate)
        return result

    # Canonical labels are preferred only after sequence verification. This
    # handles the common contract while refusing a label collision caused by
    # a mutated MPNN sequence.
    for chain in ref_chains:
        if (
            chain in available
            and identity_matches(chain, chain)
        ):
            result[chain] = chain
            available.remove(chain)

    # If labels were rewritten by a backend, allow a unique full-sequence
    # match. Equal lengths, residue order, and file order alone are never
    # sufficient; repeated identical sequences intentionally remain unmapped.
    for chain in ref_chains:
        if chain in result:
            continue
        expected = expected_sequences.get(chain, "")
        candidates = [candidate for candidate in sorted(available) if identity_matches(chain, candidate)]
        if len(candidates) == 1:
            result[chain] = candidates[0]
            available.remove(candidates[0])
    return result


def _paired_points(
    reference: Any,
    predicted: Any,
    chain_map: Mapping[str, str],
    chains: Sequence[str],
    atom_names: Sequence[str] | None = None,
    *,
    all_heavy: bool = False,
    exclude_atom_names: Sequence[str] = (),
    reference_valid_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    ref_points: list[np.ndarray] = []
    pred_points: list[np.ndarray] = []
    expected = 0
    matched = 0
    for ref_chain in chains:
        pred_chain = chain_map.get(ref_chain)
        polymer_only = _polymer_chain(reference, ref_chain)
        ref_residues = _residue_rows(reference, ref_chain, polymer_only=polymer_only)
        if atom_names is None and all_heavy:
            requested_names = [
                tuple(name for name in _heavy_atom_names(reference, residue)
                      if name not in exclude_atom_names)
                for residue in ref_residues
            ]
        else:
            requested_names = [tuple(atom_names or ())] * len(ref_residues)
        if reference_valid_only:
            # A source 5-prime terminus may have no phosphate. Require every
            # observed source atom, without requiring absent template slots.
            requested_names = [
                tuple(name for name in names
                      if name in residue["rows"]
                      and _valid_coord(reference, residue["rows"][name]))
                for residue, names in zip(ref_residues, requested_names, strict=True)
            ]
        if pred_chain is None:
            expected += sum(len(names) for names in requested_names)
            continue
        pred_residues = _residue_rows(predicted, pred_chain, polymer_only=polymer_only)
        expected += sum(len(names) for names in requested_names)
        if len(ref_residues) != len(pred_residues):
            continue
        for ref_residue, pred_residue, names in zip(
            ref_residues, pred_residues, requested_names, strict=True
        ):
            for name in names:
                ref_row = ref_residue["rows"].get(str(name))
                pred_row = pred_residue["rows"].get(str(name))
                if ref_row is None or pred_row is None:
                    continue
                if not _valid_coord(reference, ref_row) or not _valid_coord(predicted, pred_row):
                    continue
                ref_points.append(np.asarray(reference.coord[ref_row], dtype=np.float64))
                pred_points.append(np.asarray(predicted.coord[pred_row], dtype=np.float64))
                matched += 1
    return (
        np.asarray(ref_points, dtype=np.float64).reshape((-1, 3)),
        np.asarray(pred_points, dtype=np.float64).reshape((-1, 3)),
        matched,
        expected,
    )


def _kabsch(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    minimum_points: int = 3,
) -> tuple[np.ndarray, float]:
    if reference.shape != predicted.shape or reference.ndim != 2 or reference.shape[1:] != (3,):
        raise MetricUnavailable("coordinate arrays have incompatible shapes")
    if len(reference) < minimum_points:
        if minimum_points == 1:
            raise MetricUnavailable("at least one paired atom is required for a motif fit")
        raise MetricUnavailable("at least three paired atoms are required for a rigid fit")
    ref_center = reference.mean(axis=0)
    pred_center = predicted.mean(axis=0)
    ref_zero = reference - ref_center
    pred_zero = predicted - pred_center
    covariance = pred_zero.T @ ref_zero
    left, _singular, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        right_t[-1, :] *= -1.0
        rotation = left @ right_t
    transformed = pred_zero @ rotation + ref_center
    rmsd = float(np.sqrt(np.mean(np.sum((transformed - reference) ** 2, axis=1))))
    transform = np.concatenate(
        (rotation, (ref_center - pred_center @ rotation).reshape(3, 1)), axis=1
    )
    return transform, rmsd


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:, :3] + transform[:, 3]


def _rmsd_with_fit(reference: np.ndarray, predicted: np.ndarray) -> float:
    return _kabsch(reference, predicted)[1]


def _motif_rmsd_with_fit(reference: np.ndarray, predicted: np.ndarray) -> float:
    """Match Proteina-Complexa's selected-tip least-squares motif RMSD."""

    return _kabsch(reference, predicted, minimum_points=1)[1]


def _fit_then_measure(
    fit_reference: np.ndarray,
    fit_predicted: np.ndarray,
    measure_reference: np.ndarray,
    measure_predicted: np.ndarray,
) -> float:
    transform, _ = _kabsch(fit_reference, fit_predicted)
    transformed = _apply_transform(measure_predicted, transform)
    if transformed.shape != measure_reference.shape or not len(transformed):
        raise MetricUnavailable("fit and measured coordinate arrays are incompatible")
    return float(np.sqrt(np.mean(np.sum((transformed - measure_reference) ** 2, axis=1))))


def _coverage_reason(
    label: str,
    reference: np.ndarray,
    predicted: np.ndarray,
    matched: int,
    expected: int,
) -> str | None:
    """Explain why a structural metric cannot use a partial atom set."""

    if expected <= 0:
        return f"{label} has no requested atom correspondence"
    if matched != expected or len(reference) != expected or len(predicted) != expected:
        return f"{label} atom coverage incomplete ({matched}/{expected} requested atoms)"
    if len(reference) != len(predicted):
        return f"{label} atom correspondence has incompatible shapes"
    return None


def _read_json_file(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_confidence_file(path: Path) -> Any:
    """Read JSON confidence or the AF2 backend's native PAE ``.npz``."""

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            result: dict[str, Any] = {}
            if "pae" in archive:
                result["pae_matrix"] = np.asarray(archive["pae"], dtype=np.float64)
            elif len(archive.files) == 1:
                result["pae_matrix"] = np.asarray(archive[archive.files[0]], dtype=np.float64)
            keys = list(archive.files)
            # Preserve native token identity metadata. ``asym_id`` is often
            # numeric in ESMFold2 archives; the caller must resolve it using
            # exact token labels or declared per-chain token counts.
            for key in (
                "asym_id", "token_chain_ids", "pae_chain_ids", "chain_ids",
                "token_index", "residue_index",
            ):
                if key in archive:
                    result[key] = np.asarray(archive[key]).copy()
            return result
    return _read_json_file(path)


def _confidence_payload(prediction: Mapping[str, Any]) -> Mapping[str, Any]:
    values: dict[str, Any] = dict(prediction.get("_confidence_arrays") or {})
    raw_values = prediction.get("confidence")
    if isinstance(raw_values, (str, Path)):
        try:
            raw_values = _read_confidence_file(Path(raw_values))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raw_values = None
    raw_path = prediction.get("raw_confidence_path")
    if raw_path:
        try:
            raw_file = _read_confidence_file(Path(raw_path))
            if isinstance(raw_file, Mapping):
                values.update(raw_file)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    if isinstance(raw_values, Mapping):
        raw_pae_matrix = raw_values.get("pae_matrix")
        values.update(raw_values)
        nested = raw_values.get("confidence")
        if isinstance(nested, Mapping):
            values.update(nested)
        # Keep a raw matrix when the native confidence summary uses the same
        # ``pae`` key for its scalar normalized value.
        if raw_pae_matrix is not None and "pae_matrix" not in values:
            values["pae_matrix"] = raw_pae_matrix
    return values


def _lookup(payload: Mapping[str, Any], *names: str) -> Any:
    lowered = {str(key).lower(): value for key, value in payload.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            return value
    return None


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
        return float(value)
    return None


def _resolve_token_chain_labels(
    payload: Mapping[str, Any],
    prediction: Mapping[str, Any],
    predicted: Any,
    size: int,
) -> tuple[np.ndarray | None, str | None]:
    """Resolve exact PAE token labels to prediction chain labels.

    ESMFold2 may expose string ``token_chain_ids`` directly or numeric
    ``asym_id`` labels in a native archive. Numeric labels are accepted only
    when an explicit label map or the confidence payload's declared
    ``per_chain_token_count`` gives a unique correspondence; residue lengths
    and file order are never guessed.
    """

    raw = prediction.get("token_chain_ids")
    if raw is None:
        raw = prediction.get("pae_chain_ids")
    if raw is None:
        raw = _lookup(
            payload,
            "token_chain_ids", "pae_chain_ids", "chain_ids", "asym_id",
        )
    if raw is None:
        return None, "PAE has no exact token chain labels"
    values = np.asarray(raw).reshape(-1)
    if len(values) != size:
        return None, "PAE token chain label count does not match matrix"
    predicted_chains = set(_ordered_chains(predicted))
    direct = values.astype(str)
    if set(direct.tolist()).issubset(predicted_chains) and set(direct.tolist()):
        return direct, None

    explicit_map = prediction.get("token_chain_label_map")
    if explicit_map is None:
        explicit_map = _lookup(payload, "token_chain_label_map", "asym_id_to_chain")
    if isinstance(explicit_map, Mapping):
        mapped = np.asarray(
            [str(explicit_map.get(str(value), explicit_map.get(value, ""))) for value in values],
            dtype=str,
        )
        if set(mapped.tolist()).issubset(predicted_chains) and all(mapped.tolist()):
            return mapped, None
        return None, "PAE token labels have an incomplete explicit chain map"

    declared = _lookup(payload, "per_chain_token_count")
    if not isinstance(declared, Mapping):
        return None, "numeric PAE token labels have no explicit chain map or token counts"
    unique_labels = list(dict.fromkeys(values.tolist()))
    counts = {str(label): int(np.sum(values == label)) for label in unique_labels}
    remaining = set(predicted_chains)
    mapped_by_label: dict[str, str] = {}
    for label, count in counts.items():
        candidates = [
            str(chain) for chain, declared_count in declared.items()
            if str(chain) in remaining and int(declared_count) == count
        ]
        if len(candidates) != 1:
            return None, "numeric PAE token labels are not uniquely identified by declared counts"
        mapped_by_label[label] = candidates[0]
        remaining.remove(candidates[0])
    mapped = np.asarray([mapped_by_label[str(value)] for value in values], dtype=str)
    return mapped, None


def _confidence_metrics(
    metrics: dict[str, Any],
    reasons: dict[str, str],
    prediction: Mapping[str, Any],
    reference: Any,
    predicted: Any,
    chain_map: Mapping[str, str],
    design_chains: Sequence[str],
    input_data: ValidationInput | None = None,
    metric_provenance: dict[str, Any] | None = None,
) -> None:
    scoped_names = ("min_design_target_pae_angstrom", "min_target_design_pae_angstrom")
    for name in scoped_names:
        metrics[name] = None

    def scoped_unavailable(reason: str) -> None:
        for name in scoped_names:
            reasons[name] = reason

    antibody_requested = (
        input_data is not None
        and isinstance(getattr(input_data, "row", None), Mapping)
        and "heavy_role" in input_data.row
    )

    def antibody_unavailable(reason: str) -> None:
        if antibody_requested:
            _missing(metrics, reasons, "antibody_antigen_pae_angstrom", reason)

    payload = _confidence_payload(prediction)
    plddt = _lookup(payload, "plddt", "mean_plddt", "mean_pLDDT")
    if plddt is None:
        _missing(metrics, reasons, "plddt", "prediction confidence has no pLDDT")
    else:
        values = np.asarray(plddt, dtype=np.float64)
        finite = values[np.isfinite(values)] if values.ndim else values.reshape(1)
        if not len(finite):
            _missing(metrics, reasons, "plddt", "pLDDT contains no finite values")
        else:
            value = float(np.mean(finite))
            metrics["plddt"] = value / 100.0 if value > 1.0 else value
    for name, aliases in {
        "ptm": ("ptm", "pTM", "ptm_score"),
        "iptm": ("iptm", "ipTM", "iptm_score", "interface_ptm"),
        "binder_ptm": ("binder_ptm", "binder_pTM"),
        "binder_iptm": ("binder_iptm", "binder_ipTM"),
    }.items():
        value = _finite_scalar(_lookup(payload, *aliases))
        if value is None:
            _missing(metrics, reasons, name, f"prediction confidence has no {name}")
        else:
            metrics[name] = value

    # These paper-defined AF2ig scalars are deliberately separate from
    # binder pLDDT and native ColabDesign iPAE. Missing values stay unavailable.
    total_plddt = _finite_scalar(_lookup(payload, "plddt_total"))
    if total_plddt is not None and 0.0 <= total_plddt <= 1.0:
        metrics["plddt_total"] = total_plddt
    else:
        _missing(metrics, reasons, "plddt_total", "prediction confidence has no valid total complex pLDDT")
    interaction_pae = _finite_scalar(_lookup(payload, "pae_interaction"))
    if interaction_pae is not None and interaction_pae >= 0.0:
        metrics["pae_interaction"] = interaction_pae
    else:
        _missing(metrics, reasons, "pae_interaction", "prediction confidence has no bidirectional mean interface PAE")

    direct_angstrom = _finite_scalar(_lookup(payload, "ipae_angstrom", "interface_pae"))
    direct_normalized = _finite_scalar(_lookup(payload, "ipae_normalized", "normalized_ipae", "ipae"))
    if direct_angstrom is not None:
        metrics["ipae_angstrom"] = direct_angstrom
    if direct_normalized is not None:
        metrics["ipae_normalized"] = direct_normalized
    pae = _lookup(payload, "pae_matrix", "predicted_aligned_error", "predicted_aligned_error_matrix", "pae")
    if pae is None:
        if direct_angstrom is None:
            _missing(metrics, reasons, "ipae_angstrom", "prediction confidence has no PAE matrix")
        if direct_normalized is None:
            _missing(metrics, reasons, "ipae_normalized", "prediction confidence has no normalized iPAE")
        _missing(metrics, reasons, "min_interchain_pae_angstrom", "prediction confidence has no PAE matrix")
        scoped_unavailable("prediction confidence has no PAE matrix")
        antibody_unavailable("prediction confidence has no PAE matrix")
        return
    try:
        matrix = np.asarray(pae, dtype=np.float64)
    except (TypeError, ValueError):
        matrix = np.asarray([])
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        reason = "PAE is not a square matrix"
        if direct_angstrom is None:
            _missing(metrics, reasons, "ipae_angstrom", reason)
        if direct_normalized is None:
            _missing(metrics, reasons, "ipae_normalized", reason)
        _missing(metrics, reasons, "min_interchain_pae_angstrom", reason)
        scoped_unavailable(reason)
        antibody_unavailable(reason)
        return
    positions: dict[str, np.ndarray] = {}
    inverse_chain_map = {pred_chain: ref_chain for ref_chain, pred_chain in chain_map.items()}
    token_labels, token_label_reason = _resolve_token_chain_labels(
        payload, prediction, predicted, matrix.shape[0]
    )
    if token_labels is not None:
        # Native backend chain provenance is sufficient for chain-level PAE.
        # Symmetric ligand atoms can lack a unique coordinate correspondence
        # while every PAE token still has an unambiguous submitted chain label.
        declared = prediction.get("_confidence_chain_map", {})
        if isinstance(declared, Mapping):
            candidates = {str(ref): str(pred) for ref, pred in declared.items()}
            for ref_chain, pred_chain in candidates.items():
                if (
                    list(candidates.values()).count(pred_chain) == 1
                    and ref_chain in _ordered_chains(reference)
                    and pred_chain in _ordered_chains(predicted)
                    and _is_ligand_chain(reference, ref_chain)
                    and _is_ligand_chain(predicted, pred_chain)
                    and (pred_chain not in inverse_chain_map
                         or inverse_chain_map[pred_chain] == ref_chain)
                    and (ref_chain not in inverse_chain_map.values()
                         or inverse_chain_map.get(pred_chain) == ref_chain)
                ):
                    inverse_chain_map[pred_chain] = ref_chain
        for pred_chain, ref_chain in inverse_chain_map.items():
            positions[ref_chain] = np.flatnonzero(token_labels == str(pred_chain)).astype(np.int64)
    else:
        # AF2/PipeTools writes the PDB in the same context+design order as its
        # PAE tensor.  Use the prediction's actual chain order, never the
        # generation input order (which may be design+context).
        if token_label_reason is not None and _lookup(
            payload, "token_chain_ids", "pae_chain_ids", "chain_ids", "asym_id"
        ) is not None:
            reason = token_label_reason
            if direct_angstrom is None:
                _missing(metrics, reasons, "ipae_angstrom", reason)
            if direct_normalized is None:
                _missing(metrics, reasons, "ipae_normalized", reason)
            _missing(metrics, reasons, "min_interchain_pae_angstrom", reason)
            scoped_unavailable(reason)
            antibody_unavailable(reason)
            return
        # AF2/PipeTools writes the PDB in the same context+design order as its
        # PAE tensor. Use that order only when no token identity was supplied.
        offset = 0
        for pred_chain in _ordered_chains(predicted):
            ref_chain = inverse_chain_map.get(pred_chain)
            count = len(_residue_rows(predicted, pred_chain))
            if ref_chain is not None:
                positions[ref_chain] = np.arange(offset, offset + count, dtype=np.int64)
            offset += count
        if offset != matrix.shape[0]:
            positions = {}
    if not positions or sum(len(value) for value in positions.values()) != matrix.shape[0]:
        reason = "PAE dimensions do not match prediction chain/token order"
        if direct_angstrom is None:
            _missing(metrics, reasons, "ipae_angstrom", reason)
        if direct_normalized is None:
            _missing(metrics, reasons, "ipae_normalized", reason)
        _missing(metrics, reasons, "min_interchain_pae_angstrom", reason)
        scoped_unavailable(reason)
        antibody_unavailable(reason)
        return

    from .benchmark_metrics import antibody_antigen_pae, scoped_pae
    if antibody_requested:
        try:
            metrics["antibody_antigen_pae_angstrom"] = antibody_antigen_pae(
                matrix, positions, reference, input_data.row
            )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            antibody_unavailable(str(exc))
    for name, direction in (("min_design_target_pae_angstrom", "bidirectional"),
                            ("min_target_design_pae_angstrom", "target_to_design")):
        try:
            value, pairs = scoped_pae(matrix, positions, reference, design_chains, direction=direction)
            metrics[name] = value
            reasons.pop(name, None)
            if direction == "bidirectional":
                metrics["design_target_pae_pairs"] = [list(pair) for pair in pairs]
            elif metric_provenance is not None:
                metric_provenance[name] = {"direction": direction,
                                           "pairs": [[right, left] for left, right in pairs]}
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            reasons[name] = str(exc)

    # AlphaProteo's inter-chain minimum is taken over every distinct chain
    # pair, rather than only binder/context pairs. Keep that scope separate
    # from the native iPAE summary below, whose definition is binder/context.
    all_interchain_values: list[np.ndarray] = []
    reference_chains = [
        chain for chain in _ordered_chains(reference) if chain in positions
    ]
    for index, left_chain in enumerate(reference_chains):
        for right_chain in reference_chains[index + 1 :]:
            left_idx = positions[left_chain]
            right_idx = positions[right_chain]
            all_interchain_values.append(matrix[np.ix_(left_idx, right_idx)].ravel())
            all_interchain_values.append(matrix[np.ix_(right_idx, left_idx)].ravel())
    if all_interchain_values:
        flattened = np.concatenate(all_interchain_values)
        flattened = flattened[np.isfinite(flattened)]
        if len(flattened):
            metrics["min_interchain_pae_angstrom"] = float(np.min(flattened))
        else:
            _missing(
                metrics,
                reasons,
                "min_interchain_pae_angstrom",
                "inter-chain PAE contains no finite values",
            )
    else:
        _missing(
            metrics,
            reasons,
            "min_interchain_pae_angstrom",
            "prediction has no distinct inter-chain pair",
        )

    design_idx = np.concatenate([positions[chain] for chain in design_chains if chain in positions]) if any(chain in positions for chain in design_chains) else np.asarray([], dtype=np.int64)
    context_idx = np.concatenate([values for chain, values in positions.items() if chain not in design_chains]) if len(positions) > len(design_chains) else np.asarray([], dtype=np.int64)
    if not len(design_idx) or not len(context_idx):
        reason = "design/context chain positions unavailable for inter-chain PAE"
        if direct_angstrom is None:
            _missing(metrics, reasons, "ipae_angstrom", reason)
        if direct_normalized is None:
            _missing(metrics, reasons, "ipae_normalized", reason)
        # A joint protein–ligand design has no fixed context, but its two
        # generated chains still define an inter-chain PAE minimum above.
        return
    values = np.concatenate((matrix[np.ix_(design_idx, context_idx)].ravel(), matrix[np.ix_(context_idx, design_idx)].ravel()))
    values = values[np.isfinite(values)]
    if not len(values):
        if "min_interchain_pae_angstrom" not in metrics:
            _missing(metrics, reasons, "min_interchain_pae_angstrom", "inter-chain PAE contains no finite values")
        return
    mean_value = float(np.mean(values))
    if direct_angstrom is None:
        metrics["ipae_angstrom"] = mean_value
    if direct_normalized is None:
        metrics["ipae_normalized"] = mean_value / 31.0


def _motif_reference_chain(
    input_data: ValidationInput,
    reference: Any,
    input_chain: str,
    explicit_map: Mapping[str, Any] | None = None,
) -> str | None:
    """Resolve a motif chain in a source/reference structure.

    The compiler's final ``contig_string`` is expressed in output-chain
    coordinates, while an explicitly supplied source motif can retain its
    original chain label. Exact labels win; a rewritten label is accepted
    only when a unique full-sequence polymer match proves the correspondence.
    """

    if explicit_map and input_chain in explicit_map:
        candidate = str(explicit_map[input_chain])
        if candidate in _ordered_chains(reference):
            return candidate
    available = _ordered_chains(reference)
    # ``recover_reference`` deliberately returns only the requested motif
    # residues, so its exact relabeled output chain is a canonical identity
    # even though its sequence is a subset of the full input chain.
    if input_chain in available:
        return input_chain
    expected = _chain_sequence(input_data.atoms, input_chain)
    if not expected:
        return None
    # A source/reference chain may have a rewritten label. Resolve that only
    # by a unique full-sequence match; equal residue counts are insufficient.
    candidates = [
        chain for chain in available
        if _polymer_chain(reference, chain)
        and _sequence_matches(reference, chain, expected)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _predicted_motif_residue(
    input_data: ValidationInput, predicted: Any, chain_map: Mapping[str, str],
    chain: str, position: int, ins_code: str,
) -> Mapping[str, Any] | None:
    """Map a final input locator through the verified full-chain sequence.

    Refolding backends may renumber residues. ``chain_map`` has already
    verified the complete requested sequence, so its residue ordinals are
    authoritative while the prediction's numeric labels are not.
    """
    pred_chain = chain_map.get(chain)
    if pred_chain is None:
        return None
    original = _residue_rows(input_data.atoms, chain)
    refolded = _residue_rows(predicted, pred_chain)
    if len(original) != len(refolded):
        return None
    selected = _residue_at_position(original, position, ins_code)
    if selected is None:
        return None
    return refolded[original.index(selected)]


def _motif_pairs(
    input_data: ValidationInput,
    predicted: Any,
    chain_map: Mapping[str, str],
    *,
    reference: Any,
    reference_chain_map: Mapping[str, Any] | None = None,
    atom_names_override: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    ref_points: list[np.ndarray] = []
    pred_points: list[np.ndarray] = []
    matched: list[tuple[str, int, str]] = []
    missing: list[tuple[str, int, str]] = []
    for chain, positions in input_data.motif_positions.items():
        pred_chain = chain_map.get(chain)
        ref_chain = _motif_reference_chain(
            input_data, reference, chain, explicit_map=reference_chain_map
        )
        if pred_chain is None or ref_chain is None:
            missing.extend((chain, int(position), ins) for position, ins in positions)
            continue
        ref_residues = _residue_rows(reference, ref_chain)
        by_label = {
            (int(item["label_seq"]), str(item["ins_code"])): index
            for index, item in enumerate(ref_residues)
        }
        selected_atoms = input_data.tip_positions.get(chain, {})
        for position, ins_code in positions:
            ref_index = by_label.get((int(position), str(ins_code)))
            if ref_index is None and not ins_code:
                candidates = [index for index, item in enumerate(ref_residues) if int(item["label_seq"]) == int(position)]
                ref_index = candidates[0] if len(candidates) == 1 else None
            if ref_index is None:
                missing.append((chain, int(position), str(ins_code)))
                continue
            ref_residue = ref_residues[ref_index]
            pred_residue = _predicted_motif_residue(
                input_data, predicted, chain_map, chain, int(position), str(ins_code)
            )
            if pred_residue is None:
                missing.append((chain, int(position), str(ins_code)))
                continue
            atom_names = atom_names_override if atom_names_override is not None else selected_atoms.get((int(position), str(ins_code)))
            if atom_names is None:
                atom_names = tuple(sorted(ref_residue["rows"]))
            for atom_name in atom_names:
                ref_row = ref_residue["rows"].get(atom_name)
                pred_row = pred_residue["rows"].get(atom_name)
                if ref_row is None or pred_row is None:
                    missing.append((chain, int(position), str(atom_name)))
                    continue
                if not _valid_coord(reference, ref_row) or not _valid_coord(predicted, pred_row):
                    missing.append((chain, int(position), str(atom_name)))
                    continue
                ref_points.append(np.asarray(reference.coord[ref_row], dtype=np.float64))
                pred_points.append(np.asarray(predicted.coord[pred_row], dtype=np.float64))
                matched.append((chain, int(position), str(atom_name)))
    return (
        np.asarray(ref_points, dtype=np.float64).reshape((-1, 3)),
        np.asarray(pred_points, dtype=np.float64).reshape((-1, 3)),
        matched,
        missing,
    )


def _motif_reference_prediction(
    input_data: ValidationInput,
    prediction: Mapping[str, Any],
    variant: Mapping[str, Any] | None,
) -> Any | None:
    candidate = prediction.get("motif_reference_path")
    if candidate is None and variant:
        candidate = variant.get("motif_reference_path")
    if candidate is None:
        # A row-level reference is explicit metadata from the generation
        # ledger. It is intentionally never discovered by scanning
        # ``original`` or sibling task directories.
        candidate = input_data.row.get("motif_reference_path")
    if candidate is None:
        return None
    reference_path = Path(candidate)
    was_relative = not reference_path.is_absolute()
    if was_relative:
        reference_path = input_data.task_dir / reference_path
    try:
        reference_path = reference_path.resolve()
    except ValueError:
        return None
    if was_relative:
        try:
            reference_path.relative_to(input_data.task_dir.resolve())
        except ValueError:
            return None
    try:
        return read_structure(reference_path)
    except (OSError, ValueError, TypeError):
        return None


def _motif_reference_requested(
    input_data: ValidationInput,
    prediction: Mapping[str, Any],
    variant: Mapping[str, Any] | None,
) -> bool:
    if prediction.get("motif_reference_path") is not None:
        return True
    if variant and variant.get("motif_reference_path") is not None:
        return True
    return bool(input_data.row.get("motif_reference_path"))


def _one_letter_residue(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if text in _AA3_TO_1:
        return _AA3_TO_1[text]
    if len(text) == 1 and text in set(_AA3_TO_1.values()):
        return text
    return None


def _residue_at_position(
    residues: Sequence[Mapping[str, Any]], position: int, ins_code: str
) -> Mapping[str, Any] | None:
    exact = [
        item for item in residues
        if int(item["label_seq"]) == int(position)
        and str(item["ins_code"]) == str(ins_code)
    ]
    if len(exact) == 1:
        return exact[0]
    if not ins_code:
        uninserted = [
            item for item in residues
            if int(item["label_seq"]) == int(position)
        ]
        if len(uninserted) == 1:
            return uninserted[0]
    return None


def _fixed_restype_by_position(
    input_data: ValidationInput,
) -> dict[tuple[str, int, str], str]:
    result: dict[tuple[str, int, str], str] = {}
    for locator in input_data.fixed_restypes:
        chain = str(locator.get("chain", "")).strip()
        raw_position = locator.get("res_id", locator.get("resid"))
        residue = locator.get("restype", locator.get("res_name"))
        if not chain or raw_position in (None, "", ".", "?") or residue in (None, ""):
            continue
        try:
            position = int(raw_position)
        except (TypeError, ValueError):
            continue
        one_letter = _one_letter_residue(residue)
        if one_letter is not None:
            result[(chain, position, str(locator.get("ins_code", "") or ""))] = one_letter
    return result


def _variant_residue_at_position(
    input_data: ValidationInput,
    sequences: Mapping[str, str],
    chain: str,
    position: int,
    ins_code: str,
) -> str | None:
    sequence = sequences.get(chain)
    if not sequence:
        return None
    residues = _residue_rows(input_data.atoms, chain)
    source_residue = _residue_at_position(residues, position, ins_code)
    if source_residue is None:
        return None
    ordinal = residues.index(source_residue)
    return _one_letter_residue(sequence[ordinal]) if ordinal < len(sequence) else None


def _motif_sequence_identities(
    input_data: ValidationInput,
    variant: Mapping[str, Any] | None,
    motif_reference: Any | None,
    *,
    motif_reference_requested: bool = False,
) -> tuple[list[tuple[str, int, str, str | None, str | None]], list[str]]:
    """Return source and variant identities for every requested motif residue.

    ``contig_string`` positions are final output positions. Fixed restype
    records override source coordinates; otherwise the loaded generated
    structure is the source of truth. An explicit motif reference can fill
    source identities only when its chain label or full sequence proves the
    correspondence.
    """

    sequences = _variant_sequences(variant)
    if not sequences:
        row_sequence = str(input_data.row.get("design_sequence", "")).strip()
        if row_sequence and len(input_data.design_chains) == 1:
            sequences = {input_data.design_chains[0]: row_sequence.upper()}
    fixed = _fixed_restype_by_position(input_data)
    values: list[tuple[str, int, str, str | None, str | None]] = []
    missing: list[str] = []
    for chain, positions in input_data.motif_positions.items():
        input_residues = _residue_rows(input_data.atoms, chain)
        if motif_reference is not None:
            source_chain = _motif_reference_chain(input_data, motif_reference, chain)
        else:
            source_chain = None if motif_reference_requested else chain
        if motif_reference is None and not motif_reference_requested:
            source_residues = input_residues
        elif source_chain is not None:
            source_residues = _residue_rows(motif_reference, source_chain)
        else:
            # An explicit source that cannot be mapped must remain unknown;
            # silently falling back to the generated design would make a
            # missing source identity look like a successful motif recovery.
            source_residues = []
        for position, ins_code in positions:
            key = (chain, int(position), str(ins_code))
            expected = fixed.get(key)
            if expected is None:
                source_residue = _residue_at_position(source_residues, position, ins_code)
                if source_residue is not None:
                    expected = _one_letter_residue(source_residue["res_name"])
            variant_identity = _variant_residue_at_position(
                input_data, sequences, chain, position, ins_code
            )
            if expected is None:
                missing.append(f"{chain}{position}{ins_code}: source residue identity unavailable")
            values.append((chain, int(position), str(ins_code), expected, variant_identity))
    return values, missing


def _sequence_metrics(
    metrics: dict[str, Any],
    reasons: dict[str, str],
    input_data: ValidationInput,
    predicted: Any,
    chain_map: Mapping[str, str],
    variant: Mapping[str, Any] | None,
    motif_reference: Any | None = None,
    motif_reference_requested: bool = False,
) -> None:
    sequences = _variant_sequences(variant)
    if not sequences:
        row_sequence = input_data.row.get("design_sequence", "").strip()
        if row_sequence and len(input_data.design_chains) == 1:
            sequences = {input_data.design_chains[0]: row_sequence.upper()}
    recoveries: list[float] = []
    for chain in input_data.design_chains:
        expected = sequences.get(chain)
        original = _chain_sequence(input_data.atoms, chain)
        if expected is None:
            _missing(metrics, reasons, "sequence_recovery", f"no variant sequence for design chain {chain!r}")
            continue
        n = min(len(expected), len(original))
        if n == 0 or len(expected) != len(original):
            _missing(metrics, reasons, "sequence_recovery", f"variant/reference sequence length differs for chain {chain!r}")
            continue
        recoveries.append(sum(a == b for a, b in zip(expected, original, strict=True)) / n)
    if recoveries:
        metrics["sequence_recovery"] = float(np.mean(recoveries))

    if not input_data.motif_positions:
        _missing(metrics, reasons, "correct_motif_sequence", "no resolved contig motif residues")
        _missing(metrics, reasons, "motif_sequence_recovery", "no resolved contig motif residues")
        return

    identities, source_missing = _motif_sequence_identities(
        input_data,
        variant,
        motif_reference,
        motif_reference_requested=motif_reference_requested,
    )
    predicted_missing: list[str] = []
    predicted_values: list[str | None] = []
    for chain, position, ins_code, _source_identity, _variant_identity in identities:
        pred_residue = _predicted_motif_residue(
            input_data, predicted, chain_map, chain, position, ins_code
        )
        value = _one_letter_residue(pred_residue["res_name"]) if pred_residue else None
        predicted_values.append(value)
        if value is None:
            predicted_missing.append(f"{chain}{position}{ins_code}: predicted residue unavailable")

    unknown = source_missing + predicted_missing
    if unknown:
        reason = "motif sequence identity unavailable: " + "; ".join(unknown[:4])
        _missing(metrics, reasons, "correct_motif_sequence", reason)
        _missing(metrics, reasons, "motif_sequence_recovery", reason)
        return

    expected_values = [item[3] for item in identities]
    variant_values = [item[4] for item in identities]
    if any(value is None for value in expected_values):
        reason = "motif source identity is unknown"
        _missing(metrics, reasons, "correct_motif_sequence", reason)
        _missing(metrics, reasons, "motif_sequence_recovery", reason)
        return
    # If an explicit MPNN/GT variant sequence is supplied, it is itself part
    # of the identity contract: a refold cannot pass after silently changing
    # the requested fixed motif sequence.
    mismatches = [
        index for index, (expected, observed, variant_value) in enumerate(
            zip(expected_values, predicted_values, variant_values, strict=True)
        )
        if observed != expected or (variant_value is not None and variant_value != expected)
    ]
    correct = not mismatches
    metrics["correct_motif_sequence"] = correct
    metrics["motif_sequence_recovery"] = float(
        sum(index not in mismatches for index in range(len(expected_values)))
        / len(expected_values)
    )


def _ligand_clashes(predicted: Any) -> tuple[bool | None, str | None]:
    """Check raw protein-ligand distances using Complexa's 1.5 A radius.

    Invalid coordinate rows are excluded from the distance query. If any
    protein/ligand atom is invalid and no clash is observed among the valid
    rows, the answer is unknown: the available coordinates cannot prove that
    an omitted atom is clash-free.
    """

    mol = np.asarray(predicted.mol_type).astype(str)
    protein = np.flatnonzero(mol == "protein")
    ligand = np.flatnonzero(mol == "ligand")
    if not len(protein) or not len(ligand):
        return None, "prediction has no protein-ligand atom pair"
    coordinates = np.asarray(predicted.coord, dtype=np.float64)
    coordinate_valid = np.isfinite(coordinates).all(axis=1)
    declared_valid = getattr(predicted, "coord_valid", None)
    if declared_valid is not None:
        coordinate_valid &= np.asarray(declared_valid, dtype=bool)
    protein_valid = coordinate_valid[protein]
    ligand_valid = coordinate_valid[ligand]
    protein_coords = coordinates[protein][protein_valid]
    ligand_coords = coordinates[ligand][ligand_valid]
    if not len(protein_coords) or not len(ligand_coords):
        return None, "protein-ligand coordinates are incomplete"
    # Complexa's motif-binder check is a raw protein-ligand Cartesian query at
    # 1.5 A. In particular, no covalent/bonded-pair exclusion is applied. The
    # query radius is inclusive, matching scipy.spatial.cKDTree.query_ball_point.
    delta = protein_coords[:, None, :] - ligand_coords[None, :, :]
    has_clash = bool(np.any(np.sum(delta * delta, axis=-1) <= 1.5 * 1.5))
    if has_clash:
        return True, None
    if not bool(np.all(protein_valid)) or not bool(np.all(ligand_valid)):
        return None, "protein-ligand coordinates are incomplete; no clash can be proven"
    return False, None


def _mapped_ligand_points(
    reference: Any,
    predicted: Any,
    reference_chain: str,
    predicted_chain: str,
    atom_map: Mapping[int, int],
    *,
    allow_predicted_extra: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return mapped ligand heavy-atom points and observed-reference coverage."""

    try:
        ref_labels, pred_labels = _chain_values(reference), _chain_values(predicted)
        ref_modalities = np.asarray(getattr(reference, "mol_type", np.full(ref_labels.shape, "ligand"))).astype(str)
        pred_modalities = np.asarray(getattr(predicted, "mol_type", np.full(pred_labels.shape, "ligand"))).astype(str)
        ref_indices = {
            int(index)
            for index in np.flatnonzero((ref_labels == str(reference_chain)) & (ref_modalities == "ligand")).tolist()
            if _atom_element(reference, int(index)) not in {"H", "D"}
        }
        pred_indices = {
            int(index)
            for index in np.flatnonzero((pred_labels == str(predicted_chain)) & (pred_modalities == "ligand")).tolist()
            if _atom_element(predicted, int(index)) not in {"H", "D"}
        }
    except (TypeError, ValueError):
        return np.empty((0, 3)), np.empty((0, 3)), 0, 1
    expected = len(ref_indices)
    if expected == 0 or (not allow_predicted_extra and len(pred_indices) != expected) or set(atom_map) != ref_indices:
        return np.empty((0, 3)), np.empty((0, 3)), 0, expected
    ref_points: list[np.ndarray] = []
    pred_points: list[np.ndarray] = []
    for ref_index in sorted(ref_indices):
        pred_index = atom_map.get(ref_index)
        if pred_index is None or pred_index not in pred_indices:
            continue
        if not _valid_coord(reference, ref_index) or not _valid_coord(predicted, pred_index):
            continue
        ref_points.append(np.asarray(reference.coord[ref_index], dtype=np.float64))
        pred_points.append(np.asarray(predicted.coord[pred_index], dtype=np.float64))
    return (
        np.asarray(ref_points, dtype=np.float64).reshape((-1, 3)),
        np.asarray(pred_points, dtype=np.float64).reshape((-1, 3)),
        len(ref_points),
        expected,
    )


def _protein_chain_set(atoms: Any) -> bool:
    chains = _ordered_chains(atoms)
    return bool(chains) and all(_chain_mol_type(atoms, chain) == "protein" for chain in chains)


def _raw_rmsd(reference: np.ndarray, predicted: np.ndarray) -> float:
    if reference.shape != predicted.shape or reference.ndim != 2 or reference.shape[1:] != (3,):
        raise MetricUnavailable("coordinate arrays have incompatible shapes")
    if not len(reference):
        raise MetricUnavailable("no paired atoms are available")
    return float(np.sqrt(np.mean(np.sum((reference - predicted) ** 2, axis=1))))



def _complexa_binder_points(reference, predicted, chain_map, binder_chains, *, all_heavy):
    """Pair binder atoms by mapped residue ordinal, identity and atom name."""
    if not binder_chains or len(set(binder_chains)) != len(binder_chains):
        raise MetricUnavailable("Complexa alignment requires unique designed protein binder chains")
    targets = [chain_map.get(chain) for chain in binder_chains]
    if any(target is None for target in targets) or len(set(targets)) != len(targets):
        raise MetricUnavailable("designed protein binder chain mapping is incomplete or ambiguous")
    ref_points, pred_points = [], []

    def residues(atoms, chain):
        result = sorted(_residue_rows(atoms, chain, polymer_only=True),
                        key=lambda residue: (residue["label_seq"], residue["ins_code"]))
        labels = [(residue["label_seq"], residue["ins_code"]) for residue in result]
        if not result or len(set(labels)) != len(labels):
            raise MetricUnavailable("binder residue correspondence is incomplete or ambiguous")
        return result

    def atom_rows(atoms, chain, residue):
        rows = np.flatnonzero((_chain_values(atoms) == chain)
                              & (np.asarray(atoms.mol_type).astype(str) == "protein")
                              & (np.asarray(atoms.residue_uid) == residue["uid"]))
        result = {}
        for raw_row in rows:
            row = int(raw_row)
            name = str(atoms.atom_name[row])
            if all_heavy:
                if name == "OXT" or _atom_element(atoms, row) in {"H", "D", "T"}:
                    continue
            elif name != "CA":
                continue
            if name in result:
                raise MetricUnavailable("binder atom identity is duplicated")
            result[name] = row
        if not result:
            raise MetricUnavailable("binder heavy atoms are absent" if all_heavy
                                    else "binder CA coverage is incomplete")
        return result

    for chain, target in zip(binder_chains, targets, strict=True):
        if (_chain_mol_type(reference, chain) != "protein"
                or _chain_mol_type(predicted, target) != "protein"):
            raise MetricUnavailable("binder chain mapping does not identify protein")
        original, folded = residues(reference, chain), residues(predicted, target)
        if len(original) != len(folded):
            raise MetricUnavailable("binder residue correspondence is incomplete")
        for ordinal, (ref_residue, pred_residue) in enumerate(zip(original, folded, strict=True), 1):
            ref_rows = atom_rows(reference, chain, ref_residue)
            pred_rows = atom_rows(predicted, target, pred_residue)
            if all_heavy:
                if str(ref_residue["res_name"]).upper() != str(pred_residue["res_name"]).upper():
                    raise MetricUnavailable(f"binder residue identity differs at {chain} ordinal {ordinal}")
                ref_identity = {(name, _atom_element(reference, row)) for name, row in ref_rows.items()}
                pred_identity = {(name, _atom_element(predicted, row)) for name, row in pred_rows.items()}
                if ref_identity != pred_identity:
                    raise MetricUnavailable(f"binder heavy-atom identity sets differ at {chain} ordinal {ordinal}")
            if set(ref_rows) != set(pred_rows):
                raise MetricUnavailable("binder CA coverage is incomplete")
            for name in sorted(ref_rows):
                ref_row, pred_row = ref_rows[name], pred_rows[name]
                if not _valid_coord(reference, ref_row) or not _valid_coord(predicted, pred_row):
                    raise MetricUnavailable("binder heavy-atom coordinates are incomplete" if all_heavy
                                            else "binder CA coordinates are incomplete")
                ref_points.append(reference.coord[ref_row])
                pred_points.append(predicted.coord[pred_row])
    return np.asarray(ref_points, dtype=np.float64), np.asarray(pred_points, dtype=np.float64)


def _complexa_ligand_alignment(reference, predicted, chain_map, binder_chains, ligand_ref, ligand_pred):
    """Use a complete identity-matched binder heavy fit, then a CA fallback."""
    from .benchmark_metrics import _require_rigid_fit_geometry

    provenance = {"fit": None, "binder_chains": list(binder_chains), "fallback_reason": None}
    try:
        fit_ref, fit_pred = _complexa_binder_points(
            reference, predicted, chain_map, binder_chains, all_heavy=True)
        _require_rigid_fit_geometry(fit_ref, fit_pred, label="Complexa binder heavy-atom")
        value = _fit_then_measure(fit_ref, fit_pred, ligand_ref, ligand_pred)
        provenance.update(fit="binder_all_heavy", fit_atom_count=len(fit_ref))
        return value, None, provenance
    except MetricUnavailable as exc:
        provenance["fallback_reason"] = str(exc)
    try:
        fit_ref, fit_pred = _complexa_binder_points(
            reference, predicted, chain_map, binder_chains, all_heavy=False)
        _require_rigid_fit_geometry(fit_ref, fit_pred, label="Complexa binder CA")
        value = _fit_then_measure(fit_ref, fit_pred, ligand_ref, ligand_pred)
        provenance.update(fit="binder_ca", fit_atom_count=len(fit_ref))
        return value, None, provenance
    except MetricUnavailable as exc:
        reason = f"Complexa binder CA fallback unavailable: {exc}"
        provenance["unavailable_reason"] = reason
        return None, reason, provenance


def _protein_aligned_ligand_metrics(
    reference, predicted, protein_chain_map, ligand_maps, atom_maps, mapping_reasons,
    *, strict_ligand_chains=(), binder_chains=None, metric_provenance=None,
):
    """Ligand self-fit and receptor-fit RMSD, using generated coordinates as reference."""
    from .ligand_metrics import ligand_chains, named_atom_mapping, LigandMappingError, _heavy_indices

    names = ("ligand_rmsd", "context_aligned_ligand_rmsd", "receptor_aligned_ligand_rmsd")
    if binder_chains is not None:
        names += ("complexa_aligned_ligand_rmsd",)
    values, reasons = dict.fromkeys(names), {}
    reference_points, predicted_points = [], []
    predicted_count = 0
    unverified_ligands = {}
    for chain in ligand_chains(reference):
        target = ligand_maps.get(chain)
        mapping = atom_maps.get((chain, target)) if target is not None else None
        if mapping is None:
            unverified_ligands[chain] = mapping_reasons.get(
                chain, "ligand lacks a unique verified chemical correspondence")
            if chain in strict_ligand_chains:
                reason = mapping_reasons.get(chain, "generated ligand lacks a verified full-graph mapping")
                return {name: None for name in names}, {name: reason for name in names}
            try:
                mapping = named_atom_mapping(reference, predicted, chain)
                target = chain
            except LigandMappingError as exc:
                reason = f"{chain}: {exc}"
                return {name: None for name in names}, {name: reason for name in names}
        predicted_count += len(_heavy_indices(predicted, target)[0])
        ref, pred, matched, expected = _mapped_ligand_points(
            reference, predicted, chain, target, mapping, allow_predicted_extra=True,
        )
        reason = _coverage_reason(f"ligand {chain}", ref, pred, matched, expected)
        if reason is not None:
            return {name: None for name in names}, {name: reason for name in names}
        reference_points.append(ref)
        predicted_points.append(pred)
    if not reference_points:
        return {name: None for name in names}, {name: "no ligand chain" for name in names}
    ligand_ref, ligand_pred = np.concatenate(reference_points), np.concatenate(predicted_points)
    values.update(ligand_reference_atom_count=len(ligand_ref),
                  ligand_predicted_atom_count=predicted_count,
                  ligand_matched_atom_count=len(ligand_ref))
    try:
        values["ligand_rmsd"] = _rmsd_with_fit(ligand_ref, ligand_pred)
    except MetricUnavailable as exc:
        _missing(values, reasons, "ligand_rmsd", str(exc))
    receptor = tuple(chain for chain in _ordered_chains(reference)
                     if _chain_mol_type(reference, chain) == "protein")
    for metric, atoms, heavy in (
        ("context_aligned_ligand_rmsd", ("CA",), False),
        ("receptor_aligned_ligand_rmsd", None, True),
    ):
        ref, pred, matched, expected = _paired_points(
            reference, predicted, protein_chain_map, receptor, atoms, all_heavy=heavy,
            exclude_atom_names=("OXT",) if heavy else (),
        )
        reason = _coverage_reason("receptor alignment", ref, pred, matched, expected)
        if reason:
            _missing(values, reasons, metric, reason)
            continue
        try:
            values[metric] = _fit_then_measure(ref, pred, ligand_ref, ligand_pred)
        except MetricUnavailable as exc:
            _missing(values, reasons, metric, str(exc))
    if binder_chains is not None:
        name = "complexa_aligned_ligand_rmsd"
        if unverified_ligands:
            value = None
            reason = "Complexa ligand correspondence is unverified: " + "; ".join(
                f"{chain}: {why}" for chain, why in unverified_ligands.items())
            provenance = {"fit": None, "binder_chains": list(binder_chains),
                          "fallback_reason": None, "unavailable_reason": reason}
        else:
            value, reason, provenance = _complexa_ligand_alignment(
                reference, predicted, protein_chain_map, binder_chains, ligand_ref, ligand_pred)
        values[name] = value
        if reason is not None:
            reasons[name] = reason
        if metric_provenance is not None:
            metric_provenance[name] = provenance
    return values, reasons

def _na_design_metrics(
    metrics: dict[str, Any],
    reasons: dict[str, str],
    reference: Any,
    predicted: Any,
    chain_map: Mapping[str, str],
    design_chains: Sequence[str],
) -> None:
    """Measure nucleic-acid token centers and target-aligned design placement."""

    names = (
        "na_token_center_rmsd",
        "na_aligned_protein_ca_rmsd",
        "protein_aligned_na_p_rmsd",
    )
    modalities = [_chain_mol_type(reference, chain) for chain in design_chains]
    na_design = bool(modalities) and all(kind in {"rna", "dna"} for kind in modalities)
    protein_design = bool(modalities) and all(kind == "protein" for kind in modalities)
    context_chains = [
        chain for chain in _ordered_chains(reference) if chain not in design_chains
    ]

    if na_design:
        center_ref, center_pred, matched, expected = _paired_points(
            reference, predicted, chain_map, design_chains, ("C1'",)
        )
        reason = _coverage_reason("nucleic-acid C1' token centers", center_ref,
                                  center_pred, matched, expected)
        if reason is None:
            try:
                metrics["na_token_center_rmsd"] = _rmsd_with_fit(center_ref, center_pred)
            except MetricUnavailable as exc:
                reason = str(exc)
        if reason is not None:
            _missing(metrics, reasons, "na_token_center_rmsd", reason)

        protein_context = [
            chain for chain in context_chains
            if _chain_mol_type(reference, chain) == "protein"
        ]
        if protein_context:
            fit_ref, fit_pred, fit_matched, fit_expected = _paired_points(
                reference, predicted, chain_map, protein_context, ("CA",)
            )
            measure_ref, measure_pred, measure_matched, measure_expected = _paired_points(
                reference, predicted, chain_map, design_chains, ("P",),
                reference_valid_only=True,
            )
            reason = (_coverage_reason("protein C-alpha alignment", fit_ref, fit_pred,
                                       fit_matched, fit_expected)
                      or _coverage_reason("designed nucleic-acid phosphate", measure_ref,
                                          measure_pred, measure_matched, measure_expected))
            if reason is None:
                try:
                    metrics["protein_aligned_na_p_rmsd"] = _fit_then_measure(
                        fit_ref, fit_pred, measure_ref, measure_pred
                    )
                except MetricUnavailable as exc:
                    reason = str(exc)
            if reason is not None:
                _missing(metrics, reasons, "protein_aligned_na_p_rmsd", reason)
        else:
            _missing(metrics, reasons, "protein_aligned_na_p_rmsd",
                     "no protein context chain for alignment")
    else:
        _missing(metrics, reasons, "na_token_center_rmsd",
                 "design chains are not all nucleic acid")
        _missing(metrics, reasons, "protein_aligned_na_p_rmsd",
                 "design chains are not all nucleic acid")

    na_context = [
        chain for chain in context_chains
        if _chain_mol_type(reference, chain) in {"rna", "dna"}
    ]
    if protein_design and na_context:
        fit_ref, fit_pred, fit_matched, fit_expected = _paired_points(
            reference, predicted, chain_map, na_context, ("P",),
            reference_valid_only=True,
        )
        measure_ref, measure_pred, measure_matched, measure_expected = _paired_points(
            reference, predicted, chain_map, design_chains, ("CA",)
        )
        reason = (_coverage_reason("nucleic-acid phosphate alignment", fit_ref, fit_pred,
                                   fit_matched, fit_expected)
                  or _coverage_reason("designed protein C-alpha", measure_ref,
                                      measure_pred, measure_matched, measure_expected))
        if reason is None:
            try:
                metrics["na_aligned_protein_ca_rmsd"] = _fit_then_measure(
                    fit_ref, fit_pred, measure_ref, measure_pred
                )
            except MetricUnavailable as exc:
                reason = str(exc)
        if reason is not None:
            _missing(metrics, reasons, "na_aligned_protein_ca_rmsd", reason)
    else:
        _missing(metrics, reasons, "na_aligned_protein_ca_rmsd",
                 "requires protein design and nucleic-acid context")


def evaluate_prediction(
    input_data: ValidationInput,
    variant: Mapping[str, Any] | None,
    prediction: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Evaluate one completed complex/monomer prediction.

    The return value is intentionally a pair of plain dictionaries so the
    incremental JSON/CSV writer can serialize it without a model-specific
    object.  ``prediction`` may contain an optional ``chain_map`` and optional
    motif-reference path in addition to the public worker fields.
    """

    metrics: dict[str, Any] = {"complexa_aligned_ligand_rmsd": None}
    reasons: dict[str, str] = {}
    metric_provenance: dict[str, Any] = {}
    status = str(prediction.get("status", "ok")).lower()
    if status not in {"ok", "success", "complete", "completed"}:
        reason = str(prediction.get("error") or "prediction did not complete")
        for name in _METRIC_NAMES:
            _missing(metrics, reasons, name, reason)
        return {"metrics": metrics, "reasons": reasons}
    structure_path = prediction.get("structure_path")
    if not structure_path:
        reason = "prediction has no structure_path"
        for name in _METRIC_NAMES:
            _missing(metrics, reasons, name, reason)
        return {"metrics": metrics, "reasons": reasons}
    try:
        predicted = read_structure(Path(structure_path))
    except Exception as exc:
        reason = f"cannot read prediction structure: {exc}"
        for name in _METRIC_NAMES:
            _missing(metrics, reasons, name, reason)
        return {"metrics": metrics, "reasons": reasons}

    reference = input_data.atoms
    chain_map = _chain_map(reference, predicted, input_data, prediction, variant)
    ligand_chain_map: dict[str, str] = {}
    ligand_atom_maps: dict[tuple[str, str], dict[int, int]] = {}
    ligand_mapping_reasons: dict[str, str] = {}
    try:
        from .ligand_metrics import (LigandMappingError, context_ligand_atom_maps,
                                     ligand_chains, match_ligand_chains)
        from .ligands import LigandResolutionError, resolve_context_ligand_components

        ligand_chain_map, ligand_atom_maps, ligand_mapping_reasons = match_ligand_chains(
            reference, predicted
        )
        context_witnesses = {}
        context_smiles = {}
        witness_errors = {}
        for chain in ligand_chains(reference):
            if chain in input_data.design_chains:
                continue
            ligand_rows = np.flatnonzero((_chain_values(reference) == chain)
                                        & (np.asarray(reference.mol_type).astype(str) == "ligand"))
            component_values = getattr(reference, "component_uid", None)
            component_count = (len(set(np.asarray(component_values)[ligand_rows].tolist()))
                               if component_values is not None else 1)
            if chain in ligand_chain_map and component_count == 1 and not _polymer_chain(reference, chain):
                continue
            try:
                components, names = resolve_context_ligand_components(
                    input_data.task_dir, reference,
                    design_chains=input_data.design_chains, chains=(chain,))
                if len(components) == 1:
                    entry = next(iter(components.values()))
                    smiles = entry.get("smiles")
                    if isinstance(smiles, str) and smiles.strip():
                        context_smiles[chain] = smiles
                        continue
                # Predicted extra atoms are authorized only by known, source-
                # resolved CCD components. Generated/joint ligands stay strict.
                if not components or any(len(entry.get("ccd", ())) != 1
                                         for entry in components.values()):
                    continue
                polymer_mapping = ({chain: chain_map[chain]}
                                   if _polymer_chain(reference, chain) and chain in chain_map
                                   else None)
                context_witnesses.update(context_ligand_atom_maps(
                    reference, predicted, components, names,
                    predicted_chain_map=polymer_mapping))
            except (LigandResolutionError, LigandMappingError) as exc:
                witness_errors[chain] = str(exc)
        if context_witnesses or context_smiles:
            ligand_chain_map, ligand_atom_maps, ligand_mapping_reasons = match_ligand_chains(
                reference, predicted, authoritative_context_maps=context_witnesses,
                context_smiles=context_smiles)
        for chain, reason in witness_errors.items():
            if chain not in ligand_chain_map:
                ligand_mapping_reasons[chain] = f"context ligand witness unavailable: {reason}"
    except (ImportError, TypeError, ValueError) as exc:
        ligand_mapping_reasons = {
            chain: f"ligand graph mapping unavailable: {exc}"
            for chain in _ordered_chains(reference)
            if _is_ligand_chain(reference, chain)
        }
    for chain in ligand_chains(reference):
        mapped_chain = ligand_chain_map.get(chain)
        if _polymer_chain(reference, chain):
            # The verified complete polymer sequence remains an alignment
            # authority when heterogens share its public chain. A ligand-map
            # failure remains unavailable for ligand metrics only.
            if mapped_chain is not None and chain_map.get(chain) != mapped_chain:
                ligand_chain_map.pop(chain, None)
                ligand_mapping_reasons[chain] = "ligand and polymer chain identities disagree"
            continue
        explicit_chain_map = prediction.get("chain_map")
        if mapped_chain is None or (
            isinstance(explicit_chain_map, Mapping)
            and explicit_chain_map.get(chain) != mapped_chain
        ):
            chain_map.pop(chain, None)
        else:
            chain_map[chain] = mapped_chain
    design_chains = tuple(input_data.design_chains)
    if not design_chains:
        _missing(metrics, reasons, "design_rmsd", "no design_chain_id or joint design-chain ledger")
        _missing(metrics, reasons, "binder_design_rmsd", "no design chain")
        _missing(metrics, reasons, "context_aligned_design_rmsd", "no design chain")
        _missing(metrics, reasons, "complex_design_rmsd", "no design chain")
        _missing(metrics, reasons, "ligand_rmsd", "no design chain")
        _missing(metrics, reasons, "context_aligned_ligand_rmsd", "no design chain")
        _missing(metrics, reasons, "binder_scRMSD_bb3", "no design chain")
    else:
        design_modalities = [
            _chain_mol_type(reference, chain)
            for chain in design_chains
            if np.any(_chain_values(reference) == chain)
        ]
        design_is_protein = bool(design_modalities) and all(
            modality == "protein" for modality in design_modalities
        )
        design_is_protein_ligand = (
            "protein" in design_modalities
            and "ligand" in design_modalities
            and all(modality in {"protein", "ligand"} for modality in design_modalities)
        )
        design_protein_chains = tuple(
            chain for chain in design_chains
            if _chain_mol_type(reference, chain) == "protein"
        )
        design_is_ligand = bool(design_modalities) and all(
            modality == "ligand" for modality in design_modalities
        )
        context_chains = tuple(
            chain for chain in _ordered_chains(reference) if chain not in design_chains
        )
        if design_is_protein or design_is_protein_ligand:
            design_ref, design_pred, design_matched, design_expected = _paired_points(
                reference, predicted, chain_map, design_protein_chains, _BACKBONE3
            )
            binder_bb3 = True
        design_coverage_reason = _coverage_reason(
            "design", design_ref, design_pred, design_matched, design_expected
        ) if (design_is_protein or design_is_protein_ligand) else "design protein backbone metric is not applicable"
        if design_is_protein or design_is_protein_ligand:
            try:
                if design_coverage_reason is not None:
                    raise MetricUnavailable(design_coverage_reason)
                metrics["design_rmsd"] = _rmsd_with_fit(design_ref, design_pred)
                metrics["binder_scRMSD_bb3"] = metrics["design_rmsd"]
            except MetricUnavailable as exc:
                _missing(metrics, reasons, "design_rmsd", str(exc))
                _missing(metrics, reasons, "binder_scRMSD_bb3", str(exc))

            binder_ref, binder_pred, binder_matched, binder_expected = _paired_points(
                reference, predicted, chain_map, design_protein_chains, ("CA",)
            )
            binder_coverage_reason = _coverage_reason(
                "binder CA", binder_ref, binder_pred, binder_matched, binder_expected
            )
            try:
                if binder_coverage_reason is not None:
                    raise MetricUnavailable(binder_coverage_reason)
                # PXDesignBench/AF2-opt's binder designability RMSD is the
                # binder CA-only fit; AME's separate bb3 metric remains above.
                metrics["binder_design_rmsd"] = _rmsd_with_fit(binder_ref, binder_pred)
            except MetricUnavailable as exc:
                _missing(metrics, reasons, "binder_design_rmsd", str(exc))

            context_ref, context_pred, context_matched, context_expected = _paired_points(
                reference, predicted, chain_map, context_chains, ("CA",)
            )
            context_coverage_reason = _coverage_reason(
                "context", context_ref, context_pred, context_matched, context_expected
            )
            if context_coverage_reason is None and design_coverage_reason is None:
                try:
                    metrics["context_aligned_design_rmsd"] = _fit_then_measure(
                        context_ref, context_pred, design_ref, design_pred
                    )
                except MetricUnavailable as exc:
                    _missing(metrics, reasons, "context_aligned_design_rmsd", str(exc))
            else:
                _missing(
                    metrics,
                    reasons,
                    "context_aligned_design_rmsd",
                    context_coverage_reason or design_coverage_reason or "incomplete atom coverage",
                )

            if _protein_chain_set(reference) and _protein_chain_set(predicted):
                full_chains = _ordered_chains(reference)
                full_ref, full_pred, full_matched, full_expected = _paired_points(
                    reference, predicted, chain_map, full_chains, ("CA",)
                )
                full_reason = _coverage_reason(
                    "full complex CA", full_ref, full_pred, full_matched, full_expected
                )
                if set(chain_map.values()) != set(_ordered_chains(predicted)):
                    full_reason = "full complex CA chain mapping is incomplete"
                if full_reason is None:
                    try:
                        metrics["complex_design_rmsd"] = _rmsd_with_fit(full_ref, full_pred)
                    except MetricUnavailable as exc:
                        _missing(metrics, reasons, "complex_design_rmsd", str(exc))
                else:
                    _missing(metrics, reasons, "complex_design_rmsd", full_reason)
            else:
                _missing(
                    metrics,
                    reasons,
                    "complex_design_rmsd",
                    "full complex CA metric is only applicable to all-protein complexes",
                )
            ligand_values, ligand_reasons = _protein_aligned_ligand_metrics(
                reference, predicted, chain_map, ligand_chain_map, ligand_atom_maps,
                ligand_mapping_reasons,
                strict_ligand_chains=tuple(chain for chain in design_chains
                                          if _is_ligand_chain(reference, chain)),
                binder_chains=design_protein_chains,
                metric_provenance=metric_provenance,
            )
            metrics.update(ligand_values)
            reasons.update(ligand_reasons)
        elif design_is_ligand:
            for name in (
                "design_rmsd", "binder_design_rmsd", "context_aligned_design_rmsd",
                "complex_design_rmsd", "binder_scRMSD_bb3",
            ):
                _missing(metrics, reasons, name, "protein backbone metric is not applicable to ligand design")
            ligand_values: list[float] = []
            ligand_points: list[tuple[np.ndarray, np.ndarray]] = []
            for chain in design_chains:
                pred_chain = ligand_chain_map.get(chain)
                atom_map = ligand_atom_maps.get((chain, pred_chain)) if pred_chain else None
                if pred_chain is None or atom_map is None:
                    reason = ligand_mapping_reasons.get(
                        chain, "no unique ligand element/bond-order graph mapping"
                    )
                    _missing(metrics, reasons, "ligand_rmsd", f"{chain}: {reason}")
                    _missing(metrics, reasons, "context_aligned_ligand_rmsd", f"{chain}: {reason}")
                    continue
                ligand_ref, ligand_pred, ligand_matched, ligand_expected = _mapped_ligand_points(
                    reference, predicted, chain, pred_chain, atom_map
                )
                ligand_reason = _coverage_reason(
                    f"ligand {chain}", ligand_ref, ligand_pred, ligand_matched, ligand_expected
                )
                if ligand_reason is not None:
                    _missing(metrics, reasons, "ligand_rmsd", ligand_reason)
                    _missing(metrics, reasons, "context_aligned_ligand_rmsd", ligand_reason)
                    continue
                try:
                    ligand_values.append(_rmsd_with_fit(ligand_ref, ligand_pred))
                    ligand_points.append((ligand_ref, ligand_pred))
                except MetricUnavailable as exc:
                    _missing(metrics, reasons, "ligand_rmsd", str(exc))
                    _missing(metrics, reasons, "context_aligned_ligand_rmsd", str(exc))
            if ligand_values:
                metrics["ligand_rmsd"] = float(np.mean(ligand_values))
            context_is_protein = bool(context_chains) and all(
                _chain_mol_type(reference, chain) == "protein" for chain in context_chains
            )
            if ligand_points and context_is_protein:
                context_ref, context_pred, context_matched, context_expected = _paired_points(
                    reference, predicted, chain_map, context_chains, ("CA",)
                )
                context_reason = _coverage_reason(
                    "context", context_ref, context_pred, context_matched, context_expected
                )
                if context_reason is None:
                    values = []
                    for ligand_ref, ligand_pred in ligand_points:
                        try:
                            values.append(
                                _fit_then_measure(
                                    context_ref, context_pred, ligand_ref, ligand_pred
                                )
                            )
                        except MetricUnavailable as exc:
                            context_reason = str(exc)
                            break
                    if values and context_reason is None:
                        metrics["context_aligned_ligand_rmsd"] = float(np.mean(values))
                    elif context_reason is not None:
                        _missing(metrics, reasons, "context_aligned_ligand_rmsd", context_reason)
                else:
                    _missing(metrics, reasons, "context_aligned_ligand_rmsd", context_reason)
            elif "context_aligned_ligand_rmsd" not in metrics:
                _missing(
                    metrics,
                    reasons,
                    "context_aligned_ligand_rmsd",
                    "ligand context fit requires complete protein context chains",
                )
        else:
            for name in (
                "design_rmsd", "binder_design_rmsd", "context_aligned_design_rmsd",
                "complex_design_rmsd", "binder_scRMSD_bb3", "ligand_rmsd",
                "context_aligned_ligand_rmsd",
            ):
                _missing(metrics, reasons, name, "mixed or unknown design modality")

    if metrics["complexa_aligned_ligand_rmsd"] is None:
        reasons.setdefault("complexa_aligned_ligand_rmsd", "requires mapped designed protein binder and ligand atoms")
    _na_design_metrics(metrics, reasons, reference, predicted, chain_map, design_chains)

    motif_reference_requested = _motif_reference_requested(input_data, prediction, variant)
    motif_reference_structure = _motif_reference_prediction(input_data, prediction, variant)
    reference_provenance = {}
    reference_error = None
    if not motif_reference_requested and input_data.motif_positions:
        from .references import ReferenceUnavailable, recover_reference
        try:
            motif_reference_structure, reference_provenance = recover_reference(input_data)
            motif_reference_requested = True
        except (ReferenceUnavailable, OSError, ValueError, TypeError, KeyError) as exc:
            reference_error = f"source motif recovery unavailable: {type(exc).__name__}: {exc}"
            motif_reference_requested = (input_data.task_dir / "resolved_config.yaml").is_file()
    _confidence_metrics(
        metrics, reasons, prediction, reference, predicted, chain_map, design_chains, input_data,
        metric_provenance,
    )
    _sequence_metrics(
        metrics,
        reasons,
        input_data,
        predicted,
        chain_map,
        variant,
        motif_reference=motif_reference_structure,
        motif_reference_requested=motif_reference_requested,
    )

    motif_ref, motif_pred, _matched, missing = _motif_pairs(input_data, predicted, chain_map, reference=reference)
    if len(motif_ref) and not missing:
        try:
            metrics["motif_rmsd_design"] = _motif_rmsd_with_fit(motif_ref, motif_pred)
        except MetricUnavailable as exc:
            _missing(metrics, reasons, "motif_rmsd_design", str(exc))
    elif input_data.motif_positions:
        _missing(metrics, reasons, "motif_rmsd_design", "no complete motif atom correspondence")
    else:
        _missing(metrics, reasons, "motif_rmsd_design", "no contig motif was declared")
    if missing:
        reasons.setdefault("motif_rmsd_design", f"motif atoms unavailable: {missing[:3]}")

    if motif_reference_structure is None:
        reason = reference_error or (
            "explicit source/reference motif path could not be loaded"
            if motif_reference_requested
            else "source/reference motif coordinates were not supplied"
        )
        _missing(metrics, reasons, "motif_rmsd_reference", reason)
    else:
        reference_chain_map = prediction.get("motif_reference_chain_map")
        if not isinstance(reference_chain_map, Mapping) and variant:
            reference_chain_map = variant.get("motif_reference_chain_map")
        if not isinstance(reference_chain_map, Mapping):
            reference_chain_map = None
        ref_ref, ref_pred, _matched_ref, missing_ref = _motif_pairs(
            input_data,
            predicted,
            chain_map,
            reference=motif_reference_structure,
            reference_chain_map=reference_chain_map,
        )
        if len(ref_ref) and not missing_ref:
            try:
                metrics["motif_rmsd_reference"] = _motif_rmsd_with_fit(ref_ref, ref_pred)
            except MetricUnavailable as exc:
                _missing(metrics, reasons, "motif_rmsd_reference", str(exc))
        else:
            _missing(metrics, reasons, "motif_rmsd_reference", "no complete source/reference motif correspondence")
        if missing_ref:
            reasons.setdefault("motif_rmsd_reference", f"reference motif atoms unavailable: {missing_ref[:3]}")

    from .benchmark_metrics import extra_geometry
    extra, extra_reasons = extra_geometry(input_data, predicted, chain_map,
        ligand_chain_map, ligand_atom_maps, motif_reference_structure)
    metrics.update(extra)
    reasons.update(extra_reasons)

    clash, clash_reason = _ligand_clashes(predicted)
    if clash is None:
        _missing(
            metrics,
            reasons,
            "has_ligand_clashes",
            clash_reason or "protein-ligand clash status is unavailable",
        )
    else:
        metrics["has_ligand_clashes"] = clash
    return {"metrics": metrics, "reasons": reasons,
            "provenance": {"motif_reference": reference_provenance, **metric_provenance}}


def _resolve_sequence_chain(
    atoms: Any,
    preferred: str,
    expected_sequence: str | None,
) -> str | None:
    """Resolve one prediction chain by canonical label plus full sequence."""

    chains = _ordered_chains(atoms)
    expected = str(expected_sequence or "").upper()
    if preferred in chains and (not expected or _sequence_matches(atoms, preferred, expected)):
        return preferred
    if expected:
        candidates = [
            chain for chain in chains if _sequence_matches(atoms, chain, expected)
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None
    return preferred if preferred in chains else None


def paired_bound_unbound_rmsd(
    complex_prediction: Any,
    monomer_prediction: Any,
    *,
    chain: str,
    expected_sequence: str | None = None,
    ca_only: bool = False,
) -> tuple[float | None, str | None]:
    """CA-align the monomer, then measure all-heavy RMSD to its complex binder."""

    complex_chain = _resolve_sequence_chain(
        complex_prediction, chain, expected_sequence
    )
    if complex_chain is None:
        return None, "bound binder chain is absent or sequence-ambiguous"
    expected = str(expected_sequence or "").upper() or _chain_sequence(complex_prediction, complex_chain)
    monomer_chain = _resolve_sequence_chain(
        monomer_prediction, chain, expected
    )
    if monomer_chain is None:
        return None, "monomer binder chain is absent or sequence-ambiguous"
    if _chain_sequence(complex_prediction, complex_chain) != expected:
        return None, "bound binder sequence does not match requested variant"
    if _chain_sequence(monomer_prediction, monomer_chain) != expected:
        return None, "unbound binder sequence does not match requested variant"
    # BindCraft first aligns its monomer using AlignChainMover (C-alpha),
    # then measures all-heavy RMSD without another fit. Here the paired
    # complex binder is the reference instead of a hallucination trajectory.
    ca_reference, ca_predicted, ca_matched, ca_expected = _paired_points(
        complex_prediction, monomer_prediction,
        {complex_chain: monomer_chain}, (complex_chain,), ("CA",),
    )
    ca_reason = _coverage_reason(
        "bound/unbound binder CA alignment", ca_reference, ca_predicted, ca_matched, ca_expected
    )
    if ca_reason is not None:
        return None, ca_reason
    if ca_only:
        try:
            return _rmsd_with_fit(ca_reference, ca_predicted), None
        except MetricUnavailable as exc:
            return None, str(exc)
    reference, predicted, matched, expected_atoms = _paired_points(
        complex_prediction,
        monomer_prediction,
        {complex_chain: monomer_chain},
        (complex_chain,),
        None,
        all_heavy=True,
    )
    reason = _coverage_reason(
        "bound/unbound binder all-heavy", reference, predicted, matched, expected_atoms
    )
    if reason is not None:
        return None, reason
    try:
        return _fit_then_measure(ca_reference, ca_predicted, reference, predicted), None
    except MetricUnavailable as exc:
        return None, str(exc)


def _prediction_atoms(value: Any) -> Any:
    if hasattr(value, "get_annotation_categories") and hasattr(value, "coord"):
        return value
    if isinstance(value, Mapping):
        path = value.get("structure_path")
        if path:
            return read_structure(Path(path))
    path = getattr(value, "structure_path", None)
    if path:
        return read_structure(Path(path))
    raise MetricUnavailable("prediction has no readable structure")


def evaluate_pair(
    input_data: ValidationInput,
    variant: Mapping[str, Any] | None,
    complex_prediction: Mapping[str, Any] | Any,
    monomer_prediction: Mapping[str, Any] | Any,
) -> dict[str, dict[str, Any]]:
    """Evaluate the AF2 bound/unbound pair for one sequence/model/sample.

    The monomer binder is first aligned to the complex binder using CA
    atoms. All matched heavy atoms are then measured without a second fit.
    This uses BindCraft's atom-selection/alignment convention with the
    paired AF2 complex as reference, not its hallucination trajectory.
    """

    metrics: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    for label, record in (("complex", complex_prediction), ("monomer", monomer_prediction)):
        if isinstance(record, Mapping) and str(record.get("status", "ok")).lower() not in {"ok", "success", "complete", "completed"}:
            reason = str(record.get("error") or f"{label} prediction did not complete")
            _missing(metrics, reasons, "bound_unbound_rmsd", reason)
            return {"metrics": metrics, "reasons": reasons}
    try:
        complex_atoms = _prediction_atoms(complex_prediction)
        monomer_atoms = _prediction_atoms(monomer_prediction)
    except (OSError, MetricUnavailable, ValueError) as exc:
        _missing(metrics, reasons, "bound_unbound_rmsd", f"cannot read AF2 pair: {exc}")
        return {"metrics": metrics, "reasons": reasons}
    if not input_data.design_chains:
        _missing(metrics, reasons, "bound_unbound_rmsd", "no design chain is available for AF2 pairing")
        return {"metrics": metrics, "reasons": reasons}
    variant_sequences = _variant_sequences(variant)
    values: list[float] = []
    for chain in input_data.design_chains:
        value, reason = paired_bound_unbound_rmsd(
            complex_atoms,
            monomer_atoms,
            chain=chain,
            expected_sequence=variant_sequences.get(chain) or _chain_sequence(input_data.atoms, chain),
        )
        if value is not None:
            values.append(value)
        elif reason:
            reasons.setdefault("bound_unbound_rmsd", f"{chain}: {reason}")
    if values:
        metrics["bound_unbound_rmsd"] = float(np.mean(values))
    else:
        _missing(metrics, reasons, "bound_unbound_rmsd", reasons.get("bound_unbound_rmsd", "no paired binder backbone atoms"))
    ca_values = []
    for chain in input_data.design_chains:
        value, reason = paired_bound_unbound_rmsd(
            complex_atoms, monomer_atoms, chain=chain,
            expected_sequence=variant_sequences.get(chain) or _chain_sequence(input_data.atoms, chain),
            ca_only=True)
        if value is None:
            ca_values = []
            _missing(metrics, reasons, "bound_unbound_ca_rmsd", reason or "paired CA unavailable")
            break
        ca_values.append(value)
    if ca_values:
        metrics["bound_unbound_ca_rmsd"] = float(np.mean(ca_values))
    return {"metrics": metrics, "reasons": reasons}
