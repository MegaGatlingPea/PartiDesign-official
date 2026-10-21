"""Streaming AlphaFold2-pTM Initial-Guess validation backend.

This module is deliberately a small validation-facing adapter around the
vendored ColabDesign implementation.  It does not share the generation AF2
reward path: validation always runs the PipeTools-style binder pTM initial
guess protocol and, when requested, the independent pTM hallucination
protocol for the same design sequence.

The public backend yields one immutable result per model/protocol/sample.  A
caller can therefore persist ``seqN.json`` immediately and resume by passing
the completed prediction keys back to :meth:`AF2Backend.predict`.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from partidesign.paths import (community_source, community_asset, asset_root, cache_dir, offline_site, code_root)
from typing import Any

import numpy as np

from src.design.utils.reader import read_structure
from src.design.utils.writer import write_structure_atomic
from src.design.rewards.af2_mapping import (
    af2_output_suffix,
    af2_source_ledger,
    context_unknowns,
    encode_private_af2_indices,
    normalize_private_override,
    private_chain_id,
    preserve_native_pdb_confidence,
    restore_af2_residue_identity,
    validate_prepared_af2,
    write_private_af2_pdb,
)


class AF2Error(RuntimeError):
    """Base class for validation AF2 errors."""


class AF2ConfigurationError(AF2Error, ValueError):
    """The AF2 deployment or request configuration is invalid."""


class AF2CandidateError(ValueError):
    """One validation input cannot be represented by the AF2 adapter."""


class AF2UnsupportedChemistryError(AF2CandidateError):
    """Input chemistry is outside AF2-IG's protein-template capability."""

    code = "unsupported_chemistry"

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class AF2SystemicError(AF2Error):
    """Model loading or accelerator failure that invalidates the backend."""


_AA3_TO_1 = {
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
}
_AA1 = frozenset(_AA3_TO_1.values())
_SYSTEMIC_MARKERS = (
    "jax",
    "xla",
    "cuda",
    "cublas",
    "cudnn",
    "out of memory",
    "illegal memory",
    "device-side assert",
    "checkpoint",
    "colabdesign import",
)
_SEED_MODULUS = 2_147_483_647
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def signed_circular_offset(length: int) -> np.ndarray:
    """Return the AF2 cyclic relative-position matrix for one chain.

    A wrap is used only when it is strictly shorter than the linear path.
    For an even-length diametric pair the linear distance is retained.  The
    sign convention matches ColabDesign/PXDesignBench's ``offset`` input.
    """

    if isinstance(length, bool) or not isinstance(length, int) or length < 2:
        raise AF2CandidateError("cyclic AF2 offset length must be an integer >= 2")
    positions = np.arange(length, dtype=np.int32)
    linear = positions[:, None] - positions[None, :]
    absolute = np.abs(linear)
    wrapped = length - absolute
    magnitude = np.where(wrapped < absolute, -wrapped, absolute)
    return (magnitude * np.sign(linear)).astype(np.int32, copy=False)


def apply_cyclic_offset(
    model: Any,
    *,
    design_start: int,
    design_length: int,
    enabled: bool,
) -> None:
    """Patch only the design-chain block of a prepared ColabDesign model.

    The optional key is removed for every linear request, which prevents a
    cyclic request from contaminating a later request that reuses the model.
    """

    inputs = getattr(model, "_inputs", None)
    if not isinstance(inputs, Mapping):
        raise AF2CandidateError("ColabDesign model has no prepared _inputs mapping")
    # ColabDesign owns a mutable dict here.  Mapping is used in the contract
    # check above so test doubles and mapping subclasses get the same error.
    if hasattr(inputs, "pop"):
        inputs.pop("offset", None)
    if not enabled:
        return
    residues = np.asarray(inputs.get("residue_index"), dtype=np.int32)
    if residues.ndim != 1:
        raise AF2CandidateError("AF2 residue_index must be one-dimensional")
    if (
        isinstance(design_start, bool)
        or isinstance(design_length, bool)
        or design_start < 0
        or design_length < 2
        or design_start + design_length > len(residues)
    ):
        raise AF2CandidateError("AF2 cyclic design slice is outside residue_index")
    offset = residues[:, None] - residues[None, :]
    offset = np.asarray(offset, dtype=np.int32)
    offset[
        design_start : design_start + design_length,
        design_start : design_start + design_length,
    ] = signed_circular_offset(design_length)
    inputs["offset"] = offset


def _clear_request_state(model: Any) -> None:
    """Clear request tensors while retaining compiled model parameters."""

    if model is None:
        return
    for name in ("_inputs", "aux", "_tmp"):
        value = getattr(model, name, None)
        if isinstance(value, dict):
            value.clear()


def _stable_seed(
    base_seed: int,
    request_id: str,
    variant_id: str,
    protocol: str,
    model: int,
    sample: int,
) -> int:
    sequence_number = int(re.findall(r"\d+", variant_id)[-1]) if re.search(r"\d", variant_id) else 0
    sample_number = int(re.findall(r"\d+", request_id)[-1]) if re.search(r"\d", request_id) else 0
    offset = sample_number * 10000019 + sequence_number * 100003 + model * 1009 + sample * 2 + (protocol == "monomer")
    return int((base_seed + offset) % _SEED_MODULUS)


def _slug(value: Any, fallback: str) -> str:
    text = str(value) if value is not None else fallback
    text = _SAFE_NAME.sub("_", text).strip("._")
    return text or fallback


def _mapping_value(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _first_mapping_value(mapping: Any, *keys: str) -> Any:
    """Return the first present value without truth-testing NumPy arrays."""

    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _scalar(value: Any, name: str, *, unit_interval: bool = False) -> float:
    if value is None:
        raise AF2CandidateError(f"AF2 confidence metric {name} is missing")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AF2CandidateError(f"AF2 confidence metric {name} is malformed") from error
    if array.size == 0 or not np.isfinite(array).all():
        raise AF2CandidateError(f"AF2 confidence metric {name} is non-finite")
    if unit_interval and (np.any(array < 0.0) or np.any(array > 1.0)):
        raise AF2CandidateError(f"AF2 confidence metric {name} is outside [0,1]")
    if not unit_interval and np.any(array < 0.0):
        raise AF2CandidateError(f"AF2 confidence metric {name} is negative")
    return float(array.reshape(-1).mean())


def extract_confidence(
    model: Any,
    protocol: str,
    *,
    design_start: int,
    design_length: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Extract native AF2 confidence and the full PAE matrix.

    ``plddt`` is reduced over the design-chain residues only.  ColabDesign
    stores the per-residue vector in ``aux["plddt"]``; ``aux["log"]["plddt"]``
    is a scalar loss and must not be sliced as if it were that vector.  The
    scalar normalized PAE/iPAE values come from ColabDesign's native ``log``
    payload; the full matrix is returned separately for raw evidence and
    downstream interface/motif metrics.
    """

    aux = getattr(model, "aux", None)
    if not isinstance(aux, Mapping):
        raise AF2CandidateError("ColabDesign model confidence payload is missing")
    log = aux.get("log") if isinstance(aux.get("log"), Mapping) else aux
    # ``log.plddt`` is the loss (1 - mean pLDDT) in the vendored ColabDesign
    # implementation.  Prefer the residue vector from ``aux.plddt`` and use
    # ``aux.all.plddt`` only as a shape-compatible multi-model fallback.
    plddt_source = _first_mapping_value(aux, "plddt")
    if plddt_source is None:
        all_aux = aux.get("all") if isinstance(aux.get("all"), Mapping) else None
        plddt_source = _first_mapping_value(all_aux, "plddt")
        if plddt_source is not None:
            try:
                all_plddt = np.asarray(plddt_source, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise AF2CandidateError("AF2 pLDDT payload is malformed") from error
            if all_plddt.ndim < 2:
                raise AF2CandidateError(
                    "AF2 per-residue pLDDT vector is missing; log.plddt is scalar"
                )
            # ``all`` is normally [model, residue] (or a compatible leading
            # model/recycle stack). Average only leading axes and retain the
            # residue axis for the design slice below.
            plddt_source = all_plddt.mean(axis=tuple(range(all_plddt.ndim - 1)))
    ptm_source = _first_mapping_value(log, "ptm")
    if ptm_source is None:
        ptm_source = _first_mapping_value(aux, "ptm")
    pae_matrix_source = _mapping_value(aux, "pae")
    pae_log_source = _mapping_value(log, "pae")
    if pae_matrix_source is None:
        raise AF2CandidateError("AF2 full PAE matrix is missing")
    if pae_log_source is None:
        raise AF2CandidateError("AF2 normalized PAE summary is missing")
    try:
        pae = np.asarray(pae_matrix_source, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AF2CandidateError("AF2 PAE matrix is malformed") from error
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise AF2CandidateError("AF2 PAE matrix must be square")
    if not np.isfinite(pae).all() or np.any(pae < 0):
        raise AF2CandidateError("AF2 PAE matrix must be finite and non-negative")
    try:
        raw_plddt = np.asarray(plddt_source, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AF2CandidateError("AF2 pLDDT payload is malformed") from error
    if raw_plddt.ndim == 0:
        raise AF2CandidateError(
            "AF2 per-residue pLDDT vector is missing; log.plddt is scalar"
        )
    if raw_plddt.ndim > 1:
        # A direct aux payload should already be [residue]. Accept leading
        # model/recycle axes defensively, while preserving the final residue
        # axis for the requested design slice.
        raw_plddt = raw_plddt.mean(axis=tuple(range(raw_plddt.ndim - 1)))
    plddt = raw_plddt.reshape(-1)
    if design_start < 0 or design_length < 1 or design_start + design_length > len(plddt):
        raise AF2CandidateError("AF2 design slice is outside pLDDT payload")
    design_values = plddt[design_start : design_start + design_length]
    if not np.isfinite(design_values).all() or np.any(design_values < 0) or np.any(design_values > 1):
        raise AF2CandidateError("AF2 design pLDDT must be finite in [0,1]")
    plddt_normalized = float(design_values.mean())
    confidence: dict[str, Any] = {
        "plddt": plddt_normalized * 100.0,
        "plddt_normalized": plddt_normalized,
        "ptm": _scalar(ptm_source, "ptm", unit_interval=True),
        "pae": _scalar(pae_log_source, "pae", unit_interval=True),
        "pae_angstrom": _scalar(pae_log_source, "pae", unit_interval=True) * 31.0,
        "pae_matrix_mean_angstrom": float(pae.mean()),
        "pae_shape": [int(pae.shape[0]), int(pae.shape[1])],
        "plddt_scale": "0-1 native; plddt is reported on 0-100 scale",
        "pae_scale": "normalized native; *_angstrom uses factor 31",
        "protocol": protocol,
    }

    # AlphaProteo's total pLDDT covers every complex residue. Keep the
    # existing binder-only pLDDT untouched for the other AF2 criteria.
    if len(plddt) == pae.shape[0] and np.isfinite(plddt).all() and np.all((plddt >= 0) & (plddt <= 1)):
        confidence["plddt_total"] = float(plddt.mean())
    else:
        confidence["plddt_total_unavailable_reason"] = "AF2 total pLDDT must match PAE residues and be finite in [0,1]"
    if protocol == "complex":
        # The AF2ig implementation cited by AlphaProteo averages both
        # directions. Native ColabDesign iPAE remains a separate scalar.
        stop = design_start + design_length
        if pae.shape[0] == len(plddt):
            binder_indices = np.arange(design_start, stop)
            context_indices = np.concatenate((np.arange(design_start), np.arange(stop, len(plddt))))
            if len(context_indices):
                confidence["pae_interaction"] = float((
                    pae[np.ix_(binder_indices, context_indices)].mean()
                    + pae[np.ix_(context_indices, binder_indices)].mean()
                ) / 2.0)
            else:
                confidence["pae_interaction_unavailable_reason"] = "complex prediction has no context residues"
        else:
            confidence["pae_interaction_unavailable_reason"] = "PAE dimensions do not match pLDDT residue order"
        confidence["iptm"] = _scalar(
            _first_mapping_value(log, "i_ptm")
            if _first_mapping_value(log, "i_ptm") is not None
            else _first_mapping_value(aux, "i_ptm"),
            "iptm",
            unit_interval=True,
        )
        ipae = _scalar(
            _first_mapping_value(log, "i_pae")
            if _first_mapping_value(log, "i_pae") is not None
            else _first_mapping_value(aux, "i_pae"),
            "ipae",
            unit_interval=True,
        )
        confidence["ipae"] = ipae
        confidence["ipae_angstrom"] = ipae * 31.0
    else:
        confidence["iptm_unavailable_reason"] = "monomer prediction has no interface metric"
        confidence["ipae_unavailable_reason"] = "monomer prediction has no interface metric"
        confidence["ipae_angstrom_unavailable_reason"] = (
            "monomer prediction has no interface metric"
        )
    return confidence, pae


def _is_systemic(error: Exception) -> bool:
    if isinstance(error, (AF2SystemicError, ImportError, ModuleNotFoundError, MemoryError)):
        return True
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in _SYSTEMIC_MARKERS)


def expected_keys(config: Mapping[str, Any]) -> list[str]:
    """Return deterministic prediction keys without touching model assets."""

    if not isinstance(config, Mapping):
        raise AF2ConfigurationError("AF2 config must be a mapping")
    raw_mode = config.get("mode", "both")
    mode = str(raw_mode)
    if mode not in {"complex", "monomer", "both"}:
        raise AF2ConfigurationError("AF2 mode must be complex, monomer, or both")
    raw_models = config.get("models", [1])
    models = AF2Backend._models_numbers(raw_models)
    samples = AF2Backend._positive_int("samples", config.get("samples", 1))
    protocols = ("complex", "monomer") if mode == "both" else (mode,)
    return [
        f"{protocol}_model{model}_sample{sample}"
        for protocol in protocols
        for model in models
        for sample in range(samples)
    ]


def _read_chain_ids(atoms: Any) -> list[str]:
    values = getattr(atoms, "chain_id", None)
    if values is None:
        return []
    result: list[str] = []
    for value in np.asarray(values).astype(str):
        if value not in result:
            result.append(value)
    return result


def _attr_or_mapping(value: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(value, Mapping) and key in value:
            return value[key]
        try:
            return getattr(value, key)
        except AttributeError:
            pass
    return None


def _design_chain(input_value: Any) -> str:
    raw = _attr_or_mapping(input_value, "design_chains", "design_chain")
    if isinstance(raw, str):
        chains = [raw]
    elif isinstance(raw, Mapping):
        chains = [str(key) for key in raw]
    elif isinstance(raw, Sequence):
        chains = [str(item) for item in raw]
    else:
        chains = []
    if len(chains) != 1 or not chains[0]:
        raise AF2CandidateError("AF2 validation requires exactly one design protein chain")
    return chains[0]


def _context_chains(input_value: Any, design_chain: str) -> list[str]:
    raw = _attr_or_mapping(input_value, "context_chains", "context_chain_ids")
    if isinstance(raw, str):
        result = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        result = [str(item) for item in raw]
    else:
        result = []
    if not result:
        atoms = _attr_or_mapping(input_value, "atoms")
        result = [chain for chain in _read_chain_ids(atoms) if chain != design_chain]
    result = [chain for chain in result if chain != design_chain]
    if len(result) != len(set(result)):
        raise AF2CandidateError("AF2 context chain list contains duplicates")
    return result


def _jsonable(value: Any) -> Any:
    """Convert NumPy/Path values into deterministic JSON ledger values."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _array_field(atoms: Any, name: str) -> np.ndarray | None:
    value = getattr(atoms, name, None)
    return None if value is None else np.asarray(value)


def _chain_values(atoms: Any) -> np.ndarray:
    labels = _array_field(atoms, "label_asym_id")
    if labels is not None and any(str(value) for value in labels):
        return labels.astype(str)
    chains = _array_field(atoms, "chain_id")
    return np.asarray([], dtype=str) if chains is None else chains.astype(str)


def _atom_reference(atoms: Any, index: int, *, _chains: np.ndarray | None = None) -> dict[str, Any]:
    """Return a stable row/locator record for one source AtomArray atom."""

    chains = _chain_values(atoms) if _chains is None else _chains
    atom_names = _array_field(atoms, "atom_name")
    res_names = _array_field(atoms, "res_name")
    res_ids = _array_field(atoms, "res_id")
    residue_uids = _array_field(atoms, "residue_uid")
    ins_codes = _array_field(atoms, "ins_code")
    atom_uids = _array_field(atoms, "atom_uid")
    auth_chains = _array_field(atoms, "auth_asym_id")

    def at(values: np.ndarray | None, default: Any = None) -> Any:
        if values is None or index >= len(values):
            return default
        return _jsonable(values[index])

    return {
        "index": int(index),
        "atom_uid": at(atom_uids, int(index)),
        "chain": at(chains, ""),
        "author_chain": at(auth_chains, ""),
        "res_id": at(res_ids),
        "residue_uid": at(residue_uids),
        "ins_code": at(ins_codes, ""),
        "res_name": at(res_names, ""),
        "atom_name": at(atom_names, ""),
    }


def _normal_bond_order(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, str):
        named = {"single": 1, "sing": 1, "double": 2, "doub": 2,
                 "triple": 3, "trip": 3}
        if value.strip().lower() in named:
            return named[value.strip().lower()]
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AF2UnsupportedChemistryError(
            "AF2 bond order is not a supported numeric or named order",
            details={"reason": "unsupported_bond_order", "order": _jsonable(value)},
        ) from exc
    if number.is_integer():
        return int(number)
    return number


def _graph_edges(atoms: Any) -> tuple[list[tuple[int, int, int | float | None, dict[str, Any]]], dict[tuple[int, int], tuple[int | float | None, dict[str, Any]]]]:
    """Read every source bond, retaining both row indices and JSON records."""

    bonds = getattr(atoms, "bonds", None)
    if bonds is None:
        return [], {}
    try:
        array = np.asarray(bonds.as_array())
    except Exception as error:
        raise AF2UnsupportedChemistryError(
            "AF2 could not read the source bond graph",
            details={"reason": "bond_graph_unreadable", "error": str(error)},
        ) from error
    if array.ndim != 2 or array.shape[1] < 2:
        raise AF2UnsupportedChemistryError(
            "AF2 source bond graph has an invalid shape",
            details={"reason": "bond_graph_malformed", "shape": list(array.shape)},
        )
    edges: list[tuple[int, int, int | float | None, dict[str, Any]]] = []
    lookup: dict[tuple[int, int], tuple[int | float | None, dict[str, Any]]] = {}
    chains = _chain_values(atoms)
    atom_count = len(chains)
    for edge_index, raw in enumerate(array.tolist()):
        try:
            left, right = int(raw[0]), int(raw[1])
        except (TypeError, ValueError) as error:
            raise AF2UnsupportedChemistryError(
                "AF2 source bond graph contains a non-integer atom index",
                details={"reason": "bond_graph_malformed", "edge_index": edge_index},
            ) from error
        if left < 0 or right < 0 or left >= atom_count or right >= atom_count or left == right:
            raise AF2UnsupportedChemistryError(
                "AF2 source bond graph contains an invalid atom index",
                details={"reason": "bond_graph_malformed", "edge_index": edge_index, "atoms": [left, right]},
            )
        order = _normal_bond_order(raw[2] if len(raw) > 2 else None)
        record = {
            "edge_index": int(edge_index),
            "atom1": _atom_reference(atoms, left, _chains=chains),
            "atom2": _atom_reference(atoms, right, _chains=chains),
            "order": order,
        }
        key = tuple(sorted((left, right)))
        if key in lookup:
            raise AF2UnsupportedChemistryError(
                "AF2 source bond graph contains duplicate edges",
                details={"reason": "bond_graph_duplicate", "atoms": list(key)},
            )
        lookup[key] = (order, record)
        edges.append((left, right, order, record))
    return edges, lookup


def _residue_key(atoms: Any, index: int, *, _chains: np.ndarray | None = None) -> tuple[str, int | str, str]:
    residue_uids = _array_field(atoms, "residue_uid")
    chains = _chain_values(atoms) if _chains is None else _chains
    ins_codes = _array_field(atoms, "ins_code")
    if residue_uids is not None:
        return (str(chains[index]), int(residue_uids[index]), "")
    res_ids = _array_field(atoms, "res_id")
    if res_ids is None:
        raise AF2UnsupportedChemistryError(
            "AF2 cannot validate source chemistry without residue identity",
            details={"reason": "residue_identity_missing"},
        )
    ins = "" if ins_codes is None else str(ins_codes[index])
    return (str(chains[index]), int(res_ids[index]), ins)


def _residue_positions(atoms: Any) -> tuple[dict[str, list[tuple[str, int | str, str]]], dict[tuple[str, tuple[str, int | str, str]], int]]:
    chains = _chain_values(atoms)
    result: dict[str, list[tuple[str, int | str, str]]] = {}
    positions: dict[tuple[str, tuple[str, int | str, str]], int] = {}
    for index, chain_value in enumerate(chains.tolist()):
        chain = str(chain_value)
        key = _residue_key(atoms, index, _chains=chains)
        values = result.setdefault(chain, [])
        if (chain, key) not in positions:
            positions[(chain, key)] = len(values)
            values.append(key)
    return result, positions


def _single_bond(order: int | float | None) -> bool:
    return order is not None and abs(float(order) - 1.0) < 1e-6


def _resolve_locator_rows(atoms: Any, locator: Mapping[str, Any]) -> list[int]:
    """Resolve a normalized row ledger atom locator against the source graph."""

    chains = _chain_values(atoms)
    if len(chains) == 0:
        return []
    mask = np.ones(len(chains), dtype=bool)
    chain = _mapping_value(locator, "chain", "label_asym_id", "asym_id", "chain_id")
    if chain not in (None, "", ".", "?"):
        mask &= chains == str(chain)
    res_ids = _array_field(atoms, "res_id")
    res_id = _mapping_value(locator, "res_id", "auth_seq_id", "label_seq_id")
    if res_id not in (None, "", ".", "?") and res_ids is not None:
        try:
            mask &= res_ids.astype(np.int64) == int(res_id)
        except (TypeError, ValueError):
            return []
    ins_codes = _array_field(atoms, "ins_code")
    ins_code = _mapping_value(locator, "ins_code", "pdb_ins_code")
    if ins_code not in (None, "", ".", "?") and ins_codes is not None:
        mask &= ins_codes.astype(str) == str(ins_code)
    atom_name = _mapping_value(locator, "atom", "atom_name", "label_atom_id")
    atom_names = _array_field(atoms, "atom_name")
    if atom_name not in (None, "", ".", "?") and atom_names is not None:
        mask &= atom_names.astype(str) == str(atom_name)
    atom_index = _mapping_value(locator, "atom_index")
    atom_id = _mapping_value(locator, "atom_id", "atom_uid")
    if atom_index not in (None, "", ".", "?"):
        try:
            row = int(atom_index)
        except (TypeError, ValueError):
            return []
        mask &= np.arange(len(chains), dtype=np.int64) == row
    elif atom_id not in (None, "", ".", "?"):
        try:
            # The public ledger uses one-based source row/atom IDs.
            row = int(atom_id) - 1
        except (TypeError, ValueError):
            return []
        mask &= np.arange(len(chains), dtype=np.int64) == row
    return np.flatnonzero(mask).astype(np.int64).tolist()


def _explicit_offset_chains(input_value: Any) -> list[str]:
    """Normalize explicit AF2 offset intent without inferring graph edges."""

    raw = _attr_or_mapping(input_value, "cyclic_offset_chains")
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        values = list(raw)
    else:
        raise AF2UnsupportedChemistryError(
            "AF2 cyclic offset intent must be a chain-id sequence",
            details={"reason": "cyclic_offset_metadata_malformed"},
        )
    if any(not isinstance(value, str) or not value for value in values):
        raise AF2UnsupportedChemistryError(
            "AF2 cyclic offset intent contains an invalid chain id",
            details={"reason": "cyclic_offset_metadata_malformed"},
        )
    if len(values) != len(set(values)):
        raise AF2UnsupportedChemistryError(
            "AF2 cyclic offset intent contains duplicate chain ids",
            details={"reason": "cyclic_offset_metadata_ambiguous", "chains": values},
        )
    return values


def _audit_chemistry(input_value: Any, *, design_chain: str, context: Sequence[str]) -> dict[str, Any]:
    """Validate protein bond semantics and identify cyclic design chains."""

    atoms = _attr_or_mapping(input_value, "atoms")
    cycles: list[str] = []
    edge_lookup: dict[tuple[int, int], tuple[int | float | None, dict[str, Any]]] = {}
    if atoms is not None:
        edges, edge_lookup = _graph_edges(atoms)
        residue_order, positions = _residue_positions(atoms) if edges else ({}, {})
        atom_names = _array_field(atoms, "atom_name")
        chains = _chain_values(atoms)
        for left, right, order, raw_record in edges:
            left_chain, right_chain = str(chains[left]), str(chains[right])
            left_key, right_key = _residue_key(atoms, left, _chains=chains), _residue_key(atoms, right, _chains=chains)
            if left_chain != right_chain:
                raise AF2UnsupportedChemistryError(
                    "AF2-IG does not support inter-chain covalent bonds",
                    details={"reason": "inter_chain_bond", "edge": raw_record},
                )
            if left_key != right_key:
                if atom_names is None:
                    raise AF2UnsupportedChemistryError(
                        "AF2 cannot classify a cross-residue source bond",
                        details={"reason": "atom_names_missing", "edge": raw_record},
                    )
                left_name, right_name = str(atom_names[left]), str(atom_names[right])
                if (
                    left_chain == design_chain
                    and left_name == right_name == "SG"
                    and _single_bond(order)
                    and str(atoms.res_name[left]).upper() == "CYS"
                    and str(atoms.res_name[right]).upper() == "CYS"
                ):
                    # AF2-IG folds the designed peptide sequence. Its template
                    # has no explicit disulfide edge; retain the two cysteines
                    # and evaluate the same peptide confidence thresholds.
                    continue
                if {left_name, right_name} != {"C", "N"} or not _single_bond(order):
                    raise AF2UnsupportedChemistryError(
                        "AF2-IG supports only ordinary peptide bonds and one terminal cyclic C-N edge",
                        details={"reason": "unsupported_cross_residue_bond", "edge": raw_record},
                    )
                left_pos = positions.get((left_chain, left_key))
                right_pos = positions.get((right_chain, right_key))
                chain_length = len(residue_order.get(left_chain, []))
                terminal = (
                    left_chain == design_chain
                    and chain_length >= 2
                    and left_pos is not None
                    and right_pos is not None
                    and {left_pos, right_pos} == {0, chain_length - 1}
                    and ((left_name == "C" and left_pos == chain_length - 1 and right_name == "N" and right_pos == 0)
                         or (right_name == "C" and right_pos == chain_length - 1 and left_name == "N" and left_pos == 0))
                )
                if terminal:
                    cycles.append(left_chain)
                elif left_pos is not None and right_pos is not None and abs(left_pos - right_pos) == 1:
                    c_pos = left_pos if left_name == "C" else right_pos
                    n_pos = left_pos if left_name == "N" else right_pos
                    if c_pos + 1 != n_pos:
                        raise AF2UnsupportedChemistryError(
                            "AF2-IG found a reversed non-template C-N cross-residue edge",
                            details={"reason": "reversed_peptide_bond", "edge": raw_record},
                        )
                else:
                    raise AF2UnsupportedChemistryError(
                        "AF2-IG supports only adjacent ordinary peptide bonds and one terminal cyclic C-N edge",
                        details={"reason": "unsupported_cross_residue_bond", "edge": raw_record},
                    )
        for chain in dict.fromkeys(cycles):
            if chain != design_chain:
                raise AF2UnsupportedChemistryError(
                    "AF2 cyclic offset is supported only on the design chain",
                    details={"reason": "context_chain_cyclic_edge", "chain": chain},
                )
        cycle_set = [chain for chain in dict.fromkeys(cycles)]
    else:
        cycle_set = []

    declared = _attr_or_mapping(input_value, "cyclic_chains")
    if declared is None:
        declared_cycles: list[str] = []
    elif isinstance(declared, str):
        declared_cycles = [declared]
    elif isinstance(declared, Sequence):
        declared_cycles = [str(item) for item in declared]
    else:
        raise AF2UnsupportedChemistryError(
            "AF2 cyclic chain metadata is malformed",
            details={"reason": "cyclic_metadata_malformed"},
        )
    declared_set = set(declared_cycles)
    derived_set = set(cycle_set)
    if declared_set and declared_set != derived_set:
        raise AF2UnsupportedChemistryError(
            "AF2 cyclic metadata does not match the source terminal C-N graph",
            details={"reason": "cyclic_metadata_mismatch", "declared": declared_cycles, "derived": cycle_set},
        )
    if any(chain != design_chain for chain in declared_set | derived_set):
        raise AF2UnsupportedChemistryError(
            "AF2 cyclic offset is supported only on the design chain",
            details={"reason": "unsupported_cyclic_chain", "chains": sorted(declared_set | derived_set)},
        )

    # cyclic_chains remains a physical, graph-derived contract. The
    # separate offset intent is accepted only for the selected standard
    # protein design chain and never creates or relaxes a source bond.
    explicit_offset = _explicit_offset_chains(input_value)
    if explicit_offset:
        if any(chain != design_chain for chain in explicit_offset):
            raise AF2UnsupportedChemistryError(
                "AF2 cyclic offset intent is supported only on the design chain",
                details={
                    "reason": "unsupported_cyclic_offset_chain",
                    "chains": explicit_offset,
                    "design_chain": design_chain,
                },
            )
        if atoms is None:
            raise AF2UnsupportedChemistryError(
                "AF2 cyclic offset intent requires source protein residues",
                details={"reason": "cyclic_offset_residue_identity_missing"},
            )
        chains = _chain_values(atoms)
        design_rows = chains == design_chain
        if not bool(np.any(design_rows)):
            raise AF2UnsupportedChemistryError(
                "AF2 cyclic offset intent names an absent design chain",
                details={"reason": "cyclic_offset_chain_absent", "chain": design_chain},
            )
        mol_types = _array_field(atoms, "mol_type")
        if mol_types is not None:
            design_mol_types = {
                str(value).lower() for value in mol_types[design_rows].reshape(-1)
            }
            if design_mol_types != {"protein"}:
                raise AF2UnsupportedChemistryError(
                    "AF2 cyclic offset intent requires a standard protein design chain",
                    details={
                        "reason": "cyclic_offset_nonprotein_design",
                        "mol_types": sorted(design_mol_types),
                    },
                )
        residue_names = _array_field(atoms, "res_name")
        if residue_names is None:
            raise AF2UnsupportedChemistryError(
                "AF2 cyclic offset intent requires design residue identities",
                details={"reason": "cyclic_offset_residue_identity_missing"},
            )
        names = np.char.upper(np.char.strip(residue_names.astype(str)))
        bad = sorted(set(names[design_rows].tolist()) - set(_AA3_TO_1))
        if bad:
            raise AF2UnsupportedChemistryError(
                "AF2 cyclic offset intent requires standard protein residues",
                details={"reason": "cyclic_offset_nonstandard_residue", "residues": bad},
            )
        residue_order, _ = _residue_positions(atoms)
        residue_count = len(residue_order.get(design_chain, ()))
        if residue_count < 3:
            raise AF2UnsupportedChemistryError(
                "AF2 cyclic offset intent requires at least three design residues",
                details={"reason": "cyclic_offset_short_design", "length": residue_count},
            )

    effective_cycles = list(cycle_set)
    for chain in explicit_offset:
        if chain not in effective_cycles:
            effective_cycles.append(chain)

    raw_constraints = _attr_or_mapping(input_value, "bond_constraints", "bonds")
    if raw_constraints is None:
        chemistry_metadata = _attr_or_mapping(input_value, "chemistry_metadata")
        raw_constraints = _mapping_value(chemistry_metadata, "requested_constraints")
    constraints = [] if raw_constraints is None else raw_constraints
    if isinstance(constraints, str):
        constraints = [] if not constraints.strip() else [constraints]
    if isinstance(constraints, Mapping) or not isinstance(constraints, Iterable):
        raise AF2UnsupportedChemistryError(
            "AF2 requested bond ledger is malformed",
            details={"reason": "requested_bond_ledger_malformed"},
        )
    for bond_index, bond in enumerate(constraints):
        if not isinstance(bond, Mapping):
            raise AF2UnsupportedChemistryError(
                "AF2 requested bond ledger contains a non-object entry",
                details={"reason": "requested_bond_ledger_malformed", "bond_index": bond_index},
            )
        left_locator = _mapping_value(bond, "atom1", "endpoint1", "left")
        right_locator = _mapping_value(bond, "atom2", "endpoint2", "right")
        if not isinstance(left_locator, Mapping) or not isinstance(right_locator, Mapping):
            raise AF2UnsupportedChemistryError(
                "AF2 requested bond ledger requires normalized atom1/atom2 locators",
                details={"reason": "requested_bond_endpoints_missing", "bond_index": bond_index, "bond": _jsonable(bond)},
            )
        left_rows = _resolve_locator_rows(atoms, left_locator) if atoms is not None else []
        right_rows = _resolve_locator_rows(atoms, right_locator) if atoms is not None else []
        if len(left_rows) != 1 or len(right_rows) != 1:
            raise AF2UnsupportedChemistryError(
                "AF2 requested bond ledger does not resolve uniquely to source atoms",
                details={"reason": "requested_bond_unresolved", "bond_index": bond_index, "left_rows": left_rows, "right_rows": right_rows},
            )
        key = tuple(sorted((left_rows[0], right_rows[0])))
        graph_entry = edge_lookup.get(key)
        if graph_entry is None:
            raise AF2UnsupportedChemistryError(
                "AF2 requested bond is absent from the source bond graph",
                details={"reason": "requested_bond_absent", "bond_index": bond_index, "atoms": list(key)},
            )
        graph_order, _ = graph_entry
        requested_order = _normal_bond_order(_mapping_value(bond, "order", "bond_order", "value"))
        if requested_order is not None and graph_order is not None and abs(float(requested_order) - float(graph_order)) > 1e-6:
            raise AF2UnsupportedChemistryError(
                "AF2 requested bond order differs from the source graph",
                details={"reason": "requested_bond_order_mismatch", "bond_index": bond_index, "requested": requested_order, "graph": graph_order},
            )
    return {"cyclic_chains": effective_cycles}


def _assert_protein_input(input_value: Any, *, design_chain: str, context: Sequence[str]) -> dict[str, Any]:
    row = _attr_or_mapping(input_value, "row")
    modality = _attr_or_mapping(input_value, "design_modality", "modality")
    if modality is None:
        modality = _attr_or_mapping(row, "design_modality", "modality")
    if modality is not None and str(modality).lower() != "protein":
        raise AF2UnsupportedChemistryError(
            f"AF2-IG supports only protein design chains, got {modality!r}",
            details={"reason": "unsupported_design_modality", "modality": str(modality)},
        )
    atoms = _attr_or_mapping(input_value, "atoms")
    mol_types = getattr(atoms, "mol_type", None)
    if mol_types is not None:
        values = {str(value).lower() for value in np.asarray(mol_types).reshape(-1)}
        if values and values != {"protein"}:
            raise AF2UnsupportedChemistryError(
                f"AF2-IG supports only standard protein-protein inputs, got {sorted(values)}",
                details={"reason": "unsupported_molecule_types", "mol_types": sorted(values)},
            )
    residue_names = getattr(atoms, "res_name", None)
    context_unknown = False
    if residue_names is not None:
        names = np.char.upper(np.char.strip(np.asarray(residue_names).astype(str)))
        unknown = ~np.isin(names, tuple(_AA3_TO_1))
        design_rows = _chain_values(atoms) == design_chain
        context_unknown = bool(np.any(unknown & ~design_rows))
        bad = sorted(set(names[unknown & design_rows].tolist()))
        if bad:
            raise AF2UnsupportedChemistryError(
                f"AF2-IG requires standard design-chain residues, got {bad}",
                details={"reason": "nonstandard_residue", "residues": bad},
            )
    if not context:
        raise AF2CandidateError("AF2-IG complex validation requires at least one context protein chain")
    audit = _audit_chemistry(input_value, design_chain=design_chain, context=context)
    audit["context_unknown"] = context_unknown
    return audit


def validate_capability(input_value: Any) -> None:
    """Validate AF2-IG capability without importing or constructing a model."""

    design_chain = _design_chain(input_value)
    context = _context_chains(input_value, design_chain)
    _assert_protein_input(input_value, design_chain=design_chain, context=context)
    return None


def _cyclic_chain(input_value: Any, design_chain: str) -> bool:
    raw = _attr_or_mapping(input_value, "cyclic_chains")
    if raw is None:
        return False
    if isinstance(raw, str):
        chains = {raw}
    elif isinstance(raw, Sequence):
        chains = {str(item) for item in raw}
    else:
        raise AF2CandidateError("cyclic_chains must be a chain-id sequence")
    if design_chain in chains:
        return True
    if chains:
        raise AF2CandidateError(
            f"cyclic_chains does not include the selected design chain {design_chain!r}"
        )
    return False


def _input_path(input_value: Any, variant: Mapping[str, Any]) -> Path:
    raw = _mapping_value(variant, "structure_path", "input_path", "path")
    if raw is None:
        raw = _attr_or_mapping(input_value, "path", "structure_path", "input_path")
    if raw is None:
        raise AF2CandidateError("AF2 validation input has no structure path")
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_file():
        raise AF2CandidateError(f"AF2 validation structure is missing: {path}")
    return path


def _source_atoms(input_value: Any, variant: Mapping[str, Any]) -> Any:
    """Normalize every format through the authoritative standalone reader."""

    atoms = _attr_or_mapping(input_value, "atoms")
    if hasattr(atoms, "coord") and hasattr(atoms, "get_annotation_categories"):
        return atoms
    try:
        return read_structure(_input_path(input_value, variant))
    except (OSError, ValueError) as error:
        raise AF2CandidateError(f"AF2 source normalization failed: {error}") from error


def _private_pdb(input_value: Any, variant: Mapping[str, Any], output_dir: Path) -> Path:
    """Normalize source/override templates with the same occurrence/break ledger."""

    try:
        atoms = variant.get("_af2_source_atoms")
        if atoms is None:
            atoms = _source_atoms(input_value, variant)
        design_chain = _design_chain(input_value)
        ledger = variant.get("_af2_source_ledger")
        if ledger is None:
            ledger = af2_source_ledger(
                atoms, [*_context_chains(input_value, design_chain), design_chain], design_chain,
            )
        raw = _mapping_value(variant, "private_pdb", "pdb_path")
        if raw is not None:
            path = Path(str(raw)).expanduser().resolve()
            if not path.is_file():
                raise AF2CandidateError(f"AF2 private PDB is missing: {path}")
            atoms = normalize_private_override(atoms, read_structure(path), ledger)
        target = output_dir / ".af2_input.pdb"
        temporary = target.with_name(f".{target.name}.tmp")
        write_private_af2_pdb(atoms, temporary, ledger=ledger)
        os.replace(temporary, target)
        if isinstance(variant, dict):
            variant["_af2_source_ledger"] = ledger
        return target.resolve()
    except AF2CandidateError:
        raise
    except (OSError, ValueError, TypeError, IndexError) as error:
        raise AF2CandidateError(f"AF2 private template normalization failed: {error}") from error


def _restore_af2_unknown_output(model: Any) -> None:
    """Serialize native unknown receptor slots as UNK, using predicted backbone."""
    unknown = np.asarray(model._pdb["batch"]["aatype"]) == 20
    # ColabDesign takes argmax over a 20-letter target one-hot, which turns
    # an all-zero X row into ALA when serializing. Retain X only in output
    # identity; the model already received native aatype=20 at preparation.
    outputs = [model.aux]
    best = getattr(model, "_tmp", {}).get("best", {}).get("aux")
    if best is not None and best is not model.aux:
        outputs.append(best)
    for aux in outputs:
        values = aux["all"]
        aatype = np.asarray(values["aatype"]).copy()
        mask = np.asarray(values["atom_mask"]).copy()
        aatype[..., unknown] = 20
        # AF2 atom37 starts N, CA, C, CB, O. X has no asserted sidechain.
        mask[..., unknown, 3] = 0
        mask[..., unknown, 5:] = 0
        values["aatype"], values["atom_mask"] = aatype, mask


def _af2_context_unknowns(atoms: Any, design_chain: str) -> Any:
    return context_unknowns(atoms, design_chain)


def _insert_af2_context_break_gaps(atoms: Any, design_chain: str) -> None:
    """Encode each source occurrence using shared authoritative break semantics."""

    order = list(dict.fromkeys(_chain_values(atoms).tolist()))
    ledger = af2_source_ledger(atoms, order, design_chain)
    encode_private_af2_indices(atoms, ledger)


def _prepared_chain_lengths(model: Any, expected: Sequence[str]) -> list[int] | None:
    """Read the AF2 preparation ledger before ColabDesign serializes A/B."""
    pdb_data = getattr(model, "_pdb", None)
    index = pdb_data.get("idx") if isinstance(pdb_data, Mapping) else None
    if not isinstance(index, Mapping):
        # Lightweight test doubles and older ColabDesign builds lack the
        # private ledger; retain the historical direct chain-count path.
        return None
    chains = np.asarray(index.get("chain")).astype(str)
    if chains.ndim != 1:
        raise AF2CandidateError("AF2 prepared chain ledger is malformed")
    lengths: list[int] = []
    for chain in expected:
        count = int(np.count_nonzero(chains == str(chain)))
        if count <= 0:
            raise AF2CandidateError(
                f"AF2 preparation dropped expected chain {chain!r}"
            )
        lengths.append(count)
    if sum(lengths) != len(chains):
        extras = sorted(set(chains.tolist()) - {str(value) for value in expected})
        raise AF2CandidateError(
            f"AF2 preparation has unexpected chains: {extras}"
        )
    return lengths


def _rewrite_output_chain_ids(
    path: Path,
    expected: Sequence[str],
    segment_lengths: Sequence[int] | None = None,
) -> None:
    """Restore source chain identity after ColabDesign's A/B serialization."""
    if any(len(chain) != 1 for chain in expected):
        raise AF2CandidateError(
            "AF2 output chain identity requires one-character PDB chain IDs"
        )
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    observed: list[str] = []
    residue_order: list[tuple[str, str, str]] = []
    seen_residues: set[tuple[str, str, str]] = set()
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 27:
            chain = line[21].strip()
            if chain and chain not in observed:
                observed.append(chain)
            key = (chain, line[22:26], line[26])
            if chain and key not in seen_residues:
                seen_residues.add(key)
                residue_order.append(key)
    if segment_lengths is None:
        if len(observed) != len(expected):
            raise AF2CandidateError(
                f"AF2 output chain count differs: observed={observed}, expected={list(expected)}"
            )
        mapping = dict(zip(observed, expected, strict=True))
        residue_mapping = {
            (chain, res_id, ins): target
            for (chain, res_id, ins) in residue_order
            for target in (mapping[chain],)
        }
    else:
        lengths = [int(value) for value in segment_lengths]
        if len(lengths) != len(expected) or any(value <= 0 for value in lengths):
            raise AF2CandidateError(
                f"AF2 prepared chain lengths are inconsistent: lengths={lengths}, expected={list(expected)}"
            )
        if sum(lengths) != len(residue_order):
            raise AF2CandidateError(
                "AF2 output residue count differs from the prepared chain ledger: "
                f"observed={len(residue_order)}, prepared={sum(lengths)}"
            )
        residue_mapping: dict[tuple[str, str, str], str] = {}
        cursor = 0
        for target, length in zip(expected, lengths, strict=True):
            for key in residue_order[cursor:cursor + length]:
                residue_mapping[key] = target
            cursor += length
    rewritten: list[str] = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 27:
            key = (line[21].strip(), line[22:26], line[26])
            target = residue_mapping.get(key)
            if target is not None:
                line = f"{line[:21]}{target}{line[22:]}"
        rewritten.append(line)
    path.write_text("".join(rewritten), encoding="utf-8")


class AF2Backend:
    """Persistent PipeTools-style AF2-pTM complex/monomer validator."""

    def __init__(self, repo_root: Path, config: Mapping[str, Any], device: str = "cuda:0") -> None:
        if not isinstance(config, Mapping):
            raise AF2ConfigurationError("AF2 config must be a mapping")
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.config = dict(config)
        self.device = str(device)
        self.params_dir = self._resolve_asset(
            self.config.get("params_dir"),
            community_asset("colabdesign", self.repo_root) / "checkpoint" / "af2",
            base=Path.cwd() if os.environ.get("PARTIDESIGN_MODEL_DIR") else self.repo_root,
        )
        self.source_path = self._resolve_asset(
            self.config.get("source_path", self.config.get("colabdesign_source")),
            community_source("colabdesign", self.repo_root),
            base=Path.cwd() if os.environ.get("PARTIDESIGN_MODEL_DIR") else self.repo_root,
        )
        self.models: dict[str, Any] = {}
        self._last_input = None
        self._last_audit = None
        self.cache_miss_count = 0
        self.cache_hit_count = 0
        self.models_numbers = self._models_numbers(self.config.get("models", [1]))
        self.samples = self._positive_int("samples", self.config.get("samples", 1))
        self.recycles = self._positive_int("recycles", self.config.get("recycles", 3))
        raw_seed = self.config.get("seed", 101)
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise AF2ConfigurationError("AF2 seed must be an integer")
        self.seed = int(raw_seed)
        mode = str(self.config.get("mode", "both"))
        if mode not in {"complex", "monomer", "both"}:
            raise AF2ConfigurationError("AF2 mode must be complex, monomer, or both")
        # A validation request always has to produce both the binder complex
        # and the independent monomer control. Keep the generic mode helper
        # above for pure planning tests, but reject a backend configuration
        # that would silently omit either protocol.
        if mode != "both":
            raise AF2ConfigurationError(
                "AF2 validation always runs both complex and monomer protocols"
            )
        self.mode = "both"

    def validate_input(self, input: Any) -> Any:
        """Check a sample once, before any model is initialized."""
        if input is not self._last_input:
            chain = _design_chain(input)
            audit = _assert_protein_input(input, design_chain=chain,
                                          context=_context_chains(input, chain))
            self._last_input, self._last_audit = input, audit
        return self._last_audit

    @staticmethod
    def _resolve_asset(raw: Any, fallback: Path, *, base: Path | None = None) -> Path:
        path = Path(str(raw)).expanduser() if raw is not None else fallback
        if not path.is_absolute() and base is not None:
            path = base / path
        return path.resolve()

    @staticmethod
    def _positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise AF2ConfigurationError(f"AF2 {name} must be an integer >= 1")
        return int(value)

    @staticmethod
    def _models_numbers(value: Any) -> tuple[int, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            raise AF2ConfigurationError("AF2 models must be a non-empty sequence")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise AF2ConfigurationError("AF2 models must contain integers")
        result = tuple(int(item) for item in value)
        if any(item not in {1, 2} for item in result) or len(set(result)) != len(result):
            raise AF2ConfigurationError("AF2 models must be an ordered unique subset of [1, 2]")
        return result

    @property
    def protocols(self) -> tuple[str, ...]:
        return ("complex", "monomer")

    @staticmethod
    def validate_capability(input_value: Any) -> None:
        """Run the CPU-only capability check without loading AF2 assets."""

        return validate_capability(input_value)

    def expected_keys(self) -> list[str]:
        """Return this backend's ordered prediction keys without model loading."""

        return expected_keys({**self.config, "mode": "both"})

    def close(self) -> None:
        """Release transient request state and cached protocol models."""

        for model in tuple(self.models.values()):
            _clear_request_state(model)
        self.models.clear()

    def _validate_assets(self) -> None:
        if not self.params_dir.is_dir():
            raise AF2ConfigurationError(f"AF2 params_dir is not a directory: {self.params_dir}")
        for model in self.models_numbers:
            checkpoint = self.params_dir / f"params_model_{model}_ptm.npz"
            if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size < 1:
                raise AF2ConfigurationError(f"AF2 pTM checkpoint is missing or empty: {checkpoint}")
        if not self.source_path.is_dir():
            raise AF2ConfigurationError(f"AF2 source_path is not a directory: {self.source_path}")
        package = self.source_path / "colabdesign"
        if not (package / "__init__.py").is_file() and not (self.source_path / "__init__.py").is_file():
            raise AF2ConfigurationError(
                "AF2 source_path must contain the vendored colabdesign package: "
                f"{self.source_path}"
            )

    def _import_factory(self) -> Any:
        self._validate_assets()
        source = self.source_path
        import_root = source if (source / "colabdesign").is_dir() else source.parent
        source_text = str(import_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        try:
            module = importlib.import_module("colabdesign")
        except Exception as error:
            raise AF2SystemicError(f"ColabDesign import failed from {source}: {error}") from error
        origin = getattr(module, "__file__", None)
        if origin is not None:
            try:
                resolved = Path(origin).resolve()
                if source not in resolved.parents and resolved != source:
                    raise AF2SystemicError(
                        f"ambient ColabDesign import rejected: {resolved}; expected under {source}"
                    )
            except OSError as error:
                raise AF2SystemicError(f"cannot inspect ColabDesign import path: {error}") from error
        factory = getattr(module, "mk_afdesign_model", None)
        if not callable(factory):
            raise AF2SystemicError("vendored ColabDesign has no mk_afdesign_model")
        return factory

    def _build_model(self, protocol: str) -> Any:
        factory = self._import_factory()
        model_names = [f"model_{number}_ptm" for number in self.models_numbers]
        if protocol == "complex":
            kwargs = {
                "protocol": "binder",
                "use_multimer": False,
                "use_templates": True,
                "use_initial_guess": True,
                "use_initial_atom_pos": False,
                "model_names": model_names,
                "num_recycles": self.recycles,
                "data_dir": str(self.params_dir),
            }
        elif protocol == "monomer":
            kwargs = {
                "protocol": "hallucination",
                "use_templates": False,
                "initial_guess": False,
                "use_initial_atom_pos": False,
                "use_multimer": False,
                "model_names": model_names,
                "num_recycles": self.recycles,
                "data_dir": str(self.params_dir),
            }
        else:
            raise AF2ConfigurationError(f"unknown AF2 protocol: {protocol}")
        try:
            return factory(**kwargs)
        except Exception as error:
            raise AF2SystemicError(f"AF2 model construction failed for {protocol}: {error}") from error

    def _get_model(self, protocol: str) -> Any:
        if protocol in self.models:
            self.cache_hit_count += 1
            return self.models[protocol]
        model = self._build_model(protocol)
        self.models[protocol] = model
        self.cache_miss_count += 1
        return model

    def _model_local_index(self, model_number: int) -> int:
        return self.models_numbers.index(model_number)

    def _prepare(
        self, model: Any, input_value: Any, variant: Mapping[str, Any],
        protocol: str, *, audit: Mapping[str, Any],
    ) -> tuple[str, list[str], int, int, bool]:
        design_chain = _design_chain(input_value)
        context = _context_chains(input_value, design_chain)
        sequences = _mapping_value(variant, "sequences")
        sequence = _mapping_value(variant, "design_sequence", "sequence")
        if isinstance(sequences, Mapping):
            sequence = sequences.get(design_chain, sequence)
        if not isinstance(sequence, str) or not sequence or sequence.upper() != sequence or any(letter not in _AA1 for letter in sequence):
            raise AF2CandidateError("AF2 design sequence must contain canonical uppercase amino acids")
        pdb_path = variant.get("private_pdb") if protocol == "complex" else None
        ledger = variant.get("_af2_source_ledger")
        if not isinstance(ledger, Mapping):
            raise AF2CandidateError("AF2 preparation has no source occurrence ledger")
        if len(next(chain["residues"] for chain in ledger["chains"] if chain["source_chain"] == design_chain)) != len(sequence):
            raise AF2CandidateError("AF2 design sequence length differs from source occurrences")
        try:
            if protocol == "complex":
                model.prep_inputs(
                    pdb_filename=str(pdb_path),
                    chain=",".join(private_chain_id(ledger, chain) for chain in context),
                    binder_chain=private_chain_id(ledger, design_chain),
                    binder_len=len(sequence),
                    use_binder_template=True,
                    rm_target_seq=False,
                    rm_target_sc=False,
                    rm_template_ic=True,
                    ignore_missing=False,
                )
                validate_prepared_af2(model, ledger, [*context, design_chain])
            else:
                model.prep_inputs(length=len(sequence))
        except Exception as error:
            if _is_systemic(error):
                raise AF2SystemicError(f"AF2 {protocol} input preparation failed: {error}") from error
            raise AF2CandidateError(f"AF2 {protocol} input preparation failed: {error}") from error
        inputs = getattr(model, "_inputs", None)
        if not isinstance(inputs, Mapping):
            raise AF2CandidateError("AF2 model preparation produced no _inputs")
        residue_index = np.asarray(inputs.get("residue_index"))
        if residue_index.ndim != 1:
            raise AF2CandidateError("AF2 prepared residue_index is malformed")
        start = len(residue_index) - len(sequence) if protocol == "complex" else 0
        if start < 0 or start + len(sequence) > len(residue_index):
            raise AF2CandidateError("AF2 prepared design-chain slice is inconsistent")
        cyclic = design_chain in set(audit.get("cyclic_chains", ()))
        apply_cyclic_offset(
            model,
            design_start=start,
            design_length=len(sequence),
            enabled=cyclic,
        )
        return sequence, context, start, len(sequence), cyclic

    @staticmethod
    def _atomic_pdb(model: Any, path: Path) -> None:
        saver = getattr(model, "save_pdb", None)
        if not callable(saver):
            raise AF2CandidateError("ColabDesign model has no save_pdb method")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            returned = saver(str(temporary))
            if not temporary.is_file() or temporary.stat().st_size == 0:
                if isinstance(returned, bytes) and returned:
                    temporary.write_bytes(returned)
                elif isinstance(returned, str) and returned:
                    temporary.write_text(returned, encoding="utf-8")
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise AF2CandidateError("ColabDesign save_pdb produced no PDB bytes")
            os.replace(temporary, path)
        except AF2CandidateError:
            raise
        except OSError as error:
            raise AF2SystemicError(f"AF2 PDB write failed for {path}: {error}") from error
        finally:
            if temporary.exists():
                temporary.unlink()

    def predict(
        self,
        input: Any,
        variant: Mapping[str, Any],
        output_dir: Path,
        completed_keys: set[str] | Iterable[str] = (),
    ) -> Iterator[dict[str, Any]]:
        """Yield one result per requested protocol/model/sample prediction."""

        if not isinstance(variant, Mapping):
            raise AF2CandidateError("AF2 variant must be a mapping")
        completed = {str(value) for value in completed_keys}
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        seq_index = int(_mapping_value(variant, "seq_index", "sequence_index") or 0)
        variant_id = str(_mapping_value(variant, "variant_id", "id") or f"seq{seq_index}")
        sample_id = str(_mapping_value(input, "sample_id", "id") or "sample")
        # Check the unchanged input once, before loading any AF2 model.
        design_chain = _design_chain(input)
        context = _context_chains(input, design_chain)
        audit = self.validate_input(input)
        source_atoms = _source_atoms(input, variant)
        try:
            source_ledger = af2_source_ledger(source_atoms, [*context, design_chain], design_chain)
        except (ValueError, TypeError, IndexError) as error:
            raise AF2CandidateError(f"AF2 source occurrence ledger is invalid: {error}") from error
        prepared_variant = {
            **dict(variant), "output_dir": str(output_dir),
            "_af2_source_atoms": source_atoms, "_af2_source_ledger": source_ledger,
        }
        prepared_pdb = None
        with tempfile.TemporaryDirectory(prefix=f".seq{seq_index}_", dir=output_dir) as temporary:
            for protocol in self.protocols:
                model = None
                for model_number in self.models_numbers:
                    for sample in range(self.samples):
                        key = f"{protocol}_model{model_number}_sample{sample}"
                        if key in completed:
                            continue
                        try:
                            if protocol == "complex" and prepared_pdb is None:
                                prepared_pdb = _private_pdb(input, prepared_variant, Path(temporary))
                                prepared_variant["private_pdb"] = str(prepared_pdb)
                            model = self._get_model(protocol)
                            sequence, context_ids, design_start, design_length, cyclic = self._prepare(
                                model, input, prepared_variant, protocol, audit=audit,
                            )
                            seed = _stable_seed(
                                self.seed, sample_id, variant_id, protocol, model_number, sample
                            )
                            model.predict(
                                seq=sequence,
                                models=[self._model_local_index(model_number)],
                                num_recycles=self.recycles,
                                verbose=False,
                                seed=seed,
                            )
                            confidence, pae = extract_confidence(
                                model,
                                protocol,
                                design_start=design_start,
                                design_length=design_length,
                            )
                            stem = (
                                f"seq{seq_index}_{protocol}_model{model_number}_sample{sample}"
                            )
                            native_path = Path(temporary) / f"{stem}.pdb"
                            if protocol == "complex" and audit.get("context_unknown"):
                                _restore_af2_unknown_output(model)
                            self._atomic_pdb(model, native_path)
                            expected = list(context_ids) + [design_chain] if protocol == "complex" else [design_chain]
                            try:
                                folded = restore_af2_residue_identity(
                                    read_structure(native_path), source_ledger, expected,
                                )
                                structure_path = output_dir / (stem + af2_output_suffix(source_ledger, expected))
                                write_structure_atomic(folded, structure_path, entry_id=stem)
                                if structure_path.suffix == ".pdb":
                                    preserve_native_pdb_confidence(native_path, structure_path)
                            except (OSError, ValueError, TypeError, IndexError) as error:
                                raise AF2CandidateError(f"AF2 prediction source restoration failed: {error}") from error
                            yield {
                                "key": key,
                                "protocol": protocol,
                                "model": model_number,
                                "sample": sample,
                                "status": "complete",
                                "structure_path": str(structure_path),
                                "_confidence_arrays": {"pae_matrix": pae},
                                "confidence": confidence,
                                "seed": seed,
                                "source": _mapping_value(variant, "source"),
                                "source_index": _mapping_value(variant, "source_index"),
                                "cyclic": cyclic,
                            }
                        except AF2CandidateError as error:
                            yield {
                                "key": key,
                                "protocol": protocol,
                                "model": model_number,
                                "sample": sample,
                                "status": "failed",
                                "structure_path": "",
                                "confidence": {},
                                "error": str(error),
                            }
                        except Exception as error:
                            if _is_systemic(error):
                                self.models.pop(protocol, None)
                                raise AF2SystemicError(
                                    f"AF2 {protocol}/model{model_number}/sample{sample} failed systemically: {error}"
                                ) from error
                            yield {
                                "key": key,
                                "protocol": protocol,
                                "model": model_number,
                                "sample": sample,
                                "status": "failed",
                                "structure_path": "",
                                "confidence": {},
                                "error": f"{type(error).__name__}: {error}",
                            }
                        finally:
                            _clear_request_state(model)



__all__ = [
    "AF2Backend",
    "AF2CandidateError",
    "AF2ConfigurationError",
    "AF2Error",
    "AF2SystemicError",
    "AF2UnsupportedChemistryError",
    "apply_cyclic_offset",
    "extract_confidence",
    "expected_keys",
    "signed_circular_offset",
    "validate_capability",
]
