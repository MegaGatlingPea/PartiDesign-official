"""Minimal, deterministic standalone canonical compiler.

The compiler accepts the source-faithful Biotite AtomArray and native
TokenArray produced by the scoring reader/projection.  It performs only the
four inference operations needed before the scorer: Angstrom-to-nanometre
conversion, source-authoritative chain-break detection, deterministic
zero-randomness centering, and canonical tensor emission.

No training/data graph type is imported here.  The canonical field registry and
defaults live in src.scoring.schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import biotite.structure as struc
import numpy as np
import torch

from src.common.constants import (
    ATOM91_SIZE,
    DNA_ATOM27,
    ELEMENTS_128,
    LIGAND_BOND_TYPES,
    LIGAND_FORMAL_CHARGE_TO_CLASS,
    MOL_TYPE_TO_ID,
    PROTEIN_ATOM37,
    RNA_ATOM27,
    TOKEN_TYPE32,
    TOKEN_TYPE_IGNORE_INDEX,
    atom_slot_index,
    element_id,
    normalise_atom_name,
    normalise_res_name,
    polymer_slot_to_atom91,
)
from .schema import empty_canonical_sample, validate_canonical_sample
from .structure import STD_RESIDUES


_POLYMER_MOL_TYPES = frozenset(("protein", "rna", "dna"))
_COORDINATION_BOND = int(struc.BondType.COORDINATION)


def _annotation(
    atoms: Any,
    name: str,
    n: int,
    *,
    default: Any,
    dtype: Any,
) -> np.ndarray:
    if hasattr(atoms, name):
        value = np.asarray(getattr(atoms, name))
        if value.shape != (n,):
            raise ValueError(
                f"AtomArray annotation {name!r} has shape {value.shape}; "
                f"expected {(n,)}"
            )
        return value
    return np.full(n, default, dtype=dtype)


def _token_centres(tokens: Any, n_atom: int) -> np.ndarray:
    try:
        values = tokens.get_annotation("centre_atom_index")
    except (AttributeError, KeyError):
        values = [
            int(token.atom_indices[0])
            for token in tokens
            if getattr(token, "atom_indices", None)
        ]
    centres = np.asarray(values, dtype=np.int64)
    if centres.ndim != 1 or len(centres) != len(tokens):
        raise ValueError("TokenArray centre_atom_index must have one entry per token")
    if len(centres) == 0 or int(centres.min()) < 0 or int(centres.max()) >= n_atom:
        raise ValueError("token centres do not map to source atoms")
    return centres


def _token_atom_indices(tokens: Any, n_atom: int) -> tuple[tuple[int, ...], np.ndarray]:
    owner = np.full(n_atom, -1, dtype=np.int64)
    groups: list[tuple[int, ...]] = []
    for token_index, token in enumerate(tokens):
        raw = getattr(token, "atom_indices", None)
        if raw is None:
            raise ValueError(f"token {token_index} has no atom_indices")
        indices = tuple(int(index) for index in raw)
        if not indices or any(index < 0 or index >= n_atom for index in indices):
            raise ValueError(f"token {token_index} has invalid atom_indices")
        if len(set(indices)) != len(indices):
            raise ValueError(f"token {token_index} repeats an atom")
        if np.any(owner[list(indices)] >= 0):
            raise ValueError("source atom belongs to more than one token")
        owner[list(indices)] = token_index
        groups.append(indices)
    if len(tokens) == 0 or np.any(owner < 0):
        missing = np.flatnonzero(owner < 0).tolist()
        raise ValueError(f"tokens do not cover source atoms: {missing[:20]}")
    return tuple(groups), owner


def _bond_rows(atoms: Any) -> np.ndarray:
    if atoms.bonds is None:
        return np.zeros((0, 3), dtype=np.int64)
    rows = np.asarray(atoms.bonds.as_array())
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] < 3:
        raise ValueError("AtomArray bonds must have three columns")
    return rows[:, :3].astype(np.int64, copy=False)


def _canonical_bond_type(raw_type: int) -> int:
    """Match the reference target-side Biotite-to-model bond ABI."""

    raw = int(raw_type)
    if raw in (0, 1):
        return LIGAND_BOND_TYPES["single"]
    if raw in (2,):
        return LIGAND_BOND_TYPES["double"]
    if raw in (3, 4):
        return LIGAND_BOND_TYPES["triple"]
    if raw in (5, 6, 7, 9):
        return LIGAND_BOND_TYPES["aromatic"]
    if raw == 8:
        return LIGAND_BOND_TYPES["none"]
    raise ValueError(f"unexpected Biotite bond type: {raw_type}")


def _condition_bond_state(raw_type: int) -> tuple[bool, bool, int]:
    """Return the condition-side known/order flags for one source edge."""

    raw = int(raw_type)
    if raw in (0, 4):
        # ANY and QUADRUPLE preserve edge existence while their order is not
        # represented exactly by the four-state condition ABI.
        return True, False, LIGAND_BOND_TYPES["none"]
    if raw == 1:
        return True, True, LIGAND_BOND_TYPES["single"]
    if raw == 2:
        return True, True, LIGAND_BOND_TYPES["double"]
    if raw == 3:
        return True, True, LIGAND_BOND_TYPES["triple"]
    if raw in (5, 6, 7, 9):
        return True, True, LIGAND_BOND_TYPES["aromatic"]
    if raw == 8:
        return False, False, LIGAND_BOND_TYPES["none"]
    raise ValueError(f"unexpected Biotite condition bond type: {raw_type}")


def _bond_kept_mask(atoms: Any, rows: np.ndarray, carrier_mask: np.ndarray) -> np.ndarray:
    """Keep the same source edges as Protenix target bond features.

    Standard polymer connectivity is implicit in the model.  Discontinuous
    edges, nonstandard edges, and all carrier edges remain explicit.
    """

    if len(rows) == 0:
        return np.zeros(0, dtype=bool)
    n = len(atoms)
    mol = np.asarray(atoms.mol_type).astype(str)
    res_name = np.asarray(atoms.res_name).astype(str)
    res_id = _annotation(atoms, "res_id", n, default=-1, dtype=np.int64).astype(np.int64)
    chain = _annotation(atoms, "chain_id", n, default="", dtype="U1").astype(str)
    ref_space = _annotation(atoms, "ref_space_uid", n, default=-1, dtype=np.int64)
    polymer = np.isin(mol, tuple(_POLYMER_MOL_TYPES))
    standard = np.isin(np.char.upper(res_name), list(STD_RESIDUES)) & polymer
    i = rows[:, 0]
    j = rows[:, 1]
    std_std = standard[i] & standard[j]
    std_unstd = (standard[i] & ~standard[j] & polymer[j]) | (
        standard[j] & ~standard[i] & polymer[i]
    )
    inter_unstd = (
        ~standard[i] & polymer[i] & ~standard[j] & polymer[j]
        & (ref_space[i] != ref_space[j])
    )
    kept = ~(std_std | std_unstd | inter_unstd)
    kept |= (np.abs(res_id[i] - res_id[j]) > 1) | (chain[i] != chain[j])
    kept |= carrier_mask[i] | carrier_mask[j]
    return kept


def _has_named_bond(
    atoms: Any,
    left: tuple[int, ...],
    right: tuple[int, ...],
    left_names: set[str],
    right_names: set[str],
    pairs: set[tuple[int, int]] | None,
) -> bool:
    if atoms.bonds is None:
        return False
    if pairs is None:
        pairs = {
            (min(int(i), int(j)), max(int(i), int(j)))
            for i, j, _ in _bond_rows(atoms)
        }
    names = np.asarray(atoms.atom_name).astype(str)
    left_rows = [
        index for index in left
        if normalise_atom_name(names[index]) in left_names
    ]
    right_rows = [
        index for index in right
        if normalise_atom_name(names[index]) in right_names
    ]
    return any(
        (min(i, j), max(i, j)) in pairs
        for i in left_rows for j in right_rows
    )


def _chain_breaks(
    atoms: Any,
    groups: tuple[tuple[int, ...], ...],
    centres: np.ndarray,
    coords_nm: torch.Tensor,
    asym_id: np.ndarray,
    residue_index: np.ndarray,
) -> tuple[torch.Tensor, str]:
    n_token = len(groups)
    result = torch.ones(n_token, dtype=torch.bool)
    if n_token <= 1:
        return result, "ChainBreakPerResidueTransform.v1"
    mol = np.asarray(atoms.mol_type).astype(str).astype(object)
    bond_pairs = None
    if atoms.bonds is not None:
        bond_pairs = {
            (min(int(i), int(j)), max(int(i), int(j)))
            for i, j, _ in _bond_rows(atoms)
        }
    for index in range(n_token - 1):
        left, right = index, index + 1
        mol_left = str(mol[centres[left]]).strip().lower()
        mol_right = str(mol[centres[right]]).strip().lower()
        if mol_left != mol_right or mol_left == "ligand":
            result[index] = True
            continue
        same_chain = int(asym_id[left]) == int(asym_id[right])
        adjacent_residue = int(residue_index[right]) - int(residue_index[left]) in (0, 1)
        if not same_chain or not adjacent_residue:
            result[index] = True
            continue
        if mol_left == "protein":
            if atoms.bonds is not None:
                result[index] = not _has_named_bond(
                    atoms, groups[left], groups[right], {"C"}, {"N"}, bond_pairs
                )
            else:
                left_ca = next(
                    (atom_index for atom_index in groups[left]
                     if normalise_atom_name(atoms.atom_name[atom_index]) == "CA"),
                    None,
                )
                right_ca = next(
                    (atom_index for atom_index in groups[right]
                     if normalise_atom_name(atoms.atom_name[atom_index]) == "CA"),
                    None,
                )
                close = (
                    left_ca is not None and right_ca is not None
                    and torch.linalg.vector_norm(coords_nm[right_ca] - coords_nm[left_ca])
                    <= 0.4
                )
                result[index] = not bool(close)
        elif mol_left in {"rna", "dna"}:
            if atoms.bonds is not None:
                result[index] = not _has_named_bond(
                    atoms, groups[left], groups[right], {"O3'"}, {"P"}, bond_pairs
                )
            else:
                result[index] = True
        else:
            result[index] = True
    result[-1] = True
    return result, "ChainBreakPerResidueTransform.v1"


def _safe_frame(
    p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    eps = 1e-8
    e1 = (p3 - p2) / torch.linalg.vector_norm(p3 - p2).clamp_min(eps)
    raw_e2 = p1 - p2
    e3 = torch.linalg.cross(e1, raw_e2, dim=-1)
    ok = bool(torch.linalg.vector_norm(e3).item() > 1e-6)
    e3 = e3 / torch.linalg.vector_norm(e3).clamp_min(eps)
    e2 = torch.linalg.cross(e3, e1, dim=-1)
    e2 = e2 / torch.linalg.vector_norm(e2).clamp_min(eps)
    return torch.stack((e1, e2, e3), dim=-1), ok


def _charge_arrays(atoms: Any, n: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    charge_present = hasattr(atoms, "charge")
    native = (
        np.asarray(getattr(atoms, "charge"), dtype=np.float64)
        if charge_present else np.zeros(n, dtype=np.float64)
    )
    if native.shape != (n,) or not np.isfinite(native).all():
        raise ValueError("charge must be finite per atom")
    native_round = np.rint(native).astype(np.int64)
    if not np.allclose(native, native_round, atol=1e-6, rtol=0):
        raise ValueError("charge must contain integer formal-charge values")
    ref_charge_present = hasattr(atoms, "ref_charge")
    ref_mask_present = hasattr(atoms, "ref_mask")
    if ref_charge_present != ref_mask_present:
        raise ValueError("ref_charge and ref_mask must be supplied together")
    effective = native_round
    if ref_charge_present:
        ref = np.asarray(getattr(atoms, "ref_charge"), dtype=np.float64)
        mask = np.asarray(getattr(atoms, "ref_mask"), dtype=bool)
        if ref.shape != mask.shape or ref.shape != (n,):
            raise ValueError("ref_charge/ref_mask shape mismatch")
        if not np.isfinite(ref[mask]).all():
            raise ValueError("reference charges must be finite where mapped")
        ref_round = np.zeros(n, dtype=np.int64)
        ref_round[mask] = np.rint(ref[mask]).astype(np.int64)
        if not np.allclose(ref[mask], ref_round[mask], atol=1e-6, rtol=0):
            raise ValueError("reference charge must contain integer values")
        effective = native_round.copy()
        effective[mask] = ref_round[mask]
        trusted = mask.copy()
        authority = "ccd_reference"
    else:
        trusted = np.full(n, charge_present, dtype=bool)
        authority = "native_without_reference_metadata"
    supported = np.isin(effective, np.asarray(tuple(LIGAND_FORMAL_CHARGE_TO_CLASS), dtype=np.int64))
    trusted_before_vocabulary = trusted.copy()
    trusted &= supported
    dataset_source = str(getattr(atoms, "dataset_source", "")).strip()
    return effective, trusted, {
        "dataset_source": dataset_source or "not_available",
        "authority": authority,
        "native_source": False,
        "charge_annotation_present": bool(charge_present),
        "reference_annotation_present": bool(ref_charge_present),
        "trusted_atom_count": int(trusted.sum()),
        "unsupported_vocabulary_atom_count": int(
            (trusted_before_vocabulary & ~supported).sum()
        ),
    }


def _metadata_dict(value: Any) -> dict[str, Any]:
    """Convert optional request/plan metadata without importing training types."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    values = getattr(value, "__dict__", None)
    if isinstance(values, Mapping):
        return deepcopy(dict(values))
    return {}


def _source_mapping(
    atoms: Any,
    tokens: Any,
    owner: np.ndarray,
    local_slots: np.ndarray,
    atom91_slots: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
    modified_indices: list[int],
) -> dict[str, Any]:
    n_atom, n_token = len(atoms), len(tokens)
    source_raw = _annotation(atoms, "source_atom_index", n_atom, default=-1, dtype=np.int64).astype(np.int64)
    if np.any(source_raw < 0):
        crop_source = np.arange(n_atom, dtype=np.int64)
        source_status = "crop_local_only"
    else:
        crop_source = source_raw
        source_status = "source_crop_bijection_verified"
    if len(np.unique(crop_source)) != n_atom:
        raise ValueError("source_atom_index must be unique")
    source_count = int(crop_source.max(initial=-1)) + 1
    source_to_crop = np.full(source_count, -1, dtype=np.int64)
    source_to_crop[crop_source] = np.arange(n_atom, dtype=np.int64)
    try:
        crop_token_source = np.asarray(tokens.get_annotation("source_token_index"), dtype=np.int64)
    except (AttributeError, KeyError):
        crop_token_source = np.arange(n_token, dtype=np.int64)
    if crop_token_source.shape != (n_token,) or np.any(crop_token_source < 0):
        crop_token_source = np.arange(n_token, dtype=np.int64)
    if len(np.unique(crop_token_source)) != n_token:
        raise ValueError("source_token_index must be unique")
    token_count = int(crop_token_source.max(initial=-1)) + 1
    source_token_to_crop = np.full(token_count, -1, dtype=np.int64)
    source_token_to_crop[crop_token_source] = np.arange(n_token, dtype=np.int64)
    selected_condition = _annotation(
        atoms, "is_condition_atom", n_atom, default=False, dtype=bool
    ).astype(bool)
    dataset_source = str(getattr(atoms, "dataset_source", "")).strip()
    return {
        "coordinate_units": "nm",
        "source_mapping_status": source_status,
        "crop_atom_to_source": crop_source.tolist(),
        "source_atom_to_crop": source_to_crop.tolist(),
        "source_atom_count": int(len(source_to_crop)),
        "crop_atom_to_canonical_token": owner.tolist(),
        "crop_atom_to_modality_slot": local_slots.tolist(),
        "crop_atom_to_atom91_slot": atom91_slots.tolist(),
        "crop_token_to_source": crop_token_source.tolist(),
        "source_token_to_crop": source_token_to_crop.tolist(),
        "source_token_count": int(len(source_token_to_crop)),
        "crop_token_to_canonical": list(range(n_token)),
        "canonical_to_crop_tokens": [[int(index)] for index in range(n_token)],
        "modified_canonical_token_indices": [int(index) for index in modified_indices],
        "unsupported_extra_atoms": [],
        "selected_condition_atom_indices": np.flatnonzero(selected_condition).tolist(),
        "condition": {},
        "polymer_ss": {},
        "polymer_coarse_ss": {},
        "chain_break_authority": "ChainBreakPerResidueTransform.v1",
        "require_resolved_token_centers": False,
        "augmentation_rng": "torch_worker_state",
        "dataset_source": dataset_source or "not_available",
        "source_output_selection": {},
        "sample_request": _metadata_dict(getattr(atoms, "sample_request", None)),
        "sample_plan": _metadata_dict(getattr(atoms, "sample_plan", None)),
    }


def build_canonical(atoms: Any, tokens: Any, identity: str) -> dict[str, Any]:
    """Compile one source-faithful structure into the frozen 84-field ABI."""

    n_atom = len(atoms)
    if n_atom <= 0:
        raise ValueError("structure has no atoms")
    if not hasattr(atoms, "coord"):
        raise ValueError("AtomArray is missing coordinates")
    coord_ang = np.asarray(atoms.coord)
    if coord_ang.shape != (n_atom, 3) or not np.isfinite(coord_ang).all():
        raise ValueError("coordinates must be finite with shape [n_atoms, 3]")
    mol = np.char.lower(np.asarray(atoms.mol_type).astype(str))
    if not set(mol.tolist()) <= set(_POLYMER_MOL_TYPES) | {"ligand"}:
        raise ValueError("structure contains an unsupported modality")
    groups, owner = _token_atom_indices(tokens, n_atom)
    centres = _token_centres(tokens, n_atom)
    n_token = len(groups)
    is_resolved = _annotation(atoms, "is_resolved", n_atom, default=True, dtype=bool).astype(bool)
    if hasattr(atoms, "source_mol_id"):
        source_mol = _annotation(
            atoms, "source_mol_id", n_atom, default=-1, dtype=np.int64
        ).astype(np.int64)
    else:
        # The legacy field defaults source ownership to the model molecule
        # annotation when that annotation exists, and otherwise preserves the
        # explicit unavailable sentinel.
        source_mol = _annotation(
            atoms, "mol_id", n_atom, default=-1, dtype=np.int64
        ).astype(np.int64)
    model_mol = (
        _annotation(atoms, "mol_id", n_atom, default=-1, dtype=np.int64).astype(np.int64)
        if hasattr(atoms, "mol_id")
        else np.arange(n_atom, dtype=np.int64)
    )
    entity_mol = (
        _annotation(atoms, "entity_mol_id", n_atom, default=-1, dtype=np.int64).astype(np.int64)
        if hasattr(atoms, "entity_mol_id")
        else np.arange(n_atom, dtype=np.int64)
    )
    asym = _annotation(atoms, "asym_id_int", n_atom, default=-1, dtype=np.int64).astype(np.int64)
    if np.any(asym < 0):
        asym = _annotation(atoms, "asym_id", n_atom, default=-1, dtype=np.int64).astype(np.int64)
    if np.any(asym < 0):
        asym = np.zeros(n_atom, dtype=np.int64)
    entity = _annotation(atoms, "entity_id_int", n_atom, default=-1, dtype=np.int64).astype(np.int64)
    if np.any(entity < 0):
        entity = _annotation(atoms, "entity_id", n_atom, default=-1, dtype=np.int64).astype(np.int64)
    if np.any(entity < 0):
        entity = np.zeros(n_atom, dtype=np.int64)
    residue_index_atom = _annotation(atoms, "res_id", n_atom, default=-1, dtype=np.int64).astype(np.int64)
    chain = _annotation(atoms, "chain_id", n_atom, default="", dtype="U1").astype(str)
    carrier_mask_atom = _annotation(atoms, "modified_polymer_carrier_mask", n_atom, default=False, dtype=bool).astype(bool)
    proxy_mask_atom = _annotation(atoms, "modified_polymer_proxy_mask", n_atom, default=False, dtype=bool).astype(bool)
    was_modified_atom = _annotation(atoms, "was_modified_polymer", n_atom, default=False, dtype=bool).astype(bool)
    carrier_key_atom = _annotation(atoms, "modified_polymer_carrier_residue_key", n_atom, default="", dtype="U256").astype(str)
    cano_atom = _annotation(atoms, "cano_seq_resname", n_atom, default="", dtype="U16").astype(str)
    raw_res = np.asarray(atoms.res_name).astype(str)
    atom_name = np.asarray(atoms.atom_name).astype(str)
    element = np.asarray(atoms.element).astype(str)
    charge, charge_trusted, charge_provenance = _charge_arrays(atoms, n_atom)

    carrier_group = np.zeros(n_token, dtype=bool)
    modified_group = np.zeros(n_token, dtype=bool)
    source_group_mol = np.empty(n_token, dtype=object)
    model_group_mol = np.empty(n_token, dtype=object)
    canonical_res = np.empty(n_token, dtype=object)
    for token_index, atom_indices in enumerate(groups):
        centre = int(centres[token_index])
        source_group_mol[token_index] = str(mol[centre])
        carrier_group[token_index] = bool(carrier_mask_atom[list(atom_indices)].any())
        modified_group[token_index] = bool(
            was_modified_atom[list(atom_indices)].any()
            or proxy_mask_atom[list(atom_indices)].any()
            or carrier_group[token_index]
        )
        model_group_mol[token_index] = "ligand" if carrier_group[token_index] else str(mol[centre])
        if str(mol[centre]) in _POLYMER_MOL_TYPES:
            candidates = [normalise_res_name(value) for value in cano_atom[list(atom_indices)] if str(value).strip()]
            canonical_res[token_index] = candidates[0] if candidates else normalise_res_name(raw_res[centre])
        else:
            canonical_res[token_index] = normalise_res_name(raw_res[centre])
    model_mol_token = model_mol[centres].astype(np.int64, copy=True)
    used_model_ids = {int(value) for value in model_mol.tolist() if int(value) >= 0}
    next_model_id = max(1, max(used_model_ids, default=-1) + 1)
    carrier_private_ids: dict[str, int] = {}
    for token_index, atom_indices in enumerate(groups):
        if not carrier_group[token_index]:
            continue
        centre = int(centres[token_index])
        key = str(carrier_key_atom[centre])
        if not key:
            key = ":".join(
                (str(source_group_mol[token_index]), str(chain[centre]),
                 str(asym[centre]), str(residue_index_atom[centre]),
                 str(raw_res[centre]))
            )
        if key not in carrier_private_ids:
            while next_model_id in used_model_ids:
                next_model_id += 1
            carrier_private_ids[key] = next_model_id
            used_model_ids.add(next_model_id)
            next_model_id += 1
        model_mol_token[token_index] = carrier_private_ids[key]

    coords_nm_raw = torch.as_tensor(coord_ang, dtype=torch.float32) / 10.0
    condition = _annotation(atoms, "is_condition_atom", n_atom, default=False, dtype=bool).astype(bool)
    if bool(condition.any()):
        if bool((condition & ~is_resolved).any()):
            raise ValueError("is_condition_atom selects unresolved source atoms")
        center_coords = coords_nm_raw[torch.as_tensor(condition)]
        center_source = "condition_atoms"
    else:
        center_indices = torch.as_tensor(centres, dtype=torch.long)
        center_coords = coords_nm_raw[center_indices]
        center_source = "token_representative_atoms"
        center_mask = torch.as_tensor(is_resolved[centres], dtype=torch.bool)
        if not bool(center_mask.any()):
            raise ValueError(f"No resolved atoms available for center source {center_source!r}")
        center_coords = center_coords[center_mask]
    center_origin = center_coords.mean(dim=0, dtype=torch.float32)
    coords_nm = coords_nm_raw - center_origin[None, :]

    local_slots = np.full(n_atom, -1, dtype=np.int64)
    atom91_slots = np.full(n_atom, -1, dtype=np.int64)
    sample = empty_canonical_sample(n_token)
    sample["pdb_id"] = str(identity)
    sample["center_source"] = center_source
    sample["center_origin_nm"] = center_origin.clone()
    sample["source_to_augmented_affine"][3, :3] = -center_origin
    sample["post_center_translation_nm"].zero_()
    # Filled after modality slots are resolved; unsupported extras are tracked
    # separately and are not target-valid canonical atoms.
    sample["num_resolved_atoms"] = torch.tensor(0, dtype=torch.long)
    sample["num_unsupported_extra_atoms"] = torch.tensor(0, dtype=torch.long)
    sample["chain_break_after_token"], chain_break_authority = _chain_breaks(
        atoms, groups, centres, coords_nm, asym[centres], residue_index_atom[centres]
    )

    sample["mol_type_id"] = torch.as_tensor(
        [MOL_TYPE_TO_ID[str(value)] for value in model_group_mol], dtype=torch.long
    )
    sample["ligand_mask"] = sample["mol_type_id"] == MOL_TYPE_TO_ID["ligand"]
    token_types: list[int] = []
    for token_index in range(n_token):
        if model_group_mol[token_index] == "ligand":
            token_types.append(TOKEN_TYPE_IGNORE_INDEX)
        else:
            residue = normalise_res_name(str(canonical_res[token_index]))
            modality = str(source_group_mol[token_index])
            fallback = {"protein": "UNK", "rna": "N", "dna": "DN"}[modality]
            token_types.append(int(TOKEN_TYPE32.get(residue, TOKEN_TYPE32[fallback])))
    sample["token_type32"] = torch.as_tensor(token_types, dtype=torch.long)
    sample["residue_index"] = torch.as_tensor(residue_index_atom[centres], dtype=torch.long)
    sample["asym_id"] = torch.as_tensor(asym[centres], dtype=torch.long)
    sample["entity_id"] = torch.as_tensor(entity[centres], dtype=torch.long)
    sample["mol_id"] = torch.as_tensor(model_mol_token, dtype=torch.long)
    sample["source_mol_id"] = torch.as_tensor(source_mol[centres], dtype=torch.long)
    sample["entity_mol_id"] = torch.as_tensor(entity_mol[centres], dtype=torch.long)
    sample["token_center_coords_nm"] = coords_nm[torch.as_tensor(centres, dtype=torch.long)]
    sample["token_center_resolved_mask"] = torch.as_tensor(is_resolved[centres], dtype=torch.bool)
    sample["modified_polymer_token_mask"] = torch.as_tensor(modified_group, dtype=torch.bool)
    carrier_element = _annotation(atoms, "modified_polymer_carrier_element", n_atom, default="", dtype="U1").astype(str)
    unknown_element = element_id("UNK_ELEM_128")
    sample["ligand_element_type"] = torch.as_tensor(
        [element_id(str(carrier_element[int(centres[index])]))
         if carrier_group[index] else (
             element_id(element[int(centres[index])])
             if bool(sample["ligand_mask"][index]) else unknown_element
         ) for index in range(n_token)],
        dtype=torch.long,
    )

    carrier_charge = _annotation(atoms, "modified_polymer_carrier_charge", n_atom, default=0, dtype=np.int64).astype(np.int64)
    effective_token_charge = np.zeros(n_token, dtype=np.float32)
    trusted_token_charge = np.zeros(n_token, dtype=bool)
    for token_index, atom_indices in enumerate(groups):
        centre = int(centres[token_index])
        if carrier_group[token_index]:
            effective_token_charge[token_index] = float(carrier_charge[centre])
            trusted_token_charge[token_index] = int(carrier_charge[centre]) in LIGAND_FORMAL_CHARGE_TO_CLASS
        elif sample["ligand_mask"][token_index]:
            effective_token_charge[token_index] = float(charge[centre])
            trusted_token_charge[token_index] = bool(charge_trusted[centre])
    sample["ligand_charge"] = torch.as_tensor(effective_token_charge, dtype=torch.float32)
    if bool(trusted_token_charge.any()):
        for token_index in np.flatnonzero(trusted_token_charge):
            sample["ligand_formal_charge_class"][token_index] = int(
                LIGAND_FORMAL_CHARGE_TO_CLASS[int(round(effective_token_charge[token_index]))]
            )
        sample["ligand_formal_charge_mask"] = torch.as_tensor(trusted_token_charge, dtype=torch.bool)

    protein_coords = sample["protein_atom37_coords"]
    protein_mask = sample["protein_atom37_mask"]
    rna_coords = sample["rna_atom27_coords"]
    rna_mask = sample["rna_atom27_mask"]
    dna_coords = sample["dna_atom27_coords"]
    dna_mask = sample["dna_atom27_mask"]
    unsupported: list[dict[str, Any]] = []
    occupied: set[tuple[int, int]] = set()
    coordinate_rows: dict[str, list[tuple[int, int, int]]] = {
        "protein": [], "rna": [], "dna": [],
    }
    for atom_index in range(n_atom):
        token_index = int(owner[atom_index])
        if model_group_mol[token_index] == "ligand":
            # The canonical ligand head has one global atom slot (ATOM91).
            # Keep this local slot in provenance even though ligand atoms do
            # not populate the polymer atom91 coordinate bank.
            local_slots[atom_index] = 0
            continue
        modality = str(source_group_mol[token_index])
        residue = str(canonical_res[token_index])
        slot = atom_slot_index(modality, residue, atom_name[atom_index])
        if slot is None:
            unsupported.append({
                "crop_atom_index": int(atom_index),
                "canonical_token_index": token_index,
                "mol_type": modality,
                "res_name": residue,
                "atom_name": normalise_atom_name(atom_name[atom_index]),
                "resolved": bool(is_resolved[atom_index]),
                "modified_polymer": bool(modified_group[token_index]),
                "reason": "unsupported_extra",
            })
            continue
        local_slots[atom_index] = int(slot)
        atom91_slots[atom_index] = int(polymer_slot_to_atom91(modality, int(slot)))
        if not is_resolved[atom_index]:
            continue
        key = (token_index, int(slot))
        if key in occupied:
            raise ValueError(f"canonical atom slot collision at token={token_index}, slot={slot}")
        occupied.add(key)
        coordinate_rows[modality].append((token_index, int(slot), atom_index))
    # Scatter each modality once. Scalar tensor assignments for every source
    # atom add preparation overhead without changing the canonical mapping.
    coordinate_banks = {
        "protein": (protein_coords, protein_mask),
        "rna": (rna_coords, rna_mask),
        "dna": (dna_coords, dna_mask),
    }
    for modality, mapped_rows in coordinate_rows.items():
        if not mapped_rows:
            continue
        indices = torch.as_tensor(mapped_rows, dtype=torch.long)
        target_coords, target_mask = coordinate_banks[modality]
        target_coords[indices[:, 0], indices[:, 1]] = coords_nm[indices[:, 2]]
        target_mask[indices[:, 0], indices[:, 1]] = True
    sample["polymer_atom91_coords"] = torch.cat(
        (sample["protein_atom37_coords"], sample["rna_atom27_coords"], sample["dna_atom27_coords"]),
        dim=1,
    )
    sample["polymer_atom91_mask"] = torch.cat(
        (sample["protein_atom37_mask"], sample["rna_atom27_mask"], sample["dna_atom27_mask"]),
        dim=1,
    )
    rows = _bond_rows(atoms)
    target_valid_atom = is_resolved & (local_slots >= 0)
    if bool((condition & ~target_valid_atom).any()):
        raise ValueError("is_condition_atom selects an unresolved or unsupported atom")
    if bool(condition.any()):
        full_condition_token = torch.zeros(n_token, dtype=torch.bool)
        for token_index, atom_indices in enumerate(groups):
            valid = [
                atom_index for atom_index in atom_indices
                if target_valid_atom[atom_index]
            ]
            full_condition_token[token_index] = bool(
                valid and all(bool(condition[atom_index]) for atom_index in valid)
            )
        sample["is_condition_token"] = full_condition_token
        full_condition_entity = torch.zeros(n_token, dtype=torch.bool)
        entity_groups: dict[tuple[str, int], list[int]] = {}
        for token_index in range(n_token):
            if source_group_mol[token_index] in _POLYMER_MOL_TYPES:
                key = ("polymer", int(asym[centres[token_index]]))
            else:
                key = ("ligand", int(model_mol_token[token_index]))
            entity_groups.setdefault(key, []).append(token_index)
        for token_indices in entity_groups.values():
            if token_indices and bool(full_condition_token[token_indices].all().item()):
                full_condition_entity[token_indices] = True
        sample["is_condition_entity_token"] = full_condition_entity
        selected_count = int((condition & target_valid_atom).sum())
        target_count = int(target_valid_atom.sum())
        sample["condition_token_fraction"] = full_condition_token.float().mean()
        sample["condition_atom_fraction"] = torch.tensor(
            selected_count / max(target_count, 1), dtype=torch.float32
        )
        for atom_index in np.flatnonzero(condition):
            token_index = int(owner[atom_index])
            slot = int(local_slots[atom_index])
            model = str(model_group_mol[token_index])
            if model == "ligand":
                sample["ligand_motif_token_center_coords_nm"][token_index] = coords_nm[atom_index]
                sample["ligand_motif_mask"][token_index] = True
                continue
            target = {
                "protein": ("protein_motif_atom37_coords_nm", "protein_motif_atom37_mask"),
                "rna": ("rna_motif_atom27_coords_nm", "rna_motif_atom27_mask"),
                "dna": ("dna_motif_atom27_coords_nm", "dna_motif_atom27_mask"),
            }[str(source_group_mol[token_index])]
            coord_field, mask_field = target
            sample[coord_field][token_index, slot] = coords_nm[atom_index]
            sample[mask_field][token_index, slot] = True
        # A source edge is visible when both endpoints are selected.  Preserve
        # endpoint slots and exact order when the source bond carries one.
        for atom_i, atom_j, raw_type in rows:
            atom_i, atom_j, raw_type = int(atom_i), int(atom_j), int(raw_type)
            if not (condition[atom_i] and condition[atom_j]):
                continue
            if local_slots[atom_i] < 0 or local_slots[atom_j] < 0:
                continue
            token_i, token_j = int(owner[atom_i]), int(owner[atom_j])
            if token_i == token_j:
                continue
            known, order_known, bond_type = _condition_bond_state(raw_type)
            if not known:
                continue
            endpoint_i = (
                ATOM91_SIZE if model_group_mol[token_i] == "ligand"
                else int(atom91_slots[atom_i])
            )
            endpoint_j = (
                ATOM91_SIZE if model_group_mol[token_j] == "ligand"
                else int(atom91_slots[atom_j])
            )
            if token_i > token_j:
                token_i, token_j = token_j, token_i
                endpoint_i, endpoint_j = endpoint_j, endpoint_i
            sample["condition_bond_pair_mask"][token_i, token_j] = True
            sample["condition_bond_pair_mask"][token_j, token_i] = True
            sample["condition_bond_pair_endpoint_slot"][token_i, token_j] = torch.tensor(
                (endpoint_i, endpoint_j), dtype=torch.int16
            )
            sample["condition_bond_pair_endpoint_slot"][token_j, token_i] = torch.tensor(
                (endpoint_j, endpoint_i), dtype=torch.int16
            )
            if order_known:
                sample["condition_bond_type"][token_i, token_j] = bond_type
                sample["condition_bond_type"][token_j, token_i] = bond_type
                sample["condition_bond_type_known_mask"][token_i, token_j] = True
                sample["condition_bond_type_known_mask"][token_j, token_i] = True
    # Native authoring annotations may carry token-level identity/hotspot and
    # optional-center flags even when no atom motif is selected.
    sample["token_identity_condition_mask"] = torch.as_tensor(
        _annotation(atoms, "identity_condition", n_atom, default=False, dtype=bool)[centres],
        dtype=torch.bool,
    )
    sample["hotspot_mask"] = torch.as_tensor(
        _annotation(atoms, "hotspot_condition", n_atom, default=False, dtype=bool)[centres],
        dtype=torch.bool,
    )
    optional_mask = _annotation(
        atoms, "token_center_condition", n_atom, default=False, dtype=bool
    )[centres]
    sample["optional_token_center_mask"] = torch.as_tensor(optional_mask, dtype=torch.bool)
    sample["optional_token_center_coords_nm"][sample["optional_token_center_mask"]] = (
        sample["token_center_coords_nm"][sample["optional_token_center_mask"]]
    )
    sample["num_resolved_atoms"] = torch.tensor(
        int((is_resolved & (local_slots >= 0)).sum()), dtype=torch.long
    )

    modality_specs = {
        MOL_TYPE_TO_ID["protein"]: ("protein_atom37_coords", "protein_atom37_mask", ("N", "CA", "C")),
        MOL_TYPE_TO_ID["rna"]: ("rna_atom27_coords", "rna_atom27_mask", ("C3'", "C1'", "C4'")),
        MOL_TYPE_TO_ID["dna"]: ("dna_atom27_coords", "dna_atom27_mask", ("C3'", "C1'", "C4'")),
    }
    for token_index, modality in enumerate(sample["mol_type_id"].tolist()):
        spec = modality_specs.get(modality)
        if spec is None:
            continue
        coords_name, mask_name, names = spec
        atom_names_mod = PROTEIN_ATOM37 if modality == MOL_TYPE_TO_ID["protein"] else (
            RNA_ATOM27 if modality == MOL_TYPE_TO_ID["rna"] else DNA_ATOM27
        )
        slots = tuple(atom_names_mod.index(name) for name in names)
        if bool(sample[mask_name][token_index, list(slots)].all()):
            frame, ok = _safe_frame(
                sample[coords_name][token_index, slots[0]],
                sample[coords_name][token_index, slots[1]],
                sample[coords_name][token_index, slots[2]],
            )
            if ok:
                sample["token_frame"][token_index] = frame
                sample["token_frame_mask"][token_index] = True

    rows = _bond_rows(atoms)
    kept = _bond_kept_mask(atoms, rows, carrier_mask_atom)
    pair_types: dict[tuple[int, int], int] = {}
    ligand_pair_types: dict[tuple[int, int], int] = {}
    for row, keep in zip(rows, kept):
        atom_i, atom_j, raw_type = map(int, row)
        if local_slots[atom_i] < 0 or local_slots[atom_j] < 0:
            # The canonical target omits unsupported polymer extras from all
            # topology fields while retaining them in provenance.
            continue
        btype = _canonical_bond_type(raw_type)
        if btype == LIGAND_BOND_TYPES["none"]:
            continue
        token_i, token_j = int(owner[atom_i]), int(owner[atom_j])
        if token_i == token_j:
            continue
        pair = (min(token_i, token_j), max(token_i, token_j))
        if bool(keep):
            previous = pair_types.get(pair)
            if previous is not None and previous != btype:
                raise ValueError(f"conflicting token bond types for pair {pair}")
            pair_types[pair] = btype
        if (
            bool(sample["ligand_mask"][token_i])
            and bool(sample["ligand_mask"][token_j])
            and model_mol_token[token_i] == model_mol_token[token_j]
        ):
            previous = ligand_pair_types.get(pair)
            if previous is not None and previous != btype:
                raise ValueError(f"conflicting ligand bond types for pair {pair}")
            ligand_pair_types[pair] = btype
    for (token_i, token_j), btype in pair_types.items():
        sample["token_bond_type"][token_i, token_j] = btype
        sample["token_bond_type"][token_j, token_i] = btype
        sample["token_bond_pair_mask"][token_i, token_j] = True
        sample["token_bond_pair_mask"][token_j, token_i] = True
    for (token_i, token_j), btype in ligand_pair_types.items():
        sample["ligand_bond_type"][token_i, token_j] = btype
        sample["ligand_bond_type"][token_j, token_i] = btype
    same_ligand = (
        (sample["mol_id"][:, None] == sample["mol_id"][None, :])
        & sample["ligand_mask"][:, None]
        & sample["ligand_mask"][None, :]
    )
    sample["ligand_bond_pair_mask"] = same_ligand & ~torch.eye(n_token, dtype=torch.bool)

    sample["num_unsupported_extra_atoms"] = torch.tensor(len(unsupported), dtype=torch.long)
    provenance = _source_mapping(
        atoms, tokens, owner, local_slots, atom91_slots, groups,
        np.flatnonzero(modified_group).tolist(),
    )
    provenance["unsupported_extra_atoms"] = deepcopy(unsupported)
    provenance["chain_break_authority"] = chain_break_authority
    provenance["ligand_formal_charge"] = {
        **charge_provenance,
        "ligand_token_count": int(sample["ligand_mask"].sum().item()),
        "trusted_ligand_token_count": int(trusted_token_charge.sum()),
        "nonzero_ligand_token_count": int((effective_token_charge != 0).sum()),
    }
    carrier_indices = np.flatnonzero(carrier_group).tolist()
    if carrier_indices:
        carrier_atom_id = _annotation(atoms, "modified_polymer_carrier_atom_id", n_atom, default="", dtype="U1").astype(str)
        carrier_source_name = _annotation(atoms, "modified_polymer_carrier_source_atom_name", n_atom, default="", dtype="U1").astype(str)
        carrier_component = _annotation(atoms, "modified_polymer_carrier_component_id", n_atom, default="", dtype="U1").astype(str)
        hetero = _annotation(atoms, "hetero", n_atom, default=False, dtype=bool).astype(bool)
        component_boundaries: dict[str, dict[str, Any]] = {}
        for token_index in carrier_indices:
            atom_index = groups[token_index][0]
            key = str(carrier_key_atom[atom_index])
            boundary = component_boundaries.setdefault(
                key,
                {
                    "carrier_residue_key": key,
                    "model_token_indices": [],
                    "source_atom_indices": [],
                    "component_id": str(carrier_component[atom_index]),
                },
            )
            boundary["model_token_indices"].append(int(token_index))
            boundary["source_atom_indices"].append(
                int(provenance["crop_atom_to_source"][atom_index])
            )

        def endpoint_identity(atom_index: int, token_index: int, slot: int) -> dict[str, Any]:
            return {
                "model_token_index": int(token_index),
                "source_atom_index": int(provenance["crop_atom_to_source"][atom_index]),
                "atom_name": str(atom_name[atom_index]),
                "ccd_atom_id": str(carrier_atom_id[atom_index]),
                "source_atom_name": str(carrier_source_name[atom_index]),
                "source_res_name": str(raw_res[atom_index]),
                "source_res_id": int(residue_index_atom[atom_index]),
                "source_chain_id": str(chain[atom_index]),
                "source_mol_type": str(mol[atom_index]),
                "source_mol_id": int(source_mol[atom_index]),
                "source_entity_mol_id": int(entity_mol[atom_index]),
                "model_mol_id": int(model_mol_token[token_index]),
                "carrier_residue_key": str(carrier_key_atom[atom_index]),
                "component_id": str(carrier_component[atom_index]),
                "endpoint_slot": int(slot),
            }

        model_token_to_source = []
        for token_index in carrier_indices:
            atom_index = groups[token_index][0]
            model_token_to_source.append({
                "model_token_index": int(token_index),
                "source_atom_index": int(provenance["crop_atom_to_source"][atom_index]),
                "atom_name": str(atom_name[atom_index]),
                "ccd_atom_id": str(carrier_atom_id[atom_index]),
                "source_atom_name": str(carrier_source_name[atom_index]),
                "element": str(carrier_element[atom_index]),
                "formal_charge": int(carrier_charge[atom_index]),
                "res_name": str(raw_res[atom_index]),
                "res_id": int(residue_index_atom[atom_index]),
                "chain_id": str(chain[atom_index]),
                "entity_id": int(entity[atom_index]),
                "asym_id": int(asym[atom_index]),
                "mol_type": str(mol[atom_index]),
                "hetero": bool(hetero[atom_index]),
                "carrier_residue_key": str(carrier_key_atom[atom_index]),
                "source_mol_id": int(source_mol[atom_index]),
                "source_entity_mol_id": int(entity_mol[atom_index]),
                "model_mol_id": int(model_mol_token[token_index]),
                "component_id": str(carrier_component[atom_index]),
                "component_boundary": deepcopy(
                    component_boundaries[str(carrier_key_atom[atom_index])]
                ),
            })

        hard_bonds: list[dict[str, Any]] = []
        for atom_i, atom_j, raw_type in rows:
            atom_i, atom_j, raw_type = int(atom_i), int(atom_j), int(raw_type)
            if not (carrier_mask_atom[atom_i] or carrier_mask_atom[atom_j]):
                continue
            known, order_known, bond_type = _condition_bond_state(raw_type)
            if not known:
                continue
            token_i, token_j = int(owner[atom_i]), int(owner[atom_j])
            if token_i == token_j:
                continue
            slot_i = (
                ATOM91_SIZE
                if model_group_mol[token_i] == "ligand"
                else int(atom91_slots[atom_i])
            )
            slot_j = (
                ATOM91_SIZE
                if model_group_mol[token_j] == "ligand"
                else int(atom91_slots[atom_j])
            )
            if slot_i < 0 or slot_j < 0:
                raise ValueError(
                    "carrier hard bond endpoint is not represented in the canonical ABI"
                )
            if token_i > token_j:
                token_i, token_j = token_j, token_i
                atom_i, atom_j = atom_j, atom_i
                slot_i, slot_j = slot_j, slot_i
            hard_bonds.append({
                "token_i": token_i,
                "token_j": token_j,
                "bond_type": int(bond_type),
                "bond_type_known": bool(order_known),
                "bond_exists": True,
                "raw_bond_type": raw_type,
                "endpoint_i_slot": slot_i,
                "endpoint_j_slot": slot_j,
                "source_endpoint_i": endpoint_identity(atom_i, token_i, slot_i),
                "source_endpoint_j": endpoint_identity(atom_j, token_j, slot_j),
            })
        unique_hard_bonds: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in hard_bonds:
            key = (
                row["token_i"], row["token_j"],
                row["endpoint_i_slot"], row["endpoint_j_slot"],
                row["bond_type"], row["bond_type_known"],
            )
            unique_hard_bonds[key] = row
        provenance["modified_polymer_carriers"] = {
            "version": 1,
            "model_token_to_source": model_token_to_source,
            "hard_bonds": [
                unique_hard_bonds[key] for key in sorted(unique_hard_bonds)
            ],
            "expected_carrier_token_indices": [int(value) for value in carrier_indices],
            "expected_carrier_token_count": len(carrier_indices),
        }
    sample["provenance"] = provenance
    validate_canonical_sample(sample)
    return sample


__all__ = ["build_canonical"]
