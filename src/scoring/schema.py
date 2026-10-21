"""Frozen canonical input schema used by standalone PartiScore inference.

The training canonical schema is intentionally not imported here.  The scorer
needs the same checkpoint-facing fields and defaults, but its authoring path
must remain independent of samplers, transforms, and graph containers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import torch

from src.common.constants import (
    ATOM91_SIZE,
    DNA_ATOM27,
    DNA_RES_NAMES,
    ELEMENTS_128,
    LIGAND_BOND_TYPES,
    LIGAND_FORMAL_CHARGES,
    MOL_TYPE_TO_ID,
    PROTEIN_ATOM37,
    PROTEIN_RES_NAMES,
    RNA_ATOM27,
    RNA_RES_NAMES,
    TOKEN_TYPE32,
    TOKEN_TYPE_IGNORE_INDEX,
    element_id,
)


class FieldScope(str, Enum):
    TOKEN = "token"
    PAIR = "pair"
    GLOBAL = "global"
    STRING = "string"
    PROVENANCE = "provenance"


class PairSymmetry(str, Enum):
    NONE = "none"
    MATRIX = "matrix"
    ENDPOINT_SWAP = "endpoint_swap"


@dataclass(frozen=True)
class CanonicalFieldSpec:
    name: str
    scope: FieldScope
    dtype: torch.dtype | None = None
    trailing_shape: tuple[int, ...] = ()
    pad_value: int | float | bool = 0
    required: bool = True
    value_mask: str | None = None
    symmetry: PairSymmetry = PairSymmetry.NONE
    zero_diagonal: bool = False


def _field(
    name: str,
    scope: FieldScope,
    dtype: torch.dtype | None = None,
    trailing_shape: tuple[int, ...] = (),
    pad_value: int | float | bool = 0,
    *,
    required: bool = True,
    value_mask: str | None = None,
    symmetry: PairSymmetry = PairSymmetry.NONE,
    zero_diagonal: bool = False,
) -> CanonicalFieldSpec:
    return CanonicalFieldSpec(
        name=name,
        scope=scope,
        dtype=dtype,
        trailing_shape=trailing_shape,
        pad_value=pad_value,
        required=bool(required),
        value_mask=value_mask,
        symmetry=symmetry,
        zero_diagonal=zero_diagonal,
    )


# Keep this list byte-for-byte aligned with the checkpoint input ABI.  It is
# deliberately written out rather than derived from optional training fields.
_TOKEN_FIELDS = (
    _field("token_mask", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("design_token_mask", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("flow_denoising_mask", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("token_center_resolved_mask", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("mol_type_id", FieldScope.TOKEN, torch.long, pad_value=MOL_TYPE_TO_ID["protein"]),
    _field("token_type32", FieldScope.TOKEN, torch.long, pad_value=TOKEN_TYPE_IGNORE_INDEX),
    _field("residue_index", FieldScope.TOKEN, torch.long),
    _field("asym_id", FieldScope.TOKEN, torch.long),
    _field("entity_id", FieldScope.TOKEN, torch.long),
    _field("mol_id", FieldScope.TOKEN, torch.long),
    _field("source_mol_id", FieldScope.TOKEN, torch.long, pad_value=-1, required=False),
    _field("entity_mol_id", FieldScope.TOKEN, torch.long),
    _field("chain_break_after_token", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("token_center_coords_nm", FieldScope.TOKEN, torch.float32, (3,), 0.0),
    _field(
        "protein_atom37_coords", FieldScope.TOKEN, torch.float32,
        (len(PROTEIN_ATOM37), 3), 0.0, value_mask="protein_atom37_mask",
    ),
    _field("protein_atom37_mask", FieldScope.TOKEN, torch.bool, (len(PROTEIN_ATOM37),), False),
    _field(
        "rna_atom27_coords", FieldScope.TOKEN, torch.float32,
        (len(RNA_ATOM27), 3), 0.0, value_mask="rna_atom27_mask",
    ),
    _field("rna_atom27_mask", FieldScope.TOKEN, torch.bool, (len(RNA_ATOM27),), False),
    _field(
        "dna_atom27_coords", FieldScope.TOKEN, torch.float32,
        (len(DNA_ATOM27), 3), 0.0, value_mask="dna_atom27_mask",
    ),
    _field("dna_atom27_mask", FieldScope.TOKEN, torch.bool, (len(DNA_ATOM27),), False),
    _field(
        "polymer_atom91_coords", FieldScope.TOKEN, torch.float32,
        (ATOM91_SIZE, 3), 0.0, value_mask="polymer_atom91_mask",
    ),
    _field("polymer_atom91_mask", FieldScope.TOKEN, torch.bool, (ATOM91_SIZE,), False),
    _field("token_frame", FieldScope.TOKEN, torch.float32, (3, 3), 0.0),
    _field("token_frame_mask", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("ligand_mask", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("modified_polymer_token_mask", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("ligand_element_type", FieldScope.TOKEN, torch.long, pad_value=element_id("UNK_ELEM_128")),
    _field("ligand_charge", FieldScope.TOKEN, torch.float32, pad_value=0.0),
    _field(
        "ligand_formal_charge_class", FieldScope.TOKEN, torch.long,
        pad_value=0, value_mask="ligand_formal_charge_mask",
    ),
    _field("ligand_formal_charge_mask", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("is_condition_token", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("is_condition_entity_token", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("token_identity_condition_mask", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field(
        "optional_token_center_coords_nm", FieldScope.TOKEN, torch.float32,
        (3,), 0.0, value_mask="optional_token_center_mask",
    ),
    _field("optional_token_center_mask", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("hotspot_mask", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("polymer_ss_token_gt", FieldScope.TOKEN, torch.long, pad_value=0, value_mask="polymer_ss_token_gt_mask"),
    _field("polymer_ss_token_gt_mask", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("polymer_ss_token", FieldScope.TOKEN, torch.long, pad_value=0, value_mask="polymer_ss_token_condition_mask"),
    _field("polymer_ss_token_condition_mask", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("polymer_coarse_ss", FieldScope.TOKEN, torch.long, pad_value=0),
    _field(
        "protein_motif_atom37_coords_nm", FieldScope.TOKEN, torch.float32,
        (len(PROTEIN_ATOM37), 3), 0.0, value_mask="protein_motif_atom37_mask",
    ),
    _field("protein_motif_atom37_mask", FieldScope.TOKEN, torch.bool, (len(PROTEIN_ATOM37),), False),
    _field(
        "rna_motif_atom27_coords_nm", FieldScope.TOKEN, torch.float32,
        (len(RNA_ATOM27), 3), 0.0, value_mask="rna_motif_atom27_mask",
    ),
    _field("rna_motif_atom27_mask", FieldScope.TOKEN, torch.bool, (len(RNA_ATOM27),), False),
    _field(
        "dna_motif_atom27_coords_nm", FieldScope.TOKEN, torch.float32,
        (len(DNA_ATOM27), 3), 0.0, value_mask="dna_motif_atom27_mask",
    ),
    _field("dna_motif_atom27_mask", FieldScope.TOKEN, torch.bool, (len(DNA_ATOM27),), False),
    _field(
        "ligand_motif_token_center_coords_nm", FieldScope.TOKEN, torch.float32,
        (3,), 0.0, value_mask="ligand_motif_mask",
    ),
    _field("ligand_motif_mask", FieldScope.TOKEN, torch.bool, pad_value=False),
    _field("debug_role_id", FieldScope.TOKEN, torch.long, pad_value=0),
)


def _pair(
    name: str,
    dtype: torch.dtype,
    trailing_shape: tuple[int, ...] = (),
    pad_value: int | float | bool = 0,
    *,
    value_mask: str | None = None,
    symmetry: PairSymmetry = PairSymmetry.MATRIX,
) -> CanonicalFieldSpec:
    return _field(
        name, FieldScope.PAIR, dtype, trailing_shape, pad_value,
        value_mask=value_mask, symmetry=symmetry, zero_diagonal=True,
    )


_PAIR_FIELDS = (
    _pair("ligand_bond_type", torch.long, pad_value=LIGAND_BOND_TYPES["none"], value_mask="ligand_bond_pair_mask"),
    _pair("ligand_bond_pair_mask", torch.bool, pad_value=False),
    _pair("token_bond_type", torch.long, pad_value=LIGAND_BOND_TYPES["none"], value_mask="token_bond_pair_mask"),
    _pair("token_bond_pair_mask", torch.bool, pad_value=False),
    _pair("polymer_ss_pair_gt", torch.long, pad_value=0, value_mask="polymer_ss_pair_gt_mask"),
    _pair("polymer_ss_pair_gt_mask", torch.bool, pad_value=False),
    _pair("polymer_ss_pair", torch.long, pad_value=0, value_mask="polymer_ss_pair_condition_mask"),
    _pair("polymer_ss_pair_condition_mask", torch.bool, pad_value=False),
    _pair("condition_bond_pair_mask", torch.bool, pad_value=False),
    _field(
        "condition_bond_pair_endpoint_slot", FieldScope.PAIR, torch.int16,
        (2,), -1, symmetry=PairSymmetry.ENDPOINT_SWAP,
    ),
    _pair("condition_bond_type", torch.long, pad_value=LIGAND_BOND_TYPES["none"], value_mask="condition_bond_type_known_mask"),
    _pair("condition_bond_type_known_mask", torch.bool, pad_value=False),
)


_GLOBAL_FIELDS = (
    _field("sample_loss_weight", FieldScope.GLOBAL, torch.float32, (), 1.0),
    _field("row_index", FieldScope.GLOBAL, torch.long, (), -1),
    _field("crop_attempt", FieldScope.GLOBAL, torch.long, (), 0),
    _field("condition_token_fraction", FieldScope.GLOBAL, torch.float32, (), 0.0),
    _field("condition_atom_fraction", FieldScope.GLOBAL, torch.float32, (), 0.0),
    _field("num_canonical_tokens", FieldScope.GLOBAL, torch.long, (), 0),
    _field("num_resolved_atoms", FieldScope.GLOBAL, torch.long, (), 0),
    _field("num_unsupported_extra_atoms", FieldScope.GLOBAL, torch.long, (), 0),
    _field("center_origin_nm", FieldScope.GLOBAL, torch.float32, (3,), 0.0),
    _field("post_center_translation_nm", FieldScope.GLOBAL, torch.float32, (3,), 0.0),
    _field("global_rotation", FieldScope.GLOBAL, torch.float32, (3, 3), 0.0),
    _field("source_to_augmented_affine", FieldScope.GLOBAL, torch.float32, (4, 4), 0.0),
)


_STRING_FIELD_NAMES = (
    "pdb_id", "assembly_id", "split", "recipe_id", "task_family",
    "crop_method", "auxiliary_mode", "center_source", "global_rotation_convention",
)
_STRING_FIELDS = tuple(_field(name, FieldScope.STRING) for name in _STRING_FIELD_NAMES)
_PROVENANCE_FIELDS = (_field("provenance", FieldScope.PROVENANCE),)

_ALL_FIELD_SPECS = _TOKEN_FIELDS + _PAIR_FIELDS + _GLOBAL_FIELDS + _STRING_FIELDS + _PROVENANCE_FIELDS
CANONICAL_FIELD_SPECS = MappingProxyType({spec.name: spec for spec in _ALL_FIELD_SPECS})
assert len(_TOKEN_FIELDS) == 50
assert len(_PAIR_FIELDS) == 12
assert len(_GLOBAL_FIELDS) == 12
assert len(_STRING_FIELDS) == 9
assert len(CANONICAL_FIELD_SPECS) == 84

_DEFAULT_STRINGS = {
    "pdb_id": "unknown",
    "assembly_id": "unknown_source",
    "split": "unknown_source",
    "recipe_id": "not_applicable",
    "task_family": "not_applicable",
    "crop_method": "unknown",
    "auxiliary_mode": "not_applicable",
    "center_source": "not_applied",
    "global_rotation_convention": "row_vector_x_at_R",
}


def empty_canonical_sample(num_tokens: int) -> dict[str, Any]:
    """Create the complete unbatched 84-field canonical sample."""

    n = int(num_tokens)
    if n <= 0:
        raise ValueError("canonical samples require at least one token")
    sample: dict[str, Any] = {}
    for spec in _ALL_FIELD_SPECS:
        if spec.scope is FieldScope.TOKEN:
            shape = (n,) + spec.trailing_shape
            sample[spec.name] = torch.full(shape, spec.pad_value, dtype=spec.dtype)
        elif spec.scope is FieldScope.PAIR:
            shape = (n, n) + spec.trailing_shape
            sample[spec.name] = torch.full(shape, spec.pad_value, dtype=spec.dtype)
        elif spec.scope is FieldScope.GLOBAL:
            sample[spec.name] = torch.full(spec.trailing_shape, spec.pad_value, dtype=spec.dtype)
        elif spec.scope is FieldScope.STRING:
            sample[spec.name] = _DEFAULT_STRINGS[spec.name]
        else:
            sample[spec.name] = {}
    sample["token_mask"].fill_(True)
    sample["token_type32"].fill_(TOKEN_TYPE32["UNK"])
    sample["token_frame"] = torch.eye(3, dtype=torch.float32).repeat(n, 1, 1)
    sample["global_rotation"] = torch.eye(3, dtype=torch.float32)
    sample["source_to_augmented_affine"] = torch.eye(4, dtype=torch.float32)
    sample["sample_loss_weight"] = torch.tensor(1.0, dtype=torch.float32)
    sample["num_canonical_tokens"] = torch.tensor(n, dtype=torch.long)
    sample["provenance"] = {
        "coordinate_units": "nm",
        "source_mapping_status": "synthetic_identity",
        "crop_atom_to_source": list(range(n)),
        "source_atom_to_crop": list(range(n)),
        "crop_atom_to_canonical_token": list(range(n)),
        "crop_atom_to_modality_slot": [0] * n,
        "crop_atom_to_atom91_slot": [-1] * n,
        "crop_token_to_source": list(range(n)),
        "source_token_to_crop": list(range(n)),
        "crop_token_to_canonical": list(range(n)),
        "canonical_to_crop_tokens": [[i] for i in range(n)],
        "modified_canonical_token_indices": [],
        "unsupported_extra_atoms": [],
        "selected_condition_atom_indices": [],
        "require_resolved_token_centers": False,
    }
    return sample


def _assert_zero_where_hidden(value: torch.Tensor, mask: torch.Tensor, name: str) -> None:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value)
    if value.dtype == torch.bool:
        ok = ~value[~expanded]
    else:
        ok = value[~expanded] == 0
    if not bool(ok.all().item()):
        raise ValueError(f"{name} must be zero where its value mask is false")


def validate_canonical_sample(sample: Mapping[str, Any], *, exact_keys: bool = True) -> None:
    """Validate the frozen unbatched scorer sample at the model boundary."""

    if not isinstance(sample, Mapping):
        raise TypeError("canonical sample must be a mapping")
    expected = set(CANONICAL_FIELD_SPECS)
    actual = set(sample)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"canonical sample missing fields: {sorted(missing)}")
    if exact_keys and extra:
        raise ValueError(f"canonical sample has unregistered fields: {sorted(extra)}")
    token_mask = sample["token_mask"]
    if not isinstance(token_mask, torch.Tensor) or token_mask.ndim != 1 or not token_mask.numel():
        raise ValueError("token_mask must be a non-empty [N] tensor")
    n = int(token_mask.shape[0])
    if not bool(token_mask.all().item()):
        raise ValueError("unbatched canonical samples may not contain padding")
    for spec in _ALL_FIELD_SPECS:
        value = sample[spec.name]
        if spec.scope is FieldScope.STRING:
            if not isinstance(value, str):
                raise TypeError(f"{spec.name} must be a string")
            continue
        if spec.scope is FieldScope.PROVENANCE:
            if not isinstance(value, Mapping):
                raise TypeError("provenance must be a mapping")
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{spec.name} must be a tensor")
        if value.device.type != "cpu" or value.dtype != spec.dtype:
            raise ValueError(f"{spec.name} must be CPU {spec.dtype}, got {value.device}/{value.dtype}")
        shape = ((n,) + spec.trailing_shape if spec.scope is FieldScope.TOKEN else
                 (n, n) + spec.trailing_shape if spec.scope is FieldScope.PAIR else
                 spec.trailing_shape)
        if tuple(value.shape) != shape:
            raise ValueError(f"{spec.name} shape {tuple(value.shape)} != {shape}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{spec.name} contains non-finite values")
        if spec.value_mask is not None:
            _assert_zero_where_hidden(value, sample[spec.value_mask], spec.name)
        if spec.scope is FieldScope.PAIR:
            if spec.symmetry is PairSymmetry.MATRIX and not torch.equal(value, value.transpose(0, 1)):
                raise ValueError(f"{spec.name} is not symmetric")
            if spec.symmetry is PairSymmetry.ENDPOINT_SWAP:
                if not torch.equal(value, value.transpose(0, 1).flip(-1)):
                    raise ValueError(f"{spec.name} does not swap endpoint order")
            if spec.zero_diagonal:
                diagonal = value[torch.arange(n), torch.arange(n)]
                if not bool((diagonal == 0).all().item()):
                    raise ValueError(f"{spec.name} diagonal must be zero")
    mol_type = sample["mol_type_id"]
    if not bool(((mol_type >= 0) & (mol_type < len(MOL_TYPE_TO_ID))).all().item()):
        raise ValueError("mol_type_id contains an unknown modality")
    if not torch.equal(sample["ligand_mask"], mol_type == MOL_TYPE_TO_ID["ligand"]):
        raise ValueError("ligand_mask must agree with mol_type_id")
    if not bool((sample["token_center_resolved_mask"] <= sample["token_mask"]).all().item()):
        raise ValueError("token_center_resolved_mask exceeds token_mask")
    if not torch.equal(
        sample["polymer_atom91_mask"],
        torch.cat((sample["protein_atom37_mask"], sample["rna_atom27_mask"], sample["dna_atom27_mask"]), dim=-1),
    ):
        raise ValueError("polymer_atom91_mask must concatenate modality masks")
    if not torch.equal(
        sample["polymer_atom91_coords"],
        torch.cat((sample["protein_atom37_coords"], sample["rna_atom27_coords"], sample["dna_atom27_coords"]), dim=-2),
    ):
        raise ValueError("polymer_atom91_coords must concatenate modality coordinates")
    if not bool((sample["ligand_formal_charge_mask"] <= sample["ligand_mask"]).all().item()):
        raise ValueError("ligand formal-charge mask exceeds ligand mask")
    if not bool(((sample["ligand_element_type"] >= 0) & (sample["ligand_element_type"] < len(ELEMENTS_128))).all().item()):
        raise ValueError("ligand_element_type contains an unknown element")
    protein_token = mol_type == MOL_TYPE_TO_ID["protein"]
    rna_token = mol_type == MOL_TYPE_TO_ID["rna"]
    dna_token = mol_type == MOL_TYPE_TO_ID["dna"]
    ligand_token = mol_type == MOL_TYPE_TO_ID["ligand"]
    allowed_token_ids = {
        MOL_TYPE_TO_ID["protein"]: torch.as_tensor(
            [TOKEN_TYPE32[name] for name in PROTEIN_RES_NAMES], dtype=torch.long
        ),
        MOL_TYPE_TO_ID["rna"]: torch.as_tensor(
            [TOKEN_TYPE32[name] for name in RNA_RES_NAMES], dtype=torch.long
        ),
        MOL_TYPE_TO_ID["dna"]: torch.as_tensor(
            [TOKEN_TYPE32[name] for name in DNA_RES_NAMES], dtype=torch.long
        ),
    }
    for modality_id, allowed in allowed_token_ids.items():
        if not bool(torch.isin(sample["token_type32"][mol_type == modality_id], allowed).all().item()):
            raise ValueError(f"token_type32 contains an invalid id for mol_type_id={modality_id}")
    if not bool((sample["token_type32"][ligand_token] == TOKEN_TYPE_IGNORE_INDEX).all().item()):
        raise ValueError("ligand tokens must use TOKEN_TYPE_IGNORE_INDEX")
    if not bool((sample["ligand_formal_charge_mask"] <= ligand_token).all().item()):
        raise ValueError("ligand formal-charge mask exceeds ligand mask")
    charge_class = sample["ligand_formal_charge_class"]
    known_charge = sample["ligand_formal_charge_mask"]
    if not bool(
        ((charge_class[known_charge] >= 0)
         & (charge_class[known_charge] < len(LIGAND_FORMAL_CHARGES))).all().item()
    ):
        raise ValueError("ligand formal-charge class is outside the frozen vocabulary")
    if bool(known_charge.any().item()):
        expected_charge = torch.as_tensor(
            LIGAND_FORMAL_CHARGES, dtype=sample["ligand_charge"].dtype
        )[charge_class[known_charge]]
        if not torch.equal(sample["ligand_charge"][known_charge], expected_charge):
            raise ValueError("ligand_charge disagrees with ligand_formal_charge_class")
    if not bool((sample["ligand_charge"][~ligand_token] == 0).all().item()):
        raise ValueError("ligand_charge must be zero for polymer tokens")
    if bool((sample["protein_atom37_mask"] & ~protein_token[:, None]).any().item()):
        raise ValueError("protein atom masks are only valid on protein tokens")
    if bool((sample["rna_atom27_mask"] & ~rna_token[:, None]).any().item()):
        raise ValueError("RNA atom masks are only valid on RNA tokens")
    if bool((sample["dna_atom27_mask"] & ~dna_token[:, None]).any().item()):
        raise ValueError("DNA atom masks are only valid on DNA tokens")
    for mask_name, parent_name in (
        ("protein_motif_atom37_mask", "protein_atom37_mask"),
        ("rna_motif_atom27_mask", "rna_atom27_mask"),
        ("dna_motif_atom27_mask", "dna_atom27_mask"),
        ("ligand_motif_mask", "ligand_mask"),
    ):
        if bool((sample[mask_name] & ~sample[parent_name]).any().item()):
            raise ValueError(f"{mask_name} must be a subset of {parent_name}")
    if bool(
        (sample["design_token_mask"] & ~sample["token_mask"]).any().item()
        or (sample["flow_denoising_mask"] & ~sample["token_mask"]).any().item()
        or (sample["modified_polymer_token_mask"] & ~sample["token_mask"]).any().item()
        or (sample["is_condition_token"] & ~sample["token_mask"]).any().item()
        or (sample["is_condition_entity_token"] & ~sample["token_mask"]).any().item()
        or (sample["token_identity_condition_mask"] & ~sample["token_mask"]).any().item()
        or (sample["optional_token_center_mask"] & ~sample["token_mask"]).any().item()
        or (sample["hotspot_mask"] & ~sample["token_mask"]).any().item()
    ):
        raise ValueError("token-side masks must be subsets of token_mask")
    pair_valid = sample["token_mask"][:, None] & sample["token_mask"][None, :]
    if bool((sample["ligand_bond_pair_mask"] & ~pair_valid).any().item()):
        raise ValueError("ligand_bond_pair_mask contains an invalid token")
    if int(sample["num_canonical_tokens"].item()) != n:
        raise ValueError("num_canonical_tokens disagrees with token fields")
    provenance = sample["provenance"]
    if provenance.get("coordinate_units") != "nm":
        raise ValueError("canonical coordinates must be tagged as nm")
    modified = provenance.get("modified_canonical_token_indices", ())
    if not isinstance(modified, (list, tuple)) or len(set(int(x) for x in modified)) != len(modified):
        raise ValueError("modified_canonical_token_indices must be unique integer indices")
    if set(int(x) for x in modified) != set(torch.where(sample["modified_polymer_token_mask"])[0].tolist()):
        raise ValueError("modified canonical provenance disagrees with token mask")
    unsupported = provenance.get("unsupported_extra_atoms", ())
    if int(sample["num_unsupported_extra_atoms"].item()) != len(unsupported):
        raise ValueError("num_unsupported_extra_atoms disagrees with provenance")


__all__ = [
    "CANONICAL_FIELD_SPECS", "CanonicalFieldSpec", "FieldScope", "PairSymmetry",
    "empty_canonical_sample", "validate_canonical_sample",
]
