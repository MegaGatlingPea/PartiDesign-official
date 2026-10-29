from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import biotite.structure as struc
import numpy as np
import torch

from ..utils.atoms import (
    DNA_ATOM27,
    DNA_RES_ATOMS,
    ELEMENT_TO_ID,
    LIGAND_BOND_TYPES,
    MOL_TYPE_TO_ID,
    PROTEIN_ATOM37,
    PROTEIN_RES_ATOMS,
    RNA_ATOM27,
    RNA_RES_ATOMS,
    TOKEN_TYPE32,
    TOKEN_TYPE_IGNORE_INDEX,
    UNKNOWN_ELEMENT_ID,
    validate_atom_array,
)
from .model_tokens import ModelTokenLayout, build_model_token_layout
from .compile_policy import token_bucket_size
from .records import CompiledSample, DesignSampleError, ExpandedRequest
from .modified_polymer import proxy_atom_rows, proxy_inventory
from ..model.constants import normalise_atom_name


ATOM91_SIZE = len(PROTEIN_ATOM37) + len(RNA_ATOM27) + len(DNA_ATOM27)
LIGAND_ENDPOINT_SLOT = ATOM91_SIZE


class FieldScope(str, Enum):
    TOKEN = "token"
    PAIR = "pair"


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
    pad_value: float | int | bool = 0
    value_mask: str | None = None
    symmetry: PairSymmetry = PairSymmetry.NONE
    zero_diagonal: bool = False


def _field(
    name: str,
    scope: FieldScope,
    dtype: torch.dtype | None = None,
    trailing_shape: tuple[int, ...] = (),
    pad_value: float | int | bool = 0,
    *,
    value_mask: str | None = None,
    symmetry: PairSymmetry = PairSymmetry.NONE,
    zero_diagonal: bool = False,
) -> CanonicalFieldSpec:
    return CanonicalFieldSpec(name, scope, dtype, trailing_shape, pad_value, value_mask, symmetry, zero_diagonal)


def _token(name: str, dtype: torch.dtype, shape: tuple[int, ...] = (), pad: float | int | bool = 0, *, mask: str | None = None) -> CanonicalFieldSpec:
    return _field(name, FieldScope.TOKEN, dtype, shape, pad, value_mask=mask)


def _pair(name: str, dtype: torch.dtype, shape: tuple[int, ...] = (), pad: float | int | bool = 0, *, mask: str | None = None, symmetry: PairSymmetry = PairSymmetry.MATRIX) -> CanonicalFieldSpec:
    return _field(name, FieldScope.PAIR, dtype, shape, pad, value_mask=mask, symmetry=symmetry, zero_diagonal=True)


_TOKEN_FIELDS = (
    _token("token_mask", torch.bool, pad=False),
    _token("design_token_mask", torch.bool, pad=False),
    _token("mol_type_id", torch.long, pad=MOL_TYPE_TO_ID["protein"]),
    _token("token_type32", torch.long, pad=TOKEN_TYPE_IGNORE_INDEX),
    _token("residue_index", torch.long),
    _token("asym_id", torch.long),
    _token("mol_id", torch.long),
    _token("chain_break_after_token", torch.bool, pad=False),
    _token("token_center_coords_nm", torch.float32, (3,), 0.0),
    _token("protein_atom37_coords", torch.float32, (37, 3), 0.0, mask="protein_atom37_mask"),
    _token("protein_atom37_mask", torch.bool, (37,), False),
    _token("rna_atom27_coords", torch.float32, (27, 3), 0.0, mask="rna_atom27_mask"),
    _token("rna_atom27_mask", torch.bool, (27,), False),
    _token("dna_atom27_coords", torch.float32, (27, 3), 0.0, mask="dna_atom27_mask"),
    _token("dna_atom27_mask", torch.bool, (27,), False),
    _token("token_frame", torch.float32, (3, 3), 0.0),
    _token("token_frame_mask", torch.bool, pad=False),
    _token("ligand_mask", torch.bool, pad=False),
    _token("ligand_element_type", torch.long, pad=UNKNOWN_ELEMENT_ID),
    _token("ligand_charge", torch.float32, pad=0.0),
    _token("token_identity_condition_mask", torch.bool, pad=False),
    _token("optional_token_center_coords_nm", torch.float32, (3,), 0.0, mask="optional_token_center_mask"),
    _token("optional_token_center_mask", torch.bool, pad=False),
    _token("hotspot_mask", torch.bool, pad=False),
    _token("polymer_ss_token", torch.long, pad=0, mask="polymer_ss_token_condition_mask"),
    _token("polymer_ss_token_condition_mask", torch.bool, pad=False),
    _token("polymer_coarse_ss", torch.long, pad=0),
    _token("protein_motif_atom37_coords_nm", torch.float32, (37, 3), 0.0, mask="protein_motif_atom37_mask"),
    _token("protein_motif_atom37_mask", torch.bool, (37,), False),
    _token("rna_motif_atom27_coords_nm", torch.float32, (27, 3), 0.0, mask="rna_motif_atom27_mask"),
    _token("rna_motif_atom27_mask", torch.bool, (27,), False),
    _token("dna_motif_atom27_coords_nm", torch.float32, (27, 3), 0.0, mask="dna_motif_atom27_mask"),
    _token("dna_motif_atom27_mask", torch.bool, (27,), False),
    _token("ligand_motif_token_center_coords_nm", torch.float32, (3,), 0.0, mask="ligand_motif_mask"),
    _token("ligand_motif_mask", torch.bool, pad=False),
)
_PAIR_FIELDS = (
    _pair("token_bond_type", torch.long, pad=0, mask="token_bond_pair_mask"),
    _pair("token_bond_pair_mask", torch.bool, pad=False),
    _pair("polymer_ss_pair", torch.long, pad=0, mask="polymer_ss_pair_condition_mask"),
    _pair("polymer_ss_pair_condition_mask", torch.bool, pad=False),
    _pair("condition_bond_pair_mask", torch.bool, pad=False),
    _field(
        "condition_bond_pair_endpoint_slot",
        FieldScope.PAIR,
        torch.int16,
        (2,),
        -1,
        symmetry=PairSymmetry.ENDPOINT_SWAP,
        zero_diagonal=False,
    ),
    _pair("condition_bond_type", torch.long, pad=0, mask="condition_bond_type_known_mask"),
    _pair("condition_bond_type_known_mask", torch.bool, pad=False),
)
_ALL_FIELDS = _TOKEN_FIELDS + _PAIR_FIELDS
CANONICAL_FIELD_SPECS = MappingProxyType({spec.name: spec for spec in _ALL_FIELDS})
if len(CANONICAL_FIELD_SPECS) != 43:
    raise AssertionError("standalone canonical field registry is incomplete or duplicated")


_TEMPLATE_FIELDS = (
    _token("design_entity_id", torch.long, pad=-1),
    _token("template_group_id", torch.long),
    _token("template_protein_atom37_coords_nm", torch.float32, (37, 3), 0.0, mask="template_protein_atom37_mask"),
    _token("template_protein_atom37_mask", torch.bool, (37,), False),
    _token("antibody_chain_role_id", torch.long),
    _token("antibody_region_id", torch.long),
)
TEMPLATE_FIELD_SPECS = MappingProxyType({spec.name: spec for spec in _TEMPLATE_FIELDS})


def _sample_field_specs(sample: Mapping[str, Any]) -> Mapping[str, CanonicalFieldSpec]:
    present = set(sample) & set(TEMPLATE_FIELD_SPECS)
    if present and present != set(TEMPLATE_FIELD_SPECS):
        raise ValueError("antibody template fields must be supplied as one complete six-field group")
    return {**CANONICAL_FIELD_SPECS, **TEMPLATE_FIELD_SPECS} if present else CANONICAL_FIELD_SPECS


def empty_canonical_sample(num_tokens: int) -> dict[str, Any]:
    n = int(num_tokens)
    if n <= 0:
        raise ValueError("canonical inference sample requires at least one token")
    sample: dict[str, Any] = {}
    for spec in _ALL_FIELDS:
        if spec.scope == FieldScope.TOKEN:
            sample[spec.name] = torch.full((n,) + spec.trailing_shape, spec.pad_value, dtype=spec.dtype)
        elif spec.scope == FieldScope.PAIR:
            sample[spec.name] = torch.full((n, n) + spec.trailing_shape, spec.pad_value, dtype=spec.dtype)
        else:
            raise AssertionError(f"standalone runtime field has unsupported scope: {spec}")
    sample["token_mask"].fill_(True)
    sample["token_frame"] = torch.eye(3, dtype=torch.float32).repeat(n, 1, 1)
    return sample


def _model_coordinates(compiled: CompiledSample) -> np.ndarray:
    atoms = compiled.atoms
    result = np.zeros((len(atoms), 3), dtype=np.float32)
    valid = np.asarray(atoms.coord_valid, dtype=bool)
    if valid.any():
        coords = np.asarray(atoms.coord[valid], dtype=np.float64) / 10.0
        affine = compiled.source_to_model_affine
        transformed = coords @ affine[:3, :3] + affine[3, :3]
        result[valid] = transformed.astype(np.float32)
    return result


def _bond_type(raw: int) -> int:
    raw = int(raw)
    if raw in {0, 1}:
        return LIGAND_BOND_TYPES["single"]
    if raw == 2:
        return LIGAND_BOND_TYPES["double"]
    if raw in {3, 4}:
        return LIGAND_BOND_TYPES["triple"]
    if raw in {5, 6, 7, 9}:
        return LIGAND_BOND_TYPES["aromatic"]
    if raw == 8:
        return LIGAND_BOND_TYPES["none"]
    raise ValueError(f"unsupported Biotite bond type: {raw}")


def _carrier_bond_condition_type(raw: int) -> tuple[bool, int]:
    """Map source carrier bonds without the lossy target-bond coercions."""

    raw = int(raw)
    if raw == int(struc.BondType.ANY):
        return False, LIGAND_BOND_TYPES["none"]
    if raw == int(struc.BondType.SINGLE):
        return True, LIGAND_BOND_TYPES["single"]
    if raw == int(struc.BondType.DOUBLE):
        return True, LIGAND_BOND_TYPES["double"]
    if raw == int(struc.BondType.TRIPLE):
        return True, LIGAND_BOND_TYPES["triple"]
    if raw == int(struc.BondType.QUADRUPLE):
        raise ValueError("modified-polymer carrier bonds cannot use QUADRUPLE order")
    if raw in {
        int(struc.BondType.AROMATIC_SINGLE),
        int(struc.BondType.AROMATIC_DOUBLE),
        int(struc.BondType.AROMATIC_TRIPLE),
        int(struc.BondType.AROMATIC),
    }:
        return True, LIGAND_BOND_TYPES["aromatic"]
    raise ValueError(
        f"unsupported modified-polymer carrier Biotite bond type: {raw}"
    )


def _endpoint_slot(
    compiled: CompiledSample,
    atoms: struc.AtomArray,
    row: int,
) -> int:
    modality = str(atoms.mol_type[row])
    if modality == "ligand":
        return LIGAND_ENDPOINT_SLOT
    atom_name = str(atoms.atom_name[row])
    residue_uid = int(atoms.residue_uid[row])
    proxy_residue_name = compiled.model_parent_residue.get(residue_uid)
    if proxy_residue_name is not None:
        token_rows = np.where(atoms.token_uid == int(atoms.token_uid[row]))[0]
        local_matches = np.where(token_rows == int(row))[0]
        if len(local_matches) != 1:
            raise ValueError("modified-polymer endpoint row is not token-local")
        mapped_name = proxy_atom_rows(
            modality=modality,
            residue_name=str(atoms.res_name[row]),
            proxy_residue_name=proxy_residue_name,
            source_atom_names=[str(atoms.atom_name[value]) for value in token_rows],
        ).get(int(local_matches[0]))
        if mapped_name is None:
            raise ValueError(
                "topology condition endpoint is outside the modified-polymer "
                f"proxy inventory: {atoms.chain_id[row]}:{atoms.res_id[row]}:{atom_name}"
            )
        atom_name = mapped_name
    if modality == "protein":
        return PROTEIN_ATOM37.index(atom_name)
    if modality == "rna":
        return len(PROTEIN_ATOM37) + RNA_ATOM27.index(atom_name)
    if modality == "dna":
        return len(PROTEIN_ATOM37) + len(RNA_ATOM27) + DNA_ATOM27.index(atom_name)
    raise ValueError(f"unknown modality {modality!r}")


def _frame(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> tuple[torch.Tensor, bool]:
    def normalize(value: torch.Tensor) -> torch.Tensor:
        return value / value.norm().clamp_min(1.0e-8)
    e1 = normalize(p3 - p2)
    e3_raw = torch.linalg.cross(e1, p1 - p2, dim=-1)
    ok = bool((e3_raw.norm() > 1.0e-6).item())
    e3 = normalize(e3_raw)
    e2 = normalize(torch.linalg.cross(e3, e1, dim=-1))
    return torch.stack((e1, e2, e3), dim=-1), ok


def _chain_breaks(
    compiled: CompiledSample, layout: ModelTokenLayout, coords: torch.Tensor
) -> torch.Tensor:
    """Match validation's source-modality and polymer-link continuity rules."""
    atoms = compiled.atoms
    centers = layout.model_token_to_row
    groups = layout.rows_by_model_token
    breaks = torch.ones(len(centers), dtype=torch.bool)
    bond_pairs = None if atoms.bonds is None else {
        tuple(sorted((int(i), int(j)))) for i, j, _ in atoms.bonds.as_array()
    }
    names = [normalise_atom_name(name) for name in atoms.atom_name]
    forced = frozenset(int(value) for value in compiled.forced_chain_break_token_uids)
    for token, (left, right) in enumerate(zip(centers, centers[1:])):
        authoring_token = int(atoms.token_uid[left])
        if authoring_token in forced and authoring_token != int(atoms.token_uid[right]):
            continue
        modality = str(atoms.mol_type[left])
        if (
            modality != str(atoms.mol_type[right])
            or modality == "ligand"
            or int(atoms.asym_uid[left]) != int(atoms.asym_uid[right])
            or int(atoms.res_id[right]) - int(atoms.res_id[left]) not in {0, 1}
        ):
            continue
        link = ("C", "N") if modality == "protein" else ("O3'", "P")
        if bond_pairs is not None:
            breaks[token] = not any(
                tuple(sorted((i, j))) in bond_pairs
                for i in groups[token] if names[i] == link[0]
                for j in groups[token + 1] if names[j] == link[1]
            )
        elif modality == "protein":
            ca_left = next((i for i in groups[token] if names[i] == "CA"), None)
            ca_right = next((i for i in groups[token + 1] if names[i] == "CA"), None)
            if ca_left is not None and ca_right is not None:
                breaks[token] = not bool(
                    atoms.coord_valid[ca_left] and atoms.coord_valid[ca_right]
                    and torch.linalg.vector_norm(coords[ca_right] - coords[ca_left]) <= 0.4
                )
    return breaks


def compile_canonical_sample(
    compiled: CompiledSample, *, reference: bool = False,
) -> dict[str, Any]:
    """Compile the authoring IR to the exact checkpoint-facing canonical schema."""

    atoms = compiled.atoms
    validate_atom_array(atoms, phase="tokenized")
    layout = build_model_token_layout(atoms, compiled.modified_polymer_carriers)
    groups = [np.asarray(rows, dtype=np.int64) for rows in layout.rows_by_model_token]
    n = len(groups)
    sample = empty_canonical_sample(n)
    coords = torch.as_tensor(_model_coordinates(compiled), dtype=torch.float32)
    # Validation marks optional centers as conditioned atoms as well as exposing
    # the separate token-center channel. Keep the authoring flags distinct.
    coordinate_condition = np.asarray(atoms.motif_coord_condition, dtype=bool) | np.asarray(
        atoms.token_center_condition, dtype=bool
    )
    atom_to_token = np.asarray(layout.model_token_by_row, dtype=np.int64)

    for token_uid, rows in enumerate(groups):
        center = int(layout.model_token_to_row[token_uid])
        if center not in rows:
            raise ValueError("model-token representative row is outside its row group")
        source_modality = str(atoms.mol_type[center])
        is_carrier = bool(layout.carrier_token_mask[token_uid])
        modality = "ligand" if is_carrier else source_modality
        raw_residue_name = str(atoms.res_name[center])
        model_residue_name = compiled.model_parent_residue.get(
            int(atoms.residue_uid[center]), raw_residue_name
        )
        sample["mol_type_id"][token_uid] = MOL_TYPE_TO_ID[modality]
        sample["token_type32"][token_uid] = TOKEN_TYPE_IGNORE_INDEX if modality == "ligand" else TOKEN_TYPE32.get(model_residue_name, TOKEN_TYPE32[_unknown_token(modality)])
        sample["residue_index"][token_uid] = int(atoms.res_id[center])
        sample["asym_id"][token_uid] = int(atoms.asym_uid[center])
        sample["mol_id"][token_uid] = int(
            layout.model_molecule_uid_by_model_token[token_uid]
        )
        sample["design_token_mask"][token_uid] = bool(atoms.is_design[center])
        if bool(atoms.coord_valid[center]):
            sample["token_center_coords_nm"][token_uid] = coords[center]
        sample["hotspot_mask"][token_uid] = bool(atoms.hotspot_condition[rows].any())
        if bool(atoms.token_center_condition[center]) and bool(atoms.coord_valid[center]):
            sample["optional_token_center_mask"][token_uid] = True
            sample["optional_token_center_coords_nm"][token_uid] = coords[center]
        if is_carrier:
            residue_uid = int(atoms.residue_uid[center])
            carrier = compiled.modified_polymer_carriers[residue_uid]
            atom_uid = int(atoms.atom_uid[center])
            sample["token_identity_condition_mask"][token_uid] = True
            sample["ligand_mask"][token_uid] = True
            element = str(atoms.element[center]).strip().upper()
            element_id = ELEMENT_TO_ID.get(element, UNKNOWN_ELEMENT_ID)
            if element_id == UNKNOWN_ELEMENT_ID:
                raise ValueError(
                    "modified-polymer carrier element is outside the checkpoint "
                    f"vocabulary: {element!r}"
                )
            sample["ligand_element_type"][token_uid] = element_id
            sample["ligand_charge"][token_uid] = float(
                carrier.formal_charge(atom_uid)
            )
            if bool(coordinate_condition[center]) and bool(
                atoms.coord_valid[center]
            ):
                sample["ligand_motif_mask"][token_uid] = True
                sample["ligand_motif_token_center_coords_nm"][token_uid] = coords[
                    center
                ]
            continue

        identity = np.asarray(atoms.identity_condition[rows], dtype=bool)
        if identity.any() and not identity.all():
            raise ValueError(f"token {token_uid} has partial identity visibility")
        sample["token_identity_condition_mask"][token_uid] = bool(identity.all())
        if modality == "ligand":
            sample["ligand_mask"][token_uid] = True
            element = str(atoms.element[center]).strip().upper()
            sample["ligand_element_type"][token_uid] = ELEMENT_TO_ID.get(element, UNKNOWN_ELEMENT_ID)
            if reference and atoms.formal_charge[center] is not None:
                charge = float(atoms.formal_charge[center])
                if not np.isfinite(charge) or not charge.is_integer():
                    raise DesignSampleError("partial reference has an invalid ligand formal charge")
                sample["ligand_charge"][token_uid] = charge
            if bool(coordinate_condition[center]) and bool(atoms.coord_valid[center]):
                sample["ligand_motif_mask"][token_uid] = True
                sample["ligand_motif_token_center_coords_nm"][token_uid] = coords[center]
            continue

        names = PROTEIN_ATOM37 if modality == "protein" else RNA_ATOM27 if modality == "rna" else DNA_ATOM27
        inventory = PROTEIN_RES_ATOMS if modality == "protein" else RNA_RES_ATOMS if modality == "rna" else DNA_RES_ATOMS
        residue_uid = int(atoms.residue_uid[center])
        modified = residue_uid in compiled.model_parent_residue
        allowed_names = (
            set(proxy_inventory(modality, model_residue_name))
            if modified
            else set(names)
            if model_residue_name == _unknown_token(modality)
            else set(inventory[model_residue_name])
        )
        if modality == "protein" and not modified:
            allowed_names.add("OXT")
        coords_key = f"{modality}_atom{len(names)}_coords"
        mask_key = f"{modality}_atom{len(names)}_mask"
        motif_coords_key = f"{modality}_motif_atom{len(names)}_coords_nm"
        motif_mask_key = f"{modality}_motif_atom{len(names)}_mask"
        modified_names_by_local_row = (
            proxy_atom_rows(
                modality=modality,
                residue_name=raw_residue_name,
                proxy_residue_name=model_residue_name,
                source_atom_names=[str(atoms.atom_name[row]) for row in rows],
            )
            if modified
            else {}
        )
        occupied_slots: dict[str, int] = {}
        for local_row, row in enumerate(rows):
            source_atom_name = str(atoms.atom_name[row])
            atom_name = (
                modified_names_by_local_row.get(local_row)
                if modified
                else source_atom_name
            )
            if atom_name is None or atom_name not in allowed_names:
                continue
            if atom_name not in names:
                raise ValueError(f"atom {atom_name!r} is outside {modality} canonical slots")
            previous = occupied_slots.get(atom_name)
            if previous is not None:
                raise ValueError(
                    "modified-polymer source atoms alias to one proxy slot: "
                    f"rows {previous} and {int(row)} -> {atom_name!r}"
                )
            occupied_slots[atom_name] = int(row)
            slot = names.index(atom_name)
            if bool(atoms.coord_valid[row]):
                sample[coords_key][token_uid, slot] = coords[row]
                sample[mask_key][token_uid, slot] = True
                if bool(coordinate_condition[row]):
                    sample[motif_coords_key][token_uid, slot] = coords[row]
                    sample[motif_mask_key][token_uid, slot] = True

    sample["chain_break_after_token"] = _chain_breaks(compiled, layout, coords)

    for modality, names, prefix in (
        ("protein", PROTEIN_ATOM37, "protein_atom37"),
        ("rna", RNA_ATOM27, "rna_atom27"),
        ("dna", DNA_ATOM27, "dna_atom27"),
    ):
        frame_names = ("N", "CA", "C") if modality == "protein" else ("C3'", "C1'", "C4'")
        slots = [names.index(name) for name in frame_names]
        for token_uid in torch.where(sample["mol_type_id"] == MOL_TYPE_TO_ID[modality])[0].tolist():
            if bool(sample[f"{prefix}_mask"][token_uid, slots].all().item()):
                frame, ok = _frame(*sample[f"{prefix}_coords"][token_uid, slots])
                if ok:
                    sample["token_frame"][token_uid] = frame
                    sample["token_frame_mask"][token_uid] = True

    if atoms.bonds is not None:
        for row_i, row_j, raw_order in atoms.bonds.as_array():
            i, j = int(row_i), int(row_j)
            token_i, token_j = int(atom_to_token[i]), int(atom_to_token[j])
            if token_i == token_j:
                continue
            carrier_involved = bool(
                layout.carrier_token_mask[token_i]
                or layout.carrier_token_mask[token_j]
            )
            if carrier_involved:
                order_known, order = _carrier_bond_condition_type(int(raw_order))
                if not order_known:
                    continue
            else:
                order = _bond_type(int(raw_order))
            if order == 0:
                continue
            ligand_i, ligand_j = bool(sample["ligand_mask"][token_i]), bool(sample["ligand_mask"][token_j])
            discontinuous = str(atoms.chain_id[i]) != str(atoms.chain_id[j]) or abs(int(atoms.res_id[i]) - int(atoms.res_id[j])) > 1
            if ligand_i or ligand_j or discontinuous:
                # Training merges equal target bond classes at token-pair level;
                # atom endpoints are represented separately by condition slots.
                _set_symmetric_bond(sample["token_bond_type"], token_i, token_j, order)
    sample["token_bond_pair_mask"] = sample["token_bond_type"] > 0

    _materialize_topology_conditions(sample, compiled, atoms, layout)

    for token_uid, value in compiled.coarse_ss.items():
        model_token = _ss_model_token(layout, int(token_uid), field="coarse SS")
        sample["polymer_coarse_ss"][model_token] = int(value)
    for token_uid, value in compiled.ss_token.items():
        model_token = _ss_model_token(layout, int(token_uid), field="token SS")
        sample["polymer_ss_token"][model_token] = int(value)
        sample["polymer_ss_token_condition_mask"][model_token] = True
    for record in compiled.ss_pair:
        i = _ss_model_token(layout, int(record.token_uid_a), field="pair SS")
        j = _ss_model_token(layout, int(record.token_uid_b), field="pair SS")
        value = int(record.value)
        sample["polymer_ss_pair"][i, j] = sample["polymer_ss_pair"][j, i] = value
        sample["polymer_ss_pair_condition_mask"][i, j] = sample["polymer_ss_pair_condition_mask"][j, i] = True
    if compiled.antibody_metadata:
        from ..antibody.compiler import template_canonical_fields
        sample.update(template_canonical_fields(compiled, layout, sample))
    validate_canonical_sample(sample)
    return sample


def _unknown_token(modality: str) -> str:
    return {"protein": "UNK", "rna": "N", "dna": "DN"}[modality]


def _set_symmetric_bond(matrix: torch.Tensor, i: int, j: int, value: int) -> None:
    previous = int(matrix[i, j].item())
    if previous not in {0, value}:
        raise ValueError(f"conflicting canonical bond type for token pair {(i, j)}")
    matrix[i, j] = matrix[j, i] = value


def _condition_order_type(order: str | None) -> tuple[bool, int]:
    if order is None or str(order).strip().lower() in {
        "",
        "any",
        "unknown",
        "quad",
        "quadruple",
    }:
        return False, LIGAND_BOND_TYPES["none"]
    value = str(order).strip().lower()
    if value not in LIGAND_BOND_TYPES or value == "none":
        raise ValueError(f"unsupported topology condition order {order!r}")
    return True, int(LIGAND_BOND_TYPES[value])


def _ss_model_token(
    layout: ModelTokenLayout,
    authoring_token: int,
    *,
    field: str,
) -> int:
    if not 0 <= int(authoring_token) < layout.num_authoring_tokens:
        raise ValueError(f"{field} references a missing authoring token")
    model_tokens = layout.authoring_token_to_model_tokens[int(authoring_token)]
    if any(layout.carrier_token_mask[token] for token in model_tokens):
        raise ValueError(f"{field} cannot target a modified-polymer carrier")
    if len(model_tokens) != 1:
        raise ValueError(f"{field} authoring token does not map one-to-one to the model")
    return int(model_tokens[0])


def _set_condition_edge(
    sample: dict[str, Any],
    compiled: CompiledSample,
    atoms: struc.AtomArray,
    layout: ModelTokenLayout,
    row_i: int,
    row_j: int,
    *,
    order_known: bool,
    order_type: int,
    seen_pairs: dict[
        tuple[int, int], tuple[tuple[int, int], bool, int, tuple[int, int]]
    ],
    preserve_existing_order: bool = False,
) -> None:
    token_i = int(layout.model_token_by_row[int(row_i)])
    token_j = int(layout.model_token_by_row[int(row_j)])
    if token_i == token_j:
        return
    if token_i > token_j:
        token_i, token_j, row_i, row_j = token_j, token_i, row_j, row_i
    pair = (token_i, token_j)
    atom_pair = tuple(
        sorted((int(atoms.atom_uid[int(row_i)]), int(atoms.atom_uid[int(row_j)])))
    )
    slot_i = (
        LIGAND_ENDPOINT_SLOT
        if layout.carrier_token_mask[token_i]
        else _endpoint_slot(compiled, atoms, int(row_i))
    )
    slot_j = (
        LIGAND_ENDPOINT_SLOT
        if layout.carrier_token_mask[token_j]
        else _endpoint_slot(compiled, atoms, int(row_j))
    )
    signature = (atom_pair, bool(order_known), int(order_type), (slot_i, slot_j))
    previous = seen_pairs.get(pair)
    if previous is not None:
        if preserve_existing_order and previous[0] == atom_pair and previous[3] == (slot_i, slot_j):
            return
        if previous != signature:
            raise ValueError(
                f"multiple required condition edges map to token pair {pair}"
            )
        return
    seen_pairs[pair] = signature
    sample["condition_bond_pair_mask"][token_i, token_j] = True
    sample["condition_bond_pair_mask"][token_j, token_i] = True
    sample["condition_bond_pair_endpoint_slot"][token_i, token_j] = torch.tensor(
        (slot_i, slot_j), dtype=torch.int16
    )
    sample["condition_bond_pair_endpoint_slot"][token_j, token_i] = torch.tensor(
        (slot_j, slot_i), dtype=torch.int16
    )
    if order_known:
        sample["condition_bond_type"][token_i, token_j] = int(order_type)
        sample["condition_bond_type"][token_j, token_i] = int(order_type)
        sample["condition_bond_type_known_mask"][token_i, token_j] = True
        sample["condition_bond_type_known_mask"][token_j, token_i] = True


def _materialize_topology_conditions(
    sample: dict[str, Any],
    compiled: CompiledSample,
    atoms: struc.AtomArray,
    layout: ModelTokenLayout,
) -> None:
    uid_to_row = {int(uid): index for index, uid in enumerate(atoms.atom_uid)}
    seen_pairs: dict[
        tuple[int, int], tuple[tuple[int, int], bool, int, tuple[int, int]]
    ] = {}
    explicit_condition_pairs: set[tuple[int, int]] = set()

    if atoms.bonds is not None:
        for raw_row_i, raw_row_j, raw_order in atoms.bonds.as_array():
            row_i, row_j = int(raw_row_i), int(raw_row_j)
            token_i = int(layout.model_token_by_row[row_i])
            token_j = int(layout.model_token_by_row[row_j])
            if token_i == token_j or not (
                layout.carrier_token_mask[token_i]
                or layout.carrier_token_mask[token_j]
            ):
                continue
            order_known, order_type = _carrier_bond_condition_type(int(raw_order))
            same_carrier = (
                layout.carrier_residue_uid_by_model_token[token_i] is not None
                and layout.carrier_residue_uid_by_model_token[token_i]
                == layout.carrier_residue_uid_by_model_token[token_j]
            )
            if same_carrier and not order_known:
                raise ValueError(
                    "modified-polymer carrier internal bonds require exact order"
                )
            _set_condition_edge(
                sample,
                compiled,
                atoms,
                layout,
                row_i,
                row_j,
                order_known=order_known,
                order_type=order_type,
                seen_pairs=seen_pairs,
            )

    for condition in compiled.topology_conditions:
        if condition.atom_uid_a not in uid_to_row or condition.atom_uid_b not in uid_to_row:
            raise ValueError("topology condition references missing AtomUID")
        row_i = uid_to_row[condition.atom_uid_a]
        row_j = uid_to_row[condition.atom_uid_b]
        token_i = int(layout.model_token_by_row[row_i])
        token_j = int(layout.model_token_by_row[row_j])
        value = str(condition.order).strip().lower()
        carrier_residue_i = layout.carrier_residue_uid_by_model_token[token_i]
        carrier_residue_j = layout.carrier_residue_uid_by_model_token[token_j]
        same_carrier = (
            carrier_residue_i is not None
            and carrier_residue_i == carrier_residue_j
        )
        if token_i == token_j:
            if same_carrier:
                raise ValueError(
                    "topology condition contradicts the complete carrier graph"
                )
            continue
        explicit_pair = tuple(sorted((token_i, token_j)))
        if explicit_pair in explicit_condition_pairs:
            raise ValueError(
                f"multiple required condition edges map to token pair {explicit_pair}"
            )
        explicit_condition_pairs.add(explicit_pair)
        if same_carrier and (
            explicit_pair not in seen_pairs
        ):
            raise ValueError(
                "topology condition contradicts the complete carrier graph"
            )
        if (
            layout.carrier_token_mask[token_i]
            or layout.carrier_token_mask[token_j]
        ) and value in {"quad", "quadruple"}:
            raise ValueError("modified-polymer carrier bonds cannot use QUADRUPLE order")
        order_known, order_type = _condition_order_type(condition.order)
        _set_condition_edge(
            sample,
            compiled,
            atoms,
            layout,
            row_i,
            row_j,
            order_known=order_known,
            order_type=order_type,
            seen_pairs=seen_pairs,
        )

    # Context source topology is supplied information, including concrete orders.
    # Use the training knownness/class ABI, not a training recipe's stochastic
    # visibility policy. Explicit and carrier facts above retain precedence.
    # Fixed context may have several atom edges per token pair. Select its
    # representative as training does: conditioned endpoints, distance, slots.
    context_edges: dict[tuple[int, int], tuple[Any, ...]] = {}
    if atoms.bonds is not None:
        for raw_row_i, raw_row_j, raw_order in atoms.bonds.as_array():
            row_i, row_j = int(raw_row_i), int(raw_row_j)
            if bool(atoms.is_design[row_i]) or bool(atoms.is_design[row_j]):
                continue
            token_i = int(layout.model_token_by_row[row_i])
            token_j = int(layout.model_token_by_row[row_j])
            if token_i == token_j or int(raw_order) == int(struc.BondType.COORDINATION):
                continue
            # Validate the enum, but never reuse the target graph's lossy
            # ANY->single / QUADRUPLE->triple coercion as a condition.
            order_type = _bond_type(int(raw_order))
            order_known = int(raw_order) not in {
                int(struc.BondType.ANY), int(struc.BondType.QUADRUPLE),
            }
            if not order_known:
                order_type = LIGAND_BOND_TYPES["none"]
            ligand_involved = bool(sample["ligand_mask"][token_i] or sample["ligand_mask"][token_j])
            discontinuous = (
                str(atoms.chain_id[row_i]) != str(atoms.chain_id[row_j])
                or abs(int(atoms.res_id[row_i]) - int(atoms.res_id[row_j])) > 1
            )
            if not (ligand_involved or discontinuous):
                continue
            if token_i > token_j:
                token_i, token_j, row_i, row_j = token_j, token_i, row_j, row_i
            pair = (token_i, token_j)
            if pair in seen_pairs:
                continue
            rank = (
                -(int(atoms.coord_valid[row_i]) + int(atoms.coord_valid[row_j])),
                float(np.sum((atoms.coord[row_i] - atoms.coord[row_j]) ** 2)),
                _endpoint_slot(compiled, atoms, row_i),
                _endpoint_slot(compiled, atoms, row_j),
                tuple(sorted((int(atoms.atom_uid[row_i]), int(atoms.atom_uid[row_j])))),
            )
            previous = context_edges.get(pair)
            if previous is None or rank < previous[0]:
                context_edges[pair] = (rank, row_i, row_j, order_known, order_type)
    for _, row_i, row_j, order_known, order_type in context_edges.values():
        _set_condition_edge(
            sample, compiled, atoms, layout, row_i, row_j,
            order_known=order_known, order_type=order_type, seen_pairs=seen_pairs,
        )

    carrier_tokens_by_residue: dict[int, list[int]] = {}
    for token_uid, residue_uid in enumerate(
        layout.carrier_residue_uid_by_model_token
    ):
        if residue_uid is not None:
            carrier_tokens_by_residue.setdefault(int(residue_uid), []).append(token_uid)
    for token_scope in carrier_tokens_by_residue.values():
        for offset, token_i in enumerate(token_scope):
            for token_j in token_scope[offset + 1 :]:
                if bool(sample["condition_bond_pair_mask"][token_i, token_j]) and not bool(
                    sample["condition_bond_type_known_mask"][token_i, token_j]
                ):
                    raise ValueError(
                        "complete carrier graph contains a positive edge without exact order"
                    )
                sample["condition_bond_type_known_mask"][token_i, token_j] = True
                sample["condition_bond_type_known_mask"][token_j, token_i] = True

    positive_uid_pairs = {
        edge for policy in compiled.edge_policies for edge in policy.hard_positive_edges
    }
    for policy in compiled.edge_policies:
        if not policy.complete_graph and not policy.hard_negative_edges:
            continue
        if policy.component_uid is not None:
            atom_scope = np.asarray(atoms.component_uid, dtype=np.int64) == int(policy.component_uid)
        else:
            atom_scope = np.asarray(atoms.molecule_uid, dtype=np.int64) == int(policy.molecule_uid)
        atom_scope &= np.asarray(atoms.mol_type).astype(str) == "ligand"
        for uid_a, uid_b in policy.hard_negative_edges:
            if uid_a not in uid_to_row or uid_b not in uid_to_row:
                raise ValueError("hard negative edge references missing AtomUID")
            row_i, row_j = uid_to_row[uid_a], uid_to_row[uid_b]
            if not bool(atom_scope[row_i] and atom_scope[row_j]):
                raise ValueError("hard negative edge lies outside its ligand policy scope")
            token_i = int(layout.model_token_by_row[row_i])
            token_j = int(layout.model_token_by_row[row_j])
            if token_i == token_j:
                raise ValueError("hard negative edge aliases one model token")
            if (uid_a, uid_b) in positive_uid_pairs or bool(
                sample["condition_bond_pair_mask"][token_i, token_j]
            ):
                raise ValueError(
                    f"hard negative edge conflicts with a positive condition: {(uid_a, uid_b)}"
                )
            sample["condition_bond_pair_mask"][token_i, token_j] = False
            sample["condition_bond_pair_mask"][token_j, token_i] = False
            sample["condition_bond_type"][token_i, token_j] = LIGAND_BOND_TYPES["none"]
            sample["condition_bond_type"][token_j, token_i] = LIGAND_BOND_TYPES["none"]
            sample["condition_bond_type_known_mask"][token_i, token_j] = True
            sample["condition_bond_type_known_mask"][token_j, token_i] = True
        if not policy.complete_graph:
            continue
        token_scope = sorted(
            set(
                int(layout.model_token_by_row[row])
                for row in np.where(atom_scope)[0]
            )
        )
        if len(token_scope) < 1:
            raise ValueError("complete ligand edge policy resolves to no ligand tokens")
        for offset, token_i in enumerate(token_scope):
            for token_j in token_scope[offset + 1 :]:
                if bool(sample["condition_bond_pair_mask"][token_i, token_j]):
                    if not bool(sample["condition_bond_type_known_mask"][token_i, token_j]):
                        raise ValueError(
                            "complete ligand graph contains a positive edge without exact order"
                        )
                sample["condition_bond_type_known_mask"][token_i, token_j] = True
                sample["condition_bond_type_known_mask"][token_j, token_i] = True


def validate_canonical_sample(sample: Mapping[str, Any]) -> None:
    specs = _sample_field_specs(sample)
    if set(sample) != set(specs):
        raise ValueError(f"canonical keys differ: missing={set(specs)-set(sample)}, extra={set(sample)-set(specs)}")
    n = int(sample["token_mask"].shape[0])
    if n <= 0 or not bool(sample["token_mask"].all().item()):
        raise ValueError("unbatched canonical token mask must be all true and non-empty")
    for name, spec in specs.items():
        value = sample[name]
        expected_shape = (n,) + spec.trailing_shape if spec.scope == FieldScope.TOKEN else (n, n) + spec.trailing_shape if spec.scope == FieldScope.PAIR else spec.trailing_shape
        if not isinstance(value, torch.Tensor) or value.dtype != spec.dtype or tuple(value.shape) != expected_shape or value.device.type != "cpu":
            raise TypeError(f"canonical field {name} has wrong tensor ABI")
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"canonical field {name} contains non-finite values")
        if spec.scope == FieldScope.PAIR:
            expected = value.transpose(0, 1).flip(-1) if spec.symmetry == PairSymmetry.ENDPOINT_SWAP else value.transpose(0, 1)
            if not torch.equal(value, expected):
                raise ValueError(f"canonical pair field {name} violates symmetry")
    if "design_entity_id" in sample:
        if not torch.equal(sample["design_entity_id"] >= 0, sample["design_token_mask"]):
            raise ValueError("antibody semantic design IDs disagree with design_token_mask")
        template_mask = sample["template_protein_atom37_mask"]
        if not torch.equal(sample["template_group_id"] > 0, template_mask.any(-1)):
            raise ValueError("positive template geometry groups require visible atoms")
        if bool((template_mask.any(-1) & (sample["antibody_region_id"] != 1)).any()):
            raise ValueError("antibody template geometry may expose only framework residues")
        if bool((sample["template_protein_atom37_coords_nm"][~template_mask] != 0).any()):
            raise ValueError("hidden template coordinate storage must be zero")
    if not torch.equal(sample["ligand_mask"], sample["mol_type_id"] == MOL_TYPE_TO_ID["ligand"]):
        raise ValueError("ligand_mask disagrees with mol_type_id")
    condition_edge = sample["condition_bond_pair_mask"].bool()
    condition_known = sample["condition_bond_type_known_mask"].bool()
    condition_type = sample["condition_bond_type"].long()
    if not bool(((condition_type >= 0) & (condition_type <= 4)).all().item()):
        raise ValueError("bond condition type is outside the five-class ABI")
    if not torch.equal(condition_type > 0, condition_edge & condition_known):
        raise ValueError("bond condition violates the four-state edge/order ABI")


def dense_pad_canonical_collate(
    samples: Sequence[Mapping[str, Any]], *, pad_max_size: int | None = None,
    token_buckets: Sequence[int] | None = None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("dense canonical collate requires at least one sample")
    for sample in samples:
        validate_canonical_sample(sample)
    if any(set(sample) != set(samples[0]) for sample in samples[1:]):
        raise ValueError("canonical batches cannot mix antibody template and normal schemas")
    specs = _sample_field_specs(samples[0])
    lengths = [int(sample["token_mask"].shape[0]) for sample in samples]
    maximum = max(lengths)
    if token_buckets is not None:
        if pad_max_size is not None:
            raise ValueError("pad_max_size and token_buckets are mutually exclusive")
        maximum = token_bucket_size(maximum, token_buckets)
    if pad_max_size is not None:
        if int(pad_max_size) < maximum:
            raise ValueError("canonical truncation is forbidden")
        maximum = int(pad_max_size)
    batch: dict[str, Any] = {}
    for name, spec in specs.items():
        values = [sample[name] for sample in samples]
        if spec.scope == FieldScope.TOKEN:
            padded = []
            for value in values:
                output = torch.full((maximum,) + tuple(value.shape[1:]), spec.pad_value, dtype=value.dtype)
                output[: len(value)] = value
                padded.append(output)
            batch[name] = torch.stack(padded)
        elif spec.scope == FieldScope.PAIR:
            padded = []
            for value in values:
                output = torch.full((maximum, maximum) + tuple(value.shape[2:]), spec.pad_value, dtype=value.dtype)
                output[: value.shape[0], : value.shape[1]] = value
                padded.append(output)
            batch[name] = torch.stack(padded)
        else:
            raise AssertionError(f"unsupported standalone batch scope: {spec.scope}")
    return batch


@dataclass(frozen=True)
class PredictBatch:
    samples: tuple[CompiledSample, ...]
    requests: tuple[ExpandedRequest, ...]
    canonical: Mapping[str, Any]
    reference: Mapping[str, Any] | None = None
    partial_fraction: float | None = None


def _cyclic_periods(compiled: CompiledSample, canonical: Mapping[str, Any]) -> torch.Tensor:
    """Build opt-in FM-only periods independently of chemical bond conditions.

    Count source residues, not atomized carrier tokens. Ambiguous or incomplete
    residue numbering cannot define an unambiguous period. Never inspect bonds
    or generated coordinates, and never apply the task flag to context receptors.
    """
    atoms = compiled.atoms
    periods = torch.zeros_like(canonical["residue_index"])
    if not compiled.request.task.cyclic_offset:
        return periods
    design_chains = {
        chain.chain_id for chain in compiled.request.task.design_chains
        if chain.modality == "protein"
    }
    selected = set(compiled.request.task.cyclic_offset_chains)
    if selected:
        if not selected <= design_chains:
            raise DesignSampleError("cyclic_offset_chains must select design protein chains")
        design_chains = selected
    elif len(design_chains) != 1:
        raise DesignSampleError("cyclic_offset requires one design protein chain or explicit cyclic_offset_chains")
    for asym in np.unique(atoms.asym_uid):
        chain_rows = np.flatnonzero(atoms.asym_uid == asym)
        if str(atoms.chain_id[chain_rows[0]]) not in design_chains:
            continue
        if not np.all(atoms.is_design[chain_rows]):
            continue
        if not np.all(atoms.mol_type[chain_rows] == "protein"):
            raise DesignSampleError("cyclic_offset requires a protein design chain")
        if any(str(code).strip() not in {"", ".", "?"} for code in atoms.ins_code[chain_rows]):
            raise DesignSampleError("cyclic_offset does not support insertion-coded residues")
        residue_ids = np.unique(atoms.res_id[chain_rows])
        if len(residue_ids) < 3 or not np.all(np.diff(residue_ids) == 1):
            raise DesignSampleError("cyclic_offset requires at least three consecutively numbered residues")
        # A residue position must denote exactly one source residue, including
        # carriers whose several model tokens share that same position.
        if any(
            len(np.unique(atoms.residue_uid[chain_rows[atoms.res_id[chain_rows] == pos]])) != 1
            for pos in residue_ids
        ):
            raise DesignSampleError("cyclic_offset requires unique residue positions")
        mask = canonical["asym_id"] == int(asym)
        periods[mask] = len(residue_ids)
    return periods


def collate_compiled_samples(
    samples: Sequence[CompiledSample], *, pad_max_size: int | None = None,
    token_buckets: Sequence[int] | None = None,
) -> PredictBatch:
    partial = [sample.reference_atoms is not None for sample in samples]
    if any(partial) and not all(partial):
        raise DesignSampleError("partial and ordinary samples require separate batches")
    fractions = {sample.partial_fraction for sample in samples}
    if len(fractions) != 1 or (any(partial) and None in fractions):
        raise DesignSampleError("a batch requires one homogeneous partial fraction")
    if not any(partial) and fractions != {None}:
        raise DesignSampleError("partial fraction requires reference atoms")
    canonical_samples = [compile_canonical_sample(sample) for sample in samples]
    canonical = dense_pad_canonical_collate(
        canonical_samples, pad_max_size=pad_max_size, token_buckets=token_buckets,
    )
    # Runtime-only metadata: keep the frozen training canonical schema unchanged.
    # Always attach the tensor so changing cyclic content does not change keys
    # seen by the compiled FM network. AE features deliberately ignore it.
    periods = torch.zeros_like(canonical["residue_index"])
    for row, (sample, unbatched) in enumerate(zip(samples, canonical_samples, strict=True)):
        period = _cyclic_periods(sample, unbatched)
        periods[row, :len(period)] = period
    canonical["cyclic_period"] = periods
    reference = None
    if any(partial):
        reference_samples = [
            compile_canonical_sample(replace(
                sample, atoms=sample.reference_atoms, reference_atoms=None,
                partial_fraction=None,
            ), reference=True)
            for sample in samples
        ]
        for compiled, unbatched in zip(samples, reference_samples, strict=True):
            # The VAE's ordinary polymer view needs its native local frame.
            # Carriers are atomized ligand tokens and have no polymer frame.
            invalid_frames = (
                unbatched["token_mask"] & ~unbatched["ligand_mask"]
                & ~unbatched["token_frame_mask"]
            )
            if bool(invalid_frames.any()):
                tokens = torch.where(invalid_frames)[0].tolist()
                raise DesignSampleError(
                    f"partial reference {compiled.request.sample_id} has non-encodable polymer frames at model tokens {tokens}"
                )
        reference = dense_pad_canonical_collate(
            reference_samples, pad_max_size=canonical["token_mask"].shape[1],
        )
        reference["cyclic_period"] = periods
    return PredictBatch(
        samples=tuple(samples),
        requests=tuple(sample.request for sample in samples),
        canonical=canonical,
        reference=reference,
        partial_fraction=next(iter(fractions)),
    )


__all__ = [
    "CANONICAL_FIELD_SPECS", "PredictBatch", "collate_compiled_samples",
    "compile_canonical_sample", "dense_pad_canonical_collate", "empty_canonical_sample",
    "validate_canonical_sample",
]
