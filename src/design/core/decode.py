"""Terminal modality decode for standalone PartiDesign inference.

This module turns batched FM/VAE tensors into a small, private proposal.  The
proposal is not a second structural IR: stable atom inventory and all hard
facts remain owned by :class:`~src.design.core.records.CompiledSample`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch
from biotite import structure as struc
from torch import Tensor

from ..utils.chemistry import reconstruct_obabel_with_molecule
from ..model.constants import (
    DESIGN_LIGAND_ELEMENT_IDS,
    ELEMENTS_128,
    ID_TO_MOL_TYPE,
    ID_TO_TOKEN_TYPE32,
    LIGAND_BOND_TYPES,
    MOL_TYPE_TO_ID,
)
from .model_tokens import build_model_token_layout
from .omission import normalize_rm_token_type
from .records import CompiledSample


_VALID_POLYMER_TOKEN_IDS = {
    MOL_TYPE_TO_ID["protein"]: tuple(range(0, 20)),
    MOL_TYPE_TO_ID["rna"]: tuple(range(21, 25)),
    MOL_TYPE_TO_ID["dna"]: tuple(range(26, 30)),
}
_RM_TOKEN_TYPE_NAMES = frozenset(ID_TO_TOKEN_TYPE32.values())
_MOTIF_MASK_FIELDS = (
    "protein_motif_atom37_mask",
    "rna_motif_atom27_mask",
    "dna_motif_atom27_mask",
    "ligand_motif_mask",
    "optional_token_center_mask",
)
_CANONICAL_TO_BIOTITE = {
    LIGAND_BOND_TYPES["single"]: int(struc.BondType.SINGLE),
    LIGAND_BOND_TYPES["double"]: int(struc.BondType.DOUBLE),
    LIGAND_BOND_TYPES["triple"]: int(struc.BondType.TRIPLE),
    LIGAND_BOND_TYPES["aromatic"]: int(struc.BondType.AROMATIC),
}

# Exact validation-dump valence route. Aromatic consumes its minimum feasible
# order because the five-way head does not distinguish aromatic single/double.
_MIN_BOND_ORDER_HALF_UNITS = (0, 2, 4, 6, 2)
_ALLOWED_VALENCES = {
    "H": (1,),
    "B": (3, 4),
    "C": (4,),
    "N": (3, 4),
    "O": (1, 2),
    "F": (1,),
    "SI": (4,),
    "P": (3, 5),
    "S": (2, 4, 6),
    "CL": (1,),
    "AS": (3, 5),
    "SE": (2, 4, 6),
    "BR": (1,),
    "I": (1, 3, 5),
}
_NONCLASSICAL_BONDING_ELEMENTS = frozenset(("B",))


@dataclass(frozen=True, slots=True)
class DecodedProposal:
    """Private terminal model proposal keyed by canonical token UID.

    Coordinates stay in the model frame and nanometres.  Bond endpoints are
    token UIDs; ligand tokens are one atom each.  Charge logits and sanitized
    OpenBabel molecules are retained only for optional private chemistry and
    never become AtomArray annotations or public writer fields.
    """

    token_mask: np.ndarray
    design_token_mask: np.ndarray
    generated_token_mask: np.ndarray
    context_constraint_mode: str
    polymer_res_name: tuple[str | None, ...]
    token_center_nm: np.ndarray
    polymer_atom91_nm: np.ndarray
    ligand_element: tuple[str | None, ...]
    ligand_bonds: tuple[tuple[int, int, int], ...]
    decoded_ligand_molecule_uids: frozenset[int]
    private_ligand_charge_logits: np.ndarray | None = None
    private_ligand_molecules: Mapping[int, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    private_ligand_atom_uids: Mapping[int, tuple[int, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )


def _sample_tensor(
    values: Mapping[str, Any],
    key: str,
    sample_index: int,
    *,
    scalar_ok: bool = False,
) -> Tensor:
    value = values.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"decode input {key!r} must be a torch.Tensor")
    if scalar_ok and value.ndim == 0:
        return value
    if value.ndim == 0 or not 0 <= int(sample_index) < int(value.shape[0]):
        raise ValueError(f"decode input {key!r} has no sample axis entry {sample_index}")
    return value[int(sample_index)]


def _finite_cpu(value: Tensor, *, name: str) -> Tensor:
    result = value.detach().float().cpu()
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"decode tensor {name!r} contains non-finite values")
    return result


def _restricted_polymer_id(
    logits: Tensor,
    mol_type_id: int,
    rm_token_type: tuple[str, ...] | None = None,
) -> int:
    """Return the highest-scoring valid polymer token identity.

    rm_token_type is intentionally a tuple of canonical TOKEN_TYPE32 names,
    rather than model IDs. This keeps the omission policy tied to the
    checkpoint vocabulary and makes RNA/DNA C/DC distinct. The None branch
    preserves the original numerical route exactly.
    """

    valid_ids = _VALID_POLYMER_TOKEN_IDS.get(int(mol_type_id))
    if valid_ids is None:
        raise ValueError(f"polymer decode received modality id {mol_type_id}")
    if rm_token_type is not None:
        names = tuple(str(name).strip().upper() for name in rm_token_type)
        invalid = sorted(set(names).difference(_RM_TOKEN_TYPE_NAMES))
        if invalid:
            raise ValueError(
                "rm_token_type contains unsupported TOKEN_TYPE32 name(s): "
                f"{invalid}"
            )
        omitted = set(names)
        valid_ids = tuple(
            token_id
            for token_id in valid_ids
            if ID_TO_TOKEN_TYPE32[token_id] not in omitted
        )
        if not valid_ids:
            modality = ID_TO_MOL_TYPE.get(int(mol_type_id), str(mol_type_id))
            raise ValueError(
                f"rm_token_type removes every valid {modality} polymer identity"
            )
    index = torch.tensor(valid_ids, dtype=torch.long)
    return int(index[logits[index].argmax()].item())


def _decode_rm_token_type(compiled: CompiledSample | None) -> tuple[str, ...] | None:
    """Resolve the already-normalized omission policy from the request task.

    Task/config parsing owns validation. Decode only reads the resolved task
    field, while the defensive checks keep direct callers from introducing
    arbitrary model vocabulary IDs.
    """

    if compiled is None:
        return None
    task = getattr(getattr(compiled, "request", None), "task", None)
    value = getattr(task, "rm_token_type", None)
    return normalize_rm_token_type(
        value, "compiled request task rm_token_type"
    )


def _polymer_protected_mask(
    canonical_batch: Mapping[str, Any],
    *,
    sample_index: int,
    token_mask: Tensor,
    design_mask: Tensor,
    identity_mask: Tensor,
    mol_type: Tensor,
    carrier_mask: Tensor,
    compiled: CompiledSample | None,
    layout: Any | None,
) -> Tensor:
    """Return polymer tokens exempt from identity omission.

    The canonical motif/center masks are compiled condition metadata. If a
    compiled sample is available, its row-level condition annotations are
    projected through the actual model-token layout as a cross-check; source
    coordinates are never inspected or guessed here.
    """

    protected = (~design_mask) | identity_mask | carrier_mask
    protected |= mol_type == MOL_TYPE_TO_ID["ligand"]

    for field in _MOTIF_MASK_FIELDS:
        if field not in canonical_batch:
            continue
        mask = _sample_tensor(canonical_batch, field, sample_index)
        # _sample_tensor has removed B, leaving [N] or [N,K].
        mask = mask.detach().bool().cpu()
        if mask.ndim == 1:
            motif = mask
        elif mask.ndim >= 2:
            motif = mask.reshape(mask.shape[0], -1).any(dim=1)
        else:
            raise ValueError(f"canonical motif mask {field!r} has invalid rank")
        if motif.shape != token_mask.shape:
            raise ValueError(
                f"canonical motif mask {field!r} disagrees with token shape"
            )
        protected |= motif

    # Keep compiled condition source authoritative when a caller supplies a
    # hand-built canonical mapping without motif fields. This still uses
    # explicit condition annotations, never coordinate finiteness/distance.
    if compiled is not None and layout is not None:
        atoms = compiled.atoms
        try:
            row_condition = np.asarray(atoms.coordinate_condition, dtype=bool)
        except AttributeError:
            try:
                row_condition = np.asarray(
                    atoms.motif_coord_condition, dtype=bool
                ) | np.asarray(atoms.token_center_condition, dtype=bool)
            except AttributeError:
                row_condition = np.zeros(len(atoms), dtype=bool)
        model_token_by_row = np.asarray(layout.model_token_by_row, dtype=np.int64)
        if row_condition.shape != (len(model_token_by_row),):
            raise ValueError("compiled coordinate-condition metadata has invalid shape")
        token_count = int(token_mask.shape[0])
        if bool(
            ((model_token_by_row < 0) | (model_token_by_row >= token_count)).any()
        ):
            raise ValueError(
                "compiled model-token layout contains an invalid row-to-token mapping"
            )
        coordinate_tokens = np.zeros(int(token_mask.shape[0]), dtype=bool)
        coordinate_tokens[model_token_by_row[row_condition]] = True
        protected |= torch.as_tensor(coordinate_tokens, dtype=torch.bool)

    return protected


def _restricted_ligand_element_id(logits: Tensor) -> int:
    allowed = torch.tensor(DESIGN_LIGAND_ELEMENT_IDS, dtype=torch.long)
    return int(allowed[logits[allowed].argmax()].item())


def _element_cap(element_id: int) -> tuple[int | None, int | None]:
    if not 0 <= int(element_id) < len(ELEMENTS_128):
        return None, None
    allowed = _ALLOWED_VALENCES.get(ELEMENTS_128[int(element_id)])
    if allowed is None:
        return None, None
    return 2 * max(allowed), max(allowed)


def _constraint_enabled(candidate_mask: Tensor, element_type: Tensor) -> list[bool]:
    enabled = [False] * int(candidate_mask.shape[0])
    visited: set[int] = set()
    for token_idx in torch.nonzero(candidate_mask.any(dim=1)).flatten().tolist():
        if token_idx in visited:
            continue
        members = torch.nonzero(candidate_mask[token_idx]).flatten().tolist()
        members.append(token_idx)
        visited.update(members)
        elements = {
            ELEMENTS_128[int(element_type[index])]
            if 0 <= int(element_type[index]) < len(ELEMENTS_128)
            else "UNKNOWN"
            for index in members
        }
        classical = all(
            element in _ALLOWED_VALENCES
            and element not in _NONCLASSICAL_BONDING_ELEMENTS
            for element in elements
        )
        for index in members:
            enabled[index] = classical
    return enabled


def _decode_model_bond_types(
    bond_logits: Tensor,
    candidate_mask: Tensor,
    element_type: Tensor,
) -> Tensor:
    """Copy the current validation ``strategy='valence'`` path exactly."""

    logits = _finite_cpu(bond_logits, name="ligand_bond_logits")
    candidate = candidate_mask.detach().bool().cpu()
    elements = element_type.detach().long().cpu()
    if logits.ndim != 3 or logits.shape[-1] != len(LIGAND_BOND_TYPES):
        raise ValueError("ligand bond logits must be [N,N,5]")
    if candidate.shape != logits.shape[:2] or elements.shape != logits.shape[:1]:
        raise ValueError("ligand bond decoder inputs disagree on token shape")
    if bool(torch.diagonal(candidate).any().item()) or not torch.equal(
        candidate, candidate.transpose(0, 1)
    ):
        raise ValueError("ligand bond candidate mask must be symmetric and off-diagonal")

    symmetric_logits = 0.5 * (logits + logits.transpose(0, 1))
    probability = torch.softmax(symmetric_logits, dim=-1)
    bond_score = 1.0 - probability[..., LIGAND_BOND_TYPES["none"]]
    proposed = symmetric_logits.argmax(dim=-1)
    eligible = (
        torch.triu(candidate, diagonal=1)
        & (proposed != LIGAND_BOND_TYPES["none"])
    )
    accepted = torch.zeros_like(eligible)
    used_valence_x2 = [0] * int(proposed.shape[0])
    used_degree = [0] * int(proposed.shape[0])
    constraint = _constraint_enabled(candidate, elements)
    candidates = torch.nonzero(eligible).tolist()
    candidates.sort(
        key=lambda pair: (-float(bond_score[pair[0], pair[1]]), pair[0], pair[1])
    )
    for token_i, token_j in candidates:
        bond_type = int(proposed[token_i, token_j])
        order_x2 = _MIN_BOND_ORDER_HALF_UNITS[bond_type]
        cap_i, degree_i = _element_cap(int(elements[token_i])) if constraint[token_i] else (None, None)
        cap_j, degree_j = _element_cap(int(elements[token_j])) if constraint[token_j] else (None, None)
        if cap_i is not None and used_valence_x2[token_i] + order_x2 > cap_i:
            continue
        if cap_j is not None and used_valence_x2[token_j] + order_x2 > cap_j:
            continue
        if degree_i is not None and used_degree[token_i] + 1 > degree_i:
            continue
        if degree_j is not None and used_degree[token_j] + 1 > degree_j:
            continue
        accepted[token_i, token_j] = True
        used_valence_x2[token_i] += order_x2
        used_valence_x2[token_j] += order_x2
        used_degree[token_i] += 1
        used_degree[token_j] += 1

    result = torch.zeros_like(proposed, dtype=torch.long)
    result[accepted] = proposed[accepted]
    result = result + result.transpose(0, 1)
    if bool((result[~candidate] != LIGAND_BOND_TYPES["none"]).any().item()):
        raise AssertionError("decoded ligand edge escaped its candidate molecule")
    return result


def _complete_ligand_molecules(compiled: CompiledSample) -> frozenset[int]:
    atoms = compiled.atoms
    complete: set[int] = set()
    for policy in compiled.edge_policies:
        if not policy.complete_graph:
            continue
        if policy.component_uid is not None:
            rows = np.where(atoms.component_uid == int(policy.component_uid))[0]
        else:
            rows = np.where(atoms.molecule_uid == int(policy.molecule_uid))[0]
        molecule_uids = np.unique(atoms.molecule_uid[rows])
        if len(rows) == 0 or len(molecule_uids) != 1:
            raise ValueError("complete ligand edge policy has an invalid molecule scope")
        complete.add(int(molecule_uids[0]))
    return frozenset(complete)


def decode_proposal(
    canonical_batch: Mapping[str, Any],
    generated: Mapping[str, Any],
    decoder_output: Mapping[str, Any],
    *,
    sample_index: int,
    ligand_graph_method: str,
    context_constraint_mode: str,
    generated_token_mask: Tensor,
    compiled: CompiledSample | None = None,
) -> DecodedProposal:
    """Decode one batch row without mutating the authoritative authoring IR."""

    method = str(ligand_graph_method).lower()
    if method not in {"model", "obabel"}:
        raise ValueError("ligand_graph_method must be model or obabel")
    mode = str(context_constraint_mode)
    if mode not in {"rigid", "flex"}:
        raise ValueError("context_constraint_mode must be exactly 'rigid' or 'flex'")

    token_mask_t = _sample_tensor(canonical_batch, "token_mask", sample_index).detach().bool().cpu()
    design_mask_t = _sample_tensor(canonical_batch, "design_token_mask", sample_index).detach().bool().cpu()
    generated_mask_raw = _sample_tensor(
        {"generated_token_mask": generated_token_mask},
        "generated_token_mask",
        sample_index,
    )
    if generated_mask_raw.dtype is not torch.bool:
        raise TypeError("generated_token_mask must be a bool tensor")
    generated_mask_t = generated_mask_raw.detach().cpu()
    mol_type_t = _sample_tensor(canonical_batch, "mol_type_id", sample_index).detach().long().cpu()
    mol_id_t = _sample_tensor(canonical_batch, "mol_id", sample_index).detach().long().cpu()
    identity_t = _sample_tensor(
        canonical_batch, "token_identity_condition_mask", sample_index
    ).detach().bool().cpu()
    target_element_t = _sample_tensor(
        canonical_batch, "ligand_element_type", sample_index
    ).detach().long().cpu()
    shapes = {
        tuple(value.shape)
        for value in (
            token_mask_t,
            design_mask_t,
            generated_mask_t,
            mol_type_t,
            mol_id_t,
            identity_t,
            target_element_t,
        )
    }
    if len(shapes) != 1 or len(next(iter(shapes))) != 1:
        raise ValueError("canonical decode token fields disagree on [N] shape")
    if bool((design_mask_t & ~token_mask_t).any().item()):
        raise ValueError("design mask contains a padded token")
    if bool((generated_mask_t & ~token_mask_t).any().item()):
        raise ValueError("generated token mask contains a padded token")
    expected_generated = design_mask_t if mode == "rigid" else token_mask_t
    if not torch.equal(generated_mask_t, expected_generated):
        raise ValueError(
            f"{mode} generated token mask disagrees with the solver contract"
        )
    token_count = int(token_mask_t.shape[0])
    carrier_mask_t = torch.zeros(token_count, dtype=torch.bool)
    layout = None
    if compiled is not None:
        layout = build_model_token_layout(
            compiled.atoms,
            compiled.modified_polymer_carriers,
        )
        valid_token_count = int(token_mask_t.sum().item())
        if layout.num_model_tokens != valid_token_count:
            raise ValueError(
                "compiled model-token layout disagrees with the canonical token axis"
            )
        carrier_mask_t[: layout.num_model_tokens] = torch.as_tensor(
            layout.carrier_token_mask,
            dtype=torch.bool,
        )
        if bool((carrier_mask_t & ~token_mask_t).any().item()):
            raise ValueError("modified-polymer carrier mask contains a padded token")

    rm_token_type = _decode_rm_token_type(compiled)
    protected_polymer_t: Tensor | None = None
    if rm_token_type:
        protected_polymer_t = _polymer_protected_mask(
            canonical_batch,
            sample_index=sample_index,
            token_mask=token_mask_t,
            design_mask=design_mask_t,
            identity_mask=identity_t,
            mol_type=mol_type_t,
            carrier_mask=carrier_mask_t,
            compiled=compiled,
            layout=layout,
        )

    token_logits = _finite_cpu(
        _sample_tensor(decoder_output, "token_logits32", sample_index),
        name="token_logits32",
    )
    polymer_atom91 = _finite_cpu(
        _sample_tensor(decoder_output, "polymer_atom91_nm", sample_index),
        name="polymer_atom91_nm",
    )
    ligand_logits = _finite_cpu(
        _sample_tensor(decoder_output, "ligand_element_logits", sample_index),
        name="ligand_element_logits",
    )
    token_center = _finite_cpu(
        _sample_tensor(generated, "token_center", sample_index),
        name="token_center",
    )
    if token_logits.shape != (token_count, 32):
        raise ValueError("token logits must be [N,32]")
    if polymer_atom91.shape != (token_count, 91, 3):
        raise ValueError("polymer coordinates must be [N,91,3]")
    if ligand_logits.shape != (token_count, len(ELEMENTS_128)):
        raise ValueError("ligand element logits must be [N,128]")
    if token_center.shape != (token_count, 3):
        raise ValueError("generated token centers must be [N,3]")

    polymer_names: list[str | None] = [None] * token_count
    predicted_element_ids = torch.full((token_count,), -1, dtype=torch.long)
    ligand_elements: list[str | None] = [None] * token_count
    for token_uid in torch.nonzero(token_mask_t).flatten().tolist():
        mol_type_id = int(mol_type_t[token_uid])
        if mol_type_id in _VALID_POLYMER_TOKEN_IDS:
            omission = (
                rm_token_type
                if protected_polymer_t is not None
                and bool(design_mask_t[token_uid])
                and not bool(protected_polymer_t[token_uid])
                else None
            )
            token_type = _restricted_polymer_id(
                token_logits[token_uid], mol_type_id, omission
            )
            polymer_names[token_uid] = ID_TO_TOKEN_TYPE32[token_type]
        elif mol_type_id == MOL_TYPE_TO_ID["ligand"]:
            predicted_id = _restricted_ligand_element_id(ligand_logits[token_uid])
            effective_id = int(target_element_t[token_uid]) if bool(identity_t[token_uid]) else predicted_id
            if not 0 <= effective_id < len(ELEMENTS_128):
                raise ValueError("conditioned ligand element is outside the checkpoint vocabulary")
            predicted_element_ids[token_uid] = effective_id
            ligand_elements[token_uid] = ELEMENTS_128[effective_id]
        else:
            raise ValueError(f"unsupported model modality id {mol_type_id}")

    design_ligand = (
        token_mask_t
        & design_mask_t
        & (mol_type_t == MOL_TYPE_TO_ID["ligand"])
        & ~carrier_mask_t
    )
    if bool(design_ligand.any().item()) and compiled is None:
        raise ValueError("design ligand decode requires the matching CompiledSample")
    complete_molecules = frozenset() if compiled is None else _complete_ligand_molecules(compiled)
    decoded_molecules = frozenset(
        int(value)
        for value in torch.unique(mol_id_t[design_ligand]).tolist()
        if int(value) not in complete_molecules
    )

    ligand_bonds: list[tuple[int, int, int]] = []
    private_molecules: dict[int, Any] = {}
    private_atom_uids: dict[int, tuple[int, ...]] = {}
    if compiled is not None and layout is not None:
        for molecule_uid in sorted(decoded_molecules):
            members = torch.nonzero(
                design_ligand & (mol_id_t == int(molecule_uid))
            ).flatten().tolist()
            private_atom_uids[int(molecule_uid)] = tuple(
                int(compiled.atoms.atom_uid[layout.model_token_to_row[int(member)]])
                for member in members
            )
    if decoded_molecules and method == "model":
        enabled = _sample_tensor(
            decoder_output,
            "ligand_graph_decoder_enabled",
            sample_index,
            scalar_ok=True,
        )
        if not bool(torch.isfinite(enabled.detach()).all().item()) or float(enabled.detach().cpu()) <= 0.5:
            raise ValueError("model ligand graph route requires an enabled graph decoder")
        bond_logits = _sample_tensor(
            decoder_output, "ligand_bond_logits", sample_index
        )
        candidate = torch.zeros((token_count, token_count), dtype=torch.bool)
        for molecule_uid in decoded_molecules:
            members = design_ligand & (mol_id_t == int(molecule_uid))
            candidate |= members[:, None] & members[None, :]
        candidate &= ~torch.eye(token_count, dtype=torch.bool)
        decoded = _decode_model_bond_types(
            bond_logits,
            candidate,
            predicted_element_ids,
        )
        for token_i, token_j in torch.nonzero(
            torch.triu(decoded > 0, diagonal=1)
        ).tolist():
            canonical_type = int(decoded[token_i, token_j])
            ligand_bonds.append(
                (token_i, token_j, _CANONICAL_TO_BIOTITE[canonical_type])
            )
    elif decoded_molecules:
        coords_angstrom = token_center.numpy().astype(np.float64, copy=False) * 10.0
        for molecule_uid in sorted(decoded_molecules):
            members = torch.nonzero(
                design_ligand & (mol_id_t == int(molecule_uid))
            ).flatten().tolist()
            if not members:
                raise AssertionError(molecule_uid)
            elements = [ligand_elements[index] for index in members]
            if any(element is None for element in elements):
                raise ValueError("OpenBabel ligand element decode is incomplete")
            graph, molecule = reconstruct_obabel_with_molecule(
                coords_angstrom[members],
                [str(element) for element in elements],
            )
            private_molecules[int(molecule_uid)] = molecule
            for local_i, local_j, bond_type in graph.bonds:
                ligand_bonds.append(
                    (members[int(local_i)], members[int(local_j)], int(bond_type))
                )

    charge_logits: np.ndarray | None = None
    raw_charge = decoder_output.get("ligand_charge_logits")
    if raw_charge is not None:
        charge_logits = _finite_cpu(
            _sample_tensor(decoder_output, "ligand_charge_logits", sample_index),
            name="ligand_charge_logits",
        ).numpy()
        if charge_logits.shape[:1] != (token_count,):
            raise ValueError("ligand charge logits must retain the token axis")

    return DecodedProposal(
        token_mask=token_mask_t.numpy(),
        design_token_mask=design_mask_t.numpy(),
        generated_token_mask=generated_mask_t.numpy(),
        context_constraint_mode=mode,
        polymer_res_name=tuple(polymer_names),
        token_center_nm=token_center.numpy(),
        polymer_atom91_nm=polymer_atom91.numpy(),
        ligand_element=tuple(ligand_elements),
        ligand_bonds=tuple(sorted(ligand_bonds)),
        decoded_ligand_molecule_uids=decoded_molecules,
        private_ligand_charge_logits=charge_logits,
        private_ligand_molecules=MappingProxyType(private_molecules),
        private_ligand_atom_uids=MappingProxyType(private_atom_uids),
    )


__all__ = ["DecodedProposal", "decode_proposal"]
