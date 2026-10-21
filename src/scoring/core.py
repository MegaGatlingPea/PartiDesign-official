"""Shared PartiScore tensor network, geometry, and stable MDN likelihood.

Coordinates remain in nm; supervised distances and Gaussian densities use Angstrom.
This module does not import the training, dataset, or benchmark runtime.
"""
from __future__ import annotations

import math
from typing import Any, Mapping
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from src.scoring.layers.pair_bias_attn import PairBiasAttention
from src.common.constants import (
    ATOM91_SIZE,
    DNA_ATOM91_OFFSET,
    ELEMENTS_128,
    MOL_TYPE_TO_ID,
    PROTEIN_ATOM91_OFFSET,
    RNA_ATOM91_OFFSET,
    TOKEN_TYPE32,
    PROTEIN_ATOM37,
    RNA_ATOM27,
    DNA_ATOM27,
    element_id,
)

TRAIN_TARGET_CUTOFF_ANGSTROM = 7.0
_STRUCTURE_ATOM_SLOTS = ATOM91_SIZE + 1
_STRUCTURE_RBF_BINS = 16


def distributed_sum(value):
    value = value.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value)
    return value


def require_finite(value, label, identities, synchronize=False):
    bad = (~torch.isfinite(value).all()).to(torch.long)
    if synchronize:
        bad = distributed_sum(bad)
    if bad.item():
        raise FloatingPointError(f"Non-finite {label}; local sample identities: {identities}")


def mdn_log_density(logits, mu, sigma, target):
    """Stable normal-mixture log likelihood; caller selects supervised pairs first."""
    return torch.logsumexp(
        F.log_softmax(logits, -1)
        - sigma.log()
        - 0.5 * math.log(2 * math.pi)
        - 0.5 * ((target[..., None] - mu) / sigma).square(),
        -1,
    )


def _batched(value: torch.Tensor, *, batch_ndim: int) -> torch.Tensor:
    """Return a canonical tensor with an explicit batch dimension."""

    if value.ndim == batch_ndim:
        return value.unsqueeze(0)
    if value.ndim != batch_ndim + 1:
        raise ValueError(
            f"expected tensor with {batch_ndim} or {batch_ndim + 1} dimensions, "
            f"got {value.shape}"
        )
    return value


def _has_heavy_atom_fields(batch: Mapping[str, Any]) -> bool:
    """Whether a batch carries canonical heavy-atom coordinate fields."""

    if "polymer_atom91_coords" in batch and "polymer_atom91_mask" in batch:
        return True
    return any(
        coords in batch and mask in batch
        for coords, mask in (
            ("protein_atom37_coords", "protein_atom37_mask"),
            ("rna_atom27_coords", "rna_atom27_mask"),
            ("dna_atom27_coords", "dna_atom27_mask"),
        )
    )


def _canonical_heavy_atom_geometry(
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack canonical heavy atoms into 91 polymer slots plus one ligand slot.

    The canonical schema stores polymer atoms in a unified 91-slot array. The
    modality-specific fallback keeps compact synthetic fixtures usable while
    preserving the same slot semantics. A ligand token denotes one heavy atom,
    represented in the final slot by its token centre.
    """

    token_mask = _batched(batch["token_mask"].bool(), batch_ndim=1)
    bsz, n_token = token_mask.shape
    device = token_mask.device

    if "polymer_atom91_coords" in batch and "polymer_atom91_mask" in batch:
        polymer_coords = _batched(
            batch["polymer_atom91_coords"].float(), batch_ndim=3
        ).to(device=device, dtype=torch.float32)
        polymer_mask = _batched(
            batch["polymer_atom91_mask"].bool(), batch_ndim=2
        ).to(device=device)
        if polymer_coords.shape[:2] != (bsz, n_token) or polymer_coords.shape[2:] != (
            ATOM91_SIZE,
            3,
        ):
            raise ValueError(
                "polymer_atom91_coords has incompatible shape: "
                f"{polymer_coords.shape}, expected {(bsz, n_token, ATOM91_SIZE, 3)}"
            )
        if polymer_mask.shape != (bsz, n_token, ATOM91_SIZE):
            raise ValueError(
                "polymer_atom91_mask has incompatible shape: "
                f"{polymer_mask.shape}, expected {(bsz, n_token, ATOM91_SIZE)}"
            )
        coords = torch.zeros(
            bsz,
            n_token,
            _STRUCTURE_ATOM_SLOTS,
            3,
            dtype=torch.float32,
            device=device,
        )
        atom_mask = torch.zeros(
            bsz,
            n_token,
            _STRUCTURE_ATOM_SLOTS,
            dtype=torch.bool,
            device=device,
        )
        coords[..., :ATOM91_SIZE, :] = polymer_coords
        atom_mask[..., :ATOM91_SIZE] = polymer_mask
    else:
        coords = torch.zeros(
            bsz,
            n_token,
            _STRUCTURE_ATOM_SLOTS,
            3,
            dtype=torch.float32,
            device=device,
        )
        atom_mask = torch.zeros(
            bsz,
            n_token,
            _STRUCTURE_ATOM_SLOTS,
            dtype=torch.bool,
            device=device,
        )
        mol_type = _batched(batch["mol_type_id"].long(), batch_ndim=1).to(device)
        modality_fields = (
            (
                MOL_TYPE_TO_ID["protein"],
                "protein_atom37_coords",
                "protein_atom37_mask",
                PROTEIN_ATOM91_OFFSET,
            ),
            (
                MOL_TYPE_TO_ID["rna"],
                "rna_atom27_coords",
                "rna_atom27_mask",
                RNA_ATOM91_OFFSET,
            ),
            (
                MOL_TYPE_TO_ID["dna"],
                "dna_atom27_coords",
                "dna_atom27_mask",
                DNA_ATOM91_OFFSET,
            ),
        )
        for modality, coords_name, mask_name, offset in modality_fields:
            if coords_name not in batch or mask_name not in batch:
                continue
            local_coords = _batched(
                batch[coords_name].float(), batch_ndim=3
            ).to(device=device, dtype=torch.float32)
            local_mask = _batched(
                batch[mask_name].bool(), batch_ndim=2
            ).to(device=device)
            slot_count = local_mask.shape[-1]
            if local_coords.shape[:2] != (bsz, n_token) or local_coords.shape[-2:] != (
                slot_count,
                3,
            ):
                raise ValueError(f"{coords_name} has incompatible shape {local_coords.shape}")
            selected = (
                local_mask
                & token_mask[..., None].to(device)
                & (mol_type == modality)[..., None]
            )
            coords[..., offset : offset + slot_count, :] = torch.where(
                selected[..., None],
                local_coords,
                torch.zeros_like(local_coords),
            )
            atom_mask[..., offset : offset + slot_count] = selected

        if not _has_heavy_atom_fields(batch):
            raise ValueError(
                "PartiScore structure input requires canonical heavy atom "
                "coordinates and masks"
            )

    atom_mask = atom_mask.clone()
    atom_mask &= token_mask[..., None].to(device)
    finite = torch.isfinite(coords).all(dim=-1)
    if bool((atom_mask & ~finite).any().item()):
        raise ValueError("canonical heavy atom coordinates are non-finite")
    coords = torch.where(
        finite[..., None],
        coords,
        torch.zeros_like(coords),
    )

    mol_type = _batched(batch["mol_type_id"].long(), batch_ndim=1).to(device)
    ligand_mask = _batched(
        batch.get(
            "ligand_mask",
            mol_type == MOL_TYPE_TO_ID["ligand"],
        ).bool(),
        batch_ndim=1,
    ).to(device)
    if "token_center_coords_nm" not in batch:
        raise KeyError("PartiScore requires token_center_coords_nm")
    centers = _batched(
        batch["token_center_coords_nm"].float(), batch_ndim=2
    ).to(device=device, dtype=torch.float32)
    ligand_atoms = ligand_mask & token_mask
    if bool(
        (ligand_atoms & ~torch.isfinite(centers).all(dim=-1)).any().item()
    ):
        raise ValueError("ligand token centres are non-finite")
    coords[..., ATOM91_SIZE, :] = torch.where(
        ligand_atoms[..., None],
        centers,
        torch.zeros_like(centers),
    )
    atom_mask[..., ATOM91_SIZE] = ligand_atoms
    return coords, atom_mask


def _minimum_heavy_atom_distances(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Compute pairwise minimum heavy-atom distances without a 4-D batch tensor."""

    if indices[0].numel() == 0:
        return coords.new_empty((0,), dtype=torch.float32)
    b, i, j = indices
    result: list[torch.Tensor] = []
    for start in range(0, int(b.numel()), int(chunk_size)):
        sl = slice(start, start + int(chunk_size))
        left = coords[b[sl], i[sl]]
        right = coords[b[sl], j[sl]]
        left_mask = atom_mask[b[sl], i[sl]]
        right_mask = atom_mask[b[sl], j[sl]]
        distances = torch.cdist(left, right)
        valid_atoms = left_mask[..., :, None] & right_mask[..., None, :]
        distances = distances.masked_fill(~valid_atoms, float("inf"))
        result.append(distances.amin(dim=(-2, -1)).mul_(10.0).float())
    return torch.cat(result, dim=0)


def supervision_pairs(
    batch,
    cutoff_angstrom=TRAIN_TARGET_CUTOFF_ANGSTROM,
    chunk_size=1024,
):
    valid = _batched(batch["token_mask"].bool(), batch_ndim=1)
    if "source_mol_id" not in batch:
        raise ValueError("PartiScore requires source_mol_id for every real token")
    mol = _batched(batch["source_mol_id"].long(), batch_ndim=1)
    if ((mol < 0) & valid).any():
        raise ValueError("PartiScore requires valid source_mol_id for every real token")
    centers = _batched(batch["token_center_coords_nm"].float(), batch_ndim=2)
    require_finite(centers[valid], "token centers", batch.get("partiscore_id", "unknown"))
    mask = valid[:, :, None] & valid[:, None, :]
    mask &= mol[:, :, None] != mol[:, None, :]
    mask &= torch.ones_like(mask).triu(1)
    coords, atom_mask = _canonical_heavy_atom_geometry(batch)
    token_has_atom = atom_mask.any(dim=-1)
    mask &= token_has_atom[:, :, None] & token_has_atom[:, None, :]
    candidate_indices = mask.nonzero(as_tuple=True)
    targets = _minimum_heavy_atom_distances(
        coords,
        atom_mask,
        candidate_indices,
        chunk_size=max(1, int(chunk_size)),
    )
    keep = targets <= float(cutoff_angstrom)
    return tuple(index[keep] for index in candidate_indices), targets[keep]


class ScoreBlock(nn.Module):
    def __init__(self, node_dim, pair_dim, heads, dropout):
        super().__init__()
        self.attention = PairBiasAttention(
            node_dim,
            node_dim // heads,
            heads,
            True,
            node_dim,
            True,
            pair_dim,
        )
        self.node_ffn = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 4, node_dim),
        )
        self.left = nn.Linear(node_dim, pair_dim)
        self.right = nn.Linear(node_dim, pair_dim)
        self.pair_ffn = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, pair_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim * 2, pair_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, node, pair, valid):
        mask = valid[:, :, None] & valid[:, None, :]
        node = node + self.dropout(self.attention(node, pair, mask))
        node = (node + self.dropout(self.node_ffn(node))) * valid[..., None]
        pair = pair + self.dropout(self.left(node)[:, :, None] + self.right(node)[:, None, :])
        pair = (pair + self.dropout(self.pair_ffn(pair))) * mask[..., None]
        return node, pair


class StructureTokenEncoder(nn.Module):
    """Encode canonical atom identity and within-token invariant geometry."""

    def __init__(
        self,
        node_dim: int,
        *,
        atom_dim: int = 64,
        rbf_bins: int = _STRUCTURE_RBF_BINS,
        atom_chunk_size: int = 1024,
    ):
        super().__init__()
        self.atom_dim = int(atom_dim)
        self.atom_chunk_size = int(atom_chunk_size)
        self.atom_slots = _STRUCTURE_ATOM_SLOTS
        self.token_identity = nn.Embedding(len(TOKEN_TYPE32), self.atom_dim)
        self.atom_slot = nn.Embedding(self.atom_slots, self.atom_dim)
        self.atom_element = nn.Embedding(len(ELEMENTS_128), self.atom_dim)
        self.register_buffer('slot_element_ids', torch.tensor(
            [element_id(name[0]) for name in
             (*PROTEIN_ATOM37, *RNA_ATOM27, *DNA_ATOM27)] + [element_id('UNK_ELEM_128')],
            dtype=torch.long))
        self.atom_modality = nn.Embedding(4, self.atom_dim)
        self.atom_input = nn.Sequential(
            nn.Linear(self.atom_dim + int(rbf_bins) + 1, self.atom_dim * 2),
            nn.SiLU(),
            nn.Linear(self.atom_dim * 2, self.atom_dim),
        )
        self.to_node = nn.Sequential(
            nn.LayerNorm(self.atom_dim),
            nn.Linear(self.atom_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.register_buffer(
            "rbf_centers",
            torch.linspace(0.0, 2.0, int(rbf_bins), dtype=torch.float32),
        )
        self.rbf_width = 0.15

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        coords, atom_mask = _canonical_heavy_atom_geometry(batch)
        bsz, n_token, n_slot, _ = coords.shape
        if n_slot != self.atom_slots:
            raise ValueError(f"unexpected structure atom slot count {n_slot}")
        token_mask = _batched(batch["token_mask"].bool(), batch_ndim=1).to(coords.device)
        token_type = _batched(
            batch.get("token_type32", torch.zeros_like(token_mask, dtype=torch.long)).long(),
            batch_ndim=1,
        ).to(coords.device).clamp(min=0, max=len(TOKEN_TYPE32) - 1)
        mol_type = _batched(batch["mol_type_id"].long(), batch_ndim=1).to(coords.device).clamp(
            min=0, max=3
        )
        ligand_element = _batched(
            batch.get(
                "ligand_element_type",
                torch.zeros_like(token_type),
            ).long(),
            batch_ndim=1,
        ).to(coords.device).clamp(min=0, max=len(ELEMENTS_128) - 1)
        slots = torch.arange(n_slot, device=coords.device).view(1, 1, n_slot)
        slots = slots.expand(bsz, n_token, n_slot)
        elements = self.slot_element_ids.view(1, 1, n_slot).expand(bsz, n_token, n_slot).clone()
        elements[..., -1] = ligand_element
        identity = (
            self.token_identity(token_type)[..., None, :]
            + self.atom_slot(slots)
            + self.atom_element(elements)
            + self.atom_modality(mol_type)[..., None, :]
        )
        flat_coords = coords.reshape(-1, n_slot, 3)
        flat_mask = atom_mask.reshape(-1, n_slot)
        flat_rbf = coords.new_zeros(
            flat_coords.shape[0], n_slot, self.rbf_centers.numel()
        )
        for start in range(0, flat_coords.shape[0], self.atom_chunk_size):
            stop = min(start + self.atom_chunk_size, flat_coords.shape[0])
            local_coords = flat_coords[start:stop]
            local_mask = flat_mask[start:stop]
            safe_coords = torch.where(
                local_mask[..., None],
                local_coords,
                torch.zeros_like(local_coords),
            )
            distances = torch.cdist(safe_coords, safe_coords)
            pair_mask = local_mask[..., :, None] & local_mask[..., None, :]
            rbf = torch.exp(
                -(
                    (
                        distances[..., None]
                        - self.rbf_centers.to(
                            device=distances.device,
                            dtype=distances.dtype,
                        )
                    )
                    / self.rbf_width
                ).square()
            )
            neighbour_count = pair_mask.sum(dim=-2).clamp_min(1).to(rbf.dtype)
            flat_rbf[start:stop] = (
                (rbf * pair_mask[..., None].to(rbf.dtype)).sum(dim=-2)
                / neighbour_count[..., None]
            )
        atom_input = torch.cat(
            (
                identity,
                flat_rbf.reshape(bsz, n_token, n_slot, -1),
                atom_mask[..., None].to(coords.dtype),
            ),
            dim=-1,
        )
        atom_hidden = self.atom_input(atom_input) * atom_mask[..., None].to(coords.dtype)
        pooled = atom_hidden.sum(dim=-2) / atom_mask.sum(dim=-1).clamp_min(1).to(
            atom_hidden.dtype
        )[..., None]
        node = self.to_node(pooled)
        return node * token_mask[..., None].to(node.dtype)


class PartiScoreNetwork(nn.Module):
    def __init__(
        self,
        latent_dim=8,
        node_dim=256,
        pair_dim=128,
        layers=6,
        heads=8,
        dropout=0.1,
        components=10,
        readout_chunk_size=65536,
        input_mode="structure",
    ):
        super().__init__()
        if node_dim % heads or components != 10:
            raise ValueError("node_dim must divide heads; v2 uses ten Gaussian components")
        self.input_mode = str(input_mode).lower()
        if self.input_mode not in {"latent", "structure"}:
            raise ValueError(
                f"PartiScore model.input_mode must be latent|structure, got {input_mode!r}"
            )
        if self.input_mode == "latent":
            self.node_input = nn.Sequential(
                nn.LayerNorm(latent_dim), nn.Linear(latent_dim, node_dim)
            )
            self.structure_input = None
        else:
            self.node_input = None
            self.structure_input = StructureTokenEncoder(node_dim)
        self.modality = nn.Embedding(4, node_dim)
        # Equality2 + ordered modality16 + signed separation129 + bond5 + distance30.
        self.pair_input = nn.Linear(182, pair_dim)
        self.left = nn.Linear(node_dim, pair_dim)
        self.right = nn.Linear(node_dim, pair_dim)
        self.blocks = nn.ModuleList(
            [ScoreBlock(node_dim, pair_dim, heads, dropout) for _ in range(layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(node_dim * 2 + pair_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, components * 3),
        )
        self.readout_chunk_size = int(readout_chunk_size)
        self.register_buffer(
            "distance_boundaries", torch.linspace(0.1, 3.0, 29)
        )
        with torch.no_grad():
            nn.init.normal_(self.readout[-1].weight, std=1e-3)
            self.readout[-1].bias[:10].zero_()
            self.readout[-1].bias[10:20].copy_(
                torch.expm1(torch.linspace(1, 9, 10)).log()
            )
            self.readout[-1].bias[20:].fill_(math.log(math.expm1(1.9)))

    def pair_features(self, batch):
        mask = _batched(batch["token_mask"].bool(), batch_ndim=1)
        mol = _batched(batch["source_mol_id"].long(), batch_ndim=1).to(mask.device)
        if ((mol < 0) & mask).any():
            raise ValueError("Invalid source molecule ID")
        same_mol = mol[:, :, None] == mol[:, None, :]
        chain = _batched(batch["asym_id"].long(), batch_ndim=1).to(mask.device)
        same_chain = chain[:, :, None] == chain[:, None, :]
        types = _batched(batch["mol_type_id"].long(), batch_ndim=1).to(mask.device)
        ordered_types = types[:, :, None] * 4 + types[:, None, :]
        poly = types != MOL_TYPE_TO_ID["ligand"]
        seq = _batched(batch["residue_index"].long(), batch_ndim=1).to(mask.device)
        delta = (seq[:, None, :] - seq[:, :, None]).clamp(-63, 63) + 63
        delta = torch.where(same_chain, delta, 127)
        delta = torch.where(poly[:, :, None] & poly[:, None, :], delta, 128)
        centers = _batched(
            batch["token_center_coords_nm"].float(), batch_ndim=2
        ).to(mask.device)
        require_finite(
            centers[mask], "token centers", batch.get("partiscore_id", "unknown")
        )
        safe_centers = torch.where(mask[..., None], centers, torch.zeros_like(centers))
        distances = torch.cdist(safe_centers, safe_centers)
        distance = F.one_hot(
            torch.bucketize(distances, self.distance_boundaries), 30
        ).float()
        distance = distance * same_mol[..., None]
        bonds = _batched(batch["token_bond_type"].long(), batch_ndim=2).to(mask.device)
        # Canonical non-edge is zero and edge classes are single/double/triple/aromatic.
        bonds = torch.where(
            _batched(batch["token_bond_pair_mask"].bool(), batch_ndim=2).to(mask.device),
            bonds,
            0,
        )
        raw = torch.cat(
            [
                same_mol[..., None],
                same_chain[..., None],
                F.one_hot(ordered_types, 16),
                F.one_hot(delta, 129),
                F.one_hot(bonds, 5),
                distance,
            ],
            -1,
        ).float()
        return raw * (mask[:, :, None] & mask[:, None, :])[..., None]

    def forward(self, latent=None, batch=None):
        if batch is None:
            batch = latent
            latent = None
        valid = _batched(batch["token_mask"].bool(), batch_ndim=1)
        types = _batched(batch["mol_type_id"].long(), batch_ndim=1).to(valid.device)
        if self.input_mode == "latent":
            if latent is None:
                raise ValueError("latent input is required when input_mode='latent'")
            node_input = self.node_input(latent)
        else:
            node_input = self.structure_input(batch)
        node = (node_input + self.modality(types)) * valid[..., None]
        pair = self.pair_input(self.pair_features(batch))
        pair = pair + self.left(node)[:, :, None] + self.right(node)[:, None, :]
        for block in self.blocks:
            node, pair = block(node, pair, valid)
        return node, pair

    def mixture(self, node, pair, indices):
        b, i, j = indices
        with torch.autocast(device_type=node.device.type, enabled=False):
            hi, hj = node[b, i].float(), node[b, j].float()
            p = (pair[b, i, j].float() + pair[b, j, i].float()) * 0.5
            raw = self.readout(torch.cat([hi + hj, (hi - hj).abs(), p], -1))
            logits, mu, sigma = raw.chunk(3, -1)
            return logits.float(), F.softplus(mu).float(), (F.softplus(sigma) + 0.1).float()

    def likelihood(self, node, pair, indices, target):
        logs, sigmas = [], []
        for start in range(0, len(target), self.readout_chunk_size):
            sl = slice(start, start + self.readout_chunk_size)
            params = self.mixture(node, pair, tuple(v[sl] for v in indices))
            for value in params:
                require_finite(value, "MDN parameters", "selected pair chunk")
            logs.append(mdn_log_density(*params, target[sl].float()).float())
            sigmas.append(params[2])
        if not logs:
            return node.new_empty(0, dtype=torch.float32), node.new_empty(
                (0, 10), dtype=torch.float32
            )
        return torch.cat(logs).float(), torch.cat(sigmas).float()
