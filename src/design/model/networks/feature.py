import math
from collections.abc import Mapping
from typing import Any, Callable

import torch
from loguru import logger
from torch.nn import functional as F

from src.design.model.networks.feature_base import (
    Feature,
    FeatureMode,
    FeatureRegistryName,
    bin_and_one_hot,
    flow_float_dtype,
    flow_pair_mask,
    flow_token_mask,
)
from src.design.model.networks.lig_feature import (
    ConditionLigandElementSeqFeat,
    LigandChargeSeqFeat,
    LigandCoordSeqFeat,
    LigandDegreeSeqFeat,
    LigandElementSeqFeat,
    LigandMotifSeqFeat,
    TokenBondPairFeat,
)
from src.design.model.networks.nuc_feature import (
    ConditionDNATokenTypeSeqFeat,
    ConditionRNATokenTypeSeqFeat,
    DNAAtom27CoordsSeqFeat,
    DNAAtom27MotifSeqFeat,
    DNAGroupMaskSeqFeat,
    DNATokenTypeSeqFeat,
    DNATorsionAnglesSeqFeat,
    RNAAtom27CoordsSeqFeat,
    RNAAtom27MotifSeqFeat,
    RNAGroupMaskSeqFeat,
    RNATokenTypeSeqFeat,
    RNATorsionAnglesSeqFeat,
)
from src.design.model.networks.prot_feature import (
    ConditionProteinTokenTypeSeqFeat,
    ProteinAtom37CoordsSeqFeat,
    ProteinAtom37MotifSeqFeat,
    ProteinBackboneTorsionAnglesSeqFeat,
    ProteinSideChainAnglesSeqFeat,
    ProteinTokenTypeSeqFeat,
)
from src.design.model.networks.template import TemplatePairFeat, TemplateSingleSeqFeat
from src.design.model.constants import (
    CONDITION_TOKEN_BOND_ENDPOINT_VOCAB_SIZE,
    CONDITION_TOKEN_BOND_PAIR_DIM,
    MOL_TYPE_TO_ID,
)


def get_time_embedding(
    t: torch.Tensor, edim: int, max_positions: int = 2000
) -> torch.Tensor:
    assert t.ndim == 1, f"Expected time shape [B], got {tuple(t.shape)}"
    t = t * max_positions
    half_dim = edim // 2
    emb_scale = math.log(max_positions) / (half_dim - 1)
    frequencies = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb_scale
    )
    phases = t.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)
    if edim % 2 == 1:
        embedding = F.pad(embedding, (0, 1), mode="constant")
    assert embedding.shape == (t.shape[0], edim)
    return embedding


def bin_pairwise_distances(
    coords: torch.Tensor,
    min_dist: float,
    max_dist: float,
    dim: int,
    *, floating: bool = False,
) -> torch.Tensor:
    pair_dists_nm = torch.linalg.vector_norm(
        coords[:, :, None, :] - coords[:, None, :, :], dim=-1
    )
    limits = torch.linspace(min_dist, max_dist, dim - 1, device=coords.device)
    return bin_and_one_hot(pair_dists_nm, limits, floating=floating)


def _masked_coords(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask[..., None], coords, torch.zeros_like(coords))


def _expand_token_flag(flag: Any, token_mask: torch.Tensor, name: str) -> torch.Tensor:
    batch_size, n_token = token_mask.shape
    if isinstance(flag, bool):
        return torch.full_like(token_mask, flag)
    if not isinstance(flag, torch.Tensor):
        raise TypeError(f"`{name}` must be a bool or tensor")
    flag = flag.to(device=token_mask.device).bool()
    if flag.ndim == 1 and flag.shape == (batch_size,):
        return flag[:, None].expand(batch_size, n_token)
    if flag.ndim == 2 and flag.shape == (batch_size, 1):
        return flag.expand(batch_size, n_token)
    if flag.shape == token_mask.shape:
        return flag
    raise AssertionError(
        f"`{name}` must have shape [B], [B,1], or [B,N], got {tuple(flag.shape)}"
    )


class ZeroFeat(Feature):
    def __init__(self, dim_feats_out: int, mode: FeatureMode, **kwargs: Any):
        super().__init__(dim=dim_feats_out)
        self.mode = mode

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        batch_size, n_token = self.extract_bs_and_n(batch)
        shape = (
            (batch_size, n_token, self.dim)
            if self.mode == "seq"
            else (batch_size, n_token, n_token, self.dim)
        )
        return torch.zeros(
            shape, device=self.extract_device(batch), dtype=flow_float_dtype(batch)
        )


# Current universal VAE features. Their names, dimensions, and math are checkpoint ABI.


class ChainBreakPerResidueSeqFeat(Feature):
    required_keys = frozenset(("chain_break_after_token",))

    def __init__(self, **kwargs: Any):
        super().__init__(dim=1)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        assert (
            "chain_break_after_token" in batch
        ), "`chain_break_after_token` not in batch"
        return batch["chain_break_after_token"].to(dtype=torch.float32)[..., None]


class LatentVariableSeqFeat(Feature):
    required_keys = frozenset(("z_latent",))

    def __init__(self, latent_z_dim: int, **kwargs: Any):
        super().__init__(dim=latent_z_dim)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        assert "z_latent" in batch, "`z_latent` not in batch"
        return batch["z_latent"]


class MolTypeSeqFeat(Feature):
    required_keys = frozenset(("mol_type_id",))

    def __init__(self, **kwargs: Any):
        super().__init__(dim=len(MOL_TYPE_TO_ID))

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        assert "mol_type_id" in batch, "`mol_type_id` not in batch"
        mol_type = batch["mol_type_id"].long().clamp(min=0, max=self.dim - 1)
        return F.one_hot(mol_type, num_classes=self.dim).to(dtype=torch.float32)


class TokenCenterCoordsNmSeqFeat(Feature):
    required_keys = frozenset(("token_center_coords_nm",))

    def __init__(self, **kwargs: Any):
        super().__init__(dim=3)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        assert (
            "token_center_coords_nm" in batch
        ), "`token_center_coords_nm` not in batch"
        return batch["token_center_coords_nm"]


class SequenceSeparationPairFeat(Feature):
    required_keys = frozenset(("residue_index", "asym_id"))

    def __init__(self, seq_sep_dim: int, **kwargs: Any):
        super().__init__(dim=seq_sep_dim)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        assert "residue_index" in batch, "`residue_index` not in batch"
        assert "asym_id" in batch, "`asym_id` not in batch"
        indices = batch["residue_index"]
        seq_sep = (indices[:, :, None] - indices[:, None, :]).float()
        chains = batch["asym_id"]
        same_chain = chains[:, :, None] == chains[:, None, :]
        cross_chain_value = self.dim // 2 - 1 + 10
        seq_sep = torch.where(
            same_chain,
            seq_sep,
            torch.full_like(seq_sep, float(cross_chain_value)),
        )

        assert self.dim % 2 == 1 and self.dim > 3
        limits = torch.linspace(
            -(self.dim / 2.0 - 1),
            self.dim / 2.0 - 1,
            self.dim - 1,
            device=indices.device,
        )
        return bin_and_one_hot(seq_sep, limits)


class RelativeChainIndexPairFeat(Feature):
    required_keys = frozenset(("asym_id",))

    def __init__(self, **kwargs: Any):
        super().__init__(dim=3)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        assert "asym_id" in batch, "`asym_id` not in batch"
        asym_id = batch["asym_id"]
        rel_chain = torch.sign(asym_id[:, :, None] - asym_id[:, None, :]).long()
        return F.one_hot(rel_chain + 1, num_classes=3).float()


class MolTypePairFeat(Feature):
    required_keys = frozenset(("mol_type_id",))

    def __init__(self, **kwargs: Any):
        self.num_mol_types = len(MOL_TYPE_TO_ID)
        super().__init__(dim=self.num_mol_types**2)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        mol_i = batch["mol_type_id"][:, :, None].long().clamp(0, self.num_mol_types - 1)
        mol_j = batch["mol_type_id"][:, None, :].long().clamp(0, self.num_mol_types - 1)
        return F.one_hot(mol_i * self.num_mol_types + mol_j, num_classes=self.dim).to(
            dtype=torch.float32
        )


class SameMetaPairFeat(Feature):
    required_keys = frozenset(
        ("token_center_coords_nm", "asym_id", "mol_id", "mol_type_id")
    )

    def __init__(self, **kwargs: Any):
        super().__init__(dim=3)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        dtype = batch["token_center_coords_nm"].dtype
        asym = batch["asym_id"]
        mol = batch["mol_id"]
        mol_type = batch["mol_type_id"]
        return torch.stack(
            [
                (asym[:, :, None] == asym[:, None, :]).to(dtype),
                (mol[:, :, None] == mol[:, None, :]).to(dtype),
                (mol_type[:, :, None] == mol_type[:, None, :]).to(dtype),
            ],
            dim=-1,
        )


class TokenCenterPairDistancePairFeat(Feature):
    required_keys = frozenset(("token_center_coords_nm",))

    def __init__(
        self,
        token_center_pair_dist_dim: int = 30,
        token_center_pair_dist_min: float = 0.1,
        token_center_pair_dist_max: float = 3.0,
        **kwargs: Any,
    ):
        super().__init__(dim=token_center_pair_dist_dim)
        self.min_dist = float(token_center_pair_dist_min)
        self.max_dist = float(token_center_pair_dist_max)
        assert self.dim > 1 and self.max_dist > self.min_dist

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        return bin_pairwise_distances(
            batch["token_center_coords_nm"], self.min_dist, self.max_dist, self.dim
        )


class TokenFrameOrientationPairFeat(Feature):
    required_keys = frozenset(
        ("token_center_coords_nm", "token_frame", "token_frame_mask")
    )

    def __init__(self, **kwargs: Any):
        super().__init__(dim=12)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        center = batch["token_center_coords_nm"]
        frames = batch["token_frame"]
        frame_mask = batch["token_frame_mask"]
        delta = center[:, None, :, :] - center[:, :, None, :]
        dist = delta.norm(dim=-1).clamp_min(1e-6)
        local_delta = (
            torch.einsum("bnij,bnmj->bnmi", frames.transpose(-1, -2), delta)
            / dist[..., None]
        )
        frame_dot = torch.einsum("bnia,bmja->bnmij", frames, frames).flatten(
            start_dim=-2
        )
        orient_mask = (frame_mask[:, :, None] & frame_mask[:, None, :]).to(
            center.dtype
        )[..., None]
        return torch.cat([local_delta, frame_dot], dim=-1) * orient_mask


# Flow-only shared sequence features.


class FlowChainBreakSeqFeat(Feature):
    required_keys = frozenset(("token_mask", "chain_break_after_token"))

    def __init__(self, **kwargs: Any):
        super().__init__(dim=1)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        mask = flow_token_mask(batch)
        value = batch["chain_break_after_token"]
        assert value.shape == mask.shape
        return (value.bool() & mask).to(flow_float_dtype(batch))[..., None]


class FlowMolTypeSeqFeat(Feature):
    required_keys = frozenset(("token_mask", "mol_type_id"))

    def __init__(self, **kwargs: Any):
        super().__init__(dim=len(MOL_TYPE_TO_ID))

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        mask = flow_token_mask(batch)
        mol_type = batch["mol_type_id"].long()
        assert mol_type.shape == mask.shape
        safe_type = torch.where(mask, mol_type, torch.zeros_like(mol_type))
        return (
            self.one_hot(safe_type, self.dim, flow_float_dtype(batch))
            * mask[..., None]
        )


class FlowXtTokenCenterSeqFeat(Feature):
    required_keys = frozenset(("token_mask", "x_t"))

    def __init__(self, **kwargs: Any):
        super().__init__(dim=3)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        coords = batch["x_t"]["token_center"]
        mask = flow_token_mask(batch)
        assert coords.shape == mask.shape + (3,)
        return _masked_coords(coords, mask)


class FlowXtLocalLatentsSeqFeat(Feature):
    required_keys = frozenset(("token_mask", "x_t"))

    def __init__(self, latent_z_dim: int, **kwargs: Any):
        super().__init__(dim=latent_z_dim)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        latents = batch["x_t"]["local_latents"]
        mask = flow_token_mask(batch)
        assert latents.shape == mask.shape + (self.dim,)
        return _masked_coords(latents, mask)


class FlowXscTokenCenterSeqFeat(Feature):
    required_keys = frozenset(("token_mask", "x_sc", "x_sc_valid"))

    def __init__(self, **kwargs: Any):
        super().__init__(dim=3)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token_mask = flow_token_mask(batch)
        valid = (
            _expand_token_flag(batch["x_sc_valid"], token_mask, "x_sc_valid")
            & token_mask
        )
        coords = batch["x_sc"]["token_center"].detach()
        assert coords.shape == token_mask.shape + (3,)
        return _masked_coords(coords, valid)


class FlowXscLocalLatentsSeqFeat(Feature):
    required_keys = frozenset(("token_mask", "x_sc", "x_sc_valid"))

    def __init__(self, latent_z_dim: int, **kwargs: Any):
        super().__init__(dim=latent_z_dim)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token_mask = flow_token_mask(batch)
        valid = (
            _expand_token_flag(batch["x_sc_valid"], token_mask, "x_sc_valid")
            & token_mask
        )
        latents = batch["x_sc"]["local_latents"].detach()
        assert latents.shape == token_mask.shape + (self.dim,)
        return _masked_coords(latents, valid)


class FlowOptionalTokenCenterSeqFeat(Feature):
    required_keys = frozenset(
        ("token_mask", "optional_token_center_coords_nm", "optional_token_center_mask")
    )

    def __init__(self, **kwargs: Any):
        super().__init__(dim=4)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token_mask = flow_token_mask(batch)
        known = batch["optional_token_center_mask"].bool()
        assert known.shape == token_mask.shape
        coords = batch["optional_token_center_coords_nm"]
        assert coords.shape == token_mask.shape + (3,)
        return torch.cat(
            [_masked_coords(coords, known), known.to(coords.dtype)[..., None]], dim=-1
        )


class FlowHotspotSeqFeat(Feature):
    required_keys = frozenset(("token_mask", "hotspot_mask"))

    def __init__(self, **kwargs: Any):
        super().__init__(dim=1)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token_mask = flow_token_mask(batch)
        hotspot = batch["hotspot_mask"].bool()
        assert hotspot.shape == token_mask.shape
        return hotspot.to(flow_float_dtype(batch))[..., None]


class FlowPolymerSSTokenSeqFeat(Feature):
    required_keys = frozenset(
        (
            "token_mask",
            "mol_type_id",
            "polymer_ss_token",
            "polymer_ss_token_condition_mask",
        )
    )

    def __init__(self, **kwargs: Any):
        super().__init__(dim=6)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token_mask = flow_token_mask(batch)
        condition_mask = batch["polymer_ss_token_condition_mask"].bool()
        state = batch["polymer_ss_token"].long()
        assert condition_mask.shape == state.shape == token_mask.shape
        mol_type = batch["mol_type_id"]
        assert mol_type.shape == token_mask.shape
        protein = mol_type == MOL_TYPE_TO_ID["protein"]
        nucleic = (mol_type == MOL_TYPE_TO_ID["rna"]) | (
            mol_type == MOL_TYPE_TO_ID["dna"]
        )
        valid_polymer = token_mask & (protein | nucleic)
        visible = condition_mask
        safe_state = torch.where(visible, state, torch.zeros_like(state))
        channels = (
            valid_polymer & ~visible,
            visible & protein & (safe_state == 1),
            visible & protein & (safe_state == 2),
            visible & protein & (safe_state == 3),
            visible & nucleic & (safe_state == 1),
            visible & nucleic & (safe_state == 2),
        )
        return torch.stack(channels, dim=-1).to(flow_float_dtype(batch))


class FlowPolymerCoarseSSSeqFeat(Feature):
    required_keys = frozenset(
        ("token_mask", "design_token_mask", "mol_type_id", "polymer_coarse_ss")
    )

    def __init__(self, polymer_coarse_ss_emb_dim: int = 64, **kwargs: Any):
        super().__init__(dim=int(polymer_coarse_ss_emb_dim))
        assert self.dim > 0
        self.embedding = torch.nn.Embedding(3, self.dim)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token_mask = flow_token_mask(batch)
        design_mask = batch["design_token_mask"].bool()
        mol_type = batch["mol_type_id"].long()
        state = batch["polymer_coarse_ss"].long()
        assert design_mask.shape == mol_type.shape == state.shape == token_mask.shape
        design_polymer = token_mask & design_mask & (
            mol_type != MOL_TYPE_TO_ID["ligand"]
        )
        safe_state = torch.where(design_polymer, state, torch.zeros_like(state))
        return self.embedding(safe_state) * design_polymer[..., None].to(
            flow_float_dtype(batch)
        )


class FlowTimeEmbeddingSeqFeat(Feature):
    required_keys = frozenset(("token_mask", "t"))

    def __init__(self, component: str, t_emb_dim: int, **kwargs: Any):
        super().__init__(dim=t_emb_dim)
        self.component = component

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        time = batch["t"][self.component]
        _, n_token = flow_token_mask(batch).shape
        embedding = get_time_embedding(time, self.dim).to(flow_float_dtype(batch))
        return embedding[:, None, :].expand(-1, n_token, -1)


# Flow-only shared pair features.


class FlowHotspotPairFeat(Feature):
    required_keys = frozenset(("token_mask", "hotspot_mask"))

    def __init__(self, **kwargs: Any):
        super().__init__(dim=1)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token_mask = flow_token_mask(batch)
        hotspot = batch["hotspot_mask"].bool()
        assert hotspot.shape == token_mask.shape
        pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
        pair_hotspot = hotspot[:, :, None] | hotspot[:, None, :]
        return (pair_hotspot & pair_mask).to(flow_float_dtype(batch))[..., None]


class FlowRelativeSequenceSeparationPairFeat(Feature):
    required_keys = frozenset(
        (
            "token_mask",
            "mol_type_id",
            "residue_index",
            "asym_id",
        )
    )

    def __init__(self, seq_sep_dim: int, **kwargs: Any):
        super().__init__(dim=seq_sep_dim)
        assert self.dim % 2 == 1 and self.dim >= 5

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token_mask = flow_token_mask(batch)
        pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
        indices = batch["residue_index"].long()
        asym_id = batch["asym_id"].long()
        mol_type = batch["mol_type_id"].long()
        assert indices.shape == asym_id.shape == mol_type.shape == token_mask.shape
        polymer = mol_type != MOL_TYPE_TO_ID["ligand"]
        same_polymer_chain = (
            pair_mask
            & polymer[:, :, None]
            & polymer[:, None, :]
            & (asym_id[:, :, None] == asym_id[:, None, :])
        )
        seq_sep = (indices[:, :, None] - indices[:, None, :]).float()
        # Inference-only topology-derived periods; neither training nor AE
        # sequence features are changed. Carrier-touching pairs retain the
        # existing non-polymer bucket, even though carriers carry a period.
        period = batch.get("cyclic_period")
        if period is not None:
            period_i = period[:, :, None]
            period_j = period[:, None, :]
            circular_pair = same_polymer_chain & (period_i > 0) & (period_i == period_j)
            distance = seq_sep.abs()
            wrapped = period_i.to(seq_sep.dtype) - distance
            # At an even half-period preserve the original sign (antisymmetry).
            circular = torch.where(wrapped < distance, -wrapped, distance) * seq_sep.sign()
            seq_sep = torch.where(circular_pair, circular, seq_sep)
        cross_chain_value = self.dim // 2 - 1 + 10
        seq_sep = torch.where(
            same_polymer_chain,
            seq_sep,
            torch.full_like(seq_sep, float(cross_chain_value)),
        )
        limits = torch.linspace(
            -(self.dim / 2.0 - 1),
            self.dim / 2.0 - 1,
            self.dim - 1,
            device=indices.device,
        )
        feat = bin_and_one_hot(seq_sep, limits, floating=self._use_float_one_hot()).to(flow_float_dtype(batch))
        return feat * pair_mask[..., None]


class FlowRelativeChainIndexPairFeat(Feature):
    required_keys = frozenset(("token_mask", "asym_id"))

    def __init__(self, **kwargs: Any):
        super().__init__(dim=3)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        asym_id = batch["asym_id"]
        assert asym_id.shape == flow_token_mask(batch).shape
        relative = torch.sign(asym_id[:, :, None] - asym_id[:, None, :]).long() + 1
        feat = self.one_hot(relative, 3, flow_float_dtype(batch))
        return feat * flow_pair_mask(batch)[..., None]


class FlowMolPairFeat(Feature):
    required_keys = frozenset(("token_mask", "mol_type_id"))

    def __init__(self, **kwargs: Any):
        self.num_mol_types = len(MOL_TYPE_TO_ID)
        super().__init__(dim=self.num_mol_types**2)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token_mask = flow_token_mask(batch)
        mol_type = batch["mol_type_id"].long()
        safe = torch.where(token_mask, mol_type, torch.zeros_like(mol_type))
        pair_id = safe[:, :, None] * self.num_mol_types + safe[:, None, :]
        feat = self.one_hot(pair_id, self.dim, flow_float_dtype(batch))
        return feat * flow_pair_mask(batch)[..., None]


class FlowSameMetaPairFeat(Feature):
    required_keys = frozenset(("token_mask", "asym_id", "mol_type_id"))

    def __init__(self, **kwargs: Any):
        super().__init__(dim=2)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        asym_id = batch["asym_id"]
        mol_type = batch["mol_type_id"]
        feat = torch.stack(
            (
                asym_id[:, :, None] == asym_id[:, None, :],
                mol_type[:, :, None] == mol_type[:, None, :],
            ),
            dim=-1,
        ).to(flow_float_dtype(batch))
        return feat * flow_pair_mask(batch)[..., None]


class _FlowTokenCenterDistancePairFeat(Feature):
    def __init__(
        self,
        token_center_pair_dist_dim: int,
        token_center_pair_dist_min_nm: float,
        token_center_pair_dist_max_nm: float,
        **kwargs: Any,
    ):
        super().__init__(dim=token_center_pair_dist_dim)
        self.min_dist = float(token_center_pair_dist_min_nm)
        self.max_dist = float(token_center_pair_dist_max_nm)
        assert self.dim > 1 and self.max_dist > self.min_dist

    def _distogram(self, coords: torch.Tensor) -> torch.Tensor:
        return bin_pairwise_distances(
            coords, self.min_dist, self.max_dist, self.dim, floating=self._use_float_one_hot()
        ).to(coords.dtype)


class FlowXtTokenCenterDistancePairFeat(_FlowTokenCenterDistancePairFeat):
    required_keys = frozenset(("token_mask", "x_t"))

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token_mask = flow_token_mask(batch)
        coords = batch["x_t"]["token_center"]
        assert coords.shape == token_mask.shape + (3,)
        safe_coords = _masked_coords(coords, token_mask)
        return self._distogram(safe_coords) * flow_pair_mask(batch)[..., None]


class FlowXscTokenCenterDistancePairFeat(_FlowTokenCenterDistancePairFeat):
    required_keys = frozenset(("token_mask", "x_sc", "x_sc_valid"))

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token_mask = flow_token_mask(batch)
        valid = (
            _expand_token_flag(batch["x_sc_valid"], token_mask, "x_sc_valid")
            & token_mask
        )
        coords = batch["x_sc"]["token_center"].detach()
        assert coords.shape == token_mask.shape + (3,)
        pair_valid = valid[:, :, None] & valid[:, None, :]
        return self._distogram(_masked_coords(coords, valid)) * pair_valid[..., None]


class FlowOptionalTokenCenterDistancePairFeat(_FlowTokenCenterDistancePairFeat):
    required_keys = frozenset(
        ("token_mask", "optional_token_center_coords_nm", "optional_token_center_mask")
    )

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token_mask = flow_token_mask(batch)
        known = batch["optional_token_center_mask"].bool()
        assert known.shape == token_mask.shape
        coords = batch["optional_token_center_coords_nm"]
        assert coords.shape == token_mask.shape + (3,)
        pair_known = known[:, :, None] & known[:, None, :]
        return self._distogram(_masked_coords(coords, known)) * pair_known[..., None]


class FlowConditionTokenBondPairFeat(Feature):

    required_keys = frozenset(
        (
            "token_mask",
            "condition_bond_pair_mask",
            "condition_bond_pair_endpoint_slot",
        )
    )

    def __init__(self, **kwargs: Any):
        assert CONDITION_TOKEN_BOND_ENDPOINT_VOCAB_SIZE == 92
        assert CONDITION_TOKEN_BOND_PAIR_DIM == 1 + 92 + 92 == 185
        super().__init__(dim=CONDITION_TOKEN_BOND_PAIR_DIM)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token_mask = flow_token_mask(batch)
        batch_size, n_token = token_mask.shape
        pair_mask = batch["condition_bond_pair_mask"].bool()
        endpoint_slot = batch["condition_bond_pair_endpoint_slot"].long()
        assert pair_mask.shape == (batch_size, n_token, n_token)
        assert endpoint_slot.shape == pair_mask.shape + (2,)

        pair_mask = pair_mask & flow_pair_mask(batch)
        safe_slot = torch.where(
            pair_mask[..., None], endpoint_slot, torch.zeros_like(endpoint_slot)
        )
        dtype = flow_float_dtype(batch)
        visible = pair_mask[..., None].to(dtype=dtype)
        endpoint_i = self.one_hot(
            safe_slot[..., 0], CONDITION_TOKEN_BOND_ENDPOINT_VOCAB_SIZE, dtype
        )
        endpoint_j = self.one_hot(
            safe_slot[..., 1], CONDITION_TOKEN_BOND_ENDPOINT_VOCAB_SIZE, dtype
        )
        return torch.cat(
            (visible, endpoint_i * visible, endpoint_j * visible), dim=-1
        )


class FlowConditionBondOrderPairFeat(Feature):
    """Standalone mirror of the five-way known bond-order condition."""

    required_keys = frozenset(
        (
            "token_mask",
            "condition_bond_type",
            "condition_bond_type_known_mask",
        )
    )

    def __init__(self, **kwargs: Any):
        super().__init__(dim=5)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        pair_mask = flow_pair_mask(batch)
        bond_type = batch["condition_bond_type"].long()
        known = batch["condition_bond_type_known_mask"].bool()
        assert bond_type.shape == known.shape == pair_mask.shape
        known = known & pair_mask
        safe_type = torch.where(known, bond_type, torch.zeros_like(bond_type))
        return self.one_hot(safe_type, self.dim, flow_float_dtype(batch)) * known[..., None]


class FlowPolymerSSPairFeat(Feature):
    required_keys = frozenset(
        (
            "token_mask",
            "mol_type_id",
            "polymer_ss_pair",
            "polymer_ss_pair_condition_mask",
        )
    )

    def __init__(self, **kwargs: Any):
        super().__init__(dim=5)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        pair_mask = flow_pair_mask(batch)
        condition_mask = batch["polymer_ss_pair_condition_mask"].bool()
        state = batch["polymer_ss_pair"].long()
        assert condition_mask.shape == state.shape == pair_mask.shape
        visible = condition_mask
        mol_type = batch["mol_type_id"]
        assert mol_type.shape == pair_mask.shape[:2]
        protein = mol_type == MOL_TYPE_TO_ID["protein"]
        nucleic = (mol_type == MOL_TYPE_TO_ID["rna"]) | (
            mol_type == MOL_TYPE_TO_ID["dna"]
        )
        protein_pair = protein[:, :, None] & protein[:, None, :]
        nucleic_pair = nucleic[:, :, None] & nucleic[:, None, :]
        supported_pair = pair_mask & (protein_pair | nucleic_pair)
        safe_state = torch.where(visible, state, torch.zeros_like(state))
        channels = (
            supported_pair & ~visible,
            visible & protein_pair & (safe_state == 1),
            visible & protein_pair & (safe_state == 2),
            visible & nucleic_pair & (safe_state == 1),
            visible & nucleic_pair & (safe_state == 2),
        )
        return torch.stack(channels, dim=-1).to(flow_float_dtype(batch))


class FlowTimeEmbeddingPairFeat(Feature):
    required_keys = frozenset(("token_mask", "t"))

    def __init__(self, component: str, t_emb_dim: int, **kwargs: Any):
        super().__init__(dim=t_emb_dim)
        self.component = component

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        time = batch["t"][self.component]
        _, n_token = flow_token_mask(batch).shape
        embedding = get_time_embedding(time, self.dim).to(flow_float_dtype(batch))
        return embedding[:, None, None, :].expand(-1, n_token, n_token, -1)


FeatureConstructor = Callable[..., Feature]
VAE_FEATURE_REGISTRY: dict[FeatureMode, dict[str, FeatureConstructor]] = {
    "seq": {
        "chain_break_per_res": ChainBreakPerResidueSeqFeat,
        "mol_type": MolTypeSeqFeat,
        "token_center_nm": TokenCenterCoordsNmSeqFeat,
        "z_latent_seq": LatentVariableSeqFeat,
        "x1_protein_token_type": ProteinTokenTypeSeqFeat,
        "x1_protein_atom37_coords": lambda **kw: ProteinAtom37CoordsSeqFeat(
            rel=False, **kw
        ),
        "x1_protein_atom37_coords_rel": lambda **kw: ProteinAtom37CoordsSeqFeat(
            rel=True, **kw
        ),
        "x1_protein_bb_angles": ProteinBackboneTorsionAnglesSeqFeat,
        "x1_protein_sidechain_angles": ProteinSideChainAnglesSeqFeat,
        "x1_rna_token_type": RNATokenTypeSeqFeat,
        "x1_rna_atom27_coords": lambda **kw: RNAAtom27CoordsSeqFeat(
            rel=False, local=False, **kw
        ),
        "x1_rna_atom27_coords_rel": lambda **kw: RNAAtom27CoordsSeqFeat(
            rel=True, local=False, **kw
        ),
        "x1_rna_atom27_coords_local": lambda **kw: RNAAtom27CoordsSeqFeat(
            rel=True, local=True, **kw
        ),
        "x1_rna_group_masks": RNAGroupMaskSeqFeat,
        "x1_rna_torsion_angles": RNATorsionAnglesSeqFeat,
        "x1_dna_token_type": DNATokenTypeSeqFeat,
        "x1_dna_atom27_coords": lambda **kw: DNAAtom27CoordsSeqFeat(
            rel=False, local=False, **kw
        ),
        "x1_dna_atom27_coords_rel": lambda **kw: DNAAtom27CoordsSeqFeat(
            rel=True, local=False, **kw
        ),
        "x1_dna_atom27_coords_local": lambda **kw: DNAAtom27CoordsSeqFeat(
            rel=True, local=True, **kw
        ),
        "x1_dna_group_masks": DNAGroupMaskSeqFeat,
        "x1_dna_torsion_angles": DNATorsionAnglesSeqFeat,
        "x1_ligand_element": LigandElementSeqFeat,
        "x1_ligand_charge": LigandChargeSeqFeat,
        "x1_ligand_coord": LigandCoordSeqFeat,
        "x1_ligand_degree": LigandDegreeSeqFeat,
    },
    "pair": {
        "rel_seq_sep": SequenceSeparationPairFeat,
        "rel_chain_idx": RelativeChainIndexPairFeat,
        "mol_pair": MolTypePairFeat,
        "same_meta_pair": SameMetaPairFeat,
        "token_center_pair_dists": TokenCenterPairDistancePairFeat,
        "token_frame_pair_orientation": TokenFrameOrientationPairFeat,
        "token_bond_pair": TokenBondPairFeat,
    },
}

FLOW_FEATURE_REGISTRY: dict[FeatureMode, dict[str, FeatureConstructor]] = {
    "seq": {
        "chain_break_per_res": FlowChainBreakSeqFeat,
        "mol_type": FlowMolTypeSeqFeat,
        "xt_token_center": FlowXtTokenCenterSeqFeat,
        "xt_local_latents": FlowXtLocalLatentsSeqFeat,
        "xsc_token_center": FlowXscTokenCenterSeqFeat,
        "xsc_local_latents": FlowXscLocalLatentsSeqFeat,
        "optional_token_center": FlowOptionalTokenCenterSeqFeat,
        "condition_protein_token_type": ConditionProteinTokenTypeSeqFeat,
        "condition_rna_token_type": ConditionRNATokenTypeSeqFeat,
        "condition_dna_token_type": ConditionDNATokenTypeSeqFeat,
        "condition_ligand_element": ConditionLigandElementSeqFeat,
        "hotspot": FlowHotspotSeqFeat,
        "polymer_ss_token": FlowPolymerSSTokenSeqFeat,
        "protein_atom37_motif": ProteinAtom37MotifSeqFeat,
        "rna_atom27_motif": RNAAtom27MotifSeqFeat,
        "dna_atom27_motif": DNAAtom27MotifSeqFeat,
        "ligand_motif": LigandMotifSeqFeat,
        "time_emb_token_center": lambda **kw: FlowTimeEmbeddingSeqFeat(
            component="token_center", **kw
        ),
        "time_emb_local_latents": lambda **kw: FlowTimeEmbeddingSeqFeat(
            component="local_latents", **kw
        ),
        "polymer_coarse_ss": FlowPolymerCoarseSSSeqFeat,
        "soft_template_single": TemplateSingleSeqFeat,
    },
    "pair": {
        "rel_seq_sep": FlowRelativeSequenceSeparationPairFeat,
        "rel_chain_idx": FlowRelativeChainIndexPairFeat,
        "mol_pair": FlowMolPairFeat,
        "same_meta_pair": FlowSameMetaPairFeat,
        "xt_token_center_pair_dists": FlowXtTokenCenterDistancePairFeat,
        "xsc_token_center_pair_dists": FlowXscTokenCenterDistancePairFeat,
        "optional_token_center_pair_dists": FlowOptionalTokenCenterDistancePairFeat,
        "hotspot_mask_pair": FlowHotspotPairFeat,
        "condition_token_bond_pair": FlowConditionTokenBondPairFeat,
        "polymer_ss_pair": FlowPolymerSSPairFeat,
        "condition_bond_order_pair": FlowConditionBondOrderPairFeat,
        "soft_template_pair": TemplatePairFeat,
        "time_emb_token_center": lambda **kw: FlowTimeEmbeddingPairFeat(
            component="token_center", **kw
        ),
        "time_emb_local_latents": lambda **kw: FlowTimeEmbeddingPairFeat(
            component="local_latents", **kw
        ),
    },
}


def get_feature_registry(
    registry_name: FeatureRegistryName, mode: FeatureMode
) -> Mapping[str, FeatureConstructor]:
    registries = {
        "vae": VAE_FEATURE_REGISTRY,
        "flow": FLOW_FEATURE_REGISTRY,
    }
    if registry_name not in registries:
        raise ValueError(f"Unknown feature registry `{registry_name}`")
    if mode not in ("seq", "pair"):
        raise ValueError(f"Unknown feature mode `{mode}`")
    return registries[registry_name][mode]


class FeatureFactory(torch.nn.Module):
    def __init__(
        self,
        feats: list[str] | tuple[str, ...] | None,
        dim_feats_out: int,
        use_ln_out: bool,
        mode: FeatureMode,
        feature_registry: FeatureRegistryName = "vae",
        strict_feats: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        if mode not in ("seq", "pair"):
            raise ValueError(f"Feature mode must be `seq` or `pair`, got `{mode}`")
        self.mode = mode
        self.registry_name = feature_registry
        self.strict_feats = bool(strict_feats)
        self.dim_feats_out = int(dim_feats_out)
        self.feature_names = tuple(feats or ())
        self.ret_zero = not self.feature_names
        if self.ret_zero:
            logger.trace("No features requested")
            self.zero_creator = ZeroFeat(dim_feats_out=self.dim_feats_out, mode=mode)
            self.raw_dim = 0
            self.required_keys = frozenset()
            return

        registry = get_feature_registry(feature_registry, mode)
        creators = []
        for feature_name in self.feature_names:
            if feature_name not in registry:
                raise KeyError(
                    f"{feature_registry} {mode} feature `{feature_name}` is not registered"
                )
            creator = registry[feature_name](**kwargs)
            creators.append(creator)
        self.feat_creators = torch.nn.ModuleList(creators)
        self.raw_dim = sum(creator.get_dim() for creator in creators)
        self.required_keys = frozenset().union(
            *(creator.required_keys for creator in creators)
        )
        self.ln_out = (
            torch.nn.LayerNorm(self.dim_feats_out)
            if use_ln_out
            else torch.nn.Identity()
        )
        self.linear_out = torch.nn.Linear(self.raw_dim, self.dim_feats_out, bias=False)
        template_positions = [
            index
            for index, name in enumerate(self.feature_names)
            if name in {"soft_template_single", "soft_template_pair"}
        ]
        if template_positions and template_positions != [len(self.feature_names) - 1]:
            raise ValueError(
                "soft template features must be the final feature in their registry list"
            )
        self.template_feature_names = tuple(
            self.feature_names[index] for index in template_positions
        )
        self.template_prefix_dim = self.raw_dim
        self.template_tail_dim = 0
        if template_positions:
            template_index = template_positions[0]
            self.template_prefix_dim = sum(
                creator.get_dim() for creator in self.feat_creators[:template_index]
            )
            self.template_tail_dim = self.raw_dim - self.template_prefix_dim
            # Keep a fresh native model exactly on the legacy function until
            # the appended template columns receive a gradient update.
            with torch.no_grad():
                self.linear_out.weight[:, self.template_prefix_dim :].zero_()

    def get_creator(self, feature_name: str, **kwargs: Any) -> Feature:
        registry = get_feature_registry(self.registry_name, self.mode)
        if feature_name not in registry:
            raise KeyError(
                f"{self.registry_name} {self.mode} feature `{feature_name}` is not registered"
            )
        return registry[feature_name](**kwargs)

    def _check_required_keys(self, batch: Mapping[str, Any]) -> None:
        if not self.strict_feats:
            return
        missing = sorted(key for key in self.required_keys if key not in batch)
        if missing:
            raise KeyError(f"Missing required feature fields: {', '.join(missing)}")

    def _mask(self, batch: Mapping[str, Any]) -> torch.Tensor:
        if "token_mask" in batch:
            return batch["token_mask"]
        raise IOError("FeatureFactory requires `token_mask`")

    def apply_padding_mask(
        self, feature_tensor: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask = mask.bool()
        if self.mode == "seq":
            expanded = mask[..., None]
        else:
            expanded = (mask[:, :, None] & mask[:, None, :])[..., None]
        return feature_tensor * expanded.to(dtype=feature_tensor.dtype)

    def raw_features(self, batch: Mapping[str, Any]) -> torch.Tensor:
        self._check_required_keys(batch)
        if self.ret_zero:
            return self.zero_creator(batch)
        mask = self._mask(batch)
        expected_prefix = (
            mask.shape
            if self.mode == "seq"
            else (mask.shape[0], mask.shape[1], mask.shape[1])
        )
        tensors = []
        for name, creator in zip(self.feature_names, self.feat_creators):
            tensor = creator(batch)
            expected_shape = expected_prefix + (creator.get_dim(),)
            assert (
                tensor.shape == expected_shape
            ), f"Feature `{name}` returned {tuple(tensor.shape)}, expected {expected_shape}"
            tensors.append(tensor)
        return self.apply_padding_mask(torch.cat(tensors, dim=-1), mask)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        if self.ret_zero:
            return self.zero_creator(batch)
        mask = self._mask(batch)
        raw = self.raw_features(batch)
        projected = self.ln_out(self.linear_out(raw))
        return self.apply_padding_mask(projected, mask)
