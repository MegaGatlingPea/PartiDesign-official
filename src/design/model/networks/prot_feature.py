import torch
from torch.nn import functional as F

from src.design.model.networks.feature_base import (
    Feature,
    bin_and_one_hot,
    flatten_coords_with_mask,
    flow_float_dtype,
    flow_modality_mask,
)
from src.design.model.angle_utils import signed_dihedral_angle
from src.design.model.constants import MOL_TYPE_TO_ID, PROTEIN_ATOM37, PROTEIN_RES_NAMES, TOKEN_TYPE32
from src.design.model.protein_torsion import protein_chi_angles


def _protein_token_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    assert "mol_type_id" in batch, "`mol_type_id` not in batch"
    assert "token_mask" in batch, "`token_mask` not in batch"
    return (batch["mol_type_id"] == MOL_TYPE_TO_ID["protein"]) & batch["token_mask"].bool()


def _flatten_coords_with_mask(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert coords.shape[:-1] == mask.shape, (coords.shape, mask.shape)
    coords = coords * mask[..., None].to(dtype=coords.dtype)
    return torch.cat([coords.flatten(start_dim=-2), mask.to(dtype=coords.dtype)], dim=-1)


def _coords_relative_to_slot(
    coords: torch.Tensor,
    mask: torch.Tensor,
    center_slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert coords.shape[:-1] == mask.shape, (coords.shape, mask.shape)
    assert 0 <= center_slot < coords.shape[-2]
    center_mask = mask[..., center_slot]
    rel_mask = mask & center_mask[..., None]
    rel = (coords - coords[..., center_slot : center_slot + 1, :]) * rel_mask[..., None].to(dtype=coords.dtype)
    return rel, rel_mask


def _angle_bins(angle: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    feat = bin_and_one_hot(angle, torch.linspace(-torch.pi, torch.pi, 20, device=angle.device))
    return feat * valid[..., None].to(dtype=feat.dtype)


class ProteinTokenTypeSeqFeat(Feature):
    required_keys = frozenset(("token_mask", "mol_type_id", "token_type32"))

    def __init__(self, **kwargs):
        super().__init__(dim=len(PROTEIN_RES_NAMES))

    def forward(self, batch):
        assert "token_type32" in batch, "`token_type32` not in batch"
        token_mask = _protein_token_mask(batch)
        token_type = batch["token_type32"].long()
        local_type = token_type.clamp(min=0, max=self.dim - 1)
        feat = F.one_hot(local_type, num_classes=self.dim).to(dtype=torch.float32)
        return feat * token_mask[..., None].to(dtype=feat.dtype)


class ProteinAtom37CoordsSeqFeat(Feature):
    required_keys = frozenset(
        ("token_mask", "mol_type_id", "protein_atom37_coords", "protein_atom37_mask")
    )

    def __init__(self, rel: bool = False, **kwargs):
        super().__init__(dim=len(PROTEIN_ATOM37) * 4)
        self.rel = bool(rel)
        self.ca_slot = PROTEIN_ATOM37.index("CA")

    def forward(self, batch):
        assert "protein_atom37_coords" in batch, "`protein_atom37_coords` not in batch"
        assert "protein_atom37_mask" in batch, "`protein_atom37_mask` not in batch"
        coords = batch["protein_atom37_coords"]
        atom_mask = batch["protein_atom37_mask"] & _protein_token_mask(batch)[..., None]
        if self.rel:
            coords, atom_mask = _coords_relative_to_slot(coords, atom_mask, self.ca_slot)
        return _flatten_coords_with_mask(coords, atom_mask)


class ProteinBackboneTorsionAnglesSeqFeat(Feature):
    required_keys = frozenset(
        (
            "token_mask",
            "mol_type_id",
            "protein_atom37_coords",
            "protein_atom37_mask",
            "chain_break_after_token",
        )
    )

    def __init__(self, **kwargs):
        super().__init__(dim=3 * 21)
        self.n_slot = PROTEIN_ATOM37.index("N")
        self.ca_slot = PROTEIN_ATOM37.index("CA")
        self.c_slot = PROTEIN_ATOM37.index("C")

    def forward(self, batch):
        coords = batch["protein_atom37_coords"]
        atom_mask = batch["protein_atom37_mask"]
        token_mask = _protein_token_mask(batch)
        b, n_token = token_mask.shape
        feat = torch.zeros(b, n_token, 3, 21, dtype=coords.dtype, device=coords.device)
        if n_token <= 1:
            return feat.flatten(start_dim=-2)

        n = coords[:, :, self.n_slot]
        ca = coords[:, :, self.ca_slot]
        c = coords[:, :, self.c_slot]
        n_mask = atom_mask[:, :, self.n_slot]
        ca_mask = atom_mask[:, :, self.ca_slot]
        c_mask = atom_mask[:, :, self.c_slot]
        no_break = (~batch["chain_break_after_token"][:, :-1].bool()) & token_mask[:, :-1] & token_mask[:, 1:]
        pair_ok = (
            no_break
            & n_mask[:, :-1]
            & ca_mask[:, :-1]
            & c_mask[:, :-1]
            & n_mask[:, 1:]
            & ca_mask[:, 1:]
            & c_mask[:, 1:]
        )

        psi = signed_dihedral_angle(n[:, :-1], ca[:, :-1], c[:, :-1], n[:, 1:])
        omega = signed_dihedral_angle(ca[:, :-1], c[:, :-1], n[:, 1:], ca[:, 1:])
        phi = signed_dihedral_angle(c[:, :-1], n[:, 1:], ca[:, 1:], c[:, 1:])
        feat[:, :-1, 0] = _angle_bins(psi, pair_ok)
        feat[:, :-1, 1] = _angle_bins(omega, pair_ok)
        feat[:, 1:, 2] = _angle_bins(phi, pair_ok)
        return feat.flatten(start_dim=-2)


class ProteinSideChainAnglesSeqFeat(Feature):
    required_keys = frozenset(
        (
            "token_mask",
            "mol_type_id",
            "token_type32",
            "protein_atom37_coords",
            "protein_atom37_mask",
        )
    )

    def __init__(self, **kwargs):
        super().__init__(dim=4 * 21 + 4)

    def forward(self, batch):
        coords = batch["protein_atom37_coords"]
        atom_mask = batch["protein_atom37_mask"]
        token_mask = _protein_token_mask(batch)
        raw_residue_type = batch["token_type32"].long()
        angles, torsion_mask = protein_chi_angles(
            token_type32=raw_residue_type,
            atom37_coords=coords,
            atom37_mask=atom_mask,
            protein_token_mask=token_mask,
        )
        angles = angles * torsion_mask.to(angles.dtype)
        angles_feat = bin_and_one_hot(
            angles,
            torch.linspace(-torch.pi, torch.pi, 20, device=coords.device),
        )
        angles_feat = angles_feat * torsion_mask[..., None]
        feat = torch.cat([angles_feat.flatten(start_dim=-2), torsion_mask], dim=-1).to(dtype=coords.dtype)
        return feat * token_mask[..., None].to(dtype=feat.dtype)


class ConditionProteinTokenTypeSeqFeat(Feature):
    required_keys = frozenset(
        ("token_mask", "mol_type_id", "token_identity_condition_mask", "token_type32")
    )

    def __init__(self, **kwargs):
        super().__init__(dim=len(PROTEIN_RES_NAMES))

    def forward(self, batch):
        protein_mask = flow_modality_mask(batch, "protein")
        identity_mask = batch["token_identity_condition_mask"].bool()
        assert identity_mask.shape == protein_mask.shape
        visible = protein_mask & identity_mask
        target_type = batch["token_type32"].long()
        safe_type = torch.where(visible, target_type, torch.zeros_like(target_type))
        feat = F.one_hot(safe_type, num_classes=self.dim).to(flow_float_dtype(batch))
        return feat * visible[..., None]


class ProteinAtom37MotifSeqFeat(Feature):
    required_keys = frozenset(
        (
            "token_mask",
            "mol_type_id",
            "protein_motif_atom37_coords_nm",
            "protein_motif_atom37_mask",
        )
    )

    def __init__(self, **kwargs):
        super().__init__(dim=len(PROTEIN_ATOM37) * 4)

    def forward(self, batch):
        protein_mask = flow_modality_mask(batch, "protein")
        motif_mask = batch["protein_motif_atom37_mask"].bool()
        assert motif_mask.shape == protein_mask.shape + (len(PROTEIN_ATOM37),)
        coords = batch["protein_motif_atom37_coords_nm"].to(flow_float_dtype(batch))
        assert coords.shape == motif_mask.shape + (3,)
        return flatten_coords_with_mask(coords, motif_mask)
