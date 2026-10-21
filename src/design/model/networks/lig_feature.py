import torch
from torch.nn import functional as F

from src.design.model.networks.feature_base import Feature, flow_float_dtype, flow_modality_mask
from src.design.model.constants import ELEMENTS_128, LIGAND_BOND_TYPES


def _ligand_token_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    assert "ligand_mask" in batch, "`ligand_mask` not in batch"
    assert "token_mask" in batch, "`token_mask` not in batch"
    return batch["ligand_mask"].bool() & batch["token_mask"].bool()


class LigandElementSeqFeat(Feature):
    required_keys = frozenset(("token_mask", "ligand_mask", "ligand_element_type"))

    def __init__(self, **kwargs):
        super().__init__(dim=len(ELEMENTS_128))

    def forward(self, batch):
        assert "ligand_element_type" in batch, "`ligand_element_type` not in batch"
        elem = batch["ligand_element_type"].long().clamp(min=0, max=self.dim - 1)
        feat = F.one_hot(elem, num_classes=self.dim).to(dtype=torch.float32)
        return feat * _ligand_token_mask(batch)[..., None].to(dtype=feat.dtype)


class LigandChargeSeqFeat(Feature):
    required_keys = frozenset(("token_mask", "ligand_mask", "ligand_charge"))

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        assert "ligand_charge" in batch, "`ligand_charge` not in batch"
        feat = batch["ligand_charge"].to(dtype=torch.float32)[..., None]
        return feat * _ligand_token_mask(batch)[..., None].to(dtype=feat.dtype)


class LigandCoordSeqFeat(Feature):
    required_keys = frozenset(("token_mask", "ligand_mask", "token_center_coords_nm"))

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        assert (
            "token_center_coords_nm" in batch
        ), "`token_center_coords_nm` not in batch"
        feat = batch["token_center_coords_nm"]
        return feat * _ligand_token_mask(batch)[..., None].to(dtype=feat.dtype)


class LigandDegreeSeqFeat(Feature):
    required_keys = frozenset(("token_mask", "ligand_mask", "token_bond_type"))

    def __init__(self, ligand_degree_bins: int = 9, **kwargs):
        super().__init__(dim=int(ligand_degree_bins))
        assert self.dim > 0

    def forward(self, batch):
        assert "token_bond_type" in batch, "`token_bond_type` not in batch"
        bond_present = batch["token_bond_type"] > LIGAND_BOND_TYPES["none"]
        degree = bond_present.long().sum(dim=-1).clamp(min=0, max=self.dim - 1)
        feat = F.one_hot(degree, num_classes=self.dim).to(dtype=torch.float32)
        return feat * _ligand_token_mask(batch)[..., None].to(dtype=feat.dtype)


class TokenBondPairFeat(Feature):
    required_keys = frozenset(("token_bond_type", "token_bond_pair_mask"))

    def __init__(self, **kwargs):
        super().__init__(dim=len(LIGAND_BOND_TYPES))

    def forward(self, batch):
        assert "token_bond_type" in batch, "`token_bond_type` not in batch"
        assert "token_bond_pair_mask" in batch, "`token_bond_pair_mask` not in batch"
        bond_type = batch["token_bond_type"].long().clamp(min=0, max=self.dim - 1)
        feat = F.one_hot(bond_type, num_classes=self.dim).to(dtype=torch.float32)
        return feat * batch["token_bond_pair_mask"][..., None].to(dtype=feat.dtype)


class ConditionLigandElementSeqFeat(Feature):
    required_keys = frozenset(
        (
            "token_mask",
            "mol_type_id",
            "token_identity_condition_mask",
            "ligand_element_type",
        )
    )

    def __init__(self, **kwargs):
        super().__init__(dim=len(ELEMENTS_128))

    def forward(self, batch):
        ligand_mask = flow_modality_mask(batch, "ligand")
        identity_mask = batch["token_identity_condition_mask"].bool()
        assert identity_mask.shape == ligand_mask.shape
        visible = ligand_mask & identity_mask
        target_element = batch["ligand_element_type"].long()
        safe_element = torch.where(
            visible, target_element, torch.zeros_like(target_element)
        )
        feat = F.one_hot(safe_element, num_classes=self.dim).to(flow_float_dtype(batch))
        return feat * visible[..., None]


class LigandMotifSeqFeat(Feature):
    required_keys = frozenset(
        (
            "token_mask",
            "mol_type_id",
            "ligand_motif_token_center_coords_nm",
            "ligand_motif_mask",
        )
    )

    def __init__(self, **kwargs):
        super().__init__(dim=4)

    def forward(self, batch):
        ligand_mask = flow_modality_mask(batch, "ligand")
        motif_mask = batch["ligand_motif_mask"].bool()
        assert motif_mask.shape == ligand_mask.shape
        coords = batch["ligand_motif_token_center_coords_nm"].to(
            flow_float_dtype(batch)
        )
        assert coords.shape == motif_mask.shape + (3,)
        safe_coords = torch.where(
            motif_mask[..., None], coords, torch.zeros_like(coords)
        )
        return torch.cat([safe_coords, motif_mask.to(coords.dtype)[..., None]], dim=-1)
