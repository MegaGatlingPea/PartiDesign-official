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
from src.design.model.constants import (
    DNA_ATOM27,
    DNA_RES_NAMES,
    MOL_TYPE_TO_ID,
    NA_BACKBONE_ATOMS,
    NA_BASE_ATOMS,
    NA_SUGAR_ATOMS,
    RNA_ATOM27,
    RNA_RES_NAMES,
    TOKEN_TYPE32,
)


def _nuc_token_mask(batch: dict[str, torch.Tensor], mol_name: str) -> torch.Tensor:
    assert mol_name in {"rna", "dna"}
    assert "mol_type_id" in batch, "`mol_type_id` not in batch"
    assert "token_mask" in batch, "`token_mask` not in batch"
    return (batch["mol_type_id"] == MOL_TYPE_TO_ID[mol_name]) & batch["token_mask"].bool()


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


def _coords_in_token_frame(
    coords: torch.Tensor,
    mask: torch.Tensor,
    frame: torch.Tensor,
    frame_mask: torch.Tensor,
    center_slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rel, rel_mask = _coords_relative_to_slot(coords, mask, center_slot)
    local = torch.einsum("bnij,bnaj->bnai", frame.transpose(-1, -2), rel)
    local_mask = rel_mask & frame_mask[..., None]
    return local * local_mask[..., None].to(dtype=local.dtype), local_mask


def _angle_bins(angle: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    feat = bin_and_one_hot(angle, torch.linspace(-torch.pi, torch.pi, 20, device=angle.device))
    return feat * valid[..., None].to(dtype=feat.dtype)


class NucTokenTypeSeqFeat(Feature):

    def __init__(self, *, mol_name: str, res_names: tuple[str, ...], **kwargs):
        super().__init__(dim=len(res_names))
        assert mol_name in {"rna", "dna"}
        self.mol_name = mol_name
        self.res_names = res_names
        self.global_ids = tuple(TOKEN_TYPE32[name] for name in res_names)
        unknown_name = "N" if mol_name == "rna" else "DN"
        assert unknown_name in res_names
        self.unknown_local_id = res_names.index(unknown_name)
        self.required_keys = frozenset(("token_mask", "mol_type_id", "token_type32"))

    def forward(self, batch):
        assert "token_type32" in batch, "`token_type32` not in batch"
        token_mask = _nuc_token_mask(batch, self.mol_name)
        token_type = batch["token_type32"].long()
        local_type = torch.full_like(token_type, self.unknown_local_id)
        valid_type = torch.zeros_like(token_mask)
        for local_id, global_id in enumerate(self.global_ids):
            is_type = token_type == global_id
            valid_type |= is_type
            local_type = torch.where(is_type, torch.full_like(local_type, local_id), local_type)
        feat = F.one_hot(local_type.clamp(min=0, max=self.dim - 1), num_classes=self.dim).to(dtype=torch.float32)
        return feat * token_mask[..., None].to(dtype=feat.dtype)


class NucAtom27CoordsSeqFeat(Feature):

    def __init__(
        self,
        *,
        mol_name: str,
        atom_names: tuple[str, ...],
        rel: bool = False,
        local: bool = False,
        **kwargs,
    ):
        super().__init__(dim=len(atom_names) * 4)
        assert mol_name in {"rna", "dna"}
        self.mol_name = mol_name
        self.atom_names = atom_names
        self.rel = bool(rel)
        self.local = bool(local)
        self.c1_slot = atom_names.index("C1'")
        if self.local:
            assert self.rel, "local nucleotide coordinates are defined relative to C1'"
        required_keys = {
            "token_mask",
            "mol_type_id",
            f"{self.mol_name}_atom27_coords",
            f"{self.mol_name}_atom27_mask",
        }
        if self.local:
            required_keys.update(("token_frame", "token_frame_mask"))
        self.required_keys = frozenset(required_keys)

    def forward(self, batch):
        coords = batch[f"{self.mol_name}_atom27_coords"]
        atom_mask = batch[f"{self.mol_name}_atom27_mask"] & _nuc_token_mask(batch, self.mol_name)[..., None]
        if self.local:
            coords, atom_mask = _coords_in_token_frame(
                coords,
                atom_mask,
                batch["token_frame"],
                batch["token_frame_mask"],
                self.c1_slot,
            )
        elif self.rel:
            coords, atom_mask = _coords_relative_to_slot(coords, atom_mask, self.c1_slot)
        return _flatten_coords_with_mask(coords, atom_mask)


class NucGroupMaskSeqFeat(Feature):

    def __init__(self, *, mol_name: str, atom_names: tuple[str, ...], **kwargs):
        super().__init__(dim=len(atom_names) * 3)
        assert mol_name in {"rna", "dna"}
        self.mol_name = mol_name
        self.atom_names = atom_names
        self.required_keys = frozenset(
            (
                "token_mask",
                "mol_type_id",
                f"{self.mol_name}_atom27_mask",
            )
        )

    def forward(self, batch):
        atom_mask = batch[f"{self.mol_name}_atom27_mask"] & _nuc_token_mask(batch, self.mol_name)[..., None]
        backbone = torch.tensor(
            [name in NA_BACKBONE_ATOMS for name in self.atom_names],
            dtype=atom_mask.dtype,
            device=atom_mask.device,
        )
        sugar = torch.tensor(
            [name in NA_SUGAR_ATOMS for name in self.atom_names],
            dtype=atom_mask.dtype,
            device=atom_mask.device,
        )
        base = torch.tensor(
            [name in NA_BASE_ATOMS for name in self.atom_names],
            dtype=atom_mask.dtype,
            device=atom_mask.device,
        )
        mask_f = atom_mask.to(dtype=torch.float32)
        return torch.cat([mask_f * backbone, mask_f * sugar, mask_f * base], dim=-1)


class NucTorsionAnglesSeqFeat(Feature):

    def __init__(self, *, mol_name: str, atom_names: tuple[str, ...], **kwargs):
        super().__init__(dim=7 * 21)
        assert mol_name in {"rna", "dna"}
        self.mol_name = mol_name
        self.atom_names = atom_names
        self.idx = {name: i for i, name in enumerate(atom_names)}
        self.required_keys = frozenset(
            (
                "token_mask",
                "mol_type_id",
                f"{self.mol_name}_atom27_coords",
                f"{self.mol_name}_atom27_mask",
                "chain_break_after_token",
            )
        )

    def _coords(self, coords: torch.Tensor, name: str) -> torch.Tensor:
        return coords[:, :, self.idx[name], :]

    def _mask(self, atom_mask: torch.Tensor, name: str) -> torch.Tensor:
        return atom_mask[:, :, self.idx[name]]

    def forward(self, batch):
        coords = batch[f"{self.mol_name}_atom27_coords"]
        atom_mask = batch[f"{self.mol_name}_atom27_mask"]
        token_mask = _nuc_token_mask(batch, self.mol_name)
        b, n_token = token_mask.shape
        feat = torch.zeros(b, n_token, 7, 21, dtype=coords.dtype, device=coords.device)

        def c(name: str) -> torch.Tensor:
            return self._coords(coords, name)

        def m(name: str) -> torch.Tensor:
            return self._mask(atom_mask, name)

        def current(names: tuple[str, str, str, str], slot: int) -> None:
            valid = token_mask
            for name in names:
                valid = valid & m(name)
            angle = signed_dihedral_angle(c(names[0]), c(names[1]), c(names[2]), c(names[3]))
            feat[:, :, slot] = _angle_bins(angle, valid)

        current(("P", "O5'", "C5'", "C4'"), 1)
        current(("O5'", "C5'", "C4'", "C3'"), 2)
        current(("C5'", "C4'", "C3'", "O3'"), 3)

        if n_token > 1:
            no_break = (~batch["chain_break_after_token"][:, :-1].bool()) & token_mask[:, :-1] & token_mask[:, 1:]
            alpha_valid = no_break & m("O3'")[:, :-1] & m("P")[:, 1:] & m("O5'")[:, 1:] & m("C5'")[:, 1:]
            alpha = signed_dihedral_angle(c("O3'")[:, :-1], c("P")[:, 1:], c("O5'")[:, 1:], c("C5'")[:, 1:])
            feat[:, 1:, 0] = _angle_bins(alpha, alpha_valid)

            eps_valid = no_break & m("C4'")[:, :-1] & m("C3'")[:, :-1] & m("O3'")[:, :-1] & m("P")[:, 1:]
            eps = signed_dihedral_angle(c("C4'")[:, :-1], c("C3'")[:, :-1], c("O3'")[:, :-1], c("P")[:, 1:])
            feat[:, :-1, 4] = _angle_bins(eps, eps_valid)

            zeta_valid = no_break & m("C3'")[:, :-1] & m("O3'")[:, :-1] & m("P")[:, 1:] & m("O5'")[:, 1:]
            zeta = signed_dihedral_angle(c("C3'")[:, :-1], c("O3'")[:, :-1], c("P")[:, 1:], c("O5'")[:, 1:])
            feat[:, :-1, 5] = _angle_bins(zeta, zeta_valid)

        has_n9 = m("N9")
        has_n1 = m("N1")
        anchor = torch.where(has_n9[..., None], c("N9"), c("N1"))
        base_end = torch.where(has_n9[..., None], c("C4"), c("C2"))
        chi_valid = token_mask & m("O4'") & m("C1'") & ((has_n9 & m("C4")) | (has_n1 & m("C2")))
        chi = signed_dihedral_angle(c("O4'"), c("C1'"), anchor, base_end)
        feat[:, :, 6] = _angle_bins(chi, chi_valid)

        return feat.flatten(start_dim=-2)


class RNAAtom27CoordsSeqFeat(NucAtom27CoordsSeqFeat):
    def __init__(self, rel: bool = False, local: bool = False, **kwargs):
        super().__init__(mol_name="rna", atom_names=RNA_ATOM27, rel=rel, local=local, **kwargs)


class DNAAtom27CoordsSeqFeat(NucAtom27CoordsSeqFeat):
    def __init__(self, rel: bool = False, local: bool = False, **kwargs):
        super().__init__(mol_name="dna", atom_names=DNA_ATOM27, rel=rel, local=local, **kwargs)


class RNATokenTypeSeqFeat(NucTokenTypeSeqFeat):
    def __init__(self, **kwargs):
        super().__init__(mol_name="rna", res_names=RNA_RES_NAMES, **kwargs)


class DNATokenTypeSeqFeat(NucTokenTypeSeqFeat):
    def __init__(self, **kwargs):
        super().__init__(mol_name="dna", res_names=DNA_RES_NAMES, **kwargs)


class RNAGroupMaskSeqFeat(NucGroupMaskSeqFeat):
    def __init__(self, **kwargs):
        super().__init__(mol_name="rna", atom_names=RNA_ATOM27, **kwargs)


class DNAGroupMaskSeqFeat(NucGroupMaskSeqFeat):
    def __init__(self, **kwargs):
        super().__init__(mol_name="dna", atom_names=DNA_ATOM27, **kwargs)


class RNATorsionAnglesSeqFeat(NucTorsionAnglesSeqFeat):
    def __init__(self, **kwargs):
        super().__init__(mol_name="rna", atom_names=RNA_ATOM27, **kwargs)


class DNATorsionAnglesSeqFeat(NucTorsionAnglesSeqFeat):
    def __init__(self, **kwargs):
        super().__init__(mol_name="dna", atom_names=DNA_ATOM27, **kwargs)


class ConditionNucTokenTypeSeqFeat(Feature):

    def __init__(self, *, mol_name: str, res_names: tuple[str, ...], **kwargs):
        super().__init__(dim=len(res_names))
        assert mol_name in {"rna", "dna"}
        self.mol_name = mol_name
        self.global_ids = tuple(TOKEN_TYPE32[name] for name in res_names)
        self.required_keys = frozenset(
            ("token_mask", "mol_type_id", "token_identity_condition_mask", "token_type32")
        )

    def forward(self, batch):
        modality_mask = flow_modality_mask(batch, self.mol_name)
        identity_mask = batch["token_identity_condition_mask"].bool()
        assert identity_mask.shape == modality_mask.shape
        visible = modality_mask & identity_mask
        target_type = batch["token_type32"].long()
        safe_type = torch.where(
            visible,
            target_type,
            torch.full_like(target_type, self.global_ids[0]),
        )
        valid_type = torch.zeros_like(visible)
        local_type = torch.zeros_like(safe_type)
        for local_id, global_id in enumerate(self.global_ids):
            is_type = safe_type == global_id
            valid_type |= is_type
            local_type = torch.where(
                is_type, torch.full_like(local_type, local_id), local_type
            )
        feat = F.one_hot(local_type, num_classes=self.dim).to(flow_float_dtype(batch))
        return feat * visible[..., None]


class ConditionRNATokenTypeSeqFeat(ConditionNucTokenTypeSeqFeat):
    def __init__(self, **kwargs):
        super().__init__(mol_name="rna", res_names=RNA_RES_NAMES, **kwargs)


class ConditionDNATokenTypeSeqFeat(ConditionNucTokenTypeSeqFeat):
    def __init__(self, **kwargs):
        super().__init__(mol_name="dna", res_names=DNA_RES_NAMES, **kwargs)


class NucAtom27MotifSeqFeat(Feature):

    def __init__(self, *, mol_name: str, atom_names: tuple[str, ...], **kwargs):
        super().__init__(dim=len(atom_names) * 4)
        assert mol_name in {"rna", "dna"}
        self.mol_name = mol_name
        self.atom_names = atom_names
        self.coords_key = f"{mol_name}_motif_atom27_coords_nm"
        self.mask_key = f"{mol_name}_motif_atom27_mask"
        self.required_keys = frozenset(
            ("token_mask", "mol_type_id", self.coords_key, self.mask_key)
        )

    def forward(self, batch):
        modality_mask = flow_modality_mask(batch, self.mol_name)
        motif_mask = batch[self.mask_key].bool()
        assert motif_mask.shape == modality_mask.shape + (len(self.atom_names),)
        coords = batch[self.coords_key].to(flow_float_dtype(batch))
        assert coords.shape == motif_mask.shape + (3,)
        return flatten_coords_with_mask(coords, motif_mask)


class RNAAtom27MotifSeqFeat(NucAtom27MotifSeqFeat):
    def __init__(self, **kwargs):
        super().__init__(mol_name="rna", atom_names=RNA_ATOM27, **kwargs)


class DNAAtom27MotifSeqFeat(NucAtom27MotifSeqFeat):
    def __init__(self, **kwargs):
        super().__init__(mol_name="dna", atom_names=DNA_ATOM27, **kwargs)
