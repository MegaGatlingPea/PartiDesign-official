"""Geometry-only soft-template features.

The template stream deliberately stops at the raw feature boundary. The
ordinary FeatureFactory owns the projection into the model width, so a
template is subject to the same projection/normalization path as every other
registered feature. This keeps the template representation free of learned
adapter bottlenecks and makes its raw ABI explicit:

* soft_template_single: 11 channels;
* soft_template_pair: 111 channels.

Only internal, same-group geometry is exposed. Coordinates are masked before
any arithmetic, and every positive template group is independently invariant
to a rigid transform. Group labels are used only for equality, never as
numeric features.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.design.model.networks.feature_base import Feature, flow_float_dtype
from src.design.model.constants import MOL_TYPE_TO_ID, PROTEIN_ATOM37


TEMPLATE_SINGLE_DIM = 11
TEMPLATE_PAIR_DIM = 111


class TemplateFeatureFactory(nn.Module):
    """Build masked, invariant raw template geometry.

    This class has no learned parameters. It is a geometry helper used by
    the two registry features below. build_single_features intentionally
    does not construct any [B, N, N, ...] tensor; pair geometry is only
    evaluated when soft_template_pair is requested.
    """

    required_keys = frozenset(
        {
            "token_mask",
            "mol_type_id",
            "template_group_id",
            "template_protein_atom37_coords_nm",
            "template_protein_atom37_mask",
            "antibody_chain_role_id",
            "antibody_region_id",
        }
    )

    _FRAME_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O")
    _PROTEIN_ID = MOL_TYPE_TO_ID["protein"]

    def __init__(
        self,
        *,
        rbf_centers: Sequence[float] | Tensor | None = None,
        rbf_scales: Sequence[float] | Tensor = (0.05, 0.15, 0.40, 0.80),
        frame_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.frame_eps = float(frame_eps)
        if not self.frame_eps > 0:
            raise ValueError("frame_eps must be positive")

        atom_slots = tuple(PROTEIN_ATOM37.index(name) for name in self._FRAME_ATOMS)
        # These buffers are nonpersistent ABI constants and are omitted from
        # checkpoint state. Keep their canonical values on CPU while constructing
        # a meta model; a later module.to(device) moves them with the model.
        self.register_buffer(
            "_frame_atom_slots",
            torch.tensor(atom_slots, dtype=torch.long, device="cpu"),
            persistent=False,
        )

        if rbf_centers is None:
            rbf_centers_tensor = torch.linspace(
                0.0, 4.0, 16, dtype=torch.float32, device="cpu"
            )
        else:
            rbf_centers_tensor = torch.as_tensor(
                rbf_centers, dtype=torch.float32, device="cpu"
            )
        rbf_scales_tensor = torch.as_tensor(
            rbf_scales, dtype=torch.float32, device="cpu"
        )
        if rbf_centers_tensor.ndim != 1 or rbf_centers_tensor.numel() == 0:
            raise ValueError("rbf_centers must be a non-empty one-dimensional sequence")
        if rbf_scales_tensor.ndim != 1 or rbf_scales_tensor.numel() == 0:
            raise ValueError("rbf_scales must be a non-empty one-dimensional sequence")
        # Meta inputs cannot be inspected with torch.isfinite(...).all().item().
        # The explicit CPU construction above keeps the normal path checked even
        # inside torch.device("meta"); retain the guard for any future meta source.
        if not rbf_centers_tensor.is_meta and not bool(
            torch.isfinite(rbf_centers_tensor).all()
        ):
            raise ValueError("rbf_centers must be finite")
        if not rbf_scales_tensor.is_meta and (
            not bool(torch.isfinite(rbf_scales_tensor).all())
            or bool((rbf_scales_tensor <= 0).any())
        ):
            raise ValueError("rbf_scales must be finite and positive")
        self.register_buffer("rbf_centers", rbf_centers_tensor, persistent=False)
        self.register_buffer("rbf_scales", rbf_scales_tensor, persistent=False)

        self.single_dim = TEMPLATE_SINGLE_DIM
        self.pair_dim = TEMPLATE_PAIR_DIM

    @property
    def required_feature_keys(self) -> frozenset[str]:
        return self.required_keys

    def _validate_batch(
        self, batch: Mapping[str, Any]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        missing = sorted(key for key in self.required_keys if key not in batch)
        if missing:
            raise KeyError(
                "TemplateFeatureFactory missing required fields: " + ", ".join(missing)
            )
        values = {key: batch[key] for key in self.required_keys}
        non_tensors = [key for key, value in values.items() if not isinstance(value, Tensor)]
        if non_tensors:
            raise TypeError(
                "TemplateFeatureFactory fields must be tensors: "
                + ", ".join(sorted(non_tensors))
            )

        token_mask = batch["token_mask"]
        mol_type_id = batch["mol_type_id"]
        group_id = batch["template_group_id"]
        coords = batch["template_protein_atom37_coords_nm"]
        atom_mask = batch["template_protein_atom37_mask"]
        chain_role = batch["antibody_chain_role_id"]
        region_id = batch["antibody_region_id"]
        if token_mask.ndim != 2:
            raise ValueError(f"token_mask must have shape [B,N], got {tuple(token_mask.shape)}")
        batch_size, n_token = token_mask.shape
        expected_token_shape = (batch_size, n_token)
        for name, value in (
            ("mol_type_id", mol_type_id),
            ("template_group_id", group_id),
            ("antibody_chain_role_id", chain_role),
            ("antibody_region_id", region_id),
        ):
            if value.shape != expected_token_shape:
                raise ValueError(
                    f"{name} must have shape {expected_token_shape}, got {tuple(value.shape)}"
                )
        expected_coord_shape = (batch_size, n_token, len(PROTEIN_ATOM37), 3)
        if coords.shape != expected_coord_shape:
            raise ValueError(
                "template_protein_atom37_coords_nm must have shape "
                f"{expected_coord_shape}, got {tuple(coords.shape)}"
            )
        expected_atom_mask_shape = (batch_size, n_token, len(PROTEIN_ATOM37))
        if atom_mask.shape != expected_atom_mask_shape:
            raise ValueError(
                "template_protein_atom37_mask must have shape "
                f"{expected_atom_mask_shape}, got {tuple(atom_mask.shape)}"
            )
        if not torch.is_floating_point(coords):
            raise TypeError("template_protein_atom37_coords_nm must be floating point")
        return (
            token_mask,
            mol_type_id,
            group_id,
            coords,
            atom_mask,
            chain_role,
            region_id,
        )

    def _prepare_geometry(
        self, batch: Mapping[str, Any]
    ) -> tuple[Tensor, ...]:
        (
            token_mask,
            mol_type_id,
            group_id,
            coords,
            atom_mask,
            chain_role,
            region_id,
        ) = self._validate_batch(batch)
        device = token_mask.device
        dtype = flow_float_dtype(batch)

        token_mask = token_mask.to(device=device).bool()
        mol_type_id = mol_type_id.to(device=device).long()
        group_id = group_id.to(device=device).long()
        chain_role = chain_role.to(device=device).long()
        region_id = region_id.to(device=device).long()
        # Sanitize before subtraction/norm/frame construction. Invalid slots
        # are zeroed below, so hidden NaN/Inf values cannot enter any feature.
        coords = coords.to(device=device, dtype=dtype)
        coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        atom_mask = atom_mask.to(device=device).bool()

        protein = mol_type_id == self._PROTEIN_ID
        group_positive = group_id > 0
        eligible_token = token_mask & protein & group_positive
        visible_atom_mask = atom_mask & eligible_token[..., None]
        coords = torch.where(
            visible_atom_mask[..., None], coords, torch.zeros_like(coords)
        )

        frame_coords = coords.index_select(-2, self._frame_atom_slots)
        frame_mask = visible_atom_mask.index_select(-1, self._frame_atom_slots)
        n_coord, ca_coord, c_coord, _o_coord = frame_coords.unbind(dim=-2)
        n_visible, ca_visible, c_visible, _o_visible = frame_mask.unbind(dim=-1)

        ca_valid = ca_visible
        template_any = visible_atom_mask.any(dim=-1)
        sample_has_template = template_any.any(dim=-1, keepdim=True)
        # Region/role metadata remains useful for hidden geometry rows, but
        # only inside a sample that contains at least one usable template.
        single_mask = token_mask & protein & (
            (eligible_token & template_any)
            | ((chain_role > 0) & sample_has_template)
        )

        eps = self.frame_eps
        x_raw = c_coord - ca_coord
        x_norm = torch.linalg.vector_norm(x_raw, dim=-1)
        x_axis = x_raw / x_norm.clamp_min(eps)[..., None]
        n_raw = n_coord - ca_coord
        n_perp = n_raw - (n_raw * x_axis).sum(dim=-1, keepdim=True) * x_axis
        z_raw = torch.cross(x_axis, n_perp, dim=-1)
        z_norm = torch.linalg.vector_norm(z_raw, dim=-1)
        z_axis = z_raw / z_norm.clamp_min(eps)[..., None]
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        frame = torch.stack((x_axis, y_axis, z_axis), dim=-1)
        frame_valid = (
            n_visible
            & ca_visible
            & c_visible
            & (x_norm > eps)
            & (torch.linalg.vector_norm(n_perp, dim=-1) > eps)
            & (z_norm > eps)
        )
        frame = torch.where(frame_valid[..., None, None], frame, torch.zeros_like(frame))

        return (
            token_mask,
            group_id,
            chain_role,
            region_id,
            frame_coords,
            frame_mask,
            frame,
            frame_valid,
            ca_coord,
            ca_valid,
            eligible_token,
            template_any,
            single_mask,
            dtype,
        )

    def _single_from_geometry(self, geometry: tuple[Tensor, ...]) -> Tensor:
        (
            _token_mask,
            _group_id,
            chain_role,
            region_id,
            _frame_coords,
            _frame_mask,
            _frame,
            frame_valid,
            _ca_coord,
            _ca_valid,
            eligible_token,
            template_any,
            single_mask,
            dtype,
        ) = geometry
        role = chain_role.clamp(min=0, max=2)
        region = region_id.clamp(min=0, max=4)
        role_feature = F.one_hot(role, num_classes=3).to(dtype=dtype)
        region_feature = F.one_hot(region, num_classes=5).to(dtype=dtype)
        validity_feature = torch.stack(
            (eligible_token, template_any, frame_valid), dim=-1
        ).to(dtype=dtype)
        single_features = torch.cat(
            (role_feature, region_feature, validity_feature), dim=-1
        )
        # No-template samples and non-design/padding rows are exact zeros.
        single_features = single_features * single_mask[..., None].to(dtype)
        return torch.nan_to_num(single_features, nan=0.0, posinf=0.0, neginf=0.0)

    def _pair_from_geometry(self, geometry: tuple[Tensor, ...]) -> Tensor:
        (
            _token_mask,
            group_id,
            _chain_role,
            _region_id,
            frame_coords,
            frame_mask,
            frame,
            frame_valid,
            ca_coord,
            ca_valid,
            eligible_token,
            template_any,
            _single_mask,
            dtype,
        ) = geometry
        group_positive = group_id > 0
        same_group = group_id[:, :, None] == group_id[:, None, :]
        pair_mask = (
            eligible_token[:, :, None]
            & eligible_token[:, None, :]
            & same_group
            & group_positive[:, :, None]
            & group_positive[:, None, :]
            & template_any[:, :, None]
            & template_any[:, None, :]
        )

        ca_i = ca_coord[:, :, None, :]
        ca_j = ca_coord[:, None, :, :]
        ca_distance = torch.linalg.vector_norm(ca_i - ca_j, dim=-1)
        ca_pair_valid = pair_mask & ca_valid[:, :, None] & ca_valid[:, None, :]
        ca_distance = torch.where(ca_pair_valid, ca_distance, torch.zeros_like(ca_distance))
        centers = self.rbf_centers.to(device=ca_distance.device, dtype=dtype)
        scales = self.rbf_scales.to(device=ca_distance.device, dtype=dtype)
        rbf = torch.exp(
            -0.5
            * (
                (ca_distance[..., None, None] - centers[None, None, None, :, None])
                / scales[None, None, None, None, :]
            )
            ** 2
        ).flatten(start_dim=-2)
        rbf = rbf * ca_pair_valid[..., None].to(dtype)

        # Ordered atom-pair distances retain directional token-pair detail;
        # validity channels distinguish missing atoms from geometric zero.
        ordered_delta = (
            frame_coords[:, :, None, :, None, :]
            - frame_coords[:, None, :, None, :, :]
        )
        ordered_distances = torch.linalg.vector_norm(ordered_delta, dim=-1)
        ordered_valid = (
            frame_mask[:, :, None, :, None] & frame_mask[:, None, :, None, :]
        )
        ordered_distances = torch.where(
            ordered_valid, ordered_distances, torch.zeros_like(ordered_distances)
        ).flatten(start_dim=-2)
        ordered_valid_feature = ordered_valid.to(dtype=dtype).flatten(start_dim=-2)

        frame_pair_valid = frame_valid[:, :, None] & frame_valid[:, None, :] & pair_mask
        frame_i = frame[:, :, None, :, :]
        frame_j = frame[:, None, :, :, :]
        relative_frame = torch.matmul(frame_i.transpose(-1, -2), frame_j)
        relative_frame = torch.where(
            frame_pair_valid[..., None, None],
            relative_frame,
            torch.zeros_like(relative_frame),
        ).flatten(start_dim=-2)
        displacement = ca_coord[:, None, :, :] - ca_coord[:, :, None, :]
        direction = torch.matmul(
            frame_i.transpose(-1, -2), displacement[..., None]
        ).squeeze(-1)
        direction = torch.where(
            frame_pair_valid[..., None], direction, torch.zeros_like(direction)
        )

        pair_features = torch.cat(
            (
                ca_distance[..., None],
                rbf,
                ordered_distances,
                ordered_valid_feature,
                relative_frame,
                direction,
                frame_pair_valid[..., None].to(dtype),
                pair_mask[..., None].to(dtype),
            ),
            dim=-1,
        )
        pair_features = torch.nan_to_num(
            pair_features, nan=0.0, posinf=0.0, neginf=0.0
        )
        return pair_features * pair_mask[..., None].to(dtype)

    def build_single_features(self, batch: Mapping[str, Any]) -> Tensor:
        """Return the 11-channel masked single-token template feature."""

        return self._single_from_geometry(self._prepare_geometry(batch))

    def build_pair_features(self, batch: Mapping[str, Any]) -> Tensor:
        """Return the 111-channel masked same-group pair template feature."""

        return self._pair_from_geometry(self._prepare_geometry(batch))

    def build_features(self, batch: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
        """Return both raw streams, sharing one geometry preparation pass."""

        geometry = self._prepare_geometry(batch)
        return self._single_from_geometry(geometry), self._pair_from_geometry(geometry)

    def build_feature_masks(self, batch: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
        """Return the token/pair masks used by the corresponding raw streams."""

        geometry = self._prepare_geometry(batch)
        token_mask, group_id, _role, _region, _fc, _fm, _frame, _fv, _ca, _cv, eligible, any_, single_mask, _dtype = geometry
        group_positive = group_id > 0
        pair_mask = (
            eligible[:, :, None]
            & eligible[:, None, :]
            & (group_id[:, :, None] == group_id[:, None, :])
            & group_positive[:, :, None]
            & group_positive[:, None, :]
            & any_[:, :, None]
            & any_[:, None, :]
        )
        return single_mask, pair_mask

    def forward(
        self, batch: Mapping[str, Any] | None = None, **fields: Any
    ) -> tuple[Tensor, Tensor]:
        if batch is None:
            batch = fields
        elif fields:
            raise TypeError("pass either a batch mapping or keyword fields, not both")
        if not isinstance(batch, Mapping):
            raise TypeError("TemplateFeatureFactory expects a batch mapping")
        return self.build_features(batch)


class TemplateSingleSeqFeat(Feature):
    """Registry wrapper for the 11-channel template single feature."""

    required_keys = TemplateFeatureFactory.required_keys

    def __init__(self, **_kwargs: Any):
        super().__init__(dim=TEMPLATE_SINGLE_DIM)
        self.factory = TemplateFeatureFactory()

    def forward(self, batch: Mapping[str, Any]) -> Tensor:
        return self.factory.build_single_features(batch)


class TemplatePairFeat(Feature):
    """Registry wrapper for the 111-channel template pair feature."""

    required_keys = TemplateFeatureFactory.required_keys

    def __init__(self, **_kwargs: Any):
        super().__init__(dim=TEMPLATE_PAIR_DIM)
        self.factory = TemplateFeatureFactory()

    def forward(self, batch: Mapping[str, Any]) -> Tensor:
        return self.factory.build_pair_features(batch)


__all__ = [
    "TEMPLATE_PAIR_DIM",
    "TEMPLATE_SINGLE_DIM",
    "TemplateFeatureFactory",
    "TemplatePairFeat",
    "TemplateSingleSeqFeat",
]
