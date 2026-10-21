from collections.abc import Mapping
from typing import Any, Literal

import torch
from torch.nn import functional as F

from src.design.model.constants import MOL_TYPE_TO_ID


FeatureMode = Literal["seq", "pair"]
FeatureRegistryName = Literal["vae", "flow"]


def floating_one_hot(indices: torch.Tensor, num_classes: int, *,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Construct the final floating tensor without an int64 channel temporary."""
    return torch.zeros((*indices.shape, num_classes), device=indices.device,
                       dtype=dtype).scatter_(-1, indices.unsqueeze(-1), 1.0)


def bin_and_one_hot(tensor: torch.Tensor, bin_limits: torch.Tensor, *,
                    floating: bool = False) -> torch.Tensor:
    bin_indices = torch.bucketize(tensor.contiguous(), bin_limits.contiguous())
    if floating:
        return floating_one_hot(bin_indices, len(bin_limits) + 1)
    return F.one_hot(bin_indices, len(bin_limits) + 1).to(dtype=torch.float32)


def flow_float_dtype(batch: Mapping[str, Any]) -> torch.dtype:
    for outer, inner_keys in (
        ("x_t", ("token_center", "local_latents")),
        ("x_sc", ("token_center", "local_latents")),
    ):
        if outer in batch:
            nested = batch[outer]
            for inner in inner_keys:
                if inner in nested and torch.is_floating_point(nested[inner]):
                    return nested[inner].dtype
    for key in (
        "optional_token_center_coords_nm",
        "protein_motif_atom37_coords_nm",
        "rna_motif_atom27_coords_nm",
        "dna_motif_atom27_coords_nm",
        "ligand_motif_token_center_coords_nm",
    ):
        if key in batch and torch.is_floating_point(batch[key]):
            return batch[key].dtype
    return torch.float32


def flow_token_mask(batch: Mapping[str, Any]) -> torch.Tensor:
    mask = batch["token_mask"]
    assert isinstance(mask, torch.Tensor) and mask.ndim == 2
    return mask.bool()


def flow_pair_mask(batch: Mapping[str, Any]) -> torch.Tensor:
    mask = flow_token_mask(batch)
    return mask[:, :, None] & mask[:, None, :]


def flow_modality_mask(batch: Mapping[str, Any], modality: str) -> torch.Tensor:
    assert modality in MOL_TYPE_TO_ID, modality
    return flow_token_mask(batch) & (
        batch["mol_type_id"] == MOL_TYPE_TO_ID[modality]
    )


def flatten_coords_with_mask(
    coords: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    assert coords.shape[:-1] == mask.shape, (coords.shape, mask.shape)
    safe_coords = torch.where(mask[..., None], coords, torch.zeros_like(coords))
    return torch.cat(
        [safe_coords.flatten(start_dim=-2), mask.to(dtype=coords.dtype)], dim=-1
    )


class Feature(torch.nn.Module):
    required_keys: frozenset[str] = frozenset()

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.float_one_hot = False

    def _use_float_one_hot(self) -> bool:
        return self.float_one_hot and not self.training and not torch.is_grad_enabled()

    def one_hot(self, indices: torch.Tensor, num_classes: int,
                dtype: torch.dtype) -> torch.Tensor:
        if self._use_float_one_hot():
            return floating_one_hot(indices, num_classes, dtype=dtype)
        return F.one_hot(indices, num_classes=num_classes).to(dtype)

    def get_dim(self) -> int:
        return self.dim

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def extract_bs_and_n(self, batch: Mapping[str, Any]) -> tuple[int, int]:
        if "token_mask" in batch:
            tensor = batch["token_mask"]
        elif "x_t" in batch and "token_center" in batch["x_t"]:
            tensor = batch["x_t"]["token_center"]
        else:
            raise IOError("Cannot infer batch/token dimensions for feature")
        return tensor.shape[0], tensor.shape[1]

    def extract_device(self, batch: Mapping[str, Any]) -> torch.device:
        if "token_mask" in batch:
            return batch["token_mask"].device
        if "x_t" in batch and "token_center" in batch["x_t"]:
            return batch["x_t"]["token_center"].device
        raise IOError("Cannot infer device for feature")
