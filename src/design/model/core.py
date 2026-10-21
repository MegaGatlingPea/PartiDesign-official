"""Checkpoint-compatible FM/VAE core used by standalone inference."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from src.design.model.constants import (
    LIGAND_BOND_TYPES,
    MOL_TYPE_TO_ID,
    TOKEN_TYPE_IGNORE_INDEX,
    UNKNOWN_ELEMENT_ID,
)
from src.design.model.networks.local_latents import LocalLatentsTransformer
from src.design.model.solver import (
    STATE_COMPONENTS,
    STATE_DIMS,
    ProductSpaceSolver,
    validate_context_constraint_mode,
    validate_sampling_config,
)
from src.design.model.vae.autoencoder import InferenceAutoencoder


_VAE_PAIR_FIELDS = frozenset(("token_bond_type", "token_bond_pair_mask"))
_VAE_PAD_VALUES: Mapping[str, float | int | bool] = {
    "mol_type_id": MOL_TYPE_TO_ID["protein"],
    "token_type32": TOKEN_TYPE_IGNORE_INDEX,
    "ligand_element_type": UNKNOWN_ELEMENT_ID,
    "token_bond_type": LIGAND_BOND_TYPES["none"],
}


class PartiDesignInferenceModel(torch.nn.Module):
    """One combined-checkpoint model with exact ``nn`` and ``autoencoder`` roots."""

    def __init__(
        self,
        *,
        fm_network_config: Mapping[str, Any],
        ae_network_config: Mapping[str, Any],
        context_constraint_mode: str = "rigid",
    ) -> None:
        super().__init__()
        self._context_constraint_mode = validate_context_constraint_mode(
            context_constraint_mode
        )
        self.autoencoder = InferenceAutoencoder(ae_network_config)
        self.nn = LocalLatentsTransformer(**fm_network_config)
        if self.autoencoder.latent_dim != STATE_DIMS["local_latents"]:
            raise ValueError("autoencoder latent dimension must be 8")
        if int(fm_network_config["latent_z_dim"]) != self.autoencoder.latent_dim:
            raise ValueError("FM and autoencoder latent dimensions differ")
        self.solver = ProductSpaceSolver(self._context_constraint_mode)

    @property
    def context_constraint_mode(self) -> str:
        return self._context_constraint_mode

    @property
    def rigid_context(self) -> bool:
        return self._context_constraint_mode == "rigid"

    @property
    def supports_bond_order_conditioning(self) -> bool:
        """Whether this checkpoint actually consumes the five-state bond ABI."""

        builder = self.nn.pair_repr_builder
        factories = (builder.init_repr_factory, builder.cond_factory)
        return any(
            factory is not None
            and "condition_bond_order_pair" in factory.feature_names
            for factory in factories
        )

    @staticmethod
    def _design_context_masks(
        batch: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor]:
        token_mask = batch["token_mask"].bool()
        design_mask = batch["design_token_mask"].bool()
        if design_mask.shape != token_mask.shape:
            raise ValueError("design_token_mask must match token_mask")
        if bool((design_mask & ~token_mask).any().item()):
            raise ValueError("design_token_mask contains padded tokens")
        if not bool(design_mask.any(dim=-1).all().item()):
            raise ValueError("every sample must contain at least one design token")
        return design_mask, token_mask & ~design_mask

    def _masked_autoencoder_input(
        self,
        batch: Mapping[str, Any],
        active_mask: Tensor,
    ) -> dict[str, Tensor]:
        token_mask = batch["token_mask"].bool()
        if active_mask.dtype is not torch.bool or active_mask.shape != token_mask.shape:
            raise TypeError("active VAE mask must be a [B, N] bool tensor")
        if bool((active_mask & ~token_mask).any().item()):
            raise ValueError("active VAE mask contains padded tokens")

        result: dict[str, Tensor] = {}
        for key in sorted(self.autoencoder.encoder.required_batch_keys):
            value = batch.get(key)
            if not isinstance(value, Tensor):
                raise TypeError(f"canonical VAE field {key!r} must be a tensor")
            if key == "token_mask":
                result[key] = active_mask
                continue
            if key in _VAE_PAIR_FIELDS:
                visible = active_mask[:, :, None] & active_mask[:, None, :]
            else:
                visible = active_mask
            while visible.ndim < value.ndim:
                visible = visible.unsqueeze(-1)
            pad_value = _VAE_PAD_VALUES.get(key, 0)
            result[key] = torch.where(
                visible,
                value,
                torch.full_like(value, pad_value),
            )
        return result

    def _encode_fixed_context_latents(
        self,
        batch: Mapping[str, Any],
        context_mask: Tensor,
    ) -> Tensor:
        token_mask = batch["token_mask"].bool()
        token_center = batch["token_center_coords_nm"]
        batch_size, token_count = token_mask.shape
        fixed = torch.zeros(
            batch_size,
            token_count,
            STATE_DIMS["local_latents"],
            dtype=token_center.dtype,
            device=token_center.device,
        )
        active_rows = context_mask.any(dim=-1)
        if not bool(active_rows.any().item()):
            return fixed

        row_batch: dict[str, Tensor] = {}
        for key in sorted(self.autoencoder.encoder.required_batch_keys):
            value = batch.get(key)
            if not isinstance(value, Tensor):
                raise TypeError(f"canonical VAE field {key!r} must be a tensor")
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(f"canonical VAE field {key!r} has invalid batch axis")
            row_batch[key] = value[active_rows]
        row_context_mask = context_mask[active_rows]
        encoder_input = self._masked_autoencoder_input(
            row_batch,
            row_context_mask,
        )
        statistics = self.autoencoder.encode_statistics(encoder_input)
        context_mean = statistics["mean"]
        expected = row_context_mask.shape + (STATE_DIMS["local_latents"],)
        if context_mean.shape != expected:
            raise ValueError("VAE context latent mean has an invalid shape")
        context_latent = self.autoencoder.encoder.ln_z(context_mean)
        context_latent = context_latent * row_context_mask[..., None].to(
            context_latent.dtype
        )
        fixed[active_rows] = context_latent.to(dtype=fixed.dtype)
        return fixed

    def configure_optimization(self, config: Mapping[str, Any]) -> dict[str, Any]:
        from .optimization import configure_optimization
        return configure_optimization(self, config)

    def _prepare_inference_optimization(self, result: dict[str, Any]) -> dict[str, Any]:
        if getattr(self, "_inference_optimization_config", None) is None:
            return result
        from .optimization import ensure_optimization_device

        token_mask = result.get("token_mask")
        if isinstance(token_mask, Tensor):
            ensure_optimization_device(self, token_mask.device)
        return result

    def add_sampling_context(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(batch)
        if not self.rigid_context:
            result.pop("x_context", None)
            return self._prepare_inference_optimization(result)
        if "x_context" in result:
            return self._prepare_inference_optimization(result)
        _, context_mask = self._design_context_masks(result)
        token_center = result["token_center_coords_nm"]
        context_latents = self._encode_fixed_context_latents(result, context_mask)
        result["x_context"] = {
            "token_center": token_center
            * context_mask[..., None].to(token_center.dtype),
            "local_latents": context_latents,
        }
        return self._prepare_inference_optimization(result)

    def call_nn(
        self,
        batch: Mapping[str, Any],
    ) -> dict[str, dict[str, Tensor]]:
        missing = sorted(self.nn.required_batch_keys.difference(batch))
        if missing:
            raise KeyError(f"Missing FM network fields: {', '.join(missing)}")
        network_batch: dict[str, Any] = {}
        for key in sorted(self.nn.required_batch_keys):
            value = batch[key]
            if key in {"x_t", "x_sc", "t"}:
                if not isinstance(value, Mapping):
                    raise TypeError(f"FM network field {key!r} must be a mapping")
                nested: dict[str, Tensor] = {}
                for component in STATE_COMPONENTS:
                    tensor = value.get(component)
                    if not isinstance(tensor, Tensor):
                        raise TypeError(
                            f"FM network field {key}.{component} must be a tensor"
                        )
                    nested[component] = tensor
                network_batch[key] = nested
            else:
                if not isinstance(value, Tensor):
                    raise TypeError(f"FM network field {key!r} must be a tensor")
                network_batch[key] = value
        # Inference-only topology metadata is not part of the frozen canonical
        # training schema/required feature keys, but must survive this whitelist.
        if "cyclic_period" in batch:
            period = batch["cyclic_period"]
            if not isinstance(period, Tensor) or period.dtype != torch.long:
                raise TypeError("cyclic_period must be a long tensor")
            if period.shape != batch["token_mask"].shape:
                raise ValueError("cyclic_period must have shape [batch, tokens]")
            network_batch["cyclic_period"] = period
        return self.nn(network_batch)

    def predict_for_sampling(
        self,
        batch: Mapping[str, Any],
    ) -> dict[str, dict[str, Tensor]]:
        network_batch = dict(batch)
        has_self_condition = "x_sc" in batch
        if not has_self_condition:
            if self.rigid_context:
                network_batch["x_sc"] = {
                    component: batch["x_context"][component].detach().clone()
                    for component in STATE_COMPONENTS
                }
            else:
                network_batch["x_sc"] = {
                    component: torch.zeros_like(batch["x_t"][component])
                    for component in STATE_COMPONENTS
                }
        network_batch["x_sc_valid"] = torch.full(
            (batch["mask"].shape[0],),
            has_self_condition,
            dtype=torch.bool,
            device=batch["mask"].device,
        )
        return self.call_nn(network_batch)

    @torch.no_grad()
    def generate(
        self,
        batch: Mapping[str, Any],
        sampling_config: Mapping[str, Any],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        validate_sampling_config(sampling_config)
        sampling_batch = self.add_sampling_context(batch)
        sampling_batch["mask"] = sampling_batch["token_mask"].bool()
        return self.solver.full_simulation(
            batch=sampling_batch,
            predict_for_sampling=self.predict_for_sampling,
            sampling_config=sampling_config,
        )

    def decode_generated(
        self,
        canonical_batch: Mapping[str, Any],
        generated: Mapping[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        token_mask = canonical_batch["token_mask"].bool()
        if generated["token_center"].shape != token_mask.shape + (3,):
            raise ValueError("generated token centers have an invalid shape")
        if generated["local_latents"].shape != token_mask.shape + (8,):
            raise ValueError("generated local latents have an invalid shape")
        decoder_batch = dict(canonical_batch)
        vae_dtype = next(self.autoencoder.parameters()).dtype
        decoder_batch["token_center_coords_nm"] = generated["token_center"].to(
            dtype=vae_dtype
        )
        local_latents = generated["local_latents"].to(dtype=vae_dtype)
        with torch.no_grad():
            output = self.autoencoder.decode(local_latents, decoder_batch)
        return output, decoder_batch
