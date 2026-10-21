"""Inference-only universal VAE wrapper.

The training LightningModule also owns optimization, logging, validation dumps, and
diagnostic losses.  None of those participate in standalone inference.  This module
retains the exact ``encoder.*``/``decoder.*`` state namespace and only the encode/decode
operations consumed by the flow sampler.
"""

from collections.abc import Mapping
from typing import Any

import torch

from src.design.model.vae.decoder import DecoderTransformer
from src.design.model.vae.encoder import EncoderTransformer


class InferenceAutoencoder(torch.nn.Module):
    """Frozen VAE with the checkpoint-compatible module hierarchy."""

    def __init__(self, network_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.encoder = EncoderTransformer(**network_config)
        self.decoder = DecoderTransformer(**network_config)
        self.latent_dim = int(network_config["latent_z_dim"])

    @staticmethod
    def _module_input(
        module: torch.nn.Module,
        batch: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        required_keys = module.required_batch_keys
        missing = sorted(required_keys.difference(batch))
        if missing:
            raise KeyError(f"Missing VAE network fields: {', '.join(missing)}")
        module_input: dict[str, torch.Tensor] = {}
        for key in sorted(required_keys):
            value = batch[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"VAE network field {key!r} must be a tensor")
            module_input[key] = value
        return module_input

    def encode_statistics(
        self,
        batch: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        encoder_input = self._module_input(self.encoder, batch)
        output = self.encoder(encoder_input)
        mean = output["mean"]
        log_scale = output["log_scale"]
        token_mask = encoder_input["token_mask"].bool()
        if mean.shape != log_scale.shape or mean.shape[:-1] != token_mask.shape:
            raise ValueError("invalid VAE encoder output shape")
        return {"mean": mean, "log_scale": log_scale}

    def decode(
        self,
        z_latent: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        decoder_input = dict(batch)
        decoder_input["z_latent"] = z_latent
        return self.decoder(self._module_input(self.decoder, decoder_input))
