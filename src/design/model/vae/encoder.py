from typing import Dict

import torch

from src.design.model.networks.feature import FeatureFactory
from src.design.model.networks.modules.attn_n_transition import MultiheadAttnAndTransition
from src.design.model.networks.modules.pair_update import PairReprUpdate
from src.design.model.networks.modules.seq_transition_af3 import Transition


class EncoderTransformer(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        enc_cfg = kwargs["encoder"]
        self.nlayers = int(enc_cfg["nlayers"])
        self.token_dim = int(enc_cfg["token_dim"])
        self.pair_repr_dim = int(enc_cfg["pair_repr_dim"])
        self.dim_cond = int(enc_cfg["dim_cond"])
        self.update_pair_repr = bool(enc_cfg["update_pair_repr"])
        self.update_pair_repr_every_n = int(enc_cfg["update_pair_repr_every_n"])
        self.use_tri_mult = bool(enc_cfg["use_tri_mult"])
        if self.update_pair_repr or self.use_tri_mult:
            raise ValueError(
                "the fixed standalone VAE config requires "
                "update_pair_repr=false and use_tri_mult=false"
            )
        self.normalize_latent = bool(enc_cfg.get("normalize_latent", False))
        self.latent_z_dim = int(kwargs["latent_z_dim"])

        self.init_repr_factory = FeatureFactory(
            feats=enc_cfg["feats_seq"],
            dim_feats_out=self.token_dim,
            use_ln_out=bool(enc_cfg.get("use_ln_seq", False)),
            mode="seq",
            **enc_cfg,
        )

        self.cond_factory = FeatureFactory(
            feats=enc_cfg["feats_cond_seq"],
            dim_feats_out=self.dim_cond,
            use_ln_out=bool(enc_cfg.get("use_ln_cond", False)),
            mode="seq",
            **enc_cfg,
        )

        self.pair_rep_factory = FeatureFactory(
            feats=enc_cfg["feats_pair_repr"],
            dim_feats_out=self.pair_repr_dim,
            use_ln_out=bool(enc_cfg.get("use_ln_pair", False)),
            mode="pair",
            **enc_cfg,
        )
        self.required_batch_keys = frozenset(
            {"token_mask"}
            | set(self.init_repr_factory.required_keys)
            | set(self.cond_factory.required_keys)
            | set(self.pair_rep_factory.required_keys)
        )

        self.transition_c_1 = Transition(self.dim_cond, expansion_factor=2)
        self.transition_c_2 = Transition(self.dim_cond, expansion_factor=2)

        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.token_dim,
                    dim_pair=self.pair_repr_dim,
                    nheads=int(enc_cfg["nheads"]),
                    dim_cond=self.dim_cond,
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=bool(enc_cfg["use_qkln"]),
                )
                for _ in range(self.nlayers)
            ]
        )

        if self.update_pair_repr:
            self.pair_update_layers = torch.nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=self.token_dim,
                            pair_dim=self.pair_repr_dim,
                            use_tri_mult=self.use_tri_mult,
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        self.latent_decoder_mean_n_log_scale = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, 2 * self.latent_z_dim, bias=False),
        )

        self.ln_z = (
            torch.nn.LayerNorm(self.latent_z_dim)
            if self.normalize_latent
            else torch.nn.Identity()
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "token_mask" in batch, "`token_mask` not in batch"
        mask = batch["token_mask"].bool()

        cond = self.cond_factory(batch)
        cond = self.transition_c_2(self.transition_c_1(cond, mask), mask)

        seqs = self.init_repr_factory(batch) * mask[..., None].to(dtype=cond.dtype)
        pair_rep = self.pair_rep_factory(batch)

        for i, layer in enumerate(self.transformer_layers):
            seqs = layer(seqs, pair_rep, cond, mask)
            if (
                self.update_pair_repr
                and i < self.nlayers - 1
                and self.pair_update_layers[i] is not None
            ):
                pair_rep = self.pair_update_layers[i](seqs, pair_rep, mask)

        flat_out = self.latent_decoder_mean_n_log_scale(seqs) * mask[..., None].to(
            seqs.dtype
        )
        mean, log_scale = torch.chunk(flat_out, chunks=2, dim=-1)

        return {
            "mean": mean,
            "log_scale": log_scale,
        }
