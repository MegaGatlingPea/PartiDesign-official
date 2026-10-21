import torch

from src.design.model.networks.feature import FeatureFactory
from src.design.model.networks.modules.adaptive_ln_scale import AdaptiveLayerNorm


class PairReprBuilder(torch.nn.Module):
    """
    Builds initial pair representation. Essentially the pair feature factory, but potentially with
    an adaptive layer norm layer as well.
    """

    def __init__(self, feats_repr, feats_cond, dim_feats_out, dim_cond_pair, **kwargs):
        super().__init__()
        self.broadcast_time_condition = False

        self.init_repr_factory = FeatureFactory(
            feats=feats_repr,
            dim_feats_out=dim_feats_out,
            use_ln_out=True,
            mode="pair",
            **kwargs,
        )

        self.cond_factory = None  # Build a pair feature for conditioning and use it for adaln the pair representation
        if feats_cond is not None:
            if len(feats_cond) > 0:
                self.cond_factory = FeatureFactory(
                    feats=feats_cond,
                    dim_feats_out=dim_cond_pair,
                    use_ln_out=True,
                    mode="pair",
                    **kwargs,
                )
                self.adaln = AdaptiveLayerNorm(
                    dim=dim_feats_out, dim_cond=dim_cond_pair
                )

        required_keys = set(self.init_repr_factory.required_keys)
        if self.cond_factory is not None:
            required_keys.update(self.cond_factory.required_keys)
        required_keys.add("token_mask")
        self.required_keys = frozenset(required_keys)

    def forward(self, batch_nn):
        mask = batch_nn["token_mask"].bool()  # [b, n]
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        repr = self.init_repr_factory(batch_nn)  # [b, n, n, dim_feats_out]
        if self.cond_factory is not None:
            time_only = all(
                name in {"time_emb_token_center", "time_emb_local_latents"}
                for name in self.cond_factory.feature_names
            )
            if self.broadcast_time_condition and not self.training and not torch.is_grad_enabled() and time_only:
                # Time changes every step, but is constant across the pair plane
                # within a step. Compute its learned projections once per batch
                # row; keep feature/parameter order and the final padding mask.
                # A repeated tile retains the large-M BF16 GEMM behavior on
                # the validated Ampere workloads. A 16x16 tile had rare
                # projection rounding differences across the 400-step schedule;
                # use 32x32 and slice only after the learned projections.
                # This internal projection tile does not limit input length
                # or batch size; every pair receives the projected condition.
                small_batch = dict(batch_nn)
                small_batch["token_mask"] = torch.ones_like(mask[:, :32])
                cond = self.cond_factory(small_batch)
                normed_cond = self.adaln.norm_cond(cond)
                gamma = self.adaln.to_gamma(normed_cond)[:, :1, :1]
                beta = self.adaln.to_beta(normed_cond)[:, :1, :1]
                repr = (self.adaln.norm(repr) * gamma + beta) * pair_mask[..., None]
            else:
                cond = self.cond_factory(batch_nn)  # [b, n, n, dim_cond]
                repr = self.adaln(repr, cond, pair_mask)
        return repr
