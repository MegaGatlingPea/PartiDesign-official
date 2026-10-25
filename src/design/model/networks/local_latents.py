from collections.abc import Mapping
from typing import Any

import torch

from src.design.model.solver import STATE_COMPONENTS, STATE_DIMS
from src.design.model.networks.feature import FeatureFactory
from src.design.model.networks.modules.attn_n_transition import MultiheadAttnAndTransition
from src.design.model.networks.modules.pair_bias_bank import PairBiasBank
from src.design.model.networks.modules.pair_rep_initial import PairReprBuilder
from src.design.model.networks.modules.pair_update import PairReprUpdate
from src.design.model.networks.modules.seq_transition_af3 import Transition


UNINDEXED_MOTIF_MAX_TOKENS = 320


def _unindexed_motif_config(kwargs: Mapping[str, Any]) -> tuple[bool, int]:
    raw = kwargs.get("unindexed_motif")
    if raw is None:
        return False, UNINDEXED_MOTIF_MAX_TOKENS
    if not isinstance(raw, Mapping) or set(raw) != {"enabled", "max_tokens"}:
        raise ValueError(
            "network.unindexed_motif must contain exactly enabled and max_tokens"
        )
    enabled = raw["enabled"]
    max_tokens = raw["max_tokens"]
    if type(enabled) is not bool:
        raise TypeError("network.unindexed_motif.enabled must be boolean")
    if type(max_tokens) is not int or max_tokens != UNINDEXED_MOTIF_MAX_TOKENS:
        raise ValueError("network.unindexed_motif.max_tokens must be exactly 320")
    return enabled, max_tokens


class LocalLatentsTransformer(torch.nn.Module):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.nlayers = int(kwargs["nlayers"])
        self.token_dim = int(kwargs["token_dim"])
        self.nheads = int(kwargs["nheads"])
        self.pair_repr_dim = int(kwargs["pair_repr_dim"])
        self.dim_cond = int(kwargs["dim_cond"])
        self.update_pair_repr = bool(kwargs["update_pair_repr"])
        self.update_pair_repr_every_n = int(kwargs["update_pair_repr_every_n"])
        self.use_tri_mult = bool(kwargs["use_tri_mult"])
        self.use_qkln = bool(kwargs["use_qkln"])
        self.parallel_mha_transition = bool(
            kwargs.get("parallel_mha_transition", False)
        )
        (
            self.unindexed_motif_enabled,
            self.unindexed_motif_max_tokens,
        ) = _unindexed_motif_config(kwargs)

        assert self.nlayers > 0
        assert self.update_pair_repr_every_n > 0
        assert self.token_dim % self.nheads == 0
        if self.unindexed_motif_enabled:
            raise ValueError(
                "standalone inference keeps unindexed motif concat disabled"
            )
        if self.update_pair_repr or self.use_tri_mult:
            raise ValueError(
                "the standalone FM checkpoint ABI requires "
                "update_pair_repr=false and use_tri_mult=false"
            )

        self.latent_dim = int(kwargs["latent_z_dim"])
        assert self.latent_dim == STATE_DIMS["local_latents"]

        self.init_repr_factory = FeatureFactory(
            feats=kwargs["feats_seq"],
            dim_feats_out=self.token_dim,
            use_ln_out=False,
            mode="seq",
            feature_registry="flow",
            **kwargs,
        )
        self.cond_factory = FeatureFactory(
            feats=kwargs["feats_cond_seq"],
            dim_feats_out=self.dim_cond,
            use_ln_out=False,
            mode="seq",
            feature_registry="flow",
            **kwargs,
        )
        self.pair_repr_builder = PairReprBuilder(
            feats_repr=kwargs["feats_pair_repr"],
            feats_cond=kwargs["feats_pair_cond"],
            dim_feats_out=self.pair_repr_dim,
            dim_cond_pair=self.dim_cond,
            feature_registry="flow",
            **kwargs,
        )
        self.required_batch_keys = frozenset(
            {"token_mask"}
            | set(self.init_repr_factory.required_keys)
            | set(self.cond_factory.required_keys)
            | set(self.pair_repr_builder.required_keys)
        )

        self.transition_c_1 = Transition(self.dim_cond, expansion_factor=2)
        self.transition_c_2 = Transition(self.dim_cond, expansion_factor=2)
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.token_dim,
                    dim_pair=self.pair_repr_dim,
                    nheads=self.nheads,
                    dim_cond=self.dim_cond,
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=self.parallel_mha_transition,
                    use_attn_pair_bias=True,
                    use_qkln=self.use_qkln,
                )
                for _ in range(self.nlayers)
            ]
        )
        # These are plain references/configuration values.  The modules remain
        # owned by transformer_layers, while no derived pair weights or pair
        # tensors are registered on this network.
        self._attention_modules = tuple(
            layer.mhba.mha for layer in self.transformer_layers
        )
        self._attention_opt_sdpa = False
        self._attention_opt_fused_pair_bias = False
        self._attention_opt_row_chunk_size = 0
        self._attention_opt_layer_group_size = 1
        self._bioir_backend = None
        self._bioir_graph_region = None
        for attention in self._attention_modules:
            attention._attention_use_sdpa = False

        if self.update_pair_repr:
            self.pair_update_layers = torch.nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=self.token_dim,
                            pair_dim=self.pair_repr_dim,
                            use_tri_mult=self.use_tri_mult,
                        )
                        if index % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for index in range(self.nlayers - 1)
                ]
            )

        self.token_center_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(
                self.token_dim,
                STATE_DIMS["token_center"],
                bias=False,
            ),
        )
        self.local_latents_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, self.latent_dim, bias=False),
        )

    def configure_attention_optimization(
        self,
        sdpa: bool = False,
        fused_pair_bias: bool = False,
        row_chunk_size: int = 0,
        layer_group_size: int = 1,
    ) -> "LocalLatentsTransformer":
        """Configure opt-in FM attention paths before ``torch.compile``.

        The flags are ordinary Python attributes and the derived pair bias is
        recomputed inside each evaluation forward.  No option changes the
        checkpoint/state_dict surface.  The optimized paths are inference
        paths; training forwards continue to use the stock attention math.
        """

        if type(sdpa) is not bool:
            raise TypeError("sdpa must be a bool")
        if type(fused_pair_bias) is not bool:
            raise TypeError("fused_pair_bias must be a bool")
        if type(row_chunk_size) is not int or row_chunk_size < 0:
            raise ValueError("row_chunk_size must be a non-negative int")
        if type(layer_group_size) is not int or layer_group_size <= 0:
            raise ValueError("layer_group_size must be a positive int")

        self._attention_opt_sdpa = sdpa
        self._attention_opt_fused_pair_bias = fused_pair_bias
        self._attention_opt_row_chunk_size = row_chunk_size
        self._attention_opt_layer_group_size = layer_group_size
        for attention in self._attention_modules:
            attention._attention_use_sdpa = sdpa
        return self

    def forward(self, batch: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
        if self._bioir_backend is not None and not self.training and not torch.is_grad_enabled():
            return self._bioir_backend.forward(self, batch)
        return self._forward_eager(batch)

    def _forward_eager(self, batch: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
        mask = batch["token_mask"].bool()
        assert mask.ndim == 2
        if self._bioir_backend is not None and not self.training and not torch.is_grad_enabled():
            self._bioir_backend.prepare_forward(mask)
        for component in STATE_COMPONENTS:
            assert batch["x_t"][component].shape == mask.shape + (
                STATE_DIMS[component],
            )

        condition = self.cond_factory(batch)
        condition = self.transition_c_2(
            self.transition_c_1(condition, mask),
            mask,
        )
        sequence = self.init_repr_factory(batch) * mask[..., None]
        pair_representation = self.pair_repr_builder(batch)

        # Pair features are invariant across this FM stack.  In evaluation,
        # the opt-in bank computes only one bounded layer group at a time and
        # passes projected bias through the ordinary layer call.  The default
        # branch intentionally retains the original call shape and arithmetic.
        use_fused_pair_bias = (
            self._attention_opt_fused_pair_bias
            and not self.training
            and not torch.is_grad_enabled()
            and not self.update_pair_repr
        )
        if use_fused_pair_bias:
            group_size = PairBiasBank.resolve_group_size(
                pair_representation,
                self._attention_modules,
                self._attention_opt_layer_group_size,
            )
            use_fused_pair_bias = group_size > 0
        if use_fused_pair_bias:
            for group_start in range(0, self.nlayers, group_size):
                group_stop = min(self.nlayers, group_start + group_size)
                projected_bias = PairBiasBank.compute(
                    pair_representation,
                    self._attention_modules,
                    group_start,
                    group_stop,
                    row_chunk_size=self._attention_opt_row_chunk_size,
                )
                for index in range(group_start, group_stop):
                    head_start = (index - group_start) * self.nheads
                    head_stop = head_start + self.nheads
                    sequence = self.transformer_layers[index](
                        sequence,
                        pair_representation,
                        condition,
                        mask,
                        projected_bias=projected_bias[:, head_start:head_stop],
                    )
                    if self.update_pair_repr and index < self.nlayers - 1:
                        pair_update = self.pair_update_layers[index]
                        if pair_update is not None:
                            pair_representation = pair_update(
                                sequence,
                                pair_representation,
                                mask,
                            )
                del projected_bias
        else:
            for index, layer in enumerate(self.transformer_layers):
                sequence = layer(sequence, pair_representation, condition, mask)
                if self.update_pair_repr and index < self.nlayers - 1:
                    pair_update = self.pair_update_layers[index]
                    if pair_update is not None:
                        pair_representation = pair_update(
                            sequence,
                            pair_representation,
                            mask,
                        )

        output = {
            "token_center": self.token_center_linear(sequence) * mask[..., None],
            "local_latents": self.local_latents_linear(sequence) * mask[..., None],
        }
        return {component: {"v": output[component]} for component in STATE_COMPONENTS}
