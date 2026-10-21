from typing import Dict

import torch

from src.design.model.networks.feature import FeatureFactory
from src.design.model.networks.modules.attn_n_transition import MultiheadAttnAndTransition
from src.design.model.networks.modules.pair_update import PairReprUpdate
from src.design.model.networks.modules.seq_transition_af3 import Transition
from src.design.model.constants import (
    ATOM91_SIZE,
    ELEMENTS_128,
    LIGAND_BOND_TYPES,
    LIGAND_FORMAL_CHARGES,
)
from src.design.model.vae.ligand_graph_decoder import LigandGraphDecoderHead


class DecoderTransformer(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        dec_cfg = kwargs["decoder"]
        self.nlayers = int(dec_cfg["nlayers"])
        self.token_dim = int(dec_cfg["token_dim"])
        self.pair_repr_dim = int(dec_cfg["pair_repr_dim"])
        self.dim_cond = int(dec_cfg["dim_cond"])
        self.update_pair_repr = bool(dec_cfg["update_pair_repr"])
        self.update_pair_repr_every_n = int(dec_cfg["update_pair_repr_every_n"])
        self.use_tri_mult = bool(dec_cfg["use_tri_mult"])
        if self.update_pair_repr or self.use_tri_mult:
            raise ValueError(
                "the fixed standalone VAE config requires "
                "update_pair_repr=false and use_tri_mult=false"
            )
        self.use_qkln = bool(dec_cfg["use_qkln"])
        self.abs_coors = bool(dec_cfg.get("abs_coors", False))
        self.use_ligand_graph_decoder = bool(
            dec_cfg.get("use_ligand_graph_decoder", False)
        )

        self.init_repr_factory = FeatureFactory(
            feats=dec_cfg["feats_seq"],
            dim_feats_out=self.token_dim,
            use_ln_out=bool(dec_cfg.get("use_ln_seq", False)),
            mode="seq",
            **dec_cfg,
        )

        self.cond_factory = FeatureFactory(
            feats=dec_cfg["feats_cond_seq"],
            dim_feats_out=self.dim_cond,
            use_ln_out=bool(dec_cfg.get("use_ln_cond", False)),
            mode="seq",
            **dec_cfg,
        )

        self.pair_rep_factory = FeatureFactory(
            feats=dec_cfg["feats_pair_repr"],
            dim_feats_out=self.pair_repr_dim,
            use_ln_out=bool(dec_cfg.get("use_ln_pair", False)),
            mode="pair",
            **dec_cfg,
        )
        required_batch_keys = (
            {"token_mask", "z_latent", "token_center_coords_nm"}
            | set(self.init_repr_factory.required_keys)
            | set(self.cond_factory.required_keys)
            | set(self.pair_rep_factory.required_keys)
        )
        if self.use_ligand_graph_decoder:
            required_batch_keys.update(("ligand_mask", "mol_id"))
        self.required_batch_keys = frozenset(required_batch_keys)

        self.transition_c_1 = Transition(self.dim_cond, expansion_factor=2)
        self.transition_c_2 = Transition(self.dim_cond, expansion_factor=2)

        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.token_dim,
                    dim_pair=self.pair_repr_dim,
                    nheads=int(dec_cfg["nheads"]),
                    dim_cond=self.dim_cond,
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=self.use_qkln,
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

        self.token_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, 32, bias=False),
        )

        self.polymer_coord_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, ATOM91_SIZE * 3, bias=False),
        )

        self.ligand_element_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, len(ELEMENTS_128), bias=False),
        )

        self.ligand_graph_head = LigandGraphDecoderHead(
            latent_dim=int(kwargs["latent_z_dim"]),
            n_bond_types=len(LIGAND_BOND_TYPES),
            n_charge_classes=len(LIGAND_FORMAL_CHARGES),
            node_dim=int(dec_cfg.get("ligand_graph_node_dim", 256)),
            edge_dim=int(dec_cfg.get("ligand_graph_edge_dim", 128)),
            rbf_dim=int(dec_cfg.get("ligand_graph_rbf_dim", 64)),
            dist_min_nm=float(dec_cfg.get("ligand_graph_dist_min_nm", 0.05)),
            dist_max_nm=float(dec_cfg.get("ligand_graph_dist_max_nm", 3.0)),
            num_layers=int(dec_cfg.get("ligand_graph_num_layers", 3)),
            row_chunk_size=int(dec_cfg.get("ligand_graph_row_chunk_size", 64)),
        )

        if not self.use_ligand_graph_decoder:
            for param in self.ligand_graph_head.parameters():
                param.requires_grad_(False)

    def decode_ligand_graph(
        self,
        z_latent: torch.Tensor,
        input: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        assert self.use_ligand_graph_decoder
        return self.ligand_graph_head(
            z_latent=z_latent,
            token_center_coords_nm=input["token_center_coords_nm"],
            token_mask=input["token_mask"],
            ligand_mask=input["ligand_mask"],
            mol_id=input["mol_id"],
        )

    def forward(
        self, input: Dict[str, torch.Tensor], fixed_receptor: bool = False
    ) -> Dict[str, torch.Tensor]:
        del fixed_receptor
        assert "token_mask" in input, "`token_mask` not in decoder input"
        assert "z_latent" in input, "`z_latent` not in decoder input"
        mask = input["token_mask"].bool()

        cond = self.cond_factory(input)
        cond = self.transition_c_2(self.transition_c_1(cond, mask), mask)

        seqs = self.init_repr_factory(input) * mask[..., None].to(dtype=cond.dtype)
        pair_rep = self.pair_rep_factory(input)

        for i, layer in enumerate(self.transformer_layers):
            seqs = layer(seqs, pair_rep, cond, mask)
            if (
                self.update_pair_repr
                and i < self.nlayers - 1
                and self.pair_update_layers[i] is not None
            ):
                pair_rep = self.pair_update_layers[i](seqs, pair_rep, mask)

        token_logits32 = self.token_head(seqs) * mask[..., None].to(seqs.dtype)
        polymer_offsets = self.polymer_coord_head(seqs).reshape(
            seqs.shape[0], seqs.shape[1], ATOM91_SIZE, 3
        )
        if self.abs_coors:
            polymer_atom91_nm = polymer_offsets
        else:
            polymer_atom91_nm = (
                polymer_offsets + input["token_center_coords_nm"][:, :, None, :]
            )

        ligand_element_logits = self.ligand_element_head(seqs) * mask[..., None].to(
            seqs.dtype
        )
        if self.use_ligand_graph_decoder:
            ligand_graph_out = self.decode_ligand_graph(
                z_latent=input["z_latent"],
                input=input,
            )
            ligand_graph_enabled = seqs.new_tensor(1.0)
        else:
            ligand_graph_out = {
                "ligand_bond_logits": seqs.new_zeros(
                    seqs.shape[0],
                    seqs.shape[1],
                    seqs.shape[1],
                    len(LIGAND_BOND_TYPES),
                ),
                "ligand_charge_logits": seqs.new_zeros(
                    seqs.shape[0],
                    seqs.shape[1],
                    len(LIGAND_FORMAL_CHARGES),
                ),
                "ligand_graph_pair_mask": torch.zeros(
                    seqs.shape[0],
                    seqs.shape[1],
                    seqs.shape[1],
                    dtype=torch.bool,
                    device=seqs.device,
                ),
            }
            ligand_graph_enabled = seqs.new_tensor(0.0)

        return {
            "token_logits32": token_logits32,
            "polymer_atom91_nm": polymer_atom91_nm,
            "ligand_element_logits": ligand_element_logits,
            **ligand_graph_out,
            "ligand_graph_decoder_enabled": ligand_graph_enabled,
        }
