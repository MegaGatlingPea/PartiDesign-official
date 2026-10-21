from typing import Final

import torch
from torch.utils.checkpoint import checkpoint


_MAX_GRAPH_HEAD_PARAMETERS: Final[int] = 5_000_000


class InvariantMessagePassingBlock(torch.nn.Module):

    def __init__(self, node_dim: int, edge_dim: int, rbf_dim: int):
        super().__init__()
        pair_dim = 3 * node_dim + rbf_dim
        self.message_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(pair_dim),
            torch.nn.Linear(pair_dim, edge_dim, bias=False),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(edge_dim),
            torch.nn.Linear(edge_dim, node_dim, bias=False),
        )
        self.node_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * node_dim),
            torch.nn.Linear(2 * node_dim, node_dim, bias=False),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(node_dim),
            torch.nn.Linear(node_dim, node_dim, bias=False),
        )
        self.output_norm = torch.nn.LayerNorm(node_dim)

    def _row_messages(
        self,
        row_h: torch.Tensor,
        all_h: torch.Tensor,
        row_rbf: torch.Tensor,
        row_pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        h_i = row_h[:, :, None, :]
        h_j = all_h[:, None, :, :]
        pair_feat = torch.cat(
            (h_i + h_j, (h_i - h_j).abs(), h_i * h_j, row_rbf),
            dim=-1,
        )
        messages = self.message_mlp(pair_feat)
        messages = messages * row_pair_mask[..., None].to(messages.dtype)
        return messages.sum(dim=2)

    def forward(
        self,
        h: torch.Tensor,
        rbf: torch.Tensor,
        pair_mask: torch.Tensor,
        node_mask: torch.Tensor,
        row_chunk_size: int,
    ) -> torch.Tensor:
        aggregates = []
        for row_start in range(0, h.shape[1], row_chunk_size):
            row_stop = min(row_start + row_chunk_size, h.shape[1])
            inputs = (
                h[:, row_start:row_stop],
                h,
                rbf[:, row_start:row_stop],
                pair_mask[:, row_start:row_stop],
            )
            if self.training and torch.is_grad_enabled():
                row_aggregate = checkpoint(
                    self._row_messages,
                    *inputs,
                    use_reentrant=False,
                )
            else:
                row_aggregate = self._row_messages(*inputs)
            aggregates.append(row_aggregate)
        aggregate = torch.cat(aggregates, dim=1)
        degree = pair_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        aggregate = aggregate / degree.to(aggregate.dtype)
        update = self.node_mlp(torch.cat((h, aggregate), dim=-1))
        h = self.output_norm(h + update)
        return h * node_mask[..., None].to(h.dtype)


class LigandGraphDecoderHead(torch.nn.Module):
    """Decode ligand bonds and formal charges from latent nodes and distances.

    The head intentionally cannot access decoder Transformer states, element
    predictions, or any ground-truth chemical labels.  Pair geometry enters only
    through distances, making the module invariant to global rotation and
    translation.
    """

    def __init__(
        self,
        latent_dim: int,
        n_bond_types: int,
        n_charge_classes: int,
        node_dim: int = 256,
        edge_dim: int = 128,
        rbf_dim: int = 64,
        dist_min_nm: float = 0.05,
        dist_max_nm: float = 3.0,
        num_layers: int = 3,
        row_chunk_size: int = 64,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.rbf_dim = int(rbf_dim)
        self.dist_min_nm = float(dist_min_nm)
        self.dist_max_nm = float(dist_max_nm)
        self.num_layers = int(num_layers)
        self.row_chunk_size = int(row_chunk_size)
        self.n_bond_types = int(n_bond_types)
        self.n_charge_classes = int(n_charge_classes)

        assert self.latent_dim > 0
        assert self.node_dim > 0 and self.edge_dim > 0
        assert self.rbf_dim > 1
        assert self.dist_max_nm > self.dist_min_nm >= 0.0
        assert self.num_layers == 3, "The production graph head uses exactly 3 layers"
        assert self.row_chunk_size > 0
        assert self.n_bond_types > 1 and self.n_charge_classes > 1

        centers = torch.linspace(
            self.dist_min_nm,
            self.dist_max_nm,
            self.rbf_dim,
            dtype=torch.float32,
        )
        spacing = (self.dist_max_nm - self.dist_min_nm) / (self.rbf_dim - 1)
        self.register_buffer("rbf_centers_nm", centers, persistent=True)
        self.register_buffer(
            "rbf_inverse_two_sigma_sq",
            torch.tensor(0.5 / (spacing * spacing), dtype=torch.float32),
            persistent=True,
        )

        self.node_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(self.latent_dim),
            torch.nn.Linear(self.latent_dim, self.node_dim, bias=False),
            torch.nn.SiLU(),
        )
        self.message_passing = torch.nn.ModuleList(
            [
                InvariantMessagePassingBlock(
                    node_dim=self.node_dim,
                    edge_dim=self.edge_dim,
                    rbf_dim=self.rbf_dim,
                )
                for _ in range(self.num_layers)
            ]
        )
        edge_input_dim = 3 * self.node_dim + self.rbf_dim
        self.bond_classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_input_dim),
            torch.nn.Linear(edge_input_dim, self.edge_dim, bias=False),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(self.edge_dim),
            torch.nn.Linear(self.edge_dim, self.n_bond_types, bias=True),
        )
        self.charge_classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.node_dim),
            torch.nn.Linear(self.node_dim, self.edge_dim, bias=False),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(self.edge_dim),
            torch.nn.Linear(self.edge_dim, self.n_charge_classes, bias=True),
        )

        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        assert parameter_count <= _MAX_GRAPH_HEAD_PARAMETERS, parameter_count

    @staticmethod
    def candidate_mask(
        token_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
        mol_id: torch.Tensor,
    ) -> torch.Tensor:
        assert token_mask.ndim == ligand_mask.ndim == mol_id.ndim == 2
        assert token_mask.shape == ligand_mask.shape == mol_id.shape
        node_mask = token_mask.bool() & ligand_mask.bool()
        same_molecule = mol_id[:, :, None] == mol_id[:, None, :]
        diagonal = torch.eye(
            token_mask.shape[1],
            dtype=torch.bool,
            device=token_mask.device,
        )[None]
        return node_mask[:, :, None] & node_mask[:, None, :] & same_molecule & ~diagonal

    def _distance_rbf(self, coords_nm: torch.Tensor) -> torch.Tensor:
        distance_nm = torch.cdist(coords_nm.float(), coords_nm.float())
        delta = distance_nm[..., None] - self.rbf_centers_nm
        rbf = torch.exp(-delta.square() * self.rbf_inverse_two_sigma_sq)
        return rbf.to(coords_nm.dtype)

    def _bond_logits_rows(
        self,
        row_h: torch.Tensor,
        all_h: torch.Tensor,
        row_rbf: torch.Tensor,
    ) -> torch.Tensor:
        h_i = row_h[:, :, None, :]
        h_j = all_h[:, None, :, :]
        pair_feat = torch.cat(
            (h_i + h_j, (h_i - h_j).abs(), h_i * h_j, row_rbf),
            dim=-1,
        )
        return self.bond_classifier(pair_feat)

    def forward(
        self,
        z_latent: torch.Tensor,
        token_center_coords_nm: torch.Tensor,
        token_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
        mol_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert z_latent.ndim == 3 and z_latent.shape[-1] == self.latent_dim
        assert token_center_coords_nm.shape == z_latent.shape[:2] + (3,)
        assert (
            token_mask.shape == ligand_mask.shape == mol_id.shape == z_latent.shape[:2]
        )
        assert bool(torch.isfinite(z_latent).all().item())
        assert bool(torch.isfinite(token_center_coords_nm).all().item())

        node_mask = token_mask.bool() & ligand_mask.bool()
        pair_mask = self.candidate_mask(token_mask, ligand_mask, mol_id)
        h = self.node_projection(z_latent) * node_mask[..., None].to(z_latent.dtype)
        rbf = self._distance_rbf(token_center_coords_nm).to(h.dtype)
        rbf = rbf * pair_mask[..., None].to(rbf.dtype)
        for layer in self.message_passing:
            h = layer(
                h=h,
                rbf=rbf,
                pair_mask=pair_mask,
                node_mask=node_mask,
                row_chunk_size=self.row_chunk_size,
            )

        charge_logits = self.charge_classifier(h)
        charge_logits = charge_logits * node_mask[..., None].to(charge_logits.dtype)

        bond_rows = []
        for row_start in range(0, h.shape[1], self.row_chunk_size):
            row_stop = min(row_start + self.row_chunk_size, h.shape[1])
            inputs = (
                h[:, row_start:row_stop],
                h,
                rbf[:, row_start:row_stop],
            )
            if self.training and torch.is_grad_enabled():
                row_logits = checkpoint(
                    self._bond_logits_rows,
                    *inputs,
                    use_reentrant=False,
                )
            else:
                row_logits = self._bond_logits_rows(*inputs)
            bond_rows.append(row_logits)
        bond_logits = torch.cat(bond_rows, dim=1)
        bond_logits = 0.5 * (bond_logits + bond_logits.transpose(1, 2))
        bond_logits = bond_logits * pair_mask[..., None].to(bond_logits.dtype)

        return {
            "ligand_bond_logits": bond_logits,
            "ligand_charge_logits": charge_logits,
            "ligand_graph_pair_mask": pair_mask,
        }
