"""Checkpoint-only FP32 PartiScore structure inference."""
import inspect
import math
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
import torch
from .core import (PartiScoreNetwork, _canonical_heavy_atom_geometry,
                   _minimum_heavy_atom_distances, supervision_pairs)
from .batching import _batch_canonical_poses, _plan_score_batches, _inference_score_limits
from .features import SCORE_FIELDS
from .preprocessing import StructurePreprocessor
from .records import InterfaceScore, ScoreResult


def _checkpoint(path, weights):
    if weights not in {"raw", "ema"}:
        raise ValueError("weights must be raw or ema")
    path = str(Path(path).expanduser().resolve())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    config = payload.get("hyper_parameters", {}).get("config")
    if not isinstance(config, Mapping) or config.get("score_contract_version") != 2:
        raise ValueError("checkpoint requires score_contract_version=2")
    model = config.get("model")
    if not isinstance(model, Mapping) or model.get("input_mode") != "structure":
        raise ValueError("standalone scoring requires a v2 structure checkpoint")
    allowed = set(inspect.signature(PartiScoreNetwork).parameters)
    unknown = set(model) - allowed
    if unknown:
        raise ValueError(f"unsupported checkpoint model fields: {sorted(unknown)}")
    with torch.random.fork_rng(devices=[]):
        network = PartiScoreNetwork(**dict(model)).float()
    if weights == "raw":
        raw = payload.get("state_dict")
        if not isinstance(raw, Mapping):
            raise ValueError("checkpoint has no state_dict")
        state = {k[6:]: v for k, v in raw.items() if k.startswith("score.")}
    else:
        ema = payload.get("partiscore_ema")
        if not isinstance(ema, Mapping) or ema.get("version") != 1:
            raise ValueError("checkpoint has no valid v1 EMA payload")
        decay = ema.get("decay")
        expected_decay = config.get("ema", {}).get("decay", 0.999)
        if (isinstance(decay, bool) or not isinstance(decay, (int, float))
                or not math.isfinite(decay) or not 0 <= decay < 1 or decay != expected_decay):
            raise ValueError("invalid EMA decay")
        for key in ("updates", "started_at_step"):
            if isinstance(ema.get(key), bool) or not isinstance(ema.get(key), int) or ema[key] < 0:
                raise ValueError(f"invalid EMA {key}")
        state = ema.get("state")
        if not isinstance(state, Mapping) or not state:
            raise ValueError("EMA state is empty or invalid")
    network.load_state_dict(state, strict=True)
    for key, value in network.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite checkpoint tensor: {key}")
    network.requires_grad_(False)
    return network.eval(), {"checkpoint": path, "weights": weights, "model": dict(model)}


def validate_checkpoint(path, weights="raw"):
    """Validate metadata, exact scorer keys/shapes and selected weights on CPU."""
    _, metadata = _checkpoint(path, weights)
    return metadata



def _resolve_device(device):
    if str(device).startswith("npu") and not hasattr(torch, "npu"):
        import importlib
        importlib.import_module("torch_npu")
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device

def _metrics(logp, *, has_interface=True):
    count = int(logp.numel())
    if not count:
        return dict(score=None, mean_log_density=None, density_sum=0.0, pair_count=0,
                    status="no_contact" if has_interface else "no_interface")
    if not torch.isfinite(logp).all():
        raise FloatingPointError("nonfinite PartiScore log density")
    mean = float(logp.mean().item())
    density = float(logp.exp().sum().item())
    if not math.isfinite(density):
        raise FloatingPointError("nonfinite PartiScore density sum")
    return dict(score=mean + math.log(count), mean_log_density=mean,
                density_sum=density, pair_count=count, status="ok")


def _design_group_pairs(batch, cutoff_angstrom=5.0):
    """Return eligible design-group pairs and the pre-cutoff scope mask.

    Preparation assigns one source ID to each design group or context source,
    so this path can retain the network's inter-source candidate contract while
    using group metadata for the additional eligibility boundary.
    """

    if "_partiscore_design_group_id" not in batch:
        raise ValueError("design_groups scoring requires design-group metadata")
    valid = batch["token_mask"].bool()
    groups = batch["_partiscore_design_group_id"].long()
    if groups.shape != valid.shape:
        raise ValueError("design-group metadata shape does not match token_mask")
    pair_scope = valid[:, :, None] & valid[:, None, :]
    pair_scope &= torch.ones_like(pair_scope).triu(1)
    left_design = groups[:, :, None] >= 0
    right_design = groups[:, None, :] >= 0
    pair_scope &= (
        (left_design & right_design & (groups[:, :, None] != groups[:, None, :]))
        | (left_design ^ right_design)
    )
    source = batch["source_mol_id"].long()
    pair_scope &= source[:, :, None] != source[:, None, :]
    # Direct covalent design-context token pairs are represented as real graph
    # edges, but are excluded from inference likelihood aggregation.
    bond_mask = batch.get("_partiscore_covalent_bond_pair_mask")
    if bond_mask is None:
        bond_mask = batch.get("token_bond_pair_mask")
    if bond_mask is None:
        raise ValueError("design_groups scoring requires token bond metadata")
    pair_scope &= ~bond_mask.bool()
    coords, atom_mask = _canonical_heavy_atom_geometry(batch)
    pair_scope &= atom_mask.any(dim=-1)[:, :, None] & atom_mask.any(dim=-1)[:, None, :]
    candidate_indices = pair_scope.nonzero(as_tuple=True)
    targets = _minimum_heavy_atom_distances(
        coords,
        atom_mask,
        candidate_indices,
        chunk_size=1024,
    )
    keep = targets <= float(cutoff_angstrom)
    return tuple(index[keep] for index in candidate_indices), targets[keep], pair_scope


def _design_group_interfaces(prepared, batch, batch_index, indices, values, pair_scope):
    """Aggregate eligible contact pairs at existing source-partner granularity."""

    token_mask = batch["token_mask"][batch_index].bool()
    groups = batch["_partiscore_design_group_id"][batch_index].long()
    source = batch["source_mol_id"][batch_index].long()
    n_token = int(token_mask.sum().item())
    # One CPU copy of the token metadata avoids per-token device synchronizing
    # while deriving the eligible source-partner combinations.
    metadata = torch.stack((source[:n_token], groups[:n_token]), dim=1).detach().cpu().tolist()
    source_groups = {}
    for source_id, group_id in metadata:
        source_groups.setdefault(int(source_id), set()).add(int(group_id))
    source_ids = sorted(source_groups)
    potential = {
        (left, right)
        for index, left in enumerate(source_ids)
        for right in source_ids[index + 1:]
        if ((-1 in source_groups[left]) != (-1 in source_groups[right]))
        or (-1 not in source_groups[left]
            and -1 not in source_groups[right]
            and source_groups[left] != source_groups[right])
    }

    if not potential:
        return [], []

    local = indices[0] == batch_index
    pair_i = indices[1][local]
    pair_j = indices[2][local]
    pair_values = values[local]
    pair_left_source = source[pair_i]
    pair_right_source = source[pair_j]
    interfaces, masks = [], []
    for key in sorted(potential):
        mask = ((pair_left_source == key[0]) & (pair_right_source == key[1])) | (
            (pair_left_source == key[1]) & (pair_right_source == key[0])
        )
        interfaces.append(InterfaceScore(prepared.partner_labels[key[0]],
                                         prepared.partner_labels[key[1]],
                                         **_metrics(pair_values[mask])))
        masks.append(mask)
    return interfaces, masks


class PartiScorer:
    validate_checkpoint = staticmethod(validate_checkpoint)

    def __init__(self, network, *, checkpoint="", weights="raw", device="cpu",
                 batch_size=32, max_pair_elements=2097152, num_workers=16):
        self.batch_size, self.max_pair_elements = _inference_score_limits(
            self, inference_batch_size=batch_size, max_pair_elements=max_pair_elements)
        self._preprocessor = StructurePreprocessor(num_workers=num_workers)
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.network = network.to(self.device).float().eval()
        self.checkpoint, self.weights = str(checkpoint), weights

    @classmethod
    def from_checkpoint(cls, path, weights="raw", device="auto", batch_size=32,
                        max_pair_elements=2097152, num_workers=16):
        network, metadata = _checkpoint(path, weights)
        device = _resolve_device(device)
        return cls(network, checkpoint=metadata["checkpoint"], weights=weights, device=device,
                   batch_size=batch_size, max_pair_elements=max_pair_elements, num_workers=num_workers)

    def _score(self, inputs, identities, design_chains, scope, reader):
        inputs = list(inputs)
        if scope not in {"all", "design_touching", "design_groups"}:
            raise ValueError("scope must be all, design_touching, or design_groups")
        if identities is None:
            identities = [str(x) if reader else str(i) for i, x in enumerate(inputs)]
        else:
            identities = list(identities)
        chains = [None] * len(inputs) if design_chains is None else list(design_chains)
        if len(identities) != len(inputs) or len(chains) != len(inputs):
            raise ValueError("input, identity and design-chain counts must match")
        if scope in {"design_touching", "design_groups"} and design_chains is None:
            raise ValueError(f"{scope} requires per-pose design_chains")
        results = [None] * len(inputs)
        records = []
        tasks = [(index, str(value) if reader else value, str(identity), selected, bool(reader), scope)
                 for index, (value, identity, selected) in enumerate(zip(inputs, identities, chains))]
        preparation_count = 0
        for expected_index, (index, prepared, error) in enumerate(self._preprocessor.prepare(tasks)):
            preparation_count += 1
            if index != expected_index:
                raise RuntimeError("preprocessor returned out-of-order input identity")
            identity = str(identities[index])
            if error is not None:
                results[index] = ScoreResult(identity=identity, error=error,
                                             checkpoint=self.checkpoint, weights=self.weights, scope=scope)
            else:
                records.append(dict(index=index, ordinal=index, target_id=None,
                                    batch=prepared.canonical, prepared=prepared, identity=identity))
        if preparation_count != len(inputs):
            raise RuntimeError("preprocessor returned an unexpected number of results")
        planned = _plan_score_batches(self, records, inference_batch_size=self.batch_size,
                                      max_pair_elements=self.max_pair_elements)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            for group in planned:
                batch = _batch_canonical_poses([r["batch"] for r in group], [r["identity"] for r in group],
                                               fields=SCORE_FIELDS, device=self.device)
                node, pair = self.network(batch)
                if not torch.isfinite(node).all() or not torch.isfinite(pair).all():
                    raise FloatingPointError("nonfinite PartiScore network output: " + str([r["identity"] for r in group]))
                if scope == "design_groups":
                    indices, target, pair_scope = _design_group_pairs(batch, cutoff_angstrom=5.0)
                else:
                    indices, target = supervision_pairs(batch, cutoff_angstrom=5.0)
                logp, _ = self.network.likelihood(node, pair, indices, target)
                for b, record in enumerate(group):
                    prepared = record["prepared"]
                    source = batch["source_mol_id"][b]
                    local = indices[0] == b
                    left, right = source[indices[1][local]], source[indices[2][local]]
                    values = logp[local]
                    if scope == "design_groups":
                        interfaces, masks = _design_group_interfaces(
                            prepared, batch, b, indices, logp, pair_scope
                        )
                    else:
                        interfaces, masks = [], []
                        for a, c in combinations(sorted(prepared.partner_labels), 2):
                            mask = ((left == a) & (right == c)) | ((left == c) & (right == a))
                            interfaces.append(InterfaceScore(prepared.partner_labels[a], prepared.partner_labels[c],
                                                             **_metrics(values[mask])))
                            if scope == "all" or a in prepared.design_source_ids or c in prepared.design_source_ids:
                                masks.append(mask)
                    selected = torch.stack(masks).any(0) if masks else torch.zeros_like(values, dtype=torch.bool)
                    results[record["index"]] = ScoreResult(
                        identity=record["identity"], **_metrics(values[selected], has_interface=bool(masks)),
                        interfaces=interfaces, checkpoint=self.checkpoint, weights=self.weights, scope=scope)
        return results

    def close(self):
        """Release reusable CPU preparation workers; the scorer cannot be reused."""
        self._preprocessor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def score_files(self, paths, *, identities=None, design_chains=None, scope="all"):
        return self._score(paths, identities, design_chains, scope, True)

    def score_structures(self, structures, *, identities=None, design_chains=None, scope="all"):
        return self._score(structures, identities, design_chains, scope, None)
