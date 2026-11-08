"""Request-local native AF3 templates for sequence chains with distinct policy."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import numpy as np

_FIELDS = ("aatype", "atom_positions", "atom_mask")


def _chain_policies(all_tokens, templates_by_chain_id):
    chains = np.asarray(all_tokens.chain_id)
    residues = np.asarray(all_tokens.res_name)
    if chains.ndim != 1 or residues.shape != chains.shape:
        raise ValueError("AF3 template policy requires native one-dimensional token chain/residue axes")
    groups = {}
    for cid in dict.fromkeys(chains.tolist()):
        sequence = ",".join(residues[chains == cid].tolist())
        groups.setdefault(sequence, []).append(cid)
    divergent = []
    for members in groups.values():
        policies = []
        for cid in members:
            desired = templates_by_chain_id.get(cid, ())
            for known, owners in policies:
                if list(desired) == list(known):
                    owners.append(cid)
                    break
            else:
                policies.append((desired, [cid]))
        if len(policies) > 1:
            divergent.append((members, policies[1:]))
    return chains, divergent


def _native_arrays(result, token_count):
    arrays = {name: np.asarray(getattr(result, name)) for name in _FIELDS}
    aa, xyz, mask = (arrays[name] for name in _FIELDS)
    if (aa.ndim != 2 or aa.shape[1] != token_count or xyz.ndim != 4
            or xyz.shape[:2] != aa.shape or xyz.shape[-1] != 3
            or mask.shape != xyz.shape[:-1]):
        raise ValueError("AF3 template policy native output axes differ from Templates schema")
    return arrays


def _compute(cls, original, **kwargs):
    result = original(cls, **kwargs)
    chain_ids, divergent = _chain_policies(kwargs["all_tokens"], kwargs["templates_by_chain_id"])
    if not divergent:
        return result
    indices = np.asarray(kwargs["standard_token_idxs"])
    count = int(kwargs["padding_shapes"].num_tokens)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= len(chain_ids))
            or len(indices) > count):
        raise ValueError("AF3 template policy native standard-token axis is invalid")
    # Native Templates first selects standard_token_idxs, then pads this axis
    # to padding_shapes.num_tokens. Padding remains the original native result.
    labels = chain_ids[indices]
    baseline = _native_arrays(result, count)
    corrected = {name: value.copy() for name, value in baseline.items()}
    for members, policies in divergent:
        for desired, owners in policies:
            isolated = {cid: [] for cid in kwargs["templates_by_chain_id"]}
            isolated.update({cid: desired for cid in members})
            native = original(cls, **{**kwargs, "templates_by_chain_id": isolated})
            arrays = _native_arrays(native, count)
            columns = np.flatnonzero(np.isin(labels, owners))
            for name in _FIELDS:
                if arrays[name].shape != baseline[name].shape or arrays[name].dtype != baseline[name].dtype:
                    raise ValueError("AF3 template policy recomputation changed native feature shape/dtype")
                corrected[name][:, columns, ...] = arrays[name][:, columns, ...]
    return replace(result, **corrected)


@contextmanager
def sequence_template_policy(features, *, enabled):
    """Restore the exact native classmethod descriptor after one serial request."""
    if not enabled:
        yield
        return
    cls = features.Templates
    descriptor = cls.__dict__.get("compute_features")
    if not isinstance(descriptor, classmethod):
        raise ValueError("AF3 native Templates.compute_features is not a classmethod")

    def compute(current_cls, all_tokens, standard_token_idxs, padding_shapes,
                templates_by_chain_id, max_templates, logging_name):
        return _compute(
            current_cls, descriptor.__func__, all_tokens=all_tokens,
            standard_token_idxs=standard_token_idxs, padding_shapes=padding_shapes,
            templates_by_chain_id=templates_by_chain_id, max_templates=max_templates,
            logging_name=logging_name)

    cls.compute_features = classmethod(compute)
    try:
        yield
    finally:
        cls.compute_features = descriptor


__all__ = ["sequence_template_policy"]
