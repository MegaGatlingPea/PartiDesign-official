"""One-process, one-model-load CyclicMPNN batch worker."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping


def _fixed_position_argument(
    context_chains: list[str], fixed_positions: list[str]
) -> str | None:
    """Return ColabDesign's optional fixed-position selector.

    ``prep_inputs()`` treats an empty string as a malformed residue selector;
    an unconstrained cyclic monomer must pass ``None`` instead.
    """

    value = ",".join([*context_chains, *fixed_positions])
    return value or None


def _sample_requests(model, requests, config, apply_bias, seed_everything=None):
    """Prepare each current structure once and batch compatible native inputs."""
    import numpy as np
    from .aa_omission import resolve_rm_aa

    explicit_rm_aa = "rm_aa" in config and config["rm_aa"] is not None
    effective_rm_aa = resolve_rm_aa(config)
    batched = callable(getattr(model, "sample_inputs", None))
    if batched:
        import jax
        from colabdesign.mpnn.model import order_aa
    if explicit_rm_aa and effective_rm_aa:
        from colabdesign.mpnn.model import aa_order
    native_rm_aa = "" if explicit_rm_aa else effective_rm_aa
    results = [None] * len(requests)
    groups = {}
    for index, request in enumerate(requests):
        candidate_id = str(request["candidate_id"])
        try:
            context = [str(value) for value in request["context_chains"]]
            design = str(request["design_chain"])
            fixed = [str(value) for value in request["fixed_positions"]]
            if not batched and "seed" in request:
                if seed_everything is not None:
                    seed_everything(request["seed"])
                set_seed = getattr(model, "set_seed", None)
                if callable(set_seed):
                    set_seed(request["seed"])
            model.prep_inputs(
                pdb_filename=str(request["pdb_path"]),
                chain=",".join([*context, design]),
                fix_pos=_fixed_position_argument(context, fixed),
                rm_aa=native_rm_aa or None,
            )
            apply_bias(model)
            if explicit_rm_aa and effective_rm_aa:
                raw_positions = request.get("omit_positions", [])
                if not isinstance(raw_positions, list):
                    raise ValueError("omit_positions must be a list")
                lengths = tuple(int(value) for value in model._lengths)
                design_offset = sum(lengths[:-1])
                design_length = lengths[-1]
                for raw_position in raw_positions:
                    if isinstance(raw_position, bool) or not isinstance(raw_position, int):
                        raise ValueError("omit_positions must contain positive integers")
                    position = int(raw_position)
                    if not 1 <= position <= design_length:
                        raise ValueError("omit_positions contains an out-of-range position")
                    row = design_offset + position - 1
                    for amino_acid in effective_rm_aa:
                        # Apply the hard mask after any user bias.  A large
                        # finite bias could otherwise undo the omission.
                        model._inputs["bias"][row, aa_order[amino_acid]] = -np.inf
            temperature = float(request.get("temperature", config["temperature"]))
            if batched:
                # prep_inputs creates fresh arrays for each structure. Keep them
                # only for this call; values are never reused across requests.
                inputs = dict(model._inputs)
                shape = tuple((name, np.shape(value), str(np.asarray(value).dtype))
                              for name, value in sorted(inputs.items()))
                groups.setdefault((shape, int(request.get("seed", 0))), []).append((
                    index, candidate_id, inputs,
                    int(request.get("seed", 0)),
                    temperature, int(model._lengths[-1]),
                ))
            else:
                sampled = model.sample(temperature=temperature, num=1, batch=1)
                sequences = sampled.get("seq")
                if sequences is None or len(sequences) != 1:
                    raise ValueError("CyclicMPNN did not return one sequence")
                results[index] = {"candidate_id": candidate_id, "status": "ok",
                                  "sequence": str(sequences[0]).split("/")[-1]}
        except (KeyError, TypeError, ValueError) as exc:
            results[index] = {"candidate_id": candidate_id, "status": "failed",
                              "error": f"{type(exc).__name__}: {exc}"}
    def sample_group(group, keys):
        inputs = jax.tree_util.tree_map(
            lambda *values: np.stack(values), *(row[2] for row in group))
        temperatures = np.asarray([row[4] for row in group], dtype=np.float32)
        sampled = model.sample_inputs(inputs, keys, temperatures).argmax(-1)
        if len(sampled) != len(group):
            raise ValueError("CyclicMPNN returned the wrong number of sequences")
        for row, tokens in zip(group, sampled):
            index, candidate_id, _inputs, _key, _temperature, design_length = row
            sequence = "".join(order_aa[int(token)] for token in tokens[-design_length:])
            results[index] = {"candidate_id": candidate_id, "status": "ok",
                              "sequence": sequence}

    for group in groups.values():
        # Keep each candidate's original draw when a failed batch is retried.
        keys = jax.random.split(jax.random.PRNGKey(group[0][3]), len(group))
        try:
            sample_group(group, keys)
        except (KeyError, TypeError, ValueError) as exc:
            if len(group) == 1:
                row = group[0]
                results[row[0]] = {
                    "candidate_id": row[1], "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            import warnings
            warnings.warn(
                f"CyclicMPNN batch sampling failed ({type(exc).__name__}: {exc}); "
                "retrying prepared candidates individually to isolate the error.",
                RuntimeWarning,
            )
            for row_index, row in enumerate(group):
                try:
                    sample_group([row], keys[row_index:row_index + 1])
                except (KeyError, TypeError, ValueError) as candidate_error:
                    results[row[0]] = {
                        "candidate_id": row[1], "status": "failed",
                        "error": f"{type(candidate_error).__name__}: {candidate_error}",
                    }
    return results


def _run(payload: Mapping[str, Any]) -> dict[str, Any]:
    from ml_collections import ConfigDict
    from pxdbench.tools.protmpnn.main_mpnn import (
        _apply_bias_inputs,
        _make_mpnn_model,
    )

    config = payload["config"]
    requests = payload["requests"]
    if not isinstance(config, Mapping) or not isinstance(requests, list):
        raise ValueError("invalid CyclicMPNN batch protocol")
    model_config = ConfigDict(
        {
            "weights": "original",
            "model_name": str(config["model_name"]),
            "base_model_name": str(config["base_model_name"]),
            "model_weights_path": str(config["checkpoint"]),
            "backbone_noise": float(config["backbone_noise"]),
        }
    )
    started = time.perf_counter()
    model = _make_mpnn_model(model_config)
    load_seconds = time.perf_counter() - started
    results = _sample_requests(
        model, requests, config,
        lambda current: _apply_bias_inputs(
            current, str(config.get("bias_aa_json") or ""), ""),
    )
    return {
        "schema_version": 1,
        "model_loads": 1,
        "model_load_seconds": load_seconds,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.request).read_text())
    result = _run(payload)
    Path(args.result).write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
