"""Isolated CPU model loading used by the explicit check --load stage."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from .paths import community_source, community_asset, using_model_dir, code_root


def _source(name: str, *suffix: str) -> Path:
    path = community_source(name).joinpath(*suffix)
    if not path.is_dir():
        raise FileNotFoundError(f"bundled source missing: {path}")
    sys.path.insert(0, str(path))
    return path


def _finite_state(model) -> int:
    import torch
    state = model.state_dict()
    for name, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"nonfinite parameter: {name}")
    return len(state)


def _protein_load(root: Path, *, soluble: bool, checkpoint=None, ca_only=False) -> str:
    import torch
    _source("proteinmpnn")
    from protein_mpnn_utils import ProteinMPNN
    family = "soluble" if soluble else "vanilla"
    paths = sorted((community_asset("proteinmpnn") / "checkpoint" / family).glob("*.pt"))
    if not soluble:
        paths += sorted((community_asset("proteinmpnn") / "checkpoint" / "ca").glob("*.pt"))
    if checkpoint is not None:
        paths = [Path(checkpoint)]
    if not paths:
        raise FileNotFoundError("no ProteinMPNN weights")
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = ProteinMPNN(ca_only=ca_only if checkpoint else path.parent.name == "ca", num_letters=21,
            node_features=128, edge_features=128, hidden_dim=128,
            num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.0,
            k_neighbors=payload["num_edges"])
        model.load_state_dict(payload["model_state_dict"], strict=True)
        _finite_state(model)
    return f"strict CPU keys/shapes/finite tensors: {len(paths)} {family} checkpoints"


def _ligand_load(*, checkpoint=None, model_type="ligand_mpnn", packing_only=False, sequence_only=False) -> str:
    import torch
    _source("ligandmpnn")
    from model_utils import ProteinMPNN
    from sc_utils import Packer
    root = community_asset("ligandmpnn") / "checkpoint"
    paths = sorted(root.glob("ligandmpnn_v_*.pt"))
    if checkpoint is not None:
        paths = [Path(checkpoint)]
    if packing_only:
        paths = []
    elif not paths:
        raise FileNotFoundError("no LigandMPNN sequence weights")
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128,
            num_encoder_layers=3, num_decoder_layers=3, k_neighbors=payload["num_edges"],
            device=torch.device("cpu"), atom_context_num=payload["atom_context_num"] if model_type == "ligand_mpnn" else 1,
            model_type=model_type, ligand_mpnn_use_side_chain_context=0)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        _finite_state(model)
    if sequence_only:
        return f"strict CPU keys/shapes/finite tensors: {len(paths)} selected sequence weights (packing unused)"
    packer = Packer(node_features=128, edge_features=128, num_positional_embeddings=16,
        num_chain_embeddings=16, num_rbf=16, hidden_dim=128, num_encoder_layers=3,
        num_decoder_layers=3, atom_context_num=16, lower_bound=0.0, upper_bound=20.0,
        top_k=32, dropout=0.0, augment_eps=0.0, atom37_order=False,
        device=torch.device("cpu"), num_mix=3)
    payload = torch.load(root / "ligandmpnn_sc_v_32_002_16.pt", map_location="cpu", weights_only=False)
    packer.load_state_dict(payload["model_state_dict"], strict=True)
    _finite_state(packer)
    return f"strict CPU keys/shapes/finite tensors: {len(paths)} sequence weights and side-chain packer"


def _cyclic_load(checkpoint=None) -> str:
    import numpy as np
    import jax
    import jax.numpy as jnp
    import haiku as hk
    import joblib
    _source("colabdesign")
    from colabdesign.mpnn.modules import ProteinMPNN
    path = Path(checkpoint) if checkpoint else community_asset("cyclicmpnn") / "checkpoint" / "cyclicmpnn_48_010.pkl"
    payload = joblib.load(path)
    config = dict(num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
                  num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.0,
                  k_neighbors=payload.get("num_edges", 48), dropout=0.0)
    length = 4
    inputs = dict(X=jnp.ones((length, 4, 3)), mask=jnp.ones(length),
        residue_idx=jnp.arange(length), chain_idx=jnp.zeros(length, dtype=jnp.int32),
        S=jax.nn.one_hot(jnp.zeros(length, dtype=jnp.int32), 21),
        decoding_order=jnp.arange(length)[:, None])
    function = hk.transform(lambda value: ProteinMPNN(**config).score(value))
    expected = jax.eval_shape(function.init, jax.random.PRNGKey(0), inputs)
    state = payload["model_state_dict"]
    expected_keys = {(scope, name) for scope, values in expected.items() for name in values}
    keys = {(scope, name) for scope, values in state.items() for name in values}
    if keys != expected_keys:
        raise ValueError(f"CyclicMPNN parameter keys mismatch: missing={expected_keys - keys}, extra={keys - expected_keys}")
    for scope, name in keys:
        value = np.asarray(state[scope][name])
        if value.shape != expected[scope][name].shape or not np.isfinite(value).all():
            raise ValueError(f"CyclicMPNN parameter shape/values mismatch: {scope}/{name}")
    return f"strict CPU Haiku architecture keys/shapes and finite tensors: {len(keys)}"


def _af3_load(root: Path, target: str) -> str:
    """Validate one AF3 archive, SDK import, and all CPU parameter records."""
    from . import af3_assets

    layout = af3_assets.validate_installation(root, target)
    os.environ["PARTIDESIGN_AF3_CCD_DIR"] = str(layout.ccd_dir)
    source = _source("af3", "src")
    try:
        module = importlib.import_module("alphafold3")
        params = importlib.import_module("alphafold3.model.params")
    except Exception as exc:
        raise RuntimeError(f"AF3 SDK import failed: {exc}") from exc
    origin = getattr(module, "__file__", None)
    if origin is not None:
        resolved = Path(origin).resolve()
        if source not in resolved.parents and resolved != source:
            raise RuntimeError(
                f"ambient AlphaFold3 import rejected: {resolved}; expected under {source}"
            )
    import numpy as np
    weights = params.get_model_haiku_params(layout.weight.parent)
    if not weights:
        raise ValueError(f"AF3 model archive has no parameter records: {layout.weight}")
    records = 0
    for scope, values in weights.items():
        for name, value in values.items():
            array = np.asarray(value)
            if array.dtype.kind in "fc" and not np.isfinite(array).all():
                raise ValueError(f"nonfinite AF3 parameter record: {scope}/{name}")
            records += 1
    del weights
    return (
        f"strict CPU AF3 archive/SDK/CCD validation: {target}; "
        f"loaded {records} finite parameter records"
    )


def load_target(args) -> str:
    target = args.target
    if target == "partidesign":
        from src.design.model.checkpoint import load_combined_checkpoint
        path = args.checkpoint or str(Path(args.model_dir) / "ckpts" / "partidesign.ckpt")
        if args.antibody:
            from src.design.antibody.checkpoint import validate_antibody_checkpoint
            validate_antibody_checkpoint(path, variant=args.variant, require_bundle=args.require_bundle)
        from src.design.model.bundle import BUNDLE_MEMBERS, is_partidesign_bundle, _validate_bundle_schema
        if args.checkpoint is not None or args.variant is not None:
            variants = [args.variant]
        else:
            from src.design.model.checkpoint import _read_checkpoint
            document = _read_checkpoint(path, "PartiDesign checkpoint")
            if is_partidesign_bundle(document):
                validated = _validate_bundle_schema(document)
                variants = [member for member in BUNDLE_MEMBERS if member in validated]
            else:
                variants = [None]
        for variant in variants:
            model = load_combined_checkpoint(path, variant=variant,
                require_bundle=args.require_bundle or (args.checkpoint is None and variant is not None),
                context_constraint_mode=args.context_mode)
            if any(parameter.device.type != "cpu" for parameter in model.parameters()):
                raise RuntimeError("native checkpoint did not load on CPU")
            del model
        return "strict CPU native load: " + ", ".join(str(value or "standalone") for value in variants)
    if target == "partiscore":
        from src.scoring.api import validate_checkpoint
        path = args.checkpoint or Path(args.model_dir) / "ckpts" / "partiscore.ckpt"
        weights = getattr(args, "weights", "raw")
        validate_checkpoint(path, weights=weights)
        return f"strict CPU PartiScore v2 keys/shapes/finite tensors ({weights})"
    if target in {"proteinmpnn", "solublempnn"}:
        return _protein_load(Path(args.model_dir), soluble=target == "solublempnn",
            checkpoint=args.checkpoint, ca_only=args.ca_only)
    if target == "ligandmpnn":
        return _ligand_load(checkpoint=args.checkpoint, model_type=args.model_type or "ligand_mpnn",
            packing_only=args.packing_only, sequence_only=args.sequence_only)
    if target == "nampnn":
        from src.design.rewards.na_mpnn_runtime import load_na_mpnn_model
        loaded = load_na_mpnn_model(code_root(), device="cpu")
        return f"strict CPU NA-MPNN keys/shapes/finite tensors: {_finite_state(loaded.model)}"
    if target == "cyclicmpnn":
        return _cyclic_load(args.checkpoint)
    if target.startswith("esmfold2-"):
        return _esm_load(target.removeprefix("esmfold2-"))
    if target in {"af3", "af3-la"}:
        return _af3_load(Path(args.model_dir), target)
    if target == "af2":
        return _af2_load()
    if target == "chemistry":
        if args.checkpoint is not None:
            from src.design.utils.chemistry import load_trusted_ccd_components
            records = load_trusted_ccd_components(Path(args.checkpoint), {"ATP"})
            if "ATP" not in records:
                raise ValueError("trusted CCD has no ATP")
            return "task-required trusted chemistry decoded on CPU (ATP representative)"
        from src.common.chemistry.ccd import ccd_components, ccd_metadata
        metadata = ccd_metadata()
        components = ccd_components()
        if "ALA" not in metadata:
            raise ValueError("packaged native CCD has no ALA polymer metadata")
        for name in ("ALA", "ATP"):
            if name not in components:
                raise ValueError(f"packaged native CCD has no {name}")
        return f"Git-supplied native chemistry decoded on CPU ({len(components)} components)"
    raise ValueError(f"unsupported CPU load target: {target}")


def _esm_load(mode: str) -> str:
    _source("esmfold2", "esm")
    _source("esmfold2", "transformers", "src")
    modeling = importlib.import_module("transformers.models.esmfold2.modeling_esmfold2")
    path = community_asset("esmfold2") / "checkpoint" / mode
    result = modeling.ESMFold2Model.from_pretrained(str(path), load_esmc=False,
        local_files_only=True, output_loading_info=True, device_map=None)
    model, information = result
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if information.get(key):
            raise ValueError(f"ESMFold2 {mode} {key}: {information[key]}")
    _finite_state(model)
    del model
    esmc = importlib.import_module("transformers.models.esmc.modeling_esmc")
    model, information = esmc.ESMCForMaskedLM.from_pretrained(
        str(community_asset("esmfold2") / "checkpoint" / "esmc-6b"),
        local_files_only=True, output_loading_info=True, device_map=None,
        attn_implementation="eager")
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if information.get(key):
            raise ValueError(f"ESMC {key}: {information[key]}")
    _finite_state(model)
    return f"strict CPU local HF keys/shapes/finite tensors: ESMFold2 {mode} and ESMC-6B"


def _af2_load() -> str:
    _source("colabdesign")
    from colabdesign import mk_afdesign_model
    from colabdesign.af.alphafold.model import data
    import jax
    import jax.numpy as jnp
    import numpy as np
    root = community_asset("colabdesign") / "checkpoint" / "af2"
    count = 0
    for suffix in ("", "_ptm", "_multimer_v3"):
        for number in range(1, 6):
            params = data.get_model_haiku_params(f"model_{number}{suffix}", str(root))
            if not params:
                raise ValueError("empty AF2 parameter tree")
            for scope, values in params.items():
                for name, value in values.items():
                    if not np.isfinite(value).all():
                        raise ValueError(f"nonfinite AF2 parameter: {scope}/{name}")
            count += 1
    trees = 0
    for name, multimer in (("model_1_ptm", False), ("model_2_ptm", False),
                           ("model_1_multimer_v3", True)):
        built = mk_afdesign_model(protocol="hallucination", use_multimer=multimer,
            use_templates=False, model_names=[name], data_dir=str(root), num_recycles=1)
        built.prep_inputs(length=4)
        batch = {key: value for key, value in built._inputs.items()
                 if key not in {"bias", "rm_template", "rm_template_seq", "rm_template_sc"}}
        length = batch["aatype"].shape[0]
        batch["prev"] = {"prev_msa_first_row": jnp.zeros((length, 256)),
            "prev_pair": jnp.zeros((length, length, 128)),
            "prev_pos": jnp.zeros((length, 37, 3))}
        batch["use_dropout"] = False
        batch["mask_template_interchain"] = False
        expected = jax.eval_shape(built._model["runner"].init, jax.random.PRNGKey(0), batch)
        state = built._model_params[0]
        expected_keys = {(scope, key) for scope, values in expected.items() for key in values}
        keys = {(scope, key) for scope, values in state.items() for key in values}
        if keys != expected_keys:
            raise ValueError(f"AF2 {name} architecture keys mismatch: missing={expected_keys - keys}, extra={keys - expected_keys}")
        for scope, key in keys:
            if np.shape(state[scope][key]) != expected[scope][key].shape:
                raise ValueError(f"AF2 {name} architecture shape mismatch: {scope}/{key}")
        trees += 1
    return (f"CPU AF2 local load/finite trees: {count} weights; exact Haiku keys/shapes "
            f"for {trees} supported runtime architectures (PTM 1/2 and multimer-v3)")


def community_smoke(target: str, root: Path, out: Path | None) -> str:
    # A model-presence or CPU-load result must never masquerade as real inference.
    from .backend_smoke import run_backend_smoke
    return run_backend_smoke(target, root, out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--variant")
    parser.add_argument("--require-bundle", action="store_true")
    parser.add_argument("--context-mode", default="auto")
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--model-type")
    parser.add_argument("--ca-only", action="store_true")
    parser.add_argument("--packing-only", action="store_true")
    parser.add_argument("--sequence-only", action="store_true")
    parser.add_argument("--antibody", action="store_true")
    args = parser.parse_args(argv)
    try:
        with using_model_dir(args.model_dir):
            print(load_target(args))
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
