"""Strict combined-checkpoint loading and the standalone compile boundary."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from partidesign.paths import resource_root

from src.design.model.bundle import is_partidesign_bundle, select_bundle_checkpoint
from src.design.model.core import PartiDesignInferenceModel
from src.design.model.solver import (
    STATE_COMPONENTS,
    STATE_DIMS,
    validate_context_constraint_mode,
    validate_sampling_config,
)


_COMPILE_KEYS = frozenset(
    ("enabled", "backend", "mode", "dynamic", "fullgraph")
)
_REQUESTED_CONTEXT_MODES = frozenset(("auto", "flex", "rigid"))
_UNINDEXED_MOTIF_MAX_TOKENS = 320
_CHECKPOINT_CONTRACT_KEY = "partidesign_checkpoint"


@dataclass(frozen=True, slots=True)
class CheckpointMigrationProbe:
    """Strict in-memory provenance for a legacy-FM/current-AE migration."""

    context_constraint_mode: str
    network_tensor_count: int
    donor_autoencoder_tensor_count: int
    shared_autoencoder_tensor_count: int
    sampling_schema_migration: str
    replaced_source_autoencoder_keys: tuple[str, ...]
    introduced_donor_autoencoder_keys: tuple[str, ...]


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return value


def _plain(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item, f"{label}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{label}[]") for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"{label} contains unsupported value {type(value)!r}")


def _network_unindexed_motif_contract(
    config: Mapping[str, Any],
) -> tuple[bool, int]:
    result = _plain(_as_mapping(config, "checkpoint cfg_exp.network"), "network")
    if "unindexed_motif" not in result:
        enabled = False
        max_tokens = _UNINDEXED_MOTIF_MAX_TOKENS
    else:
        raw = result["unindexed_motif"]
        raw = _as_mapping(raw, "checkpoint cfg_exp.network.unindexed_motif")
        if set(raw) != {"enabled", "max_tokens"}:
            raise ValueError(
                "checkpoint network.unindexed_motif must contain exactly enabled and max_tokens"
            )
        enabled = raw["enabled"]
        max_tokens = raw["max_tokens"]
        if type(enabled) is not bool:
            raise TypeError("checkpoint network.unindexed_motif.enabled must be bool")
        if type(max_tokens) is not int or max_tokens != _UNINDEXED_MOTIF_MAX_TOKENS:
            raise ValueError(
                "checkpoint network.unindexed_motif.max_tokens must be exactly 320"
            )
    return enabled, max_tokens


def _validate_checkpoint_unindexed_motif_metadata(
    checkpoint: Mapping[str, Any],
    network_config: Mapping[str, Any],
) -> tuple[bool, int]:
    nested = _network_unindexed_motif_contract(network_config)
    if _CHECKPOINT_CONTRACT_KEY not in checkpoint:
        if nested[0]:
            raise ValueError(
                "enabled unindexed motif checkpoint requires explicit root metadata"
            )
        return nested
    metadata = checkpoint[_CHECKPOINT_CONTRACT_KEY]
    metadata = _as_mapping(metadata, f"checkpoint {_CHECKPOINT_CONTRACT_KEY}")
    has_enabled = "unindexed_motif_enabled" in metadata
    has_max_tokens = "unindexed_motif_max_tokens" in metadata
    if not has_enabled and not has_max_tokens:
        if nested[0]:
            raise ValueError(
                "enabled unindexed motif checkpoint requires explicit root metadata"
            )
        return nested
    if has_enabled != has_max_tokens:
        raise ValueError(
            "checkpoint unindexed motif metadata must contain both enabled and max_tokens"
        )
    explicit = (
        metadata["unindexed_motif_enabled"],
        metadata["unindexed_motif_max_tokens"],
    )
    if type(explicit[0]) is not bool or type(explicit[1]) is not int:
        raise TypeError("checkpoint unindexed motif root metadata has invalid types")
    if explicit[1] != _UNINDEXED_MOTIF_MAX_TOKENS:
        raise ValueError("checkpoint unindexed motif max_tokens must be exactly 320")
    if explicit != nested:
        raise ValueError(
            "checkpoint root metadata and nested config disagree on unindexed motif"
        )
    return nested


def _standalone_network_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate checkpoint metadata, then force the public dynamic-N model path."""

    result = _plain(_as_mapping(config, "checkpoint cfg_exp.network"), "network")
    _, max_tokens = _network_unindexed_motif_contract(result)
    result["unindexed_motif"] = {
        "enabled": False,
        "max_tokens": max_tokens,
    }
    return result


def _fixed_ae_network_config() -> dict[str, Any]:
    path = resource_root() / "configs" / "resources" / "model" / "ae_config.yaml"
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"cannot read packaged AE configuration: {path}") from error
    config = _plain(_as_mapping(document, "fixed AE config"), "fixed AE config")
    if config.get("latent_z_dim") != STATE_DIMS["local_latents"]:
        raise ValueError("packaged AE config must use latent_z_dim=8")
    for branch in ("encoder", "decoder"):
        branch_config = _as_mapping(config.get(branch), f"fixed AE config.{branch}")
        if branch_config.get("update_pair_repr") is not False:
            raise ValueError(f"fixed AE config.{branch}.update_pair_repr must be false")
        if branch_config.get("use_tri_mult") is not False:
            raise ValueError(f"fixed AE config.{branch}.use_tri_mult must be false")
    return config


def _checkpoint_experiment_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    hyper_parameters = _as_mapping(
        checkpoint.get("hyper_parameters"),
        "checkpoint.hyper_parameters",
    )
    config = _plain(
        _as_mapping(hyper_parameters.get("cfg_exp"), "checkpoint cfg_exp"),
        "checkpoint cfg_exp",
    )
    for key in ("network", "flow", "sampling"):
        _as_mapping(config.get(key), f"checkpoint cfg_exp.{key}")
    return config


def _validate_checkpoint_flow(config: Mapping[str, Any]) -> str:
    mode = validate_context_constraint_mode(config.get("context_constraint_mode"))
    product = _as_mapping(
        config.get("product_flowmatcher"),
        "checkpoint cfg_exp.flow.product_flowmatcher",
    )
    if set(product) != set(STATE_COMPONENTS) or len(product) != len(STATE_COMPONENTS):
        raise ValueError(
            "checkpoint product_flowmatcher components must be exactly "
            f"{STATE_COMPONENTS}"
        )
    for component in STATE_COMPONENTS:
        component_config = _as_mapping(
            product[component],
            f"checkpoint cfg_exp.flow.product_flowmatcher.{component}",
        )
        if component_config.get("dim") != STATE_DIMS[component]:
            raise ValueError(f"checkpoint {component} flow dimension is incompatible")
        if component_config.get("zero_com_noise") is not False:
            raise ValueError(
                f"checkpoint {component} requires unsupported zero_com_noise"
            )
    return mode


def _resolve_context_constraint_mode(
    checkpoint_mode: str,
    requested_mode: str,
) -> str:
    checkpoint_mode = validate_context_constraint_mode(checkpoint_mode)
    if not isinstance(requested_mode, str) or requested_mode not in _REQUESTED_CONTEXT_MODES:
        raise ValueError(
            "model.context_constraint_mode must be exactly auto, flex, or rigid, got "
            f"{requested_mode!r}"
        )
    if requested_mode != "auto" and requested_mode != checkpoint_mode:
        raise ValueError(
            "model.context_constraint_mode mismatch: requested "
            f"{requested_mode!r}, checkpoint declares {checkpoint_mode!r}"
        )
    return checkpoint_mode


def _validate_checkpoint_sampling(config: Mapping[str, Any]) -> None:
    keys = set(config)
    if keys not in ({"args", "model"}, {"args", "model", "ligand_graph"}):
        raise ValueError("checkpoint sampling has an unsupported top-level schema")
    args = dict(_as_mapping(config.get("args"), "checkpoint sampling.args"))
    allowed_args = {"nsteps", "self_cond", "save_trajectory_every"}
    if set(args) != allowed_args:
        raise ValueError(
            "checkpoint sampling.args must contain nsteps, self_cond, and "
            "save_trajectory_every"
        )
    if args.pop("save_trajectory_every") != 0:
        raise ValueError("combined checkpoint must disable trajectory collection")
    # Saved sampling modes describe checkpoint provenance. Runtime sampling is
    # validated independently and uses shared polymer/ligand SC updates.
    model = _plain(
        _as_mapping(config.get("model"), "checkpoint sampling.model"),
        "checkpoint sampling.model",
    )
    saved_ligand_modes: set[str] = set()
    for component in STATE_COMPONENTS:
        component_config = _as_mapping(
            model.get(component), f"checkpoint sampling.model.{component}"
        )
        step = _as_mapping(
            component_config.get("simulation_step_params"),
            f"checkpoint {component} simulation_step_params",
        )
        modes = _as_mapping(
            step.get("sampling_mode"), f"checkpoint {component} sampling_mode"
        )
        if (
            set(modes) != {"polymer", "ligand"}
            or modes["polymer"] != "sc"
            or type(modes["ligand"]) is not str
            or modes["ligand"] not in {"vf", "sc"}
        ):
            raise ValueError(
                f"checkpoint {component} requires polymer=sc and ligand=vf or sc"
            )
        saved_ligand_modes.add(modes["ligand"])
        model[component]["simulation_step_params"]["sampling_mode"] = {
            "polymer": "sc", "ligand": "sc"
        }
    if len(saved_ligand_modes) != 1:
        raise ValueError("checkpoint components must use the same ligand sampling mode")
    normalized = {
        "args": args,
        "model": model,
    }
    if "ligand_graph" in config:
        normalized["ligand_graph"] = config["ligand_graph"]
    validate_sampling_config(normalized)


def _validate_legacy_flex_sampling(config: Mapping[str, Any]) -> None:
    """Recognize only the exact pre-modality flex sampling ABI.

    This schema used score correction for every token and predates both the
    polymer/ligand mode split and TSR parameters.  It is accepted only by the
    migration probe; ordinary checkpoint loading continues to reject it.
    """

    top = _as_mapping(config, "legacy checkpoint sampling")
    if set(top) != {"args", "model"}:
        raise ValueError("legacy flex sampling keys must be exactly args/model")
    args = _as_mapping(top.get("args"), "legacy checkpoint sampling.args")
    if dict(args) != {
        "nsteps": 400,
        "self_cond": True,
        "save_trajectory_every": 0,
    }:
        raise ValueError("legacy flex sampling args are not the recognized 400-step ABI")
    model = _as_mapping(top.get("model"), "legacy checkpoint sampling.model")
    if set(model) != set(STATE_COMPONENTS):
        raise ValueError("legacy flex sampling model components are incompatible")
    expected_schedule = {
        "token_center": ("log", "1/t"),
        "local_latents": ("power", "tan"),
    }
    for component in STATE_COMPONENTS:
        component_config = _as_mapping(
            model[component],
            f"legacy checkpoint sampling.model.{component}",
        )
        if set(component_config) != {"schedule", "gt", "simulation_step_params"}:
            raise ValueError(f"legacy {component} sampling keys are incompatible")
        schedule = _as_mapping(
            component_config["schedule"],
            f"legacy {component} schedule",
        )
        gt = _as_mapping(component_config["gt"], f"legacy {component} gt")
        schedule_mode, gt_mode = expected_schedule[component]
        if dict(schedule) != {"mode": schedule_mode, "p": 2.0}:
            raise ValueError(f"legacy {component} schedule is incompatible")
        if dict(gt) != {"mode": gt_mode, "p": 1.0, "clamp_val": None}:
            raise ValueError(f"legacy {component} score correction is incompatible")
        step = _as_mapping(
            component_config["simulation_step_params"],
            f"legacy {component} simulation step",
        )
        if dict(step) != {
            "sampling_mode": "sc",
            "sc_scale_noise": 0.1,
            "sc_scale_score": 1.0,
            "t_lim_ode": 0.98,
            "t_lim_ode_below": 0.02,
            "center_every_step": False,
        }:
            raise ValueError(f"legacy {component} simulation step is incompatible")


def _validate_state_dict(
    state_dict: Mapping[str, Any],
    expected: Mapping[str, torch.Tensor],
) -> None:
    if not state_dict:
        raise ValueError("checkpoint state_dict is empty")
    if not all(isinstance(key, str) for key in state_dict):
        raise TypeError("checkpoint state_dict keys must be strings")
    _validate_state_namespaces(state_dict)

    actual_keys = set(state_dict)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:5]}")
        raise ValueError("combined checkpoint state mismatch: " + "; ".join(details))

    for key in sorted(expected_keys):
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"checkpoint state {key!r} must be a tensor")
        if value.shape != expected[key].shape:
            raise ValueError(
                f"checkpoint state {key!r} shape {tuple(value.shape)} does not match "
                f"expected {tuple(expected[key].shape)}"
            )


def _validate_state_namespaces(state_dict: Mapping[str, Any]) -> None:
    unexpected_roots = sorted(
        key
        for key in state_dict
        if not key.startswith("nn.") and not key.startswith("autoencoder.")
    )
    if unexpected_roots:
        preview = ", ".join(unexpected_roots[:5])
        raise ValueError(f"checkpoint contains mixed/unexpected state roots: {preview}")
    for root in ("nn.", "autoencoder."):
        if not any(key.startswith(root) for key in state_dict):
            raise ValueError(f"checkpoint is missing the {root} state namespace")


def _read_checkpoint(path_value: str | Path, label: str) -> Mapping[str, Any]:
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        document = torch.load(
            str(path),
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except Exception as error:
        raise RuntimeError(f"failed to read {label}: {path}") from error
    return _as_mapping(document, label)



def _read_selected_checkpoint(
    checkpoint: str | Path,
    *,
    variant: str | None = None,
    require_bundle: bool = False,
) -> tuple[Mapping[str, Any], Path, str | None]:
    """Select one member or retain an explicitly supplied combined file."""
    path = Path(checkpoint).expanduser()
    document = _read_checkpoint(path, "PartiDesign checkpoint")
    if not is_partidesign_bundle(document):
        if require_bundle:
            raise ValueError("route_ckpt requires a multi-branch PartiDesign checkpoint")
        return document, path, None
    if variant is None:
        raise ValueError("a model bundle requires route=true; use route_ckpt to select a branch")
    return select_bundle_checkpoint(document, variant), path, variant


def _validated_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    requested_mode: str,
    allow_legacy_flex_sampling: bool = False,
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    state_dict = _as_mapping(
        checkpoint.get("state_dict"),
        "checkpoint.state_dict",
    )
    experiment_config = _checkpoint_experiment_config(checkpoint)
    network_config = _as_mapping(
        experiment_config["network"],
        "checkpoint cfg_exp.network",
    )
    _validate_checkpoint_unindexed_motif_metadata(checkpoint, network_config)
    checkpoint_mode = _validate_checkpoint_flow(
        _as_mapping(experiment_config["flow"], "checkpoint cfg_exp.flow")
    )
    resolved_mode = _resolve_context_constraint_mode(
        checkpoint_mode,
        requested_mode,
    )
    sampling = _as_mapping(
        experiment_config["sampling"],
        "checkpoint cfg_exp.sampling",
    )
    if allow_legacy_flex_sampling:
        if resolved_mode != "flex":
            raise ValueError("legacy sampling migration is valid only for a flex checkpoint")
        _validate_legacy_flex_sampling(sampling)
    else:
        _validate_checkpoint_sampling(sampling)
    return state_dict, network_config, resolved_mode


def _meta_model(
    network_config: Mapping[str, Any],
    context_constraint_mode: str,
) -> PartiDesignInferenceModel:
    with torch.device("meta"):
        return PartiDesignInferenceModel(
            fm_network_config=_standalone_network_config(network_config),
            ae_network_config=_fixed_ae_network_config(),
            context_constraint_mode=context_constraint_mode,
        )


def _strict_assign_state(
    model: PartiDesignInferenceModel,
    state_dict: Mapping[str, Any],
) -> None:
    _validate_state_dict(state_dict, model.state_dict())
    incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint assignment reported incompatible keys")


def probe_legacy_checkpoint_migration(
    source_checkpoint: str | Path,
    *,
    autoencoder_donor_checkpoint: str | Path,
) -> CheckpointMigrationProbe:
    """Strictly prove a legacy-FM/current-AE merge without writing a checkpoint.

    The source supplies every ``nn.*`` tensor and its explicit context mode.  The
    donor supplies every current ``autoencoder.*`` tensor.  All shared source/donor
    autoencoder tensors must be bitwise identical, and the resulting in-memory state
    must strict-assign into the checkpoint-specific FM network plus packaged current
    VAE architecture.  No missing value is synthesized and neither input is modified.
    """

    source = _read_checkpoint(source_checkpoint, "source combined checkpoint")
    source_state, source_network, source_mode = _validated_checkpoint_contract(
        source,
        requested_mode="auto",
        allow_legacy_flex_sampling=True,
    )
    donor = _read_checkpoint(
        autoencoder_donor_checkpoint,
        "autoencoder donor combined checkpoint",
    )
    donor_state, _donor_network, _donor_mode = _validated_checkpoint_contract(
        donor,
        requested_mode="auto",
    )
    _validate_state_namespaces(source_state)
    _validate_state_namespaces(donor_state)

    source_nn = {
        key: value for key, value in source_state.items() if key.startswith("nn.")
    }
    source_ae = {
        key: value
        for key, value in source_state.items()
        if key.startswith("autoencoder.")
    }
    donor_ae = {
        key: value
        for key, value in donor_state.items()
        if key.startswith("autoencoder.")
    }
    if not source_nn or not source_ae or not donor_ae:
        raise ValueError(
            "legacy migration requires source nn/autoencoder and donor autoencoder tensors"
        )

    shared = sorted(set(source_ae).intersection(donor_ae))
    if not shared:
        raise ValueError("source and donor autoencoders have no shared tensor ABI")
    for key in shared:
        source_value = source_ae[key]
        donor_value = donor_ae[key]
        if not isinstance(source_value, torch.Tensor) or not isinstance(
            donor_value, torch.Tensor
        ):
            raise TypeError(f"shared autoencoder state {key!r} must be tensors")
        if source_value.shape != donor_value.shape or not torch.equal(
            source_value, donor_value
        ):
            raise ValueError(
                f"source/donor shared autoencoder tensor differs: {key}"
            )

    source_only = tuple(sorted(set(source_ae).difference(donor_ae)))
    donor_only = tuple(sorted(set(donor_ae).difference(source_ae)))
    if not source_only or not donor_only:
        raise ValueError(
            "legacy migration probe requires explicit source-only and donor-only "
            "autoencoder tensors"
        )
    merged_state = {**source_nn, **donor_ae}
    model = _meta_model(source_network, source_mode)
    _strict_assign_state(model, merged_state)
    if model.context_constraint_mode != source_mode:
        raise AssertionError("migration probe changed checkpoint context mode")
    return CheckpointMigrationProbe(
        context_constraint_mode=source_mode,
        network_tensor_count=len(source_nn),
        donor_autoencoder_tensor_count=len(donor_ae),
        shared_autoencoder_tensor_count=len(shared),
        sampling_schema_migration=(
            "legacy_all_sc_without_tsr_to_current_validated_runtime_sampling"
        ),
        replaced_source_autoencoder_keys=source_only,
        introduced_donor_autoencoder_keys=donor_only,
    )


def resolve_checkpoint_context_constraint_mode(
    checkpoint: str | Path,
    *,
    requested_mode: str = "auto",
    variant: str | None = None,
    require_bundle: bool = False,
) -> str:
    """Resolve explicit nested checkpoint metadata without constructing the model."""

    checkpoint_mapping, _source, _member = _read_selected_checkpoint(
        checkpoint, variant=variant, require_bundle=require_bundle,
    )
    _state_dict, _network_config, resolved_mode = _validated_checkpoint_contract(
        checkpoint_mapping,
        requested_mode=requested_mode,
    )
    return resolved_mode


def load_combined_checkpoint(
    checkpoint: str | Path,
    *,
    context_constraint_mode: str = "auto",
    variant: str | None = None,
    require_bundle: bool = False,
) -> PartiDesignInferenceModel:
    """Load a selected bundle member or one strict combined Lightning checkpoint.

    The checkpoint is memory-mapped on CPU and assigned into a meta-constructed model,
    avoiding a second full model-sized allocation before the caller moves it to its
    inference device.
    """

    checkpoint_mapping, source, member = _read_selected_checkpoint(
        checkpoint, variant=variant, require_bundle=require_bundle,
    )
    state_dict, network_config, resolved_mode = _validated_checkpoint_contract(
        checkpoint_mapping,
        requested_mode=context_constraint_mode,
    )
    model = _meta_model(network_config, resolved_mode)
    _strict_assign_state(model, state_dict)
    model._checkpoint_source = str(source)
    model._checkpoint_member = member
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def validate_compile_config(config: Mapping[str, Any]) -> None:
    mapping = _as_mapping(config, "model.compile")
    if set(mapping) != set(_COMPILE_KEYS) or len(mapping) != len(_COMPILE_KEYS):
        raise ValueError(f"model.compile keys must be exactly {sorted(_COMPILE_KEYS)}")
    for key in ("enabled", "dynamic", "fullgraph"):
        if type(mapping[key]) is not bool:
            raise TypeError(f"model.compile.{key} must be bool")
    for key in ("backend", "mode"):
        if not isinstance(mapping[key], str) or not mapping[key].strip():
            raise TypeError(f"model.compile.{key} must be a non-empty string")


def compile_denoiser(
    model: PartiDesignInferenceModel,
    compile_config: Mapping[str, Any],
) -> PartiDesignInferenceModel:
    """Compile only ``model.nn`` with a stable Inductor padding policy.

    Capture the policy in the compiler options so that lazy compilation and
    later recompilation use it without changing process-global Torch settings.
    Compile errors propagate without fallback.
    """

    if not isinstance(model, PartiDesignInferenceModel):
        raise TypeError("compile_denoiser requires PartiDesignInferenceModel")
    validate_compile_config(compile_config)
    if getattr(model, "_denoiser_compiled", False):
        raise RuntimeError("the denoiser has already been compiled")
    if not compile_config["enabled"]:
        model._inference_compile_config = dict(compile_config)
        return model
    compile_kwargs: dict[str, Any] = {
        "backend": compile_config["backend"],
        "dynamic": compile_config["dynamic"],
        "fullgraph": compile_config["fullgraph"],
    }
    compiler_options: dict[str, Any] = {}
    if compile_config["backend"] == "inductor":
        from torch._inductor import list_mode_options

        # Torch rejects mode and options together. Expanding the requested mode
        # preserves its settings while pinning the benchmark-dependent pad_mm
        # choice responsible for independent compilations using different GEMMs.
        compiler_options.update(
            list_mode_options(compile_config["mode"], compile_config["dynamic"])
        )
        compiler_options["shape_padding"] = False
        compile_kwargs["options"] = compiler_options
    else:
        compile_kwargs["mode"] = compile_config["mode"]
    model.nn = torch.compile(model.nn, **compile_kwargs)
    model._denoiser_compiled = True
    model._inference_compile_config = dict(compile_config)
    model._inference_compiler_options = dict(compiler_options)
    return model
