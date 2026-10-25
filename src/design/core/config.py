from __future__ import annotations

from dataclasses import dataclass
import math
import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.errors import OmegaConfBaseException

from ..model.solver import validate_sampling_config
from .records import DesignConfigError
from .compile_policy import validate_bucket_compile_policy, validate_compile_enabled
from .omission import normalize_rm_token_type
from ..rewards.interface import normalize_fixed_interface
from ..rewards.aa_omission import resolve_rm_aa
from partidesign.paths import inference_config_dir, community_asset


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: yaml.Loader, node: yaml.Node, deep: bool = False) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise DesignConfigError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


@dataclass(frozen=True)
class LoadedDesignConfig:
    data: DictConfig
    invocation_dir: Path


def _inference_config_dir() -> Path:
    """Return the release-bundle inference config directory."""

    config_dir = inference_config_dir()
    if not config_dir.is_dir():
        raise DesignConfigError(f"inference config directory does not exist: {config_dir}")
    return config_dir


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.suffix.lower() != ".yaml":
        raise DesignConfigError(f"YAML path must end in .yaml: {source}")
    try:
        payload = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except DesignConfigError:
        raise
    except Exception as exc:
        raise DesignConfigError(f"cannot read YAML {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DesignConfigError(f"YAML root must be a mapping: {source}")
    if not all(isinstance(key, str) for key in payload):
        raise DesignConfigError(f"YAML mapping keys must be strings: {source}")
    return payload


def _closed(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        raise DesignConfigError(f"unknown {where} keys: {unknown}")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DesignConfigError(f"{where} must be a mapping")
    return value


def _int(value: Any, where: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DesignConfigError(f"{where} must be an integer")
    if value < minimum:
        raise DesignConfigError(f"{where} must be >= {minimum}")
    return int(value)


def _positive_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DesignConfigError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not result > 0:
        raise DesignConfigError(f"{where} must be finite and > 0")
    return result


def _finite_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DesignConfigError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DesignConfigError(f"{where} must be finite")
    return result


def _nonnegative_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DesignConfigError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise DesignConfigError(f"{where} must be finite and >= 0")
    return result


_ESMFOLD2_OPTIMIZATION_MODES = frozenset({"off", "exact", "fast", "auto"})
_ESMFOLD2_OPTIMIZATION_LEVERS = frozenset(
    {
        "tg",
        "sg",
        "eg",
        "pb",
        "t1",
        "t3",
        "t5",
        "t6",
        "t10",
        "mk",
        "ls",
        "rg",
        "ax",
        "af",
        "t15",
        "xte",
        "t16",
        "tx",
        "disto",
        "xln",
        "ro",
        "kd",
        "dit",
        "x2b",
        "x3",
        "x6",
        "x7",
        "x8",
        "x10",
        "x4",
        "swa",
        "fc",
    }
)


def _esmfold2_optimization_defaults() -> dict[str, Any]:
    """Return the automatic internal policy for omitted ESMFold2 optimization."""

    return {
        "mode": "auto",
        "disable": [],
    }


def _validate_esmfold2_optimization(
    value: Any,
    where: str = "reward.esmfold2.optimization",
) -> dict[str, Any]:
    optimization = {**_esmfold2_optimization_defaults(), **_mapping(value, where)}
    _closed(
        optimization,
        {
            "mode",
            "disable",
        },
        where,
    )
    mode = optimization.get("mode")
    if not isinstance(mode, str) or mode not in _ESMFOLD2_OPTIMIZATION_MODES:
        raise DesignConfigError(
            f"{where}.mode must be off, exact, fast, or auto"
        )
    disable = optimization.get("disable")
    if not isinstance(disable, list):
        raise DesignConfigError(f"{where}.disable must be a list of strings")
    seen: set[str] = set()
    for lever in disable:
        if not isinstance(lever, str):
            raise DesignConfigError(f"{where}.disable must be a list of strings")
        if lever not in _ESMFOLD2_OPTIMIZATION_LEVERS:
            raise DesignConfigError(
                f"{where}.disable contains unsupported lever {lever!r}"
            )
        if lever in seen:
            raise DesignConfigError(f"{where}.disable contains duplicate {lever!r}")
        seen.add(lever)
    return {
        "mode": mode,
        "disable": list(disable),
    }


def _resolve_path(value: Any, invocation_dir: Path, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise DesignConfigError(f"{where} must be a non-empty local path string or null")
    if "://" in value:
        raise DesignConfigError(f"{where} does not accept a URI: {value!r}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = invocation_dir / path
    return str(path.resolve())


def _validate_model(model: Mapping[str, Any], invocation_dir: Path) -> None:
    _closed(model, {"checkpoint", "precision", "context_constraint_mode", "compile", "optimization", "execution_backend", "bioir"}, "model")
    execution_backend = model.get("execution_backend", "auto")
    if not isinstance(execution_backend, str) or execution_backend not in {"auto", "native", "bioir"}:
        raise DesignConfigError("model.execution_backend must be auto, native, or bioir")
    model["execution_backend"] = execution_backend
    bioir = {"cuda_graph": True, "max_graphs": 4, "max_tokens": 512}
    if "bioir" in model:
        bioir.update(_mapping(model["bioir"], "model.bioir"))
    _closed(bioir, {"cuda_graph", "max_graphs", "max_tokens"}, "model.bioir")
    if type(bioir["cuda_graph"]) is not bool:
        raise DesignConfigError("model.bioir.cuda_graph must be boolean")
    for key in ("max_graphs", "max_tokens"):
        _int(bioir[key], "model.bioir." + key, minimum=1)
    model["bioir"] = bioir
    if model.get("precision") != "bf16-mixed":
        raise DesignConfigError("model.precision must be 'bf16-mixed'")
    context_constraint_mode = model.get("context_constraint_mode")
    if not isinstance(context_constraint_mode, str) or context_constraint_mode not in {
        "auto",
        "rigid",
        "flex",
    }:
        raise DesignConfigError(
            "model.context_constraint_mode must be auto, rigid, or flex"
        )
    compile_cfg = _mapping(model.get("compile"), "model.compile")
    _closed(compile_cfg, {"enabled", "backend", "mode", "dynamic", "fullgraph", "token_buckets"}, "model.compile")
    validate_compile_enabled(compile_cfg.get("enabled"))
    for key in ("dynamic", "fullgraph"):
        if not isinstance(compile_cfg.get(key), bool):
            raise DesignConfigError(f"model.compile.{key} must be boolean")
    if compile_cfg.get("backend") != "inductor" or compile_cfg.get("mode") != "default":
        raise DesignConfigError("model.compile backend/mode must be inductor/default")
    if compile_cfg.get("fullgraph") is not True:
        raise DesignConfigError("model.compile.fullgraph is fixed to true")
    validate_bucket_compile_policy(compile_cfg)
    if execution_backend == "bioir" and (
        compile_cfg["enabled"] is True or compile_cfg.get("token_buckets") is not None
    ):
        raise DesignConfigError(
            "BioIR requires model.compile.enabled=false or auto, and token_buckets=null"
        )
    from ..model.optimization import DEFAULT_OPTIMIZATION, validate_optimization_config
    # Hydra ++ overrides may supply a single optional field. Keep defaults in
    # Python rather than requiring an optimization block in user YAML.
    optimization = DEFAULT_OPTIMIZATION
    if "optimization" in model:
        requested = _mapping(model["optimization"], "model.optimization")
        optimization = {**DEFAULT_OPTIMIZATION, **requested}
    try:
        validate_optimization_config(optimization)
    except (TypeError, ValueError) as error:
        raise DesignConfigError(str(error)) from error
    if "optimization" in model:
        model["optimization"] = optimization
    model["checkpoint"] = _resolve_path(model.get("checkpoint"), invocation_dir, "model.checkpoint")


def validate_sampling_preflight(sampling: Mapping[str, Any]) -> None:
    """Apply the solver-owned exact numerical schema at every public preflight."""

    try:
        validate_sampling_config(sampling)
    except (TypeError, ValueError) as exc:
        raise DesignConfigError(f"invalid sampling config: {exc}") from exc


def _validate_search_reward(
    search: Mapping[str, Any],
    reward: Mapping[str, Any],
    *,
    invocation_dir: Path,
) -> None:
    _closed(
        search,
        {"algorithm", "selection_scope", "max_batch_size", "best_of_n"},
        "search",
    )
    if search.get("selection_scope") != "task_length_plan":
        raise DesignConfigError("search.selection_scope must be task_length_plan")
    algorithm = search.get("algorithm")
    if algorithm is not None and not isinstance(algorithm, str):
        raise DesignConfigError("search.algorithm must be null or a string")
    if algorithm not in {None, "best_of_n"}:
        raise DesignConfigError(f"unsupported search.algorithm: {algorithm!r}")
    max_batch = search.get("max_batch_size")
    if max_batch is not None:
        _int(max_batch, "search.max_batch_size", minimum=1)
    best = _mapping(search.get("best_of_n"), "search.best_of_n")
    _closed(best, {"replicas"}, "search.best_of_n")
    _int(best.get("replicas"), "search.best_of_n.replicas", minimum=1)
    target = reward.get("_target_")
    if algorithm is None:
        _closed(reward, {"_target_"}, "reward")
        if target is not None:
            raise DesignConfigError("null search requires null reward")
    elif target == "src.design.rewards.clash.ClashReward":
        _closed(
            reward,
            {"_target_", "design_design_weight", "design_context_weight"},
            "reward",
        )
        design_design = _nonnegative_float(
            reward.get("design_design_weight"), "reward.design_design_weight"
        )
        design_context = _nonnegative_float(
            reward.get("design_context_weight"), "reward.design_context_weight"
        )
        if design_design == 0 and design_context == 0:
            raise DesignConfigError("at least one clash reward weight must be positive")
    elif target == "src.design.rewards.partiscore.PartiScoreReward":
        if algorithm != "best_of_n":
            raise DesignConfigError(
                "PartiScore reward is available only with best_of_n search"
            )
        _validate_partiscore_reward(reward, invocation_dir)
    elif target == "src.design.rewards.esmfold2.ESMFold2Reward":
        if algorithm != "best_of_n":
            raise DesignConfigError(
                "ESMFold2 reward is available only with best_of_n search"
            )
        _validate_esmfold2_reward(reward, invocation_dir)
    elif target == "src.design.rewards.af3.AF3Reward":
        if algorithm != "best_of_n":
            raise DesignConfigError("AF3 reward is available only with best_of_n search")
        _validate_af3_reward(reward, invocation_dir)
    elif target == "src.design.rewards.af2.AF2Reward":
        if algorithm != "best_of_n":
            raise DesignConfigError(
                "AF2 reward is available only with best_of_n search"
            )
        _validate_af2_reward(reward, invocation_dir)
    else:
        raise DesignConfigError(
            "non-null search requires the clash, PartiScore, ESMFold2, AF2, or AF3 reward"
        )


def _optional_local_path(
    mapping: Mapping[str, Any],
    key: str,
    invocation_dir: Path,
    where: str,
) -> None:
    value = mapping.get(key)
    if value is not None:
        mapping[key] = _resolve_path(value, invocation_dir, f"{where}.{key}")


def _validate_mpnn_common(
    config: Mapping[str, Any],
    *,
    invocation_dir: Path,
    where: str,
    extra_keys: set[str],
    allow_proteinmpnn_constraints: bool,
) -> None:
    _closed(
        config,
        {
            "model_type",
            "checkpoint",
            "temperature",
            "backbone_noise",
            "omit_aas",
            "rm_aa",
            "bias_aa",
            "pssm_path",
            "pssm_multi",
            "pssm_threshold",
            "pssm_log_odds",
            "pssm_bias",
            "tied_positions_path",
            *extra_keys,
        },
        where,
    )
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise DesignConfigError(f"{where}.model_type must be a non-empty string")
    _optional_local_path(config, "checkpoint", invocation_dir, where)
    if config.get("checkpoint") is None:
        raise DesignConfigError(f"{where}.checkpoint must be a local path")
    _optional_local_path(config, "pssm_path", invocation_dir, where)
    _optional_local_path(config, "tied_positions_path", invocation_dir, where)
    _positive_float(config.get("temperature"), f"{where}.temperature")
    _nonnegative_float(config.get("backbone_noise"), f"{where}.backbone_noise")
    try:
        config["omit_aas"] = resolve_rm_aa(config)
    except ValueError as exc:
        raise DesignConfigError(f"{where}.{exc}") from exc
    if not isinstance(config.get("omit_aas"), str):
        raise DesignConfigError(f"{where}.omit_aas must be a string")
    omit_aas = str(config["omit_aas"])
    if any(amino_acid not in "ACDEFGHIKLMNPQRSTVWY" for amino_acid in omit_aas):
        raise DesignConfigError(f"{where}.omit_aas must contain amino-acid letters")
    if set(omit_aas) == set("ACDEFGHIKLMNPQRSTVWY"):
        raise DesignConfigError(f"{where}.omit_aas cannot omit every amino acid")
    bias = _mapping(config.get("bias_aa"), f"{where}.bias_aa")
    for amino_acid, value in bias.items():
        if amino_acid not in "ACDEFGHIKLMNPQRSTVWY":
            raise DesignConfigError(f"{where}.bias_aa keys must be amino-acid letters")
        _finite_float(value, f"{where}.bias_aa.{amino_acid}")
    pssm_multi = _nonnegative_float(config.get("pssm_multi"), f"{where}.pssm_multi")
    if pssm_multi > 1:
        raise DesignConfigError(f"{where}.pssm_multi must be <= 1")
    _finite_float(config.get("pssm_threshold"), f"{where}.pssm_threshold")
    if not isinstance(config.get("pssm_log_odds"), bool) or not isinstance(
        config.get("pssm_bias"), bool
    ):
        raise DesignConfigError(f"{where}.pssm_log_odds/pssm_bias must be boolean")
    pssm_threshold = float(config["pssm_threshold"])
    if not allow_proteinmpnn_constraints:
        if (
            config.get("pssm_path") is not None
            or config.get("tied_positions_path") is not None
            or config.get("pssm_log_odds")
            or config.get("pssm_bias")
            or pssm_multi > 0
            or pssm_threshold != 0.0
        ):
            raise DesignConfigError(
                f"{where} does not support ProteinMPNN PSSM/tied controls"
            )
        return
    if config.get("pssm_path") is None and (
        config.get("pssm_log_odds")
        or config.get("pssm_bias")
        or pssm_multi > 0
        or pssm_threshold != 0.0
    ):
        raise DesignConfigError(f"{where} PSSM controls require pssm_path")
    if config.get("pssm_path") is not None and not (
        config.get("pssm_log_odds") or config.get("pssm_bias")
    ):
        raise DesignConfigError(
            f"{where}.pssm_path requires pssm_log_odds=true or pssm_bias=true"
        )
    if pssm_threshold != 0.0 and not config.get("pssm_log_odds"):
        raise DesignConfigError(
            f"{where}.pssm_threshold requires pssm_log_odds=true"
        )
    if pssm_multi > 0 and not config.get("pssm_bias"):
        raise DesignConfigError(f"{where}.pssm_multi > 0 requires pssm_bias=true")


def _validate_esmfold2_reward(
    reward: Mapping[str, Any],
    invocation_dir: Path | None = None,
    *,
    allow_na_mpnn: bool = True,
) -> None:
    _closed(
        reward,
        {"_target_", "structure_source", "sequence", "esmfold2", "score"},
        "reward",
    )
    structure_source = reward.get("structure_source")
    if structure_source not in {"fold", "design"}:
        raise DesignConfigError("reward.structure_source must be fold or design")

    sequence = _mapping(reward.get("sequence"), "reward.sequence")
    _closed(
        sequence,
        {
            "source",
            "fixed_positions",
            "fixed_interface",
            "proteinmpnn",
            "ligandmpnn",
            "cyclicmpnn",
            "auto_proteinmpnn",
            "solublempnn_defaults",
            "solublempnn",
            "na_mpnn",  # shared options do not grant source eligibility
        },
        "reward.sequence",
    )
    source = sequence.get("source")
    allowed_sources = {"auto", "design", "proteinmpnn", "ligandmpnn", "cyclicmpnn"}
    if allow_na_mpnn:
        allowed_sources.add("na_mpnn")
    if source not in allowed_sources:
        raise DesignConfigError(
            "reward.sequence.source must be auto, design, proteinmpnn, "
            "ligandmpnn, cyclicmpnn"
            + (", or na_mpnn" if allow_na_mpnn else "")
        )
    if structure_source == "design" and source != "design":
        raise DesignConfigError(
            "reward.structure_source=design requires reward.sequence.source=design"
        )
    fixed_positions = sequence.get("fixed_positions")
    if not isinstance(fixed_positions, list):
        raise DesignConfigError(
            "reward.sequence.fixed_positions must be a list of CHAIN:POSITION strings"
        )
    seen_fixed: set[tuple[str, int]] = set()
    for value in fixed_positions:
        if not isinstance(value, str) or value.count(":") != 1:
            raise DesignConfigError(
                "reward.sequence.fixed_positions must be a list of CHAIN:POSITION strings"
            )
        chain, raw_position = value.split(":", 1)
        try:
            position = int(raw_position)
        except ValueError as exc:
            raise DesignConfigError(
                f"reward.sequence.fixed_positions entry {value!r} has an invalid position"
            ) from exc
        if not chain or position < 1:
            raise DesignConfigError(
                f"reward.sequence.fixed_positions entry {value!r} is invalid"
            )
        key = (chain, position)
        if key in seen_fixed:
            raise DesignConfigError(
                f"reward.sequence.fixed_positions contains duplicate {value!r}"
            )
        seen_fixed.add(key)
    try:
        sequence["fixed_interface"] = normalize_fixed_interface(
            sequence.get("fixed_interface")
        )
    except ValueError as exc:
        raise DesignConfigError(f"reward.sequence.fixed_interface: {exc}") from exc
    root = Path.cwd() if invocation_dir is None else invocation_dir
    protein = _mapping(sequence.get("proteinmpnn"), "reward.sequence.proteinmpnn")
    _validate_mpnn_common(
        protein,
        invocation_dir=root,
        where="reward.sequence.proteinmpnn",
        extra_keys={"ca_only", "use_soluble_model"},
        allow_proteinmpnn_constraints=True,
    )
    if not isinstance(protein.get("ca_only"), bool) or not isinstance(
        protein.get("use_soluble_model"), bool
    ):
        raise DesignConfigError(
            "reward.sequence.proteinmpnn ca_only/use_soluble_model must be boolean"
        )
    protein_contract = {
        "protein_mpnn": (False, False),
        "soluble_mpnn": (False, True),
        "ca_protein_mpnn": (True, False),
    }
    if protein["model_type"] not in protein_contract:
        raise DesignConfigError(
            "reward.sequence.proteinmpnn.model_type must be protein_mpnn, "
            "soluble_mpnn, or ca_protein_mpnn"
        )
    expected_ca, expected_soluble = protein_contract[protein["model_type"]]
    if protein["ca_only"] != expected_ca or protein["use_soluble_model"] != expected_soluble:
        raise DesignConfigError(
            "reward.sequence.proteinmpnn model_type conflicts with "
            "ca_only/use_soluble_model"
        )
    for name in ("auto_proteinmpnn", "solublempnn_defaults", "solublempnn"):
        if name not in sequence:
            continue
        soluble = _mapping(sequence[name], "reward.sequence." + name)
        _validate_mpnn_common(soluble, invocation_dir=root,
                              where="reward.sequence." + name,
                              extra_keys={"ca_only", "use_soluble_model"},
                              allow_proteinmpnn_constraints=True)
        if name == "auto_proteinmpnn":
            contract = protein_contract.get(soluble.get("model_type"))
            if contract is None or (soluble.get("ca_only"), soluble.get("use_soluble_model")) != contract:
                raise DesignConfigError("reward.sequence.auto_proteinmpnn has conflicting backend flags")
            continue
        if (soluble.get("model_type") != "soluble_mpnn"
                or soluble.get("ca_only") is not False
                or soluble.get("use_soluble_model") is not True):
            raise DesignConfigError("reward.sequence." + name
                                    + " must select the SolubleMPNN contract")
    ligand = _mapping(sequence.get("ligandmpnn"), "reward.sequence.ligandmpnn")
    _validate_mpnn_common(
        ligand,
        invocation_dir=root,
        where="reward.sequence.ligandmpnn",
        extra_keys={
            "use_ligand_atoms",
            "use_sidechain_context",
            "pack_side_chains",
            "repack_everything",
            "homo_oligomer",
            "symmetry_residues_path",
            "symmetry_weights_path",
        },
        allow_proteinmpnn_constraints=False,
    )
    for key in (
        "use_ligand_atoms",
        "use_sidechain_context",
        "pack_side_chains",
        "repack_everything",
        "homo_oligomer",
    ):
        if not isinstance(ligand.get(key), bool):
            raise DesignConfigError(f"reward.sequence.ligandmpnn.{key} must be boolean")
    ligand_model_types = {
        "ligand_mpnn",
        "protein_mpnn",
        "soluble_mpnn",
    }
    if ligand["model_type"] not in ligand_model_types:
        raise DesignConfigError(
            "reward.sequence.ligandmpnn.model_type is not supported by the pinned "
            "LigandMPNN suite"
        )
    if float(ligand["backbone_noise"]) != 0.0:
        raise DesignConfigError(
            "reward.sequence.ligandmpnn.backbone_noise must be 0; the pinned "
            "LigandMPNN backend does not expose backbone noise"
        )
    if ligand["homo_oligomer"]:
        raise DesignConfigError(
            "reward.sequence.ligandmpnn.homo_oligomer is incompatible with the "
            "one-design-chain contract; use explicit symmetry residues"
        )
    _optional_local_path(
        ligand,
        "symmetry_residues_path",
        root,
        "reward.sequence.ligandmpnn",
    )
    _optional_local_path(
        ligand,
        "symmetry_weights_path",
        root,
        "reward.sequence.ligandmpnn",
    )
    if ligand.get("symmetry_weights_path") is not None and ligand.get(
        "symmetry_residues_path"
    ) is None:
        raise DesignConfigError(
            "reward.sequence.ligandmpnn.symmetry_weights_path requires "
            "symmetry_residues_path"
        )
    cyclic = _mapping(sequence.get("cyclicmpnn"), "reward.sequence.cyclicmpnn")
    _validate_mpnn_common(
        cyclic,
        invocation_dir=root,
        where="reward.sequence.cyclicmpnn",
        extra_keys={"base_model"},
        allow_proteinmpnn_constraints=False,
    )
    if not isinstance(cyclic.get("base_model"), str) or not cyclic["base_model"]:
        raise DesignConfigError(
            "reward.sequence.cyclicmpnn.base_model must be a non-empty string"
        )
    if cyclic["model_type"] != "cyclicmpnn":
        raise DesignConfigError(
            "reward.sequence.cyclicmpnn.model_type must be cyclicmpnn"
        )

    if allow_na_mpnn or "na_mpnn" in sequence:
        na = _mapping(sequence.get("na_mpnn"), "reward.sequence.na_mpnn")
        _closed(
            na,
            {"model_type", "checkpoint", "temperature", "seed", "rm_aa", "omit_aas"},
            "reward.sequence.na_mpnn",
        )
        if na.get("model_type") != "na_mpnn":
            raise DesignConfigError(
                "reward.sequence.na_mpnn.model_type must be na_mpnn"
            )
        checkpoint = _resolve_path(
            na.get("checkpoint"), root, "reward.sequence.na_mpnn.checkpoint"
        )
        if checkpoint is None:
            raise DesignConfigError(
                "reward.sequence.na_mpnn.checkpoint must be a local path"
            )
        expected_checkpoint = (
            community_asset("na_mpnn", root) / "checkpoint" / "s_19137.pt"
        ).resolve()
        if allow_na_mpnn and Path(checkpoint) != expected_checkpoint:
            raise DesignConfigError(
                "reward.sequence.na_mpnn.checkpoint must resolve to the pinned "
                f"asset: {expected_checkpoint}"
            )
        na["checkpoint"] = checkpoint
        temperature = _positive_float(
            na.get("temperature"), "reward.sequence.na_mpnn.temperature"
        )
        if temperature > 10:
            raise DesignConfigError(
                "reward.sequence.na_mpnn.temperature must be <= 10"
            )
        _int(na.get("seed"), "reward.sequence.na_mpnn.seed", minimum=0)
        try:
            na["omit_aas"] = resolve_rm_aa(na)
        except ValueError as exc:
            raise DesignConfigError(f"reward.sequence.na_mpnn.{exc}") from exc
        if any(letter not in "ACDEFGHIKLMNPQRSTVWY" for letter in na["omit_aas"]):
            raise DesignConfigError("reward.sequence.na_mpnn.rm_aa must contain protein amino-acid letters")

    fold = _mapping(reward.get("esmfold2"), "reward.esmfold2")
    _closed(
        fold,
        {
            "mode",
            "loops",
            "steps",
            "lm_dropout",
            "lm_mask_pct",
            "attention_backend",
            "context_embedding_cache",
            "timeout_seconds",
            "optimization",
        },
        "reward.esmfold2",
    )
    if fold.get("mode") not in {"fast", "standard"}:
        raise DesignConfigError("reward.esmfold2.mode must be fast or standard")
    _int(fold.get("loops"), "reward.esmfold2.loops", minimum=1)
    _int(fold.get("steps"), "reward.esmfold2.steps", minimum=1)
    _int(
        fold.get("timeout_seconds"),
        "reward.esmfold2.timeout_seconds",
        minimum=1,
    )
    dropout = _nonnegative_float(
        fold.get("lm_dropout"), "reward.esmfold2.lm_dropout"
    )
    if dropout >= 1:
        raise DesignConfigError("reward.esmfold2.lm_dropout must be < 1")
    mask = fold.get("lm_mask_pct")
    if mask is not None:
        mask = _nonnegative_float(mask, "reward.esmfold2.lm_mask_pct")
        if mask > 1:
            raise DesignConfigError("reward.esmfold2.lm_mask_pct must be <= 1")
    if fold.get("attention_backend") not in {"sdpa", "flash_attention_2", "eager"}:
        raise DesignConfigError(
            "reward.esmfold2.attention_backend must be sdpa, flash_attention_2, or eager"
        )
    if fold.get("context_embedding_cache") not in {None, True, False}:
        raise DesignConfigError(
            "reward.esmfold2.context_embedding_cache must be null, true, or false"
        )
    if "optimization" not in fold:
        fold["optimization"] = _esmfold2_optimization_defaults()
    else:
        fold["optimization"] = _validate_esmfold2_optimization(
            fold["optimization"]
        )

    score = _mapping(reward.get("score"), "reward.score")
    _closed(
        score,
        {
            *({"method"} if allow_na_mpnn else set()),
            "monomer_plddt_weight",
            "complex_plddt_weight",
            "complex_iptm_weight",
        },
        "reward.score",
    )
    method = score.get("method", "confidence")
    if not isinstance(method, str) or method not in {"confidence", "partiscore"}:
        raise DesignConfigError("reward.score.method must be confidence or partiscore")
    if method == "partiscore" and structure_source != "fold":
        raise DesignConfigError("reward.score.method=partiscore requires structure_source=fold")
    monomer = _nonnegative_float(
        score.get("monomer_plddt_weight"), "reward.score.monomer_plddt_weight"
    )
    complex_plddt = _nonnegative_float(
        score.get("complex_plddt_weight"), "reward.score.complex_plddt_weight"
    )
    complex_iptm = _nonnegative_float(
        score.get("complex_iptm_weight"), "reward.score.complex_iptm_weight"
    )
    if monomer <= 0 or complex_plddt + complex_iptm <= 0:
        raise DesignConfigError(
            "ESMFold2 monomer and complex reward weights must be positive in total"
        )


def _validate_partiscore_reward(
    reward: Mapping[str, Any], invocation_dir: Path | None = None
) -> None:
    """Validate the lightweight PartiScore reward and its sequence controls.

    PartiScore ranks the candidate structure directly, so it has no fold or
    structure-source block.  The sequence subtree deliberately reuses the
    closed ESMFold2 validator with NA-MPNN disabled; the synthetic fold/score
    values are validator-only defaults and never load fold assets.
    """

    _closed(reward, {"_target_", "sequence"}, "reward")
    sequence = _mapping(reward.get("sequence"), "reward.sequence")
    _validate_esmfold2_reward(
        {
            "_target_": "src.design.rewards.esmfold2.ESMFold2Reward",
            "structure_source": "fold",
            "sequence": sequence,
            "esmfold2": {
                "mode": "fast",
                "loops": 1,
                "steps": 1,
                "lm_dropout": 0.0,
                "lm_mask_pct": None,
                "attention_backend": "sdpa",
                "context_embedding_cache": None,
                "timeout_seconds": 1,
            },
            "score": {
                "monomer_plddt_weight": 1.0,
                "complex_plddt_weight": 0.25,
                "complex_iptm_weight": 0.75,
            },
        },
        invocation_dir,
        allow_na_mpnn=False,
    )
    ligand = _mapping(sequence.get("ligandmpnn"), "reward.sequence.ligandmpnn")
    for key in ("pack_side_chains", "repack_everything"):
        if ligand.get(key) is True:
            raise DesignConfigError(
                f"reward.sequence.ligandmpnn.{key} must be false for PartiScore; "
                "packing is mandatory and controlled internally"
            )


def _validate_af3_reward(reward: Mapping[str, Any], invocation_dir: Path | None = None) -> None:
    _closed(reward, {"_target_", "structure_source", "sequence", "af3", "score"}, "reward")
    # Reuse the proven sequence/fixed-position validators without constructing
    # an ESM model or exposing its parameters as AF3 options.
    _validate_esmfold2_reward({
        "_target_": "src.design.rewards.esmfold2.ESMFold2Reward",
        "structure_source": reward.get("structure_source"),
        "sequence": reward.get("sequence"),
        "esmfold2": {"mode": "fast", "loops": 1, "steps": 1,
                     "lm_dropout": 0.0, "lm_mask_pct": None,
                     "attention_backend": "sdpa", "context_embedding_cache": None,
                     "timeout_seconds": 1},
        "score": {"method": "confidence", "monomer_plddt_weight": 1.0,
                  "complex_plddt_weight": 1.0, "complex_iptm_weight": 1.0},
    }, invocation_dir, allow_na_mpnn=True)
    fold = _mapping(reward.get("af3"), "reward.af3")
    _closed(fold, {"model", "mode", "samples", "recycles", "seed", "attention_backend",
                   "buckets", "timeout_seconds"}, "reward.af3")
    if fold.get("model") not in {"af3", "af3-la"}:
        raise DesignConfigError("reward.af3.model must be af3 or af3-la")
    if fold.get("mode") != "complex":
        raise DesignConfigError("AF3 reward predicts the complete candidate: mode must be complex")
    for key in ("samples", "recycles", "timeout_seconds"):
        _int(fold.get(key), f"reward.af3.{key}", minimum=1)
    if fold["samples"] != 1:
        raise DesignConfigError("AF3 reward requires one diffusion sample per candidate; validation supports multiple samples")
    _int(fold.get("seed"), "reward.af3.seed", minimum=0)
    if fold["seed"] >= 2**32:
        raise DesignConfigError("reward.af3.seed must be a uint32 integer")
    if fold.get("attention_backend") not in {"triton", "cudnn", "xla"}:
        raise DesignConfigError("reward.af3.attention_backend must be triton, cudnn, or xla")
    buckets = fold.get("buckets")
    if (not isinstance(buckets, list) or not buckets
            or any(type(n) is not int or n < 1 for n in buckets)
            or buckets != sorted(set(buckets))):
        raise DesignConfigError("reward.af3.buckets must be strictly increasing positive integers")
    score = _mapping(reward.get("score"), "reward.score")
    _closed(score, {"method", "plddt_weight", "ptm_weight", "iptm_weight"}, "reward.score")
    if score.get("method") not in {"confidence", "partiscore"}:
        raise DesignConfigError("reward.score.method must be confidence or partiscore")
    if score["method"] == "partiscore" and reward.get("structure_source") != "fold":
        raise DesignConfigError("reward.score.method=partiscore requires structure_source=fold")
    weights = {key: _nonnegative_float(score.get(key), f"reward.score.{key}")
               for key in ("plddt_weight", "ptm_weight", "iptm_weight")}
    if weights["plddt_weight"] + weights["ptm_weight"] <= 0:
        raise DesignConfigError("AF3 monomer pLDDT/pTM weights must be positive in total")


def _validate_af2_reward(
    reward: Mapping[str, Any], invocation_dir: Path | None = None
) -> None:
    _closed(
        reward,
        {"_target_", "structure_source", "sequence", "af2", "score"},
        "reward",
    )
    # Reuse the exact closed MPNN, output-source, and score contract.
    _validate_esmfold2_reward(
        {
            "_target_": "src.design.rewards.esmfold2.ESMFold2Reward",
            "structure_source": reward.get("structure_source"),
            "sequence": reward.get("sequence"),
            "esmfold2": {
                "mode": "fast",
                "loops": 1,
                "steps": 1,
                "lm_dropout": 0.0,
                "lm_mask_pct": None,
                "attention_backend": "sdpa",
                "context_embedding_cache": None,
                "timeout_seconds": 1,
            },
            "score": reward.get("score"),
        },
        invocation_dir,
        allow_na_mpnn=False,
    )
    af2 = _mapping(reward.get("af2"), "reward.af2")
    _closed(
        af2,
        {
            "model",
            "model_num",
            "num_recycles",
            "use_initial_guess",
            "msa_policy",
            "timeout_seconds",
        },
        "reward.af2",
    )
    if af2.get("model") not in {"multimer", "ptm"}:
        raise DesignConfigError("reward.af2.model must be multimer or ptm")
    model_num = _int(af2.get("model_num"), "reward.af2.model_num", minimum=1)
    if model_num > 5:
        raise DesignConfigError("reward.af2.model_num must be <= 5")
    _int(af2.get("num_recycles"), "reward.af2.num_recycles", minimum=0)
    _int(af2.get("timeout_seconds"), "reward.af2.timeout_seconds", minimum=1)
    if not isinstance(af2.get("use_initial_guess"), bool):
        raise DesignConfigError("reward.af2.use_initial_guess must be a boolean")
    if af2.get("msa_policy") != "none":
        raise DesignConfigError("reward.af2.msa_policy must be none")


def capture_reward_auto_backend(config, explicit_overlay, overrides=()):
    """Record caller options before inherited vanilla defaults lose provenance.

    Auto selects SolubleMPNN by default. Options explicitly supplied under the
    native proteinmpnn slot apply to that automatic backend, even when their
    values equal a shipped vanilla default. Explicit backend selectors win too.
    """
    sequence = OmegaConf.select(config, "reward.sequence")
    if sequence is None or sequence.get("source") != "auto":
        return
    if sequence.get("auto_proteinmpnn") is not None:
        return
    policy = load_yaml_mapping(_inference_config_dir() / "sequence.yaml")
    effective = copy.deepcopy(policy["solublempnn_defaults"])
    from partidesign.paths import asset_root
    effective["checkpoint"] = str(asset_root() / effective["checkpoint"])
    reward = explicit_overlay.get("reward", {})
    user_sequence = reward.get("sequence", {}) if isinstance(reward, Mapping) else {}
    user_backend = user_sequence.get("proteinmpnn", {}) if isinstance(user_sequence, Mapping) else {}
    if isinstance(user_backend, Mapping):
        effective.update(copy.deepcopy(dict(user_backend)))
    from hydra.core.override_parser.overrides_parser import OverridesParser
    prefix = "reward.sequence.proteinmpnn"
    explicit_keys = set(user_backend) if isinstance(user_backend, Mapping) else set()
    for item in OverridesParser.create().parse_overrides(list(overrides)):
        key = item.key_or_group
        if key == prefix:
            value = OmegaConf.to_container(sequence.proteinmpnn, resolve=True)
            effective.update(value)
            explicit_keys.update(value)
        elif key.startswith(prefix + "."):
            name = key[len(prefix) + 1:]
            effective[name] = item.value()
            explicit_keys.add(name)
    if "model_type" in explicit_keys:
        flags = {"protein_mpnn": (False, False), "soluble_mpnn": (False, True),
                 "ca_protein_mpnn": (True, False)}.get(effective["model_type"])
        if flags is not None:
            for key, value in zip(("ca_only", "use_soluble_model"), flags):
                if key not in explicit_keys:
                    effective[key] = value
        if "checkpoint" not in explicit_keys:
            variant = {"protein_mpnn": "vanilla", "soluble_mpnn": "soluble",
                       "ca_protein_mpnn": "ca"}.get(effective["model_type"])
            if variant is not None:
                effective["checkpoint"] = str(asset_root() / f"community_models/proteinmpnn/checkpoint/{variant}/v_48_020.pt")
    named = user_sequence.get("solublempnn") if isinstance(user_sequence, Mapping) else None
    if isinstance(named, Mapping):
        effective.update(copy.deepcopy(dict(named)))
    with open_dict(sequence):
        sequence.auto_proteinmpnn = effective


def _attach_sequence_policy(data):
    reward = data.get("reward")
    if not isinstance(reward, dict) or reward.get("_target_") is None:
        return
    sequence = reward.get("sequence")
    if not isinstance(sequence, dict):
        return
    policy = load_yaml_mapping(_inference_config_dir() / "sequence.yaml")
    # Shared policy annotations do not belong to the closed runtime schema.
    sequence.pop("auto", None)
    if "solublempnn_defaults" not in sequence:
        from partidesign.paths import asset_root
        defaults = copy.deepcopy(policy.get("solublempnn_defaults", {}))
        if defaults:
            path = Path(str(defaults["checkpoint"]))
            if not path.is_absolute():
                defaults["checkpoint"] = str(asset_root() / path)
            sequence["solublempnn_defaults"] = defaults


def validate_design_config(config: DictConfig, invocation_dir: Path) -> DictConfig:
    try:
        data = OmegaConf.to_container(config, resolve=True)
    except OmegaConfBaseException as exc:
        raise DesignConfigError(f"cannot resolve design config: {exc}") from exc
    if not isinstance(data, dict):
        raise DesignConfigError("resolved design config must be a mapping")
    _closed(
        data,
        {
            "model", "sampling", "tasks", "search", "reward", "task_names", "tasks_file",
            "seed", "batch_size", "num_workers", "pin_memory", "run_name", "output_dir", "output_format",
            "ccd_path", "postprocess", "scorer", "route", "route_ckpt", "rm_token_type",
        },
        "top-level config",
    )
    _attach_sequence_policy(data)
    removal = normalize_rm_token_type(data.get("rm_token_type"))
    data["rm_token_type"] = None if removal is None else list(removal)
    data.setdefault("route", True)
    data.setdefault("route_ckpt", None)
    if type(data["route"]) is not bool:
        raise DesignConfigError("route must be boolean")
    if data["route_ckpt"] is not None and (
        not isinstance(data["route_ckpt"], str)
        or data["route_ckpt"] not in {"omni", "prot", "nuc", "ligand", "antibody", "pep"}
    ):
        raise DesignConfigError("route_ckpt must be null, omni, prot, nuc, ligand, antibody, or pep")
    _int(data.get("seed"), "seed", minimum=0)
    _int(data.get("batch_size"), "batch_size", minimum=1)
    _int(data.get("num_workers"), "num_workers", minimum=0)
    if not isinstance(data.get("pin_memory"), bool):
        raise DesignConfigError("pin_memory must be a boolean")
    if not isinstance(data.get("run_name"), str) or not data["run_name"]:
        raise DesignConfigError("run_name must be a non-empty string")
    if not isinstance(data.get("output_format"), str) or data["output_format"] not in {"cif", "pdb"}:
        raise DesignConfigError("output_format must be cif or pdb")
    task_names = data.get("task_names")
    if task_names is not None:
        if not isinstance(task_names, list) or not task_names:
            raise DesignConfigError("task_names must be null or a non-empty list")
        if not all(isinstance(name, str) and name for name in task_names):
            raise DesignConfigError("task_names entries must be non-empty strings")
        if len(task_names) != len(set(task_names)):
            raise DesignConfigError("task_names contains duplicates")
    tasks = _mapping(data.get("tasks"), "tasks")
    tasks_file = _resolve_path(data.get("tasks_file"), invocation_dir, "tasks_file")
    if tasks_file is not None:
        if tasks:
            raise DesignConfigError("tasks_file is mutually exclusive with packaged tasks")
        task_path = Path(tasks_file)
        if task_path.suffix.lower() != ".yaml":
            raise DesignConfigError("tasks_file must end in .yaml")
        if not task_path.is_file():
            raise DesignConfigError(f"tasks_file does not exist: {task_path}")
        tasks = load_yaml_mapping(task_path)
        data["tasks"] = tasks
    elif not tasks:
        raise DesignConfigError("no tasks supplied; select a packaged group or tasks_file")
    data["tasks_file"] = tasks_file
    data["output_dir"] = _resolve_path(data.get("output_dir"), invocation_dir, "output_dir")
    data["ccd_path"] = _resolve_path(data.get("ccd_path"), invocation_dir, "ccd_path")
    _validate_model(_mapping(data.get("model"), "model"), invocation_dir)
    validate_sampling_preflight(_mapping(data.get("sampling"), "sampling"))
    _validate_search_reward(
        _mapping(data.get("search"), "search"),
        _mapping(data.get("reward"), "reward"),
        invocation_dir=invocation_dir,
    )
    selected = data.get("task_names") or list(tasks)
    needs_scorer = any(
        isinstance(tasks.get(name), Mapping) and tasks[name].get("score") is True
        for name in selected
    ) or data["reward"].get("_target_") == "src.design.rewards.partiscore.PartiScoreReward" or (
        data["reward"].get("_target_") in {"src.design.rewards.esmfold2.ESMFold2Reward", "src.design.rewards.af3.AF3Reward"}
        and data["reward"].get("score", {}).get("method", "confidence") == "partiscore"
    )
    if needs_scorer:
        scorer = _mapping(data.get("scorer"), "scorer")
        _closed(scorer, {"checkpoint", "weights", "device", "batch_size", "max_pair_elements", "num_workers"}, "scorer")
        scorer["checkpoint"] = _resolve_path(scorer.get("checkpoint"), invocation_dir, "scorer.checkpoint")
        if scorer["checkpoint"] is None:
            raise DesignConfigError("scorer.checkpoint is required when PartiScore is enabled")
        if not isinstance(scorer.get("weights"), str) or scorer.get("weights") not in {"raw", "ema"}:
            raise DesignConfigError("scorer.weights must be raw or ema")
        if not isinstance(scorer.get("device"), str) or not scorer["device"]:
            raise DesignConfigError("scorer.device must be a non-empty string")
        _int(scorer.get("batch_size"), "scorer.batch_size", minimum=1)
        _int(scorer.get("max_pair_elements"), "scorer.max_pair_elements", minimum=1)
        _int(scorer.get("num_workers", 16), "scorer.num_workers", minimum=0)
    post = _mapping(data.get("postprocess"), "postprocess")
    _closed(
        post,
        {"force_field", "charge_method", "platform", "use_ray"},
        "postprocess",
    )
    if post.get("force_field") != "amber19sb":
        raise DesignConfigError("postprocess.force_field must be amber19sb")
    charge_method = post.get("charge_method")
    if not isinstance(charge_method, str) or charge_method not in {
        "gasteiger",
        "am1bcc",
    }:
        raise DesignConfigError(
            "postprocess.charge_method must be gasteiger or am1bcc"
        )
    platform = post.get("platform")
    if (
        not isinstance(platform, str)
        or platform not in {"CUDA", "CPU"}
        or not isinstance(post.get("use_ray"), bool)
    ):
        raise DesignConfigError("invalid postprocess platform/use_ray")
    selected_postprocess = any(
        isinstance(tasks.get(name), Mapping)
        and tasks[name].get("postprocess", False) is True
        and tasks[name].get("override", True) is True
        for name in selected
    )
    if selected_postprocess:
        if data["output_format"] != "cif":
            raise DesignConfigError("enabled postprocessing requires CIF output")
        if (platform, post["use_ray"]) not in {("CUDA", True), ("CPU", False)}:
            raise DesignConfigError(
                "enabled postprocessing accepts only CUDA+Ray or CPU+no-Ray"
            )
    return OmegaConf.create(data)


_REWARD_OMISSION_BACKENDS = ("proteinmpnn", "ligandmpnn", "cyclicmpnn", "na_mpnn")


def normalize_reward_rm_aa_overlay(overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Keep explicit legacy settings above the new stock Cys-removal default."""
    result = copy.deepcopy(dict(overlay))
    reward = result.get("reward", {})
    sequence = reward.get("sequence", {}) if isinstance(reward, Mapping) else {}
    if isinstance(sequence, Mapping):
        for name in _REWARD_OMISSION_BACKENDS:
            backend = sequence.get(name)
            if isinstance(backend, dict) and "omit_aas" in backend and "rm_aa" not in backend:
                backend["rm_aa"] = None
    return result


def normalize_reward_rm_aa_overrides(overrides: Sequence[str]) -> list[str]:
    """An explicit legacy CLI field clears an inherited canonical default."""
    result = list(overrides)
    keys = {item.split("=", 1)[0].lstrip("+") for item in overrides}
    for name in _REWARD_OMISSION_BACKENDS:
        base = f"reward.sequence.{name}"
        if base + ".omit_aas" in keys and base + ".rm_aa" not in keys:
            result.append(base + ".rm_aa=null")
    return result


def compose_design_config(
    config: str | Path | Mapping[str, Any] | DictConfig | None = None,
    *,
    overrides: Sequence[str] = (),
    invocation_dir: str | Path | None = None,
) -> LoadedDesignConfig:
    base_dir = Path.cwd() if invocation_dir is None else Path(invocation_dir)
    base_dir = base_dir.expanduser().resolve()
    try:
        with initialize_config_dir(
            config_dir=str(_inference_config_dir()), version_base=None
        ):
            composed = compose(config_name="design", overrides=normalize_reward_rm_aa_overrides(overrides))
    except (HydraException, OmegaConfBaseException) as exc:
        raise DesignConfigError(f"invalid Hydra config or override: {exc}") from exc
    overlay = {}
    if config is not None:
        if isinstance(config, (str, Path)):
            config_path = Path(config).expanduser()
            if not config_path.is_absolute():
                config_path = base_dir / config_path
            overlay = load_yaml_mapping(config_path.resolve())
        elif isinstance(config, DictConfig):
            overlay = OmegaConf.to_container(config, resolve=False)
        elif isinstance(config, Mapping):
            overlay = dict(config)
        else:
            raise DesignConfigError(f"unsupported config input: {type(config).__name__}")
        overlay = normalize_reward_rm_aa_overlay(overlay)
        if "defaults" in overlay:
            raise DesignConfigError("external resolved configs may not contain Hydra defaults")
        try:
            # Diagnostic overrides are supported without putting optimization
            # controls in ordinary YAML. Only these known optional sections
            # are opened; the usual closed validation still checks their keys.
            model_overlay = overlay.get("model", {})
            if isinstance(model_overlay, Mapping) and "optimization" in model_overlay:
                from ..model.optimization import DEFAULT_OPTIMIZATION
                with open_dict(composed.model):
                    composed.model.optimization = dict(DEFAULT_OPTIMIZATION)
            reward_overlay = overlay.get("reward", {})
            fold_overlay = reward_overlay.get("esmfold2", {}) if isinstance(reward_overlay, Mapping) else {}
            if isinstance(fold_overlay, Mapping) and "optimization" in fold_overlay and "esmfold2" in composed.reward:
                with open_dict(composed.reward.esmfold2):
                    composed.reward.esmfold2.optimization = _esmfold2_optimization_defaults()
                    OmegaConf.set_struct(composed.reward.esmfold2.optimization, False)
            # External mappings are checked by the closed native validator;
            # Hydra struct mode must allow task names and optional backend blocks.
            OmegaConf.set_struct(composed, False)
            composed = OmegaConf.merge(composed, OmegaConf.create(overlay))
        except OmegaConfBaseException as exc:
            raise DesignConfigError(f"invalid config overlay: {exc}") from exc
    capture_reward_auto_backend(composed, overlay, normalize_reward_rm_aa_overrides(overrides))
    return LoadedDesignConfig(validate_design_config(composed, base_dir), base_dir)
