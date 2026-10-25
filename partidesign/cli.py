"""Local PartiDesign commands; model imports happen only on dispatch."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import sys
from pathlib import Path

from . import __version__
from ._lightning_notices import configure_lightning_notices
from .paths import model_dir, using_model_dir


class CLIParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        from .ui import print_help
        print_help(self, file)


class SingleValue(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        seen = getattr(namespace, "_single_options", set())
        if self.dest in seen:
            parser.error(f"{option_string} may be supplied only once")
        setattr(namespace, "_single_options", seen | {self.dest})
        setattr(namespace, self.dest, values)


def _single(parser, *names, **kwargs):
    parser.add_argument(*names, action=SingleValue, **kwargs)


def _model_location(parser):
    _single(parser, "--model-dir", type=Path,
            help="Local model root: flag > PARTIDESIGN_MODEL_DIR > PartiDesign repository root")


def _task_options(parser, *, generation=False):
    _single(parser, "--config", type=Path, help="Closed main configuration overlay (.yaml)")
    parser.add_argument("--task", action="append", help="Select a task name (repeatable)")
    _single(parser, "--ckpt", type=Path, help="Explicit combined checkpoint or five/six-member bundle")
    _single(parser, "--route", choices=("true", "false"), help="Use checkpoint routing (default: configured true)")
    _single(parser, "--route-ckpt", choices=("null", "omni", "prot", "nuc", "ligand", "antibody", "pep"),
            help="Bundle selector (pep requires a version-2 six-member bundle); null means automatic selection")
    _single(parser, "--execution-backend", choices=("auto", "native", "bioir"),
            help="FM execution backend (default: auto); BioIR requires its separate CUDA 13 environment")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Existing Hydra configuration override (repeatable)")
    if generation:
        _single(parser, "--out", type=Path, help="Output directory; otherwise ./inference/RUN_NAME")
        _single(parser, "--run-name", help="Run label (default: configured design)")
        _single(parser, "--seed", type=int, help="Nonnegative global seed (default: configured 2026)")
        _single(parser, "--batch-size", type=int, help="Positive FM batch size (default: configured 16)")
        _single(parser, "--num-workers", type=int, help="Nonnegative data workers (default: configured 4)")
        _single(parser, "--compile", choices=("auto", "true", "false"),
                help="Compile FM trunk (default: auto); sampling remains 400 steps")
        _single(parser, "--compile-strategy", choices=("auto", "dynamic", "bucket"),
                help="Automatic execution policy (default), dynamic compilation, or static token buckets")
        parser.add_argument("--show-config", action="store_true", help="Print resolved configuration without generation")


def _msa_options(parser):
    """Add the lightweight AF3 MSA policy flags shared by validate/check."""
    _single(parser, "--msa", choices=("auto", "off", "cache-only"),
            help="AF3 target/context MSA policy")
    _single(parser, "--msa-cache-dir", help="AF3 MSA cache directory")
    _single(parser, "--msa-profile", help="Recorded AF3 MSA cache profile")
    _single(parser, "--msa-database-dir", help="AF3 public database root")
    _single(parser, "--msa-binaries-dir", help="Directory containing jackhmmer/nhmmer")
    _single(parser, "--msa-workers", type=int,
            help="Concurrent AF3 MSA searches (positive integer)")
    _single(parser, "--msa-cpus",
            help="AF3 MSA CPU budget: auto or a positive integer")
    _single(parser, "--jackhmmer-cpus", type=int,
            help="Threads per AF3 jackhmmer process")
    _single(parser, "--nhmmer-cpus", type=int,
            help="Threads per AF3 nhmmer process")
    _single(parser, "--msa-timeout", type=int,
            help="AF3 MSA search timeout in seconds")
    parser.add_argument("--msa-refresh", action="store_true",
                        help="Refresh matching AF3 MSA cache generations")
    _single(parser, "--msa-templates", choices=("true", "false"),
            help="AF3 native structure-template search: true or false")
    _single(parser, "--msa-template-date",
            help="AF3 maximum template date (YYYY-MM-DD)")


def build_parser() -> argparse.ArgumentParser:
    from .download import TARGETS
    parser = CLIParser(prog="partidesign", description="Local multimodal and antibody design")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Choose model weights interactively, or download explicit targets",
                                   description="Open a model-selection dialog when no targets are given in a terminal.")
    download.add_argument("targets", nargs="*", choices=TARGETS, metavar="TARGET",
                          help="Resource targets; omit to select interactively")
    _model_location(download)
    download.add_argument("--all", action="store_true", help="Prepare every registered target")
    download.add_argument("--list", action="store_true", help="List local status and sources without network")
    download.add_argument("--force", action="store_true", help="Replace only after a complete new download succeeds")
    design = commands.add_parser("design", help="Generate normal or antibody tasks from one task YAML")
    design.add_argument("task_file", type=Path, metavar="TASK_FILE.yaml")
    _model_location(design)
    _task_options(design, generation=True)
    check = commands.add_parser("check", help="Read-only checks, optional CPU loading or explicit real smoke")
    check.add_argument("task_file", nargs="?", type=Path, metavar="TASK_FILE.yaml")
    _model_location(check)
    _task_options(check)
    choice = check.add_mutually_exclusive_group()
    choice.add_argument("--model", action="append", choices=TARGETS, help="Check a resource target (repeatable)")
    choice.add_argument("--all", action="store_true", help="Check all registered resource targets")
    check.add_argument("--load", action="store_true", help="Strict-load requested models on CPU")
    check.add_argument("--smoke", action="store_true", help="Run bounded real models and read back results (CUDA for GPU backends)")
    _single(check, "--out", type=Path, help="Keep smoke outputs here; otherwise use a temporary directory")
    _msa_options(check)

    relax = commands.add_parser("relax", help="Retry OpenMM for existing inference samples without generation")
    relax.add_argument("task_file", type=Path, metavar="TASK_FILE.yaml")
    _model_location(relax)
    _single(relax, "--config", type=Path, help="Closed main configuration overlay (.yaml)")
    relax.add_argument("--task", action="append", help="Select a task name (repeatable)")
    _single(relax, "--out", type=Path, required=True, help="Existing inference run root containing task result.csv files")
    _single(relax, "--limit", type=int, help="Positive maximum retry samples per task")
    _single(relax, "--platform", choices=("CUDA", "CPU"), help="CUDA uses all visible GPUs; CPU is a sequential diagnostic")
    _single(relax, "--charge-method", choices=("gasteiger", "am1bcc"))
    relax.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                       help="Existing Hydra configuration override (repeatable)")
    relax.add_argument("--rerun", action="store_true", help="Also retry completed samples and replace owned refined staging files")
    relax.add_argument("--plan", action="store_true", help="Count selected existing samples without relaxation")

    score = commands.add_parser("score", help="Score direct-child PDB/CIF files into per-directory score.csv")
    score.add_argument("input_dirs", nargs="+", type=Path, metavar="INPUT_DIR")
    _model_location(score)
    _single(score, "--config", type=Path, help="Closed score configuration overlay (.yaml)")
    _single(score, "--ckpt", type=Path, help="Explicit PartiScore checkpoint")
    _single(score, "--device", help="Scoring device (default: configured auto)")
    _single(score, "--batch-size", type=int, help="Scorer inference batch size")
    _single(score, "--num-workers", type=int, help="Scorer structure-preparation workers")
    _single(score, "--overwrite", choices=("true", "false"),
            help="Replace existing score.csv files (default: configured false)")
    score.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                       help="Typed score configuration override (repeatable)")
    score.add_argument("--show-config", action="store_true",
                       help="Print resolved scoring configuration without scoring")

    cluster = commands.add_parser("cluster", help="Cluster successful original design polymers on CPU")
    from src.cluster.cli import add_arguments as cluster_arguments
    cluster_arguments(cluster)

    postfilter = commands.add_parser("postfilter", help="Relax and filter successful protein complex refolds")
    from src.postfilter.cli import add_arguments as postfilter_arguments
    postfilter_arguments(postfilter)

    # Constants and configuration parsing in this package do not import models.
    from src.validate.config import METHODS
    validate = commands.add_parser("validate", help="Validate existing run/task result.csv files")
    validate.add_argument("runs", nargs="+", metavar="RUN_OR_TASK_PATH")
    _model_location(validate)
    _single(validate, "--config", type=Path)
    validate.add_argument("--task", action="append", help="Select task names (repeatable)")
    validate.add_argument("--method", action="append", choices=METHODS, help="Validation methods (repeatable)")
    validate.add_argument("--source", action="append", help="SOURCE:COUNT slots; gt or proteinmpnn/solublempnn/ligandmpnn/cyclicmpnn/nampnn, each also accepts -fi (6 A); repeatable; fold default gt:1+solublempnn:1, nucleic sequence-design default gt:1+nampnn:1")
    validate.add_argument("--criteria", action="append", help="Filter preset names (repeatable; default none)")
    _single(validate, "--presets", type=Path)
    _single(validate, "--worker-gpus", type=float, help="GPU fraction per worker in (0,1]")
    _single(validate, "--worker-cpus", type=float, help="Positive CPU allocation per worker")
    maintenance = validate.add_mutually_exclusive_group()
    maintenance.add_argument("--refilter", action="store_true", help="Rewrite reports from persisted metrics, without models")
    maintenance.add_argument("--recompute-metrics", action="store_true", help="Rewrite metrics from persisted structures, without models")
    maintenance.add_argument("--plan", action="store_true", help="Print lightweight plan without writing or model work")
    validate.add_argument("--score", action="store_true", help="PartiScore originals and successful refolds")
    rerun = validate.add_mutually_exclusive_group()
    rerun.add_argument("--rerun", action="store_true", help="Regenerate requested sequence slots")
    rerun.add_argument("--rerun-refold", action="store_true",
                       help="AF3/AF3-LA only: reuse saved sequence slots and regenerate refolds")
    _msa_options(validate)
    _single(validate, "--limit", type=int, help="Positive maximum sample count per task")
    return parser


def _absolute(path: Path, cwd: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else cwd / path).resolve()


_SCORE_CONFIG_KEYS = frozenset({"input_dirs", "overwrite", "scorer"})
_SCORE_SCORER_KEYS = frozenset({
    "checkpoint", "weights", "device", "batch_size", "max_pair_elements", "num_workers",
})


def _load_score_yaml(path: Path) -> dict:
    """Load one score mapping with duplicate and root-key checks deferred."""
    import yaml

    class _UniqueKeyLoader(yaml.SafeLoader):
        pass

    def _mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"duplicate YAML key: {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping,
    )
    path = Path(path)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"score configuration must be YAML: {path}")
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"cannot read score configuration {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"score configuration root must be a mapping: {path}")
    if not all(isinstance(key, str) for key in payload):
        raise ValueError(f"score configuration keys must be strings: {path}")
    return dict(payload)


def _merge_score_mapping(base: Mapping, overlay: Mapping) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_score_mapping(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_score_mapping(config: Mapping) -> None:
    unknown = sorted(set(config).difference(_SCORE_CONFIG_KEYS))
    if unknown:
        raise ValueError(f"unknown score configuration fields: {unknown}")
    scorer = config.get("scorer")
    if not isinstance(scorer, Mapping):
        raise ValueError("score configuration scorer must be a mapping")
    unknown = sorted(set(scorer).difference(_SCORE_SCORER_KEYS))
    if unknown:
        raise ValueError(f"unknown score scorer fields: {unknown}")
    checkpoint = scorer.get("checkpoint")
    if not isinstance(checkpoint, (str, Path)) or not str(checkpoint).strip():
        raise ValueError("score scorer.checkpoint must be a nonempty path")
    if not isinstance(scorer.get("weights"), str) or not scorer["weights"].strip():
        raise ValueError("score scorer.weights must be a nonempty string")
    if not isinstance(scorer.get("device"), str) or not scorer["device"].strip():
        raise ValueError("score scorer.device must be a nonempty string")
    for key, minimum in (("batch_size", 1), ("max_pair_elements", 1), ("num_workers", 0)):
        value = scorer.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"score scorer.{key} must be an integer >= {minimum}")
    if not isinstance(config.get("overwrite"), bool):
        raise ValueError("score overwrite must be boolean")
    input_dirs = config.get("input_dirs")
    if isinstance(input_dirs, (str, Path)):
        input_dirs = [input_dirs]
    if not isinstance(input_dirs, Sequence) or isinstance(input_dirs, (str, bytes)):
        raise ValueError("score input_dirs must be a list of directories")
    if not all(isinstance(value, (str, Path)) and str(value).strip() for value in input_dirs):
        raise ValueError("score input_dirs entries must be nonempty paths")


def _score_path_list(values, cwd: Path) -> list[str]:
    if isinstance(values, (str, Path)):
        values = [values]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("score input_dirs must be a list of directories")
    return [str(_absolute(Path(value), cwd)) for value in values]


def _score_override_values(args, *, config_dir: Path, explicit_fields: set[str]):
    """Parse typed score overrides and return (updates, keys)."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    try:
        parsed = OverridesParser.create().parse_overrides(args.override)
    except Exception as error:
        raise ValueError(f"invalid score override: {error}") from error
    keys = [item.key_or_group for item in parsed]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate --override configuration key")
    updates = []
    for item in parsed:
        key = item.key_or_group
        if item.is_sweep_override() or item.is_delete():
            raise ValueError("score sweep/delete overrides are not supported")
        if any(key == field or key.startswith(field + ".") or field.startswith(key + ".")
               for field in explicit_fields):
            raise ValueError(f"--override {key!r} conflicts with an explicit CLI parameter")
        value = item.value()
        if key == "scorer":
            if not isinstance(value, str) or not value:
                raise ValueError("score scorer override must select a named scorer group")
            group = config_dir / "scorer" / f"{value}.yaml"
            if not group.is_file():
                raise ValueError(f"unknown score scorer group: {value}")
            selected = _load_score_yaml(group)
            updates.append((key, selected))
        elif key in {"input_dirs", "overwrite"}:
            updates.append((key, value))
        elif key.startswith("scorer."):
            field = key.split(".", 1)[1]
            if field not in _SCORE_SCORER_KEYS:
                raise ValueError(f"unknown score override field: {key}")
            updates.append((key, value))
        else:
            raise ValueError(f"unknown score override field: {key}")
    return updates, keys


def build_score_config(args, root: Path, *, invocation_dir: Path | None = None) -> dict:
    """Resolve the standalone score contract before calling ``src.scoring``."""
    from .paths import inference_config_dir

    root = Path(root).expanduser().resolve()
    cwd = Path.cwd().resolve() if invocation_dir is None else Path(invocation_dir).resolve()
    config_dir = inference_config_dir()
    default_path = config_dir / "score.yaml"
    config = _load_score_yaml(default_path)
    overlay = {}
    if args.config is not None:
        config_path = _absolute(args.config, cwd)
        overlay = _load_score_yaml(config_path)
        config = _merge_score_mapping(config, overlay)
    if not isinstance(overlay, Mapping):
        raise ValueError("score configuration overlay must be a mapping")
    unknown = sorted(set(overlay).difference(_SCORE_CONFIG_KEYS))
    if unknown:
        raise ValueError(f"unknown score configuration fields: {unknown}")
    overlay_scorer = overlay.get("scorer", {})
    if overlay_scorer is not None and not isinstance(overlay_scorer, Mapping):
        raise ValueError("score configuration scorer must be a mapping")
    if isinstance(overlay_scorer, Mapping):
        unknown = sorted(set(overlay_scorer).difference(_SCORE_SCORER_KEYS))
        if unknown:
            raise ValueError(f"unknown score scorer fields: {unknown}")

    explicit_fields = {"input_dirs"}
    if args.overwrite is not None:
        explicit_fields.add("overwrite")
    if args.ckpt is not None:
        explicit_fields.add("scorer.checkpoint")
    if args.device is not None:
        explicit_fields.add("scorer.device")
    if args.batch_size is not None:
        explicit_fields.add("scorer.batch_size")
    if args.num_workers is not None:
        explicit_fields.add("scorer.num_workers")
    updates, override_keys = _score_override_values(
        args, config_dir=config_dir, explicit_fields=explicit_fields,
    )
    checkpoint_override = "scorer.checkpoint" in override_keys
    scorer_group_override = "scorer" in override_keys
    for key, value in updates:
        if key == "scorer":
            config["scorer"] = dict(value)
        elif key == "input_dirs":
            config[key] = value
        elif key == "overwrite":
            config[key] = value
        else:
            config["scorer"][key.split(".", 1)[1]] = value

    # Positional directories are the command's explicit input selection.
    if args.input_dirs:
        config["input_dirs"] = list(args.input_dirs)
    if args.overwrite is not None:
        config["overwrite"] = args.overwrite == "true"
    if args.ckpt is not None:
        config["scorer"]["checkpoint"] = str(_absolute(args.ckpt, cwd))
    if args.device is not None:
        config["scorer"]["device"] = args.device
    if args.batch_size is not None:
        config["scorer"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["scorer"]["num_workers"] = args.num_workers

    # Validate raw scalar types before normalizing paths; otherwise a null or
    # numeric checkpoint could be stringified into an apparently valid path.
    _validate_score_mapping(config)
    config["input_dirs"] = _score_path_list(config.get("input_dirs"), cwd)
    checkpoint = Path(str(config["scorer"]["checkpoint"])).expanduser()
    overlay_checkpoint = isinstance(overlay_scorer, Mapping) and "checkpoint" in overlay_scorer
    if args.ckpt is None and (scorer_group_override or (
            not checkpoint_override and not overlay_checkpoint)) and not checkpoint.is_absolute():
        config["scorer"]["checkpoint"] = str(root / checkpoint)
    elif args.ckpt is None and (checkpoint_override or overlay_checkpoint) and not checkpoint.is_absolute():
        config["scorer"]["checkpoint"] = str(_absolute(checkpoint, cwd))
    else:
        config["scorer"]["checkpoint"] = str(checkpoint)
    _validate_score_mapping(config)
    if not config["input_dirs"]:
        raise ValueError("score requires at least one input directory")
    return config


def build_design_runner(args, root: Path, *, invocation_dir: Path | None = None):
    """Normalize the new frontend without changing the legacy compose priority."""
    from hydra import compose, initialize_config_dir
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from hydra.errors import HydraException
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException
    from src.design.core.api import DesignRunner
    from src.design.core.config import load_yaml_mapping, normalize_reward_rm_aa_overlay, capture_reward_auto_backend
    from src.design.core.records import DesignConfigError
    from src.design.core.compile_policy import DEFAULT_TOKEN_BUCKETS, validate_compile_enabled
    from .paths import inference_config_dir

    cwd = Path.cwd().resolve() if invocation_dir is None else invocation_dir.resolve()
    fields = {"tasks_file": str(_absolute(args.task_file, cwd))}
    mapping = {"ckpt": "model.checkpoint", "out": "output_dir", "run_name": "run_name",
               "seed": "seed", "batch_size": "batch_size", "num_workers": "num_workers",
               "route": "route", "route_ckpt": "route_ckpt", "compile": "model.compile.enabled",
               "execution_backend": "model.execution_backend"}
    for argument, key in mapping.items():
        value = getattr(args, argument, None)
        if value is None:
            continue
        if argument in {"ckpt", "out"}:
            value = str(_absolute(value, cwd))
        elif argument == "route" or (argument == "compile" and value != "auto"):
            value = value == "true"
        elif argument == "route_ckpt" and value == "null":
            value = None
        fields[key] = value
    if args.task is not None:
        fields["task_names"] = args.task
    strategy = getattr(args, "compile_strategy", None)
    if strategy == "auto":
        if getattr(args, "compile", None) not in {None, "auto"}:
            raise DesignConfigError("--compile-strategy auto conflicts with --compile true or false")
        fields["model.compile.enabled"] = "auto"
    if strategy is not None:
        fields["model.compile.dynamic"] = strategy != "bucket"
    try:
        parsed = OverridesParser.create().parse_overrides(args.override)
        keys = [item.key_or_group for item in parsed]
        if len(keys) != len(set(keys)):
            raise DesignConfigError("duplicate --override configuration key")
        for key in keys:
            if any(key == field or key.startswith(field + ".") or field.startswith(key + ".")
                   for field in fields):
                raise DesignConfigError(f"--override {key!r} conflicts with an explicit CLI parameter")
        for item in parsed:
            if item.is_sweep_override() or item.is_delete():
                raise DesignConfigError("sweep/delete overrides are not supported by the local CLI")
            if (strategy in {"auto", "dynamic"} and item.key_or_group == "model.compile.token_buckets"
                    and item.value() is not None):
                raise DesignConfigError(f"--compile-strategy {strategy} conflicts with a token_buckets override")
        config_dir = inference_config_dir()
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            base = compose(config_name="design")
            selected = compose(config_name="design", overrides=args.override)
        overlay = {} if args.config is None else load_yaml_mapping(_absolute(args.config, cwd))
        overlay = normalize_reward_rm_aa_overlay(overlay)
        if "defaults" in overlay:
            raise DesignConfigError("external configuration overlays may not contain Hydra defaults")
        OmegaConf.set_struct(base, False)
        base = OmegaConf.merge(base, OmegaConf.create(overlay))
        for item in parsed:
            key = item.key_or_group
            # Group selection replaces that group; scalar overrides are applied last.
            value = OmegaConf.select(selected, key) if (config_dir / key).is_dir() else item.value()
            OmegaConf.update(base, key, value, merge=False, force_add=True)
            if key.startswith("reward.sequence.") and key.endswith(".omit_aas"):
                canonical = key.rsplit(".", 1)[0] + ".rm_aa"
                if canonical not in keys:
                    OmegaConf.update(base, canonical, None, merge=False, force_add=True)
        if not OmegaConf.is_dict(base.model):
            raise DesignConfigError("model must be a configuration mapping; use --ckpt and --route-ckpt to select weights")
        stock_model_override = "model" in keys and (config_dir / "model").is_dir()
        explicit_checkpoint = (
            getattr(args, "ckpt", None) is not None
            or "model.checkpoint" in keys
            or (not stock_model_override and isinstance(overlay.get("model"), dict)
                and "checkpoint" in overlay["model"])
        )
        if not explicit_checkpoint:
            stock = Path(str(base.model.checkpoint))
            OmegaConf.update(base, "model.checkpoint", str(stock if stock.is_absolute() else root / stock))
        explicit_scorer = (
            isinstance(overlay.get("scorer"), dict) and "checkpoint" in overlay["scorer"]
            or any(key == "scorer" or key == "scorer.checkpoint" for key in keys)
        )
        if not explicit_scorer:
            OmegaConf.update(base, "scorer.checkpoint", str(root / "ckpts" / "partiscore.ckpt"))
        # Only stock checkpoint defaults move to the model root. Explicit user
        # paths retain invocation-directory semantics in the native validator.
        # ProteinMPNN has two stock variants; choose that variant from the
        # selected backend fields instead of assuming the soluble weights.
        def stock_relative_checkpoint(backend, backend_config):
            if backend == "proteinmpnn":
                model_type = str(backend_config.get("model_type") or "")
                use_soluble = backend_config.get("use_soluble_model", False)
                if not isinstance(use_soluble, bool):
                    raise DesignConfigError(
                        "reward.sequence.proteinmpnn.use_soluble_model must be boolean"
                    )
                if model_type in {"soluble_mpnn", "solublempnn"} or use_soluble:
                    variant = "soluble"
                elif model_type == "ca_protein_mpnn":
                    variant = "ca"
                elif model_type in {"protein_mpnn", "proteinmpnn", ""}:
                    variant = "vanilla"
                else:
                    raise DesignConfigError(
                        "unsupported stock protein sequence model_type: " + model_type
                    )
                return f"community_models/proteinmpnn/checkpoint/{variant}/v_48_020.pt"
            return {
                "ligandmpnn": "community_models/ligandmpnn/checkpoint/ligandmpnn_v_32_010_25.pt",
                "cyclicmpnn": "community_models/cyclicmpnn/checkpoint/cyclicmpnn_48_010.pkl",
                "na_mpnn": "community_models/na_mpnn/checkpoint/s_19137.pt",
            }[backend]

        relative_defaults = {
            "proteinmpnn": None,
            "ligandmpnn": "community_models/ligandmpnn/checkpoint/ligandmpnn_v_32_010_25.pt",
            "cyclicmpnn": "community_models/cyclicmpnn/checkpoint/cyclicmpnn_48_010.pkl",
            "na_mpnn": "community_models/na_mpnn/checkpoint/s_19137.pt",
        }
        reward_overlay = overlay.get("reward", {})
        sequence_overlay = reward_overlay.get("sequence", {}) if isinstance(reward_overlay, dict) else {}
        for backend, relative in relative_defaults.items():
            key = f"reward.sequence.{backend}.checkpoint"
            value = OmegaConf.select(base, key)
            user_backend = sequence_overlay.get(backend, {}) if isinstance(sequence_overlay, dict) else {}
            backend_prefix = f"reward.sequence.{backend}"
            # A checkpoint named in an external overlay remains explicit even
            # when a reward group is selected on the command line.
            explicit = isinstance(user_backend, dict) and "checkpoint" in user_backend
            explicit |= backend_prefix in keys or key in keys
            if value is not None and not explicit:
                # A stock group override is still a default, not a user path.
                path = Path(str(value))
                if not path.is_absolute() and str(path).startswith("community_models/"):
                    relative = stock_relative_checkpoint(
                        backend,
                        OmegaConf.to_container(
                            OmegaConf.select(base, f"reward.sequence.{backend}"),
                            resolve=True,
                        ) or {},
                    )
                    path = Path(relative)
                    OmegaConf.update(base, key, str(root / path))
        if strategy is not None:
            buckets = OmegaConf.select(base, "model.compile.token_buckets")
            fields["model.compile.token_buckets"] = (
                None if strategy != "bucket" else
                list(DEFAULT_TOKEN_BUCKETS) if buckets is None else buckets
            )
            if strategy != "auto":
                enabled = validate_compile_enabled(fields.get(
                    "model.compile.enabled", OmegaConf.select(base, "model.compile.enabled"),
                ))
                fields["model.compile.enabled"] = enabled is not False
        for key, value in fields.items():
            OmegaConf.update(base, key, value, merge=False, force_add=True)
        if (strategy in {"dynamic", "bucket"}
                and OmegaConf.select(base, "model.execution_backend") == "bioir"):
            raise DesignConfigError(
                "BioIR execution conflicts with --compile-strategy dynamic or bucket"
            )
        capture_reward_auto_backend(base, overlay, args.override)
        # Strategy consumes the bucket override above; do not reapply a raw null
        # after selecting the bucket defaults or resetting the dynamic policy.
        native_overrides = [value for value in args.override
                            if value.split("=", 1)[0].lstrip("+") != "reward"
                            and not (strategy is not None and
                                     value.split("=", 1)[0].lstrip("+") in {
                                         "model.compile.token_buckets", "model.compile.enabled",
                                     })]
        reward_presets = {
            "src.design.rewards.clash.ClashReward": "clash",
            "src.design.rewards.partiscore.PartiScoreReward": "partiscore",
            "src.design.rewards.esmfold2.ESMFold2Reward": "esmfold2",
            "src.design.rewards.af3.AF3Reward": str(OmegaConf.select(base, "reward.af3.model") or "af3-la"),
            "src.design.rewards.af2.AF2Reward": "af2",
        }
        target = OmegaConf.select(base, "reward._target_")
        if target in reward_presets:
            native_overrides.insert(0, "reward=" + reward_presets[target])
        from .paths import using_model_dir
        with using_model_dir(root):
            return DesignRunner(config=base, overrides=native_overrides, invocation_dir=cwd)
    except DesignConfigError:
        raise
    except (HydraException, OmegaConfBaseException, ValueError, TypeError) as error:
        raise DesignConfigError(f"invalid CLI configuration: {error}") from error


def _validate_arguments(args, parser):
    if args.command == "download":
        if args.all and args.targets:
            parser.error("download --all and explicit targets are mutually exclusive")
        if args.list and args.force:
            parser.error("download --list cannot be combined with --force")
        if len(args.targets) != len(set(args.targets)):
            parser.error("duplicate download target")
    if args.command == "check":
        if args.out is not None and not args.smoke:
            parser.error("check --out requires --smoke")
        if args.task_file is not None and (args.model or args.all):
            parser.error("task checks cannot be combined with --model or --all")
        if args.task_file is None and (
            args.config is not None or args.task is not None or args.ckpt is not None
            or args.route is not None or args.route_ckpt is not None or args.override
            or args.execution_backend is not None
        ):
            parser.error("task-specific check options require TASK_FILE.yaml")
        if args.model and len(args.model) != len(set(args.model)):
            parser.error("duplicate check model")
        msa_values = (
            args.msa, args.msa_cache_dir, args.msa_profile, args.msa_database_dir,
            args.msa_binaries_dir, args.msa_workers, args.msa_cpus,
            args.jackhmmer_cpus, args.nhmmer_cpus, args.msa_timeout,
            args.msa_templates, args.msa_template_date,
        )
        if (any(value is not None for value in msa_values) or args.msa_refresh) and (
            not args.model or not any(target in {"af3", "af3-la"} for target in args.model)
        ):
            parser.error("AF3 MSA options require check --model af3 or --model af3-la")
    if args.command == "validate" and (args.refilter or args.recompute_metrics) and (args.rerun or args.score):
        mode = "--refilter" if args.refilter else "--recompute-metrics"
        parser.error(f"{mode} cannot be combined with --rerun or --score")
    if args.command == "validate" and args.rerun_refold and (
        args.plan or args.refilter or args.recompute_metrics
    ):
        parser.error("--rerun-refold cannot be combined with --plan, --refilter, or --recompute-metrics")
    if args.command == "validate" and args.msa_refresh and (
        args.plan or args.refilter or args.recompute_metrics
    ):
        parser.error("--msa-refresh cannot be combined with --plan, --refilter, or --recompute-metrics")


def _validation_argv(args) -> list[str]:
    argv = []
    for value in args.runs:
        path = Path(value).expanduser()
        candidate = Path.cwd() / "inference" / path
        bare_name = not path.is_absolute() and "/" not in value and "\\" not in value
        argv.append(str(candidate.resolve()) if bare_name and candidate.exists() else value)
    for name in ("config", "task", "method", "source", "criteria", "presets",
                 "worker_gpus", "worker_cpus", "limit", "msa", "msa_cache_dir",
                 "msa_profile", "msa_database_dir", "msa_binaries_dir",
                 "msa_workers", "msa_cpus", "jackhmmer_cpus", "nhmmer_cpus",
                 "msa_timeout", "msa_templates", "msa_template_date"):
        value = getattr(args, name)
        if value is None:
            continue
        for item in value if isinstance(value, list) else [value]:
            argv.extend(["--" + name.replace("_", "-"), str(item)])
    for name in ("refilter", "recompute_metrics", "plan", "score", "rerun",
                 "rerun_refold", "msa_refresh"):
        if getattr(args, name):
            argv.append("--" + name.replace("_", "-"))
    return argv


def main(argv: Sequence[str] | None = None) -> int:
    configure_lightning_notices()
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(args, parser)
    interactive_download = (args.command == "download" and not args.targets
                            and not args.all and not args.list)
    if interactive_download:
        from .interactive import can_prompt
        if not can_prompt():
            parser.error("download needs TARGET(s), --all or --list outside an interactive terminal")
    try:
        if args.command == "cluster":
            from src.cluster.cli import run as cluster_run
            return cluster_run(args)
        if args.command == "postfilter":
            from src.postfilter.cli import run as postfilter_run
            return postfilter_run(args)
        root = model_dir(args.model_dir)
        if args.command == "download":
            from .download import TARGETS, download_targets, list_targets, selection_options
            targets = list(TARGETS) if args.all else args.targets
            from .ui import DownloadDisplay, show_inventory
            if args.list:
                records = [item for item in list_targets(root) if not targets or item["target"] in targets]
                show_inventory(records, root)
                return 0
            if interactive_download:
                from .interactive import select_downloads
                targets = select_downloads(selection_options(root, force=args.force), root)
                if targets is None:
                    print("Cancelled")
                    return 0
            # Keep the downloaded asset root explicit while exposing the
            # immutable source/resource root through partidesign.paths.  This
            # matters for source checkouts where --model-dir is external: the
            # AF3 CCD converter must import the vendored SDK from the checkout,
            # never from ROOT/community_models.
            with using_model_dir(root):
                with DownloadDisplay(root, targets) as display:
                    records = download_targets(targets, root, force=args.force, progress=display.on_event)
                    display.finish(records)
            return int(any(item["status"] in {"failed", "unavailable"} for item in records))
        with using_model_dir(root):
            if args.command == "design":
                runner = build_design_runner(args, root)
                if args.show_config:
                    from omegaconf import OmegaConf
                    print(OmegaConf.to_yaml(runner.config, resolve=True), end="")
                    return 0
                from .ui import show_context
                show_context("Design", root, "Tasks: " + ", ".join(task.name for task in runner.tasks))
                result = runner.run()
                return 2 if result.failed or result.postprocess_failed or result.score_failed else 0
            if args.command == "relax":
                from .ui import show_context
                from src.design.postproc.resume import recover_postprocessing
                runner = build_design_runner(args, root)
                show_context("Relax", root, "Existing outputs: " + str(runner.config.output_dir))
                result = recover_postprocessing(
                    runner, limit=args.limit, rerun=args.rerun, plan=args.plan,
                    platform=args.platform, charge_method=args.charge_method)
                if args.plan:
                    print(f"Tasks: {result['tasks']}; samples to retry: {result['samples']}; platform: {result['platform']}")
                    return 0
                print(f"Relaxed: {result.postprocessed}; failed: {result.postprocess_failed}; scored: {result.scored}; score failed: {result.score_failed}")
                return 2 if result.postprocess_failed or result.score_failed else 0
            if args.command == "check":
                from .checking import run_check
                return run_check(args, root)
            if args.command == "score":
                config = build_score_config(args, root)
                if args.show_config:
                    import yaml
                    print(yaml.safe_dump(config, sort_keys=False), end="")
                    return 0
                from src.scoring.cli import run as score_run
                return int(score_run(config) or 0)
            from src.validate.__main__ import main as validate_main
            if not args.plan:
                from .ui import show_context
                show_context("Validate", root)
            try:
                return int(validate_main(_validation_argv(args)) or 0)
            except (ValueError, OSError, RuntimeError) as error:
                # Match the native validation entry point's failure status.
                from src.validate.progress import emit
                emit({"phase": "failed", "error": str(error)}, error=True)
                return 2
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        from .ui import show_error
        show_error(f"{type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
