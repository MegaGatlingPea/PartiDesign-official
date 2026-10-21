"""Strict, small validation configuration independent of generation Hydra."""
from __future__ import annotations

import argparse
import copy
import math
from datetime import date
from pathlib import Path
import os
from partidesign.paths import asset_root, ENV_MODEL_DIR
from typing import Any

METHODS = ("af2_ig", "esmfold2-fast", "esmfold2-standard", "af3", "af3-la", "partiscore", "vina_dock")
SOURCES = (
    "gt",
    "protein_mpnn",
    "soluble_mpnn",
    "ligand_mpnn",
    "cyclic_mpnn",
    "na_mpnn",
)
_SOURCE_ALIASES = {
    "gt": "gt",
    "protein_mpnn": "protein_mpnn",
    "proteinmpnn": "protein_mpnn",
    "protein_mpnn-fi": "protein_mpnn",
    "proteinmpnn-fi": "protein_mpnn",
    "soluble_mpnn": "soluble_mpnn",
    "solublempnn": "soluble_mpnn",
    "soluble_mpnn-fi": "soluble_mpnn",
    "solublempnn-fi": "soluble_mpnn",
    "ligand_mpnn": "ligand_mpnn",
    "ligandmpnn": "ligand_mpnn",
    "ligand_mpnn-fi": "ligand_mpnn",
    "ligandmpnn-fi": "ligand_mpnn",
    "cyclic_mpnn": "cyclic_mpnn",
    "cyclicmpnn": "cyclic_mpnn",
    "cyclic_mpnn-fi": "cyclic_mpnn",
    "cyclicmpnn-fi": "cyclic_mpnn",
    "na_mpnn": "na_mpnn",
    "nampnn": "na_mpnn",
    "na_mpnn-fi": "na_mpnn",
    "nampnn-fi": "na_mpnn",
}

# The cache/config layer imports only the standard library, never the AF3 SDK.
from .af3_msa import DEFAULT_MSA_CONFIG as _EXPORTED_MSA_CONFIG
from .af3_msa import validate_msa_settings

DEFAULT_MSA_CONFIG = copy.deepcopy(_EXPORTED_MSA_CONFIG)

DEFAULTS = {
    "methods": ["esmfold2-fast"],
    # The fold-validation default is resolved per input after its designed
    # modality is known. Protein/cyclic inputs use SolubleMPNN; nucleic-acid
    # inputs use NA-MPNN. Keep a protein default here for config consumers
    # that only have a parser-level view; runner source resolution supplies
    # the modality-aware list at execution time.
    "sequence": {"sources": [{"source": "soluble_mpnn", "count": 1}], "seed": 101},
    "af2": {"models": [1], "samples": 1, "recycles": 3, "seed": 101},
    "esmfold2": {"samples": 1, "mode": "fast", "loops": 20, "steps": 100,
                 "lm_dropout": 0.3, "seed": 101},
    "af3": {"samples": 1, "mode": "complex", "recycles": 10, "seed": 101,
            "attention_backend": "triton",
            "buckets": [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120],
            "timeout_seconds": 1800,
            "msa": copy.deepcopy(DEFAULT_MSA_CONFIG)},
    "scorer": {"enabled": False, "original": True, "refold": True,
               "checkpoint": "ckpts/partiscore.ckpt", "weights": "raw",
               "batch_size": 32, "num_workers": 0},
    "vina": {
        "cpu": 1,
        "seed": 101,
        "exhaustiveness": 8,
        "n_poses": 20,
        "box_margin": 5.0,
        "box_min_size": 20.0,
        "box_max_size": 40.0,
        "target_chains": None,
        "ligand_chain": None,
        "structure_stage": "postprocessed",
    },
    "structure_stage": "base",
    "criteria": [], "worker_gpus": None, "worker_cpus": 1.0,
    "unfiltered_success": False,
    "rerun": False, "rerun_refold": False,
}


def load_yaml(path: Path) -> dict:
    import yaml
    try:
        value = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


MPNN_FIELDS = {"temperature", "backbone_noise", "omit_aas", "rm_aa", "bias_aa", "pssm_multi",
               "pssm_threshold", "pssm_log_odds", "pssm_bias", "seed", "checkpoint",
               "pssm_path", "tied_positions_path", "model_type", "base_model",
               "base_model_name", "use_soluble_model", "ca_only", "use_ligand_atoms",
               "use_sidechain_context", "pack_side_chains", "repack_everything",
               "homo_oligomer", "symmetry_residues_path", "symmetry_weights_path",
               "bias_aa_json", "asset_root", "fixed_interface"}
EXTRA_FIELDS = {
    "sequence": {
        "protein_mpnn", "soluble_mpnn", "proteinmpnn", "solublempnn",
        "ligand_mpnn", "ligandmpnn", "cyclic_mpnn", "cyclicmpnn", "na_mpnn", "nampnn",
    },
    "af2": {"mode", "params_dir", "source_path", "colabdesign_source"},
    "esmfold2": {"python", "runtime_python", "cpus", "gpus", "lm_mask_pct", "attention_backend",
                "esmc_attention_backend", "context_embedding_cache", "timeout_seconds", "optimization"},
}


def _settings(value, context):
    if not isinstance(value, dict) or set(value) - MPNN_FIELDS:
        raise ValueError(f"{context} must be a mapping containing supported MPNN settings")
    numeric = {"temperature", "backbone_noise", "pssm_multi", "pssm_threshold"}
    for key in numeric & value.keys():
        if isinstance(value[key], bool) or not isinstance(value[key], (float, int)) or not math.isfinite(value[key]):
            raise ValueError(f"{context}.{key} must be numeric")
    for key in {"pssm_log_odds", "pssm_bias"} & value.keys():
        if not isinstance(value[key], bool):
            raise ValueError(f"{context}.{key} must be boolean")
    if "bias_aa" in value and not isinstance(value["bias_aa"], dict):
        raise ValueError(f"{context}.bias_aa must be a mapping")
    for key in {"checkpoint", "pssm_path", "tied_positions_path", "omit_aas"} & value.keys():
        if value[key] is not None and not isinstance(value[key], str):
            raise ValueError(f"{context}.{key} must be a string")
    if "rm_aa" in value:
        from src.design.rewards.aa_omission import resolve_rm_aa

        try:
            resolve_rm_aa(value)
        except ValueError as exc:
            raise ValueError(f"{context}.rm_aa: {exc}") from exc
    if "seed" in value and (type(value["seed"]) is not int or value["seed"] < 0):
        raise ValueError(f"{context}.seed must be a nonnegative integer")
    if "fixed_interface" in value:
        from .sequences import _normalize_fixed_interface

        _normalize_fixed_interface(value["fixed_interface"])


def _validate_esmfold_optimization(value: Any) -> None:
    if not isinstance(value, dict) or set(value) - {"mode", "disable"}:
        raise ValueError("esmfold2.optimization requires mode and disable fields")
    if "disable" in value and not isinstance(value["disable"], list):
        raise ValueError("esmfold2.optimization.disable must be a list")


def _merge_mapping(result: dict, override: dict, *, path: str = "") -> None:
    """Deep-merge one closed mapping while retaining legacy extra fields."""
    extras = EXTRA_FIELDS.get(path, set())
    for key, value in override.items():
        if key not in result:
            if key not in extras:
                if path:
                    raise ValueError(f"Unknown {path} configuration fields: {key}")
                raise ValueError(f"Unknown configuration field: {key}")
            result[key] = copy.deepcopy(value)
            if path == "sequence":
                _settings(value, f"sequence.{key}")
            elif path == "esmfold2" and key == "optimization":
                _validate_esmfold_optimization(value)
            continue
        current = result[key]
        if isinstance(current, dict):
            if not isinstance(value, dict):
                label = f"{path}." if path else ""
                raise ValueError(f"{label}{key} must be a mapping")
            _merge_mapping(current, value, path=f"{path}.{key}" if path else key)
        else:
            result[key] = copy.deepcopy(value)


def merge(base: dict, override: dict) -> dict:
    if not isinstance(base, dict) or not isinstance(override, dict):
        raise ValueError("configuration and overlay must be mappings")
    result = copy.deepcopy(base)
    _merge_mapping(result, override)
    return result


def normalize_sources(sources: Any) -> list[dict]:
    if not isinstance(sources, list) or not sources:
        raise ValueError("sequence.sources must be a nonempty ordered list")
    from .sequences import normalize_sequence_sources

    result = normalize_sequence_sources(sources, default_gt=False)
    for item in result:
        _settings(item["settings"], f"source {item['source']}")
    return result


def _validate_vina(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("vina must be a mapping")
    integer_fields = ("cpu", "seed", "exhaustiveness", "n_poses")
    for key in integer_fields:
        if type(value.get(key)) is not int:
            raise ValueError(f"vina.{key} must be an integer")
        if key != "seed" and value[key] < 1:
            raise ValueError(f"vina.{key} must be a positive integer")
        if key == "seed" and value[key] < 0:
            raise ValueError("vina.seed must be a nonnegative integer")
    for key in ("box_margin", "box_min_size", "box_max_size"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) \
                or not math.isfinite(float(raw)) or float(raw) <= 0:
            raise ValueError(f"vina.{key} must be a positive finite number")
    if float(value["box_min_size"]) > float(value["box_max_size"]):
        raise ValueError("vina.box_min_size must not exceed vina.box_max_size")
    targets = value.get("target_chains")
    if targets is not None:
        if not isinstance(targets, list) or any(
            not isinstance(item, str) or not item.strip() for item in targets
        ):
            raise ValueError("vina.target_chains must be null or a list of nonempty strings")
        if len(set(targets)) != len(targets):
            raise ValueError("vina.target_chains must contain unique chain IDs")
    ligand_chain = value.get("ligand_chain")
    if ligand_chain is not None and (
        not isinstance(ligand_chain, str) or not ligand_chain.strip()
    ):
        raise ValueError("vina.ligand_chain must be null or a nonempty string")
    if value.get("structure_stage") != "postprocessed":
        raise ValueError("vina.structure_stage must be 'postprocessed'")
    return value


def validate_af3_settings(value: Any) -> dict[str, Any]:
    """Validate shared AF3/AF3-LA settings, including optional MSA search."""
    if not isinstance(value, dict) or set(value) != set(DEFAULTS["af3"]):
        raise ValueError("af3 must contain only the documented AF3 settings")
    for key in ("samples", "recycles"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"af3.{key} must be a positive integer")
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2**32:
        raise ValueError("af3.seed must be a uint32 integer")
    if not isinstance(value["mode"], str) or value["mode"] not in {"complex", "monomer", "both"}:
        raise ValueError("af3.mode must be complex, monomer, or both")
    if not isinstance(value["attention_backend"], str) or value["attention_backend"] not in {"triton", "cudnn", "xla"}:
        raise ValueError("af3.attention_backend must be triton, cudnn, or xla")
    buckets = value["buckets"]
    if (not isinstance(buckets, list) or not buckets
            or any(type(n) is not int or n < 1 for n in buckets)
            or buckets != sorted(set(buckets))):
        raise ValueError("af3.buckets must be strictly increasing positive integers")
    timeout = value["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("af3.timeout_seconds must be finite and positive")
    validate_msa_settings(value["msa"])
    return value


def _positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed < 1 or str(parsed) != str(value):
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _msa_cpu_arg(value: str) -> int | str:
    if value == "auto":
        return value
    return _positive_int_arg(value)


def _strict_bool_arg(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _iso_date_arg(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected an ISO date (YYYY-MM-DD)") from exc
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("expected an ISO date (YYYY-MM-DD)")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate existing PartiDesign run/task result.csv files")
    parser.add_argument("runs", nargs="+", help="Run paths, task paths, or names under inference/")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", action="append", help="Select task names (repeatable)")
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument(
        "--source",
        action="append",
        help=(
            "Explicit SOURCE:COUNT, repeat in desired order. SOURCE accepts gt, "
            "protein_mpnn, protein_mpnn-fi, proteinmpnn, proteinmpnn-fi, "
            "soluble_mpnn, soluble_mpnn-fi, solublempnn, solublempnn-fi, "
            "ligand_mpnn, ligand_mpnn-fi, ligandmpnn, ligandmpnn-fi, "
            "cyclic_mpnn, cyclic_mpnn-fi, cyclicmpnn, cyclicmpnn-fi, "
            "na_mpnn, na_mpnn-fi, nampnn, or nampnn-fi."
        ),
    )
    parser.add_argument("--criteria", action="append", help="Named preset; repeatable")
    parser.add_argument("--presets", type=Path)
    parser.add_argument("--worker-gpus", type=float)
    parser.add_argument("--worker-cpus", type=float)
    parser.add_argument("--refilter", action="store_true", help="Rebuild reports/filter persisted metrics without models")
    parser.add_argument("--recompute-metrics", action="store_true",
                        help="Re-evaluate persisted refold structures without models")
    parser.add_argument("--plan", action="store_true", help="Read input inventory/configuration without output or model work")
    parser.add_argument("--score", action="store_true", help="PartiScore original and successful refold structures")
    rerun = parser.add_mutually_exclusive_group()
    rerun.add_argument("--rerun", action="store_true", help="Regenerate all requested sequence slots")
    rerun.add_argument(
        "--rerun-refold", action="store_true",
        help="AF3/AF3-LA only: reuse saved sequence slots and regenerate refolds",
    )
    parser.add_argument("--msa", choices=("auto", "off", "cache-only"),
                        help="AF3 target/context MSA policy")
    parser.add_argument("--msa-cache-dir", help="AF3 MSA cache directory")
    parser.add_argument("--msa-profile", help="Recorded AF3 MSA cache profile")
    parser.add_argument("--msa-database-dir", help="AF3 public database root")
    parser.add_argument("--msa-binaries-dir", help="Directory containing jackhmmer/nhmmer")
    parser.add_argument("--msa-workers", type=_positive_int_arg,
                        help="Concurrent AF3 MSA searches (positive integer)")
    parser.add_argument("--msa-cpus", type=_msa_cpu_arg,
                        help="AF3 MSA CPU budget: auto or a positive integer")
    parser.add_argument("--jackhmmer-cpus", type=_positive_int_arg,
                        help="Threads per AF3 jackhmmer process")
    parser.add_argument("--nhmmer-cpus", type=_positive_int_arg,
                        help="Threads per AF3 nhmmer process")
    parser.add_argument("--msa-timeout", type=_positive_int_arg,
                        help="AF3 MSA search timeout in seconds")
    parser.add_argument("--msa-refresh", action="store_true",
                        help="Refresh matching AF3 MSA cache generations")
    parser.add_argument("--msa-templates", type=_strict_bool_arg,
                        help="AF3 native structure-template search: true or false")
    parser.add_argument("--msa-template-date", type=_iso_date_arg,
                        help="AF3 maximum template date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, help="Maximum samples per task for a bounded run")
    args = parser.parse_args(argv)
    overlay = load_yaml(args.config) if args.config else {}
    cfg = merge(DEFAULTS, overlay)
    if ENV_MODEL_DIR in os.environ:
        scorer_overlay = overlay.get("scorer", {})
        if isinstance(scorer_overlay, dict) and "checkpoint" in scorer_overlay:
            cfg["scorer"]["checkpoint"] = str(Path(cfg["scorer"]["checkpoint"]).expanduser().resolve())
        else:
            cfg["scorer"]["checkpoint"] = str(asset_root() / "ckpts" / "partiscore.ckpt")
    if args.method:
        cfg["methods"] = args.method
    explicit_sources = args.source is not None or (
        isinstance(overlay.get("sequence"), dict)
        and "sources" in overlay["sequence"]
    )
    # Execution metadata, not a user-authored config field. The runner uses
    # it to distinguish an explicit source list from the modality-aware
    # parser default when it resolves one input at a time.
    cfg["_sequence_sources_explicit"] = bool(explicit_sources)
    if (
        not explicit_sources
        and isinstance(cfg.get("methods"), list)
        and any(method in {"partiscore", "vina_dock"} for method in cfg["methods"])
    ):
        cfg["sequence"]["sources"] = [{"source": "gt", "count": 1}]
    if args.source:
        values = []
        for value in args.source:
            try:
                name, count = value.rsplit(":", 1)
                values.append({"source": name, "count": int(count)})
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid --source {value!r}; expected SOURCE:COUNT") from exc
        cfg["sequence"]["sources"] = values
    cfg["sequence"]["sources"] = normalize_sources(cfg["sequence"]["sources"])
    if args.criteria is not None:
        cfg["criteria"] = args.criteria
        cfg["unfiltered_success"] = False
    if args.worker_gpus is not None:
        cfg["worker_gpus"] = args.worker_gpus
    if args.worker_cpus is not None:
        cfg["worker_cpus"] = args.worker_cpus
    if args.score:
        cfg["scorer"]["enabled"] = True
    if args.rerun:
        cfg["rerun"] = True
    msa_cli_values = {
        "mode": args.msa,
        "cache_dir": args.msa_cache_dir,
        "cache_profile": args.msa_profile,
        "database_dir": args.msa_database_dir,
        "binaries_dir": args.msa_binaries_dir,
        "search_workers": args.msa_workers,
        "cpu_budget": args.msa_cpus,
        "jackhmmer_n_cpu": args.jackhmmer_cpus,
        "nhmmer_n_cpu": args.nhmmer_cpus,
        "timeout_seconds": args.msa_timeout,
        "templates": args.msa_templates,
        "max_template_date": args.msa_template_date,
    }
    msa_cli_explicit = any(value is not None for value in msa_cli_values.values()) or args.msa_refresh
    if msa_cli_explicit:
        cfg["af3"]["msa"].update({
            key: value for key, value in msa_cli_values.items() if value is not None
        })
        if args.msa_refresh:
            cfg["af3"]["msa"]["refresh"] = True
    if args.rerun_refold:
        cfg["rerun_refold"] = True
    worker_gpus = cfg.get("worker_gpus")
    if worker_gpus is not None:
        if (isinstance(worker_gpus, bool)
                or not isinstance(worker_gpus, (int, float))
                or not math.isfinite(float(worker_gpus))
                or not 0 < float(worker_gpus) <= 1):
            raise ValueError("worker_gpus must be finite and in (0, 1]")
        cfg["worker_gpus"] = float(worker_gpus)
    worker_cpus = cfg.get("worker_cpus")
    if (isinstance(worker_cpus, bool)
            or not isinstance(worker_cpus, (int, float))
            or not math.isfinite(float(worker_cpus))
            or float(worker_cpus) <= 0):
        raise ValueError("worker_cpus must be a positive finite number")
    cfg["worker_cpus"] = float(worker_cpus)
    if not isinstance(cfg["methods"], list) or not cfg["methods"] or any(m not in METHODS for m in cfg["methods"]):
        raise ValueError(f"methods must be a nonempty subset of {METHODS}")
    if msa_cli_explicit and not any(method in {"af3", "af3-la"} for method in cfg["methods"]):
        raise ValueError("MSA command options require an AF3/AF3-LA validation method")
    if len(cfg["methods"]) != len(set(cfg["methods"])):
        raise ValueError("Duplicate validation methods")
    if type(cfg["rerun"]) is not bool or type(cfg["rerun_refold"]) is not bool:
        raise ValueError("rerun and rerun_refold must be boolean")
    if cfg["rerun"] and cfg["rerun_refold"]:
        raise ValueError("rerun and rerun_refold are mutually exclusive")
    if cfg["rerun_refold"] and any(method not in {"af3", "af3-la"} for method in cfg["methods"]):
        raise ValueError("--rerun-refold is supported only with AF3 or AF3-LA")
    if cfg["rerun_refold"] and (args.plan or args.refilter or args.recompute_metrics):
        raise ValueError("--rerun-refold cannot be combined with --plan, --refilter, or --recompute-metrics")
    if args.msa_refresh and (args.plan or args.refilter or args.recompute_metrics):
        raise ValueError("--msa-refresh cannot be combined with --plan, --refilter, or --recompute-metrics")
    if cfg.get("structure_stage") not in {"base", "postprocessed"}:
        raise ValueError("structure_stage must be 'base' or 'postprocessed'")
    _validate_vina(cfg.get("vina"))
    validate_af3_settings(cfg.get("af3"))
    if "vina_dock" in cfg["methods"] and len(cfg["methods"]) != 1:
        raise ValueError("vina_dock cannot be combined with other validation methods")
    if "vina_dock" in cfg["methods"] and cfg["scorer"]["enabled"]:
        raise ValueError("vina_dock does not use scorer/refold")
    if "partiscore" in cfg["methods"] and any(s["source"] != "gt" for s in cfg["sequence"]["sources"]):
        raise ValueError("Standalone partiscore requires GT structure coordinates; use --score with a refold method for MPNN sequences")
    if "vina_dock" in cfg["methods"] and any(s["source"] != "gt" for s in cfg["sequence"]["sources"]):
        raise ValueError("vina_dock requires GT-only validation sequences")
    if not isinstance(cfg["criteria"], list) or any(not isinstance(x, str) for x in cfg["criteria"]):
        raise ValueError("criteria must be a list of preset names")
    if type(cfg["unfiltered_success"]) is not bool:
        raise ValueError("unfiltered_success must be boolean")
    if cfg["unfiltered_success"] and cfg["criteria"]:
        raise ValueError("unfiltered_success requires empty criteria")
    for group in ("af2", "esmfold2"):
        if type(cfg[group].get("samples")) is not int or cfg[group]["samples"] < 1:
            raise ValueError(f"{group}.samples must be a positive integer")
    models = cfg["af2"].get("models")
    if not isinstance(models, list) or not models or any(type(m) is not int or m not in (1, 2) for m in models) or len(set(models)) != len(models):
        raise ValueError("af2.models must be an ordered unique subset of [1, 2]")
    if cfg["af2"].get("mode", "both") != "both":
        raise ValueError("AF2 validation requires both complex and monomer predictions")
    for group, keys in (("af2", ("recycles",)), ("esmfold2", ("loops", "steps"))):
        for key in keys:
            if type(cfg[group].get(key)) is not int or cfg[group][key] < 1:
                raise ValueError(f"{group}.{key} must be a positive integer")
    for group in ("sequence", "af2", "esmfold2"):
        if type(cfg[group].get("seed")) is not int or cfg[group]["seed"] < 0:
            raise ValueError(f"{group}.seed must be a nonnegative integer")
    if cfg["esmfold2"].get("mode") not in {"fast", "standard"}:
        raise ValueError("esmfold2.mode must be fast or standard")
    dropout = cfg["esmfold2"].get("lm_dropout")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout <= 1:
        raise ValueError("esmfold2.lm_dropout must be between 0 and 1")
    for key in ("enabled", "original", "refold"):
        if type(cfg["scorer"][key]) is not bool:
            raise ValueError(f"scorer.{key} must be boolean")
    for key, minimum in (("batch_size", 1), ("num_workers", 0)):
        if type(cfg["scorer"][key]) is not int or cfg["scorer"][key] < minimum:
            raise ValueError(f"scorer.{key} must be an integer >= {minimum}")
    if not isinstance(cfg["scorer"]["checkpoint"], str) or not isinstance(cfg["scorer"]["weights"], str) or cfg["scorer"]["weights"] not in {"raw", "ema"}:
        raise ValueError("scorer checkpoint must be a path and weights must be raw or ema")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    return args, cfg


def config_for_method(cfg: dict, method: str) -> dict:
    """Resolve selected weights in an isolated per-method configuration."""
    resolved = copy.deepcopy(cfg)
    if method in {"esmfold2-fast", "esmfold2-standard"}:
        resolved.setdefault("esmfold2", {})["mode"] = method.removeprefix("esmfold2-")
    return resolved
