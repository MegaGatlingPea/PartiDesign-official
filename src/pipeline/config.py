"""Closed, model-free configuration for the iterative design pipeline.

Task authoring YAML is retained verbatim.  Workflow overlays only control the
orchestrator; a selected task's original molecular conditions are never edited.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any, Mapping

from partidesign.paths import inference_config_dir, model_dir as resolve_model_dir
from src.validate.config import DEFAULTS as VALIDATE_DEFAULTS
from src.validate.config import merge as merge_validation
from src.validate.config import validate_af3_settings, _settings, _validate_vina

METHODS = ("esmfold2-fast", "af2_ig", "af3")
PROFILES = ("auto", "protein", "nuc", "ppi")
_ROUTES = {"monomer", "protein", "nuc", "ligand", "peptide", "antibody", "ame", "joint"}
_ROUTE_ALIASES = {"protein_context": "protein", "nucleic_context": "nuc",
                  "nuc_context": "nuc", "ligand_context": "ligand",
                  "protein_protein": "protein", "protein_na": "nuc",
                  "nucleic": "nuc", "na": "nuc"}
_ROOT_FIELDS = {"preset", "recycle", "seed", "run_name", "output_dir", "from_run",
                "resume", "generation", "iteration", "openmm",
                "final", "rounds", "tasks", "presets", "criteria_routes"}
_TASK_FIELDS = {"preset", "recycle", "iteration", "openmm", "final",
                "rounds", "presets", "criteria_routes"}
_ITERATION_FIELDS = {"sequence", "refold", "filter", "selection", "transition"}
_FINAL_FIELDS = {"method", "validation", "criteria", "criteria_route", "criteria_mode", "pool",
                 "prediction_selection"}
_SEQUENCE_FIELDS = {"include_gt", "mpnn"}
_MPNN_FIELDS = {"source", "count", "settings"}
_REFOLD_FIELDS = {"method", "conformers", "validation"}
_FILTER_FIELDS = {"criteria", "criteria_route", "criteria_mode"}
_TRANSITION_FIELDS = {"relax", "partial"}
_PARTIAL_FIELDS = {"enabled", "fraction", "replicas", "include_parent"}
_POOL_FIELDS = {"source", "max_pass"}
_GENERATION_FIELDS = {"config", "ckpt", "batch_size", "num_workers", "route",
                      "route_ckpt", "compile", "compile_strategy",
                      "execution_backend", "override"}
_OPENMM_FIELDS = {"force_field", "charge_method", "platform", "use_ray",
                  "worker_gpus", "worker_cpus"}
_SELECTION_FIELDS = {"mode", "num_clusters", "max_pass", "max_pass_ratio",
                     "per_parent", "per_family", "rank_by"}
_NATIVE_VALIDATION_FIELDS = set(VALIDATE_DEFAULTS) - {
    "methods", "criteria", "unfiltered_success", "rerun", "rerun_refold",
}
_MPNN_SOURCES = {"auto", "protein_mpnn", "soluble_mpnn", "ligand_mpnn",
                 "cyclic_mpnn", "na_mpnn"}

BUILTINS = {
    "preset": "auto", "recycle": 3, "seed": 2026, "run_name": "pipeline",
    "output_dir": None, "from_run": None, "resume": False,
    "generation": {"config": None, "ckpt": None, "batch_size": 16,
                   "num_workers": 4, "route": True, "route_ckpt": None,
                   "compile": "auto", "compile_strategy": "auto",
                   "execution_backend": "auto", "override": []},
    "iteration": {
        "sequence": {"include_gt": True, "mpnn": {"source": "auto", "count": 4, "settings": {}}},
        "refold": {"method": "esmfold2-fast", "conformers": 1, "validation": {}},
        "filter": {"criteria": None, "criteria_route": "auto", "criteria_mode": "all"},
        "selection": {"mode": "top", "max_pass": 200, "max_pass_ratio": None,
                      "per_parent": 1, "per_family": 2, "rank_by": "design_plddt"},
        "transition": {"relax": {"enabled": False},
                       "partial": {"enabled": True, "fraction": 0.625, "replicas": 4,
                                   "include_parent": True}},
    },
    "rounds": {1: {"sequence": {"mpnn": {"count": 1}},
                   "selection": {"mode": "cluster", "num_clusters": 200,
                                "per_parent": None, "rank_by": "design_plddt"}}},
    "openmm": {"force_field": "amber19sb", "charge_method": "gasteiger",
               "platform": "CUDA", "use_ray": False,
               "worker_gpus": 0.125, "worker_cpus": 1.0},
    "final": {"method": "af3", "validation": {"af3": {"samples": 5}}, "criteria": None,
              "prediction_selection": "best_ranked",
              "criteria_route": "auto", "criteria_mode": "all",
              "pool": {"source": "history_best", "max_pass": 200}},
    "presets": {}, "criteria_routes": {}, "tasks": {},
}


def _mapping(value: Any, label: str, fields: set[str] | None = None) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a mapping with string keys")
    if fields is not None and set(value) - fields:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(set(value) - fields))}")
    return value


def _integer(value: Any, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, minimum: float = 0,
            maximum: float | None = None, include_min: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or
            (value < minimum if include_min else value <= minimum)
            or (maximum is not None and value > maximum)):
        relation = ">=" if include_min else ">"
        raise ValueError(f"{label} must be finite, {relation} {minimum}"
                         + (f", and <= {maximum}" if maximum is not None else ""))
    return float(value)


def _choice(value: Any, choices: set | tuple, label: str) -> Any:
    if not isinstance(value, (str, type(None))) or value not in choices:
        raise ValueError(f"{label} must be one of {tuple(choices)}")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _names(value: Any, label: str, *, nullable: bool = False) -> list[str] | None:
    if value is None and nullable:
        return None
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"{label} must be a list of nonempty names")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicate names")
    return list(value)


def _path(value: Any, base: Path, label: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{label} must be a nonempty path")
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def _load(path: Path) -> dict:
    import yaml
    try:
        value = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    return _mapping(value, str(path))


def _merge(base: Mapping, overlay: Mapping) -> dict:
    """List replacement and selection-discriminator replacement."""
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if key == "selection" and isinstance(value, dict) and "mode" in value:
            result[key] = copy.deepcopy(value)
        elif key == "rounds" and isinstance(value, dict):
            rounds = copy.deepcopy(result.get(key, {}))
            for raw_round, row in value.items():
                number = _round_number(raw_round)
                rounds[number] = _merge(rounds.get(number, {}), row)
            result[key] = rounds
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _round_number(value: Any) -> int:
    if type(value) is int:
        return _integer(value, "round number")
    if isinstance(value, str) and value.isdecimal() and str(int(value)) == value:
        return _integer(int(value), "round number")
    raise ValueError(f"Round keys must be positive integers, got {value!r}")


def _check_overlay(value: dict, label: str, *, task: bool = False) -> None:
    _mapping(value, label, _TASK_FIELDS if task else _ROOT_FIELDS)
    for key, fields in (("iteration", _ITERATION_FIELDS),
                        ("generation", _GENERATION_FIELDS),
                        ("openmm", _OPENMM_FIELDS), ("final", _FINAL_FIELDS)):
        if key in value:
            _mapping(value[key], f"{label}.{key}", fields)
    if "rounds" in value:
        if not isinstance(value["rounds"], dict):
            raise ValueError(f"{label}.rounds must be a mapping keyed by round")
        seen = set()
        for key, row in value["rounds"].items():
            number = _round_number(key)
            if number in seen:
                raise ValueError(f"Duplicate round {number} in {label}")
            seen.add(number)
            _check_iteration(row, f"{label}.rounds.{number}", partial=True)
    if "tasks" in value:
        _mapping(value["tasks"], f"{label}.tasks")
        for name, row in value["tasks"].items():
            _check_overlay(row, f"{label}.tasks.{name}", task=True)
    # Validate scalars even when a later overlay would mask an invalid setting.
    for key in ("recycle",):
        if key in value:
            _integer(value[key], f"{label}.{key}")
    if "preset" in value:
        _choice(value["preset"], PROFILES, f"{label}.preset")
    if "iteration" in value:
        _check_iteration(value["iteration"], f"{label}.iteration", partial=True)
    if "final" in value:
        final = value["final"]
        if "method" in final:
            _choice(final["method"], {"af3", "af2_ig"}, f"{label}.final.method")
        _check_filter({k: v for k, v in final.items() if k in _FILTER_FIELDS},
                      f"{label}.final")
        if "prediction_selection" in final:
            _choice(final["prediction_selection"], {"best_ranked", "all"},
                    f"{label}.final.prediction_selection")
        if "validation" in final:
            _check_validation(final["validation"], f"{label}.final.validation")
        if "pool" in final:
            _pool(final["pool"], f"{label}.final.pool", partial=True)


def _check_iteration(value: dict, label: str, *, partial: bool = False) -> None:
    _mapping(value, label, _ITERATION_FIELDS)
    if "sequence" in value:
        sequence = _mapping(value["sequence"], f"{label}.sequence", _SEQUENCE_FIELDS)
        if "include_gt" in sequence:
            _boolean(sequence["include_gt"], f"{label}.sequence.include_gt")
        if "mpnn" in sequence:
            mpnn = _mapping(sequence["mpnn"], f"{label}.sequence.mpnn", _MPNN_FIELDS)
            if "count" in mpnn:
                _integer(mpnn["count"], f"{label}.sequence.mpnn.count", 0)
            if "source" in mpnn:
                _choice(mpnn["source"], _MPNN_SOURCES, f"{label}.sequence.mpnn.source")
            if "settings" in mpnn:
                _settings(mpnn["settings"], f"{label}.sequence.mpnn.settings")
        if not partial and not sequence["include_gt"] and not sequence["mpnn"]["count"]:
            raise ValueError(f"{label}.sequence requires GT or at least one MPNN proposal")
    if "refold" in value:
        refold = _mapping(value["refold"], f"{label}.refold", _REFOLD_FIELDS)
        if "method" in refold:
            _choice(refold["method"], METHODS, f"{label}.refold.method")
        if "conformers" in refold:
            _integer(refold["conformers"], f"{label}.refold.conformers")
        if "validation" in refold:
            _check_validation(refold["validation"], f"{label}.refold.validation")
    if "filter" in value:
        _check_filter(value["filter"], f"{label}.filter")
    if "selection" in value:
        _selection(value["selection"], f"{label}.selection", partial=partial)
    if "transition" in value:
        transition = _mapping(value["transition"], f"{label}.transition", _TRANSITION_FIELDS)
        if "relax" in transition:
            relax = _mapping(transition["relax"], f"{label}.transition.relax", {"enabled"})
            if "enabled" in relax:
                _boolean(relax["enabled"], f"{label}.transition.relax.enabled")
        if "partial" in transition:
            _partial(transition["partial"], f"{label}.transition.partial", overlay=partial)


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _check_filter(value: dict, label: str) -> None:
    _mapping(value, label, _FILTER_FIELDS)
    if "criteria" in value:
        _names(value["criteria"], f"{label}.criteria", nullable=True)
    if "criteria_mode" in value:
        _choice(value["criteria_mode"], {"all", "any"}, f"{label}.criteria_mode")
    if "criteria_route" in value:
        _route(value["criteria_route"], f"{label}.criteria_route")


def _check_validation(value: dict, label: str) -> None:
    validation = _mapping(value, label, _NATIVE_VALIDATION_FIELDS)
    sequence = validation.get("sequence", {})
    if isinstance(sequence, dict) and "sources" in sequence:
        raise ValueError(f"{label}.sequence.sources is pipeline-owned; "
                         "configure sequence.include_gt and sequence.mpnn instead")
    scorer = validation.get("scorer", {})
    if isinstance(scorer, dict) and scorer.get("enabled", False) is not False:
        raise ValueError("Pipeline does not run an additional PartiScore stage")


def _partial(value: dict, label: str, *, overlay: bool = False) -> dict:
    _mapping(value, label, _PARTIAL_FIELDS)
    for key in ("enabled", "include_parent"):
        if key in value:
            _boolean(value[key], f"{label}.{key}")
    if "replicas" in value:
        _integer(value["replicas"], f"{label}.replicas")
    if "fraction" in value:
        fraction = _number(value["fraction"], f"{label}.fraction", maximum=1)
        if round(fraction * 400) < 1:
            raise ValueError(f"{label}.fraction must retain at least one of 400 sampling steps")
    if overlay:
        return copy.deepcopy(value)
    result = {"enabled": value.get("enabled", False), "replicas": value.get("replicas", 4),
              "include_parent": value.get("include_parent", True)}
    if result["enabled"]:
        if "fraction" not in value:
            raise ValueError(f"{label}.fraction is required when partial is enabled")
        result["fraction"] = float(value["fraction"])
    return result


def _pool(value: dict, label: str, *, partial: bool = False) -> dict:
    _mapping(value, label, _POOL_FIELDS)
    if "source" in value:
        _choice(value["source"], {"history_best", "last"}, f"{label}.source")
    if "max_pass" in value:
        _integer(value["max_pass"], f"{label}.max_pass")
    return copy.deepcopy(value) if partial else {
        "source": value.get("source", "history_best"), "max_pass": value.get("max_pass", 200)}


def _route(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be auto or a named context route")
    value = _ROUTE_ALIASES.get(value, value)
    if value != "auto" and value not in _ROUTES:
        raise ValueError(f"{label}: unknown criteria route {value!r}")
    return value


def _selection(value: dict, label: str, *, partial: bool = False) -> dict:
    _mapping(value, label, _SELECTION_FIELDS)
    mode = value.get("mode")
    if mode is not None:
        _choice(mode, {"cluster", "top"}, f"{label}.mode")
    if mode == "cluster" and {"max_pass", "max_pass_ratio"} & value.keys():
        raise ValueError(f"{label}: cluster and top limits are mutually exclusive")
    if mode == "top" and "num_clusters" in value:
        raise ValueError(f"{label}: top selection cannot contain num_clusters")
    for key in ("num_clusters", "max_pass", "per_parent", "per_family"):
        if key in value and value[key] is not None:
            _integer(value[key], f"{label}.{key}")
    if "max_pass_ratio" in value and value["max_pass_ratio"] is not None:
        _number(value["max_pass_ratio"], f"{label}.max_pass_ratio", maximum=1)
    if "rank_by" in value:
        _string(value["rank_by"], f"{label}.rank_by")
        if value["rank_by"] != "design_plddt":
            raise ValueError(f"{label}.rank_by must be design_plddt")
    if partial:
        return copy.deepcopy(value)
    if mode is None:
        raise ValueError(f"{label}.mode is required")
    result = {"mode": mode, "rank_by": value.get("rank_by", "design_plddt"),
              "per_parent": value.get("per_parent", None if mode == "cluster" else 1)}
    if mode == "top" or "per_family" in value:
        result["per_family"] = value.get("per_family", 2)
    if mode == "cluster":
        result["num_clusters"] = _integer(value.get("num_clusters", 200),
                                          f"{label}.num_clusters")
    else:
        result["max_pass"] = value.get("max_pass", 200)
        result["max_pass_ratio"] = value.get("max_pass_ratio")
        if result["max_pass"] is None and result["max_pass_ratio"] is None:
            raise ValueError(f"{label}: top selection requires max_pass or max_pass_ratio")
    return result


def _modalities(task: dict, name: str) -> set[str]:
    _mapping(task, f"task {name}")
    if task.get("format") == "antibody":
        return {"protein"}
    chains = task.get("design_chains")
    if not isinstance(chains, list) or not chains:
        raise ValueError(f"task {name}: design_chains must be a nonempty list")
    result = set()
    seen = set()
    for row in chains:
        _mapping(row, f"task {name}.design_chains entry")
        if len(row) != 1:
            raise ValueError(f"task {name}: each design_chains entry must name one chain")
        chain, modality = next(iter(row.items()))
        _string(chain, f"task {name} chain")
        _choice(modality, {"protein", "peptide", "rna", "dna", "ligand"},
                f"task {name} design modality")
        if chain in seen:
            raise ValueError(f"task {name}: duplicate design chain {chain}")
        seen.add(chain)
        if modality in {"protein", "peptide"}:
            result.add("protein")
        elif modality in {"rna", "dna"}:
            result.add("nuc")
        elif modality == "ligand":
            result.add("ligand")
        else:
            raise ValueError(f"task {name}: unsupported design modality {modality!r}")
    if not result & {"protein", "nuc"}:
        raise ValueError(f"task {name}: pure ligand design is not supported by pipeline")
    return result


def _preset_name(value: Any, modalities: set[str], name: str) -> str:
    _choice(value, PROFILES, f"task {name}.preset")
    polymers = modalities & {"protein", "nuc"}
    if value == "ppi":
        if modalities != {"protein"}:
            raise ValueError(f"task {name}: preset ppi requires protein/peptide design")
        return value
    if value != "auto":
        if value not in polymers:
            raise ValueError(f"task {name}: preset {value} is incompatible with design modality")
        return value
    if len(polymers) != 1:
        raise ValueError(f"task {name}: mixed protein/nuc design requires an explicit preset "
                         "and applicable criteria_route or criteria")
    return next(iter(polymers))


def _presets(value: dict, label: str) -> dict:
    _mapping(value, label)
    for name, preset in value.items():
        _mapping(preset, f"{label}.{name}", {"methods", "reduction", "rules"})
        methods = preset.get("methods")
        if not isinstance(methods, list) or not methods or any(
                not isinstance(method, str) or method not in
                {"af2_ig", "esmfold2-fast", "esmfold2-standard", "af3", "af3-la",
                 "partiscore", "vina_dock"} for method in methods):
            raise ValueError(f"Invalid methods for criterion {name}")
        _choice(preset.get("reduction"), {"all_and_mean", "any_prediction",
                                           "best_ranked", "any_af2_pair"},
                f"criterion {name}.reduction")
        rules = preset.get("rules")
        if not isinstance(rules, list) or not rules:
            raise ValueError(f"Criterion {name} requires nonempty rules")
        for rule in rules:
            _mapping(rule, f"criterion {name} rule", {"metric", "op", "value", "scope"})
            _string(rule.get("metric"), f"criterion {name}.metric")
            _choice(rule.get("op"), {">", ">=", "<", "<=", "=="},
                    f"criterion {name}.op")
            _number(rule.get("value"), f"criterion {name}.value",
                    minimum=-math.inf, include_min=True)
            _choice(rule.get("scope", "complex"), {"complex", "monomer", "paired", "prediction"},
                    f"criterion {name}.scope")
    return copy.deepcopy(value)


def _criteria_routes(value: dict, definitions: dict) -> dict:
    _mapping(value, "criteria_routes", _ROUTES)
    result = {}
    for route, spec in value.items():
        _mapping(spec, f"criteria_routes.{route}", {"screen", "final"})
        result[route] = {}
        for stage in ("screen", "final"):
            names = spec.get(stage)
            if isinstance(names, dict):
                _mapping(names, f"criteria_routes.{route}.{stage}", set(METHODS))
                result[route][stage] = {
                    method: _names(items, f"criteria_routes.{route}.{stage}.{method}")
                    for method, items in names.items()
                }
                all_names = [name for items in result[route][stage].values() for name in items]
            else:
                result[route][stage] = _names(names, f"criteria_routes.{route}.{stage}")
                all_names = result[route][stage]
            unknown = set(all_names) - definitions.keys()
            if unknown:
                from .criteria import include_native_presets
                include_native_presets(definitions, unknown)
                unknown -= definitions.keys()
            if unknown:
                raise ValueError(f"Unknown criteria in route {route}: {sorted(unknown)}")
    return result


def criteria_for(task_config: dict, stage: dict | str, context_route: str,
                 method: str | None = None) -> tuple[list[str], dict]:
    """Resolve native presets after runtime knows the actual complex context.

    A round dictionary or the string 'final' is accepted as stage.  Returned
    definitions are isolated copies and can be passed directly to validate's
    filter evaluator without writing an intermediate preset file.
    """
    is_final = stage == "final" if isinstance(stage, str) else stage is task_config.get("final")
    if isinstance(stage, str):
        if stage != "final":
            raise ValueError("stage must be a round dictionary or 'final'")
        row = task_config["final"]
    else:
        row = stage
    selected_method = method or row["method"]
    if selected_method not in METHODS:
        raise ValueError(f"Unsupported pipeline method: {selected_method}")
    names = row.get("criteria")
    if names is None:
        route = row.get("criteria_route", "auto")
        if route == "auto":
            route = _route(context_route, "runtime context route")
        if route == "auto":
            raise ValueError("An ambiguous context requires explicit criteria or criteria_route")
        routing = task_config["criteria_routes"].get(route)
        if routing is None:
            raise ValueError(f"No {task_config['preset']} criteria route for {route}")
        names = routing["final" if is_final else "screen"]
        if isinstance(names, dict):
            if selected_method not in names:
                raise ValueError(f"No {selected_method} criteria for {route}; configure explicit criteria")
            names = names[selected_method]
    definitions = task_config["presets"]
    for name in names:
        if name not in definitions:
            raise ValueError(f"Unknown criteria preset: {name}")
        if selected_method not in definitions[name]["methods"]:
            raise ValueError(f"Criterion {name} does not apply to {selected_method}")
    return list(names), copy.deepcopy(definitions)


def _validation_defaults() -> dict:
    default_file = inference_config_dir() / "validate.yaml"
    defaults = copy.deepcopy(VALIDATE_DEFAULTS)
    if default_file.is_file():
        defaults = merge_validation(defaults, _load(default_file))
    # Keep pipeline-only protocol choices in one place, over native validate defaults.
    defaults["af2"]["mode"] = "both"
    defaults["af3"]["buckets"] = list(range(128, 1025, 128))
    return defaults


def _native_validation(overrides: dict, method: str, samples: int, seed: int,
                       base: Path, worker_gpus: Any, worker_cpus: Any, *, defaults=None) -> dict:
    _check_validation(overrides, "validation")
    defaults = copy.deepcopy(_validation_defaults() if defaults is None else defaults)
    # Sequence ownership belongs to the workflow.  These are harmless GT
    # placeholders until runtime resolves each parent's auto MPNN route.
    defaults["sequence"]["sources"] = [{"source": "gt", "label": "self", "count": 1,
                                        "settings": {}}]
    for group in ("sequence", "af2", "esmfold2", "af3"):
        defaults[group]["seed"] = seed
    cfg = merge_validation(defaults, overrides)
    # Validate caller values before workflow-owned sample/mode normalization:
    # bool-as-int or invalid dormant native settings must not be hidden by it.
    for group in ("af2", "esmfold2", "af3"):
        _integer(cfg[group]["samples"], f"validation.{group}.samples")
    if cfg["esmfold2"]["mode"] != "fast":
        raise ValueError("pipeline esmfold2.mode must be fast; select method at iteration.refold.method")
    if cfg["structure_stage"] != "base":
        raise ValueError("pipeline validation.structure_stage must be base")
    for key in ("enabled", "original", "refold"):
        if type(cfg["scorer"][key]) is not bool:
            raise ValueError(f"validation.scorer.{key} must be boolean")
    for key, minimum in (("batch_size", 1), ("num_workers", 0)):
        _integer(cfg["scorer"][key], f"validation.scorer.{key}", minimum)
    _choice(cfg["scorer"]["weights"], {"raw", "ema"}, "validation.scorer.weights")
    _validate_vina(cfg["vina"])
    cfg["methods"] = [method]
    cfg["criteria"] = []
    cfg["sequence"]["sources"] = [{"source": "gt", "label": "self", "count": 1,
                                   "settings": {}}]
    cfg["_sequence_sources_explicit"] = True
    cfg["unfiltered_success"] = False
    cfg["rerun"] = False
    cfg["rerun_refold"] = False
    cfg["structure_stage"] = "base"
    cfg["scorer"]["enabled"] = False
    if worker_gpus is not None:
        cfg["worker_gpus"] = worker_gpus
    if worker_cpus is not None:
        cfg["worker_cpus"] = worker_cpus
    selected_group = {"esmfold2-fast": "esmfold2", "af2_ig": "af2", "af3": "af3"}[method]
    cfg[selected_group]["samples"] = samples
    cfg["esmfold2"]["mode"] = "fast"
    for group in ("sequence", "af2", "esmfold2", "af3"):
        _integer(cfg[group]["seed"], f"validation.{group}.seed", 0)
    for group in ("af2", "esmfold2", "af3"):
        _integer(cfg[group]["samples"], f"validation.{group}.samples")
    for group, fields in (("af2", ("recycles",)), ("esmfold2", ("loops", "steps"))):
        for key in fields:
            _integer(cfg[group][key], f"validation.{group}.{key}")
    models = cfg["af2"]["models"]
    if (not isinstance(models, list) or not models or
            any(type(model) is not int or model not in (1, 2) for model in models)
            or len(models) != len(set(models))):
        raise ValueError("validation.af2.models must be an ordered unique subset of [1, 2]")
    if cfg["af2"].get("mode", "both") != "both":
        raise ValueError("AF2 validation requires its native complex/monomer paired protocol")
    _number(cfg["esmfold2"]["lm_dropout"], "validation.esmfold2.lm_dropout",
            maximum=1, include_min=True)
    for key in ("lm_mask_pct",):
        if key in cfg["esmfold2"]:
            _number(cfg["esmfold2"][key], f"validation.esmfold2.{key}",
                    maximum=1, include_min=True)
    for key in ("timeout_seconds", "cpus", "gpus"):
        if key in cfg["esmfold2"]:
            _number(cfg["esmfold2"][key], f"validation.esmfold2.{key}")
    if cfg["worker_gpus"] is not None:
        cfg["worker_gpus"] = _number(cfg["worker_gpus"], "validation.worker_gpus", maximum=1)
    cfg["worker_cpus"] = _number(cfg["worker_cpus"], "validation.worker_cpus")
    validate_af3_settings(cfg["af3"])
    # A scorer path remains part of native config even with the stage disabled.
    cfg["scorer"]["checkpoint"] = _path(cfg["scorer"]["checkpoint"], base,
                                         "validation.scorer.checkpoint")
    return cfg


def _generation(value: dict, base: Path, model_root: Path) -> dict:
    _mapping(value, "generation", _GENERATION_FIELDS)
    result = copy.deepcopy(value)
    _integer(result["batch_size"], "generation.batch_size")
    _integer(result["num_workers"], "generation.num_workers", 0)
    if type(result["route"]) is not bool:
        raise ValueError("generation.route must be boolean")
    _choice(result["route_ckpt"], {None, "omni", "prot", "nuc", "ligand", "antibody", "pep"},
            "generation.route_ckpt")
    if (type(result["compile"]) not in {str, bool}
            or result["compile"] not in {"auto", True, False, "true", "false"}):
        raise ValueError("generation.compile must be auto, true or false")
    if type(result["compile"]) is bool:
        result["compile"] = "true" if result["compile"] else "false"
    _choice(result["compile_strategy"], {"auto", "dynamic", "bucket"}, "generation.compile_strategy")
    _choice(result["execution_backend"], {"auto", "native", "bioir"}, "generation.execution_backend")
    if (not isinstance(result["override"], list)
            or any(not isinstance(x, str) or "=" not in x for x in result["override"])):
        raise ValueError("generation.override must be a list of KEY=VALUE strings")
    result["config"] = (_path(result["config"], base, "generation.config")
                        if result["config"] is not None
                        else str(inference_config_dir() / "design.yaml"))
    result["ckpt"] = (_path(result["ckpt"], base, "generation.ckpt")
                      if result["ckpt"] is not None
                      else str(model_root / "ckpts" / "partidesign.ckpt"))
    if not Path(result["config"]).is_file():
        raise ValueError(f"Generation configuration does not exist: {result['config']}")
    # The runner enforces these controls on effective generation, without editing
    # original authoring tasks or making a second score/reward/OpenMM pass.
    result["workflow_controls"] = {
        "score": False, "postprocess": False, "reward": None, "search": None,
    }
    return result


def build_config(args: Any, repo_root: str | Path, *,
                 invocation_dir: str | Path | None = None) -> dict:
    """Build isolated effective tasks without importing models or creating output."""
    base = Path(invocation_dir or Path.cwd()).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    task_arg = getattr(args, "task_file", None) or getattr(args, "tasks_file", None)
    task_file = Path(_path(task_arg, base, "task_file"))
    from .interface import unpack_document, workflow_overlay
    task_payload, embedded = unpack_document(_load(task_file))
    if not task_payload:
        raise ValueError("Task file must contain at least one authoring task")
    for name, task in task_payload.items():
        _mapping(task, f"task {name}")
    overlay_path = getattr(args, "config", None)
    external = workflow_overlay(_load(Path(_path(overlay_path, base, "config")))) if overlay_path else {}
    overlay = _merge(embedded, external)
    _check_overlay(overlay, "workflow")
    configured_tasks = overlay.get("tasks", {})
    unknown = set(configured_tasks) - task_payload.keys()
    if unknown:
        raise ValueError(f"Unknown configured tasks: {sorted(unknown)}")
    selected = getattr(args, "task", None)
    task_names = _names(selected, "--task") if selected is not None else list(task_payload)
    if not task_names:
        raise ValueError("Select at least one task")
    unknown = set(task_names) - task_payload.keys()
    if unknown:
        raise ValueError(f"Unknown selected tasks: {sorted(unknown)}")
    # Validate all explicit CLI values, including options scoped to rounds
    # outside a recycle=1 run. Invalid dormant flags must not be ignored.
    for attr, minimum in (("recycle", 1), ("first_mpnn_count", 0),
                          ("mpnn_count", 0), ("num_clusters", 1), ("max_pass", 1),
                          ("partial_replicas", 1)):
        if getattr(args, attr, None) is not None:
            _integer(getattr(args, attr), f"--{attr.replace('_', '-')}", minimum)
    if getattr(args, "max_pass_ratio", None) is not None:
        _number(args.max_pass_ratio, "--max-pass-ratio", maximum=1)
    if getattr(args, "partial_fraction", None) is not None:
        _partial({"fraction": args.partial_fraction}, "--partial", overlay=True)
    if getattr(args, "relax", None) is not None:
        _choice(args.relax, {"true", "false"}, "--relax")
    global_cfg = _merge(BUILTINS, {k: v for k, v in overlay.items() if k != "tasks"})
    for name, attr in (("seed", "seed"), ("run_name", "run_name"),
                       ("output_dir", "out"), ("from_run", "from_run")):
        value = getattr(args, attr, None)
        if value is not None:
            global_cfg[name] = value
    if getattr(args, "resume", False):
        global_cfg["resume"] = True
    if type(global_cfg["resume"]) is not bool:
        raise ValueError("resume must be boolean")
    _integer(global_cfg["seed"], "seed", 0)
    run_name = _string(global_cfg["run_name"], "run_name")
    if run_name in {".", ".."} or Path(run_name).name != run_name or "/" in run_name or "\\" in run_name:
        raise ValueError("run_name must be one directory name")
    output_dir = (_path(global_cfg["output_dir"], base, "output_dir")
                  if global_cfg["output_dir"] is not None
                  else str(base / "inference" / run_name))
    from_run = (_path(global_cfg["from_run"], base, "from_run")
                if global_cfg["from_run"] is not None else None)
    if from_run is not None and not Path(from_run).is_dir():
        raise ValueError(f"from_run does not exist: {from_run}")
    model_arg = getattr(args, "model_dir", None)
    model_root = resolve_model_dir(model_arg, invocation_dir=base)
    generation = copy.deepcopy(global_cfg["generation"])
    for key in _GENERATION_FIELDS - {"config", "override"}:
        value = getattr(args, key, None)
        if value is not None:
            if key == "route" and isinstance(value, str):
                value = value == "true"
            if key == "route_ckpt" and value == "null":
                value = None
            generation[key] = value
    if getattr(args, "override", None):
        generation["override"] = list(getattr(args, "override"))
    generation = _generation(generation, base, model_root)
    worker_gpus, worker_cpus = getattr(args, "worker_gpus", None), getattr(args, "worker_cpus", None)
    if worker_gpus is not None:
        _number(worker_gpus, "--worker-gpus", maximum=1)
    if worker_cpus is not None:
        _number(worker_cpus, "--worker-cpus")
    custom_presets_path = getattr(args, "presets", None)
    custom_presets = {}
    if custom_presets_path:
        custom = _load(Path(_path(custom_presets_path, base, "--presets")))
        _mapping(custom, "--presets file", {"presets"})
        custom_presets = _presets(custom.get("presets"), "custom presets")
    tasks = {}
    profiles = {}
    validation_defaults = _validation_defaults()
    for name in task_names:
        task_override = configured_tasks.get(name, {})
        modalities = _modalities(task_payload[name], name)
        profile = getattr(args, "preset", None)
        if profile is None:
            profile = task_override.get("preset", overlay.get("preset", "auto"))
        profile = _preset_name(profile, modalities, name)
        if profile == "ppi" and len(task_payload[name]["design_chains"]) != 1:
            raise ValueError(f"task {name}: preset ppi requires one protein/peptide design chain")
        if profile not in profiles:
            from .criteria import profile_presets
            preset = _load(inference_config_dir() / "pipeline" / f"{profile}.yaml")
            _check_overlay(preset, f"preset {profile}")
            profiles[profile] = _merge({"presets": profile_presets(profile)}, preset)
        task_cfg = _merge(BUILTINS, profiles[profile])
        task_cfg = _merge(task_cfg, {k: v for k, v in overlay.items() if k != "tasks"})
        # Each round is assembled in precedence order below, so global round
        # rows must remain separate from task-level values.
        global_rounds = task_cfg.pop("rounds", {})
        shared_iteration = copy.deepcopy(task_cfg["iteration"])
        task_cfg = _merge(task_cfg, {k: v for k, v in task_override.items() if k != "rounds"})
        task_cfg["preset"] = profile
        recycle = getattr(args, "recycle", None)
        recycle = task_cfg["recycle"] if recycle is None else recycle
        _integer(recycle, f"task {name}.recycle")
        task_rounds = task_override.get("rounds", {})
        extra_rounds = {int(k) for k in global_rounds} | {_round_number(k) for k in task_rounds}
        if any(number > recycle for number in extra_rounds if number != 1):
            raise ValueError(f"task {name}: round override exceeds recycle={recycle}")
        definitions = _presets(_merge(task_cfg["presets"], custom_presets), f"task {name}.presets")
        from .criteria import include_native_presets
        requested = []
        for block in (task_cfg["iteration"], *global_rounds.values(), *task_rounds.values()):
            requested.extend(block.get("filter", {}).get("criteria") or [])
        requested.extend(task_cfg["final"].get("criteria") or [])
        requested.extend(getattr(args, "criteria", None) or [])
        requested.extend(getattr(args, "final_criteria", None) or [])
        include_native_presets(definitions, requested)
        routes = _criteria_routes(task_cfg["criteria_routes"], definitions)
        rounds = []
        # Task controls follow global round controls, then task-round and CLI.
        task_iteration = task_override.get("iteration", {})
        for number in range(1, recycle + 1):
            row = copy.deepcopy(shared_iteration)
            row = _merge(row, global_rounds.get(number, {}))
            row = _merge(row, task_iteration)
            matching = next((value for key, value in task_rounds.items()
                             if _round_number(key) == number), {})
            row = _merge(row, matching)
            if getattr(args, "method", None) is not None:
                row["refold"]["method"] = args.method
            count_attr = "first_mpnn_count" if number == 1 else "mpnn_count"
            if getattr(args, count_attr, None) is not None:
                row["sequence"]["mpnn"]["count"] = getattr(args, count_attr)
            if getattr(args, "criteria", None) is not None:
                row["filter"]["criteria"] = list(args.criteria)
            if number == 1 and getattr(args, "num_clusters", None) is not None:
                row["selection"] = {"mode": "cluster", "num_clusters": args.num_clusters,
                                    "per_parent": row["selection"].get("per_parent"),
                                    "rank_by": row["selection"].get("rank_by", "design_plddt")}
            if number > 1 and any(getattr(args, key, None) is not None
                                  for key in ("max_pass", "max_pass_ratio")):
                selection = row["selection"]
                if selection.get("mode") != "top":
                    selection = {"mode": "top", "max_pass": 200, "max_pass_ratio": None,
                                 "per_parent": selection.get("per_parent", 1),
                                 "per_family": selection.get("per_family", 2),
                                 "rank_by": selection.get("rank_by", "design_plddt")}
                for key in ("max_pass", "max_pass_ratio"):
                    if getattr(args, key, None) is not None:
                        selection[key] = getattr(args, key)
                row["selection"] = selection
            if number < recycle:
                if getattr(args, "relax", None) is not None:
                    row["transition"]["relax"]["enabled"] = args.relax == "true"
                if getattr(args, "partial_fraction", None) is not None:
                    row["transition"]["partial"].update(
                        enabled=True, fraction=args.partial_fraction)
                if getattr(args, "partial_replicas", None) is not None:
                    row["transition"]["partial"]["replicas"] = args.partial_replicas
            _check_iteration(row, f"task {name}.rounds.{number}")
            refold, sequence = row["refold"], row["sequence"]
            if refold["method"] == "af2_ig" and "nuc" in modalities:
                raise ValueError(f"task {name}: af2_ig does not support nucleic-acid design")
            partial = _partial(row["transition"]["partial"],
                               f"task {name}.rounds.{number}.transition.partial")
            if number == recycle:
                partial["enabled"] = False
                partial.pop("fraction", None)
            normalized_round = {
                "round": number, "method": refold["method"],
                "mpnn_count": sequence["mpnn"]["count"], "samples": refold["conformers"],
                "include_gt": sequence["include_gt"],
                "sequence_source": sequence["mpnn"]["source"],
                "sequence_settings": copy.deepcopy(sequence["mpnn"]["settings"]),
                **copy.deepcopy(row["filter"]),
                "selection": _selection(row["selection"], f"task {name}.rounds.{number}.selection"),
                "relax_enabled": number < recycle and row["transition"]["relax"]["enabled"],
                "partial": partial,
            }
            if number == 1 or number == recycle:
                normalized_round["selection"].pop("per_family", None)
            normalized_round["criteria_route"] = _route(normalized_round["criteria_route"], "criteria_route")
            normalized_round["validation"] = _native_validation(
                refold["validation"], refold["method"], refold["conformers"],
                global_cfg["seed"], base, worker_gpus, worker_cpus,
                defaults=validation_defaults)
            rounds.append(normalized_round)
        final = copy.deepcopy(task_cfg["final"])
        if getattr(args, "final_criteria", None) is not None:
            final["criteria"] = list(args.final_criteria)
        _check_filter({key: value for key, value in final.items() if key in _FILTER_FIELDS},
                      f"task {name}.final")
        final_method = _choice(final["method"], {"af3", "af2_ig"}, f"task {name}.final.method")
        if final_method == "af2_ig" and "nuc" in modalities:
            raise ValueError(f"task {name}: af2_ig does not support nucleic-acid design")
        final["pool"] = _pool(final["pool"], f"task {name}.final.pool")
        final["criteria_route"] = _route(final["criteria_route"], "final.criteria_route")
        final_group = "af2" if final_method == "af2_ig" else "af3"
        final_samples = final["validation"].get(final_group, {}).get("samples", 5)
        final["validation"] = _native_validation(final["validation"], final_method, final_samples,
                                                  global_cfg["seed"], base,
                                                  worker_gpus, worker_cpus,
                                                  defaults=validation_defaults)
        openmm = copy.deepcopy(task_cfg["openmm"])
        _mapping(openmm, f"task {name}.openmm", _OPENMM_FIELDS)
        for key in ("force_field", "charge_method"):
            _string(openmm[key], f"openmm.{key}")
        _choice(openmm["platform"], {"CUDA", "CPU", "OpenCL", "Reference"}, "openmm.platform")
        if type(openmm["use_ray"]) is not bool:
            raise ValueError("openmm.use_ray must be boolean")
        _number(openmm["worker_gpus"], "openmm.worker_gpus", maximum=1)
        _number(openmm["worker_cpus"], "openmm.worker_cpus")
        normalized = {"preset": profile, "recycle": recycle, "rounds": rounds,
                      "openmm": openmm, "final": final,
                      "presets": definitions, "criteria_routes": routes}
        if len(modalities & {"protein", "nuc"}) > 1:
            for row in [*rounds, final]:
                if row["criteria"] is None and row["criteria_route"] == "auto":
                    raise ValueError(f"task {name}: mixed design requires explicit criteria_route "
                                     "or criteria for both iteration and final")
        # Explicit routes/names fail in planning, without waiting for models.
        for stage in [*rounds, final]:
            if stage["criteria"] is not None or stage["criteria_route"] != "auto":
                criteria_for(normalized, "final" if stage is final else stage,
                             stage["criteria_route"])
        tasks[name] = normalized
    return {"task_file": str(task_file), "task_payload": copy.deepcopy(task_payload),
            "task_names": task_names, "run_name": run_name, "output_dir": output_dir,
            "seed": global_cfg["seed"], "model_dir": str(model_root),
            "from_run": from_run, "resume": global_cfg["resume"],
            "generation": generation, "tasks": tasks}
