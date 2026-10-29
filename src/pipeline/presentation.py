"""Public workflow snapshots and short model-free execution plans."""
from __future__ import annotations
import copy
from pathlib import Path


def _delta(value, baseline):
    if not isinstance(value, dict) or not isinstance(baseline, dict):
        return copy.deepcopy(value)
    if "mode" in value and value.get("mode") != baseline.get("mode"):
        return copy.deepcopy(value)
    return {key: _delta(item, baseline.get(key)) for key, item in value.items()
            if key not in baseline or item != baseline[key]}


def public_config(cfg):
    from .config import BUILTINS, _native_validation, _load, _merge, _modalities
    from .criteria import profile_presets, motif_reference_preset
    from src.validate.filters import load_presets
    from partidesign.paths import inference_config_dir
    native_defaults = {}
    profiles = {}
    native_presets = load_presets(inference_config_dir()/"criteria.yaml")

    def validation_view(stage, profile, final=False):
        method = stage["method"]
        group = {"af2_ig": "af2", "af3": "af3", "esmfold2-fast": "esmfold2"}[method]
        count = stage["validation"][group]["samples"]
        key = profile, method, count, final
        if key not in native_defaults:
            preset = profiles[profile][1]
            default = _merge(BUILTINS, preset)
            overrides = (default["final"]["validation"] if final else
                         default["iteration"]["refold"]["validation"])
            native_defaults[key] = _native_validation(overrides, method, count, cfg["seed"], Path.cwd(), None, None)
        baseline = native_defaults[key]
        settings = stage["validation"]
        differences = _delta(settings, baseline)
        delta = differences.pop(group, {})
        result = {"method": method}
        if final:
            delta["samples"] = count
            if group == "af2": delta["models"] = settings[group]["models"]
        else:
            result["conformers"] = stage["samples"]
        if delta: result[group] = delta
        result.update(differences)
        return result

    def filtering(stage):
        result = {"criteria": copy.deepcopy(stage["criteria"])} if stage["criteria"] is not None else {}
        if stage["criteria_route"] != "auto": result["criteria_route"] = stage["criteria_route"]
        if stage["criteria_mode"] != "all": result["criteria_mode"] = stage["criteria_mode"]
        return result

    def iteration(stage, profile):
        mpnn = {"source": stage["sequence_source"], "count": stage["mpnn_count"]}
        if stage["sequence_settings"]: mpnn["settings"] = copy.deepcopy(stage["sequence_settings"])
        selection = {k: copy.deepcopy(v) for k, v in stage["selection"].items() if k != "rank_by"}
        partial = ({k: copy.deepcopy(v) for k, v in stage["partial"].items() if k != "enabled"}
                   if stage["partial"]["enabled"] else False)
        return {"mpnn": mpnn, "include_gt": stage["include_gt"],
                "refold": validation_view(stage, profile), "filter": filtering(stage),
                "selection": selection, "relax": stage["relax_enabled"], "partial": partial}

    tasks = {}
    for name in cfg["task_names"]:
        settings = cfg["tasks"][name]
        profile = settings["preset"]
        if profile not in profiles:
            profiles[profile] = (profile_presets(profile), _load(inference_config_dir()/"pipeline"/(profile+".yaml")))
        rounds = [iteration(stage, profile) for stage in settings["rounds"]]
        common = copy.deepcopy(rounds[1] if len(rounds) > 1 else rounds[0])
        # Last-round transition is automatically disabled; it is not an override.
        overrides = {}
        for stage, row in zip(settings["rounds"], rounds):
            row = copy.deepcopy(row)
            if stage["round"] == settings["recycle"]:
                row["partial"], row["relax"] = common["partial"], common["relax"]
            delta = _delta(row, common)
            if delta: overrides[stage["round"]] = delta
        final = settings["final"]
        final_view = {**validation_view(final, profile, final=True), **filtering(final),
                      "pool": copy.deepcopy(final["pool"]),
                      "prediction_selection": final["prediction_selection"]}
        flow = {"preset": settings["preset"], "recycle": settings["recycle"],
                "iteration": common, "final": final_view}
        if overrides: flow["rounds"] = overrides
        definitions, preset = profiles[profile]
        custom = {}
        for criterion, value in settings["presets"].items():
            if criterion == "pipeline_motif":
                nucleic = _modalities(cfg["task_payload"][name], name) == {"nuc"}
                source_name = "af3_na_motif" if nucleic else "af3_ame"
                source = settings["presets"].get(source_name, native_presets[source_name])
                if value == motif_reference_preset({source_name: source}, nucleic=nucleic):
                    continue
            if criterion in definitions:
                delta = _delta(value, definitions[criterion])
                if delta: custom[criterion] = delta
            elif criterion not in native_presets or value != native_presets[criterion]:
                custom[criterion] = copy.deepcopy(value)
        if custom: flow["presets"] = custom
        routes = _delta(settings["criteria_routes"], preset["criteria_routes"])
        if routes: flow["criteria_routes"] = routes
        openmm = _delta(settings["openmm"], BUILTINS["openmm"])
        if openmm: flow["openmm"] = openmm
        tasks[name] = flow
    pipeline = {"seed": cfg["seed"], "run_name": cfg["run_name"]}
    if cfg["from_run"]: pipeline["from_run"] = cfg["from_run"]
    generation = {k: copy.deepcopy(v) for k, v in cfg["generation"].items()
                  if k in BUILTINS["generation"] and k not in {"config", "ckpt"}
                  and v != BUILTINS["generation"][k]}
    for key, default in (("config", inference_config_dir()/"design.yaml"),
                         ("ckpt", Path(cfg["model_dir"])/"ckpts"/"partidesign.ckpt")):
        if cfg["generation"][key] != str(default): generation[key] = cfg["generation"][key]
    if generation: pipeline["generation"] = generation
    pipeline["tasks"] = tasks
    return pipeline


def public_tasks(cfg):
    """Snapshot selected native tasks separately from workflow settings."""
    return {name: copy.deepcopy(cfg["task_payload"][name]) for name in cfg["task_names"]}


def plan_view(cfg):
    result = {"output": cfg["output_dir"], "seed": cfg["seed"], "tasks": {}}
    if cfg.get("motif_policy") == "first_and_final":
        result["motif"] = {"applies_to": "coordinate motifs", "check": [1, "final"],
                           "reference": "original motif"}
    for name in cfg["task_names"]:
        task, authored = cfg["tasks"][name], cfg["task_payload"][name]
        if cfg["from_run"]:
            start = {"from_run": cfg["from_run"]}
        elif authored.get("partial"):
            start = {"sample": authored["partial"]["reference"],
                     "fraction": authored["partial"]["fraction"], "replicas": authored["sample_num"]}
        else:
            start = {"sample_num": authored.get("sample_num", 1), "design_chains": authored["design_chains"],
                     "design_length": authored.get("design_length")}
        rows = {}
        for stage in task["rounds"]:
            selection = {k: v for k, v in stage["selection"].items() if k != "rank_by" and v is not None}
            row = {"gt": int(stage["include_gt"]), "mpnn": stage["mpnn_count"],
                   "refold": stage["method"], "conformers": stage["samples"],
                   "filter": stage["criteria"] if stage["criteria"] is not None else "auto",
                   "selection": selection}
            if stage["round"] < task["recycle"]:
                row["relax"] = stage["relax_enabled"]
                row["partial"] = ({k: v for k, v in stage["partial"].items() if k != "enabled"}
                                   if stage["partial"]["enabled"] else False)
            rows[stage["round"]] = row
        final = task["final"]
        group = "af2" if final["method"] == "af2_ig" else "af3"
        final_view = {"method": final["method"], "samples": final["validation"][group]["samples"],
                      "pool": final["pool"], "filter": final["criteria"] if final["criteria"] is not None else "auto"}
        if group == "af2":
            final_view.update(models=final["validation"][group]["models"],
                              require_all_models=True, paired_complex_monomer=True)
        result["tasks"][name] = {"start": start, "preset": task["preset"], "rounds": rows,
                                 "rank_by": "design_plddt", "final": final_view}
        if cfg.get("motif_policy") == "first_and_final":
            result["tasks"][name]["motif_filter"] = copy.deepcopy(task["presets"]["pipeline_motif"]["rules"])
    return result
