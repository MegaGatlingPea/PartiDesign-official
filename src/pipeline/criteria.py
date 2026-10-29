"""Pipeline criteria derived from the native validation definitions."""
from __future__ import annotations

import copy

from partidesign.paths import inference_config_dir
from src.validate.filters import load_presets

_NATIVE_PRESETS = (
    "af2_ig_protein",
    "af2_ig_peptide",
    "af3_basic",
    "af3_protein_monomer",
    "af3_na_monomer",
    "af3_protein_na_complex",
    "af3_protein_bind_na",
    "af3_na_bind_protein",
    "af3_protein_bind_ligand",
    "af3_na_bind_ligand",
    "af3_antibody",
    "af3_ame",
)
# Confidence, sequence correctness, and clash thresholds retain native values.
_RELAXED_METRICS = frozenset({
    "bound_unbound_rmsd",
    "design_rmsd",
    "na_token_center_rmsd",
    "complex_ca_p_rmsd",
    "na_aligned_protein_ca_rmsd",
    "protein_aligned_na_p_rmsd",
    "min_design_target_pae_angstrom",
    "binder_design_rmsd",
    "complexa_aligned_ligand_rmsd",
    "na_p_self_rmsd",
    "na_aligned_ligand_rmsd",
    "antibody_antigen_pae_angstrom",
    "antigen_aligned_antibody_ca_rmsd",
    "binder_scRMSD_bb3",
    "motif_rmsd_reference",
})


def _relaxed(preset: dict) -> dict:
    result = copy.deepcopy(preset)
    for rule in result["rules"]:
        if rule["metric"] in _RELAXED_METRICS:
            rule["value"] *= 1.5
    return result


def _protein_binder(preset: dict) -> dict:
    result = copy.deepcopy(preset)
    result["methods"] = ["esmfold2-fast", "af3"]
    thresholds = {
        "binder_iptm": 0.6,
        "binder_ptm": 0.7,
        "complex_design_rmsd": 4.0,
    }
    for rule in result["rules"]:
        if rule["metric"] in thresholds:
            rule["value"] = thresholds[rule["metric"]]
    return result


def profile_presets(profile: str) -> dict:
    """Return independent native definitions with the profile's fixed deltas."""
    if profile not in {"protein", "nuc", "ppi"}:
        raise ValueError(f"Unknown pipeline criteria profile: {profile}")
    native = load_presets(inference_config_dir() / "criteria.yaml")
    if profile == "ppi":
        result = {
            "pipeline_protein_binder": _protein_binder(native["af3_basic"]),
            "pipeline_af2_ig_protein": _relaxed(native["af2_ig_protein"]),
            "pipeline_af2_ig_peptide": _relaxed(native["af2_ig_peptide"]),
            "af2_ig_protdbench": copy.deepcopy(native["af2_ig_protdbench"]),
            "af2_ig_peptide": copy.deepcopy(native["af2_ig_peptide"]),
            "af3_basic": copy.deepcopy(native["af3_basic"]),
        }
        result["af2_ig_protdbench"]["reduction"] = "all_and_mean"
        return result
    result = {name: copy.deepcopy(native[name]) for name in _NATIVE_PRESETS}
    result.update({
        f"pipeline_{name}": _relaxed(native[name])
        for name in _NATIVE_PRESETS
    })
    result["pipeline_protein_binder"] = _protein_binder(native["af3_basic"])
    return result


def include_native_presets(definitions, names):
    """Resolve explicitly requested validate names without changing legacy configs."""
    missing = set(names) - definitions.keys()
    if missing:
        native = load_presets(inference_config_dir() / "criteria.yaml")
        for name in sorted(missing & native.keys()):
            definitions[name] = copy.deepcopy(native[name])
    return definitions


MOTIF_GEOMETRY_METRICS = frozenset({
    "motif_rmsd_reference", "motif_rmsd_design", "motif_rmsd_reference_bb3",
})


def motif_reference_preset(native=None, *, nucleic=False):
    """Use validate's strict motif line, without its unrelated AME rules."""
    if native is None:
        native = load_presets(inference_config_dir() / "criteria.yaml")
    source_name = "af3_na_motif" if nucleic else "af3_ame"
    source = native[source_name]
    rules = [copy.deepcopy(rule) for rule in source["rules"]
             if rule["metric"] == "motif_rmsd_reference"]
    if not rules:
        raise ValueError(f"validate {source_name} must define the strict motif RMSD rule")
    return {"methods": ["esmfold2-fast", "af2_ig", "af3"],
            "reduction": "any_prediction", "rules": rules}


def without_motif_geometry(names, definitions):
    """Leave nonmotif criteria and identity rules active in later iterations."""
    result = dict(definitions)
    kept = []
    for name in names:
        preset = definitions[name]
        rules = [rule for rule in preset["rules"] if rule["metric"] not in MOTIF_GEOMETRY_METRICS]
        if rules:
            result[name] = dict(preset, rules=rules)
            kept.append(name)
    return kept, result
