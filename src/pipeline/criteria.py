"""Pipeline criteria derived from the native validation definitions."""
from __future__ import annotations

import copy

from partidesign.paths import inference_config_dir
from src.validate.filters import load_presets
from src.validate.methods import REFOLD_METHODS

_NATIVE_PRESETS = (
    "af2_ig_protein",
    "af2_ig_peptide",
    "af3_basic",
    "af3_peptide",
    "af3_alphaproteo",
    "af3_protein_monomer",
    "af3_na_monomer",
    "af3_protein_na_complex",
    "af3_protein_bind_na",
    "af3_na_bind_protein",
    "af3_protein_bind_ligand",
    "af3_na_bind_ligand",
    "af3_antibody",
    "af3_ame",
    "af3_ame_strict",
    "af3_protein_bind_na_strict",
    "af3_na_bind_protein_strict",
    "af3_protein_bind_ligand_strict",
    "af3_na_bind_ligand_strict",
)
# PAE and geometry upper bounds relax; pTM/ipTM, identity and clash rules stay fixed.
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
    "min_interchain_pae_angstrom",
    "rfd4_min_ipae_angstrom",
    "rfd4_backbone_rmsd",
    "rfd4_target_aligned_binder_rmsd",
    "rfd4_target_backbone_aligned_binder_rmsd",
    "rfd4_backbone_aligned_ligand_rmsd",
    "rfd4_motif_rmsd_reference",
})


def _relaxed(preset: dict) -> dict:
    result = copy.deepcopy(preset)
    for rule in result["rules"]:
        if rule["metric"] in _RELAXED_METRICS:
            rule["value"] *= 1.5
    return result


def _protein_binder(preset: dict) -> dict:
    result = copy.deepcopy(preset)
    result["methods"] = [method for method, spec in REFOLD_METHODS.items() if spec.family != "af2"]
    thresholds = {
        "binder_iptm": 0.6,
        "binder_ptm": 0.7,
        "complex_design_rmsd": 4.0,
    }
    for rule in result["rules"]:
        if rule["metric"] in thresholds:
            rule["value"] = thresholds[rule["metric"]]
    return result


def _peptide_binder(preset: dict) -> dict:
    result = _protein_binder(preset)
    for rule in result["rules"]:
        if rule["metric"] == "rfd4_min_ipae_angstrom":
            rule["value"] = 3.5
    return result


def _af2_antibody(native: dict) -> dict:
    """Keep AF2 protein thresholds and the antibody's geometric epitope check."""
    result = copy.deepcopy(native["af2_ig_protein"])
    geometry = [copy.deepcopy(rule) for rule in native["af3_antibody"]["rules"]
                if rule["metric"] == "rfd4_target_aligned_binder_rmsd"]
    if not geometry:
        raise ValueError("validate af3_antibody must define target-aligned binder RMSD")
    result["rules"].extend(geometry)
    return result


def profile_presets(profile: str) -> dict:
    """Return independent native definitions with the profile's fixed deltas."""
    if profile not in {"protein", "nuc", "ppi", "antibody"}:
        raise ValueError(f"Unknown pipeline criteria profile: {profile}")
    native = load_presets(inference_config_dir() / "criteria.yaml")
    if profile == "ppi":
        result = {
            "pipeline_protein_binder": _protein_binder(native["af3_basic"]),
            "pipeline_peptide_binder": _peptide_binder(native["af3_peptide"]),
            "pipeline_af2_ig_protein": _relaxed(native["af2_ig_protein"]),
            "pipeline_af2_ig_peptide": _relaxed(native["af2_ig_peptide"]),
            "af2_ig_protdbench": copy.deepcopy(native["af2_ig_protdbench"]),
            "af2_ig_peptide": copy.deepcopy(native["af2_ig_peptide"]),
            "af3_basic": copy.deepcopy(native["af3_basic"]),
            "af3_peptide": copy.deepcopy(native["af3_peptide"]),
        }
        result["af2_ig_protdbench"]["reduction"] = "all_and_mean"
        return result
    result = {name: copy.deepcopy(native[name]) for name in _NATIVE_PRESETS}
    result.update({
        f"pipeline_{name}": _relaxed(native[name])
        for name in _NATIVE_PRESETS
    })
    result["pipeline_protein_binder"] = _protein_binder(native["af3_basic"])
    result["pipeline_peptide_binder"] = _peptide_binder(native["af3_peptide"])
    antibody = _af2_antibody(native)
    result["pipeline_af2_ig_antibody"] = _relaxed(antibody)
    result["pipeline_af2_ig_antibody_final"] = antibody
    return result


def include_native_presets(definitions, names):
    """Resolve explicitly requested validate names without changing legacy configs."""
    missing = set(names) - definitions.keys()
    if missing:
        native = load_presets(inference_config_dir() / "criteria.yaml")
        for name in sorted(missing & native.keys()):
            definitions[name] = copy.deepcopy(native[name])
        for name in sorted(missing):
            if name.startswith("pipeline_") and name[9:] in native:
                definitions[name] = _relaxed(native[name[9:]])
    return definitions


MOTIF_GEOMETRY_METRICS = frozenset({
    "motif_rmsd_reference", "motif_rmsd_design", "motif_rmsd_reference_bb3",
    "rfd4_motif_rmsd_reference",
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
    return {"methods": list(REFOLD_METHODS),
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


def with_antibody_contacts(names, definitions):
    """Keep shared contact gates mandatory under either all/any criteria mode.

    Attach the rules to each selected same-prediction criterion, rather than
    allowing a different conformer to satisfy the contact gate independently.
    """
    contacts = definitions["antibody_contacts"]
    names = list(names) or ["antibody_contacts"]
    for name in names:
        if name == "antibody_contacts":
            continue
        preset = definitions[name]
        rules = list(preset["rules"])
        additions = copy.deepcopy(contacts["rules"])
        if (preset["reduction"] in {"any_prediction", "best_ranked"}
                and {rule.get("scope", "complex") for rule in rules} == {"prediction"}):
            # Native prediction-scoped rules still select one complete complex.
            # Mixing scope names would make validate reject this conjunction.
            for rule in additions:
                rule["scope"] = "prediction"
        rules.extend(rule for rule in additions if rule not in rules)
        definitions[name] = dict(preset, rules=rules)
    return names, definitions
