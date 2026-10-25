"""Strict per-refold rules; predictive confidence stays with validation."""
from __future__ import annotations
import math
import operator
from pathlib import Path

BC1_METRICS = frozenset("bc1_" + n for n in (
    "binder_energy_score", "surface_hydrophobicity", "shape_complementarity", "dg", "dsasa",
    "interface_residues", "interface_hbonds", "interface_unsat_hbonds", "binder_loop_percent",
    "binder_helix_percent", "binder_sheet_percent", "packstat", "dg_dsasa", "interface_hydrophobicity"))
BC2_METRICS = frozenset("bc2_" + n for n in (
    "backbone_clashes", "all_atom_clashes", "interface_residues", "interface_buried_area",
    "interface_buried_area_fraction", "surface_hydrophobicity", "interface_hydrophobicity",
    "binder_chain_breaks", "binder_helix_fraction", "binder_beta_sheet_fraction", "binder_loop_fraction",
    "interface_helix_fraction", "interface_beta_sheet_fraction", "interface_loop_fraction", "exposed_loop_fraction",
    "terminus_exposure", "binder_length", "binder_mass_kda", "binder_pi", "binder_net_charge",
    "binder_cysteines", "receptor_chains_contacted")) | frozenset("bc2_interface_aa_"+a for a in "ACDEFGHIKLMNPQRSTVWY")
METHOD_METRICS = {"rosetta": BC1_METRICS, "bc2": BC2_METRICS}
OPS = {">":operator.gt, ">=":operator.ge, "<":operator.lt, "<=":operator.le, "==":operator.eq}

def finite_number(value):
    return type(value) in (int,float) and math.isfinite(value)

def load_presets(path: Path):
    from .config import load_yaml
    data=load_yaml(path)
    if set(data)!={"presets"} or not isinstance(data["presets"],dict) or not data["presets"]:
        raise ValueError("postfilter presets require a nonempty presets mapping")
    for name,preset in data["presets"].items():
        if not isinstance(name,str) or not name or not isinstance(preset,dict):
            raise ValueError("every preset needs a name and mapping")
        if set(preset)-{"methods","reduction","rules","description"}:
            raise ValueError(f"unknown preset fields: {name}")
        methods=preset.get("methods")
        if not isinstance(methods,list) or not methods or any(m not in METHOD_METRICS for m in methods) or len(set(methods))!=len(methods):
            raise ValueError(f"invalid postfilter methods: {name}")
        if preset.get("reduction")!="all":
            raise ValueError(f"postfilter requires per-refold reduction all: {name}")
        allowed=set().union(*(METHOD_METRICS[m] for m in methods))
        if not isinstance(preset.get("rules"),list) or not preset["rules"]:
            raise ValueError(f"preset needs rules: {name}")
        for rule in preset["rules"]:
            if not isinstance(rule,dict) or set(rule)-{"metric","op","value","scope"}:
                raise ValueError(f"invalid rule fields: {name}")
            if rule.get("metric") not in allowed or rule.get("op") not in OPS or rule.get("scope","complex")!="complex" or not finite_number(rule.get("value")):
                raise ValueError(f"invalid metric/operator/threshold/scope: {name}: {rule}")
    return data["presets"]

def validate_criteria(names,presets):
    names=list(dict.fromkeys(names))
    if not names: raise ValueError("postfilter requires a criterion")
    for name in names:
        if name not in presets: raise ValueError(f"unknown postfilter criterion: {name}")
    return names

def evaluate(metrics,name,preset):
    missing,failed=[],[]
    for rule in preset["rules"]:
        key=rule["metric"]; value=metrics.get(key)
        if not finite_number(value): missing.append(f"required metric unavailable: {key}")
        elif not OPS[rule["op"]](value,rule["value"]): failed.append(f"{key}={value} does not satisfy {rule['op']} {rule['value']}")
    return {"criteria":name,"status":"unavailable" if missing else "failed" if failed else "passed","reasons":missing+failed}
