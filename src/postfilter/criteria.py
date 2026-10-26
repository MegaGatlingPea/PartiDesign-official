"""Strict per-refold rules; predictive confidence stays with validation."""
from __future__ import annotations
import math
import operator
from pathlib import Path

BC1_METRICS = frozenset("bc1_"+n for n in (
    "binder_energy_score","surface_hydrophobicity","shape_complementarity","dg","dsasa",
    "interface_residues","interface_hbonds","interface_unsat_hbonds","binder_loop_percent"))
BC2_METRICS = frozenset(("bc2_backbone_clashes","bc2_interface_residues"))
METHOD_METRICS = {"rosetta": BC1_METRICS, "bc2": BC2_METRICS}
OPS = {">":operator.gt, ">=":operator.ge, "<":operator.lt, "<=":operator.le, "==":operator.eq}

def finite_number(value):
    return type(value) in (int,float) and math.isfinite(value)

def load_presets(path: Path):
    from .config import load_yaml
    data=load_yaml(path)
    if set(data)!={"defaults","protocol","presets"} or not isinstance(data["presets"],dict) or not data["presets"]:
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
    defaults=data["defaults"]
    if not isinstance(defaults,dict) or set(defaults)!={"protein","peptide"}:
        raise ValueError("defaults must declare protein and peptide preset lists")
    for names in defaults.values():
        if not isinstance(names,list) or not names or any(not isinstance(n,str) for n in names):
            raise ValueError("default criteria must be nonempty preset name lists")
        validate_criteria(names,data["presets"])
    protocol=data["protocol"]
    if not isinstance(protocol,dict) or set(protocol)!={"scorefunction","relax"} or protocol["scorefunction"]!="beta_nov16":
        raise ValueError("protocol requires the BC1 beta_nov16 scorefunction and relax settings")
    relax=protocol["relax"]
    if not isinstance(relax,dict) or set(relax)!={"max_iter","repeats"} or any(type(v) is not int or v<1 for v in relax.values()):
        raise ValueError("relax requires positive max_iter and repeats")
    return data

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

def profile_for_kind(kind,bundle):
    if kind not in bundle["defaults"]: raise ValueError(f"unsupported binder kind: {kind}")
    names=list(bundle["defaults"][kind])
    required={rule["metric"] for name in names for rule in bundle["presets"][name]["rules"]}
    return names,required
