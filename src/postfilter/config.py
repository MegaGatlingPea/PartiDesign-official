"""Closed postfilter settings, without model imports."""
from __future__ import annotations
import copy
import os
import platform
from pathlib import Path
import shutil
import sys

DEFAULT_CRITERIA=("bc1_protein_structural","bc2_protein_geometry")

def load_yaml(path):
    import yaml
    class UniqueLoader(yaml.SafeLoader): pass
    def mapping(loader,node,deep=False):
        result={}
        for k,v in node.value:
            key=loader.construct_object(k,deep=deep)
            if not isinstance(key,str) or key in result: raise ValueError(f"duplicate or nonstring YAML key: {key!r}")
            result[key]=loader.construct_object(v,deep=deep)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping)
    value=yaml.load(Path(path).read_text(),Loader=UniqueLoader)
    if not isinstance(value,dict): raise ValueError(f"configuration must be a mapping: {path}")
    return value

def _merge(base,overlay):
    for key,value in overlay.items():
        if key not in base: raise ValueError(f"unknown postfilter setting: {key}")
        if isinstance(base[key],dict) and key!="path_mappings":
            if not isinstance(value,dict): raise ValueError(f"setting must be a mapping: {key}")
            _merge(base[key],value)
        else: base[key]=value

def resolve_tool(name,configured=None):
    if configured: return str(Path(configured).expanduser().absolute())
    from partidesign.paths import resource_root
    names=("mkdssp","dssp") if name=="dssp" else ("DAlphaBall.gcc","DAlphaBall")
    roots=[Path(sys.executable).parent]
    if platform.system()=="Linux" and platform.machine().lower() in {"x86_64","amd64"}:
        roots.insert(0,resource_root()/"community_models"/"rosetta"/"bin")
    for root in roots:
        for filename in names:
            path=root/filename
            if path.is_file() and os.access(path,os.X_OK): return str(path)
    return next((found for n in names if (found:=shutil.which(n))),None)

def load_config(path=None):
    from partidesign.paths import inference_config_dir
    config=copy.deepcopy(load_yaml(inference_config_dir()/"postfilter.yaml"))
    if path is not None: _merge(config,load_yaml(path))
    for name in ("workers","threads","timeout"):
        if type(config[name]) is not int or config[name]<1: raise ValueError(f"{name} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"]<0: raise ValueError("seed must be nonnegative integer")
    if config["scorefunction"]!="beta_nov16": raise ValueError("only the BC1 beta_nov16 protocol is supported")
    for name in ("max_iter","repeats"):
        if type(config["relax"][name]) is not int or config["relax"][name]<1: raise ValueError(f"relax.{name} must be positive")
    if type(config["relax"]["constrain_start"]) is not bool: raise ValueError("relax.constrain_start must be boolean")
    mappings=config["path_mappings"]
    if not isinstance(mappings,dict) or any(not isinstance(k,str) or not isinstance(v,str) or not Path(k).is_absolute() or not Path(v).is_absolute() for k,v in mappings.items()):
        raise ValueError("path_mappings require explicit absolute old and new roots")
    for tool in ("dalphaball","dssp"):
        value=config[tool]
        if value is not None and (not isinstance(value,str) or not value.strip()): raise ValueError(f"{tool} must be a path or null")
        config[tool]=resolve_tool(tool,value)
    return config

def dependency_status(config):
    import importlib.util
    result={"pyrosetta":"available" if importlib.util.find_spec("pyrosetta") else "missing"}
    for name in ("dalphaball","dssp"):
        value=config.get(name)
        result[name]="available" if value and Path(value).is_file() and os.access(value,os.X_OK) else "missing"
    return result
