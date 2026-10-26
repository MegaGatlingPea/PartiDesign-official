"""Closed postfilter settings, without model imports."""
from __future__ import annotations
import copy
import os
import platform
from pathlib import Path
import shutil
import sys

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
    if path is not None:
        overlay=load_yaml(path)
        unknown=set(overlay)-set(config)
        if unknown:
            raise ValueError("unsupported runtime settings: "+", ".join(sorted(unknown))+
                             "; scientific defaults belong in postfilter_presets.yaml")
        config.update(overlay)
    if config["workers"] is not None and (type(config["workers"]) is not int or config["workers"]<1):
        raise ValueError("workers must be a positive integer or null for automatic CPU allocation")
    for name in ("threads","timeout"):
        if type(config[name]) is not int or config[name]<1: raise ValueError(f"{name} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"]<0: raise ValueError("seed must be a nonnegative integer")
    for name in ("dalphaball","dssp"):
        value=config[name]
        if value is not None and (not isinstance(value,str) or not value.strip()): raise ValueError(f"{name} must be a path or null")
        config[name]=resolve_tool(name,value)
    return config

def dependency_status(config,required_tools=None):
    import importlib.util
    result={"pyrosetta":"available" if importlib.util.find_spec("pyrosetta") else "missing"}
    for name in sorted(set(required_tools) if required_tools is not None else {"dalphaball","dssp"}):
        value=config.get(name)
        result[name]="available" if value and Path(value).is_file() and os.access(value,os.X_OK) else "missing"
    return result
