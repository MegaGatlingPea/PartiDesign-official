"""Atomic publication of small postfilter records."""
import json
import os
from pathlib import Path
import tempfile

def read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as stream: value=json.load(stream)
    except FileNotFoundError: return None
    if not isinstance(value,dict): raise ValueError(f"State must be a JSON object: {path}")
    return value

def atomic_text(path,content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(descriptor,"w",encoding="utf-8") as stream: stream.write(content)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def atomic_json(path,data):
    atomic_text(path,json.dumps(data,allow_nan=False,separators=(",",":"))+"\n")
