"""Small JSON helpers for independent postfilter state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any


def _plain_path(path: str | os.PathLike) -> Path:
    path = Path(os.path.abspath(Path(path).expanduser()))
    for item in (*reversed(path.parents), path):
        try:
            mode = item.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"Path has a symlink component: {item}")
    return path


def read_json(path: str | os.PathLike) -> dict | None:
    """Read an ordinary JSON object, returning None only for a missing file."""
    path = _plain_path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(mode):
        raise ValueError(f"State is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_invalid_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot read JSON state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"State must be a JSON object: {path}")
    return value


def _invalid_constant(value: str):
    raise ValueError(f"Non-finite JSON value: {value}")


def atomic_json(path: str | os.PathLike, data: Any) -> None:
    """Publish complete, finite JSON through a same-directory replacement."""
    content = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path = _plain_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _plain_path(path)
    if path.exists() and not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"State is not a regular file: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
