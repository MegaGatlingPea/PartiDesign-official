"""Public polymer token-removal names, independent of model execution."""
from __future__ import annotations

from typing import Any

from ..model.constants import TOKEN_TYPE32
from .records import DesignConfigError


def normalize_rm_token_type(value: Any, where: str = "rm_token_type") -> tuple[str, ...] | None:
    """Resolve canonical token names; null keeps the task-specific policy."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise DesignConfigError(f"{where} must be null or a list of canonical token names")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise DesignConfigError(f"{where}[{index}] must be a non-empty token name")
        name = item.strip().upper()
        if name not in TOKEN_TYPE32:
            raise DesignConfigError(f"{where}[{index}] unknown token name {item!r}")
        if name not in result:
            result.append(name)
    return tuple(result)


def normalize_task_rm_token_type(value: Any, where: str = "rm_token_type") -> tuple[str, ...]:
    """Public tasks exclude free protein Cys unless explicitly overridden."""
    policy = normalize_rm_token_type(value, where)
    return ("CYS",) if policy is None else policy
