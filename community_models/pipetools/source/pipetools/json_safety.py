from __future__ import annotations

import math
from typing import Any


def contains_nonfinite(value: Any) -> bool:
    """Return whether a JSON-like value contains NaN or infinity."""

    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_nonfinite(item) for item in value)
    return False


def normalize_nonfinite(
    value: Any,
    *,
    path: str = "",
) -> tuple[Any, list[dict[str, str]]]:
    """Replace non-finite JSON numbers with null and return exact JSON pointers.

    Fold backends can report undefined chain-pair confidence cells as NaN.
    JSON's non-standard NaN token is unsuitable for private worker state,
    but dropping the surrounding matrix would also discard valid confidence.
    This helper preserves the complete shape, substitutes ``None`` only for the
    undefined cells, and records why each cell is unavailable.
    """

    unavailable: list[dict[str, str]] = []

    def visit(item: Any, pointer: str) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            if math.isnan(item):
                kind = "NaN"
            elif item > 0:
                kind = "Infinity"
            else:
                kind = "-Infinity"
            unavailable.append(
                {
                    "path": pointer or "/",
                    "reason": f"upstream confidence value is non-finite ({kind})",
                }
            )
            return None
        if isinstance(item, dict):
            return {
                key: visit(child, f"{pointer}/{_escape_pointer(str(key))}")
                for key, child in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [
                visit(child, f"{pointer}/{index}")
                for index, child in enumerate(item)
            ]
        return item

    return visit(value, path), unavailable


def _escape_pointer(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")
