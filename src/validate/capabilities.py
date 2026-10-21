"""Cheap preflight flags; parse and capability checks happen in the worker."""
from __future__ import annotations


def classify_inputs(plans: list[dict], config: dict) -> None:
    """Mark rows for dispatch without loading every input before model startup.

    Full chemistry, chain and AF2 capability checks are performed once by the
    worker that owns the sample. This keeps startup proportional to the CSV
    inventory and avoids parsing every structure twice.
    """
    del config
    for plan in plans:
        for entry in plan["samples"].values():
            entry["capability"] = (
                {"status": "invalid", "reason": entry["error"]}
                if entry.get("error")
                else {"status": "eligible"}
            )
