"""Shared resolution of the protein amino-acid omission setting.

``rm_aa`` is the unified spelling used by reward and validation callers.  The
older ``omit_aas`` key remains supported for existing configurations and is
used only when ``rm_aa`` is absent or explicitly ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


STANDARD_PROTEIN_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def resolve_rm_aa(settings: Mapping[str, Any]) -> str:
    """Return the effective protein omission string for one settings mapping.

    An explicit non-``None`` ``rm_aa`` value wins over legacy ``omit_aas``;
    this includes the empty string, which deliberately disables a configured
    legacy omission.  The helper validates only the unified value and leaves
    legacy ``omit_aas`` validation to its existing configuration contract.
    """

    if not isinstance(settings, Mapping):
        raise ValueError("amino-acid omission settings must be a mapping")
    if "rm_aa" in settings and settings["rm_aa"] is not None:
        value = settings["rm_aa"]
        if not isinstance(value, str):
            raise ValueError("rm_aa must be a string or null")
        if any(letter not in STANDARD_PROTEIN_AAS for letter in value):
            raise ValueError(
                "rm_aa must contain standard protein one-letter amino-acid letters"
            )
        return value

    # Validation source settings historically allowed ``omit_aas: null``;
    # preserve that no-omission behavior while leaving non-string legacy
    # values to the existing validation error below.
    legacy = settings.get("omit_aas", "")
    if legacy is None:
        return ""
    if not isinstance(legacy, str):
        raise ValueError("omit_aas must be a string")
    return legacy


__all__ = ["STANDARD_PROTEIN_AAS", "resolve_rm_aa"]
