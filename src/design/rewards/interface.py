"""Shared fixed-interface position selection for polymer sequence design.

The interface producer deliberately operates on the candidate AtomArray and
returns only stable ``(chain_id, one_based_token_position)`` entries.  It does
not alter coordinates or any of the existing hard-condition annotations.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from typing import Any

import numpy as np


_DEFAULT_CUTOFF = 6.0
_ALLOWED_KEYS = frozenset({"enabled", "cutoff", "target_chains"})
_POLYMER_MODALITIES = frozenset({"protein", "rna", "dna"})
_ION_ELEMENTS = frozenset(
    {
        "LI", "NA", "K", "RB", "CS",
        "MG", "CA", "SR", "BA",
        "AL", "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI",
        "CU", "ZN", "Y", "ZR", "NB", "MO", "RU", "RH", "PD",
        "AG", "CD", "IN", "SN", "SB", "HG", "PB", "BI",
        "F", "CL", "BR", "I",
    }
)
_ION_NAMES = _ION_ELEMENTS | frozenset(
    {"SOD", "POT", "CAL", "MAG", "ZIN", "FER", "MN2", "FE2", "FE3"}
)


def normalize_fixed_interface(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate and normalize the closed fixed-interface configuration.

    ``None`` is the canonical disabled value.  A mapping must explicitly state
    ``enabled``; cutoff and target-chain selection have stable defaults.  The
    returned dictionary is a detached, ordinary mapping so it can be stored in
    an OmegaConf-resolved configuration without retaining caller-owned lists.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("fixed_interface must be null or a mapping")
    unknown = sorted(
        set(value).difference(_ALLOWED_KEYS),
        key=lambda item: str(item),
    )
    if unknown:
        raise ValueError(f"fixed_interface has unknown keys: {unknown}")
    if "enabled" not in value or not isinstance(value["enabled"], bool):
        raise ValueError("fixed_interface.enabled must be a boolean")

    cutoff = value.get("cutoff", _DEFAULT_CUTOFF)
    if isinstance(cutoff, bool) or not isinstance(cutoff, Real):
        raise ValueError("fixed_interface.cutoff must be numeric")
    cutoff = float(cutoff)
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("fixed_interface.cutoff must be finite and > 0")

    raw_targets = value.get("target_chains")
    if raw_targets is not None:
        # OmegaConf exposes resolved list fields as ListConfig while direct
        # callers use a plain list.  Keep tuples and other arbitrary
        # sequences out of the public schema.
        if not isinstance(raw_targets, list) and type(raw_targets).__name__ != "ListConfig":
            raise ValueError("fixed_interface.target_chains must be a list or null")
        targets: list[str] = []
        seen: set[str] = set()
        for chain in list(raw_targets):
            if (
                not isinstance(chain, str)
                or not chain
                or chain != chain.strip()
                or any(character.isspace() for character in chain)
                or chain in {".", "?"}
                or ":" in chain
            ):
                raise ValueError(
                    "fixed_interface.target_chains entries must be non-empty chain IDs"
                )
            if chain in seen:
                raise ValueError(
                    f"fixed_interface.target_chains contains duplicate {chain!r}"
                )
            seen.add(chain)
            targets.append(chain)
    else:
        targets = None

    return {
        "enabled": bool(value["enabled"]),
        "cutoff": cutoff,
        "target_chains": targets,
    }


def _candidate_atoms(candidate: Any) -> Any:
    if isinstance(candidate, Mapping):
        atoms = candidate.get("atoms")
    else:
        atoms = getattr(candidate, "atoms", None)
    if atoms is None:
        raise ValueError("fixed-interface selection requires candidate.atoms")
    return atoms


def _required_array(atoms: Any, name: str, count: int) -> np.ndarray:
    try:
        values = np.asarray(getattr(atoms, name))
    except AttributeError as exc:
        raise ValueError(f"fixed-interface candidate lacks {name!r} annotation") from exc
    if values.shape != (count,):
        raise ValueError(f"fixed-interface annotation {name!r} must have shape [{count}]")
    return values


def _chain_order(chain_ids: np.ndarray) -> tuple[str, ...]:
    result: list[str] = []
    for value in chain_ids.astype(str).tolist():
        chain = str(value)
        if chain not in result:
            result.append(chain)
    return tuple(result)


def _is_ion_chain(
    rows: np.ndarray,
    *,
    elements: np.ndarray,
    heavy: np.ndarray,
    residue_names: np.ndarray | None = None,
) -> bool:
    heavy_rows = rows[heavy[rows]]
    if len(heavy_rows) != 1:
        return False
    row = heavy_rows[0]
    element = str(elements[row]).strip().upper()
    residue = "" if residue_names is None else str(residue_names[row]).strip().upper()
    return element in _ION_ELEMENTS or residue in _ION_NAMES


def _valid_heavy_rows(
    rows: np.ndarray,
    *,
    heavy: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    return rows[heavy[rows] & valid[rows]]


def interface_fixed_positions(
    candidate: Any,
    design_chain: str | None,
    config: Mapping[str, Any] | None,
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...]]:
    """Return interface-fixed design positions and resolved target chains.

    A position is fixed when any valid heavy atom in its design token is at or
    below the inclusive cutoff from any valid heavy atom in a selected target
    chain.  Auto targets use molecular partner chains, omit ions, and retain a
    joint ligand even when that ligand is marked as a design entity.
    """

    normalized = normalize_fixed_interface(config)
    if normalized is None or not normalized["enabled"]:
        return (), ()
    if not isinstance(design_chain, str) or not design_chain:
        raise ValueError("fixed-interface selection requires one design chain")

    atoms = _candidate_atoms(candidate)
    count = len(atoms)
    if count == 0:
        raise ValueError("fixed-interface selection requires a non-empty candidate")
    chain_ids = _required_array(atoms, "chain_id", count).astype(str)
    internal_chain_ids = chain_ids
    public = getattr(atoms, "label_asym_id", None)
    if public is not None:
        public = _required_array(atoms, "label_asym_id", count).astype(str)
        missing = np.isin(public, ("", ".", "?"))
        if not missing.all():
            if missing.any():
                raise ValueError("fixed-interface public chain labels are partially missing")
            chain_ids = public
    aliases: dict[str, set[str]] = {}
    for internal, public_chain in zip(internal_chain_ids, chain_ids, strict=True):
        aliases.setdefault(str(internal), set()).add(str(public_chain))
    for public_chain in _chain_order(chain_ids):
        if len(set(internal_chain_ids[chain_ids == public_chain])) != 1:
            raise ValueError("fixed-interface public chain label has ambiguous internal aliases")

    def resolve_chain(name: str) -> str:
        choices = set(aliases.get(name, ()))
        if name in set(chain_ids):
            choices.add(name)
        if len(choices) != 1:
            raise ValueError(f"fixed-interface chain {name!r} does not resolve uniquely")
        return next(iter(choices))

    planner_design_chain = design_chain
    design_chain = resolve_chain(design_chain)
    token_values = _required_array(atoms, "token_uid", count)
    if token_values.dtype.kind not in "iu":
        raise ValueError("fixed-interface token_uid must be integer-valued")
    token_uids = np.asarray(token_values, dtype=np.int64)
    if (token_uids < 0).any():
        raise ValueError("fixed-interface token_uid must be non-negative")
    modalities = _required_array(atoms, "mol_type", count).astype(str)
    modalities = np.char.lower(np.char.strip(modalities))
    residue_names = (
        _required_array(atoms, "res_name", count).astype(str)
        if hasattr(atoms, "res_name")
        else None
    )
    residue_uids = (
        _required_array(atoms, "residue_uid", count)
        if hasattr(atoms, "residue_uid")
        else None
    )
    is_design = np.asarray(
        getattr(atoms, "is_design", np.ones(count, dtype=bool)), dtype=bool
    )
    if is_design.shape != (count,):
        raise ValueError(f"fixed-interface annotation 'is_design' must have shape [{count}]")
    try:
        coordinates = np.asarray(atoms.coord, dtype=np.float64)
    except AttributeError as exc:
        raise ValueError("fixed-interface candidate lacks 'coord'") from exc
    if coordinates.shape != (count, 3):
        raise ValueError(f"fixed-interface coordinates must have shape [{count}, 3]")
    raw_valid = getattr(atoms, "coord_valid", np.ones(count, dtype=bool))
    valid = np.asarray(raw_valid, dtype=bool)
    if valid.shape != (count,):
        raise ValueError(f"fixed-interface annotation 'coord_valid' must have shape [{count}]")
    if bool((valid & ~np.isfinite(coordinates).all(axis=1)).any()):
        raise ValueError("fixed-interface valid coordinates must be finite")
    finite = np.isfinite(coordinates).all(axis=1)
    if hasattr(atoms, "element"):
        elements = np.char.upper(np.char.strip(np.asarray(atoms.element).astype(str)))
        if elements.shape != (count,):
            raise ValueError(f"fixed-interface annotation 'element' must have shape [{count}]")
        heavy = ~np.isin(elements, ("H", "D", "T"))
    else:
        elements = np.full(count, "", dtype="<U1")
        heavy = np.ones(count, dtype=bool)
    valid &= finite

    chains = _chain_order(chain_ids)
    design_rows = np.flatnonzero(chain_ids == design_chain)
    if not len(design_rows):
        raise ValueError(f"fixed-interface design chain {design_chain!r} does not resolve")
    design_modalities = set(modalities[design_rows].tolist())
    if design_modalities not in ({"protein"}, {"rna"}, {"dna"}):
        raise ValueError(
            f"fixed-interface design chain {design_chain!r} must be one polymer modality"
        )

    # Match the planner's token ordering: token UIDs are stable and positions
    # are counted independently within each polymer chain.
    token_rows: dict[int, np.ndarray] = {}
    for raw_uid in sorted(set(int(value) for value in token_uids.tolist())):
        rows = np.flatnonzero(token_uids == raw_uid)
        row_chains = set(chain_ids[rows].tolist())
        row_modalities = set(modalities[rows].tolist())
        if len(row_chains) != 1 or len(row_modalities) != 1:
            raise ValueError("fixed-interface token spans ambiguous chain or modality")
        if residue_names is not None and len(set(residue_names[rows].tolist())) != 1:
            raise ValueError("fixed-interface token spans ambiguous residue identity")
        if residue_uids is not None and len(set(int(value) for value in residue_uids[rows])) != 1:
            raise ValueError("fixed-interface token spans ambiguous residue UID")
        token_design = is_design[rows]
        if bool(token_design.any()) != bool(token_design.all()):
            raise ValueError("fixed-interface token has partial design annotation")
        token_rows[raw_uid] = rows

    design_tokens: list[tuple[int, int, np.ndarray]] = []
    position = 0
    for token_uid in sorted(token_rows):
        rows = token_rows[token_uid]
        if str(chain_ids[rows[0]]) != design_chain:
            continue
        position += 1
        if bool(is_design[rows].all()):
            heavy_rows = _valid_heavy_rows(rows, heavy=heavy, valid=valid)
            if not len(heavy_rows):
                raise ValueError(
                    f"fixed-interface design token {design_chain}:{position} has no valid heavy coordinates"
                )
            design_tokens.append((token_uid, position, heavy_rows))
    if not design_tokens:
        raise ValueError(
            f"fixed-interface design chain {design_chain!r} has no design tokens"
        )

    def molecular_target_rows(chain: str) -> np.ndarray:
        rows = np.flatnonzero(chain_ids == chain)
        eligible = (
            np.isin(modalities[rows], tuple(_POLYMER_MODALITIES))
            & ~is_design[rows]
        ) | (modalities[rows] == "ligand")
        # Mixed molecular context can share a source chain. Exclude ions per
        # token rather than treating the entire chain as one chemical entity.
        for uid in set(token_uids[rows].tolist()):
            token = token_rows[int(uid)]
            if modalities[token[0]] == "ligand" and _is_ion_chain(
                token, elements=elements, heavy=heavy, residue_names=residue_names
            ):
                eligible[np.isin(rows, token)] = False
        return _valid_heavy_rows(rows[eligible], heavy=heavy, valid=valid)

    requested_targets = normalized["target_chains"]
    if requested_targets is None:
        target_chains = [
            chain for chain in chains
            if chain != design_chain and len(molecular_target_rows(chain))
        ]
    else:
        target_chains = [resolve_chain(chain) for chain in requested_targets]
        if len(set(target_chains)) != len(target_chains):
            raise ValueError("fixed-interface target aliases resolve to duplicate chains")
        for chain in target_chains:
            if chain == design_chain:
                raise ValueError("fixed-interface target_chains cannot include the design chain")
            if not len(molecular_target_rows(chain)):
                raise ValueError(
                    f"fixed-interface target chain {chain!r} is an ion, non-molecular, "
                    "or has no valid heavy context coordinates"
                )

    target_rows = [molecular_target_rows(chain) for chain in target_chains]
    if target_rows:
        target_indices = np.concatenate(target_rows, axis=0)
        target_coordinates = coordinates[target_indices]
    else:
        target_coordinates = np.empty((0, 3), dtype=np.float64)

    cutoff_squared = float(normalized["cutoff"]) ** 2
    positions: list[tuple[str, int]] = []
    for _token_uid, token_position, rows in design_tokens:
        if len(target_coordinates) == 0:
            continue
        distances = coordinates[rows, None, :] - target_coordinates[None, :, :]
        if float(np.min(np.sum(distances * distances, axis=2))) <= cutoff_squared:
            positions.append((planner_design_chain, token_position))
    return tuple(positions), tuple(target_chains)


__all__ = ["interface_fixed_positions", "normalize_fixed_interface"]
