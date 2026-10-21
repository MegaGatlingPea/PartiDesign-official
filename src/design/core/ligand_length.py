"""Resolve pocket-based ligand atom counts for standalone task requests.

The resolver is intentionally outside the compiler.  It reads the complete
normalized input only to estimate a ligand size, then replaces the parsed
sentinel with an ordinary integer ``LengthPlan`` before model/runtime setup or
Best-of-N replica expansion.  The model-facing crop remains untouched.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .records import (
    DesignInputError,
    DesignSampleError,
    ExpandedRequest,
    LengthPlan,
)


_HYDROGEN_ELEMENTS = frozenset({"H", "D", "T"})
_HOTSPOT_RADIUS_ANGSTROM = 8.0
_PAIR_BLOCK_SIZE = 2048
_GeometryKey = tuple[str | None, str, str | None, str | None]
_Geometry = tuple[int, float, str]
_FULL_INPUT_SOURCE = "full_input"
_HOTSPOT_SOURCE = "full_input_hotspot_8A"


def base_request_seed(global_seed: int, request: ExpandedRequest) -> int:
    """Derive a repeatable seed from the global seed and base task slot."""

    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")
    if global_seed < 0:
        raise ValueError("global_seed must be non-negative")
    return int((global_seed + int(request.task_slot_ordinal) * 1000003) % (2**63))


def _source_design_exclusion(source: Any, chain_id: str) -> np.ndarray:
    try:
        chains = np.asarray(source.chain_id).astype(str)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DesignInputError("normalized input is missing valid chain identifiers") from exc
    return chains == str(chain_id)


def _finite_heavy_mask(source: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        coords = np.asarray(source.coord, dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DesignInputError(
            "normalized input coordinates must be a finite Nx3 array"
        ) from exc
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise DesignInputError("normalized input coordinates must have shape (N, 3)")
    try:
        elements = np.asarray(source.element).astype(str)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DesignInputError("normalized input is missing valid atom elements") from exc
    if elements.ndim != 1 or elements.shape[0] != coords.shape[0]:
        raise DesignInputError("normalized input atom elements do not match coordinates")
    try:
        coord_valid = np.asarray(source.coord_valid, dtype=bool)
    except (AttributeError, TypeError, ValueError):
        coord_valid = np.ones(coords.shape[0], dtype=bool)
    if coord_valid.ndim != 1 or coord_valid.shape[0] != coords.shape[0]:
        raise DesignInputError("normalized input coordinate validity does not match atoms")
    normalized_elements = np.char.upper(np.char.strip(elements))
    finite = np.isfinite(coords).all(axis=1)
    heavy = ~np.isin(normalized_elements, list(_HYDROGEN_ELEMENTS))
    return coords, finite & coord_valid & heavy


def _hotspot_mask(
    source: Any,
    *,
    selector: str,
    design_excluded: np.ndarray,
    finite_heavy: np.ndarray,
) -> np.ndarray:
    from .crop import source_selector

    selected_residues = source_selector(source, selector)
    if not selected_residues:
        raise DesignSampleError("hotspot selector matched no input residues")
    try:
        residue_uids = np.asarray(source.residue_uid, dtype=np.int64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DesignInputError("normalized input is missing residue identifiers") from exc
    if residue_uids.shape != design_excluded.shape:
        raise DesignInputError("normalized input residue identifiers do not match atoms")
    selected = np.isin(residue_uids, np.asarray(selected_residues, dtype=np.int64))
    if not np.any(selected):
        raise DesignSampleError("hotspot selector matched no input atoms")
    if np.any(selected & design_excluded):
        raise DesignSampleError("hotspot may select context residues only")
    result = selected & finite_heavy
    if not np.any(result):
        raise DesignInputError("hotspot contains no finite heavy atoms")
    return result


def _within_hotspot_radius(
    context_coords: np.ndarray,
    hotspot_coords: np.ndarray,
) -> np.ndarray:
    """Select context atoms within 8 A of any selected hotspot heavy atom."""

    radius_sq = _HOTSPOT_RADIUS_ANGSTROM**2
    result = np.zeros(len(context_coords), dtype=bool)
    # Keep both dimensions of the pairwise temporary bounded for a full-input
    # source with many receptor atoms; selection is an atom union, not a
    # residue expansion.
    block_size = int(_PAIR_BLOCK_SIZE)
    if block_size <= 0:
        raise RuntimeError("hotspot pair block size must be positive")
    for start in range(0, len(context_coords), block_size):
        stop = min(start + block_size, len(context_coords))
        local = np.zeros(stop - start, dtype=bool)
        for hotspot_start in range(0, len(hotspot_coords), block_size):
            hotspot_stop = min(hotspot_start + block_size, len(hotspot_coords))
            delta = (
                context_coords[start:stop, None, :]
                - hotspot_coords[None, hotspot_start:hotspot_stop, :]
            )
            distances_sq = np.einsum("ijk,ijk->ij", delta, delta, dtype=np.float64)
            local |= np.any(distances_sq <= radius_sq, axis=1)
            if bool(local.all()):
                break
        result[start:stop] = local
    return result


def _pocket_coordinates(request: ExpandedRequest, source: Any) -> tuple[np.ndarray, str]:
    chain_id = request.task.length_distribution_chain
    if chain_id is None:
        raise DesignSampleError("request does not select a ligand length distribution")
    coords, finite_heavy = _finite_heavy_mask(source)
    design_excluded = _source_design_exclusion(source, chain_id)
    context_mask = finite_heavy & ~design_excluded
    if not np.any(context_mask):
        raise DesignInputError("ligand pocket is empty: input has no context heavy atoms")

    if request.task.hotspot is None:
        return coords[context_mask], _FULL_INPUT_SOURCE

    hotspot_mask = _hotspot_mask(
        source,
        selector=request.task.hotspot,
        design_excluded=design_excluded,
        finite_heavy=finite_heavy,
    )
    context_coords = coords[context_mask]
    selected = _within_hotspot_radius(context_coords, coords[hotspot_mask])
    pocket = context_coords[selected]
    if not len(pocket):
        raise DesignInputError("hotspot pocket is empty: no context heavy atom within 8 A")
    return pocket, _HOTSPOT_SOURCE


def _read_full_input(request: ExpandedRequest, ccd_path: Path | None) -> Any:
    from ..utils.reader import read_structure

    input_path = request.task.input_path
    if input_path is None:
        raise DesignInputError(
            "design_length.from_distribution requires an input structure for pocket sizing"
        )
    declared_modalities = {
        chain.chain_id: chain.modality
        for chain in request.task.design_chains
        if chain.modality in {"protein", "rna", "dna"}
    }
    return read_structure(
        Path(input_path),
        ccd_path=ccd_path,
        declared_polymer_chain_modalities=declared_modalities,
    )


def _manual_metadata(request: ExpandedRequest) -> dict[str, Any]:
    metadata = dict(request.length_metadata)
    chain = next(
        (item for item in request.task.design_chains if item.modality == "ligand"),
        None,
    )
    if chain is None:
        return metadata
    raw_count = request.plan.lengths.get(chain.chain_id)
    if isinstance(raw_count, bool) or not isinstance(raw_count, (int, np.integer)):
        return metadata
    metadata.update(
        {
            "ligand_length_source": "manual",
            "ligand_sampled_atom_count": int(raw_count),
        }
    )
    return metadata


def _resolve_distribution_request(
    request: ExpandedRequest,
    *,
    global_seed: int,
    ccd_path: Path | None,
    geometry_cache: dict[_GeometryKey, _Geometry],
) -> ExpandedRequest:
    chain_id = request.task.length_distribution_chain
    if chain_id is None:
        return replace(request, length_metadata=_manual_metadata(request))
    if len(request.task.design_chains) != 1:
        raise DesignSampleError(
            "design_length.from_distribution requires exactly one design chain"
        )
    chain = request.task.design_chains[0]
    if chain.chain_id != chain_id or chain.modality != "ligand":
        raise DesignSampleError(
            "design_length.from_distribution requires exactly one ligand design chain"
        )

    # Keep histogram/TargetDiff asset loading on the actual distribution path;
    # manual requests must remain a no-op with respect to that optional asset.
    from ..utils.pocket_size import pocket_space_size, sample_ligand_atom_count

    geometry_key = (
        None if request.task.input_path is None else str(request.task.input_path),
        chain_id, request.task.hotspot,
        None if ccd_path is None else str(ccd_path),
    )
    if geometry_key not in geometry_cache:
        source = _read_full_input(request, ccd_path)
        pocket_coords, pocket_source = _pocket_coordinates(request, source)
        if len(pocket_coords) < 2:
            raise DesignInputError(
                "ligand pocket requires at least two finite heavy atoms for sizing"
            )
        geometry_cache[geometry_key] = (
            int(len(pocket_coords)), float(pocket_space_size(pocket_coords)), pocket_source,
        )
    pocket_atom_count, pocket_size, pocket_source = geometry_cache[geometry_key]
    rng = np.random.default_rng(base_request_seed(global_seed, request))
    sampled_count, bin_index = sample_ligand_atom_count(pocket_size, rng)
    if (
        isinstance(sampled_count, bool)
        or not isinstance(sampled_count, (int, np.integer))
        or int(sampled_count) < 2
    ):
        raise DesignSampleError(
            "ligand atom-count distribution returned fewer than two heavy atoms"
        )
    if isinstance(bin_index, bool) or not isinstance(bin_index, (int, np.integer)):
        raise DesignSampleError("ligand atom-count distribution returned an invalid bin")

    ligand_conditions = request.task.condition.get("ligand", {})
    motif_entry = (
        ligand_conditions.get(chain_id)
        if isinstance(ligand_conditions, Mapping)
        else None
    )
    if isinstance(motif_entry, Mapping) and motif_entry.get("mode") == "atom_motif":
        motif_atoms = motif_entry.get("atoms")
        if isinstance(motif_atoms, list) and len(motif_atoms) > int(sampled_count):
            raise DesignSampleError(
                "atom_motif design_length is smaller than selected motif"
            )

    metadata = dict(request.length_metadata)
    metadata.update(
        {
            "ligand_length_source": "from_distribution",
            "ligand_pocket_atom_count": pocket_atom_count,
            "ligand_pocket_size_angstrom": pocket_size,
            "ligand_length_bin": int(bin_index),
            "ligand_sampled_atom_count": int(sampled_count),
            "ligand_pocket_source": pocket_source,
        }
    )
    plan = LengthPlan(
        request.plan.index,
        {**dict(request.plan.lengths), chain_id: int(sampled_count)},
    )
    return replace(request, plan=plan, length_metadata=metadata)


def resolve_length_distributions(
    requests: Sequence[ExpandedRequest],
    *,
    global_seed: int,
    ccd_path: Path | None = None,
) -> tuple[ExpandedRequest, ...]:
    """Resolve distribution sentinels once for each expanded base request."""

    # Invocation-local geometry only: never cache random draws or mutable source
    # AtomArrays, and never retain a stale geometry across separate runner calls.
    geometry_cache: dict[_GeometryKey, _Geometry] = {}
    return tuple(
        _resolve_distribution_request(
            request,
            global_seed=global_seed,
            ccd_path=ccd_path,
            geometry_cache=geometry_cache,
        )
        for request in requests
    )


__all__ = ["base_request_seed", "resolve_length_distributions"]
