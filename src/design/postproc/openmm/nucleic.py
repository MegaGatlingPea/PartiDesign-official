"""Private whole-nucleic-acid product selection for OpenMM postprocessing.

The public structure and its biotite.structure.BondList remain the sole source of
atom inventory and connectivity. This module only selects design-owned RNA/DNA
components, includes explicitly bonded design bridges, and delegates product
chemistry to product_graph.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from . import product_graph


_NUCLEIC_MODALITIES = frozenset(("rna", "dna"))
_BRIDGE_MODALITIES = frozenset(("protein", "ligand"))
_SUPPORTED_MODALITIES = _NUCLEIC_MODALITIES | _BRIDGE_MODALITIES


def _authoritative_adjacency(atoms: Any) -> list[set[int]]:
    """Return undirected BondList adjacency without inventing any edges."""

    from ..runner import PostprocessPreparationError

    adjacency = [set() for _ in range(len(atoms))]
    if atoms.bonds is None:
        return adjacency
    for raw_i, raw_j, _raw_order in atoms.bonds.as_array():
        i, j = int(raw_i), int(raw_j)
        if i < 0 or i >= len(atoms) or j < 0 or j >= len(atoms):
            raise PostprocessPreparationError(
                "authoritative bond endpoint is outside the accepted structure"
            )
        if i == j:
            raise PostprocessPreparationError("authoritative bond contains a self-edge")
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


def whole_nucleic_records(candidate: Any) -> tuple[Any, ...]:
    """Build private records for design nucleic acids and explicit design bridges.

    Every design-owned RNA/DNA atom is a seed, independent of chain ID. A seed
    component may expand only through authoritative bonds into design-owned
    RNA/DNA, protein, or ligand atoms. Any edge from such a component to a
    context atom is rejected here; cross-chain/context handling belongs to the
    integration owner. Disconnected selected components are sent separately to
    build_product_records.
    """

    from ..runner import PostprocessPreparationError

    atoms = candidate.atoms
    try:
        is_design = np.asarray(atoms.is_design, dtype=bool)
        modalities = np.asarray(atoms.mol_type).astype(str)
    except AttributeError as exc:
        raise PostprocessPreparationError(
            "accepted structure lacks design ownership or modality annotations"
        ) from exc
    if len(is_design) != len(atoms) or len(modalities) != len(atoms):
        raise PostprocessPreparationError(
            "design ownership and modality annotations do not match the structure"
        )

    nucleic_seeds = {
        int(row)
        for row in np.flatnonzero(
            is_design & np.isin(modalities, tuple(_NUCLEIC_MODALITIES))
        )
    }
    if not nucleic_seeds:
        return ()

    adjacency = _authoritative_adjacency(atoms)
    assigned: set[int] = set()
    groups: list[tuple[int, ...]] = []
    for seed in sorted(nucleic_seeds):
        if seed in assigned:
            continue
        group = {seed}
        queue = [seed]
        while queue:
            row = queue.pop()
            for neighbor in sorted(adjacency[row]):
                if not bool(is_design[neighbor]):
                    raise PostprocessPreparationError(
                        "whole-nucleic optimization does not support context attachment"
                    )
                modality = str(modalities[neighbor])
                if modality not in _SUPPORTED_MODALITIES:
                    raise PostprocessPreparationError(
                        "whole-nucleic optimization supports nucleic, protein, and "
                        "ligand design bridges only"
                    )
                if neighbor not in group:
                    group.add(neighbor)
                    queue.append(neighbor)
        assigned.update(group)
        groups.append(tuple(sorted(group)))

    records = product_graph.build_product_records(candidate, groups)
    return tuple(
        replace(record, record_id=f"nucleic-{ordinal}")
        for ordinal, record in enumerate(records)
    )


__all__ = ["whole_nucleic_records"]
