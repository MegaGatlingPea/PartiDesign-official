"""Graph-only partition for local covalent parameterization.

This module does not assign force-field parameters or infer reaction chemistry.
The complete receptor residue at each attachment is part of the local product;
three-bond context provides every boundary torsion's real heavy endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class CovalentRegion:
    moving_rows: tuple[int, ...]
    product_rows: tuple[int, ...]
    context_rows: tuple[int, ...]
    halo_rows: tuple[int, ...]
    attachment_edges: tuple[tuple[int, int, int], ...]
    boundary_edges: tuple[tuple[int, int, int], ...]


def plan_covalent_region(atoms, moving_rows) -> CovalentRegion:
    """Partition explicit moving atoms; never grow motion into the receptor.

    The caller must provide complete design units. A design atom outside the
    selection connected to it is rejected rather than silently frozen. The
    returned halo is a parameterization scaffold, not extra moving atoms.
    """
    from ..runner import PostprocessPreparationError

    moving = set(map(int, moving_rows))
    if not moving or any(r < 0 or r >= len(atoms) for r in moving):
        raise PostprocessPreparationError("local covalent region has invalid moving rows")
    if any(not bool(atoms.is_design[r]) for r in moving):
        raise PostprocessPreparationError("local covalent moving rows include context")
    if atoms.bonds is None:
        raise PostprocessPreparationError("local covalent region requires explicit topology")
    edges = [tuple(map(int, edge)) for edge in atoms.bonds.as_array()]
    adjacency = [set() for _ in atoms]
    for i, j, order in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    attachments = []
    linked_context = set()
    for i, j, order in edges:
        if (i in moving) == (j in moving):
            continue
        inside, outside = (i, j) if i in moving else (j, i)
        if bool(atoms.is_design[outside]):
            raise PostprocessPreparationError("local covalent selection splits a design unit")
        if str(atoms.mol_type[outside]) not in {"protein", "rna", "dna"}:
            raise PostprocessPreparationError("local covalent receptor endpoint must be polymer")
        attachments.append((inside, outside, order))
        linked_context.add(outside)
    if not attachments:
        raise PostprocessPreparationError("local covalent region has no receptor attachment")

    def residue_key(r):
        return str(atoms.chain_id[r]), int(atoms.res_id[r]), str(atoms.ins_code[r])

    linked_residues = {residue_key(r) for r in linked_context}
    context = {r for r in range(len(atoms)) if residue_key(r) in linked_residues}
    if any(bool(atoms.is_design[r]) for r in context):
        raise PostprocessPreparationError("attachment receptor residue has mixed design ownership")
    product = moving | context
    halo = set(product)
    frontier = set(product)
    for _ in range(3):
        frontier = set().union(*(adjacency[r] for r in frontier)) - halo if frontier else set()
        halo.update(frontier)
    # Include complete rings/residues rather than cutting an aromatic residue.
    halo_residues = {residue_key(r) for r in halo if str(atoms.mol_type[r]) != "ligand"}
    halo.update(r for r in range(len(atoms)) if residue_key(r) in halo_residues)
    relevant = [(i, j, t) for i, j, t in edges if i in halo or j in halo]
    if any(t not in {1, 2, 3, 5, 6, 7, 9} for _, _, t in relevant):
        raise PostprocessPreparationError("local covalent chemistry requires concrete bond orders")
    boundary = [(i, j, t) for i, j, t in edges if (i in product) != (j in product)]
    if any(i not in halo or j not in halo for i, j, _ in boundary):
        raise AssertionError("local parameterization halo lost a boundary endpoint")
    return CovalentRegion(
        tuple(sorted(moving)), tuple(sorted(product)), tuple(sorted(context)),
        tuple(sorted(halo - product)), tuple(sorted(attachments)), tuple(sorted(boundary)),
    )
