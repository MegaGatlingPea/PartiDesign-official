"""Materialize private fixed-context chemistry while retaining public row ownership."""
from __future__ import annotations

from dataclasses import replace
from biotite import structure as struc
import numpy as np

from ..runner import _atom_key, PostprocessPreparationError


def materialize_fixed_context(atoms, payload):
    """Return a private view, rebound payload, and explicit CCD overrides."""
    view = atoms
    components = {}
    mapping = {}
    if payload.linked_context_residues:
        from .linked_context import linked_context_view
        view, mapping, components = linked_context_view(atoms, payload.linked_context_residues)
        # PDB-derived polymer connections may retain ANY orders. Resolve only
        # named backbone edges in the validated closed fixed polymer product.
        from .context_polymer import _components
        from ..runner import _nonbonded_metal_relation
        covalent = view.copy()
        covalent.bonds = struc.BondList(len(view), np.asarray([
            (i, j, order) for i, j, order in view.bonds.as_array()
            if not _nonbonded_metal_relation(view, int(i), int(j), int(order))
        ], dtype=np.uint32).reshape(-1, 3))
        seed_keys = {mapping[key] for spec in payload.linked_context_residues for key in spec.heavy_keys}
        selected = {row for group in _components(covalent)
                    if any(_atom_key(view, row) in seed_keys for row in group) for row in group}
        edges = view.bonds.as_array().copy()
        removed = set()
        # Historical readers can join two adjacent numbered cyclic products.
        # A saturated ester carbonyl cannot also carry a distant peptide N.
        # Remove that inconsistent bridge only from the private fixed graph.
        for ordinal, (raw_i, raw_j, raw_order) in enumerate(edges):
            i, j = int(raw_i), int(raw_j)
            if i not in selected or j not in selected or int(raw_order) != int(struc.BondType.ANY):
                continue
            if view.mol_type[i] != "protein" or view.mol_type[j] != "protein":
                continue
            if {str(view.element[i]).upper(), str(view.element[j]).upper()} != {"C", "O"}:
                continue
            carbon, oxygen = (i, j) if str(view.element[i]).upper() == "C" else (j, i)
            if str(view.atom_name[carbon]) != "C":
                continue
            on, oo = view.bonds.get_bonds(oxygen)
            alcohol = [(int(n), int(o)) for n, o in zip(on, oo) if int(n) != carbon]
            if (len(alcohol) != 1 or alcohol[0][1] != int(struc.BondType.SINGLE)
                    or str(view.element[alcohol[0][0]]).upper() != "C"):
                continue
            neighbors, orders = view.bonds.get_bonds(carbon)
            if not any(int(o) == 2 and str(view.element[n]).upper() == "O" for n, o in zip(neighbors, orders)):
                continue
            if np.linalg.norm(view.coord[carbon] - view.coord[oxygen]) > 2.0:
                continue
            for other, order in zip(neighbors, orders):
                other = int(other)
                if (other in selected and view.mol_type[other] == "protein"
                        and str(view.atom_name[other]) == "N" and str(view.element[other]).upper() == "N"
                        and int(view.res_id[other]) == int(view.res_id[carbon]) + 1
                        and view.chain_id[other] == view.chain_id[carbon]
                        and np.linalg.norm(view.coord[carbon] - view.coord[other]) > 3.0):
                    for index, (left, right, _) in enumerate(edges):
                        if {int(left), int(right)} == {carbon, other}:
                            removed.add(index)
        if removed:
            edges = np.asarray([edge for index, edge in enumerate(edges) if index not in removed], dtype=np.uint32)
            view.bonds = struc.BondList(len(view), edges)
        for edge in edges:
            i, j, order = map(int, edge)
            if order != int(struc.BondType.ANY) or i not in selected or j not in selected:
                continue
            left, right = _atom_key(view, i), _atom_key(view, j)
            if view.is_design[i] or view.is_design[j] or left[:3] == right[:3] or left[0] != right[0]:
                continue
            names = {left[4], right[4]}
            elements = {str(view.element[i]).upper(), str(view.element[j]).upper()}
            protein_link = (view.mol_type[i] == view.mol_type[j] == "protein"
                            and names == {"C", "N"} and elements == {"C", "N"})
            nucleic_link = (view.mol_type[i] == view.mol_type[j] and view.mol_type[i] in ("rna", "dna")
                            and names == {"O3'", "P"} and elements == {"O", "P"})
            ester_link = False
            if view.mol_type[i] == view.mol_type[j] == "protein" and elements == {"C", "O"}:
                carbon, oxygen = (i, j) if str(view.element[i]).upper() == "C" else (j, i)
                cn, co = view.bonds.get_bonds(carbon)
                on, oo = view.bonds.get_bonds(oxygen)
                carbonyl = [(int(n), int(o)) for n, o in zip(cn, co)
                            if int(n) != oxygen]
                alcohol = [(int(n), int(o)) for n, o in zip(on, oo)
                           if int(n) != carbon]
                # A concrete carbonyl C and a one-neighbor alcohol O can
                # only form a neutral single ester link in this closed graph.
                ester_link = (str(view.atom_name[carbon]) == "C"
                              and sorted(o for _, o in carbonyl) == [1, 2]
                              and any(o == 2 and str(view.element[n]).upper() == "O" for n, o in carbonyl)
                              and len(alcohol) == 1 and alcohol[0][1] == 1
                              and str(view.element[alcohol[0][0]]).upper() == "C")
            if protein_link or nucleic_link or ester_link:
                edge[2] = int(struc.BondType.SINGLE)
        view.bonds = struc.BondList(len(view), edges)

    def rekey(key):
        return mapping.get(key, key)

    def records(values):
        return tuple(replace(record, heavy_keys=tuple(rekey(k) for k in record.heavy_keys))
                     for record in values)

    payload = replace(
        payload,
        linked_context_residues=(),
        gaff_ligands=records(payload.gaff_ligands),
        peptide_records=records(payload.peptide_records),
        local_product_records=records(payload.local_product_records),
        native_moving_keys=tuple(rekey(k) for k in payload.native_moving_keys),
        local_moving_keys=tuple(rekey(k) for k in payload.local_moving_keys),
        context_formal_charges=tuple((rekey(k), charge) for k, charge in payload.context_formal_charges),
        cobalt_hexammine_groups=tuple(tuple(rekey(k) for k in group)
                                      for group in payload.cobalt_hexammine_groups),
        fixed_coordination_records=records(payload.fixed_coordination_records),
        fixed_ligand_completions=records(payload.fixed_ligand_completions),
    )
    if payload.fixed_ligand_completions:
        from .fixed_ligand import expand_fixed_ligands
        view, completed = expand_fixed_ligands(
            view, payload.fixed_ligand_completions,
            formal_charge_facts=payload.context_formal_charges)
        payload = replace(payload, gaff_ligands=payload.gaff_ligands + completed,
                          fixed_ligand_completions=())
    return view, payload, components


def restore_public_context(original, private, refined):
    """Export coordinates only; private atoms and identities never enter the CIF."""
    if len(private) != len(refined) or len(private) < len(original):
        raise PostprocessPreparationError("private refinement changed heavy-atom row ownership")
    if [_atom_key(private, i) for i in range(len(private))] != [
        _atom_key(refined, i) for i in range(len(refined))
    ] or not np.array_equal(private.element, refined.element):
        raise PostprocessPreparationError("private refinement changed heavy-atom identity")
    if not np.isfinite(refined.coord).all():
        raise PostprocessPreparationError("private refinement returned non-finite coordinates")
    if not np.array_equal(private.element[:len(original)], original.element):
        raise PostprocessPreparationError("private context does not preserve public heavy rows")
    result = original.copy()
    result.coord = np.asarray(refined.coord[:len(original)], dtype=np.float32).copy()
    return result
