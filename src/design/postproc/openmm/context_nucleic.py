"""Whole fixed modified nucleic context in a private, charged GAFF view.

Canonical RNA/DNA stays on native Amber. A modified fixed RNA/DNA component
uses its complete connected graph, the native phosphodiester anion convention,
and explicit source charge-state facts. No private completion atom is exported.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from biotite import structure as struc

_CANONICAL = frozenset(("A", "C", "G", "U", "DA", "DC", "DG", "DT"))


def has_modified_nucleic_context(atoms):
    return bool(np.any(~np.asarray(atoms.is_design, dtype=bool)
                      & np.isin(atoms.mol_type, ("rna", "dna"))
                      & ~np.isin(atoms.res_name, tuple(_CANONICAL))))


def _private_terminal_phosphates(atoms):
    from .refine import OpenMMPostprocessError
    private = atoms.copy()
    additions = []
    bonds = list(atoms.bonds.as_array())
    for row in np.flatnonzero(~atoms.is_design & np.isin(atoms.mol_type, ("rna", "dna"))
                              & (atoms.atom_name == "P")):
        neighbors, orders = atoms.bonds.get_bonds(int(row))
        if len(neighbors) != 3:
            continue
        names = {str(atoms.atom_name[n]) for n in neighbors}
        same = all((atoms.chain_id[n], atoms.res_id[n], atoms.ins_code[n])
                   == (atoms.chain_id[row], atoms.res_id[row], atoms.ins_code[row])
                   for n in neighbors)
        if not same or names != {"OP1", "OP2", "O5'"}:
            raise OpenMMPostprocessError("modified nucleic terminal phosphate is incomplete or ambiguous")
        if sorted(map(int, orders)) != [1, 1, 2] or any(atoms.element[n] != "O" for n in neighbors):
            raise OpenMMPostprocessError("modified nucleic terminal phosphate has unsupported bond chemistry")
        vectors = atoms.coord[neighbors].astype(float) - atoms.coord[row]
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths < 1e-6):
            raise OpenMMPostprocessError("modified nucleic terminal phosphate has a degenerate oxygen frame")
        direction = -(vectors / lengths[:, None]).sum(axis=0)
        if np.linalg.norm(direction) < 1e-6:
            direction = np.cross(vectors[0], vectors[1])
        if np.linalg.norm(direction) < 1e-6:
            raise OpenMMPostprocessError("modified nucleic terminal phosphate has a degenerate oxygen frame")
        extra = atoms[[int(row)]].copy()
        extra.atom_name[0], extra.element[0] = "OP3", "O"
        extra.coord[0] = atoms.coord[row] + 1.6 * direction / np.linalg.norm(direction)
        extra.is_design[0], extra.coord_valid[0] = False, True
        extra.atom_uid[0] = int(np.max(atoms.atom_uid)) + 1 + len(additions)
        for name in ("centre_atom_mask", "identity_condition", "motif_coord_condition",
                     "token_center_condition", "hotspot_condition"):
            if name in extra.get_annotation_categories():
                getattr(extra, name)[0] = False
        new_row = len(atoms) + len(additions)
        additions.append(extra)
        bonds.append((int(row), new_row, int(struc.BondType.SINGLE)))
    if additions:
        private = struc.concatenate([atoms, *additions])
        private.bonds = struc.BondList(len(private), np.asarray(bonds, dtype=np.uint32))
    return private


def _charged_records(private, groups, source_charges, component_overrides=None):
    from .context_polymer import _atom_key
    from .product_graph import build_product_records
    from .refine import OpenMMPostprocessError
    selected = {int(row) for group in groups for row in group}
    by_key = {_atom_key(private, row): row for row in selected}
    declared = {by_key[key]: charge for key, charge in source_charges if key in by_key}
    charges = {_atom_key(private, row): charge for row, charge in declared.items()}
    for row in sorted(selected):
        if str(private.element[row]).upper() != "P":
            continue
        neighbors, orders = private.bonds.get_bonds(row)
        if (len(neighbors) != 4 or any(str(private.element[n]).upper() != "O" for n in neighbors)
                or sorted(map(int, orders)) != [1, 1, 1, 2]):
            raise OpenMMPostprocessError("modified nucleic phosphate needs a closed tetrahedral oxygen graph")
        terminal = [int(n) for n in neighbors if len(private.bonds.get_bonds(int(n))[0]) == 1]
        single = [int(n) for n, order in zip(neighbors, orders, strict=True)
                  if int(order) == 1 and int(n) in terminal]
        if not single:
            raise OpenMMPostprocessError("modified nucleic phosphate has no nonbridging single oxygen")
        if any(declared.get(n, 0) not in (0, -1) for n in terminal):
            raise OpenMMPostprocessError("modified nucleic phosphate has an unsupported explicit charge state")
        # One anion per native phosphodiester or mono-protonated terminal
        # monophosphate. Explicit additional deprotonation is honored by count.
        count = max(1, sum(declared.get(n) == -1 for n in terminal))
        if count > len(single):
            raise OpenMMPostprocessError("modified nucleic phosphate charge contradicts its heavy graph")
        for n in terminal:
            charges[_atom_key(private, n)] = 0
        # Preserve explicit single-oxygen assignments, then deterministic CCD
        # names. A private terminal OP3 remains protonated unless requested.
        single.sort(key=lambda n: (declared.get(n) != -1,
                                   str(private.atom_name[n]) == "OP3",
                                   str(private.atom_name[n])))
        for n in single[:count]:
            charges[_atom_key(private, n)] = -1
    candidate = SimpleNamespace(atoms=private, private_ligand_molecules={},
                                private_ligand_errors={})
    records = build_product_records(candidate, groups, formal_charge_overrides=charges,
                                    component_overrides=component_overrides)
    for record in records:
        for key, element, hydrogens in zip(record.heavy_keys, record.elements,
                                            record.hydrogen_counts, strict=True):
            if element == "P" and hydrogens:
                raise OpenMMPostprocessError("modified nucleic product cannot contain a P-H bond")
    return tuple(replace(record, record_id=f"nucleic-context-{ordinal}")
                 for ordinal, record in enumerate(records))


def minimize_modified_nucleic_context(atoms, *, payload, platform_name, charge_method, component_overrides=None, move_context=False):
    from .context_polymer import (_atom_key, _components, _build_parameterization_topology,
                                  _repair_parameterization_topology,
                                  _private_heavy_atom_array, _map_back)
    from .refine import OpenMMPostprocessError
    from ..runner import _nonbonded_metal_relation
    from .peptide import minimize_peptides
    if payload.local_product_records or payload.covalent_link is not None:
        raise OpenMMPostprocessError("modified nucleic context cannot be combined with local/covalent products")
    if atoms.bonds is None:
        raise OpenMMPostprocessError("modified nucleic context requires an authoritative bond graph")
    view = atoms.copy()
    view.bonds = struc.BondList(len(view), np.asarray([
        (i, j, order) for i, j, order in atoms.bonds.as_array()
        if not _nonbonded_metal_relation(atoms, int(i), int(j), int(order))
    ], dtype=np.uint32).reshape(-1, 3))
    topology, positions = _build_parameterization_topology(view, payload)
    split, (repaired_topology, repaired_positions) = _repair_parameterization_topology(topology, positions)
    private = _private_heavy_atom_array(view, view, split, repaired_topology, repaired_positions)
    private = _private_terminal_phosphates(private)
    groups = []
    for group in _components(private):
        if not any(private.mol_type[row] in ("rna", "dna")
                   and private.res_name[row] not in _CANONICAL
                   and not private.is_design[row] for row in group):
            continue
        if any(private.is_design[row] or private.mol_type[row] not in ("rna", "dna") for row in group):
            raise OpenMMPostprocessError("modified nucleic context must be a closed fixed RNA/DNA component")
        groups.append(group)
    if not groups:
        raise OpenMMPostprocessError("modified nucleic context has no closed parameterizable component")
    records = _charged_records(private, groups, payload.context_formal_charges, component_overrides)
    design_keys = tuple(_atom_key(atoms, int(row)) for row in np.flatnonzero(atoms.is_design))
    private_payload = replace(payload, peptide_records=payload.peptide_records + records,
                              native_moving_keys=design_keys)
    refined, initial, final, force = minimize_peptides(
        private, payload=private_payload, platform_name=platform_name,
        charge_method=charge_method, fixed_product_record_ids=tuple(r.record_id for r in records),
        move_context=move_context)
    if not np.isfinite([initial, final, force]).all():
        raise OpenMMPostprocessError("modified nucleic minimization returned non-finite summaries")
    return _map_back(atoms, private, refined, groups, move_context=move_context), initial, final, force
