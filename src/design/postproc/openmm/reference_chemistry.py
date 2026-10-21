"""Match a private NA halo to its already parameterized Amber complement.

The isolated CCD phosphoric-acid state is not the native Amber phosphodiester
state. Only the uniquely identified unchanged nonbridging phosphate OH may be
deprotonated here, and only outside the product core. No reaction, heavy atom,
tautomer or general protonation search is inferred.
"""
from collections import Counter
from dataclasses import replace
import logging

from .mapping import scaffold_atom_keys, heavy_parent


def match_nucleic_reference(records, atoms, core_heavy, reference_atoms, scaffold, topology):
    from .refine import atom_key, OpenMMPostprocessError

    native_keys = scaffold_atom_keys(topology, reference_atoms)
    native_h = Counter(heavy_parent(key) for key in native_keys if heavy_parent(key) != key)
    native_graph = {key: set() for key in native_keys if heavy_parent(key) == key}
    for bond in scaffold.bonds:
        left, right = native_keys[bond.atom1.idx], native_keys[bond.atom2.idx]
        if left in native_graph and right in native_graph:
            native_graph[left].add(right)
            native_graph[right].add(left)
    source = {atom_key(atoms, row): row for row in range(len(atoms))}
    result = []
    for record in records:
        graph = {i: {} for i in range(len(record.heavy_keys))}
        for i, j, order in record.bonds:
            graph[i][j] = graph[j][i] = order
        charges, counts = list(record.formal_charges), list(record.hydrogen_counts)
        changed = []
        for i, key in enumerate(record.heavy_keys):
            row = source[key]
            if key in core_heavy or atoms.is_design[row] or str(atoms.mol_type[row]) not in {'rna', 'dna'}:
                continue
            if key not in native_graph:
                raise OpenMMPostprocessError('NA reference is missing a public context atom')
            if counts[i] == native_h[key]:
                continue
            neighbors = graph[i]
            allowed = (str(atoms.res_name[row]) in {'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT'}
                       and record.elements[i] == 'O' and charges[i] == 0
                       and counts[i] == 1 and native_h[key] == 0 and len(neighbors) == 1)
            phosphorus = next(iter(neighbors), -1)
            if allowed:
                pkey = record.heavy_keys[phosphorus]
                p_neighbors = graph[phosphorus]
                allowed = (record.elements[phosphorus] == 'P' and charges[phosphorus] == 0
                           and counts[phosphorus] == 0 and neighbors[phosphorus] == 'single'
                           and len(p_neighbors) == 4
                           and all(record.elements[j] == 'O' for j in p_neighbors)
                           and sorted(p_neighbors.values()) == ['double', 'single', 'single', 'single']
                           and sum(len(graph[j]) == 2 for j in p_neighbors) == 2
                           and native_graph[key] == {pkey}
                           and native_graph.get(pkey) == {record.heavy_keys[j] for j in p_neighbors})
            if not allowed:
                raise OpenMMPostprocessError(
                    f'NA halo protonation differs from native Amber outside the supported phosphodiester state: {key}')
            # Removing the reference's absent proton changes formal charge by
            # exactly -1. Never derive formal charge by rounding partial charges.
            charges[i], counts[i] = -1, 0
            changed.append(key)
        if changed:
            logging.getLogger(__name__).info('Matched private NA halo phosphate state to native Amber: %s', changed)
        result.append(replace(record, formal_charges=tuple(charges), hydrogen_counts=tuple(counts)))
    return tuple(result)


def nucleic_core_charge_target(record, atoms, structure, final_keys, core_keys):
    """Conserve explicit whole NA-product charge with the Amber complement fixed.

    Amber NA terminal residues carry fractional residue sums. The fitted core
    may therefore have a fractional sum although its connected product's total
    formal charge is an integer. Protein-only policy is handled by the caller.
    """
    from .refine import atom_key, OpenMMPostprocessError

    source_types = {atom_key(atoms, row): str(atoms.mol_type[row]) for row in range(len(atoms))}
    modalities = {source_types[key] for key in record.heavy_keys}
    if not modalities.intersection({'rna', 'dna'}):
        return None
    if 'protein' in modalities:
        raise OpenMMPostprocessError('mixed protein/NA covalent product charge partition is not supported')
    expected_heavy = set(record.heavy_keys)
    selected = {i for i, key in enumerate(final_keys) if heavy_parent(key) in expected_heavy}
    if {final_keys[i] for i in selected if heavy_parent(final_keys[i]) == final_keys[i]} != expected_heavy:
        raise OpenMMPostprocessError('NA charge target does not cover its full public product')
    # A private heavy atom or another component may not hide outside the charge budget.
    for bond in structure.bonds:
        if (bond.atom1.idx in selected) != (bond.atom2.idx in selected):
            raise OpenMMPostprocessError('NA charge target has an unaccounted private/component boundary')
    complement_charge = sum(structure.atoms[i].charge for i in selected if final_keys[i] not in core_keys)
    return float(sum(record.formal_charges) - complement_charge)
