"""Assemble a local product with an unchanged, parameterized Amber complement."""
from .mapping import scaffold_atom_keys, product_atom_keys, heavy_parent
from .hybrid import merge_parameterized_region, constrain_core_charge


def assemble_product(atoms, region, *, reference_atoms, scaffold, scaffold_topology,
                     product, molecule, local_record=None, full_record=None,
                     product_keys=None, full_records=None):
    """Require explicit topology and chemistry; return parameters plus stable keys.

    The changed core consists of complete linked context residues and the design
    unit. Its GAFF model replaces all Amber terms touching that core (including
    Amber-specific CMAP); the complement retains native terms and atom charges.
    Public heavy atoms are neither deleted nor invented by this assembly.
    """
    from .refine import atom_key, OpenMMPostprocessError

    base_keys = scaffold_atom_keys(scaffold_topology, reference_atoms)
    local_keys = (product_atom_keys(molecule, local_record, full_record)
                  if product_keys is None else tuple(product_keys))
    records = (full_record,) if full_records is None else tuple(full_records)
    if not records or any(record is None for record in records) or len(local_keys) != molecule.GetNumAtoms():
        raise OpenMMPostprocessError("hybrid product chemistry mapping is incomplete")
    public_keys = tuple(atom_key(atoms, row) for row in range(len(atoms)))
    core_heavy = {public_keys[row] for row in region.product_rows}
    core_keys = {key for key in local_keys if key is not None and heavy_parent(key) in core_heavy}
    if not core_heavy.issubset(core_keys):
        raise OpenMMPostprocessError("local parameterization is missing core heavy atoms")
    final_keys = tuple(key for key in base_keys if heavy_parent(key) not in core_heavy) + tuple(
        key for key in local_keys if key in core_keys
    )
    if len(final_keys) != len(set(final_keys)) or not set(public_keys).issubset(final_keys):
        raise OpenMMPostprocessError("hybrid assembly does not preserve public atom inventory")
    expected = set()
    for bond in scaffold.bonds:
        left, right = base_keys[bond.atom1.idx], base_keys[bond.atom2.idx]
        if heavy_parent(left) not in core_heavy and heavy_parent(right) not in core_heavy:
            expected.add(frozenset((left, right)))
    for i, j, _ in atoms.bonds.as_array():
        left, right = public_keys[int(i)], public_keys[int(j)]
        if left in core_heavy or right in core_heavy:
            expected.add(frozenset((left, right)))
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left, right = local_keys[i], local_keys[j]
        if (left in core_keys or right in core_keys) and (
            molecule.GetAtomWithIdx(i).GetAtomicNum() == 1 or molecule.GetAtomWithIdx(j).GetAtomicNum() == 1
        ):
            if left is None or right is None:
                raise OpenMMPostprocessError("private cap hydrogen reaches product core")
            expected.add(frozenset((left, right)))
    public_edges = {frozenset((public_keys[int(i)], public_keys[int(j)])) for i, j, _ in atoms.bonds.as_array()}
    if not public_edges.issubset(expected):
        raise OpenMMPostprocessError("native scaffold lost an authoritative public bond")
    merged = merge_parameterized_region(
        scaffold, product, scaffold_keys=base_keys, product_keys=local_keys,
        core_keys=core_keys, final_keys=final_keys, expected_bonds=expected,
    )
    charge_audit = _constrain_product_charges(atoms, merged, final_keys, core_heavy, core_keys, records)
    return merged, final_keys, charge_audit


def _constrain_product_charges(atoms, structure, final_keys, core_heavy, core_keys, records):
    """Fit each disconnected product independently, never transfer charge between them."""
    from .refine import OpenMMPostprocessError
    from .reference_chemistry import nucleic_core_charge_target
    plans, assigned = [], set()
    for record in records:
        heavy = set(record.heavy_keys)
        owned = {key for key in core_keys if heavy_parent(key) in heavy}
        if not owned or assigned.intersection(owned):
            raise OpenMMPostprocessError('product charge partitions are empty or overlap')
        assigned.update(owned)
        formal = sum(q for key, q in zip(record.heavy_keys, record.formal_charges, strict=True) if key in core_heavy)
        target = nucleic_core_charge_target(record, atoms, structure, final_keys, owned)
        if target is None:
            target = formal
        plans.append((record.record_id, owned, formal, target))
    if assigned != core_keys:
        raise OpenMMPostprocessError('product charge partitions do not cover the complete core')
    reports = [dict(record_id=name, **constrain_core_charge(structure, final_keys, owned, formal,
                                                          target_charge=target))
               for name, owned, formal, target in plans]
    if len(reports) == 1:
        return reports[0]
    return dict(core_formal_charge=sum(row['core_formal_charge'] for row in reports),
                core_target_partial_charge=sum(row['core_target_partial_charge'] for row in reports),
                core_partial_charge_before=sum(row['core_partial_charge_before'] for row in reports),
                corrected_atoms=sum(row['corrected_atoms'] for row in reports), components=reports)
