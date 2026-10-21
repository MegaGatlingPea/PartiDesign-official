"""Stable public-heavy/private-hydrogen keys for hybrid force-field assembly."""
from collections import defaultdict


def hydrogen_key(parent, ordinal):
    return ("private-H", parent, int(ordinal))


def heavy_parent(key):
    if isinstance(key, tuple) and len(key) == 3 and key[0] == "private-H":
        return key[1]
    return key


def scaffold_atom_keys(topology, reference_atoms):
    """Map by preserved src row IDs, not OpenMM atom order or residue index."""
    from .refine import atom_key, OpenMMPostprocessError

    atoms = list(topology.atoms())
    graph = defaultdict(list)
    for a, b in topology.bonds():
        graph[a.index].append(b.index)
        graph[b.index].append(a.index)
    result = [None] * len(atoms)
    seen_rows = set()
    source_keys = [atom_key(reference_atoms, row) for row in range(len(reference_atoms))]
    if len(source_keys) != len(set(source_keys)):
        raise OpenMMPostprocessError("private scaffold reference identities are ambiguous")
    source_rows = {key: row for row, key in enumerate(source_keys)}
    for atom in atoms:
        if atom.element is None:
            raise OpenMMPostprocessError("private scaffold atom has no element")
        if atom.element.atomic_number == 1:
            continue
        identity = (atom.residue.chain.id, int(atom.residue.id), atom.residue.insertionCode,
                    atom.residue.name, atom.name)
        # Modeller.addHydrogens recreates topology atoms and may discard IDs.
        # Its complete unchanged heavy identity is still an exact unique map;
        # neither private chain ordinal nor atom traversal order is sufficient.
        row = int(atom.id[4:]) if atom.id and atom.id.startswith("src:") else source_rows.get(identity, -1)
        if row < 0 or row >= len(reference_atoms) or row in seen_rows:
            raise OpenMMPostprocessError("private scaffold source row mapping is invalid")
        if source_keys[row] != identity:
            raise OpenMMPostprocessError("private scaffold heavy identity changed")
        seen_rows.add(row)
        result[atom.index] = atom_key(reference_atoms, row)
    counts = defaultdict(int)
    for atom in atoms:
        if atom.element.atomic_number != 1:
            continue
        neighbors = graph[atom.index]
        if len(neighbors) != 1 or result[neighbors[0]] is None:
            raise OpenMMPostprocessError("private scaffold H lacks a unique heavy parent")
        parent = result[neighbors[0]]
        result[atom.index] = hydrogen_key(parent, counts[parent])
        counts[parent] += 1
    if len(result) != len(set(result)):
        raise OpenMMPostprocessError("private scaffold atom keys are not unique")
    return tuple(result)


def product_atom_keys(molecule, local_record, full_record):
    """Label actual-product hydrogens; surplus private cap H receive None.

    H atoms on the same parent are chemically equivalent for ownership. Their
    coordinates always come from that parent's owning force-field structure.
    """
    from .refine import OpenMMPostprocessError

    full_counts = dict(zip(full_record.heavy_keys, full_record.hydrogen_counts, strict=True))
    heavy_count = len(local_record.heavy_keys)
    result = list(local_record.heavy_keys) + [None] * (molecule.GetNumAtoms() - heavy_count)
    counts = defaultdict(int)
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        if index < heavy_count:
            if atom.GetAtomicNum() == 1:
                raise OpenMMPostprocessError("local product order is not heavy-first")
            continue
        neighbors = list(atom.GetNeighbors())
        if atom.GetAtomicNum() != 1 or len(neighbors) != 1 or neighbors[0].GetIdx() >= heavy_count:
            raise OpenMMPostprocessError("local product H lacks a unique heavy parent")
        parent = result[neighbors[0].GetIdx()]
        if parent not in full_counts:
            raise OpenMMPostprocessError("local product parent absent from complete chemistry")
        ordinal = counts[parent]
        counts[parent] += 1
        if ordinal < full_counts[parent]:
            result[index] = hydrogen_key(parent, ordinal)
    for key, count in zip(local_record.heavy_keys, local_record.hydrogen_counts, strict=True):
        if counts[key] != count or count < full_counts[key]:
            raise OpenMMPostprocessError("private local H inventory disagrees with explicit product")
    return tuple(result)
