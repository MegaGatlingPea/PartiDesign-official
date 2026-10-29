"""Private GAFF products for fixed modified protein context.

A modified protein residue in a fixed context cannot use the ordinary Amber
residue templates.  This module creates a private parameterization view of the
whole closed heavy-atom component containing that residue.  The public
``AtomArray`` is never changed or exported with private repair atoms.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from biotite import structure as struc

from ...utils.atoms import PROTEIN_RES_ATOMS
from ...utils.ccd import load_packaged_polymer_ccd


_STANDARD_PROTEIN = frozenset(PROTEIN_RES_ATOMS) - {"UNK"}
_PROTEIN_MODIFIED = frozenset({"SCH", "SAR", "MVA"})
_NEW_ATOM_PREFIX = "private:missing-heavy:"


def _error_type() -> type[Exception]:
    from .refine import OpenMMPostprocessError

    return OpenMMPostprocessError


def _atom_key(atoms: Any, row: int) -> tuple[str, int, str, str, str]:
    return (
        str(atoms.chain_id[row]),
        int(atoms.res_id[row]),
        str(atoms.ins_code[row]),
        str(atoms.res_name[row]),
        str(atoms.atom_name[row]),
    )


def _components(atoms: Any) -> tuple[tuple[int, ...], ...]:
    count = len(atoms)
    adjacency = [set() for _ in range(count)]
    if atoms.bonds is not None:
        for raw_i, raw_j, _ in atoms.bonds.as_array():
            i, j = int(raw_i), int(raw_j)
            if i < 0 or j < 0 or i >= count or j >= count or i == j:
                raise _error_type()("modified protein context has an invalid bond graph")
            adjacency[i].add(j)
            adjacency[j].add(i)
    remaining = set(range(count))
    components: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        component = {start}
        queue = [start]
        while queue:
            row = queue.pop()
            for neighbor in adjacency[row]:
                if neighbor not in component:
                    component.add(neighbor)
                    queue.append(neighbor)
        remaining.difference_update(component)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _modified_rows(atoms: Any) -> np.ndarray:
    modalities = np.asarray(atoms.mol_type).astype(str)
    names = np.asarray(atoms.res_name).astype(str)
    return np.flatnonzero(
        (modalities == "protein") & ~np.isin(names, tuple(_STANDARD_PROTEIN))
    ).astype(np.int64)


def has_modified_protein_context(atoms: Any) -> bool:
    """Return whether a fixed connected protein context needs private GAFF.

    Modified residues that are entirely design-owned remain on the normal
    whole-peptide route.  A modified residue with any fixed atom selects this
    route; the minimizer then validates the complete component and reports a
    candidate-local error for unsupported chemistry.
    """

    modified = set(map(int, _modified_rows(atoms)))
    if not modified:
        return False
    is_design = np.asarray(atoms.is_design, dtype=bool)
    for component in _components(atoms):
        component_modified = modified.intersection(component)
        if component_modified and not is_design[list(component_modified)].all():
            return True
    return False


def _build_parameterization_topology(atoms: Any, payload: Any):
    from ..runner import _ambertools_on_path
    from .refine import _build_openmm_topology

    with _ambertools_on_path():
        topology, positions, _, _ = _build_openmm_topology(atoms, payload)
    return topology, positions


def _repair_parameterization_topology(topology: Any, positions: Any):
    from .preparation import repair_standard_heavy_atoms
    from .scaffold import split_private_backbone_breaks

    split = split_private_backbone_breaks(topology)
    return split, repair_standard_heavy_atoms(split, positions)


def _positions_angstrom(positions: Any, count: int) -> np.ndarray:
    from openmm import unit

    if hasattr(positions, "value_in_unit"):
        values = np.asarray(positions.value_in_unit(unit.angstrom), dtype=np.float64)
    else:
        values = np.asarray(positions, dtype=np.float64)
    if values.shape != (count, 3) or not np.isfinite(values).all():
        raise _error_type()("private repaired coordinates are invalid")
    return values


def _residue_key(residue: Any) -> tuple[str, int, str, str]:
    try:
        residue_id = int(residue.id)
    except (TypeError, ValueError) as exc:
        raise _error_type()("private topology has a non-integral residue ID") from exc
    return (
        str(residue.chain.id),
        residue_id,
        str(residue.insertionCode),
        str(residue.name),
    )


def _source_row(atom: Any, source_atoms: Any) -> int | None:
    atom_id = atom.id
    if not isinstance(atom_id, str) or not atom_id.startswith("src:"):
        return None
    raw = atom_id[4:]
    try:
        row = int(raw)
    except (TypeError, ValueError):
        raise _error_type()(f"private topology has an invalid source atom ID: {atom_id!r}")
    if row < 0 or row >= len(source_atoms):
        raise _error_type()(f"private topology source atom ID is out of range: {atom_id!r}")
    expected = _atom_key(source_atoms, row)
    observed = (*_residue_key(atom.residue), str(atom.name))
    if observed != expected:
        raise _error_type()(f"private topology changed source atom identity: {atom_id!r}")
    element = getattr(atom.element, "symbol", None)
    if str(element).upper() != str(source_atoms.element[row]).upper():
        raise _error_type()(f"private topology changed source element: {atom_id!r}")
    return row

def _missing_slot(atom: Any) -> tuple[int, int, str] | None:
    atom_id = atom.id
    if not isinstance(atom_id, str) or not atom_id.startswith(_NEW_ATOM_PREFIX):
        return None
    fields = atom_id[len(_NEW_ATOM_PREFIX) :].split(":", 2)
    if len(fields) != 3:
        raise _error_type()(f"private topology has an invalid missing-atom ID: {atom_id!r}")
    try:
        chain_index, residue_index = int(fields[0]), int(fields[1])
    except ValueError as exc:
        raise _error_type()(f"private topology has an invalid missing-atom ID: {atom_id!r}") from exc
    if chain_index < 0 or residue_index < 0 or not fields[2]:
        raise _error_type()(f"private topology has an invalid missing-atom ID: {atom_id!r}")
    if str(atom.name) != fields[2]:
        raise _error_type()(f"private topology missing-atom ID/name mismatch: {atom_id!r}")
    return chain_index, residue_index, fields[2]


def _ccd_order(residue_name: str, left_name: str, right_name: str) -> int:
    ccd = load_packaged_polymer_ccd()
    component = ccd.get(str(residue_name))
    if component is None or component.polymer_modality not in {"protein", "rna", "dna"}:
        raise _error_type()(
            f"no packaged polymer CCD component for repaired residue {residue_name!r}"
        )
    pair = {str(left_name), str(right_name)}
    facts = [fact for fact in component.bonds if {fact.atom_id_1, fact.atom_id_2} == pair]
    if len(facts) != 1:
        raise _error_type()(
            f"no unique packaged CCD bond for {residue_name}:{left_name}-{right_name}"
        )
    order = {"single": struc.BondType.SINGLE, "double": struc.BondType.DOUBLE,
             "triple": struc.BondType.TRIPLE, "aromatic": struc.BondType.AROMATIC}.get(
                 str(facts[0].order).lower()
             )
    if order is None:
        raise _error_type()(f"packaged CCD bond order is not concrete: {facts[0].order!r}")
    return int(order)


def _new_edge_order(left: Any, right: Any, private: Any) -> int:
    left_key = _atom_key(private, left)
    right_key = _atom_key(private, right)
    if left_key[:4] == right_key[:4]:
        return _ccd_order(left_key[3], left_key[4], right_key[4])
    if {left_key[4], right_key[4]} == {"C", "N"}:
        c_key = left_key if left_key[4] == "C" else right_key
        n_key = right_key if left_key[4] == "C" else left_key
        if (
            c_key[0] == n_key[0]
            and n_key[1] == c_key[1] + 1
            and c_key[3] in _STANDARD_PROTEIN
            and n_key[3] in _STANDARD_PROTEIN
        ):
            return int(struc.BondType.SINGLE)
    # PDBFixer may expose a missing phosphodiester edge when a standard RNA/DNA
    # residue gained a heavy atom.  Only the canonical O3'--P edge is accepted
    # here; a source C3'--P edge remains authoritative through ``source_edges``.
    if {left_key[4], right_key[4]} == {"O3'", "P"}:
        o3_key = left_key if left_key[4] == "O3'" else right_key
        p_key = right_key if left_key[4] == "O3'" else left_key
        ccd = load_packaged_polymer_ccd()
        left_component = ccd.get(str(o3_key[3]))
        right_component = ccd.get(str(p_key[3]))
        if (
            left_component is not None
            and right_component is not None
            and left_component.polymer_modality in {"rna", "dna"}
            and right_component.polymer_modality in {"rna", "dna"}
            and str(private[int(left if left_key[4] == "O3'" else right)].mol_type).lower() in {"rna", "dna"}
            and str(private[int(right if right_key[4] == "P" else left)].mol_type).lower() in {"rna", "dna"}
            and o3_key[0] == p_key[0]
            and p_key[1] == o3_key[1] + 1
            and _ccd_order(o3_key[3], "C3'", "O3'")
            == int(struc.BondType.SINGLE)
        ):
            # The P atom belongs to the following residue.  Its own CCD facts
            # must also contain the incoming O5'--P bond before this edge is
            # allowed to be synthesized by the repaired graph.
            incoming = _ccd_order(p_key[3], "O5'", "P")
            if incoming == int(struc.BondType.SINGLE):
                return int(struc.BondType.SINGLE)
    raise _error_type()(
        "repaired private graph contains an unsupported new inter-residue edge"
    )


def _extra_row(atoms: Any, base_row: int, name: str, element: str, coord: np.ndarray, uid: int):
    extra = atoms[[base_row]].copy()
    extra.atom_name[0] = str(name)
    extra.element[0] = str(element).upper()
    extra.coord[0] = np.asarray(coord, dtype=np.float32)
    extra.coord_valid[0] = True
    extra.is_design[0] = False
    for annotation in (
        "centre_atom_mask",
        "identity_condition",
        "motif_coord_condition",
        "token_center_condition",
        "hotspot_condition",
    ):
        if annotation in extra.get_annotation_categories():
            getattr(extra, annotation)[0] = False
    if "atom_uid" in extra.get_annotation_categories():
        extra.atom_uid[0] = int(uid)
    return extra


def _private_heavy_atom_array(
    public_atoms: Any,
    source_atoms: Any,
    source_topology: Any,
    repaired_topology: Any,
    repaired_positions: Any,
):
    """Convert repaired OpenMM heavy atoms to a private AtomArray.

    ``source_atoms`` may contain private terminal atoms (currently OXT) used
    only for Amber/GAFF template matching.  Public rows are always the prefix
    supplied in ``public_atoms``; source-only rows and PDBFixer additions are
    appended to the private view.
    """

    error = _error_type()
    public_keys = [_atom_key(public_atoms, row) for row in range(len(public_atoms))]
    source_keys = [_atom_key(source_atoms, row) for row in range(len(source_atoms))]
    if len(public_keys) != len(set(public_keys)) or len(source_keys) != len(set(source_keys)):
        raise error("original heavy-atom keys are not unique")
    if source_keys[: len(public_keys)] != public_keys:
        raise error("private template preparation reordered public heavy atoms")
    source_elements = np.asarray(source_atoms.element).astype(str)
    if np.any(np.isin(np.char.upper(source_elements), ("H", "D", "T"))):
        raise error("modified protein context does not accept explicit public hydrogens")

    source_residue_rows: dict[tuple[int, int], list[int]] = {}
    for chain_index, chain in enumerate(source_topology.chains()):
        for residue_index, residue in enumerate(chain.residues()):
            rows: list[int] = []
            for atom in residue.atoms():
                row = _source_row(atom, source_atoms)
                if row is not None:
                    rows.append(row)
            if rows:
                source_residue_rows[(chain_index, residue_index)] = rows

    output_positions = _positions_angstrom(repaired_positions, repaired_topology.getNumAtoms())
    output_row_by_atom: dict[Any, int] = {}
    source_private_rows: dict[int, int] = {}
    extras: list[Any] = []
    next_uid = int(np.max(np.asarray(public_atoms.atom_uid, dtype=np.int64))) + 1 if len(public_atoms) else 0
    seen_source_rows: set[int] = set()
    seen_new_slots: set[tuple[int, int, str]] = set()
    residue_key_by_slot: dict[tuple[int, int], tuple[str, int, str, str]] = {}
    for slot, rows in source_residue_rows.items():
        if rows:
            residue_key_by_slot[slot] = _atom_key(source_atoms, rows[0])[:4]

    # Existing source atoms are authoritative.  Public atoms remain in their
    # original order; private template atoms such as OXT are appended first.
    for atom in repaired_topology.atoms():
        if getattr(atom.element, "atomic_number", 0) <= 1:
            continue
        source_row = _source_row(atom, source_atoms)
        if source_row is not None:
            if source_row in seen_source_rows:
                raise error("private topology duplicated a source heavy atom")
            seen_source_rows.add(source_row)
            if source_row < len(public_atoms):
                output_row_by_atom[atom] = source_row
            else:
                element = getattr(atom.element, "symbol", None)
                if not element:
                    raise error("private template heavy atom lacks an element")
                next_uid += 1
                # Retain the source-only residue identity and annotations.
                source_extra = source_atoms[[source_row]].copy()
                source_extra.coord[0] = np.asarray(output_positions[atom.index], dtype=np.float32)
                source_extra.coord_valid[0] = True
                source_extra.is_design[0] = False
                for annotation in (
                    "centre_atom_mask",
                    "identity_condition",
                    "motif_coord_condition",
                    "token_center_condition",
                    "hotspot_condition",
                ):
                    if annotation in source_extra.get_annotation_categories():
                        getattr(source_extra, annotation)[0] = False
                if "atom_uid" in source_extra.get_annotation_categories():
                    source_extra.atom_uid[0] = int(next_uid - 1)
                extras.append(source_extra)
                source_private_rows[source_row] = len(public_atoms) + len(extras) - 1
                output_row_by_atom[atom] = source_private_rows[source_row]
            continue
        slot = _missing_slot(atom)
        if slot is None:
            raise error(f"private topology contains an untracked heavy atom: {atom.id!r}")
        if slot in seen_new_slots:
            raise error("private topology duplicated a repaired heavy atom")
        seen_new_slots.add(slot)
        residue_key = residue_key_by_slot.get(slot[:2])
        if residue_key is None:
            raise error("repaired heavy atom belongs to a residue absent from the source")
        base_rows = source_residue_rows[slot[:2]]
        base_source_row = base_rows[0]
        base_public_row = min(base_source_row, len(public_atoms) - 1)
        key = (*residue_key, slot[2])
        if key in source_keys:
            raise error(f"repaired heavy atom collides with a source key: {key}")
        element = getattr(atom.element, "symbol", None)
        if not element:
            raise error("repaired heavy atom lacks an element")
        extra = _extra_row(
            public_atoms,
            base_public_row,
            slot[2],
            str(element),
            output_positions[atom.index],
            next_uid,
        )
        next_uid += 1
        extras.append(extra)
        output_row_by_atom[atom] = len(public_atoms) + len(extras) - 1

    if seen_source_rows != set(range(len(source_atoms))):
        missing = sorted(set(range(len(source_atoms))).difference(seen_source_rows))
        raise error(f"private topology lost source heavy atoms: {missing[:5]}")

    if extras:
        private = struc.concatenate([public_atoms, *extras])
    else:
        private = public_atoms.copy()
    private.coord = np.asarray(private.coord, dtype=np.float32)
    private.bonds = struc.BondList(len(private))

    source_edges: dict[tuple[int, int], int] = {}
    if source_atoms.bonds is not None:
        for raw_i, raw_j, raw_order in source_atoms.bonds.as_array():
            i, j = int(raw_i), int(raw_j)
            if i >= len(public_atoms):
                i = source_private_rows.get(i, -1)
            if j >= len(public_atoms):
                j = source_private_rows.get(j, -1)
            if i < 0 or j < 0:
                raise error("private source graph references an untracked atom")
            pair = tuple(sorted((i, j)))
            if pair in source_edges:
                raise error("source heavy graph contains duplicate edges")
            source_edges[pair] = int(raw_order)

    repaired_edges: dict[tuple[int, int], int] = {}
    for bond in repaired_topology.bonds():
        left, right = output_row_by_atom.get(bond[0]), output_row_by_atom.get(bond[1])
        if left is None or right is None:
            # Bonds to explicit hydrogens generated for an existing GAFF ligand
            # are not part of the private heavy graph.
            continue
        pair = tuple(sorted((int(left), int(right))))
        if pair in repaired_edges:
            raise error("private repaired graph contains duplicate edges")
        if pair in source_edges:
            repaired_edges[pair] = source_edges[pair]
        else:
            repaired_edges[pair] = _new_edge_order(pair[0], pair[1], private)

    if not set(source_edges).issubset(repaired_edges):
        raise error("private repaired graph lost a source heavy edge")
    for pair, raw_order in repaired_edges.items():
        private.bonds.add_bond(pair[0], pair[1], raw_order)
    return private

def _context_groups(private: Any, original_length: int) -> tuple[tuple[int, ...], ...]:
    error = _error_type()
    modified = set(map(int, _modified_rows(private)))
    if not modified:
        return ()
    is_design = np.asarray(private.is_design, dtype=bool)
    groups: list[tuple[int, ...]] = []
    used: set[int] = set()
    for component in _components(private):
        component_modified = modified.intersection(component)
        if not component_modified:
            continue
        if used.intersection(component):
            continue
        used.update(component)
        fixed_modified = {
            row for row in component_modified if not bool(is_design[row])
        }
        if not fixed_modified:
            # A wholly design-owned modified component belongs to the normal
            # peptide route; only fixed modified components become context.
            continue
        if any(is_design[row] for row in component):
            raise error("modified protein context component contains design atoms")
        if any(str(private.mol_type[row]) != "protein" for row in component):
            raise error("modified protein context component contains non-protein atoms")
        if not all(row < original_length or not bool(private.is_design[row]) for row in component):
            raise error("repaired context atom has invalid design ownership")
        groups.append(tuple(sorted(component)))
    if not groups:
        raise error("modified protein context has no fixed closed component")
    return tuple(groups)


def _context_records(private: Any, groups: Sequence[Sequence[int]], component_overrides=None):
    from .product_graph import build_product_records

    candidate = SimpleNamespace(
        atoms=private,
        private_ligand_molecules={},
        private_ligand_errors={},
    )
    records = build_product_records(candidate, groups, component_overrides=component_overrides)
    return tuple(replace(record, record_id=f"context-{ordinal}") for ordinal, record in enumerate(records))


def _map_back(
    original: Any,
    private: Any,
    private_refined: Any,
    groups: Sequence[Sequence[int]],
    *,
    move_context: bool = False,
):
    error = _error_type()
    original_keys = [_atom_key(original, row) for row in range(len(original))]
    private_keys = [_atom_key(private, row) for row in range(len(private))]
    refined_keys = [_atom_key(private_refined, row) for row in range(len(private_refined))]
    if len(private_keys) != len(set(private_keys)) or refined_keys != private_keys:
        raise error("private minimization changed heavy-atom identity")
    if len(private_refined) != len(private) or not np.array_equal(
        np.asarray(private_refined.element).astype(str), np.asarray(private.element).astype(str)
    ):
        raise error("private minimization changed heavy-atom elements")
    if not np.isfinite(np.asarray(private_refined.coord, dtype=np.float64)).all():
        raise error("private minimization returned non-finite coordinates")
    context_rows = sorted({int(row) for group in groups for row in group})
    if not move_context and context_rows and not np.array_equal(
        np.asarray(private_refined.coord)[context_rows], np.asarray(private.coord)[context_rows]
    ):
        raise error("fixed modified protein context moved during minimization")
    private_row_by_key = {key: row for row, key in enumerate(private_keys)}
    if set(original_keys).difference(private_row_by_key):
        raise error("private minimization lost an original public atom")
    refined = original.copy()
    for row, key in enumerate(original_keys):
        refined.coord[row] = np.asarray(private_refined.coord[private_row_by_key[key]], dtype=np.float32)
    if refined.bonds is None or original.bonds is None:
        if refined.bonds is not original.bonds:
            raise error("public bond graph was changed")
    elif not np.array_equal(refined.bonds.as_array(), original.bonds.as_array()):
        raise error("public bond graph was changed")
    if not np.isfinite(np.asarray(refined.coord, dtype=np.float64)).all():
        raise error("public minimization returned non-finite coordinates")
    return refined


def minimize_modified_protein_context(
    atoms: Any,
    *,
    payload: Any,
    platform_name: str,
    charge_method: str,
    component_overrides=None,
    move_context: bool = False,
):
    """Minimize a modified fixed protein context with private GAFF chemistry."""

    error = _error_type()
    if payload.local_product_records or payload.covalent_link is not None:
        raise error("modified protein context cannot be combined with local/covalent payloads")
    if atoms.bonds is None:
        raise error("modified protein context requires an authoritative bond graph")
    design_key_set = {
        _atom_key(atoms, int(row)) for row in np.flatnonzero(atoms.is_design)
    }
    design_key_set.update(tuple(record_key) for record in payload.peptide_records for record_key in record.heavy_keys)
    design_key_set.update(tuple(record_key) for record_key in payload.native_moving_keys)
    public_key_set = {_atom_key(atoms, row) for row in range(len(atoms))}
    if not design_key_set.issubset(public_key_set):
        raise error("payload design ownership references a non-public heavy atom")
    design_keys = tuple(
        _atom_key(atoms, int(row))
        for row in range(len(atoms))
        if _atom_key(atoms, row) in design_key_set
    )
    from .scaffold import complete_reference_termini
    source_atoms = complete_reference_termini(atoms)
    source_topology, source_positions = _build_parameterization_topology(source_atoms, payload)
    repair_input, repaired = _repair_parameterization_topology(source_topology, source_positions)
    repaired_topology, repaired_positions = repaired
    private = _private_heavy_atom_array(atoms, source_atoms, repair_input, repaired_topology, repaired_positions)
    private_key_set = {_atom_key(private, row) for row in range(len(private))}
    if not design_key_set.issubset(private_key_set):
        raise error("private repaired topology lost a design heavy atom")
    private.is_design[:] = False
    for row, key in enumerate(_atom_key(private, index) for index in range(len(private))):
        if key in design_key_set:
            private.is_design[row] = True
    groups = _context_groups(private, len(atoms))
    records = _context_records(private, groups, component_overrides)
    if not records:
        raise error("modified protein context has no parameterizable product records")

    from dataclasses import replace as dataclass_replace
    from .peptide import minimize_peptides

    private_payload = dataclass_replace(
        payload,
        peptide_records=tuple(payload.peptide_records) + tuple(records),
        native_moving_keys=design_keys,
    )
    context_ids = tuple(record.record_id for record in records)
    private_refined, initial, final, force = minimize_peptides(
        private,
        payload=private_payload,
        platform_name=platform_name,
        charge_method=charge_method,
        fixed_product_record_ids=context_ids,
        move_context=move_context,
    )
    try:
        initial_value, final_value, force_value = float(initial), float(final), float(force)
    except (TypeError, ValueError) as exc:
        raise error("modified protein minimization returned invalid summaries") from exc
    if not np.isfinite([initial_value, final_value, force_value]).all():
        raise error("modified protein minimization returned non-finite summaries")
    refined = _map_back(atoms, private, private_refined, groups, move_context=move_context)
    return refined, initial_value, final_value, force_value


__all__ = ["has_modified_protein_context", "minimize_modified_protein_context"]
