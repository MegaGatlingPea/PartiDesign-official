"""Atom correspondence for ligand validation metrics.

Graph matching requires a full element/bond-order isomorphism when atom names
differ. Chemically equivalent automorphisms are accepted with a deterministic
mapping. For retained atom names, named_atom_mapping compares all observed
reference heavy atoms using names, elements and connectivity, allowing refold
CCD completion to add atoms without reference coordinates.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class LigandMappingError(ValueError):
    """The ligand correspondence is absent, incomplete, or ambiguous."""


# Biotite distinguishes the aromatic single/double/triple encodings (5, 6 and
# 7) from the generic aromatic encoding (9).  They are different source
# facts, but are equivalent witnesses for atom correspondence.  Keep the
# distinction in the graph and only use this relation when comparing the
# endpoint orders.
_AROMATIC_ORDER_EQUIVALENTS = {
    5.0: frozenset({1.0, 5.0, 9.0}),
    6.0: frozenset({2.0, 6.0, 9.0}),
    7.0: frozenset({3.0, 7.0, 9.0}),
    9.0: frozenset({5.0, 6.0, 7.0, 9.0}),
}


_ATOM_FACT_ANNOTATIONS = {
    "charge": ("formal_charge", "charge", "pdbx_formal_charge", "atom_charge"),
    "stereo": (
        "atom_stereo", "stereo", "stereochemistry", "stereo_config",
        "chirality", "pdbx_stereo_config",
    ),
    "isotope": ("isotope", "mass_number"),
    "hydrogen_count": (
        "hydrogen_count", "implicit_hydrogens", "implicit_h_count",
        "num_hydrogens", "h_count",
    ),
}


def _unknown_fact(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return True
    text = str(value).strip().lower()
    return text in {"", ".", "?", "none", "unknown", "unspecified", "nan"}


def _canonical_fact(value: Any) -> Any | None:
    """Return a stable scalar for an optional authoritative atom fact."""

    if isinstance(value, np.generic):
        value = value.item()
    if _unknown_fact(value):
        return None
    if isinstance(value, (bool, int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return int(number) if number.is_integer() else round(number, 8)
    text = str(value).strip().lower()
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else round(number, 8)


def _atom_facts(atoms: Any, rows: np.ndarray) -> dict[int, tuple[tuple[str, Any], ...]]:
    """Read optional charge/stereo facts without inventing missing chemistry.

    Reader versions and small test fixtures use different annotation names.
    A fact is a constraint only when the annotation exists and the value is
    explicit on both sides of a comparison; absent values remain wildcards.
    """

    from src.common.chemistry.charges import formal_charge_values
    charges = formal_charge_values(atoms, rows=rows)
    values_by_field: dict[str, np.ndarray] = {}
    for field, names in _ATOM_FACT_ANNOTATIONS.items():
        if field == "charge":
            continue
        for name in names:
            if not hasattr(atoms, name):
                continue
            try:
                values = np.asarray(getattr(atoms, name), dtype=object)
            except (TypeError, ValueError):
                continue
            element_count = len(np.asarray(getattr(atoms, "element", values)))
            if values.ndim == 1 and len(values) >= element_count:
                values_by_field[field] = values
                break
    result: dict[int, tuple[tuple[str, Any], ...]] = {}
    for position, raw_index in enumerate(rows.tolist()):
        index = int(raw_index)
        facts = []
        if charges is not None and charges[position] is not None:
            facts.append(("charge", charges[position]))
        for field, values in values_by_field.items():
            value = _canonical_fact(values[index])
            if value is not None:
                facts.append((field, value))
        result[index] = tuple(sorted(facts))
    return result


def _facts_compatible(left: tuple[tuple[str, Any], ...], right: tuple[tuple[str, Any], ...]) -> bool:
    left_values = dict(left)
    right_values = dict(right)
    return all(
        left_values[field] == right_values[field]
        for field in left_values.keys() & right_values.keys()
    )


def _order_compatible(left: float, right: float) -> bool:
    if left == right:
        return True
    if left in _AROMATIC_ORDER_EQUIVALENTS:
        return right in _AROMATIC_ORDER_EQUIVALENTS[left]
    if right in _AROMATIC_ORDER_EQUIVALENTS:
        return left in _AROMATIC_ORDER_EQUIVALENTS[right]
    return False


def _chain_values(atoms: Any) -> np.ndarray:
    categories = atoms.get_annotation_categories()
    if "label_asym_id" in categories:
        return np.asarray(atoms.label_asym_id).astype(str)
    return np.asarray(atoms.chain_id).astype(str)


def ligand_chains(atoms: Any) -> list[str]:
    values = _chain_values(atoms)
    mol_type = np.asarray(getattr(atoms, "mol_type", np.asarray([""] * len(values)))).astype(str)
    return list(dict.fromkeys(values[mol_type == "ligand"].tolist()))


def _element_values(atoms: Any, indices: np.ndarray) -> dict[int, str]:
    if not hasattr(atoms, "element"):
        raise LigandMappingError("ligand structure has no explicit element annotation")
    values = np.asarray(atoms.element).astype(str)
    result: dict[int, str] = {}
    for raw_index in indices.tolist():
        value = values[int(raw_index)].strip().upper()
        if not value or value in {".", "?"}:
            raise LigandMappingError("ligand has an unknown heavy-atom element")
        result[int(raw_index)] = value
    return result


def _heavy_indices(atoms: Any, chain: str) -> tuple[np.ndarray, dict[int, str]]:
    chains = _chain_values(atoms)
    selected = chains == str(chain)
    if hasattr(atoms, "mol_type"):
        selected &= np.asarray(atoms.mol_type).astype(str) == "ligand"
    rows = np.flatnonzero(selected).astype(np.int64)
    if not len(rows):
        raise LigandMappingError(f"ligand chain {chain!r} is absent")
    elements = _element_values(atoms, rows)
    heavy = np.asarray(
        [index for index in rows.tolist() if elements[int(index)] not in {"H", "D"}],
        dtype=np.int64,
    )
    if not len(heavy):
        raise LigandMappingError(f"ligand chain {chain!r} has no heavy atoms")
    return heavy, {int(index): elements[int(index)] for index in heavy.tolist()}


def _bond_order(value: Any) -> float:
    if isinstance(value, str):
        text = value.strip().lower()
        aliases = {"single": 1.0, "double": 2.0, "triple": 3.0, "aromatic": 4.0}
        if text in aliases:
            return aliases[text]
        try:
            value = float(text)
        except (TypeError, ValueError) as exc:
            raise LigandMappingError(f"unknown ligand bond order {value!r}") from exc
    try:
        order = float(value)
    except (TypeError, ValueError) as exc:
        raise LigandMappingError(f"unknown ligand bond order {value!r}") from exc
    if not np.isfinite(order) or order <= 0:
        raise LigandMappingError(f"invalid ligand bond order {value!r}")
    return order


def _adjacency(atoms: Any, indices: np.ndarray) -> dict[int, dict[int, float]]:
    bonds = getattr(atoms, "bonds", None)
    if bonds is None:
        raise LigandMappingError("ligand structure has no explicit bond graph")
    try:
        rows = np.asarray(bonds.as_array())
    except (AttributeError, TypeError, ValueError) as exc:
        raise LigandMappingError("ligand bond graph cannot be read") from exc
    index_set = {int(index) for index in indices.tolist()}
    graph = {index: {} for index in index_set}
    if rows.ndim != 2 or rows.shape[1] < 3:
        raise LigandMappingError("ligand bond graph has an invalid shape")
    for raw in rows.tolist():
        try:
            left, right = int(raw[0]), int(raw[1])
            order = _bond_order(raw[2])
        except (TypeError, ValueError, IndexError) as exc:
            raise LigandMappingError("ligand bond graph contains an invalid edge") from exc
        if left == right or left not in index_set or right not in index_set:
            continue
        previous = graph[left].get(right)
        if previous is not None and previous != order:
            raise LigandMappingError("ligand graph has conflicting duplicate bond orders")
        graph[left][right] = order
        graph[right][left] = order
    return graph


def _bond_stereo_facts(
    atoms: Any,
    indices: np.ndarray,
) -> dict[tuple[int, int], Any]:
    """Read optional per-edge stereo columns without requiring them."""

    bonds = getattr(atoms, "bonds", None)
    if bonds is None:
        return {}
    index_set = {int(index) for index in indices.tolist()}
    try:
        rows = np.asarray(bonds.as_array())
    except (AttributeError, TypeError, ValueError):
        return {}
    result: dict[tuple[int, int], Any] = {}
    for raw in rows.tolist():
        if len(raw) < 4:
            continue
        left, right = int(raw[0]), int(raw[1])
        if left not in index_set or right not in index_set or left == right:
            continue
        value = _canonical_fact(raw[3])
        if value is None:
            continue
        result[tuple(sorted((left, right)))] = value
    return result


def _node_signature(
    index: int,
    elements: Mapping[int, str],
    graph: Mapping[int, Mapping[int, float]],
) -> tuple[str, tuple[tuple[str, float], ...]]:
    neighbors = tuple(sorted((elements[neighbor], _order_key(order)) for neighbor, order in graph[index].items()))
    return elements[index], neighbors


def _order_key(order: float) -> tuple[str, float]:
    """Return the bond class used for candidate pruning."""

    if order in (5.0, 6.0, 7.0, 9.0):
        return "aromatic", 0.0 if order == 9.0 else order - 4.0
    return "plain", float(order)


def _external_attachment_facts(
    atoms: Any,
    component_rows: np.ndarray,
    heavy_rows: np.ndarray,
) -> dict[int, tuple[tuple[Any, ...], ...]]:
    """Return coordinate-free facts for bonds leaving a ligand component.

    The neighbour name and coordinates are deliberately omitted because
    refolders can rename/reorder their context atoms.  Element, modality and
    endpoint order are stable chemistry facts and are compared whenever both
    structures carry an explicit external bond.
    """

    bonds = getattr(atoms, "bonds", None)
    if bonds is None:
        return {int(row): () for row in heavy_rows.tolist()}
    component_set = {int(row) for row in component_rows.tolist()}
    heavy_set = {int(row) for row in heavy_rows.tolist()}
    elements = np.asarray(getattr(atoms, "element", ())).astype(str)
    modalities = np.asarray(
        getattr(atoms, "mol_type", np.asarray([""] * len(elements)))
    ).astype(str)
    result: dict[int, list[tuple[Any, ...]]] = {row: [] for row in heavy_set}
    try:
        rows = np.asarray(bonds.as_array())
    except (AttributeError, TypeError, ValueError):
        return {row: () for row in heavy_set}
    for raw in rows.tolist():
        if len(raw) < 3:
            continue
        left, right = int(raw[0]), int(raw[1])
        if (left in component_set) == (right in component_set):
            continue
        ligand_row, neighbour = (
            (left, right) if left in component_set else (right, left)
        )
        if ligand_row not in heavy_set:
            continue
        element = str(elements[neighbour]).strip().upper() if neighbour < len(elements) else ""
        modality = str(modalities[neighbour]).strip().lower() if neighbour < len(modalities) else ""
        try:
            order = _bond_order(raw[2])
        except LigandMappingError:
            raise
        result[ligand_row].append((order, element, modality))
    return {row: tuple(sorted(values, key=repr)) for row, values in result.items()}


def _external_compatible(
    left: tuple[tuple[Any, ...], ...],
    right: tuple[tuple[Any, ...], ...],
) -> bool:
    # An absent graph on one side is an unknown fact, not evidence of a
    # missing attachment.  Once both sides provide an explicit attachment,
    # all endpoint order/element/modality facts must agree.
    return not left or not right or left == right


def _stable_node_key(
    elements: Mapping[int, str],
    graph: Mapping[int, Mapping[int, float]],
    facts: Mapping[int, tuple[tuple[str, Any], ...]],
    external: Mapping[int, tuple[tuple[Any, ...], ...]],
    index: int,
    names: Mapping[int, str] | None = None,
) -> tuple[Any, ...]:
    return (
        elements[index],
        facts.get(index, ()),
        tuple(sorted((elements[n], _order_key(order)) for n, order in graph[index].items())),
        external.get(index, ()),
        str((names or {}).get(index, "")),
        index,
    )



def _context_component_groups(atoms: Any, chain: str) -> list[tuple[int, np.ndarray]]:
    """Return ligand rows grouped by the reader-stable component UID."""

    chains = _chain_values(atoms)
    modalities = np.asarray(
        getattr(atoms, "mol_type", np.asarray([""] * len(chains)))
    ).astype(str)
    rows = np.flatnonzero(
        (chains == str(chain)) & (modalities == "ligand")
    ).astype(np.int64)
    if not len(rows):
        raise LigandMappingError(f"ligand chain {chain!r} is absent")
    values = getattr(atoms, "component_uid", None)
    if values is None:
        raise LigandMappingError(
            "context ligand witness requires reader component_uid"
        )
    values = np.asarray(values)
    groups: list[tuple[int, np.ndarray]] = []
    seen: list[int] = []
    for raw_row in rows.tolist():
        try:
            uid = int(values[int(raw_row)])
        except (TypeError, ValueError) as exc:
            raise LigandMappingError(
                "context ligand component_uid is not an integer"
            ) from exc
        if uid not in seen:
            seen.append(uid)
            groups.append((uid, []))
        for current, members in groups:
            if current == uid:
                members.append(int(raw_row))
                break
    return [
        (uid, np.asarray(members, dtype=np.int64))
        for uid, members in groups
    ]


def _context_component_name(atoms: Any, rows: np.ndarray) -> str:
    values = {
        str(value).strip().upper()
        for value in np.asarray(atoms.res_name).astype(str)[rows]
    }
    if len(values) != 1:
        raise LigandMappingError("context ligand component has mixed IDs")
    value = next(iter(values))
    if not value or value in {".", "?"}:
        raise LigandMappingError("context ligand component has no ID")
    return value


def _context_locator(atoms: Any, rows: np.ndarray) -> tuple[Any, ...]:
    """Return stable non-coordinate locators used for repeated components."""

    chains = _chain_values(atoms)

    def unique(name: str, default: Any = None) -> Any:
        if not hasattr(atoms, name):
            return default
        values = np.asarray(getattr(atoms, name))[rows]
        values = [
            value.item() if isinstance(value, np.generic) else value
            for value in values.tolist()
        ]
        unique_values = list(dict.fromkeys(values))
        return unique_values[0] if len(unique_values) == 1 else default

    return (
        str(chains[int(rows[0])]) if len(rows) else "",
        unique("res_id"),
        unique("ins_code", ""),
        unique("auth_seq_id"),
        unique("label_seq_id"),
        unique("residue_uid"),
        unique("molecule_uid"),
    )


def _context_locator_score(
    source: tuple[Any, ...],
    target: tuple[Any, ...],
) -> int:
    score = 0
    if source[0] and target[0] and source[0] == target[0]:
        score += 1
    for index, weight in ((1, 8), (2, 4), (3, 6), (4, 5), (5, 2), (6, 1)):
        left, right = source[index], target[index]
        if (
            left is not None
            and right is not None
            and str(left) not in {"", ".", "?"}
            and left == right
        ):
            score += weight
    return score



def _context_component_graph_mapping(
    reference: Any,
    predicted: Any,
    reference_rows: np.ndarray,
    predicted_rows: np.ndarray,
    reference_atom_names: Mapping[int, str],
    expected_component: str,
) -> dict[int, int]:
    """Prove one observed component maps into one native predicted CCD."""

    expected_component = str(expected_component).strip().upper()
    if not expected_component or expected_component in {".", "?"}:
        raise LigandMappingError("context ligand CCD identity is empty")
    predicted_component = _context_component_name(predicted, predicted_rows)
    if predicted_component != expected_component:
        raise LigandMappingError(
            f"context ligand predicted CCD {predicted_component!r} "
            f"differs from authoritative {expected_component!r}"
        )
    ref_heavy = np.asarray(
        [
            int(row)
            for row in reference_rows.tolist()
            if str(reference.element[int(row)]).strip().upper() not in {"H", "D"}
        ],
        dtype=np.int64,
    )
    pred_heavy = np.asarray(
        [
            int(row)
            for row in predicted_rows.tolist()
            if str(predicted.element[int(row)]).strip().upper() not in {"H", "D"}
        ],
        dtype=np.int64,
    )
    if not len(ref_heavy):
        raise LigandMappingError("context ligand component has no observed heavy atoms")
    if len(pred_heavy) < len(ref_heavy):
        raise LigandMappingError("predicted context ligand lacks observed heavy atoms")
    reference_uids = getattr(reference, "atom_uid", None)
    if reference_uids is None:
        raise LigandMappingError("context ligand witness requires reference atom_uid")
    reference_uids = np.asarray(reference_uids)
    native_by_row: dict[int, str] = {}
    for row in ref_heavy.tolist():
        try:
            uid = int(reference_uids[int(row)])
        except (TypeError, ValueError) as exc:
            raise LigandMappingError(
                "context ligand reference atom_uid is invalid"
            ) from exc
        raw_name = reference_atom_names.get(uid)
        if raw_name in (None, "", ".", "?"):
            raise LigandMappingError(
                f"context ligand witness lacks native name for atom_uid {uid}"
            )
        native_by_row[int(row)] = str(raw_name).strip()
    if len(set(native_by_row.values())) != len(native_by_row):
        raise LigandMappingError(
            "context ligand witness has duplicate native atom names"
        )
    predicted_names: dict[str, int] = {}
    for row in pred_heavy.tolist():
        name = str(predicted.atom_name[int(row)]).strip()
        if not name or name in {".", "?"} or name in predicted_names:
            raise LigandMappingError(
                "predicted context ligand native atom names are absent or duplicated"
            )
        predicted_names[name] = int(row)
    mapping: dict[int, int] = {}
    for ref_row, native_name in native_by_row.items():
        pred_row = predicted_names.get(native_name)
        if pred_row is None:
            raise LigandMappingError(
                f"predicted context ligand lacks native atom {native_name!r}"
            )
        mapping[ref_row] = pred_row
        ref_element = str(reference.element[ref_row]).strip().upper()
        pred_element = str(predicted.element[pred_row]).strip().upper()
        if ref_element != pred_element:
            raise LigandMappingError(
                f"context ligand native atom {native_name!r} has a different element"
            )

    ref_graph = _adjacency(reference, ref_heavy)
    pred_graph = _adjacency(predicted, pred_heavy)
    inverse = {pred: ref for ref, pred in mapping.items()}
    for ref_left, neighbours in ref_graph.items():
        for ref_right, ref_order in neighbours.items():
            if ref_left >= ref_right:
                continue
            pred_left = mapping[ref_left]
            pred_right = mapping[ref_right]
            pred_order = pred_graph[pred_left].get(pred_right)
            if pred_order is None or not _order_compatible(ref_order, pred_order):
                raise LigandMappingError(
                    "context ligand observed bond graph differs from predicted induced graph"
                )
    for pred_left, neighbours in pred_graph.items():
        if pred_left not in inverse:
            continue
        for pred_right, pred_order in neighbours.items():
            if pred_right not in inverse or pred_left >= pred_right:
                continue
            ref_left = inverse[pred_left]
            ref_right = inverse[pred_right]
            ref_order = ref_graph[ref_left].get(ref_right)
            if ref_order is None or not _order_compatible(ref_order, pred_order):
                raise LigandMappingError(
                    "predicted context ligand has an incompatible observed bond"
                )

    ref_facts = _atom_facts(reference, ref_heavy)
    pred_facts = _atom_facts(predicted, pred_heavy)
    for ref_row, pred_row in mapping.items():
        ref_values = dict(ref_facts.get(ref_row, ()))
        pred_values = dict(pred_facts.get(pred_row, ()))
        if any(pred_values.get(field) != value for field, value in ref_values.items()):
            raise LigandMappingError(
                "context ligand authoritative atom facts differ"
            )
    ref_stereo = _bond_stereo_facts(reference, ref_heavy)
    pred_stereo = _bond_stereo_facts(predicted, pred_heavy)
    for ref_edge, value in ref_stereo.items():
        pred_edge = tuple(sorted((mapping[ref_edge[0]], mapping[ref_edge[1]])))
        if pred_stereo.get(pred_edge) != value:
            raise LigandMappingError(
                "context ligand authoritative bond stereo differs"
            )

    ref_external = _external_attachment_facts(
        reference, np.asarray(reference_rows, dtype=np.int64), ref_heavy
    )
    pred_external = _external_attachment_facts(
        predicted, np.asarray(predicted_rows, dtype=np.int64), pred_heavy
    )
    for ref_row, pred_row in mapping.items():
        if ref_external.get(ref_row, ()) != pred_external.get(pred_row, ()):
            raise LigandMappingError(
                "context ligand explicit attachment differs"
            )
    return mapping



def _context_chain_atom_mapping(
    reference: Any,
    predicted: Any,
    reference_chain: str,
    predicted_chain: str,
    resolved_components: Mapping[tuple[str, int], Mapping[str, Any]],
    reference_atom_names: Mapping[int, str],
) -> dict[int, int]:
    ref_groups = _context_component_groups(reference, reference_chain)
    pred_groups = _context_component_groups(predicted, predicted_chain)
    expected: dict[int, Mapping[str, Any]] = {}
    for raw_key, value in resolved_components.items():
        try:
            chain, uid = raw_key
            key = (str(chain), int(uid))
        except (TypeError, ValueError) as exc:
            raise LigandMappingError("context ligand component key is invalid") from exc
        if key[0] == str(reference_chain):
            expected[key[1]] = value
    ref_uids = {uid for uid, _ in ref_groups}
    if set(expected) != ref_uids:
        raise LigandMappingError(
            f"context ligand witness does not cover reference chain {reference_chain!r}"
        )
    if len(ref_groups) != len(pred_groups):
        raise LigandMappingError(
            "context ligand component count differs from authoritative witness"
        )
    choices: dict[int, list[tuple[int, dict[int, int], int]]] = {}
    for ref_uid, ref_rows in ref_groups:
        entry = expected[ref_uid]
        ccd = entry.get("ccd") if isinstance(entry, Mapping) else None
        if not isinstance(ccd, (list, tuple)) or len(ccd) != 1:
            raise LigandMappingError(
                "context extra-atom witness requires one authoritative CCD per component"
            )
        ccd_name = str(ccd[0]).strip().upper()
        choices[ref_uid] = []
        for target_index, (_pred_uid, pred_rows) in enumerate(pred_groups):
            try:
                mapping = _context_component_graph_mapping(
                    reference, predicted, ref_rows, pred_rows,
                    reference_atom_names, ccd_name,
                )
            except LigandMappingError:
                continue
            score = _context_locator_score(
                _context_locator(reference, ref_rows),
                _context_locator(predicted, pred_rows),
            )
            choices[ref_uid].append((target_index, mapping, score))
        if not choices[ref_uid]:
            raise LigandMappingError(
                f"context ligand component UID {ref_uid} has no verified predicted CCD"
            )
    order = sorted(choices, key=lambda uid: (len(choices[uid]), uid))
    assignment: dict[int, tuple[int, dict[int, int]]] = {}
    used: set[int] = set()

    def visit(depth: int) -> bool:
        if depth == len(order):
            return True
        ref_uid = order[depth]
        candidates = sorted(
            choices[ref_uid],
            key=lambda value: (-value[2], value[0]),
        )
        for target_index, mapping, _score in candidates:
            if target_index in used:
                continue
            used.add(target_index)
            assignment[ref_uid] = (target_index, mapping)
            if visit(depth + 1):
                return True
            assignment.pop(ref_uid)
            used.remove(target_index)
        return False

    if not visit(0):
        raise LigandMappingError(
            f"context ligand components cannot be assigned for {reference_chain!r}"
        )
    result: dict[int, int] = {}
    for ref_uid, (_target_index, mapping) in assignment.items():
        for ref_row, pred_row in mapping.items():
            if ref_row in result and result[ref_row] != pred_row:
                raise LigandMappingError("context ligand atom map is inconsistent")
            result[ref_row] = pred_row
    return result


def context_atom_mapping(
    reference: Any,
    predicted: Any,
    reference_chain: str,
    predicted_chain: str,
    resolved_components: Mapping[tuple[str, int], Mapping[str, Any]],
    reference_atom_names: Mapping[int, str],
) -> dict[int, int]:
    """Return a verified row-index map for one context ligand chain.

    The witness must come from resolve_context_ligand_components. It permits
    native predicted CCD atoms absent from the observed reference only after
    proving every observed atom, induced graph edge/order, authoritative fact,
    and explicit attachment.
    """

    return _context_chain_atom_mapping(
        reference, predicted, reference_chain, predicted_chain,
        resolved_components, reference_atom_names,
    )


def context_ligand_atom_maps(
    reference: Any,
    predicted: Any,
    resolved_components: Mapping[tuple[str, int], Mapping[str, Any]],
    reference_atom_names: Mapping[int, str],
    *,
    predicted_chain_map: Mapping[str, str] | None = None,
) -> dict[tuple[str, str], dict[int, int]]:
    """Return verified row-index maps for all resolved context ligand chains."""

    reference_chains = sorted({str(key[0]) for key in resolved_components})
    target_chains = ligand_chains(predicted)
    result: dict[tuple[str, str], dict[int, int]] = {}
    for reference_chain in reference_chains:
        if predicted_chain_map is not None:
            predicted_chain = predicted_chain_map.get(reference_chain)
            if predicted_chain is None:
                raise LigandMappingError(
                    f"context ligand chain {reference_chain!r} has no predicted chain"
                )
            candidates = [str(predicted_chain)]
        else:
            candidates = sorted(target_chains)
        matches: list[tuple[str, dict[int, int]]] = []
        for predicted_chain in candidates:
            try:
                mapping = _context_chain_atom_mapping(
                    reference, predicted, reference_chain, predicted_chain,
                    resolved_components, reference_atom_names,
                )
            except LigandMappingError:
                continue
            matches.append((predicted_chain, mapping))
        if not matches:
            raise LigandMappingError(
                f"context ligand chain {reference_chain!r} has no verified predicted chain"
            )
        if predicted_chain_map is None and len(matches) > 1:
            canonical = [item for item in matches if item[0] == reference_chain]
            if len(canonical) == 1:
                matches = canonical
            else:
                raise LigandMappingError(
                    f"context ligand chain {reference_chain!r} has ambiguous predicted chains"
                )
        predicted_chain, mapping = matches[0]
        result[(reference_chain, predicted_chain)] = mapping
    return result


def _trusted_smiles_graphs(
    reference: Any,
    predicted: Any,
    ref_indices: np.ndarray,
    pred_indices: np.ndarray,
    smiles: str,
) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    """Prove both complete graphs against a resolved context SMILES identity.

    The trusted molecule supplies its aromatic and concrete Kekule orders.
    Normalization is local to this comparison; atom/bond arrays and ordinary
    graph matching are unchanged. Missing or extra heavy atoms are rejected.
    """

    from ..design.rewards.af3_chemistry import (
        ChemistryError, map_source_to_molecule, molecule_graph, source_graph,
        trusted_smiles,
    )

    try:
        molecule = trusted_smiles(smiles)
        trusted = molecule_graph(molecule)
        expected = set(range(len(trusted.names)))
        graphs = []
        for atoms, indices in ((reference, ref_indices), (predicted, pred_indices)):
            source = source_graph(atoms, indices, preserve_aromatic_orders=True)
            mapping = map_source_to_molecule(source, molecule)
            if len(indices) != len(expected) or set(mapping) != expected:
                raise ChemistryError("context SMILES requires every trusted heavy atom")
            by_trusted = {
                trusted_index: int(indices[source_index])
                for source_index, trusted_index in enumerate(mapping)
            }
            graph = {int(index): {} for index in indices.tolist()}
            for left, right, order in trusted.bonds:
                left_row, right_row = by_trusted[left], by_trusted[right]
                graph[left_row][right_row] = float(order)
                graph[right_row][left_row] = float(order)
            graphs.append(graph)
    except (ChemistryError, TypeError, ValueError) as exc:
        raise LigandMappingError(f"trusted context SMILES graph proof failed: {exc}") from exc
    return graphs[0], graphs[1]


def atom_mapping(
    reference: Any,
    predicted: Any,
    reference_chain: str,
    predicted_chain: str,
    *,
    authoritative_mapping: Mapping[int, int] | None = None,
    allow_predicted_extra: bool = False,
    trusted_smiles: str | None = None,
) -> dict[int, int]:
    """Return a deterministic reference-atom -> predicted-atom isomorphism.

    Graph automorphisms are chemically equivalent.  They are therefore
    accepted and resolved by a stable search order instead of being reported
    as an ambiguity.  Atom names, formal charge/stereo annotations and
    explicit external attachments constrain that search when both structures
    provide the corresponding fact.  ``authoritative_mapping`` is an
    optional input-side witness for callers that already have one; it is
    validated against the same graph contract before reuse. Predicted extra
    atoms are accepted only with that witness and only after induced-graph
    validation. A separately resolved fixed-context ``trusted_smiles`` can
    prove equivalent aromatic/Kekule representations of two complete graphs;
    all atom facts, stereo and external-attachment checks still apply.
    """

    ref_indices, ref_elements = _heavy_indices(reference, reference_chain)
    pred_indices, pred_elements = _heavy_indices(predicted, predicted_chain)
    if trusted_smiles is not None and allow_predicted_extra:
        raise LigandMappingError("context SMILES cannot authorize predicted extra atoms")
    if allow_predicted_extra and authoritative_mapping is None:
        raise LigandMappingError(
            "predicted extra ligand atoms require an authoritative context witness"
        )
    if allow_predicted_extra:
        if len(pred_indices) < len(ref_indices):
            raise LigandMappingError(
                f"ligand heavy-atom count differs ({len(ref_indices)} > {len(pred_indices)})"
            )
    elif len(ref_indices) != len(pred_indices):
        raise LigandMappingError(
            f"ligand heavy-atom count differs ({len(ref_indices)} != {len(pred_indices)})"
        )
    ref_graph = _adjacency(reference, ref_indices)
    pred_graph = _adjacency(predicted, pred_indices)
    if trusted_smiles is not None:
        ref_graph, pred_graph = _trusted_smiles_graphs(
            reference, predicted, ref_indices, pred_indices, trusted_smiles
        )
    ref_bond_stereo = _bond_stereo_facts(reference, ref_indices)
    pred_bond_stereo = _bond_stereo_facts(predicted, pred_indices)
    ref_facts = _atom_facts(reference, ref_indices)
    pred_facts = _atom_facts(predicted, pred_indices)
    ref_component_rows = np.flatnonzero(
        _chain_values(reference) == str(reference_chain)
    ).astype(np.int64)
    pred_component_rows = np.flatnonzero(
        _chain_values(predicted) == str(predicted_chain)
    ).astype(np.int64)
    ref_external = _external_attachment_facts(reference, ref_component_rows, ref_indices)
    pred_external = _external_attachment_facts(predicted, pred_component_rows, pred_indices)

    def neighbor_signature_compatible(ref_index: int, pred_index: int) -> bool:
        """Compare local element/order multisets with aromatic equivalence."""

        reference_neighbors = sorted(
            (
                ref_elements[neighbor],
                order,
            )
            for neighbor, order in ref_graph[ref_index].items()
        )
        predicted_neighbors = sorted(
            (
                pred_elements[neighbor],
                order,
            )
            for neighbor, order in pred_graph[pred_index].items()
        )
        if len(reference_neighbors) != len(predicted_neighbors):
            return False
        used: set[int] = set()

        def visit(depth: int) -> bool:
            if depth == len(reference_neighbors):
                return True
            element, order = reference_neighbors[depth]
            for candidate, (candidate_element, candidate_order) in enumerate(
                predicted_neighbors
            ):
                if candidate in used:
                    continue
                if element != candidate_element or not _order_compatible(order, candidate_order):
                    continue
                used.add(candidate)
                if visit(depth + 1):
                    return True
                used.remove(candidate)
            return False

        return visit(0)

    def valid_pair(ref_index: int, pred_index: int) -> bool:
        return (
            ref_elements[ref_index] == pred_elements[pred_index]
            and neighbor_signature_compatible(ref_index, pred_index)
            and _facts_compatible(
                ref_facts.get(ref_index, ()), pred_facts.get(pred_index, ())
            )
            and _external_compatible(
                ref_external.get(ref_index, ()), pred_external.get(pred_index, ())
            )
        )

    def valid_mapping(mapping: Mapping[int, int]) -> bool:
        ref_set = set(ref_indices.tolist())
        pred_set = set(pred_indices.tolist())
        if set(mapping) != ref_set or len(set(mapping.values())) != len(mapping):
            return False
        if allow_predicted_extra:
            if not set(mapping.values()) <= pred_set:
                return False
        elif set(mapping.values()) != pred_set:
            return False
        for ref_index, pred_index in mapping.items():
            if allow_predicted_extra:
                if ref_elements[int(ref_index)] != pred_elements[int(pred_index)]:
                    return False
                reference_facts = dict(ref_facts.get(int(ref_index), ()))
                predicted_facts = dict(pred_facts.get(int(pred_index), ()))
                if any(
                    predicted_facts.get(field) != value
                    for field, value in reference_facts.items()
                ):
                    return False
                reference_external = ref_external.get(int(ref_index), ())
                if reference_external and reference_external != pred_external.get(
                    int(pred_index), ()
                ):
                    return False
            elif not valid_pair(int(ref_index), int(pred_index)):
                return False
        inverse = {int(pred): int(ref) for ref, pred in mapping.items()}
        for ref_left in ref_indices.tolist():
            for ref_right, ref_order in ref_graph[int(ref_left)].items():
                if ref_right not in mapping:
                    return False
                pred_order = pred_graph[int(mapping[ref_left])].get(mapping[ref_right])
                if pred_order is None or not _order_compatible(ref_order, pred_order):
                    return False
                ref_stereo = ref_bond_stereo.get(tuple(sorted((int(ref_left), int(ref_right)))))
                pred_stereo = pred_bond_stereo.get(
                    tuple(sorted((int(mapping[ref_left]), int(mapping[ref_right]))))
                )
                if ref_stereo is not None and (
                    (allow_predicted_extra and pred_stereo is None)
                    or (pred_stereo is not None and ref_stereo != pred_stereo)
                ):
                    return False
        if allow_predicted_extra:
            for pred_left, neighbours in pred_graph.items():
                if pred_left not in inverse:
                    continue
                for pred_right, pred_order in neighbours.items():
                    if pred_right not in inverse or pred_left >= pred_right:
                        continue
                    ref_order = ref_graph[inverse[pred_left]].get(inverse[pred_right])
                    if ref_order is None or not _order_compatible(ref_order, pred_order):
                        return False
        return True

    if authoritative_mapping is not None:
        try:
            witness = {
                int(reference_index): int(predicted_index)
                for reference_index, predicted_index in authoritative_mapping.items()
            }
        except (TypeError, ValueError) as exc:
            raise LigandMappingError("authoritative ligand atom mapping is invalid") from exc
        if not valid_mapping(witness):
            raise LigandMappingError("authoritative ligand atom mapping violates graph chemistry")
        return witness

    candidates = {
        index: [
            candidate for candidate in pred_indices.tolist()
            if valid_pair(int(index), int(candidate))
        ]
        for index in ref_indices.tolist()
    }
    if any(not values for values in candidates.values()):
        raise LigandMappingError("ligand graph has no element/bond-order correspondence")

    ref_names = {
        int(index): str(getattr(reference, "atom_name")[int(index)]).strip()
        for index in ref_indices.tolist()
        if hasattr(reference, "atom_name")
    }
    pred_names = {
        int(index): str(getattr(predicted, "atom_name")[int(index)]).strip()
        for index in pred_indices.tolist()
        if hasattr(predicted, "atom_name")
    }
    # A retained input name is an authoritative witness.  Use it whenever the
    # two structures share the complete name set and the witness validates;
    # otherwise names remain only a deterministic tie-break for automorphisms.
    if ref_names and pred_names and set(ref_names.values()) == set(pred_names.values()):
        by_name = {
            ref_index: next(
                pred_index for pred_index, name in pred_names.items()
                if name == ref_names[ref_index]
            )
            for ref_index in ref_names
        }
        if valid_mapping(by_name):
            return by_name

    def compatible(ref_index: int, pred_index: int, mapping: Mapping[int, int]) -> bool:
        for assigned_ref, assigned_pred in mapping.items():
            ref_order = ref_graph[ref_index].get(assigned_ref)
            pred_order = pred_graph[pred_index].get(assigned_pred)
            if ref_order is None and pred_order is None:
                continue
            if ref_order is None or pred_order is None or not _order_compatible(ref_order, pred_order):
                return False
        return True

    order = sorted(
        ref_indices.tolist(),
        key=lambda index: (
            len(candidates[int(index)]),
            -len(ref_graph[int(index)]),
            _stable_node_key(
                ref_elements, ref_graph, ref_facts, ref_external,
                int(index), ref_names,
            ),
        ),
    )

    def search(mapping: dict[int, int], used: set[int]) -> bool:
        if len(mapping) == len(ref_indices):
            return True
        ref_index = int(order[len(mapping)])
        candidates_for_ref = sorted(
            candidates[ref_index],
            key=lambda candidate: _stable_node_key(
                pred_elements, pred_graph, pred_facts, pred_external,
                int(candidate), pred_names,
            ),
        )
        for candidate in candidates_for_ref:
            candidate = int(candidate)
            if candidate in used or not compatible(int(ref_index), candidate, mapping):
                continue
            mapping[int(ref_index)] = candidate
            used.add(candidate)
            if search(mapping, used):
                return True
            used.remove(candidate)
            del mapping[int(ref_index)]
        return False

    mapping: dict[int, int] = {}
    if not search(mapping, set()) or not valid_mapping(mapping):
        raise LigandMappingError("ligand element/bond-order graph is not isomorphic")
    return dict(mapping)


def match_ligand_chains(
    reference: Any,
    predicted: Any,
    *,
    authoritative_context_maps: Mapping[
        tuple[str, str], Mapping[int, int]
    ] | None = None,
    context_smiles: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[tuple[str, str], dict[int, int]], dict[str, str]]:
    """Match ligand chains using strict graphs or verified context witnesses.

    ``authoritative_context_maps`` is the output of
    :func:`context_ligand_atom_maps`.  It is accepted only for the keyed
    reference/predicted chain pair and is revalidated by :func:`atom_mapping`.
    Without it, generated and joint ligands retain strict equal-heavy-atom
    graph matching. ``context_smiles`` must come from independently resolved
    fixed context chemistry and authorizes only equal, complete heavy graphs.
    """

    reference_chains = ligand_chains(reference)
    predicted_chains = ligand_chains(predicted)
    available = set(predicted_chains)
    chain_map: dict[str, str] = {}
    atom_maps: dict[tuple[str, str], dict[int, int]] = {}
    reasons: dict[str, str] = {}
    witnesses = authoritative_context_maps or {}
    smiles_by_chain = context_smiles or {}

    for reference_chain in reference_chains:
        keyed = [
            (str(predicted_chain), mapping)
            for (ref_chain, predicted_chain), mapping in witnesses.items()
            if str(ref_chain) == reference_chain
        ]
        if keyed:
            keyed = [item for item in keyed if item[0] in available]
            if len(keyed) != 1:
                reasons[reference_chain] = (
                    "authoritative context ligand chain mapping is absent or ambiguous"
                )
                continue
            predicted_chain, witness = keyed[0]
            try:
                mapping = atom_mapping(
                    reference, predicted, reference_chain, predicted_chain,
                    authoritative_mapping=witness,
                    allow_predicted_extra=True,
                )
            except LigandMappingError as exc:
                reasons[reference_chain] = f"authoritative context map rejected: {exc}"
                continue
            chain_map[reference_chain] = predicted_chain
            atom_maps[(reference_chain, predicted_chain)] = mapping
            available.remove(predicted_chain)
            continue

        # A canonical chain label is a strong identity hint, but the graph
        # proof is still mandatory. Prefer it when the strict proof succeeds.
        if reference_chain in available:
            try:
                canonical_mapping = atom_mapping(
                    reference, predicted, reference_chain, reference_chain,
                    trusted_smiles=smiles_by_chain.get(reference_chain),
                )
            except LigandMappingError:
                canonical_mapping = None
            if canonical_mapping is not None:
                chain_map[reference_chain] = reference_chain
                atom_maps[(reference_chain, reference_chain)] = canonical_mapping
                available.remove(reference_chain)
                continue

        candidates: list[tuple[str, dict[int, int]]] = []
        for predicted_chain in sorted(available):
            try:
                mapping = atom_mapping(
                    reference, predicted, reference_chain, predicted_chain,
                    trusted_smiles=smiles_by_chain.get(reference_chain),
                )
            except LigandMappingError:
                continue
            candidates.append((predicted_chain, mapping))
        if len(candidates) == 1:
            predicted_chain, mapping = candidates[0]
            chain_map[reference_chain] = predicted_chain
            atom_maps[(reference_chain, predicted_chain)] = mapping
            available.remove(predicted_chain)
        elif len(candidates) > 1:
            reasons[reference_chain] = "ligand graph mapping is ambiguous"
        else:
            reasons[reference_chain] = "no unique ligand element/bond-order graph mapping"
    return chain_map, atom_maps, reasons

def named_atom_mapping(reference: Any, predicted: Any, chain: str) -> dict[int, int]:
    """Map every observed reference heavy atom by its retained name and connectivity.

    Refolded CCD completion may add atoms absent in the generated coordinates.
    Their coordinates have no reference and are not part of this comparison.
    Bond connectivity on the observed atoms must agree; aromatic/Kekule bond
    order representations do not change the atom-name correspondence.
    """
    ref_rows, ref_elements = _heavy_indices(reference, chain)
    pred_rows, pred_elements = _heavy_indices(predicted, chain)
    def by_name(atoms, rows):
        result = {}
        for row in rows.tolist():
            name = str(atoms.atom_name[row])
            if not name or name in result:
                raise LigandMappingError("ligand atom names are absent or duplicated")
            result[name] = int(row)
        return result
    ref_names, pred_names = by_name(reference, ref_rows), by_name(predicted, pred_rows)
    if not set(ref_names) <= set(pred_names):
        raise LigandMappingError("refold lacks named reference ligand atoms")
    mapping = {row: pred_names[name] for name, row in ref_names.items()}
    if any(ref_elements[a] != pred_elements[b] for a, b in mapping.items()):
        raise LigandMappingError("named ligand atom elements differ")
    ref_graph = _adjacency(reference, ref_rows)
    pred_graph = _adjacency(predicted, pred_rows)
    mapped = set(mapping.values())
    for source, target in mapping.items():
        if {mapping[neighbor] for neighbor in ref_graph[source]} != set(pred_graph[target]) & mapped:
            raise LigandMappingError("named reference ligand connectivity differs")
        for neighbor, ref_order in ref_graph[source].items():
            pred_order = pred_graph[target].get(mapping[neighbor])
            if pred_order is None or not _order_compatible(ref_order, pred_order):
                raise LigandMappingError("named reference ligand bond order differs")
    return mapping
