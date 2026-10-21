"""Context-ligand chemistry resolution for standalone ESMFold2 validation.

Validation inputs deliberately do not grow a second ligand_entries field.
Instead, context chemistry is recovered from the immutable task snapshot and
the authoritative source structure. Coordinates are never used to infer
bonds, charges, aromaticity, or a component identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import load_yaml


class LigandResolutionError(ValueError):
    """A context ligand has no unambiguous authoritative chemistry."""


_PLACEHOLDER_COMPONENTS = frozenset({
    "", ".", "?", "UNL", "LIG", "LIGAND", "UNK", "UNKNOWN", "MOL", "MOV",
})


@dataclass(frozen=True)
class _Graph:
    """Heavy-atom graph in the order supplied by the caller."""

    names: tuple[str, ...]
    elements: tuple[str, ...]
    edges: tuple[tuple[int, int, str], ...]
    adjacency: tuple[dict[int, str], ...]
    atom_facts: tuple[tuple[tuple[str, Any], ...], ...] = ()
    external: tuple[tuple[tuple[Any, ...], ...], ...] = ()


_ORDER_NAMES = {
    1: "single",
    2: "double",
    3: "triple",
    # Biotite 4 is QUADRUPLE; aromatic bonds are 5-7 or generic 9.
    5: "aromatic",
    6: "aromatic",
    7: "aromatic",
    9: "aromatic",
}


def _repo_root(task: Path) -> Path:
    task = Path(task).expanduser().resolve()
    for parent in (task, *task.parents):
        if (parent / "src").is_dir() and (parent / "configs").is_dir():
            return parent
    # This is only a fallback for a test/embedded checkout. A missing
    # resolved snapshot remains an explicit error below.
    return task.parents[2] if len(task.parents) > 2 else task.parent


def _task_config(
    task: Path, *, allow_missing: bool = False
) -> tuple[dict[str, Any], dict[str, Any], Path | None]:
    task = Path(task).expanduser().resolve()
    snapshot = task / "resolved_config.yaml"
    if not snapshot.is_file():
        if allow_missing:
            return {}, {}, None
        raise LigandResolutionError(
            f"context ligand chemistry requires {snapshot}"
        )
    try:
        config = load_yaml(snapshot)
    except Exception as exc:
        raise LigandResolutionError(
            f"cannot read resolved validation task config {snapshot}: {exc}"
        ) from exc
    tasks = config.get("tasks")
    if not isinstance(tasks, Mapping):
        if allow_missing:
            return dict(config), {}, None
        raise LigandResolutionError(f"{snapshot} has no tasks mapping")
    task_config = tasks.get(task.name)
    if not isinstance(task_config, Mapping):
        if allow_missing:
            return dict(config), {}, None
        raise LigandResolutionError(
            f"{snapshot} has no resolved task {task.name!r}"
        )
    raw_input = task_config.get("input")
    if not isinstance(raw_input, str) or not raw_input.strip():
        if allow_missing:
            return dict(config), dict(task_config), None
        raise LigandResolutionError(
            f"resolved task {task.name!r} has no authoritative input structure"
        )
    source = Path(raw_input).expanduser()
    if not source.is_absolute():
        source = _repo_root(task) / source
    source = source.resolve()
    return dict(config), dict(task_config), source


def _design_chains(task_config: Mapping[str, Any]) -> set[str]:
    raw = task_config.get("design_chains", ())
    if isinstance(raw, Mapping):
        return {str(key) for key in raw}
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return set()
    result: set[str] = set()
    for item in raw:
        if isinstance(item, Mapping):
            result.update(str(key) for key in item)
    return result


def _ligand_conditions(task_config: Mapping[str, Any]) -> Mapping[str, Any]:
    condition = task_config.get("condition", {})
    if not isinstance(condition, Mapping):
        return {}
    value = condition.get("ligand", {})
    return value if isinstance(value, Mapping) else {}


def _condition_for_chain(task_config: Mapping[str, Any], chain: str) -> Mapping[str, Any] | None:
    value = _ligand_conditions(task_config).get(chain)
    return value if isinstance(value, Mapping) else None


def _canonical_chains(atoms: Any) -> np.ndarray:
    labels = getattr(atoms, "label_asym_id", None)
    if labels is not None:
        values = np.asarray(labels).astype(str)
        if values.size and all(value not in {"", ".", "?"} for value in values):
            return values
    return np.asarray(atoms.chain_id).astype(str)


def _normalize_element(value: Any) -> str:
    text = str(value).strip()
    if not text or text in {".", "?"}:
        raise LigandResolutionError("ligand graph has an unknown atom element")
    try:
        from rdkit import Chem
        symbol = text[:1].upper() + text[1:].lower()
        atomic_number = int(Chem.GetPeriodicTable().GetAtomicNumber(symbol))
        if atomic_number <= 1:
            raise ValueError
        return Chem.GetPeriodicTable().GetElementSymbol(atomic_number).upper()
    except Exception as exc:
        raise LigandResolutionError(
            f"ligand graph has unsupported atom element {text!r}"
        ) from exc


def _bond_order(value: Any) -> str:
    try:
        result = _ORDER_NAMES[int(value)]
    except (KeyError, TypeError, ValueError) as exc:
        raise LigandResolutionError(
            f"ligand graph has unsupported or unknown bond order {value!r}"
        ) from exc
    return result


def _rows_for_chain(atoms: Any, chain: str) -> np.ndarray:
    chains = _canonical_chains(atoms)
    modalities = np.asarray(getattr(atoms, "mol_type")).astype(str)
    rows = np.flatnonzero((chains == str(chain)) & (modalities == "ligand"))
    if len(rows) == 0:
        raise LigandResolutionError(
            f"context ligand chain {chain!r} is absent from the validation structure"
        )
    return rows


def _component_rows(atoms: Any, rows: Sequence[int]) -> list[np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    component = getattr(atoms, "component_uid", None)
    if component is not None:
        values = np.asarray(component, dtype=np.int64)[rows]
        groups: list[np.ndarray] = []
        for value in dict.fromkeys(values.tolist()):
            groups.append(rows[values == value])
        return groups
    names = np.asarray(atoms.res_name).astype(str)[rows]
    groups = []
    for value in dict.fromkeys(names.tolist()):
        groups.append(rows[names == value])
    return groups


_ATOM_FACT_ANNOTATIONS = {
    "charge": ("formal_charge", "charge", "pdbx_formal_charge", "atom_charge"),
    "stereo": (
        "atom_stereo", "stereo", "stereochemistry", "stereo_config",
        "chirality", "pdbx_stereo_config",
    ),
}


def _canonical_fact(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    text = str(value).strip().lower()
    if text in {"", ".", "?", "none", "unknown", "unspecified", "nan"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return text
    if not np.isfinite(number):
        return None
    return int(number) if number.is_integer() else round(number, 8)


def _atom_facts(atoms: Any, rows: Sequence[int]) -> tuple[tuple[tuple[str, Any], ...], ...]:
    from src.common.chemistry.charges import formal_charge_values
    rows = np.asarray(rows, dtype=np.int64)
    try:
        charges = formal_charge_values(atoms, rows=rows)
    except ValueError as exc:
        raise LigandResolutionError(str(exc)) from exc
    fields: dict[str, np.ndarray] = {}
    length = len(np.asarray(getattr(atoms, "element")))
    for field, names in _ATOM_FACT_ANNOTATIONS.items():
        if field == "charge":
            continue
        for name in names:
            if not hasattr(atoms, name):
                continue
            values = np.asarray(getattr(atoms, name), dtype=object)
            if values.ndim == 1 and len(values) >= length:
                fields[field] = values
                break
    result = []
    for position, row in enumerate(rows.tolist()):
        values = []
        if charges is not None and charges[position] is not None:
            values.append(("charge", charges[position]))
        for field, array in fields.items():
            value = _canonical_fact(array[int(row)])
            if value is not None:
                values.append((field, value))
        result.append(tuple(sorted(values)))
    return tuple(result)


def _external_attachment_facts(
    atoms: Any,
    component_rows: Sequence[int],
    heavy_rows: Sequence[int],
) -> tuple[tuple[tuple[Any, ...], ...], ...]:
    bonds = getattr(atoms, "bonds", None)
    heavy_rows = np.asarray(heavy_rows, dtype=np.int64)
    if bonds is None:
        return tuple(() for _ in heavy_rows)
    component_set = {int(row) for row in np.asarray(component_rows, dtype=np.int64).tolist()}
    heavy_set = {int(row) for row in heavy_rows.tolist()}
    elements = np.asarray(getattr(atoms, "element")).astype(str)
    modalities = np.asarray(
        getattr(atoms, "mol_type", np.asarray([""] * len(elements)))
    ).astype(str)
    result: dict[int, list[tuple[Any, ...]]] = {int(row): [] for row in heavy_rows.tolist()}
    try:
        raw_bonds = bonds.as_array().tolist()
    except (AttributeError, TypeError, ValueError) as exc:
        raise LigandResolutionError("ligand bond graph cannot be read") from exc
    for raw in raw_bonds:
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
        order = _bond_order(raw[2])
        element = str(elements[neighbour]).strip().upper() if neighbour < len(elements) else ""
        modality = str(modalities[neighbour]).strip().lower() if neighbour < len(modalities) else ""
        result[ligand_row].append((order, element, modality))
    return tuple(
        tuple(sorted(result[int(row)], key=repr)) for row in heavy_rows.tolist()
    )


def _graph_from_atoms(atoms: Any, rows: Sequence[int]) -> _Graph:
    rows = np.asarray(rows, dtype=np.int64)
    elements_all = np.asarray(getattr(atoms, "element")).astype(str)
    heavy = [
        int(row) for row in rows.tolist()
        if elements_all[int(row)].strip().upper() not in {"H", "D", "T"}
    ]
    if not heavy:
        raise LigandResolutionError("context ligand has no heavy atoms")
    names = tuple(str(atoms.atom_name[row]).strip() for row in heavy)
    if any(not name or name in {".", "?"} for name in names):
        raise LigandResolutionError("context ligand has missing atom names")
    if len(names) != len(set(names)):
        raise LigandResolutionError("context ligand atom names are ambiguous")
    elements = tuple(_normalize_element(elements_all[row]) for row in heavy)
    row_to_local = {row: index for index, row in enumerate(heavy)}
    edges: dict[tuple[int, int], str] = {}
    source = getattr(atoms, "bonds", None)
    if source is None:
        raise LigandResolutionError("context ligand has no complete bond graph")
    for raw_i, raw_j, raw_order in source.as_array().tolist():
        row_i, row_j = int(raw_i), int(raw_j)
        if row_i not in row_to_local or row_j not in row_to_local:
            continue
        if row_i == row_j:
            raise LigandResolutionError("context ligand graph has a self-bond")
        left, right = sorted((row_to_local[row_i], row_to_local[row_j]))
        order = _bond_order(raw_order)
        key = (left, right)
        previous = edges.get(key)
        if previous is not None and previous != order:
            raise LigandResolutionError("context ligand graph has conflicting bond orders")
        edges[key] = order
    edge_values = tuple(
        (left, right, order)
        for (left, right), order in sorted(edges.items())
    )
    adjacency = [dict() for _ in heavy]
    for left, right, order in edge_values:
        adjacency[left][right] = order
        adjacency[right][left] = order
    if len(heavy) > 1:
        seen = {0}
        pending = [0]
        while pending:
            current = pending.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        if len(seen) != len(heavy):
            raise LigandResolutionError("context ligand component graph is disconnected")
    return _Graph(
        names,
        elements,
        edge_values,
        tuple(adjacency),
        _atom_facts(atoms, heavy),
        _external_attachment_facts(atoms, rows, heavy),
    )


def _graph_identity(graph: _Graph, component: str) -> dict[str, Any]:
    return {
        "component": str(component).upper(),
        "atoms": [
            {"name": name, "element": element}
            for name, element in sorted(zip(graph.names, graph.elements))
        ],
        "bonds": [
            {"atom1": graph.names[left], "atom2": graph.names[right], "order": order}
            for left, right, order in graph.edges
        ],
    }


def _rdkit_graph(molecule: Any, *, preserve_kekule: bool = False) -> _Graph:
    from rdkit import Chem
    molecule = Chem.RemoveHs(Chem.Mol(molecule), sanitize=not preserve_kekule)
    names = tuple(
        atom.GetSymbol().upper() for atom in molecule.GetAtoms()
    )
    atom_facts = []
    for atom in molecule.GetAtoms():
        facts: list[tuple[str, Any]] = []
        charge = int(atom.GetFormalCharge())
        # Neutral charge is authoritative for an explicit SMILES too.  Keep
        # zero so a target-side explicit non-zero charge cannot be accepted as
        # an unspecified fact.
        facts.append(("charge", charge))
        chiral = str(atom.GetChiralTag())
        if chiral != "CHI_UNSPECIFIED":
            facts.append(("stereo", chiral.lower()))
        isotope = int(atom.GetIsotope())
        if isotope:
            facts.append(("isotope", isotope))
        atom_facts.append(tuple(sorted(facts)))
    edges: list[tuple[int, int, str]] = []
    adjacency = [dict() for _ in names]
    for bond in molecule.GetBonds():
        bond_type = bond.GetBondType()
        if bond.GetIsAromatic() or bond_type == Chem.BondType.AROMATIC:
            order = "aromatic"
        elif bond_type == Chem.BondType.SINGLE:
            order = "single"
        elif bond_type == Chem.BondType.DOUBLE:
            order = "double"
        elif bond_type == Chem.BondType.TRIPLE:
            order = "triple"
        else:
            raise LigandResolutionError(
                f"SMILES contains unsupported bond type {bond_type}"
            )
        left, right = sorted((int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())))
        edges.append((left, right, order))
        adjacency[left][right] = order
        adjacency[right][left] = order
    return _Graph(
        names=names,
        elements=names,
        edges=tuple(sorted(edges)),
        adjacency=tuple(adjacency),
        atom_facts=tuple(atom_facts),
        external=(),
    )


def _graph_mapping(
    observed: _Graph, expected: _Graph, *, kekule: _Graph | None = None,
) -> dict[int, int]:
    """Return one deterministic target-to-authority graph bijection.

    Multiple solutions are graph automorphisms, not chemistry mismatches.  A
    stable candidate order chooses one while preserving every available atom,
    endpoint-order, charge/stereo and external-attachment constraint.
    """

    if len(observed.names) != len(expected.names):
        raise LigandResolutionError(
            f"ligand graph atom count differs (target {len(observed.names)}, "
            f"authority {len(expected.names)})"
        )
    if sorted(observed.elements) != sorted(expected.elements):
        raise LigandResolutionError("ligand graph element inventory differs")
    if len(observed.edges) != len(expected.edges):
        raise LigandResolutionError("ligand graph bond count differs")

    def facts_compatible(left: tuple[tuple[str, Any], ...], right: tuple[tuple[str, Any], ...]) -> bool:
        left_values, right_values = dict(left), dict(right)
        return all(
            left_values[field] == right_values[field]
            for field in left_values.keys() & right_values.keys()
        )

    def external_compatible(
        left_graph: _Graph,
        left_index: int,
        right_graph: _Graph,
        right_index: int,
    ) -> bool:
        if not left_graph.external or not right_graph.external:
            return True
        return left_graph.external[left_index] == right_graph.external[right_index]

    def order_compatible(left: str, right: str, first: int, second: int) -> bool:
        if left == right:
            return True
        return (
            kekule is not None and right == "aromatic"
            and left == kekule.adjacency[first].get(second)
        )

    def valid_pair(observed_index: int, expected_index: int) -> bool:
        return (
            observed.elements[observed_index] == expected.elements[expected_index]
            and len(observed.adjacency[observed_index]) == len(expected.adjacency[expected_index])
            and (kekule is not None or sorted(observed.adjacency[observed_index].values())
                == sorted(expected.adjacency[expected_index].values()))
            and facts_compatible(
                observed.atom_facts[observed_index] if observed.atom_facts else (),
                expected.atom_facts[expected_index] if expected.atom_facts else (),
            )
            and external_compatible(
                observed,
                observed_index,
                expected,
                expected_index,
            )
        )

    def valid_mapping(mapping: Mapping[int, int]) -> bool:
        if set(mapping) != set(range(len(observed.names))):
            return False
        if set(mapping.values()) != set(range(len(expected.names))):
            return False
        for observed_index, expected_index in mapping.items():
            if not valid_pair(observed_index, expected_index):
                return False
        for observed_left, observed_right, observed_order in observed.edges:
            expected_left = mapping[observed_left]
            expected_right = mapping[observed_right]
            expected_order = expected.adjacency[expected_left].get(expected_right)
            if expected_order is None or not order_compatible(observed_order, expected_order, expected_left, expected_right):
                return False
        return True

    # Stable source names are authoritative when the complete name inventory
    # survives a round-trip.  This also keeps source-to-native AtomUID names
    # aligned for CCD-backed contexts.
    if set(observed.names) == set(expected.names) and len(set(observed.names)) == len(observed.names):
        expected_by_name = {name: i for i, name in enumerate(expected.names)}
        named = {i: expected_by_name[name] for i, name in enumerate(observed.names)}
        if valid_mapping(named):
            return named

    candidates: dict[int, list[int]] = {}
    for observed_index in range(len(observed.names)):
        candidates[observed_index] = [
            expected_index
            for expected_index in range(len(expected.names))
            if valid_pair(observed_index, expected_index)
        ]
        if not candidates[observed_index]:
            raise LigandResolutionError("ligand graph has no compatible atom mapping")
    order = sorted(
        range(len(observed.names)),
        key=lambda index: (len(candidates[index]), -len(observed.adjacency[index]), index),
    )
    mapping: dict[int, int] = {}
    used: set[int] = set()

    def visit(depth: int) -> bool:
        if depth == len(order):
            return True
        observed_index = order[depth]
        for expected_index in sorted(candidates[observed_index], key=lambda value: (expected.names[value], value)):
            if expected_index in used:
                continue
            valid = True
            for previous_observed, previous_expected in mapping.items():
                observed_order = observed.adjacency[observed_index].get(previous_observed)
                expected_order = expected.adjacency[expected_index].get(previous_expected)
                if observed_order is None and expected_order is None:
                    continue
                if observed_order is None or expected_order is None or not order_compatible(observed_order, expected_order, expected_index, previous_expected):
                    valid = False
                    break
            if not valid:
                continue
            mapping[observed_index] = expected_index
            used.add(expected_index)
            if visit(depth + 1):
                return True
            used.remove(expected_index)
            mapping.pop(observed_index)
        return False

    if not visit(0) or not valid_mapping(mapping):
        raise LigandResolutionError("target ligand graph does not match authoritative chemistry")
    return dict(mapping)


def _canonical_smiles_names(molecule: Any) -> tuple[str, ...]:
    try:
        from src.design.rewards.esmfold2_input import _canonical_smiles_atom_names
        return tuple(_canonical_smiles_atom_names(molecule))
    except Exception as exc:
        raise LigandResolutionError(
            f"cannot derive native ESMFold2 SMILES atom names: {exc}"
        ) from exc


def _normalize_ccd(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    result = []
    for item in values:
        text = str(item).strip().upper()
        if not text or text in {".", "?"}:
            raise LigandResolutionError("CCD context ligand has an empty component ID")
        result.append(text)
    return result


def _explicit_chemistry(
    task_config: Mapping[str, Any],
    chain: str,
) -> tuple[str, Any] | None:
    spec = _condition_for_chain(task_config, chain)
    if spec is None:
        return None
    mode = str(spec.get("mode", "")).strip().lower()
    if mode == "smiles":
        smiles = spec.get("smiles")
        if not isinstance(smiles, str) or not smiles.strip():
            raise LigandResolutionError(
                f"context ligand {chain!r} has mode=smiles but no SMILES"
            )
        from rdkit import Chem
        molecule = Chem.MolFromSmiles(smiles.strip())
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise LigandResolutionError(f"context ligand {chain!r} has invalid SMILES")
        molecule = Chem.RemoveHs(molecule)
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise LigandResolutionError(
                f"context ligand {chain!r} SMILES must be one connected component"
            )
        return "smiles", (smiles.strip(), molecule)
    if mode in {"ccd", "component", "components"}:
        raw = spec.get("ccd", spec.get("components", spec.get("component")))
        return "ccd", _normalize_ccd(raw)
    # type_bond/atom_motif and unknown condition modes do not define complete
    # ligand chemistry; the source CCD is the only safe fallback.
    return None


def _source_components(source: Any, chain: str) -> list[tuple[str, np.ndarray, _Graph]]:
    rows = _rows_for_chain(source, chain)
    result = []
    for component_rows in _component_rows(source, rows):
        names = {
            str(value).strip().upper()
            for value in np.asarray(source.res_name).astype(str)[component_rows]
        }
        if len(names) != 1:
            raise LigandResolutionError(
                f"source ligand chain {chain!r} has a component with mixed IDs"
            )
        component = next(iter(names))
        if component in _PLACEHOLDER_COMPONENTS:
            raise LigandResolutionError(
                f"source ligand chain {chain!r} has no usable CCD identity"
            )
        result.append((component, component_rows, _graph_from_atoms(source, component_rows)))
    if not result:
        raise LigandResolutionError(f"source ligand chain {chain!r} has no components")
    return result


def _component_locator(atoms: Any, rows: Sequence[int]) -> tuple[Any, ...]:
    """Return non-coordinate source locators for one ligand occurrence."""

    rows = np.asarray(rows, dtype=np.int64)
    chains = _canonical_chains(atoms)

    def unique_annotation(name: str, default: Any = None) -> Any:
        if not hasattr(atoms, name):
            return default
        values = np.asarray(getattr(atoms, name))[rows]
        unique = list(dict.fromkeys(values.tolist()))
        return unique[0] if len(unique) == 1 else default

    return (
        str(chains[int(rows[0])]) if len(rows) else "",
        unique_annotation("res_id"),
        unique_annotation("ins_code", ""),
        unique_annotation("auth_seq_id"),
        unique_annotation("label_seq_id"),
        unique_annotation("residue_uid"),
        unique_annotation("molecule_uid"),
    )


def _locator_score(source: tuple[Any, ...], target: tuple[Any, ...]) -> int:
    score = 0
    # Chain IDs may be rewritten by a backend, so they are only a weak hint.
    if source[0] and target[0] and source[0] == target[0]:
        score += 1
    for index, weight in ((1, 8), (2, 4), (3, 6), (4, 5), (5, 2), (6, 1)):
        left, right = source[index], target[index]
        if left is not None and right is not None and str(left) not in {"", ".", "?"} and left == right:
            score += weight
    return score


def _component_assignment(
    atoms: Any,
    target_groups: Sequence[np.ndarray],
    source: Any,
    expected: Sequence[tuple[str, np.ndarray, _Graph]],
) -> dict[int, tuple[int, dict[int, int]]]:
    """Assign source occurrences to target groups using graph and locators.

    Component IDs are a hint because the writer may replace them with
    file-local ``LIGxx`` aliases.  Every accepted pair still needs a complete
    element/bond/authoritative-fact graph proof.  Residue/molecule locators
    disambiguate repeated instances such as the six EDO components in 8TFD.
    """

    if len(target_groups) != len(expected):
        raise LigandResolutionError("context ligand component count differs from authoritative source")
    target_records = []
    for group in target_groups:
        values = {
            str(value).strip().upper()
            for value in np.asarray(atoms.res_name).astype(str)[group]
        }
        if len(values) != 1:
            raise LigandResolutionError("context ligand has a component with mixed IDs")
        target_records.append((next(iter(values)), _component_locator(atoms, group), _graph_from_atoms(atoms, group)))
    source_records = [
        (component, _component_locator(source, rows), graph)
        for component, rows, graph in expected
    ]
    candidates: dict[int, list[tuple[int, dict[int, int], tuple[int, int, int]]] ] = {}
    for source_index, (source_id, source_locator, source_graph) in enumerate(source_records):
        choices = []
        for target_index, (target_id, target_locator, target_graph) in enumerate(target_records):
            try:
                mapping = _graph_mapping(target_graph, source_graph)
            except LigandResolutionError:
                continue
            score = (
                int(target_id == source_id),
                _locator_score(source_locator, target_locator),
                -target_index,
            )
            choices.append((target_index, mapping, score))
        if not choices:
            raise LigandResolutionError(
                f"context ligand source component {source_id!r} has no verified target graph"
            )
        candidates[source_index] = choices

    order = sorted(
        candidates,
        key=lambda index: (
            len(candidates[index]),
            -max(choice[2][0] * 100 + choice[2][1] for choice in candidates[index]),
            index,
        ),
    )
    assignment: dict[int, tuple[int, dict[int, int]]] = {}
    used: set[int] = set()
    source_ids = {source_id for source_id, _, _ in source_records}

    # Repeated occurrences of one CCD (for example EDO) are chemically
    # interchangeable once each occurrence has passed graph verification.
    # Keep this path bounded for large repeated-component contexts while still
    # using the stable locator-first choice order.
    if len(source_ids) == 1:
        def find_first(depth: int) -> bool:
            if depth == len(order):
                return True
            source_index = order[depth]
            choices = sorted(
                candidates[source_index],
                key=lambda choice: (choice[2][0], choice[2][1], choice[2][2]),
                reverse=True,
            )
            for target_index, mapping, _score in choices:
                if target_index in used:
                    continue
                used.add(target_index)
                assignment[source_index] = (target_index, mapping)
                if find_first(depth + 1):
                    return True
                assignment.pop(source_index)
                used.remove(target_index)
            return False

        if not find_first(0):
            raise LigandResolutionError(
                "context ligand source occurrences cannot be assigned uniquely"
            )
        return dict(assignment)

    best_assignment: dict[int, tuple[int, dict[int, int]]] | None = None
    best_score: tuple[int, int] | None = None
    tied_best = False

    def visit(depth: int, id_score: int, locator_score: int) -> None:
        nonlocal best_assignment, best_score, tied_best
        if depth == len(order):
            score = (id_score, locator_score)
            if best_score is None or score > best_score:
                best_score = score
                best_assignment = dict(assignment)
                tied_best = False
            elif score == best_score:
                tied_best = True
            return
        source_index = order[depth]
        choices = sorted(
            candidates[source_index],
            key=lambda choice: (choice[2][0], choice[2][1], choice[2][2]),
            reverse=True,
        )
        for target_index, mapping, score in choices:
            if target_index in used:
                continue
            used.add(target_index)
            assignment[source_index] = (target_index, mapping)
            visit(depth + 1, id_score + score[0], locator_score + score[1])
            assignment.pop(source_index)
            used.remove(target_index)

    visit(0, 0, 0)
    if best_assignment is None:
        raise LigandResolutionError("context ligand source occurrences cannot be assigned uniquely")
    if tied_best and len(source_ids) > 1:
        raise LigandResolutionError(
            "context ligand source occurrences have ambiguous locator matches"
        )
    return best_assignment


def _source_chain(source: Any, target_chain: str, target_atoms: Any | None = None) -> str:
    chains = _canonical_chains(source)
    modalities = np.asarray(source.mol_type).astype(str)
    ligand_chains = [
        chain for chain in dict.fromkeys(chains.tolist())
        if bool(np.any((chains == chain) & (modalities == "ligand")))
    ]
    if target_chain in ligand_chains:
        return target_chain
    if len(ligand_chains) == 1:
        return str(ligand_chains[0])
    if target_atoms is not None:
        target_groups = _component_rows(
            target_atoms,
            _rows_for_chain(target_atoms, target_chain),
        )
        candidates: list[str] = []
        for source_chain in ligand_chains:
            try:
                _component_assignment(
                    target_atoms,
                    target_groups,
                    source,
                    _source_components(source, str(source_chain)),
                )
            except LigandResolutionError:
                continue
            candidates.append(str(source_chain))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise LigandResolutionError(
                f"authoritative source has ambiguous ligand-chain graph matches for {target_chain!r}"
            )
    if not ligand_chains:
        raise LigandResolutionError(
            f"authoritative source has no ligand chain for context {target_chain!r}"
        )
    raise LigandResolutionError(
        f"authoritative source has multiple ligand chains; cannot map {target_chain!r}"
    )


def _load_source(source_path: Path) -> Any:
    if not source_path.is_file():
        raise LigandResolutionError(
            f"authoritative source structure is missing: {source_path}"
        )
    try:
        from src.design.utils.reader import read_structure
        return read_structure(source_path)
    except Exception as exc:
        raise LigandResolutionError(
            f"cannot read authoritative source structure {source_path}: {exc}"
        ) from exc


def _resolve_smiles(
    atoms: Any,
    chain: str,
    smiles: str,
    molecule: Any,
) -> tuple[dict[str, Any], dict[int, str]]:
    target_rows = _rows_for_chain(atoms, chain)
    target_groups = _component_rows(atoms, target_rows)
    if len(target_groups) != 1:
        raise LigandResolutionError(
            f"context ligand {chain!r} SMILES requires one target component"
        )
    target = _graph_from_atoms(atoms, target_groups[0])
    authority = _rdkit_graph(molecule)
    try:
        mapping = _graph_mapping(target, authority)
    except LigandResolutionError:
        # An authoritative aromatic molecule also supplies its concrete
        # Kekule orders. Accept that representation without loosening the
        # ordinary source-graph matcher or inventing any missing bond.
        if not any(order == "aromatic" for _, _, order in authority.edges):
            raise
        from rdkit import Chem
        concrete = Chem.Mol(molecule)
        try:
            Chem.Kekulize(concrete, clearAromaticFlags=True)
        except (ValueError, RuntimeError) as exc:
            raise LigandResolutionError("trusted SMILES cannot supply Kekule bond orders") from exc
        mapping = _graph_mapping(target, authority, kekule=_rdkit_graph(concrete, preserve_kekule=True))
    names = _canonical_smiles_names(molecule)
    uids = np.asarray(getattr(atoms, "atom_uid", np.arange(len(atoms))), dtype=np.int64)
    target_heavy = [
        int(row) for row in target_groups[0].tolist()
        if str(atoms.element[row]).strip().upper() not in {"H", "D", "T"}
    ]
    return {"smiles": smiles}, {
        int(uids[row]): names[int(mapping[index])]
        for index, row in enumerate(target_heavy)
    }


def _resolve_ccd_or_source(
    atoms: Any,
    chain: str,
    source: Any,
    source_chain: str,
    explicit_ccd: list[str] | None,
) -> tuple[dict[str, Any], dict[int, str]]:
    target_groups = _component_rows(atoms, _rows_for_chain(atoms, chain))
    expected = _source_components(source, source_chain)
    if explicit_ccd is not None:
        expected_ids = [component.upper() for component, _, _ in expected]
        if sorted(expected_ids) != sorted(explicit_ccd):
            raise LigandResolutionError(
                f"context ligand {chain!r} CCD definition {explicit_ccd!r} "
                f"does not match source CCD components "
                f"{expected_ids!r}"
            )
    if len(target_groups) != len(expected):
        raise LigandResolutionError(
            f"context ligand {chain!r} component count differs from authoritative source"
        )
    assignments = _component_assignment(atoms, target_groups, source, expected)
    target_names = []
    for source_index, (target_index, mapping) in assignments.items():
        component, source_rows, source_graph = expected[source_index]
        target_graph = _graph_from_atoms(atoms, target_groups[target_index])
        target_heavy = [
            int(row) for row in target_groups[target_index].tolist()
            if str(atoms.element[row]).strip().upper() not in {"H", "D", "T"}
        ]
        uids = np.asarray(getattr(atoms, "atom_uid", np.arange(len(atoms))), dtype=np.int64)
        for local, row in enumerate(target_heavy):
            target_names.append((int(uids[row]), source_graph.names[mapping[local]]))
    target_names_map = dict(target_names)
    return {"ccd": list(explicit_ccd or [component for component, _, _ in expected])}, target_names_map


def _context_ligand_chains(atoms: Any, design_chains: set[str]) -> list[str]:
    chains = _canonical_chains(atoms)
    modalities = np.asarray(atoms.mol_type).astype(str)
    return [
        str(chain) for chain in dict.fromkeys(chains.tolist())
        if str(chain) not in design_chains
        and bool(np.any((chains == chain) & (modalities == "ligand")))
    ]


def _component_uid(atoms: Any, rows: Sequence[int]) -> int:
    """Return the reader-stable component UID for one target occurrence."""

    values = getattr(atoms, "component_uid", None)
    if values is None:
        raise LigandResolutionError(
            "component-level ligand resolution requires reader component_uid"
        )
    rows = np.asarray(rows, dtype=np.int64)
    unique = list(dict.fromkeys(np.asarray(values)[rows].tolist()))
    if len(unique) != 1:
        raise LigandResolutionError(
            "target ligand component has an ambiguous component_uid"
        )
    try:
        return int(unique[0])
    except (TypeError, ValueError) as exc:
        raise LigandResolutionError(
            f"target ligand component_uid {unique[0]!r} is not an integer"
        ) from exc


def resolve_context_ligand_components(
    task: Path,
    atoms: Any,
    *,
    design_chains: Sequence[str] = (),
    chains: Sequence[str] | None = None,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[int, str]]:
    """Resolve each context ligand component independently.

    The key ``(logical_chain_id, component_uid)`` uses the target reader's
    stable component UID, while chemistry and native atom names come from a
    verified SMILES or source graph.  Source component order is never used as
    the target order: aliases, repeated components and rewritten residue
    labels are assigned by graph proof plus source locators.

    This is an internal component-level view for mixed private payloads.  The
    legacy :func:`resolve_context_ligands` aggregate tuple remains available
    for pure-chain callers.
    """

    _, task_config, source_path = _task_config(task)
    target_chains = list(chains) if chains is not None else _context_ligand_chains(
        atoms, set(str(value) for value in design_chains)
    )
    target_chains = [str(value) for value in target_chains]
    if not target_chains:
        return {}, {}

    source: Any | None = None
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    atom_names: dict[int, str] = {}
    for chain in target_chains:
        target_groups = _component_rows(atoms, _rows_for_chain(atoms, chain))
        explicit = _explicit_chemistry(task_config, chain)
        if explicit is not None and explicit[0] == "smiles":
            if len(target_groups) != 1:
                raise LigandResolutionError(
                    f"context ligand {chain!r} SMILES requires one target component"
                )
            entry, mapping = _resolve_smiles(
                atoms, chain, explicit[1][0], explicit[1][1]
            )
            entries[(chain, _component_uid(atoms, target_groups[0]))] = entry
            atom_names.update(mapping)
            continue

        if source is None:
            source = _load_source(source_path)
        source_chain = _source_chain(source, chain, atoms)
        expected = _source_components(source, source_chain)
        if explicit is not None:
            explicit_ids = list(explicit[1])
            expected_ids = [component.upper() for component, _, _ in expected]
            if sorted(expected_ids) != sorted(explicit_ids):
                raise LigandResolutionError(
                    f"context ligand {chain!r} CCD definition {explicit_ids!r} "
                    f"does not match source CCD components {expected_ids!r}"
                )
        assignments = _component_assignment(atoms, target_groups, source, expected)
        for source_index, (target_index, mapping) in sorted(
            assignments.items(), key=lambda item: item[1][0]
        ):
            component, _source_rows, source_graph = expected[source_index]
            target_group = target_groups[target_index]
            uid = _component_uid(atoms, target_group)
            key = (chain, uid)
            if key in entries:
                raise LigandResolutionError(
                    f"context ligand component key {key!r} is duplicated"
                )
            entries[key] = {"ccd": [component]}
            target_heavy = [
                int(row) for row in target_group.tolist()
                if str(atoms.element[row]).strip().upper() not in {"H", "D", "T"}
            ]
            uids = np.asarray(
                getattr(atoms, "atom_uid", np.arange(len(atoms))), dtype=np.int64
            )
            for local, row in enumerate(target_heavy):
                atom_uid = int(uids[row])
                native_name = source_graph.names[mapping[local]]
                previous = atom_names.get(atom_uid)
                if previous is not None and previous != native_name:
                    raise LigandResolutionError(
                        f"target atom_uid {atom_uid} has conflicting native names"
                    )
                atom_names[atom_uid] = native_name
    return entries, atom_names


def resolve_context_ligands(
    task: Path,
    atoms: Any,
    *,
    design_chains: Sequence[str] = (),
    chains: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    """Resolve every context ligand to native ESMFold2 chemistry and atom names.

    The second return value maps target atom_uid to the output atom name
    expected by a SMILES/CCD/native ESMFold2 ledger.
    """

    _, task_config, source_path = _task_config(task)
    target_chains = list(chains) if chains is not None else _context_ligand_chains(
        atoms, set(str(value) for value in design_chains)
    )
    target_chains = [str(value) for value in target_chains]
    if not target_chains:
        return {}, {}
    source: Any | None = None
    entries: dict[str, dict[str, Any]] = {}
    atom_names: dict[int, str] = {}
    for chain in target_chains:
        explicit = _explicit_chemistry(task_config, chain)
        if explicit is not None and explicit[0] == "smiles":
            entry, mapping = _resolve_smiles(atoms, chain, explicit[1][0], explicit[1][1])
        else:
            if source is None:
                source = _load_source(source_path)
            source_chain = _source_chain(source, chain, atoms)
            if explicit is not None:
                entry, mapping = _resolve_ccd_or_source(
                    atoms, chain, source, source_chain, list(explicit[1])
                )
            else:
                entry, mapping = _resolve_ccd_or_source(
                    atoms, chain, source, source_chain, None
                )
        entries[chain] = entry
        atom_names.update(mapping)
    return entries, atom_names


def ligand_request_identity(task: Path) -> dict[str, Any]:
    """Return coordinate-free chemistry identity for one resolved task.

    Source CIF coordinates and motif/reference coordinates are intentionally
    absent. A source-backed context component contributes only its component
    ID, atom inventory, and complete internal bond graph.
    """

    _, task_config, source_path = _task_config(task, allow_missing=True)
    if not task_config:
        return {"schema_version": 1, "context_ligands": {}}
    design = _design_chains(task_config)
    source: Any | None = None
    # The source is needed to discover context ligand chains when the task
    # condition does not explicitly name their chemistry.
    if source_path is not None:
        try:
            source = _load_source(source_path)
        except LigandResolutionError:
            if not any(
                chain not in design
                for chain in _ligand_conditions(task_config)
            ):
                source = None
            else:
                raise
    result: dict[str, Any] = {"schema_version": 1, "context_ligands": {}}
    chains: list[str] = []
    if source is not None:
        source_chains = _canonical_chains(source)
        modalities = np.asarray(source.mol_type).astype(str)
        chains.extend(
            str(chain) for chain in dict.fromkeys(source_chains.tolist())
            if str(chain) not in design
            and bool(np.any((source_chains == chain) & (modalities == "ligand")))
        )
    chains.extend(
        str(chain) for chain in _ligand_conditions(task_config)
        if str(chain) not in design
    )
    for chain in dict.fromkeys(chains):
        explicit = _explicit_chemistry(task_config, chain)
        if explicit is not None and explicit[0] == "smiles":
            _, molecule = explicit[1]
            from rdkit import Chem
            result["context_ligands"][chain] = {
                "mode": "smiles",
                "smiles": Chem.MolToSmiles(
                    molecule, canonical=True, isomericSmiles=True
                ),
            }
            continue
        if explicit is not None and explicit[0] == "ccd":
            result["context_ligands"][chain] = {
                "mode": "ccd", "ccd": list(explicit[1]),
            }
            continue
        if source is None:
            raise LigandResolutionError(
                f"context ligand {chain!r} has no authoritative chemistry"
            )
        source_chain = _source_chain(source, chain)
        result["context_ligands"][chain] = {
            "mode": "source_ccd",
            "components": [
                _graph_identity(graph, component)
                for component, _, graph in _source_components(source, source_chain)
            ],
        }
    return result


__all__ = [
    "LigandResolutionError",
    "ligand_request_identity",
    "resolve_context_ligand_components",
    "resolve_context_ligands",
]
