"""Private native Amber support for cobalt(III) hexammine components.

The public structure contains only the seven heavy atoms of a CoN6
component.  This module expands a validated component to a private CO6
residue, using the AmberTools CO6.off/frcmod.CO6 files shipped with the
active environment.  The public residue name, atom names, IDs, topology and
coordinates are retained; only private ammonia hydrogens are appended.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import itertools
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from ..runner import AtomKey, PostprocessPreparationError


_CO6_N_ROLES = frozenset({"N1", "N2", "N3", "N4", "N5", "N6"})
_CO6_TYPE_PREFIX = "PDCO6_"
_CO6_PARAMETER_PREFIX = "PD_CO6_"
_CO6_BOND_LENGTH_NM = 0.20702
_CO6_H_BOND_LENGTH_NM = 0.1037
_CO6_DISTANCE_TOLERANCE_NM = 0.050
_CO6_TRANS_DOT_MAX = -0.82
_CO6_CIS_DOT_MAX = 0.45
_CO6_MIN_PRIVATE_DISTANCE_NM = 1.0e-5


def _error(message: str) -> PostprocessPreparationError:
    return PostprocessPreparationError(message)


def _system_error(message: str, cause: BaseException | None = None) -> RuntimeError:
    from .refine import OpenMMSystemError

    error = OpenMMSystemError(message)
    if cause is not None:
        error.__cause__ = cause
    return error


def _element_symbol(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise _error("cobalt hexammine atom has an empty element")
    return text[:1].upper() + text[1:].lower()


def _atom_key_from_openmm(atom: Any) -> AtomKey:
    residue = atom.residue
    try:
        residue_id = int(residue.id)
    except (TypeError, ValueError) as exc:
        raise _error("cobalt topology residue IDs must be integer-like") from exc
    return (
        str(residue.chain.id),
        residue_id,
        str(residue.insertionCode),
        str(residue.name),
        str(atom.name),
    )


def _atom_key_from_array(atoms: Any, row: int) -> AtomKey:
    try:
        return (
            str(atoms.chain_id[row]),
            int(atoms.res_id[row]),
            str(atoms.ins_code[row]),
            str(atoms.res_name[row]),
            str(atoms.atom_name[row]),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise _error("cobalt heavy-group rows do not expose AtomKey fields") from exc


def _normalize_rows(rows: Sequence[int], count: int) -> tuple[int, ...]:
    if isinstance(rows, (str, bytes)):
        raise _error("cobalt heavy group rows must be a sequence of indices")
    try:
        raw_rows = tuple(rows)
    except TypeError as exc:
        raise _error("cobalt heavy group rows must be a sequence of indices") from exc
    if len(raw_rows) != 7:
        raise _error("cobalt hexammine requires exactly seven heavy atoms")
    result: list[int] = []
    for raw in raw_rows:
        if isinstance(raw, (bool, np.bool_)):
            raise _error("cobalt heavy group rows must be integer indices")
        try:
            row = int(raw)
        except (TypeError, ValueError) as exc:
            raise _error("cobalt heavy group rows must be integer indices") from exc
        if row != raw or row < 0 or row >= count:
            raise _error("cobalt heavy group row is outside the accepted structure")
        result.append(row)
    if len(set(result)) != len(result):
        raise _error("cobalt heavy group contains a duplicate atom")
    return tuple(result)


def _single_bond_order(raw: object) -> bool:
    try:
        return raw is not None and abs(float(raw) - 1.0) <= 1.0e-7
    except (TypeError, ValueError):
        return False


def _all_pairings(values: tuple[int, ...]) -> Iterable[tuple[tuple[int, int], ...]]:
    if not values:
        yield ()
        return
    first = values[0]
    for position in range(1, len(values)):
        second = values[position]
        remainder = values[1:position] + values[position + 1:]
        for tail in _all_pairings(remainder):
            yield ((first, second),) + tail


def _octahedral_pairs(
    vectors: np.ndarray,
    *,
    distance_nm: np.ndarray | None = None,
) -> tuple[tuple[int, int], ...]:
    """Return a unique trans-pairing for six normalized Co--N vectors."""

    if vectors.shape != (6, 3) or not np.isfinite(vectors).all():
        raise _error("cobalt hexammine N vectors are non-finite or incomplete")
    lengths = np.linalg.norm(vectors, axis=1) if distance_nm is None else distance_nm
    if np.any(lengths < 1.0e-8):
        raise _error("cobalt hexammine Co--N geometry is degenerate")
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    dot = np.clip(unit_vectors @ unit_vectors.T, -1.0, 1.0)
    candidates: list[tuple[float, tuple[tuple[int, int], ...]]] = []
    for pairing in _all_pairings(tuple(range(6))):
        score = sum(float((dot[i, j] + 1.0) ** 2) for i, j in pairing)
        candidates.append((score, pairing))
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_score, best = candidates[0]
    if best_score > 3.0 * (1.0 + _CO6_TRANS_DOT_MAX) ** 2:
        raise _error("cobalt hexammine N geometry has no octahedral trans pairing")
    if len(candidates) > 1 and candidates[1][0] - best_score < 1.0e-4:
        raise _error("cobalt hexammine trans-pair assignment is ambiguous")
    if any(float(dot[i, j]) > _CO6_TRANS_DOT_MAX for i, j in best):
        raise _error("cobalt hexammine trans angle is not octahedral")
    for pair_i, pair_j in itertools.combinations(best, 2):
        for i in pair_i:
            for j in pair_j:
                if abs(float(dot[i, j])) > _CO6_CIS_DOT_MAX:
                    raise _error("cobalt hexammine cis angles are not octahedral")
    return tuple(sorted(tuple(sorted(pair)) for pair in best))


def _validate_geometry(
    coords_nm: np.ndarray,
    co_index: int,
    n_indices: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    if coords_nm.shape != (7, 3) or not np.isfinite(coords_nm).all():
        raise _error("cobalt hexammine coordinates must be finite seven-atom XYZ")
    co = coords_nm[co_index]
    vectors = np.asarray([coords_nm[index] - co for index in n_indices], dtype=float)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(
        np.abs(lengths - _CO6_BOND_LENGTH_NM) > _CO6_DISTANCE_TOLERANCE_NM
    ):
        raise _error("cobalt hexammine Co--N distances are outside CO6 geometry")
    return _octahedral_pairs(vectors, distance_nm=lengths)


def _validate_array_graph(atoms: Any, rows: tuple[int, ...]) -> tuple[AtomKey, ...]:
    if getattr(atoms, "bonds", None) is None:
        raise _error("cobalt heavy group lacks an authoritative BondList")
    keys = tuple(_atom_key_from_array(atoms, row) for row in rows)
    if len(set(keys)) != len(keys):
        raise _error("cobalt heavy-group AtomKeys are not unique")
    if len({key[:4] for key in keys}) != 1:
        raise _error("cobalt hexammine heavy atoms must share one residue")
    elements = [_element_symbol(atoms.element[row]) for row in rows]
    co_local = [index for index, value in enumerate(elements) if value == "Co"]
    n_local = [index for index, value in enumerate(elements) if value == "N"]
    if len(co_local) != 1 or len(n_local) != 6 or any(
        value not in {"Co", "N"} for value in elements
    ):
        raise _error("cobalt hexammine requires one Co and six N heavy atoms")
    selected = set(rows)
    internal: list[tuple[int, int]] = []
    for raw_i, raw_j, raw_order in atoms.bonds.as_array():
        i, j = int(raw_i), int(raw_j)
        if i == j:
            raise _error("cobalt heavy graph contains a self-bond")
        if (i in selected) != (j in selected):
            raise _error("cobalt hexammine has an external heavy-atom edge")
        if i not in selected:
            continue
        if not _single_bond_order(raw_order):
            raise _error("cobalt hexammine heavy edges must be SINGLE")
        internal.append((i, j))
    if len(internal) != 6:
        raise _error("cobalt hexammine requires exactly six internal heavy edges")
    co_row = rows[co_local[0]]
    n_rows = {rows[index] for index in n_local}
    degree = {row: 0 for row in rows}
    for raw_i, raw_j in internal:
        degree[raw_i] += 1
        degree[raw_j] += 1
        if co_row not in (raw_i, raw_j):
            raise _error("cobalt hexammine internal graph is not a Co--N star")
        other = raw_j if raw_i == co_row else raw_i
        if other not in n_rows:
            raise _error("cobalt hexammine internal graph is not a Co--N star")
    if degree[co_row] != 6 or any(degree[row] != 1 for row in n_rows):
        raise _error("cobalt hexammine internal graph is not a full CoN6 star")
    coords_angstrom = np.asarray(atoms.coord[list(rows)], dtype=float)
    _validate_geometry(coords_angstrom * 0.1, co_local[0], tuple(n_local))
    return keys


def validate_cobalt_heavy_group(atoms: Any, rows: Sequence[int]) -> tuple[AtomKey, ...]:
    """Validate one authoritative seven-heavy-atom CoN6 component.

    The returned keys retain the caller's row order.  Every selected atom must
    belong to one residue, the authoritative graph must be exactly six
    concrete SINGLE Co--N edges with no external edge, and the coordinates
    must admit an unambiguous octahedral trans pairing.
    """

    normalized = _normalize_rows(rows, len(atoms))
    return _validate_array_graph(atoms, normalized)


@dataclass(frozen=True)
class _BaseAtom:
    name: str
    type_name: str
    element_symbol: str
    charge: float


@dataclass(frozen=True)
class _CobaltParameters:
    atoms: Mapping[str, _BaseAtom]
    bonds: tuple[tuple[str, str], ...]
    type_names: Mapping[str, str]
    center_angles: Mapping[tuple[str, str], tuple[float, float]]


class _CobaltMatcher:
    """ForceField template matcher keyed by preserved residue identity."""

    def __init__(self) -> None:
        self.templates: dict[tuple[str, str, str, str], Any] = {}
        self.angle_pairs: dict[
            tuple[str, str, str, str], tuple[tuple[str, str, float, float], ...]
        ] = {}

    def __call__(
        self,
        _forcefield: Any,
        residue: Any,
        _bonded_to_atom: Any,
        _ignore_external_bonds: bool,
        _ignore_extra_particles: bool,
    ) -> Any:
        key = (
            str(residue.chain.id),
            str(residue.id),
            str(residue.insertionCode),
            str(residue.name),
        )
        return self.templates.get(key)


class _CobaltAngleGenerator:
    """Restore N--Co--N angles after OpenMM graph matching.

    OpenMM's residue matcher may permute symmetry-equivalent nitrogen atoms.
    The ParmEd-generated force field therefore omits all N--Co--N center-angle
    entries; this generator adds the exact 15 terms from the geometry-derived
    trans pairing recorded for each prepared residue.
    """

    def __init__(self, matcher: _CobaltMatcher) -> None:
        self.matcher = matcher

    def createForce(self, _system: Any, _data: Any, _nonbonded_method: Any,
                    _nonbonded_cutoff: Any, _args: Any) -> None:
        return None

    def postprocessSystem(self, system: Any, data: Any, _args: Any) -> None:
        import openmm

        forces = [
            force for force in system.getForces()
            if type(force) is openmm.HarmonicAngleForce
        ]
        force = forces[0] if forces else openmm.HarmonicAngleForce()
        if not forces:
            system.addForce(force)
        atoms_by_residue: dict[
            tuple[str, str, str, str], dict[str, Any]
        ] = {}
        for atom in data.atoms:
            key = _residue_match_key(atom.residue)
            atoms_by_residue.setdefault(key, {})[str(atom.name)] = atom
        for residue_key, terms in self.matcher.angle_pairs.items():
            atom_map = atoms_by_residue.get(residue_key)
            if atom_map is None:
                raise _system_error("cobalt angle residue disappeared during OpenMM setup")
            cobalt = next(
                (atom for atom in atom_map.values()
                 if _element_symbol(atom.element.symbol) == "Co"),
                None,
            )
            if cobalt is None:
                raise _system_error("cobalt angle residue has no Co atom")
            for left_name, right_name, angle, k in terms:
                try:
                    left, right = atom_map[left_name], atom_map[right_name]
                except KeyError as exc:
                    raise _system_error(
                        "cobalt N--Co--N angle endpoint disappeared"
                    ) from exc
                force.addAngle(left.index, cobalt.index, right.index, angle, k)


def _residue_match_key(residue: Any) -> tuple[str, str, str, str]:
    return (
        str(residue.chain.id),
        str(residue.id),
        str(residue.insertionCode),
        str(residue.name),
    )


def _parameter_files() -> tuple[Path, Path]:
    root = Path(sys.prefix) / "dat" / "contrib" / "others"
    return root / "CO6.off", root / "frcmod.CO6"


def _load_cobalt_parameters(forcefield: Any) -> _CobaltParameters:
    cached = getattr(forcefield, "_pd_cobalt_parameters", None)
    if cached is not None:
        return cached
    if not hasattr(forcefield, "loadFile") or not hasattr(
        forcefield, "registerResidueTemplate"
    ):
        raise _error("cobalt native route requires an OpenMM ForceField")
    off_path, frcmod_path = _parameter_files()
    if not off_path.is_file() or not frcmod_path.is_file():
        raise _system_error(
            "installed AmberTools CO6.off/frcmod.CO6 assets are unavailable"
        )
    try:
        from parmed.amber.parameters import AmberParameterSet
        from parmed.openmm.parameters import OpenMMParameterSet

        amber = AmberParameterSet(str(off_path), str(frcmod_path))
        parameter_set = OpenMMParameterSet.from_parameterset(amber)
        xml_buffer = StringIO()
        parameter_set.write(xml_buffer)
        xml_root = ET.fromstring(xml_buffer.getvalue())
    except Exception as exc:
        raise _system_error("Amber CO6 parameter conversion failed", exc)

    residue_set = getattr(amber, "residues", {})
    residue = residue_set.get("CO6")
    if residue is None:
        raise _system_error("Amber CO6.off does not define a CO6 residue")
    base_atoms: dict[str, _BaseAtom] = {}
    for atom in residue.atoms:
        try:
            from openmm.app import element

            base_atoms[str(atom.name)] = _BaseAtom(
                str(atom.name),
                str(atom.type),
                _element_symbol(
                        element.Element.getByAtomicNumber(int(atom.atomic_number)).symbol
                ),
                float(atom.charge),
            )
        except Exception as exc:
            raise _system_error(
                "Amber CO6.off contains an unreadable atom record", exc
            )
    bonds: list[tuple[str, str]] = []
    for bond in residue.bonds:
        try:
            bonds.append((str(bond.atom1.name), str(bond.atom2.name)))
        except Exception as exc:
            raise _system_error(
                "Amber CO6.off contains an unreadable bond record", exc
            )
    n_types: dict[str, list[str]] = {}
    for name in _CO6_N_ROLES:
        atom = base_atoms.get(name)
        if atom is None:
            raise _error("Amber CO6.off is missing one of N1..N6")
        n_types.setdefault(atom.type_name, []).append(name)
    if sorted(len(value) for value in n_types.values()) != [2, 2, 2]:
        raise _system_error("Amber CO6.off does not encode three trans N type pairs")
    type_names = {
        source: _CO6_TYPE_PREFIX + source
        for source in {atom.type_name for atom in base_atoms.values()}
    }

    atom_types = xml_root.find("AtomTypes")
    if atom_types is None:
        raise _system_error("ParmEd CO6 conversion omitted AtomTypes")
    center_angles: dict[tuple[str, str], tuple[float, float]] = {}
    angle_force = xml_root.find("HarmonicAngleForce")
    if angle_force is None:
        raise _system_error("ParmEd CO6 conversion omitted angle parameters")
    source_n_types = set(n_types)
    for angle in angle_force.findall("Angle"):
        class1 = angle.attrib.get("class1")
        class2 = angle.attrib.get("class2")
        class3 = angle.attrib.get("class3")
        if (
            class2 == "CO"
            and class1 in source_n_types
            and class3 in source_n_types
        ):
            try:
                key = tuple(sorted((class1, class3)))
                value = (
                    float(angle.attrib["angle"]),
                    float(angle.attrib["k"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise _system_error(
                    "ParmEd CO6 conversion emitted invalid center-angle parameters",
                    exc,
                )
            previous = center_angles.get(key)
            if previous is not None and previous != value:
                raise _system_error(
                    "ParmEd CO6 conversion emitted conflicting center-angle parameters"
                )
            center_angles[key] = value
    expected_center_pairs = {
        tuple(sorted(pair))
        for pair in itertools.combinations_with_replacement(
            sorted(source_n_types), 2
        )
    }
    if set(center_angles) != expected_center_pairs:
        raise _system_error("ParmEd CO6 conversion omitted a center-angle parameter")
    for angle in list(angle_force.findall("Angle")):
        class1 = angle.attrib.get("class1")
        class2 = angle.attrib.get("class2")
        class3 = angle.attrib.get("class3")
        if (
            class2 == "CO"
            and class1 in source_n_types
            and class3 in source_n_types
        ):
            angle_force.remove(angle)
    for atom_type in atom_types.findall("Type"):
        source_name = atom_type.attrib.get("name")
        if source_name in type_names:
            atom_type.attrib["name"] = type_names[source_name]
        source_class = atom_type.attrib.get("class")
        if source_class in type_names:
            atom_type.attrib["class"] = type_names[source_class]
    residues = xml_root.find("Residues")
    if residues is not None:
        xml_root.remove(residues)
    for node in xml_root.iter():
        for attr in ("type", "class", "class1", "class2", "class3", "class4"):
            value = node.attrib.get(attr)
            if value in type_names:
                node.attrib[attr] = type_names[value]
    try:
        forcefield.loadFile(StringIO(ET.tostring(xml_root, encoding="unicode")))
    except Exception as exc:
        raise _system_error(
            "Amber CO6 force terms could not be loaded into ForceField", exc
        )
    result = _CobaltParameters(base_atoms, tuple(bonds), type_names, center_angles)
    setattr(forcefield, "_pd_cobalt_parameters", result)
    matcher = getattr(forcefield, "_pd_cobalt_matcher", None)
    if matcher is None:
        matcher = _CobaltMatcher()
        forcefield.registerTemplateMatcher(matcher)
        setattr(forcefield, "_pd_cobalt_matcher", matcher)
    if getattr(forcefield, "_pd_cobalt_angle_generator", None) is None:
        forcefield.registerGenerator(_CobaltAngleGenerator(matcher))
        setattr(forcefield, "_pd_cobalt_angle_generator", True)
    return result


def _topology_group(
    topology: Any,
    group: Sequence[AtomKey],
) -> tuple[tuple[Any, ...], Any, tuple[Any, ...]]:
    by_key: dict[AtomKey, Any] = {}
    for atom in topology.atoms():
        key = _atom_key_from_openmm(atom)
        if key in by_key:
            raise _error("cobalt topology AtomKeys are not unique")
        by_key[key] = atom
    if len(group) != 7 or len(set(group)) != 7:
        raise _error("cobalt hexammine requires seven unique heavy keys")
    try:
        selected = tuple(by_key[key] for key in group)
    except KeyError as exc:
        raise _error(f"cobalt heavy key is absent from topology: {exc.args[0]}") from exc
    residues = {atom.residue for atom in selected}
    if len(residues) != 1:
        raise _error("cobalt hexammine heavy atoms must share one residue")
    residue = next(iter(residues))
    residue_atoms = tuple(residue.atoms())
    if len(residue_atoms) != 7 or set(residue_atoms) != set(selected):
        raise _error("cobalt residue is not an exact seven-heavy-atom component")
    co = tuple(atom for atom in selected if _element_symbol(atom.element.symbol) == "Co")
    nitrogens = tuple(atom for atom in selected if _element_symbol(atom.element.symbol) == "N")
    if len(co) != 1 or len(nitrogens) != 6 or len(co) + len(nitrogens) != 7:
        raise _error("cobalt topology group requires one Co and six N atoms")
    adjacency: dict[Any, list[Any]] = {atom: [] for atom in selected}
    external = False
    internal: list[tuple[Any, Any]] = []
    selected_set = set(selected)
    for bond in topology.bonds():
        left, right = bond
        left_in, right_in = left in selected_set, right in selected_set
        if left_in != right_in:
            external = True
            continue
        if left_in and right_in:
            if not _single_bond_order(bond.order):
                raise _error("cobalt hexammine heavy edges must be SINGLE")
            adjacency[left].append(right)
            adjacency[right].append(left)
            internal.append((left, right))
    if external or len(internal) != 6:
        raise _error("cobalt topology group must be a closed six-edge CoN6 star")
    cobalt = co[0]
    if len(adjacency[cobalt]) != 6 or set(adjacency[cobalt]) != set(nitrogens):
        raise _error("cobalt topology group is not a full CoN6 star")
    if any(adjacency[nitrogen] != [cobalt] for nitrogen in nitrogens):
        raise _error("cobalt topology group contains an unexpected N edge")
    return selected, residue, nitrogens


def _positions_nm(positions: Any, count: int) -> tuple[Any, np.ndarray, np.ndarray]:
    from openmm import unit

    if hasattr(positions, "value_in_unit") and hasattr(positions, "unit"):
        output_unit = positions.unit
        original = np.asarray(positions.value_in_unit(output_unit), dtype=float)
        xyz_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
    else:
        output_unit = unit.nanometer
        original = np.asarray(positions, dtype=float)
        xyz_nm = original.copy()
    if original.shape != (count, 3) or xyz_nm.shape != (count, 3):
        raise _error("cobalt positions must have shape (N, 3)")
    if not np.isfinite(original).all() or not np.isfinite(xyz_nm).all():
        raise _error("cobalt positions must be finite")
    return output_unit, original, xyz_nm


def _private_h_directions(outward: np.ndarray) -> np.ndarray:
    axis = np.asarray(outward, dtype=float)
    axis /= np.linalg.norm(axis)
    canonical = np.eye(3, dtype=float)
    reference = canonical[int(np.argmin(np.abs(canonical @ axis)))]
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    angles = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
    return np.asarray(
        [
            axis / 3.0
            + np.sqrt(8.0) / 3.0
            * (np.cos(angle) * first + np.sin(angle) * second)
            for angle in angles
        ],
        dtype=float,
    )


def _template_for_group(
    forcefield: Any,
    parameters: _CobaltParameters,
    *,
    group_index: int,
    cobalt: Any,
    nitrogens: Sequence[Any],
    trans_pairs: Sequence[tuple[int, int]],
    hydrogen_names: Mapping[Any, Sequence[str]],
) -> tuple[Any, tuple[tuple[str, str, float, float], ...]]:
    from openmm.app import ForceField, element

    by_type: dict[str, list[str]] = {}
    for role in sorted(_CO6_N_ROLES):
        base = parameters.atoms[role]
        by_type.setdefault(base.type_name, []).append(role)
    if sorted(len(value) for value in by_type.values()) != [2, 2, 2]:
        raise _error("Amber CO6 parameter types do not contain three N pairs")
    type_order = sorted(by_type)
    actual_pairs = sorted(
        (
            tuple(sorted((nitrogens[i], nitrogens[j]), key=_atom_key_from_openmm))
            for i, j in trans_pairs
        ),
        key=lambda pair: tuple(_atom_key_from_openmm(atom) for atom in pair),
    )
    expected_name_pairs = {
        frozenset(role for role in by_type[type_name])
        for type_name in type_order
    }
    actual_name_pairs = {
        frozenset(atom.name for atom in pair) for pair in actual_pairs
    }
    if actual_name_pairs == expected_name_pairs:
        pair_types = {
            frozenset(by_type[type_name]): type_name for type_name in type_order
        }
        typed_pairs = [
            (pair, pair_types[frozenset(atom.name for atom in pair)])
            for pair in actual_pairs
        ]
    else:
        # Source atom numbering is private input evidence.  When it does not
        # describe the CO6 template pairs, assign the three equivalent type
        # pairs deterministically from the geometry-derived pair order.
        typed_pairs = list(zip(actual_pairs, type_order, strict=True))
    actual_to_role: dict[Any, str] = {}
    actual_to_type: dict[Any, str] = {}
    for pair, type_name in typed_pairs:
        for actual, role in zip(pair, sorted(by_type[type_name]), strict=True):
            actual_to_role[actual] = role
            actual_to_type[actual] = type_name
    angle_terms: list[tuple[str, str, float, float]] = []
    for left, right in itertools.combinations(
        sorted(nitrogens, key=_atom_key_from_openmm), 2
    ):
        type_key = tuple(sorted((actual_to_type[left], actual_to_type[right])))
        try:
            angle, k = parameters.center_angles[type_key]
        except KeyError as exc:
            raise _error("Amber CO6 center-angle parameters are incomplete") from exc
        angle_terms.append((left.name, right.name, angle, k))
    role_to_actual: dict[str, str] = {
        "CO": cobalt.name,
        **{role: actual.name for actual, role in actual_to_role.items()},
    }
    for actual, names in hydrogen_names.items():
        role = actual_to_role[actual]
        for index, name in enumerate(names, start=1):
            role_to_actual[f"H{role[1:]}{index}"] = name
    template = ForceField._TemplateData(
        f"{_CO6_PARAMETER_PREFIX}{group_index:04d}"
    )
    for role, base_atom in parameters.atoms.items():
        actual_name = role_to_actual.get(role)
        if actual_name is None:
            raise _error(f"Amber CO6 atom role {role} has no topology mapping")
        template.addAtom(
            ForceField._TemplateAtomData(
                actual_name,
                parameters.type_names[base_atom.type_name],
                element.get_by_symbol(base_atom.element_symbol),
                {"charge": base_atom.charge},
            )
        )
    for role_left, role_right in parameters.bonds:
        try:
            template.addBondByName(role_to_actual[role_left], role_to_actual[role_right])
        except KeyError as exc:
            raise _error("Amber CO6 bond role has no topology mapping") from exc
    forcefield.registerResidueTemplate(template)
    return template, tuple(angle_terms)


def prepare_cobalt_hexammine(
    topology: Any,
    positions: Any,
    forcefield: Any,
    *,
    heavy_groups: Sequence[Sequence[AtomKey]],
) -> tuple[Any, Any]:
    """Append private CO6 ammonia hydrogens for validated heavy groups.

    All source atoms and bonds are copied verbatim into a new topology.  Each
    selected residue must be one exact seven-heavy-atom closed CoN6 component;
    unsupported or ambiguous groups raise a candidate-local preparation error.
    """

    if isinstance(heavy_groups, (str, bytes)):
        raise _error("cobalt heavy groups must be a sequence of AtomKey groups")
    groups = tuple(tuple(group) for group in heavy_groups)
    if not groups:
        return topology, positions
    group_keys = [key for group in groups for key in group]
    if len(group_keys) != len(set(group_keys)):
        raise _error("cobalt heavy groups overlap")
    source_atoms = list(topology.atoms())
    output_unit, original_positions, xyz_nm = _positions_nm(
        positions, len(source_atoms)
    )
    prepared: list[
        tuple[tuple[Any, ...], Any, tuple[Any, ...], tuple[tuple[int, int], ...]]
    ] = []
    residue_keys: set[tuple[str, str, str, str]] = set()
    for group in groups:
        selected, residue, nitrogens = _topology_group(topology, group)
        residue_key = _residue_match_key(residue)
        if residue_key in residue_keys:
            raise _error("one cobalt residue has more than one heavy group")
        residue_keys.add(residue_key)
        cobalt = next(
            atom for atom in selected if _element_symbol(atom.element.symbol) == "Co"
        )
        n_ordered = tuple(sorted(nitrogens, key=_atom_key_from_openmm))
        selected_indices = [atom.index for atom in (cobalt,) + n_ordered]
        coords = xyz_nm[selected_indices]
        vectors = coords[1:] - coords[0]
        distances = np.linalg.norm(vectors, axis=1)
        trans_pairs = _octahedral_pairs(vectors, distance_nm=distances)
        prepared.append((selected, residue, n_ordered, trans_pairs))

    from openmm import unit
    from openmm.app import Topology, element

    parameters = _load_cobalt_parameters(forcefield)
    matcher = getattr(forcefield, "_pd_cobalt_matcher", None)
    if matcher is None:
        raise _error("cobalt ForceField matcher registration failed")
    prepared_by_residue = {
        residue: (selected, nitrogens, trans_pairs)
        for selected, residue, nitrogens, trans_pairs in prepared
    }

    rebuilt = Topology()
    periodic_box = topology.getPeriodicBoxVectors()
    if periodic_box is not None:
        rebuilt.setPeriodicBoxVectors(periodic_box)
    old_to_new: dict[Any, Any] = {}
    residue_map: dict[Any, Any] = {}
    rebuilt_positions: list[np.ndarray] = []
    private_coordinates: list[np.ndarray] = []
    serial = int(getattr(forcefield, "_pd_cobalt_serial", 0))
    for chain in topology.chains():
        new_chain = rebuilt.addChain(chain.id)
        for residue in chain.residues():
            new_residue = rebuilt.addResidue(
                residue.name,
                new_chain,
                id=residue.id,
                insertionCode=residue.insertionCode,
            )
            residue_map[residue] = new_residue
            for atom in residue.atoms():
                kwargs: dict[str, Any] = {"id": atom.id}
                formal_charge = getattr(atom, "formalCharge", None)
                if formal_charge is not None:
                    kwargs["formalCharge"] = formal_charge
                old_to_new[atom] = rebuilt.addAtom(
                    atom.name,
                    atom.element,
                    new_residue,
                    **kwargs,
                )
                rebuilt_positions.append(original_positions[atom.index])
            group = prepared_by_residue.get(residue)
            if group is None:
                continue
            selected, nitrogens, trans_pairs = group
            cobalt = next(
                atom
                for atom in selected
                if _element_symbol(atom.element.symbol) == "Co"
            )
            new_nitrogens = tuple(old_to_new[atom] for atom in nitrogens)
            cobalt_position = xyz_nm[cobalt.index]
            hydrogen_names: dict[Any, tuple[str, ...]] = {}
            group_positions: list[np.ndarray] = []
            for n_index, (old_nitrogen, _new_nitrogen) in enumerate(
                zip(nitrogens, new_nitrogens, strict=True)
            ):
                direction = xyz_nm[old_nitrogen.index] - cobalt_position
                direction /= np.linalg.norm(direction)
                names = tuple(
                    f"PDH{serial:04d}_{n_index + 1}_{h_index}"
                    for h_index in range(1, 4)
                )
                hydrogen_names[old_to_new[old_nitrogen]] = names
                for name, vector in zip(
                    names,
                    _private_h_directions(direction),
                    strict=True,
                ):
                    coordinate = xyz_nm[old_nitrogen.index] + (
                        _CO6_H_BOND_LENGTH_NM * vector
                    )
                    if np.linalg.norm(
                        coordinate - xyz_nm[old_nitrogen.index]
                    ) < 1.0e-8:
                        raise _error(
                            "cobalt private hydrogen coordinate is degenerate"
                        )
                    hydrogen = rebuilt.addAtom(
                        name,
                        element.hydrogen,
                        new_residue,
                        id=f"private:cobalt-hexammine:{serial}:{n_index}:{name}",
                    )
                    rebuilt.addBond(
                        old_to_new[old_nitrogen], hydrogen, order=1
                    )
                    rebuilt_positions.append(
                        unit.Quantity(
                            np.asarray(coordinate), unit.nanometer
                        ).value_in_unit(output_unit)
                    )
                    group_positions.append(coordinate)
            if len(group_positions) != 18:
                raise _error(
                    "cobalt hexammine did not create exactly eighteen H atoms"
                )
            for left, right in itertools.combinations(group_positions, 2):
                if np.linalg.norm(left - right) < _CO6_MIN_PRIVATE_DISTANCE_NM:
                    raise _error("cobalt private hydrogen coordinates collide")
            for coordinate in group_positions:
                if any(
                    np.linalg.norm(coordinate - previous)
                    < _CO6_MIN_PRIVATE_DISTANCE_NM
                    for previous in private_coordinates
                ):
                    raise _error("cobalt private hydrogen coordinates collide")
                if any(
                    np.linalg.norm(coordinate - heavy) < _CO6_MIN_PRIVATE_DISTANCE_NM
                    for heavy in xyz_nm
                ):
                    raise _error("cobalt private hydrogen overlaps a heavy atom")
            template, angle_terms = _template_for_group(
                forcefield,
                parameters,
                group_index=serial,
                cobalt=old_to_new[cobalt],
                nitrogens=new_nitrogens,
                trans_pairs=trans_pairs,
                hydrogen_names=hydrogen_names,
            )
            residue_key = _residue_match_key(new_residue)
            matcher.templates[residue_key] = template
            matcher.angle_pairs[residue_key] = angle_terms
            private_coordinates.extend(group_positions)
            serial += 1
    for bond in topology.bonds():
        rebuilt.addBond(
            old_to_new[bond[0]],
            old_to_new[bond[1]],
            bond.type,
            bond.order,
        )

    setattr(forcefield, "_pd_cobalt_serial", serial)

    return rebuilt, unit.Quantity(np.asarray(rebuilt_positions), output_unit)


__all__ = [
    "prepare_cobalt_hexammine",
    "validate_cobalt_heavy_group",
]
