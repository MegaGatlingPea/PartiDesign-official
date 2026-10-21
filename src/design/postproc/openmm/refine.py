"""One-sample whole-system OpenMM restrained minimization."""

from __future__ import annotations

import copy
import logging
import math
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from biotite import structure as struc
from rdkit import Chem

from ..runner import (
    AtomKey,
    GaffLigandRecord,
    SampleParameterizationPayload,
    _ambertools_on_path,
)


class OpenMMPostprocessError(RuntimeError):
    """One accepted sample cannot complete the requested minimization."""


class OpenMMSystemError(RuntimeError):
    """The requested OpenMM platform/runtime is systemically unavailable."""


_SYSTEMIC_RUNTIME_MARKERS = (
    "cuda",
    "hip error",
    "opencl",
    "device lost",
    "out of memory",
    "failed to allocate",
    "cannot allocate",
    "bad_alloc",
    "resource temporarily unavailable",
    "context was destroyed",
    "invalid context",
)


def _is_systemic_runtime_error(error: Exception) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}".lower()
        if any(marker in text for marker in _SYSTEMIC_RUNTIME_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


@contextmanager
def _sample_postprocess_stage(label: str) -> Iterator[None]:
    """Translate one tightly bounded third-party sample operation."""

    try:
        yield
    except (OpenMMPostprocessError, OpenMMSystemError):
        raise
    except (
        AssertionError,
        AttributeError,
        FileNotFoundError,
        ImportError,
        NameError,
        NotImplementedError,
        TypeError,
    ):
        raise
    except Exception as error:
        if isinstance(error, MemoryError) or _is_systemic_runtime_error(error):
            raise OpenMMSystemError(
                f"{label} encountered a systemic platform/runtime failure"
            ) from error
        logging.getLogger(__name__).debug(
            "%s failed (%s): %s", label, type(error).__name__, str(error)[:2000])
        raise OpenMMPostprocessError(
            f"{label} failed ({type(error).__name__}): {str(error)[:500]}"
        ) from error


def _validate_minimized_state(
    initial_energy: float,
    final_energy: float,
    positions: np.ndarray,
    *,
    label: str,
) -> None:
    """Reject invalid numerical output, not structural or energetic quality."""

    if not math.isfinite(final_energy):
        raise OpenMMPostprocessError(f"{label} returned non-finite energy")
    if not np.isfinite(np.asarray(positions, dtype=np.float64)).all():
        raise OpenMMPostprocessError(f"{label} returned non-finite coordinates")
    if not math.isfinite(initial_energy):
        logging.getLogger(__name__).warning(
            "%s started with non-finite energy (%s), but returned finite final "
            "energy and coordinates; retaining minimized output",
            label,
            initial_energy,
        )


_ORDER_NAME_TO_RDKIT = {
    "single": Chem.BondType.SINGLE,
    "double": Chem.BondType.DOUBLE,
    "triple": Chem.BondType.TRIPLE,
    "aromatic": Chem.BondType.AROMATIC,
}


def atom_key(atom_array: struc.AtomArray, row: int) -> AtomKey:
    return (
        str(atom_array.chain_id[row]),
        int(atom_array.res_id[row]),
        str(atom_array.ins_code[row]),
        str(atom_array.res_name[row]),
        str(atom_array.atom_name[row]),
    )


def _normalized_element(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise OpenMMPostprocessError("empty element in accepted base CIF")
    return text[0].upper() + text[1:].lower()


def _ensure_ambertools_toolkit() -> Any:
    """Make the active environment's AmberTools wrapper available to OpenFF."""

    from openff.toolkit import GLOBAL_TOOLKIT_REGISTRY
    from openff.toolkit.utils.toolkits import AmberToolsToolkitWrapper

    for toolkit in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits:
        if toolkit.toolkit_name == "AmberTools":
            return toolkit
    toolkit = AmberToolsToolkitWrapper()
    GLOBAL_TOOLKIT_REGISTRY.register_toolkit(toolkit)
    return toolkit


def _validate_preassigned_charges(molecule: Any) -> None:
    """Keep GAFF 0.15.1 from replacing valid all-zero user charges with AM1-BCC."""

    if molecule.partial_charges is None:
        raise OpenMMPostprocessError("private ligand partial charges are absent")
    charges = np.asarray(
        molecule.partial_charges.m_as("elementary_charge"), dtype=np.float64
    )
    if charges.shape != (molecule.n_atoms,) or not np.isfinite(charges).all():
        raise OpenMMPostprocessError("private ligand partial charges are invalid")
    expected = float(molecule.total_charge.m_as("elementary_charge"))
    if not math.isclose(
        float(charges.sum()), expected, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise OpenMMPostprocessError(
            "private ligand partial charges do not conserve formal charge"
        )


def _gaff_mol2_text(molecule: Any) -> str:
    """Serialize the accepted explicit graph without the MDL 999-atom limit."""
    _validate_preassigned_charges(molecule)
    if not molecule.conformers:
        raise OpenMMPostprocessError("GAFF molecule lacks an accepted conformer")
    xyz = np.asarray(molecule.conformers[0].m_as("angstrom"), dtype=float)
    if xyz.shape != (molecule.n_atoms, 3) or not np.isfinite(xyz).all():
        raise OpenMMPostprocessError("GAFF accepted conformer is invalid")
    charges = molecule.partial_charges.m_as("elementary_charge")
    names = [atom.name for atom in molecule.atoms]
    if len(set(names)) != molecule.n_atoms or any(
        not name or len(name) > 5 or any(c.isspace() for c in name) for name in names
    ):
        raise OpenMMPostprocessError("GAFF atom names exceed Amber MOL2 naming limits")
    lines = ["@<TRIPOS>MOLECULE", "PARTIDESIGN",
             f"{molecule.n_atoms} {molecule.n_bonds} 1 0 0",
             "SMALL", "USER_CHARGES", "", "@<TRIPOS>ATOM"]
    for index, atom in enumerate(molecule.atoms):
        x, y, z = xyz[index]
        # Element labels carry no guessed hybridization. Antechamber derives
        # GAFF types from the accepted explicit bond graph.
        lines.append(f"{index + 1} {atom.name} {x:.8f} {y:.8f} {z:.8f} "
                     f"{atom.symbol} 1 MOL {charges[index]:.12f}")
    lines.append("@<TRIPOS>BOND")
    for index, bond in enumerate(molecule.bonds):
        order = float(bond.bond_order)
        if order not in (1.0, 2.0, 3.0):
            raise OpenMMPostprocessError("GAFF input needs concrete bond orders")
        lines.append(f"{index + 1} {bond.atom1_index + 1} "
                     f"{bond.atom2_index + 1} {int(order)}")
    lines.extend(("@<TRIPOS>SUBSTRUCTURE", "1 MOL 1", ""))
    return "\n".join(lines)


def _read_verified_gaff_types(path: str | Path, molecule: Any) -> None:
    """Reject truncated/reordered/chemically changed Antechamber output."""
    sections: dict[str, list[list[str]]] = {}
    section = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            sections.setdefault(section, [])
        elif section and line.strip():
            sections[section].append(line.split())
    rows = sections.get("@<TRIPOS>ATOM", [])
    if len(rows) != molecule.n_atoms or any(len(row) < 9 for row in rows):
        raise OpenMMPostprocessError("Antechamber returned incomplete GAFF atom types")
    xyz = np.asarray(molecule.conformers[0].m_as("angstrom"), dtype=float)
    charges = molecule.partial_charges.m_as("elementary_charge")
    for index, (row, atom) in enumerate(zip(rows, molecule.atoms, strict=True)):
        # Amber's intermediate AC format truncates labels to four characters.
        # Numeric atom IDs, coordinates, charges and complete edge mapping are
        # authoritative here; ffxml retains the original unique OpenFF names.
        if int(row[0]) != index + 1 or row[1] not in (atom.name, atom.name[:4]):
            raise OpenMMPostprocessError("Antechamber changed GAFF atom identity/order")
        output_xyz = np.asarray(list(map(float, row[2:5])))
        if not np.isfinite(output_xyz).all() or not np.allclose(
            output_xyz, xyz[index], rtol=0, atol=5.1e-4
        ):
            raise OpenMMPostprocessError("Antechamber changed the accepted conformer")
        if not math.isclose(float(row[8]), float(charges[index]),
                            rel_tol=0, abs_tol=1.0e-5):
            raise OpenMMPostprocessError("Antechamber changed preassigned charges")
        if not row[5] or row[5] in ("DU", "du", "UNK"):
            raise OpenMMPostprocessError("Antechamber returned an unknown GAFF atom type")
        atom.gaff_type = row[5]
    expected = {tuple(sorted((b.atom1_index + 1, b.atom2_index + 1))): b
                for b in molecule.bonds}
    seen = set()
    for row in sections.get("@<TRIPOS>BOND", []):
        if len(row) < 4:
            raise OpenMMPostprocessError("Antechamber returned incomplete GAFF bonds")
        pair = tuple(sorted((int(row[1]), int(row[2]))))
        if pair not in expected or pair in seen:
            raise OpenMMPostprocessError("Antechamber changed GAFF connectivity")
        bond = expected[pair]
        order = row[3].lower()
        if order == "am":
            order = "1"
        if order == "ar":
            if not bond.is_aromatic:
                raise OpenMMPostprocessError("Antechamber changed GAFF bond chemistry")
        elif order not in ("1", "2", "3") or (
            not bond.is_aromatic and int(order) != bond.bond_order
        ):
            raise OpenMMPostprocessError("Antechamber changed GAFF bond chemistry")
        seen.add(pair)
    if seen != set(expected):
        raise OpenMMPostprocessError("Antechamber lost accepted GAFF bonds")


def _gaff_parameter_key(section: str, names: tuple[str, ...]) -> tuple[str, ...]:
    if section == "IMPROPER":
        outer = sorted((names[0], names[1], names[3]))
        return outer[0], outer[1], names[2], outer[2]
    if section == "DIHE":
        a, b, c, d = names
        return (d, c, b, a) if b > c or (b == c and a > d) else names
    return min(names, names[::-1])


def _gaff_parameter_requests(
    molecule: Any, parmchk_file: Path
) -> tuple[str, dict[str, set[tuple[str, ...]]]]:
    """Request each actual GAFF graph environment once, without atom quartets.

    FRCMOD numeric fields are input placeholders. AmberTools replaces all of
    them using -att 1 before any parameters reach the OpenMM force field.
    Improper centers come from its installed PARMCHK database. Missing-angle
    empirical estimates use ideal hybridization geometry of the sanitized graph,
    never a possibly distorted accepted conformer's measured angle.
    """
    flags = {}
    for line in parmchk_file.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] == "PARM":
            flags[fields[1]] = int(fields[2])
    types = [str(atom.gaff_type) for atom in molecule.atoms]
    rdkit = molecule.to_rdkit()
    ideal_angles = {
        Chem.HybridizationType.SP: 180.0,
        Chem.HybridizationType.SP2: 120.0,
        Chem.HybridizationType.SP3: math.degrees(math.acos(-1.0 / 3.0)),
    }
    angle_hints = {}
    unknown = set(types).difference(flags)
    if unknown or any(not value or len(value) > 2 for value in types):
        raise OpenMMPostprocessError(
            f"GAFF types absent from the installed PARMCHK database: {sorted(unknown)}"
        )
    neighbors = [[] for _ in types]
    for bond in molecule.bonds:
        i, j = int(bond.atom1_index), int(bond.atom2_index)
        neighbors[i].append(j)
        neighbors[j].append(i)
    neighbors = [sorted(rows) for rows in neighbors]
    entries: dict[str, dict[tuple[str, ...], None]] = {
        name: {} for name in ("MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON")
    }

    def add(section: str, rows: tuple[int, ...]) -> None:
        key = tuple(types[row] for row in rows)
        key = _gaff_parameter_key(section, key)
        entries[section][key] = None
        if section == "ANGLE":
            angle_hints.setdefault(
                key, ideal_angles.get(rdkit.GetAtomWithIdx(rows[1]).GetHybridization(), 0.0)
            )

    for i in range(len(types)):
        add("MASS", (i,))
        add("NONBON", (i,))
        for j in neighbors[i]:
            if j > i:
                add("BOND", (i, j))
            for k in neighbors[j]:
                if k != i:
                    add("ANGLE", (i, j, k))
                if k <= j or k == i:
                    continue
                for l in neighbors[k]:
                    if l != j:
                        add("DIHE", (i, j, k, l))
        if flags[types[i]] == 1 and len(neighbors[i]) >= 3:
            # Amber's improper_id2 skips flagged atoms with fewer than three
            # explicit neighbors; they do not define an improper quartet.
            if len(neighbors[i]) != 3:
                raise OpenMMPostprocessError(
                    "GAFF improper center has more than three explicit neighbors"
                )
            a, b, d = neighbors[i]
            add("IMPROPER", (a, b, i, d))
    lines = ["PartiDesign GAFF parameter requests"]
    suffixes = {
        "MASS": " 0.000 0.000",
        "BOND": "    0.000    0.000",
        "ANGLE": "    0.000    0.000",
        "DIHE": "   1    0.000         0.000           1.000",
        "IMPROPER": "         0.0          0.0         1.0",
        "NONBON": " 0.000 0.000",
    }
    for section, keys in entries.items():
        lines.extend((section,))
        for key in keys:
            suffix = (f"    0.000 {angle_hints[key]:12.6f}" if section == "ANGLE"
                      else suffixes[section])
            lines.append("-".join(f"{name:<2}" for name in key) + suffix)
        lines.append("")
    return "\n".join(lines) + "\n", {section: set(keys) for section, keys in entries.items()}


def _clean_gaff_frcmod(
    path: Path, requested: Mapping[str, set[tuple[str, ...]]]
) -> None:
    """Apply Amber's signed-periodicity cleanup and require complete output.

    The upstream FRCMOD request path omits the MOL2 path's internal cleanup.
    Keeping repeated positive tails would overwrite multi-term proper torsions
    when ParmEd reads them. Distinct continuation/final periods remain intact.
    """
    widths = {"MASS": 1, "BOND": 2, "ANGLE": 3, "DIHE": 4,
              "IMPROPER": 4, "NONBON": 1}
    seen: dict[str, dict[tuple[Any, ...], tuple[float, ...]]] = {
        section: {} for section in widths
    }
    covered = {section: set() for section in widths}
    output = []
    section = None
    for line in path.read_text().splitlines():
        text = line.strip()
        if text in widths:
            section = text
            output.append(line)
            continue
        if not text or section is None:
            output.append(line)
            continue
        if "ATTN" in text:
            raise OpenMMPostprocessError("GAFF parameter completion needs revision: " + text)
        count = widths[section]
        if count == 1:
            fields = text.split()
            names = (fields[0],)
            numeric = fields[1:3]
        else:
            names = tuple(line[3 * i:3 * i + 2].strip() for i in range(count))
            fields = line[3 * count - 1:].split()
            numeric = fields[:4 if section == "DIHE" else 3 if section == "IMPROPER" else 2]
        expected_fields = 4 if section == "DIHE" else 3 if section == "IMPROPER" else 2
        try:
            values = tuple(float(value) for value in numeric)
        except ValueError as error:
            raise OpenMMPostprocessError("GAFF returned malformed numeric parameters") from error
        if len(values) != expected_fields or not all(math.isfinite(value) for value in values):
            raise OpenMMPostprocessError("GAFF returned incomplete or non-finite parameters")
        if any(not name for name in names):
            raise OpenMMPostprocessError("GAFF returned malformed parameter types")
        invalid = (
            (section == "MASS" and values[0] <= 0)
            or (section == "BOND" and (values[0] <= 0 or values[1] <= 0))
            or (section == "ANGLE" and (values[0] <= 0 or not 0 < values[1] <= 180))
            or (section == "NONBON" and (values[0] <= 0 or values[1] < 0))
            or (section == "DIHE" and (values[0] <= 0 or values[-1] == 0))
            or (section == "IMPROPER" and (values[0] < 0 or values[-1] == 0))
        )
        if invalid:
            raise OpenMMPostprocessError("GAFF returned invalid or uncomputed parameters")
        key = _gaff_parameter_key(section, names)
        covered[section].add(key)
        unique = key + (values[-1],) if section == "DIHE" else key
        if unique in seen[section]:
            if seen[section][unique] != values:
                raise OpenMMPostprocessError(
                    f"GAFF returned conflicting repeated parameters in {section} {key}: "
                    f"{seen[section][unique]} vs {values}"
                )
            continue
        seen[section][unique] = values
        output.append(line)
    for section, keys in requested.items():
        if keys.difference(covered[section]):
            raise OpenMMPostprocessError(f"GAFF returned incomplete {section} parameter coverage")
    path.write_text("\n".join(output) + "\n")


def _gaff_parameter_record(
    section: str, line: str
) -> tuple[tuple[str, ...], tuple[Any, ...], tuple[float, ...]] | None:
    """Parse one numeric FRCMOD record for conflict repair bookkeeping."""
    widths = {"MASS": 1, "BOND": 2, "ANGLE": 3, "DIHE": 4,
              "IMPROPER": 4, "NONBON": 1}
    count = widths[section]
    text = line.strip()
    if not text or "ATTN" in text:
        return None
    if count == 1:
        fields = text.split()
        if len(fields) < 3:
            return None
        names = (fields[0],)
        numeric = fields[1:3]
    else:
        names = tuple(line[3 * i:3 * i + 2].strip() for i in range(count))
        fields = line[3 * count - 1:].split()
        numeric = fields[:4 if section == "DIHE" else 3 if section == "IMPROPER" else 2]
    expected = 4 if section == "DIHE" else 3 if section == "IMPROPER" else 2
    if len(numeric) != expected:
        return None
    try:
        values = tuple(float(value) for value in numeric)
    except ValueError:
        return None
    key = _gaff_parameter_key(section, names)
    unique = key + (values[-1],) if section == "DIHE" else key
    return key, unique, values


def _gaff_parameter_conflict_keys(path: Path) -> dict[str, set[tuple[str, ...]]]:
    """Return FRCMOD keys whose same-period records disagree."""
    widths = {"MASS": 1, "BOND": 2, "ANGLE": 3, "DIHE": 4,
              "IMPROPER": 4, "NONBON": 1}
    section = None
    seen: dict[str, dict[tuple[Any, ...], tuple[float, ...]]] = {
        name: {} for name in widths
    }
    conflicts: dict[str, set[tuple[str, ...]]] = {}
    for line in path.read_text().splitlines():
        text = line.strip()
        if text in widths:
            section = text
            continue
        if section is None:
            continue
        record = _gaff_parameter_record(section, line)
        if record is None:
            continue
        key, unique, values = record
        previous = seen[section].get(unique)
        if previous is not None and previous != values:
            conflicts.setdefault(section, set()).add(key)
        else:
            seen[section][unique] = values
    return conflicts


def _replace_gaff_parameter_conflicts(
    path: Path,
    replacement_path: Path,
    conflicts: Mapping[str, set[tuple[str, ...]]],
) -> None:
    """Replace only ambiguous records with Amber's graph-aware probe output."""
    widths = {"MASS": 1, "BOND": 2, "ANGLE": 3, "DIHE": 4,
              "IMPROPER": 4, "NONBON": 1}
    replacement_lines: dict[str, list[str]] = {section: [] for section in conflicts}
    section = None
    for line in replacement_path.read_text().splitlines():
        text = line.strip()
        if text in widths:
            section = text
            continue
        if section not in conflicts:
            continue
        record = _gaff_parameter_record(section, line)
        if record is not None and record[0] in conflicts[section]:
            replacement_lines[section].append(line)
    missing = [
        (section, key)
        for section, keys in conflicts.items()
        if not replacement_lines[section]
        for key in keys
    ]
    if missing:
        raise OpenMMPostprocessError(
            f"GAFF conflict repair did not return replacement parameters: {missing}"
        )

    output: list[str] = []
    section = None
    inserted: set[str] = set()
    for line in path.read_text().splitlines():
        text = line.strip()
        if text in widths:
            section = text
            output.append(line)
            if section in replacement_lines:
                output.extend(replacement_lines[section])
                inserted.add(section)
            continue
        if section in conflicts:
            record = _gaff_parameter_record(section, line)
            if record is not None and record[0] in conflicts[section]:
                continue
        output.append(line)
    if set(replacement_lines).difference(inserted):
        raise OpenMMPostprocessError(
            "GAFF conflict repair could not locate a target FRCMOD section"
        )
    path.write_text("\n".join(output) + "\n")


def _gaff_parameter_probe_mol2(
    molecule: Any, requested: Mapping[str, set[tuple[str, ...]]]
) -> str:
    """Build compact typed MOL2 probes for Amber's graph-aware ``parmchk2``.

    Requesting a FRCMOD by atom-type key makes ``parmchk2`` print every source
    family present in GAFF. A small disconnected MOL2 containing one graph
    environment per requested key lets Amber perform its normal graph-aware
    selection without scanning or ambiguously combining an entire ligand.
    """
    element_by_type: dict[str, str] = {}
    for atom in molecule.atoms:
        type_name = str(atom.gaff_type)
        element = _normalized_element(atom.symbol)
        previous = element_by_type.setdefault(type_name, element)
        if previous != element:
            raise OpenMMPostprocessError(
                f"GAFF type {type_name!r} maps to multiple elements: "
                f"{previous!r} and {element!r}"
            )

    atoms: list[tuple[str, str]] = []
    bonds: list[tuple[int, int]] = []

    def add_probe(key: tuple[str, ...], *, improper: bool = False) -> None:
        start = len(atoms) + 1
        for type_name in key:
            if type_name not in element_by_type:
                raise OpenMMPostprocessError(
                    f"GAFF probe has no element for atom type {type_name!r}"
                )
            atoms.append((element_by_type[type_name], type_name))
        if len(key) == 2:
            bonds.append((start, start + 1))
        elif len(key) == 3:
            bonds.extend(((start, start + 1), (start + 1, start + 2)))
        elif len(key) == 4:
            if improper:
                bonds.extend(
                    ((start, start + 2), (start + 1, start + 2), (start + 2, start + 3))
                )
            else:
                bonds.extend(
                    ((start, start + 1), (start + 1, start + 2), (start + 2, start + 3))
                )

    # MASS and NONBON share one atom environment. Keep the probe graph small
    # even for a large ligand with many repeated atom types.
    for key in sorted(requested.get("MASS", ())):
        add_probe(key)
    probed_types = {key[0] for key in requested.get("MASS", ())}
    for section in ("BOND", "ANGLE", "DIHE", "IMPROPER"):
        for key in sorted(requested.get(section, ())):
            add_probe(key, improper=section == "IMPROPER")
    for key in sorted(requested.get("NONBON", ())):
        if key[0] not in probed_types:
            add_probe(key)
            probed_types.add(key[0])
    if not atoms:
        raise OpenMMPostprocessError("GAFF parameter probe graph is empty")
    if len(atoms) > 9999:
        raise OpenMMPostprocessError("GAFF parameter probe graph exceeds MOL2 atom names")

    lines = [
        "@<TRIPOS>MOLECULE",
        "PARTIDESIGN_GAFF_PROBES",
        f"{len(atoms)} {len(bonds)} 1 0 0",
        "SMALL",
        "USER_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    for index, (element, type_name) in enumerate(atoms, start=1):
        # The type field is authoritative for parmchk2; the short name only
        # needs a valid element-leading MOL2 label.
        name = (f"{element[:2]}{index:03d}" if len(element) > 1 else f"{element}{index:04d}")
        lines.append(
            f"{index:6d} {name:<5} {1.5 * index:10.4f} {0.0:10.4f} {0.0:10.4f} "
            f"{type_name:<2} 1 MOL {0.0: .12f}"
        )
    lines.append("@<TRIPOS>BOND")
    for index, (atom_i, atom_j) in enumerate(bonds, start=1):
        lines.append(f"{index:6d} {atom_i:4d} {atom_j:4d} 1")
    lines.extend(("@<TRIPOS>SUBSTRUCTURE", "     1 MOL 1", ""))
    return "\n".join(lines)


class _PrechargedGAFFMixin:
    """Narrow 0.15.1 override: assigned finite charges are always authoritative."""

    def _molecule_has_user_charges(self, molecule: Any) -> bool:
        _validate_preassigned_charges(molecule)
        return True

    def generate_residue_template(self, molecule: Any) -> str:
        # GAFF 0.15.1 unconditionally embeds a new conformer. The private
        # molecule already carries the accepted structure plus its hydrogens;
        # embedding a complete polymer is expensive and unnecessary here.
        class AcceptedConformerMolecule(type(molecule)):
            def generate_conformers(self, *args, **kwargs):
                conformers = self.conformers
                if not conformers:
                    raise OpenMMPostprocessError("GAFF molecule lacks an accepted conformer")
                for conformer in conformers:
                    xyz = np.asarray(conformer.m_as("angstrom"), dtype=float)
                    if xyz.shape != (self.n_atoms, 3) or not np.isfinite(xyz).all():
                        raise OpenMMPostprocessError("GAFF accepted conformer is invalid")
                return None

        accepted = AcceptedConformerMolecule(molecule)
        self._accepted_gaff_molecule = accepted
        try:
            return super().generate_residue_template(accepted)
        finally:
            del self._accepted_gaff_molecule

    def _run_antechamber(self, molecule_filename, input_format="sdf",
                         gaff_mol2_filename=None, frcmod_filename=None, verbosity=0):
        import shutil
        import subprocess
        import tempfile
        from ..runner import _ambertools_executable

        molecule = self._accepted_gaff_molecule
        charge = float(molecule.total_charge.m_as("elementary_charge"))
        if not math.isclose(charge, round(charge), abs_tol=1.0e-6):
            raise OpenMMPostprocessError("GAFF molecular charge must be integral")
        major = str(self._gaff_major_version)
        if major not in ("1", "2"):
            raise ValueError("unsupported GAFF major version")
        with tempfile.TemporaryDirectory(prefix="partidesign-gaff-") as directory:
            root = Path(directory)
            (root / "input.mol2").write_text(_gaff_mol2_text(molecule))
            shutil.copyfile(self.gaff_dat_filename, root / "gaff.dat")
            # Only atom typing is requested: accepted bonds, coordinates,
            # atom order/names, and preassigned charges remain authoritative.
            parmchk_file = (_ambertools_executable("parmchk2").parent.parent
                            / "dat" / "antechamber" / "PARMCHK.DAT")
            if not parmchk_file.is_file():
                raise OpenMMPostprocessError("installed AmberTools PARMCHK database is absent")
            requested = None
            commands = [
                [str(_ambertools_executable("antechamber")),
                 "-i", "input.mol2", "-fi", "mol2", "-o", "output.mol2",
                 "-fo", "mol2", "-at", "gaff" if major == "1" else "gaff2",
                 "-j", "1", "-an", "no", "-du", "no", "-seq", "no",
                 "-nc", str(round(charge)), "-dr", "no", "-s", str(verbosity)],
                [str(_ambertools_executable("parmchk2")),
                 "-i", "requests.frcmod", "-f", "frcmod", "-p", "gaff.dat",
                 "-o", "output.frcmod", "-s", major, "-a", "Y", "-att", "1", "-fc", "2",
                 "-c", str(parmchk_file)],
            ]
            for command, output_name in zip(commands, ("output.mol2", "output.frcmod"), strict=True):
                result = subprocess.run(command, cwd=root, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                output = root / output_name
                if result.returncode or not output.is_file() or not output.stat().st_size:
                    raise OpenMMPostprocessError(
                        f"{Path(command[0]).name} failed to produce valid {output_name}: "
                        + result.stdout[-2000:])
                self._check_for_errors(result.stdout)
                if output_name == "output.mol2":
                    _read_verified_gaff_types(output, molecule)
                    requests, requested = _gaff_parameter_requests(molecule, parmchk_file)
                    (root / "requests.frcmod").write_text(requests)
                else:
                    try:
                        _clean_gaff_frcmod(output, requested)
                    except OpenMMPostprocessError:
                        conflicts = _gaff_parameter_conflict_keys(output)
                        if not conflicts:
                            raise
                        conflict_requested = {
                            section: set(keys) for section, keys in conflicts.items()
                        }
                        (root / "parameter_probes.mol2").write_text(
                            _gaff_parameter_probe_mol2(molecule, conflict_requested)
                        )
                        repair = subprocess.run(
                            [str(_ambertools_executable("parmchk2")),
                             "-i", "parameter_probes.mol2", "-f", "mol2", "-p", "gaff.dat",
                             "-o", "conflict.frcmod", "-s", major, "-a", "Y",
                             "-att", "1", "-fc", "2", "-c", str(parmchk_file)],
                            cwd=root, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                        )
                        replacement = root / "conflict.frcmod"
                        if repair.returncode or not replacement.is_file() or not replacement.stat().st_size:
                            raise OpenMMPostprocessError(
                                "GAFF conflict repair failed to produce valid parameters: "
                                + repair.stdout[-2000:]
                            )
                        self._check_for_errors(repair.stdout)
                        _replace_gaff_parameter_conflicts(output, replacement, conflicts)
                        _clean_gaff_frcmod(output, requested)
            shutil.copyfile(root / "output.mol2", gaff_mol2_filename)
            shutil.copyfile(root / "output.frcmod", frcmod_filename)
        return gaff_mol2_filename, frcmod_filename

    def _read_gaff_atom_types_from_mol2(self, gaff_mol2_filename, molecule):
        _read_verified_gaff_types(gaff_mol2_filename, molecule)


def _precharged_gaff_generator(
    generator_type: type,
    *,
    molecules: Sequence[Any],
) -> Any:
    class PrechargedGAFFTemplateGenerator(_PrechargedGAFFMixin, generator_type):
        pass

    return PrechargedGAFFTemplateGenerator(
        molecules=molecules,
        forcefield="gaff-2.2.20",
    )


def _build_private_ligand(
    record: GaffLigandRecord,
    atom_array: struc.AtomArray,
    row_by_key: Mapping[AtomKey, int],
) -> Chem.Mol:
    rw = Chem.RWMol()
    for element, charge, hydrogen_count in zip(
        record.elements, record.formal_charges, record.hydrogen_counts, strict=True
    ):
        atom = Chem.Atom(_normalized_element(element))
        atom.SetFormalCharge(int(charge))
        # Restore the validated product hydrogen state before aromatic
        # sanitization; an aromatic N cannot infer whether it was [nH].
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(int(hydrogen_count))
        rw.AddAtom(atom)
    for atom_i, atom_j, order_name in record.bonds:
        rw.AddBond(int(atom_i), int(atom_j), _ORDER_NAME_TO_RDKIT[order_name])
        if order_name == "aromatic":
            bond = rw.GetBondBetweenAtoms(int(atom_i), int(atom_j))
            bond.SetIsAromatic(True)
            rw.GetAtomWithIdx(int(atom_i)).SetIsAromatic(True)
            rw.GetAtomWithIdx(int(atom_j)).SetIsAromatic(True)
    molecule = rw.GetMol()
    conformer = Chem.Conformer(len(record.heavy_keys))
    for local_index, key in enumerate(record.heavy_keys):
        if key not in row_by_key:
            raise OpenMMPostprocessError(f"GAFF heavy key is absent from base CIF: {key}")
        xyz = np.asarray(atom_array.coord[row_by_key[key]], dtype=np.float64)
        conformer.SetAtomPosition(local_index, tuple(float(value) for value in xyz))
    molecule.AddConformer(conformer, assignId=True)
    Chem.SanitizeMol(molecule)
    molecule_h = Chem.AddHs(molecule, addCoords=True)
    from .hydrogens import repair_added_hydrogen_geometry
    with _sample_postprocess_stage("private ligand hydrogen geometry"):
        repair_added_hydrogen_geometry(molecule_h, len(record.heavy_keys))
    Chem.SanitizeMol(molecule_h)
    observed_h = []
    for local_index in range(len(record.heavy_keys)):
        atom = molecule_h.GetAtomWithIdx(local_index)
        observed_h.append(
            sum(1 for neighbour in atom.GetNeighbors() if neighbour.GetAtomicNum() == 1)
        )
    if tuple(observed_h) != tuple(record.hydrogen_counts):
        raise OpenMMPostprocessError(
            "private ligand hydrogen state differs from the selected chemistry"
        )
    return molecule_h


def _openmm_bond_order(raw_type: int) -> float | None:
    mapping = {
        int(struc.BondType.ANY): None,
        int(struc.BondType.SINGLE): 1.0,
        int(struc.BondType.DOUBLE): 2.0,
        int(struc.BondType.TRIPLE): 3.0,
        int(struc.BondType.AROMATIC): 1.5,
        int(struc.BondType.AROMATIC_SINGLE): 1.5,
        int(struc.BondType.AROMATIC_DOUBLE): 1.5,
        int(struc.BondType.AROMATIC_TRIPLE): 1.5,
    }
    if int(raw_type) == int(struc.BondType.COORDINATION):
        raise OpenMMPostprocessError(
            "metal coordination is outside the supported OpenMM routes"
        )
    if int(raw_type) not in mapping:
        raise OpenMMPostprocessError(f"unsupported authoritative bond type {raw_type}")
    return mapping[int(raw_type)]


def _build_openmm_topology(
    atom_array: struc.AtomArray,
    payload: SampleParameterizationPayload,
):
    from openff.toolkit.topology import Molecule
    from openmm import unit
    from openmm.app import Element, Topology

    if atom_array.bonds is None:
        raise OpenMMPostprocessError("accepted structure lacks an authoritative BondList")
    keys = [atom_key(atom_array, row) for row in range(len(atom_array))]
    if len(keys) != len(set(keys)):
        raise OpenMMPostprocessError("original-heavy atom keys are not unique")
    row_by_key = {key: row for row, key in enumerate(keys)}
    gaff_by_key: dict[AtomKey, tuple[GaffLigandRecord, int]] = {}
    private_molecules: list[Molecule] = []
    private_rdkit: dict[str, Chem.Mol] = {}
    for record in payload.gaff_ligands:
        molecule_h = _build_private_ligand(record, atom_array, row_by_key)
        private_rdkit[record.record_id] = molecule_h
        private_molecules.append(
            Molecule.from_rdkit(
                molecule_h,
                allow_undefined_stereo=True,
                hydrogens_are_explicit=True,
            )
        )
        for local_index, key in enumerate(record.heavy_keys):
            if key in gaff_by_key:
                raise OpenMMPostprocessError("one heavy atom belongs to two GAFF records")
            gaff_by_key[key] = (record, local_index)

    hierarchy: OrderedDict[str, OrderedDict[tuple[int, str, str], list[int]]] = OrderedDict()
    for row, key in enumerate(keys):
        chain_id, res_id, ins_code, res_name, _ = key
        hierarchy.setdefault(chain_id, OrderedDict()).setdefault(
            (res_id, ins_code, res_name), []
        ).append(row)
    topology = Topology()
    openmm_atom_by_row: dict[int, Any] = {}
    openmm_atom_by_key: dict[AtomKey, Any] = {}
    positions_angstrom: list[np.ndarray] = []
    for chain_id, residues in hierarchy.items():
        chain = topology.addChain(chain_id)
        for (res_id, ins_code, res_name), rows in residues.items():
            residue = topology.addResidue(
                res_name, chain, id=str(res_id), insertionCode=ins_code
            )
            seen_names: set[str] = set()
            for row in rows:
                name = str(atom_array.atom_name[row])
                if name in seen_names:
                    raise OpenMMPostprocessError(f"duplicate atom name in residue: {name}")
                seen_names.add(name)
                key = keys[row]
                omm_atom = topology.addAtom(
                    name,
                    Element.getBySymbol(_normalized_element(atom_array.element[row])),
                    residue,
                    id=f"src:{row}",
                )
                openmm_atom_by_row[row] = omm_atom
                openmm_atom_by_key[key] = omm_atom
                positions_angstrom.append(np.asarray(atom_array.coord[row], dtype=np.float64))

            ligand_records = {
                gaff_by_key[key][0].record_id
                for key in (keys[row] for row in rows)
                if key in gaff_by_key
            }
            if len(ligand_records) > 1:
                raise OpenMMPostprocessError("one residue contains multiple GAFF molecules")
            if ligand_records:
                record_id = next(iter(ligand_records))
                molecule_h = private_rdkit[record_id]
                record = next(r for r in payload.gaff_ligands if r.record_id == record_id)
                conformer = molecule_h.GetConformer()
                heavy_count = len(record.heavy_keys)
                next_h = 1
                rd_to_omm = {
                    local_index: openmm_atom_by_key[key]
                    for local_index, key in enumerate(record.heavy_keys)
                }
                for rd_index in range(heavy_count, molecule_h.GetNumAtoms()):
                    rd_atom = molecule_h.GetAtomWithIdx(rd_index)
                    if rd_atom.GetAtomicNum() != 1:
                        raise OpenMMPostprocessError("private ligand atom order is not heavy-first")
                    name = f"H{next_h}"
                    while name in seen_names:
                        next_h += 1
                        name = f"H{next_h}"
                    seen_names.add(name)
                    omm_atom = topology.addAtom(name, Element.getBySymbol("H"), residue)
                    rd_to_omm[rd_index] = omm_atom
                    point = conformer.GetAtomPosition(rd_index)
                    positions_angstrom.append(np.asarray((point.x, point.y, point.z)))
                    next_h += 1
                for bond in molecule_h.GetBonds():
                    rd_i, rd_j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if rd_i < heavy_count and rd_j < heavy_count:
                        continue
                    topology.addBond(
                        rd_to_omm[rd_i], rd_to_omm[rd_j], order=bond.GetBondTypeAsDouble()
                    )

    coordination_edges = {
        tuple(sorted((record.heavy_keys[i], record.heavy_keys[j])))
        for record in payload.fixed_coordination_records
        for i, j, order in record.bonds if order == "dative"
    }
    expected_edges: set[tuple[AtomKey, AtomKey]] = set()
    for row_i, row_j, raw_type in atom_array.bonds.as_array():
        i, j = int(row_i), int(row_j)
        from ..runner import _nonbonded_metal_relation
        if _nonbonded_metal_relation(atom_array, i, j, int(raw_type)):
            continue
        pair = tuple(sorted((keys[i], keys[j])))
        order = 1.0 if pair in coordination_edges else _openmm_bond_order(int(raw_type))
        topology.addBond(openmm_atom_by_row[i], openmm_atom_by_row[j], order=order)
        expected_edges.add(tuple(sorted((keys[i], keys[j]))))
    actual_edges = {
        tuple(sorted((keys[int(a.id.removeprefix("src:"))], keys[int(b.id.removeprefix("src:"))])))
        for a, b in topology.bonds()
        if a.id and b.id and a.id.startswith("src:") and b.id.startswith("src:")
    }
    if actual_edges != expected_edges:
        raise OpenMMPostprocessError("OpenMM topology changed the authoritative heavy graph")
    return (
        topology,
        unit.Quantity(np.asarray(positions_angstrom), unit.angstrom),
        private_molecules,
        keys,
    )


def _create_ordinary_system(
    forcefield: Any,
    modeller: Any,
    private_molecules: Sequence[Any],
    gaff_template_generator: Any,
    *,
    charge_method: str,
    nonbonded_method: Any,
    constraints: Any,
) -> Any:
    with _sample_postprocess_stage("GAFF/template parameterization"):
        if private_molecules:
            ambertools = _ensure_ambertools_toolkit()
            if charge_method not in {"gasteiger", "am1bcc"}:
                raise ValueError(f"unsupported charge method {charge_method!r}")
            for molecule in private_molecules:
                molecule.assign_partial_charges(
                    partial_charge_method=charge_method,
                    toolkit_registry=ambertools,
                    normalize_partial_charges=True,
                    **({"use_conformers": molecule.conformers} if charge_method == "am1bcc" else {}),
                )
                _validate_preassigned_charges(molecule)
            generator = _precharged_gaff_generator(
                gaff_template_generator,
                molecules=private_molecules,
            )
            forcefield.registerTemplateGenerator(generator.generator)
        return forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=nonbonded_method,
            constraints=constraints,
            rigidWater=True,
        )


def _run_openmm_minimizer(simulation: Any, unit: Any) -> None:
    with _sample_postprocess_stage("OpenMM minimization"):
        simulation.minimizeEnergy(
            tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=500,
        )


def _ordinary_minimize(
    atom_array: struc.AtomArray,
    *,
    platform_name: str,
    payload: SampleParameterizationPayload,
    charge_method: str,
) -> tuple[struc.AtomArray, float, float, float]:
    from openmm import CustomExternalForce, Platform, VerletIntegrator, unit
    from openmm.app import ForceField, HBonds, Modeller, NoCutoff, Simulation
    from openmmforcefields.generators import GAFFTemplateGenerator

    from .scaffold import split_private_backbone_breaks

    # Repair only the parameterization view: an input chain may contain real
    # backbone gaps even without an inference crop. Never bridge those gaps or
    # export the private terminal atoms added for Amber template matching.
    keys = [atom_key(atom_array, row) for row in range(len(atom_array))]
    private_atoms = atom_array
    with _ambertools_on_path():
        topology, positions, private_molecules, _ = _build_openmm_topology(
            private_atoms,
            payload,
        )
    from .preparation import repair_standard_heavy_atoms, add_polymer_hydrogens
    from .native import prepare_native_nucleic, native_hydrogen_variants

    topology = split_private_backbone_breaks(topology)
    with _sample_postprocess_stage("standard heavy-atom completion"):
        topology, positions = repair_standard_heavy_atoms(topology, positions)
    forcefield = ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    from .protein import prepare_native_protein
    with _sample_postprocess_stage("native protein endpoints"):
        topology, positions = prepare_native_protein(topology, positions, forcefield)
    with _sample_postprocess_stage("native nucleic endpoints"):
        topology, positions = prepare_native_nucleic(topology, positions, forcefield)
    if payload.cobalt_hexammine_groups:
        from .cobalt import prepare_cobalt_hexammine
        with _sample_postprocess_stage("cobalt hexammine parameterization"):
            topology, positions = prepare_cobalt_hexammine(
                topology, positions, forcefield,
                heavy_groups=payload.cobalt_hexammine_groups)
    coordination_fixed = set()
    if payload.fixed_coordination_records:
        from .coordination import prepare_fixed_coordination
        with _sample_postprocess_stage("fixed coordination parameterization"):
            topology, positions, fixed_keys = prepare_fixed_coordination(
                topology, positions, forcefield, payload.fixed_coordination_records)
            coordination_fixed.update(fixed_keys)
    modeller = Modeller(topology, positions)
    with _sample_postprocess_stage("OpenMM hydrogenation"):
        add_polymer_hydrogens(
            modeller, forcefield=None,
            variants=native_hydrogen_variants(topology, forcefield))

    key_set = set(keys)
    source_atoms = {
        (
            atom.residue.chain.id,
            int(atom.residue.id),
            atom.residue.insertionCode,
            atom.residue.name,
            atom.name,
        ): atom
        for atom in modeller.topology.atoms()
        if (
            atom.residue.chain.id,
            int(atom.residue.id),
            atom.residue.insertionCode,
            atom.residue.name,
            atom.name,
        )
        in key_set
    }
    if set(source_atoms) != key_set:
        raise OpenMMPostprocessError("hydrogenation lost or duplicated an original heavy atom")
    cobalt_fixed = {key for group in payload.cobalt_hexammine_groups for key in group}
    if not cobalt_fixed.issubset(key_set):
        raise OpenMMPostprocessError("cobalt context references absent original atoms")
    fixed_context_keys = cobalt_fixed | coordination_fixed
    fixed_context_residues = {key[:4] for key in fixed_context_keys}
    with _ambertools_on_path():
        system = _create_ordinary_system(
            forcefield,
            modeller,
            private_molecules,
            GAFFTemplateGenerator,
            charge_method=charge_method,
            nonbonded_method=NoCutoff,
            constraints=None if payload.native_moving_keys or fixed_context_keys else HBonds,
        )
    restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter(
        "k", 10.0 * unit.kilocalories_per_mole / unit.angstrom**2
    )
    for name in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    key_by_particle: dict[int, AtomKey] = {}
    native_moving = set(payload.native_moving_keys)
    if not native_moving.issubset(key_set):
        raise OpenMMPostprocessError("native moving keys are absent from the original structure")
    moving_residues = {key[:4] for key in native_moving}
    for atom, position in zip(modeller.topology.atoms(), modeller.positions):
        key: AtomKey = (
            atom.residue.chain.id,
            int(atom.residue.id),
            atom.residue.insertionCode,
            atom.residue.name,
            atom.name,
        )
        if (native_moving and key[:4] not in moving_residues) or key[:4] in fixed_context_residues:
            system.setParticleMass(atom.index, 0)
        if key in key_set:
            xyz = np.asarray(position.value_in_unit(unit.nanometer), dtype=float)
            if (not native_moving or key in native_moving) and key not in fixed_context_keys:
                restraint.addParticle(atom.index, xyz.tolist())
            key_by_particle[atom.index] = key
    system.addForce(restraint)
    from .stability import guard_zero_lj_exceptions
    guard_zero_lj_exceptions(system, modeller.topology)

    try:
        platform = Platform.getPlatformByName(platform_name)
    except (OpenMMPostprocessError, OpenMMSystemError):
        raise
    except Exception as error:
        raise OpenMMSystemError(f"OpenMM platform {platform_name!r} unavailable") from error
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    try:
        properties = ({"DeviceIndex": "0", "Precision": "mixed", "UseBlockingSync": "true"}
                      if platform_name == "CUDA" else {"Threads": "1"} if platform_name == "CPU" else {})
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    except (OpenMMPostprocessError, OpenMMSystemError):
        raise
    except Exception as error:
        raise OpenMMSystemError("OpenMM Context initialization failed") from error
    with _sample_postprocess_stage("OpenMM position initialization"):
        simulation.context.setPositions(modeller.positions)
    with _sample_postprocess_stage("OpenMM initial-state evaluation"):
        initial = simulation.context.getState(getEnergy=True)
        initial_energy = float(
            initial.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        if not math.isfinite(initial_energy):
            raise OpenMMPostprocessError("OpenMM initial energy is non-finite before minimization")
    _run_openmm_minimizer(simulation, unit)
    with _sample_postprocess_stage("OpenMM final-state evaluation"):
        state = simulation.context.getState(
            getEnergy=True,
            getForces=True,
            getPositions=True,
        )
        final_energy = float(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        forces = np.asarray(
            state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
        )
        if not np.isfinite(forces).all():
            raise OpenMMPostprocessError("OpenMM minimization returned non-finite forces")
        force_rows = [i for i in range(system.getNumParticles())
                      if system.getParticleMass(i).value_in_unit(unit.dalton) > 0]
        rms_force = float(np.sqrt(np.mean(np.sum(forces[force_rows] ** 2, axis=-1))))
        positions_nm = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        )
    _validate_minimized_state(
        initial_energy,
        final_energy,
        positions_nm,
        label="OpenMM minimization",
    )
    refined = copy.deepcopy(atom_array)
    row_by_key = {atom_key(refined, row): row for row in range(len(refined))}
    recovered: set[AtomKey] = set()
    for particle_index, key in key_by_particle.items():
        row = row_by_key[key]
        if (native_moving and key not in native_moving) or key in fixed_context_keys:
            if not np.allclose(positions_nm[particle_index] * 10.0, atom_array.coord[row], rtol=0, atol=1e-5):
                raise OpenMMPostprocessError("native polymer minimization moved fixed context")
        else:
            refined.coord[row] = positions_nm[particle_index] * 10.0
        recovered.add(key)
    if recovered != key_set:
        raise OpenMMPostprocessError("minimized heavy-atom map-back is incomplete")
    return refined, initial_energy, final_energy, rms_force


def refine_one(
    base_cif: str | Path,
    refined_cif: str | Path,
    *,
    config: Mapping[str, Any],
    payload: SampleParameterizationPayload,
) -> tuple[float, float, float]:
    """Refine one accepted base CIF without changing its public topology/identity."""

    from ...utils.reader import read_structure
    atom_array = read_structure(Path(base_cif), ccd_path=None)
    if payload.output_atom_aliases:
        from ..runner import payload_for_output
        payload = payload_for_output(payload, atom_array)
    # Design flags are not serialized in the public CIF. Recover only the
    # accepted payload's explicit ownership before selecting a context route.
    moving_keys = set(getattr(payload, "native_moving_keys", ())) | set(getattr(payload, "local_moving_keys", ()))
    moving_keys.update(key for record in getattr(payload, "peptide_records", ())
                       for key in getattr(record, "heavy_keys", ()))
    if moving_keys:
        public_keys = {atom_key(atom_array, row) for row in range(len(atom_array))}
        if not moving_keys.issubset(public_keys):
            raise OpenMMPostprocessError("payload design ownership references absent public atoms")
        atom_array.is_design[:] = [atom_key(atom_array, row) in moving_keys
                                  for row in range(len(atom_array))]
    public_atoms = atom_array
    component_overrides = {}
    private_context = bool(getattr(payload, "linked_context_residues", ())
                           or getattr(payload, "fixed_ligand_completions", ()))
    if private_context:
        from .context_view import materialize_fixed_context
        with _sample_postprocess_stage("fixed context identity and completion"):
            atom_array, payload, component_overrides = materialize_fixed_context(atom_array, payload)

    def write_result(refined):
        from .peptide import write_peptide_coordinates
        if private_context:
            from .context_view import restore_public_context
            refined = restore_public_context(public_atoms, atom_array, refined)
        write_peptide_coordinates(base_cif, refined_cif, public_atoms, refined)

    charge_method = config.get("charge_method", "gasteiger")
    if charge_method not in {"gasteiger", "am1bcc"}:
        raise ValueError(f"unsupported charge method {charge_method!r}")
    from ...utils.atoms import PROTEIN_RES_ATOMS
    modified_context = ((np.asarray(atom_array.mol_type) == "protein")
                        & ~np.asarray(atom_array.is_design, dtype=bool)
                        & ~np.isin(atom_array.res_name, tuple(set(PROTEIN_RES_ATOMS) - {"UNK"})))
    modified_nucleic = (~np.asarray(atom_array.is_design, dtype=bool)
                        & np.isin(atom_array.mol_type, ("rna", "dna"))
                        & ~np.isin(atom_array.res_name, ("A", "C", "G", "U", "DA", "DC", "DG", "DT")))
    if modified_context.any() and modified_nucleic.any():
        raise OpenMMPostprocessError("combined modified protein and nucleic context is unsupported")
    if modified_nucleic.any():
        from .context_nucleic import minimize_modified_nucleic_context
        from .peptide import write_peptide_coordinates
        with _sample_postprocess_stage("modified nucleic context parameterization"):
            refined, initial_energy, final_energy, rms_force = minimize_modified_nucleic_context(
                atom_array, payload=payload,
                platform_name=str(config.get("platform", "CUDA")).upper(),
                charge_method=charge_method, component_overrides=component_overrides)
        with _sample_postprocess_stage("modified nucleic coordinate serialization"):
            write_result(refined)
        return initial_energy, final_energy, rms_force
    if modified_context.any():
        from .context_polymer import minimize_modified_protein_context
        from .peptide import write_peptide_coordinates
        with _sample_postprocess_stage("modified protein context parameterization"):
            refined, initial_energy, final_energy, rms_force = minimize_modified_protein_context(
                atom_array, payload=payload,
                platform_name=str(config.get("platform", "CUDA")).upper(),
                charge_method=charge_method, component_overrides=component_overrides)
        with _sample_postprocess_stage("modified context coordinate serialization"):
            write_result(refined)
        return initial_energy, final_energy, rms_force
    if payload.local_product_records:
        from .local_refine import minimize_local_products
        from .peptide import write_peptide_coordinates

        refined, initial_energy, final_energy, rms_force = minimize_local_products(
            atom_array, payload=payload,
            platform_name=str(config.get("platform", "CUDA")).upper(),
            charge_method=charge_method,
        )
        with _sample_postprocess_stage("local product serialization"):
            write_result(refined)
        return initial_energy, final_energy, rms_force
    if payload.peptide_records:
        from .peptide import minimize_peptides, write_peptide_coordinates

        refined, initial_energy, final_energy, rms_force = minimize_peptides(
            atom_array, payload=payload,
            platform_name=str(config.get("platform", "CUDA")).upper(),
            charge_method=charge_method,
        )
        with _sample_postprocess_stage("peptide serialization"):
            write_result(refined)
        return initial_energy, final_energy, rms_force
    component_uid = np.asarray(atom_array.component_uid, dtype=np.int64)
    cross_component = []
    if atom_array.bonds is not None:
        cross_component = [
            (int(i), int(j), int(t))
            for i, j, t in atom_array.bonds.as_array()
            if component_uid[int(i)] != component_uid[int(j)]
        ]
    if payload.covalent_link is not None:
        from .covalent import minimize_covalent

        refined, initial_energy, final_energy, rms_force = minimize_covalent(
            atom_array,
            platform_name=str(config.get("platform", "CUDA")).upper(),
            payload=payload,
            charge_method=charge_method,
        )
    else:
        if payload.covalent_link is not None:
            raise OpenMMPostprocessError("covalent payload has no authoritative PL edge")
        refined, initial_energy, final_energy, rms_force = _ordinary_minimize(
            atom_array,
            platform_name=str(config.get("platform", "CUDA")).upper(),
            payload=payload,
            charge_method=charge_method,
        )
    if not math.isfinite(final_energy):
        raise OpenMMPostprocessError("postprocess energy summary is non-finite")
    # Refinement updates only coordinates in the accepted public CIF. Re-running
    # the design serializer could allocate fresh LIGxx identifiers for context
    # molecules or ions whose private source identities are no longer in memory.
    from .peptide import write_peptide_coordinates
    with _sample_postprocess_stage("refined coordinate serialization"):
        write_result(refined)
    return initial_energy, final_energy, rms_force


__all__ = [
    "OpenMMPostprocessError",
    "OpenMMSystemError",
    "atom_key",
    "refine_one",
]
