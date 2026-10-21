"""Private AmberTools backend for the single supported CYS--ligand product."""

from __future__ import annotations

import copy
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from biotite import structure as struc
from rdkit import Chem
from rdkit.Chem import AllChem

from ..runner import AtomKey, GaffLigandRecord, SampleParameterizationPayload
from .refine import (
    OpenMMPostprocessError,
    OpenMMSystemError,
    _validate_minimized_state,
    atom_key,
)


_RDKIT_ORDER = {
    "single": Chem.BondType.SINGLE,
    "double": Chem.BondType.DOUBLE,
    "triple": Chem.BondType.TRIPLE,
    "aromatic": Chem.BondType.AROMATIC,
}
_BIOTITE_ORDER = {
    int(struc.BondType.SINGLE): "single",
    int(struc.BondType.DOUBLE): "double",
    int(struc.BondType.TRIPLE): "triple",
    int(struc.BondType.AROMATIC): "aromatic",
    int(struc.BondType.AROMATIC_SINGLE): "aromatic",
    int(struc.BondType.AROMATIC_DOUBLE): "aromatic",
    int(struc.BondType.AROMATIC_TRIPLE): "aromatic",
}
_CHAIN_CODES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_ANTECHAMBER_AM1_BCC_SQM_KEYWORDS = (
    "qm_theory='AM1', grms_tol=0.01, scfconv=1.d-10, ndiis_attempts=700"
)
_SQM_INPUT_ASSIGNMENT = re.compile(
    r"(?i)\b(qm_theory|grms_tol|scfconv|ndiis_attempts)\s*=\s*"
    r"(?:'([^']*)'|\"([^\"]*)\"|([^,\s/]+))"
)
_SQM_OUTPUT_HAMILTONIAN = re.compile(
    r"(?im)(?:^\s*(?:\|\s*)?QMMM:\s*(AM1|PM3)\b|"
    r"Selected\s+Hamiltonian\s*:?\s*(AM1|PM3)\b)"
)


@dataclass(frozen=True)
class _Mol2Atom:
    name: str
    coord: tuple[float, float, float]
    atom_type: str
    charge: float
    element: str
    parent_heavy: int


@dataclass(frozen=True)
class _Patch:
    atoms: tuple[_Mol2Atom, ...]
    bonds: tuple[tuple[int, int, str], ...]
    ligand_frcmod: str
    link_frcmod: str


_CACHE: dict[tuple[Any, ...], _Patch] = {}


def _error(message: str) -> OpenMMPostprocessError:
    return OpenMMPostprocessError(message)


_UNRESOLVED_FRCMOD = re.compile(
    r"(?ix)(?:"
    r"\bATTN\b|\bNaN\b|(?<![A-Za-z])[-+]?Inf(?:inity)?(?![A-Za-z])|"
    r"\b(?:undefined|unknown|missing|placeholder|TBD)\b|"
    r"\bneeds?\s+revision\b|\?{2,}|X{4,}"
    r")"
)
_REQUIRED_FRCMOD_SECTIONS = {"MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON"}


def _validated_frcmod(text: str, *, label: str) -> str:
    """Reject truncated or visibly unresolved private parameter supplements."""

    if not text.strip():
        raise _error(f"{label} FRCMOD is empty")
    sections = {line.strip() for line in text.splitlines()}
    if not _REQUIRED_FRCMOD_SECTIONS.issubset(sections):
        raise _error(f"{label} FRCMOD is missing required sections")
    # Parmchk2's harmless first-line title is often ``Remark line goes here``;
    # unresolved parameter markers, if any, occur in the section body.
    body = "\n".join(text.splitlines()[1:])
    if _UNRESOLVED_FRCMOD.search(body):
        raise _error(f"{label} FRCMOD contains an unresolved parameter placeholder")
    return text


def _element(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise _error("empty element in authoritative heavy topology")
    return text[0].upper() + text[1:].lower()


def _order(raw: int) -> str:
    if int(raw) not in _BIOTITE_ORDER:
        raise _error(f"covalent route requires concrete ordinary bond order, got {raw}")
    return _BIOTITE_ORDER[int(raw)]


def _tool(name: str) -> Path:
    path = Path(sys.executable).resolve().parent / name
    if not path.is_file() or not os.access(path, os.X_OK):
        raise OpenMMSystemError(f"required AmberTools executable is unavailable: {path}")
    return path


def _run(name: str, args: Sequence[str], work: Path) -> None:
    executable = _tool(name)
    env = dict(os.environ)
    env["AMBERHOME"] = str(executable.parent.parent)
    env["PATH"] = os.pathsep.join((str(executable.parent), env.get("PATH", "")))
    result = subprocess.run(
        [str(executable), *args], cwd=work, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    (work / f"{name}.log").write_text(result.stdout or "", encoding="utf-8")
    if result.returncode:
        raise _error(
            f"{name} failed with exit code {result.returncode}; "
            "private tool output was suppressed"
        )


def _antechamber_args(
    input_name: str,
    total_charge: int,
    charge_method: str,
) -> tuple[str, ...]:
    """Select fast Gasteiger or the fully pinned optional AM1-BCC recipe."""

    if charge_method not in {"gasteiger", "am1bcc"}:
        raise ValueError(f"unsupported charge method {charge_method!r}")
    arguments = [
        "-i", input_name,
        "-fi", "sdf",
        "-o", "product.mol2",
        "-fo", "mol2",
        "-c", "gas" if charge_method == "gasteiger" else "bcc",
        "-at", "gaff2",
        "-nc", str(int(total_charge)),
        "-rn", "ADD",
    ]
    if charge_method == "am1bcc":
        arguments.extend(("-ek", _ANTECHAMBER_AM1_BCC_SQM_KEYWORDS))
    arguments.extend(("-pf", "y"))
    return tuple(arguments)


def _validate_antechamber_sqm(work: Path) -> None:
    """Prove that antechamber actually ran the fixed private AM1 recipe."""

    paths = (work / "sqm.in", work / "sqm.out")
    if not all(path.is_file() for path in paths):
        raise _error("antechamber did not produce both private SQM provenance files")
    try:
        sqm_input, sqm_output = (
            path.read_text(encoding="utf-8") for path in paths
        )
    except (OSError, UnicodeError) as exc:
        raise _error("private SQM provenance files could not be read") from exc

    assignments: dict[str, list[str]] = defaultdict(list)
    for match in _SQM_INPUT_ASSIGNMENT.finditer(sqm_input):
        value = next(group for group in match.groups()[1:] if group is not None)
        assignments[match.group(1).lower()].append(value.strip())
    if any(len(assignments[name]) != 1 for name in (
        "qm_theory", "grms_tol", "scfconv", "ndiis_attempts",
    )):
        raise _error("private SQM input does not match the fixed AM1-BCC recipe")

    def _number(name: str) -> Decimal:
        try:
            value = Decimal(
                assignments[name][0].replace("d", "e").replace("D", "e")
            )
            if not value.is_finite():
                raise InvalidOperation
            return value
        except InvalidOperation as exc:
            raise _error(
                "private SQM input does not match the fixed AM1-BCC recipe"
            ) from exc

    if (
        assignments["qm_theory"][0].upper() != "AM1"
        or _number("grms_tol") != Decimal("0.01")
        or _number("scfconv") != Decimal("1e-10")
        or _number("ndiis_attempts") != Decimal("700")
    ):
        raise _error("private SQM input does not match the fixed AM1-BCC recipe")

    reported = {
        model.upper()
        for match in _SQM_OUTPUT_HAMILTONIAN.findall(sqm_output)
        for model in match
        if model
    }
    if reported != {"AM1"}:
        raise _error("private SQM output did not report Hamiltonian AM1")


def _scope(
    atoms: struc.AtomArray, payload: SampleParameterizationPayload,
) -> tuple[list[AtomKey], dict[AtomKey, int], list[tuple[int, int, int]], GaffLigandRecord, int, int]:
    if atoms.bonds is None:
        raise _error("accepted covalent structure lacks an authoritative BondList")
    keys = [atom_key(atoms, row) for row in range(len(atoms))]
    if len(keys) != len(set(keys)):
        raise _error("authoritative heavy-atom keys are not unique")
    if not np.isfinite(np.asarray(atoms.coord)).all():
        raise _error("authoritative heavy coordinates are non-finite")
    row_by_key = {key: row for row, key in enumerate(keys)}
    edges = [(int(i), int(j), int(t)) for i, j, t in atoms.bonds.as_array()]
    link = payload.covalent_link
    if link is None:
        raise _error("cross-component topology has no private covalent-link record")
    records = [record for record in payload.gaff_ligands if record.record_id == link.ligand_record_id]
    if len(records) != 1 or len(payload.gaff_ligands) != 1:
        raise _error("initial covalent route requires exactly one topology-rich CIF ligand")
    record = records[0]
    if link.polymer_key not in row_by_key or link.ligand_key not in row_by_key:
        raise _error("private covalent endpoints are absent from the accepted CIF")
    sg_row, ligand_row = row_by_key[link.polymer_key], row_by_key[link.ligand_key]
    component = np.asarray(atoms.component_uid, dtype=np.int64)
    cross = [(i, j, t) for i, j, t in edges if component[i] != component[j]]
    if len(cross) != 1:
        raise _error("initial covalent route requires exactly one cross-component link")
    i, j, raw = cross[0]
    if {keys[i], keys[j]} != {link.polymer_key, link.ligand_key} or _order(raw) != "single":
        raise _error("authoritative link is not the declared ordinary single bond")
    if link.order != "single" or link.remove_polymer_hydrogen != "HG":
        raise _error("supported product removes HG from one single CYS-SG link")
    if str(atoms.mol_type[sg_row]) != "protein" or str(atoms.mol_type[ligand_row]) != "ligand":
        raise _error("supported link is protein CYS-SG to ligand")
    if str(atoms.res_name[sg_row]) != "CYS" or str(atoms.atom_name[sg_row]) != "SG" or _element(atoms.element[sg_row]) != "S":
        raise _error("polymer endpoint is not standard CYS SG")
    residue = (str(atoms.chain_id[sg_row]), int(atoms.res_id[sg_row]), str(atoms.ins_code[sg_row]))
    residue_rows = [r for r in range(len(atoms)) if (str(atoms.chain_id[r]), int(atoms.res_id[r]), str(atoms.ins_code[r])) == residue]
    if {str(atoms.atom_name[r]) for r in residue_rows} != {"N", "CA", "C", "O", "CB", "SG"}:
        raise _error("CYS endpoint is terminal, modified, or missing heavy atoms")
    names = {str(atoms.atom_name[r]): r for r in residue_rows}
    residue_set = set(residue_rows)
    external_n = any((i == names["N"] and j not in residue_set and str(atoms.atom_name[j]) == "C") or (j == names["N"] and i not in residue_set and str(atoms.atom_name[i]) == "C") for i, j, _ in edges)
    external_c = any((i == names["C"] and j not in residue_set and str(atoms.atom_name[j]) == "N") or (j == names["C"] and i not in residue_set and str(atoms.atom_name[i]) == "N") for i, j, _ in edges)
    if not external_n or not external_c:
        raise _error("CYS SG endpoint is not an internal residue")
    if link.ligand_key not in record.heavy_keys:
        raise _error("linked ligand atom is absent from its private CIF record")
    ligand_rows = [row_by_key.get(key) for key in record.heavy_keys]
    if any(row is None for row in ligand_rows):
        raise _error("private ligand heavy-key mapping is incomplete")
    ligand_rows_i = [int(row) for row in ligand_rows]
    if len({str(atoms.atom_name[r]) for r in ligand_rows_i}) != len(ligand_rows_i):
        raise _error("CIF ligand atom names are not unique")
    if {_element(atoms.element[r]) for r in [ligand_row]} & {"Fe", "Zn", "Mg", "Ca", "Mn", "Co", "Ni", "Cu"}:
        raise _error("metal coordination is outside the covalent route")
    local = {key: index for index, key in enumerate(record.heavy_keys)}
    ligand_set = set(ligand_rows_i)
    authoritative = {
        tuple(sorted((local[keys[i]], local[keys[j]]))): _order(t)
        for i, j, t in edges if i in ligand_set and j in ligand_set
    }
    private = {tuple(sorted((int(i), int(j)))): order for i, j, order in record.bonds}
    if authoritative != private:
        raise _error("private ligand graph is not the complete concrete-order CIF graph")
    if sum(record.formal_charges) != link.product_total_charge:
        raise _error("private ligand charge differs from product integer charge")
    linked_local = local[link.ligand_key]
    if record.hydrogen_counts[linked_local] < link.remove_ligand_hydrogen_count:
        raise _error("product removes more ligand hydrogens than are available")
    return keys, row_by_key, edges, record, sg_row, ligand_row


def _product(record: GaffLigandRecord, linked: int, remove_h: int) -> tuple[Chem.Mol, set[int], dict[int, int]]:
    cap = Chem.MolFromSmiles("CC(=O)N[C@@H](CS)C(=O)NC")
    assert cap is not None
    sulfur = next(atom.GetIdx() for atom in cap.GetAtoms() if atom.GetSymbol() == "S")
    rw = Chem.RWMol()
    for index, (element, charge, count) in enumerate(zip(record.elements, record.formal_charges, record.hydrogen_counts)):
        atom = Chem.Atom(_element(element))
        atom.SetFormalCharge(int(charge)); atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(int(count) - (remove_h if index == linked else 0))
        rw.AddAtom(atom)
    for i, j, order in record.bonds:
        rw.AddBond(int(i), int(j), _RDKIT_ORDER[order])
        if order == "aromatic":
            bond = rw.GetBondBetweenAtoms(int(i), int(j)); bond.SetIsAromatic(True)
            rw.GetAtomWithIdx(int(i)).SetIsAromatic(True); rw.GetAtomWithIdx(int(j)).SetIsAromatic(True)
    ligand = rw.GetMol(); Chem.SanitizeMol(ligand)
    offset = cap.GetNumAtoms()
    combined = Chem.RWMol(Chem.CombineMols(cap, ligand))
    combined.AddBond(sulfur, offset + linked, Chem.BondType.SINGLE)
    product = combined.GetMol(); Chem.SanitizeMol(product)
    product = Chem.AddHs(product)
    if AllChem.EmbedMolecule(product, randomSeed=12648430, useRandomCoords=True) != 0:
        raise _error("could not embed capped local CYS-ligand product")
    AllChem.MMFFOptimizeMolecule(product, maxIters=200)
    ligand_heavy = {offset + i for i in range(record.heavy_keys.__len__())}
    selected = set(ligand_heavy); parent: dict[int, int] = {}
    for atom in product.GetAtoms():
        if atom.GetAtomicNum() == 1:
            heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
            if len(heavy) == 1 and heavy[0] in ligand_heavy:
                selected.add(atom.GetIdx()); parent[atom.GetIdx()] = heavy[0] - offset
    observed = [0] * len(record.heavy_keys)
    for index in parent.values(): observed[index] += 1
    expected = list(record.hydrogen_counts); expected[linked] -= remove_h
    if observed != expected:
        raise _error("capped product does not realize the declared ligand hydrogen state")
    if any(n.GetAtomicNum() == 1 for n in product.GetAtomWithIdx(sulfur).GetNeighbors()):
        raise _error("capped product incorrectly retains CYS HG")
    return product, selected, parent


def _parse_mol2(path: Path, expected: Sequence[str]) -> tuple[list[tuple[str, tuple[float, float, float], str, float]], list[tuple[int, int, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        atom_start = lines.index("@<TRIPOS>ATOM") + 1
        bond_start = lines.index("@<TRIPOS>BOND") + 1
    except ValueError as exc:
        raise _error("AmberTools produced malformed MOL2") from exc
    atoms = []
    for line in lines[atom_start:bond_start - 1]:
        fields = line.split()
        if fields: atoms.append((fields[1], tuple(map(float, fields[2:5])), fields[5], float(fields[8])))
    bonds = []
    for line in lines[bond_start:]:
        if line.startswith("@<TRIPOS>"): break
        fields = line.split()
        if fields: bonds.append((int(fields[1]) - 1, int(fields[2]) - 1, fields[3]))
    if len(atoms) != len(expected):
        raise _error("AmberTools changed capped-product atom count")
    return atoms, bonds


def _write_mol2(path: Path, atoms: Sequence[_Mol2Atom], bonds: Sequence[tuple[int, int, str]], coordinates: Mapping[int, np.ndarray] | None = None) -> None:
    lines = ["@<TRIPOS>MOLECULE", "LIG", f" {len(atoms)} {len(bonds)} 1 0 0", "SMALL", "USER_CHARGES", "", "@<TRIPOS>ATOM"]
    for index, atom in enumerate(atoms):
        xyz = np.asarray(atom.coord if coordinates is None or index not in coordinates else coordinates[index])
        lines.append(f"{index+1:7d} {atom.name:<8s} {xyz[0]:10.4f} {xyz[1]:10.4f} {xyz[2]:10.4f} {atom.atom_type:<8s} 1 LIG {atom.charge:12.6f}")
    lines.append("@<TRIPOS>BOND")
    for serial, (i, j, order) in enumerate(bonds, 1): lines.append(f"{serial:6d} {i+1:5d} {j+1:5d} {order}")
    lines.extend(("@<TRIPOS>SUBSTRUCTURE", "     1 LIG         1 TEMP              0 ****  ****    0 ROOT", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def _cross_frcmod(
    product: Chem.Mol,
    selected_sorted: Sequence[int],
    ligand_atoms: Sequence[_Mol2Atom],
    linked_local: int,
    work: Path,
) -> tuple[str, str]:
    """Transfer only link-spanning GAFF2 terms onto ff19SB CYS atom types."""

    _run("parmchk2", ["-i", "product.mol2", "-f", "mol2", "-o", "product.frcmod", "-s", "2", "-a", "Y", "-w", "Y"], work)
    ligand_frcmod = _validated_frcmod(
        (work / "product.frcmod").read_text(encoding="utf-8"),
        label="ligand supplement",
    )
    (work / "product.leap").write_text(
        "\n".join(("source leaprc.gaff2", "loadAmberParams product.frcmod", "adduct = loadMol2 product.mol2", "saveAmberParm adduct product.prmtop product.inpcrd", "quit", "")),
        encoding="utf-8",
    )
    _run("tleap", ["-f", "product.leap"], work)
    import parmed as pmd

    parm = pmd.load_file(str(work / "product.prmtop"), str(work / "product.inpcrd"))
    if len(parm.atoms) != product.GetNumAtoms():
        raise _error("local product PRMTOP changed atom count")
    ligand_local = {product_index: local for local, product_index in enumerate(selected_sorted)}
    heavy_selected = [i for i in selected_sorted if product.GetAtomWithIdx(i).GetAtomicNum() > 1]
    linked_product = heavy_selected[linked_local]
    sulfur_candidates = [
        neighbor.GetIdx() for neighbor in product.GetAtomWithIdx(linked_product).GetNeighbors()
        if neighbor.GetIdx() not in ligand_local and neighbor.GetSymbol() == "S"
    ]
    if len(sulfur_candidates) != 1:
        raise _error("capped local product has no unique CYS sulfur link")
    sulfur_product = sulfur_candidates[0]
    cap_heavy_types = {
        0: "CT", 1: "C", 2: "O", 3: "N", 4: "XC", 5: "2C",
        6: "S", 7: "C", 8: "O", 9: "N", 10: "CT",
    }
    cap_h_types = {0: "HC", 3: "H", 4: "H1", 5: "H1", 9: "H", 10: "H1"}
    actual_types: list[str] = []
    for index, atom in enumerate(product.GetAtoms()):
        if index in ligand_local:
            actual_types.append(ligand_atoms[ligand_local[index]].atom_type)
        elif atom.GetAtomicNum() > 1:
            if index not in cap_heavy_types:
                raise _error("unexpected capped-CYS heavy atom while transferring parameters")
            actual_types.append(cap_heavy_types[index])
        else:
            parents = [neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() > 1]
            if len(parents) != 1 or parents[0] not in cap_h_types:
                raise _error("unexpected capped-CYS hydrogen while transferring parameters")
            actual_types.append(cap_h_types[parents[0]])

    def term_types(indices: Sequence[int]) -> str:
        return "-".join(actual_types[index] for index in indices)

    bond_rows: dict[str, tuple[float, float]] = {}
    angle_rows: dict[str, tuple[float, float]] = {}
    link_set = {sulfur_product, linked_product}
    for bond in parm.bonds:
        ids = (bond.atom1.idx, bond.atom2.idx)
        if set(ids) == link_set:
            if bond.type is None or not _nonzero((bond.type.k, bond.type.req)):
                raise _error("local GAFF2 link bond is missing, zero, or non-finite")
            bond_rows[term_types(ids)] = (float(bond.type.k), float(bond.type.req))
    for angle in parm.angles:
        ids = (angle.atom1.idx, angle.atom2.idx, angle.atom3.idx)
        if link_set.issubset(ids):
            if angle.type is None or not _nonzero((angle.type.k, angle.type.theteq)):
                raise _error("local GAFF2 link angle is missing, zero, or non-finite")
            angle_rows[term_types(ids)] = (float(angle.type.k), float(angle.type.theteq))
    proper_rows, improper_rows = _link_dihedral_rows(
        parm,
        actual_types,
        (sulfur_product, linked_product),
    )
    if len(bond_rows) != 1 or not angle_rows or not proper_rows:
        raise _error("local product did not produce complete link-spanning parameter families")
    lines = ["GAFF2 capped CYS-ligand product parameters", "MASS", "", "BOND"]
    lines.extend(f"{name:<14s} {k:10.4f} {req:10.4f}" for name, (k, req) in bond_rows.items())
    lines.append(""); lines.append("ANGLE")
    lines.extend(f"{name:<17s} {k:10.4f} {theta:10.4f}" for name, (k, theta) in angle_rows.items())
    lines.append(""); lines.append("DIHE")
    lines.extend(f"{name:<23s} 1 {pk:10.5f} {phase:10.4f} {periodicity:8.3f}" for name, pk, phase, periodicity in proper_rows)
    lines.append(""); lines.append("IMPROPER")
    lines.extend(f"{name:<23s} {pk:10.5f} {phase:10.4f} {periodicity:8.3f}" for name, pk, phase, periodicity in improper_rows)
    lines.extend(("", "NONBON", ""))
    link_frcmod = _validated_frcmod(
        "\n".join(lines),
        label="link override",
    )
    return ligand_frcmod, link_frcmod


def _compile_patch(
    record: GaffLigandRecord,
    payload: SampleParameterizationPayload,
    work: Path,
    *,
    charge_method: str,
) -> _Patch:
    link = payload.covalent_link; assert link is not None
    linked = record.heavy_keys.index(link.ligand_key)
    key = (charge_method, tuple(link.chemistry_signature), link.product_total_charge, link.remove_ligand_hydrogen_count, record.elements, record.formal_charges, record.bonds, record.hydrogen_counts, linked)
    if key in _CACHE: return _CACHE[key]
    molecule, selected, parent = _product(record, linked, link.remove_ligand_hydrogen_count)
    sdf = work / "product.sdf"; writer = Chem.SDWriter(str(sdf)); writer.write(molecule); writer.close()
    _run(
        "antechamber",
        _antechamber_args(sdf.name, link.product_total_charge, charge_method),
        work,
    )
    if charge_method == "am1bcc":
        _validate_antechamber_sqm(work)
    elif any((work / name).exists() for name in ("sqm.in", "sqm.out")):
        raise _error("Gasteiger parameterization unexpectedly produced SQM files")
    parsed_atoms, parsed_bonds = _parse_mol2(work / "product.mol2", [a.GetSymbol() for a in molecule.GetAtoms()])
    if charge_method == "gasteiger":
        serialized_charges = np.asarray(
            [atom[3] for atom in parsed_atoms], dtype=np.float64
        )
        serialized_charge = float(serialized_charges.sum())
        if not np.isfinite(serialized_charges).all() or not math.isclose(
            serialized_charge,
            float(link.product_total_charge),
            rel_tol=0.0,
            abs_tol=5.0e-4,
        ):
            raise _error(
                "Gasteiger MOL2 charge does not conserve the declared integer charge"
            )
    selected_sorted = sorted(selected); product_to_local = {value: index for index, value in enumerate(selected_sorted)}
    ligand_atoms = []
    hydrogen_serial = 1
    heavy_indices = [i for i in selected_sorted if molecule.GetAtomWithIdx(i).GetAtomicNum() > 1]
    heavy_pos = {value: index for index, value in enumerate(heavy_indices)}
    for product_index in selected_sorted:
        atom = molecule.GetAtomWithIdx(product_index); parsed = parsed_atoms[product_index]
        if atom.GetAtomicNum() > 1:
            p = heavy_pos[product_index]; name = record.heavy_keys[p][-1]
        else:
            p = parent[product_index]; name = f"H{hydrogen_serial}"; hydrogen_serial += 1
        ligand_atoms.append(_Mol2Atom(name, parsed[1], parsed[2], parsed[3], atom.GetSymbol(), p))
    ligand_bonds = tuple((product_to_local[i], product_to_local[j], order) for i, j, order in parsed_bonds if i in product_to_local and j in product_to_local)
    _write_mol2(work / "ligand.mol2", ligand_atoms, ligand_bonds)
    ligand_frcmod, link_frcmod = _cross_frcmod(
        molecule,
        selected_sorted,
        ligand_atoms,
        linked,
        work,
    )
    patch = _Patch(
        tuple(ligand_atoms),
        ligand_bonds,
        ligand_frcmod,
        link_frcmod,
    ); _CACHE[key] = patch
    return patch


def _write_scaffold(atoms: struc.AtomArray, ligand_rows: set[int], sg_row: int, path: Path) -> tuple[int, int]:
    chains = list(dict.fromkeys(map(str, atoms.chain_id)))
    if len(chains) > len(_CHAIN_CODES): raise _error("too many chains for private Amber mapping")
    chain_map = dict(zip(chains, _CHAIN_CODES))
    residues: list[tuple[str, int, str, str]] = []; rows: dict[tuple[str, int, str, str], list[int]] = defaultdict(list)
    for row in range(len(atoms)):
        if row in ligand_rows: continue
        residue = (str(atoms.chain_id[row]), int(atoms.res_id[row]), str(atoms.ins_code[row]), str(atoms.res_name[row]))
        if residue not in rows: residues.append(residue)
        rows[residue].append(row)
    sg_res = (str(atoms.chain_id[sg_row]), int(atoms.res_id[sg_row]), str(atoms.ins_code[sg_row]), str(atoms.res_name[sg_row]))
    lines = []; serial = 1; prior = None
    for residue_index, residue in enumerate(residues, start=1):
        chain, _auth_resid, _ins, resname = residue
        if prior is not None and prior != chain: lines.append("TER")
        outname = "CYX" if residue == sg_res else resname
        for row in rows[residue]:
            x, y, z = atoms.coord[row]; name = str(atoms.atom_name[row]); element = _element(atoms.element[row])
            lines.append(f"ATOM  {serial:5d} {name:>4s} {outname:>3s} {chain_map[chain]}{residue_index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}")
            serial += 1
        prior = chain
    lines.extend(("TER", "END", "")); path.write_text("\n".join(lines), encoding="utf-8")
    return residues.index(sg_res) + 1, len(residues)


def _map_and_audit(atoms: struc.AtomArray, keys: Sequence[AtomKey], edges: Sequence[tuple[int, int, int]], sg_row: int, ligand_row: int, expected_total_charge: int, prmtop_path: Path, inpcrd_path: Path) -> tuple[Any, Any, dict[int, AtomKey], tuple[int, int], Any]:
    import parmed as pmd
    from openmm import unit
    from openmm.app import AmberInpcrdFile, AmberPrmtopFile
    parm = pmd.load_file(str(prmtop_path), str(inpcrd_path)); prmtop = AmberPrmtopFile(str(prmtop_path)); inpcrd = AmberInpcrdFile(str(inpcrd_path))
    observed_total_charge = float(sum(float(atom.charge) for atom in parm.atoms))
    if not math.isclose(observed_total_charge, float(expected_total_charge), rel_tol=0.0, abs_tol=5e-4):
        raise _error(
            f"private PRMTOP charge is not the declared integer product charge: "
            f"{observed_total_charge:.6f} != {expected_total_charge}"
        )
    positions = np.asarray(inpcrd.positions.value_in_unit(unit.angstrom)); heavy = [a.idx for a in parm.atoms if float(a.mass) > 2.0]
    if len(heavy) != len(keys):
        raise _error(f"PRMTOP changed original-heavy atom count: {len(keys)} -> {len(heavy)}")
    candidates = []
    expected_mass = {"C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974, "S": 32.06, "F": 18.998, "Cl": 35.45, "Br": 79.904, "I": 126.90}
    for row in range(len(atoms)):
        for particle in heavy:
            symbol = _element(atoms.element[row])
            if symbol in expected_mass and abs(float(parm.atoms[particle].mass) - expected_mass[symbol]) < 0.8:
                distance = float(np.linalg.norm(positions[particle] - atoms.coord[row]))
                if distance <= 0.02: candidates.append((distance, row, particle))
    assigned_rows = set(); assigned_particles = set(); particle_to_key = {}
    for _, row, particle in sorted(candidates):
        if row not in assigned_rows and particle not in assigned_particles:
            assigned_rows.add(row); assigned_particles.add(particle); particle_to_key[particle] = keys[row]
    if assigned_rows != set(range(len(keys))) or assigned_particles != set(heavy): raise _error("PRMTOP heavy map is not bijective")
    expected = {tuple(sorted((keys[i], keys[j]))) for i, j, t in edges if t != int(struc.BondType.COORDINATION)}
    if any(t == int(struc.BondType.COORDINATION) for _, _, t in edges): raise _error("metal coordination is outside the covalent route")
    actual = {tuple(sorted((particle_to_key[b.atom1.idx], particle_to_key[b.atom2.idx]))) for b in parm.bonds if b.atom1.idx in particle_to_key and b.atom2.idx in particle_to_key}
    if actual != expected: raise _error("PRMTOP heavy-edge set differs from authoritative BondList")
    reverse = {key: particle for particle, key in particle_to_key.items()}; link = (reverse[keys[sg_row]], reverse[keys[ligand_row]])
    _audit_terms(parm, link)
    return prmtop, inpcrd, particle_to_key, link, parm


def _nonzero(values: Iterable[float]) -> bool:
    values = [float(value) for value in values]
    return bool(values) and all(math.isfinite(value) for value in values) and any(abs(value) > 0 for value in values)


def _valid_dihedral_type(dtype: Any) -> bool:
    phi_k = float(dtype.phi_k)
    periodicity = float(dtype.per)
    phase = float(dtype.phase)
    return (
        all(math.isfinite(value) for value in (phi_k, periodicity, phase))
        and abs(phi_k) > 0.0
        and abs(periodicity) > 0.0
    )


def _bond_graph(parm: Any) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = defaultdict(set)
    for bond in parm.bonds:
        i, j = int(bond.atom1.idx), int(bond.atom2.idx)
        graph[i].add(j)
        graph[j].add(i)
    return graph


def _expected_link_impropers(
    graph: Mapping[int, set[int]],
    link: tuple[int, int],
) -> set[tuple[int, frozenset[int]]]:
    """Enumerate topology-defined three-neighbour centers spanning the link."""

    expected: set[tuple[int, frozenset[int]]] = set()
    for center, other in (link, tuple(reversed(link))):
        neighbors = graph.get(center, set())
        if other in neighbors and len(neighbors) == 3:
            expected.add((center, frozenset((center, *neighbors))))
    return expected


def _canonical_central_link_proper(
    ids: tuple[int, int, int, int],
    link: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Orient a proper whose central bond is the link; reject edge-only overlap."""

    if ids[1:3] == link:
        return ids
    if ids[1:3] == tuple(reversed(link)):
        return tuple(reversed(ids))
    return None


def _link_dihedral_rows(
    parm: Any,
    atom_types: Sequence[str],
    link: tuple[int, int],
) -> tuple[
    list[tuple[str, float, float, float]],
    list[tuple[str, float, float, float]],
]:
    """Select only topology-expected central propers and three-neighbour impropers."""

    graph = _bond_graph(parm)
    expected_impropers = _expected_link_impropers(graph, link)
    proper_rows: list[tuple[str, float, float, float]] = []
    improper_rows: list[tuple[str, float, float, float]] = []
    for dihedral in parm.dihedrals:
        ids = (
            int(dihedral.atom1.idx),
            int(dihedral.atom2.idx),
            int(dihedral.atom3.idx),
            int(dihedral.atom4.idx),
        )
        if bool(dihedral.improper):
            id_set = set(ids)
            memberships = {
                (center, frozenset(id_set))
                for center in link
                if id_set.difference({center}).issubset(graph.get(center, set()))
            }
            if expected_impropers.isdisjoint(memberships):
                continue
            rows = improper_rows
        else:
            if _canonical_central_link_proper(ids, link) is None:
                continue
            rows = proper_rows
        dtype = dihedral.type
        if dtype is None or not _valid_dihedral_type(dtype):
            raise _error(
                "local GAFF2 link proper/improper is missing, zero, or non-finite"
            )
        rows.append(
            (
                "-".join(atom_types[index] for index in ids),
                float(dtype.phi_k),
                float(dtype.phase),
                float(dtype.per),
            )
        )
    _audit_link_impropers(parm, link)
    return proper_rows, improper_rows


def _audit_link_impropers(
    parm: Any,
    link: tuple[int, int],
) -> set[tuple[int, frozenset[int]]]:
    """Require every topology-expected link improper to exist with real parameters."""

    graph = _bond_graph(parm)
    expected = _expected_link_impropers(graph, link)
    observed: set[tuple[int, frozenset[int]]] = set()
    link_set = set(link)
    for dihedral in parm.dihedrals:
        if not bool(dihedral.improper):
            continue
        ids = (
            int(dihedral.atom1.idx),
            int(dihedral.atom2.idx),
            int(dihedral.atom3.idx),
            int(dihedral.atom4.idx),
        )
        id_set = set(ids)
        if not link_set.issubset(id_set):
            continue
        memberships = {
            (center, frozenset(id_set))
            for center in link
            if id_set.difference({center}).issubset(graph.get(center, set()))
        }
        matched = expected.intersection(memberships)
        if not matched:
            continue
        dtype = dihedral.type
        if dtype is None or not _valid_dihedral_type(dtype):
            raise _error("link improper is zero or non-finite")
        observed.update(matched)
    if not expected.issubset(observed):
        raise _error("link-spanning improper coverage is incomplete")
    return expected


def _audit_terms(parm: Any, link: tuple[int, int]) -> None:
    a, b = link; graph = _bond_graph(parm); link_bonds = []
    for bond in parm.bonds:
        i, j = bond.atom1.idx, bond.atom2.idx
        if {i, j} == {a, b}: link_bonds.append(bond)
    if len(link_bonds) != 1 or link_bonds[0].type is None or not _nonzero((link_bonds[0].type.k, link_bonds[0].type.req)): raise _error("link bond parameter is absent, duplicate, zero, or non-finite")
    expected_angles = {frozenset((n, a, b)) for n in graph[a] if n != b} | {frozenset((a, b, n)) for n in graph[b] if n != a}
    observed_angles = set()
    for angle in parm.angles:
        ids = (angle.atom1.idx, angle.atom2.idx, angle.atom3.idx)
        if a in ids and b in ids:
            if angle.type is None or not _nonzero((angle.type.k, angle.type.theteq)): raise _error("link angle parameter is zero or non-finite")
            observed_angles.add(frozenset(ids))
    if not expected_angles.issubset(observed_angles): raise _error("link-spanning angle coverage is incomplete")
    expected_propers = {(left, a, b, right) for left in graph[a] if left != b for right in graph[b] if right != a}; observed = set()
    for dihedral in parm.dihedrals:
        ids = (dihedral.atom1.idx, dihedral.atom2.idx, dihedral.atom3.idx, dihedral.atom4.idx)
        if dihedral.improper:
            continue
        canonical = _canonical_central_link_proper(ids, link)
        if canonical is None:
            continue
        dtype = dihedral.type
        if dtype is None or not _valid_dihedral_type(dtype): raise _error("link proper is zero or non-finite")
        observed.add(canonical)
    if not expected_propers.issubset(observed): raise _error("link-spanning proper coverage is incomplete")
    _audit_link_impropers(parm, link)


def _audit_exclusions(system: Any, parm: Any, link: tuple[int, int]) -> None:
    from openmm import NonbondedForce, unit
    nonbonded = next((system.getForce(i) for i in range(system.getNumForces()) if isinstance(system.getForce(i), NonbondedForce)), None)
    if nonbonded is None: raise _error("Amber System has no NonbondedForce")
    exceptions = {}
    for index in range(nonbonded.getNumExceptions()):
        i, j, q, sigma, epsilon = nonbonded.getExceptionParameters(index); exceptions[tuple(sorted((int(i), int(j))))] = (q, sigma, epsilon)
    graph: dict[int, set[int]] = defaultdict(set)
    for bond in parm.bonds:
        i, j = bond.atom1.idx, bond.atom2.idx; graph[i].add(j); graph[j].add(i)
    a, b = link

    def distances(start: int, blocked: tuple[int, int]) -> dict[int, int]:
        result = {start: 0}; queue = deque([start])
        while queue:
            node = queue.popleft()
            if result[node] == 2: continue
            for neighbor in graph[node]:
                if {node, neighbor} == set(blocked) or neighbor in result: continue
                result[neighbor] = result[node] + 1; queue.append(neighbor)
        return result

    side_a = distances(a, link); side_b = distances(b, link)
    for atom_i, distance_i in side_a.items():
        for atom_j, distance_j in side_b.items():
            distance = distance_i + 1 + distance_j
            if distance > 3: continue
            pair = tuple(sorted((atom_i, atom_j)))
            if pair not in exceptions: raise _error("link-spanning Amber exclusions/scaling are incomplete")
            q, _, epsilon = exceptions[pair]
            if distance <= 2 and (abs(q.value_in_unit(unit.elementary_charge**2)) > 1e-10 or abs(epsilon.value_in_unit(unit.kilojoule_per_mole)) > 1e-10): raise _error("link 1-2/1-3 pair is not excluded")
            if distance == 3:
                qi, sigma_i, epsilon_i = nonbonded.getParticleParameters(atom_i)
                qj, sigma_j, epsilon_j = nonbonded.getParticleParameters(atom_j)
                observed_q = q.value_in_unit(unit.elementary_charge**2)
                expected_q = (
                    qi.value_in_unit(unit.elementary_charge)
                    * qj.value_in_unit(unit.elementary_charge)
                    / 1.2
                )
                observed_sigma = exceptions[pair][1].value_in_unit(unit.nanometer)
                expected_sigma = 0.5 * (
                    sigma_i.value_in_unit(unit.nanometer)
                    + sigma_j.value_in_unit(unit.nanometer)
                )
                observed_epsilon = epsilon.value_in_unit(unit.kilojoule_per_mole)
                expected_epsilon = math.sqrt(
                    epsilon_i.value_in_unit(unit.kilojoule_per_mole)
                    * epsilon_j.value_in_unit(unit.kilojoule_per_mole)
                ) / 2.0
                if not (
                    math.isclose(observed_q, expected_q, rel_tol=1e-6, abs_tol=1e-8)
                    and math.isclose(observed_sigma, expected_sigma, rel_tol=1e-6, abs_tol=1e-8)
                    and math.isclose(observed_epsilon, expected_epsilon, rel_tol=1e-6, abs_tol=1e-8)
                ):
                    raise _error("link 1-4 electrostatic/Lennard-Jones scaling is incorrect")


def _full_leap_text(
    *,
    cys_index: int,
    ligand_residue_index: int,
    ligand_atom_name: str,
) -> str:
    """Load ligand supplements before the hybrid cross-force-field overrides."""

    return "\n".join(
        (
            "source leaprc.protein.ff19SB",
            "source leaprc.DNA.OL21",
            "source leaprc.RNA.OL3",
            "source leaprc.gaff2",
            "loadAmberParams product.frcmod",
            "loadAmberParams link.frcmod",
            "scaffold = loadPdb scaffold.pdb",
            "lig = loadMol2 ligand.mol2",
            "adduct = combine { scaffold lig }",
            f"bond adduct.{cys_index}.SG adduct.{ligand_residue_index}.{ligand_atom_name}",
            "check adduct",
            "saveAmberParm adduct full.prmtop full.inpcrd",
            "quit",
            "",
        )
    )


class CovalentPatchCompiler:
    """Compile one private PRMTOP and prove its authoritative heavy topology."""

    def compile(self, atoms: struc.AtomArray, payload: SampleParameterizationPayload, work: Path, *, charge_method: str) -> tuple[Any, Any, dict[int, AtomKey], tuple[int, int], Any]:
        keys, row_by_key, edges, record, sg_row, ligand_row = _scope(atoms, payload)
        patch = _compile_patch(
            record,
            payload,
            work,
            charge_method=charge_method,
        )
        (work / "product.frcmod").write_text(patch.ligand_frcmod, encoding="utf-8")
        (work / "link.frcmod").write_text(patch.link_frcmod, encoding="utf-8")
        ligand_rows = {row_by_key[key] for key in record.heavy_keys}; cys_index, residue_count = _write_scaffold(atoms, ligand_rows, sg_row, work / "scaffold.pdb")
        (work / "scaffold.leap").write_text("\n".join(("source leaprc.protein.ff19SB", "source leaprc.DNA.OL21", "source leaprc.RNA.OL3", "source leaprc.gaff2", "scaffold = loadPdb scaffold.pdb", "saveAmberParm scaffold scaffold.prmtop scaffold.inpcrd", "quit", "")), encoding="utf-8")
        _run("tleap", ["-f", "scaffold.leap"], work)
        import parmed as pmd
        scaffold_parm = pmd.load_file(str(work / "scaffold.prmtop"), str(work / "scaffold.inpcrd"))
        scaffold_charge = float(sum(float(atom.charge) for atom in scaffold_parm.atoms))
        link = payload.covalent_link; assert link is not None
        expected_total_charge = int(round(scaffold_charge)) + int(link.product_total_charge)
        target_ligand_charge = float(expected_total_charge) - scaffold_charge
        current_ligand_charge = float(sum(atom.charge for atom in patch.atoms))
        correction = target_ligand_charge - current_ligand_charge
        if not math.isfinite(correction) or abs(correction) > 1.0:
            raise _error("reaction-site charge conservation requires an implausible correction")
        # The local product supplies the selected partial charges.  One deterministic correction on
        # the linked ligand atom conserves the whole private PRMTOP integer charge while
        # leaving every non-reaction-site charge unchanged.
        linked_local = record.heavy_keys.index(link.ligand_key)
        charged_atoms = list(patch.atoms)
        linked_atom = charged_atoms[linked_local]
        charged_atoms[linked_local] = _Mol2Atom(
            linked_atom.name, linked_atom.coord, linked_atom.atom_type,
            linked_atom.charge + correction, linked_atom.element, linked_atom.parent_heavy,
        )
        heavy_count = len(record.heavy_keys); coordinates = {}; accepted = [np.asarray(atoms.coord[row_by_key[key]]) for key in record.heavy_keys]
        for index, xyz in enumerate(accepted): coordinates[index] = xyz
        for index in range(heavy_count, len(charged_atoms)):
            parent = charged_atoms[index].parent_heavy; coordinates[index] = accepted[parent] + np.asarray(charged_atoms[index].coord) - np.asarray(charged_atoms[parent].coord)
        _write_mol2(work / "ligand.mol2", charged_atoms, patch.bonds, coordinates)
        (work / "full.leap").write_text(
            _full_leap_text(
                cys_index=cys_index,
                ligand_residue_index=residue_count + 1,
                ligand_atom_name=link.ligand_key[-1],
            ),
            encoding="utf-8",
        )
        _run("tleap", ["-f", "full.leap"], work)
        log = (work / "tleap.log").read_text(encoding="utf-8")
        if any(marker in log for marker in ("FATAL:", "Error!", "does not have a type", "Could not find", "No torsion terms")):
            raise _error("tleap reported an unmatched template or missing parameter")
        if any(
            not path.is_file() or path.stat().st_size == 0
            for path in (work / "full.prmtop", work / "full.inpcrd")
        ):
            raise _error("tleap did not create the full Amber topology")
        return _map_and_audit(atoms, keys, edges, sg_row, ligand_row, expected_total_charge, work / "full.prmtop", work / "full.inpcrd")


def minimize_covalent(atoms: struc.AtomArray, *, platform_name: str, payload: SampleParameterizationPayload, charge_method: str) -> tuple[struc.AtomArray, float, float, float]:
    """Run the fixed restrained-vacuum minimizer for the one supported link."""

    _scope(atoms, payload)
    with tempfile.TemporaryDirectory(prefix="partidesign-covalent-") as temporary:
        prmtop, inpcrd, particle_to_key, link, parm = CovalentPatchCompiler().compile(
            atoms,
            payload,
            Path(temporary),
            charge_method=charge_method,
        )
        from openmm import CustomExternalForce, Platform, VerletIntegrator, unit
        from openmm.app import HBonds, NoCutoff, Simulation
        system = prmtop.createSystem(nonbondedMethod=NoCutoff, constraints=HBonds, rigidWater=True); _audit_exclusions(system, parm, link)
        restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"); restraint.addGlobalParameter("k", 10.0 * unit.kilocalories_per_mole / unit.angstrom**2)
        for name in ("x0", "y0", "z0"): restraint.addPerParticleParameter(name)
        for particle in sorted(particle_to_key): restraint.addParticle(particle, np.asarray(inpcrd.positions[particle].value_in_unit(unit.nanometer)).tolist())
        system.addForce(restraint)
        from .stability import guard_zero_lj_exceptions
        guard_zero_lj_exceptions(system, prmtop.topology)
        try: platform = Platform.getPlatformByName(platform_name)
        except Exception as exc: raise OpenMMSystemError(f"OpenMM platform {platform_name!r} unavailable") from exc
        properties = {"DeviceIndex": "0", "Precision": "mixed", "UseBlockingSync": "true"} if platform_name == "CUDA" else {"Threads": "1"} if platform_name == "CPU" else {}
        simulation = Simulation(prmtop.topology, system, VerletIntegrator(0.001 * unit.picoseconds), platform, properties); simulation.context.setPositions(inpcrd.positions)
        initial = float(simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        if not np.isfinite(initial):
            raise OpenMMPostprocessError("covalent initial energy is non-finite before minimization")
        simulation.minimizeEnergy(tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=500)
        state = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True); final = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)); rms = float(np.sqrt(np.mean(np.sum(forces * forces, axis=-1))))
        positions = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        _validate_minimized_state(initial, final, positions, label="covalent minimization")
        refined = copy.deepcopy(atoms); row_by_key = {atom_key(refined, row): row for row in range(len(refined))}
        if set(particle_to_key.values()) != set(row_by_key): raise _error("heavy-coordinate map-back key set is incomplete")
        for particle, key in particle_to_key.items(): refined.coord[row_by_key[key]] = positions[particle] * 10.0
        return refined, initial, final, rms


__all__ = ["CovalentPatchCompiler", "minimize_covalent"]
