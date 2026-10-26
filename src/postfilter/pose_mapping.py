"""Exact public identities and coordinates across a private protein Pose bridge."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import string
import tempfile
from typing import Any

import numpy as np
from biotite.structure.io import pdbx

from src.design.utils.atoms import PROTEIN_RES_ATOMS
from src.design.utils.reader import _read_pdb, read_structure
from src.design.utils.source_mapping import COMPLETE_BACKBONE_CRITERIA

from .topology import head_tail_chains


class MappingError(ValueError):
    """A structure cannot be represented without changing its chemistry."""


_AA = dict(zip(
    ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
     "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"),
    "ARNDCQEGHILKMFPSTWYV", strict=True,
))
_MISSING = {"", ".", "?"}
_ALIASES = string.ascii_uppercase + string.ascii_lowercase + string.digits
_COORD_COLUMNS = {"Cartn_x", "Cartn_y", "Cartn_z"}


def _column(category: Any, name: str, default: str = "") -> list[str]:
    if name not in category:
        return [default] * category.row_count
    return category[name].as_array(str).tolist()


def _clean(value: str) -> str:
    return "" if value in _MISSING else value


def _pdb_document(path: Path) -> Any:
    """Create a faithful CIF view; PDB has no source CIF categories to retain."""
    lines = path.read_text().splitlines()
    if any(line.startswith("LINK  ") for line in lines):
        raise MappingError("PDB LINK chemistry requires an explicit source CIF mapping")
    model_ids = [line[10:14].strip() or "1" for line in lines if line.startswith("MODEL ")]
    if len(model_ids) > 1:
        raise MappingError("multiple source PDB models are unsupported")
    rows = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    if any(line[16:17].strip() for line in rows):
        raise MappingError("source alternate conformations are unsupported")
    if any(line[17:20].strip().upper() not in _AA for line in rows):
        raise MappingError("native postfilter supports only standard amino acids")
    if any(line[76:78].strip().upper() in {"H", "D", "T"}
           or line[12:16].strip().lstrip("123").startswith("H") for line in rows):
        raise MappingError("explicit source hydrogens are unsupported")
    _parsed, _declared, serial_to_atom = _read_pdb(path)
    atoms = read_structure(path, canonical_polymer_order=True)
    if len(atoms) != len(rows) or len(serial_to_atom) != len(rows):
        raise MappingError("PD reader would discard or merge source PDB atoms")
    seen_residues, previous_locators = set(), {}
    for index in range(len(atoms)):
        chain = str(atoms.label_asym_id[index])
        occurrence = (chain, int(atoms.residue_uid[index]))
        if occurrence in seen_residues:
            continue
        seen_residues.add(occurrence)
        try:
            locator = (int(atoms.auth_seq_id[index]), _clean(str(atoms.auth_ins_code[index])))
        except (TypeError, ValueError) as exc:
            raise MappingError("PDB residue order requires integral author numbering; use explicit CIF labels") from exc
        if chain in previous_locators and locator <= previous_locators[chain]:
            raise MappingError("PDB residue order is ambiguous after author numbering; use explicit CIF labels")
        previous_locators[chain] = locator
    by_serial = {int(line[6:11]): line for line in rows}
    serials = {index: serial for serial, index in serial_to_atom.items()}
    sequences: dict[tuple[str, int], int] = {}
    next_sequence: dict[str, int] = {}
    columns: dict[str, list[str]] = {name: [] for name in (
        "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
        "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
        "auth_atom_id", "auth_comp_id", "auth_asym_id", "auth_seq_id",
        "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
        "B_iso_or_equiv", "pdbx_PDB_model_num", "pdbx_formal_charge",
    )}
    for index in range(len(atoms)):
        line = by_serial[serials[index]]
        chain = str(atoms.label_asym_id[index])
        residue_key = (chain, int(atoms.residue_uid[index]))
        if residue_key not in sequences:
            next_sequence[chain] = next_sequence.get(chain, 0) + 1
            sequences[residue_key] = next_sequence[chain]
        values = {
            "group_PDB": line[:6].strip(), "id": str(serials[index]),
            "type_symbol": str(atoms.element[index]), "label_atom_id": str(atoms.atom_name[index]),
            "label_alt_id": ".", "label_comp_id": str(atoms.res_name[index]),
            "label_asym_id": chain, "label_entity_id": str(atoms.label_entity_id[index]),
            "label_seq_id": str(sequences[residue_key]),
            "auth_atom_id": str(atoms.atom_name[index]), "auth_comp_id": str(atoms.res_name[index]),
            "auth_asym_id": str(atoms.auth_asym_id[index]), "auth_seq_id": str(atoms.auth_seq_id[index]),
            "pdbx_PDB_ins_code": str(atoms.auth_ins_code[index]) or "?",
            "Cartn_x": line[30:38].strip(), "Cartn_y": line[38:46].strip(),
            "Cartn_z": line[46:54].strip(), "occupancy": line[54:60].strip() or "1",
            "B_iso_or_equiv": line[60:66].strip() or "0", "pdbx_PDB_model_num": model_ids[0] if model_ids else "1",
            "pdbx_formal_charge": "." if atoms.formal_charge[index] is None else str(atoms.formal_charge[index]),
        }
        for name in columns:
            columns[name].append(values[name])
    bonds = {(min(int(a), int(b)), max(int(a), int(b))) for a, b, _ in atoms.bonds.as_array()
             if int(atoms.residue_uid[a]) != int(atoms.residue_uid[b])}
    for line in lines:
        if not line.startswith("SSBOND"):
            continue
        if any(line[start:stop].strip() not in {"", "1555", "1_555"} for start, stop in ((59, 65), (66, 72))):
            raise MappingError("symmetry-related PDB disulfides are unsupported")
        endpoints = []
        for chain, seq, ins in ((line[15:16], line[17:21], line[21:22]),
                                (line[29:30], line[31:35], line[35:36])):
            matches = [i for i in range(len(atoms)) if str(atoms.auth_asym_id[i]) == chain.strip()
                       and str(atoms.auth_seq_id[i]) == seq.strip()
                       and str(atoms.auth_ins_code[i]) == ins.strip()
                       and str(atoms.res_name[i]) == "CYS" and str(atoms.atom_name[i]) == "SG"]
            if len(matches) != 1:
                raise MappingError("PDB SSBOND endpoint is absent or ambiguous")
            endpoints.append(matches[0])
        bonds.add(tuple(sorted(endpoints)))
    cif = pdbx.CIFFile()
    block = pdbx.CIFBlock()
    cif["postfilter_source"] = block
    block["atom_site"] = pdbx.CIFCategory(columns)
    block["struct_conn_type"] = pdbx.CIFCategory({"id": ["covale"],
        "criteria": [COMPLETE_BACKBONE_CRITERIA]})
    connections: dict[str, list[str]] = {name: [] for name in ("id", "conn_type_id", "pdbx_value_order")}
    for prefix in ("ptnr1", "ptnr2"):
        for name in ("label_asym_id", "label_seq_id", "label_comp_id", "label_atom_id"):
            connections[f"{prefix}_{name}"] = []
    for ordinal, (left, right) in enumerate(sorted(bonds), 1):
        connections["id"].append(f"link{ordinal}")
        names = {str(atoms.atom_name[i]) for i in (left, right)}
        connections["conn_type_id"].append("disulf" if names == {"SG"} else "covale")
        connections["pdbx_value_order"].append("SING")
        for prefix, index in (("ptnr1", left), ("ptnr2", right)):
            for name in ("label_asym_id", "label_seq_id", "label_comp_id", "label_atom_id"):
                connections[f"{prefix}_{name}"].append(columns[name][index])
    if bonds:
        block["struct_conn"] = pdbx.CIFCategory(connections)
    return cif


def _source_document(path: Path) -> Any:
    if path.suffix.lower() == ".pdb":
        return _pdb_document(path)
    if path.suffix.lower() != ".cif":
        raise MappingError("native postfilter accepts source PDB or CIF only")
    return pdbx.CIFFile.read(path)


@dataclass(frozen=True)
class ResidueOccurrence:
    ordinal: int
    chain: str
    label_seq: str
    auth_chain: str
    auth_seq: str
    insertion: str
    name: str
    atom_indices: tuple[int, ...]
    segment: int
    private_chain: str
    private_number: int


@dataclass
class SourceStructure:
    path: Path
    cif: Any
    block_name: str
    atoms: Any
    coordinates: np.ndarray
    source_rows: tuple[int, ...]
    residues: tuple[ResidueOccurrence, ...]
    atom_residues: tuple[int, ...]
    bonds: tuple[tuple[int, int, int], ...]
    disulfides: tuple[tuple[int, int], ...]
    binder_chains: tuple[str, ...]
    target_chains: tuple[str, ...]
    head_tail_pairs: tuple[tuple[int, int], ...] = ()

    @property
    def binder_indices(self) -> tuple[int, ...]:
        return tuple(r.ordinal for r in self.residues if r.chain in self.binder_chains)

    @property
    def target_indices(self) -> tuple[int, ...]:
        return tuple(r.ordinal for r in self.residues if r.chain in self.target_chains)

    @property
    def partners(self) -> str:
        target = "".join(dict.fromkeys(r.private_chain for r in self.residues if r.chain in self.target_chains))
        binder = "".join(dict.fromkeys(r.private_chain for r in self.residues if r.chain in self.binder_chains))
        return f"{target}_{binder}"

    @property
    def sequences(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for residue in self.residues:
            result[residue.chain] = result.get(residue.chain, "") + _AA[residue.name]
        return result

    def write_private_pdb(self, path: Path) -> None:
        serial = 0
        previous_segment = None
        lines: list[str] = []
        for residue in self.residues:
            if previous_segment is not None and residue.segment != previous_segment:
                lines.append("TER")
            previous_segment = residue.segment
            for index in residue.atom_indices:
                serial += 1
                if serial > 99999:
                    raise MappingError("private PDB bridge exceeds atom serial capacity")
                name, element = str(self.atoms.atom_name[index]), str(self.atoms.element[index])
                x, y, z = self.coordinates[index]
                if any(len(f"{value:8.3f}") > 8 for value in (x, y, z)):
                    raise MappingError("coordinate exceeds private PDB bridge capacity")
                atom_field = f" {name:<3}" if len(name) < 4 else name
                lines.append(f"ATOM  {serial:5d} {atom_field} {residue.name:>3} "
                    f"{residue.private_chain}{residue.private_number:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2}  ")
        lines.extend(("TER", "END"))
        Path(path).write_text("\n".join(lines) + "\n")


def read_source(path: str | Path, binder_chains: tuple[str, ...],
                target_chains: tuple[str, ...]) -> SourceStructure:
    """Read the PD heavy graph and source float64 coordinates without omissions."""
    path = Path(path).resolve()
    if path.suffix.lower() == ".pdb":
        document = _pdb_document(path)
        with tempfile.TemporaryDirectory(prefix="partidesign-postfilter-pdb-") as directory:
            converted = Path(directory) / "source.cif"
            document.write(converted)
            result = read_source(converted, binder_chains, target_chains)
        result.path = path
        return result
    cif = _source_document(path)
    if len(cif) != 1:
        raise MappingError("native postfilter requires exactly one CIF data block")
    block_name = next(iter(cif))
    block = cif[block_name]
    if "atom_site" not in block or not block["atom_site"].row_count:
        raise MappingError("source CIF has no atoms")
    if "struct_conn" in block:
        connections = block["struct_conn"]
        kinds = _column(connections, "conn_type_id")
        for row, kind in enumerate(kinds):
            if kind.lower() in {"covale", "disulf", "modres"}:
                for name in ("ptnr1_symmetry", "ptnr2_symmetry"):
                    if _column(connections, name)[row] not in {"", ".", "?", "1555", "1_555"}:
                        raise MappingError("symmetry-related covalent source bonds are unsupported")
    category = block["atom_site"]
    columns = {name: _column(category, name, default) for name, default in {
        "id": "", "label_asym_id": "", "label_seq_id": ".", "auth_asym_id": "",
        "auth_seq_id": ".", "pdbx_PDB_ins_code": "", "label_comp_id": "",
        "label_atom_id": "", "type_symbol": "", "label_alt_id": ".",
        "pdbx_PDB_model_num": "1", "Cartn_x": "", "Cartn_y": "", "Cartn_z": "",
    }.items()}
    if len(set(columns["pdbx_PDB_model_num"])) != 1:
        raise MappingError("multiple source models are unsupported")
    if any(value not in _MISSING for value in columns["label_alt_id"]):
        raise MappingError("source alternate conformations are unsupported")
    if any(value.upper() not in _AA for value in columns["label_comp_id"]):
        raise MappingError("native postfilter supports only the twenty standard amino acids")
    if any(value.upper() in {"H", "D", "T"} for value in columns["type_symbol"]):
        raise MappingError("explicit source hydrogens require an unsupported public atom mapping")
    ids = columns["id"]
    if any(value in _MISSING for value in ids) or len(set(ids)) != len(ids):
        raise MappingError("source atom_site.id must uniquely identify every atom")
    rows_by_key: dict[tuple[str, ...], int] = {}
    for row in range(category.row_count):
        chain = columns["label_asym_id"][row]
        auth_chain = columns["auth_asym_id"][row]
        key = (chain, columns["label_seq_id"][row],
               chain if auth_chain in _MISSING else auth_chain,
               columns["auth_seq_id"][row], _clean(columns["pdbx_PDB_ins_code"][row]),
               columns["label_comp_id"][row].upper(), columns["label_atom_id"][row],
               columns["type_symbol"][row].upper())
        if key in rows_by_key:
            raise MappingError("source contains ambiguous duplicate atom occurrences")
        rows_by_key[key] = row
    atoms = read_structure(path, canonical_polymer_order=True)
    if len(atoms) != category.row_count:
        raise MappingError("PD reader would discard source atoms")
    if any(str(value) != "protein" for value in atoms.mol_type):
        raise MappingError("native postfilter supports protein polymer atoms only")
    source_rows: list[int] = []
    for index in range(len(atoms)):
        key = (str(atoms.label_asym_id[index]), str(atoms.label_seq_id[index]),
               str(atoms.auth_asym_id[index]), str(atoms.auth_seq_id[index]),
               str(atoms.auth_ins_code[index]), str(atoms.res_name[index]),
               str(atoms.atom_name[index]), str(atoms.element[index]))
        try:
            source_rows.append(rows_by_key[key])
        except KeyError as exc:
            raise MappingError("PD reader changed a public atom identity") from exc
    if len(set(source_rows)) != category.row_count:
        raise MappingError("source atom mapping is not one-to-one")
    coordinates = np.asarray([[float(columns[axis][row]) for axis in
                               ("Cartn_x", "Cartn_y", "Cartn_z")]
                              for row in source_rows], dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise MappingError("source coordinates must be finite")
    binder_chains, target_chains = tuple(binder_chains), tuple(target_chains)
    binder, target = set(binder_chains), set(target_chains)
    chains = set(str(value) for value in atoms.label_asym_id)
    if (not binder or not target or binder & target or binder | target != chains
            or len(binder) != len(binder_chains) or len(target) != len(target_chains)):
        raise MappingError("explicit binder and target sets must partition source chains")
    groups: dict[int, list[int]] = {}
    for index, uid in enumerate(atoms.residue_uid):
        groups.setdefault(int(uid), []).append(index)
    atom_residues = [0] * len(atoms)
    residue_rows = list(groups.values())
    for ordinal, indices in enumerate(residue_rows, 1):
        for index in indices:
            atom_residues[index] = ordinal
        name = str(atoms.res_name[indices[0]])
        expected = set(PROTEIN_RES_ATOMS[name])
        actual = [str(atoms.atom_name[i]) for i in indices]
        if len(set(actual)) != len(actual) or set(actual) - expected - {"OXT"} or expected - set(actual):
            raise MappingError(f"incomplete or unsupported heavy atoms at residue occurrence {ordinal}: "
                f"missing={sorted(expected - set(actual))}, extra={sorted(set(actual) - expected - {'OXT'})}")
        for index in indices:
            if str(atoms.element[index]) != str(atoms.atom_name[index])[0]:
                raise MappingError(f"element differs from standard atom {atoms.atom_name[index]}")
    bonds = tuple(sorted((min(int(a), int(b)), max(int(a), int(b)), int(order))
                         for a, b, order in atoms.bonds.as_array()))
    try:
        closed_chains = set(head_tail_chains({"annotations": {
            "label_asym_id": atoms.label_asym_id.tolist(),
            "residue_uid": atoms.residue_uid.tolist(),
            "label_seq_id": atoms.label_seq_id.tolist(),
            "atom_name": atoms.atom_name.tolist(), "res_name": atoms.res_name.tolist(),
        }, "bonds": bonds}))
    except (KeyError, TypeError, ValueError) as exc:
        raise MappingError(f"unsupported source peptide topology: {exc}") from exc
    chain_ordinals: dict[str, list[int]] = {}
    for ordinal, indices in enumerate(residue_rows, 1):
        chain_ordinals.setdefault(str(atoms.label_asym_id[indices[0]]), []).append(ordinal)
    head_tail_pairs = tuple((indices[0], indices[-1]) for chain, indices in chain_ordinals.items()
                            if chain in closed_chains)
    closure_edges = {(last, first) for first, last in head_tail_pairs}
    polymer_links: set[tuple[int, int]] = set()
    disulfides: list[tuple[int, int]] = []
    for a, b, _order in bonds:
        ra, rb = atom_residues[a], atom_residues[b]
        if ra == rb:
            continue
        names = {str(atoms.atom_name[a]), str(atoms.atom_name[b])}
        ca, cb = str(atoms.label_asym_id[a]), str(atoms.label_asym_id[b])
        if (ca in binder) != (cb in binder):
            raise MappingError("covalent bonds across binder and target are unsupported")
        if names == {"C", "N"} and ca == cb:
            carbon, nitrogen = (ra, rb) if str(atoms.atom_name[a]) == "C" else (rb, ra)
            if _order != 1:
                raise MappingError("inter-residue peptide bonds must be single bonds")
            if (carbon, nitrogen) in closure_edges:
                continue
            if nitrogen != carbon + 1:
                raise MappingError("unsupported inter-residue C-N chemistry")
            polymer_links.add((carbon, nitrogen))
        elif names == {"SG"} and str(atoms.res_name[a]) == str(atoms.res_name[b]) == "CYS":
            disulfides.append((min(ra, rb), max(ra, rb)))
        else:
            raise MappingError("unsupported inter-residue covalent chemistry")
    residues: list[ResidueOccurrence] = []
    segment, private_number = -1, 0
    previous_chain = None
    seen_chains: set[str] = set()
    for ordinal, indices in enumerate(residue_rows, 1):
        first = indices[0]
        chain = str(atoms.label_asym_id[first])
        new_segment = chain != previous_chain or (ordinal - 1, ordinal) not in polymer_links
        if chain != previous_chain and chain in seen_chains:
            raise MappingError("source chain occurrences are interleaved ambiguously")
        seen_chains.add(chain)
        if new_segment:
            segment += 1
            private_number = 0
        if segment >= len(_ALIASES):
            raise MappingError("too many connected segments for the private PDB bridge")
        private_number += 1
        if private_number > 9999:
            raise MappingError("connected segment exceeds private residue capacity")
        residues.append(ResidueOccurrence(ordinal, chain, str(atoms.label_seq_id[first]),
            str(atoms.auth_asym_id[first]), str(atoms.auth_seq_id[first]),
            str(atoms.auth_ins_code[first]), str(atoms.res_name[first]), tuple(indices),
            segment, _ALIASES[segment], private_number))
        previous_chain = chain
    return SourceStructure(path, cif, block_name, atoms, coordinates, tuple(source_rows),
        tuple(residues), tuple(atom_residues), bonds, tuple(sorted(set(disulfides))),
        binder_chains, target_chains, head_tail_pairs)


@dataclass
class PoseMapping:
    # (atom number, residue number), indexed in exact source heavy-atom order.
    atom_ids: tuple[tuple[int, int], ...]
    pose_residues: tuple[int, ...]
    separation_jump: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _pose_edges(pose: Any, mapping: PoseMapping) -> set[tuple[int, int]]:
    reverse = {value: index for index, value in enumerate(mapping.atom_ids)}
    edges: set[tuple[int, int]] = set()
    for (atom_number, residue_number), index in reverse.items():
        residue = pose.residue(residue_number)
        for neighbor in residue.bonded_neighbor(atom_number):
            other = reverse.get((int(neighbor), residue_number))
            if other is not None:
                edges.add(tuple(sorted((index, other))))
        for connection in range(1, residue.n_possible_residue_connections() + 1):
            if int(residue.residue_connect_atom_index(connection)) != atom_number:
                continue
            partner = int(residue.residue_connection_partner(connection))
            if not partner:
                continue
            other_connection = int(residue.residue_connection_conn_id(connection))
            other_atom = int(pose.residue(partner).residue_connect_atom_index(other_connection))
            other = reverse.get((other_atom, partner))
            if other is not None:
                edges.add(tuple(sorted((index, other))))
    return edges


def verify_pose(source: SourceStructure, pose: Any, mapping: PoseMapping) -> None:
    if int(pose.total_residue()) != len(source.residues):
        raise MappingError("Rosetta added or removed source residues")
    if len(set(mapping.atom_ids)) != len(source.atoms):
        raise MappingError("Pose atom mapping is not one-to-one")
    for occurrence, pose_index in zip(source.residues, mapping.pose_residues, strict=True):
        residue = pose.residue(pose_index)
        if not residue.is_protein() or str(residue.name3()).strip() != occurrence.name:
            raise MappingError("Rosetta changed a standard residue identity")
        source_names = {str(source.atoms.atom_name[i]) for i in occurrence.atom_indices}
        pose_heavy = {str(residue.atom_name(a)).strip() for a in
                      range(1, residue.natoms() + 1) if not residue.atom_is_hydrogen(a)
                      and not residue.atom_type(a).is_virtual()}
        if pose_heavy - source_names - {"OXT"} or source_names - pose_heavy:
            raise MappingError(f"Rosetta changed source heavy atoms at occurrence {occurrence.ordinal} {occurrence.name}: "
                               f"missing={sorted(source_names - pose_heavy)}, extra={sorted(pose_heavy - source_names - {'OXT'})}")
    for index, (atom_number, residue_number) in enumerate(mapping.atom_ids):
        residue = pose.residue(residue_number)
        if str(residue.atom_name(atom_number)).strip() != str(source.atoms.atom_name[index]):
            raise MappingError("Rosetta changed a source atom name")
        if str(residue.atom_type(atom_number).element()).upper() != str(source.atoms.element[index]):
            raise MappingError("Rosetta changed a source element")
    expected = {(a, b) for a, b, _ in source.bonds}
    actual = _pose_edges(pose, mapping)
    if actual != expected:
        raise MappingError(f"Rosetta source-atom bond graph differs: "
                           f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")


def load_pose_mapping(source: SourceStructure, pr: Any, work_dir: Path) -> tuple[Any, PoseMapping]:
    """Load in the native child, restoring every source heavy coordinate."""
    private_pdb = Path(work_dir) / "bridge.pdb"
    source.write_private_pdb(private_pdb)
    pose = pr.pose_from_pdb(str(private_pdb))
    by_private = {(r.private_chain, r.private_number): r for r in source.residues}
    pose_residues = [0] * len(source.residues)
    atom_ids: list[tuple[int, int] | None] = [None] * len(source.atoms)
    for pose_index in range(1, pose.total_residue() + 1):
        info = pose.pdb_info()
        occurrence = by_private.get((str(info.chain(pose_index)), int(info.number(pose_index))))
        if occurrence is None or pose_residues[occurrence.ordinal - 1]:
            raise MappingError("Rosetta returned an unknown or duplicate private residue")
        pose_residues[occurrence.ordinal - 1] = pose_index
        residue = pose.residue(pose_index)
        for index in occurrence.atom_indices:
            name = str(source.atoms.atom_name[index])
            if not residue.has(name):
                raise MappingError(f"Rosetta dropped source atom {occurrence.ordinal}:{name}")
            atom_number = int(residue.atom_index(name))
            atom_ids[index] = (atom_number, pose_index)
            xyz = pr.rosetta.numeric.xyzVector_double_t(*map(float, source.coordinates[index]))
            pose.set_xyz(pr.rosetta.core.id.AtomID(atom_number, pose_index), xyz)
    if not all(pose_residues) or any(value is None for value in atom_ids):
        raise MappingError("Rosetta omitted source residues or atoms")
    for left, right in source.disulfides:
        pr.rosetta.core.conformation.form_disulfide(
            pose.conformation(), pose_residues[left - 1], pose_residues[right - 1])
    if source.head_tail_pairs:
        for first, last in source.head_tail_pairs:
            first_index, last_index = pose_residues[first - 1], pose_residues[last - 1]
            pr.rosetta.core.pose.remove_lower_terminus_type_from_pose_residue(pose, first_index)
            pr.rosetta.core.pose.remove_upper_terminus_type_from_pose_residue(pose, last_index)
            pose.conformation().declare_chemical_bond(last_index, "C", first_index, "N")
            pr.rosetta.core.pose.add_variant_type_to_pose_residue(
                pose, pr.rosetta.core.chemical.CUTPOINT_LOWER, last_index)
            pr.rosetta.core.pose.add_variant_type_to_pose_residue(
                pose, pr.rosetta.core.chemical.CUTPOINT_UPPER, first_index)
        # Terminal/closure variants change native AtomIDs. Resolve them again
        # only after every chemistry operation, then restore the source values.
        for occurrence, pose_index in zip(source.residues, pose_residues, strict=True):
            residue = pose.residue(pose_index)
            for index in occurrence.atom_indices:
                name = str(source.atoms.atom_name[index])
                if not residue.has(name):
                    raise MappingError(f"cyclization dropped source atom {occurrence.ordinal}:{name}")
                atom_ids[index] = (int(residue.atom_index(name)), pose_index)
    # Chemistry operations may rebuild native atoms and move public heavy atoms.
    # Restore source precision again after all chemical variants are installed.
    for index, (atom_number, pose_index) in enumerate(atom_ids):
        pose.set_xyz(pr.rosetta.core.id.AtomID(atom_number, pose_index),
                     pr.rosetta.numeric.xyzVector_double_t(*map(float, source.coordinates[index])))
    mapping = PoseMapping(tuple(atom_ids), tuple(pose_residues))
    verify_pose(source, pose, mapping)
    if not np.array_equal(pose_coordinates(pose, mapping), source.coordinates):
        raise MappingError("source coordinate precision was not restored in Pose")
    return pose, mapping


def pose_coordinates(pose: Any, mapping: PoseMapping) -> np.ndarray:
    return np.asarray([[float(value) for value in pose.residue(residue).xyz(atom)]
                       for atom, residue in mapping.atom_ids], dtype=np.float64)


def _category_values(cif: Any, *, exclude_coordinates: bool) -> dict:
    result = {}
    for block_name in cif:
        result[block_name] = {}
        for category_name in cif[block_name]:
            category = cif[block_name][category_name]
            result[block_name][category_name] = {
                column_name: _column(category, column_name) for column_name in category
                if not (exclude_coordinates and category_name == "atom_site"
                        and column_name in _COORD_COLUMNS)
            }
    return result


def validate_relaxed_file(source: SourceStructure, path: Path) -> SourceStructure:
    result = read_source(path, source.binder_chains, source.target_chains)
    if _category_values(result.cif, exclude_coordinates=True) != _category_values(
            source.cif, exclude_coordinates=True):
        raise MappingError("relaxed CIF changed source categories or public atom identities")
    if result.bonds != source.bonds or result.source_rows != source.source_rows:
        raise MappingError("relaxed CIF changed the actual source bond graph")
    return result


def write_relaxed_cif(source: SourceStructure, pose: Any, mapping: PoseMapping,
                      path: Path) -> None:
    """Verify the native graph, then atomically replace only the owned output."""
    path = Path(path).resolve()
    if path == source.path:
        raise MappingError("relaxed output cannot overwrite the source structure")
    verify_pose(source, pose, mapping)
    coordinates = pose_coordinates(pose, mapping)
    if not np.isfinite(coordinates).all():
        raise MappingError("Rosetta returned nonfinite coordinates")
    cif = _source_document(source.path)
    if _category_values(cif, exclude_coordinates=False) != _category_values(
            source.cif, exclude_coordinates=False):
        raise MappingError("source structure changed during native execution")
    category = cif[source.block_name]["atom_site"]
    for axis, column in enumerate(("Cartn_x", "Cartn_y", "Cartn_z")):
        values = _column(category, column)
        for index, row in enumerate(source.source_rows):
            values[row] = format(float(coordinates[index, axis]), ".17g")
        category[column] = pdbx.CIFColumn(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".cif", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        cif.write(temporary)
        result = validate_relaxed_file(source, temporary)
        if not np.array_equal(result.coordinates, coordinates):
            raise MappingError("relaxed CIF did not retain native coordinate precision")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def native_state_document(source: SourceStructure, pose: Any, mapping: PoseMapping) -> dict:
    """Primitive private geometry needed to reproduce scoring after a restart."""
    verify_pose(source, pose, mapping)
    residues = []
    for pose_index in range(1, pose.total_residue() + 1):
        residue = pose.residue(pose_index)
        names = [str(residue.atom_name(i)).strip() for i in range(1, residue.natoms() + 1)]
        bonds = sorted({tuple(sorted((names[i - 1], names[int(j) - 1])))
                        for i in range(1, residue.natoms() + 1) for j in residue.bonded_neighbor(i)})
        connections = []
        for connection in range(1, residue.n_possible_residue_connections() + 1):
            partner = int(residue.residue_connection_partner(connection))
            if partner:
                atom = int(residue.residue_connect_atom_index(connection))
                partner_connection = int(residue.residue_connection_conn_id(connection))
                partner_atom = int(pose.residue(partner).residue_connect_atom_index(partner_connection))
                connections.append([names[atom - 1], partner,
                    str(pose.residue(partner).atom_name(partner_atom)).strip()])
        type_name = str(residue.type().name())
        residues.append({"pose_index": pose_index, "name3": str(residue.name3()).strip(),
            "type_name": type_name, "variants": type_name.split(":")[1:], "atom_names": names,
            "elements": [str(residue.atom_type(i).element()).upper() for i in range(1, residue.natoms() + 1)],
            "coordinates": [[float(v) for v in residue.xyz(i)] for i in range(1, residue.natoms() + 1)],
            "bonds": [list(edge) for edge in bonds], "connections": sorted(connections)})
    state = {"version": 1, "residues": residues,
        "atom_ids": [list(value) for value in mapping.atom_ids],
        "pose_residues": list(mapping.pose_residues),
        "public_categories": _category_values(source.cif, exclude_coordinates=True),
        "public_coordinates": pose_coordinates(pose, mapping).tolist(),
        "relax_effective": mapping.details.get("relax_effective")}
    validate_native_state_document(source, state, expected_coordinates=pose_coordinates(pose, mapping))
    return state


def validate_native_state_document(source: SourceStructure, state: dict,
                                   *, expected_coordinates: np.ndarray | None = None) -> None:
    """Validate saved public correspondence and actual mapped native bonds."""
    if not isinstance(state, dict) or state.get("version") != 1:
        raise MappingError("private native state has an unsupported version")
    if state.get("public_categories") != _category_values(source.cif, exclude_coordinates=True):
        raise MappingError("private native state changed public identities or categories")
    try:
        coordinates = np.asarray(state["public_coordinates"], dtype=np.float64)
        expected = source.coordinates if expected_coordinates is None else expected_coordinates
        if coordinates.shape != expected.shape or not np.isfinite(coordinates).all() or not np.allclose(
                coordinates, expected, rtol=0, atol=1e-12):
            raise MappingError("private native state does not match completed public coordinates")
        residues = state["residues"]
        if not isinstance(residues, list) or len(residues) != len(source.residues):
            raise MappingError("private native state has different residue occurrences")
        atom_ids = state["atom_ids"]
        pose_residues = state["pose_residues"]
        if (not isinstance(atom_ids, list) or len(atom_ids) != len(source.atoms)
                or sorted(pose_residues) != list(range(1, len(residues) + 1))):
            raise MappingError("private native state has invalid public atom mapping")
        reverse: dict[tuple[int, str], int] = {}
        all_atoms: dict[tuple[int, str], tuple[str, np.ndarray]] = {}
        native_edges: set[tuple[tuple[int, str], tuple[int, str]]] = set()
        for pose_index, saved in enumerate(residues, 1):
            if saved["pose_index"] != pose_index or saved["name3"] not in _AA:
                raise MappingError("private native state has invalid residue identity")
            if not isinstance(saved["type_name"], str) or not saved["type_name"]:
                raise MappingError("private native state lacks the native residue type")
            if saved["variants"] != saved["type_name"].split(":")[1:]:
                raise MappingError("private native state variant list is inconsistent")
            names, elements = saved["atom_names"], saved["elements"]
            xyz = np.asarray(saved["coordinates"], dtype=np.float64)
            if (not isinstance(names, list) or not names or len(names) != len(set(names))
                    or any(not isinstance(name, str) or not name for name in names)
                    or not isinstance(elements, list) or len(elements) != len(names)
                    or xyz.shape != (len(names), 3) or not np.isfinite(xyz).all()):
                raise MappingError("private native state has malformed atom coordinates")
            for name, element, coordinate in zip(names, elements, xyz, strict=True):
                all_atoms[(pose_index, name)] = (element, coordinate)
            for left, right in saved["bonds"]:
                if left not in names or right not in names or left == right:
                    raise MappingError("private native state has an invalid intra-residue bond")
                native_edges.add(tuple(sorted(((pose_index, left), (pose_index, right)))))
            for name, partner, other_name in saved["connections"]:
                if name not in names or type(partner) is not int or not 1 <= partner <= len(residues):
                    raise MappingError("private native state has an invalid residue connection")
                native_edges.add(tuple(sorted(((pose_index, name), (partner, other_name)))))
        for occurrence, pose_index in zip(source.residues, pose_residues, strict=True):
            if residues[pose_index - 1]["name3"] != occurrence.name:
                raise MappingError("private native state changed a source residue")
        for index, pair in enumerate(atom_ids):
            if not isinstance(pair, list) or len(pair) != 2 or any(type(i) is not int for i in pair):
                raise MappingError("private native state has an invalid AtomID")
            atom, residue_index = pair
            if not 1 <= residue_index <= len(residues):
                raise MappingError("private native state AtomID names an absent residue")
            names = residues[residue_index - 1]["atom_names"]
            if not 1 <= atom <= len(names):
                raise MappingError("private native state AtomID names an absent atom")
            name = names[atom - 1]
            key = (residue_index, name)
            if (key in reverse or name != str(source.atoms.atom_name[index])
                    or all_atoms[key][0] != str(source.atoms.element[index])
                    or residue_index != pose_residues[source.atom_residues[index] - 1]
                    or not np.allclose(all_atoms[key][1], coordinates[index], rtol=0, atol=1e-12)):
                raise MappingError("private native state does not map public atoms exactly")
            reverse[key] = index
        actual = {tuple(sorted((reverse[left], reverse[right]))) for left, right in native_edges
                  if left in reverse and right in reverse}
        if actual != {(a, b) for a, b, _ in source.bonds}:
            raise MappingError("private native state has a different public bond graph")
        if any(left not in all_atoms or right not in all_atoms for left, right in native_edges):
            raise MappingError("private native state has a bond to an absent atom")
    except MappingError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise MappingError(f"private native state is malformed: {exc}") from exc


def restore_native_state(source: SourceStructure, pose: Any, mapping: PoseMapping,
                         state: dict, pr: Any) -> None:
    """Restore exact saved native variants, hydrogens and terminal preparation."""
    validate_native_state_document(source, state)
    if (state["pose_residues"] != list(mapping.pose_residues)
            or state["atom_ids"] != [list(value) for value in mapping.atom_ids]):
        raise MappingError("reloaded Pose changed the saved public atom mapping")
    for pose_index, saved in enumerate(state["residues"], 1):
        residue = pose.residue(pose_index)
        if str(residue.type().name()) != saved["type_name"]:
            types = pose.residue_type_set_for_pose(pr.rosetta.core.chemical.FULL_ATOM_t)
            residue_type = types.name_map(saved["type_name"])
            replacement = pr.rosetta.core.conformation.ResidueFactory.create_residue(
                residue_type, residue, pose.conformation())
            pose.replace_residue(pose_index, replacement, False)
            residue = pose.residue(pose_index)
        names = [str(residue.atom_name(i)).strip() for i in range(1, residue.natoms() + 1)]
        elements = [str(residue.atom_type(i).element()).upper() for i in range(1, residue.natoms() + 1)]
        if names != saved["atom_names"] or elements != saved["elements"]:
            raise MappingError("reloaded native atoms differ from the saved residue type")
        for atom_number, coordinate in enumerate(saved["coordinates"], 1):
            pose.set_xyz(pr.rosetta.core.id.AtomID(atom_number, pose_index),
                         pr.rosetta.numeric.xyzVector_double_t(*map(float, coordinate)))
    verify_pose(source, pose, mapping)
    current = native_state_document(source, pose, mapping)
    for key in ("residues", "atom_ids", "pose_residues", "public_categories", "public_coordinates"):
        if current[key] != state[key]:
            raise MappingError(f"restored native state differs in {key}")
    if state.get("relax_effective") is not None:
        if not isinstance(state["relax_effective"], dict):
            raise MappingError("saved effective relax settings are malformed")
        mapping.details["relax_effective"] = dict(state["relax_effective"])
