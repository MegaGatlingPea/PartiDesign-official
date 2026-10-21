"""Static antibody scaffold interpretation on canonical CIF label numbering.

The licensed BoltzGen YAML/CIF assets are data only. No BoltzGen runtime is
imported. Official edit windows are retained verbatim rather than relabelled
as IMGT intervals.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
import re

import numpy as np
import yaml
from biotite.structure.io.pdbx import CIFFile

from partidesign.paths import resource_root

from ..core.records import DesignConfigError
from ..utils.atoms import PROTEIN_ATOM37

ASSET_DIR = resource_root() / "configs" / "resources" / "antibody" / "scaffolds"
SOURCE_REVISION = "a3149cf18eeb58648d1abbb27539bd73f746cdda"
_AA = dict(zip(
    ("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL").split(),
    "ARNDCQEGHILKMFPSTWYV",
))


def positions(selector: object, available: tuple[int, ...], where: str) -> tuple[int, ...]:
    if selector is None:
        return available
    if not isinstance(selector, (str, int)) or isinstance(selector, bool):
        raise DesignConfigError(f"{where} must use canonical label_seq_id ranges")
    selected: list[int] = []
    for term in str(selector).split(","):
        term = term.strip()
        if re.fullmatch(r"\d+", term):
            lo = hi = int(term)
        elif re.fullmatch(r"\d*\.\.\d*", term) and term != "..":
            a, b = term.split("..")
            lo, hi = int(a) if a else min(available), int(b) if b else max(available)
        else:
            raise DesignConfigError(f"invalid {where} selector {term!r}")
        if lo < 1 or lo > hi:
            raise DesignConfigError(f"invalid {where} interval {term!r}")
        selected.extend(range(lo, hi + 1))
    if len(selected) != len(set(selected)) or not set(selected) <= set(available):
        raise DesignConfigError(f"{where} selects duplicate or unavailable source residues")
    return tuple(sorted(selected))


def _count_interval(value: object, where: str) -> tuple[int, int]:
    if type(value) is int and value > 0:
        return value, value
    if isinstance(value, str) and re.fullmatch(r"[1-9]\d*\.\.[1-9]\d*", value):
        lo, hi = map(int, value.split(".."))
        if lo <= hi:
            return lo, hi
    raise DesignConfigError(f"{where} must be a positive count or inclusive interval")


@dataclass(frozen=True)
class ScaffoldChain:
    role: str
    source_chain: str
    source_positions: tuple[int, ...]
    sequence: str
    cdr_positions: tuple[tuple[int, ...], ...]
    official_lengths: tuple[tuple[int, int], ...]
    excluded_positions: tuple[int, ...]
    insertions: tuple[tuple[int, int, int], ...]
    visible: tuple[bool, ...]
    coords_nm: np.ndarray
    atom_mask: np.ndarray
    author_numbers: tuple[str, ...]
    insertion_codes: tuple[str, ...]
    source_present: tuple[bool, ...] = ()
    boundary_unresolved: tuple[int, ...] = ()


@dataclass(frozen=True)
class Scaffold:
    name: str
    yaml_path: Path
    cif_path: Path
    chains: tuple[ScaffoldChain, ...]
    reset_res_index: tuple[str, ...]


def official_names() -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in ASSET_DIR.rglob("*.yaml")))


def scaffold_path(name: str) -> Path:
    hits = [p for p in ASSET_DIR.rglob("*.yaml") if p.stem == name]
    if len(hits) != 1:
        raise DesignConfigError(f"unknown official antibody scaffold {name!r}; choices={official_names()}")
    return hits[0]


@lru_cache(maxsize=64)
def load_scaffold(path: Path, role_pairs: tuple[tuple[str, str], ...] = ()) -> Scaffold:
    try:
        return _load_scaffold(path, role_pairs)
    except DesignConfigError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        raise DesignConfigError(f"invalid antibody scaffold {path}: malformed YAML/CIF field ({exc})") from exc


def _validate_scaffold_rows(spec: Mapping[str, Any], path: Path) -> None:
    for key, wrapper in (("include", "chain"), ("design", "chain"),
                         ("exclude", "chain"), ("reset_res_index", "chain"),
                         ("structure_groups", "group"), ("design_insertions", "insertion")):
        rows = spec.get(key, [])
        if not isinstance(rows, list):
            raise DesignConfigError(f"scaffold {path}: {key} must be a list")
        ids = []
        for i, row in enumerate(rows):
            where = f"scaffold {path}: {key}[{i}].{wrapper}"
            if not isinstance(row, Mapping) or set(row) != {wrapper} or not isinstance(row[wrapper], Mapping):
                raise DesignConfigError(f"{where} must be a mapping")
            value = row[wrapper]
            if not isinstance(value.get("id"), str) or not value["id"]:
                raise DesignConfigError(f"{where}.id must be a nonempty label chain")
            ids.append(value["id"])
            allowed = {"id", "res_index"}
            if key == "structure_groups":
                allowed.add("visibility")
            if key == "design_insertions":
                allowed.add("num_residues")
            if set(value) - allowed:
                raise DesignConfigError(f"{where}: unsupported fields {sorted(set(value)-allowed)}")
        if key in {"include", "design", "reset_res_index"} and len(ids) != len(set(ids)):
            raise DesignConfigError(f"scaffold {path}: duplicate chain in {key}")


def _load_scaffold(path: Path, role_pairs: tuple[tuple[str, str], ...] = ()) -> Scaffold:
    path = Path(path).resolve()
    try:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DesignConfigError(f"cannot read antibody scaffold {path}") from exc
    if not isinstance(spec, Mapping):
        raise DesignConfigError("antibody scaffold YAML must be a mapping")
    allowed = {"path", "include", "design", "structure_groups", "exclude", "design_insertions", "reset_res_index"}
    if set(spec) - allowed:
        raise DesignConfigError(f"unsupported scaffold fields: {sorted(set(spec)-allowed)}")
    _validate_scaffold_rows(spec, path)
    source = spec.get("path")
    if not isinstance(source, str) or "://" in source:
        raise DesignConfigError("scaffold path must reference a local CIF")
    cif_path = (path.parent / source).resolve()
    try:
        block = CIFFile.read(cif_path).block
    except (OSError, ValueError) as exc:
        raise DesignConfigError(f"cannot read scaffold CIF {cif_path}") from exc
    asym = block["struct_asym"]
    entity_by_chain = dict(zip(asym["id"].as_array(str), asym["entity_id"].as_array(str)))
    poly = block["entity_poly_seq"]
    entity, nums, names = (poly[k].as_array(str) for k in ("entity_id", "num", "mon_id"))
    site = block["atom_site"]
    labels = site["label_asym_id"].as_array(str)
    seqids = site["label_seq_id"].as_array(str)
    atom_names = site["label_atom_id"].as_array(str)
    model = site["pdbx_PDB_model_num"].as_array(str) if "pdbx_PDB_model_num" in site else np.full(len(labels), "1")
    alt = site["label_alt_id"].as_array(str) if "label_alt_id" in site else np.full(len(labels), ".")
    occupancy = site["occupancy"].as_array(float) if "occupancy" in site else np.ones(len(labels))
    xyz = np.column_stack([site[k].as_array(float) for k in ("Cartn_x", "Cartn_y", "Cartn_z")]) / 10.0
    author = site["auth_seq_id"].as_array(str)
    ins = site["pdbx_PDB_ins_code"].as_array(str)
    includes = spec.get("include")
    if not isinstance(includes, list) or len(includes) not in (1, 2):
        raise DesignConfigError("scaffold must include one VHH or paired VH/VL chains")
    include_chains = [row["chain"]["id"] for row in includes]
    roles = dict(role_pairs) if role_pairs else dict(zip(("VHH",) if len(includes) == 1 else ("VH", "VL"), include_chains))
    if set(roles) not in ({"VHH"}, {"VH", "VL"}) or set(roles.values()) != set(include_chains):
        raise DesignConfigError("scaffold roles must map exactly included VHH or VH/VL chains")
    result: list[ScaffoldChain] = []
    for role, chain in roles.items():
        if chain not in entity_by_chain:
            raise DesignConfigError(f"scaffold label chain {chain!r} has no polymer entity")
        seq = {int(n): _AA.get(str(aa), "X") for e, n, aa in zip(entity, nums, names) if e == entity_by_chain[chain]}
        include_row = next(row["chain"] for row in includes if row["chain"]["id"] == chain)
        included = positions(include_row.get("res_index"), tuple(sorted(seq)), f"include.{chain}")
        if any(seq[n] == "X" for n in included):
            raise DesignConfigError("antibody scaffold V domains must contain standard protein identities")
        design_rows = [r["chain"] for r in spec.get("design", []) if r["chain"]["id"] == chain]
        if len(design_rows) != 1 or not isinstance(design_rows[0].get("res_index"), str):
            raise DesignConfigError("scaffold needs exactly three explicit source CDR windows per chain")
        terms = design_rows[0]["res_index"].split(",")
        if len(terms) != 3:
            raise DesignConfigError("scaffold needs three CDR windows")
        cdrs = tuple(positions(t, included, f"CDR.{chain}") for t in terms)
        if len(set(sum(cdrs, ()))) != sum(map(len, cdrs)):
            raise DesignConfigError("scaffold CDR windows overlap")
        excluded = tuple(sorted(n for r in spec.get("exclude", []) if r["chain"]["id"] == chain for n in positions(r["chain"].get("res_index"), included, f"exclude.{chain}")))
        if not set(excluded) <= set(sum(cdrs, ())):
            raise DesignConfigError("scaffold excludes framework residues")
        insertions = []
        for row in spec.get("design_insertions", []):
            row = row["insertion"]
            if row["id"] != chain:
                continue
            anchor = row["res_index"]
            if type(anchor) is not int or anchor not in sum(cdrs, ()):
                raise DesignConfigError("scaffold insertion anchor must be inside a CDR window")
            lo, hi = _count_interval(row["num_residues"], "insertion.num_residues")
            insertions.append((anchor, lo, hi))
        intervals = []
        for window in cdrs:
            base = len(window) - len(set(window) & set(excluded))
            additions = [(a, b) for n, a, b in insertions if n in window]
            intervals.append((base + sum(a for a, b in additions), base + sum(b for a, b in additions)))
        visible = np.ones(len(included), dtype=bool)
        for row in spec.get("structure_groups", []):
            row = row["group"]
            if row["id"] == chain:
                selected = set(positions(row.get("res_index"), included, "structure_groups"))
                visible[[n in selected for n in included]] = int(row["visibility"]) > 0
        # Hidden CDR geometry is never an antibody template input, including fixed CDRs.
        visible[[n in sum(cdrs, ()) for n in included]] = False
        coords = np.zeros((len(included), 37, 3), dtype=np.float32)
        mask = np.zeros((len(included), 37), dtype=bool)
        authors, codes = [""] * len(included), [""] * len(included)
        present = np.zeros(len(included), dtype=bool)
        for i, n in enumerate(included):
            rows = np.flatnonzero((labels == chain) & (seqids == str(n)) & (model == model[0]))
            present[i] = len(rows) > 0
            for j in sorted(rows, key=lambda j: (-occupancy[j], alt[j] not in (".", "?", ""), alt[j])):
                authors[i], codes[i] = author[j], "" if ins[j] in (".", "?") else ins[j]
                if atom_names[j] in ("N", "CA", "C", "O"):
                    slot = PROTEIN_ATOM37.index(atom_names[j])
                    if not mask[i, slot] and np.isfinite(xyz[j]).all():
                        coords[i, slot], mask[i, slot] = xyz[j], True
        if not present.any():
            raise DesignConfigError("scaffold included V domain has no resolved source residues")
        # Official parse_file trims unresolved ends of each included span;
        # internal missing residues remain tokens on the label-sequence axis.
        first, last = int(np.flatnonzero(present)[0]), int(np.flatnonzero(present)[-1]) + 1
        boundary_unresolved = included[:first] + included[last:]
        included = included[first:last]
        if not set(sum(cdrs, ())) <= set(included):
            raise DesignConfigError("scaffold boundary trimming removes a declared CDR window")
        coords, mask, visible, present = (value[first:last] for value in (coords, mask, visible, present))
        authors, codes = authors[first:last], codes[first:last]
        mask &= visible[:, None]
        coords[~mask] = 0
        coords.flags.writeable = mask.flags.writeable = False
        result.append(ScaffoldChain(role, chain, included, "".join(seq[n] for n in included), cdrs, tuple(intervals), excluded, tuple(insertions), tuple(visible.tolist()), coords, mask, tuple(authors), tuple(codes), tuple(present.tolist()), boundary_unresolved))
    reset = tuple(r["chain"]["id"] for r in spec.get("reset_res_index", []))
    if set(reset) != set(include_chains):
        raise DesignConfigError("scaffold requires reset_res_index on every variable chain")
    return Scaffold(path.stem, path, cif_path, tuple(result), reset)
