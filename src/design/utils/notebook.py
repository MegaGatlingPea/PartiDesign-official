"""Lightweight helpers for interactive task preparation.

Use the native reader, parser, cropper and compiler. Reload a saved input before
selecting residues so that selections use the exported mmCIF label numbering.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import base64
from functools import lru_cache
from numbers import Real
import json
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import biotite.structure as struc
from biotite.structure.io import pdbx
import numpy as np
import requests
import yaml
from scipy.spatial import cKDTree

from ..core.compiler import (
    _SOURCE_SEGMENT, _contig_gap_range, _resolve_atom_selector, compile_request,
    prepare_task_source,
)
from ..core.crop import source_selector
from ..core.ligand_length import base_request_seed, resolve_length_distributions
from ..core.records import DesignConfigError, DesignInputError, ExpandedRequest
from ..core.tasks import parse_task
from .atoms import (
    DNA_RES_ATOMS, ELEMENTS_128, PROTEIN_RES_ATOMS, RNA_RES_ATOMS, SIDECHAIN_TIP_ATOMS,
)
from .ligand_identity import (
    build_validated_ligand_identity,
    compute_ligand_component_signature,
)
from .chemistry import TrustedCCDComponent
from .reader import read_structure
from .writer import StructureWriteError, write_cif


MAX_PREVIEW_PLANS = 8
MAX_PREVIEW_DESIGN_TOKENS = 4096
MAX_PREVIEW_SOURCE_ATOMS = 100_000
MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024


@lru_cache(maxsize=8)
def _trusted_ccd_preview(
    path_text: str, component_ids: tuple[str, ...], modified_ns: int,
) -> dict[str, TrustedCCDComponent]:
    """Read local chemical definitions in a short-lived subprocess."""
    program = """
import base64, json, sys
from pathlib import Path
from src.design.utils.chemistry import load_trusted_ccd_components
records = load_trusted_ccd_components(Path(sys.argv[1]), json.loads(sys.argv[2]))
print(json.dumps({k: base64.b64encode(v.rdkit_binary).decode('ascii') for k, v in records.items()}))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", program, path_text, json.dumps(component_ids)],
            cwd=Path(__file__).resolve().parents[3], capture_output=True,
            text=True, timeout=60, check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise DesignInputError("The local CCD check timed out after 60 seconds.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip().splitlines()
        raise DesignInputError(f"The local CCD check failed: {detail[-1] if detail else exc.returncode}") from exc
    records = json.loads(result.stdout)
    return {key: TrustedCCDComponent(key, base64.b64decode(value, validate=True))
            for key, value in records.items()}


def fetch_rcsb(
    pdb_id: str, output_dir: str | Path, assembly_id: int | str | None = None,
) -> Path:
    """Download an RCSB entry or biological assembly as mmCIF.

    Publish the file only after a complete download and format check.
    """
    identifier = str(pdb_id).strip()
    if re.fullmatch(r"[0-9][A-Za-z0-9]{3}", identifier):
        identifier = identifier.upper()
    elif re.fullmatch(r"pdb_[A-Za-z0-9]{8}", identifier, re.IGNORECASE):
        identifier = identifier.lower()
    else:
        raise ValueError("Enter a four-character PDB ID, such as 1CRN, or pdb_XXXXXXXX.")
    suffix = ""
    if assembly_id is not None:
        if isinstance(assembly_id, bool) or not re.fullmatch(r"[1-9][0-9]*", str(assembly_id)):
            raise ValueError("The assembly ID must be a positive integer.")
        suffix = f"-assembly{assembly_id}"
    destination = Path(output_dir).expanduser().resolve() / f"{identifier}{suffix}.cif"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix=".cif", dir=destination.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    url = f"https://files.rcsb.org/download/{identifier}{suffix}.cif"
    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            size = 0
            with temporary_path.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise ValueError("The RCSB download exceeds the 256 MiB limit.")
                    stream.write(chunk)
        cif = pdbx.CIFFile.read(temporary_path)
        if not cif or "atom_site" not in cif[next(iter(cif))]:
            raise DesignInputError("The downloaded file contains no mmCIF atom_site data.")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def _chains(atoms: struc.AtomArray) -> np.ndarray:
    return np.asarray([
        str(label) if str(label) not in {"", ".", "?"} else str(chain)
        for label, chain in zip(atoms.label_asym_id, atoms.chain_id, strict=True)
    ])


def load_input(
    path: str | Path, *, keep_chains: Sequence[str] | None = None,
    exclude_res_names: Sequence[str] = (),
) -> struc.AtomArray:
    """Read the first model, resolve alternate locations and remove water/hydrogen.

    Optional filtering preserves existing bonds, author numbering and chain
    breaks. Chain selections use normalized label_asym_id values.
    """
    atoms = read_structure(Path(path).expanduser().resolve())
    mask = np.ones(len(atoms), dtype=bool)
    if keep_chains is not None:
        selected = {keep_chains} if isinstance(keep_chains, str) else set(keep_chains)
        missing = selected.difference(_chains(atoms).tolist())
        if missing:
            raise DesignInputError(f"These chains are absent from the input: {sorted(missing)}")
        mask &= np.isin(_chains(atoms), list(selected))
    excluded = ({exclude_res_names} if isinstance(exclude_res_names, str)
                else set(exclude_res_names))
    if excluded:
        mask &= ~np.isin(atoms.res_name, [str(value).strip().upper() for value in excluded])
    if not mask.any():
        raise DesignInputError("No non-water heavy atoms remain after cleaning.")
    return atoms[mask]


def chain_summary(atoms: struc.AtomArray) -> list[dict[str, Any]]:
    """Summarize observed residues/components by label chain ID and modality."""
    chains = _chains(atoms)
    result = []
    for chain, modality in dict.fromkeys(zip(chains.tolist(), atoms.mol_type.tolist())):
        rows = (chains == chain) & (atoms.mol_type == modality)
        result.append({
            "chain": chain, "modality": modality,
            "residues": int(len(np.unique(atoms.residue_uid[rows]))),
            "atoms": int(rows.sum()),
        })
    return result


def _hotspot_chain_group(values: Sequence[str] | str, role: str) -> set[str]:
    if isinstance(values, str):
        values = [values]
    try:
        selected = set(values)
    except (TypeError, ValueError) as exc:
        raise DesignInputError(f"Select at least one {role} chain.") from exc
    if not selected or any(not isinstance(chain, str) or not chain.strip() for chain in selected):
        raise DesignInputError(f"Select at least one {role} chain.")
    return selected


def calculate_hotspots(
    atoms: struc.AtomArray,
    context_chains: Sequence[str] | str,
    reference_chains: Sequence[str] | str,
    cutoff: float = 6.0,
) -> list[dict[str, Any]]:
    """Find Context residues within ``cutoff`` angstroms of any Reference atom.

    Use all finite, resolved heavy atoms in the selected chains, independently
    of any Motif selection. Return source-order residue records with the minimum
    heavy-atom distance, normalized selector, source row/UID and author metadata.
    No contacts is a valid empty result. The input is never modified.
    """
    context = _hotspot_chain_group(context_chains, "Context")
    reference = _hotspot_chain_group(reference_chains, "Reference")
    if context & reference:
        raise DesignInputError("Context and Reference chains must not overlap.")
    if (isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, Real)
            or not np.isfinite(cutoff) or cutoff <= 0):
        raise DesignInputError("The hotspot cutoff must be a positive finite distance in angstroms.")
    if not len(atoms):
        raise DesignInputError("Import a structure before calculating hotspots.")
    chains = _chains(atoms)
    existing = set(chains.tolist())
    for selected, role in ((context, "Context"), (reference, "Reference")):
        missing = selected - existing
        if missing:
            raise DesignInputError(f"These {role} chains are absent from the input: {sorted(missing)}")
    coordinates = np.asarray(atoms.coord, dtype=np.float64)
    elements = np.char.upper(np.char.strip(np.asarray(atoms.element).astype(str)))
    heavy = np.isin(elements, ELEMENTS_128[1:118])
    valid = np.asarray(atoms.coord_valid, dtype=bool) & np.isfinite(coordinates).all(axis=1) & heavy
    context_rows = np.flatnonzero(valid & np.isin(chains, list(context)))
    reference_rows = np.flatnonzero(valid & np.isin(chains, list(reference)))
    if not len(context_rows):
        raise DesignInputError("Context has no resolved heavy-atom coordinates.")
    if not len(reference_rows):
        raise DesignInputError("Reference has no resolved heavy-atom coordinates.")
    # One nearest-neighbor distance per Context atom; no atom-pair matrix.
    nearest = cKDTree(coordinates[reference_rows]).query(
        coordinates[context_rows], k=1, workers=1,
    )[0]
    minimum_by_uid: dict[int, float] = {}
    for row, distance in zip(context_rows, nearest, strict=True):
        if distance <= cutoff:
            uid = int(atoms.residue_uid[row])
            minimum_by_uid[uid] = min(minimum_by_uid.get(uid, float("inf")), float(distance))
    if not minimum_by_uid:
        return []

    first_rows: dict[int, int] = {}
    for row, value in enumerate(atoms.residue_uid):
        first_rows.setdefault(int(value), row)
    # Validate locator uniqueness in one pass, including non-contact residues.
    locators: dict[int, tuple[str, int, str]] = {}
    numeric_counts: dict[tuple[str, int], int] = {}
    explicit_counts: dict[tuple[str, int, str], int] = {}
    for uid, row in first_rows.items():
        chain = str(chains[row])
        if chain not in context:
            continue
        try:
            number = int(atoms.label_seq_id[row])
        except ValueError:
            number = int(atoms.res_id[row])
        insertion = str(atoms.ins_code[row])
        if insertion in {".", "?"}:
            insertion = ""
        locator = (chain, number, insertion)
        locators[uid] = locator
        numeric_counts[(chain, number)] = numeric_counts.get((chain, number), 0) + 1
        explicit_counts[locator] = explicit_counts.get(locator, 0) + 1

    result = []
    for uid in sorted(minimum_by_uid, key=first_rows.__getitem__):
        row = first_rows[uid]
        chain, number, insertion = locators[uid]
        count = explicit_counts[(chain, number, insertion)] if insertion else numeric_counts[(chain, number)]
        prefix = chain if re.fullmatch(r"[A-Za-z]", chain) else f"[{chain}]"
        selector = f"{prefix}{number}{insertion}"
        if number <= 0 or count != 1 or _SOURCE_SEGMENT.fullmatch(selector) is None:
            raise DesignInputError("A hotspot residue number is ambiguous. Save and reload the cleaned input first.")
        author = {field: str(getattr(atoms, field)[row])
                  for field in ("auth_asym_id", "auth_seq_id", "auth_ins_code")}
        author_chain = author["auth_asym_id"] if author["auth_asym_id"] not in {"", ".", "?"} else chain
        author_number = author["auth_seq_id"] if author["auth_seq_id"] not in {"", ".", "?"} else str(number)
        author_insertion = author["auth_ins_code"] if author["auth_ins_code"] not in {".", "?"} else ""
        result.append({
            "selector": selector, "uid": uid, "row": row, "chain": chain,
            "name": str(atoms.res_name[row]), "modality": str(atoms.mol_type[row]),
            **author, "auth_locator": f"{author_chain}:{author_number}{author_insertion}",
            "minimum_distance": minimum_by_uid[uid],
        })
    return result


def _same_contents(left: Path, right: Path) -> bool:
    if not right.is_file() or left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as first, right.open("rb") as second:
        while chunk := first.read(1024 * 1024):
            if chunk != second.read(1024 * 1024):
                return False
    return True


def stage_input(path: str | Path, raw_dir: str | Path) -> Path:
    """Stage an original input without overwriting a different file."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DesignInputError("The input structure file could not be found.")
    directory = Path(raw_dir).expanduser().resolve()
    if source.is_relative_to(directory):
        return source
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        name = source.name if number == 1 else f"{source.stem}_{number}{source.suffix}"
        destination = directory / name
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() and _same_contents(source, destination):
                return destination
            number += 1
            continue
        try:
            with destination.open("xb") as output, source.open("rb") as original:
                shutil.copyfileobj(original, output, length=1024 * 1024)
        except FileExistsError:
            number += 1
            continue
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return destination


# Training-compatible non-backbone atom group, restricted to observed atoms.
_NA_MOTIF_NON_BACKBONE = frozenset({
    "O4'", "C1'", "C2'", "O2'", "N9", "C8", "N7", "C5", "C6", "N6",
    "O6", "N1", "C2", "N2", "O2", "N3", "C4", "N4", "O4", "C7",
})
_POLYMER_INVENTORIES = {
    "protein": PROTEIN_RES_ATOMS, "rna": RNA_RES_ATOMS, "dna": DNA_RES_ATOMS,
}


def _source_residue(atoms: struc.AtomArray, selector: str) -> tuple[int, np.ndarray]:
    if not isinstance(selector, str) or _SOURCE_SEGMENT.fullmatch(selector.strip()) is None:
        raise DesignInputError("Select one input residue, for example A12.")
    try:
        selected = source_selector(atoms, selector.strip())
    except ValueError as exc:
        raise DesignInputError(f"The input residue could not be uniquely located: {selector}") from exc
    if len(selected) != 1:
        raise DesignInputError(f"An atom selection must refer to one residue: {selector}")
    uid = int(selected[0])
    rows = np.flatnonzero(atoms.residue_uid == uid)
    if len(set(atoms.mol_type[rows])) != 1 or str(atoms.mol_type[rows[0]]) not in _POLYMER_INVENTORIES:
        raise DesignInputError("Select a protein, RNA or DNA residue.")
    return uid, rows


def _residue_atom_choices(atoms: struc.AtomArray, rows: np.ndarray) -> tuple[list[str], dict[str, Any]]:
    row = int(rows[0])
    modality, name = str(atoms.mol_type[row]), str(atoms.res_name[row])
    resolved = rows[np.asarray(atoms.coord_valid[rows], dtype=bool)]
    names = [str(value) for value in atoms.atom_name[resolved]]
    if len(names) != len(set(names)):
        raise DesignInputError(f"Residue {name} has duplicate atom names and cannot be selected by atom.")
    canonical = _POLYMER_INVENTORIES[modality].get(name)
    choices: dict[str, Any] = {
        "full": {"label": "All observed atoms", "enabled": bool(names), "atoms": None,
                 "missing_atoms": [], "reason": "" if names else "No resolved atoms are available."},
    }
    if modality == "protein":
        tip = list(SIDECHAIN_TIP_ATOMS.get(name, ()))
        missing = [atom for atom in tip if atom not in names]
        enabled = canonical is not None and bool(tip) and not missing
        reason = ("" if enabled else "No side-chain tip preset is available for this residue." if not tip
                  else "Unresolved side-chain tip atoms: " + ", ".join(missing))
        choices["tip"] = {"label": "Side-chain tip", "enabled": enabled,
                          "atoms": tip, "missing_atoms": missing, "reason": reason}
        choices["custom"] = {"label": "Choose atoms", "enabled": bool(names),
                             "atoms": names, "missing_atoms": [],
                             "reason": "" if names else "No resolved atoms are available."}
    else:
        expected = ([] if canonical is None else
                    [atom for atom in canonical if atom in _NA_MOTIF_NON_BACKBONE])
        selected = [atom for atom in expected if atom in names]
        missing = [atom for atom in expected if atom not in names]
        enabled = canonical is not None and bool(selected)
        reason = ("Partial-atom presets are unavailable for modified nucleotides." if canonical is None else
                  "" if enabled else "No base or non-backbone atoms are resolved for this residue.")
        choices["sugar_base"] = {"label": "Base and non-backbone atoms", "enabled": enabled,
                                 "atoms": selected, "missing_atoms": missing,
                                 "reason": reason}
    return names, choices


def motif_atom_names(
    atoms: struc.AtomArray, selector: str, choice: str | Sequence[str] | None,
) -> list[str] | None:
    """Resolve one residue's Motif atom selection; None keeps all observed atoms.

    Protein selections may name individual atoms. Nucleic-acid selections use
    the full or trained non-backbone group. Missing group atoms are reported
    by inspect_contig and are never fabricated.
    """
    _, rows = _source_residue(atoms, selector)
    names, presets = _residue_atom_choices(atoms, rows)
    modality = str(atoms.mol_type[int(rows[0])])
    if (choice is None or (isinstance(choice, str) and choice in {"all", "full"})
            or (isinstance(choice, Sequence) and not isinstance(choice, str) and not choice)):
        preset = presets["full"]
        if not preset["enabled"]:
            raise DesignInputError(f"{selector}: {preset['reason']}")
        return None
    if isinstance(choice, str):
        if choice not in presets or choice == "custom":
            raise DesignInputError(f"{selector} does not support this atom selection.")
        preset = presets[choice]
        if not preset["enabled"]:
            raise DesignInputError(f"{selector}: {preset['reason']}")
        return list(preset["atoms"])
    if modality != "protein":
        raise DesignInputError("For nucleic acids, choose all observed atoms or the base and non-backbone preset.")
    if not isinstance(choice, Sequence) or not choice or not all(isinstance(atom, str) and atom for atom in choice):
        raise DesignInputError(f"Select at least one atom for {selector}.")
    chosen = list(choice)
    if len(chosen) != len(set(chosen)):
        raise DesignInputError(f"The atom selection for {selector} contains duplicate names.")
    missing = [atom for atom in chosen if atom not in names]
    if missing:
        raise DesignInputError(f"These atoms in {selector} are missing or unresolved: {', '.join(missing)}")
    return chosen


def _source_residue_record(
    atoms: struc.AtomArray, uid: int, *, layout_min: int, layout_max: int,
) -> dict[str, Any]:
    rows = np.flatnonzero(atoms.residue_uid == uid)
    row = int(rows[0])
    chain = str(atoms.chain_id[row])
    prefix = chain if re.fullmatch(r"[A-Za-z]", chain) else f"[{chain}]"
    label = str(atoms.label_seq_id[row])
    try:
        number = int(label)
    except ValueError:
        number = int(atoms.res_id[row])
    selector = f"{prefix}{number}{str(atoms.ins_code[row])}"
    # Selectors use label numbering; author numbering is display metadata.
    _, checked_rows = _source_residue(atoms, selector)
    if int(atoms.residue_uid[int(checked_rows[0])]) != uid:
        raise DesignInputError("This residue number is ambiguous. Save and reload the cleaned input first.")
    names, atom_choices = _residue_atom_choices(atoms, rows)
    author = {field: str(getattr(atoms, field)[row])
              for field in ("auth_asym_id", "auth_seq_id", "auth_ins_code")}
    auth_locator = f"{author['auth_asym_id']}:{author['auth_seq_id']}{author['auth_ins_code']}"
    chain_uids = list(dict.fromkeys(int(value) for value in atoms.residue_uid[atoms.chain_id == chain]))
    return {"selector": selector, "uid": uid, "row": row, "chain": chain,
            "name": str(atoms.res_name[row]), "modality": str(atoms.mol_type[row]),
            **author, "auth_locator": auth_locator, "available_atoms": names,
            "atom_choices": atom_choices, "source_index": chain_uids.index(uid) + 1,
            "layout_min_position": layout_min, "layout_max_position": layout_max}


def inspect_contig(atoms: struc.AtomArray, chain: str, token: str) -> dict[str, Any]:
    """Inspect Contig bounds and copied residues without allocating gap layouts."""
    if not isinstance(chain, str) or not chain or any(value in chain for value in "]/"):
        raise DesignInputError("Select a valid Motif input chain.")
    if not isinstance(token, str) or not token.strip():
        raise DesignInputError("Enter a Contig, for example 10/A12-20/10.")
    segments = [part.strip() for part in token.split("/")]
    minimum = maximum = 0
    copied: set[int] = set()
    residues: list[dict[str, Any]] = []
    for segment in segments:
        matched = _SOURCE_SEGMENT.fullmatch(segment)
        if matched is None:
            try:
                low, high = _contig_gap_range(segment)
            except (ValueError, OverflowError) as exc:
                raise DesignInputError(f"Invalid Contig segment: {segment or 'empty segment'}") from exc
            minimum += low
            maximum += high
            if maximum > MAX_PREVIEW_DESIGN_TOKENS:
                raise DesignInputError("This preview supports Contigs of at most 4096 residues.")
            continue
        selected_chain = matched.group(1) or matched.group(2)
        if selected_chain != chain:
            raise DesignInputError("All copied Contig residues must belong to the selected input chain.")
        try:
            selected = source_selector(atoms, segment)
        except ValueError as exc:
            raise DesignInputError(f"The input segment in the Contig could not be found: {segment}") from exc
        for value in selected:
            uid = int(value)
            if uid in copied:
                raise DesignInputError(f"The Contig uses an input residue more than once: {segment}")
            copied.add(uid)
            minimum += 1
            maximum += 1
            if maximum > MAX_PREVIEW_DESIGN_TOKENS:
                raise DesignInputError("This preview supports Contigs of at most 4096 residues.")
            residues.append(_source_residue_record(
                atoms, uid, layout_min=minimum, layout_max=maximum,
            ))
    if not residues:
        raise DesignInputError("The Contig must include at least one input Motif residue.")
    modalities = {record["modality"] for record in residues}
    if len(modalities) != 1:
        raise DesignInputError("A Motif input chain must contain one design modality.")
    return {"chain": chain, "token": "/".join(segments),
            "minimum_length": minimum, "maximum_length": maximum,
            "copied_count": len(residues), "residues": residues}


def build_motif_condition(
    atoms: struc.AtomArray, chain: str, token: str,
    choices: Mapping[str, str | Sequence[str] | None] | None = None,
) -> dict[str, Any]:
    """Build condition.contig from per-residue selections, preserving input identity."""
    info = inspect_contig(atoms, chain, token)
    if choices is None:
        choices = {}
    if not isinstance(choices, Mapping):
        raise DesignInputError("Specify Motif atom selections by residue.")
    copied_uids = {record["uid"] for record in info["residues"]}
    canonical = {record["uid"]: record["selector"] for record in info["residues"]}
    selected_uids: set[int] = set()
    partial: dict[str, list[str]] = {}
    for selector, choice in choices.items():
        uid, _ = _source_residue(atoms, selector)
        if uid not in copied_uids:
            raise DesignInputError(f"The atom selection is outside the current Contig: {selector}")
        if uid in selected_uids:
            raise DesignInputError(f"The same Motif residue was configured more than once: {selector}")
        selected_uids.add(uid)
        names = motif_atom_names(atoms, selector, choice)
        if names is not None:
            partial[canonical[uid]] = names
    return {"contig": {chain: {"token": info["token"], "polymer_atom": partial or None}}}


def _source_identity_proofs(atoms: struc.AtomArray) -> dict[int, Any]:
    """Preserve component names using the exact input heavy-atom graph.

    Export the supplied graph without ligand reconstruction. Ambiguous atom
    names or unknown bond orders are rejected.
    """
    result = {}
    for uid in np.unique(atoms.component_uid[atoms.mol_type == "ligand"]):
        rows = np.flatnonzero(atoms.component_uid == uid)
        signature = compute_ligand_component_signature(atoms, rows)
        if signature is None:
            name = str(atoms.res_name[rows[0]])
            raise StructureWriteError(f"Cannot preserve input ligand {name!r}: atom names or bond orders are ambiguous.")
        result[int(uid)] = build_validated_ligand_identity(
            str(atoms.res_name[rows[0]]), signature[0], signature[1],
        )
    return result


def _bond_signature(atoms: struc.AtomArray) -> set[tuple[int, int, int]]:
    if atoms.bonds is None:
        return set()
    return {(min(int(a), int(b)), max(int(a), int(b)), int(order))
            for a, b, order in atoms.bonds.as_array()}


def save_input(atoms: struc.AtomArray, path: str | Path) -> Path:
    """Save a cleaned input as mmCIF and verify its heavy-atom graph.

    Preserve author numbering, insertion codes and label chain IDs. Reload the
    returned file for both interactive selections and the task input.
    """
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".cif":
        raise ValueError("Save the input as .cif to preserve its topology and numbering.")
    if not len(atoms):
        raise DesignInputError("An empty input structure cannot be saved.")
    if not np.asarray(atoms.coord_valid, dtype=bool).all():
        raise DesignInputError("Every retained input atom must have observed coordinates.")
    exported = atoms.copy()
    exported.identity_condition[:] = True
    proofs = _source_identity_proofs(exported)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix=".cif", dir=output.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        write_cif(exported, temporary_path, entry_id=output.stem,
                  validated_ligand_identities=proofs)
        restored = read_structure(temporary_path)
        if len(restored) != len(exported):
            raise StructureWriteError("The atom count changed when reading back the saved input.")
        for field in ("atom_name", "res_name", "element", "mol_type", "label_asym_id"):
            if not np.array_equal(getattr(exported, field), getattr(restored, field)):
                raise StructureWriteError(f"The saved input changed the {field} field.")
        for field in ("auth_asym_id", "auth_seq_id", "auth_ins_code"):
            # A missing source author value is allowed to acquire the native
            # writer's canonical locator; only concrete locators can be lost.
            retained = ((np.asarray(exported.source_format) != "")
                        & ~np.isin(getattr(exported, field), ("", ".", "?")))
            if not np.array_equal(getattr(exported, field)[retained], getattr(restored, field)[retained]):
                raise StructureWriteError(f"The saved input changed the author field {field}.")
        if not np.allclose(exported.coord, restored.coord, atol=2e-6, rtol=0):
            raise StructureWriteError("The saved input changed the coordinates.")
        if _bond_signature(exported) != _bond_signature(restored):
            raise StructureWriteError("The saved input changed the supplied bond graph.")
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


_EXAMPLE_GROUPS = {
    "protein": "Protein", "peptide": "Peptide", "nucleic_acids": "Nucleic acids",
    "ligand": "Ligand", "antibody": "Antibody", "motif": "Motif",
    "protein_ligand_additional": "Additional protein-ligand",
}
_EXAMPLE_CRITERION_FILES = frozenset({"custom_presets.yaml"})


def list_example_tasks(repo_root: str | Path) -> list[dict[str, Any]]:
    """List the native runnable examples, excluding validation criteria YAML.

    Payloads are independent copies of the complete task mappings. Source
    paths retain their native repository-relative form until preparation.
    """
    root = Path(repo_root).expanduser().resolve()
    directory = root / "configs" / "examples"
    if not directory.is_dir():
        raise DesignInputError(f"The native examples directory is absent: {directory}")
    result = []
    for path in sorted(directory.glob("*.yaml")):
        if path.name in _EXAMPLE_CRITERION_FILES:
            continue
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise DesignConfigError(f"Example file must contain named tasks: {path}")
        group = _EXAMPLE_GROUPS.get(path.stem, path.stem.replace("_", " ").capitalize())
        for name, payload in document.items():
            if not isinstance(name, str) or not isinstance(payload, Mapping):
                raise DesignConfigError(f"Invalid named example task in {path}")
            human_name = name.replace("_", " ")
            result.append({
                "key": f"{path.stem}/{name}", "label": f"{group}: {human_name}",
                "group": group, "name": name, "path": path,
                "payload": deepcopy(dict(payload)),
            })
    return result


def _saved_source_mapping(
    source: struc.AtomArray, saved: struc.AtomArray,
) -> tuple[dict[int, int], dict[int, int]]:
    """Map source residue UIDs through save_input's verified atom-row order."""
    if len(source) != len(saved):
        raise DesignInputError("The saved example source changed its atom inventory.")
    source_to_saved: dict[int, int] = {}
    saved_to_source: dict[int, int] = {}
    saved_rows: dict[int, int] = {}
    for row, (raw_source, raw_saved) in enumerate(zip(
        source.residue_uid, saved.residue_uid, strict=True,
    )):
        original, restored = int(raw_source), int(raw_saved)
        if (source_to_saved.setdefault(original, restored) != restored
                or saved_to_source.setdefault(restored, original) != original):
            raise DesignInputError("An example source residue became ambiguous after saving.")
        saved_rows.setdefault(restored, row)
    return source_to_saved, saved_rows


def _saved_source_locator(atoms: struc.AtomArray, uid: int, row: int) -> str:
    chain = str(atoms.chain_id[row])
    prefix = chain if re.fullmatch(r"[A-Za-z]", chain) else f"[{chain}]"
    try:
        number = int(atoms.label_seq_id[row])
    except ValueError:
        number = int(atoms.res_id[row])
    insertion = str(atoms.ins_code[row])
    if insertion in {".", "?"}:
        insertion = ""
    selector = f"{prefix}{number}{insertion}"
    if number <= 0 or source_selector(atoms, selector) != [uid]:
        raise DesignInputError(f"The saved example residue has an ambiguous selector: {selector}")
    return selector


def _remap_example_source_selectors(
    payload: dict[str, Any], task: Any, source: struc.AtomArray,
    saved: struc.AtomArray,
) -> None:
    """Translate only source-addressed fields; final design positions stay native."""
    uid_mapping, saved_rows = _saved_source_mapping(source, saved)

    def translate(text: str, *, contig: bool = False) -> str:
        if not isinstance(text, str) or not text.strip():
            raise DesignInputError("An example source selector must be a nonempty string.")
        terms: list[str] = []
        for raw_term in text.split("/"):
            term = raw_term.strip()
            match = _SOURCE_SEGMENT.fullmatch(term)
            if contig and match is None:
                try:
                    _contig_gap_range(term)
                except (ValueError, OverflowError) as exc:
                    raise DesignInputError(f"Unsupported example Contig segment: {term!r}") from exc
                terms.append(term)
                continue
            original_uids = source_selector(source, term)
            target_uids = [uid_mapping[int(uid)] for uid in original_uids]
            # A whole-chain selector survives normalization without expansion,
            # including chains containing more than one component type.
            if match is None and source_selector(saved, term) == target_uids:
                terms.append(term)
                continue
            locators = [
                _saved_source_locator(saved, uid, saved_rows[uid]) for uid in target_uids
            ]
            candidate = locators[0]
            if len(locators) > 1:
                last = _SOURCE_SEGMENT.fullmatch(locators[-1])
                first = _SOURCE_SEGMENT.fullmatch(locators[0])
                assert first is not None and last is not None
                if (first.group(1) or first.group(2)) == (last.group(1) or last.group(2)):
                    candidate += f"-{last.group(3)}{last.group(4)}"
                else:
                    candidate = "/".join(locators)
            if source_selector(saved, candidate) != target_uids:
                candidate = "/".join(locators)
            if source_selector(saved, candidate) != target_uids:
                raise DesignInputError(f"The example source selector could not be preserved: {text!r}")
            terms.append(candidate)
        return "/".join(terms)

    source_fields = payload.setdefault("antigen", {}) if task.format == "antibody" else payload
    if source_fields.get("hotspot") is not None:
        source_fields["hotspot"] = translate(source_fields["hotspot"])
    crop = source_fields.get("crop")
    if isinstance(crop, Mapping):
        for field in ("include", "exclude"):
            if field in crop:
                crop[field] = [translate(value) for value in crop[field]]
        for entry in crop.get("proximity", []):
            entry["anchor"] = translate(entry["anchor"])
    condition = payload.get("condition") or {}
    for entry in (condition.get("contig") or {}).values():
        entry["token"] = translate(entry["token"], contig=True)
        if entry.get("polymer_atom") is not None:
            remapped = {}
            for selector, atoms in entry["polymer_atom"].items():
                target = translate(selector)
                if target in remapped:
                    raise DesignInputError("Example polymer_atom selectors address the same saved residue.")
                remapped[target] = atoms
            entry["polymer_atom"] = remapped
    design_chains = {chain.chain_id for chain in task.design_chains}
    for declaration in condition.get("bond", []) or []:
        endpoints = [declaration.get("atom1"), declaration.get("atom2")]
        endpoints.extend(declaration.get("omit_atoms", []) or [])
        for endpoint in endpoints:
            if not isinstance(endpoint, Mapping) or endpoint.get("chain") in design_chains:
                continue
            row = _resolve_atom_selector(source, endpoint, where="example bond")
            if str(source.mol_type[row]) == "ligand":
                continue
            uid = uid_mapping[int(source.residue_uid[row])]
            restored_row = saved_rows[uid]
            endpoint["resid"] = int(saved.label_seq_id[restored_row])
            # Preserve an explicit insertion selector, including an empty one.
            if "ins_code" in endpoint:
                endpoint["ins_code"] = str(saved.ins_code[restored_row])


def prepare_example_task(
    repo_root: str | Path, key: str, output_dir: str | Path,
) -> dict[str, Any]:
    """Prepare one complete native preset against a verified saved source CIF.

    Use the native source preparation to retain declared peptide modalities.
    Save the full pre-crop source and translate source selectors so the original
    crop, hotspots, copied Motif identities and final design conditions survive.
    """
    root = Path(repo_root).expanduser().resolve()
    entry = next((entry for entry in list_example_tasks(root) if entry["key"] == key), None)
    if entry is None:
        raise DesignConfigError(f"Unknown native example task: {key!r}")
    task, document = _task_document(entry["name"], entry["payload"], root)
    payload = yaml.safe_load(document)[entry["name"]]
    atoms = None
    input_file = None
    if task.input_path is not None:
        prepared = prepare_task_source(task)
        source = prepared.asym_order_source
        if source is None:
            raise DesignInputError("The native example source could not be prepared.")
        input_file = save_input(source, Path(output_dir) / "source.cif")
        atoms = load_input(input_file)
        _remap_example_source_selectors(payload, task, source, atoms)
        source_fields = payload["antigen"] if task.format == "antibody" else payload
        source_fields["input"] = str(input_file)
    # Validate the complete normalized mapping using the same native schema.
    parse_task(entry["name"], payload, invocation_dir=root)
    return {"name": entry["name"], "payload": payload, "atoms": atoms,
            "input_file": input_file, "format": task.format, "key": key}


def _task_document(task_name: str, payload: Mapping[str, Any], invocation_dir: str | Path):
    invocation = Path(invocation_dir).expanduser().resolve()
    plain = _plain(deepcopy(payload))
    task = parse_task(task_name, plain, invocation_dir=invocation)
    if task.input_path is not None:
        source_fields = plain.setdefault("antigen", {}) if task.format == "antibody" else plain
        source_fields["input"] = str(task.input_path)
    if task.antibody is not None and isinstance(plain["antibody"].get("scaffold"), Mapping):
        plain["antibody"]["scaffold"]["path"] = str(task.antibody.scaffold_path)
    return task, yaml.safe_dump({task_name: plain}, sort_keys=False, allow_unicode=True)


def save_task(
    task_name: str, payload: Mapping[str, Any], path: str | Path, *,
    invocation_dir: str | Path,
) -> Path:
    """Validate and save a YAML file containing one named task."""
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".yaml":
        raise ValueError("The task file must use the .yaml extension.")
    _, document = _task_document(task_name, payload, invocation_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def preview_task(
    task_name: str, payload: Mapping[str, Any], *, invocation_dir: str | Path,
    output_dir: str | Path, context_mode: str = "auto",
) -> dict[str, Any]:
    """Compile representative length plans and preview supplied conditions.

    Check at most eight plans without expanding the full sample budget. The
    preview shows the first checked plan, not a generated structure.
    """
    if context_mode not in {"auto", "rigid", "flex"}:
        raise ValueError("Context mode must be auto, rigid or flex.")
    lengths = payload.get("design_length") if isinstance(payload, Mapping) else None
    if isinstance(lengths, Mapping):
        for value in lengths.values():
            if isinstance(value, Mapping) and set(value) == {"min", "max", "step"}:
                lo, hi, step = (value[key] for key in ("min", "max", "step"))
                if all(type(x) is int for x in (lo, hi, step)) and step > 0:
                    if (hi - lo) // step + 1 > 10_000:
                        raise ValueError("This preview supports at most 10,000 length plans.")
    task, document = _task_document(task_name, payload, invocation_dir)
    prepared = prepare_task_source(task)
    if prepared.source is not None and len(prepared.source) > MAX_PREVIEW_SOURCE_ATOMS:
        raise ValueError("This preview supports at most 100,000 input atoms. Keep fewer chains or residues.")
    ligand_conditions = task.condition.get("ligand", {}) or {}
    ccd_ids = tuple(sorted({str(entry["ccd"]) for entry in ligand_conditions.values()
        if isinstance(entry, Mapping) and entry.get("mode") == "ccd" and "ccd" in entry})) if isinstance(ligand_conditions, Mapping) else ()
    trusted_ccd = None
    if ccd_ids:
        from partidesign.paths import community_asset
        asset = community_asset("esmfold2", Path(__file__).resolve().parents[3]) / "common" / "ccd.pkl"
        if not asset.is_file():
            raise DesignInputError(f"CCD preview requires local chemical definitions: {asset}")
        trusted_ccd = _trusted_ccd_preview(str(asset), ccd_ids, asset.stat().st_mtime_ns)
    plan_indices = sorted(set(np.linspace(
        0, len(task.length_plans) - 1, min(MAX_PREVIEW_PLANS, len(task.length_plans)),
        dtype=int,
    ).tolist()))
    requests_to_check = tuple(ExpandedRequest(
        task=task, plan=task.length_plans[index],
        sample_id=f"sample_{index * task.sample_num:06d}",
        task_slot_ordinal=index * task.sample_num,
        global_ordinal=index * task.sample_num,
    ) for index in plan_indices)
    requests_to_check = resolve_length_distributions(requests_to_check, global_seed=0)
    first = None
    for request in requests_to_check:
        token_bound = 0
        for chain in task.design_chains:
            bound = int(request.plan.lengths.get(chain.chain_id, 0))
            sequence = task.condition.get("sequence", {}) or {}
            full = sequence.get(chain.chain_id) if isinstance(sequence, Mapping) else None
            if isinstance(full, str):
                bound = max(bound, len(re.findall(r"\[[^\]]*\]|.", full)))
            contigs = task.condition.get("contig", {}) or {}
            entry = contigs.get(chain.chain_id) if isinstance(contigs, Mapping) else None
            if isinstance(entry, Mapping) and isinstance(entry.get("token"), str):
                contig_bound = 0
                for segment in entry["token"].split("/"):
                    segment = segment.strip()
                    if re.fullmatch(r"\d+(?:-\d+)?", segment):
                        contig_bound += int(segment.rsplit("-", 1)[-1])
                    elif prepared.source is not None:
                        contig_bound += len(source_selector(prepared.source, segment))
                if type(entry.get("motif_max_length")) is int:
                    contig_bound = min(contig_bound, entry["motif_max_length"])
                bound = max(bound, contig_bound)
            if chain.chain_id in (task.condition.get("backbone", []) or []) and prepared.source is not None:
                rows = prepared.source.chain_id == chain.chain_id
                size = int(rows.sum()) if chain.modality == "ligand" else len(np.unique(prepared.source.residue_uid[rows]))
                bound = max(bound, size)
            token_bound += bound
        if token_bound > MAX_PREVIEW_DESIGN_TOKENS:
            raise ValueError("This preview supports at most 4096 design tokens.")
        compiled = compile_request(request, prepared_source=prepared,
            trusted_ccd_components=trusted_ccd,
            rng=np.random.default_rng(base_request_seed(0, request)))
        if first is None:
            first = compiled
    assert first is not None
    atoms = first.atoms
    used = (atoms.coord_valid & ~atoms.is_design if task.format == "antibody" else
            atoms.coord_valid & (
                ~atoms.is_design | atoms.motif_coord_condition | atoms.token_center_condition
            ))
    input_preview = None
    notes = [
        f"Checked {len(requests_to_check)} representative length plans.",
        "The view shows retained Context and supplied Motif/backbone coordinates. New design positions have no coordinates yet.",
        "No structures have been generated.",
    ]
    if task.format == "antibody":
        notes[1] = ("The view shows supplied antigen Context coordinates. "
                    "Antibody design positions have no generated coordinates yet.")
        if task.antibody.template:
            notes.append("The antibody framework supplies a separate soft geometry template; "
                         "it is not a hard Motif/backbone coordinate condition and is not shown here.")
    if len(plan_indices) < len(task.length_plans):
        notes.append(f"Checked {MAX_PREVIEW_PLANS} of {len(task.length_plans)} length plans.")
    if task.length_distribution_chain:
        notes.append("Pocket-based ligand lengths were resolved for representative requests.")
    if ccd_ids:
        notes.append("CCD ligand chemistry was checked using local definitions.")
    if context_mode == "flex":
        notes.append("Flexible Context is shown in its input conformation.")
    output = Path(output_dir).expanduser().resolve()
    preview_path = output / f"{task.name}_input_preview.cif"
    if used.any():
        output.mkdir(parents=True, exist_ok=True)
        visible = atoms[used]
        from .visualization import condition_b_factors
        visible.set_annotation("b_factor", condition_b_factors(visible))
        # Ligand backbone compilation intentionally copies coordinates only:
        # its element/name fields are unknown. The native compiler copies the
        # source rows in order. Verify that exact correspondence, then decorate
        # this display-only array with the supplied source labels/graph.
        for chain in task.design_chains:
            if chain.modality != "ligand" or chain.chain_id not in (task.condition.get("backbone", []) or []):
                continue
            rows = np.flatnonzero(visible.is_design & (visible.chain_id == chain.chain_id))
            source_rows = (np.empty(0, dtype=int) if prepared.source is None else
                           np.flatnonzero(prepared.source.chain_id == chain.chain_id))
            if (not len(rows) or len(rows) != len(source_rows)
                    or not np.array_equal(visible.coord[rows], prepared.source.coord[source_rows])):
                visible = visible[~(visible.is_design & (visible.chain_id == chain.chain_id))]
                notes.append(f"Ligand backbone {chain.chain_id} could not be mapped to the input for display and was omitted.")
                continue
            reference = prepared.source[source_rows]
            for field in ("res_name", "atom_name", "element", "formal_charge", "hetero",
                          "res_id", "ins_code", "auth_asym_id", "auth_seq_id", "auth_ins_code", "source_format"):
                getattr(visible, field)[rows] = getattr(reference, field)
            edges = _bond_signature(visible)
            for a, b, order in _bond_signature(reference):
                left, right = int(rows[a]), int(rows[b])
                edges.add((min(left, right), max(left, right), order))
            visible.bonds = struc.BondList(len(visible), np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 3))
            notes.append(f"Ligand backbone {chain.chain_id} uses input atom labels and bonds for display only; the coordinate condition does not fix chemistry.")
        # Some coordinate-only conditions deliberately hide identity. Use a
        # separate copy to preserve their source names without changing colors.
        if len(visible):
            visible.identity_condition[:] = True
            write_cif(visible, preview_path, entry_id=f"{task.name}_conditions",
                      validated_ligand_identities=_source_identity_proofs(visible))
            input_preview = preview_path
        else:
            preview_path.unlink(missing_ok=True)
    else:
        preview_path.unlink(missing_ok=True)
        notes.append("No coordinate conditions were supplied, so there is no structure to preview.")
    return {"task": task, "yaml": document, "input_preview": input_preview,
            "request_count": task.sample_num * len(task.length_plans),
            "checked_count": len(requests_to_check), "notes": notes}


__all__ = [
    "fetch_rcsb", "stage_input", "load_input", "chain_summary", "calculate_hotspots", "save_input",
    "inspect_contig", "motif_atom_names", "build_motif_condition", "save_task", "preview_task",
    "list_example_tasks", "prepare_example_task",
]
