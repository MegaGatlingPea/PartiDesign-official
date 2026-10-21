"""CPU-only backbone extraction, US-align scoring, and PX greedy clustering.

Only final design polymer chains contribute: proteins use CA and RNA/DNA use
C3'.  Side chains, covalent linkers outside those traces, and bond/cyclic
chemistry are not separate clustering features.  Each design chain remains a
separate chain for US-align; context is never included.
"""

from __future__ import annotations

from collections.abc import Sequence
import gzip
import math
import os
from pathlib import Path
import re
import string
import subprocess

import numpy as np


_USALIGN_TIMEOUT_SECONDS = 300
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
_TM_SCORE = re.compile(
    r"^\s*TM-score\s*=\s*(\S+)\s+"
    r"\(normalized by length of Structure_([12])\s*:",
    re.MULTILINE,
)


def _read_structure(path: Path):
    # Biotite is the only structural dependency.  In particular, do not import
    # design/validation readers that may initialize model dependencies.
    from biotite.structure.io import pdb, pdbx

    suffixes = [suffix.lower() for suffix in path.suffixes]
    compressed = bool(suffixes and suffixes[-1] == ".gz")
    suffix = suffixes[-2] if compressed and len(suffixes) > 1 else path.suffix.lower()
    opener = gzip.open if compressed else open
    with opener(path, "rt") as handle:
        if suffix in {".cif", ".mmcif"}:
            cif = pdbx.CIFFile.read(handle)
            if "label_asym_id" not in cif.block["atom_site"]:
                raise ValueError(f"{path}: CIF atom_site requires label_asym_id")
            atoms = pdbx.get_structure(
                cif,
                model=1,
                altloc="first",
                use_author_fields=False,
                extra_fields=["label_asym_id"],
                include_bonds=False,
            )
            # AtomArray's default chain_id width is only four characters.
            # Keep the full official label rather than its truncated copy.
            atoms.set_annotation("chain_id", np.asarray(atoms.label_asym_id).astype(str))
            return atoms
        if suffix in {".pdb", ".ent"}:
            return pdb.get_structure(
                pdb.PDBFile.read(handle),
                model=1,
                altloc="first",
                include_bonds=False,
            )
    raise ValueError(f"{path}: expected a CIF or PDB structure")


def prepare_structure(
    structure: Path,
    design_chains: tuple[dict, ...],
    output: Path,
) -> tuple[str, ...]:
    """Write a design-only CA/C3' trace and return its modality composition.

    The result.csv metadata field named "chain" is the final label_asym_id.
    CIF labels, rather than author IDs, select chains.  The returned sorted
    modality tuple is independent of metadata chain order.  Output is CIF, or
    PDB when output ends in .pdb/.ent; private PDB chain IDs are reassigned to
    keep distinct design chains distinct.
    """
    from biotite import structure as struc
    from biotite.structure.io import pdb, pdbx

    structure, output = Path(structure), Path(output)
    if structure.resolve() == output.resolve():
        raise ValueError("alignment output must differ from the source structure")
    if not design_chains:
        raise ValueError("at least one design polymer chain is required")

    declared: list[tuple[str, str]] = []
    expected_counts: dict[str, int] = {}
    seen: set[str] = set()
    for index, entry in enumerate(design_chains):
        if not isinstance(entry, dict):
            raise ValueError(f"design_chains[{index}] must be a mapping")
        chain = entry.get("chain")
        modality = entry.get("modality")
        if not isinstance(chain, str) or not chain.strip():
            raise ValueError(f"design_chains[{index}] requires the final chain label")
        if chain in seen:
            raise ValueError(f"duplicate design chain {chain!r}")
        if modality not in {"protein", "rna", "dna"}:
            raise ValueError(f"design chain {chain!r}: unsupported polymer modality {modality!r}")
        if "residue_count" in entry:
            count = entry["residue_count"]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"design chain {chain!r}: residue_count must be a positive integer")
            expected_counts[chain] = count
        seen.add(chain)
        declared.append((chain, modality))

    atoms = _read_structure(structure)
    labels = np.asarray(atoms.chain_id).astype(str)
    parts = []
    for chain, modality in declared:
        chain_atoms = atoms[labels == chain]
        if not len(chain_atoms):
            raise ValueError(f"{structure}: design chain {chain!r} is absent")
        anchor = "CA" if modality == "protein" else "C3'"
        starts = struc.get_residue_starts(chain_atoms, add_exclusive_stop=True)
        selected = []
        for start, stop in zip(starts[:-1], starts[1:]):
            residue = chain_atoms[start:stop]
            matches = np.flatnonzero(np.asarray(residue.atom_name) == anchor)
            if len(matches) != 1:
                residue_id = f"{int(residue.res_id[0])}{residue.ins_code[0]}"
                raise ValueError(
                    f"{structure}: design chain {chain!r} residue {residue_id} "
                    f"requires exactly one {anchor} atom; found {len(matches)}"
                )
            selected.append(int(start + matches[0]))
        trace = chain_atoms[np.asarray(selected, dtype=int)]
        expected_count = expected_counts.get(chain)
        if expected_count is not None and len(trace) != expected_count:
            raise ValueError(
                f"{structure}: design chain {chain!r} residue_count mismatch: "
                f"expected {expected_count}, found {len(trace)}"
            )
        if not np.all(np.isfinite(trace.coord)):
            raise ValueError(f"{structure}: design chain {chain!r} has non-finite {anchor} coordinates")
        # Modified polymer residues may be HETATM in the final file, but their
        # polymer anchors must still enter US-align's normal ATOM trace.
        trace.hetero[:] = False
        parts.append(trace)

    trace = struc.concatenate(parts)
    suffix = output.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        chain_ids = string.ascii_uppercase + string.ascii_lowercase + string.digits
        if len(parts) > len(chain_ids):
            raise ValueError("PDB alignment output supports at most 62 design chains; use CIF")
        trace.set_annotation(
            "chain_id",
            np.concatenate(
                [np.full(len(part), chain_ids[index]) for index, part in enumerate(parts)]
            ),
        )
        # Source numbering can exceed PDB limits.  Only trace order and chain
        # boundaries matter, so use dense residue IDs in this temporary file.
        trace.res_id = np.concatenate([np.arange(1, len(part) + 1) for part in parts])
        trace.ins_code[:] = ""
        file = pdb.PDBFile()
        pdb.set_structure(file, trace)
    elif suffix in {".cif", ".mmcif"}:
        file = pdbx.CIFFile()
        pdbx.set_structure(file, trace)
    else:
        raise ValueError("alignment output must end in .cif, .mmcif, .pdb, or .ent")
    file.write(output)
    return tuple(sorted(modality for _, modality in declared))


def score_pair(
    first: Path,
    second: Path,
    *,
    executable: str,
    multimer: bool = False,
    molecule_type: str = "auto",
) -> float:
    """Return the mean of US-align's two length-normalized TM-scores.

    A pair requires one successful standard-precision US-align invocation.
    RNA and DNA share US-align's RNA mode.  Mixed polymer complexes use auto.
    Execution and malformed/non-finite/out-of-range results are explicit errors.
    """
    molecule_modes = {
        "auto": "auto",
        "prot": "protein",
        "protein": "protein",
        "rna": "RNA",
        "dna": "RNA",
    }
    if not isinstance(molecule_type, str) or molecule_type.lower() not in molecule_modes:
        raise ValueError(f"unsupported US-align molecule type {molecule_type!r}")
    if not isinstance(executable, str) or not executable.strip():
        raise ValueError("US-align executable must be a non-empty path or command")

    command = [
        executable,
        str(Path(first).resolve()),
        str(Path(second).resolve()),
        "-mol",
        molecule_modes[molecule_type.lower()],
    ]
    if multimer:
        command.extend(["-mm", "1", "-ter", "0", "-split", "2"])
    environment = os.environ.copy()
    environment.update({name: "1" for name in _THREAD_VARIABLES})
    try:
        completed = subprocess.run(
            command,
            shell=False,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            timeout=_USALIGN_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            f"US-align failed for {first} and {second} with exit {exc.returncode}"
            + (f": {detail[-2000:]}" if detail else "")
        ) from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"US-align failed for {first} and {second}: {exc}") from exc

    normalized: dict[str, float] = {}
    for token, structure_id in _TM_SCORE.findall(completed.stdout):
        if structure_id in normalized:
            raise RuntimeError(f"US-align returned duplicate Structure_{structure_id} TM-scores")
        try:
            value = float(token)
        except ValueError as exc:
            raise RuntimeError(f"US-align returned an invalid TM-score {token!r}") from exc
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise RuntimeError(f"US-align returned a TM-score outside finite [0, 1]: {token!r}")
        normalized[structure_id] = value
    if set(normalized) != {"1", "2"}:
        raise RuntimeError("US-align did not return both Structure_1 and Structure_2 TM-scores")
    return (normalized["1"] + normalized["2"]) / 2


def greedy_clusters(
    names: Sequence[str],
    scores: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """PX greedy stars: the largest unassigned inclusive neighborhood wins.

    Ties follow the existing input order.  The center is the first member;
    other members follow input order.  Members need only match their center,
    so a star can contain pairs scoring below the threshold.
    """
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise ValueError("cluster threshold must be a finite number in [0, 1]")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("cluster threshold must be a finite number in [0, 1]")
    matrix = np.asarray(scores, dtype=float)
    size = len(names)
    if matrix.shape != (size, size):
        raise ValueError(f"score matrix must have shape {(size, size)}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("score matrix must contain only finite values in [0, 1]")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
        raise ValueError("score matrix must be symmetric")
    adjacent = matrix >= threshold
    # Each remaining sample is its own neighbor even when the caller leaves
    # the diagonal unset.
    np.fill_diagonal(adjacent, True)
    remaining = np.ones(size, dtype=bool)
    clusters: list[list[int]] = []
    while np.any(remaining):
        candidates = np.flatnonzero(remaining)
        counts = adjacent[np.ix_(candidates, candidates)].sum(axis=1)
        center = int(candidates[np.argmax(counts)])
        members = np.flatnonzero(remaining & adjacent[center])
        clusters.append([center] + [int(index) for index in members if index != center])
        remaining[members] = False
    return clusters
