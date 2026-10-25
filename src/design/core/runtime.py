"""Lightning ``Trainer.predict`` runtime for standalone PartiDesign inference.

The runtime deliberately carries the authoring ``CompiledSample`` beside the
checkpoint-facing canonical tensors.  It does not create a second structure IR,
and it keeps accounting in memory instead of emitting manifests or journals.
"""

from __future__ import annotations

import csv
import copy
import json
import inspect
import logging
import math
import multiprocessing
import re
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from partidesign.paths import community_asset
from typing import Any, TypeAlias

import numpy as np
import torch
from biotite import structure as struc
from rdkit import Chem
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.fabric.utilities.seed import pl_worker_init_function
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler

from .assembly import (
    _head_to_tail_design_protein_asyms, assemble_candidate, normalize_cyclic_product,
)
from .batch import PredictBatch, collate_compiled_samples
from .compiler import compile_request
from .compile_policy import active_token_buckets
from .execution_policy import apply_execution_policy, resolve_execution_policy
from .preflight import (
    PreflightKey, source_interfaces, preflight_key, unique_preflight_requests,
)
from .config import LoadedDesignConfig, validate_sampling_preflight
from .decode import decode_proposal
from ..model.checkpoint import (
    compile_denoiser,
    load_combined_checkpoint,
    resolve_checkpoint_context_constraint_mode,
)
from ..model.routing import resolve_model_route, prepare_routing_sources, task_model_preferences
from ..model.constants import DESIGN_LIGAND_ELEMENTS
from ..model.solver import SimulationState
from ..postproc.runner import (
    PostprocessJob,
    PostprocessPreparationError,
    SampleParameterizationPayload,
    build_parameterization_payload,
    preflight_postprocessing,
    run_postprocessing,
)
from .records import (
    CompiledSample,
    DesignConfigError,
    DesignInputError,
    DesignRunResult,
    DesignSampleError,
    ExpandedRequest,
    SampleFailure,
    SearchCandidate,
    TaskSpec,
)
from .model_tokens import build_model_token_layout
from ..rewards.clash import ClashReward, validate_vdw_elements
from ..rewards.partiscore import PartiScoreReward, ranking_score, reward_record
from ..rewards.esmfold2 import ESMFold2Reward, resolve_sequence_source
from ..rewards.esmfold2_input import (
    _explicit_covalent_edges,
    _ligand_entry,
    candidate_chain_modalities,
)
from ..rewards.esmfold2_runner import FoldRewardResult, run_fold_reward_batch
from ..rewards.af2 import AF2Reward
from ..rewards.af2_runner import run_af2_reward_batch
from ..rewards.mpnn import generate_mpnn_sequences
from .search import FKBranchRejected, run_fk_request, stable_candidate_ordinal
from ..utils.writer import (
    StructureWriteError,
    write_structure_atomic,
)
from ..utils.reader import read_structure
from ..utils.chemistry import TrustedCCDComponent, load_trusted_ccd_components


PreparedItem: TypeAlias = CompiledSample | SampleFailure
TerminalResult: TypeAlias = SearchCandidate | SampleFailure
TerminalCandidateFactory: TypeAlias = Callable[..., Sequence[TerminalResult]]
FKCandidateFactory: TypeAlias = Callable[..., Sequence[TerminalResult]]
FoldRewardRunner: TypeAlias = Callable[..., tuple[tuple[FoldRewardResult, ...], tuple[Any, ...]]]
CheckpointModeResolver: TypeAlias = Callable[..., str]


def _finalize_postprocess_files(
    jobs: Sequence[PostprocessJob],
    results: Mapping[Any, Any],
    tasks: Mapping[str, TaskSpec],
) -> tuple[PostprocessJob, ...]:
    """Promote successful refinements before scoring; never touch failed bases."""
    finalized = []
    for job in jobs:
        if results[job.key].succeeded and tasks[job.task_name].override:
            if job.base_cif == job.refined_cif:
                raise RuntimeError("postprocess staging and base paths must differ")
            if job.base_cif.is_symlink() or job.refined_cif.is_symlink():
                raise RuntimeError("postprocess promotion requires regular output files")
            if not job.base_cif.is_file() or not job.refined_cif.is_file():
                raise RuntimeError("postprocess promotion is missing a completed output file")
            os.replace(job.refined_cif, job.base_cif)
            job = replace(job, refined_cif=job.base_cif)
        finalized.append(job)
    return tuple(finalized)


def _run_deferred_fold_reward_batch(*args: Any, reward_config: Mapping[str, Any], **kwargs: Any):
    target = reward_config.get("_target_")
    if target == "src.design.rewards.af2.AF2Reward":
        return run_af2_reward_batch(*args, reward_config=reward_config, **kwargs)
    if target == "src.design.rewards.esmfold2.ESMFold2Reward":
        return run_fold_reward_batch(*args, reward_config=reward_config, **kwargs)
    if target == "src.design.rewards.af3.AF3Reward":
        from ..rewards.af3_runner import run_af3_reward_batch
        return run_af3_reward_batch(*args, reward_config=reward_config, **kwargs)
    raise DesignConfigError("deferred fold reward target is not implemented")
ResultKey: TypeAlias = tuple[str, str]
FailureWire: TypeAlias = tuple[str, str, str, str]
GroupKey: TypeAlias = tuple[str, tuple[int, ...]]
SlotWire: TypeAlias = tuple[str, str, int, tuple[int, ...]]
CandidateSummary: TypeAlias = tuple[
    str, int, str, tuple[int, ...], float, int, str
]
PostprocessFailureWire: TypeAlias = tuple[str, str, str]
# The last field is the optional compact design ledger.  Keeping the legacy
# fields in their original positions preserves the existing single-chain CSV
# writer and its byte-compatible rows.
ResultRowWire: TypeAlias = tuple[Any, ...]
_POSTPROCESS_CSV_BATCH_SIZE = 32
_POSTPROCESS_CSV_FLUSH_SECONDS = 2.0

_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
_RNA3_TO_1 = {"A": "A", "G": "G", "C": "C", "U": "U"}
_DNA3_TO_1 = {"DA": "A", "DG": "G", "DC": "C", "DT": "T"}
_UNKNOWN_PROXY_TO_1 = {"protein": {"UNK": "X"}, "rna": {"N": "N"}, "dna": {"DN": "N"}}

_DEFERRED_HYDROGEN_ELEMENTS = frozenset({"H", "D", "T"})
_DEFERRED_BOND_NAMES = {
    int(struc.BondType.ANY): "?",
    int(struc.BondType.SINGLE): "single",
    int(struc.BondType.DOUBLE): "double",
    int(struc.BondType.TRIPLE): "triple",
    int(struc.BondType.AROMATIC): "aromatic",
    int(struc.BondType.AROMATIC_SINGLE): "aromatic",
    int(struc.BondType.AROMATIC_DOUBLE): "aromatic",
    int(struc.BondType.AROMATIC_TRIPLE): "aromatic",
}
_DEFERRED_ORDER_TO_BIOTITE = {
    "?": int(struc.BondType.ANY),
    "single": int(struc.BondType.SINGLE),
    "double": int(struc.BondType.DOUBLE),
    "triple": int(struc.BondType.TRIPLE),
    "aromatic": int(struc.BondType.AROMATIC),
}


def _deferred_component_groups(
    atoms: Any,
    chain: str,
) -> list[tuple[int, np.ndarray]]:
    """Return ligand component occurrences in source/file row order."""

    component_uid = np.asarray(atoms.component_uid, dtype=np.int64)
    rows_by_uid: dict[int, list[int]] = {}
    for row in np.flatnonzero(
        (np.asarray(atoms.chain_id).astype(str) == str(chain))
        & (np.asarray(atoms.mol_type).astype(str) == "ligand")
    ):
        uid = int(component_uid[row])
        if uid < 0:
            raise ValueError("deferred fold ligand component UID is negative")
        rows_by_uid.setdefault(uid, []).append(int(row))
    return [
        (uid, np.asarray(rows, dtype=np.int64))
        for uid, rows in sorted(rows_by_uid.items(), key=lambda item: min(item[1]))
    ]


def _deferred_component_signature(
    atoms: Any,
    rows: np.ndarray,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...]]:
    """Build a coordinate-free heavy-atom inventory and complete graph."""

    elements = np.asarray(atoms.element).astype(str)
    heavy = [
        int(row)
        for row in np.asarray(rows, dtype=np.int64)
        if str(elements[int(row)]).strip().upper() not in _DEFERRED_HYDROGEN_ELEMENTS
    ]
    if not heavy:
        raise ValueError("deferred fold trusted ligand has no heavy atoms")
    names = [str(atoms.atom_name[row]).strip() for row in heavy]
    normalized_elements = [str(elements[row]).strip().upper() for row in heavy]
    if (
        any(not name or name in {".", "?"} for name in names)
        or any(not element or element in {".", "?"} for element in normalized_elements)
        or len(names) != len(set(names))
    ):
        raise ValueError("deferred fold ligand atom inventory is not immutable")
    atom_signature = tuple(sorted(zip(names, normalized_elements, strict=True)))
    row_to_name = dict(zip(heavy, names, strict=True))
    heavy_rows = set(heavy)
    edges: dict[tuple[str, str], str] = {}
    if atoms.bonds is not None:
        for raw_left, raw_right, raw_order in atoms.bonds.as_array():
            left, right = int(raw_left), int(raw_right)
            if left not in heavy_rows or right not in heavy_rows:
                continue
            order = _DEFERRED_BOND_NAMES.get(int(raw_order))
            if order is None:
                raise ValueError("deferred fold ligand graph has unsupported bond order")
            edge = tuple(sorted((row_to_name[left], row_to_name[right])))
            if edge[0] == edge[1]:
                raise ValueError("deferred fold ligand graph has a self-edge")
            previous = edges.get(edge)
            if previous is not None and previous != order:
                raise ValueError("deferred fold ligand graph has conflicting bond orders")
            edges[edge] = order
    edge_signature = tuple(
        (left, right, order)
        for (left, right), order in sorted(edges.items())
    )
    return atom_signature, edge_signature



def _restore_deferred_kekule_encoding(
    source: Any,
    source_rows: np.ndarray,
    folded: Any,
    folded_rows: np.ndarray,
) -> bool:
    """Check known Kekule orders and restore only proven counterpart encoding.

    Biotite's AROMATIC_SINGLE/DOUBLE/TRIPLE retain a concrete Kekule order.
    The existing CCD template rule permits the matching plain order; generic
    AROMATIC cannot prove a plain-order counterpart. An already-identical
    aromatic signature keeps its existing generic/subtype behavior, but two
    known conflicting base orders must never hide behind the aromatic label.
    """
    from ..utils.reader import _template_order_matches

    source_signature = _deferred_component_signature(source, source_rows)
    folded_signature = _deferred_component_signature(folded, folded_rows)
    if source_signature[0] != folded_signature[0]:
        return False

    def named_edges(atoms: Any, rows: np.ndarray):
        names = {
            int(row): str(atoms.atom_name[row]).strip()
            for row in rows
            if str(atoms.element[row]).strip().upper()
            not in _DEFERRED_HYDROGEN_ELEMENTS
        }
        # BondList stores one entry per unordered atom-index pair; the
        # signature check above already rejects duplicate heavy atom names.
        edges = {}
        if atoms.bonds is not None:
            for index, (raw_left, raw_right, raw_order) in enumerate(
                atoms.bonds.as_array()
            ):
                left, right = int(raw_left), int(raw_right)
                if left in names and right in names:
                    pair = tuple(sorted((names[left], names[right])))
                    edges[pair] = (index, int(raw_order))
        return edges

    source_edges = named_edges(source, source_rows)
    folded_edges = named_edges(folded, folded_rows)
    if source_edges.keys() != folded_edges.keys():
        return False
    same_signature = source_signature == folded_signature
    generic_aromatic = int(struc.BondType.AROMATIC)
    for pair, (_index, source_order) in source_edges.items():
        folded_order = folded_edges[pair][1]
        if _template_order_matches(source_order, folded_order):
            continue
        # Preserve the pre-existing exact aromatic-signature path when one
        # side has no specific Kekule order. This never enables the fallback
        # from a generic aromatic bond to a plain single/double/triple bond.
        if same_signature and generic_aromatic in (source_order, folded_order):
            continue
        return False
    if same_signature:
        return True

    # Keep coordinates, atom order and external bonds exactly as returned.
    # Restore the source witness encoding so the final writer can revalidate
    # its unchanged CCD identity instead of downgrading it to a local LIG ID.
    bonds = folded.bonds.as_array().copy()
    for pair, (_index, source_order) in source_edges.items():
        bonds[folded_edges[pair][0], 2] = source_order
    folded.bonds = struc.BondList(len(folded), bonds)
    return True

def _deferred_semantic_atom_keys(
    atoms: Any,
    *,
    atom_name_overrides: Mapping[int, str] | None = None,
) -> dict[tuple[str, int, str], int]:
    """Map chain/residue ordinal/atom name without relying on coordinates or UIDs."""

    result: dict[tuple[str, int, str], int] = {}
    chain_values = np.asarray(atoms.chain_id).astype(str)
    res_id = np.asarray(atoms.res_id, dtype=np.int64)
    ins_code = np.asarray(atoms.ins_code).astype(str)
    res_name = np.asarray(atoms.res_name).astype(str)
    component_uid = np.asarray(atoms.component_uid, dtype=np.int64)
    overrides = {} if atom_name_overrides is None else {
        int(uid): str(name) for uid, name in atom_name_overrides.items()
    }
    for chain in sorted(set(chain_values.tolist())):
        chain_rows = np.flatnonzero(chain_values == chain)
        residue_rows: dict[tuple[int, str], list[int]] = {}
        for row in chain_rows:
            residue_rows.setdefault(
                (int(res_id[row]), str(ins_code[row])), []
            ).append(int(row))
        residue_ordinals: dict[tuple[int, str], int] = {}
        for ordinal, residue_key in enumerate(sorted(residue_rows), start=1):
            rows = np.asarray(residue_rows[residue_key], dtype=np.int64)
            if (
                len(set(component_uid[rows].tolist())) != 1
                or len(set(res_name[rows].tolist())) != 1
            ):
                raise ValueError(
                    "deferred fold residue locator is ambiguous: "
                    f"{(str(chain), *residue_key)!r}"
                )
            residue_ordinals[residue_key] = ordinal
        for row in chain_rows:
            residue_key = (int(res_id[row]), str(ins_code[row]))
            atom_name = overrides.get(
                int(atoms.atom_uid[row]), str(atoms.atom_name[row])
            )
            key = (str(chain), residue_ordinals[residue_key], atom_name)
            if key in result:
                raise ValueError(f"deferred fold atom endpoint is ambiguous: {key}")
            result[key] = int(row)
    return result


def _merge_deferred_bond_order(previous: int | None, incoming: int) -> int:
    """Merge a hard edge, keeping concrete orders stronger than ANY."""

    incoming = int(incoming)
    if previous is None:
        return incoming
    previous = int(previous)
    any_order = int(struc.BondType.ANY)
    if previous == any_order:
        return incoming
    if incoming == any_order or previous == incoming:
        return previous
    raise ValueError("deferred fold topology conflicts with authoritative bond order")


def _deferred_source_atom_keys(
    candidate: SearchCandidate, *, canonical_ligand_names: bool = True,
) -> dict[tuple[str, int, str], int]:
    source = candidate.atoms
    source_atom_name_overrides: dict[int, str] = {}
    # ESMFold2 serializes private ligands through SMILES; AF2 retains the
    # source ligand names. Match the naming scheme actually used by the fold.
    ligand_chains = (
        dict.fromkeys(np.asarray(source.chain_id).astype(str).tolist())
        if canonical_ligand_names else ()
    )
    for chain in ligand_chains:
        rows = np.flatnonzero(
            (np.asarray(source.chain_id).astype(str) == chain)
            & (np.asarray(source.mol_type).astype(str) == "ligand")
        )
        if not any(
            int(value) in candidate.private_ligand_molecules
            for value in np.asarray(source.molecule_uid, dtype=np.int64)[rows]
        ):
            continue
        try:
            _payload, atom_map = _ligand_entry(candidate, chain)
        except (KeyError, ValueError):
            # The ESMFold2 path has already validated this ledger before a
            # fold can reach final output.  AF2 may not have one; in that
            # case retain source names and fail closed if an endpoint cannot
            # be resolved below.
            continue
        source_atom_name_overrides.update(
            {int(uid): str(value[1]) for uid, value in atom_map.items()}
        )

    return _deferred_semantic_atom_keys(
        source, atom_name_overrides=source_atom_name_overrides,
    )


def _restore_deferred_fold_output(
    candidate: SearchCandidate,
    folded: Any,
    *,
    canonical_ligand_names: bool = True,
) -> Any:
    """Restore source-faithful identity/topology after a deferred fold.

    The fold parser allocates fresh component/atom UIDs and may collapse
    explicit input topology into its own graph.  This adapter is deliberately
    coordinate-blind: it matches trusted CCD components by chain occurrence
    plus complete heavy-atom graph, then overlays authoritative source edges
    through semantic residue/atom endpoints.  Any missing or changed fact is
    rejected instead of being downgraded to a local ligand name.
    """

    source = candidate.atoms
    identities = dict(candidate.validated_ligand_identities)
    explicit_edges = _explicit_covalent_edges(candidate)
    if (
        not identities
        and not explicit_edges
        and not candidate.private_ligand_molecules
        and not candidate.private_ligand_errors
    ):
        # Keep the byte/annotation path for an ordinary fold untouched.  The
        # restoration adapter is only meaningful when a trusted identity,
        # source-explicit topology fact, or private chemistry sidecar needs to
        # cross the fold UID boundary.
        return folded
    result = copy.deepcopy(folded)
    source_uids = np.asarray(source.component_uid, dtype=np.int64)
    folded_uids = np.asarray(result.component_uid, dtype=np.int64)
    target_for_source: dict[int, int] = {}

    for source_uid, identity in sorted(identities.items()):
        source_uid = int(source_uid)
        source_rows = np.flatnonzero(source_uids == source_uid)
        if len(source_rows) == 0:
            raise ValueError(
                f"deferred fold trusted ligand component UID {source_uid} is absent"
            )
        if set(np.asarray(source.mol_type)[source_rows].astype(str)) != {"ligand"}:
            raise ValueError("deferred fold trusted identity references a non-ligand")
        source_chains = set(np.asarray(source.chain_id)[source_rows].astype(str))
        if len(source_chains) != 1:
            raise ValueError("deferred fold trusted ligand spans multiple chains")
        source_signature = _deferred_component_signature(source, source_rows)
        expected_signature = (identity.atom_signature, identity.edge_signature)
        if source_signature != expected_signature:
            raise ValueError(
                f"deferred fold trusted source ligand {identity.component_id!r} changed"
            )
        chain = next(iter(source_chains))
        source_components = _deferred_component_groups(source, chain)
        source_occurrences = [uid for uid, _rows in source_components]
        try:
            occurrence = source_occurrences.index(source_uid)
        except ValueError as exc:
            raise ValueError("deferred fold trusted ligand occurrence is absent") from exc
        folded_components = _deferred_component_groups(result, chain)
        if occurrence >= len(folded_components):
            raise ValueError(
                f"deferred fold omitted trusted ligand {identity.component_id!r}"
            )
        target_uid, target_rows = folded_components[occurrence]
        if target_uid in target_for_source.values():
            raise ValueError("deferred fold trusted ligand occurrence is ambiguous")
        valid_encoding = _restore_deferred_kekule_encoding(
            source, source_rows, result, target_rows,
        )
        if (
            not valid_encoding
            or _deferred_component_signature(result, target_rows) != expected_signature
        ):
            raise ValueError(
                f"deferred fold changed trusted ligand {identity.component_id!r}"
            )
        target_for_source[source_uid] = int(target_uid)
        result.res_name[target_rows] = identity.component_id
        # read_structure intentionally initializes condition annotations to
        # false.  A revalidated trusted sidecar needs the source identity fact
        # restored before the final writer checks it.
        result.identity_condition[target_rows] = True

    # Re-key matched components to the source UIDs expected by the immutable
    # sidecar.  Preserve all other folded components, allocating fresh IDs on
    # collision rather than allowing a polymer/ligand UID alias.
    all_rows_by_uid: dict[int, list[int]] = {}
    for row, raw_uid in enumerate(np.asarray(result.component_uid, dtype=np.int64)):
        uid = int(raw_uid)
        if uid < 0:
            raise ValueError("deferred fold component UID is negative")
        all_rows_by_uid.setdefault(uid, []).append(int(row))
    folded_groups = [
        (uid, np.asarray(rows, dtype=np.int64))
        for uid, rows in sorted(all_rows_by_uid.items(), key=lambda item: min(item[1]))
    ]
    target_to_source = {target: source for source, target in target_for_source.items()}
    reserved = set(target_for_source)
    used: set[int] = set()
    next_uid = max(
        [*reserved, *(int(value) for value in folded_uids.tolist()), -1]
    ) + 1
    reassigned = folded_uids.copy()
    for target_uid, target_rows in folded_groups:
        if target_uid in target_to_source:
            new_uid = int(target_to_source[target_uid])
        elif target_uid not in reserved and target_uid not in used:
            new_uid = int(target_uid)
        else:
            while next_uid in reserved or next_uid in used:
                next_uid += 1
            new_uid = int(next_uid)
            next_uid += 1
        if new_uid in used:
            raise ValueError("deferred fold component UID remap is ambiguous")
        used.add(new_uid)
        reassigned[target_rows] = new_uid
    result.component_uid = reassigned

    # The fold reader assigns fresh molecule UIDs from file-local identity
    # keys.  Private ligand chemistry is keyed by the source molecule UID, so
    # carry that identity across the same ordered ligand-component occurrences
    # instead of relying on factorization order to happen to match.  A changed
    # ligand inventory is a fold/sample mismatch and must fail closed.
    if "molecule_uid" not in result.get_annotation_categories():
        raise ValueError("deferred fold output lacks molecule UID annotations")
    source_chains = set(np.asarray(source.chain_id).astype(str).tolist())
    folded_chains = set(np.asarray(result.chain_id).astype(str).tolist())
    source_ligand_chains = {
        chain
        for chain in source_chains
        if len(_deferred_component_groups(source, chain))
    }
    folded_ligand_chains = {
        chain
        for chain in folded_chains
        if len(_deferred_component_groups(result, chain))
    }
    if source_ligand_chains != folded_ligand_chains:
        raise ValueError("deferred fold changed the ligand chain inventory")
    remapped_molecule_uids = np.asarray(result.molecule_uid, dtype=np.int64).copy()
    for chain in sorted(source_ligand_chains):
        source_groups = _deferred_component_groups(source, chain)
        folded_groups = _deferred_component_groups(result, chain)
        if len(source_groups) != len(folded_groups):
            raise ValueError(
                f"deferred fold changed ligand component count on chain {chain!r}"
            )
        for (_source_component, source_rows), (
            _folded_component,
            folded_rows,
        ) in zip(source_groups, folded_groups, strict=True):
            source_molecules = np.unique(
                np.asarray(source.molecule_uid, dtype=np.int64)[source_rows]
            )
            folded_molecules = np.unique(
                np.asarray(result.molecule_uid, dtype=np.int64)[folded_rows]
            )
            if len(source_molecules) != 1 or int(source_molecules[0]) < 0:
                raise ValueError(
                    "deferred fold source ligand component has ambiguous molecule UID"
                )
            if len(folded_molecules) != 1 or int(folded_molecules[0]) < 0:
                raise ValueError(
                    "deferred fold ligand component has ambiguous molecule UID"
                )
            remapped_molecule_uids[folded_rows] = int(source_molecules[0])
    result.molecule_uid = remapped_molecule_uids

    source_key_map = _deferred_source_atom_keys(
        candidate, canonical_ligand_names=canonical_ligand_names,
    )
    folded_key_map = _deferred_semantic_atom_keys(result)

    # Private RDKit atom indices follow source ligand rows.  A SMILES fold
    # serializes the same atoms in a different order; restore that row contract
    # through the input's explicit name ledger, keeping predicted coordinates
    # and chemistry attached to their atoms.  AtomArray slicing remaps bonds.
    source_key_by_row = {row: key for key, row in source_key_map.items()}
    row_order = np.arange(len(result), dtype=np.int64)
    for molecule_uid in candidate.private_ligand_molecules:
        source_rows = np.flatnonzero(
            np.asarray(source.molecule_uid, dtype=np.int64) == int(molecule_uid)
        )
        folded_rows = np.flatnonzero(
            np.asarray(result.molecule_uid, dtype=np.int64) == int(molecule_uid)
        )
        try:
            matched_rows = [
                folded_key_map[source_key_by_row[int(row)]] for row in source_rows
            ]
        except KeyError as exc:
            raise ValueError(
                "deferred fold omitted a private ligand atom endpoint"
            ) from exc
        if not len(source_rows) or set(matched_rows) != set(folded_rows.tolist()):
            raise ValueError("deferred fold changed private ligand atom inventory")
        row_order[folded_rows] = matched_rows
    if not np.array_equal(row_order, np.arange(len(result))):
        result = result[row_order]
        folded_key_map = _deferred_semantic_atom_keys(result)
    source_uid_to_row = {int(uid): row for row, uid in enumerate(source.atom_uid)}
    existing: dict[tuple[int, int], int] = {}
    if result.bonds is not None:
        for raw_left, raw_right, raw_order in result.bonds.as_array():
            left, right = sorted((int(raw_left), int(raw_right)))
            existing[(left, right)] = _merge_deferred_bond_order(
                existing.get((left, right)), int(raw_order)
            )

    for uid_a, uid_b, order_name in explicit_edges:
        try:
            source_row_a = source_uid_to_row[int(uid_a)]
            source_row_b = source_uid_to_row[int(uid_b)]
        except KeyError as exc:
            raise ValueError("deferred fold topology endpoint is absent from source") from exc
        source_key_a = next(
            key for key, row in source_key_map.items() if row == source_row_a
        )
        source_key_b = next(
            key for key, row in source_key_map.items() if row == source_row_b
        )
        try:
            folded_row_a = folded_key_map[source_key_a]
            folded_row_b = folded_key_map[source_key_b]
        except KeyError as exc:
            raise ValueError(
                "deferred fold omitted an authoritative topology endpoint"
            ) from exc
        pair = tuple(sorted((int(folded_row_a), int(folded_row_b))))
        desired = _DEFERRED_ORDER_TO_BIOTITE[str(order_name).strip().lower()]
        if desired == int(struc.BondType.ANY):
            # ANY is existence-only: retain a concrete model/CCD edge when
            # present, but materialize the required edge if it is absent.
            existing[pair] = _merge_deferred_bond_order(existing.get(pair), desired)
        else:
            # A concrete source fact is authoritative over a fold/parser
            # prediction, including a conflicting concrete order.
            existing[pair] = desired

    result.bonds = struc.BondList(
        len(result),
        np.asarray(
            [(*pair, order) for pair, order in sorted(existing.items())],
            dtype=np.uint32,
        ).reshape((-1, 3)),
    )
    return result


def _restore_fold_source_partners(source: Any, folded: Any) -> Any:
    """Carry source ownership across sequence changes without matching sidechain atoms."""
    def residues(atoms: Any) -> dict[str, list[np.ndarray]]:
        chains = np.asarray(atoms.chain_id).astype(str)
        output = {}
        for chain in dict.fromkeys(chains.tolist()):
            groups: dict[tuple[int, str], list[int]] = {}
            for row in np.flatnonzero(chains == chain):
                groups.setdefault((int(atoms.res_id[row]), str(atoms.ins_code[row])), []).append(int(row))
            output[chain] = [np.asarray(rows, dtype=np.int64) for _key, rows in sorted(groups.items())]
        return output

    source_groups, fold_groups = residues(source), residues(folded)
    if set(source_groups) != set(fold_groups):
        raise ValueError("fold source-partner mapping changed the source chain set")
    source_ids = np.asarray(source.source_mol_id)
    if not np.issubdtype(source_ids.dtype, np.integer) or bool((source_ids < 0).any()):
        raise ValueError("fold source-partner IDs must be nonnegative integers")
    mapped = np.empty(len(folded), dtype=np.int64)
    for chain, groups in source_groups.items():
        if len(groups) != len(fold_groups[chain]):
            raise ValueError("fold source-partner mapping changed the source residue count")
        for source_rows, fold_rows in zip(groups, fold_groups[chain], strict=True):
            ids = np.unique(source_ids[source_rows])
            if len(ids) != 1:
                raise ValueError("fold source-partner mapping is ambiguous within a residue")
            mapped[fold_rows] = int(ids[0])
    result = folded.copy()
    result.set_annotation("source_mol_id", mapped)
    return result


def _restore_fold_design_state(candidate: SearchCandidate, folded: Any) -> Any:
    """Restore task design ownership after a fold reader rebuilds annotations.

    ``read_structure`` intentionally returns source-neutral atoms, so a folded
    structure otherwise loses the ``is_design`` mask required by the private
    whole-peptide payload route.  Match source residues by chain/author residue
    identity where possible and use the task's declared design chains for fold
    residues that have no source counterpart.
    """

    source = candidate.atoms
    design_chains = {
        str(chain.chain_id) for chain in candidate.base_request.task.design_chains
    }
    source_flags: dict[tuple[str, int, str], bool] = {}
    source_chains = np.asarray(source.chain_id).astype(str)
    source_res_ids = np.asarray(source.res_id, dtype=np.int64)
    source_ins = np.asarray(source.ins_code).astype(str)
    source_design = np.asarray(source.is_design, dtype=bool)
    for row in range(len(source)):
        key = (str(source_chains[row]), int(source_res_ids[row]), str(source_ins[row]))
        flag = bool(source_design[row])
        previous = source_flags.get(key)
        if previous is not None and previous != flag:
            raise ValueError("fold design ownership is inconsistent within a source residue")
        source_flags[key] = flag

    result = folded.copy()
    folded_chains = np.asarray(result.chain_id).astype(str)
    folded_res_ids = np.asarray(result.res_id, dtype=np.int64)
    folded_ins = np.asarray(result.ins_code).astype(str)
    restored = np.asarray(
        [
            source_flags.get(
                (str(chain), int(res_id), str(ins)),
                str(chain) in design_chains,
            )
            for chain, res_id, ins in zip(
                folded_chains, folded_res_ids, folded_ins, strict=True
            )
        ],
        dtype=bool,
    )
    result.is_design = restored
    return result


def _request_key(request: ExpandedRequest) -> ResultKey:
    return request.task.name, request.sample_id


def _result_key(result: TerminalResult) -> ResultKey:
    if isinstance(result, SearchCandidate):
        return _request_key(result.base_request)
    return result.task_name, result.sample_id


def _primary_design_identity(request: ExpandedRequest) -> tuple[str, str]:
    chains = request.task.design_chains
    if len(chains) != 1:
        return "", "joint"
    return chains[0].chain_id, chains[0].modality


def _is_joint_request(request: ExpandedRequest) -> bool:
    return len(request.task.design_chains) > 1


def _design_rows(
    atoms: Any,
    chain_id: str,
    *,
    design_only: bool = True,
) -> np.ndarray:
    chain_mask = np.asarray(atoms.chain_id).astype(str) == str(chain_id)
    if not design_only:
        return np.flatnonzero(chain_mask)
    return np.flatnonzero(chain_mask & np.asarray(atoms.is_design, dtype=bool))


def _canonical_ligand_smiles(molecule: Any) -> str:
    molecule = Chem.RemoveHs(Chem.Mol(molecule), sanitize=True)
    if molecule.GetNumAtoms() == 0 or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("design ligand graph is disconnected")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _canonical_condition_smiles(task: TaskSpec, chain_id: str) -> str | None:
    condition = task.condition
    ligand = condition.get("ligand", {}) if isinstance(condition, Mapping) else {}
    entry = ligand.get(chain_id) if isinstance(ligand, Mapping) else None
    raw = entry.get("smiles") if isinstance(entry, Mapping) else None
    if not isinstance(raw, str):
        return None
    try:
        molecule = Chem.MolFromSmiles(raw)
        if molecule is None:
            return None
        return _canonical_ligand_smiles(molecule)
    except (TypeError, ValueError):
        return None


def _ligand_molecule_for_owner(
    atoms: Any,
    rows: np.ndarray,
    *,
    private_ligand_molecules: Mapping[int, Any] | None = None,
    trusted_ligand_components: Mapping[int, TrustedCCDComponent] | None = None,
) -> Any | None:
    molecule_uids = list(dict.fromkeys(int(value) for value in atoms.molecule_uid[rows]))
    if len(molecule_uids) != 1:
        raise ValueError("design ligand must be one connected molecule")
    molecule_uid = molecule_uids[0]
    molecule = (
        None
        if private_ligand_molecules is None
        else private_ligand_molecules.get(molecule_uid)
    )
    if molecule is not None:
        return molecule
    component = (
        None
        if trusted_ligand_components is None
        else trusted_ligand_components.get(molecule_uid)
    )
    if component is None:
        return None
    try:
        return Chem.Mol(component.rdkit_binary)
    except Exception as exc:
        raise ValueError("trusted ligand private chemistry cannot be decoded") from exc


def _sequence_from_atoms(
    atoms: Any,
    *,
    chain_id: str,
    modality: str,
    model_parent_residue: Mapping[int, str] | None = None,
    fixed_modified_residue_names: Mapping[int, str] | None = None,
    private_ligand_molecules: Mapping[int, Any] | None = None,
    private_ligand_errors: Mapping[int, str] | None = None,
    trusted_ligand_components: Mapping[int, TrustedCCDComponent] | None = None,
    fallback_smiles: str | None = None,
    strict_ligand: bool = False,
    design_only: bool = True,
) -> str:
    rows = _design_rows(atoms, chain_id, design_only=design_only)
    if not len(rows):
        raise ValueError("design output chain is absent from candidate")
    model_parent_residue = {} if model_parent_residue is None else model_parent_residue
    fixed_modified_residue_names = (
        {} if fixed_modified_residue_names is None else fixed_modified_residue_names
    )
    if modality == "ligand":
        molecule = _ligand_molecule_for_owner(
            atoms,
            rows,
            private_ligand_molecules=private_ligand_molecules,
            trusted_ligand_components=trusted_ligand_components,
        )
        if molecule is not None:
            return _canonical_ligand_smiles(molecule)
        molecule_uids = list(dict.fromkeys(int(value) for value in atoms.molecule_uid[rows]))
        private_error = (
            None
            if private_ligand_errors is None or len(molecule_uids) != 1
            else private_ligand_errors.get(molecule_uids[0])
        )
        if strict_ligand:
            if private_error is not None:
                raise ValueError(f"design ligand private chemistry failed: {private_error}")
            raise ValueError("design ligand lacks its sanitized private molecule")
        return "" if fallback_smiles is None else str(fallback_smiles)
    mapping = (
        _AA3_TO_1
        if modality == "protein"
        else _RNA3_TO_1
        if modality == "rna"
        else _DNA3_TO_1
    )
    seen: set[int] = set()
    sequence: list[str] = []
    for row in rows:
        token_uid = int(atoms.token_uid[row])
        if token_uid in seen:
            continue
        seen.add(token_uid)
        residue_uid = int(atoms.residue_uid[row])
        fixed_modified = fixed_modified_residue_names.get(residue_uid)
        if fixed_modified is not None:
            sequence.append(f"[{fixed_modified}]")
            continue
        residue = str(atoms.res_name[row])
        model_residue = model_parent_residue.get(residue_uid, residue)
        one_letter = mapping.get(model_residue)
        if one_letter is None:
            one_letter = _UNKNOWN_PROXY_TO_1[modality].get(model_residue)
        if one_letter is None:
            raise ValueError(
                f"design {modality} sequence contains unsupported residue {residue!r}"
            )
        sequence.append(one_letter)
    return "".join(sequence)


def _candidate_design_sequence(
    candidate: SearchCandidate,
    *,
    chain_id: str,
    modality: str,
) -> str:
    return _sequence_from_atoms(
        candidate.atoms,
        chain_id=chain_id,
        modality=modality,
        model_parent_residue=candidate.model_parent_residue,
        fixed_modified_residue_names=candidate.fixed_modified_residue_names,
        private_ligand_molecules=candidate.private_ligand_molecules,
        private_ligand_errors=candidate.private_ligand_errors,
        strict_ligand=True,
        design_only=False,
    )


def _motif_output_fields(
    owner: CompiledSample | SearchCandidate,
    chain_id: str,
    modality: str,
) -> tuple[str, str]:
    """Return compiler-resolved motif metadata for one final design chain.

    In particular, this must not reinterpret the task YAML: contig gap lengths
    are sampled during compilation and source selectors are remapped to final
    output-token positions there.  Ligand motif metadata has a separate graph
    contract and deliberately stays blank in this CSV schema.
    """

    if modality == "ligand":
        return "", ""
    fields = owner.motif_output_fields.get(chain_id)
    if fields is None:
        return "", ""
    return fields.contig_string, fields.tip_atoms


def _design_counts(
    owner: CompiledSample,
    chain_id: str,
    layout: Any | None = None,
) -> tuple[int, int]:
    rows = _design_rows(owner.atoms, chain_id)
    if not len(rows):
        raise DesignSampleError(
            f"compiled design chain {chain_id!r} has no atom inventory"
        )
    authoring = len(np.unique(np.asarray(owner.atoms.token_uid[rows], dtype=np.int64)))
    if layout is None:
        layout = build_model_token_layout(
            owner.atoms,
            owner.modified_polymer_carriers,
        )
    model = len({int(layout.model_token_by_row[int(row)]) for row in rows})
    if authoring <= 0 or model <= 0:
        raise DesignSampleError("design-chain token counts must be positive")
    return int(authoring), int(model)


def _build_design_ledger(
    compiled: CompiledSample,
    candidate: SearchCandidate | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build the ordered, sample-local output ledger.

    ``compiled.atoms`` is the source-faithful authoring axis.  A candidate's
    atoms carry the final decoded identities, while the compiler carrier map
    supplies the model-token expansion needed for search grouping and output
    reporting.  The ledger remains a primitive JSON payload at the CSV boundary.
    """

    layout = build_model_token_layout(
        compiled.atoms,
        compiled.modified_polymer_carriers,
    )
    source_fixed = _fixed_modified_residue_names(compiled)
    source_trusted = compiled.trusted_ligand_components
    ledger: list[dict[str, Any]] = []
    for chain in compiled.request.task.design_chains:
        authoring_count, model_count = _design_counts(compiled, chain.chain_id, layout)
        source_smiles = _canonical_condition_smiles(
            compiled.request.task,
            chain.chain_id,
        )
        source_sequence = _sequence_from_atoms(
            compiled.atoms,
            chain_id=chain.chain_id,
            modality=chain.modality,
            model_parent_residue=compiled.model_parent_residue,
            fixed_modified_residue_names=source_fixed,
            trusted_ligand_components=source_trusted,
            fallback_smiles=source_smiles,
        )
        if chain.modality == "ligand" and source_sequence and source_smiles is None:
            source_smiles = source_sequence
        final_sequence = ""
        final_smiles: str | None = None
        if candidate is not None:
            final_smiles = _canonical_condition_smiles(
                compiled.request.task,
                chain.chain_id,
            )
            final_sequence = _sequence_from_atoms(
                candidate.atoms,
                chain_id=chain.chain_id,
                modality=chain.modality,
                model_parent_residue=candidate.model_parent_residue,
                fixed_modified_residue_names=candidate.fixed_modified_residue_names,
                private_ligand_molecules=candidate.private_ligand_molecules,
                private_ligand_errors=candidate.private_ligand_errors,
                fallback_smiles=None,
                design_only=False,
            )
            if chain.modality == "ligand" and final_sequence:
                final_smiles = final_sequence
            if chain.modality == "ligand":
                final_rows = _design_rows(candidate.atoms, chain.chain_id, design_only=False)
                authoring_count = int(len(np.unique(candidate.atoms.token_uid[final_rows])))
        entry: dict[str, Any] = {
            "chain": chain.chain_id,
            "modality": chain.modality,
            "source_sequence": source_sequence,
            "final_sequence": final_sequence,
            "authoring_token_count": authoring_count,
            "model_token_count": model_count,
        }
        if chain.modality == "ligand":
            canonical_smiles = final_smiles or source_smiles
            if canonical_smiles:
                entry["canonical_smiles"] = canonical_smiles
        ledger.append(entry)
    return tuple(ledger)


def _request_design_ledger(request: ExpandedRequest) -> tuple[dict[str, Any], ...]:
    """Provide a deterministic ledger even when compilation failed early."""

    sequence_by_chain = request.task.condition.get("sequence", {})
    if not isinstance(sequence_by_chain, Mapping):
        sequence_by_chain = {}
    result: list[dict[str, Any]] = []
    for chain in request.task.design_chains:
        planned = request.plan.lengths.get(chain.chain_id)
        raw_sequence = sequence_by_chain.get(chain.chain_id)
        source_sequence = raw_sequence.upper() if isinstance(raw_sequence, str) else ""
        if planned is None and isinstance(raw_sequence, str):
            planned = len(raw_sequence)
        if planned is None:
            planned = 0
        entry: dict[str, Any] = {
            "chain": chain.chain_id,
            "modality": chain.modality,
            "source_sequence": source_sequence,
            "final_sequence": "",
            "authoring_token_count": int(planned),
            "model_token_count": int(planned),
        }
        if chain.modality == "ligand":
            smiles = _canonical_condition_smiles(request.task, chain.chain_id)
            if smiles:
                entry["source_sequence"] = smiles
                entry["canonical_smiles"] = smiles
        result.append(entry)
    return tuple(result)


def _ledger_json(entries: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(
        list(entries),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _constraint_output_fields(
    candidate: SearchCandidate,
    output_atoms: Any,
    *,
    output_format: str = "cif",
    canonical_ligand_names: bool = True,
) -> dict[str, str]:
    """Report design constraints using the structure being written.

    This is a constraint ledger, not an inventory or geometry validation of all
    output bonds. Requested any/unknown orders remain distinct from predictions.
    """
    source = candidate.atoms
    design = np.asarray(source.is_design, dtype=bool)
    polymer = np.isin(np.asarray(source.mol_type).astype(str), ("protein", "rna", "dna"))
    fixed_rows = np.flatnonzero(
        design & polymer & np.asarray(source.identity_condition, dtype=bool)
    )
    uid_to_row = {int(uid): row for row, uid in enumerate(source.atom_uid)}
    conditions = []
    for condition in candidate.topology_conditions:
        left = uid_to_row[int(condition.atom_uid_a)]
        right = uid_to_row[int(condition.atom_uid_b)]
        if not (design[left] or design[right]):
            continue
        # Ligand internal chemistry is already reported through SMILES. Only
        # ligand-polymer or inter-residue polymer-polymer constraints belong here.
        if not (polymer[left] or polymer[right]):
            continue
        if (
            polymer[left] and polymer[right]
            and int(source.residue_uid[left]) == int(source.residue_uid[right])
        ):
            continue
        conditions.append(condition)
    if not len(fixed_rows) and not conditions:
        return {"fixed_restypes": "[]", "bond_constraints": "[]"}

    if output_atoms is not source:
        source_keys = _deferred_source_atom_keys(
            candidate, canonical_ligand_names=canonical_ligand_names,
        )
        key_by_row = {row: key for key, row in source_keys.items()}
        final_keys = _deferred_semantic_atom_keys(output_atoms)

    def output_row(row: int) -> int:
        if output_atoms is source:
            return int(row)
        try:
            return final_keys[key_by_row[int(row)]]
        except KeyError as exc:
            raise ValueError("output constraint endpoint is absent from final structure") from exc

    def residue_locator(row: int) -> dict[str, Any]:
        chain = str(output_atoms.chain_id[row])
        if output_format == "cif":
            chain = str(output_atoms.label_asym_id[row]) or chain
        is_polymer = str(output_atoms.mol_type[row]) in {"protein", "rna", "dna"}
        locator: dict[str, Any] = {
            "chain": chain,
            # Nonpolymer mmCIF label_seq_id is '.', not an internal residue ID.
            "res_id": (
                int(output_atoms.res_id[row])
                if output_format != "cif" or is_polymer else None
            ),
        }
        insertion = str(output_atoms.ins_code[row]).strip()
        if insertion:
            locator["ins_code"] = insertion
        return locator

    fixed_by_residue: dict[int, list[int]] = {}
    for row in fixed_rows:
        fixed_by_residue.setdefault(int(source.residue_uid[row]), []).append(int(row))
    fixed = []
    for rows in fixed_by_residue.values():
        row = rows[0]
        anchor_name = "CA" if str(source.mol_type[row]) == "protein" else "C1'"
        anchor = next(
            (index for index in rows if str(source.atom_name[index]) == anchor_name), row
        )
        locator = residue_locator(output_row(anchor))
        locator["restype"] = str(source.res_name[row])
        fixed.append(locator)

    def atom_locator(uid: int) -> dict[str, Any]:
        row = output_row(uid_to_row[int(uid)])
        return {
            **residue_locator(row),
            "atom": str(output_atoms.atom_name[row]),
            # 1-based atom_site.id (or PDB atom serial) disambiguates nonpolymers.
            "atom_id": row + 1,
        }

    bonds = []
    seen_bonds = set()
    for condition in conditions:
        signature = (*condition.canonical_uids(), str(condition.order), bool(condition.required_output))
        if signature in seen_bonds:
            continue
        seen_bonds.add(signature)
        endpoints = sorted(
            (atom_locator(condition.atom_uid_a), atom_locator(condition.atom_uid_b)),
            key=lambda value: value["atom_id"],
        )
        bonds.append({
            "atom1": endpoints[0], "atom2": endpoints[1],
            "order": str(condition.order),
            "required_output": bool(condition.required_output),
        })
    fixed.sort(key=lambda value: (value["chain"], value["res_id"], value.get("ins_code", "")))
    bonds.sort(key=lambda value: (value["atom1"]["atom_id"], value["atom2"]["atom_id"], value["order"]))
    return {
        "fixed_restypes": json.dumps(fixed, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        "bond_constraints": json.dumps(bonds, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
    }


def _candidate_generation_metadata(candidate: SearchCandidate) -> dict[str, Any]:
    """Primitive reporting only; sampled search groups never shrink on salvage."""
    result = dict(candidate.base_request.length_metadata)
    fragments = candidate.ligand_fragment_metadata
    if fragments:
        result["ligand_fragment_details"] = json.dumps(
            {str(uid): dict(record) for uid, record in sorted(fragments.items())},
            sort_keys=True, allow_nan=False, separators=(",", ":"),
        )
        if len(fragments) == 1:
            record = next(iter(fragments.values()))
            for key in ("original_atom_count", "retained_atom_count", "fragment_count", "salvaged"):
                result[f"ligand_{key}"] = record[key]
    return result


def _antibody_output_metadata(candidate, atoms, path, *, fold=None, sequence_source="design", structure_source="design", reward_score=None):
    if candidate.base_request.task.format != "antibody":
        return {}
    from ..antibody.results import build_antibody_result, antibody_result_context
    result = build_antibody_result(candidate, atoms, path, fold=fold)
    result.update(sequence_source=sequence_source, structure_source=structure_source,
                  reward_score=reward_score if reward_score is not None and math.isfinite(reward_score) else None)
    # Primitive in-memory wire data only; no public JSON files or cells.
    return {"_antibody_result": json.dumps(result, allow_nan=False),
            "_antibody_context": json.dumps(antibody_result_context(candidate), allow_nan=False)}



def _write_candidate_output(candidate, atoms, path, *, entry_id, **metadata_options):
    """Publish antibody CIF and its row together, measured after serialization."""
    if getattr(candidate.base_request.task, "format", "normal") != "antibody":
        write_structure_atomic(atoms, path, entry_id=entry_id,
                               validated_ligand_identities=candidate.validated_ligand_identities)
        return {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Metadata errors must not publish a CIF that the CSV reports as failed.
    with tempfile.TemporaryDirectory(prefix=".antibody-output-", dir=path.parent) as work:
        temporary = Path(work) / path.name
        write_structure_atomic(atoms, temporary, entry_id=entry_id,
                               validated_ligand_identities=candidate.validated_ligand_identities)
        final_atoms = read_structure(temporary)
        fields = _antibody_output_metadata(candidate, final_atoms, path, **metadata_options)
        os.replace(temporary, path)
    return fields


def _candidate_fallback_ledger(candidate: SearchCandidate) -> tuple[dict[str, Any], ...]:
    """Best-effort ledger for direct helper use without the compiled carrier map."""

    result: list[dict[str, Any]] = []
    for index, chain in enumerate(candidate.base_request.task.design_chains):
        rows = _design_rows(candidate.atoms, chain.chain_id, design_only=False)
        authoring = len(np.unique(candidate.atoms.token_uid[rows])) if len(rows) else 0
        model = (
            int(candidate.actual_joint_length[index])
            if index < len(candidate.actual_joint_length)
            else authoring
        )
        try:
            sequence = _candidate_design_sequence(
                candidate,
                chain_id=chain.chain_id,
                modality=chain.modality,
            )
        except ValueError:
            sequence = ""
        entry = {
            "chain": chain.chain_id,
            "modality": chain.modality,
            "source_sequence": "",
            "final_sequence": sequence,
            "authoring_token_count": int(authoring),
            "model_token_count": int(model),
        }
        if chain.modality == "ligand" and sequence:
            entry["canonical_smiles"] = sequence
        result.append(entry)
    return tuple(result)


def _output_residue_groups(atoms: Any, rows: np.ndarray) -> list[tuple[tuple[int, str], np.ndarray]]:
    """Group one chain's rows into physical residues in file order."""

    result: list[tuple[tuple[int, str], np.ndarray]] = []
    positions: dict[tuple[int, str], list[int]] = {}
    for raw_row in np.asarray(rows, dtype=np.int64).tolist():
        row = int(raw_row)
        key = (int(atoms.res_id[row]), str(atoms.ins_code[row]))
        positions.setdefault(key, []).append(row)
    for key, group in positions.items():
        result.append((key, np.asarray(group, dtype=np.int64)))
    return result


def _output_residue_count(
    atoms: Any,
    rows: np.ndarray,
    *,
    modality: str,
) -> int:
    """Count physical residues, retaining ligand component boundaries."""

    groups = _output_residue_groups(atoms, rows)
    categories_method = getattr(atoms, "get_annotation_categories", None)
    categories = (
        set(categories_method())
        if callable(categories_method)
        else set()
    )
    if modality != "ligand" or (
        "component_uid" not in categories and not hasattr(atoms, "component_uid")
    ):
        return len(groups)
    component_values = np.asarray(atoms.component_uid, dtype=np.int64)
    locators = {
        (int(component_values[row]), key[0], key[1])
        for key, group in groups
        for row in [int(group[0])]
    }
    return len(locators)


def _output_polymer_sequence(
    candidate: SearchCandidate,
    output_atoms: Any,
    *,
    chain_id: str,
    modality: str,
    output_rows: np.ndarray,
) -> str:
    """Render the sequence represented by the final structure rows.

    Fold readers can replace the authoring residue UID and parent-code a fixed
    CCD residue for model input. Pair source and output residues by the
    restored chain and residue locator, with a same-chain positional fallback
    only after the complete one-to-one residue count is proven. This keeps
    CCD identity attached to its occurrence without guessing across chains.
    """

    mapping = (
        _AA3_TO_1
        if modality == "protein"
        else _RNA3_TO_1
        if modality == "rna"
        else _DNA3_TO_1
    )
    source_chain_rows = np.flatnonzero(
        np.asarray(candidate.atoms.chain_id).astype(str) == str(chain_id)
    )
    source_groups = _output_residue_groups(candidate.atoms, source_chain_rows)
    output_groups = _output_residue_groups(output_atoms, output_rows)
    if not source_groups or len(source_groups) != len(output_groups):
        raise ValueError(
            f"final output residue mapping changed chain {chain_id!r} length"
        )
    source_by_key = {key: group for key, group in source_groups}
    output_keys = [key for key, _group in output_groups]
    source_keys = [key for key, _group in source_groups]
    if len(source_by_key) == len(source_groups) and set(source_keys) == set(output_keys):
        paired = [
            (source_by_key[key], output_group)
            for key, output_group in output_groups
        ]
    else:
        # The fold restoration already proved this exact chain's residue
        # count and chain identity. Pairing within that one chain is therefore
        # a residue occurrence mapping, never an ordinal chain-role guess.
        paired = [
            (source_group[1], output_group)
            for source_group, output_group in zip(source_groups, output_groups, strict=True)
        ]

    sequence: list[str] = []
    for source_rows, output_group in paired:
        source_uids = np.unique(
            np.asarray(candidate.atoms.residue_uid, dtype=np.int64)[source_rows]
        )
        if len(source_uids) != 1:
            raise ValueError(
                f"source residue mapping is ambiguous on chain {chain_id!r}"
            )
        residue_uid = int(source_uids[0])
        output_names = {
            str(value).strip().upper()
            for value in np.asarray(output_atoms.res_name)[output_group].tolist()
        }
        if len(output_names) != 1:
            raise ValueError(
                f"final output residue identity is ambiguous on chain {chain_id!r}"
            )
        output_residue = next(iter(output_names))
        fixed_modified = candidate.fixed_modified_residue_names.get(residue_uid)
        if fixed_modified is not None:
            observed_component = output_residue.strip("[]")
            if observed_component != str(fixed_modified).upper():
                raise ValueError(
                    f"final output CCD identity for residue {residue_uid} changed "
                    f"from {fixed_modified!r} to {output_residue!r}"
                )
            sequence.append(f"[{fixed_modified}]")
            continue
        if output_residue.startswith("[") and output_residue.endswith("]"):
            sequence.append(output_residue)
            continue
        one_letter = mapping.get(output_residue)
        if one_letter is None:
            parent = candidate.model_parent_residue.get(residue_uid)
            if parent is not None:
                one_letter = mapping.get(str(parent).upper())
                if one_letter is None:
                    one_letter = _UNKNOWN_PROXY_TO_1[modality].get(str(parent).upper())
        if one_letter is None:
            one_letter = _UNKNOWN_PROXY_TO_1[modality].get(output_residue)
        if one_letter is None:
            raise ValueError(
                f"final output {modality} sequence contains unsupported residue {output_residue!r}"
            )
        sequence.append(one_letter)
    return "".join(sequence)


def _output_design_chain_rows(
    candidate: SearchCandidate,
    output_atoms: Any,
) -> tuple[dict[str, Any], ...]:
    """Describe every declared design chain in the final structure.

    Task chain IDs identify ownership, while public metadata names the final
    mmCIF label_asym_id. Residue counts use physical residue locators, so
    atomized modified-polymer carriers count as one residue.
    """

    chain_values = np.asarray(output_atoms.chain_id).astype(str)
    labels = np.asarray(
        getattr(output_atoms, "label_asym_id", output_atoms.chain_id)
    ).astype(str)
    declared = tuple(candidate.base_request.task.design_chains)
    declared_ids = [str(chain.chain_id) for chain in declared]

    rows_by_declared: dict[str, np.ndarray] = {}
    for chain_id in declared_ids:
        rows = np.flatnonzero(chain_values == chain_id)
        if len(rows):
            rows_by_declared[chain_id] = rows
    missing = [chain_id for chain_id in declared_ids if chain_id not in rows_by_declared]
    if missing:
        raise ValueError(
            "final output design-chain mapping is incomplete: "
            f"declared={declared_ids!r}, missing={missing!r}"
        )

    result: list[dict[str, Any]] = []
    for chain in declared:
        chain_id = str(chain.chain_id)
        rows = rows_by_declared[chain_id]
        final_labels = list(dict.fromkeys(labels[rows].tolist()))
        if len(final_labels) != 1 or not final_labels[0]:
            raise ValueError(
                f"final output design chain {chain_id!r} has ambiguous label_asym_id"
            )
        residue_groups = _output_residue_groups(output_atoms, rows)
        if not residue_groups:
            raise ValueError(
                f"final output design chain {chain_id!r} has no residues"
            )
        residue_count = _output_residue_count(
            output_atoms,
            rows,
            modality=str(chain.modality),
        )
        if chain.modality == "ligand":
            try:
                sequence = _candidate_design_sequence(
                    candidate,
                    chain_id=chain_id,
                    modality=chain.modality,
                )
            except ValueError:
                # A legacy direct terminal candidate may not carry the private
                # RDKit sidecar; the task condition still supplies a
                # canonical, in-memory SMILES fallback.
                sequence = _canonical_condition_smiles(
                    candidate.base_request.task, chain_id
                ) or ""
                if not sequence:
                    raise
        else:
            sequence = _output_polymer_sequence(
                candidate,
                output_atoms,
                chain_id=chain_id,
                modality=chain.modality,
                output_rows=rows,
            )
        entry = {
            "chain": final_labels[0],
            "modality": str(chain.modality),
            "sequence": sequence,
            "residue_count": residue_count,
        }
        sub_modality = getattr(chain, "sub_modality", None)
        if sub_modality is not None:
            if chain.modality != "protein" or sub_modality != "peptide":
                raise ValueError("invalid output design-chain sub-modality")
            entry["sub_modality"] = sub_modality
        task = candidate.base_request.task
        if getattr(task, "cyclic_offset", False):
            selected = set(getattr(task, "cyclic_offset_chains", ()))
            if not selected:
                selected = {
                    item.chain_id for item in declared if item.modality == "protein"
                }
            if chain_id in selected:
                entry["cyclic_offset"] = True
        result.append(entry)
    return tuple(result)


def _design_chains_json(
    candidate: SearchCandidate,
    output_atoms: Any,
) -> str:
    """Serialize the public final-output design-chain metadata."""

    return json.dumps(
        list(_output_design_chain_rows(candidate, output_atoms)),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _result_row_design_chain_labels(
    row: ResultRowWire | None,
) -> list[str]:
    """Read final labels from public metadata, then legacy row identities."""

    if row is None:
        return []
    raw_metadata = row[17] if len(row) >= 18 else ""
    if isinstance(raw_metadata, str) and raw_metadata.strip():
        try:
            parsed = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, list):
            labels = [
                str(item["chain"])
                for item in parsed
                if isinstance(item, Mapping)
                and isinstance(item.get("chain"), str)
                and item["chain"].strip()
            ]
            if len(labels) == len(parsed):
                return list(dict.fromkeys(labels))
    # Older rows carried the authoring chain scalar. This is the only scalar
    # fallback; task configuration chain IDs are never used for final scoring.
    if len(row) >= 5:
        scalar = str(row[4] or "").strip()
        if scalar:
            labels = [
                value.strip()
                for value in re.split(r"[,;|]", scalar)
                if value.strip()
            ]
            if labels:
                return list(dict.fromkeys(labels))
    if len(row) < 2:
        return []
    filename = Path(str(row[1])).stem
    marker = "_joint_"
    if marker not in filename:
        return []
    labels: list[str] = []
    for segment in filename.split(marker, 1)[1].split("__"):
        for modality in ("ligand", "prot", "rna", "dna"):
            token = f"_{modality}_"
            if token not in segment:
                continue
            encoded = segment.split(token, 1)[0]
            decoded = re.sub(
                r"_x([0-9A-Fa-f]{2})",
                lambda match: chr(int(match.group(1), 16)),
                encoded,
            )
            if decoded:
                labels.append(decoded)
            break
    return list(dict.fromkeys(labels))


def _safe_filename_component(value: str) -> str:
    """Encode a chain ID into an ASCII filename component without collisions."""

    result: list[str] = []
    for byte in str(value).encode("utf-8"):
        character = chr(byte)
        if (ord("A") <= byte <= ord("Z")) or (
            ord("a") <= byte <= ord("z")
        ) or (ord("0") <= byte <= ord("9")) or character == ".":
            result.append(character)
        else:
            # Underscore itself is escaped, so an encoded ``_xHH`` triplet can
            # never be confused with literal input text.
            result.append(f"_x{byte:02X}")
    return "".join(result) or "_x00"


def _legacy_owner_scalars(
    owner: CompiledSample | SearchCandidate,
) -> tuple[str, str, int | str]:
    request = owner.request if isinstance(owner, CompiledSample) else owner.base_request
    if _is_joint_request(request):
        return "joint", "", ""
    chain = request.task.design_chains[0]
    rows = _design_rows(
        owner.atoms,
        chain.chain_id,
        design_only=not isinstance(owner, SearchCandidate),
    )
    token_length = len(set(int(value) for value in owner.atoms.token_uid[rows]))
    return chain.modality, chain.chain_id, int(token_length)


def _candidate_output_identity(
    candidate: SearchCandidate,
    *,
    global_rank: int | None,
    output_format: str,
    sequence_override: str | None = None,
    ledger_entries: Sequence[Mapping[str, Any]] | None = None,
    joint_chain_ids: Sequence[str] | None = None,
) -> tuple[str, int, str, str, int | str, str, str, str]:
    request = candidate.base_request
    chain_id, modality = _primary_design_identity(request)
    # Public sample numbering is deliberately task-local.  ``global_ordinal``
    # remains an internal RNG/candidate-ordering fact and must never leak into
    # paths or CSV identity.
    base_sample_id = int(request.task_slot_ordinal) + 1
    if _is_joint_request(request):
        if global_rank is None:
            return f"sample{base_sample_id}.{output_format}", base_sample_id, "joint", "", "", "", "", ""
        filename_chain_ids = (
            tuple(str(value) for value in joint_chain_ids)
            if joint_chain_ids is not None
            else tuple(str(chain.chain_id) for chain in request.task.design_chains)
        )
        chain_ids = "-".join(
            _safe_filename_component(chain_id)
            for chain_id in filename_chain_ids
        )
        filename = f"sample{base_sample_id}_{chain_ids}_rank{int(global_rank)}"
        return filename + f".{output_format}", base_sample_id, "joint", "", "", "", "", ""
    rows = _design_rows(candidate.atoms, chain_id, design_only=False)
    token_length = len(set(int(value) for value in candidate.atoms.token_uid[rows]))
    short = {"protein": "prot", "rna": "rna", "dna": "dna", "ligand": "ligand"}[modality]
    filename = f"sample{base_sample_id}.{output_format}"
    if global_rank is not None:
        filename = (
            f"sample{base_sample_id}_{short}_{chain_id}_length{token_length}_"
            f"rank{global_rank}.{output_format}"
        )
    sequence = sequence_override or _candidate_design_sequence(
        candidate, chain_id=chain_id, modality=modality
    )
    contig, tip = _motif_output_fields(candidate, chain_id, modality)
    return filename, base_sample_id, modality, chain_id, token_length, sequence, contig, tip


def _failed_replica_row(
    compiled: CompiledSample,
    *,
    reward_method: str,
    stage: str,
    message: str,
) -> ResultRowWire:
    request = compiled.request
    modality, chain_id, token_length = _legacy_owner_scalars(compiled)
    contig, tip = _motif_output_fields(compiled, chain_id, modality)
    ledger = _ledger_json(_build_design_ledger(compiled))
    return (
        request.task.name,
        "",
        int(request.task_slot_ordinal) + 1,
        modality,
        chain_id,
        token_length,
        reward_method,
        "",
        "",
        contig,
        tip,
        "failed",
        stage,
        message,
        "{}",
        ledger,
        "{}",
        "",
    )


def _failed_request_row(
    request: ExpandedRequest,
    *,
    reward_method: str,
    stage: str,
    message: str,
) -> ResultRowWire:
    ledger = _ledger_json(_request_design_ledger(request))
    if _is_joint_request(request):
        modality, chain_id, token_length = "joint", "", ""
    else:
        chain_id, modality = _primary_design_identity(request)
        planned = request.plan.lengths.get(chain_id)
        if planned is None:
            sequence = request.task.condition.get("sequence", {})
            raw_sequence = (
                sequence.get(chain_id) if isinstance(sequence, Mapping) else None
            )
            planned = len(raw_sequence) if isinstance(raw_sequence, str) else 0
        token_length = int(planned)
    # Compilation did not produce a sampled layout, so publishing the raw task
    # contig here would misrepresent it as resolved candidate metadata.
    contig, tip = "", ""
    return (
        request.task.name,
        "",
        int(request.task_slot_ordinal) + 1,
        modality,
        chain_id,
        token_length,
        reward_method,
        "",
        "",
        contig,
        tip,
        "failed",
        stage,
        message,
        "{}",
        ledger,
        "{}",
        "",
    )


def _af3_reward_marker():
    from ..rewards.af3 import AF3Reward
    return AF3Reward()


def _reward_method(config: Mapping[str, Any]) -> str:
    target = config.get("_target_")
    if target == "src.design.rewards.esmfold2.ESMFold2Reward":
        return "esmfold2"
    if target == "src.design.rewards.af2.AF2Reward":
        return "af2" + str(config["af2"]["model"])
    if target == "src.design.rewards.af3.AF3Reward":
        return str(config["af3"]["model"])
    if target == "src.design.rewards.clash.ClashReward":
        return "clash"
    if target == "src.design.rewards.partiscore.PartiScoreReward":
        return "partiscore"
    raise RuntimeError("search reward method is not implemented")


def _failure_wire(failure: SampleFailure) -> FailureWire:
    return (
        failure.task_name,
        failure.sample_id,
        failure.stage,
        failure.message,
    )


def _move_predict_batch(batch: PredictBatch, device: torch.device) -> PredictBatch:
    return PredictBatch(
        samples=batch.samples,
        requests=batch.requests,
        canonical={
            key: value.to(device=device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.canonical.items()
        },
    )


def _all_gather_objects(value: Any) -> list[Any]:
    _, world_size = _distributed_rank_world()
    if world_size == 1:
        return [value]
    gathered: list[Any] = [None] * world_size
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def _gather_objects_to_rank_zero(value: Any) -> list[Any] | None:
    """Gather a private object without materializing it on peer ranks."""

    rank, world_size = _distributed_rank_world()
    if world_size == 1:
        return [value]
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    torch.distributed.gather_object(value, gathered, dst=0)
    return gathered


def actual_joint_length(compiled: CompiledSample) -> tuple[int, ...]:
    """Return search-group lengths in declared design-chain order.

    Counts are always taken from the checkpoint-facing model-token projection;
    ordinary standard single-chain counts remain numerically unchanged because
    their authoring and model token axes are one-to-one.
    """

    layout = build_model_token_layout(
        compiled.atoms,
        compiled.modified_polymer_carriers,
    )
    return tuple(
        _design_counts(compiled, chain.chain_id, layout)[1]
        for chain in compiled.request.task.design_chains
    )


def _fixed_modified_residue_names(compiled: CompiledSample) -> dict[int, str]:
    """Return primitive candidate-report metadata from compiler carrier records."""

    return dict(
        sorted(
            (
                int(residue_uid),
                str(carrier.report_residue_name),
            )
            for residue_uid, carrier in compiled.modified_polymer_carriers.items()
        )
    )


def _fixed_modified_parent_residue(compiled: CompiledSample) -> dict[int, str]:
    """Return canonical-parent hints only for fixed modified carriers."""

    return dict(
        sorted(
            (
                int(residue_uid),
                str(carrier.standard_parent),
            )
            for residue_uid, carrier in compiled.modified_polymer_carriers.items()
            if carrier.standard_parent is not None
        )
    )


def _numbered_search_candidate(
    compiled: CompiledSample,
    candidate: SearchCandidate,
    *,
    terminal_local_ordinal: int,
    terminal_width: int,
) -> SearchCandidate:
    request = compiled.request
    if _request_key(candidate.base_request) != _request_key(request):
        raise RuntimeError("terminal candidate does not match its compiled request")
    ordinal = stable_candidate_ordinal(
        request.global_ordinal,
        terminal_local_ordinal,
        terminal_width,
    )
    if ordinal > np.iinfo(np.int64).max:
        raise OverflowError("global candidate ordinal exceeds int64")
    return SearchCandidate(
        candidate_id=(
            f"{request.task.name}/{request.sample_id}/"
            f"candidate_{terminal_local_ordinal:06d}"
        ),
        candidate_ordinal=ordinal,
        base_request=request,
        actual_joint_length=actual_joint_length(compiled),
        atoms=candidate.atoms,
        topology_conditions=compiled.topology_conditions,
        model_parent_residue=dict(compiled.model_parent_residue),
        antibody_metadata=dict(compiled.antibody_metadata),
        fixed_modified_residue_names=_fixed_modified_residue_names(compiled),
        fixed_modified_parent_residue=_fixed_modified_parent_residue(compiled),
        validated_ligand_identities=dict(compiled.validated_ligand_identities),
        private_ligand_molecules=dict(candidate.private_ligand_molecules),
        private_ligand_errors=dict(candidate.private_ligand_errors),
        ligand_fragment_metadata=dict(candidate.ligand_fragment_metadata),
        motif_output_fields=dict(compiled.motif_output_fields),
    )


def _copy_private_ligand_molecules(
    molecules: Mapping[int, Any],
) -> dict[int, Any]:
    if not molecules:
        return {}
    try:
        from rdkit import Chem
    except ImportError:
        # A private molecule payload can exist only when its chemistry backend is
        # installed; do not silently drop it during FK resampling.
        raise RuntimeError("private ligand molecule copy requires RDKit")
    return {int(uid): Chem.Mol(molecule) for uid, molecule in molecules.items()}


def _candidate_with_atoms(
    candidate: SearchCandidate,
    atoms: Any,
) -> SearchCandidate:
    """Carry owner-local private chemistry onto finalized output atoms.

    A deferred fold may replace the candidate ``AtomArray`` (including the
    designed protein sequence) while retaining the private ligand chemistry
    that was validated during base generation.  ``copy.copy`` deliberately
    preserves every non-coordinate sidecar; the postprocess payload builder
    then revalidates those sidecars against the finalized structure and fails
    closed if the fold changed the chemistry identity.
    """

    finalized = copy.copy(candidate)
    finalized.atoms = atoms
    return finalized


def build_terminal_candidates(
    model: torch.nn.Module,
    batch: PredictBatch,
    *,
    sampling_config: Mapping[str, Any],
    ligand_graph_method: str,
) -> tuple[TerminalResult, ...]:
    """Thin terminal adapter over the model, WP5 decode, and assembly APIs."""

    generated, generation_info = model.generate(batch.canonical, sampling_config)
    generated_token_mask = generation_info.get("denoising_mask")
    if not isinstance(generated_token_mask, torch.Tensor):
        raise RuntimeError("model generation omitted the denoising_mask contract")
    context_constraint_mode = getattr(model, "context_constraint_mode", None)
    if context_constraint_mode not in {"rigid", "flex"}:
        raise RuntimeError("model omitted a valid context_constraint_mode contract")
    decoder_output, _ = model.decode_generated(batch.canonical, generated)
    results: list[TerminalResult] = []
    for sample_index, compiled in enumerate(batch.samples):
        request = compiled.request
        try:
            proposal = decode_proposal(
                batch.canonical,
                generated,
                decoder_output,
                sample_index=sample_index,
                ligand_graph_method=ligand_graph_method,
                context_constraint_mode=context_constraint_mode,
                generated_token_mask=generated_token_mask,
                compiled=compiled,
            )
            assembled = assemble_candidate(compiled, proposal)
            results.append(
                SearchCandidate(
                    candidate_id=(
                        f"{request.task.name}/{request.sample_id}/candidate_000000"
                    ),
                    candidate_ordinal=request.global_ordinal,
                    base_request=request,
                    actual_joint_length=actual_joint_length(compiled),
                    atoms=assembled.atoms,
                    topology_conditions=compiled.topology_conditions,
                    model_parent_residue=dict(compiled.model_parent_residue),
                    antibody_metadata=dict(compiled.antibody_metadata),
                    fixed_modified_residue_names=_fixed_modified_residue_names(compiled),
                    fixed_modified_parent_residue=_fixed_modified_parent_residue(
                        compiled
                    ),
                    validated_ligand_identities=dict(compiled.validated_ligand_identities),
                    private_ligand_molecules=dict(assembled.private_ligand_molecules),
                    private_ligand_errors=dict(assembled.private_ligand_errors),
                    ligand_fragment_metadata=dict(assembled.ligand_fragment_metadata),
                    motif_output_fields=dict(compiled.motif_output_fields),
                )
            )
        except (DesignInputError, DesignSampleError, ValueError) as exc:
            results.append(
                SampleFailure(
                    task_name=request.task.name,
                    sample_id=request.sample_id,
                    stage="decode_assembly",
                    message=str(exc),
                )
            )
    return tuple(results)


def _clone_simulation_state(state: SimulationState) -> SimulationState:
    return SimulationState(
        step_index=state.step_index,
        x={key: value.detach().clone() for key, value in state.x.items()},
        previous_clean=(
            None
            if state.previous_clean is None
            else {
                key: value.detach().clone()
                for key, value in state.previous_clean.items()
            }
        ),
    )


def build_fk_terminal_candidates(
    model: torch.nn.Module,
    compiled: CompiledSample,
    batch: PredictBatch,
    *,
    sampling_config: Mapping[str, Any],
    ligand_graph_method: str,
    search_config: Mapping[str, Any],
    reward: ClashReward,
) -> tuple[TerminalResult, ...]:
    """Run one request-local FK beam through the segmented standalone solver."""

    if len(batch.samples) != 1 or batch.samples[0] is not compiled:
        raise ValueError("FK terminal factory requires one matching CompiledSample")
    solver = getattr(model, "solver", None)
    if solver is None:
        raise TypeError("FK requires the standalone segmented ProductSpaceSolver")
    sampling_batch = model.add_sampling_context(batch.canonical)
    fk_config = search_config["fk_steering"]
    beam_width = int(fk_config["beam_width"])
    initial_states = [
        solver.initialize_simulation(
            batch=sampling_batch,
            sampling_config=sampling_config,
        )
        for _ in range(beam_width)
    ]
    def advance_segment(
        state: SimulationState,
        start_step: int,
        end_step: int,
        branch_index: int,
    ) -> SimulationState:
        del branch_index
        return solver.partial_simulation(
            batch=sampling_batch,
            predict_for_sampling=model.predict_for_sampling,
            sampling_config=sampling_config,
            state=_clone_simulation_state(state),
            start_step=start_step,
            end_step=end_step,
        )

    def terminal_lookahead(
        state: SimulationState,
        end_step: int,
        terminal_step: int,
        attempt_ordinal: int,
    ) -> SearchCandidate:
        terminal_state = _clone_simulation_state(state)
        if end_step < terminal_step:
            terminal_state = solver.partial_simulation(
                batch=sampling_batch,
                predict_for_sampling=model.predict_for_sampling,
                sampling_config=sampling_config,
                state=terminal_state,
                start_step=end_step,
                end_step=terminal_step,
            )
        generated = dict(terminal_state.x)
        token_mask = batch.canonical.get("token_mask")
        design_mask = batch.canonical.get("design_token_mask")
        if not isinstance(token_mask, torch.Tensor) or not isinstance(
            design_mask, torch.Tensor
        ):
            raise RuntimeError("FK batch omitted canonical token masks")
        context_constraint_mode = getattr(model, "context_constraint_mode", None)
        if context_constraint_mode == "rigid":
            generated_token_mask = design_mask.bool()
        elif context_constraint_mode == "flex":
            generated_token_mask = token_mask.bool()
        else:
            raise RuntimeError("model omitted a valid context_constraint_mode contract")
        decoder_output, _ = model.decode_generated(batch.canonical, generated)
        try:
            proposal = decode_proposal(
                batch.canonical,
                generated,
                decoder_output,
                sample_index=0,
                ligand_graph_method=ligand_graph_method,
                context_constraint_mode=context_constraint_mode,
                generated_token_mask=generated_token_mask,
                compiled=compiled,
            )
            # Preserve this exact terminal assembly: OpenBabel/private chemistry is
            # not rerun after terminal resampling.
            assembled = assemble_candidate(compiled, proposal)
        except (DesignInputError, DesignSampleError, ValueError) as exc:
            raise FKBranchRejected(str(exc)) from exc
        return SearchCandidate(
            candidate_id=f"fk-lookahead-{attempt_ordinal:012d}",
            candidate_ordinal=attempt_ordinal,
            base_request=compiled.request,
            actual_joint_length=actual_joint_length(compiled),
            atoms=assembled.atoms,
            topology_conditions=compiled.topology_conditions,
            model_parent_residue=dict(compiled.model_parent_residue),
            antibody_metadata=dict(compiled.antibody_metadata),
            fixed_modified_residue_names=_fixed_modified_residue_names(compiled),
            fixed_modified_parent_residue=_fixed_modified_parent_residue(compiled),
            validated_ligand_identities=dict(compiled.validated_ligand_identities),
            private_ligand_molecules=dict(assembled.private_ligand_molecules),
            private_ligand_errors=dict(assembled.private_ligand_errors),
            ligand_fragment_metadata=dict(assembled.ligand_fragment_metadata),
        )

    selected = run_fk_request(
        initial_states,
        step_checkpoints=tuple(int(value) for value in search_config["step_checkpoints"]),
        n_branch=int(fk_config["n_branch"]),
        temperature=float(fk_config["temperature"]),
        advance_segment=advance_segment,
        terminal_lookahead=terminal_lookahead,
        score_lookahead=lambda candidates: reward.score_batch(candidates, device="cpu"),
    )
    # Resampling is with replacement.  Give every retained terminal beam a unique
    # object while preserving the exact assembly selected above.
    return tuple(
        SearchCandidate(
            candidate_id=candidate.candidate_id,
            candidate_ordinal=candidate.candidate_ordinal,
            base_request=candidate.base_request,
            actual_joint_length=candidate.actual_joint_length,
            atoms=candidate.atoms.copy(),
            topology_conditions=candidate.topology_conditions,
            model_parent_residue=dict(candidate.model_parent_residue),
            antibody_metadata=dict(getattr(candidate, "antibody_metadata", {})),
            fixed_modified_residue_names=dict(candidate.fixed_modified_residue_names),
            fixed_modified_parent_residue=dict(
                candidate.fixed_modified_parent_residue
            ),
            validated_ligand_identities=dict(candidate.validated_ligand_identities),
            private_ligand_molecules=_copy_private_ligand_molecules(
                candidate.private_ligand_molecules
            ),
            private_ligand_errors=dict(candidate.private_ligand_errors),
            ligand_fragment_metadata=dict(candidate.ligand_fragment_metadata),
        )
        for candidate in selected
    )


def _compile_runtime_request(
    request: ExpandedRequest,
    *,
    ccd_path: Path | None,
    trusted_ccd_components: Mapping[str, TrustedCCDComponent],
    global_seed: int,
    prepared_source: Any = None,
) -> CompiledSample:
    """Compile one request with a worker- and preflight-independent RNG."""

    options = {}
    if isinstance(prepared_source, Exception):
        raise prepared_source
    if prepared_source is not None:
        options["prepared_source"] = prepared_source
    return compile_request(
        request,
        ccd_path=ccd_path,
        trusted_ccd_components=trusted_ccd_components,
        rng=np.random.default_rng(int(global_seed) + int(request.global_ordinal)),
        **options,
    )


class _CompiledRequestDataset(Dataset[PreparedItem]):
    def __init__(
        self,
        requests: Sequence[ExpandedRequest],
        *,
        ccd_path: Path | None,
        trusted_ccd_components: Mapping[str, TrustedCCDComponent],
        global_seed: int = 0,
        clash_preflight: bool = False,
        prepared_sources: Mapping[str, Any] | None = None,
    ) -> None:
        self.requests = tuple(requests)
        self.ccd_path = ccd_path
        self.trusted_ccd_components = dict(trusted_ccd_components)
        self.global_seed = int(global_seed)
        self.clash_preflight = bool(clash_preflight)
        self.prepared_sources = dict(prepared_sources or {})

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, index: int) -> PreparedItem:
        request = self.requests[index]
        try:
            options = dict(ccd_path=self.ccd_path,
                           trusted_ccd_components=self.trusted_ccd_components,
                           global_seed=self.global_seed)
            if request.task.name in self.prepared_sources:
                options["prepared_source"] = self.prepared_sources[request.task.name]
            compiled = _compile_runtime_request(request, **options)
        except DesignInputError as exc:
            return SampleFailure(
                request.task.name, request.sample_id, "input", str(exc)
            )
        except DesignSampleError as exc:
            return SampleFailure(
                request.task.name, request.sample_id, "compile", str(exc)
            )
        if self.clash_preflight:
            atoms = compiled.atoms
            authoritative = (~atoms.is_design) | atoms.identity_condition
            try:
                validate_vdw_elements(atoms.element[authoritative])
            except ValueError as exc:
                return SampleFailure(
                    request.task.name,
                    request.sample_id,
                    "reward_preflight",
                    str(exc),
                )
        return compiled


def _distributed_rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _release_generation_process_group() -> tuple[int, int]:
    """Release the generation process group and retain its pre-release identity.

    OpenMM postprocessing runs only on rank zero after generation accounting.  A
    destroyed process group reports ``(0, 1)`` through ``_distributed_rank_world``;
    callers therefore must use the returned values for deciding which ranks
    continue with postprocessing.
    """

    rank, world_size = _distributed_rank_world()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return rank, world_size


_LIGHTNING_MAX_SEED = 0xFFFFFFFF
_TORCH_MAX_SEED = (1 << 64) - 1


def _validate_global_seed(global_seed: int) -> int:
    """Validate the base seed accepted by Lightning's final ``reset_seed``."""

    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global seed must be an integer")
    if global_seed < 0:
        raise ValueError("global seed must be nonnegative")
    if global_seed > _LIGHTNING_MAX_SEED:
        raise ValueError(
            "global seed must be at most 4294967295 because Lightning accepts "
            "only 32-bit seeds"
        )
    return global_seed


def _global_rank_seed(global_seed: int, global_rank: int) -> int:
    """Return the exact Torch seed for one global DDP rank.

    The rank-zero stream is intentionally ``global_seed`` and every other rank
    advances that same base by its global rank.  There is no folding, hashing, or
    modulo arithmetic: overflow is reported as an invalid seed configuration.
    """

    base_seed = _validate_global_seed(global_seed)
    if isinstance(global_rank, bool) or not isinstance(global_rank, int):
        raise TypeError("global rank must be an integer")
    if global_rank < 0:
        raise ValueError("global rank must be nonnegative")
    rank_seed = base_seed + global_rank
    if rank_seed > _TORCH_MAX_SEED:
        raise ValueError(
            "global seed plus global rank exceeds Torch's 64-bit seed range"
        )
    return rank_seed


class _ExactShardBatchSampler(Sampler[list[int]]):
    """Batch an exact shard and emit one empty control batch for an idle rank."""

    def __init__(
        self,
        sampler: int | Sampler[int],
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        if drop_last:
            raise ValueError("exact inference sharding never drops requests")
        self.size = int(sampler) if isinstance(sampler, int) else len(sampler)
        self.batch_size = int(batch_size)
        self.sampler = (
            SequentialSampler(range(self.size)) if isinstance(sampler, int) else sampler
        )
        self.drop_last = False
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def __iter__(self):
        rank, world_size = _distributed_rank_world()
        indices = list(range(rank, self.size, world_size))
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        maximum_local_size = (self.size + world_size - 1) // world_size
        maximum_batches = max(
            1,
            (maximum_local_size + self.batch_size - 1) // self.batch_size,
        )
        # Empty collective-control batches equalize predict-step counts without
        # owning, compiling, generating, writing, or accounting a fake sample.
        batches.extend([[] for _ in range(maximum_batches - len(batches))])
        yield from batches

    def __len__(self) -> int:
        _, world_size = _distributed_rank_world()
        maximum_local_size = (self.size + world_size - 1) // world_size
        return max(1, (maximum_local_size + self.batch_size - 1) // self.batch_size)


def _collate_prepared(
    items: Sequence[PreparedItem], *, token_buckets: Sequence[int] | None = None,
) -> tuple[PredictBatch | None, tuple[SampleFailure, ...]]:
    compiled = tuple(item for item in items if isinstance(item, CompiledSample))
    failures = tuple(item for item in items if isinstance(item, SampleFailure))
    return (
        collate_compiled_samples(compiled, token_buckets=token_buckets) if compiled else None,
        failures,
    )


def _require_bond_order_checkpoint_support(
    model: torch.nn.Module,
    canonical: Mapping[str, Any],
) -> None:
    """Reject concrete/non-edge bond conditions that a checkpoint would ignore.

    The older rigid checkpoint still consumes endpoint/existence conditions, so
    unknown-positive bonds remain valid.  A true entry in
    ``condition_bond_type_known_mask`` additionally needs the five-state order
    feature.  Test doubles without a declared capability stay injectable; every
    production :class:`PartiDesignInferenceModel` declares the property.
    """

    known = canonical.get("condition_bond_type_known_mask")
    if not isinstance(known, torch.Tensor):
        raise TypeError(
            "canonical condition_bond_type_known_mask must be a tensor"
        )
    if known.dtype != torch.bool:
        raise TypeError(
            "canonical condition_bond_type_known_mask must be boolean"
        )
    if not bool(known.any().item()):
        return
    supported = getattr(model, "supports_bond_order_conditioning", None)
    if supported is None:
        return
    if type(supported) is not bool:
        raise TypeError(
            "model supports_bond_order_conditioning must be boolean"
        )
    if not supported:
        raise DesignConfigError(
            "model checkpoint does not consume condition_bond_order_pair; "
            "concrete bond orders and known non-edges would be ignored"
        )


class _StandalonePredictModule(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        sampling_config: Mapping[str, Any],
        ligand_graph_method: str,
        output_dir: Path,
        output_format: str,
        terminal_candidate_factory: TerminalCandidateFactory,
        fk_candidate_factory: FKCandidateFactory,
        search_config: Mapping[str, Any],
        reward_config: Mapping[str, Any],
        requests: Sequence[ExpandedRequest],
        postprocess_enabled: bool,
        fold_reward_runner: FoldRewardRunner,
        global_seed: int,
        scorer_config: Mapping[str, Any] | None = None,
        token_buckets: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.token_buckets = None if token_buckets is None else tuple(token_buckets)
        self.scorer_config = dict(scorer_config or {})
        self._partiscore = None
        self._local_score_records: dict[tuple[str, str], dict[str, Any]] = {}
        self._global_score_records: dict[tuple[str, str], dict[str, Any]] = {}
        self._score_ranking_finalized_tasks: set[str] = set()
        self._local_reward_records: dict[str, dict[str, Any]] = {}
        self._prepared_fold_atoms: dict[str, Any] = {}
        self._local_sequence_sources: dict[str, str] = {}
        self.inference_model = model
        self.sampling_config = dict(sampling_config)
        self.ligand_graph_method = ligand_graph_method
        self.output_dir = output_dir
        self.output_format = output_format
        self.terminal_candidate_factory = terminal_candidate_factory
        self.fk_candidate_factory = fk_candidate_factory
        self.search_config = dict(search_config)
        self.search_algorithm = self.search_config.get("algorithm")
        self.reward_config = dict(reward_config)
        self.reward_method = (
            _reward_method(self.reward_config)
            if self.search_algorithm is not None
            else ""
        )
        self.fold_reward_runner = fold_reward_runner
        self.global_seed = int(global_seed)
        _validate_global_seed(self.global_seed)
        self._rank_rng_seeded = False
        reward_target = self.reward_config.get("_target_")
        self.reward = (
            None
            if self.search_algorithm is None
            else ClashReward(
                design_design_weight=float(reward_config["design_design_weight"]),
                design_context_weight=float(reward_config["design_context_weight"]),
            )
            if reward_target == "src.design.rewards.clash.ClashReward"
            else ESMFold2Reward()
            if reward_target == "src.design.rewards.esmfold2.ESMFold2Reward"
            else PartiScoreReward(self._get_partiscore)
            if reward_target == "src.design.rewards.partiscore.PartiScoreReward"
            else AF2Reward()
            if reward_target == "src.design.rewards.af2.AF2Reward"
            else _af3_reward_marker()
            if reward_target == "src.design.rewards.af3.AF3Reward"
            else None
        )
        if self.search_algorithm is not None and self.reward is None:
            raise DesignConfigError("runtime reward target is not implemented")
        self.deferred_fold_reward = isinstance(self.reward, (ESMFold2Reward, AF2Reward)) or (self.reward is not None and reward_target == "src.design.rewards.af3.AF3Reward")
        self.deferred_partiscore_reward = (
            isinstance(self.reward, PartiScoreReward)
            and self.reward_config.get("sequence", {}).get("source", "design") != "design"
        )
        # Postprocessing is a task-level switch.  The root postprocess mapping
        # only carries shared OpenMM execution parameters and must not make an
        # otherwise plain task produce private chemistry payloads.
        self._postprocess_task_names = frozenset(
            request.task.name
            for request in requests
            if bool(getattr(request.task, "postprocess", False))
            and request.task.override
        )
        del postprocess_enabled  # task switches are the sole enablement source
        self.postprocess_enabled = bool(self._postprocess_task_names)
        # Exact non-padding sharding legitimately leaves a rank empty when the
        # request count is smaller than the visible device count.
        self.allow_zero_length_dataloader_with_multiple_devices = True
        self._local_succeeded: list[ResultKey] = []
        self._local_failures: list[FailureWire] = []
        self._global_succeeded: tuple[ResultKey, ...] | None = None
        self._global_failures: tuple[FailureWire, ...] | None = None
        self._local_slot_groups: list[SlotWire] = []
        self._local_candidates: dict[str, SearchCandidate] = {}
        self._local_design_ledgers: dict[str, tuple[dict[str, Any], ...]] = {}
        self._local_fold_results: dict[str, FoldRewardResult] = {}
        self._local_batch_seconds: list[float] = []
        self._global_batch_seconds: tuple[tuple[int, int, float], ...] = ()
        self._local_result_rows: list[ResultRowWire] = []
        self._global_result_rows: tuple[ResultRowWire, ...] = ()
        self._request_by_key = {_request_key(request): request for request in requests}
        if len(self._request_by_key) != len(requests):
            raise DesignConfigError("runtime requests must have unique IDs")
        self._task_names = tuple(dict.fromkeys(request.task.name for request in requests))
        self._task_score_enabled = any(request.task.score for request in requests)
        self._local_parameterization_payloads: dict[
            ResultKey, SampleParameterizationPayload
        ] = {}
        self._local_postprocess_failures: dict[ResultKey, str] = {}
        self._global_parameterization_payloads: tuple[
            SampleParameterizationPayload, ...
        ] = ()
        self._global_postprocess_failures: tuple[PostprocessFailureWire, ...] = ()
        self._global_postprocess_status: dict[ResultKey, str] = {}
        self._global_postprocess_messages: dict[ResultKey, str] = {}
        self._postprocess_pending_tasks: set[str] = set()
        self._postprocess_pending_updates = 0
        self._postprocess_last_flush_monotonic = time.monotonic()

    def _task_postprocess_enabled(self, task_name: str) -> bool:
        return str(task_name) in self._postprocess_task_names

    def _postprocess_result_keys(
        self,
        rows: Sequence[ResultRowWire],
    ) -> set[ResultKey]:
        """Return actual successful filenames requested for postprocessing."""

        return {
            (str(row[0]), str(row[1]))
            for row in rows
            if len(row) >= 12
            and row[11] == "ok"
            and isinstance(row[1], str)
            and bool(row[1])
            and self._task_postprocess_enabled(str(row[0]))
        }

    def _get_partiscore(self) -> Any:
        if self._partiscore is None:
            from src.scoring import PartiScorer

            options = {key: value for key, value in self.scorer_config.items() if key != "checkpoint"}
            # FM can already be offloaded to CPU: strategy retains the owner device.
            options["device"] = str(self.trainer.strategy.root_device if self._trainer is not None else self.device)
            started = time.perf_counter()
            self._partiscore = PartiScorer.from_checkpoint(self.scorer_config["checkpoint"], **options)
            self._partiscore_load_seconds = time.perf_counter() - started
        return self._partiscore

    def _prepare_fold_output(self, candidate: SearchCandidate, fold: FoldRewardResult) -> Any:
        if candidate.candidate_id not in self._prepared_fold_atoms:
            folded = read_structure(
                fold.fold_cif,
                declared_polymer_chain_modalities=candidate_chain_modalities(candidate),
            )
            restored = _restore_deferred_fold_output(
                candidate, folded,
                canonical_ligand_names=self.reward_method == "esmfold2",
            )
            restored = _restore_fold_design_state(candidate, restored)
            if "source_mol_id" in candidate.atoms.get_annotation_categories():
                restored = _restore_fold_source_partners(candidate.atoms, restored)
            cyclic_asyms = _head_to_tail_design_protein_asyms(candidate)
            if cyclic_asyms:
                cyclic_chains = tuple(dict.fromkeys(
                    str(candidate.atoms.chain_id[np.flatnonzero(candidate.atoms.asym_uid == asym)[0]])
                    for asym in sorted(cyclic_asyms)
                ))
                restored = normalize_cyclic_product(restored, cyclic_chains)
            self._prepared_fold_atoms[candidate.candidate_id] = restored
        return self._prepared_fold_atoms[candidate.candidate_id]

    def _score_output_paths(self, entries: Sequence[tuple[str, str, Path, str]]) -> dict:
        """Read each successful actual output once, with primitive result sidecars."""
        if not entries:
            return {}
        row_by_key: dict[ResultKey, ResultRowWire] = {}
        for row in (*self._local_result_rows, *self._global_result_rows):
            if len(row) >= 2:
                row_by_key[(str(row[0]), str(row[1]))] = row
        identities = [f"{task}/{path.name}" for task, _base, path, _source in entries]
        results = self._get_partiscore().score_files(
            [path for _task, _base, path, _source in entries],
            identities=identities,
            design_chains=[
                _result_row_design_chain_labels(row_by_key.get((task, base)))
                for task, base, _path, _source in entries
            ],
            scope="design_groups",
        )
        if [result.identity for result in results] != identities:
            raise RuntimeError("PartiScore final-file result IDs differ from submitted files")
        records = {}
        for (task, base, path, source), result in zip(entries, results, strict=True):
            if result.status in {"ok", "no_contact"}:
                ranking_score(result)  # Successful numeric errors are fatal here too.
            elif result.status not in {"error", "no_interface"}:
                raise RuntimeError(f"unknown PartiScore result status {result.status!r}")
            prefix = "refined_partiscore" if source == "refined" else "partiscore"
            metadata = {
                f"{prefix}_score": result.score,
                f"{prefix}_pair_count": result.pair_count,
                f"{prefix}_status": result.status,
                f"{prefix}_error": result.error,
                f"{prefix}_source": source,
                f"{prefix}_weights": result.weights,
                f"{prefix}_checkpoint": str(result.checkpoint),
            }
            row = row_by_key.get((task, base))
            if row is not None and len(row) >= 17:
                output_metadata = json.loads(row[16])
                if "_antibody_result" in output_metadata:
                    from ..antibody.scoring import antibody_score_record
                    antibody_row = json.loads(output_metadata["_antibody_result"])
                    binder = [antibody_row[key] for key in ("heavy_chain", "light_chain") if antibody_row.get(key)]
                    metadata.update(antibody_score_record(result, binder))
            if source == "refined":
                metadata["refined_filename"] = path.name
            interfaces = [
                {"filename": path.name, "source": source, **interface.as_dict()}
                for interface in result.interfaces
            ]
            record = {"metadata": metadata, "interfaces": interfaces}
            # This is the distributed boundary: strings, finite numbers, nulls only.
            json.dumps(record, allow_nan=False)
            records[(task, base)] = record
        return records

    def _score_local_outputs(self) -> None:
        task_by_name = {request.task.name: request.task for request in self._request_by_key.values()}
        entries = [
            (row[0], row[1], self.output_dir / row[0] / row[1],
             "fold" if self.deferred_fold_reward and self.reward_config["structure_source"] == "fold" else "design")
            for row in self._local_result_rows
            if row[11] == "ok" and task_by_name[row[0]].score
            and not self._task_postprocess_enabled(row[0])
            and (row[0], row[1]) not in self._local_score_records
        ]
        self._local_score_records.update(self._score_output_paths(entries))

    def score_refined_outputs(self, jobs: Sequence[Any], results: Mapping[Any, Any]) -> None:
        task_by_name = {request.task.name: request.task for request in self._request_by_key.values()}
        original_source = (
            "fold" if self.deferred_fold_reward
            and self.reward_config["structure_source"] == "fold" else "design"
        )
        entries = [
            (
                job.task_name,
                job.base_cif.name,
                job.refined_cif if results[job.payload.key].succeeded else job.base_cif,
                "refined" if results[job.payload.key].succeeded else original_source,
            )
            for job in jobs if task_by_name[job.task_name].score
        ]
        # Preparation failures have no OpenMM job, but their original output is
        # still a valid, explicitly requested scoring target.
        job_keys = {(job.task_name, job.base_cif.name) for job in jobs}
        entries.extend(
            (task, filename, self.output_dir / task / filename, original_source)
            for task, filename in self.global_postprocess_keys()
            if (task, filename) not in job_keys and task_by_name[task].score
        )
        for key, record in self._score_output_paths(entries).items():
            base = self._global_score_records.setdefault(key, {"metadata": {}, "interfaces": []})
            base["metadata"].update(record["metadata"])
            base["interfaces"].extend(record["interfaces"])
        if entries:
            # Keep every completed postprocess status durable before the final
            # score write publishes its task rows.
            self.flush_postprocess_status(force=True)
            self._write_result_csv(
                affected_tasks={str(entry[0]) for entry in entries}
            )

    def score_accounting(self, task_name: str | None = None) -> tuple[int, int]:
        succeeded = failed = 0
        for (record_task, _filename), record in self._global_score_records.items():
            if task_name is not None and record_task != task_name:
                continue
            for key, value in record["metadata"].items():
                if key.endswith("partiscore_status"):
                    if value == "error":
                        failed += 1
                    else:
                        succeeded += 1
        return succeeded, failed

    def finalize_score_rankings(self) -> None:
        """Publish task ranks only after global final-file scoring is complete."""
        if not self._task_score_enabled:
            return
        rank, world_size = _distributed_rank_world()
        publication_error: Exception | None = None
        status = None
        if rank == 0:
            try:
                self._publish_final_score_rankings()
                status = ("ok", "")
            except Exception as error:
                publication_error = error
                status = ("fatal", f"{type(error).__name__}: {error}")
        if world_size > 1:
            statuses = [status]
            torch.distributed.broadcast_object_list(statuses, src=0)
            status = statuses[0]
        if status is None:
            raise RuntimeError("rank zero did not publish final-score ranking status")
        if status[0] == "fatal":
            if publication_error is not None:
                raise publication_error
            raise RuntimeError(f"rank-zero final-score ranking failed: {status[1]}")

    def _publish_final_score_rankings(self) -> None:
        from ..utils.score_ranking import finalize_score_ranking, _rename_owned_files

        if self._global_succeeded is None or self._global_failures is None:
            raise RuntimeError("final-score ranking requires completed global accounting")
        task_by_name = {request.task.name: request.task for request in self._request_by_key.values()}
        finalized = set(getattr(self, "_score_ranking_finalized_tasks", set()))
        for task_name in self._task_names:
            task = task_by_name[task_name]
            if not task.score or task_name in finalized:
                continue
            indexed = [(index, row) for index, row in enumerate(self._global_result_rows) if row[0] == task_name]
            expected = {(task_name, str(row[1])) for _index, row in indexed if row[11] == "ok" and row[1]}
            records = {key: record for key, record in self._global_score_records.items() if key[0] == task_name}
            if set(records) != expected:
                raise RuntimeError(f"final-score ranking requires complete scoring for task {task_name!r}")
            if self._task_postprocess_enabled(task_name) and any(
                self._global_postprocess_status.get(key) not in {"succeeded", "failed"} for key in expected
            ):
                raise RuntimeError(f"final-score ranking requires completed postprocessing for task {task_name!r}")

            rows = []
            for index, row in indexed:
                metadata = records.get((task_name, row[1]), {}).get("metadata", {})
                if getattr(task, "format", "normal") == "antibody":
                    prefix, field = "partiscore_antigen", "partiscore_antigen"
                else:
                    refined = self._global_postprocess_status.get((task_name, row[1])) == "succeeded"
                    prefix = "refined_partiscore" if refined else "partiscore"
                    field = f"{prefix}_score"
                if row[11] == "ok" and metadata.get(f"{prefix}_status") not in {"ok", "no_contact", "no_interface", "error"}:
                    raise RuntimeError(f"final-score ranking is missing the actual retained-structure score: {task_name}/{row[1]}")
                value = metadata.get(field) if metadata.get(f"{prefix}_status") == "ok" else None
                if value is not None and (isinstance(value, bool) or not math.isfinite(float(value))):
                    raise RuntimeError("final-score ranking received an invalid successful score")
                rows.append({"_row_index": index, "filename": row[1], "base_sample_id": row[2],
                             "global_rank": row[7], "final_partiscore": value,
                             **({"refined_filename": metadata["refined_filename"]} if "refined_filename" in metadata else {})})
            interfaces = [interface for record in records.values() for interface in record["interfaces"]]
            task_dir = self.output_dir / task_name
            table_paths = (task_dir / "result.csv", task_dir / "partiscore.csv")
            old_tables = {}
            for path in table_paths:
                if path.is_symlink():
                    raise RuntimeError(f"final-score table may not be a symlink: {path}")
                old_tables[path] = path.read_bytes() if path.exists() else None
            state_fields = ("_global_result_rows", "_global_score_records", "_local_score_records",
                            "_global_postprocess_status", "_global_postprocess_messages",
                            "_global_postprocess_failures", "_local_postprocess_failures",
                            "_global_parameterization_payloads", "_local_parameterization_payloads")
            old_state = {name: getattr(self, name) for name in state_fields}
            rename_map = {}
            try:
                ranked, updated_interfaces, rename_map = finalize_score_ranking(task_dir, rows, interfaces)
                ranked_by_index = {item["_row_index"]: item for item in ranked}
                updated_rows = []
                for index, row in enumerate(self._global_result_rows):
                    item = ranked_by_index.get(index)
                    if item is None:
                        updated_rows.append(row)
                        continue
                    metadata = json.loads(row[16]) if len(row) >= 17 else {}
                    metadata.update(final_partiscore=item["final_partiscore"], score_rank=item["score_rank"])
                    if "_antibody_result" in metadata:
                        antibody = json.loads(metadata["_antibody_result"])
                        antibody.update(filename=item["filename"], final_partiscore=item["final_partiscore"], score_rank=item["score_rank"])
                        metadata["_antibody_result"] = json.dumps(antibody, allow_nan=False)
                    values = list(row)
                    values[1] = item["filename"]
                    while len(values) < 17:
                        values.append("[]" if len(values) == 15 else "{}")
                    values[16] = json.dumps(metadata, allow_nan=False)
                    updated_rows.append(tuple(values))
                self._global_result_rows = tuple(updated_rows)
                score_records = {key: record for key, record in self._global_score_records.items() if key[0] != task_name}
                row_by_filename = {row[1]: ranked_by_index[index] for index, row in indexed}
                interface_offset = 0
                for key, record in records.items():
                    item = row_by_filename[key[1]]
                    metadata = dict(record["metadata"])
                    metadata.update(final_partiscore=item["final_partiscore"], score_rank=item["score_rank"])
                    if "refined_filename" in item:
                        metadata["refined_filename"] = item["refined_filename"]
                    count = len(record["interfaces"])
                    score_records[(task_name, item["filename"])] = {**record, "metadata": metadata,
                        "interfaces": updated_interfaces[interface_offset:interface_offset + count]}
                    interface_offset += count
                self._global_score_records = score_records

                def file_key(key: ResultKey) -> ResultKey:
                    return (key[0], rename_map.get(key[1], key[1])) if key[0] == task_name else key

                self._local_score_records = {file_key(key): score_records[file_key(key)]
                    if key[0] == task_name else record for key, record in self._local_score_records.items()}
                for name in ("_global_postprocess_status", "_global_postprocess_messages", "_local_postprocess_failures"):
                    setattr(self, name, {file_key(key): value for key, value in getattr(self, name).items()})
                self._global_postprocess_failures = tuple(
                    (*file_key((wire[0], wire[1])), wire[2]) for wire in self._global_postprocess_failures)
                self._global_parameterization_payloads = tuple(
                    replace(payload, sample_id=file_key(payload.key)[1]) if payload.task_name == task_name else payload
                    for payload in self._global_parameterization_payloads)
                self._local_parameterization_payloads = {file_key(key):
                    replace(payload, sample_id=file_key(key)[1]) if key[0] == task_name else payload
                    for key, payload in self._local_parameterization_payloads.items()}
                # Succeeded/failed requests and _request_by_key retain their
                # original identities; only final output file references move.
                self._score_ranking_finalized_tasks = finalized | {task_name}
                self._write_result_csv(affected_tasks={task_name})
                finalized.add(task_name)
            except BaseException as error:
                try:
                    if rename_map:
                        _rename_owned_files(task_dir, {destination: source for source, destination in rename_map.items()})
                    for name, value in old_state.items():
                        setattr(self, name, value)
                    self._score_ranking_finalized_tasks = set(finalized)
                    for path, content in old_tables.items():
                        if content is None:
                            path.unlink(missing_ok=True)
                            continue
                        descriptor, temporary = tempfile.mkstemp(prefix=".score-rank-csv-", dir=task_dir)
                        try:
                            with os.fdopen(descriptor, "wb") as stream:
                                stream.write(content)
                            os.replace(temporary, path)
                        finally:
                            Path(temporary).unlink(missing_ok=True)
                    for path in table_paths:
                        path.with_suffix(".csv.tmp").unlink(missing_ok=True)
                except BaseException as rollback_error:
                    raise RuntimeError(f"final-score table publication failed ({error}); rollback failed ({rollback_error})") from error
                raise

    def transfer_batch_to_device(
        self,
        batch: tuple[PredictBatch | None, tuple[SampleFailure, ...]],
        device: torch.device,
        dataloader_idx: int,
    ) -> tuple[PredictBatch | None, tuple[SampleFailure, ...]]:
        predict_batch, failures = batch
        if predict_batch is None:
            return None, failures
        return _move_predict_batch(predict_batch, device), failures

    def on_predict_start(self) -> None:
        """Set one rank-local Torch stream after Lightning's final seed reset."""

        self._ensure_rank_torch_rng_stream()

    def _ensure_rank_torch_rng_stream(self) -> None:
        """Seed every generation route once using ``base_seed + global_rank``."""

        if getattr(self, "_rank_rng_seeded", False):
            return
        rank, world_size = _distributed_rank_world()
        rank_seed = _global_rank_seed(self.global_seed, rank)
        torch.manual_seed(rank_seed)
        self._rank_rng_seeded = True
        self._log_rank_torch_rng_seed(rank, world_size, rank_seed)

    def _log_rank_torch_rng_seed(
        self, rank: int, world_size: int, rank_seed: int,
    ) -> None:
        """Append the effective rank seed to each selected task's run log.

        DDP uses Lightning's ``popen`` launcher, so child ranks do not inherit
        the parent process's configured ``FileHandler``.  A small O_APPEND write
        keeps the per-rank seed visible without adding another runtime callback.
        """

        output_dir = getattr(self, "output_dir", None)
        task_names = getattr(self, "_task_names", ())
        if not isinstance(output_dir, Path):
            return
        line = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} INFO "
            f"Torch RNG seed rank={rank}/{world_size}: {rank_seed}\n"
        )
        for task_name in task_names:
            path = output_dir / str(task_name) / "logs" / "design.log"
            try:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            except OSError:
                # Logging must never change candidate generation or accounting.
                continue

    def _capture_parameterization(
        self,
        candidate: SearchCandidate,
        key: ResultKey,
        *,
        atoms: Any | None = None,
    ) -> None:
        if not self._task_postprocess_enabled(candidate.base_request.task.name):
            return
        if key in self._local_parameterization_payloads or key in self._local_postprocess_failures:
            raise RuntimeError(f"duplicate owner-local postprocess preparation for {key}")
        try:
            payload_candidate = (
                candidate if atoms is None else _candidate_with_atoms(candidate, atoms)
            )
            payload = build_parameterization_payload(
                payload_candidate,
                task_name=key[0],
                sample_id=key[1],
            )
        except PostprocessPreparationError as error:
            self._local_postprocess_failures[key] = str(error)
            return
        except (RuntimeError, ValueError) as error:
            # A finalized fold has crossed a private-chemistry UID boundary;
            # stale sidecars are sample-local postprocess failures.  Preserve
            # the stricter behavior for ordinary terminal output, where this
            # indicates an internal candidate invariant instead.
            if atoms is None:
                raise
            self._local_postprocess_failures[key] = (
                f"{type(error).__name__}: {error}"
            )
            return
        if payload.key != key:
            raise RuntimeError("parameterization payload key differs from output slot")
        self._local_parameterization_payloads[key] = payload

    def predict_step(
        self,
        batch: tuple[PredictBatch | None, tuple[SampleFailure, ...]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, tuple[Any, ...]]:
        predict_batch, prepared_failures = batch
        failures = list(prepared_failures)
        self._local_failures.extend(_failure_wire(item) for item in failures)
        row_replicas = (
            int(self.search_config["best_of_n"]["replicas"])
            if self.search_algorithm == "best_of_n"
            else 1
        )
        for failure in failures:
            key = (failure.task_name, failure.sample_id)
            request = self._request_by_key.get(key)
            if request is None:
                raise RuntimeError("prepared failure has no expanded request")
            for _replica in range(row_replicas):
                self._local_result_rows.append(
                    _failed_request_row(
                        request,
                        reward_method=self.reward_method,
                        stage=failure.stage,
                        message=failure.message,
                    )
                )
        if predict_batch is None:
            return {}

        _require_bond_order_checkpoint_support(
            self.inference_model,
            predict_batch.canonical,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        if self.search_algorithm is None:
            self._predict_plain(predict_batch)
        elif self.search_algorithm == "best_of_n":
            self._predict_best_of_n(predict_batch)
        elif self.search_algorithm == "fk_steering":
            self._predict_fk(predict_batch)
        else:
            raise DesignConfigError(
                f"runtime search algorithm is not implemented: {self.search_algorithm!r}"
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._local_batch_seconds.append(time.perf_counter() - started)
        return {}

    def _predict_plain(self, predict_batch: PredictBatch) -> None:
        failures: list[SampleFailure] = []
        succeeded: list[ResultKey] = []

        terminal_results = tuple(
            self.terminal_candidate_factory(
                self.inference_model,
                predict_batch,
                sampling_config=self.sampling_config,
                ligand_graph_method=self.ligand_graph_method,
            )
        )
        expected = {_request_key(request) for request in predict_batch.requests}
        compiled_by_key = {
            _request_key(compiled.request): compiled
            for compiled in predict_batch.samples
        }
        if set(compiled_by_key) != expected:
            raise RuntimeError("compiled plain batch IDs differ from its requests")
        result_by_key: dict[ResultKey, TerminalResult] = {}
        for result in terminal_results:
            if not isinstance(result, (SearchCandidate, SampleFailure)):
                raise TypeError(
                    "terminal candidate factory returned an unsupported result"
                )
            key = _result_key(result)
            if key in result_by_key:
                raise RuntimeError(f"terminal factory duplicated result {key}")
            result_by_key[key] = result
        if set(result_by_key) != expected:
            raise RuntimeError(
                "terminal factory result IDs differ from the compiled batch: "
                f"missing={sorted(expected-set(result_by_key))}, "
                f"extra={sorted(set(result_by_key)-expected)}"
            )

        for key in sorted(expected):
            result = result_by_key[key]
            if isinstance(result, SampleFailure):
                failures.append(result)
                self._local_result_rows.append(
                    _failed_replica_row(
                        compiled_by_key[key],
                        reward_method="",
                        stage=result.stage,
                        message=result.message,
                    )
                )
                continue
            ledger_entries: tuple[dict[str, Any], ...] | None = None
            try:
                ledger_entries = _build_design_ledger(
                    compiled_by_key[key],
                    result,
                )
                identity = _candidate_output_identity(
                    result,
                    global_rank=None,
                    output_format=self.output_format,
                    ledger_entries=ledger_entries,
                )
                (
                    filename,
                    base_id,
                    modality,
                    chain_id,
                    token_length,
                    sequence,
                    contig,
                    tip,
                ) = identity
                output_path = (
                    self.output_dir / result.base_request.task.name / filename
                )
                constraint_fields = _constraint_output_fields(
                    result, result.atoms, output_format=self.output_format,
                )
                design_chains = _design_chains_json(result, result.atoms)
                antibody_fields = _write_candidate_output(
                    result, result.atoms, output_path,
                    entry_id=f"{result.base_request.task.name}_{result.base_request.sample_id}",
                )
            except (OSError, StructureWriteError, ValueError) as exc:
                failures.append(
                    SampleFailure(key[0], key[1], "write", str(exc))
                )
                self._local_result_rows.append(
                    self._failed_candidate_row(
                        result,
                        stage="write",
                        message=str(exc),
                        ledger_entries=ledger_entries,
                    )
                )
            else:
                # Payload identity follows the actual file written.  Search
                # winners may carry a rank-qualified filename; the base
                # request sample ID is intentionally not a postprocess key.
                self._capture_parameterization(result, (key[0], filename))
                succeeded.append(key)
                self._local_result_rows.append(
                    (
                        result.base_request.task.name,
                        filename,
                        base_id,
                        modality,
                        chain_id,
                        token_length,
                        "",
                        "",
                        sequence,
                        contig,
                        tip,
                        "ok",
                        "",
                        "",
                        "{}",
                        _ledger_json(ledger_entries),
                        json.dumps({**_candidate_generation_metadata(result), **constraint_fields, **antibody_fields}, allow_nan=False),
                        design_chains,
                    )
                )
        self._local_succeeded.extend(succeeded)
        self._local_failures.extend(_failure_wire(item) for item in failures)

    def _predict_best_of_n(self, predict_batch: PredictBatch) -> None:
        replicas = int(self.search_config["best_of_n"]["replicas"])
        for compiled in predict_batch.samples:
            lengths = actual_joint_length(compiled)
            request = compiled.request
            self._local_slot_groups.append(
                (
                    request.task.name,
                    request.sample_id,
                    request.global_ordinal,
                    lengths,
                )
            )

        specs = [
            (compiled, replica)
            for compiled in predict_batch.samples
            for replica in range(replicas)
        ]
        configured_chunk = self.search_config.get("max_batch_size")
        chunk_size = len(specs) if configured_chunk is None else int(configured_chunk)
        for start in range(0, len(specs), chunk_size):
            chunk = specs[start : start + chunk_size]
            repeated = collate_compiled_samples(
                [compiled for compiled, _ in chunk], token_buckets=self.token_buckets,
            )
            repeated = _move_predict_batch(repeated, self.device)
            terminal_results = tuple(
                self.terminal_candidate_factory(
                    self.inference_model,
                    repeated,
                    sampling_config=self.sampling_config,
                    ligand_graph_method=self.ligand_graph_method,
                )
            )
            if len(terminal_results) != len(chunk):
                raise RuntimeError(
                    "Best-of-N terminal factory must return one ordered result per replica"
                )
            for (compiled, replica), result in zip(chunk, terminal_results):
                if not isinstance(result, (SearchCandidate, SampleFailure)):
                    raise TypeError(
                        "terminal candidate factory returned an unsupported result"
                    )
                if _result_key(result) != _request_key(compiled.request):
                    raise RuntimeError(
                        "Best-of-N terminal result does not match replica request order"
                    )
                if isinstance(result, SampleFailure):
                    self._local_result_rows.append(
                        _failed_replica_row(
                            compiled,
                            reward_method=self.reward_method,
                            stage=result.stage,
                            message=result.message,
                        )
                    )
                    continue
                candidate = _numbered_search_candidate(
                    compiled,
                    result,
                    terminal_local_ordinal=replica,
                    terminal_width=replicas,
                )
                if candidate.candidate_id in self._local_candidates:
                    raise RuntimeError(
                        f"duplicate Best-of-N candidate ID {candidate.candidate_id}"
                    )
                self._local_candidates[candidate.candidate_id] = candidate
                self._local_design_ledgers[candidate.candidate_id] = _build_design_ledger(
                    compiled,
                    candidate,
                )
                self._write_original_candidate(candidate, replica=replica)

    def _write_original_candidate(
        self, candidate: SearchCandidate, *, replica: int,
    ) -> None:
        """Persist the decoded candidate before any deferred sequence/reward work."""
        if not (self.deferred_fold_reward or self.deferred_partiscore_reward):
            return
        requested = str(self.reward_config.get("sequence", {}).get("source", "design"))
        if requested == "design":
            return
        try:
            if self.deferred_partiscore_reward:
                from ..rewards.sequence import resolve_reward_sequence_source

                source = resolve_reward_sequence_source(candidate, requested)
            else:
                source = resolve_sequence_source(candidate, requested)
        except ValueError:
            # Keep the original even if later sequence planning rejects it.
            # The deferred reward owns candidate-local routing failure reporting.
            source = requested
        if source == "design":
            return
        base_id = int(candidate.base_request.task_slot_ordinal) + 1
        filename = f"sample{base_id}_candidate{replica + 1}.{self.output_format}"
        path = self.output_dir / candidate.base_request.task.name / "original" / filename
        # This file is deliberately absent from scoring/postprocess queues and
        # result rows. A write failure must not silently lose the requested copy.
        write_structure_atomic(
            candidate.atoms, path,
            entry_id=f"{candidate.base_request.task.name}_sample{base_id}_candidate{replica + 1}",
            validated_ligand_identities=candidate.validated_ligand_identities,
        )

    def _predict_fk(self, predict_batch: PredictBatch) -> None:
        if self.reward is None:
            raise AssertionError("FK requires a reward")
        beam_width = int(self.search_config["fk_steering"]["beam_width"])
        for compiled in predict_batch.samples:
            request = compiled.request
            lengths = actual_joint_length(compiled)
            self._local_slot_groups.append(
                (
                    request.task.name,
                    request.sample_id,
                    request.global_ordinal,
                    lengths,
                )
            )
            single = collate_compiled_samples((compiled,), token_buckets=self.token_buckets)
            single = _move_predict_batch(single, self.device)
            terminal_results = tuple(
                self.fk_candidate_factory(
                    self.inference_model,
                    compiled,
                    single,
                    sampling_config=self.sampling_config,
                    ligand_graph_method=self.ligand_graph_method,
                    search_config=self.search_config,
                    reward=self.reward,
                )
            )
            if len(terminal_results) > beam_width:
                raise RuntimeError("FK terminal factory exceeded beam_width")
            for terminal_local_ordinal, result in enumerate(terminal_results):
                if not isinstance(result, (SearchCandidate, SampleFailure)):
                    raise TypeError("FK terminal factory returned an unsupported result")
                if _result_key(result) != _request_key(request):
                    raise RuntimeError("FK terminal result does not match its request")
                if isinstance(result, SampleFailure):
                    continue
                candidate = _numbered_search_candidate(
                    compiled,
                    result,
                    terminal_local_ordinal=terminal_local_ordinal,
                    terminal_width=beam_width,
                )
                if candidate.candidate_id in self._local_candidates:
                    raise RuntimeError(f"duplicate FK candidate ID {candidate.candidate_id}")
                self._local_candidates[candidate.candidate_id] = candidate
                self._local_design_ledgers[candidate.candidate_id] = _build_design_ledger(
                    compiled,
                    candidate,
                )

    def on_predict_epoch_end(self) -> None:
        if self.deferred_fold_reward or self.deferred_partiscore_reward:
            # The FM must be offloaded before MPNN and structural reward models start.  Runtime
            # finishes search after Trainer.predict returns and all ranks have
            # entered the explicit offload barrier.
            return
        if self.search_algorithm in {"best_of_n", "fk_steering"}:
            self._finish_search()
        elif self.search_algorithm is not None:
            raise DesignConfigError(
                f"runtime search algorithm is not implemented: {self.search_algorithm!r}"
            )
        self._collect_global_accounting()

    def finish_deferred_partiscore_reward(
        self, *, repo_root: Path, work_root: Path,
        timing_logger: logging.Logger | None = None,
    ) -> None:
        """Replace candidates with sequence-consistent packed structures before ranking."""
        if not self.deferred_partiscore_reward:
            raise RuntimeError("deferred PartiScore reward is not enabled")
        from ..rewards.sequence import plan_reward_sequence_source
        from ..rewards.packing import pack_mpnn_candidates

        rank, _world_size = _distributed_rank_world()
        local_error = None
        prepared: dict[str, SearchCandidate] = {}
        failures: dict[str, str] = {}
        sources: dict[str, str] = {}
        try:
            sequence = self.reward_config["sequence"]
            planned = []
            for candidate in self._local_candidates.values():
                try:
                    plan = plan_reward_sequence_source(
                        candidate, requested=str(sequence["source"]),
                        user_fixed_positions=tuple(sequence.get("fixed_positions", ())),
                        fixed_interface=sequence.get("fixed_interface"),
                        reward_method="partiscore",
                    )
                except ValueError as error:
                    failures[candidate.candidate_id] = str(error)
                    continue
                sources[candidate.candidate_id] = plan.source
                planned.append((candidate, plan))
            sequence_started = time.perf_counter()
            sequence_timings: dict[str, float] = {}
            antibody_plans = {}
            if any(candidate.base_request.task.format == "antibody" for candidate, _plan in planned):
                from ..rewards.mpnn import build_antibody_mpnn_plan, MPNNSequenceError
                accepted = []
                for candidate, plan in planned:
                    try:
                        if candidate.base_request.task.format == "antibody":
                            antibody_plans[candidate.candidate_id] = build_antibody_mpnn_plan(
                                candidate, sequence_plan=plan,
                            )
                        accepted.append((candidate, plan))
                    except MPNNSequenceError as exc:
                        failures[candidate.candidate_id] = str(exc)
                        sources.pop(candidate.candidate_id, None)
                planned = accepted
            plans, sequence_failures = generate_mpnn_sequences(
                planned, reward_config=self.reward_config, repo_root=repo_root,
                work_dir=work_root / "mpnn", timings=sequence_timings,
                **({"antibody_plans": antibody_plans} if antibody_plans else {}),
            )
            if timing_logger is not None:
                source_counts = {source: list(sources.values()).count(source) for source in sorted(set(sources.values()))}
                timing_text = " ".join(f"{name}={value:.3f}s" for name, value in sorted(sequence_timings.items()))
                timing_logger.info(
                    "PartiScore sequence generation completed in %.3fs candidates=%d sources=%s %s",
                    time.perf_counter() - sequence_started, len(planned), source_counts, timing_text,
                )
            expected_planned = {candidate.candidate_id for candidate, _plan in planned}
            if set(plans) & set(sequence_failures) or set(plans) | set(sequence_failures) != expected_planned:
                raise RuntimeError("MPNN candidate accounting is incomplete or duplicated")
            failures.update(sequence_failures)
            pack_items = [(candidate, plans[candidate.candidate_id])
                          for candidate, _plan in planned if candidate.candidate_id in plans]
            packing_started = time.perf_counter()
            packing_timings: dict[str, float] = {}
            prepared, packing_failures = pack_mpnn_candidates(
                pack_items, reward_config=self.reward_config, repo_root=repo_root,
                work_dir=work_root / "packing", timings=packing_timings,
                **({"antibody_plans": {candidate.candidate_id: antibody_plans[candidate.candidate_id]
                    for candidate, _plan in pack_items if candidate.candidate_id in antibody_plans}} if antibody_plans else {}),
            )
            if timing_logger is not None:
                timing_logger.info(
                    "PartiScore packing completed in %.3fs candidates=%d runtime_load=%.3fs batch_work=%.3fs",
                    time.perf_counter() - packing_started, len(pack_items),
                    packing_timings.get("runtime_load_seconds", 0.0),
                    packing_timings.get("batch_work_seconds", 0.0),
                )
            if set(prepared) & set(packing_failures) or set(prepared) | set(packing_failures) != set(plans):
                raise RuntimeError("packing candidate accounting is incomplete or duplicated")
            failures.update(packing_failures)
            if set(prepared) & set(failures) or set(prepared) | set(failures) != set(self._local_candidates):
                raise RuntimeError("PartiScore preparation lost a candidate")
            for candidate_id, candidate in prepared.items():
                original = self._local_candidates[candidate_id]
                if candidate.candidate_id != candidate_id or (
                    candidate.candidate_ordinal != original.candidate_ordinal
                    or candidate.base_request is not original.base_request
                    or candidate.actual_joint_length != original.actual_joint_length
                ):
                    raise RuntimeError("packing changed candidate identity or ownership")
            if any(not isinstance(message, str) or not message for message in failures.values()):
                raise RuntimeError("PartiScore preparation returned an invalid failure")
        except Exception as error:
            local_error = f"rank {rank}: {type(error).__name__}: {error}"
        errors = _all_gather_objects(local_error)
        if any(error is not None for error in errors):
            raise RuntimeError("PartiScore preparation failed: " + "; ".join(error for error in errors if error is not None))
        for candidate_id, message in failures.items():
            self._local_result_rows.append(self._failed_candidate_row(
                self._local_candidates[candidate_id], stage="reward", message=message,
            ))
        self._local_candidates = prepared
        self._local_sequence_sources = {cid: sources[cid] for cid in prepared}
        ledger_error = None
        try:
            # Keep source sequence and token counts from compilation, update only final identity.
            for cid, candidate in prepared.items():
                if sources[cid] == "design":
                    continue
                entries = self._local_design_ledgers.get(cid, ())
                self._local_design_ledgers[cid] = tuple(
                    {**entry, "final_sequence": _sequence_from_atoms(
                        candidate.atoms, chain_id=str(entry["chain"]), modality=str(entry["modality"]),
                        model_parent_residue=candidate.model_parent_residue,
                        fixed_modified_residue_names=candidate.fixed_modified_residue_names,
                        private_ligand_molecules=candidate.private_ligand_molecules,
                        private_ligand_errors=candidate.private_ligand_errors,
                        design_only=False,
                    )} for entry in entries
                )
        except Exception as error:
            ledger_error = f"rank {rank}: {type(error).__name__}: {error}"
        errors = _all_gather_objects(ledger_error)
        if any(error is not None for error in errors):
            raise RuntimeError("packed candidate ledger failed: " + "; ".join(error for error in errors if error is not None))
        self._finish_search(timing_logger=timing_logger)
        self._collect_global_accounting()

    def finish_deferred_fold_reward(
        self,
        *,
        repo_root: Path,
        work_root: Path,
    ) -> None:
        if not self.deferred_fold_reward:
            raise RuntimeError("deferred fold reward is not enabled")
        rank, _world_size = _distributed_rank_world()
        results: tuple[FoldRewardResult, ...] = ()
        local_error: str | None = None
        try:
            results, fold_failures = self.fold_reward_runner(
                tuple(self._local_candidates.values()),
                reward_config=self.reward_config,
                repo_root=repo_root,
                owner_work_root=work_root,
                **({"stage_output_root": self.output_dir}
                   if any(candidate.base_request.task.format == "antibody" for candidate in self._local_candidates.values())
                   and self.reward_config.get("_target_") in {"src.design.rewards.esmfold2.ESMFold2Reward", "src.design.rewards.af3.AF3Reward"} else {}),
            )
        except Exception as error:
            local_error = f"rank {rank}: {type(error).__name__}: {error}"
        errors = _all_gather_objects(local_error)
        real_errors = [error for error in errors if error is not None]
        if real_errors:
            raise RuntimeError(
                "fold reward worker failed: " + "; ".join(real_errors)
            )
        self._local_fold_results = {result.candidate_id: result for result in results}
        if len(self._local_fold_results) != len(results):
            raise RuntimeError("fold reward duplicated a candidate result")
        failed_ids: set[str] = set()
        for failure in fold_failures:
            candidate_id = getattr(failure, "candidate_id", None)
            message = getattr(failure, "message", None)
            if not isinstance(candidate_id, str) or not isinstance(message, str):
                raise RuntimeError("fold reward returned an invalid failure")
            if candidate_id in failed_ids or candidate_id in self._local_fold_results:
                raise RuntimeError("fold reward duplicated a candidate outcome")
            candidate = self._local_candidates.get(candidate_id)
            if candidate is None:
                raise RuntimeError("fold reward failed an unknown candidate")
            failed_ids.add(candidate_id)
            self._local_result_rows.append(
                self._failed_candidate_row(
                    candidate,
                    stage="reward",
                    message=message,
                )
            )
        expected_ids = set(self._local_candidates)
        observed_ids = set(self._local_fold_results) | failed_ids
        if observed_ids != expected_ids:
            raise RuntimeError(
                "fold reward candidate accounting is incomplete: "
                f"missing={sorted(expected_ids-observed_ids)}, "
                f"extra={sorted(observed_ids-expected_ids)}"
            )
        self._local_candidates = {
            candidate_id: candidate
            for candidate_id, candidate in self._local_candidates.items()
            if candidate_id in self._local_fold_results
        }
        score_override = {result.candidate_id: result.score for result in results}
        if self.reward_method in {"esmfold2", "af3", "af3-la"} and self.reward_config.get("score", {}).get("method", "confidence") == "partiscore":
            local_error = None
            try:
                prepared = []
                prepared_ids = []
                for candidate_id, candidate in self._local_candidates.items():
                    try:
                        atoms = self._prepare_fold_output(candidate, self._local_fold_results[candidate_id])
                    except (DesignInputError, OSError, StructureWriteError, ValueError) as error:
                        self._local_result_rows.append(self._failed_candidate_row(candidate, stage="reward", message=str(error)))
                        continue
                    prepared.append(atoms)
                    prepared_ids.append(candidate_id)
                scored = self._get_partiscore().score_structures(
                    prepared, identities=prepared_ids,
                    design_chains=[[chain.chain_id for chain in self._local_candidates[cid].base_request.task.design_chains] for cid in prepared_ids],
                    scope="design_groups",
                ) if prepared else []
                if [result.identity for result in scored] != prepared_ids:
                    raise RuntimeError("fold PartiScore results differ from prepared IDs")
                score_override = {}
                for result in scored:
                    candidate = self._local_candidates[result.identity]
                    antibody_metrics = {}
                    if candidate.base_request.task.format == "antibody":
                        from ..antibody.scoring import antibody_binder_chains, antibody_score_components, antibody_score_record
                        binder = antibody_binder_chains(candidate)
                        antibody_metrics = antibody_score_record(result, binder)
                        result, _internal = antibody_score_components(result, binder)
                    if result.status in {"error", "no_interface"}:
                        self._local_result_rows.append(self._failed_candidate_row(
                            self._local_candidates[result.identity], stage="reward",
                            message=result.error or f"PartiScore: {result.status}",
                        ))
                        continue
                    score_override[result.identity] = ranking_score(result)
                    self._local_reward_records[result.identity] = {**reward_record(result, source="fold"), **antibody_metrics}
                self._local_candidates = {cid: candidate for cid, candidate in self._local_candidates.items() if cid in score_override}
            except Exception as error:
                local_error = f"rank {rank}: {type(error).__name__}: {error}"
            errors = _all_gather_objects(local_error)
            if any(error is not None for error in errors):
                raise RuntimeError("fold PartiScore failed: " + "; ".join(error for error in errors if error is not None))
        self._finish_search(score_by_id_override=score_override)
        self._collect_global_accounting()

    def _failed_candidate_row(
        self,
        candidate: SearchCandidate,
        *,
        stage: str,
        message: str,
        global_rank: int | None = None,
        ledger_entries: Sequence[Mapping[str, Any]] | None = None,
    ) -> ResultRowWire:
        request = candidate.base_request
        modality, chain_id, token_length = _legacy_owner_scalars(candidate)
        if _is_joint_request(request):
            sequence = ""
        else:
            try:
                sequence = _candidate_design_sequence(
                    candidate, chain_id=chain_id, modality=modality
                )
            except ValueError:
                sequence = ""
        contig, tip = _motif_output_fields(candidate, chain_id, modality)
        if ledger_entries is None:
            ledger_entries = self._local_design_ledgers.get(candidate.candidate_id)
        if ledger_entries is None:
            ledger_entries = _candidate_fallback_ledger(candidate)
        return (
            request.task.name,
            "",
            int(request.task_slot_ordinal) + 1,
            modality,
            chain_id,
            token_length,
            self.reward_method,
            "" if global_rank is None else str(global_rank),
            sequence,
            contig,
            tip,
            "failed",
            stage,
            message,
            "{}",
            _ledger_json(ledger_entries),
            json.dumps(_candidate_generation_metadata(candidate), allow_nan=False),
            "",
        )

    def local_deferred_fold_counts(self) -> tuple[int, int]:
        """Return primitive owner-local candidate/result counts for progress logs."""

        if not self.deferred_fold_reward:
            return (0, 0)
        return (len(self._local_candidates), len(self._local_fold_results))

    def _collect_parameterization_payloads(self) -> None:
        rank, world_size = _distributed_rank_world()
        local_error: str | None = None
        local: tuple[
            tuple[SampleParameterizationPayload, ...],
            tuple[PostprocessFailureWire, ...],
        ] | None = None
        try:
            local_success = self._postprocess_result_keys(self._local_result_rows)
            local_payload_keys = set(self._local_parameterization_payloads)
            local_failure_keys = set(self._local_postprocess_failures)
            if local_payload_keys & local_failure_keys:
                raise RuntimeError("postprocess payload/failure keys overlap on one owner")
            if local_payload_keys | local_failure_keys != local_success:
                raise RuntimeError(
                    "owner-local postprocess preparation differs from base-success IDs"
                )
            local = (
                tuple(self._local_parameterization_payloads.values()),
                tuple(
                    (key[0], key[1], message)
                    for key, message in self._local_postprocess_failures.items()
                ),
            )
        except Exception as error:
            local_error = f"rank {rank}: {type(error).__name__}: {error}"
        local_errors = _all_gather_objects(local_error)
        real_local_errors = [error for error in local_errors if error is not None]
        if real_local_errors:
            raise RuntimeError(
                "owner-local postprocess preparation is invalid: "
                + "; ".join(real_local_errors)
            )
        if local is None:
            raise RuntimeError("owner-local postprocess preparation was not produced")

        gathered = _gather_objects_to_rank_zero(local)
        payloads: tuple[SampleParameterizationPayload, ...] = ()
        failures: tuple[PostprocessFailureWire, ...] = ()
        root_error: Exception | None = None
        root_status: tuple[str, str] | None = None
        if rank == 0:
            try:
                if gathered is None or len(gathered) != world_size:
                    raise RuntimeError("rank-zero private payload gather is incomplete")
                payloads = tuple(
                    payload
                    for rank_payload in gathered
                    for payload in rank_payload[0]
                )
                failures = tuple(
                    failure
                    for rank_payload in gathered
                    for failure in rank_payload[1]
                )
                payload_keys = [payload.key for payload in payloads]
                failure_keys = [(failure[0], failure[1]) for failure in failures]
                all_keys = payload_keys + failure_keys
                if len(all_keys) != len(set(all_keys)):
                    raise RuntimeError(
                        "global postprocess preparation duplicated a sample key"
                    )
                root_status = ("ok", "")
            except Exception as error:
                root_error = error
                root_status = ("fatal", f"{type(error).__name__}: {error}")
        if world_size > 1:
            status_list = [root_status]
            torch.distributed.broadcast_object_list(status_list, src=0)
            root_status = status_list[0]
        if root_status is None:
            raise RuntimeError("rank zero did not validate private postprocess payloads")
        if root_status[0] == "fatal":
            if rank == 0 and root_error is not None:
                raise root_error
            raise RuntimeError(
                f"rank-zero private payload validation failed: {root_status[1]}"
            )
        self._global_parameterization_payloads = payloads if rank == 0 else ()
        self._global_postprocess_failures = failures if rank == 0 else ()

    def _collect_global_accounting(self) -> None:
        rank, world_size = _distributed_rank_world()
        if self._task_score_enabled:
            scoring_error = None
            try:
                self._score_local_outputs()
            except Exception as error:
                scoring_error = f"rank {rank}: {type(error).__name__}: {error}"
            errors = _all_gather_objects(scoring_error)
            if any(error is not None for error in errors):
                raise RuntimeError("final-file PartiScore failed: " + "; ".join(error for error in errors if error is not None))
        local = (
            tuple(self._local_succeeded),
            tuple(self._local_failures),
            tuple((rank, index, value) for index, value in enumerate(self._local_batch_seconds)),
            tuple(self._local_result_rows),
        )
        if self._task_score_enabled:
            local += (tuple((task, filename, json.dumps(record, allow_nan=False))
                            for (task, filename), record in self._local_score_records.items()),)
        gathered = _all_gather_objects(local)
        if world_size > 1:
            torch.distributed.barrier()
        self._global_succeeded = tuple(
            key for item in gathered if item is not None for key in item[0]
        )
        self._global_failures = tuple(
            wire for item in gathered if item is not None for wire in item[1]
        )
        self._global_batch_seconds = tuple(
            timing for item in gathered if item is not None for timing in item[2]
        )
        self._global_result_rows = tuple(
            row for item in gathered if item is not None for row in item[3]
        )
        if self._task_score_enabled:
            scoring = [entry for item in gathered if item is not None for entry in item[4]]
            keys = [(task, filename) for task, filename, _record in scoring]
            if len(set(keys)) != len(keys):
                raise RuntimeError("duplicate global PartiScore output identity")
            scored_tasks = {
                request.task.name
                for request in self._request_by_key.values()
                if request.task.score
                and not self._task_postprocess_enabled(request.task.name)
            }
            expected = [(row[0], row[1]) for row in self._global_result_rows if row[0] in scored_tasks and row[11] == "ok"]
            if len(set(expected)) != len(expected) or set(keys) != set(expected):
                raise RuntimeError("global PartiScore accounting differs from successful output IDs")
            self._global_score_records = {(task, filename): json.loads(record) for task, filename, record in scoring}
        if not self._task_score_enabled:
            if rank == 0 and self._global_result_rows:
                self._write_result_csv()
            return
        publication_error = None
        publication_status = None
        if rank == 0:
            try:
                if self._global_result_rows:
                    self._write_result_csv()
                publication_status = ("ok", "")
            except Exception as error:
                publication_error = error
                publication_status = ("fatal", f"{type(error).__name__}: {error}")
        if world_size > 1:
            statuses = [publication_status]
            torch.distributed.broadcast_object_list(statuses, src=0)
            publication_status = statuses[0]
        if publication_status is None:
            raise RuntimeError("rank zero did not publish result-table status")
        if publication_status[0] == "fatal":
            if publication_error is not None:
                raise publication_error
            raise RuntimeError(f"rank-zero result-table publication failed: {publication_status[1]}")

    def _write_interrupted_result_csv(self) -> None:
        """Publish existing final rows after failure, without scoring or collectives."""
        rank, world_size = _distributed_rank_world()
        if rank != 0:
            return
        if not self._global_result_rows:
            # Unselected search candidates are not final output slots. A failed
            # DDP rank must not attempt a collective or publish a partial shard.
            if (
                self.search_algorithm is not None
                or world_size != 1
                or _launcher_world_size() != 1
                or not self._local_result_rows
            ):
                return
            self._global_result_rows = tuple(self._local_result_rows)
            self._global_score_records = dict(self._local_score_records)
        self._write_result_csv(interrupted=True)

    def _write_result_csv(
        self,
        *,
        interrupted: bool = False,
        affected_tasks: Sequence[str] | None = None,
    ) -> None:
        task_by_name = {r.task.name: r.task for r in self._request_by_key.values()}
        affected_task_names = None if affected_tasks is None else {
            str(task_name) for task_name in affected_tasks
        }
        if affected_task_names is not None:
            unknown_tasks = affected_task_names.difference(self._task_names)
            if unknown_tasks:
                raise RuntimeError(
                    "result CSV affected-task filter contains unknown tasks: "
                    f"{sorted(unknown_tasks)}"
                )
        antibody_tasks = {name for name, task in task_by_name.items() if getattr(task, "format", "normal") == "antibody"}
        length_metadata_by_base = {
            (request.task.name, int(request.task_slot_ordinal) + 1): dict(request.length_metadata)
            for request in self._request_by_key.values()
        }
        base_fields = [
            "filename",
            "design_chains",
            "base_sample_id",
            "design_modality",
            "design_chain_id",
            "token_length",
            "reward_method",
            "global_rank",
            "design_sequence",
            "contig_string",
            "tip_atoms",
            "status",
            "failure_stage",
            "failure_message",
        ]
        decoded_by_task: dict[str, list[dict[str, Any]]] = {
            task_name: [] for task_name in self._task_names
        }
        has_partiscore_reward = self.reward_method == "partiscore" or (
            self.deferred_fold_reward
            and self.reward_config.get("score", {}).get("method") == "partiscore"
        )
        metric_fields_by_task: dict[str, set[str]] = {
            task_name: ({"partiscore_score"} if has_partiscore_reward else set())
            for task_name in self._task_names
        }
        multi_task_by_name = {
            task_name: any(
                _is_joint_request(request)
                for request in self._request_by_key.values()
                if request.task.name == task_name
            )
            for task_name in self._task_names
        }
        result_rows = list(self._global_result_rows)
        represented = {(row[0], int(row[2])) for row in result_rows}
        antibody_failures = (getattr(self, "_global_failures", None) or ()) if antibody_tasks else ()
        requests_by_failure = ({(r.task.name, r.sample_id): r for r in self._request_by_key.values()}
                               if antibody_failures else {})
        for failure in antibody_failures:
            request = requests_by_failure.get((failure[0], failure[1]))
            if request is not None and request.task.name in antibody_tasks:
                identity = (request.task.name, int(request.task_slot_ordinal) + 1)
                if identity not in represented:
                    result_rows.append(_failed_request_row(request, reward_method=self.reward_method,
                                                          stage=failure[2], message=failure[3]))
                    represented.add(identity)
        for row in result_rows:
            task_name = row[0]
            if task_name not in decoded_by_task:
                raise RuntimeError(
                    f"result row belongs to an unknown task {task_name!r}"
                )
            if affected_task_names is not None and task_name not in affected_task_names:
                continue
            legacy_fields = [name for name in base_fields if name != "design_chains"]
            values = dict(zip(legacy_fields, row[1:14], strict=True))
            raw_design_chains = row[17] if len(row) >= 18 else ""
            if values["status"] == "ok":
                if not raw_design_chains:
                    raise RuntimeError("successful result row has no design_chains metadata")
                try:
                    parsed_design_chains = json.loads(raw_design_chains)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("result design_chains payload is not valid JSON") from exc
                if not isinstance(parsed_design_chains, list):
                    raise RuntimeError("result design_chains payload must be a list")
                seen_labels: set[str] = set()
                for index, entry in enumerate(parsed_design_chains):
                    if not isinstance(entry, Mapping):
                        raise RuntimeError(
                            f"result design_chains[{index}] payload must be an object"
                        )
                    missing = {"chain", "modality", "sequence", "residue_count"}.difference(entry)
                    if missing:
                        raise RuntimeError(
                            f"result design_chains[{index}] is missing {sorted(missing)}"
                        )
                    label = entry["chain"]
                    modality = entry["modality"]
                    sequence_value = entry["sequence"]
                    residue_count = entry["residue_count"]
                    if (
                        not isinstance(label, str)
                        or not label
                        or label in seen_labels
                        or not isinstance(modality, str)
                        or modality not in {"protein", "rna", "dna", "ligand"}
                        or not isinstance(sequence_value, str)
                        or not sequence_value
                        or isinstance(residue_count, bool)
                        or not isinstance(residue_count, int)
                        or residue_count < 1
                    ):
                        raise RuntimeError(
                            f"result design_chains[{index}] payload is invalid"
                        )
                    if "sub_modality" in entry and (
                        modality != "protein" or entry["sub_modality"] != "peptide"
                    ):
                        raise RuntimeError("result design-chain sub_modality is invalid")
                    if "cyclic_offset" in entry and (
                        modality != "protein" or type(entry["cyclic_offset"]) is not bool
                        or (entry["cyclic_offset"] and residue_count < 3)
                    ):
                        raise RuntimeError("result design-chain cyclic_offset is invalid")
                    seen_labels.add(label)
                values["design_chains"] = raw_design_chains
            else:
                values["design_chains"] = ""
            metrics = json.loads(row[14])
            if not isinstance(metrics, dict):
                raise RuntimeError("result metric payload must be a mapping")
            if len(row) >= 16:
                ledger = row[15]
            else:
                ledger = "[]"
            if not isinstance(ledger, str):
                raise RuntimeError("result design ledger payload must be text")
            if multi_task_by_name[task_name]:
                parsed_ledger = json.loads(ledger)
                if not isinstance(parsed_ledger, list):
                    raise RuntimeError("result design ledger payload must be a list")
            for name, value in metrics.items():
                if not isinstance(name, str) or not isinstance(value, (int, float)):
                    raise RuntimeError("result metric payload is not primitive")
                if not math.isfinite(float(value)):
                    raise RuntimeError("result metric payload contains non-finite values")
            if set(metrics).intersection((*base_fields, "error", "postprocess_status", "fixed_restypes", "bond_constraints", "final_partiscore", "score_rank")):
                raise RuntimeError("result metric name conflicts with a public result field")
            values.update(metrics)
            metric_fields_by_task[task_name].update(metrics)
            metadata = dict(length_metadata_by_base.get((task_name, int(row[2])), {}))
            metadata.update(json.loads(row[16]) if len(row) >= 17 else {})
            metadata.update(self._global_score_records.get((task_name, row[1]), {}).get("metadata", {}))
            if task_by_name[task_name].score:
                # Intermediate status writes must not advertise provisional
                # ranks, even when some final-file scores are already present.
                finalized = task_name in getattr(self, "_score_ranking_finalized_tasks", set())
                values["final_partiscore"] = metadata.get("final_partiscore", "") if finalized else ""
                values["score_rank"] = metadata.get("score_rank", "") if finalized else ""
            for name, value in metadata.items():
                if not isinstance(name, str) or (value is not None and not isinstance(value, (str, int, float, bool))):
                    raise RuntimeError("scoring metadata is not JSON primitive")
                if isinstance(value, float) and not math.isfinite(value):
                    raise RuntimeError("scoring metadata contains a non-finite number")
            postprocess_key = (task_name, str(row[1]))
            postprocess_enabled = self._task_postprocess_enabled(task_name)
            values["postprocess_status"] = (
                "disabled" if not postprocess_enabled else
                "not_run" if values["status"] != "ok" or not values["filename"] else
                self._global_postprocess_status.get(postprocess_key, "pending")
            )
            # Keep the detailed internal accounting above, but expose one compact
            # public row. Errors from different stages share one readable cell.
            errors: list[str] = []
            if values["status"] != "ok":
                stage = values["failure_stage"] or "inference"
                errors.append(f"{stage}: {values['failure_message'] or values['status']}")
            if self._global_postprocess_status.get(postprocess_key) == "failed":
                errors.append(
                    "postprocess: "
                    + (self._global_postprocess_messages.get(postprocess_key) or "failed")
                )
            for prefix in ("partiscore", "refined_partiscore"):
                message = metadata.get(f"{prefix}_error")
                if message or metadata.get(f"{prefix}_status") == "error":
                    errors.append(f"partiscore: {message or 'failed'}")
            for name in ("fixed_restypes", "bond_constraints"):
                values[name] = metadata.get(name, "") if values["status"] == "ok" else ""
            values["error"] = " | ".join(dict.fromkeys(errors))
            if task_name in antibody_tasks:
                from ..antibody.results import ANTIBODY_RESULT_FIELDS
                antibody_values = json.loads(metadata.get("_antibody_result", "{}"))
                final_scores = self._global_score_records.get((task_name, row[1]), {}).get("metadata", {})
                # Reuse already-computed reward scores only for the same final
                # coordinate source, with no successful subsequent refinement.
                if not final_scores and metadata.get("reward_structure_source") == antibody_values.get("structure_source") and values["postprocess_status"] in {"disabled", "failed"}:
                    final_scores = metadata
                for prefix in ("partiscore_antigen", "partiscore_internal"):
                    if final_scores.get(f"{prefix}_status") == "error":
                        message = final_scores.get(f"{prefix}_error") or "failed"
                        errors.append(f"{prefix}: {message}")
                antibody_values.update({name: final_scores[name] for name in (
                    "partiscore_antigen", "partiscore_antigen_pair_count",
                    "partiscore_internal", "partiscore_internal_pair_count") if name in final_scores})
                stages = [values["failure_stage"]] if values["failure_stage"] else []
                if self._global_postprocess_status.get(postprocess_key) == "failed":
                    stages.append("postprocess")
                if any(final_scores.get(f"{prefix}_status") == "error" for prefix in ("partiscore", "refined_partiscore", "partiscore_antigen", "partiscore_internal")):
                    stages.append("score")
                antibody_values.update(filename=values["filename"], reward_method=values["reward_method"],
                                       rank=values["global_rank"],
                                       status=("failed" if values["status"] != "ok" else "partial" if errors else "ok"),
                                       postprocess_status=values["postprocess_status"], error_stage=";".join(dict.fromkeys(stages)),
                                       error=" | ".join(dict.fromkeys(errors)))
                if task_by_name[task_name].score:
                    antibody_values.update(final_partiscore=values["final_partiscore"], score_rank=values["score_rank"])
                values["_antibody_public"] = {name: antibody_values.get(name, "") for name in ANTIBODY_RESULT_FIELDS}
            decoded_by_task[task_name].append(values)
        for task_name, decoded in decoded_by_task.items():
            if affected_task_names is not None and task_name not in affected_task_names:
                continue
            if interrupted and not decoded:
                continue
            fields = (
                base_fields[:11] + ["fixed_restypes", "bond_constraints"]
                + sorted(metric_fields_by_task[task_name])
                + (["final_partiscore", "score_rank"] if task_by_name[task_name].score else [])
                + ["postprocess_status", "error"]
            )
            decoded.sort(
                key=lambda item: (
                    int(item["base_sample_id"]),
                    int(item["global_rank"] or 10**9),
                    str(item["status"]),
                    str(item["filename"]),
                )
            )
            if task_name in getattr(self, "_score_ranking_finalized_tasks", set()):
                decoded.sort(key=lambda item: (item.get("score_rank", "") == "", item.get("score_rank") or 0))
            if task_name in antibody_tasks:
                from ..antibody.results import antibody_result_fields
                fields = list(antibody_result_fields(score=bool(task_by_name[task_name].score)))
                public_rows = [item["_antibody_public"] for item in decoded]
            else:
                public_rows = decoded
            path = self.output_dir / task_name / "result.csv"
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=fields, extrasaction="raise"
                )
                writer.writeheader()
                writer.writerows({name: item.get(name, "") for name in fields} for item in public_rows)
            os.replace(temporary, path)
            interfaces = [
                interface for (task, _filename), record in self._global_score_records.items()
                if task == task_name for interface in record["interfaces"]
            ]
            if task_name not in antibody_tasks and not interrupted and any(request.task.name == task_name and request.task.score for request in self._request_by_key.values()):
                interface_path = self.output_dir / task_name / "partiscore.csv"
                interfaces = [{name: value for name, value in item.items() if name != "status"}
                              for item in interfaces]
                interface_fields = ["filename", "source", "partner_a", "partner_b", "score", "mean_log_density", "density_sum", "pair_count", "error"]
                interface_fields += sorted(set().union(*(set(item) for item in interfaces)) - set(interface_fields))
                temporary = interface_path.with_suffix(".csv.tmp")
                with temporary.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=interface_fields)
                    writer.writeheader()
                    writer.writerows(interfaces)
                os.replace(temporary, interface_path)

    def _finish_search(
        self,
        *,
        score_by_id_override: Mapping[str, float] | None = None,
        timing_logger: logging.Logger | None = None,
    ) -> None:
        if self.reward is None:
            raise AssertionError("search requires a reward")
        rank, world_size = _distributed_rank_world()
        gathered_slots = _all_gather_objects(tuple(self._local_slot_groups))
        slots: list[SlotWire] = [
            slot for rank_slots in gathered_slots for slot in rank_slots
        ]
        slot_by_key: dict[ResultKey, SlotWire] = {}
        slots_by_group: dict[GroupKey, list[SlotWire]] = {}
        for slot in slots:
            key = (slot[0], slot[1])
            if key in slot_by_key:
                raise RuntimeError(f"search slot group duplicated request {key}")
            slot_by_key[key] = slot
            group = (slot[0], tuple(slot[3]))
            slots_by_group.setdefault(group, []).append(slot)
        for group_slots in slots_by_group.values():
            group_slots.sort(key=lambda item: item[2])

        candidates = list(self._local_candidates.values())
        score_by_id: dict[str, float] = {}
        if score_by_id_override is not None:
            score_by_id = {
                str(candidate_id): float(score)
                for candidate_id, score in score_by_id_override.items()
            }
            if set(score_by_id) != {candidate.candidate_id for candidate in candidates}:
                raise RuntimeError("deferred reward scores differ from candidate IDs")
        elif isinstance(self.reward, PartiScoreReward):
            local_error = None
            try:
                score_started = time.perf_counter()
                scores = self.reward.score_batch(candidates, device="cpu")
                if timing_logger is not None:
                    timing_logger.info(
                        "PartiScore structural scoring completed in %.3fs candidates=%d scorer_load=%.3fs",
                        time.perf_counter() - score_started, len(candidates),
                        float(getattr(self, "_partiscore_load_seconds", 0.0)),
                    )
                score_by_id = dict(zip(scores.candidate_ids, scores.total.tolist(), strict=True))
                self._local_reward_records.update({
                    cid: {**record,
                          "reward_sequence_source": self._local_sequence_sources.get(cid, "design"),
                          "reward_structure_source": (
                              "packed" if self._local_sequence_sources.get(cid, "design") != "design" else "design"
                          )}
                    for cid, record in scores.records.items()
                })
                for candidate in candidates:
                    if candidate.candidate_id in scores.failures:
                        self._local_result_rows.append(self._failed_candidate_row(
                            candidate, stage="reward", message=scores.failures[candidate.candidate_id]
                        ))
                candidates = [candidate for candidate in candidates if candidate.candidate_id in score_by_id]
            except Exception as error:
                local_error = f"rank {rank}: {type(error).__name__}: {error}"
            errors = _all_gather_objects(local_error)
            if any(error is not None for error in errors):
                raise RuntimeError("PartiScore reward failed: " + "; ".join(error for error in errors if error is not None))
        else:
            configured_chunk = self.search_config.get("max_batch_size")
            chunk_size = len(candidates) or 1
            if configured_chunk is not None:
                chunk_size = int(configured_chunk)
            for start in range(0, len(candidates), chunk_size):
                chunk = candidates[start : start + chunk_size]
                scores = self.reward.score_batch(chunk, device="cpu")
                score_by_id.update(
                    (candidate_id, float(score))
                    for candidate_id, score in zip(
                        scores.candidate_ids, scores.total.tolist()
                    )
                )

        selection_started = time.perf_counter()
        for candidate in candidates:
            group = (
                candidate.base_request.task.name,
                tuple(candidate.actual_joint_length),
            )
            if group not in slots_by_group:
                raise RuntimeError(f"search candidate has no requested group {group}")
        summaries: tuple[CandidateSummary, ...] = tuple(
            (
                candidate.candidate_id,
                candidate.candidate_ordinal,
                candidate.base_request.task.name,
                tuple(candidate.actual_joint_length),
                score_by_id[candidate.candidate_id],
                rank,
                candidate.base_request.sample_id,
            )
            for candidate in candidates
        )
        gathered_summaries = _all_gather_objects(summaries)

        selection_payload: tuple[
            tuple[tuple[str, int, int], ...], tuple[FailureWire, ...]
        ] | None = None
        selection_status: tuple[
            str,
            tuple[
                tuple[tuple[str, int, int], ...], tuple[FailureWire, ...]
            ] | None,
            str,
        ] | None = None
        root_selection_error: Exception | None = None
        if rank == 0:
            try:
                by_group: dict[GroupKey, list[CandidateSummary]] = {}
                candidate_ids: set[str] = set()
                for rank_summaries in gathered_summaries:
                    for summary in rank_summaries:
                        if summary[0] in candidate_ids:
                            raise RuntimeError(
                                f"duplicate global search candidate ID {summary[0]}"
                            )
                        candidate_ids.add(summary[0])
                        group = (summary[2], tuple(summary[3]))
                        by_group.setdefault(group, []).append(summary)
                unexpected = set(by_group).difference(slots_by_group)
                if unexpected:
                    raise RuntimeError(
                        "search summaries contain unexpected groups: "
                        f"{sorted(unexpected)}"
                    )
                assignments: list[tuple[str, int, int]] = []
                missing: list[FailureWire] = []
                for group, group_slots in sorted(slots_by_group.items()):
                    ranked = sorted(
                        by_group.get(group, ()), key=lambda item: (-item[4], item[1])
                    )
                    assignments.extend(
                        (summary[0], summary[5], global_rank)
                        for global_rank, summary in enumerate(ranked, start=1)
                    )
                    succeeded_samples = {summary[6] for summary in ranked}
                    for slot in group_slots:
                        if slot[1] not in succeeded_samples:
                            missing.append(
                                (
                                    slot[0],
                                    slot[1],
                                    "search",
                                    "no valid replica remained for this base sample",
                                )
                            )
                selection_payload = (tuple(assignments), tuple(missing))
                selection_status = ("ok", selection_payload, "")
            except Exception as error:
                root_selection_error = error
                selection_status = (
                    "fatal",
                    None,
                    f"{type(error).__name__}: {error}",
                )
        if world_size > 1:
            status_list = [selection_status]
            torch.distributed.broadcast_object_list(status_list, src=0)
            selection_status = status_list[0]
        if selection_status is None:
            raise RuntimeError("rank zero did not broadcast search selection status")
        if selection_status[0] == "fatal":
            if rank == 0 and root_selection_error is not None:
                raise root_selection_error
            raise RuntimeError(
                f"rank-zero search selection failed: {selection_status[2]}"
            )
        selection_payload = selection_status[1]
        if selection_payload is None:
            raise RuntimeError("rank zero did not broadcast search selection")
        assignments, missing = selection_payload
        if rank == 0:
            self._local_failures.extend(missing)
        successful_base_keys: set[ResultKey] = set()
        assigned_base_keys: set[ResultKey] = set()
        write_failure_by_base: dict[ResultKey, str] = {}
        for candidate_id, owner_rank, global_rank in assignments:
            if owner_rank != rank:
                continue
            candidate = self._local_candidates.get(candidate_id)
            if candidate is None:
                raise RuntimeError(
                    f"winner owner {rank} cannot find candidate {candidate_id!r}"
                )
            task_name = candidate.base_request.task.name
            sample_id = candidate.base_request.sample_id
            key = (task_name, sample_id)
            assigned_base_keys.add(key)
            fold = self._local_fold_results.get(candidate_id)
            ledger_entries = self._local_design_ledgers.get(candidate_id)
            try:
                output_atoms = candidate.atoms
                if self.deferred_fold_reward and self.reward_config["structure_source"] == "fold":
                    if fold is None:
                        raise RuntimeError("winner owner lacks its fold result")
                    output_atoms = self._prepare_fold_output(candidate, fold)
                design_chain_rows = _output_design_chain_rows(candidate, output_atoms)
                design_chains = json.dumps(
                    list(design_chain_rows),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                identity = _candidate_output_identity(
                    candidate,
                    global_rank=global_rank,
                    output_format=self.output_format,
                    sequence_override=(None if fold is None else fold.design_sequence),
                    ledger_entries=ledger_entries,
                    joint_chain_ids=(
                        [entry["chain"] for entry in design_chain_rows]
                        if _is_joint_request(candidate.base_request)
                        else None
                    ),
                )
                (
                    filename,
                    base_id,
                    modality,
                    chain_id,
                    token_length,
                    sequence,
                    contig,
                    tip,
                ) = identity
                output_path = self.output_dir / task_name / filename
                constraint_fields = _constraint_output_fields(
                    candidate, output_atoms, output_format=self.output_format,
                    canonical_ligand_names=self.reward_method == "esmfold2",
                )
                actual_sequence_source = (fold.sequence_source if fold is not None else
                                          self._local_reward_records.get(candidate_id, {}).get("reward_sequence_source", "design"))
                if candidate.base_request.task.format == "antibody" and fold is not None and self.reward_config["structure_source"] != "fold":
                    # The ranking refold may use MPNN, but the emitted FM CIF
                    # still carries its original generated sequence.
                    actual_sequence_source = "design"
                actual_structure_source = ("fold" if fold is not None and self.reward_config["structure_source"] == "fold" else
                                           "packed" if fold is None and actual_sequence_source != "design" else "design")
                antibody_fields = _write_candidate_output(
                    candidate, output_atoms, output_path, entry_id=f"{task_name}_{sample_id}", fold=fold,
                    sequence_source=actual_sequence_source, structure_source=actual_structure_source,
                    reward_score=score_by_id[candidate_id])
            except (OSError, RuntimeError, StructureWriteError, ValueError) as exc:
                write_failure_by_base.setdefault(key, str(exc))
                self._local_result_rows.append(
                    self._failed_candidate_row(
                        candidate,
                        stage="write",
                        message=str(exc),
                        global_rank=global_rank,
                    )
                )
            else:
                self._capture_parameterization(
                    candidate,
                    (task_name, filename),
                    atoms=output_atoms,
                )
                successful_base_keys.add(key)
                metrics = (
                    ({"clash_score": score_by_id[candidate_id]}
                     if self.reward_method == "clash" else {})
                    if fold is None
                    else dict(fold.metrics)
                )
                reward_record = self._local_reward_records.get(candidate_id, {})
                if reward_record.get("reward_score") is not None:
                    # The score that selected this candidate is distinct from
                    # optional final-file/refined scoring performed afterward.
                    metrics["partiscore_score"] = reward_record["reward_score"]
                self._local_result_rows.append(
                    (
                        task_name,
                        filename, base_id, modality, chain_id, token_length,
                        self.reward_method, str(global_rank),
                        sequence, contig, tip, "ok", "", "",
                        json.dumps(metrics, sort_keys=True, allow_nan=False),
                        _ledger_json(ledger_entries or _candidate_fallback_ledger(candidate)),
                        json.dumps({**_candidate_generation_metadata(candidate),
                                    **self._local_reward_records.get(candidate_id, {}),
                                    **constraint_fields, **antibody_fields},
                                   allow_nan=False),
                        design_chains,
                    )
                )
        for key in sorted(successful_base_keys):
            self._local_succeeded.append(key)
        for key in sorted(assigned_base_keys - successful_base_keys):
            self._local_failures.append(
                (key[0], key[1], "write", write_failure_by_base[key])
            )
        if timing_logger is not None:
            timing_logger.info(
                "PartiScore ranking and output writing completed in %.3fs winners=%d",
                time.perf_counter() - selection_started, len(assignments),
            )

    def global_accounting(
        self,
    ) -> tuple[tuple[ResultKey, ...], tuple[FailureWire, ...]]:
        if self._global_succeeded is None or self._global_failures is None:
            raise RuntimeError("predict accounting was requested before epoch end")
        return self._global_succeeded, self._global_failures

    def global_batch_timings(self) -> tuple[tuple[int, int, float], ...]:
        return self._global_batch_seconds

    def global_postprocess_preparation(
        self,
    ) -> tuple[
        tuple[SampleParameterizationPayload, ...],
        tuple[PostprocessFailureWire, ...],
    ]:
        if not self.postprocess_enabled:
            return (), ()
        self._collect_parameterization_payloads()
        return (
            self._global_parameterization_payloads,
            self._global_postprocess_failures,
        )

    def global_postprocess_keys(self) -> tuple[ResultKey, ...]:
        """Return successful enabled-task output identities in result order."""

        if self._global_result_rows is None:
            raise RuntimeError("postprocess keys were requested before accounting")
        task_order = {name: index for index, name in enumerate(self._task_names)}
        rows = [
            row
            for row in self._global_result_rows
            if len(row) >= 12
            and row[11] == "ok"
            and row[1]
            and self._task_postprocess_enabled(str(row[0]))
        ]
        rows.sort(
            key=lambda row: (
                task_order.get(str(row[0]), 10**9),
                int(row[2]),
                int(str(row[7])) if str(row[7]).isdigit() else 10**9,
                str(row[1]),
            )
        )
        keys = tuple((str(row[0]), str(row[1])) for row in rows)
        if len(keys) != len(set(keys)):
            raise RuntimeError("successful postprocess output identities are duplicated")
        return keys

    def _ensure_postprocess_csv_flush_state(self) -> None:
        # A few focused tests construct this module without running __init__.
        if "_postprocess_pending_tasks" not in self.__dict__:
            self._postprocess_pending_tasks = set()
        if "_postprocess_pending_updates" not in self.__dict__:
            self._postprocess_pending_updates = 0
        if "_postprocess_last_flush_monotonic" not in self.__dict__:
            self._postprocess_last_flush_monotonic = time.monotonic()

    def flush_postprocess_status(self, *, force: bool = False) -> bool:
        """Durably write pending postprocess statuses for affected tasks.

        Driver callbacks can arrive much faster than a large result.csv can be
        rewritten.  Flush after a bounded number of outcomes or elapsed time;
        callers use ``force=True`` at lifecycle boundaries and on errors.
        """

        self._ensure_postprocess_csv_flush_state()
        if not self._postprocess_pending_tasks:
            if force:
                self._postprocess_pending_updates = 0
                self._postprocess_last_flush_monotonic = time.monotonic()
            return False
        if not self._global_result_rows:
            return False
        elapsed = time.monotonic() - self._postprocess_last_flush_monotonic
        if not force and (
            self._postprocess_pending_updates < _POSTPROCESS_CSV_BATCH_SIZE
            and elapsed < _POSTPROCESS_CSV_FLUSH_SECONDS
        ):
            return False
        affected_tasks = set(self._postprocess_pending_tasks)
        # Keep pending state until the write succeeds so exception cleanup can
        # retry the durable publication.
        self._write_result_csv(affected_tasks=affected_tasks)
        self._postprocess_pending_tasks.clear()
        self._postprocess_pending_updates = 0
        self._postprocess_last_flush_monotonic = time.monotonic()
        return True

    def set_postprocess_status(
        self,
        statuses: Mapping[ResultKey, str],
        messages: Mapping[ResultKey, str] | None = None,
    ) -> None:
        """Attach final postprocess status to result.csv rows on rank zero."""

        normalized = {
            (str(key[0]), str(key[1])): str(value)
            for key, value in statuses.items()
        }
        unknown = set(normalized).difference(self.global_postprocess_keys())
        if unknown:
            raise RuntimeError(
                "postprocess status contains unknown output identities: "
                f"{sorted(unknown)}"
            )
        self._global_postprocess_status = {}
        self._global_postprocess_messages = {}
        self.update_postprocess_status(
            normalized,
            messages,
            write=False,
        )
        if self._global_result_rows:
            self._write_result_csv()
        self._ensure_postprocess_csv_flush_state()
        self._postprocess_pending_tasks.clear()
        self._postprocess_pending_updates = 0
        self._postprocess_last_flush_monotonic = time.monotonic()

    def update_postprocess_status(
        self,
        statuses: Mapping[ResultKey, str],
        messages: Mapping[ResultKey, str] | None = None,
        *,
        write: bool = True,
    ) -> None:
        """Merge postprocess outcomes and durably publish affected task rows.

        Postprocessing results arrive one completed job at a time.  Replacing
        the status mapping on every callback would make earlier rows appear
        pending again, so this method deliberately updates only the supplied
        identities and rewrites only their task CSVs.
        """

        self._ensure_postprocess_csv_flush_state()
        normalized = {
            (str(key[0]), str(key[1])): str(value)
            for key, value in statuses.items()
        }
        known = set(self.global_postprocess_keys())
        unknown = set(normalized).difference(known)
        if unknown:
            raise RuntimeError(
                "postprocess status contains unknown output identities: "
                f"{sorted(unknown)}"
            )
        normalized_messages = {
            (str(key[0]), str(key[1])): str(value)
            for key, value in (messages or {}).items()
            if value
        }
        unknown_messages = set(normalized_messages).difference(normalized)
        if unknown_messages:
            raise RuntimeError(
                "postprocess message contains identities without status: "
                f"{sorted(unknown_messages)}"
            )
        self._global_postprocess_status.update(normalized)
        for key in normalized:
            message = normalized_messages.get(key)
            if message:
                self._global_postprocess_messages[key] = message
            else:
                self._global_postprocess_messages.pop(key, None)

        successful = {
            key for key, status in normalized.items() if status == "succeeded"
        }
        if successful and self._global_result_rows:
            rows = []
            for row in self._global_result_rows:
                key = (str(row[0]), str(row[1]))
                metadata = json.loads(row[16]) if len(row) >= 17 else {}
                if key in successful and "_antibody_result" in metadata:
                    from ..antibody.results import refresh_antibody_result
                    atoms = read_structure(self.output_dir / row[0] / row[1])
                    result = refresh_antibody_result(
                        json.loads(metadata["_antibody_result"]),
                        atoms,
                        json.loads(metadata["_antibody_context"]),
                    )
                    metadata["_antibody_result"] = json.dumps(
                        result,
                        allow_nan=False,
                    )
                    row = (*row[:16], json.dumps(metadata, allow_nan=False), *row[17:])
                rows.append(row)
            self._global_result_rows = tuple(rows)
        if normalized:
            self._postprocess_pending_tasks.update(key[0] for key in normalized)
            self._postprocess_pending_updates += len(normalized)
        if write:
            self.flush_postprocess_status(force=True)


class _PredictProgressLogCallback(Callback):
    """Mirror Lightning's terminal progress in the ordinary run log."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._total_batches = 0
        self._report_every = 1

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if not trainer.is_global_zero:
            return
        counts = trainer.num_predict_batches
        self._total_batches = (
            sum(int(value) for value in counts)
            if isinstance(counts, (list, tuple))
            else int(counts)
        )
        self._report_every = max(1, math.ceil(self._total_batches / 20))
        self._logger.info(
            "predict progress: 0/%d rank-local batches", self._total_batches
        )

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del pl_module, outputs, batch, dataloader_idx
        if not trainer.is_global_zero or self._total_batches <= 0:
            return
        completed = batch_idx + 1
        if (
            completed == self._total_batches
            or completed % self._report_every == 0
        ):
            self._logger.info(
                "predict progress: %d/%d rank-local batches",
                completed,
                self._total_batches,
            )


def _prepare_output_dir(
    output_dir: Path,
    resolved_config: Any,
    *,
    task_names: Sequence[str],
) -> None:
    """Prepare one isolated output leaf for every selected task."""

    if output_dir.exists():
        if not output_dir.is_dir():
            raise DesignConfigError(f"output_dir is not a directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True)

    selected_task_names = frozenset(task_names)
    for child in output_dir.iterdir():
        if child.is_symlink():
            raise DesignConfigError(
                "output_dir may contain only real task output directories; "
                f"symlinks are unsupported: {child}"
            )
        if child.name in selected_task_names:
            continue
        logs_dir = child / "logs"
        required_files = (
            child / "resolved_config.yaml",
            child / "result.csv",
            logs_dir / "design.log",
        )
        completed_task_leaf = (
            child.is_dir()
            and not logs_dir.is_symlink()
            and logs_dir.is_dir()
            and all(
                path.is_file() and not path.is_symlink()
                for path in required_files
            )
        )
        if not completed_task_leaf:
            raise DesignConfigError(
                "output_dir may contain only completed task output directories; "
                f"unexpected root entry: {child}"
            )

    for task_name in task_names:
        task_dir = output_dir / task_name
        if task_dir.exists():
            if not task_dir.is_dir():
                raise DesignConfigError(
                    f"task output path is not a directory: {task_dir}"
                )
            if any(task_dir.iterdir()):
                raise DesignConfigError(
                    "task output directory must be new or empty "
                    f"(resume is unsupported): {task_dir}"
                )

    for task_name in task_names:
        task_dir = output_dir / task_name
        task_dir.mkdir(exist_ok=True)
        (task_dir / "logs").mkdir()
        temporary = task_dir / ".resolved_config.yaml.tmp"
        temporary.write_text(
            OmegaConf.to_yaml(resolved_config, resolve=True), encoding="utf-8"
        )
        os.replace(temporary, task_dir / "resolved_config.yaml")


class _TaskLogFilter(logging.Filter):
    """Keep sample-scoped messages in their owning task leaf only."""

    def __init__(self, task_name: str) -> None:
        super().__init__()
        self._task_name = task_name

    def filter(self, record: logging.LogRecord) -> bool:
        record_task = getattr(record, "task_name", None)
        return record_task is None or record_task == self._task_name


def _log_task(
    logger: logging.Logger,
    task_name: str,
    level: int,
    message: str,
    *args: Any,
) -> None:
    logger.log(level, message, *args, extra={"task_name": task_name})


def _runtime_logger(
    output_dir: Path,
    *,
    task_names: Sequence[str],
) -> logging.Logger:
    logger = logging.getLogger(f"src.design.core.runtime.{id(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for task_name in task_names:
        handler = logging.FileHandler(
            output_dir / task_name / "logs" / "design.log", encoding="utf-8"
        )
        handler.setFormatter(formatter)
        handler.addFilter(_TaskLogFilter(task_name))
        logger.addHandler(handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in tuple(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


_OUTPUT_READY_TIMEOUT_SECONDS = 90.0
_OUTPUT_READY_POLL_SECONDS = 0.05
_OUTPUT_LAUNCH_ID_ENV = "PARTIDESIGN_OUTPUT_LAUNCH_ID"


def _launcher_world_size() -> int:
    raw = os.environ.get("WORLD_SIZE")
    if raw is None:
        return 1
    try:
        world_size = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid launcher WORLD_SIZE={raw!r}") from exc
    if world_size < 1:
        raise RuntimeError(f"invalid launcher WORLD_SIZE={raw!r}")
    return world_size


def _output_launch_identity() -> str:
    inherited = os.environ.get(_OUTPUT_LAUNCH_ID_ENV)
    if inherited:
        return inherited
    run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    if run_id:
        identity = f"torchelastic:{run_id}"
    else:
        address = os.environ.get("MASTER_ADDR")
        port = os.environ.get("MASTER_PORT")
        world_size = os.environ.get("WORLD_SIZE")
        if address and port and world_size:
            identity = f"torchrun:{address}:{port}:{world_size}"
        else:
            # Lightning's subprocess launcher is created later by Trainer, so
            # the original rank zero has no WORLD_SIZE yet.  Its children
            # inherit this private token and therefore wait on the same marker.
            identity = f"parent:{os.getpid()}:{time.time_ns()}"
    os.environ[_OUTPUT_LAUNCH_ID_ENV] = identity
    return identity


def _output_status_payload(
    output_dir: Path,
    *,
    state: str,
    message: str = "",
) -> str:
    if state not in {"ready", "failed"}:
        raise ValueError(f"invalid output-preparation state {state!r}")
    return json.dumps(
        {
            "launch_identity": _output_launch_identity(),
            "output_dir": str(output_dir.expanduser().resolve()),
            "state": state,
            "message": message,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _output_ready_marker(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    root = resolved.parent
    while not root.is_dir():
        parent = root.parent
        if parent == root:
            raise RuntimeError(
                f"output path has no existing coordination ancestor: {resolved}"
            )
        root = parent
    # Rendezvous uses the launch name and output path directly, not a digest.
    from urllib.parse import quote
    launch = quote(_output_launch_identity(), safe="")
    directory = root / ".partidesign-output-status"
    for offset in range(0, len(launch), 128):
        directory /= "launch_" + launch[offset:offset + 128]
    directory = directory / "outputs" / resolved.relative_to(root)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "status.json"


def _publish_output_status(
    output_dir: Path,
    *,
    state: str,
    message: str = "",
) -> Path:
    marker = _output_ready_marker(output_dir)
    temporary = marker.with_name(f"{marker.name}.tmp.{os.getpid()}")
    temporary.write_text(
        _output_status_payload(output_dir, state=state, message=message),
        encoding="utf-8",
    )
    os.replace(temporary, marker)
    return marker


def _publish_output_ready(output_dir: Path) -> Path:
    return _publish_output_status(output_dir, state="ready")


def _publish_output_failed(output_dir: Path, error: BaseException) -> Path:
    return _publish_output_status(
        output_dir,
        state="failed",
        message=f"{type(error).__name__}: {error}",
    )


def _output_ack_path(marker: Path, rank: int) -> Path:
    return marker.with_name(f"{marker.name}.ack.{rank}")


def _acknowledge_output_status(marker: Path) -> None:
    rank = _launcher_local_rank()
    acknowledgement = _output_ack_path(marker, rank)
    temporary = acknowledgement.with_name(
        f"{acknowledgement.name}.tmp.{os.getpid()}"
    )
    temporary.write_text(_output_launch_identity(), encoding="utf-8")
    os.replace(temporary, acknowledgement)


def _wait_for_output_status_acks(
    marker: Path,
    *,
    peer_count: int,
    timeout_seconds: float = 5.0,
) -> None:
    if peer_count <= 0:
        return
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if len(tuple(marker.parent.glob(f"{marker.name}.ack.*"))) >= peer_count:
            return
        time.sleep(_OUTPUT_READY_POLL_SECONDS)


def _wait_for_output_ready(
    output_dir: Path,
    *,
    timeout_seconds: float = _OUTPUT_READY_TIMEOUT_SECONDS,
) -> Path:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("output-ready timeout must be finite and positive")
    marker = _output_ready_marker(output_dir)
    expected_identity = _output_launch_identity()
    expected_output = str(output_dir.expanduser().resolve())
    deadline = time.monotonic() + timeout_seconds
    saw_wrong_payload = False
    while time.monotonic() < deadline:
        if marker.is_file():
            try:
                observed = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                observed = None
            if not isinstance(observed, dict):
                saw_wrong_payload = True
            elif (
                observed.get("launch_identity") != expected_identity
                or observed.get("output_dir") != expected_output
            ):
                saw_wrong_payload = True
            elif observed.get("state") == "failed":
                _acknowledge_output_status(marker)
                message = observed.get("message")
                detail = message if isinstance(message, str) else "unknown failure"
                raise RuntimeError(
                    f"rank-zero output preparation failed: {detail}"
                )
            elif observed.get("state") == "ready" and output_dir.is_dir():
                _acknowledge_output_status(marker)
                return marker
            elif observed.get("state") not in {"ready", "failed"}:
                saw_wrong_payload = True
        time.sleep(_OUTPUT_READY_POLL_SECONDS)
    detail = (
        " (a marker with the wrong launch identity was ignored)"
        if saw_wrong_payload
        else ""
    )
    raise RuntimeError(
        "timed out waiting for rank-zero output preparation"
        f" after {timeout_seconds:.3f}s{detail}: {output_dir}"
    )


def _remove_output_ready(marker: Path | None) -> None:
    if marker is not None:
        marker.unlink(missing_ok=True)
        for path in marker.parent.glob(f"{marker.name}.ack.*"):
            path.unlink(missing_ok=True)
        for path in marker.parent.glob(f"{marker.name}.tmp.*"):
            path.unlink(missing_ok=True)
        directory = marker.parent
        while directory.name != ".partidesign-output-status":
            try:
                directory.rmdir()
            except OSError:
                break
            directory = directory.parent
        if directory.name == ".partidesign-output-status":
            try:
                directory.rmdir()
            except OSError:
                pass


def _launcher_local_rank() -> int:
    raw = os.environ.get("RANK", os.environ.get("LOCAL_RANK"))
    if raw is None:
        return 0
    try:
        rank = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid launcher rank={raw!r}") from exc
    if rank < 0:
        raise RuntimeError(f"invalid launcher rank={raw!r}")
    return rank


def _default_trainer(*, precision: str) -> Trainer:
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    return Trainer(
        accelerator=accelerator,
        devices="auto" if accelerator == "gpu" else 1,
        strategy="auto",
        precision=precision if accelerator == "gpu" else "32-true",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        # Lightning owns the terminal TQDM bar: it correctly follows predict
        # batches and stays rank-zero-only under the built-in DDP strategy.
        enable_progress_bar=True,
        inference_mode=True,
        use_distributed_sampler=False,
    )


def _trusted_ccd_asset_path() -> Path:
    """Return the one release-owned private chemistry asset path."""

    return (
        community_asset("esmfold2", Path(__file__).resolve().parents[3])
        / "common"
        / "ccd.pkl"
    )


def _trusted_ccd_loader_process(
    connection: Any,
    path_text: str,
    requests: tuple[ExpandedRequest, ...],
    ccd_path: Path | None,
) -> None:
    """Isolated one-shot decoder for the large trusted CCD pickle."""

    try:
        component_ids: set[str] = set()
        for request in requests:
            source = None
            ligand_condition = request.task.condition.get("ligand", {})
            for chain in request.task.design_chains:
                if chain.modality != "ligand":
                    continue
                entry = (
                    ligand_condition.get(chain.chain_id)
                    if isinstance(ligand_condition, Mapping)
                    else None
                )
                if not isinstance(entry, Mapping) or entry.get("mode") not in {"type_bond", "ccd"}:
                    continue
                if entry.get("mode") == "ccd":
                    component_id = entry.get("ccd")
                    if (set(entry) != {"mode", "ccd"} or not isinstance(component_id, str)
                            or not component_id or not component_id.isascii()
                            or not component_id.isalnum() or component_id != component_id.upper()):
                        raise DesignSampleError("CCD ligand requires exactly mode and an uppercase alphanumeric ccd ID")
                    component_ids.add(component_id)
                    continue
                if source is None:
                    source = read_structure(request.task.input_path, ccd_path=ccd_path)
                rows = np.flatnonzero(
                    (np.asarray(source.chain_id).astype(str) == chain.chain_id)
                    & (np.asarray(source.mol_type).astype(str) == "ligand")
                )
                names = {
                    str(source.res_name[int(row)]).strip().upper() for row in rows
                }
                if len(rows) == 0 or len(names) != 1:
                    raise DesignSampleError(
                        f"type_bond preflight lacks one named ligand component for chain "
                        f"{chain.chain_id!r}"
                    )
                component_ids.update(names)
        if not component_ids:
            raise DesignSampleError("type_bond preflight lacks named ligand components")
        records = load_trusted_ccd_components(Path(path_text), component_ids)
        for request in requests:
            compile_request(
                request,
                ccd_path=ccd_path,
                trusted_ccd_components=records,
            )
        connection.send(
            (True, {key: value.rdkit_binary for key, value in records.items()})
        )
    except Exception as error:
        connection.send((False, f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _load_trusted_ccd_components_isolated(
    path: Path,
    requests: Sequence[ExpandedRequest],
    ccd_path: Path | None,
) -> Mapping[str, TrustedCCDComponent]:
    """Decode the 417MB snapshot once without retaining its heap in runtime."""

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_trusted_ccd_loader_process,
        args=(child, str(path), tuple(requests), ccd_path),
        daemon=False,
    )
    process.start()
    child.close()
    success = False
    payload: Any = "trusted CCD isolated preflight produced no result"
    try:
        if not parent.poll(120.0):
            process.terminate()
            raise ValueError("trusted CCD isolated preflight timed out")
        try:
            success, payload = parent.recv()
        except EOFError:
            payload = "trusted CCD isolated preflight closed its protocol unexpectedly"
    finally:
        parent.close()
        process.join(timeout=10.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        raise ValueError("trusted CCD isolated preflight did not exit")
    if process.exitcode != 0:
        raise ValueError(
            f"trusted CCD isolated preflight exited with code {process.exitcode}"
        )
    if not success or not isinstance(payload, dict):
        raise ValueError(f"trusted CCD isolated preflight failed: {payload}")
    return {
        key: TrustedCCDComponent(key, value) for key, value in payload.items()
    }


def _uses_type_bond(task: TaskSpec) -> bool:
    condition = task.condition
    ligand = condition.get("ligand", {}) if isinstance(condition, Mapping) else {}
    if not isinstance(ligand, Mapping):
        return False
    return any(
        chain.modality == "ligand"
        and isinstance(ligand.get(chain.chain_id), Mapping)
        and ligand[chain.chain_id].get("mode") in {"type_bond", "ccd"}
        for chain in task.design_chains
    )


def _preflight_trusted_ccd_components(
    requests: Sequence[ExpandedRequest],
    *,
    ccd_path: Path | None,
) -> Mapping[str, TrustedCCDComponent]:
    """Resolve and validate hard-ligand chemistry once in the parent process."""

    probe_requests: list[ExpandedRequest] = []
    seen_tasks: set[str] = set()
    for request in requests:
        if request.task.name in seen_tasks or not _uses_type_bond(request.task):
            continue
        seen_tasks.add(request.task.name)
        probe_requests.append(request)
    if not probe_requests:
        return {}
    records = _load_trusted_ccd_components_isolated(
        _trusted_ccd_asset_path(), probe_requests, ccd_path
    )
    return records


def _compile_preflight_request(
    request: ExpandedRequest, *,
    compiled_requests: Mapping[PreflightKey, CompiledSample | Exception] | None = None,
    **kwargs: Any,
) -> CompiledSample:
    if compiled_requests is None:
        return _compile_runtime_request(request, **kwargs)
    result = compiled_requests[preflight_key(request)]
    if isinstance(result, Exception):
        raise result
    return result


def _sequence_route_candidate(compiled: CompiledSample) -> SearchCandidate:
    """Materialize the guaranteed post-decode identity used by auto routing.

    Flexible design-protein identities are ``UNK`` in the compiler IR, while
    the decoder is restricted to the 20 standard amino acids.  A standard
    placeholder therefore makes preflight eligibility match every possible
    decoded identity without changing hard modified identities or context.
    """

    atoms = compiled.atoms.copy()
    fixed_modified_residue_uids = {
        int(value) for value in compiled.modified_polymer_carriers
    }
    for token_uid in np.unique(np.asarray(atoms.token_uid, dtype=np.int64)):
        rows = np.flatnonzero(
            np.asarray(atoms.token_uid, dtype=np.int64) == int(token_uid)
        )
        if set(np.asarray(atoms.mol_type[rows]).astype(str)) != {"protein"}:
            continue
        if not bool(np.asarray(atoms.is_design[rows], dtype=bool).all()):
            continue
        if int(atoms.residue_uid[rows[0]]) in fixed_modified_residue_uids:
            continue
        identity = np.asarray(atoms.identity_condition[rows], dtype=bool)
        if bool(identity.any()) != bool(identity.all()):
            raise DesignSampleError(
                "compiled design-protein token has partial identity conditioning"
            )
        if not bool(identity.any()):
            atoms.res_name[rows] = "ALA"
    request = compiled.request
    return SearchCandidate(
        candidate_id=(
            f"{request.task.name}/{request.sample_id}/sequence_route_preflight"
        ),
        candidate_ordinal=request.global_ordinal,
        base_request=request,
        actual_joint_length=actual_joint_length(compiled),
        atoms=atoms,
        topology_conditions=compiled.topology_conditions,
        model_parent_residue=dict(compiled.model_parent_residue),
        antibody_metadata=dict(compiled.antibody_metadata),
        fixed_modified_residue_names=_fixed_modified_residue_names(compiled),
        fixed_modified_parent_residue=_fixed_modified_parent_residue(compiled),
        validated_ligand_identities=dict(compiled.validated_ligand_identities),
    )


def _validate_af2_sequence_routes(
    requests: Sequence[ExpandedRequest], reward_config: Mapping[str, Any],
) -> None:
    """Reject unsupported NA routes from task declarations, without compilation."""
    if reward_config.get("_target_") != "src.design.rewards.af2.AF2Reward":
        return
    source = reward_config.get("sequence", {}).get("source")
    if source == "na_mpnn":
        raise DesignConfigError("NA-MPNN sequence source requires ESMFold2 or AF3")
    if source == "auto" and any(
        chain.modality in {"rna", "dna"}
        for request in unique_preflight_requests(requests)
        for chain in request.task.design_chains
    ):
        raise DesignConfigError(
            "AF2 reward auto sequence routing cannot use NA-MPNN; "
            "use ESMFold2, AF3, or sequence.source=design"
        )


def _preflight_partiscore_packing_assets(
    requests: Sequence[ExpandedRequest], reward_config: Mapping[str, Any], *,
    ccd_path: Path | None, trusted_ccd_components: Mapping[str, TrustedCCDComponent],
    global_seed: int,
    compiled_requests: Mapping[PreflightKey, CompiledSample | Exception] | None = None,
) -> None:
    if reward_config.get("_target_") != "src.design.rewards.partiscore.PartiScoreReward":
        return
    source = reward_config.get("sequence", {}).get("source", "design")
    if source == "design":
        return
    from ..rewards.sequence import resolve_reward_sequence_source
    required = source != "auto"
    if source == "auto":
        for request in (unique_preflight_requests(requests) if compiled_requests is not None else requests):
            try:
                compiled = _compile_preflight_request(
                    request, compiled_requests=compiled_requests, ccd_path=ccd_path, trusted_ccd_components=trusted_ccd_components,
                    global_seed=global_seed,
                )
                route = resolve_reward_sequence_source(
                    _sequence_route_candidate(compiled), "auto", reward_method="partiscore",
                )
            except (DesignInputError, DesignSampleError, ValueError):
                continue
            if route != "design":
                required = True
                break
    if required:
        from ..rewards.packing import validate_packing_assets
        try:
            validate_packing_assets(Path(__file__).resolve().parents[3])
        except RuntimeError as error:
            raise DesignConfigError(f"PartiScore packing preflight failed: {error}") from error



def _call_checkpoint_function(function: Callable[..., Any], *args: Any,
                              route_options: Mapping[str, Any], **kwargs: Any) -> Any:
    """Pass routing options without changing existing injected loader callbacks."""
    parameters = inspect.signature(function).parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    for key, value in route_options.items():
        parameter = parameters.get(key)
        if accepts_kwargs or (parameter is not None and parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
        }):
            kwargs[key] = value
    return function(*args, **kwargs)


def run_runtime(
    loaded: LoadedDesignConfig,
    tasks: tuple[TaskSpec, ...],
    requests: tuple[ExpandedRequest, ...],
    *,
    terminal_candidate_factory: TerminalCandidateFactory = build_terminal_candidates,
    fk_candidate_factory: FKCandidateFactory = build_fk_terminal_candidates,
    model_loader: Callable[..., torch.nn.Module] = load_combined_checkpoint,
    checkpoint_mode_resolver: CheckpointModeResolver = (
        resolve_checkpoint_context_constraint_mode
    ),
    model_compiler: Callable[[Any, Mapping[str, Any]], Any] = compile_denoiser,
    trainer_factory: Callable[..., Trainer] = _default_trainer,
    fold_reward_runner: FoldRewardRunner = _run_deferred_fold_reward_batch,
    prepared_sources: Mapping[str, Any] | None = None,
) -> DesignRunResult:
    """Execute standalone inference with exact in-memory ID accounting."""

    runtime_started = time.perf_counter()
    config = loaded.data
    try:
        _validate_global_seed(config.seed)
    except (TypeError, ValueError) as error:
        raise DesignConfigError(f"seed is invalid: {error}") from error
    sampling_config = OmegaConf.to_container(config.sampling, resolve=True)
    if not isinstance(sampling_config, dict):
        raise DesignConfigError("sampling config must resolve to a mapping")
    validate_sampling_preflight(sampling_config)
    if config.search.algorithm not in {None, "best_of_n"}:
        raise DesignConfigError(
            "public inference search.algorithm must be null or best_of_n"
        )
    search_enabled = config.search.algorithm == "best_of_n"
    reward_config = OmegaConf.to_container(config.reward, resolve=True)
    if not isinstance(reward_config, dict):
        raise DesignConfigError("reward config must resolve to a mapping")
    reward_target = str(reward_config.get("_target_"))
    clash_enabled = search_enabled and reward_target == "src.design.rewards.clash.ClashReward"
    if clash_enabled:
        try:
            validate_vdw_elements(DESIGN_LIGAND_ELEMENTS)
        except ValueError as exc:
            raise DesignConfigError(
                f"clash reward generated element vocabulary is invalid: {exc}"
            ) from exc
    needs_scorer = any(task.score for task in tasks) or reward_target == "src.design.rewards.partiscore.PartiScoreReward" or (
        reward_target in {"src.design.rewards.esmfold2.ESMFold2Reward", "src.design.rewards.af3.AF3Reward"}
        and reward_config.get("score", {}).get("method", "confidence") == "partiscore"
    )
    scorer_config = OmegaConf.to_container(config.get("scorer", {}), resolve=True) if needs_scorer else {}
    checkpoint = config.model.checkpoint
    ccd_path = None if config.ccd_path is None else Path(str(config.ccd_path))
    automatic_route = bool(config.get("route", True)) and config.get("route_ckpt") is None
    routing_sources = dict(prepared_sources or {})
    if automatic_route:
        missing_sources = [task for task in tasks if task.name not in routing_sources]
        routing_sources.update(prepare_routing_sources(missing_sources, ccd_path=ccd_path))
    model_variant, route_mode, route_reason = resolve_model_route(
        tasks, enabled=config.get("route", True), override=config.get("route_ckpt"),
        prepared_sources=routing_sources, ccd_path=ccd_path,
    )
    route_preferences = (task_model_preferences(tasks, prepared_sources=routing_sources, ccd_path=ccd_path)
                         if automatic_route else ())
    route_options = {
        "variant": model_variant,
        "require_bundle": route_mode == "forced",
    }
    if checkpoint is None:
        raise DesignConfigError("model.checkpoint is required for generation")
    if config.output_dir is None:
        raise DesignConfigError("output_dir is required")
    output_dir = Path(str(config.output_dir))
    expected = tuple(_request_key(request) for request in requests)
    if not expected or len(set(expected)) != len(expected):
        raise DesignConfigError("expanded request IDs must be non-empty and unique")
    try:
        trusted_ccd_components = _preflight_trusted_ccd_components(
            requests,
            ccd_path=ccd_path,
        )
    except (DesignInputError, DesignSampleError, OSError, ValueError) as error:
        raise DesignConfigError(
            f"trusted ligand chemistry preflight failed: {error}"
        ) from error
    needs_interface_preflight = reward_target == "src.design.rewards.partiscore.PartiScoreReward" or (
        reward_target in {"src.design.rewards.esmfold2.ESMFold2Reward", "src.design.rewards.af3.AF3Reward"}
        and reward_config.get("score", {}).get("method", "confidence") == "partiscore"
    )
    sequence_source = reward_config.get("sequence", {}).get("source")
    if needs_interface_preflight:
        interfaces = source_interfaces(requests)
        for request in unique_preflight_requests(requests):
            if interfaces[preflight_key(request)] is False:
                raise DesignConfigError(
                    f"task {request.task.name!r} PartiScore reward requires "
                    "a design-touching source-partner interface"
                )
    # Structural preparation happens once in the real dataset. Actual scoring,
    # sequence backends and writers own geometry, asset and format failures.
    _validate_af2_sequence_routes(requests, reward_config)
    if sequence_source != "auto":
        _preflight_partiscore_packing_assets(
            (), reward_config, ccd_path=ccd_path,
            trusted_ccd_components=trusted_ccd_components, global_seed=int(config.seed),
        )
    postprocess_config = OmegaConf.to_container(config.postprocess, resolve=True)
    if not isinstance(postprocess_config, dict):
        raise DesignConfigError("postprocess config must resolve to a mapping")
    postprocess_enabled = any(
        bool(getattr(task, "postprocess", False)) and task.override for task in tasks
    )
    if postprocess_enabled:
        preflight_postprocessing(postprocess_config)
        from ..postproc.context import preload_context_chemistry
        preload_context_chemistry(tasks, ccd_path=ccd_path)
    if postprocess_enabled and reward_target == "src.design.rewards.af2.AF2Reward":
        raise DesignConfigError(
            "AF2 fold rewards cannot be combined with OpenMM postprocessing"
        )

    if any(task.format == "antibody" for task in tasks):
        from ..antibody.checkpoint import validate_antibody_checkpoint
        _call_checkpoint_function(
            validate_antibody_checkpoint, checkpoint, route_options=route_options,
            task_formats=tuple(task.format for task in tasks),
        )

    context_preflight_started = time.perf_counter()
    requested_context_mode = str(config.model.context_constraint_mode)
    try:
        resolved_context_mode = _call_checkpoint_function(
            checkpoint_mode_resolver, checkpoint, route_options=route_options,
            requested_mode=requested_context_mode,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise DesignConfigError(
            "model checkpoint context_constraint_mode preflight failed: "
            f"{error}"
        ) from error
    if resolved_context_mode not in {"rigid", "flex"}:
        raise DesignConfigError(
            "model checkpoint resolved an invalid context_constraint_mode: "
            f"{resolved_context_mode!r}"
        )
    config.model.context_constraint_mode = resolved_context_mode
    context_preflight_elapsed = time.perf_counter() - context_preflight_started
    execution_policy_started = time.perf_counter()
    execution_policy = resolve_execution_policy(
        OmegaConf.to_container(config.model, resolve=True), requests,
        OmegaConf.to_container(config.search, resolve=True),
    )
    apply_execution_policy(config, execution_policy)
    execution_backend = execution_policy.resolved_backend
    execution_policy_elapsed = time.perf_counter() - execution_policy_started

    launcher_rank = _launcher_local_rank()
    launcher_world_size = _launcher_world_size()
    output_ready_marker: Path | None = None
    logger: logging.Logger | None = None
    module: _StandalonePredictModule | None = None
    try:
        if launcher_rank == 0:
            task_names = tuple(task.name for task in tasks)
            try:
                output_init_started = time.perf_counter()
                _prepare_output_dir(
                    output_dir,
                    config,
                    task_names=task_names,
                )
                logger = _runtime_logger(
                    output_dir,
                    task_names=task_names,
                )
                logger.info("FM execution policy: %s", execution_policy.metadata())
                logger.info("FM execution policy startup completed in %.3fs", execution_policy_elapsed)
                logger.info("runtime validation/preflight completed in %.3fs", time.perf_counter() - runtime_started)
                logger.info("checkpoint context constraint preflight completed in %.3fs", context_preflight_elapsed)
                logger.info("output leaf initialization completed in %.3fs", time.perf_counter() - output_init_started)
            except Exception as error:
                if launcher_world_size > 1:
                    output_ready_marker = _publish_output_failed(output_dir, error)
                    _wait_for_output_status_acks(
                        output_ready_marker,
                        peer_count=launcher_world_size - 1,
                    )
                raise
            # Publish before a single-process Trainer is constructed: it may
            # subsequently launch Lightning DDP children that re-enter here.
            output_ready_marker = _publish_output_ready(output_dir)
        else:
            if launcher_world_size < 2:
                raise RuntimeError("nonzero launcher rank requires WORLD_SIZE >= 2")
            _wait_for_output_ready(output_dir)

        resolved_seed = int(config.seed)
        seed_everything(resolved_seed, workers=True)
        if logger is not None:
            logger.info(
                "checkpoint context_constraint_mode resolved %s -> %s",
                requested_context_mode,
                resolved_context_mode,
            )
            for task in tasks:
                task_request_count = sum(
                    request.task.name == task.name for request in requests
                )
                _log_task(logger, task.name, logging.INFO, "global seed=%d", resolved_seed)
                _log_task(
                    logger,
                    task.name,
                    logging.INFO,
                    "starting %d expanded requests for task %s",
                    task_request_count,
                    task.name,
                )
        load_started = time.perf_counter()
        model = _call_checkpoint_function(
            model_loader, str(checkpoint), route_options=route_options,
            context_constraint_mode=resolved_context_mode,
        )
        if logger is not None:
            logger.info(
                "checkpoint source=%s branch=%s route=%s reason=%s",
                getattr(model, "_checkpoint_source", checkpoint),
                getattr(model, "_checkpoint_member", None) or "standalone",
                route_mode if getattr(model, "_checkpoint_member", None) else "direct",
                route_reason if getattr(model, "_checkpoint_member", None) else "explicit standalone checkpoint",
            )
        model_context_mode = getattr(
            model,
            "context_constraint_mode",
            resolved_context_mode,
        )
        if model_context_mode != resolved_context_mode:
            raise RuntimeError(
                "loaded model context_constraint_mode differs from preflight: "
                f"{model_context_mode!r} != {resolved_context_mode!r}"
            )
        if logger is not None:
            logger.info("checkpoint model load completed in %.3fs", time.perf_counter() - load_started)
            actual_member = getattr(model, "_checkpoint_member", None) or "standalone"
            for preference in route_preferences:
                _log_task(logger, preference.task_name, logging.INFO,
                          "model routing preference=%s reason=%s; submission member=%s",
                          preference.member, preference.reason, actual_member)
        if hasattr(model, "configure_optimization"):
            optimization_started = time.perf_counter()
            from ..model.optimization import DEFAULT_OPTIMIZATION
            optimization = config.model.get("optimization", DEFAULT_OPTIMIZATION)
            decision = model.configure_optimization(dict(optimization))
            if logger is not None:
                logger.info("FM inference optimization: %s", decision)
                logger.info("FM optimization configuration completed in %.3fs", time.perf_counter() - optimization_started)
        compile_started = time.perf_counter()
        compile_config = OmegaConf.to_container(config.model.compile, resolve=True)
        token_buckets = active_token_buckets(compile_config)
        if execution_backend == "bioir":
            from ..model.bioir import configure_bioir, bioir_status
            model = configure_bioir(
                model, OmegaConf.to_container(config.model.bioir, resolve=True),
            )
            if logger is not None:
                logger.info("FM execution backend BioIR: %s", bioir_status(model))
        else:
            model = model_compiler(model, compile_config)
        if logger is not None:
            compiler_options = getattr(model, "_inference_compiler_options", None)
            if compiler_options:
                logger.info("FM compiler options: %s", compiler_options)
            logger.info("FM compile wrapper setup completed in %.3fs", time.perf_counter() - compile_started)
            logger.info(
                "checkpoint load and compile wrapper setup completed in %.3fs",
                time.perf_counter() - load_started,
            )
        search_config = OmegaConf.to_container(config.search, resolve=True)
        assert isinstance(search_config, dict) and isinstance(reward_config, dict)
        module = _StandalonePredictModule(
            model,
            sampling_config=sampling_config,
            ligand_graph_method=str(config.sampling.ligand_graph.method),
            output_dir=output_dir,
            output_format=str(config.output_format),
            terminal_candidate_factory=terminal_candidate_factory,
            fk_candidate_factory=fk_candidate_factory,
            search_config=search_config,
            reward_config=reward_config,
            requests=requests,
            scorer_config=scorer_config,
            postprocess_enabled=postprocess_enabled,
            fold_reward_runner=fold_reward_runner,
            global_seed=int(config.seed),
            token_buckets=token_buckets,
        )
        dataset = _CompiledRequestDataset(
            requests,
            ccd_path=ccd_path,
            trusted_ccd_components=trusted_ccd_components,
            global_seed=int(config.seed),
            clash_preflight=clash_enabled,
            prepared_sources=routing_sources,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=_ExactShardBatchSampler(
                len(dataset), int(config.batch_size)
            ),
            num_workers=int(config.num_workers),
            collate_fn=partial(_collate_prepared, token_buckets=token_buckets),
            worker_init_fn=pl_worker_init_function,
            persistent_workers=bool(int(config.num_workers) > 0),
            pin_memory=bool(config.pin_memory),
        )
        trainer = trainer_factory(precision=str(config.model.precision))
        if logger is not None:
            trainer.callbacks.append(_PredictProgressLogCallback(logger))
            logger.info("predict data loader and trainer initialization completed in %.3fs", time.perf_counter() - load_started)
        trainer.predict(
            module,
            dataloaders=loader,
            return_predictions=False,
        )
        if execution_backend == "bioir" and logger is not None:
            from ..model.bioir import bioir_status
            logger.info("FM BioIR execution statistics: %s", bioir_status(module.inference_model))
        if module.deferred_partiscore_reward:
            rank, world_size = _distributed_rank_world()
            offload_error = None
            offload_started = time.perf_counter()
            try:
                module.inference_model.to(torch.device("cpu"))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as error:
                offload_error = f"rank {rank}: {type(error).__name__}: {error}"
            errors = _all_gather_objects(offload_error)
            if world_size > 1:
                torch.distributed.barrier()
            if any(error is not None for error in errors):
                raise RuntimeError("model offload before PartiScore preparation failed: " + "; ".join(error for error in errors if error is not None))
            if logger is not None:
                logger.info("FM model offload before PartiScore completed in %.3fs", time.perf_counter() - offload_started)
            started = time.perf_counter()
            temporary_parent = Path(tempfile.mkdtemp(prefix=f"partidesign-packing-rank{rank}-"))
            try:
                module.finish_deferred_partiscore_reward(
                    repo_root=Path(__file__).resolve().parents[3],
                    work_root=temporary_parent / "owner-batch",
                    timing_logger=logger,
                )
            finally:
                shutil.rmtree(temporary_parent, ignore_errors=True)
            if logger is not None:
                logger.info("PartiScore sequence/packing/ranking completed in %.3fs", time.perf_counter() - started)
        if module.deferred_fold_reward:
            rank, world_size = _distributed_rank_world()
            offload_error: str | None = None
            try:
                module.inference_model.to(torch.device("cpu"))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as error:
                offload_error = f"rank {rank}: {type(error).__name__}: {error}"
            offload_errors = _all_gather_objects(offload_error)
            if world_size > 1:
                torch.distributed.barrier()
            real_offload_errors = [item for item in offload_errors if item is not None]
            if real_offload_errors:
                raise RuntimeError(
                    "model offload before ESMFold2 reward failed: "
                    + "; ".join(real_offload_errors)
                )
            candidate_counts = _all_gather_objects(
                module.local_deferred_fold_counts()[0]
            )
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in candidate_counts
            ):
                raise RuntimeError("fold reward candidate counts are invalid")
            global_candidate_count = sum(candidate_counts)
            if logger is not None:
                logger.info(
                    "fold reward progress: 0/%d candidates across %d owner batches",
                    global_candidate_count,
                    world_size,
                )
            temporary_parent = Path(
                tempfile.mkdtemp(prefix=f"partidesign-fold-rank{rank}-")
            )
            try:
                module.finish_deferred_fold_reward(
                    repo_root=Path(__file__).resolve().parents[3],
                    work_root=temporary_parent / "owner-batch",
                )
            finally:
                shutil.rmtree(temporary_parent, ignore_errors=True)
            result_counts = _all_gather_objects(
                module.local_deferred_fold_counts()[1]
            )
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in result_counts
            ):
                raise RuntimeError("fold reward result counts are invalid")
            if logger is not None:
                logger.info(
                    "fold reward progress: %d/%d candidates completed",
                    sum(result_counts),
                    global_candidate_count,
                )
        succeeded_wires, failure_wires = module.global_accounting()
        if logger is not None:
            timings = module.global_batch_timings()
            first = [value for _, index, value in timings if index == 0]
            if first:
                logger.info(
                    "first predict batch (compile warmup + sampling) %.3fs",
                    max(first),
                )
            steady = [value for _, index, value in timings if index > 0]
            if steady:
                logger.info(
                    "steady predict batches n=%d median=%.3fs range=[%.3f, %.3f]s",
                    len(steady),
                    float(np.median(np.asarray(steady, dtype=np.float64))),
                    min(steady),
                    max(steady),
                )
        succeeded: set[ResultKey] = set()
        failed: dict[ResultKey, SampleFailure] = {}
        for raw_key in succeeded_wires:
            key = (str(raw_key[0]), str(raw_key[1]))
            if key in succeeded or key in failed:
                raise RuntimeError(f"duplicate terminal accounting for {key}")
            succeeded.add(key)
        for wire in failure_wires:
            if not isinstance(wire, (tuple, list)) or len(wire) != 4:
                raise TypeError("predict outcome contains an invalid failure tuple")
            failure = SampleFailure(*(str(value) for value in wire))
            key = (failure.task_name, failure.sample_id)
            if key in succeeded or key in failed:
                raise RuntimeError(f"duplicate terminal accounting for {key}")
            failed[key] = failure
            if logger is not None:
                _log_task(
                    logger,
                    failure.task_name,
                    logging.ERROR,
                    "%s/%s stage=%s: %s",
                    failure.task_name,
                    failure.sample_id,
                    failure.stage,
                    failure.message,
                )
        expected_set = set(expected)
        observed = succeeded | set(failed)
        if observed != expected_set:
            raise RuntimeError(
                "terminal accounting differs from expected request IDs: "
                f"missing={sorted(expected_set-observed)}, "
                f"extra={sorted(observed-expected_set)}"
            )
        postprocessed = 0
        postprocess_failed = 0
        if postprocess_enabled:
            postprocess_started = time.perf_counter()
            rank, world_size = _distributed_rank_world()
            offload_error: str | None = None
            try:
                module.inference_model.to(torch.device("cpu"))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as error:
                offload_error = f"rank {rank}: {type(error).__name__}: {error}"
            offload_errors = _all_gather_objects(offload_error)
            if world_size > 1:
                torch.distributed.barrier()
            real_offload_errors = [item for item in offload_errors if item is not None]
            if real_offload_errors:
                raise RuntimeError(
                    "model offload before OpenMM postprocess failed: "
                    + "; ".join(real_offload_errors)
                )

            parameterization_payloads, preparation_failures = (
                module.global_postprocess_preparation()
            )
            payload_by_key: dict[ResultKey, SampleParameterizationPayload] = {}
            preparation_by_key: dict[ResultKey, str] = {}
            validation_error: Exception | None = None
            validation_status: tuple[str, str] | None = None
            if rank == 0:
                try:
                    for payload in parameterization_payloads:
                        if payload.key in payload_by_key:
                            raise RuntimeError(
                                "duplicate gathered parameterization payload "
                                f"{payload.key}"
                            )
                        payload_by_key[payload.key] = payload
                    for wire in preparation_failures:
                        if not isinstance(wire, (tuple, list)) or len(wire) != 3:
                            raise TypeError(
                                "postprocess preparation contains an invalid failure"
                            )
                        key = (str(wire[0]), str(wire[1]))
                        if key in preparation_by_key:
                            raise RuntimeError(
                                "duplicate gathered postprocess preparation failure "
                                f"{key}"
                            )
                        preparation_by_key[key] = str(wire[2])
                    if set(payload_by_key) & set(preparation_by_key):
                        raise RuntimeError(
                            "postprocess payload/failure accounting overlaps"
                        )
                    expected_postprocess_keys = set(module.global_postprocess_keys())
                    if set(payload_by_key) | set(preparation_by_key) != expected_postprocess_keys:
                        raise RuntimeError(
                            "postprocess preparation differs from successful enabled-task output IDs"
                        )
                    validation_status = ("ok", "")
                except Exception as error:
                    validation_error = error
                    validation_status = (
                        "fatal",
                        f"{type(error).__name__}: {error}",
                    )
            if world_size > 1:
                validation_list = [validation_status]
                torch.distributed.broadcast_object_list(validation_list, src=0)
                validation_status = validation_list[0]
            if validation_status is None:
                raise RuntimeError(
                    "rank zero did not publish postprocess payload validation"
                )
            if validation_status[0] == "fatal":
                if rank == 0 and validation_error is not None:
                    raise validation_error
                raise RuntimeError(
                    "rank-zero postprocess payload validation failed: "
                    f"{validation_status[1]}"
                )

            # All generation collectives, including private payload validation,
            # are complete.  OpenMM is a rank-zero driver operation and may run
            # for minutes, so release the generation process group before the
            # driver enters the job loop.  Save rank/world before destruction:
            # after destroy, _distributed_rank_world() necessarily reports (0, 1).
            generation_rank, generation_world_size = _release_generation_process_group()
            from ..postproc.context import clear_context_chemistry_cache
            clear_context_chemistry_cache()
            if generation_world_size > 1 and generation_rank != 0:
                return DesignRunResult(
                    output_dir,
                    len(succeeded),
                    len(failed),
                    0,
                    0,
                    0,
                    0,
                )

            if logger is not None:
                logger.info(
                    "postprocess payload preparation completed in %.3fs candidates=%d",
                    time.perf_counter() - postprocess_started,
                    len(payload_by_key),
                )
            expected_order = {
                key: index
                for index, key in enumerate(module.global_postprocess_keys())
            }
            ordered_payloads = sorted(
                payload_by_key.values(),
                key=lambda payload: expected_order[payload.key],
            )
            jobs = tuple(
                PostprocessJob(
                    task_name=payload.task_name,
                    sample_id=payload.sample_id,
                    base_cif=output_dir / payload.task_name / payload.sample_id,
                    refined_cif=(
                        output_dir
                        / payload.task_name
                        / Path(payload.sample_id).with_suffix(".refined.cif").name
                    ),
                    payload=payload,
                )
                for payload in ordered_payloads
            )
            tasks_by_name = {task.name: task for task in tasks}
            jobs_by_key = {job.key: job for job in jobs}
            finalized_jobs: dict[ResultKey, PostprocessJob] = {}
            completed_result_keys: set[ResultKey] = set()

            preparation_statuses = {
                key: "failed" for key in preparation_by_key
            }
            module.update_postprocess_status(
                preparation_statuses,
                preparation_by_key,
                write=False,
            )
            # Preparation failures are known before the job loop and must be
            # durable even when no OpenMM job is eventually submitted.
            module.flush_postprocess_status(force=True)

            def _record_postprocess_result(
                job: PostprocessJob,
                result: Any,
            ) -> None:
                key = result.key
                if key not in jobs_by_key:
                    raise RuntimeError(
                        "postprocess callback returned an unknown job identity: "
                        f"{key}"
                    )
                if key in completed_result_keys:
                    raise RuntimeError(
                        "postprocess callback returned a duplicate job identity: "
                        f"{key}"
                    )
                if job.key != key:
                    raise RuntimeError(
                        "postprocess callback job/result identities differ"
                    )
                finalized_jobs[key] = _finalize_postprocess_files(
                    (job,),
                    {key: result},
                    tasks_by_name,
                )[0]
                completed_result_keys.add(key)
                status = "succeeded" if result.succeeded else "failed"
                module.update_postprocess_status(
                    {key: status},
                    {} if result.succeeded else {
                        key: result.message or "postprocess failed"
                    },
                    write=False,
                )
                module.flush_postprocess_status()

            openmm_started = time.perf_counter()
            results = (
                run_postprocessing(
                    jobs,
                    config=postprocess_config,
                    on_result=_record_postprocess_result,
                )
                if jobs
                else ()
            )
            if logger is not None:
                logger.info(
                    "OpenMM postprocess execution completed in %.3fs jobs=%d",
                    time.perf_counter() - openmm_started, len(jobs),
                )
            result_by_key = {result.key: result for result in results}
            if (
                len(result_by_key) != len(results)
                or set(result_by_key) != set(payload_by_key)
                or completed_result_keys != set(payload_by_key)
            ):
                raise RuntimeError(
                    "postprocess result/callback IDs differ from submitted payload IDs"
                )
            finalized_jobs_for_scoring = tuple(
                finalized_jobs[job.key] for job in jobs
            )
            refined_scoring_started = time.perf_counter()
            module.score_refined_outputs(
                finalized_jobs_for_scoring,
                result_by_key,
            )
            if logger is not None:
                logger.info(
                    "refined-output scoring completed in %.3fs",
                    time.perf_counter() - refined_scoring_started,
                )
            failure_messages: list[PostprocessFailureWire] = [
                (key[0], key[1], message)
                for key, message in preparation_by_key.items()
            ]
            postprocessed = sum(
                int(result.succeeded) for result in result_by_key.values()
            )
            for key, result in result_by_key.items():
                if not result.succeeded:
                    failure_messages.append(
                        (key[0], key[1], result.message or "postprocess failed")
                    )
                elif (
                    logger is not None
                    and result.initial_energy_kj_mol is not None
                    and result.final_energy_kj_mol is not None
                ):
                    rms_force = result.rms_force_kj_mol_nm
                    rms_text = (
                        f"{rms_force:.8g}"
                        if rms_force is not None and math.isfinite(rms_force)
                        else "non-finite"
                    )
                    _log_task(
                        logger,
                        key[0],
                        logging.INFO,
                        "%s/%s stage=postprocess "
                        "initial_energy_kj_mol=%.8g "
                        "final_energy_kj_mol=%.8g "
                        "rms_force_kj_mol_nm=%s diagnostic_only=true",
                        key[0],
                        key[1],
                        result.initial_energy_kj_mol,
                        result.final_energy_kj_mol,
                        rms_text,
                    )
            postprocess_failed = len(failure_messages)
            expected_postprocess_count = len(module.global_postprocess_keys())
            if postprocessed + postprocess_failed != expected_postprocess_count:
                raise RuntimeError(
                    "postprocess accounting differs from successful enabled-task outputs"
                )
            if set(module._global_postprocess_status) != set(module.global_postprocess_keys()):
                raise RuntimeError(
                    "postprocess status accounting differs from successful enabled-task outputs"
                )
            module.flush_postprocess_status(force=True)
            if logger is not None:
                for task_name, sample_id, message in failure_messages:
                    _log_task(
                        logger,
                        task_name,
                        logging.ERROR,
                        "%s/%s stage=postprocess: %s",
                        task_name,
                        sample_id,
                        message,
                    )
                logger.info(
                    "postprocess completed succeeded=%d failed=%d total=%.3fs",
                    postprocessed,
                    postprocess_failed,
                    time.perf_counter() - postprocess_started,
                )
        # No callback/intermediate table write may publish a task ranking.
        # Every final structure score is now gathered (and OpenMM is complete).
        module.finalize_score_rankings()
        if logger is not None:
            for task in tasks:
                task_name = task.name
                message = "completed succeeded=%d failed=%d"
                counts = (
                    sum(key[0] == task_name for key in succeeded),
                    sum(key[0] == task_name for key in failed),
                )
                if task.score:
                    message += " scored=%d score_failed=%d"
                    counts += module.score_accounting(task_name)
                _log_task(logger, task_name, logging.INFO, message, *counts)
        rank, world_size = _distributed_rank_world()
        scoring_counts = module.score_accounting() if rank == 0 else ((0, 0) if not needs_scorer else None)
        if needs_scorer and world_size > 1:
            counts = [scoring_counts]
            torch.distributed.broadcast_object_list(counts, src=0)
            scoring_counts = counts[0]
        return DesignRunResult(
            output_dir,
            len(succeeded),
            len(failed),
            postprocessed,
            postprocess_failed,
            *scoring_counts,
        )
    except Exception as exc:
        if logger is not None:
            logger.error(
                "stage=run_fatal type=%s message=%s",
                type(exc).__name__,
                exc,
            )
        if module is not None:
            try:
                module.flush_postprocess_status(force=True)
            except Exception as csv_error:
                if logger is not None:
                    logger.error(
                        "stage=postprocess_status_flush type=%s message=%s",
                        type(csv_error).__name__,
                        csv_error,
                    )
            try:
                module._write_interrupted_result_csv()
            except Exception as csv_error:
                if logger is not None:
                    logger.error(
                        "stage=interrupted_result_csv type=%s message=%s",
                        type(csv_error).__name__,
                        csv_error,
                    )
        raise
    finally:
        try:
            if module is not None:
                try:
                    module.flush_postprocess_status(force=True)
                except Exception as csv_error:
                    if logger is not None:
                        logger.error(
                            "stage=postprocess_status_flush_finally type=%s message=%s",
                            type(csv_error).__name__,
                            csv_error,
                        )
            if module is not None and module._partiscore is not None:
                module._partiscore.close()
        finally:
            if launcher_rank == 0:
                _remove_output_ready(output_ready_marker)
            if logger is not None:
                _close_logger(logger)


__all__ = [
    "FKCandidateFactory",
    "TerminalCandidateFactory",
    "actual_joint_length",
    "build_fk_terminal_candidates",
    "build_terminal_candidates",
    "run_runtime",
]
