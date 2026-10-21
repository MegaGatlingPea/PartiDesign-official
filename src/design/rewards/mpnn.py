"""Offline batched MPNN adapters.

The model implementations and checkpoints live under ``community_models``.
This module translates finalized PartiDesign candidates into strict native
inputs, performs one native batch per compatible target, and validates the
returned sequences.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from partidesign.paths import (community_source, community_asset, asset_root, cache_dir, offline_site, code_root)
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from biotite import structure as struc
from biotite.structure.io import pdbx

from ..core.records import SearchCandidate
from ..model.constants import DNA_RES_ATOMS, PROTEIN_RES_ATOMS, RNA_RES_ATOMS
from ..utils.writer import write_pdb, write_structure_atomic
from .aa_omission import STANDARD_PROTEIN_AAS, resolve_rm_aa
from .esmfold2 import SequenceSourcePlan, protein_tokens
from .mpnn_cache import MPNNCacheError, run_mpnn_cached


class MPNNSequenceError(RuntimeError):
    """One candidate could not be converted, designed, or validated."""


class MPNNSystemError(RuntimeError):
    """The selected offline backend or runtime is unavailable or unhealthy."""


def _release_pythonpath(*sources: Path) -> str:
    """Return the release-owned import roots for a native worker.

    Native workers inherit the process environment, but their import search
    path is a release boundary. Keep the package root and immutable bundled
    source trees explicit; an arbitrary caller PYTHONPATH or model asset
    root must never become executable Python code.
    """

    values: list[str] = []
    seen: set[str] = set()
    for path in (code_root(), offline_site(), *sources):
        value = str(Path(path).expanduser().resolve())
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return os.pathsep.join(values)


@dataclass(frozen=True, slots=True)
class AntibodyMPNNPlan:
    """Explicit opt-in paired/single-chain antibody redesign side-car."""

    design_chain_ids: tuple[str, ...]
    mutable_residue_uids: tuple[int, ...]

    def __post_init__(self) -> None:
        chains = tuple(str(value) for value in self.design_chain_ids)
        if not chains or any(not value for value in chains):
            raise ValueError("antibody design_chain_ids must be non-empty")
        if len(chains) != len(set(chains)):
            raise ValueError("antibody design_chain_ids contains duplicates")
        raw_uids = tuple(self.mutable_residue_uids)
        normalized: list[int] = []
        for value in raw_uids:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError("antibody mutable residue UIDs must be integers")
            uid = int(value)
            if uid < 0:
                raise ValueError("antibody mutable residue UIDs must be non-negative")
            normalized.append(uid)
        if len(normalized) != len(set(normalized)):
            raise ValueError("antibody mutable residue UIDs contains duplicates")
        object.__setattr__(self, "design_chain_ids", chains)
        object.__setattr__(self, "mutable_residue_uids", tuple(normalized))

    @property
    def mutable_uid_set(self) -> frozenset[int]:
        return frozenset(int(value) for value in self.mutable_residue_uids)


@dataclass(frozen=True, slots=True)
class _ValidatedAntibodyMPNNPlan:
    plan: AntibodyMPNNPlan
    tokens_by_uid: Mapping[int, Any]
    tokens_by_chain: Mapping[str, tuple[Any, ...]]


_PROTEIN_AA3 = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL",
    }
)


def _antibody_annotation_values(atoms: Any, name: str) -> np.ndarray | None:
    if name not in atoms.get_annotation_categories():
        return None
    return np.asarray(getattr(atoms, name))


def build_antibody_mpnn_plan(
    candidate: SearchCandidate,
    *,
    design_chain_ids: Sequence[str] | None = None,
    mutable_residue_uids: Sequence[int] | None = None,
    sequence_plan: SequenceSourcePlan | None = None,
) -> AntibodyMPNNPlan:
    """Build an antibody side-car from explicit values or compiler annotations."""

    atoms = candidate.atoms
    chain_values = np.asarray(atoms.chain_id).astype(str)
    protein_mask = np.asarray(atoms.mol_type).astype(str) == "protein"
    if design_chain_ids is None:
        roles = _antibody_annotation_values(atoms, "antibody_chain_role_id")
        if roles is None:
            raise MPNNSequenceError(
                "antibody plan requires design_chain_ids or antibody_chain_role_id"
            )
        roles = np.asarray(roles, dtype=np.int64)
        design_chain_ids = tuple(
            sorted(
                {
                    str(chain)
                    for chain in chain_values[protein_mask & (roles > 0)].tolist()
                }
            )
        )
    if mutable_residue_uids is None:
        mutable_annotation = _antibody_annotation_values(
            atoms, "antibody_sequence_mutable"
        )
        if mutable_annotation is None:
            raise MPNNSequenceError(
                "antibody plan requires mutable_residue_uids or "
                "antibody_sequence_mutable"
            )
        residue_values = np.asarray(atoms.residue_uid, dtype=np.int64)
        mutable_annotation = np.asarray(mutable_annotation, dtype=bool)
        selected: list[int] = []
        for uid in dict.fromkeys(int(value) for value in residue_values):
            rows = np.flatnonzero(residue_values == uid)
            values = mutable_annotation[rows]
            if bool(values.any()) != bool(values.all()):
                raise MPNNSequenceError(
                    f"antibody mutable annotation is partial for residue UID {uid}"
                )
            if bool(values.all()):
                selected.append(uid)
        mutable_residue_uids = tuple(selected)
    if sequence_plan is not None:
        # Pipeline-derived mutability must also honor explicit user/FI fixes.
        fixed = set(sequence_plan.fixed_positions)
        protected = {
            int(token.residue_uid) for token in protein_tokens(atoms)
            if (str(token.chain_id), int(token.position)) in fixed
        }
        mutable_residue_uids = tuple(
            uid for uid in mutable_residue_uids if int(uid) not in protected
        )
    try:
        return AntibodyMPNNPlan(
            tuple(str(value) for value in design_chain_ids),
            tuple(mutable_residue_uids),
        )
    except (TypeError, ValueError) as exc:
        raise MPNNSequenceError(str(exc)) from exc


def _validate_antibody_mpnn_plan(
    candidate: SearchCandidate,
    plan: AntibodyMPNNPlan,
    *,
    sequence_plan: SequenceSourcePlan | None = None,
) -> _ValidatedAntibodyMPNNPlan:
    """Validate chain/UID ownership before any native model invocation."""

    if not isinstance(plan, AntibodyMPNNPlan):
        raise MPNNSequenceError("antibody plan must be an AntibodyMPNNPlan")
    tokens = protein_tokens(candidate.atoms)
    tokens_by_uid: dict[int, Any] = {}
    tokens_by_chain: dict[str, list[Any]] = {}
    for token in tokens:
        uid = int(token.residue_uid)
        if uid in tokens_by_uid:
            raise MPNNSequenceError(f"residue UID {uid} maps to multiple protein tokens")
        tokens_by_uid[uid] = token
        tokens_by_chain.setdefault(str(token.chain_id), []).append(token)
    known_chains = set(tokens_by_chain)
    chain_values = np.asarray(candidate.atoms.chain_id).astype(str)
    roles = _antibody_annotation_values(candidate.atoms, "antibody_chain_role_id")
    mutable_annotation = _antibody_annotation_values(
        candidate.atoms, "antibody_sequence_mutable"
    )
    residue_values = np.asarray(candidate.atoms.residue_uid, dtype=np.int64)
    for chain in plan.design_chain_ids:
        if chain not in known_chains:
            raise MPNNSequenceError(
                f"antibody design chain {chain!r} has no protein tokens"
            )
        rows = np.flatnonzero(chain_values == chain)
        modalities = {str(value) for value in candidate.atoms.mol_type[rows]}
        if modalities != {"protein"}:
            raise MPNNSequenceError(
                f"antibody design chain {chain!r} is not a pure protein chain"
            )
        if roles is not None and not bool(np.all(np.asarray(roles[rows], dtype=np.int64) > 0)):
            raise MPNNSequenceError(
                f"antibody design chain {chain!r} lacks a positive role annotation"
            )
    mutable = plan.mutable_uid_set
    for uid in mutable:
        token = tokens_by_uid.get(uid)
        if token is None:
            raise MPNNSequenceError(
                f"antibody mutable residue UID {uid} is missing from candidate"
            )
        if str(token.chain_id) not in plan.design_chain_ids:
            raise MPNNSequenceError(
                f"antibody mutable residue UID {uid} belongs to non-design chain "
                f"{token.chain_id!r}"
            )
        if str(token.residue_name).upper() not in _PROTEIN_AA3:
            raise MPNNSequenceError(
                f"antibody mutable residue UID {uid} is not a standard protein residue"
            )
        if mutable_annotation is not None:
            rows = np.flatnonzero(residue_values == uid)
            values = np.asarray(mutable_annotation[rows], dtype=bool)
            if bool(values.any()) != bool(values.all()):
                raise MPNNSequenceError(
                    f"antibody mutable annotation is partial for residue UID {uid}"
                )
            if not bool(values.all()):
                raise MPNNSequenceError(
                    f"antibody mutable residue UID {uid} is not annotation-selected"
                )
    if sequence_plan is not None:
        fixed = {
            (str(chain), int(position))
            for chain, position in sequence_plan.fixed_positions
        }
        for uid in mutable:
            token = tokens_by_uid[uid]
            if (str(token.chain_id), int(token.position)) in fixed:
                raise MPNNSequenceError(
                    f"antibody mutable residue UID {uid} is also identity-fixed"
                )
    return _ValidatedAntibodyMPNNPlan(
        plan=plan,
        tokens_by_uid=tokens_by_uid,
        tokens_by_chain={chain: tuple(values) for chain, values in tokens_by_chain.items()},
    )


def _antibody_compatibility_plan(
    sequence_plan: SequenceSourcePlan,
    selection: _ValidatedAntibodyMPNNPlan,
) -> SequenceSourcePlan:
    """Retain the historical scalar design-chain field for private helpers."""

    fixed = {
        (str(chain), int(position))
        for chain, position in sequence_plan.fixed_positions
    }
    mutable = selection.plan.mutable_uid_set
    for chain, tokens in selection.tokens_by_chain.items():
        for token in tokens:
            if (
                chain not in selection.plan.design_chain_ids
                or int(token.residue_uid) not in mutable
            ):
                fixed.add((chain, int(token.position)))
    return replace(
        sequence_plan,
        design_chain=selection.plan.design_chain_ids[0],
        fixed_positions=tuple(sorted(fixed)),
    )


def _antibody_fixed_payload(
    prepared: Any,
    sample: Any,
    selection: _ValidatedAntibodyMPNNPlan,
) -> tuple[list[str], list[str], dict[str, list[int]]]:
    """Build native ProteinMPNN chain/fixed masks from residue UIDs."""

    design_aliases = sorted(
        prepared.alias_by_chain[chain]
        for chain in selection.plan.design_chain_ids
    )
    protein_aliases = sorted(
        prepared.alias_by_chain[record.chain_id]
        for record in sample.chains
        if record.modality == "protein"
    )
    fixed_aliases = [alias for alias in protein_aliases if alias not in design_aliases]
    uid_by_chain_position = {
        (str(token.chain_id), int(token.position)): int(token.residue_uid)
        for token in selection.tokens_by_uid.values()
    }
    fixed: dict[str, list[int]] = {}
    mutable = selection.plan.mutable_uid_set
    for chain, residues in prepared.residues_by_chain.items():
        alias = prepared.alias_by_chain[chain]
        if alias not in protein_aliases:
            continue
        positions: list[int] = []
        for residue in residues:
            uid = uid_by_chain_position.get((chain, int(residue.canonical_position)))
            if uid is None or uid not in mutable:
                positions.append(int(residue.temporary_position))
        fixed[alias] = positions
    return design_aliases, fixed_aliases, fixed


def _validate_antibody_sequences(
    sequence_plan: SequenceSourcePlan,
    selection: _ValidatedAntibodyMPNNPlan,
    chain_sequences: Mapping[str, str],
) -> SequenceSourcePlan:
    """Validate every physical chain and update only selected residue UIDs."""

    expected = dict(sequence_plan.chain_sequences)
    if set(chain_sequences) != set(expected):
        raise MPNNSequenceError("antibody MPNN output chain set differs from input")
    mutable = selection.plan.mutable_uid_set
    for chain, source in expected.items():
        observed = str(chain_sequences[chain]).strip().upper()
        if len(observed) != len(source):
            raise MPNNSequenceError(
                f"antibody MPNN chain {chain!r} has length {len(observed)}; "
                f"expected {len(source)}"
            )
        if any(letter not in "ACDEFGHIKLMNPQRSTVWY" for letter in observed):
            raise MPNNSequenceError(
                f"antibody MPNN chain {chain!r} contains invalid amino acid"
            )
        for token in selection.tokens_by_chain.get(chain, ()):
            position = int(token.position)
            if (
                int(token.residue_uid) not in mutable
                and observed[position - 1] != source[position - 1]
            ):
                raise MPNNSequenceError(
                    f"antibody MPNN changed protected residue {chain}:{position}"
                )
    updated = dict(expected)
    for chain in selection.plan.design_chain_ids:
        updated[chain] = str(chain_sequences[chain]).strip().upper()
    return replace(
        sequence_plan,
        chain_sequences=updated,
        design_chain=selection.plan.design_chain_ids[0],
    )


_NA_PDB_CHAIN_ALIASES = tuple(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)
_POLYMER_RES_ATOMS = {
    "protein": PROTEIN_RES_ATOMS,
    "rna": RNA_RES_ATOMS,
    "dna": DNA_RES_ATOMS,
}
_CUDA_DEVICE_TOKEN = re.compile(
    r"(?:[0-9]+|GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}|"
    r"MIG-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12})"
)


def _stable_chain_order(atoms: Any) -> list[str]:
    """Return the existing authoring chain order without row-order dependence."""

    chain_values = np.asarray(atoms.chain_id).astype(str)
    asym_values = np.asarray(atoms.asym_uid, dtype=np.int64)
    token_values = np.asarray(atoms.token_uid, dtype=np.int64)
    chains = sorted(set(chain_values.tolist()))

    def key(chain: str) -> tuple[int, int, str]:
        rows = np.flatnonzero(chain_values == chain)
        asym_uids = asym_values[rows]
        token_uids = token_values[rows]
        return (
            int(asym_uids.min()) if len(asym_uids) else 2**63 - 1,
            int(token_uids.min()) if len(token_uids) else 2**63 - 1,
            chain,
        )

    return sorted(chains, key=key)


_MPNN_POLYMER_MODALITIES = frozenset({"protein", "rna", "dna", "nuc"})


@dataclass(frozen=True, slots=True)
class _MPNNIdentityGroup:
    """One private MPNN residue/component and its source identity."""

    chain_id: str
    modality: str
    rows: tuple[int, ...]
    position: int
    source_res_id: int
    source_ins_code: str


def _mpnn_identity_groups(
    atoms: Any,
    chain_order: Sequence[str],
) -> dict[str, list[_MPNNIdentityGroup]]:
    """Build the single identity map shared by the ledger and private CIF.

    Polymer authoring tokens are residue-level, whereas ligand authoring tokens
    are atom-level.  Ligand components therefore group by ``component_uid``;
    source residue IDs and insertion codes are retained only as provenance and
    never used as the private MPNN position.
    """

    chain_values = np.asarray(atoms.chain_id).astype(str)
    token_values = np.asarray(atoms.token_uid, dtype=np.int64)
    component_values = np.asarray(atoms.component_uid, dtype=np.int64)
    residue_values = np.asarray(atoms.res_id, dtype=np.int64)
    insertion_values = np.asarray(atoms.ins_code).astype(str)
    atom_values = np.asarray(atoms.atom_uid, dtype=np.int64)
    groups_by_chain: dict[str, list[_MPNNIdentityGroup]] = {}

    for chain in chain_order:
        chain_rows = np.flatnonzero(chain_values == chain)
        modalities = {str(value) for value in atoms.mol_type[chain_rows]}
        if len(modalities) != 1:
            raise MPNNSequenceError(f"chain {chain!r} spans multiple modalities")
        modality = next(iter(modalities))
        if modality not in _MPNN_POLYMER_MODALITIES and modality != "ligand":
            raise MPNNSequenceError(
                f"MPNN cannot canonicalize unsupported chain modality {modality!r}"
            )

        if modality in _MPNN_POLYMER_MODALITIES:
            group_ids = sorted({int(value) for value in token_values[chain_rows]})
            if any(value < 0 for value in group_ids):
                raise MPNNSequenceError(
                    f"MPNN chain {chain!r} has an unfinished polymer token ledger"
                )
            grouped_rows = [
                (group_id, chain_rows[token_values[chain_rows] == group_id])
                for group_id in group_ids
            ]
        else:
            group_ids = sorted({int(value) for value in component_values[chain_rows]})
            if any(value < 0 for value in group_ids):
                raise MPNNSequenceError(
                    f"MPNN ligand chain {chain!r} has an unfinished component ledger"
                )
            grouped_rows = [
                (group_id, chain_rows[component_values[chain_rows] == group_id])
                for group_id in group_ids
            ]
            # A component's private position must be stable under atom-row
            # shuffling. Source identity orders distinct components; UID and
            # atom UID only break ties when the source identifiers collide.
            grouped_rows.sort(
                key=lambda item: (
                    int(residue_values[int(item[1][0])]),
                    str(insertion_values[int(item[1][0])]).strip(),
                    int(item[0]),
                    int(atom_values[item[1]].min()),
                )
            )

        groups: list[_MPNNIdentityGroup] = []
        for position, (group_id, raw_rows) in enumerate(grouped_rows, start=1):
            rows = tuple(
                sorted(
                    (int(index) for index in raw_rows),
                    key=lambda index: (
                        str(atoms.atom_name[index]),
                        int(atom_values[index]),
                        index,
                    ),
                )
            )
            source_keys = {
                (
                    int(residue_values[index]),
                    str(insertion_values[index]).strip(),
                )
                for index in rows
            }
            if len(source_keys) != 1:
                raise MPNNSequenceError(
                    f"MPNN {modality} component {chain}:{group_id} spans "
                    "multiple source residue identities"
                )
            source_res_id, source_ins_code = next(iter(source_keys))
            groups.append(
                _MPNNIdentityGroup(
                    chain_id=chain,
                    modality=modality,
                    rows=rows,
                    position=position,
                    source_res_id=source_res_id,
                    source_ins_code=source_ins_code,
                )
            )
        groups_by_chain[chain] = groups
    return groups_by_chain


def _sanitized_process_log_tail(path: Path, *, limit: int = 2_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unavailable"
    text = "".join(character if character.isprintable() else " " for character in text)
    return text[-limit:].strip() or "empty"


def _structured_asset(path_value: Any, *, expected: type, where: str) -> Any:
    path = Path(str(path_value)).expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise MPNNSystemError(f"{where} asset is missing or unsafe: {path}")
    try:
        value = yaml.safe_load(path.read_text())
    except Exception as exc:
        raise MPNNSystemError(f"cannot read {where} asset: {type(exc).__name__}") from exc
    if not isinstance(value, expected):
        raise MPNNSystemError(f"{where} asset must contain {expected.__name__}")
    return value


def _backend_assets(config: Mapping[str, Any], source: str) -> None:
    checkpoint = Path(str(config["checkpoint"])).expanduser().absolute()
    if not checkpoint.is_file() or checkpoint.is_symlink() or checkpoint.stat().st_size == 0:
        raise MPNNSystemError(f"{source} checkpoint is missing or unsafe: {checkpoint}")


def _has_explicit_rm_aa(settings: Mapping[str, Any]) -> bool:
    """Return whether the unified policy overrides legacy ``omit_aas``."""

    return "rm_aa" in settings and settings["rm_aa"] is not None


def _check_rm_aa_has_allowed_free_position(
    backend: Mapping[str, Any], positions: Sequence[int]
) -> None:
    """Reject an explicit policy that removes every free protein residue type."""

    if (
        _has_explicit_rm_aa(backend)
        and positions
        and set(resolve_rm_aa(backend)) == STANDARD_PROTEIN_AAS
    ):
        raise MPNNSequenceError(
            "rm_aa omits all standard amino acids at free design positions"
        )


def _free_design_protein_positions(
    candidate: SearchCandidate,
    plan: SequenceSourcePlan,
) -> tuple[int, ...]:
    """Return free design-identity positions for a private MPNN view.

    Native MPNN omission controls are vocabulary-wide by default.  Explicit
    ``rm_aa`` uses the native per-residue controls instead, so coordinate-only
    motif tokens stay sequence-free and are not converted into fixed identity.
    """

    if plan.design_chain is None:
        return ()
    atoms = candidate.atoms
    tokens = protein_tokens(atoms)
    fixed = {(str(chain), int(position)) for chain, position in plan.fixed_positions}
    atom_count = len(np.asarray(atoms.token_uid))
    motif = np.asarray(
        getattr(atoms, "motif_coord_condition", np.zeros(atom_count, dtype=bool)),
        dtype=bool,
    )
    center = np.asarray(
        getattr(atoms, "token_center_condition", np.zeros(atom_count, dtype=bool)),
        dtype=bool,
    )
    token_uids = np.asarray(atoms.token_uid, dtype=np.int64)
    positions: list[int] = []
    for token in tokens:
        key = (str(token.chain_id), int(token.position))
        if (
            token.chain_id != plan.design_chain
            or not token.is_design
            or token.identity_conditioned
            or key in fixed
        ):
            continue
        rows = np.flatnonzero(token_uids == int(token.token_uid))
        if bool(motif[rows].any()) or bool(center[rows].any()):
            continue
        positions.append(int(token.position))
    return tuple(sorted(set(positions)))


def _protein_omit_payload(
    candidate: SearchCandidate,
    plan: SequenceSourcePlan,
    *,
    name: str,
    design_alias: str,
    backend: Mapping[str, Any],
    chain_aliases: Sequence[str] = (),
) -> dict[str, Any] | None:
    if not _has_explicit_rm_aa(backend):
        return None
    omission = resolve_rm_aa(backend)
    positions = _free_design_protein_positions(candidate, plan)
    _check_rm_aa_has_allowed_free_position(backend, positions)
    if not omission or not positions:
        return None
    payload: dict[str, Any] = {
        design_alias: [[list(positions), omission]],
    }
    # ProteinMPNN indexes this mapping for every parsed protein chain, even
    # when no per-residue omission applies to a context chain.
    for alias in chain_aliases:
        payload.setdefault(str(alias), [])
    return {name: payload}


def _ligand_omit_payload(
    candidate: SearchCandidate,
    plan: SequenceSourcePlan,
    prepared: Any,
    *,
    design_alias: str,
    backend: Mapping[str, Any],
) -> dict[str, str] | None:
    if not _has_explicit_rm_aa(backend):
        return None
    omission = resolve_rm_aa(backend)
    positions = _free_design_protein_positions(candidate, plan)
    _check_rm_aa_has_allowed_free_position(backend, positions)
    if not omission or not positions:
        return None
    residues = {
        int(residue.canonical_position): int(residue.temporary_position)
        for residue in prepared.residues_by_chain[plan.design_chain]
    }
    result: dict[str, str] = {}
    for position in positions:
        try:
            temporary = residues[int(position)]
        except KeyError as exc:
            raise MPNNSequenceError(
                f"LigandMPNN omission position {position} is absent from the prepared chain"
            ) from exc
        result[f"{design_alias}{temporary}"] = omission
    return result


def _antibody_omit_payload(
    selection: _ValidatedAntibodyMPNNPlan,
    prepared: Any,
    *,
    name: str,
    ligand: bool,
    backend: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build per-residue omission for explicitly mutable antibody residues."""

    if not _has_explicit_rm_aa(backend):
        return None
    omission = resolve_rm_aa(backend)
    if not omission:
        return None
    uid_by_chain_position = {
        (str(token.chain_id), int(token.position)): int(token.residue_uid)
        for token in selection.tokens_by_uid.values()
    }
    mutable = selection.plan.mutable_uid_set
    values: dict[str, Any] = {}
    positions: list[int] = []
    for chain in selection.plan.design_chain_ids:
        alias = prepared.alias_by_chain[chain]
        for residue in prepared.residues_by_chain[chain]:
            uid = uid_by_chain_position.get(
                (chain, int(residue.canonical_position))
            )
            if uid not in mutable:
                continue
            position = int(residue.temporary_position)
            positions.append(position)
            if ligand:
                values[f"{alias}{position}"] = omission
            else:
                values.setdefault(alias, []).append(position)
    _check_rm_aa_has_allowed_free_position(backend, positions)
    if not values:
        return None
    if ligand:
        return values
    protein_aliases = {
        str(prepared.alias_by_chain[chain])
        for chain in selection.tokens_by_chain
    }
    return {
        name: {
            alias: (
                [[values[alias], omission]]
                if alias in values
                else []
            )
            for alias in sorted(protein_aliases)
        }
    }


def _protein_model_contract(config: Mapping[str, Any]) -> tuple[str, bool, bool]:
    """Return the exact official ProteinMPNN family selected by public config."""

    model_type = str(config["model_type"])
    expected = {
        "protein_mpnn": ("vanilla", False, False),
        "soluble_mpnn": ("soluble", False, True),
        "ca_protein_mpnn": ("ca", True, False),
    }
    if model_type not in expected:
        raise MPNNSystemError(
            "proteinmpnn model_type must be protein_mpnn, soluble_mpnn, or "
            "ca_protein_mpnn"
        )
    family, ca_only, soluble = expected[model_type]
    if bool(config["ca_only"]) != ca_only or bool(config["use_soluble_model"]) != soluble:
        raise MPNNSystemError(
            f"proteinmpnn model_type={model_type} conflicts with ca_only/"
            "use_soluble_model"
        )
    return family, ca_only, soluble


def _ligand_checkpoint_flag(model_type: str) -> str:
    flags = {
        "ligand_mpnn": "--checkpoint_ligand_mpnn",
        "protein_mpnn": "--checkpoint_protein_mpnn",
        "soluble_mpnn": "--checkpoint_soluble_mpnn",
    }
    try:
        return flags[model_type]
    except KeyError as exc:
        raise MPNNSystemError(
            f"unsupported LigandMPNN-suite model_type {model_type!r}"
        ) from exc


def _offline_environment(repo_root: Path) -> dict[str, str]:
    source = community_source("pipetools")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from pipetools.errors import AdapterError  # type: ignore[import-not-found]
    from pipetools.runtime import (  # type: ignore[import-not-found]
        resolve_owner_cuda_visible_device,
    )

    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PIP_NO_INDEX": "1",
            "PIPETOOLS_NETWORK_POLICY": "fold-offline",
            "PIPETOOLS_MODEL_ROOT": str(asset_root(repo_root) / "community_models"),
            "PIPETOOLS_MPNN_PYTHON": sys.executable,
        }
    )
    for key in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
    ):
        env.pop(key, None)
    try:
        env["CUDA_VISIBLE_DEVICES"] = resolve_owner_cuda_visible_device(env)
    except AdapterError as exc:
        raise MPNNSystemError(str(exc)) from exc
    env["PYTHONPATH"] = _release_pythonpath(source)
    return env


def _load_pipetools(repo_root: Path) -> tuple[Any, Any, Any, Any, Any]:
    del repo_root
    source = community_source("pipetools")
    if not source.is_dir():
        raise MPNNSystemError(f"offline PipeTools source is missing: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from pipetools.adapters.mpnn import (  # type: ignore[import-not-found]
        prepare_mpnn_input,
        run_ligandmpnn,
        run_proteinmpnn,
    )
    from pipetools.config import SequenceConfig  # type: ignore[import-not-found]
    from pipetools.model import (  # type: ignore[import-not-found]
        CanonicalSample,
        ChainRecord,
        ResidueRecord,
    )
    from pipetools.runtime import RuntimePaths  # type: ignore[import-not-found]
    return (
        (prepare_mpnn_input, run_ligandmpnn, run_proteinmpnn),
        SequenceConfig,
        (CanonicalSample, ChainRecord, ResidueRecord),
        RuntimePaths,
        source,
    )


def _mpnn_break_before(
    atoms: Any,
    groups: Sequence[_MPNNIdentityGroup],
) -> set[int]:
    """Return group indices that must start a new private context chain.

    The public AtomArray remains untouched.  Explicit residue-number gaps,
    missing peptide bonds when a bond graph is available, and large CA jumps
    are all treated as receptor breaks.  Design-chain breaks are rejected by
    the caller because the design contract is one continuous chain.
    """
    if len(groups) < 2:
        return set()
    residue_names = np.asarray(atoms.atom_name).astype(str)
    residue_ids = np.asarray(atoms.res_id, dtype=np.int64)
    bond_pairs = None
    if getattr(atoms, "bonds", None) is not None:
        bond_pairs = {
            tuple(sorted((int(left), int(right))))
            for left, right, _order in atoms.bonds.as_array().tolist()
        }
    coordinates = np.asarray(getattr(atoms, "coord", np.zeros((len(atoms), 3))), dtype=float)
    valid = np.asarray(
        getattr(atoms, "coord_valid", np.isfinite(coordinates).all(axis=1)),
        dtype=bool,
    )
    breaks: set[int] = set()
    for index, (left, right) in enumerate(zip(groups, groups[1:]), start=1):
        if right.source_res_id - left.source_res_id not in {0, 1}:
            breaks.add(index)
            continue
        left_names = {
            residue_names[row].strip().upper(): row for row in left.rows
        }
        right_names = {
            residue_names[row].strip().upper(): row for row in right.rows
        }
        if "C" in left_names and "N" in right_names and bond_pairs is not None:
            if tuple(sorted((left_names["C"], right_names["N"]))) not in bond_pairs:
                breaks.add(index)
                continue
        if "CA" in left_names and "CA" in right_names:
            left_row, right_row = left_names["CA"], right_names["CA"]
            if (
                valid[left_row]
                and valid[right_row]
                and np.isfinite(coordinates[[left_row, right_row]]).all()
                and float(np.linalg.norm(coordinates[right_row] - coordinates[left_row])) > 4.8
            ):
                breaks.add(index)
    return breaks


def _split_mpnn_context_groups(
    atoms: Any,
    chain: str,
    groups: Sequence[_MPNNIdentityGroup],
    *,
    design_chain: str,
) -> list[tuple[str, list[_MPNNIdentityGroup]]]:
    # Generated design polymers are continuous by contract. Coordinate quality
    # must not redefine their connectivity or prevent sequence design.
    breaks = set() if chain == design_chain else _mpnn_break_before(atoms, groups)
    segments: list[tuple[str, list[_MPNNIdentityGroup]]] = []
    current: list[_MPNNIdentityGroup] = []
    segment_index = 0
    for index, group in enumerate(groups):
        if index in breaks and current:
            segment_id = (
                chain if segment_index == 0 else f"{chain}__break{segment_index}"
            )
            segments.append((segment_id, current))
            segment_index += 1
            current = []
        current.append(group)
    if current:
        segment_id = chain if segment_index == 0 else f"{chain}__break{segment_index}"
        segments.append((segment_id, current))
    return segments


def _canonical_sample(
    candidate: SearchCandidate,
    plan: SequenceSourcePlan,
    work_dir: Path,
    model_types: tuple[Any, Any, Any],
    *,
    design_chain_ids: Sequence[str] | None = None,
) -> Any:
    CanonicalSample, ChainRecord, ResidueRecord = model_types
    atoms = candidate.atoms
    if plan.design_chain is None:
        raise MPNNSequenceError("MPNN requires exactly one design protein chain")
    explicit_design_chains = (
        None
        if design_chain_ids is None
        else frozenset(str(chain) for chain in design_chain_ids)
    )
    chain_order = _stable_chain_order(atoms)
    identity_groups = _mpnn_identity_groups(atoms, chain_order)
    segmented: dict[str, list[tuple[str, list[_MPNNIdentityGroup]]]] = {
        chain: _split_mpnn_context_groups(
            atoms,
            chain,
            identity_groups[chain],
            design_chain=(
                chain
                if explicit_design_chains is not None
                and chain in explicit_design_chains
                else plan.design_chain
            ),
        )
        for chain in chain_order
    }
    chain_records: list[Any] = []
    residue_records: list[Any] = []
    private_chain_by_row: dict[int, str] = {}
    private_position_by_row: dict[int, int] = {}
    for chain in chain_order:
        rows = np.flatnonzero(np.asarray(atoms.chain_id).astype(str) == chain)
        modalities = set(str(value) for value in atoms.mol_type[rows])
        if len(modalities) != 1:
            raise MPNNSequenceError(f"chain {chain!r} spans multiple modalities")
        modality = next(iter(modalities))
        source_sequence = plan.chain_sequences.get(chain, "")
        for segment_id, segment_groups in segmented[chain]:
            first_position = segment_groups[0].position
            last_position = segment_groups[-1].position
            sequence = (
                source_sequence[first_position - 1:last_position]
                if source_sequence
                else ""
            )
            chain_records.append(
                ChainRecord(
                    chain_id=segment_id,
                    entity_id=str(int(atoms.entity_uid[rows[0]]) + 1),
                    modality=modality,
                    sequence=sequence,
                    entity_key=f"entity:{int(atoms.entity_uid[rows[0]])}",
                    source_chain_id=chain,
                )
            )
            for position, group in enumerate(segment_groups, start=1):
                row = group.rows[0]
                for atom_row in group.rows:
                    private_chain_by_row[int(atom_row)] = segment_id
                    private_position_by_row[int(atom_row)] = position
                residue_records.append(
                    ResidueRecord(
                        chain_id=segment_id,
                        position=position,
                        ins_code="",
                        source_chain_id=chain,
                        source_res_id=group.source_res_id,
                        source_ins_code=group.source_ins_code,
                        res_name=str(atoms.res_name[row]),
                        atom_names=tuple(
                            str(atoms.atom_name[index]) for index in group.rows
                        ),
                        atom_elements=tuple(
                            str(atoms.element[index]) for index in group.rows
                        ),
                    )
                )
    by_chain = {record.chain_id: record for record in chain_records}
    design = by_chain[plan.design_chain]
    private_atoms = copy.deepcopy(atoms)
    base_asym = int(np.asarray(atoms.asym_uid, dtype=np.int64).max()) + 1
    private_asym = {
        record.chain_id: base_asym + index
        for index, record in enumerate(chain_records)
    }
    for row, segment_id in private_chain_by_row.items():
        private_atoms.chain_id[row] = segment_id
        if hasattr(private_atoms, "asym_uid"):
            private_atoms.asym_uid[row] = private_asym[segment_id]
        for annotation in ("label_asym_id", "auth_asym_id"):
            if hasattr(private_atoms, annotation):
                getattr(private_atoms, annotation)[row] = segment_id
    canonical_cif = work_dir / "candidate.cif"
    write_structure_atomic(private_atoms, canonical_cif, entry_id=candidate.candidate_id)
    # Normalize only this owner-local transient CIF to canonical positions.
    cif = pdbx.CIFFile.read(canonical_cif)
    block = cif[next(iter(cif))]
    atom_site = block["atom_site"]
    if len(atom_site["label_seq_id"]) != len(atoms):
        raise MPNNSequenceError("private MPNN CIF changed atom row count")
    label_seq_id = list(atom_site["label_seq_id"].as_array().astype(str))
    if len(private_position_by_row) != len(atoms):
        raise MPNNSequenceError("private MPNN identity map does not cover every atom")
    for row, position in private_position_by_row.items():
        label_seq_id[row] = str(position)
    atom_site["label_seq_id"] = label_seq_id
    if "pdbx_PDB_ins_code" in atom_site:
        atom_site["pdbx_PDB_ins_code"] = ["."] * len(atoms)
    cif.write(canonical_cif)
    motif = [
        (chain, int(position), "")
        for chain, position in plan.fixed_positions
        if chain == plan.design_chain
    ]
    return CanonicalSample(
        sample_id=candidate.candidate_id,
        source_path=canonical_cif,
        canonical_cif=canonical_cif,
        canonical_json=work_dir / "candidate.json",
        inference_json=work_dir / "candidate.input.json",
        chains=chain_records,
        design_chain=design,
        context_chains=[record for record in chain_records if record is not design],
        residues=residue_records,
        motif_residues=motif,
    )

def _runtime_paths(repo_root: Path, RuntimePaths: Any) -> Any:
    models = asset_root(repo_root) / "community_models"
    return RuntimePaths(
        repo_root=repo_root,
        model_root=models,
        cache_root=cache_dir("partidesign_reward", repo_root),
        mpnn_python=Path(sys.executable),
        proteinmpnn_root=models / "proteinmpnn",
        ligandmpnn_root=models / "ligandmpnn",
        esmfold2_root=models / "esmfold2",
    )


def _validate_sequence(
    plan: SequenceSourcePlan,
    sequence: str,
) -> SequenceSourcePlan:
    if plan.design_chain is None:
        raise MPNNSequenceError("MPNN returned a sequence without a design chain")
    source = plan.chain_sequences[plan.design_chain]
    sequence = sequence.strip().upper()
    if len(sequence) != len(source) or any(letter not in "ACDEFGHIKLMNPQRSTVWY" for letter in sequence):
        raise MPNNSequenceError("MPNN returned an invalid design-chain sequence")
    for chain, position in plan.fixed_positions:
        if chain == plan.design_chain and sequence[position - 1] != source[position - 1]:
            raise MPNNSequenceError(
                f"MPNN changed fixed residue {chain}{position}"
            )
    sequences = dict(plan.chain_sequences)
    sequences[plan.design_chain] = sequence
    return replace(plan, chain_sequences=sequences)


def _validate_na_sequence(
    plan: SequenceSourcePlan,
    sequence: str,
    *,
    modality: str,
) -> SequenceSourcePlan:
    """Validate a DNA/RNA worker result and replace only the design chain."""

    if plan.design_chain is None:
        raise MPNNSequenceError("NA-MPNN returned a sequence without a design chain")
    source = plan.chain_sequences.get(plan.design_chain)
    if source is None:
        raise MPNNSequenceError("NA-MPNN plan lacks its design-chain sequence")
    alphabet = {"dna": frozenset("ACGT"), "rna": frozenset("ACGU")}.get(modality)
    if alphabet is None:
        raise MPNNSequenceError(f"NA-MPNN design modality is unsupported: {modality!r}")
    sequence = sequence.strip().upper()
    if len(sequence) != len(source) or any(letter not in alphabet for letter in sequence):
        raise MPNNSequenceError(
            f"NA-MPNN returned an invalid {modality.upper()} design-chain sequence"
        )
    for chain, position in plan.fixed_positions:
        if chain == plan.design_chain and sequence[position - 1] != source[position - 1]:
            raise MPNNSequenceError(
                f"NA-MPNN changed fixed residue {chain}{position}"
            )
    sequences = dict(plan.chain_sequences)
    sequences[plan.design_chain] = sequence
    return replace(plan, chain_sequences=sequences)


def _na_private_pdb(
    candidate: SearchCandidate,
    plan: SequenceSourcePlan,
    output_path: Path,
) -> tuple[dict[str, str], str]:
    """Write a canonical, private standard-polymer PDB for the NA worker.

    The public candidate keeps its source chain/residue identifiers and bond
    graph.  This owner-local view only supplies one-character chain aliases,
    position-only polymer numbering, and coordinates; no private topology is
    allowed to leak into the public output.
    """

    if plan.design_chain is None:
        raise MPNNSequenceError("NA-MPNN plan lacks its design chain")
    atoms = getattr(candidate, "atoms", None)
    if atoms is None or len(atoms) == 0:
        raise MPNNSequenceError("NA-MPNN candidate has no atoms")
    source = copy.deepcopy(atoms)
    polymer_modalities = frozenset({"protein", "rna", "dna"})
    modality_values = np.asarray(source.mol_type).astype(str)
    # The native NA worker accepts only standard polymer atoms.  Keep the
    # public candidate untouched and remove ligand/non-polymer context from
    # this private view before any PDB aliasing or writing occurs.
    keep = np.isin(modality_values, tuple(polymer_modalities))
    modified_names = getattr(candidate, "fixed_modified_residue_names", {})
    modified_parents = getattr(candidate, "fixed_modified_parent_residue", {})
    for residue_uid, component_id in modified_names.items():
        rows = np.flatnonzero(
            np.asarray(source.residue_uid, dtype=np.int64) == int(residue_uid)
        )
        if not len(rows):
            raise MPNNSequenceError(
                f"NA-MPNN modified residue {component_id!r} is absent"
            )
        modalities = {str(value) for value in source.mol_type[rows]}
        if len(modalities) != 1:
            raise MPNNSequenceError(
                f"NA-MPNN modified residue {component_id!r} has ambiguous modality"
            )
        modality = next(iter(modalities))
        if modality not in polymer_modalities:
            keep[rows] = False
            continue
        parent = modified_parents.get(int(residue_uid))
        if parent not in _POLYMER_RES_ATOMS.get(modality, {}):
            raise MPNNSequenceError(
                f"NA-MPNN modified residue {component_id!r} lacks a canonical parent"
            )
        if modality == "protein" and component_id == "MSE" and parent == "MET":
            selenium = rows[np.asarray(source.atom_name[rows]).astype(str) == "SE"]
            source.atom_name[selenium] = "SD"
            source.element[selenium] = "S"
        allowed = set(_POLYMER_RES_ATOMS[modality][parent])
        for row in rows:
            if str(source.atom_name[row]) not in allowed:
                keep[row] = False
        source.res_name[rows] = parent
        source.hetero[rows] = False
    if not bool(keep.all()):
        source = source[keep].copy()
    chain_values = np.asarray(source.chain_id).astype(str)
    chain_order = _stable_chain_order(source)
    if len(chain_order) > len(_NA_PDB_CHAIN_ALIASES):
        raise MPNNSequenceError("NA-MPNN private PDB has too many chains for PDB aliases")
    alias_by_chain = {
        chain: _NA_PDB_CHAIN_ALIASES[index]
        for index, chain in enumerate(chain_order)
    }
    if plan.design_chain not in alias_by_chain:
        raise MPNNSequenceError(
            f"NA-MPNN design chain is absent from candidate: {plan.design_chain!r}"
        )

    position_by_token: dict[tuple[str, int], int] = {}
    for chain in chain_order:
        token_uids = sorted(
            {
                int(value)
                for value in np.asarray(
                    source.token_uid[chain_values == chain], dtype=np.int64
                )
            }
        )
        if any(token_uid < 0 for token_uid in token_uids):
            raise MPNNSequenceError("NA-MPNN candidate has an unfinished token ledger")
        position_by_token.update(
            {
                (chain, token_uid): position
                for position, token_uid in enumerate(token_uids, start=1)
            }
        )
    for row, chain in enumerate(chain_values):
        token_uid = int(source.token_uid[row])
        key = (chain, token_uid)
        private_chain = alias_by_chain[chain]
        private_position = position_by_token[key]
        source.chain_id[row] = private_chain
        source.label_asym_id[row] = private_chain
        source.auth_asym_id[row] = private_chain
        source.res_id[row] = private_position
        source.label_seq_id[row] = str(private_position)
        source.auth_seq_id[row] = str(private_position)
        source.ins_code[row] = ""
        source.auth_ins_code[row] = ""
        source.hetero[row] = False

    # Explicit conditions are represented in the model's fixed-position mask,
    # not in the transient PDB.  Dropping the graph also lets standard
    # polymer input with a cyclic/inter-chain condition remain PDB-readable.
    source.bonds = struc.BondList(len(source))
    try:
        write_pdb(source, output_path)
    except Exception as exc:
        raise MPNNSequenceError(
            f"cannot materialize NA-MPNN private PDB: {type(exc).__name__}"
        ) from exc
    return alias_by_chain, alias_by_chain[plan.design_chain]


def _resolve_na_owner_cuda_visible_device(environment: Mapping[str, str]) -> str:
    """Resolve one physical GPU for a standalone NA worker.

    ``CUDA_VISIBLE_DEVICES`` is a physical-device namespace owned by the
    parent.  A multi-device parent selects its owner with ``LOCAL_RANK``;
    when the parent already exposes one device, that device remains the owner
    regardless of local rank.  An absent list follows the standalone CUDA
    ordering policy and selects the local rank (or device zero).

    This is deliberately local to the NA adapter: the NA subprocess must not
    import the PipeTools runtime merely to resolve its physical device.
    """

    raw_rank = environment.get("LOCAL_RANK")
    if raw_rank is None:
        local_rank = 0
    else:
        try:
            local_rank = int(raw_rank)
        except (TypeError, ValueError) as exc:
            raise MPNNSystemError(
                f"Invalid LOCAL_RANK for NA-MPNN owner GPU: {raw_rank!r}"
            ) from exc
        if local_rank < 0:
            raise MPNNSystemError(
                f"Invalid LOCAL_RANK for NA-MPNN owner GPU: {raw_rank!r}"
            )

    raw_visible = environment.get("CUDA_VISIBLE_DEVICES")
    if raw_visible is None:
        return str(local_rank)
    if not raw_visible.strip():
        raise MPNNSystemError(
            "CUDA_VISIBLE_DEVICES explicitly hides every NA-MPNN owner GPU"
        )
    visible = [value.strip() for value in raw_visible.split(",")]
    if any(not value or _CUDA_DEVICE_TOKEN.fullmatch(value) is None for value in visible):
        raise MPNNSystemError(
            "CUDA_VISIBLE_DEVICES does not name a usable NA-MPNN GPU set"
        )
    if len(set(visible)) != len(visible):
        raise MPNNSystemError(
            "CUDA_VISIBLE_DEVICES contains duplicate NA-MPNN GPU owners"
        )
    if len(visible) == 1:
        return visible[0]
    if local_rank >= len(visible):
        raise MPNNSystemError(
            "LOCAL_RANK does not resolve one NA-MPNN owner GPU"
        )
    return visible[local_rank]


def _na_offline_environment(repo_root: Path) -> dict[str, str]:
    """Return the NA worker environment without importing PipeTools."""

    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        environment.pop(key, None)
    environment["PYTHONPATH"] = _release_pythonpath(
        community_source("na_mpnn")
    )
    environment["CUDA_VISIBLE_DEVICES"] = _resolve_na_owner_cuda_visible_device(
        environment
    )
    return environment


def _batched_na_mpnn_sequences(
    items: Sequence[tuple[SearchCandidate, SequenceSourcePlan]],
    *,
    reward_config: Mapping[str, Any],
    repo_root: Path,
    work_dir: Path,
) -> tuple[dict[str, SequenceSourcePlan], dict[str, str]]:
    """Run one vendored NA-MPNN worker process for one owner batch."""

    if not items:
        return {}, {}
    backend = reward_config["sequence"]["na_mpnn"]
    if backend.get("model_type") != "na_mpnn":
        raise MPNNSystemError("na_mpnn model_type must be exactly 'na_mpnn'")
    omission = resolve_rm_aa(backend)
    try:
        configured_path = Path(str(backend["checkpoint"])).expanduser()
        if not configured_path.is_absolute():
            configured_path = repo_root / configured_path
        configured_checkpoint = configured_path.resolve(strict=False)
        expected_checkpoint = (
            community_asset("na_mpnn", repo_root) / "checkpoint" / "s_19137.pt"
        ).resolve()
    except (KeyError, OSError, RuntimeError) as exc:
        raise MPNNSystemError("NA-MPNN checkpoint path is invalid") from exc
    if configured_checkpoint != expected_checkpoint:
        raise MPNNSystemError(
            "NA-MPNN checkpoint must resolve to the vendored pinned asset"
        )
    try:
        temperature = float(backend["temperature"])
        seed = int(backend["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MPNNSystemError("NA-MPNN backend temperature/seed is invalid") from exc
    if not np.isfinite(temperature) or not 0.0 < temperature <= 10.0:
        raise MPNNSystemError("NA-MPNN temperature must be finite and in (0, 10]")
    if isinstance(backend["seed"], bool) or seed < 0 or seed != backend["seed"]:
        raise MPNNSystemError("NA-MPNN seed must be a non-negative integer")

    work_dir.mkdir(parents=True, exist_ok=False)
    requests: list[dict[str, Any]] = []
    plan_by_id: dict[str, tuple[SequenceSourcePlan, str]] = {}
    failures: dict[str, str] = {}
    for candidate_index, (candidate, plan) in enumerate(items):
        try:
            candidate_id = str(candidate.candidate_id)
            safe = f"candidate_{candidate_index}"
            candidate_root = work_dir / "prepared" / safe
            candidate_root.mkdir(parents=True)
            _aliases, design_alias = _na_private_pdb(
                candidate,
                plan,
                candidate_root / "candidate.pdb",
            )
            design_rows = np.flatnonzero(
                np.asarray(candidate.atoms.chain_id).astype(str) == plan.design_chain
            )
            modalities = {
                str(value) for value in np.asarray(candidate.atoms.mol_type)[design_rows]
            }
            if modalities not in ({"dna"}, {"rna"}):
                raise MPNNSequenceError(
                    "NA-MPNN design chain must be exactly one DNA or RNA modality"
                )
            modality = next(iter(modalities))
            sequence = plan.chain_sequences.get(plan.design_chain)
            if sequence is None:
                raise MPNNSequenceError("NA-MPNN plan lacks its design-chain sequence")
            length = len(sequence)
            fixed = sorted(
                int(position)
                for chain, position in plan.fixed_positions
                if chain == plan.design_chain
            )
            if any(position < 1 or position > length for position in fixed):
                raise MPNNSequenceError("NA-MPNN fixed position is outside design chain")
            fixed_set = set(fixed)
            designable = [position for position in range(1, length + 1) if position not in fixed_set]
            if not designable:
                raise MPNNSequenceError("NA-MPNN has no designable positions")
            requests.append(
                {
                    "candidate_id": candidate_id,
                    "pdb_path": str(candidate_root / "candidate.pdb"),
                    "design_chain": design_alias,
                    "designable_positions": designable,
                    "fixed_positions": fixed,
                    "seed": seed,
                    "temperature": temperature,
                }
            )
            plan_by_id[candidate_id] = (plan, modality)
        except MPNNSequenceError as exc:
            failures[str(candidate.candidate_id)] = str(exc)

    if not requests:
        return {}, failures
    request_path = work_dir / "request.json"
    result_path = work_dir / "result.json"
    request_path.write_text(
        json.dumps(
            {"schema_version": 1, "rm_aa": omission, "requests": requests},
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "src.design.rewards.na_mpnn_worker",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    cache_stats = _run_cached_mpnn(
        command, backend='na_mpnn', source='na_mpnn', cwd=repo_root,
        env=_na_offline_environment(repo_root), work_dir=work_dir,
    )

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MPNNSystemError("NA-MPNN worker result is unreadable") from exc
    if (not isinstance(payload, Mapping) or not {"schema_version", "model_loads", "results"}.issubset(payload)
            or set(payload) - {"schema_version", "model_loads", "results", "model_load_count", "cache_hit"}):
        raise MPNNSystemError("NA-MPNN worker result protocol is invalid")
    if (payload.get("schema_version") != 1 or type(payload.get("model_loads")) is not int
            or payload["model_loads"] != cache_stats["model_loads_this_request"]):
        raise MPNNSystemError("NA-MPNN worker result protocol is invalid")
    values = payload.get("results")
    if not isinstance(values, list) or len(values) != len(requests):
        raise MPNNSystemError("NA-MPNN worker result count differs from requests")

    result: dict[str, SequenceSourcePlan] = {}
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping) or set(value) - {
            "candidate_id", "status", "sequence", "error"
        }:
            raise MPNNSystemError("NA-MPNN worker returned an invalid candidate result")
        candidate_id = value.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in seen or candidate_id not in plan_by_id:
            raise MPNNSystemError("NA-MPNN candidate accounting is invalid")
        seen.add(candidate_id)
        plan, modality = plan_by_id[candidate_id]
        if value.get("status") == "failed":
            failures[candidate_id] = str(value.get("error") or "NA-MPNN candidate failed")
            continue
        if value.get("status") != "ok" or not isinstance(value.get("sequence"), str):
            raise MPNNSystemError("NA-MPNN worker returned an invalid candidate status")
        try:
            result[candidate_id] = _validate_na_sequence(
                plan,
                value["sequence"],
                modality=modality,
            )
        except MPNNSequenceError as exc:
            failures[candidate_id] = str(exc)
    if seen != set(plan_by_id):
        raise MPNNSystemError("NA-MPNN worker omitted a candidate result")
    return result, failures


def _native_config(source: str, config: Mapping[str, Any], SequenceConfig: Any) -> Any:
    options: dict[str, Any] = {
        "batch_size": 1,
        "seed": 101,
    }
    omission = resolve_rm_aa(config)
    if source == "proteinmpnn":
        options["backbone_noise"] = float(config["backbone_noise"])
        options["sampling_temp"] = str(config["temperature"])
        if omission:
            options["omit_aas"] = omission
        if config.get("bias_aa"):
            options["bias_aa"] = dict(config["bias_aa"])
        if config.get("pssm_path"):
            options["pssm"] = _structured_asset(
                config["pssm_path"], expected=dict, where="ProteinMPNN PSSM"
            )
            options["pssm_multi"] = float(config["pssm_multi"])
            options["pssm_threshold"] = float(config["pssm_threshold"])
            options["pssm_log_odds_flag"] = bool(config["pssm_log_odds"])
            options["pssm_bias_flag"] = bool(config["pssm_bias"])
        if config.get("tied_positions_path"):
            options["tied_positions"] = _structured_asset(
                config["tied_positions_path"],
                expected=list,
                where="ProteinMPNN tied positions",
            )
        options.update(
            {
                "model_name": Path(str(config["checkpoint"])).stem,
                "use_soluble_model": bool(config["use_soluble_model"]),
                "ca_only": bool(config["ca_only"]),
            }
        )
        return SequenceConfig(model="proteinmpnn", count=1, options=options)
    options.update(
        {
            "temperature": float(config["temperature"]),
            "model_type": str(config["model_type"]),
            "model_name": Path(str(config["checkpoint"])).name,
            "ligand_mpnn_use_atom_context": bool(config["use_ligand_atoms"]),
            "ligand_mpnn_use_side_chain_context": bool(config["use_sidechain_context"]),
        }
    )
    if config["pack_side_chains"]:
        options["pack_side_chains"] = True
    if config["repack_everything"]:
        options["repack_everything"] = True
    if omission:
        options["omit_aa"] = omission
    if config.get("bias_aa"):
        options["bias_aa"] = dict(config["bias_aa"])
    if config.get("symmetry_residues_path"):
        options["symmetry_residues"] = _structured_asset(
            config["symmetry_residues_path"],
            expected=list,
            where="LigandMPNN symmetry residues",
        )
    if config.get("symmetry_weights_path"):
        options["symmetry_weights"] = _structured_asset(
            config["symmetry_weights_path"],
            expected=list,
            where="LigandMPNN symmetry weights",
        )
    return SequenceConfig(model="ligandmpnn", count=1, options=options)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _read_fasta_records(path: Path, *, where: str) -> list[str]:
    try:
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    except OSError as exc:
        raise MPNNSystemError(f"{where} output FASTA is unavailable") from exc
    records: list[str] = []
    sequence: list[str] = []
    seen_header = False
    for line in lines:
        if line.startswith(">"):
            if seen_header:
                records.append("".join(sequence))
            seen_header = True
            sequence = []
        elif not seen_header:
            raise MPNNSystemError(f"{where} output FASTA is malformed")
        else:
            sequence.append(line)
    if seen_header:
        records.append("".join(sequence))
    if len(records) < 2 or any(not value for value in records):
        raise MPNNSystemError(
            f"{where} must emit one native and at least one designed sequence"
        )
    return records


def _read_design_sequences(
    path: Path,
    design_segment: int,
    *,
    separator: str,
    where: str,
) -> list[str]:
    records = _read_fasta_records(path, where=where)
    sequences: list[str] = []
    for record in records[1:]:
        segments = record.split(separator)
        if not 0 <= design_segment < len(segments):
            raise MPNNSystemError(f"{where} output lacks the design chain")
        sequences.append(segments[design_segment])
    return sequences


def _read_single_design_sequence(path: Path, design_segment: int) -> str:
    sequences = _read_design_sequences(
        path, design_segment, separator=":", where="batched MPNN"
    )
    if len(sequences) != 1:
        raise MPNNSystemError("batched MPNN emitted more than one designed sequence")
    return sequences[0]


def _list_option(value: Any, *, where: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise MPNNSystemError(f"{where} must contain a list")
    return value


def _symmetry_cli(values: list[Any], *, where: str) -> str:
    groups: list[str] = []
    for value in values:
        if not isinstance(value, list) or not value:
            raise MPNNSystemError(f"{where} groups must be non-empty lists")
        groups.append(",".join(str(item) for item in value))
    return "|".join(groups)


def _ligand_suite_command(
    *,
    repo_root: Path,
    work_dir: Path,
    paths_json: Path,
    fixed_json: Path,
    redesigned_json: Path,
    output: Path,
    backend: Mapping[str, Any],
    symmetry_residues_cli: str | None,
    symmetry_weights_cli: str | None,
    omit_path: Path | None = None,
    batch_size: int = 1,
    chains_to_design: Sequence[str] | None = None,
) -> list[str]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise MPNNSystemError("LigandMPNN batch_size must be a positive integer")
    model_type = str(backend["model_type"])
    checkpoint = Path(str(backend["checkpoint"])).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path(repo_root) / checkpoint
    checkpoint = checkpoint.resolve()
    omission = resolve_rm_aa(backend)
    if _has_explicit_rm_aa(backend):
        omission = ""
    command = [
        sys.executable,
        str(community_source("ligandmpnn") / "run.py"),
        "--model_type", model_type,
        "--pdb_path_multi", str(paths_json),
        "--fixed_residues_multi", str(fixed_json),
        "--redesigned_residues_multi", str(redesigned_json),
        "--out_folder", str(output),
        "--seed", str(int(backend.get("seed", 101))),
        "--batch_size", str(batch_size),
        "--number_of_batches", "1",
        "--temperature", str(float(backend["temperature"])),
        "--verbose", "0",
        "--omit_AA", omission,
        _ligand_checkpoint_flag(model_type), str(checkpoint),
        "--ligand_mpnn_use_atom_context",
        str(int(bool(backend["use_ligand_atoms"]))),
        "--ligand_mpnn_use_side_chain_context",
        str(int(bool(backend["use_sidechain_context"]))),
        "--pack_side_chains", str(int(bool(backend["pack_side_chains"]))),
        "--repack_everything", str(int(bool(backend["repack_everything"]))),
        "--homo_oligomer", str(int(bool(backend["homo_oligomer"]))),
    ]
    if chains_to_design is not None:
        values = [str(value) for value in chains_to_design]
        if not values or len(values) != len(set(values)):
            raise MPNNSystemError(
                "LigandMPNN chains_to_design must be non-empty and unique"
            )
        command.extend(["--chains_to_design", ",".join(values)])
    bias = backend.get("bias_aa") or {}
    if bias:
        command.extend(
            [
                "--bias_AA",
                ",".join(
                    f"{amino_acid}:{float(value)}"
                    for amino_acid, value in sorted(bias.items())
                ),
            ]
        )
    if backend["pack_side_chains"]:
        sidechain_checkpoint = (
            community_asset("ligandmpnn", repo_root) / "checkpoint"
            / "ligandmpnn_sc_v_32_002_16.pt"
        )
        if (
            not sidechain_checkpoint.is_file()
            or sidechain_checkpoint.is_symlink()
            or sidechain_checkpoint.stat().st_size == 0
        ):
            raise MPNNSystemError(
                f"LigandMPNN side-chain checkpoint is missing or unsafe: "
                f"{sidechain_checkpoint}"
            )
        command.extend(
            [
                "--checkpoint_path_sc",
                str(sidechain_checkpoint),
                "--number_of_packs_per_design", "1",
            ]
        )
    if symmetry_residues_cli is not None:
        if symmetry_weights_cli is None:
            raise MPNNSystemError(
                "LigandMPNN symmetry residues require resolved symmetry weights"
            )
        command.extend(
            [
                "--symmetry_residues", symmetry_residues_cli,
                "--symmetry_weights", symmetry_weights_cli,
            ]
        )
    elif symmetry_weights_cli is not None:
        raise MPNNSystemError(
            "LigandMPNN symmetry weights require resolved symmetry residues"
        )
    if omit_path is not None:
        command.extend(["--omit_AA_per_residue_multi", str(omit_path)])
    return command


def _resolved_ligand_symmetry(
    raw_residues: list[Any],
    raw_weights: list[Any],
    prepared: Any,
    plan: SequenceSourcePlan,
) -> tuple[str | None, str | None]:
    if not raw_residues:
        if raw_weights:
            raise MPNNSystemError(
                "LigandMPNN symmetry weights require symmetry residues"
            )
        return None, None
    if plan.design_chain is None:
        raise MPNNSequenceError("LigandMPNN symmetry lacks a design chain")
    if raw_weights and len(raw_weights) != len(raw_residues):
        raise MPNNSystemError(
            "LigandMPNN symmetry weights must match every residue group"
        )
    residues_by_position = {
        int(residue.canonical_position): residue
        for residue in prepared.residues_by_chain[plan.design_chain]
    }
    alias = prepared.alias_by_chain[plan.design_chain]
    resolved_residues: list[list[str]] = []
    resolved_weights: list[list[float]] = []
    used: set[int] = set()
    for group_index, raw_group in enumerate(raw_residues):
        if not isinstance(raw_group, list) or len(raw_group) < 2:
            raise MPNNSystemError(
                "LigandMPNN symmetry groups must contain at least two residues"
            )
        group: list[str] = []
        for token in raw_group:
            text = str(token)
            prefix = f"{plan.design_chain}:"
            if not text.startswith(prefix):
                raise MPNNSystemError(
                    "LigandMPNN symmetry may target only the design chain"
                )
            try:
                position = int(text[len(prefix) :])
                residue = residues_by_position[position]
            except (ValueError, KeyError) as exc:
                raise MPNNSystemError(
                    f"LigandMPNN symmetry residue {text!r} did not resolve"
                ) from exc
            if position in used:
                raise MPNNSystemError(
                    f"LigandMPNN symmetry residue {text!r} is repeated"
                )
            used.add(position)
            group.append(f"{alias}{int(residue.temporary_position)}")
        resolved_residues.append(group)
        raw_group_weights = (
            raw_weights[group_index]
            if raw_weights
            else [1.0 / len(group)] * len(group)
        )
        if not isinstance(raw_group_weights, list) or len(raw_group_weights) != len(group):
            raise MPNNSystemError(
                "LigandMPNN symmetry weights must match every residue group"
            )
        values = np.asarray(raw_group_weights, dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0.0) or not np.isclose(
            values.sum(), 1.0, rtol=1e-6, atol=1e-6
        ):
            raise MPNNSystemError(
                "LigandMPNN symmetry weights must be finite, non-negative, and sum to 1"
            )
        resolved_weights.append([float(value) for value in values])
    return (
        _symmetry_cli(resolved_residues, where="LigandMPNN symmetry residues"),
        _symmetry_cli(resolved_weights, where="LigandMPNN symmetry weights"),
    )


def _proteinmpnn_command(
    *,
    repo_root: Path,
    backend: Mapping[str, Any],
    jsonl_path: Path,
    chain_path: Path,
    fixed_path: Path,
    output: Path,
    bias_path: Path | None,
    pssm_path: Path | None,
    tied_path: Path | None,
    omit_path: Path | None = None,
    batch_size: int = 1,
) -> list[str]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise MPNNSystemError("ProteinMPNN batch_size must be a positive integer")
    _family, ca_only, use_soluble = _protein_model_contract(backend)
    checkpoint = Path(str(backend["checkpoint"])).expanduser().absolute()
    omission = resolve_rm_aa(backend)
    if _has_explicit_rm_aa(backend):
        omission = ""
    command = [
        sys.executable,
        str(community_source("proteinmpnn") / "protein_mpnn_run.py"),
        "--jsonl_path", str(jsonl_path),
        "--chain_id_jsonl", str(chain_path),
        "--fixed_positions_jsonl", str(fixed_path),
        "--out_folder", str(output),
        "--path_to_model_weights", str(checkpoint.parent),
        "--model_name", checkpoint.stem,
        "--num_seq_per_target", str(batch_size),
        "--batch_size", str(batch_size),
        "--sampling_temp", str(float(backend["temperature"])),
        "--backbone_noise", str(float(backend["backbone_noise"])),
        "--seed", str(int(backend.get("seed", 101))),
        "--suppress_print", "1",
        "--omit_AAs", omission + "X",
    ]
    if ca_only:
        command.append("--ca_only")
    if use_soluble:
        command.append("--use_soluble_model")
    if bias_path is not None:
        command.extend(["--bias_AA_jsonl", str(bias_path)])
    if pssm_path is not None:
        command.extend(
            [
                "--pssm_jsonl", str(pssm_path),
                "--pssm_multi", str(float(backend["pssm_multi"])),
                "--pssm_threshold", str(float(backend["pssm_threshold"])),
                "--pssm_log_odds_flag", str(int(bool(backend["pssm_log_odds"]))),
                "--pssm_bias_flag", str(int(bool(backend["pssm_bias"]))),
            ]
        )
    if tied_path is not None:
        command.extend(["--tied_positions_jsonl", str(tied_path)])
    if omit_path is not None:
        command.extend(["--omit_AA_jsonl", str(omit_path)])
    return command


def _proteinmpnn_constraint_payloads(
    sample: Any,
    prepared: Any,
    plan: SequenceSourcePlan,
    backend: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Map public canonical selectors into official ProteinMPNN JSON payloads."""

    if plan.design_chain is None:
        raise MPNNSequenceError("ProteinMPNN plan lacks its design chain")
    design_alias = prepared.alias_by_chain[plan.design_chain]
    name = prepared.pdb_path.stem
    fixed_positions = sorted(
        int(position)
        for chain, position in plan.fixed_positions
        if chain == plan.design_chain
    )
    fixed = {name: {design_alias: fixed_positions}}
    pssm: dict[str, Any] | None = None
    pssm_path = backend.get("pssm_path")
    if pssm_path:
        raw = _structured_asset(pssm_path, expected=dict, where="ProteinMPNN PSSM")
        chain_value = raw.get(plan.design_chain)
        if chain_value is None:
            chain_value = raw.get(design_alias)
        if chain_value is None or len(raw) != 1 or not isinstance(chain_value, Mapping):
            raise MPNNSystemError(
                "ProteinMPNN PSSM must contain exactly the canonical design chain"
            )
        length = len(prepared.residues_by_chain[plan.design_chain])
        alphabet = str(chain_value.get("alphabet", "ACDEFGHIKLMNPQRSTVWYX"))
        canonical_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        if len(alphabet) != len(canonical_alphabet) or set(alphabet) != set(canonical_alphabet):
            raise MPNNSystemError(
                "ProteinMPNN PSSM alphabet must be a permutation of "
                f"{canonical_alphabet}"
            )
        order = [alphabet.index(amino_acid) for amino_acid in canonical_alphabet]

        def vector(field: str, default: float) -> np.ndarray:
            value = np.asarray(chain_value.get(field, [default] * length), dtype=float)
            if value.shape != (length,):
                raise MPNNSystemError(
                    f"ProteinMPNN PSSM {field} must have shape ({length},)"
                )
            return value

        def matrix(field: str, default: float) -> np.ndarray:
            value = np.asarray(
                chain_value.get(field, [[default] * len(order) for _ in range(length)]),
                dtype=float,
            )
            if value.shape != (length, len(order)):
                raise MPNNSystemError(
                    f"ProteinMPNN PSSM {field} must have shape "
                    f"({length}, {len(order)})"
                )
            return value[:, order]

        coef = vector("pssm_coef", 1.0)
        bias = matrix("pssm_bias", 0.0)
        log_odds = matrix("pssm_log_odds", 10000.0)
        if not (np.isfinite(coef).all() and np.isfinite(bias).all() and np.isfinite(log_odds).all()):
            raise MPNNSystemError("ProteinMPNN PSSM contains non-finite values")
        if np.any((coef < 0.0) | (coef > 1.0)):
            raise MPNNSystemError("ProteinMPNN PSSM coefficients must be in [0, 1]")
        if backend["pssm_bias"] and (
            np.any(bias < 0.0)
            or not np.allclose(bias.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
        ):
            raise MPNNSystemError(
                "ProteinMPNN PSSM bias rows must be probability distributions"
            )
        if backend["pssm_log_odds"] and "pssm_log_odds" not in chain_value:
            raise MPNNSystemError(
                "ProteinMPNN PSSM requires pssm_log_odds values when the flag is enabled"
            )
        pssm = {
            name: {
                design_alias: {
                    "pssm_coef": coef.tolist(),
                    "pssm_bias": bias.tolist(),
                    "pssm_log_odds": log_odds.tolist(),
                }
            }
        }

    tied: dict[str, Any] | None = None
    tied_path = backend.get("tied_positions_path")
    if tied_path:
        raw_groups = _structured_asset(
            tied_path, expected=list, where="ProteinMPNN tied positions"
        )
        groups: list[dict[str, list[int]]] = []
        used: set[int] = set()
        for raw_group in raw_groups:
            if not isinstance(raw_group, list) or len(raw_group) < 2:
                raise MPNNSystemError(
                    "ProteinMPNN tied-position groups must contain at least two residues"
                )
            positions: list[int] = []
            for token in raw_group:
                text = str(token)
                prefix = f"{plan.design_chain}:"
                if not text.startswith(prefix):
                    raise MPNNSystemError(
                        "ProteinMPNN tied positions may target only the design chain"
                    )
                try:
                    position = int(text[len(prefix) :])
                except ValueError as exc:
                    raise MPNNSystemError(
                        f"invalid ProteinMPNN tied position {text!r}"
                    ) from exc
                if not 1 <= position <= len(prepared.residues_by_chain[plan.design_chain]):
                    raise MPNNSystemError(
                        f"ProteinMPNN tied position {text!r} is out of range"
                    )
                if position in fixed_positions or position in used:
                    raise MPNNSystemError(
                        f"ProteinMPNN tied position {text!r} is fixed or repeated"
                    )
                positions.append(position)
                used.add(position)
            groups.append({design_alias: positions})
        tied = {name: groups}
    return fixed, pssm, tied


def _read_proteinmpnn_design_sequences(path: Path) -> list[str]:
    records = _read_fasta_records(path, where="ProteinMPNN")
    return [record.split("/")[0] for record in records[1:]]


def _read_proteinmpnn_design_sequence(path: Path) -> str:
    sequences = _read_proteinmpnn_design_sequences(path)
    if len(sequences) != 1:
        raise MPNNSystemError("ProteinMPNN emitted more than one designed sequence")
    return sequences[0]


def _read_proteinmpnn_chain_sequences(path: Path) -> list[list[str]]:
    """Read every designed chain segment from one native ProteinMPNN FASTA."""
    records = _read_fasta_records(path, where="ProteinMPNN")
    result: list[list[str]] = []
    for record in records[1:]:
        segments = [segment.strip().upper() for segment in record.split("/")]
        if not segments or any(not segment for segment in segments):
            raise MPNNSystemError("ProteinMPNN emitted an empty designed chain segment")
        result.append(segments)
    return result


def _antibody_chain_sequences_from_segments(
    sequence_plan: SequenceSourcePlan,
    selection: _ValidatedAntibodyMPNNPlan,
    prepared: Any,
    segments: Sequence[str],
) -> dict[str, str]:
    design_aliases = sorted(
        prepared.alias_by_chain[chain]
        for chain in selection.plan.design_chain_ids
    )
    if len(segments) != len(design_aliases):
        raise MPNNSystemError(
            "ProteinMPNN emitted a different number of antibody design chains"
        )
    chain_sequences = dict(sequence_plan.chain_sequences)
    for alias, sequence in zip(design_aliases, segments, strict=True):
        chain = prepared.chain_by_alias[alias]
        chain_sequences[chain] = sequence
    return chain_sequences


def _batched_antibody_proteinmpnn_sequences(
    items: Sequence[tuple[SearchCandidate, SequenceSourcePlan]],
    *,
    antibody_plans: Mapping[str, AntibodyMPNNPlan],
    reward_config: Mapping[str, Any],
    repo_root: Path,
    work_dir: Path,
) -> tuple[dict[str, SequenceSourcePlan], dict[str, str]]:
    """Run native ProteinMPNN with multiple physical antibody chains."""

    if not items:
        return {}, {}
    failures: dict[str, str] = {}
    result: dict[str, SequenceSourcePlan] = {}
    active_items: list[tuple[SearchCandidate, SequenceSourcePlan]] = []
    for candidate, sequence_plan in items:
        try:
            selection = _validate_antibody_mpnn_plan(
                candidate,
                antibody_plans[candidate.candidate_id],
                sequence_plan=sequence_plan,
            )
            if selection.plan.mutable_residue_uids:
                active_items.append((candidate, sequence_plan))
            else:
                result[candidate.candidate_id] = sequence_plan
        except MPNNSequenceError as exc:
            failures[candidate.candidate_id] = str(exc)
    if not active_items:
        return result, failures
    items = active_items
    backend = reward_config["sequence"]["proteinmpnn"]
    _backend_assets(backend, "proteinmpnn")
    _family, ca_only, _use_soluble = _protein_model_contract(backend)
    work_dir.mkdir(parents=True, exist_ok=False)
    api, _SequenceConfig, model_types, _RuntimePaths, _ = _load_pipetools(repo_root)
    prepare_mpnn_input = api[0]
    records: list[dict[str, Any]] = []
    parser = community_source("proteinmpnn") / "protein_mpnn_utils.py"
    if str(parser.parent) not in sys.path:
        sys.path.insert(0, str(parser.parent))
    from protein_mpnn_utils import parse_PDB  # type: ignore[import-not-found]

    for candidate_index, (candidate, sequence_plan) in enumerate(items):
        candidate_id = candidate.candidate_id
        try:
            antibody = antibody_plans[candidate_id]
            selection = _validate_antibody_mpnn_plan(
                candidate, antibody, sequence_plan=sequence_plan
            )
            if not antibody.mutable_residue_uids:
                result[candidate_id] = sequence_plan
                continue
            compatibility = _antibody_compatibility_plan(sequence_plan, selection)
            safe = f"candidate_{candidate_index}"
            candidate_root = work_dir / "prepared" / safe
            candidate_root.mkdir(parents=True)
            sample = _canonical_sample(
                candidate,
                compatibility,
                candidate_root,
                model_types,
                design_chain_ids=selection.plan.design_chain_ids,
            )
            prepared = prepare_mpnn_input(
                sample,
                candidate_root / "native-input",
                ca_only=ca_only,
                materialize=True,
            )
            parsed = parse_PDB(str(prepared.pdb_path), ca_only=ca_only)
            if not isinstance(parsed, list) or len(parsed) != 1:
                raise MPNNSystemError("ProteinMPNN parser did not return one structure")
            parsed_row = parsed[0]
            parsed_row["name"] = safe
            design_aliases, fixed_aliases, fixed = _antibody_fixed_payload(
                prepared, sample, selection
            )
            omit = _antibody_omit_payload(
                selection,
                prepared,
                name=safe,
                ligand=False,
                backend=backend,
            )
            records.append(
                {
                    "candidate": candidate,
                    "plan": sequence_plan,
                    "selection": selection,
                    "prepared": prepared,
                    "safe": safe,
                    "pdb_bytes": prepared.pdb_path.read_bytes(),
                    "parsed": parsed_row,
                    "chain_ids": (design_aliases, fixed_aliases),
                    "fixed": fixed,
                    # Target names are serialization keys, not constraints.
                    "omit": None if omit is None else omit[safe],
                }
            )
        except (MPNNSequenceError, ValueError) as exc:
            failures[candidate_id] = str(exc)

    if not records:
        return result, failures
    groups: list[list[dict[str, Any]]] = []
    for record in records:
        for group in groups:
            first = group[0]
            if (
                record["pdb_bytes"] == first["pdb_bytes"]
                and record["chain_ids"] == first["chain_ids"]
                and record["fixed"] == first["fixed"]
                and record["omit"] == first["omit"]
            ):
                group.append(record)
                break
        else:
            groups.append([record])

    for batch_index, group in enumerate(groups):
        batch_dir = work_dir / f"batch_{batch_index}"
        batch_dir.mkdir()
        first = group[0]
        name = first["safe"]
        jsonl_path = batch_dir / "parsed.jsonl"
        jsonl_path.write_text(
            json.dumps(first["parsed"], sort_keys=True, allow_nan=True) + "\n"
        )

        def jsonl(name: str, payload: Any) -> Path:
            path = batch_dir / name
            path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            return path

        chain_path = jsonl("chain_id.jsonl", {name: first["chain_ids"]})
        fixed_path = jsonl("fixed_positions.jsonl", {name: first["fixed"]})
        omit_payload = None if first["omit"] is None else {name: first["omit"]}
        output = batch_dir / "output"
        command = _proteinmpnn_command(
            repo_root=repo_root,
            backend=backend,
            jsonl_path=jsonl_path,
            chain_path=chain_path,
            fixed_path=fixed_path,
            output=output,
            bias_path=None,
            pssm_path=None,
            tied_path=None,
            omit_path=(
                jsonl("omit_AA.jsonl", omit_payload)
                if omit_payload
                else None
            ),
            batch_size=len(group),
        )
        _run_cached_mpnn(
            command,
            backend="proteinmpnn",
            source="proteinmpnn",
            cwd=community_source("proteinmpnn"),
            env=_offline_environment(repo_root),
            work_dir=batch_dir,
        )
        sequences = _read_proteinmpnn_chain_sequences(
            output / "seqs" / f"{name}.fa"
        )
        if len(sequences) != len(group):
            raise MPNNSystemError(
                "ProteinMPNN returned a different number of sequences than its native batch"
            )
        for record, segments in zip(group, sequences, strict=True):
            candidate_id = record["candidate"].candidate_id
            try:
                chain_sequences = _antibody_chain_sequences_from_segments(
                    record["plan"],
                    record["selection"],
                    record["prepared"],
                    segments,
                )
                result[candidate_id] = _validate_antibody_sequences(
                    record["plan"], record["selection"], chain_sequences
                )
            except MPNNSequenceError as exc:
                failures[candidate_id] = str(exc)
    return result, failures


def _read_ligandmpnn_chain_sequences(path: Path) -> list[list[str]]:
    records = _read_fasta_records(path, where="LigandMPNN")
    return [
        [segment.strip().upper() for segment in record.split(":")]
        for record in records[1:]
    ]


def _batched_antibody_ligandmpnn_sequences(
    items: Sequence[tuple[SearchCandidate, SequenceSourcePlan]],
    *,
    antibody_plans: Mapping[str, AntibodyMPNNPlan],
    reward_config: Mapping[str, Any],
    repo_root: Path,
    work_dir: Path,
) -> tuple[dict[str, SequenceSourcePlan], dict[str, str]]:
    """Run native LigandMPNN over H/L while retaining antigen atom context."""

    if not items:
        return {}, {}
    failures: dict[str, str] = {}
    result: dict[str, SequenceSourcePlan] = {}
    active_items: list[tuple[SearchCandidate, SequenceSourcePlan]] = []
    for candidate, sequence_plan in items:
        try:
            selection = _validate_antibody_mpnn_plan(
                candidate,
                antibody_plans[candidate.candidate_id],
                sequence_plan=sequence_plan,
            )
            if selection.plan.mutable_residue_uids:
                active_items.append((candidate, sequence_plan))
            else:
                result[candidate.candidate_id] = sequence_plan
        except MPNNSequenceError as exc:
            failures[candidate.candidate_id] = str(exc)
    if not active_items:
        return result, failures
    items = active_items
    backend = reward_config["sequence"]["ligandmpnn"]
    _backend_assets(backend, "ligandmpnn")
    if not bool(backend.get("use_ligand_atoms", False)):
        raise MPNNSystemError(
            "antibody LigandMPNN requires use_ligand_atoms=true for antigen context"
        )
    work_dir.mkdir(parents=True, exist_ok=False)
    api, _SequenceConfig, model_types, _RuntimePaths, _ = _load_pipetools(repo_root)
    prepare_mpnn_input = api[0]
    input_dir = work_dir / "inputs"
    input_dir.mkdir()
    records: list[dict[str, Any]] = []
    for candidate_index, (candidate, sequence_plan) in enumerate(items):
        candidate_id = candidate.candidate_id
        try:
            antibody = antibody_plans[candidate_id]
            selection = _validate_antibody_mpnn_plan(
                candidate, antibody, sequence_plan=sequence_plan
            )
            if not antibody.mutable_residue_uids:
                result[candidate_id] = sequence_plan
                continue
            compatibility = _antibody_compatibility_plan(sequence_plan, selection)
            safe = f"candidate_{candidate_index}"
            candidate_root = work_dir / "prepared" / safe
            candidate_root.mkdir(parents=True)
            sample = _canonical_sample(
                candidate,
                compatibility,
                candidate_root,
                model_types,
                design_chain_ids=selection.plan.design_chain_ids,
            )
            prepared = prepare_mpnn_input(
                sample,
                candidate_root / "native-input",
                ca_only=False,
                materialize=True,
            )
            protein_aliases = sorted(
                prepared.alias_by_chain[record.chain_id]
                for record in sample.chains
                if record.modality == "protein"
            )
            design_aliases = sorted(
                prepared.alias_by_chain[chain]
                for chain in selection.plan.design_chain_ids
            )
            uid_by_chain_position = {
                (str(token.chain_id), int(token.position)): int(token.residue_uid)
                for token in selection.tokens_by_uid.values()
            }
            mutable = selection.plan.mutable_uid_set
            fixed_tokens: list[str] = []
            redesigned_tokens: list[str] = []
            for chain, residues in prepared.residues_by_chain.items():
                alias = prepared.alias_by_chain[chain]
                if alias not in protein_aliases:
                    continue
                for residue in residues:
                    uid = uid_by_chain_position.get(
                        (chain, int(residue.canonical_position))
                    )
                    token = f"{alias}{int(residue.temporary_position)}"
                    if uid in mutable:
                        redesigned_tokens.append(token)
                    else:
                        fixed_tokens.append(token)
            omit = _antibody_omit_payload(
                selection,
                prepared,
                name=safe,
                ligand=True,
                backend=backend,
            )
            pdb_path = input_dir / f"{safe}.pdb"
            shutil.copyfile(prepared.pdb_path, pdb_path)
            records.append(
                {
                    "candidate": candidate,
                    "plan": sequence_plan,
                    "selection": selection,
                    "prepared": prepared,
                    "safe": safe,
                    "pdb_path": pdb_path,
                    "pdb_bytes": pdb_path.read_bytes(),
                    "chains_to_design": design_aliases,
                    "fixed": " ".join(sorted(fixed_tokens)),
                    "redesigned": " ".join(sorted(redesigned_tokens)),
                    "protein_aliases": protein_aliases,
                    "omit": omit,
                }
            )
        except (MPNNSequenceError, ValueError) as exc:
            failures[candidate_id] = str(exc)
    if not records:
        return result, failures
    groups: list[list[dict[str, Any]]] = []
    for record in records:
        for group in groups:
            first = group[0]
            if (
                record["pdb_bytes"] == first["pdb_bytes"]
                and record["chains_to_design"] == first["chains_to_design"]
                and record["fixed"] == first["fixed"]
                and record["redesigned"] == first["redesigned"]
                and record["omit"] == first["omit"]
            ):
                group.append(record)
                break
        else:
            groups.append([record])

    for batch_index, group in enumerate(groups):
        batch_dir = work_dir / f"batch_{batch_index}"
        batch_dir.mkdir()
        first = group[0]
        paths_json = batch_dir / "pdb_paths.json"
        fixed_json = batch_dir / "fixed_residues.json"
        redesigned_json = batch_dir / "redesigned_residues.json"
        _write_json(paths_json, {str(first["pdb_path"]): ""})
        _write_json(fixed_json, {str(first["pdb_path"]): first["fixed"]})
        _write_json(redesigned_json, {str(first["pdb_path"]): first["redesigned"]})
        omit_payload = (
            {str(first["pdb_path"]): first["omit"]}
            if first["omit"] is not None
            else None
        )
        omit_path = None
        if omit_payload is not None:
            omit_path = batch_dir / "omit_AA_per_residue_multi.json"
            _write_json(omit_path, omit_payload)
        output = batch_dir / "output"
        command = _ligand_suite_command(
            repo_root=repo_root,
            work_dir=batch_dir,
            paths_json=paths_json,
            fixed_json=fixed_json,
            redesigned_json=redesigned_json,
            output=output,
            backend=backend,
            symmetry_residues_cli=None,
            symmetry_weights_cli=None,
            chains_to_design=first["chains_to_design"],
            omit_path=omit_path,
            batch_size=len(group),
        )
        _run_cached_mpnn(
            command,
            backend="ligandmpnn",
            source="ligandmpnn",
            cwd=community_source("ligandmpnn"),
            env=_offline_environment(repo_root),
            work_dir=batch_dir,
        )
        sequences = _read_ligandmpnn_chain_sequences(
            output / "seqs" / f"{first['safe']}.fa"
        )
        if len(sequences) != len(group):
            raise MPNNSystemError(
                "LigandMPNN returned a different number of sequences than its native batch"
            )
        for record, segments in zip(group, sequences, strict=True):
            candidate_id = record["candidate"].candidate_id
            try:
                if len(segments) != len(record["protein_aliases"]):
                    raise MPNNSequenceError(
                        "LigandMPNN emitted a different number of protein chains"
                    )
                chain_sequences = dict(record["plan"].chain_sequences)
                for alias, sequence in zip(record["protein_aliases"], segments, strict=True):
                    chain = record["prepared"].chain_by_alias[alias]
                    chain_sequences[chain] = sequence
                result[candidate_id] = _validate_antibody_sequences(
                    record["plan"], record["selection"], chain_sequences
                )
            except MPNNSequenceError as exc:
                failures[candidate_id] = str(exc)
    return result, failures


def _batched_proteinmpnn_sequences(
    items: Sequence[tuple[SearchCandidate, SequenceSourcePlan]],
    *,
    reward_config: Mapping[str, Any],
    repo_root: Path,
    work_dir: Path,
) -> tuple[dict[str, SequenceSourcePlan], dict[str, str]]:
    """Run ProteinMPNN in native batches while preserving candidate mapping.

    The official entry point batches draws for one target.  Candidates with
    different structures or fixed-position payloads therefore remain separate
    native targets; duplicate validation slots share one target and consume
    distinct rows from one native batch.
    """

    if not items:
        return {}, {}
    backend = reward_config["sequence"]["proteinmpnn"]
    _backend_assets(backend, "proteinmpnn")
    _family, ca_only, _use_soluble = _protein_model_contract(backend)
    work_dir.mkdir(parents=True, exist_ok=False)
    api, _SequenceConfig, model_types, _RuntimePaths, _ = _load_pipetools(repo_root)
    prepare_mpnn_input = api[0]
    failures: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    parser = community_source("proteinmpnn") / "protein_mpnn_utils.py"
    if str(parser.parent) not in sys.path:
        sys.path.insert(0, str(parser.parent))
    from protein_mpnn_utils import parse_PDB  # type: ignore[import-not-found]

    for candidate_index, (candidate, plan) in enumerate(items):
        try:
            safe = f"candidate_{candidate_index}"
            candidate_root = work_dir / "prepared" / safe
            candidate_root.mkdir(parents=True)
            sample = _canonical_sample(candidate, plan, candidate_root, model_types)
            prepared = prepare_mpnn_input(
                sample,
                candidate_root / "native-input",
                ca_only=ca_only,
                materialize=True,
            )
            parsed = parse_PDB(str(prepared.pdb_path), ca_only=ca_only)
            if not isinstance(parsed, list) or len(parsed) != 1:
                raise MPNNSystemError("ProteinMPNN parser did not return one structure")
            parsed_row = parsed[0]
            parsed_row["name"] = safe
            if plan.design_chain is None:
                raise MPNNSequenceError("ProteinMPNN plan lacks its design chain")
            design_alias = prepared.alias_by_chain[plan.design_chain]
            all_aliases = sorted(
                prepared.alias_by_chain[record.chain_id]
                for record in sample.chains
                if record.modality == "protein"
            )
            fixed, pssm, tied = _proteinmpnn_constraint_payloads(
                sample, prepared, plan, backend
            )
            omit = _protein_omit_payload(
                candidate,
                plan,
                name=safe,
                design_alias=design_alias,
                backend=backend,
                chain_aliases=all_aliases,
            )
            source_key = prepared.pdb_path.stem
            records.append(
                {
                    "candidate": candidate,
                    "plan": plan,
                    "safe": safe,
                    "pdb_bytes": prepared.pdb_path.read_bytes(),
                    "parsed": parsed_row,
                    "chain_ids": (
                        [design_alias],
                        [value for value in all_aliases if value != design_alias],
                    ),
                    "fixed": fixed[source_key],
                    "pssm": None if pssm is None else pssm[source_key],
                    "tied": None if tied is None else tied[source_key],
                    # Match omission by content just like fixed/PSSM/tied.
                    "omit": None if omit is None else omit[safe],
                }
            )
        except MPNNSequenceError as exc:
            failures[candidate.candidate_id] = str(exc)

    if not records:
        return {}, failures

    # Compare actual prepared structures and constraint payloads in memory to
    # select native batch members; nothing is persisted for this grouping.
    groups: list[list[dict[str, Any]]] = []
    for record in records:
        for group in groups:
            first = group[0]
            if (
                record["pdb_bytes"] == first["pdb_bytes"]
                and record["chain_ids"] == first["chain_ids"]
                and record["fixed"] == first["fixed"]
                and record["pssm"] == first["pssm"]
                and record["tied"] == first["tied"]
                and record["omit"] == first["omit"]
            ):
                group.append(record)
                break
        else:
            groups.append([record])

    result: dict[str, SequenceSourcePlan] = {}
    model_loads = 0
    started = time.perf_counter()
    bias = backend.get("bias_aa") or {}
    for batch_index, group in enumerate(groups):
        batch_dir = work_dir / f"batch_{batch_index}"
        batch_dir.mkdir()
        first = group[0]
        jsonl_path = batch_dir / "parsed.jsonl"
        jsonl_path.write_text(
            json.dumps(first["parsed"], sort_keys=True, allow_nan=True) + "\n"
        )

        def jsonl(name: str, payload: Any) -> Path:
            path = batch_dir / name
            path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            return path

        name = first["safe"]
        chain_path = jsonl("chain_id.jsonl", {name: first["chain_ids"]})
        fixed_path = jsonl("fixed_positions.jsonl", {name: first["fixed"]})
        pssm_payload = {} if first["pssm"] is None else {name: first["pssm"]}
        tied_payload = {} if first["tied"] is None else {name: first["tied"]}
        omit_payload = None if first["omit"] is None else {name: first["omit"]}
        output = batch_dir / "output"
        command = _proteinmpnn_command(
            repo_root=repo_root,
            backend=backend,
            jsonl_path=jsonl_path,
            chain_path=chain_path,
            fixed_path=fixed_path,
            output=output,
            bias_path=jsonl("bias_aa.jsonl", bias) if bias else None,
            pssm_path=jsonl("pssm.jsonl", pssm_payload) if pssm_payload else None,
            tied_path=jsonl("tied_positions.jsonl", tied_payload) if tied_payload else None,
            omit_path=(
                jsonl("omit_AA.jsonl", omit_payload)
                if omit_payload
                else None
            ),
            batch_size=len(group),
        )
        cache_stats = _run_cached_mpnn(
            command,
            backend="proteinmpnn",
            source="proteinmpnn",
            cwd=community_source("proteinmpnn"),
            env=_offline_environment(repo_root),
            work_dir=batch_dir,
        )
        model_loads += cache_stats["model_loads_this_request"]
        sequences = _read_proteinmpnn_design_sequences(
            output / "seqs" / f"{name}.fa"
        )
        if len(sequences) != len(group):
            raise MPNNSystemError(
                "ProteinMPNN returned a different number of sequences than its native batch"
            )
        for record, sequence in zip(group, sequences):
            candidate_id = record["candidate"].candidate_id
            try:
                result[candidate_id] = _validate_sequence(record["plan"], sequence)
            except MPNNSequenceError as exc:
                failures[candidate_id] = str(exc)

    return result, failures


def _batched_ligand_suite_sequences(
    items: Sequence[tuple[SearchCandidate, SequenceSourcePlan]],
    *,
    source: str,
    reward_config: Mapping[str, Any],
    repo_root: Path,
    work_dir: Path,
) -> tuple[dict[str, SequenceSourcePlan], dict[str, str]]:
    """Run one native batch per compatible LigandMPNN-suite target.

    The vendored LigandMPNN entry point batches draws for one PDB.  Compatible
    duplicate slots are consequently represented by one input PDB and one
    native ``batch_size``; distinct structures or residue controls get their
    own native invocation while reusing the resident backend model.
    """

    if source not in {"proteinmpnn", "ligandmpnn"}:
        raise AssertionError("batched LigandMPNN suite received the wrong source")
    if not items:
        return {}, {}
    backend = reward_config["sequence"][source]
    _backend_assets(backend, source)
    if source == "proteinmpnn":
        return _batched_proteinmpnn_sequences(
            items,
            reward_config=reward_config,
            repo_root=repo_root,
            work_dir=work_dir,
        )

    work_dir.mkdir(parents=True, exist_ok=False)
    api, _SequenceConfig, model_types, _RuntimePaths, _ = _load_pipetools(repo_root)
    prepare_mpnn_input = api[0]
    input_dir = work_dir / "inputs"
    input_dir.mkdir()
    raw_symmetry_residues = _list_option(
        _structured_asset(
            backend["symmetry_residues_path"],
            expected=list,
            where="LigandMPNN symmetry residues",
        )
        if backend.get("symmetry_residues_path")
        else [],
        where="LigandMPNN symmetry residues",
    )
    raw_symmetry_weights = _list_option(
        _structured_asset(
            backend["symmetry_weights_path"],
            expected=list,
            where="LigandMPNN symmetry weights",
        )
        if backend.get("symmetry_weights_path")
        else [],
        where="LigandMPNN symmetry weights",
    )
    symmetry_contract: tuple[str | None, str | None] | None = None
    failures: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for candidate_index, (candidate, plan) in enumerate(items):
        try:
            safe = f"candidate_{candidate_index}"
            candidate_root = work_dir / "prepared" / safe
            candidate_root.mkdir(parents=True)
            sample = _canonical_sample(candidate, plan, candidate_root, model_types)
            prepared = prepare_mpnn_input(
                sample,
                candidate_root / "native-input",
                ca_only=False,
                materialize=True,
            )
            current_symmetry = _resolved_ligand_symmetry(
                raw_symmetry_residues,
                raw_symmetry_weights,
                prepared,
                plan,
            )
            if plan.design_chain is None:
                raise MPNNSequenceError("MPNN plan lacks its design chain")
            design_alias = prepared.alias_by_chain[plan.design_chain]
            protein_aliases = sorted(
                prepared.alias_by_chain[record.chain_id]
                for record in sample.chains
                if record.modality == "protein"
            )
            design_segment = protein_aliases.index(design_alias)
            fixed = {
                (prepared.alias_by_chain[chain], int(position))
                for chain, position in plan.fixed_positions
                if chain == plan.design_chain
            }
            design_residues = prepared.residues_by_chain[plan.design_chain]
            redesigned = [
                f"{design_alias}{residue.temporary_position}"
                for residue in design_residues
                if (design_alias, residue.canonical_position) not in fixed
            ]
            omit = _ligand_omit_payload(
                candidate,
                plan,
                prepared,
                design_alias=design_alias,
                backend=backend,
            )
        except MPNNSequenceError as exc:
            failures[candidate.candidate_id] = str(exc)
            continue
        if symmetry_contract is None:
            symmetry_contract = current_symmetry
        elif current_symmetry != symmetry_contract:
            raise MPNNSystemError(
                "batched LigandMPNN candidates resolve different symmetry contracts"
            )
        pdb_path = input_dir / f"{safe}.pdb"
        shutil.copyfile(prepared.pdb_path, pdb_path)
        records.append(
            {
                "candidate": candidate,
                "plan": plan,
                "safe": safe,
                "pdb_path": pdb_path,
                "pdb_bytes": pdb_path.read_bytes(),
                "fixed": " ".join(
                    f"{alias}{position}" for alias, position in sorted(fixed)
                ),
                "redesigned": " ".join(redesigned),
                "design_segment": design_segment,
                "omit": omit,
            }
        )

    if not records:
        return {}, failures

    # Equality selects duplicate native inputs with identical controls; this
    # grouping decision is kept in memory for the current invocation.
    groups: list[list[dict[str, Any]]] = []
    for record in records:
        for group in groups:
            first = group[0]
            if (
                record["pdb_bytes"] == first["pdb_bytes"]
                and record["fixed"] == first["fixed"]
                and record["redesigned"] == first["redesigned"]
                and record["design_segment"] == first["design_segment"]
                and record["omit"] == first["omit"]
            ):
                group.append(record)
                break
        else:
            groups.append([record])

    result: dict[str, SequenceSourcePlan] = {}
    model_loads = 0
    started = time.perf_counter()
    for batch_index, group in enumerate(groups):
        batch_dir = work_dir / f"batch_{batch_index}"
        batch_dir.mkdir()
        first = group[0]
        first_path = str(first["pdb_path"])
        paths_json = batch_dir / "pdb_paths.json"
        fixed_json = batch_dir / "fixed_residues.json"
        redesigned_json = batch_dir / "redesigned_residues.json"
        _write_json(paths_json, {first_path: ""})
        _write_json(fixed_json, {first_path: first["fixed"]})
        _write_json(redesigned_json, {first_path: first["redesigned"]})
        omit_payload = (
            {first_path: first["omit"]}
            if first["omit"] is not None
            else None
        )
        omit_path = None
        if omit_payload is not None:
            omit_path = batch_dir / "omit_AA_per_residue_multi.json"
            _write_json(omit_path, omit_payload)
        output = batch_dir / "output"
        command = _ligand_suite_command(
            repo_root=repo_root,
            work_dir=batch_dir,
            paths_json=paths_json,
            fixed_json=fixed_json,
            redesigned_json=redesigned_json,
            output=output,
            backend=backend,
            symmetry_residues_cli=(
                None if symmetry_contract is None else symmetry_contract[0]
            ),
            symmetry_weights_cli=(
                None if symmetry_contract is None else symmetry_contract[1]
            ),
            omit_path=omit_path,
            batch_size=len(group),
        )
        cache_stats = _run_cached_mpnn(
            command,
            backend="ligandmpnn",
            source=source,
            cwd=community_source("ligandmpnn"),
            env=_offline_environment(repo_root),
            work_dir=batch_dir,
        )
        model_loads += cache_stats["model_loads_this_request"]
        sequences = _read_design_sequences(
            output / "seqs" / f"{first['safe']}.fa",
            int(first["design_segment"]),
            separator=":",
            where="LigandMPNN",
        )
        if len(sequences) != len(group):
            raise MPNNSystemError(
                "LigandMPNN returned a different number of sequences than its native batch"
            )
        for record, sequence in zip(group, sequences):
            candidate_id = record["candidate"].candidate_id
            try:
                result[candidate_id] = _validate_sequence(record["plan"], sequence)
            except MPNNSequenceError as exc:
                failures[candidate_id] = str(exc)

    elapsed = time.perf_counter() - started
    return result, failures


def _batched_cyclicmpnn_sequences(
    items: Sequence[tuple[SearchCandidate, SequenceSourcePlan]],
    *,
    reward_config: Mapping[str, Any],
    repo_root: Path,
    work_dir: Path,
    timings: dict[str, float] | None = None,
) -> tuple[dict[str, SequenceSourcePlan], dict[str, str]]:
    if not items:
        return {}, {}
    backend = reward_config["sequence"]["cyclicmpnn"]
    _backend_assets(backend, "cyclicmpnn")
    explicit_rm_aa = _has_explicit_rm_aa(backend)
    omission = resolve_rm_aa(backend)
    work_dir.mkdir(parents=True, exist_ok=False)
    api, _SequenceConfig, model_types, _RuntimePaths, _ = _load_pipetools(repo_root)
    prepare_mpnn_input = api[0]
    requests: list[dict[str, Any]] = []
    plan_by_id: dict[str, SequenceSourcePlan] = {}
    failures: dict[str, str] = {}
    bias_path = ""
    if backend.get("bias_aa"):
        path = work_dir / "bias-aa.json"
        _write_json(path, dict(backend["bias_aa"]))
        bias_path = str(path)
    for candidate_index, (candidate, plan) in enumerate(items):
        try:
            safe = f"candidate_{candidate_index}"
            candidate_root = work_dir / "prepared" / safe
            candidate_root.mkdir(parents=True)
            sample = _canonical_sample(candidate, plan, candidate_root, model_types)
            prepared = prepare_mpnn_input(
                sample,
                candidate_root / "native-input",
                ca_only=False,
                materialize=True,
            )
            if plan.design_chain is None:
                raise MPNNSequenceError("CyclicMPNN plan lacks its design chain")
            design_alias = prepared.alias_by_chain[plan.design_chain]
            context_aliases = [
                prepared.alias_by_chain[record.chain_id]
                for record in sample.context_chains
                if record.modality == "protein"
            ]
            fixed = [
                f"{prepared.alias_by_chain[chain]}{position}"
                for chain, position in plan.fixed_positions
                if chain == plan.design_chain
            ]
            omit_positions = (
                list(_free_design_protein_positions(candidate, plan))
                if explicit_rm_aa
                else []
            )
            _check_rm_aa_has_allowed_free_position(backend, omit_positions)
            requests.append(
                {
                    "candidate_id": candidate.candidate_id,
                    # Keep the prepared PDB in its candidate-owned directory
                    # for the worker lifetime instead of copying it into a
                    # second batch input tree.
                    "pdb_path": str(prepared.pdb_path),
                    "design_chain": design_alias,
                    "context_chains": context_aliases,
                    "fixed_positions": fixed,
                    "seed": int(backend.get("seed", 101)),
                    "temperature": float(backend["temperature"]),
                    "omit_positions": omit_positions,
                }
            )
            plan_by_id[candidate.candidate_id] = plan
        except MPNNSequenceError as exc:
            failures[candidate.candidate_id] = str(exc)
    if not requests:
        return {}, failures
    request_path = work_dir / "request.json"
    result_path = work_dir / "result.json"
    _write_json(
        request_path,
        {
            "schema_version": 1,
            "config": {
                "model_name": str(backend["base_model"]),
                "base_model_name": str(backend["base_model"]),
                "checkpoint": str(backend["checkpoint"]),
                "temperature": float(backend["temperature"]),
                "backbone_noise": float(backend["backbone_noise"]),
                "omit_aas": "" if explicit_rm_aa else omission,
                "rm_aa": omission if explicit_rm_aa else None,
                "bias_aa_json": bias_path,
            },
            "requests": requests,
        },
    )
    command = [
        sys.executable,
        "-m",
        "src.design.rewards.cyclicmpnn_worker",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    env = _offline_environment(repo_root)
    env["PYTHONPATH"] = _release_pythonpath(
        community_source("cyclicmpnn"),
        community_source("colabdesign"),
    )
    cache_stats = _run_cached_mpnn(
        command, backend='cyclicmpnn', source='cyclicmpnn', cwd=repo_root,
        env=env, work_dir=work_dir,
    )
    raw = json.loads(result_path.read_text())
    if (not isinstance(raw, Mapping) or type(raw.get("model_loads")) is not int
            or raw["model_loads"] != cache_stats["model_loads_this_request"]):
        raise MPNNSystemError("CyclicMPNN batch protocol is invalid")
    if timings is not None:
        for name in ("model_load_seconds",):
            value = raw.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                timings[f"cyclicmpnn_{name}"] = float(value)
    values = raw.get("results")
    if not isinstance(values, list) or len(values) != len(requests):
        raise MPNNSystemError("CyclicMPNN batch result count differs from requests")
    result: dict[str, SequenceSourcePlan] = {}
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise MPNNSystemError("CyclicMPNN returned a non-mapping result")
        candidate_id = str(value.get("candidate_id"))
        if candidate_id in seen or candidate_id not in plan_by_id:
            raise MPNNSystemError("CyclicMPNN candidate accounting is invalid")
        seen.add(candidate_id)
        if value.get("status") != "ok":
            failures[candidate_id] = str(value.get("error") or "CyclicMPNN failed")
            continue
        try:
            result[candidate_id] = _validate_sequence(
                plan_by_id[candidate_id], str(value["sequence"])
            )
        except MPNNSequenceError as exc:
            failures[candidate_id] = str(exc)
    return result, failures


def _run_cached_mpnn(
    command: Sequence[str], *, backend: str, source: str, cwd: Path,
    env: Mapping[str, str], work_dir: Path,
) -> dict[str, Any]:
    """Keep model lifetimes independent of each reward batch's output directory."""
    log_path = work_dir / "process.log"
    try:
        stats = run_mpnn_cached(
            command, backend=backend, cwd=cwd, env=env,
            log_path=log_path, timeout=7200,
        )
    except (MPNNCacheError, OSError, subprocess.SubprocessError) as exc:
        tail = _sanitized_process_log_tail(log_path)
        detail = str(exc).strip() or type(exc).__name__
        raise MPNNSystemError(
            f"{source} batched worker failed: {type(exc).__name__}: {detail}; "
            f"process_log_tail={tail}"
        ) from exc
    if (not isinstance(stats, Mapping)
            or type(stats.get("model_loads_this_request")) is not int
            or stats["model_loads_this_request"] < 0):
        raise MPNNSystemError(f"{source} resident cache returned invalid load metrics")
    return dict(stats)


def _with_batch_seed(
    reward_config: Mapping[str, Any],
    *,
    source: str,
    seed: int | None,
) -> Mapping[str, Any]:
    """Return a source-local config with an optional validation batch seed.

    Reward callers omit ``seed`` and retain their configured backend seed.  A
    validation batch supplies one common seed; copy only the selected backend
    section so the caller's nested configuration is never mutated.
    """

    if seed is None:
        return reward_config
    sequence = reward_config.get("sequence")
    if not isinstance(sequence, Mapping):
        raise MPNNSequenceError("MPNN reward configuration lacks sequence backends")
    backend = sequence.get(source)
    if not isinstance(backend, Mapping):
        raise MPNNSequenceError(f"MPNN reward configuration lacks backend {source!r}")
    effective_backend = dict(backend)
    effective_backend["seed"] = seed
    effective_sequence = dict(sequence)
    effective_sequence[source] = effective_backend
    effective_config = dict(reward_config)
    effective_config["sequence"] = effective_sequence
    return effective_config


def generate_mpnn_sequences(
    items: Sequence[tuple[SearchCandidate, SequenceSourcePlan]],
    *,
    reward_config: Mapping[str, Any],
    repo_root: Path,
    work_dir: Path,
    seed: int | None = None,
    timings: dict[str, float] | None = None,
    antibody_plans: Mapping[str, AntibodyMPNNPlan] | None = None,
) -> tuple[dict[str, SequenceSourcePlan], dict[str, str]]:
    """Generate one sequence per candidate, reusing resident backend models.

    Candidates are grouped by native source, and each source group is sent to
    its backend's batch worker.  ``seed`` is an optional common seed for one
    validation batch; it does not alter reward callers that omit it.
    """

    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise MPNNSequenceError("MPNN batch seed must be a non-negative integer")
    result: dict[str, SequenceSourcePlan] = {}
    failures: dict[str, str] = {}
    item_ids = {candidate.candidate_id for candidate, _plan in items}
    if len(item_ids) != len(items):
        raise MPNNSequenceError("MPNN candidates must have distinct candidate IDs")
    antibody_plans = {} if antibody_plans is None else dict(antibody_plans)
    unknown_antibody_ids = set(antibody_plans) - item_ids
    if unknown_antibody_ids:
        raise MPNNSequenceError(
            "antibody_plans contains unknown candidate IDs: "
            + ", ".join(sorted(unknown_antibody_ids))
        )
    grouped: dict[str, list[tuple[SearchCandidate, SequenceSourcePlan]]] = {}
    for candidate, plan in items:
        if plan.source == "design":
            result[candidate.candidate_id] = plan
        else:
            grouped.setdefault(plan.source, []).append((candidate, plan))
    requested = str(
        reward_config.get("sequence", {}).get("source", "design")
    )
    for source, group in sorted(grouped.items()):
        effective_reward_config = reward_config
        if source == "proteinmpnn" and requested == "auto":
            # Reward auto uses the native proteinmpnn source key for its
            # SolubleMPNN route.  Normalize only when that group exists, so
            # nucleic auto candidates that remain on design do not require a
            # protein backend or accidentally load one.
            from .sequence import normalize_reward_sequence_config

            try:
                effective_reward_config = normalize_reward_sequence_config(
                    reward_config,
                    requested=requested,
                    require_soluble=True,
                )
            except ValueError as exc:
                raise MPNNSequenceError(str(exc)) from exc
        effective_config = _with_batch_seed(
            effective_reward_config, source=source, seed=seed
        )
        if source == "na_mpnn":
            generated, rejected = _batched_na_mpnn_sequences(
                group,
                reward_config=effective_config,
                repo_root=repo_root,
                work_dir=work_dir / source,
            )
            result.update(generated)
            failures.update(rejected)
            continue
        if source in {"proteinmpnn", "ligandmpnn"}:
            antibody_group = [
                (candidate, plan)
                for candidate, plan in group
                if candidate.candidate_id in antibody_plans
            ]
            generic_group = [
                (candidate, plan)
                for candidate, plan in group
                if candidate.candidate_id not in antibody_plans
            ]
            if antibody_group:
                if source == "proteinmpnn":
                    generated, rejected = _batched_antibody_proteinmpnn_sequences(
                        antibody_group,
                        antibody_plans=antibody_plans,
                        reward_config=effective_config,
                        repo_root=repo_root,
                        work_dir=work_dir / f"{source}_antibody",
                    )
                else:
                    generated, rejected = _batched_antibody_ligandmpnn_sequences(
                        antibody_group,
                        antibody_plans=antibody_plans,
                        reward_config=effective_config,
                        repo_root=repo_root,
                        work_dir=work_dir / f"{source}_antibody",
                    )
                result.update(generated)
                failures.update(rejected)
            if generic_group:
                generated, rejected = _batched_ligand_suite_sequences(
                    generic_group,
                    source=source,
                    reward_config=effective_config,
                    repo_root=repo_root,
                    work_dir=work_dir / source,
                )
                result.update(generated)
                failures.update(rejected)
            continue
        if source != "cyclicmpnn":
            raise MPNNSequenceError(f"unsupported MPNN source {source!r}")
        source_started = time.perf_counter()
        generated, rejected = _batched_cyclicmpnn_sequences(
            group,
            reward_config=effective_config,
            repo_root=repo_root,
            work_dir=work_dir / source,
            timings=timings,
        )
        if timings is not None:
            timings[f"{source}_total_seconds"] = time.perf_counter() - source_started
        result.update(generated)
        failures.update(rejected)
    return result, failures


__all__ = [
    "AntibodyMPNNPlan",
    "MPNNSequenceError",
    "MPNNSystemError",
    "build_antibody_mpnn_plan",
    "generate_mpnn_sequences",
]
