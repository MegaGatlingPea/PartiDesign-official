"""Offline ESMFold2 reward planning and score reduction.

The heavy ESMFold2/MPNN processes are launched only after the PartiDesign model
has been offloaded.  This module owns the model-free, deterministic contract:
sequence-source selection, fixed protein tokens, private fold input identity,
and reduction of ESMFold2 confidence values into a ranking score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from biotite.structure import AtomArray

from ..core.records import SearchCandidate
from ..model.constants import DNA_RES_ATOMS, PROTEIN_RES_ATOMS, RNA_RES_ATOMS
from .interface import interface_fixed_positions, normalize_fixed_interface


_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
_RNA3_TO_1 = {"A": "A", "G": "G", "C": "C", "U": "U"}
_DNA3_TO_1 = {"DA": "A", "DG": "G", "DC": "C", "DT": "T"}
_POLYMER_MODALITIES = frozenset({"protein", "rna", "dna"})
_POLYMER_RES_TO_1 = {
    "protein": _AA3_TO_1,
    "rna": _RNA3_TO_1,
    "dna": _DNA3_TO_1,
}
_POLYMER_RES_ATOMS = {
    "protein": PROTEIN_RES_ATOMS,
    "rna": RNA_RES_ATOMS,
    "dna": DNA_RES_ATOMS,
}
_PROTEIN_BACKBONE = frozenset({"N", "CA", "C", "O", "OXT"})


@dataclass(frozen=True, slots=True)
class ProteinToken:
    chain_id: str
    token_uid: int
    position: int
    residue_name: str
    is_design: bool
    identity_conditioned: bool
    # Kept as a trailing default so existing positional ProteinToken callers
    # and serialized SequenceSourcePlan consumers remain unchanged.
    modality: str = "protein"
    residue_uid: int = -1


@dataclass(frozen=True, slots=True)
class SequenceSourcePlan:
    source: str
    design_chain: str | None
    chain_sequences: Mapping[str, str]
    fixed_positions: tuple[tuple[str, int], ...]
    interface_positions: tuple[tuple[str, int], ...] = ()
    interface_target_chains: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ESMFold2Confidence:
    candidate_id: str
    design_plddt: float
    interface_iptm: float | None
    ptm: float | None = None
    interface_iplddt: float | None = None
    motif_rmsd_angstrom: float | None = None


@dataclass(frozen=True, slots=True)
class ESMFold2ScoreBatch:
    candidate_ids: tuple[str, ...]
    total: torch.Tensor
    design_plddt: torch.Tensor
    interface_iptm: torch.Tensor

    def __post_init__(self) -> None:
        size = len(self.candidate_ids)
        for name in ("total", "design_plddt", "interface_iptm"):
            value = getattr(self, name)
            if value.dtype != torch.float32 or value.shape != (size,):
                raise ValueError(f"{name} must be float32[{size}]")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} contains non-finite values")


def _token_rows(atoms: AtomArray) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for token_uid in sorted(set(int(value) for value in atoms.token_uid)):
        rows = np.flatnonzero(np.asarray(atoms.token_uid, dtype=np.int64) == token_uid)
        if len(rows):
            result.append(rows)
    return result


def protein_tokens(atoms: AtomArray) -> tuple[ProteinToken, ...]:
    return _polymer_tokens(atoms, modalities=frozenset({"protein"}))


def polymer_tokens(atoms: AtomArray) -> tuple[ProteinToken, ...]:
    """Return deterministic token rows for standard polymer modalities.

    ProteinToken is retained as the public record name for ABI compatibility;
    its trailing modality field makes the same ledger usable for RNA and DNA
    without changing existing protein callers. Mixed or non-polymer tokens
    are intentionally omitted here. The NA eligibility validator below
    inspects those rows directly and rejects them with a candidate-local
    explanation rather than silently treating them as context.
    """

    return _polymer_tokens(atoms, modalities=_POLYMER_MODALITIES)


def _polymer_tokens(
    atoms: AtomArray,
    *,
    modalities: frozenset[str],
) -> tuple[ProteinToken, ...]:
    positions: dict[str, int] = {}
    result: list[ProteinToken] = []
    for rows in _token_rows(atoms):
        token_modalities = set(str(value) for value in atoms.mol_type[rows])
        if len(token_modalities) != 1 or not token_modalities <= modalities:
            continue
        chain_ids = set(str(value) for value in atoms.chain_id[rows])
        residues = set(str(value) for value in atoms.res_name[rows])
        if len(chain_ids) != 1 or len(residues) != 1:
            raise ValueError("one protein token has inconsistent chain/residue identity")
        chain = next(iter(chain_ids))
        modality = next(iter(token_modalities))
        positions[chain] = positions.get(chain, 0) + 1
        identity = np.asarray(atoms.identity_condition[rows], dtype=bool)
        if bool(identity.any()) != bool(identity.all()):
            raise ValueError("one protein token has partial identity conditioning")
        result.append(
            ProteinToken(
                chain_id=chain,
                token_uid=int(atoms.token_uid[rows[0]]),
                position=positions[chain],
                residue_name=next(iter(residues)),
                is_design=bool(np.asarray(atoms.is_design[rows], dtype=bool).all()),
                identity_conditioned=bool(identity.all()),
                modality=modality,
                residue_uid=int(atoms.residue_uid[rows[0]]),
            )
        )
    return tuple(result)


_UNKNOWN_SEQUENCE_LETTER = {"protein": "X", "rna": "N", "dna": "N"}


def _token_model_residue(
    candidate: SearchCandidate,
    token: ProteinToken,
) -> str | None:
    """Return a canonical model residue, or ``None`` for a fixed unknown scaffold."""

    mapping = _POLYMER_RES_TO_1[token.modality]
    if token.residue_name in mapping:
        return token.residue_name
    if token.residue_uid in candidate.fixed_modified_residue_names:
        return candidate.fixed_modified_parent_residue.get(token.residue_uid)
    parent = candidate.model_parent_residue.get(token.residue_uid)
    if parent in mapping:
        return str(parent)
    return None


def _sequence_letter(candidate: SearchCandidate, token: ProteinToken) -> str:
    residue = _token_model_residue(candidate, token)
    if residue is None:
        if (
            token.residue_uid in candidate.fixed_modified_residue_names
            or token.residue_uid in candidate.model_parent_residue
        ):
            return _UNKNOWN_SEQUENCE_LETTER[token.modality]
        raise ValueError(
            "sequence planning requires standard or fixed modified polymer identity, "
            f"got {token.modality}:{token.residue_name!r}"
        )
    return _POLYMER_RES_TO_1[token.modality][residue]


def _protein_sequences(
    candidate: SearchCandidate,
    tokens: Sequence[ProteinToken],
) -> dict[str, str]:
    by_chain: dict[str, list[str]] = {}
    for token in tokens:
        letter = _sequence_letter(candidate, token)
        by_chain.setdefault(token.chain_id, []).append(letter)
    return {chain: "".join(sequence) for chain, sequence in by_chain.items()}


def _polymer_sequences(
    candidate: SearchCandidate,
    tokens: Sequence[ProteinToken],
) -> dict[str, str]:
    by_chain: dict[str, list[str]] = {}
    modality_by_chain: dict[str, str] = {}
    for token in tokens:
        modality = str(token.modality)
        letter = _sequence_letter(candidate, token)
        previous = modality_by_chain.setdefault(token.chain_id, modality)
        if previous != modality:
            raise ValueError(
                f"polymer chain {token.chain_id!r} mixes {previous} and {modality}"
            )
        by_chain.setdefault(token.chain_id, []).append(letter)
    return {chain: "".join(sequence) for chain, sequence in by_chain.items()}


def _design_protein_chains(tokens: Sequence[ProteinToken]) -> tuple[str, ...]:
    return tuple(sorted({token.chain_id for token in tokens if token.is_design}))


def _design_chain_ids(atoms: AtomArray) -> tuple[str, ...]:
    design = np.asarray(atoms.is_design, dtype=bool)
    return tuple(
        sorted({str(value) for value in np.asarray(atoms.chain_id)[design].tolist()})
    )


def _na_design_is_present(atoms: AtomArray) -> bool:
    design = np.asarray(atoms.is_design, dtype=bool)
    if not bool(design.any()):
        return False
    for chain in _design_chain_ids(atoms):
        rows = np.flatnonzero(
            (np.asarray(atoms.chain_id).astype(str) == chain) & design
        )
        if any(str(value) in {"rna", "dna"} for value in atoms.mol_type[rows]):
            return True
    return False


def _modified_proxy_error(
    candidate: SearchCandidate,
    *,
    allowed_modalities: frozenset[str] = _POLYMER_MODALITIES,
) -> str | None:
    """Return why a fixed modified occurrence cannot form a safe MPNN view."""

    if candidate.model_parent_residue:
        return "MPNN rejects unresolved proxy-only polymer residues"
    atoms = candidate.atoms
    for residue_uid, component_id in candidate.fixed_modified_residue_names.items():
        rows = np.flatnonzero(
            np.asarray(atoms.residue_uid, dtype=np.int64) == int(residue_uid)
        )
        if not len(rows):
            return f"modified residue {component_id!r} is absent"
        modalities = {str(value) for value in atoms.mol_type[rows]}
        if len(modalities) != 1:
            return f"modified residue {component_id!r} has ambiguous modality"
        modality = next(iter(modalities))
        if modality not in allowed_modalities:
            return f"MPNN cannot proxy modified {modality} residue {component_id!r}"
        parent = candidate.fixed_modified_parent_residue.get(int(residue_uid))
        if parent not in _POLYMER_RES_TO_1[modality]:
            return (
                f"modified {modality} residue {component_id!r} lacks an "
                "unambiguous standard parent"
            )
        identity = np.asarray(atoms.identity_condition[rows], dtype=bool)
        if not bool(identity.all()):
            return f"modified residue {component_id!r} is not hard identity-fixed"
        atom_names = [str(value) for value in atoms.atom_name[rows]]
        if len(atom_names) != len(set(atom_names)):
            return f"modified residue {component_id!r} has ambiguous atom names"
        present = set(atom_names)
        if modality == "protein":
            required = {"N", "CA", "C", "O"}
        else:
            base_anchor = "N9" if parent in {"A", "G", "DA", "DG"} else "N1"
            required = {"C1'", "C3'", "C4'", base_anchor}
        missing = sorted(required - present)
        if missing:
            return (
                f"modified residue {component_id!r} lacks proxy frame atoms "
                f"{missing}"
            )
        allowed = set(_POLYMER_RES_ATOMS[modality][parent])
        retained_rows = np.asarray(
            [row for row in rows if str(atoms.atom_name[row]) in allowed],
            dtype=np.int64,
        )
        if not len(retained_rows):
            return f"modified residue {component_id!r} has no proxy atoms"
        if not bool(np.asarray(atoms.coord_valid[retained_rows], dtype=bool).all()):
            return f"modified residue {component_id!r} lacks valid proxy coordinates"
        retained_coordinates = np.asarray(atoms.coord[retained_rows], dtype=np.float64)
        if not bool(np.isfinite(retained_coordinates).all()):
            return f"modified residue {component_id!r} has non-finite proxy coordinates"
        frame_names = (
            ("N", "CA", "C")
            if modality == "protein"
            else ("C1'", "C3'", "C4'")
        )
        row_by_name = {
            str(atoms.atom_name[row]): int(row)
            for row in rows
            if str(atoms.atom_name[row]) in frame_names
        }
        frame = np.asarray(
            [atoms.coord[row_by_name[name]] for name in frame_names],
            dtype=np.float64,
        )
        centered = frame - frame.mean(axis=0, keepdims=True)
        if int(np.linalg.matrix_rank(centered, tol=1e-6)) < 2:
            return f"modified residue {component_id!r} has a degenerate proxy frame"
    return None


def _validate_na_candidate(
    candidate: SearchCandidate,
) -> tuple[str, tuple[ProteinToken, ...]]:
    """Validate and return the sole design NA chain and its polymer ledger."""

    atoms = candidate.atoms
    proxy_error = _modified_proxy_error(candidate)
    if proxy_error is not None:
        raise ValueError(proxy_error)
    if len(atoms) == 0:
        raise ValueError("na_mpnn requires a non-empty candidate")
    token_ids = np.asarray(atoms.token_uid, dtype=np.int64)
    if bool((token_ids < 0).any()):
        raise ValueError("na_mpnn requires a finalized token ledger")

    chain_order: list[str] = []
    for raw_chain in np.asarray(atoms.chain_id).astype(str):
        chain = str(raw_chain)
        if chain not in chain_order:
            chain_order.append(chain)
    design_chains = _design_chain_ids(atoms)
    if len(design_chains) != 1:
        raise ValueError("na_mpnn requires exactly one design chain")
    design_chain = design_chains[0]

    for chain in chain_order:
        rows = np.flatnonzero(np.asarray(atoms.chain_id).astype(str) == chain)
        modalities = {str(value) for value in atoms.mol_type[rows]}
        if not modalities:
            raise ValueError(f"na_mpnn chain {chain!r} has no modality")
        design_flags = np.asarray(atoms.is_design[rows], dtype=bool)
        if bool(design_flags.any()) != bool(design_flags.all()):
            raise ValueError(
                f"na_mpnn chain {chain!r} mixes design and context rows"
            )
        # A design chain must be a single canonical NA modality. Context may
        # contain protein/NA/ligand atoms, including mixed modalities sharing
        # one source chain; private NA input filters non-polymer atoms later.
        modality = next(iter(modalities)) if len(modalities) == 1 else None
        if design_flags.any():
            if modality not in {"rna", "dna"}:
                raise ValueError(
                    f"na_mpnn design chain {chain!r} must be entirely RNA or DNA"
                )
        seen_positions: set[tuple[int, str]] = set()
        token_uids = sorted(
            {int(value) for value in np.asarray(atoms.token_uid[rows], dtype=np.int64)}
        )
        for token_uid in token_uids:
            token = np.flatnonzero(
                np.asarray(atoms.token_uid, dtype=np.int64) == token_uid
            )
            token_chain = {str(value) for value in atoms.chain_id[token]}
            if token_chain != {chain}:
                raise ValueError(f"na_mpnn chain {chain!r} has an ambiguous token")
            token_modalities = {str(value) for value in atoms.mol_type[token]}
            residues = {str(value) for value in atoms.res_name[token]}
            if len(token_modalities) != 1 or len(residues) != 1:
                raise ValueError(f"na_mpnn chain {chain!r} has an ambiguous token")
            token_modality = next(iter(token_modalities))
            if modality is not None and token_modality != modality:
                raise ValueError(f"na_mpnn chain {chain!r} has an ambiguous token")
            design_values = np.asarray(atoms.is_design[token], dtype=bool)
            if bool(design_values.any()) != bool(design_values.all()):
                raise ValueError(f"na_mpnn chain {chain!r} has an ambiguous token")
            identity_values = np.asarray(atoms.identity_condition[token], dtype=bool)
            if bool(identity_values.any()) != bool(identity_values.all()):
                raise ValueError(f"na_mpnn chain {chain!r} has partial identity")
            # Ligand/non-polymer context is valid for public candidates but is
            # deliberately absent from the NA worker ledger.
            if token_modality not in _POLYMER_MODALITIES:
                continue
            residue = next(iter(residues))
            residue_uid = int(atoms.residue_uid[token[0]])
            is_modified = residue_uid in candidate.fixed_modified_residue_names
            model_residue = (
                candidate.fixed_modified_parent_residue.get(residue_uid)
                if is_modified
                else residue
            )
            if model_residue not in _POLYMER_RES_TO_1[token_modality]:
                raise ValueError(
                    f"na_mpnn rejects unresolved/noncanonical {token_modality} residue "
                    f"{residue!r}"
                )
            atom_names = [str(value) for value in atoms.atom_name[token].tolist()]
            if len(atom_names) != len(set(atom_names)):
                raise ValueError(
                    f"na_mpnn chain {chain!r} has duplicate/ambiguous atom rows"
                )
            if token_modality in {"rna", "dna"}:
                expected_atoms = set(_POLYMER_RES_ATOMS[token_modality][model_residue])
                # A sequence generator needs an unambiguous base anchor and
                # sugar frame, but terminal phosphate/other crystallographic
                # atoms may be absent in a valid source structure. Reject
                # unknown identities; permit missing optional heavy atoms.
                required = {"C1'"}
                if model_residue in {"A", "G", "DA", "DG"}:
                    required.add("N9")
                else:
                    required.add("N1")
                if not required <= set(atom_names):
                    raise ValueError(
                        f"na_mpnn chain {chain!r} has incomplete residue "
                        f"{residue!r}"
                    )
                unknown = set(atom_names) - expected_atoms
                if unknown and not is_modified:
                    raise ValueError(
                        f"na_mpnn chain {chain!r} has noncanonical atoms "
                        f"{sorted(unknown)!r}"
                    )
            key = (int(atoms.res_id[token[0]]), str(atoms.ins_code[token[0]]))
            if key in seen_positions:
                raise ValueError(
                    f"na_mpnn chain {chain!r} has ambiguous residue numbering"
                )
            seen_positions.add(key)

    ledger = polymer_tokens(atoms)
    if not ledger:
        raise ValueError("na_mpnn candidate has no canonical polymer tokens")
    design_tokens = [token for token in ledger if token.chain_id == design_chain]
    if not design_tokens:
        raise ValueError("na_mpnn design chain has no canonical tokens")
    modality = {token.modality for token in design_tokens}
    if modality not in ({"rna"}, {"dna"}):
        raise ValueError("na_mpnn design chain must be entirely RNA or entirely DNA")
    return design_chain, ledger


def has_required_head_to_tail_single_bond(candidate: SearchCandidate, chain_id: str) -> bool:
    atoms = candidate.atoms
    uid_to_row = {int(uid): row for row, uid in enumerate(atoms.atom_uid)}
    chain_tokens = [
        token for token in protein_tokens(atoms) if token.chain_id == chain_id
    ]
    if not chain_tokens:
        return False
    first_uid = chain_tokens[0].token_uid
    last_uid = chain_tokens[-1].token_uid
    for condition in candidate.topology_conditions:
        if not condition.required_output or condition.order != "single":
            continue
        left = uid_to_row.get(int(condition.atom_uid_a))
        right = uid_to_row.get(int(condition.atom_uid_b))
        if left is None or right is None:
            continue
        endpoints = {
            (int(atoms.token_uid[left]), str(atoms.atom_name[left])),
            (int(atoms.token_uid[right]), str(atoms.atom_name[right])),
        }
        if endpoints == {(first_uid, "N"), (last_uid, "C")}:
            return True
    return False


def has_cyclic_offset(candidate: SearchCandidate, chain_id: str) -> bool:
    """Return whether a task explicitly requests cyclic-offset handling.

    Cyclic offset is an input intent used by downstream refolding.  It does
    not add or imply a chemical edge in the candidate graph.  Older callers
    often carry a ``SimpleNamespace`` task without the newer fields, so the
    task attributes are intentionally read through ``getattr``.
    """

    request = getattr(candidate, "base_request", None)
    task = getattr(request, "task", None)
    if not bool(getattr(task, "cyclic_offset", False)):
        return False

    design_chains = _design_protein_chains(protein_tokens(candidate.atoms))
    if chain_id not in design_chains:
        return False

    selected = getattr(task, "cyclic_offset_chains", ())
    if selected is None:
        selected = ()
    elif isinstance(selected, str):
        selected = (selected,)
    else:
        try:
            selected = tuple(selected)
        except TypeError:
            selected = ()
    selected = {str(value) for value in selected}
    if selected:
        if str(chain_id) in selected:
            return True
        # Validation metadata names final CIF labels, while direct AtomArray
        # callers may retain different author chain IDs in the token ledger.
        labels = getattr(candidate.atoms, "label_asym_id", None)
        if labels is not None:
            rows = np.asarray(candidate.atoms.chain_id).astype(str) == str(chain_id)
            final_labels = set(np.asarray(labels).astype(str)[rows].tolist())
            return len(final_labels) == 1 and bool(final_labels & selected)
        return False
    return len(design_chains) == 1


def _all_entities_are_standard_protein(candidate: SearchCandidate) -> bool:
    atoms = candidate.atoms
    if set(str(value) for value in atoms.mol_type) != {"protein"}:
        return False
    if _modified_proxy_error(
        candidate, allowed_modalities=frozenset({"protein"})
    ) is not None:
        return False
    fixed = set(int(value) for value in candidate.fixed_modified_residue_names)
    return all(
        str(atoms.res_name[row]) in _AA3_TO_1
        or int(atoms.residue_uid[row]) in fixed
        for row in range(len(atoms))
    )


def _has_nonprotein_entities(atoms: AtomArray) -> bool:
    modality = np.asarray(atoms.mol_type).astype(str)
    return bool(np.any(modality != "protein"))


def resolve_sequence_source(candidate: SearchCandidate, requested: str) -> str:
    atoms = candidate.atoms
    if candidate.base_request.task.format == "antibody":
        if requested not in {"auto", "design", "proteinmpnn", "ligandmpnn"}:
            raise ValueError("antibody sequence source must be auto, design, proteinmpnn or ligandmpnn")
        required = {"antibody_chain_role_id", "antibody_sequence_mutable"}
        if not required <= set(atoms.get_annotation_categories()):
            raise ValueError("antibody sequence planning requires compiler residue annotations")
        if not bool(atoms.antibody_sequence_mutable.any()):
            return "design"
        source = "proteinmpnn" if requested == "auto" else requested
        if source == "design":
            return source
        error = _modified_proxy_error(candidate, allowed_modalities=frozenset({"protein"}))
        if error is not None:
            raise ValueError(error)
        tokens = protein_tokens(atoms)
        if any(_token_model_residue(candidate, token) not in _AA3_TO_1 for token in tokens if token.is_design):
            raise ValueError("antibody MPNN requires canonical design protein identities")
        # ProteinMPNN consumes the protein subset; nonprotein antigen atoms
        # remain fixed context. LigandMPNN additionally consumes atomic context.
        return source
    if requested == "na_mpnn":
        _validate_na_candidate(candidate)
        return "na_mpnn"
    if requested == "auto" and _na_design_is_present(atoms):
        # NA involvement is deliberately candidate-local. An ineligible NA
        # design must go through the ordinary finalized-candidate design path,
        # never through a protein or ligand MPNN selected from partial rows.
        try:
            _validate_na_candidate(candidate)
        except ValueError:
            return "design"
        return "na_mpnn"
    tokens = protein_tokens(atoms)
    design_chains = _design_protein_chains(tokens)
    protein_proxy_error = _modified_proxy_error(
        candidate, allowed_modalities=frozenset({"protein"})
    )
    has_modified = bool(candidate.fixed_modified_residue_names)
    if requested != "auto":
        source = requested
    elif len(design_chains) != 1:
        source = "design"
    elif (
        not has_modified
        and (
            has_required_head_to_tail_single_bond(candidate, design_chains[0])
            or has_cyclic_offset(candidate, design_chains[0])
        )
        and _all_entities_are_standard_protein(candidate)
    ):
        source = "cyclicmpnn"
    elif _all_entities_are_standard_protein(candidate):
        source = "proteinmpnn"
    elif _has_nonprotein_entities(atoms) and protein_proxy_error is None:
        source = "ligandmpnn"
    else:
        source = "design"

    if source == "design":
        return source
    if len(design_chains) != 1:
        raise ValueError(f"{source} requires exactly one design protein chain")
    if protein_proxy_error is not None:
        raise ValueError(protein_proxy_error)
    if any(
        _token_model_residue(candidate, token) not in _AA3_TO_1
        for token in tokens
        if token.is_design
    ):
        raise ValueError(f"{source} requires canonical or safely proxied design residues")
    if source == "proteinmpnn" and not _all_entities_are_standard_protein(candidate):
        raise ValueError("proteinmpnn accepts only all-standard-protein systems")
    if source == "cyclicmpnn":
        if has_modified or not _all_entities_are_standard_protein(candidate):
            raise ValueError("cyclicmpnn accepts only standard-protein context")
        if not (
            has_required_head_to_tail_single_bond(candidate, design_chains[0])
            or has_cyclic_offset(candidate, design_chains[0])
        ):
            raise ValueError(
                "cyclicmpnn requires a required-output head-to-tail single bond "
                "or explicit cyclic offset intent"
            )
    return source


def _identity_dependent_bond_tokens(candidate: SearchCandidate) -> set[int]:
    atoms = candidate.atoms
    uid_to_row = {int(uid): row for row, uid in enumerate(atoms.atom_uid)}
    result: set[int] = set()
    for condition in candidate.topology_conditions:
        for atom_uid in (condition.atom_uid_a, condition.atom_uid_b):
            row = uid_to_row.get(int(atom_uid))
            if row is None or str(atoms.mol_type[row]) != "protein":
                continue
            residue = str(atoms.res_name[row])
            atom_name = str(atoms.atom_name[row])
            if residue in PROTEIN_RES_ATOMS and atom_name not in _PROTEIN_BACKBONE:
                result.add(int(atoms.token_uid[row]))
    return result


def _na_topology_tokens(candidate: SearchCandidate) -> set[int]:
    """Return every design NA token touched by a declared topology condition."""

    atoms = candidate.atoms
    uid_to_row = {int(uid): row for row, uid in enumerate(atoms.atom_uid)}
    result: set[int] = set()
    for condition in candidate.topology_conditions:
        for atom_uid in (condition.atom_uid_a, condition.atom_uid_b):
            row = uid_to_row.get(int(atom_uid))
            if row is None:
                continue
            if (
                bool(atoms.is_design[row])
                and str(atoms.mol_type[row]) in {"rna", "dna"}
            ):
                result.add(int(atoms.token_uid[row]))
    return result


def _na_fixed_positions(
    candidate: SearchCandidate,
    tokens: Sequence[ProteinToken],
) -> set[tuple[str, int]]:
    atoms = candidate.atoms
    topology_tokens = _na_topology_tokens(candidate)
    fixed: set[tuple[str, int]] = set()
    for token in tokens:
        rows = np.flatnonzero(
            np.asarray(atoms.token_uid, dtype=np.int64) == int(token.token_uid)
        )
        if not len(rows):
            continue
        motif = bool(
            np.asarray(atoms.motif_coord_condition[rows], dtype=bool).any()
            or np.asarray(atoms.token_center_condition[rows], dtype=bool).any()
        )
        if (
            not token.is_design
            or token.identity_conditioned
            or motif
            or token.token_uid in topology_tokens
        ):
            fixed.add((token.chain_id, token.position))
    return fixed


def plan_sequence_source(
    candidate: SearchCandidate,
    *,
    requested: str,
    user_fixed_positions: Sequence[str] = (),
    fixed_interface: Mapping[str, Any] | None = None,
) -> SequenceSourcePlan:
    source = resolve_sequence_source(candidate, requested)
    if candidate.base_request.task.format == "antibody" and fixed_interface:
        raise ValueError("antibody tasks use explicit mutable residue selection; fixed_interface is unsupported")
    if source == "na_mpnn":
        design_chain, tokens = _validate_na_candidate(candidate)
        sequences = _polymer_sequences(candidate, tokens)
        fixed_values = _na_fixed_positions(candidate, tokens)
    else:
        tokens = protein_tokens(candidate.atoms)
        sequences = _protein_sequences(candidate, tokens) if tokens else {}
        design_chains = _design_protein_chains(tokens)
        design_chain = design_chains[0] if len(design_chains) == 1 else None
        identity_bond_tokens = _identity_dependent_bond_tokens(candidate)
        fixed_values = {
            (token.chain_id, token.position)
            for token in tokens
            if (not token.is_design)
            or token.identity_conditioned
            or token.token_uid in identity_bond_tokens
        }
    known_tokens = tokens
    if source == "design" and _na_design_is_present(candidate.atoms):
        # The finalized design path does not run an MPNN, but retaining its
        # generic polymer positions lets a user-supplied fixed selector remain
        # candidate-local instead of turning an otherwise valid NA fallback
        # into an unrelated protein-ledger error.
        known_tokens = polymer_tokens(candidate.atoms)
    known = {(token.chain_id, token.position) for token in known_tokens}
    for raw in user_fixed_positions:
        chain, separator, position_text = str(raw).partition(":")
        if not separator or not position_text.isdigit() or int(position_text) < 1:
            raise ValueError(f"invalid MPNN fixed position {raw!r}")
        key = (chain, int(position_text))
        if key not in known:
            raise ValueError(f"MPNN fixed position does not resolve: {raw!r}")
        fixed_values.add(key)
    interface_positions, interface_target_chains = interface_fixed_positions(
        candidate,
        design_chain,
        fixed_interface,
    )
    fixed_values.update(interface_positions)
    fixed = tuple(sorted(fixed_values))
    return SequenceSourcePlan(
        source,
        design_chain,
        sequences,
        fixed,
        interface_positions,
        interface_target_chains,
    )


def score_confidences(
    values: Sequence[ESMFold2Confidence],
    *,
    monomer_plddt_weight: float,
    complex_plddt_weight: float,
    complex_iptm_weight: float,
    device: torch.device | str = "cpu",
) -> ESMFold2ScoreBatch:
    candidate_ids = tuple(value.candidate_id for value in values)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("ESMFold2 confidence candidate IDs must be unique")
    plddt_values: list[float] = []
    interface_values: list[float] = []
    total_values: list[float] = []
    for value in values:
        plddt = float(value.design_plddt)
        if not np.isfinite(plddt) or not 0 <= plddt <= 100:
            raise ValueError("ESMFold2 design pLDDT must be finite in [0,100]")
        confidence = plddt / 100.0
        if value.interface_iptm is None:
            interface = 0.0
            total = float(monomer_plddt_weight) * confidence
        else:
            interface = float(value.interface_iptm)
            if not np.isfinite(interface) or not 0 <= interface <= 1:
                raise ValueError("ESMFold2 interface ipTM must be finite in [0,1]")
            total = (
                float(complex_plddt_weight) * confidence
                + float(complex_iptm_weight) * interface
            )
        plddt_values.append(confidence)
        interface_values.append(interface)
        total_values.append(total)
    return ESMFold2ScoreBatch(
        candidate_ids,
        torch.tensor(total_values, dtype=torch.float32, device=device),
        torch.tensor(plddt_values, dtype=torch.float32, device=device),
        torch.tensor(interface_values, dtype=torch.float32, device=device),
    )


class ESMFold2Reward:
    """Configuration marker; heavy scoring is deferred until FM offload."""

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)


__all__ = [
    "ESMFold2Confidence",
    "ESMFold2Reward",
    "ESMFold2ScoreBatch",
    "ProteinToken",
    "SequenceSourcePlan",
    "has_cyclic_offset",
    "has_required_head_to_tail_single_bond",
    "interface_fixed_positions",
    "normalize_fixed_interface",
    "plan_sequence_source",
    "polymer_tokens",
    "protein_tokens",
    "resolve_sequence_source",
    "score_confidences",
]
