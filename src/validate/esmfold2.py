"""One-variant-at-a-time ESMFold2 validation adapter."""
from __future__ import annotations

import copy
from dataclasses import replace as dataclass_replace
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from partidesign.paths import (community_source, community_asset, asset_root, cache_dir, offline_site, code_root, ENV_MODEL_DIR)
from typing import Any, Iterator, Mapping

from .sequences import _decode_jsonish, _field, _initial_sequences, _row_field
from .ligands import (LigandResolutionError, resolve_context_ligands,
                      resolve_context_ligand_components)


class ESMFold2Error(RuntimeError):
    """An ESMFold2 validation request or result is invalid."""


_RECEPTOR_SEGMENT_LEDGER = "partidesign_receptor_segments"


def _canonical_chain_values(atoms: Any) -> Any:
    """Return canonical label_asym_id values when available."""

    import numpy as np

    labels = getattr(atoms, "label_asym_id", None)
    if labels is not None:
        values = np.asarray(labels).astype(str)
        if values.size and all(value not in {"", ".", "?"} for value in values):
            return values
    return np.asarray(atoms.chain_id).astype(str)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _chain_order(atoms: Any) -> tuple[str, ...]:
    result: list[str] = []
    for value in (str(item) for item in _canonical_chain_values(atoms)):
        if value not in result:
            result.append(value)
    return tuple(result)


def _token_positions(atoms: Any) -> dict[tuple[str, int], int]:
    import numpy as np

    token_values = getattr(atoms, "token_uid", None)
    if token_values is None:
        raise ESMFold2Error("ESMFold2 input requires a finalized token_uid ledger")
    chains = _canonical_chain_values(atoms)
    tokens = np.asarray(token_values, dtype=np.int64)
    if bool((tokens < 0).any()):
        raise ESMFold2Error("ESMFold2 input requires finalized token_uid values")
    result: dict[tuple[str, int], int] = {}
    for chain in _chain_order(atoms):
        rows = np.flatnonzero(chains == chain)
        ligand = np.asarray(atoms.mol_type).astype(str)[rows] == "ligand"
        values = sorted({int(tokens[row]) for row in rows[~ligand]})
        result.update({(chain, token): position for position, token in enumerate(values)})
        if not bool(ligand.any()):
            continue
        component_values = getattr(atoms, "component_uid", None)
        if component_values is None:
            raise ESMFold2Error("ligand input requires a finalized component_uid ledger")
        components = np.asarray(component_values, dtype=np.int64)
        for offset, component in enumerate(dict.fromkeys(components[rows[ligand]].tolist())):
            for row in rows[ligand & (components[rows] == component)]:
                result[(chain, int(tokens[row]))] = len(values) + offset
    return result


def _mapped_atom_name(atoms: Any, row: int, ligand_atom_names: Mapping[int, str] | None) -> str:
    raw = str(atoms.atom_name[row])
    if not ligand_atom_names:
        return raw
    modalities = getattr(atoms, "mol_type", None)
    if modalities is None or str(modalities[row]) != "ligand":
        return raw
    uids = getattr(atoms, "atom_uid", None)
    key = int(uids[row]) if uids is not None else int(row)
    return str(ligand_atom_names.get(key, raw))


def _mapped_locator_name(
    atoms: Any,
    chain: str,
    position: int,
    atom_name: str,
    positions: Mapping[tuple[str, int], int],
    ligand_atom_names: Mapping[int, str] | None,
) -> str:
    if not ligand_atom_names:
        return str(atom_name)
    token_values = [
        token for (candidate_chain, token), candidate_position in positions.items()
        if candidate_chain == str(chain) and int(candidate_position) == int(position)
    ]
    if not token_values:
        return str(atom_name)
    chains = _canonical_chain_values(atoms)
    tokens = __import__("numpy").asarray(atoms.token_uid, dtype="int64")
    names = __import__("numpy").asarray(atoms.atom_name).astype(str)
    rows = __import__("numpy").flatnonzero(
        (chains == str(chain)) & __import__("numpy").isin(tokens, token_values)
        & (names == str(atom_name))
    )
    if len(rows) != 1:
        return str(atom_name)
    return _mapped_atom_name(atoms, int(rows[0]), ligand_atom_names)


def _endpoint(
    atoms: Any, value: Any, positions: Mapping[tuple[str, int], int],
    ligand_atom_names: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    import numpy as np

    atom_uid = None
    if isinstance(value, Mapping):
        # ``atom_id`` in result.csv is a one-based source row/serial locator;
        # only an explicit AtomUID/uid can bypass structure resolution.
        for key in ("atom_uid", "uid"):
            if key in value:
                try:
                    atom_uid = int(value[key])
                except (TypeError, ValueError) as exc:
                    raise ESMFold2Error("bond endpoint atom UID is invalid") from exc
                break
    else:
        try:
            atom_uid = int(value)
        except (TypeError, ValueError):
            atom_uid = None
    if atom_uid is not None:
        uids = np.asarray(atoms.atom_uid, dtype=np.int64)
        rows = np.flatnonzero(uids == atom_uid)
        if len(rows) != 1:
            raise ESMFold2Error(f"bond endpoint AtomUID {atom_uid} is not unique")
        row = int(rows[0])
        canonical_chains = _canonical_chain_values(atoms)
        chain, token = str(canonical_chains[row]), int(atoms.token_uid[row])
        return {
            "chain_id": chain,
            "residue_index": int(positions[(chain, token)]),
            "atom_name": _mapped_atom_name(atoms, row, ligand_atom_names),
        }
    if isinstance(value, Mapping):
        chain = value.get("chain_id", value.get("chain"))
        atom_name = value.get("atom_name", value.get("atom"))
        if chain is not None and atom_name is not None:
            if "residue_index" in value or (
                "position" in value and value.get("res_id") in (None, "", ".", "?")
            ):
                position = (
                    int(value["residue_index"])
                    if "residue_index" in value
                    else int(value["position"]) - 1
                )
                if position < 0:
                    raise ESMFold2Error("bond endpoint residue position is negative")
                return {
                    "chain_id": str(chain),
                    "residue_index": position,
                    "atom_name": _mapped_locator_name(
                        atoms, str(chain), position, str(atom_name),
                        positions, ligand_atom_names,
                    ),
                }
            # result.csv bond constraints use one-based atom_site/PDB serials
            # and label residue locators. Resolve that public ledger against
            # the parsed structure before producing ESMFold2's zero-based
            # token endpoint.
            if value.get("atom_id") not in (None, "", ".", "?"):
                try:
                    row = int(value["atom_id"]) - 1
                except (TypeError, ValueError) as exc:
                    raise ESMFold2Error("bond endpoint atom_id is invalid") from exc
                if row < 0 or row >= len(atoms.atom_uid):
                    raise ESMFold2Error("bond endpoint atom_id is out of range")
            else:
                chains = np.asarray(
                    getattr(atoms, "label_asym_id", getattr(atoms, "chain_id"))
                ).astype(str)
                mask = chains == str(chain)
                if value.get("res_id") not in (None, "", ".", "?"):
                    mask &= np.asarray(atoms.res_id, dtype=np.int64) == int(value["res_id"])
                if value.get("ins_code") not in (None, "", ".", "?"):
                    mask &= np.asarray(atoms.ins_code).astype(str) == str(value["ins_code"])
                mask &= np.asarray(atoms.atom_name).astype(str) == str(atom_name)
                rows = np.flatnonzero(mask)
                if len(rows) != 1:
                    raise ESMFold2Error("bond endpoint locator is not unique")
                row = int(rows[0])
            resolved_chain = str(_canonical_chain_values(atoms)[row])
            resolved_token = int(atoms.token_uid[row])
            return {
                "chain_id": resolved_chain,
                "residue_index": int(positions[(resolved_chain, resolved_token)]),
                "atom_name": _mapped_atom_name(atoms, row, ligand_atom_names),
            }
        if chain is not None and atom_name is None:
            raise ESMFold2Error("bond endpoint has no atom name")
        if atom_name is not None and chain is None:
            raise ESMFold2Error("bond endpoint has no chain")
        if "residue_index" in value or "position" in value:
            chain = value.get("chain_id", value.get("chain"))
            atom_name = value.get("atom_name", value.get("atom"))
            if chain is None or atom_name is None:
                raise ESMFold2Error("bond endpoint lacks chain or atom name")
            position = (
                int(value["residue_index"])
                if "residue_index" in value
                else int(value["position"]) - 1
            )
            if position < 0:
                raise ESMFold2Error("bond endpoint residue position is negative")
            return {
                "chain_id": str(chain),
                "residue_index": position,
                "atom_name": _mapped_locator_name(
                    atoms, str(chain), position, str(atom_name),
                    positions, ligand_atom_names,
                ),
            }
    raise ESMFold2Error("bond endpoint lacks an AtomUID or chain position")


def _bond_payload(
    input_record: Any,
    atoms: Any,
    *,
    ligand_atom_names: Mapping[int, str] | None = None,
    ligand_entries: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    raw = _field(input_record, "covalent_bonds", None)
    if raw is None:
        raw = _field(input_record, "bond_constraints", ())
    raw = _decode_jsonish(raw, default=())
    if isinstance(raw, Mapping):
        raw = raw.get("bonds", raw.get("constraints", ()))
    if not raw:
        # ``read_structure`` retains the source CIF/PDB bond graph.  Result
        # rows from older inference runs may not have a bond_constraints
        # column, so recover only non-template edges here: standard adjacent
        # peptide C-N links are inferred by ESMFold2, while cyclic,
        # cross-chain, disulfide, and ligand edges remain explicit.
        import numpy as np
        source = getattr(atoms, "bonds", None)
        if source is not None:
            chains = _canonical_chain_values(atoms)
            tokens = np.asarray(atoms.token_uid, dtype=np.int64)
            atom_names = np.asarray(atoms.atom_name).astype(str)
            modalities = np.asarray(atoms.mol_type).astype(str)
            molecule_uids = getattr(atoms, "molecule_uid", None)
            if molecule_uids is not None:
                molecule_uids = np.asarray(molecule_uids, dtype=np.int64)
            component_uids = getattr(atoms, "component_uid", None)
            if component_uids is not None:
                component_uids = np.asarray(component_uids, dtype=np.int64)
            if ligand_entries is None:
                ligand_entries = _field(input_record, "ligand_entries", {}) or {}
            row_smiles = _row_field(input_record, "design_sequence", "")
            values: list[dict[str, Any]] = []
            for left, right, order in source.as_array().tolist():
                i, j = int(left), int(right)
                if int(tokens[i]) == int(tokens[j]):
                    # Residue/CCD-internal chemistry is reconstructed by the
                    # native input builder and cannot be represented as a
                    # token-to-token ESMFold2 restraint.
                    continue
                if modalities[i] == modalities[j] == "ligand":
                    same_molecule = (
                        molecule_uids is not None
                        and int(molecule_uids[i]) == int(molecule_uids[j])
                    )
                    if same_molecule:
                        same_component = (
                            component_uids is None
                            or int(component_uids[i]) == int(component_uids[j])
                        )
                        # A private SMILES or one CCD component recreates the
                        # internal graph.  A bond between CCD components is
                        # retained for the native chemistry ledger.
                        chain = str(chains[i])
                        current_entry = (
                            ligand_entries.get(chain)
                            if isinstance(ligand_entries, Mapping)
                            else None
                        )
                        has_smiles = bool(
                            isinstance(current_entry, Mapping)
                            and str(current_entry.get("smiles", "")).strip()
                        )
                        design_values = getattr(atoms, "is_design", None)
                        has_smiles = has_smiles or bool(
                            modalities[i] == "ligand"
                            and design_values is not None
                            and bool(np.asarray(design_values, dtype=bool)[i])
                            and isinstance(row_smiles, str)
                            and row_smiles.strip()
                        )
                        if same_component or has_smiles:
                            continue
                same_chain = chains[i] == chains[j]
                adjacent_peptide = (
                    same_chain and abs(int(tokens[i]) - int(tokens[j])) == 1
                    and {atom_names[i], atom_names[j]} == {"C", "N"}
                )
                adjacent_nucleic = (
                    same_chain and abs(int(tokens[i]) - int(tokens[j])) == 1
                    and {atom_names[i], atom_names[j]} == {"O3'", "P"}
                )
                if adjacent_peptide or adjacent_nucleic:
                    continue
                values.append({
                    "endpoint1": {"atom_uid": int(atoms.atom_uid[i])},
                    "endpoint2": {"atom_uid": int(atoms.atom_uid[j])},
                    "bond_order": {1: "single", 2: "double", 3: "triple", 4: "aromatic"}.get(int(order), "?"),
                })
            raw = values
    if not isinstance(raw, (list, tuple)):
        return []
    positions = _token_positions(atoms)
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        if item.get("is_covalent") is False or str(item.get("kind", "covalent")).lower() in {
            "noncovalent", "coordination", "contact",
        }:
            continue
        left = item.get(
            "endpoint1", item.get("atom1", item.get("atom_a", item.get("atom_uid_a")))
        )
        right = item.get(
            "endpoint2", item.get("atom2", item.get("atom_b", item.get("atom_uid_b")))
        )
        if left is None or right is None:
            raise ESMFold2Error("covalent bond lacks both endpoints")
        order = str(item.get("bond_order", item.get("order", item.get("type", "?")))).lower()
        order = {"sing": "single", "doub": "double", "trip": "triple",
                 "arom": "aromatic", "any": "?", "unknown": "?"}.get(order, order)
        if order not in {"?", "single", "double", "triple", "aromatic"}:
            order = "?"
        result.append({
            "endpoint1": _endpoint(atoms, left, positions, ligand_atom_names),
            "endpoint2": _endpoint(atoms, right, positions, ligand_atom_names),
            "bond_order": order,
        })
    return result


def _private_segment_id(chain: str, index: int, used: set[str]) -> str:
    # ESMFold2's CIF writer truncates label_asym_id values at 16 characters.
    tag = f"__ef2b{index}"
    candidate = f"{chain[:16 - len(tag)]}{tag}"
    suffix = 1
    while candidate in used:
        collision = f"_{suffix}"
        candidate = f"{chain[:16 - len(tag) - len(collision)]}{tag}{collision}"
        suffix += 1
    used.add(candidate)
    return candidate


def _read_native_confidence_arrays(path: Path) -> dict[str, Any]:
    """Read native token confidence once, preserving the metrics PAE field name."""
    import numpy as np

    # Atom confidence is not consumed by source token metrics; leave those
    # larger arrays in the untouched private native archive.
    keys = ("plddt", "plddt_ca", "pae", "pde", "token_index", "residue_index", "asym_id",
            "entity_id", "sym_id", "token_attention_mask")
    with np.load(path, allow_pickle=False) as archive:
        try:
            return {"pae_matrix" if key == "pae" else key: archive[key] for key in keys}
        except KeyError as exc:
            raise ESMFold2Error(f"native ESMFold2 token confidence field is missing: {exc}") from exc


def _logical_confidence_arrays(
    arrays: Mapping[str, Any], payload: Mapping[str, Any], confidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from src.design.rewards.esmfold2_mapping import logical_confidence_arrays

    try:
        return logical_confidence_arrays(arrays, payload, confidence)
    except ValueError as exc:
        raise ESMFold2Error(str(exc)) from exc


def _generic_payload(
    input_record: Any,
    variant: Mapping[str, Any],
    *,
    resolved_context_ligands: tuple[Mapping[str, Any], Mapping[int, str]] | None = None,
) -> dict[str, Any]:
    import numpy as np

    atoms = _field(input_record, "atoms")
    if atoms is None:
        raise ESMFold2Error("validation input has no parsed atoms")
    chains = _canonical_chain_values(atoms)
    residue_names = np.asarray(atoms.res_name).astype(str)
    modalities = np.asarray(atoms.mol_type).astype(str)
    design = tuple(str(value) for value in (_field(input_record, "design_chains", ()) or ()))
    if not design:
        mask = np.asarray(getattr(atoms, "is_design"), dtype=bool)
        design = tuple(chain for chain in _chain_order(atoms)
                       if bool(np.any(mask & (chains == chain))))
    if not design:
        raise ESMFold2Error("ESMFold2 validation input has no design chain")
    sequences = dict(variant.get("sequences", {}))
    initial = _initial_sequences(input_record)
    # MPNN uses canonical parent letters. Preserve fixed CCD occurrences in
    # refold input, rather than silently treating those letters as real restypes.
    modified_candidate = None
    from .sequences import _AA3_TO_1, _candidate_adapter, _fixed_modified_map

    modified_names, _ = _fixed_modified_map(input_record)
    configured_ligands = _field(input_record, "ligand_entries", {}) or {}
    ligands = dict(configured_ligands) if isinstance(configured_ligands, Mapping) else {}
    ligand_atom_names: dict[int, str] = {}
    context_ligand_chains = [
        chain for chain in _chain_order(atoms)
        if chain not in design
        and bool(np.any((chains == chain) & (modalities == "ligand")))
    ]
    if resolved_context_ligands is not None:
        ligands.update(resolved_context_ligands[0])
        ligand_atom_names.update(resolved_context_ligands[1])
    elif context_ligand_chains:
        task_dir = _field(input_record, "task_dir", None)
        if task_dir is None:
            raise ESMFold2Error(
                "context ligand validation requires the task resolved_config.yaml"
            )
        try:
            resolved_ligands, ligand_atom_names = resolve_context_ligands(
                Path(task_dir),
                atoms,
                design_chains=design,
                chains=context_ligand_chains,
            )
        except LigandResolutionError as exc:
            raise ESMFold2Error(str(exc)) from exc
        ligands.update(resolved_ligands)
    row_design_sequence = _row_field(input_record, "design_sequence", "")
    entity_values = getattr(atoms, "entity_uid", None)
    if entity_values is None:
        entity_values = np.arange(len(chains), dtype=np.int64)
    else:
        entity_values = np.asarray(entity_values, dtype=np.int64)
    label_entities = getattr(atoms, "label_entity_id", None)
    entity_index: dict[int, int] = {}
    entity_copies: dict[int, int] = {}
    entries: list[dict[str, Any]] = []
    for chain in _chain_order(atoms):
        rows = np.flatnonzero(chains == chain)
        kinds = {str(value) for value in modalities[rows]}
        if len(kinds) != 1:
            raise ESMFold2Error(f"chain {chain!r} contains mixed modalities")
        modality = next(iter(kinds))
        entity_key = int(entity_values[int(rows[0])])
        if entity_key not in entity_index:
            entity_index[entity_key] = len(entity_index) + 1
            entity_copies[entity_key] = 0
        entity_copies[entity_key] += 1
        entry: dict[str, Any] = {
            "chain_id": chain, "modality": modality,
            "role": "design" if chain in design else "context",
            "modifications": [],
            "model_entity_index": entity_index[entity_key],
            "model_entity_copy": entity_copies[entity_key],
            "output_entity_id": (
                str(label_entities[int(rows[0])])
                if label_entities is not None and str(label_entities[int(rows[0])]).strip()
                else f"entity{entity_index[entity_key]}"
            ),
        }
        if modality == "ligand":
            ligand = ligands.get(chain) if isinstance(ligands, Mapping) else None
            if isinstance(ligand, Mapping):
                entry.update(copy.deepcopy(dict(ligand)))
            elif chain in design and isinstance(row_design_sequence, str) and row_design_sequence.strip():
                entry["smiles"] = row_design_sequence.strip()
            else:
                raise ESMFold2Error(f"ligand_entries is missing chain {chain!r}")
        elif modality in {"protein", "rna", "dna"}:
            sequence = sequences.get(chain, initial.get(chain))
            if not isinstance(sequence, str) or not sequence:
                raise ESMFold2Error(f"missing sequence for polymer chain {chain!r}")
            residue_uids = getattr(atoms, "residue_uid", None)
            has_modified_residue = bool(
                modified_names
                and residue_uids is not None
                and any(int(residue_uids[row]) in modified_names for row in rows)
            )
            has_nonstandard_context_residue = (
                chain not in design
                and modality == "protein"
                and bool({str(value).strip().upper() for value in residue_names[rows]}
                         .difference(_AA3_TO_1))
            )
            if modified_candidate is None and (
                has_modified_residue or has_nonstandard_context_residue
            ):
                modified_candidate = _candidate_adapter(
                    input_record, str(_field(input_record, "sample_id", "sample"))
                )
                modified_names = dict(
                    getattr(modified_candidate, "fixed_modified_residue_names", {})
                )
                has_modified_residue = bool(
                    residue_uids is not None
                    and any(int(residue_uids[row]) in modified_names for row in rows)
                )
            if modified_candidate is not None and has_modified_residue:
                if residue_uids is None:
                    raise ESMFold2Error(
                        f"modified chain {chain!r} has no residue UID ledger"
                    )
                from src.design.rewards.esmfold2_input import _polymer_payload

                try:
                    native_sequence, modifications = _polymer_payload(
                        modified_candidate, chain, modality
                    )
                except ValueError as exc:
                    raise ESMFold2Error(str(exc)) from exc
                if len(sequence) != len(native_sequence):
                    raise ESMFold2Error(
                        f"sequence length changed for modified chain {chain!r}"
                    )
                native_modifications: list[dict[str, Any]] = []
                for modification in modifications:
                    position = int(modification["position"])
                    component = str(modification.get("ccd", "")).upper()
                    if chain not in design and component == "X":
                        # ``X`` is an unsupported source component, not a CCD
                        # entry. Keep the candidate-local ALA/GLY proxy.
                        if sequence[position] == "X":
                            sequence = sequence[:position] + native_sequence[position] + sequence[position + 1:]
                        if sequence[position] != native_sequence[position]:
                            raise ESMFold2Error(
                                "sequence generator changed an unsupported "
                                f"receptor position {chain}:{position + 1}"
                            )
                        continue
                    if chain not in design and sequence[position] == "X":
                        sequence = sequence[:position] + native_sequence[position] + sequence[position + 1:]
                    if sequence[position] != native_sequence[position]:
                        raise ESMFold2Error(
                            "sequence generator changed a fixed modified "
                            f"position {chain}:{position + 1}"
                        )
                    native_modifications.append(modification)
                entry["modifications"] = native_modifications
            entry["sequence"] = sequence
            if modality in {"protein", "rna"}:
                entry["msa"] = None
        else:
            raise ESMFold2Error(f"unsupported ESMFold2 modality {modality!r}")
        entries.append(entry)
    covalent_bonds = _bond_payload(
        input_record, atoms, ligand_atom_names=ligand_atom_names,
        ligand_entries=ligands,
    )
    payload = {
        "schema_version": 1,
        "sample_id": str(_field(input_record, "sample_id", "sample")),
        "variant_id": f"seq{int(variant.get('seq_index', 0))}",
        "design_chain": design[0], "design_chains": list(design),
        "msa_policy": "none", "sequences": entries,
        "covalent_bonds": covalent_bonds,
    }
    from src.design.rewards.esmfold2_mapping import apply_source_mapping

    try:
        return apply_source_mapping(payload, atoms, design_chains=design)
    except ValueError as exc:
        raise ESMFold2Error(str(exc)) from exc



def _mixed_context_payload(input_record: Any, variant: Mapping[str, Any]) -> dict[str, Any] | None:
    """Split heterogeneous context chains privately, keeping all components and links."""
    import numpy as np

    atoms = _field(input_record, "atoms")
    chains = _canonical_chain_values(atoms)
    kinds = np.asarray(atoms.mol_type).astype(str)
    design = tuple(_field(input_record, "design_chains", ()) or ())
    mixed = [chain for chain in _chain_order(atoms) if chain not in design
             and len(set(kinds[chains == chain])) > 1]
    if not mixed:
        return None
    for chain in mixed:
        if len(set(kinds[chains == chain]).difference({"ligand"})) != 1:
            raise ESMFold2Error(f"chain {chain!r} contains multiple polymer modalities")
    task = _field(input_record, "task_dir")
    if task is None:
        raise ESMFold2Error("mixed context requires the task resolved_config.yaml")
    context_ligands = [chain for chain in _chain_order(atoms) if chain not in design
                       and bool(np.any((chains == chain) & (kinds == "ligand")))]
    try:
        components, atom_names = resolve_context_ligand_components(
            Path(task), atoms, design_chains=design, chains=context_ligands)
    except LigandResolutionError as exc:
        raise ESMFold2Error(str(exc)) from exc
    positions = _token_positions(atoms)
    private = atoms.copy()
    private_chains = chains.copy().astype(object)
    component_uids = np.asarray(atoms.component_uid, dtype=np.int64)
    used = set(_chain_order(atoms))
    ligands: dict[str, Any] = {}
    splits: dict[str, tuple[str, int, int]] = {}
    source_rows: dict[str, dict[str, Any]] = {}
    from src.design.utils.source_mapping import source_locator
    from src.design.rewards.esmfold2_mapping import _source_metadata

    endpoint_map: dict[tuple[str, int], tuple[str, int]] = {}
    for chain in context_ligands:
        rows = np.flatnonzero((chains == chain) & (kinds == "ligand"))
        for index, uid in enumerate(dict.fromkeys(component_uids[rows].tolist())):
            selected = rows[component_uids[rows] == uid]
            physical = _private_segment_id(chain, index + 1, used) if chain in mixed else (
                chain if index == 0 else _private_segment_id(chain, index + 1, used))
            private_chains[selected] = physical
            ligands[physical] = components[(chain, int(uid))]
            logical_position = positions[(chain, int(atoms.token_uid[selected[0]]))]
            endpoint_map[(chain, logical_position)] = (physical, 0)
            splits[physical] = (chain, logical_position, logical_position + 1)
            source_rows[physical] = {
                "physical_chain_id": physical, "logical_chain_id": chain,
                "logical_start": logical_position, "logical_stop": logical_position + 1,
                "observed_positions": [logical_position],
                "source_locators": [source_locator(atoms, int(selected[0]))],
                "source_metadata": [_source_metadata(atoms, selected.tolist())],
            }
    private.set_annotation("chain_id", np.asarray(private_chains, dtype=str))
    private.set_annotation("label_asym_id", np.asarray(private_chains, dtype=str))
    public_bonds = _bond_payload(input_record, atoms, ligand_atom_names=atom_names)
    bonds = copy.deepcopy(public_bonds)
    for bond in bonds:
        for name in ("endpoint1", "endpoint2"):
            endpoint = bond[name]
            mapped = endpoint_map.get((endpoint["chain_id"], endpoint["residue_index"]))
            if mapped is not None:
                endpoint["chain_id"], endpoint["residue_index"] = mapped
    record = {name: _field(input_record, name, None) for name in (
        "task_dir", "sample_id", "row", "fixed_restypes", "fixed_positions",
        "motif_positions", "tip_positions", "cyclic_chains", "cyclic_offset_chains", "contig_string", "tip_atoms")}
    record.update(atoms=private, design_chains=design, covalent_bonds=bonds,
                  ligand_entries=_field(input_record, "ligand_entries", {}))
    payload = _generic_payload(record, variant,
        resolved_context_ligands=(ligands, atom_names))
    old_ledger = payload.get(_RECEPTOR_SEGMENT_LEDGER, {})
    logical = copy.deepcopy(old_ledger.get("logical_sequences", payload["sequences"]))
    # The original chain appears once in confidence/role metadata. Attached
    # components are model entries, not additional public design chains.
    removed = set(splits).difference(_chain_order(atoms))
    logical = [entry for entry in logical if entry["chain_id"] not in removed]
    logical_by_chain = {entry["chain_id"]: entry for entry in logical}
    logical = [logical_by_chain[chain] for chain in _chain_order(atoms)
               if chain in logical_by_chain]
    for physical, (chain, start, stop) in splits.items():
        if chain in mixed or physical != chain:
            component_entry = next(entry for entry in payload["sequences"]
                                   if entry["chain_id"] == physical)
            logical_by_chain[chain].setdefault("context_components", []).append(
                {"position": start, "entry": copy.deepcopy(component_entry)})
    segment_rows = [copy.deepcopy(row) for row in old_ledger.get("segments", [])
                    if row["physical_chain_id"] not in splits]
    segment_rows.extend(source_rows.values())
    # Distinct physical entities are required even when a source output grouped
    # protein and heterogens under one logical entity identifier.
    entity_indices: dict[tuple[Any, ...], int] = {}
    entity_copies: dict[tuple[Any, ...], int] = {}
    for entry in payload["sequences"]:
        key = (entry["output_entity_id"], entry["modality"], entry.get("sequence"),
               tuple(entry.get("ccd", ())), entry.get("smiles"))
        if key not in entity_indices:
            entity_indices[key] = len(entity_indices) + 1
            entity_copies[key] = 0
        entity_copies[key] += 1
        entry["model_entity_index"] = entity_indices[key]
        entry["model_entity_copy"] = entity_copies[key]
        entry["output_entity_id"] = f"entity{entity_indices[key]}"
    payload[_RECEPTOR_SEGMENT_LEDGER] = {
        "mapping_version": 2,
        "logical_sequences": logical, "logical_covalent_bonds": public_bonds,
        "segments": segment_rows, "mixed_context": True,
        "source_polymer_sequences": old_ledger.get("source_polymer_sequences", {}),
        "source_polymer_sequence_values": old_ledger.get("source_polymer_sequence_values", []),
        "preserved_backbone_edges": old_ledger.get("preserved_backbone_edges", []),
        "mixed_component_chains": [chain for chain in splits if splits[chain][0] in mixed],
    }
    return payload


def _restore_mixed_output(path: Path, payload: Mapping[str, Any]) -> None:
    from src.design.rewards.esmfold2_mapping import restore_source_output

    try:
        restore_source_output(path, payload)
    except ValueError as exc:
        raise ESMFold2Error(str(exc)) from exc


def _build_payload(
    input_record: Any,
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = _field(input_record, "candidate", None)
    if candidate is not None:
        from src.design.rewards.esmfold2 import SequenceSourcePlan
        from src.design.rewards.esmfold2_input import build_esmfold2_input

        design = tuple(str(value) for value in (_field(input_record, "design_chains", ()) or ()))
        if not design:
            raise ESMFold2Error("candidate ESMFold2 input has no design chain")
        plan = SequenceSourcePlan(
            source=str(variant.get("source", "gt")), design_chain=design[0],
            chain_sequences=dict(variant.get("sequences", {})), fixed_positions=tuple(),
        )
        payload = build_esmfold2_input(candidate, plan)
        payload["sample_id"] = str(_field(input_record, "sample_id", candidate.candidate_id))
        payload["variant_id"] = f"seq{int(variant.get('seq_index', 0))}"
        return payload
    supplied = _field(input_record, "esmfold2_payload", None)
    if supplied is None:
        supplied = _field(input_record, "fold_payload", None)
    if isinstance(supplied, Mapping):
        payload = copy.deepcopy(dict(supplied))
        payload["sample_id"] = str(_field(input_record, "sample_id", payload.get("sample_id", "sample")))
        payload["variant_id"] = f"seq{int(variant.get('seq_index', 0))}"
        from src.design.rewards.esmfold2_mapping import apply_source_mapping, update_mapped_sequences

        atoms = _field(input_record, "atoms")
        if _RECEPTOR_SEGMENT_LEDGER in payload:
            try:
                return update_mapped_sequences(payload, atoms, variant.get("sequences", {}))
            except ValueError as exc:
                raise ESMFold2Error(str(exc)) from exc
        values = variant.get("sequences")
        if isinstance(values, Mapping):
            for entry in payload.get("sequences", ()):
                if isinstance(entry, dict) and entry.get("chain_id") in values:
                    entry["sequence"] = str(values[entry["chain_id"]])
        atoms = _field(input_record, "atoms")
        design = tuple(str(value) for value in payload.get("design_chains", ()))
        try:
            return apply_source_mapping(payload, atoms, design_chains=design)
        except ValueError as exc:
            raise ESMFold2Error(str(exc)) from exc
    mixed = _mixed_context_payload(input_record, variant)
    return mixed if mixed is not None else _generic_payload(input_record, variant)


def _runtime_paths(repo_root: Path, RuntimePaths: Any, config: Mapping[str, Any] | None = None) -> Any:
    runtime = RuntimePaths.discover(repo_root)
    settings = config.get("esmfold2", config) if isinstance(config, Mapping) else {}
    configured = settings.get("python", settings.get("runtime_python")) if isinstance(settings, Mapping) else None
    if configured:
        candidate = Path(str(configured)).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() if os.environ.get(ENV_MODEL_DIR) else repo_root) / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise ValueError(f"Configured ESMFold2 runtime Python is missing: {candidate}")
        return dataclass_replace(runtime, esmfold2_python=candidate)
    if runtime.esmfold2_python is not None and runtime.esmfold2_python.is_file():
        return runtime
    candidate = Path(sys.executable).resolve()
    return dataclass_replace(runtime, esmfold2_python=candidate) if candidate.is_file() else runtime


def _esm_config(config: Mapping[str, Any], repo_root: Path | None = None) -> Any:
    settings = config.get("esmfold2", config)
    if not isinstance(settings, Mapping):
        settings = {}
    if repo_root is not None:
        pipetools_source = community_source("pipetools", repo_root)
        if str(pipetools_source) not in sys.path:
            sys.path.insert(0, str(pipetools_source))
    from pipetools.config import ESMFold2Config

    return ESMFold2Config(
        mode=str(settings.get("mode", "fast")), samples=int(settings.get("samples", 1)),
        cpus=float(settings.get("cpus", 4.0)), gpus=float(settings.get("gpus", 1.0)),
        seed=int(settings.get("seed", 101)), loops=int(settings.get("loops", 20)),
        steps=int(settings.get("steps", 100)), lm_dropout=float(settings.get("lm_dropout", 0.3)),
        lm_mask_pct=settings.get("lm_mask_pct"),
        esmc_attention_backend=str(settings.get("attention_backend", settings.get("esmc_attention_backend", "sdpa"))),
        context_embedding_cache=settings.get("context_embedding_cache"),
        timeout_seconds=int(settings.get("timeout_seconds", 7200)),
        optimization=dict(settings.get("optimization", {"mode": "auto", "disable": []})),
    )


class _PersistentESMFold2:
    """One model-owned ESMFold2 process serving sequential requests."""

    def __init__(
        self,
        repo_root: Path,
        runtime: Any,
        config: Any,
        log_dir: Path | None = None,
    ):
        self.repo_root = repo_root
        self.runtime = runtime
        self.config = config
        cache_root = cache_dir("partidesign_validate", repo_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(
            prefix=f"validate_esmfold2_{os.getpid()}_",
            dir=cache_root,
        ))
        self.startup_path = self.root / "startup.json"
        self.stderr_path: Path | None = None
        self._log_dir: Path | None = None
        self._stderr_lock = threading.RLock()
        self._stderr_tail = bytearray()
        self._stderr_thread: threading.Thread | None = None
        self.process: subprocess.Popen[str] | None = None
        self._stdout_buffer = bytearray()
        self.set_log_dir(log_dir)
        try:
            self._start()
        except Exception:
            process, self.process = self.process, None
            if process is not None:
                self._terminate_process(process)
            else:
                self._join_stderr()
            shutil.rmtree(self.root, ignore_errors=True)
            raise

    def set_log_dir(self, log_dir: Path | None) -> None:
        """Route subsequent native lines to a task/method-owned log file."""

        with self._stderr_lock:
            if log_dir is None:
                self._log_dir = None
                self.stderr_path = None
                return
            self._log_dir = Path(log_dir).expanduser().resolve()
            self.stderr_path = self._log_dir / f"worker_{os.getpid()}.log"

    def _emit_stderr_line(self, line: str) -> None:
        if not line:
            return
        with self._stderr_lock:
            self._stderr_tail.extend((line + "\n").encode("utf-8", errors="replace"))
            if len(self._stderr_tail) > 8192:
                del self._stderr_tail[:-8192]
            path = self.stderr_path
            from .progress import emit

            emit(
                {"source": "esmfold2_native_stderr", "line": line},
                path,
                error=True,
            )

    def _drain_stderr(self, stream: Any) -> None:
        """Drain native stderr while the protocol reader waits for stdout."""

        pending = bytearray()
        carriage_return = False
        try:
            while True:
                try:
                    chunk = os.read(stream.fileno(), 65536)
                except (OSError, ValueError):
                    break
                if not chunk:
                    break
                for value in chunk:
                    if value == 13:  # tqdm commonly redraws with CR.
                        self._emit_stderr_line(pending.decode("utf-8", errors="replace"))
                        pending.clear()
                        carriage_return = True
                    elif value == 10:
                        if carriage_return:
                            carriage_return = False
                        else:
                            self._emit_stderr_line(pending.decode("utf-8", errors="replace"))
                            pending.clear()
                    else:
                        carriage_return = False
                        pending.append(value)
        finally:
            if pending:
                self._emit_stderr_line(pending.decode("utf-8", errors="replace"))

    def _join_stderr(self, stream: Any = None) -> None:
        thread = self._stderr_thread
        if thread is None:
            return
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        thread.join(timeout=30)
        self._stderr_thread = None

    @staticmethod
    def _close_stream(stream: Any) -> None:
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    def _terminate_process(self, process: Any) -> None:
        try:
            if process.poll() is None:
                process.kill()
        except (OSError, ValueError):
            pass
        try:
            process.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass
        stderr = getattr(process, "stderr", None)
        for stream in (
            getattr(process, "stdin", None),
            getattr(process, "stdout", None),
            stderr,
        ):
            self._close_stream(stream)
        self._join_stderr()

    def _stderr_failure_context(self) -> tuple[Path | None, str]:
        with self._stderr_lock:
            path = self.stderr_path
            tail = bytes(self._stderr_tail).decode("utf-8", errors="replace")
        return path, tail[-4000:]

    def _protocol_error(self, message: str) -> ESMFold2Error:
        path, tail = self._stderr_failure_context()
        location = f"; stderr log: {path}" if path is not None else ""
        suffix = f"; stderr tail: {tail}" if tail else ""
        return ESMFold2Error(f"{message}{location}{suffix}")

    def _start(self) -> None:
        from pipetools.adapters.esmfold2 import _effective_model_contract

        effective_config = _effective_model_contract(self.config)
        self.startup_path.write_text(json.dumps({
            "schema_version": 1,
            "mode": self.config.mode,
            "paths": {
                "fold_checkpoint": str(self.runtime.esmfold2_checkpoint(self.config.mode)),
                "esmc_checkpoint": str(self.runtime.esmfold2_esmc_checkpoint),
                "ccd": str(self.runtime.esmfold2_ccd),
                "ccd_bonds": str(self.runtime.esmfold2_ccd_bonds),
                "components_cif": str(self.runtime.esmfold2_common / "components.cif"),
            },
            "config": effective_config,
        }, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(self.runtime.esmfold2_python),
                str(self.repo_root / "src" / "validate" / "esmfold2_worker.py"),
                "--startup-json", str(self.startup_path),
            ],
            cwd=self.runtime.esmfold2_source,
            env=self.runtime.esmfold2_subprocess_env(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0,
        )
        stderr = self.process.stderr
        if stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(stderr,),
                name=f"esmfold2-stderr-{os.getpid()}",
                daemon=True,
            )
            self._stderr_thread.start()
        ready = self._read_line(timeout=float(self.config.timeout_seconds))
        if ready.get("type") != "ready" or ready.get("schema_version") != 1:
            raise self._protocol_error("persistent ESMFold2 worker did not become ready")

    def _read_line(self, *, timeout: float) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdout is None:
            raise ESMFold2Error("persistent ESMFold2 worker has no stdout")
        deadline = time.monotonic() + max(1.0, timeout)
        while True:
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                raw_line = bytes(self._stdout_buffer[:newline])
                del self._stdout_buffer[: newline + 1]
                try:
                    value = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    # Native imports may emit a diagnostic line.  The protocol
                    # remains framed by JSON objects and ignores that diagnostic.
                    continue
                if isinstance(value, dict):
                    return value
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._protocol_error("persistent ESMFold2 worker timed out")
            ready, _, _ = select.select([process.stdout.fileno()], [], [], remaining)
            if not ready:
                raise self._protocol_error("persistent ESMFold2 worker timed out")
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                raise self._protocol_error(
                    f"persistent ESMFold2 worker exited with {process.poll()}"
                )
            self._stdout_buffer.extend(chunk)

    def submit(self, request: Mapping[str, Any]) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdin is None:
            raise ESMFold2Error("persistent ESMFold2 worker is unavailable")
        process.stdin.write((json.dumps(dict(request), sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        process.stdin.flush()
        return self._read_line(timeout=float(self.config.timeout_seconds))

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            self._join_stderr()
            return
        try:
            if process.stdin is not None:
                process.stdin.write((json.dumps({"type": "close"}) + "\n").encode("utf-8"))
                process.stdin.flush()
                process.stdin.close()
            process.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self._terminate_process(process)
        finally:
            for stream in (
                getattr(process, "stdin", None),
                getattr(process, "stdout", None),
                getattr(process, "stderr", None),
            ):
                self._close_stream(stream)
            self._join_stderr()
        shutil.rmtree(self.root, ignore_errors=True)


class ESMFold2Backend:
    """Persistent model-owned ESMFold2 service for validation."""

    def __init__(self, repo_root: Path, config: Mapping[str, Any] | None = None, device: Any = None):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.config = dict(config or {})
        self.device = device
        self._closed = False
        self._runtime = None
        self._model_config = None
        self._service: _PersistentESMFold2 | None = None
        self._log_dir: Path | None = None
        self._log_dir_explicit = False

    @staticmethod
    def _default_log_dir(output_dir: Path) -> Path:
        """Use the conventional task/method log sibling for direct callers."""

        output_dir = Path(output_dir).expanduser().resolve()
        if output_dir.parent.name == "refolds":
            return output_dir.parent.parent / "logs"
        if output_dir.name == "refolds":
            return output_dir.parent / "logs"
        return output_dir / "logs"

    def _update_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = None if log_dir is None else Path(log_dir).expanduser().resolve()
        if self._service is not None:
            setter = getattr(self._service, "set_log_dir", None)
            if setter is not None:
                setter(self._log_dir)

    def set_log_dir(self, log_dir: Path | None) -> None:
        """Set the current task/method log route without reloading the model."""

        self._log_dir_explicit = log_dir is not None
        self._update_log_dir(log_dir)

    def _ensure_service(self) -> tuple[Any, Any]:
        if self._service is not None and self._model_config is not None:
            return self._service, self._model_config
        config = _esm_config(self.config, self.repo_root)
        source = community_source("pipetools", self.repo_root)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from pipetools.runtime import RuntimePaths

        runtime = _runtime_paths(self.repo_root, RuntimePaths, self.config)
        runtime.validate_esmfold2(config.mode)
        cache_root = cache_dir("partidesign_validate", self.repo_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        self._runtime = runtime
        self._model_config = config
        self._service = _PersistentESMFold2(
            self.repo_root, runtime, config, log_dir=self._log_dir
        )
        if self._log_dir is not None:
            setter = getattr(self._service, "set_log_dir", None)
            if setter is not None:
                setter(self._log_dir)
        return self._service, config

    def close(self) -> None:
        self._closed = True
        if self._service is not None:
            self._service.close()
            self._service = None

    def predict(
        self, input: Any, variant: Mapping[str, Any], output_dir: Path,
        completed_keys: set[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        if self._closed:
            raise ESMFold2Error("ESMFold2 backend is closed")
        completed = set(completed_keys or ())
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self._log_dir_explicit:
            self._update_log_dir(self._default_log_dir(output_dir))
        seq_index = int(variant.get("seq_index", 0))
        variant_id = f"seq{seq_index}"
        try:
            service, config = self._ensure_service()
        except Exception as exc:
            failed_config = _esm_config(self.config)
            for sample_index in range(int(failed_config.samples)):
                key = f"sample{sample_index}"
                if key not in completed:
                    yield {
                        "key": key, "protocol": "complex", "model": failed_config.mode,
                        "sample": sample_index, "status": "failed",
                        "structure_path": None, "confidence": {}, "confidence_path": None,
                        "raw_confidence_path": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            return
        expected = {f"sample{i}" for i in range(int(config.samples))}
        if expected <= completed:
            return
        payload = _build_payload(input, variant)
        payload["sample_id"] = str(_field(input, "sample_id", payload.get("sample_id", "sample")))
        payload["variant_id"] = variant_id
        with tempfile.TemporaryDirectory(prefix=f".{variant_id}_", dir=output_dir) as temporary:
            request_root = Path(temporary)
            input_json = request_root / f"{variant_id}.json"
            _write_json_atomic(input_json, payload)
            sample_id = str(payload["sample_id"])
            raw_root = request_root / "raw"
            raw_root.mkdir(parents=True, exist_ok=True)
            for sample_index in range(int(config.samples)):
                key = f"sample{sample_index}"
                if key in completed:
                    continue
                attempt_dir = raw_root / f"sample{sample_index}"
                sample_seed = int(config.seed) + sample_index
                request_id = uuid.uuid4().hex
                request = {
                    "request_id": request_id,
                    "input_json": str(input_json.resolve()),
                    "output_dir": str(attempt_dir.resolve()),
                    "expected_prefix": f"{sample_id}__{variant_id}__sample{sample_index}",
                    "expected_samples": 1,
                    "sample_index": sample_index,
                    "seed": sample_seed,
                }
                try:
                    response = service.submit(request)
                except Exception as exc:
                    service.close()
                    self._service = None
                    for remaining in range(sample_index, int(config.samples)):
                        remaining_key = f"sample{remaining}"
                        if remaining_key not in completed:
                            yield {
                                "key": remaining_key, "protocol": "complex", "model": config.mode,
                                "sample": remaining, "status": "failed",
                                "structure_path": None, "confidence": {},
                                "confidence_path": None, "raw_confidence_path": None,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                    return
                if response.get("type") != "result" or response.get("status") != "ok":
                    error = str(response.get("error") or "ESMFold2 worker returned no outcome")
                    if response.get("fatal"):
                        service.close()
                        self._service = None
                        for remaining in range(sample_index, int(config.samples)):
                            remaining_key = f"sample{remaining}"
                            if remaining_key not in completed:
                                yield {
                                    "key": remaining_key, "protocol": "complex", "model": config.mode,
                                    "sample": remaining, "status": "failed",
                                    "structure_path": None, "confidence": {}, "error": error,
                                }
                        return
                    yield {
                        "key": key, "protocol": "complex", "model": config.mode,
                        "sample": sample_index, "status": "failed",
                        "structure_path": None, "confidence": {},
                        "confidence_path": None, "raw_confidence_path": None,
                        "error": error,
                    }
                    continue
                artifacts = response.get("artifacts", [])
                if not isinstance(artifacts, list) or len(artifacts) != 1:
                    raise ESMFold2Error("ESMFold2 worker returned an incomplete one-sample artifact list")
                artifact = artifacts[0]
                if not isinstance(artifact, Mapping):
                    raise ESMFold2Error("ESMFold2 worker artifact is malformed")
                structure = Path(str(artifact.get("structure", "")))
                confidence_path = Path(str(artifact.get("confidence", "")))
                raw_confidence = Path(str(artifact.get("raw_confidence", "")))
                attempt_root = attempt_dir.resolve()
                for artifact_path in (structure, confidence_path, raw_confidence):
                    if artifact_path.is_symlink() or not artifact_path.is_file():
                        raise ESMFold2Error("ESMFold2 worker artifact is missing or symlinked")
                    if not artifact_path.resolve().is_relative_to(attempt_root):
                        raise ESMFold2Error("ESMFold2 worker artifact escaped its request directory")
                confidence_record = json.loads(confidence_path.read_text(encoding="utf-8"))
                if not isinstance(confidence_record, Mapping) or not isinstance(confidence_record.get("confidence"), Mapping):
                    raise ESMFold2Error("ESMFold2 confidence record is malformed")
                from src.design.rewards.esmfold2_mapping import mapped_confidence_summary

                confidence = mapped_confidence_summary(confidence_record["confidence"])
                if isinstance(confidence_record.get("context_embedding_cache"), Mapping):
                    confidence["context_embedding_cache"] = dict(confidence_record["context_embedding_cache"])
                final_structure = output_dir / f"{variant_id}__s{sample_index:03d}.cif"
                arrays = _read_native_confidence_arrays(raw_confidence)
                arrays = _logical_confidence_arrays(arrays, payload, confidence)
                _restore_mixed_output(structure, payload)
                # Publish one final structure; all request files remain temporary.
                structure.replace(final_structure)
                prediction = {
                    "key": key, "protocol": "complex", "model": config.mode,
                    "sample": sample_index, "status": "complete",
                    "structure_path": str(final_structure),
                    "confidence": confidence, "_confidence_arrays": arrays, "error": None,
                    # Native output preserves the explicitly submitted chain labels.
                    # This maps chain-level confidence, never ligand atom coordinates.
                    "_confidence_chain_map": {
                        str(entry["chain_id"]): next(
                            (str(segment["logical_chain_id"]) for segment in
                             payload.get(_RECEPTOR_SEGMENT_LEDGER, {}).get("segments", [])
                             if segment["physical_chain_id"] == entry["chain_id"]),
                            str(entry["chain_id"]))
                        for entry in payload["sequences"] if entry.get("modality") == "ligand"
                    },
                }
                shutil.rmtree(attempt_dir)
                yield prediction



__all__ = ["ESMFold2Backend", "ESMFold2Error"]
