"""Private source occurrence mapping shared by both ESMFold2 execution paths."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SOURCE_MAPPING = "partidesign_receptor_segments"
_POLYMERS = {"protein", "rna", "dna"}


def _rows(group: Any) -> list[int]:
    return [int(value) for value in group["row_indices"]]


def _source_metadata(atoms: Any, rows: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("source_graph_complete", "source_break_before", "source_polymer_position"):
        values = getattr(atoms, name, None)
        if values is not None:
            value = values[rows[0]]
            result[name] = bool(value) if name != "source_polymer_position" else int(value)
    return result


def _ligand_groups(atoms: Any, chain: str) -> list[list[int]]:
    chains = np.asarray(getattr(atoms, "label_asym_id", atoms.chain_id)).astype(str)
    selected = np.flatnonzero((chains == chain) & (np.asarray(atoms.mol_type).astype(str) == "ligand"))
    components = np.asarray(atoms.component_uid, dtype=np.int64)
    return [selected[components[selected] == uid].tolist()
            for uid in dict.fromkeys(components[selected].tolist())]


def apply_source_mapping(
    payload: Mapping[str, Any], atoms: Any, *, design_chains: tuple[str, ...],
) -> dict[str, Any]:
    """Turn observed logical entries into reversible physical model entities once."""
    from ..utils.source_mapping import plan_refold_positions, source_locator

    if SOURCE_MAPPING in payload:
        raise ValueError("ESMFold2 input source mapping was already applied")
    result = copy.deepcopy(dict(payload))
    logical = result.get("sequences")
    if not isinstance(logical, list) or not logical:
        raise ValueError("ESMFold2 source mapping lacks logical entries")
    sequences = {str(entry["chain_id"]): str(entry["sequence"])
                 for entry in logical if entry.get("modality") in _POLYMERS}
    plan = plan_refold_positions(atoms, design_chains=design_chains, chain_sequences=sequences)
    plans = {str(chain["logical_chain_id"]): chain for chain in plan["chains"]}
    physical: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    endpoints: dict[tuple[str, int], tuple[str, int]] = {}
    entity_ids: dict[tuple[Any, ...], int] = {}
    entity_copies: dict[tuple[Any, ...], int] = {}
    source_sequences: dict[str, Any] = {}
    sequence_values: list[list[str]] = []
    sequence_indices: dict[tuple[str, ...], int] = {}
    occurrence_by_row: dict[int, tuple[str, int]] = {}
    outputs: dict[str, tuple[Any, ...]] = {}
    used_chain_ids = {str(entry["chain_id"]) for entry in logical}
    used_chain_ids.update(str(entity["chain_id"]) for plan in plans.values() for entity in plan["entities"])
    for entry in logical:
        chain = str(entry["chain_id"])
        if entry.get("modality") not in _POLYMERS:
            groups = _ligand_groups(atoms, chain)
            expected = len(entry.get("ccd", ())) if "ccd" in entry else 1
            if len(groups) != expected:
                raise ValueError(f"ESMFold2 ligand source occurrence count differs for {chain!r}")
            entities = [{"chain_id": chain, "observed_positions": list(range(len(groups)))}]
            locators = [source_locator(atoms, group[0]) for group in groups]
            source_metadata = [_source_metadata(atoms, group) for group in groups]
            for ordinal, group in enumerate(groups):
                occurrence_by_row.update({row: (chain, ordinal) for row in group})
        else:
            chainplan = plans.get(chain)
            if chainplan is None:
                raise ValueError(f"ESMFold2 source plan lacks chain {chain!r}")
            observed = chainplan["observed"]
            if len(observed) != len(entry["sequence"]):
                raise ValueError(f"ESMFold2 observed sequence length differs for {chain!r}")
            entities = chainplan["entities"]
            locators = [copy.deepcopy(dict(group["locator"])) for group in observed]
            source_metadata = [_source_metadata(atoms, _rows(group)) for group in observed]
            for ordinal, group in enumerate(observed):
                source_metadata[ordinal]["source_break_before"] = ordinal in chainplan["break_before"]
                occurrence_by_row.update({row: (chain, ordinal) for row in _rows(group)})
                full_value = group.get("source_polymer_sequence")
                if full_value is not None:
                    full_value = tuple(str(value) for value in full_value)
                    if full_value not in sequence_indices:
                        sequence_indices[full_value] = len(sequence_values)
                        sequence_values.append(list(full_value))
                    source_metadata[ordinal]["source_polymer_sequence_index"] = sequence_indices[full_value]
            if chainplan["role"] == "context":
                full = next((group.get("source_polymer_sequence") for group in observed
                             if group.get("source_polymer_sequence") is not None), None)
                if full is not None:
                    source_sequences[chain] = list(full)
        for entity in entities:
            item = copy.deepcopy(entry)
            physical_chain = str(entity["chain_id"])
            if len(physical_chain) > 16:
                if physical_chain == chain:
                    raise ValueError("ESMFold2 source chain identity exceeds its native CIF width")
                serial = 1
                while True:
                    tag = f"__pdfrag{serial}"
                    alias = f"{chain[:16 - len(tag)]}{tag}"
                    if alias not in used_chain_ids:
                        physical_chain = alias
                        used_chain_ids.add(alias)
                        break
                    serial += 1
            positions = list(entity["observed_positions"])
            item["chain_id"] = physical_chain
            if entry.get("modality") in _POLYMERS:
                item["sequence"] = str(entity["sequence"])
                inverse = {position: index for index, position in enumerate(positions)
                           if position is not None}
                modifications = []
                for modification in entry.get("modifications", []):
                    position = modification.get("position")
                    if position in inverse:
                        modified = copy.deepcopy(modification)
                        modified["position"] = inverse[position]
                        modifications.append(modified)
                from .esmfold2_input import _STANDARD_RES_ATOMS

                components = entity.get("components", [])
                for position, observed in enumerate(positions):
                    if observed is None and components[position] not in _STANDARD_RES_ATOMS[entry["modality"]]:
                        # An unobserved confirmed CCD identity has the native
                        # full CCD inventory, with no fabricated source atoms.
                        modifications.append({"position": position, "ccd": str(components[position])})
                item["modifications"] = modifications
            if physical_chain != chain or len(entities) > 1:
                item["output_entity_id"] = f"{entry['output_entity_id']}__{physical_chain}"
            signature = (entry["model_entity_index"], item.get("modality"),
                         item.get("sequence"), str(item.get("modifications", [])),
                         str(item.get("ccd", [])), item.get("smiles"))
            if len(entities) > 1:
                signature += (physical_chain,)
            entity = entity_ids.setdefault(signature, len(entity_ids) + 1)
            entity_copies[signature] = entity_copies.get(signature, 0) + 1
            item["model_entity_index"] = entity
            item["model_entity_copy"] = entity_copies[signature]
            output_id = str(item["output_entity_id"])
            if output_id in outputs and outputs[output_id] != signature:
                output_id = f"{output_id}__ef2entity{entity}"
            outputs[output_id] = signature
            item["output_entity_id"] = output_id
            physical.append(item)
            row = {"physical_chain_id": physical_chain, "logical_chain_id": chain,
                   "observed_positions": positions,
                   "source_locators": [None if index is None else locators[index] for index in positions],
                   "source_metadata": [None if index is None else source_metadata[index] for index in positions]}
            mapped = [index for index in positions if index is not None]
            row["logical_start"] = min(mapped)
            row["logical_stop"] = max(mapped) + 1
            rows.append(row)
            for index, position in enumerate(positions):
                if position is not None:
                    key = (chain, int(position))
                    if key in endpoints:
                        raise ValueError("ESMFold2 source occurrence is mapped more than once")
                    endpoints[key] = (physical_chain, index)
    bonds = copy.deepcopy(result.get("covalent_bonds", []))
    for bond in bonds:
        for name in ("endpoint1", "endpoint2"):
            endpoint = bond[name]
            key = (str(endpoint["chain_id"]), int(endpoint["residue_index"]))
            if key not in endpoints:
                raise ValueError("ESMFold2 covalent endpoint is absent from its source map")
            endpoint["chain_id"], endpoint["residue_index"] = endpoints[key]
    preserved_backbone = []
    logical_bonds = copy.deepcopy(payload.get("covalent_bonds", []))
    physical_by_chain = {entry["chain_id"]: entry for entry in physical}

    def bond_key(bond: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(sorted((str(bond[name]["chain_id"]), int(bond[name]["residue_index"]), str(bond[name]["atom_name"]))
                            for name in ("endpoint1", "endpoint2")))

    existing_bonds = {bond_key(bond): bond for bond in bonds}
    bond_list = getattr(atoms, "bonds", None)
    if bond_list is not None:
        for left, right, _order in bond_list.as_array():
            left, right = int(left), int(right)
            if left not in occurrence_by_row or right not in occurrence_by_row:
                continue
            left_chain, left_position = occurrence_by_row[left]
            right_chain, right_position = occurrence_by_row[right]
            names = (str(atoms.atom_name[left]), str(atoms.atom_name[right]))
            if left_chain == right_chain and left_position != right_position and set(names) in ({"C", "N"}, {"O3'", "P"}):
                preserved_backbone.append([left_chain, left_position, names[0], right_position, names[1], int(_order)])
                left_native = endpoints[(left_chain, left_position)]
                right_native = endpoints[(right_chain, right_position)]
                native_implicit = (left_native[0] == right_native[0] and abs(left_native[1] - right_native[1]) == 1
                                   and int(_order) == 1)
                for physical_chain, position in (left_native, right_native):
                    if any(int(modification["position"]) == position and modification.get("infer_standard_polymer_bonds") is False
                           for modification in physical_by_chain[physical_chain].get("modifications", [])):
                        native_implicit = False
                if not native_implicit:
                    from .esmfold2_input import _BIOTITE_TO_ORDER, _merge_covalent_order

                    physical_bond = {"endpoint1": {"chain_id": left_native[0], "residue_index": left_native[1], "atom_name": names[0]},
                                     "endpoint2": {"chain_id": right_native[0], "residue_index": right_native[1], "atom_name": names[1]},
                                     "bond_order": _BIOTITE_TO_ORDER.get(int(_order), "?")}
                    key = bond_key(physical_bond)
                    previous = existing_bonds.get(key)
                    if previous is None:
                        bonds.append(physical_bond)
                        existing_bonds[key] = physical_bond
                        logical_bonds.append({"endpoint1": {"chain_id": left_chain, "residue_index": left_position, "atom_name": names[0]},
                                              "endpoint2": {"chain_id": right_chain, "residue_index": right_position, "atom_name": names[1]},
                                              "bond_order": physical_bond["bond_order"]})
                    else:
                        previous["bond_order"] = _merge_covalent_order(previous["bond_order"], physical_bond["bond_order"],
                                                                       conflict_message="ESMFold2 source backbone order conflicts with its explicit input")
    result["sequences"] = physical
    result["covalent_bonds"] = bonds
    result[SOURCE_MAPPING] = {"mapping_version": 2, "logical_sequences": logical,
                              "logical_covalent_bonds": logical_bonds,
                              "segments": rows, "source_polymer_sequences": source_sequences,
                              "source_polymer_sequence_values": sequence_values,
                              "preserved_backbone_edges": preserved_backbone}
    return result


def source_mapping(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    ledger = payload.get(SOURCE_MAPPING)
    if ledger is None:
        return None
    if not isinstance(ledger, Mapping) or ledger.get("mapping_version") != 2:
        raise ValueError("ESMFold2 cached input lacks exact source occurrence mapping; rebuild it")
    logical = ledger.get("logical_sequences")
    rows = ledger.get("segments")
    physical = payload.get("sequences")
    if not all(isinstance(value, list) and value for value in (logical, rows, physical)):
        raise ValueError("ESMFold2 source occurrence mapping is malformed")
    mapping = {str(row["physical_chain_id"]): row for row in rows}
    if len(mapping) != len(rows) or set(mapping) != {str(entry["chain_id"]) for entry in physical}:
        raise ValueError("ESMFold2 physical occurrence mapping is incomplete")
    logical_ids = [str(entry["chain_id"]) for entry in logical]
    if len(set(logical_ids)) != len(logical_ids):
        raise ValueError("ESMFold2 logical occurrence mapping is duplicate")
    coverage: dict[str, set[int]] = {chain: set() for chain in logical_ids}
    for entry in physical:
        row = mapping[str(entry["chain_id"])]
        positions, locators = row["observed_positions"], row["source_locators"]
        expected = len(entry["sequence"]) if entry["modality"] in _POLYMERS else len(entry.get("ccd", ())) or 1
        if (row["logical_chain_id"] not in logical_ids or len(positions) != expected
                or len(locators) != expected or len(row.get("source_metadata", [])) != expected):
            raise ValueError("ESMFold2 source occurrence mapping length differs")
        for position, locator in zip(positions, locators, strict=True):
            if position is None:
                if locator is not None:
                    raise ValueError("ESMFold2 filler residue has a false source locator")
            elif isinstance(position, bool) or not isinstance(position, int) or position < 0 or not isinstance(locator, Mapping):
                raise ValueError("ESMFold2 source occurrence locator is malformed")
            else:
                logical_chain = str(row["logical_chain_id"])
                if position in coverage[logical_chain]:
                    raise ValueError("ESMFold2 source occurrence is duplicated across native entities")
                coverage[logical_chain].add(position)
    for entry in logical:
        expected = len(entry["sequence"]) if entry["modality"] in _POLYMERS else len(entry.get("ccd", ())) or 1
        expected += len(entry.get("context_components", ()))
        if coverage[str(entry["chain_id"])] != set(range(expected)):
            raise ValueError("ESMFold2 source occurrence mapping dropped or invented an observed position")
    return {"physical_chain_ids": tuple(str(entry["chain_id"]) for entry in physical),
            "logical_sequences": logical, "logical_covalent_bonds": ledger["logical_covalent_bonds"],
            "physical_to_logical": mapping, "logical_index": {chain: index for index, chain in enumerate(logical_ids)},
            "mixed_context": bool(ledger.get("mixed_context", False))}


def update_mapped_sequences(
    payload: Mapping[str, Any], atoms: Any, sequences: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse one exact supplied mapping after checking the current source atoms."""
    from ..utils.source_mapping import canonical_chain_ids, polymer_residue_groups, source_locator

    mapping = source_mapping(payload)
    if mapping is None:
        raise ValueError("ESMFold2 supplied input lacks exact source mapping")
    result = copy.deepcopy(dict(payload))
    locators = {}
    for chain in dict.fromkeys(canonical_chain_ids(atoms).tolist()):
        groups = polymer_residue_groups(atoms, chain)
        values = [group["locator"] for group in groups]
        values.extend(source_locator(atoms, group[0]) for group in _ligand_groups(atoms, chain))
        locators[chain] = values
    for row in mapping["physical_to_logical"].values():
        observed = locators.get(str(row["logical_chain_id"]), [])
        for position, locator in zip(row["observed_positions"], row["source_locators"], strict=True):
            if position is not None and (position >= len(observed) or locator != observed[position]):
                raise ValueError("ESMFold2 supplied mapping differs from current source locators; rebuild it")
    logical = result[SOURCE_MAPPING]["logical_sequences"]
    logical_by_chain = {str(entry["chain_id"]): entry for entry in logical}
    for chain, sequence in sequences.items():
        entry = logical_by_chain.get(str(chain))
        if entry is None or entry.get("modality") not in _POLYMERS:
            continue
        value = str(sequence)
        old = entry["sequence"]
        if len(value) != len(old):
            raise ValueError("ESMFold2 supplied observed sequence length changed")
        for modification in entry.get("modifications", []):
            position = int(modification["position"])
            if value[position] != old[position]:
                raise ValueError("ESMFold2 supplied sequence changed a fixed modified position")
        entry["sequence"] = value
    entities, copies, output_ids = {}, {}, {}
    for entry in result["sequences"]:
        row = mapping["physical_to_logical"][str(entry["chain_id"])]
        if entry["modality"] in _POLYMERS:
            value = list(entry["sequence"])
            logical_sequence = logical_by_chain[str(row["logical_chain_id"])]["sequence"]
            for native, observed in enumerate(row["observed_positions"]):
                if observed is not None:
                    value[native] = logical_sequence[observed]
            entry["sequence"] = "".join(value)
        signature = (entry["model_entity_index"], entry.get("sequence"), str(entry.get("modifications", [])))
        entity = entities.setdefault(signature, len(entities) + 1)
        copies[signature] = copies.get(signature, 0) + 1
        entry["model_entity_index"], entry["model_entity_copy"] = entity, copies[signature]
        output = str(entry["output_entity_id"])
        if output in output_ids and output_ids[output] != signature:
            output = f"{output}__ef2entity{entity}"
        output_ids[output] = signature
        entry["output_entity_id"] = output
    return result


def mapped_token_inventory(
    asym: Any, residues: Any, token_mask: Any, payload: Mapping[str, Any],
    *, mapping: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mapping is None:
        mapping = source_mapping(payload)
    chains = np.asarray(asym, dtype=np.int64)
    positions = np.asarray(residues, dtype=np.int64)
    retained = np.asarray(token_mask, dtype=bool).copy()
    if chains.shape != positions.shape or chains.shape != retained.shape or chains.ndim != 1:
        raise ValueError("ESMFold2 native token identity shape differs")
    labels = np.full(chains.shape, "", dtype=object)
    logical_asym = chains.copy()
    logical_positions = positions.copy()
    if mapping is None:
        raise ValueError("ESMFold2 input lacks exact source occurrence mapping")
    for index in np.flatnonzero(retained):
        physical_index = int(chains[index])
        if not 0 <= physical_index < len(mapping["physical_chain_ids"]):
            raise ValueError("ESMFold2 native token chain index is invalid")
        row = mapping["physical_to_logical"][mapping["physical_chain_ids"][physical_index]]
        position = int(positions[index])
        if not 0 <= position < len(row["observed_positions"]):
            raise ValueError("ESMFold2 native token residue index is outside its source map")
        observed = row["observed_positions"][position]
        if observed is None:
            retained[index] = False
            continue
        chain = str(row["logical_chain_id"])
        labels[index] = chain
        logical_asym[index] = mapping["logical_index"][chain]
        logical_positions[index] = int(observed)
    return labels.astype(str), logical_asym, logical_positions, retained


def logical_confidence_arrays(
    arrays: Mapping[str, Any], payload: Mapping[str, Any], confidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if SOURCE_MAPPING not in payload:
        return dict(arrays)
    mapping = source_mapping(payload)
    asym = np.asarray(arrays["asym_id"])
    labels, chains, positions, retained = mapped_token_inventory(
        asym, arrays["residue_index"], arrays.get("token_attention_mask", np.ones(asym.shape, dtype=bool)), payload, mapping=mapping)
    selected = np.flatnonzero(retained)
    result = {f"native_{name}": value for name, value in arrays.items()}
    token_fields = {"plddt", "plddt_ca", "token_index", "residue_index", "asym_id", "entity_id", "sym_id", "token_attention_mask"}
    for name, value in arrays.items():
        array = np.asarray(value)
        if name in {"pae", "pae_matrix", "pde"}:
            if array.shape != (len(asym), len(asym)):
                raise ValueError("ESMFold2 pair confidence shape differs from native tokens")
            result[name] = array[np.ix_(selected, selected)]
        elif name in token_fields:
            if array.shape != asym.shape:
                raise ValueError("ESMFold2 token confidence shape differs from native tokens")
            result[name] = array[selected]
        elif name not in {"plddt_per_atom", "pair_chains_iptm", "ptm", "iptm", "complex_plddt", "complex_iplddt"}:
            result[name] = value
    if confidence is not None:
        for raw_name, summary_name, scale in (("ptm", "ptm", 1.0), ("iptm", "iptm", 1.0),
                                              ("complex_plddt", "plddt", 100.0), ("complex_iplddt", "complex_iplddt", 100.0)):
            if summary_name in confidence:
                result[raw_name] = np.asarray(float(confidence[summary_name]) / scale)
        pair = confidence.get("chain_pair_iptm")
        if isinstance(pair, Mapping):
            logical_ids = [str(entry["chain_id"]) for entry in mapping["logical_sequences"]]
            result["pair_chains_iptm"] = np.asarray([[float(pair[left][right]) for right in logical_ids] for left in logical_ids])
    result["asym_id"] = chains[selected]
    result["residue_index"] = positions[selected]
    result["token_chain_ids"] = labels[selected]
    result["native_mapped_token_mask"] = retained
    return result


def restored_source_atoms(path: Path, payload: Mapping[str, Any]) -> Any:
    """Restore exact source locators while retaining predicted atoms and real bonds."""
    mapping = source_mapping(payload)
    if mapping is None:
        raise ValueError("ESMFold2 output cannot be restored without exact source mapping")
    return _restored_source_atoms(path, payload, mapping)


def _restored_source_atoms(path: Path, payload: Mapping[str, Any], mapping: Mapping[str, Any]) -> Any:
    from ..utils.reader import read_structure

    atoms = read_structure(path, declared_polymer_chain_modalities={
        str(entry["chain_id"]): str(entry["modality"]) for entry in payload["sequences"]})
    chains = np.asarray(atoms.label_asym_id).astype(str)
    kinds = np.asarray(atoms.mol_type).astype(str)
    output_rows: list[int] = []
    restorations: list[tuple[list[int], Mapping[str, Any], Mapping[str, Any], str, int]] = []
    for entry in payload["sequences"]:
        chain = str(entry["chain_id"])
        selected = np.flatnonzero(chains == chain)
        occurrences = np.asarray(atoms.residue_uid, dtype=np.int64)
        groups = [selected[occurrences[selected] == uid].tolist()
                  for uid in dict.fromkeys(occurrences[selected].tolist())]
        row = mapping["physical_to_logical"][chain]
        if len(groups) != len(row["observed_positions"]):
            raise ValueError("ESMFold2 output residue coverage differs from its source map")
        for local, group in enumerate(groups):
            observed = row["observed_positions"][local]
            if observed is None:
                continue
            restorations.append((group, row["source_locators"][local], row["source_metadata"][local] or {},
                                 str(row["logical_chain_id"]), int(observed)))
    restorations.sort(key=lambda item: (mapping["logical_index"][item[3]], item[4]))
    output_rows = [index for group, *_ in restorations for index in group]
    result = atoms[np.asarray(output_rows, dtype=np.int64)].copy()
    row_identity: list[tuple[str, int]] = []
    full_sequences: list[Any] = []
    values: dict[str, list[Any]] = {name: [] for name in (
        "chain_id", "res_id", "ins_code", "label_asym_id", "label_seq_id", "auth_asym_id", "auth_seq_id", "auth_ins_code",
        "source_format", "source_graph_complete", "source_break_before", "source_polymer_position", "asym_uid", "entity_uid", "residue_uid", "component_uid")}
    token_values: list[int] = []
    next_token = 0
    for residue_uid, (group, locator, metadata, logical_chain, _observed) in enumerate(restorations):
        count = len(group)
        row_identity.extend([(logical_chain, _observed)] * count)
        full_index = metadata.get("source_polymer_sequence_index")
        full = (payload[SOURCE_MAPPING]["source_polymer_sequence_values"][int(full_index)] if full_index is not None
                else payload[SOURCE_MAPPING].get("source_polymer_sequences", {}).get(logical_chain))
        full_sequences.extend([None if full is None else tuple(full)] * count)
        for name in ("chain_id", "res_id", "ins_code", "label_asym_id", "label_seq_id", "auth_asym_id", "auth_seq_id", "auth_ins_code", "source_format"):
            values[name].extend([locator[name]] * count)
        for name, default in (("source_graph_complete", True), ("source_break_before", False), ("source_polymer_position", 0)):
            values[name].extend([metadata.get(name, default)] * count)
        for name in ("asym_uid", "entity_uid"):
            values[name].extend([mapping["logical_index"][logical_chain]] * count)
        for name in ("residue_uid", "component_uid"):
            values[name].extend([residue_uid] * count)
        if kinds[group[0]] == "ligand":
            token_values.extend(range(next_token, next_token + count))
            next_token += count
        else:
            token_values.extend([next_token] * count)
            next_token += 1
    integer = {"res_id", "source_polymer_position", "asym_uid", "entity_uid", "residue_uid", "component_uid"}
    boolean = {"source_graph_complete", "source_break_before"}
    for name, data in values.items():
        result.set_annotation(name, np.asarray(data, dtype=np.int64 if name in integer else bool if name in boolean else str))
    result.set_annotation("token_uid", np.asarray(token_values, dtype=np.int64))
    result.set_annotation("atom_uid", np.arange(len(result), dtype=np.int64))
    full_annotation = np.empty(len(result), dtype=object)
    for index, sequence in enumerate(full_sequences):
        full_annotation[index] = sequence
    result.set_annotation("source_polymer_sequence", full_annotation)
    # A native chain can connect consecutive positions even when the observed
    # source graph declares a real break. Keep only source-authorized bridges.
    if result.bonds is not None:
        from biotite.structure import BondList

        allowed = set()
        for edge in payload[SOURCE_MAPPING].get("preserved_backbone_edges", []):
            chain, left_position, left_name, right_position, right_name = edge[:5]
            allowed.add(tuple(sorted(((chain, int(left_position), left_name), (chain, int(right_position), right_name)))))
        kept: dict[tuple[int, int], int] = {}
        for left, right, order in result.bonds.as_array():
            left, right = int(left), int(right)
            left_chain, left_position = row_identity[left]
            right_chain, right_position = row_identity[right]
            names = (str(result.atom_name[left]), str(result.atom_name[right]))
            boundary = left_chain == right_chain and left_position != right_position and set(names) in ({"C", "N"}, {"O3'", "P"})
            later = left if left_position > right_position else right
            signature = tuple(sorted(((left_chain, left_position, names[0]), (right_chain, right_position, names[1]))))
            if boundary and bool(result.source_break_before[later]) and signature not in allowed:
                continue
            kept[tuple(sorted((left, right)))] = int(order)
        rows_by_atom: dict[tuple[str, int, str], list[int]] = {}
        for row, (chain, position) in enumerate(row_identity):
            rows_by_atom.setdefault((chain, position, str(result.atom_name[row])), []).append(row)

        def mapped_edge(chain: str, left_position: int, left_name: str, right_position: int, right_name: str) -> tuple[int, int]:
            left_rows = rows_by_atom.get((chain, left_position, left_name), [])
            right_rows = rows_by_atom.get((chain, right_position, right_name), [])
            if len(left_rows) != 1 or len(right_rows) != 1:
                raise ValueError("ESMFold2 restored backbone endpoint is absent or ambiguous")
            return tuple(sorted((left_rows[0], right_rows[0])))

        # The prepared native sequence owns standard connectivity; predicted
        # bond distances cannot remove it during the private CIF read.
        for entry in payload["sequences"]:
            if entry["modality"] not in _POLYMERS:
                continue
            segment = mapping["physical_to_logical"][entry["chain_id"]]
            positions = segment["observed_positions"]
            exact = {int(modification["position"]) for modification in entry.get("modifications", [])
                     if modification.get("infer_standard_polymer_bonds") is False}
            for local in range(1, len(positions)):
                left_position, right_position = positions[local - 1], positions[local]
                if (left_position is None or right_position is None or local in exact or local - 1 in exact
                        or bool((segment["source_metadata"][local] or {}).get("source_break_before", False))):
                    continue
                left_name, right_name = ("C", "N") if entry["modality"] == "protein" else ("O3'", "P")
                edge = mapped_edge(str(segment["logical_chain_id"]), int(left_position), left_name, int(right_position), right_name)
                kept.setdefault(edge, 1)
        for source_edge in payload[SOURCE_MAPPING].get("preserved_backbone_edges", []):
            chain, left_position, left_name, right_position, right_name = source_edge[:5]
            edge = mapped_edge(str(chain), int(left_position), str(left_name), int(right_position), str(right_name))
            order = int(source_edge[5]) if len(source_edge) > 5 else 1
            if order != 0 or edge not in kept:
                kept[edge] = order
        edges = [(left, right, order) for (left, right), order in sorted(kept.items())]
        result.bonds = BondList(len(result), np.asarray(edges, dtype=np.int64).reshape(-1, 3))
    # The retained graph is authoritative: removing filler atoms removes their
    # edges, and the public writer must not infer a new edge across that hole.
    result.set_annotation("source_graph_complete", np.ones(len(result), dtype=bool))
    return result


def _project_continuous_protein_cif(
    path: Path, payload: Mapping[str, Any], mapping: Mapping[str, Any], target: Path,
) -> Path | None:
    """Publish verified ordinary native columns with an explicit observed backbone graph."""
    ledger = payload[SOURCE_MAPPING]
    if (target.suffix.lower() != ".cif" or payload.get("covalent_bonds")
            or ledger.get("logical_covalent_bonds") or ledger.get("source_inter_residue_bonds")
            or ledger.get("source_polymer_sequences") or ledger.get("source_polymer_sequence_values")):
        return None
    from .esmfold2_input import _AA3_TO_1
    from ..utils.atoms import PROTEIN_RES_ATOMS

    aa_components = {symbol: component for component, symbol in _AA3_TO_1.items()}
    expected = {}
    for entry in payload["sequences"]:
        chain = str(entry["chain_id"])
        segment = mapping["physical_to_logical"][chain]
        if (entry.get("modality") != "protein" or entry.get("modifications")
                or segment["logical_chain_id"] != chain
                or segment["observed_positions"] != list(range(len(entry["sequence"])))):
            return None
        for position, (symbol, locator) in enumerate(zip(entry["sequence"], segment["source_locators"], strict=True)):
            seq_id = str(position + 1)
            if (symbol not in aa_components or locator["chain_id"] != chain or locator["label_asym_id"] != chain
                    or str(locator["res_id"]) != seq_id or locator["label_seq_id"] != seq_id
                    or locator["auth_asym_id"] != chain or locator["auth_seq_id"] != seq_id
                    or locator["ins_code"] or locator["auth_ins_code"]):
                return None
            expected[(chain, seq_id)] = aa_components[symbol]
    if any(set((edge[2], edge[4])) != {"C", "N"} for edge in ledger.get("preserved_backbone_edges", [])):
        return None
    from biotite.structure.io import pdbx
    from ..utils.writer import _write_cif_fast, _safe_block_name, _chem_comp_category, COMPLETE_BACKBONE_CRITERIA

    file = pdbx.CIFFile.read(path)
    if len(file) != 1:
        raise ValueError("ESMFold2 native CIF must contain exactly one structure")
    block = next(iter(file.values()))
    # Nontrivial native topology needs the established full restoration path.
    if any(category in block for category in ("struct_conn", "chem_comp_bond", "chem_comp_atom")):
        return None
    site = block["atom_site"]
    columns = {name: site[name].as_array(str) for name in site}
    chains, positions = columns["label_asym_id"], columns["label_seq_id"]
    atom_names, components, elements = columns["label_atom_id"], columns["label_comp_id"], columns["type_symbol"]
    auth_chains = columns.get("auth_asym_id", chains)
    auth_positions = columns.get("auth_seq_id", positions)
    insertions = columns.get("pdbx_PDB_ins_code", np.full(len(chains), "."))
    inventory, atom_rows = {}, {}
    for row, (chain, position, atom, component, element, auth_chain, auth_position, ins) in enumerate(zip(
            chains, positions, atom_names, components, elements, auth_chains, auth_positions, insertions, strict=True)):
        key = (str(chain), str(position))
        atom_key = (*key, str(atom))
        if (expected.get(key) != component or auth_chain != chain or auth_position != position
                or ins not in {"", ".", "?"} or atom_key in atom_rows or element.upper() != atom[0]):
            raise ValueError("ESMFold2 native protein inventory differs from its source occurrence map")
        atom_rows[atom_key] = row
        inventory.setdefault(key, set()).add(str(atom))
    if set(inventory) != set(expected) or any(inventory[key] != set(PROTEIN_RES_ATOMS[component]) for key, component in expected.items()):
        raise ValueError("ESMFold2 native protein atom coverage differs from its source occurrence map")
    if not all(np.isfinite(columns[name].astype(float)).all() for name in ("Cartn_x", "Cartn_y", "Cartn_z")):
        raise ValueError("ESMFold2 native protein coordinates are nonfinite")
    # Source graph and native sequence facts, never predicted distances, own edges.
    edges = {}
    for entry in payload["sequences"]:
        chain = str(entry["chain_id"])
        segment = mapping["physical_to_logical"][chain]
        for position in range(1, len(segment["observed_positions"])):
            if not bool((segment["source_metadata"][position] or {}).get("source_break_before", False)):
                edge = tuple(sorted(((chain, str(position), "C"), (chain, str(position + 1), "N"))))
                edges[edge] = 1
    for chain, left, left_name, right, right_name, *orders in ledger.get("preserved_backbone_edges", []):
        edge = tuple(sorted(((str(chain), str(int(left) + 1), str(left_name)), (str(chain), str(int(right) + 1), str(right_name)))))
        order = int(orders[0]) if orders else 1
        if order != 0 or edge not in edges:
            edges[edge] = order
    connections = {name: [] for name in ("id", "conn_type_id", "ptnr1_label_asym_id", "ptnr1_label_comp_id",
        "ptnr1_label_seq_id", "ptnr1_label_atom_id", "ptnr1_auth_asym_id", "ptnr1_auth_seq_id", "pdbx_ptnr1_PDB_ins_code",
        "ptnr2_label_asym_id", "ptnr2_label_comp_id", "ptnr2_label_seq_id", "ptnr2_label_atom_id", "ptnr2_auth_asym_id",
        "ptnr2_auth_seq_id", "pdbx_ptnr2_PDB_ins_code", "pdbx_value_order")}
    order_names = {0: "?", 1: "sing", 2: "doub", 3: "trip"}
    for edge, order in sorted(edges.items()):
        if order not in order_names or any(endpoint not in atom_rows for endpoint in edge):
            raise ValueError("ESMFold2 source backbone endpoint or order is invalid")
        connections["id"].append(f"covale_{len(connections['id']) + 1:06d}")
        connections["conn_type_id"].append("covale")
        connections["pdbx_value_order"].append(order_names[order])
        for side, (chain, position, atom) in zip(("ptnr1", "ptnr2"), edge, strict=True):
            for suffix, value in (("label_asym_id", chain), ("label_comp_id", expected[(chain, position)]),
                                  ("label_seq_id", position), ("label_atom_id", atom), ("auth_asym_id", chain), ("auth_seq_id", position)):
                connections[f"{side}_{suffix}"].append(value)
            connections[f"pdbx_{side}_PDB_ins_code"].append(".")
    logical_ids = list(mapping["logical_index"])
    entity_ids = {chain: str(mapping["logical_index"][chain] + 1) for chain in logical_ids}
    columns["label_entity_id"] = [entity_ids[str(chain)] for chain in chains]
    columns["auth_asym_id"], columns["auth_seq_id"] = auth_chains, auth_positions
    unique_components = list(dict.fromkeys(expected.values()))
    categories = {"atom_site": columns,
                  "entity": {"id": list(entity_ids.values()), "type": ["polymer"] * len(entity_ids)},
                  "struct_asym": {"id": logical_ids, "entity_id": [entity_ids[chain] for chain in logical_ids]},
                  "chem_comp": _chem_comp_category(comp_id=np.asarray(unique_components), mol_type=np.full(len(unique_components), "protein")),
                  "struct_conn_type": {"id": ["covale"], "criteria": [COMPLETE_BACKBONE_CRITERIA], "reference": ["."]}}
    if connections["id"]:
        categories["struct_conn"] = connections
    import os
    import tempfile

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".tmp.cif", dir=target.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        _write_cif_fast(_safe_block_name(str(payload.get("sample_id", "refold"))), categories, temporary_path)
        os.replace(temporary_path, target)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return target


def restore_source_output(path: Path, payload: Mapping[str, Any], *, destination: Path | None = None) -> Path:
    from ..utils.writer import write_structure_atomic

    if SOURCE_MAPPING not in payload:
        if destination is not None and destination != path:
            import shutil
            shutil.copyfile(path, destination)
            return destination
        return path
    mapping = source_mapping(payload)
    target = Path(destination or path)
    projected = _project_continuous_protein_cif(path, payload, mapping, target)
    if projected is not None:
        return projected
    restored = _restored_source_atoms(path, payload, mapping)
    identities = {}
    if bool(np.any(restored.mol_type == "ligand")):
        from ..utils.ligand_identity import match_ligand_component_against_ccd
        from ..utils.source_mapping import source_locator

        # Only declared CCD occurrences can retain a named CCD component.
        # SMILES and unproved graphs retain the writer's existing local LIGxx IDs.
        declared = {}
        segments = {row["physical_chain_id"]: row for row in payload[SOURCE_MAPPING]["segments"]}
        for entry in payload["sequences"]:
            if entry.get("modality") != "ligand" or "ccd" not in entry:
                continue
            segment = segments[entry["chain_id"]]
            for component, locator in zip(entry["ccd"], segment["source_locators"], strict=True):
                if locator is not None:
                    declared[tuple(sorted(locator.items()))] = str(component)
        component_uids = np.asarray(restored.component_uid, dtype=np.int64)
        for uid in dict.fromkeys(component_uids[restored.mol_type == "ligand"].tolist()):
            rows = np.flatnonzero(component_uids == uid)
            expected = declared.get(tuple(sorted(source_locator(restored, int(rows[0])).items())))
            if expected is None or set(restored.res_name[rows].astype(str)) != {expected}:
                continue
            proof_atoms = restored[rows].copy()
            proof_atoms.identity_condition[:] = True
            identity = match_ligand_component_against_ccd(proof_atoms, range(len(proof_atoms)))
            if identity is not None:
                restored.identity_condition[rows] = True
                identities[int(uid)] = identity
    return write_structure_atomic(restored, destination or path,
                                  entry_id=str(payload.get("sample_id", "refold")),
                                  validated_ligand_identities=identities)


def mapped_confidence_summary(native: Mapping[str, Any]) -> dict[str, Any]:
    mapped = native.get("partidesign_source_confidence")
    if not isinstance(mapped, Mapping):
        return dict(native)
    result = dict(native)
    result.update(mapped)
    result["native_confidence"] = {key: value for key, value in native.items()
                                   if key != "partidesign_source_confidence"}
    return result


def _tm_normalization(length: int) -> float:
    return 1.24 * (max(int(length), 19) - 15) ** (1.0 / 3.0) - 1.8


def _softmax(values: Any, np: Any) -> Any:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def native_binder_confidence(
    output: Mapping[str, Any],
    features: Mapping[str, Any],
    chain_infos: list[Any],
    sample_index: int,
    payload: Mapping[str, Any],
    np: Any,
    segments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive chain-specific pTM/iPTM from native PAE logits.

    The formulas follow Protenix's chain_ptm/chain_iptm definitions while
    retaining ESMFold2's native PAE bins (0..32 A).  The declared design-chain
    ledger selects the binder mask; global pTM/iPTM are never substituted.
    """

    logits = output["pae_logits"][sample_index].detach().float().cpu().numpy()
    asym = features["asym_id"][0].detach().cpu().numpy().astype(np.int64)
    token_mask = features["token_attention_mask"][0].detach().cpu().numpy().astype(bool)
    if segments is None:
        # Preserve the native integer-mask path for ordinary unbroken inputs.
        token_chain_labels = asym
        chain_labels = {str(chain.chain_id): int(chain.asym_id) for chain in chain_infos}
    else:
        token_chain_labels, _logical_asym, _logical_positions, token_mask = mapped_token_inventory(
            asym, features["residue_index"][0].detach().cpu().numpy(), token_mask, payload, mapping=segments
        )
        chain_labels = {str(entry["chain_id"]): str(entry["chain_id"])
                        for entry in segments["logical_sequences"]}
    entries = payload["sequences"] if segments is None else segments["logical_sequences"]
    if (
        logits.ndim != 3
        or logits.shape[0] != token_chain_labels.shape[0]
        or logits.shape[1] != token_chain_labels.shape[0]
    ):
        raise RuntimeError("ESMFold2 native PAE logits/token ledger shape differs")
    bins = logits.shape[-1]
    centers = (np.arange(bins, dtype=np.float64) + 0.5) * (32.0 / bins)
    chain_masks: dict[str, Any] = {}
    for entry in entries:
        chain = str(entry.get("chain_id", ""))
        if not chain or chain in chain_masks:
            raise RuntimeError("ESMFold2 confidence logical chain ledger is malformed")
        chain_masks[chain] = token_mask & (token_chain_labels == chain_labels[chain])

    def ptm_for(mask: Any) -> float | None:
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return None
        selected = logits[np.ix_(indices, indices)]
        probs = _softmax(selected, np)
        weights = 1.0 / (1.0 + (centers / _tm_normalization(len(indices))) ** 2)
        expected = np.sum(probs * weights, axis=-1)
        value = float(np.max(np.mean(expected, axis=-1)))
        return value if math.isfinite(value) else None

    def iptm_for(left: Any, right: Any) -> float | None:
        indices = np.flatnonzero(left | right)
        if len(indices) == 0:
            return None
        selected = logits[np.ix_(indices, indices)]
        probs = _softmax(selected, np)
        weights = 1.0 / (1.0 + (centers / _tm_normalization(len(indices))) ** 2)
        expected = np.sum(probs * weights, axis=-1)
        selected_chains = token_chain_labels[indices]
        different_chain = selected_chains[:, None] != selected_chains[None, :]
        denominator = np.maximum(np.sum(different_chain, axis=-1), 1)
        row_values = np.sum(expected * different_chain, axis=-1) / denominator
        value = float(np.max(row_values))
        return value if math.isfinite(value) else None

    declared_design = [str(value) for value in payload.get("design_chains", [])]
    design = [chain for chain in declared_design if chain in chain_masks]
    context = list(dict.fromkeys(
        str(entry.get("chain_id"))
        for entry in entries
        if entry.get("role") == "context"
        and str(entry.get("chain_id")) in chain_masks
    ))
    chain_ptm = {chain: ptm_for(chain_masks[chain]) for chain in design}
    pair_values: dict[str, dict[str, float | None]] = {}
    chain_iptm: dict[str, float | None] = {}
    for binder in design:
        pair_values[binder] = {}
        values: list[float] = []
        for partner in context:
            value = iptm_for(chain_masks[binder], chain_masks[partner])
            pair_values[binder][partner] = value
            if value is not None:
                values.append(value)
        chain_iptm[binder] = float(np.mean(values)) if values else None
    ptm_values = [value for value in chain_ptm.values() if value is not None]
    iptm_values = [value for value in chain_iptm.values() if value is not None]
    if segments is None:
        asym_to_chain = {value: chain for chain, value in chain_labels.items()}
        output_chain_ids = [asym_to_chain.get(int(value), "") if bool(valid) else ""
                            for value, valid in zip(asym, token_mask, strict=True)]
    else:
        output_chain_ids = token_chain_labels[token_mask].tolist()
    result = {
        "binder_ptm": min(ptm_values) if ptm_values else None,
        "binder_iptm": min(iptm_values) if iptm_values else None,
        "binder_ptm_by_chain": chain_ptm,
        "binder_iptm_by_chain": chain_iptm,
        "binder_pair_iptm": pair_values,
        "token_chain_ids": output_chain_ids,
        "native_metric_definition": {
            "binder_ptm": "per declared design chain: max row mean of TM expectation over that chain; d0 uses that chain token count",
            "binder_iptm": "minimum over declared design chains of mean pair iPTM against each context chain; each pair uses its own design+context token mask and d0",
            "pae_logits": "native ESMFold2 PAE head logits with 0..32 A bin centers",
            "aggregation": "minimum across multiple declared design chains; missing context yields null binder_iptm",
        },
    }
    if segments is not None:
        # Confidence follows restored public chains, including all retained
        # heterogens. Pair scores are recomputed from the logical token masks;
        # averaging physical-fragment iPTM would use incompatible d0 values.
        result["per_chain_token_count"] = {
            chain: int(mask.sum()) for chain, mask in chain_masks.items()}
        if "plddt" in output:
            plddt = output["plddt"][sample_index].detach().float().cpu().numpy()
            result["per_chain_plddt"] = {
                chain: float(np.mean(plddt[mask]) * 100.0)
                for chain, mask in chain_masks.items() if bool(mask.any())}
        valid = np.flatnonzero(token_mask)
        probabilities = _softmax(logits[np.ix_(valid, valid)], np)
        weights = 1.0 / (1.0 + (centers / _tm_normalization(len(valid))) ** 2)
        expected = np.sum(probabilities * weights, axis=-1)
        selected_labels = token_chain_labels[valid]
        result["ptm"] = float(np.max(np.mean(expected, axis=-1)))
        different = selected_labels[:, None] != selected_labels[None, :]
        result["iptm"] = float(np.max(np.sum(expected * different, axis=-1) / np.maximum(different.sum(axis=-1), 1)))
        result["chain_pair_iptm"] = {
            left: {right: float(np.mean(expected[np.ix_(selected_labels == left, selected_labels == right)]))
                   for right in chain_masks}
            for left in chain_masks}
        result["native_metric_definition"]["chain_pair_iptm"] = (
            "native ESMFold2 directional pair mean of TM expectation; d0 uses all mapped tokens")
        for name in ("pae", "pde"):
            if name in output:
                matrix = output[name][sample_index].detach().float().cpu().numpy()
                result[f"{name}_mean"] = float(matrix[np.ix_(valid, valid)].mean())
        if all(name in features for name in ("atom_to_token", "atom_attention_mask", "distogram_atom_idx", "mol_type")) and all(
            name in output for name in ("plddt_per_atom", "sample_atom_coords")
        ):
            atom_to_token = features["atom_to_token"][0].detach().cpu().numpy().astype(np.int64)
            atom_mask = features["atom_attention_mask"][0].detach().cpu().numpy().astype(bool)
            atom_mask &= token_mask[atom_to_token]
            atom_plddt = output["plddt_per_atom"][sample_index].detach().float().cpu().numpy()
            if not atom_mask.any():
                raise RuntimeError("ESMFold2 mapped confidence has no retained atoms")
            result["plddt"] = float(atom_plddt[atom_mask].mean() * 100.0)
            representative = features["distogram_atom_idx"][0].detach().cpu().numpy().astype(np.int64)
            coordinates = output["sample_atom_coords"].detach().float().cpu().numpy()
            coordinates = coordinates[0, sample_index] if coordinates.ndim == 4 else coordinates[sample_index]
            token_xyz = coordinates[representative]
            distances = np.sqrt(np.sum((token_xyz[:, None] - token_xyz[None, :]) ** 2, axis=-1))
            is_ligand = features["mol_type"][0].detach().cpu().numpy().astype(np.int64) == 4
            interface = (distances < 8.0) & (token_chain_labels[:, None] != token_chain_labels[None, :])
            interface &= token_mask[:, None] & token_mask[None, :]
            token_weight = np.where(is_ligand, 2.0, interface.any(axis=-1).astype(float))
            atom_weight = atom_mask.astype(float) * token_weight[atom_to_token]
            result["complex_iplddt"] = float(np.sum(atom_plddt * atom_weight) / (atom_weight.sum() + 1e-6) * 100.0)
        unchanged = all(row["physical_chain_id"] == row["logical_chain_id"]
                        and row["observed_positions"] == list(range(len(row["observed_positions"])))
                        for row in segments["physical_to_logical"].values())
        if unchanged:
            for name in ("ptm", "iptm", "plddt", "complex_iplddt", "chain_pair_iptm", "pae_mean", "pde_mean"):
                result.pop(name, None)
    return result


