"""Private AF3 source/entity mapping validation.

The sequence planner is shared with the other refold backends.  This module
checks AF3's native residue positions independently of the source atom order.
It does not load a model or invent source occurrences for unresolved positions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any


_LOCATOR_FIELDS = (
    "chain_id", "res_id", "ins_code", "label_asym_id", "label_seq_id",
    "auth_asym_id", "auth_seq_id", "auth_ins_code", "source_format",
)


def validate_source_mapping(payload: Mapping[str, Any]) -> None:
  """Reject obsolete or inconsistent private prepared input mappings."""
  chains = payload.get("chains", ())
  forward = payload.get("source_to_native_chain")
  reverse = payload.get("native_to_source_chain")
  residues = payload.get("source_residue_map")
  ledger = payload.get("source_atom_ledger")
  source_bonds = payload.get("source_inter_residue_bonds")
  full_sequences = payload.get("source_polymer_sequences")
  if not all(isinstance(value, Mapping) for value in (forward, reverse, residues)):
    raise ValueError("AF3 prepared input has no explicit source/entity mapping; rebuild it")
  if not isinstance(ledger, Sequence) or isinstance(ledger, (str, bytes)):
    raise ValueError("AF3 prepared input has no source atom ledger; rebuild it")
  if (not isinstance(source_bonds, (tuple, list)) or not isinstance(full_sequences, Mapping)):
    raise ValueError("AF3 prepared input predates explicit source topology/full sequences; rebuild it")
  chain_by_id: dict[str, Mapping[str, Any]] = {}
  for chain in chains:
    native = str(chain["id"])
    if not re.fullmatch(r"[A-Z]+", native):
      raise ValueError(f"AF3 native chain ID is not uppercase alphabetic: {native!r}")
    if native in chain_by_id:
      raise ValueError(f"AF3 duplicate native chain {native!r}")
    chain_by_id[native] = chain
    if chain.get("role") not in {"design", "context"} or chain.get("input_semantics") not in {
        "design", "complete_sequence", "observed_fragments", "ligand",
    }:
      raise ValueError("AF3 prepared input predates explicit refold positions; rebuild it")
    if chain["role"] == "design" and chain.get("msa_reference") is not None:
      raise ValueError("AF3 design chains must remain query-only")
  native_ids = set(chain_by_id)
  if set(reverse) != native_ids or set(residues) != native_ids:
    raise ValueError("AF3 native chain mapping must cover each entity exactly once")
  flattened = []
  for source, native_values in forward.items():
    if not isinstance(native_values, (tuple, list)) or not native_values:
      raise ValueError("AF3 prepared input uses a legacy scalar chain mapping; rebuild it")
    for native in native_values:
      if native not in native_ids or reverse[native] != source:
        raise ValueError("AF3 source/entity chain mappings disagree")
      if chain_by_id[native].get("source_id") != source:
        raise ValueError("AF3 native entity has a different logical source chain")
      flattened.append(native)
  if len(flattened) != len(native_ids) or set(flattened) != native_ids:
    raise ValueError("AF3 source/entity chain mapping is not a partition")
  for source, sequence in full_sequences.items():
    if (source not in forward or not isinstance(sequence, (tuple, list)) or not sequence
        or any(not isinstance(component, str) or not component for component in sequence)):
      raise ValueError("AF3 confirmed full source sequence is missing or malformed")
    if any(chain_by_id[native]["role"] != "context" for native in forward[source]):
      raise ValueError("AF3 full source sequences may describe only context logical chains")
  ledger_by_row = {}
  for atom in ledger:
    if not isinstance(atom, Mapping) or any(key not in atom for key in _LOCATOR_FIELDS):
      raise ValueError("AF3 source atom ledger lacks source locators; rebuild it")
    row = atom.get("row")
    if type(row) is not int or row < 0 or row in ledger_by_row:
      raise ValueError("AF3 source atom rows are not unique non-negative integers")
    ledger_by_row[row] = atom
  mapped_rows: set[int] = set()
  row_occurrences: dict[int, tuple[str, int]] = {}
  observed_ordinals: dict[str, set[int]] = {}
  source_full_positions: dict[str, dict[int, int]] = {}
  for native, chain in chain_by_id.items():
    sequence_length = (len(chain.get("ccd_ids", ())) if chain["modality"] == "ligand"
                       else len(chain["sequence"]))
    mapping = residues[native]
    if not isinstance(mapping, (tuple, list)) or len(mapping) != sequence_length:
      raise ValueError(f"AF3 source residue map length differs from native chain {native!r}")
    source = reverse[native]
    if chain["input_semantics"] == "complete_sequence" and source not in full_sequences:
      raise ValueError("AF3 complete sequence entity has no confirmed full source sequence; rebuild it")
    for modification in chain.get("modifications", ()):
      if (not isinstance(modification, (tuple, list)) or len(modification) != 2
          or not isinstance(modification[0], str) or not modification[0]
          or type(modification[1]) is not int or not 1 <= modification[1] <= sequence_length):
        raise ValueError("AF3 modification must name a CCD and an in-range 1-based native position")
    for item in mapping:
      if item is None:
        if chain["role"] != "context" or chain["input_semantics"] != "complete_sequence":
          raise ValueError("AF3 unmapped native positions require confirmed context full sequence")
        continue
      required = _LOCATOR_FIELDS + (
          "source_chain_id", "source_res_id", "source_ins_code", "source_observed_index",
          "source_residue_uid", "source_token_uid", "row_indices",
          "source_polymer_position", "source_break_before", "source_graph_complete", "source_entity_id",
      )
      if not isinstance(item, Mapping) or any(key not in item for key in required):
        raise ValueError("AF3 source residue mapping lacks explicit occurrences; rebuild it")
      if item["source_chain_id"] != source or item["chain_id"] != source:
        raise ValueError("AF3 residue mapping disagrees with logical source chain")
      if item["source_res_id"] != item["res_id"] or item["source_ins_code"] != item["ins_code"]:
        raise ValueError("AF3 compatibility and canonical source locators disagree")
      if (type(item["source_polymer_position"]) is not int or item["source_polymer_position"] < 0
          or type(item["source_break_before"]) is not bool or type(item["source_graph_complete"]) is not bool
          or not isinstance(item["source_entity_id"], str)):
        raise ValueError("AF3 source occurrence topology/full position facts are malformed")
      ordinal = item["source_observed_index"]
      seen = observed_ordinals.setdefault(source, set())
      if type(ordinal) is not int or ordinal < 0 or ordinal in seen:
        raise ValueError("AF3 observed source residue is mapped more than once")
      seen.add(ordinal)
      if source in full_sequences:
        position = item["source_polymer_position"]
        if (not 1 <= position <= len(full_sequences[source]) or item["res_id"] != position
            or item["label_seq_id"] != str(position)):
          raise ValueError("AF3 confirmed context full position conflicts with source label/residue position")
        positions = source_full_positions.setdefault(source, {})
        positions[ordinal] = position
      rows = item["row_indices"]
      if not isinstance(rows, (tuple, list)) or not rows:
        raise ValueError("AF3 source occurrence has no atom rows")
      for row in rows:
        if type(row) is not int or row in mapped_rows or row not in ledger_by_row:
          raise ValueError("AF3 source atom occurrence is absent or mapped more than once")
        atom = ledger_by_row[row]
        if any(atom[key] != item[key] for key in _LOCATOR_FIELDS):
          raise ValueError("AF3 source atom and residue locators disagree")
        mapped_rows.add(row)
        row_occurrences[row] = (source, ordinal)
  selected_rows = {row for row, atom in ledger_by_row.items() if atom["chain_id"] in forward}
  if mapped_rows != selected_rows:
    raise ValueError("AF3 source mapping does not retain every selected source atom")
  for source, ordinals in observed_ordinals.items():
    if ordinals != set(range(len(ordinals))):
      raise ValueError(f"AF3 observed residue order is incomplete on logical chain {source!r}")
  for source, positions in source_full_positions.items():
    ordered = [positions[index] for index in sorted(positions)]
    if len(set(ordered)) != len(ordered) or ordered != sorted(ordered):
      raise ValueError("AF3 confirmed source full positions are not unique and ordered")
  from biotite.structure import BondType

  valid_orders = {int(order) for order in BondType}
  seen_bonds = set()
  for bond in source_bonds:
    if (not isinstance(bond, (tuple, list)) or len(bond) != 3
        or any(type(value) is not int for value in bond)
        or bond[0] not in mapped_rows or bond[1] not in mapped_rows
        or row_occurrences[bond[0]] == row_occurrences[bond[1]] or bond[2] not in valid_orders):
      raise ValueError("AF3 source inter-residue bond has an invalid mapped row/order/occurrence")
    key = tuple(sorted(bond[:2]))
    if key in seen_bonds:
      raise ValueError("AF3 source inter-residue bond is repeated or has conflicting orders")
    seen_bonds.add(key)
  for pair in payload.get("bonded_atom_pairs", ()):
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
      raise ValueError("AF3 bond must contain exactly two native atom endpoints")
    for endpoint in pair:
      if (not isinstance(endpoint, (tuple, list)) or len(endpoint) != 3
          or endpoint[0] not in native_ids or type(endpoint[1]) is not int
          or not 1 <= endpoint[1] <= len(residues[endpoint[0]])
          or not isinstance(endpoint[2], str) or not endpoint[2]):
        raise ValueError("AF3 bond endpoint has an invalid native chain/residue/atom position")
      item = residues[endpoint[0]][endpoint[1] - 1]
      if item is None or endpoint[2] not in {ledger_by_row[row]["atom_name"] for row in item["row_indices"]}:
        raise ValueError("AF3 bond endpoint has no observed source atom occurrence")


def _plain(value: Any) -> Any:
  if isinstance(value, Mapping):
    return {str(key): _plain(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [_plain(item) for item in value]
  return value


def validate_prepared_request(prepared: Mapping[str, Any], current: Mapping[str, Any]) -> None:
  """Compare source/request facts directly, independently of MSA cache state."""
  validate_source_mapping(prepared)
  fields = (
      "name", "protocol", "rng_seeds", "bonded_atom_pairs", "user_ccd",
      "source_to_native_chain", "native_to_source_chain", "cyclic_chains", "cyclic_lengths",
      "source_residue_map", "source_atom_ledger", "native_component_map", "component_rows",
      "source_inter_residue_bonds", "source_polymer_sequences",
  )
  for key in fields:
    if _plain(prepared.get(key)) != _plain(current.get(key)):
      raise ValueError(f"AF3 prepared input {key} differs from current source/request; rebuild it")
  def chain_facts(payload: Mapping[str, Any]) -> Any:
    return _plain([{key: value for key, value in chain.items() if key != "msa_reference"}
                   for chain in payload["chains"]])
  if chain_facts(prepared) != chain_facts(current):
    raise ValueError("AF3 prepared native sequences/entities differ from current source/request; rebuild it")
