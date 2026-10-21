"""Private AF3 source projection for structures and native token confidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .af3_input import AF3InputError


def _category(block: Any, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
  from biotite.structure.io.pdbx import CIFCategory

  if not rows:
    if name in block:
      del block[name]
    return
  columns = list(dict.fromkeys(key for row in rows for key in row))
  block[name] = CIFCategory({key: [row.get(key, "?") for row in rows]
                            for key in columns}, name=name)


def rebuild_source_categories(
    block: Any, payload: Mapping[str, Any], rows: list[dict[str, Any]],
    native_lookup: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
) -> None:
  """Rebuild standard CIF identities and all observed polymer connections.

  Native fragment entities are one logical source entity in the published CIF.
  Only a confirmed complete sequence can create entity_poly_seq rows for source
  context; observed fragments never pretend to be a complete sequence.
  """
  from ..utils.source_mapping import COMPLETE_BACKBONE_CRITERIA

  chain_order = list(payload["source_to_native_chain"])
  chain_index = {chain: index for index, chain in enumerate(chain_order)}
  rows.sort(key=lambda row: (chain_index[row["chain_id"]],
                            row["source_observed_index"], row["native_atom_index"]))
  atom_site = block["atom_site"]
  selected = [row["native_atom_index"] for row in rows]
  for name in list(atom_site.keys()):
    atom_site[name] = atom_site[name].as_array()[selected].tolist()
  entities = {chain: str(index + 1) for index, chain in enumerate(chain_order)}
  replacements = {
      "id": [str(index + 1) for index in range(len(rows))],
      "label_asym_id": [row["label_asym_id"] for row in rows],
      "auth_asym_id": [row["auth_asym_id"] for row in rows],
      "label_seq_id": [row["label_seq_id"] for row in rows],
      "auth_seq_id": [row["auth_seq_id"] for row in rows],
      "pdbx_PDB_ins_code": [row["auth_ins_code"] or "?" for row in rows],
      "label_entity_id": [entities[row["chain_id"]] for row in rows],
      "label_comp_id": [row["res_name"] for row in rows],
      "label_atom_id": [row["atom_name"] for row in rows],
  }
  if "auth_atom_id" in atom_site:
    replacements["auth_atom_id"] = replacements["label_atom_id"]
  if "auth_comp_id" in atom_site:
    replacements["auth_comp_id"] = replacements["label_comp_id"]
  for name, values in replacements.items():
    atom_site[name] = values

  by_chain: dict[str, list[dict[str, Any]]] = {}
  by_residue: dict[tuple[str, int], list[dict[str, Any]]] = {}
  for row in rows:
    by_chain.setdefault(row["chain_id"], []).append(row)
    by_residue.setdefault((row["chain_id"], row["source_observed_index"]), []).append(row)
  entries = {str(chain["id"]): chain for chain in payload["chains"]}
  ledger_by_row = {int(atom["row"]): atom for atom in payload["source_atom_ledger"]}
  entity_rows, poly_rows, sequence_rows, poly_scheme, nonpoly_scheme = [], [], [], [], []
  for chain in chain_order:
    values = by_chain.get(chain)
    if not values:
      raise AF3InputError(f"published source chain {chain!r} has no observed atoms")
    modality = str(values[0]["mol_type"])
    entity = entities[chain]
    entity_rows.append({"id": entity, "type": "non-polymer" if modality == "ligand" else "polymer"})
    residues = [group[0] for key, group in by_residue.items() if key[0] == chain]
    if modality != "ligand":
      poly_rows.append({"entity_id": entity, "type": {
          "protein": "polypeptide(L)", "rna": "polyribonucleotide",
          "dna": "polydeoxyribonucleotide",
      }[modality], "pdbx_strand_id": residues[0]["auth_asym_id"]})
      # The shared source reader captures an authoritative full sequence once.
      # Confirmed full sequence positions must be legal public label positions.
      full = payload.get("source_polymer_sequences", {}).get(chain)
      full_positions_match = not full or all(
          int(residue["label_seq_id"]) == int(payload["source_residue_map"][residue["native_chain_id"]][residue["native_res_id"] - 1].get("source_polymer_position", 0)) for residue in residues
      )
      if not full_positions_match:
        raise AF3InputError(f"source chain {chain!r} label positions disagree with its confirmed full sequence")
      if full:
        sequence_rows.extend({"entity_id": entity, "num": str(position), "mon_id": str(component), "hetero": "n"}
                             for position, component in enumerate(full, start=1))
      elif all(entries[native]["input_semantics"] == "design" for native in payload["source_to_native_chain"][chain]):
        # A design entity itself defines the complete generated sequence.
        if [residue["label_seq_id"] for residue in residues] == [str(index + 1) for index in range(len(residues))]:
          sequence_rows.extend({"entity_id": entity, "num": residue["label_seq_id"], "mon_id": residue["res_name"], "hetero": "n"}
                               for residue in residues)
      for residue in residues:
        poly_scheme.append({"asym_id": chain, "entity_id": entity,
                            "seq_id": residue["label_seq_id"], "mon_id": residue["res_name"],
                            "pdb_seq_num": residue["auth_seq_id"], "auth_seq_num": residue["auth_seq_id"],
                            "pdb_mon_id": residue["res_name"], "auth_mon_id": residue["res_name"],
                            "pdb_strand_id": residue["auth_asym_id"], "pdb_ins_code": residue["auth_ins_code"] or "."})
    else:
      for position, residue in enumerate(residues, start=1):
        nonpoly_scheme.append({"asym_id": chain, "entity_id": entity,
                               "mon_id": residue["res_name"], "ndb_seq_num": str(position),
                               "pdb_seq_num": residue["auth_seq_id"], "auth_seq_num": residue["auth_seq_id"],
                               "pdb_mon_id": residue["res_name"], "auth_mon_id": residue["res_name"],
                               "pdb_strand_id": residue["auth_asym_id"], "pdb_ins_code": residue["auth_ins_code"] or "."})
  _category(block, "entity", entity_rows)
  _category(block, "struct_asym", [{"id": chain, "entity_id": entities[chain]} for chain in chain_order])
  _category(block, "entity_poly", poly_rows)
  _category(block, "entity_poly_seq", sequence_rows)
  _category(block, "pdbx_poly_seq_scheme", poly_scheme)
  _category(block, "pdbx_nonpoly_scheme", nonpoly_scheme)

  source_atoms = {int(row["source_atom_row"]): row for row in rows if row["source_atom_row"] is not None}
  connections: dict[tuple[int, int], dict[str, Any]] = {}
  row_indexes = {id(row): index for index, row in enumerate(rows)}

  def connect(left: Mapping[str, Any], right: Mapping[str, Any], order: str = "SING", kind: str = "covale") -> None:
    key = tuple(sorted((row_indexes[id(left)], row_indexes[id(right)])))
    record: dict[str, Any] = {"conn_type_id": kind, "pdbx_value_order": order}
    for side, atom in (("ptnr1", left), ("ptnr2", right)):
      record.update({f"{side}_label_asym_id": atom["label_asym_id"],
                     f"{side}_label_seq_id": atom["label_seq_id"],
                     f"{side}_label_comp_id": atom["res_name"], f"{side}_label_atom_id": atom["atom_name"],
                     f"{side}_auth_asym_id": atom["auth_asym_id"], f"{side}_auth_seq_id": atom["auth_seq_id"],
                     f"pdbx_{side}_PDB_ins_code": atom["auth_ins_code"] or ".", f"{side}_symmetry": "1_555"})
    previous = connections.get(key)
    if previous is not None and previous["pdbx_value_order"] != order and "?" not in {previous["pdbx_value_order"], order}:
      raise AF3InputError("source/native output bond orders disagree")
    connections[key] = record

  # Preserve native non-backbone connections only when both endpoints retain a
  # source occurrence. Native filler bonds cannot enter the restored graph.
  native_conn = block.get("struct_conn")
  if native_conn is not None:
    arrays = {str(name): native_conn[name].as_array(str) for name in native_conn.keys()}
    for index in range(native_conn.row_count):
      endpoints = []
      for side in ("ptnr1", "ptnr2"):
        try:
          residue = arrays[f"{side}_label_seq_id"][index]
          if residue in {".", "?"}:
            residue = arrays[f"{side}_auth_seq_id"][index]
          key = (arrays[f"{side}_label_asym_id"][index], int(residue),
                 arrays[f"{side}_label_comp_id"][index], arrays[f"{side}_label_atom_id"][index])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
          raise AF3InputError("native AF3 CIF struct_conn endpoint is malformed") from exc
        atom = native_lookup.get(key)
        if atom is None:
          mapping = payload["source_residue_map"].get(key[0], ())
          if 1 <= key[1] <= len(mapping) and mapping[key[1] - 1] is None:
            endpoints = []
            break
          raise AF3InputError(f"native CIF bond endpoint has no atom mapping: {key!r}")
        endpoints.append(atom)
      if endpoints:
        connect(*endpoints, order=arrays.get("pdbx_value_order", ["SING"] * native_conn.row_count)[index],
                kind=arrays.get("conn_type_id", ["covale"] * native_conn.row_count)[index])

  orders = {0: "?", 1: "SING", 2: "DOUB", 3: "TRIP", 9: "AROM"}
  for left, right, order in payload.get("source_inter_residue_bonds", ()):
    first, second = source_atoms.get(int(left)), source_atoms.get(int(right))
    if first is None or second is None:
      # Source OXT can be omitted by AF3. Every other hard source endpoint is
      # already required by the residue inventory check.
      missing = [ledger_by_row[int(row)] for row, atom in ((left, first), (right, second)) if atom is None]
      if any(atom["atom_name"] != "OXT" for atom in missing):
        raise AF3InputError("source chemical edge lost an output atom endpoint")
      continue
    if int(order) not in orders:
      raise AF3InputError(f"source inter-residue bond order {order!r} cannot be represented in CIF")
    connect(first, second, orders[int(order)])

  # Within an explicitly planned native fragment, consecutive mapped residues
  # are connected. No adjacency across fragments or unobserved filler is used.
  native_groups = _native_residue_groups(rows)
  for native, mapping in payload["source_residue_map"].items():
    modality = entries[native]["modality"]
    if modality == "ligand":
      continue
    left_name, right_name = ("C", "N") if modality == "protein" else ("O3'", "P")
    by_native_residue = {residue: {row["atom_name"]: row for row in values}
                         for (chain, residue), values in native_groups.items() if chain == native}
    for position, (left, right) in enumerate(zip(mapping, mapping[1:]), start=1):
      if left is None or right is None:
        continue
      if bool(right.get("source_break_before", False)) and entries[native]["role"] != "design":
        continue
      first = by_native_residue.get(position, {}).get(left_name)
      second = by_native_residue.get(position + 1, {}).get(right_name)
      if first is not None and second is not None:
        connect(first, second)
    if native in payload.get("cyclic_chains", ()) and entries[native]["role"] == "design" and len(mapping) > 1:
      first = by_native_residue.get(len(mapping), {}).get("C")
      second = by_native_residue.get(1, {}).get("N")
      if first is not None and second is not None:
        connect(first, second)
  connection_rows = [{"id": f"bond{index}", **record} for index, record in enumerate(connections.values(), start=1)]
  _category(block, "struct_conn", connection_rows)
  types = list(dict.fromkeys(["covale", *[record["conn_type_id"] for record in connection_rows]]))
  _category(block, "struct_conn_type", [{"id": kind, "criteria": COMPLETE_BACKBONE_CRITERIA if kind == "covale" else "Native supplied connection.", "reference": "."} for kind in types])


def _native_residue_groups(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], list[Mapping[str, Any]]]:
  result: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
  for row in rows:
    result.setdefault((str(row["native_chain_id"]), int(row["native_res_id"])), []).append(row)
  return result


def project_source_confidence(
    payload: Mapping[str, Any], native_chain_ids: Sequence[str],
    native_res_ids: Sequence[int], full_pae: Any, full_pde: Any,
) -> dict[str, Any]:
  """Project native token matrices in the exact restored logical/source order."""
  pae, pde = np.asarray(full_pae), np.asarray(full_pde)
  size = len(native_chain_ids)
  if len(native_res_ids) != size or pae.shape != (size, size) or pde.shape != (size, size):
    raise AF3InputError("native confidence token locators and PAE/PDE dimensions differ")
  order = {chain: index for index, chain in enumerate(payload["source_to_native_chain"])}
  selected: list[tuple[int, Mapping[str, Any]]] = []
  mask = []
  for index, (chain, position) in enumerate(zip(native_chain_ids, native_res_ids, strict=True)):
    mapping = payload["source_residue_map"].get(str(chain))
    if mapping is None or position < 1 or position > len(mapping):
      raise AF3InputError("native confidence token has no explicit source residue position")
    source = mapping[position - 1]
    mask.append(source is not None)
    if source is not None:
      selected.append((index, source))
  selected.sort(key=lambda item: (order[item[1]["chain_id"]], item[1]["source_observed_index"], item[0]))
  indexes = [index for index, _source in selected]
  locators = [dict(source) for _index, source in selected]
  return {
      "native_full_pae": full_pae, "native_full_pde": full_pde,
      "native_token_chain_ids": list(native_chain_ids), "native_token_res_ids": list(native_res_ids),
      "native_token_source_mask": mask, "source_token_indices": indexes,
      "source_token_residue_map": locators,
      "full_pae": pae[np.ix_(indexes, indexes)], "full_pde": pde[np.ix_(indexes, indexes)],
      "token_chain_ids": [source["chain_id"] for source in locators],
      "source_token_chain_ids": [source["chain_id"] for source in locators],
      "token_res_ids": [source["res_id"] for source in locators],
      "token_ins_codes": [source["ins_code"] for source in locators],
  }
