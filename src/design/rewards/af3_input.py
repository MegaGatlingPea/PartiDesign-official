"""Native AlphaFold 3 input and output ledgers.

The standalone design package owns the boundary between its canonical
``AtomArray`` records and the public AlphaFold 3 input dataclasses.  This
module deliberately does not import the AlphaFold 3 SDK at module import time:
CPU validation and planning can therefore inspect inputs without loading JAX
or the native CIF extension.  ``AF3Input.to_folding_input`` performs the lazy
SDK conversion in the long lived AF3 worker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any

import numpy as np


class AF3InputError(ValueError):
  """Raised when a design record cannot be represented as native AF3 input."""


_PROTEIN_1_TO_3 = dict(
    zip(
        "ARNDCQEGHILKMFPSTWYV",
        (
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ),
    )
)
_PROTEIN_3_TO_1 = {value: key for key, value in _PROTEIN_1_TO_3.items()}
_RNA_CODES = {"A", "G", "C", "U", "N"}
_DNA_CODES = {"A", "G", "C", "T", "N", "DA", "DG", "DC", "DT", "DN"}
_RNA_MOD_PARENT = {
    "PSU": "U",
    "5MC": "C",
    "OMU": "U",
    "1MA": "A",
    "2MG": "G",
    "M2G": "G",
    "7MG": "G",
    "OMG": "G",
    "H2U": "U",
    "5MU": "U",
}
_DNA_MOD_PARENT = {
    "5CM": "DC",
    "5HM": "DC",
    "5FC": "DC",
    "5HC": "DC",
}
_PROTEIN_MOD_PARENT = {
    "MSE": "MET",
    "FME": "MET",
    "SEP": "SER",
    "TPO": "THR",
    "PTR": "TYR",
    "SEC": "CYS",
    "CSO": "CYS",
    "CME": "CYS",
    "PYL": "LYS",
    "KCX": "LYS",
    "LLP": "LYS",
    "MLZ": "LYS",
    "HYP": "PRO",
}

# These are common CCD IDs shipped by the public AF3 database.  Unrecognised
# component names are represented in a stable user CCD namespace below.  The
# worker may pass its complete CCD name set to avoid this conservative list.
_COMMON_CCD_IDS = frozenset(
    {
        "HOH",
        "WAT",
        "DOD",
        "NA",
        "K",
        "CL",
        "CA",
        "MG",
        "MN",
        "FE",
        "ZN",
        "CU",
        "ATP",
        "ADP",
        "AMP",
        "GTP",
        "GDP",
        "CTP",
        "UTP",
        "NAD",
        "FAD",
        "HEM",
        "SAM",
    }
)
_BRACKETED_COMPONENT = re.compile(r"\[([^\]]+)\]")
_CCD_DATA_BLOCK = re.compile(r"(?m)^data_([^\s]+)\s*$")


def _mapping(value: Any) -> Mapping[str, Any]:
  if isinstance(value, Mapping):
    return value
  if hasattr(value, "__dict__"):
    return vars(value)
  return {}


def _record_atoms(record: Any) -> Any:
  atoms = record.get("atoms") if isinstance(record, Mapping) else getattr(record, "atoms", None)
  if atoms is None:
    raise AF3InputError("AF3 input record must expose an atoms AtomArray")
  return atoms


def _record_design_chains(record: Any, atoms: Any | None = None) -> dict[str, str]:
  raw = record.get("design_chains") if isinstance(record, Mapping) else getattr(record, "design_chains", None)
  if raw is None and not isinstance(record, Mapping):
    base = getattr(record, "base_request", None)
    task = getattr(base, "task", None)
    raw = getattr(task, "design_chains", None)
  result: dict[str, str] = {}

  def _inferred_modality(chain_id: str) -> str:
    if atoms is None:
      return "protein"
    from ..utils.source_mapping import canonical_chain_ids

    chain_values = canonical_chain_ids(atoms)
    rows = np.flatnonzero(chain_values == str(chain_id))
    if len(rows) == 0:
      return "protein"
    modality_values = {
        str(value).strip().lower()
        for value in np.asarray(atoms.mol_type, dtype=object)[rows]
    }
    if len(modality_values) != 1:
      raise AF3InputError(
          f"chain {chain_id!r} contains mixed modalities: {sorted(modality_values)}"
      )
    return next(iter(modality_values))

  if isinstance(raw, Mapping):
    for key, value in raw.items():
      result[str(key)] = str(value.get("modality", value) if isinstance(value, Mapping) else value)
    return result
  for item in raw or ():
    if isinstance(item, str):
      result[item] = _inferred_modality(item)
    elif isinstance(item, Mapping):
      if len(item) == 1 and "chain_id" not in item:
        key, value = next(iter(item.items()))
        result[str(key)] = str(value)
      else:
        key = item.get("chain_id", item.get("id"))
        if key is not None:
          result[str(key)] = str(item.get("modality", "protein"))
    else:
      key = getattr(item, "chain_id", getattr(item, "id", None))
      if key is not None:
        result[str(key)] = str(getattr(item, "modality", "protein"))
  return result


def _record_ligand_entries(record: Any) -> Mapping[str, Any]:
  value = (
      record.get("ligand_entries")
      if isinstance(record, Mapping)
      else getattr(record, "ligand_entries", None)
  )
  return value if isinstance(value, Mapping) else {}


def _private_ligand_molecule(
    record: Any,
    variant: Any,
    atoms: Any,
    chain_id: str,
    rows: np.ndarray,
) -> Any | None:
  """Find a private molecule sidecar without serializing it into native JSON."""
  keys: list[Any] = [str(chain_id)]
  if hasattr(atoms, "molecule_uid"):
    keys.extend(
        int(value)
        for value in dict.fromkeys(np.asarray(atoms.molecule_uid)[rows].tolist())
    )
  for owner in (variant, record):
    owner_map = _mapping(owner)
    for field_name in (
        "private_ligand_molecules",
        "ligand_molecules",
    ):
      container = owner_map.get(field_name)
      if container is None:
        container = getattr(owner, field_name, None)
      if not isinstance(container, Mapping):
        continue
      for key in keys:
        if key in container:
          return container[key]
  return None


def _variant_mapping(variant: Any) -> Mapping[str, Any]:
  return _mapping(variant)


def _variant_sequences(variant: Any) -> dict[str, Any]:
  raw = _variant_mapping(variant).get("sequences", {})
  result: dict[str, Any] = {}
  if isinstance(raw, Mapping):
    for key, value in raw.items():
      if isinstance(value, Mapping):
        # Preserve ligand-only entries (ccd_ids/smiles) for the chain builder;
        # polymer entries are reduced to their explicit sequence below.
        if "sequence" in value or "seq" in value:
          result[str(key)] = value.get("sequence", value.get("seq", ""))
        else:
          result[str(key)] = dict(value)
      else:
        result[str(key)] = value
  elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
    for value in raw:
      entry = _mapping(value)
      key = entry.get("chain_id", entry.get("id"))
      if key is not None:
        if any(
            field_name in entry
            for field_name in ("molecule", "smiles", "ccd_ids", "ccdCodes")
        ):
          # Keep private/generated ligand chemistry in-process.  It is
          # consumed into a named CCD before the native JSON payload.
          result[str(key)] = dict(entry)
        else:
          result[str(key)] = entry.get("sequence", entry.get("seq", ""))
  return result


def _variant_settings(record: Any, variant: Any) -> dict[str, Any]:
  result: dict[str, Any] = {}
  record_map = _mapping(record)
  variant_map = _variant_mapping(variant)
  for source in (record_map.get("settings"), variant_map.get("settings")):
    if isinstance(source, Mapping):
      result.update(source)
  result.update({key: value for key, value in record_map.items() if key in {
      "protocol", "mode", "cyclic", "is_cyclic", "cyclic_offset", "cycle",
      "user_ccd", "userCCD", "custom_ccd", "customCCD",
      "fixed_modified_residue_names", "sample", "sample_id",
  }})
  result.update({key: value for key, value in variant_map.items() if key in {
      "protocol", "mode", "cyclic", "is_cyclic", "cyclic_offset",
      "cycle", "user_ccd", "userCCD", "custom_ccd", "customCCD",
      "fixed_modified_residue_names", "sample", "sample_id",
  }})
  return result


def _chain_rows(atoms: Any, chain_id: str) -> np.ndarray:
  from ..utils.source_mapping import canonical_chain_ids

  return np.flatnonzero(canonical_chain_ids(atoms) == str(chain_id))


def _residue_groups(atoms: Any, rows: np.ndarray) -> list[np.ndarray]:
  if len(rows) == 0:
    return []
  if np.all(np.asarray(atoms.mol_type)[rows] != "ligand"):
    from ..utils.source_mapping import canonical_chain_ids, polymer_residue_groups

    chain = str(canonical_chain_ids(atoms)[int(rows[0])])
    return [np.asarray(group["row_indices"], dtype=np.int64)
            for group in polymer_residue_groups(atoms, chain)]
  groups: list[np.ndarray] = []
  seen: set[Any] = set()
  residue_uids = np.asarray(getattr(atoms, "residue_uid", np.full(len(atoms), -1)))
  if residue_uids.shape == (len(atoms),) and not np.all(residue_uids == -1):
    keys = residue_uids[rows]
  else:
    ins_codes = np.asarray(getattr(atoms, "ins_code", np.full(len(atoms), "")))
    keys = np.asarray(
        [(int(atoms.res_id[row]), str(ins_codes[row])) for row in rows],
        dtype=object,
    )
  for key in keys.tolist():
    key = tuple(key) if isinstance(key, list) else key
    if key in seen:
      continue
    seen.add(key)
    if np.asarray(keys).ndim == 2:
      mask = np.all(
          np.asarray(keys) == np.asarray(key, dtype=object).reshape(1, -1), axis=1
      )
    else:
      mask = np.asarray(keys) == key
    groups.append(rows[mask])
  return groups


def _name_for_group(atoms: Any, rows: np.ndarray) -> str:
  names = {str(value).strip().upper() for value in np.asarray(atoms.res_name)[rows]}
  if len(names) != 1:
    raise AF3InputError(f"residue has mixed component names: {sorted(names)}")
  return next(iter(names))


def _one_letter(name: str, modality: str) -> str:
  name = str(name).strip().upper()
  if modality == "protein":
    return _PROTEIN_3_TO_1.get(_PROTEIN_MOD_PARENT.get(name, name), "X")
  if modality == "rna":
    parent = _RNA_MOD_PARENT.get(name, name)
    return parent if parent in _RNA_CODES else "N"
  if modality == "dna":
    parent = _DNA_MOD_PARENT.get(name, name)
    return {
        "A": "A",
        "G": "G",
        "C": "C",
        "T": "T",
        "N": "N",
        "DA": "A",
        "DG": "G",
        "DC": "C",
        "DT": "T",
        "DN": "N",
    }.get(parent, "N")
  return "X"


def _native_modification_letter(name: str, modality: str) -> str:
  """Use the SDK's actual query encoding for a fixed CCD occurrence."""
  try:
    from alphafold3.constants import residue_names
  except ImportError:
    # Planning ordinary/common modifications remains SDK-independent. The
    # installed SDK supplies the authoritative encoding before actual folding.
    return _one_letter(name, modality)
  return residue_names.letters_three_to_one(name, default="X" if modality == "protein" else "N")


def _sequence_tokens(value: Any, modality: str) -> list[str]:
  if value is None:
    return []
  text = str(value).strip()
  if not text:
    return []
  tokens: list[str] = []
  cursor = 0
  for match in re.finditer(r"[A-Za-z]|\[[A-Za-z0-9_]+\]", text):
    if match.start() != cursor:
      raise AF3InputError(f"invalid {modality} sequence near {text[cursor:]!r}")
    token = match.group(0)
    tokens.append(token[1:-1].upper() if token.startswith("[") else token.upper())
    cursor = match.end()
  if cursor != len(text):
    raise AF3InputError(f"invalid {modality} sequence near {text[cursor:]!r}")
  return tokens


def _native_chain_id(source_id: str, used: set[str], ordinal: int) -> str:
  candidate = str(source_id)
  if re.fullmatch(r"[A-Z]+", candidate) and candidate not in used:
    used.add(candidate)
    return candidate
  # Uppercase alphabetic identifiers remain valid AF3 IDs after 26 chains.
  index = int(ordinal)
  while True:
    number = index + 1
    letters = ""
    while number:
      number, rem = divmod(number - 1, 26)
      letters = chr(ord("A") + rem) + letters
    if letters not in used:
      used.add(letters)
      return letters
    index += 1


def _stable_custom_ids(
    atoms: Any,
    chain_ids: Sequence[str],
    known_ccd_ids: set[str] | None,
    native_ccd: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
  """Assign deterministic UCCD IDs only when native chemistry is unavailable.

  A declared/native CCD entry is accepted only when its atom names, elements,
  charges (when authored), and heavy-atom graph match the source residue.
  Mismatches intentionally fall through to a private user CCD rather than
  silently substituting a same-named component.
  """
  known = set(_COMMON_CCD_IDS)
  if known_ccd_ids is not None:
    known.update(str(value).upper() for value in known_ccd_ids)
  remap: dict[str, str] = {}
  components: dict[str, list[dict[str, Any]]] = {}
  native_occurrences: dict[str, bool] = {}
  next_ordinal = 1
  for chain_id in chain_ids:
    rows = _chain_rows(atoms, chain_id)
    modality = {str(value) for value in np.asarray(atoms.mol_type)[rows]}
    if modality != {"ligand"}:
      continue
    for group in _residue_groups(atoms, rows):
      source_name = _name_for_group(atoms, group)
      native_match = False
      if native_ccd is not None and source_name in native_ccd:
        try:
          from .af3_chemistry import native_component_matches
          native_match = native_component_matches(atoms, group, native_ccd[source_name])
        except Exception as exc:
          raise AF3InputError(
              f"native CCD component {source_name!r} cannot be validated"
          ) from exc
      uses_native = source_name in known and (
          native_ccd is None or source_name not in native_ccd or native_match
      )
      previous_native = native_occurrences.get(source_name)
      if previous_native is not None and previous_native != uses_native:
        raise AF3InputError(
            f"source CCD component {source_name!r} has inconsistent native/custom occurrences"
        )
      native_occurrences[source_name] = uses_native
      if uses_native:
        continue
      if source_name not in remap:
        remap[source_name] = f"UCCD_{next_ordinal:03d}"
        next_ordinal += 1
      native_name = remap[source_name]
      components.setdefault(native_name, []).append({
          "source_name": source_name,
          "rows": group.copy(),
      })
  return remap, components


def _bond_order_name(order: int) -> tuple[str, str]:
  # Biotite's enum values are stable, but accepting string-like values keeps
  # this adapter usable with lightweight AtomArray fixtures.
  mapping = {1: ("SING", "N"), 2: ("DOUB", "N"), 3: ("TRIP", "N"), 4: ("AROM", "Y")}
  return mapping.get(int(order), ("SING", "N"))


def _named_molecule_signature(materialized: Any) -> tuple[tuple[str, ...], str]:
  """Compare trusted chemistry/stereo in the fixed source atom-name namespace."""
  from rdkit import Chem

  names = tuple(sorted(materialized.atom_names))
  name_numbers = {name: index + 1 for index, name in enumerate(names)}
  molecule = Chem.Mol(materialized.molecule)
  for atom, name in zip(molecule.GetAtoms(), materialized.atom_names, strict=True):
    atom.SetAtomMapNum(name_numbers[name])
  params = Chem.SmilesWriteParams()
  params.canonical = True
  params.doIsomericSmiles = True
  params.allBondsExplicit = True
  params.allHsExplicit = True
  # Include enhanced/atropisomer stereo while excluding coordinates and
  # arbitrary caller atom properties from the chemical identity comparison.
  flags = (
      Chem.CXSmilesFields.CX_ENHANCEDSTEREO
      | Chem.CXSmilesFields.CX_BOND_ATROPISOMER
      | Chem.CXSmilesFields.CX_BOND_CFG
      | Chem.CXSmilesFields.CX_RADICALS
  )
  return names, Chem.MolToCXSmiles(molecule, params, flags)


def _source_ledger(atoms: Any) -> list[dict[str, Any]]:
  from ..utils.source_mapping import source_locator

  result: list[dict[str, Any]] = []
  for row in range(len(atoms)):
    result.append({
        "row": int(row),
        "atom_uid": int(getattr(atoms, "atom_uid", np.arange(len(atoms)))[row]),
        "chain_id": str(atoms.chain_id[row]),
        "res_id": int(atoms.res_id[row]),
        "ins_code": str(getattr(atoms, "ins_code", np.full(len(atoms), ""))[row]),
        "residue_uid": int(getattr(atoms, "residue_uid", np.full(len(atoms), -1))[row]),
        "token_uid": int(getattr(atoms, "token_uid", np.full(len(atoms), -1))[row]),
        "res_name": str(atoms.res_name[row]),
        "atom_name": str(atoms.atom_name[row]),
        "element": str(getattr(atoms, "element", np.full(len(atoms), ""))[row]),
        "mol_type": str(atoms.mol_type[row]),
        "is_design": bool(getattr(atoms, "is_design", np.zeros(len(atoms), dtype=bool))[row]),
        **source_locator(atoms, row),
    })
  return result


def _source_occurrence_facts(atoms: Any, row: int) -> dict[str, Any]:
  """Small source facts kept once per observed residue, never per filler."""
  def value(field_name: str, default: Any) -> Any:
    values = getattr(atoms, field_name, None)
    return default if values is None else values[row]

  return {
      "source_polymer_position": int(value("source_polymer_position", 0)),
      "source_break_before": bool(value("source_break_before", False)),
      "source_graph_complete": bool(value("source_graph_complete", False)),
      "source_entity_id": str(value("label_entity_id", "")),
  }


@dataclass(frozen=True, slots=True)
class AF3Input:
  """SDK-independent AF3 request plus stable source/product ledgers."""

  name: str
  chains: tuple[Mapping[str, Any], ...]
  rng_seeds: tuple[int, ...]
  bonded_atom_pairs: tuple[tuple[tuple[str, int, str], tuple[str, int, str]], ...]
  user_ccd: str | None
  source_to_native_chain: Mapping[str, tuple[str, ...]]
  native_to_source_chain: Mapping[str, str]
  cyclic_chains: tuple[str, ...] = ()
  cyclic_lengths: Mapping[str, int] = field(default_factory=dict)
  source_residue_map: Mapping[str, tuple[Mapping[str, Any] | None, ...]] = field(default_factory=dict)
  source_atom_ledger: tuple[Mapping[str, Any], ...] = ()
  source_inter_residue_bonds: tuple[tuple[int, int, int], ...] = ()
  source_polymer_sequences: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
  native_component_map: Mapping[str, str] = field(default_factory=dict)
  component_rows: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
  protocol: str = "complex"
  msa_context: Mapping[str, Any] = field(default_factory=dict)

  @classmethod
  def from_payload(cls, payload: Mapping[str, Any]) -> "AF3Input":
    """Restore a private prepared request without rebuilding its chemistry."""
    from .af3_mapping import validate_source_mapping

    try:
      validate_source_mapping(payload)
    except (TypeError, KeyError, ValueError) as exc:
      raise AF3InputError(str(exc)) from exc
    values = dict(payload)
    for key in ("chains", "rng_seeds", "cyclic_chains", "source_atom_ledger"):
      if key in values:
        values[key] = tuple(values[key])
    values["bonded_atom_pairs"] = tuple(
        (tuple(left), tuple(right)) for left, right in values.get("bonded_atom_pairs", ())
    )
    values["source_inter_residue_bonds"] = tuple(
        tuple(bond) for bond in values["source_inter_residue_bonds"]
    )
    values["source_polymer_sequences"] = {
        source: tuple(sequence) for source, sequence in values["source_polymer_sequences"].items()
    }
    for key in ("source_residue_map", "component_rows"):
      values[key] = {name: tuple(rows) for name, rows in values.get(key, {}).items()}
    values["source_to_native_chain"] = {
        name: tuple(native_ids) for name, native_ids in values["source_to_native_chain"].items()
    }
    return cls(**values)

  @property
  def chain_ids(self) -> tuple[str, ...]:
    return tuple(str(entry["id"]) for entry in self.chains)

  def to_folding_input(self) -> Any:
    """Convert to the SDK's ``folding_input.Input`` lazily."""
    try:
      from alphafold3.common import folding_input
    except Exception as exc:  # pragma: no cover - exercised in SDK env
      raise AF3InputError("AlphaFold 3 SDK is not importable") from exc

    native_chains: list[Any] = []
    for entry in self.chains:
      modality = str(entry["modality"])
      kwargs = {
          "id": str(entry["id"]),
          "description": entry.get("description"),
      }
      if modality in {"protein", "rna", "dna"}:
        kwargs["sequence"] = str(entry["sequence"])
      modifications = tuple(
          (str(item[0]), int(item[1]))
          for item in entry.get("modifications", ())
      )
      if modality == "protein":
        reference = entry.get("msa_reference")
        templates = ()
        paired_msa = unpaired_msa = ""
        if reference is not None:
          _validate_msa_reference(entry, reference)
          paired_msa = Path(reference["paired_msa_path"]).read_text()
          unpaired_msa = Path(reference["unpaired_msa_path"]).read_text()
          templates = tuple(folding_input.Template(
              mmcif=Path(template["mmcif_path"]).read_text(),
              query_to_template_map={int(k): int(v) for k, v in template["query_to_template_map"].items()},
          ) for template in reference.get("templates", ()))
        native_chains.append(
            folding_input.ProteinChain(
                ptms=modifications,
                paired_msa=paired_msa,
                unpaired_msa=unpaired_msa,
                templates=templates,
                **kwargs,
            )
        )
      elif modality == "rna":
        reference = entry.get("msa_reference")
        unpaired_msa = ""
        if reference is not None:
          _validate_msa_reference(entry, reference)
          unpaired_msa = Path(reference["unpaired_msa_path"]).read_text()
        native_chains.append(
            folding_input.RnaChain(
                modifications=modifications,
                unpaired_msa=unpaired_msa,
                **kwargs,
            )
        )
      elif modality == "dna":
        native_chains.append(
            folding_input.DnaChain(
                modifications=modifications,
                **kwargs,
            )
        )
      elif modality == "ligand":
        if entry.get("smiles") is not None:
          native_chains.append(
              folding_input.Ligand(
                  id=str(entry["id"]),
                  smiles=str(entry["smiles"]),
                  description=entry.get("description"),
              )
          )
        else:
          native_chains.append(
              folding_input.Ligand(
                  id=str(entry["id"]),
                  ccd_ids=tuple(entry["ccd_ids"]),
                  description=entry.get("description"),
              )
          )
      else:
        raise AF3InputError(f"unsupported AF3 modality {modality!r}")
    return folding_input.Input(
        name=self.name,
        chains=native_chains,
        rng_seeds=self.rng_seeds,
        bonded_atom_pairs=self.bonded_atom_pairs or None,
        user_ccd=self.user_ccd,
    )

  def payload(self) -> dict[str, Any]:
    """Return the small worker payload used by ``AF3Worker.predict``."""
    return {
        "name": self.name,
        "protocol": self.protocol,
        "chains": [dict(entry) for entry in self.chains],
        "rng_seeds": list(self.rng_seeds),
        "bonded_atom_pairs": self.bonded_atom_pairs,
        "user_ccd": self.user_ccd,
        "source_to_native_chain": dict(self.source_to_native_chain),
        "native_to_source_chain": dict(self.native_to_source_chain),
      "cyclic_chains": list(self.cyclic_chains),
      "cyclic_lengths": dict(self.cyclic_lengths),
      "source_residue_map": {
          key: [None if item is None else dict(item) for item in value]
          for key, value in self.source_residue_map.items()
      },
        "source_atom_ledger": list(self.source_atom_ledger),
        "source_inter_residue_bonds": self.source_inter_residue_bonds,
        "source_polymer_sequences": {
            source: list(sequence) for source, sequence in self.source_polymer_sequences.items()
        },
        "native_component_map": dict(self.native_component_map),
        "component_rows": {key: list(value) for key, value in self.component_rows.items()},
        "msa_context": dict(self.msa_context),
    }


def _a3m_rows(text: str, *, path: Path) -> list[str]:
  """Parse native FASTA rows without mistaking line wrapping for columns."""
  rows: list[str] = []
  pieces: list[str] | None = None
  for raw in text.splitlines():
    line = raw.strip()
    if not line:
      continue
    if line.startswith(">"):
      if not line[1:].strip():
        raise AF3InputError(f"Cached A3M has an empty header: {path}")
      if pieces is not None:
        rows.append("".join(pieces))
      pieces = []
    elif pieces is None:
      raise AF3InputError(f"Cached A3M has no FASTA header: {path}")
    else:
      pieces.append(line)
  if pieces is not None:
    rows.append("".join(pieces))
  return rows


def _template_index(value: Any) -> int:
  if type(value) is int:
    return value
  if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
    return int(value)
  raise AF3InputError(f"Cached template index is not an integer: {value!r}")


def _validate_template_reference(template: Mapping[str, Any], query_length: int) -> None:
  raw_path = template.get("mmcif_path")
  if not isinstance(raw_path, (str, Path)) or not str(raw_path):
    raise AF3InputError("Cached template has no mmCIF path")
  path = Path(raw_path)
  if not path.is_file():
    raise AF3InputError(f"Missing cached template mmCIF: {path}")
  raw_mapping = template.get("query_to_template_map")
  if not isinstance(raw_mapping, Mapping):
    raise AF3InputError("Cached template query mapping must be a mapping")
  mapping = {_template_index(key): _template_index(value) for key, value in raw_mapping.items()}
  if len(mapping) != len(raw_mapping):
    raise AF3InputError("Cached template query indices collide after integer conversion")
  if any(key < 0 or key >= query_length or value < 0 for key, value in mapping.items()):
    raise AF3InputError("Cached template query/template index is out of range")
  # Validate against the same native residue sequence that AF3's template
  # features index, including unresolved template residues. Atom row count
  # and maximum author number are not template coordinate dimensions.
  try:
    from alphafold3 import structure

    native = structure.from_mmcif(
        mmcif_string=path.read_text(), fix_mse_residues=True,
        fix_arginines=True, include_water=False, include_bonds=False, include_other=True,
    )
    polymer_chains = native.polymer_auth_asym_id_to_label_asym_id()
    if len(polymer_chains) != 1:
      raise AF3InputError("Cached template must contain exactly one polymer chain")
    label = next(iter(polymer_chains.values()))
    length = len(native.chain_single_letter_sequence(include_missing_residues=True)[label])
  except AF3InputError:
    raise
  except Exception as exc:
    raise AF3InputError(f"Cached template is not native AF3-readable: {path}") from exc
  if any(value >= length for value in mapping.values()):
    raise AF3InputError(f"Cached template residue index is out of range for {path}")


def _validate_msa_reference(chain: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
  if not isinstance(reference, Mapping):
    raise AF3InputError("MSA reference must be a mapping")
  if reference.get("modality") != chain["modality"] or reference.get("sequence") != chain["sequence"]:
    raise AF3InputError(f"MSA cache query differs from native chain {chain['id']}")
  for key in ("unpaired_msa_path", "paired_msa_path"):
    if key == "paired_msa_path" and chain["modality"] != "protein":
      continue
    raw_path = reference.get(key)
    if not isinstance(raw_path, (str, Path)) or not str(raw_path):
      raise AF3InputError(f"Missing cached {key}")
    path = Path(raw_path)
    if not path.is_file():
      raise AF3InputError(f"Missing cached {key}: {path}")
    try:
      rows = _a3m_rows(path.read_text(), path=path)
    except (OSError, UnicodeError) as exc:
      raise AF3InputError(f"Cached A3M cannot be read: {path}") from exc
    # Native A3M first row must be the exact normalized FINAL native query.
    if not rows or rows[0] != chain["sequence"]:
      raise AF3InputError(f"Cached A3M query differs from native chain {chain['id']}")
    for index, row in enumerate(rows):
      if not re.fullmatch(r"[A-Za-z-]+", row):
        raise AF3InputError(f"Cached A3M row {index} contains invalid residues: {path}")
      columns = "".join(character for character in row if not character.islower())
      if len(columns) != len(chain["sequence"]):
        raise AF3InputError(
            f"Cached A3M row {index} has {len(columns)} query columns, "
            f"expected {len(chain['sequence'])}: {path}"
        )
  templates = reference.get("templates", ())
  if not isinstance(templates, (tuple, list)):
    raise AF3InputError("Cached templates must be a sequence")
  if chain["modality"] != "protein" and templates:
    raise AF3InputError("Native AF3 templates are supported only for protein chains")
  for template in templates:
    if not isinstance(template, Mapping):
      raise AF3InputError("Cached template must be a mapping")
    _validate_template_reference(template, len(chain["sequence"]))


def msa_queries(payload: Mapping[str, Any], design_chains: Sequence[str]) -> list[dict[str, str]]:
  """Native per-chain queries; designed chains remain query-only."""
  design = set(design_chains)
  return [{"native_id": str(chain["id"]), "source_id": str(chain["source_id"]),
           "modality": str(chain["modality"]), "sequence": str(chain["sequence"])}
          for chain in payload["chains"]
          if chain["modality"] in {"protein", "rna"}
          and chain.get("role") != "design" and chain.get("source_id") not in design]


def attach_msa_references(payload: Mapping[str, Any], design_chains: Sequence[str], references: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
  """Attach one immutable cache reference independently to each native chain."""
  result = dict(payload, chains=[dict(chain) for chain in payload["chains"]])
  queries = {query["native_id"]: query for query in msa_queries(payload, design_chains)}
  contexts = []
  for chain in result["chains"]:
    if str(chain["id"]) not in queries:
      continue
    matching = [ref for ref in references if (ref["modality"], ref["sequence"]) == (chain["modality"], chain["sequence"])]
    if len(matching) != 1:
      raise AF3InputError(f"Expected one ready MSA reference for chain {chain['id']}")
    reference = dict(matching[0])
    _validate_msa_reference(chain, reference)
    chain["msa_reference"] = reference
    contexts.append({"native_id": chain["id"], "source_id": chain["source_id"],
                     **{key: reference.get(key) for key in ("modality", "sequence", "entry_id", "profile_id", "profile_text", "cache_dir", "generation", "generation_id", "provenance")}})
  result["msa_context"] = {"chains": contexts, "resolve_msa_overlaps": bool(contexts),
                           "max_templates": 4 if any(c.get("msa_reference", {}).get("templates") for c in result["chains"]) else 0}
  return result


def _render_user_ccd(
    atoms: Any,
    components: Mapping[str, Sequence[Mapping[str, Any]]],
    source_to_native: Mapping[str, str],
    *,
    molecule_overrides: Mapping[str, Any] | None = None,
) -> str | None:
  """Render coordinate-free custom CCDs through the chemistry boundary.

  ``source_to_native`` remains part of the private compatibility signature;
  atom names and graph rows are the authoritative identity in the helper.
  """
  del source_to_native
  try:
    from .af3_chemistry import ChemistryError, render_user_ccd
    return render_user_ccd(atoms, components, molecule_overrides)
  except ChemistryError as exc:
    raise AF3InputError(str(exc)) from exc

def _user_ccd_component_ids(value: Any) -> set[str]:
  if not isinstance(value, str):
    return set()
  return {match.group(1).strip().upper() for match in _CCD_DATA_BLOCK.finditer(value)}


def _user_ccd_blocks(value: str) -> dict[str, str]:
  matches = list(_CCD_DATA_BLOCK.finditer(value))
  if not matches:
    raise AF3InputError("user CCD must contain at least one data_ component block")
  blocks: dict[str, str] = {}
  for index, match in enumerate(matches):
    component_id = match.group(1).strip().upper()
    if component_id in blocks:
      raise AF3InputError(f"user CCD contains duplicate component block {component_id!r}")
    end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
    blocks[component_id] = value[match.start():end].strip()
  return blocks


def _parse_user_ccd(value: str | None) -> Mapping[str, Any]:
  if value is None:
    return {}
  try:
    from alphafold3.cpp import cif_dict
    parsed = cif_dict.parse_multi_data_cif(value)
  except Exception as exc:
    raise AF3InputError(f"user CCD cannot be parsed: {type(exc).__name__}: {exc}") from exc
  if not isinstance(parsed, Mapping):
    raise AF3InputError("user CCD parser returned no component mapping")
  blocks = _user_ccd_blocks(value)
  normalized: dict[str, Any] = {}
  for raw_id, entry in parsed.items():
    component_id = str(raw_id).strip().upper()
    if component_id in normalized:
      raise AF3InputError(f"user CCD contains duplicate parsed component {component_id!r}")
    normalized[component_id] = entry
  if set(blocks) != set(normalized):
    raise AF3InputError(
        "user CCD data_ blocks do not match parsed component identities"
    )
  return normalized


def _merge_user_ccd(
    provided: str | None,
    generated: str | None,
) -> str | None:
  if provided is None:
    return generated
  if generated is None:
    return provided
  provided_blocks = _user_ccd_blocks(provided)
  generated_blocks = _user_ccd_blocks(generated)
  merged = [provided.rstrip()]
  for component_id, block in generated_blocks.items():
    previous = provided_blocks.get(component_id)
    if previous is not None:
      if previous != block:
        raise AF3InputError(
            f"user CCD component {component_id!r} conflicts with generated chemistry"
        )
      continue
    merged.append(block)
  return "\n\n".join(part for part in merged if part.strip()) + "\n"


def _ccd_entry_has_explicit_charges(entry: Mapping[str, Any]) -> bool:
  values = entry.get("_chem_comp_atom.charge", ())
  if values is None or isinstance(values, (str, bytes)):
    return bool(values)
  try:
    values = list(values)
  except TypeError:
    return True
  return any(
      str(value).strip().lower() not in {"", ".", "?", "none", "nan"}
      for value in values
  )


def build_af3_input(
    input_record: Any,
    variant: Any,
    *,
    seed: int = 101,
    sample: int = 0,
    protocol: str | None = None,
    known_ccd_ids: set[str] | None = None,
    ccd: Mapping[str, Any] | None = None,
    ccd_asset_root: str | Path | None = None,
) -> AF3Input:
  """Build a native AF3 request from a canonical design record.

  ``input_record`` may be a ``CompiledSample``, ``SearchCandidate`` or a
  mapping exposing the same ``atoms`` and ``design_chains`` attributes.
  Coordinates, templates and MSAs are never read.  This makes complex AF3
  scoring a sequence/chemistry request rather than a template leak.  ``ccd``
  is an optional caller-owned native CCD mapping; ``ccd_asset_root`` resolves
  the shared native pickle through the process-cached chemistry loader.
  """
  atoms = _record_atoms(input_record)
  if len(atoms) == 0:
    raise AF3InputError("AF3 input record has no atoms")
  design = _record_design_chains(input_record, atoms)
  sequences = _variant_sequences(variant)
  record_ligand_entries = _record_ligand_entries(input_record)

  def _source_entry(source_id: str) -> Mapping[str, Any]:
    value = _mapping(sequences.get(source_id))
    if value:
      return value
    return _mapping(record_ligand_entries.get(source_id))
  settings = _variant_settings(input_record, variant)
  record_map = _mapping(input_record)
  variant_map = _variant_mapping(variant)
  provided_user_ccd = (
      variant_map.get(
          "user_ccd",
          variant_map.get(
              "userCCD",
              variant_map.get("custom_ccd", variant_map.get("customCCD")),
          ),
      )
      or record_map.get("user_ccd", record_map.get("userCCD"))
  )
  if provided_user_ccd is not None and not isinstance(provided_user_ccd, str):
    raise AF3InputError("user CCD must be a CIF string")
  provided_user_components = _parse_user_ccd(provided_user_ccd)
  native_ccd = ccd
  if native_ccd is not None and not isinstance(native_ccd, Mapping):
    raise AF3InputError("AF3 native CCD inventory must be a mapping")
  if native_ccd is None and ccd_asset_root is not None:
    try:
      from .af3_chemistry import ChemistryError, load_native_ccd
      native_ccd = load_native_ccd(ccd_asset_root)
    except ChemistryError as exc:
      raise AF3InputError(str(exc)) from exc
  native_ccd_ids = (
      {str(value).strip().upper() for value in native_ccd.keys()}
      if native_ccd is not None
      else set()
  )
  available_ccd_ids = (
      set(_COMMON_CCD_IDS)
      | native_ccd_ids
      | {str(value).strip().upper() for value in (known_ccd_ids or ())}
      | set(provided_user_components)
      | set(_PROTEIN_MOD_PARENT)
      | set(_RNA_MOD_PARENT)
      | set(_DNA_MOD_PARENT)
  )
  if protocol is None:
    protocol = str(settings.get("protocol", settings.get("mode", "complex"))).lower()
  if protocol not in {"complex", "monomer"}:
    raise AF3InputError(f"AF3 protocol must be complex or monomer, got {protocol!r}")

  from ..utils.source_mapping import canonical_chain_ids

  atom_chain_ids = canonical_chain_ids(atoms)
  source_chain_ids: list[str] = []
  for value in atom_chain_ids.tolist():
    if value not in source_chain_ids:
      source_chain_ids.append(value)
  for value in sequences:
    if value not in source_chain_ids:
      source_chain_ids.append(value)
  # Monomer mode intentionally drops rigid context chains.  Ligand design
  # chains remain because they are part of the design request.
  if protocol == "monomer" and design:
    source_chain_ids = [value for value in source_chain_ids if value in design]

  smiles_by_source: dict[str, str] = {}
  for source_id in source_chain_ids:
    raw_entry = _source_entry(source_id)
    if raw_entry.get("smiles") is not None:
      smiles_by_source[source_id] = str(raw_entry["smiles"])
  smiles_bonded_sources: set[str] = set()
  if getattr(atoms, "bonds", None) is not None and smiles_by_source:
    ins_codes = np.asarray(
        getattr(atoms, "ins_code", np.full(len(atoms), "")), dtype=object
    )
    for row_i, row_j, _order in atoms.bonds.as_array().tolist():
      row_i, row_j = int(row_i), int(row_j)
      chain_i, chain_j = str(atom_chain_ids[row_i]), str(atom_chain_ids[row_j])
      if chain_i not in smiles_by_source and chain_j not in smiles_by_source:
        continue
      if chain_i != chain_j:
        if chain_i in smiles_by_source:
          smiles_bonded_sources.add(chain_i)
        if chain_j in smiles_by_source:
          smiles_bonded_sources.add(chain_j)
        continue
      # A cross-residue ligand link also needs named CCD endpoints.
      if (int(atoms.res_id[row_i]), str(ins_codes[row_i])) != (
          int(atoms.res_id[row_j]), str(ins_codes[row_j])
      ) and chain_i in smiles_by_source:
        smiles_bonded_sources.add(chain_i)

  fixed_modified = settings.get("fixed_modified_residue_names", {})
  if not isinstance(fixed_modified, Mapping):
    fixed_modified = {}
  if not fixed_modified and not isinstance(input_record, Mapping):
    candidate_fixed = getattr(input_record, "fixed_modified_residue_names", {})
    if isinstance(candidate_fixed, Mapping):
      fixed_modified = candidate_fixed

  from ..utils.source_mapping import plan_refold_positions, source_locator

  observed_sequences: dict[str, str] = {}
  polymer_names: dict[str, list[str]] = {}
  for source_id in source_chain_ids:
    rows = _chain_rows(atoms, source_id)
    modalities = {str(value) for value in np.asarray(atoms.mol_type)[rows]}
    if len(modalities) > 1:
      raise AF3InputError(f"chain {source_id!r} contains mixed modalities: {sorted(modalities)}")
    modality = next(iter(modalities), str(design.get(source_id, "protein")))
    if modality == "ligand":
      continue
    groups = _residue_groups(atoms, rows)
    names = []
    for group in groups:
      name = _name_for_group(atoms, group)
      residue_uid = int(getattr(atoms, "residue_uid", np.full(len(atoms), -1))[group[0]])
      names.append(str(fixed_modified.get(
          residue_uid, fixed_modified.get(str(residue_uid), name)
      )).strip().upper())
    tokens = _sequence_tokens(sequences.get(source_id), modality)
    if tokens and len(tokens) != len(groups):
      raise AF3InputError(
          f"chain {source_id!r} sequence has {len(tokens)} residues but "
          f"the source structure has {len(groups)} residues"
      )
    observed_sequences[source_id] = (
        "".join(token if len(token) == 1 else _one_letter(token, modality) for token in tokens)
        if tokens else "".join(_one_letter(name, modality) for name in names)
    ).upper()
    if not observed_sequences[source_id]:
      raise AF3InputError(f"polymer chain {source_id!r} has an empty sequence")
    if modality in {"rna", "dna"}:
      alphabet = "ACGUN" if modality == "rna" else "ACGTN"
      if any(letter not in alphabet for letter in observed_sequences[source_id]):
        raise AF3InputError(f"{modality.upper()} sequence must use {alphabet}; unknown nucleic residues use N")
    polymer_names[source_id] = names
  try:
    selected_rows = np.flatnonzero(np.isin(atom_chain_ids, source_chain_ids))
    planning_atoms = atoms if len(selected_rows) == len(atoms) else atoms[selected_rows]
    position_plan = plan_refold_positions(
        planning_atoms, design_chains=tuple(design), chain_sequences=observed_sequences
    )
    if planning_atoms is not atoms:
      # The worker's atom ledger always refers to the original source rows.
      for chain_plan in position_plan["chains"]:
        for group in chain_plan["observed"]:
          group["row_indices"] = [int(selected_rows[row]) for row in group["row_indices"]]
  except (ValueError, KeyError) as exc:
    raise AF3InputError(f"AF3 source positions cannot be planned: {exc}") from exc
  polymer_plans = {str(plan["logical_chain_id"]): plan for plan in position_plan["chains"]}

  # Allocate each original source's first native ID before allocating extra
  # fragments, so a fragment cannot steal an unchanged design/context ID.
  primary_native: dict[str, str] = {}
  source_to_native: dict[str, tuple[str, ...]] = {}
  native_to_source: dict[str, str] = {}
  used_native: set[str] = set()
  for ordinal, source_id in enumerate(source_chain_ids):
    primary_native[source_id] = _native_chain_id(source_id, used_native, ordinal)
  for source_id in source_chain_ids:
    entity_count = len(polymer_plans[source_id]["entities"]) if source_id in polymer_names else 1
    if entity_count == 0:
      raise AF3InputError(f"polymer chain {source_id!r} has no native entities")
    native_ids = [primary_native[source_id]]
    native_ids.extend(
        _native_chain_id(f"{source_id}__pdfrag{index + 1}", used_native, len(used_native))
        for index in range(1, entity_count)
    )
    source_to_native[source_id] = tuple(native_ids)
    native_to_source.update({native: source_id for native in native_ids})

  explicit_ccd_ids: set[str] = set()
  for source_id in source_chain_ids:
    raw_entry = _source_entry(source_id)
    declared_ccd = raw_entry.get("ccd_ids", raw_entry.get("ccdCodes", ()))
    if isinstance(declared_ccd, Sequence) and not isinstance(declared_ccd, (str, bytes)):
      explicit_ccd_ids.update(str(component).upper() for component in declared_ccd)
  custom_component_map, custom_components = _stable_custom_ids(
      atoms,
      source_chain_ids,
      available_ccd_ids | explicit_ccd_ids,
      native_ccd,
  )
  unbound_smiles_names: set[str] = set()
  retained_component_names: set[str] = set()
  for candidate_source in source_chain_ids:
    candidate_rows = _chain_rows(atoms, candidate_source)
    candidate_groups = _residue_groups(atoms, candidate_rows)
    candidate_names = {_name_for_group(atoms, group) for group in candidate_groups}
    if candidate_source in smiles_by_source and candidate_source not in smiles_bonded_sources:
      unbound_smiles_names.update(candidate_names)
    else:
      retained_component_names.update(candidate_names)
  for source_name in unbound_smiles_names - retained_component_names:
    component_id = custom_component_map.pop(source_name, None)
    if component_id is not None:
      custom_components.pop(component_id, None)
  chain_entries: list[dict[str, Any]] = []
  residue_native_position: dict[tuple[str, int, str], tuple[str, int]] = {}
  source_residue_map: dict[str, tuple[Mapping[str, Any] | None, ...]] = {}
  atom_ins_codes = np.asarray(
      getattr(atoms, "ins_code", np.full(len(atoms), "")), dtype=object
  )
  cyclic_candidate: set[str] = set()
  molecule_overrides: dict[str, Any] = {}
  trusted_occurrences: dict[str, bool] = {}
  materialized_smiles_sources: set[str] = set()
  reserved_component_ids = set(custom_components) | set(custom_component_map.values()) | available_ccd_ids

  def _allocate_custom_component_id() -> str:
    ordinal = 1
    while True:
      candidate = f"UCCD_{ordinal:03d}"
      ordinal += 1
      if candidate not in reserved_component_ids:
        reserved_component_ids.add(candidate)
        return candidate

  def _set_molecule_override(component_id: str, materialized: Any) -> None:
    previous = molecule_overrides.get(component_id)
    if previous is not None:
      if _named_molecule_signature(previous) != _named_molecule_signature(materialized):
        raise AF3InputError(
            f"custom CCD component {component_id!r} has inconsistent trusted definitions"
        )
      return
    molecule_overrides[component_id] = materialized

  def _validate_provided_component(component_id: str, rows: Sequence[int]) -> None:
    normalized_id = str(component_id).strip().upper()
    entry = provided_user_components.get(normalized_id)
    if entry is None:
      return
    try:
      from .af3_chemistry import (
          ChemistryError,
          ccd_has_explicit_stereo,
          native_component_matches,
          source_graph,
      )
      source = source_graph(atoms, rows)
      if (
          any(charge is None for charge in source.charges)
          and _ccd_entry_has_explicit_charges(entry)
      ):
        raise ChemistryError(
            f"user CCD component {normalized_id!r} asserts charges unavailable in "
            "the authoritative source graph"
        )
      if ccd_has_explicit_stereo(entry):
        raise ChemistryError(
            f"user CCD component {normalized_id!r} asserts stereo unavailable in "
            "the authoritative source graph"
        )
      if not native_component_matches(atoms, rows, entry):
        raise ChemistryError(
            f"user CCD component {normalized_id!r} does not match its source graph"
        )
    except ChemistryError as exc:
      raise AF3InputError(str(exc)) from exc

  for source_id in source_chain_ids:
    rows = _chain_rows(atoms, source_id)
    declared_modality = str(design.get(source_id, ""))
    if len(rows):
      modalities = {str(value) for value in np.asarray(atoms.mol_type)[rows]}
      if len(modalities) != 1:
        raise AF3InputError(
            f"chain {source_id!r} contains mixed modalities: {sorted(modalities)}"
        )
      inferred_modality = next(iter(modalities))
      if declared_modality in {"protein", "rna", "dna", "ligand"} and declared_modality != inferred_modality:
        raise AF3InputError(
            f"chain {source_id!r} declares {declared_modality!r} but atoms are {inferred_modality!r}"
        )
      modality = declared_modality if declared_modality in {"protein", "rna", "dna", "ligand"} else inferred_modality
    else:
      modality = declared_modality or "protein"
    native_id = source_to_native[source_id][0]
    groups = _residue_groups(atoms, rows)
    role = "design" if source_id in design else "context"
    if modality == "ligand":
      source_residue_map[native_id] = tuple({
          **source_locator(atoms, int(group[0])),
          **_source_occurrence_facts(atoms, int(group[0])),
          "source_chain_id": source_id,
          "source_res_id": int(np.asarray(atoms.res_id)[group[0]]),
          "source_ins_code": str(atom_ins_codes[group[0]]),
          "source_observed_index": index,
          "source_residue_uid": int(getattr(atoms, "residue_uid", np.full(len(atoms), -1))[group[0]]),
          "source_token_uid": int(getattr(atoms, "token_uid", np.full(len(atoms), -1))[group[0]]),
          "row_indices": [int(row) for row in group],
      } for index, group in enumerate(groups))
      for position, group in enumerate(groups, start=1):
        first_row = int(group[0])
        residue_key = (source_id, int(atoms.res_id[first_row]), str(atom_ins_codes[first_row]))
        if residue_key in residue_native_position:
          raise AF3InputError(f"chain {source_id!r} has ambiguous AF3 bond endpoint {residue_key[1:]}")
        residue_native_position[residue_key] = (native_id, position)
      raw_entry = _source_entry(source_id)
      declared_ccd = raw_entry.get("ccd_ids", raw_entry.get("ccdCodes", ()))
      if isinstance(declared_ccd, str):
        declared_ccd = (declared_ccd,)
      elif not isinstance(declared_ccd, Sequence):
        declared_ccd = ()
      declared_ccd = tuple(
          str(value).strip().upper() for value in declared_ccd if str(value).strip()
      )
      smiles = raw_entry.get("smiles")
      trusted_molecule = None
      if smiles is None:
        trusted_molecule = raw_entry.get("molecule")
        if trusted_molecule is None:
          trusted_molecule = _private_ligand_molecule(
              input_record, variant, atoms, source_id, rows
          )
      uses_trusted = smiles is not None or trusted_molecule is not None
      for group in groups:
        source_name = _name_for_group(atoms, group)
        previous_trusted = trusted_occurrences.get(source_name)
        if previous_trusted is not None and previous_trusted != uses_trusted:
          raise AF3InputError(
              f"source CCD component {source_name!r} mixes trusted and implicit occurrences"
          )
        trusted_occurrences[source_name] = uses_trusted
      if smiles is not None:
        if len(groups) != 1:
          raise AF3InputError(
              f"SMILES ligand chain {source_id!r} must map to exactly one source residue"
          )
        try:
          from .af3_chemistry import ChemistryError, materialize_smiles, source_graph
          source_graph_value = source_graph(
              atoms, groups[0], preserve_aromatic_orders=True
          )
          materialized, source_to_smiles = materialize_smiles(
              source_graph_value, str(smiles)
          )
        except ChemistryError as exc:
          raise AF3InputError(
              f"SMILES ligand chain {source_id!r} has no exact source-row mapping"
          ) from exc
        atom_uids = np.asarray(
            getattr(atoms, "atom_uid", np.arange(len(atoms))), dtype=np.int64
        )
        smiles_atom_map = tuple({
            "source_row": int(row),
            "source_atom_uid": int(atom_uids[row]),
            "source_atom_name": str(atoms.atom_name[row]),
            "compiler_atom_index": int(trusted_index),
            "compiler_atom_name": f"A{int(trusted_index) + 1}",
            "native_atom_name": materialized.atom_names[int(trusted_index)],
        } for row, trusted_index in zip(groups[0].tolist(), source_to_smiles, strict=True))
        # Use a named CCD for every explicit SMILES. Native SMILES atom names
        # differ from compiler A1/A2 names; materialization preserves charge,
        # stereochemistry and exact source identities for both bonded and free ligands.
        source_name = _name_for_group(atoms, groups[0])
        component_id = custom_component_map.get(source_name)
        if component_id is None:
          component_id = _allocate_custom_component_id()
        entries = custom_components.setdefault(component_id, [])
        if not any(np.array_equal(np.asarray(entry["rows"]), groups[0]) for entry in entries):
          entries.append({"source_name": source_name, "rows": groups[0].copy()})
        custom_component_map[source_name] = component_id
        _set_molecule_override(component_id, materialized)
        materialized_smiles_sources.add(source_id)
        chain_entries.append({
            "id": native_id,
            "source_id": source_id,
            "modality": "ligand",
            "role": role,
            "input_semantics": "ligand",
            "ccd_ids": (component_id,),
            "smiles_atom_map": smiles_atom_map,
        })
        continue

      if trusted_molecule is not None:
        if len(groups) != 1:
          raise AF3InputError(
              f"trusted molecule ligand chain {source_id!r} must map to exactly one source residue"
          )
        try:
          from .af3_chemistry import (
              ChemistryError,
              materialize_trusted_molecule,
              source_graph,
          )
          source_graph_value = source_graph(
              atoms, groups[0], preserve_aromatic_orders=True
          )
          materialized, source_to_molecule = materialize_trusted_molecule(
              source_graph_value, trusted_molecule
          )
        except ChemistryError as exc:
          raise AF3InputError(
              f"trusted molecule ligand chain {source_id!r} has no exact source-row mapping"
          ) from exc
        atom_uids = np.asarray(
            getattr(atoms, "atom_uid", np.arange(len(atoms))), dtype=np.int64
        )
        molecule_atom_map = tuple({
            "source_row": int(row),
            "source_atom_uid": int(atom_uids[row]),
            "source_atom_name": str(atoms.atom_name[row]),
            "compiler_atom_index": int(trusted_index),
            "compiler_atom_name": f"A{int(trusted_index) + 1}",
            "native_atom_name": materialized.atom_names[int(trusted_index)],
        } for row, trusted_index in zip(groups[0].tolist(), source_to_molecule, strict=True))
        source_name = _name_for_group(atoms, groups[0])
        component_id = custom_component_map.get(source_name)
        if component_id is None:
          component_id = _allocate_custom_component_id()
        entries = custom_components.setdefault(component_id, [])
        if not any(np.array_equal(np.asarray(entry["rows"]), groups[0]) for entry in entries):
          entries.append({"source_name": source_name, "rows": groups[0].copy()})
        custom_component_map[source_name] = component_id
        _set_molecule_override(component_id, materialized)
        chain_entries.append({
            "id": native_id,
            "source_id": source_id,
            "modality": "ligand",
            "role": role,
            "input_semantics": "ligand",
            "ccd_ids": (component_id,),
            "molecule_atom_map": molecule_atom_map,
        })
        continue

      if declared_ccd:
        ccd_ids = list(declared_ccd)
        if len(ccd_ids) == len(groups):
          declared_pairs = tuple(zip(ccd_ids, groups, strict=True))
        elif len(ccd_ids) == 1:
          ccd_ids *= len(groups)
          declared_pairs = tuple(zip(ccd_ids, groups, strict=True))
        else:
          raise AF3InputError(
              f"ligand chain {source_id!r} has {len(ccd_ids)} CCD components "
              f"for {len(groups)} source residues"
          )
        for component_id, group in declared_pairs:
          _validate_provided_component(component_id, group)
          if native_ccd is not None and component_id in native_ccd:
            try:
              from .af3_chemistry import native_component_matches
              if not native_component_matches(atoms, group, native_ccd[component_id]):
                raise AF3InputError(
                    f"declared CCD component {component_id!r} does not match "
                    f"source residue on chain {source_id!r}"
                )
            except AF3InputError:
              raise
            except Exception as exc:
              raise AF3InputError(
                  f"declared CCD component {component_id!r} cannot be validated"
              ) from exc
      else:
        ccd_ids = [
            custom_component_map.get(_name_for_group(atoms, group), _name_for_group(atoms, group))
            for group in groups
        ]
        for component_id, group in zip(ccd_ids, groups, strict=True):
          _validate_provided_component(component_id, group)
      if not ccd_ids:
        ccd_ids = [
            str(value)
            for value in raw_entry.get("ccd_ids", raw_entry.get("ccdCodes", ()))
        ]
      if not ccd_ids:
        raise AF3InputError(f"ligand chain {source_id!r} has no CCD component")
      chain_entries.append({
          "id": native_id,
          "source_id": source_id,
          "modality": "ligand",
          "role": role,
          "input_semantics": "ligand",
          "ccd_ids": tuple(ccd_ids),
      })
      continue

    plan = polymer_plans[source_id]
    observed = plan["observed"]
    if len(observed) != len(groups):
      raise AF3InputError(f"shared source residue groups differ on chain {source_id!r}")
    names = polymer_names[source_id]
    complete_sequence = plan["input_semantics"] == "complete_sequence"
    for native_id, entity in zip(source_to_native[source_id], plan["entities"], strict=True):
      sequence = str(entity["sequence"])
      observed_positions = entity["observed_positions"]
      components = entity.get("components", ())
      if not sequence or not re.fullmatch(r"[A-Z]+", sequence):
        raise AF3InputError(f"invalid native polymer sequence on chain {source_id!r}")
      if len(observed_positions) != len(sequence) or (components and len(components) != len(sequence)):
        raise AF3InputError(f"native residue position/component map length differs on chain {source_id!r}")
      modifications: list[tuple[str, int]] = []
      native_residues: list[Mapping[str, Any] | None] = []
      for position, observed_index in enumerate(observed_positions, start=1):
        group = None
        if observed_index is None:
          native_residues.append(None)
          if not complete_sequence or not components:
            raise AF3InputError("unmapped AF3 positions require a confirmed complete polymer sequence")
          name = str(components[position - 1]).upper()
        else:
          if type(observed_index) is not int or not 0 <= observed_index < len(observed):
            raise AF3InputError(f"invalid observed/native residue position on chain {source_id!r}")
          source_group = observed[observed_index]
          group = np.asarray(source_group["row_indices"], dtype=np.int64)
          if not np.array_equal(group, groups[observed_index]):
            raise AF3InputError(f"shared source residue row order differs on chain {source_id!r}")
          locator = dict(source_group["locator"])
          native_residues.append({
              **locator,
              **_source_occurrence_facts(atoms, int(group[0])),
              "source_chain_id": source_id,
              "source_res_id": int(locator["res_id"]),
              "source_ins_code": str(locator["ins_code"]),
              "source_observed_index": observed_index,
              "source_residue_uid": int(source_group["residue_uid"]),
              "source_token_uid": int(source_group["token_uid"]),
              "row_indices": [int(row) for row in group],
          })
          residue_key = (source_id, int(locator["res_id"]), str(locator["ins_code"]))
          if residue_key in residue_native_position:
            raise AF3InputError(f"chain {source_id!r} has an ambiguous native residue occurrence")
          residue_native_position[residue_key] = (native_id, position)
          name = names[observed_index]
        standard = (
            (modality == "protein" and name in _PROTEIN_3_TO_1)
            or (modality == "rna" and name in _RNA_CODES)
            or (modality == "dna" and name in _DNA_CODES)
        )
        if not standard and name not in {"UNK", "N", "DN"}:
          if name not in available_ccd_ids:
            raise AF3InputError(
                f"polymer modification {name!r} on chain {source_id!r} "
                "has no native or provided CCD component"
            )
          if group is not None:
            _validate_provided_component(name, group)
          modifications.append((name, position))
      sequence_letters = list(sequence)
      for name, position in modifications:
        sequence_letters[position - 1] = _native_modification_letter(name, modality)
      sequence = "".join(sequence_letters)
      source_residue_map[native_id] = tuple(native_residues)
      chain_entries.append({
          "id": native_id,
          "source_id": source_id,
          "modality": modality,
          "role": role,
          "input_semantics": str(plan["input_semantics"]),
          "sequence": sequence,
          "modifications": tuple(modifications),
          "description": None,
      })

  # Identify cyclic design chains from explicit settings or the canonical
  # head-tail C--N topology fact.  The polymer-polymer bond itself is omitted
  # from AF3's bond list below, but its chain-local metadata is retained.
  explicit_cycle = settings.get("cyclic", settings.get("is_cyclic", settings.get("cycle", False)))
  explicit_offsets = settings.get("cyclic_offset")
  if isinstance(explicit_cycle, Mapping):
    explicit_cycle_ids = {str(key) for key, value in explicit_cycle.items() if bool(value)}
  elif isinstance(explicit_offsets, Mapping):
    explicit_cycle_ids = {str(key) for key, value in explicit_offsets.items() if value is True}
  elif explicit_offsets is True:
    # A scalar cyclic_offset is the compact single-chain form used by the
    # design planner.  It is metadata, not a residue-index override: the
    # native pipeline still receives contiguous chain-local residue indices.
    polymer_design = {
        chain_id
        for chain_id, modality in design.items()
        if modality in {"protein", "rna", "dna"}
    }
    explicit_cycle_ids = set(polymer_design or source_chain_ids[:1])
  elif explicit_cycle:
    polymer_design = {
        chain_id
        for chain_id, modality in design.items()
        if modality in {"protein", "rna", "dna"}
    }
    explicit_cycle_ids = set(polymer_design or source_chain_ids[:1])
  else:
    explicit_cycle_ids = set()
  uid_to_row = {
      int(value): row for row, value in enumerate(getattr(atoms, "atom_uid", np.arange(len(atoms))))
  }
  # ValidationInput is a slotted dataclass, so it is not exposed by vars().
  # Read the canonical offset intent directly rather than requiring a settings alias.
  declared_offsets = (input_record.get("cyclic_offset_chains", ()) if isinstance(input_record, Mapping)
                      else getattr(input_record, "cyclic_offset_chains", ()))
  if isinstance(declared_offsets, (str, bytes)):
    raise AF3InputError("cyclic_offset_chains must be a sequence of chain IDs")
  explicit_cycle_ids.update(str(chain) for chain in (declared_offsets or ()))
  cyclic_source_ids = set(explicit_cycle_ids)
  cyclic_groups = {
      source: [np.asarray(group["row_indices"], dtype=np.int64) for group in plan["observed"]]
      for source, plan in polymer_plans.items()
  }
  if getattr(atoms, "bonds", None) is not None:
    for row_i, row_j, order in atoms.bonds.as_array().tolist():
      if int(order) not in {1, 0}:
        continue
      row_i, row_j = int(row_i), int(row_j)
      if str(atoms.mol_type[row_i]) != "protein" or str(atoms.mol_type[row_j]) != "protein":
        continue
      if str(atom_chain_ids[row_i]) != str(atom_chain_ids[row_j]):
        continue
      names_pair = {str(atoms.atom_name[row_i]).upper(), str(atoms.atom_name[row_j]).upper()}
      if names_pair != {"N", "C"}:
        continue
      chain = str(atom_chain_ids[row_i])
      chain_groups = cyclic_groups.get(chain, ())
      carbon_row = row_i if str(atoms.atom_name[row_i]).upper() == "C" else row_j
      nitrogen_row = row_i if str(atoms.atom_name[row_i]).upper() == "N" else row_j
      if (len(chain_groups) > 1 and carbon_row in chain_groups[-1]
          and nitrogen_row in chain_groups[0]):
        cyclic_source_ids.add(chain)
  cyclic_source_ids.intersection_update(source_chain_ids)
  for chain in cyclic_source_ids:
    if set(np.asarray(atoms.mol_type)[_chain_rows(atoms, chain)].astype(str)) != {"protein"}:
      raise AF3InputError("AF3 cyclic offset requires a protein chain")
    if len(source_to_native[chain]) != 1:
      raise AF3InputError("AF3 cyclic offset cannot span separate fragment entities")
  cyclic_native = tuple(source_to_native[source_id][0] for source_id in source_chain_ids if source_id in cyclic_source_ids)
  chain_by_id = {entry["id"]: entry for entry in chain_entries}
  cyclic_lengths = {
      source_to_native[source_id][0]: len(chain_by_id[source_to_native[source_id][0]]["sequence"])
      for source_id in cyclic_source_ids
  }

  # Convert concrete bonds while intentionally omitting every polymer-polymer
  # edge.  AF3 infers the polymer backbone from sequence and consumes only
  # ligand/cross-component covalent links here.
  bond_keys: dict[tuple[tuple[str, int, str], tuple[str, int, str]], None] = {}
  smiles_chains = set(smiles_by_source) - materialized_smiles_sources
  if getattr(atoms, "bonds", None) is not None:
    allowed = set(source_chain_ids)
    for row_i, row_j, _order in atoms.bonds.as_array().tolist():
      row_i, row_j = int(row_i), int(row_j)
      chain_i, chain_j = str(atom_chain_ids[row_i]), str(atom_chain_ids[row_j])
      if chain_i not in allowed or chain_j not in allowed:
        continue
      if chain_i in smiles_chains or chain_j in smiles_chains:
        # An unbonded native SMILES ligand has no external endpoint to encode.
        # Any endpoint-bearing SMILES was materialized into a named UCCD above.
        continue
      if chain_i == chain_j and (
          int(atoms.res_id[row_i]), str(atom_ins_codes[row_i])
      ) == (int(atoms.res_id[row_j]), str(atom_ins_codes[row_j])):
        # Internal ligand chemistry is already carried by its CCD graph.
        continue
      if str(atoms.mol_type[row_i]) != "ligand" and str(atoms.mol_type[row_j]) != "ligand":
        continue
      pos_i = residue_native_position.get((
          chain_i, int(atoms.res_id[row_i]), str(atom_ins_codes[row_i])
      ))
      pos_j = residue_native_position.get((
          chain_j, int(atoms.res_id[row_j]), str(atom_ins_codes[row_j])
      ))
      if pos_i is None or pos_j is None:
        raise AF3InputError("AF3 bond endpoint has no source residue mapping")
      if chain_i == chain_j and pos_i == pos_j:
        # Internal component edges already belong to the exact CCD graph.
        continue
      left = (pos_i[0], int(pos_i[1]), str(atoms.atom_name[row_i]))
      right = (pos_j[0], int(pos_j[1]), str(atoms.atom_name[row_j]))
      key = tuple(sorted((left, right)))
      bond_keys[key] = None

  generated_user_ccd = _render_user_ccd(
      atoms, custom_components, custom_component_map, molecule_overrides=molecule_overrides
  )
  user_ccd = _merge_user_ccd(provided_user_ccd, generated_user_ccd)

  sample_seed = int(seed) + int(sample)
  if sample_seed < 0 or sample_seed > 2**32 - 1:
    raise AF3InputError("AF3 sample seed must fit an unsigned 32-bit integer")
  name = str(
      variant_map.get("name")
      or variant_map.get("sample_id")
      or record_map.get("name")
      or record_map.get("sample_id")
      or "af3"
  )
  if not name.strip():
    name = "af3"
  seq_index = variant_map.get("seq_index")
  if seq_index is not None:
    if type(seq_index) is not int or seq_index < 0:
      raise AF3InputError("AF3 sequence index must be a non-negative integer")
    # All sequence variants share the validation sample output directory.
    # Include the slot in native names so different sequences never alias CIFs.
    name = f"{name}_seq{seq_index}"
  if Path(name).name != name or name in {".", ".."}:
    raise AF3InputError("AF3 input name must be a safe single path component")
  component_rows = {
      component_id: tuple(int(row) for entry in entries for row in np.asarray(entry["rows"]).tolist())
      for component_id, entries in custom_components.items()
  }
  source_polymer_sequences: dict[str, tuple[str, ...]] = {}
  for source_id, plan in polymer_plans.items():
    if plan["role"] != "context":
      continue
    full_sequence = next((group["source_polymer_sequence"] for group in plan["observed"]
                          if group.get("source_polymer_sequence") is not None and len(group["source_polymer_sequence"])), None)
    if full_sequence is not None:
      source_polymer_sequences[source_id] = tuple(full_sequence)
  source_occurrence_by_row = {
      row: (source, item["source_observed_index"])
      for native, entries in source_residue_map.items()
      for source in (native_to_source[native],)
      for item in entries if item is not None
      for row in item["row_indices"]
  }
  source_inter_residue_bonds = ()
  if getattr(atoms, "bonds", None) is not None:
    source_inter_residue_bonds = tuple(
        (int(left), int(right), int(order))
        for left, right, order in atoms.bonds.as_array().tolist()
        if int(left) in source_occurrence_by_row and int(right) in source_occurrence_by_row
        and source_occurrence_by_row[int(left)] != source_occurrence_by_row[int(right)]
    )
  result = AF3Input(
      name=name,
      chains=tuple(chain_entries),
      rng_seeds=(sample_seed,),
      bonded_atom_pairs=tuple((key[0], key[1]) for key in sorted(bond_keys)),
      user_ccd=user_ccd,
      source_to_native_chain=source_to_native,
      native_to_source_chain=native_to_source,
      cyclic_chains=cyclic_native,
      cyclic_lengths=cyclic_lengths,
      source_residue_map=source_residue_map,
      source_atom_ledger=tuple(_source_ledger(atoms)),
      source_inter_residue_bonds=source_inter_residue_bonds,
      source_polymer_sequences=source_polymer_sequences,
      native_component_map=custom_component_map,
      component_rows=component_rows,
      protocol=protocol,
  )
  from .af3_mapping import validate_source_mapping

  try:
    validate_source_mapping(result.payload())
  except ValueError as exc:
    raise AF3InputError(str(exc)) from exc
  return result


__all__ = [
    "AF3Input",
    "AF3InputError",
    "build_af3_input",
]
