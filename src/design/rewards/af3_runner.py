"""Owner-local native AF3 reward execution.

The runner owns only candidate planning, one persistent AF3 backend per reward
batch, and the AF3 confidence reduction. Native SDK loading and inference live
behind :mod:af3_backend; no ESMFold2 model is instantiated here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..core.records import SearchCandidate
from ..utils.reader import read_structure
from .af3_backend import AF3Backend, AF3BackendError
from .af3_input import AF3InputError
from .esmfold2 import ESMFold2Confidence
from .sequence import plan_reward_sequence_source, normalize_reward_sequence_config
from .esmfold2_input import candidate_chain_modalities
from .esmfold2_runner import (
    FoldRewardFailure,
    FoldRewardResult,
    _reported_polymer_sequences,
    motif_rmsd_angstrom,
)
from .mpnn import (
    MPNNSequenceError,
    MPNNSystemError,
    build_antibody_mpnn_plan,
    generate_mpnn_sequences,
)

def plan_sequence_source(*args: Any, **kwargs: Any) -> Any:
    """Compatibility seam forwarding runner planning to the reward helper."""
    return plan_reward_sequence_source(*args, **kwargs)


def _write_antibody_stage(
    candidate: SearchCandidate,
    plan: Any,
    variant: Mapping[str, Any],
    reward_config: Mapping[str, Any],
    candidate_root: Path,
    stage_output_root: Path | None,
) -> None:
  """Keep the antibody finalization carrier private to the owner work tree."""

  import os

  stage = {
      "candidate": candidate,
      "plan": plan,
      "variant": dict(variant),
      "reward_config": dict(reward_config),
  }
  import torch as _torch

  _torch.save(stage, candidate_root / "finalized_antibody.pt")
  if stage_output_root is None:
    return
  destination = (
      stage_output_root
      / candidate.base_request.task.name
      / "antibody_stages"
      / f"{candidate.base_request.sample_id}_candidate_{candidate.candidate_ordinal}.pt"
  )
  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_suffix(".tmp")
  _torch.save(stage, temporary)
  os.replace(temporary, destination)


def _design_chain_ids(candidate: SearchCandidate) -> tuple[str, ...]:
  """Return candidate-authoritative design chains in first-seen order."""

  chain_values = np.asarray(candidate.atoms.chain_id).astype(str)
  design_mask = np.asarray(candidate.atoms.is_design, dtype=bool)
  result: list[str] = []
  for chain in chain_values:
    if chain in result:
      continue
    rows = np.flatnonzero(chain_values == chain)
    if bool(design_mask[rows].any()):
      result.append(str(chain))
  if result:
    return tuple(result)
  for item in candidate.base_request.task.design_chains:
    chain = str(getattr(item, "chain_id", getattr(item, "id", "")))
    if chain and chain not in result:
      result.append(chain)
  return tuple(result)


def _all_chain_ids(candidate: SearchCandidate) -> tuple[str, ...]:
  result: list[str] = []
  for value in np.asarray(candidate.atoms.chain_id).astype(str):
    if value not in result:
      result.append(str(value))
  return tuple(result)


def _variant_from_plan(
    candidate: SearchCandidate,
    plan: Any,
    *,
    name: str,
) -> dict[str, Any]:
  """Adapt the finalized slots plan to the SDK-independent AF3 builder ABI."""

  design_chain_ids = _design_chain_ids(candidate)
  design = set(design_chain_ids)
  modalities = candidate_chain_modalities(candidate)
  sequences = [
      {
          "chain_id": str(chain),
          "role": "design" if chain in design else "context",
          "modality": str(modalities.get(chain, "protein")),
          "sequence": str(sequence),
      }
      for chain, sequence in plan.chain_sequences.items()
  ]
  task = candidate.base_request.task
  authored = task.condition.get("ligand", {}) if isinstance(getattr(task, "condition", None), Mapping) else {}
  identities = getattr(candidate, "validated_ligand_identities", {})
  private_molecules = getattr(candidate, "private_ligand_molecules", {})
  atoms = candidate.atoms
  labels = np.asarray(atoms.chain_id).astype(str)
  for chain in _all_chain_ids(candidate):
    if modalities.get(chain) != "ligand":
      continue
    entry = {"chain_id": chain, "role": "design" if chain in design else "context", "modality": "ligand"}
    declared = authored.get(chain, {}) if isinstance(authored, Mapping) else {}
    if isinstance(declared, Mapping) and declared.get("mode") == "smiles":
      entry["smiles"] = declared["smiles"]
    else:
      rows = np.flatnonzero(labels == chain)
      component_uids = list(dict.fromkeys(np.asarray(atoms.component_uid)[rows].tolist()))
      from ..utils.ligand_identity import ligand_identity_matches_component
      ccd_ids = []
      for uid in component_uids:
        identity = identities.get(int(uid))
        if identity is None or not ligand_identity_matches_component(
            atoms, rows[np.asarray(atoms.component_uid)[rows] == uid].tolist(), identity
        ):
          ccd_ids = []
          break
        ccd_ids.append(identity.component_id)
      if ccd_ids:
        entry["ccd_ids"] = ccd_ids
      else:
        molecule_uids = list(dict.fromkeys(np.asarray(atoms.molecule_uid)[rows].tolist()))
        molecule = private_molecules.get(int(molecule_uids[0])) if len(molecule_uids) == 1 else None
        if molecule is not None:
          entry["molecule"] = molecule
    # Coordinate-free named CCD is still available for source components when
    # no authored SMILES/private molecule carrier exists.
    sequences = [item for item in sequences if item["chain_id"] != chain]
    sequences.append(entry)
  settings: dict[str, Any] = {}
  if bool(getattr(task, "cyclic_offset", False)):
    cyclic_chains = tuple(getattr(task, "cyclic_offset_chains", ()))
    if not cyclic_chains:
      cyclic_chains = design_chain_ids
    settings["cyclic_offset"] = {str(chain): True for chain in cyclic_chains}
  return {
      "name": str(name),
      "sample_id": str(name),
      "sequences": sequences,
      "settings": settings,
      "fixed_modified_residue_names": dict(
          getattr(candidate, "fixed_modified_residue_names", {})
      ),
  }


def _aliases(
    raw: Mapping[str, Any],
    source_chain: str,
) -> tuple[str, ...]:
  """Return source/native aliases for confidence maps from mixed SDK versions."""

  result: list[str] = [str(source_chain)]
  for mapping_name in ("source_to_native_chain", "native_to_source_chain"):
    mapping = _field(raw, mapping_name)
    if not isinstance(mapping, Mapping):
      continue
    if mapping_name == "source_to_native_chain":
      value = mapping.get(str(source_chain))
      if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result.extend(str(item) for item in value)
      elif value is not None:
        result.append(str(value))
    else:
      for native, source in mapping.items():
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
          matches = str(source_chain) in {str(item) for item in source}
        else:
          matches = str(source) == str(source_chain)
        if matches:
          result.append(str(native))
  # Newer worker records carry aligned source/native identifiers even when
  # aliases are not serialized as dictionaries.
  source_ids = _field(raw, "source_chain_ids", "source_unique_chain_ids")
  native_ids = _field(
      raw,
      "unique_chain_ids",
      "native_chain_ids",
      "native_unique_chain_ids",
  )
  if (
      isinstance(source_ids, Sequence)
      and not isinstance(source_ids, (str, bytes, bytearray))
      and isinstance(native_ids, Sequence)
      and not isinstance(native_ids, (str, bytes, bytearray))
  ):
    for source, native in zip(source_ids, native_ids, strict=False):
      if str(source) == str(source_chain):
        result.append(str(native))
  # Accept the explicit alias maps used by SDK adapters. Values may be a
  # single native identifier or a list of equivalent identifiers.
  for alias_name in ("source_chain_aliases", "chain_aliases", "source_aliases"):
    aliases = _field(raw, alias_name)
    if not isinstance(aliases, Mapping):
      continue
    value = aliases.get(str(source_chain))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
      result.extend(str(item) for item in value)
    elif value is not None:
      result.append(str(value))
    for alias, canonical in aliases.items():
      if isinstance(canonical, Sequence) and not isinstance(canonical, (str, bytes, bytearray)):
        matches = str(source_chain) in {str(item) for item in canonical}
      else:
        matches = str(canonical) == str(source_chain)
      if matches:
        result.append(str(alias))
  return tuple(dict.fromkeys(result))


def _mapped_native_chain_ids(
    raw: Mapping[str, Any],
    source_chain: str,
) -> tuple[str, ...] | None:
  """Resolve native confidence IDs without a colliding source-name alias.

  None means a legacy record has no authoritative mapping. An empty tuple
  means a mapping exists but does not contain the requested source chain.
  Source-keyed pLDDT/atom-count dictionaries continue to use _aliases.
  """

  result: list[str] = []
  mapped = False
  for mapping_name in ("source_to_native_chain", "native_to_source_chain"):
    mapping = _field(raw, mapping_name)
    if not isinstance(mapping, Mapping) or not mapping:
      continue
    mapped = True
    if mapping_name == "source_to_native_chain":
      value = mapping.get(str(source_chain))
      if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result.extend(str(item) for item in value)
      elif value is not None:
        result.append(str(value))
    else:
      for native, source in mapping.items():
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
          matches = str(source_chain) in {str(item) for item in source}
        else:
          matches = str(source) == str(source_chain)
        if matches:
          result.append(str(native))
  source_ids = _field(raw, "source_chain_ids", "source_unique_chain_ids")
  native_ids = _field(
      raw, "unique_chain_ids", "native_chain_ids", "native_unique_chain_ids", "chain_ids"
  )
  if (
      isinstance(source_ids, Sequence)
      and not isinstance(source_ids, (str, bytes, bytearray))
      and len(source_ids) > 0
      and isinstance(native_ids, Sequence)
      and not isinstance(native_ids, (str, bytes, bytearray))
      and len(native_ids) > 0
  ):
    mapped = True
    for source, native in zip(source_ids, native_ids, strict=False):
      if str(source) == str(source_chain):
        result.append(str(native))
  return tuple(dict.fromkeys(result)) if mapped else None


def _finite_unit(value: Any, label: str) -> float:
  try:
    result = float(np.asarray(value).reshape(-1)[0])
  except (TypeError, ValueError, IndexError) as exc:
    raise ValueError(f"AF3 {label} is not numeric") from exc
  if not np.isfinite(result) or not 0.0 <= result <= 1.0:
    raise ValueError(f"AF3 {label} must be finite in [0,1]")
  return result


def _finite_nonnegative(value: Any, label: str) -> float:
  try:
    result = float(np.asarray(value).reshape(-1)[0])
  except (TypeError, ValueError, IndexError) as exc:
    raise ValueError(f"AF3 {label} is not numeric") from exc
  if not np.isfinite(result) or result < 0.0:
    raise ValueError(f"AF3 {label} must be finite and non-negative")
  return result



def _field(
    raw: Mapping[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
  """Read a native confidence field from either schema location.

  The worker currently emits confidence fields at prediction-record level,
  while the SDK adapter may group them under confidence. Keeping this
  lookup local lets the reduction enforce the same validation for both
  layouts without copying or mutating native records.
  """

  containers: list[Mapping[str, Any]] = [raw]
  nested = raw.get("confidence")
  if isinstance(nested, Mapping):
    containers.append(nested)
  for container in containers:
    for name in names:
      if name in container:
        return container[name]
  return default


def _map_value(mapping: Mapping[Any, Any], aliases: Sequence[str]) -> Any:
  for alias in aliases:
    if alias in mapping:
      return mapping[alias]
    for key, value in mapping.items():
      if str(key) == alias:
        return value
  return None


def _chain_metric(
    raw: Mapping[str, Any],
    field: str,
    chain: str,
) -> float | None:
  value = _field(raw, field)
  mapped = _mapped_native_chain_ids(raw, chain)
  aliases = _aliases(raw, chain) if mapped is None else mapped
  order = _native_chain_order(raw)
  native_aliases = (
      [native for native in order if native in aliases]
      if mapped is None else list(mapped)
  )
  if isinstance(value, Mapping):
    by_chain = value
  elif isinstance(value, (Sequence, np.ndarray)) and not isinstance(value, (str, bytes, bytearray)):
    by_chain = dict(zip(order, value, strict=False)) if order else {}
  else:
    by_chain = {}
  if len(native_aliases) > 1:
    values, weights = [], []
    token_weights = _native_token_weights(raw)
    for native in native_aliases:
      item = _map_value(by_chain, (native,))
      if item is None:
        return None
      weight = float(token_weights.get(native, 1))
      if weight > 0:
        values.append(_finite_unit(item, f"{field}[{native}]"))
        weights.append(weight)
    return float(np.average(values, weights=weights)) if values else None
  if isinstance(value, Mapping):
    value = _map_value(value, aliases)
  elif (
      isinstance(value, Sequence)
      or isinstance(value, np.ndarray)
  ) and not isinstance(value, (str, bytes, bytearray)):
    chain_order = _native_chain_order(raw)
    if chain_order:
      by_chain = {
          str(key): item for key, item in zip(chain_order, value, strict=False)
      }
      value = _map_value(by_chain, aliases)
    else:
      value = None
  if value is None:
    return None
  return _finite_unit(value, f"{field}[{chain}]")


def _native_token_weights(raw: Mapping[str, Any]) -> Mapping[str, Any]:
  weights = _field(raw, "native_per_chain_source_token_count", "native_per_chain_token_count")
  return weights if isinstance(weights, Mapping) else {}


def _pair_value(
    pair: Any,
    left_aliases: Sequence[str],
    right_aliases: Sequence[str],
    chain_order: Sequence[str] = (),
    token_weights: Mapping[str, Any] | None = None,
    *,
    strict: bool = False,
) -> float | None:
  # Each logical source chain can own several native fragment entities. Use
  # every relevant pair, weighted by its observed token counts; selecting the
  # first alias silently discarded the remaining receptor fragments.
  token_weights = token_weights or {}
  indexes = {str(chain): index for index, chain in enumerate(chain_order)}
  if strict:
    lefts = [chain for chain in dict.fromkeys(left_aliases) if float(token_weights.get(chain, 1)) > 0]
    rights = [chain for chain in dict.fromkeys(right_aliases) if float(token_weights.get(chain, 1)) > 0]
    if not lefts or not rights or any(chain not in indexes for chain in (*lefts, *rights)):
      return None
  else:
    lefts = [str(chain) for chain in chain_order if str(chain) in left_aliases] or list(dict.fromkeys(left_aliases))
    rights = [str(chain) for chain in chain_order if str(chain) in right_aliases] or list(dict.fromkeys(right_aliases))
  values, weights = [], []
  matrix = None
  if not isinstance(pair, Mapping):
    try:
      matrix = np.asarray(pair, dtype=np.float64)
    except (TypeError, ValueError):
      return None
  for left in lefts:
    for right in rights:
      value = None
      if isinstance(pair, Mapping):
        row = _map_value(pair, (left,))
        if isinstance(row, Mapping):
          value = _map_value(row, (right,))
        if value is None:
          for key in (f"{left}:{right}", f"{left},{right}", f"{left}_{right}", (left, right)):
            if key in pair:
              value = pair[key]
              break
      elif matrix is not None and left in indexes and right in indexes:
        try:
          value = matrix[indexes[left], indexes[right]]
          # Preserve a missing nullable entry before NumPy converts None to NaN.
          if strict and pair[indexes[left]][indexes[right]] is None:
            return None
        except IndexError:
          return None
      if strict and value is None:
        return None
      if value is not None:
        weight = float(token_weights.get(left, 1)) * float(token_weights.get(right, 1))
        if weight > 0:
          values.append(_finite_unit(value, "chain-pair iPTM"))
          weights.append(weight)
  if not values:
    return None
  return float(np.average(values, weights=weights))


def _aligned_chain_value(
    raw: Mapping[str, Any],
    value: Any,
    aliases: Sequence[str],
) -> Any:
  if isinstance(value, Mapping):
    return _map_value(value, aliases)
  if (
      isinstance(value, Sequence)
      or isinstance(value, np.ndarray)
  ) and not isinstance(value, (str, bytes, bytearray)):
    order = _native_chain_order(raw)
    if order:
      by_chain = {
          str(key): item for key, item in zip(order, value, strict=False)
      }
      return _map_value(by_chain, aliases)
  return None


def _chain_plddt(
    raw: Mapping[str, Any],
    chain: str,
) -> tuple[float, int] | None:
  values = _field(raw, "per_chain_plddt", "chain_plddt")
  if values is None:
    return None
  aliases = _aliases(raw, chain)
  value = _aligned_chain_value(raw, values, aliases)
  if value is None:
    return None
  atom_counts = _field(
      raw,
      "per_chain_atom_counts",
      "per_chain_atom_count",
      "chain_atom_counts",
      "chain_atom_count",
      "atom_counts_by_chain",
  )
  count = _aligned_chain_value(raw, atom_counts, aliases)
  if count is None and atom_counts is not None and len(_native_chain_order(raw)) <= 1:
    # A single-chain native adapter may serialize its actual atom count as a
    # scalar.  Never use token counts as a proxy for multi-chain weighting.
    count = atom_counts
  if count is None:
    raise ValueError(
        f"AF3 pLDDT for chain {chain!r} lacks an actual atom count"
    )
  try:
    score = float(np.asarray(value).reshape(-1)[0])
    count = int(np.asarray(count).reshape(-1)[0])
  except (TypeError, ValueError, IndexError) as exc:
    raise ValueError(f"AF3 pLDDT for chain {chain!r} is malformed") from exc
  if 0.0 <= score <= 1.0:
    score *= 100.0
  if not np.isfinite(score) or not 0.0 <= score <= 100.0 or count < 1:
    raise ValueError(f"AF3 pLDDT for chain {chain!r} is invalid")
  return score, count


def _design_plddt(
    raw: Mapping[str, Any],
    design_chains: Sequence[str],
) -> float:
  per_chain_values = _field(raw, "per_chain_plddt", "chain_plddt")
  if per_chain_values is not None:
    values: list[float] = []
    counts: list[int] = []
    for chain in design_chains:
      item = _chain_plddt(raw, chain)
      if item is None:
        raise ValueError(
            f"AF3 prediction lacks design-chain pLDDT for chain {chain!r}"
        )
      values.append(item[0])
      counts.append(item[1])
    if values:
      # Native AF3 pLDDT is an atomic mean.  Weight only by native atom
      # counts, never by token/residue counts (which misweight ligands and
      # modified residues).
      return float(
          np.average(
              np.asarray(values, dtype=np.float64),
              weights=np.asarray(counts, dtype=np.float64),
          )
      )
  value = _field(raw, "mean_plddt", "native_plddt")
  if value is None:
    value = _field(raw, "plddt_fraction", "plddt")
    if value is not None:
      value = float(np.asarray(value).reshape(-1)[0]) * 100.0
  if value is None:
    raise ValueError("AF3 prediction lacks design-chain pLDDT")
  try:
    score = float(np.asarray(value).reshape(-1)[0])
  except (TypeError, ValueError, IndexError) as exc:
    raise ValueError("AF3 design pLDDT is malformed") from exc
  if 0.0 <= score <= 1.0:
    score *= 100.0
  if not np.isfinite(score) or not 0.0 <= score <= 100.0:
    raise ValueError("AF3 design pLDDT is invalid")
  return score


def _native_chain_order(raw: Mapping[str, Any]) -> tuple[str, ...]:
  for field in (
      "unique_chain_ids",
      "native_unique_chain_ids",
      "native_chain_ids",
      "chain_ids",
      "token_chain_ids",
      "source_chain_ids",
      "source_token_chain_ids",
  ):
    values = _field(raw, field)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
      continue
    result: list[str] = []
    for value in values:
      value = str(value)
      if value not in result:
        result.append(value)
    if result:
      return tuple(result)
  return ()


def _design_target_iptm(
    raw: Mapping[str, Any],
    design_chains: Sequence[str],
    target_chains: Sequence[str],
) -> float | None:
  if not target_chains:
    return None
  pair = _field(raw, "chain_pair_iptm", "pair_iptm")
  values: list[float] = []
  if pair is not None:
    chain_order = _native_chain_order(raw)
    token_weights = _native_token_weights(raw)
    for design in design_chains:
      mapped_design = _mapped_native_chain_ids(raw, design)
      design_aliases = _aliases(raw, design) if mapped_design is None else mapped_design
      for target in target_chains:
        mapped_target = _mapped_native_chain_ids(raw, target)
        target_aliases = _aliases(raw, target) if mapped_target is None else mapped_target
        strict = mapped_design is not None or mapped_target is not None
        if strict:
          if not design_aliases or not target_aliases:
            return None
          # A zero-observed-token source has no applicable interface pairs.
          if (
              not any(float(token_weights.get(chain, 1)) > 0 for chain in design_aliases)
              or not any(float(token_weights.get(chain, 1)) > 0 for chain in target_aliases)
          ):
            continue
        value = _pair_value(
            pair,
            design_aliases,
            target_aliases,
            chain_order,
            token_weights,
            strict=strict,
        )
        if strict and value is None:
          return None
        if value is not None:
          values.append(value)
  if values:
    return float(np.mean(np.asarray(values, dtype=np.float64)))
  # A present pair matrix is authoritative.  Do not fall back to a global
  # iPTM when it lacks the design-to-target entries: that value may include
  # an excluded design-design interface (for example VH--VL).
  if pair is not None:
    return None
  # A global iPTM is safe only for a single design/target pair. Multichain
  # antibody/global scores may include the excluded VH-VL internal interface.
  if len(design_chains) != 1 or len(target_chains) != 1:
    return None
  fallback = _field(raw, "iptm", "native_iptm")
  if fallback is None:
    return None
  return _finite_unit(fallback, "iPTM")


def _design_ptm(
    raw: Mapping[str, Any],
    design_chains: Sequence[str],
    *,
    multichain: bool | None = None,
) -> float:
  if multichain is None:
    native_chains = _native_chain_order(raw)
    multichain = len(design_chains) > 1 or len(native_chains) > 1

  # Multichain reductions require native chain-scoped pTM. A global pTM can
  # include excluded design-design interfaces in antibody rankings.
  scoped_fields = ("chain_ptm", "per_chain_ptm", "ptm_by_chain")
  for field in scoped_fields:
    scoped = _field(raw, field)
    if scoped is None:
      continue
    values: list[float] = []
    missing: list[str] = []
    for chain in design_chains:
      value = _chain_metric(raw, field, chain)
      if value is None:
        missing.append(str(chain))
      else:
        values.append(value)
    if missing:
      raise ValueError(
          "AF3 multichain prediction lacks chain-scoped pTM for design "
          + "chain(s): "
          + ", ".join(missing)
      )
    if values:
      return float(np.mean(np.asarray(values, dtype=np.float64)))

  # Some SDK adapters expose chain-scoped values in ordinary pTM.
  global_ptm = _field(raw, "ptm")
  if isinstance(global_ptm, (Mapping, Sequence, np.ndarray)) and not isinstance(
      global_ptm, (str, bytes, bytearray)
  ):
    values = []
    missing = []
    for chain in design_chains:
      value = _chain_metric(raw, "ptm", chain)
      if value is None:
        missing.append(str(chain))
      else:
        values.append(value)
    if missing:
      raise ValueError(
          "AF3 multichain prediction lacks chain-scoped pTM for design "
          + "chain(s): "
          + ", ".join(missing)
      )
    if values:
      return float(np.mean(np.asarray(values, dtype=np.float64)))

  if multichain:
    raise ValueError(
        "AF3 multichain prediction lacks required chain-scoped pTM"
    )
  fallback = _field(raw, "ptm", "native_ptm")
  if fallback is None or isinstance(fallback, Mapping):
    raise ValueError("AF3 prediction lacks scalar pTM")
  return _finite_unit(fallback, "pTM")


def _af3_confidence(
    candidate: SearchCandidate,
    raw: Mapping[str, Any],
    *,
    protocol: str,
) -> tuple[ESMFold2Confidence, float, float | None]:
  protocol = str(protocol).strip().lower()
  design_chains = _design_chain_ids(candidate)
  if not design_chains:
    raise ValueError("AF3 candidate has no design chain")
  all_chains = _all_chain_ids(candidate)
  excluded = set(design_chains)
  if candidate.base_request.task.format == "antibody":
    roles = getattr(candidate.atoms, "antibody_chain_role_id", None)
    if roles is None:
      raise ValueError("AF3 antibody confidence requires compiled chain roles")
    labels = np.asarray(candidate.atoms.chain_id).astype(str)
    excluded.update(labels[np.asarray(roles) > 0].tolist())
  target_chains = (
      tuple(chain for chain in all_chains if chain not in excluded)
      if protocol == "complex" else ()
  )
  design_plddt = _design_plddt(raw, design_chains)
  ptm = _design_ptm(
      raw,
      design_chains,
      # A monomer with one design chain is the only unambiguous global pTM
      # fallback.
      multichain=(len(design_chains) > 1 or bool(target_chains)),
  )
  interface_iptm = (
      _design_target_iptm(raw, design_chains, target_chains)
      if protocol == "complex"
      else None
  )
  if target_chains and interface_iptm is None:
    raise ValueError("AF3 complex prediction lacks design-target iPTM")
  return (
      ESMFold2Confidence(
          candidate_id=candidate.candidate_id,
          design_plddt=design_plddt,
          interface_iptm=interface_iptm,
          ptm=ptm,
          interface_iplddt=None,
      ),
      design_plddt / 100.0,
      interface_iptm,
  )


def _confidence_score(
    confidence: ESMFold2Confidence,
    *,
    score_config: Mapping[str, Any],
    protocol: str,
) -> float:
  weights = {
      "plddt": _finite_nonnegative(
          score_config.get("plddt_weight", 0.0), "score.plddt_weight"
      ),
      "ptm": _finite_nonnegative(
          score_config.get("ptm_weight", 0.2), "score.ptm_weight"
      ),
      "iptm": _finite_nonnegative(
          score_config.get("iptm_weight", 0.8), "score.iptm_weight"
      ),
  }
  if weights["plddt"] + weights["ptm"] <= 0.0:
    raise ValueError("AF3 pLDDT and pTM weights must have positive total")
  values = {
      "plddt": confidence.design_plddt / 100.0,
      "ptm": _finite_unit(confidence.ptm, "pTM"),
  }
  if protocol == "complex" and confidence.interface_iptm is not None:
    values["iptm"] = _finite_unit(confidence.interface_iptm, "iPTM")
  applicable = [key for key in values if weights[key] > 0.0]
  denominator = float(sum(weights[key] for key in applicable))
  if denominator <= 0.0:
    raise ValueError("AF3 applicable score weights must have positive total")
  score = float(sum(weights[key] * values[key] for key in applicable) / denominator)
  if not np.isfinite(score):
    raise ValueError("AF3 score is non-finite")
  return score


def _prediction_path(raw: Mapping[str, Any]) -> Path:
  value = raw.get("structure_path", raw.get("cif_path", raw.get("pdb_path")))
  if value is None or not str(value):
    raise ValueError("AF3 prediction lacks a structure path")
  path = Path(str(value)).expanduser()
  if not path.is_file():
    raise ValueError(f"AF3 structure path is missing: {path}")
  return path


def _af3_metrics(
    raw: Mapping[str, Any],
    score: float,
    confidence: ESMFold2Confidence,
) -> dict[str, float]:
  metrics = {
      "af3_score": float(score),
      "af3_plddt": float(confidence.design_plddt / 100.0),
      "af3_ptm": float(confidence.ptm),
  }
  if confidence.interface_iptm is not None:
    metrics["af3_iptm"] = float(confidence.interface_iptm)
  native_plddt = _field(raw, "mean_plddt", "native_plddt")
  if native_plddt is not None:
    try:
      value = float(np.asarray(native_plddt).reshape(-1)[0])
      if np.isfinite(value):
        metrics["af3_native_plddt"] = value / 100.0 if value > 1.0 else value
    except (TypeError, ValueError, IndexError):
      pass
  return metrics


def _result_from_prediction(
    candidate: SearchCandidate,
    plan: Any,
    variant: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    score_config: Mapping[str, Any],
) -> FoldRewardResult:
  fold_cif = _prediction_path(prediction)
  protocol = str(prediction.get("protocol", "complex")).lower()
  confidence, _plddt_fraction, _interface_iptm = _af3_confidence(
      candidate,
      prediction,
      protocol=protocol,
  )
  score = _confidence_score(
      confidence,
      score_config=score_config,
      protocol=protocol,
  )
  folded = read_structure(
      fold_cif,
      declared_polymer_chain_modalities=candidate_chain_modalities(candidate),
  )
  confidence = ESMFold2Confidence(
      candidate_id=confidence.candidate_id,
      design_plddt=confidence.design_plddt,
      interface_iptm=confidence.interface_iptm,
      ptm=confidence.ptm,
      interface_iplddt=None,
      motif_rmsd_angstrom=motif_rmsd_angstrom(candidate, folded),
  )
  reported = _reported_polymer_sequences(candidate, variant)
  design_chains = _design_chain_ids(candidate)
  cache = prediction.get("cache")
  if not isinstance(cache, Mapping):
    cache = {
        "model": str(prediction.get("model", "af3")),
        "protocol": protocol,
        "sample": int(prediction.get("sample", 0)),
    }
  return FoldRewardResult(
      candidate_id=candidate.candidate_id,
      score=float(score),
      confidence=confidence,
      fold_cif=fold_cif,
      sequence_source=str(plan.source),
      cache=dict(cache),
      design_sequence=(
          reported.get(design_chains[0]) if len(design_chains) == 1 else None
      ),
      metrics=_af3_metrics(prediction, score, confidence),
  )


def run_af3_reward_batch(
    candidates: Sequence[SearchCandidate],
    *,
    reward_config: Mapping[str, Any],
    repo_root: Path,
    owner_work_root: Path,
    stage_output_root: Path | None = None,
) -> tuple[tuple[FoldRewardResult, ...], tuple[FoldRewardFailure, ...]]:
  """Run sequence planning, one persistent AF3 service, and score reduction."""

  if not candidates:
    return (), ()
  if owner_work_root.exists() or owner_work_root.is_symlink():
    raise RuntimeError(f"AF3 reward work root already exists: {owner_work_root}")
  owner_work_root.mkdir(parents=True)

  sequence_config = reward_config["sequence"]
  af3_config = reward_config["af3"]
  score_config = reward_config.get("score", {})
  planned: list[tuple[SearchCandidate, Any]] = []
  failures: list[FoldRewardFailure] = []
  for candidate in candidates:
    try:
      plan = plan_sequence_source(
          candidate,
          requested=str(sequence_config.get("source", "auto")),
          user_fixed_positions=tuple(sequence_config.get("fixed_positions", ())),
          fixed_interface=sequence_config.get("fixed_interface"),
          reward_method="esmfold2",
      )
      planned.append((candidate, plan))
    except (MPNNSequenceError, ValueError, OSError) as error:
      failures.append(FoldRewardFailure(candidate.candidate_id, str(error)))

  antibody_plans: dict[str, Any] = {}
  if any(
      candidate.base_request.task.format == "antibody"
      for candidate, _plan in planned
  ):
    accepted: list[tuple[SearchCandidate, Any]] = []
    for candidate, plan in planned:
      try:
        if candidate.base_request.task.format == "antibody":
          antibody_plans[candidate.candidate_id] = build_antibody_mpnn_plan(
            candidate, sequence_plan=plan,
          )
        accepted.append((candidate, plan))
      except MPNNSequenceError as error:
        failures.append(FoldRewardFailure(candidate.candidate_id, str(error)))
    planned = accepted

  effective_reward_config = reward_config
  if (
      str(sequence_config.get("source", "auto")) == "auto"
      and any(plan.source == "proteinmpnn" for _candidate, plan in planned)
  ):
    try:
      effective_reward_config = normalize_reward_sequence_config(
          reward_config,
          requested="auto",
          require_soluble=True,
      )
    except ValueError as error:
      raise MPNNSystemError(str(error)) from error

  try:
    sequence_plans, mpnn_failures = generate_mpnn_sequences(
        planned,
        reward_config=effective_reward_config,
        repo_root=repo_root,
        work_dir=owner_work_root / "mpnn-batches",
        **({"antibody_plans": antibody_plans} if antibody_plans else {}),
    )
  except MPNNSystemError:
    raise
  except MPNNSequenceError as error:
    raise MPNNSystemError(f"batched MPNN preparation failed: {error}") from error
  failures.extend(
      FoldRewardFailure(candidate_id, message)
      for candidate_id, message in sorted(mpnn_failures.items())
  )

  ledgers: dict[str, tuple[SearchCandidate, Any, Mapping[str, Any], Path]] = {}
  for candidate_index, (candidate, _initial_plan) in enumerate(planned):
    if candidate.candidate_id in mpnn_failures:
      continue
    candidate_root = owner_work_root / f"candidate_{candidate_index}"
    try:
      plan = sequence_plans[candidate.candidate_id]
      variant = _variant_from_plan(
          candidate,
          plan,
          name=f"candidate_{candidate_index}",
      )
      candidate_root.mkdir(parents=True, exist_ok=False)
      if candidate.base_request.task.format == "antibody":
        _write_antibody_stage(
            candidate,
            plan,
            variant,
            effective_reward_config,
            candidate_root,
            stage_output_root,
        )
      ledgers[candidate.candidate_id] = (candidate, plan, variant, candidate_root)
    except (KeyError, MPNNSequenceError, ValueError, OSError) as error:
      failures.append(FoldRewardFailure(candidate.candidate_id, str(error)))

  if not ledgers:
    return (), tuple(failures)

  method = str(af3_config.get("model", "af3-la"))
  device: Any = "cpu"
  if torch.cuda.is_available():
    device = torch.device("cuda", torch.cuda.current_device())
  backend = AF3Backend(
      repo_root,
      af3_config,
      method=method,
      device=device,
  )
  results: list[FoldRewardResult] = []
  try:
    for candidate, _initial_plan in planned:
      entry = ledgers.get(candidate.candidate_id)
      if entry is None:
        continue
      _candidate, plan, variant, candidate_root = entry
      try:
        predictions = list(backend.predict(
            candidate,
            variant,
            candidate_root / "af3",
        ))
      except AF3BackendError:
        raise
      except (AF3InputError, KeyError, ValueError, OSError) as error:
        failures.append(FoldRewardFailure(candidate.candidate_id, str(error)))
        continue
      if len(predictions) != 1:
        failures.append(
            FoldRewardFailure(
                candidate.candidate_id,
                f"AF3 returned {len(predictions)} predictions; expected exactly one",
            )
        )
        continue
      try:
        results.append(
            _result_from_prediction(
                candidate,
                plan,
                variant,
                predictions[0],
                score_config=score_config,
            )
        )
      except (KeyError, TypeError, ValueError, IndexError, OSError) as error:
        failures.append(
            FoldRewardFailure(
                candidate.candidate_id,
                f"AF3 candidate result is invalid: {type(error).__name__}: {error}",
            )
        )
  finally:
    close = getattr(backend, "close", None)
    if close is not None:
      close()
  return tuple(results), tuple(failures)


__all__ = [
    "FoldRewardFailure",
    "FoldRewardResult",
    "run_af3_reward_batch",
]
