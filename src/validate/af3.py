"""Validation boundary for native AlphaFold 3 predictions.

This module contains only schema and confidence checks.  It deliberately does
not import ``src.validate`` or any ESM runtime; callers can use it with a
lightweight fake backend in CPU planning tests and with ``AF3Backend`` in the
native worker process.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import json

import numpy as np

from src.design.rewards.af3_backend import AF3Backend, AF3BackendError


class AF3ValidationError(ValueError):
  """Raised when a native AF3 prediction violates the public ABI."""


def _scalar(value: Any, name: str) -> float:
  try:
    array = np.asarray(value)
    if array.size != 1 or isinstance(value, (bool, np.bool_)):
      raise ValueError("not one numeric value")
    result = float(array.reshape(-1)[0])
  except (TypeError, ValueError, IndexError) as exc:
    raise AF3ValidationError(f"AF3 {name} must be a finite scalar") from exc
  if not np.isfinite(result):
    raise AF3ValidationError(f"AF3 {name} must be finite")
  return result


def _finite_array(value: Any, name: str) -> None:
  if value is None:
    return
  try:
    array = np.asarray(value, dtype=np.float64)
  except (TypeError, ValueError) as exc:
    raise AF3ValidationError(f"AF3 {name} is not numeric") from exc
  if not np.isfinite(array).all():
    raise AF3ValidationError(f"AF3 {name} contains non-finite values")


def validate_af3_prediction(
    prediction: Mapping[str, Any],
    *,
    expected_model: str | None = None,
    expected_protocol: str | None = None,
    expected_sample: int | None = None,
    require_structure: bool = False,
) -> dict[str, Any]:
  """Validate and return one normalized AF3 prediction dictionary."""
  if not isinstance(prediction, Mapping):
    raise AF3ValidationError("AF3 prediction must be a mapping")
  result = dict(prediction)
  model = str(result.get("model", ""))
  if model not in {"af3", "af3-la"}:
    raise AF3ValidationError(f"unsupported AF3 model label {model!r}")
  protocol = str(result.get("protocol", ""))
  if protocol not in {"complex", "monomer"}:
    raise AF3ValidationError(f"unsupported AF3 protocol {protocol!r}")
  raw_sample = result.get("sample")
  if isinstance(raw_sample, (bool, np.bool_)):
    raise AF3ValidationError("AF3 prediction sample must be an integer")
  if isinstance(raw_sample, (float, np.floating)) and not float(raw_sample).is_integer():
    raise AF3ValidationError("AF3 prediction sample must be an integer")
  try:
    sample = int(raw_sample)
  except (KeyError, TypeError, ValueError) as exc:
    raise AF3ValidationError("AF3 prediction sample must be an integer") from exc
  if sample < 0:
    raise AF3ValidationError("AF3 prediction sample must be non-negative")
  key = str(result.get("key", ""))
  expected_key = f"{protocol}_sample{sample}"
  if key != expected_key:
    raise AF3ValidationError(
        f"AF3 prediction key must be {expected_key!r}, got {key!r}"
    )
  if expected_model is not None and model != str(expected_model):
    raise AF3ValidationError(f"AF3 model is {model!r}, expected {expected_model!r}")
  if expected_protocol is not None and protocol != str(expected_protocol):
    raise AF3ValidationError(
        f"AF3 protocol is {protocol!r}, expected {expected_protocol!r}"
    )
  if expected_sample is not None and sample != int(expected_sample):
    raise AF3ValidationError(
        f"AF3 sample is {sample}, expected {int(expected_sample)}"
    )

  native_plddt = _scalar(result.get("native_plddt"), "native_plddt")
  plddt = _scalar(result.get("plddt"), "plddt")
  if not 0.0 <= native_plddt <= 100.0:
    raise AF3ValidationError("native AF3 pLDDT must lie in [0, 100]")
  if not 0.0 <= plddt <= 1.0:
    raise AF3ValidationError("validation AF3 pLDDT must lie in [0, 1]")
  if not np.isclose(plddt, native_plddt / 100.0, atol=1e-5, rtol=1e-5):
    raise AF3ValidationError("AF3 pLDDT fraction does not match native pLDDT")
  result["model"] = model
  result["protocol"] = protocol
  result["sample"] = sample
  result["key"] = key
  result["native_plddt"] = native_plddt
  result["plddt"] = plddt

  metadata = result.get("metadata")
  if not isinstance(metadata, Mapping):
    raise AF3ValidationError("AF3 prediction metadata must be a mapping")
  if metadata.get("key") != key:
    raise AF3ValidationError("AF3 metadata key does not match prediction key")
  raw_metadata_sample = metadata.get("sample", -1)
  if isinstance(raw_metadata_sample, (bool, np.bool_)):
    raise AF3ValidationError("AF3 metadata sample is not an integer")
  if isinstance(raw_metadata_sample, (float, np.floating)) and not float(raw_metadata_sample).is_integer():
    raise AF3ValidationError("AF3 metadata sample is not an integer")
  try:
    metadata_sample = int(raw_metadata_sample)
  except (TypeError, ValueError) as exc:
    raise AF3ValidationError("AF3 metadata sample is not an integer") from exc
  if str(metadata.get("protocol")) != protocol or metadata_sample != sample:
    raise AF3ValidationError("AF3 metadata protocol/sample does not match prediction")

  # Native confidence uses one square token matrix per output. PDE is never
  # a substitute for PAE, even though their shapes are identical.
  matrices = {}
  for name, alias in (("full_pae", "pae"), ("full_pde", "pde")):
    value = result.get(name, result.get(alias))
    if value is None:
      raise AF3ValidationError("AF3 prediction must expose separate full PAE and PDE")
    _finite_array(value, name)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or not array.shape[0]:
      raise AF3ValidationError(f"AF3 {name} must be a nonempty square token matrix")
    if np.any(array < 0):
      raise AF3ValidationError(f"AF3 {name} must be non-negative")
    matrices[name] = array
    result[name] = value
  if matrices["full_pae"].shape != matrices["full_pde"].shape:
    raise AF3ValidationError("AF3 PAE and PDE token dimensions differ")
  labels = result.get("token_chain_ids")
  if labels is not None:
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != matrices["full_pae"].shape[0]:
      raise AF3ValidationError("AF3 token chain labels do not match confidence matrices")
    if any(not str(label) for label in labels):
      raise AF3ValidationError("AF3 token chain labels must not be empty")
  if result.get("source_token_indices") is not None:
    try:
      indexes = np.asarray(result["source_token_indices"], dtype=np.int64)
      mask = np.asarray(result["native_token_source_mask"], dtype=bool)
      native_pae = np.asarray(result["native_full_pae"], dtype=np.float64)
      native_pde = np.asarray(result["native_full_pde"], dtype=np.float64)
      locators = result["source_token_residue_map"]
      valid = (indexes.ndim == mask.ndim == 1
               and len(indexes) == matrices["full_pae"].shape[0]
               and native_pae.shape == native_pde.shape == (len(mask), len(mask))
               and len(set(indexes.tolist())) == len(indexes)
               and set(indexes.tolist()) == set(np.flatnonzero(mask).tolist())
               and len(locators) == len(indexes)
               and [str(item["chain_id"]) for item in locators] == list(map(str, result["token_chain_ids"]))
               and [int(item["res_id"]) for item in locators] == list(map(int, result["token_res_ids"]))
               and np.array_equal(native_pae[np.ix_(indexes, indexes)], matrices["full_pae"])
               and np.array_equal(native_pde[np.ix_(indexes, indexes)], matrices["full_pde"]))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
      raise AF3ValidationError("AF3 source confidence projection is malformed") from exc
    if not valid:
      raise AF3ValidationError("AF3 source confidence projection disagrees with native token mapping")
  for name in ("ptm", "iptm"):
    if result.get(name) is not None:
      value = _scalar(result[name], name)
      if not 0.0 <= value <= 1.0:
        raise AF3ValidationError(f"AF3 {name} must lie in [0, 1]")
      result[name] = value
  if require_structure:
    structure_path = result.get("cif_path", result.get("pdb_path"))
    if not structure_path or not Path(str(structure_path)).is_file():
      raise AF3ValidationError("AF3 prediction structure path does not exist")
  return result


def normalize_af3_confidence(
    prediction: Mapping[str, Any], design_chains: Sequence[str],
    *, antibody_chains: Sequence[str] = (),
) -> dict[str, Any]:
  """Expose native confidence to metrics, including model-free recomputation.

  Matrices remain in the private native confidence file after sequence state is
  compacted. Chain arrays retain native ordering; token labels are mapped to the
  source identities of the published CIF. Binder confidence excludes interfaces
  between design chains, including the antibody VH/VL interface.
  """
  from src.design.rewards.af3_runner import _design_ptm, _design_target_iptm

  result = dict(prediction)
  raw = {}
  path_value = result.get("raw_confidence_path")
  if path_value:
    path = Path(str(path_value))
    if any(item.is_symlink() for item in (path, *path.parents)):
      raise AF3ValidationError("AF3 confidence path must not contain a symlink")
    try:
      raw = json.loads(path.read_text())
    except (OSError, ValueError, TypeError) as exc:
      raise AF3ValidationError("AF3 native confidence file is unreadable") from exc
    if not isinstance(raw, dict):
      raise AF3ValidationError("AF3 native confidence file must be a mapping")
  confidence = dict(raw)
  confidence.update(result.get("confidence") or {})
  # Top-level native values are authoritative before the private file is written.
  for name in (
      "full_pae", "full_pde", "chain_ptm", "chain_pair_iptm", "chain_ids",
      "native_chain_ids", "source_chain_ids", "token_chain_ids", "ptm", "iptm",
      "native_to_source_chain", "source_to_native_chain", "mean_plddt",
      "native_full_pae", "native_full_pde", "native_token_chain_ids", "native_token_res_ids",
      "native_token_source_mask", "source_token_indices", "source_token_residue_map",
      "source_token_chain_ids", "token_res_ids", "token_ins_codes", "token_chain_label_map",
      "native_per_chain_source_token_count", "native_per_chain_token_count",
  ):
    if result.get(name) is not None:
      confidence[name] = result[name]
  restored = {str(native): str(source) for native, source in
              dict(confidence.get("native_to_source_chain") or {}).items()}
  if not restored:
    native = confidence.get("native_chain_ids", confidence.get("chain_ids", ()))
    source = confidence.get("source_chain_ids", ())
    if len(native) == len(source):
      restored = dict(zip(map(str, native), map(str, source)))
  confidence["native_to_source_chain"] = restored
  partition: dict[str, list[str]] = {}
  for native, source in restored.items():
    partition.setdefault(source, []).append(native)
  confidence["source_to_native_chain"] = partition
  labels = confidence.get("source_token_chain_ids", confidence.get("token_chain_ids"))
  if labels is not None:
    if confidence.get("source_token_indices") is None:
      labels = [restored.get(str(chain), str(chain)) for chain in labels]
    else:
      labels = list(map(str, labels))
    result["token_chain_ids"] = labels
    confidence["token_chain_ids"] = labels
  pae = confidence.get("full_pae", confidence.get("pae_matrix"))
  pde = confidence.get("full_pde", confidence.get("pde_matrix"))
  if confidence.get("source_token_indices") is not None and (pae is None or pde is None):
    try:
      indexes = np.asarray(confidence["source_token_indices"], dtype=np.int64)
      pae = np.asarray(confidence["native_full_pae"])[np.ix_(indexes, indexes)]
      pde = np.asarray(confidence["native_full_pde"])[np.ix_(indexes, indexes)]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
      raise AF3ValidationError("AF3 native confidence has no valid source projection") from exc
    confidence.update(full_pae=pae, full_pde=pde)
  if pae is None or pde is None:
    raise AF3ValidationError("AF3 native confidence must contain PAE and PDE")
  native_plddt = result.get("native_plddt", confidence.get("native_plddt", confidence.get("mean_plddt")))
  result.update(full_pae=pae, full_pde=pde, native_plddt=native_plddt,
                plddt=_scalar(native_plddt, "native_plddt") / 100.0)
  for name in ("native_full_pae", "native_full_pde", "native_token_chain_ids", "native_token_res_ids",
               "native_token_source_mask", "source_token_indices", "source_token_residue_map", "token_res_ids"):
    if name in confidence:
      result[name] = confidence[name]
  result = validate_af3_prediction(result)
  confidence["plddt"] = result["native_plddt"]
  confidence["pae_matrix"] = result["full_pae"]
  chains = list(dict.fromkeys(map(str, labels or confidence.get("source_chain_ids", ()))))
  design = [str(chain) for chain in design_chains if str(chain) in chains]
  excluded = set(design) | set(map(str, antibody_chains))
  targets = [chain for chain in chains if chain not in excluded]
  if design:
    try:
      confidence["binder_ptm"] = _design_ptm(confidence, design, multichain=len(chains) > 1)
      confidence["binder_iptm"] = _design_target_iptm(confidence, design, targets)
    except ValueError as exc:
      raise AF3ValidationError(str(exc)) from exc
  result["confidence"] = confidence
  result["_confidence_chain_map"] = {chain: chain for chain in chains}
  return result


def validate_af3_predictions(
    predictions: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
  """Validate a batch and reject duplicate or unstable sample keys."""
  if isinstance(predictions, Mapping):
    entries = list(predictions.values())
  elif isinstance(predictions, Sequence) and not isinstance(predictions, (str, bytes)):
    entries = list(predictions)
  else:
    raise AF3ValidationError("AF3 predictions must be a sequence or mapping")
  result = [validate_af3_prediction(item, **kwargs) for item in entries]
  keys = [item["key"] for item in result]
  if len(set(keys)) != len(keys):
    raise AF3ValidationError("AF3 prediction keys must be unique")
  return result


class AF3Validator:
  """Thin validation wrapper around a persistent AF3 backend."""

  def __init__(self, backend: AF3Backend | Any, *, require_structure: bool = False) -> None:
    self.backend = backend
    self.require_structure = require_structure
    self._closed = False

  def predict(
      self,
      input_record: Any,
      variant: Any,
      directory: str | Path,
      completed_keys: Iterable[str] = (),
      *,
      prepared_inputs: Mapping[str, Mapping[str, Any]] | None = None,
  ) -> Iterator[dict[str, Any]]:
    if self._closed:
      raise AF3ValidationError("AF3 validator is closed")
    extra = {} if prepared_inputs is None else {"prepared_inputs": prepared_inputs}
    predictions = self.backend.predict(
        input_record,
        variant,
        directory,
        completed_keys=completed_keys,
        **extra,
    )
    seen = set()
    try:
      for prediction in predictions:
        result = validate_af3_prediction(
            prediction, expected_model=getattr(self.backend, "method", None),
            require_structure=self.require_structure,
        )
        if result["key"] in seen:
          raise AF3ValidationError("AF3 prediction keys must be unique")
        seen.add(result["key"])
        yield result
    finally:
      close = getattr(predictions, "close", None)
      if close is not None:
        close()

  def set_log_dir(self, log_dir: str | Path | None) -> None:
    setter = getattr(self.backend, "set_log_dir", None)
    if setter is not None:
      setter(log_dir)

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    close = getattr(self.backend, "close", None)
    if close is not None:
      close()

  def __enter__(self) -> "AF3Validator":
    return self

  def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
    self.close()


__all__ = [
    "AF3Backend",
    "AF3BackendError",
    "AF3ValidationError",
    "AF3Validator",
    "normalize_af3_confidence",
    "validate_af3_prediction",
    "validate_af3_predictions",
]
