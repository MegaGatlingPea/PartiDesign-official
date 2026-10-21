"""Public AlphaFold 3 backend used by the standalone design package.

The backend is intentionally small: conversion to native AF3 input lives in
``af3_input`` and the long-lived SDK/JAX state lives in ``af3_worker``.  This
class owns request fan-out (complex/monomer) and the stable prediction key
contract consumed by reward planning.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
import os
import shutil
import tempfile
from typing import Any

from .af3_input import AF3Input, AF3InputError, build_af3_input
from .af3_worker import AF3Worker, _get, _normalise_method


class AF3BackendError(RuntimeError):
  """Raised when an AF3 backend request cannot be completed."""

  def __init__(self, message: str, *, fatal: bool = False) -> None:
    super().__init__(message)
    self.fatal = bool(fatal)


def _config_section(config: Any) -> Any:
  """Return the shared AF3 section while accepting already-normalized config."""
  if isinstance(config, Mapping):
    for name in ("af3",):
      section = config.get(name)
      if isinstance(section, Mapping):
        return section
    return config
  for name in ("af3",):
    section = getattr(config, name, None)
    if section is not None:
      return section
  return config


def _protocols(config: Any) -> tuple[str, ...]:
  mode = str(_get(config, "mode", "complex")).strip().lower()
  if mode == "both":
    return ("complex", "monomer")
  if mode in {"complex", "monomer"}:
    return (mode,)
  raise AF3BackendError("AF3 mode must be complex, monomer, or both")


def _variant_with_ligand_smiles(input_record: Any, variant: Any) -> Any:
  """Share trusted design/context ligand chemistry across prepare and predict."""
  native_variant = dict(variant) if isinstance(variant, Mapping) else variant
  if not isinstance(native_variant, dict):
    return native_variant
  ligand_entries = dict(getattr(input_record, "ligand_entries", {}) or {})
  task_dir = getattr(input_record, "task_dir", None)
  if task_dir is not None and (Path(task_dir) / "resolved_config.yaml").is_file():
    from src.validate.inputs import ValidationInput

    if isinstance(input_record, ValidationInput):
      from src.validate.ligands import (
          LigandResolutionError, _task_config, resolve_context_ligands,
      )

      try:
        # Most de novo tasks have no source input. Only an explicit context
        # SMILES needs the resolver's authoritative-source requirements.
        _, task_config, _ = _task_config(Path(task_dir), allow_missing=True)
        if not task_config:
          raise LigandResolutionError(
              f"{Path(task_dir) / 'resolved_config.yaml'} has no resolved task {Path(task_dir).name!r}"
          )
        conditions = task_config.get("condition")
        if conditions is None:
          conditions = {}
        if not isinstance(conditions, Mapping):
          raise LigandResolutionError("resolved task condition must be a mapping")
        specifications = conditions.get("ligand")
        if specifications is None:
          specifications = {}
        if not isinstance(specifications, Mapping):
          raise LigandResolutionError("resolved task ligand condition must be a mapping")
        context = []
        for chain, spec in specifications.items():
          chain = str(chain)
          if chain in input_record.design_chains:
            continue
          if not isinstance(spec, Mapping):
            raise LigandResolutionError(f"context ligand {chain!r} condition must be a mapping")
          if str(spec.get("mode", "")).strip().lower() == "smiles":
            context.append(chain)
        if context:
          resolved, _ = resolve_context_ligands(
              Path(task_dir), input_record.atoms,
              design_chains=input_record.design_chains, chains=context,
          )
          ligand_entries.update(resolved)
      except LigandResolutionError as exc:
        raise AF3BackendError(f"AF3 context ligand chemistry is invalid: {exc}") from exc
  if ligand_entries:
    sequences = dict(native_variant.get("sequences") or {})
    for chain, entry in ligand_entries.items():
      sequences[str(chain)] = {"smiles": entry["smiles"]}
    native_variant["sequences"] = sequences
  return native_variant


def prepare_af3_inputs(input_record: Any, variant: Mapping[str, Any], config: Any,
                       repo_root: str | Path) -> dict[str, dict[str, Any]]:
  """Build exact protocol requests before CPU searches, without a GPU backend."""
  from partidesign.paths import af3_asset
  section = _config_section(config)
  native_variant = _variant_with_ligand_smiles(input_record, variant)
  return {protocol: build_af3_input(
      input_record, native_variant, seed=int(_get(section, "seed", 101)), sample=0,
      protocol=protocol, ccd_asset_root=af3_asset("af3", Path(repo_root)) / "ccd",
  ).payload() for protocol in _protocols(section)}


def _publish_prediction_artifacts(
    prediction: dict[str, Any],
    *,
    request_dir: Path,
    protocol_dir: Path,
) -> None:
  """Move one missing sample's private artifacts into its final directory."""
  moved: dict[Path, Path] = {}
  for field in (
      "structure_path",
      "cif_path",
      "native_cif_path",
      "pdb_path",
      "confidence_path",
      "raw_confidence_path",
  ):
    value = prediction.get(field)
    if value is None or not str(value):
      continue
    source = Path(str(value)).expanduser().resolve()
    if source in moved:
      prediction[field] = str(moved[source])
      continue
    # Lightweight test workers may return an already-published path.  Only
    # relocate artifacts emitted under this request's private directory.
    if not source.is_relative_to(request_dir):
      continue
    if not source.exists():
      continue
    destination = protocol_dir / source.name
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
      raise AF3BackendError(f"AF3 artifact is not an owned regular file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Completed sample keys are skipped before this function. A missing,
    # interrupted or explicitly rerun slot may replace its own stale artifacts.
    os.replace(source, destination)
    moved[source] = destination
    prediction[field] = str(destination)


def _as_prediction_list(value: Any) -> list[dict[str, Any]]:
  """Normalize a worker's one-or-many result without losing mapping values."""
  if isinstance(value, Mapping):
    # A native worker emits a single prediction mapping only in lightweight
    # test doubles; a keyed mapping is also accepted for service adapters.
    if any(key in value for key in ("model", "protocol", "sample", "cif_path")):
      return [dict(value)]
    result: list[dict[str, Any]] = []
    for key, prediction in value.items():
      if not isinstance(prediction, Mapping):
        raise AF3BackendError(f"worker result for {key!r} is not a mapping")
      item = dict(prediction)
      item.setdefault("key", str(key))
      result.append(item)
    return result
  if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
    result = []
    for index, prediction in enumerate(value):
      if not isinstance(prediction, Mapping):
        raise AF3BackendError(f"worker result {index} is not a mapping")
      result.append(dict(prediction))
    return result
  raise AF3BackendError("AF3 worker returned neither a prediction mapping nor a sequence")


class AF3Backend:
  """Shared native AF3 service with ESM/AF2-compatible prediction records.

  ``predict`` streams prediction dictionaries.  A request configured with
  ``mode='both'`` contributes complex records followed by monomer records;
  each record uses a zero-based key such as ``complex_sample0`` or
  ``monomer_sample0``.  The worker is constructed once and reused across all
  calls, so callers should keep this object alive for the owner process.
  """

  def __init__(
      self,
      repo_root: str | Path,
      config: Any,
      method: str = "af3-la",
      device: str | int | None = None,
      *,
      worker: Any | None = None,
  ) -> None:
    self.repo_root = Path(repo_root).expanduser().resolve()
    self.config = config
    self.method = _normalise_method(method)
    self.device = device if device is not None else _get(config, "device", None)
    self._worker = (
        worker
        if worker is not None
        else AF3Worker(self.repo_root, config, self.method, device=self.device)
    )
    from partidesign.paths import af3_asset
    self._ccd_asset_root = (af3_asset("af3", self.repo_root) / "ccd"
                            if worker is None else None)
    self._log_dir: Path | None = None
    self._log_dir_explicit = False
    self._closed = False

  @property
  def worker(self) -> Any:
    """Return the persistent service for callers needing the payload ABI."""
    return self._worker

  @property
  def model_runner(self) -> Any | None:
    return getattr(self._worker, "model_runner", None)

  def set_log_dir(self, log_dir: str | Path | None) -> None:
    """Route child-native diagnostics without replacing its resident model."""
    self._log_dir_explicit = log_dir is not None
    self._log_dir = None if log_dir is None else Path(log_dir).expanduser().resolve()
    setter = getattr(self._worker, "set_log_dir", None)
    if setter is not None:
      try:
        setter(self._log_dir)
      except Exception as exc:
        if isinstance(exc, AF3BackendError):
          raise
        raise AF3BackendError(
            f"AF3 log setup failed: {type(exc).__name__}: {exc}",
            fatal=bool(getattr(exc, "fatal", False)),
        ) from exc

  def predict(
      self,
      input_record: Any,
      variant: Any,
      directory: str | Path,
      completed_keys: Iterable[str] = (),
      *,
      prepared_inputs: Mapping[str, Mapping[str, Any]] | None = None,
  ) -> Iterator[dict[str, Any]]:
    """Stream configured AF3 protocols so completed results survive later failures."""
    if self._closed:
      raise AF3BackendError("AF3 backend is closed")
    section = _config_section(self.config)
    samples = int(_get(section, "samples", 1))
    if samples < 1:
      raise AF3BackendError("AF3 samples must be positive")
    seed = int(_get(section, "seed", 101))
    output_root = Path(directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    completed = {str(value) for value in completed_keys}
    if not self._log_dir_explicit:
      self.set_log_dir(output_root / "logs")

    native_variant = variant
    variant_prepared = False
    for protocol in _protocols(section):
      expected = {f"{protocol}_sample{index}" for index in range(samples)}
      if expected <= completed:
        continue
      # ``samples`` is a ModelRunner setting, not a request loop.  Building and
      # featurising one input per protocol is what lets a persistent runner
      # return all diffusion samples without reloading weights or recompiling
      # one graph per sample.
      if not variant_prepared:
        native_variant = _variant_with_ligand_smiles(input_record, variant)
        variant_prepared = True
      current_input = build_af3_input(
          input_record,
          native_variant,
          seed=seed,
          sample=0,
          protocol=protocol,
          ccd_asset_root=self._ccd_asset_root,
      )
      if prepared_inputs is None:
        native_input = current_input
      else:
        from .af3_mapping import validate_prepared_request

        try:
          prepared = prepared_inputs[protocol]
          validate_prepared_request(prepared, current_input.payload())
          native_input = AF3Input.from_payload(prepared)
        except (KeyError, TypeError, ValueError, AF3InputError) as exc:
          raise AF3BackendError(f"AF3 prepared {protocol} input is incompatible: {exc}") from exc
      protocol_dir = output_root / protocol
      protocol_dir.mkdir(parents=True, exist_ok=True)
      request_dir = Path(tempfile.mkdtemp(prefix=".af3-request-", dir=protocol_dir))
      try:
        try:
          raw_predictions = self._worker.predict(native_input, request_dir)
        except Exception as exc:
          if isinstance(exc, AF3BackendError):
            raise
          raise AF3BackendError(
              f"AF3 {protocol} prediction failed: {type(exc).__name__}: {exc}",
              fatal=bool(getattr(exc, "fatal", False)),
          ) from exc
        predictions = _as_prediction_list(raw_predictions)
        if len(predictions) != samples:
          raise AF3BackendError(
              f"AF3 worker returned {len(predictions)} {protocol} samples; "
              f"configured for {samples}"
          )
        seen_keys: set[str] = set()
        for fallback_sample, prediction in enumerate(predictions):
          raw_sample = prediction.get("sample", fallback_sample)
          try:
            sample = int(raw_sample)
          except (TypeError, ValueError) as exc:
            raise AF3BackendError(
                f"AF3 worker returned non-integer {protocol} sample {raw_sample!r}"
            ) from exc
          if sample < 0 or sample >= samples:
            raise AF3BackendError(
                f"AF3 worker returned out-of-range {protocol} sample {sample}"
            )
          key = f"{protocol}_sample{sample}"
          if key in seen_keys:
            raise AF3BackendError(f"AF3 worker returned duplicate sample {key}")
          seen_keys.add(key)
          if key in completed:
            # This sample was already atomically published by an earlier run;
            # leave its bytes untouched and let temporary files be removed.
            continue
          _publish_prediction_artifacts(
              prediction, request_dir=request_dir, protocol_dir=protocol_dir
          )
          metadata = dict(prediction.get("metadata") or {})
          metadata.update({"key": key, "protocol": protocol, "sample": sample})
          prediction.update({
              "key": key,
              "model": self.method,
              "protocol": protocol,
              "status": prediction.get("status", "complete"),
              "structure_path": prediction.get(
                  "structure_path", prediction.get("cif_path", prediction.get("pdb_path", ""))
              ),
              "sample": sample,
              "sample_id": key,
              "prediction_id": f"{self.method}_{key}",
              "metadata": metadata,
              "msa_context": dict(native_input.msa_context),
              "source_atom_ledger": prediction.get(
                  "source_atom_ledger", list(native_input.source_atom_ledger)
              ),
              "source_inter_residue_bonds": native_input.source_inter_residue_bonds,
              "source_polymer_sequences": {
                  source: list(sequence) for source, sequence in native_input.source_polymer_sequences.items()
              },
              "source_to_native_chain": dict(native_input.source_to_native_chain),
              "native_to_source_chain": dict(native_input.native_to_source_chain),
              "native_component_map": dict(native_input.native_component_map),
              "source_component_map": {
                  str(native): str(source)
                  for source, native in native_input.native_component_map.items()
              },
              "source_residue_map": {
                  key: [None if item is None else dict(item) for item in value]
                  for key, value in native_input.source_residue_map.items()
              },
          })
          yield prediction
      finally:
        shutil.rmtree(request_dir, ignore_errors=True)

  def predict_payload(
      self,
      payload: AF3Input | Mapping[str, Any],
      workdir: str | Path,
  ) -> list[dict[str, Any]]:
    """Forward the worker payload ABI for the reward facade."""
    if self._closed:
      raise AF3BackendError("AF3 backend is closed")
    try:
      return _as_prediction_list(self._worker.predict(payload, workdir))
    except Exception as exc:
      if isinstance(exc, AF3BackendError):
        raise
      raise AF3BackendError(
          f"AF3 prediction failed: {type(exc).__name__}: {exc}",
          fatal=bool(getattr(exc, "fatal", False)),
      ) from exc

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    close = getattr(self._worker, "close", None)
    if close is not None:
      close()

  def __enter__(self) -> "AF3Backend":
    return self

  def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
    self.close()


__all__ = ["AF3Backend", "AF3BackendError"]
