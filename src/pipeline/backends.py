"""Owner-local native sequence/refold adapters and durable prediction reuse.

Prediction evidence is cached before the native metric cleaner discards arrays.
Reference-dependent metrics are always evaluated on a private copy per parent.
"""
from __future__ import annotations

import base64
import binascii
import copy
from dataclasses import is_dataclass, replace
import json
from math import isfinite, prod
import os
from pathlib import Path
import sqlite3
import tempfile
from threading import RLock
from typing import Any, Mapping
import zlib

import numpy as np

_METHODS = {"esmfold2-fast", "af2_ig", "af3"}
_ARTIFACT_FIELDS = (
    "structure_path", "cif_path", "native_cif_path", "pdb_path",
    "confidence_path", "raw_confidence_path",
)
_IDENTITY_FIELDS = (
    "chain_id", "label_asym_id", "res_id", "ins_code", "res_name",
    "atom_name", "element", "mol_type", "atom_uid", "residue_uid", "token_uid",
    "label_entity_id", "auth_asym_id", "auth_seq_id", "auth_ins_code",
    "source_polymer_position", "source_break_before", "source_graph_complete",
)
_ARRAY_MARKER = "__pipeline_ndarray__"
_COMPRESSED_ARRAY_MARKER = "__pipeline_ndarray_zlib__"
_ARRAY_ALIAS_MARKER = "__pipeline_ndarray_alias__"


class NativeBackendError(RuntimeError):
    """Native execution did not produce complete reusable evidence."""
    fatal = True


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _encode(value):
    if isinstance(value, np.ndarray):
        return {_ARRAY_MARKER: value.tolist(), "dtype": value.dtype.str}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    raise TypeError(f"Unsupported native evidence value: {type(value).__name__}")


def _encode_evidence(value):
    """Compact raw arrays without changing canonical prediction-request JSON."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            if value.dtype.fields is not None:
                raise TypeError("Object-containing structured evidence arrays are unsupported")
            # Object string/label arrays stay JSON values, never raw pointers
            # or pickles. Explicit shape also preserves empty object arrays.
            return {_ARRAY_MARKER: [_encode_evidence(item) for item in value.flat],
                    "dtype": value.dtype.str, "shape": list(value.shape)}
        return {_COMPRESSED_ARRAY_MARKER: base64.b64encode(
                    zlib.compress(value.tobytes(order="C"), level=1)).decode("ascii"),
                "dtype": np.lib.format.dtype_to_descr(value.dtype),
                "shape": list(value.shape)}
    if isinstance(value, Mapping):
        encoded = {str(key): _encode_evidence(item) for key, item in value.items()}
        # Native and mapped confidence arrays often contain the exact same
        # compressed bytes. Keep one payload, without hashes or disk reads.
        for name, item in tuple(encoded.items()):
            logical = name.removeprefix("native_")
            if (logical != name and isinstance(item, dict)
                    and _COMPRESSED_ARRAY_MARKER in item
                    and item == encoded.get(logical)):
                encoded[name] = {_ARRAY_ALIAS_MARKER: logical}
        return encoded
    if isinstance(value, (tuple, list)):
        return [_encode_evidence(item) for item in value]
    return _encode(value)


def _evidence_shape(value):
    if not isinstance(value, list) or any(type(axis) is not int or axis < 0 for axis in value):
        raise ValueError("ndarray evidence shape must contain nonnegative integers")
    count = prod(value)
    if any(axis > np.iinfo(np.intp).max for axis in value) or count > np.iinfo(np.intp).max:
        raise ValueError("ndarray evidence shape exceeds platform index size")
    return tuple(value), count


def _decode_compressed_array(value):
    if set(value) != {_COMPRESSED_ARRAY_MARKER, "dtype", "shape"}:
        raise ValueError("compressed ndarray evidence has invalid fields")
    try:
        descriptor = value["dtype"]
        if not isinstance(descriptor, (str, list)):
            raise ValueError("invalid dtype descriptor")
        dtype = np.lib.format.descr_to_dtype(descriptor)
        if dtype.hasobject:
            raise ValueError("compressed ndarray evidence cannot contain object pointers")
        shape, count = _evidence_shape(value["shape"])
        expected = count * dtype.itemsize
        if expected >= np.iinfo(np.intp).max:
            raise ValueError("ndarray evidence byte count exceeds platform index size")
        encoded = value[_COMPRESSED_ARRAY_MARKER]
        if not isinstance(encoded, str):
            raise ValueError("compressed ndarray data must be base64 text")
        compressed = base64.b64decode(encoded, validate=True)
        decoder = zlib.decompressobj()
        # Bound decompression to the declared bytes plus one: oversized,
        # truncated, concatenated and trailing-data streams all fail closed.
        raw = decoder.decompress(compressed, expected + 1)
        if (len(raw) != expected or not decoder.eof
                or decoder.unused_data or decoder.unconsumed_tail):
            raise ValueError("compressed ndarray data does not match its declared shape/dtype")
        if dtype.itemsize == 0:
            return np.empty(shape, dtype=dtype)
        # Own writable bytes, retaining endian, NaN payload and record padding
        # exactly; native metric code may mutate a private prediction copy.
        return np.frombuffer(bytearray(raw), dtype=dtype, count=count).reshape(shape)
    except (binascii.Error, zlib.error, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid compressed ndarray evidence: {exc}") from exc


def _decode(value):
    if isinstance(value, dict):
        if _COMPRESSED_ARRAY_MARKER in value:
            return _decode_compressed_array(value)
        if set(value) == {_ARRAY_MARKER, "dtype"}:
            # Existing raw_predictions.json files retain their list encoding.
            return np.asarray(value[_ARRAY_MARKER], dtype=value["dtype"])
        if _ARRAY_MARKER in value:
            if set(value) != {_ARRAY_MARKER, "dtype", "shape"}:
                raise ValueError("ndarray evidence has invalid fields")
            dtype = np.dtype(value["dtype"])
            if dtype != np.dtype(object):
                raise ValueError("shaped JSON ndarray evidence requires object dtype")
            shape, count = _evidence_shape(value["shape"])
            items = value[_ARRAY_MARKER]
            if not isinstance(items, list) or len(items) != count:
                raise ValueError("object ndarray evidence does not match its declared shape")
            result = np.empty(shape, dtype=object)
            for index, item in enumerate(items):
                result.flat[index] = _decode(item)
            return result
        if _ARRAY_ALIAS_MARKER in value:
            raise ValueError("ndarray aliases require a sibling array")
        aliases = {key: item[_ARRAY_ALIAS_MARKER] for key, item in value.items()
                   if isinstance(item, dict) and set(item) == {_ARRAY_ALIAS_MARKER}}
        result = {key: None if key in aliases else _decode(item)
                  for key, item in value.items()}
        for key, target in aliases.items():
            array = result.get(target) if isinstance(target, str) else None
            if not isinstance(array, np.ndarray):
                raise ValueError("ndarray alias does not identify a sibling array")
            # Decoded fields remain independent and writable, as in old caches.
            result[key] = (np.frombuffer(bytearray(array.tobytes(order="C")), dtype=array.dtype)
                           .reshape(array.shape) if array.dtype.itemsize
                           else np.empty(array.shape, dtype=array.dtype))
        return result
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _canonical(value):
    return json.dumps(_encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            encoded = _encode_evidence(value)
            json.dump(encoded, stream, sort_keys=True, allow_nan=False)
            stream.flush()
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return encoded


def _stat(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _referenced_files(value, root):
    """Record actual MSA/template/explicit asset files, not just their names."""
    found = {}
    if isinstance(value, Mapping):
        for name, item in value.items():
            if isinstance(item, (str, Path)) and (
                "path" in str(name) or str(name) in {"checkpoint", "params_dir", "model_dir"}
            ):
                path = Path(item).expanduser()
                path = path if path.is_absolute() else Path(root) / path
                found[str(path.resolve())] = _stat(path)
            found.update(_referenced_files(item, root))
    elif isinstance(value, (tuple, list)):
        for item in value:
            found.update(_referenced_files(item, root))
    return found


def _design_arrays(input_record):
    atoms = _get(input_record, "atoms")
    design = tuple(_get(input_record, "design_chains", ()))
    if atoms is None or not design:
        raise ValueError("pipeline requires parsed atoms and declared design chains")
    labels = np.asarray(getattr(atoms, "label_asym_id", atoms.chain_id)).astype(str)
    modalities = np.asarray(atoms.mol_type).astype(str)
    designed = np.isin(labels, design)
    if not all(np.any(designed & (labels == chain)) for chain in design):
        raise ValueError("pipeline design chain is absent from parsed source atoms")
    kinds = set(modalities[designed])
    if not kinds or not kinds <= {"protein", "rna", "dna"}:
        raise ValueError("pipeline supports polymer design only; pure ligand design is unsupported")
    return atoms, design, labels, modalities, designed, kinds


def _occurrence_identity(input_record, *, coordinates=False):
    """Conservative source occurrence identity required by restored evidence."""
    atoms = _get(input_record, "atoms")
    if atoms is None:
        raise ValueError("pipeline prediction requires parsed source atoms")
    result = {name: _get(atoms, name) for name in _IDENTITY_FIELDS
              if _get(atoms, name) is not None}
    bonds = _get(atoms, "bonds")
    result["bonds"] = [] if bonds is None else bonds.as_array()
    result["design_chains"] = list(_get(input_record, "design_chains", ()))
    result["cyclic_chains"] = list(_get(input_record, "cyclic_chains", ()))
    result["cyclic_offset_chains"] = list(_get(input_record, "cyclic_offset_chains", ()))
    if coordinates:
        result["initial_guess_coordinates"] = np.asarray(atoms.coord)
        valid = _get(atoms, "coord_valid")
        if valid is not None:
            result["coord_valid"] = valid
    return result


def _restoration_identity(input_record):
    """Stable occurrences; obsolete mutable side chains are not model inputs.

    Free design residues are represented by source locators and modality.
    Their replacement chemistry comes from the native sequence/modification
    payload. Context occurrence inventories and inter-residue links remain
    strict so a restored CIF can be shared without relocating any residue.
    """
    atoms, design, labels, modalities, designed, _ = _design_arrays(input_record)
    rows = {}
    for index in range(len(atoms)):
        key = (str(labels[index]), int(atoms.res_id[index]), str(atoms.ins_code[index]))
        rows.setdefault(key, []).append(index)
    groups = []
    locators = {}
    locator_fields = ("label_asym_id", "label_seq_id", "auth_asym_id", "auth_seq_id", "auth_ins_code",
                      )
    for key, indices in rows.items():
        first = indices[0]
        residue = {"chain_id": key[0], "res_id": key[1], "ins_code": key[2],
                   "modality": str(modalities[first]), "is_design": bool(designed[first])}
        for name in locator_fields:
            values = getattr(atoms, name, None)
            if values is not None:
                residue[name] = values[first]
        if not designed[first] and str(modalities[first]) not in {"protein","rna","dna"}:
            residue["context_atoms"] = [
                (str(atoms.res_name[i]), str(atoms.atom_name[i]), str(atoms.element[i]))
                for i in indices
            ]
        groups.append(residue)
        for index in indices:
            locators[index] = (*key, str(atoms.atom_name[index]))
    bonds = getattr(atoms, "bonds", None)
    edges = []
    for a, b, order in ([] if bonds is None else bonds.as_array()):
        a, b = int(a), int(b)
        # Ordinary internal mutable-AA topology is superseded by the actual
        # requested identity. Cross-residue and context graphs remain facts.
        if (locators[a][:3] == locators[b][:3] and
                str(modalities[a]) in {"protein","rna","dna"}):
            continue
        edges.append((*sorted((locators[a], locators[b])), int(order)))
    return {"residues": groups, "edges": sorted(edges), "design_chains": list(design),
            "cyclic_chains": list(_get(input_record, "cyclic_chains", ())),
            "cyclic_offset_chains": list(_get(input_record, "cyclic_offset_chains", ()))}


def _model_payload_identity(payload, method):
    if method == "af3":
        # Source atom ledgers/old design sequences are restoration carriers,
        # not AF3 conditioning. Stable occurrence identity is kept separately.
        fields = ("protocol", "chains", "rng_seeds", "bonded_atom_pairs", "user_ccd",
                  "cyclic_chains", "cyclic_lengths", "msa_context",
                  "source_to_native_chain", "native_to_source_chain")
        return {protocol: {name: value[name] for name in fields if name in value}
                for protocol, value in payload.items()}
    if method == "esmfold2-fast":
        return {name: value for name, value in payload.items()
                if name not in {"sample_id", "variant_id", "name", "partidesign_receptor_segments"}}
    return payload


def _alias_input(input_record):
    if isinstance(input_record, Mapping):
        result = dict(input_record)
        result["sample_id"] = "request"
        return result
    if is_dataclass(input_record):
        return replace(input_record, sample_id="request")
    result = copy.copy(input_record)
    result.sample_id = "request"
    return result


def _alias_variant(variant):
    result = copy.deepcopy(dict(variant))
    result.update(seq_index=0, variant_id="seq0", sample_id="request", name="request",
                  source="gt", sequence_source="gt", sequence_branch="self", source_index=0)
    # These are provenance/input-pose aliases. Native chemistry settings,
    # modifications, cyclic intent, MSA and templates remain untouched.
    for name in ("id", "input_path", "path", "structure_path", "output_dir"):
        result.pop(name, None)
    return result


def _rebind_af3_sources(input_record, variant, config, predictions, repo_root):
    """Restore each parent's source evidence without changing shared raw data."""
    from src.design.rewards.af3_backend import prepare_af3_inputs
    prepared = prepare_af3_inputs(_alias_input(input_record), _alias_variant(variant),
                                  config["af3"], repo_root)
    atoms = input_record.atoms
    labels = np.asarray(getattr(atoms, "label_asym_id", atoms.chain_id)).astype(str)
    uid = getattr(atoms, "atom_uid", np.arange(len(atoms)))
    current_atoms = {(str(labels[i]), int(atoms.res_id[i]), str(atoms.ins_code[i]),
                      str(atoms.atom_name[i])): int(uid[i]) for i in range(len(atoms))}
    fields = ("source_atom_ledger", "source_inter_residue_bonds", "source_polymer_sequences",
              "source_residue_map", "source_to_native_chain", "native_to_source_chain",
              "native_component_map")
    for prediction in predictions:
        payload = prepared[prediction["protocol"]]
        for field in fields:
            if field in payload:
                prediction[field] = copy.deepcopy(payload[field])
        if "native_component_map" in payload:
            prediction["source_component_map"] = {
                str(native): str(source) for source, native in payload["native_component_map"].items()
            }
        for field in ("product_atom_ledger", "restored_product_atom_ledger", "native_product_atom_ledger"):
            for item in prediction.get(field, ()):
                if field == "native_product_atom_ledger" and "source_chain_id" not in item:
                    continue
                chain = item.get("source_chain_id", item.get("chain_id"))
                position = item.get("source_res_id", item.get("res_id"))
                insertion = item.get("source_ins_code", item.get("ins_code", ""))
                if position is not None:
                    item["source_atom_uid"] = current_atoms.get(
                        (str(chain), int(position), str(insertion), str(item.get("atom_name"))))


def _validation_config(config, method):
    from src.validate.config import DEFAULTS, config_for_method, merge
    if method not in _METHODS:
        raise ValueError(f"unsupported pipeline prediction method: {method!r}")
    defaults = copy.deepcopy(DEFAULTS)
    defaults["_sequence_sources_explicit"] = True
    result = config_for_method(merge(defaults, copy.deepcopy(dict(config))), method)
    if result["scorer"].get("enabled"):
        raise ValueError("pipeline native refolding does not run an extra PartiScore scorer")
    return result


def _score_fraction(value, *, native_percent=False):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 0:
        raise ValueError("design confidence must be scalar")
    result = float(array)
    if native_percent or result > 1.0:
        result /= 100.0
    if not isfinite(result) or not 0 <= result <= 1:
        raise ValueError("design confidence must be finite in [0,1]")
    return result


def _confidence_field(prediction, *names):
    for container in (prediction, prediction.get("confidence", {})):
        if isinstance(container, Mapping):
            for name in names:
                if container.get(name) is not None:
                    return container[name]
    return None


def design_score(input_record, method, raw_prediction):
    """Design-only pLDDT, normalized to [0,1], with native weighting."""
    design = tuple(map(str, _get(input_record, "design_chains", ())))
    if not design:
        raise ValueError("prediction has no declared design chain")
    if method == "af2_ig":
        confidence = raw_prediction.get("confidence", {})
        if "plddt_normalized" in confidence:
            return _score_fraction(confidence["plddt_normalized"])
        if "plddt" in confidence:
            # Native AF2 extract_confidence explicitly reduces the design slice.
            return _score_fraction(confidence["plddt"], native_percent=True)
        raise ValueError("AF2 prediction lacks native design-chain pLDDT")
    if method == "af3":
        from src.design.rewards.af3_runner import _chain_plddt
        values = [_chain_plddt(raw_prediction, chain) for chain in design]
        if any(value is None for value in values):
            raise ValueError("AF3 prediction lacks design-chain atomic confidence")
        return float(sum(score * count for score, count in values)
                     / sum(count for _, count in values) / 100.0)
    if method != "esmfold2-fast":
        raise ValueError(f"unsupported pipeline score method: {method!r}")
    arrays = raw_prediction.get("_confidence_arrays", {})
    labels = arrays.get("token_chain_ids")
    plddt = arrays.get("plddt", arrays.get("plddt_ca"))
    if labels is not None and plddt is not None:
        labels, plddt = np.asarray(labels).astype(str), np.asarray(plddt, dtype=np.float64)
        if labels.shape != plddt.shape or labels.ndim != 1:
            raise ValueError("ESMFold2 token confidence/chain labels differ")
        mask = np.isin(labels, design)
        attention = arrays.get("token_attention_mask")
        if attention is not None:
            if np.asarray(attention).shape != mask.shape:
                raise ValueError("ESMFold2 token attention mask differs from confidence")
            mask &= np.asarray(attention, dtype=bool)
        if not all(np.any(mask & (labels == chain)) for chain in design):
            raise ValueError("ESMFold2 prediction lacks a design-chain token confidence")
        selected = plddt[mask]
        if not np.isfinite(selected).all() or np.any(selected < 0):
            raise ValueError("ESMFold2 design confidence is invalid")
        if np.max(selected) > 1:
            selected = selected / 100.0
        if np.any(selected > 1):
            raise ValueError("ESMFold2 design confidence exceeds one")
        return float(selected.mean())
    values = _confidence_field(raw_prediction, "per_chain_plddt")
    counts = _confidence_field(raw_prediction, "per_chain_token_count", "per_chain_token_counts")
    if not isinstance(values, Mapping) or not isinstance(counts, Mapping):
        raise ValueError("ESMFold2 prediction lacks design-chain token confidence/counts")
    total = weight = 0.0
    for chain in design:
        if chain not in values or chain not in counts:
            raise ValueError(f"ESMFold2 prediction lacks design confidence for {chain!r}")
        count = counts[chain]
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("ESMFold2 design token count must be a positive integer")
        total += _score_fraction(values[chain], native_percent=True) * int(count)
        weight += int(count)
    return total / weight


class NativeOwner:
    """One resource owner; native workers stay resident across tasks/rounds."""

    def __init__(self, repo_root, cache_dir, device="cuda:0"):
        self.repo = Path(repo_root).expanduser().resolve()
        self.cache = Path(cache_dir).expanduser().resolve()
        self.device = str(device)
        self._workers = {}
        self._generators = {}
        self._msa_caches = {}
        self._asset_cache = {}
        self._request_ids = {}
        self._registry = None
        self._registry_lock = RLock()
        self._validated_predictions = set()
        self._last_prediction = None
        self._execution = RLock()
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError("pipeline native owner is closed")

    def _assets(self, method, config):
        """Inspect only the native checkpoint/chemistry files, once per owner."""
        from partidesign.paths import af3_asset, community_asset
        if method == "esmfold2-fast":
            default_models = community_asset("esmfold2", self.repo).parent
            root = Path(os.environ.get("PIPETOOLS_MODEL_ROOT", default_models)).expanduser().resolve() / "esmfold2"
            key = _canonical([method, str(root)])
            if key not in self._asset_cache:
                files = []
                for directory in (root / "checkpoint/fast", root / "checkpoint/esmc-6b"):
                    index = directory / "model.safetensors.index.json"
                    if index.is_file():
                        manifest = json.loads(index.read_text())
                        files.extend(directory / name for name in sorted(set(manifest["weight_map"].values())))
                        files.append(index)
                    else:
                        weight = directory / "model.safetensors"
                        files.append(weight if weight.is_file() else directory / "pytorch_model.bin")
                    files.append(directory / "config.json")
                files.extend(root / "checkpoint/esmc-6b" / name for name in
                             ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"))
                files.extend((root / "common/ccd.pkl", root / "common/ccd_bonds.sqlite3"))
                self._asset_cache[key] = [_stat(path) for path in files]
        elif method == "af2_ig":
            section = config["af2"]
            root = Path(section.get("params_dir") or community_asset("colabdesign", self.repo) / "checkpoint/af2")
            root = root if root.is_absolute() else self.repo / root
            key = _canonical([method, str(root), section["models"]])
            if key not in self._asset_cache:
                self._asset_cache[key] = [_stat(root / f"params_model_{model}_ptm.npz")
                                          for model in section["models"]]
        else:
            section = config["af3"]
            root = af3_asset("af3", self.repo)
            checkpoint = Path(section.get("checkpoint") or root / "checkpoints/af3.bin.zst")
            checkpoint = checkpoint if checkpoint.is_absolute() else self.repo / checkpoint
            key = _canonical([method, str(checkpoint), str(root)])
            if key not in self._asset_cache:
                weight = checkpoint / "af3.bin.zst" if checkpoint.is_dir() else checkpoint
                files = [weight, *(root / "ccd" / name for name in
                                    ("ccd.pickle", "chemical_component_sets.pickle", "components.cif"))]
                self._asset_cache[key] = [_stat(path) for path in files]
        return self._asset_cache[key]

    def _worker(self, method, config, assets):
        from src.validate.runner import ValidationWorker
        section = {"af2_ig": "af2", "esmfold2-fast": "esmfold2", "af3": "af3"}[method]
        key = _canonical([method, config[section], assets])
        if key not in self._workers:
            self._workers[key] = ValidationWorker(self.repo, method, config, [], {}, self.device)
        return self._workers[key]

    def _generator(self, config):
        from src.validate.sequences import SequenceGenerator
        sequence = copy.deepcopy(config.get("sequence", {}))
        sequence.pop("seed", None)
        sequence.pop("sources", None)
        key = _canonical(sequence)
        if key not in self._generators:
            self._generators[key] = SequenceGenerator(self.repo, config, self.device)
        return self._generators[key]

    def _payload(self, input_record, variant, method, config):
        if method == "esmfold2-fast":
            from src.validate.esmfold2 import _build_payload
            return _build_payload(input_record, variant), None
        if method == "af2_ig":
            from src.validate.af2 import _source_atoms
            override = variant.get("private_pdb", variant.get("pdb_path"))
            source_atoms = _source_atoms(input_record, variant)
            if override is None:
                return {"sequences": variant.get("sequences", {}),
                        "initial_guess": _occurrence_identity(input_record, coordinates=True),
                        "private_pdb": None, "source_atoms": source_atoms.coord}, None
            from io import StringIO
            from src.validate.af2 import _design_chain, _context_chains, read_structure
            from src.design.rewards.af2_mapping import (
                af2_source_ledger, af2_preparation_ledger, normalize_private_override, private_af2_pdb,
            )
            design = _design_chain(input_record)
            ledger = af2_source_ledger(source_atoms, [*_context_chains(input_record, design), design], design)
            effective = normalize_private_override(source_atoms, read_structure(override), ledger)
            template = StringIO()
            private_af2_pdb(effective, ledger=ledger).write(template)
            return {"sequences": variant.get("sequences", {}),
                    "initial_guess": _occurrence_identity(input_record),
                    "private_pdb": {"template": template.getvalue(),
                                    "source_ledger": af2_preparation_ledger(ledger)}}, None
        from src.design.rewards.af3_backend import prepare_af3_inputs
        from src.design.rewards.af3_input import attach_msa_references, msa_queries
        from src.validate.af3_msa import AF3MSACache, validate_msa_settings
        payloads = prepare_af3_inputs(input_record, variant, config["af3"], self.repo)
        settings = validate_msa_settings(config["af3"].get("msa"))
        if settings["mode"] != "off":
            key = _canonical(settings)
            if key not in self._msa_caches:
                self._msa_caches[key] = AF3MSACache(self.repo, settings)
            cache = self._msa_caches[key]
            queries = {(query["modality"], query["sequence"])
                       for payload in payloads.values()
                       for query in msa_queries(payload, input_record.design_chains)}
            references = [cache.ensure(modality, sequence) for modality, sequence in sorted(queries)]
            payloads = {protocol: attach_msa_references(payload, input_record.design_chains, references)
                        for protocol, payload in payloads.items()}
        return payloads, payloads

    def generate(self, input_record, round_config, *, task_name, round_number,
                 parent_id, seed, work_dir):
        from src.validate.sequences import _candidate_adapter
        from src.design.rewards.esmfold2 import _validate_na_candidate
        from src.design.rewards.sequence import resolve_reward_sequence_source
        self._check_open()
        count = round_config.get("mpnn_count", 1 if round_number == 1 else 4)
        if type(count) is not int or count < 0:
            raise ValueError("mpnn_count must be a non-negative integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("MPNN seed must be a non-negative integer")
        config = _validation_config(round_config.get("validation", {}), round_config.get("method", "esmfold2-fast"))
        atoms, design, labels, modalities, designed, kinds = _design_arrays(input_record)
        include_gt = round_config.get("include_gt", True)
        sources = [{"source": "gt", "count": 1}] if include_gt else []
        passthrough = None
        task_index = round_config.get("_task_index", 0)
        parent_index = round_config.get("_parent_index", 0)
        stable_seed = (seed + task_index * 10000019 + int(round_number) * 100003
                       + parent_index * 1009) % (2**31 - 1)
        if count:
            candidate = _candidate_adapter(input_record, "pipeline-route")
            requested = round_config.get("sequence_source", "auto")
            if requested == "auto":
                if kinds & {"rna", "dna"}:
                    if len(design) != 1 or len(kinds) != 1:
                        passthrough = "na_mpnn_requires_exactly_one_design_na_chain"
                        native_source = "design"
                    else:
                        _validate_na_candidate(candidate)
                        native_source = "na_mpnn"
                else:
                    native_source = resolve_reward_sequence_source(candidate, "auto", reward_method="esmfold2")
                public = {"proteinmpnn": "soluble_mpnn", "ligandmpnn": "ligand_mpnn",
                          "na_mpnn": "na_mpnn", "design": None}.get(native_source)
                if native_source not in {"proteinmpnn", "ligandmpnn", "na_mpnn", "design"}:
                    raise ValueError(f"reward auto selected an unsupported pipeline source: {native_source}")
            else:
                public = requested
                native_source = requested
                if requested == "na_mpnn":
                    _validate_na_candidate(candidate)
            if public is None:
                passthrough = passthrough or "reward_auto_gt_passthrough"
            else:
                backend = config.get("sequence", {})
                settings = copy.deepcopy(round_config.get("sequence_settings", {}))
                fi = backend.get(public, backend.get(native_source, {}))
                if isinstance(fi, Mapping) and "fixed_interface" in fi:
                    settings.setdefault("fixed_interface", copy.deepcopy(fi["fixed_interface"]))
                ligand_context = bool(np.any((modalities == "ligand") & ~designed))
                if public == "na_mpnn" and ligand_context:
                    supplied = settings.get("fixed_interface")
                    if supplied is not None and not supplied.get("enabled", False):
                        raise ValueError("nucleic-acid ligand binding requires fixed_interface.enabled=true")
                    target_chains = None if supplied is None else supplied.get("target_chains")
                    ligand_chains = set(labels[(modalities == "ligand") & ~designed])
                    if target_chains is not None and not ligand_chains.issubset(set(target_chains)):
                        raise ValueError("nucleic-acid fixed interface must include every ligand context chain")
                    settings["fixed_interface"] = {"enabled": True, **(supplied or {})}
                settings["seed"] = stable_seed
                sources.append({"source": public, "count": count,
                                **({"label": "self_fi"} if public == "na_mpnn" and ligand_context else {}),
                                "settings": settings})
        if not sources:
            raise ValueError("The selected sequence backend cannot redesign this candidate and GT is disabled")
        with self._execution:
            self._check_open()
            variants = list(self._generator(config).generate(input_record, sources, Path(work_dir)))
        for variant in variants:
            if variant.get("status") not in {"complete", "ok"}:
                raise NativeBackendError(variant.get("error") or "Native MPNN did not produce a sequence")
            if passthrough is not None:
                variant.setdefault("settings", {}).update(
                    backend="identity_passthrough", sampled=False, reason=passthrough,
                    requested_mpnn_count=count)
            variant["pipeline_mpnn_seed"] = stable_seed
        return variants

    def _request_id(self, request_text):
        """SQLite assigns stable names to exact actual-input request text."""
        with self._registry_lock:
            if request_text in self._request_ids:
                return self._request_ids[request_text]
            if self._registry is None:
                self.cache.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(self.cache / "requests.sqlite3", timeout=60,
                                             check_same_thread=False)
                try:
                    connection.execute("PRAGMA journal_mode=WAL")
                    connection.execute("PRAGMA synchronous=NORMAL")
                    connection.execute("CREATE TABLE IF NOT EXISTS requests "
                                       "(id INTEGER PRIMARY KEY, request TEXT NOT NULL UNIQUE)")
                    connection.commit()
                except Exception:
                    connection.close()
                    raise
                self._registry = connection
            self._registry.execute("INSERT OR IGNORE INTO requests(request) VALUES (?)",
                                   (request_text,))
            identifier = self._registry.execute("SELECT id FROM requests WHERE request=?",
                                                (request_text,)).fetchone()[0]
            self._registry.commit()
            self._request_ids[request_text] = identifier
            return identifier

    def _memory_prediction(self, request_key):
        cached = self._last_prediction
        if cached is None or cached[0] != request_key:
            return None
        # Only the primary structure is needed immediately by metrics/writing.
        if all(Path(value["structure_path"]).is_file() for value in cached[1]):
            return copy.deepcopy(cached[1])
        self._last_prediction = None
        self._validated_predictions.discard(request_key)
        return None

    def _remember_prediction(self, request_key, predictions):
        self._validated_predictions.add(request_key)
        # One raw result avoids retaining every large PAE matrix in RAM.
        self._last_prediction = (request_key, predictions)
        return copy.deepcopy(predictions)

    def predict(self, input_record, variant, method, validation_config, *, task_name):
        from src.validate.storage import ensure_directory, locked
        from src.validate.planning import expected_keys, prediction_metadata_valid
        self._check_open()
        config = _validation_config(validation_config, method)
        _design_arrays(input_record)
        if method=="esmfold2-fast" and set(getattr(input_record,"cyclic_offset_chains",()))-set(getattr(input_record,"cyclic_chains",())):
            raise ValueError("ESMFold2 uses explicit ring bonds; offset-only inputs require af2_ig or af3")
        canonical_input, canonical_variant = _alias_input(input_record), _alias_variant(variant)
        payload, prepared = self._payload(canonical_input, canonical_variant, method, config)
        assets = self._assets(method, config)
        section = {"af2_ig": "af2", "esmfold2-fast": "esmfold2", "af3": "af3"}[method]
        conditioning = _model_payload_identity(payload, method)
        request = {"version": 2, "task": str(task_name), "method": method,
                   "config": config[section], "weights": assets, "input": conditioning,
                   "source_occurrences": (_occurrence_identity(canonical_input)
                                          if method == "af2_ig" else _restoration_identity(canonical_input)),
                   "referenced_files": _referenced_files([conditioning, config[section]], self.repo)}
        request_text = _canonical(request)
        request_key = f"request_{self._request_id(request_text)}"
        cached = self._memory_prediction(request_key)
        if cached is not None:
            return request_key, cached, True
        root = ensure_directory(self.cache / request_key)
        state_path = root / "raw_predictions.json"
        with locked(root / "prediction.lock"):
            self._check_open()
            cached = self._memory_prediction(request_key)
            if cached is not None:
                return request_key, cached, True
            if state_path.is_file():
                state = json.loads(state_path.read_text())
                if state.get("request_key") != request_key:
                    raise NativeBackendError("prediction cache request identity is corrupt")
                if request_key in self._validated_predictions:
                    valid = all(Path(value["structure_path"]).is_file()
                                for value in state["predictions"])
                else:
                    valid = all(Path(path).is_file() for path in state.get("artifacts", ()))
                if valid:
                    return request_key, self._remember_prediction(
                        request_key, _decode(state["predictions"])), True
            output = ensure_directory(root / "native")
            with self._execution:
                self._check_open()
                worker = self._worker(method, config, assets)
                worker.log_dir = ensure_directory(self.cache.parent / "logs")
                backend = worker._models()
                extra = {} if prepared is None else {"prepared_inputs": prepared}
                predictions = list(backend.predict(canonical_input, canonical_variant, output,
                                                   completed_keys=set(), **extra))
            expected = expected_keys(method, config)
            received = [value.get("key") for value in predictions]
            if len(received) != len(set(received)) or set(received) != set(expected):
                raise NativeBackendError("native backend returned incomplete or duplicate prediction slots")
            for prediction in predictions:
                if prediction.get("status") != "complete":
                    raise NativeBackendError(prediction.get("error") or "native prediction failed")
                if not prediction_metadata_valid(prediction["key"], prediction, method):
                    raise NativeBackendError("native backend returned incompatible prediction metadata")
            artifacts = set()
            for prediction in predictions:
                for field in _ARTIFACT_FIELDS:
                    value = prediction.get(field)
                    if value:
                        path = Path(value).resolve()
                        if not path.is_file():
                            raise NativeBackendError(f"native prediction artifact is missing: {path}")
                        artifacts.add(str(path))
                if not prediction.get("structure_path"):
                    raise NativeBackendError("native prediction has no restored structure")
            saved = _atomic_json(state_path, {"request_key": request_key, "predictions": predictions,
                                              "artifacts": sorted(artifacts)})
            # Decode the exact committed representation once, without another
            # compression or a large ndarray.tolist() allocation.
            return request_key, self._remember_prediction(
                request_key, _decode(saved["predictions"])), False

    def evaluate(self, input_record, variant, method, validation_config, raw_predictions):
        from src.validate.planning import expected_keys
        from src.validate.runner import _compact_sequence
        from src.design.utils.reader import read_structure
        self._check_open()
        metric_config = dict(validation_config)
        motif_geometry = metric_config.pop("_pipeline_motif_geometry", True)
        config = _validation_config(metric_config, method)
        metric_variant = variant if motif_geometry else {
            **variant, "_pipeline_motif_geometry": False,
        }
        if (motif_geometry and getattr(input_record, "motif_positions", {})
                and not variant.get("motif_reference_path")
                and not input_record.row.get("motif_reference_path")):
            # One parent evaluation can contain many conformers or AF2 pairs.
            # The original source mapping is identical for those predictions.
            metric_variant = {**variant, "_pipeline_motif_reference_cache": {}}
        # Parsed native structures are ephemeral. Keep an already-parsed
        # object by reference, while isolating mutable confidence/metric data.
        parsed_by_key = {value["key"]: value["_parsed_atoms"] for value in raw_predictions
                         if value.get("_parsed_atoms") is not None}
        predictions = [copy.deepcopy({key: value for key, value in prediction.items()
                                      if key != "_parsed_atoms"})
                       for prediction in raw_predictions]
        if method == "af3":
            _rebind_af3_sources(input_record, variant, config, predictions, self.repo)
        with self._execution:
            worker = self._worker(method, config, self._assets(method, config))
            worker.log_dir = self.cache.parent / "logs"
            for prediction in predictions:
                if method == "af3":
                    native_ranking = prediction.get("metadata", {}).get("ranking_score")
                    if native_ranking is not None:
                        prediction.setdefault("confidence", {})["ranking_score"] = float(native_ranking)
                # Save the design-only rank before the native cleaner removes
                # arrays and source maps needed for the weighted reduction.
                prediction["pipeline_design_score"] = design_score(input_record, method, prediction)
                key = prediction["key"]
                if key not in parsed_by_key:
                    parsed_by_key[key] = read_structure(Path(prediction["structure_path"]))
                prediction["_parsed_atoms"] = parsed_by_key[key]
                try:
                    worker._metrics(input_record, metric_variant, prediction)
                finally:
                    prediction.pop("_parsed_atoms", None)
                if prediction.get("metrics_status") != "complete":
                    raise NativeBackendError(prediction.get("error") or
                                             "native metric evaluation failed without complete evidence")
        by_key = {prediction["key"]: prediction for prediction in predictions}
        paired, paired_reasons = {}, {}
        if method == "af2_ig":
            from src.validate.metrics import evaluate_pair
            for model in config["af2"]["models"]:
                for sample in range(config["af2"]["samples"]):
                    suffix = f"model{model}_sample{sample}"
                    a = by_key.get(f"complex_{suffix}", {})
                    b = by_key.get(f"monomer_{suffix}", {})
                    if a.get("status") == b.get("status") == "complete":
                        pair = evaluate_pair(input_record, variant,
                                             parsed_by_key[a["key"]], parsed_by_key[b["key"]])
                        paired[suffix] = pair["metrics"]
                        paired_reasons[suffix] = pair.get("reasons", {})
        expected = expected_keys(method, config)
        complete = all(by_key.get(key, {}).get("status") == "complete"
                       and by_key.get(key, {}).get("metrics_status") == "complete" for key in expected)
        complexes = [p for p in predictions if p.get("protocol") == "complex"]
        best = max(complexes or predictions, key=lambda p: p["pipeline_design_score"], default=None)
        return {"status": "complete" if complete else "partial", "method": method,
                "sequence": _compact_sequence(dict(variant)), "expected_predictions": expected,
                "predictions": by_key, "paired_metrics": paired, "paired_reasons": paired_reasons,
                "metrics": {} if best is None else copy.deepcopy(best.get("metrics", {})),
                "design_score": None if best is None else best["pipeline_design_score"],
                "_parsed_atoms": parsed_by_key}

    def close(self):
        with self._execution:
            if self._closed:
                return
            self._closed = True
            errors = []
            for owner in [*self._workers.values(), *self._generators.values()]:
                try:
                    owner.close()
                except Exception as exc:
                    errors.append(exc)
            self._workers.clear()
            self._generators.clear()
            self._msa_caches.clear()
            self._asset_cache.clear()
            self._request_ids.clear()
            self._validated_predictions.clear()
            self._last_prediction = None
            with self._registry_lock:
                if self._registry is not None:
                    self._registry.close()
                    self._registry = None
            if errors:
                raise errors[0]


__all__ = ["NativeOwner", "NativeBackendError", "design_score"]
