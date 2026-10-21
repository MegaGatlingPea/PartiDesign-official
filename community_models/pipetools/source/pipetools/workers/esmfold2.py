from __future__ import annotations

import argparse
import gc

import importlib
import inspect
import io
import json
import os
import tempfile
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..ccd_bonds import CCDBondCacheReader
from ..esmfold2_evidence import (
    atomic_number,
    decode_atom_name_rows,
    encode_atom_name_rows,
)
from ..output_identity import rewrite_output_entity_identity


_BATCH_SCHEMA_VERSION = 1
_INPUT_SCHEMA_VERSION = 1
_UNAVAILABLE_CONFIDENCE = (
    {
        "path": "has_clash",
        "reason": "ESMFold2 does not emit a native clash flag",
    },
)


class _SystemicWorkerError(RuntimeError):
    """Model/runtime failure that must terminate the persistent worker."""


def _is_cuda_oom(torch: Any, error: BaseException) -> bool:
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    return oom_type is not None and isinstance(error, oom_type)


@dataclass(slots=True)
class _ModelBundle:
    torch: Any
    np: Any
    model: Any
    builder: Any
    types: Any
    build_chains_from_input: Any
    seed_context: Any
    lm_dropout_context: Any
    ccd_bond_reader: CCDBondCacheReader

    # The model-owned RDKit CCD dictionary is the only atom-inventory authority
    # used for modified-polymer occurrence validation.  Keep it beside the
    # builder so request conversion never reaches back into PartiDesign data.
    ccd_molecules: dict[str, Any] | None = None
    context_embedding_cache: dict[str, Any] = field(default_factory=dict)
    context_cache_signature: str | None = None
    context_cache_hits: int = 0
    context_cache_misses: int = 0
    optimization: Any = None


def run_batch(request_json: Path) -> None:
    request_path = _safe_existing_file(request_json, "ESMFold2 batch request")
    payload = json.loads(request_path.read_text())
    validated = _validate_batch_payload(payload)
    result_json = validated["result_json"]
    if result_json.exists() or result_json.is_symlink():
        raise RuntimeError(f"ESMFold2 batch result target is unsafe: {result_json}")

    bundle = _load_model_bundle(
        validated["mode"],
        validated["paths"],
        backend=validated["config"]["esmc_attention_backend"],
        optimization_config=validated["config"].get("optimization", {"mode": "off"}),

    )
    results: list[dict[str, Any]] = []
    for request_index, request in enumerate(validated["requests"]):
        try:
            artifacts = _execute_request(
                bundle,
                request,
                validated["config"],
                validated["mode"],
            )
            result = {
                "request_id": request["request_id"],
                "status": "ok",
                "error": None,
                "artifacts": artifacts,
            }
        except Exception as exc:
            if _systemic_worker_failure(exc):
                raise
            if getattr(bundle, "optimization", None) is not None:
                bundle.optimization.error()
            message = _error_text(exc)
            output_dir = request["output_dir"]
            if not output_dir.exists() and not output_dir.is_symlink():
                output_dir.mkdir(parents=True, exist_ok=False)
            if output_dir.is_dir() and not output_dir.is_symlink():
                _atomic_json(
                    output_dir / "error.json",
                    {
                        "request_id": request["request_id"],
                        "error": message,
                        "traceback": traceback.format_exc()[-16000:],
                    },
                )
            result = {
                "request_id": request["request_id"],
                "status": "error",
                "error": message,
                "artifacts": [],
            }
        finally:
            _release_request_memory(bundle)
        results.append(result)

    response = {
        "schema_version": _BATCH_SCHEMA_VERSION,
        "batch_id": validated["batch_id"],
        "mode": validated["mode"],
        "model_load_count": 1,
        "esmc_load_count": 1,
        "ccd_load_count": 1,
        "ccd_bond_cache_load_count": 1,
        "request_count": len(validated["requests"]),
        "results": results,
    }
    if getattr(bundle, "optimization", None) is not None:
        response["optimization"] = bundle.optimization.evidence()
    _atomic_json(result_json, response)
    print(
        "ESMFold2 persistent worker: "
        f"mode={validated['mode']} model_load_count=1 esmc_load_count=1 "
        "ccd_load_count=1 ccd_bond_cache_load_count=1 "
        f"request_count={len(validated['requests'])}",
        flush=True,
    )
    _close_model_bundle(bundle)


def _systemic_worker_failure(error: Exception) -> bool:
    """Classify only explicitly typed model/runtime failures as run-fatal.

    Candidate-controlled CCD IDs and atom names are embedded in validation
    messages.  Substring matching words such as ``cuda`` would therefore let
    an ordinary candidate error abort every other request in the batch.
    """

    if isinstance(
        error,
        (
            _SystemicWorkerError,
            AssertionError,
            ImportError,
            ModuleNotFoundError,
            MemoryError,
            NotImplementedError,
        ),
    ):
        return True
    return False


def _release_request_memory(bundle: Any) -> None:
    """Release request-local tensors while retaining the loaded model bundle."""

    gc.collect()
    torch = getattr(bundle, "torch", None)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and hasattr(cuda, "empty_cache"):
        cuda.empty_cache()
    if getattr(bundle, "optimization", None) is not None:
        bundle.optimization.after_request()


def _close_model_bundle(bundle: Any) -> None:
    if getattr(bundle, "optimization", None) is not None:
        bundle.optimization.clear("owner_close")
    reader = getattr(bundle, "ccd_bond_reader", None)
    if reader is not None:
        close = getattr(reader, "close", None)
        if not callable(close):
            raise RuntimeError("ESMFold2 CCD bond reader cannot be closed")
        close()


def _load_model_bundle(
    mode: str,
    paths: dict[str, Path],
    *,
    backend: str,
    optimization_config: dict[str, Any] | None = None,

) -> _ModelBundle:
    initialization_started = time.perf_counter()
    visible_devices = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if len(visible_devices) != 1:
        raise RuntimeError(
            "ESMFold2 worker must own exactly one CUDA_VISIBLE_DEVICES entry"
        )
    ccd_bond_reader = CCDBondCacheReader(
        paths["ccd_bonds"], source_cif=paths["components_cif"]
    )
    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")
    # Reward ranking must not depend on allocator history or request order.
    # Set the strongest deterministic PyTorch contract before creating CUDA
    # model state; custom fused kernels are still covered by the real repeated
    # request gate below.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    modeling = importlib.import_module(
        "transformers.models.esmfold2.modeling_esmfold2"
    )
    esmfold2 = importlib.import_module("esm.models.esmfold2")
    processor = importlib.import_module("esm.models.esmfold2.processor")
    prepare_input = importlib.import_module("esm.models.esmfold2.prepare_input")
    conformers = importlib.import_module("esm.models.esmfold2.conformers")

    if not torch.cuda.is_available():
        raise RuntimeError("ESMFold2 requires a CUDA GPU")
    model = modeling.ESMFold2Model.from_pretrained(
        str(paths["fold_checkpoint"]),
        load_esmc=False,
        local_files_only=True,
    )
    model = model.cuda().eval()
    esmc = _load_esmc(model, paths["esmc_checkpoint"], torch, backend)
    # The upstream ESMFold2 implementation exposes this as its production
    # Triton path.  It must not be combined with torch.compile (the upstream
    # API documents them as alternative backends).
    model.set_kernel_backend("fused")
    fused_modules = sum(
        getattr(module, "_kernel_backend", None) == "fused"
        for module in model.modules()
    )
    if fused_modules < 1:
        raise RuntimeError("ESMFold2 failed to activate fused Triton kernels")
    torch.cuda.synchronize()
    msa_enabled = bool(model.config.msa_encoder.enabled)
    if msa_enabled != (mode == "standard"):
        raise RuntimeError(
            f"ESMFold2 {mode} checkpoint has msa_encoder.enabled={msa_enabled}"
        )
    ccd_molecules = conformers.load_ccd(paths["ccd"].parent)
    if not isinstance(ccd_molecules, dict) or not ccd_molecules:
        raise RuntimeError("ESMFold2 CCD molecule cache is empty")
    builder = esmfold2.ESMFold2InputBuilder(ccd_cache=paths["ccd"].parent)
    optimization = None
    if optimization_config is not None and optimization_config.get("mode", "off") != "off":
        from ..esmfold2_optimization import OptimizationSession
        optimization = OptimizationSession(model, builder, torch, optimization_config, mode)
    return _ModelBundle(
        torch=torch,
        np=np,
        model=model,
        builder=builder,
        types=esmfold2,
        build_chains_from_input=prepare_input.build_chains_from_input,
        seed_context=processor._seed_context,
        lm_dropout_context=processor._lm_dropout_context,
        ccd_bond_reader=ccd_bond_reader,

        ccd_molecules=ccd_molecules,
        optimization=optimization,
    )


def _load_esmc(model: Any, checkpoint: Path, torch: Any, backend: str) -> Any:
    """Load the ESM-C encoder with the requested upstream attention backend."""
    if backend not in {"sdpa", "flash_attention_2", "eager"}:
        raise RuntimeError(f"Unsupported ESM-C attention backend: {backend}")
    esmc_module = importlib.import_module("transformers.models.esmc.modeling_esmc")
    esmc_class = getattr(esmc_module, "ESMCModel")
    esmc = esmc_class.from_pretrained(
        str(checkpoint), local_files_only=True, attn_implementation=backend,
        torch_dtype=torch.bfloat16,
    ).to(device=model.device, dtype=torch.bfloat16).eval()
    for parameter in esmc.parameters():
        parameter.requires_grad_(False)
    if backend == "flash_attention_2" and not bool(
        getattr(esmc, "_use_flash_attn", False)
    ):
        raise RuntimeError(
            "ESMFold2 FlashAttention 2 was requested but is not active"
        )
    model._esmc = esmc
    model._esmc_fp8 = False
    return esmc


def _context_embedding_cache_plan(
    spec: dict[str, Any],
    chain_infos: list[Any],
    config: dict[str, Any],
) -> tuple[bool, bool, list[Any]]:
    """Resolve safe per-request ESM-C cache use from the sequence ledger."""
    sequences = spec["sequences"]
    design_chains = set(spec["design_chains"])
    design = [item for item in sequences if item["chain_id"] in design_chains]
    if not design:
        raise RuntimeError("ESMFold2 input lacks its design-chain ledger entries")
    context_proteins = {
        str(item["chain_id"])
        for item in sequences
        if item["role"] == "context" and item["modality"] == "protein"
    }
    context_asym = [
        chain.asym_id
        for chain in chain_infos
        if chain.chain_id in context_proteins
    ]
    eligible = (
        config.get("lm_mask_pct") == 0
        and all(item["modality"] != "protein" for item in design)
        and bool(context_asym)
    )
    enabled = config.get("context_embedding_cache") is not False and eligible
    return enabled, eligible, context_asym


def _context_modification_ledger_descriptor(spec: dict[str, Any]) -> tuple[Any, ...]:
    """Build a direct descriptor for fixed protein context chains.

    The cache compares this occurrence descriptor directly, so atom-tokenized
    modified polymers cannot reuse another context's hidden states. Atom names
    remain canonicalized independently of source row order. Design-chain
    content is intentionally excluded.
    """

    sequences = spec.get("sequences") if isinstance(spec, dict) else None
    if not isinstance(sequences, list):
        raise RuntimeError("ESMFold2 context-cache sequence ledger is malformed")
    canonical: list[dict[str, Any]] = []
    seen_chain_ids: set[str] = set()
    for entry in sequences:
        if not isinstance(entry, dict):
            raise RuntimeError("ESMFold2 context-cache sequence entry is malformed")
        if entry.get("role") != "context" or entry.get("modality") != "protein":
            continue
        chain_id = entry.get("chain_id")
        if not isinstance(chain_id, str) or not chain_id or chain_id in seen_chain_ids:
            raise RuntimeError("ESMFold2 context-cache chain identity is malformed")
        seen_chain_ids.add(chain_id)
        raw_modifications = entry.get("modifications", [])
        if raw_modifications is None:
            raw_modifications = []
        if not isinstance(raw_modifications, list):
            raise RuntimeError(
                f"ESMFold2 context-cache modification ledger is malformed for {chain_id}"
            )
        modifications: list[dict[str, Any]] = []
        seen_positions: set[int] = set()
        sequence = entry.get("sequence")
        if sequence is None and not raw_modifications:
            sequence_length: int | None = None
        elif isinstance(sequence, str) and sequence:
            sequence_length = len(sequence)
        else:
            raise RuntimeError(
                f"ESMFold2 context-cache protein sequence is malformed for {chain_id}"
            )
        for raw in raw_modifications:
            if not isinstance(raw, dict):
                raise RuntimeError(
                    f"ESMFold2 context-cache modification is malformed for {chain_id}"
                )
            keys = frozenset(raw)
            if keys == _MODIFICATION_LEGACY_KEYS:
                strict = False
            elif keys == _MODIFICATION_STRICT_KEYS:
                strict = True
            else:
                raise RuntimeError(
                    f"ESMFold2 context-cache modification fields are malformed for {chain_id}"
                )
            position = raw.get("position")
            ccd = raw.get("ccd")
            if (
                not isinstance(position, int)
                or isinstance(position, bool)
                or position < 0
                or (
                    sequence_length is not None
                    and position >= sequence_length
                )
                or position in seen_positions
                or not isinstance(ccd, str)
                or not ccd
                or ccd.strip() != ccd
            ):
                raise RuntimeError(
                    f"ESMFold2 context-cache modification occurrence is malformed for {chain_id}"
                )
            seen_positions.add(position)
            if strict:
                atom_names = raw.get("atom_names")
                infer_standard_polymer_bonds = raw.get(
                    "infer_standard_polymer_bonds"
                )
                if (
                    not isinstance(atom_names, list)
                    or not atom_names
                    or len(atom_names) != len(set(atom_names))
                    or any(
                        not isinstance(name, str)
                        or not name
                        or name.strip() != name
                        or len(name) > 4
                        for name in atom_names
                    )
                    or not isinstance(infer_standard_polymer_bonds, bool)
                    or infer_standard_polymer_bonds
                ):
                    raise RuntimeError(
                        "ESMFold2 context-cache strict modification is malformed"
                    )
                canonical_atom_names: list[str] | None = sorted(atom_names)
                infer_value = False
            else:
                canonical_atom_names = None
                # Legacy callers omit this field; use the vendored constructor's
                # default in the canonical digest rather than conflating it with
                # strict false inference.
                infer_value = True
            modifications.append(
                {
                    "position": position,
                    "ccd": ccd,
                    "atom_names": canonical_atom_names,
                    "infer_standard_polymer_bonds": infer_value,
                }
            )
        modifications.sort(key=lambda item: item["position"])
        canonical.append({"chain_id": chain_id, "modifications": modifications})
    canonical.sort(key=lambda item: item["chain_id"])
    return tuple(
        (
            item["chain_id"],
            tuple(
                (
                    modification["position"],
                    modification["ccd"],
                    (
                        tuple(modification["atom_names"])
                        if modification["atom_names"] is not None
                        else None
                    ),
                    modification["infer_standard_polymer_bonds"],
                )
                for modification in item["modifications"]
            ),
        )
        for item in canonical
    )



def _cached_context_lm_hidden_states(
    bundle: _ModelBundle,
    features: dict[str, Any],
    context_asym: list[Any],
    context_modification_descriptor: tuple[Any, ...] = (),
) -> Any:
    """Reuse one bounded, ordered set of protein-context ESM-C states."""
    torch = bundle.torch
    input_ids = features["input_ids"]
    asym_id = features["asym_id"]
    residue_index = features["residue_index"]
    mol_type = features["mol_type"]
    token_mask = features["token_attention_mask"]
    uncached_mask = (mol_type == 0) & token_mask.bool()
    planned: list[tuple[Any, Any, int]] = []
    signature_parts: list[Any] = []
    for ordinal, asym in enumerate(context_asym):
        mask = uncached_mask & (asym_id == asym)
        indices = mask[0].nonzero(as_tuple=True)[0]
        if not len(indices):
            raise RuntimeError(
                "ESMFold2 context-cache ledger references an absent protein chain"
            )
        token_values: list[Any] = []
        for value in (
            input_ids[0, indices],
            asym_id[0, indices],
            residue_index[0, indices],
            mol_type[0, indices],
            token_mask[0, indices],
        ):
            cpu_value = value.detach().cpu().contiguous()
            token_values.append(
                (
                    tuple(cpu_value.shape),
                    str(cpu_value.dtype),
                    tuple(cpu_value.reshape(-1).tolist()),
                )
            )
        signature_parts.append((ordinal, tuple(token_values)))
        planned.append((mask, indices, ordinal))
        uncached_mask &= ~mask

    signature = (context_modification_descriptor, tuple(signature_parts))
    if not planned or uncached_mask.any():
        raise RuntimeError("ESMFold2 cache plan did not cover all protein tokens")


    if signature != bundle.context_cache_signature:
        # Bound resident GPU memory to exactly the current ordered context set.
        # Consecutive replicas with the same context reuse it; a new context
        # atomically evicts every tensor from the preceding set.
        bundle.context_embedding_cache.clear()
        bundle.context_cache_signature = signature

    result = None
    pending_cache: list[tuple[int, Any]] = []
    cache_hits = 0
    cache_misses = 0
    for mask, indices, key in planned:
        cached = bundle.context_embedding_cache.get(key)
        if cached is None:
            try:
                hidden = bundle.model._compute_lm_hidden_states(
                    input_ids,
                    asym_id,
                    residue_index,
                    mol_type,
                    mask,
                    lm_mask_pct=0.0,
                )
            except Exception as error:
                if _is_cuda_oom(bundle.torch, error):
                    raise
                raise _SystemicWorkerError(
                    "ESMFold2 ESM-C forward failed: "
                    f"{type(error).__name__}: {error}"
                ) from error
            cached = hidden[:, indices].detach()
            pending_cache.append((key, cached))
            cache_misses += 1
        else:
            cache_hits += 1
        if result is None:
            result = torch.zeros(
                (
                    input_ids.shape[0],
                    input_ids.shape[1],
                    cached.shape[2],
                    cached.shape[3],
                ),
                device=cached.device,
                dtype=cached.dtype,
            )
        result[:, indices] = cached
    if result is None:
        raise RuntimeError("ESMFold2 context cache produced no hidden states")
    for key, cached in pending_cache:
        bundle.context_embedding_cache[key] = cached
    bundle.context_cache_hits += cache_hits
    bundle.context_cache_misses += cache_misses
    return result


def _uncached_lm_hidden_states(
    bundle: _ModelBundle,
    features: dict[str, Any],
) -> Any:
    """Compute deterministic ESM-C states without retaining a GPU tensor."""

    try:
        return bundle.model._compute_lm_hidden_states(
            features["input_ids"],
            features["asym_id"],
            features["residue_index"],
            features["mol_type"],
            features["token_attention_mask"].bool(),
            lm_mask_pct=0.0,
        )
    except Exception as error:
        if _is_cuda_oom(bundle.torch, error):
            raise
        raise _SystemicWorkerError(
            "ESMFold2 ESM-C forward failed: "
            f"{type(error).__name__}: {error}"
        ) from error


def _lm_hidden_states_for_request(
    bundle: _ModelBundle,
    features: dict[str, Any],
    spec: dict[str, Any],
    chain_infos: list[Any],
    config: dict[str, Any],
) -> tuple[Any | None, dict[str, Any]]:
    """Resolve request-local ESM-C execution and bounded cache telemetry.

    Full hidden states are never persistent: protein designs change their LM
    input, while non-protein design tokens receive no ESM-C state worth caching.
    Only the current ordered fixed-protein-context set may remain resident.
    """

    context_enabled, cache_eligible, context_asym = _context_embedding_cache_plan(
        spec,
        chain_infos,
        config,
    )
    cache_enabled = context_enabled
    cache_before = (bundle.context_cache_hits, bundle.context_cache_misses)
    lm_cache_started = time.perf_counter()
    if cache_enabled:
        context_modification_descriptor = _context_modification_ledger_descriptor(spec)
        hidden = _cached_context_lm_hidden_states(
            bundle,
            features,
            context_asym,
            context_modification_descriptor,
        )
    elif config.get("lm_mask_pct") == 0:
        # Compute every protein-design request explicitly, but never retain its
        # hidden states in either owner-GPU cache.
        hidden = _uncached_lm_hidden_states(bundle, features)
    else:
        # Non-zero LM masking remains owned by the upstream model forward.
        hidden = None
    lm_cache_seconds = time.perf_counter() - lm_cache_started
    context_hits = bundle.context_cache_hits - cache_before[0]
    context_misses = bundle.context_cache_misses - cache_before[1]
    visible_context_entries = (
        len(bundle.context_embedding_cache) if cache_enabled else 0
    )
    return hidden, {
        "enabled": cache_enabled,
        "eligible": cache_eligible,
        "hits": context_hits,
        "misses": context_misses,
        "entries": visible_context_entries,
        "full_hits": 0,
        "full_misses": 0,
        "full_entries": 0,
        "context_hits": context_hits,
        "context_misses": context_misses,
        "context_entries": visible_context_entries,
        "lm_embedding_prep": lm_cache_seconds,
    }


def _decode_diffusion_results(
    bundle: _ModelBundle,
    output: Any,
    features: dict[str, Any],
    chain_infos: list[Any],
    *,
    samples: int,
    expected_prefix: str,
    expected_samples: int,
) -> list[Any]:
    """Decode the native model result while preserving the fatal protocol boundary."""

    try:
        decoded = bundle.builder.decode(
            output,
            features,
            chain_infos,
            num_diffusion_samples=samples,
            complex_id=expected_prefix,
        )
    except Exception as error:
        raise _SystemicWorkerError(
            f"ESMFold2 decode protocol failed: {type(error).__name__}: {error}"
        ) from error
    decoded_results = decoded if isinstance(decoded, list) else [decoded]
    if len(decoded_results) != expected_samples:
        raise _SystemicWorkerError(
            "ESMFold2 worker protocol returned an unexpected diffusion sample count: "
            f"{len(decoded_results)} != {expected_samples}"
        )
    return decoded_results


def _execute_request(
    bundle: _ModelBundle,
    request: dict[str, Any],
    config: dict[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    request_started = time.perf_counter()
    output_dir = request["output_dir"]
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError(f"ESMFold2 request output target is unsafe: {output_dir}")
    output_dir.mkdir(parents=True)
    input_bytes = request["input_json"].read_bytes()
    staged_input = output_dir / "input.json"
    staged_input.write_bytes(input_bytes)
    spec = json.loads(input_bytes)
    structure_input, preview_chains, expected_covalent_edges = (
        _structure_input_from_payload(
            spec,
            bundle.types,
            bundle.build_chains_from_input,
            ccd_molecules=bundle.ccd_molecules,
            seed=config["seed"],
        )
    )
    expected_chain_ids = [entry["chain_id"] for entry in spec["sequences"]]
    if [chain.chain_id for chain in preview_chains] != expected_chain_ids:
        raise RuntimeError("ESMFold2 changed canonical label_asym_id chain order")

    torch = bundle.torch
    attempt_stage = "device input preparation"

    def _request_attempt() -> tuple[Any, dict[str, Any], list[Any], dict[str, Any], float, float, float]:
        nonlocal attempt_stage
        input_conversion_started = time.perf_counter()
        attempt_stage = "device input preparation"
        try:
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats()
            features, chain_infos = bundle.builder.prepare_input(
                structure_input,
                seed=config["seed"],
                device=bundle.model.device,
            )
        except Exception as error:
            if _is_cuda_oom(torch, error):
                raise
            raise _SystemicWorkerError(
                "ESMFold2 device input preparation failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        if [chain.chain_id for chain in chain_infos] != expected_chain_ids:
            raise RuntimeError("ESMFold2 prepared input changed label_asym_id identity")
        _apply_prepared_entity_identity(spec["sequences"], features, chain_infos)
        _validate_expected_token_bonds(features, expected_covalent_edges)
        if bundle.optimization is not None:
            attempt_stage = "optimization input preparation"
            bundle.optimization.before_request(features)
        input_conversion_seconds = time.perf_counter() - input_conversion_started
        attempt_stage = "LM embedding preparation"
        lm_hidden_states, cache_stats = _lm_hidden_states_for_request(
            bundle,
            features,
            spec,
            chain_infos,
            config,
        )
        lm_cache_seconds = float(cache_stats["lm_embedding_prep"])
        seed_context = (
            bundle.seed_context(config["seed"])
            if config["seed"] is not None
            else nullcontext()
        )
        attempt_stage = "model forward"
        try:
            with torch.no_grad():
                with seed_context:
                    with bundle.lm_dropout_context(bundle.model, config["lm_dropout"]):
                        forward_started = time.perf_counter()

                        def _model_forward():
                            return bundle.model(
                                **features,
                                lm_hidden_states=lm_hidden_states,
                                num_loops=config["loops"],
                                num_sampling_steps=config["steps"],
                                num_diffusion_samples=config["samples"],
                                lm_mask_pct=config["lm_mask_pct"],
                                msa_max_depth=1,
                                msa_column_mask_rate=0.1,
                                msa_subsample_at_inference=False,
                            )

                        output = _model_forward()
            del lm_hidden_states
            torch.cuda.synchronize()
        except Exception as error:
            if _is_cuda_oom(torch, error):
                raise
            raise _SystemicWorkerError(
                f"ESMFold2 model forward failed: {type(error).__name__}: {error}"
            ) from error
        forward_seconds = time.perf_counter() - forward_started
        return (
            output,
            features,
            chain_infos,
            cache_stats,
            input_conversion_seconds,
            lm_cache_seconds,
            forward_seconds,
        )

    try:
        attempt_result = (
            bundle.optimization.request(_request_attempt)
            if bundle.optimization is not None
            else _request_attempt()
        )
    except Exception as error:
        if _is_cuda_oom(torch, error):
            raise _SystemicWorkerError(
                f"ESMFold2 {attempt_stage} failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        raise
    (
        output,
        features,
        chain_infos,
        cache_stats,
        input_conversion_seconds,
        lm_cache_seconds,
        forward_seconds,
    ) = attempt_result
    prepared_atom_inventory = _prepared_atom_inventory(features, bundle.np)
    # The CCD leaving-atom rule is checked against the in-memory prepared
    # tensors.  These labels are consumed for validation only and never enter
    # the confidence NPZ.
    _leaving_atom_omission_arrays(
        spec,
        prepared_atom_inventory,
        bundle.ccd_bond_reader,
        bundle.np,
    )
    output_started = time.perf_counter()
    if os.environ.get("PARTIDESIGN_ESMFOLD2_CAPTURE_COORDS") == "1":
        bundle.np.save(output_dir / "diagnostic_sample_coordinates.npy", output["sample_atom_coords"].detach().float().cpu().numpy(), allow_pickle=False)
    decoded_results = _decode_diffusion_results(
        bundle,
        output,
        features,
        chain_infos,
        samples=config["samples"],
        expected_prefix=request["expected_prefix"],
        expected_samples=request["expected_samples"],
    )

    artifacts: list[dict[str, Any]] = []
    for sample_index, result in enumerate(decoded_results):
        prefix = f"{request['expected_prefix']}_sample_{sample_index}"
        structure_path = output_dir / f"{prefix}.cif"
        confidence_path = output_dir / f"{prefix}.confidence.json"
        raw_confidence_path = output_dir / f"{prefix}.confidence.npz"
        cif_text = result.complex.to_mmcif()
        cif_text = _inject_explicit_topology(
            cif_text,
            spec,
            bundle.ccd_bond_reader,
            chain_infos=chain_infos,
            features=features,
        )
        _validate_output_atom_inventory_in_memory(
            cif_text,
            spec["sequences"],
            prepared_atom_inventory,
        )
        _atomic_text(structure_path, cif_text)
        confidence = _confidence_payload(
            output,
            features,
            chain_infos,
            sample_index,
            mode,
            config,
        )
        if "partidesign_receptor_segments" in spec:
            from src.design.rewards.esmfold2_mapping import native_binder_confidence, source_mapping

            confidence["partidesign_source_confidence"] = native_binder_confidence(
                output, features, chain_infos, sample_index, spec, bundle.np,
                source_mapping(spec),
            )
        confidence_record = {
            "confidence": confidence,
            "confidence_unavailable": list(_UNAVAILABLE_CONFIDENCE),
            "context_embedding_cache": cache_stats,
            "effective_config": {
                **config,
                "mode": mode,
                "external_msa_search": False,
                "msa_max_depth": 1,
                "msa_column_mask_rate": 0.1,
                "msa_subsample_at_inference": False,
            },
        }
        _atomic_json(confidence_path, confidence_record)
        _write_raw_confidence(
            raw_confidence_path, output, features, spec,
            bundle.ccd_bond_reader, sample_index, bundle.np,
        )
        artifacts.append({
            "sample_index": sample_index,
            "structure": str(structure_path.resolve()),
            "confidence": str(confidence_path.resolve()),
            "raw_confidence": str(raw_confidence_path.resolve()),
        })
    return artifacts


_MODIFICATION_LEGACY_KEYS = frozenset({"position", "ccd"})
_MODIFICATION_STRICT_KEYS = frozenset(
    {"position", "ccd", "atom_names", "infer_standard_polymer_bonds"}
)
_PRIVATE_COVALENT_ORDER_TO_MMCIF = {
    "?": "?",
    "single": "sing",
    "double": "doub",
    "triple": "trip",
    "aromatic": "arom",
}
_PRIVATE_COVALENT_ORDERS = frozenset(_PRIVATE_COVALENT_ORDER_TO_MMCIF)


def _mmcif_covalent_order(raw: Any) -> str:
    if not isinstance(raw, str) or raw not in _PRIVATE_COVALENT_ORDERS:
        raise RuntimeError(
            "ESMFold2 private covalent bond order must be one of ?, single, "
            "double, triple, aromatic"
        )
    return _PRIVATE_COVALENT_ORDER_TO_MMCIF[raw]


def _load_ccd_molecules(
    ccd_molecules: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the model-owned RDKit CCD dictionary.

    The ESMFold2 bundle already loads this dictionary during model setup.  The
    lazy import is retained for focused conversion tests and for callers that
    invoke ``_structure_input_from_payload`` directly; it never consults
    PartiDesign's pkl.gz inputs.
    """

    if ccd_molecules is not None:
        if not isinstance(ccd_molecules, dict):
            raise RuntimeError("ESMFold2 CCD molecule cache is malformed")
        return ccd_molecules
    try:
        conformers = importlib.import_module("esm.models.esmfold2.conformers")
        loaded = conformers.load_ccd()
    except Exception as error:
        raise RuntimeError(
            "ESMFold2 model-owned CCD molecule cache is unavailable: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise RuntimeError("ESMFold2 CCD molecule cache is malformed")
    return loaded


def _ccd_heavy_atom_names(
    ccd: str,
    ccd_molecules: dict[str, Any] | None,
) -> tuple[str, ...]:
    """Read the exact heavy-atom name allowlist from ESM's RDKit CCD cache."""

    molecules = _load_ccd_molecules(ccd_molecules)
    if ccd not in molecules:
        raise RuntimeError(f"ESMFold2 CCD component {ccd!r} is not in ccd.pkl")
    molecule = molecules[ccd]
    Chem = importlib.import_module("rdkit.Chem")
    try:
        # Match ESM's own atom reader: significant hydrogens may be present in
        # the pickle, but the model/output contract is heavy-atom only.
        molecule = Chem.RemoveHs(molecule, sanitize=False)
        names: list[str] = []
        for atom in molecule.GetAtoms():
            props = atom.GetPropsAsDict()
            name = props.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("an atom lacks its CCD name")
            if atom.GetSymbol().strip().upper() in {"H", "D", "T"}:
                continue
            names.append(name)
    except Exception as error:
        raise RuntimeError(
            f"ESMFold2 CCD component {ccd!r} has invalid RDKit atom data: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not names or len(names) != len(set(names)):
        raise RuntimeError(
            f"ESMFold2 CCD component {ccd!r} has an invalid heavy-atom allowlist"
        )
    return tuple(names)


def _validated_modification(
    raw: Any,
    sequence_length: int,
    ccd_molecules: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    """Validate one legacy or strict private modification occurrence."""

    if not isinstance(raw, dict):
        raise RuntimeError("ESMFold2 polymer modification entry is malformed")
    keys = frozenset(raw)
    if keys == _MODIFICATION_LEGACY_KEYS:
        strict = False
    elif keys == _MODIFICATION_STRICT_KEYS:
        strict = True
    else:
        raise RuntimeError(
            "ESMFold2 polymer modification must contain either {position,ccd} "
            "or the complete private atom-inventory fields"
        )
    position = raw.get("position")
    ccd = raw.get("ccd")
    if (
        not isinstance(position, int)
        or isinstance(position, bool)
        or not 0 <= position < sequence_length
        or not isinstance(ccd, str)
        or not ccd
        or ccd.strip() != ccd
    ):
        raise RuntimeError("ESMFold2 polymer modification position/CCD is malformed")

    # Validate existence even for legacy callers.  The worker's model bundle
    # supplies ccd_molecules from ccd.pkl, so unknown names cannot silently
    # fall back to a backbone-only proxy.
    allowed_names = _ccd_heavy_atom_names(ccd, ccd_molecules)
    if not strict:
        return {"position": position, "ccd": ccd}, False

    atom_names = raw.get("atom_names")
    infer_standard_polymer_bonds = raw.get("infer_standard_polymer_bonds")
    if (
        not isinstance(atom_names, list)
        or not atom_names
        or any(
            not isinstance(name, str)
            or not name
            or name.strip() != name
            or len(name) > 4
            for name in atom_names
        )
        or len(atom_names) != len(set(atom_names))
        or any(name not in allowed_names for name in atom_names)
        or not isinstance(infer_standard_polymer_bonds, bool)
        or infer_standard_polymer_bonds
    ):
        raise RuntimeError(
            f"ESMFold2 polymer modification {ccd!r} has an invalid exact heavy-atom ledger"
        )
    return {
        "position": position,
        "ccd": ccd,
        "atom_names": list(atom_names),
        "infer_standard_polymer_bonds": False,
    }, True


def _modification_from_payload(
    types: Any,
    raw: Any,
    sequence_length: int,
    ccd_molecules: dict[str, Any] | None,
) -> Any:
    """Construct a vendored Modification without dropping strict fields."""

    normalized, strict = _validated_modification(
        raw, sequence_length, ccd_molecules
    )
    kwargs = dict(normalized)
    constructor = getattr(types, "Modification", None)
    if not callable(constructor):
        raise RuntimeError("ESMFold2 vendored Modification type is unavailable")
    if strict:
        try:
            parameters = inspect.signature(constructor).parameters
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            supported = {
                name
                for name in ("atom_names", "infer_standard_polymer_bonds")
                if name in parameters or accepts_kwargs
            }
        except (TypeError, ValueError):
            # Some extension/factory constructors do not expose a signature;
            # let the actual call decide, while still failing closed below.
            supported = {"atom_names", "infer_standard_polymer_bonds"}
        if supported != {"atom_names", "infer_standard_polymer_bonds"}:
            raise RuntimeError(
                "ESMFold2 strict modification fields are unsupported by the "
                "vendored Modification constructor"
            )
    try:
        return constructor(**kwargs)
    except TypeError as error:
        if strict:
            raise RuntimeError(
                "ESMFold2 vendored Modification rejected strict private fields"
            ) from error
        # Keep the old two-field call contract explicit for constructors that
        # expose additional optional parameters but reject keyword expansion.
        try:
            return constructor(position=normalized["position"], ccd=normalized["ccd"])
        except TypeError:
            raise RuntimeError(
                "ESMFold2 vendored Modification rejected legacy fields"
            ) from error


def _structure_input_from_payload(
    payload: dict[str, Any],
    types: Any,
    build_chains_from_input: Any,
    *,
    seed: int,
    ccd_molecules: dict[str, Any] | None = None,
) -> tuple[Any, list[Any], list[tuple[int, int]]]:
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _INPUT_SCHEMA_VERSION
        or payload.get("msa_policy") != "none"
    ):
        raise RuntimeError("Invalid PipeTools ESMFold2 input schema")
    raw_sequences = payload.get("sequences")
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise RuntimeError("ESMFold2 input has no sequences")
    sequences: list[Any] = []
    seen: set[str] = set()
    design_count = 0
    for entry in raw_sequences:
        if not isinstance(entry, dict):
            raise RuntimeError("ESMFold2 sequence entry is malformed")
        chain_id = entry.get("chain_id")
        if not isinstance(chain_id, str) or not chain_id or chain_id in seen:
            raise RuntimeError("ESMFold2 chain IDs must be non-empty and unique")
        seen.add(chain_id)
        role = entry.get("role")
        if role not in {"design", "context"}:
            raise RuntimeError(f"ESMFold2 chain {chain_id} has an invalid role")
        design_count += int(role == "design")
        modifications_raw = entry.get("modifications", [])
        if modifications_raw is None:
            modifications_raw = []
        if not isinstance(modifications_raw, list):
            raise RuntimeError(
                f"ESMFold2 chain {chain_id} modification ledger is malformed"
            )
        modification_positions: set[int] = set()
        modifications: list[Any] = []
        sequence_length = (
            len(entry.get("sequence", ""))
            if isinstance(entry.get("sequence"), str)
            else -1
        )
        for item in modifications_raw:
            modification = _modification_from_payload(
                types,
                item,
                sequence_length,
                ccd_molecules,
            )
            position = item.get("position") if isinstance(item, dict) else None
            if position in modification_positions:
                raise RuntimeError(
                    f"ESMFold2 chain {chain_id} has duplicate modification position"
                )
            modification_positions.add(position)
            modifications.append(modification)
        modifications = modifications or None
        modality = entry.get("modality")
        if modality == "protein":
            if not isinstance(entry.get("sequence"), str) or not entry["sequence"]:
                raise RuntimeError(f"ESMFold2 chain {chain_id} sequence is malformed")
            if entry.get("msa") is not None:
                raise RuntimeError("External protein MSA is forbidden for ESMFold2")
            sequence = types.ProteinInput(
                id=chain_id,
                sequence=entry["sequence"],
                modifications=modifications,
                msa=None,
            )
        elif modality == "rna":
            if not isinstance(entry.get("sequence"), str) or not entry["sequence"]:
                raise RuntimeError(f"ESMFold2 chain {chain_id} sequence is malformed")
            if entry.get("msa") is not None:
                raise RuntimeError("External RNA MSA is forbidden for ESMFold2")
            sequence = types.RNAInput(
                id=chain_id,
                sequence=entry["sequence"],
                modifications=modifications,
                msa=None,
            )
        elif modality == "dna":
            if not isinstance(entry.get("sequence"), str) or not entry["sequence"]:
                raise RuntimeError(f"ESMFold2 chain {chain_id} sequence is malformed")
            sequence = types.DNAInput(
                id=chain_id,
                sequence=entry["sequence"],
                modifications=modifications,
            )
        elif modality == "ligand":
            has_smiles = isinstance(entry.get("smiles"), str) and bool(entry["smiles"])
            has_ccd = isinstance(entry.get("ccd"), list) and bool(entry["ccd"])
            if has_smiles == has_ccd:
                raise RuntimeError(
                    f"ESMFold2 ligand {chain_id} requires exactly one of smiles or ccd"
                )
            if has_ccd:
                if any(
                    not isinstance(component, str) or not component.strip()
                    for component in entry["ccd"]
                ):
                    raise RuntimeError(
                        f"ESMFold2 ligand {chain_id} has a malformed CCD ledger"
                    )
                for component in entry["ccd"]:
                    _ccd_heavy_atom_names(component, ccd_molecules)
            sequence = types.LigandInput(
                id=chain_id,
                smiles=entry["smiles"] if has_smiles else None,
                ccd=entry["ccd"] if has_ccd else None,
            )
        else:
            raise RuntimeError(
                f"ESMFold2 chain {chain_id} has unsupported modality {modality!r}"
            )
        sequences.append(sequence)
    design_chains = payload.get("design_chains")
    if (
        not isinstance(design_chains, list)
        or not design_chains
        or len(design_chains) != len(set(design_chains))
        or set(design_chains)
        != {entry["chain_id"] for entry in raw_sequences if entry["role"] == "design"}
        or design_count != len(design_chains)
        or payload.get("design_chain") != design_chains[0]
    ):
        raise RuntimeError("ESMFold2 input has an invalid design-chain ledger")

    raw_bonds = payload.get("covalent_bonds", [])
    provisional_bonds = _provisional_covalent_bonds(raw_bonds, types)
    # CCD leaving atoms are removed only when the upstream tokenizer knows a
    # chain participates in a covalent bond.  Build the atom-name ledger with
    # placeholder ordinals first; resolving against an unbonded preview would
    # shift every atom after a removed leaving atom.
    preliminary = types.StructurePredictionInput(
        sequences=sequences,
        covalent_bonds=provisional_bonds or None,
    )
    chains, tokens, atoms = build_chains_from_input(preliminary, seed=seed)
    if [chain.chain_id for chain in chains] != [entry["chain_id"] for entry in raw_sequences]:
        raise RuntimeError("ESMFold2 upstream changed chain identity or order")
    _apply_chain_entity_identity(raw_sequences, chains, tokens)
    numeric_bonds: list[Any] = []
    expected_edges: list[tuple[int, int]] = []
    for bond_index, raw_bond in enumerate(raw_bonds, start=1):
        left = _numeric_endpoint(
            raw_bond["endpoint1"], chains, tokens, atoms, bond_index, 1
        )
        right = _numeric_endpoint(
            raw_bond["endpoint2"], chains, tokens, atoms, bond_index, 2
        )
        numeric_bonds.append(
            types.CovalentBond(
                chain_id1=left[0],
                res_idx1=left[1],
                atom_idx1=left[2],
                chain_id2=right[0],
                res_idx2=right[1],
                atom_idx2=right[2],
            )
        )
        expected_edges.append((left[3], right[3]))
    final = types.StructurePredictionInput(
        sequences=sequences,
        covalent_bonds=numeric_bonds or None,
    )
    return final, chains, expected_edges


def _apply_chain_entity_identity(
    raw_sequences: list[dict[str, Any]],
    chains: list[Any],
    tokens: list[Any],
) -> None:
    """Override upstream sequence-only dedup with variant-aware entity IDs."""

    if len(raw_sequences) != len(chains):
        raise RuntimeError("ESMFold2 entity identity chain cardinality differs")
    expected_entities = [entry.get("model_entity_index") for entry in raw_sequences]
    if (
        any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in expected_entities
        )
        or set(expected_entities) != set(range(1, max(expected_entities, default=0) + 1))
    ):
        raise RuntimeError("ESMFold2 model entity identity ledger is malformed")
    copies_by_entity: dict[int, set[int]] = {}
    output_by_entity: dict[int, str] = {}
    entity_by_output: dict[str, int] = {}
    for entry, chain in zip(raw_sequences, chains, strict=True):
        expected_chain = entry.get("chain_id")
        model_entity = entry.get("model_entity_index")
        model_copy = entry.get("model_entity_copy")
        output_entity = entry.get("output_entity_id")
        if (
            getattr(chain, "chain_id", None) != expected_chain
            or not isinstance(model_copy, int)
            or isinstance(model_copy, bool)
            or model_copy < 1
            or not isinstance(output_entity, str)
            or not output_entity
        ):
            raise RuntimeError("ESMFold2 variant entity identity ledger is malformed")
        copies_by_entity.setdefault(model_entity, set()).add(model_copy)
        if output_by_entity.setdefault(model_entity, output_entity) != output_entity:
            raise RuntimeError("ESMFold2 one model entity has multiple output IDs")
        if entity_by_output.setdefault(output_entity, model_entity) != model_entity:
            raise RuntimeError("ESMFold2 output entity ID aliases model entities")
        chain.entity_id = model_entity - 1
        chain.sym_id = model_copy - 1
        for token in tokens:
            if getattr(token, "asym_id", None) == chain.asym_id:
                token.entity_id = model_entity - 1
                token.sym_id = model_copy - 1
    for entity, copies in copies_by_entity.items():
        if copies != set(range(1, len(copies) + 1)):
            raise RuntimeError(
                f"ESMFold2 model entity {entity} copy ordinals are not contiguous"
            )


def _apply_prepared_entity_identity(
    raw_sequences: list[dict[str, Any]],
    features: dict[str, Any],
    chains: list[Any],
) -> None:
    tokens = [token for chain in chains for token in chain.tokens]
    _apply_chain_entity_identity(raw_sequences, chains, tokens)
    entity_id = features.get("entity_id")
    sym_id = features.get("sym_id")
    if entity_id is None or sym_id is None:
        raise RuntimeError("ESMFold2 prepared features lack entity/sym identity")
    if tuple(entity_id.shape) != tuple(sym_id.shape) or len(entity_id.shape) != 2:
        raise RuntimeError("ESMFold2 prepared entity/sym feature shapes differ")
    for chain in chains:
        indices = [token.token_index for token in chain.tokens]
        if not indices:
            raise RuntimeError(f"ESMFold2 prepared chain {chain.chain_id} has no tokens")
        entity_id[0, indices] = chain.entity_id
        sym_id[0, indices] = chain.sym_id


def _validate_expected_token_bonds(
    features: dict[str, Any], expected_edges: list[tuple[int, int]]
) -> None:
    if not expected_edges:
        return
    token_bonds = features.get("token_bonds")
    if token_bonds is None or not hasattr(token_bonds, "shape"):
        raise RuntimeError("ESMFold2 prepared input lacks token_bonds evidence")
    shape = tuple(int(value) for value in token_bonds.shape)
    if len(shape) != 4 or shape[0] != 1 or shape[3] != 1:
        raise RuntimeError(
            "ESMFold2 token_bonds must have shape [1, tokens, tokens, 1]"
        )
    matrix = token_bonds[0, :, :, 0]
    limit = int(matrix.shape[0])
    if int(matrix.shape[1]) != limit:
        raise RuntimeError("ESMFold2 token_bonds matrix is not square")
    for left, right in expected_edges:
        if not 0 <= left < limit or not 0 <= right < limit:
            raise RuntimeError("ESMFold2 explicit covalent edge is outside token_bonds")
        forward = matrix[left, right]
        reverse = matrix[right, left]
        forward = forward.item() if hasattr(forward, "item") else forward
        reverse = reverse.item() if hasattr(reverse, "item") else reverse
        if float(forward) <= 0 or float(reverse) <= 0:
            raise RuntimeError(
                f"ESMFold2 dropped explicit covalent token edge {left}<->{right}"
            )


def _provisional_covalent_bonds(raw: Any, types: Any) -> list[Any]:
    if not isinstance(raw, list):
        raise RuntimeError("ESMFold2 covalent bonds must be a list")
    provisional: list[Any] = []
    for bond_index, bond in enumerate(raw, start=1):
        if not isinstance(bond, dict) or set(bond) != {
            "endpoint1",
            "endpoint2",
            "bond_order",
        }:
            raise RuntimeError(f"ESMFold2 covalent bond {bond_index} is malformed")
        bond_order = bond["bond_order"]
        if (
            not isinstance(bond_order, str)
            or not bond_order
            or bond_order.strip() != bond_order
            or bond_order not in _PRIVATE_COVALENT_ORDERS
        ):
            raise RuntimeError(
                f"ESMFold2 covalent bond {bond_index} order is malformed"
            )
        left = _validated_endpoint(bond["endpoint1"], bond_index, 1)
        right = _validated_endpoint(bond["endpoint2"], bond_index, 2)
        provisional.append(
            types.CovalentBond(
                chain_id1=left[0],
                res_idx1=left[1],
                atom_idx1=0,
                chain_id2=right[0],
                res_idx2=right[1],
                atom_idx2=0,
            )
        )
    return provisional


def _validated_endpoint(
    endpoint: Any, bond_index: int, side: int
) -> tuple[str, int, str]:
    if not isinstance(endpoint, dict) or set(endpoint) != {
        "chain_id",
        "residue_index",
        "atom_name",
    }:
        raise RuntimeError(
            f"ESMFold2 covalent bond {bond_index} endpoint {side} is malformed"
        )
    chain_id = endpoint["chain_id"]
    residue_index = endpoint["residue_index"]
    atom_name = endpoint["atom_name"]
    if (
        not isinstance(chain_id, str)
        or not chain_id
        or not isinstance(residue_index, int)
        or isinstance(residue_index, bool)
        or residue_index < 0
        or not isinstance(atom_name, str)
        or not atom_name
        or atom_name.strip() != atom_name
    ):
        raise RuntimeError(
            f"ESMFold2 covalent bond {bond_index} endpoint {side} is malformed"
        )
    return chain_id, residue_index, atom_name


def _numeric_endpoint(
    endpoint: Any,
    chains: list[Any],
    tokens: list[Any],
    atoms: list[Any],
    bond_index: int,
    side: int,
) -> tuple[str, int, int, int]:
    chain_id, residue_index, atom_name = _validated_endpoint(
        endpoint, bond_index, side
    )
    chain_matches = [chain for chain in chains if chain.chain_id == chain_id]
    if len(chain_matches) != 1:
        raise RuntimeError(
            f"ESMFold2 covalent bond {bond_index} endpoint {side} chain is ambiguous"
        )
    chain = chain_matches[0]
    token_by_index = {token.token_index: token for token in tokens}
    residue_atoms = []
    for atom in atoms:
        token = token_by_index.get(atom.token_index)
        if (
            token is not None
            and token.asym_id == chain.asym_id
            and token.residue_index == residue_index
            and getattr(atom, "is_valid", True)
        ):
            residue_atoms.append(atom)
    matches = [
        index
        for index, atom in enumerate(residue_atoms)
        if atom.name == atom_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"ESMFold2 covalent bond {bond_index} endpoint {side} atom "
            f"{chain_id}:{residue_index + 1}:{atom_name} resolved {len(matches)} times"
        )
    matched = residue_atoms[matches[0]]
    return chain_id, residue_index, matches[0], int(matched.token_index)


def _confidence_payload(
    output: dict[str, Any],
    features: dict[str, Any],
    chain_infos: list[Any],
    sample_index: int,
    mode: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    mask = features["token_attention_mask"][0].bool()
    valid_count = int(mask.sum().item())
    if valid_count < 1:
        raise RuntimeError("ESMFold2 confidence has no valid tokens")

    def scalar(name: str, scale: float = 1.0) -> float:
        value = float(output[name][sample_index].detach().float().cpu().item()) * scale
        if not (-float("inf") < value < float("inf")):
            raise RuntimeError(f"ESMFold2 confidence {name} is non-finite")
        return value

    plddt = output["plddt"][sample_index].detach().float().cpu()
    per_chain = {
        chain.chain_id: float(
            plddt[[token.token_index for token in chain.tokens]].mean().item() * 100.0
        )
        for chain in chain_infos
    }
    per_chain_token_count = {
        chain.chain_id: len(chain.tokens) for chain in chain_infos
    }
    pair = output["pair_chains_iptm"][sample_index].detach().float().cpu()
    pair_chains: dict[str, dict[str, float]] = {}
    for left in chain_infos:
        pair_chains[left.chain_id] = {
            right.chain_id: float(pair[left.asym_id, right.asym_id].item())
            for right in chain_infos
        }
    valid_indices = mask.detach().cpu().nonzero(as_tuple=False).flatten()

    def valid_pair_mean(name: str) -> float:
        matrix = output[name][sample_index].detach().float().cpu()
        selected = matrix.index_select(0, valid_indices).index_select(
            1, valid_indices
        )
        return float(selected.mean().item())

    pae_mean = valid_pair_mean("pae")
    pde_mean = valid_pair_mean("pde")
    return {
        "model": "esmfold2",
        "mode": mode,
        "plddt": scalar("complex_plddt", 100.0),
        "ptm": scalar("ptm"),
        "iptm": scalar("iptm"),
        "complex_iplddt": scalar("complex_iplddt", 100.0),
        "pae_mean": pae_mean,
        "pde_mean": pde_mean,
        "per_chain_plddt": per_chain,
        "per_chain_token_count": per_chain_token_count,
        "chain_pair_iptm": pair_chains,
        "sampling": {
            "seed": config["seed"],
            "loops": config["loops"],
            "steps": config["steps"],
            "samples": config["samples"],
            "lm_dropout": config["lm_dropout"],
            "lm_mask_pct": config["lm_mask_pct"],
        },
    }


def _write_raw_confidence(
    path: Path,
    output: dict[str, Any],
    features: dict[str, Any],
    input_payload: dict[str, Any],
    ccd_bond_reader: CCDBondCacheReader,
    sample_index: int,
    np: Any,
) -> None:
    """Write confidence tensors and token labels consumed by downstream metrics.

    Model preparation graphs stay in memory. The archive is deliberately small:
    confidence arrays plus labels needed to map PAE rows back to chains/residues.
    """
    del input_payload, ccd_bond_reader
    arrays: dict[str, Any] = {}
    for name in (
        "plddt",
        "plddt_per_atom",
        "plddt_ca",
        "pae",
        "pde",
        "ptm",
        "iptm",
        "complex_plddt",
        "complex_iplddt",
        "pair_chains_iptm",
    ):
        arrays[name] = output[name][sample_index].detach().float().cpu().numpy()
    for name in (
        "token_index",
        "residue_index",
        "asym_id",
        "entity_id",
        "sym_id",
        "token_attention_mask",
    ):
        if name not in features:
            raise RuntimeError(f"ESMFold2 output lacks token label {name}")
        arrays[name] = features[name][0].detach().cpu().numpy()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _leaving_atom_omission_arrays(
    input_payload: dict[str, Any],
    arrays: dict[str, Any],
    ccd_bond_reader: CCDBondCacheReader,
    np: Any,
) -> dict[str, Any]:
    """Record exact heavy CCD leaving atoms absent from prepared ligand tokens."""

    sequences = input_payload.get("sequences")
    bonds = input_payload.get("covalent_bonds")
    if not isinstance(sequences, list) or not isinstance(bonds, list):
        raise RuntimeError("ESMFold2 private chemistry ledger is malformed")
    bonded_chains: set[str] = set()
    for bond in bonds:
        if not isinstance(bond, dict):
            raise RuntimeError("ESMFold2 private covalent bond is malformed")
        for side in (1, 2):
            endpoint = bond.get(f"endpoint{side}")
            chain_id = endpoint.get("chain_id") if isinstance(endpoint, dict) else None
            if not isinstance(chain_id, str) or not chain_id:
                raise RuntimeError("ESMFold2 private covalent endpoint is malformed")
            bonded_chains.add(chain_id)

    atom_mask = np.asarray(arrays["atom_attention_mask"]).astype(bool, copy=False)
    token_mask = np.asarray(arrays["token_attention_mask"]).astype(bool, copy=False)
    atom_to_token = np.asarray(arrays["atom_to_token"]).astype(int, copy=False)
    asym_id = np.asarray(arrays["asym_id"]).astype(int, copy=False)
    residue_index = np.asarray(arrays["residue_index"]).astype(int, copy=False)
    valid_atom_indices = np.flatnonzero(atom_mask)
    valid_token_indices = atom_to_token[valid_atom_indices]
    if (
        np.any(valid_token_indices < 0)
        or np.any(valid_token_indices >= token_mask.size)
        or not np.all(token_mask[valid_token_indices])
    ):
        raise RuntimeError("ESMFold2 prepared atom-to-token ledger is invalid")
    names = decode_atom_name_rows(
        np.asarray(arrays["ref_atom_name_chars"])[valid_atom_indices]
    )
    prepared = {
        (
            int(asym_id[token]),
            int(residue_index[token]),
            atom_name,
        )
        for token, atom_name in zip(valid_token_indices, names, strict=True)
    }

    records: list[tuple[int, int, str, int]] = []
    for chain_index, entry in enumerate(sequences):
        if not isinstance(entry, dict):
            raise RuntimeError("ESMFold2 private sequence entry is malformed")
        chain_id = entry.get("chain_id")
        components = entry.get("ccd")
        if (
            entry.get("modality") != "ligand"
            or chain_id not in bonded_chains
            or not isinstance(components, list)
        ):
            continue
        for component_index, component in enumerate(components):
            if not isinstance(component, str) or not component:
                raise RuntimeError("ESMFold2 private ligand CCD ledger is malformed")
            for atom in ccd_bond_reader.query_leaving_atoms(component):
                number = atomic_number(atom.type_symbol)
                if number == 1:
                    continue
                key = (chain_index, component_index, atom.atom_id)
                if key not in prepared:
                    records.append((*key, number))

    if len(records) != len({record[:3] for record in records}):
        raise RuntimeError("ESMFold2 leaving-atom omission ledger is duplicate")
    records.sort(key=lambda value: (value[0], value[1], value[2]))
    return {
        "omission_chain_index": np.asarray(
            [record[0] for record in records], dtype=np.int64
        ),
        "omission_residue_index": np.asarray(
            [record[1] for record in records], dtype=np.int64
        ),
        "omission_atom_name_chars": encode_atom_name_rows(
            [record[2] for record in records]
        ),
        "omission_atomic_number": np.asarray(
            [record[3] for record in records], dtype=np.int64
        ),
    }


def _inject_explicit_topology(
    cif_text: str,
    spec: dict[str, Any],
    ccd_bond_reader: CCDBondCacheReader,
    *,
    chain_infos: list[Any] | None = None,
    features: dict[str, Any] | None = None,
) -> str:
    """Attach CCD bond order and explicit covalent links without distance guesses."""

    smiles_entries = [
        entry
        for entry in spec.get("sequences", [])
        if isinstance(entry, dict)
        and isinstance(entry.get("smiles"), str)
        and bool(entry["smiles"])
    ]
    ccd_ligand_uses_lig = any(
        isinstance(entry, dict)
        and isinstance(entry.get("ccd"), list)
        and "LIG" in entry["ccd"]
        for entry in spec.get("sequences", [])
    )
    if len(smiles_entries) > 1 or (smiles_entries and ccd_ligand_uses_lig):
        # Every private SMILES is currently emitted by the upstream decoder
        # under the placeholder component ``LIG``.  Merging two independent
        # graphs (including a CCD component already named ``LIG``) under that
        # one component would silently corrupt topology.
        # The public LIG01/LIG02 writer policy is not available at this
        # serialization boundary, so reject the candidate before parsing or
        # writing its output.
        raise RuntimeError(
            "ESMFold2 cannot serialize multiple private graphs under one LIG "
            "component"
        )
    np = importlib.import_module("numpy")
    pdbx = importlib.import_module("biotite.structure.io.pdbx")
    cif_file = pdbx.CIFFile.read(io.StringIO(cif_text))
    block = next(iter(cif_file.values()))
    _rewrite_output_ligand_residues(
        block,
        spec,
        chain_infos,
        features,
        pdbx,
        np,
    )
    rewrite_output_entity_identity(block, spec, pdbx, np)
    _validate_output_modified_occurrences(block, spec)

    component_ids: list[str] = []
    for entry in spec["sequences"]:
        ccd_components = entry.get("ccd", [])
        if ccd_components is not None:
            if not isinstance(ccd_components, list):
                raise RuntimeError("ESMFold2 private CCD ledger is malformed")
            component_ids.extend(ccd_components)
        modifications = entry.get("modifications", [])
        if modifications is None:
            modifications = []
        if not isinstance(modifications, list):
            raise RuntimeError("ESMFold2 private modification ledger is malformed")
        component_ids.extend(modification["ccd"] for modification in modifications)
    component_ids = list(dict.fromkeys(component_ids))
    atom_site = block["atom_site"]
    present_component_atoms = _component_atom_intersection(
        atom_site["label_comp_id"].as_array(str),
        atom_site["label_asym_id"].as_array(str),
        atom_site["label_seq_id"].as_array(str),
        atom_site["label_atom_id"].as_array(str),
    )
    bond_rows: list[dict[str, str]] = []
    ordinal = 1
    for component_id in component_ids:
        present = present_component_atoms.get(component_id)
        if not present:
            raise RuntimeError(
                f"ESMFold2 output lacks CCD component {component_id} atoms"
            )
        for bond in ccd_bond_reader.query(component_id):
            if bond.atom_id_1 not in present or bond.atom_id_2 not in present:
                continue
            bond_rows.append(
                {
                    "pdbx_ordinal": str(ordinal),
                    "comp_id": component_id,
                    "atom_id_1": bond.atom_id_1,
                    "atom_id_2": bond.atom_id_2,
                    "value_order": bond.value_order,
                    "pdbx_aromatic_flag": bond.aromatic_flag,
                    "pdbx_stereo_config": bond.stereo_config,
                }
            )
            ordinal += 1
    # ESMFold2 atomizes a SMILES ligand but its mmCIF decoder does not emit the
    # corresponding local component graph.  Rebuild that graph from the exact
    # private SMILES using the same canonical atom-name algorithm as the model
    # tokenizer.  This is serialization of an authoritative input graph, not
    # distance-based bond inference.
    Chem = importlib.import_module("rdkit.Chem")
    AllChem = importlib.import_module("rdkit.Chem.AllChem")
    order_name = {
        str(Chem.BondType.SINGLE): "SING",
        str(Chem.BondType.DOUBLE): "DOUB",
        str(Chem.BondType.TRIPLE): "TRIP",
        str(Chem.BondType.AROMATIC): "AROM",
    }
    for entry in spec["sequences"]:
        smiles = entry.get("smiles")
        if not isinstance(smiles, str) or not smiles:
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise RuntimeError("ESMFold2 private SMILES cannot be parsed for output")
        molecule = Chem.AddHs(molecule)
        ranks = list(AllChem.CanonicalRankAtoms(molecule))
        for atom, rank in zip(molecule.GetAtoms(), ranks, strict=True):
            atom.SetProp("name", atom.GetSymbol().upper() + str(int(rank) + 1))
        molecule = Chem.RemoveHs(molecule)
        for bond in molecule.GetBonds():
            try:
                value_order = order_name[str(bond.GetBondType())]
            except KeyError as exc:
                raise RuntimeError("ESMFold2 SMILES has unsupported bond order") from exc
            bond_rows.append(
                {
                    "pdbx_ordinal": str(ordinal),
                    "comp_id": "LIG",
                    "atom_id_1": bond.GetBeginAtom().GetProp("name"),
                    "atom_id_2": bond.GetEndAtom().GetProp("name"),
                    "value_order": value_order,
                    "pdbx_aromatic_flag": "Y" if bond.GetIsAromatic() else "N",
                    "pdbx_stereo_config": "N",
                }
            )
            ordinal += 1
        present = present_component_atoms.get("LIG", frozenset())
        expected_atom_names = {
            atom.GetProp("name")
            for atom in molecule.GetAtoms()
            if atom.HasProp("name")
        }
        missing = sorted(expected_atom_names.difference(present))
        if missing:
            raise RuntimeError(
                "ESMFold2 SMILES canonical atom names are absent from output "
                f"atom_site: missing={missing}"
            )
    if bond_rows:
        _merge_chem_comp_bond(block, pdbx, np, bond_rows)

    connection_rows: list[dict[str, str]] = []
    for index, bond in enumerate(spec.get("covalent_bonds", []), start=1):
        left = _output_endpoint(block, bond["endpoint1"])
        right = _output_endpoint(block, bond["endpoint2"])
        connection_rows.append(
            {
                "id": str(index),
                "conn_type_id": "covale",
                "pdbx_value_order": _mmcif_covalent_order(bond["bond_order"]),
                "ptnr1_label_asym_id": bond["endpoint1"]["chain_id"],
                "ptnr2_label_asym_id": bond["endpoint2"]["chain_id"],
                "ptnr1_label_comp_id": left[0],
                "ptnr2_label_comp_id": right[0],
                "ptnr1_label_seq_id": left[1],
                "ptnr2_label_seq_id": right[1],
                "ptnr1_label_atom_id": bond["endpoint1"]["atom_name"],
                "ptnr2_label_atom_id": bond["endpoint2"]["atom_name"],
                "pdbx_ptnr1_PDB_ins_code": ".",
                "pdbx_ptnr2_PDB_ins_code": ".",
            }
        )
    if connection_rows:
        _merge_struct_conn(block, pdbx, np, connection_rows)
    output = io.StringIO()
    cif_file.write(output)
    return output.getvalue()


def _validate_output_modified_occurrences(block: Any, spec: dict[str, Any]) -> None:
    """Require strict modified-polymer occurrences to retain exact atom names."""

    atom_site = block["atom_site"]
    arrays = {
        name: atom_site[name].as_array(str)
        for name in ("label_asym_id", "label_seq_id", "label_comp_id", "label_atom_id")
    }
    for entry in spec.get("sequences", []):
        if not isinstance(entry, dict) or entry.get("modality") not in {
            "protein",
            "rna",
            "dna",
        }:
            continue
        chain_id = entry.get("chain_id")
        sequence = entry.get("sequence")
        modifications = entry.get("modifications", [])
        if not isinstance(chain_id, str) or not isinstance(sequence, str):
            raise RuntimeError("ESMFold2 output modified-polymer ledger is malformed")
        if modifications is None:
            modifications = []
        if not isinstance(modifications, list):
            raise RuntimeError("ESMFold2 output modification ledger is malformed")
        for modification in modifications:
            if not isinstance(modification, dict):
                raise RuntimeError("ESMFold2 output modification entry is malformed")
            if set(modification) == _MODIFICATION_LEGACY_KEYS:
                continue
            if set(modification) != _MODIFICATION_STRICT_KEYS:
                raise RuntimeError("ESMFold2 output modification entry is malformed")
            position = modification.get("position")
            atom_names = modification.get("atom_names")
            if (
                not isinstance(position, int)
                or isinstance(position, bool)
                or not 0 <= position < len(sequence)
                or not isinstance(atom_names, list)
                or not atom_names
                or len(atom_names) != len(set(atom_names))
                or modification.get("infer_standard_polymer_bonds") is not False
            ):
                raise RuntimeError("ESMFold2 output strict modification is malformed")
            rows = [
                index
                for index, (observed_chain, observed_position) in enumerate(
                    zip(arrays["label_asym_id"], arrays["label_seq_id"], strict=True)
                )
                if str(observed_chain) == chain_id
                and str(observed_position) == str(position + 1)
            ]
            observed_names = [str(arrays["label_atom_id"][index]) for index in rows]
            if (
                len(observed_names) != len(atom_names)
                or set(observed_names) != set(atom_names)
            ):
                missing = sorted(set(atom_names).difference(observed_names))
                extra = sorted(set(observed_names).difference(atom_names))
                raise RuntimeError(
                    f"ESMFold2 output modified occurrence {chain_id}:{position + 1} "
                    "does not retain its exact atom-name allowlist: "
                    f"missing={missing}, extra={extra}"
                )
            observed_components = {
                str(arrays["label_comp_id"][index]) for index in rows
            }
            if observed_components != {str(modification["ccd"])}:
                raise RuntimeError(
                    f"ESMFold2 output modified occurrence {chain_id}:{position + 1} "
                    "changed CCD identity"
                )


def _component_atom_intersection(
    comp_ids: Any,
    asym_ids: Any,
    seq_ids: Any,
    atom_ids: Any,
) -> dict[str, frozenset[str]]:
    """Return atoms present in every output instance of each component.

    A single mmCIF ``chem_comp_bond`` graph applies to all instances sharing a
    ``comp_id``.  ESMFold2 omits hydrogens and can remove a CCD leaving atom at
    a covalent link, so only endpoints present in every concrete instance are
    serializable without inventing an atom or an instance-specific component.
    """

    values = tuple(zip(comp_ids, asym_ids, seq_ids, atom_ids, strict=True))
    instances: dict[tuple[str, str, str], set[str]] = {}
    for raw_comp, raw_asym, raw_seq, raw_atom in values:
        key = (str(raw_comp), str(raw_asym), str(raw_seq))
        instances.setdefault(key, set()).add(str(raw_atom))
    by_component: dict[str, list[set[str]]] = {}
    for (component, _asym, _seq), names in instances.items():
        by_component.setdefault(component, []).append(names)
    return {
        component: frozenset(set.intersection(*instance_names))
        for component, instance_names in by_component.items()
    }


def _rewrite_output_ligand_residues(
    block: Any,
    spec: dict[str, Any],
    chain_infos: list[Any] | None,
    features: dict[str, Any] | None,
    pdbx: Any,
    np: Any,
) -> None:
    """Restore per-CCD residue identity lost by the upstream decoder.

    ESMFold2 atom-tokenizes non-polymers correctly, including a distinct
    ``residue_index`` for every CCD in a glycan.  Its current
    ``build_molecular_complex_from_features()`` decoder then deliberately
    collapses every non-polymer chain into one mmCIF residue.  That loses both
    chemical identity and the residue coordinate needed by explicit bonds.

    Rebuild those atom-site annotations from the exact prepared-token ledger.
    Rows are matched in model output order and atom names are checked before
    any rewrite; no coordinate or distance inference is involved.
    """

    ligand_entries = {
        entry["chain_id"]: entry
        for entry in spec.get("sequences", [])
        if (
            isinstance(entry, dict)
            and entry.get("modality") == "ligand"
            and isinstance(entry.get("ccd"), list)
            and bool(entry["ccd"])
        )
    }
    multi_component = {
        chain_id
        for chain_id, entry in ligand_entries.items()
        if isinstance(entry.get("ccd"), list) and len(entry["ccd"]) > 1
    }
    if not ligand_entries:
        return
    if chain_infos is None or features is None:
        if multi_component:
            raise RuntimeError(
                "ESMFold2 multi-CCD output requires the prepared token ledger"
            )
        return
    if "atom_attention_mask" not in features or "ref_atom_name_chars" not in features:
        raise RuntimeError("ESMFold2 output rewrite lacks prepared atom features")

    atom_mask = _unbatch_feature_array(features["atom_attention_mask"], np).astype(
        bool
    )
    atom_name_chars = _unbatch_feature_array(features["ref_atom_name_chars"], np)
    if atom_name_chars.ndim != 2 or atom_name_chars.shape[0] != atom_mask.shape[0]:
        raise RuntimeError("ESMFold2 prepared atom-name ledger has an invalid shape")

    atom_site = block["atom_site"]
    label_asym = atom_site["label_asym_id"].as_array(str)
    label_atom = atom_site["label_atom_id"].as_array(str)
    rewritten_comp = atom_site["label_comp_id"].as_array(str).astype(object)
    rewritten_label_seq = atom_site["label_seq_id"].as_array(str).astype(object)
    rewritten_auth_seq = atom_site["auth_seq_id"].as_array(str).astype(object)
    rewritten_auth_comp = (
        atom_site["auth_comp_id"].as_array(str).astype(object)
        if "auth_comp_id" in atom_site
        else None
    )

    chain_info_by_id: dict[str, Any] = {}
    for chain_info in chain_infos:
        chain_id = getattr(chain_info, "chain_id", None)
        if not isinstance(chain_id, str) or not chain_id or chain_id in chain_info_by_id:
            raise RuntimeError("ESMFold2 prepared chain ledger is malformed")
        chain_info_by_id[chain_id] = chain_info

    for chain_id, entry in ligand_entries.items():
        ccd = entry.get("ccd")
        if not isinstance(ccd, list) or not ccd or not all(
            isinstance(component, str) and component for component in ccd
        ):
            raise RuntimeError(f"ESMFold2 ligand {chain_id} has an invalid CCD ledger")
        chain_info = chain_info_by_id.get(chain_id)
        if chain_info is None:
            raise RuntimeError(f"ESMFold2 prepared output lacks ligand chain {chain_id}")
        expected_rows: list[tuple[int, str, str]] = []
        observed_residues: dict[int, str] = {}
        for token in getattr(chain_info, "tokens", []):
            residue_index = getattr(token, "residue_index", None)
            residue_name = getattr(token, "residue_name", None)
            atom_start = getattr(token, "atom_start", None)
            atom_count = getattr(token, "atom_count", None)
            if (
                not isinstance(residue_index, int)
                or not 0 <= residue_index < len(ccd)
                or residue_name != ccd[residue_index]
                or not isinstance(atom_start, int)
                or not isinstance(atom_count, int)
                or atom_count < 1
            ):
                raise RuntimeError(
                    f"ESMFold2 prepared ligand token ledger differs for chain {chain_id}"
                )
            if observed_residues.setdefault(residue_index, residue_name) != residue_name:
                raise RuntimeError(
                    f"ESMFold2 prepared ligand residue is inconsistent for chain {chain_id}"
                )
            for atom_index in range(atom_start, atom_start + atom_count):
                if not 0 <= atom_index < atom_mask.shape[0]:
                    raise RuntimeError("ESMFold2 prepared ligand atom index is invalid")
                if not atom_mask[atom_index]:
                    continue
                atom_name = "".join(
                    chr(int(value) + 32)
                    for value in atom_name_chars[atom_index]
                    if int(value) != 0
                ).strip()
                if not atom_name:
                    raise RuntimeError("ESMFold2 prepared ligand atom name is empty")
                expected_rows.append((residue_index, residue_name, atom_name))
        if set(observed_residues) != set(range(len(ccd))):
            raise RuntimeError(
                f"ESMFold2 prepared ligand residue coverage differs for chain {chain_id}"
            )

        output_rows = [
            row
            for row, observed_chain in enumerate(label_asym)
            if str(observed_chain) == chain_id
        ]
        if len(output_rows) != len(expected_rows):
            raise RuntimeError(
                f"ESMFold2 ligand atom coverage differs for chain {chain_id}: "
                f"{len(output_rows)} != {len(expected_rows)}"
            )
        for row, (residue_index, component, expected_atom) in zip(
            output_rows, expected_rows
        ):
            if str(label_atom[row]) != expected_atom:
                raise RuntimeError(
                    f"ESMFold2 ligand atom order differs for chain {chain_id}: "
                    f"{label_atom[row]} != {expected_atom}"
                )
            position = str(residue_index + 1)
            rewritten_comp[row] = component
            rewritten_label_seq[row] = position
            rewritten_auth_seq[row] = position
            if rewritten_auth_comp is not None:
                rewritten_auth_comp[row] = component

    atom_site["label_comp_id"] = pdbx.CIFColumn(
        data=pdbx.CIFData(array=np.asarray(rewritten_comp, dtype=np.str_))
    )
    atom_site["label_seq_id"] = pdbx.CIFColumn(
        data=pdbx.CIFData(array=np.asarray(rewritten_label_seq, dtype=np.str_))
    )
    atom_site["auth_seq_id"] = pdbx.CIFColumn(
        data=pdbx.CIFData(array=np.asarray(rewritten_auth_seq, dtype=np.str_))
    )
    if rewritten_auth_comp is not None:
        atom_site["auth_comp_id"] = pdbx.CIFColumn(
            data=pdbx.CIFData(array=np.asarray(rewritten_auth_comp, dtype=np.str_))
        )


def _unbatch_feature_array(value: Any, np: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim < 2 or array.shape[0] != 1:
        raise RuntimeError("ESMFold2 prepared feature has an invalid batch shape")
    return array[0]


def _prepared_atom_inventory(features: dict[str, Any], np: Any) -> dict[str, Any]:
    """Materialize only the prepared atom labels needed for in-memory checks."""
    required = (
        "token_attention_mask",
        "atom_attention_mask",
        "asym_id",
        "residue_index",
        "atom_to_token",
        "ref_element",
        "ref_atom_name_chars",
    )
    missing = [name for name in required if name not in features]
    if missing:
        raise RuntimeError(
            "ESMFold2 prepared input lacks atom inventory fields: "
            + ", ".join(missing)
        )
    return {
        name: _unbatch_feature_array(features[name], np)
        for name in required
    }


def _validate_output_atom_inventory_in_memory(
    cif_text: str,
    sequences: list[dict[str, Any]],
    prepared: dict[str, Any],
) -> None:
    """Compare the current CIF atom rows with prepared tensors without dumping them."""
    try:
        from biotite.structure.io import pdbx
        from ..adapters.esmfold2 import _validate_output_atom_inventory

        cif_file = pdbx.CIFFile.read(io.StringIO(cif_text))
        blocks = list(cif_file.values())
        if len(blocks) != 1:
            raise RuntimeError(
                "ESMFold2 output atom inventory requires one CIF block"
            )
        atom_site = blocks[0]["atom_site"]
        columns = {
            name: atom_site[name].as_array(str)
            for name in (
                "label_asym_id",
                "label_seq_id",
                "label_comp_id",
                "label_atom_id",
                "type_symbol",
            )
        }
        _validate_output_atom_inventory(
            columns,
            sequences,
            None,
            raw_confidence_arrays=prepared,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"ESMFold2 in-memory output atom inventory validation failed: {exc}"
        ) from exc


def _cif_category(pdbx: Any, np: Any, rows: list[dict[str, str]]) -> Any:
    columns = {
        name: pdbx.CIFColumn(
            data=pdbx.CIFData(
                array=np.array([row[name] for row in rows], dtype=np.str_)
            )
        )
        for name in rows[0]
    }
    return pdbx.CIFCategory(columns=columns)


def _cif_rows(category: Any) -> tuple[list[str], list[dict[str, str]]]:
    """Extract a CIF category without discarding columns unknown to us."""

    names = [str(name) for name in category.keys()]
    if not names:
        return [], []
    arrays = {name: category[name].as_array(str) for name in names}
    row_count = len(next(iter(arrays.values())))
    if any(len(values) != row_count for values in arrays.values()):
        raise RuntimeError("ESMFold2 existing CIF topology category is ragged")
    rows = [
        {name: str(arrays[name][index]) for name in names}
        for index in range(row_count)
    ]
    return names, rows


def _normalized_topology_order(value: Any) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RuntimeError("ESMFold2 topology contains an empty bond order")
    if value == ".":
        raise RuntimeError("ESMFold2 topology contains an empty bond order")
    return value.upper()


def _merge_topology_order(
    existing: str,
    incoming: str,
    *,
    conflict_message: str,
) -> str:
    existing_normalized = _normalized_topology_order(existing)
    incoming_normalized = _normalized_topology_order(incoming)
    if existing_normalized == "?":
        return incoming
    if incoming_normalized == "?":
        return existing
    if existing_normalized != incoming_normalized:
        raise RuntimeError(conflict_message)
    # Use the incoming producer's spelling for equal concrete orders.  CCD
    # rows and explicit private links use different mmCIF vocabularies/case;
    # preserving the authoritative incoming representation keeps downstream
    # exact-order validation stable while still deduplicating the edge.
    return incoming


def _chem_bond_key(row: dict[str, str]) -> tuple[str, tuple[str, str]]:
    required = ("comp_id", "atom_id_1", "atom_id_2")
    if any(not row.get(name) or row[name] in {".", "?"} for name in required):
        raise RuntimeError("ESMFold2 chem_comp_bond contains an empty authoritative field")
    _normalized_topology_order(row.get("value_order"))
    atom_pair = tuple(sorted((row["atom_id_1"], row["atom_id_2"])))
    if atom_pair[0] == atom_pair[1]:
        raise RuntimeError("ESMFold2 chem_comp_bond contains a self-edge")
    return row["comp_id"], atom_pair


def _chem_bond_signature(row: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (name, "." if value == "" else value)
            for name, value in row.items()
            if name
            not in {"pdbx_ordinal", "atom_id_1", "atom_id_2", "value_order"}
        )
    )


def _merge_chem_comp_bond(
    block: Any,
    pdbx: Any,
    np: Any,
    incoming: list[dict[str, str]],
) -> None:
    """Merge CCD rows into an existing category with hard conflict checks."""

    if not incoming:
        return
    existing_columns: list[str]
    existing_rows: list[dict[str, str]]
    if "chem_comp_bond" in block:
        existing_columns, existing_rows = _cif_rows(block["chem_comp_bond"])
    else:
        existing_columns, existing_rows = [], []
    required = {"comp_id", "atom_id_1", "atom_id_2", "value_order"}
    if any(not required.issubset(row) for row in incoming):
        raise RuntimeError("ESMFold2 injected chem_comp_bond row is incomplete")

    columns = list(existing_columns)
    for row in incoming:
        for name in row:
            if name not in columns:
                columns.append(name)
    normalized_existing: list[dict[str, str]] = []
    seen: dict[
        tuple[str, tuple[str, str]], tuple[tuple[tuple[str, str], ...], int]
    ] = {}
    for raw in existing_rows:
        key = _chem_bond_key(raw)
        row = {name: raw.get(name, ".") for name in columns}
        signature = _chem_bond_signature(row)
        previous = seen.get(key)
        if previous is not None:
            previous_signature, previous_index = previous
            if previous_signature != signature:
                raise RuntimeError(
                    "ESMFold2 existing chem_comp_bond has a hard edge conflict"
                )
            normalized_existing[previous_index]["value_order"] = _merge_topology_order(
                normalized_existing[previous_index]["value_order"],
                row["value_order"],
                conflict_message=(
                    "ESMFold2 existing chem_comp_bond has a hard edge conflict"
                ),
            )
            # Exact chemistry duplicate: retain one authoritative row.
            continue
        seen[key] = (signature, len(normalized_existing))
        normalized_existing.append(row)

    used_ordinals: set[int] = set()
    for row in normalized_existing:
        raw_ordinal = row.get("pdbx_ordinal", ".")
        try:
            ordinal = int(raw_ordinal)
        except ValueError:
            continue
        if ordinal > 0:
            used_ordinals.add(ordinal)
    next_ordinal = max(used_ordinals, default=0) + 1
    merged = list(normalized_existing)
    for raw in incoming:
        row = {name: raw.get(name, ".") for name in columns}
        key = _chem_bond_key(row)
        signature = _chem_bond_signature(row)
        previous = seen.get(key)
        if previous is not None:
            previous_signature, previous_index = previous
            if previous_signature != signature:
                raise RuntimeError(
                    "ESMFold2 injected chem_comp_bond conflicts with existing topology"
                )
            merged[previous_index]["value_order"] = _merge_topology_order(
                merged[previous_index]["value_order"],
                row["value_order"],
                conflict_message=(
                    "ESMFold2 injected chem_comp_bond conflicts with existing topology"
                ),
            )
            continue
        if "pdbx_ordinal" in columns:
            while next_ordinal in used_ordinals:
                next_ordinal += 1
            row["pdbx_ordinal"] = str(next_ordinal)
            used_ordinals.add(next_ordinal)
            next_ordinal += 1
        seen[key] = (signature, len(merged))
        merged.append(row)
    if merged:
        block["chem_comp_bond"] = _cif_category(pdbx, np, merged)


_STRUCT_CONN_ENDPOINT_FIELDS = (
    "label_asym_id",
    "label_comp_id",
    "label_seq_id",
    "label_atom_id",
    "PDB_ins_code",
)


def _struct_conn_endpoint(row: dict[str, str], side: int) -> tuple[str, ...]:
    values = tuple(
        row.get(
            (
                f"pdbx_ptnr{side}_PDB_ins_code"
                if field == "PDB_ins_code"
                else f"ptnr{side}_{field}"
            ),
            ".",
        )
        for field in _STRUCT_CONN_ENDPOINT_FIELDS
    )
    # CIF writers commonly spell a missing insertion code as either an empty
    # string or a dot.  Treat those two spellings as one endpoint identity;
    # the four label fields above remain mandatory and are never guessed.
    values = tuple("." if value == "" else value for value in values)
    if any(value in {"", ".", "?"} for value in values[:4]):
        raise RuntimeError("ESMFold2 struct_conn contains an empty endpoint")
    return values


def _struct_conn_key(row: dict[str, str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    left = _struct_conn_endpoint(row, 1)
    right = _struct_conn_endpoint(row, 2)
    if left == right:
        raise RuntimeError("ESMFold2 struct_conn contains a self-edge")
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _struct_conn_signature(row: dict[str, str]) -> tuple[tuple[str, str], ...]:
    endpoint_fields: dict[int, list[tuple[str, str]]] = {1: [], 2: []}
    base_fields: list[tuple[str, str]] = []
    for name, value in row.items():
        if name in {"id", "pdbx_value_order"}:
            continue
        side: int | None = None
        suffix = name
        for prefix, candidate_side in (
            ("pdbx_ptnr1_", 1),
            ("pdbx_ptnr2_", 2),
            ("ptnr1_", 1),
            ("ptnr2_", 2),
        ):
            if name.startswith(prefix):
                side = candidate_side
                suffix = name[len(prefix) :]
                break
        normalized = "." if value == "" else value
        if side is None:
            base_fields.append((name, normalized))
        else:
            endpoint_fields[side].append((suffix, normalized))
    # Sort the complete endpoint-side field groups so that ptnr1/ptnr2
    # reversal produces the same signature, including optional symmetry or
    # insertion-code fields.
    endpoint_signature = tuple(
        sorted(tuple(sorted(fields)) for fields in endpoint_fields.values())
    )
    return tuple(sorted(base_fields + [("__endpoint_fields__", repr(endpoint_signature))]))


def _merge_struct_conn(
    block: Any,
    pdbx: Any,
    np: Any,
    incoming: list[dict[str, str]],
) -> None:
    """Merge explicit inter-component links while preserving existing links."""

    if not incoming:
        return
    if "struct_conn" in block:
        existing_columns, existing_rows = _cif_rows(block["struct_conn"])
    else:
        existing_columns, existing_rows = [], []
    required = {
        "id",
        "conn_type_id",
        "pdbx_value_order",
        "ptnr1_label_asym_id",
        "ptnr2_label_asym_id",
        "ptnr1_label_comp_id",
        "ptnr2_label_comp_id",
        "ptnr1_label_seq_id",
        "ptnr2_label_seq_id",
        "ptnr1_label_atom_id",
        "ptnr2_label_atom_id",
    }
    if any(not required.issubset(row) for row in incoming):
        raise RuntimeError("ESMFold2 injected struct_conn row is incomplete")
    columns = list(existing_columns)
    for row in incoming:
        for name in row:
            if name not in columns:
                columns.append(name)
    normalized_existing: list[dict[str, str]] = []
    seen: dict[
        tuple[tuple[str, ...], tuple[str, ...]], tuple[tuple[tuple[str, str], ...], int]
    ] = {}
    used_ids: set[str] = set()
    for raw in existing_rows:
        row = {name: raw.get(name, ".") for name in columns}
        row_id = row.get("id", ".")
        if row_id in used_ids:
            raise RuntimeError("ESMFold2 existing struct_conn contains duplicate IDs")
        used_ids.add(row_id)
        key = _struct_conn_key(row)
        signature = _struct_conn_signature(row)
        previous = seen.get(key)
        if previous is not None:
            previous_signature, previous_index = previous
            if previous_signature != signature:
                raise RuntimeError("ESMFold2 existing struct_conn has a hard edge conflict")
            normalized_existing[previous_index]["pdbx_value_order"] = _merge_topology_order(
                normalized_existing[previous_index]["pdbx_value_order"],
                row["pdbx_value_order"],
                conflict_message="ESMFold2 existing struct_conn has a hard edge conflict",
            )
            continue
        seen[key] = (signature, len(normalized_existing))
        normalized_existing.append(row)
    numeric_ids = [int(value) for value in used_ids if value.isdigit()]
    next_id = max(numeric_ids, default=0) + 1
    merged = list(normalized_existing)
    for raw in incoming:
        row = {name: raw.get(name, ".") for name in columns}
        key = _struct_conn_key(row)
        signature = _struct_conn_signature(row)
        previous = seen.get(key)
        if previous is not None:
            previous_signature, previous_index = previous
            if previous_signature != signature:
                raise RuntimeError(
                    "ESMFold2 injected struct_conn conflicts with existing topology"
                )
            merged[previous_index]["pdbx_value_order"] = _merge_topology_order(
                merged[previous_index]["pdbx_value_order"],
                row["pdbx_value_order"],
                conflict_message=(
                    "ESMFold2 injected struct_conn conflicts with existing topology"
                ),
            )
            continue
        candidate_id = str(next_id)
        while candidate_id in used_ids:
            next_id += 1
            candidate_id = str(next_id)
        row["id"] = candidate_id
        used_ids.add(candidate_id)
        next_id += 1
        seen[key] = (signature, len(merged))
        merged.append(row)
    if merged:
        block["struct_conn"] = _cif_category(pdbx, np, merged)


def _output_endpoint(block: Any, endpoint: dict[str, Any]) -> tuple[str, str]:
    atom_site = block["atom_site"]
    arrays = {
        name: atom_site[name].as_array(str)
        for name in (
            "label_asym_id",
            "label_comp_id",
            "label_seq_id",
            "auth_seq_id",
            "label_atom_id",
        )
    }
    chain_id = endpoint["chain_id"]
    residues: list[tuple[str, str, str]] = []
    rows_by_residue: dict[tuple[str, str, str], list[int]] = {}
    for row, observed_chain in enumerate(arrays["label_asym_id"]):
        if str(observed_chain) != chain_id:
            continue
        identity = (
            str(arrays["label_seq_id"][row]),
            str(arrays["auth_seq_id"][row]),
            str(arrays["label_comp_id"][row]),
        )
        if identity not in rows_by_residue:
            residues.append(identity)
            rows_by_residue[identity] = []
        rows_by_residue[identity].append(row)
    residue_index = endpoint["residue_index"]
    if not isinstance(residue_index, int) or not 0 <= residue_index < len(residues):
        raise RuntimeError(
            f"ESMFold2 output lacks covalent residue {chain_id}:{residue_index + 1}"
        )
    identity = residues[residue_index]
    atom_matches = [
        row
        for row in rows_by_residue[identity]
        if str(arrays["label_atom_id"][row]) == endpoint["atom_name"]
    ]
    if len(atom_matches) != 1:
        raise RuntimeError(
            "ESMFold2 output covalent atom did not resolve uniquely: "
            f"{chain_id}:{residue_index + 1}:{endpoint['atom_name']}"
        )
    label_seq_id, _auth_seq_id, comp_id = identity
    return comp_id, label_seq_id






def _validate_batch_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != _BATCH_SCHEMA_VERSION:
        raise RuntimeError("ESMFold2 batch request has an unsupported schema")
    mode = payload.get("mode")
    if mode not in {"fast", "standard"}:
        raise RuntimeError("ESMFold2 batch request has an invalid mode")
    batch_id = payload.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise RuntimeError("ESMFold2 batch request has an invalid batch identifier")

    raw_paths = payload.get("paths")
    if not isinstance(raw_paths, dict) or set(raw_paths) != {
        "fold_checkpoint",
        "esmc_checkpoint",
        "ccd",
        "ccd_bonds",
        "components_cif",
    }:
        raise RuntimeError("ESMFold2 batch model paths are incomplete")
    paths = {
        "fold_checkpoint": _safe_existing_directory(
            Path(raw_paths["fold_checkpoint"]), "ESMFold2 fold checkpoint"
        ),
        "esmc_checkpoint": _safe_existing_directory(
            Path(raw_paths["esmc_checkpoint"]), "ESMC checkpoint"
        ),
        "ccd": _safe_existing_file(Path(raw_paths["ccd"]), "ESMFold2 CCD"),
        "ccd_bonds": _safe_existing_file(
            Path(raw_paths["ccd_bonds"]), "ESMFold2 CCD bond cache"
        ),
        "components_cif": _safe_existing_file(
            Path(raw_paths["components_cif"]), "CCD components CIF"
        ),
    }
    common = paths["components_cif"].parent
    esmfold2_root = common.parent
    expected_paths = {
        "fold_checkpoint": esmfold2_root / "checkpoint" / mode,
        "esmc_checkpoint": esmfold2_root / "checkpoint" / "esmc-6b",
        "ccd": common / "ccd.pkl",
        "ccd_bonds": common / "ccd_bonds.sqlite3",
        "components_cif": common / "components.cif",
    }
    if any(paths[name] != expected for name, expected in expected_paths.items()):
        raise RuntimeError(
            "ESMFold2 batch paths do not match the canonical model asset layout"
        )
    driver_ccd_raw = os.environ.get("ESMCFOLD_CCD_PATH")
    if driver_ccd_raw is not None:
        driver_ccd = _safe_existing_file(
            Path(driver_ccd_raw), "ESMFold2 driver CCD"
        )
        if paths["ccd"] != driver_ccd:
            raise RuntimeError(
                "ESMFold2 batch CCD does not match ESMCFOLD_CCD_PATH"
            )
    config = payload.get("config")
    required_config = {
        "seed",
        "loops",
        "steps",
        "samples",
        "lm_dropout",
        "lm_mask_pct",
    }
    optional_config = {"esmc_attention_backend", "context_embedding_cache", "optimization"}
    if (
        not isinstance(config, dict)
        or not required_config <= set(config)
        or set(config) - (required_config | optional_config)
    ):
        raise RuntimeError("ESMFold2 batch sampling config is incomplete")
    # Backward-compatible for older batch payloads; public config now
    # always serializes this explicitly.
    config = {
        **config,
        "esmc_attention_backend": config.get("esmc_attention_backend", "sdpa"),
        "context_embedding_cache": config.get("context_embedding_cache", False),
    }
    if not all(
        isinstance(config[name], int) and not isinstance(config[name], bool)
        for name in ("seed", "loops", "steps", "samples")
    ):
        raise RuntimeError("ESMFold2 integer sampling config is malformed")
    if min(config["loops"], config["steps"], config["samples"]) < 1:
        raise RuntimeError("ESMFold2 loops, steps and samples must be positive")
    for name in ("lm_dropout", "lm_mask_pct"):
        if not isinstance(config[name], (int, float)) or isinstance(config[name], bool):
            raise RuntimeError(f"ESMFold2 {name} is malformed")
    if not 0 <= config["lm_dropout"] < 1 or not 0 <= config["lm_mask_pct"] <= 1:
        raise RuntimeError("ESMFold2 masking/dropout config is outside [0, 1]")
    if config["esmc_attention_backend"] not in {"sdpa", "flash_attention_2", "eager"}:
        raise RuntimeError("ESMFold2 attention backend is malformed")
    if config["context_embedding_cache"] not in {None, True, False}:
        raise RuntimeError("ESMFold2 context embedding cache flag is malformed")

    from ..esmfold2_optimization import validate_config
    if "optimization" in config:
        validate_config(config["optimization"])
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list) or not raw_requests:
        raise RuntimeError("ESMFold2 batch request is empty")
    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_requests:
        if not isinstance(raw, dict) or set(raw) != {
            "request_id",
            "input_json",
            "output_dir",
            "expected_prefix",
            "expected_samples",
        }:
            raise RuntimeError("ESMFold2 batch request item is malformed")
        request_id = raw["request_id"]
        if not isinstance(request_id, str) or not request_id or request_id in seen:
            raise RuntimeError("ESMFold2 batch request IDs are invalid or duplicate")
        seen.add(request_id)
        expected_prefix = raw["expected_prefix"]
        if (
            not isinstance(expected_prefix, str)
            or not expected_prefix
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in expected_prefix)
        ):
            raise RuntimeError("ESMFold2 expected output prefix is unsafe")
        expected_samples = raw["expected_samples"]
        if expected_samples != config["samples"]:
            raise RuntimeError("ESMFold2 request sample count disagrees with batch config")
        requests.append(
            {
                **raw,
                "input_json": _safe_existing_file(
                    Path(raw["input_json"]), "ESMFold2 request input"
                ),
                "output_dir": Path(raw["output_dir"]).expanduser().resolve(),
            }
        )
    return {
        "batch_id": batch_id,
        "mode": mode,
        "paths": paths,
        "config": config,
        "requests": requests,
        "result_json": Path(payload["result_json"]).expanduser().resolve(),
    }


def _safe_existing_file(path: Path, description: str) -> Path:
    raw = Path(path).expanduser()
    _reject_symlink_components(raw, description)
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"{description} is missing or unsafe: {raw}")
    return raw.resolve()


def _safe_existing_directory(path: Path, description: str) -> Path:
    raw = Path(path).expanduser()
    _reject_symlink_components(raw, description)
    if raw.is_symlink() or not raw.is_dir():
        raise RuntimeError(f"{description} is missing or unsafe: {raw}")
    return raw.resolve()


def _reject_symlink_components(path: Path, description: str) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise RuntimeError(
                f"{description} is missing or unsafe: {path} "
                f"(symlink component: {component})"
            )


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _error_text(exc: BaseException) -> str:
    message = str(exc).strip() or type(exc).__name__
    return message[-8000:]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipetools.workers.esmfold2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    batch = subparsers.add_parser("batch")
    batch.add_argument("--request-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "batch":
        run_batch(args.request_json)
        return
    raise RuntimeError(f"Unsupported ESMFold2 worker command {args.command!r}")


if __name__ == "__main__":
    main()
