"""Lossless, inference-only PartiDesign checkpoint bundles.

The bundle keeps one shared autoencoder state and one inference-model state per
route.  It deliberately drops training-only checkpoint members and retains
only the configuration and optional contract metadata needed to select an
ordinary combined checkpoint for the existing loader.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


BUNDLE_VERSION_KEY = "partidesign_bundle_version"
LEGACY_BUNDLE_MEMBERS = ("omni", "prot", "nuc", "ligand", "antibody")
BUNDLE_MEMBERS = (*LEGACY_BUNDLE_MEMBERS, "pep")
LEGACY_BUNDLE_VERSION = 1
BUNDLE_VERSION = 2
_BUNDLE_MEMBERS_BY_VERSION = {
    LEGACY_BUNDLE_VERSION: LEGACY_BUNDLE_MEMBERS,
    BUNDLE_VERSION: BUNDLE_MEMBERS,
}
_CHECKPOINT_CONTRACT_KEY = "partidesign_checkpoint"
_AUTOENCODER_KEYS = frozenset(("state_dict",))
_MODEL_KEYS = frozenset(("state_dict", "hyper_parameters", _CHECKPOINT_CONTRACT_KEY))
_CFG_EXP_KEYS = ("network", "flow", "sampling")
_ABI_METADATA_KEYS = (
    "schema_version",
    "context_constraint_mode",
    "unindexed_motif_enabled",
    "unindexed_motif_max_tokens",
)
_BUNDLE_CONTRACT_KEYS = frozenset(
    (*_ABI_METADATA_KEYS, "checkpoint_kind", "complete_training_resume")
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return value


def _state_mapping(value: Any, label: str, prefix: str) -> Mapping[str, torch.Tensor]:
    state = _mapping(value, label)
    if not state:
        raise ValueError(f"{label} must not be empty")
    for key, tensor in state.items():
        if not key.startswith(prefix):
            raise ValueError(f"{label} key {key!r} must start with {prefix!r}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label}[{key!r}] must be a tensor")
    return state


def _validate_bundle_schema(document: Any) -> Mapping[str, Any]:
    bundle = _mapping(document, "bundle")
    version = bundle.get(BUNDLE_VERSION_KEY)
    if type(version) is not int:
        raise TypeError(f"{BUNDLE_VERSION_KEY} must be an integer")
    members = _BUNDLE_MEMBERS_BY_VERSION.get(version)
    if members is None:
        raise ValueError(
            f"{BUNDLE_VERSION_KEY} must be exactly 1 or 2, got {version!r}"
        )
    expected_root_keys = frozenset((BUNDLE_VERSION_KEY, "autoencoder", *members))
    if set(bundle) != expected_root_keys:
        raise ValueError(
            "bundle root keys must be exactly "
            f"{sorted(expected_root_keys)} for version {version}"
        )

    autoencoder = _mapping(bundle["autoencoder"], "bundle.autoencoder")
    if set(autoencoder) != _AUTOENCODER_KEYS:
        raise ValueError("bundle.autoencoder keys must be exactly ['state_dict']")
    _state_mapping(
        autoencoder["state_dict"],
        "bundle.autoencoder.state_dict",
        "autoencoder.",
    )

    for member in members:
        payload = _mapping(bundle[member], f"bundle.{member}")
        if set(payload) - _MODEL_KEYS or "state_dict" not in payload or "hyper_parameters" not in payload:
            raise ValueError(
                f"bundle.{member} keys must contain state_dict and hyper_parameters "
                f"and may contain only {_CHECKPOINT_CONTRACT_KEY}"
            )
        _state_mapping(payload["state_dict"], f"bundle.{member}.state_dict", "nn.")
        hyper_parameters = _mapping(
            payload["hyper_parameters"],
            f"bundle.{member}.hyper_parameters",
        )
        if set(hyper_parameters) != {"cfg_exp"}:
            raise ValueError(
                f"bundle.{member}.hyper_parameters keys must be exactly ['cfg_exp']"
            )
        cfg_exp = _mapping(
            hyper_parameters["cfg_exp"],
            f"bundle.{member}.hyper_parameters.cfg_exp",
        )
        if set(cfg_exp) != set(_CFG_EXP_KEYS):
            raise ValueError(
                f"bundle.{member}.hyper_parameters.cfg_exp keys must be exactly "
                f"{list(_CFG_EXP_KEYS)}"
            )
        for key in _CFG_EXP_KEYS:
            _mapping(cfg_exp[key], f"bundle.{member}.hyper_parameters.cfg_exp.{key}")
        if _CHECKPOINT_CONTRACT_KEY in payload:
            metadata = _mapping(
                payload[_CHECKPOINT_CONTRACT_KEY],
                f"bundle.{member}.{_CHECKPOINT_CONTRACT_KEY}",
            )
            if (
                set(metadata) - _BUNDLE_CONTRACT_KEYS
                or "checkpoint_kind" not in metadata
                or "complete_training_resume" not in metadata
            ):
                raise ValueError(
                    f"bundle.{member}.{_CHECKPOINT_CONTRACT_KEY} contains unsupported "
                    "or missing inference metadata keys"
                )
            if metadata["checkpoint_kind"] != "inference":
                raise ValueError(
                    f"bundle.{member}.{_CHECKPOINT_CONTRACT_KEY}.checkpoint_kind "
                    "must be inference"
                )
            if metadata["complete_training_resume"] is not False:
                raise ValueError(
                    f"bundle.{member}.{_CHECKPOINT_CONTRACT_KEY}.complete_training_resume "
                    "must be false"
                )
    return bundle


def is_partidesign_bundle(document: Any) -> bool:
    """Return whether *document* declares the PartiDesign bundle marker.

    This is deliberately a marker check rather than a schema validator.  A
    malformed marked document must fail closed in ``select_bundle_checkpoint``
    instead of being silently reinterpreted as a legacy combined checkpoint.
    """

    return isinstance(document, Mapping) and BUNDLE_VERSION_KEY in document


def select_bundle_checkpoint(document: Mapping[str, Any], member: str) -> dict[str, Any]:
    """Materialize one normal route as an ordinary combined checkpoint mapping.

    The returned mapping has the exact ``state_dict``/``hyper_parameters``
    shape consumed by the combined checkpoint loader.  It is a fresh shallow
    container; tensor values are shared with the loaded bundle document and
    are never converted or copied.
    """

    if member not in BUNDLE_MEMBERS:
        raise ValueError(f"bundle member must be one of {BUNDLE_MEMBERS}")
    bundle = _validate_bundle_schema(document)
    members = _BUNDLE_MEMBERS_BY_VERSION[bundle[BUNDLE_VERSION_KEY]]
    if member not in members:
        raise ValueError(
            f"bundle version {bundle[BUNDLE_VERSION_KEY]} does not include {member!r}; "
            "append the member or build a version-2 six-member bundle"
        )
    autoencoder_state = bundle["autoencoder"]["state_dict"]
    member_payload = bundle[member]
    member_state = member_payload["state_dict"]
    state_dict = dict(member_state)
    state_dict.update(autoencoder_state)
    selected: dict[str, Any] = {
        "state_dict": state_dict,
        "hyper_parameters": copy.deepcopy(member_payload["hyper_parameters"]),
    }
    if _CHECKPOINT_CONTRACT_KEY in member_payload:
        selected[_CHECKPOINT_CONTRACT_KEY] = copy.deepcopy(
            member_payload[_CHECKPOINT_CONTRACT_KEY]
        )
    return selected


def _load_checkpoint(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        document = torch.load(
            str(path),
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except Exception as error:
        raise RuntimeError(f"failed to read {label}: {path}") from error
    return _mapping(document, label)


def _checkpoint_helpers() -> tuple[Any, Any, Any]:
    # Keep this import lazy: checkpoint.py imports the inference model and the
    # bundle module must remain importable while that loader is being extended.
    from .checkpoint import (  # pylint: disable=import-outside-toplevel
        _meta_model,
        _strict_assign_state,
        _validated_checkpoint_contract,
    )

    return _validated_checkpoint_contract, _meta_model, _strict_assign_state


def _split_state(
    state: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    state = _mapping(state, label)
    nn_state: dict[str, torch.Tensor] = {}
    autoencoder_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label}[{key!r}] must be a tensor")
        if key.startswith("nn."):
            nn_state[key] = value
        elif key.startswith("autoencoder."):
            autoencoder_state[key] = value
        else:
            raise ValueError(
                f"{label} contains unsupported state root for key {key!r}"
            )
    if not nn_state or not autoencoder_state:
        raise ValueError(f"{label} must contain both nn.* and autoencoder.* tensors")
    return nn_state, autoencoder_state


def _sanitize_cfg_exp(cfg_exp: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Retain only the three configuration trees consumed by inference."""

    cfg_exp = _mapping(cfg_exp, label)
    missing = [key for key in _CFG_EXP_KEYS if key not in cfg_exp]
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")
    return {
        key: copy.deepcopy(
            _mapping(cfg_exp[key], f"{label}.{key}")
        )
        for key in _CFG_EXP_KEYS
    }


def _sanitize_contract(metadata: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Keep inference ABI metadata while dropping training provenance."""

    metadata = _mapping(metadata, label)
    sanitized = {
        key: copy.deepcopy(metadata[key])
        for key in _ABI_METADATA_KEYS
        if key in metadata
    }
    sanitized["checkpoint_kind"] = "inference"
    sanitized["complete_training_resume"] = False
    return sanitized


def _assert_shared_autoencoder(
    autoencoders: Mapping[str, Mapping[str, torch.Tensor]],
    members: tuple[str, ...] | None = None,
) -> Mapping[str, torch.Tensor]:
    members = BUNDLE_MEMBERS if members is None else tuple(members)
    if not members:
        raise ValueError("bundle must contain at least one member")
    reference_member = members[0]
    reference = autoencoders[reference_member]
    reference_keys = set(reference)
    for member in members[1:]:
        current = autoencoders[member]
        if set(current) != reference_keys:
            raise ValueError(
                "shared autoencoder keys differ between "
                f"{reference_member} and {member}"
            )
        for key in sorted(reference_keys):
            expected = reference[key]
            actual = current[key]
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                raise ValueError(
                    f"shared autoencoder tensor differs in shape or dtype: {key}"
                )
            if not torch.equal(expected, actual):
                raise ValueError(f"shared autoencoder tensor values differ: {key}")
    return reference


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _compare_state_tensors(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    label: str,
) -> None:
    if set(expected) != set(actual):
        raise ValueError(f"temporary bundle changed {label} tensor keys")
    for key in expected:
        expected_tensor = expected[key]
        actual_tensor = actual[key]
        if (
            not isinstance(expected_tensor, torch.Tensor)
            or not isinstance(actual_tensor, torch.Tensor)
            or expected_tensor.shape != actual_tensor.shape
            or expected_tensor.dtype != actual_tensor.dtype
            or not torch.equal(expected_tensor, actual_tensor)
        ):
            raise ValueError(f"temporary bundle changed tensor {label}.{key}")


def _validate_temporary_bundle(
    expected: Mapping[str, Any],
    reloaded: Mapping[str, Any],
) -> None:
    """Validate the complete serialized bundle before publication."""

    _validate_bundle_schema(reloaded)
    members = _BUNDLE_MEMBERS_BY_VERSION[reloaded[BUNDLE_VERSION_KEY]]
    for member in members:
        # Exercise the exact combined-checkpoint view for every branch, not
        # just the root envelope.
        select_bundle_checkpoint(reloaded, member)

    if set(expected) != set(reloaded):
        raise ValueError("temporary bundle changed root keys")
    _compare_state_tensors(
        expected["autoencoder"]["state_dict"],
        reloaded["autoencoder"]["state_dict"],
        "autoencoder",
    )
    for member in members:
        expected_payload = expected[member]
        actual_payload = reloaded[member]
        _compare_state_tensors(
            expected_payload["state_dict"],
            actual_payload["state_dict"],
            f"{member}.state_dict",
        )
        if expected_payload["hyper_parameters"] != actual_payload["hyper_parameters"]:
            raise ValueError(f"temporary bundle changed {member} hyper_parameters")
        if expected_payload.get(_CHECKPOINT_CONTRACT_KEY) != actual_payload.get(_CHECKPOINT_CONTRACT_KEY):
            raise ValueError(f"temporary bundle changed {member} contract metadata")


def _atomic_publish(
    document: Mapping[str, Any],
    destination: Path,
    *,
    replace: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            torch.save(document, handle)
            handle.flush()
            os.fsync(handle.fileno())

        reloaded = _load_checkpoint(temporary, "temporary PartiDesign bundle")
        _validate_temporary_bundle(document, reloaded)

        if replace:
            os.replace(temporary, destination)
        else:
            # A hard link publishes without replacing a destination created by
            # a concurrent writer.  Both paths are in the same directory/filesystem.
            os.link(temporary, destination)
        _fsync_directory(destination.parent)
        if temporary.exists():
            temporary.unlink()
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _prepare_member_payload(
    member: str,
    source_path: Path,
    validate_contract: Any,
    meta_model: Any,
    strict_assign: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    source = _load_checkpoint(source_path, f"{member} source checkpoint")
    validated_state, network_config, context_mode = validate_contract(
        source,
        requested_mode="auto",
    )
    validated_state = _mapping(validated_state, f"{member} validated state_dict")
    # Force the existing strict inference ABI check before anything can be
    # published.  The helper constructs on meta, so this does not allocate a
    # real model-sized parameter set on CPU.
    model = meta_model(network_config, context_mode)
    strict_assign(model, validated_state)
    if member == "antibody":
        # The antibody template feature contract is separate from the
        # shared FM contract and must be checked before publication.
        from ..antibody.checkpoint import (  # pylint: disable=import-outside-toplevel
            validate_antibody_checkpoint_metadata,
        )

        validate_antibody_checkpoint_metadata(source)
    nn_state, autoencoder_state = _split_state(
        validated_state,
        f"{member} validated state_dict",
    )

    hyper_parameters = _mapping(
        source.get("hyper_parameters"),
        f"{member} source hyper_parameters",
    )
    cfg_exp = _mapping(
        hyper_parameters.get("cfg_exp"),
        f"{member} source hyper_parameters.cfg_exp",
    )
    payload: dict[str, Any] = {
        "state_dict": nn_state,
        "hyper_parameters": {
            "cfg_exp": _sanitize_cfg_exp(
                cfg_exp,
                f"{member} source hyper_parameters.cfg_exp",
            )
        },
    }
    if _CHECKPOINT_CONTRACT_KEY in source:
        payload[_CHECKPOINT_CONTRACT_KEY] = _sanitize_contract(
            _mapping(
                source[_CHECKPOINT_CONTRACT_KEY],
                f"{member} source {_CHECKPOINT_CONTRACT_KEY}",
            ),
            f"{member} source {_CHECKPOINT_CONTRACT_KEY}",
        )
    return payload, autoencoder_state


def build_checkpoint_bundle(
    sources: Mapping[str, Path],
    destination: Path,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    """Build and atomically publish a lossless inference-only bundle.

    ``sources`` must contain exactly one checkpoint path for every bundle
    member.  Existing destinations are never overwritten unless ``replace``
    is explicitly true.
    """

    if type(replace) is not bool:
        raise TypeError("replace must be a bool")
    source_mapping = _mapping(sources, "sources")
    source_keys = set(source_mapping)
    if source_keys == set(LEGACY_BUNDLE_MEMBERS):
        members = LEGACY_BUNDLE_MEMBERS
        bundle_version = LEGACY_BUNDLE_VERSION
    elif source_keys == set(BUNDLE_MEMBERS):
        members = BUNDLE_MEMBERS
        bundle_version = BUNDLE_VERSION
    else:
        raise ValueError(
            "sources keys must be exactly the five legacy members or all six "
            f"members: {LEGACY_BUNDLE_MEMBERS} or {BUNDLE_MEMBERS}"
        )
    source_paths = {
        member: Path(source_mapping[member]).expanduser().resolve()
        for member in members
    }
    destination_path = Path(destination).expanduser().resolve()
    if destination_path in set(source_paths.values()):
        raise ValueError("bundle output must be distinct from every source checkpoint")
    if destination_path.exists() and not replace:
        raise FileExistsError(f"bundle output already exists: {destination_path}")

    validate_contract, meta_model, strict_assign = _checkpoint_helpers()
    autoencoder_states: dict[str, dict[str, torch.Tensor]] = {}
    member_payloads: dict[str, dict[str, Any]] = {}

    for member in members:
        payload, autoencoder_state = _prepare_member_payload(
            member,
            source_paths[member],
            validate_contract,
            meta_model,
            strict_assign,
        )
        autoencoder_states[member] = autoencoder_state
        member_payloads[member] = payload

    shared_autoencoder = _assert_shared_autoencoder(autoencoder_states, members)
    document: dict[str, Any] = {
        BUNDLE_VERSION_KEY: bundle_version,
        "autoencoder": {"state_dict": dict(shared_autoencoder)},
    }
    document.update(member_payloads)
    _validate_bundle_schema(document)
    _atomic_publish(document, destination_path, replace=replace)

    return {
        "source_bytes": {
            member: int(source_paths[member].stat().st_size)
            for member in members
        },
        "output_bytes": int(destination_path.stat().st_size),
    }


def append_pep_to_bundle(
    base_bundle: Path,
    pep_source: Path,
    destination: Path,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    """Append a validated pep checkpoint to an existing version-1 bundle.

    The five legacy member payloads are retained as loaded.  Only the bundle
    version and new pep payload are added; the shared autoencoder must compare
    exactly before the atomic publisher is called.
    """

    if type(replace) is not bool:
        raise TypeError("replace must be a bool")
    base_path = Path(base_bundle).expanduser().resolve()
    pep_path = Path(pep_source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if destination_path in {base_path, pep_path}:
        raise ValueError("bundle output must be distinct from every input")
    if destination_path.exists() and not replace:
        raise FileExistsError(f"bundle output already exists: {destination_path}")

    base_document = _load_checkpoint(base_path, "base PartiDesign bundle")
    if not is_partidesign_bundle(base_document):
        raise ValueError("base bundle must declare the PartiDesign bundle marker")
    _validate_bundle_schema(base_document)
    if base_document[BUNDLE_VERSION_KEY] != LEGACY_BUNDLE_VERSION:
        raise ValueError(
            "pep append requires an existing version-1 five-member bundle"
        )

    validate_contract, meta_model, strict_assign = _checkpoint_helpers()
    pep_payload, pep_autoencoder = _prepare_member_payload(
        "pep",
        pep_path,
        validate_contract,
        meta_model,
        strict_assign,
    )
    base_autoencoder = base_document["autoencoder"]["state_dict"]
    _assert_shared_autoencoder(
        {"base": base_autoencoder, "pep": pep_autoencoder},
        ("base", "pep"),
    )

    document = dict(base_document)
    document[BUNDLE_VERSION_KEY] = BUNDLE_VERSION
    document["pep"] = pep_payload
    _validate_bundle_schema(document)
    _atomic_publish(document, destination_path, replace=replace)
    return {
        "source_bytes": {
            "base_bundle": int(base_path.stat().st_size),
            "pep": int(pep_path.stat().st_size),
        },
        "output_bytes": int(destination_path.stat().st_size),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for member in LEGACY_BUNDLE_MEMBERS:
        parser.add_argument(
            f"--{member}",
            type=Path,
            help=f"{member} source checkpoint (required for a direct five/six-member build)",
        )
    parser.add_argument(
        "--pep",
        type=Path,
        help="pep source checkpoint; optional for legacy five-member builds",
    )
    parser.add_argument(
        "--base-bundle",
        type=Path,
        help="existing version-1 five-member bundle to extend with --pep",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="atomically replace an existing output bundle",
    )
    args = parser.parse_args()
    legacy_sources = {
        member: getattr(args, member)
        for member in LEGACY_BUNDLE_MEMBERS
    }
    if args.base_bundle is not None:
        if args.pep is None:
            parser.error("--base-bundle requires --pep")
        conflicts = [member for member, value in legacy_sources.items() if value is not None]
        if conflicts:
            parser.error(
                "--base-bundle/--pep cannot be combined with source options: "
                + ", ".join(f"--{member}" for member in conflicts)
            )
    else:
        missing = [member for member, value in legacy_sources.items() if value is None]
        if missing:
            parser.error(
                "a direct build requires all legacy sources: "
                + ", ".join(f"--{member}" for member in missing)
            )
    return args


def main() -> None:
    args = _parse_args()
    if args.base_bundle is not None:
        stats = append_pep_to_bundle(
            args.base_bundle,
            args.pep,
            args.output,
            replace=args.replace,
        )
    else:
        sources = {
            member: getattr(args, member)
            for member in LEGACY_BUNDLE_MEMBERS
        }
        if args.pep is not None:
            sources["pep"] = args.pep
        stats = build_checkpoint_bundle(
            sources,
            args.output,
            replace=args.replace,
        )
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
