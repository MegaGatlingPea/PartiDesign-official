from __future__ import annotations

import math
import re
from pathlib import Path
from dataclasses import replace
from typing import Any, Mapping, Sequence

from .omission import normalize_rm_token_type, normalize_task_rm_token_type
from .records import (
    CropProximitySpec,
    CropSpec,
    DesignChainSpec,
    DesignConfigError,
    ExpandedRequest,
    LengthPlan,
    PartialSpec,
    TaskSpec,
)


_TASK_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_MODALITIES = {"protein", "rna", "dna", "ligand"}
_AUTHORING_SUB_MODALITIES = {"peptide"}
_TASK_KEYS = {
    "input", "sample_num", "design_chains", "design_length", "condition", "hotspot", "crop", "score", "postprocess", "override",
    "cyclic_offset", "cyclic_offset_chains", "format", "rm_token_type", "partial",
}


def _closed(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        raise DesignConfigError(f"unknown {where} keys: {unknown}")


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DesignConfigError(f"{where} must be a positive integer")
    return int(value)


def _resolve_local_path(value: Any, invocation_dir: Path, where: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise DesignConfigError(f"{where} must be a non-empty path string")
    if "://" in value:
        raise DesignConfigError(f"{where} does not accept a URI")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = invocation_dir / path
    return path.resolve()


def _parse_design_chains(value: Any, task_name: str) -> tuple[DesignChainSpec, ...]:
    if not isinstance(value, list) or not value:
        raise DesignConfigError(f"task {task_name!r} design_chains must be a non-empty list")
    result: list[DesignChainSpec] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or len(item) != 1:
            raise DesignConfigError(
                f"task {task_name!r} design_chains[{index}] must be a one-entry mapping"
            )
        chain_id, modality = next(iter(item.items()))
        if not isinstance(chain_id, str) or not chain_id or len(chain_id) > 32:
            raise DesignConfigError(f"task {task_name!r} has an invalid design chain ID")
        if not isinstance(modality, str) or (
            modality not in _MODALITIES and modality not in _AUTHORING_SUB_MODALITIES
        ):
            raise DesignConfigError(
                f"task {task_name!r} chain {chain_id!r} has invalid modality {modality!r}"
            )
        if modality == "peptide":
            result.append(DesignChainSpec(chain_id, "protein", "peptide"))
        else:
            result.append(DesignChainSpec(chain_id, modality))
    ids = [item.chain_id for item in result]
    if len(ids) != len(set(ids)):
        raise DesignConfigError(f"task {task_name!r} repeats a design chain")
    return tuple(result)


def _validate_public_smiles(condition: Mapping[str, Any], task_name: str) -> None:
    ligand = condition.get("ligand")
    if ligand is None:
        return
    if not isinstance(ligand, Mapping):
        raise DesignConfigError(f"task {task_name!r} condition.ligand must be a mapping")
    for chain_id, raw_entry in ligand.items():
        if not isinstance(raw_entry, Mapping):
            raise DesignConfigError(
                f"task {task_name!r} condition.ligand.{chain_id} must be a mapping"
            )
        if raw_entry.get("mode") == "ccd":
            component_id = raw_entry.get("ccd")
            if (set(raw_entry) != {"mode", "ccd"} or not isinstance(component_id, str)
                    or not component_id or not component_id.isascii()
                    or not component_id.isalnum() or component_id != component_id.upper()):
                raise DesignConfigError("CCD ligand requires exactly mode and an uppercase alphanumeric ccd ID")
            continue
        if raw_entry.get("mode") != "smiles":
            continue
        if set(raw_entry) != {"mode", "smiles"} or not isinstance(
            raw_entry.get("smiles"), str
        ):
            raise DesignConfigError(
                f"task {task_name!r} SMILES ligand requires exactly mode and smiles"
            )
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise DesignConfigError("SMILES task preflight requires RDKit") from exc
        molecule = Chem.MolFromSmiles(raw_entry["smiles"])
        if molecule is None:
            raise DesignConfigError(
                f"task {task_name!r} has an invalid SMILES for chain {chain_id!r}"
            )
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise DesignConfigError(
                f"task {task_name!r} SMILES for chain {chain_id!r} "
                "must contain exactly one connected component"
            )


def _validate_contig_length_bounds(
    condition: Mapping[str, Any], task_name: str
) -> None:
    """Validate optional final-length windows before any model/output side effect."""

    contig = condition.get("contig")
    if contig is None:
        return
    if not isinstance(contig, Mapping):
        raise DesignConfigError(f"task {task_name!r} condition.contig must be a mapping")
    allowed = {"token", "polymer_atom", "motif_min_length", "motif_max_length"}
    for chain_id, entry in contig.items():
        if not isinstance(chain_id, str) or not chain_id:
            raise DesignConfigError(f"task {task_name!r} condition.contig has an invalid chain ID")
        if not isinstance(entry, Mapping):
            raise DesignConfigError(
                f"task {task_name!r} condition.contig.{chain_id} must be a mapping"
            )
        _closed(entry, allowed, f"task {task_name!r} condition.contig.{chain_id}")
        has_min = "motif_min_length" in entry
        has_max = "motif_max_length" in entry
        if has_min != has_max:
            raise DesignConfigError(
                f"task {task_name!r} condition.contig.{chain_id} requires both "
                "motif_min_length and motif_max_length"
            )
        if not has_min:
            continue
        minimum = _positive_int(
            entry["motif_min_length"],
            f"task {task_name!r} condition.contig.{chain_id}.motif_min_length",
        )
        maximum = _positive_int(
            entry["motif_max_length"],
            f"task {task_name!r} condition.contig.{chain_id}.motif_max_length",
        )
        if minimum > maximum:
            raise DesignConfigError(
                f"task {task_name!r} condition.contig.{chain_id} motif length bounds "
                "must form an inclusive interval"
            )


def _grid_values(value: Any, where: str) -> list[int]:
    if isinstance(value, list):
        if not value:
            raise DesignConfigError(f"{where} list must be non-empty")
        values = [_positive_int(item, f"{where} entry") for item in value]
        if len(values) != len(set(values)):
            raise DesignConfigError(f"{where} contains duplicate lengths")
        return values
    if isinstance(value, Mapping):
        _closed(value, {"min", "max", "step"}, where)
        if set(value) != {"min", "max", "step"}:
            raise DesignConfigError(f"{where} grid requires min, max, and step")
        minimum = _positive_int(value["min"], f"{where}.min")
        maximum = _positive_int(value["max"], f"{where}.max")
        step = _positive_int(value["step"], f"{where}.step")
        if minimum > maximum or (maximum - minimum) % step != 0:
            raise DesignConfigError(f"{where} must be an inclusive aligned grid")
        return list(range(minimum, maximum + 1, step))
    raise DesignConfigError(f"{where} must be a list or inclusive grid")


def _distribution_chain(
    value: Any,
    design_chains: tuple[DesignChainSpec, ...],
    task_name: str,
) -> str | None:
    """Return the one chain selected by the public distribution sentinel.

    The sentinel is deliberately parsed at task level, before ordinary length
    plans are expanded. It is not an integer plan and is resolved by the API
    after request expansion, so every Best-of-N replica receives the same
    sampled base-request length.
    """

    if not isinstance(value, Mapping):
        return None
    sentinel_chains = [
        str(chain_id)
        for chain_id, raw_length in value.items()
        if raw_length == "from_distribution"
    ]
    if not sentinel_chains:
        return None
    if len(value) != 1 or len(sentinel_chains) != 1:
        raise DesignConfigError(
            f"task {task_name!r} design_length.from_distribution requires exactly "
            "one design ligand chain"
        )
    chain_id = sentinel_chains[0]
    if len(design_chains) != 1:
        raise DesignConfigError(
            f"task {task_name!r} design_length.from_distribution does not support "
            "multiple design chains"
        )
    chain = design_chains[0]
    if chain.chain_id != chain_id or chain.modality != "ligand":
        raise DesignConfigError(
            f"task {task_name!r} design_length.from_distribution requires exactly "
            "one ligand design chain"
        )
    return chain_id


def _validate_distribution_conditions(
    condition: Mapping[str, Any],
    chain_id: str,
    task_name: str,
) -> None:
    """Reject conditions whose ligand atom inventory cannot be sampled."""

    ligand = condition.get("ligand")
    if isinstance(ligand, Mapping) and chain_id in ligand:
        entry = ligand[chain_id]
        if isinstance(entry, Mapping):
            mode = entry.get("mode")
            if mode in {"smiles", "ccd", "type_bond", "backbone"}:
                raise DesignConfigError(
                    f"task {task_name!r} design_length.from_distribution is "
                    f"incompatible with fixed ligand mode {mode!r}"
                )

    backbone = condition.get("backbone")
    if isinstance(backbone, (list, tuple, set, frozenset)) and chain_id in backbone:
        raise DesignConfigError(
            f"task {task_name!r} design_length.from_distribution is incompatible "
            "with a fixed ligand backbone"
        )


def _parse_selector_list(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise DesignConfigError(f"{where} must be a non-empty list")
    selectors: list[str] = []
    for index, raw_selector in enumerate(value):
        if not isinstance(raw_selector, str) or not raw_selector.strip():
            raise DesignConfigError(
                f"{where}[{index}] must be a non-empty selector string"
            )
        selector = raw_selector.strip()
        if selector in selectors:
            raise DesignConfigError(f"{where} contains duplicate selector {selector!r}")
        selectors.append(selector)
    return tuple(selectors)


def _parse_crop(value: Any, task_name: str) -> CropSpec | None:
    if value is None:
        return None
    where = f"task {task_name!r} crop"
    if not isinstance(value, Mapping) or not value:
        raise DesignConfigError(f"{where} must be a non-empty mapping")
    _closed(value, {"include", "proximity", "exclude"}, where)

    include = (
        _parse_selector_list(value["include"], f"{where}.include")
        if "include" in value
        else None
    )
    exclude = (
        _parse_selector_list(value["exclude"], f"{where}.exclude")
        if "exclude" in value
        else ()
    )

    proximity_specs: list[CropProximitySpec] = []
    if "proximity" in value:
        raw_proximity = value["proximity"]
        if not isinstance(raw_proximity, list) or not raw_proximity:
            raise DesignConfigError(f"{where}.proximity must be a non-empty list")
        for index, raw_entry in enumerate(raw_proximity):
            entry_where = f"{where}.proximity[{index}]"
            if not isinstance(raw_entry, Mapping):
                raise DesignConfigError(f"{entry_where} must be a mapping")
            _closed(raw_entry, {"anchor", "radius"}, entry_where)
            if set(raw_entry) != {"anchor", "radius"}:
                raise DesignConfigError(
                    f"{entry_where} requires exactly anchor and radius"
                )
            raw_anchor = raw_entry["anchor"]
            if not isinstance(raw_anchor, str) or not raw_anchor.strip():
                raise DesignConfigError(
                    f"{entry_where}.anchor must be a non-empty selector string"
                )
            anchor = raw_anchor.strip()
            raw_radius = raw_entry["radius"]
            if isinstance(raw_radius, bool) or not isinstance(raw_radius, (int, float)):
                raise DesignConfigError(
                    f"{entry_where}.radius must be a finite positive number"
                )
            try:
                radius = float(raw_radius)
            except (OverflowError, ValueError) as exc:
                raise DesignConfigError(
                    f"{entry_where}.radius must be a finite positive number"
                ) from exc
            if not math.isfinite(radius) or radius <= 0.0:
                raise DesignConfigError(
                    f"{entry_where}.radius must be a finite positive number"
                )
            proximity_spec = CropProximitySpec(anchor, radius)
            if proximity_spec in proximity_specs:
                raise DesignConfigError(
                    f"{where}.proximity contains duplicate entries"
                )
            proximity_specs.append(proximity_spec)

    return CropSpec(
        include=include,
        proximity=tuple(proximity_specs),
        exclude=exclude,
    )


def _parse_length_plans(
    value: Any,
    design_chains: tuple[DesignChainSpec, ...],
    task_name: str,
) -> tuple[LengthPlan, ...]:
    if value is None:
        return (LengthPlan(0, {}),)
    if not isinstance(value, Mapping) or not value:
        raise DesignConfigError(f"task {task_name!r} design_length must be a non-empty mapping")
    chain_ids = tuple(chain.chain_id for chain in design_chains)
    polymer_ids = tuple(
        chain.chain_id for chain in design_chains if chain.modality != "ligand"
    )
    if "joint" in value:
        if set(value) != {"joint"}:
            raise DesignConfigError(
                f"task {task_name!r} joint design_length must contain only joint"
            )
        if len(design_chains) == 1:
            raise DesignConfigError(
                f"task {task_name!r} joint design_length is only valid for multiple design chains"
            )
        if not polymer_ids:
            raise DesignConfigError(
                f"task {task_name!r} multi-ligand-only tasks cannot provide joint lengths"
            )
        raw_plans = value["joint"]
        if not isinstance(raw_plans, list) or not raw_plans:
            raise DesignConfigError(
                f"task {task_name!r} design_length.joint must be a non-empty list"
            )
        plans: list[LengthPlan] = []
        seen: set[tuple[int, ...]] = set()
        for index, raw_plan in enumerate(raw_plans):
            where = f"task {task_name!r} design_length.joint[{index}]"
            if not isinstance(raw_plan, Mapping) or not raw_plan:
                raise DesignConfigError(f"{where} must be a non-empty mapping")
            row_keys = tuple(raw_plan)
            if set(row_keys) != set(polymer_ids):
                missing = [chain_id for chain_id in polymer_ids if chain_id not in raw_plan]
                extra = [chain_id for chain_id in row_keys if chain_id not in polymer_ids]
                raise DesignConfigError(
                    f"{where} must contain exactly polymer chains; missing={missing}, extra={extra}"
                )
            if row_keys != polymer_ids:
                raise DesignConfigError(
                    f"{where} keys must follow declared polymer chain order {polymer_ids}"
                )
            lengths = {
                chain_id: _positive_int(raw_plan[chain_id], f"{where}.{chain_id}")
                for chain_id in polymer_ids
            }
            signature = tuple(lengths[chain_id] for chain_id in polymer_ids)
            if signature in seen:
                raise DesignConfigError(f"{where} duplicates an earlier joint plan")
            seen.add(signature)
            plans.append(LengthPlan(index, lengths))
        return tuple(plans)

    if len(design_chains) == 1:
        if len(value) != 1:
            raise DesignConfigError(
                f"task {task_name!r} single-chain design_length must name its only chain"
            )
        chain_id, length_value = next(iter(value.items()))
        if chain_id not in chain_ids:
            raise DesignConfigError(
                f"task {task_name!r} design_length has unknown chain {chain_id!r}"
            )
        if length_value == "from_distribution":
            # The public sentinel is resolved after request expansion by the
            # API. Keep a placeholder plan here so ordinary expansion still
            # produces exactly one base request per sample.
            return (LengthPlan(0, {}),)
        return tuple(
            LengthPlan(index, {chain_id: length})
            for index, length in enumerate(
                _grid_values(length_value, f"task {task_name!r} design_length.{chain_id}")
            )
        )

    if not polymer_ids:
        raise DesignConfigError(
            f"task {task_name!r} multi-ligand-only tasks cannot provide non-joint lengths"
        )
    raise DesignConfigError(
        f"task {task_name!r} multiple design chains require explicit joint lengths"
    )


def _parse_partial(value: Any, *, task_name: str, invocation_dir: Path) -> PartialSpec:
    where = f"task {task_name!r} partial"
    if not isinstance(value, Mapping):
        raise DesignConfigError(f"{where} must be a mapping")
    _closed(value, {"reference", "result_csv", "fraction"}, where)
    if not {"reference", "fraction"} <= set(value):
        raise DesignConfigError(f"{where} requires reference and fraction")
    reference = _resolve_local_path(value["reference"], invocation_dir, f"{where}.reference")
    if reference is None or reference.suffix.lower() != ".cif":
        raise DesignConfigError(f"{where}.reference must name a candidate CIF")
    result_csv = (
        _resolve_local_path(value["result_csv"], invocation_dir, f"{where}.result_csv")
        if "result_csv" in value else None
    )
    if "result_csv" in value and result_csv is None:
        raise DesignConfigError(f"{where}.result_csv must name a result CSV")
    raw_fraction = value["fraction"]
    if isinstance(raw_fraction, bool) or not isinstance(raw_fraction, (int, float)):
        raise DesignConfigError(f"{where}.fraction must be finite in (0, 1]")
    try:
        fraction = float(raw_fraction)
    except (OverflowError, ValueError) as exc:
        raise DesignConfigError(f"{where}.fraction must be finite in (0, 1]") from exc
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise DesignConfigError(f"{where}.fraction must be finite in (0, 1]")
    if round(fraction * 400) < 1:
        raise DesignConfigError(f"{where}.fraction must select at least one of 400 steps")
    return PartialSpec(reference=reference, result_csv=result_csv, fraction=fraction)


def parse_task(
    name: str,
    payload: Any,
    *,
    invocation_dir: Path,
) -> TaskSpec:
    if not isinstance(name, str) or not _TASK_NAME.fullmatch(name):
        raise DesignConfigError(f"invalid task name: {name!r}")
    if not isinstance(payload, Mapping):
        raise DesignConfigError(f"task {name!r} must be a mapping")
    task_format = payload.get("format", "normal")
    if not isinstance(task_format, str) or task_format not in {"normal", "antibody"}:
        raise DesignConfigError(f"task {name!r} format must be normal or antibody")
    if task_format == "antibody":
        from ..antibody.schema import parse_antibody_task
        return parse_antibody_task(name, payload, invocation_dir=invocation_dir)
    _closed(payload, _TASK_KEYS, f"task {name!r}")
    partial = (
        _parse_partial(payload["partial"], task_name=name, invocation_dir=invocation_dir)
        if "partial" in payload else None
    )
    sample_num = _positive_int(payload.get("sample_num"), f"task {name!r} sample_num")
    chains = _parse_design_chains(payload.get("design_chains"), name)
    condition = payload.get("condition", {})
    if condition is None:
        condition = {}
    if not isinstance(condition, Mapping):
        raise DesignConfigError(f"task {name!r} condition must be a mapping")
    _validate_public_smiles(condition, name)
    _validate_contig_length_bounds(condition, name)
    distribution_chain = _distribution_chain(payload.get("design_length"), chains, name)
    if distribution_chain is not None:
        _validate_distribution_conditions(condition, distribution_chain, name)
    hotspot = payload.get("hotspot")
    if hotspot is not None and (not isinstance(hotspot, str) or not hotspot.strip()):
        raise DesignConfigError(f"task {name!r} hotspot must be a non-empty string")
    score = payload.get("score", False)
    if not isinstance(score, bool):
        raise DesignConfigError(f"task {name!r} score must be boolean")
    postprocess = payload.get("postprocess", False)
    if type(postprocess) is not bool:
        raise DesignConfigError(f"task {name!r} postprocess must be boolean")
    override = payload.get("override", True)
    if type(override) is not bool:
        raise DesignConfigError(f"task {name!r} override must be boolean")
    cyclic_offset = payload.get("cyclic_offset", False)
    if type(cyclic_offset) is not bool:
        raise DesignConfigError(f"task {name!r} cyclic_offset must be boolean")
    if cyclic_offset and not any(chain.modality == "protein" for chain in chains):
        raise DesignConfigError(f"task {name!r} cyclic_offset requires a protein design chain")
    selected_cyclic = payload.get("cyclic_offset_chains")
    protein_chains = {chain.chain_id for chain in chains if chain.modality == "protein"}
    if "cyclic_offset_chains" in payload:
        if not cyclic_offset:
            raise DesignConfigError("cyclic_offset_chains requires cyclic_offset: true")
        if (
            not isinstance(selected_cyclic, list) or not selected_cyclic
            or any(not isinstance(chain, str) or chain not in protein_chains for chain in selected_cyclic)
            or len(set(selected_cyclic)) != len(selected_cyclic)
        ):
            raise DesignConfigError("cyclic_offset_chains must uniquely select design protein chains")
    elif cyclic_offset and len(protein_chains) > 1:
        raise DesignConfigError("multiple protein design chains require cyclic_offset_chains")
    length_plans = ((LengthPlan(0, {}),)
                    if partial is not None and "design_length" not in payload else
                    _parse_length_plans(payload.get("design_length"), chains, name))
    if partial is not None:
        length_plans = (LengthPlan(0, {}),)
    return TaskSpec(
        name=name,
        input_path=_resolve_local_path(payload.get("input"), invocation_dir, f"task {name!r} input"),
        sample_num=sample_num,
        design_chains=chains,
        length_plans=length_plans,
        condition=dict(condition),
        hotspot=hotspot,
        crop=_parse_crop(payload.get("crop"), name),
        score=score,
        cyclic_offset=cyclic_offset,
        cyclic_offset_chains=tuple(selected_cyclic or ()),
        postprocess=postprocess,
        override=override,
        length_distribution_chain=distribution_chain if partial is None else None,
        rm_token_type=normalize_task_rm_token_type(payload.get("rm_token_type"), f"task {name!r} rm_token_type"),
        partial=partial,
    )


def select_and_parse_tasks(
    tasks: Mapping[str, Any],
    *,
    task_names: Sequence[str] | None,
    invocation_dir: Path,
    global_rm_token_type: Sequence[str] | None = None,
) -> tuple[TaskSpec, ...]:
    if not isinstance(tasks, Mapping) or not tasks:
        raise DesignConfigError("tasks must be a non-empty name-to-dict mapping")
    available = list(tasks)
    selected = available if task_names is None else list(task_names)
    if not selected:
        raise DesignConfigError("task selection may not be empty")
    missing = [name for name in selected if name not in tasks]
    if missing:
        raise DesignConfigError(f"selected tasks do not exist: {missing}")
    if len(selected) != len(set(selected)):
        raise DesignConfigError("selected tasks contain duplicates")
    formats = {}
    for name in selected:
        payload = tasks[name]
        if not isinstance(payload, Mapping):
            raise DesignConfigError(f"task {name!r} must be a mapping")
        value = payload.get("format", "normal")
        if not isinstance(value, str) or value not in {"normal", "antibody"}:
            raise DesignConfigError(f"task {name!r} format must be normal or antibody")
        formats[name] = value
    if len(set(formats.values())) != 1:
        raise DesignConfigError(f"normal and antibody tasks cannot share one submission: {formats}")
    parsed = tuple(parse_task(name, tasks[name], invocation_dir=invocation_dir) for name in selected)
    global_policy = normalize_rm_token_type(global_rm_token_type)
    if global_policy is not None:
        parsed = tuple(replace(task, rm_token_type=global_policy) for task in parsed)
    return parsed


def expand_requests(tasks: Sequence[TaskSpec]) -> tuple[ExpandedRequest, ...]:
    requests: list[ExpandedRequest] = []
    global_ordinal = 0
    for task in tasks:
        task_ordinal = 0
        for plan in task.length_plans:
            for _ in range(task.sample_num):
                sample_id = f"sample_{task_ordinal:06d}"
                requests.append(
                    ExpandedRequest(
                        task=task,
                        plan=plan,
                        sample_id=sample_id,
                        task_slot_ordinal=task_ordinal,
                        global_ordinal=global_ordinal,
                    )
                )
                task_ordinal += 1
                global_ordinal += 1
    keys = [(request.task.name, request.sample_id) for request in requests]
    if len(keys) != len(set(keys)):
        raise AssertionError("expanded request IDs are not globally unique")
    return tuple(requests)
