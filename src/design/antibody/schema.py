"""Closed task-level antibody schema, lowered to ordinary structural design."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..core.crop import source_selector
from ..core.omission import normalize_task_rm_token_type
from ..core.records import (
    DesignChainSpec,
    DesignConfigError,
    DesignSampleError,
    LengthPlan,
    TaskSpec,
)
from .scaffolds import Scaffold, ScaffoldChain, load_scaffold, scaffold_path


@dataclass(frozen=True)
class RoleDesign:
    role: str
    chain_id: str
    cdrs: tuple[int, ...]
    lengths: tuple[tuple[int, int], ...]
    framework_positions: tuple[int, ...] = ()
    official_edits: tuple[bool, ...] = (False, False, False)
    paratope_cdrs: tuple[int, ...] = (1, 2, 3)
    paratope_framework_positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class AntibodySpec:
    scaffold_path: Path
    source_roles: tuple[tuple[str, str], ...]
    chains: tuple[RoleDesign, ...]
    template: bool = True
    alternatives: tuple[AntibodySpec, ...] = ()
    selection_name: str | None = None


@dataclass(frozen=True)
class _FrameworkResidueView:
    """Coordinate-free source residue metadata for the shared selector parser."""

    chain_id: np.ndarray
    residue_uid: np.ndarray
    label_seq_id: np.ndarray
    res_id: np.ndarray
    ins_code: np.ndarray


def _framework_metadata_view(
    scaffold: Scaffold,
) -> tuple[_FrameworkResidueView, dict[int, tuple[str, str, int]]]:
    """Build the minimal source view needed by ``source_selector``.

    ``ScaffoldChain.source_positions`` retains internal unresolved positions on
    the label-sequence axis.  Giving each one a metadata-only residue row lets
    framework selectors address those positions without requiring coordinates.
    """

    chains: list[str] = []
    residue_uids: list[int] = []
    labels: list[str] = []
    residue_ids: list[int] = []
    insertion_codes: list[str] = []
    source_by_uid: dict[int, tuple[str, str, int]] = {}
    for chain in scaffold.chains:
        for index, position in enumerate(chain.source_positions):
            uid = len(residue_uids)
            chains.append(chain.source_chain)
            residue_uids.append(uid)
            labels.append(str(position))
            residue_ids.append(position)
            insertion_codes.append(chain.insertion_codes[index])
            source_by_uid[uid] = (chain.role, chain.source_chain, position)
    return (
        _FrameworkResidueView(
            chain_id=np.asarray(chains, dtype=str),
            residue_uid=np.asarray(residue_uids, dtype=np.int64),
            label_seq_id=np.asarray(labels, dtype=str),
            res_id=np.asarray(residue_ids, dtype=np.int64),
            ins_code=np.asarray(insertion_codes, dtype=str),
        ),
        source_by_uid,
    )


def _framework_selector_prefix(chain_id: str) -> str:
    if len(chain_id) == 1 and chain_id.isascii() and chain_id.isalpha():
        return chain_id
    return f"[{chain_id}]"


def _framework_selector_hint(role: str, source_chain: str) -> str:
    prefix = _framework_selector_prefix(source_chain)
    return (
        f"{role}.framework must be boolean or a residue selector string using "
        f"source label_seq_id, e.g. '5/10-12' within {role}; explicit source "
        f"TEMPLATE selectors such as '{prefix}5/{prefix}10-12' are also accepted; "
        "integer and comma/'..' selectors are unsupported"
    )


def _resolve_framework_selector(
    selector: str,
    *,
    role: str,
    source_chain: ScaffoldChain,
    framework_available: tuple[int, ...],
    metadata_view: _FrameworkResidueView,
    source_by_uid: Mapping[int, tuple[str, str, int]],
) -> tuple[int, ...]:
    hint = _framework_selector_hint(role, source_chain.source_chain)
    # The enclosing VH/VL/VHH role already identifies the template chain.
    # Only supply its prefix; endpoint/range semantics stay in source_selector.
    prefix = _framework_selector_prefix(source_chain.source_chain)
    terms = [term.strip() for term in selector.split("/")]
    resolved_selector = "/".join(
        f"{prefix}{term}" if term[:1].isdigit() else term for term in terms
    )
    try:
        selected_uids = source_selector(metadata_view, resolved_selector)
    except DesignSampleError as exc:
        raise DesignConfigError(f"{hint}: {exc}") from exc

    available = set(framework_available)
    selected: list[int] = []
    for raw_uid in selected_uids:
        source = source_by_uid.get(int(raw_uid))
        if source is None:
            raise DesignConfigError(f"{hint}: selector resolved an unknown source residue")
        selected_role, selected_chain, position = source
        if selected_role != role or selected_chain != source_chain.source_chain:
            raise DesignConfigError(
                f"{role}.framework selector {selector!r} must select only the "
                f"{role} source TEMPLATE chain {source_chain.source_chain!r}"
            )
        selected.append(position)
    selected_set = set(selected)
    non_framework = sorted(selected_set - available)
    if non_framework:
        raise DesignConfigError(
            f"{role}.framework selector {selector!r} selects non-framework "
            f"(CDR) source positions {non_framework}"
        )
    return tuple(position for position in source_chain.source_positions if position in selected_set)


def _closed(value: Any, keys: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DesignConfigError(f"{where} must be a mapping")
    if set(value) - keys:
        raise DesignConfigError(f"unknown {where} keys: {sorted(set(value)-keys)}")
    return value


def _length(value: Any, native: int, official: tuple[int, int], where: str) -> tuple[int, int]:
    if value == "native":
        return native, native
    if value == "official":
        return official
    if type(value) is int and value > 0:
        return value, value
    if isinstance(value, Mapping) and set(value) == {"min", "max"}:
        lo, hi = value["min"], value["max"]
        if type(lo) is int and type(hi) is int and 0 < lo <= hi:
            return lo, hi
    raise DesignConfigError(f"{where} must be native, official, a positive count or {{min, max}}")


def _merge_antibody_settings(common: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge per-scaffold authoring recursively; every list replaces its parent."""
    result = deepcopy(dict(common))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_antibody_settings(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@lru_cache(maxsize=256)
def _custom_selection_path(path: str, invocation_dir: Path) -> str:
    from ..core.tasks import _resolve_local_path

    resolved = _resolve_local_path(path, invocation_dir, "antibody.scaffold.path")
    if resolved is None:
        raise DesignConfigError("custom antibody scaffold needs a local YAML path")
    return str(resolved)


def _selection_name(source: Any, invocation_dir: Path) -> str:
    if isinstance(source, str) and source:
        return source
    if isinstance(source, Mapping):
        path = source.get("path")
        if isinstance(path, str) and path:
            return _custom_selection_path(path, invocation_dir)
    raise DesignConfigError("antibody.scaffold choices must name an official scaffold or a custom scaffold mapping")


def _antibody_choices(raw: Mapping[str, Any], invocation_dir: Path) -> tuple[tuple[str, dict[str, Any]], ...]:
    sources = raw.get("scaffold")
    if not isinstance(sources, list) or not sources:
        raise DesignConfigError("antibody.scaffold list must be nonempty")
    common = {key: value for key, value in raw.items() if key != "scaffold"}
    choices = []
    seen = set()
    for index, item in enumerate(sources):
        if isinstance(item, Mapping) and "scaffold" in item:
            wrapper = _closed(item, {"scaffold", "design", "paratope", "template"},
                              f"antibody.scaffold[{index}]")
            source = wrapper["scaffold"]
            settings = _merge_antibody_settings(common, {
                key: value for key, value in wrapper.items() if key != "scaffold"})
        else:
            source = item
            settings = deepcopy(common)
        identity = _selection_name(source, invocation_dir)
        if identity in seen:
            raise DesignConfigError(f"antibody.scaffold list repeats scaffold {identity!r}; each choice needs a unique source")
        seen.add(identity)
        settings["scaffold"] = deepcopy(source)
        choices.append((identity, settings))
    return tuple(choices)


def _saved_scaffold_name(saved_row: Mapping[str, Any]) -> str | None:
    """Use the public scalar column without reading private context files."""
    identity = saved_row.get("scaffold")
    if isinstance(identity, str) and identity.strip():
        return identity.strip()
    context = saved_row.get("_antibody_context")
    if isinstance(context, str) and context.strip():
        try:
            context = json.loads(context)
        except (TypeError, ValueError) as exc:
            raise DesignSampleError("invalid saved antibody context") from exc
    if isinstance(context, Mapping):
        identity = context.get("scaffold")
        if isinstance(identity, str) and identity.strip():
            return identity.strip()
    return None


def resolve_antibody_authoring(
    authored_task: Mapping[str, Any], saved_row: Mapping[str, Any], *,
    invocation_dir: Path | None = None,
) -> dict[str, Any]:
    """Pin list authoring to a saved choice without loading any scaffold assets.

    Scalar tasks return unchanged. Custom list identities use full YAML paths;
    callers with relative custom paths can supply their captured invocation_dir.
    Saved public ``scaffold`` takes precedence over the in-memory native context.
    """
    raw = authored_task.get("antibody")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("scaffold"), list):
        return authored_task if isinstance(authored_task, dict) else dict(authored_task)
    choices = _antibody_choices(raw, Path.cwd() if invocation_dir is None else Path(invocation_dir))
    identity = _saved_scaffold_name(saved_row)
    if identity is None and len(choices) == 1:
        identity = choices[0][0]
    selected = next((settings for name, settings in choices if name == identity), None)
    if selected is None:
        raise DesignSampleError(
            "partial antibody scaffold list requires a saved concrete scaffold choice; "
            "for a CIF-only reference specify one concrete antibody.scaffold")
    source = selected["scaffold"]
    if isinstance(source, Mapping):
        selected["scaffold"] = dict(source, path=identity)
    result = deepcopy(dict(authored_task))
    result["antibody"] = selected
    return result


def resolve_antibody_spec(spec: AntibodySpec, saved_row: Mapping[str, Any]) -> AntibodySpec:
    """Resolve a partial request before interpreting any saved CDR lengths."""
    if not spec.alternatives:
        # Pipeline authoring may already have pinned a custom list choice to
        # a scalar mapping. Keep its saved full-path identity on later rounds.
        identity = _saved_scaffold_name(saved_row)
        if spec.selection_name is None and identity == str(spec.scaffold_path):
            return replace(spec, selection_name=identity)
        return spec
    identity = _saved_scaffold_name(saved_row)
    if identity is None and len(spec.alternatives) == 1:
        return spec.alternatives[0]
    selected = next((choice for choice in spec.alternatives
                     if choice.selection_name == identity), None)
    if selected is None:
        raise DesignSampleError(
            "partial antibody scaffold list requires a saved concrete scaffold choice; "
            "for a CIF-only reference specify one concrete antibody.scaffold")
    return selected


def parse_antibody_task(name: str, payload: Mapping[str, Any], *, invocation_dir: Path) -> TaskSpec:
    from ..core.tasks import _parse_crop, _parse_partial, _positive_int, _resolve_local_path

    _closed(payload, {"format", "sample_num", "antibody", "antigen", "score", "postprocess", "override", "rm_token_type", "partial"}, f"task {name!r}")
    partial = (_parse_partial(payload["partial"], task_name=name, invocation_dir=invocation_dir)
               if "partial" in payload else None)
    raw = _closed(payload.get("antibody"), {"scaffold", "chains", "design", "paratope", "template"}, "antibody")
    source = raw.get("scaffold")
    if isinstance(source, list):
        if len(source) > 1 and partial is not None and partial.result_csv is None:
            raise DesignConfigError(
                "partial antibody.scaffold list needs saved native result_csv metadata; "
                "for a CIF-only reference specify one concrete antibody.scaffold")
        alternatives = []
        first_task = None
        expected_roles = None
        for identity, settings in _antibody_choices(raw, invocation_dir):
            choice_task = parse_antibody_task(name, dict(payload, antibody=settings), invocation_dir=invocation_dir)
            choice = choice_task.antibody
            role_chains = {chain.role: chain.chain_id for chain in choice.chains}
            if expected_roles is None:
                expected_roles = role_chains
                first_task = choice_task
            elif role_chains != expected_roles:
                raise DesignConfigError("antibody.scaffold choices must share the same roles and output chains; VHH and VH/VL cannot mix")
            alternatives.append(replace(choice, selection_name=identity))
        return replace(first_task, antibody=replace(first_task.antibody, alternatives=tuple(alternatives)))
    source_roles: tuple[tuple[str, str], ...] = ()
    if isinstance(source, str):
        source_path = scaffold_path(source)
    elif isinstance(source, Mapping):
        _closed(source, {"path", "roles", "numbering"}, "antibody.scaffold")
        if source.get("numbering") != "label_seq_id":
            raise DesignConfigError("custom antibody scaffold requires numbering: label_seq_id")
        source_path = _resolve_local_path(source.get("path"), invocation_dir, "antibody.scaffold.path")
        if source_path is None:
            raise DesignConfigError("custom antibody scaffold needs a local YAML path")
        role_map = source.get("roles")
        if not isinstance(role_map, Mapping) or set(role_map) not in ({"VH", "VL"}, {"VHH"}):
            raise DesignConfigError("custom scaffold roles must be {VH, VL} or {VHH}")
        if any(not isinstance(v, str) or not v for v in role_map.values()):
            raise DesignConfigError("custom scaffold roles need nonempty CIF label chain IDs")
        source_roles = tuple(sorted(role_map.items()))
    else:
        raise DesignConfigError("antibody.scaffold must name an official scaffold or a custom scaffold mapping")
    scaffold = load_scaffold(source_path, source_roles)
    source_roles = tuple((c.role, c.source_chain) for c in scaffold.chains)
    framework_view, framework_records = _framework_metadata_view(scaffold)
    roles = {c.role for c in scaffold.chains}
    if partial is not None and set(partial.cdr_lengths) - roles:
        raise DesignConfigError("partial.cdr_lengths contains a role absent from the antibody scaffold")
    chains = raw.get("chains", {role: "L" if role == "VL" else "H" for role in roles})
    if not isinstance(chains, Mapping) or set(chains) != roles:
        raise DesignConfigError(f"antibody.chains must assign exactly roles {sorted(roles)}")
    if any(not isinstance(c, str) or not c or len(c) > 32 for c in chains.values()) or len(set(chains.values())) != len(chains):
        raise DesignConfigError("antibody output chain IDs must be distinct nonempty strings of at most 32 characters")
    design = raw.get("design", {role: {"cdrs": ["CDR1", "CDR2", "CDR3"]} for role in roles})
    if not isinstance(design, Mapping) or set(design) - roles:
        raise DesignConfigError("antibody.design contains unknown roles")
    paratope = raw.get("paratope", {})
    if not isinstance(paratope, Mapping) or set(paratope) - roles:
        raise DesignConfigError("antibody.paratope contains unknown roles")
    planned: list[RoleDesign] = []
    sizes: dict[str, int] = {}
    for source_chain in scaffold.chains:
        role = source_chain.role
        selection = _closed(design.get(role, {}), {"cdrs", "lengths", "framework"}, f"antibody.design.{role}")
        cdrs = selection.get("cdrs", [])
        if not isinstance(cdrs, list) or any(c not in ("CDR1", "CDR2", "CDR3") for c in cdrs) or len(set(cdrs)) != len(cdrs):
            raise DesignConfigError("cdrs must be a unique list drawn from CDR1/CDR2/CDR3; [] is a reference task")
        chosen = tuple(sorted(int(c[-1]) for c in cdrs))
        lengths = selection.get("lengths", {})
        if not isinstance(lengths, Mapping) or set(lengths) - set(cdrs):
            raise DesignConfigError("CDR lengths may only be provided for selected CDRs")
        intervals = tuple(_length(lengths.get(f"CDR{i}", "native"), len(window), source_chain.official_lengths[i-1], f"{role}.lengths.CDR{i}") for i, window in enumerate(source_chain.cdr_positions, 1))
        framework = selection.get("framework", False)
        fr_available = tuple(n for n in source_chain.source_positions if n not in sum(source_chain.cdr_positions, ()))
        if framework is False:
            mutable_fr = ()
        elif framework is True:
            mutable_fr = fr_available
        elif isinstance(framework, str):
            mutable_fr = _resolve_framework_selector(
                framework,
                role=role,
                source_chain=source_chain,
                framework_available=fr_available,
                metadata_view=framework_view,
                source_by_uid=framework_records,
            )
        else:
            raise DesignConfigError(_framework_selector_hint(role, source_chain.source_chain))
        paratope_selection = _closed(paratope.get(role, {}), {"cdrs", "framework"}, f"antibody.paratope.{role}")
        paratope_cdrs = paratope_selection.get("cdrs", ["CDR1", "CDR2", "CDR3"])
        if not isinstance(paratope_cdrs, list) or any(c not in ("CDR1", "CDR2", "CDR3") for c in paratope_cdrs) or len(set(paratope_cdrs)) != len(paratope_cdrs):
            raise DesignConfigError(f"antibody.paratope.{role}.cdrs must be a unique list drawn from CDR1/CDR2/CDR3")
        chosen_paratope = tuple(sorted(int(c[-1]) for c in paratope_cdrs))
        paratope_framework = paratope_selection.get("framework", False)
        if paratope_framework is False:
            paratope_fr = ()
        elif paratope_framework is True:
            paratope_fr = fr_available
        elif isinstance(paratope_framework, str):
            paratope_fr = _resolve_framework_selector(
                paratope_framework,
                role=role,
                source_chain=source_chain,
                framework_available=fr_available,
                metadata_view=framework_view,
                source_by_uid=framework_records,
            )
        else:
            raise DesignConfigError(_framework_selector_hint(role, source_chain.source_chain))
        official_edits = tuple(lengths.get(f"CDR{i}") == "official" for i in (1, 2, 3))
        planned.append(RoleDesign(role, chains[role], chosen, intervals, mutable_fr, official_edits,
                                  paratope_cdrs=chosen_paratope, paratope_framework_positions=paratope_fr))
        sizes[chains[role]] = len(source_chain.sequence) + sum(intervals[i-1][0] - len(source_chain.cdr_positions[i-1]) for i in chosen)
    template = raw.get("template", True)
    if type(template) is not bool:
        raise DesignConfigError("antibody.template must be boolean")
    target = payload.get("antigen", {})
    if target is None:
        target = {}
    target = _closed(target, {"input", "crop", "hotspot"}, "antigen")
    target_path = _resolve_local_path(target.get("input"), invocation_dir, "antigen.input")
    if target_path is not None and not target_path.is_file():
        raise DesignConfigError(f"antigen input does not exist: {target_path}")
    if target_path is None and (target.get("crop") is not None or target.get("hotspot") is not None):
        raise DesignConfigError("antigen crop/hotspot requires antigen.input")
    hotspot = target.get("hotspot")
    if hotspot is not None and (not isinstance(hotspot, str) or not hotspot.strip()):
        raise DesignConfigError("antigen.hotspot must be a nonempty selector string")
    controls = {key: payload.get(key, default) for key, default in (("score", False), ("postprocess", False), ("override", True))}
    if any(type(v) is not bool for v in controls.values()):
        raise DesignConfigError("score, postprocess and override must be boolean")
    return TaskSpec(
        name=name, input_path=target_path,
        sample_num=_positive_int(payload.get("sample_num"), f"task {name!r} sample_num"),
        design_chains=tuple(DesignChainSpec(c.chain_id, "protein") for c in planned),
        length_plans=(LengthPlan(0, sizes),), condition={}, hotspot=hotspot,
        crop=_parse_crop(target.get("crop"), name), **controls,
        format="antibody", antibody=AntibodySpec(source_path, source_roles, tuple(planned), template),
        rm_token_type=normalize_task_rm_token_type(payload.get("rm_token_type"), f"task {name!r} rm_token_type"),
        partial=partial,
    )
