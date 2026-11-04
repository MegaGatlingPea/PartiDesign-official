"""Normalize declared reference structures into native, candidate-owned tasks."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import time

from .state import Candidate, atomic_json, write_csv


def configuration_task(seed):
    """Compose model settings without treating a seed as de novo generation."""
    from src.design.utils.reader import read_structure
    from .state import residues

    authored = copy.deepcopy(seed["authored"])
    authored.pop("partial", None)
    authored["sample_num"] = 1
    if authored.get("format", "normal") == "normal":
        atoms = read_structure(seed["path"])
        lengths = {chain: len(residues(atoms, chain))
                   for entry in authored["design_chains"] for chain in entry}
        authored["design_length"] = ({next(iter(lengths)): [next(iter(lengths.values()))]}
                                     if len(lengths) == 1 else {"joint": [lengths]})
    return seed["task"], authored


def candidate_authoring(cfg, candidate):
    if not cfg.get("seed_input"):
        authored = cfg["task_payload"][candidate.task]
    else:
        import yaml

        registry = cfg.get("_seed_authored")
        if registry is None:
            registry = {}
            for seed in cfg["seed_input"]["rows"]:
                metadata = seed_metadata(cfg, seed["task"], seed["sample_id"])
                path = metadata / "resolved_config.yaml"
                if path.is_file():
                    registry[(seed["task"], seed["sample_id"])] = yaml.safe_load(path.read_text())["tasks"][seed["task"]]
                else:
                    registry[(seed["task"], seed["sample_id"])] = seed["authored"]
            cfg["_seed_authored"] = registry
        authored = registry[(candidate.task, candidate.origin_id)]
    if (authored.get("format") == "antibody"
            and isinstance(authored["antibody"].get("scaffold"), list)):
        from src.design.antibody.schema import resolve_antibody_authoring

        resolved = cfg.setdefault("_antibody_authored", {})
        key = (candidate.task, candidate.origin_id)
        if key not in resolved:
            resolved[key] = resolve_antibody_authoring(authored, candidate.row)
        return resolved[key]
    return authored


def seed_metadata(cfg, task, sample_id):
    return Path(cfg["output_dir"]) / ".work" / "seeds" / task / sample_id


def _merge_graph_conditions(authored, record):
    """Retain explicit non-template chemistry without fixing free sequences."""
    from src.design.core.records import TopologyCondition
    from src.validate.sequences import _graph_topology_conditions
    from .state import residues

    graph = _graph_topology_conditions(record, TopologyCondition)
    if not graph:
        return authored
    atoms = record.atoms
    labels = getattr(atoms, "label_asym_id", atoms.chain_id)
    rows = {int(uid): i for i, uid in enumerate(atoms.atom_uid)}
    positions = {}
    for chain in record.design_chains:
        for position, indices in enumerate(residues(atoms, chain), 1):
            positions.update((int(i), position) for i in indices)
    if not authored.get("condition"):
        authored["condition"] = {}
    condition = authored["condition"]
    bonds = condition.setdefault("bond", [])
    sequence = condition.setdefault("sequence", {})
    from src.design.core.compiler import _identity_invariant_atom
    from src.validate.sequences import _AA3_TO_1, _RNA3_TO_1, _DNA3_TO_1
    maps = {"protein": _AA3_TO_1, "rna": _RNA3_TO_1, "dna": _DNA3_TO_1}
    for edge in graph:
        endpoints = []
        for uid in (edge.atom_uid_a, edge.atom_uid_b):
            i = rows[uid]
            chain = str(labels[i])
            atom = str(atoms.atom_name[i])
            modality = str(atoms.mol_type[i])
            endpoint = {"chain": chain, "atom": atom}
            if modality != "ligand":
                position = positions[i] if i in positions else int(atoms.label_seq_id[i])
                endpoint["resid"] = position
            endpoints.append(endpoint)
            if chain not in record.design_chains or _identity_invariant_atom(modality, atom):
                continue
            name = str(atoms.res_name[i])
            residue = maps.get(modality, {}).get(name, "[" + name + "]")
            current = sequence.setdefault(chain, [])
            if isinstance(current, str):
                from .state import tokens
                if tokens(current)[position - 1] != residue:
                    raise ValueError(f"source bond requires {chain}{position} identity {residue}")
            else:
                matches = [entry for entry in current
                           if (entry["resid"] if entry["resid"] > 0 else len(residues(atoms, chain)) + entry["resid"] + 1) == position]
                if matches and any(entry["residue"] != residue for entry in matches):
                    raise ValueError(f"source bond conflicts with fixed identity at {chain}{position}")
                if not matches:
                    current.append({"resid": position, "residue": residue})
        order = "any" if edge.order == "?" else edge.order
        proposed = {"atom1": endpoints[0], "atom2": endpoints[1], "order": order}
        pair = {tuple(sorted(endpoint.items())) for endpoint in endpoints}
        duplicate = False
        for existing in bonds:
            # Exact native aliases are compiled by the same resolver below.
            previous = {tuple(sorted(existing[key].items())) for key in ("atom1", "atom2")}
            if previous == pair:
                previous_order = existing.get("order", "any")
                if previous_order in {"any", "unknown"}:
                    existing["order"] = order
                elif order not in {"any", "unknown", previous_order}:
                    raise ValueError("source and annotation specify conflicting bond orders")
                duplicate = True
                break
        if not duplicate:
            bonds.append(proposed)
    if not sequence:
        condition.pop("sequence", None)
    return authored


def _freeze_source(cfg, seed):
    from src.design.utils.reader import read_structure
    from src.design.utils.writer import write_structure_atomic

    metadata = seed_metadata(cfg, seed["task"], seed["sample_id"])
    metadata.mkdir(parents=True, exist_ok=True)
    source = Path(seed["path"])
    frozen = metadata / "source.cif"
    if source.suffix.lower() == ".cif":
        shutil.copy2(source, frozen)
    else:
        write_structure_atomic(read_structure(source), frozen, entry_id=seed["sample_id"])
    return frozen


def _prepare_one(cfg, seed, *, frozen=None):
    import yaml
    from dataclasses import replace
    from src.design.core.tasks import parse_task, expand_requests
    from src.design.core.compiler import compile_request
    from src.design.core.partial import load_partial_reference
    from src.design.core.runtime import reference_result_row, reference_output_atoms
    from src.design.utils.writer import write_structure_atomic
    from .state import residues

    task, sample = seed["task"], seed["sample_id"]
    folder = Path(cfg["output_dir"]) / task / "start"
    metadata = seed_metadata(cfg, task, sample)
    source = Path(seed["path"])
    frozen = _freeze_source(cfg, seed) if frozen is None else frozen
    authored = copy.deepcopy(seed["authored"])
    authored["sample_num"] = 1
    if authored.get("format", "normal") == "normal" and Path(authored.get("input") or source) == source:
        authored["input"] = str(frozen)
    partial = dict(authored.get("partial") or {}, reference=str(frozen),
                   fraction=seed["initial_partial"].get("fraction", 0.625))
    authored["partial"] = partial
    parsed = parse_task(task, authored, invocation_dir=metadata)
    if parsed.partial.result_csv is not None:
        original = load_partial_reference(replace(parsed.partial, reference=source), task=parsed)
        reference_csv = metadata / "reference.csv"
        write_csv(reference_csv, [dict(original.record.row, filename=str(frozen))])
        authored["partial"]["result_csv"] = str(reference_csv)
        parsed = parse_task(task, authored, invocation_dir=metadata)
    if parsed.format == "normal":
        reference = load_partial_reference(parsed.partial, task=parsed)
        if authored.get("design_length") is not None:
            check = dict(authored)
            check.pop("partial")
            requested = parse_task(task, check, invocation_dir=metadata)
            actual = {chain.chain_id: len(residues(reference.record.atoms, chain.chain_id))
                      for chain in parsed.design_chains}
            if not any(dict(plan.lengths) == actual for plan in requested.length_plans):
                raise ValueError(f"design_length does not contain the seed's actual lengths: {actual}")
        authored = _merge_graph_conditions(authored, reference.record)
        parsed = parse_task(task, authored, invocation_dir=metadata)
    compiled = compile_request(expand_requests((parsed,))[0])
    path = folder / (sample + ".cif")
    folder.mkdir(parents=True, exist_ok=True)
    write_structure_atomic(reference_output_atoms(compiled), path, entry_id=sample,
                           validated_ligand_identities=compiled.validated_ligand_identities)
    row = reference_result_row(compiled, filename=str(path))
    context = row.pop("_antibody_context", None)
    if context is not None:
        context_path = metadata / "antibody_context.json"
        atomic_json(context_path, context)
        row["_antibody_context_path"] = str(context_path)
    row.update(filename=str(path), source_status=seed.get("source_status", ""),
               _pipeline_metadata=str(metadata))
    (metadata / "resolved_config.yaml").write_text(yaml.safe_dump(
        {"tasks": {task: authored}, "task_names": [task]}, sort_keys=False))
    return Candidate(sample, task, "", sample, str(path), row, source="seed", origin_path=str(source))


def prepare_seeds(cfg):
    """Compile each declared seed once and export native constraint metadata."""
    result = {name: [] for name in cfg["task_names"]}
    from concurrent.futures import ThreadPoolExecutor
    rows = cfg["seed_input"]["rows"]
    with ThreadPoolExecutor(max_workers=min(4, len(rows))) as pool:
        copies = [pool.submit(_freeze_source, cfg, seed) for seed in rows]
        for seed, future in zip(rows, copies):
            try:
                result[seed["task"]].append(_prepare_one(cfg, seed, frozen=future.result()))
            except (ValueError, OSError) as exc:
                raise ValueError(f"annotation.csv row {seed['row_number']} ({seed['sample_id']}): {exc}") from exc
    for task, candidates in result.items():
        write_csv(Path(cfg["output_dir"]) / task / "start" / "result.csv",
                  [dict(c.row, candidate_id=c.candidate_id, origin_id=c.origin_id) for c in candidates])
    return result


def initialize_seeds(owner, cfg, seeds):
    """Run the reference-start stage before the first refold round."""
    from dataclasses import replace

    settings = {(row["task"], row["sample_id"]): row["initial_partial"] for row in cfg["seed_input"]["rows"]}
    result = {}
    for task, parents in seeds.items():
        folder = Path(cfg["output_dir"]) / task / "start"
        state = folder / ".work" / "initial.json"
        if state.is_file():
            saved = json.loads(state.read_text())
            restored = [Candidate(**row) for row in saved["candidates"]]
            if all(Path(c.path).is_file() for c in restored):
                result[task] = restored
                if saved.get("summary"):
                    cfg.setdefault("_initial_summaries", []).append(saved["summary"])
                continue
        started = time.monotonic()
        # An interrupted initial phase is not a completed native batch. Replace
        # only its owned replicas, retaining imported inputs and their ledgers.
        for partial_folder in (folder / "partial", folder / ".work" / "partial"):
            if partial_folder.is_dir() and not partial_folder.is_symlink():
                shutil.rmtree(partial_folder)
        enabled = [c for c in parents if settings[(task, c.origin_id)]["enabled"]]
        retained = [c for c in parents if not settings[(task, c.origin_id)]["enabled"]
                    or settings[(task, c.origin_id)]["include_parent"]]
        children = []
        outcome = {"failed": 0, "model_loads": 0}
        if enabled:
            if hasattr(owner, "activate_partial"):
                owner.activate_partial(cfg["effective_generation"], parent_count=len(enabled))
            references = [{"candidate_id": c.candidate_id, "path": c.path,
                           "result_csv": str(folder / "result.csv"),
                           "authored": candidate_authoring(cfg, c), "settings": settings[(task, c.origin_id)]}
                          for c in enabled]
            outcome = owner.partial_many(references, authored={}, effective_generation=cfg["effective_generation"],
                                         settings=settings[(task, enabled[0].origin_id)],
                                         output=folder / "partial", seed=cfg["seed"])
            by_id = {c.candidate_id: c for c in parents}
            for index, item in enumerate(outcome["candidates"], 1):
                parent = by_id[item["parent_id"]]
                row = dict(item["row"], _pipeline_metadata=parent.row["_pipeline_metadata"],
                           source_status=parent.row.get("source_status", ""))
                if parent.row.get("_antibody_context_path"):
                    row["_antibody_context_path"] = parent.row["_antibody_context_path"]
                    row.pop("_antibody_context", None)
                children.append(replace(parent, candidate_id=f"initial_p{index:07d}",
                                        parent_id=parent.candidate_id, path=item["path"], row=row, source="partial"))
        result[task] = retained + children
        summary = {"task": task, "stage": "start", "seed_count": len(parents),
                   "retained_parent_count": len(retained), "partial_success_count": len(children),
                   "partial_failed_count": outcome.get("failed", 0),
                   "partial_model_loads": outcome.get("model_loads", 0),
                   "selected_count": len(result[task]), "seconds": time.monotonic() - started}
        cfg.setdefault("_initial_summaries", []).append(summary)
        cfg.setdefault("_initial_recomputed", set()).add(task)
        print(f"[{task}] start: {len(parents)} seeds, {len(children)} partial children, "
              f"{len(retained)} retained parents", flush=True)
        atomic_json(state, {"candidates": [c.saved() for c in result[task]], "summary": summary})
    return result
