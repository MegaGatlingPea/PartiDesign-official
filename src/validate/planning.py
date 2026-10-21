"""Lightweight task and sequence plans; never imports a model or scans files."""
from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path


def resolve_tasks(runs, repo_root: Path, task_names=None) -> list[Path]:
    selected = set(task_names) if task_names else None
    tasks = []
    for value in runs:
        raw = Path(value).expanduser()
        bare_name = not raw.is_absolute() and "/" not in str(value) and "\\" not in str(value)
        local_run = Path.cwd() / "inference" / raw
        if bare_name and local_run.is_dir():
            root = local_run
        else:
            root = raw if raw.exists() else repo_root / "inference" / raw
        root = root.resolve()
        if not root.is_dir():
            raise ValueError(f"Run or task does not exist: {value}")
        candidates = [root] if (root / "result.csv").is_file() else sorted(root.iterdir())
        for path in candidates:
            if path.name in {"validate", "original"} or path.name.startswith("."):
                continue
            if path.is_dir() and (path / "result.csv").is_file() and (selected is None or path.name in selected):
                if path not in tasks:
                    tasks.append(path)
    if not tasks:
        raise ValueError("No selected task/result.csv files")
    return tasks


def inventory(task: Path, limit=None) -> dict:
    """Read the result table and resolve input paths without content hashing."""
    samples = {}
    with (task / "result.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "filename" not in reader.fieldnames:
            raise ValueError(f"Missing filename column: {task / 'result.csv'}")
        for index, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"Extra CSV fields at {task}/result.csv:{index}")
            row = {k: (v or "") for k, v in row.items()}
            name = row["filename"].strip()
            path = (task / name).resolve()
            sample = Path(name).stem if name else f"row{index}"
            if not sample or sample in {".", ".."} or sample in samples:
                raise ValueError(f"Duplicate or invalid sample identity: {sample}")
            error = None
            try:
                relative = path.relative_to(task)
                if not name or any(part in {"validate", "original"} for part in relative.parts):
                    raise ValueError("filename must name a generated input")
                if path.suffix.lower() not in {".pdb", ".cif"}:
                    raise ValueError("filename must end in .cif or .pdb")
                if not path.is_file():
                    raise ValueError(f"input structure does not exist: {path}")
            except (OSError, ValueError) as exc:
                error = str(exc)
            samples[sample] = {"path": str(path), "row": row, "error": error}
            if limit is not None and len(samples) >= limit:
                break
    return samples


def _valid_length_range(value) -> list[int] | None:
    """Expand one validated design-length range without importing design code.

    The design task parser accepts only positive integer min, max and step
    fields, with an endpoint aligned to the step. Invalid mappings
    deliberately return None so callers retain existing fallback behavior.
    """
    if not isinstance(value, Mapping) or set(value) != {"min", "max", "step"}:
        return None
    minimum, maximum, step = (value[key] for key in ("min", "max", "step"))
    if any(type(item) is not int or item <= 0 for item in (minimum, maximum, step)):
        return None
    if minimum > maximum or (maximum - minimum) % step != 0:
        return None
    return list(range(minimum, maximum + 1, step))


def generation_budget(task: Path, recorded: int, limit=None) -> int:
    if limit is not None:
        return recorded
    snapshot = task / "resolved_config.yaml"
    if not snapshot.is_file():
        return recorded
    from .config import load_yaml
    task_cfg = load_yaml(snapshot).get("tasks", {}).get(task.name, {})
    count = task_cfg.get("sample_num")
    if type(count) is not int or count < 1:
        return recorded
    lengths = task_cfg.get("design_length", {})
    combinations = 1
    if isinstance(lengths, dict):
        for key, value in lengths.items():
            if isinstance(value, list):
                combinations *= len(value)
            elif key != "joint":
                grid = _valid_length_range(value)
                if grid is not None:
                    combinations *= len(grid)
    declared = count * combinations
    if recorded > declared:
        raise ValueError(f"recorded generation rows ({recorded}) exceed declared budget ({declared})")
    return declared


def resolve_sequence_sources(row: Mapping, method: str, cfg: Mapping) -> list[dict]:
    """Resolve the sequence branches planned for one result row.

    The parser-level source list is a protein-shaped placeholder when source
    provenance is implicit. Validation resolves that placeholder after the
    row's designed modality is known. Keep the same light, metadata-only
    resolution in the saved plan so report denominators describe the slots
    the worker was actually asked to produce.

    Plans written before per-row source metadata existed have no provenance
    marker. Treat those source lists as authored to preserve their historical
    benchmark labels and counts.
    """
    from .sequences import default_validation_sources, normalize_sequence_sources

    sequence_cfg = cfg.get("sequence", {}) if isinstance(cfg, Mapping) else {}
    configured = sequence_cfg.get("sources", []) if isinstance(sequence_cfg, Mapping) else []
    marker = cfg.get("_sequence_sources_explicit") if isinstance(cfg, Mapping) else None
    explicit = True if marker is None else bool(marker)
    if explicit:
        return normalize_sequence_sources(configured, default_gt=False)
    return default_validation_sources(
        row,
        method=method,
        configured_sources=configured,
        explicit=False,
    )


def sequence_sources_by_sample(plan: Mapping) -> dict[str, list[dict]] | None:
    """Return planned source branches per sample when provenance is available.

    None denotes a legacy plan whose single global source list remains
    authoritative. A false parser provenance marker is enough to derive the
    per-row defaults for old plans that predate this field.
    """
    samples = plan.get("samples", {})
    if not isinstance(samples, Mapping):
        return None
    saved = plan.get("sequence_sources_by_sample")
    cfg = plan.get("config", {})
    if saved is None:
        marker = cfg.get("_sequence_sources_explicit") if isinstance(cfg, Mapping) else None
        if marker is not False:
            return None
        return {
            str(sample): resolve_sequence_sources(entry.get("row", {}), plan.get("method"), cfg)
            for sample, entry in samples.items()
            if isinstance(sample, str) and isinstance(entry, Mapping)
        }
    if not isinstance(saved, Mapping):
        raise ValueError("Malformed saved sequence source plan")
    result = {}
    for sample, specs in saved.items():
        if not isinstance(sample, str) or not isinstance(specs, (list, tuple)):
            raise ValueError("Malformed saved sequence source plan")
        result[sample] = resolve_sequence_sources(
            {}, plan.get("method"),
            {**cfg, "_sequence_sources_explicit": True,
             "sequence": {"sources": specs}},
        )
    return result


def sequence_slot_count(plan: Mapping, sample: str, source_map: Mapping | None = None) -> int:
    """Return the expected sequence slots for one sample."""
    if source_map is not None and sample in source_map:
        return sum(spec["count"] for spec in source_map[sample])
    count = plan.get("sequence_count", 0)
    if type(count) is not int or count < 0:
        raise ValueError("Malformed saved validation plan sequence count")
    return count


def make_plan(task: Path, method: str, cfg: dict, repo: Path, limit=None) -> dict:
    from .config import config_for_method

    cfg = config_for_method(cfg, method)
    del repo  # Kept in the signature for callers that build plans from a repo root.
    samples = inventory(task, limit)
    planned = generation_budget(task, len(samples), limit)
    lengths, basis = length_budgets(task, samples, planned, limit)
    sources_by_sample = {
        sample: resolve_sequence_sources(entry.get("row", {}), method, cfg)
        for sample, entry in samples.items()
    }
    return {
        "task_dir": str(task),
        "method": method,
        "samples": samples,
        "generation_planned": planned,
        "length_budgets": lengths,
        "length_budget_basis": basis,
        "sequence_count": sum(s["count"] for s in cfg["sequence"]["sources"]),
        "sequence_sources_by_sample": sources_by_sample,
        "config": cfg,
    }


def expected_keys(method: str, cfg: dict) -> list[str]:
    if method == "af2_ig":
        return [f"{protocol}_model{model}_sample{sample}"
                for protocol in ("complex", "monomer") for model in cfg["af2"]["models"]
                for sample in range(cfg["af2"]["samples"])]
    if method in {"esmfold2-fast", "esmfold2-standard"}:
        return [f"sample{i}" for i in range(cfg["esmfold2"]["samples"])]
    if method in {"af3", "af3-la"}:
        mode = cfg["af3"]["mode"]
        protocols = ("complex", "monomer") if mode == "both" else (mode,)
        return [f"{protocol}_sample{i}" for protocol in protocols
                for i in range(cfg["af3"]["samples"])]
    return ["original"]


def prediction_metadata_valid(key: str, prediction: dict, method: str) -> bool:
    """A backend prediction key binds its protocol/model/sample slot."""
    import re

    if not isinstance(prediction, dict) or not isinstance(key, str) or prediction.get("key") != key:
        return False
    if method == "af2_ig":
        match = re.fullmatch(r"(complex|monomer)_model([12])_sample(\d+)", key)
        return bool(match and prediction.get("protocol") == match[1]
                    and type(prediction.get("model")) is int
                    and prediction["model"] == int(match[2])
                    and type(prediction.get("sample")) is int
                    and prediction["sample"] == int(match[3]))
    if method in {"esmfold2-fast", "esmfold2-standard"}:
        match = re.fullmatch(r"sample(\d+)", key)
        return bool(match and prediction.get("protocol") == "complex"
                    and prediction.get("model") in {method.removeprefix("esmfold2-"), method}
                    and type(prediction.get("sample")) is int
                    and prediction["sample"] == int(match[1]))
    if method in {"af3", "af3-la"}:
        match = re.fullmatch(r"(complex|monomer)_sample(\d+)", key)
        return bool(match and prediction.get("protocol") == match[1]
                    and prediction.get("model") == method
                    and type(prediction.get("sample")) is int
                    and prediction["sample"] == int(match[2]))
    return (method in {"partiscore", "vina_dock"} and key == "original"
            and prediction.get("protocol") == "complex"
            and prediction.get("model") == method
            and type(prediction.get("sample")) is int
            and prediction["sample"] == 0)


def row_length_group(row):
    if row.get("heavy_role") in {"VH", "VHH"}:
        chains = [(row.get(f"{prefix}_chain"), row.get(f"{prefix}_sequence"))
                  for prefix in ("heavy", "light")]
        if not all(chain and sequence for chain, sequence in chains if chain or sequence) or not chains[0][0]:
            return "unknown"
        return "|".join(f"{chain}:protein:{len(sequence)}" for chain, sequence in sorted(
            (chain, sequence) for chain, sequence in chains if chain))
    import json
    try:
        entries = json.loads(row.get("design_chains") or "[]")
        polymers = [entry for entry in entries if entry["modality"] in {"protein", "rna", "dna"}]
        if entries and not polymers:
            return "ligand"
        if polymers:
            return "|".join(f"{e['chain']}:{e['modality']}:{e['residue_count']}" for e in sorted(polymers, key=lambda e: e["chain"]))
    except (ValueError, TypeError, KeyError):
        pass
    return "unknown"


def length_budgets(task, samples, planned, limit=None):
    from collections import Counter
    recorded = Counter(row_length_group(entry["row"]) for entry in samples.values())
    if limit is not None:
        return dict(recorded), "selected_rows"
    snapshot = task / "resolved_config.yaml"
    if snapshot.is_file():
        from .config import load_yaml
        from itertools import product
        cfg = load_yaml(snapshot).get("tasks", {}).get(task.name, {})
        raw_chains = cfg.get("design_chains", [])
        chains = dict(item for spec in raw_chains for item in spec.items()) if isinstance(raw_chains, list) else raw_chains
        lengths = cfg.get("design_length", {})
        count = cfg.get("sample_num")
        if isinstance(chains, dict) and isinstance(lengths, dict) and type(count) is int:
            # Task YAML retains authored peptide intent; result rows use the
            # canonical protein chemistry needed by refold/report consumers.
            chains = {chain: "protein" if kind == "peptide" else kind
                      for chain, kind in chains.items()}
            polymers = sorted((chain, kind) for chain, kind in chains.items() if kind in {"protein", "rna", "dna"})
            if not polymers and chains:
                return {"ligand": planned}, "declared_generation_budget"
            grids = []
            for chain, kind in polymers:
                value = lengths.get(chain)
                if isinstance(value, list):
                    values = value
                else:
                    values = _valid_length_range(value)
                    if values is None:
                        values = [value]
                if not values or any(type(v) is not int or v <= 0 for v in values):
                    grids = []
                    break
                grids.append([(chain, kind, v) for v in values])
            if grids:
                allocations = {"|".join(f"{c}:{kind}:{length}" for c, kind, length in values): count
                               for values in product(*grids)}
                if sum(allocations.values()) == planned:
                    return allocations, "declared_generation_budget"
    if planned > len(samples):
        recorded["unknown"] += planned - len(samples)
    return dict(recorded), "recorded_lengths_with_unallocated_failures"
