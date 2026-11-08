"""CPU ligand grouping from authoritative metadata, without structure readers."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Iterable

from .inputs import ClusterJob, _names, _read_csv, discover_jobs, load_job
from .runner import MAX_WORKERS, _THREAD_VARS, _sample_key, _single_core, _write_reports


_DETAIL_FIELDS = (
    "run", "task", "method", "criterion", "cohort", "group_by", "candidate_id",
    "structure_path", "ligands", "chemical_group_key", "group_id", "group_size",
    "representative", "representative_file", "is_representative", "passing_branches",
    "representative_policy", "chemistry_source",
)
_SUMMARY_FIELDS = (
    "run", "task", "method", "criterion", "cohort", "group_by", "n_candidates",
    "n_identities", "n_groups", "n_singletons", "largest_group",
    "representative_policy", "chemistry_source",
)


@dataclass(frozen=True)
class ChemicalJob:
    task_dir: Path
    task_name: str
    method: str
    report_dir: Path
    source: str


@dataclass(frozen=True)
class ChemicalSample:
    candidate_id: str
    structure: Path
    # Screen inputs have no output-chain locator in the manifest; None is
    # intentional and never inferred by reading the predicted structure.
    ligands: tuple[tuple[str | None, str], ...]


@dataclass(frozen=True)
class ChemicalInput:
    samples: dict[str, ChemicalSample]
    cohorts: dict[str, tuple[str, ...]]
    sources: dict[str, dict[str, tuple[str, ...]]]


def _natural_key(name: str):
    return _sample_key(name), name


def _screen_task(task_dir: Path) -> dict:
    manifest = task_dir / ".screen.json"
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read screen manifest {manifest}: {exc}") from exc
    if not isinstance(value, dict) or value.get("kind") != "partidesign.screen.v1":
        raise ValueError(f"{manifest}: expected kind 'partidesign.screen.v1'")
    task = value.get("task")
    if not isinstance(task, dict) or not isinstance(task.get("name"), str) or not task["name"]:
        raise ValueError(f"{manifest}: task.name must be a nonempty string")
    methods = task.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError(f"{manifest}: task.methods must be a nonempty list")
    _names(methods)
    if any(name in {".", ".."} or "/" in name or "\\" in name for name in methods):
        raise ValueError(f"{manifest}: task.methods must contain simple method names")
    if len(methods) != len(set(methods)):
        raise ValueError(f"{manifest}: duplicate task.methods")
    if not isinstance(task.get("rows"), list):
        raise ValueError(f"{manifest}: task.rows must be a list")
    return task


def discover_chemical_jobs(paths, *, tasks=(), methods=()) -> list[ChemicalJob]:
    """Visit only the existing validation levels and immediate screen children."""
    roots = (paths,) if isinstance(paths, (str, Path)) else tuple(paths)
    tasks, methods = _names(tasks), _names(methods)
    selected_tasks, selected_methods = set(tasks), set(methods)
    result = [ChemicalJob(job.task_dir, job.task_dir.name, job.method,
                          job.method_dir, "validation")
              for job in discover_jobs(roots, tasks=tasks, methods=methods)]
    seen = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if (root / ".screen.json").is_file():
            task_dirs = [root]
        else:
            task_dirs = [child for child in sorted(root.iterdir())
                         if child.is_dir() and not child.name.startswith(".")
                         and (child / ".screen.json").is_file()]
        for task_dir in task_dirs:
            task_dir = task_dir.resolve()
            if task_dir in seen:
                continue
            seen.add(task_dir)
            task = _screen_task(task_dir)
            if selected_tasks and task["name"] not in selected_tasks:
                continue
            for method in sorted(task["methods"]):
                if selected_methods and method not in selected_methods:
                    continue
                result.append(ChemicalJob(task_dir, task["name"], method,
                                          task_dir / "cluster" / method, "screen"))
    return result


def _load_validation(job: ChemicalJob, criteria: Iterable[str]) -> ChemicalInput:
    data = load_job(ClusterJob(job.task_dir, job.report_dir), criteria=criteria,
                    require_structures=False)
    samples = {}
    for name, sample in data.samples.items():
        ligands = tuple(sorted((chain["chain"], chain["sequence"])
                               for chain in sample.design_chains
                               if chain["modality"] == "ligand"))
        if ligands:
            samples[name] = ChemicalSample(name, sample.structure, ligands)
    cohorts = {criterion: tuple(name for name in names if name in samples)
               for criterion, names in data.groups.items()}
    return ChemicalInput(samples, cohorts, data.sources)


def _load_screen(job: ChemicalJob, criteria: Iterable[str]) -> ChemicalInput:
    if _names(criteria):
        raise ValueError("--criteria applies to validation success tables; screen outputs have no passing criteria")
    task = _screen_task(job.task_dir)
    manifest = job.task_dir / ".screen.json"
    candidates = {}
    for index, row in enumerate(task["rows"]):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"{manifest}: task.rows[{index}].id must be a nonempty string")
        if row["id"] in candidates:
            raise ValueError(f"{manifest}: duplicate candidate id {row['id']!r}")
        candidates[row["id"]] = row
    scores_path = job.task_dir / "scores.csv"
    fields, rows = _read_csv(scores_path)
    required = {"id", "modality", "method", "status", "structure_path"}
    if not required.issubset(fields):
        raise ValueError(f"{scores_path}: missing columns: {', '.join(sorted(required.difference(fields)))}")
    samples = {}
    seen = set()
    for index, row in enumerate(rows, 2):
        if row["method"] != job.method:
            continue
        name = row["id"]
        context = f"{scores_path}:{index} ({name!r})"
        if name in seen:
            raise ValueError(f"{context}: duplicate candidate/method row")
        seen.add(name)
        if row["status"] != "complete" or row["modality"] != "ligand":
            continue
        authored = candidates.get(name)
        if authored is None:
            raise ValueError(f"{context}: candidate is missing from authoritative {manifest.name}")
        if authored.get("modality") != "ligand" or authored.get("error"):
            raise ValueError(f"{context}: complete ligand conflicts with screen manifest metadata")
        smiles = authored.get("seq")
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(f"{manifest}: candidate {name!r} has no authoritative ligand SMILES in seq")
        if not row["structure_path"]:
            raise ValueError(f"{context}: complete candidate has no structure_path")
        # Candidate IDs are labels, never filenames. The screen-owned path is
        # report provenance only; chemical grouping never opens the structure.
        structure = Path(row["structure_path"]).expanduser()
        if not structure.is_absolute():
            structure = job.task_dir / structure
        samples[name] = ChemicalSample(name, structure.resolve(), ((None, smiles),))
    return ChemicalInput(samples, {"": tuple(sorted(samples, key=_natural_key))}, {})


def load_chemical_job(job: ChemicalJob, *, criteria=()) -> ChemicalInput:
    if job.source == "screen":
        return _load_screen(job, criteria)
    return _load_validation(job, criteria)


def _chemical_identity(smiles: str, *, context: str, group_by: str):
    from rdkit import Chem, rdBase
    from rdkit.Chem.Scaffolds import MurckoScaffold
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"{context}: missing authoritative ligand SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"{context}: invalid authoritative ligand SMILES {smiles!r}")
    # Canonical ordering alone: no tautomer conversion, charge neutralization,
    # fragment stripping, generic scaffold conversion or stereochemistry loss.
    identity = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if group_by == "identity":
        return identity, identity
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        return identity, "identity:" + identity
    return identity, "murcko:" + Chem.MolToSmiles(
        scaffold, canonical=True, isomericSmiles=True)


def group_candidates(data: ChemicalInput, *, group_by: str):
    if group_by not in {"identity", "murcko"}:
        raise ValueError("--group-by must be identity or murcko")
    identities, keys = {}, {}
    for name in sorted(data.samples, key=_natural_key):
        identity, group = [], []
        for chain, smiles in data.samples[name].ligands:
            canonical, key = _chemical_identity(
                smiles, context=f"candidate {name!r}, ligand {chain!r}", group_by=group_by)
            identity.append((chain, canonical))
            group.append((chain, key))
        identities[name], keys[name] = tuple(identity), tuple(group)
    cohorts = {}
    for criterion, names in sorted(data.cohorts.items()):
        groups = {}
        for name in sorted(names, key=_natural_key):
            groups.setdefault(keys[name], []).append(name)
        cohorts[criterion] = tuple(tuple(members) for members in groups.values())
    return identities, keys, cohorts


def process_chemical_job(job: ChemicalJob, group_by: str, criteria: tuple[str, ...],
                         scratch_root: Path, *, plan=False) -> dict:
    _single_core()
    started = time.monotonic()
    data = load_chemical_job(job, criteria=criteria)
    identities, keys, cohorts = group_candidates(data, group_by=group_by)
    result = {"task": str(job.task_dir), "method": job.method, "group_by": group_by,
              "candidates": len(data.samples), "cohorts": len(cohorts),
              "groups": sum(len(groups) for groups in cohorts.values())}
    if plan:
        return result
    base = dict(run=job.task_dir.parent.name, task=job.task_name, method=job.method,
                cohort="screen_complete" if job.source == "screen" else "validation_pass",
                group_by=group_by, representative_policy="first_natural_id",
                chemistry_source=".screen.json:task.rows[].seq" if job.source == "screen"
                else "result.csv:design_chains[].sequence")
    details, summaries = [], []
    for criterion, groups in cohorts.items():
        names = data.cohorts[criterion]
        summaries.append(dict(base, criterion=criterion, n_candidates=len(names),
                              n_identities=len({identities[name] for name in names}),
                              n_groups=len(groups), n_singletons=sum(len(g) == 1 for g in groups),
                              largest_group=max(map(len, groups), default=0)))
        for group_id, members in enumerate(groups, 1):
            representative = members[0]
            for name in members:
                ligands = [{"chain": chain, "canonical_isomeric_smiles": smiles}
                           for chain, smiles in identities[name]]
                details.append(dict(
                    base, criterion=criterion, candidate_id=name,
                    structure_path=str(data.samples[name].structure),
                    ligands=json.dumps(ligands, separators=(",", ":")),
                    chemical_group_key=json.dumps(keys[name], separators=(",", ":")),
                    group_id=group_id, group_size=len(members), representative=representative,
                    representative_file=str(data.samples[representative].structure),
                    is_representative=int(name == representative),
                    passing_branches=json.dumps(data.sources.get(criterion, {}).get(name, ())),
                ))
    job.report_dir.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    report = f"chemical_{group_by}_clusters.csv"
    summary = f"chemical_{group_by}_summary.csv"
    _write_reports(job.report_dir, details, summaries, scratch_root, report_specs=(
        (report, _DETAIL_FIELDS, details), (summary, _SUMMARY_FIELDS, summaries)))
    result.update(report=str(job.report_dir / report),
                  summary=str(job.report_dir / summary),
                  seconds=round(time.monotonic() - started, 3),
                  pid=os.getpid(), thread_limit=1)
    return result


def run(paths, *, tasks=(), methods=(), criteria=(), group_by, workers=16, plan=False) -> int:
    if group_by not in {"identity", "murcko"}:
        raise ValueError("--group-by must be identity or murcko")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise ValueError("--workers must be an integer in [1,16]")
    criteria = _names(criteria)
    jobs = discover_chemical_jobs(paths, tasks=tasks, methods=methods)
    if not jobs:
        raise ValueError("No validation success.csv tables or screen task manifests matched")
    if criteria and any(job.source == "screen" for job in jobs):
        raise ValueError("--criteria applies to validation success tables; screen outputs have no passing criteria")
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    active = min(workers, available, len(jobs))
    print(f"Chemical grouping directories: {len(jobs)}; group_by={group_by}; "
          f"workers: {active}; CPU threads per worker: 1", flush=True)
    saved_environment = {name: os.environ.get(name) for name in _THREAD_VARS}
    _single_core()
    failures = 0
    scratch_root = Path.cwd() / ".cache" / "cluster"
    try:
        if plan:
            for job in jobs:
                result = process_chemical_job(job, group_by, criteria, scratch_root, plan=True)
                print("PLAN " + json.dumps(result, sort_keys=True), flush=True)
            return 0
        with ProcessPoolExecutor(max_workers=active,
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_single_core) as pool:
            pending = {pool.submit(process_chemical_job, job, group_by, criteria, scratch_root): job
                       for job in jobs}
            for future in as_completed(pending):
                job = pending[future]
                try:
                    result = future.result()
                except Exception as error:
                    failures += 1
                    print(f"FAILED {job.task_dir} method={job.method}: "
                          f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
                    continue
                print("DONE " + json.dumps(result, sort_keys=True), flush=True)
    finally:
        for name, value in saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    print(f"Completed directories: {len(jobs) - failures}/{len(jobs)}", flush=True)
    return 2 if failures else 0
