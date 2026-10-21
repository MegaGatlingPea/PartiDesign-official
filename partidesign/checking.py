"""Read-only CPU preflight and explicitly requested model execution."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .paths import code_root, inference_config_dir, resource_root, community_source


class Report:
    def __init__(self, display=None) -> None:
        self.failed = 0
        self.display = display

    def emit(self, label: str, status: str, detail: str) -> None:
        if self.display is None:
            print(f"{status} {label}: {detail}", flush=True)
        else:
            self.display.emit(label, status, detail)
        self.failed += status == "FAIL"

    def attempt(self, label: str, callback: Callable[[], str]) -> bool:
        if self.display is not None:
            self.display.begin(label)
        try:
            detail = callback()
        except Exception as error:
            self.emit(label, "FAIL", f"{type(error).__name__}: {error}")
            return False
        self.emit(label, "PASS", detail)
        return True


def _cpu_child_environment() -> dict[str, str]:
    """Return the CPU checker environment with a closed package path."""
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        JAX_PLATFORMS="cpu",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        PIP_NO_INDEX="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    environment["PYTHONPATH"] = str(code_root().resolve())
    return environment


def _environment() -> str:
    modules = ("torch", "lightning", "hydra", "omegaconf", "yaml", "numpy", "biotite", "rdkit")
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError("missing core dependencies: " + ", ".join(missing))
    directory = inference_config_dir()
    required = ("design.yaml", "score.yaml", "relax.yaml", "validate.yaml", "criteria.yaml", "sequence.yaml")
    absent = [str(directory / name) for name in required if not (directory / name).is_file()]
    native_ccd = resource_root() / "configs" / "resources" / "chemistry" / "ccd.json.gz"
    if not native_ccd.is_file():
        absent.append(str(native_ccd))
    if absent:
        raise FileNotFoundError("missing packaged resources: " + ", ".join(absent))
    return f"Python {sys.version.split()[0]}; core packages discoverable and packaged inference resources"


def _cpu_child(target: str, root: Path, *, checkpoint: str | None = None,
               variant: str | None = None, require_bundle: bool = False,
               context_mode: str = "auto", weights: str = "raw", model_type: str | None = None,
               ca_only: bool = False, packing_only: bool = False, sequence_only: bool = False, antibody: bool = False) -> str:
    command = [sys.executable, "-m", "partidesign.model_checks", target,
               "--model-dir", str(root)]
    if checkpoint is not None:
        command += ["--checkpoint", checkpoint]
    if variant is not None:
        command += ["--variant", variant]
    if require_bundle:
        command.append("--require-bundle")
    command += ["--context-mode", context_mode, "--weights", weights]
    if model_type:
        command += ["--model-type", model_type]
    if ca_only:
        command.append("--ca-only")
    if packing_only:
        command.append("--packing-only")
    if sequence_only:
        command.append("--sequence-only")
    if antibody:
        command.append("--antibody")
    env = _cpu_child_environment()
    # Source checkout execution also works from an unrelated invocation directory.
    result = subprocess.run(
        command, cwd=str(code_root()), env=env, capture_output=True, text=True
    )
    if result.returncode:
        tail = (result.stderr or result.stdout).strip()[-3000:]
        raise RuntimeError(f"CPU load exited {result.returncode}: {tail}")
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "strict CPU load"


def _source_assets(target: str) -> str:
    names = {"proteinmpnn": "proteinmpnn", "solublempnn": "proteinmpnn",
             "ligandmpnn": "ligandmpnn", "cyclicmpnn": "cyclicmpnn",
             "nampnn": "na_mpnn", "af2": "colabdesign",
             "esmfold2-fast": "esmfold2", "esmfold2-standard": "esmfold2",
             "af3": "af3", "af3-la": "af3"}
    if target in names:
        source = community_source(names[target])
        if not source.is_dir():
            label = "AF3 SDK" if target in {"af3", "af3-la"} else "bundled backend source"
            raise FileNotFoundError(f"{label} source is missing: {source}")
        if target in {"af3", "af3-la"}:
            package = source / "src" / "alphafold3"
            converter = package / "constants" / "converters" / "ccd_pickle_gen.py"
            if not package.is_dir() or not converter.is_file():
                raise FileNotFoundError(f"AF3 SDK source is incomplete: {source}")
    return "bundled source available"


def _msa_settings_for_check(args, root: Path) -> dict:
    """Build an AF3 MSA settings mapping without loading an AF3 model."""
    from src.validate.config import DEFAULT_MSA_CONFIG, validate_msa_settings

    settings = copy.deepcopy(DEFAULT_MSA_CONFIG)
    values = {
        "mode": args.msa,
        "cache_dir": args.msa_cache_dir,
        "cache_profile": args.msa_profile,
        "database_dir": args.msa_database_dir,
        "binaries_dir": args.msa_binaries_dir,
        "search_workers": args.msa_workers,
        "cpu_budget": args.msa_cpus,
        "jackhmmer_n_cpu": args.jackhmmer_cpus,
        "nhmmer_n_cpu": args.nhmmer_cpus,
        "timeout_seconds": args.msa_timeout,
        "templates": args.msa_templates,
        "max_template_date": args.msa_template_date,
    }
    for key, value in values.items():
        if value is not None:
            if key in {"templates"}:
                value = value == "true"
            if key == "cpu_budget" and value != "auto":
                try:
                    value = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError("af3.msa.cpu_budget must be 'auto' or a positive integer") from exc
            if key in {"cache_dir", "database_dir", "binaries_dir"}:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = root / path
                value = str(path)
            settings[key] = value
    if args.msa_refresh:
        settings["refresh"] = True
    validate_msa_settings(settings)
    return settings


def _msa_cache_check(args, root: Path, modalities: tuple[str, ...]) -> str:
    settings = _msa_settings_for_check(args, root)
    if settings["mode"] == "off":
        return "MSA disabled; database and HMMER checks skipped"
    from src.validate.af3_msa import AF3MSACache

    # Keep this adapter tolerant of the small constructor spelling used by
    # standalone cache implementations while calling only their read-only
    # readiness method.  No search task, model, or GPU is started here.
    parameters = inspect.signature(AF3MSACache).parameters
    positional = []
    keyword = {}
    # Prefer a named settings/config argument.  The native implementation
    # takes (repo_root, settings), so passing settings positionally would
    # accidentally fill repo_root before the explicit root keyword below.
    if "config" in parameters:
        keyword["config"] = settings
    elif "settings" in parameters:
        keyword["settings"] = settings
    else:
        keyword.update(settings)
    for name in ("repo_root", "root", "model_root"):
        if name in parameters:
            keyword[name] = root
            break
    if "readonly" in parameters:
        keyword["readonly"] = True
    cache = AF3MSACache(*positional, **keyword)
    result = cache.check(modalities)
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        rendered = json.dumps(result, sort_keys=True, default=str)
        if result.get("ready") is False:
            raise RuntimeError(f"AF3 MSA resources are not ready: {rendered}")
        return rendered
    return str(result)


def _task_preflight(runner) -> str:
    import numpy as np
    from src.design.core.batch import collate_compiled_samples
    from src.design.core.compiler import prepare_task_source, compile_request
    from src.design.core.records import DesignInputError, DesignSampleError
    from src.design.core.runtime import _preflight_trusted_ccd_components
    from src.design.core.config import validate_sampling_preflight

    validate_sampling_preflight(runner.config.sampling)
    ccd = None if runner.config.ccd_path is None else Path(str(runner.config.ccd_path))
    representatives = []
    seen_tasks = set()
    for request in runner.requests:
        if request.task.name in seen_tasks:
            continue
        seen_tasks.add(request.task.name)
        representatives.append(request)
    representatives = tuple(representatives)
    trusted = _preflight_trusted_ccd_components(representatives, ccd_path=ccd)
    tokens = []
    compiled_samples = []
    sources = dict(getattr(runner, "_routing_sources", {}))
    for request in representatives:
        if (request.task.format == "normal" and request.task.input_path is not None
                and request.task.name not in sources):
            try:
                sources[request.task.name] = prepare_task_source(request.task, ccd_path=ccd)
            except (DesignInputError, DesignSampleError, OSError) as error:
                sources[request.task.name] = error
        options = {}
        if request.task.name in sources:
            prepared = sources[request.task.name]
            if isinstance(prepared, Exception):
                raise prepared
            options["prepared_source"] = prepared
        compiled = compile_request(request, ccd_path=ccd,
            trusted_ccd_components=trusted,
            rng=np.random.default_rng(int(runner.config.seed) + int(request.global_ordinal)), **options)
        compiled_samples.append(compiled)
        batch = collate_compiled_samples([compiled])
        canonical = batch.canonical
        count = int(canonical["token_mask"].sum())
        if not count:
            raise ValueError(f"{request.task.name}: no model tokens after compilation")
        tokens.append(f"{request.task.name}/plan{request.plan.index}={count}")
    runner._check_compiled_samples = tuple(compiled_samples)
    runner._routing_sources = sources
    return ("CPU masks/crop/chemistry compiled with fixed seed; "
            f"{len(runner.tasks)} selected tasks, {len(representatives)} representative "
            f"requests, {len(runner.requests)} planned requests: " + ", ".join(tokens)
            + "; other length/crop stochastic outcomes were not checked "
            "(stochastic length/crop outcomes beyond this seed are not certified)")


def _required_task_targets(runner) -> list[str]:
    from omegaconf import OmegaConf
    from src.design.core.runtime import _uses_type_bond
    cfg = OmegaConf.to_container(runner.config, resolve=True)
    targets = []
    if any(task.score for task in runner.tasks):
        targets.append("partiscore")
    if any(_uses_type_bond(task) for task in runner.tasks):
        targets.append("chemistry")
    reward = cfg["reward"]
    target = str(reward.get("_target_") or "")
    if target.endswith("ESMFold2Reward"):
        mode = reward.get("esmfold2", {}).get("mode", "fast")
        targets.append("esmfold2-" + mode)
    elif target.endswith("AF2Reward"):
        targets.append("af2")
    elif target.endswith("PartiScoreReward"):
        targets.append("partiscore")
    requested = reward.get("sequence", {}).get("source", "design")
    if target and requested != "design":
        from src.design.core.runtime import _sequence_route_candidate, _validate_af2_sequence_routes
        from src.design.rewards.sequence import resolve_reward_sequence_source
        _validate_af2_sequence_routes(runner.requests, reward)
        method = "partiscore" if target.endswith("PartiScoreReward") else "esmfold2"
        for compiled in getattr(runner, "_check_compiled_samples", ()):
            source = resolve_reward_sequence_source(_sequence_route_candidate(compiled),
                requested, reward_method=method)
            if source == "design":
                continue
            if source == "proteinmpnn":
                family = reward["sequence"]["proteinmpnn"]["model_type"]
                source = "solublempnn" if family == "soluble_mpnn" else "proteinmpnn"
            if source == "na_mpnn":
                source = "nampnn"
            targets.append(source)
            if method == "partiscore":
                targets.append("ligandmpnn")
    return list(dict.fromkeys(targets))


def _task_target_options(runner, target: str) -> tuple[list[Path] | None, dict]:
    from .paths import community_asset
    cfg = runner.config
    if target == "partiscore":
        path = Path(str(cfg.scorer.checkpoint))
        return [path], {"checkpoint": str(path), "weights": str(cfg.scorer.weights)}
    sequence = cfg.reward.get("sequence", {})
    names = {"proteinmpnn": "proteinmpnn", "solublempnn": "proteinmpnn",
             "cyclicmpnn": "cyclicmpnn", "nampnn": "na_mpnn", "ligandmpnn": "ligandmpnn"}
    if target in names:
        name = names[target]
        if target == "ligandmpnn" and str(cfg.reward.get("_target_", "")).endswith("PartiScoreReward"):
            requested = str(sequence.get("source", "design"))
            routes = []
            from src.design.core.runtime import _sequence_route_candidate
            from src.design.rewards.sequence import resolve_reward_sequence_source
            for compiled in getattr(runner, "_check_compiled_samples", ()):
                routes.append(resolve_reward_sequence_source(_sequence_route_candidate(compiled),
                    requested, reward_method="partiscore"))
            if "ligandmpnn" not in routes:
                path = community_asset("ligandmpnn") / "checkpoint" / "ligandmpnn_sc_v_32_002_16.pt"
                return [path], {"packing_only": True}
        backend = sequence.get(name, {})
        if "checkpoint" in backend:
            path = Path(str(backend.checkpoint))
            options = {"checkpoint": str(path)}
            if target in {"proteinmpnn", "solublempnn", "ligandmpnn"}:
                options["model_type"] = str(backend.model_type)
                options["ca_only"] = bool(backend.get("ca_only", False))
            paths = [path]
            if target == "ligandmpnn":
                options["sequence_only"] = not str(cfg.reward.get("_target_", "")).endswith("PartiScoreReward")
            if target == "ligandmpnn" and str(cfg.reward.get("_target_", "")).endswith("PartiScoreReward"):
                paths.append(community_asset("ligandmpnn") / "checkpoint" / "ligandmpnn_sc_v_32_002_16.pt")
            return paths, options
    if target == "chemistry":
        path = community_asset("esmfold2") / "common" / "ccd.pkl"
        return [path], {"checkpoint": str(path)}
    return None, {}


def _readback(directory: Path, expected: dict[str, int]) -> str:
    import csv
    import numpy as np
    from src.design.utils.reader import read_structure
    tables = list(directory.rglob("result.csv"))
    if not tables:
        raise ValueError("smoke produced no result.csv")
    count = 0
    observed = {}
    for table in tables:
        with table.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"empty result.csv: {table}")
        if table.parent.name in observed:
            raise ValueError(f"duplicate task output: {table}")
        observed[table.parent.name] = len(rows)
        for row in rows:
            name = row.get("filename", "")
            if not name or row.get("error"):
                raise ValueError(f"unsuccessful result row: {table}: {row.get('error') or name}")
            path = (table.parent / name).resolve()
            path.relative_to(table.parent.resolve())
            atoms = read_structure(path)
            if len(atoms) == 0 or not np.isfinite(atoms.coord).all():
                raise ValueError(f"empty or nonfinite structure: {path}")
            count += 1
    if observed != expected:
        raise ValueError(f"output accounting differs: expected={expected}, observed={observed}")
    return f"{count} successful CIF/CSV rows independently read back"


def _native_smoke(runner, args) -> str:
    from .cli import build_parser, build_design_runner
    from .paths import model_dir
    manager = tempfile.TemporaryDirectory(prefix="partidesign-smoke-") if args.out is None else nullcontext(str(args.out.expanduser().resolve()))
    with manager as directory:
        output = Path(directory) / "design"
        if runner is None:
            task_file = Path(directory) / "smoke_tasks.yaml"
            task_file.parent.mkdir(parents=True, exist_ok=True)
            task_file.write_text("protein_smoke:\n  sample_num: 1\n  design_chains: [{P: protein}]\n  design_length: {P: [12]}\n  score: false\n  postprocess: false\n")
            smoke_args = build_parser().parse_args(["design", str(task_file), "--out", str(output),
                "--batch-size", "1", "--num-workers", "0", "--compile", "false"])
            runner = build_design_runner(smoke_args, model_dir(args.model_dir))
        else:
            # Smoke executes one representative request per task at the original length.
            first = {}
            for request in runner.requests:
                first.setdefault(request.task.name, request)
            runner.requests = tuple(first.values())
            runner.config.output_dir = str(output)
            runner.config.batch_size = 1
            runner.config.num_workers = 0
        expected = {}
        for request in runner.requests:
            expected[request.task.name] = expected.get(request.task.name, 0) + 1
        from src.design.core.runtime import run_runtime
        from lightning.pytorch import Trainer
        import torch
        def one_device_trainer(*, precision):
            gpu = torch.cuda.is_available()
            return Trainer(accelerator="gpu" if gpu else "cpu", devices=1,
                precision=precision if gpu else "32-true", logger=False,
                enable_checkpointing=False, enable_model_summary=False,
                inference_mode=True, use_distributed_sampler=False)
        options = {}
        if hasattr(runner, "_routing_sources"):
            options["prepared_sources"] = runner._routing_sources
        result = run_runtime(runner.loaded, runner.tasks, runner.requests,
                             trainer_factory=one_device_trainer, **options)
        if result.failed or result.postprocess_failed or result.score_failed:
            raise RuntimeError(f"smoke failed: generated={result.failed}, packing={result.postprocess_failed}, scoring={result.score_failed}")
        return ("400-step " + ("compiled" if runner.config.model.compile.enabled else "eager")
                + " smoke on one device; " + _readback(output, expected))


def run_check(args, root: Path) -> int:
    from .ui import CheckDisplay
    with CheckDisplay(root) as display:
        return _run_check(args, root, Report(display))


def _run_check(args, root: Path, report: Report) -> int:
    from .download import TARGETS, list_targets
    report.emit("model-dir", "PASS", str(root))
    if not report.attempt("environment", _environment):
        return 1
    runner = None
    route = None
    if args.task_file is not None:
        from .cli import build_design_runner
        from src.design.model.routing import resolve_model_route, prepare_routing_sources, task_model_preferences
        holders = []
        if not report.attempt("task-configuration", lambda: _runner_summary(args, root, holders)):
            return 1
        runner = holders[0]
        route_holder = []
        def route_summary():
            ccd = None if runner.config.ccd_path is None else Path(str(runner.config.ccd_path))
            automatic = runner.config.route and runner.config.route_ckpt is None
            sources = prepare_routing_sources(runner.tasks, ccd_path=ccd) if automatic else {}
            runner._routing_sources = sources
            resolved = resolve_model_route(runner.tasks, enabled=runner.config.route,
                override=runner.config.route_ckpt, prepared_sources=sources, ccd_path=ccd)
            route_holder.append(resolved)
            detail = f"{resolved[0] or 'standalone'} ({resolved[1]}): {resolved[2]}"
            if automatic:
                preferences = task_model_preferences(runner.tasks, prepared_sources=sources)
                detail += "; " + "; ".join(f"{p.task_name} -> {p.member}: {p.reason}" for p in preferences)
            return detail
        if not report.attempt("route", route_summary):
            return 1
        route = route_holder[0]
        report.attempt("task-structure", lambda: _task_preflight(runner))
        if any(task.postprocess for task in runner.tasks):
            report.attempt("postprocess-resources", lambda: _postprocess_resources(runner))
        else:
            report.emit("postprocess", "SKIP", "disabled for selected tasks")
        target_holder = []
        if not report.attempt("task-backends", lambda: _target_summary(runner, target_holder)):
            return 1
        targets = target_holder[0]
        checkpoint = Path(str(runner.config.model.checkpoint))
        report.attempt("partidesign-assets", lambda: _file_summary(checkpoint))
        if args.load or args.smoke:
            report.attempt("partidesign-load", lambda: _cpu_child("partidesign", root,
                checkpoint=str(checkpoint), variant=route[0], require_bundle=route[1] == "forced",
                context_mode=str(runner.config.model.context_constraint_mode),
                antibody=any(task.format == "antibody" for task in runner.tasks)))
    else:
        targets = list(TARGETS) if args.all else (args.model or ["partidesign"])
    if any(target in {"af3", "af3-la"} for target in targets):
        if not report.attempt(
            "af3-msa",
            lambda: _msa_cache_check(args, root, ("protein", "rna")),
        ):
            # Continue with the ordinary AF3 resource/source checks so the
            # report still identifies every missing local prerequisite.
            pass
    records = {item["target"]: item for item in list_targets(root)}
    ready_targets = []
    for target in targets:
        record = records[target]
        selected_paths, options = _task_target_options(runner, target) if runner is not None else (None, {})
        if selected_paths is not None:
            if not report.attempt(target + "-assets", lambda paths=selected_paths:
                    "; ".join(_file_summary(path) for path in paths)):
                continue
        elif record["status"] != "installed":
            report.emit(target + "-assets", "FAIL", record.get("error") or "required resource is missing")
            continue
        if not report.attempt(target + "-source", lambda target=target: _source_assets(target)):
            continue
        if selected_paths is None:
            report.emit(target + "-assets", "PASS", "complete local resource set")
        if args.load or args.smoke:
            if report.attempt(target + "-load", lambda target=target, options=options: _cpu_child(target, root, **options)):
                ready_targets.append(target)
        else:
            report.emit(target + "-load", "NOT_RUN", "--load was not requested")
    for target in TARGETS:
        if target not in targets and not (runner is not None and target == "partidesign"):
            report.emit(target, "SKIP", "unused optional model")
    if args.smoke:
        if runner is not None:
            if not report.failed:
                report.attempt("native-smoke", lambda: _native_smoke(runner, args))
            else:
                report.emit("native-smoke", "NOT_RUN", "required task preflight failed")
        elif targets == ["partidesign"] and "partidesign" in ready_targets:
            report.attempt("native-smoke", lambda: _native_smoke(None, args))
        else:
            from .model_checks import community_smoke
            for target in ready_targets:
                report.attempt(target + "-smoke", lambda target=target: community_smoke(target, root, args.out))
    return 1 if report.failed else 0


def _runner_summary(args, root, holders) -> str:
    from .cli import build_design_runner
    runner = build_design_runner(args, root)
    holders.append(runner)
    return f"{len(runner.tasks)} selected tasks, {len(runner.requests)} expanded requests"


def _file_summary(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"checkpoint is missing or empty: {path}")
    return str(path)

def _target_summary(runner, holders) -> str:
    targets = _required_task_targets(runner)
    holders.append(targets)
    return ", ".join(targets) if targets else "native generation only"


def _postprocess_resources(runner) -> str:
    from omegaconf import OmegaConf
    from src.design.postproc.runner import preflight_postprocessing
    config = OmegaConf.to_container(runner.config.postprocess, resolve=True)
    platform = str(config.get("platform", "CUDA")).upper()
    ray = bool(config.get("use_ray", True))
    if (platform, ray) not in {("CUDA", True), ("CPU", False)}:
        raise ValueError("OpenMM accepts only CUDA+Ray or CPU/no-Ray")
    if str(runner.config.reward.get("_target_", "")).endswith("AF2Reward"):
        raise ValueError("AF2 fold rewards cannot be combined with OpenMM postprocessing")
    if ray and importlib.util.find_spec("ray") is None:
        raise ValueError("Ray dependency required by configured CUDA postprocessing")
    # Validate force fields and AmberTools using the native CPU diagnostic. A
    # default read-only check must never activate a CUDA OpenMM context.
    preflight_postprocessing(dict(config, platform="CPU", use_ray=False))
    return "native CPU dependency/force-field/AmberTools check; configured GPU context NOT_RUN (requires --smoke)"
