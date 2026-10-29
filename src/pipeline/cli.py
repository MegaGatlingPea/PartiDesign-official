"""Public pipeline arguments; importing this module never loads a model."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


class _SingleValue(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        seen = getattr(namespace, "_single_options", set())
        if self.dest in seen:
            parser.error(f"{option_string} may be supplied only once")
        setattr(namespace, "_single_options", seen | {self.dest})
        setattr(namespace, self.dest, self.const if self.nargs == 0 else values)


def _single(parser, *names, **kwargs):
    parser.add_argument(*names, action=_SingleValue, **kwargs)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("task_file", type=Path, help="Native design task YAML; select task names with --task")
    _single(parser, "--preset", choices=("auto", "protein", "nuc", "ppi"),
            help="Workflow preset, resolved per task (default: auto)")
    parser.add_argument("--task", action="append", help="Select a task name (repeatable)")
    _single(parser, "--config", type=Path, help="Pipeline workflow YAML (recycle, per-round operations and final validation)")
    _single(parser, "--recycle", type=int, help="Total rounds including first validation (default: 3)")
    _single(parser, "--first-mpnn-count", type=int, help="Auto MPNN proposals per first-round structure (default: 1)")
    _single(parser, "--mpnn-count", type=int, help="Auto MPNN proposals per later-round parent (default: 4)")
    _single(parser, "--method", choices=("esmfold2-fast", "af2_ig", "af3"),
            help="Iteration refold backend (default: esmfold2-fast)")
    _single(parser, "--num-clusters", type=int, help="First-round number of sequence clusters (default: 200)")
    _single(parser, "--max-pass", type=int, help="Later-round task-wide upper limit (default: 200)")
    _single(parser, "--max-pass-ratio", type=float, help="Later-round fraction of unique proposed design sequences")
    _single(parser, "--relax", choices=("true", "false"),
            help="Whole-complex OpenMM between rounds (default: false)")
    _single(parser, "--partial-fraction", type=float,
            help="Partial diffusion fraction of the 400-step grid (default: 0.625, last 250 intervals from index 150)")
    _single(parser, "--partial-replicas", type=int,
            help="Partial diffusion proposals per selected parent (default: 4)")
    parser.add_argument("--criteria", action="append", help="Iteration criteria preset (repeatable)")
    parser.add_argument("--final-criteria", action="append", help="Final validation criteria preset (repeatable)")
    _single(parser, "--presets", type=Path, help="Additional native criteria definitions YAML")
    _single(parser, "--from-run", type=Path, help="Use existing generation results without modifying them")
    _single(parser, "--resume", nargs=0, const=True, default=False,
            help="Resume an existing run with identical effective configuration")
    _single(parser, "--keep-cache", nargs=0, const=True, default=False,
            help="Keep raw prediction caches after completion for metric recomputation and prediction reuse")
    _single(parser, "--model-dir", type=Path, help="Local model root")
    _single(parser, "--ckpt", type=Path, help="Generation checkpoint or routing bundle")
    _single(parser, "--batch-size", type=int, help="FM batch size (default: native 16)")
    _single(parser, "--num-workers", type=int, help="FM data workers (default: native 4)")
    _single(parser, "--route", choices=("true", "false"), help="Enable generation checkpoint routing")
    _single(parser, "--route-ckpt", choices=("null", "omni", "prot", "nuc", "ligand", "antibody", "pep"),
            help="Generation bundle member; null selects automatically")
    _single(parser, "--compile", choices=("auto", "true", "false"), help="Generation compile policy")
    _single(parser, "--compile-strategy", choices=("auto", "dynamic", "bucket"),
            help="Generation compile strategy")
    _single(parser, "--execution-backend", choices=("auto", "native", "bioir"),
            help="Generation execution backend")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Native generation override (repeatable)")
    _single(parser, "--out", type=Path, help="Run directory (default: inference/RUN_NAME)")
    _single(parser, "--run-name", help="Run directory label (default: pipeline)")
    _single(parser, "--seed", type=int, help="Stable workflow seed (default: 2026)")
    _single(parser, "--worker-gpus", type=float, help="Fractional validation GPU budget per worker")
    _single(parser, "--worker-cpus", type=float, help="Validation CPUs per worker (default: native 1)")
    _single(parser, "--plan", nargs=0, const=True, default=False,
            help="Show effective configuration/capabilities without models or output")
    _single(parser, "--show-config", nargs=0, const=True, default=False,
            help="Show full internal configuration for diagnostics")


def run(args):
    from .runner import run as run_workflow
    return run_workflow(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run design, refinement, and validation in one command")
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
