"""Lightweight parser shared by the installed and module entry points."""
from __future__ import annotations

import argparse
from pathlib import Path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("paths", nargs="+", type=Path, metavar="RUN_TASK_OR_SUCCESS",
                        help="Run, task or validation-success directories; chemical modes also accept screen outputs")
    parser.add_argument("--task", action="append", help="Select task names (repeatable)")
    parser.add_argument("--method", action="append", help="Select existing validation or screen methods (repeatable)")
    parser.add_argument("--criteria", action="append", help="Select passing criteria (repeatable; kept separate)")
    parser.add_argument("--group-by", choices=("identity", "murcko"),
                        help="Group designed ligands by exact chemical identity or Murcko scaffold; default: polymer TM clustering")
    parser.add_argument("--tm-threshold", type=float, default=0.5,
                        help="Mean of the two length-normalized TM-scores (default: 0.5)")
    parser.add_argument("--workers", type=int, default=16,
                        help="One CPU process per selected directory/method; concurrency 1..16 (default: 16)")
    parser.add_argument("--usalign", type=Path,
                        help="USalign executable; default: active Python environment, then PATH")
    parser.add_argument("--plan", action="store_true",
                        help="Inspect selected inputs and comparison/group counts without writing")


def run(args: argparse.Namespace) -> int:
    from .runner import run as run_cluster
    return run_cluster(args.paths, tasks=args.task or (), methods=args.method or (),
                       criteria=args.criteria or (), threshold=args.tm_threshold,
                       workers=args.workers, executable=args.usalign, plan=args.plan,
                       group_by=args.group_by)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.cluster",
                                     description="Cluster design polymers with US-align or explicitly group designed ligands by chemistry")
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted")
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        import sys
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
