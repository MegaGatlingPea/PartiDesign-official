"""Lightweight postfilter argument declarations and dispatch."""
from pathlib import Path

def add_arguments(parser):
    parser.add_argument("inputs",nargs="*",metavar="RUN_OR_TASK_OR_SUCCESS")
    parser.add_argument("--config",type=Path,help="Closed postfilter YAML overlay")
    parser.add_argument("--presets",type=Path,help="Postfilter preset YAML")
    parser.add_argument("--criteria",action="append",help="Postfilter preset; repeat to require all")
    parser.add_argument("--upstream-criteria",action="append",help="Named passed validation criterion to select")
    parser.add_argument("--method",action="append",choices=("af2_ig","esmfold2-fast","esmfold2-standard","af3","af3-la"))
    parser.add_argument("--task",action="append",help="Select task name")
    parser.add_argument("--limit",type=int,help="Positive maximum concrete refolds per task")
    parser.add_argument("--out",type=Path,help="Independent output directory, for a single selected task")
    parser.add_argument("--show-config",action="store_true",help="Print settings without structure or model loading")
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--plan",action="store_true",help="Read successful refolds and report work/dependency gaps without result writes")
    mode.add_argument("--refilter",action="store_true",help="Rejudge saved metrics without native work")
    mode.add_argument("--recompute-metrics",action="store_true",help="Recompute metrics on saved geometry without another relax")
    mode.add_argument("--rerun",action="store_true",help="Explicitly rerun selected refolds and replace owned outputs")

def run(args):
    from .runner import run as run_postfilter
    return run_postfilter(args)
