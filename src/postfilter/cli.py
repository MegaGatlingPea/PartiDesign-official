"""Small public interface for default-preset Ray postfiltering."""
from pathlib import Path

def add_arguments(parser):
    parser.add_argument("inputs",nargs="+",metavar="RUN_OR_TASK_OR_SUCCESS")
    parser.add_argument("--out",type=Path,help="Independent output directory; multiple tasks use task-name subdirectories")
    parser.add_argument("--config",type=Path,help="Runtime YAML overlay for CPU resources, seed and tool paths")
    parser.add_argument("--task",action="append",help="Select task name; repeat for multiple tasks")
    parser.add_argument("--plan",action="store_true",help="Show selected candidates, default presets and CPU resources without calculations")
    parser.add_argument("--rerun",action="store_true",help="Reprocess selected samples, including completed samples")

def run(args):
    from .runner import run as run_postfilter
    return run_postfilter(args)
