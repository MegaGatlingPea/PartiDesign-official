"""Public sequence-only screen arguments and dispatch."""
from pathlib import Path

def add_arguments(parser):
    from partidesign.cli import _model_location, _msa_options, _single
    parser.add_argument("screen_file",type=Path,metavar="SCREEN.yaml",help="Named screen tasks with sequence contexts and id,modality,seq CSV candidates")
    parser.add_argument("--task",action="append",help="Select a screen task (repeatable)")
    _single(parser,"--out",type=Path,default=Path("screen_results"),help="Screen output directory")
    _model_location(parser)
    _single(parser,"--worker-gpus",type=float,help="GPU fraction per resident worker in (0,1], same as pipeline/validate")
    _single(parser,"--worker-cpus",type=float,help="Positive CPU allocation per resident worker")
    parser.add_argument("--plan",action="store_true",help="Show jobs, resource reservations and context policies without prediction or writing")
    parser.add_argument("--resume",action="store_true",help="Reuse matching complete requests and finish incomplete stages")
    _msa_options(parser)

def run(args):
    import json
    from partidesign.paths import code_root, model_dir, using_model_dir
    from .config import load_tasks
    from .runner import make_plan, run as run_screen
    if args.resume and args.msa_refresh:
        raise ValueError("MSA refresh changes pinned evidence; use a new --out without --resume")
    tasks = load_tasks(args.screen_file,selected=args.task or (),args=args)
    plan = make_plan(tasks,args)
    if args.plan:
        print(json.dumps(plan,indent=2))
        return 0
    root = model_dir(args.model_dir)
    with using_model_dir(root):
        return run_screen(tasks,args,code_root(),root)
