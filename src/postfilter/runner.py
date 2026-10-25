"""Stage successful refolds, isolate native work, and publish independent results."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile

from .config import DEFAULT_CRITERIA,dependency_status,load_config
from .criteria import evaluate,load_presets,validate_criteria

def _geometry_settings(config):
    try:version=importlib.metadata.version("pyrosetta")
    except importlib.metadata.PackageNotFoundError:version=None
    return {"protocol_version":1,"pyrosetta_version":version,**{k:config[k] for k in ("scorefunction","seed","relax","threads")}}

def _metric_settings(config):
    tools={}
    for key in ("dssp","dalphaball"):
        path=config.get(key)
        if path and Path(path).is_file():
            st=Path(path).stat();tools[key]={"path":path,"size":st.st_size,"mtime_ns":st.st_mtime_ns}
        else:tools[key]=None
    return {"definition_version":2,"tools":tools}

@contextmanager
def _lock(output):
    import fcntl
    from .storage import _plain_path
    root=_plain_path(output)
    if root.is_symlink():raise ValueError("postfilter output cannot be a symlink")
    state=_plain_path(root/".state");state.mkdir(parents=True,exist_ok=True)
    path=_plain_path(state/"run.lock")
    if path.is_symlink():raise ValueError("postfilter lock cannot be a symlink")
    with path.open("a+") as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError(f"postfilter output is in use: {root}") from exc
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)

def _physical_candidate(candidate):
    return {k:v for k,v in candidate.to_dict().items() if k not in {"passing_criteria"}}

def _promote_pair(staging,private_staging,destination,private_destination,original_presence=None):
    """Publish owned geometry/state, rolling back ordinary write errors."""
    from .storage import _plain_path
    pairs=((_plain_path(private_staging),_plain_path(private_destination)),(_plain_path(staging),_plain_path(destination)))
    backups=[];published=[]
    for source,target in pairs:
        target.parent.mkdir(parents=True,exist_ok=True)
        backup=staging.parent/(target.name+".previous")
        was_present=target.exists() if original_presence is None else original_presence[str(target)]
        if was_present and not backup.exists():shutil.copy2(target,backup)
        backups.append((target,backup if was_present and backup.exists() else None))
    try:
        for source,target in pairs:
            _plain_path(target)
            os.replace(source,target);published.append(target)
    except Exception:
        for target,backup in reversed(backups):
            if backup is not None:os.replace(backup,target)
            elif target in published:target.unlink(missing_ok=True)
        raise
    for _,backup in backups:
        if backup is not None:backup.unlink(missing_ok=True)

def _still_passed(candidate, config):
    from .inputs import refresh_candidate
    current = refresh_candidate(candidate, path_mappings=config["path_mappings"])
    if _physical_candidate(current) != _physical_candidate(candidate):
        raise ValueError("upstream prediction identity or source changed during postfilter")

def _process(candidate, output, config, names, presets, mode):
    from .records import snapshot_refold
    from .schema import Candidate
    from .storage import _plain_path, atomic_json, read_json
    from .metrics_bc2 import compute_input_metrics

    identity = Path(candidate.identity)
    if identity.is_absolute() or ".." in identity.parts:
        raise ValueError("invalid candidate identity")
    record_path = _plain_path(output / "records" / identity.with_suffix(".json"))
    pending_path = _plain_path(record_path.with_suffix(".pending.json"))
    destination = _plain_path(output / "structures" / identity.with_suffix(".cif"))
    private_destination = _plain_path(output / ".state" / "native" / identity.with_suffix(".json"))
    geometry_config, metric_config = _geometry_settings(config), _metric_settings(config)
    previous, pending = read_json(record_path), read_json(pending_path)
    record = {
        "version": 1, "identity": candidate.identity, "candidate": candidate.to_dict(),
        "status": "failed", "geometry_status": "not_run", "metrics": {}, "errors": {},
        "decisions": {}, "relaxed_path": None, "error": "",
        "geometry_config": geometry_config, "metric_config": metric_config,
    }
    can_update_record = False
    resumed_attempt = False

    def same_request(saved):
        return (_physical_candidate(Candidate.from_dict(saved["candidate"])) == _physical_candidate(candidate)
                and saved.get("geometry_config") == geometry_config)

    def check_pair(saved):
        return (destination.is_file() and private_destination.is_file()
                and snapshot_refold(destination) == saved.get("output_snapshot")
                and read_json(private_destination) == saved.get("private_state_snapshot"))

    try:
        if snapshot_refold(candidate.refold_path) != candidate.source_snapshot:
            raise ValueError("source refold changed after selection")
        baseline = previous
        if pending and mode not in {"rerun", "refilter"} and same_request(pending):
            baseline = pending
            resumed_attempt = True
            # A process may stop between a pair publication and its state update.
            if pending.get("publication_ready"):
                if not check_pair(pending):
                    staging = _plain_path(pending["staging_path"])
                    if not staging.is_relative_to(output / ".state"):
                        raise ValueError("pending geometry is outside owned staging")
                    private_staging = _plain_path(staging.with_suffix(".native.json"))
                    if not staging.is_file() or snapshot_refold(staging) != pending.get("output_snapshot"):
                        raise ValueError("interrupted geometry publication cannot be recovered")
                    # This is actual saved native geometry, not newly inferred atoms.
                    if not private_staging.is_file():
                        atomic_json(private_staging, pending["private_state_snapshot"])
                    if read_json(private_staging) != pending.get("private_state_snapshot"):
                        raise ValueError("interrupted private geometry changed")
                    _promote_pair(staging, private_staging, destination, private_destination,
                                  pending.get("publication_previous"))
                baseline = copy.deepcopy(pending)
                baseline.update(geometry_status="complete", relaxed_path=str(destination))

        if baseline and mode != "rerun":
            old_settings = baseline.get("geometry_config", {})
            compare_settings = geometry_config
            if mode == "refilter":
                old_settings = {k: v for k, v in old_settings.items() if k != "pyrosetta_version"}
                compare_settings = {k: v for k, v in compare_settings.items() if k != "pyrosetta_version"}
            if (_physical_candidate(Candidate.from_dict(baseline["candidate"])) != _physical_candidate(candidate)
                    or old_settings != compare_settings):
                raise ValueError("input, roles or relax protocol changed; use --rerun")
            record = copy.deepcopy(baseline)
            record["candidate"] = candidate.to_dict()
            if record.get("geometry_status") == "complete":
                if not destination.is_file() or snapshot_refold(destination) != record.get("output_snapshot"):
                    raise ValueError("saved relaxed structure is absent or changed; use --rerun")
                if mode != "refilter" and not check_pair(record):
                    raise ValueError("saved private geometry is absent or changed; use --rerun")

        if mode == "refilter":
            if not previous or record.get("geometry_status") != "complete":
                raise ValueError("no completed postfilter geometry to refilter")
            can_update_record = True
        else:
            if mode == "recompute_metrics" and record.get("geometry_status") != "complete":
                raise ValueError("no completed geometry; metric recomputation cannot relax")
            need_geometry = record.get("geometry_status") != "complete" or mode == "rerun"
            need_metrics = (need_geometry or mode == "recompute_metrics"
                            or record.get("metric_config") != metric_config or record.get("status") != "complete")
            if baseline and not need_geometry and baseline.get("metric_config") != metric_config and mode != "recompute_metrics":
                raise ValueError("metric settings changed; use --recompute-metrics")
            can_update_record = record.get("geometry_status") == "complete"
            if need_metrics:
                from .worker import execute_native
                native_config = copy.deepcopy(config)
                native_config["mode"] = "relax" if need_geometry else "metrics_only"
                cached_native = record.get("native_result", {})
                use_cached_native = (not need_geometry and mode == "resume"
                                     and record.get("metric_config") == metric_config
                                     and record.get("native_result_metric_config") == metric_config
                                     and not record.get("metrics_attempt_incomplete")
                                     and cached_native.get("status") == "complete")
                if use_cached_native:
                    native = cached_native
                else:
                    if need_geometry:
                        if resumed_attempt and record.get("staging_path"):
                            staging = _plain_path(record["staging_path"])
                            if not staging.is_relative_to(output / ".state"):
                                raise ValueError("pending job escaped owned staging")
                        else:
                            staging = Path(tempfile.mkdtemp(prefix="candidate-", dir=output / ".state")) / "relaxed.cif"
                        native_config["private_state_path"] = str(staging.with_suffix(".native.json"))
                    else:
                        staging = destination
                        native_config["private_state_path"] = str(private_destination)
                    reuse_job = (resumed_attempt and record.get("metrics_attempt_incomplete")
                                 and record.get("native_mode") == native_config["mode"])
                    job_directory = (Path(record["job_directory"]) if reuse_job and record.get("job_directory")
                                     else Path(tempfile.mkdtemp(prefix="native-job-", dir=output / ".state")))
                    job_directory = _plain_path(job_directory)
                    if not job_directory.is_relative_to(output / ".state"):
                        raise ValueError("native job escaped owned state")
                    native_config["job_directory"] = str(job_directory)
                    record.update(geometry_config=geometry_config, metric_config=metric_config,
                                  status="running", staging_path=str(staging), job_directory=str(job_directory),
                                  metrics_attempt_incomplete=True,native_mode=native_config["mode"])
                    # Preserve the last committed record until new geometry is valid.
                    atomic_json(pending_path, record)
                    native = execute_native(candidate, staging, native_config)
                    complete = native.get("geometry_status") == "complete" or native.get("status") == "complete"
                    if not complete:
                        record.update(metrics_attempt_incomplete=False,native_result=native,
                                      native_result_metric_config=metric_config)
                        atomic_json(pending_path,record)
                    if complete and staging.is_file():
                        private_staging = _plain_path(native.get("private_state_path") or native_config["private_state_path"])
                        if not private_staging.is_relative_to(output / ".state") and private_staging != private_destination:
                            raise ValueError("native state escaped owned output")
                        record.update(output_snapshot=snapshot_refold(staging),
                                      private_state_snapshot=read_json(private_staging),
                                      private_state_path=str(private_destination), native_result=native,
                                      native_result_metric_config=metric_config,metrics_attempt_incomplete=False,
                                      publication_ready=True,
                                      publication_previous={str(destination):destination.exists(),
                                                            str(private_destination):private_destination.exists()})
                        atomic_json(pending_path, record)
                        if need_geometry:
                            _promote_pair(staging, private_staging, destination, private_destination,
                                          record["publication_previous"])
                        record.update(geometry_status="complete", relaxed_path=str(destination))
                        can_update_record = True
                        atomic_json(record_path, record)
                    if record.get("geometry_status") != "complete":
                        raise RuntimeError(native.get("error") or "native geometry failed")

                record["metrics"] = dict(native.get("metrics", {}))
                record["errors"] = dict(native.get("errors", {}))
                record["native_details"] = native.get("details", {})
                input_metrics = compute_input_metrics(candidate, config)
                record["metrics"].update(input_metrics["metrics"])
                record["errors"].update(input_metrics["errors"])
                record["input_details"] = input_metrics.get("details", {})
                record["native_result"] = native
                record["status"] = "complete" if native.get("status") == "complete" else "failed"
                record["error"] = native.get("error") or ""

        _still_passed(candidate, config)
        if snapshot_refold(candidate.refold_path) != candidate.source_snapshot:
            raise ValueError("source refold changed before publication")
        record["decisions"] = {name: evaluate(record.get("metrics", {}), name, presets[name]) for name in names}
        atomic_json(record_path, record)
        if mode != "refilter":
            pending_path.unlink(missing_ok=True)
        return record
    except Exception as exc:
        record["status"], record["error"] = "failed", str(exc)
        record["decisions"] = {name: {"criteria": name, "status": "unavailable", "reasons": [str(exc)]} for name in names}
        if can_update_record:
            atomic_json(record_path, record)
        if record.get("staging_path") and mode != "refilter":
            atomic_json(pending_path, record)
        return record


def run(args):
    from partidesign.paths import inference_config_dir
    config=load_config(args.config)
    presets=load_presets(args.presets or inference_config_dir()/"postfilter_presets.yaml")
    names=validate_criteria(args.criteria or DEFAULT_CRITERIA,presets)
    if args.limit is not None and args.limit<1:raise ValueError("--limit must be positive")
    if args.show_config:
        print(json.dumps({"config":config,"criteria":{n:presets[n] for n in names}},indent=2));return 0
    if not args.inputs:raise ValueError("postfilter requires RUN_OR_TASK_OR_SUCCESS")
    from .inputs import select_refolds
    selected=select_refolds(args.inputs,methods=args.method,upstream_criteria=args.upstream_criteria,
                            task_names=args.task,limit=args.limit,path_mappings=config["path_mappings"])
    nonerrors={"not_ppi","no_quality_success","not_complex"}
    selection_errors=[d for d in selected.diagnostics if d.get("code") not in nonerrors]
    if not selected.task_dirs:
        reasons="; ".join(d.get("reason",d.get("code","selection error")) for d in selected.diagnostics)
        raise ValueError(reasons or "No validation method matched the selected inputs/tasks/methods")
    deps=dependency_status(config)
    if args.plan:
        print(json.dumps({"selected":len(selected.candidates),"tasks":selected.task_dirs,"dependencies":deps,"selection_counts":selected.counts,
                          "candidates":[{k:v for k,v in c.to_dict().items() if k!="source_snapshot"} for c in selected.candidates],
                          "diagnostics":selected.diagnostics},indent=2));return 2 if selection_errors else 0
    mode="refilter" if args.refilter else "recompute_metrics" if args.recompute_metrics else "rerun" if args.rerun else "resume"
    if selected.candidates and mode!="refilter" and any(v!="available" for v in deps.values()):
        raise ValueError(f"postfilter dependency gap: {deps}; see docs/postfilter.md")
    if args.out and len(selected.task_dirs)!=1:raise ValueError("--out requires exactly one selected task")
    from .storage import atomic_json,read_json
    from .reporting import write_reports
    failures=bool(selection_errors)
    for task in selected.task_dirs:
        candidates=[c for c in selected.candidates if c.task_dir==task]
        output=Path(args.out).expanduser().absolute() if args.out else Path(task)/"postfilter"
        if output.resolve()==Path(task).resolve() or output.resolve().is_relative_to(Path(task)/"validate"):
            raise ValueError("postfilter output must be independent of upstream validation")
        from .storage import _plain_path
        output=_plain_path(output)
        prior=read_json(output/".state"/"plan.json")
        if prior is None and output.exists() and any(output.iterdir()):
            raise ValueError("postfilter output must be new/empty or contain its saved task plan")
        if prior is not None and (prior.get("version")!=1 or prior.get("task_dir")!=task):
            raise ValueError("postfilter output belongs to a different task or unsupported state")
        diagnostics=[d for d in selected.diagnostics if d.get("task_dir") in (None,task)]
        with _lock(output):
            state=output/".state"/"plan.json"
            previous_plan=read_json(state) if state.is_file() else None
            atomic_json(output/".state"/"report_status.json",{"status":"running"})
            if previous_plan is None:
                atomic_json(state,{"version":1,"task_dir":task,"status":"running","config":config,
                                   "criteria":names,"identities":[c.identity for c in candidates]})
            # Per-candidate settings are checked before replacing this saved plan.
            with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
                records=list(pool.map(lambda candidate:_process(candidate,output,config,names,presets,mode),candidates))
            errors=any(r["status"]!="complete" or any(d["status"]=="unavailable" for d in r["decisions"].values()) for r in records)
            counts={**selected.counts.get(task,{}),"selected":len(candidates),
                    "metadata_error_events":sum(d.get("code") not in nonerrors for d in diagnostics),
                    "limited":args.limit is not None,"limit":args.limit,
                    "previous_not_selected":len(set(previous_plan.get("identities",[]))-set(c.identity for c in candidates)) if previous_plan else 0}
            summary=write_reports(output,records,diagnostics,counts)
            if not errors:
                atomic_json(state,{"version":1,"task_dir":task,"status":"complete","config":config,
                                   "criteria":names,"identities":[c.identity for c in candidates]})
            atomic_json(output/"resolved_config.json",{"config":config,"criteria":{n:presets[n] for n in names}})
            print(json.dumps({"task":task,"output":str(output),"selected":summary["selected"],"passed":summary["passed"],"execution_failures":summary["execution_failures"]}))
            failures|=errors
    return 2 if failures else 0
