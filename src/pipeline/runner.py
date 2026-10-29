"""Phase scheduling for no-training sequence/structure optimization."""
from __future__ import annotations
from argparse import Namespace
import copy
from dataclasses import is_dataclass, replace
import json
import errno
import os
from pathlib import Path
import shutil
import sys
import time

from .config import build_config, criteria_for
from .layout import (TASK_LAYOUT, design_folder, metadata_folder, refold_folder,
                     run_work, stage_folder, stage_state, task_layout)
from .selection import SelectionCandidate, select_candidates, sequence_key
from .state import (Candidate, atomic_json, assert_invariants, chain_sequences,
                    has_special_polymer_bond, jsonable, load_candidate, refreshed_row, residues, write_csv)


def generation_runner(cfg, repo):
    """Compose with the design frontend, then enforce workflow-owned stages."""
    from omegaconf import OmegaConf
    from partidesign.cli import build_design_runner
    from partidesign.paths import inference_config_dir
    from src.design.core.api import DesignRunner
    opts = dict(cfg["generation"])
    opts.pop("workflow_controls", None)
    if Path(opts["config"]) == inference_config_dir() / "design.yaml":
        opts["config"] = None
    for key in ("ckpt", "config"):
        if opts.get(key) is not None:
            opts[key] = Path(opts[key])
    if opts.get("compile_strategy")=="auto" and opts.get("compile") in {"true","false"}:
        opts["compile_strategy"]=None
    for flag in ("route",):
        if isinstance(opts.get(flag), bool):
            opts[flag] = str(opts[flag]).lower()
    args = Namespace(**opts, task_file=Path(cfg["task_file"]) if cfg.get("task_file") else None,
                     task=cfg["task_names"],
                     out=Path(cfg["output_dir"]) / "generation",
                     run_name="generation", seed=cfg["seed"])
    original = build_design_runner(args, Path(cfg["model_dir"]), invocation_dir=Path.cwd(),
                                   task_payload=cfg["task_payload"])
    native = OmegaConf.to_container(original.config, resolve=True)
    native["tasks_file"] = None
    native["tasks"] = copy.deepcopy(native["tasks"])
    for task in native["tasks"].values():
        task["score"] = False
        task["postprocess"] = False
    native["search"]["algorithm"] = None
    native["reward"] = {"_target_": None}
    native["output_dir"] = str(Path(cfg["output_dir"]) / "generation")
    native["run_name"] = "generation"
    return DesignRunner(native, invocation_dir=Path.cwd())


def context_route(record, task):
    if record.row.get("heavy_role"):
        return "antibody"
    modalities = {chain.modality for chain in chain_sequences(record)}
    if len(modalities) > 1 and not modalities <= {"rna", "dna"}:
        return "joint"
    if record.tip_atoms and modalities=={"protein"} and "ligand" in set(record.atoms.mol_type):
        return "ame"
    kinds = {str(x) for i,x in enumerate(record.atoms.mol_type)
             if str(record.atoms.chain_id[i]) not in record.design_chains}
    kinds -= {"ion", "water"}
    if len(kinds) > 1:
        # A mixed receptor needs authored criteria/route, never a name guess.
        return "auto"
    if not kinds:
        return "monomer"
    kind = next(iter(kinds))
    if kind == "protein":
        if modalities == {"protein"} and max(len(residues(record.atoms,c)) for c in record.design_chains) <= 30:
            return "peptide"
        return "protein"
    if kind in {"rna","dna"}:
        return "nuc"
    if kind == "ligand":
        return "ligand"
    return "auto"


def decisions(record, names, definitions, mode="all"):
    from src.validate.filters import evaluate
    evaluated = {name:evaluate(record,name,definitions[name]) for name in names}
    passing = [set(row.get("passing_prediction_keys", ())) for row in evaluated.values()
               if row["status"] == "passed"]
    if mode == "all":
        keys = set.intersection(*passing) if len(passing)==len(names) and passing else set()
    else:
        keys = set.union(*passing) if passing else set()
    # A later round requires a whole complex, never the unbound AF2 monomer.
    keys = {key for key in keys if record["predictions"].get(key,{}).get("protocol") == "complex"}
    return keys, evaluated


def final_prediction_view(record, selection="best_ranked", *, models=None):
    """Choose one final complex per AF2 model with its matched controls."""
    if selection == "all":
        return record, {"mode": "all"}
    from math import isfinite
    ranked = []
    for key in record["expected_predictions"]:
        if record["method"] == "af2_ig" and not key.startswith("complex_"):
            continue
        try:
            prediction = record["predictions"][key]
            if prediction["protocol"] != "complex":
                continue
            if record["method"] == "af2_ig":
                metrics = prediction["metrics"]
                score = 0.8 * float(metrics["iptm"]) + 0.2 * float(metrics["ptm"])
            else:
                score = float(prediction["confidence"]["ranking_score"])
            if not isfinite(score):
                raise ValueError("non-finite ranking score")
        except (KeyError, TypeError, ValueError) as exc:
            if record["method"] == "af2_ig":
                return dict(record, status="partial"), {
                    "mode": "best_ranked", "status": "unavailable",
                    "reason": f"AF2 prediction {key!r} lacks complete ranking metrics"}
            raise ValueError(f"Final {record['method']} prediction {key!r} lacks ranking confidence") from exc
        ranked.append((-score, key))
    if record["method"] == "af2_ig":
        by_model = {}
        for item in ranked:
            model = item[1].removeprefix("complex_").split("_sample", 1)[0]
            by_model.setdefault(model, []).append(item)
        model_names = [f"model{model}" for model in models] if models is not None else list(by_model)
        if not model_names or any(not by_model.get(model) for model in model_names):
            return dict(record, status="partial"), {
                "mode": "best_ranked", "status": "unavailable",
                "reason": "not every configured AF2 model has a ranked complex"}
        selected = {model: min(by_model[model]) for model in model_names}
        negative_score, key = min(selected.values())
        selected_models = {}
        expected, pairs = [], []
        for model, (negative, complex_key) in selected.items():
            suffix = complex_key.removeprefix("complex_")
            monomer_key = "monomer_" + suffix
            expected.extend((complex_key, monomer_key))
            pairs.append(suffix)
            selected_models[model] = {
                "selected_prediction_key": complex_key, "paired_prediction_key": monomer_key,
                "ranking_score": -negative}
        view = dict(record, expected_predictions=expected,
                    predictions={key: record["predictions"][key] for key in expected
                                 if key in record["predictions"]},
                    paired_metrics={suffix: record["paired_metrics"][suffix] for suffix in pairs
                                    if suffix in record["paired_metrics"]},
                    paired_reasons={suffix: record.get("paired_reasons", {}).get(suffix, {})
                                    for suffix in pairs})
        return view, {
            "mode": "best_ranked", "selected_models": selected_models,
            "selected_prediction_key": key, "paired_prediction_key": "monomer_" + key.removeprefix("complex_"),
            "ranking_score": -negative_score}
    negative_score, key = min(ranked)
    view = dict(record, expected_predictions=[key], predictions={key: record["predictions"][key]})
    return view, {"mode": "best_ranked", "selected_prediction_key": key,
                  "ranking_score": -negative_score}


def _copy_snapshot(source, destination, task):
    import yaml
    source = metadata_folder(source, task)
    path = Path(source) / "resolved_config.yaml"
    if path.is_file():
        snapshot = yaml.safe_load(path.read_text())
    else:
        snapshot = {"tasks":{task:{}}}
    snapshot=_absolute_references(snapshot,_source_bases(source,snapshot))
    snapshot["tasks"]={task:snapshot.get("tasks",{}).get(task,{})}
    snapshot["task_names"]=[task]
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"resolved_config.yaml").write_text(yaml.safe_dump(snapshot,sort_keys=False))
    return snapshot


def import_candidates(source, target, names, *, layout=None):
    from concurrent.futures import ThreadPoolExecutor
    from src.validate.inputs import discover_inputs
    result={name:[] for name in names}
    snapshots={}
    bases={}
    new_layout = layout == TASK_LAYOUT
    folders = {name: Path(target) / name / "design" if new_layout else Path(target) / name
               for name in names}
    public_sources = {name: Path(source)/name/"design" for name in names}
    if any((folder/"result.csv").is_file() for folder in public_sources.values()):
        entries=((name,entry) for name in names for entry in discover_inputs(
            public_sources[name] if (public_sources[name]/"result.csv").is_file() else Path(source)/name))
    else:
        entries=((entry.task_dir.name,entry) for entry in discover_inputs(source,names))
    with ThreadPoolExecutor(max_workers=min(4,os.cpu_count() or 1)) as pool:
        copies=[]
        for name,entry in entries:
            if entry.error:
                raise ValueError(entry.error)
            if entry.row.get("error"):
                continue
            folder=folders[name]
            if name not in snapshots:
                metadata=folder/".work"/name if new_layout else folder
                snapshots[name]=_copy_snapshot(entry.task_dir,metadata,name)
                bases[name]=_source_bases(entry.task_dir,snapshots[name])
            path=(folder if new_layout else folder/"inputs")/(entry.sample_id+entry.path.suffix)
            copies.append(pool.submit(_publish,entry.path,path))
            row=_absolute_references(dict(entry.row),bases[name])
            row["filename"]=str(path)
            result[name].append(Candidate(entry.sample_id,name,"",entry.sample_id,str(path),row,
                                          origin_path=str((entry.row.get("origin_path") or entry.path)
                                                          if new_layout else entry.path)))
        for future in copies:
            future.result()
    for name in names:
        if not result[name]:
            raise ValueError(f"No valid generated structures for task {name!r} in {source}")
        write_csv(folders[name]/"result.csv",[
            dict(c.row,candidate_id=c.candidate_id,**({"origin_path":c.origin_path} if new_layout else {}))
            for c in result[name]])
    return result


def _stage_inputs(parents, folder, previous, *, task=None, layout=None):
    task = task or (parents[0].task if parents else folder.name)
    metadata=folder/".work"/task if layout==TASK_LAYOUT else folder
    _copy_snapshot(previous,metadata,task)
    result=[]
    for parent in parents:
        source=Path(parent.path)
        inputs=folder/".work"/"inputs" if layout==TASK_LAYOUT else folder/"inputs"
        target=inputs/(parent.candidate_id+source.suffix)
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.resolve()!=target.resolve():
            _publish(source,target)
        result.append(replace(parent,path=str(target),row=dict(parent.row,filename=str(target))))
    return result


def _load_stage(path, *, folder=None, task=None):
    if not path.is_file():
        return None
    value=json.loads(path.read_text())
    if value.get("status")!="complete":
        return None
    folder=Path(folder) if folder is not None else path.parent
    metadata=metadata_folder(folder,task) if task is not None else folder
    if (any(not (folder/name).is_file() for name in ("result.csv","selection.csv")) or
            not (metadata/"resolved_config.yaml").is_file()):
        return None
    result=[Candidate(**c) for c in value["candidates"]]
    archive=[Candidate(**c) for c in value.get("archive",[])]
    if any(not Path(c.path).is_file() for c in [*result,*archive]):
        return None
    return result,dict(value["summary"],_archive=archive)


def _candidate_record(candidate, folder, **kwargs):
    """Parse within the stage, then attach the correctly named task metadata."""
    record=load_candidate(candidate,folder,**kwargs)
    metadata=metadata_folder(folder,candidate.task)
    if metadata!=Path(folder):
        if is_dataclass(record):
            record=replace(record,task_dir=metadata)
        else:
            record=copy.copy(record)
            vars(record).update(task_dir=metadata)
    return record


def _relax(candidate, input_record, folder, config):
    from src.validate.sequences import _candidate_adapter
    from src.design.postproc.runner import build_parameterization_payload, PostprocessPreparationError
    from src.design.postproc.openmm.refine import refine_one, OpenMMPostprocessError
    import yaml
    snapshot=yaml.safe_load((metadata_folder(folder,candidate.task)/"resolved_config.yaml").read_text())
    authored=snapshot.get("tasks",{}).get(candidate.task,{})
    native=_candidate_adapter(input_record,candidate.candidate_id)
    from src.design.core.tasks import parse_task
    native.base_request.task=parse_task(candidate.task,authored,invocation_dir=Path(__file__).resolve().parents[2])
    target=folder/"relaxed"/(candidate.candidate_id+".cif")
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        _restore_context_molecules(native,input_record)
        payload=build_parameterization_payload(native,task_name=candidate.task,sample_id=candidate.candidate_id)
        energies=refine_one(candidate.path,target,config=config,payload=payload,move_context=True)
        row=refreshed_row(input_record,target)
        row.update(postprocess_status="succeeded",structure_stage="relaxed",refold_path=candidate.path)
        updated=replace(candidate,path=str(target),row=row)
        child=_candidate_record(updated,folder)
        assert_invariants(input_record,child)
    except (OpenMMPostprocessError,PostprocessPreparationError) as exc:
        return None,{"status":"failed","error":f"{type(exc).__name__}: {exc}"}
    return updated,{"status":"complete","initial_energy":energies[0],
                    "final_energy":energies[1],"rms_force":energies[2],"path":str(target)}


def _selection_family(parent):
    """One previous refold and its partial children compete as a family."""
    return parent.parent_id if parent.source == "partial" else parent.candidate_id


def run_stage(owner, cfg, task_name, stage, parents, previous, *, final=False, force=False):
    from .backends import design_score
    task_cfg=cfg["tasks"][task_name]
    number=stage.get("round",task_cfg["recycle"])
    folder=stage_folder(cfg,task_name,number,final=final)
    state=stage_state(cfg,folder)
    method=stage["method"]
    requested_method=method
    first_record=None
    refold_override=None
    policy_checked=bool(not final and method!="esmfold2-fast" and parents)
    if policy_checked:
        first_record=_candidate_record(parents[0],previous)
        if has_special_polymer_bond(first_record):
            method="esmfold2-fast"
            validation=copy.deepcopy(stage["validation"])
            requested_group={"af2_ig":"af2","af3":"af3"}[requested_method]
            validation["methods"]=[method]
            validation["esmfold2"]["samples"]=stage["samples"]
            validation["esmfold2"]["seed"]=validation[requested_group]["seed"]
            stage=dict(stage,method=method,validation=validation)
            refold_override="special_polymer_polymer_bond"
            print(f"[{task_name}] round {number}: {requested_method} -> {method} "
                  "(special polymer-polymer bond)",flush=True)
    saved=None if force else _load_stage(state,folder=folder,task=task_name)
    if saved is not None and (not policy_checked or saved[1].get("method")==method):
        print(f"[{task_name}] {'final' if final else f'round {number}'} resumed",flush=True)
        return saved[0],dict(saved[1],_resumed=True)
    # Only this invalidated phase's owned products are replaced. Upstream
    # generation, parent structures and the shared raw prediction cache stay intact.
    for name in ("relaxed","partial","success"):
        owned=folder/name
        if owned.is_dir() and not owned.is_symlink():
            shutil.rmtree(owned)
    if task_layout(cfg):
        for owned in (folder/"refold",folder/"predictions",folder/".work"/"predictions",
                      folder/".work"/"partial",folder/".work"/"refolds"):
            if owned.is_dir() and not owned.is_symlink():
                shutil.rmtree(owned)
    folder.mkdir(parents=True,exist_ok=True)
    atomic_json(state,{"status":"running"})
    parents=_stage_inputs(parents,folder,previous,task=task_name,layout=cfg.get("_layout"))
    if first_record is not None:
        metadata={"path":Path(parents[0].path),"row":parents[0].row,
                  "task_dir":metadata_folder(folder,task_name)}
        if is_dataclass(first_record):
            first_record=replace(first_record,**metadata)
        else:
            first_record=copy.copy(first_record)
            vars(first_record).update(metadata)
    if hasattr(owner, "activate"):
        owner.activate(method,stage["validation"],task_cfg["openmm"])
    proposals=[]
    unique_sequences=set()
    for parent_index,parent in enumerate(parents):
        record=first_record if parent_index==0 and first_record is not None else _candidate_record(parent,folder)
        route=context_route(record,cfg["task_payload"][task_name])
        names,presets=criteria_for(task_cfg,"final" if final else stage,route,method)
        if method=="af2_ig":
            from src.validate.af2 import validate_capability
            validate_capability(record)
        effective=dict(stage,mpnn_count=0 if final else stage["mpnn_count"],
                       include_gt=True if final else stage.get("include_gt",True),
                       _task_index=list(cfg["task_payload"]).index(task_name),_parent_index=parent_index)
        variants=owner.generate(record,effective,task_name=task_name,round_number=number,
                                parent_id=parent.candidate_id,seed=cfg["seed"],
                                work_dir=folder/(".work/mpnn" if task_layout(cfg) else ".mpnn")/parent.candidate_id)
        for variant in variants:
            if variant.get("status") not in {"complete","ok"}:
                raise RuntimeError(f"Sequence generation failed for {parent.candidate_id}: {variant.get('error')}")
            chains=chain_sequences(record,variant)
            unique_sequences.add(sequence_key(chains))
            proposals.append((parent,record,variant,chains,names,presets))
    count={"task":task_name,"stage":"final" if final else (f"round_{number}" if task_layout(cfg) else f"recycle_{number:03d}"),
           "method":method,"parent_count":len(parents),"proposal_count":len(proposals),
           "unique_sequence_count":len(unique_sequences),"unique_request_count":0,
           "cache_hits":0,"actual_prediction_count":0,"passing_count":0,
           "duplicate_count":0,"selected_count":0,"family_limited_count":0,
           "relax_success_count":0,"relax_failed_count":0,
           "partial_success_count":0,"partial_failed_count":0,"partial_model_loads":0}
    if refold_override is not None:
        count.update(requested_method=requested_method,refold_override=refold_override)
    actual_requests=set()
    eligible=[]
    candidate_by_id={}
    input_by_id={}
    group={"af2_ig":"af2","af3":"af3","esmfold2-fast":"esmfold2"}[method]
    score_profile=json.dumps([method,stage["validation"].get(group,{})],sort_keys=True)
    result_rows=[]
    motif_policy=cfg.get("motif_policy")=="first_and_final"
    check_motif=not motif_policy or final or number==1
    metric_settings=(stage["validation"] if check_motif else
                     dict(stage["validation"], _pipeline_motif_geometry=False))
    jobs=[(p[1],p[2],method,stage["validation"],task_name) for p in proposals]
    outcomes=(owner.predict_many(jobs) if hasattr(owner,"predict_many") else
              (owner.predict(r,v,m,c,task_name=t) for r,v,m,c,t in jobs))
    for ordinal,((parent,input_record,variant,chains,names,presets),(request_key,raw,hit)) in enumerate(_paired_outcomes(proposals,outcomes),1):
        actual_requests.add(request_key)
        count["cache_hits"]+=int(hit)
        if not hit:
            count["actual_prediction_count"]+=len(raw)
        record=owner.evaluate(input_record,variant,method,metric_settings,raw)
        parsed_by_key=record.pop("_parsed_atoms",{})
        evaluation_record=record
        if final:
            evaluation_record,selection_info=final_prediction_view(
                record,stage.get("prediction_selection","best_ranked"),
                models=stage["validation"]["af2"]["models"] if method=="af2_ig" else None)
            record["prediction_selection"]=selection_info
            if "selected_prediction_key" in selection_info:
                best=record["predictions"][selection_info["selected_prediction_key"]]
                record["metrics"]=copy.deepcopy(best.get("metrics",{}))
                record["design_score"]=best.get("pipeline_design_score")
        if not check_motif:
            from .criteria import without_motif_geometry
            names,presets=without_motif_geometry(names,presets)
        if not names and not check_motif:
            passing={key for key,prediction in evaluation_record["predictions"].items()
                     if prediction["protocol"]=="complex" and evaluation_record["status"]=="complete"}
            evaluated={}
        else:
            passing,evaluated=decisions(evaluation_record,names,presets,stage.get("criteria_mode","all"))
        if motif_policy and check_motif and bool(getattr(input_record,"motif_positions",{})):
            motif_keys,motif_decision=decisions(evaluation_record,["pipeline_motif"],presets)
            passing &= motif_keys
            evaluated.update(motif_decision)
        if final and method=="af2_ig" and selection_info.get("selected_models"):
            selected_models=selection_info["selected_models"]
            all_models_passed=all(value["selected_prediction_key"] in passing for value in selected_models.values())
            selection_info["status"]="passed" if all_models_passed else "filtered"
            passing={selection_info["selected_prediction_key"]} if all_models_passed else set()
        elif final and method=="af2_ig" and selection_info["mode"]=="all":
            passing_models={key.removeprefix("complex_").split("_sample",1)[0] for key in passing}
            all_models_passed=all(f"model{model}" in passing_models for model in stage["validation"]["af2"]["models"])
            selection_info["status"]="passed" if all_models_passed else "filtered"
            if not all_models_passed:
                passing=set()
        record.update(decisions=evaluated,request_key=request_key,cache_hit=hit,
                      parent_id=parent.candidate_id,origin_id=parent.origin_id)
        public=((folder/"predictions" if final else folder/".work"/"predictions")
                if task_layout(cfg) else folder/"validate"/method)
        public.mkdir(parents=True,exist_ok=True)
        for prediction in record["predictions"].values():
            source=Path(prediction["structure_path"]) if prediction.get("structure_path") else None
            published={}
            if source and source.is_file():
                path=public/f"proposal{ordinal:06d}__{prediction['key']}{source.suffix}"
                _publish(source,path)
                prediction["structure_path"]=str(path)
                published[str(source.resolve())]=str(path)
            if task_layout(cfg):
                for field in ("cif_path","native_cif_path","pdb_path","confidence_path","raw_confidence_path"):
                    value=prediction.get(field)
                    artifact=Path(value) if value else None
                    if artifact is None or not artifact.is_file():
                        continue
                    identity=str(artifact.resolve())
                    if identity not in published:
                        target=public/f"proposal{ordinal:06d}__{prediction['key']}__{field.removesuffix('_path')}{artifact.suffix}"
                        _publish(artifact,target)
                        published[identity]=str(target)
                    prediction[field]=published[identity]
        atomic_json(public/f"proposal{ordinal:06d}.json",record)
        if not passing:
            result_rows.append({"candidate_id":f"proposal{ordinal:06d}","parent_id":parent.candidate_id,
                                "origin_id":parent.origin_id,"source":variant["source"],"status":"filtered",
                                "request_key":request_key,"cache_hit":hit,"decisions":evaluated,
                                "sequences":variant["sequences"]})
        raw_by_key={p["key"]:p for p in raw}
        for key in sorted(passing):
            prediction=record["predictions"][key]
            candidate_id=f"{'final' if final else f'r{number:03d}'}_c{len(result_rows)+1:07d}"
            source=Path(prediction["structure_path"])
            refolds=refold_folder(cfg,folder,final=final)
            if task_layout(cfg) and not final and source.suffix.lower()!=".cif":
                refolds=folder/".work"/"refolds"
            target=refolds/(candidate_id+source.suffix)
            target.parent.mkdir(parents=True,exist_ok=True)
            _publish(source,target)
            try:
                from .state import restore_context_aliases
                atoms=restore_context_aliases(input_record,target,atoms=parsed_by_key.get(key))
                row=refreshed_row(input_record,target,atoms=atoms)
                row.update(postprocess_status="not_run",
                           structure_stage=f"{method}_validation" if final else "refold")
                row.pop("refold_path",None)
                score=design_score(input_record,method,raw_by_key[key])
                candidate=Candidate(candidate_id,task_name,parent.candidate_id,parent.origin_id,
                                    str(target),row,variant["source"],request_key+":"+key,score,
                                    origin_path=parent.origin_path,
                                    lineage_id=parent.lineage_id or candidate_id)
                child=_candidate_record(candidate,folder,atoms=atoms)
                assert_invariants(input_record,child)
                actual_chains=chain_sequences(child)
                candidate=replace(candidate,score_profile=score_profile,
                                  sequence_identity=json.dumps(sequence_key(actual_chains)))
            except ValueError as exc:
                if task_layout(cfg) and not final:
                    target.unlink(missing_ok=True)
                result_rows.append({"candidate_id":candidate_id,"parent_id":parent.candidate_id,
                                    "origin_id":parent.origin_id,"source":variant["source"],
                                    "prediction_key":request_key+":"+key,
                                    "status":"condition_failed","error":str(exc),"request_key":request_key})
                print(f"[{task_name}] {candidate_id}: inherited-condition check failed: {exc}",flush=True)
                continue
            if task_layout(cfg) and not final and Path(candidate.path).suffix.lower()!=".cif":
                from src.design.utils.writer import write_structure_atomic
                target=refold_folder(cfg,folder)/(candidate.candidate_id+".cif")
                write_structure_atomic(child.atoms,target,entry_id=candidate.candidate_id)
                row=dict(row,filename=str(target))
                candidate=replace(candidate,path=str(target),row=row)
                child=replace(child,path=target,row=row)
            candidate_by_id[candidate_id]=candidate
            input_by_id[candidate_id]=child
            eligible.append(SelectionCandidate(candidate_id,parent.candidate_id,score,actual_chains,
                                               family_id=_selection_family(parent)))
            result_rows.append({**row,"candidate_id":candidate_id,"parent_id":parent.candidate_id,
                                "origin_id":parent.origin_id,"origin_path":parent.origin_path,
                                "lineage_id":candidate.lineage_id,"source":variant["source"],
                                "prediction_key":candidate.prediction_key,"design_score":score,
                                "status":"passed","cache_hit":hit})
        print(f"[{task_name}] {'final' if final else f'round {number}'} proposal {ordinal}/{len(proposals)}"
              f" ({method}, {'cache' if hit else 'predicted'}, passing={len(passing)})",flush=True)
    count["unique_request_count"]=len(actual_requests)
    count["passing_count"]=len(eligible)
    selection=({"mode":"top","max_pass":None,"max_pass_ratio":None,"per_parent":None}
               if final else stage["selection"])
    selection={k:v for k,v in selection.items() if k!="rank_by"}
    selected=select_candidates(eligible,unique_count=len(unique_sequences),**selection)
    count.update(selected_count=len(selected.selected),duplicate_count=selected.counts["duplicate_count"],
                 family_limited_count=selected.counts["family_limited_count"])
    rows=[{"candidate_id":c.candidate_id,"parent_id":c.parent_id,
           "family_id":c.family_id or c.parent_id,"design_score":c.score,
           **dict(selected.decisions[c.candidate_id])} for c in eligible]
    chosen=[]
    for item in selected.selected:
        candidate=candidate_by_id[item.candidate_id]
        current=input_by_id[item.candidate_id]
        if Path(candidate.path).suffix.lower()!=".cif":
            from src.design.utils.writer import write_structure_atomic
            target=refold_folder(cfg,folder,final=final)/(candidate.candidate_id+".cif")
            write_structure_atomic(current.atoms,target,entry_id=candidate.candidate_id)
            candidate=replace(candidate,path=str(target),row=dict(candidate.row,filename=str(target)))
            current=replace(current,path=target,row=candidate.row)
        chosen.append((candidate,current))
    archive=[candidate for candidate,_ in chosen] if not final else []
    survivors=[]
    if final:
        for candidate,current in chosen:
            destination=folder/"success"/Path(candidate.path).name
            destination.parent.mkdir(parents=True,exist_ok=True)
            _publish(candidate.path,destination)
            survivors.append(replace(candidate,path=str(destination),row=dict(candidate.row,filename=str(destination))))
            if task_layout(cfg):
                for row in result_rows:
                    if row["candidate_id"]==candidate.candidate_id:
                        row["filename"]=str(destination)
    else:
        if number<task_cfg["recycle"] and stage.get("relax_enabled",True):
            jobs=[(c,current,folder,task_cfg["openmm"]) for c,current in chosen]
            outcomes=(owner.relax_many(jobs) if hasattr(owner,"relax_many") else
                      ((owner.relax(*job) if hasattr(owner,"relax") else _relax(*job)) for job in jobs))
            for (candidate,_),(updated,relaxation) in _paired_outcomes(chosen,outcomes):
                for row in rows:
                    if row["candidate_id"]==candidate.candidate_id:
                        row["relaxation"]=relaxation
                if updated is None:
                    count["relax_failed_count"]+=1
                else:
                    survivors.append(updated)
                    count["relax_success_count"]+=1
        else:
            survivors=[candidate for candidate,_ in chosen]
        partial=stage.get("partial",{"enabled":False})
        if number<task_cfg["recycle"] and partial["enabled"] and survivors:
            reference_csv=(folder/".work"/"partial_inputs.csv" if task_layout(cfg)
                           else folder/"partial"/"inputs.csv")
            write_csv(reference_csv,[dict(c.row,filename=c.path,candidate_id=c.candidate_id) for c in survivors])
            references=[{"candidate_id":c.candidate_id,"path":c.path,"result_csv":str(reference_csv)} for c in survivors]
            result=owner.partial_many(references,authored=cfg["task_payload"][task_name],
                effective_generation=cfg["effective_generation"],settings=partial,
                output=folder/"partial",seed=(cfg["seed"]+number*100003+cfg["task_names"].index(task_name)*10000019)%(2**32-1))
            parents_by_id={c.candidate_id:c for c in survivors}
            children=[]
            for index,item in enumerate(result["candidates"],1):
                parent=parents_by_id[item["parent_id"]]
                child=Candidate(f"r{number:03d}_p{index:07d}",task_name,parent.candidate_id,parent.origin_id,
                    item["path"],dict(item["row"],filename=item["path"],structure_stage="partial"),"partial",origin_path=parent.origin_path,
                    lineage_id=parent.lineage_id)
                children.append(child)
            count["partial_success_count"]=len(children)
            count["partial_failed_count"]=result.get("failed",0)
            count["partial_model_loads"]=result.get("model_loads",0)
            survivors=([*survivors,*children] if partial["include_parent"] else children)
    write_csv(folder/"result.csv",result_rows)
    write_csv(folder/"selection.csv",rows)
    atomic_json(state,{"status":"complete","candidates":[c.saved() for c in survivors],
                                     "archive":[c.saved() for c in archive],"summary":count})
    print(f"[{task_name}] {count['stage']}: selected {count['selected_count']}, "
          f"optimized {count['relax_success_count']}, continuing {len(survivors)}",flush=True)
    return survivors,dict(count,_resumed=False,_archive=archive)


def final_pool(history, last, settings):
    """Keep lineage bests within score protocols, then allocate protocol quotas."""
    from collections import deque
    candidates=list(last)
    if settings["source"]=="history_best":
        best={}
        for candidate in history:
            key=(candidate.lineage_id or candidate.origin_id,candidate.score_profile)
            previous=best.get(key)
            if previous is None or (-candidate.score,candidate.candidate_id)<(-previous.score,previous.candidate_id):
                best[key]=candidate
        candidates=list(best.values())
    profiles={}
    for candidate in candidates:
        profiles.setdefault(candidate.score_profile,[]).append(candidate)
    queues=[deque(sorted(group,key=lambda c:(-c.score,c.candidate_id)))
            for _,group in sorted(profiles.items())]
    selected=[]
    seen=set()
    while queues and len(selected)<settings["max_pass"]:
        remaining=[]
        for queue in queues:
            while queue:
                candidate=queue.popleft()
                identity=candidate.sequence_identity
                if not identity:
                    identity=json.dumps(sequence_key(chain_sequences(load_candidate(candidate,Path(candidate.path).parent))))
                if identity not in seen:
                    selected.append(candidate)
                    seen.add(identity)
                    break
            if queue:
                remaining.append(queue)
            if len(selected)==settings["max_pass"]:
                break
        queues=remaining
    return selected


def _configure_motif_policy(cfg, *, legacy=False):
    """New runs check original motif geometry at first round and final only."""
    if legacy:
        return
    from .config import _modalities
    from .criteria import motif_reference_preset
    from src.validate.filters import load_presets
    from partidesign.paths import inference_config_dir
    cfg["motif_policy"] = "first_and_final"
    native = load_presets(inference_config_dir() / "criteria.yaml")
    for name, task in cfg["tasks"].items():
        nucleic = _modalities(cfg["task_payload"][name], name) == {"nuc"}
        source_name = "af3_na_motif" if nucleic else "af3_ame"
        source = task["presets"].get(source_name, native[source_name])
        strict = motif_reference_preset({source_name: source}, nucleic=nucleic)
        task["presets"].setdefault("pipeline_motif", strict)


def run(args):
    import yaml
    from partidesign.paths import using_model_dir
    repo=Path(__file__).resolve().parents[2]
    cfg=build_config(args,repo)
    legacy=bool(cfg["resume"] and (Path(cfg["output_dir"])/"resolved_config.yaml").is_file())
    _configure_motif_policy(cfg, legacy=legacy)
    from .presentation import plan_view, public_config, public_tasks
    if args.plan and not args.show_config:
        print(yaml.safe_dump(plan_view(cfg), sort_keys=False), end="")
        return 0
    if args.plan or args.show_config:
        budget=[]
        for name in cfg["task_names"]:
            task=cfg["tasks"][name]
            authored=cfg["task_payload"][name]
            budget.append({"task":name,"initial_mode":"from_run" if cfg["from_run"] else ("partial" if authored.get("partial") else "de_novo"),
                           "sample_num":authored.get("sample_num",1),
                           "design_length":authored.get("design_length"),
                           "recycle":task["recycle"],"methods":[r["method"] for r in task["rounds"]],
                           "sequences_per_parent":[int(r["include_gt"])+r["mpnn_count"] for r in task["rounds"]],
                           "selection":[r["selection"] for r in task["rounds"]],
                           "openmm_count_per_surviving_lineage":sum(r["relax_enabled"] for r in task["rounds"]),
                           "partial":[r["partial"] for r in task["rounds"]],
                           "structure_multipliers":[(r["partial"]["replicas"]+int(r["partial"]["include_parent"]))
                                                    if r["partial"]["enabled"] else 1 for r in task["rounds"]],
                           "final_pool":task["final"]["pool"],
                           "final_samples_per_sequence":task["final"]["validation"]["af2" if task["final"]["method"]=="af2_ig" else "af3"]["samples"],
                           "final_structures_per_sequence":(2*len(task["final"]["validation"]["af2"]["models"])*task["final"]["validation"]["af2"]["samples"]
                                                            if task["final"]["method"]=="af2_ig" else task["final"]["validation"]["af3"]["samples"]),
                           "final_prediction_selection":task["final"]["prediction_selection"],
                           "final":task["final"]["method"]+" / GT-only"})
        print(yaml.safe_dump({"effective_config":cfg,"budget":budget,
                              "generation_expansion":"sample_num follows native per-length-plan expansion",
                              "capabilities":{"design":"protein, peptide, RNA, DNA",
                                              "af2_ig":"one standard protein design chain with protein context; paired protocols",
                                              "cyclic_offset":"native AF2/AF3; ESM uses explicit covalent bonds",
                                              "context_motion":True,"pure_ligand_design":False,
                                              "polymer_bond_iteration":"special polymer-polymer bonds force esmfold2-fast; final AF3 is unchanged"}},
                             sort_keys=False),end="")
        return 0
    native=generation_runner(cfg,repo)
    from omegaconf import OmegaConf
    cfg["effective_generation"]=OmegaConf.to_container(native.config,resolve=True)
    root=Path(cfg["output_dir"])
    resume=cfg.pop("resume")
    snapshot=(root/"resolved_config.yaml" if legacy else root/".work"/"resolved_config.yaml")
    if root.exists() and not resume:
        raise ValueError(f"Run directory already exists: {root}; use --resume with identical configuration")
    if resume:
        if not snapshot.is_file():
            raise ValueError("--resume requires an existing resolved_config.yaml")
        previous=yaml.safe_load(snapshot.read_text())
        if previous!=cfg:
            raise ValueError("Effective configuration changed; start a new run instead of --resume")
    root.mkdir(parents=True,exist_ok=True)
    if not resume:
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_text(yaml.safe_dump(jsonable(cfg),sort_keys=False))
        (root/"config.yaml").write_text(yaml.safe_dump(public_config(cfg),sort_keys=False))
        (root/"tasks.yaml").write_text(yaml.safe_dump(public_tasks(cfg),sort_keys=False))
    # Layout is runtime state, deliberately absent from strict effective-config
    # comparison so existing runs keep their saved computation and paths.
    if not legacy:
        cfg["_layout"]=TASK_LAYOUT
        # Only relocate native publication. Its parsed tasks and expanded
        # requests remain the exact instances composed for config comparison.
        OmegaConf.update(native.config,"output_dir",str(root/".work"/"generation"),force_add=True)
    work=run_work(cfg)
    (work/"logs").mkdir(parents=True,exist_ok=True)
    from src.validate.storage import locked
    from .backends import NativeOwner
    started=time.monotonic()
    lock=work/"run.lock" if task_layout(cfg) else root/".run.lock"
    with locked(lock), using_model_dir(cfg["model_dir"]):
        initial_state=work/"generation.json" if task_layout(cfg) else root/".generation.json"
        if initial_state.is_file():
            initial={name:[Candidate(**c) for c in rows] for name,rows in json.loads(initial_state.read_text()).items()}
            if any(not Path(c.path).is_file() for rows in initial.values() for c in rows):
                raise ValueError("Saved generation inputs are missing; restore them before resuming")
        else:
            if cfg["from_run"]:
                source=Path(cfg["from_run"])
            else:
                native.run()
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
                source=work/"generation"
            initial=import_candidates(source,root if task_layout(cfg) else root/"generation",
                                      cfg["task_names"],layout=cfg.get("_layout"))
            atomic_json(initial_state,{name:[c.saved() for c in rows] for name,rows in initial.items()})
        owner=_OwnerPool(repo,root,cfg)
        summaries=[]
        live=initial
        history={name:[] for name in cfg["task_names"]}
        invalidated={name:False for name in cfg["task_names"]}
        previous_dirs={name:design_folder(cfg,name) for name in cfg["task_names"]}
        try:
            groups={}
            for name in cfg["task_names"]:
                methods=[r["method"] for r in cfg["tasks"][name]["rounds"]]
                if initial[name] and any(method!="esmfold2-fast" for method in methods):
                    record=_candidate_record(initial[name][0],previous_dirs[name])
                    if has_special_polymer_bond(record):
                        methods=["esmfold2-fast"]*len(methods)
                signature=tuple(methods)
                groups.setdefault(signature,[]).append(name)
            # Compatible tasks advance together; heterogeneous backends do not
            # force a complete model reload on every otherwise identical round.
            for names in groups.values():
                maximum=max(cfg["tasks"][n]["recycle"] for n in names)
                for number in range(1,maximum+1):
                    for name in names:
                        task=cfg["tasks"][name]
                        if number>task["recycle"] or not live[name]:
                            continue
                        stage=task["rounds"][number-1]
                        live[name],summary=run_stage(owner,cfg,name,stage,live[name],previous_dirs[name],
                                                     force=invalidated[name])
                        history[name].extend(summary.pop("_archive",[]))
                        resumed=summary.pop("_resumed")
                        invalidated[name]=invalidated[name] or not resumed
                        summaries.append(summary)
                        previous_dirs[name]=stage_folder(cfg,name,number)
                        write_csv(root/"summary.csv",summaries)
            for name in cfg["task_names"]:
                pool=cfg["tasks"][name]["final"]["pool"]
                live[name]=final_pool(history[name],live[name],pool)
                if not live[name]:
                    continue
                live[name],summary=run_stage(owner,cfg,name,cfg["tasks"][name]["final"],live[name],
                                             previous_dirs[name],final=True,force=invalidated[name])
                summary.pop("_resumed",None)
                summary.pop("_archive",None)
                summaries.append(summary)
                write_csv(root/"summary.csv",summaries)
        finally:
            owner.close()
        atomic_json(work/"summary.json",{"status":"complete","seconds":time.monotonic()-started,
                                         "stages":summaries,"final_counts":{n:len(c) for n,c in live.items()}})
        if not getattr(args,"keep_cache",False):
            from .retention import prune_prediction_cache
            cache=(work/"prediction_cache" if task_layout(cfg) else root/".prediction_cache")
            removed=prune_prediction_cache(cache)
            if removed:
                print(f"Removed {removed} completed prediction caches; structures and metrics retained. "
                      "Use --keep-cache to retain raw confidence data.",flush=True)
    print(f"Pipeline complete: {root}",flush=True)
    return 0


class _ResourceOwner:
    """One Ray reservation owns MPNN, refold and direct OpenMM."""
    def __init__(self, repo, cache):
        from .backends import NativeOwner
        self.owner=NativeOwner(Path(repo),Path(cache),device="cuda:0")
        self.fm=None
    def generate(self,*args,**kwargs):
        return self.owner.generate(*args,**kwargs)
    def predict(self,*args,**kwargs):
        return self.owner.predict(*args,**kwargs)
    def relax(self,*args,**kwargs):
        return _relax(*args,**kwargs)
    def partial(self,*args,**kwargs):
        if self.fm is None:
            from .fm import PartialOwner
            self.fm=PartialOwner()
        return self.fm.run(*args,**kwargs)
    def close(self):
        self.owner.close()
        if self.fm is not None:
            self.fm.close()

class _OwnerPool:
    def __init__(self,repo,root,cfg):
        from .backends import NativeOwner
        self.repo,self.root,self.cfg=repo,root,cfg
        self.cache=(run_work(cfg)/"prediction_cache" if task_layout(cfg) else root/".prediction_cache")
        self.evaluator=NativeOwner(repo,self.cache,device="cpu")
        self.actors=[]
        self.identity=None
        self.started=False

    def activate(self,method,validation,openmm):
        import math
        import os
        import ray
        import torch
        gpu=validation.get("worker_gpus")
        gpu=float(gpu if gpu is not None else {"af2_ig":0.25,"af3":0.5}.get(method,1.0))
        gpu=max(gpu,float(openmm.get("worker_gpus",0)) if openmm["platform"]!="CPU" else 0)
        cpu=max(float(validation.get("worker_cpus",1)),float(openmm.get("worker_cpus",1)))
        if not 0<gpu<=1:
            raise ValueError("pipeline workers require a GPU budget in (0,1]")
        identity=(method,gpu,cpu)
        if identity==self.identity:
            return
        self._close_actors()
        if not self.started:
            if ray.is_initialized():
                raise RuntimeError("pipeline owns one Ray runtime; another runtime is already active")
            gpus=torch.cuda.device_count()
            if not gpus:
                raise RuntimeError("No CUDA GPUs are visible for pipeline")
            budgets=[]
            for task in self.cfg["tasks"].values():
                for stage in [*task["rounds"],task["final"]]:
                    settings=stage["validation"]
                    fraction=settings.get("worker_gpus")
                    fraction=float(fraction if fraction is not None else
                                   {"af2_ig":0.25,"af3":0.5}.get(stage["method"],1.0))
                    fraction=max(fraction,float(task["openmm"]["worker_gpus"])
                                 if task["openmm"]["platform"]!="CPU" else 0)
                    per_cpu=max(float(settings.get("worker_cpus",1)),float(task["openmm"]["worker_cpus"]))
                    budgets.append(math.ceil(math.floor(1/fraction)*gpus*per_cpu))
            ray.init(address="local",num_gpus=gpus,num_cpus=min(os.cpu_count() or 1,max(budgets)),
                     include_dashboard=False,log_to_driver=True)
            self.started=True
        capacity=math.floor(1/gpu)*int(ray.cluster_resources().get("GPU",0))
        capacity=min(capacity,int(ray.cluster_resources().get("CPU",0)//cpu))
        if capacity<1:
            raise RuntimeError(f"Insufficient resources for a worker requesting {gpu:g} GPU and {cpu:g} CPU")
        remote=ray.remote(num_gpus=gpu,num_cpus=cpu,max_restarts=0,
                          runtime_env={"env_vars":{"XLA_PYTHON_CLIENT_PREALLOCATE":"false"}})(_ResourceOwner)
        cache=getattr(self,"cache",self.root/".prediction_cache")
        self.actors=[remote.remote(str(self.repo),str(cache)) for _ in range(capacity)]
        self.identity=identity
        print(f"Resident {method} pool: {capacity} workers, {gpu:g} GPU / {cpu:g} CPU each",flush=True)


    def generate(self,*args,**kwargs):
        import ray
        return ray.get(self.actors[0].generate.remote(*args,**kwargs))

    def predict_many(self,jobs):
        import ray
        from collections import deque
        pending=deque()
        for index,job in enumerate(jobs):
            record,variant,method,validation,task=job
            pending.append(self.actors[index%len(self.actors)].predict.remote(
                record,variant,method,validation,task_name=task))
            if len(pending)>=2*len(self.actors):
                yield ray.get(pending.popleft())
        while pending:
            yield ray.get(pending.popleft())

    def evaluate(self,*args,**kwargs):
        return self.evaluator.evaluate(*args,**kwargs)

    def relax(self,*args,**kwargs):
        import ray
        return ray.get(self.actors[0].relax.remote(*args,**kwargs))

    def relax_many(self,jobs):
        import ray
        pending=[]
        for index,job in enumerate(jobs):
            pending.append(self.actors[index%len(self.actors)].relax.remote(*job))
        for item in pending:
            yield ray.get(item)

    def partial_many(self,references,**kwargs):
        import ray
        count=min(len(references),len(self.actors))
        jobs=[]
        for index in range(count):
            options=dict(kwargs,output=Path(kwargs["output"])/f"worker_{index+1:03d}",
                         seed=(kwargs["seed"]+index*1009)%(2**32-1))
            if task_layout(getattr(self,"cfg",{})):
                public=Path(kwargs["output"])
                options.update(output=public.parent/".work"/"partial"/f"worker_{index+1:03d}",
                               public_output=public)
            jobs.append(self.actors[index].partial.remote(references[index::count],**options))
        result={"candidates":[],"failed":0,"model_loads":0}
        for outcome in ray.get(jobs):
            result["candidates"].extend(outcome["candidates"])
            result["failed"]+=outcome.get("failed",0)
            result["model_loads"]+=outcome.get("model_loads",0)
        return result

    def _close_actors(self, *, abort=False):
        actors,self.actors=self.actors,[]
        self.identity=None
        if not actors:
            return
        import ray
        error=None
        try:
            if not abort:
                ray.get([actor.close.remote() for actor in actors],timeout=30)
        except BaseException as exc:
            error=exc
        for actor in actors:
            try:
                ray.kill(actor,no_restart=True)
            except BaseException as exc:
                if error is None:
                    error=exc
                else:
                    error.add_note(f"Actor cleanup also failed: {exc!r}")
        if error is not None:
            raise error

    def close(self):
        # run() closes from finally. A failed stage can leave prefetched actor
        # work busy, so abort it without queueing close behind more predictions.
        primary=sys.exc_info()[1]
        error=None
        try:
            self._close_actors(abort=primary is not None)
        except BaseException as exc:
            error=exc
        try:
            self.evaluator.close()
        except BaseException as exc:
            if error is None:
                error=exc
            else:
                error.add_note(f"Evaluator cleanup also failed: {exc!r}")
        if self.started:
            self.started=False
            try:
                import ray
                ray.shutdown()
            except BaseException as exc:
                if error is None:
                    error=exc
                else:
                    error.add_note(f"Ray shutdown also failed: {exc!r}")
        if error is not None:
            if primary is None:
                raise error
            if error is not primary:
                primary.add_note(f"Pipeline cleanup also failed: {error!r}")
                for note in getattr(error,"__notes__",()):
                    primary.add_note(note)



def _paired_outcomes(proposals,outcomes):
    from itertools import zip_longest
    sentinel=object()
    for proposal,outcome in zip_longest(proposals,outcomes,fillvalue=sentinel):
        if proposal is sentinel or outcome is sentinel:
            raise RuntimeError("Prediction worker returned an incomplete or extra outcome stream")
        yield proposal,outcome

def _source_bases(source, snapshot):
    source=Path(source).resolve()
    roots=[]
    for origin in (source,Path(snapshot.get("tasks_file") or source).parent,Path(__file__).resolve().parents[2]):
        for parent in (origin,*origin.parents):
            if (parent/"src").is_dir() and (parent/"configs").is_dir():
                roots.append(parent)
                break
    # External authoring folders need not contain a checkout marker.
    authoring=Path(snapshot.get("tasks_file") or source).expanduser()
    authoring=authoring.parent if snapshot.get("tasks_file") else source
    return list(dict.fromkeys([*roots,source,authoring,Path.cwd()]))


def _absolute_references(value,bases,key=""):
    if isinstance(value,dict):
        return {k:_absolute_references(v,bases,str(k)) for k,v in value.items()}
    if isinstance(value,list):
        return [_absolute_references(v,bases,key) for v in value]
    if not isinstance(value,str) or not value:
        return value
    relevant=(key in {"input","reference","source","source_file","reference_file"} or
              key.endswith(("_path","_file")))
    if relevant and "\n" not in value:
        path=Path(value).expanduser()
        if path.is_absolute():
            return str(path)
        for base in bases:
            if (base/path).is_file():
                return str((base/path).resolve())
    return value




def _publish(source,target):
    source,target=Path(source),Path(target)
    if source==target:
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        os.link(source,target)
    except FileExistsError:
        target.unlink()
        _publish(source,target)
    except OSError as exc:
        if exc.errno!=errno.EXDEV:
            raise
        shutil.copy2(source,target)


def _restore_context_molecules(candidate,input_record):
    atoms=candidate.atoms
    import numpy as np
    if not np.any((atoms.mol_type=="ligand") & ~atoms.is_design):
        return
    from rdkit import Chem
    from src.validate.ligands import resolve_context_ligand_components,_canonical_smiles_names
    from src.design.utils.chemistry import private_molecule_from_trusted_graph
    entries,names=resolve_context_ligand_components(input_record.task_dir,atoms,
                                                    design_chains=input_record.design_chains)
    for (chain,component),entry in entries.items():
        if "smiles" not in entry:
            continue
        rows=np.flatnonzero((atoms.label_asym_id==chain)&(atoms.component_uid==component))
        trusted=Chem.RemoveHs(Chem.MolFromSmiles(entry["smiles"]))
        indices={name:i for i,name in enumerate(_canonical_smiles_names(trusted))}
        trusted=Chem.RenumberAtoms(trusted,[indices[names[int(atoms.atom_uid[r])]] for r in rows])
        candidate.private_ligand_molecules[int(atoms.molecule_uid[rows[0]])]=private_molecule_from_trusted_graph(atoms,rows,trusted)
