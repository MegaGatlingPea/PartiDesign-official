"""Independent, atomic postfilter tables for concrete refold identities."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import tempfile

BASE_FIELDS=("task","method","sample","seq_index","sequence_source","source_index",
             "prediction_key","passing_criteria","input_refold","relaxed_refold","status")

def _atomic(path,text):
    from .storage import _plain_path
    path=_plain_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    _plain_path(path)
    descriptor,name=tempfile.mkstemp(prefix="."+path.name+"-",dir=path.parent)
    try:
        with os.fdopen(descriptor,"w") as stream:
            stream.write(text);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def write_reports(output_dir,records,diagnostics=(),selection_counts=None):
    from io import StringIO
    output=Path(output_dir)
    rows=[];metric_names=set();criteria_names=set();success=[]
    for record in records:
        candidate=record["candidate"]
        row={key:candidate.get(key,"") for key in ("task","method","sample","seq_index","source_index","prediction_key")}
        row.update(sequence_source=candidate["source"],passing_criteria=json.dumps(candidate["passing_criteria"],separators=(",",":")),
                   input_refold=candidate["refold_path"],relaxed_refold=record.get("relaxed_path",""),status=record["status"])
        metrics=record.get("metrics",{});metric_names.update(metrics);row.update(metrics)
        decisions=record.get("decisions",{});criteria_names.update(decisions)
        for name,decision in decisions.items():
            row[name+".status"]=decision["status"]
            row[name+".reason"]="; ".join(decision.get("reasons",[]))
        row["error"]=record.get("error","")
        row["metric_errors"]=json.dumps(record.get("errors",{}),sort_keys=True,separators=(",",":"))
        rows.append(row)
        passed=bool(decisions) and all(d.get("status")=="passed" for d in decisions.values())
        if record["status"]=="complete" and passed:
            path=Path(row["relaxed_refold"])
            owned=(output/"structures").absolute()
            if owned.is_symlink() or ".." in path.parts:
                raise ValueError("successful structure has a symlink or unsafe owned path")
            try:relative=path.absolute().relative_to(owned)
            except ValueError:raise ValueError("successful structure is outside owned structures") from None
            cursor=owned
            for part in relative.parts:
                cursor=cursor/part
                if cursor.is_symlink():raise ValueError("successful structure path contains a symlink")
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to((output/"structures").resolve()):
                raise ValueError("successful postfilter structure is absent or outside owned structures")
            success.append(row)
    fields=list(BASE_FIELDS)+sorted(metric_names)+[field for name in sorted(criteria_names) for field in (name+".status",name+".reason")]+["metric_errors","error"]
    def table(values):
        stream=StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(values);return stream.getvalue()
    summary={"status":"complete","selected":len(rows),"relax_complete":sum(r.get("relaxed_path") is not None and bool(r.get("relaxed_path")) for r in records),
             "execution_failures":sum(r["status"]!="complete" for r in records),"passed":len(success),
             "metrics_complete":sum(r["status"]=="complete" and not r.get("errors") for r in records),
             "required_metrics_complete":sum(bool(r.get("decisions")) and all(
                 d.get("status")!="unavailable" for d in r["decisions"].values()) for r in records),
             "criteria":{name:{state:sum(r.get("decisions",{}).get(name,{}).get("status")==state for r in records) for state in ("passed","failed","unavailable")} for name in sorted(criteria_names)},
             "selection_counts":selection_counts or {},"diagnostics":list(diagnostics)}
    rendered={"metrics.csv":table(rows),"success.csv":table(success),"summary.json":json.dumps(summary,indent=2,allow_nan=False)+"\n"}
    _atomic(output/".state"/"report_status.json",'{"status":"writing"}\n')
    for name,text in rendered.items():_atomic(output/name,text)
    _atomic(output/".state"/"report_status.json",'{"status":"complete"}\n')
    return summary
