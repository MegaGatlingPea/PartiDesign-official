"""Select concrete protein refolds from saved named-quality decisions."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import json
import os
from pathlib import Path
import stat

from src.validate.planning import expected_keys, prediction_metadata_valid, sequence_sources_by_sample
from .records import snapshot_refold
from .schema import Candidate, Selection
from .storage import _plain_path, read_json

METHODS = ("af2_ig", "esmfold2-fast", "esmfold2-standard", "af3", "af3-la")
_PEPTIDE = {"peptide", "cyclic-peptide", "cyclic_peptide"}
_AA = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


class _Rejected(ValueError):
    def __init__(self, code, reason):
        super().__init__(reason)
        self.code = code


def _reject(code, reason):
    raise _Rejected(code, reason)


def _names(values):
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError("Selection names must be nonempty strings")
    return set(values)


def _mappings(values):
    if values is None:
        return ()
    if not isinstance(values, Mapping):
        raise ValueError("path_mappings must map absolute old roots to absolute new roots")
    result = []
    for old, new in values.items():
        if not isinstance(old, (str, os.PathLike)) or not isinstance(new, (str, os.PathLike)):
            raise ValueError("Path mapping roots must be paths")
        old, new = Path(old).expanduser(), Path(new).expanduser()
        if not old.is_absolute() or not new.is_absolute() or ".." in old.parts or ".." in new.parts:
            raise ValueError("Path mapping roots must be absolute and normalized")
        result.append((old, new))
    return tuple(sorted(result, key=lambda pair: len(pair[0].parts), reverse=True))


def _mapped(value, mappings):
    if not isinstance(value, str) or not value:
        _reject("metadata_incomplete", "Saved path is missing or is not a string")
    path = Path(value).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        _reject("metadata_mismatch", f"Saved path must be absolute and normalized: {value}")
    for old, new in mappings:
        if path.is_relative_to(old):
            return new / path.relative_to(old)
    return path


def _diagnostic(selection, code, reason, **context):
    selection.diagnostics.append({"stage": "source-selection", "code": code, "reason": str(reason), **context})


def _task_counts(selection, task_dir):
    return selection.counts.setdefault(str(task_dir), {
        "nominated": 0, "deduplicated": 0, "non_ppi": 0,
        "selected_before_limit": 0, "limited_out": 0,
    })


def _method_roots(root):
    if root.name == "success":
        yield root.parent
    elif root.parent.name == "validate":
        yield root
    elif root.name == "validate":
        for method in METHODS:
            path = root / method
            if path.is_dir():
                yield path
    elif (root / "validate").is_dir():
        yield from _method_roots(root / "validate")
    else:
        with os.scandir(root) as entries:
            for entry in sorted(entries, key=lambda entry: entry.name):
                if entry.name.startswith(".") or entry.name in {"original", "postfilter"}:
                    continue
                if Path(entry.name).suffix.lower() in {".cif", ".pdb", ".csv"}:
                    continue
                if entry.is_dir(follow_symlinks=False) and (root / entry.name / "validate").is_dir():
                    yield from _method_roots(root / entry.name / "validate")


def _regular_refold(path, root):
    if not path.is_relative_to(root) or path.suffix.lower() not in {".cif", ".pdb"}:
        _reject("invalid_refold", f"Selected structure is outside its sample refolds directory: {path}")
    path = _plain_path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        _reject("missing_refold", f"Selected refold is missing: {path}")
    if not stat.S_ISREG(mode):
        _reject("invalid_refold", f"Selected refold is not a regular file: {path}")
    return path


def _slots(plan, sample, source_map):
    from src.validate.sequences import normalize_sequence_sources
    sources = source_map.get(sample) if source_map is not None else None
    if source_map is not None and sources is None:
        _reject("metadata_incomplete", f"Saved source plan omits sample {sample}")
    if sources is None:
        sources = plan.get("config", {}).get("sequence", {}).get("sources")
        if not sources:
            _reject("metadata_incomplete", "Saved sequence source plan is missing")
        sources = normalize_sequence_sources(sources, default_gt=False)
    slots = []
    for source in sources:
        for source_index in range(source["count"]):
            slots.append({"seq_index": len(slots), "source": source["source"],
                          "source_index": source_index, "sequence_branch": source["label"]})
    if source_map is None and plan.get("sequence_count") != len(slots):
        _reject("metadata_mismatch", "Saved sequence count disagrees with source slots")
    return slots


def _checked_record(record, *, task, method, sample, slot, expected):
    if not isinstance(record, dict):
        _reject("metadata_incomplete", "Planned sequence record is missing")
    for field, wanted in (("task", task), ("method", method), ("sample", sample)):
        if record.get(field) != wanted:
            _reject("metadata_mismatch", f"Sequence record {field} does not match its saved plan")
    sequence = record.get("sequence")
    if not isinstance(sequence, dict):
        _reject("metadata_incomplete", "Sequence metadata is missing")
    for field in ("seq_index", "source", "source_index"):
        if sequence.get(field) != slot[field] or (field != "source" and type(sequence.get(field)) is not int):
            _reject("metadata_mismatch", f"Sequence {field} does not match its planned slot")
    branch = sequence.get("sequence_branch", "self" if sequence.get("source") == "gt" else sequence.get("source"))
    if branch != slot["sequence_branch"] or ("label" in sequence and sequence["label"] != branch):
        _reject("metadata_mismatch", "Sequence branch does not match its planned source branch")
    sequences = sequence.get("sequences")
    if (not isinstance(sequences, dict) or not sequences
            or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in sequences.items())):
        _reject("metadata_incomplete", "Actual saved sequence map is missing or malformed")
    if record.get("expected_predictions") != expected:
        _reject("metadata_mismatch", "Expected prediction identities disagree with saved method settings")
    if record.get("status") != "complete":
        _reject("incomplete_record", "Sequence record is not complete")
    if not isinstance(record.get("predictions"), dict):
        _reject("metadata_incomplete", "Prediction metadata is missing")
    return sequence


def _passing_keys(record, criteria, selection, context):
    decisions = record.get("decisions")
    if not isinstance(decisions, dict):
        _reject("metadata_incomplete", "Saved named quality decisions are missing")
    selected, passed = defaultdict(set), False
    for name, decision in decisions.items():
        if not isinstance(name, str) or not name.strip() or name != name.strip() or name.lower() == "unfiltered" or (criteria is not None and name not in criteria):
            continue
        if not isinstance(decision, dict):
            _diagnostic(selection, "metadata_incomplete", f"Malformed quality decision {name}", **context)
            continue
        if decision.get("status") != "passed":
            continue
        passed = True
        if decision.get("criteria", name) != name:
            _diagnostic(selection, "metadata_mismatch", f"Decision name disagrees with criteria {name}", **context)
            continue
        keys = decision.get("passing_prediction_keys")
        if not isinstance(keys, list) or not keys or any(not isinstance(key, str) or not key for key in keys):
            _diagnostic(selection, "metadata_incomplete", f"Passed decision {name} has no concrete passing prediction keys", **context)
            continue
        for key in keys:
            selected[key].add(name)
    if not passed:
        _diagnostic(selection, "no_quality_success", "No selected named quality criterion has saved passed status", **context)
    return selected


def _task_definition(task_dir):
    path = _plain_path(task_dir / "resolved_config.yaml")
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return {}
    if not stat.S_ISREG(mode):
        _reject("metadata_incomplete", "Saved task configuration is not a regular file")
    import yaml
    try:
        saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (ValueError, OSError, yaml.YAMLError) as exc:
        _reject("metadata_incomplete", f"Cannot read saved task definition: {exc}")
    if not isinstance(saved, dict) or not isinstance(saved.get("tasks"), dict):
        _reject("metadata_incomplete", "Saved task configuration has no tasks mapping")
    task = saved["tasks"].get(task_dir.name)
    if not isinstance(task, dict):
        _reject("metadata_incomplete", "Saved task configuration omits the selected task")
    return task


def _peptide_intent(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"modality", "sub_modality", "design_modality", "type", "format"} and isinstance(item, str) and item.lower() in _PEPTIDE:
                return True
            if key == "design_chains":
                declarations = item if isinstance(item, list) else [item]
                if any(isinstance(declaration, dict) and any(
                    isinstance(kind, str) and kind.lower() in _PEPTIDE for kind in declaration.values()
                ) for declaration in declarations):
                    return True
            if _peptide_intent(item):
                return True
    elif isinstance(value, list):
        return any(_peptide_intent(item) for item in value)
    return False


def _design_entries(row):
    value = row.get("design_chains")
    try:
        entries = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        _reject("metadata_incomplete", "Saved design_chains is not valid JSON")
    if not isinstance(entries, list) or not entries or any(not isinstance(entry, dict) for entry in entries):
        _reject("metadata_incomplete", "Complete saved design-chain role metadata is required")
    labels = []
    for entry in entries:
        if _peptide_intent(entry):
            _reject("not_ppi", "Explicit peptide design intent is outside the protein interface scope")
        if "sub_modality" in entry and entry["sub_modality"] != "peptide":
            _reject("metadata_incomplete", "Saved design-chain sub_modality is unknown")
        if entry.get("modality") != "protein":
            _reject("not_ppi", "All designed partners must be explicitly declared protein")
        chain = entry.get("chain")
        if (not isinstance(chain, str) or not chain or chain != chain.strip()
                or not isinstance(entry.get("sequence"), str) or not entry["sequence"]
                or type(entry.get("residue_count")) is not int or entry["residue_count"] < 1):
            _reject("metadata_incomplete", "Saved design-chain entries are incomplete")
        if chain in labels:
            _reject("metadata_mismatch", "Saved design-chain labels are duplicated")
        labels.append(chain)
    return labels


def _chain_list(value):
    if isinstance(value, str):
        value = value.split(",") if value else []
    if not isinstance(value, list) or any(not isinstance(chain, str) or not chain.strip() for chain in value):
        _reject("metadata_incomplete", "Antigen chain roles are missing or malformed")
    result = [chain.strip() for chain in value]
    if len(set(result)) != len(result):
        _reject("metadata_mismatch", "Antigen chain roles are duplicated")
    return result


def _roles(row, record, sequence, task_definition):
    if _peptide_intent(row) or _peptide_intent(task_definition) or _peptide_intent(record.get("input_metadata", {})):
        _reject("not_ppi", "Explicit peptide intent is outside the protein interface scope")
    if row.get("design_modality") not in {None, "", "protein", "joint", "antibody"}:
        _reject("not_ppi", "Saved design modality is not explicitly protein")
    authored = task_definition.get("design_chains", [])
    authored = authored if isinstance(authored, list) else [authored]
    for declaration in authored:
        if isinstance(declaration, dict) and any(kind != "protein" for kind in declaration.values()):
            _reject("not_ppi", "Authored design partners are not exclusively protein")
    saved_sequences = sequence["sequences"]
    antibody = bool(row.get("heavy_role") or row.get("heavy_chain") or row.get("antigen_chains") or task_definition.get("format") == "antibody")
    if antibody:
        heavy, light = row.get("heavy_chain"), row.get("light_chain")
        if row.get("heavy_role") not in {"VH", "VHH"} or not isinstance(heavy, str) or not heavy:
            _reject("metadata_incomplete", "Explicit antibody heavy chain and role are required")
        if row["heavy_role"] == "VHH" and light:
            _reject("metadata_mismatch", "VHH cannot claim a light-chain role")
        if light and not isinstance(light, str):
            _reject("metadata_incomplete", "Antibody light-chain role is malformed")
        binder = [heavy] + ([light] if light else [])
        target = _chain_list(row.get("antigen_chains"))
    else:
        binder = _design_entries(row)
        target = [chain for chain in saved_sequences if chain not in binder]
    if not binder or not target:
        _reject("not_ppi", "Both explicit binder and protein-context target assemblies are required")
    if len(set(binder)) != len(binder) or set(binder).intersection(target):
        _reject("metadata_mismatch", "Binder and target chain roles overlap")
    if set(binder + target) != set(saved_sequences):
        _reject("metadata_mismatch", "Assembly roles do not cover the actual saved sequence map")
    metadata = record.get("input_metadata")
    declared = metadata.get("design_chains") if isinstance(metadata, dict) else None
    if (not isinstance(declared, list) or any(not isinstance(chain, str) for chain in declared)
            or len(set(declared)) != len(declared) or set(declared) != set(binder)):
        _reject("metadata_mismatch", "Record design-chain labels disagree with binder roles")
    return binder, target


def _structure_sequences(snapshot):
    annotations = snapshot["annotations"]
    if any(kind != "protein" for kind in annotations["mol_type"]):
        _reject("not_ppi", "Refold contains nucleic-acid, ligand or unknown nonprotein atoms")
    from src.design.utils.atoms import COMMON_STANDARD_PARENT
    residues, chains = {}, defaultdict(list)
    for chain, uid, name in zip(annotations["label_asym_id"], annotations["residue_uid"], annotations["res_name"], strict=True):
        key = (chain, uid)
        if key in residues:
            if residues[key] != name:
                _reject("invalid_refold", "A refold residue has inconsistent chemical identity")
            continue
        residues[key] = name
        letter = _AA.get(name) or _AA.get(COMMON_STANDARD_PARENT.get(name))
        if letter is None:
            _reject("unsupported_chemistry", f"No unambiguous protein sequence identity for component {name}")
        chains[chain].append(letter)
    return {chain: "".join(letters) for chain, letters in chains.items()}


def _map_chains(saved, actual):
    if len(saved) != len(actual):
        _reject("metadata_mismatch", "Refold chain inventory does not match the saved sequence map")
    assigned = {}
    for chain, sequence in saved.items():
        if chain in actual:
            if actual[chain] != sequence:
                _reject("sequence_mismatch", f"Refold chain {chain} differs from its actual saved sequence")
            assigned[chain] = chain
    remaining = {chain for chain in actual if chain not in assigned.values()}
    for chain, sequence in saved.items():
        if chain in assigned:
            continue
        options = [label for label in remaining if actual[label] == sequence]
        if len(options) != 1:
            _reject("ambiguous_chain_map" if options else "sequence_mismatch",
                    f"Saved chain {chain} has {len(options)} unique-sequence matches in the refold")
        assigned[chain] = options[0]
        remaining.remove(options[0])
    return assigned


def _checked_prediction(key, prediction, record, sequence, method):
    if key not in record["expected_predictions"]:
        _reject("metadata_mismatch", "Passing key is not an expected prediction identity")
    if not prediction_metadata_valid(key, prediction, method):
        _reject("metadata_mismatch", "Passing prediction protocol/model/sample identity is invalid")
    if prediction.get("protocol") != "complex":
        _reject("not_complex", "A passing monomer or paired comparison cannot nominate a complex sibling")
    if prediction.get("status") != "complete" or prediction.get("metrics_status") != "complete":
        _reject("incomplete_prediction", "Passing prediction or its saved metric state is incomplete")
    if not isinstance(prediction.get("metrics"), dict) or not isinstance(prediction.get("reasons", {}), dict):
        _reject("metadata_incomplete", "Passing prediction metric containers are malformed")
    for field in ("source", "source_index", "seq_index", "sequence_branch"):
        if field in prediction and (prediction[field] != sequence.get(field)
                or (field in {"source_index", "seq_index"} and type(prediction[field]) is not int)):
            _reject("metadata_mismatch", f"Prediction {field} disagrees with its sequence slot")
    metadata = prediction.get("metadata", {})
    if not isinstance(metadata, dict):
        _reject("metadata_incomplete", "Prediction metadata is malformed")
    for field in ("key", "protocol", "model", "sample"):
        if field in metadata and (metadata[field] != prediction.get(field)
                or type(metadata[field]) is not type(prediction.get(field))):
            _reject("metadata_mismatch", f"Nested prediction {field} disagrees with its slot identity")



def _method_plan(root, task_dir, mappings):
    root = _plain_path(root)
    plan = read_json(root / ".state" / "plan.json")
    if not plan or plan.get("method") != root.name:
        _reject("metadata_incomplete", "Saved method plan is missing or has the wrong method")
    if _mapped(plan.get("task_dir"), mappings) != task_dir:
        _reject("metadata_mismatch", "Saved task_dir differs from method ownership; an explicit path mapping is required after relocation")
    if not isinstance(plan.get("samples"), dict):
        _reject("metadata_incomplete", "Saved plan has no sample mapping")
    return plan


def _candidate_from_state(root, sample, slot, record_path, record, sequence,
                          binder, target, key, names, mappings):
    prediction = record["predictions"].get(key)
    _checked_prediction(key, prediction, record, sequence, root.name)
    path = _mapped(prediction.get("structure_path"), mappings)
    if root.name in {"af3", "af3-la"} and prediction.get("native_cif_path"):
        if path == _mapped(prediction["native_cif_path"], mappings):
            _reject("invalid_refold", "AF3 requires its restored structure_path, not its native CIF")
    path = _regular_refold(path, root / "refolds" / sample)
    snapshot = snapshot_refold(path)
    actual = _structure_sequences(snapshot)
    assigned = _map_chains(sequence["sequences"], actual)
    task_dir = root.parent.parent
    return Candidate(
        task_dir=str(task_dir), method_dir=str(root), task=task_dir.name,
        method=root.name, sample=sample, seq_index=slot["seq_index"],
        source=slot["source"], source_index=slot["source_index"],
        prediction_key=key, refold_path=str(path), record_path=str(record_path),
        binder_chains=tuple(assigned[chain] for chain in binder),
        target_chains=tuple(assigned[chain] for chain in target),
        sequences=actual, passing_criteria=tuple(sorted(names)), source_snapshot=snapshot,
    )


def refresh_candidate(candidate: Candidate, *, path_mappings=None) -> Candidate:
    """Revalidate one current slot within its originally selected quality names.

    Return current physical state for the caller's direct comparison with the
    selected Candidate. Other samples, sequence records and predictions are
    never loaded, and generated input paths are never inspected.
    """
    mappings = _mappings(path_mappings)
    root = _mapped(candidate.method_dir, ())
    task_dir = _mapped(candidate.task_dir, ())
    if (candidate.method not in METHODS or root.name != candidate.method
            or root.parent.name != "validate" or root.parent.parent != task_dir
            or task_dir.name != candidate.task):
        _reject("metadata_mismatch", "Candidate method and task ownership disagree")
    sample = candidate.sample
    if not isinstance(sample, str) or not sample or sample in {".", ".."} or Path(sample).name != sample:
        _reject("metadata_mismatch", "Saved sample identity is not a safe directory name")
    if type(candidate.seq_index) is not int or candidate.seq_index < 0:
        _reject("metadata_mismatch", "Candidate sequence slot is invalid")
    if not isinstance(candidate.prediction_key, str) or not candidate.prediction_key:
        _reject("metadata_mismatch", "Candidate prediction identity is invalid")
    criteria = _names(candidate.passing_criteria)
    if not criteria:
        _reject("no_quality_success", "Candidate has no selected named quality criteria")
    record_path = root / "refolds" / sample / f"seq{candidate.seq_index}.json"
    if _mapped(candidate.record_path, ()) != record_path:
        _reject("metadata_mismatch", "Candidate record path disagrees with its planned slot")
    try:
        plan = _method_plan(root, task_dir, mappings)
        entry = plan["samples"].get(sample)
        if not isinstance(entry, dict) or not isinstance(entry.get("row"), dict):
            _reject("metadata_incomplete", "Current plan omits the candidate sample or its row metadata")
        narrow_plan = {**plan, "samples": {sample: entry}}
        saved_sources = plan.get("sequence_sources_by_sample")
        if isinstance(saved_sources, Mapping):
            narrow_plan["sequence_sources_by_sample"] = (
                {sample: saved_sources[sample]} if sample in saved_sources else {}
            )
        source_map = sequence_sources_by_sample(narrow_plan)
        slot = next((slot for slot in _slots(narrow_plan, sample, source_map)
                     if slot["seq_index"] == candidate.seq_index), None)
        if slot is None:
            _reject("metadata_mismatch", "Candidate sequence slot is no longer planned")
        expected = expected_keys(candidate.method, plan["config"])
        if not expected or len(set(expected)) != len(expected):
            _reject("metadata_incomplete", "Saved method settings do not define unique prediction slots")
        record = read_json(record_path)
        sequence = _checked_record(record, task=candidate.task, method=candidate.method,
                                   sample=sample, slot=slot, expected=expected)
        selection = Selection()
        nominations = _passing_keys(record, criteria, selection, {})
        names = nominations.get(candidate.prediction_key)
        if not names:
            reasons = "; ".join(item["reason"] for item in selection.diagnostics)
            _reject("no_quality_success", reasons or "Current named quality decisions no longer nominate this prediction")
        binder, target = _roles(entry["row"], record, sequence, _task_definition(task_dir))
        return _candidate_from_state(root, sample, slot, record_path, record, sequence,
                                     binder, target, candidate.prediction_key, names, mappings)
    except (KeyError, TypeError) as exc:
        _reject("metadata_incomplete", f"Malformed current candidate state: {exc}")


def select_refolds(inputs, *, methods=None, upstream_criteria=None, task_names=None,
                   limit=None, path_mappings=None) -> Selection:
    """Select only saved, named-quality-passed complex protein predictions."""
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]
    selected_methods, criteria, tasks = _names(methods), _names(upstream_criteria), _names(task_names)
    if selected_methods is not None and selected_methods.difference(METHODS):
        raise ValueError(f"Unsupported postfilter source methods: {sorted(selected_methods.difference(METHODS))}")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    mappings = _mappings(path_mappings)
    selection = Selection()
    roots, seen = [], set()
    for value in inputs:
        raw = Path(value).expanduser()
        context = {"input": str(raw)}
        if raw.name == "result.csv" or raw.suffix.lower() in {".cif", ".pdb"}:
            _diagnostic(selection, "invalid_input", "Select a run, task, validation method or success directory", **context)
            continue
        try:
            root = _plain_path(raw)
            if not root.is_dir():
                _reject("invalid_input", f"Input directory does not exist: {root}")
            discovered = list(_method_roots(root))
            if not discovered:
                _diagnostic(selection, "metadata_incomplete", "No supported saved validation method plan was found", **context)
            for method_root in discovered:
                if method_root not in roots:
                    roots.append(method_root)
        except (OSError, ValueError) as exc:
            _diagnostic(selection, getattr(exc, "code", "invalid_input"), exc, **context)
    candidates = []
    for root in roots:
        method, task_dir = root.name, root.parent.parent
        context = {"task_dir": str(task_dir), "method_dir": str(root), "task": task_dir.name, "method": method}
        if tasks is not None and task_dir.name not in tasks:
            continue
        if selected_methods is not None and method not in selected_methods:
            continue
        try:
            if method not in METHODS:
                _reject("unsupported_method", f"Unsupported source method: {method}")
            root = _plain_path(root)
            plan = _method_plan(root, task_dir, mappings)
            samples = plan["samples"]
            if str(task_dir) not in selection.task_dirs:
                selection.task_dirs.append(str(task_dir))
            task_counts = _task_counts(selection, task_dir)
            source_map = sequence_sources_by_sample(plan)
            expected = expected_keys(method, plan["config"])
            if not expected or len(set(expected)) != len(expected):
                _reject("metadata_incomplete", "Saved method settings do not define unique prediction slots")
            definition = _task_definition(task_dir)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            _diagnostic(selection, getattr(exc, "code", "metadata_incomplete"), exc, **context)
            continue
        for sample, entry in samples.items():
            sample_context = {**context, "sample": sample}
            try:
                if not isinstance(sample, str) or not sample or sample in {".", ".."} or Path(sample).name != sample:
                    _reject("metadata_mismatch", "Saved sample identity is not a safe directory name")
                if not isinstance(entry, dict) or not isinstance(entry.get("row"), dict):
                    _reject("metadata_incomplete", "Saved sample row metadata is missing")
                slots = _slots(plan, sample, source_map)
            except (KeyError, TypeError, ValueError) as exc:
                _diagnostic(selection, getattr(exc, "code", "metadata_incomplete"), exc, **sample_context)
                continue
            for slot in slots:
                sequence_context = {**sample_context, "seq_index": slot["seq_index"]}
                record_path = root / "refolds" / sample / f"seq{slot['seq_index']}.json"
                nominations = {}
                try:
                    record = read_json(record_path)
                    sequence = _checked_record(record, task=task_dir.name, method=method,
                                               sample=sample, slot=slot, expected=expected)
                    for key, names in _passing_keys(record, criteria, selection, sequence_context).items():
                        identity = (str(root), sample, slot["seq_index"], key)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        nominations[key] = names
                        task_counts["nominated"] += len(names)
                        task_counts["deduplicated"] += 1
                    if not nominations:
                        continue
                    binder, target = _roles(entry["row"], record, sequence, definition)
                except (OSError, TypeError, ValueError) as exc:
                    if getattr(exc, "code", None) == "not_ppi":
                        task_counts["non_ppi"] += len(nominations)
                    _diagnostic(selection, getattr(exc, "code", "metadata_incomplete"), exc, **sequence_context)
                    continue
                for key, names in nominations.items():
                    prediction_context = {**sequence_context, "prediction_key": key}
                    try:
                        candidates.append(_candidate_from_state(
                            root, sample, slot, record_path, record, sequence,
                            binder, target, key, names, mappings,
                        ))
                    except (KeyError, OSError, TypeError, ValueError) as exc:
                        _diagnostic(selection, getattr(exc, "code", "invalid_refold"), exc, **prediction_context)
    claims = defaultdict(list)
    for candidate in candidates:
        try:
            file_state = Path(candidate.refold_path).lstat()
        except OSError as exc:
            _diagnostic(selection, "missing_refold", exc, task_dir=candidate.task_dir,
                        method_dir=candidate.method_dir, sample=candidate.sample,
                        seq_index=candidate.seq_index, prediction_key=candidate.prediction_key)
            continue
        claims[(file_state.st_dev, file_state.st_ino)].append(candidate)
    for owners in claims.values():
        if len(owners) > 1:
            for candidate in owners:
                _diagnostic(selection, "conflicting_refold", f"Different prediction identities claim the same refold: {candidate.refold_path}",
                            task_dir=candidate.task_dir, method_dir=candidate.method_dir, sample=candidate.sample,
                            seq_index=candidate.seq_index, prediction_key=candidate.prediction_key)
        else:
            selection.candidates.append(owners[0])
            _task_counts(selection, owners[0].task_dir)["selected_before_limit"] += 1
    if limit is not None:
        retained = defaultdict(int)
        limited = []
        for candidate in selection.candidates:
            if retained[candidate.task_dir] < limit:
                limited.append(candidate)
                retained[candidate.task_dir] += 1
            else:
                _task_counts(selection, candidate.task_dir)["limited_out"] += 1
        selection.candidates = limited
    return selection
