"""Lightweight screen YAML/CSV parsing; no prediction SDK imports."""
from __future__ import annotations
import copy
import csv
from pathlib import Path
import re
from src.validate.af3_msa import validate_msa_settings
from src.validate.msa_options import cli_settings

METHODS = ("esmfold2-fast", "esmfold2-standard", "af3", "af3-la", "mol1", "mol1-constraint", "boltz2")
MODALITIES = ("protein", "peptide", "rna", "dna", "ligand")
_TASK_FIELDS = {"methods", "samples", "contexts", "candidates", "ranking", "msa"}
_CHAIN_FIELDS = {"type", "sequence", "pocket", "msa"}
_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_CHAIN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_TOKEN = re.compile(r"[A-Za-z]")
_ALPHABETS = {"protein": set("ACDEFGHIKLMNPQRSTVWY"), "rna": set("ACGU"), "dna": set("ACGT")}

def canonical_modality(value):
    if value not in MODALITIES:
        raise ValueError(f"modality must be one of {', '.join(MODALITIES)}")
    return "protein" if value == "peptide" else value

def validate_sequence(value, modality):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("sequence must be a nonempty string")
    value = value.strip()
    kind = canonical_modality(modality)
    if kind == "ligand":
        from rdkit import Chem, rdBase
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(value)
        if molecule is None or not molecule.GetNumHeavyAtoms() or len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError("ligand seq must be one valid connected SMILES")
        return value
    value = value.upper()
    tokens = _TOKEN.findall(value)
    if "".join(tokens) != value or not tokens:
        raise ValueError("invalid polymer sequence syntax")
    if any(len(token) == 1 and token not in _ALPHABETS[kind] for token in tokens):
        raise ValueError(f"sequence contains an invalid {kind} symbol")
    return value

def residue_count(sequence):
    return len(_TOKEN.findall(sequence))

def _closed(value, allowed, label):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{label} must be a mapping")
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(extra))}")

def load_yaml(path):
    import yaml
    class UniqueLoader(yaml.SafeLoader):
        pass
    # YAML 1.2 booleans: MSA mode `off` is text, not PyYAML's legacy False.
    UniqueLoader.yaml_implicit_resolvers = {
        key: [(tag, pattern) for tag, pattern in rules if tag != 'tag:yaml.org,2002:bool']
        for key, rules in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    UniqueLoader.add_implicit_resolver('tag:yaml.org,2002:bool', re.compile(r'^(?:true|false|True|False|TRUE|FALSE)$'), list('tTfF'))
    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError(f"YAML keys must be unique strings: {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        result = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid screen YAML: {exc}") from exc
    if not isinstance(result, dict) or not result:
        raise ValueError("screen YAML must contain named tasks")
    return result

def read_candidates(path):
    rows, seen = [], set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != 3 or set(reader.fieldnames) != {"id", "modality", "seq"}:
            raise ValueError("candidate CSV must have exactly id,modality,seq columns")
        for number, raw in enumerate(reader, 1):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"candidate CSV row {number} has the wrong number of fields")
            identifier = raw["id"].strip()
            if not identifier or identifier in seen:
                raise ValueError(f"candidate IDs must be nonempty and unique: {identifier!r}")
            seen.add(identifier)
            row = {"id": identifier, "modality": raw["modality"].strip().lower(), "seq": raw["seq"].strip(), "row_index": number, "error": None}
            try:
                row["seq"] = validate_sequence(row["seq"], row["modality"])
            except ValueError as exc:
                row["error"] = str(exc)
            rows.append(row)
    if not rows:
        raise ValueError("candidate CSV contains no candidates")
    return rows

def load_tasks(path, *, selected=(), args=None, cwd=None):
    cwd = Path(cwd or Path.cwd())
    raw = load_yaml(path)
    chosen = list(selected) or list(raw)
    if len(set(chosen)) != len(chosen) or set(chosen) - set(raw):
        raise ValueError("selected screen tasks must be unique names present in the YAML")
    override = cli_settings(args) if args is not None else {}
    tasks = {}
    for name in chosen:
        if not _NAME.fullmatch(name):
            raise ValueError(f"Invalid screen task name: {name!r}")
        value = raw[name]
        _closed(value, _TASK_FIELDS, f"screen task {name}")
        methods = value.get("methods")
        if (not isinstance(methods, list) or not methods or any(method not in METHODS for method in methods) or len(set(methods)) != len(methods)):
            raise ValueError(f"{name}.methods must be a unique nonempty list from {', '.join(METHODS)}")
        samples = value.get("samples", 1)
        if type(samples) is not int or samples < 1:
            raise ValueError(f"{name}.samples must be a positive integer")
        contexts = value.get("contexts")
        if not isinstance(contexts, dict) or not contexts:
            raise ValueError(f"{name}.contexts must contain at least one sequence-defined chain")
        shared = value.get("msa", {})
        validate_msa_settings(shared)
        chains = []
        for identifier, entry in contexts.items():
            if not isinstance(identifier, str) or not _CHAIN.fullmatch(identifier) or identifier in {".", ".."}:
                raise ValueError(f"Invalid context chain ID: {identifier!r}")
            _closed(entry, _CHAIN_FIELDS, f"{name}.contexts.{identifier}")
            kind = canonical_modality(entry.get("type"))
            sequence = validate_sequence(entry.get("sequence"), entry["type"])
            chain = {"id": identifier, "role": "context", "modality": kind,
                     "smiles" if kind == "ligand" else "sequence": sequence}
            if entry["type"] == "peptide":
                chain["sub_modality"] = "peptide"
            pocket = entry.get("pocket", [])
            if not isinstance(pocket, list) or any(type(i) is not int or i < 1 for i in pocket):
                raise ValueError(f"{name}.{identifier}.pocket must contain positive 1-based residue positions")
            if pocket and kind == "ligand":
                raise ValueError("SMILES characters are not residue positions; ligand context pocket lists are unsupported")
            if pocket and max(pocket) > residue_count(sequence):
                raise ValueError(f"{name}.{identifier}.pocket position is outside its sequence")
            if pocket:
                chain["pocket"] = list(dict.fromkeys(pocket))
            local = entry.get("msa", {})
            validate_msa_settings(local)
            chain["msa"] = validate_msa_settings({**shared, **local, **override})
            chains.append(chain)
        ranking = value.get("ranking")
        if ranking is not None:
            _closed(ranking, {"metric", "order"}, f"{name}.ranking")
            if not isinstance(ranking.get("metric"), str) or not ranking["metric"].strip():
                raise ValueError("ranking.metric must be a nonempty metric name")
            if ranking.get("order", "desc") not in {"asc", "desc"}:
                raise ValueError("ranking.order must be asc or desc")
            ranking = {"metric": ranking["metric"], "order": ranking.get("order", "desc")}
        filename = value.get("candidates")
        if not isinstance(filename, str) or not filename.strip() or "://" in filename:
            raise ValueError(f"{name}.candidates must be a local CSV path")
        candidate_path = Path(filename).expanduser()
        candidate_path = (candidate_path if candidate_path.is_absolute() else cwd / candidate_path).resolve()
        tasks[name] = dict(name=name, methods=list(methods), samples=samples, contexts=chains,
                           candidates=str(candidate_path), rows=read_candidates(candidate_path), ranking=ranking)
    return tasks

def request_for(task, row):
    if row.get("error"):
        raise ValueError(row["error"])
    labels = {chain["id"] for chain in task["contexts"]}
    candidate_id, index = "D", 1
    while candidate_id in labels:
        candidate_id = f"D{index}"
        index += 1
    modality = canonical_modality(row["modality"])
    candidate = {"id": candidate_id, "role": "design", "modality": modality,
                 "smiles" if modality == "ligand" else "sequence": row["seq"]}
    if row["modality"] == "peptide":
        candidate["sub_modality"] = "peptide"
    return {"version": 1, "name": f"screen_{row['row_index']:06d}",
            "chains": [*copy.deepcopy(task["contexts"]), candidate], "design_chains": [candidate_id]}

def affinity_eligible(request):
    designed = [c for c in request["chains"] if c["role"] == "design"]
    contexts = [c for c in request["chains"] if c["role"] == "context"]
    return (len(designed) == 1 and designed[0]["modality"] == "ligand"
            and any(c["modality"] == "protein" for c in contexts)
            and all(c["modality"] in {"protein", "ligand"} for c in contexts))

def ranking_for(task, modality, method):
    if task.get("ranking"):
        return task["ranking"]
    metric = ("boltz2_affinity_probability_binary" if method == "boltz2" else "ligand_iptm") if modality == "ligand" else "design_ipsae"
    return {"metric": metric, "order": "desc"}
