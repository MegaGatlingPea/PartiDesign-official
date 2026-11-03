"""Resident Mol-1 inference with the shared context MSA/template cache."""
from __future__ import annotations

import copy
from dataclasses import is_dataclass, replace
from collections.abc import Mapping
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from partidesign.paths import community_asset, community_source
from src.design.rewards.af3_input import build_af3_input, msa_queries, attach_msa_references
from src.design.rewards.af3_backend import _variant_with_ligand_smiles, _protocols

ADAPTER_REVISION = "native_covalent_v2"


def _conditioning_input(input_record):
    """Materialize declared model-only links on a private chemistry view."""
    from .esmfold2 import _bond_payload, _canonical_chain_values, _token_positions
    from .sequences import _field, _decode_jsonish
    raw = _field(input_record, "covalent_bonds", None)
    if raw is None:
        raw = _field(input_record, "bond_constraints", ())
    if not _decode_jsonish(raw, default=()):
        return input_record, set()
    atoms = _field(input_record, "atoms")
    declared = _bond_payload(input_record, atoms)
    if not declared:
        return input_record, set()
    positions = _token_positions(atoms)
    chains = _canonical_chain_values(atoms)
    rows = {}
    for i, (chain, token, name) in enumerate(zip(chains, atoms.token_uid, atoms.atom_name)):
        key = (str(chain), positions[(str(chain), int(token))], str(name))
        rows.setdefault(key, []).append(i)
    graph = [] if atoms.bonds is None else atoms.bonds.as_array().tolist()
    existing = {tuple(sorted((int(a), int(b)))) for a, b, _ in graph}
    added = set()
    for bond in declared:
        pair = []
        for endpoint in (bond["endpoint1"], bond["endpoint2"]):
            key = (endpoint["chain_id"], endpoint["residue_index"], endpoint["atom_name"])
            matches = rows.get(key, ())
            if len(matches) != 1:
                raise ValueError(f"Mol-1 covalent bond endpoint is not unique: {endpoint}")
            pair.append(matches[0])
        edge = tuple(sorted(pair))
        if edge in existing and atoms.residue_uid[edge[0]] == atoms.residue_uid[edge[1]]:
            continue  # CCD-internal chemistry keeps its native bond orders.
        if bond["bond_order"] not in {"single", "?"}:
            raise ValueError("Mol-1 explicit inter-residue bonds support single or unspecified bond orders")
        if edge not in existing:
            added.add(edge)
            existing.add(edge)
            graph.append([*edge, 1])
    if not added:
        return input_record, set()
    from biotite.structure import BondList
    private_atoms = atoms.copy()
    private_atoms.bonds = BondList(len(atoms), np.asarray(graph, dtype=np.uint32))
    if isinstance(input_record, Mapping):
        private = dict(input_record, atoms=private_atoms)
    elif is_dataclass(input_record):
        private = replace(input_record, atoms=private_atoms)
    else:
        private = copy.copy(input_record)
        private.atoms = private_atoms
    return private, added


def _fixed_inventory_ccd(input_record, repo_root):
    """Use Mol-1's own cached CCD for explicitly fixed exceptional inventories."""
    from .sequences import _field, _row_field
    declared = _field(input_record, "fixed_restypes")
    if declared is None:
        declared = _row_field(input_record, "fixed_restypes", ())
    if isinstance(declared, str):
        declared = json.loads(declared) if declared.strip() else ()
    if isinstance(declared, Mapping):
        declared = declared.get("residues", declared)
        if isinstance(declared, Mapping):
            declared = (declared,)
    if not isinstance(declared, (list, tuple)) or not any(
            isinstance(item, Mapping) and "atom_names" in item for item in declared):
        return None
    from src.design.rewards.af3_input import _fixed_atom_inventories, _record_atoms
    atoms = _record_atoms(input_record)
    inventories = _fixed_atom_inventories(input_record, atoms)
    if not inventories:
        return None
    import sys
    os.environ["MOL_ROOT_DIR"] = str(community_asset("mol1", repo_root))
    source = str(community_source("mol1", repo_root))
    if source not in sys.path:
        sys.path.insert(0, source)
    from mol.data.core import ccd
    native = ccd.biotite_load_ccd_cif()
    names = set(str(atoms.res_name[row]) for row in range(len(atoms))
                if int(atoms.residue_uid[row]) in inventories)
    result = {}
    for name in names:
        if name not in native:
            raise ValueError(f"Mol-1 has no authoritative CCD for fixed inventory {name!r}")
        result[name] = {
            f"_{category}.{column}": values.as_array(str).tolist()
            for category, table in native[name].items() for column, values in table.items()
        }
    return result


def prepare_mol1_inputs(input_record, variant, config, repo_root, *, cache=None):
    """Prepare chemistry and context searches without loading a prediction model."""
    variant = _variant_with_ligand_smiles(input_record, variant)
    conditioned, added = _conditioning_input(input_record)
    fixed_ccd = _fixed_inventory_ccd(conditioned, repo_root)
    payloads = {protocol: build_af3_input(
        conditioned, variant, seed=config["seed"], protocol=protocol, ccd=fixed_ccd,
    ).payload() for protocol in _protocols(config)}
    for payload in payloads.values():
        if added:
            payload["mol1_covalent_bonds"] = mol1_covalent_bonds(payload)
            payload["source_inter_residue_bonds"] = [
                edge for edge in payload["source_inter_residue_bonds"]
                if tuple(sorted((int(edge[0]), int(edge[1])))) not in added]
        if payload["name"].startswith("af3"):
            payload["name"] = "mol1" + payload["name"][3:]
    from .af3_msa import AF3MSACache
    settings = config["msa"]
    queries = {(q["modality"], q["sequence"]) for payload in payloads.values()
               for q in msa_queries(payload, input_record.design_chains, settings=settings)}
    refs = []
    if queries:
        cache = cache or AF3MSACache(repo_root, settings)
        refs = [cache.ensure(modality, seq) for modality, seq in sorted(queries)]
    return {protocol: attach_msa_references(payload, input_record.design_chains, refs, settings=settings)
            for protocol, payload in payloads.items()}


def mol1_job(payload, directory):
    """Convert canonical chain/chemistry ledgers; never run native online searches."""
    directory = Path(directory)
    sequences, entities = [], {}
    template_files = {}
    text_files = {}

    def text_file(path):
        path = str(path)
        if path not in text_files:
            text_files[path] = Path(path).read_text()
        return text_files[path]

    for number, chain in enumerate(payload["chains"], 1):
        entity = {"count": 1, "id": [chain["id"]]}
        modality = chain["modality"]
        entities[chain["id"]] = str(number)
        if modality in {"protein", "rna", "dna"}:
            entity["sequence"] = chain["sequence"]
            key = {"protein": "proteinChain", "rna": "rnaSequence", "dna": "dnaSequence"}[modality]
            if chain.get("modifications"):
                entity["modifications"] = [
                    ({"ptmType": "CCD_" + name, "ptmPosition": pos} if modality == "protein"
                     else {"modificationType": "CCD_" + name, "basePosition": pos})
                    for name, pos in chain["modifications"]]
            # Inline A3M is consumed by the SDK directly, not the online CLI.
            entity["unpairedMsa"] = f">query\n{chain['sequence']}\n"
            reference = chain.get("msa_reference")
            if reference:
                entity["unpairedMsa"] = text_file(reference["unpaired_msa_path"])
                if reference.get("paired_msa_path"):
                    entity["pairedMsa"] = text_file(reference["paired_msa_path"])
                if reference.get("templates"):
                    template_key = tuple(
                        (str(t["mmcif_path"]),
                         tuple(sorted((int(a), int(b)) for a, b in t["query_to_template_map"].items())),
                         t.get("release_date")) for t in reference["templates"])
                    path = template_files.get(template_key)
                    if path is None:
                        templates = []
                        for template in reference["templates"]:
                            mapping = sorted((int(a), int(b)) for a, b in template["query_to_template_map"].items())
                            item = {"mmcif": text_file(template["mmcif_path"]),
                                    "queryIndices": [a for a, _ in mapping],
                                    "templateIndices": [b for _, b in mapping]}
                            if template.get("release_date") is not None:
                                item["release_date"] = template["release_date"]
                            templates.append(item)
                        path = directory / f"templates_{chain['id']}.json"
                        path.write_text(json.dumps(templates))
                        template_files[template_key] = path
                    entity["templatesPath"] = str(path)
        else:
            key = "ligand"
            entity["ligand"] = chain.get("smiles") or "CCD_" + "_".join(chain["ccd_ids"])
            if not chain.get("smiles"):
                entity["ccdCodes"] = list(chain["ccd_ids"])
        sequences.append({key: entity})
    return {"name": payload["name"], "sequences": sequences,
            "covalent_bonds": mol1_covalent_bonds(payload), "preserve_modified_residues": True}


def mol1_covalent_bonds(payload):
    """Canonical native bond input shared by execution and request deduplication."""
    if "mol1_covalent_bonds" in payload:
        return payload["mol1_covalent_bonds"]
    entities = {chain["id"]: str(i) for i, chain in enumerate(payload["chains"], 1)}
    endpoints = {}
    for native_chain, residues in payload["source_residue_map"].items():
        for pos, residue in enumerate(residues, 1):
            if residue is not None:
                endpoints[(residue["source_chain_id"], int(residue["source_res_id"]),
                           residue.get("source_ins_code", ""))] = (native_chain, pos)
    bonds = {tuple(sorted(tuple(e) for e in pair)) for pair in payload["bonded_atom_pairs"]}
    ledger = {int(row["row"]): row for row in payload["source_atom_ledger"]}
    for left, right, order in payload["source_inter_residue_bonds"]:
        a, b = ledger[left], ledger[right]
        if a["mol_type"] not in {"protein", "rna", "dna"} or b["mol_type"] not in {"protein", "rna", "dna"}:
            continue
        x = endpoints.get((a["chain_id"], int(a["res_id"]), a.get("ins_code", "")))
        y = endpoints.get((b["chain_id"], int(b["res_id"]), b.get("ins_code", "")))
        if x is None or y is None:
            continue
        forward = (a["atom_name"], b["atom_name"]) in {("C", "N"), ("O3'", "P")}
        reverse = (b["atom_name"], a["atom_name"]) in {("C", "N"), ("O3'", "P")}
        normal = x[0] == y[0] and ((y[1] == x[1] + 1 and forward) or
                                      (x[1] == y[1] + 1 and reverse))
        if not normal:
            if int(order) not in {0, 1}:
                raise ValueError("Mol-1 explicit inter-residue bonds support single or unspecified bond orders")
            bonds.add(tuple(sorted(((x[0], x[1], a["atom_name"]), (y[0], y[1], b["atom_name"])))))
    covalent = []
    for a, b in sorted(bonds):
        covalent.append({"entity1": entities[a[0]], "copy1": 1, "position1": str(a[1]), "atom1": a[2],
                         "entity2": entities[b[0]], "copy2": 1, "position2": str(b[1]), "atom2": b[2]})
    return covalent


def _register_user_ccd(text):
    """Install only this request's authoritative custom components in the SDK."""
    if not text:
        return
    from io import StringIO
    import gemmi
    from biotite.structure.io.pdbx import CIFFile, CIFColumn
    from pdbeccdutils.core import ccd_reader
    from mol.data.core import ccd
    blocks = CIFFile.read(StringIO(text))
    native = ccd.biotite_load_ccd_cif()
    # Load the native RDKit cache before adding request-local components.
    ccd.get_component_rdkit_mol("ALA")
    for name, block in blocks.items():
        atom = block["chem_comp_atom"]
        n = atom.row_count
        for field, values in {
            "alt_atom_id": atom["atom_id"].as_array(),
            "pdbx_component_atom_id": atom["atom_id"].as_array(),
            "pdbx_leaving_atom_flag": ["N"] * n,
            **{f"model_Cartn_{axis}": atom[f"pdbx_model_Cartn_{axis}_ideal"].as_array() for axis in "xyz"},
        }.items():
            if field not in atom:
                atom[field] = CIFColumn(values)
        if "one_letter_code" not in block["chem_comp"]:
            block["chem_comp"]["one_letter_code"] = CIFColumn(["?"])
        for field, default in {"pdbx_modified_date": "?", "pdbx_release_status": "REL"}.items():
            if field not in block["chem_comp"]:
                block["chem_comp"][field] = CIFColumn([default])
        # PDBe requires complete atom-column groups, including model coordinates.
        # Parse the completed block rather than the minimal incoming CCD text.
        stream = StringIO()
        one = CIFFile()
        one[name] = block
        one.write(stream)
        parsed = gemmi.cif.read_string(stream.getvalue())
        component = ccd_reader._parse_pdb_mmcif(parsed.find_block(name)).component
        mol = component.mol
        if not mol.GetNumConformers():
            raise ValueError(f"Mol-1 custom component {name} has no reference conformer")
        mol.atom_map = {a.GetProp("name"): a.GetIdx() for a in mol.GetAtoms()}
        mol.ref_conf_id = mol.GetConformer().GetId()
        mol.ref_mask = np.isfinite(mol.GetConformer().GetPositions()).all(axis=1)
        native[name] = block
        ccd._ccd_rdkit_mols[name] = mol
    # These small lookup caches may contain a component alias from an earlier
    # request. Keep the already-loaded native chemistry and refresh the lookups.
    for function in (ccd.get_component_atom_array, ccd.get_one_letter_code,
                     ccd.get_mol_type, ccd.get_ccd_ref_info):
        function.cache_clear()


class _Mol1CCD:
    """Expose the resident SDK chemistry in the common restoration format."""
    def get(self, name):
        from mol.data.core import ccd
        block = ccd.biotite_load_ccd_cif().get(name)
        if block is None:
            return None
        return {f"_{category}.{column}": values.as_array().tolist()
                for category, fields in block.items() for column, values in fields.items()}


class Mol1Backend:
    """One native model per validation worker, on either CUDA runtime."""
    def __init__(self, repo_root, config, device=None):
        self.repo = Path(repo_root)
        self.config = config
        self.device = device or "cuda:0"
        self.runner = self.cache = None
        self.log_dir = None

    def set_log_dir(self, directory):
        self.log_dir = Path(directory)

    def _load(self, directory):
        if self.runner is not None:
            return
        import sys
        root = community_asset("mol1", self.repo)
        os.environ["MOL_ROOT_DIR"] = str(root)
        os.environ.setdefault("LAYERNORM_TYPE", "torch")
        source = str(community_source("mol1", self.repo))
        if source not in sys.path:
            sys.path.insert(0, source)
        from mol.config.base import configs as base
        from mol.config.data import data_configs
        from mol.config.inference import inference_configs
        from mol.config.model_type import model_configs
        from mol.config.parser import parse_configs
        from mol.runner.inference import InferenceRunner
        from mol.data.inference.infer_dataloader import InferenceDataset
        from mol.data.template.template_utils import TemplateHitFeaturizer
        from ml_collections import ConfigDict
        cfg = parse_configs({**base, "data": data_configs, **inference_configs}, fill_required_with_null=True)
        cfg.update(ConfigDict(model_configs["mol-1-base"]))
        cfg.device = str(self.device)
        cfg.dump_dir = str(directory)
        cfg.load_checkpoint_path = str(Path(self.config.get("checkpoint") or root / "checkpoint/mol-1-base.pt").resolve())
        cfg.model.N_cycle = self.config["recycles"]
        cfg.sample_diffusion.N_step = self.config["steps"]
        cfg.sample_diffusion.N_sample = self.config["samples"]
        cfg.dtype = self.config["dtype"]
        cfg.triangle_multiplicative = self.config["triangle_multiplicative"]
        cfg.triangle_attention = self.config["triangle_attention"]
        cfg.need_atom_confidence = True
        import torch
        with torch.device(self.device):
            self.runner = InferenceRunner(cfg)
        self.dataset = object.__new__(InferenceDataset)
        self.dataset.use_msa = True
        self.dataset.msa_pair_as_unpair = True
        self.dataset.use_rna_msa = True
        self.dataset.use_template = True
        # Cached templates have exact alignments, so neither Kalign nor PDBe is used.
        self.dataset.online_template_featurizer = TemplateHitFeaturizer(
            mmcif_dir=str(root / "mmcif"), fetch_remote=False)

    def predict(self, input_record, variant, output, *, completed_keys=(), prepared_inputs=None):
        from .af3_msa import AF3MSACache
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if prepared_inputs is None:
            self.cache = self.cache or AF3MSACache(self.repo, self.config["msa"])
            prepared_inputs = prepare_mol1_inputs(input_record, variant, self.config, self.repo, cache=self.cache)
        wanted = [(protocol, payload) for protocol, payload in prepared_inputs.items()
                  if any(f"{protocol}_sample{i}" not in completed_keys for i in range(self.config["samples"]))]
        if not wanted:
            return
        self._load(output)
        import torch
        from mol.utils.seed import seed_everything
        for protocol, payload in wanted:
            folder = output / protocol
            folder.mkdir(parents=True, exist_ok=True)
            seed_everything(self.config["seed"], deterministic=False)
            job = mol1_job(payload, folder)
            _register_user_ccd(payload.get("user_ccd"))
            try:
                data, atoms, _ = self.dataset.process_one(job)
            finally:
                from mol.utils.file_io import _JSON_FILE_CACHE
                for entity in job["sequences"]:
                    for chain in entity.values():
                        path = chain.get("templatesPath")
                        if path is not None:
                            _JSON_FILE_CACHE.pop(str(Path(path)), None)
            predicted = self.runner.predict(data)
            for sample, (summary, full) in enumerate(zip(predicted["summary_confidence"], predicted["full_data"], strict=True)):
                if f"{protocol}_sample{sample}" in completed_keys:
                    continue
                yield self._prediction(predicted["coordinate"][sample], summary, full, atoms, data,
                                       payload, protocol, sample, folder)

    def _prediction(self, coordinates, summary, full, atoms, data, payload, protocol, sample, folder):
        from mol.data.utils import save_structure_cif
        from src.design.rewards.af3_worker import _native_prediction, _token_residue_atom_names
        def plain(value):
            if hasattr(value, "detach"):
                return value.detach().float().cpu().numpy()
            return value
        native = atoms.copy()
        native.b_factor = np.asarray(plain(full["atom_plddt"])) * 100.0
        native_path = folder / f"complex_sample{sample}.mol1.cif"
        save_structure_cif(native, coordinates, str(native_path), data["entity_poly_type"], payload["name"], save_wo_unresolved=False)
        text = native_path.read_text()
        structure = SimpleNamespace(
            to_mmcif=lambda: text, atom_b_factor=native.b_factor,
            iter_atoms=lambda: iter([{"chain_id": str(native.chain_id[i]), "res_id": int(native.res_id[i]),
                                    "res_name": str(native.res_name[i]), "atom_name": str(native.atom_name[i]),
                                    "atom_element": str(native.element[i])} for i in range(len(native))]))
        atom_to_token = np.asarray(plain(full["atom_to_token_idx"]), dtype=int)
        n_token = int(atom_to_token.max()) + 1
        first = [int(np.flatnonzero(atom_to_token == i)[0]) for i in range(n_token)]
        metadata = {key: plain(value) for key, value in summary.items()}
        metadata.update(token_chain_ids=[str(native.chain_id[i]) for i in first],
                        token_res_ids=[int(native.res_id[i]) for i in first],
                        token_residue_atom_names=_token_residue_atom_names(
                            native.atom_name, n_token, atom_to_token_idx=atom_to_token),
                        chain_ids=list(dict.fromkeys(str(x) for x in native.chain_id)))
        result = _native_prediction(
            SimpleNamespace(predicted_structure=structure, metadata=metadata,
                            numerical_data={"full_pae": plain(full["token_pair_pae"]),
                                            "full_pde": plain(full["token_pair_pde"])}),
            worker=SimpleNamespace(method="mol1", _sdk={},
                                   _ccd_cache={("mol1", payload.get("user_ccd") or ""): _Mol1CCD()}),
            payload=payload, protocol=protocol,
            sample=sample, directory=folder,
            native_frame_mask=np.asarray(plain(full["token_has_frame"]), dtype=bool))
        native_path.unlink()
        result["confidence"]["ranking_score"] = float(metadata["ranking_score"])
        return result

    def close(self):
        self.runner = None
        self.cache = None
