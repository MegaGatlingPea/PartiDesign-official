"""Resident native Boltz-2 inference for standalone validation.

Searches belong to the shared validation MSA service. This module never invokes
the Boltz CLI, downloads assets, or starts an independent GPU scheduler.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from partidesign.paths import community_asset, community_source

ADAPTER_REVISION = "shared_context_exact_templates_v1"
AFFINITY_FIELDS = (
    "affinity_pred_value", "affinity_probability_binary",
    "affinity_pred_value1", "affinity_probability_binary1",
    "affinity_pred_value2", "affinity_probability_binary2",
)
_HOST_BATCH_FIELDS = frozenset({
    "all_coords", "all_resolved_mask", "crop_to_all_atom_map", "chain_symmetries",
    "amino_acids_symmetries", "ligand_symmetries", "record", "affinity_mw",
})


def _plain(value):
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return value


def _scalar(value, name):
    array = np.asarray(_plain(value))
    if array.size != 1:
        raise ValueError(f"Boltz-2 {name} must be one scalar")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"Boltz-2 {name} must be finite")
    if "probability" in name and not 0 <= result <= 1:
        raise ValueError(f"Boltz-2 {name} must be in [0, 1]")
    return result


def affinity_metrics(prediction):
    """Return six native, explicitly named metrics; never reinterpret units."""
    confidence = prediction.get("confidence", {})
    metadata = prediction.get("metadata", {})
    requested = bool(prediction.get("boltz2_affinity_requested", metadata.get("boltz2_affinity_requested", False)))
    keys = ["boltz2_" + name for name in AFFINITY_FIELDS]
    if not requested and not any(key in confidence for key in keys):
        return {}
    if not all(key in confidence for key in keys):
        raise ValueError("Boltz-2 requested affinity output is incomplete")
    return {key: _scalar(confidence[key], key) for key in keys}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _source_imports(repo):
    source = community_source("boltz", repo) / "src"
    if not (source / "boltz/__init__.py").is_file():
        raise ValueError(f"Boltz-2 source is missing: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import boltz
    if not Path(boltz.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError("A different Boltz package is already imported")


class _BoltzCCD:
    """Expose the resident native molecule dictionary to common restoration."""
    def __init__(self, molecules):
        self.molecules = molecules

    def get(self, name):
        molecule = self.molecules.get(name)
        if molecule is None:
            return None
        from rdkit import Chem
        atoms = list(molecule.GetAtoms())
        heavy = [atom for atom in atoms if atom.GetAtomicNum() > 1]
        names = {atom.GetIdx(): atom.GetProp("name") for atom in heavy}
        bonds = [bond for bond in molecule.GetBonds()
                 if bond.GetBeginAtomIdx() in names and bond.GetEndAtomIdx() in names]
        order = {Chem.BondType.SINGLE: "SING", Chem.BondType.DOUBLE: "DOUB",
                 Chem.BondType.TRIPLE: "TRIP", Chem.BondType.AROMATIC: "AROM"}
        return {
            "_chem_comp_atom.atom_id": [names[a.GetIdx()] for a in heavy],
            "_chem_comp_atom.type_symbol": [a.GetSymbol().upper() for a in heavy],
            "_chem_comp_atom.charge": [a.GetFormalCharge() for a in heavy],
            "_chem_comp_atom.pdbx_aromatic_flag": ["Y" if a.GetIsAromatic() else "N" for a in heavy],
            "_chem_comp_bond.atom_id_1": [names[b.GetBeginAtomIdx()] for b in bonds],
            "_chem_comp_bond.atom_id_2": [names[b.GetEndAtomIdx()] for b in bonds],
            "_chem_comp_bond.value_order": [order[b.GetBondType()] for b in bonds],
            "_chem_comp_bond.pdbx_aromatic_flag": ["Y" if b.GetIsAromatic() else "N" for b in bonds],
        }


def bind_native_chemistry(payload, target):
    """Prove SMILES identity, then map native atom names to source atom UIDs.

    ChEMBL is allowed to standardize its input internally, but a changed graph,
    charge or supplied stereochemistry cannot become another validation input.
    Only the private canonical payload is changed; source ledgers stay intact.
    """
    from rdkit import Chem
    result = deepcopy(payload)
    if payload.get("sequence_request") is not None:
        result["native_ligand_chemistry"] = {}
        chains = {str(chain["name"]): chain for chain in target.structure.chains}
        for entry in result["chains"]:
            if entry["modality"] != "ligand":
                continue
            chain = chains[entry["id"]]
            residues = target.structure.residues[int(chain["res_idx"]):int(chain["res_idx"]) + int(chain["res_num"])]
            if len(residues) != 1:
                raise ValueError("Boltz-2 sequence ligand must remain one exact native component")
            component = str(residues[0]["name"])
            authority = Chem.RemoveHs(Chem.MolFromSmiles(entry["smiles"]))
            molecule = target.extra_mols.get(component)
            if molecule is None:
                raise ValueError("Boltz-2 sequence ligand native molecule is missing")
            molecule = Chem.RemoveHs(molecule)
            identity = Chem.MolToSmiles(authority, isomericSmiles=True)
            if identity != Chem.MolToSmiles(molecule, isomericSmiles=True):
                raise ValueError("Boltz-2 standardization changed authoritative sequence ligand chemistry or stereochemistry")
            mapping = molecule.GetSubstructMatch(authority, useChirality=True)
            if (len(mapping) != authority.GetNumAtoms() or len(mapping) != molecule.GetNumAtoms()
                    or authority.GetNumBonds() != molecule.GetNumBonds()):
                raise ValueError("Boltz-2 sequence ligand heavy graph is not bijective")
            entry["ccd_ids"] = [component]
            result["native_ligand_chemistry"][entry["id"]] = {
                "component_id": component, "canonical_isomeric_smiles": identity,
                "smiles_atom_to_native_name": [molecule.GetAtomWithIdx(i).GetProp("name") for i in mapping]}
        return result
    chains = {str(chain["name"]): chain for chain in target.structure.chains}
    for entry in result["chains"]:
        if entry["modality"] != "ligand":
            continue
        chain = chains[str(entry["id"])]
        start = int(chain["res_idx"])
        residues = target.structure.residues[start:start + int(chain["res_num"])]
        if len(residues) != 1:
            raise ValueError("Boltz-2 ligand validation requires one component per ligand chain")
        component = str(residues[0]["name"])
        source_names = {str(row["res_name"]) for row in result["source_atom_ledger"]
                        if str(row["chain_id"]) == str(entry["source_id"])}
        if len(source_names) != 1:
            raise ValueError("Boltz-2 ligand source component is ambiguous")
        source_name = next(iter(source_names))
        result.setdefault("native_component_map", {})[source_name] = component
        entry["ccd_ids"] = [component]
        if not entry.get("smiles"):
            continue
        authority = Chem.MolFromSmiles(entry["smiles"])
        native = target.extra_mols.get(component)
        if authority is None or native is None:
            raise ValueError("Boltz-2 authoritative ligand molecule is missing")
        authority = Chem.RemoveHs(authority)
        if (Chem.MolToSmiles(authority, isomericSmiles=True)
                != Chem.MolToSmiles(native, isomericSmiles=True)):
            raise ValueError("Boltz-2 standardization changed authoritative ligand chemistry or stereochemistry")
        if authority.GetNumAtoms() != native.GetNumAtoms() or authority.GetNumBonds() != native.GetNumBonds():
            raise ValueError("Boltz-2 ligand heavy graph changed")
        mapping = native.GetSubstructMatch(authority, useChirality=True)
        if len(mapping) != authority.GetNumAtoms():
            raise ValueError("Boltz-2 ligand has no exact stereochemical atom mapping")
        rows = entry.get("smiles_atom_map", ())
        if len(rows) != authority.GetNumAtoms():
            raise ValueError("Boltz-2 ligand source atom mapping is incomplete")
        updated = []
        for row in rows:
            index = int(row["compiler_atom_index"])
            if not 0 <= index < len(mapping):
                raise ValueError("Boltz-2 source SMILES atom index is invalid")
            atom = native.GetAtomWithIdx(mapping[index])
            updated.append({**row, "native_atom_name": atom.GetProp("name")})
        entry["smiles_atom_map"] = updated
        for residue in result["source_residue_map"][entry["id"]]:
            if residue is not None:
                residue["native_component_id"] = component
    return result


def _native_structure_view(path):
    from biotite.structure.io import pdbx
    text = Path(path).read_text()
    atom = next(iter(pdbx.CIFFile.deserialize(text).values()))["atom_site"]
    def column(name):
        return atom[name].as_array().tolist()
    names, chains, residues, components, elements = (column(name) for name in (
        "label_atom_id", "label_asym_id", "auth_seq_id", "label_comp_id", "type_symbol"))
    atoms = [dict(chain_id=str(chain), res_id=int(residue), res_name=str(component),
                  atom_name=str(name), atom_element=str(element))
             for name, chain, residue, component, element in zip(names, chains, residues, components, elements, strict=True)]
    return SimpleNamespace(to_mmcif=lambda: text, iter_atoms=lambda: iter(atoms),
                           atom_b_factor=np.asarray(column("B_iso_or_equiv"), dtype=float))


class Boltz2Backend:
    """One pair of native checkpoint instances per existing validation worker."""
    def __init__(self, repo_root, config, device=None):
        self.repo = Path(repo_root)
        self.config = deepcopy(config)
        self.device = str(device or "cuda:0")
        self.log_dir = None
        self.models = {}
        self.cache = None
        self.sdk = None
        self._msa_cache = {}
        self._sequence_msa_cache = {}
        from .pocket import SourcePocketCache
        self.pocket_cache = SourcePocketCache()
        self._closed = False

    def set_log_dir(self, directory):
        self.log_dir = Path(directory)

    def _load_data(self):
        if self.sdk is not None:
            return
        _source_imports(self.repo)
        from partidesign.boltz_assets import ensure_boltz_mols
        from boltz.data.mol import load_canonicals
        from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
        from boltz.data.feature.featurizerv2 import Boltz2Featurizer
        root = community_asset("boltz", self.repo)
        moldir = ensure_boltz_mols(root)
        self.sdk = dict(root=root, moldir=moldir, ccd=load_canonicals(moldir),
                        tokenizer=Boltz2Tokenizer(), featurizer=Boltz2Featurizer())

    def _load(self, directory=None, *, kinds=None):
        """Load strict native checkpoints once; supports CPU-only asset checks."""
        self._load_data()
        for kind in (kinds if kinds is not None else (["confidence", "affinity"] if self.config["affinity"] else ["confidence"])):
            if kind not in self.models:
                self.models[kind] = self._load_model(kind)

    def _load_model(self, kind):
        import torch
        from boltz.main import Boltz2DiffusionParams, BoltzSteeringParams, PairformerArgsV2, MSAModuleArgs
        from boltz.model.models.boltz2 import Boltz2
        for name in ("CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"):
            os.environ.setdefault(name, "1")
        torch.set_float32_matmul_precision("highest")
        diffusion = Boltz2DiffusionParams()
        diffusion.step_scale = 1.5
        affinity = kind == "affinity"
        checkpoint = self.sdk["root"] / "checkpoint" / ("boltz2_aff.ckpt" if affinity else "boltz2_conf.ckpt")
        if not checkpoint.is_file():
            raise ValueError(f"Boltz-2 checkpoint is missing: {checkpoint}; run partidesign download boltz2")
        predict_args = dict(
            recycling_steps=5 if affinity else self.config["recycles"],
            sampling_steps=self.config["affinity_steps"] if affinity else self.config["steps"],
            diffusion_samples=self.config["affinity_samples"] if affinity else self.config["samples"],
            max_parallel_samples=1, write_confidence_summary=not affinity,
            write_full_pae=not affinity, write_full_pde=not affinity)
        steering = BoltzSteeringParams()
        if affinity:
            predict_args["keys_dict_out"] = ["iptm"]
            steering.fk_steering = False
            steering.physical_guidance_update = False
            steering.contact_guidance_update = False
        options = dict(strict=True, map_location="cpu", diffusion_process_args=asdict(diffusion),
                       ema=False, pairformer_args=asdict(PairformerArgsV2()),
                       msa_args=asdict(MSAModuleArgs(subsample_msa=False, num_subsampled_msa=1024, use_paired_feature=True)),
                       predict_args=predict_args, steering_args=asdict(steering))
        if affinity:
            # Upstream affinity loading retains the checkpoint's kernel setting.
            options["affinity_mw_correction"] = False
            if not self.device.startswith("cuda"):
                options["use_kernels"] = False
        else:
            options["use_kernels"] = self.device.startswith("cuda")
        model = Boltz2.load_from_checkpoint(checkpoint, **options)
        model.eval()
        model.to(self.device)
        model.setup("predict")
        return model

    def _parse(self, payload, directory, *, require_affinity=None):
        self._load_data()
        from .boltz2_msa import boltz2_job
        from boltz.data import const
        from boltz.data.parse.schema import parse_boltz_schema
        from boltz.data.parse.a3m import parse_a3m
        from boltz.data.types import Input
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        affinity_unavailable_reason = None
        try:
            affinity_job = boltz2_job(payload, directory, affinity=True)
        except ValueError as exc:
            required = self.config["affinity"] if require_affinity is None else require_affinity
            if required:
                raise
            affinity_job = None
            affinity_unavailable_reason = str(exc)
        # Eligible structure-only records still retain their own native pose.
        job = affinity_job or boltz2_job(payload, directory, affinity=False)
        if job.get("templates"):
            raise ValueError("Boltz-2 templates must use exact shared mapping features")
        name = str(payload["name"])
        if name in {"", ".", ".."} or Path(name).name != name:
            raise ValueError("Boltz-2 input name must be a file basename")
        required = self.config["affinity"] if require_affinity is None else require_affinity
        try:
            target = parse_boltz_schema(name, job, self.sdk["ccd"], self.sdk["moldir"], boltz_2=True)
            mapped = bind_native_chemistry(payload, target)
        except ValueError as exc:
            if required or affinity_job is None:
                raise
            # Affinity standardization may change an otherwise valid charged
            # structure input. Structure-only mode retains its exact chemistry.
            job = boltz2_job(payload, directory, affinity=False)
            target = parse_boltz_schema(name, job, self.sdk["ccd"], self.sdk["moldir"], boltz_2=True)
            mapped = bind_native_chemistry(payload, target)
            mapped["boltz2_affinity_unavailable_reason"] = str(exc)
        if affinity_job is None:
            mapped["boltz2_affinity_unavailable_reason"] = affinity_unavailable_reason
        msas = {}
        for chain in target.record.chains:
            if chain.msa_id == -1:
                continue
            if chain.msa_id == 0:
                raise ValueError("Boltz-2 auto MSA is forbidden; shared MSA or explicit query-only is required")
            path = Path(chain.msa_id)
            if path.suffix != ".a3m" or not path.is_file():
                raise ValueError(f"Boltz-2 shared A3M is missing: {path}")
            if str(path) not in self._msa_cache:
                self._msa_cache[str(path)] = parse_a3m(path, taxonomy=None, max_seqs=const.max_msa_seqs)
            msas[int(chain.chain_id)] = self._msa_cache[str(path)]
        structures = directory / "structures"
        structures.mkdir(exist_ok=True)
        target.structure.dump(structures / f"{target.record.id}.npz")
        target.record.dump(directory / "record.json")
        _write_json(directory / "job.json", job)
        native = Input(target.structure, msas, record=target.record,
                       residue_constraints=target.residue_constraints, templates=None, extra_mols=target.extra_mols)
        self._save_native_input(native, directory, structures)
        return target, native, mapped, structures

    @staticmethod
    def _save_native_input(native, directory, structures):
        """Persist real native MSA, constraints, molecules and conformers."""
        from rdkit import Chem
        msas = {}
        for chain, msa in native.msa.items():
            path = directory / f"native_msa_{chain}.npz"
            msa.dump(path)
            msas[str(chain)] = str(path)
        constraints_path = None
        if native.residue_constraints is not None:
            path = directory / "native_residue_constraints.npz"
            native.residue_constraints.dump(path)
            constraints_path = str(path)
        molecule_path = directory / "native_molecules.json"
        _write_json(molecule_path, {name: molecule.ToBinary(Chem.PropertyPickleOptions.AllProps).hex()
                                    for name, molecule in (native.extra_mols or {}).items()})
        _write_json(directory / "native_input.json", {
            "schema_version": 1, "record_path": str(directory / "record.json"),
            "structure_path": str(structures / f"{native.record.id}.npz"), "msa_paths": msas,
            "residue_constraints_path": constraints_path, "molecules_path": str(molecule_path)})

    def _saved_native_input(self, resume, pose_path):
        self._load_data()
        from rdkit import Chem
        from boltz.data.types import Input, Record, MSA, ResidueConstraints, StructureV2
        manifest_path = Path(resume.get("native_input_path", Path(resume["processed_dir"]) / "native_input.json"))
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError("Saved Boltz-2 native input schema is unsupported")
        paths = [manifest_path, Path(manifest["record_path"]), Path(manifest["structure_path"]),
                 Path(manifest["molecules_path"]), *(Path(path) for path in manifest["msa_paths"].values())]
        if manifest.get("residue_constraints_path"):
            paths.append(Path(manifest["residue_constraints_path"]))
        for path in paths:
            if not path.is_file() or any(item.is_symlink() for item in (path, *path.parents)):
                raise ValueError(f"Saved Boltz-2 native input is missing or symlinked: {path}")
        molecules = {name: Chem.Mol(bytes.fromhex(value)) for name, value in
                     json.loads(Path(manifest["molecules_path"]).read_text()).items()}
        record = Record.load(Path(manifest["record_path"]))
        original = StructureV2.load(Path(manifest["structure_path"]))
        pose = StructureV2.load(pose_path)
        for label, old, new in (("chains", original.chains, pose.chains),
                                ("residues", original.residues, pose.residues), ("atoms", original.atoms, pose.atoms)):
            if len(old) != len(new):
                raise ValueError(f"Saved Boltz-2 {label} inventory differs from affinity pose")
            for name in old.dtype.names:
                if name not in {"coords", "is_present"} and name in new.dtype.names and not np.array_equal(old[name], new[name]):
                    raise ValueError(f"Saved Boltz-2 {label}.{name} differs from affinity pose")
        msas = {int(chain): MSA.load(Path(path)) for chain, path in manifest["msa_paths"].items()}
        constraints = (None if not manifest.get("residue_constraints_path")
                       else ResidueConstraints.load(Path(manifest["residue_constraints_path"])))
        native = Input(pose, msas, record=record, residue_constraints=constraints, templates=None, extra_mols=molecules)
        return native, Path(manifest["structure_path"]).parent, paths

    def _features(self, native, payload, *, affinity=False):
        from .boltz2_msa import shared_template_features, input_evidence
        from boltz.data import const
        from boltz.data.crop.affinity import AffinityCropper
        from boltz.data.mol import load_molecules
        data = self.sdk["tokenizer"].tokenize(native)
        if affinity:
            data = AffinityCropper().crop(data, max_tokens=256, max_atoms=2048)
        molecules = {**self.sdk["ccd"], **native.extra_mols}
        missing = set(data.tokens["res_name"].tolist()) - molecules.keys()
        molecules.update(load_molecules(self.sdk["moldir"], missing))
        options = native.record.inference_options
        features = self.sdk["featurizer"].process(
            data, molecules=molecules, random=np.random.default_rng(42), training=False,
            max_atoms=None, max_tokens=None, max_seqs=const.max_msa_seqs,
            pad_to_max_seqs=False, single_sequence_prop=0.0, compute_frames=True,
            inference_pocket_constraints=None if options is None else options.pocket_constraints,
            inference_contact_constraints=None if options is None else options.contact_constraints,
            compute_constraint_features=True, override_method="other" if affinity else None,
            compute_affinity=affinity)
        template = shared_template_features(data, payload, self.sdk["ccd"], self.sdk["moldir"], max_tokens=None)
        if template is not None:
            features.update(template)
        evidence = input_evidence(data, features, payload)
        features["record"] = native.record
        return data, features, molecules, evidence

    def _forward(self, kind, features, *, seed):
        import torch
        from pytorch_lightning import seed_everything
        from boltz.data.module.inferencev2 import collate
        if kind not in self.models:
            self._load(kinds=(kind,))
        seed_everything(seed, workers=False)
        batch = collate([features])
        for key in batch:
            if key not in _HOST_BATCH_FIELDS:
                batch[key] = batch[key].to(self.device)
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda"))
        with torch.inference_mode(), autocast:
            prediction = self.models[kind].predict_step(batch, 0)
        if prediction.get("exception"):
            raise RuntimeError(f"Boltz-2 {kind} prediction failed (native OOM/exception)")
        return prediction, batch

    @staticmethod
    def _sample_prediction(prediction, sample):
        fields = {"coords", "plddt", "pae", "pde", "confidence_score", "complex_plddt",
                  "complex_iplddt", "complex_pde", "complex_ipde", "ptm", "iptm",
                  "ligand_iptm", "protein_iptm"}
        result = {"exception": False, "masks": prediction["masks"]}
        for key in fields & prediction.keys():
            result[key] = prediction[key][sample:sample + 1]
        if "pair_chains_iptm" in prediction:
            result["pair_chains_iptm"] = {
                left: {right: values[sample:sample + 1] for right, values in row.items()}
                for left, row in prediction["pair_chains_iptm"].items()}
        return result

    def _prediction(self, target, payload, data, features, molecules, folder, sample, native_path, summary):
        from src.design.rewards.af3_worker import _native_prediction
        chain_names = {int(chain["asym_id"]): str(chain["name"]) for chain in target.structure.chains}
        metadata = {"token_chain_ids": [chain_names[int(token["asym_id"])] for token in data.tokens],
                    "token_res_ids": [int(token["res_idx"]) + 1 for token in data.tokens],
                    "chain_ids": list(chain_names.values()),
                    "ptm": summary["ptm"], "iptm": summary["iptm"],
                    "confidence_score": summary["confidence_score"],
                    "chain_ptm": {chain_names[int(i)]: value for i, value in summary["chains_ptm"].items()},
                    "chain_pair_iptm": {chain_names[int(i)]: {chain_names[int(j)]: value for j, value in row.items()}
                                        for i, row in summary["pair_chains_iptm"].items()}}
        metadata.update({"boltz2_native_" + key: value for key, value in summary.items()
                         if isinstance(value, (int, float))})
        for name in ["pae", "pde"]:
            with np.load(native_path.parent / f"{name}_{target.record.id}_model_0.npz", allow_pickle=False) as arrays:
                if name == "pae":
                    pae = arrays[name].copy()
                else:
                    pde = arrays[name].copy()
        frames = np.asarray(_plain(features["frame_resolved_mask"]), dtype=bool)
        if frames.ndim == 2:
            frames = frames[0]
        if payload.get("sequence_request") is not None:
            from .sequence_metrics import native_sequence_prediction
            from src.design.rewards.af3_worker import _token_residue_atom_names
            structure = _native_structure_view(native_path)
            atoms = list(structure.iter_atoms())
            atom_to_token = np.full(len(atoms), -1, dtype=int)
            for i, token in enumerate(data.tokens):
                start, count = int(token["atom_idx"]), int(token["atom_num"])
                atom_to_token[start:start + count] = i
            metadata["token_residue_atom_names"] = _token_residue_atom_names(
                [atom["atom_name"] for atom in atoms], len(data.tokens), atom_to_token_idx=atom_to_token)
            metadata["native_ligand_chemistry"] = payload.get("native_ligand_chemistry", {})
            return native_sequence_prediction(
                SimpleNamespace(predicted_structure=structure, metadata=metadata,
                                numerical_data={"full_pae": pae, "full_pde": pde}),
                payload=payload, model="boltz2", sample=sample, directory=folder,
                atom_to_token=atom_to_token, native_frame_mask=frames)
        result = _native_prediction(
            SimpleNamespace(predicted_structure=_native_structure_view(native_path), metadata=metadata,
                            numerical_data={"full_pae": pae, "full_pde": pde}),
            worker=SimpleNamespace(method="boltz2", _sdk={},
                                   _ccd_cache={("boltz2", payload.get("user_ccd") or ""): _BoltzCCD(molecules)}),
            payload=payload, protocol="complex", sample=sample, directory=folder,
            native_frame_mask=frames)
        result["confidence"].update({"boltz2_native_" + key: value for key, value in summary.items()
                                     if isinstance(value, (int, float))})
        result["confidence"]["confidence_score"] = float(summary["confidence_score"])
        result["native_ranking_score"] = float(summary["confidence_score"])
        result["metadata"]["native_ranking_score"] = result["native_ranking_score"]
        return result

    def predict(self, input_record, variant, output, *, completed_keys=(), prepared_inputs=None):
        wanted = [i for i in range(self.config["samples"]) if f"complex_sample{i}" not in completed_keys]
        if not wanted:
            return
        from .boltz2_msa import prepare_boltz2_inputs
        from .af3_msa import AF3MSACache
        self._load_data()
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if prepared_inputs is None:
            self.cache = self.cache or AF3MSACache(self.repo, self.config["msa"])
            prepared_inputs = prepare_boltz2_inputs(input_record, variant, self.config, self.repo,
                                                   cache=self.cache, pocket_cache=self.pocket_cache)
        if set(prepared_inputs) != {"complex"}:
            raise ValueError("Boltz-2 validation currently supports complex protocol only")
        yield from self._predict_payload(prepared_inputs["complex"], output,
                                         completed_keys=completed_keys, affinity_requested=self.config["affinity"])

    def predict_sequences(self, request, output, *, completed_keys=()):
        """Native polymer/SMILES structure prediction; affinity is a later phase."""
        if self._closed:
            raise ValueError("Boltz-2 backend is closed")
        if all(f"complex_sample{i}" in completed_keys for i in range(self.config["samples"])):
            return
        from .sequence_inputs import prepare_sequence_payload
        payload = prepare_sequence_payload(request, self.repo, "boltz2", self.config,
                                           cache=self._sequence_msa_cache)
        yield from self._predict_payload(payload, output, completed_keys=completed_keys, affinity_requested=False)

    def _predict_payload(self, payload, output, *, completed_keys=(), affinity_requested=False):
        self._load_data()
        from boltz.data.write.writer import BoltzWriter
        if self._closed:
            raise ValueError("Boltz-2 backend is closed")
        output = Path(output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        wanted = [i for i in range(self.config["samples"]) if f"complex_sample{i}" not in completed_keys]
        if not wanted:
            return
        processed = output / ".boltz2" / "processed"
        target, native, payload, structures = self._parse(payload, processed, require_affinity=bool(affinity_requested))
        payload_path = processed / "payload.json"
        _write_json(payload_path, payload)
        data, features, molecules, evidence = self._features(native, payload)
        evidence.update(adapter_revision=ADAPTER_REVISION, phase="structure", seed=self.config["seed"],
                        recycling_steps=self.config["recycles"], sampling_steps=self.config["steps"],
                        diffusion_samples=self.config["samples"], max_parallel_samples=1)
        evidence_path = output / ".boltz2" / "structure_model_input.json"
        _write_json(evidence_path, evidence)
        prediction, batch = self._forward("confidence", features, seed=self.config["seed"])
        for sample in wanted:
            folder = output / f"complex_sample{sample}"
            native_folder = output / ".boltz2" / f"complex_sample{sample}"
            writer = BoltzWriter(structures, native_folder, output_format="mmcif", boltz2=True)
            selected = self._sample_prediction(prediction, sample)
            writer.write_on_batch_end(None, None, selected, [], batch, 0, 0)
            native_dir = native_folder / target.record.id
            native_path = native_dir / f"{target.record.id}_model_0.cif"
            summary = json.loads((native_dir / f"confidence_{target.record.id}_model_0.json").read_text())
            result = self._prediction(target, payload, data, features, molecules, folder, sample, native_path, summary)
            structure_path = native_dir / f"pre_affinity_{target.record.id}.npz"
            eligible = bool(target.record.affinity)
            if eligible and not structure_path.is_file():
                raise ValueError("Boltz-2 native writer did not publish this sample's affinity structure")
            result["boltz2_affinity_requested"] = False
            result["metadata"].update(boltz2_adapter_revision=ADAPTER_REVISION,
                                      boltz2_affinity_requested=False,
                                      boltz2_affinity_eligible=eligible,
                                      boltz2_affinity_unavailable_reason=payload.get("boltz2_affinity_unavailable_reason"),
                                      boltz2_affinity_resume={
                                          "schema_version": 1, "payload_path": str(payload_path),
                                          "native_input_path": str(processed / "native_input.json"),
                                          "processed_dir": str(processed), "record_id": target.record.id,
                                          "pre_affinity_structure_path": str(structure_path) if eligible else None,
                                          "native_output_dir": str(native_dir), "structure_sample": sample,
                                          "native_structure_path": str(native_path)},
                                      boltz2_model_input_evidence_path=str(evidence_path))
            self._persist_phase_metadata(result)
            if affinity_requested:
                result = self.add_affinity(result)
            yield result

    @staticmethod
    def _persist_phase_metadata(result):
        raw_path = Path(result["raw_confidence_path"])
        raw = json.loads(raw_path.read_text())
        raw.update({key: value for key, value in result["confidence"].items() if key.startswith("boltz2_")})
        raw["boltz2_affinity_requested"] = bool(result.get("boltz2_affinity_requested", False))
        raw["boltz2_phase_metadata"] = result["metadata"]
        _write_json(raw_path, raw)

    def add_affinity(self, prediction, output=None):
        """Enrich one saved structure using native affinity sampling and winner.

        The pre-affinity full structure is an input pose. The scored pose is
        the selected *internal cropped diffusion sample* saved separately.
        All required request/input artifacts are durable across owner restarts.
        """
        if self._closed:
            raise ValueError("Boltz-2 backend is closed")
        result = deepcopy(dict(prediction))
        metadata = result.setdefault("metadata", {})
        resume = metadata.get("boltz2_affinity_resume")
        if result.get("raw_confidence_path"):
            raw = json.loads(Path(result["raw_confidence_path"]).read_text())
            phase_metadata = raw.get("boltz2_phase_metadata", {})
            durable_resume = phase_metadata.get("boltz2_affinity_resume")
            if (isinstance(durable_resume, Mapping) and phase_metadata.get("key") == result.get("key")
                    and (resume is None or resume == durable_resume)):
                resume = durable_resume
                metadata.update(phase_metadata)
                result["boltz2_affinity_requested"] = bool(raw.get("boltz2_affinity_requested", False))
                result.setdefault("confidence", {}).update({key: value for key, value in raw.items()
                                                            if key.startswith("boltz2_") and key not in {
                                                                "boltz2_phase_metadata", "boltz2_affinity_requested"}})
        if not isinstance(resume, Mapping) or resume.get("schema_version") != 1:
            raise ValueError("Saved Boltz-2 prediction has no durable native affinity input")
        if not metadata.get("boltz2_affinity_eligible") or not resume.get("pre_affinity_structure_path"):
            raise ValueError("Saved Boltz-2 structure is not eligible for native ligand affinity")
        sample = int(resume["structure_sample"])
        seed = int(self.config["seed"]) + sample
        complete = (result.get("boltz2_affinity_requested")
                    and metadata.get("boltz2_affinity_diffusion_samples") == self.config["affinity_samples"]
                    and metadata.get("boltz2_affinity_sampling_steps") == self.config["affinity_steps"]
                    and metadata.get("boltz2_affinity_seed") == seed)
        required = [metadata.get(key, "") for key in (
            "boltz2_affinity_raw_path", "boltz2_affinity_samples_path", "boltz2_affinity_input_evidence_path")]
        required.extend(metadata.get("affinity_required_artifacts", ()))
        if complete and all(Path(str(path)).is_file() for path in required):
            affinity_metrics(result)
            return result
        payload_path, pose_path = Path(resume["payload_path"]), Path(resume["pre_affinity_structure_path"])
        for path in (payload_path, pose_path):
            if not path.is_file() or any(item.is_symlink() for item in (path, *path.parents)):
                raise ValueError(f"Saved Boltz-2 native affinity input is missing or symlinked: {path}")
        payload = json.loads(payload_path.read_text())
        if output is None:
            native_folder = Path(resume["native_output_dir"]).parent
        else:
            native_folder = Path(output).expanduser().resolve() / ".boltz2" / f"affinity_{result['key']}"
        native, structures, input_artifacts = self._saved_native_input(resume, pose_path)
        from boltz.data.write.writer import BoltzAffinityWriter
        if not native.record.affinity or native.record.id != resume["record_id"]:
            raise ValueError("Saved Boltz-2 native record does not identify the eligible affinity binder")
        payload = bind_native_chemistry(payload, SimpleNamespace(structure=native.structure, extra_mols=native.extra_mols))
        data, features, _, evidence = self._features(native, payload, affinity=True)
        native_dir = native_folder / native.record.id
        native_dir.mkdir(parents=True, exist_ok=True)
        evidence.update(adapter_revision=ADAPTER_REVISION, phase="affinity", sample=sample,
                        source_structure=str(pose_path), payload_path=str(payload_path), seed=seed,
                        diffusion_samples=self.config["affinity_samples"], sampling_steps=self.config["affinity_steps"],
                        recycling_steps=5, max_parallel_samples=1, affinity_mw_correction=False,
                        scored_pose_definition="native internal diffusion sample selected by descending iptm")
        evidence_path = native_dir / "affinity_model_input.json"
        _write_json(evidence_path, evidence)
        affinity, affinity_batch = self._forward("affinity", features, seed=seed)
        values = {name: _scalar(affinity[name], name) for name in AFFINITY_FIELDS}
        samples_path, selected_index, selected_iptm = self._save_affinity_samples(affinity, features, data, native_dir)
        BoltzAffinityWriter(structures, native_folder).write_on_batch_end(
            None, None, affinity, [], affinity_batch, 0, 0)
        result.setdefault("confidence", {}).update({"boltz2_" + name: value for name, value in values.items()})
        result["boltz2_affinity_requested"] = True
        metadata.update(boltz2_affinity_requested=True, boltz2_affinity_structure_sample=sample,
                        boltz2_affinity_seed=seed, boltz2_affinity_diffusion_samples=self.config["affinity_samples"],
                        boltz2_affinity_sampling_steps=self.config["affinity_steps"],
                        boltz2_affinity_raw_path=str(native_dir / f"affinity_{native.record.id}.json"),
                        boltz2_affinity_input_evidence_path=str(evidence_path),
                        boltz2_affinity_samples_path=str(samples_path),
                        boltz2_affinity_selected_sample_index=selected_index,
                        boltz2_affinity_selected_iptm=selected_iptm,
                        boltz2_affinity_scored_pose_path=str(samples_path),
                        boltz2_affinity_scored_pose_kind="internal_cropped_diffusion_sample",
                        boltz2_affinity_crop_mapping_path=str(native_dir / "affinity_crop_mapping.json"))
        metadata["affinity_required_artifacts"] = [str(path) for path in (
            samples_path, evidence_path, native_dir / "affinity_crop_mapping.json",
            native_dir / f"affinity_{native.record.id}.json", payload_path, pose_path,
            *input_artifacts)]
        affinity_metrics(result)
        self._persist_phase_metadata(result)
        return result

    @staticmethod
    def _save_affinity_samples(affinity, features, data, directory):
        """Retain the exact native winner and every internally sampled pose."""
        import torch
        if "coords" not in affinity or "iptm" not in affinity:
            raise ValueError("Native Boltz-2 affinity omitted internal sampled coordinates or iPTM evidence")
        iptm_tensor = affinity["iptm"]
        coords = np.asarray(_plain(affinity["coords"]))
        iptm = np.asarray(_plain(iptm_tensor)).reshape(-1)
        if (coords.ndim != 3 or coords.shape[0] != len(iptm) or coords.shape[-1] != 3
                or not len(iptm) or not np.isfinite(coords).all() or not np.isfinite(iptm).all()):
            raise ValueError("Native Boltz-2 affinity internal pose/iPTM axes are malformed")
        # Match the original model's exact torch.argsort selection, including
        # its native device/tie behavior; NumPy sorting is not substituted.
        selected = int(torch.argsort(iptm_tensor, descending=True)[0].item())
        arrays = {"coords": coords, "iptm": iptm, "selected_index": np.asarray(selected),
                  "selected_coords": coords[selected], "selected_iptm": np.asarray(iptm[selected])}
        for key in ("crop_to_all_atom_map", "atom_to_token", "atom_pad_mask", "token_pad_mask",
                    "asym_id", "residue_index", "mol_type", "ref_atom_name_chars", "ref_element"):
            if key in features:
                value = features[key]
                arrays[key] = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
                if arrays[key].dtype.kind == "O":
                    raise ValueError(f"Native affinity mapping {key} is not a numeric array")
        arrays["crop_token_index"] = np.asarray(data.tokens["token_idx"], dtype=int)
        arrays["crop_token_asym_id"] = np.asarray(data.tokens["asym_id"], dtype=int)
        arrays["crop_token_residue_index"] = np.asarray(data.tokens["res_idx"], dtype=int)
        arrays["crop_token_atom_index"] = np.asarray(data.tokens["atom_idx"], dtype=int)
        arrays["crop_token_atom_count"] = np.asarray(data.tokens["atom_num"], dtype=int)
        full_atom_indices = np.concatenate([
            np.arange(int(token["atom_idx"]), int(token["atom_idx"]) + int(token["atom_num"]), dtype=int)
            for token in data.tokens])
        atom_mask = np.asarray(arrays["atom_pad_mask"], dtype=bool)
        if atom_mask.ndim != 1 or atom_mask.shape != (coords.shape[1],) or int(atom_mask.sum()) != len(full_atom_indices):
            raise ValueError("Native affinity crop atom inventory differs from sampled coordinate axis")
        expected_tokens = np.repeat(np.arange(len(data.tokens)), data.tokens["atom_num"].astype(int))
        if not np.array_equal(np.argmax(arrays["atom_to_token"][atom_mask], axis=-1), expected_tokens):
            raise ValueError("Native affinity atom-to-token feature differs from crop token atom ordering")
        if "crop_to_all_atom_map" not in arrays:
            mapping = np.full(len(atom_mask), -1, dtype=int)
            mapping[atom_mask] = full_atom_indices
            arrays["crop_to_all_atom_map"] = mapping
        chain_names = {int(chain["asym_id"]): str(chain["name"]) for chain in data.structure.chains}
        ledger = []
        for token in data.tokens:
            start, count = int(token["atom_idx"]), int(token["atom_num"])
            for atom_index in range(start, start + count):
                atom = data.structure.atoms[atom_index]
                ledger.append({"full_atom_index": atom_index, "native_token_index": int(token["token_idx"]),
                               "chain_id": chain_names[int(token["asym_id"])],
                               "residue_index": int(token["res_idx"]) + 1, "atom_name": str(atom["name"])})
        path = Path(directory) / "affinity_samples.npz"
        np.savez_compressed(path, **arrays)
        _write_json(Path(directory) / "affinity_crop_mapping.json", {
            "sample_archive": str(path), "selected_index": selected, "selected_iptm": float(iptm[selected]),
            "selection": "torch.argsort(native_iptm, descending=True)[0]",
            "token_atom_ledger": ledger, "coordinate_axis": "native cropped feature atom order; use crop_to_all_atom_map"})
        return path, selected, float(iptm[selected])

    def close(self):
        self._closed = True
        self.models.clear()
        self.cache = None
        self.sdk = None
        self._msa_cache.clear()
        self._sequence_msa_cache.clear()
        self.pocket_cache = None
