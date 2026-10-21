from __future__ import annotations

import copy
import io
import json
import math
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure
from biotite.structure.info import residue as residue_template

from ..config import SequenceConfig
from ..errors import AdapterError, ConfigError, StructureError
from ..model import CanonicalSample, ResidueRecord, SequenceVariant
from ..runtime import RuntimePaths


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AA1_TO_3 = {one: three for three, one in AA3_TO_1.items()}
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
CHAIN_ALIASES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_RESIDUE_SUFFIX_RE = re.compile(r"^(-?\d+)([A-Za-z]?)$")


@dataclass(frozen=True, slots=True)
class MPNNResidue:
    chain_id: str
    alias: str
    temporary_position: int
    canonical_position: int
    canonical_ins_code: str
    source_chain_id: str
    source_res_id: int
    source_ins_code: str
    aa: str | None


@dataclass(slots=True)
class MPNNInput:
    pdb_path: Path
    alias_by_chain: dict[str, str]
    chain_by_alias: dict[str, str]
    residues_by_chain: dict[str, list[MPNNResidue]]
    pdb_text: str | None = None
    alias_ledger: dict[str, Any] | None = None

    def protein_chain_ids(self, sample: CanonicalSample) -> list[str]:
        return [chain.chain_id for chain in sample.chains if chain.modality == "protein"]


@dataclass(slots=True)
class MPNNProvenancePlan:
    """Deterministic, model-free description of one MPNN invocation.

    Every managed input file and command is constructed before any upstream
    process starts.  The same object is materialized and executed by the real
    adapters and reused while decoding their outputs.
    """

    adapter: str
    work_dir: Path
    prepared: MPNNInput
    fixed_residues: tuple[MPNNResidue, ...]
    redesigned_residues: tuple[MPNNResidue, ...]
    argv: tuple[str, ...]
    analysis_argv: dict[str, tuple[str, ...]]
    scoring_argv: tuple[tuple[str, ...], ...]
    expected_files: dict[Path, str]

    @property
    def input_pdb_text(self) -> str | None:
        return self.prepared.pdb_text

    @property
    def commands(self) -> dict[str, Any]:
        return {
            "design": list(self.argv),
            "analysis": {
                name: list(argv) for name, argv in self.analysis_argv.items()
            },
            "scoring": [list(argv) for argv in self.scoring_argv],
        }

    def materialize(self) -> None:
        """Write exactly the files described by this plan, without running MPNN."""

        for path in self.expected_files:
            symlink = _managed_symlink_component(path, self.work_dir)
            if symlink is not None:
                raise AdapterError(
                    f"Refusing to materialize MPNN provenance through symlink: {symlink}"
                )
        for path, content in self.expected_files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)



PROTEINMPNN_OPTIONS = {
    "model_name",
    "ca_only",
    "use_soluble_model",
    "seed",
    "save_score",
    "save_probs",
    "score_only",
    "path_to_fasta",
    "conditional_probs_only",
    "conditional_probs_only_backbone",
    "unconditional_probs_only",
    "backbone_noise",
    "batch_size",
    "max_length",
    "sampling_temp",
    "fixed_residues",
    "redesigned_residues",
    "omit_aas",
    "omit_aa_per_residue",
    "bias_aa",
    "bias_by_residue",
    "pssm",
    "pssm_multi",
    "pssm_threshold",
    "pssm_log_odds_flag",
    "pssm_bias_flag",
    "tied_positions",
}


LIGANDMPNN_OPTIONS = {
    "model_type",
    "model_name",
    "seed",
    "batch_size",
    "temperature",
    "save_stats",
    "fixed_residues",
    "redesigned_residues",
    "bias_aa",
    "bias_aa_per_residue",
    "omit_aa",
    "omit_aa_per_residue",
    "symmetry_residues",
    "symmetry_weights",
    "homo_oligomer",
    "ligand_mpnn_use_atom_context",
    "ligand_mpnn_cutoff_for_score",
    "ligand_mpnn_use_side_chain_context",
    "transmembrane_buried",
    "transmembrane_interface",
    "global_transmembrane_label",
    "pack_side_chains",
    "number_of_packs_per_design",
    "sc_num_denoising_steps",
    "sc_num_samples",
    "repack_everything",
    "force_hetatm",
    "packed_suffix",
    "pack_with_ligand_context",
    "verbose",
    "zero_indexed",
    "use_sequence",
    "autoregressive_score",
    "single_aa_score",
    "score_samples",
}


def build_mpnn_provenance(
    sample: CanonicalSample,
    config: SequenceConfig,
    runtime: RuntimePaths,
    work_dir: Path,
) -> MPNNProvenancePlan:
    """Build a complete deterministic MPNN request without running a model."""

    if config.model == "proteinmpnn":
        return build_proteinmpnn_provenance(sample, config, runtime, work_dir)
    if config.model == "ligandmpnn":
        return build_ligandmpnn_provenance(sample, config, runtime, work_dir)
    raise ConfigError(
        "MPNN provenance can only be built for sequence.model=proteinmpnn or ligandmpnn"
    )


def build_proteinmpnn_provenance(
    sample: CanonicalSample,
    config: SequenceConfig,
    runtime: RuntimePaths,
    work_dir: Path,
) -> MPNNProvenancePlan:
    """Return exact ProteinMPNN files and commands, with no filesystem writes."""

    if config.model != "proteinmpnn":
        raise ConfigError("ProteinMPNN builder requires sequence.model=proteinmpnn")
    _reject_unknown_options(config.options, PROTEINMPNN_OPTIONS, "proteinmpnn")
    options = dict(config.options)
    ca_only = _as_bool(options.get("ca_only", False), "ca_only")
    work_dir = Path(work_dir).resolve()
    prepared = prepare_mpnn_input(
        sample,
        work_dir,
        ca_only=ca_only,
        materialize=False,
    )
    _validate_proteinmpnn_scalars(options)
    fixed, redesigned = _design_masks(sample, prepared, options)
    batch_size = _resolve_batch_size(config.count, options.get("batch_size"))
    temperatures = str(options.get("sampling_temp", "0.1")).split()
    if len(temperatures) != 1:
        raise ConfigError(
            "proteinmpnn sampling_temp must contain exactly one temperature so "
            "sequence.count remains exact"
        )

    output_dir = work_dir / "output"
    argv = [
        str(runtime.mpnn_python),
        str(runtime.proteinmpnn_source / "protein_mpnn_run.py"),
        "--pdb_path",
        str(prepared.pdb_path),
        "--pdb_path_chains",
        prepared.alias_by_chain[sample.design_chain.chain_id],
        "--out_folder",
        str(output_dir),
        "--path_to_model_weights",
        str(_proteinmpnn_weight_dir(runtime, options, ca_only)),
        "--num_seq_per_target",
        str(config.count),
        "--batch_size",
        str(batch_size),
        "--sampling_temp",
        temperatures[0],
        "--seed",
        str(options.get("seed", 101)),
        "--suppress_print",
        "1",
    ]
    if ca_only:
        argv.append("--ca_only")
    if _as_bool(options.get("use_soluble_model", False), "use_soluble_model"):
        if ca_only:
            raise ConfigError("ProteinMPNN ca_only and use_soluble_model cannot be combined")
        argv.append("--use_soluble_model")
    _append_values(
        argv,
        options,
        {
            "model_name": "--model_name",
            "save_score": "--save_score",
            "save_probs": "--save_probs",
            "backbone_noise": "--backbone_noise",
            "max_length": "--max_length",
            "pssm_multi": "--pssm_multi",
            "pssm_threshold": "--pssm_threshold",
            "pssm_log_odds_flag": "--pssm_log_odds_flag",
            "pssm_bias_flag": "--pssm_bias_flag",
        },
        bool_as_int={
            "save_score",
            "save_probs",
            "pssm_log_odds_flag",
            "pssm_bias_flag",
        },
    )
    expected_files = _prepared_input_files(prepared)
    _add_proteinmpnn_constraint_files(
        argv,
        sample,
        prepared,
        options,
        fixed,
        redesigned,
        work_dir,
        expected_files,
    )
    external_snapshot = _plan_path_to_fasta_snapshot(
        options,
        work_dir,
        expected_files,
    )
    analysis_argv = _build_proteinmpnn_analysis_argv(
        argv,
        options,
        work_dir,
        external_snapshot,
    )
    return MPNNProvenancePlan(
        adapter="proteinmpnn",
        work_dir=work_dir,
        prepared=prepared,
        fixed_residues=tuple(fixed),
        redesigned_residues=tuple(redesigned),
        argv=tuple(argv),
        analysis_argv={name: tuple(command) for name, command in analysis_argv.items()},
        scoring_argv=(),
        expected_files=expected_files,
    )


def build_ligandmpnn_provenance(
    sample: CanonicalSample,
    config: SequenceConfig,
    runtime: RuntimePaths,
    work_dir: Path,
) -> MPNNProvenancePlan:
    """Return exact LigandMPNN files and commands, with no filesystem writes."""

    if config.model != "ligandmpnn":
        raise ConfigError("LigandMPNN builder requires sequence.model=ligandmpnn")
    _reject_unknown_options(config.options, LIGANDMPNN_OPTIONS, "ligandmpnn")
    options = dict(config.options)
    work_dir = Path(work_dir).resolve()
    prepared = prepare_mpnn_input(
        sample,
        work_dir,
        ca_only=False,
        materialize=False,
    )
    _validate_ligandmpnn_scalars(options)
    fixed, redesigned = _design_masks(sample, prepared, options)
    batch_size = _resolve_batch_size(config.count, options.get("batch_size"))
    model_type = str(options.get("model_type", "ligand_mpnn"))
    checkpoint_flag, checkpoint = _ligandmpnn_checkpoint(runtime, model_type, options)
    output_dir = work_dir / "output"
    argv = [
        str(runtime.mpnn_python),
        str(runtime.ligandmpnn_source / "run.py"),
        "--pdb_path",
        str(prepared.pdb_path),
        "--out_folder",
        str(output_dir),
        "--chains_to_design",
        prepared.alias_by_chain[sample.design_chain.chain_id],
        "--model_type",
        model_type,
        checkpoint_flag,
        str(checkpoint),
        "--checkpoint_path_sc",
        str(runtime.ligandmpnn_root / "checkpoint" / "ligandmpnn_sc_v_32_002_16.pt"),
        "--batch_size",
        str(batch_size),
        "--number_of_batches",
        str(config.count // batch_size),
        "--seed",
        str(options.get("seed", 101)),
        "--fasta_seq_separation",
        ":",
    ]
    if fixed:
        argv.extend(["--fixed_residues", " ".join(_ligand_tokens(fixed))])
    if redesigned:
        argv.extend(["--redesigned_residues", " ".join(_ligand_tokens(redesigned))])
    expected_files = _prepared_input_files(prepared)
    _add_ligandmpnn_options(
        argv,
        sample,
        prepared,
        options,
        fixed,
        redesigned,
        work_dir,
        expected_files,
    )
    if sample.motif_atoms:
        requested = options.get("ligand_mpnn_use_side_chain_context", True)
        if not _as_bool(requested, "ligand_mpnn_use_side_chain_context"):
            raise ConfigError(
                "atomic motif requires ligand_mpnn_use_side_chain_context=true"
            )
        _set_or_replace(argv, "--ligand_mpnn_use_side_chain_context", "1")
    scoring_argv = _build_ligandmpnn_scoring_argv(
        sample,
        prepared,
        options,
        fixed,
        redesigned,
        argv,
        output_dir,
        runtime,
        work_dir,
        config.count,
    )
    return MPNNProvenancePlan(
        adapter="ligandmpnn",
        work_dir=work_dir,
        prepared=prepared,
        fixed_residues=tuple(fixed),
        redesigned_residues=tuple(redesigned),
        argv=tuple(argv),
        analysis_argv={},
        scoring_argv=tuple(tuple(command) for command in scoring_argv),
        expected_files=expected_files,
    )


def reconstruct_mpnn_native_variants(
    sample: CanonicalSample,
    config: SequenceConfig,
    runtime: RuntimePaths,
    work_dir: Path | None = None,
    *,
    plan: MPNNProvenancePlan | None = None,
) -> list[SequenceVariant]:
    """Rebuild variants and scores from completed native MPNN outputs.

    Decode native FASTA and numeric outputs using the supplied input plan.
    No saved input or command-log matching is performed.
    """

    return _reconstruct_mpnn_native_variants(
        sample,
        config,
        runtime,
        work_dir=work_dir,
        plan=plan,
    )


def _reconstruct_mpnn_native_variants(
    sample: CanonicalSample,
    config: SequenceConfig,
    runtime: RuntimePaths,
    *,
    work_dir: Path | None,
    plan: MPNNProvenancePlan | None,
) -> list[SequenceVariant]:
    resolved = _resolve_native_output_plan(
        sample, config, runtime, work_dir=work_dir, plan=plan
    )
    if resolved.adapter == "proteinmpnn":
        return _reconstruct_proteinmpnn_native_variants(sample, config, resolved)
    if resolved.adapter == "ligandmpnn":
        return _reconstruct_ligandmpnn_native_variants(
            sample, config, runtime, resolved
        )
    raise AdapterError(f"Unsupported MPNN provenance adapter: {resolved.adapter}")






def run_proteinmpnn(
    sample: CanonicalSample,
    config: SequenceConfig,
    runtime: RuntimePaths,
    work_dir: Path,
) -> list[SequenceVariant]:
    plan = build_proteinmpnn_provenance(sample, config, runtime, work_dir)
    work_dir = plan.work_dir
    plan.materialize()
    options = dict(config.options)
    argv = list(plan.argv)
    _run_command(argv, runtime.proteinmpnn_source, runtime, work_dir / "process.log")
    _run_proteinmpnn_auxiliary(
        argv,
        options,
        runtime,
        work_dir,
        plan.analysis_argv,
    )
    return _reconstruct_mpnn_native_variants(
        sample,
        config,
        runtime,
        work_dir=None,
        plan=plan,
    )


def run_ligandmpnn(
    sample: CanonicalSample,
    config: SequenceConfig,
    runtime: RuntimePaths,
    work_dir: Path,
) -> list[SequenceVariant]:
    plan = build_ligandmpnn_provenance(sample, config, runtime, work_dir)
    work_dir = plan.work_dir
    plan.materialize()
    options = dict(config.options)
    output_dir = work_dir / "output"
    argv = list(plan.argv)
    _run_command(argv, runtime.ligandmpnn_source, runtime, work_dir / "process.log")
    _run_ligandmpnn_scoring(
        sample,
        plan.prepared,
        options,
        list(plan.fixed_residues),
        list(plan.redesigned_residues),
        argv,
        output_dir,
        runtime,
        work_dir,
        config.count,
        plan.scoring_argv,
    )
    return _reconstruct_mpnn_native_variants(
        sample,
        config,
        runtime,
        work_dir=None,
        plan=plan,
    )


def _resolve_native_output_plan(
    sample: CanonicalSample,
    config: SequenceConfig,
    runtime: RuntimePaths,
    *,
    work_dir: Path | None,
    plan: MPNNProvenancePlan | None,
) -> MPNNProvenancePlan:
    if work_dir is None and plan is None:
        raise AdapterError("MPNN native reconstruction requires work_dir or plan")
    if work_dir is not None:
        requested = Path(work_dir).expanduser()
        if requested.is_symlink():
            raise AdapterError(
                f"Refusing MPNN native reconstruction through symlink work_dir: {requested}"
            )
        requested = requested.resolve()
        if plan is not None and requested != plan.work_dir:
            raise AdapterError(
                f"MPNN work_dir differs from supplied plan: {requested} != {plan.work_dir}"
            )
    else:
        assert plan is not None
        requested = plan.work_dir
    return plan if plan is not None else build_mpnn_provenance(sample, config, runtime, requested)


def _reconstruct_proteinmpnn_native_variants(
    sample: CanonicalSample,
    config: SequenceConfig,
    plan: MPNNProvenancePlan,
) -> list[SequenceVariant]:
    prepared = plan.prepared
    options = dict(config.options)
    output_dir = plan.work_dir / "output"
    design_artifacts = _proteinmpnn_design_artifacts(
        output_dir, options, config.count
    )
    analysis_artifacts = _proteinmpnn_analysis_artifacts(
        options, plan.work_dir, plan.analysis_argv
    )

    fasta = output_dir / "seqs" / f"{prepared.pdb_path.stem}.fa"
    generated = _read_fasta(fasta)[1:]
    if len(generated) != config.count:
        raise AdapterError(
            f"ProteinMPNN produced {len(generated)} generated sequences; expected "
            f"exactly {config.count}; see {plan.work_dir / 'process.log'}"
        )
    observed = prepared.residues_by_chain[sample.design_chain.chain_id]
    shared_artifacts = {
        "design": design_artifacts,
        "analysis": analysis_artifacts,
    }
    variants: list[SequenceVariant] = []
    for index, (header, sequence) in enumerate(generated):
        sequence = sequence.replace("/", "").replace(":", "")
        if len(sequence) != len(observed):
            raise AdapterError(
                f"ProteinMPNN sequence {index} has {len(sequence)} observed residues; "
                f"expected {len(observed)}"
            )
        design_sequence = _reconstruct_design_sequence(sample, observed, sequence, plan.fixed_residues)
        chain_sequences = {chain.chain_id: chain.sequence for chain in sample.chains}
        chain_sequences[sample.design_chain.chain_id] = design_sequence
        scores = _header_fields(header)
        scores["artifacts"] = copy.deepcopy(shared_artifacts)
        variants.append(
            SequenceVariant(
                variant_id=f"proteinmpnn_{index:03d}",
                source="proteinmpnn",
                design_chain=sample.design_chain.chain_id,
                design_sequence=design_sequence,
                chain_sequences=chain_sequences,
                scores=scores,
            )
        )
    _require_distinct_design_sequences(variants, config.count, "ProteinMPNN")
    _validate_proteinmpnn_header_scores(
        generated, design_artifacts, options, config.count, plan.work_dir
    )
    return variants


def _reconstruct_ligandmpnn_native_variants(
    sample: CanonicalSample,
    config: SequenceConfig,
    runtime: RuntimePaths,
    plan: MPNNProvenancePlan,
) -> list[SequenceVariant]:
    prepared = plan.prepared
    options = dict(config.options)
    output_dir = plan.work_dir / "output"
    packed_artifacts = _ligandmpnn_packed_artifacts(
        output_dir, prepared, options, config.count
    )
    stats_artifact = _ligandmpnn_stats_artifact(output_dir, prepared, options)
    score_artifacts = _ligandmpnn_scoring_artifacts(
        sample,
        prepared,
        options,
        runtime,
        plan.work_dir,
        config.count,
        plan.scoring_argv,
    )

    fasta = output_dir / "seqs" / f"{prepared.pdb_path.stem}.fa"
    generated = _read_fasta(fasta)[1:]
    if len(generated) != config.count:
        raise AdapterError(
            f"LigandMPNN produced {len(generated)} generated sequences; expected "
            f"exactly {config.count}; see {plan.work_dir / 'process.log'}"
        )
    protein_chain_ids = prepared.protein_chain_ids(sample)
    ordered_aliases = sorted(
        prepared.alias_by_chain[chain] for chain in protein_chain_ids
    )
    chain_for_segment = [prepared.chain_by_alias[alias] for alias in ordered_aliases]
    native_observed = {
        chain_id: "".join(
            residue.aa or "X" for residue in prepared.residues_by_chain[chain_id]
        )
        for chain_id in protein_chain_ids
    }
    variants: list[SequenceVariant] = []
    for index, (header, sequence) in enumerate(generated):
        header_scores = _header_fields(header)
        segments = sequence.split(":")
        if len(segments) != len(chain_for_segment):
            raise AdapterError(
                f"LigandMPNN sequence {index} has {len(segments)} protein-chain segments; "
                f"expected {len(chain_for_segment)}"
            )
        observed_by_chain = dict(zip(chain_for_segment, segments, strict=True))
        for chain_id, observed_sequence in observed_by_chain.items():
            expected_length = len(prepared.residues_by_chain[chain_id])
            if len(observed_sequence) != expected_length:
                raise AdapterError(
                    f"LigandMPNN chain {chain_id} has length {len(observed_sequence)}; "
                    f"expected {expected_length}"
                )
            if (
                chain_id != sample.design_chain.chain_id
                and observed_sequence != native_observed[chain_id]
            ):
                raise AdapterError(
                    f"LigandMPNN changed immutable context protein chain {chain_id}"
                )
        design_sequence = _reconstruct_design_sequence(
            sample,
            prepared.residues_by_chain[sample.design_chain.chain_id],
            observed_by_chain[sample.design_chain.chain_id],
            plan.fixed_residues,
        )
        chain_sequences = {chain.chain_id: chain.sequence for chain in sample.chains}
        chain_sequences[sample.design_chain.chain_id] = design_sequence
        if packed_artifacts:
            header_scores["packed_pdbs"] = packed_artifacts[index]
        if stats_artifact is not None:
            header_scores["stats_path"] = stats_artifact
        if score_artifacts:
            header_scores["score"] = copy.deepcopy(score_artifacts[index])
        variants.append(
            SequenceVariant(
                variant_id=f"ligandmpnn_{index:03d}",
                source="ligandmpnn",
                design_chain=sample.design_chain.chain_id,
                design_sequence=design_sequence,
                chain_sequences=chain_sequences,
                scores=header_scores,
            )
        )
    _require_distinct_design_sequences(variants, config.count, "LigandMPNN")
    for index, (header, _) in enumerate(generated):
        _validate_ligandmpnn_header(_header_fields(header), index, options)
    return variants




def _require_distinct_design_sequences(
    variants: list[SequenceVariant], expected_count: int, model: str
) -> None:
    distinct = {variant.design_sequence for variant in variants}
    if len(variants) != expected_count or len(distinct) != expected_count:
        raise AdapterError(
            f"{model} produced {len(distinct)} distinct design-chain sequences "
            f"for sequence.count={expected_count}; duplicate designs are forbidden"
        )


def prepare_mpnn_input(
    sample: CanonicalSample,
    work_dir: Path,
    *,
    ca_only: bool,
    materialize: bool = True,
) -> MPNNInput:
    if len(sample.chains) > len(CHAIN_ALIASES):
        raise StructureError(
            f"MPNN temporary PDB supports at most {len(CHAIN_ALIASES)} entities; "
            f"found {len(sample.chains)}"
        )
    alias_by_chain = {
        chain.chain_id: CHAIN_ALIASES[index] for index, chain in enumerate(sample.chains)
    }
    chain_by_alias = {alias: chain for chain, alias in alias_by_chain.items()}
    try:
        array = get_structure(
            CIFFile.read(sample.canonical_cif),
            model=1,
            use_author_fields=False,
        )
    except Exception as exc:
        raise StructureError(f"Cannot parse canonical CIF for MPNN: {exc}") from exc
    expected = set(alias_by_chain)
    mask = np.isin(array.chain_id, list(expected))
    array = array[mask].copy()
    original_chain_ids = np.asarray(array.chain_id).copy()
    present = set(str(value) for value in array.chain_id)
    missing = sorted(expected - present)
    if missing:
        raise StructureError(f"Canonical CIF lacks MPNN entities: {missing}")

    ledger = {
        (record.chain_id, record.position, record.ins_code): record
        for record in sample.residues
    }
    residues_by_chain: dict[str, list[MPNNResidue]] = {}
    residue_name_aliases: list[dict[str, Any]] = []
    keep_rows = np.ones(len(array), dtype=bool)
    fixed_design_positions = {
        (str(chain_id), int(position), str(ins_code))
        for chain_id, position, ins_code in sample.motif_residues
    }
    for chain in sample.chains:
        chain_id = chain.chain_id
        chain_array = array[original_chain_ids == chain_id]
        is_polymer = chain.modality in {"protein", "nuc"}
        residue_keys = sorted(
            {
                (
                    int(atom.res_id),
                    "" if is_polymer else str(atom.ins_code).strip(),
                )
                for atom in chain_array
            },
            key=lambda key: (int(key[0]), str(key[1])),
        )
        mapped: list[MPNNResidue] = []
        for temporary_position, (position, ins_code) in enumerate(residue_keys, start=1):
            record = ledger.get((chain_id, position, ins_code))
            if record is None:
                raise StructureError(
                    f"MPNN residue {chain_id}:{position}{ins_code} is missing from canonical ledger"
                )
            residue_mask = (original_chain_ids == chain_id) & (array.res_id == position)
            if not is_polymer:
                residue_mask &= array.ins_code == ins_code
            source_res_name = record.res_name.upper()
            if chain.modality != "protein":
                cif_res_names = {
                    str(value).upper() for value in array.res_name[residue_mask]
                }
                if len(cif_res_names) != 1:
                    raise StructureError(
                        f"MPNN residue {chain_id}:{position}{ins_code} maps to "
                        f"multiple CIF component IDs"
                    )
                source_res_name = next(iter(cif_res_names))
            aa = AA3_TO_1.get(record.res_name.upper()) if chain.modality == "protein" else None
            temporary_res_name = source_res_name
            if chain.modality == "protein":
                if not (1 <= record.position <= len(chain.sequence)):
                    raise StructureError(
                        f"Canonical protein position {chain_id}:{record.position} lies outside "
                        f"sequence length {len(chain.sequence)}"
                    )
                native = chain.sequence[record.position - 1]
                if aa is None:
                    if (
                        chain_id == sample.design_chain.chain_id
                        and (chain_id, int(record.position), "")
                        not in fixed_design_positions
                    ):
                        raise ConfigError(
                            f"MPNN does not support non-standard design-chain residue "
                            f"{record.res_name} at "
                            f"{record.source_chain_id}{record.source_res_id}"
                        )
                    temporary_res_name = AA1_TO_3.get(native, "")
                    if not temporary_res_name:
                        raise ConfigError(
                            "MPNN cannot represent non-standard context residue "
                            f"{record.res_name} at "
                            f"{record.source_chain_id}{record.source_res_id}: "
                            f"canonical parent {native!r} is not a standard amino acid"
                        )
                    aa = native
                    residue_name_aliases.append(
                        {
                            "chain": chain_id,
                            "canonical_position": record.position,
                            "source_res_name": source_res_name,
                            "temporary_res_name": temporary_res_name,
                        }
                    )
                if native != aa:
                    raise StructureError(
                        f"Canonical sequence/coordinate mismatch at {chain_id}:{record.position}: "
                        f"JSON={native}, CIF={aa}"
                    )
            elif not re.fullmatch(r"[A-Za-z0-9]{1,3}", temporary_res_name):
                # This name is private to the coordinate-only MPNN PDB.  The
                # public CIF/SMILES/CCD identity remains unchanged; LigandMPNN
                # consumes the atom names, elements and coordinates here.
                temporary_res_name = (
                    "LIG" if chain.modality == "ligand" else "NCL"
                )
                residue_name_aliases.append(
                        {
                            "chain": chain_id,
                            "canonical_position": record.position,
                            "source_res_name": source_res_name,
                            "temporary_res_name": temporary_res_name,
                        }
                    )
            mapped.append(
                MPNNResidue(
                    chain_id=chain_id,
                    alias=alias_by_chain[chain_id],
                    temporary_position=temporary_position,
                    canonical_position=record.position,
                    canonical_ins_code=record.ins_code,
                    source_chain_id=record.source_chain_id,
                    source_res_id=record.source_res_id,
                    source_ins_code=record.source_ins_code,
                    aa=aa,
                )
            )
            if temporary_res_name != source_res_name:
                array.res_name[residue_mask] = temporary_res_name
                if chain.modality == "protein":
                    array.hetero[residue_mask] = False
                    if record.res_name.upper() == "MSE" and temporary_res_name == "MET":
                        selenium = residue_mask & (array.atom_name == "SE")
                        array.atom_name[selenium] = "SD"
                        array.element[selenium] = "S"
                    try:
                        template = residue_template(temporary_res_name)
                    except Exception as exc:
                        raise ConfigError(
                            "MPNN cannot materialize canonical parent residue "
                            f"{temporary_res_name!r}"
                        ) from exc
                    allowed_atoms = {
                        str(atom_name)
                        for atom_name, element in zip(
                            template.atom_name,
                            template.element,
                            strict=True,
                        )
                        if str(element).strip().upper() not in {"H", "D", "T"}
                    }
                    for row in np.flatnonzero(residue_mask):
                        if str(array.atom_name[row]) not in allowed_atoms:
                            keep_rows[row] = False
            array.res_id[residue_mask] = temporary_position
            array.ins_code[residue_mask] = ""
        residues_by_chain[chain_id] = mapped
        array.chain_id[original_chain_ids == chain_id] = alias_by_chain[chain_id]

    if not bool(keep_rows.all()):
        array = array[keep_rows].copy()
        original_chain_ids = original_chain_ids[keep_rows]

    design_residues = residues_by_chain[sample.design_chain.chain_id]
    required_atoms = {"CA"} if ca_only else {"N", "CA", "C", "O"}
    source_records = {
        (record.chain_id, record.position, record.ins_code): record
        for record in sample.residues
    }
    for residue in design_residues:
        record = source_records[
            (residue.chain_id, residue.canonical_position, residue.canonical_ins_code)
        ]
        absent = sorted(required_atoms - set(record.atom_names))
        if absent:
            raise ConfigError(
                f"MPNN design residue {record.source_chain_id}{record.source_res_id} "
                f"lacks required backbone atoms {absent}"
            )
    canonical_positions = [residue.canonical_position for residue in design_residues]
    if len(canonical_positions) != len(set(canonical_positions)):
        raise ConfigError("MPNN design chain has ambiguous canonical residue positions")

    pdb_path = work_dir / "input_alias.pdb"
    pdb_file = PDBFile()
    pdb_file.set_structure(array)
    pdb_buffer = io.StringIO()
    pdb_file.write(pdb_buffer)
    pdb_text = pdb_buffer.getvalue()
    mapping = {
        "alias_by_chain": alias_by_chain,
        "residue_name_aliases": residue_name_aliases,
        "residues": {
            chain: [
                {
                    "temporary_position": residue.temporary_position,
                    "canonical_position": residue.canonical_position,
                    "source_chain": residue.source_chain_id,
                    "source_residue": residue.source_res_id,
                    "source_insertion": residue.source_ins_code,
                }
                for residue in residues
            ]
            for chain, residues in residues_by_chain.items()
        },
    }
    prepared = MPNNInput(
        pdb_path,
        alias_by_chain,
        chain_by_alias,
        residues_by_chain,
        pdb_text,
        mapping,
    )
    if materialize:
        work_dir.mkdir(parents=True, exist_ok=True)
        pdb_path.write_text(pdb_text)
    return prepared


def _design_masks(
    sample: CanonicalSample,
    prepared: MPNNInput,
    options: dict[str, Any],
) -> tuple[list[MPNNResidue], list[MPNNResidue]]:
    design_observed = prepared.residues_by_chain[sample.design_chain.chain_id]
    user_fixed = _resolve_residue_values(
        options.get("fixed_residues"), sample, prepared, "fixed_residues"
    )
    redesigned = _resolve_residue_values(
        options.get("redesigned_residues"), sample, prepared, "redesigned_residues"
    )
    motif_fixed: list[MPNNResidue] = []
    lookup = {
        (residue.canonical_position, residue.canonical_ins_code): residue
        for residue in design_observed
    }
    for chain_id, position, ins_code in sample.motif_residues:
        if chain_id != sample.design_chain.chain_id:
            raise ConfigError("motif contains a non-design-chain residue")
        try:
            motif_fixed.append(lookup[(position, ins_code)])
        except KeyError as exc:
            raise ConfigError(
                f"motif residue {chain_id}{position}{ins_code} has no MPNN coordinates"
            ) from exc
    fixed = _dedupe_residues([*user_fixed, *motif_fixed])
    if redesigned:
        redesigned_keys = {_residue_key(residue) for residue in redesigned}
        fixed_keys = {_residue_key(residue) for residue in fixed}
        conflict = redesigned_keys & fixed_keys
        if conflict:
            raise ConfigError(
                f"redesigned_residues conflicts with fixed/motif residues: {sorted(conflict)}"
            )
        fixed = _dedupe_residues(
            [
                residue
                for residue in design_observed
                if _residue_key(residue) not in redesigned_keys
            ]
            + fixed
        )
    return fixed, redesigned


def _add_proteinmpnn_constraint_files(
    argv: list[str],
    sample: CanonicalSample,
    prepared: MPNNInput,
    options: dict[str, Any],
    fixed: list[MPNNResidue],
    redesigned: list[MPNNResidue],
    work_dir: Path,
    expected_files: dict[Path, str] | None = None,
) -> None:
    name = prepared.pdb_path.stem
    alias = prepared.alias_by_chain[sample.design_chain.chain_id]
    effective_designed = _effective_designed(sample, prepared, fixed, redesigned)
    if fixed:
        payload = {name: {alias: sorted(res.temporary_position for res in fixed)}}
        argv.extend(
            [
                "--fixed_positions_jsonl",
                str(
                    _write_json(
                        work_dir / "fixed_positions.jsonl", payload, expected_files
                    )
                ),
            ]
        )
    omit = options.get("omit_aa_per_residue")
    if omit:
        grouped: dict[str, list[int]] = {}
        for token, amino_acids in _require_mapping(omit, "omit_aa_per_residue").items():
            residue = _resolve_one_residue(token, sample, prepared, "omit_aa_per_residue")
            amino_acids = str(amino_acids)
            _validate_amino_acids(amino_acids, "omit_aa_per_residue")
            _reject_all_standard_amino_acids(
                amino_acids, f"omit_aa_per_residue.{token}"
            )
            grouped.setdefault(amino_acids, []).append(residue.temporary_position)
        payload = {name: {alias: [[positions, aas] for aas, positions in grouped.items()]}}
        argv.extend(
            [
                "--omit_AA_jsonl",
                str(_write_json(work_dir / "omit_AA.jsonl", payload, expected_files)),
            ]
        )
    if "omit_aas" in options:
        amino_acids = "".join(sorted(set(str(options["omit_aas"]) + "X")))
        _validate_amino_acids(amino_acids, "omit_aas")
        _reject_all_standard_amino_acids(amino_acids, "omit_aas")
        argv.extend(["--omit_AAs", amino_acids])
    bias = options.get("bias_aa")
    if bias:
        payload = {
            str(key): _finite_float(value, f"bias_aa.{key}")
            for key, value in _require_mapping(bias, "bias_aa").items()
        }
        _validate_amino_acids("".join(payload), "bias_aa")
        argv.extend(
            [
                "--bias_AA_jsonl",
                str(_write_json(work_dir / "bias_AA.jsonl", payload, expected_files)),
            ]
        )
    per_residue_bias = options.get("bias_by_residue")
    if per_residue_bias:
        arrays = {
            prepared.alias_by_chain[chain.chain_id]: np.zeros(
                (len(prepared.residues_by_chain[chain.chain_id]), len(AA_ALPHABET)),
                dtype=float,
            )
            for chain in sample.chains
            if chain.modality == "protein"
        }
        for token, values in _require_mapping(per_residue_bias, "bias_by_residue").items():
            residue = _resolve_one_residue(token, sample, prepared, "bias_by_residue")
            for amino_acid, value in _require_mapping(values, f"bias_by_residue.{token}").items():
                _validate_amino_acids(str(amino_acid), "bias_by_residue")
                arrays[residue.alias][
                    residue.temporary_position - 1,
                    AA_ALPHABET.index(str(amino_acid)),
                ] = _finite_float(value, f"bias_by_residue.{token}.{amino_acid}")
        payload = {name: {chain: values.tolist() for chain, values in arrays.items()}}
        argv.extend(
            [
                "--bias_by_res_jsonl",
                str(
                    _write_json(
                        work_dir / "bias_by_res.jsonl", payload, expected_files
                    )
                ),
            ]
        )
    pssm = options.get("pssm")
    if pssm:
        remapped = _remap_proteinmpnn_pssm(pssm, sample, prepared, options)
        argv.extend(
            [
                "--pssm_jsonl",
                str(
                    _write_json(
                        work_dir / "pssm.jsonl",
                        {name: remapped},
                        expected_files,
                    )
                ),
            ]
        )
    elif _as_bool(options.get("pssm_bias_flag", False), "pssm_bias_flag") or _as_bool(
        options.get("pssm_log_odds_flag", False), "pssm_log_odds_flag"
    ):
        raise ConfigError("ProteinMPNN PSSM flags require sequence.options.pssm")
    tied = options.get("tied_positions")
    if tied:
        groups = _require_sequence(tied, "tied_positions")
        payload_groups: list[dict[str, list[int]]] = []
        used: set[tuple[str, int]] = set()
        for group in groups:
            residues = [
                _resolve_one_residue(token, sample, prepared, "tied_positions")
                for token in _require_sequence(group, "tied_positions group")
            ]
            _validate_coupled_group(
                residues,
                effective_designed,
                used,
                "tied_positions",
            )
            by_alias: dict[str, list[int]] = {}
            for residue in residues:
                by_alias.setdefault(residue.alias, []).append(residue.temporary_position)
            payload_groups.append(by_alias)
        argv.extend(
            [
                "--tied_positions_jsonl",
                str(
                    _write_json(
                        work_dir / "tied_positions.jsonl",
                        {name: payload_groups},
                        expected_files,
                    )
                ),
            ]
        )


def _add_ligandmpnn_options(
    argv: list[str],
    sample: CanonicalSample,
    prepared: MPNNInput,
    options: dict[str, Any],
    fixed: list[MPNNResidue],
    redesigned: list[MPNNResidue],
    work_dir: Path,
    expected_files: dict[Path, str] | None = None,
) -> None:
    effective_designed = _effective_designed(sample, prepared, fixed, redesigned)
    _append_values(
        argv,
        options,
        {
            "temperature": "--temperature",
            "save_stats": "--save_stats",
            "ligand_mpnn_use_atom_context": "--ligand_mpnn_use_atom_context",
            "ligand_mpnn_cutoff_for_score": "--ligand_mpnn_cutoff_for_score",
            "ligand_mpnn_use_side_chain_context": "--ligand_mpnn_use_side_chain_context",
            "global_transmembrane_label": "--global_transmembrane_label",
            "pack_side_chains": "--pack_side_chains",
            "number_of_packs_per_design": "--number_of_packs_per_design",
            "sc_num_denoising_steps": "--sc_num_denoising_steps",
            "sc_num_samples": "--sc_num_samples",
            "repack_everything": "--repack_everything",
            "force_hetatm": "--force_hetatm",
            "packed_suffix": "--packed_suffix",
            "pack_with_ligand_context": "--pack_with_ligand_context",
            "verbose": "--verbose",
        },
        bool_as_int={
            "save_stats",
            "ligand_mpnn_use_atom_context",
            "ligand_mpnn_use_side_chain_context",
            "pack_side_chains",
            "repack_everything",
            "force_hetatm",
            "pack_with_ligand_context",
            "verbose",
        },
    )
    if _as_bool(options.get("zero_indexed", False), "zero_indexed"):
        argv.extend(["--zero_indexed", "1"])
    for option, flag in (
        ("bias_aa", "--bias_AA"),
        ("omit_aa", "--omit_AA"),
    ):
        if option not in options:
            continue
        if option == "bias_aa":
            values = _require_mapping(options[option], option)
            _validate_amino_acids("".join(str(key) for key in values), option)
            text = ",".join(
                f"{key}:{_finite_float(value, f'{option}.{key}')}"
                for key, value in values.items()
            )
        else:
            text = str(options[option])
            _validate_amino_acids(text, option)
        argv.extend([flag, text])
    for option, flag in (
        ("bias_aa_per_residue", "--bias_AA_per_residue"),
        ("omit_aa_per_residue", "--omit_AA_per_residue"),
    ):
        if option not in options:
            continue
        remapped: dict[str, Any] = {}
        for token, values in _require_mapping(options[option], option).items():
            residue = _resolve_one_residue(token, sample, prepared, option)
            key = f"{residue.alias}{residue.temporary_position}"
            if option == "bias_aa_per_residue":
                mapping = _require_mapping(values, f"{option}.{token}")
                _validate_amino_acids("".join(str(aa) for aa in mapping), option)
                remapped[key] = {
                    str(aa): _finite_float(value, f"{option}.{token}.{aa}")
                    for aa, value in mapping.items()
                }
            else:
                _validate_amino_acids(str(values), option)
                _reject_all_standard_amino_acids(str(values), f"{option}.{token}")
                remapped[key] = str(values)
        path = _write_json(
            work_dir / f"{option}.json", remapped, expected_files
        )
        argv.extend([flag, str(path)])
    for option, flag in (
        ("transmembrane_buried", "--transmembrane_buried"),
        ("transmembrane_interface", "--transmembrane_interface"),
    ):
        if option in options:
            residues = _resolve_residue_values(options[option], sample, prepared, option)
            argv.extend([flag, " ".join(_ligand_tokens(residues))])
    symmetry = options.get("symmetry_residues")
    if symmetry:
        groups = []
        resolved_groups: list[list[MPNNResidue]] = []
        used: set[tuple[str, int]] = set()
        for group in _require_sequence(symmetry, "symmetry_residues"):
            residues = [
                _resolve_one_residue(token, sample, prepared, "symmetry_residues")
                for token in _require_sequence(group, "symmetry_residues group")
            ]
            _validate_coupled_group(
                residues,
                effective_designed,
                used,
                "symmetry_residues",
            )
            resolved_groups.append(residues)
            groups.append(",".join(_ligand_tokens(residues)))
        argv.extend(["--symmetry_residues", "|".join(groups)])
        if "symmetry_weights" in options:
            weight_groups = _require_sequence(options["symmetry_weights"], "symmetry_weights")
            if len(weight_groups) != len(groups):
                raise ConfigError("symmetry_weights must have one group per symmetry_residues group")
        else:
            weight_groups = [
                [1.0 / len(residues)] * len(residues) for residues in resolved_groups
            ]
        formatted = []
        for residues, weights in zip(resolved_groups, weight_groups, strict=True):
            weight_values = [
                _finite_float(value, "symmetry weight")
                for value in _require_sequence(weights, "symmetry weight group")
            ]
            if len(weight_values) != len(residues):
                raise ConfigError("each symmetry weight group must match its residue group")
            if any(value < 0.0 for value in weight_values) or not math.isclose(
                sum(weight_values), 1.0, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ConfigError(
                    "each symmetry weight group must be non-negative and sum to 1"
                )
            formatted.append(",".join(str(value) for value in weight_values))
        argv.extend(["--symmetry_weights", "|".join(formatted)])
    elif "symmetry_weights" in options:
        raise ConfigError("symmetry_weights requires symmetry_residues")


def _validate_proteinmpnn_scalars(options: dict[str, Any]) -> None:
    if "seed" in options:
        seed = _integer(options["seed"], "seed")
        if seed < 0:
            raise ConfigError("ProteinMPNN seed must be non-negative")
    if "model_name" in options:
        model_name = str(options["model_name"])
        if Path(model_name).name != model_name:
            raise ConfigError("ProteinMPNN model_name must be a model basename, not a path")
    temperature = _finite_float(options.get("sampling_temp", 0.1), "sampling_temp")
    if temperature <= 0.0:
        raise ConfigError("ProteinMPNN sampling_temp must be positive")
    if "backbone_noise" in options and _finite_float(
        options["backbone_noise"], "backbone_noise"
    ) < 0.0:
        raise ConfigError("ProteinMPNN backbone_noise must be non-negative")
    if "max_length" in options:
        _positive_int(options["max_length"], "max_length")
    for field in (
        "save_score",
        "save_probs",
        "score_only",
        "conditional_probs_only",
        "conditional_probs_only_backbone",
        "unconditional_probs_only",
        "pssm_log_odds_flag",
        "pssm_bias_flag",
    ):
        if field in options:
            _as_bool(options[field], field)
    if "path_to_fasta" in options and not _as_bool(
        options.get("score_only", False), "score_only"
    ):
        raise ConfigError("ProteinMPNN path_to_fasta requires score_only=true")
    pssm_multi = _finite_float(options.get("pssm_multi", 0.0), "pssm_multi")
    if not 0.0 <= pssm_multi <= 1.0:
        raise ConfigError("ProteinMPNN pssm_multi must be in [0, 1]")
    if pssm_multi > 0.0 and not _as_bool(
        options.get("pssm_bias_flag", False), "pssm_bias_flag"
    ):
        raise ConfigError("ProteinMPNN pssm_multi > 0 requires pssm_bias_flag=true")
    if "pssm_threshold" in options and not _as_bool(
        options.get("pssm_log_odds_flag", False), "pssm_log_odds_flag"
    ):
        raise ConfigError("ProteinMPNN pssm_threshold requires pssm_log_odds_flag=true")
    if "pssm_threshold" in options:
        _finite_float(options["pssm_threshold"], "pssm_threshold")


def _validate_ligandmpnn_scalars(options: dict[str, Any]) -> None:
    if "seed" in options:
        seed = _integer(options["seed"], "seed")
        if seed < 0:
            raise ConfigError("LigandMPNN seed must be non-negative")
    temperature = _finite_float(options.get("temperature", 0.1), "temperature")
    if temperature <= 0.0:
        raise ConfigError("LigandMPNN temperature must be positive")
    model_type = str(options.get("model_type", "ligand_mpnn"))
    if "homo_oligomer" in options:
        raise ConfigError(
            "LigandMPNN homo_oligomer is incompatible with PipeTools' exactly-one-design-chain contract; "
            "use explicit symmetry_residues within the design chain"
        )
    ligand_only = {
        "ligand_mpnn_use_atom_context",
        "ligand_mpnn_cutoff_for_score",
        "ligand_mpnn_use_side_chain_context",
    }
    wrong_ligand = sorted(ligand_only & options.keys()) if model_type != "ligand_mpnn" else []
    if wrong_ligand:
        raise ConfigError(f"{wrong_ligand} require LigandMPNN model_type=ligand_mpnn")
    per_residue_membrane = {"transmembrane_buried", "transmembrane_interface"}
    wrong_membrane = sorted(per_residue_membrane & options.keys())
    if wrong_membrane and model_type != "per_residue_label_membrane_mpnn":
        raise ConfigError(
            f"{wrong_membrane} require model_type=per_residue_label_membrane_mpnn"
        )
    if "global_transmembrane_label" in options:
        if model_type != "global_label_membrane_mpnn":
            raise ConfigError(
                "global_transmembrane_label requires model_type=global_label_membrane_mpnn"
            )
        if int(options["global_transmembrane_label"]) not in {0, 1}:
            raise ConfigError("global_transmembrane_label must be 0 or 1")
    if "ligand_mpnn_cutoff_for_score" in options and _finite_float(
        options["ligand_mpnn_cutoff_for_score"], "ligand_mpnn_cutoff_for_score"
    ) < 0.0:
        raise ConfigError("ligand_mpnn_cutoff_for_score must be non-negative")
    boolean_fields = {
        "save_stats",
        "ligand_mpnn_use_atom_context",
        "ligand_mpnn_use_side_chain_context",
        "pack_side_chains",
        "repack_everything",
        "force_hetatm",
        "pack_with_ligand_context",
        "verbose",
        "zero_indexed",
        "use_sequence",
        "autoregressive_score",
        "single_aa_score",
    }
    for field in boolean_fields & options.keys():
        _as_bool(options[field], field)
    autoregressive = _as_bool(
        options.get("autoregressive_score", False), "autoregressive_score"
    )
    single_aa = _as_bool(options.get("single_aa_score", False), "single_aa_score")
    if autoregressive and single_aa:
        raise ConfigError(
            "LigandMPNN autoregressive_score and single_aa_score are mutually exclusive"
        )
    scoring = autoregressive or single_aa
    if ({"use_sequence", "score_samples"} & options.keys()) and not scoring:
        raise ConfigError(
            "LigandMPNN use_sequence/score_samples require one score mode"
        )
    if scoring:
        _positive_int(options.get("score_samples", 1), "score_samples")
    packing_fields = {
        "number_of_packs_per_design",
        "sc_num_denoising_steps",
        "sc_num_samples",
        "repack_everything",
        "force_hetatm",
        "packed_suffix",
        "pack_with_ligand_context",
    }
    packing_enabled = _as_bool(options.get("pack_side_chains", False), "pack_side_chains")
    if packing_fields & options.keys() and not packing_enabled:
        raise ConfigError(
            f"{sorted(packing_fields & options.keys())} require pack_side_chains=true"
        )
    if packing_enabled:
        for field, default in (
            ("number_of_packs_per_design", 4),
            ("sc_num_denoising_steps", 3),
            ("sc_num_samples", 16),
        ):
            _positive_int(options.get(field, default), field)
        suffix = str(options.get("packed_suffix", "_packed"))
        if not suffix or Path(suffix).name != suffix or any(value in suffix for value in ("/", "\\")):
            raise ConfigError("packed_suffix must be a non-empty filename suffix")


def _effective_designed(
    sample: CanonicalSample,
    prepared: MPNNInput,
    fixed: list[MPNNResidue],
    redesigned: list[MPNNResidue],
) -> set[tuple[str, int]]:
    if redesigned:
        return {_residue_key(residue) for residue in redesigned}
    fixed_keys = {_residue_key(residue) for residue in fixed}
    return {
        _residue_key(residue)
        for residue in prepared.residues_by_chain[sample.design_chain.chain_id]
        if _residue_key(residue) not in fixed_keys
    }


def _validate_coupled_group(
    residues: list[MPNNResidue],
    effective_designed: set[tuple[str, int]],
    used: set[tuple[str, int]],
    field: str,
) -> None:
    keys = [_residue_key(residue) for residue in residues]
    if len(keys) < 2:
        raise ConfigError(f"each {field} group must contain at least two residues")
    if len(set(keys)) != len(keys):
        raise ConfigError(f"a {field} group contains a duplicate residue")
    overlap = set(keys) & used
    if overlap:
        raise ConfigError(f"{field} residue appears in more than one group: {sorted(overlap)}")
    invalid = set(keys) - effective_designed
    if invalid:
        raise ConfigError(
            f"{field} groups must contain only redesigned, non-motif residues; invalid={sorted(invalid)}"
        )
    used.update(keys)


def _build_proteinmpnn_analysis_argv(
    design_argv: list[str],
    options: dict[str, Any],
    work_dir: Path,
    path_to_fasta_snapshot: Path | None = None,
) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    modes = [
        ("score_only", "--score_only"),
        ("conditional_probs_only", "--conditional_probs_only"),
        ("conditional_probs_only_backbone", "--conditional_probs_only_backbone"),
        ("unconditional_probs_only", "--unconditional_probs_only"),
    ]
    for option, flag in modes:
        if not _as_bool(options.get(option, False), option):
            continue
        auxiliary = list(design_argv)
        output_index = auxiliary.index("--out_folder") + 1
        auxiliary[output_index] = str(work_dir / "analysis" / option)
        if option == "conditional_probs_only_backbone":
            auxiliary.extend(["--conditional_probs_only", "1", flag, "1"])
        else:
            auxiliary.extend([flag, "1"])
        if option == "score_only" and options.get("path_to_fasta"):
            fasta = path_to_fasta_snapshot
            if fasta is None:
                # Compatibility for direct private-helper callers.  Production
                # always supplies the builder-managed immutable snapshot.
                fasta = Path(str(options["path_to_fasta"])).expanduser().resolve()
            if not fasta.is_file():
                if path_to_fasta_snapshot is not None:
                    # The snapshot is expected but intentionally not written
                    # until plan.materialize().
                    pass
                else:
                    raise ConfigError(f"path_to_fasta does not exist: {fasta}")
            auxiliary.extend(["--path_to_fasta", str(fasta)])
        commands[option] = auxiliary
    return commands


def _run_proteinmpnn_auxiliary(
    design_argv: list[str],
    options: dict[str, Any],
    runtime: RuntimePaths,
    work_dir: Path,
    planned_commands: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, list[str]]:
    commands = (
        {name: list(argv) for name, argv in planned_commands.items()}
        if planned_commands is not None
        else _build_proteinmpnn_analysis_argv(design_argv, options, work_dir)
    )
    for option, auxiliary in commands.items():
        _run_command(
            auxiliary,
            runtime.proteinmpnn_source,
            runtime,
            work_dir / "analysis" / f"{option}.log",
        )
    return _proteinmpnn_analysis_artifacts(
        options,
        work_dir,
        {name: tuple(command) for name, command in commands.items()},
    )


def _proteinmpnn_analysis_artifacts(
    options: dict[str, Any],
    work_dir: Path,
    planned_commands: dict[str, tuple[str, ...]],
) -> dict[str, list[str]]:
    mode_flags = {
        "score_only": "--score_only",
        "conditional_probs_only": "--conditional_probs_only",
        "conditional_probs_only_backbone": "--conditional_probs_only_backbone",
        "unconditional_probs_only": "--unconditional_probs_only",
    }
    enabled = {
        name
        for name in mode_flags
        if _as_bool(options.get(name, False), name)
    }
    if set(planned_commands) != enabled:
        raise AdapterError(
            "ProteinMPNN analysis plan differs from enabled options: "
            f"planned={sorted(planned_commands)}, enabled={sorted(enabled)}"
        )
    analysis_root = work_dir / "analysis"
    result: dict[str, list[str]] = {}
    expected_npz: set[Path] = set()
    for mode in mode_flags:
        mode_root = analysis_root / mode
        paths = _artifact_glob(
            mode_root,
            "*.npz",
            work_dir,
            f"ProteinMPNN {mode}",
            recursive=True,
        )
        if mode not in enabled:
            if paths:
                raise AdapterError(
                    f"ProteinMPNN {mode} artifacts exist although the mode is disabled"
                )
            continue
        if not paths:
            raise AdapterError(
                f"ProteinMPNN {mode} completed without an NPZ artifact in {mode_root}"
            )
        for path in paths:
            _validate_npz(path, work_dir=work_dir)
        expected_npz.update(path.absolute() for path in paths)
        result[mode] = [str(path.resolve()) for path in paths]

    all_npz = {
        path.absolute()
        for path in _artifact_glob(
            analysis_root,
            "*.npz",
            work_dir,
            "ProteinMPNN analysis",
            recursive=True,
        )
    }
    if all_npz != expected_npz:
        extras = sorted(str(path) for path in all_npz - expected_npz)
        raise AdapterError(
            f"ProteinMPNN analysis contains unbound NPZ artifacts: {extras}"
        )
    return result


def _proteinmpnn_weight_dir(
    runtime: RuntimePaths, options: dict[str, Any], ca_only: bool
) -> Path:
    if ca_only:
        family = "ca"
    elif _as_bool(options.get("use_soluble_model", False), "use_soluble_model"):
        family = "soluble"
    else:
        family = "vanilla"
    path = runtime.proteinmpnn_root / "checkpoint" / family
    model_name = str(options.get("model_name", "v_48_020"))
    checkpoint = path / f"{model_name}.pt"
    if not checkpoint.is_file():
        raise AdapterError(f"ProteinMPNN checkpoint is missing: {checkpoint}")
    return path


def _proteinmpnn_design_artifacts(
    output_dir: Path, options: dict[str, Any], count: int
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for option, directory, required_key in (
        ("save_score", "scores", "score"),
        ("save_probs", "probs", "probs"),
    ):
        root = output_dir / directory
        paths = _artifact_glob(
            root,
            "*.npz",
            output_dir.parent,
            f"ProteinMPNN {option}",
        )
        enabled = _as_bool(options.get(option, False), option)
        if not enabled:
            if paths:
                raise AdapterError(
                    f"ProteinMPNN {option} artifacts exist although the option is disabled"
                )
            continue
        if len(paths) != 1:
            raise AdapterError(
                f"ProteinMPNN {option} expected one NPZ artifact, found {len(paths)}"
            )
        arrays = _validate_npz(paths[0], work_dir=output_dir.parent)
        if required_key not in arrays or arrays[required_key].ndim == 0:
            raise AdapterError(f"ProteinMPNN {option} artifact lacks {required_key}")
        if arrays[required_key].shape[0] != count:
            raise AdapterError(
                f"ProteinMPNN {option} artifact contains {arrays[required_key].shape[0]} "
                f"designs; expected {count}"
            )
        if option == "save_score":
            for key in ("score", "global_score"):
                if key not in arrays or arrays[key].shape != (count,):
                    raise AdapterError(
                        f"ProteinMPNN save_score artifact {key} must have shape ({count},)"
                    )
        else:
            probs = arrays["probs"]
            required = {"log_probs", "S", "mask", "chain_order"}
            missing = sorted(required - set(arrays))
            if missing:
                raise AdapterError(
                    f"ProteinMPNN save_probs artifact lacks keys: {missing}"
                )
            if probs.ndim != 3 or probs.shape[-1] != len(AA_ALPHABET):
                raise AdapterError(
                    "ProteinMPNN save_probs probabilities must have shape "
                    f"(designs, residues, {len(AA_ALPHABET)})"
                )
            if arrays["log_probs"].shape != probs.shape:
                raise AdapterError(
                    "ProteinMPNN save_probs log_probs shape differs from probs"
                )
            if arrays["S"].shape != probs.shape[:2]:
                raise AdapterError(
                    "ProteinMPNN save_probs sequence tensor shape differs from probs"
                )
            sums = np.sum(probs, axis=-1)
            if not np.allclose(sums, 1.0, rtol=1e-4, atol=1e-5):
                raise AdapterError(
                    "ProteinMPNN save_probs probabilities are not normalized"
                )
        result[option] = [str(paths[0].resolve())]
    return result


def _validate_proteinmpnn_header_scores(
    generated: list[tuple[str, str]],
    design_artifacts: dict[str, list[str]],
    options: dict[str, Any],
    count: int,
    work_dir: Path,
) -> None:
    headers = [_header_fields(header) for header, _ in generated]
    expected_temperature = float(options.get("sampling_temp", 0.1))
    for index, values in enumerate(headers):
        for key in ("T", "sample", "score", "global_score", "seq_recovery"):
            if type(values.get(key)) is not float or not math.isfinite(values[key]):
                raise AdapterError(
                    f"ProteinMPNN generated FASTA header {index} lacks finite numeric {key}"
                )
        if values["sample"] != float(index + 1):
            raise AdapterError(
                f"ProteinMPNN FASTA sample index {values['sample']} does not match {index + 1}"
            )
        if not math.isclose(values["T"], expected_temperature, rel_tol=0.0, abs_tol=1e-7):
            raise AdapterError(
                f"ProteinMPNN FASTA temperature {values['T']} differs from config "
                f"{expected_temperature}"
            )
        if not 0.0 <= values["seq_recovery"] <= 1.0:
            raise AdapterError(
                f"ProteinMPNN FASTA seq_recovery is outside [0, 1]: "
                f"{values['seq_recovery']}"
            )
    score_paths = design_artifacts.get("save_score", [])
    if not score_paths:
        return
    if len(score_paths) != 1:
        raise AdapterError("ProteinMPNN save_score manifest must contain one NPZ")
    arrays = _validate_npz(Path(score_paths[0]), work_dir=work_dir)
    for index in range(count):
        for header_key, array_key in (
            ("score", "score"),
            ("global_score", "global_score"),
        ):
            header_value = headers[index][header_key]
            native_value = float(arrays[array_key][index])
            if not math.isclose(
                header_value, native_value, rel_tol=0.0, abs_tol=5.1e-4
            ):
                raise AdapterError(
                    f"ProteinMPNN FASTA {header_key} for variant {index} "
                    f"differs from save_score NPZ: {header_value} != {native_value}"
                )


def _validate_ligandmpnn_header(
    values: dict[str, Any], index: int, options: dict[str, Any]
) -> None:
    for key in (
        "id",
        "T",
        "seed",
        "overall_confidence",
        "ligand_confidence",
        "seq_rec",
    ):
        if type(values.get(key)) is not float or not math.isfinite(values[key]):
            raise AdapterError(
                f"LigandMPNN generated FASTA header {index} lacks finite numeric {key}"
            )
    zero_indexed = _as_bool(options.get("zero_indexed", False), "zero_indexed")
    expected_id = index if zero_indexed else index + 1
    if values["id"] != float(expected_id):
        raise AdapterError(
            f"LigandMPNN FASTA id {values['id']} does not match {expected_id}"
        )
    expected_temperature = float(options.get("temperature", 0.1))
    if not math.isclose(values["T"], expected_temperature, rel_tol=0.0, abs_tol=1e-7):
        raise AdapterError(
            f"LigandMPNN FASTA temperature {values['T']} differs from config "
            f"{expected_temperature}"
        )
    configured_seed = int(options.get("seed", 101))
    if values["seed"] != float(configured_seed):
        raise AdapterError(
            f"LigandMPNN FASTA seed {values['seed']} differs from config {configured_seed}"
        )
    for key in ("overall_confidence", "ligand_confidence", "seq_rec"):
        if not 0.0 <= values[key] <= 1.0:
            raise AdapterError(
                f"LigandMPNN FASTA {key} is outside [0, 1]: {values[key]}"
            )


def _ligandmpnn_packed_artifacts(
    output_dir: Path,
    prepared: MPNNInput,
    options: dict[str, Any],
    count: int,
) -> list[list[str]]:
    packed_root = output_dir / "packed"
    disk_paths = _artifact_glob(
        packed_root,
        "*.pdb",
        output_dir.parent,
        "LigandMPNN packed PDB",
    )
    if not _as_bool(options.get("pack_side_chains", False), "pack_side_chains"):
        if disk_paths:
            raise AdapterError(
                "LigandMPNN packed PDB artifacts exist although packing is disabled"
            )
        return []
    pack_count = _positive_int(
        options.get("number_of_packs_per_design", 4),
        "number_of_packs_per_design",
    )
    suffix = str(options.get("packed_suffix", "_packed"))
    zero_indexed = _as_bool(options.get("zero_indexed", False), "zero_indexed")
    result: list[list[str]] = []
    expected_paths: list[Path] = []
    for index in range(count):
        output_index = index if zero_indexed else index + 1
        paths = [
            output_dir
            / "packed"
            / f"{prepared.pdb_path.stem}{suffix}_{output_index}_{pack_index}.pdb"
            for pack_index in range(1, pack_count + 1)
        ]
        for path in paths:
            _validate_pdb_artifact(
                path, output_dir.parent, "LigandMPNN packed PDB"
            )
        expected_paths.extend(paths)
        result.append([str(path.resolve()) for path in paths])
    if {path.absolute() for path in disk_paths} != {
        path.absolute() for path in expected_paths
    }:
        raise AdapterError(
            "LigandMPNN packed PDB set differs from the configured variant/pack layout"
        )
    return result


def _ligandmpnn_stats_artifact(
    output_dir: Path, prepared: MPNNInput, options: dict[str, Any]
) -> str | None:
    stats_root = output_dir / "stats"
    disk_paths = _artifact_glob(
        stats_root,
        "*.pt",
        output_dir.parent,
        "LigandMPNN stats",
    )
    if not _as_bool(options.get("save_stats", False), "save_stats"):
        if disk_paths:
            raise AdapterError(
                "LigandMPNN stats artifacts exist although save_stats is disabled"
            )
        return None
    path = output_dir / "stats" / f"{prepared.pdb_path.stem}.pt"
    _read_regular_artifact(
        path, output_dir.parent, "LigandMPNN stats artifact"
    )
    if {value.absolute() for value in disk_paths} != {path.absolute()}:
        raise AdapterError(
            "LigandMPNN stats PT set/count differs from the expected native artifact"
        )
    return str(path.resolve())


def _build_ligandmpnn_scoring_argv(
    sample: CanonicalSample,
    prepared: MPNNInput,
    options: dict[str, Any],
    fixed: list[MPNNResidue],
    redesigned: list[MPNNResidue],
    design_argv: list[str],
    output_dir: Path,
    runtime: RuntimePaths,
    work_dir: Path,
    count: int,
) -> list[list[str]]:
    autoregressive = _as_bool(
        options.get("autoregressive_score", False), "autoregressive_score"
    )
    single_aa = _as_bool(options.get("single_aa_score", False), "single_aa_score")
    if not (autoregressive or single_aa):
        return []
    model_type = str(options.get("model_type", "ligand_mpnn"))
    checkpoint_flag, checkpoint = _ligandmpnn_checkpoint(runtime, model_type, options)
    score_samples = _positive_int(options.get("score_samples", 1), "score_samples")
    use_sequence = _as_bool(options.get("use_sequence", True), "use_sequence")
    zero_indexed = _as_bool(options.get("zero_indexed", False), "zero_indexed")
    commands: list[list[str]] = []
    for index in range(count):
        output_index = index if zero_indexed else index + 1
        backbone = (
            output_dir
            / "backbones"
            / f"{prepared.pdb_path.stem}_{output_index}.pdb"
        )
        score_dir = work_dir / "analysis" / f"ligand_score_{index:03d}"
        argv = [
            str(runtime.mpnn_python),
            str(runtime.ligandmpnn_source / "score.py"),
            "--pdb_path",
            str(backbone),
            "--out_folder",
            str(score_dir),
            "--chains_to_design",
            prepared.alias_by_chain[sample.design_chain.chain_id],
            "--model_type",
            model_type,
            checkpoint_flag,
            str(checkpoint),
            "--seed",
            str(options.get("seed", 101)),
            "--batch_size",
            "1",
            "--number_of_batches",
            str(score_samples),
            "--use_sequence",
            str(int(use_sequence)),
            "--autoregressive_score",
            str(int(autoregressive)),
            "--single_aa_score",
            str(int(single_aa)),
        ]
        if fixed:
            argv.extend(["--fixed_residues", " ".join(_ligand_tokens(fixed))])
        if redesigned:
            argv.extend(["--redesigned_residues", " ".join(_ligand_tokens(redesigned))])
        for flag in (
            "--ligand_mpnn_use_atom_context",
            "--ligand_mpnn_cutoff_for_score",
            "--ligand_mpnn_use_side_chain_context",
            "--global_transmembrane_label",
            "--transmembrane_buried",
            "--transmembrane_interface",
            "--symmetry_residues",
            "--verbose",
        ):
            _copy_argv_value(design_argv, argv, flag)
        commands.append(argv)
    return commands


def _run_ligandmpnn_scoring(
    sample: CanonicalSample,
    prepared: MPNNInput,
    options: dict[str, Any],
    fixed: list[MPNNResidue],
    redesigned: list[MPNNResidue],
    design_argv: list[str],
    output_dir: Path,
    runtime: RuntimePaths,
    work_dir: Path,
    count: int,
    planned_commands: tuple[tuple[str, ...], ...] | None = None,
) -> None:
    commands = (
        [list(command) for command in planned_commands]
        if planned_commands is not None
        else _build_ligandmpnn_scoring_argv(
            sample,
            prepared,
            options,
            fixed,
            redesigned,
            design_argv,
            output_dir,
            runtime,
            work_dir,
            count,
        )
    )
    if not commands:
        return
    if len(commands) != count:
        raise AdapterError(
            f"LigandMPNN planned {len(commands)} scoring commands; expected {count}"
        )
    for argv in commands:
        backbone = Path(_require_argv_value(argv, "--pdb_path"))
        score_dir = Path(_require_argv_value(argv, "--out_folder"))
        _validate_pdb_artifact(
            backbone, work_dir, "LigandMPNN generated scoring backbone"
        )
        log_path = score_dir / "process.log"
        _run_command(argv, runtime.ligandmpnn_source, runtime, log_path)


def _ligandmpnn_scoring_artifacts(
    sample: CanonicalSample,
    prepared: MPNNInput,
    options: dict[str, Any],
    runtime: RuntimePaths,
    work_dir: Path,
    count: int,
    planned_commands: tuple[tuple[str, ...], ...],
) -> list[dict[str, Any]]:
    """Read current native scoring outputs once, without log reconstruction."""
    artifacts = []
    expected_length = sum(len(prepared.residues_by_chain[chain.chain_id])
                          for chain in sample.chains if chain.modality == "protein")
    for command in planned_commands:
        argv = list(command)
        backbone = Path(_require_argv_value(argv, "--pdb_path"))
        directory = Path(_require_argv_value(argv, "--out_folder"))
        path = directory / f"{backbone.stem}.pt"
        if _managed_symlink_component(path, work_dir) is not None or not path.is_file():
            raise AdapterError(f"Missing or symlinked LigandMPNN score output: {path}")
        values = _read_ligand_score(path)
        shape = values["probs"].shape
        samples = int(_require_argv_value(argv, "--number_of_batches"))
        if shape != (samples, expected_length, len(AA_ALPHABET)):
            raise AdapterError(f"LigandMPNN score shape {shape} differs from requested dimensions")
        artifacts.append({
            "path": str(path.resolve()), "samples": samples, "length": expected_length,
            "alphabet": len(AA_ALPHABET),
            "mode": "autoregressive" if _as_bool(options.get("autoregressive_score", False), "autoregressive_score") else "single_aa",
            "use_sequence": _as_bool(options.get("use_sequence", True), "use_sequence"),
        })
    return artifacts


def _read_ligand_score(path: Path) -> dict[str, Any]:
    """Load native numeric NumPy outputs with PyTorch's restricted unpickler."""
    import torch
    from numpy._core import multiarray

    numeric_types = [type(np.dtype(name)) for name in
                     ("float16", "float32", "float64", "int32", "int64", "bool")]
    allowed = [np.ndarray, np.dtype, multiarray._reconstruct, multiarray.scalar,
               (multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
               (multiarray.scalar, "numpy.core.multiarray.scalar"), *numeric_types]
    try:
        with torch.serialization.safe_globals(allowed):
            values = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AdapterError(f"Cannot read LigandMPNN score output {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise AdapterError("LigandMPNN score output must be a dictionary")
    arrays = {}
    for name in ("probs", "log_probs", "logits"):
        value = values.get(name)
        if not isinstance(value, (np.ndarray, torch.Tensor)):
            raise AdapterError(f"LigandMPNN score output lacks numeric {name}")
        array = np.asarray(value)
        if array.ndim != 3 or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise AdapterError(f"LigandMPNN {name} must be a finite rank-3 array")
        arrays[name] = array
    if len({array.shape for array in arrays.values()}) != 1:
        raise AdapterError("LigandMPNN score array shapes differ")
    probs = arrays["probs"]
    if np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(-1), 1, atol=1e-4):
        raise AdapterError("LigandMPNN probabilities are not normalized")
    if not np.allclose(np.exp(arrays["log_probs"]), probs, rtol=1e-4, atol=1e-6):
        raise AdapterError("LigandMPNN probabilities differ from log probabilities")
    return arrays



def _artifact_glob(
    root: Path,
    pattern: str,
    work_dir: Path,
    label: str,
    *,
    recursive: bool = False,
) -> list[Path]:
    symlink = _managed_symlink_component(root, work_dir)
    if symlink is not None:
        raise AdapterError(f"Refusing {label} through symlink: {symlink}")
    if not root.exists():
        return []
    if not root.is_dir():
        raise AdapterError(f"{label} root is not a directory: {root}")
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    paths = sorted(iterator)
    for path in paths:
        symlink = _managed_symlink_component(path, work_dir)
        if symlink is not None:
            raise AdapterError(f"Refusing {label} through symlink: {symlink}")
    return paths


def _read_regular_artifact(path: Path, work_dir: Path, label: str) -> bytes:
    symlink = _managed_symlink_component(path, work_dir)
    if symlink is not None:
        raise AdapterError(f"Refusing {label} through symlink: {symlink}")
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise AdapterError(f"{label} is not a regular file: {path}")
        data = path.read_bytes()
        after = path.lstat()
    except FileNotFoundError as exc:
        raise AdapterError(f"{label} is missing: {path}") from exc
    except OSError as exc:
        raise AdapterError(f"Cannot read {label} {path}: {exc}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(data) != after.st_size:
        raise AdapterError(f"{label} changed while it was being read: {path}")
    if not data:
        raise AdapterError(f"{label} is empty: {path}")
    return data


def _validate_pdb_artifact(path: Path, work_dir: Path, label: str) -> None:
    data = _read_regular_artifact(path, work_dir, label)
    try:
        text = data.decode("utf-8")
        array = PDBFile.read(io.StringIO(text)).get_structure(model=1)
    except Exception as exc:
        raise AdapterError(f"Cannot parse {label} {path}: {exc}") from exc
    if len(array) == 0:
        raise AdapterError(f"{label} has no atoms: {path}")


def _copy_argv_value(source: list[str], destination: list[str], flag: str) -> None:
    if flag not in source:
        return
    index = source.index(flag)
    if index + 1 >= len(source):
        raise AdapterError(f"Internal MPNN argv flag has no value: {flag}")
    destination.extend([flag, source[index + 1]])


def _require_argv_value(argv: list[str], flag: str) -> str:
    if argv.count(flag) != 1:
        raise AdapterError(f"Internal MPNN argv must contain {flag} exactly once")
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise AdapterError(f"Internal MPNN argv flag has no value: {flag}")
    return argv[index + 1]


def _validate_npz(
    path: Path, *, work_dir: Path | None = None
) -> dict[str, np.ndarray]:
    if work_dir is None:
        if path.is_symlink():
            raise AdapterError(
                f"Refusing ProteinMPNN NPZ artifact through symlink: {path}"
            )
        work_dir = path.parent
    data = _read_regular_artifact(path, work_dir, "ProteinMPNN NPZ artifact")
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except Exception as exc:
        raise AdapterError(f"Cannot parse ProteinMPNN NPZ artifact {path}: {exc}") from exc
    if not arrays:
        raise AdapterError(f"ProteinMPNN NPZ artifact is empty: {path}")
    for key, array in arrays.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise AdapterError(f"ProteinMPNN NPZ {path} contains non-finite {key}")
    return arrays


def _ligandmpnn_checkpoint(
    runtime: RuntimePaths, model_type: str, options: dict[str, Any]
) -> tuple[str, Path]:
    defaults = {
        "protein_mpnn": ("--checkpoint_protein_mpnn", "proteinmpnn_v_48_020.pt"),
        "ligand_mpnn": ("--checkpoint_ligand_mpnn", "ligandmpnn_v_32_010_25.pt"),
        "soluble_mpnn": ("--checkpoint_soluble_mpnn", "solublempnn_v_48_020.pt"),
        "per_residue_label_membrane_mpnn": (
            "--checkpoint_per_residue_label_membrane_mpnn",
            "per_residue_label_membrane_mpnn_v_48_020.pt",
        ),
        "global_label_membrane_mpnn": (
            "--checkpoint_global_label_membrane_mpnn",
            "global_label_membrane_mpnn_v_48_020.pt",
        ),
    }
    try:
        flag, default_name = defaults[model_type]
    except KeyError as exc:
        raise ConfigError(f"Unsupported LigandMPNN model_type {model_type!r}") from exc
    filename = str(options.get("model_name", default_name))
    if Path(filename).name != filename:
        raise ConfigError("LigandMPNN model_name must be a checkpoint filename, not a path")
    checkpoint = runtime.ligandmpnn_root / "checkpoint" / filename
    if not checkpoint.is_file():
        raise AdapterError(f"LigandMPNN checkpoint is missing: {checkpoint}")
    return flag, checkpoint


def _resolve_residue_values(
    value: Any,
    sample: CanonicalSample,
    prepared: MPNNInput,
    field: str,
) -> list[MPNNResidue]:
    if value is None or value == "" or value == []:
        return []
    if isinstance(value, str):
        tokens: Iterable[Any] = value.replace(",", " ").split()
    else:
        tokens = _require_sequence(value, field)
    return _dedupe_residues(
        [_resolve_one_residue(token, sample, prepared, field) for token in tokens]
    )


def _resolve_one_residue(
    token: Any,
    sample: CanonicalSample,
    prepared: MPNNInput,
    field: str,
) -> MPNNResidue:
    text = str(token).strip()
    chain = sample.design_chain
    suffix = text
    explicit_chain = chain.chain_id if text.startswith(chain.chain_id) else None
    if explicit_chain is not None:
        suffix = text[len(chain.chain_id) :]
    if explicit_chain is None and not _RESIDUE_SUFFIX_RE.fullmatch(suffix):
        raise ConfigError(
            f"{field} residue {text!r} must target the single design chain "
            f"{chain.chain_id} (mmCIF label_asym_id)"
        )
    match = _RESIDUE_SUFFIX_RE.fullmatch(suffix)
    if match is None:
        raise ConfigError(f"Invalid residue selector {text!r} in {field}")
    res_id = int(match.group(1))
    ins_code = match.group(2)
    observed = prepared.residues_by_chain[chain.chain_id]
    source = [
        residue
        for residue in observed
        if residue.source_res_id == res_id and residue.source_ins_code == ins_code
    ]
    if len(source) == 1:
        return source[0]
    canonical = [
        residue
        for residue in observed
        if residue.canonical_position == res_id and residue.canonical_ins_code == ins_code
    ]
    matches = source or canonical
    if len(matches) != 1:
        raise ConfigError(f"{field} residue {text!r} did not resolve uniquely")
    return matches[0]


def _resolve_chain_name(
    value: str, sample: CanonicalSample, prepared: MPNNInput
) -> str:
    canonical = [chain.chain_id for chain in sample.chains if chain.chain_id == value]
    if len(canonical) != 1 or canonical[0] not in prepared.alias_by_chain:
        raise ConfigError(
            f"Chain {value!r} did not resolve uniquely by mmCIF label_asym_id for MPNN"
        )
    return canonical[0]


def _remap_proteinmpnn_pssm(
    value: Any,
    sample: CanonicalSample,
    prepared: MPNNInput,
    options: dict[str, Any],
) -> dict[str, dict[str, list[Any]]]:
    """Validate upstream PSSM arrays and remap full-sequence rows over gaps.

    Upstream ProteinMPNN consumes one coefficient vector and two ``L x 21``
    matrices per temporary PDB chain.  A task YAML may provide either the
    observed-coordinate length or the full canonical sequence length; the
    latter is projected through the source/canonical residue ledger.
    """
    result: dict[str, dict[str, list[Any]]] = {}
    use_bias = _as_bool(options.get("pssm_bias_flag", False), "pssm_bias_flag")
    use_log_odds = _as_bool(
        options.get("pssm_log_odds_flag", False), "pssm_log_odds_flag"
    )
    for chain_name, raw_values in _require_mapping(value, "pssm").items():
        canonical = _resolve_chain_name(str(chain_name), sample, prepared)
        if canonical != sample.design_chain.chain_id:
            raise ConfigError(
                "ProteinMPNN PSSM may target only the unique design chain "
                f"{sample.design_chain.chain_id}"
            )
        chain = next(chain for chain in sample.chains if chain.chain_id == canonical)
        if chain.modality != "protein":
            raise ConfigError(f"PSSM chain {chain_name!r} is not protein")
        values = _require_mapping(raw_values, f"pssm.{chain_name}")
        allowed = {"alphabet", "pssm_coef", "pssm_bias", "pssm_log_odds"}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ConfigError(f"Unknown pssm.{chain_name} keys: {unknown}")
        if use_bias and "pssm_bias" not in values:
            raise ConfigError(
                f"pssm.{chain_name}.pssm_bias is required when pssm_bias_flag=true"
            )
        if use_log_odds and "pssm_log_odds" not in values:
            raise ConfigError(
                f"pssm.{chain_name}.pssm_log_odds is required when "
                "pssm_log_odds_flag=true"
            )
        alphabet = str(values.get("alphabet", AA_ALPHABET))
        if len(alphabet) != len(AA_ALPHABET) or set(alphabet) != set(AA_ALPHABET):
            raise ConfigError(
                f"pssm.{chain_name}.alphabet must be a permutation of {AA_ALPHABET}"
            )
        column_order = [alphabet.index(amino_acid) for amino_acid in AA_ALPHABET]
        observed = prepared.residues_by_chain[canonical]
        observed_length = len(observed)
        canonical_length = len(chain.sequence)

        def row_indices(length: int, field: str) -> list[int]:
            if length == observed_length:
                return list(range(observed_length))
            if length == canonical_length:
                return [residue.canonical_position - 1 for residue in observed]
            raise ConfigError(
                f"pssm.{chain_name}.{field} has {length} rows; expected "
                f"{observed_length} observed or {canonical_length} canonical rows"
            )

        if "pssm_coef" in values:
            coef = np.asarray(values["pssm_coef"], dtype=float)
            if coef.ndim != 1:
                raise ConfigError(f"pssm.{chain_name}.pssm_coef must be one-dimensional")
            coef = coef[row_indices(len(coef), "pssm_coef")]
        else:
            coef = np.ones(observed_length, dtype=float)
        if np.any((coef < 0.0) | (coef > 1.0)):
            raise ConfigError(f"pssm.{chain_name}.pssm_coef values must be in [0, 1]")

        def matrix(field: str, default: float) -> np.ndarray:
            if field not in values:
                return np.full((observed_length, len(AA_ALPHABET)), default, dtype=float)
            array = np.asarray(values[field], dtype=float)
            if array.ndim != 2 or array.shape[1] != len(AA_ALPHABET):
                raise ConfigError(
                    f"pssm.{chain_name}.{field} must have shape L x {len(AA_ALPHABET)}"
                )
            array = array[row_indices(array.shape[0], field)]
            return array[:, column_order]

        bias = matrix("pssm_bias", 0.0)
        log_odds = matrix("pssm_log_odds", 10000.0)
        if not (
            np.isfinite(coef).all()
            and np.isfinite(bias).all()
            and np.isfinite(log_odds).all()
        ):
            raise ConfigError(f"pssm.{chain_name} contains non-finite values")
        if "pssm_bias" in values:
            if np.any(bias < 0.0) or not np.allclose(
                bias.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6
            ):
                raise ConfigError(
                    f"pssm.{chain_name}.pssm_bias rows must be non-negative probability "
                    "distributions summing to 1"
                )
        result[prepared.alias_by_chain[canonical]] = {
            "pssm_coef": coef.tolist(),
            "pssm_bias": bias.tolist(),
            "pssm_log_odds": log_odds.tolist(),
        }
    design_alias = prepared.alias_by_chain[sample.design_chain.chain_id]
    if set(result) != {design_alias}:
        raise ConfigError("pssm must contain exactly the unique design chain")
    return result


def _reconstruct_design_sequence(
    sample: CanonicalSample,
    residues: list[MPNNResidue],
    observed_sequence: str,
    fixed_residues: tuple[MPNNResidue, ...] = (),
) -> str:
    if len(residues) != len(observed_sequence):
        raise AdapterError("Cannot reconstruct design sequence from mismatched observed length")
    sequence = list(sample.design_chain.sequence)
    for residue, amino_acid in zip(residues, observed_sequence, strict=True):
        if amino_acid not in AA_ALPHABET[:-1]:
            raise AdapterError(
                f"MPNN returned unsupported amino acid {amino_acid!r} at "
                f"{residue.source_chain_id}{residue.source_res_id}"
            )
        if residue in fixed_residues and amino_acid != residue.aa:
            raise AdapterError(f"MPNN changed fixed residue {residue.source_chain_id}{residue.source_res_id}")
        sequence[residue.canonical_position - 1] = amino_acid
    return "".join(sequence)


def _read_fasta(path: Path, work_dir: Path | None = None) -> list[tuple[str, str]]:
    if work_dir is None:
        if path.parent.name == "seqs" and path.parent.parent.name == "output":
            work_dir = path.parent.parent.parent
        else:
            work_dir = path.parent
    data = _read_regular_artifact(path, work_dir, "MPNN native FASTA")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AdapterError(f"MPNN FASTA is not UTF-8: {path}") from exc
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence)))
            header = line[1:]
            sequence = []
        elif header is None:
            raise AdapterError(f"Invalid FASTA without header: {path}")
        else:
            sequence.append(line)
    if header is not None:
        records.append((header, "".join(sequence)))
    if not records:
        raise AdapterError(f"MPNN FASTA is empty: {path}")
    return records


def _header_fields(header: str) -> dict[str, Any]:
    values: dict[str, Any] = {"header": header}
    seen = {"header"}
    for part in header.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise AdapterError(f"MPNN FASTA header contains an empty field: {header}")
        if key in seen:
            raise AdapterError(
                f"MPNN FASTA header contains duplicate/reserved field {key!r}: {header}"
            )
        seen.add(key)
        try:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise AdapterError(
                    f"MPNN FASTA header contains non-finite {key}: {header}"
                )
            values[key] = numeric
        except ValueError:
            values[key] = value
    return values


def _run_command(
    argv: list[str], cwd: Path, runtime: RuntimePaths, log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=runtime.subprocess_env(block_network=True),
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        log_path.write_text(
            f"argv={json.dumps(argv, allow_nan=False)}\nexit_code=timeout\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n"
        )
        raise AdapterError(f"MPNN subprocess timed out after 3600s; see {log_path}") from exc
    log_path.write_text(
        f"argv={json.dumps(argv, allow_nan=False)}\n"
        f"exit_code={completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}\n"
    )
    if completed.returncode != 0:
        raise AdapterError(
            f"MPNN subprocess failed with exit {completed.returncode}; see {log_path}"
        )


def _prepared_input_files(prepared: MPNNInput) -> dict[Path, str]:
    if prepared.pdb_text is None:
        return {}
    return {prepared.pdb_path: prepared.pdb_text}


def _plan_path_to_fasta_snapshot(
    options: dict[str, Any],
    work_dir: Path,
    expected_files: dict[Path, str],
) -> Path | None:
    value = options.get("path_to_fasta")
    if not value:
        return None
    source = Path(str(value)).expanduser()
    if not source.is_absolute():
        source = Path.cwd() / source
    source = source.absolute()
    if not source.is_file():
        raise ConfigError(f"path_to_fasta does not exist: {source}")
    try:
        data = source.read_bytes()
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"path_to_fasta is not UTF-8 text: {source}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read path_to_fasta {source}: {exc}") from exc
    snapshot = work_dir / "managed_inputs" / "path_to_fasta.fasta"
    expected_files[snapshot] = text
    return snapshot


def _managed_symlink_component(path: Path, work_dir: Path) -> Path | None:
    try:
        relative = path.relative_to(work_dir)
    except ValueError as exc:
        raise AdapterError(
            f"Managed MPNN provenance path escapes work_dir: {path}"
        ) from exc
    current = work_dir
    if current.is_symlink():
        return current
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return current
    return None


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"


def _write_json(
    path: Path,
    payload: Any,
    expected_files: dict[Path, str] | None = None,
) -> Path:
    content = _compact_json(payload)
    if expected_files is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    else:
        expected_files[path] = content
    return path


def _append_values(
    argv: list[str],
    options: dict[str, Any],
    mapping: dict[str, str],
    *,
    bool_as_int: set[str] | None = None,
) -> None:
    bool_as_int = bool_as_int or set()
    for option, flag in mapping.items():
        if option not in options:
            continue
        value = options[option]
        if option in bool_as_int:
            value = int(_as_bool(value, option))
        argv.extend([flag, str(value)])


def _set_or_replace(argv: list[str], flag: str, value: str) -> None:
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv.extend([flag, value])


def _resolve_batch_size(count: int, value: Any | None) -> int:
    """Resolve an exact MPNN batch size without surplus sequences.

    The default uses one batch through 256 sequences. Above that, it selects
    the largest exact divisor strictly below 256, as required by LigandMPNN's
    batch_size * number_of_batches output contract. Explicit batch sizes
    remain available for deliberate memory tuning.
    """

    if count < 1:
        raise ConfigError("sequence.count must be positive")
    if value is None:
        if count <= 256:
            return count
        for candidate in range(255, 0, -1):
            if count % candidate == 0:
                return candidate
        raise AssertionError("batch size 1 must divide every positive count")
    try:
        batch_size = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("MPNN batch_size must be an integer") from exc
    if batch_size < 1 or batch_size > count or count % batch_size:
        raise ConfigError(
            f"MPNN batch_size={batch_size} must be positive, no greater than "
            f"sequence.count={count}, and divide it exactly"
        )
    return batch_size


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    raise ConfigError(f"{field} must be true/false")


def _reject_unknown_options(options: dict[str, Any], allowed: set[str], adapter: str) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ConfigError(f"Unknown {adapter} sequence.options keys: {unknown}")


def _require_mapping(value: Any, field: str) -> dict[Any, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _require_sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{field} must be a list")
    return list(value)


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ConfigError(f"{field} must be finite")
    return result


def _positive_int(value: Any, field: str) -> int:
    result = _integer(value, field)
    if result <= 0:
        raise ConfigError(f"{field} must be a positive integer")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be an integer") from exc
    if str(value).strip() not in {str(result), f"{result}.0"}:
        raise ConfigError(f"{field} must be an integer")
    return result


def _validate_amino_acids(value: str, field: str) -> None:
    invalid = sorted(set(value) - set(AA_ALPHABET))
    if invalid:
        raise ConfigError(f"{field} contains invalid amino acids: {invalid}")


def _reject_all_standard_amino_acids(value: str, field: str) -> None:
    standard = set(AA_ALPHABET[:-1])
    if standard.issubset(set(value)):
        raise ConfigError(f"{field} cannot omit all 20 standard amino acids")


def _ligand_tokens(residues: list[MPNNResidue]) -> list[str]:
    return [f"{residue.alias}{residue.temporary_position}" for residue in residues]


def _residue_key(residue: MPNNResidue) -> tuple[str, int]:
    return residue.chain_id, residue.temporary_position


def _dedupe_residues(residues: list[MPNNResidue]) -> list[MPNNResidue]:
    return list({_residue_key(residue): residue for residue in residues}.values())
