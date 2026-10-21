"""Bounded real adapter execution; run each backend in a fresh process."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .paths import code_root, offline_site, resource_root, using_model_dir


def _backend_worker_environment() -> dict[str, str]:
    """Return the checker child environment with a closed import path."""
    environment = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        PIP_NO_INDEX="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(code_root().resolve()), str(offline_site().resolve()))
    )
    return environment


def _fixture(target: str, directory: Path):
    import numpy as np
    import biotite.structure as struc
    from src.design.antibody.scaffolds import load_scaffold, scaffold_path
    from src.design.utils.reader import read_structure
    from src.design.utils.writer import write_structure_atomic
    from src.validate.inputs import load_input
    if target == "nampnn":
        source = resource_root() / "configs" / "examples" / "structures" / "nuc_design" / "rna_backbone_motif.pdb"
        atoms = read_structure(source)
        chains = np.asarray(atoms.chain_id).astype(str)
        modality = np.asarray(atoms.mol_type).astype(str)
        choices = list(dict.fromkeys(chains[np.isin(modality, ["rna", "dna"])]))
        if not choices:
            raise ValueError("packaged NA smoke structure has no nucleic chain")
        design = choices[0]
        positions = list(dict.fromkeys(atoms.res_id[chains == design]))[:8]
        design_atoms = atoms[(chains == design) & np.isin(atoms.res_id, positions)].copy()
        # Keep the native 8-nt RNA as design and add an owner-local duplicate
        # as fixed nucleic context.  The shift keeps the two chains separated
        # while preserving the source residue/atom/bond schema on both chains.
        context = design_atoms.copy()
        context_chain = "C" if design != "C" else "D"
        context.chain_id[:] = context_chain
        context.label_asym_id[:] = context_chain
        context.auth_asym_id[:] = context_chain
        context.label_entity_id[:] = context_chain
        context.coord = np.asarray(context.coord, dtype=np.float32).copy()
        context.coord[:, 0] += 20.0
        from src.design.utils.atoms import assign_identity_uids
        atoms = struc.concatenate([design_atoms, context])
        assign_identity_uids(atoms)
    else:
        scaffold = load_scaffold(scaffold_path("adalimumab.6cr1"))
        atoms = read_structure(scaffold.cif_path)
        design = next(chain.source_chain for chain in scaffold.chains if chain.role == "VH")
        selected = np.zeros(len(atoms), dtype=bool)
        for chain in scaffold.chains:
            positions = chain.source_positions
            if target != "partiscore":
                positions = positions[:12]
            selected |= (np.asarray(atoms.chain_id).astype(str) == chain.source_chain) & np.isin(atoms.res_id, positions)
        atoms = atoms[selected]
        if target == "cyclicmpnn":
            atoms = atoms[np.asarray(atoms.chain_id).astype(str) == design]
            first = int(min(atoms.res_id))
            last = int(max(atoms.res_id))
            n = np.flatnonzero((atoms.res_id == first) & (atoms.atom_name == "N"))
            c = np.flatnonzero((atoms.res_id == last) & (atoms.atom_name == "C"))
            if len(n) != 1 or len(c) != 1:
                raise ValueError("cyclic smoke has ambiguous termini")
            if atoms.bonds is None:
                atoms.bonds = struc.BondList(len(atoms))
            atoms.bonds.add_bond(int(c[0]), int(n[0]), struc.BondType.SINGLE)
    path = directory / "input.cif"
    write_structure_atomic(atoms, path, entry_id="backend_smoke")
    row = {"filename": path.name, "design_chain_id": design,
           "design_modality": "rna" if target == "nampnn" else "protein",
           "fixed_restypes": "[]", "bond_constraints": "[]"}
    rows = np.flatnonzero(np.asarray(atoms.chain_id).astype(str) == design)
    if not len(rows):
        raise ValueError(f"smoke design chain has no atoms: {design}")
    first = int(rows[0])
    row["fixed_restypes"] = json.dumps([{"chain": design,
        "res_id": int(atoms.res_id[first]), "ins_code": str(atoms.ins_code[first]),
        "restype": str(atoms.res_name[first])}])
    return load_input(path, row, directory)


def _variants(record, source: str, directory: Path):
    import numpy as np
    from src.validate.sequences import SequenceGenerator
    generator = SequenceGenerator(code_root())
    try:
        values = list(generator.generate(record, [{"source": source, "count": 1}], directory))
    finally:
        generator.close()
    for value in values:
        if value.get("status") != "complete" or value.get("error"):
            raise RuntimeError(f"{source} sequence generation failed: {value.get('error')}")
    selected = next((value for value in values if value["source"] == source), None)
    if selected is None:
        raise ValueError(f"no {source} variant was executed")
    from src.validate.sequences import _initial_sequences
    original = _initial_sequences(record)
    sequences = selected.get("sequences", {})
    for chain, sequence in original.items():
        if chain not in sequences or len(sequences[chain]) != len(sequence):
            raise ValueError(f"MPNN changed chain/residue accounting: {chain}")
        if chain not in record.design_chains and sequences[chain] != sequence:
            raise ValueError(f"MPNN changed context sequence: {chain}")

    # Resolve each fixed-restype ledger entry to its actual 1-based sequence
    # position.  res_id is an author/source locator, not a Python index.
    atoms = record.atoms
    chain_values = np.asarray(
        getattr(atoms, "label_asym_id", getattr(atoms, "chain_id"))
    ).astype(str)
    residue_values = np.asarray(atoms.res_id, dtype=np.int64)
    ins_values = np.asarray(atoms.ins_code).astype(str)
    uid_values = np.asarray(atoms.residue_uid, dtype=np.int64)
    fixed_count = 0
    for chain, locators in record.fixed_positions.items():
        if chain not in original or chain not in sequences:
            raise ValueError(f"fixed residue chain is absent from sequence output: {chain}")
        residue_positions = {}
        uid_positions = {}
        for row_index in np.flatnonzero(chain_values == str(chain)).tolist():
            key = (int(residue_values[row_index]), str(ins_values[row_index]))
            if key not in residue_positions:
                residue_positions[key] = len(residue_positions) + 1
                uid_positions[int(uid_values[row_index])] = residue_positions[key]
        for locator in locators:
            position = None
            raw_uid = locator.get("residue_uid", locator.get("res_uid"))
            if raw_uid not in (None, "", ".", "?"):
                position = uid_positions.get(int(raw_uid))
            if position is None:
                raw_res_id = locator.get("res_id", locator.get("position"))
                raw_ins = str(locator.get("ins_code", ""))
                if raw_res_id not in (None, "", ".", "?"):
                    position = residue_positions.get((int(raw_res_id), raw_ins))
            if position is None or position < 1 or position > len(original[chain]):
                raise ValueError(f"fixed residue locator does not resolve: {chain}:{locator}")
            expected = original[chain][position - 1]
            observed = sequences[chain][position - 1]
            if observed != expected:
                raise ValueError(
                    f"MPNN changed fixed residue {chain}:{position}: "
                    f"expected {expected!r}, got {observed!r}"
                )
            fixed_count += 1

    context_chains = tuple(str(chain) for chain in record.context_chains)
    context_residue_count = sum(len(original.get(chain, "")) for chain in context_chains)
    selected["_smoke_fixed_count"] = fixed_count
    selected["_smoke_context_chain_count"] = len(context_chains)
    selected["_smoke_context_residue_count"] = context_residue_count
    return selected


def _packing(record, variant, directory: Path) -> str:
    import numpy as np
    from src.validate.sequences import _candidate_adapter
    from src.design.core.records import SearchCandidate
    from src.design.rewards.esmfold2 import plan_sequence_source
    from src.design.rewards.packing import pack_mpnn_candidates
    from src.design.utils.writer import write_structure_atomic
    adapted = _candidate_adapter(record, "smoke")
    candidate = SearchCandidate(**{field.name: getattr(adapted, field.name)
        for field in fields(SearchCandidate) if hasattr(adapted, field.name)})
    plan = plan_sequence_source(candidate, requested="ligandmpnn")
    plan = replace(plan, chain_sequences=dict(variant["sequences"]))
    prepared, failures = pack_mpnn_candidates([(candidate, plan)],
        reward_config={"sequence": {"packing": {"device": "cuda"}}},
        repo_root=code_root(), work_dir=directory)
    if failures or "smoke" not in prepared:
        raise RuntimeError(f"packing failed: {failures}")
    packed = prepared["smoke"].atoms
    original = candidate.atoms
    protected = (~np.asarray(original.is_design, dtype=bool)) | np.isin(original.atom_name, ["N", "CA", "C", "O"])
    by_uid = {int(uid): index for index, uid in enumerate(packed.atom_uid)}
    for index in np.flatnonzero(protected):
        uid = int(original.atom_uid[index])
        if uid not in by_uid or not np.array_equal(original.coord[index], packed.coord[by_uid[uid]]):
            raise ValueError("packing changed protected context/backbone coordinates")
    path = write_structure_atomic(packed, directory / "packed.cif", entry_id="packed_smoke")
    _structure(path)
    return "real side-chain packing; protected context/backbone unchanged"


def _structure(path: str | Path) -> None:
    import numpy as np
    from src.design.utils.reader import read_structure
    atoms = read_structure(path)
    if not len(atoms) or not np.isfinite(atoms.coord).all():
        raise ValueError(f"invalid predicted structure: {path}")


def _execute(target: str, root: Path, directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=False)
    if target == "chemistry":
        from .model_checks import load_target
        args = argparse.Namespace(target=target, model_dir=str(root), checkpoint=None)
        return load_target(args)
    if target == "partidesign":
        from .checking import _native_smoke
        args = argparse.Namespace(out=directory, model_dir=root)
        return _native_smoke(None, args)
    record = _fixture(target, directory)
    sources = {"proteinmpnn": "protein_mpnn", "solublempnn": "soluble_mpnn",
               "ligandmpnn": "ligand_mpnn", "cyclicmpnn": "cyclic_mpnn", "nampnn": "na_mpnn"}
    if target in sources:
        variant = _variants(record, sources[target], directory / "sequences")
        detail = (
            f"real {sources[target]} forward/sample; "
            f"fixed positions={variant.get('_smoke_fixed_count', 0)}, "
            f"context chains={variant.get('_smoke_context_chain_count', 0)} "
            f"({variant.get('_smoke_context_residue_count', 0)} residues); "
            "design length and identities checked"
        )
        if target == "ligandmpnn":
            detail += "; " + _packing(record, variant, directory / "packing")
        return detail
    if target == "partiscore":
        from src.scoring.api import PartiScorer
        scorer = PartiScorer.from_checkpoint(root / "ckpts" / "partiscore.ckpt",
            device="cpu", num_workers=0, batch_size=1)
        try:
            result = scorer.score_files([str(record.path)],
                design_chains=[list(record.design_chains)], scope="design_groups")[0]
            metrics = result.as_dict()
            if metrics.get("status") not in {"ok", "no_contact", "no_interface"}:
                raise ValueError(f"unexpected PartiScore result: {metrics}")
            if metrics.get("status") != "ok":
                raise ValueError("score smoke fixture did not execute an eligible contact score")
        finally:
            scorer.close()
        return "real CPU PartiScore forward on a Fab interface; finite contact score"
    variant = _variants(record, "gt", directory / "sequences")
    if target.startswith("esmfold2-"):
        from src.validate.esmfold2 import ESMFold2Backend
        backend = ESMFold2Backend(code_root(), {"mode": target.removeprefix("esmfold2-"),
            "samples": 1, "loops": 2, "steps": 10, "lm_dropout": 0.0})
    elif target == "af2":
        from src.validate.af2 import AF2Backend
        backend = AF2Backend(code_root(), {"models": [1], "samples": 1, "recycles": 1, "mode": "both"})
    elif target in {"af3", "af3-la"}:
        from .af3_assets import layout
        assets = layout(root, target)
        os.environ["PARTIDESIGN_AF3_CCD_DIR"] = str(assets.ccd_dir)
        from src.validate.af3 import AF3Backend
        backend = AF3Backend(
            code_root(),
            {
                "target": target,
                "variant": target,
                "checkpoint": str(assets.weight),
                "model_dir": str(assets.weight.parent),
                "ccd_dir": str(assets.ccd_dir),
                "samples": 1,
            },
            device="cuda:0", method=target,
        )
    else:
        raise ValueError(f"unknown smoke target: {target}")
    try:
        predictions = list(backend.predict(record, variant, directory / "refolds"))
    finally:
        backend.close()
    if not predictions:
        raise ValueError("backend returned no predictions")
    for prediction in predictions:
        if prediction.get("status") != "complete":
            raise RuntimeError(f"backend prediction failed: {prediction.get('error')}")
        _structure(prediction["structure_path"])
    return f"real bounded {target} inference; {len(predictions)} structures independently read back"


def run_backend_smoke(target: str, root: Path, out: Path | None) -> str:
    manager = tempfile.TemporaryDirectory(prefix="partidesign-backend-smoke-") if out is None else nullcontext(str(out.expanduser().resolve()))
    with manager as base:
        directory = Path(base) / target
        env = _backend_worker_environment()
        if target in {"af3", "af3-la"}:
            from .af3_assets import layout
            env["PARTIDESIGN_AF3_CCD_DIR"] = str(layout(root, target).ccd_dir)
        command = [sys.executable, "-m", "partidesign.backend_smoke", target,
                   "--model-dir", str(root), "--directory", str(directory)]
        result = subprocess.run(
            command, cwd=str(code_root()), env=env, capture_output=True, text=True
        )
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip()[-4000:])
        return result.stdout.strip().splitlines()[-1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with using_model_dir(args.model_dir):
            print(_execute(args.target, args.model_dir, args.directory))
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
