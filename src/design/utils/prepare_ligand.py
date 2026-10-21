"""Prepare a topology-rich ligand CIF from a SMILES string."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from biotite import structure as struc
from rdkit import Chem
from rdkit.Chem import AllChem

from .writer import write_cif


_RDKIT_TO_BIOTITE = {
    Chem.BondType.SINGLE: struc.BondType.SINGLE,
    Chem.BondType.DOUBLE: struc.BondType.DOUBLE,
    Chem.BondType.TRIPLE: struc.BondType.TRIPLE,
    Chem.BondType.AROMATIC: struc.BondType.AROMATIC,
}


def ligand_from_smiles(
    smiles: str,
    *,
    chain_id: str = "L",
    component_id: str = "LIG",
    seed: int = 2026,
) -> struc.AtomArray:
    """Return one finite heavy-atom ligand with concrete internal topology."""

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("SMILES could not be parsed")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("one ligand CIF must contain one connected SMILES component")
    molecule_h = Chem.AddHs(molecule)
    embed_params = AllChem.ETKDGv3()
    embed_params.randomSeed = int(seed)
    status = AllChem.EmbedMolecule(molecule_h, embed_params)
    if status != 0:
        raise ValueError("RDKit could not embed a finite ligand conformer")
    if AllChem.MMFFHasAllMoleculeParams(molecule_h):
        AllChem.MMFFOptimizeMolecule(molecule_h, maxIters=500)
    elif AllChem.UFFHasAllMoleculeParams(molecule_h):
        AllChem.UFFOptimizeMolecule(molecule_h, maxIters=500)
    heavy = Chem.RemoveHs(molecule_h)
    conformer = heavy.GetConformer()
    atom_array = struc.AtomArray(heavy.GetNumAtoms())
    coordinates = np.asarray(
        [list(conformer.GetAtomPosition(i)) for i in range(heavy.GetNumAtoms())],
        dtype=np.float32,
    )
    if not np.isfinite(coordinates).all():
        raise ValueError("embedded coordinates are not finite")
    atom_array.coord = coordinates
    atom_array.chain_id[:] = str(chain_id)
    atom_array.res_id[:] = 1
    atom_array.ins_code[:] = ""
    atom_array.res_name[:] = str(component_id)
    atom_array.hetero[:] = True

    counts: Counter[str] = Counter()
    names: list[str] = []
    elements: list[str] = []
    for atom in heavy.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] += 1
        names.append(f"{symbol.upper()}{counts[symbol]}")
        elements.append(symbol.upper())
    atom_array.atom_name[:] = names
    atom_array.element[:] = elements

    annotations = {
        "mol_type": np.full(len(atom_array), "ligand", dtype="U8"),
        "is_design": np.zeros(len(atom_array), dtype=bool),
        "coord_valid": np.ones(len(atom_array), dtype=bool),
        "component_uid": np.zeros(len(atom_array), dtype=np.int64),
        "entity_uid": np.zeros(len(atom_array), dtype=np.int64),
        "asym_uid": np.zeros(len(atom_array), dtype=np.int64),
        "molecule_uid": np.zeros(len(atom_array), dtype=np.int64),
        "residue_uid": np.zeros(len(atom_array), dtype=np.int64),
        "token_uid": np.arange(len(atom_array), dtype=np.int64),
        "atom_uid": np.arange(len(atom_array), dtype=np.int64),
        "label_asym_id": np.full(len(atom_array), str(chain_id), dtype="U128"),
        "auth_asym_id": np.full(len(atom_array), "", dtype="U128"),
        "auth_seq_id": np.full(len(atom_array), "", dtype="U128"),
        "auth_ins_code": np.full(len(atom_array), "", dtype="U128"),
    }
    for name, values in annotations.items():
        atom_array.set_annotation(name, values)

    bonds: list[list[int]] = []
    for bond in heavy.GetBonds():
        bond_type = _RDKIT_TO_BIOTITE.get(bond.GetBondType())
        if bond_type is None:
            raise ValueError(f"unsupported SMILES bond type {bond.GetBondType()}")
        bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), int(bond_type)])
    atom_array.bonds = struc.BondList(
        len(atom_array),
        np.asarray(bonds, dtype=np.uint32).reshape((-1, 3)),
    )
    return atom_array


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert one connected SMILES ligand to a topology-rich CIF."
    )
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chain", default="L")
    parser.add_argument("--component", default="LIG")
    parser.add_argument("--seed", default=2026, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.suffix.lower() != ".cif":
        raise ValueError("prepared ligand output must use the .cif extension")
    ligand = ligand_from_smiles(
        args.smiles,
        chain_id=args.chain,
        component_id=args.component,
        seed=args.seed,
    )
    write_cif(ligand, args.output, entry_id=args.output.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ligand_from_smiles", "main"]
