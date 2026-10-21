"""RDKit/QED and CPU Vina Dock validation of successfully postprocessed ligands.

Vina is a standalone validation method; this module is never a reward producer.
Authoritative bonds and row SMILES supply chemistry; coordinates never supply bonds.
"""
from __future__ import annotations
from pathlib import Path
from typing import Mapping
import math
import numpy as np


def require_structure_stage(input_data, stage):
    if stage == "base":
        return {"structure_stage": "base", "structure_path": str(input_data.path)}
    if stage != "postprocessed":
        raise ValueError("structure_stage must be base or postprocessed")
    from .ligands import _task_config
    _, task, _ = _task_config(input_data.task_dir)
    if task.get("postprocess") is not True or task.get("override", True) is not True:
        raise ValueError("postprocessed structure required: task must enable postprocess and override")
    # Current inference promotes successful OpenMM output to the base filename.
    # Candidate-local failures retain the base and write the explicit stage error.
    errors = str(input_data.row.get("error") or "").split(" | ")
    if any(part.strip().startswith("postprocess:") for part in errors):
        raise ValueError("postprocessed structure unavailable: " + str(input_data.row["error"]))
    if input_data.row.get("postprocess_status") != "succeeded":
        raise ValueError("postprocessed structure unavailable: explicit OpenMM success is missing")
    if input_data.path.suffix.lower() != ".cif":
        raise ValueError("current OpenMM promotion requires a CIF result")
    return {"structure_stage": "postprocessed", "structure_path": str(input_data.path),
            "stage_evidence": str(input_data.task_dir / "result.csv"),
            "postprocess_status": "succeeded",
            "promotion": "successful OpenMM atomically replaces result-row CIF"}


def pocket_box(input_data, ligand_chain, config):
    """Fixed per-task box from original conditioning ligand, never the generated ligand."""
    from .ligands import _task_config, _source_chain, _canonical_chains
    from ..design.utils.reader import read_structure
    _, _, path = _task_config(input_data.task_dir)
    source = read_structure(path)
    chain = _source_chain(source, ligand_chain)
    indices = np.flatnonzero((_canonical_chains(source) == chain)
        & (np.asarray(source.mol_type).astype(str) == "ligand")
        & ~np.isin(np.char.upper(np.asarray(source.element).astype(str)), ["H", "D", "T"]))
    if not len(indices) or not np.isfinite(source.coord[indices]).all() or not np.all(source.coord_valid[indices]):
        raise ValueError("original source ligand has no complete finite pocket coordinates")
    coords = np.asarray(source.coord[indices], float)
    size = coords.max(0) - coords.min(0) + 2 * config["box_margin"]
    size = np.maximum(size, config["box_min_size"])
    if np.any(size > config["box_max_size"]):
        raise ValueError("source pocket exceeds configured maximum box size")
    return ((coords.max(0) + coords.min(0)) / 2).tolist(), size.tolist(), {
        "source_path": str(path), "source_chain": chain, "source_heavy_atoms": len(indices)}


def molecule_from_input(input_data, chain):
    """Match complete heavy graph to authoritative SMILES and attach refined coordinates.

    Graph-equivalent automorphisms are allowed. The first deterministic mapping
    consistent with supplied stereochemistry is used; no bond is inferred from distance.
    """
    from rdkit import Chem
    from .ligands import _graph_from_atoms, _rdkit_graph, _canonical_chains
    entry = input_data.ligand_entries.get(chain)
    if not entry or not entry.get("smiles"):
        raise ValueError("designed ligand requires authoritative result-row SMILES")
    molecule = Chem.MolFromSmiles(entry["smiles"])
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("RDKit cannot recognize one connected ligand from row SMILES")
    molecule = Chem.RemoveHs(molecule)
    atoms = input_data.atoms
    rows = np.flatnonzero((_canonical_chains(atoms) == chain)
        & ~np.isin(np.char.upper(np.asarray(atoms.element).astype(str)), ["H", "D", "T"]))
    if not len(rows) or not np.isfinite(atoms.coord[rows]).all() or not np.all(atoms.coord_valid[rows]):
        raise ValueError("refined ligand coordinates are incomplete")
    observed, authority = _graph_from_atoms(atoms, rows), _rdkit_graph(molecule)
    if len(rows) != molecule.GetNumAtoms() or len(observed.edges) != len(authority.edges):
        raise ValueError("refined ligand heavy graph differs from row SMILES")
    for a, b, _ in atoms.bonds.as_array():
        if (int(a) in set(rows)) != (int(b) in set(rows)):
            raise ValueError("covalently attached ligand is unsupported by free-ligand Vina Dock")
    candidates = {i: [j for j, element in enumerate(authority.elements)
                     if element == observed.elements[i]
                     and len(authority.adjacency[j]) == len(observed.adjacency[i])
                     and all(dict(authority.atom_facts[j]).get(field) == value
                             for field, value in observed.atom_facts[i]
                             if field == "charge")]
                  for i in range(len(rows))}
    order = sorted(candidates, key=lambda i: (len(candidates[i]), -len(observed.adjacency[i]), i))
    mapping, used, attempts = {}, set(), 0
    answer = None
    target_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False)
    atom_stereo = {a.GetIdx(): a.GetProp("_CIPCode") for a in molecule.GetAtoms() if a.HasProp("_CIPCode")}
    bond_stereo = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()): b.GetStereo() for b in molecule.GetBonds()
                   if b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)}

    def prove():
        nonlocal answer, attempts
        attempts += 1
        # Check concrete CIF orders after sanitization using the trusted charge/H
        # inventory. This permits an equivalent Kekule encoding of aromatic SMILES.
        witness = Chem.RWMol(Chem.RenumberAtoms(molecule, [mapping[i] for i in range(len(rows))]))
        for bond in list(witness.GetBonds()):
            witness.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        types = {"single": Chem.BondType.SINGLE, "double": Chem.BondType.DOUBLE,
                 "triple": Chem.BondType.TRIPLE, "aromatic": Chem.BondType.AROMATIC}
        for a, b, bond_order in observed.edges:
            witness.AddBond(a, b, types[bond_order])
        try:
            witness = witness.GetMol()
            for atom in witness.GetAtoms():
                atom.SetIsAromatic(False)
            for bond in witness.GetBonds():
                if bond.GetBondType() == Chem.BondType.AROMATIC:
                    bond.GetBeginAtom().SetIsAromatic(True)
                    bond.GetEndAtom().SetIsAromatic(True)
            Chem.SanitizeMol(witness)
            if Chem.MolToSmiles(witness, isomericSmiles=False) != target_smiles:
                return
            result = Chem.Mol(molecule)
            conf = Chem.Conformer(result.GetNumAtoms())
            conf.Set3D(True)
            for i, j in mapping.items():
                conf.SetAtomPosition(j, tuple(map(float, atoms.coord[rows[i]])))
            result.RemoveAllConformers()
            result.AddConformer(conf)
            if atom_stereo or bond_stereo:
                check = Chem.Mol(result)
                Chem.AssignStereochemistryFrom3D(check, replaceExistingTags=True)
                Chem.AssignStereochemistry(check, cleanIt=True, force=True)
                if any(not check.GetAtomWithIdx(j).HasProp("_CIPCode")
                       or check.GetAtomWithIdx(j).GetProp("_CIPCode") != code for j, code in atom_stereo.items()):
                    return
                if any(check.GetBondBetweenAtoms(a, b).GetStereo() != value for (a, b), value in bond_stereo.items()):
                    return
            answer = result
        except (ValueError, RuntimeError):
            return

    def search(depth):
        if answer is not None or attempts >= 4096:
            return
        if depth == len(order):
            prove()
            return
        i = order[depth]
        for j in candidates[i]:
            if j in used:
                continue
            # Check complete edge inventory. Concrete order equivalence is
            # proved above rather than guessed across aromatic/Kekule graphs.
            if any((k in observed.adjacency[i]) != (v in authority.adjacency[j]) for k, v in mapping.items()):
                continue
            mapping[i], used_before = j, set(used)
            used.add(j)
            search(depth + 1)
            used.clear()
            used.update(used_before)
            mapping.pop(i)
            if answer is not None:
                break
    search(0)
    if answer is None:
        raise ValueError("refined graph/coordinates cannot preserve authoritative SMILES chemistry/stereochemistry")
    return answer


def _receptor_topology(atoms, chains):
    """Build a private OpenMM topology from the supplied heavy-atom graph."""
    from openmm import unit
    from openmm.app import Topology, element
    from .ligands import _canonical_chains

    chain_values = _canonical_chains(atoms)
    selected = np.isin(chain_values, chains) & (
        np.asarray(atoms.mol_type).astype(str) == "protein"
    )
    selected &= ~np.isin(
        np.char.upper(np.asarray(atoms.element).astype(str)), ["H", "D", "T"]
    )
    receptor = atoms[selected].copy()
    if (
        not len(receptor)
        or not np.isfinite(receptor.coord).all()
        or not np.asarray(receptor.coord_valid, dtype=bool).all()
    ):
        raise ValueError("protein receptor heavy-atom coordinates are incomplete")

    aliases = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    if len(chains) > len(aliases):
        raise ValueError("too many receptor chains for private PDB preparation")
    original_chains = _canonical_chains(receptor)
    for ordinal, chain in enumerate(chains):
        indices = np.flatnonzero(original_chains == chain)
        if not len(indices):
            raise ValueError("requested receptor chain absent: " + chain)
        receptor.chain_id[indices] = aliases[ordinal]
        receptor.ins_code[indices] = ""

    topology = Topology()
    chain_objects = {}
    residue_objects = {}
    atom_objects = {}
    atom_info = {}
    residue_ordinals = {}
    original_atom_names = {}
    positions = []
    for index in range(len(receptor)):
        chain_id = str(receptor.chain_id[index])
        if chain_id not in chain_objects:
            chain_objects[chain_id] = topology.addChain(chain_id)
        chain = chain_objects[chain_id]
        residue_uid = int(receptor.residue_uid[index])
        residue_key = (chain_id, residue_uid)
        if residue_key not in residue_objects:
            ordinal = residue_ordinals.get(chain_id, 0) + 1
            residue_ordinals[chain_id] = ordinal
            residue_objects[residue_key] = topology.addResidue(
                str(receptor.res_name[index]),
                chain,
                id=str(ordinal),
                insertionCode="",
            )
        try:
            atom_element = element.get_by_symbol(str(receptor.element[index]))
        except Exception as error:
            raise ValueError(
                "unsupported receptor element: " + str(receptor.element[index])
            ) from error
        atom = topology.addAtom(
            str(receptor.atom_name[index]),
            atom_element,
            residue_objects[residue_key],
            id=str(index),
        )
        atom_objects[index] = atom
        residue_key_text = f"{chain_id}:{residue_ordinals[chain_id]}"
        atom_name = str(receptor.atom_name[index])
        atom_info[index] = (residue_key_text, atom_name)
        original_atom_names.setdefault(residue_key_text, set()).add(atom_name)
        positions.append(np.asarray(receptor.coord[index], dtype=float) / 10.0)

    source_bonds = []
    for left, right, order in np.asarray(receptor.bonds.as_array(), dtype=int):
        left = int(left)
        right = int(right)
        order = int(order)
        if left not in atom_objects or right not in atom_objects:
            raise ValueError("receptor bond endpoint is outside the selected graph")
        topology.addBond(atom_objects[left], atom_objects[right], order=order)
        source_bonds.append((atom_info[left], atom_info[right], order))
    return (
        receptor,
        topology,
        unit.Quantity(np.asarray(positions), unit.nanometer),
        source_bonds,
        original_atom_names,
    )


def _template_heavy_names(template):
    return {
        template.atom_names[index]
        for index, atom in enumerate(template.mol.GetAtoms())
        if atom.GetAtomicNum() > 1
    }


def _bonds_by_name(mol, names):
    return {
        frozenset((names[bond.GetBeginAtomIdx()], names[bond.GetEndAtomIdx()])): bond
        for bond in mol.GetBonds()
        if bond.GetBeginAtom().GetAtomicNum() > 1
        and bond.GetEndAtom().GetAtomicNum() > 1
    }


def _template_bonds_by_name(template):
    return _bonds_by_name(template.mol, template.atom_names)


def _template_order_compatible(source_order, template_bond):
    """Compare one source order while accepting aromatic/Kekule encodings."""
    source_order = int(source_order)
    if source_order == 0:
        return False
    source_aromatic_order = {
        5: 1,  # aromatic single
        6: 2,  # aromatic double
        7: 3,  # aromatic triple
    }
    if template_bond.GetIsAromatic():
        # A source aromatic edge may be encoded as an aromatic subtype or
        # as either Kekule single/double edge.  Generic aromatic is 9.
        return source_order in {1, 2, 5, 6, 9}
    template_order = {
        "SINGLE": 1,
        "DOUBLE": 2,
        "TRIPLE": 3,
    }.get(str(template_bond.GetBondType()))
    if template_order is None:
        return False
    if source_order in source_aromatic_order:
        return source_aromatic_order[source_order] == template_order
    return source_order == template_order


def _template_orders_compatible(source_edges, template, template_edges):
    """Check source orders against one valid template resonance form."""
    from rdkit import Chem

    mismatches = []
    for edge, source_order in source_edges.items():
        template_bond = template_edges.get(edge)
        if template_bond is None:
            return False
        if _template_order_compatible(source_order, template_bond):
            continue
        template_order = {
            "SINGLE": 1,
            "DOUBLE": 2,
            "TRIPLE": 3,
        }.get(str(template_bond.GetBondType()))
        if (
            template_bond.GetIsAromatic()
            or int(source_order) not in {1, 2, 3}
            or template_order is None
        ):
            return False
        mismatches.append(edge)
    if not mismatches:
        return True

    # Keep explicit hydrogens and total charge in the candidate molecule.
    expected_atoms = template.mol.GetNumAtoms()
    expected_hydrogens = sum(
        atom.GetAtomicNum() == 1 for atom in template.mol.GetAtoms()
    )
    expected_total_h = tuple(
        atom.GetTotalNumHs() for atom in template.mol.GetAtoms()
    )
    expected_charge = sum(atom.GetFormalCharge() for atom in template.mol.GetAtoms())
    flags = (
        Chem.ResonanceFlags.KEKULE_ALL
        | Chem.ResonanceFlags.ALLOW_CHARGE_SEPARATION
    )
    try:
        supplier = Chem.ResonanceMolSupplier(template.mol, int(flags), 128)
        for resonance in supplier:
            if resonance.GetNumAtoms() != expected_atoms:
                continue
            if sum(
                atom.GetAtomicNum() == 1 for atom in resonance.GetAtoms()
            ) != expected_hydrogens:
                continue
            if tuple(
                atom.GetTotalNumHs() for atom in resonance.GetAtoms()
            ) != expected_total_h:
                continue
            if sum(atom.GetFormalCharge() for atom in resonance.GetAtoms()) != expected_charge:
                continue
            resonance_edges = _bonds_by_name(resonance, template.atom_names)
            if all(
                edge in resonance_edges
                and _template_order_compatible(
                    source_edges[edge], resonance_edges[edge]
                )
                for edge in source_edges
            ):
                return True
    except (TypeError, ValueError, RuntimeError):
        return False
    return False


def _private_receptor_raw_mols(
    topology, positions, source_bonds, original_atom_names
):
    """Build Meeko raw residue molecules without coordinate bond perception."""
    from openmm import unit
    from rdkit import Chem
    from meeko import ResidueChemTemplates

    templates = ResidueChemTemplates.create_from_defaults()
    positions_angstrom = np.asarray(
        positions.value_in_unit(unit.nanometer) * 10.0, dtype=float
    )
    source_internal_orders = {}
    for (left_key, left_name), (right_key, right_name), order in source_bonds:
        if left_key != right_key:
            continue
        protected = original_atom_names.get(left_key)
        edge_names = {left_name, right_name}
        if protected is None or not edge_names <= protected:
            raise ValueError(
                f"source receptor internal bond is outside original heavy graph: "
                f"{left_key}:{left_name}-{right_name}"
            )
        edge = frozenset(edge_names)
        previous = source_internal_orders.setdefault(left_key, {}).get(edge)
        if previous is not None and previous != int(order):
            raise ValueError(
                f"conflicting source receptor bond order in {left_key}: "
                f"{left_name}-{right_name}"
            )
        source_internal_orders[left_key][edge] = int(order)

    raw_input_mols = {}
    raw_indices = {}
    template_edges = {}
    for chain in topology.chains():
        for ordinal, residue in enumerate(chain.residues(), start=1):
            residue_key = f"{chain.id}:{ordinal}"
            atoms = {atom.name: atom for atom in residue.atoms()}
            if len(atoms) != sum(1 for _ in residue.atoms()):
                raise ValueError("duplicate receptor atom name in residue " + residue_key)
            heavy_names = {
                atom.name
                for atom in residue.atoms()
                if atom.element is not None and atom.element.atomic_number > 1
            }
            protected = set(original_atom_names.get(residue_key, ()))
            if not protected:
                raise ValueError(
                    f"missing original heavy-atom inventory for {residue_key}"
                )
            if not protected <= heavy_names:
                raise ValueError(
                    f"repair dropped an original heavy atom in {residue_key}"
                )
            source_edges = source_internal_orders.get(residue_key, {})
            candidates = []
            for template_name in templates.ambiguous.get(
                residue.name, [residue.name]
            ):
                template = templates.residue_templates.get(template_name)
                if template is None or _template_heavy_names(template) != heavy_names:
                    continue
                candidate_edges = _template_bonds_by_name(template)
                if any(edge not in candidate_edges for edge in source_edges):
                    continue
                if not _template_orders_compatible(
                    source_edges, template, candidate_edges
                ):
                    continue
                if any(
                    edge <= protected and edge not in source_edges
                    for edge in candidate_edges
                ):
                    continue
                candidates.append((template_name, template, candidate_edges))
            if not candidates:
                raise ValueError(
                    f"no Meeko template preserves original heavy graph for "
                    f"{residue_key}"
                )
            template_name, template, candidate_edges = candidates[0]
            raw_mol = Chem.RemoveHs(Chem.Mol(template.mol))
            Chem.SanitizeMol(raw_mol)
            template_heavy_indices = [
                index
                for index, atom in enumerate(template.mol.GetAtoms())
                if atom.GetAtomicNum() > 1
            ]
            raw_name_to_index = {
                template.atom_names[template_index]: raw_index
                for raw_index, template_index in enumerate(template_heavy_indices)
            }
            conformer = Chem.Conformer(raw_mol.GetNumAtoms())
            for atom_name, raw_index in raw_name_to_index.items():
                atom = atoms.get(atom_name)
                if atom is None:
                    raise ValueError(
                        f"template atom {atom_name} missing from receptor residue {residue_key}"
                    )
                conformer.SetAtomPosition(
                    raw_index,
                    tuple(positions_angstrom[atom.index]),
                )
            raw_mol.AddConformer(conformer, assignId=True)
            raw_input_mols[residue_key] = (raw_mol, residue.name)
            raw_indices[residue_key] = raw_name_to_index
            template_edges[residue_key] = candidate_edges

    explicit_bonds = {}
    bonded_endpoints = set()
    for (left_key, left_name), (right_key, right_name), order in source_bonds:
        if left_key == right_key:
            edge = frozenset((left_name, right_name))
            if edge not in template_edges.get(left_key, {}):
                raise ValueError(
                    f"unsupported receptor internal bond in {left_key}: "
                    f"{left_name}-{right_name}"
                )
            continue
        if order != 1:
            raise ValueError(
                "unsupported non-single receptor cross-residue bond: "
                f"{left_key}:{left_name}-{right_key}:{right_name}"
            )
        if left_key not in raw_indices or right_key not in raw_indices:
            raise ValueError("receptor bond references an unavailable residue")
        if left_name not in raw_indices[left_key] or right_name not in raw_indices[right_key]:
            raise ValueError("receptor bond references an unavailable atom")
        if (right_key, left_key) in explicit_bonds:
            bond_key = (right_key, left_key)
            pair = (raw_indices[right_key][right_name], raw_indices[left_key][left_name])
        else:
            bond_key = (left_key, right_key)
            pair = (raw_indices[left_key][left_name], raw_indices[right_key][right_name])
        explicit_bonds.setdefault(bond_key, []).append(pair)
        bonded_endpoints.update(((left_key, left_name), (right_key, right_name)))

    blunt_ends = []
    for residue_key, name_to_index in raw_indices.items():
        for atom_name in ("N", "C"):
            if atom_name in name_to_index and (residue_key, atom_name) not in bonded_endpoints:
                blunt_ends.append((residue_key, name_to_index[atom_name]))
    return templates, raw_input_mols, explicit_bonds, blunt_ends


def _prepare_receptor_with_provenance(input_data, chains):
    from meeko import MoleculePreparation, PDBQTWriterLegacy, Polymer
    from .ligands import _canonical_chains
    from ..design.postproc.openmm.preparation import repair_standard_heavy_atoms

    atoms = input_data.atoms
    chain_values = _canonical_chains(atoms)
    selected = np.isin(chain_values, chains) & (
        np.asarray(atoms.mol_type).astype(str) == "protein"
    )
    selected &= ~np.isin(
        np.char.upper(np.asarray(atoms.element).astype(str)), ["H", "D", "T"]
    )
    receptor = atoms[selected].copy()
    if not len(receptor):
        raise ValueError("protein receptor heavy-atom coordinates are incomplete")
    (
        receptor,
        topology,
        positions,
        source_bonds,
        original_atom_names,
    ) = _receptor_topology(receptor, chains)
    original_atom_count = topology.getNumAtoms()
    repaired_topology, repaired_positions = repair_standard_heavy_atoms(
        topology, positions
    )
    added_heavy_atoms = repaired_topology.getNumAtoms() - original_atom_count
    templates, raw_input_mols, explicit_bonds, blunt_ends = _private_receptor_raw_mols(
        repaired_topology,
        repaired_positions,
        source_bonds,
        original_atom_names,
    )
    polymer = Polymer(
        raw_input_mols,
        explicit_bonds,
        templates,
        MoleculePreparation(),
        blunt_ends=blunt_ends,
        forgive_extra_bonds=False,
    )
    ignored = polymer.get_ignored_monomers()
    if ignored or len(polymer.monomers) != len(raw_input_mols):
        raise ValueError("Meeko could not prepare every receptor residue: " + str(ignored))
    rigid, flexible = PDBQTWriterLegacy.write_from_polymer(polymer)
    if not rigid.strip() or flexible:
        raise ValueError("Meeko did not produce one rigid receptor PDBQT")
    return rigid, {
        "method": "repair_standard_heavy_atoms",
        "added_heavy_atoms": int(added_heavy_atoms),
    }


def prepare_receptor(input_data, chains):
    rigid, _ = _prepare_receptor_with_provenance(input_data, chains)
    return rigid


def _prepare_ligand_pdbqt(molecule):
    """Prepare one Vina ligand with charges omitted from the scoring input."""
    from rdkit import Chem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    preparations = MoleculePreparation(charge_model="zero").prepare(
        Chem.AddHs(molecule, addCoords=True)
    )
    if len(preparations) != 1:
        raise ValueError("Meeko ligand preparation did not yield one PDBQT")
    ligand_pdbqt, ok, error = PDBQTWriterLegacy.write_string(preparations[0])
    if not ok:
        raise ValueError("Meeko ligand PDBQT preparation failed: " + error)
    return ligand_pdbqt, {
        "charge_model": "zero",
        "reason": "AutoDock Vina ignores user-supplied partial charges",
    }


def evaluate_ligand_docking(input_data, *, config: Mapping, work_dir: Path):
    metrics = {"rdkit_valid": False, "docking_success": False, "qed": None, "vina_dock": None}
    result = {"metrics": metrics, "reasons": {}, "provenance": {}}
    stage = "postprocess"
    try:
        result["provenance"].update(require_structure_stage(input_data, config["structure_stage"]))
        from .ligands import _canonical_chains
        atoms = input_data.atoms
        values = _canonical_chains(atoms)
        ligands = [chain for chain in input_data.design_chains
                   if np.all(np.asarray(atoms.mol_type)[values == chain] == "ligand")]
        chain = config.get("ligand_chain")
        if chain is None:
            if len(ligands) != 1:
                raise ValueError("Vina requires one unambiguous designed ligand chain")
            chain = ligands[0]
        elif chain not in ligands:
            raise ValueError("vina.ligand_chain must identify a designed ligand")
        stage = "preparation"
        molecule = molecule_from_input(input_data, chain)
        from rdkit.Chem import QED
        metrics["rdkit_valid"] = True
        metrics["qed"] = float(QED.qed(molecule))
        center, size, box = pocket_box(input_data, chain, config)
        targets = config.get("target_chains")
        if targets is None:
            targets = list(dict.fromkeys(values[np.asarray(atoms.mol_type).astype(str) == "protein"].tolist()))
        if not targets or any(c in input_data.design_chains for c in targets):
            raise ValueError("Vina receptor must be fixed protein context chains")
        ligand_pdbqt, ligand_provenance = _prepare_ligand_pdbqt(molecule)
        result["provenance"]["ligand_preparation"] = ligand_provenance
        receptor_pdbqt, receptor_provenance = _prepare_receptor_with_provenance(
            input_data, targets
        )
        result["provenance"]["receptor_preparation"] = receptor_provenance
        work_dir = Path(work_dir)
        from .storage import ensure_directory, atomic_text
        ensure_directory(work_dir)
        receptor_path = work_dir / "receptor.pdbqt"
        atomic_text(receptor_path, receptor_pdbqt)
        stage = "docking"
        import vina, meeko
        from vina import Vina
        engine = Vina(sf_name="vina", cpu=config["cpu"], seed=config["seed"], verbosity=0)
        engine.set_receptor(str(receptor_path))
        engine.set_ligand_from_string(ligand_pdbqt)
        engine.compute_vina_maps(center=center, box_size=size)
        engine.dock(exhaustiveness=config["exhaustiveness"], n_poses=config["n_poses"])
        energies = np.asarray(engine.energies(n_poses=config["n_poses"]), dtype=float)
        if energies.ndim != 2 or not len(energies) or not np.isfinite(energies[:, 0]).all():
            raise ValueError("Vina Dock returned no finite docked pose affinities")
        metrics["vina_dock"] = float(energies[:, 0].min())
        metrics["docking_success"] = True
        result["provenance"].update(ligand_chain=chain, target_chains=targets,
            pocket_box={"center": center, "size": size, **box},
            protocol={key: config[key] for key in ("cpu", "seed", "exhaustiveness", "n_poses")},
            affinity_unit="kcal/mol", vina_version=vina.__version__, meeko_version=meeko.__version__)
    except Exception as exc:
        result["error_stage"] = stage
        result["reasons"][stage] = f"{type(exc).__name__}: {exc}"
    return result
