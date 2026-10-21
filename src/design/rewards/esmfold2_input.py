"""Translate one finalized PartiDesign candidate into offline ESMFold2 JSON."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
from biotite import structure as struc
from rdkit import Chem
from rdkit.Chem import AllChem

from ..core.records import SearchCandidate
from ..model.constants import DNA_RES_ATOMS, PROTEIN_RES_ATOMS, RNA_RES_ATOMS
from .esmfold2 import SequenceSourcePlan
from ..utils.ligand_identity import ligand_identity_matches_component


_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
_RNA3_TO_1 = {"A": "A", "G": "G", "C": "C", "U": "U"}
_DNA3_TO_1 = {"DA": "A", "DG": "G", "DC": "C", "DT": "T"}
_UNKNOWN_SEQUENCE_LETTER = {"protein": "X", "rna": "N", "dna": "N"}
_STANDARD_RES_ATOMS = {
    "protein": PROTEIN_RES_ATOMS,
    "rna": RNA_RES_ATOMS,
    "dna": DNA_RES_ATOMS,
}
_BIOTITE_TO_ORDER = {
    int(struc.BondType.ANY): "?",
    int(struc.BondType.SINGLE): "single",
    int(struc.BondType.DOUBLE): "double",
    int(struc.BondType.TRIPLE): "triple",
    int(struc.BondType.AROMATIC): "aromatic",
    int(struc.BondType.AROMATIC_SINGLE): "aromatic",
    int(struc.BondType.AROMATIC_DOUBLE): "aromatic",
    int(struc.BondType.AROMATIC_TRIPLE): "aromatic",
}


def _merge_covalent_order(
    previous: str | None,
    incoming: str,
    *,
    conflict_message: str,
) -> str:
    """Merge existence-only conditions without weakening concrete chemistry."""

    normalized = "?" if incoming.lower() in {"any", "unknown", "?"} else incoming
    if normalized not in {"?", "single", "double", "triple", "aromatic"}:
        raise ValueError("ESMFold2 covalent edge has unsupported bond order")
    if previous is None or previous == "?":
        return normalized
    if normalized == "?" or previous == normalized:
        return previous
    raise ValueError(conflict_message)


def _chain_order(candidate: SearchCandidate) -> list[str]:
    """Return the model-authored chain order independently of atom row order."""

    atoms = candidate.atoms
    chain_values = np.asarray(atoms.chain_id).astype(str)

    def order_key(chain: str) -> tuple[int, int, str]:
        rows = np.flatnonzero(chain_values == chain)
        asym_uids = {
            int(value) for value in np.asarray(atoms.asym_uid[rows], dtype=np.int64)
        }
        if len(asym_uids) != 1 or next(iter(asym_uids)) < 0:
            raise ValueError(f"ESMFold2 chain {chain!r} has an unfinished asym ledger")
        token_uids = {
            int(value) for value in np.asarray(atoms.token_uid[rows], dtype=np.int64)
        }
        if not token_uids or min(token_uids) < 0:
            raise ValueError(f"ESMFold2 chain {chain!r} has an unfinished token ledger")
        return next(iter(asym_uids)), min(token_uids), chain

    return sorted(set(chain_values.tolist()), key=order_key)


def candidate_chain_modalities(
    candidate: SearchCandidate,
) -> dict[str, str]:
    """Return the authoritative modality for every candidate chain.

    ESMFold2's internal mmCIF serializer may omit ``_chem_comp.type`` even
    though its private input and output ``_entity_poly`` ledger still identify
    a modified component as polymer.  The finalized candidate is the trusted
    source for this private read boundary.  The map also protects a true
    ligand whose CCD ID happens to look like a standard or modified polymer
    residue when the private fold file omits ``_chem_comp``.
    """

    atoms = candidate.atoms
    chain_values = np.asarray(atoms.chain_id).astype(str)
    result: dict[str, str] = {}
    for chain in _chain_order(candidate):
        rows = np.flatnonzero(chain_values == chain)
        modalities = {str(value) for value in atoms.mol_type[rows]}
        if len(modalities) != 1:
            raise ValueError(f"ESMFold2 chain {chain!r} spans multiple modalities")
        modality = next(iter(modalities))
        if modality not in {"protein", "rna", "dna", "ligand"}:
            raise ValueError(
                f"ESMFold2 chain {chain!r} has unsupported modality {modality!r}"
            )
        result[chain] = modality
    return result


def _chain_role(candidate: SearchCandidate, chain: str) -> str:
    rows = np.flatnonzero(np.asarray(candidate.atoms.chain_id).astype(str) == chain)
    design = np.asarray(candidate.atoms.is_design[rows], dtype=bool)
    if bool(design.any()) != bool(design.all()):
        raise ValueError(f"ESMFold2 chain {chain!r} mixes design and context atoms")
    return "design" if bool(design.all()) else "context"


def _preserve_exact_occurrence(candidate: SearchCandidate, rows: np.ndarray) -> bool:
    """Return whether source atom omissions are a hard identity constraint."""

    atoms = candidate.atoms
    design = np.asarray(atoms.is_design[rows], dtype=bool)
    identity = np.asarray(atoms.identity_condition[rows], dtype=bool)
    if bool(design.any()) != bool(design.all()):
        raise ValueError("ESMFold2 polymer occurrence has partial design state")
    if bool(identity.any()) != bool(identity.all()):
        raise ValueError("ESMFold2 polymer occurrence has partial identity state")
    # Context and explicitly fixed motif/design residues retain their exact
    # occurrence inventory.  A freely designable standard residue must remain
    # on the native standard-token path so an MPNN sequence may change it.
    return not bool(design.all()) or bool(identity.all())


def _polymer_payload(
    candidate: SearchCandidate,
    chain: str,
    modality: str,
) -> tuple[str, list[dict[str, Any]]]:
    atoms = candidate.atoms
    rows = np.flatnonzero(
        (np.asarray(atoms.chain_id).astype(str) == chain)
        & (np.asarray(atoms.mol_type).astype(str) == modality)
    )
    mapping = _AA3_TO_1 if modality == "protein" else _RNA3_TO_1 if modality == "rna" else _DNA3_TO_1
    sequence: list[str] = []
    modifications: list[dict[str, Any]] = []
    token_uids = sorted(
        {int(value) for value in np.asarray(atoms.token_uid[rows], dtype=np.int64)}
    )
    if any(token_uid < 0 for token_uid in token_uids):
        raise ValueError("ESMFold2 polymer token ledger is unfinished")
    for token_uid in token_uids:
        token_rows = np.flatnonzero(
            np.asarray(atoms.token_uid, dtype=np.int64) == token_uid
        )
        if {str(value) for value in atoms.chain_id[token_rows]} != {chain}:
            raise ValueError("ESMFold2 polymer token spans multiple chains")
        if {str(value) for value in atoms.mol_type[token_rows]} != {modality}:
            raise ValueError("ESMFold2 polymer token spans multiple modalities")
        residue_uids = {
            int(value) for value in np.asarray(atoms.residue_uid[token_rows], dtype=np.int64)
        }
        residues = {str(value) for value in atoms.res_name[token_rows]}
        if len(residue_uids) != 1 or len(residues) != 1:
            raise ValueError("ESMFold2 polymer token has ambiguous residue identity")
        residue_uid = next(iter(residue_uids))
        residue = next(iter(residues))
        component_id = candidate.fixed_modified_residue_names.get(residue_uid)
        if component_id is not None:
            if residue != component_id:
                raise ValueError(
                    "ESMFold2 modified occurrence disagrees with candidate residue name"
                )
            parent = candidate.fixed_modified_parent_residue.get(residue_uid)
            if parent is not None and parent not in mapping:
                raise ValueError(
                    f"ESMFold2 modified {modality} parent is not canonical: {parent!r}"
                )
            sequence.append(
                mapping[parent]
                if parent is not None
                else _UNKNOWN_SEQUENCE_LETTER[modality]
            )
            atom_names: list[str] = []
            for token_row in token_rows:
                element = str(atoms.element[token_row]).strip().upper()
                if element in {"H", "D", "T"}:
                    continue
                atom_name = str(atoms.atom_name[token_row])
                if not atom_name:
                    raise ValueError("ESMFold2 modified occurrence has an empty atom name")
                atom_names.append(atom_name)
            if not atom_names or len(atom_names) != len(set(atom_names)):
                raise ValueError(
                    "ESMFold2 modified occurrence heavy atom names must be non-empty and unique"
                )
            # The occurrence contract is an exact inventory, not a source-row
            # ordering promise.  Canonicalize the ledger so equivalent inputs
            # produce the same reward payload and cache identity even when the
            # source mmCIF orders atom_site rows differently.
            atom_names.sort()
            modifications.append(
                {
                    "position": len(sequence) - 1,
                    "ccd": component_id,
                    "atom_names": atom_names,
                    "infer_standard_polymer_bonds": False,
                }
            )
            continue
        expected_atoms = _STANDARD_RES_ATOMS[modality].get(residue)
        if expected_atoms is not None:
            atom_names = [
                str(atoms.atom_name[token_row])
                for token_row in token_rows
                if str(atoms.element[token_row]).strip().upper() not in {"H", "D", "T"}
            ]
            if not atom_names or len(atom_names) != len(set(atom_names)):
                raise ValueError(
                    "ESMFold2 standard polymer occurrence heavy atom names must be "
                    "non-empty and unique"
                )
            if (
                set(atom_names) != set(expected_atoms)
                and _preserve_exact_occurrence(candidate, token_rows)
            ):
                # A standard identity with an authored omission (for example an
                # isopeptide leaving atom) is still an exact CCD occurrence.
                # Atom-tokenize only this exceptional residue; fully populated
                # standard residues retain the native one-token path unchanged.
                sequence.append(mapping[residue])
                modifications.append(
                    {
                        "position": len(sequence) - 1,
                        "ccd": residue,
                        "atom_names": sorted(atom_names),
                        "infer_standard_polymer_bonds": False,
                    }
                )
                continue
        model_residue = candidate.model_parent_residue.get(residue_uid, residue)
        letter = mapping.get(model_residue)
        if letter is None and residue_uid in candidate.model_parent_residue:
            letter = _UNKNOWN_SEQUENCE_LETTER[modality]
        if letter is None:
            raise ValueError(
                f"ESMFold2 does not accept unresolved {modality} residue {residue!r}"
            )
        sequence.append(letter)
    if not sequence:
        raise ValueError(f"ESMFold2 chain {chain!r} has no {modality} tokens")
    return "".join(sequence), modifications


def _is_exact_polymer_occurrence(candidate: SearchCandidate, row: int) -> bool:
    """Return whether a residue needs explicit occurrence/boundary chemistry."""

    atoms = candidate.atoms
    residue_uid = int(atoms.residue_uid[row])
    if residue_uid in candidate.fixed_modified_residue_names:
        return True
    modality = str(atoms.mol_type[row])
    residue = str(atoms.res_name[row])
    expected = _STANDARD_RES_ATOMS.get(modality, {}).get(residue)
    if expected is None:
        return False
    rows = np.flatnonzero(
        np.asarray(atoms.residue_uid, dtype=np.int64) == residue_uid
    )
    present = {
        str(atoms.atom_name[index])
        for index in rows
        if str(atoms.element[index]).strip().upper() not in {"H", "D", "T"}
    }
    return present != set(expected) and _preserve_exact_occurrence(candidate, rows)


def _canonical_smiles_atom_names(molecule: Chem.Mol) -> tuple[str, ...]:
    source = Chem.Mol(molecule)
    source = Chem.RemoveHs(source)
    with_h = Chem.AddHs(source)
    ranks = list(AllChem.CanonicalRankAtoms(with_h))
    names: list[str] = []
    for atom, rank in zip(source.GetAtoms(), ranks[: source.GetNumAtoms()]):
        name = f"{atom.GetSymbol().upper()}{int(rank) + 1}"
        if len(name) > 4:
            raise ValueError("ESMFold2 canonical SMILES atom name exceeds four characters")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("ESMFold2 canonical SMILES atom names are not unique")
    return tuple(names)


def _ligand_entry(
    candidate: SearchCandidate,
    chain: str,
) -> tuple[dict[str, Any], dict[int, tuple[int, str]]]:
    atoms = candidate.atoms
    chain_rows = np.flatnonzero(
        (np.asarray(atoms.chain_id).astype(str) == chain)
        & (np.asarray(atoms.mol_type).astype(str) == "ligand")
    )
    molecule_first_token: dict[int, int] = {}
    for row in chain_rows:
        molecule_uid = int(atoms.molecule_uid[row])
        token_uid = int(atoms.token_uid[row])
        molecule_first_token[molecule_uid] = min(
            token_uid, molecule_first_token.get(molecule_uid, token_uid)
        )
    molecule_uids = sorted(
        molecule_first_token,
        key=lambda value: (molecule_first_token[value], value),
    )
    if not molecule_uids:
        raise ValueError(f"ESMFold2 ligand chain {chain!r} has no molecule")
    atom_map: dict[int, tuple[int, str]] = {}
    if len(molecule_uids) == 1 and molecule_uids[0] in candidate.private_ligand_molecules:
        molecule_uid = molecule_uids[0]
        molecule = Chem.RemoveHs(Chem.Mol(candidate.private_ligand_molecules[molecule_uid]))
        rows = np.flatnonzero(
            (np.asarray(atoms.chain_id).astype(str) == chain)
            & (np.asarray(atoms.molecule_uid, dtype=np.int64) == molecule_uid)
        )
        if molecule.GetNumAtoms() != len(rows):
            raise ValueError("private ligand atom order differs from finalized candidate")

        # A complete, independently validated CCD graph is the most stable
        # representation for ESMFold2: it preserves canonical CCD atom names
        # and lets the worker use the same compact CCD bond/leaving-atom
        # ledger as final output.  Generated or otherwise unvalidated ligands
        # continue through the private-SMILES path below.
        component_uids = {
            int(value) for value in np.asarray(atoms.component_uid[rows], dtype=np.int64)
        }
        if len(component_uids) == 1:
            component_uid = next(iter(component_uids))
            identity = candidate.validated_ligand_identities.get(component_uid)
            if identity is not None and ligand_identity_matches_component(
                atoms, rows, identity
            ):
                for row in rows:
                    atom_map[int(atoms.atom_uid[row])] = (
                        0,
                        str(atoms.atom_name[row]),
                    )
                return {"ccd": [identity.component_id]}, atom_map

        original_smiles: str | None = None
        ligand_condition = candidate.base_request.task.condition.get("ligand", {})
        if isinstance(ligand_condition, Mapping):
            chain_condition = ligand_condition.get(chain)
            if (
                isinstance(chain_condition, Mapping)
                and chain_condition.get("mode") == "smiles"
                and isinstance(chain_condition.get("smiles"), str)
            ):
                original_smiles = str(chain_condition["smiles"])
        smiles = original_smiles or Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
        reparsed = Chem.MolFromSmiles(smiles)
        if reparsed is None or reparsed.GetNumAtoms() != molecule.GetNumAtoms():
            raise ValueError("private ligand cannot be represented by one canonical SMILES")
        # Canonical SMILES parsing may reorder atoms. Match the original molecule
        # to the canonical serialized graph before assigning the upstream names.
        match = molecule.GetSubstructMatch(reparsed)
        if len(match) != molecule.GetNumAtoms():
            reverse = reparsed.GetSubstructMatch(molecule)
            if len(reverse) != molecule.GetNumAtoms():
                raise ValueError("private ligand SMILES atom mapping is not bijective")
            original_to_canonical = tuple(int(value) for value in reverse)
        else:
            original_to_canonical = [0] * len(match)
            for canonical_index, original_index in enumerate(match):
                original_to_canonical[int(original_index)] = canonical_index
            original_to_canonical = tuple(original_to_canonical)
        canonical_names = _canonical_smiles_atom_names(reparsed)
        for local, row in enumerate(rows):
            atom_map[int(atoms.atom_uid[row])] = (
                0,
                canonical_names[original_to_canonical[local]],
            )
        return {"smiles": smiles}, atom_map

    component_first_token: dict[int, int] = {}
    for row in chain_rows:
        component_uid = int(atoms.component_uid[row])
        token_uid = int(atoms.token_uid[row])
        component_first_token[component_uid] = min(
            token_uid, component_first_token.get(component_uid, token_uid)
        )
    component_uids = sorted(
        component_first_token,
        key=lambda value: (component_first_token[value], value),
    )
    components: list[str] = []
    for residue_index, component_uid in enumerate(component_uids):
        rows = np.flatnonzero(
            (np.asarray(atoms.chain_id).astype(str) == chain)
            & (np.asarray(atoms.component_uid, dtype=np.int64) == component_uid)
        )
        names = set(str(value) for value in atoms.res_name[rows])
        if len(names) != 1:
            raise ValueError("one CCD ligand component has multiple component names")
        component = next(iter(names))
        identity = candidate.validated_ligand_identities.get(component_uid)
        if identity is not None:
            if not ligand_identity_matches_component(atoms, rows, identity):
                raise ValueError("validated CCD ligand identity no longer matches candidate")
            component = identity.component_id
        if not component or component in {"UNL", "LIG"}:
            raise ValueError(
                "ligand lacks private SMILES and a usable local CCD component"
            )
        components.append(component)
        for row in rows:
            atom_map[int(atoms.atom_uid[row])] = (
                residue_index,
                str(atoms.atom_name[row]),
            )
    return {"ccd": components}, atom_map


def _entity_identity(entries: list[dict[str, Any]]) -> None:
    entity_by_signature: dict[str, int] = {}
    copies: dict[int, int] = {}
    for entry in entries:
        signature = json.dumps(
            {
                "modality": entry["modality"],
                "sequence": entry.get("sequence"),
                "smiles": entry.get("smiles"),
                "ccd": entry.get("ccd"),
                "modifications": entry.get("modifications", []),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        entity = entity_by_signature.setdefault(signature, len(entity_by_signature) + 1)
        copies[entity] = copies.get(entity, 0) + 1
        entry["model_entity_index"] = entity
        entry["model_entity_copy"] = copies[entity]
        entry["output_entity_id"] = str(entity)


def _is_template_edge(candidate: SearchCandidate, left: int, right: int) -> bool:
    """Return whether upstream sequence/SMILES/CCD already reconstructs an edge."""

    atoms = candidate.atoms
    if int(atoms.token_uid[left]) == int(atoms.token_uid[right]):
        return True
    modalities = {str(atoms.mol_type[left]), str(atoms.mol_type[right])}
    if modalities == {"ligand"} and int(atoms.molecule_uid[left]) == int(
        atoms.molecule_uid[right]
    ):
        molecule_uid = int(atoms.molecule_uid[left])
        if molecule_uid in candidate.private_ligand_molecules:
            # The private SMILES path reconstructs the complete connected graph.
            return True
        # A CCD list reconstructs only each component's internal graph.  A real
        # bond between two components of the same authored molecule must remain
        # explicit so it is not silently lost at the reward boundary.
        return int(atoms.component_uid[left]) == int(atoms.component_uid[right])
    if len(modalities) != 1 or "ligand" in modalities:
        return False
    if (
        _is_exact_polymer_occurrence(candidate, left)
        or _is_exact_polymer_occurrence(candidate, right)
    ):
        # PartiDesign disables upstream's generic modified-boundary inference.
        # Every real boundary, closure, or crosslink touching the occurrence is
        # therefore carried with exact named endpoints below.
        return False
    if int(atoms.asym_uid[left]) != int(atoms.asym_uid[right]):
        return False
    positions = [int(atoms.res_id[left]), int(atoms.res_id[right])]
    if abs(positions[0] - positions[1]) != 1:
        return False
    earlier, later = (
        (left, right) if positions[0] < positions[1] else (right, left)
    )
    endpoint = (str(atoms.atom_name[earlier]), str(atoms.atom_name[later]))
    modality = next(iter(modalities))
    if modality == "protein":
        return endpoint == ("C", "N")
    if modality in {"rna", "dna"}:
        return endpoint == ("O3'", "P")
    return False


def _explicit_covalent_edges(
    candidate: SearchCandidate,
    *,
    include_topology_conditions: bool = True,
) -> tuple[tuple[int, int, str], ...]:
    """Return explicit non-template edges without conflating condition and output.

    ESMFold2 consumes every non-template declared topology condition as an input
    bond; sequence/SMILES/CCD-internal edges are reconstructed upstream.  Other
    reward adapters, notably AF2, may request only the authoritative edges that
    are already present in the candidate ``BondList`` for final-output overlay.
    """

    atoms = candidate.atoms
    uid_to_row = {int(uid): row for row, uid in enumerate(atoms.atom_uid)}
    by_uid: dict[tuple[int, int], str] = {}
    if include_topology_conditions:
        for condition in candidate.topology_conditions:
            uid_a, uid_b = int(condition.atom_uid_a), int(condition.atom_uid_b)
            try:
                row_a, row_b = uid_to_row[uid_a], uid_to_row[uid_b]
            except KeyError as error:
                raise ValueError(
                    "ESMFold2 covalent endpoint is absent from candidate"
                ) from error
            if _is_template_edge(candidate, row_a, row_b):
                continue
            key = tuple(sorted((uid_a, uid_b)))
            previous = by_uid.get(key)
            order = str(condition.order)
            by_uid[key] = _merge_covalent_order(
                previous,
                order,
                conflict_message="ESMFold2 explicit bond order conflict",
            )

    bonds = atoms.bonds
    if bonds is not None:
        for left, right, raw_order in bonds.as_array():
            left, right = int(left), int(right)
            if _is_template_edge(candidate, left, right):
                continue
            order = _BIOTITE_TO_ORDER.get(int(raw_order))
            if order is None:
                if int(raw_order) == int(struc.BondType.COORDINATION):
                    continue
                raise ValueError("ESMFold2 special covalent edge has unknown bond order")
            key = tuple(
                sorted((int(atoms.atom_uid[left]), int(atoms.atom_uid[right])))
            )
            previous = by_uid.get(key)
            by_uid[key] = _merge_covalent_order(
                previous,
                order,
                conflict_message="ESMFold2 authoritative bond order conflict",
            )

    result: list[tuple[int, int, str]] = []
    for (uid_a, uid_b), order in sorted(by_uid.items()):
        try:
            row_a, row_b = uid_to_row[uid_a], uid_to_row[uid_b]
        except KeyError as error:
            raise ValueError("ESMFold2 covalent endpoint is absent from candidate") from error
        result.append((uid_a, uid_b, order))
    return tuple(result)


def build_esmfold2_input(
    candidate: SearchCandidate,
    sequence_plan: SequenceSourcePlan,
) -> dict[str, Any]:
    atoms = candidate.atoms
    entries: list[dict[str, Any]] = []
    ligand_maps: dict[str, dict[int, tuple[int, str]]] = {}
    design_chains: list[str] = []
    for chain in _chain_order(candidate):
        rows = np.flatnonzero(np.asarray(atoms.chain_id).astype(str) == chain)
        modalities = set(str(value) for value in atoms.mol_type[rows])
        if len(modalities) != 1:
            raise ValueError(f"ESMFold2 chain {chain!r} spans multiple modalities")
        modality = next(iter(modalities))
        role = _chain_role(candidate, chain)
        design_chains.extend([chain] if role == "design" else [])
        entry: dict[str, Any] = {
            "chain_id": chain,
            "modality": modality,
            "role": role,
            "modifications": [],
        }
        if modality == "ligand":
            payload, atom_map = _ligand_entry(candidate, chain)
            entry.update(payload)
            ligand_maps[chain] = atom_map
        else:
            sequence, modifications = _polymer_payload(candidate, chain, modality)
            if (
                (chain == sequence_plan.design_chain
                 or (candidate.base_request.task.format == "antibody" and role == "design"))
                and modality in {"protein", "rna", "dna"}
                and chain in sequence_plan.chain_sequences
            ):
                planned = str(sequence_plan.chain_sequences[chain])
                if len(planned) != len(sequence):
                    raise ValueError(
                        f"ESMFold2 sequence length changed for chain {chain!r}"
                    )
                for modification in modifications:
                    position = int(modification["position"])
                    if planned[position] != sequence[position]:
                        raise ValueError(
                            "ESMFold2 sequence generator changed a fixed modified "
                            f"position {chain}:{position + 1}"
                        )
                sequence = planned
            entry["sequence"] = sequence
            entry["modifications"] = modifications
            if modality in {"protein", "rna"}:
                entry["msa"] = None
        entries.append(entry)
    if not design_chains:
        raise ValueError("ESMFold2 reward requires at least one design chain")
    _entity_identity(entries)

    uid_to_row = {int(uid): row for row, uid in enumerate(atoms.atom_uid)}
    position_by_token: dict[int, int] = {}
    for entry in entries:
        chain = entry["chain_id"]
        rows = np.flatnonzero(np.asarray(atoms.chain_id).astype(str) == chain)
        if entry["modality"] == "ligand":
            continue
        token_uids = sorted(
            {int(value) for value in np.asarray(atoms.token_uid[rows], dtype=np.int64)}
        )
        if any(token_uid < 0 for token_uid in token_uids):
            raise ValueError("ESMFold2 polymer token ledger is unfinished")
        position_by_token.update(
            {token_uid: index for index, token_uid in enumerate(token_uids)}
        )

    bonds: list[dict[str, Any]] = []
    for atom_uid_a, atom_uid_b, order in _explicit_covalent_edges(candidate):
        endpoints: list[dict[str, Any]] = []
        for atom_uid in (atom_uid_a, atom_uid_b):
            row = uid_to_row.get(int(atom_uid))
            if row is None:
                raise ValueError("ESMFold2 covalent endpoint is absent from candidate")
            chain = str(atoms.chain_id[row])
            modality = str(atoms.mol_type[row])
            if modality == "ligand":
                try:
                    residue_index, atom_name = ligand_maps[chain][int(atom_uid)]
                except KeyError as exc:
                    raise ValueError("ESMFold2 ligand atom mapping is incomplete") from exc
            else:
                residue_index = position_by_token[int(atoms.token_uid[row])]
                atom_name = str(atoms.atom_name[row])
            endpoints.append(
                {
                    "chain_id": chain,
                    "residue_index": residue_index,
                    "atom_name": atom_name,
                }
            )
        bonds.append(
            {
                "endpoint1": endpoints[0],
                "endpoint2": endpoints[1],
                "bond_order": order,
            }
        )
    from .esmfold2_mapping import apply_source_mapping

    return apply_source_mapping({
        "schema_version": 1,
        "sample_id": candidate.candidate_id,
        "variant_id": "sequence_000",
        # ``design_chain`` remains as a compatibility identity for the copied
        # PipeTools artifact ledger.  ``design_chains`` is authoritative for
        # scoring and cache eligibility and supports mixed protein+nucleic
        # design without changing the one-protein-chain MPNN contract.
        "design_chain": design_chains[0],
        "design_chains": design_chains,
        "msa_policy": "none",
        "sequences": entries,
        "covalent_bonds": bonds,
    }, atoms, design_chains=tuple(design_chains))


__all__ = [
    "build_esmfold2_input",
    "candidate_chain_modalities",
]
