"""Hard reconciliation and source-frame assembly for terminal candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import warnings

import biotite.structure as struc
import numpy as np
from rdkit import Chem

from ..utils.atoms import (
    DNA_ATOM27,
    DNA_RES_ATOMS,
    PROTEIN_ATOM37,
    PROTEIN_RES_ATOMS,
    RNA_ATOM27,
    RNA_RES_ATOMS,
    validate_atom_array,
)
from ..utils.chemistry import (
    private_molecule_from_authoring_graph,
    private_molecule_from_trusted_graph,
    reconcile_required_edges,
    trusted_ccd_private_molecule,
)
from .decode import DecodedProposal
from .modified_polymer import (
    ModifiedPolymerRecoveryWarning,
    plan_modified_polymer_coordinates,
    proxy_atom_name,
    proxy_atom_rows,
    proxy_inventory,
)
from ..model.constants import LIGAND_FORMAL_CHARGES
from .model_tokens import build_model_token_layout
from .records import CompiledSample, SearchCandidate


_INVENTORIES = {
    "protein": PROTEIN_RES_ATOMS,
    "rna": RNA_RES_ATOMS,
    "dna": DNA_RES_ATOMS,
}
_SLOTS = {
    "protein": (0, PROTEIN_ATOM37),
    "rna": (len(PROTEIN_ATOM37), RNA_ATOM27),
    "dna": (len(PROTEIN_ATOM37) + len(RNA_ATOM27), DNA_ATOM27),
}
_COMMON_POLYMER_ATOMS = {
    modality: frozenset.intersection(
        *(frozenset(atom_names) for atom_names in inventories.values())
    )
    for modality, inventories in _INVENTORIES.items()
}
_ORDER_TO_BIOTITE = {
    "single": int(struc.BondType.SINGLE),
    "double": int(struc.BondType.DOUBLE),
    "triple": int(struc.BondType.TRIPLE),
    "aromatic": int(struc.BondType.AROMATIC),
}


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    """Authoritative structure plus private, non-writer chemistry sidecars."""

    atoms: struc.AtomArray
    private_ligand_charge_logits: np.ndarray | None = None
    private_ligand_molecules: Mapping[int, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    private_ligand_errors: Mapping[int, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    ligand_fragment_metadata: Mapping[int, Mapping[str, Any]] = field(
        default_factory=lambda: MappingProxyType({})
    )


def _source_coordinates(compiled: CompiledSample, model_nm: np.ndarray) -> np.ndarray:
    values = np.asarray(model_nm, dtype=np.float64)
    if values.shape[-1:] != (3,) or not np.isfinite(values).all():
        raise ValueError("generated model coordinates must be finite [...,3]")
    affine = compiled.model_to_source_affine
    return (values @ affine[:3, :3] + affine[3, :3]) * 10.0


def _bond_map(atoms: struc.AtomArray) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    if atoms.bonds is None:
        return result
    for raw_i, raw_j, raw_type in atoms.bonds.as_array():
        key = tuple(sorted((int(raw_i), int(raw_j))))
        bond_type = int(raw_type)
        previous = result.get(key)
        if previous is not None and previous != bond_type:
            raise ValueError(f"conflicting BondList orders for atom rows {key}")
        result[key] = bond_type
    return result


def _set_bond_map(
    atoms: struc.AtomArray,
    edges: Mapping[tuple[int, int], int],
) -> None:
    rows = np.asarray(
        [[i, j, bond_type] for (i, j), bond_type in sorted(edges.items())],
        dtype=np.uint32,
    ).reshape((-1, 3))
    atoms.bonds = struc.BondList(len(atoms), rows)


def _merge_edge(
    edges: dict[tuple[int, int], int],
    row_i: int,
    row_j: int,
    bond_type: int,
) -> None:
    key = tuple(sorted((int(row_i), int(row_j))))
    if key[0] == key[1]:
        raise ValueError("assembly cannot create a self-edge")
    previous = edges.get(key)
    if previous is None or previous == int(struc.BondType.ANY):
        edges[key] = int(bond_type)
    elif int(bond_type) != int(struc.BondType.ANY) and previous != int(bond_type):
        raise ValueError(f"assembly bond order conflict for rows {key}")


def _complete_component_uids(compiled: CompiledSample) -> frozenset[int]:
    return frozenset(
        int(policy.component_uid)
        for policy in compiled.edge_policies
        if policy.complete_graph and policy.component_uid is not None
    )


def _design_protein_terminal_rows(
    atoms: struc.AtomArray,
) -> dict[int, tuple[int, int]]:
    """Return exact first-N/last-C rows for unambiguous design protein asyms."""

    design_protein = (
        np.asarray(atoms.is_design, dtype=bool)
        & (np.asarray(atoms.mol_type).astype(str) == "protein")
    )
    result: dict[int, tuple[int, int]] = {}
    for raw_asym in np.unique(atoms.asym_uid[design_protein]):
        asym_uid = int(raw_asym)
        rows = np.where(design_protein & (atoms.asym_uid == asym_uid))[0]
        if len(rows) < 2:
            continue
        residue_positions = np.asarray(atoms.res_id[rows], dtype=np.int64)
        for position in np.unique(residue_positions):
            position_rows = rows[residue_positions == position]
            if len(np.unique(atoms.residue_uid[position_rows])) != 1:
                break
        else:
            first_position = int(residue_positions.min())
            last_position = int(residue_positions.max())
            first_rows = rows[
                (residue_positions == first_position)
                & (np.asarray(atoms.atom_name[rows]).astype(str) == "N")
            ]
            last_rows = rows[
                (residue_positions == last_position)
                & (np.asarray(atoms.atom_name[rows]).astype(str) == "C")
            ]
            if len(first_rows) == 1 and len(last_rows) == 1:
                result[asym_uid] = (int(first_rows[0]), int(last_rows[0]))
    return result


def _head_to_tail_design_protein_asyms(
    compiled: CompiledSample | SearchCandidate,
) -> frozenset[int]:
    """Find exact required first-N/last-C single closures, independent of endpoint order."""

    atoms = compiled.atoms
    uid_to_row = {int(uid): row for row, uid in enumerate(atoms.atom_uid)}
    terminal_rows = _design_protein_terminal_rows(atoms)
    result: set[int] = set()
    for condition in compiled.topology_conditions:
        if not condition.required_output or condition.order != "single":
            continue
        if (
            int(condition.atom_uid_a) not in uid_to_row
            or int(condition.atom_uid_b) not in uid_to_row
        ):
            raise ValueError("topology condition references missing AtomUID")
        rows = np.asarray(
            [uid_to_row[int(condition.atom_uid_a)], uid_to_row[int(condition.atom_uid_b)]],
            dtype=np.int64,
        )
        if not bool(np.asarray(atoms.is_design[rows], dtype=bool).all()):
            continue
        if set(str(value) for value in atoms.mol_type[rows]) != {"protein"}:
            continue
        asym_uids = np.unique(atoms.asym_uid[rows])
        if len(asym_uids) != 1:
            continue
        asym_uid = int(asym_uids[0])
        if asym_uid not in terminal_rows:
            continue
        if set(rows.tolist()) == set(terminal_rows[asym_uid]):
            result.add(asym_uid)

    # A source graph can carry a real terminal C-N single edge even when an
    # older request omitted the corresponding topology condition. Preserve
    # that physical closure without changing the compiled condition ledger.
    if atoms.bonds is not None:
        for raw_i, raw_j, raw_order in atoms.bonds.as_array():
            if int(raw_order) != int(struc.BondType.SINGLE):
                continue
            row_i, row_j = int(raw_i), int(raw_j)
            if (
                row_i < 0
                or row_j < 0
                or row_i >= len(atoms)
                or row_j >= len(atoms)
            ):
                raise ValueError("source bond references an out-of-range atom row")
            if int(atoms.asym_uid[row_i]) != int(atoms.asym_uid[row_j]):
                continue
            asym_uid = int(atoms.asym_uid[row_i])
            if asym_uid not in terminal_rows:
                continue
            if {row_i, row_j} == set(terminal_rows[asym_uid]):
                result.add(asym_uid)

    request = getattr(compiled, "request", getattr(compiled, "base_request", None))
    task = getattr(request, "task", None)
    if bool(getattr(task, "cyclic_offset", False)):
        design_chain_ids = {
            str(chain.chain_id)
            for chain in (getattr(task, "design_chains", ()) or ())
            if str(getattr(chain, "modality", "")) == "protein"
        }
        selected = getattr(task, "cyclic_offset_chains", ())
        if selected is None:
            selected = ()
        elif isinstance(selected, str):
            selected = (selected,)
        else:
            try:
                selected = tuple(selected)
            except TypeError:
                selected = ()
        selected = {str(value) for value in selected}
        if selected:
            selected &= design_chain_ids
        elif len(design_chain_ids) == 1:
            selected = set(design_chain_ids)
        else:
            selected = set()
        chain_values = np.asarray(atoms.chain_id).astype(str)
        for chain_id in sorted(selected):
            chain_rows = np.where(
                np.asarray(atoms.is_design, dtype=bool)
                & (np.asarray(atoms.mol_type).astype(str) == "protein")
                & (chain_values == chain_id)
            )[0]
            asyms = np.unique(atoms.asym_uid[chain_rows])
            if len(asyms) != 1 or int(asyms[0]) not in terminal_rows:
                raise ValueError(
                    f"cyclic offset chain {chain_id!r} lacks unambiguous terminal N/C endpoints"
                )
            result.add(int(asyms[0]))
    return frozenset(result)


def normalize_cyclic_product(
    atoms: struc.AtomArray,
    chain_ids: Sequence[str],
) -> struc.AtomArray:
    """Return a copied output with selected protein chains closed physically.

    This helper is deliberately output-only. It removes an unhard terminal
    OXT and adds the exact existing terminal C--N edge, preserving all
    coordinates and AtomUIDs. It never changes the input array.
    """

    if isinstance(chain_ids, str):
        raw_selected_ids = (chain_ids,)
    else:
        raw_selected_ids = tuple(str(value) for value in chain_ids)
    if len(set(raw_selected_ids)) != len(raw_selected_ids):
        raise ValueError("cyclic product selection contains duplicate chain IDs")
    selected_ids = raw_selected_ids
    output = atoms.copy()
    if not selected_ids:
        return output
    atom_uids = np.asarray(atoms.atom_uid, dtype=np.int64)
    if len(np.unique(atom_uids)) != len(atom_uids):
        raise ValueError("cyclic product requires unique AtomUIDs")
    if "label_asym_id" in atoms.get_annotation_categories():
        labels = np.asarray(atoms.label_asym_id).astype(str)
    else:
        labels = np.asarray(atoms.chain_id).astype(str)
    chains = np.asarray(atoms.chain_id).astype(str)
    asym_values = np.asarray(atoms.asym_uid, dtype=np.int64)
    selected_asyms: set[int] = set()
    for chain_id in selected_ids:
        matches = np.where((chains == chain_id) | (labels == chain_id))[0]
        if not len(matches):
            raise ValueError(f"cyclic product chain {chain_id!r} is absent")
        asyms = np.unique(asym_values[matches])
        if len(asyms) != 1:
            raise ValueError(f"cyclic product chain {chain_id!r} is ambiguous")
        asym_uid = int(asyms[0])
        if asym_uid in selected_asyms:
            raise ValueError(f"cyclic product chain {chain_id!r} duplicates a selected asym")
        selected_asyms.add(asym_uid)

    terminal_pairs: list[tuple[int, int]] = []
    remove_rows: set[int] = set()
    motif = np.asarray(
        getattr(atoms, "motif_coord_condition", np.zeros(len(atoms), dtype=bool)),
        dtype=bool,
    )
    center = np.asarray(
        getattr(atoms, "token_center_condition", np.zeros(len(atoms), dtype=bool)),
        dtype=bool,
    )
    for asym_uid in sorted(selected_asyms):
        rows = np.where(asym_values == asym_uid)[0]
        if not bool(np.asarray(atoms.is_design[rows], dtype=bool).all()) or set(
            np.asarray(atoms.mol_type[rows]).astype(str)
        ) != {"protein"}:
            raise ValueError("cyclic product selection must contain design protein rows only")
        positions = np.asarray(atoms.res_id[rows], dtype=np.int64)
        if len(np.unique(positions)) < 2:
            raise ValueError("cyclic product requires distinct first and last residues")
        ins_codes = np.asarray(
            getattr(atoms, "ins_code", np.full(len(atoms), "", dtype="<U1"))
        ).astype(str)
        if any(str(ins_codes[row]).strip() not in {"", ".", "?"} for row in rows):
            raise ValueError("cyclic product does not support insertion-coded residues")
        for position in np.unique(positions):
            position_rows = rows[positions == position]
            if len(np.unique(atoms.residue_uid[position_rows])) != 1:
                raise ValueError("cyclic product has ambiguous residue positions")
        first = int(positions.min())
        last = int(positions.max())
        first_n = rows[
            (positions == first)
            & (np.asarray(atoms.atom_name[rows]).astype(str) == "N")
        ]
        last_c = rows[
            (positions == last)
            & (np.asarray(atoms.atom_name[rows]).astype(str) == "C")
        ]
        if len(first_n) != 1 or len(last_c) != 1:
            raise ValueError("cyclic product lacks unambiguous terminal N/C endpoints")
        oxt_rows = rows[
            (positions == last)
            & (np.asarray(atoms.atom_name[rows]).astype(str) == "OXT")
        ]
        if any(bool(motif[row] or center[row]) for row in oxt_rows):
            raise ValueError("cyclic product conflicts with hard terminal OXT")
        remove_rows.update(int(row) for row in oxt_rows)
        terminal_pairs.append((int(atoms.atom_uid[last_c[0]]), int(atoms.atom_uid[first_n[0]])))

    keep = np.ones(len(atoms), dtype=bool)
    if remove_rows:
        keep[sorted(remove_rows)] = False
    output = atoms[keep]
    if output.bonds is not None:
        for raw_i, raw_j, _raw_order in output.bonds.as_array():
            row_i, row_j = int(raw_i), int(raw_j)
            if (
                row_i < 0
                or row_j < 0
                or row_i >= len(output)
                or row_j >= len(output)
            ):
                raise ValueError("cyclic product bond references an invalid atom row")
            if row_i == row_j:
                raise ValueError("cyclic product bond cannot be a self-edge")
    uid_to_row = {int(uid): row for row, uid in enumerate(output.atom_uid)}
    edges = _bond_map(output)
    for c_uid, n_uid in terminal_pairs:
        if c_uid not in uid_to_row or n_uid not in uid_to_row:
            raise ValueError("cyclic product terminal endpoint disappeared")
        _merge_edge(
            edges,
            uid_to_row[c_uid],
            uid_to_row[n_uid],
            int(struc.BondType.SINGLE),
        )
    _set_bond_map(output, edges)
    return output


def _modified_polymer_token_coordinate_plan(
    compiled: CompiledSample,
    proposal: DecodedProposal,
    token_uid: int,
    rows: np.ndarray,
    *,
    source_polymer: np.ndarray,
    preserve_backbone_coordinates: bool,
    emit_warning: bool,
) -> dict[int, np.ndarray]:
    """Plan one modified residue while keeping generated common atoms exact."""

    atoms = compiled.atoms
    if len(rows) == 0:
        raise ValueError("modified-polymer token has no authoring rows")
    modalities = {str(value) for value in atoms.mol_type[rows]}
    residue_uids = np.unique(atoms.residue_uid[rows])
    residue_names = {str(value) for value in atoms.res_name[rows]}
    if len(modalities) != 1 or len(residue_uids) != 1 or len(residue_names) != 1:
        raise ValueError("modified-polymer token has ambiguous source identity")
    modality = next(iter(modalities))
    residue_uid = int(residue_uids[0])
    residue_name = next(iter(residue_names))
    try:
        proxy_residue_name = compiled.model_parent_residue[residue_uid]
    except KeyError as exc:
        raise ValueError("modified-polymer token has no model proxy") from exc
    offset, slot_names = _SLOTS[modality]
    generated_by_name = {
        atom_name: np.asarray(
            source_polymer[token_uid, offset + slot_names.index(atom_name)],
            dtype=np.float32,
        )
        for atom_name in proxy_inventory(modality, proxy_residue_name)
    }
    # Motif geometry is input-only; backbone-only output behavior is unchanged.
    preserve_source_coord_mask = np.zeros(len(rows), dtype=bool)
    if preserve_backbone_coordinates:
        proxy_names_by_local_row = proxy_atom_rows(
            modality=modality,
            residue_name=residue_name,
            proxy_residue_name=proxy_residue_name,
            source_atom_names=[str(atoms.atom_name[row]) for row in rows],
        )
        for local_row, row in enumerate(rows):
            hard_coordinate = bool(
                atoms.coord_valid[row]
                and (
                    atoms.token_center_condition[row]
                    and not atoms.motif_coord_condition[row]
                )
            )
            preserve_source_coord_mask[local_row] = hard_coordinate
            atom_name = proxy_names_by_local_row.get(local_row)
            if atom_name is None:
                if hard_coordinate and proxy_atom_name(
                    modality=modality,
                    residue_name=residue_name,
                    proxy_residue_name=proxy_residue_name,
                    source_atom_name=str(atoms.atom_name[row]),
                ) is not None:
                    raise ValueError(
                        "hard modified-polymer atom aliases to an occupied proxy slot"
                    )
                continue
            if hard_coordinate:
                generated_by_name[atom_name] = np.asarray(
                    atoms.coord[row], dtype=np.float32
                )

    plan = plan_modified_polymer_coordinates(
        modality=modality,
        residue_name=residue_name,
        proxy_residue_name=proxy_residue_name,
        source_atom_names=[str(value) for value in atoms.atom_name[rows]],
        source_coordinates=np.asarray(atoms.coord[rows], dtype=np.float32),
        source_coord_valid=np.asarray(atoms.coord_valid[rows], dtype=bool),
        generated_proxy_coordinates=generated_by_name,
        preserve_source_coord_mask=preserve_source_coord_mask,
    )
    retained_rows = {
        int(rows[int(local_row)])
        for local_row in (*plan.common_coordinates, *plan.extra_coordinates)
    }
    residue_row_set = {int(row) for row in rows}
    if atoms.bonds is not None:
        for raw_i, raw_j, _ in atoms.bonds.as_array():
            row_i, row_j = int(raw_i), int(raw_j)
            if row_i in residue_row_set:
                local_row, other_row = row_i, row_j
            elif row_j in residue_row_set:
                local_row, other_row = row_j, row_i
            else:
                continue
            if local_row in retained_rows:
                continue
            if int(atoms.residue_uid[other_row]) != residue_uid:
                raise ValueError(
                    "modified-polymer fallback would remove an authoritative "
                    "cross-residue bond endpoint"
                )
    if emit_warning and plan.fallback_reason is not None:
        warnings.warn(
            ModifiedPolymerRecoveryWarning(
                str(atoms.chain_id[rows[0]]),
                int(atoms.res_id[rows[0]]),
                residue_name,
                plan.fallback_reason,
            ),
            stacklevel=3,
        )
    coordinates: dict[int, np.ndarray] = {}
    for local_row, coordinate in plan.common_coordinates.items():
        coordinates[int(atoms.atom_uid[int(rows[int(local_row)])])] = np.asarray(
            coordinate, dtype=np.float32
        )
    for local_row, coordinate in plan.extra_coordinates.items():
        coordinates[int(atoms.atom_uid[int(rows[int(local_row)])])] = np.asarray(
            coordinate, dtype=np.float32
        )
    return coordinates


def _activate_and_fill(
    compiled: CompiledSample,
    proposal: DecodedProposal,
) -> struc.AtomArray:
    atoms = compiled.atoms.copy()
    validate_atom_array(atoms, phase="tokenized")
    layout = build_model_token_layout(
        compiled.atoms,
        compiled.modified_polymer_carriers,
    )
    authoring_token_count = layout.num_authoring_tokens
    model_token_count = layout.num_model_tokens
    if authoring_token_count <= 0 or model_token_count <= 0:
        raise ValueError("compiled sample has no canonical tokens")
    carrier_authoring_tokens = frozenset(
        int(layout.model_token_to_authoring_token[model_token])
        for model_token, is_carrier in enumerate(layout.carrier_token_mask)
        if is_carrier
    )
    omitted_by_token = {
        int(token_uid): frozenset(str(atom_name) for atom_name in atom_names)
        for token_uid, atom_names in compiled.omitted_atom_names_by_token_uid.items()
    }
    for token_uid, atom_names in omitted_by_token.items():
        if token_uid < 0 or token_uid >= authoring_token_count:
            raise ValueError("compiled omit-atoms sidecar references a missing token")
        if not atom_names:
            raise ValueError("compiled omit-atoms sidecar contains an empty atom set")
        # Carrier omit/leaving atoms have already been physically removed by the
        # compiler.  This legacy sidecar belongs only to standard/proxy rebuilds.
        if token_uid in carrier_authoring_tokens:
            continue
        token_rows = np.where(atoms.token_uid == token_uid)[0]
        if not bool(np.asarray(atoms.is_design[token_rows], dtype=bool).all()):
            raise ValueError("compiled omit-atoms sidecar references context")
        if set(str(value) for value in atoms.mol_type[token_rows]) == {"ligand"}:
            raise ValueError("compiled omit-atoms sidecar references a ligand")
        leaked = atom_names.intersection(
            str(value) for value in atoms.atom_name[token_rows]
        )
        if leaked:
            raise ValueError(
                "compiled omit-atoms sidecar disagrees with the atom inventory: "
                f"token {token_uid} still contains {sorted(leaked)}"
            )
    if not (
        proposal.token_mask.shape
        == proposal.design_token_mask.shape
        == proposal.generated_token_mask.shape
    ):
        raise ValueError("proposal token/design/generated masks disagree")
    if proposal.token_mask.ndim != 1 or len(proposal.token_mask) < model_token_count:
        raise ValueError("proposal does not cover the compiled token inventory")
    if proposal.token_center_nm.shape != (len(proposal.token_mask), 3):
        raise ValueError("proposal token centers have an invalid shape")
    if proposal.polymer_atom91_nm.shape != (len(proposal.token_mask), 91, 3):
        raise ValueError("proposal polymer coordinates have an invalid shape")
    if proposal.context_constraint_mode not in {"rigid", "flex"}:
        raise ValueError("proposal has an invalid context constraint mode")
    for name, mask in (
        ("token", proposal.token_mask),
        ("design", proposal.design_token_mask),
        ("generated", proposal.generated_token_mask),
    ):
        if np.asarray(mask).dtype != np.bool_:
            raise TypeError(f"proposal {name} mask must be boolean")
    if not bool(
        np.asarray(proposal.token_mask[:model_token_count], dtype=bool).all()
    ):
        raise ValueError("proposal masks a real compiled token as padding")
    if bool(
        np.asarray(proposal.token_mask[model_token_count:], dtype=bool).any()
    ):
        raise ValueError("proposal token mask marks padded tail tokens as valid")
    if bool(
        np.asarray(proposal.design_token_mask[model_token_count:], dtype=bool).any()
    ):
        raise ValueError("proposal design mask marks padded tail tokens as design")
    if bool(np.any(proposal.generated_token_mask & ~proposal.token_mask)):
        raise ValueError("proposal generated mask contains a padded token")

    # ``proposal`` retains the batch-wide padded token axis, while the
    # authoritative ``CompiledSample`` owns only its unpadded token inventory.
    # Compare like-for-like: shorter rows in a heterogeneous batch must not fail
    # merely because another sample established a larger padding length.
    expected_design = np.zeros(model_token_count, dtype=bool)
    for model_token, raw_rows in enumerate(layout.rows_by_model_token):
        rows = np.asarray(raw_rows, dtype=np.int64)
        design_values = np.asarray(atoms.is_design[rows], dtype=bool)
        if not bool((design_values == design_values[0]).all()):
            raise ValueError("one model token spans design and context rows")
        expected_design[model_token] = bool(design_values[0])
    if not np.array_equal(
        proposal.design_token_mask[:model_token_count], expected_design
    ):
        raise ValueError("proposal design mask does not match CompiledSample")
    expected_generated = (
        expected_design
        if proposal.context_constraint_mode == "rigid"
        else proposal.token_mask[:model_token_count]
    )
    if not np.array_equal(
        proposal.generated_token_mask[:model_token_count], expected_generated
    ):
        raise ValueError(
            f"proposal {proposal.context_constraint_mode} generated mask does not "
            "match CompiledSample"
        )

    source_polymer = _source_coordinates(compiled, proposal.polymer_atom91_nm)
    source_centers = _source_coordinates(compiled, proposal.token_center_nm)
    if proposal.context_constraint_mode == "rigid":
        active = ~np.asarray(atoms.is_design, dtype=bool)
    else:
        active = np.zeros(len(atoms), dtype=bool)
        for atom_uid, coordinate in _flex_context_coordinate_plan(
            compiled,
            proposal,
            source_polymer=source_polymer,
            source_centers=source_centers,
            emit_warnings=True,
        ).items():
            rows = np.where(atoms.atom_uid == int(atom_uid))[0]
            if len(rows) != 1:
                raise ValueError("flex context AtomUID is missing or ambiguous")
            row = int(rows[0])
            active[row] = True
            atoms.coord[row] = coordinate
            atoms.coord_valid[row] = True
    complete_components = _complete_component_uids(compiled)
    head_to_tail_asyms = _head_to_tail_design_protein_asyms(compiled)
    terminal_protein_asym_by_token: dict[int, int] = {}
    design_protein = np.asarray(atoms.is_design, dtype=bool) & (
        np.asarray(atoms.mol_type).astype(str) == "protein"
    )
    for asym_uid in np.unique(atoms.asym_uid[design_protein]):
        chain_rows = np.where(
            design_protein & (atoms.asym_uid == int(asym_uid))
        )[0]
        terminal_position = int(np.asarray(atoms.res_id[chain_rows], dtype=np.int64).max())
        terminal_rows = chain_rows[atoms.res_id[chain_rows] == terminal_position]
        terminal_tokens = np.unique(atoms.token_uid[terminal_rows])
        if len(terminal_tokens) != 1:
            raise ValueError("design protein terminal residue does not map to one token")
        terminal_protein_asym_by_token[int(terminal_tokens[0])] = int(asym_uid)

    for authoring_token in range(authoring_token_count):
        rows = np.where(atoms.token_uid == authoring_token)[0]
        if len(rows) == 0:
            raise AssertionError(authoring_token)
        model_tokens = layout.authoring_token_to_model_tokens[authoring_token]
        if not model_tokens:
            raise AssertionError(authoring_token)
        design = bool(atoms.is_design[rows[0]])
        if not design:
            continue
        modality = str(atoms.mol_type[rows[0]])
        if set(str(value) for value in atoms.mol_type[rows]) != {modality}:
            raise ValueError("one token spans multiple modalities")
        is_carrier = all(
            bool(layout.carrier_token_mask[model_token])
            for model_token in model_tokens
        )
        if is_carrier:
            if len(model_tokens) != len(rows):
                raise ValueError(
                    "carrier authoring token does not expand one model token per row"
                )
            residue_uids = np.unique(atoms.residue_uid[rows])
            if len(residue_uids) != 1 or int(residue_uids[0]) not in (
                compiled.modified_polymer_carriers
            ):
                raise ValueError("carrier token has ambiguous source residue identity")
            for row in rows:
                row = int(row)
                model_token = int(layout.model_token_by_row[row])
                if not bool(proposal.generated_token_mask[model_token]):
                    raise ValueError("design carrier atom is absent from generated mask")
                active[row] = True
                hard_coordinate = bool(
                    atoms.coord_valid[row]
                    and (
                        atoms.token_center_condition[row]
                        and not atoms.motif_coord_condition[row]
                    )
                )
                if not hard_coordinate:
                    atoms.coord[row] = source_centers[model_token].astype(
                        np.float32
                    )
                atoms.coord_valid[row] = True
            continue
        if any(layout.carrier_token_mask[model_token] for model_token in model_tokens):
            raise ValueError("authoring token mixes carrier and ordinary model tokens")
        if len(model_tokens) != 1:
            raise ValueError("ordinary authoring token does not map one-to-one")
        model_token = int(model_tokens[0])
        if modality == "ligand":
            if len(rows) != 1:
                raise ValueError("each ligand token must own exactly one atom row")
            row = int(rows[0])
            active[row] = True
            hard_identity = bool(atoms.identity_condition[row])
            element = str(atoms.element[row]).strip().upper()
            if not hard_identity:
                predicted = proposal.ligand_element[model_token]
                if predicted is None:
                    raise ValueError(
                        f"ligand model token {model_token} has no decoded element"
                    )
                element = str(predicted).upper()
                atoms.element[row] = element
            elif not element:
                raise ValueError("hard ligand identity has no element")
            hard_coordinate = bool(
                atoms.coord_valid[row]
                and (
                    atoms.token_center_condition[row]
                    and not atoms.motif_coord_condition[row]
                )
            )
            if not hard_coordinate:
                atoms.coord[row] = source_centers[model_token].astype(np.float32)
            atoms.coord_valid[row] = True
            continue

        identity = np.asarray(atoms.identity_condition[rows], dtype=bool)
        if identity.any() and not identity.all():
            raise ValueError(
                f"polymer authoring token {authoring_token} has partial hard identity"
            )
        if identity.all():
            residue_names = set(str(value) for value in atoms.res_name[rows])
            if len(residue_names) != 1:
                raise ValueError("hard polymer identity disagrees within one token")
            residue_name = next(iter(residue_names))
        else:
            residue_name = proposal.polymer_res_name[model_token]
            if residue_name is None:
                raise ValueError(
                    f"polymer model token {model_token} has no decoded identity"
                )
        terminal_asym_uid = terminal_protein_asym_by_token.get(authoring_token)
        head_to_tail_terminal = terminal_asym_uid in head_to_tail_asyms
        residue_uids = np.unique(atoms.residue_uid[rows])
        if len(residue_uids) != 1:
            raise ValueError("one polymer token spans multiple residue UIDs")
        residue_uid = int(residue_uids[0])
        if residue_uid in compiled.model_parent_residue:
            if not bool(identity.all()):
                raise ValueError(
                    "a design residue cannot request a free nonstandard identity"
                )
            coordinates = _modified_polymer_token_coordinate_plan(
                compiled,
                proposal,
                model_token,
                rows,
                source_polymer=source_polymer,
                preserve_backbone_coordinates=True,
                emit_warning=True,
            )
            required_rows: set[int] = set()
            for atom_uid, coordinate in coordinates.items():
                matches = rows[atoms.atom_uid[rows] == int(atom_uid)]
                if len(matches) != 1:
                    raise ValueError(
                        "modified-polymer coordinate plan references a missing row"
                    )
                row = int(matches[0])
                if head_to_tail_terminal and str(atoms.atom_name[row]) == "OXT":
                    if bool(
                        atoms.motif_coord_condition[row]
                        or atoms.token_center_condition[row]
                    ):
                        raise ValueError(
                            "head-to-tail closure conflicts with hard terminal OXT"
                        )
                    continue
                required_rows.add(row)
                active[row] = True
                atoms.coord[row] = np.asarray(coordinate, dtype=np.float32)
                atoms.coord_valid[row] = True
            if not required_rows:
                raise ValueError(
                    "modified-polymer design token has no representable proxy atoms"
                )
            center_rows = rows[np.asarray(atoms.centre_atom_mask[rows], dtype=bool)]
            if len(center_rows) != 1 or int(center_rows[0]) not in required_rows:
                raise ValueError(
                    "modified-polymer design projection cannot retain its representative atom"
                )
            continue
        inventory = _INVENTORIES.get(modality, {}).get(str(residue_name))
        if inventory is None:
            raise ValueError(
                f"decoded residue {modality}:{residue_name} has no authoring inventory"
            )
        if (
            modality == "protein"
            and terminal_asym_uid is not None
            and not head_to_tail_terminal
            and "OXT" not in inventory
        ):
            inventory = (*inventory, "OXT")
        omitted_atom_names = omitted_by_token.get(authoring_token, frozenset())
        invalid_omissions = omitted_atom_names.difference(inventory)
        if invalid_omissions:
            raise ValueError(
                "compiled omit-atoms sidecar is incompatible with decoded identity: "
                f"token {authoring_token} does not contain {sorted(invalid_omissions)}"
            )
        inventory = tuple(
            atom_name
            for atom_name in inventory
            if atom_name not in omitted_atom_names
        )
        offset, slot_names = _SLOTS[modality]
        required_rows: set[int] = set()
        for atom_name in inventory:
            matches = rows[atoms.atom_name[rows] == atom_name]
            if len(matches) != 1:
                raise ValueError(
                    f"token {authoring_token} inventory has {len(matches)} rows for {atom_name!r}"
                )
            row = int(matches[0])
            required_rows.add(row)
            active[row] = True
            atoms.res_name[row] = str(residue_name)
            slot = offset + slot_names.index(atom_name)
            hard_coordinate = bool(
                atoms.coord_valid[row]
                and (
                    atoms.token_center_condition[row]
                    and not atoms.motif_coord_condition[row]
                )
            )
            if not hard_coordinate:
                atoms.coord[row] = source_polymer[model_token, slot].astype(
                    np.float32
                )
            atoms.coord_valid[row] = True
        inactive_rows = set(int(value) for value in rows) - required_rows
        for row in inactive_rows:
            if head_to_tail_terminal and str(atoms.atom_name[row]) == "OXT":
                if bool(
                    atoms.motif_coord_condition[row]
                    or atoms.token_center_condition[row]
                ):
                    raise ValueError(
                        "head-to-tail closure conflicts with hard terminal OXT"
                    )
                continue
            if bool(
                atoms.identity_condition[row]
                or atoms.motif_coord_condition[row]
                or atoms.token_center_condition[row]
            ):
                raise ValueError("decoded identity would remove a hard polymer atom")

    output = atoms[active]
    output.coord = np.asarray(output.coord, dtype=np.float32)
    if len(output) == 0 or not bool(np.asarray(output.coord_valid).all()):
        raise ValueError("terminal assembly contains an unresolved output coordinate")
    if not np.isfinite(output.coord).all():
        raise ValueError("terminal assembly contains non-finite coordinates")

    # A partial atom motif is a newly completed ligand and must receive a
    # generated file-local component ID in the writer. Complete type_bond
    # components retain their input component name; SMILES already uses UNL.
    for component_uid in np.unique(
        output.component_uid[output.is_design & (output.mol_type == "ligand")]
    ):
        if int(component_uid) in complete_components:
            continue
        rows = np.where(output.component_uid == component_uid)[0]
        output.res_name[rows] = "UNL"
    return output


def _flex_context_coordinate_plan(
    compiled: CompiledSample,
    proposal: DecodedProposal,
    *,
    source_polymer: np.ndarray | None = None,
    source_centers: np.ndarray | None = None,
    emit_warnings: bool = False,
) -> dict[int, np.ndarray]:
    """Map every representable flex-context atom to its generated coordinate.

    Standard context keeps its exact identity and inventory.  A modified or
    otherwise non-standard polymer keeps its original residue name, uses exact
    generated coordinates for source/proxy intersections, and recovers source
    extras only through the guarded residue-local Kabsch plan.  No retained
    flex row silently falls back to its original source coordinate.
    """

    atoms = compiled.atoms
    layout = build_model_token_layout(
        atoms,
        compiled.modified_polymer_carriers,
    )
    if source_polymer is None:
        source_polymer = _source_coordinates(compiled, proposal.polymer_atom91_nm)
    if source_centers is None:
        source_centers = _source_coordinates(compiled, proposal.token_center_nm)
    result: dict[int, np.ndarray] = {}
    context_rows = np.where(~np.asarray(atoms.is_design, dtype=bool))[0]
    for authoring_token in np.unique(atoms.token_uid[context_rows]):
        authoring_token = int(authoring_token)
        rows = np.where(
            (~np.asarray(atoms.is_design, dtype=bool))
            & (atoms.token_uid == authoring_token)
        )[0]
        if len(rows) == 0:
            raise AssertionError(authoring_token)
        model_tokens = layout.authoring_token_to_model_tokens[authoring_token]
        is_carrier = all(
            bool(layout.carrier_token_mask[model_token])
            for model_token in model_tokens
        )
        if is_carrier:
            if len(model_tokens) != len(rows):
                raise ValueError(
                    "carrier authoring token does not expand one model token per row"
                )
            for row in rows:
                row = int(row)
                model_token = int(layout.model_token_by_row[row])
                if not bool(proposal.generated_token_mask[model_token]):
                    raise ValueError(
                        "flex context carrier atom is absent from generated mask"
                    )
                # Compiler context rows carry motif visibility so the model sees
                # their source geometry.  In flex mode that visibility is not a
                # hard output coordinate constraint: every carrier atom follows
                # its generated atomwise token center.
                result[int(atoms.atom_uid[row])] = source_centers[
                    model_token
                ].astype(np.float32)
            continue
        if any(layout.carrier_token_mask[model_token] for model_token in model_tokens):
            raise ValueError("authoring token mixes carrier and ordinary model tokens")
        if len(model_tokens) != 1:
            raise ValueError("ordinary authoring token does not map one-to-one")
        model_token = int(model_tokens[0])
        if not bool(proposal.generated_token_mask[model_token]):
            raise ValueError("flex context token is absent from generated mask")
        modality_values = {str(value) for value in atoms.mol_type[rows]}
        if len(modality_values) != 1:
            raise ValueError("one flex context token spans multiple modalities")
        modality = next(iter(modality_values))
        if modality == "ligand":
            if len(rows) != 1:
                raise ValueError("each flex context ligand token must own one atom")
            row = int(rows[0])
            result[int(atoms.atom_uid[row])] = source_centers[model_token].astype(
                np.float32
            )
            continue

        inventory = _INVENTORIES.get(modality)
        if inventory is None:
            raise ValueError(f"unsupported flex context modality {modality!r}")
        residue_uids = np.unique(atoms.residue_uid[rows])
        residue_names = {str(value) for value in atoms.res_name[rows]}
        if len(residue_uids) != 1 or len(residue_names) != 1:
            raise ValueError("one flex context token has ambiguous residue identity")
        residue_uid = int(residue_uids[0])
        residue_name = next(iter(residue_names))
        parent_name = compiled.model_parent_residue.get(residue_uid)
        if parent_name is not None:
            modified_coordinates = _modified_polymer_token_coordinate_plan(
                compiled,
                proposal,
                model_token,
                rows,
                source_polymer=source_polymer,
                preserve_backbone_coordinates=False,
                emit_warning=emit_warnings,
            )
            if not modified_coordinates:
                raise ValueError(
                    "flex modified-polymer token has no representable proxy atoms"
                )
            result.update(modified_coordinates)
            center_rows = rows[np.asarray(atoms.centre_atom_mask[rows], dtype=bool)]
            retained_uids = set(modified_coordinates)
            if (
                len(center_rows) != 1
                or int(atoms.atom_uid[int(center_rows[0])]) not in retained_uids
            ):
                raise ValueError(
                    "flex modified-polymer projection cannot retain its representative atom"
                )
            continue
        if residue_name in inventory:
            allowed_names = set(inventory[residue_name])
        else:
            allowed_names = set(_COMMON_POLYMER_ATOMS[modality])
        if modality == "protein" and bool(np.any(atoms.atom_name[rows] == "OXT")):
            allowed_names.add("OXT")

        offset, slot_names = _SLOTS[modality]
        retained_rows: list[int] = []
        for row in rows:
            atom_name = str(atoms.atom_name[row])
            if atom_name not in allowed_names or atom_name not in slot_names:
                continue
            slot = offset + slot_names.index(atom_name)
            result[int(atoms.atom_uid[row])] = source_polymer[
                model_token, slot
            ].astype(np.float32)
            retained_rows.append(int(row))
        if not retained_rows:
            raise ValueError("flex context token has no representable proxy atoms")
        center_rows = rows[np.asarray(atoms.centre_atom_mask[rows], dtype=bool)]
        if len(center_rows) != 1 or int(center_rows[0]) not in retained_rows:
            raise ValueError(
                "flex context projection cannot retain its representative atom"
            )
    return result


def _merge_generated_polymer_bonds(
    compiled: CompiledSample,
    atoms: struc.AtomArray,
) -> None:
    edges = _bond_map(atoms)
    carrier_residue_uids = frozenset(
        int(value) for value in compiled.modified_polymer_carriers
    )
    carrier_row_mask = np.isin(
        np.asarray(atoms.residue_uid, dtype=np.int64),
        tuple(carrier_residue_uids),
    )
    design_polymer = (
        atoms.is_design
        & (atoms.mol_type != "ligand")
        & ~carrier_row_mask
    )
    carrier_authoring_tokens_by_asym: dict[int, frozenset[int]] = {}
    for asym_uid in np.unique(compiled.atoms.asym_uid):
        scoped = (
            np.asarray(compiled.atoms.asym_uid, dtype=np.int64) == int(asym_uid)
        ) & np.isin(
            np.asarray(compiled.atoms.residue_uid, dtype=np.int64),
            tuple(carrier_residue_uids),
        )
        carrier_authoring_tokens_by_asym[int(asym_uid)] = frozenset(
            int(value) for value in np.unique(compiled.atoms.token_uid[scoped])
        )
    for asym_uid in np.unique(atoms.asym_uid[design_polymer]):
        rows = np.where(design_polymer & (atoms.asym_uid == asym_uid))[0]
        if not len(rows):
            continue
        canonical_rows: list[int] = []
        canonical_residue_names: list[str] = []
        canonical_atom_names: list[str] = []
        modified_proxy_names: dict[int, str] = {}
        for residue_uid in np.unique(atoms.residue_uid[rows]):
            residue_uid = int(residue_uid)
            proxy_residue_name = compiled.model_parent_residue.get(residue_uid)
            if proxy_residue_name is None:
                continue
            residue_rows = rows[atoms.residue_uid[rows] == residue_uid]
            raw_names = {str(value) for value in atoms.res_name[residue_rows]}
            modalities = {str(value) for value in atoms.mol_type[residue_rows]}
            if len(raw_names) != 1 or len(modalities) != 1:
                raise ValueError("modified polymer has ambiguous bond identity")
            local_names = proxy_atom_rows(
                modality=next(iter(modalities)),
                residue_name=next(iter(raw_names)),
                proxy_residue_name=proxy_residue_name,
                source_atom_names=[str(atoms.atom_name[row]) for row in residue_rows],
            )
            modified_proxy_names.update(
                {
                    int(residue_rows[local_row]): atom_name
                    for local_row, atom_name in local_names.items()
                }
            )
        occupied: set[tuple[int, str]] = set()
        for row in rows:
            row = int(row)
            modality = str(atoms.mol_type[row])
            residue_uid = int(atoms.residue_uid[row])
            raw_residue_name = str(atoms.res_name[row])
            proxy_residue_name = compiled.model_parent_residue.get(residue_uid)
            if proxy_residue_name is None:
                atom_name = str(atoms.atom_name[row])
                inventory = _INVENTORIES[modality].get(raw_residue_name)
                if inventory is None or (
                    atom_name not in inventory
                    and not (modality == "protein" and atom_name == "OXT")
                ):
                    continue
                model_residue_name = raw_residue_name
            else:
                atom_name = modified_proxy_names.get(row)
                if atom_name is None:
                    if modality == "protein" and str(atoms.atom_name[row]) == "OXT":
                        atom_name = "OXT"
                    else:
                        continue
                model_residue_name = proxy_residue_name
            key = (residue_uid, atom_name)
            if key in occupied:
                raise ValueError(
                    "generated polymer contains duplicate proxy atom aliases"
                )
            occupied.add(key)
            canonical_rows.append(row)
            canonical_residue_names.append(model_residue_name)
            canonical_atom_names.append(atom_name)
        if not canonical_rows:
            raise ValueError("generated polymer has no canonical bond inventory")
        canonical = atoms[np.asarray(canonical_rows, dtype=np.int64)]
        canonical.res_name = np.asarray(canonical_residue_names, dtype="<U16")
        canonical.atom_name = np.asarray(canonical_atom_names, dtype="<U32")
        try:
            local = struc.connect_via_residue_names(
                canonical, inter_residue=True
            )
        except Exception as exc:
            raise ValueError("cannot construct canonical generated-polymer bonds") from exc
        for local_i, local_j, bond_type in local.as_array():
            output_i = int(canonical_rows[int(local_i)])
            output_j = int(canonical_rows[int(local_j)])
            token_i = int(atoms.token_uid[output_i])
            token_j = int(atoms.token_uid[output_j])
            if int(atoms.residue_uid[output_i]) != int(
                atoms.residue_uid[output_j]
            ) and any(
                min(token_i, token_j) < carrier_token < max(token_i, token_j)
                for carrier_token in carrier_authoring_tokens_by_asym.get(
                    int(asym_uid), frozenset()
                )
            ):
                # Excluding a carrier from the template view must not make its
                # two polymer neighbours look adjacent and synthesize a bridge.
                continue
            _merge_edge(
                edges,
                output_i,
                output_j,
                int(bond_type),
            )
    _set_bond_map(atoms, edges)


def _replace_decoded_ligand_bonds(
    compiled: CompiledSample,
    atoms: struc.AtomArray,
    proposal: DecodedProposal,
) -> None:
    layout = build_model_token_layout(
        compiled.atoms,
        compiled.modified_polymer_carriers,
    )
    edges = _bond_map(atoms)
    decoded_rows: dict[int, set[int]] = {}
    for molecule_uid in proposal.decoded_ligand_molecule_uids:
        rows = set(
            np.where(
                atoms.is_design
                & (atoms.mol_type == "ligand")
                & (atoms.molecule_uid == int(molecule_uid))
            )[0].tolist()
        )
        if not rows:
            raise ValueError("decoded ligand molecule has no final atom rows")
        decoded_rows[int(molecule_uid)] = rows
        for key in list(edges):
            if key[0] in rows and key[1] in rows:
                del edges[key]

    token_to_row: dict[int, int] = {}
    for row in np.where(atoms.mol_type == "ligand")[0]:
        row = int(row)
        atom_uid = int(atoms.atom_uid[row])
        try:
            model_token = int(layout.model_token_by_atom_uid[atom_uid])
        except KeyError as exc:
            raise ValueError("final ligand AtomUID is absent from model layout") from exc
        previous = token_to_row.get(model_token)
        if previous is not None and previous != row:
            raise ValueError("ligand model token maps to multiple final atoms")
        token_to_row[model_token] = row
    for token_i, token_j, bond_type in proposal.ligand_bonds:
        if int(token_i) not in token_to_row or int(token_j) not in token_to_row:
            raise ValueError("decoded ligand bond endpoint is absent")
        row_i, row_j = token_to_row[int(token_i)], token_to_row[int(token_j)]
        molecule_i, molecule_j = int(atoms.molecule_uid[row_i]), int(atoms.molecule_uid[row_j])
        if molecule_i != molecule_j or molecule_i not in decoded_rows:
            raise ValueError("decoded ligand edge crosses its planned molecule")
        _merge_edge(edges, row_i, row_j, int(bond_type))
    _set_bond_map(atoms, edges)


def _uid_bond_orders(atoms: struc.AtomArray) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    if atoms.bonds is None:
        return result
    for row_i, row_j, bond_type in atoms.bonds.as_array():
        key = tuple(
            sorted((int(atoms.atom_uid[int(row_i)]), int(atoms.atom_uid[int(row_j)])))
        )
        previous = result.get(key)
        if previous is not None and previous != int(bond_type):
            raise ValueError("authoritative hard graph contains conflicting orders")
        result[key] = int(bond_type)
    return result


def _join_required_bond_order(
    previous: int, order: int, key: tuple[int, int]
) -> int:
    """Retain explicit aromatic subtype when compatible hard facts overlap."""
    if previous == order:
        return order
    aromatic = int(struc.BondType.AROMATIC)
    specific = {
        int(struc.BondType.AROMATIC_SINGLE),
        int(struc.BondType.AROMATIC_DOUBLE),
        int(struc.BondType.AROMATIC_TRIPLE),
    }
    if previous == aromatic and order in specific:
        return order
    if order == aromatic and previous in specific:
        return previous
    raise ValueError(f"hard bond order conflict for AtomUID pair {key}")


def _reconcile_hard_graph(
    compiled: CompiledSample,
    atoms: struc.AtomArray,
) -> None:
    source_orders = _uid_bond_orders(compiled.atoms)
    required: dict[tuple[int, int], int] = {}
    carrier_atom_uids_by_residue: dict[int, frozenset[int]] = {}
    for residue_uid in compiled.modified_polymer_carriers:
        rows = np.where(compiled.atoms.residue_uid == int(residue_uid))[0]
        if not len(rows):
            raise ValueError("modified-polymer carrier disappeared from source atoms")
        carrier_atom_uids_by_residue[int(residue_uid)] = frozenset(
            int(value) for value in compiled.atoms.atom_uid[rows]
        )
    def add_required(uid_a: int, uid_b: int, order: int) -> None:
        key = tuple(sorted((int(uid_a), int(uid_b))))
        previous = required.get(key)
        required[key] = (
            int(order)
            if previous is None
            else _join_required_bond_order(previous, int(order), key)
        )

    for condition in compiled.topology_conditions:
        if not condition.required_output:
            continue
        raw_order = str(condition.order).strip().lower()
        order = _ORDER_TO_BIOTITE.get(raw_order)
        if order is None and raw_order in {"any", "unknown"}:
            order = int(struc.BondType.ANY)
        if order is None:
            raise ValueError(f"unsupported required bond order {condition.order!r}")
        add_required(condition.atom_uid_a, condition.atom_uid_b, order)
    for policy in compiled.edge_policies:
        if not policy.required_output:
            continue
        for uid_a, uid_b in policy.hard_positive_edges:
            key = tuple(sorted((int(uid_a), int(uid_b))))
            order = source_orders.get(key)
            if order is None:
                raise ValueError("hard edge policy has no authoritative bond order")
            add_required(uid_a, uid_b, order)

    complete_scopes: list[
        tuple[list[int], tuple[tuple[int, int], ...]]
    ] = []
    for policy in compiled.edge_policies:
        if not policy.complete_graph:
            continue
        if policy.component_uid is not None:
            rows = np.where(atoms.component_uid == int(policy.component_uid))[0]
        else:
            rows = np.where(atoms.molecule_uid == int(policy.molecule_uid))[0]
        if not len(rows):
            raise ValueError("complete hard graph scope disappeared after decode")
        uids = [int(value) for value in atoms.atom_uid[rows]]
        complete_scopes.append((uids, policy.hard_positive_edges))

    exact_carrier_orders = {
        int(struc.BondType.SINGLE),
        int(struc.BondType.DOUBLE),
        int(struc.BondType.TRIPLE),
        int(struc.BondType.AROMATIC_SINGLE),
        int(struc.BondType.AROMATIC_DOUBLE),
        int(struc.BondType.AROMATIC_TRIPLE),
        int(struc.BondType.AROMATIC),
    }
    output_uids = frozenset(int(value) for value in atoms.atom_uid)
    for residue_uid, carrier_uids in carrier_atom_uids_by_residue.items():
        if not carrier_uids.issubset(output_uids):
            raise ValueError(
                f"modified-polymer carrier {residue_uid} lost an output atom"
            )
        positive_edges: list[tuple[int, int]] = []
        for (uid_a, uid_b), order in source_orders.items():
            if uid_a not in carrier_uids or uid_b not in carrier_uids:
                continue
            if int(order) not in exact_carrier_orders:
                if int(order) == int(struc.BondType.QUADRUPLE):
                    raise ValueError(
                        "modified-polymer carrier bonds cannot use QUADRUPLE order"
                    )
                raise ValueError(
                    "modified-polymer carrier internal bonds require exact order"
                )
            positive_edges.append((uid_a, uid_b))
            add_required(uid_a, uid_b, int(order))
        complete_scopes.append(
            (sorted(carrier_uids), tuple(sorted(positive_edges)))
        )

    reconcile_required_edges(
        atoms,
        ((uid_a, uid_b, order) for (uid_a, uid_b), order in sorted(required.items())),
        complete_components=complete_scopes,
    )

def _ligand_mode(
    compiled: CompiledSample,
    chain_id: str,
) -> str:
    condition = compiled.request.task.condition
    if not isinstance(condition, Mapping):
        return "free"
    if chain_id in set(condition.get("backbone", []) or []):
        return "backbone"
    ligand_config = condition.get("ligand", {}) or {}
    if not isinstance(ligand_config, Mapping):
        return "free"
    entry = ligand_config.get(chain_id)
    if not isinstance(entry, Mapping):
        return "free"
    return str(entry.get("mode", "free")).strip().lower()


def _required_ligand_endpoint_rows(
    compiled: CompiledSample,
    atoms: struc.AtomArray,
) -> Mapping[int, frozenset[int]]:
    """Return hard/explicit ligand endpoint rows keyed by molecule UID."""

    uid_to_row = {int(uid): row for row, uid in enumerate(atoms.atom_uid)}
    protected: dict[int, set[int]] = {}
    endpoint_pairs: list[tuple[int, int]] = []
    for condition in compiled.topology_conditions:
        if condition.required_output:
            endpoint_pairs.append(
                (int(condition.atom_uid_a), int(condition.atom_uid_b))
            )
    for policy in compiled.edge_policies:
        if policy.required_output:
            endpoint_pairs.extend(
                (int(uid_a), int(uid_b))
                for uid_a, uid_b in policy.hard_positive_edges
            )
    for uid_a, uid_b in endpoint_pairs:
        if uid_a not in uid_to_row or uid_b not in uid_to_row:
            raise ValueError("hard ligand endpoint is absent from assembly output")
        row_a, row_b = uid_to_row[uid_a], uid_to_row[uid_b]
        ligand_a = str(atoms.mol_type[row_a]) == "ligand"
        ligand_b = str(atoms.mol_type[row_b]) == "ligand"
        for row, is_ligand in ((row_a, ligand_a), (row_b, ligand_b)):
            if is_ligand:
                molecule_uid = int(atoms.molecule_uid[row])
                protected.setdefault(molecule_uid, set()).add(int(row))
    return MappingProxyType(
        {
            molecule_uid: frozenset(sorted(rows))
            for molecule_uid, rows in sorted(protected.items())
        }
    )


def _ligand_fragment_components(
    atoms: struc.AtomArray,
    rows: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    row_set = {int(row) for row in np.asarray(rows, dtype=np.int64).tolist()}
    adjacency = {row: set() for row in row_set}
    for (row_i, row_j), _ in _bond_map(atoms).items():
        if row_i in row_set and row_j in row_set:
            adjacency[row_i].add(row_j)
            adjacency[row_j].add(row_i)
    components: list[tuple[int, ...]] = []
    remaining = set(row_set)
    while remaining:
        start = min(remaining)
        stack = [start]
        remaining.remove(start)
        component: list[int] = []
        while stack:
            row = stack.pop()
            component.append(row)
            for neighbor in sorted(adjacency[row], reverse=True):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda component: component[0]))


def _validate_generated_ligand_component(
    compiled: CompiledSample,
    proposal: DecodedProposal,
    atoms: struc.AtomArray,
    molecule_uid: int,
    rows: Sequence[int],
) -> Any:
    """Sanitize one candidate fragment using the proposal's private charges."""

    source = proposal.private_ligand_molecules.get(int(molecule_uid))
    if source is not None:
        source_uids = proposal.private_ligand_atom_uids.get(int(molecule_uid))
        if source_uids is None:
            source_uids = tuple(
                int(compiled.atoms.atom_uid[row])
                for row in np.where(
                    (compiled.atoms.molecule_uid == int(molecule_uid))
                    & compiled.atoms.is_design
                    & (compiled.atoms.mol_type == "ligand")
                )[0]
            )
        if len(source_uids) != source.GetNumAtoms():
            raise ValueError("private ligand AtomUID map disagrees with molecule")
        source_index = {int(uid): index for index, uid in enumerate(source_uids)}
        retained_uids = [int(atoms.atom_uid[row]) for row in rows]
        if any(uid not in source_index for uid in retained_uids):
            raise ValueError("private ligand AtomUID map lost a retained atom")
        charges = [
            source.GetAtomWithIdx(source_index[uid]).GetFormalCharge()
            for uid in retained_uids
        ]
    else:
        charge_logits = proposal.private_ligand_charge_logits
        if charge_logits is None or charge_logits.ndim != 2:
            raise ValueError("generated ligand private chemistry has no charge logits")
        if charge_logits.shape[1] != len(LIGAND_FORMAL_CHARGES):
            raise ValueError("generated ligand private charge logits have invalid shape")
        layout = build_model_token_layout(
            compiled.atoms,
            compiled.modified_polymer_carriers,
        )
        charges = []
        for row in rows:
            atom_uid = int(atoms.atom_uid[row])
            try:
                model_token = int(layout.model_token_by_atom_uid[atom_uid])
            except KeyError as exc:
                raise ValueError(
                    "generated ligand private charge AtomUID is absent from model layout"
                ) from exc
            if not 0 <= model_token < charge_logits.shape[0]:
                raise ValueError("generated ligand private charge token is out of range")
            charge_class = int(np.argmax(charge_logits[model_token]))
            charges.append(int(LIGAND_FORMAL_CHARGES[charge_class]))
    return private_molecule_from_authoring_graph(atoms, rows, charges)


def _ligand_fragment_salvage(
    compiled: CompiledSample,
    atoms: struc.AtomArray,
    proposal: DecodedProposal | None = None,
) -> tuple[struc.AtomArray, Mapping[int, Mapping[str, Any]]]:
    """Select one safe generated ligand fragment without inventing bonds."""

    ligand_rows = np.where(np.asarray(atoms.mol_type) == "ligand")[0]
    endpoint_rows = _required_ligand_endpoint_rows(compiled, atoms)
    keep = np.ones(len(atoms), dtype=bool)
    metadata: dict[int, Mapping[str, Any]] = {}
    for molecule_uid in sorted(
        int(value) for value in np.unique(atoms.molecule_uid[ligand_rows])
    ):
        rows = np.where(
            (np.asarray(atoms.molecule_uid) == molecule_uid)
            & (np.asarray(atoms.mol_type) == "ligand")
        )[0]
        rows = np.asarray(rows, dtype=np.int64)
        if not bool(np.asarray(atoms.is_design[rows], dtype=bool).all()):
            continue
        original_uids = tuple(int(value) for value in atoms.atom_uid[rows])
        components = _ligand_fragment_components(atoms, rows)
        chain_ids = {str(value) for value in atoms.chain_id[rows]}
        if len(chain_ids) != 1:
            raise ValueError("ligand molecule spans multiple chains")
        mode = _ligand_mode(compiled, next(iter(chain_ids)))
        variable = mode in {"free", "atom_motif"}
        protected_rows = set(endpoint_rows.get(molecule_uid, ()))
        protected_rows.update(
            int(row)
            for row in rows
            if bool(atoms.identity_condition[row])
            or bool(atoms.motif_coord_condition[row])
            or bool(atoms.token_center_condition[row])
        )
        selected_fragment_index: int | None = None
        if variable:
            containing = [
                index
                for index, component in enumerate(components)
                if protected_rows.intersection(component)
                and protected_rows.issubset(component)
            ]
            if protected_rows and len(containing) != 1:
                raise ValueError("protected ligand atoms span disconnected fragments")
            ranked = (
                containing
                if containing
                else sorted(
                    range(len(components)),
                    key=lambda index: (-len(components[index]), components[index][0]),
                )
            )
            if proposal is not None and len(components) > 1:
                selected_fragment_index = None
                for index in ranked:
                    candidate_rows = tuple(components[index])
                    try:
                        _validate_generated_ligand_component(
                            compiled,
                            proposal,
                            atoms,
                            molecule_uid,
                            candidate_rows,
                        )
                    except ValueError:
                        continue
                    selected_fragment_index = index
                    break
                if selected_fragment_index is None:
                    raise ValueError(
                        "generated ligand has no chemically valid retained fragment"
                    )
            elif containing:
                selected_fragment_index = containing[0]
            else:
                selected_fragment_index = max(
                    range(len(components)),
                    key=lambda index: (len(components[index]), -components[index][0]),
                )
            selected_rows = set(components[selected_fragment_index])
            keep[rows] = False
            keep[np.asarray(sorted(selected_rows), dtype=np.int64)] = True
        else:
            selected_rows = set(int(row) for row in rows)
        retained_rows = tuple(sorted(selected_rows))
        retained_uids = tuple(int(value) for value in atoms.atom_uid[list(retained_rows)])
        fragment_uids = tuple(
            tuple(int(value) for value in atoms.atom_uid[list(component)])
            for component in components
        )
        metadata[molecule_uid] = MappingProxyType(
            {
                "original_atom_count": len(rows),
                "retained_atom_count": len(retained_rows),
                "fragment_count": len(components),
                "salvaged": bool(len(retained_rows) < len(rows)),
                "original_atom_uids": original_uids,
                "retained_atom_uids": retained_uids,
                "dropped_atom_uids": tuple(
                    uid for uid in original_uids if uid not in set(retained_uids)
                ),
                "fragment_atom_uids": fragment_uids,
                "selected_fragment_index": selected_fragment_index,
                "protected_atom_uids": tuple(
                    int(atoms.atom_uid[row]) for row in sorted(protected_rows)
                ),
                "mode": mode,
            }
        )
    if bool((~keep).any()):
        atoms = atoms[keep]
    return atoms, MappingProxyType(metadata)


def _renumber_output_tokens(atoms: struc.AtomArray) -> None:
    """Make surviving token UIDs contiguous in their original row order."""

    mapping: dict[int, int] = {}
    next_uid = 0
    normalized: list[int] = []
    for raw_uid in np.asarray(atoms.token_uid).tolist():
        old_uid = int(raw_uid)
        if old_uid not in mapping:
            mapping[old_uid] = next_uid
            next_uid += 1
        normalized.append(mapping[old_uid])
    atoms.token_uid = np.asarray(normalized, dtype=np.int64)



def _assert_rigid_context(
    compiled: CompiledSample,
    output: struc.AtomArray,
) -> None:
    original = compiled.atoms
    output_uid_to_row = {int(uid): row for row, uid in enumerate(output.atom_uid)}
    context_rows = np.where(~original.is_design)[0]
    categories = original.get_annotation_categories()
    for row in context_rows:
        uid = int(original.atom_uid[row])
        if uid not in output_uid_to_row:
            raise AssertionError("rigid context atom was removed")
        final_row = output_uid_to_row[uid]
        if not np.array_equal(original.coord[row], output.coord[final_row]):
            raise AssertionError("rigid context coordinate changed during assembly")
        for category in categories:
            if original.get_annotation(category)[row] != output.get_annotation(category)[final_row]:
                raise AssertionError(
                    f"rigid context annotation {category!r} changed during assembly"
                )
    original_orders = _uid_bond_orders(original)
    output_orders = _uid_bond_orders(output)
    context_uids = set(int(value) for value in original.atom_uid[context_rows])
    original_context_edges = {
        key: value
        for key, value in original_orders.items()
        if key[0] in context_uids and key[1] in context_uids
    }
    output_context_edges = {
        key: value
        for key, value in output_orders.items()
        if key[0] in context_uids and key[1] in context_uids
    }
    if original_context_edges != output_context_edges:
        raise AssertionError("rigid context topology changed during assembly")


def _assert_flex_context(
    compiled: CompiledSample,
    proposal: DecodedProposal,
    output: struc.AtomArray,
) -> None:
    original = compiled.atoms
    expected = _flex_context_coordinate_plan(compiled, proposal)
    output_uid_to_row = {int(uid): row for row, uid in enumerate(output.atom_uid)}
    output_context_uids = set(
        int(value) for value in output.atom_uid[~np.asarray(output.is_design, dtype=bool)]
    )
    if output_context_uids != set(expected):
        raise AssertionError("flex context output inventory differs from its proxy plan")
    original_uid_to_row = {int(uid): row for row, uid in enumerate(original.atom_uid)}
    for uid, coordinate in expected.items():
        original_row = original_uid_to_row[int(uid)]
        final_row = output_uid_to_row[int(uid)]
        if not np.array_equal(output.coord[final_row], coordinate):
            raise AssertionError("flex context did not use the generated coordinate")
        if not bool(output.coord_valid[final_row]):
            raise AssertionError("flex context generated coordinate is not marked valid")
        for category in original.get_annotation_categories():
            if category == "coord_valid":
                continue
            if (
                original.get_annotation(category)[original_row]
                != output.get_annotation(category)[final_row]
            ):
                raise AssertionError(
                    f"flex context annotation {category!r} changed during assembly"
                )

    original_orders = _uid_bond_orders(original)
    output_orders = _uid_bond_orders(output)
    expected_uids = set(expected)
    original_context_edges = {
        key: value
        for key, value in original_orders.items()
        if key[0] in expected_uids and key[1] in expected_uids
    }
    output_context_edges = {
        key: value
        for key, value in output_orders.items()
        if key[0] in expected_uids and key[1] in expected_uids
    }
    if original_context_edges != output_context_edges:
        raise AssertionError("flex context topology changed during assembly")


def _private_ligand_payloads(
    compiled: CompiledSample,
    proposal: DecodedProposal,
    atoms: struc.AtomArray,
) -> tuple[Mapping[int, Any], Mapping[int, str]]:
    """Build optional OpenMM-facing chemistry without changing public atoms."""

    condition = compiled.request.task.condition
    ligand_config = condition.get("ligand", {}) if isinstance(condition, Mapping) else {}
    if ligand_config is None:
        ligand_config = {}
    if not isinstance(ligand_config, Mapping):
        raise ValueError("compiled ligand condition must remain a mapping")
    backbone = condition.get("backbone", []) if isinstance(condition, Mapping) else []
    backbone_chains = set(backbone or [])
    molecules: dict[int, Any] = {}
    errors: dict[int, str] = {}
    charge_logits = proposal.private_ligand_charge_logits
    layout = build_model_token_layout(
        compiled.atoms,
        compiled.modified_polymer_carriers,
    )

    design_ligand = atoms.is_design & (atoms.mol_type == "ligand")
    for molecule_uid in np.unique(atoms.molecule_uid[design_ligand]):
        molecule_uid = int(molecule_uid)
        rows = np.where(design_ligand & (atoms.molecule_uid == molecule_uid))[0]
        chains = set(str(value) for value in atoms.chain_id[rows])
        if len(chains) != 1:
            errors[molecule_uid] = "one private ligand payload spans multiple chains"
            continue
        chain = next(iter(chains))
        entry = ligand_config.get(chain)
        mode = "backbone" if chain in backbone_chains else "free"
        if entry is not None:
            if not isinstance(entry, Mapping):
                errors[molecule_uid] = "compiled ligand mode is not a mapping"
                continue
            mode = str(entry.get("mode"))

        try:
            if mode == "smiles":
                smiles = entry.get("smiles") if isinstance(entry, Mapping) else None
                if not isinstance(smiles, str):
                    raise ValueError("SMILES private chemistry has no source string")
                source = Chem.MolFromSmiles(smiles)
                if source is None:
                    raise ValueError("SMILES private chemistry cannot be reparsed")
                source = Chem.RemoveHs(source)
                if source.GetNumAtoms() != len(rows):
                    raise ValueError("SMILES private chemistry changed heavy-atom count")
                expected = [str(atoms.element[row]).upper() for row in rows]
                observed = [atom.GetSymbol().upper() for atom in source.GetAtoms()]
                if observed != expected:
                    raise ValueError("SMILES private atom order disagrees with authoring rows")
                molecules[molecule_uid] = private_molecule_from_trusted_graph(
                    atoms, rows, source
                )
                continue
            elif molecule_uid in proposal.private_ligand_molecules:
                source = proposal.private_ligand_molecules[molecule_uid]
                source_uids = proposal.private_ligand_atom_uids.get(molecule_uid)
                if source_uids is None:
                    source_uids = tuple(
                        int(compiled.atoms.atom_uid[row])
                        for row in np.where(
                            (compiled.atoms.molecule_uid == molecule_uid)
                            & compiled.atoms.is_design
                            & (compiled.atoms.mol_type == "ligand")
                        )[0]
                    )
                if source.GetNumAtoms() != len(source_uids):
                    raise ValueError("obabel private molecule changed atom count")
                source_index = {int(uid): index for index, uid in enumerate(source_uids)}
                retained_uids = [int(atoms.atom_uid[row]) for row in rows]
                if any(uid not in source_index for uid in retained_uids):
                    raise ValueError("private ligand AtomUID map lost a retained atom")
                charges = [
                    source.GetAtomWithIdx(source_index[uid]).GetFormalCharge()
                    for uid in retained_uids
                ]
                molecules[molecule_uid] = private_molecule_from_authoring_graph(
                    atoms, rows, charges
                )
                continue
            elif mode in {"type_bond", "ccd"}:
                component = compiled.trusted_ligand_components.get(molecule_uid)
                if component is None:
                    raise ValueError("hard ligand lacks matched trusted CCD chemistry")
                component_names = {
                    str(atoms.res_name[int(row)]).strip().upper() for row in rows
                }
                if len(component_names) != 1:
                    raise ValueError("hard ligand component identity is ambiguous")
                source = trusted_ccd_private_molecule(
                    atoms, rows, next(iter(component_names)), component
                )
                molecules[molecule_uid] = private_molecule_from_trusted_graph(
                    atoms, rows, source
                )
                continue
            else:
                if charge_logits is None:
                    raise ValueError("model private chemistry has no charge logits")
                if charge_logits.ndim != 2 or charge_logits.shape[1] != len(
                    LIGAND_FORMAL_CHARGES
                ):
                    raise ValueError("model private charge logits have invalid shape")
                charges = []
                for row in rows:
                    atom_uid = int(atoms.atom_uid[row])
                    try:
                        model_token = int(layout.model_token_by_atom_uid[atom_uid])
                    except KeyError as exc:
                        raise ValueError(
                            "model private charge AtomUID is absent from model layout"
                        ) from exc
                    if not 0 <= model_token < charge_logits.shape[0]:
                        raise ValueError("model private charge token is out of range")
                    charge_class = int(np.argmax(charge_logits[model_token]))
                    charges.append(int(LIGAND_FORMAL_CHARGES[charge_class]))
            molecules[molecule_uid] = private_molecule_from_authoring_graph(
                atoms,
                rows,
                charges,
            )
        except Exception as exc:
            prefix = "motif_bond_reconciliation: " if mode == "atom_motif" else ""
            errors[molecule_uid] = f"{prefix}{type(exc).__name__}: {exc}"
    return MappingProxyType(molecules), MappingProxyType(errors)


def assemble_candidate(
    compiled: CompiledSample,
    proposal: DecodedProposal,
) -> AssemblyResult:
    """Return one reconciled, source-frame authoritative terminal candidate."""

    output = _activate_and_fill(compiled, proposal)
    _merge_generated_polymer_bonds(compiled, output)
    _replace_decoded_ligand_bonds(compiled, output, proposal)
    _reconcile_hard_graph(compiled, output)
    output, fragment_metadata = _ligand_fragment_salvage(compiled, output, proposal)
    cyclic_asyms = _head_to_tail_design_protein_asyms(compiled)
    if cyclic_asyms:
        cyclic_chain_ids_list: list[str] = []
        for asym_uid in sorted(cyclic_asyms):
            rows = np.where(output.asym_uid == int(asym_uid))[0]
            if not len(rows):
                raise ValueError("cyclic product asym disappeared during assembly")
            cyclic_chain_ids_list.append(str(output.chain_id[int(rows[0])]))
        cyclic_chain_ids = tuple(dict.fromkeys(cyclic_chain_ids_list))
        output = normalize_cyclic_product(output, cyclic_chain_ids)
    output.coord = np.asarray(output.coord, dtype=np.float32)
    if not bool(output.coord_valid.all()) or not np.isfinite(output.coord).all():
        raise ValueError("assembled terminal candidate is not fully resolved")
    if "charge" in output.get_annotation_categories():
        raise AssertionError("public formal charge leaked into authoring output")
    if proposal.context_constraint_mode == "rigid":
        _assert_rigid_context(compiled, output)
    else:
        _assert_flex_context(compiled, proposal, output)
    _renumber_output_tokens(output)
    if compiled.antibody_metadata:
        from ..antibody.compiler import attach_candidate_annotations
        attach_candidate_annotations(compiled, output)
    validate_atom_array(output, phase="tokenized")
    private_molecules, private_errors = _private_ligand_payloads(
        compiled,
        proposal,
        output,
    )
    return AssemblyResult(
        atoms=output,
        private_ligand_charge_logits=proposal.private_ligand_charge_logits,
        private_ligand_molecules=private_molecules,
        private_ligand_errors=private_errors,
        ligand_fragment_metadata=fragment_metadata,
    )


__all__ = ["AssemblyResult", "assemble_candidate", "normalize_cyclic_product"]
