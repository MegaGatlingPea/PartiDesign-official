"""Source-preserving modified-polymer projection for structure scoring.

Only this deterministic inference operation and its direct chemistry helpers
are retained from PartiDesign. It never samples training tasks, crops structures,
or augments coordinates. Source atom/bond coverage remains authoritative.
"""
from typing import Any, Mapping
import biotite.structure as struc
import numpy as np
from .structure import STD_RESIDUES, AtomArrayTokenizer
from src.common.chemistry.ccd import ccd_polymer_components
from src.common.constants import (ID_TO_TOKEN_TYPE32, PROTEIN_RES_ATOMS, TOKEN_TYPE32,
    atom_slot_index, normalise_atom_name, normalise_res_name)
_POLYMER_MOL_TYPES = frozenset(("protein", "rna", "dna"))
_HYDROGEN_ELEMENTS = frozenset(("H", "D", "T"))
_CCD_BOND_ORDER = {"single": int(struc.BondType.SINGLE), "double": int(struc.BondType.DOUBLE),
 "triple": int(struc.BondType.TRIPLE), "aromatic": int(struc.BondType.AROMATIC)}

def _projection_ccd_components():
    return ccd_polymer_components()

def _normalised_literal_name(value: object) -> str:
    return str(value).strip().upper().replace("*", "'")

def _resolve_ccd_atom_id_with_reason(
    component: Mapping[str, Any], source_name: object
) -> tuple[str | None, str]:
    """Resolve a source atom and distinguish unknown from ambiguous aliases."""

    name = _normalised_literal_name(source_name)
    atoms = component.get("atoms", ())
    literal_ids: dict[str, set[str]] = {}
    for atom in atoms:
        if isinstance(atom, Mapping) and atom.get("atom_id"):
            literal_ids.setdefault(
                _normalised_literal_name(atom.get("atom_id")), set()
            ).add(str(atom.get("atom_id")))
    literal_matches = literal_ids.get(name, set())
    if len(literal_matches) == 1:
        return next(iter(literal_matches)), "literal"
    if len(literal_matches) > 1:
        return None, "ambiguous"
    aliases = {
        str(atom.get("atom_id"))
        for atom in atoms
        if isinstance(atom, Mapping)
        and atom.get("atom_id")
        and name
        in {
            _normalised_literal_name(alias)
            for alias in atom.get("aliases", ())
        }
    }
    if len(aliases) != 1:
        return None, "ambiguous" if len(aliases) > 1 else "unknown"
    return next(iter(aliases)), "alias"

def _ccd_atoms_by_id(component: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(atom["atom_id"]): atom
        for atom in component.get("atoms", ())
        if isinstance(atom, Mapping) and atom.get("atom_id")
    }

def _overlay_ccd_bonds(
    atom_array: Any,
    retained_indices: np.ndarray,
    source_to_ccd: Mapping[int, str],
    component: Mapping[str, Any],
) -> None:
    """Overlay exact CCD internal orders while retaining all source bonds."""

    if len(atom_array) == 0:
        return
    source_index_to_new = {
        int(source_idx): new_idx
        for new_idx, source_idx in enumerate(retained_indices.tolist())
    }
    pair_to_order: dict[tuple[int, int], int] = {}
    if atom_array.bonds is not None:
        for atom_i, atom_j, bond_type in atom_array.bonds.as_array():
            pair_to_order[
                (min(int(atom_i), int(atom_j)), max(int(atom_i), int(atom_j)))
            ] = int(bond_type)

    ccd_to_new = {
        ccd_id: source_index_to_new[int(source_idx)]
        for source_idx, ccd_id in source_to_ccd.items()
        if int(source_idx) in source_index_to_new
    }
    for row in component.get("bonds", ()):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            continue
        ccd_i, ccd_j, raw_order = (
            str(row[0]),
            str(row[1]),
            str(row[2]).lower(),
        )
        if ccd_i not in ccd_to_new or ccd_j not in ccd_to_new:
            continue
        order = _CCD_BOND_ORDER.get(raw_order)
        if order is None:
            continue
        pair = tuple(sorted((ccd_to_new[ccd_i], ccd_to_new[ccd_j])))
        pair_to_order[pair] = order

    bonds = struc.BondList(len(atom_array))
    for (atom_i, atom_j), bond_type in sorted(pair_to_order.items()):
        bonds.add_bond(atom_i, atom_j, bond_type)
    atom_array.bonds = bonds

def _canonical_res_name(mol_type: str, cano_seq_resname: str) -> str:
    res_name = normalise_res_name(cano_seq_resname)
    if mol_type == "protein":
        token_type = TOKEN_TYPE32.get(res_name, TOKEN_TYPE32["UNK"])
    elif mol_type == "rna":
        token_type = TOKEN_TYPE32.get(res_name, TOKEN_TYPE32["N"])
    elif mol_type == "dna":
        token_type = TOKEN_TYPE32.get(res_name, TOKEN_TYPE32["DN"])
    else:
        raise ValueError(f"modified polymer has invalid mol_type={mol_type!r}")
    canonical = ID_TO_TOKEN_TYPE32[int(token_type)]
    assert canonical in STD_RESIDUES, (mol_type, cano_seq_resname, canonical)
    return canonical

def _is_canonical_atom(mol_type: str, res_name: str, atom_name: str) -> bool:
    atom_name = normalise_atom_name(atom_name)
    if mol_type == "protein":
        valid_atoms = PROTEIN_RES_ATOMS.get(res_name, PROTEIN_RES_ATOMS["UNK"])
        return atom_name in valid_atoms or (res_name != "UNK" and atom_name == "OXT")
    return atom_slot_index(mol_type, res_name, atom_name) is not None

def _literal_atom_name(atom_name: str) -> str:
    return str(atom_name).strip().upper().replace("*", "'")

def _prefer_exact_canonical_atom_names(
    atom_names: np.ndarray,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    """Resolve alias collisions only when one atom already has the canonical name."""

    assert atom_names.ndim == candidate_mask.ndim == 1
    assert atom_names.shape == candidate_mask.shape
    selected = candidate_mask.copy()
    normalised = np.asarray(
        [normalise_atom_name(name) for name in atom_names], dtype=object
    )
    literal = np.asarray(
        [_literal_atom_name(name) for name in atom_names], dtype=object
    )
    for canonical_name in np.unique(normalised[selected]):
        colliding = np.flatnonzero(selected & (normalised == canonical_name))
        if len(colliding) <= 1:
            continue
        exact = colliding[literal[colliding] == canonical_name]
        if len(exact) == 1:
            selected[colliding] = False
            selected[int(exact[0])] = True
    return selected

def _normalised_bond_rows(atom_array) -> list[tuple[int, int, int]]:
    if atom_array.bonds is None:
        return []
    return sorted(
        (min(int(i), int(j)), max(int(i), int(j)), int(bond_type))
        for i, j, bond_type in atom_array.bonds.as_array()
    )

def _annotation_array(
    atom_array: Any,
    name: str,
    size: int,
    *,
    default: Any,
    dtype: Any,
) -> np.ndarray:
    """Return a length-checked per-atom annotation or a length-correct default."""

    if hasattr(atom_array, name):
        values = np.asarray(getattr(atom_array, name))
        if values.shape != (size,):
            raise ValueError(
                f"AtomArray annotation {name!r} has shape {values.shape}, "
                f"expected {(size,)}"
            )
        return values
    return np.full(size, default, dtype=dtype)

def _source_bond_order(raw_type: int) -> int | None:
    """Normalize source concrete bond encodings to the CCD order vocabulary."""

    raw = int(raw_type)
    if raw == int(struc.BondType.ANY):
        return None
    if raw in {
        int(struc.BondType.AROMATIC_SINGLE),
        int(struc.BondType.AROMATIC_DOUBLE),
        7,
        int(struc.BondType.AROMATIC),
    }:
        return int(struc.BondType.AROMATIC)
    return raw

def _ccd_internal_bond_conflict(
    atom_array: Any,
    start: int,
    stop: int,
    source_to_ccd: Mapping[int, str],
    component: Mapping[str, Any],
) -> str | None:
    """Reject concrete source/CCD disagreements before overlaying the graph."""

    ccd_pairs: dict[tuple[str, str], int] = {}
    for row in component.get("bonds", ()):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            continue
        order = _CCD_BOND_ORDER.get(str(row[2]).lower())
        if order is None:
            continue
        pair = tuple(sorted((str(row[0]), str(row[1]))))
        previous = ccd_pairs.get(pair)
        if previous is not None and previous != order:
            return "component_conflicting_ccd_bond_order"
        ccd_pairs[pair] = order

    if atom_array.bonds is None:
        return None
    for atom_i, atom_j, raw_type in atom_array.bonds.as_array():
        atom_i = int(atom_i)
        atom_j = int(atom_j)
        if not (start <= atom_i < stop and start <= atom_j < stop):
            continue
        ccd_i = source_to_ccd.get(atom_i)
        ccd_j = source_to_ccd.get(atom_j)
        if ccd_i is None or ccd_j is None:
            continue
        ccd_order = ccd_pairs.get(tuple(sorted((ccd_i, ccd_j))))
        source_order = _source_bond_order(int(raw_type))
        if ccd_order is not None and source_order is not None and source_order != ccd_order:
            return (
                "source_ccd_bond_order_conflict:"
                f"{ccd_i}-{ccd_j}:source={int(raw_type)}:ccd={ccd_order}"
            )
    return None

def _set_centre_atom_mask(atom_array) -> None:
    centre_mask = np.zeros(len(atom_array), dtype=np.int64)
    starts = np.asarray(
        struc.get_residue_starts(atom_array, add_exclusive_stop=True),
        dtype=np.int64,
    )
    for start, stop in zip(starts[:-1], starts[1:]):
        mol_type = str(atom_array.mol_type[start]).strip().lower()
        res_name = normalise_res_name(atom_array.res_name[start])
        if mol_type == "ligand" or res_name not in STD_RESIDUES:
            centre_mask[start:stop] = 1
            continue
        preferred = "CA" if mol_type == "protein" else "C1'"
        names = np.asarray(atom_array.atom_name[start:stop]).astype(str)
        matches = np.flatnonzero(names == preferred)
        centre_mask[start + int(matches[0]) if len(matches) else start] = 1
    atom_array.set_annotation("centre_atom_mask", centre_mask)

def canonicalize_modified_polymer_carrier(
    token_array: Any,
    atom_array: Any,
) -> tuple[Any, Any, np.ndarray, dict[str, int]]:
    """Project modified polymers by exact atom name and retain the induced bonds."""

    n_atom_before = len(atom_array)
    n_token_before = len(token_array)
    assert n_atom_before > 0 and n_token_before > 0
    modified_mask = np.asarray(
        getattr(atom_array, "modified_res_mask", np.zeros(n_atom_before, dtype=bool)),
        dtype=bool,
    )
    mol_types = np.char.lower(np.asarray(atom_array.mol_type).astype(str))
    project_mask = modified_mask & np.isin(mol_types, tuple(_POLYMER_MOL_TYPES))
    if not bool(project_mask.any()):
        return (
            token_array,
            atom_array,
            np.arange(n_token_before, dtype=np.int64),
            {
                "modified_residue_count": 0,
                "dropped_modified_residue_count": 0,
                "atom_count_before": n_atom_before,
                "atom_count_after": n_atom_before,
                "bond_count_before": len(_normalised_bond_rows(atom_array)),
                "bond_count_after": len(_normalised_bond_rows(atom_array)),
                "token_count_before": n_token_before,
                "token_count_after": n_token_before,
            },
        )

    keep_atom = np.ones(n_atom_before, dtype=bool)
    projected_res_name = np.asarray(atom_array.res_name).copy()
    carrier_mask_before = np.zeros(n_atom_before, dtype=bool)
    carrier_atom_id_before = np.full(n_atom_before, "", dtype="<U32")
    carrier_source_atom_name_before = np.full(n_atom_before, "", dtype="<U32")
    carrier_element_before = np.full(n_atom_before, "", dtype="<U4")
    carrier_charge_before = np.zeros(n_atom_before, dtype=np.int64)
    carrier_component_before = np.full(n_atom_before, "", dtype="<U8")
    carrier_residue_key_before = np.full(n_atom_before, "", dtype="<U256")
    carrier_records: list[dict[str, Any]] = []
    ccd_fallback_reasons: list[dict[str, Any]] = []
    components = _projection_ccd_components()
    starts = np.asarray(
        struc.get_residue_starts(atom_array, add_exclusive_stop=True),
        dtype=np.int64,
    )
    chain_values = _annotation_array(
        atom_array, "chain_id", n_atom_before, default="", dtype="<U1"
    )
    asym_values = _annotation_array(
        atom_array, "asym_id_int", n_atom_before, default=-1, dtype=np.int64
    )
    res_id_values = _annotation_array(
        atom_array, "res_id", n_atom_before, default=-1, dtype=np.int64
    )
    ins_code_values = _annotation_array(
        atom_array, "ins_code", n_atom_before, default="", dtype="<U1"
    )
    modified_residue_count = 0
    dropped_modified_residue_count = 0
    for start, stop in zip(starts[:-1], starts[1:]):
        if not bool(project_mask[start:stop].any()):
            continue
        assert bool(project_mask[start:stop].all()), (
            "modified_res_mask must be residue-uniform",
            int(start),
            int(stop),
        )
        mol_type = str(mol_types[start])
        assert bool((mol_types[start:stop] == mol_type).all())
        source_res_name = str(atom_array.res_name[start]).strip()
        component = components.get(source_res_name.upper())
        source_atom_names = np.asarray(atom_array.atom_name[start:stop]).astype(str)
        source_elements = np.asarray(atom_array.element[start:stop]).astype(str)
        residue_key = ":".join(
            (
                mol_type,
                str(chain_values[start]),
                str(asym_values[start]),
                str(res_id_values[start]),
                str(ins_code_values[start]),
                source_res_name,
            )
        )
        ccd_reason: str | None = None
        source_to_ccd: dict[int, str] = {}
        ccd_atoms: dict[str, Mapping[str, Any]] = {}
        if not isinstance(component, Mapping):
            ccd_reason = "component_missing"
        elif str(component.get("polymer_modality", "")).lower() != mol_type:
            ccd_reason = "component_modality_mismatch"
        elif component.get("validation_issues"):
            ccd_reason = "component_validation_issues"
        else:
            ccd_atoms = _ccd_atoms_by_id(component)
            if not ccd_atoms or not component.get("bonds"):
                ccd_reason = "component_graph_incomplete"
            else:
                for local_idx, atom_name in enumerate(source_atom_names, start=int(start)):
                    element = source_elements[local_idx - int(start)].strip().upper()
                    # Only the exact hydrogen isotopes are omitted from the
                    # CCD intersection. Element symbols such as HG (mercury)
                    # are heavy atoms and must be mapped when the CCD carries
                    # them explicitly.
                    if element in _HYDROGEN_ELEMENTS:
                        continue
                    ccd_atom_id, resolution_reason = _resolve_ccd_atom_id_with_reason(
                        component, atom_name
                    )
                    if ccd_atom_id is None or ccd_atom_id not in ccd_atoms:
                        ccd_reason = (
                            "ambiguous_source_atom_alias"
                            if resolution_reason == "ambiguous"
                            else "source_heavy_atom_not_in_ccd"
                        )
                        source_to_ccd.clear()
                        break
                    ccd_element = str(ccd_atoms[ccd_atom_id].get("element", "")).strip().upper()
                    if not element or not ccd_element:
                        ccd_reason = "source_ccd_element_unknown"
                        source_to_ccd.clear()
                        break
                    if ccd_element != element:
                        ccd_reason = "source_ccd_element_mismatch"
                        source_to_ccd.clear()
                        break
                    if ccd_atom_id in source_to_ccd.values():
                        ccd_reason = "ambiguous_source_atom_alias"
                        source_to_ccd.clear()
                        break
                    source_to_ccd[local_idx] = ccd_atom_id
                if not source_to_ccd:
                    ccd_reason = ccd_reason or "source_has_no_heavy_atoms"
                elif ccd_reason is None:
                    ccd_reason = _ccd_internal_bond_conflict(
                        atom_array,
                        int(start),
                        int(stop),
                        source_to_ccd,
                        component,
                    )
                    if ccd_reason is None:
                        local_keep = np.zeros(stop - start, dtype=bool)
                        for source_idx in source_to_ccd:
                            local_keep[source_idx - int(start)] = True
                        if not bool(local_keep.any()):
                            ccd_reason = "no_common_heavy_atoms"
                        else:
                            keep_atom[start:stop] = local_keep
                            carrier_mask_before[start:stop] = local_keep
                            for source_idx, ccd_atom_id in source_to_ccd.items():
                                local_idx = source_idx - int(start)
                                ccd_atom = ccd_atoms[ccd_atom_id]
                                carrier_atom_id_before[source_idx] = ccd_atom_id
                                carrier_source_atom_name_before[source_idx] = source_atom_names[
                                    local_idx
                                ]
                                carrier_element_before[source_idx] = str(
                                    ccd_atom.get("element", source_elements[local_idx])
                                ).strip().upper()
                                carrier_charge_before[source_idx] = int(
                                    ccd_atom.get("formal_charge", 0) or 0
                                )
                                carrier_component_before[source_idx] = source_res_name.upper()
                                carrier_residue_key_before[source_idx] = residue_key
                            carrier_records.append(
                                {
                                    "start": int(start),
                                    "stop": int(stop),
                                    "source_to_ccd": dict(source_to_ccd),
                                    "component": component,
                                }
                            )
                            modified_residue_count += 1
                            continue
        if ccd_reason is not None:
            ccd_fallback_reasons.append(
                {"residue_key": residue_key, "res_name": source_res_name, "reason": ccd_reason}
            )
        cano_names = {
            normalise_res_name(value)
            for value in np.asarray(atom_array.cano_seq_resname[start:stop]).astype(str)
        }
        assert len(cano_names) == 1, (int(start), int(stop), sorted(cano_names))
        canonical_res_name = _canonical_res_name(mol_type, next(iter(cano_names)))
        local_atom_names = source_atom_names
        local_keep = np.asarray(
            [
                _is_canonical_atom(mol_type, canonical_res_name, atom_name)
                for atom_name in local_atom_names
            ],
            dtype=bool,
        )
        local_keep = _prefer_exact_canonical_atom_names(
            local_atom_names,
            local_keep,
        )
        required_center = "CA" if mol_type == "protein" else "C1'"
        retained_atom_names = {
            normalise_atom_name(atom_name)
            for atom_name in local_atom_names[local_keep]
        }
        if required_center not in retained_atom_names:
            local_keep[:] = False
        if not bool(local_keep.any()):
            raise ValueError(
                "modified-polymer residue has no valid proxy representation: "
                f"{residue_key}; reason={ccd_reason or 'proxy_no_retained_atoms'}"
            )
        retained_names = local_atom_names[local_keep]
        normalised_names = [normalise_atom_name(name) for name in retained_names]
        assert len(normalised_names) == len(set(normalised_names)), (
            "modified residue maps multiple atoms to one canonical atom name",
            str(atom_array.res_name[start]),
            canonical_res_name,
            normalised_names,
        )
        keep_atom[start:stop] = local_keep
        projected_res_name[start:stop] = canonical_res_name
        modified_residue_count += 1
        dropped_modified_residue_count += int(not bool(local_keep.any()))

    if not bool(keep_atom.any()):
        raise ValueError("modified-polymer projection removed every crop atom")

    old_bonds = _normalised_bond_rows(atom_array)
    old_to_new_atom = np.full(n_atom_before, -1, dtype=np.int64)
    old_to_new_atom[keep_atom] = np.arange(int(keep_atom.sum()), dtype=np.int64)
    expected_bonds = sorted(
        (
            min(int(old_to_new_atom[i]), int(old_to_new_atom[j])),
            max(int(old_to_new_atom[i]), int(old_to_new_atom[j])),
            int(bond_type),
        )
        for i, j, bond_type in old_bonds
        if keep_atom[i] and keep_atom[j]
    )

    projected_atom_array = atom_array[keep_atom]
    projected_atom_array.res_name = projected_res_name[keep_atom]
    retained_old_indices = np.flatnonzero(keep_atom).astype(np.int64)
    projected_carrier_mask = carrier_mask_before[keep_atom]
    projected_atom_names = np.asarray(projected_atom_array.atom_name).astype(str)
    projected_source_atom_names = np.asarray(
        carrier_source_atom_name_before[keep_atom]
    ).astype(str)
    for projected_idx, source_idx in enumerate(retained_old_indices.tolist()):
        if projected_carrier_mask[projected_idx]:
            projected_atom_names[projected_idx] = carrier_atom_id_before[source_idx]
    projected_atom_array.atom_name = projected_atom_names
    projected_atom_array.set_annotation(
        "modified_polymer_carrier_source_atom_name", projected_source_atom_names
    )
    for record in carrier_records:
        _overlay_ccd_bonds(
            projected_atom_array,
            retained_old_indices,
            record["source_to_ccd"],
            record["component"],
        )
    if hasattr(projected_atom_array, "cano_seq_resname"):
        cano_seq = np.asarray(projected_atom_array.cano_seq_resname).astype(
            "<U3", copy=True
        )
        cano_seq[project_mask[keep_atom] & ~carrier_mask_before[keep_atom]] = (
            projected_res_name[keep_atom][
                project_mask[keep_atom] & ~carrier_mask_before[keep_atom]
            ]
        )
        projected_atom_array.cano_seq_resname = cano_seq
    projected_modified_mask = modified_mask[keep_atom].copy()
    projected_from_modified = project_mask[keep_atom].copy()
    projected_modified_mask[projected_from_modified & ~carrier_mask_before[keep_atom]] = False
    projected_atom_array.set_annotation(
        "modified_res_mask", projected_modified_mask
    )
    projected_atom_array.set_annotation(
        "was_modified_polymer", projected_from_modified
    )
    projected_carrier_mask = carrier_mask_before[keep_atom]
    projected_atom_array.set_annotation(
        "modified_polymer_proxy_mask",
        projected_from_modified & ~projected_carrier_mask,
    )
    projected_atom_array.set_annotation(
        "modified_polymer_carrier_mask", projected_carrier_mask
    )
    projected_atom_array.set_annotation(
        "modified_polymer_carrier_atom_id", carrier_atom_id_before[keep_atom]
    )
    projected_atom_array.set_annotation(
        "modified_polymer_carrier_element", carrier_element_before[keep_atom]
    )
    projected_atom_array.set_annotation(
        "modified_polymer_carrier_charge", carrier_charge_before[keep_atom]
    )
    projected_atom_array.set_annotation(
        "modified_polymer_carrier_component_id", carrier_component_before[keep_atom]
    )
    projected_atom_array.set_annotation(
        "modified_polymer_carrier_residue_key", carrier_residue_key_before[keep_atom]
    )
    _set_centre_atom_mask(projected_atom_array)
    actual_bonds = _normalised_bond_rows(projected_atom_array)
    if carrier_records:
        expected_pairs = {(i, j) for i, j, _ in expected_bonds}
        actual_pairs = {(i, j) for i, j, _ in actual_bonds}
        assert expected_pairs <= actual_pairs, (
            "CCD carrier overlay dropped a retained source bond"
        )
    else:
        assert actual_bonds == expected_bonds, (
            "Biotite AtomArray slicing did not preserve the retained-atom induced bond graph"
        )

    old_atom_to_source_token = np.full(n_atom_before, -1, dtype=np.int64)
    try:
        source_token_indices = np.asarray(
            token_array.get_annotation("source_token_index"), dtype=np.int64
        )
    except (AttributeError, KeyError):
        source_token_indices = None
    for token_idx, token in enumerate(token_array):
        atom_indices = np.asarray(token.atom_indices, dtype=np.int64)
        assert bool((old_atom_to_source_token[atom_indices] == -1).all())
        old_atom_to_source_token[atom_indices] = (
            int(source_token_indices[token_idx])
            if source_token_indices is not None
            else token_idx
        )
    assert bool((old_atom_to_source_token >= 0).all())
    retained_atom_to_source_token = old_atom_to_source_token[keep_atom]

    projected_token_array = AtomArrayTokenizer(projected_atom_array).get_token_array()
    new_centres = np.asarray(
        projected_token_array.get_annotation("centre_atom_index"), dtype=np.int64
    )
    if source_token_indices is not None:
        projected_source_tokens = retained_atom_to_source_token[new_centres]
        assert len(np.unique(projected_source_tokens)) == len(projected_source_tokens)
        projected_token_array.set_annotation(
            "source_token_index", projected_source_tokens
        )

    new_atom_to_token = np.full(len(projected_atom_array), -1, dtype=np.int64)
    for token_idx, token in enumerate(projected_token_array):
        atom_indices = np.asarray(token.atom_indices, dtype=np.int64)
        assert bool((new_atom_to_token[atom_indices] == -1).all())
        new_atom_to_token[atom_indices] = token_idx
    assert bool((new_atom_to_token >= 0).all())
    old_centres = np.asarray(
        token_array.get_annotation("centre_atom_index"), dtype=np.int64
    )
    old_token_to_new = np.full(n_token_before, -1, dtype=np.int64)
    retained_old_tokens = keep_atom[old_centres]
    retained_new_atoms = old_to_new_atom[old_centres[retained_old_tokens]]
    old_token_to_new[retained_old_tokens] = new_atom_to_token[retained_new_atoms]

    return (
        projected_token_array,
        projected_atom_array,
        old_token_to_new,
        {
            "modified_residue_count": modified_residue_count,
            "dropped_modified_residue_count": dropped_modified_residue_count,
            "ccd_carrier_residue_count": len(carrier_records),
            "ccd_carrier_fallback_count": len(ccd_fallback_reasons),
            "ccd_carrier_fallback_reasons": ccd_fallback_reasons,
            "atom_count_before": n_atom_before,
            "atom_count_after": len(projected_atom_array),
            "bond_count_before": len(old_bonds),
            "bond_count_after": len(actual_bonds),
            "token_count_before": n_token_before,
            "token_count_after": len(projected_token_array),
        },
    )
