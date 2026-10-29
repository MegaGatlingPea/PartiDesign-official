"""Resolve one actual candidate into the native condition and VAE reference views.

The task remains the authority for all permissions. Result metadata supplies final
output locators and resolved motif placement, never a new set of fixed conditions.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import biotite.structure as struc
import numpy as np

from .records import (
    CompiledSample, DesignSampleError, ExpandedRequest, LengthPlan, PartialSpec,
)


@dataclass(frozen=True)
class PartialReference:
    """Read-only task-preparation snapshot, safe to send to loader workers."""

    record: Any
    design_metadata: tuple[dict[str, Any], ...]
    result_csv: Path | None = None
    terminal_op3_rows: tuple[int, ...] = field(init=False, default=())

    def __post_init__(self):
        _validate_fixed_restype_ledger(self.record)
        object.__setattr__(self, "terminal_op3_rows", _validated_terminal_op3_rows(self.record.atoms))



def _validate_fixed_restype_ledger(record) -> None:
    """A fixed locator must identify one exact actual residue, never a proxy."""
    from src.validate.inputs import _locator_rows, _normalise_locators, _parse_json_list

    entries = _normalise_locators(
        _parse_json_list(record.row.get("fixed_restypes", ""), field="fixed_restypes",
                         row_context="partial reference"),
        field="fixed_restypes", row_context="partial reference",
    )
    atoms = record.atoms
    for index, locator in enumerate(entries):
        rows = _locator_rows(atoms, locator)
        if not rows or len(set(int(atoms.residue_uid[row]) for row in rows)) != 1:
            raise DesignSampleError(
                f"partial fixed_restypes[{index}] must resolve one actual residue"
            )
        expected = locator.get("restype")
        actual = {str(atoms.res_name[row]) for row in rows}
        if not isinstance(expected, str) or not expected or actual != {expected}:
            raise DesignSampleError(
                f"partial fixed_restypes[{index}] restype disagrees with actual CIF identity"
            )


def _validated_terminal_op3_rows(atoms) -> tuple[int, ...]:
    """Apply the existing graph-proven standard first-residue OP3 exception."""
    from .compiler import _STANDARD_BY_MODALITY

    result = []
    graph = None if atoms.bonds is None else atoms.bonds.as_array()
    for raw_row in np.flatnonzero(np.asarray(atoms.atom_name).astype(str) == "OP3"):
        row = int(raw_row)
        modality = str(atoms.mol_type[row])
        chain = str(_labels(atoms)[row])
        groups = _residues(atoms, chain)
        residue = np.flatnonzero(atoms.residue_uid == atoms.residue_uid[row])
        phosphorus = residue[atoms.atom_name[residue] == "P"]
        oxygens = residue[atoms.atom_name[residue] == "OP3"]
        valid = (
            modality in {"rna", "dna"}
            and str(atoms.res_name[row]) in _STANDARD_BY_MODALITY[modality]
            and len(groups) > 0
            and int(atoms.residue_uid[groups[0][0]]) == int(atoms.residue_uid[row])
            and len(phosphorus) == 1 and len(oxygens) == 1
            and str(atoms.element[row]).upper() == "O"
            and str(atoms.element[phosphorus[0]]).upper() == "P"
        )
        if valid and graph is not None:
            incident = graph[(graph[:, 0] == row) | (graph[:, 1] == row)]
            valid = (
                len(incident) == 1 and int(incident[0, 2]) == int(struc.BondType.SINGLE)
                and {int(incident[0, 0]), int(incident[0, 1])} == {row, int(phosphorus[0])}
            )
        else:
            valid = False
        if not valid:
            raise DesignSampleError("partial reference has an unsupported or unproven terminal OP3")
        result.append(row)
    return tuple(result)


def load_partial_reference(spec: PartialSpec, *, task=None) -> PartialReference:
    """Read one actual CIF, with saved locators when a result CSV is supplied."""
    from src.validate.inputs import load_input, _parse_design_chains

    if not spec.reference.is_file():
        raise DesignSampleError(f"partial reference does not exist: {spec.reference}")
    result_csv = spec.result_csv
    if result_csv is None:
        return _load_cif_reference(spec.reference, task)
    if not result_csv.is_file():
        raise DesignSampleError(f"partial result CSV does not exist: {result_csv}")
    try:
        with result_csv.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        raise DesignSampleError(f"cannot read partial result CSV: {exc}") from exc
    matches = []
    for row in rows:
        filename = str(row.get("filename") or "").strip()
        if not filename:
            continue
        path = Path(filename).expanduser()
        if not path.is_absolute():
            path = result_csv.parent / path
        if path.resolve() == spec.reference:
            matches.append(row)
    if len(matches) != 1:
        raise DesignSampleError(
            f"partial reference requires exactly one matching result row; found {len(matches)}"
        )
    row = matches[0]
    required = {"design_chains", "fixed_restypes", "bond_constraints",
                "contig_string", "tip_atoms"}
    if required.difference(row):
        raise DesignSampleError(
            f"partial result row lacks output locators: {sorted(required.difference(row))}"
        )
    if str(row.get("error") or "").strip():
        raise DesignSampleError("partial reference result row reports a generation error")
    if not str(row.get("fixed_restypes") or "").strip() or not str(
        row.get("bond_constraints") or ""
    ).strip():
        raise DesignSampleError("partial reference requires resolved constraint metadata")
    try:
        metadata = _parse_design_chains(row["design_chains"], row_context="partial reference")
        record = load_input(spec.reference, row, spec.reference.parent)
    except (OSError, ValueError, TypeError) as exc:
        raise DesignSampleError(f"invalid partial reference: {exc}") from exc
    if not metadata:
        raise DesignSampleError("partial reference has no resolved design chains")
    if record.reasons:
        raise DesignSampleError("invalid partial reference: " + "; ".join(record.reasons))
    return PartialReference(record=record, design_metadata=tuple(metadata), result_csv=result_csv)


def _load_cif_reference(path: Path, task) -> PartialReference:
    """Use explicit task roles and actual identities without a sidecar."""
    from src.validate.inputs import load_input, _parse_design_chains
    from src.validate.sequences import _AA3_TO_1, _RNA3_TO_1, _DNA3_TO_1
    from .compiler import _STANDARD_BY_MODALITY

    if task is None:
        raise DesignSampleError("CIF-only partial reference requires the original task design_chains")
    row = {"filename": path.name,
           "design_chain_id": json.dumps([chain.chain_id for chain in task.design_chains]),
           "fixed_restypes": "[]", "bond_constraints": "[]",
           "contig_string": "", "tip_atoms": "", "error": ""}
    try:
        record = load_input(path, row, path.parent)
        atoms = record.atoms
        maps = {"protein": _AA3_TO_1, "rna": _RNA3_TO_1, "dna": _DNA3_TO_1}
        metadata = []
        cyclic = []
        for chain in task.design_chains:
            groups = _residues(atoms, chain.chain_id)
            if not groups:
                raise DesignSampleError(
                    f"CIF-only partial design chain {chain.chain_id!r} must match an actual CIF chain label"
                )
            if chain.modality == "ligand":
                from rdkit import Chem
                from ..postproc.recovery import _generated_private, _target_graph
                rows = np.flatnonzero(_labels(atoms) == chain.chain_id)
                entry = (task.condition.get("ligand", {}) or {}).get(chain.chain_id, {})
                molecule = (_generated_private(atoms, rows, entry["smiles"])
                            if entry.get("mode") == "smiles" else _target_graph(atoms, rows))
                metadata.append({"chain": chain.chain_id, "modality": "ligand",
                                 "sequence": Chem.MolToSmiles(molecule), "residue_count": len(groups)})
                continue
            sequence = []
            for rows in groups:
                names = set(str(value) for value in atoms.res_name[rows])
                if len(names) != 1 or set(atoms.mol_type[rows]) != {chain.modality}:
                    raise DesignSampleError("partial reference has mixed or incorrect design modality")
                name = names.pop()
                sequence.append(maps[chain.modality][name]
                                if name in _STANDARD_BY_MODALITY[chain.modality]
                                else f"[{name}]")
            entry = {"chain": chain.chain_id, "modality": chain.modality,
                     "sequence": "".join(sequence), "residue_count": len(groups)}
            if chain.sub_modality:
                entry["sub_modality"] = chain.sub_modality
            if (chain.modality == "protein" and task.cyclic_offset
                    and (not task.cyclic_offset_chains or chain.chain_id in task.cyclic_offset_chains)):
                entry["cyclic_offset"] = True
                cyclic.append(chain.chain_id)
            metadata.append(entry)
        metadata = _parse_design_chains(metadata, row_context="CIF-only partial reference")
        row.update(design_chains=json.dumps(metadata, separators=(",", ":")),
                   design_modality=("joint" if len(task.design_chains) > 1
                                    else task.design_chains[0].modality))
        record = replace(record, row=row, cyclic_offset_chains=tuple(cyclic))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise DesignSampleError(f"invalid CIF-only partial reference: {exc}") from exc
    if record.reasons:
        raise DesignSampleError("invalid partial reference: " + "; ".join(record.reasons))
    return PartialReference(record=record, design_metadata=tuple(metadata))

def _labels(atoms) -> np.ndarray:
    return np.asarray(getattr(atoms, "label_asym_id", atoms.chain_id)).astype(str)


def _position(atoms, row: int) -> tuple[int, str]:
    raw = getattr(atoms, "label_seq_id", atoms.res_id)[row]
    try:
        position = int(raw)
    except (ValueError, TypeError):
        position = int(atoms.res_id[row])
    return position, str(atoms.ins_code[row])


def _context_key(atoms, row: int) -> tuple[str, int, str]:
    # Native CIF output emits the normalized physical res_id. PDB input keeps
    # the original author number in label_seq_id, so that field is not a
    # shared axis between the original source and its emitted/refolded output.
    labels = getattr(atoms, "label_asym_id", atoms.chain_id)
    return str(labels[row]), int(atoms.res_id[row]), str(atoms.ins_code[row])


def _residues(atoms, chain: str) -> list[np.ndarray]:
    groups: dict[int, list[int]] = {}
    for row in np.flatnonzero(_labels(atoms) == chain):
        groups.setdefault(int(atoms.residue_uid[row]), []).append(int(row))
    return [np.asarray(rows, dtype=np.int64) for rows in groups.values()]


def _design_mapping(task, reference: PartialReference) -> dict[str, str]:
    specs = task.design_chains
    metadata = reference.design_metadata
    if len(specs) != len(metadata):
        raise DesignSampleError("partial reference design-chain count disagrees with original task")
    original = {chain.chain_id for chain in specs}
    actual = {str(entry["chain"]) for entry in metadata}
    if original == actual:
        by_chain = {str(entry["chain"]): entry for entry in metadata}
        ordered = [by_chain[chain.chain_id] for chain in specs]
    elif original & actual:
        raise DesignSampleError("partial reference design-chain mapping is ambiguous")
    else:
        # Native design_chains preserves authoring chain order even when final
        # output labels differ. Do not infer a mapping from sequences or geometry.
        ordered = list(metadata)
    result = {}
    atoms = reference.record.atoms
    for chain, entry in zip(specs, ordered, strict=True):
        output = str(entry["chain"])
        if entry["modality"] != chain.modality:
            raise DesignSampleError(f"partial reference modality disagrees for {chain.chain_id!r}")
        groups = _residues(atoms, output)
        if not groups:
            raise DesignSampleError(f"partial reference design chain {output!r} is absent")
        if chain.modality != "ligand" and len(groups) != int(entry["residue_count"]):
            raise DesignSampleError(f"partial reference residue count disagrees for {output!r}")
        if any(set(atoms.mol_type[rows]) != {chain.modality} for rows in groups):
            raise DesignSampleError(f"partial reference has mixed or incorrect modality in {output!r}")
        result[chain.chain_id] = output
    return result


class _ExactContigResolver:
    """Reuse native contig parsing only when every integer choice is unique."""

    def integers(self, minimum, maximum):
        if maximum != minimum + 1:
            raise DesignSampleError(
                "CIF-only partial contig gaps are ambiguous; write fixed gap lengths in task YAML"
            )
        return minimum


def _resolved_layouts(request, prepared, reference, chain_map):
    """Map source motifs through the saved final locators without a random draw."""
    from src.validate.references import source_mapping
    from .compiler import _source_selector, _chain_residue_uids

    atoms = reference.record.atoms
    condition = prepared.condition
    contigs = condition.get("contig", {}) or {}
    lengths, layouts = {}, {}
    reverse = {output: original for original, output in chain_map.items()}
    positions = {}
    for chain in request.task.design_chains:
        output = chain_map[chain.chain_id]
        groups = _residues(atoms, output)
        if chain.modality != "ligand":
            lengths[chain.chain_id] = len(groups)
            positions[chain.chain_id] = {
                _position(atoms, int(rows[0])): index
                for index, rows in enumerate(groups)
            }
            if len(positions[chain.chain_id]) != len(groups):
                raise DesignSampleError("partial reference has ambiguous residue locators")
        else:
            entry = (condition.get("ligand", {}) or {}).get(chain.chain_id, {})
            mode = entry.get("mode", "free")
            if chain.chain_id in (condition.get("backbone", []) or []):
                mode = "backbone"
            if mode in {"free", "atom_motif"}:
                lengths[chain.chain_id] = sum(len(rows) for rows in groups)
    if contigs:
        if prepared.source is None:
            raise DesignSampleError("partial source motifs require the original input structure")
        if reference.result_csv is None:
            from .compiler import _parse_contig, _contig_length_bounds
            for chain, entry in contigs.items():
                count = lengths[chain]
                bounds = _contig_length_bounds(entry, chain_id=chain)
                if bounds is not None and not bounds[0] <= count <= bounds[1]:
                    raise DesignSampleError("partial reference length is outside task contig bounds")
                layouts[chain] = _parse_contig(
                    entry.get("token"), chain_id=chain, source=prepared.source,
                    rng=_ExactContigResolver(), motif_length_bounds=(count, count),
                )
        else:
            motif_positions = {
                reverse[chain]: values
                for chain, values in reference.record.motif_positions.items()
                if chain in reverse
            }
            if set(contigs) != set(motif_positions):
                raise DesignSampleError("partial motif locators do not cover original task contigs")
            remapped = replace(reference.record, motif_positions=motif_positions)
            try:
                mapping = source_mapping(remapped, {"condition": dict(condition)}, prepared.source)
            except (ValueError, TypeError, KeyError) as exc:
                raise DesignSampleError(f"partial source motif mapping failed: {exc}") from exc
            for chain in contigs:
                layouts[chain] = [None] * lengths[chain]
            for destination, origin in mapping:
                chain, number, insertion = destination
                index = positions[chain].get((int(number), str(insertion)))
                if index is None:
                    raise DesignSampleError(f"partial motif output residue is absent: {destination}")
                selector = f"[{origin[0]}]{origin[1]}{origin[2]}"
                uids = _source_selector(prepared.source, selector)
                if len(uids) != 1 or layouts[chain][index] is not None:
                    raise DesignSampleError(f"partial original motif selector is ambiguous: {selector}")
                layouts[chain][index] = int(uids[0])
    for chain in request.task.design_chains:
        if chain.modality == "ligand" or chain.chain_id not in (
            condition.get("backbone", []) or []
        ):
            continue
        if prepared.source is None:
            raise DesignSampleError("partial backbone conditions require the original input")
        if len(_chain_residue_uids(prepared.source, chain.chain_id)) != lengths[chain.chain_id]:
            raise DesignSampleError("partial backbone length disagrees with original source")
    return lengths, layouts



def _augment_standard_context(prepared, reference, chain_map):
    """Admit completed current standard-context atoms without adding conditions."""
    from .compiler import _STANDARD_BY_MODALITY
    from ..utils.atoms import concatenate_atom_arrays

    source = prepared.source
    if source is None:
        return prepared
    current = reference.record.atoms
    actual_design = set(chain_map.values())
    actual_groups = {}
    for rows in _group_rows(current, np.flatnonzero(~np.isin(_labels(current), list(actual_design)))):
        row = int(rows[0])
        key = _context_key(current, row)
        if key in actual_groups:
            raise DesignSampleError(f"partial context residue mapping is ambiguous: {key}")
        actual_groups[key] = rows
    additions = []
    next_uid = int(np.max(source.atom_uid)) + 1
    for rows in _group_rows(source, np.flatnonzero(~source.is_design)):
        first = int(rows[0])
        modality = str(source.mol_type[first])
        name = str(source.res_name[first])
        if modality not in _STANDARD_BY_MODALITY or name not in _STANDARD_BY_MODALITY[modality]:
            continue
        key = _context_key(source, first)
        actual_rows = actual_groups.get(key)
        if actual_rows is None:
            continue  # The common projection reports the precise absent locator.
        if set(current.res_name[actual_rows]) != {name}:
            raise DesignSampleError("partial reference changed original context identity")
        names = {str(source.atom_name[row]) for row in rows}
        extra = np.asarray([
            int(row) for row in actual_rows if str(current.atom_name[row]) not in names
        ], dtype=np.int64)
        if not len(extra):
            continue
        from .compiler import _INVENTORIES
        allowed = set(_INVENTORIES[modality][name])
        if modality == "protein":
            allowed.add("OXT")
        if any(int(row) in reference.terminal_op3_rows for row in extra):
            allowed.add("OP3")
        if not set(current.atom_name[extra]) <= allowed:
            raise DesignSampleError("partial context completion contains noncanonical atoms")
        part = current[extra].copy()
        # Existing source row indices remain unchanged, preserving all declared
        # bond and carrier origin mappings. Native tokenization groups by UIDs.
        for field in ("entity_uid", "asym_uid", "component_uid", "molecule_uid", "residue_uid"):
            getattr(part, field)[:] = getattr(source, field)[first]
        part.atom_uid = np.arange(next_uid, next_uid + len(part), dtype=np.int64)
        next_uid += len(part)
        for field in ("chain_id", "label_asym_id", "label_entity_id", "res_id", "label_seq_id", "ins_code"):
            getattr(part, field)[:] = getattr(source, field)[first]
        part.is_design[:] = False
        part.coord[:] = 0
        part.coord_valid[:] = False
        part.motif_coord_condition[:] = False
        part.token_center_condition[:] = False
        part.bonds = None
        additions.append(part)
    if not additions:
        return prepared
    augmented = concatenate_atom_arrays((source, *additions))
    from .compiler import _merge_current_bonds
    terminal_edges = []
    for row in np.flatnonzero(augmented.atom_name == "OP3"):
        residue = np.flatnonzero(augmented.residue_uid == augmented.residue_uid[row])
        phosphorus = residue[augmented.atom_name[residue] == "P"]
        if len(phosphorus) != 1:
            raise DesignSampleError("partial context OP3 has no unique same-residue phosphorus")
        terminal_edges.append((int(row), int(phosphorus[0]), int(struc.BondType.SINGLE)))
    _merge_current_bonds(augmented, terminal_edges)
    return replace(prepared, source=augmented)


def _required_names(compiled, rows, actual_name: str, prepared) -> set[str]:
    from .compiler import _INVENTORIES
    from ..utils.ccd import resolve_ccd_component

    atoms = compiled.atoms
    first = int(rows[0])
    modality = str(atoms.mol_type[first])
    if modality == "ligand":
        return {str(atoms.atom_name[row]) for row in rows}
    inventory = _INVENTORIES[modality]
    if actual_name in inventory and actual_name not in {"UNK", "N", "DN"}:
        required = set(inventory[actual_name])
    else:
        residue_uid = int(atoms.residue_uid[first])
        if residue_uid not in compiled.modified_polymer_carriers and residue_uid not in compiled.model_parent_residue:
            raise DesignSampleError(f"partial reference has unresolved polymer identity {actual_name!r}")
        try:
            resolution = resolve_ccd_component(
                actual_name, embedded=prepared.embedded_ccd,
                configured=prepared.configured_ccd, packaged=prepared.packaged_ccd,
            )
        except (ValueError, KeyError) as exc:
            raise DesignSampleError(f"partial reference requires complete CCD chemistry for {actual_name!r}: {exc}") from exc
        component = resolution.component
        if resolution.status != "resolved" or component is None:
            raise DesignSampleError(f"partial reference requires complete CCD chemistry for {actual_name!r}")
        required = {
            atom.atom_id for atom in component.atoms
            if str(atom.element).upper() not in {"H", "D", "T"} and not atom.leaving
        }
    # Terminal OXT is optional in the native heavy-atom decoder contract.
    required.discard("OXT")
    for token_uid in np.unique(atoms.token_uid[rows]):
        required.difference_update(compiled.omitted_atom_names_by_token_uid.get(int(token_uid), ()))
    return required



def _validate_compiled_ledger(compiled, record, actual_to_native) -> None:
    """Require saved design constraints to remain original-task permissions."""
    from src.validate.inputs import _locator_rows

    atoms, current = compiled.atoms, record.atoms
    for index, locator in enumerate(record.fixed_restypes):
        actual_rows = _locator_rows(current, locator)
        native_rows = [actual_to_native[int(row)] for row in actual_rows]
        residue_uids = {int(atoms.residue_uid[row]) for row in native_rows}
        if len(residue_uids) != 1:
            raise DesignSampleError(f"partial fixed_restypes[{index}] has no unique native residue")
        rows = np.flatnonzero(atoms.residue_uid == next(iter(residue_uids)))
        if not bool(np.asarray(atoms.identity_condition[rows], dtype=bool).all()):
            raise DesignSampleError(
                f"partial CSV fixed_restypes[{index}] is incompatible with original task identity permissions"
            )

    conditions = {edge.canonical_uids(): edge for edge in compiled.topology_conditions}
    polymer = np.isin(np.asarray(atoms.mol_type).astype(str), ("protein", "rna", "dna"))
    for index, bond in enumerate(record.bond_constraints):
        if not bond.get("required_output", True):
            continue
        endpoints = []
        for name in ("atom1", "atom2"):
            actual_rows = _locator_rows(current, bond[name])
            if len(actual_rows) != 1:
                raise DesignSampleError(f"partial bond_constraints[{index}] has no unique actual atom")
            endpoints.append(actual_to_native[int(actual_rows[0])])
        left, right = endpoints
        # The native public hard-constraint ledger contains design-touching
        # polymer-ligand and inter-residue polymer edges. Other source graph
        # facts retain the native reader's checks without becoming new promises.
        if not (bool(atoms.is_design[left]) or bool(atoms.is_design[right])):
            continue
        if not (polymer[left] or polymer[right]):
            continue
        if polymer[left] and polymer[right] and int(atoms.residue_uid[left]) == int(atoms.residue_uid[right]):
            continue
        pair = tuple(sorted((int(atoms.atom_uid[left]), int(atoms.atom_uid[right]))))
        condition = conditions.get(pair)
        if condition is None or not condition.required_output:
            raise DesignSampleError(
                f"partial CSV required bond_constraints[{index}] is incompatible with original task topology permissions"
            )
        requested = str(bond.get("order", "any")).strip().lower()
        original = str(condition.order).strip().lower()
        if requested not in {"any", "unknown"} and requested != original:
            raise DesignSampleError(
                f"partial CSV required bond_constraints[{index}] order is incompatible with original task topology condition"
            )



def _optional_whole_residue_oxt_rows(compiled, reference) -> set[int]:
    """Keep native optional terminal OXT distinct from explicit atom promises."""
    from src.validate.inputs import _parse_tip_positions

    atoms = compiled.atoms
    contigs = compiled.request.task.condition.get("contig", {}) or {}
    required_endpoints = {
        int(uid)
        for edge in compiled.topology_conditions if edge.required_output
        for uid in (edge.atom_uid_a, edge.atom_uid_b)
    }
    optional = set()
    for chain in compiled.request.task.design_chains:
        if chain.modality != "protein" or chain.chain_id not in contigs:
            continue
        fields = compiled.motif_output_fields.get(chain.chain_id)
        if fields is None:
            continue
        selections = _parse_tip_positions(fields.tip_atoms).get(chain.chain_id, {})
        rows = np.flatnonzero(atoms.chain_id == chain.chain_id)
        terminal = int(np.max(atoms.res_id[rows]))
        for row in rows[(atoms.res_id[rows] == terminal) & (atoms.atom_name[rows] == "OXT")]:
            row = int(row)
            # A whole-residue contig copies every observed source atom. Its
            # terminal OXT is optional in the same native output inventory used
            # by _required_names(); an explicit atom or topology promise is not.
            if (
                _position(atoms, row) not in selections
                and not bool(atoms.token_center_condition[row])
                and int(atoms.atom_uid[row]) not in required_endpoints
            ):
                optional.add(row)

    # Context coordinates are implicitly conditioned as whole residues. Apply
    # the same native terminal-OXT output policy, preserving literal OXT tips.
    labels = _labels(atoms)
    context = np.flatnonzero(~np.asarray(atoms.is_design, dtype=bool))
    for chain in dict.fromkeys(labels[context].tolist()):
        rows = context[labels[context] == chain]
        terminal_rows = _group_rows(atoms, rows)[-1]
        first = int(terminal_rows[0])
        if str(atoms.mol_type[first]) != "protein":
            continue
        selections = reference.record.tip_positions.get(chain, {}).get(_position(atoms, first), ())
        if "OXT" in selections:
            continue
        for row in terminal_rows[
            (atoms.atom_name[terminal_rows] == "OXT")
            & (np.char.upper(atoms.element[terminal_rows].astype(str)) == "O")
        ]:
            row = int(row)
            if (
                not bool(atoms.token_center_condition[row])
                and int(atoms.atom_uid[row]) not in required_endpoints
            ):
                optional.add(row)
    return optional


def _project_reference(compiled, reference, prepared, chain_map):
    """Overlay current sequence/geometry on the native author's stable row axis."""
    condition = compiled.atoms
    current = reference.record.atoms
    result = condition.copy()
    result.coord[:] = 0
    result.coord_valid[:] = False
    optional_oxt_rows = _optional_whole_residue_oxt_rows(compiled, reference)
    mapped: dict[int, int] = {}
    seen_current: set[int] = set()
    groups = []
    design_ids = {chain.chain_id for chain in compiled.request.task.design_chains}
    for chain in compiled.request.task.design_chains:
        native_groups = []
        for uid in dict.fromkeys(condition.residue_uid[condition.chain_id == chain.chain_id].tolist()):
            native_groups.append(np.flatnonzero(condition.residue_uid == uid))
        actual_groups = _residues(current, chain_map[chain.chain_id])
        if chain.modality == "ligand":
            native_groups = [np.flatnonzero(condition.chain_id == chain.chain_id)]
            actual_groups = [np.flatnonzero(_labels(current) == chain_map[chain.chain_id])]
        if len(native_groups) != len(actual_groups):
            raise DesignSampleError(f"partial compiled residue count disagrees for {chain.chain_id!r}")
        groups.extend(zip(native_groups, actual_groups, strict=True))
    context = np.flatnonzero(~np.isin(condition.chain_id, list(design_ids)))
    actual_design = set(chain_map.values())
    context_by_key = {}
    for rows in _group_rows(current, np.flatnonzero(~np.isin(_labels(current), list(actual_design)))):
        row = int(rows[0])
        key = _context_key(current, row)
        if key in context_by_key:
            raise DesignSampleError(f"partial context residue mapping is ambiguous: {key}")
        context_by_key[key] = rows
    for rows in _group_rows(condition, context):
        row = int(rows[0])
        key = _context_key(condition, row)
        actual_rows = context_by_key.pop(key, None)
        if actual_rows is None:
            raise DesignSampleError(f"partial reference lacks original context residue {key}")
        groups.append((rows, actual_rows))
    if context_by_key:
        raise DesignSampleError("partial reference contains context absent from the original task")

    for native_rows, actual_rows in groups:
        first, actual_first = int(native_rows[0]), int(actual_rows[0])
        modality = str(condition.mol_type[first])
        names = {str(current.res_name[row]) for row in actual_rows}
        if len(names) != 1 or set(current.mol_type[actual_rows]) != {modality}:
            raise DesignSampleError("partial residue identity/modality is ambiguous")
        actual_name = next(iter(names))
        is_ligand = modality == "ligand"
        if not is_ligand and bool(condition.identity_condition[native_rows].all()):
            if {str(condition.res_name[row]) for row in native_rows} != {actual_name}:
                raise DesignSampleError("partial reference changed an original fixed residue identity")
        actual_by_name = {str(current.atom_name[row]): int(row) for row in actual_rows}
        if len(actual_by_name) != len(actual_rows):
            raise DesignSampleError("partial reference has duplicate atom names in a residue")
        required = _required_names(compiled, native_rows, actual_name, prepared)
        if not is_ligand and not required <= set(actual_by_name):
            raise DesignSampleError(
                f"partial reference lacks complete geometry for {actual_name}: missing {sorted(required-set(actual_by_name))}"
            )
        native_by_name = {str(condition.atom_name[row]): int(row) for row in native_rows}
        if len(native_by_name) != len(native_rows):
            raise DesignSampleError("partial authoring atom inventory is ambiguous")
        free_ligand = is_ligand and not bool(condition.identity_condition[native_rows].any())
        mixed_ligand = is_ligand and not free_ligand and not bool(
            condition.identity_condition[native_rows].all()
        )
        if free_ligand:
            if len(native_rows) != len(actual_rows):
                raise DesignSampleError("partial free-ligand atom count disagrees")
            pairs = zip(native_rows, actual_rows, strict=True)
        elif mixed_ligand:
            if len(native_rows) != len(actual_rows):
                raise DesignSampleError("partial mixed-ligand atom count disagrees")
            available = dict(actual_by_name)
            pairs = []
            for row in native_rows[condition.identity_condition[native_rows]]:
                name = str(condition.atom_name[row])
                if name not in available:
                    raise DesignSampleError("partial reference lacks an original fixed ligand atom")
                pairs.append((int(row), available.pop(name)))
            unmatched = []
            for row in native_rows[~condition.identity_condition[native_rows]]:
                name = str(condition.atom_name[row])
                if name in available:
                    pairs.append((int(row), available.pop(name)))
                else:
                    unmatched.append(int(row))
            # Salvage may leave gaps in generated atom names. Keep surviving
            # named slots first, then map the remaining free inventory stably.
            pairs.extend(zip(unmatched, (available[name] for name in sorted(available)), strict=True))
        else:
            extra = set(actual_by_name) - set(native_by_name)
            if extra:
                raise DesignSampleError(f"partial current atoms are outside native inventory: {sorted(extra)}")
            pairs = ((native_by_name[name], row) for name, row in actual_by_name.items())
        result.res_name[native_rows] = actual_name
        for native_row, actual_row in pairs:
            native_row, actual_row = int(native_row), int(actual_row)
            if actual_row in seen_current:
                raise DesignSampleError("partial reference atom mapping is not one-to-one")
            if not bool(current.coord_valid[actual_row]) or not np.isfinite(current.coord[actual_row]).all():
                raise DesignSampleError("partial reference requires finite complete current coordinates")
            free_atom = is_ligand and not bool(condition.identity_condition[native_row])
            if not free_atom and str(condition.element[native_row]).upper() != str(current.element[actual_row]).upper():
                raise DesignSampleError("partial reference atom elements disagree with native inventory")
            mapped[native_row] = actual_row
            seen_current.add(actual_row)
            result.coord[native_row] = current.coord[actual_row]
            result.coord_valid[native_row] = True
            result.formal_charge[native_row] = current.formal_charge[actual_row]
            if free_atom:
                result.element[native_row] = current.element[actual_row]
                result.atom_name[native_row] = current.atom_name[actual_row]
        hard_rows = native_rows[
            np.asarray(condition.motif_coord_condition[native_rows], dtype=bool)
            | np.asarray(condition.token_center_condition[native_rows], dtype=bool)
        ]
        if any(int(row) not in mapped and int(row) not in optional_oxt_rows for row in hard_rows):
            raise DesignSampleError("partial reference lacks an original conditioned atom")
    if len(seen_current) != len(current):
        raise DesignSampleError("partial reference has unmatched current atoms")
    reverse = {actual: native for native, actual in mapped.items()}
    _validate_compiled_ledger(compiled, reference.record, reverse)
    edges = []
    if current.bonds is not None:
        edges = [
            (reverse[int(a)], reverse[int(b)], int(order))
            for a, b, order in current.bonds.as_array()
        ]
    result.bonds = struc.BondList(len(result), np.asarray(edges, dtype=np.uint32).reshape(-1, 3))
    graph = {tuple(sorted((a, b))): order for a, b, order in edges}
    uid_rows = {int(uid): row for row, uid in enumerate(condition.atom_uid)}
    for edge in compiled.topology_conditions:
        pair = tuple(sorted((uid_rows[int(edge.atom_uid_a)], uid_rows[int(edge.atom_uid_b)])))
        if edge.required_output and pair not in graph:
            raise DesignSampleError("partial reference lacks an original required topology edge")
        accepted_orders = {
            "single": {1}, "double": {2}, "triple": {3}, "aromatic": {5, 6, 7, 9},
        }
        if pair in graph and edge.order in accepted_orders and graph[pair] not in accepted_orders[edge.order]:
            raise DesignSampleError("partial reference changed an original topology bond order")
    return result


def _group_rows(atoms, rows):
    groups = {}
    for row in rows:
        groups.setdefault(int(atoms.residue_uid[row]), []).append(int(row))
    return [np.asarray(group, dtype=np.int64) for group in groups.values()]


def compile_partial_request(
    request: ExpandedRequest, *, ccd_path: Path | None = None,
    prepared_source=None, trusted_ccd_components: Mapping[str, Any] | None = None,
) -> CompiledSample:
    """Compile current candidate geometry with native task permissions and locators."""
    from .compiler import compile_request, prepare_task_source, _validate_prepared_task_source, _affines

    task = request.task
    if task.partial is None:
        raise DesignSampleError("partial compilation requires task.partial")
    if task.format != "normal":
        raise DesignSampleError("partial references require the normal task format")
    prepared = prepared_source
    if prepared is None:
        prepared = prepare_task_source(task, ccd_path=ccd_path)
    effective_ccd = _validate_prepared_task_source(prepared, task, ccd_path)
    reference = prepared.partial_reference or load_partial_reference(task.partial, task=task)
    chain_map = _design_mapping(task, reference)
    lengths, layouts = _resolved_layouts(request, prepared, reference, chain_map)
    plan = LengthPlan(0, lengths)
    native_task = replace(task, partial=None, length_plans=(plan,), length_distribution_chain=None)
    native_request = replace(request, task=native_task, plan=plan)
    source_inventory = _augment_standard_context(prepared, reference, chain_map)
    native_source = replace(source_inventory, task=native_task, partial_reference=None)
    extra_atoms = {}
    current = reference.record.atoms
    for original, output in chain_map.items():
        for position, rows in enumerate(_residues(current, output), start=1):
            extra = [int(row) for row in rows if int(row) in reference.terminal_op3_rows]
            if extra:
                extra_atoms.setdefault(original, {})[position] = current[np.asarray(extra, dtype=np.int64)]
    compiled = compile_request(
        native_request, ccd_path=effective_ccd, prepared_source=native_source,
        trusted_ccd_components=trusted_ccd_components,
        resolved_polymer_layouts=layouts, polymer_extra_atoms=extra_atoms,
    )
    reference_atoms = _project_reference(compiled, reference, prepared, chain_map)
    # Both flow conditions and clean state use the selected current complex.
    # Identity/topology permissions and original motif source locators stay native.
    condition = compiled.atoms
    conditioned = (condition.motif_coord_condition | condition.token_center_condition) & condition.coord_valid
    present = conditioned & reference_atoms.coord_valid
    condition.coord[present] = reference_atoms.coord[present]
    # Whole-residue optional atoms absent from the current snapshot have no
    # current coordinate condition; literal required endpoints failed above.
    absent = conditioned & ~reference_atoms.coord_valid
    condition.coord[absent] = 0
    condition.coord_valid[absent] = False
    compiled.source_to_model_affine, compiled.model_to_source_affine = _affines(reference_atoms)
    # Native assembly retains chain_id for task ownership and label_asym_id for
    # the actual final-output locators. No second assembly implementation exists.
    motif_output = dict(compiled.motif_output_fields)
    from .compiler import _final_chain_selector
    from .records import MotifOutputFields
    for original, output in chain_map.items():
        rows = compiled.atoms.chain_id == original
        compiled.atoms.label_asym_id[rows] = output
        reference_atoms.label_asym_id[rows] = output
        if original in motif_output and original != output:
            before, after = _final_chain_selector(original), _final_chain_selector(output)
            field = motif_output[original]
            def relabel(text, delimiter):
                return delimiter.join(
                    after + part[len(before):] if part.startswith(before) else part
                    for part in text.split(delimiter)
                )
            motif_output[original] = MotifOutputFields(
                relabel(field.contig_string, "/"), relabel(field.tip_atoms, ";"),
            )
    compiled.motif_output_fields = motif_output
    labels = np.asarray(compiled.atoms.label_asym_id).astype(str)
    for label in set(labels):
        if len(np.unique(compiled.atoms.asym_uid[labels == label])) != 1:
            raise DesignSampleError("partial output design labels collide with retained context")
    actual_partial = replace(task.partial, result_csv=reference.result_csv)
    compiled.request = replace(request, plan=plan, task=replace(task, partial=actual_partial))
    compiled.reference_atoms = reference_atoms
    compiled.partial_fraction = task.partial.fraction
    return compiled


__all__ = ["PartialReference", "load_partial_reference", "compile_partial_request"]
