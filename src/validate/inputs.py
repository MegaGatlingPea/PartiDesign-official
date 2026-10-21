"""Input discovery and structure identity for standalone validation.

The validator consumes the public ``result.csv`` row and the one structure
named by that row.  This module intentionally does not walk a task's other
PDB/CIF files: a validation run must be reproducible from the row identity.
The project's controlled reader is used so explicit CIF/PDB topology remains
available through ``ValidationInput.atoms.bonds``.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from ..design.utils.atoms import (
    UID_FIELDS,
    assign_identity_uids,
    tokenize_atom_array,
    validate_atom_array,
)
from ..design.utils.reader import read_structure


class ValidationInputError(ValueError):
    """A row or structure cannot be used as a validation input."""


@dataclass(frozen=True, slots=True)
class InputDescriptor:
    """A row-level descriptor yielded before structure parsing.

    Descriptors are deliberately JSON-like.  The orchestrator can account for
    a missing file or malformed row without constructing an AtomArray, then
    call :func:`load_input` for rows that are eligible for metric evaluation.
    """

    run_dir: Path
    task_dir: Path
    result_path: Path
    path: Path
    row: dict[str, str]
    sample_id: str
    row_index: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationInput:
    """One generated structure plus its resolved public result metadata.

    ``atoms`` is the exact output of ``src.design.utils.reader.read_structure``
    and therefore includes the reader's complete template and explicit bond
    graph.  ``fixed_restypes`` and ``bond_constraints`` retain the JSON
    records from the row; ``fixed_positions`` is the residue-locator index
    used by downstream sequence/motif checks.
    """

    path: Path
    task_dir: Path
    sample_id: str
    row: dict[str, str]
    atoms: Any
    design_chains: tuple[str, ...]
    fixed_restypes: list[dict[str, Any]]
    bond_constraints: list[dict[str, Any]]
    contig_string: str
    tip_atoms: str
    cyclic_chains: tuple[str, ...]
    fixed_positions: dict[str, list[dict[str, Any]]]
    motif_positions: dict[str, tuple[tuple[int, str], ...]]
    tip_positions: dict[str, dict[tuple[int, str], tuple[str, ...]]]
    reasons: tuple[str, ...] = ()
    # Explicit model/refold intent; chemical cyclic_chains remains graph-derived.
    cyclic_offset_chains: tuple[str, ...] = ()

    @property
    def metadata_states(self) -> dict[str, str]:
        return {field: "known" if _row_text(self.row, field) else "unknown"
                for field in ("fixed_restypes", "bond_constraints")}

    @property
    def chemistry_metadata(self) -> dict:
        atoms = self.atoms
        chains = np.asarray(getattr(atoms, "label_asym_id", atoms.chain_id)).astype(str)
        def locator(index):
            return {"atom_index": index, "chain": str(chains[index]),
                    "res_id": int(atoms.res_id[index]), "ins_code": str(atoms.ins_code[index]),
                    "restype": str(atoms.res_name[index]), "atom": str(atoms.atom_name[index])}
        graph = [] if getattr(atoms, "bonds", None) is None else [
            {"atom1": locator(int(left)), "atom2": locator(int(right)), "order": int(order)}
            for left, right, order in atoms.bonds.as_array().tolist()]
        return {"schema_version": 1, "metadata_states": self.metadata_states,
                "graph_bonds": graph, "requested_constraints": self.bond_constraints,
                "fixed_restypes": self.fixed_restypes, "cyclic_chains": list(self.cyclic_chains),
                "cyclic_offset_chains": list(self.cyclic_offset_chains),
                "graph_definition": "complete reader template and explicit source bond graph"}

    @property
    def context_chains(self) -> tuple[str, ...]:
        """All loaded output chains outside the row-declared design set."""

        values = np.asarray(
            getattr(self.atoms, "label_asym_id", getattr(self.atoms, "chain_id"))
        ).astype(str)
        return tuple(
            chain for chain in dict.fromkeys(values.tolist())
            if chain not in self.design_chains
        )

    @property
    def design_modality(self) -> str:
        return _row_text(self.row, "design_modality")

    @property
    def ligand_entries(self) -> dict[str, dict[str, str]]:
        """Expose validated per-chain ligand SMILES to refold adapters."""

        if "design_chains" not in self.row:
            return {}
        metadata = _parse_design_chains(
            self.row.get("design_chains", ""),
            row_context=f"{self.task_dir.name}/{self.sample_id}",
        )
        return {
            str(item["chain"]): {
                "mode": "smiles",
                "smiles": str(item["sequence"]),
            }
            for item in metadata
            if item["modality"] == "ligand"
        }


_SOURCE_SEGMENT = re.compile(
    r"^(?:\[([^\]]+)\]|([A-Za-z]))(\d+)([^\d\-/]*)"
    r"(?:-(\d+)([^\d\-/]*))?$"
)
_TIP_SEGMENT = re.compile(
    r"^(?:\[([^\]]+)\]|([A-Za-z]))(\d+)([^:]*):(.+)$"
)
_JOINT_FILENAME_SEGMENT = re.compile(
    r"^(?P<chain>.+)_(?P<modality>prot|rna|dna|ligand)_"
)


def _row_text(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name, "")
    if value is None:
        return ""
    # CSV readers return strings; accepting scalar values here keeps the API
    # convenient for tests and programmatic callers without hiding mappings.
    return str(value).strip()


def _parse_json_list(value: Any, *, field: str, row_context: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValidationInputError(
            f"{row_context}: {field} is not valid JSON"
        ) from exc
    if not isinstance(parsed, list):
        raise ValidationInputError(f"{row_context}: {field} must be a JSON list")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, Mapping):
            raise ValidationInputError(
                f"{row_context}: {field}[{index}] must be an object"
            )
        result.append(dict(item))
    return result


_DESIGN_CHAIN_MODALITIES = frozenset({"protein", "rna", "dna", "ligand"})
_DESIGN_CHAIN_SUB_MODALITIES = frozenset({"peptide"})
_DESIGN_CHAIN_FIELDS = ("chain", "modality", "sequence", "residue_count")


def _parse_design_chains(
    value: Any,
    *,
    row_context: str,
) -> list[dict[str, Any]]:
    """Parse the public final-output design-chain metadata.

    None and an empty value represent an explicitly present empty list.
    The caller uses the CSV key's presence to decide whether legacy scalar
    fallback is allowed; this helper only validates the value itself.
    """

    if value is None:
        parsed: Any = []
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            parsed = []
        else:
            try:
                parsed = json.loads(text)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValidationInputError(
                    f"{row_context}: design_chains is not valid JSON"
                ) from exc
    else:
        parsed = value
    if not isinstance(parsed, list):
        raise ValidationInputError(
            f"{row_context}: design_chains must be a JSON list"
        )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_item in enumerate(parsed):
        if not isinstance(raw_item, Mapping):
            raise ValidationInputError(
                f"{row_context}: design_chains[{index}] must be an object"
            )
        missing = [field for field in _DESIGN_CHAIN_FIELDS if field not in raw_item]
        if missing:
            raise ValidationInputError(
                f"{row_context}: design_chains[{index}] missing {', '.join(missing)}"
            )
        chain = raw_item["chain"]
        modality = raw_item["modality"]
        sequence = raw_item["sequence"]
        residue_count = raw_item["residue_count"]
        if (
            not isinstance(chain, str)
            or not chain.strip()
            or chain != chain.strip()
        ):
            raise ValidationInputError(
                f"{row_context}: design_chains[{index}].chain must be a non-empty string"
            )
        chain = chain.strip()
        if chain in seen:
            raise ValidationInputError(
                f"{row_context}: design_chains contains duplicate chain {chain!r}"
            )
        seen.add(chain)
        if not isinstance(modality, str) or modality not in _DESIGN_CHAIN_MODALITIES:
            raise ValidationInputError(
                f"{row_context}: design_chains[{index}].modality is invalid"
            )
        has_sub_modality = "sub_modality" in raw_item
        sub_modality = raw_item.get("sub_modality")
        if has_sub_modality and (
            not isinstance(sub_modality, str)
            or sub_modality not in _DESIGN_CHAIN_SUB_MODALITIES
        ):
            raise ValidationInputError(
                f"{row_context}: design_chains[{index}].sub_modality is invalid"
            )
        if has_sub_modality and modality != "protein":
            raise ValidationInputError(
                f"{row_context}: design_chains[{index}].sub_modality=peptide requires modality=protein"
            )
        if not isinstance(sequence, str) or not sequence:
            raise ValidationInputError(
                f"{row_context}: design_chains[{index}].sequence must be a non-empty string"
            )
        if (
            isinstance(residue_count, bool)
            or not isinstance(residue_count, (int, np.integer))
            or int(residue_count) < 1
        ):
            raise ValidationInputError(
                f"{row_context}: design_chains[{index}].residue_count must be a positive integer"
            )
        if "cyclic_offset" in raw_item and (
            modality != "protein" or type(raw_item["cyclic_offset"]) is not bool
            or (raw_item["cyclic_offset"] and int(residue_count) < 3)
        ):
            raise ValidationInputError(
                f"{row_context}: design_chains[{index}].cyclic_offset requires a boolean and a protein chain of at least three residues"
            )
        normalized = dict(raw_item)
        normalized.update(
            chain=chain,
            modality=str(modality),
            sequence=sequence,
            residue_count=int(residue_count),
        )
        result.append(normalized)
    return result


def _antibody_chain_metadata(row, atoms, row_context):
    from biotite.sequence import ProteinSequence
    metadata, fixed = [], []
    role = row.get("heavy_role", "")
    if role not in {"VH", "VHH"}:
        raise ValidationInputError(f"{row_context}: invalid antibody heavy_role")
    if role == "VHH" and row.get("light_chain"):
        raise ValidationInputError(f"{row_context}: VHH cannot contain a light chain")
    labels = np.asarray(getattr(atoms, "label_asym_id", atoms.chain_id)).astype(str)
    regions = ("fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4")
    seen = set()
    for prefix in ("heavy", "light"):
        chain = row.get(f"{prefix}_chain", "")
        sequence = row.get(f"{prefix}_sequence", "")
        if not chain:
            if prefix == "heavy" or sequence or any(row.get(f"{prefix}_{region}") or row.get(f"{prefix}_{region}_range") for region in regions):
                raise ValidationInputError(f"{row_context}: missing {prefix} chain")
            continue
        if chain in seen:
            raise ValidationInputError(f"{row_context}: antibody roles share a chain")
        seen.add(chain)
        rows = np.flatnonzero(labels == chain)
        if not len(rows) or not np.all(atoms.mol_type[rows] == "protein"):
            raise ValidationInputError(f"{row_context}: antibody chain {chain!r} is absent or nonprotein")
        residues = {}
        for index in rows:
            key = (int(atoms.res_id[index]), str(atoms.ins_code[index]))
            residues.setdefault(key, int(index))
        positions = [position for position, insertion in residues]
        if positions != list(range(1, len(residues)+1)) or any(insertion for position, insertion in residues):
            raise ValidationInputError(f"{row_context}: antibody positions must be final dense label_seq_id")
        actual = "".join(ProteinSequence.convert_letter_3to1(str(atoms.res_name[index])) for index in residues.values())
        if sequence != actual:
            raise ValidationInputError(f"{row_context}: {prefix}_sequence differs from CIF")
        cursor = 1
        for region in regions:
            part = row.get(f"{prefix}_{region}", "")
            end = cursor + len(part)-1
            expected = (str(cursor) if cursor == end else f"{cursor}-{end}") if part else ""
            if row.get(f"{prefix}_{region}_range", "") != expected:
                raise ValidationInputError(f"{row_context}: incorrect {prefix}_{region}_range")
            cursor = end+1
        if "".join(row.get(f"{prefix}_{region}", "") for region in regions) != actual:
            raise ValidationInputError(f"{row_context}: {prefix} FR/CDR segments do not reconstruct CIF sequence")
        mutable = set()
        for interval in row.get(f"{prefix}_design_positions", "").split(","):
            if not interval:
                continue
            match = re.fullmatch(r"([1-9][0-9]*)(?:-([1-9][0-9]*))?", interval)
            if match is None:
                raise ValidationInputError(f"{row_context}: invalid {prefix} design positions")
            start, end = int(match[1]), int(match[2] or match[1])
            if end < start or end > len(actual) or mutable.intersection(range(start,end+1)):
                raise ValidationInputError(f"{row_context}: invalid {prefix} design position range")
            mutable.update(range(start,end+1))
        metadata.append(dict(chain=chain, modality="protein", sequence=actual, residue_count=len(actual)))
        for (position, insertion), index in residues.items():
            if position not in mutable:
                fixed.append(dict(chain=chain, res_id=position, restype=str(atoms.res_name[index])))
    return metadata, fixed


def _decode_filename_component(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        try:
            return chr(int(match.group(1), 16))
        except ValueError:
            return match.group(0)

    return re.sub(r"_x([0-9A-Fa-f]{2})", replace, value)


def _chain_list_from_value(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    # The public runtime currently emits one chain for ordinary tasks.  The
    # separators below are only a compatibility path for joint rows produced
    # by early development builds; chain IDs themselves are not split on '_'.
    return [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]


def _infer_joint_design_chains(row: Mapping[str, Any]) -> list[str]:
    """Recover joint-chain names from the runtime's stable filename grammar."""

    filename = _row_text(row, "filename")
    stem = Path(filename).stem
    if "_joint_" not in stem:
        return []
    suffix = stem.split("_joint_", 1)[1]
    result: list[str] = []
    for segment in suffix.split("__"):
        match = _JOINT_FILENAME_SEGMENT.match(segment)
        if match is not None:
            chain = _decode_filename_component(match.group("chain"))
            if chain and chain not in result:
                result.append(chain)
    return result


def _locator_chain(locator: Mapping[str, Any]) -> str:
    value = locator.get("chain", locator.get("label_asym_id", ""))
    return str(value).strip()


def _normalise_locators(
    entries: Iterable[Mapping[str, Any]], *, field: str, row_context: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(entries):
        item = dict(raw)
        chain = _locator_chain(item)
        if not chain:
            raise ValidationInputError(
                f"{row_context}: {field}[{index}] has no chain locator"
            )
        item["chain"] = chain
        if "res_id" in item and item["res_id"] not in (None, "", ".", "?"):
            try:
                item["res_id"] = int(item["res_id"])
            except (TypeError, ValueError) as exc:
                raise ValidationInputError(
                    f"{row_context}: {field}[{index}] has a non-integer res_id"
                ) from exc
        if item.get("ins_code") in (None, ".", "?"):
            item["ins_code"] = ""
        elif "ins_code" in item:
            item["ins_code"] = str(item["ins_code"])
        if "atom_id" in item and item["atom_id"] not in (None, "", ".", "?"):
            try:
                item["atom_id"] = int(item["atom_id"])
            except (TypeError, ValueError) as exc:
                raise ValidationInputError(
                    f"{row_context}: {field}[{index}] has a non-integer atom_id"
                ) from exc
        result.append(item)
    return result


def _bond_endpoint_entries(bonds: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    for bond in bonds:
        for key in ("atom1", "atom2"):
            endpoint = bond.get(key)
            if isinstance(endpoint, Mapping):
                yield dict(endpoint)


def _residue_key_from_row(atoms: Any, row: int) -> tuple[str, int, str]:
    chain = str(
        getattr(atoms, "label_asym_id", getattr(atoms, "chain_id"))[row]
    )
    return chain, int(atoms.residue_uid[row]), str(atoms.ins_code[row])


def _residue_order(atoms: Any, chain: str) -> list[int]:
    chain_values = np.asarray(
        getattr(atoms, "label_asym_id", getattr(atoms, "chain_id"))
    ).astype(str)
    rows = np.flatnonzero(chain_values == chain)
    result: list[int] = []
    seen: set[int] = set()
    for row in rows.tolist():
        uid = int(atoms.residue_uid[row])
        if uid not in seen:
            seen.add(uid)
            result.append(uid)
    return result


def _cyclic_chains_from_graph(atoms: Any) -> tuple[str, ...]:
    """Return chains with an actual terminal backbone C--N graph edge."""

    chain_values = np.asarray(
        getattr(atoms, "label_asym_id", getattr(atoms, "chain_id"))
    ).astype(str)
    residue_values = np.asarray(atoms.residue_uid, dtype=np.int64)
    mol_values = np.asarray(atoms.mol_type).astype(str)
    atom_names = np.asarray(atoms.atom_name).astype(str)
    first_last: dict[str, tuple[int, int]] = {}
    for chain in dict.fromkeys(chain_values.tolist()):
        if not np.any((chain_values == chain) & (mol_values == "protein")):
            continue
        residues = _residue_order(atoms, chain)
        if len(residues) >= 2:
            first_last[chain] = (residues[0], residues[-1])
    found: set[str] = set()
    if getattr(atoms, "bonds", None) is None:
        return ()
    standard = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
    for raw_i, raw_j, _order in atoms.bonds.as_array().tolist():
        if int(_order) != 1:
            continue
        i, j = int(raw_i), int(raw_j)
        if any(str(atoms.res_name[k]).upper() not in standard or mol_values[k] != "protein" for k in (i, j)):
            continue
        if chain_values[i] != chain_values[j]:
            continue
        if {atom_names[i], atom_names[j]} != {"C", "N"}:
            continue
        chain = str(chain_values[i])
        terminals = first_last.get(chain)
        if terminals is None:
            continue
        first_uid, last_uid = terminals
        uid_pair = {int(residue_values[i]), int(residue_values[j])}
        if uid_pair != {first_uid, last_uid}:
            continue
        c_row = i if atom_names[i] == "C" else j
        n_row = i if atom_names[i] == "N" else j
        if int(residue_values[c_row]) == last_uid and int(residue_values[n_row]) == first_uid:
            found.add(chain)
    return tuple(chain for chain in dict.fromkeys(chain_values.tolist()) if chain in found)


def _parse_contig_positions(value: str) -> dict[str, tuple[tuple[int, str], ...]]:
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    if not value.strip():
        return {}
    for raw_segment in value.split("/"):
        segment = raw_segment.strip()
        match = _SOURCE_SEGMENT.fullmatch(segment)
        if match is None:
            # Numeric segments are newly designed gaps, not source motifs.
            continue
        chain = match.group(1) or match.group(2)
        start = int(match.group(3))
        start_ins = match.group(4) or ""
        end = int(match.group(5)) if match.group(5) else start
        end_ins = match.group(6) or ""
        if end < start:
            continue
        if end == start:
            grouped[chain].append((start, start_ins))
            continue
        for position in range(start, end + 1):
            grouped[chain].append((position, start_ins if position == start else end_ins if position == end else ""))
    return {chain: tuple(values) for chain, values in grouped.items()}


def _parse_tip_positions(value: str) -> dict[str, dict[tuple[int, str], tuple[str, ...]]]:
    grouped: dict[str, dict[tuple[int, str], tuple[str, ...]]] = defaultdict(dict)
    if not value.strip():
        return {}
    for raw_segment in value.split(";"):
        segment = raw_segment.strip()
        if not segment:
            continue
        match = _TIP_SEGMENT.fullmatch(segment)
        if match is None:
            raise ValidationInputError(f"invalid tip_atoms segment: {segment!r}")
        chain = match.group(1) or match.group(2)
        position = int(match.group(3))
        ins_code = match.group(4) or ""
        names = tuple(atom.strip() for atom in match.group(5).split(",") if atom.strip())
        if not names:
            raise ValidationInputError(f"tip_atoms segment has no atoms: {segment!r}")
        key = (position, ins_code)
        previous = grouped[chain].get(key)
        if previous is not None and previous != names:
            raise ValidationInputError(f"duplicate conflicting tip_atoms segment: {segment!r}")
        grouped[chain][key] = names
    return {chain: dict(values) for chain, values in grouped.items()}


def _fixed_position_index(
    fixed_restypes: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for locator in fixed_restypes:
        chain = _locator_chain(locator)
        if chain:
            result[chain].append(dict(locator))
    return {chain: list(values) for chain, values in result.items()}


def _locator_rows(atoms: Any, locator: Mapping[str, Any]) -> list[int]:
    chains = np.asarray(
        getattr(atoms, "label_asym_id", getattr(atoms, "chain_id"))
    ).astype(str)
    mask = chains == _locator_chain(locator)
    if locator.get("res_id") not in (None, "", ".", "?"):
        mask &= np.asarray(atoms.res_id, dtype=np.int64) == int(locator["res_id"])
    if locator.get("ins_code") not in (None, "", ".", "?"):
        mask &= np.asarray(atoms.ins_code).astype(str) == str(locator["ins_code"])
    atom_name = locator.get("atom")
    if atom_name not in (None, "", ".", "?"):
        mask &= np.asarray(atoms.atom_name).astype(str) == str(atom_name)
    atom_id = locator.get("atom_id")
    if atom_id not in (None, "", ".", "?"):
        # The public constraint ledger stores 1-based atom_site/PDB serials;
        # generated output is read in that same stable row order.
        expected = int(atom_id) - 1
        mask &= np.arange(len(np.asarray(atoms.coord)), dtype=np.int64) == expected
    return np.flatnonzero(mask).astype(np.int64).tolist()


def _validate_constraint_identity(
    atoms: Any,
    fixed_restypes: Iterable[Mapping[str, Any]],
    bond_constraints: Iterable[Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    for index, locator in enumerate(fixed_restypes):
        rows = _locator_rows(atoms, locator)
        if not rows:
            reasons.append(f"fixed_restypes[{index}] does not resolve to an output residue")
    graph: dict[tuple[int, int], int] = {}
    if getattr(atoms, "bonds", None) is not None:
        graph = {
            tuple(sorted((int(left), int(right)))): int(_order)
            for left, right, _order in atoms.bonds.as_array().tolist()
        }
    for index, bond in enumerate(bond_constraints):
        endpoints: list[int] = []
        for endpoint_name in ("atom1", "atom2"):
            rows = _locator_rows(atoms, bond[endpoint_name])
            if len(rows) != 1:
                reasons.append(
                    f"bond_constraints[{index}].{endpoint_name} resolves to {len(rows)} atoms"
                )
            else:
                endpoints.append(rows[0])
        if len(endpoints) == 2:
            pair = tuple(sorted(endpoints))
            if pair not in graph:
                if bond.get("required_output", True):
                    reasons.append(f"bond_constraints[{index}] is absent from the source bond graph")
                continue
            requested = str(bond.get("order", "any")).lower()
            accepted = {"any": set(range(10)), "single": {1}, "double": {2}, "triple": {3},
                        "aromatic": {5, 6, 7, 9}}
            if requested not in accepted or graph[pair] not in accepted[requested]:
                reasons.append(f"bond_constraints[{index}] order {requested!r} disagrees with source graph order {graph[pair]}")
    return reasons


def _finalize_loaded_identity(atoms: Any, design_chains: Iterable[str]) -> Any:
    """Finalize the canonical authoring identity required by model workers.

    ``read_structure()`` deliberately stops at the identity phase, leaving
    ``token_uid`` unset. Validation workers consume the same AtomArray ABI as
    generation, so finalize UIDs and representative tokens at this boundary
    while leaving coordinates, source labels, and the explicit bond graph
    untouched. Lightweight test doubles are returned unchanged.
    """

    if not hasattr(atoms, "get_annotation_categories") or not hasattr(atoms, "token_uid"):
        return atoms
    categories = set(atoms.get_annotation_categories())
    if not set(UID_FIELDS).issubset(categories):
        return atoms
    try:
        if any(
            np.any(np.asarray(atoms.get_annotation(name), dtype=np.int64) < 0)
            for name in UID_FIELDS
            if name != "token_uid"
        ):
            assign_identity_uids(atoms)
        # The validation input is an authoring structure, not a compiled
        # carrier assembly. No carrier residues are invented here; unusual
        # modified polymers fail closed with the tokenizer's precise error.
        tokenize_atom_array(atoms, carrier_residue_uids=())
        validate_atom_array(atoms, phase="tokenized")
    except Exception as exc:
        raise ValidationInputError(f"cannot finalize input identity/token axis: {exc}") from exc

    selected = {str(value) for value in design_chains}
    chain_values = np.asarray(atoms.chain_id).astype(str)
    if "label_asym_id" in categories:
        chain_values = np.asarray(atoms.label_asym_id).astype(str)
    atoms.is_design = np.isin(chain_values, sorted(selected))
    return atoms


def discover_inputs(
    run_dirs: Path | str | Iterable[Path | str],
    task_names: Iterable[str] | None = None,
) -> Iterator[InputDescriptor]:
    """Yield one descriptor per ``result.csv`` row in selected task folders.

    ``run_dirs`` may be one run path or an iterable of runs.  Only direct task
    children and their named result-file paths are visited.  ``validate`` and
    ``original`` are never treated as input tasks.
    """

    if isinstance(run_dirs, (str, Path)):
        roots = [Path(run_dirs)]
    else:
        roots = [Path(value) for value in run_dirs]
    selected = None if task_names is None else {str(value) for value in task_names}
    for raw_run in roots:
        run_dir = raw_run.expanduser().resolve()
        if not run_dir.is_dir():
            continue
        if (run_dir / "result.csv").is_file():
            # A direct task directory is useful for retrying one failed task;
            # its parent remains the run identity in the descriptor.
            task_dirs = [run_dir]
            descriptor_run_dir = run_dir.parent
        else:
            task_dirs = sorted(
                path
                for path in run_dir.iterdir()
                if path.is_dir() and path.name not in {"validate", "original"} and not path.name.startswith(".")
            )
            descriptor_run_dir = run_dir
        for task_dir in task_dirs:
            if selected is not None and task_dir.name not in selected:
                continue
            result_path = task_dir / "result.csv"
            if not result_path.is_file():
                continue
            seen_samples: set[str] = set()
            with result_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row_index, raw_row in enumerate(reader, start=2):
                    row = {str(key): ("" if value is None else str(value)) for key, value in raw_row.items() if key is not None}
                    filename = _row_text(row, "filename")
                    candidate = (task_dir / filename).resolve() if filename else task_dir / ""
                    sample_id = Path(filename).stem if filename else _row_text(row, "base_sample_id") or f"row{row_index}"
                    descriptor_error: str | None = None
                    if sample_id in seen_samples:
                        descriptor_error = f"duplicate sample identity in task result.csv: {sample_id!r}"
                    seen_samples.add(sample_id)
                    if not filename:
                        descriptor_error = descriptor_error or "result row has no filename"
                    try:
                        relative_candidate = candidate.relative_to(task_dir.resolve())
                    except ValueError:
                        relative_candidate = None
                        descriptor_error = descriptor_error or "result filename escapes task directory"
                    if relative_candidate is not None and any(part in {"validate", "original"} for part in relative_candidate.parts):
                        descriptor_error = descriptor_error or "result filename points into validate/original"
                    yield InputDescriptor(
                        run_dir=descriptor_run_dir,
                        task_dir=task_dir,
                        result_path=result_path,
                        path=candidate,
                        row=row,
                        sample_id=sample_id,
                        row_index=row_index,
                        error=descriptor_error,
                    )


def load_input(
    path: Path | str,
    row: Mapping[str, Any],
    task_dir: Path | str,
) -> ValidationInput:
    """Load exactly one row-named structure and resolve its metadata."""

    task_root = Path(task_dir).expanduser().resolve()
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = task_root / source
    source = source.resolve()
    try:
        source.relative_to(task_root)
    except ValueError as exc:
        raise ValidationInputError(
            f"input path escapes task directory: {source}"
        ) from exc
    relative_source = source.relative_to(task_root)
    if any(part in {"validate", "original"} for part in relative_source.parts):
        raise ValidationInputError(f"input path points into validate/original: {source}")
    if not source.is_file():
        raise ValidationInputError(f"input structure does not exist: {source}")
    if source.suffix.lower() not in {".cif", ".pdb"}:
        raise ValidationInputError(f"unsupported validation structure suffix: {source.suffix}")

    row_copy = {str(key): ("" if value is None else str(value)) for key, value in row.items()}
    sample_id = source.stem
    row_context = f"{task_root.name}/{sample_id}"
    fixed_restypes = _normalise_locators(
        _parse_json_list(row_copy.get("fixed_restypes", ""), field="fixed_restypes", row_context=row_context),
        field="fixed_restypes",
        row_context=row_context,
    )
    raw_bonds = _parse_json_list(
        row_copy.get("bond_constraints", ""), field="bond_constraints", row_context=row_context
    )
    bond_constraints: list[dict[str, Any]] = []
    for index, raw_bond in enumerate(raw_bonds):
        bond = dict(raw_bond)
        for endpoint_name in ("atom1", "atom2"):
            endpoint = bond.get(endpoint_name)
            if not isinstance(endpoint, Mapping):
                raise ValidationInputError(
                    f"{row_context}: bond_constraints[{index}] missing {endpoint_name}"
                )
            bond[endpoint_name] = _normalise_locators(
                [endpoint],
                field=f"bond_constraints[{index}].{endpoint_name}",
                row_context=row_context,
            )[0]
        bond_constraints.append(bond)

    antibody_row = "heavy_role" in row
    explicit_design_chains = "design_chains" in row
    design_chain_metadata = (
        _parse_design_chains(row.get("design_chains"), row_context=row_context)
        if explicit_design_chains
        else None
    )
    if explicit_design_chains:
        # Keep programmatic callers and report rebuilds on the same compact
        # representation used by result.csv.
        row_copy["design_chains"] = json.dumps(
            design_chain_metadata,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    atoms = read_structure(source)
    if hasattr(atoms, "label_asym_id"):
        labels = np.asarray(atoms.label_asym_id).astype(str)
        authors = np.asarray(atoms.chain_id).astype(str)
        if any(len(set(labels[authors == author])) > 1 for author in set(authors)):
            raise ValidationInputError("Repeated author chain spans multiple TER/label segments; use unique source chain IDs")
    if antibody_row:
        design_chain_metadata, fixed_restypes = _antibody_chain_metadata(row_copy, atoms, row_context)
        row_copy["design_chains"] = json.dumps(design_chain_metadata, separators=(",", ":"))
        row_copy["design_modality"] = "joint" if len(design_chain_metadata)>1 else "protein"
        row_copy["fixed_restypes"] = json.dumps(fixed_restypes, separators=(",", ":"))
        row_copy["bond_constraints"] = "[]"
    if design_chain_metadata is not None:
        # The final CIF label_asym_id values are authoritative when the new
        # metadata is present. Never replace a complete list with constraint
        # endpoints: unconstrained design chains are part of the contract too.
        design_chains = [str(item["chain"]) for item in design_chain_metadata]
    else:
        design_chains = _chain_list_from_value(_row_text(row_copy, "design_chain_id"))
        if not design_chains and _row_text(row_copy, "design_modality") == "joint":
            design_chains = _infer_joint_design_chains(row_copy)

    atoms = _finalize_loaded_identity(atoms, design_chains)
    cyclic_offset_chains = tuple(
        str(item["chain"]) for item in (design_chain_metadata or ())
        if item.get("cyclic_offset", False)
    )
    for chain in cyclic_offset_chains:
        labels = np.asarray(getattr(atoms, "label_asym_id", atoms.chain_id)).astype(str)
        selected = labels == chain
        if not np.any(selected) or not np.all(np.asarray(atoms.mol_type)[selected] == "protein"):
            raise ValidationInputError(f"{row_context}: cyclic_offset chain {chain!r} is not a loaded protein design chain")
        if len(_residue_order(atoms, chain)) < 3:
            raise ValidationInputError(f"{row_context}: cyclic_offset chain {chain!r} requires at least three residues")
    # Fixed restypes describe designed polymers only. Keep legacy receptor
    # metadata in row_copy, but do not turn it into design constraints.
    fixed_restypes = [item for item in fixed_restypes
                      if _locator_chain(item) in design_chains]

    contig_string = _row_text(row_copy, "contig_string")
    tip_atoms = _row_text(row_copy, "tip_atoms")
    motif_positions = _parse_contig_positions(contig_string)
    tip_positions = _parse_tip_positions(tip_atoms)
    reasons: list[str] = _validate_constraint_identity(atoms, fixed_restypes, bond_constraints)
    for chain, positions in motif_positions.items():
        if design_chains and chain not in design_chains:
            reasons.append(f"contig motif chain {chain!r} is not listed in design_chains")
        if not positions:
            reasons.append(f"contig motif chain {chain!r} has no positions")
    return ValidationInput(
        path=source,
        task_dir=task_root,
        sample_id=sample_id,
        row=row_copy,
        atoms=atoms,
        design_chains=tuple(design_chains),
        fixed_restypes=fixed_restypes,
        bond_constraints=bond_constraints,
        contig_string=contig_string,
        tip_atoms=tip_atoms,
        cyclic_chains=_cyclic_chains_from_graph(atoms),
        cyclic_offset_chains=cyclic_offset_chains,
        fixed_positions=_fixed_position_index(fixed_restypes),
        motif_positions=motif_positions,
        tip_positions=tip_positions,
        reasons=tuple(reasons),
    )
