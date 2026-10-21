from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import gzip
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import biotite.structure as struc
import numpy as np

from ..utils.atoms import (
    COMMON_STANDARD_PARENT,
    DNA_ATOM27,
    DNA_RES_ATOMS,
    PROTEIN_ATOM37,
    PROTEIN_RES_ATOMS,
    RNA_ATOM27,
    RNA_RES_ATOMS,
    SIDECHAIN_TIP_ATOMS,
    assign_identity_uids,
    concatenate_atom_arrays,
    create_atom_array,
    tokenize_atom_array,
    validate_atom_array,
)
from ..utils.reader import load_ccd_standard_parents, read_structure
from ..utils import reader as _reader_module
from ..utils.chemistry import (
    TrustedCCDComponent,
    project_trusted_ccd_component,
    trusted_ccd_private_molecule,
)
from ..utils.ligand_identity import (
    build_validated_ligand_identity,
    compute_ligand_component_signature,
    validated_ligand_identities_from_ccd,
)
from ..utils.ccd import (
    CCDComponent,
    load_ccd_components,
    load_embedded_ccd,
    load_packaged_polymer_ccd,
    _LazyTypedComponents,
    resolve_ccd_component,
)
from .crop import project_source_crop
from .model_tokens import ModelTokenLayout, build_model_token_layout
from .records import (
    CompiledSample,
    DesignChainSpec,
    DesignInputError,
    DesignSampleError,
    EdgePolicy,
    ExpandedRequest,
    ModifiedPolymerCarrier,
    MotifOutputFields,
    TaskSpec,
    TokenPairCondition,
    TopologyCondition,
)


_AA1_TO_3 = {
    one: three
    for one, three in zip(
        "ARNDCQEGHILKMFPSTWYV",
        ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"),
    )
}
_AA3_TO_1 = {value: key for key, value in _AA1_TO_3.items()}
_PUBLIC_ALPHABET = {
    "protein": set(_AA1_TO_3),
    "rna": set("ACGU"),
    "dna": set("ACGT"),
}
_UNKNOWN_RESIDUE = {"protein": "UNK", "rna": "N", "dna": "DN"}
_INVENTORIES = {
    "protein": PROTEIN_RES_ATOMS,
    "rna": RNA_RES_ATOMS,
    "dna": DNA_RES_ATOMS,
}
_CENTER_ATOM = {"protein": "CA", "rna": "C1'", "dna": "C1'"}
_ORDER_TO_BIOTITE = {
    "single": int(struc.BondType.SINGLE),
    "double": int(struc.BondType.DOUBLE),
    "triple": int(struc.BondType.TRIPLE),
    "aromatic": int(struc.BondType.AROMATIC),
}
_UNKNOWN_POSITIVE_ORDERS = frozenset({"any", "unknown"})
_BIOTITE_TO_ORDER = {value: key for key, value in _ORDER_TO_BIOTITE.items()}
_AROMATIC_BIOTITE_ORDERS = frozenset(
    int(value)
    for value in (
        struc.BondType.AROMATIC,
        struc.BondType.AROMATIC_SINGLE,
        struc.BondType.AROMATIC_DOUBLE,
        struc.BondType.AROMATIC_TRIPLE,
    )
)
_STANDARD_BY_MODALITY = {
    "protein": frozenset(PROTEIN_RES_ATOMS),
    "rna": frozenset(RNA_RES_ATOMS),
    "dna": frozenset(DNA_RES_ATOMS),
}
_PARENT_MODALITY = {
    name: modality
    for modality, names in _STANDARD_BY_MODALITY.items()
    for name in names
}

_SOURCE_SEGMENT = re.compile(
    r"^(?:\[([^\]]+)\]|([A-Za-z]))(\d+)([^\d\-/]*)"
    r"(?:-(\d+)([^\d\-/]*))?$"
)
_FINAL_RANGE = re.compile(r"^(?:\[([^\]]+)\]|([A-Za-z]))?(\-?\d+)(?:-(\-?\d+))?$")


@dataclass(frozen=True, slots=True)
class _CarrierCandidate:
    source_residue_uid: int
    component: CCDComponent
    canonical_by_origin_uid: Mapping[int, str]
    standard_parent: str | None = None


@dataclass(frozen=True, slots=True)
class _CarrierOccurrence:
    source_residue_uid: int
    component: CCDComponent
    rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PreparedTaskSource:
    """Immutable source-stage facts shared by expanded requests of one task.

    The source AtomArray instances are owned by this snapshot.  Callers of
    compile_request receive deep copies before any per-request materialization
    mutates annotations or bonds.
    """

    task: TaskSpec
    ccd_path: Path | None
    source: struc.AtomArray | None
    asym_order_source: struc.AtomArray | None
    source_declared_bonds: tuple[tuple[int, int, int], ...]
    source_carrier_candidates: Mapping[int, _CarrierCandidate]
    source_parent_by_residue: Mapping[int, str]
    source_modified_residue_uids: frozenset[int]
    forced_chain_break_keys: tuple[tuple[str, int, str], ...]
    parent_sources: Mapping[str, str | None]
    embedded_ccd: Mapping[str, CCDComponent]
    configured_ccd: Mapping[str, CCDComponent]
    packaged_ccd: Mapping[str, CCDComponent]
    condition: Mapping[str, Any]
    design_ids: frozenset[str]

    def __getstate__(self) -> dict[str, Any]:
        """Return a pickle-safe copy of the immutable source snapshot.

        ``MappingProxyType`` deliberately protects a prepared snapshot from
        accidental mutation, but it is not itself pickleable.  Keep the public
        immutable representation and serialize only plain copies of its maps;
        :meth:`__setstate__` reinstates the same protections in the worker.
        """

        return {
            "task": self.task,
            "ccd_path": self.ccd_path,
            "source": _clone_source_atom_array(self.source),
            "asym_order_source": _clone_source_atom_array(self.asym_order_source),
            "source_declared_bonds": tuple(self.source_declared_bonds),
            "source_carrier_candidates": {
                int(residue_uid): {
                    "source_residue_uid": int(candidate.source_residue_uid),
                    "component": candidate.component,
                    "canonical_by_origin_uid": dict(
                        candidate.canonical_by_origin_uid
                    ),
                    "standard_parent": candidate.standard_parent,
                }
                for residue_uid, candidate in self.source_carrier_candidates.items()
            },
            "source_parent_by_residue": dict(self.source_parent_by_residue),
            "source_modified_residue_uids": frozenset(
                self.source_modified_residue_uids
            ),
            "forced_chain_break_keys": tuple(self.forced_chain_break_keys),
            "parent_sources": dict(self.parent_sources),
            "embedded_ccd": dict(self.embedded_ccd),
            "configured_ccd": dict(self.configured_ccd),
            "packaged_ccd": (
                self.packaged_ccd if isinstance(self.packaged_ccd, _LazyTypedComponents)
                else dict(self.packaged_ccd)
            ),
            "condition": deepcopy(dict(self.condition)),
            "design_ids": frozenset(self.design_ids),
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore the immutable source snapshot after worker deserialization."""

        candidates = {
            int(residue_uid): _CarrierCandidate(
                source_residue_uid=int(candidate["source_residue_uid"]),
                component=candidate["component"],
                canonical_by_origin_uid=dict(
                    candidate["canonical_by_origin_uid"]
                ),
                standard_parent=candidate["standard_parent"],
            )
            for residue_uid, candidate in state["source_carrier_candidates"].items()
        }
        object.__setattr__(self, "task", state["task"])
        object.__setattr__(self, "ccd_path", state["ccd_path"])
        object.__setattr__(self, "source", _clone_source_atom_array(state["source"]))
        object.__setattr__(
            self,
            "asym_order_source",
            _clone_source_atom_array(state["asym_order_source"]),
        )
        object.__setattr__(
            self,
            "source_declared_bonds",
            tuple(state["source_declared_bonds"]),
        )
        object.__setattr__(
            self,
            "source_carrier_candidates",
            _freeze_carrier_candidates(candidates),
        )
        object.__setattr__(
            self,
            "source_parent_by_residue",
            MappingProxyType(dict(state["source_parent_by_residue"])),
        )
        object.__setattr__(
            self,
            "source_modified_residue_uids",
            frozenset(state["source_modified_residue_uids"]),
        )
        object.__setattr__(
            self,
            "forced_chain_break_keys",
            tuple(state["forced_chain_break_keys"]),
        )
        object.__setattr__(
            self,
            "parent_sources",
            MappingProxyType(dict(state["parent_sources"])),
        )
        object.__setattr__(
            self,
            "embedded_ccd",
            MappingProxyType(dict(state["embedded_ccd"])),
        )
        object.__setattr__(
            self,
            "configured_ccd",
            MappingProxyType(dict(state["configured_ccd"])),
        )
        object.__setattr__(
            self,
            "packaged_ccd",
            (state["packaged_ccd"] if isinstance(state["packaged_ccd"], _LazyTypedComponents)
             else MappingProxyType(dict(state["packaged_ccd"]))),
        )
        object.__setattr__(
            self,
            "condition",
            MappingProxyType(deepcopy(dict(state["condition"]))),
        )
        object.__setattr__(
            self,
            "design_ids",
            frozenset(state["design_ids"]),
        )

@dataclass(frozen=True, slots=True)
class _SequenceResidue:
    """One parsed ``condition.sequence`` residue.

    Bracketed residues are kept as a typed sidecar until the authoring
    inventory is materialized.  This is intentionally separate from the
    standard residue-name list: putting ``[MSE]`` into that list would make it
    look like a model vocabulary token and would also change the canonical
    standard-sequence path.
    """

    name: str
    bracket_component: str | None = None

    @property
    def is_bracketed(self) -> bool:
        return self.bracket_component is not None


def _source_order_name(order: int) -> str:
    value = int(order)
    if value == int(struc.BondType.ANY):
        return "any"
    if value in _AROMATIC_BIOTITE_ORDERS:
        return "aromatic"
    result = _BIOTITE_TO_ORDER.get(value)
    if result is None:
        raise DesignInputError(f"unsupported source covalent bond order {value}")
    return result


@lru_cache(maxsize=8)
def _configured_polymer_ccd(path_text: str) -> Mapping[str, CCDComponent]:
    """Load v3/raw polymer CCD while leaving the legacy ligand v2 lane alone."""

    path = Path(path_text)
    if path.name.endswith(".json.gz"):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                schema = json.load(stream).get("schema_version")
        except Exception as exc:
            raise DesignInputError(f"cannot inspect configured CCD {path}: {exc}") from exc
        if schema == 2:
            return {}
    return load_ccd_components(path, source_kind="configured")


def _polymer_ccd_sources(
    input_path: Path | None,
    ccd_path: Path | None,
) -> tuple[
    Mapping[str, CCDComponent],
    Mapping[str, CCDComponent],
    Mapping[str, CCDComponent],
]:
    packaged = load_packaged_polymer_ccd()
    configured: Mapping[str, CCDComponent] = (
        {} if ccd_path is None else _configured_polymer_ccd(str(ccd_path.resolve()))
    )
    embedded: Mapping[str, CCDComponent] = (
        {}
        if input_path is None or input_path.suffix.lower() not in {".cif", ".mmcif"}
        else load_embedded_ccd(input_path)
    )
    return embedded, configured, packaged


def _resolve_proxy_parent(
    component_id: str,
    modality: str,
    *,
    embedded: Mapping[str, CCDComponent],
    configured: Mapping[str, CCDComponent],
    packaged: Mapping[str, CCDComponent],
) -> str:
    unknown = _UNKNOWN_RESIDUE[modality]

    def visit(current: str, seen: frozenset[str]) -> str:
        normalized = str(current).strip().upper()
        if _PARENT_MODALITY.get(normalized) == modality:
            return normalized
        if normalized in seen:
            return unknown
        resolution = resolve_ccd_component(
            normalized,
            embedded=embedded,
            configured=configured,
            packaged=packaged,
            include_packaged=False,
        )
        if resolution.status == "missing":
            builtin = COMMON_STANDARD_PARENT.get(normalized)
            return (
                builtin
                if builtin is not None and _PARENT_MODALITY.get(builtin) == modality
                else unknown
            )
        component = resolution.component
        if component is None or (
            component.polymer_modality is not None
            and component.polymer_modality != modality
        ):
            return unknown
        parents = component.parent_ids
        if parents is None or len(parents) != 1:
            return unknown
        return visit(parents[0], seen | {normalized})

    return visit(component_id, frozenset())


def _ccd_parent_hints(
    embedded: Mapping[str, CCDComponent],
    configured: Mapping[str, CCDComponent],
    packaged: Mapping[str, CCDComponent],
) -> dict[str, str]:
    """Return modality-safe reader hints; model proxy selection is separate."""

    result: dict[str, str] = {}
    from src.common.chemistry.ccd import polymer_metadata
    if isinstance(packaged, _LazyTypedComponents):
        for component_id, metadata in polymer_metadata().items():
            modality = metadata["polymer_modality"]
            if modality in _UNKNOWN_RESIDUE:
                result[component_id] = _UNKNOWN_RESIDUE[modality]
        mappings = (configured, embedded)
    else:
        mappings = (packaged, configured, embedded)
    for mapping in mappings:
        for component_id, component in mapping.items():
            if component.polymer_modality in _UNKNOWN_RESIDUE:
                result[component_id] = _UNKNOWN_RESIDUE[component.polymer_modality]
    return result


def _source_declared_bonds(
    path: Path,
    source: struc.AtomArray,
    *,
    parent_sources: Mapping[str, str | None],
    declared_polymer_chain_modalities: Mapping[str, str],
) -> tuple[tuple[int, int, int], ...]:
    """Re-read only file-declared bonds, excluding reader polymer templates.

    ``read_structure()`` intentionally carries legacy residue-name templates for
    ordinary polymers.  Those edges have no provenance annotation, so a
    carrier must recover the pre-template file facts here; otherwise a PDB
    CONECT ``ANY`` edge could be silently refined to template ``SINGLE``.
    The repeated atom read is checked row-for-row before its indices are used.
    """

    suffix = path.suffix.lower()
    if suffix == ".pdb":
        raw_atoms, bonds, _ = _reader_module._read_pdb(
            path,
            parent_sources=parent_sources,
            declared_polymer_chain_modalities=declared_polymer_chain_modalities,
        )
    elif suffix in {".cif", ".mmcif"}:
        raw_atoms, block, _, altloc_ledger = _reader_module._read_cif_rows(
            path,
            parent_sources=parent_sources,
            declared_polymer_chain_modalities=declared_polymer_chain_modalities,
        )
        bonds = _reader_module._cif_component_bonds(
            block, raw_atoms
        ) + _reader_module._cif_struct_conn_bonds(
            block, raw_atoms, altloc_ledger
        )
    else:
        raise DesignInputError(f"unsupported source structure suffix {suffix!r}")
    identity_fields = (
        "chain_id",
        "res_id",
        "ins_code",
        "res_name",
        "atom_name",
        "element",
        "atom_uid",
    )
    if len(raw_atoms) != len(source) or any(
        not np.array_equal(
            raw_atoms.get_annotation(field), source.get_annotation(field)
        )
        for field in identity_fields
    ):
        raise DesignInputError(
            "file-declared bond rows do not align with normalized source AtomUIDs"
        )
    return tuple((int(i), int(j), int(order)) for i, j, order in bonds)


def _parent_sources(
    input_path: Path | None,
    ccd_path: Path | None,
) -> dict[str, str | None]:
    """Resolve parents with input-CIF > release CCD > built-in precedence."""

    result: dict[str, str | None] = dict(COMMON_STANDARD_PARENT)
    if ccd_path is not None and not ccd_path.name.endswith(".json.gz"):
        result.update(load_ccd_standard_parents(ccd_path))
    if input_path is not None and input_path.suffix.lower() in {".cif", ".mmcif"}:
        result.update(load_ccd_standard_parents(input_path))
    return result


def _identifier_sort_key(value: object) -> tuple[int, object]:
    text = str(value).strip()
    if text in {"", ".", "?"}:
        return (2, "")
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _insertion_sort_key(value: object) -> tuple[int, str]:
    """Order the base residue before its alphabetic insertion variants."""

    text = str(value).strip()
    return (0, "") if text in {"", ".", "?"} else (1, text)


def _residue_order_key(
    atoms: struc.AtomArray,
    residue_uid: int,
) -> tuple[tuple[int, object], tuple[int, object], tuple[int, object], str, int]:
    """Return a stable structural identity key for one authoring residue.

    Residue rows in an AtomArray need not be stored in authoring order.  Prefer
    the explicit label sequence identity, then the normalized residue ID and
    insertion/author fields, and use the internal UID only as a final tie-break
    for genuinely ambiguous duplicate identities.
    """

    rows = np.where(np.asarray(atoms.residue_uid, dtype=np.int64) == int(residue_uid))[0]
    if not len(rows):
        raise DesignSampleError(f"residue UID {residue_uid} is absent")
    labels = sorted({str(value).strip() for value in atoms.label_seq_id[rows]})
    res_ids = sorted({int(value) for value in atoms.res_id[rows]})
    ins_codes = sorted({str(value).strip() for value in atoms.ins_code[rows]})
    auth_seq_ids = sorted({str(value).strip() for value in atoms.auth_seq_id[rows]})
    names = sorted({str(value).strip().upper() for value in atoms.res_name[rows]})
    label = labels[0] if labels else ""
    structural = label if label not in {"", ".", "?"} else (res_ids[0] if res_ids else "")
    return (
        _identifier_sort_key(structural),
        _insertion_sort_key(ins_codes[0] if ins_codes else ""),
        _identifier_sort_key(auth_seq_ids[0] if auth_seq_ids else ""),
        names[0] if names else "",
        int(residue_uid),
    )


def _model_parent_by_residue(
    atoms: struc.AtomArray,
    parent_sources: Mapping[str, str | None],
    *,
    excluded_residue_uids: frozenset[int] = frozenset(),
) -> dict[int, str]:
    result: dict[int, str] = {}
    for residue_uid in np.unique(atoms.residue_uid[atoms.mol_type != "ligand"]):
        if int(residue_uid) in excluded_residue_uids:
            continue
        rows = np.where(atoms.residue_uid == residue_uid)[0]
        modality = str(atoms.mol_type[rows[0]])
        names = set(str(value) for value in atoms.res_name[rows])
        if len(names) != 1:
            raise DesignSampleError(f"polymer residue UID {residue_uid} has mixed names")
        name = next(iter(names)).upper()
        if name in _STANDARD_BY_MODALITY[modality]:
            continue
        parent = parent_sources.get(name)
        if parent is None:
            parent = _UNKNOWN_RESIDUE[modality]
        if _PARENT_MODALITY.get(parent) != modality:
            raise DesignSampleError(
                f"modified polymer {name!r} parent {parent!r} changes modality"
            )
        result[int(residue_uid)] = parent
    return result


def _source_carrier_candidates(
    source: struc.AtomArray,
    source_bonds: Sequence[tuple[int, int, int]],
    *,
    embedded: Mapping[str, CCDComponent],
    configured: Mapping[str, CCDComponent],
    packaged: Mapping[str, CCDComponent],
) -> tuple[dict[int, _CarrierCandidate], dict[int, str], set[int]]:
    """Classify source non-standard polymers as CCD carriers or proxy fallbacks."""

    candidates: dict[int, _CarrierCandidate] = {}
    proxy_parent: dict[int, str] = {}
    modified_residue_uids: set[int] = set()
    for raw_residue_uid in np.unique(
        source.residue_uid[source.mol_type != "ligand"]
    ):
        residue_uid = int(raw_residue_uid)
        rows = np.where(source.residue_uid == residue_uid)[0]
        modality = str(source.mol_type[rows[0]])
        names = {str(value).strip().upper() for value in source.res_name[rows]}
        if len(names) != 1:
            raise DesignInputError(
                f"source polymer residue UID {residue_uid} has mixed component IDs"
            )
        component_id = next(iter(names))
        if component_id in _STANDARD_BY_MODALITY[modality]:
            continue
        modified_residue_uids.add(residue_uid)
        resolved_parent = _resolve_proxy_parent(
            component_id,
            modality,
            embedded=embedded,
            configured=configured,
            packaged=packaged,
        )
        proxy_parent[residue_uid] = resolved_parent
        resolution = resolve_ccd_component(
            component_id,
            embedded=embedded,
            configured=configured,
            packaged=packaged,
            include_packaged=False,
        )
        component = resolution.component
        if component is not None and component.polymer_modality is None:
            raw_type = (component.raw_chem_type or "").upper()
            if "SACCHARIDE" in raw_type or "CARBOHYDRATE" in raw_type:
                raise DesignInputError(
                    f"CCD component {component_id!r} is carbohydrate/ligand chemistry"
                )
        if resolution.status != "resolved" or component is None:
            continue
        if component.polymer_modality != modality:
            raise DesignInputError(
                f"CCD component {component_id!r} modality "
                f"{component.polymer_modality!r} conflicts with source {modality!r}"
            )

        canonical_by_origin: dict[int, str] = {}
        valid = True
        for row in rows.tolist():
            source_name = str(source.atom_name[row])
            try:
                canonical = component.resolve_atom_id(source_name)
            except DesignInputError:
                valid = False
                break
            if canonical is None or canonical in canonical_by_origin.values():
                valid = False
                break
            ccd_atom = component.atom_by_id(canonical)
            source_element = str(source.element[row]).strip().upper()
            if (
                ccd_atom is None
                or ccd_atom.element is None
                or source_element != ccd_atom.element.upper()
                or source_element in {"H", "D", "T"}
            ):
                valid = False
                break
            canonical_by_origin[int(source.atom_uid[row])] = canonical
        if not valid or not canonical_by_origin:
            continue

        ccd_edges = {
            tuple(sorted((bond.atom_id_1, bond.atom_id_2))): bond.order
            for bond in component.bonds
        }
        row_set = set(int(value) for value in rows.tolist())
        for raw_i, raw_j, raw_order in source_bonds:
            row_i, row_j = int(raw_i), int(raw_j)
            if row_i not in row_set or row_j not in row_set:
                continue
            pair = tuple(
                sorted(
                    (
                        canonical_by_origin[int(source.atom_uid[row_i])],
                        canonical_by_origin[int(source.atom_uid[row_j])],
                    )
                )
            )
            ccd_order = ccd_edges.get(pair)
            if ccd_order is None:
                raise DesignInputError(
                    f"source positive edge {component_id}:{pair[0]}-{pair[1]} "
                    "is absent from the authoritative CCD graph"
                )
            source_order = _source_order_name(int(raw_order))
            if ccd_order != "aromatic":
                # Aromatic subtypes retain an explicit Kekule order. Compare
                # that fact to a CCD single/double/triple declaration locally;
                # generic AROMATIC has no such order and must still conflict.
                source_order = {
                    int(struc.BondType.AROMATIC_SINGLE): "single",
                    int(struc.BondType.AROMATIC_DOUBLE): "double",
                    int(struc.BondType.AROMATIC_TRIPLE): "triple",
                }.get(int(raw_order), source_order)
            if source_order != "any" and source_order != ccd_order:
                raise DesignInputError(
                    f"source/CCD bond order conflict for {component_id}:"
                    f"{pair[0]}-{pair[1]} ({source_order} versus {ccd_order})"
                )
        candidates[residue_uid] = _CarrierCandidate(
            source_residue_uid=residue_uid,
            component=component,
            canonical_by_origin_uid=dict(canonical_by_origin),
            standard_parent=(
                resolved_parent
                if resolved_parent in _STANDARD_BY_MODALITY[modality]
                and resolved_parent != _UNKNOWN_RESIDUE[modality]
                else None
            ),
        )
    return candidates, proxy_parent, modified_residue_uids


def _native_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DesignSampleError(f"{where} must be a mapping")
    return value


def _unique_residue_uids(atoms: struc.AtomArray, indices: np.ndarray) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for row in indices.tolist():
        uid = int(atoms.residue_uid[row])
        if uid not in seen:
            seen.add(uid)
            result.append(uid)
    return result


def _chain_residue_uids(atoms: struc.AtomArray, chain_id: str) -> list[int]:
    return _unique_residue_uids(atoms, np.where(atoms.chain_id == chain_id)[0])


def _source_residue_term(atoms: struc.AtomArray, term: str) -> list[int]:
    match = _SOURCE_SEGMENT.fullmatch(str(term).strip())
    if match is None:
        raise DesignSampleError(f"invalid source residue selector: {term!r}")
    chain_id = match.group(1) or match.group(2)
    if not chain_id or "]" in chain_id or "/" in chain_id:
        raise DesignSampleError(f"invalid source chain selector: {term!r}")
    start_number, start_ins = int(match.group(3)), match.group(4) or ""
    end_number = None if match.group(5) is None else int(match.group(5))
    end_ins = match.group(6) or ""
    residues = _chain_residue_uids(atoms, chain_id)
    if not residues:
        raise DesignSampleError(f"source selector chain {chain_id!r} does not exist")

    def endpoint(number: int, ins_code: str) -> int:
        found: list[int] = []
        for uid in residues:
            rows = np.where(atoms.residue_uid == uid)[0]
            label_seq_id = str(atoms.label_seq_id[rows[0]])
            try:
                structural_number = int(label_seq_id)
            except ValueError:
                structural_number = int(atoms.res_id[rows[0]])
            if structural_number != number:
                continue
            # CIF selectors use label_seq_id; PDB insertion selection uses ins_code.
            if ins_code and str(atoms.ins_code[rows[0]]) != ins_code:
                continue
            found.append(uid)
        if len(found) != 1:
            raise DesignSampleError(
                f"source selector endpoint {chain_id}{number}{ins_code} matched {len(found)} residues"
            )
        return found[0]

    start_uid = endpoint(start_number, start_ins)
    if end_number is None:
        return [start_uid]
    end_uid = endpoint(end_number, end_ins)
    start_index, end_index = residues.index(start_uid), residues.index(end_uid)
    if start_index > end_index:
        raise DesignSampleError(f"source residue range is reversed: {term!r}")
    return residues[start_index : end_index + 1]


def _source_selector(atoms: struc.AtomArray, text: str) -> list[int]:
    if not isinstance(text, str) or not text.strip():
        raise DesignSampleError("source selector must be a non-empty string")
    result: list[int] = []
    seen: set[int] = set()
    for term in text.split("/"):
        for uid in _source_residue_term(atoms, term):
            if uid not in seen:
                seen.add(uid)
                result.append(uid)
    return result


def _sample_closed_integer(minimum: int, maximum: int, rng: Any) -> int:
    """Draw from one inclusive integer interval with the inference RNG."""

    if minimum > maximum:
        raise AssertionError("empty closed integer interval")
    if hasattr(rng, "integers"):
        return int(rng.integers(minimum, maximum + 1))
    return int(rng.randint(minimum, maximum + 1))


def _contig_gap_range(segment: str) -> tuple[int, int]:
    if re.fullmatch(r"\d+", segment):
        value = int(segment)
        return value, value
    match = re.fullmatch(r"(\d+)-(\d+)", segment)
    if match is None:
        raise DesignSampleError(f"invalid contig gap segment: {segment!r}")
    minimum, maximum = int(match.group(1)), int(match.group(2))
    if minimum > maximum:
        raise DesignSampleError(f"reversed contig gap range: {segment!r}")
    return minimum, maximum


def _contig_length_bounds(
    entry: Mapping[str, Any], *, chain_id: str
) -> tuple[int, int] | None:
    has_min = "motif_min_length" in entry
    has_max = "motif_max_length" in entry
    if has_min != has_max:
        raise DesignSampleError(
            f"condition.contig.{chain_id} requires both motif_min_length and motif_max_length"
        )
    if not has_min:
        return None
    minimum, maximum = entry["motif_min_length"], entry["motif_max_length"]
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, int)
        or not isinstance(maximum, int)
        or minimum <= 0
        or maximum <= 0
        or minimum > maximum
    ):
        raise DesignSampleError(
            f"condition.contig.{chain_id} motif length bounds must be positive inclusive integers"
        )
    return int(minimum), int(maximum)


def _parse_contig(
    token: str,
    *,
    chain_id: str,
    source: struc.AtomArray,
    rng: Any,
    motif_length_bounds: tuple[int, int] | None = None,
) -> list[int | None]:
    if not isinstance(token, str) or not token.strip():
        raise DesignSampleError(f"condition.contig.{chain_id}.token must be non-empty")
    segments: list[list[int] | tuple[int, int]] = []
    for raw_segment in token.split("/"):
        segment = raw_segment.strip()
        source_match = _SOURCE_SEGMENT.fullmatch(segment)
        if source_match is None:
            segments.append(_contig_gap_range(segment))
            continue
        selected_chain = source_match.group(1) or source_match.group(2)
        if selected_chain != chain_id:
            raise DesignSampleError(
                f"contig for {chain_id!r} cannot copy source chain {selected_chain!r}"
            )
        segments.append(_source_residue_term(source, segment))

    fixed_length = sum(len(segment) for segment in segments if isinstance(segment, list))
    gap_ranges = [segment for segment in segments if isinstance(segment, tuple)]
    if motif_length_bounds is None:
        gap_values = [_sample_closed_integer(minimum, maximum, rng) for minimum, maximum in gap_ranges]
    else:
        requested_minimum, requested_maximum = motif_length_bounds
        feasible_minimum = fixed_length + sum(minimum for minimum, _ in gap_ranges)
        feasible_maximum = fixed_length + sum(maximum for _, maximum in gap_ranges)
        target_minimum = max(requested_minimum, feasible_minimum)
        target_maximum = min(requested_maximum, feasible_maximum)
        if target_minimum > target_maximum:
            raise DesignSampleError(
                f"contig for {chain_id!r} cannot satisfy motif length interval "
                f"[{requested_minimum}, {requested_maximum}] (feasible "
                f"[{feasible_minimum}, {feasible_maximum}])"
            )
        remaining = _sample_closed_integer(target_minimum, target_maximum, rng) - fixed_length
        gap_values = []
        for index, (minimum, maximum) in enumerate(gap_ranges):
            suffix = gap_ranges[index + 1 :]
            suffix_minimum = sum(value[0] for value in suffix)
            suffix_maximum = sum(value[1] for value in suffix)
            allowed_minimum = max(minimum, remaining - suffix_maximum)
            allowed_maximum = min(maximum, remaining - suffix_minimum)
            if allowed_minimum > allowed_maximum:
                raise AssertionError("contig length sampler lost a feasible gap assignment")
            chosen = _sample_closed_integer(allowed_minimum, allowed_maximum, rng)
            gap_values.append(chosen)
            remaining -= chosen
        if remaining != 0:
            raise AssertionError("contig length sampler did not consume the target length")

    layout: list[int | None] = []
    gap_index = 0
    for segment in segments:
        if isinstance(segment, tuple):
            layout.extend([None] * gap_values[gap_index])
            gap_index += 1
        else:
            layout.extend(segment)
    if not layout:
        raise DesignSampleError(f"contig for {chain_id!r} resolved to zero residues")
    return layout


def _final_chain_selector(chain_id: str) -> str:
    """Format one final-chain selector with the public contig quoting rules."""

    if re.fullmatch(r"[A-Za-z]", chain_id):
        return chain_id
    if "]" in chain_id or "/" in chain_id:
        raise DesignSampleError(
            f"design chain {chain_id!r} cannot be represented in motif output"
        )
    return f"[{chain_id}]"


def _resolved_motif_output_fields(
    atoms: struc.AtomArray,
    *,
    chain_id: str,
    partial_atoms_by_position: Mapping[int, Sequence[str]],
) -> MotifOutputFields | None:
    """Serialize the sampled final motif mask, never the unresolved task string."""

    motif_rows = np.flatnonzero(np.asarray(atoms.motif_coord_condition, dtype=bool))
    motif_positions = {
        int(atoms.res_id[row])
        for row in motif_rows
        if str(atoms.chain_id[row]) == chain_id
    }
    if not motif_positions:
        return None

    chain_rows = np.flatnonzero(np.asarray(atoms.chain_id).astype(str) == chain_id)
    if not len(chain_rows):
        raise AssertionError("resolved motif chain has no atom inventory")
    length = int(np.asarray(atoms.res_id[chain_rows], dtype=np.int64).max())
    prefix = _final_chain_selector(chain_id)
    parts: list[str] = []
    position = 1
    while position <= length:
        is_motif = position in motif_positions
        end = position
        while end < length and ((end + 1) in motif_positions) == is_motif:
            end += 1
        if is_motif:
            part = f"{prefix}{position}"
            if end != position:
                part += f"-{end}"
            parts.append(part)
        else:
            parts.append(str(end - position + 1))
        position = end + 1

    tip_atoms: list[str] = []
    for position, atom_names in sorted(partial_atoms_by_position.items()):
        if position not in motif_positions:
            raise AssertionError(
                "partial motif atom selection has no visible coordinate motif token"
            )
        tip_atoms.append(
            f"{prefix}{position}:{','.join(str(name) for name in atom_names)}"
        )
    return MotifOutputFields("/".join(parts), ";".join(tip_atoms))


def _public_sequence(value: str, modality: str, where: str) -> list[str]:
    if not isinstance(value, str) or not value:
        raise DesignSampleError(f"{where} must be a non-empty sequence string")
    letters = list(value.upper())
    bad = sorted(set(letters).difference(_PUBLIC_ALPHABET[modality]))
    if bad:
        raise DesignSampleError(f"{where} contains unsupported symbols: {bad}")
    if modality == "protein":
        return [_AA1_TO_3[letter] for letter in letters]
    if modality == "rna":
        return letters
    return [f"D{letter}" for letter in letters]


def _public_residue(value: Any, modality: str, where: str) -> str:
    names = _public_sequence(value, modality, where)
    if len(names) != 1:
        raise DesignSampleError(f"{where} must contain exactly one residue symbol")
    return names[0]


def _source_residue_name(atoms: struc.AtomArray, residue_uid: int) -> str:
    rows = np.where(atoms.residue_uid == residue_uid)[0]
    if not len(rows):
        raise AssertionError(residue_uid)
    return str(atoms.res_name[rows[0]])


def _element_for_polymer_atom(atom_name: str) -> str:
    name = atom_name.strip().upper()
    if name.startswith("OP") or name.startswith("O"):
        return "O"
    if name.startswith("N"):
        return "N"
    if name.startswith("S"):
        return "S"
    if name.startswith("P"):
        return "P"
    return "C"


def _materialize_polymer_rows(
    chain: DesignChainSpec,
    residue_names: Sequence[str],
    *,
    model_residue_names: Sequence[str] | None = None,
    source: struc.AtomArray | None = None,
    layout: Sequence[int | None] | None = None,
    carrier_components_by_position: Mapping[int, CCDComponent] | None = None,
    minimum_proxy_positions: Collection[int] = (),
) -> struc.AtomArray:
    inventory = _INVENTORIES[chain.modality]
    union_inventory = {
        "protein": tuple(PROTEIN_ATOM37),
        "rna": tuple(RNA_ATOM27),
        "dna": tuple(DNA_ATOM27),
    }[chain.modality]
    if model_residue_names is None:
        model_residue_names = residue_names
    if len(model_residue_names) != len(residue_names):
        raise AssertionError("polymer output/model residue layouts disagree")
    if layout is None:
        layout = [None] * len(residue_names)
    if len(layout) != len(residue_names):
        raise AssertionError("polymer source/output residue layouts disagree")
    carrier_components_by_position = (
        {} if carrier_components_by_position is None else carrier_components_by_position
    )
    minimum_proxy_positions = {int(value) for value in minimum_proxy_positions}

    parts: list[struc.AtomArray] = []
    for position, (residue_name, model_residue_name, source_uid) in enumerate(
        zip(residue_names, model_residue_names, layout, strict=True),
        start=1,
    ):
        if (
            source is not None
            and source_uid is not None
            and residue_name != model_residue_name
        ):
            source_rows = np.where(source.residue_uid == int(source_uid))[0]
            if not len(source_rows):
                raise AssertionError(f"copied source residue UID {source_uid} disappeared")
            part = source[source_rows]
            part.coord = np.asarray(part.coord, dtype=np.float32)
            part.chain_id[:] = chain.chain_id
            part.label_asym_id[:] = chain.chain_id
            part.label_entity_id[:] = chain.chain_id
            part.res_id[:] = position
            part.label_seq_id[:] = str(position)
            parts.append(part)
            continue

        carrier_component = carrier_components_by_position.get(position)
        if carrier_component is not None:
            retained_atoms = tuple(
                atom
                for atom in carrier_component.atoms
                if str(atom.element or "").upper() not in {"H", "D", "T"}
                and not bool(atom.leaving)
            )
            if not retained_atoms:
                raise DesignSampleError(
                    f"CCD component {carrier_component.component_id!r} has no "
                    "retained heavy atoms"
                )
            part = create_atom_array(len(retained_atoms))
            part.chain_id[:] = chain.chain_id
            part.label_asym_id[:] = chain.chain_id
            part.label_entity_id[:] = chain.chain_id
            part.res_id[:] = position
            part.label_seq_id[:] = str(position)
            part.res_name[:] = residue_name
            part.atom_name = np.asarray(
                [atom.atom_id for atom in retained_atoms], dtype="<U32"
            )
            part.element = np.asarray(
                [str(atom.element).upper() for atom in retained_atoms], dtype="<U3"
            )
            part.formal_charge = np.asarray(
                [atom.formal_charge for atom in retained_atoms], dtype=object
            )
            if carrier_component.has_full_finite_ideal_coordinates(
                atom.atom_id for atom in retained_atoms
            ):
                # Preserve the established authoring behavior when the CCD
                # supplies a complete ideal conformer. A de-novo carrier
                # without one must remain entirely coordinate-invalid: mixing
                # a partial ideal conformer with generated coordinates would
                # silently turn some atoms into hard geometric conditions.
                part.coord = np.asarray(
                    [atom.ideal_coord for atom in retained_atoms], dtype=np.float32
                )
                part.coord_valid[:] = True
            part.mol_type[:] = chain.modality
            part.hetero[:] = False
            parts.append(part)
            continue

        if model_residue_name == _UNKNOWN_RESIDUE[chain.modality]:
            # Identity-hidden positions must be assembly-closed: every decoded
            # standard identity can select its atoms without allocating new rows
            # or changing stable AtomUIDs after model execution.
            atom_names = (
                inventory[model_residue_name]
                if position in minimum_proxy_positions
                else union_inventory
            )
        else:
            try:
                atom_names = inventory[model_residue_name]
            except KeyError as exc:
                raise DesignSampleError(
                    f"no model inventory for {chain.modality} residue {model_residue_name!r}"
                ) from exc
        if (
            chain.modality == "protein"
            and position == len(residue_names)
            and "OXT" not in atom_names
        ):
            # OXT is a terminal-state atom, not part of every amino-acid
            # inventory.  Materialize it before model execution so fixed
            # sequences and contigs have the same stable-UID terminal row as
            # identity-hidden protein chains.
            atom_names = (*atom_names, "OXT")
        part = create_atom_array(len(atom_names))
        part.chain_id[:] = chain.chain_id
        part.label_asym_id[:] = chain.chain_id
        part.label_entity_id[:] = chain.chain_id
        part.res_id[:] = position
        part.label_seq_id[:] = str(position)
        part.res_name[:] = residue_name
        part.atom_name = np.asarray(atom_names, dtype="<U32")
        part.element = np.asarray(
            [_element_for_polymer_atom(atom_name) for atom_name in atom_names],
            dtype="<U3",
        )
        part.mol_type[:] = chain.modality
        part.hetero[:] = False
        parts.append(part)
    return concatenate_atom_arrays(parts)


def _source_chain(source: struc.AtomArray | None, chain: DesignChainSpec) -> np.ndarray:
    if source is None:
        return np.empty(0, dtype=np.int64)
    indices = np.where(source.chain_id == chain.chain_id)[0]
    if not len(indices):
        return indices
    modalities = set(str(value) for value in source.mol_type[indices])
    if modalities != {chain.modality}:
        raise DesignSampleError(
            f"input design chain {chain.chain_id!r} modality {sorted(modalities)} "
            f"does not equal declared {chain.modality!r}"
        )
    if chain.modality == "ligand" and len(np.unique(source.component_uid[indices])) != 1:
        raise DesignSampleError(
            f"design ligand chain {chain.chain_id!r} must resolve to one component"
        )
    return indices


def _resolve_final_position(position: Any, length: int, where: str) -> int:
    if isinstance(position, bool) or not isinstance(position, int) or position == 0:
        raise DesignSampleError(f"{where} resid must be a nonzero integer")
    resolved = position if position > 0 else length + position + 1
    if not 1 <= resolved <= length:
        raise DesignSampleError(f"{where} resid {position} is outside final length {length}")
    return resolved


def _sequence_bracket_component(value: str, *, where: str) -> str:
    component_id = value.strip().upper()
    if (
        not component_id
        or value.strip() != value
        or any(character.isspace() for character in value)
        or any(character in component_id for character in "[]")
        or len(component_id) > 16
    ):
        raise DesignSampleError(
            f"{where} bracketed residue must be a bare CCD component ID"
        )
    return component_id


def _parse_sequence_string(
    value: str, modality: str, where: str
) -> list[_SequenceResidue]:
    if not isinstance(value, str) or not value:
        raise DesignSampleError(f"{where} must be a non-empty sequence string")
    result: list[_SequenceResidue] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character == "]":
            raise DesignSampleError(f"{where} has an unmatched closing bracket")
        if character == "[":
            closing = value.find("]", index + 1)
            if closing < 0:
                raise DesignSampleError(f"{where} has an unterminated CCD bracket")
            raw_component = value[index + 1 : closing]
            component_id = _sequence_bracket_component(
                raw_component, where=f"{where}[{len(result)}]"
            )
            result.append(_SequenceResidue(component_id, component_id))
            index = closing + 1
            continue
        if character.isspace():
            raise DesignSampleError(f"{where} contains whitespace")
        result.append(
            _SequenceResidue(
                _public_sequence(character, modality, where)[0]
            )
        )
        index += 1
    if not result:
        raise DesignSampleError(f"{where} must contain at least one residue")
    return result


def _parse_sequence_residue(
    value: Any, modality: str, where: str
) -> _SequenceResidue:
    if not isinstance(value, str) or not value:
        raise DesignSampleError(f"{where} must contain exactly one residue symbol")
    if value.startswith("[") or value.endswith("]"):
        if not (value.startswith("[") and value.endswith("]")):
            raise DesignSampleError(f"{where} has malformed CCD brackets")
        component_id = _sequence_bracket_component(value[1:-1], where=where)
        return _SequenceResidue(component_id, component_id)
    parsed = _parse_sequence_string(value, modality, where)
    if len(parsed) != 1 or parsed[0].is_bracketed:
        raise DesignSampleError(f"{where} must contain exactly one residue symbol")
    return parsed[0]


def _sequence_condition(
    condition: Mapping[str, Any], chain: DesignChainSpec
) -> tuple[list[_SequenceResidue] | None, list[tuple[int, _SequenceResidue]]]:
    sequence_by_chain = condition.get("sequence", {})
    if sequence_by_chain is None:
        sequence_by_chain = {}
    sequence_by_chain = _native_mapping(sequence_by_chain, "condition.sequence")
    value = sequence_by_chain.get(chain.chain_id)
    if value is None:
        return None, []
    if isinstance(value, str):
        return _parse_sequence_string(
            value, chain.modality, f"condition.sequence.{chain.chain_id}"
        ), []
    if not isinstance(value, list) or not value:
        raise DesignSampleError(
            f"condition.sequence.{chain.chain_id} must be full sequence or non-empty sparse list"
        )
    sparse: list[tuple[int, str]] = []
    for index, item in enumerate(value):
        row = _native_mapping(item, f"condition.sequence.{chain.chain_id}[{index}]")
        if set(row) != {"resid", "residue"}:
            raise DesignSampleError("sparse sequence rows require exactly resid and residue")
        sparse.append(
            (
                row["resid"],
                _parse_sequence_residue(
                    row["residue"], chain.modality,
                    f"condition.sequence.{chain.chain_id}[{index}].residue",
                ),
            )
        )
    return None, sparse


def _resolve_bracket_residue(
    sequence_residue: _SequenceResidue,
    chain: DesignChainSpec,
    *,
    embedded: Mapping[str, CCDComponent],
    configured: Mapping[str, CCDComponent],
    packaged: Mapping[str, CCDComponent],
) -> tuple[CCDComponent | None, str]:
    """Resolve one bracket token to a CCD carrier or a model proxy.

    A known non-matching polymer modality is an explicit configuration error;
    only a missing/unusable component falls back to the modality-safe proxy.
    """

    if not sequence_residue.is_bracketed:
        raise AssertionError("standard sequence residue has no bracket component")
    component_id = str(sequence_residue.bracket_component)
    resolution = resolve_ccd_component(
        component_id,
        embedded=embedded,
        configured=configured,
        packaged=packaged,
        include_packaged=False,
    )
    component = resolution.component
    if component is not None:
        if component.polymer_modality is None:
            raise DesignSampleError(
                f"CCD component {component_id!r} does not declare a "
                "protein/RNA/DNA polymer modality"
            )
        if component.polymer_modality != chain.modality:
            raise DesignSampleError(
                f"CCD component {component_id!r} modality "
                f"{component.polymer_modality!r} conflicts with design chain "
                f"{chain.chain_id!r} modality {chain.modality!r}"
            )
    proxy_parent = _resolve_proxy_parent(
        component_id,
        chain.modality,
        embedded=embedded,
        configured=configured,
        packaged=packaged,
    )
    if resolution.status == "resolved" and component is not None:
        retained = tuple(
            atom
            for atom in component.atoms
            if str(atom.element or "").upper() not in {"H", "D", "T"}
            and not bool(atom.leaving)
        )
        if retained:
            return component, proxy_parent
    return None, proxy_parent


def _resolve_sequence_residue(
    residue: _SequenceResidue,
    chain: DesignChainSpec,
    *,
    embedded: Mapping[str, CCDComponent],
    configured: Mapping[str, CCDComponent],
    packaged: Mapping[str, CCDComponent],
) -> tuple[_SequenceResidue, CCDComponent | None, str]:
    if not residue.is_bracketed:
        return residue, None, _UNKNOWN_RESIDUE[chain.modality]
    component, proxy_parent = _resolve_bracket_residue(
        residue,
        chain,
        embedded=embedded,
        configured=configured,
        packaged=packaged,
    )
    return residue, component, proxy_parent


def _copy_residue_coordinates(
    target: struc.AtomArray,
    target_position: int,
    source: struc.AtomArray,
    source_residue_uid: int,
    *,
    all_atoms: bool,
) -> None:
    target_indices = np.where(target.res_id == target_position)[0]
    source_indices = np.where(source.residue_uid == source_residue_uid)[0]
    source_by_name = {str(source.atom_name[index]): int(index) for index in source_indices}
    for target_index in target_indices:
        source_index = source_by_name.get(str(target.atom_name[target_index]))
        if source_index is None:
            continue
        # Until _finalize_uids(), atom_uid is the sole source-origin identity.
        # Copy it even for coordinate-invalid rows so later source-edge recovery
        # never falls back to a mutable chain/residue/atom locator.
        target.atom_uid[target_index] = source.atom_uid[source_index]
        if (target.res_name[target_index] == source.res_name[source_index]
                and source.formal_charge[source_index] is not None):
            target.formal_charge[target_index] = source.formal_charge[source_index]
        if not bool(source.coord_valid[source_index]):
            continue
        if all_atoms or str(target.atom_name[target_index]) == _CENTER_ATOM[str(target.mol_type[target_index])]:
            target.coord[target_index] = source.coord[source_index]
            target.coord_valid[target_index] = True
            if all_atoms:
                target.motif_coord_condition[target_index] = True
            else:
                target.token_center_condition[target_index] = True


def _compile_polymer_chain(
    request: ExpandedRequest,
    chain: DesignChainSpec,
    source: struc.AtomArray | None,
    condition: Mapping[str, Any],
    rng: Any,
    source_parent_by_residue: Mapping[int, str],
    source_modified_residue_uids: set[int],
    embedded_ccd: Mapping[str, CCDComponent],
    configured_ccd: Mapping[str, CCDComponent],
    packaged_ccd: Mapping[str, CCDComponent],
) -> tuple[
    struc.AtomArray,
    list[int | None],
    MotifOutputFields | None,
    Mapping[int, CCDComponent],
    Mapping[int, str],
    Mapping[int, str],
]:
    source_indices = _source_chain(source, chain)
    existing = bool(len(source_indices))
    backbone_raw = condition.get("backbone", []) or []
    if not isinstance(backbone_raw, list) or not all(isinstance(item, str) for item in backbone_raw):
        raise DesignSampleError("condition.backbone must be a list of design chain IDs")
    backbone = chain.chain_id in backbone_raw
    contig_map = condition.get("contig", {}) or {}
    contig_map = _native_mapping(contig_map, "condition.contig")
    contig_value = contig_map.get(chain.chain_id)
    if backbone and contig_value is not None:
        raise DesignSampleError(f"backbone and contig are mutually exclusive for {chain.chain_id!r}")
    full_sequence, sparse = _sequence_condition(condition, chain)
    resolved_full_sequence = (
        None
        if full_sequence is None
        else [
            _resolve_sequence_residue(
                residue,
                chain,
                embedded=embedded_ccd,
                configured=configured_ccd,
                packaged=packaged_ccd,
            )
            for residue in full_sequence
        ]
    )
    resolved_sparse_tokens = [
        (
            position,
            *_resolve_sequence_residue(
                residue,
                chain,
                embedded=embedded_ccd,
                configured=configured_ccd,
                packaged=packaged_ccd,
            ),
        )
        for position, residue in sparse
    ]

    layout: list[int | None]
    if contig_value is not None:
        if source is None or not existing:
            raise DesignSampleError(f"contig requires existing input design chain {chain.chain_id!r}")
        contig_entry = _native_mapping(contig_value, f"condition.contig.{chain.chain_id}")
        if set(contig_entry).difference(
            {"token", "polymer_atom", "motif_min_length", "motif_max_length"}
        ):
            raise DesignSampleError(f"unknown contig keys for {chain.chain_id!r}")
        layout = _parse_contig(
            contig_entry.get("token"),
            chain_id=chain.chain_id,
            source=source,
            rng=rng,
            motif_length_bounds=_contig_length_bounds(
                contig_entry, chain_id=chain.chain_id
            ),
        )
    elif backbone:
        if source is None or not existing:
            raise DesignSampleError(f"backbone requires existing input design chain {chain.chain_id!r}")
        layout = list(_chain_residue_uids(source, chain.chain_id))
    elif full_sequence is not None:
        layout = [None] * len(full_sequence)
    elif chain.chain_id in request.plan.lengths:
        layout = [None] * int(request.plan.lengths[chain.chain_id])
    else:
        raise DesignSampleError(
            f"polymer chain {chain.chain_id!r} has no resolved length fact"
        )

    if chain.chain_id in request.plan.lengths and int(request.plan.lengths[chain.chain_id]) != len(layout):
        raise DesignSampleError(f"independent length facts disagree for {chain.chain_id!r}")
    if full_sequence is not None and len(full_sequence) != len(layout):
        raise DesignSampleError(f"full sequence and structural length disagree for {chain.chain_id!r}")

    model_residue_names: list[str] = []
    residue_names: list[str] = []
    for uid in layout:
        copied = uid is not None and source is not None
        modified = copied and int(uid) in source_modified_residue_uids
        if copied and (contig_value is not None or (backbone and modified)):
            raw_name = _source_residue_name(source, int(uid))
            residue_names.append(raw_name)
            model_residue_names.append(
                source_parent_by_residue.get(int(uid), raw_name)
            )
        else:
            unknown = _UNKNOWN_RESIDUE[chain.modality]
            residue_names.append(unknown)
            model_residue_names.append(unknown)

    carrier_components_by_position: dict[int, CCDComponent] = {}
    carrier_parents_by_position: dict[int, str] = {}
    proxy_parents_by_position: dict[int, str] = {}

    def apply_sequence_residue(
        position: int,
        residue: _SequenceResidue,
        component: CCDComponent | None,
        proxy_parent: str,
    ) -> None:
        copied_uid = layout[position - 1]
        copied_modified = (
            copied_uid is not None
            and int(copied_uid) in source_modified_residue_uids
        )
        if copied_uid is not None and residue.is_bracketed:
            if source is None:
                raise AssertionError("copied sequence residue has no source")
            source_name = _source_residue_name(source, int(copied_uid)).strip().upper()
            if not copied_modified:
                raise DesignSampleError(
                    "bracket CCD identity cannot replace a copied standard residue "
                    f"at {chain.chain_id}:{position}"
                )
            if source_name != residue.name:
                raise DesignSampleError(
                    f"bracket CCD identity conflicts with copied source residue at "
                    f"{chain.chain_id}:{position} ({source_name!r} versus {residue.name!r})"
                )
            # A source-backed modified carrier is authoritative.  Do not replace
            # its source atom inventory with a synthetic ideal-coordinate one.
            return

        copied_identity_locked = (
            copied_uid is not None
            and (contig_value is not None or copied_modified)
        )
        if copied_identity_locked and model_residue_names[position - 1] != residue.name:
            raise DesignSampleError(
                f"sequence conflicts with copied identity at {chain.chain_id}:{position}"
            )
        if copied_uid is None:
            residue_names[position - 1] = residue.name
            if component is not None:
                model_residue_names[position - 1] = _UNKNOWN_RESIDUE[chain.modality]
                carrier_components_by_position[position] = component
                carrier_parents_by_position[position] = proxy_parent
            else:
                if residue.is_bracketed:
                    model_residue_names[position - 1] = proxy_parent
                    proxy_parents_by_position[position] = proxy_parent
                else:
                    model_residue_names[position - 1] = residue.name
        elif not copied_identity_locked:
            model_residue_names[position - 1] = residue.name

    if resolved_full_sequence is not None:
        for index, (residue, component, proxy_parent) in enumerate(
            resolved_full_sequence, start=1
        ):
            apply_sequence_residue(index, residue, component, proxy_parent)

    resolved_sparse: dict[int, _SequenceResidue] = {}
    for raw_position, residue, component, proxy_parent in resolved_sparse_tokens:
        position = _resolve_final_position(
            raw_position, len(layout), f"sequence.{chain.chain_id}"
        )
        current = resolved_sparse.get(position)
        if current is not None and current != residue:
            raise DesignSampleError(
                f"contradictory sparse sequence at {chain.chain_id}:{position}"
            )
        resolved_sparse[position] = residue
        apply_sequence_residue(position, residue, component, proxy_parent)

    if chain.modality == "protein" and contig_value is not None and source is not None:
        for position, residue_uid in enumerate(layout, start=1):
            if residue_uid is None or position == len(layout):
                continue
            source_rows = np.where(source.residue_uid == int(residue_uid))[0]
            hard_oxt = source_rows[
                (source.atom_name[source_rows] == "OXT")
                & np.asarray(source.coord_valid[source_rows], dtype=bool)
            ]
            if len(hard_oxt):
                raise DesignSampleError(
                    "contig terminal-state conflict: hard source OXT maps to "
                    f"internal final position {chain.chain_id}:{position}"
                )

    atoms = _materialize_polymer_rows(
        chain,
        residue_names,
        model_residue_names=model_residue_names,
        source=source,
        layout=layout,
        carrier_components_by_position=carrier_components_by_position,
        minimum_proxy_positions=proxy_parents_by_position,
    )
    # A source UNK motif contributes backbone coordinates as model conditions;
    # its amino-acid identity remains a decoder prediction.
    unknown_residue = _UNKNOWN_RESIDUE[chain.modality]
    hard_identity_positions = (
        {
            index + 1
            for index, uid in enumerate(layout)
            if uid is not None and residue_names[index] != unknown_residue
        }
        if contig_value is not None
        else set()
    )
    if backbone:
        hard_identity_positions.update(
            index + 1
            for index, uid in enumerate(layout)
            if uid is not None and int(uid) in source_modified_residue_uids
        )
    if full_sequence is not None:
        hard_identity_positions.update(range(1, len(layout) + 1))
    hard_identity_positions.update(resolved_sparse)
    atoms.identity_condition[np.isin(atoms.res_id, list(hard_identity_positions))] = True

    partial_atoms_by_position: dict[int, tuple[str, ...]] = {}
    if contig_value is not None and source is not None:
        for position, residue_uid in enumerate(layout, start=1):
            if residue_uid is not None:
                _copy_residue_coordinates(atoms, position, source, residue_uid, all_atoms=True)
        raw_polymer_atom = _native_mapping(
            contig_value, f"condition.contig.{chain.chain_id}"
        ).get("polymer_atom")
        polymer_atom = (
            {}
            if raw_polymer_atom is None
            else _native_mapping(
                raw_polymer_atom,
                f"condition.contig.{chain.chain_id}.polymer_atom",
            )
        )
        copied_positions = {uid: index + 1 for index, uid in enumerate(layout) if uid is not None}
        for selector, atom_selection in polymer_atom.items():
            selected_uids = _source_selector(source, selector)
            if len(selected_uids) != 1 or selected_uids[0] not in copied_positions:
                raise DesignSampleError(f"polymer_atom selector {selector!r} is not one copied residue")
            position = copied_positions[selected_uids[0]]
            indices = np.where(atoms.res_id == position)[0]
            atoms.motif_coord_condition[indices] = False
            residue_name = model_residue_names[position - 1]
            if atom_selection == "tip":
                if chain.modality != "protein" or residue_name not in SIDECHAIN_TIP_ATOMS:
                    raise DesignSampleError(f"tip preset unsupported for {chain.modality}:{residue_name}")
                names = list(SIDECHAIN_TIP_ATOMS[residue_name])
            elif isinstance(atom_selection, list) and atom_selection and all(isinstance(item, str) for item in atom_selection):
                if len(atom_selection) != len(set(atom_selection)):
                    raise DesignSampleError(f"duplicate polymer_atom name for {selector!r}")
                names = atom_selection
            else:
                raise DesignSampleError(f"polymer_atom {selector!r} must be non-empty list or tip")
            for name in names:
                matches = indices[atoms.atom_name[indices] == name]
                if len(matches) != 1 or not bool(atoms.coord_valid[matches[0]]):
                    raise DesignSampleError(f"polymer_atom {selector!r}:{name} is missing or unresolved")
                atoms.motif_coord_condition[matches[0]] = True
            partial_atoms_by_position[position] = tuple(names)
    elif backbone and source is not None:
        for position, residue_uid in enumerate(layout, start=1):
            assert residue_uid is not None
            _copy_residue_coordinates(atoms, position, source, residue_uid, all_atoms=False)

    motif_output = _resolved_motif_output_fields(
        atoms,
        chain_id=chain.chain_id,
        partial_atoms_by_position=partial_atoms_by_position,
    )
    return (
        atoms,
        layout,
        motif_output,
        carrier_components_by_position,
        carrier_parents_by_position,
        proxy_parents_by_position,
    )


def _new_ligand_rows(chain_id: str, count: int) -> struc.AtomArray:
    if count <= 0:
        raise DesignSampleError(f"ligand {chain_id!r} atom count must be positive")
    atoms = create_atom_array(count)
    atoms.chain_id[:] = chain_id
    atoms.label_asym_id[:] = chain_id
    atoms.label_entity_id[:] = chain_id
    atoms.res_id[:] = 1
    atoms.label_seq_id[:] = "."
    atoms.res_name[:] = "UNL"
    atoms.atom_name = np.asarray([f"X{index+1}" for index in range(count)], dtype="<U32")
    atoms.mol_type[:] = "ligand"
    atoms.hetero[:] = True
    return atoms


def _rdkit_ligand(chain_id: str, smiles: str) -> struc.AtomArray:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise DesignSampleError("SMILES mode requires RDKit") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise DesignSampleError("condition.ligand SMILES is invalid or empty")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise DesignSampleError(
            "condition.ligand SMILES must contain exactly one connected component"
        )
    try:
        molecule = Chem.RemoveHs(molecule, sanitize=True)
    except Exception as exc:
        raise DesignSampleError("condition.ligand SMILES hydrogen normalization failed") from exc
    if molecule.GetNumAtoms() == 0 or any(
        atom.GetAtomicNum() <= 1 for atom in molecule.GetAtoms()
    ):
        raise DesignSampleError(
            "condition.ligand SMILES must contain at least one heavy atom"
        )
    atoms = _new_ligand_rows(chain_id, molecule.GetNumAtoms())
    atoms.atom_name = np.asarray([f"A{index+1}" for index in range(len(atoms))], dtype="<U32")
    atoms.element = np.asarray([atom.GetSymbol().upper() for atom in molecule.GetAtoms()], dtype="<U3")
    atoms.formal_charge = np.asarray([atom.GetFormalCharge() for atom in molecule.GetAtoms()], dtype=object)
    atoms.identity_condition[:] = True
    bond_rows: list[tuple[int, int, int]] = []
    rdkit_order = {
        str(Chem.BondType.SINGLE): int(struc.BondType.SINGLE),
        str(Chem.BondType.DOUBLE): int(struc.BondType.DOUBLE),
        str(Chem.BondType.TRIPLE): int(struc.BondType.TRIPLE),
        str(Chem.BondType.AROMATIC): int(struc.BondType.AROMATIC),
    }
    for bond in molecule.GetBonds():
        order = rdkit_order.get(str(bond.GetBondType()))
        if order is None:
            raise DesignSampleError(f"unsupported SMILES bond type: {bond.GetBondType()}")
        bond_rows.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), order))
    if bond_rows:
        atoms.bonds = struc.BondList(len(atoms), np.asarray(bond_rows, dtype=np.uint32))
    return atoms


def _subset_source_component(source: struc.AtomArray, indices: np.ndarray) -> struc.AtomArray:
    result = source[indices]
    result.coord = np.asarray(result.coord, dtype=np.float32)
    for uid_name in ("entity_uid", "asym_uid", "component_uid", "molecule_uid", "residue_uid"):
        result.get_annotation(uid_name).fill(-1)
    result.token_uid.fill(-1)
    return result


def _ccd_ligand(
    chain_id: str, component_id: str,
    trusted_components: Mapping[str, TrustedCCDComponent] | None,
) -> struc.AtomArray:
    """Materialize named CCD chemistry, never CCD geometry, as design atoms."""
    from rdkit import Chem

    component = (trusted_components or {}).get(component_id)
    if component is None:
        raise DesignSampleError(f"CCD ligand {component_id!r} requires preflighted trusted chemistry")
    if component.component_id != component_id:
        raise DesignSampleError("CCD ligand component identity mismatch")
    molecule = Chem.RemoveHs(Chem.Mol(component.rdkit_binary), sanitize=True)
    if molecule.GetNumAtoms() == 0 or len(Chem.GetMolFrags(molecule)) != 1:
        raise DesignSampleError("CCD ligand must be one non-empty connected component")
    names = [atom.GetProp("name").strip() if atom.HasProp("name") else "" for atom in molecule.GetAtoms()]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise DesignSampleError("CCD ligand requires unique non-empty atom names")
    if any(atom.GetAtomicNum() <= 1 for atom in molecule.GetAtoms()):
        raise DesignSampleError("CCD ligand requires concrete heavy-atom elements")
    atoms = _new_ligand_rows(chain_id, molecule.GetNumAtoms())
    atoms.res_name[:] = component_id
    atoms.atom_name = np.asarray(names, dtype="<U32")
    atoms.element = np.asarray([atom.GetSymbol().upper() for atom in molecule.GetAtoms()], dtype="<U3")
    atoms.formal_charge = np.asarray([atom.GetFormalCharge() for atom in molecule.GetAtoms()], dtype=object)
    atoms.identity_condition[:] = True
    orders = {Chem.BondType.SINGLE: struc.BondType.SINGLE,
              Chem.BondType.DOUBLE: struc.BondType.DOUBLE,
              Chem.BondType.TRIPLE: struc.BondType.TRIPLE,
              Chem.BondType.AROMATIC: struc.BondType.AROMATIC}
    bonds = []
    for bond in molecule.GetBonds():
        if bond.GetBondType() not in orders:
            raise DesignSampleError("CCD ligand has an unsupported bond order")
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), int(orders[bond.GetBondType()])))
    atoms.bonds = struc.BondList(len(atoms), np.asarray(bonds, dtype=np.uint32).reshape(-1, 3))
    return atoms


def _compile_ligand_chain(
    request: ExpandedRequest,
    chain: DesignChainSpec,
    source: struc.AtomArray | None,
    condition: Mapping[str, Any],
    trusted_ccd_components: Mapping[str, TrustedCCDComponent] | None = None,
) -> tuple[struc.AtomArray, str]:
    source_indices = _source_chain(source, chain)
    existing = bool(len(source_indices))
    backbone = chain.chain_id in (condition.get("backbone", []) or [])
    ligand_map = _native_mapping(condition.get("ligand", {}) or {}, "condition.ligand")
    ligand_entry = ligand_map.get(chain.chain_id)
    if backbone and ligand_entry is not None:
        raise DesignSampleError(f"ligand backbone and ligand mode conflict for {chain.chain_id!r}")
    mode = "backbone" if backbone else "free"
    entry: Mapping[str, Any] = {}
    if ligand_entry is not None:
        entry = _native_mapping(ligand_entry, f"condition.ligand.{chain.chain_id}")
        mode = entry.get("mode")
        if mode not in {"smiles", "ccd", "type_bond", "atom_motif"}:
            raise DesignSampleError(f"unsupported ligand mode for {chain.chain_id!r}: {mode!r}")

    plan_length = request.plan.lengths.get(chain.chain_id)
    if mode == "ccd":
        component_id = entry.get("ccd")
        if (set(entry) != {"mode", "ccd"} or not isinstance(component_id, str)
                or not component_id or not component_id.isascii()
                or not component_id.isalnum() or component_id != component_id.upper()):
            raise DesignSampleError("CCD ligand requires exactly mode and an uppercase alphanumeric ccd ID")
        if existing:
            raise DesignSampleError("CCD ligand chain already exists in input; use type_bond for source components")
        if plan_length is not None:
            raise DesignSampleError("CCD ligand fixes atom count and rejects design_length")
        # Reuse the complete named hard-graph lane after authoring construction.
        return _ccd_ligand(chain.chain_id, component_id, trusted_ccd_components), "type_bond"
    if mode == "smiles":
        if set(entry) != {"mode", "smiles"} or not isinstance(entry.get("smiles"), str):
            raise DesignSampleError("SMILES ligand requires exactly mode and smiles")
        atoms = _rdkit_ligand(chain.chain_id, entry["smiles"])
        if plan_length is not None:
            raise DesignSampleError("SMILES atom count cannot be combined with design_length")
        return atoms, mode
    if mode in {"backbone", "type_bond", "atom_motif"} and (source is None or not existing):
        raise DesignSampleError(f"ligand mode {mode!r} requires an existing input component")
    if mode == "backbone":
        if plan_length is not None:
            raise DesignSampleError("existing ligand backbone fixes atom count and rejects design_length")
        atoms = _new_ligand_rows(chain.chain_id, len(source_indices))
        atoms.coord = np.asarray(source.coord[source_indices], dtype=np.float32).copy()
        atoms.coord_valid[:] = source.coord_valid[source_indices]
        atoms.token_center_condition[:] = atoms.coord_valid
        return atoms, mode
    if mode == "type_bond":
        if set(entry) != {"mode"}:
            raise DesignSampleError("type_bond accepts no partial element/atom options")
        if plan_length is not None:
            raise DesignSampleError("type_bond fixes atom count and rejects design_length")
        atoms = _subset_source_component(source, source_indices)
        atoms.coord[:] = 0
        atoms.coord_valid[:] = False
        atoms.motif_coord_condition[:] = False
        atoms.token_center_condition[:] = False
        atoms.identity_condition[:] = True
        return atoms, mode
    if mode == "atom_motif":
        if set(entry) != {"mode", "atoms"}:
            raise DesignSampleError("atom_motif requires exactly mode and atoms")
        atom_names = entry.get("atoms")
        if not isinstance(atom_names, list) or not atom_names or not all(isinstance(name, str) for name in atom_names):
            raise DesignSampleError("atom_motif atoms must be a non-empty atom-name list")
        if len(atom_names) != len(set(atom_names)):
            raise DesignSampleError("atom_motif atom names must be unique")
        if plan_length is None:
            raise DesignSampleError("atom_motif requires explicit design_length")
        selected: list[int] = []
        for name in atom_names:
            matches = source_indices[source.atom_name[source_indices] == name]
            if len(matches) != 1:
                raise DesignSampleError(f"atom_motif name {name!r} matched {len(matches)} input atoms")
            selected.append(int(matches[0]))
        if int(plan_length) < len(selected):
            raise DesignSampleError("atom_motif design_length is smaller than selected motif")
        motif = _subset_source_component(source, np.asarray(selected, dtype=np.int64))
        motif.identity_condition[:] = True
        motif.motif_coord_condition[:] = motif.coord_valid
        if int(plan_length) == len(selected):
            return motif, mode
        remainder = _new_ligand_rows(chain.chain_id, int(plan_length) - len(selected))
        existing_names = set(str(name) for name in motif.atom_name)
        for index in range(len(remainder)):
            candidate = f"X{index+1}"
            while candidate in existing_names:
                candidate = f"X_{candidate}"
            remainder.atom_name[index] = candidate
            existing_names.add(candidate)
        return concatenate_atom_arrays((motif, remainder)), mode

    if plan_length is None:
        raise DesignSampleError(f"free ligand {chain.chain_id!r} requires design_length")
    return _new_ligand_rows(chain.chain_id, int(plan_length)), mode


def _context_array(source: struc.AtomArray, design_chain_ids: set[str]) -> struc.AtomArray:
    context = source[~np.isin(source.chain_id, list(design_chain_ids))]
    context.coord = np.asarray(context.coord, dtype=np.float32)
    context.is_design[:] = False
    context.identity_condition[:] = True
    context.motif_coord_condition[:] = context.coord_valid
    context.token_center_condition[:] = False
    return context


def _pre_uid_residue_key(atoms: struc.AtomArray, row: int) -> tuple[object, ...]:
    return (
        bool(atoms.is_design[row]),
        str(atoms.chain_id[row]),
        int(atoms.res_id[row]),
        str(atoms.ins_code[row]),
        -1 if bool(atoms.is_design[row]) else int(atoms.asym_uid[row]),
    )


def _select_carrier_occurrences(
    atoms: struc.AtomArray,
    candidates: Mapping[int, _CarrierCandidate],
) -> tuple[_CarrierOccurrence, ...]:
    candidate_by_origin = {
        int(origin_uid): candidate
        for candidate in candidates.values()
        for origin_uid in candidate.canonical_by_origin_uid
    }
    grouped: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for row in range(len(atoms)):
        if str(atoms.mol_type[row]) != "ligand":
            grouped[_pre_uid_residue_key(atoms, row)].append(row)

    result: list[_CarrierOccurrence] = []
    for rows in grouped.values():
        origins = [int(atoms.atom_uid[row]) for row in rows]
        matching = {
            candidate_by_origin[origin].source_residue_uid
            for origin in origins
            if origin in candidate_by_origin
        }
        if not matching:
            continue
        if len(matching) != 1:
            raise DesignInputError("one final residue mixes multiple source CCD occurrences")
        source_residue_uid = next(iter(matching))
        candidate = candidates[source_residue_uid]
        if (
            any(origin not in candidate.canonical_by_origin_uid for origin in origins)
            or (
                candidate.source_residue_uid >= 0
                and set(origins) != set(candidate.canonical_by_origin_uid)
            )
            or len(origins) != len(set(origins))
        ):
            raise DesignInputError(
                f"source CCD occurrence {candidate.component.component_id!r} "
                "was not copied one-to-one"
            )
        for row, origin in zip(rows, origins, strict=True):
            atoms.atom_name[row] = candidate.canonical_by_origin_uid[origin]
        result.append(
            _CarrierOccurrence(
                source_residue_uid=source_residue_uid,
                component=candidate.component,
                rows=tuple(rows),
            )
        )
    return tuple(result)


def _automatic_carrier_leaving_rows(
    atoms: struc.AtomArray,
    source: struc.AtomArray | None,
    source_bonds: Sequence[tuple[int, int, int]],
    occurrences: Sequence[_CarrierOccurrence],
    condition: Mapping[str, Any],
) -> set[int]:
    if not occurrences:
        return set()
    row_to_occurrence = {
        row: occurrence_index
        for occurrence_index, occurrence in enumerate(occurrences)
        for row in occurrence.rows
    }
    source_origins = (
        set()
        if source is None
        else {int(value) for value in np.asarray(source.atom_uid, dtype=np.int64)}
    )
    final_rows_by_origin: dict[int, list[int]] = defaultdict(list)
    for row, raw_origin in enumerate(np.asarray(atoms.atom_uid, dtype=np.int64)):
        origin = int(raw_origin)
        if origin in source_origins:
            final_rows_by_origin[origin].append(row)

    linkages_by_occurrence: dict[int, set[tuple[int, int, str]]] = defaultdict(set)
    if source is not None:
        for raw_i, raw_j, raw_order in source_bonds:
            source_i, source_j = int(raw_i), int(raw_j)
            if int(source.residue_uid[source_i]) == int(source.residue_uid[source_j]):
                continue
            origins = (
                int(source.atom_uid[source_i]),
                int(source.atom_uid[source_j]),
            )
            mapped = tuple(final_rows_by_origin.get(origin, []) for origin in origins)
            for side, endpoint_rows in enumerate(mapped):
                carrier_rows = [row for row in endpoint_rows if row in row_to_occurrence]
                if not carrier_rows:
                    continue
                if len(endpoint_rows) != 1:
                    raise DesignInputError(
                        "a repeated source occurrence makes an external carrier "
                        f"edge origin {origins!r} ambiguous"
                    )
                other_rows = mapped[1 - side]
                if len(other_rows) > 1:
                    raise DesignInputError(
                        "a repeated source occurrence makes an external carrier "
                        f"edge origin {origins!r} ambiguous"
                    )
                if not other_rows:
                    continue
                row = carrier_rows[0]
                linkages_by_occurrence[row_to_occurrence[row]].add(
                    (row, other_rows[0], _source_order_name(int(raw_order)))
                )

    declarations = condition.get("bond", []) or []
    if not isinstance(declarations, list):
        raise DesignSampleError("condition.bond must be a list")
    for index, declaration in enumerate(declarations):
        item = _native_mapping(declaration, f"condition.bond[{index}]")
        endpoints = (
            _resolve_atom_selector(
                atoms, item.get("atom1"), where=f"condition.bond[{index}].atom1"
            ),
            _resolve_atom_selector(
                atoms, item.get("atom2"), where=f"condition.bond[{index}].atom2"
            ),
        )
        for endpoint, other in (endpoints, endpoints[::-1]):
            occurrence_index = row_to_occurrence.get(endpoint)
            if occurrence_index is None or other in occurrences[occurrence_index].rows:
                continue
            linkages_by_occurrence[occurrence_index].add(
                (endpoint, other, str(item.get("order", "")).strip().lower())
            )

    result: set[int] = set()
    for occurrence_index, linkages in linkages_by_occurrence.items():
        occurrence = occurrences[occurrence_index]
        component = occurrence.component
        raw_type = " ".join((component.raw_chem_type or "").upper().split())
        if (
            component.polymer_modality != "protein"
            or "PEPTIDE-LIKE" in raw_type
            or not raw_type.endswith("PEPTIDE LINKING")
        ):
            # Nucleic linkages, caps, peptide-like components, staples, and
            # other non-canonical chemistry require an explicit omit_atoms
            # declaration.  A leaving flag alone never proves a reaction.
            continue
        atom_by_id = {atom.atom_id: atom for atom in component.atoms}
        adjacency: dict[str, set[str]] = defaultdict(set)
        for bond in component.bonds:
            adjacency[bond.atom_id_1].add(bond.atom_id_2)
            adjacency[bond.atom_id_2].add(bond.atom_id_1)
        present_by_name = {
            str(atoms.atom_name[row]): row for row in occurrence.rows
        }
        if len(present_by_name) != len(occurrence.rows):
            raise DesignInputError("carrier occurrence has ambiguous canonical atom names")
        for endpoint_row, other_row, linkage_order in linkages:
            if linkage_order != "single":
                continue
            endpoint_name = str(atoms.atom_name[endpoint_row])
            other_name = str(atoms.atom_name[other_row])
            same_asym = (
                str(atoms.chain_id[endpoint_row]) == str(atoms.chain_id[other_row])
                and (
                    bool(atoms.is_design[endpoint_row])
                    and bool(atoms.is_design[other_row])
                    or int(atoms.asym_uid[endpoint_row])
                    == int(atoms.asym_uid[other_row])
                )
            )
            endpoint_position = int(atoms.res_id[endpoint_row])
            other_position = int(atoms.res_id[other_row])
            standard_backbone_link = (
                same_asym
                and str(atoms.mol_type[other_row]) == "protein"
                and (
                    (
                        endpoint_name == "C"
                        and other_name == "N"
                        and other_position == endpoint_position + 1
                    )
                    or (
                        endpoint_name == "N"
                        and other_name == "C"
                        and other_position == endpoint_position - 1
                    )
                )
            )
            if not standard_backbone_link:
                continue
            endpoint_atom = atom_by_id.get(endpoint_name)
            if endpoint_atom is None:
                raise DesignInputError("carrier linkage endpoint is absent from CCD atoms")
            if endpoint_atom.leaving:
                raise DesignInputError(
                    f"carrier linkage endpoint {component.component_id}:{endpoint_name} "
                    "is itself a CCD leaving atom"
                )
            candidates = [
                atom.atom_id
                for atom in component.atoms
                if atom.leaving
                and atom.atom_id in present_by_name
                and adjacency.get(atom.atom_id, set()) == {endpoint_name}
            ]
            if len(candidates) > 1:
                raise DesignInputError(
                    f"carrier linkage endpoint {component.component_id}:{endpoint_name} "
                    f"has ambiguous leaving atoms {sorted(candidates)}"
                )
            if candidates:
                result.add(present_by_name[candidates[0]])
    return result


def _finalize_uids(
    atoms: struc.AtomArray, *, source: struc.AtomArray | None = None
) -> None:
    prior_asym_uid = np.asarray(atoms.asym_uid, dtype=np.int64).copy()
    entity_keys = [
        (str(chain), "design" if bool(design) else str(label_entity))
        for chain, design, label_entity in zip(atoms.chain_id, atoms.is_design, atoms.label_entity_id)
    ]
    asym_keys = [
        ("design", str(chain))
        if bool(design) or int(source_asym_uid) < 0
        else ("context", int(source_asym_uid))
        for chain, design, source_asym_uid in zip(
            atoms.chain_id, atoms.is_design, prior_asym_uid
        )
    ]
    residue_keys = list(zip(asym_keys, atoms.res_id.tolist(), atoms.ins_code.tolist()))
    component_keys = [
        (asym, "polymer") if mol != "ligand" else (asym, residue, "ligand")
        for asym, residue, mol in zip(asym_keys, residue_keys, atoms.mol_type)
    ]
    molecule_keys = [
        (asym, "polymer") if mol != "ligand" else (asym, "ligand")
        for asym, mol in zip(asym_keys, atoms.mol_type)
    ]
    assign_identity_uids(
        atoms,
        entity_keys=entity_keys,
        asym_keys=asym_keys,
        component_keys=component_keys,
        molecule_keys=molecule_keys,
        residue_keys=residue_keys,
    )
    if source is None:
        return

    # Atom storage is context-first, but the checkpoint's relative-chain feature
    # uses the sign of asymmetric chain IDs. Keep the input's relative order for
    # existing chains instead of moving every design chain after its context.
    # Remap only asym_uid: the other stable authoring identities must not change.
    source_order = {
        int(uid): index
        for index, uid in enumerate(dict.fromkeys(source.asym_uid.tolist()))
    }
    source_asym_by_chain: dict[str, int] = {}
    for chain in dict.fromkeys(source.chain_id.tolist()):
        values = np.unique(source.asym_uid[source.chain_id == chain])
        if len(values) == 1:
            source_asym_by_chain[str(chain)] = int(values[0])

    key_order: dict[tuple[str, object], int] = {}
    next_order = len(source_order)
    for key in dict.fromkeys(asym_keys):
        if key[0] == "context":
            if int(key[1]) not in source_order:
                raise DesignSampleError("compiled context asymmetry is absent from source")
            key_order[key] = source_order[int(key[1])]
        elif str(key[1]) in source_asym_by_chain:
            key_order[key] = source_order[source_asym_by_chain[str(key[1])]]
        else:
            # Genuinely new chains keep their declared order after source chains.
            key_order[key] = next_order
            next_order += 1
    if len(set(key_order.values())) != len(key_order):
        raise DesignSampleError("multiple compiled chains resolve to one source asymmetry")
    compact = {value: index for index, value in enumerate(sorted(key_order.values()))}
    atoms.asym_uid = np.asarray(
        [compact[key_order[key]] for key in asym_keys], dtype=np.int64
    )


def _validate_label_asym_bijection(atoms: struc.AtomArray) -> None:
    labels = np.asarray(atoms.label_asym_id).astype(str)
    asym_uids = np.asarray(atoms.asym_uid, dtype=np.int64)
    for label in dict.fromkeys(labels.tolist()):
        if not label:
            raise DesignSampleError("label_asym_id must be non-empty")
        matching_uids = np.unique(asym_uids[labels == label])
        if len(matching_uids) != 1:
            raise DesignSampleError(
                f"label_asym_id {label!r} names multiple internal asym segments"
            )
    for asym_uid in np.unique(asym_uids):
        matching_labels = np.unique(labels[asym_uids == asym_uid])
        if len(matching_labels) != 1:
            raise DesignSampleError(
                f"internal asym {int(asym_uid)} has multiple label_asym_id values"
            )


def _merge_current_bonds(
    atoms: struc.AtomArray,
    extra: Sequence[tuple[int, int, int]] = (),
) -> None:
    merged: dict[tuple[int, int], int] = {}
    if atoms.bonds is not None:
        for row_i, row_j, order in atoms.bonds.as_array():
            merged[tuple(sorted((int(row_i), int(row_j))))] = int(order)
    for row_i, row_j, order in extra:
        key = tuple(sorted((int(row_i), int(row_j))))
        previous = merged.get(key)
        if previous is not None and previous != int(order):
            if previous == int(struc.BondType.ANY):
                merged[key] = int(order)
            elif int(order) != int(struc.BondType.ANY):
                raise DesignSampleError(f"conflicting bond orders for final atom rows {key}")
        else:
            merged[key] = int(order)
    rows = np.asarray([(i, j, order) for (i, j), order in sorted(merged.items())], dtype=np.uint32)
    atoms.bonds = struc.BondList(
        len(atoms), rows if rows.size else np.empty((0, 3), dtype=np.uint32)
    )


def _polymer_template_bonds(
    atoms: struc.AtomArray,
    *,
    excluded_residue_uids: frozenset[int] = frozenset(),
    design_only: bool = False,
) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for asym_uid in np.unique(atoms.asym_uid):
        mask = (atoms.asym_uid == asym_uid) & (atoms.mol_type != "ligand")
        if design_only:
            mask &= np.asarray(atoms.is_design, dtype=bool)
        indices = np.where(mask)[0]
        if not len(indices):
            continue
        try:
            local = struc.connect_via_residue_names(atoms[indices], inter_residue=True)
        except Exception:
            continue
        for i, j, order in local.as_array():
            row_i, row_j = int(indices[int(i)]), int(indices[int(j)])
            if (
                int(atoms.residue_uid[row_i]) in excluded_residue_uids
                or int(atoms.residue_uid[row_j]) in excluded_residue_uids
            ):
                continue
            if _reader_module.polymer_template_bond_is_supported(atoms, row_i, row_j):
                result.append((row_i, row_j, int(order)))
    return result


def _strict_peptide_linking(component: CCDComponent | None) -> bool:
    """Return whether a CCD component can participate in an ordinary peptide link.

    This deliberately excludes peptide-like components and caps.  The raw CCD
    chemical type is authoritative; a polymer modality alone is not enough to
    infer a backbone reaction.
    """

    if component is None or component.polymer_modality != "protein":
        return False
    raw_type = " ".join((component.raw_chem_type or "").upper().split())
    return raw_type.endswith("PEPTIDE LINKING") and "PEPTIDE-LIKE" not in raw_type


def _carrier_boundary_peptide_bonds(
    atoms: struc.AtomArray,
    carrier_residue_uids: frozenset[int],
    *,
    embedded: Mapping[str, CCDComponent],
    configured: Mapping[str, CCDComponent],
    packaged: Mapping[str, CCDComponent],
) -> list[tuple[int, int, int]]:
    """Infer only ordinary C--N links at atomized-carrier residue boundaries.

    Standard polymer template generation intentionally skips every edge touching
    a carrier, because its atomwise model representation has no residue-token
    template to supply those edges.  Recover the two normal boundaries here,
    while leaving standard--standard, head--tail, peptide-like, cap, and other
    nonstandard chemistry to existing source/condition facts.

    The returned edges are structural AtomArray facts, not public
    ``TopologyCondition`` records.  This keeps a routine peptide boundary from
    being mislabeled as a user-declared nonstandard bond while still making the
    edge available to the canonical target/condition bond channels and writer.
    """

    if not carrier_residue_uids:
        return []

    components_by_residue: dict[int, CCDComponent | None] = {}
    rows_by_residue: dict[int, np.ndarray] = {}
    residue_groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for row in range(len(atoms)):
        residue_uid = int(atoms.residue_uid[row])
        if residue_uid not in rows_by_residue:
            rows_by_residue[residue_uid] = np.where(
                np.asarray(atoms.residue_uid, dtype=np.int64) == residue_uid
            )[0]
            residue_groups[(int(atoms.asym_uid[row]), str(atoms.chain_id[row]))].append(
                residue_uid
            )

    for residue_uid, rows in rows_by_residue.items():
        names = {str(value).strip().upper() for value in atoms.res_name[rows]}
        if len(names) != 1:
            raise DesignInputError(
                f"polymer residue UID {residue_uid} has mixed component IDs"
            )
        component_id = next(iter(names))
        resolution = resolve_ccd_component(
            component_id,
            embedded=embedded,
            configured=configured,
            packaged=packaged,
            include_packaged=False,
        )
        components_by_residue[residue_uid] = resolution.component

    result: list[tuple[int, int, int]] = []
    single = int(struc.BondType.SINGLE)
    for residue_uids in residue_groups.values():
        residue_uids.sort(key=lambda uid: _residue_order_key(atoms, uid))
        for left_uid, right_uid in zip(residue_uids, residue_uids[1:]):
            if left_uid not in carrier_residue_uids and right_uid not in carrier_residue_uids:
                continue
            left_component = components_by_residue.get(left_uid)
            right_component = components_by_residue.get(right_uid)
            if not (
                _strict_peptide_linking(left_component)
                and _strict_peptide_linking(right_component)
            ):
                continue

            left_rows = rows_by_residue[left_uid]
            right_rows = rows_by_residue[right_uid]
            if int(atoms.res_id[right_rows[0]]) != int(atoms.res_id[left_rows[0]]) + 1:
                # Residue order alone is not enough: a structural gap must not
                # synthesize an ordinary peptide bond across the missing ID.
                continue
            left_c = left_rows[atoms.atom_name[left_rows] == "C"]
            right_n = right_rows[atoms.atom_name[right_rows] == "N"]
            # A partial source atom intersection is valid, but it must not cause
            # us to invent a boundary whose endpoint is absent or ambiguous.
            if len(left_c) != 1 or len(right_n) != 1:
                continue
            result.append((int(left_c[0]), int(right_n[0]), single))
    return result


def _drop_carrier_incident_bonds(
    atoms: struc.AtomArray,
    carrier_residue_uids: frozenset[int],
) -> None:
    retained: list[tuple[int, int, int]] = []
    if atoms.bonds is not None:
        for raw_i, raw_j, raw_order in atoms.bonds.as_array():
            row_i, row_j = int(raw_i), int(raw_j)
            if (
                int(atoms.residue_uid[row_i]) in carrier_residue_uids
                or int(atoms.residue_uid[row_j]) in carrier_residue_uids
            ):
                continue
            retained.append((row_i, row_j, int(raw_order)))
    atoms.bonds = struc.BondList(
        len(atoms),
        np.asarray(retained, dtype=np.uint32)
        if retained
        else np.empty((0, 3), dtype=np.uint32),
    )


def _finalize_carriers(
    atoms: struc.AtomArray,
    origin_uid_by_row: np.ndarray,
    candidates: Mapping[int, _CarrierCandidate],
) -> tuple[
    dict[int, ModifiedPolymerCarrier],
    list[tuple[int, int, int]],
    list[TopologyCondition],
]:
    candidate_by_origin = {
        int(origin_uid): candidate
        for candidate in candidates.values()
        for origin_uid in candidate.canonical_by_origin_uid
    }
    selected: dict[int, _CarrierCandidate] = {}
    for raw_residue_uid in np.unique(atoms.residue_uid[atoms.mol_type != "ligand"]):
        residue_uid = int(raw_residue_uid)
        rows = np.where(atoms.residue_uid == residue_uid)[0]
        matches = {
            candidate_by_origin[int(origin_uid)].source_residue_uid
            for origin_uid in origin_uid_by_row[rows]
            if int(origin_uid) in candidate_by_origin
        }
        if not matches:
            continue
        if len(matches) != 1 or any(
            int(origin_uid) not in candidate_by_origin for origin_uid in origin_uid_by_row[rows]
        ):
            raise DesignInputError("final carrier residue lost its one-to-one source origin")
        selected[residue_uid] = candidates[next(iter(matches))]
        atoms.identity_condition[rows] = True

    source_max_molecule_uid = (
        -1 if not len(atoms) else int(np.asarray(atoms.molecule_uid).max())
    )
    carriers: dict[int, ModifiedPolymerCarrier] = {}
    internal_edges: list[tuple[int, int, int]] = []
    topology: list[TopologyCondition] = []
    for ordinal, residue_uid in enumerate(sorted(selected), start=1):
        candidate = selected[residue_uid]
        component = candidate.component
        rows = np.where(atoms.residue_uid == residue_uid)[0]
        row_by_name = {str(atoms.atom_name[row]): int(row) for row in rows}
        if len(row_by_name) != len(rows):
            raise DesignInputError("final carrier has duplicate canonical atom IDs")
        atom_by_id = {atom.atom_id: atom for atom in component.atoms}
        charges = tuple(
            (
                int(atoms.atom_uid[row]),
                int(atom_by_id[name].formal_charge or 0),
            )
            for name, row in sorted(row_by_name.items())
        )
        provenance = tuple(
            f"{item.kind}:{item.location}"
            for item in component.provenance
        )
        carriers[residue_uid] = ModifiedPolymerCarrier(
            residue_uid=residue_uid,
            component_id=component.component_id,
            model_molecule_uid=source_max_molecule_uid + ordinal,
            formal_charge_by_atom_uid=charges,
            provenance=provenance,
            report_residue_name=component.component_id,
            standard_parent=candidate.standard_parent,
        )
        for bond in component.bonds:
            row_i = row_by_name.get(bond.atom_id_1)
            row_j = row_by_name.get(bond.atom_id_2)
            if row_i is None or row_j is None:
                continue
            internal_edges.append((row_i, row_j, bond.biotite_order))
            uid_i, uid_j = sorted(
                (int(atoms.atom_uid[row_i]), int(atoms.atom_uid[row_j]))
            )
            topology.append(TopologyCondition(uid_i, uid_j, bond.order, True))
    return carriers, internal_edges, topology


def _inherit_carrier_source_edges(
    source: struc.AtomArray | None,
    source_bonds: Sequence[tuple[int, int, int]],
    atoms: struc.AtomArray,
    origin_uid_by_row: np.ndarray,
    carriers: Mapping[int, ModifiedPolymerCarrier],
) -> tuple[list[TopologyCondition], list[tuple[int, int, int]]]:
    if source is None or not carriers:
        return [], []
    final_rows_by_origin: dict[int, list[int]] = defaultdict(list)
    source_origins = {int(value) for value in np.asarray(source.atom_uid, dtype=np.int64)}
    for row, raw_origin in enumerate(origin_uid_by_row):
        origin = int(raw_origin)
        if origin in source_origins:
            final_rows_by_origin[origin].append(row)

    topology: list[TopologyCondition] = []
    structural: list[tuple[int, int, int]] = []
    carrier_residue_uids = set(int(value) for value in carriers)
    for raw_i, raw_j, raw_order in source_bonds:
        source_i, source_j = int(raw_i), int(raw_j)
        if int(source.residue_uid[source_i]) == int(source.residue_uid[source_j]):
            continue
        origins = (int(source.atom_uid[source_i]), int(source.atom_uid[source_j]))
        mapped = tuple(final_rows_by_origin.get(origin, []) for origin in origins)
        mapped_carrier = tuple(
            any(int(atoms.residue_uid[row]) in carrier_residue_uids for row in rows)
            for rows in mapped
        )
        if not any(mapped_carrier):
            continue
        if any(len(rows) > 1 for rows in mapped):
            raise DesignInputError(
                f"source external edge origins {origins!r} map ambiguously after copying"
            )
        for source_row, rows in zip((source_i, source_j), mapped, strict=True):
            if rows:
                final_row = rows[0]
                if (
                    bool(atoms.is_design[final_row])
                    and str(atoms.mol_type[final_row]) in _STANDARD_BY_MODALITY
                ):
                    residue_rows = np.where(
                        atoms.residue_uid == atoms.residue_uid[final_row]
                    )[0]
                    if not bool(atoms.identity_condition[residue_rows].all()) and not _identity_invariant_atom(
                        str(atoms.mol_type[final_row]), str(atoms.atom_name[final_row])
                    ):
                        raise DesignInputError(
                            "carrier external edge targets an identity-dependent "
                            "atom of a non-fixed design polymer"
                        )
            elif (
                bool(source.is_design[source_row])
                and str(source.mol_type[source_row]) in _STANDARD_BY_MODALITY
                and not _identity_invariant_atom(
                    str(source.mol_type[source_row]), str(source.atom_name[source_row])
                )
            ):
                raise DesignInputError(
                    "carrier external edge loses an identity-dependent design-polymer endpoint"
                )
        if not mapped[0] or not mapped[1]:
            continue
        row_i, row_j = mapped[0][0], mapped[1][0]
        if row_i == row_j:
            raise DesignInputError("source external edge collapsed to one final atom")
        order_name = _source_order_name(int(raw_order))
        uid_i, uid_j = sorted((int(atoms.atom_uid[row_i]), int(atoms.atom_uid[row_j])))
        topology.append(TopologyCondition(uid_i, uid_j, order_name, True))
        structural.append((row_i, row_j, int(raw_order)))
    return topology, structural


def _inherit_type_bond_source_edges(
    source: struc.AtomArray | None,
    source_bonds: Sequence[tuple[int, int, int]],
    atoms: struc.AtomArray,
    origin_uid_by_row: np.ndarray,
    ligand_modes: Mapping[str, str],
) -> tuple[list[TopologyCondition], list[tuple[int, int, int]]]:
    """Retain source edges crossing a hard ``type_bond`` ligand boundary.

    ``_subset_source_component()`` deliberately keeps only the selected
    component's internal BondList.  Its source ``atom_uid`` values are kept
    as temporary origins until ``_finalize_uids()``; use that ledger here so
    cross-component edges are restored without reconstructing identity from
    residue or atom names.
    """

    if source is None or not source_bonds:
        return [], []
    type_bond_chains = {
        str(chain_id)
        for chain_id, mode in ligand_modes.items()
        if mode == "type_bond"
    }
    if not type_bond_chains:
        return [], []

    final_rows_by_origin: dict[int, list[int]] = defaultdict(list)
    source_origins = {
        int(value) for value in np.asarray(source.atom_uid, dtype=np.int64)
    }
    for row, raw_origin in enumerate(np.asarray(origin_uid_by_row, dtype=np.int64)):
        origin = int(raw_origin)
        if origin in source_origins:
            final_rows_by_origin[origin].append(row)

    type_bond_origins = {
        int(source.atom_uid[row])
        for row in range(len(source))
        if str(source.chain_id[row]) in type_bond_chains
        and str(source.mol_type[row]) == "ligand"
    }
    if not type_bond_origins:
        return [], []

    topology: list[TopologyCondition] = []
    structural: list[tuple[int, int, int]] = []
    for raw_i, raw_j, raw_order in source_bonds:
        source_i, source_j = int(raw_i), int(raw_j)
        if int(raw_order) == int(struc.BondType.COORDINATION):
            # Coordination/metal edges are intentionally outside the
            # covalent type_bond contract and are not serialized here.
            continue
        origins = (
            int(source.atom_uid[source_i]),
            int(source.atom_uid[source_j]),
        )
        selected = (
            origins[0] in type_bond_origins,
            origins[1] in type_bond_origins,
        )
        if sum(selected) != 1:
            # Internal component edges are already retained by the
            # type_bond subset; unrelated source edges are out of scope.
            continue
        mapped = tuple(final_rows_by_origin.get(origin, []) for origin in origins)
        if any(len(rows) > 1 for rows in mapped):
            raise DesignInputError(
                f"source type_bond edge origins {origins!r} map ambiguously after compilation"
            )
        if not mapped[0] or not mapped[1]:
            # Crop/materialization may legitimately remove the non-selected
            # source endpoint.  Only restore edges whose two source atoms
            # remain in the compiled request.
            continue
        selected_row = mapped[0][0] if selected[0] else mapped[1][0]
        selected_chain = str(atoms.chain_id[selected_row])
        if (
            not bool(atoms.is_design[selected_row])
            or str(atoms.mol_type[selected_row]) != "ligand"
            or selected_chain not in type_bond_chains
        ):
            raise DesignInputError(
                "source type_bond edge selected endpoint changed modality or ownership"
            )
        row_i, row_j = mapped[0][0], mapped[1][0]
        if row_i == row_j:
            raise DesignInputError("source type_bond edge collapsed to one final atom")
        order_name = _source_order_name(int(raw_order))
        uid_i, uid_j = sorted(
            (int(atoms.atom_uid[row_i]), int(atoms.atom_uid[row_j]))
        )
        topology.append(TopologyCondition(uid_i, uid_j, order_name, True))
        structural.append((row_i, row_j, int(raw_order)))
    return topology, structural


def _resolve_atom_selector(
    atoms: struc.AtomArray,
    endpoint: Any,
    *,
    where: str,
) -> int:
    """Resolve one public atom selector against the current AtomArray.

    This helper is usable before UID finalization as well as after
    tokenization, so design-chain negative residue positions have one shared
    normalization path for bond endpoints and ``omit_atoms`` selectors.
    """

    item = _native_mapping(endpoint, f"{where} endpoint")
    allowed = {"chain", "resid", "ins_code", "atom"}
    if set(item).difference(allowed) or not {"chain", "atom"} <= set(item):
        raise DesignSampleError(
            f"{where} endpoint requires chain/atom and optional resid/ins_code"
        )
    chain, atom_name = item["chain"], item["atom"]
    if not isinstance(chain, str) or not isinstance(atom_name, str):
        raise DesignSampleError(f"{where} endpoint chain and atom must be strings")
    chain_rows = np.where(atoms.chain_id == chain)[0]
    if not len(chain_rows):
        raise DesignSampleError(f"{where} endpoint chain {chain!r} does not exist")
    modalities = {str(value) for value in atoms.mol_type[chain_rows]}
    if len(modalities) != 1:
        raise DesignSampleError(f"{where} endpoint chain {chain!r} has mixed modalities")
    ligand = modalities == {"ligand"}
    mask = (atoms.chain_id == chain) & (atoms.atom_name == atom_name)
    if ligand:
        if "resid" in item or "ins_code" in item:
            raise DesignSampleError(
                f"{where} designed ligand endpoints use stable atom name only"
            )
    else:
        if "resid" not in item:
            raise DesignSampleError(f"{where} polymer endpoint requires resid")
        chain_is_design = bool(atoms.is_design[chain_rows[0]])
        if chain_is_design:
            position = _resolve_final_position(
                item["resid"],
                int(atoms.res_id[chain_rows].max()),
                f"{where}.{chain}",
            )
            mask &= atoms.res_id == position
        else:
            if (
                isinstance(item["resid"], bool)
                or not isinstance(item["resid"], int)
                or item["resid"] <= 0
            ):
                raise DesignSampleError(
                    f"{where} context polymer resid must be a positive structural identifier"
                )
            mask &= atoms.label_seq_id == str(item["resid"])
        if "ins_code" in item:
            mask &= atoms.ins_code == str(item["ins_code"])
    matches = np.where(mask)[0]
    if len(matches) != 1:
        raise DesignSampleError(
            f"{where} endpoint {dict(item)!r} matched {len(matches)} final atoms"
        )
    return int(matches[0])


def _resolve_final_atom(
    atoms: struc.AtomArray,
    endpoint: Any,
    *,
    addressable_design_ligands: set[str],
) -> int:
    item = _native_mapping(endpoint, "bond endpoint")
    chain = item.get("chain")
    row = _resolve_atom_selector(atoms, endpoint, where="bond")
    chain_rows = np.where(atoms.chain_id == chain)[0]
    ligand = set(atoms.mol_type[chain_rows]) == {"ligand"}
    if ligand:
        if bool(atoms.is_design[row]) and chain not in addressable_design_ligands:
            raise DesignSampleError(
                "design ligand atom endpoints require CIF type_bond or selected atom_motif identity"
            )
        if not bool(atoms.identity_condition[row]):
            raise DesignSampleError("ligand bond endpoint identity is not revealed/addressable")
    else:
        token_rows = np.where(atoms.token_uid == atoms.token_uid[row])[0]
        atom_name = str(atoms.atom_name[row])
        modality = str(atoms.mol_type[row])
        chain_is_design = bool(atoms.is_design[chain_rows[0]])
        identity_invariant_atom = _identity_invariant_atom(modality, atom_name)
        if not bool(atoms.identity_condition[token_rows].all()) and not (
            chain_is_design and identity_invariant_atom
        ):
            raise DesignSampleError(
                "required polymer endpoint needs sequence/contig-fixed token identity "
                "or an identity-invariant design-polymer atom"
            )
    return row


def _is_head_to_tail_closure(
    atoms: struc.AtomArray,
    atom_i: int,
    atom_j: int,
    order: str,
) -> bool:
    """Return whether one exact PP condition is a design-protein head-tail closure."""

    if order != "single":
        return False
    rows = np.asarray([atom_i, atom_j], dtype=np.int64)
    if not bool(np.asarray(atoms.is_design[rows], dtype=bool).all()):
        return False
    if set(str(value) for value in atoms.mol_type[rows]) != {"protein"}:
        return False
    asym_uids = np.unique(atoms.asym_uid[rows])
    if len(asym_uids) != 1:
        return False
    chain_rows = np.where(
        np.asarray(atoms.is_design, dtype=bool)
        & (np.asarray(atoms.mol_type).astype(str) == "protein")
        & (atoms.asym_uid == int(asym_uids[0]))
    )[0]
    first_position = int(np.asarray(atoms.res_id[chain_rows], dtype=np.int64).min())
    last_position = int(np.asarray(atoms.res_id[chain_rows], dtype=np.int64).max())
    endpoints = {
        (str(atoms.atom_name[row]), int(atoms.res_id[row])) for row in rows
    }
    return endpoints == {("N", first_position), ("C", last_position)}


def _identity_invariant_atom(modality: str, atom_name: str) -> bool:
    """Whether every standard identity for a modality owns this atom name."""

    if modality == "protein" and atom_name == "OXT":
        # OXT is materialized only for a terminal state, but is independent of
        # the terminal amino-acid identity once that row exists.
        return True
    inventories = _INVENTORIES.get(modality, {})
    return bool(inventories) and all(
        atom_name in inventory for inventory in inventories.values()
    )


def _residue_key(atoms: struc.AtomArray, row: int) -> tuple[str, int, str]:
    return (
        str(atoms.chain_id[row]),
        int(atoms.res_id[row]),
        str(atoms.ins_code[row]),
    )


def _remove_atom_rows(
    atoms: struc.AtomArray,
    omitted_rows: set[int],
) -> struc.AtomArray:
    """Drop atoms and remap only the surviving incident BondList edges."""

    if not omitted_rows:
        return atoms
    keep = np.ones(len(atoms), dtype=bool)
    keep[np.asarray(sorted(omitted_rows), dtype=np.int64)] = False
    old_to_new = np.full(len(atoms), -1, dtype=np.int64)
    old_to_new[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    old_bonds = () if atoms.bonds is None else atoms.bonds.as_array()
    remapped: list[tuple[int, int, int]] = []
    for raw_i, raw_j, raw_order in old_bonds:
        old_i, old_j = int(raw_i), int(raw_j)
        new_i, new_j = int(old_to_new[old_i]), int(old_to_new[old_j])
        if new_i >= 0 and new_j >= 0:
            remapped.append((new_i, new_j, int(raw_order)))
    result = atoms[keep]
    result.coord = np.asarray(result.coord, dtype=np.float32)
    result.bonds = struc.BondList(
        len(result),
        np.asarray(remapped, dtype=np.uint32)
        if remapped
        else np.empty((0, 3), dtype=np.uint32),
    )
    return result


def _collect_omitted_atoms(
    atoms: struc.AtomArray,
    condition: Mapping[str, Any],
    *,
    automatic_rows: set[int] | None = None,
    ligand_modes: Mapping[str, str] | None = None,
) -> tuple[struc.AtomArray, dict[tuple[str, int, str], tuple[str, ...]]]:
    """Validate and apply all bond-local ``omit_atoms`` selectors.

    Selectors are resolved against the complete pre-UID AtomArray.  The
    returned residue-key sidecar is converted to final token UIDs only after
    UID assignment and tokenization.
    """

    declarations = condition.get("bond", []) or []
    if not isinstance(declarations, list):
        raise DesignSampleError("condition.bond must be a list")
    automatic_rows = set() if automatic_rows is None else set(automatic_rows)
    omitted_rows: set[int] = set(automatic_rows)
    omitted_by_residue: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for row in automatic_rows:
        if row < 0 or row >= len(atoms):
            raise DesignInputError("automatic leaving-atom row is out of range")
        if bool(atoms.motif_coord_condition[row]) or bool(
            atoms.token_center_condition[row]
        ):
            raise DesignInputError(
                "automatic linkage leaving-atom removal would erase a hard coordinate"
            )
        omitted_by_residue[_residue_key(atoms, row)].add(str(atoms.atom_name[row]))
    endpoint_rows: set[int] = set()
    for index, declaration in enumerate(declarations):
        row = _native_mapping(declaration, f"condition.bond[{index}]")
        endpoint_i = _resolve_atom_selector(
            atoms,
            row.get("atom1"),
            where=f"condition.bond[{index}].atom1",
        )
        endpoint_j = _resolve_atom_selector(
            atoms,
            row.get("atom2"),
            where=f"condition.bond[{index}].atom2",
        )
        if endpoint_i == endpoint_j:
            raise DesignSampleError("bond condition cannot be a self-edge")
        endpoint_rows.update((endpoint_i, endpoint_j))
        if "omit_atoms" not in row:
            continue
        raw_omit = row["omit_atoms"]
        if not isinstance(raw_omit, list) or not raw_omit:
            raise DesignSampleError(
                f"condition.bond[{index}].omit_atoms must be a non-empty list"
            )
        endpoint_residues = {
            _residue_key(atoms, endpoint_i),
            _residue_key(atoms, endpoint_j),
        }
        for omit_index, selector in enumerate(raw_omit):
            omit_row = _resolve_atom_selector(
                atoms,
                selector,
                where=f"condition.bond[{index}].omit_atoms[{omit_index}]",
            )
            modality = str(atoms.mol_type[omit_row])
            if modality not in {"protein", "rna", "dna", "ligand"} or not bool(
                atoms.is_design[omit_row]
            ):
                raise DesignSampleError(
                    "omit_atoms may target only design polymer or fixed ligand atoms"
                )
            if modality == "ligand":
                chain_id = str(atoms.chain_id[omit_row])
                component_rows = np.where(
                    (atoms.chain_id == chain_id)
                    & (atoms.mol_type == "ligand")
                )[0]
                if (
                    (ligand_modes or {}).get(chain_id) != "type_bond"
                    or not bool(atoms.identity_condition[component_rows].all())
                ):
                    raise DesignSampleError(
                        "ligand omit_atoms requires fixed complete CCD/type_bond chemistry"
                    )
            if _residue_key(atoms, omit_row) not in endpoint_residues:
                raise DesignSampleError(
                    "omitted atom residue must match one of the bond endpoint residues"
                )
            if omit_row in {endpoint_i, endpoint_j}:
                raise DesignSampleError("omit_atoms cannot remove a bond endpoint")
            atom_name = str(atoms.atom_name[omit_row])
            representative = _CENTER_ATOM.get(modality)
            if representative is not None and atom_name == representative:
                raise DesignSampleError(
                    f"omit_atoms cannot remove representative atom {representative!r}"
                )
            if bool(atoms.motif_coord_condition[omit_row]) or bool(
                atoms.token_center_condition[omit_row]
            ):
                raise DesignSampleError(
                    "omit_atoms cannot remove hard motif or token-center coordinates"
                )
            if not bool(atoms.identity_condition[omit_row]) and not _identity_invariant_atom(
                modality, atom_name
            ):
                raise DesignSampleError(
                    "omit_atoms cannot remove an identity-dependent atom from a non-fixed identity"
                )
            omitted_rows.add(omit_row)
            omitted_by_residue[_residue_key(atoms, omit_row)].add(atom_name)
    if omitted_rows.intersection(endpoint_rows):
        raise DesignSampleError("omit_atoms cannot remove a declared bond endpoint")
    return _remove_atom_rows(atoms, omitted_rows), {
        key: tuple(sorted(names)) for key, names in omitted_by_residue.items()
    }


def _finalize_omitted_atom_sidecar(
    atoms: struc.AtomArray,
    omitted_by_residue: Mapping[tuple[str, int, str], Sequence[str]],
) -> dict[int, tuple[str, ...]]:
    result: dict[int, tuple[str, ...]] = {}
    for residue_key, names in omitted_by_residue.items():
        chain, resid, ins_code = residue_key
        rows = np.where(
            (atoms.chain_id == chain)
            & (atoms.res_id == resid)
            & (atoms.ins_code == ins_code)
        )[0]
        if not len(rows):
            raise DesignSampleError(
                f"omitted atom residue {chain}:{resid}{ins_code} disappeared during compilation"
            )
        # A fixed ligand uses the already trimmed authoring inventory; only
        # polymer reconstruction needs an omitted-name sidecar per residue token.
        if np.all(atoms.mol_type[rows] == "ligand"):
            continue
        token_uids = np.unique(atoms.token_uid[rows])
        if len(token_uids) != 1:
            raise DesignSampleError(
                f"omitted atom residue {chain}:{resid}{ins_code} maps to multiple tokens"
            )
        token_uid = int(token_uids[0])
        prior = set(result.get(token_uid, ()))
        prior.update(str(name) for name in names)
        result[token_uid] = tuple(sorted(prior))
    return result


def _explicit_topology(
    atoms: struc.AtomArray,
    condition: Mapping[str, Any],
    ligand_modes: Mapping[str, str],
) -> tuple[list[TopologyCondition], list[tuple[int, int, int]]]:
    declarations = condition.get("bond", []) or []
    if not isinstance(declarations, list):
        raise DesignSampleError("condition.bond must be a list")
    topology: dict[tuple[int, int], TopologyCondition] = {}
    structural: list[tuple[int, int, int]] = []
    addressable_design_ligands = {
        chain for chain, mode in ligand_modes.items() if mode in {"type_bond", "atom_motif"}
    }
    for index, declaration in enumerate(declarations):
        row = _native_mapping(declaration, f"condition.bond[{index}]")
        order_name = str(row.get("order", "any")).strip().lower()
        if set(row).difference({"atom1", "atom2", "order", "omit_atoms"}) or not {
            "atom1", "atom2"
        } <= set(row) or order_name not in {
            *_ORDER_TO_BIOTITE,
            *_UNKNOWN_POSITIVE_ORDERS,
        }:
            raise DesignSampleError(
                "bond rows require atom1 and atom2 plus an optional supported "
                "exact or unknown-positive order"
            )
        atom_i = _resolve_final_atom(
            atoms, row["atom1"], addressable_design_ligands=addressable_design_ligands
        )
        atom_j = _resolve_final_atom(
            atoms, row["atom2"], addressable_design_ligands=addressable_design_ligands
        )
        if atom_i == atom_j:
            raise DesignSampleError("bond condition cannot be a self-edge")
        uid_pair = tuple(sorted((int(atoms.atom_uid[atom_i]), int(atoms.atom_uid[atom_j]))))
        # Every user-declared bond fixes existence and must be serialized.
        # An unknown-positive order does not invent ``single``: CCD or model
        # chemistry may refine it later, otherwise the writer emits ``?``.
        required_output = True
        record = TopologyCondition(
            uid_pair[0], uid_pair[1], order_name, required_output
        )
        previous = topology.get(uid_pair)
        if previous is not None:
            if previous.order != record.order:
                raise DesignSampleError(f"conflicting explicit bond orders for AtomUID pair {uid_pair}")
            continue
        topology[uid_pair] = record
        if order_name in _ORDER_TO_BIOTITE:
            structural.append((atom_i, atom_j, _ORDER_TO_BIOTITE[order_name]))
    return list(topology.values()), structural


def _apply_hotspot(atoms: struc.AtomArray, source: struc.AtomArray | None, selector: str | None) -> None:
    if selector is None:
        return
    if source is None:
        raise DesignSampleError("hotspot requires an input structure")
    source_uids = _source_selector(source, selector)
    selected_keys: list[tuple[str, int, str]] = []
    for uid in source_uids:
        rows = np.where(source.residue_uid == uid)[0]
        if bool(source.is_design[rows].any()):
            raise DesignSampleError("hotspot may select context residues only")
        selected_keys.append((str(source.chain_id[rows[0]]), int(source.res_id[rows[0]]), str(source.ins_code[rows[0]])))
    for chain, resid, ins_code in selected_keys:
        rows = np.where(
            (atoms.chain_id == chain) & (atoms.res_id == resid) & (atoms.ins_code == ins_code)
        )[0]
        if not len(rows) or bool(atoms.is_design[rows].any()):
            raise DesignSampleError(f"hotspot residue {chain}{resid}{ins_code} is not retained context")
        atoms.hotspot_condition[rows] = True


def _final_range_positions(text: str, *, expected_chain: str | None, chain_lengths: Mapping[str, int]) -> tuple[str, list[int]]:
    match = _FINAL_RANGE.fullmatch(str(text).strip())
    if match is None:
        raise DesignSampleError(f"invalid final-design selector: {text!r}")
    chain = match.group(1) or match.group(2) or expected_chain
    if chain is None or chain not in chain_lengths:
        raise DesignSampleError(f"final-design selector has unknown/missing chain: {text!r}")
    length = chain_lengths[chain]
    start = _resolve_final_position(int(match.group(3)), length, f"selector.{chain}")
    end = start if match.group(4) is None else _resolve_final_position(int(match.group(4)), length, f"selector.{chain}")
    if start > end:
        raise DesignSampleError(f"final-design range is reversed: {text!r}")
    return chain, list(range(start, end + 1))


def _secondary_structure(
    atoms: struc.AtomArray,
    condition: Mapping[str, Any],
    carriers: Mapping[int, ModifiedPolymerCarrier] | None = None,
) -> tuple[dict[int, int], dict[int, int], tuple[TokenPairCondition, ...]]:
    config = condition.get("secondary_structure")
    if config is None:
        return {}, {}, ()
    config = _native_mapping(config, "condition.secondary_structure")
    if set(config).difference({"coarse", "token", "pair"}):
        raise DesignSampleError("unknown secondary_structure channel")
    chain_lengths = {
        str(chain): int(atoms.res_id[atoms.chain_id == chain].max())
        for chain in np.unique(atoms.chain_id[atoms.is_design & (atoms.mol_type != "ligand")])
    }
    chain_modalities = {
        chain: str(atoms.mol_type[np.where(atoms.chain_id == chain)[0][0]])
        for chain in chain_lengths
    }
    carrier_residue_uids = set() if carriers is None else {
        int(residue_uid) for residue_uid in carriers
    }
    carrier_token_uids = {
        int(token_uid)
        for token_uid, residue_uid in zip(
            np.asarray(atoms.token_uid, dtype=np.int64),
            np.asarray(atoms.residue_uid, dtype=np.int64),
        )
        if int(residue_uid) in carrier_residue_uids
    }
    coarse: dict[int, int] = {}
    coarse_values = {"alpha": 1, "beta": 2, "paired": 1, "unpaired": 2}
    for chain, name in _native_mapping(config.get("coarse", {}) or {}, "secondary_structure.coarse").items():
        if chain not in chain_lengths or name not in coarse_values:
            raise DesignSampleError(f"invalid coarse secondary structure {chain}:{name}")
        if chain_modalities[chain] == "protein" and name not in {"alpha", "beta"}:
            raise DesignSampleError("protein coarse SS accepts alpha or beta")
        if chain_modalities[chain] in {"rna", "dna"} and name not in {"paired", "unpaired"}:
            raise DesignSampleError("nucleic coarse SS accepts paired or unpaired")
        for token_uid in np.unique(atoms.token_uid[atoms.chain_id == chain]):
            if int(token_uid) in carrier_token_uids:
                continue
            coarse[int(token_uid)] = coarse_values[name]

    token_values = {"helix": 1, "sheet": 2, "loop": 3, "paired": 1, "unpaired": 2}
    ss_token: dict[int, int] = {}
    declared_blocks: dict[tuple[str, str], set[tuple[int, ...]]] = defaultdict(set)
    token_config = _native_mapping(config.get("token", {}) or {}, "secondary_structure.token")
    for chain, classes in token_config.items():
        if chain not in chain_lengths:
            raise DesignSampleError(f"secondary_structure.token has unknown chain {chain!r}")
        for class_name, ranges in _native_mapping(classes, f"secondary_structure.token.{chain}").items():
            if class_name not in token_values or not isinstance(ranges, str):
                raise DesignSampleError(f"invalid token SS class/range {chain}:{class_name}")
            allowed = {"helix", "sheet", "loop"} if chain_modalities[chain] == "protein" else {"paired", "unpaired"}
            if class_name not in allowed:
                raise DesignSampleError(f"SS class {class_name!r} is incompatible with {chain_modalities[chain]}")
            for raw_range in ranges.split(","):
                _, positions = _final_range_positions(raw_range, expected_chain=chain, chain_lengths=chain_lengths)
                declared_blocks[(chain, class_name)].add(tuple(positions))
                for position in positions:
                    rows = np.where((atoms.chain_id == chain) & (atoms.res_id == position))[0]
                    token_uid = int(atoms.token_uid[rows[0]])
                    value = token_values[class_name]
                    if token_uid in ss_token and ss_token[token_uid] != value:
                        raise DesignSampleError(f"overlapping contradictory SS token rule at {chain}:{position}")
                    ss_token[token_uid] = value

    pair_values = {"adjacent": 2, "not_adjacent": 1, "paired": 2, "unpaired": 1}
    pair_records: dict[tuple[int, int], TokenPairCondition] = {}
    pair_config = _native_mapping(config.get("pair", {}) or {}, "secondary_structure.pair")
    for class_name, entries in pair_config.items():
        if class_name not in pair_values or not isinstance(entries, list):
            raise DesignSampleError(f"invalid secondary_structure pair class {class_name!r}")
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                raise DesignSampleError("SS pair entries must contain exactly two selectors")
            chain_a, positions_a = _final_range_positions(entry[0], expected_chain=None, chain_lengths=chain_lengths)
            chain_b, positions_b = _final_range_positions(entry[1], expected_chain=None, chain_lengths=chain_lengths)
            endpoint_token_uids = []
            for chain, positions in ((chain_a, positions_a), (chain_b, positions_b)):
                for position in positions:
                    rows = np.where((atoms.chain_id == chain) & (atoms.res_id == position))[0]
                    endpoint_token_uids.append(int(atoms.token_uid[rows[0]]))
            if any(token_uid not in ss_token for token_uid in endpoint_token_uids):
                raise DesignSampleError(
                    "SS pair endpoints require compatible token-channel conditions"
                )
            if class_name in {"adjacent", "not_adjacent"}:
                if chain_a != chain_b or chain_modalities[chain_a] != "protein":
                    raise DesignSampleError("protein SS block pairs must be same-chain protein")
                declared_protein_blocks = (
                    declared_blocks.get((chain_a, "helix"), set())
                    | declared_blocks.get((chain_a, "sheet"), set())
                )
                if tuple(positions_a) not in declared_protein_blocks or tuple(positions_b) not in declared_protein_blocks:
                    raise DesignSampleError(
                        "protein SS pair endpoints must exactly match declared helix/sheet blocks"
                    )
            else:
                if chain_modalities[chain_a] not in {"rna", "dna"} or chain_modalities[chain_b] not in {"rna", "dna"}:
                    raise DesignSampleError("nucleic SS pairs require RNA/DNA endpoints")
                if class_name == "unpaired" and chain_a != chain_b:
                    raise DesignSampleError("known-negative nucleic SS pairs must stay within one chain")
            for position_a in positions_a:
                for position_b in positions_b:
                    rows_a = np.where((atoms.chain_id == chain_a) & (atoms.res_id == position_a))[0]
                    rows_b = np.where((atoms.chain_id == chain_b) & (atoms.res_id == position_b))[0]
                    token_a, token_b = int(atoms.token_uid[rows_a[0]]), int(atoms.token_uid[rows_b[0]])
                    if token_a not in ss_token or token_b not in ss_token:
                        raise DesignSampleError(
                            "SS pair endpoints require compatible token-channel conditions"
                        )
                    if class_name in {"adjacent", "not_adjacent"}:
                        if ss_token[token_a] not in {1, 2} or ss_token[token_b] not in {1, 2}:
                            raise DesignSampleError("protein SS pair endpoints cannot be loop tokens")
                    else:
                        expected_token_state = 1 if class_name == "paired" else 2
                        if ss_token[token_a] != expected_token_state or ss_token[token_b] != expected_token_state:
                            raise DesignSampleError(
                                "nucleic SS pair state conflicts with token-channel classes"
                            )
                    if token_a == token_b:
                        raise DesignSampleError("SS pair diagonal is invalid")
                    key = tuple(sorted((token_a, token_b)))
                    value = pair_values[class_name]
                    previous = pair_records.get(key)
                    if previous is not None and previous.value != value:
                        raise DesignSampleError(f"contradictory SS pair declaration for tokens {key}")
                    pair_records[key] = TokenPairCondition(key[0], key[1], value)
    return coarse, ss_token, tuple(pair_records[key] for key in sorted(pair_records))


def _reject_carrier_secondary_structure(
    atoms: struc.AtomArray,
    carriers: Mapping[int, ModifiedPolymerCarrier],
    coarse: Mapping[int, int],
    ss_token: Mapping[int, int],
    ss_pair: Sequence[TokenPairCondition],
) -> None:
    if not carriers:
        return
    carrier_tokens = {
        int(token_uid)
        for token_uid, residue_uid in zip(atoms.token_uid, atoms.residue_uid)
        if int(residue_uid) in carriers
    }
    pair_tokens = {
        token_uid
        for record in ss_pair
        for token_uid in record.canonical_tokens()
    }
    touched = carrier_tokens.intersection(
        set(int(value) for value in coarse)
        | set(int(value) for value in ss_token)
        | pair_tokens
    )
    if touched:
        raise DesignSampleError(
            "secondary-structure conditions cannot target modified-polymer "
            f"carrier tokens {sorted(touched)}"
        )


def _affines(atoms: struc.AtomArray) -> tuple[np.ndarray, np.ndarray]:
    visible = atoms.coord_valid & (atoms.motif_coord_condition | atoms.token_center_condition)
    if not visible.any():
        visible = atoms.coord_valid & atoms.centre_atom_mask
    center_nm = np.zeros(3, dtype=np.float64)
    if visible.any():
        center_nm = np.asarray(atoms.coord[visible], dtype=np.float64).mean(axis=0) / 10.0
    forward = np.eye(4, dtype=np.float64)
    forward[3, :3] = -center_nm
    inverse = np.eye(4, dtype=np.float64)
    inverse[3, :3] = center_nm
    return forward, inverse


def _edge_policies(
    atoms: struc.AtomArray,
    ligand_modes: Mapping[str, str],
    explicit_topology: Sequence[TopologyCondition],
) -> tuple[EdgePolicy, ...]:
    policies: list[EdgePolicy] = []
    bond_rows = np.empty((0, 3), dtype=np.uint32) if atoms.bonds is None else atoms.bonds.as_array()
    for chain_id, mode in ligand_modes.items():
        rows = np.where((atoms.chain_id == chain_id) & (atoms.mol_type == "ligand"))[0]
        if not len(rows):
            continue
        component_uids = np.unique(atoms.component_uid[rows])
        if len(component_uids) != 1:
            raise DesignSampleError(f"ligand chain {chain_id!r} does not have one component UID")
        row_set = set(rows.tolist())
        hard_edges = tuple(
            (min(int(atoms.atom_uid[i]), int(atoms.atom_uid[j])), max(int(atoms.atom_uid[i]), int(atoms.atom_uid[j])))
            for i, j, _ in bond_rows
            if int(i) in row_set and int(j) in row_set
        )
        if mode in {"smiles", "type_bond"}:
            policies.append(
                EdgePolicy(
                    component_uid=int(component_uids[0]),
                    complete_graph=True,
                    hard_positive_edges=hard_edges,
                    required_output=True,
                )
            )
        elif mode == "atom_motif":
            row_uids = {int(atoms.atom_uid[row]) for row in rows}
            explicit_edges = tuple(
                record.canonical_uids()
                for record in explicit_topology
                if record.required_output
                and set(record.canonical_uids()) <= row_uids
            )
            if not explicit_edges:
                continue
            policies.append(
                EdgePolicy(
                    component_uid=int(component_uids[0]),
                    complete_graph=False,
                    hard_positive_edges=explicit_edges,
                    required_output=True,
                )
            )
    return tuple(policies)


def _source_atom_key(atoms: struc.AtomArray, row: int) -> tuple[str, str, str, str]:
    return (
        str(atoms.label_asym_id[row]),
        str(atoms.label_seq_id[row]),
        str(atoms.res_name[row]),
        str(atoms.atom_name[row]),
    )


def _inherit_atom_motif_pl_edges(
    source: struc.AtomArray | None,
    atoms: struc.AtomArray,
    ligand_modes: Mapping[str, str],
) -> tuple[list[TopologyCondition], list[tuple[int, int, int]]]:
    if source is None or source.bonds is None:
        return [], []
    motif_chains = {chain for chain, mode in ligand_modes.items() if mode == "atom_motif"}
    if not motif_chains:
        return [], []
    final_by_key: dict[tuple[str, str, str, str], int] = {}
    for row in range(len(atoms)):
        key = _source_atom_key(atoms, row)
        if all(value not in {"", ".", "?"} for value in (key[0], key[2], key[3])):
            if key in final_by_key:
                raise DesignSampleError(f"retained source atom identity is ambiguous: {key}")
            final_by_key[key] = row
    topology: list[TopologyCondition] = []
    structural: list[tuple[int, int, int]] = []
    for raw_i, raw_j, raw_order in source.bonds.as_array():
        source_i, source_j = int(raw_i), int(raw_j)
        ligand_i = str(source.mol_type[source_i]) == "ligand"
        ligand_j = str(source.mol_type[source_j]) == "ligand"
        if ligand_i == ligand_j:
            continue
        motif_source = source_i if ligand_i else source_j
        context_source = source_j if ligand_i else source_i
        if str(source.chain_id[motif_source]) not in motif_chains:
            continue
        motif_final = final_by_key.get(_source_atom_key(source, motif_source))
        context_final = final_by_key.get(_source_atom_key(source, context_source))
        if motif_final is None or context_final is None:
            # Unselected motif-ligand atoms deliberately do not inherit their PL edge.
            continue
        order = int(raw_order)
        if order in {
            int(struc.BondType.AROMATIC), int(struc.BondType.AROMATIC_SINGLE),
            int(struc.BondType.AROMATIC_DOUBLE), int(struc.BondType.AROMATIC_TRIPLE),
        }:
            raise DesignSampleError("aromatic polymer-ligand covalent edges are unsupported")
        order_name = _BIOTITE_TO_ORDER.get(order)
        if order_name is None:
            raise DesignSampleError("atom_motif inherited PL edge requires an exact supported bond order")
        uid_i, uid_j = sorted((int(atoms.atom_uid[motif_final]), int(atoms.atom_uid[context_final])))
        topology.append(TopologyCondition(uid_i, uid_j, order_name, True))
        structural.append((motif_final, context_final, order))
    return topology, structural


def _ligand_mode_topology(
    atoms: struc.AtomArray,
    ligand_modes: Mapping[str, str],
) -> list[TopologyCondition]:
    if atoms.bonds is None:
        return []
    result: list[TopologyCondition] = []
    for raw_i, raw_j, raw_order in atoms.bonds.as_array():
        row_i, row_j = int(raw_i), int(raw_j)
        if str(atoms.mol_type[row_i]) != "ligand" or str(atoms.mol_type[row_j]) != "ligand":
            continue
        chain_i, chain_j = str(atoms.chain_id[row_i]), str(atoms.chain_id[row_j])
        if chain_i != chain_j or chain_i not in ligand_modes:
            continue
        mode = ligand_modes[chain_i]
        if mode not in {"smiles", "type_bond"}:
            continue
        order_name = (
            "aromatic"
            if int(raw_order) in _AROMATIC_BIOTITE_ORDERS
            else _BIOTITE_TO_ORDER.get(int(raw_order))
        )
        if order_name is None:
            raise DesignSampleError(f"conditioned ligand edge has unsupported/unknown order: {raw_order}")
        uid_i, uid_j = sorted((int(atoms.atom_uid[row_i]), int(atoms.atom_uid[row_j])))
        result.append(TopologyCondition(uid_i, uid_j, order_name, True))
    return result


def _merge_topology_conditions(
    atoms: struc.AtomArray,
    records: Sequence[TopologyCondition],
    model_layout: ModelTokenLayout,
) -> tuple[TopologyCondition, ...]:
    uid_to_token = dict(model_layout.model_token_by_atom_uid)
    by_uid: dict[tuple[int, int], TopologyCondition] = {}
    token_pair_to_uid: dict[tuple[int, int], tuple[int, int]] = {}
    for record in records:
        uid_pair = record.canonical_uids()
        if uid_pair[0] not in uid_to_token or uid_pair[1] not in uid_to_token:
            raise DesignSampleError(f"topology condition references missing AtomUID pair {uid_pair}")
        previous = by_uid.get(uid_pair)
        if previous is not None:
            previous_unknown = previous.order in _UNKNOWN_POSITIVE_ORDERS
            current_unknown = record.order in _UNKNOWN_POSITIVE_ORDERS
            if previous.order == record.order:
                merged_order = previous.order
            elif previous_unknown and not current_unknown:
                merged_order = record.order
            elif current_unknown and not previous_unknown:
                merged_order = previous.order
            else:
                raise DesignSampleError(
                    f"conflicting topology facts for AtomUID pair {uid_pair}"
                )
            by_uid[uid_pair] = TopologyCondition(
                uid_pair[0],
                uid_pair[1],
                merged_order,
                previous.required_output or record.required_output,
            )
            continue
        token_pair = tuple(sorted((uid_to_token[uid_pair[0]], uid_to_token[uid_pair[1]])))
        existing_uid_pair = token_pair_to_uid.get(token_pair)
        if existing_uid_pair is not None and existing_uid_pair != uid_pair:
            raise DesignSampleError(
                f"multiple distinct atom edges map to model token pair {token_pair}"
            )
        token_pair_to_uid[token_pair] = uid_pair
        by_uid[uid_pair] = record
    return tuple(by_uid[key] for key in sorted(by_uid))


def _clone_source_atom_array(source: struc.AtomArray | None) -> struc.AtomArray | None:
    return None if source is None else source.copy()


def _freeze_carrier_candidates(
    candidates: Mapping[int, _CarrierCandidate],
) -> Mapping[int, _CarrierCandidate]:
    return MappingProxyType(
        {
            int(residue_uid): _CarrierCandidate(
                source_residue_uid=int(candidate.source_residue_uid),
                component=candidate.component,
                canonical_by_origin_uid=MappingProxyType(
                    dict(candidate.canonical_by_origin_uid)
                ),
                standard_parent=candidate.standard_parent,
            )
            for residue_uid, candidate in candidates.items()
        }
    )


def _clone_carrier_candidates(
    candidates: Mapping[int, _CarrierCandidate],
) -> dict[int, _CarrierCandidate]:
    return {
        int(residue_uid): _CarrierCandidate(
            source_residue_uid=int(candidate.source_residue_uid),
            component=candidate.component,
            canonical_by_origin_uid=dict(candidate.canonical_by_origin_uid),
            standard_parent=candidate.standard_parent,
        )
        for residue_uid, candidate in candidates.items()
    }


def _source_path_key(path: Path | None) -> Path | None:
    return None if path is None else Path(path).resolve()


def _validate_prepared_task_source(
    prepared_source: PreparedTaskSource,
    task: TaskSpec,
    ccd_path: Path | None,
) -> Path | None:
    if not isinstance(prepared_source, PreparedTaskSource):
        raise DesignSampleError("prepared_source must be a PreparedTaskSource")
    if prepared_source.task is not task:
        raise DesignSampleError(
            "prepared source belongs to a different task identity"
        )
    if _source_path_key(prepared_source.task.input_path) != _source_path_key(
        task.input_path
    ):
        raise DesignSampleError("prepared source input path does not match task")
    if ccd_path is not None and _source_path_key(prepared_source.ccd_path) != _source_path_key(
        ccd_path
    ):
        raise DesignSampleError("prepared source CCD path does not match request")
    return prepared_source.ccd_path if ccd_path is None else Path(ccd_path)


def prepare_task_source(
    task: TaskSpec,
    *,
    ccd_path: Path | None = None,
) -> PreparedTaskSource:
    """Read, validate, and crop one task's deterministic source stage."""

    ccd_path = None if ccd_path is None else Path(ccd_path)
    condition = _native_mapping(task.condition, "condition")
    ligand_condition = _native_mapping(
        condition.get("ligand", {}) or {},
        "condition.ligand",
    )
    strict_type_bond_chains = frozenset(
        str(chain_id)
        for chain_id, raw_entry in ligand_condition.items()
        if isinstance(raw_entry, Mapping) and raw_entry.get("mode") == "type_bond"
    )
    embedded_ccd, configured_ccd, packaged_ccd = _polymer_ccd_sources(
        task.input_path, ccd_path
    )
    parent_sources = _ccd_parent_hints(
        embedded_ccd, configured_ccd, packaged_ccd
    )
    parent_sources.update(_parent_sources(task.input_path, ccd_path))
    # Parent hints from a concrete CCD component outrank the historical
    # COMMON_STANDARD_PARENT table.  The latter is only a last-resort name
    # fallback; for example packaged 5HM explicitly declares RNA parent C,
    # while the old kernel entry incorrectly names DC.  Apply the source
    # precedence used by resolve_ccd_component (packaged < configured < input).
    from src.common.chemistry.ccd import ccd_polymer_parent_ids
    if isinstance(packaged_ccd, _LazyTypedComponents):
        for component_id, parents in ccd_polymer_parent_ids().items():
            if parents is not None:
                parent_sources[component_id] = parents[0] if len(parents) == 1 else None
        parent_mappings = (configured_ccd, embedded_ccd)
    else:
        parent_mappings = (packaged_ccd, configured_ccd, embedded_ccd)
    for mapping in parent_mappings:
        for component_id, component in mapping.items():
            if component.parent_ids is None:
                continue
            # A present but empty/ambiguous CCD parent declaration is a
            # concrete fact that cannot select a proxy.  Store None to mask
            # COMMON_STANDARD_PARENT and any lower-precedence name fallback;
            # only parent_ids=None is allowed to inherit lower sources.
            parent_sources[component_id] = (
                component.parent_ids[0]
                if len(component.parent_ids) == 1
                else None
            )
    declared_polymer_chain_modalities = {
        chain.chain_id: chain.modality
        for chain in task.design_chains
    }
    # Antibody V chains are de novo; similarly named input chains remain
    # antigen context until the explicit antigen crop has been applied.
    antibody_source = task.antibody is not None
    if antibody_source:
        declared_polymer_chain_modalities = {}
    source = (
        None
        if task.input_path is None
        else read_structure(
            task.input_path,
            ccd_path=ccd_path,
            strict_type_bond_chains=strict_type_bond_chains,
            polymer_parent_sources=parent_sources,
            declared_polymer_chain_modalities=declared_polymer_chain_modalities,
        )
    )
    # Preserve the complete input's chain order even when a later crop removes
    # some source chains. Projection must not redefine this model-facing order.
    if task.crop is not None and source is None:
        raise DesignSampleError("task crop requires an input structure")
    allowed_condition = {"contig", "sequence", "backbone", "secondary_structure", "ligand", "bond"}
    unknown = set(condition).difference(allowed_condition)
    if unknown:
        raise DesignSampleError(f"unknown condition keys: {sorted(unknown)}")
    design_ids = {chain.chain_id for chain in task.design_chains}
    source_carrier_candidates: dict[int, _CarrierCandidate] = {}
    source_parent_by_residue: dict[int, str] = {}
    source_modified_residue_uids: set[int] = set()
    source_declared_bonds: tuple[tuple[int, int, int], ...] = ()
    forced_chain_break_keys: tuple[tuple[str, int, str], ...] = ()
    asym_order_source: struc.AtomArray | None = None
    if source is not None:
        source_declared_bonds = _source_declared_bonds(
            task.input_path,
            source,
            parent_sources=parent_sources,
            declared_polymer_chain_modalities=declared_polymer_chain_modalities,
        )
        for chain in task.design_chains:
            source_rows = np.where(source.chain_id == chain.chain_id)[0]
            if len(source_rows) and len(np.unique(source.asym_uid[source_rows])) > 1:
                raise DesignSampleError(
                    f"design chain {chain.chain_id!r} matches multiple TER segments "
                    "and is ambiguous"
                )
        source.is_design = (
            np.isin(source.chain_id, list(design_ids))
            if not antibody_source
            else np.zeros(len(source), dtype=bool)
        )
        asym_order_source = source.copy()
        if task.crop is not None:
            crop_result = project_source_crop(
                source,
                source_declared_bonds,
                task.crop,
                design_chain_ids=set() if antibody_source else design_ids,
                hotspot=task.hotspot,
                condition=condition,
            )
            source = crop_result.source
            source_declared_bonds = crop_result.declared_bonds
            forced_chain_break_keys = crop_result.forced_chain_breaks
        if antibody_source:
            collisions = set(source.chain_id) & design_ids
            if collisions:
                raise DesignSampleError(
                    f"antibody output chains collide with retained antigen chains {sorted(collisions)}; "
                    "set antibody.chains explicitly"
                )
        (
            source_carrier_candidates,
            source_parent_by_residue,
            source_modified_residue_uids,
        ) = _source_carrier_candidates(
            source,
            source_declared_bonds,
            embedded=embedded_ccd,
            configured=configured_ccd,
            packaged=packaged_ccd,
        )
        for residue_uid, parent in source_parent_by_residue.items():
            parent_sources[_source_residue_name(source, residue_uid).upper()] = parent
    return PreparedTaskSource(
        task=task,
        ccd_path=ccd_path,
        source=_clone_source_atom_array(source),
        asym_order_source=_clone_source_atom_array(asym_order_source),
        source_declared_bonds=tuple(source_declared_bonds),
        source_carrier_candidates=_freeze_carrier_candidates(source_carrier_candidates),
        source_parent_by_residue=MappingProxyType(dict(source_parent_by_residue)),
        source_modified_residue_uids=frozenset(source_modified_residue_uids),
        forced_chain_break_keys=tuple(forced_chain_break_keys),
        parent_sources=MappingProxyType(dict(parent_sources)),
        embedded_ccd=MappingProxyType(dict(embedded_ccd)),
        configured_ccd=MappingProxyType(dict(configured_ccd)),
        packaged_ccd=packaged_ccd,
        condition=MappingProxyType(deepcopy(dict(condition))),
        design_ids=frozenset(design_ids),
    )


def compile_request(
    request: ExpandedRequest,
    *,
    ccd_path: Path | None = None,
    prepared_source: PreparedTaskSource | None = None,
    trusted_ccd_components: Mapping[str, TrustedCCDComponent] | None = None,
    rng: Any | None = None,
) -> CompiledSample:
    """Compile one expanded request into the sole authoring AtomArray IR."""

    if request.task.format == "antibody":
        from ..antibody.compiler import compile_antibody_request
        return compile_antibody_request(request, compile_normal=compile_request,
            ccd_path=ccd_path, trusted_ccd_components=trusted_ccd_components, rng=rng)

    if rng is None:
        rng = np.random
    if prepared_source is None:
        prepared_source = prepare_task_source(
            request.task,
            ccd_path=ccd_path,
        )
        effective_ccd_path = prepared_source.ccd_path
    else:
        effective_ccd_path = _validate_prepared_task_source(
            prepared_source, request.task, ccd_path
        )
    ccd_path = effective_ccd_path
    condition = prepared_source.condition
    embedded_ccd = dict(prepared_source.embedded_ccd)
    configured_ccd = dict(prepared_source.configured_ccd)
    packaged_ccd = prepared_source.packaged_ccd
    parent_sources = dict(prepared_source.parent_sources)
    source = _clone_source_atom_array(prepared_source.source)
    asym_order_source = _clone_source_atom_array(prepared_source.asym_order_source)
    source_declared_bonds = tuple(prepared_source.source_declared_bonds)
    source_carrier_candidates = _clone_carrier_candidates(
        prepared_source.source_carrier_candidates
    )
    source_parent_by_residue = dict(prepared_source.source_parent_by_residue)
    source_modified_residue_uids = set(
        prepared_source.source_modified_residue_uids
    )
    forced_chain_break_keys = tuple(prepared_source.forced_chain_break_keys)
    design_ids = set(prepared_source.design_ids)
    parts: list[struc.AtomArray] = []
    if source is not None:
        context = _context_array(source, design_ids)
        if len(context):
            parts.append(context)
    ligand_modes: dict[str, str] = {}
    actual_lengths: dict[str, int] = {}
    motif_output_fields: dict[str, MotifOutputFields] = {}
    synthetic_carrier_specs: list[tuple[str, int, CCDComponent, str]] = []
    bracket_proxy_specs: list[tuple[str, int, str]] = []
    for chain in request.task.design_chains:
        if chain.modality == "ligand":
            part, mode = _compile_ligand_chain(
                request, chain, source, condition, trusted_ccd_components
            )
            ligand_modes[chain.chain_id] = mode
            actual_lengths[chain.chain_id] = len(part)
        else:
            (
                part,
                layout,
                motif_output,
                carrier_components_by_position,
                carrier_parents_by_position,
                proxy_parents_by_position,
            ) = _compile_polymer_chain(
                request,
                chain,
                source,
                condition,
                rng,
                source_parent_by_residue,
                source_modified_residue_uids,
                embedded_ccd,
                configured_ccd,
                packaged_ccd,
            )
            actual_lengths[chain.chain_id] = len(layout)
            if motif_output is not None:
                motif_output_fields[chain.chain_id] = motif_output
            synthetic_carrier_specs.extend(
                (
                    chain.chain_id,
                    position,
                    component,
                    carrier_parents_by_position[position],
                )
                for position, component in carrier_components_by_position.items()
            )
            bracket_proxy_specs.extend(
                (chain.chain_id, position, parent)
                for position, parent in proxy_parents_by_position.items()
            )
        part.is_design[:] = True
        parts.append(part)
    if not parts:
        raise DesignSampleError("compiled sample contains no atoms")
    atoms = concatenate_atom_arrays(parts)
    # De-novo bracket CCD residues need a source-origin sidecar so they can
    # reuse the exact atomized carrier/model-token lane.  Negative temporary
    # origins are disjoint from reader-assigned source AtomUIDs and are
    # replaced by _finalize_uids() below.
    for ordinal, (chain_id, position, component, resolved_parent) in enumerate(
        synthetic_carrier_specs, start=1
    ):
        rows = np.where(
            (atoms.chain_id == chain_id) & (atoms.res_id == int(position))
        )[0]
        if not len(rows):
            raise DesignSampleError(
                f"bracket CCD residue {chain_id}:{position} disappeared before carrier selection"
            )
        row_by_name = {str(atoms.atom_name[row]): int(row) for row in rows}
        if len(row_by_name) != len(rows):
            raise DesignSampleError(
                f"bracket CCD residue {chain_id}:{position} has duplicate atom IDs"
            )
        canonical_by_origin: dict[int, str] = {}
        for atom_index, row in enumerate(rows, start=1):
            origin_uid = -(ordinal * 1_000_000 + atom_index)
            atoms.atom_uid[row] = origin_uid
            canonical_by_origin[origin_uid] = str(atoms.atom_name[row])
        synthetic_key = -ordinal
        source_carrier_candidates[synthetic_key] = _CarrierCandidate(
            source_residue_uid=synthetic_key,
            component=component,
            canonical_by_origin_uid=canonical_by_origin,
            standard_parent=(
                resolved_parent
                if resolved_parent in _STANDARD_BY_MODALITY[component.polymer_modality]
                and resolved_parent
                != _UNKNOWN_RESIDUE[component.polymer_modality]
                else None
            ),
        )
    carrier_occurrences = _select_carrier_occurrences(
        atoms, source_carrier_candidates
    )
    automatic_leaving_rows = _automatic_carrier_leaving_rows(
        atoms, source, source_declared_bonds, carrier_occurrences, condition
    )
    precursor_atoms = atoms
    atoms, omitted_by_residue = _collect_omitted_atoms(
        atoms, condition, automatic_rows=automatic_leaving_rows,
        ligand_modes=ligand_modes,
    )
    if not any(key[0] in ligand_modes for key in omitted_by_residue):
        precursor_atoms = None
    # Slicing/removal preserves this source origin; _finalize_uids() is the
    # first and only operation below that deliberately overwrites atom_uid.
    origin_uid_by_row = np.asarray(atoms.atom_uid, dtype=np.int64).copy()
    _finalize_uids(atoms, source=asym_order_source)
    _validate_label_asym_bijection(atoms)
    (
        modified_polymer_carriers,
        carrier_internal_edges,
        carrier_internal_topology,
    ) = _finalize_carriers(
        atoms, origin_uid_by_row, source_carrier_candidates
    )
    carrier_residue_uids = frozenset(modified_polymer_carriers)
    model_parent_residue = _model_parent_by_residue(
        atoms,
        parent_sources,
        excluded_residue_uids=carrier_residue_uids,
    )
    for chain_id, position, proxy_parent in bracket_proxy_specs:
        rows = np.where(
            (atoms.chain_id == chain_id) & (atoms.res_id == int(position))
        )[0]
        if not len(rows):
            raise DesignSampleError(
                f"bracket proxy residue {chain_id}:{position} disappeared after UID finalization"
            )
        residue_uids = np.unique(atoms.residue_uid[rows])
        if len(residue_uids) != 1:
            raise DesignSampleError(
                f"bracket proxy residue {chain_id}:{position} has ambiguous residue UID"
            )
        residue_uid = int(residue_uids[0])
        existing_parent = model_parent_residue.get(residue_uid)
        if (
            existing_parent is not None
            and existing_parent != proxy_parent
            and existing_parent != _UNKNOWN_RESIDUE[str(atoms.mol_type[rows[0]])]
        ):
            raise DesignSampleError(
                f"bracket proxy parent conflicts at {chain_id}:{position}: "
                f"{existing_parent!r} versus {proxy_parent!r}"
            )
        model_parent_residue[residue_uid] = proxy_parent
    carrier_source_topology, carrier_source_edges = _inherit_carrier_source_edges(
        source,
        source_declared_bonds,
        atoms,
        origin_uid_by_row,
        modified_polymer_carriers,
    )
    _drop_carrier_incident_bonds(atoms, carrier_residue_uids)
    type_bond_source_topology, type_bond_source_edges = (
        _inherit_type_bond_source_edges(
            source,
            source_declared_bonds,
            atoms,
            origin_uid_by_row,
            ligand_modes,
        )
    )
    carrier_boundary_edges = _carrier_boundary_peptide_bonds(
        atoms,
        carrier_residue_uids,
        embedded=embedded_ccd,
        configured=configured_ccd,
        packaged=packaged_ccd,
    )
    boundary_uid_pairs = {
        tuple(
            sorted(
                (
                    int(atoms.atom_uid[row_i]),
                    int(atoms.atom_uid[row_j]),
                )
            )
        )
        for row_i, row_j, _ in carrier_boundary_edges
    }
    if boundary_uid_pairs:
        # A source CONECT/struct_conn edge at a strict peptide boundary is
        # still source-provenance chemistry, but its unknown-positive order is
        # resolved by the same CCD rule as the structural BondList edge.  Keep
        # one concrete source condition so canonical compilation cannot receive
        # both ``any`` and ``single`` facts for the same token pair.
        carrier_source_topology = [
            (
                TopologyCondition(
                    record.atom_uid_a,
                    record.atom_uid_b,
                    "single",
                    record.required_output,
                )
                if record.canonical_uids() in boundary_uid_pairs
                and record.order in _UNKNOWN_POSITIVE_ORDERS
                else record
            )
            for record in carrier_source_topology
        ]
    _merge_current_bonds(
        atoms,
        (
            *_polymer_template_bonds(
                atoms,
                excluded_residue_uids=carrier_residue_uids,
                design_only=request.task.crop is not None,
            ),
            *carrier_internal_edges,
            *carrier_source_edges,
            *type_bond_source_edges,
            *carrier_boundary_edges,
        ),
    )
    tokenize_atom_array(atoms, carrier_residue_uids=carrier_residue_uids)
    forced_chain_break_token_uids: tuple[int, ...] = ()
    if forced_chain_break_keys:
        token_uids: set[int] = set()
        for label_asym_id, res_id, ins_code in forced_chain_break_keys:
            rows = np.where(
                (atoms.label_asym_id == label_asym_id)
                & (atoms.res_id == int(res_id))
                & (atoms.ins_code == ins_code)
            )[0]
            if len(rows) == 0:
                raise DesignSampleError(
                    "crop forced chain-break residue is absent after compilation"
                )
            token_uids.update(int(value) for value in np.unique(atoms.token_uid[rows]))
        forced_chain_break_token_uids = tuple(sorted(token_uids))
    model_layout = build_model_token_layout(atoms, modified_polymer_carriers)
    omitted_atom_names_by_token_uid = _finalize_omitted_atom_sidecar(
        atoms, omitted_by_residue
    )
    _apply_hotspot(atoms, source, request.task.hotspot)
    topology_conditions, structural_edges = _explicit_topology(
        atoms,
        condition,
        ligand_modes,
    )
    explicit_topology = tuple(topology_conditions)
    inherited_topology, inherited_structural = _inherit_atom_motif_pl_edges(
        source, atoms, ligand_modes
    )
    _merge_current_bonds(atoms, (*structural_edges, *inherited_structural))
    topology_conditions = _merge_topology_conditions(
        atoms,
        (
            *carrier_internal_topology,
            *carrier_source_topology,
            *type_bond_source_topology,
            *topology_conditions,
            *inherited_topology,
            *_ligand_mode_topology(atoms, ligand_modes),
        ),
        model_layout,
    )
    coarse, ss_token, ss_pair = _secondary_structure(
        atoms,
        condition,
        modified_polymer_carriers,
    )
    _reject_carrier_secondary_structure(
        atoms,
        modified_polymer_carriers,
        coarse,
        ss_token,
        ss_pair,
    )
    edge_policies = _edge_policies(atoms, ligand_modes, explicit_topology)
    validate_atom_array(atoms, phase="tokenized")
    trusted_ligand_components: dict[int, TrustedCCDComponent] = {}
    type_bond_rows = (
        atoms.is_design
        & (atoms.mol_type == "ligand")
        & np.isin(
            atoms.chain_id,
            [
                chain_id
                for chain_id, mode in ligand_modes.items()
                if mode == "type_bond"
            ],
        )
    )
    identity_scan_atoms = atoms[~type_bond_rows]
    validated_ligand_identities = (
        validated_ligand_identities_from_ccd(
            identity_scan_atoms,
            ccd_path=ccd_path,
        )
        if np.any(identity_scan_atoms.mol_type == "ligand")
        else {}
    )
    for chain_id, mode in ligand_modes.items():
        if mode != "type_bond":
            continue
        rows = np.where(
            atoms.is_design
            & (atoms.mol_type == "ligand")
            & (atoms.chain_id == chain_id)
        )[0]
        molecule_uids = np.unique(atoms.molecule_uid[rows])
        component_names = {str(value).strip().upper() for value in atoms.res_name[rows]}
        if len(rows) == 0 or len(molecule_uids) != 1 or len(component_names) != 1:
            raise DesignSampleError(
                f"type_bond chain {chain_id!r} lacks one uniquely named hard component"
            )
        # Pure compiler/feature tests may omit the owner-local private asset.
        # Production runtime always injects its resolved release path; only
        # that downstream path requires canonical SMILES/private chemistry.
        if trusted_ccd_components is None:
            continue
        component_id = next(iter(component_names))
        component = trusted_ccd_components.get(component_id)
        if component is None:
            raise DesignSampleError(
                f"type_bond chain {chain_id!r} has no preflighted trusted CCD component"
            )
        omitted_names = omitted_by_residue.get(_residue_key(atoms, int(rows[0])), ())
        try:
            if omitted_names:
                precursor_rows = np.where(
                    precursor_atoms.is_design
                    & (precursor_atoms.mol_type == "ligand")
                    & (precursor_atoms.chain_id == chain_id)
                )[0]
                trusted_ccd_private_molecule(
                    precursor_atoms, precursor_rows, component_id, component
                )
                component = project_trusted_ccd_component(component, omitted_names)
            trusted_ccd_private_molecule(
                atoms,
                rows,
                component_id,
                component,
            )
        except (OSError, ValueError) as exc:
            raise DesignSampleError(
                f"type_bond chain {chain_id!r} trusted chemistry mismatch: {exc}"
            ) from exc
        trusted_ligand_components[int(molecule_uids[0])] = component
        if omitted_names:
            # This is a declared product fragment, not the complete precursor
            # CCD identity. The writer emits its actual graph under a local ID.
            continue
        component_uids = np.unique(atoms.component_uid[rows])
        if len(component_uids) != 1:
            raise DesignSampleError(
                f"type_bond chain {chain_id!r} lacks one output component identity"
            )
        signature = compute_ligand_component_signature(atoms, rows)
        if signature is None:
            raise DesignSampleError(
                f"type_bond chain {chain_id!r} lacks a complete concrete identity graph"
            )
        identity = build_validated_ligand_identity(
            component_id,
            signature[0],
            signature[1],
        )
        component_uid = int(component_uids[0])
        previous_identity = validated_ligand_identities.get(component_uid)
        if previous_identity is not None and previous_identity != identity:
            raise DesignSampleError(
                f"type_bond chain {chain_id!r} has conflicting CCD identity proofs"
            )
        validated_ligand_identities[component_uid] = identity
    forward, inverse = _affines(atoms)
    return CompiledSample(
        request=request,
        atoms=atoms,
        source_to_model_affine=forward,
        model_to_source_affine=inverse,
        topology_conditions=topology_conditions,
        edge_policies=edge_policies,
        model_parent_residue=model_parent_residue,
        modified_polymer_carriers=modified_polymer_carriers,
        coarse_ss=coarse,
        ss_token=ss_token,
        ss_pair=ss_pair,
        motif_output_fields=motif_output_fields,
        trusted_ligand_components=trusted_ligand_components,
        validated_ligand_identities=validated_ligand_identities,
        omitted_atom_names_by_token_uid=omitted_atom_names_by_token_uid,
        forced_chain_break_token_uids=forced_chain_break_token_uids,
    )


__all__ = ["PreparedTaskSource", "compile_request", "prepare_task_source"]
