"""Typed, offline CCD records for modified standalone polymers.

This module is deliberately independent from the ligand schema-2 topology asset and
from every training module.  It parses either the packaged polymer schema-v3 asset or
raw PDB CCD/mmCIF component definitions, merges sources without silently overwriting
conflicting chemistry, and exposes immutable primitive records for the compiler.

The module does not decide which leaving atoms are present in a concrete polymer
linkage and does not build a source-backed carrier.  Callers must make those contextual
decisions and may use :meth:`CCDComponent.resolve_atom_id` for literal-first alias
resolution.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping as ABCMapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
from partidesign.paths import resource_root
import re
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Sequence

import biotite.structure as struc
from biotite.structure.io import pdbx

from ..core.records import DesignInputError
from src.common.chemistry.ccd import (
    CCDResourceError,
    ccd_polymer_component_ids,
    ccd_polymer_components,
    clear_ccd_caches as clear_shared_ccd_caches,
)


CCD_SCHEMA_VERSION = 3
PACKAGED_POLYMER_CCD_PATH = (
    resource_root() / "configs" / "resources" / "chemistry" / "ccd.json.gz"
)
PACKAGED_CCD_SOURCE_NAME = "ccd.json.gz"
PACKAGED_CCD_SOURCE_SIZE = 0
UNIFIED_CCD_SCHEMA_VERSION = 4
MAX_CCD_V3_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_CCD_V3_UNCOMPRESSED_BYTES = 256 * 1024 * 1024

CCDSourceKind = Literal["embedded", "configured", "packaged"]
CCDResolutionStatus = Literal["resolved", "missing", "unusable"]
PolymerModality = Literal["protein", "rna", "dna"]
BondOrder = Literal["single", "double", "triple", "aromatic", "any"]

_POLYMER_MODALITIES = frozenset({"protein", "rna", "dna"})
_HYDROGEN_ELEMENTS = frozenset({"H", "D", "T"})
_MISSING = frozenset({"", ".", "?"})
_ORDER_ALIASES: dict[str, BondOrder] = {
    "SING": "single",
    "SINGLE": "single",
    "DOUB": "double",
    "DOUBLE": "double",
    "TRIP": "triple",
    "TRIPLE": "triple",
    "AROM": "aromatic",
    "AROMATIC": "aromatic",
}
_ORDER_TO_BIOTITE = {
    "single": int(struc.BondType.SINGLE),
    "double": int(struc.BondType.DOUBLE),
    "triple": int(struc.BondType.TRIPLE),
    "aromatic": int(struc.BondType.AROMATIC),
    "any": int(struc.BondType.ANY),
}


@dataclass(frozen=True, slots=True)
class CCDProvenance:
    """Source location of one CCD component record."""

    kind: CCDSourceKind
    location: str
    source_name: str
    source_size: int


@dataclass(frozen=True, slots=True)
class CCDAtom:
    """One canonical heavy atom and its concrete CCD facts."""

    atom_id: str
    aliases: tuple[str, ...]
    element: str | None
    formal_charge: int | None
    leaving: bool | None
    ideal_coord: tuple[float, float, float] | None


@dataclass(frozen=True, slots=True)
class CCDBond:
    """One exact, canonicalized heavy-heavy CCD bond."""

    atom_id_1: str
    atom_id_2: str
    order: BondOrder

    @property
    def biotite_order(self) -> int:
        return _ORDER_TO_BIOTITE[self.order]


@dataclass(frozen=True, slots=True)
class CCDComponent:
    """Typed CCD facts for one component.

    ``parent_ids is None`` means the source omitted the parent declaration;
    ``parent_ids == ()`` means the source concretely declared no parent.  The table
    presence flags make the same distinction for empty versus absent atom/bond tables
    when parsing an embedded input CIF.
    """

    component_id: str
    raw_chem_type: str | None
    polymer_modality: PolymerModality | None
    parent_ids: tuple[str, ...] | None
    component_formal_charge: int | None
    atoms: tuple[CCDAtom, ...]
    bonds: tuple[CCDBond, ...]
    provenance: tuple[CCDProvenance, ...]
    atom_table_present: bool = True
    bond_table_present: bool = True
    validation_issues: tuple[str, ...] = ()

    def atom_by_id(self, atom_id: str) -> CCDAtom | None:
        """Return a literal canonical atom, or ``None`` when it is absent."""

        name = str(atom_id).strip()
        return next((atom for atom in self.atoms if atom.atom_id == name), None)

    def resolve_atom_id(self, source_name: str) -> str | None:
        """Resolve an observed atom name, preferring a literal canonical ID.

        An alias is accepted only when it identifies exactly one atom.  This prevents
        a permissive alias table from silently choosing chemistry.
        """

        name = str(source_name).strip()
        if not name:
            return None
        literal = self.atom_by_id(name)
        if literal is not None:
            return literal.atom_id
        matches = tuple(
            atom.atom_id for atom in self.atoms if name in atom.aliases
        )
        if not matches:
            return None
        if len(matches) != 1:
            raise DesignInputError(
                f"CCD component {self.component_id!r} alias {name!r} is ambiguous: "
                f"{sorted(matches)}"
            )
        return matches[0]

    def has_full_finite_ideal_coordinates(
        self, atom_ids: Iterable[str] | None = None
    ) -> bool:
        """Return whether every explicitly selected atom has a finite ideal position.

        No leaving-atom policy is applied here.  A future ``[CCD]`` caller must pass
        the exact contextually retained atom IDs.
        """

        selected = (
            tuple(atom.atom_id for atom in self.atoms)
            if atom_ids is None
            else tuple(str(value).strip() for value in atom_ids)
        )
        if not selected:
            return False
        by_id = {atom.atom_id: atom for atom in self.atoms}
        if len(selected) != len(set(selected)) or any(name not in by_id for name in selected):
            return False
        return all(
            by_id[name].ideal_coord is not None
            and all(math.isfinite(value) for value in by_id[name].ideal_coord or ())
            for name in selected
        )


@dataclass(frozen=True, slots=True)
class CCDResolution:
    """Result of resolving a component across embedded/configured/packaged sources."""

    status: CCDResolutionStatus
    component_id: str
    component: CCDComponent | None
    provenance: tuple[CCDProvenance, ...]
    reason: str | None = None


@dataclass(slots=True)
class _MutableComponent:
    component_id: str
    raw_chem_type: str | None = None
    parent_ids: tuple[str, ...] | None = None
    component_formal_charge: int | None = None
    atoms: dict[str, CCDAtom] | None = None
    bonds: list[tuple[str, str, BondOrder]] | None = None
    atom_table_present: bool = False
    bond_table_present: bool = False
    validation_issues: list[str] | None = None

    def __post_init__(self) -> None:
        if self.atoms is None:
            self.atoms = {}
        if self.bonds is None:
            self.bonds = []
        if self.validation_issues is None:
            self.validation_issues = []


def normalize_polymer_modality(raw_chem_type: object) -> PolymerModality | None:
    """Map a PDB CCD chemical type to the standalone polymer modality.

    Saccharide/carbohydrate types are rejected before peptide-like matching and remain
    ligands even when a task declares a polymer design chain.
    """

    text = " ".join(str(raw_chem_type).strip().upper().split())
    if text in _MISSING:
        return None
    if "SACCHARIDE" in text or "CARBOHYDRATE" in text:
        return None
    if "PEPTIDE" in text or "PROTEIN" in text:
        return "protein"
    if "DNA" in text or "DEOXYRIBO" in text:
        return "dna"
    if "RNA" in text or "RIBONUCLE" in text:
        return "rna"
    return None


def _is_carbohydrate(raw_chem_type: str | None) -> bool:
    text = "" if raw_chem_type is None else raw_chem_type.upper()
    return "SACCHARIDE" in text or "CARBOHYDRATE" in text


def _normal_component_id(value: object, *, where: str) -> str:
    component_id = str(value).strip().upper()
    if component_id in _MISSING:
        raise DesignInputError(f"{where} has a missing CCD component ID")
    return component_id


def _normal_atom_id(value: object, *, component_id: str) -> str:
    atom_id = str(value).strip()
    if atom_id in _MISSING:
        raise DesignInputError(
            f"CCD component {component_id!r} has a missing canonical atom ID"
        )
    return atom_id


def _normal_element(value: object) -> str | None:
    text = str(value).strip().upper()
    if text in _MISSING:
        return None
    if len(text) > 3 or not text.isalpha():
        raise DesignInputError(f"invalid CCD element {value!r}")
    return text


def _normal_optional_text(value: object) -> str | None:
    text = str(value).strip()
    return None if text in _MISSING else text


def _normal_int(value: object, *, field: str) -> int | None:
    text = str(value).strip()
    if text in _MISSING:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise DesignInputError(f"{field} is not an integer: {value!r}") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise DesignInputError(f"{field} is not an integer: {value!r}")
    return int(number)


def _normal_flag(value: object, *, field: str) -> bool | None:
    text = str(value).strip().upper()
    if text in _MISSING:
        return None
    if text == "Y":
        return True
    if text == "N":
        return False
    raise DesignInputError(f"{field} must be Y or N, got {value!r}")


def _normal_ideal_coord(
    values: Sequence[object], *, field: str
) -> tuple[float, float, float] | None:
    text = tuple(str(value).strip() for value in values)
    present = tuple(value not in _MISSING for value in text)
    if not any(present):
        return None
    if not all(present):
        return None
    try:
        result = tuple(float(value) for value in text)
    except ValueError as exc:
        raise DesignInputError(f"{field} is not numeric") from exc
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise DesignInputError(f"{field} is not finite")
    return result  # type: ignore[return-value]


def _normal_parent_ids(value: object) -> tuple[str, ...] | None:
    text = str(value).strip()
    if text in _MISSING:
        return None
    values = {
        _normal_component_id(token, where="chem_comp parent")
        for token in re.split(r"[,;\s]+", text)
        if token
    }
    return tuple(sorted(values))


def _normal_order(value: object, *, allow_unknown: bool = True) -> BondOrder:
    text = str(value).strip().upper()
    if allow_unknown and text in _MISSING:
        return "any"
    order = _ORDER_ALIASES.get(text)
    if order is None:
        raise DesignInputError(f"unsupported exact CCD bond order {value!r}")
    return order


def _column(category: Any, name: str, length: int, default: str = "") -> Sequence[str]:
    if name not in category:
        return [default] * length
    return category[name].as_array(str, default)


def _assign_scalar(
    builder: _MutableComponent,
    attribute: str,
    value: object,
    *,
    field: str,
) -> None:
    if value is None:
        return
    previous = getattr(builder, attribute)
    if previous is not None and previous != value:
        raise DesignInputError(
            f"CCD component {builder.component_id!r} has conflicting {field}: "
            f"{previous!r} versus {value!r}"
        )
    setattr(builder, attribute, value)


def _merge_atom_facts(
    component_id: str, previous: CCDAtom, current: CCDAtom
) -> CCDAtom:
    def scalar(field: str, high: object, low: object) -> object:
        if high is not None and low is not None and high != low:
            raise DesignInputError(
                f"CCD component {component_id!r} atom {previous.atom_id!r} "
                f"has conflicting {field}: {low!r} versus {high!r}"
            )
        return high if high is not None else low

    high_coord = current.ideal_coord
    low_coord = previous.ideal_coord
    if high_coord is not None and low_coord is not None and any(
        abs(left - right) > 1e-6
        for left, right in zip(high_coord, low_coord, strict=True)
    ):
        raise DesignInputError(
            f"CCD component {component_id!r} atom {previous.atom_id!r} "
            "has conflicting ideal coordinates"
        )
    return CCDAtom(
        atom_id=previous.atom_id,
        aliases=tuple(sorted(set(previous.aliases) | set(current.aliases))),
        element=scalar("element", current.element, previous.element),  # type: ignore[arg-type]
        formal_charge=scalar(
            "formal charge", current.formal_charge, previous.formal_charge
        ),  # type: ignore[arg-type]
        leaving=scalar("leaving flag", current.leaving, previous.leaving),  # type: ignore[arg-type]
        ideal_coord=high_coord if high_coord is not None else low_coord,
    )


def _parse_cif(
    path: Path,
    *,
    source_kind: CCDSourceKind,
    polymer_only: bool,
    complete_source: bool,
) -> Mapping[str, CCDComponent]:
    try:
        cif = pdbx.CIFFile.read(path)
    except Exception as exc:
        raise DesignInputError(f"cannot parse CCD CIF {path}: {exc}") from exc
    source_size = path.stat().st_size
    source_has_atom_table = any("chem_comp_atom" in cif[name] for name in cif.keys())
    provenance = CCDProvenance(
        kind=source_kind,
        location=str(path),
        source_name=path.name,
        source_size=source_size,
    )
    builders: dict[str, _MutableComponent] = {}
    spellings: dict[str, str] = {}

    def component(value: object, *, where: str) -> _MutableComponent:
        spelling = str(value).strip()
        component_id = _normal_component_id(spelling, where=where)
        previous_spelling = spellings.get(component_id)
        if previous_spelling is not None and previous_spelling != spelling:
            raise DesignInputError(
                f"CCD component ID has case-colliding spellings "
                f"{previous_spelling!r} and {spelling!r}"
            )
        spellings[component_id] = spelling
        return builders.setdefault(component_id, _MutableComponent(component_id))

    for block_name in cif.keys():
        block = cif[block_name]
        selected_ids: set[str] | None = None
        if "chem_comp" in block:
            category = block["chem_comp"]
            count = category.row_count
            ids = _column(category, "id", count, str(block_name))
            raw_types = _column(category, "type", count)
            parents = _column(category, "mon_nstd_parent_comp_id", count)
            charges = _column(category, "pdbx_formal_charge", count)
            selected_ids = set()
            for raw_id, raw_type, raw_parent, raw_charge in zip(
                ids, raw_types, parents, charges, strict=True
            ):
                component_id = _normal_component_id(raw_id, where="chem_comp")
                type_text = _normal_optional_text(raw_type)
                if polymer_only and normalize_polymer_modality(type_text) is None:
                    continue
                selected_ids.add(component_id)
                builder = component(raw_id, where="chem_comp")
                _assign_scalar(
                    builder, "raw_chem_type", type_text, field="chemical type"
                )
                parent_ids = _normal_parent_ids(raw_parent)
                if parent_ids is None and complete_source:
                    parent_ids = ()
                _assign_scalar(
                    builder,
                    "parent_ids",
                    parent_ids,
                    field="parent IDs",
                )
                _assign_scalar(
                    builder,
                    "component_formal_charge",
                    _normal_int(
                        raw_charge,
                        field=f"CCD component {component_id!r} formal charge",
                    ),
                    field="component formal charge",
                )
        elif polymer_only:
            selected_ids = set()

        if "chem_comp_atom" in block:
            category = block["chem_comp_atom"]
            count = category.row_count
            component_ids = _column(category, "comp_id", count, str(block_name))
            atom_ids = _column(category, "atom_id", count)
            alt_ids = _column(category, "alt_atom_id", count)
            component_atom_ids = _column(category, "pdbx_component_atom_id", count)
            elements = _column(category, "type_symbol", count)
            charges = _column(category, "charge", count)
            leaving = _column(category, "pdbx_leaving_atom_flag", count)
            ideal_x = _column(category, "pdbx_model_Cartn_x_ideal", count)
            ideal_y = _column(category, "pdbx_model_Cartn_y_ideal", count)
            ideal_z = _column(category, "pdbx_model_Cartn_z_ideal", count)
            for row in range(count):
                component_id = _normal_component_id(
                    component_ids[row], where="chem_comp_atom"
                )
                if selected_ids is not None and component_id not in selected_ids:
                    continue
                builder = component(component_ids[row], where="chem_comp_atom")
                builder.atom_table_present = True
                atom_id = _normal_atom_id(atom_ids[row], component_id=component_id)
                aliases = tuple(
                    sorted(
                        {
                            alias
                            for alias in (
                                str(alt_ids[row]).strip(),
                                str(component_atom_ids[row]).strip(),
                            )
                            if alias not in _MISSING and alias != atom_id
                        }
                    )
                )
                atom = CCDAtom(
                    atom_id=atom_id,
                    aliases=aliases,
                    element=_normal_element(elements[row]),
                    formal_charge=_normal_int(
                        charges[row],
                        field=f"CCD component {component_id!r} atom {atom_id!r} charge",
                    ),
                    leaving=_normal_flag(
                        leaving[row],
                        field=f"CCD component {component_id!r} atom {atom_id!r} leaving flag",
                    ),
                    ideal_coord=_normal_ideal_coord(
                        (ideal_x[row], ideal_y[row], ideal_z[row]),
                        field=f"CCD component {component_id!r} atom {atom_id!r} ideal coordinate",
                    ),
                )
                previous = builder.atoms.get(atom_id)
                builder.atoms[atom_id] = (
                    atom
                    if previous is None
                    else _merge_atom_facts(component_id, previous, atom)
                )

        if "chem_comp_bond" in block:
            category = block["chem_comp_bond"]
            count = category.row_count
            component_ids = _column(category, "comp_id", count, str(block_name))
            atom_1 = _column(category, "atom_id_1", count)
            atom_2 = _column(category, "atom_id_2", count)
            orders = _column(category, "value_order", count)
            for raw_component, raw_left, raw_right, raw_order in zip(
                component_ids, atom_1, atom_2, orders, strict=True
            ):
                component_id = _normal_component_id(
                    raw_component, where="chem_comp_bond"
                )
                if selected_ids is not None and component_id not in selected_ids:
                    continue
                builder = component(raw_component, where="chem_comp_bond")
                builder.bond_table_present = True
                left = _normal_atom_id(raw_left, component_id=component_id)
                right = _normal_atom_id(raw_right, component_id=component_id)
                if left == right:
                    raise DesignInputError(
                        f"CCD component {component_id!r} has a self bond at {left!r}"
                    )
                builder.bonds.append((left, right, _normal_order(raw_order)))

    result: dict[str, CCDComponent] = {}
    for component_id in sorted(builders):
        builder = builders[component_id]
        all_atoms = dict(builder.atoms)
        heavy_atoms = {
            atom_id: atom
            for atom_id, atom in all_atoms.items()
            if atom.element not in _HYDROGEN_ELEMENTS
        }
        issues = list(builder.validation_issues)
        if (
            builder.component_formal_charge is not None
            and all_atoms
            and all(atom.formal_charge is not None for atom in all_atoms.values())
        ):
            atom_charge = sum(
                int(atom.formal_charge)
                for atom in all_atoms.values()
                if atom.formal_charge is not None
            )
            if atom_charge != builder.component_formal_charge:
                issues.append(
                    f"component formal charge {builder.component_formal_charge} "
                    f"does not equal atom charge sum {atom_charge}"
                )

        bonds_by_pair: dict[tuple[str, str], BondOrder] = {}
        # Structure CIFs may declare bond facts without any component atom table.
        # Preserve these references for authoritative-source merge; final
        # completeness validation still rejects any unresolved endpoint.
        defer_endpoints = (
            source_kind == "embedded"
            and not complete_source
            and not source_has_atom_table
            and not builder.atom_table_present
        )
        for raw_left, raw_right, order in builder.bonds:
            if not defer_endpoints and (raw_left not in all_atoms or raw_right not in all_atoms):
                issues.append(
                    f"bond endpoint is absent from atom table: {raw_left}-{raw_right}"
                )
                continue
            if not defer_endpoints and (raw_left not in heavy_atoms or raw_right not in heavy_atoms):
                continue
            pair = tuple(sorted((raw_left, raw_right)))
            previous = bonds_by_pair.get(pair)
            if previous is not None and previous != order:
                raise DesignInputError(
                    f"CCD component {component_id!r} bond {pair} has conflicting orders"
                )
            if previous is not None:
                raise DesignInputError(
                    f"CCD component {component_id!r} has duplicate bond {pair}"
                )
            bonds_by_pair[pair] = order

        raw_type = builder.raw_chem_type
        modality = normalize_polymer_modality(raw_type)
        if polymer_only and modality is None:
            continue
        result[component_id] = CCDComponent(
            component_id=component_id,
            raw_chem_type=raw_type,
            polymer_modality=modality,
            parent_ids=builder.parent_ids,
            component_formal_charge=builder.component_formal_charge,
            atoms=tuple(heavy_atoms[name] for name in sorted(heavy_atoms)),
            bonds=tuple(
                CCDBond(left, right, bonds_by_pair[(left, right)])
                for left, right in sorted(bonds_by_pair)
            ),
            provenance=(provenance,),
            atom_table_present=builder.atom_table_present or complete_source,
            bond_table_present=builder.bond_table_present or complete_source,
            validation_issues=tuple(sorted(set(issues))),
        )
    return MappingProxyType(result)


def _expect_mapping(value: object, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DesignInputError(f"{where} must be a mapping")
    return value


def _parse_json_atom(component_id: str, value: object) -> CCDAtom:
    row = _expect_mapping(value, where=f"CCD component {component_id!r} atom")
    expected = {
        "atom_id",
        "aliases",
        "element",
        "formal_charge",
        "leaving",
        "ideal_coord",
    }
    if set(row) != expected:
        raise DesignInputError(
            f"CCD component {component_id!r} atom must contain exactly {sorted(expected)}"
        )
    atom_id = _normal_atom_id(row["atom_id"], component_id=component_id)
    raw_aliases = row["aliases"]
    if not isinstance(raw_aliases, list) or not all(
        isinstance(alias, str) and alias.strip() not in _MISSING
        for alias in raw_aliases
    ):
        raise DesignInputError(f"CCD component {component_id!r} has invalid aliases")
    aliases = tuple(sorted({alias.strip() for alias in raw_aliases if alias.strip() != atom_id}))
    raw_charge = row["formal_charge"]
    if raw_charge is not None and (
        isinstance(raw_charge, bool) or not isinstance(raw_charge, int)
    ):
        raise DesignInputError(f"CCD component {component_id!r} has invalid atom charge")
    raw_leaving = row["leaving"]
    if raw_leaving is not None and not isinstance(raw_leaving, bool):
        raise DesignInputError(f"CCD component {component_id!r} has invalid leaving flag")
    raw_coord = row["ideal_coord"]
    if raw_coord is None:
        ideal_coord = None
    elif (
        isinstance(raw_coord, list)
        and len(raw_coord) == 3
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in raw_coord
        )
    ):
        ideal_coord = tuple(float(item) for item in raw_coord)
    else:
        raise DesignInputError(f"CCD component {component_id!r} has invalid ideal coordinate")
    return CCDAtom(
        atom_id=atom_id,
        aliases=aliases,
        element=_normal_element(row["element"]),
        formal_charge=raw_charge,
        leaving=raw_leaving,
        ideal_coord=ideal_coord,  # type: ignore[arg-type]
    )


def _load_v3(path: Path, *, source_kind: CCDSourceKind) -> Mapping[str, CCDComponent]:
    try:
        compressed_size = path.stat().st_size
        if compressed_size > MAX_CCD_V3_COMPRESSED_BYTES:
            raise DesignInputError(
                f"polymer CCD v3 exceeds {MAX_CCD_V3_COMPRESSED_BYTES} compressed bytes"
            )
        with gzip.open(path, "rb") as stream:
            encoded = stream.read(MAX_CCD_V3_UNCOMPRESSED_BYTES + 1)
        if len(encoded) > MAX_CCD_V3_UNCOMPRESSED_BYTES:
            raise DesignInputError(
                f"polymer CCD v3 exceeds {MAX_CCD_V3_UNCOMPRESSED_BYTES} uncompressed bytes"
            )
        payload = json.loads(encoded.decode("utf-8"))
    except DesignInputError:
        raise
    except Exception as exc:
        raise DesignInputError(f"cannot parse polymer CCD v3 {path}: {exc}") from exc
    root = _expect_mapping(payload, where="polymer CCD v3 root")
    if set(root) != {"schema_version", "source", "components"}:
        raise DesignInputError(
            "polymer CCD v3 must contain exactly schema_version, source, and components"
        )
    if root["schema_version"] != CCD_SCHEMA_VERSION:
        raise DesignInputError("unsupported polymer CCD schema")
    source = _expect_mapping(root["source"], where="polymer CCD v3 source")
    if not {"filename", "size"}.issubset(source):
        raise DesignInputError("invalid polymer CCD v3 source provenance")
    filename = source["filename"]
    size = source["size"]
    if (
        not isinstance(filename, str)
        or not filename
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise DesignInputError("invalid polymer CCD v3 source provenance")
    provenance = CCDProvenance(
        kind=source_kind,
        location=str(path),
        source_name=filename,
        source_size=size,
    )
    raw_components = _expect_mapping(root["components"], where="polymer CCD v3 components")
    result: dict[str, CCDComponent] = {}
    for raw_component_id, raw_entry in raw_components.items():
        if not isinstance(raw_component_id, str):
            raise DesignInputError("polymer CCD v3 component IDs must be strings")
        component_id = _normal_component_id(raw_component_id, where="polymer CCD v3")
        if component_id in result:
            raise DesignInputError(
                f"duplicate normalized polymer CCD component ID {component_id!r}"
            )
        entry = _expect_mapping(
            raw_entry, where=f"polymer CCD v3 component {component_id!r}"
        )
        expected = {
            "raw_chem_type",
            "polymer_modality",
            "parent_ids",
            "component_formal_charge",
            "atoms",
            "bonds",
            "validation_issues",
        }
        if set(entry) != expected:
            raise DesignInputError(
                f"polymer CCD v3 component {component_id!r} must contain exactly "
                f"{sorted(expected)}"
            )
        raw_type = entry["raw_chem_type"]
        if not isinstance(raw_type, str) or not raw_type.strip():
            raise DesignInputError(f"polymer CCD v3 component {component_id!r} lacks type")
        modality = entry["polymer_modality"]
        if modality not in _POLYMER_MODALITIES or modality != normalize_polymer_modality(raw_type):
            raise DesignInputError(
                f"polymer CCD v3 component {component_id!r} has invalid modality"
            )
        if _is_carbohydrate(raw_type):
            raise DesignInputError(
                f"polymer CCD v3 cannot contain carbohydrate {component_id!r}"
            )
        raw_parents = entry["parent_ids"]
        if not isinstance(raw_parents, list) or not all(
            isinstance(parent, str) and parent.strip() not in _MISSING
            for parent in raw_parents
        ):
            raise DesignInputError(
                f"polymer CCD v3 component {component_id!r} has invalid parent IDs"
            )
        parent_ids = tuple(
            sorted(
                {
                    _normal_component_id(parent, where="polymer CCD v3 parent")
                    for parent in raw_parents
                }
            )
        )
        component_charge = entry["component_formal_charge"]
        if component_charge is not None and (
            isinstance(component_charge, bool) or not isinstance(component_charge, int)
        ):
            raise DesignInputError(
                f"polymer CCD v3 component {component_id!r} has invalid formal charge"
            )
        raw_atoms = entry["atoms"]
        if not isinstance(raw_atoms, list):
            raise DesignInputError(f"polymer CCD v3 component {component_id!r} has invalid atoms")
        atoms = tuple(_parse_json_atom(component_id, value) for value in raw_atoms)
        if tuple(atom.atom_id for atom in atoms) != tuple(
            sorted(atom.atom_id for atom in atoms)
        ) or len({atom.atom_id for atom in atoms}) != len(atoms):
            raise DesignInputError(
                f"polymer CCD v3 component {component_id!r} atoms are not unique and sorted"
            )
        raw_bonds = entry["bonds"]
        if not isinstance(raw_bonds, list):
            raise DesignInputError(f"polymer CCD v3 component {component_id!r} has invalid bonds")
        bonds: list[CCDBond] = []
        seen_pairs: set[tuple[str, str]] = set()
        atom_ids = {atom.atom_id for atom in atoms}
        for raw_bond in raw_bonds:
            if not isinstance(raw_bond, list) or len(raw_bond) != 3:
                raise DesignInputError(
                    f"polymer CCD v3 component {component_id!r} has invalid bond"
                )
            left = _normal_atom_id(raw_bond[0], component_id=component_id)
            right = _normal_atom_id(raw_bond[1], component_id=component_id)
            pair = tuple(sorted((left, right)))
            if left == right or left not in atom_ids or right not in atom_ids or pair in seen_pairs:
                raise DesignInputError(
                    f"polymer CCD v3 component {component_id!r} has invalid bond {pair}"
                )
            seen_pairs.add(pair)
            bonds.append(
                CCDBond(
                    pair[0],
                    pair[1],
                    _normal_order(raw_bond[2], allow_unknown=False),
                )
            )
        if tuple((bond.atom_id_1, bond.atom_id_2) for bond in bonds) != tuple(
            sorted((bond.atom_id_1, bond.atom_id_2) for bond in bonds)
        ):
            raise DesignInputError(
                f"polymer CCD v3 component {component_id!r} bonds are not sorted"
            )
        raw_issues = entry["validation_issues"]
        if not isinstance(raw_issues, list) or not all(
            isinstance(issue, str) and issue for issue in raw_issues
        ):
            raise DesignInputError(
                f"polymer CCD v3 component {component_id!r} has invalid validation issues"
            )
        result[component_id] = CCDComponent(
            component_id=component_id,
            raw_chem_type=raw_type.strip(),
            polymer_modality=modality,
            parent_ids=parent_ids,
            component_formal_charge=component_charge,
            atoms=atoms,
            bonds=tuple(bonds),
            provenance=(provenance,),
            validation_issues=tuple(sorted(set(raw_issues))),
        )
    return MappingProxyType(result)


class _LazyTypedComponents(ABCMapping[str, CCDComponent]):
    """Typed polymer projection that validates and materializes requested IDs."""

    __slots__ = ("_records", "_provenance", "_cache")

    def __init__(self, records: Mapping[str, Mapping[str, Any]], provenance: CCDProvenance) -> None:
        self._records = records
        self._provenance = provenance
        self._cache: dict[str, CCDComponent] = {}

    def __getitem__(self, component_id: str) -> CCDComponent:
        normalized = str(component_id).strip().upper()
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached
        try:
            entry = self._records[normalized]
        except KeyError:
            raise KeyError(normalized) from None
        value = _parse_unified_entry(normalized, entry, self._provenance)
        self._cache[normalized] = value
        return value

    def __iter__(self) -> Iterator[str]:
        # The shared view's natural iterator drains the component stream so that
        # raw parity audits can enumerate every record.  Typed callers commonly
        # enumerate IDs only for parent/modality hints; use the leading metadata
        # index there and leave atom/bond records lazy for keyed lookups.
        try:
            return iter(ccd_polymer_component_ids(self._provenance.location))
        except (CCDResourceError, OSError, ValueError):
            return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __reduce__(self):
        # The shared record view owns a live gzip cursor and lock.  Reopen the
        # typed mapping from its provenance in the child instead of pickling
        # either the stream or materialized component cache.
        return (
            reload_typed_ccd_components,
            (self._provenance.location, self._provenance.kind),
        )


def _parse_unified_entry(
    component_id: str,
    entry: Mapping[str, Any],
    provenance: CCDProvenance,
) -> CCDComponent:
    if not isinstance(entry, Mapping):
        raise DesignInputError(f"merged CCD polymer {component_id!r} must be a mapping")
    expected = {
        "raw_chem_type",
        "polymer_modality",
        "parent_ids",
        "component_formal_charge",
        "atoms",
        "bonds",
        "validation_issues",
    }
    if set(entry) != expected:
        raise DesignInputError(
            f"merged CCD polymer {component_id!r} must contain exactly {sorted(expected)}"
        )
    raw_type = entry["raw_chem_type"]
    modality = entry["polymer_modality"]
    if not isinstance(raw_type, str) or modality not in _POLYMER_MODALITIES:
        raise DesignInputError(f"merged CCD polymer {component_id!r} has invalid type")
    if modality != normalize_polymer_modality(raw_type) or _is_carbohydrate(raw_type):
        raise DesignInputError(f"merged CCD polymer {component_id!r} has invalid modality")
    raw_parents = entry["parent_ids"]
    if not isinstance(raw_parents, list) or not all(
        isinstance(parent, str) and parent.strip() not in _MISSING
        for parent in raw_parents
    ):
        raise DesignInputError(f"merged CCD polymer {component_id!r} has invalid parents")
    parent_ids = tuple(sorted({
        _normal_component_id(parent, where="merged CCD parent")
        for parent in raw_parents
    }))
    charge = entry["component_formal_charge"]
    if charge is not None and (isinstance(charge, bool) or not isinstance(charge, int)):
        raise DesignInputError(f"merged CCD polymer {component_id!r} has invalid charge")
    raw_atoms = entry["atoms"]
    if not isinstance(raw_atoms, list):
        raise DesignInputError(f"merged CCD polymer {component_id!r} has invalid atoms")
    atoms = tuple(_parse_json_atom(component_id, atom) for atom in raw_atoms)
    if tuple(atom.atom_id for atom in atoms) != tuple(sorted(atom.atom_id for atom in atoms)):
        raise DesignInputError(f"merged CCD polymer {component_id!r} atoms are not sorted")
    raw_bonds = entry["bonds"]
    if not isinstance(raw_bonds, list):
        raise DesignInputError(f"merged CCD polymer {component_id!r} has invalid bonds")
    atom_ids = {atom.atom_id for atom in atoms}
    bonds: list[CCDBond] = []
    seen: set[tuple[str, str]] = set()
    for row in raw_bonds:
        if not isinstance(row, list) or len(row) != 3:
            raise DesignInputError(f"merged CCD polymer {component_id!r} has invalid bond")
        left = _normal_atom_id(row[0], component_id=component_id)
        right = _normal_atom_id(row[1], component_id=component_id)
        pair = tuple(sorted((left, right)))
        if left == right or left not in atom_ids or right not in atom_ids or pair in seen:
            raise DesignInputError(f"merged CCD polymer {component_id!r} has invalid bond {pair}")
        seen.add(pair)
        bonds.append(CCDBond(pair[0], pair[1], _normal_order(row[2], allow_unknown=False)))
    if tuple((bond.atom_id_1, bond.atom_id_2) for bond in bonds) != tuple(
        sorted((bond.atom_id_1, bond.atom_id_2) for bond in bonds)
    ):
        raise DesignInputError(f"merged CCD polymer {component_id!r} bonds are not sorted")
    issues = entry["validation_issues"]
    if not isinstance(issues, list) or not all(isinstance(issue, str) and issue for issue in issues):
        raise DesignInputError(f"merged CCD polymer {component_id!r} has invalid issues")
    return CCDComponent(
        component_id=component_id,
        raw_chem_type=raw_type.strip(),
        polymer_modality=modality,
        parent_ids=parent_ids,
        component_formal_charge=charge,
        atoms=atoms,
        bonds=tuple(bonds),
        provenance=(provenance,),
        validation_issues=tuple(sorted(set(issues))),
    )

def _load_unified(path: Path, *, source_kind: CCDSourceKind) -> Mapping[str, CCDComponent]:
    """Project the unified schema-4 polymer view into the historical typed API."""
    try:
        records = ccd_polymer_components(path)
    except CCDResourceError as exc:
        raise DesignInputError(f"cannot parse merged CCD {path}: {exc}") from exc
    provenance = CCDProvenance(
        kind=source_kind,
        location=str(path),
        source_name=PACKAGED_CCD_SOURCE_NAME,
        source_size=path.stat().st_size,
    )
    return _LazyTypedComponents(records, provenance)


@lru_cache(maxsize=8)
def _load_unified_cached(path_text: str, source_kind: CCDSourceKind) -> Mapping[str, CCDComponent]:
    return _load_unified(Path(path_text), source_kind=source_kind)


@lru_cache(maxsize=8)
def _load_v3_cached(path_text: str, source_kind: CCDSourceKind) -> Mapping[str, CCDComponent]:
    return _load_v3(Path(path_text), source_kind=source_kind)


@lru_cache(maxsize=4)
def _load_cif_cached(
    path_text: str,
    source_kind: CCDSourceKind,
    polymer_only: bool,
    complete_source: bool,
) -> Mapping[str, CCDComponent]:
    return _parse_cif(
        Path(path_text),
        source_kind=source_kind,
        polymer_only=polymer_only,
        complete_source=complete_source,
    )


def load_ccd_components(
    path: Path | str,
    *,
    source_kind: CCDSourceKind = "configured",
) -> Mapping[str, CCDComponent]:
    """Load a complete configured/package CCD v3 asset or raw components CIF.

    Raw complete CCD input is filtered to protein/RNA/DNA component types.  Use
    :func:`load_embedded_ccd` for a possibly partial structure-input definition.
    """

    if source_kind not in {"embedded", "configured", "packaged"}:
        raise ValueError(f"invalid CCD source kind {source_kind!r}")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise DesignInputError(f"CCD source does not exist: {resolved}")
    if resolved.name == "ccd.json.gz":
        return _load_unified_cached(str(resolved), source_kind)
    if resolved.name.endswith(".json.gz"):
        # Preserve explicit legacy v3 paths while accepting a renamed copy of
        # the unified resource during packaging and caller migration.
        try:
            return _load_unified_cached(str(resolved), source_kind)
        except DesignInputError:
            return _load_v3_cached(str(resolved), source_kind)
    if resolved.suffix.lower() in {".cif", ".mmcif"}:
        return _load_cif_cached(str(resolved), source_kind, True, True)
    raise DesignInputError(
        f"CCD source must be schema-v3 .json.gz or raw .cif/.mmcif: {resolved}"
    )


def load_packaged_polymer_ccd() -> Mapping[str, CCDComponent]:
    """Load the packaged polymer view from the unified native CCD asset."""

    return load_ccd_components(PACKAGED_POLYMER_CCD_PATH, source_kind="packaged")


def reload_typed_ccd_components(
    path: Path | str,
    source_kind: CCDSourceKind,
) -> Mapping[str, CCDComponent]:
    """Reload a lazy typed CCD mapping from provenance in a child process."""
    return load_ccd_components(path, source_kind=source_kind)


def load_embedded_ccd(path: Path | str) -> Mapping[str, CCDComponent]:
    """Load possibly partial embedded component definitions from an input CIF."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise DesignInputError(f"embedded CCD input does not exist: {resolved}")
    if resolved.suffix.lower() not in {".cif", ".mmcif"}:
        raise DesignInputError(f"embedded CCD input must be CIF: {resolved}")
    return _load_cif_cached(str(resolved), "embedded", False, False)


def _merge_scalar(
    component_id: str,
    field: str,
    high: object,
    low: object,
) -> object:
    if high is not None and low is not None and high != low:
        raise DesignInputError(
            f"CCD component {component_id!r} has conflicting {field}: "
            f"{high!r} versus {low!r}"
        )
    return high if high is not None else low


def _merge_components(high: CCDComponent, low: CCDComponent) -> CCDComponent:
    component_id = high.component_id
    if component_id != low.component_id:
        raise ValueError("cannot merge different CCD component IDs")
    high_type = (
        None
        if high.raw_chem_type is None
        else " ".join(high.raw_chem_type.upper().split())
    )
    low_type = (
        None
        if low.raw_chem_type is None
        else " ".join(low.raw_chem_type.upper().split())
    )
    _merge_scalar(component_id, "chemical type", high_type, low_type)
    raw_type = high.raw_chem_type if high.raw_chem_type is not None else low.raw_chem_type
    modality = _merge_scalar(
        component_id,
        "polymer modality",
        high.polymer_modality,
        low.polymer_modality,
    )
    parent_ids = _merge_scalar(
        component_id, "parent IDs", high.parent_ids, low.parent_ids
    )
    component_charge = _merge_scalar(
        component_id,
        "component formal charge",
        high.component_formal_charge,
        low.component_formal_charge,
    )

    atom_by_id: dict[str, CCDAtom] = {}
    if low.atom_table_present:
        atom_by_id.update({atom.atom_id: atom for atom in low.atoms})
    if high.atom_table_present:
        for atom in high.atoms:
            previous = atom_by_id.get(atom.atom_id)
            atom_by_id[atom.atom_id] = (
                atom
                if previous is None
                else _merge_atom_facts(component_id, previous, atom)
            )

    bonds_by_pair: dict[tuple[str, str], CCDBond] = {}
    if low.bond_table_present:
        bonds_by_pair.update(
            {(bond.atom_id_1, bond.atom_id_2): bond for bond in low.bonds}
        )
    if high.bond_table_present:
        for bond in high.bonds:
            pair = (bond.atom_id_1, bond.atom_id_2)
            previous = bonds_by_pair.get(pair)
            if previous is not None and previous.order != bond.order:
                if previous.order == "any":
                    bonds_by_pair[pair] = bond
                elif bond.order != "any":
                    raise DesignInputError(
                        f"CCD component {component_id!r} bond {pair} has conflicting "
                        f"orders {bond.order!r} versus {previous.order!r}"
                    )
            else:
                bonds_by_pair[pair] = bond

    provenance = tuple(dict.fromkeys(high.provenance + low.provenance))
    return CCDComponent(
        component_id=component_id,
        raw_chem_type=raw_type,
        polymer_modality=modality,  # type: ignore[arg-type]
        parent_ids=parent_ids,  # type: ignore[arg-type]
        component_formal_charge=component_charge,  # type: ignore[arg-type]
        atoms=tuple(atom_by_id[name] for name in sorted(atom_by_id)),
        bonds=tuple(bonds_by_pair[pair] for pair in sorted(bonds_by_pair)),
        provenance=provenance,
        atom_table_present=high.atom_table_present or low.atom_table_present,
        bond_table_present=high.bond_table_present or low.bond_table_present,
        validation_issues=tuple(
            sorted(set(high.validation_issues) | set(low.validation_issues))
        ),
    )


def _unusable_reasons(component: CCDComponent) -> tuple[str, ...]:
    reasons = list(component.validation_issues)
    if component.polymer_modality not in _POLYMER_MODALITIES:
        reasons.append(
            "carbohydrate is ligand-only"
            if _is_carbohydrate(component.raw_chem_type)
            else "component has no protein/RNA/DNA modality"
        )
    if not component.atom_table_present:
        reasons.append("component has no atom table")
    elif not component.atoms:
        reasons.append("component has no heavy atoms")
    if component.component_formal_charge is None:
        reasons.append("component formal charge is missing")
    for atom in component.atoms:
        if atom.element is None:
            reasons.append(f"atom {atom.atom_id!r} element is missing")
        if atom.formal_charge is None:
            reasons.append(f"atom {atom.atom_id!r} formal charge is missing")
        if atom.leaving is None:
            reasons.append(f"atom {atom.atom_id!r} leaving flag is missing")
    for bond in component.bonds:
        if bond.order == "any":
            reasons.append(
                f"bond {bond.atom_id_1!r}-{bond.atom_id_2!r} has unknown-positive order"
            )
    if not component.bond_table_present and len(component.atoms) > 1:
        reasons.append("component has no exact bond table")
    elif component.bond_table_present and len(component.atoms) > 1 and not component.bonds:
        reasons.append("component bond table has no exact heavy-heavy bond")
    atom_ids = {atom.atom_id for atom in component.atoms}
    for bond in component.bonds:
        if bond.atom_id_1 not in atom_ids or bond.atom_id_2 not in atom_ids:
            reasons.append(
                f"bond endpoint is missing: {bond.atom_id_1}-{bond.atom_id_2}"
            )
    return tuple(sorted(set(reasons)))


def resolve_ccd_component(
    component_id: str,
    *,
    embedded: Mapping[str, CCDComponent] | None = None,
    configured: Mapping[str, CCDComponent] | None = None,
    packaged: Mapping[str, CCDComponent] | None = None,
    include_packaged: bool = True,
) -> CCDResolution:
    """Resolve one component with embedded > configured > packaged precedence.

    Precedence fills missing facts only.  Every shared concrete conflict raises
    :class:`DesignInputError`; it never becomes an ``unusable`` proxy fallback.
    """

    normalized = _normal_component_id(component_id, where="CCD resolution")
    packaged_mapping: Mapping[str, CCDComponent] | None = packaged
    if packaged_mapping is None and include_packaged:
        packaged_mapping = load_packaged_polymer_ccd()
    candidates = tuple(
        mapping[normalized]
        for mapping in (packaged_mapping, configured, embedded)
        if mapping is not None and normalized in mapping
    )
    if not candidates:
        return CCDResolution("missing", normalized, None, (), "component is absent")
    merged = candidates[0]
    for higher in candidates[1:]:
        merged = _merge_components(higher, merged)
    reasons = _unusable_reasons(merged)
    if reasons:
        return CCDResolution(
            "unusable",
            normalized,
            merged,
            merged.provenance,
            "; ".join(reasons),
        )
    return CCDResolution("resolved", normalized, merged, merged.provenance)


def component_to_v3_entry(component: CCDComponent) -> dict[str, object]:
    """Convert one complete polymer component to deterministic schema-v3 primitives."""

    if component.raw_chem_type is None or component.polymer_modality not in _POLYMER_MODALITIES:
        raise DesignInputError(
            f"cannot serialize non-polymer CCD component {component.component_id!r}"
        )
    if (
        component.parent_ids is None
        or not component.atom_table_present
        or not component.bond_table_present
        or any(bond.order == "any" for bond in component.bonds)
    ):
        raise DesignInputError(
            f"cannot serialize incomplete CCD component {component.component_id!r}"
        )
    return {
        "raw_chem_type": component.raw_chem_type,
        "polymer_modality": component.polymer_modality,
        "parent_ids": list(component.parent_ids),
        "component_formal_charge": component.component_formal_charge,
        "atoms": [
            {
                "atom_id": atom.atom_id,
                "aliases": list(atom.aliases),
                "element": atom.element,
                "formal_charge": atom.formal_charge,
                "leaving": atom.leaving,
                "ideal_coord": (
                    None if atom.ideal_coord is None else list(atom.ideal_coord)
                ),
            }
            for atom in component.atoms
        ],
        "bonds": [
            [bond.atom_id_1, bond.atom_id_2, bond.order]
            for bond in component.bonds
        ],
        "validation_issues": list(component.validation_issues),
    }


def make_v3_payload(
    components: Mapping[str, CCDComponent],
    *,
    source_name: str,
    source_size: int,
) -> dict[str, object]:
    """Return a sorted schema-v3 payload ready for deterministic JSON encoding."""

    if not source_name or source_size < 0:
        raise ValueError("invalid CCD source provenance")
    entries: dict[str, object] = {}
    for component_id in sorted(components):
        component = components[component_id]
        if component_id != component.component_id:
            raise DesignInputError("CCD component mapping key does not match its record")
        if component.polymer_modality not in _POLYMER_MODALITIES or _is_carbohydrate(
            component.raw_chem_type
        ):
            continue
        entries[component_id] = component_to_v3_entry(component)
    return {
        "schema_version": CCD_SCHEMA_VERSION,
        "source": {
            "filename": source_name,
            "size": source_size,
        },
        "components": entries,
    }


def encode_v3_payload(payload: Mapping[str, object]) -> bytes:
    """Encode schema-v3 JSON deterministically before gzip compression."""

    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def clear_ccd_caches() -> None:
    """Clear bounded loader caches; intended for tests and explicit asset replacement."""

    _load_v3_cached.cache_clear()
    _load_unified_cached.cache_clear()
    _load_cif_cached.cache_clear()
    clear_shared_ccd_caches()


__all__ = [
    "BondOrder",
    "CCDAtom",
    "CCDBond",
    "CCDComponent",
    "CCDProvenance",
    "CCDResolution",
    "CCDResolutionStatus",
    "CCD_SCHEMA_VERSION",
    "MAX_CCD_V3_COMPRESSED_BYTES",
    "PACKAGED_CCD_SOURCE_NAME",
    "PACKAGED_CCD_SOURCE_SIZE",
    "PACKAGED_POLYMER_CCD_PATH",
    "PolymerModality",
    "clear_ccd_caches",
    "component_to_v3_entry",
    "encode_v3_payload",
    "load_ccd_components",
    "load_embedded_ccd",
    "load_packaged_polymer_ccd",
    "reload_typed_ccd_components",
    "make_v3_payload",
    "normalize_polymer_modality",
    "resolve_ccd_component",
]
