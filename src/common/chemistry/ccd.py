"""Shared lazy accessors for the merged native CCD resource.

Schema 4 stores one record per component.  The metadata object is serialized before
components so classification and one-letter consumers can stop a gzip stream after
that leading object.  Graph views share one incremental component stream and cache
only records requested by callers; iterating a view remains available for offline
parity and audit checks.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping as ABCMapping
from functools import lru_cache
import gzip
import json
import os
import threading
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from partidesign.paths import resource_root

CCD_SCHEMA_VERSION = 4
DEFAULT_CCD_PATH = resource_root() / "configs" / "resources" / "chemistry" / "ccd.json.gz"
_HYDROGEN_ELEMENTS = frozenset({"H", "D", "T"})
_BOND_ORDER = {"single": 1, "double": 2, "triple": 3, "aromatic": 4, "any": 0}
_JSON_CHUNK_BYTES = 64 * 1024


class CCDResourceError(ValueError):
    """Raised when the merged CCD resource is missing or malformed."""


def _resolve_path(path: str | Path | None) -> Path:
    if path is None:
        return DEFAULT_CCD_PATH
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else candidate.resolve()


class _JSONStream:
    """Small incremental JSON decoder over a binary or text stream."""

    __slots__ = ("_stream", "_decoder", "_buffer", "_pos", "_eof")

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._decoder = json.JSONDecoder()
        self._buffer = ""
        self._pos = 0
        self._eof = False

    def _fill(self) -> None:
        if self._eof:
            return
        chunk = self._stream.read(_JSON_CHUNK_BYTES)
        if not chunk:
            self._eof = True
            return
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        self._buffer = self._buffer[self._pos :] + chunk
        self._pos = 0

    def _skip_whitespace(self) -> None:
        while True:
            while self._pos < len(self._buffer) and self._buffer[self._pos].isspace():
                self._pos += 1
            if self._pos < len(self._buffer) or self._eof:
                return
            self._fill()

    def peek(self) -> str:
        self._skip_whitespace()
        while self._pos >= len(self._buffer) and not self._eof:
            self._fill()
            self._skip_whitespace()
        if self._pos >= len(self._buffer):
            raise CCDResourceError("unexpected end of CCD JSON")
        return self._buffer[self._pos]

    def expect(self, expected: str) -> None:
        actual = self.peek()
        if actual != expected:
            raise CCDResourceError(f"expected {expected!r} in CCD JSON, got {actual!r}")
        self._pos += 1

    def value(self) -> Any:
        self._skip_whitespace()
        while True:
            try:
                value, end = self._decoder.raw_decode(self._buffer, self._pos)
            except json.JSONDecodeError as exc:
                if self._eof:
                    raise CCDResourceError(f"invalid CCD JSON: {exc}") from exc
                self._fill()
                self._skip_whitespace()
                continue
            self._pos = end
            return value

    def delimiter(self) -> str:
        self._skip_whitespace()
        while self._pos >= len(self._buffer) and not self._eof:
            self._fill()
            self._skip_whitespace()
        if self._pos >= len(self._buffer):
            raise CCDResourceError("unexpected end of CCD JSON after value")
        value = self._buffer[self._pos]
        self._pos += 1
        if value not in {",", "}"}:
            raise CCDResourceError(f"expected CCD JSON delimiter, got {value!r}")
        return value


def _validate_header(
    schema_version: object,
    sources: object,
    metadata: object | None = None,
) -> None:
    if schema_version != CCD_SCHEMA_VERSION:
        raise CCDResourceError(f"unsupported merged CCD schema {schema_version!r}")
    if not isinstance(sources, Mapping):
        raise CCDResourceError("merged CCD sources must be a mapping")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise CCDResourceError("merged CCD metadata must be a mapping")


def _expand_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    # Schema-4 resources use compact source-shaped metadata.  Accept the original
    # per-component shape too so explicit transition resources remain readable.
    if {"polymer_types", "one_letter_codes"} <= set(metadata):
        polymer_types = metadata["polymer_types"]
        one_letter_codes = metadata["one_letter_codes"]
        if not isinstance(polymer_types, Mapping) or not isinstance(one_letter_codes, Mapping):
            raise CCDResourceError("compact CCD metadata views must be mappings")
        result: dict[str, dict[str, Any]] = {}
        for raw_id, value in one_letter_codes.items():
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise CCDResourceError("CCD metadata IDs must be nonempty strings")
            result.setdefault(raw_id.strip().upper(), {})["one_letter_code"] = value
        for raw_id, value in polymer_types.items():
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise CCDResourceError("CCD metadata IDs must be nonempty strings")
            if not isinstance(value, list) or len(value) != 2:
                raise CCDResourceError(f"CCD polymer type metadata {raw_id!r} is invalid")
            row = result.setdefault(raw_id.strip().upper(), {})
            row["polymer_modality"], row["raw_chem_type"] = value
        parent_values = metadata.get("polymer_parent_ids")
        if parent_values is not None:
            if not isinstance(parent_values, Mapping):
                raise CCDResourceError("CCD polymer parent metadata must be a mapping")
            for raw_id, value in parent_values.items():
                if not isinstance(raw_id, str) or not raw_id.strip():
                    raise CCDResourceError("CCD metadata IDs must be nonempty strings")
                if value is not None and (
                    not isinstance(value, list)
                    or not all(isinstance(parent, str) for parent in value)
                ):
                    raise CCDResourceError(
                        f"CCD polymer parent metadata {raw_id!r} is invalid"
                    )
                result.setdefault(raw_id.strip().upper(), {})["parent_ids"] = (
                    None if value is None else list(value)
                )
        return MappingProxyType({key: MappingProxyType(value) for key, value in result.items()})
    result: dict[str, Mapping[str, Any]] = {}
    for raw_id, value in metadata.items():
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise CCDResourceError("CCD metadata IDs must be nonempty strings")
        if not isinstance(value, Mapping):
            raise CCDResourceError(f"CCD metadata {raw_id!r} must be a mapping")
        result[raw_id.strip().upper()] = MappingProxyType(dict(value))
    return MappingProxyType(result)


def _validate_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(metadata, Mapping):
        raise CCDResourceError("merged CCD metadata must be a mapping")
    return _expand_metadata(metadata)


def _read_metadata_stream(path: Path) -> Mapping[str, Mapping[str, Any]]:
    if not path.is_file():
        raise CCDResourceError(f"CCD resource is missing: {path}")
    try:
        with gzip.open(path, "rb") if path.name.endswith(".gz") else path.open("rb") as stream:
            decoder = _JSONStream(stream)
            decoder.expect("{")
            schema_version: object | None = None
            sources: object | None = None
            while True:
                key = decoder.value()
                if not isinstance(key, str):
                    raise CCDResourceError("CCD root keys must be strings")
                decoder.expect(":")
                if key == "schema_version":
                    schema_version = decoder.value()
                elif key == "sources":
                    sources = decoder.value()
                elif key == "metadata":
                    metadata = decoder.value()
                    _validate_header(schema_version, sources, metadata)
                    return _validate_metadata(metadata)
                elif key == "components":
                    raise CCDResourceError("CCD metadata must precede components")
                else:
                    raise CCDResourceError(f"unexpected CCD root field {key!r}")
                if decoder.delimiter() == "}":
                    raise CCDResourceError("CCD root is missing metadata")
    except CCDResourceError:
        raise
    except Exception as exc:
        raise CCDResourceError(f"cannot parse CCD metadata {path}: {exc}") from exc


def _read_metadata_fields(path: Path, fields: frozenset[str]) -> Mapping[str, Any]:
    if not path.is_file():
        raise CCDResourceError(f"CCD resource is missing: {path}")
    try:
        with gzip.open(path, "rb") if path.name.endswith(".gz") else path.open("rb") as stream:
            decoder = _JSONStream(stream)
            decoder.expect("{")
            schema_version: object | None = None
            sources: object | None = None
            while True:
                key = decoder.value()
                if not isinstance(key, str):
                    raise CCDResourceError("CCD root keys must be strings")
                decoder.expect(":")
                if key == "schema_version":
                    schema_version = decoder.value()
                elif key == "sources":
                    sources = decoder.value()
                elif key == "metadata":
                    _validate_header(schema_version, sources)
                    decoder.expect("{")
                    found: dict[str, Any] = {}
                    while True:
                        if decoder.peek() == "}":
                            decoder.expect("}")
                            break
                        metadata_key = decoder.value()
                        if not isinstance(metadata_key, str):
                            raise CCDResourceError("CCD metadata keys must be strings")
                        decoder.expect(":")
                        value = decoder.value()
                        if metadata_key in fields:
                            found[metadata_key] = value
                        delimiter = decoder.delimiter()
                        if delimiter == "}":
                            break
                    if not fields <= found.keys():
                        missing = sorted(fields - found.keys())
                        raise CCDResourceError(f"CCD metadata is missing {missing}")
                    return MappingProxyType(found)
                elif key == "components":
                    raise CCDResourceError("CCD metadata must precede components")
                else:
                    raise CCDResourceError(f"unexpected CCD root field {key!r}")
                if decoder.delimiter() == "}":
                    raise CCDResourceError("CCD root is missing metadata")
    except CCDResourceError:
        raise
    except Exception as exc:
        raise CCDResourceError(f"cannot parse CCD metadata {path}: {exc}") from exc


def _read_compact_metadata_field(path: Path, field: str) -> Any:
    """Read one compact leading-index field without walking its member keys."""
    if not path.is_file():
        raise CCDResourceError(f"CCD resource is missing: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            decoder = json.JSONDecoder()
            text = ""
            metadata_start = -1
            field_start = -1
            while field_start < 0:
                chunk = stream.read(131072)
                if not chunk:
                    raise CCDResourceError(f"CCD metadata field {field!r} is missing")
                text += chunk
                if metadata_start < 0:
                    metadata_start = text.find('"metadata"')
                if metadata_start >= 0:
                    field_start = text.find(json.dumps(field), metadata_start)
            colon = text.find(":", field_start + len(field) + 2)
            if colon < 0:
                raise CCDResourceError(f"CCD metadata field {field!r} lacks a value")
            value_start = colon + 1
            while True:
                try:
                    value, _end = decoder.raw_decode(text, value_start)
                    break
                except json.JSONDecodeError:
                    chunk = stream.read(131072)
                    if not chunk:
                        raise CCDResourceError(f"CCD metadata field {field!r} is truncated")
                    text += chunk
            header_text = text[:metadata_start].rstrip()
            if header_text.endswith(","):
                header_text = header_text[:-1]
            header = json.loads(header_text + "}")
            if not isinstance(header, Mapping):
                raise CCDResourceError("merged CCD header must be a mapping")
            _validate_header(header.get("schema_version"), header.get("sources"))
            return value
    except CCDResourceError:
        raise
    except Exception as exc:
        raise CCDResourceError(f"cannot parse compact CCD metadata {path}: {exc}") from exc


@lru_cache(maxsize=8)
def _metadata_fields_cached(path_text: str, fields: tuple[str, ...]) -> Mapping[str, Any]:
    path = Path(path_text)
    if len(fields) == 1:
        try:
            return MappingProxyType({fields[0]: _read_compact_metadata_field(path, fields[0])})
        except CCDResourceError as exc:
            # Explicit transition resources may use the expanded metadata shape.
            message = str(exc)
            if "metadata field" not in message and "metadata is missing" not in message:
                raise
    return _read_metadata_fields(path, frozenset(fields))


def _read_full_document(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise CCDResourceError(f"CCD resource is missing: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8") if path.name.endswith(".gz") else path.open(
            "rt", encoding="utf-8"
        ) as stream:
            payload = json.load(stream)
    except Exception as exc:
        raise CCDResourceError(f"cannot parse CCD resource {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CCDResourceError(f"CCD resource root must be a mapping: {path}")
    required = {"schema_version", "sources", "metadata", "components"}
    if set(payload) != required:
        raise CCDResourceError(f"merged CCD root must contain exactly {sorted(required)}")
    _validate_header(payload["schema_version"], payload["sources"], payload["metadata"])
    if not isinstance(payload["components"], Mapping):
        raise CCDResourceError("merged CCD components must be a mapping")
    # Keep the audit helper's historical per-component metadata shape even though
    # the shipped leading index is compact.
    expanded = dict(payload)
    expanded["metadata"] = _expand_metadata(payload["metadata"])
    return expanded


def load_ccd_document(path: str | Path | None = None) -> Mapping[str, Any]:
    """Load the complete document for build and audit tooling.

    Runtime consumers use the lazy accessors below.  This explicit audit helper is
    intentionally the full-document path.
    """
    return _read_full_document(_resolve_path(path))


@lru_cache(maxsize=4)
def _metadata_cached(path_text: str) -> Mapping[str, Mapping[str, Any]]:
    return _read_metadata_stream(Path(path_text))


def ccd_metadata(path: str | Path | None = None) -> Mapping[str, Mapping[str, Any]]:
    return _metadata_cached(str(_resolve_path(path)))


class _PolymerMetadataValue(ABCMapping[str, Any]):
    __slots__ = ("_values",)

    def __init__(self, values: list[Any]) -> None:
        if not isinstance(values, list) or len(values) != 2:
            raise CCDResourceError("CCD polymer type metadata must contain two values")
        self._values = values

    def __getitem__(self, key: str) -> Any:
        if key == "polymer_modality":
            return self._values[0]
        if key == "raw_chem_type":
            return self._values[1]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("polymer_modality", "raw_chem_type"))

    def __len__(self) -> int:
        return 2


class _PolymerMetadataProjection(ABCMapping[str, Mapping[str, Any]]):
    __slots__ = ("_raw", "_cache")

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self._raw = raw
        self._cache: dict[str, Mapping[str, Any]] = {}

    def __getitem__(self, key: str) -> Mapping[str, Any]:
        normalized = _fast_id(key)
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached
        values = self._raw.get(normalized)
        if values is None:
            raise KeyError(normalized)
        result = _PolymerMetadataValue(values)
        self._cache[normalized] = result
        return result

    def get(self, key: object, default: Any = None) -> Mapping[str, Any] | Any:
        try:
            normalized = _fast_id(key)
            cached = self._cache.get(normalized)
            if cached is not None:
                return cached
            values = self._raw.get(normalized)
            if values is None:
                return default
            result = _PolymerMetadataValue(values)
            self._cache[normalized] = result
            return result
        except (CCDResourceError, KeyError):
            return default

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)


@lru_cache(maxsize=4)
def _polymer_type_pairs_cached(path_text: str) -> Mapping[str, list[Any]]:
    raw = _metadata_fields_cached(path_text, ("polymer_types",)).get("polymer_types")
    if not isinstance(raw, Mapping):
        raise CCDResourceError("CCD polymer type metadata must be a mapping")
    # The builder validates every pair offline.  Keep the parsed mapping directly;
    # individual values are checked when a requested record is projected.
    return raw


def polymer_type_pairs(path: str | Path | None = None) -> Mapping[str, list[Any]]:
    return _polymer_type_pairs_cached(str(_resolve_path(path)))


@lru_cache(maxsize=4)
def _polymer_metadata_cached(path_text: str) -> Mapping[str, Mapping[str, Any]]:
    return _PolymerMetadataProjection(_polymer_type_pairs_cached(path_text))


def polymer_metadata(path: str | Path | None = None) -> Mapping[str, Mapping[str, Any]]:
    return _polymer_metadata_cached(str(_resolve_path(path)))


@lru_cache(maxsize=4)
def _polymer_component_ids_cached(path_text: str) -> tuple[str, ...]:
    raw = _metadata_fields_cached(path_text, ("polymer_types",)).get("polymer_types")
    if isinstance(raw, Mapping):
        result = tuple(_normal_id(value) for value in raw)
    else:
        result = tuple(
            key for key, value in _metadata_cached(path_text).items()
            if "polymer_modality" in value and "raw_chem_type" in value
        )
    if not result or len(result) != len(set(result)):
        raise CCDResourceError("CCD polymer metadata IDs are invalid")
    return result


def ccd_polymer_component_ids(path: str | Path | None = None) -> tuple[str, ...]:
    """Return polymer IDs from the leading metadata without opening components."""
    return _polymer_component_ids_cached(str(_resolve_path(path)))


@lru_cache(maxsize=4)
def _polymer_parent_ids_cached(path_text: str) -> Mapping[str, list[str] | None]:
    raw = _metadata_fields_cached(path_text, ("polymer_parent_ids",)).get(
        "polymer_parent_ids"
    )
    if not isinstance(raw, Mapping):
        # Transition compatibility for an expanded metadata resource.
        raw = {
            key: value.get("parent_ids")
            for key, value in _metadata_cached(path_text).items()
            if "polymer_modality" in value and "raw_chem_type" in value
        }
    result: dict[str, list[str] | None] = {}
    for raw_id, value in raw.items():
        component_id = _normal_id(raw_id)
        if value is not None and (
            not isinstance(value, list)
            or not all(isinstance(parent, str) for parent in value)
        ):
            raise CCDResourceError(
                f"CCD polymer parent metadata {component_id!r} is invalid"
            )
        result[component_id] = None if value is None else list(value)
    if not result:
        raise CCDResourceError("CCD polymer parent metadata must not be empty")
    return MappingProxyType(result)


def polymer_parent_ids(
    path: str | Path | None = None,
) -> Mapping[str, list[str] | None]:
    """Return exact polymer parent declarations from leading metadata only."""
    return _polymer_parent_ids_cached(str(_resolve_path(path)))


def ccd_polymer_parent_ids(
    path: str | Path | None = None,
) -> Mapping[str, list[str] | None]:
    """Compatibility alias for the shared compact parent metadata accessor."""
    return polymer_parent_ids(path)


@lru_cache(maxsize=4)
def _one_letter_cached(path_text: str) -> Mapping[str, str | None]:
    raw = _metadata_fields_cached(path_text, ("one_letter_codes",)).get("one_letter_codes")
    if isinstance(raw, Mapping):
        result: dict[str, str | None] = {}
        for raw_id, code in raw.items():
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise CCDResourceError("CCD metadata IDs must be nonempty strings")
            if code is not None and not isinstance(code, str):
                raise CCDResourceError(f"invalid one-letter code for {raw_id!r}")
            result[raw_id.strip().upper()] = code
        if not result:
            raise CCDResourceError("CCD one-letter metadata must not be empty")
        return MappingProxyType(result)
    # Transition compatibility for an expanded metadata resource.
    result = {
        key: metadata["one_letter_code"]
        for key, metadata in _metadata_cached(path_text).items()
        if "one_letter_code" in metadata
    }
    if not result:
        raise CCDResourceError("CCD one-letter metadata must not be empty")
    return MappingProxyType(result)


def ccd_one_letter_codes(path: str | Path | None = None) -> Mapping[str, str | None]:
    return _one_letter_cached(str(_resolve_path(path)))


def _normal_id(raw: object) -> str:
    value = str(raw).strip().upper()
    if not value or value in {".", "?"}:
        raise CCDResourceError("CCD component IDs must be nonempty")
    return value


def _fast_id(raw: object) -> str:
    if isinstance(raw, str) and raw and raw == raw.strip() and raw == raw.upper() and raw not in {".", "?"}:
        return raw
    return _normal_id(raw)


def _polymer_atoms(polymer: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    rows = polymer.get("atoms")
    if not isinstance(rows, list):
        raise CCDResourceError("CCD polymer atoms must be a list")
    names: list[str] = []
    elements: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise CCDResourceError("CCD polymer atom must be a mapping")
        atom_id = str(row.get("atom_id", "")).strip()
        element = row.get("element")
        if not atom_id or not isinstance(element, str) or not element:
            raise CCDResourceError("CCD polymer atom lacks ID or element")
        if atom_id in elements:
            raise CCDResourceError(f"duplicate CCD polymer atom {atom_id!r}")
        names.append(atom_id)
        elements[atom_id] = element
    return names, elements


def _polymer_bonds(polymer: Mapping[str, Any], atom_ids: set[str]) -> list[list[Any]]:
    rows = polymer.get("bonds")
    if not isinstance(rows, list):
        raise CCDResourceError("CCD polymer bonds must be a list")
    result: list[list[Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, list) or len(row) != 3:
            raise CCDResourceError("CCD polymer bond must have three values")
        left, right, raw_order = str(row[0]), str(row[1]), str(row[2]).lower()
        pair = tuple(sorted((left, right)))
        if left == right or left not in atom_ids or right not in atom_ids or pair in seen:
            raise CCDResourceError(f"invalid CCD polymer bond {pair!r}")
        if raw_order not in _BOND_ORDER or raw_order == "any":
            raise CCDResourceError(f"invalid CCD polymer bond order {raw_order!r}")
        seen.add(pair)
        result.append([left, right, _BOND_ORDER[raw_order]])
    return result


def _topology_view(
    component_id: str, view: Mapping[str, Any], polymer: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    mode = view.get("mode")
    atoms, elements, bonds = view.get("atoms"), view.get("elements"), view.get("bonds")
    if mode not in {"full", "delta"} or not isinstance(atoms, list) or not isinstance(elements, list):
        raise CCDResourceError(f"invalid CCD topology view for {component_id!r}")
    if len(atoms) != len(elements) or not all(isinstance(a, str) and a for a in atoms):
        raise CCDResourceError(f"invalid CCD topology atoms for {component_id!r}")
    if len(set(atoms)) != len(atoms) or not all(isinstance(e, str) and e for e in elements):
        raise CCDResourceError(f"invalid CCD topology elements for {component_id!r}")
    if not isinstance(bonds, list):
        raise CCDResourceError(f"invalid CCD topology bonds for {component_id!r}")
    atom_set = set(atoms)
    normalized_bonds: list[list[Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in bonds:
        if (
            not isinstance(row, list) or len(row) != 3
            or not isinstance(row[0], str) or not isinstance(row[1], str)
            or row[0] == row[1]
            or (mode == "full" and (row[0] not in atom_set or row[1] not in atom_set))
            or isinstance(row[2], bool) or not isinstance(row[2], int)
            or row[2] not in {1, 2, 3, 4}
        ):
            raise CCDResourceError(f"invalid CCD topology bond for {component_id!r}")
        pair = tuple(sorted((row[0], row[1])))
        if pair in seen:
            raise CCDResourceError(f"duplicate CCD topology bond {pair!r}")
        seen.add(pair)
        normalized_bonds.append([row[0], row[1], row[2]])
    if mode == "full":
        if polymer is not None:
            raise CCDResourceError(f"full topology {component_id!r} unexpectedly has polymer data")
        full_names, full_elements, full_bonds = list(atoms), list(elements), normalized_bonds
    else:
        if not isinstance(polymer, Mapping):
            raise CCDResourceError(f"delta topology {component_id!r} lacks polymer data")
        polymer_names, polymer_elements = _polymer_atoms(polymer)
        if set(atoms) & set(polymer_names):
            raise CCDResourceError(f"delta topology {component_id!r} repeats a polymer atom")
        if any(element not in _HYDROGEN_ELEMENTS for element in elements):
            raise CCDResourceError(f"delta topology {component_id!r} has a non-hydrogen atom")
        full_names = sorted(set(polymer_names) | set(atoms))
        element_map = dict(polymer_elements)
        element_map.update(dict(zip(atoms, elements, strict=True)))
        full_elements = [element_map[name] for name in full_names]
        full_bonds = _polymer_bonds(polymer, set(polymer_names)) + normalized_bonds
    return MappingProxyType({"atoms": full_names, "elements": full_elements, "bonds": full_bonds})


class _ComponentStore:
    """Incrementally parse component records from one shared gzip stream."""

    __slots__ = (
        "_decoder", "_stream", "_path", "_pid", "_lock", "_expected",
        "_polymer_expected", "_polymer_ids", "_records", "_order", "_done",
    )

    def __init__(
        self,
        decoder: _JSONStream,
        stream: Any,
        path: Path,
        expected: int,
        polymer_expected: int,
        polymer_ids: frozenset[str],
    ) -> None:
        self._decoder = decoder
        self._stream = stream
        self._path = path
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._expected = expected
        self._polymer_expected = polymer_expected
        self._polymer_ids = polymer_ids
        self._records: dict[str, Mapping[str, Any]] = {}
        self._order: list[str] = []
        self._done = False

    def _close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.close()

    def _prepare_for_process(self) -> None:
        """Reopen the stream after fork without inheriting a live cursor/lock."""

        pid = os.getpid()
        if self._pid == pid:
            return
        # A fork can copy an RLock held by a different parent thread.  Replace the
        # lock before acquiring it, then reopen a fresh cursor in the child.
        self._lock = threading.RLock()
        self._close()
        replacement = _open_component_store(self._path).store
        self._decoder = replacement._decoder
        self._stream = replacement._stream
        self._records.clear()
        self._order.clear()
        self._done = False
        self._pid = pid
        replacement._stream = None

    def _guard(self):
        self._prepare_for_process()
        return self._lock

    def _read_one(self) -> None:
        if self._done:
            return
        if self._decoder.peek() == "}":
            self._decoder.expect("}")
            self._done = True
            self._close()
            if len(self._order) != self._expected:
                raise CCDResourceError(
                    f"CCD component inventory mismatch: {len(self._order)} != {self._expected}"
                )
            return
        raw_id = self._decoder.value()
        if not isinstance(raw_id, str):
            raise CCDResourceError("CCD component IDs must be strings")
        self._decoder.expect(":")
        value = self._decoder.value()
        if not isinstance(value, Mapping):
            raise CCDResourceError(f"CCD component {raw_id!r} must be a mapping")
        component_id = _normal_id(raw_id)
        if component_id in self._records:
            raise CCDResourceError(f"duplicate CCD component {component_id!r}")
        self._records[component_id] = value
        self._order.append(component_id)
        delimiter = self._decoder.delimiter()
        if delimiter == "}":
            self._done = True
            self._close()
            if len(self._order) != self._expected:
                raise CCDResourceError(
                    f"CCD component inventory mismatch: {len(self._order)} != {self._expected}"
                )

    def get(self, component_id: str) -> Mapping[str, Any]:
        normalized = _fast_id(component_id)
        self._prepare_for_process()
        # Protect both the shared-record lookup and any cursor advancement.  View
        # caches bypass this path on warm hits, so the lock is paid only for raw
        # stream access while concurrent readers remain serialized safely.
        with self._lock:
            value = self._records.get(normalized)
            while value is None and not self._done:
                self._read_one()
                value = self._records.get(normalized)
            if value is None:
                raise KeyError(normalized)
            return value

    def drain(self) -> None:
        with self._guard():
            while not self._done:
                self._read_one()

    def __del__(self) -> None:
        self._close()


class _RawComponentView(ABCMapping[str, Mapping[str, Any]]):
    def __init__(self, store: _ComponentStore, count: int) -> None:
        self._store = store
        self._count = count

    def __getitem__(self, component_id: str) -> Mapping[str, Any]:
        return self._store.get(component_id)

    def get(self, component_id: object, default: Any = None) -> Mapping[str, Any] | Any:
        try:
            return self._store.get(_fast_id(component_id))
        except (CCDResourceError, KeyError):
            return default

    def __iter__(self) -> Iterator[str]:
        self._store.drain()
        with self._store._guard():
            return iter(tuple(self._store._order))

    def __len__(self) -> int:
        return self._count


class _ProjectedComponentView(ABCMapping[str, Mapping[str, Any]]):
    def __init__(self, store: _ComponentStore, field: str, count: int) -> None:
        self._store = store
        self._field = field
        self._count = count
        self._cache: dict[str, Mapping[str, Any]] = {}

    def __getitem__(self, component_id: str) -> Mapping[str, Any]:
        normalized = _fast_id(component_id)
        if normalized not in self._store._polymer_ids:
            raise KeyError(normalized)
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached
        with self._store._guard():
            cached = self._cache.get(normalized)
            if cached is not None:
                return cached
            record = self._store.get(normalized)
            value = record.get(self._field)
            if not isinstance(value, Mapping):
                raise KeyError(normalized)
            result = MappingProxyType(dict(value)) if self._field == "polymer" else value
            self._cache[normalized] = result
            return result

    def get(self, component_id: object, default: Any = None) -> Mapping[str, Any] | Any:
        try:
            normalized = _fast_id(component_id)
            if normalized not in self._store._polymer_ids:
                return default
            cached = self._cache.get(normalized)
            if cached is not None:
                return cached
            with self._store._guard():
                cached = self._cache.get(normalized)
                if cached is not None:
                    return cached
                record = self._store.get(normalized)
                value = record.get(self._field)
                if not isinstance(value, Mapping):
                    return default
                result = MappingProxyType(dict(value)) if self._field == "polymer" else value
                self._cache[normalized] = result
                return result
        except (CCDResourceError, KeyError):
            return default

    def __iter__(self) -> Iterator[str]:
        self._store.drain()
        with self._store._guard():
            result: list[str] = []
            for component_id in self._store._order:
                record = self._store._records[component_id]
                if isinstance(record.get(self._field), Mapping):
                    if component_id not in self._cache:
                        value = record[self._field]
                        self._cache[component_id] = (
                            MappingProxyType(dict(value)) if self._field == "polymer" else value
                        )
                    result.append(component_id)
            return iter(result)

    def __len__(self) -> int:
        return self._count


class _TopologyComponentView(ABCMapping[str, Mapping[str, Any]]):
    def __init__(self, store: _ComponentStore, count: int) -> None:
        self._store = store
        self._count = count
        self._cache: dict[str, Mapping[str, Any]] = {}

    def __getitem__(self, component_id: str) -> Mapping[str, Any]:
        normalized = _fast_id(component_id)
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached
        with self._store._guard():
            cached = self._cache.get(normalized)
            if cached is not None:
                return cached
            record = self._store.get(normalized)
            view = record.get("topology")
            if not isinstance(view, Mapping):
                raise KeyError(normalized)
            result = _topology_view(normalized, view, record.get("polymer"))
            self._cache[normalized] = result
            return result

    def get(self, component_id: object, default: Any = None) -> Mapping[str, Any] | Any:
        try:
            normalized = _fast_id(component_id)
            cached = self._cache.get(normalized)
            if cached is not None:
                return cached
            with self._store._guard():
                cached = self._cache.get(normalized)
                if cached is not None:
                    return cached
                record = self._store.get(normalized)
                view = record.get("topology")
                if not isinstance(view, Mapping):
                    return default
                result = _topology_view(normalized, view, record.get("polymer"))
                self._cache[normalized] = result
                return result
        except (CCDResourceError, KeyError):
            return default

    def __iter__(self) -> Iterator[str]:
        self._store.drain()
        with self._store._guard():
            result: list[str] = []
            for component_id in self._store._order:
                if isinstance(self._store._records[component_id].get("topology"), Mapping):
                    if component_id not in self._cache:
                        record = self._store._records[component_id]
                        self._cache[component_id] = _topology_view(
                            component_id, record["topology"], record.get("polymer")
                        )
                    result.append(component_id)
            return iter(result)

    def __len__(self) -> int:
        return self._count


class _ComponentStreamContext:
    __slots__ = ("store",)

    def __init__(self, store: _ComponentStore) -> None:
        self.store = store


def _open_component_store(path: Path) -> _ComponentStreamContext:
    if not path.is_file():
        raise CCDResourceError(f"CCD resource is missing: {path}")
    try:
        stream = gzip.open(path, "rb") if path.name.endswith(".gz") else path.open("rb")
        decoder = _JSONStream(stream)
        decoder.expect("{")
        schema_version: object | None = None
        sources: Mapping[str, Any] | None = None
        metadata_seen = False
        while True:
            key = decoder.value()
            if not isinstance(key, str):
                raise CCDResourceError("CCD root keys must be strings")
            decoder.expect(":")
            if key == "schema_version":
                schema_version = decoder.value()
            elif key == "sources":
                value = decoder.value()
                if not isinstance(value, Mapping):
                    raise CCDResourceError("merged CCD sources must be a mapping")
                sources = value
            elif key == "metadata":
                metadata = decoder.value()
                _validate_header(schema_version, sources, metadata)
                metadata_seen = True
            elif key == "components":
                if not metadata_seen or sources is None:
                    raise CCDResourceError("CCD components must follow metadata")
                _validate_header(schema_version, sources, metadata)
                component_info = sources.get("topology")
                expected = component_info.get("entry_count") if isinstance(component_info, Mapping) else None
                polymer_info = sources.get("polymer_components")
                polymer_expected = polymer_info.get("entry_count") if isinstance(polymer_info, Mapping) else None
                if not isinstance(expected, int) or expected < 0:
                    raise CCDResourceError("CCD topology source count is invalid")
                if not isinstance(polymer_expected, int) or polymer_expected < 0:
                    raise CCDResourceError("CCD polymer source count is invalid")
                if isinstance(metadata, Mapping) and {"polymer_types", "one_letter_codes"} <= set(metadata):
                    polymer_ids = frozenset(str(key).strip().upper() for key in metadata["polymer_types"])
                else:
                    expanded_metadata = _expand_metadata(metadata)
                    polymer_ids = frozenset(
                        key for key, value in expanded_metadata.items()
                        if "polymer_modality" in value and "raw_chem_type" in value
                    )
                decoder.expect("{")
                return _ComponentStreamContext(
                    _ComponentStore(decoder, stream, path, expected, polymer_expected, polymer_ids)
                )
            else:
                raise CCDResourceError(f"unexpected CCD root field {key!r}")
            if decoder.delimiter() == "}":
                raise CCDResourceError("CCD root is missing components")
    except CCDResourceError:
        try:
            stream.close()
        except UnboundLocalError:
            pass
        raise
    except Exception as exc:
        try:
            stream.close()
        except UnboundLocalError:
            pass
        raise CCDResourceError(f"cannot parse CCD resource {path}: {exc}") from exc


@lru_cache(maxsize=4)
def _component_store_cached(path_text: str) -> _ComponentStore:
    return _open_component_store(Path(path_text)).store


@lru_cache(maxsize=4)
def _components_view_cached(path_text: str) -> Mapping[str, Mapping[str, Any]]:
    store = _component_store_cached(path_text)
    return _RawComponentView(store, store._expected)


@lru_cache(maxsize=4)
def _polymer_view_cached(path_text: str) -> Mapping[str, Mapping[str, Any]]:
    store = _component_store_cached(path_text)
    return _ProjectedComponentView(store, "polymer", store._polymer_expected)


@lru_cache(maxsize=4)
def _topology_view_cached(path_text: str) -> Mapping[str, Mapping[str, Any]]:
    store = _component_store_cached(path_text)
    return _TopologyComponentView(store, store._expected)


def ccd_components(path: str | Path | None = None) -> Mapping[str, Mapping[str, Any]]:
    return _components_view_cached(str(_resolve_path(path)))


def ccd_polymer_components(path: str | Path | None = None) -> Mapping[str, Mapping[str, Any]]:
    return _polymer_view_cached(str(_resolve_path(path)))


def ccd_topology_components(path: str | Path | None = None) -> Mapping[str, Mapping[str, Any]]:
    return _topology_view_cached(str(_resolve_path(path)))


@lru_cache(maxsize=4)
def _topology_ids_cached(path_text: str) -> tuple[str, ...]:
    raw = _metadata_fields_cached(path_text, ("topology_ids",)).get("topology_ids")
    if not isinstance(raw, list) or not raw:
        raise CCDResourceError("CCD topology index must be a nonempty list")
    result = tuple(_normal_id(value) for value in raw)
    if len(result) != len(set(result)):
        raise CCDResourceError("CCD topology index contains duplicate IDs")
    return result


def ccd_topology_ids(path: str | Path | None = None) -> tuple[str, ...]:
    """Return the leading topology ID index without parsing component graphs."""
    return _topology_ids_cached(str(_resolve_path(path)))


def clear_ccd_caches() -> None:
    _metadata_cached.cache_clear()
    _metadata_fields_cached.cache_clear()
    _polymer_type_pairs_cached.cache_clear()
    _polymer_metadata_cached.cache_clear()
    _polymer_component_ids_cached.cache_clear()
    _polymer_parent_ids_cached.cache_clear()
    _one_letter_cached.cache_clear()
    _components_view_cached.cache_clear()
    _polymer_view_cached.cache_clear()
    _topology_view_cached.cache_clear()
    _topology_ids_cached.cache_clear()
    _component_store_cached.cache_clear()


__all__ = [
    "CCDResourceError", "CCD_SCHEMA_VERSION", "DEFAULT_CCD_PATH",
    "ccd_components", "ccd_metadata", "ccd_one_letter_codes",
    "ccd_polymer_components", "ccd_topology_components", "ccd_topology_ids", "clear_ccd_caches",
    "load_ccd_document", "polymer_metadata", "polymer_parent_ids",
    "ccd_polymer_parent_ids", "polymer_type_pairs", "ccd_polymer_component_ids",
]
