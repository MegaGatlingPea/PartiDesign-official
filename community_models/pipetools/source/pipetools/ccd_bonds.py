from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO
from urllib.parse import quote

from .errors import PipeToolsError


SCHEMA_VERSION = 2

_BOND_TABLE = "chem_comp_bond"
_LEAVING_ATOM_TABLE = "chem_comp_leaving_atom"
_METADATA_TABLE = "metadata"
_METADATA_KEYS = {
    "schema_version",
    "source_size",
    "bond_row_count",
    "component_count",
    "leaving_atom_row_count",
    "leaving_atom_component_count",
}
_REQUIRED_BOND_TAGS = {
    "comp_id": "_chem_comp_bond.comp_id",
    "atom_id_1": "_chem_comp_bond.atom_id_1",
    "atom_id_2": "_chem_comp_bond.atom_id_2",
    "value_order": "_chem_comp_bond.value_order",
    "aromatic_flag": "_chem_comp_bond.pdbx_aromatic_flag",
    "stereo_config": "_chem_comp_bond.pdbx_stereo_config",
    "ordinal": "_chem_comp_bond.pdbx_ordinal",
}
_REQUIRED_ATOM_TAGS = {
    "comp_id": "_chem_comp_atom.comp_id",
    "atom_id": "_chem_comp_atom.atom_id",
    "type_symbol": "_chem_comp_atom.type_symbol",
    "leaving_atom_flag": "_chem_comp_atom.pdbx_leaving_atom_flag",
    "ordinal": "_chem_comp_atom.pdbx_ordinal",
}
_EXPECTED_SCHEMAS = {
    _METADATA_TABLE: (
        ("key", "TEXT", 1, 1),
        ("value", "TEXT", 1, 0),
    ),
    _BOND_TABLE: (
        ("comp_id", "TEXT", 1, 1),
        ("atom_id_1", "TEXT", 1, 0),
        ("atom_id_2", "TEXT", 1, 0),
        ("value_order", "TEXT", 1, 0),
        ("aromatic_flag", "TEXT", 1, 0),
        ("stereo_config", "TEXT", 1, 0),
        ("ordinal", "INTEGER", 1, 2),
    ),
    _LEAVING_ATOM_TABLE: (
        ("comp_id", "TEXT", 1, 1),
        ("atom_id", "TEXT", 1, 0),
        ("type_symbol", "TEXT", 1, 0),
        ("ordinal", "INTEGER", 1, 2),
    ),
}
_MAX_LINE_CHARACTERS = 4 * 1024 * 1024
_MAX_TOKEN_CHARACTERS = 4 * 1024 * 1024
_INSERT_BATCH_SIZE = 10_000


class CCDBondCacheError(PipeToolsError):
    """A CCD bond cache cannot be built or trusted safely."""


@dataclass(frozen=True, slots=True)
class CCDBond:
    comp_id: str
    atom_id_1: str
    atom_id_2: str
    value_order: str
    aromatic_flag: str
    stereo_config: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class CCDBondCacheMetadata:
    schema_version: int
    source_size: int
    bond_row_count: int
    component_count: int
    leaving_atom_row_count: int
    leaving_atom_component_count: int


@dataclass(frozen=True, slots=True)
class CCDLeavingAtom:
    comp_id: str
    atom_id: str
    type_symbol: str
    ordinal: int


class CCDBondCacheReader:
    """A validated, read-only cache handle for repeated component queries."""

    def __init__(
        self,
        cache_path: str | os.PathLike[str],
        *,
        source_cif: str | os.PathLike[str] | None = None,
        validated_metadata: CCDBondCacheMetadata | None = None,
    ) -> None:
        self.path = Path(cache_path).expanduser().resolve()
        # Callers can pass metadata from an earlier structural SQLite check.
        # Otherwise the reader performs the structural check once before reuse.
        self.metadata = (
            validated_metadata
            if validated_metadata is not None
            else validate_ccd_bond_cache(self.path, source_cif)
        )
        self._connection: sqlite3.Connection | None = _readonly_connection(self.path)

    def query(self, comp_id: str) -> tuple[CCDBond, ...]:
        _validate_comp_id(comp_id)
        if self._connection is None:
            raise CCDBondCacheError("CCD bond cache reader is closed")
        try:
            rows = self._connection.execute(
                """
                SELECT comp_id, atom_id_1, atom_id_2, value_order,
                       aromatic_flag, stereo_config, ordinal
                FROM chem_comp_bond
                WHERE comp_id = ?
                ORDER BY ordinal ASC
                """,
                (comp_id,),
            ).fetchall()
        except sqlite3.DatabaseError as error:
            raise CCDBondCacheError(
                f"Failed to query CCD component {comp_id!r}: {error}"
            ) from error
        return tuple(CCDBond(*row) for row in rows)

    def query_leaving_atoms(self, comp_id: str) -> tuple[CCDLeavingAtom, ...]:
        _validate_comp_id(comp_id)
        if self._connection is None:
            raise CCDBondCacheError("CCD bond cache reader is closed")
        try:
            rows = self._connection.execute(
                """
                SELECT comp_id, atom_id, type_symbol, ordinal
                FROM chem_comp_leaving_atom
                WHERE comp_id = ?
                ORDER BY ordinal ASC
                """,
                (comp_id,),
            ).fetchall()
        except sqlite3.DatabaseError as error:
            raise CCDBondCacheError(
                f"Failed to query CCD leaving atoms for {comp_id!r}: {error}"
            ) from error
        return tuple(CCDLeavingAtom(*row) for row in rows)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "CCDBondCacheReader":
        if self._connection is None:
            raise CCDBondCacheError("CCD bond cache reader is closed")
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class _CIFToken:
    value: str
    quoted: bool
    line: int
    column: int


class _TokenStream:
    def __init__(self, tokens: Iterator[_CIFToken]) -> None:
        self._tokens = tokens
        self._pending: _CIFToken | None = None

    def next(self) -> _CIFToken | None:
        if self._pending is not None:
            token = self._pending
            self._pending = None
            return token
        return next(self._tokens, None)

    def push(self, token: _CIFToken) -> None:
        if self._pending is not None:
            raise AssertionError("CIF token pushback overflow")
        self._pending = token



class _BondBatchWriter:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._pending: list[tuple[str, str, str, str, str, str, int]] = []
        self.row_count = 0

    def add(self, bond: CCDBond, *, line: int) -> None:
        self._pending.append(
            (
                bond.comp_id,
                bond.atom_id_1,
                bond.atom_id_2,
                bond.value_order,
                bond.aromatic_flag,
                bond.stereo_config,
                bond.ordinal,
            )
        )
        self.row_count += 1
        if len(self._pending) >= _INSERT_BATCH_SIZE:
            self.flush(line=line)

    def flush(self, *, line: int) -> None:
        if not self._pending:
            return
        try:
            self._connection.executemany(
                """
                INSERT INTO chem_comp_bond (
                    comp_id, atom_id_1, atom_id_2, value_order,
                    aromatic_flag, stereo_config, ordinal
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                self._pending,
            )
        except sqlite3.IntegrityError as error:
            raise CCDBondCacheError(
                "Duplicate or invalid _chem_comp_bond primary key at or before "
                f"CIF line {line}: {error}"
            ) from error
        self._pending.clear()


class _LeavingAtomBatchWriter:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._pending: list[tuple[str, str, str, int]] = []
        self.row_count = 0

    def add(self, atom: CCDLeavingAtom, *, line: int) -> None:
        self._pending.append(
            (atom.comp_id, atom.atom_id, atom.type_symbol, atom.ordinal)
        )
        self.row_count += 1
        if len(self._pending) >= _INSERT_BATCH_SIZE:
            self.flush(line=line)

    def flush(self, *, line: int) -> None:
        if not self._pending:
            return
        try:
            self._connection.executemany(
                """
                INSERT INTO chem_comp_leaving_atom (
                    comp_id, atom_id, type_symbol, ordinal
                ) VALUES (?, ?, ?, ?)
                """,
                self._pending,
            )
        except sqlite3.IntegrityError as error:
            raise CCDBondCacheError(
                "Duplicate or invalid _chem_comp_atom leaving-atom key at or "
                f"before CIF line {line}: {error}"
            ) from error
        self._pending.clear()


def build_ccd_bond_cache(
    components_cif: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
) -> Path:
    """Stream the deployed ``components.cif`` into an atomic bond cache.

    The source is tokenized incrementally and never loaded as one document.  A
    valid pre-existing cache remains untouched unless the replacement has been
    completely built, fsynced, and validated.
    """

    source = Path(components_cif).expanduser().resolve()
    destination = Path(cache_path).expanduser().resolve()
    if not source.is_file():
        raise CCDBondCacheError(f"CCD components source is not a file: {source}")
    if source == destination:
        raise CCDBondCacheError("CCD source and cache paths must be different")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(temporary)
        _create_schema(connection)
        bond_writer = _BondBatchWriter(connection)
        leaving_writer = _LeavingAtomBatchWriter(connection)
        source_size = _parse_source(
            source, bond_writer, leaving_writer
        )
        bond_writer.flush(line=0)
        leaving_writer.flush(line=0)
        if bond_writer.row_count == 0:
            raise CCDBondCacheError(
                "CCD source contains no _chem_comp_bond rows"
            )

        component_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT 1 FROM chem_comp_bond GROUP BY comp_id"
                ")"
            ).fetchone()[0]
        )
        metadata = CCDBondCacheMetadata(
            schema_version=SCHEMA_VERSION,
            source_size=source_size,
            bond_row_count=bond_writer.row_count,
            component_count=component_count,
            leaving_atom_row_count=leaving_writer.row_count,
            leaving_atom_component_count=int(
                connection.execute(
                    "SELECT COUNT(*) FROM ("
                    "SELECT 1 FROM chem_comp_leaving_atom GROUP BY comp_id"
                    ")"
                ).fetchone()[0]
            ),
        )
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            _metadata_rows(metadata),
        )
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.commit()
        check = connection.execute("PRAGMA integrity_check").fetchall()
        if check != [("ok",)]:
            raise CCDBondCacheError(
                f"New CCD cache failed SQLite integrity_check: {check!r}"
            )
        connection.close()
        connection = None

        _fsync_file(temporary)
        _validate_cache(
            temporary,
            expected_source_size=source_size,
        )
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return destination
    except CCDBondCacheError:
        raise
    except (OSError, UnicodeError, sqlite3.DatabaseError) as error:
        raise CCDBondCacheError(
            f"Failed to build CCD bond cache from {source}: {error}"
        ) from error
    finally:
        if connection is not None:
            connection.close()
        _remove_temporary_database(temporary)


def validate_ccd_bond_cache(
    cache_path: str | os.PathLike[str],
    source_cif: str | os.PathLike[str] | None = None,
) -> CCDBondCacheMetadata:
    """Validate the SQLite schema, chemical rows, and internal row counts.

    source_cif remains accepted for callers that share the historical reader
    API. The cache itself is authoritative after its structural checks.
    """

    cache = Path(cache_path).expanduser().resolve()
    return _validate_cache(cache)


def query_component_bonds(
    cache_path: str | os.PathLike[str],
    comp_id: str,
    *,
    source_cif: str | os.PathLike[str] | None = None,
) -> tuple[CCDBond, ...]:
    """Return one component's authoritative bonds in ascending ordinal order."""

    with CCDBondCacheReader(cache_path, source_cif=source_cif) as reader:
        return reader.query(comp_id)


def query_component_leaving_atoms(
    cache_path: str | os.PathLike[str],
    comp_id: str,
    *,
    source_cif: str | os.PathLike[str] | None = None,
) -> tuple[CCDLeavingAtom, ...]:
    """Return authoritative heavy/light leaving-atom flags for one CCD."""

    with CCDBondCacheReader(cache_path, source_cif=source_cif) as reader:
        return reader.query_leaving_atoms(comp_id)


def _validate_comp_id(comp_id: str) -> None:
    if not isinstance(comp_id, str) or not comp_id or "\x00" in comp_id:
        raise CCDBondCacheError("comp_id must be a non-empty string without NUL")


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode = DELETE")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute(
        """
        CREATE TABLE metadata (
            key TEXT NOT NULL PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE chem_comp_bond (
            comp_id TEXT NOT NULL,
            atom_id_1 TEXT NOT NULL,
            atom_id_2 TEXT NOT NULL,
            value_order TEXT NOT NULL,
            aromatic_flag TEXT NOT NULL,
            stereo_config TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (comp_id, ordinal)
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE chem_comp_leaving_atom (
            comp_id TEXT NOT NULL,
            atom_id TEXT NOT NULL,
            type_symbol TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (comp_id, ordinal),
            UNIQUE (comp_id, atom_id)
        ) WITHOUT ROWID
        """
    )


def _parse_source(
    source: Path,
    bond_writer: _BondBatchWriter,
    leaving_writer: _LeavingAtomBatchWriter,
) -> int:
    source_size = source.stat().st_size
    with source.open(
        "r", encoding="utf-8-sig", errors="strict", newline=None
    ) as text:
        _parse_cif(
            _TokenStream(_tokenize_cif(text)), bond_writer, leaving_writer
        )
    return source_size


def _parse_cif(
    tokens: _TokenStream,
    bond_writer: _BondBatchWriter,
    leaving_writer: _LeavingAtomBatchWriter,
) -> None:
    while (token := tokens.next()) is not None:
        lowered = token.value.lower()
        if not token.quoted and lowered == "loop_":
            _parse_loop(
                tokens,
                bond_writer,
                leaving_writer,
                loop_token=token,
            )
            continue
        if not token.quoted and token.value.startswith("_"):
            if lowered.startswith("_chem_comp_bond."):
                _parse_scalar_bond_category(
                    tokens, bond_writer, first_tag=token
                )
                continue
            if lowered.startswith("_chem_comp_atom."):
                _parse_scalar_atom_category(
                    tokens, leaving_writer, first_tag=token
                )
                continue
            value = tokens.next()
            if value is None or _is_control_token(value):
                raise CCDBondCacheError(
                    f"Missing scalar value for {token.value!r} at CIF line {token.line}"
                )
            continue
        if not token.quoted and _is_block_marker(lowered):
            continue
        if not token.quoted and lowered == "stop_":
            raise CCDBondCacheError(
                f"Unexpected stop_ outside a loop at CIF line {token.line}"
            )
        raise CCDBondCacheError(
            f"Unexpected CIF token {token.value!r} at line {token.line}, "
            f"column {token.column}"
        )


def _parse_scalar_bond_category(
    tokens: _TokenStream,
    writer: _BondBatchWriter,
    *,
    first_tag: _CIFToken,
) -> None:
    """Parse the legal one-row scalar form used by some CCD components."""

    fields: dict[str, _CIFToken] = {}
    tag = first_tag
    while True:
        lowered = tag.value.lower()
        if lowered in fields:
            raise CCDBondCacheError(
                f"Duplicate scalar {tag.value!r} at CIF line {tag.line}"
            )
        value = tokens.next()
        if value is None or _is_control_token(value):
            raise CCDBondCacheError(
                f"Missing scalar value for {tag.value!r} at CIF line {tag.line}"
            )
        fields[lowered] = value
        following = tokens.next()
        if following is None:
            break
        if (
            not following.quoted
            and following.value.lower().startswith("_chem_comp_bond.")
        ):
            tag = following
            continue
        tokens.push(following)
        break

    missing = sorted(set(_REQUIRED_BOND_TAGS.values()) - set(fields))
    if missing:
        raise CCDBondCacheError(
            "Scalar _chem_comp_bond category is missing required fields: "
            + ", ".join(missing)
        )
    ordered = [fields[tag] for tag in _REQUIRED_BOND_TAGS.values()]
    indexes = {field: index for index, field in enumerate(_REQUIRED_BOND_TAGS)}
    writer.add(_bond_from_row(ordered, indexes), line=first_tag.line)
    writer.flush(line=first_tag.line)


def _parse_scalar_atom_category(
    tokens: _TokenStream,
    writer: _LeavingAtomBatchWriter,
    *,
    first_tag: _CIFToken,
) -> None:
    """Parse a legal one-row scalar ``_chem_comp_atom`` category."""

    fields: dict[str, _CIFToken] = {}
    tag = first_tag
    while True:
        lowered = tag.value.lower()
        if lowered in fields:
            raise CCDBondCacheError(
                f"Duplicate scalar {tag.value!r} at CIF line {tag.line}"
            )
        value = tokens.next()
        if value is None or _is_control_token(value):
            raise CCDBondCacheError(
                f"Missing scalar value for {tag.value!r} at CIF line {tag.line}"
            )
        fields[lowered] = value
        following = tokens.next()
        if following is None:
            break
        if (
            not following.quoted
            and following.value.lower().startswith("_chem_comp_atom.")
        ):
            tag = following
            continue
        tokens.push(following)
        break

    missing = sorted(set(_REQUIRED_ATOM_TAGS.values()) - set(fields))
    if missing:
        raise CCDBondCacheError(
            "Scalar _chem_comp_atom category is missing required fields: "
            + ", ".join(missing)
        )
    ordered = [fields[tag] for tag in _REQUIRED_ATOM_TAGS.values()]
    indexes = {field: index for index, field in enumerate(_REQUIRED_ATOM_TAGS)}
    atom = _leaving_atom_from_row(ordered, indexes)
    if atom is not None:
        writer.add(atom, line=first_tag.line)
    writer.flush(line=first_tag.line)


def _parse_loop(
    tokens: _TokenStream,
    bond_writer: _BondBatchWriter,
    leaving_writer: _LeavingAtomBatchWriter,
    *,
    loop_token: _CIFToken,
) -> None:
    tags: list[_CIFToken] = []
    while (token := tokens.next()) is not None:
        if token.quoted or not token.value.startswith("_"):
            tokens.push(token)
            break
        tags.append(token)
    if not tags:
        raise CCDBondCacheError(
            f"loop_ at CIF line {loop_token.line} has no data-name columns"
        )

    lowered_tags = [token.value.lower() for token in tags]
    if len(set(lowered_tags)) != len(lowered_tags):
        raise CCDBondCacheError(
            f"loop_ at CIF line {loop_token.line} has duplicate columns"
        )
    bond_tags = [
        tag for tag in lowered_tags if tag.startswith("_chem_comp_bond.")
    ]
    atom_tags = [
        tag for tag in lowered_tags if tag.startswith("_chem_comp_atom.")
    ]
    bond_indexes: dict[str, int] | None = None
    atom_indexes: dict[str, int] | None = None
    if bond_tags:
        if len(bond_tags) != len(tags):
            raise CCDBondCacheError(
                "_chem_comp_bond loop mixes columns from different categories "
                f"at CIF line {loop_token.line}"
            )
        missing = sorted(set(_REQUIRED_BOND_TAGS.values()) - set(lowered_tags))
        if missing:
            raise CCDBondCacheError(
                "_chem_comp_bond loop is missing required columns: "
                + ", ".join(missing)
            )
        bond_indexes = {
            field: lowered_tags.index(tag)
            for field, tag in _REQUIRED_BOND_TAGS.items()
        }
    elif atom_tags:
        if len(atom_tags) != len(tags):
            raise CCDBondCacheError(
                "_chem_comp_atom loop mixes columns from different categories "
                f"at CIF line {loop_token.line}"
            )
        missing = sorted(set(_REQUIRED_ATOM_TAGS.values()) - set(lowered_tags))
        if missing:
            raise CCDBondCacheError(
                "_chem_comp_atom loop is missing required columns: "
                + ", ".join(missing)
            )
        atom_indexes = {
            field: lowered_tags.index(tag)
            for field, tag in _REQUIRED_ATOM_TAGS.items()
        }

    row: list[_CIFToken] = []
    row_count = 0
    last_line = loop_token.line
    while True:
        token = tokens.next()
        if token is None:
            if row:
                raise CCDBondCacheError(
                    f"Incomplete loop row at end of CIF (started near line {row[0].line})"
                )
            break
        last_line = token.line
        if _is_control_token(token):
            if row:
                raise CCDBondCacheError(
                    "Incomplete loop row before control token "
                    f"{token.value!r} at CIF line {token.line}"
                )
            if token.value.lower() != "stop_":
                tokens.push(token)
            break
        row.append(token)
        if len(row) == len(tags):
            row_count += 1
            if bond_indexes is not None:
                bond_writer.add(
                    _bond_from_row(row, bond_indexes),
                    line=row[0].line,
                )
            elif atom_indexes is not None:
                atom = _leaving_atom_from_row(row, atom_indexes)
                if atom is not None:
                    leaving_writer.add(atom, line=row[0].line)
            row = []

    if (bond_indexes is not None or atom_indexes is not None) and row_count == 0:
        raise CCDBondCacheError(
            f"target CCD loop at CIF line {loop_token.line} has no rows"
        )
    bond_writer.flush(line=last_line)
    leaving_writer.flush(line=last_line)


def _bond_from_row(
    row: list[_CIFToken], column_indexes: dict[str, int]
) -> CCDBond:
    values = {
        field: row[index].value for field, index in column_indexes.items()
    }
    for field in ("comp_id", "atom_id_1", "atom_id_2"):
        value = values[field]
        if not value or value in {"?", "."} or "\x00" in value:
            raise CCDBondCacheError(
                f"Invalid {field} in _chem_comp_bond row at CIF line {row[0].line}"
            )
    for field in ("value_order", "aromatic_flag", "stereo_config"):
        if not values[field] or "\x00" in values[field]:
            raise CCDBondCacheError(
                f"Invalid {field} in _chem_comp_bond row at CIF line {row[0].line}"
            )
    try:
        ordinal = int(values["ordinal"], 10)
    except ValueError as error:
        raise CCDBondCacheError(
            "_chem_comp_bond.pdbx_ordinal must be an integer at "
            f"CIF line {row[0].line}: {values['ordinal']!r}"
        ) from error
    if ordinal <= 0:
        raise CCDBondCacheError(
            "_chem_comp_bond.pdbx_ordinal must be positive at "
            f"CIF line {row[0].line}"
        )
    return CCDBond(
        comp_id=values["comp_id"],
        atom_id_1=values["atom_id_1"],
        atom_id_2=values["atom_id_2"],
        value_order=values["value_order"],
        aromatic_flag=values["aromatic_flag"],
        stereo_config=values["stereo_config"],
        ordinal=ordinal,
    )


def _leaving_atom_from_row(
    row: list[_CIFToken], column_indexes: dict[str, int]
) -> CCDLeavingAtom | None:
    values = {
        field: row[index].value for field, index in column_indexes.items()
    }
    flag = values["leaving_atom_flag"].upper()
    if flag not in {"Y", "N"}:
        raise CCDBondCacheError(
            "_chem_comp_atom.pdbx_leaving_atom_flag must be Y or N at "
            f"CIF line {row[0].line}: {values['leaving_atom_flag']!r}"
        )
    if flag == "N":
        return None
    for field in ("comp_id", "atom_id", "type_symbol"):
        value = values[field]
        if not value or value in {"?", "."} or "\x00" in value:
            raise CCDBondCacheError(
                f"Invalid {field} in _chem_comp_atom row at CIF line {row[0].line}"
            )
    try:
        ordinal = int(values["ordinal"], 10)
    except ValueError as error:
        raise CCDBondCacheError(
            "_chem_comp_atom.pdbx_ordinal must be an integer at "
            f"CIF line {row[0].line}: {values['ordinal']!r}"
        ) from error
    if ordinal <= 0:
        raise CCDBondCacheError(
            "_chem_comp_atom.pdbx_ordinal must be positive at "
            f"CIF line {row[0].line}"
        )
    return CCDLeavingAtom(
        comp_id=values["comp_id"],
        atom_id=values["atom_id"],
        type_symbol=values["type_symbol"].upper(),
        ordinal=ordinal,
    )


def _tokenize_cif(handle: TextIO) -> Iterator[_CIFToken]:
    quote_character: str | None = None
    quoted_parts: list[str] = []
    quoted_length = 0
    token_line = 0
    token_column = 0
    in_semicolon_text = False
    semicolon_parts: list[str] = []
    semicolon_length = 0

    line_number = 0
    while True:
        line = handle.readline(_MAX_LINE_CHARACTERS + 1)
        if not line:
            break
        line_number += 1
        if len(line) > _MAX_LINE_CHARACTERS:
            raise CCDBondCacheError(
                f"CIF physical line {line_number} exceeds bounded parser limit"
            )

        if in_semicolon_text:
            if line.startswith(";"):
                trailing = line[1:].rstrip("\n")
                if trailing.strip() and not trailing.lstrip().startswith("#"):
                    raise CCDBondCacheError(
                        "Unexpected content after semicolon text terminator at "
                        f"CIF line {line_number}"
                    )
                yield _CIFToken(
                    "".join(semicolon_parts), True, token_line, token_column
                )
                in_semicolon_text = False
                semicolon_parts = []
                semicolon_length = 0
                continue
            semicolon_length = _append_bounded(
                semicolon_parts,
                line,
                semicolon_length,
                line=token_line,
            )
            continue

        index = 0
        if quote_character is None and line.startswith(";"):
            in_semicolon_text = True
            token_line = line_number
            token_column = 1
            semicolon_length = _append_bounded(
                semicolon_parts,
                line[1:],
                semicolon_length,
                line=token_line,
            )
            continue

        while index < len(line):
            if quote_character is not None:
                closing = line.find(quote_character, index)
                while closing >= 0:
                    following = closing + 1
                    if following == len(line) or line[following].isspace() or line[following] == "#":
                        quoted_length = _append_bounded(
                            quoted_parts,
                            line[index:closing],
                            quoted_length,
                            line=token_line,
                        )
                        yield _CIFToken(
                            "".join(quoted_parts),
                            True,
                            token_line,
                            token_column,
                        )
                        quote_character = None
                        quoted_parts = []
                        quoted_length = 0
                        index = following
                        break
                    quoted_length = _append_bounded(
                        quoted_parts,
                        line[index : closing + 1],
                        quoted_length,
                        line=token_line,
                    )
                    index = closing + 1
                    closing = line.find(quote_character, index)
                if quote_character is not None:
                    quoted_length = _append_bounded(
                        quoted_parts,
                        line[index:],
                        quoted_length,
                        line=token_line,
                    )
                    index = len(line)
                continue

            while index < len(line) and line[index].isspace():
                index += 1
            if index >= len(line) or line[index] == "#":
                break
            if line[index] in {"'", '"'}:
                quote_character = line[index]
                quoted_parts = []
                quoted_length = 0
                token_line = line_number
                token_column = index + 1
                index += 1
                continue

            start = index
            while index < len(line) and not line[index].isspace():
                index += 1
            value = line[start:index]
            if len(value) > _MAX_TOKEN_CHARACTERS:
                raise CCDBondCacheError(
                    f"CIF token at line {line_number} exceeds bounded parser limit"
                )
            yield _CIFToken(value, False, line_number, start + 1)

    if quote_character is not None:
        raise CCDBondCacheError(
            f"Unterminated quoted CIF token beginning at line {token_line}"
        )
    if in_semicolon_text:
        raise CCDBondCacheError(
            f"Unterminated semicolon CIF text beginning at line {token_line}"
        )


def _append_bounded(
    parts: list[str], value: str, length: int, *, line: int
) -> int:
    new_length = length + len(value)
    if new_length > _MAX_TOKEN_CHARACTERS:
        raise CCDBondCacheError(
            f"CIF token beginning at line {line} exceeds bounded parser limit"
        )
    parts.append(value)
    return new_length


def _is_control_token(token: _CIFToken) -> bool:
    if token.quoted:
        return False
    lowered = token.value.lower()
    return token.value.startswith("_") or _is_block_marker(lowered) or lowered in {
        "loop_",
        "stop_",
    }


def _is_block_marker(lowered: str) -> bool:
    return (
        lowered == "global_"
        or lowered == "save_"
        or lowered.startswith("save_")
        or lowered.startswith("data_")
    )


def _metadata_rows(
    metadata: CCDBondCacheMetadata,
) -> tuple[tuple[str, str], ...]:
    return (
        ("schema_version", str(metadata.schema_version)),
        ("source_size", str(metadata.source_size)),
        ("bond_row_count", str(metadata.bond_row_count)),
        ("component_count", str(metadata.component_count)),
        ("leaving_atom_row_count", str(metadata.leaving_atom_row_count)),
        (
            "leaving_atom_component_count",
            str(metadata.leaving_atom_component_count),
        ),
    )


def _validate_cache(
    cache: Path,
    *,
    expected_source_size: int | None = None,
) -> CCDBondCacheMetadata:
    if not cache.is_file():
        raise CCDBondCacheError(f"CCD bond cache is not a file: {cache}")
    try:
        with closing(_readonly_connection(cache)) as connection:
            objects = {
                (str(row[0]), str(row[1]))
                for row in connection.execute(
                    """
                    SELECT type, name FROM sqlite_master
                    WHERE name NOT LIKE 'sqlite_%'
                    """
                )
            }
            expected_objects = {
                ("table", _METADATA_TABLE),
                ("table", _BOND_TABLE),
                ("table", _LEAVING_ATOM_TABLE),
            }
            if objects != expected_objects:
                raise CCDBondCacheError(
                    f"Unexpected CCD cache objects: {sorted(objects)!r}"
                )
            for table, expected in _EXPECTED_SCHEMAS.items():
                observed = tuple(
                    (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
                    for row in connection.execute(f"PRAGMA table_info({table})")
                )
                if observed != expected:
                    raise CCDBondCacheError(
                        f"CCD cache table {table!r} has an unexpected schema"
                    )
            user_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if user_version != SCHEMA_VERSION:
                raise CCDBondCacheError(
                    f"Unsupported CCD cache user_version {user_version}"
                )
            integrity = connection.execute("PRAGMA integrity_check").fetchall()
            if integrity != [("ok",)]:
                raise CCDBondCacheError(
                    f"CCD cache failed SQLite integrity_check: {integrity!r}"
                )

            raw_metadata = dict(
                connection.execute("SELECT key, value FROM metadata").fetchall()
            )
            if not _METADATA_KEYS <= set(raw_metadata):
                raise CCDBondCacheError(
                    "CCD cache metadata is missing required fields"
                )
            metadata = _parse_metadata(raw_metadata)
            actual_rows = int(
                connection.execute("SELECT COUNT(*) FROM chem_comp_bond").fetchone()[0]
            )
            actual_components = int(
                connection.execute(
                    "SELECT COUNT(*) FROM ("
                    "SELECT 1 FROM chem_comp_bond GROUP BY comp_id"
                    ")"
                ).fetchone()[0]
            )
            if actual_rows != metadata.bond_row_count:
                raise CCDBondCacheError(
                    "CCD cache bond-row count does not match metadata"
                )
            if actual_components != metadata.component_count:
                raise CCDBondCacheError(
                    "CCD cache component count does not match metadata"
                )
            actual_leaving_rows = int(
                connection.execute(
                    "SELECT COUNT(*) FROM chem_comp_leaving_atom"
                ).fetchone()[0]
            )
            actual_leaving_components = int(
                connection.execute(
                    "SELECT COUNT(*) FROM ("
                    "SELECT 1 FROM chem_comp_leaving_atom GROUP BY comp_id"
                    ")"
                ).fetchone()[0]
            )
            if actual_leaving_rows != metadata.leaving_atom_row_count:
                raise CCDBondCacheError(
                    "CCD cache leaving-atom row count does not match metadata"
                )
            if (
                actual_leaving_components
                != metadata.leaving_atom_component_count
            ):
                raise CCDBondCacheError(
                    "CCD cache leaving-atom component count does not match metadata"
                )
            invalid_rows = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM chem_comp_bond
                    WHERE comp_id = '' OR atom_id_1 = '' OR atom_id_2 = ''
                       OR value_order = '' OR aromatic_flag = ''
                       OR stereo_config = '' OR ordinal <= 0
                       OR typeof(ordinal) != 'integer'
                    """
                ).fetchone()[0]
            )
            if invalid_rows:
                raise CCDBondCacheError(
                    f"CCD cache contains {invalid_rows} invalid bond rows"
                )
            invalid_leaving_rows = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM chem_comp_leaving_atom
                    WHERE comp_id = '' OR atom_id = '' OR type_symbol = ''
                       OR ordinal <= 0 OR typeof(ordinal) != 'integer'
                    """
                ).fetchone()[0]
            )
            if invalid_leaving_rows:
                raise CCDBondCacheError(
                    "CCD cache contains "
                    f"{invalid_leaving_rows} invalid leaving-atom rows"
                )
    except CCDBondCacheError:
        raise
    except sqlite3.DatabaseError as error:
        raise CCDBondCacheError(
            f"CCD bond cache is not a valid SQLite database: {cache}: {error}"
        ) from error

    if expected_source_size is not None and metadata.source_size != expected_source_size:
        raise CCDBondCacheError("CCD cache source size metadata is incorrect")
    return metadata


def _parse_metadata(raw: dict[str, str]) -> CCDBondCacheMetadata:
    try:
        schema_version = _parse_nonnegative_integer(
            raw["schema_version"], "schema_version"
        )
        source_size = _parse_nonnegative_integer(raw["source_size"], "source_size")
        bond_row_count = _parse_nonnegative_integer(
            raw["bond_row_count"], "bond_row_count"
        )
        component_count = _parse_nonnegative_integer(
            raw["component_count"], "component_count"
        )
        leaving_atom_row_count = _parse_nonnegative_integer(
            raw["leaving_atom_row_count"], "leaving_atom_row_count"
        )
        leaving_atom_component_count = _parse_nonnegative_integer(
            raw["leaving_atom_component_count"],
            "leaving_atom_component_count",
        )
    except (KeyError, TypeError) as error:
        raise CCDBondCacheError("CCD cache metadata is malformed") from error
    if schema_version != SCHEMA_VERSION:
        raise CCDBondCacheError(
            f"Unsupported CCD cache schema_version {schema_version}"
        )
    if bond_row_count <= 0 or component_count <= 0:
        raise CCDBondCacheError("CCD cache count metadata must be positive")
    return CCDBondCacheMetadata(
        schema_version=schema_version,
        source_size=source_size,

        bond_row_count=bond_row_count,
        component_count=component_count,
        leaving_atom_row_count=leaving_atom_row_count,
        leaving_atom_component_count=leaving_atom_component_count,
    )


def _parse_nonnegative_integer(value: str, name: str) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise CCDBondCacheError(f"CCD cache {name} metadata is malformed")
    parsed = int(value)
    if parsed < 0:
        raise CCDBondCacheError(f"CCD cache {name} metadata is negative")
    return parsed


def _readonly_connection(path: Path) -> sqlite3.Connection:
    encoded_path = quote(os.fspath(path), safe="/")
    connection = sqlite3.connect(
        f"file:{encoded_path}?mode=ro",
        uri=True,
        timeout=30,
    )
    connection.execute("PRAGMA query_only = ON")
    return connection



def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_temporary_database(path: Path) -> None:
    for candidate in (
        path,
        Path(f"{path}-journal"),
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
    ):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "CCDBond",
    "CCDBondCacheError",
    "CCDBondCacheMetadata",
    "CCDBondCacheReader",
    "CCDLeavingAtom",
    "SCHEMA_VERSION",
    "build_ccd_bond_cache",
    "query_component_bonds",
    "query_component_leaving_atoms",
    "validate_ccd_bond_cache",
]
