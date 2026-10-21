"""Exact-query AlphaFold 3 MSA/template cache.

The AF3 SDK is imported only in the isolated data-only subprocess started by
the cache.  Configuration and cache inspection therefore stay lightweight.
"""
from __future__ import annotations

import contextlib
import copy
import datetime as _dt
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from partidesign.paths import af3_source, asset_root

DEFAULT_MSA_CONFIG: dict[str, Any] = {
    "mode": "auto",
    "cache_dir": ".cache/af3_msa",
    "cache_profile": None,
    "database_dir": None,
    "binaries_dir": None,
    "search_workers": 2,
    "cpu_budget": "auto",
    "jackhmmer_n_cpu": 7,
    "nhmmer_n_cpu": 7,
    "timeout_seconds": 14400,
    "database_revision": None,
    "refresh": False,
    "templates": True,
    "max_template_date": "2021-09-30",
}
_MODALITIES = ("protein", "rna")
_DATABASES = {
    "small_bfd": "bfd-first_non_consensus_sequences.fasta",
    "mgnify": "mgy_clusters_2022_05.fa",
    "uniref90": "uniref90_2022_05.fa",
    "uniprot": "uniprot_all_2021_04.fa",
    "nt_rna": "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",
    "rfam": "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",
    "rnacentral": "rnacentral_active_seq_id_90_cov_80_linclust.fasta",
    "seqres": "pdb_seqres_2022_09_28.fasta",
    "mmcif": "mmcif_files",
}
_PROTEIN_DB = ("uniref90", "small_bfd", "mgnify", "uniprot")
_RNA_DB = ("nt_rna", "rfam", "rnacentral")
_BINARIES = ("jackhmmer", "nhmmer", "hmmalign", "hmmbuild", "hmmsearch")
_SEARCH_PARAMETERS = {
    "protein_databases": list(_PROTEIN_DB),
    "rna_databases": list(_RNA_DB),
    "protein_e_value": 1e-4,
    "protein_iterations": 1,
    "protein_max_sequences": [10000, 5000, 5000, 50000],
    "rna_e_value": 1e-3,
    "rna_max_sequences": 10000,
    "deduplicate": True,
}
_A3M_CHARS = re.compile(r"^[A-Za-z0-9.*_:-]+$")
_SCHEMA = 1


class MSAError(RuntimeError):
    """Base cache/search error."""


class MSAConfigError(ValueError):
    """Invalid MSA configuration or request."""


class MSAResourceError(MSAError):
    """Required AF3/HMMER resource is missing."""


class MSACacheMiss(MSAError):
    """A cache-only request has no exact ready entry."""


class MSAReadOnlyError(MSAError):
    """A write/search was requested from a read-only cache."""


class MSASearchError(MSAError):
    """Native AF3 data-only search failed."""


class MSASearchTimeout(MSASearchError):
    """Native AF3 data-only search exceeded its deadline."""


def _unwrap(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MSAConfigError("MSA settings must be a mapping")
    if "msa" in value:
        if set(value) != {"msa"} or not isinstance(value["msa"], Mapping):
            raise MSAConfigError("msa must be the only top-level MSA mapping")
        return value["msa"]
    return value


def validate_msa_settings(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    """Deep-merge defaults and validate the closed MSA settings schema."""
    if settings is None:
        raise MSAConfigError("MSA settings must be a mapping")
    raw = _unwrap(settings)
    unknown = set(raw) - set(DEFAULT_MSA_CONFIG)
    if unknown:
        raise MSAConfigError(f"Unknown MSA settings: {sorted(unknown)}")
    result = copy.deepcopy(DEFAULT_MSA_CONFIG)
    result.update(copy.deepcopy(dict(raw)))
    if not isinstance(result["mode"], str) or result["mode"] not in {"auto", "off", "cache-only"}:
        raise MSAConfigError("mode must be auto, off, or cache-only")
    for key in ("cache_dir",):
        if isinstance(result[key], os.PathLike):
            result[key] = os.fspath(result[key])
        if not isinstance(result[key], str) or not result[key].strip():
            raise MSAConfigError(f"{key} must be a nonempty path string")
    for key in ("database_dir", "binaries_dir", "database_revision"):
        if isinstance(result[key], os.PathLike):
            result[key] = os.fspath(result[key])
        if result[key] is not None and (
            not isinstance(result[key], str) or not result[key].strip()
        ):
            raise MSAConfigError(f"{key} must be null or a nonempty string")
    profile = result["cache_profile"]
    if profile is not None and (
        isinstance(profile, bool) or not isinstance(profile, (str, int))
    ):
        raise MSAConfigError("cache_profile must be null, integer, or string")
    if isinstance(profile, str) and not profile.strip():
        raise MSAConfigError("cache_profile must be null, integer, or nonempty string")
    if isinstance(profile, int) and profile < 1:
        raise MSAConfigError("cache_profile must be null, positive integer, or string")
    if type(result["search_workers"]) is not int or result["search_workers"] < 1:
        raise MSAConfigError("search_workers must be a positive integer")
    for key in ("jackhmmer_n_cpu", "nhmmer_n_cpu"):
        if type(result[key]) is not int or result[key] < 1:
            raise MSAConfigError(f"{key} must be a positive integer")
    budget = result["cpu_budget"]
    if budget != "auto" and (type(budget) is not int or budget < 1):
        raise MSAConfigError("cpu_budget must be auto or a positive integer")
    timeout = result["timeout_seconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0
    ):
        raise MSAConfigError("timeout_seconds must be a positive finite number")
    if type(result["refresh"]) is not bool or type(result["templates"]) is not bool:
        raise MSAConfigError("refresh and templates must be boolean")
    if result["refresh"] and result["mode"] != "auto":
        raise MSAConfigError("MSA refresh requires mode auto; cache-only/off cannot publish a new generation")
    if result["cache_profile"] is not None and result["mode"] != "cache-only":
        raise MSAConfigError("cache_profile requires mode cache-only")
    if not isinstance(result["max_template_date"], str):
        raise MSAConfigError("max_template_date must be an ISO date string")
    try:
        parsed_date = _dt.date.fromisoformat(result["max_template_date"])
    except ValueError as exc:
        raise MSAConfigError("max_template_date must be an ISO date string") from exc
    if parsed_date.isoformat() != result["max_template_date"]:
        raise MSAConfigError("max_template_date must be an ISO date string")
    return result


def resolve_msa_settings(
    settings: Mapping[str, Any] | None, repo_root: str | os.PathLike[str]
) -> dict[str, Any]:
    result = validate_msa_settings(settings)
    root = asset_root(repo_root).expanduser().resolve()
    for key in ("cache_dir", "database_dir", "binaries_dir"):
        if result[key] is None:
            continue
        value = Path(result[key]).expanduser()
        if not value.is_absolute():
            value = root / value
        result[key] = str(value.resolve())
    return result


def _sequence(sequence: str) -> str:
    if not isinstance(sequence, str):
        raise MSAConfigError("sequence must be a string")
    value = "".join(sequence.split()).upper()
    if not value:
        raise MSAConfigError("sequence must be nonempty")
    return value


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fasta(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    title: str | None = None
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if title is not None:
                rows.append((title, "".join(parts)))
            title = line[1:].strip()
            if not title:
                raise MSASearchError("empty FASTA header")
            parts = []
        elif title is None:
            raise MSASearchError("A3M has no FASTA header")
        else:
            parts.append(line.replace(" ", ""))
    if title is not None:
        rows.append((title, "".join(parts)))
    if not rows:
        raise MSASearchError("empty A3M")
    return rows


def _validate_a3m(text: str, query: str, name: str) -> int:
    rows = _fasta(text)
    query = _sequence(query)
    for index, (_, row) in enumerate(rows):
        if not row or not _A3M_CHARS.fullmatch(row):
            raise MSASearchError(f"{name} row {index} contains invalid sequence")
        columns = "".join(char for char in row if not char.islower())
        if len(columns) != len(query):
            raise MSASearchError(
                f"{name} row {index} has {len(columns)} columns, expected {len(query)}"
            )
        if index == 0 and columns.replace("-", "") != query:
            raise MSASearchError(f"{name} first row does not match query")
    return len(rows)


def _database_dir() -> Path | None:
    raw = os.environ.get("AF3_DB_DIR") or os.environ.get("ALPHAFOLD3_DB_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    path = Path("/mnt/software/AlphaFold3/alphafold3/public_databases")
    return path if path.is_dir() else None


def _binaries_dir() -> Path | None:
    raw = os.environ.get("AF3_BINARIES_DIR") or os.environ.get("ALPHAFOLD3_BINARIES_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    active = Path(sys.prefix) / "bin"
    if all((active / name).is_file() for name in _BINARIES):
        return active
    path = Path("/mnt/software/miniforge3/envs/Alphafold3/bin")
    return path if path.is_dir() else None


_NATIVE_SOURCE = r'''
import datetime
import json
import os
from pathlib import Path
import sys

request_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
request = json.loads(request_path.read_text())
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, request["source_root"])

from alphafold3.common import folding_input
from alphafold3.data.pipeline import DataPipeline, DataPipelineConfig

paths = request["paths"]
config = DataPipelineConfig(
    jackhmmer_binary_path=paths["jackhmmer"],
    nhmmer_binary_path=paths["nhmmer"],
    hmmalign_binary_path=paths["hmmalign"],
    hmmsearch_binary_path=paths["hmmsearch"],
    hmmbuild_binary_path=paths["hmmbuild"],
    small_bfd_database_path=paths["small_bfd"],
    mgnify_database_path=paths["mgnify"],
    uniprot_cluster_annot_database_path=paths["uniprot"],
    uniref90_database_path=paths["uniref90"],
    ntrna_database_path=paths["nt_rna"],
    rfam_database_path=paths["rfam"],
    rna_central_database_path=paths["rnacentral"],
    seqres_database_path=paths["seqres"],
    pdb_database_path=paths["mmcif"],
    jackhmmer_n_cpu=int(request["jackhmmer_n_cpu"]),
    nhmmer_n_cpu=int(request["nhmmer_n_cpu"]),
    max_template_date=datetime.date.fromisoformat(request["max_template_date"]),
)
pipeline = DataPipeline(config)
sequence = request["sequence"]
if request["modality"] == "protein":
    chain = folding_input.ProteinChain(
        id="query", sequence=sequence, ptms=[],
        templates=None if request["templates"] else [],
    )
    processed = pipeline.process_protein_chain(chain)
    templates = [{
        "mmcif": item.mmcif,
        "query_to_template_map": {
            str(key): int(value) for key, value in item.query_to_template_map.items()
        },
    } for item in processed.templates]
    result = {
        "modality": "protein", "sequence": sequence,
        "unpaired_msa": processed.unpaired_msa,
        "paired_msa": processed.paired_msa, "templates": templates,
    }
else:
    chain = folding_input.RnaChain(
        id="query", sequence=sequence, modifications=[]
    )
    processed = pipeline.process_rna_chain(chain)
    result = {
        "modality": "rna", "sequence": sequence,
        "unpaired_msa": processed.unpaired_msa, "templates": [],
    }
partial = output_path.with_name(output_path.name + ".partial")
partial.write_text(json.dumps(result, ensure_ascii=True))
os.replace(partial, output_path)
'''


class AF3MSACache:
    """Persistent cache with exact text identity and immutable generations."""

    def __init__(
        self,
        repo_root: str | os.PathLike[str],
        settings: Mapping[str, Any] | None,
        readonly: bool = False,
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.settings = resolve_msa_settings(settings, self.repo_root)
        self.source_root = af3_source(self.repo_root)
        self.readonly = bool(readonly)
        self.cache_dir = self._cache_dir()
        self.index_path = self.cache_dir / "index.sqlite"
        self.work_dir = self.cache_dir / ".work"
        self.lock_dir = self.cache_dir / ".locks"
        self._native_probe: bool | None = None
        if not self.readonly and self.settings["mode"] == "auto":
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.work_dir.mkdir(parents=True, exist_ok=True)
            self.lock_dir.mkdir(parents=True, exist_ok=True)
            self._schema()

    def _cache_dir(self) -> Path:
        path = Path(self.settings["cache_dir"]).expanduser()
        return (self.repo_root / path).resolve() if not path.is_absolute() else path.resolve()

    def _db_root(self) -> Path | None:
        raw = self.settings["database_dir"]
        return Path(raw).resolve() if raw is not None else _database_dir()

    def _bin_root(self) -> Path | None:
        raw = self.settings["binaries_dir"]
        return Path(raw).resolve() if raw is not None else _binaries_dir()

    def _db_path(self, name: str) -> Path | None:
        root = self._db_root()
        return None if root is None else root / _DATABASES[name]

    def _bin_path(self, name: str) -> str | None:
        root = self._bin_root()
        if root is not None:
            return str(root / name)
        return shutil.which(name)

    def _needed_db(self, modality: str) -> tuple[str, ...]:
        if modality == "protein":
            return _PROTEIN_DB + (("seqres", "mmcif") if self.settings["templates"] else ())
        if modality == "rna":
            return _RNA_DB
        raise MSAConfigError(f"unsupported modality: {modality}")

    def _needed_bins(self, modality: str) -> tuple[str, ...]:
        if modality == "protein":
            return ("jackhmmer", "hmmsearch", "hmmbuild") if self.settings["templates"] else ("jackhmmer",)
        if modality == "rna":
            return ("nhmmer", "hmmalign", "hmmbuild")
        raise MSAConfigError(f"unsupported modality: {modality}")

    def _resources(self, modality: str) -> dict[str, Any]:
        databases = {}
        missing_db = []
        for name in self._needed_db(modality):
            path = self._db_path(name)
            ok = path is not None and path.exists()
            item = {"path": str(path) if path is not None else None, "exists": bool(ok)}
            if ok:
                try:
                    stat = path.stat()
                    item.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
                except OSError:
                    ok = False
            if not ok:
                missing_db.append(name)
            databases[name] = item
        binaries = {}
        missing_bin = []
        for name in self._needed_bins(modality):
            path = self._bin_path(name)
            ok = path is not None and Path(path).is_file() and os.access(path, os.X_OK)
            binaries[name] = {"path": path, "exists": bool(ok)}
            if not ok:
                missing_bin.append(name)
        return {
            "databases": databases,
            "binaries": binaries,
            "missing_databases": missing_db,
            "missing_binaries": missing_bin,
            "resources_ready": not missing_db and not missing_bin,
        }

    def _native_ready(self) -> bool:
        if self._native_probe is not None:
            return self._native_probe
        source = self.source_root / "src"
        code = (
            "import alphafold3.data.pipeline\n"
            "from alphafold3.common import folding_input\n"
            "folding_input.ProteinChain(id='q', sequence='A', ptms=[])\n"
            "folding_input.RnaChain(id='q', sequence='A', modifications=[])\n"
        )
        if not source.is_dir():
            self._native_probe = False
            return False
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
            JAX_PLATFORM_NAME="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false",
        )
        try:
            result = subprocess.run(
                [sys.executable, "-B", "-c", code],
                cwd=self.repo_root,
                env={
                    **env,
                    "PYTHONPATH": str(source),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            self._native_probe = False
        else:
            self._native_probe = result.returncode == 0
        return self._native_probe

    def _connect(self, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            if not self.index_path.is_file():
                raise OSError(f"missing cache index: {self.index_path}")
            uri = "file:" + quote(str(self.index_path), safe="/") + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=30)
        else:
            conn = sqlite3.connect(self.index_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
        conn.row_factory = sqlite3.Row
        return conn

    def _schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_profile_text TEXT NOT NULL UNIQUE,
                    profile_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    modality TEXT NOT NULL,
                    exact_query TEXT NOT NULL,
                    profile_id INTEGER NOT NULL REFERENCES profiles(id),
                    state TEXT NOT NULL DEFAULT 'missing',
                    ready_generation INTEGER,
                    failure_kind TEXT,
                    failure_message TEXT,
                    updated_at REAL NOT NULL,
                    UNIQUE(modality, exact_query, profile_id)
                );
                CREATE TABLE IF NOT EXISTS generations (
                    entry_id INTEGER NOT NULL REFERENCES entries(id),
                    generation_id INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    dir_name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(entry_id, generation_id),
                    UNIQUE(entry_id, dir_name)
                );
                """
            )

    @staticmethod
    def _tool_version(path: str | None) -> str:
        if path is None or not Path(path).is_file():
            return "missing"
        try:
            result = subprocess.run(
                [path, "-h"], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return "unavailable"
        output = result.stdout or ""
        for line in output.splitlines():
            if re.match(r"^\s*#\s*HMMER\b", line):
                return line.strip()[:256]
        lines = output.strip().splitlines()
        return lines[0][:256] if lines else f"exit:{result.returncode}"

    def _source_version(self) -> str:
        """Read the selected AF3 source version without importing its model."""
        source = self.source_root
        pyproject = source / "pyproject.toml"
        try:
            payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return "unknown"
        project = payload.get("project")
        if isinstance(project, Mapping):
            version = project.get("version")
            if isinstance(version, str) and version.strip():
                return version.strip()
        tool = payload.get("tool")
        if isinstance(tool, Mapping):
            setuptools_scm = tool.get("setuptools_scm")
            if isinstance(setuptools_scm, Mapping):
                fallback = setuptools_scm.get("fallback_version")
                if isinstance(fallback, str) and fallback.strip():
                    return fallback.strip()
        return "unknown"

    def _profile(self) -> dict[str, Any]:
        source_version = self._source_version()
        dbs = {}
        for name in _DATABASES:
            path = self._db_path(name)
            item = {
                "path": str(path.resolve()) if path is not None else None,
                "release": self.settings["database_revision"] or _DATABASES[name],
            }
            if path is not None and path.exists():
                try:
                    stat = path.stat()
                    item.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
                except OSError as exc:
                    item.update(size=None, mtime_ns=None, stat_error=str(exc))
            else:
                item.update(size=None, mtime_ns=None)
            dbs[name] = item
        tools = {
            name: {"path": self._bin_path(name), "version": self._tool_version(self._bin_path(name))}
            for name in _BINARIES
        }
        return {
            "schema_version": _SCHEMA,
            "adapter_version": "af3_msa_cache_v1",
            "source_api": "alphafold3.data.pipeline.DataPipeline",
            "af3_version": source_version,
            "tool_versions": tools,
            "databases": dbs,
            "database_revision": self.settings["database_revision"],
            "result_params": copy.deepcopy(_SEARCH_PARAMETERS),
            "template_settings": {
                "enabled": self.settings["templates"],
                "max_template_date": self.settings["max_template_date"],
            },
        }

    def _profile_id(self, payload: Mapping[str, Any], create: bool) -> int | None:
        text = _json(payload)
        try:
            with self._connect(readonly=not create) as conn:
                if create:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute(
                        "INSERT OR IGNORE INTO profiles(canonical_profile_text, profile_json, created_at)"
                        " VALUES (?, ?, ?)", (text, text, time.time())
                    )
                    row = conn.execute(
                        "SELECT id FROM profiles WHERE canonical_profile_text=?", (text,)
                    ).fetchone()
                    conn.commit()
                else:
                    row = conn.execute(
                        "SELECT id FROM profiles WHERE canonical_profile_text=?", (text,)
                    ).fetchone()
        except (OSError, sqlite3.Error):
            return None
        if row is None:
            return None
        profile_id = int(row["id"])
        if create:
            directory = self.cache_dir / "profiles" / f"profile{profile_id:06d}"
            directory.mkdir(parents=True, exist_ok=True)
            manifest = directory / "manifest.json"
            if not manifest.exists():
                self._atomic(manifest, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return profile_id

    def _selected_profile(self) -> int | None:
        selector = self.settings["cache_profile"]
        if selector is None:
            return None
        try:
            with self._connect(readonly=True) as conn:
                if isinstance(selector, int) and not isinstance(selector, bool):
                    row = conn.execute("SELECT id FROM profiles WHERE id=?", (selector,)).fetchone()
                else:
                    value = str(selector).strip()
                    row = conn.execute(
                        "SELECT id FROM profiles WHERE canonical_profile_text=?", (value,)
                    ).fetchone()
                    if row is None and value.isdigit():
                        row = conn.execute("SELECT id FROM profiles WHERE id=?", (int(value),)).fetchone()
                    if row is None and value.startswith("profile") and value[7:].isdigit():
                        row = conn.execute("SELECT id FROM profiles WHERE id=?", (int(value[7:]),)).fetchone()
                    if row is None:
                        manifest = Path(value).expanduser()
                        if not manifest.is_absolute():
                            manifest = asset_root(self.repo_root) / manifest
                        if manifest.is_dir():
                            manifest = manifest / "manifest.json"
                        if manifest.is_file():
                            try:
                                payload = json.loads(manifest.read_text())
                            except (OSError, ValueError):
                                payload = None
                            if isinstance(payload, Mapping):
                                row = conn.execute(
                                    "SELECT id FROM profiles WHERE canonical_profile_text=?",
                                    (_json(payload),),
                                ).fetchone()
        except (OSError, sqlite3.Error):
            return None
        return None if row is None else int(row["id"])

    def _compatible_profile(self, payload: Any, modality: str) -> bool:
        """Compare search semantics without consulting this machine's resources.

        Paths, mtimes, tool installations and SDK version are provenance. A
        completed generation already contains its alignment and template data.
        An explicitly requested database revision remains a compatibility gate.
        """
        if not isinstance(payload, Mapping):
            return False
        if (payload.get("schema_version") != _SCHEMA
                or payload.get("adapter_version") != "af3_msa_cache_v1"
                or payload.get("source_api") != "alphafold3.data.pipeline.DataPipeline"):
            return False
        revision = self.settings["database_revision"]
        if revision is not None and payload.get("database_revision") != revision:
            return False
        params = payload.get("result_params")
        if not isinstance(params, Mapping):
            return False
        for key, expected in _SEARCH_PARAMETERS.items():
            if (key == "deduplicate" or key.startswith(modality + "_")) and params.get(key) != expected:
                return False
        if modality == "protein":
            template_settings = payload.get("template_settings")
            if not isinstance(template_settings, Mapping):
                return False
            if template_settings.get("enabled") is not self.settings["templates"]:
                return False
            if (self.settings["templates"] and template_settings.get("max_template_date")
                    != self.settings["max_template_date"]):
                return False
        return True

    def _cached_candidates(self, modality: str, sequence: str | None = None) -> list[sqlite3.Row]:
        """Newest published, compatible generations; explicit profiles pin selection."""
        selected = self._selected_profile() if self.settings["cache_profile"] is not None else None
        if self.settings["cache_profile"] is not None and selected is None:
            return []
        clauses = ["e.modality=?", "g.state='published'"]
        args: list[Any] = [modality]
        if sequence is not None:
            clauses.append("e.exact_query=?")
            args.append(sequence)
        if selected is not None:
            clauses.append("e.profile_id=?")
            args.append(selected)
        try:
            with self._connect(readonly=True) as conn:
                rows = conn.execute(
                    "SELECT e.id, e.profile_id, e.exact_query, p.profile_json "
                    "FROM entries e JOIN profiles p ON p.id=e.profile_id "
                    "JOIN generations g ON g.entry_id=e.id AND g.generation_id=e.ready_generation "
                    "WHERE " + " AND ".join(clauses) +
                    " ORDER BY g.created_at DESC, e.id DESC", args,
                ).fetchall()
        except (OSError, sqlite3.Error):
            return []
        candidates = []
        for row in rows:
            try:
                payload = json.loads(row["profile_json"])
            except (TypeError, ValueError):
                continue
            if self._compatible_profile(payload, modality):
                candidates.append(row)
        return candidates

    @staticmethod
    def _atomic(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_temporary = tempfile.mkstemp(
            prefix=path.name + ".tmp-", suffix=".partial", dir=path.parent
        )
        temporary = Path(raw_temporary)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @contextlib.contextmanager
    def _lock(self, entry_id: int):
        if self.readonly:
            raise MSAReadOnlyError("cache is read-only")
        path = self.lock_dir / f"entry{entry_id:06d}.lock"
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        with path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _entry(self, modality: str, sequence: str, profile_id: int) -> int:
        if self.readonly:
            raise MSAReadOnlyError("cache is read-only")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT OR IGNORE INTO entries(modality, exact_query, profile_id, state, updated_at)"
                " VALUES (?, ?, ?, 'missing', ?)",
                (modality, sequence, profile_id, time.time()),
            )
            row = conn.execute(
                "SELECT id FROM entries WHERE modality=? AND exact_query=? AND profile_id=?",
                (modality, sequence, profile_id),
            ).fetchone()
            conn.commit()
        if row is None:
            raise MSAError("cache entry allocation failed")
        return int(row["id"])

    def _reference(self, entry_id: int, profile_id: int | None = None) -> dict[str, Any] | None:
        try:
            with self._connect(readonly=True) as conn:
                entry = conn.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
                if entry is None or entry["ready_generation"] is None:
                    return None
                if profile_id is not None and int(entry["profile_id"]) != profile_id:
                    return None
                profile = conn.execute("SELECT canonical_profile_text FROM profiles WHERE id=?", (int(entry["profile_id"]),)).fetchone()
                generation = conn.execute(
                    "SELECT * FROM generations WHERE entry_id=? AND generation_id=?",
                    (entry_id, int(entry["ready_generation"])),
                ).fetchone()
        except (OSError, sqlite3.Error):
            return None
        if profile is None or generation is None or generation["state"] != "published":
            return None
        modality = str(entry["modality"])
        base = self.cache_dir / modality / f"entry{entry_id:06d}"
        directory = (base / str(generation["dir_name"])).resolve()
        if directory.parent != base.resolve() or not directory.is_dir():
            return None
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
        except (OSError, ValueError):
            return None
        if (
            not isinstance(manifest, Mapping)
            or type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != _SCHEMA
        ):
            return None
        if any(
            (
                manifest.get("status") != "ready",
                manifest.get("entry_id") != entry_id,
                manifest.get("generation_id") != int(entry["ready_generation"]),
                manifest.get("profile_id") != int(entry["profile_id"]),
                manifest.get("modality") != modality,
                manifest.get("exact_query") != entry["exact_query"],
            )
        ):
            return None

        def file_for(name: str) -> Path | None:
            path = (directory / name).resolve()
            return path if path.parent == directory and path.is_file() else None

        unpaired = file_for(str(manifest.get("unpaired_msa", "")))
        paired = file_for(str(manifest.get("paired_msa", ""))) if modality == "protein" else None
        if unpaired is None or (modality == "protein" and paired is None):
            return None
        try:
            _validate_a3m(unpaired.read_text(), str(entry["exact_query"]), "unpaired")
            if paired is not None:
                _validate_a3m(paired.read_text(), str(entry["exact_query"]), "paired")
        except (OSError, UnicodeError, MSASearchError):
            return None
        templates = []
        template_items = manifest.get("templates", [])
        if not isinstance(template_items, list):
            return None
        for item in template_items:
            if not isinstance(item, Mapping) or not isinstance(item.get("query_to_template_map"), Mapping):
                return None
            path = file_for(str(item.get("mmcif", "")))
            if path is None:
                return None
            try:
                mapping = {int(key): int(value) for key, value in item["query_to_template_map"].items()}
            except (TypeError, ValueError):
                return None
            if any(key < 0 or key >= len(entry["exact_query"]) or value < 0
                   for key, value in mapping.items()):
                return None
            templates.append({"mmcif_path": str(path), "query_to_template_map": mapping})
        return {
            "modality": modality,
            "sequence": str(entry["exact_query"]),
            "exact_query": str(entry["exact_query"]),
            "entry_id": entry_id,
            "profile_id": int(entry["profile_id"]),
            "profile_text": profile["canonical_profile_text"],
            "cache_dir": str(self.cache_dir),
            "generation_id": int(entry["ready_generation"]),
            "generation": int(entry["ready_generation"]),
            "unpaired_msa_path": str(unpaired),
            **({"paired_msa_path": str(paired)} if paired is not None else {}),
            "templates": templates,
            "provenance": "native-searched",
            "status": "ready",
            "no_hit": bool(manifest.get("no_hit", False)),
        }

    def _generation_indexed(
        self, entry_id: int, generation_id: int, dir_name: str
    ) -> bool | None:
        """Return whether a generation directory has a durable index row."""
        try:
            with self._connect(readonly=True) as conn:
                row = conn.execute(
                    "SELECT state, dir_name FROM generations "
                    "WHERE entry_id=? AND generation_id=?",
                    (entry_id, generation_id),
                ).fetchone()
        except Exception:
            # If the index cannot be inspected, retaining the directory is the
            # safe choice: it may already be referenced by a published row.
            return None
        return bool(
            row is not None
            and row["state"] == "published"
            and row["dir_name"] == dir_name
        )

    def _reconcile_entry(self, entry_id: int) -> None:
        """Remove only generation directories absent from the entry index."""
        try:
            with self._connect(readonly=True) as conn:
                indexed = {
                    int(row["generation_id"]): str(row["dir_name"])
                    for row in conn.execute(
                        "SELECT generation_id, dir_name FROM generations WHERE entry_id=?",
                        (entry_id,),
                    ).fetchall()
                }
        except Exception:
            return
        for modality in _MODALITIES:
            base = self.cache_dir / modality / f"entry{entry_id:06d}"
            if not base.is_dir():
                continue
            try:
                children = tuple(base.iterdir())
            except OSError:
                continue
            for child in children:
                match = re.fullmatch(r"generation([0-9]+)", child.name)
                if match is None or not child.is_dir() or child.is_symlink():
                    continue
                generation_id = int(match.group(1))
                if indexed.get(generation_id) == child.name:
                    continue
                # The directory is outside the durable index.  Never remove a
                # directory that any generation row names, including rows in a
                # non-published state.
                if child.name in indexed.values():
                    continue
                shutil.rmtree(child, ignore_errors=True)

    def lookup(self, modality: str, sequence: str) -> dict[str, Any] | None:
        if modality not in _MODALITIES:
            raise MSAConfigError(f"unsupported modality: {modality}")
        sequence = _sequence(sequence)
        for row in self._cached_candidates(modality, sequence):
            reference = self._reference(int(row["id"]), int(row["profile_id"]))
            if reference is not None:
                return reference
        return None

    def _native_search(self, modality: str, sequence: str) -> dict[str, Any]:
        resources = self._resources(modality)
        if not resources["resources_ready"]:
            raise MSAResourceError(
                f"{modality} resources unavailable: "
                f"databases={resources['missing_databases']} "
                f"binaries={resources['missing_binaries']}"
            )
        source = self.source_root / "src"
        if not source.is_dir():
            raise MSAResourceError(f"AF3 source tree missing: {source}")
        if self.readonly:
            raise MSAReadOnlyError("native search is disabled read-only")
        paths = {}
        for name in _DATABASES:
            value = self._db_path(name)
            paths[name] = str(value) if value is not None else ""
        for name in _BINARIES:
            paths[name] = self._bin_path(name) or ""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f"native-{modality}-", dir=self.work_dir))
        request = temporary / "request.json"
        output = temporary / "result.json"
        payload = {
            "source_root": str(source), "modality": modality, "sequence": sequence,
            "templates": bool(self.settings["templates"]),
            "max_template_date": self.settings["max_template_date"],
            "jackhmmer_n_cpu": self.settings["jackhmmer_n_cpu"],
            "nhmmer_n_cpu": self.settings["nhmmer_n_cpu"], "paths": paths,
        }
        self._atomic(request, json.dumps(payload, sort_keys=True))
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
            JAX_PLATFORM_NAME="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false",
            PYTHONPATH=str(source), TMPDIR=str(temporary),
            PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        )
        process = subprocess.Popen(
            [sys.executable, "-c", _NATIVE_SOURCE, str(request), str(output)],
            cwd=self.repo_root, env=env, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        old_handlers: dict[int, Any] = {}

        def cancel_child(signum: int, _frame: Any) -> None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except OSError:
                pass
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                old_handlers[signum] = signal.signal(signum, cancel_child)
            except (ValueError, OSError):
                pass
        try:
            try:
                stdout, stderr = process.communicate(timeout=float(self.settings["timeout_seconds"]))
            except subprocess.TimeoutExpired as exc:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    process.communicate()
                raise MSASearchTimeout(
                    f"{modality} native search timed out after {self.settings['timeout_seconds']} seconds"
                ) from exc
            if process.returncode:
                detail = (stderr or stdout or "").strip()[-8000:]
                raise MSASearchError(f"native search exited {process.returncode}: {detail}")
            try:
                result = json.loads(output.read_text())
            except (OSError, ValueError) as exc:
                raise MSASearchError("native search result was not published") from exc
            if not isinstance(result, Mapping):
                raise MSASearchError("native search result must be a mapping")
            return dict(result)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    process.communicate()
            for signum, handler in old_handlers.items():
                try:
                    signal.signal(signum, handler)
                except (ValueError, OSError):
                    pass
            shutil.rmtree(temporary, ignore_errors=True)

    def _normal_result(
        self, modality: str, sequence: str, result: Mapping[str, Any]
    ) -> tuple[str, str | None, list[dict[str, Any]], bool]:
        if result.get("modality", modality) != modality:
            raise MSASearchError("native result modality differs from request")
        if result.get("sequence", sequence) != sequence:
            raise MSASearchError("native result query differs from requested query")
        unpaired = result.get("unpaired_msa")
        if not isinstance(unpaired, str):
            raise MSASearchError("native result has no unpaired MSA")
        unpaired_depth = _validate_a3m(unpaired, sequence, "unpaired")
        paired = None
        paired_depth = 1
        if modality == "protein":
            paired = result.get("paired_msa")
            if not isinstance(paired, str):
                raise MSASearchError("native protein result has no paired MSA")
            paired_depth = _validate_a3m(paired, sequence, "paired")
        elif result.get("paired_msa") not in (None, ""):
            raise MSASearchError("RNA native result unexpectedly has paired MSA")
        raw_templates = result.get("templates", []) or []
        if not self.settings["templates"]:
            raw_templates = []
        if not isinstance(raw_templates, Sequence) or isinstance(raw_templates, (str, bytes)):
            raise MSASearchError("native templates must be a sequence")
        templates = []
        for item in raw_templates:
            if not isinstance(item, Mapping) or not isinstance(item.get("mmcif"), str):
                raise MSASearchError("native template is malformed")
            mapping = item.get("query_to_template_map", {})
            if not isinstance(mapping, Mapping):
                raise MSASearchError("native template map is malformed")
            try:
                clean = {int(key): int(value) for key, value in mapping.items()}
            except (TypeError, ValueError) as exc:
                raise MSASearchError("native template map is not integral") from exc
            templates.append({"mmcif": item["mmcif"], "query_to_template_map": clean})
        no_hit = bool(result.get("no_hit", unpaired_depth == 1 and paired_depth == 1 and not templates))
        return unpaired, paired, templates, no_hit

    def _publish(
        self, entry_id: int, profile_id: int, modality: str, sequence: str,
        unpaired: str, paired: str | None, templates: Sequence[Mapping[str, Any]], no_hit: bool,
    ) -> dict[str, Any]:
        if self.readonly:
            raise MSAReadOnlyError("cannot publish read-only cache")
        base = self.cache_dir / modality / f"entry{entry_id:06d}"
        base.mkdir(parents=True, exist_ok=True)
        maximum = 0
        if base.is_dir():
            for item in base.iterdir():
                match = re.fullmatch(r"generation([0-9]+)", item.name)
                if match:
                    maximum = max(maximum, int(match.group(1)))
        try:
            with self._connect(readonly=True) as conn:
                row = conn.execute(
                    "SELECT MAX(generation_id) FROM generations WHERE entry_id=?", (entry_id,)
                ).fetchone()
                if row and row[0] is not None:
                    maximum = max(maximum, int(row[0]))
        except (OSError, sqlite3.Error):
            pass
        generation_id = maximum + 1
        stage = Path(tempfile.mkdtemp(prefix=f"entry{entry_id:06d}-", dir=self.work_dir))
        final: Path | None = None
        final_moved = False
        try:
            self._atomic(stage / "unpaired.a3m", unpaired)
            manifest: dict[str, Any] = {
                "schema_version": _SCHEMA, "status": "ready", "modality": modality,
                "exact_query": sequence, "entry_id": entry_id, "profile_id": profile_id,
                "generation_id": generation_id, "unpaired_msa": "unpaired.a3m",
                "templates": [], "no_hit": no_hit, "provenance": "native-searched",
            }
            if modality == "protein":
                if paired is None:
                    raise MSASearchError("protein publication requires paired MSA")
                self._atomic(stage / "paired.a3m", paired)
                manifest["paired_msa"] = "paired.a3m"
            for index, item in enumerate(templates, 1):
                filename = f"template{index:06d}.mmcif"
                self._atomic(stage / filename, str(item["mmcif"]))
                manifest["templates"].append({
                    "mmcif": filename,
                    "query_to_template_map": {
                        str(key): int(value) for key, value in item["query_to_template_map"].items()
                    },
                })
            self._atomic(stage / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            final = base / f"generation{generation_id:06d}"
            os.replace(stage, final)
            final_moved = True
            try:
                with self._connect() as conn:
                    conn.execute(
                        "INSERT INTO generations(entry_id, generation_id, state, dir_name, created_at)"
                        " VALUES (?, ?, 'published', ?, ?)",
                        (entry_id, generation_id, final.name, time.time()),
                    )
                    conn.execute(
                        "UPDATE entries SET state='ready', ready_generation=?,"
                        " failure_kind=NULL, failure_message=NULL, updated_at=? WHERE id=?",
                        (generation_id, time.time(), entry_id),
                    )
            except BaseException:
                # The directory became visible before the index transaction.
                # Remove it only when a read confirms that no published row
                # names it; an already-indexed generation is durable evidence
                # that it must be preserved.
                if (
                    final_moved
                    and final is not None
                    and self._generation_indexed(entry_id, generation_id, final.name) is False
                    and final.is_dir()
                    and not final.is_symlink()
                ):
                    shutil.rmtree(final, ignore_errors=True)
                raise
            reference = self._reference(entry_id, profile_id)
            if reference is None:
                raise MSAError("published generation was not readable")
            return reference
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    def _failure(self, entry_id: int, error: BaseException, had_ready: bool) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE entries SET state=?, failure_kind=?, failure_message=?, updated_at=? WHERE id=?",
                    ("ready" if had_ready else "failed", type(error).__name__, str(error)[:4000], time.time(), entry_id),
                )
        except sqlite3.Error:
            pass

    def ensure(self, modality: str, sequence: str) -> dict[str, Any] | None:
        if modality not in _MODALITIES:
            raise MSAConfigError(f"unsupported modality: {modality}")
        sequence = _sequence(sequence)
        mode = self.settings["mode"]
        if mode == "off":
            return None
        if mode == "cache-only":
            result = self.lookup(modality, sequence)
            if result is None:
                raise MSACacheMiss(f"no ready cache entry for {modality} query")
            return result
        if not self.settings["refresh"]:
            result = self.lookup(modality, sequence)
            if result is not None:
                return result
        if self.readonly:
            raise MSAReadOnlyError("auto ensure cannot search read-only")
        profile_id = self._profile_id(self._profile(), create=True)
        if profile_id is None:
            raise MSAError("could not allocate profile")
        entry_id = self._entry(modality, sequence, profile_id)
        with self._lock(entry_id):
            self._reconcile_entry(entry_id)
            ready = self._reference(entry_id, profile_id)
            if ready is not None and not self.settings["refresh"]:
                return ready
            try:
                result = self._native_search(modality, sequence)
                unpaired, paired, templates, no_hit = self._normal_result(modality, sequence, result)
                return self._publish(
                    entry_id, profile_id, modality, sequence,
                    unpaired, paired, templates, no_hit,
                )
            except MSAError as exc:
                self._failure(entry_id, exc, ready is not None)
                raise
            except Exception as exc:
                self._failure(entry_id, exc, ready is not None)
                raise MSASearchError(str(exc)) from exc

    def _ready_entries(self, profile_id: int, modality: str) -> tuple[int, int] | None:
        """Count indexed and readable ready entries for one profile/modality."""
        try:
            with self._connect(readonly=True) as conn:
                rows = conn.execute(
                    "SELECT id FROM entries "
                    "WHERE profile_id=? AND modality=? AND ready_generation IS NOT NULL",
                    (profile_id, modality),
                ).fetchall()
        except (OSError, sqlite3.Error):
            return None
        valid = sum(
            self._reference(int(row["id"]), profile_id) is not None for row in rows
        )
        return len(rows), valid

    def check(self, modalities: Sequence[str] | str | None = None) -> dict[str, Any]:
        if modalities is None:
            wanted = list(_MODALITIES)
        elif isinstance(modalities, str):
            wanted = [modalities]
        else:
            wanted = list(modalities)
        if not wanted or any(item not in _MODALITIES for item in wanted):
            raise MSAConfigError(f"unsupported modalities: {wanted}")
        wanted = list(dict.fromkeys(wanted))
        index_exists = self.index_path.is_file()
        cache: dict[str, Any] = {
            "path": str(self.cache_dir), "index_path": str(self.index_path),
            "index_exists": index_exists,
            "readable": os.access(self.index_path, os.R_OK) if index_exists else False,
        }
        if index_exists:
            try:
                with self._connect(readonly=True) as conn:
                    cache["profiles"] = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
            except (OSError, sqlite3.Error) as exc:
                cache["readable"] = False
                cache["error"] = str(exc)
        cache["ready_entries"] = 0
        cache["ready_entries_by_modality"] = {}
        result: dict[str, Any] = {
            "mode": self.settings["mode"], "readonly": self.readonly,
            "templates": self.settings["templates"],
            "max_template_date": self.settings["max_template_date"],
            "cache": cache, "modalities": {}, "query_availability": "unknown",
        }
        if self.settings["mode"] == "off":
            for item in wanted:
                report = {"ready": True, "reason": "mode off"}
                result["modalities"][item] = report
                result[item] = report
            result["ready"] = True
            return result
        selector = self.settings["cache_profile"]
        native_ready = None
        for item in wanted:
            candidates = self._cached_candidates(item)
            seen, profiles = set(), set()
            for row in candidates:
                if row["exact_query"] in seen:
                    continue
                if self._reference(int(row["id"]), int(row["profile_id"])) is not None:
                    seen.add(row["exact_query"])
                    profiles.add(int(row["profile_id"]))
            report = {
                "cache_profile": selector,
                "selected_profile_id": next(iter(profiles)) if len(profiles) == 1 else None,
                "compatible_profile_ids": sorted(profiles),
                "indexed_ready_entries": len(candidates), "ready_entries": len(seen),
                "query_availability": "unknown", "native_import": None,
            }
            cache["ready_entries"] += len(seen)
            cache["ready_entries_by_modality"][item] = len(seen)
            if seen:
                report.update(ready=True, search_resources_checked=False,
                              reason="readable compatible cache entries available; uncached queries need search resources")
            elif self.settings["mode"] == "cache-only":
                report.update(ready=False, search_resources_checked=False,
                              reason="no readable compatible cache entries for modality")
            else:
                report.update(self._resources(item))
                if report["resources_ready"] and native_ready is None:
                    native_ready = self._native_ready()
                report["native_import"] = native_ready
                report["search_resources_checked"] = True
                report["ready"] = bool(report["resources_ready"] and native_ready)
            result["modalities"][item] = report
            result[item] = report
        result["ready"] = all(result["modalities"][item]["ready"] for item in wanted)
        return result


__all__ = [
    "AF3MSACache", "DEFAULT_MSA_CONFIG", "MSACacheMiss", "MSAConfigError",
    "MSAError", "MSAReadOnlyError", "MSAResourceError", "MSASearchError",
    "MSASearchTimeout", "resolve_msa_settings", "validate_msa_settings",
]
