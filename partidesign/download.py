"""Catalog and safely provision local PartiDesign model resources.

The module is intentionally stdlib-only: model implementations, dependency
installation, and source checkouts belong to the repository/environment. A
resource with no verified primary-source URL or supported local derivation
remains unavailable until a complete local copy is present.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
import contextvars
import gzip
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterator
from urllib.request import urlopen


_DOWNLOAD_TIMEOUT = 60
_CHUNK_SIZE = 1024 * 1024
_ARIA_CONNECTIONS = 8
_ARIA_CONNECT_TIMEOUT = 15
_ARIA_RETRIES = 3
_ARIA_RETRY_WAIT = 2
_ARIA_TAIL_LINES = 32
_ARIA_TAIL_CHARS = 8192

_ARIA_PROGRESS_RE = re.compile(
    r"(?<![0-9])(?P<completed>[0-9][0-9,]*)\s*B?\s*/\s*"
    r"(?P<total>[0-9][0-9,]*)\s*B?(?=\s*(?:\(|\]|$))"
)
_ARIA_GID_RE = re.compile(r"\[#(?:[0-9A-Fa-f]+)\]\s*")


@dataclass(frozen=True, slots=True)
class _TransferContext:
    label: str


_PROGRESS: contextvars.ContextVar[Callable[[dict[str, object]], object] | None] = (
    contextvars.ContextVar("partidesign_download_progress", default=None)
)
_TRANSFER: contextvars.ContextVar[_TransferContext | None] = contextvars.ContextVar(
    "partidesign_download_transfer", default=None
)


class _AriaFailure(RuntimeError):
    """aria2c failed in a way that can be retried with urllib."""


class _ProgressCallbackError(RuntimeError):
    """A progress consumer failed and must not trigger an aria fallback."""


@dataclass(frozen=True, slots=True)
class _AssetSpec:
    relative: str
    url: str | None = None
    archive_member: str | None = None
    gzip: bool = False


@dataclass(frozen=True, slots=True)
class _TargetSpec:
    name: str
    assets: tuple[_AssetSpec, ...]
    archive_url: str | None = None
    with_git_repo: bool = False
    packaged_resource: bool = False

    @property
    def source_kind(self) -> str:
        if self.with_git_repo:
            return "git_repo"
        return "download_url" if self.has_source else "unconfigured"

    @property
    def has_source(self) -> bool:
        return self.archive_url is not None or any(asset.url for asset in self.assets)


def _asset(relative: str, url: str | None = None) -> _AssetSpec:
    return _AssetSpec(relative=relative, url=url)


def _direct_assets(prefix: str, names: Iterable[str], base_url: str) -> tuple[_AssetSpec, ...]:
    return tuple(_asset(f"{prefix}/{name}", f"{base_url}/{name}") for name in names)


def _mapped_assets(
    prefix: str, mapping: Iterable[tuple[str, str]], base_url: str
) -> tuple[_AssetSpec, ...]:
    return tuple(
        _asset(f"{prefix}/{destination}", f"{base_url}/{source}")
        for destination, source in mapping
    )


# Official ProteinMPNN asset commit 8907e66 contains the vanilla and CA weight filenames used by the inventory.
_PROTEINMPNN_URL = "https://raw.githubusercontent.com/dauparas/ProteinMPNN/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
_LIGANDMPNN_URL = "https://files.ipd.uw.edu/pub/ligandmpnn"
_NA_MPNN_URL = "https://raw.githubusercontent.com/baker-laboratory/NA-MPNN/9fabc2482092b725e067969fba21297a806b6fda/models/design_model/s_19137.pt"
_ALPHAFOLD_PARAMS_URL = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"

_PROTEINMPNN_VANILLA = ("v_48_002.pt", "v_48_010.pt", "v_48_020.pt", "v_48_030.pt")
_PROTEINMPNN_CA = ("v_48_002.pt", "v_48_010.pt", "v_48_020.pt")
_SOLUBLEMPNN = (
    "solublempnn_v_48_002.pt",
    "solublempnn_v_48_010.pt",
    "solublempnn_v_48_020.pt",
    "solublempnn_v_48_030.pt",
)
_LIGANDMPNN = (
    "ligandmpnn_v_32_005_25.pt",
    "ligandmpnn_v_32_010_25.pt",
    "ligandmpnn_v_32_020_25.pt",
    "ligandmpnn_v_32_030_25.pt",
)


def _af2_assets() -> tuple[_AssetSpec, ...]:
    names = tuple(
        f"params_model_{model}{suffix}.npz"
        for model in range(1, 6)
        for suffix in ("", "_ptm", "_multimer_v3")
    )
    return tuple(
        _AssetSpec(
            relative=f"community_models/colabdesign/checkpoint/af2/{name}",
            archive_member=name,
        )
        for name in names
    )


_ESMFOLD2_BASES = {
    "fast": "https://huggingface.co/biohub/ESMFold2-Fast/resolve/c6c7958d63f5f2f1f0fed0bb9462316f8ccceea6",
    "standard": "https://huggingface.co/biohub/ESMFold2/resolve/8fc3ff471022fdce52c77030685eb775de0c00a3",
}
_ESMFOLD2_CIF = "community_models/esmfold2/common/components.cif"
_ESMFOLD2_BONDS = "community_models/esmfold2/common/ccd_bonds.sqlite3"
_ESMFOLD2_CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"

_ESMC6B_BASE = "https://huggingface.co/biohub/ESMC-6B/resolve/45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"


def _esmfold2_assets(mode: str) -> tuple[_AssetSpec, ...]:
    root = f"community_models/esmfold2/checkpoint/{mode}"
    common = "community_models/esmfold2/common"
    esmc = "community_models/esmfold2/checkpoint/esmc-6b"
    assets = [
        *_direct_assets(root, ("config.json", "model.safetensors"), _ESMFOLD2_BASES[mode]),
        _asset(f"{common}/ccd.pkl", f"{_ESMFOLD2_BASES['standard']}/ccd.pkl"),
        # The source dictionary is downloaded; its bond cache is derived locally.
        _AssetSpec(_ESMFOLD2_CIF, _ESMFOLD2_CCD_URL, gzip=True),
        _asset(_ESMFOLD2_BONDS),
    ]
    assets.extend(_direct_assets(esmc, (
        "config.json", "model.safetensors.index.json", "special_tokens_map.json",
        "tokenizer_config.json", "tokenizer.json",
        *(f"model-{index:05d}-of-00006.safetensors" for index in range(1, 7)),
    ), _ESMC6B_BASE))
    return tuple(assets)


_CATALOG: dict[str, _TargetSpec] = {
    "partidesign": _TargetSpec("partidesign", (_asset("ckpts/partidesign.ckpt"),)),
    "partiscore": _TargetSpec(
        "partiscore", (_asset("ckpts/partiscore.ckpt"),), with_git_repo=True
    ),
    "proteinmpnn": _TargetSpec(
        "proteinmpnn",
        (
            *_direct_assets(
                "community_models/proteinmpnn/checkpoint/vanilla",
                _PROTEINMPNN_VANILLA,
                f"{_PROTEINMPNN_URL}/vanilla_model_weights",
            ),
            *_direct_assets(
                "community_models/proteinmpnn/checkpoint/ca",
                _PROTEINMPNN_CA,
                f"{_PROTEINMPNN_URL}/ca_model_weights",
            ),
        ),
    ),
    "solublempnn": _TargetSpec(
        "solublempnn",
        _mapped_assets(
            "community_models/proteinmpnn/checkpoint/soluble",
            tuple((name.removeprefix("solublempnn_"), name) for name in _SOLUBLEMPNN),
            _LIGANDMPNN_URL,
        ),
    ),
    "ligandmpnn": _TargetSpec(
        "ligandmpnn",
        (
            *_direct_assets("community_models/ligandmpnn/checkpoint", _LIGANDMPNN, _LIGANDMPNN_URL),
            _asset(
                "community_models/ligandmpnn/checkpoint/ligandmpnn_sc_v_32_002_16.pt",
                f"{_LIGANDMPNN_URL}/ligandmpnn_sc_v_32_002_16.pt",
            ),
        ),
    ),
    "cyclicmpnn": _TargetSpec(
        "cyclicmpnn",
        (
            _asset("community_models/cyclicmpnn/checkpoint/cyclicmpnn_48_010.pkl"),
            _asset("community_models/colabdesign/source/colabdesign/mpnn/weights/v_48_020.pkl"),
        ),
        with_git_repo=True,
    ),
    "nampnn": _TargetSpec("nampnn", (_asset("community_models/na_mpnn/checkpoint/s_19137.pt", _NA_MPNN_URL),)),
    "esmfold2-fast": _TargetSpec("esmfold2-fast", _esmfold2_assets("fast")),
    "esmfold2-standard": _TargetSpec("esmfold2-standard", _esmfold2_assets("standard")),
    "af2": _TargetSpec("af2", _af2_assets(), archive_url=_ALPHAFOLD_PARAMS_URL),
    "af3": _TargetSpec(
        "af3",
        (
            _asset(
                "community_models/af3/checkpoints/af3.bin.zst",
                "https://storage.googleapis.com/alphafold3/af3.bin.zst",
            ),
            _asset(
                "community_models/af3/ccd/components.cif.gz",
                "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz",
            ),
        ),
    ),
    "af3-la": _TargetSpec(
        "af3-la",
        (
            _asset(
                "community_models/af3-la/checkpoints/af3_leaving_atom.bin.zst",
                "https://storage.googleapis.com/alphafold3/af3_leaving_atom.bin.zst",
            ),
            _asset(
                "community_models/af3/ccd/components.cif.gz",
                "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz",
            ),
        ),
    ),
    "chemistry": _TargetSpec(
        "chemistry", (_asset("configs/resources/chemistry/ccd.json.gz"),),
        with_git_repo=True, packaged_resource=True,
    ),
}

TARGETS = tuple(_CATALOG)


class _DownloadError(RuntimeError):
    """An expected resource could not be safely staged or published."""


def _root_path(root: Path) -> Path:
    return Path(root).expanduser()


def _relative_parts(relative: str) -> tuple[str, ...]:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise _DownloadError(f"unsafe catalog path: {relative!r}")
    if any("\\" in part or "\x00" in part for part in path.parts):
        raise _DownloadError(f"unsafe catalog path: {relative!r}")
    return tuple(path.parts)


def _destination(root: Path, relative: str) -> Path:
    parts = _relative_parts(relative)
    root = _root_path(root)
    resolved_root = root.resolve(strict=False)
    if root.exists() and root.is_symlink():
        raise _DownloadError(f"model root is a symlink: {root}")
    cursor = root
    for part in parts[:-1]:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise _DownloadError(f"catalog parent is a symlink: {cursor}")
    destination = root.joinpath(*parts)
    if destination.exists():
        if destination.is_symlink():
            raise _DownloadError(f"catalog destination is a symlink: {destination}")
        if not destination.is_file():
            raise _DownloadError(f"catalog destination is not a file: {destination}")
    try:
        destination.parent.resolve(strict=False).relative_to(resolved_root)
    except ValueError as exc:
        raise _DownloadError(f"catalog path escapes model root: {relative!r}") from exc
    return destination


def _safe_existing_file(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    except OSError:
        return False


def _target_root(root: Path, spec: _TargetSpec) -> Path:
    if spec.packaged_resource:
        from .paths import resource_root
        return resource_root()
    return root


def _missing_assets(root: Path, spec: _TargetSpec) -> tuple[list[_AssetSpec], str | None]:
    root = _target_root(root, spec)
    missing: list[_AssetSpec] = []
    unsafe: list[str] = []
    for asset in spec.assets:
        try:
            destination = _destination(root, asset.relative)
        except _DownloadError as exc:
            unsafe.append(str(exc))
            continue
        if not _safe_existing_file(destination):
            missing.append(asset)
    if unsafe:
        return missing, "; ".join(unsafe)
    return missing, None


def _record(spec: _TargetSpec, root: Path, status: str, error: str | None = None) -> dict[str, object]:
    root = _target_root(root, spec)
    paths: list[str] = []
    for asset in spec.assets:
        try:
            paths.append(str(_destination(root, asset.relative)))
        except _DownloadError:
            # Keep reporting the catalog text even when a caller supplied an
            # unsafe root or a future catalog entry needs review.
            paths.append(str(_root_path(root) / asset.relative))
    return {
        "target": spec.name,
        "status": status,
        "error": error,
        "paths": paths,
        "source_kind": spec.source_kind,
        "sources": list(dict.fromkeys(
            ([spec.archive_url] if spec.archive_url else [])
            + [asset.url for asset in spec.assets if asset.url]
        )),
    }


def _af3_record(
    spec: _TargetSpec, root: Path, status: str, error: str | None = None
) -> dict[str, object]:
    from .af3_assets import required_relatives, target_sources

    paths: list[str] = []
    try:
        relatives = required_relatives(spec.name)
    except ValueError:
        relatives = tuple(asset.relative for asset in spec.assets)
    for relative in relatives:
        try:
            paths.append(str(_destination(root, relative)))
        except _DownloadError:
            paths.append(str(_root_path(root) / relative))
    sources = [spec.archive_url] if spec.archive_url else []
    sources.extend(asset.url for asset in spec.assets if asset.url)
    if not sources:
        sources.extend(target_sources(spec.name))
    return {
        "target": spec.name,
        "status": status,
        "error": error,
        "paths": paths,
        "source_kind": spec.source_kind,
        "sources": list(dict.fromkeys(sources)),
    }


def _af3_missing_assets(root: Path, target: str) -> tuple[list[str], str | None]:
    from .af3_assets import required_relatives

    missing: list[str] = []
    unsafe: list[str] = []
    for relative in required_relatives(target):
        try:
            destination = _destination(root, relative)
        except _DownloadError as exc:
            unsafe.append(str(exc))
            continue
        if not _safe_existing_file(destination):
            missing.append(relative)
    if unsafe:
        return missing, "; ".join(unsafe)
    return missing, None


def _format_error(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def list_targets(root: Path) -> list[dict[str, object]]:
    """Return local status for every catalog target without network access."""
    records: list[dict[str, object]] = []
    for name in TARGETS:
        spec = _CATALOG[name]
        af3 = name in {"af3", "af3-la"}
        missing, unsafe = _af3_missing_assets(root, name) if af3 else _missing_assets(root, spec)
        record = _af3_record if af3 else _record
        if unsafe:
            records.append(record(spec, root, "unavailable", unsafe))
        elif not missing:
            records.append(record(spec, root, "installed"))
        else:
            joined = ", ".join(
                asset if isinstance(asset, str) else asset.relative for asset in missing
            )
            detail = f"missing expected assets: {joined}"
            if spec.with_git_repo:
                detail += "; supplied with git repo; copy or restore these files from a PartiDesign checkout"
            records.append(record(spec, root, "unavailable", detail))
    return records


def selection_options(root: Path, *, force: bool = False) -> list[dict[str, object]]:
    """Describe selectable targets using the downloader's own read-only rules."""
    options = []
    for record in list_targets(root):
        name = str(record["target"])
        spec = _CATALOG[name]
        af3 = name in {"af3", "af3-la"}
        missing, unsafe = _af3_missing_assets(root, name) if af3 else _missing_assets(root, spec)
        reason = ""
        if unsafe:
            reason = "Unsafe local path; repair it before downloading."
        elif not missing and not force:
            pass  # Selecting an installed resource is a harmless skip.
        elif af3:
            from .af3_assets import direct_relatives
            by_relative = {asset.relative: asset for asset in spec.assets}
            if any(not by_relative.get(relative) or not by_relative[relative].url
                   for relative in direct_relatives(name)):
                reason = "No weight URL configured; provide the authorized local weights."
        elif not spec.has_source:
            reason = ("Supplied with Git; restore the files from the repository."
                      if spec.with_git_repo else "No download URL configured; provide local files.")
        else:
            selected_assets = spec.assets if force else missing
            if any(not asset.url and not (asset.archive_member and spec.archive_url)
                   and not (name in {"esmfold2-fast", "esmfold2-standard"}
                            and asset.relative == _ESMFOLD2_BONDS)
                   for asset in selected_assets):
                reason = "Some required files have no download URL; provide them locally."
        options.append({**record, "selectable": not bool(reason), "selection_detail": reason})
    return options


def _make_temp(directory: Path, *, prefix: str, suffix: str = ".part") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    os.close(descriptor)
    return Path(raw)


def _emit(event: dict[str, object]) -> None:
    callback = _PROGRESS.get()
    if callback is not None:
        try:
            callback(event)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise _ProgressCallbackError(str(exc)) from exc


def _transfer_label(destination: Path) -> str:
    context = _TRANSFER.get()
    return context.label if context is not None else destination.name


@contextmanager
def _transfer_context(label: str) -> Iterator[None]:
    token = _TRANSFER.set(_TransferContext(label=label))
    try:
        yield
    finally:
        _TRANSFER.reset(token)


def _transfer_start(label: str, backend: str, total: int | None = None) -> None:
    _emit({"event": "transfer_start", "label": label, "backend": backend, "total": total})


def _transfer_progress(label: str, backend: str, completed: int, total: int | None) -> None:
    _emit(
        {
            "event": "transfer_progress",
            "label": label,
            "backend": backend,
            "completed": completed,
            "total": total,
        }
    )


def _transfer_done(label: str, backend: str, completed: int | None, total: int | None) -> None:
    event: dict[str, object] = {
        "event": "transfer_done",
        "label": label,
        "backend": backend,
        "total": total,
    }
    if completed is not None:
        event["completed"] = completed
    _emit(event)


def _copy_response(
    response: BinaryIO,
    destination: Path,
    *,
    label: str | None = None,
    backend: str | None = None,
    emit_progress: bool = True,
) -> tuple[int, int | None]:
    expected_raw = None
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            expected_raw = headers.get("Content-Length")
        except Exception:
            expected_raw = None
    expected = None
    if expected_raw is not None:
        try:
            expected = int(expected_raw)
        except (TypeError, ValueError):
            raise _DownloadError("HTTP Content-Length is invalid") from None
        if expected <= 0:
            raise _DownloadError("downloaded resource is empty")
    received = 0
    with destination.open("wb") as stream:
        while True:
            chunk = response.read(_CHUNK_SIZE)
            if not chunk:
                break
            stream.write(chunk)
            received += len(chunk)
            if emit_progress and backend is not None:
                _transfer_progress(label or destination.name, backend, received, expected)
    if received == 0:
        raise _DownloadError("downloaded resource is empty")
    if expected is not None and received != expected:
        raise _DownloadError(f"interrupted download: received {received} of {expected} bytes")
    return received, expected


def _fetch_urllib(url: str, destination: Path) -> None:
    label = _transfer_label(destination)
    _transfer_start(label, "urllib")
    try:
        response = urlopen(url, timeout=_DOWNLOAD_TIMEOUT)
        with response:
            status = getattr(response, "status", None)
            if status is None:
                getcode = getattr(response, "getcode", None)
                status = getcode() if callable(getcode) else None
            if status is not None and int(status) >= 400:
                raise _DownloadError(f"HTTP status {status} for {url}")
            received, expected = _copy_response(
                response,
                destination,
                label=label,
                backend="urllib",
            )
        _transfer_done(label, "urllib", received, expected)
    except _DownloadError:
        raise
    except KeyboardInterrupt:
        destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        raise _DownloadError(f"download failed from {url}: {_format_error(exc)}") from exc


def _aria_sidecar(destination: Path) -> Path:
    return Path(f"{destination}.aria2")


def _cleanup_aria_files(destination: Path) -> None:
    _aria_sidecar(destination).unlink(missing_ok=True)


def _terminate_process(process: Any) -> None:
    """Best-effort cancellation used when an aria2c transfer is interrupted."""
    try:
        process.terminate()
    except KeyboardInterrupt:
        raise
    except Exception:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    except KeyboardInterrupt:
        raise
    except Exception:
        pass
    else:
        return
    try:
        process.kill()
    except KeyboardInterrupt:
        raise
    except Exception:
        pass
    try:
        process.wait(timeout=5)
    except KeyboardInterrupt:
        raise
    except Exception:
        pass


def _aria_command(executable: str, url: str, destination: Path) -> list[str]:
    return [
        executable,
        "--no-conf=true",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--check-certificate=true",
        "--connect-timeout=" + str(_ARIA_CONNECT_TIMEOUT),
        "--enable-color=false",
        "--file-allocation=none",
        "--human-readable=false",
        "--max-connection-per-server=" + str(_ARIA_CONNECTIONS),
        "--max-tries=" + str(_ARIA_RETRIES),
        "--min-split-size=1M",
        "--out=" + destination.name,
        "--retry-wait=" + str(_ARIA_RETRY_WAIT),
        "--show-console-readout=true",
        "--split=" + str(_ARIA_CONNECTIONS),
        "--timeout=" + str(_DOWNLOAD_TIMEOUT),
        "--dir=" + str(destination.parent),
        "--summary-interval=1",
        "--console-log-level=notice",
        "--download-result=hide",
        url,
    ]


def _aria_progress(line: str) -> tuple[int, int | None] | None:
    match = _ARIA_PROGRESS_RE.search(line)
    if match is None:
        return None
    try:
        completed = int(match.group("completed").replace(",", ""))
        total = int(match.group("total").replace(",", ""))
    except (TypeError, ValueError):
        return None
    if completed < 0 or total < 0 or completed > total:
        return None
    return completed, total or None


def _aria_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value if isinstance(value, str) else ""


def _aria_output(
    stdout: object,
    stderr: object = None,
    *,
    destination: Path | None = None,
) -> str:
    lines: list[str] = []
    for value in (stdout, stderr):
        text = _aria_text(value)
        if text:
            lines.extend(text.splitlines())
    cleaned: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or _aria_progress(line) is not None:
            continue
        line = _ARIA_GID_RE.sub("", line)
        if destination is not None:
            line = line.replace(destination.name, "<download>")
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)[-_ARIA_TAIL_CHARS:]


def _fetch_aria2c(executable: str, url: str, destination: Path) -> None:
    label = _transfer_label(destination)
    _transfer_start(label, "aria2c")
    process: Any | None = None
    tail: deque[str] = deque()
    tail_chars = 0
    last_progress: tuple[int, int | None] | None = None

    def consume(raw_line: object) -> None:
        nonlocal last_progress, tail_chars
        line = _aria_text(raw_line).strip()
        if not line:
            return
        tail.append(line)
        tail_chars += len(line)
        while tail and (len(tail) > _ARIA_TAIL_LINES or tail_chars > _ARIA_TAIL_CHARS):
            tail_chars -= len(tail.popleft())
        parsed = _aria_progress(line)
        if parsed is None:
            return
        completed, total = parsed
        if last_progress is not None:
            previous, _previous_total = last_progress
            if completed < previous:
                return
        if parsed == last_progress:
            return
        _transfer_progress(label, "aria2c", completed, total)
        last_progress = parsed

    def consume_stream(stream: object) -> None:
        reader = getattr(stream, "read", None)
        if callable(reader):
            buffered: list[str] = []
            while True:
                chunk = _aria_text(reader(1))
                if not chunk:
                    break
                for character in chunk:
                    if character in "\r\n":
                        consume("".join(buffered))
                        buffered.clear()
                    else:
                        buffered.append(character)
            if buffered:
                consume("".join(buffered))
            return
        for raw_line in stream:  # type: ignore[union-attr]
            for line in re.split(r"[\r\n]+", _aria_text(raw_line)):
                consume(line)

    try:
        try:
            process = subprocess.Popen(
                _aria_command(executable, url, destination),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
            )
            stream = getattr(process, "stdout", None)
            if stream is not None:
                consume_stream(stream)
                process.wait()
            else:
                try:
                    stdout, stderr = process.communicate()
                except KeyboardInterrupt:
                    _terminate_process(process)
                    raise
                except Exception as exc:
                    _terminate_process(process)
                    raise _AriaFailure(f"aria2c transfer failed: {_format_error(exc)}") from exc
                for value in (stdout, stderr):
                    text = _aria_text(value)
                    if text:
                        for raw_line in text.splitlines():
                            consume(raw_line)
        except _ProgressCallbackError:
            _terminate_process(process)
            raise
        except KeyboardInterrupt:
            if process is not None:
                _terminate_process(process)
            raise
        except Exception as exc:
            if process is not None:
                _terminate_process(process)
            raise _AriaFailure(f"aria2c could not run: {_format_error(exc)}") from exc
        if process.returncode != 0:
            details = _aria_output("\n".join(tail), destination=destination)
            suffix = f": {details}" if details else ""
            raise _AriaFailure(f"aria2c exited with status {process.returncode}{suffix}")
        try:
            if destination.is_symlink() or not destination.is_file():
                raise _AriaFailure("aria2c completed without producing a regular file")
            with destination.open("rb") as stream:
                if not stream.read(1):
                    raise _AriaFailure("aria2c produced an empty file")
            completed = destination.stat().st_size
        except OSError as exc:
            raise _AriaFailure(f"aria2c output is unavailable: {_format_error(exc)}") from exc
        total = last_progress[1] if last_progress is not None else None
        _transfer_done(label, "aria2c", completed, total)
    finally:
        try:
            if process is not None:
                stream = getattr(process, "stdout", None)
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
        except KeyboardInterrupt:
            raise
        except Exception:
            pass
        finally:
            _cleanup_aria_files(destination)


def _fetch(url: str, destination: Path) -> None:
    """Fetch one URL, preferring aria2c and falling back to urllib."""
    executable = shutil.which("aria2c")
    if executable:
        try:
            _fetch_aria2c(executable, url, destination)
            return
        except KeyboardInterrupt:
            destination.unlink(missing_ok=True)
            _cleanup_aria_files(destination)
            raise
        except _AriaFailure as exc:
            label = _transfer_label(destination)
            _emit(
                {
                    "event": "fallback",
                    "label": label,
                    "reason": str(exc),
                }
            )
            destination.unlink(missing_ok=True)
            _cleanup_aria_files(destination)
    _fetch_urllib(url, destination)


def _archive_name(raw_name: str) -> str:
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise _DownloadError(f"unsafe archive member path: {raw_name!r}")
    raw_parts = PurePosixPath(raw_name).parts
    if raw_name.startswith("/") or ".." in raw_parts:
        raise _DownloadError(f"unsafe archive member path: {raw_name!r}")
    normalized = posixpath.normpath(raw_name)
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise _DownloadError(f"unsafe archive member path: {raw_name!r}")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _stage_direct(root: Path, asset: _AssetSpec) -> tuple[Path, Path]:
    if not asset.url:
        raise _DownloadError(f"no verified download URL for {asset.relative}")
    destination = _destination(root, asset.relative)
    temporary = _make_temp(destination.parent, prefix=f".{destination.name}.", suffix=".part")
    compressed: Path | None = None
    try:
        if asset.gzip:
            compressed = _make_temp(destination.parent, prefix=f".{destination.name}.", suffix=".gz.part")
        with _transfer_context(destination.name + (".gz" if asset.gzip else "")):
            _fetch(asset.url, compressed if compressed is not None else temporary)
        if compressed is not None:
            with gzip.open(compressed, "rb") as stream:
                _copy_response(stream, temporary, emit_progress=False)
    except (Exception, KeyboardInterrupt):
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if compressed is not None:
            compressed.unlink(missing_ok=True)
    return temporary, destination


def _stage_archive(root: Path, spec: _TargetSpec, assets: Sequence[_AssetSpec]) -> list[tuple[Path, Path]]:
    if not spec.archive_url:
        raise _DownloadError(f"target {spec.name} has no verified archive URL")
    archive = _make_temp(root, prefix=f".{spec.name}.", suffix=".tar.part")
    staged: list[tuple[Path, Path]] = []
    try:
        with _transfer_context(f"{spec.name}.tar"):
            _fetch(spec.archive_url, archive)
        expected_names = {
            asset.archive_member: asset for asset in assets if asset.archive_member is not None
        }
        if len(expected_names) != len(assets):
            raise _DownloadError("archive target has an incomplete member mapping")
        with tarfile.open(archive, mode="r:*") as stream:
            members: dict[str, tarfile.TarInfo] = {}
            for member in stream.getmembers():
                normalized = _archive_name(member.name)
                if member.issym() or member.islnk():
                    raise _DownloadError(f"archive contains a link member: {member.name!r}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise _DownloadError(f"archive contains unsupported member: {member.name!r}")
                if normalized in expected_names:
                    if normalized in members:
                        raise _DownloadError(f"archive contains duplicate member: {member.name!r}")
                    members[normalized] = member
            missing = sorted(set(expected_names) - set(members))
            if missing:
                raise _DownloadError("archive is missing expected members: " + ", ".join(missing))
            for member_name, asset in expected_names.items():
                destination = _destination(root, asset.relative)
                temporary = _make_temp(destination.parent, prefix=f".{destination.name}.", suffix=".part")
                try:
                    extracted = stream.extractfile(members[member_name])
                    if extracted is None:
                        raise _DownloadError(f"archive member is not readable: {member_name!r}")
                    with extracted:
                        _copy_response(extracted, temporary)
                except (Exception, KeyboardInterrupt):
                    temporary.unlink(missing_ok=True)
                    raise
                staged.append((temporary, destination))
    except (Exception, KeyboardInterrupt):
        for temporary, _destination_path in staged:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        archive.unlink(missing_ok=True)
    return staged


def _publish(staged: Sequence[tuple[Path, Path]]) -> None:
    """Publish all staged files and roll back if a filesystem operation fails."""
    completed: list[tuple[Path, Path | None]] = []
    current: tuple[Path, Path | None] | None = None
    try:
        for temporary, destination in staged:
            backup: Path | None = None
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise _DownloadError(f"unsafe publish destination: {destination}")
                backup = _make_temp(destination.parent, prefix=f".{destination.name}.", suffix=".backup")
                backup.unlink(missing_ok=True)
                os.replace(destination, backup)
            current = (destination, backup)
            os.replace(temporary, destination)
            completed.append(current)
            current = None
    except (Exception, KeyboardInterrupt):
        if current is not None:
            destination, backup = current
            destination.unlink(missing_ok=True)
            if backup is not None and backup.exists():
                os.replace(backup, destination)
        for destination, backup in reversed(completed):
            destination.unlink(missing_ok=True)
            if backup is not None and backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        for _destination, backup in completed:
            if backup is not None:
                backup.unlink(missing_ok=True)
        if current is not None and current[1] is not None:
            current[1].unlink(missing_ok=True)


def _download_af3_one(root: Path, spec: _TargetSpec, *, force: bool) -> dict[str, object]:
    """Stage one AF3 weight plus shared CCD artifacts as one publication."""
    from . import af3_assets

    target = spec.name
    missing, unsafe = _af3_missing_assets(root, target)
    if unsafe:
        return _af3_record(spec, root, "failed", unsafe)
    if not missing and not force:
        return _af3_record(spec, root, "installed")

    direct_relatives = af3_assets.direct_relatives(target)
    by_relative = {asset.relative: asset for asset in spec.assets}
    direct_assets: list[_AssetSpec] = []
    for relative in direct_relatives:
        asset = by_relative.get(relative)
        if asset is None or not asset.url:
            return _af3_record(
                spec,
                root,
                "failed" if force else "unavailable",
                f"no verified download URL for: {relative}",
            )
        direct_assets.append(asset)
    assets = direct_assets if force else [asset for asset in direct_assets if asset.relative in missing]

    staged: list[tuple[Path, Path]] = []
    derived_stage: Path | None = None
    try:
        for asset in assets:
            staged.append(_stage_direct(root, asset))

        ccd_relative = af3_assets.AF3_SHARED_CCD_RELATIVES[0]
        ccd_destination = _destination(root, ccd_relative)
        source_gz = next(
            (temporary for temporary, destination in staged if destination == ccd_destination),
            ccd_destination,
        )
        derived_missing = any(
            relative in missing for relative in af3_assets.AF3_SHARED_CCD_RELATIVES[1:]
        )
        if force or derived_missing:
            ccd_destination.parent.mkdir(parents=True, exist_ok=True)
            derived_stage = Path(
                tempfile.mkdtemp(prefix=".af3-derived-", dir=ccd_destination.parent)
            )
            prepared = af3_assets.stage_derived(
                source_gz,
                derived_stage,
                source_root=af3_assets.source_path(root),
                libcifpp_data_dir=ccd_destination.parent,
            )
            for filename, temporary in zip(
                af3_assets.AF3_DERIVED_CCD_NAMES,
                prepared,
                strict=True,
            ):
                destination = _destination(
                    root, f"community_models/af3/ccd/{filename}"
                )
                staged.append((temporary, destination))
        if not staged:
            raise _DownloadError("AF3 target has no downloadable or derived assets")
        _publish(staged)
        return _af3_record(spec, root, "downloaded")
    except Exception as exc:
        return _af3_record(spec, root, "failed", _format_error(exc))
    finally:
        for temporary, _destination_path in staged:
            temporary.unlink(missing_ok=True)
        if derived_stage is not None:
            shutil.rmtree(derived_stage, ignore_errors=True)


def _esmfold2_ccd_tools():
    """Use the stdlib-only CCD builder shipped with the code, not the model root."""
    from .paths import community_source

    source = str(community_source("pipetools"))
    sys.path.insert(0, source)
    try:
        from pipetools.ccd_bonds import (
            CCDBondCacheError,
            build_ccd_bond_cache,
            validate_ccd_bond_cache,
        )
    finally:
        sys.path.remove(source)
    return CCDBondCacheError, build_ccd_bond_cache, validate_ccd_bond_cache


def _download_esmfold2_one(root: Path, spec: _TargetSpec, *, force: bool) -> dict[str, object]:
    """Provision model files and shared CCD chemistry as one publication."""
    missing, unsafe = _missing_assets(root, spec)
    if unsafe:
        return _record(spec, root, "failed", unsafe)

    staged: list[tuple[Path, Path]] = []
    try:
        cache_error, build_cache, validate_cache = _esmfold2_ccd_tools()
        cache_destination = _destination(root, _ESMFOLD2_BONDS)
        rebuild = force or any(
            asset.relative in {_ESMFOLD2_CIF, _ESMFOLD2_BONDS} for asset in missing
        )
        if not rebuild:
            try:
                validate_cache(cache_destination)
            except cache_error:
                rebuild = True
        if not missing and not rebuild and not force:
            return _record(spec, root, "installed")

        assets = spec.assets if force else missing
        for asset in assets:
            if asset.relative != _ESMFOLD2_BONDS:
                staged.append(_stage_direct(root, asset))

        if rebuild:
            source_destination = _destination(root, _ESMFOLD2_CIF)
            source = next(
                (temporary for temporary, destination in staged if destination == source_destination),
                source_destination,
            )
            cache_stage = _make_temp(
                cache_destination.parent, prefix=".ccd_bonds.", suffix=".part"
            )
            staged.append((cache_stage, cache_destination))
            _emit({"event": "prepare", "label": "ESMFold2 CCD bond cache"})
            build_cache(source, cache_stage)
            validate_cache(cache_stage)
        if not staged:
            raise _DownloadError("ESMFold2 target has no downloadable or derived assets")
        _publish(staged)
        return _record(spec, root, "downloaded")
    except Exception as exc:
        return _record(spec, root, "failed", _format_error(exc))
    finally:
        for temporary, _destination_path in staged:
            temporary.unlink(missing_ok=True)


def _download_one(root: Path, spec: _TargetSpec, *, force: bool) -> dict[str, object]:
    if spec.name in {"esmfold2-fast", "esmfold2-standard"}:
        return _download_esmfold2_one(root, spec, force=force)
    if spec.name in {"af3", "af3-la"}:
        return _download_af3_one(root, spec, force=force)
    missing, unsafe = _missing_assets(root, spec)
    if unsafe:
        return _record(spec, root, "failed", unsafe)
    complete = not missing
    if complete and not force:
        return _record(spec, root, "installed")
    if not spec.has_source:
        if spec.with_git_repo:
            detail = "files are supplied with git repo; copy or restore the required files from a PartiDesign checkout"
            if force:
                detail = "--force cannot refresh Git-supplied assets; " + detail
            return _record(spec, root, "failed" if force else "unavailable", detail)
        if force:
            return _record(
                spec,
                root,
                "failed",
                "--force cannot refresh a resource with no verified download URL",
            )
        return _record(
            spec,
            root,
            "unavailable",
            "no verified download URL; local resource must be provisioned separately",
        )
    assets = list(spec.assets) if force else missing
    without_source = [
        asset.relative for asset in assets if not asset.url and not asset.archive_member
    ]
    if without_source:
        message = "no verified download URL for: " + ", ".join(without_source)
        return _record(spec, root, "failed" if force else "unavailable", message)
    staged: list[tuple[Path, Path]] = []
    try:
        direct = [asset for asset in assets if asset.url]
        if direct:
            staged.extend(_stage_direct(root, asset) for asset in direct)
        archive_assets = [asset for asset in assets if asset.archive_member]
        if archive_assets:
            staged.extend(_stage_archive(root, spec, archive_assets))
        if not staged:
            raise _DownloadError("target has no downloadable assets")
        _publish(staged)
        return _record(spec, root, "downloaded")
    except Exception as exc:
        return _record(spec, root, "failed", _format_error(exc))
    finally:
        for temporary, _destination in staged:
            temporary.unlink(missing_ok=True)


def download_targets(
    targets: Sequence[str],
    root: Path,
    force: bool = False,
    *,
    progress: Callable[[dict[str, object]], object] | None = None,
) -> list[dict[str, object]]:
    """Download selected targets, continuing after independent failures.

    ``progress`` receives small event dictionaries for target and transfer
    lifecycle changes.  A context variable keeps the established private
    ``_fetch(url, destination)`` call shape available to callers and tests.
    """
    token = _PROGRESS.set(progress)
    records: list[dict[str, object]] = []
    try:
        for raw_target in targets:
            name = str(raw_target)
            _emit({"event": "target_start", "target": name})
            spec = _CATALOG.get(name)
            if spec is None:
                record = {
                    "target": name,
                    "status": "failed",
                    "error": f"unknown target: {name}",
                    "paths": [],
                    "source_kind": "unconfigured",
                    "sources": [],
                }
            else:
                record = _download_one(_root_path(root), spec, force=bool(force))
            records.append(record)
            _emit({"event": "target_done", "target": name, "record": record})
        return records
    finally:
        _PROGRESS.reset(token)


__all__ = ["TARGETS", "download_targets", "list_targets"]
