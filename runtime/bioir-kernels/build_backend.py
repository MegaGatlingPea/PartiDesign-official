"""Repackage a pinned upstream BioIR wheel as a separate kernel distribution.

The backend uses only the Python standard library. NVIDIA package files and
licenses are copied byte for byte; no native compilation or Torch import occurs.
"""

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import os
from pathlib import Path
import platform
import sys
import tempfile
from urllib.request import Request, urlopen
import zipfile
from email import policy
from email.parser import BytesParser


NAME = "partidesign-bioir-kernels"
VERSION = "0.2.0+partidesign.1"
WHEEL_TAG = "cp312-cp312-manylinux_2_34_x86_64"
DIST_INFO = f"partidesign_bioir_kernels-{VERSION}.dist-info"
WHEEL_FILENAME = f"partidesign_bioir_kernels-{VERSION}-{WHEEL_TAG}.whl"
UPSTREAM_DIST_INFO = "bionemo_ir-0.2.0.dist-info"
UPSTREAM_FILENAME = "bionemo_ir-0.2.0-cp312-cp312-manylinux_2_34_x86_64.whl"
UPSTREAM_URL = (
    "https://files.pythonhosted.org/packages/5f/d3/"
    "b2e9b3f1aa5b24bf499a0bd3816739a26d7b2b2aee477a36a940ca5e9302/"
    + UPSTREAM_FILENAME
)
UPSTREAM_SHA256 = "afc64f92a3b4ac96d23840136201fe0e21317ef2918437690718e6c4cc0d74f0"
UPSTREAM_SIZE = 37328770
REQUIREMENTS = (
    "torch>=2.11,<2.12",
    "cuda-python>=13,<14",
    "cuequivariance>=0.11.1,<0.14",
    "cuequivariance-ops-cu13>=0.11.1,<0.14",
    "cuequivariance-ops-torch-cu13>=0.11.1,<0.14",
    "triton>=3.6,<3.9",
    "numpy>=2.2,<3",
    "nvidia-cutlass-dsl[cu13]==4.5.2",
    "nvidia-cudnn-frontend==1.23.0",
    "apache-tvm-ffi>=0.1.6,<0.2",
    "lru-dict==1.4.1",
    "einops>=0.8",
    "pydantic>=2,<3",
    "PyYAML>=6.0.3",
    "huggingface-hub>=0.36,<1",
)


def _check_platform() -> None:
    if (
        sys.implementation.name != "cpython"
        or sys.version_info[:2] != (3, 12)
        or sys.platform != "linux"
        or platform.machine().lower() != "x86_64"
    ):
        raise RuntimeError(
            f"{NAME} {VERSION} requires Linux x86_64 and CPython 3.12; "
            "the pinned upstream wheel does not support this platform/Python."
        )
    try:
        libc_name, libc_version = os.confstr("CS_GNU_LIBC_VERSION").split()
        libc_tuple = tuple(int(part) for part in libc_version.split(".")[:2])
    except (AttributeError, OSError, TypeError, ValueError):
        libc_name, libc_version = platform.libc_ver()
        try:
            libc_tuple = tuple(int(part) for part in libc_version.split(".")[:2])
        except ValueError:
            libc_tuple = ()
    if libc_name not in {"glibc", "GNU libc"} or libc_tuple < (2, 34):
        raise RuntimeError(
            f"{NAME} requires glibc >= 2.34 for its pinned manylinux wheel; "
            f"detected {libc_name or 'unknown libc'} {libc_version or 'unknown'}."
        )


def _cache_path() -> Path:
    base = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser()
    return base / "partidesign" / "bioir-kernels" / UPSTREAM_FILENAME


def _verify_file(path: Path) -> None:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > UPSTREAM_SIZE:
                break
            digest.update(chunk)
    if size != UPSTREAM_SIZE or digest.hexdigest() != UPSTREAM_SHA256:
        raise RuntimeError(
            f"BioIR upstream wheel integrity check failed for {path}: "
            f"expected SHA256 {UPSTREAM_SHA256} and {UPSTREAM_SIZE} bytes; "
            f"received {digest.hexdigest()} and {size} bytes. "
            "Remove an invalid cached file before retrying."
        )


def _download() -> Path:
    cached = _cache_path()
    if cached.exists():
        _verify_file(cached)
        return cached
    cached.parent.mkdir(parents=True, exist_ok=True)
    descriptor, pending_name = tempfile.mkstemp(
        prefix=UPSTREAM_FILENAME + ".", suffix=".tmp", dir=cached.parent
    )
    pending = Path(pending_name)
    request = Request(UPSTREAM_URL, headers={"User-Agent": f"{NAME}/{VERSION}"})
    try:
        with os.fdopen(descriptor, "wb") as out:
            with urlopen(request, timeout=120) as response:
                if not response.geturl().startswith("https://"):
                    raise RuntimeError("The upstream wheel download must remain on HTTPS.")
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > UPSTREAM_SIZE:
                        raise RuntimeError("Upstream wheel exceeds its pinned file size.")
                    out.write(chunk)
        _verify_file(pending)
        os.replace(pending, cached)
    except OSError as exc:
        raise RuntimeError(f"Could not download pinned BioIR wheel from {UPSTREAM_URL}: {exc}") from exc
    finally:
        pending.unlink(missing_ok=True)
    return cached


def _checked_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    names = set()
    for member in archive.infolist():
        name = member.filename
        components = name.rstrip("/").split("/")
        if (
            not name
            or name.startswith("/")
            or "\\" in name
            or any(part in {"", ".", ".."} for part in components)
            or components[0] not in {"bionemo_ir", UPSTREAM_DIST_INFO}
            or name in names
            or (member.external_attr >> 16) & 0o170000 == 0o120000
        ):
            raise RuntimeError(f"Unsupported or unsafe upstream wheel member: {name!r}")
        names.add(name)
        if not member.is_dir():
            members.append(member)
    required = {
        f"{UPSTREAM_DIST_INFO}/METADATA",
        f"{UPSTREAM_DIST_INFO}/WHEEL",
        f"{UPSTREAM_DIST_INFO}/RECORD",
        f"{UPSTREAM_DIST_INFO}/licenses/LICENSE",
        "bionemo_ir/__init__.py",
    }
    if not required.issubset(names):
        raise RuntimeError("The pinned BioIR wheel is missing required package or license files.")
    return members


def _metadata(archive: zipfile.ZipFile) -> bytes:
    message = BytesParser(policy=policy.compat32).parsebytes(
        archive.read(f"{UPSTREAM_DIST_INFO}/METADATA")
    )
    for header in ("Name", "Version", "Summary", "Requires-Python", "Requires-Dist"):
        del message[header]
    message["Name"] = NAME
    message["Version"] = VERSION
    message["Summary"] = "PartiDesign kernel API runtime derived from NVIDIA BioNeMo IR 0.2.0"
    message["Requires-Python"] = ">=3.12,<3.13"
    for requirement in REQUIREMENTS:
        message["Requires-Dist"] = requirement
    return message.as_bytes(policy=policy.compat32.clone(max_line_length=0, linesep="\n"))


def _metadata_files(archive: zipfile.ZipFile, members: list[zipfile.ZipInfo]) -> dict[str, bytes]:
    result = {}
    prefix = UPSTREAM_DIST_INFO + "/"
    for member in members:
        if not member.filename.startswith(prefix):
            continue
        relative = member.filename[len(prefix):]
        if relative in {"RECORD", "RECORD.jws", "RECORD.p7s", "entry_points.txt"}:
            continue
        if relative == "METADATA":
            result[relative] = _metadata(archive)
        elif relative == "WHEEL":
            result[relative] = (
                "Wheel-Version: 1.0\n"
                f"Generator: {NAME} ({VERSION})\n"
                "Root-Is-Purelib: false\n"
                f"Tag: {WHEEL_TAG}\n\n"
            ).encode("utf-8")
        else:
            result[relative] = archive.read(member)
    return result


def get_requires_for_build_wheel(config_settings=None) -> list[str]:
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None) -> str:
    _check_platform()
    with tempfile.TemporaryDirectory(prefix="partidesign-bioir-") as temporary:
        upstream = _download()
        with zipfile.ZipFile(upstream) as archive:
            files = _metadata_files(archive, _checked_members(archive))
            destination = Path(metadata_directory) / DIST_INFO
            for relative, content in files.items():
                path = destination / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    return DIST_INFO


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None) -> str:
    _check_platform()
    destination = Path(wheel_directory)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="partidesign-bioir-") as temporary:
        temporary = Path(temporary)
        upstream = _download()
        with zipfile.ZipFile(upstream) as archive:
            members = _checked_members(archive)
            metadata_files = _metadata_files(archive, members)
            if metadata_directory is not None:
                prepared = Path(metadata_directory)
                if prepared.name != DIST_INFO:
                    prepared = prepared / DIST_INFO
                for relative, content in metadata_files.items():
                    path = prepared / relative
                    if not path.is_file() or path.read_bytes() != content:
                        raise RuntimeError(f"Prepared wheel metadata differs: {relative}")
            rows = []
            built = temporary / WHEEL_FILENAME
            with zipfile.ZipFile(built, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
                prefix = UPSTREAM_DIST_INFO + "/"
                for member in members:
                    info = copy.copy(member)
                    if member.filename.startswith(prefix):
                        relative = member.filename[len(prefix):]
                        if relative not in metadata_files:
                            continue
                        info.filename = f"{DIST_INFO}/{relative}"
                        content = metadata_files[relative]
                    else:
                        content = archive.read(member)
                    out.writestr(info, content)
                    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
                    rows.append((info.filename, "sha256=" + digest, str(len(content))))
                record_name = f"{DIST_INFO}/RECORD"
                rows.append((record_name, "", ""))
                record = io.StringIO(newline="")
                csv.writer(record, lineterminator="\n").writerows(rows)
                info = zipfile.ZipInfo(record_name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                out.writestr(info, record.getvalue().encode("utf-8"))
        # Keep incomplete wheels out of the frontend's output directory.
        with built.open("rb") as source, (destination / WHEEL_FILENAME).open("wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
    return WHEEL_FILENAME
