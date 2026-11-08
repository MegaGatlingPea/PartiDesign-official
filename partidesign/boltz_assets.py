"""Pinned local Boltz-2 assets; no online searches, imports or unpickling."""
from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import uuid

HF_REVISION = "6fdef46d763fee7fbb83ca5501ccceff43b85607"
HF_BASE = f"https://huggingface.co/boltz-community/boltz-2/resolve/{HF_REVISION}"
ASSETS = {
    "boltz2_conf.ckpt": (2_286_561_469, "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1"),
    "boltz2_aff.ckpt": (2_062_139_170, "dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e"),
    "mols.tar": (1_855_662_080, "39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7"),
}
MOLS_FILE_COUNT = 45_227
MOLS_UNPACKED_BYTES = 1_820_698_819
_CHUNK = 1024 * 1024


def _check_layout(root):
    for path in (root, root / "common"):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f"Boltz asset parent must be an ordinary directory: {path}")


@lru_cache(maxsize=8)
def _read_inventory(path, device, inode, size, mtime, expected_size, expected_count, expected_total):
    del device, inode, mtime
    if size != expected_size:
        raise ValueError(f"Boltz archive size mismatch: {size}; expected {expected_size}")
    entries, names = [], set()
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            raw = member.name
            if (raw.startswith("/") or "\\" in raw or "\x00" in raw
                    or any(p in {"", ".", ".."} for p in raw.split("/"))):
                raise ValueError(f"Unsafe Boltz archive path: {raw!r}")
            parts = PurePosixPath(raw).parts
            if member.isdir() and parts == ("mols",):
                continue
            if (not member.isfile() or len(parts) != 2 or parts[0] != "mols"
                    or not parts[1].endswith(".pkl") or member.size <= 0):
                raise ValueError(f"Unsupported Boltz archive member: {raw!r}")
            if raw in names or member.offset_data + member.size > size:
                raise ValueError(f"Duplicate or truncated Boltz archive member: {raw!r}")
            names.add(raw)
            entries.append((parts[1], member.size, member.offset_data))
    if len(entries) != expected_count or sum(x[1] for x in entries) != expected_total:
        raise ValueError("Boltz archive inventory differs from the pinned release")
    return tuple(entries)


def _inventory(archive):
    if archive.is_symlink() or not archive.is_file():
        raise FileNotFoundError(f"Regular Boltz archive required: {archive}")
    s = archive.stat()
    return _read_inventory(str(archive), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns,
                           ASSETS["mols.tar"][0], MOLS_FILE_COUNT, MOLS_UNPACKED_BYTES)


def _verify_stream(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    while chunk := stream.read(_CHUNK):
        digest.update(chunk)
    if digest.hexdigest() != ASSETS["mols.tar"][1]:
        raise ValueError("Boltz archive SHA256 differs from the pinned release")


def _validate_tree(directory, entries, *, stream=None):
    if not directory.is_dir():
        raise FileNotFoundError(f"Extracted Boltz molecules required: {directory}")
    expected = {name for name, _, _ in entries}
    observed = {entry.name for entry in directory.iterdir()}
    if observed != expected:
        raise ValueError(f"Incomplete Boltz molecules: {len(observed & expected)}/{len(expected)} files")
    for name, size, offset in entries:
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Invalid Boltz molecule: {path}")
        if stream is not None:
            stream.seek(offset)
            remaining = size
            with path.open("rb") as local:
                while remaining:
                    block = stream.read(min(_CHUNK, remaining))
                    if not block or local.read(len(block)) != block:
                        raise ValueError(f"Boltz molecule content differs from archive: {path}")
                    remaining -= len(block)
                if local.read(1):
                    raise ValueError(f"Boltz molecule trailing content: {path}")


def inspect_boltz_mols(root, *, verify=False):
    """Read-only inventory, optionally byte-matching every file to pinned archive."""
    root = Path(root).expanduser()
    directory, archive = root / "common/mols", root / "common/mols.tar"
    item = dict(path=str(directory), relative="community_models/boltz/common/mols",
                bytes=None, expected_bytes=MOLS_UNPACKED_BYTES, expected_sha256=None,
                integrity="missing", detail="molecule archive must be extracted")
    try:
        _check_layout(root)
        if not directory.exists() and not directory.is_symlink():
            return item
        entries = _inventory(archive)
        if verify:
            with archive.open("rb") as stream:
                _verify_stream(stream)
                _validate_tree(directory, entries, stream=stream)
        else:
            _validate_tree(directory, entries)
        item.update(bytes=MOLS_UNPACKED_BYTES, integrity="size_verified",
                    detail=f"{len(entries):,} molecules " +
                    ("byte-matched to the pinned archive" if verify else "present; release names and sizes match"))
    except (OSError, ValueError, tarfile.TarError) as error:
        item.update(integrity="invalid", detail=f"{type(error).__name__}: {error}")
    return item


@contextmanager
def _lock(path):
    import fcntl
    if path.is_symlink():
        raise ValueError(f"Boltz extraction lock must not be a symlink: {path}")
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def ensure_boltz_mols(root, *, verify=False):
    """Reuse a complete tree or atomically extract the pinned archive.

    Complete directory symlinks may be reused read-only. Invalid symlinks are
    never repaired. Replaced incomplete directories remain as hidden backups.
    verify=True also checks archive SHA256 and exact extracted file contents.
    """
    root = Path(root).expanduser()
    _check_layout(root)
    archive, common = root / "common/mols.tar", root / "common"
    entries = _inventory(archive)
    destination = common / "mols"
    def check_existing():
        if verify:
            with archive.open("rb") as stream:
                _verify_stream(stream)
                _validate_tree(destination, entries, stream=stream)
        else:
            _validate_tree(destination, entries)

    try:
        check_existing()
        return destination
    except (OSError, ValueError):
        if destination.is_symlink():
            raise ValueError(f"Incomplete Boltz molecule symlink is not modified: {destination}")
    common.mkdir(parents=True, exist_ok=True)
    with _lock(common / ".mols.lock"):
        _check_layout(root)
        try:
            check_existing()
            return destination
        except (OSError, ValueError):
            if destination.is_symlink():
                raise ValueError(f"Incomplete Boltz molecule symlink is not modified: {destination}")
        if destination.exists() and not destination.is_dir():
            raise ValueError(f"Boltz molecule destination is not a directory: {destination}")
        staging = Path(tempfile.mkdtemp(prefix=".mols.staging-", dir=common))
        staged = staging / "mols"
        staged.mkdir()
        try:
            with archive.open("rb") as stream:
                _verify_stream(stream)
                for name, size, offset in entries:
                    stream.seek(offset)
                    remaining = size
                    with (staged / name).open("xb") as out:
                        while remaining:
                            block = stream.read(min(_CHUNK, remaining))
                            if not block:
                                raise ValueError(f"Truncated Boltz archive while extracting {name}")
                            out.write(block)
                            remaining -= len(block)
            _validate_tree(staged, entries)
            previous = None
            if destination.exists():
                previous = common / (".mols.replaced-" + uuid.uuid4().hex)
                destination.rename(previous)
            try:
                staged.rename(destination)
            except Exception:
                if previous is not None:
                    previous.rename(destination)
                raise
            return destination
        finally:
            shutil.rmtree(staging)


__all__ = ["HF_REVISION", "HF_BASE", "ASSETS", "ensure_boltz_mols", "inspect_boltz_mols"]
