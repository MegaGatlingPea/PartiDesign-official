"""AlphaFold 3 resource layout and CCD preparation helpers.

AF3 and AF3-leaving-atom share one vendored SDK and one CCD snapshot while
keeping their model archives in separate target directories.  This module only
stages generated CCD artifacts; callers publish the returned files atomically.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import gzip
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
from typing import Any

from .paths import code_root, community_source, offline_site


AF3_TARGETS = ("af3", "af3-la")
AF3_WEIGHT_URLS = {
    "af3": "https://storage.googleapis.com/alphafold3/af3.bin.zst",
    "af3-la": "https://storage.googleapis.com/alphafold3/af3_leaving_atom.bin.zst",
}
AF3_CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"
AF3_WEIGHT_RELATIVES = {
    "af3": "community_models/af3/checkpoints/af3.bin.zst",
    "af3-la": "community_models/af3-la/checkpoints/af3_leaving_atom.bin.zst",
}
AF3_SHARED_CCD_RELATIVES = (
    "community_models/af3/ccd/components.cif.gz",
    "community_models/af3/ccd/components.cif",
    "community_models/af3/ccd/ccd.pickle",
    "community_models/af3/ccd/chemical_component_sets.pickle",
)
AF3_DIRECT_RELATIVES = (
    AF3_WEIGHT_RELATIVES["af3"],
    AF3_WEIGHT_RELATIVES["af3-la"],
    AF3_SHARED_CCD_RELATIVES[0],
)
AF3_DERIVED_CCD_NAMES = (
    "components.cif",
    "ccd.pickle",
    "chemical_component_sets.pickle",
)
# The official CCD is large and conversion is intentionally bounded.  The
# subprocess is terminated rather than left behind if a host stalls.
CCD_CONVERSION_TIMEOUT = 30 * 60


@dataclass(frozen=True, slots=True)
class AF3Layout:
    target: str
    root: Path
    weight: Path
    ccd_dir: Path
    components_cif_gz: Path
    components_cif: Path
    ccd_pickle: Path
    chemical_component_sets: Path

    @property
    def required(self) -> tuple[Path, ...]:
        return (
            self.weight,
            self.components_cif_gz,
            self.components_cif,
            self.ccd_pickle,
            self.chemical_component_sets,
        )

    @property
    def direct(self) -> tuple[Path, ...]:
        return (self.weight, self.components_cif_gz)

    @property
    def derived(self) -> tuple[Path, ...]:
        return (self.components_cif, self.ccd_pickle, self.chemical_component_sets)


def validate_target(target: str) -> str:
    target = str(target)
    if target not in AF3_TARGETS:
        raise ValueError(f"unknown AF3 target: {target}")
    return target


def layout(root: str | Path, target: str) -> AF3Layout:
    target = validate_target(target)
    root = Path(root).expanduser()
    ccd_dir = root / "community_models" / "af3" / "ccd"
    return AF3Layout(
        target=target,
        root=root,
        weight=root / AF3_WEIGHT_RELATIVES[target],
        ccd_dir=ccd_dir,
        components_cif_gz=ccd_dir / "components.cif.gz",
        components_cif=ccd_dir / "components.cif",
        ccd_pickle=ccd_dir / "ccd.pickle",
        chemical_component_sets=ccd_dir / "chemical_component_sets.pickle",
    )


def required_relatives(target: str) -> tuple[str, ...]:
    target = validate_target(target)
    return (
        AF3_WEIGHT_RELATIVES[target],
        *AF3_SHARED_CCD_RELATIVES,
    )


def direct_relatives(target: str) -> tuple[str, ...]:
    target = validate_target(target)
    return (AF3_WEIGHT_RELATIVES[target], AF3_SHARED_CCD_RELATIVES[0])


def target_sources(target: str) -> tuple[str, ...]:
    target = validate_target(target)
    return (AF3_WEIGHT_URLS[target], AF3_CCD_URL)


def _safe_file(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    except OSError:
        return False


def missing_paths(root: str | Path, target: str) -> tuple[Path, ...]:
    return tuple(path for path in layout(root, target).required if not _safe_file(path))


def is_complete(root: str | Path, target: str) -> bool:
    return not missing_paths(root, target)


def source_path(root: str | Path | None = None) -> Path:
    """Return immutable SDK source; ``root`` only owns downloaded assets.

    The AF3 SDK is code shipped by the checkout or under the installed
    ``partidesign/_resources`` tree.  A caller's model root contains weights and
    generated CCD files, so it must never replace the SDK source root.  Keep the
    argument for the existing download helper/API, but deliberately resolve
    source through ``resource_root`` via :func:`community_source`.
    """
    del root
    return community_source("af3")


def _af3_pythonpath(*sources: Path) -> str:
    """Build converter imports from explicit package and SDK roots only."""
    values: list[str] = []
    seen: set[str] = set()
    for path in (*sources, code_root(), offline_site()):
        value = str(Path(path).expanduser().resolve())
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return os.pathsep.join(values)


def _validate_pickle(path: Path, *, kind: str) -> Any:
    if not _safe_file(path):
        raise ValueError(f"generated AF3 {kind} is missing or empty: {path}")
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)
    except Exception as exc:
        raise ValueError(f"generated AF3 {kind} is not readable: {path}: {exc}") from exc
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"generated AF3 {kind} is empty or not a mapping: {path}")
    if kind == "chemical_component_sets":
        required = {"glycans_linking", "glycans_other", "ions"}
        if set(value) != required:
            raise ValueError(
                f"generated AF3 chemical component sets have keys {sorted(value)}, "
                f"expected {sorted(required)}"
            )
        if any(not isinstance(items, (set, frozenset)) for items in value.values()):
            raise ValueError("generated AF3 chemical component sets contain invalid values")
    return value


def validate_derived(paths: tuple[Path, Path, Path]) -> None:
    components, ccd_pickle, sets_pickle = paths
    if not _safe_file(components):
        raise ValueError(f"generated AF3 components.cif is missing or empty: {components}")
    _validate_pickle(ccd_pickle, kind="ccd")
    _validate_pickle(sets_pickle, kind="chemical_component_sets")


_CONVERTER = textwrap.dedent(
    """
    import pickle
    import sys
    from alphafold3.constants.converters import ccd_pickle_gen
    from alphafold3.constants.converters import chemical_component_sets_gen

    source, ccd_output, sets_output = sys.argv[1:4]
    ccd_pickle_gen.main(["", source, ccd_output])
    with open(ccd_output, "rb") as stream:
        ccd = pickle.load(stream)
    sets = chemical_component_sets_gen.find_ions_and_glycans_in_ccd(ccd)
    with open(sets_output, "wb") as stream:
        pickle.dump(sets, stream, protocol=pickle.HIGHEST_PROTOCOL)
    """
).strip()


def stage_derived(
    source_gz: str | Path,
    output_dir: str | Path,
    *,
    source_root: str | Path | None = None,
    libcifpp_data_dir: str | Path | None = None,
    timeout: float = CCD_CONVERSION_TIMEOUT,
) -> tuple[Path, Path, Path]:
    """Generate and validate CCD artifacts in an unpublished staging dir.

    The active interpreter runs the SDK's converter and the official
    find_ions_and_glycans_in_ccd export.  output_dir is caller-owned temporary
    storage; no generated file is published here.
    """
    source_gz = Path(source_gz).expanduser()
    output_dir = Path(output_dir).expanduser()
    if not _safe_file(source_gz):
        raise FileNotFoundError(f"AF3 compressed CCD is missing or empty: {source_gz}")
    output_dir.mkdir(parents=True, exist_ok=True)
    components = output_dir / "components.cif"
    ccd_pickle = output_dir / "ccd.pickle"
    sets_pickle = output_dir / "chemical_component_sets.pickle"

    def cleanup_generated() -> None:
        for path in (components, ccd_pickle, sets_pickle):
            path.unlink(missing_ok=True)

    try:
        with gzip.open(source_gz, "rb") as source, components.open("wb") as destination:
            while (chunk := source.read(1024 * 1024)):
                destination.write(chunk)
    except Exception as exc:
        cleanup_generated()
        raise RuntimeError(f"cannot decompress AF3 CCD {source_gz}: {exc}") from exc
    if not _safe_file(components):
        raise ValueError(f"decompressed AF3 CCD is empty: {components}")

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # The converter reads the raw CCD directory through the same environment
    # used by the SDK build helper.  The temporary directory is still inside
    # the shared CCD parent, so publication remains atomic.
    shared_ccd_dir = (
        Path(libcifpp_data_dir).expanduser()
        if libcifpp_data_dir is not None
        else output_dir.parent
    )
    environment["LIBCIFPP_DATA_DIR"] = str(shared_ccd_dir)
    # Resource lookup happens while converter modules import.  Set this before
    # starting the SDK subprocess so generated CCD resources resolve to the
    # shared model-root directory instead of a vendored package path.
    environment["PARTIDESIGN_AF3_CCD_DIR"] = str(shared_ccd_dir)
    source_src = (
        Path(source_root).expanduser() / "src"
        if source_root is not None
        else community_source("af3") / "src"
    )
    environment["PYTHONPATH"] = _af3_pythonpath(source_src)
    try:
        result = subprocess.run(
            [sys.executable, "-c", _CONVERTER, str(components), str(ccd_pickle), str(sets_pickle)],
            cwd=None if source_root is None else str(Path(source_root).expanduser()),
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        cleanup_generated()
        raise TimeoutError(f"AF3 CCD conversion exceeded {timeout:g}s") from exc
    except OSError as exc:
        cleanup_generated()
        raise RuntimeError(f"cannot run AF3 CCD converter: {exc}") from exc
    if result.returncode:
        cleanup_generated()
        detail = (result.stderr or result.stdout or "").strip()[-4000:]
        raise RuntimeError(f"AF3 CCD conversion failed ({result.returncode}): {detail}")
    try:
        validate_derived((components, ccd_pickle, sets_pickle))
    except Exception:
        cleanup_generated()
        raise
    return components, ccd_pickle, sets_pickle


def validate_installation(root: str | Path, target: str) -> AF3Layout:
    """Validate the complete local target set without importing the SDK."""
    value = layout(root, target)
    missing = [str(path) for path in value.required if not _safe_file(path)]
    if missing:
        raise FileNotFoundError(
            f"{value.target} requires model weight and shared CCD assets: " + ", ".join(missing)
        )
    validate_derived((value.components_cif, value.ccd_pickle, value.chemical_component_sets))
    return value


__all__ = [
    "AF3_CCD_URL",
    "AF3Layout",
    "AF3_SHARED_CCD_RELATIVES",
    "AF3_TARGETS",
    "AF3_WEIGHT_RELATIVES",
    "AF3_WEIGHT_URLS",
    "CCD_CONVERSION_TIMEOUT",
    "direct_relatives",
    "is_complete",
    "layout",
    "missing_paths",
    "required_relatives",
    "source_path",
    "stage_derived",
    "target_sources",
    "validate_derived",
    "validate_installation",
    "validate_target",
]
