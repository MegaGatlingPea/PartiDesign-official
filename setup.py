"""Build configuration for the standalone PartiDesign distribution.

The checkout keeps model files and large local assets next to the source.  The
wheel only receives the selected source closure and an immutable copy of the
small/runtime data under partidesign/_resources.  The copy is made in the
build directory so a build never creates or edits a source-tree resource
directory.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Iterable

from setuptools import find_namespace_packages, setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist
from setuptools.command.egg_info import egg_info as _egg_info
from setuptools.command.build import build as _build


ROOT = Path(__file__).resolve().parent
_RESOURCE_PACKAGE = Path("partidesign") / "_resources"

_BLOCKED_COMMUNITY_DIRS = {
    "__pycache__",
    "backup",
    "backups",
    "build",
    "cache",
    "checkpoint",
    "checkpoints",
    "dist",
    "egg-info",
    "env",
    "environments",
    "log",
    "logs",
    "notebook",
    "notebooks",
    "test",
    "test_data",
    "tests",
    "venv",
    "weights",
}
_BINARY_SUFFIXES = {
    ".a",
    ".bin",
    ".c",
    ".dll",
    ".dylib",
    ".o",
    ".pyc",
    ".so",
}
_WEIGHT_SUFFIXES = {
    ".ckpt",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
}


def _allowed_package(name: str) -> bool:
    if name in {"partidesign", "src.design", "src.validate", "src.scoring", "src.common", "src.cluster", "src.postfilter"}:
        return True
    if name.startswith("src.design."):
        return True
    if name.startswith("src.validate.") and ".tests" not in name:
        return True
    if name.startswith("src.common."):
        return True
    if name.startswith("src.scoring."):
        return True
    if name.startswith("src.postfilter."):
        return True
    if name.startswith("src.cluster."):
        return True
    return False


def _packages() -> list[str]:
    names = {
        name
        for name in find_namespace_packages(
            where=str(ROOT),
            include=["partidesign", "src", "src.*"],
        )
        if _allowed_package(name)
    }
    names.update({"partidesign"})
    return sorted(names)


def _iter_python_files(root: Path, *, exclude_dirs: set[str] | None = None) -> Iterable[Path]:
    excluded = _BLOCKED_COMMUNITY_DIRS if exclude_dirs is None else exclude_dirs
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts):
            continue
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            continue
        yield path


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_python_tree(source_relative: str, build_root: Path, *, exclude_dirs: set[str] | None = None) -> None:
    source = ROOT / source_relative
    destination_root = build_root / source_relative
    for path in _iter_python_files(source, exclude_dirs=exclude_dirs):
        _copy_file(path, destination_root / path.relative_to(source))


def _copy_selected_files(relative_paths: Iterable[str], build_root: Path) -> None:
    for relative in relative_paths:
        source = ROOT / relative
        if source.is_file():
            _copy_file(source, build_root / relative)


def _git_community_files() -> list[Path]:
    command = ["git", "ls-files", "-z", "--", "community_models"]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
    except OSError:
        result = None
    if result is not None and result.returncode == 0:
        names = [item for item in result.stdout.decode().split("\x00") if item]
        if names:
            return [ROOT / item for item in names]
    community_root = ROOT / "community_models"
    return sorted(path for path in community_root.rglob("*") if path.is_file())


def _allowed_community_file(path: Path) -> bool:
    relative = path.relative_to(ROOT / "community_models")
    parts = relative.parts
    lowered = {part.lower() for part in parts}
    # Platform executables stay in Git; portable wheels ship their source.
    if parts[:2] == ("rosetta", "bin"):
        return False
    if any(part.endswith(".egg-info") or part.startswith(".") for part in parts):
        return False
    # AF3's downloaded CCD snapshot is generated runtime state, not source.
    # Git metadata normally excludes it, but wheel builds from an sdist use the
    # no-Git fallback below; keep the fallback from copying the multi-hundred-MB
    # CIF/pickle files into ``partidesign/_resources``.  A hand-authored README
    # remains eligible if a future checkout adds one.
    if len(parts) >= 2 and tuple(part.lower() for part in parts[:2]) == ("af3", "ccd"):
        return path.name.lower() == "readme.md"
    if any(part.lower().startswith("weight") for part in parts[:-1]):
        return path.name == "__init__.py"
    # Full ESMFold2 CCD chemistry is an external runtime asset.  Compact
    # native CCD tables live under configs/resources and are packaged; the
    # multi-hundred-megabyte common/ files must never leak through the sdist
    # no-Git fallback or into a wheel.
    if len(parts) >= 2 and parts[0].lower() == "esmfold2" and parts[1].lower() == "common":
        return False
    if lowered & _BLOCKED_COMMUNITY_DIRS:
        return False
    if any(part.lower().startswith("test_") for part in parts):
        return False
    if path.name.startswith("test_") or path.name.endswith("_test.py"):
        return False
    if path.suffix.lower() == ".ipynb":
        return False
    suffix = path.suffix.lower()
    if parts[:3] == ("rosetta", "source", "dalphaball") and suffix in {".c", ".h", ".f"}:
        return True
    if suffix in _BINARY_SUFFIXES:
        return False
    if suffix in _WEIGHT_SUFFIXES:
        # ccd.pkl is a chemistry data table, not a model weight.
        if relative.as_posix() != "esmfold2/common/ccd.pkl":
            return False
    return True


def _copy_resources(build_lib: Path) -> None:
    destination_root = build_lib / _RESOURCE_PACKAGE
    if destination_root.exists():
        shutil.rmtree(destination_root)
    resource_paths: list[Path] = []
    for root in (ROOT / "configs", ROOT / "licenses"):
        if root.is_dir():
            resource_paths.extend(path for path in root.rglob("*")
                                  if path.is_file()
                                  and not path.is_relative_to(ROOT / "configs" / "customs")
                                  and not any(part in _BLOCKED_COMMUNITY_DIRS or part.startswith(".")
                                              for part in path.relative_to(root).parts)
                                  and path.suffix not in {".pyc", ".log", ".out", ".err"})
    resource_paths.extend(path for path in _git_community_files() if path.is_file() and _allowed_community_file(path))
    # Curated native tool source and licenses are part of the portable wheel.
    tool_root = ROOT / "community_models" / "rosetta"
    if tool_root.is_dir():
        resource_paths.extend(path for path in tool_root.rglob("*")
                              if path.is_file() and _allowed_community_file(path))
    for source in sorted(set(resource_paths)):
        _copy_file(source, destination_root / source.relative_to(ROOT))


def _copy_source_closure(build_lib: Path) -> None:
    # Reused build directories must not retain removed source modules or data.
    source_destination = build_lib / "src"
    if source_destination.exists():
        shutil.rmtree(source_destination)
    _copy_python_tree("src/design", build_lib, exclude_dirs=set())
    _copy_python_tree("src/validate", build_lib, exclude_dirs={"tests", "__pycache__"})
    _copy_python_tree("src/scoring", build_lib, exclude_dirs=set())
    _copy_python_tree("src/cluster", build_lib, exclude_dirs=set())
    _copy_python_tree("src/postfilter", build_lib, exclude_dirs={"__pycache__", "tests"})
    _copy_python_tree("src/common", build_lib, exclude_dirs={"__pycache__"})
    _copy_selected_files(
        ("src/__init__.py", "src/design/utils/notebook_molstar.js", "src/design/utils/notebook_command.js", "src/design/utils/notebook_help.js", "src/design/utils/notebook_preset_load.js", "src/design/utils/notebook_action.js", "src/design/utils/notebook_theme.css"), build_lib
    )


class Build(_build):
    def initialize_options(self) -> None:
        super().initialize_options()
        self.build_base = str(ROOT / ".cache" / "build" / "package")


class EggInfo(_egg_info):
    def initialize_options(self) -> None:
        super().initialize_options()
        base = ROOT / ".cache" / "build" / "metadata"
        base.mkdir(parents=True, exist_ok=True)
        self.egg_base = str(base)


class BuildPy(_build_py):
    def run(self) -> None:
        super().run()
        build_lib = Path(self.build_lib)
        _copy_source_closure(build_lib)
        _copy_resources(build_lib)


class SourceDist(_sdist):
    def initialize_options(self) -> None:
        super().initialize_options()
        self.dist_dir = str(ROOT / ".cache" / "dist")

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        filtered = [
            name for name in files
            if not any(part.endswith(".egg-info") for part in Path(name).parts)
        ]
        super().make_release_tree(base_dir, filtered)


setup(
    packages=_packages(),
    package_dir={"": "."},
    cmdclass={"build": Build, "egg_info": EggInfo, "build_py": BuildPy, "sdist": SourceDist},
    include_package_data=True,
    package_data={
        "partidesign": ["_resources/**/*", "_resources/**"],
        "src.design.utils": ["notebook_molstar.js", "notebook_command.js", "notebook_help.js", "notebook_preset_load.js", "notebook_action.js", "notebook_theme.css"],
    },
)
