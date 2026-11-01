# Modifications Copyright 2026 PartiDesign contributors
# Inference-only integration: omit training/content-digest helpers.
# Copyright 2026 Molence Bio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build-key helpers for the fused LayerNorm CUDA extension.

This module deliberately has no PyTorch import at module scope.  Keeping the key
generation logic independent makes it possible to validate cache isolation on a
CPU-only development machine.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


_MODULE_COMPONENT_RE = re.compile(r"[^a-zA-Z0-9]+")
_CUDA_RELEASE_RE = re.compile(r"\brelease\s+(\d+\.\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ExtensionBuildInfo:
    """Runtime inputs which uniquely identify a compiled CUDA extension."""

    name: str
    device_index: int
    device_name: str
    capability: tuple[int, int]
    python_tag: str
    torch_tag: str
    cuda_tag: str
    nvcc_tag: str
    nvcc_path: Optional[str]
    abi_tag: str

    @property
    def capability_tag(self) -> str:
        return f"sm{self.capability[0]}{self.capability[1]}"


def _compact_tag(value: object, fallback: str = "unknown") -> str:
    compact = _MODULE_COMPONENT_RE.sub("_", str(value)).strip("_")
    return compact.lower() or fallback


def detect_nvcc(
    cuda_home: Optional[os.PathLike[str] | str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[Optional[str], str]:
    """Return the nvcc path and a version/build version for the cache key."""

    env = os.environ if environ is None else environ
    configured_home = cuda_home or env.get("CUDA_HOME")
    nvcc_path: Optional[str] = None
    if configured_home:
        candidate = Path(configured_home) / "bin" / "nvcc"
        if candidate.is_file():
            nvcc_path = str(candidate)
    if nvcc_path is None:
        nvcc_path = shutil.which("nvcc", path=env.get("PATH"))
    if nvcc_path is None:
        return None, "nvccunknown"

    try:
        output = subprocess.check_output(
            [nvcc_path, "--version"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return nvcc_path, "nvccunknown"

    release_match = _CUDA_RELEASE_RE.search(output)
    release = release_match.group(1) if release_match else "unknown"
    return nvcc_path, f"nvcc{_compact_tag(release)}"



def make_extension_name(
    *,
    python_version: tuple[int, int],
    torch_version: object,
    torch_cuda_version: object,
    capability: tuple[int, int],
    cxx11_abi: Optional[bool],
    python_abi: object = None,
    nvcc_tag: object = "nvccunknown",
) -> str:
    """Build a short module name backed by the complete compatibility key."""

    python_tag = f"py{python_version[0]}{python_version[1]}"
    capability_tag = f"sm{capability[0]}{capability[1]}"
    # The pinned source revision plus runtime ABI separates native build products.
    # PyTorch/Ninja handles compilation; no source-content inventory is read here.
    return "_".join(("mol1_e6e6486_ln", python_tag, capability_tag,
                     _compact_tag(torch_version), _compact_tag(torch_cuda_version),
                     _compact_tag(python_abi), f"abi{int(bool(cxx11_abi))}",
                     _compact_tag(nvcc_tag)))


def resolve_cuda_device_index(
    cuda: Any,
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    """Resolve the process-local CUDA device before ``torch.cuda.set_device``.

    ``torchrun`` exports ``LOCAL_RANK`` before Python imports the model.  Some
    launchers instead expose only one CUDA device per process, so an out-of-range
    local rank falls back to ``current_device()`` and then device zero.
    """

    env = os.environ if environ is None else environ
    if not cuda.is_available():
        raise RuntimeError(
            "Fast LayerNorm requires CUDA. Set LAYERNORM_TYPE=torch to use "
            "PyTorch LayerNorm on CPU-only hosts."
        )

    device_count = int(cuda.device_count())
    if device_count < 1:
        raise RuntimeError("CUDA is available but no visible CUDA devices were found")

    local_rank = env.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            rank_device = int(local_rank)
        except ValueError:
            rank_device = -1
        if 0 <= rank_device < device_count:
            return rank_device

    try:
        current_device = int(cuda.current_device())
    except (AssertionError, RuntimeError):
        current_device = 0
    return current_device if 0 <= current_device < device_count else 0


def collect_extension_build_info(
    torch_module: Any,
    build_inputs: Sequence[os.PathLike[str] | str],
    environ: Optional[Mapping[str, str]] = None,
    cuda_home: Optional[os.PathLike[str] | str] = None,
) -> ExtensionBuildInfo:
    """Collect the current runtime ABI and accelerator into one build key."""

    device_index = resolve_cuda_device_index(torch_module.cuda, environ)
    capability_values = tuple(
        int(part)
        for part in torch_module.cuda.get_device_capability(device_index)
    )
    if len(capability_values) != 2:
        raise RuntimeError(f"invalid CUDA capability: {capability_values!r}")
    capability = (capability_values[0], capability_values[1])

    try:
        device_name = str(torch_module.cuda.get_device_name(device_index))
    except (AssertionError, RuntimeError):
        device_name = "unknown"

    python_version = (sys.version_info.major, sys.version_info.minor)
    torch_version = getattr(torch_module, "__version__", "unknown")
    torch_cuda_version = getattr(
        getattr(torch_module, "version", None), "cuda", None
    )
    cxx11_abi = getattr(
        getattr(torch_module, "_C", None), "_GLIBCXX_USE_CXX11_ABI", None
    )
    python_abi = sysconfig.get_config_var("SOABI")
    nvcc_path, nvcc_tag = detect_nvcc(cuda_home=cuda_home, environ=environ)
    name = make_extension_name(
        python_version=python_version,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        capability=capability,
        cxx11_abi=cxx11_abi,
        python_abi=python_abi,
        nvcc_tag=nvcc_tag,
    )
    return ExtensionBuildInfo(
        name=name,
        device_index=device_index,
        device_name=device_name,
        capability=capability,
        python_tag=(
            f"py{python_version[0]}{python_version[1]}_"
            f"{_compact_tag(python_abi)}"
        ),
        torch_tag=f"torch{_compact_tag(torch_version)}",
        cuda_tag=(
            f"cu{_compact_tag(torch_cuda_version)}"
            if torch_cuda_version
            else "cuunknown"
        ),
        nvcc_tag=nvcc_tag,
        nvcc_path=nvcc_path,
        abi_tag="abiunknown" if cxx11_abi is None else f"abi{int(cxx11_abi)}",
    )
