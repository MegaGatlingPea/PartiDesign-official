# Copyright 2024 ByteDance and/or its affiliates.
# Modifications Copyright 2026 Molence Bio
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
#
# This file has been modified by Molence Bio
# Based on Protenix: https://github.com/bytedance/Protenix
# Upstream revision: bd54a05d047b8925a241056f36d619700604068a
# Upstream file: protenix/model/layer_norm/torch_ext_compile.py
# Changes: Build targets for Blackwell GPUs (SM100-121); cached builds are reused.


import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from torch.utils.cpp_extension import get_default_build_root, load


logger = logging.getLogger(__name__)


def _resolve_build_directory(
    name: str,
    build_directory: Optional[str],
) -> Path:
    if build_directory is not None:
        resolved = Path(build_directory)
    else:
        cache_root = os.environ.get("TORCH_EXTENSIONS_DIR")
        if cache_root is None:
            cache_root = os.path.join(
                get_default_build_root(),
                "mol",
            )
        resolved = Path(cache_root) / name
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


_STALE_LOCK_SECONDS = 300


def _clear_stale_lock(build_directory: Path) -> None:
    """Remove a lock left behind by a killed build.

    ``torch.utils.cpp_extension.load`` guards the build directory with a
    ``lock`` file through ``FileBaton``. A process killed mid-build leaves the
    file behind, and every later process then waits for it forever in
    ``FileBaton.wait()`` -- the symptom is a job that burns its whole wall
    clock with no ninja or nvcc process running and nothing in the log.

    A lock older than the longest plausible build is treated as abandoned. The
    threshold is far above a real compile (~90 s here) and far below the length
    of a job, so a concurrent build is never interrupted.
    """
    lock = build_directory / "lock"
    try:
        age = time.time() - lock.stat().st_mtime
    except OSError:
        return
    if age < _STALE_LOCK_SECONDS:
        return
    try:
        lock.unlink()
    except OSError:
        return
    logger.warning(
        "Removed a stale extension build lock (%.0f s old): %s", age, lock
    )


def _load_cached_extension(name: str, build_directory: Path) -> Optional[Any]:
    library_path = build_directory / f"{name}.so"
    if not library_path.is_file():
        return None

    try:
        spec = importlib.util.spec_from_file_location(name, library_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    except (ImportError, OSError):
        sys.modules.pop(name, None)
        logger.warning(
            "Fast LayerNorm cache entry could not be loaded; rebuilding: %s",
            library_path,
        )
        return None

    logger.info("Fast LayerNorm cache hit: %s", library_path)
    return module


def compile(
    name: str,
    sources: list[str],
    extra_include_paths: list[str],
    build_directory: Optional[str] = None,
) -> Any:
    resolved_build_directory = _resolve_build_directory(name, build_directory)
    _clear_stale_lock(resolved_build_directory)
    cached_extension = _load_cached_extension(name, resolved_build_directory)
    if cached_extension is not None:
        return cached_extension

    logger.info("Fast LayerNorm cache miss: %s", resolved_build_directory)

    # Query supported architectures from nvcc (resolved via PyTorch's
    # CUDA_HOME so we use the same toolchain as cpp_extension.load).
    import re
    import shutil
    import subprocess

    from torch.utils.cpp_extension import CUDA_HOME

    _nvcc = shutil.which("nvcc")
    if CUDA_HOME:
        _candidate = os.path.join(CUDA_HOME, "bin", "nvcc")
        if os.path.isfile(_candidate):
            _nvcc = _candidate

    _supported = set()
    try:
        out = subprocess.check_output(
            [_nvcc, "--list-gpu-arch"], text=True, stderr=subprocess.STDOUT
        )
        _supported = set(re.findall(r"compute_(\d+)", out))
    except Exception:
        _supported = {"70", "80", "86", "90"}  # safe defaults

    _wanted = [
        ("70", "70"),
        ("80", "80"),
        ("86", "86"),
        ("89", "89"),
        ("90", "90"),
        ("100", "100"),
        ("103", "103"),  # Blackwell B300/GB300
        ("120", "120"),  # RTX PRO 6000 Blackwell / RTX 50 series
        ("121", "121"),  # GB10-class Blackwell
    ]
    gencode_flags = []
    available = [(compute, sm) for compute, sm in _wanted if compute in _supported]
    for idx, (compute, sm) in enumerate(available):
        code = f"sm_{sm}"
        if idx == len(available) - 1:
            code = f"[sm_{sm},compute_{compute}]"
        gencode_flags += ["-gencode", f"arch=compute_{compute},code={code}"]
    if not gencode_flags:
        gencode_flags = ["-gencode", "arch=compute_80,code=sm_80"]

    # Build TORCH_CUDA_ARCH_LIST dynamically from supported architectures
    _arch_list = [f"{int(c)//10}.{int(c)%10}" for c, _ in available]
    if _arch_list:
        _arch_list[-1] += "+PTX"
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(_arch_list) if _arch_list else "8.0"

    return load(
        name=name,
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=[
            "-O3",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
            "-maxrregcount=32",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]
        + gencode_flags,
        verbose=True,
        build_directory=str(resolved_build_directory),
    )
