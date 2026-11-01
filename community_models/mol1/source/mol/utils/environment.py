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
"""Read-only environment diagnostics for the ``doctor`` command.

Nothing here loads a model or runs inference: the report must come back in
about a second so it is usable as the first thing to try when something is
wrong. Every probe answers a question that has actually cost us time.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mol.config.inference import inference_configs
from mol.data.sources import RELATIVE_PATH, checkpoint_key

# Optional packages the GPU kernels need. Absent is fine -- the triangle
# settings fall back to the PyTorch path -- but "installed yet not importable"
# is the case worth naming, because it looks identical to absent from the
# outside while usually meaning a CUDA-major-version mismatch.
GPU_OPTIONAL_MODULES = (
    "cuequivariance_torch",
    "cuequivariance_ops_torch",
    "triton",
)

# Binaries the MSA and template preprocessing shell out to.
SEARCH_BINARIES = ("hmmsearch", "hmmbuild", "nhmmer", "hmmalign", "kalign")

# The shared-data download script uses either one when installed and falls back
# to Python without them. Reported separately from the search binaries.
DOWNLOAD_BINARIES = ("wget", "curl")

# Caches the inference runner expects under MOL_ROOT_DIR.
RUNTIME_CACHES = (
    "common/components.cif",
    "common/components.cif.rdkit_mol.pkl",
)


@dataclass
class ModuleStatus:
    name: str
    summary: str


def _exception(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _distribution_version(module_name: str) -> Optional[str]:
    """Version of the distribution that provides an importable module.

    The import name and the distribution name can differ:
    ``cuequivariance_ops_torch`` ships as ``cuequivariance-ops-torch-cu13`` or
    ``-cu12``, so the distributions providing the module are looked up first.
    """
    try:
        candidates = importlib.metadata.packages_distributions().get(module_name, [])
    except Exception:
        candidates = []
    for distribution in [*candidates, module_name.replace("_", "-")]:
        try:
            return importlib.metadata.version(distribution)
        except Exception:
            continue
    return None


def probe_optional_module(name: str) -> ModuleStatus:
    """Import a module once and report installed / absent / broken."""
    version = _distribution_version(name)
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        return ModuleStatus(name, "not installed")
    except Exception as error:  # installed but unusable
        return ModuleStatus(
            name, f"import failed ({version or 'unknown version'}): {_exception(error)}"
        )
    return ModuleStatus(name, version or "installed (version unknown)")


def _nvidia_smi() -> str:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return "not found"
    try:
        out = subprocess.run(
            [
                binary,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except Exception as error:
        return f"present but failed: {_exception(error)}"
    return "; ".join(line.strip() for line in out.stdout.strip().splitlines())


def _torch_report(lines: list[str]) -> Optional[object]:
    try:
        import torch
    except Exception as error:
        lines.append(f"- PyTorch: NOT IMPORTABLE -- {_exception(error)}")
        return None

    lines.append(f"- PyTorch: {torch.__version__}")
    lines.append(f"- torch CUDA version: {torch.version.cuda or 'none'}")
    try:
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        probe_error = None
    except Exception as error:
        cuda_available, device_count, probe_error = False, 0, _exception(error)
    lines.append(f"- torch.cuda.is_available: {cuda_available}")
    lines.append(f"- CUDA device count: {device_count}")
    if probe_error:
        lines.append(f"- CUDA probe error: {probe_error}")

    # The arch list is the probe that pays for itself. A build without the
    # GPU's architecture, and without a compute_XX PTX entry to JIT from,
    # fails on a bare matmul with "no kernel image is available" -- which
    # reads like a broken install rather than a wrong wheel.
    arch_list = list(getattr(torch.cuda, "get_arch_list", lambda: [])())
    lines.append(f"- torch CUDA arch list: {', '.join(arch_list) or 'none'}")
    for index in range(device_count):
        name = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        total = torch.cuda.get_device_properties(index).total_memory / (1024**3)
        covered = f"sm_{major}{minor}" in arch_list
        ptx = any(entry.startswith("compute_") for entry in arch_list)
        verdict = (
            "covered by this build"
            if covered
            else ("not in arch list, but PTX present to JIT from" if ptx
                  else "NOT SUPPORTED by this build and no PTX fallback")
        )
        lines.append(
            f"- CUDA device {index}: {name}, compute capability {major}.{minor}, "
            f"{total:.0f} GiB -- {verdict}"
        )

    mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
    mps_available = bool(mps_built and torch.backends.mps.is_available())
    lines.append(f"- Apple MPS: built={mps_built} available={mps_available}")
    return torch


def format_doctor_report() -> str:
    """Build the whole report as text. Never raises."""
    lines = [
        "Mol-1 environment",
        f"- Python: {sys.version.split()[0]} ({sys.executable})",
        f"- Platform: {platform.platform()}",
    ]
    torch = _torch_report(lines)
    lines.append(f"- nvidia-smi: {_nvidia_smi()}")

    lines.append("- GPU optional packages:")
    statuses = [probe_optional_module(name) for name in GPU_OPTIONAL_MODULES]
    for status in statuses:
        lines.append(f"  - {status.name}: {status.summary}")

    # What auto will actually choose. This is the question users ask.
    if torch is not None:
        try:
            from mol.runner.inference import select_inference_device

            device = select_inference_device("auto")
            lines.append(f"- device=auto resolves to: {device}")
            from mol.model.triangular.layers import resolve_auto_kernel

            kernel = resolve_auto_kernel("triangle_attention", device)
            lines.append(f"- triangle_*=auto resolves to: {kernel}")
        except Exception as error:
            lines.append(f"- auto resolution failed: {_exception(error)}")

    lines.append("- Preprocessing binaries (MSA and templates):")
    for binary in SEARCH_BINARIES:
        found = shutil.which(binary)
        lines.append(f"  - {binary}: {found or 'not found'}")

    available = [name for name in DOWNLOAD_BINARIES if shutil.which(name)]
    lines.append(
        "- Downloader for the shared caches: "
        + (", ".join(available) if available
           else "none (the download script falls back to Python)")
    )

    root = os.environ.get("MOL_ROOT_DIR")
    lines.append(f"- MOL_ROOT_DIR: {root or 'unset (falls back to the home directory)'}")
    base = Path(root) if root else Path.home()
    weights = RELATIVE_PATH[checkpoint_key(inference_configs["model_name"])]
    lines.append(
        f"  - {weights}: "
        + ("present" if (base / weights).exists()
           else "missing (downloaded by the first prediction)")
    )
    for relative in RUNTIME_CACHES:
        path = base / relative
        lines.append(f"  - {relative}: {'present' if path.exists() else 'missing'}")

    extensions = os.environ.get("TORCH_EXTENSIONS_DIR")
    lines.append(f"- TORCH_EXTENSIONS_DIR: {extensions or 'unset (PyTorch default)'}")
    lines.append(f"- LAYERNORM_TYPE: {os.environ.get('LAYERNORM_TYPE') or 'unset (fast LayerNorm)'}")

    opt_core_dir = os.environ.get("OPT_CORE_DIR")
    try:
        from mol.model.triangular.flashpairformer import _import_trimul

        _import_trimul()
        opt_core = "importable"
    except Exception as error:
        opt_core = f"unavailable: {_exception(error)}"
    lines.append(f"- opt_core (FlashPairformer TMK3): {opt_core}")
    lines.append(f"  - OPT_CORE_DIR: {opt_core_dir or 'unset'}")

    return "\n".join(lines)
