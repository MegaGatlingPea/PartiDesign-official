# BioIR kernel runtime for PartiDesign

This source directory builds **`partidesign-bioir-kernels==0.2.0+partidesign.1`**, a
separately named distribution providing the original `bionemo_ir` Python
namespace for PartiDesign's BioIR kernel integration. It is derived from NVIDIA's
BioNeMo Inference Runtime 0.2.0 wheel. It does not represent an official NVIDIA
release or support the complete BioIR model, preprocessing, serving, or pipeline
APIs.

The build backend uses only Python's standard library. It downloads the pinned
upstream wheel over HTTPS, verifies its exact size and SHA256, and repackages it
without importing Torch or compiling CUDA code. Every `bionemo_ir/` member,
including Python modules, data and the native extension, is retained byte for
byte. All upstream license and third-party notice files are preserved. Only the
distribution metadata and `RECORD` are rewritten; official console entry points
and invalidated wheel signatures, if present, are omitted.

## Install

Use the complete `partidesign_cu13` environment described in the repository's
installation documentation. To install this component with normal dependency
resolution from the repository root:

```bash
python -m pip install ./runtime/bioir-kernels
```

The available upstream binary supports **Linux x86_64, glibc 2.34 or newer, and
CPython 3.12**. The backend rejects unsupported build platforms. The first build needs a network connection to `files.pythonhosted.org`. The
verified upstream wheel is cached at
`${XDG_CACHE_HOME:-$HOME/.cache}/partidesign/bioir-kernels/` under its original
filename. Metadata preparation and wheel building can reuse this cache offline;
both the file size and SHA256 are verified on every reuse. Downloads use a
temporary file in the cache directory and are published with an atomic rename
only after verification. An invalid cached file is rejected; remove it and
retry to download a fresh copy. No SDK or compiled artifacts are left in the
source directory.

`partidesign-bioir-kernels` and official `bionemo-ir` share a Python namespace;
install them in separate environments. The distribution version identifies this
packaging revision. The unchanged `bionemo_ir.__version__` remains `0.2.0`.

## Dependency scope

The declared requirements cover the audited PartiDesign kernel import closure,
including eagerly imported registry/hub utilities: Torch 2.11, CUDA Python 13,
cuEquivariance 0.11.1–0.13, Triton 3.6–3.8, NumPy 2, CUTLASS DSL 4.5.2, cuDNN
frontend 1.23.0, TVM FFI, lru-dict, einops, Pydantic, PyYAML and Hugging Face Hub.

The upstream full-pipeline pins for SciPy, RDKit, Ray, Biotite and Biopython are
not kernel dependencies and are excluded from this distribution. PartiDesign's
complete environment supplies its own versions for the features that use those
libraries. Preserving pipeline source files supports dynamic kernel imports; it
does not make all upstream pipeline features available with these requirements.
The original upstream long description and attribution remain in `METADATA`;
this README describes the supported scope of the derived distribution.

Building and verifying a wheel establishes packaging and byte integrity only.
Full-environment dependency checks and real GPU validation are required before
claiming compatibility or performance for a particular machine.

## Pinned source and license

- Upstream project: [NVIDIA BioNeMo Inference Runtime](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime)
- Upstream version: `0.2.0`
- Wheel: [`bionemo_ir-0.2.0-cp312-cp312-manylinux_2_34_x86_64.whl`](https://files.pythonhosted.org/packages/5f/d3/b2e9b3f1aa5b24bf499a0bd3816739a26d7b2b2aee477a36a940ca5e9302/bionemo_ir-0.2.0-cp312-cp312-manylinux_2_34_x86_64.whl)
- SHA256: `afc64f92a3b4ac96d23840136201fe0e21317ef2918437690718e6c4cc0d74f0`

NVIDIA's copyright and Apache-2.0 license remain applicable to its code. The
wheel also preserves `THIRD_PARTY_NOTICES.md` and all bundled third-party license
texts under the renamed distribution's `licenses/` directory. No NVIDIA kernel
source or binary is modified by this packaging component.
