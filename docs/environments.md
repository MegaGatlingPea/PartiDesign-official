# Environments

Run environment commands from the repository root. Model assets stay in `ckpts/`
and `community_models/`, or in `PARTIDESIGN_MODEL_DIR`; switching environments
uses the same weights.

| Definition | Environment | Python | PyTorch | JAX GPU plugins | BioIR |
| --- | --- | --- | --- | --- | --- |
| `partidesign_cu12.yaml` | `partidesign` | 3.12.6 | 2.9.1+cu128 | 0.10.2 / CUDA 12 | No |
| `partidesign_cu13.yaml` | `partidesign_cu13` | 3.12.6 | 2.11.0+cu130 | 0.10.2 / CUDA 13 | Optional backend |

Both definitions include design, antibody and peptide inference, PartiScore,
ProteinMPNN, solubleMPNN, LigandMPNN, CyclicMPNN, NA-MPNN, both ESMFold2 variants,
AF2, AF3 and AF3-LA, HMMER MSA/template tools, USalign clustering, and
OpenMM/PDBFixer/OpenFF/AmberTools refinement with Vina/Meeko/OpenBabel docking.
Both also declare DSSP 4.6.1, libmcfp 2.0.1 and PyRosetta 2026.39 for the
CPU [postfilter workflow](postfilter.md).
The corresponding weights and local MSA databases remain separate downloads.
See [downloads](download.md) and [validation](validate.md).

## Install CUDA 12

The CUDA 12 definition retains the environment name `partidesign`:

```bash
conda env create -f partidesign_cu12.yaml
conda activate partidesign
python -m pip check
partidesign --help
```

## Protein complex postfilter

Both `partidesign_cu12.yaml` and `partidesign_cu13.yaml` include the CPU postfilter
dependencies: the CPython 3.12 PyRosetta `2026.39+release.49e05e797c` build,
DSSP `4.6.1`, compatible `libmcfp 2.0.1`, and `packaging 24.2`. Use the same
environment for design, validation and postfiltering; no separate postfilter
environment is required. Each definition retains its own PyTorch/JAX CUDA stack.

After creating either environment from its normal YAML:

```bash
conda activate partidesign
# Or: conda activate partidesign_cu13
python -m pip check
partidesign postfilter --show-config
```

For an environment created before these dependencies were added, follow the
[manual postfilter setup](postfilter.md#dependencies-and-execution-settings).
Postfilter runs on CPU and needs no new folding predictions. A source checkout
provides DAlphaBall; wheel-only users must provision that executable separately.

## Install CUDA 13

Use Linux x86_64 with glibc >=2.34 and NVIDIA driver >=580. The YAML supplies the
CUDA runtime and compiler tools, including an explicit CUDA 13 OpenMM build.
Do not override them with an older toolkit in `CUDA_ROOT` or `LD_LIBRARY_PATH`.

```bash
conda env create -f partidesign_cu13.yaml
conda activate partidesign_cu13
python -m pip check
partidesign --help
python -c "import torch, jax; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name()); print(jax.devices())"
```

Use a complete checkout: the YAML installs local vendored backends, the BioIR
kernel runtime, and editable PartiDesign/AF3 packages. Relative install paths
require the repository root. Do not use `--no-deps` or ignore resolver errors.

## Move from CUDA 12

Create `partidesign_cu13` alongside `partidesign`, then activate it for the same
commands and model directory. This keeps the working CUDA 12 environment:

```bash
conda activate partidesign_cu13
partidesign check --all
# Switch back when needed:
conda activate partidesign
```

There is no `partidesign update` command. An in-place update of the CUDA 12 prefix
can leave both JAX plugin families or conflicting BioIR providers installed, so
use the separate environment. Recreate an old minimal experimental
`partidesign_cu13` environment from the complete YAML.

## BioIR provider and execution

The YAML installs `partidesign-bioir-kernels==0.2.0+partidesign.1` from
[`runtime/bioir-kernels`](../runtime/bioir-kernels/README.md). This separately
named runtime covers the kernel API used by PartiDesign. Its build verifies the
SHA256 of NVIDIA's pinned 0.2.0 wheel and retains the package files, native
kernels and licenses. The complete upstream SDK pins scientific packages for
other model pipelines; those pipeline dependencies are outside this runtime's
scope. Installation uses the normal dependency resolver.

Do not also install `bionemo-ir`: both distributions supply `bionemo_ir`, and
PartiDesign rejects duplicate providers. A compatible complete official 0.2.x
SDK can still be used in an independently resolved environment.

Both environments default to `model.execution_backend: auto` and
`model.compile.enabled: auto`. Submissions with at least 100 expanded candidates
use native dynamic compilation. Smaller submissions use BioIR when usable, or
native eager execution when unavailable. BioIR uses CUDA Graphs for eligible
inputs and its eager kernels otherwise. The count includes selected
tasks, length plans, `sample_num`, and Best-of-N replicas, and is resolved once
for the submission. The CUDA 12 environment uses native eager for small jobs
when no compatible BioIR provider is installed.

To require BioIR explicitly in the CUDA 13 environment:

```bash
partidesign check configs/examples/protein.yaml \
  --task protein_fixture --execution-backend bioir
partidesign design configs/examples/protein.yaml \
  --task protein_fixture --execution-backend bioir --out ./inference/bioir
```

An explicit BioIR request reports missing or incompatible dependencies as an
error. Automatic small-job selection can fall back to native eager for a
controlled dependency or startup incompatibility; this startup selection does
not suppress out-of-memory or input errors or retry a failed job. Within BioIR,
graph-ineligible inputs and recoverable capture/replay failures can use BioIR
eager execution, including recovery from graph allocation failures. Device-fatal
CUDA errors propagate.
Select `--execution-backend native` to require native execution; its automatic
compilation still follows the 100-candidate threshold. CPU checks and
`design --show-config` do not probe GPU capabilities, so automatic selection is
pending until runtime. See the
[execution guide](design.md#experimental-bioir-execution) for limits and measured
performance. Physical 50-series GPU validation is still pending.

## Verify features

`pip check` checks dependency metadata; `check --all` checks assets and source
availability. Neither alone proves GPU execution. Use `check --model TARGET --load`
for CPU loading and `--smoke` for bounded real inference:

```bash
partidesign check --model partiscore --smoke
partidesign check --model proteinmpnn --smoke
partidesign check --model esmfold2-fast --smoke
partidesign check --model af3 --msa off --smoke
```

Generation, GPU scoring, MSA/template search, OpenMM CUDA refinement and Vina
validation exercise different paths. Run the relevant workflow before a large
job; see [check](check.md), [score](score.md), [relax](relax.md),
[validate](validate.md), and [cluster](cluster.md).

Vina benchmark validation uses a reference ligand in the original source to
define its docking box. Use the formal ligand benchmark inputs for that
workflow; protein-only SMILES design examples do not supply this reference.
