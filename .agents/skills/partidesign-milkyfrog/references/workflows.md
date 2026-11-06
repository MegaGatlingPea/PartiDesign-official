# Common PartiDesign workflows

Use this reference for setup, generation, saved-structure operations, and output
handling. Read [task-recipes.md](task-recipes.md) for molecular task authoring,
[validation.md](validation.md) for post-generation evaluation, or
[postfilter.md](postfilter.md) for optional CPU screening of successful protein
complex refolds. Repository paths below refer to the dynamically located
active checkout, not this skill folder.

For a first generation command, read [A first run](../../../../docs/design.md#a-first-run).
See [Capabilities](../../../../README.md#capabilities) for illustrated protein, DNA,
RNA, and ligand design.

## Locate the runtime and check prerequisites

First establish the user's working directory, active Python environment, and
selected model root. If needed, inspect the active package without loading a
model:

```bash
command -v partidesign
python -c "import sys, partidesign; from partidesign.paths import code_root, resource_root, model_dir; print(sys.executable); print(partidesign.__file__); print('Code:', code_root()); print('Resources:', resource_root()); print('Models:', model_dir())"
partidesign --help
```

Use the matching `README.md` and `docs/environments.md` for setup and tested
versions. `partidesign_cu12.yaml` creates `partidesign`;
`partidesign_cu13.yaml` creates `partidesign_cu13` with the optional BioIR provider.
Both target Linux/NVIDIA and support PyRosetta, DSSP, and DAlphaBall for CPU
postfiltering; a source checkout provides DAlphaBall. The default uv route uses
`bash install.sh`, followed by `source .venv/bin/activate`. Select
`--installer conda` explicitly for the named Conda/pip route.
It pairs a locked virtual environment with native Conda dependencies. Keep the
pair intact, use separate paths for CUDA 13, and follow `docs/environments.md`
for `--plan`, updates and `--pip-only` recovery. Do not substitute
`pip install .[full]` for the uv installer.
Create or change an environment only when that is part of the requested work;
do not install dependencies merely to answer a usage question. For requested
execution, inspect `nvidia-smi` and the active PyTorch/CUDA build if readiness is
uncertain. A CPU asset check does not establish GPU readiness.

Interactive help uses centered bordered tables and left-aligned section titles.
Its small MilkyFrog header displays original bundled PNG/JPEG images only when the terminal
supports native images: root `--help` uses the standing image; other human views
choose from six images once per process, independently of model seeds. Unsupported terminals use
text only. Redirected help, `NO_COLOR`, and `TERM=dumb` stay plain; do not treat
terminal styling as a change to command flags, JSON plans, or scientific output.
See `docs/commands.md#terminal-appearance` for SSH and terminal support.

Read `docs/download.md` and `docs/check.md` for asset setup. Begin with inventory
and ordinary checks, which do not download or run model inference:

```bash
partidesign download --list
partidesign check
```

Bare `check` shows four compact sections: Machine overview, PartiDesign models,
Sequence design, and Structure prediction. The machine rows show CPU/GPU usage,
RAM and free model-disk space; model rows show Usage, Size and Status.
In interactive `download`, a checkbox selects a download and a separate ✓ marks
existing files. Fast and standard share the same ESM-C safetensors.

The PartiDesign and PartiScore checkpoints are available from the pinned
[project HF release](https://huggingface.co/MegaGatlingPea/PartiDesign):

```bash
partidesign download partidesign partiscore
```

They are stored as `ckpts/partidesign.ckpt` (4,909,825,762 bytes) and
`ckpts/partiscore.ckpt` (25,350,931 bytes) under the selected model root.
The generation bundle includes the model, autoencoder and experiment configuration;
a flow-matching-only training checkpoint is insufficient. An explicit compatible
path remains available through `design --ckpt FILE` and the matching task check.

Model-root priority is `--model-dir`, `PARTIDESIGN_MODEL_DIR`, then the detected
checkout. An external root contains `ckpts/` and `community_models/`; use
`partidesign download partidesign partiscore --model-dir PATH` for project weights,
and copy any needed Git-supplied CyclicMPNN files into their matching locations.
Code and static chemistry resources stay with the installed package. Selecting
a model root does not change where task inputs and outputs resolve.

For ordinary protein, nucleic-acid and antibody pipelines, download both ESMFold2 variants:

```bash
partidesign download esmfold2-fast esmfold2-standard
```

The PPI preset uses fast iteration and AF2-IG final; prepare `esmfold2-fast`,
`af2`, and the selected MPNN. For explicitly selected Mol-1 validation, use:

```bash
partidesign download mol1
partidesign check --model mol1 --msa off
```

Mol-1 code is bundled under `community_models/mol1/source`; the download fills
`checkpoint/` and `common/` under the selected model root. It reuses the AF3
context MSA/template cache rather than a separate search installation. Protein
MSA and templates default to ColabFold; RNA is searched only with `source: af3`,
and DNA is not searched. `partidesign msa` prepares the same tasks before a run.
The shared cache stays under `<model_dir>/.cache/af3_msa/`; see `docs/msa.md`.
Parameter usage terms are separate from the source-code license.

Prepare only backends needed by the requested workflow. Basic generation needs
no separate predictor or sequence-redesign model. Validation backend targets
are listed in [validation.md](validation.md). `check --load` strictly loads
models on CPU and may need substantial RAM; `check --smoke` runs real bounded
jobs. Neither flag belongs in a parser/config-only check. PartiScore smoke can
run on CPU; native-generation smoke requires its bundle and CUDA.

## Generate from a checked task

Read `docs/design.md`; for task fields use `docs/task_schema.md` and the relevant
section of `configs/examples/README.md`. Inspect the named task's sample count,
length plans, scoring, refinement, and search settings before running it.

The bundled `protein_fixture` requests five length-40 proteins. These examples
use batch size one and automatic execution while retaining all five requests:

```bash
partidesign design configs/examples/protein.yaml \
  --task protein_fixture --run-name first_run \
  --batch-size 1 --show-config

partidesign check configs/examples/protein.yaml --task protein_fixture
```

`--show-config` resolves configuration and basic task fields without loading
weights, probing the GPU, or generating structures. The ordinary task check
additionally inspects sources, routing, environment, and requested assets.
Detailed compilation and stochastic length/crop validity have further
execution-time checks. Use the
same checkpoint, routing, and runtime overlays for checking and generation.

For user-requested execution after prerequisites pass:

```bash
CUDA_VISIBLE_DEVICES=0 partidesign design configs/examples/protein.yaml \
  --task protein_fixture --run-name first_run \
  --batch-size 1
```

`--task` is repeatable; omitting it selects every task in the file. `sample_num`
is per length plan; multiple polymer design chains use explicit `joint` length
rows, not a Cartesian product. `--batch-size` does not reduce this budget. A
smaller diagnostic trial requires an intentional task copy/change, not a silent
change to the request.

Execution and compilation default to `auto`. The expanded count includes all
selected tasks, length plans, `sample_num`, and Best-of-N replicas. At 100
candidates or more, automatic execution uses native dynamic compilation; below
100 it uses BioIR when usable, otherwise native eager. BioIR uses graph replay
for eligible inputs and eager kernels otherwise. Selection is made once per
submission. CPU checks leave automatic GPU selection pending;
the runtime log and saved config report the request, resolved choice, reason,
and count. Use `docs/design.md`, Execution and compilation, for explicit backend,
compile, and strategy combinations. Keep automatic defaults in examples unless
the user requests an explicit policy; automatic fallback is not an out-of-memory
retry.

Automatic bundle routing is the normal choice. Normal and antibody submissions
must be separate. Force `--route-ckpt` or disable routing only with a compatible
checkpoint and a reason from the request; the `pep` member needs a version-2
six-member bundle. `--config` is a runtime overlay, not a second task YAML.
Avoid conflicting dedicated flags and `--override` values for the same field.

Generation writes `<run-root>/<task>/resolved_config.yaml`, `result.csv`,
`logs/design.log`, and successful CIF files. The default run root is
`inference/<run-name>`; `--out` supplies the root, with task names appended.
Each selected task directory must be new or empty. Preserve a requested output
path; if it is occupied, report the conflict and resolve it without clearing
saved results. A new run name/root is the normal way to generate again.

Use `result.csv` to identify final structures and per-sample errors. Failed
samples have no placeholder CIF. Normal rows include `error` and
`postprocess_status`; antibody result columns differ. Scoring/refinement can
change final filenames, so do not infer success or identity from filenames alone.

## Optional Best-of-N ranking

Search and reward are disabled by default. Add them only when requested; search,
task `score: true` (final PartiScore), and `postprocess: true` (OpenMM refinement)
are separate choices. Preview an optional clash-reward run with:

```bash
partidesign design configs/examples/protein.yaml \
  --task protein_fixture --run-name ranked_run \
  --batch-size 1 \
  --override search=best_of_n --override reward=clash \
  --override search.best_of_n.replicas=4 --show-config
```

Here five base requests × four replicas plan 20 candidates. Clash reward needs
no extra learned reward weights. Successful replicas are retained with ranking
information; Best-of-N does not promise one saved structure per base request.
Read the selected reward config before choosing another reward or its assets.

## Score existing structures

Read `docs/score.md`. Input directories must contain direct-child CIF/PDB files;
the command does not recurse, and each directory must use one format. A
generation `result.csv` is not required. Pass additional directories as further
positional arguments.

```bash
partidesign score inference/first_run/protein_fixture --show-config
partidesign score inference/first_run/protein_fixture --device cpu
```

Execution writes a separate `score.csv` in each input directory, using PartiScore
without the generation bundle. Inspect status/error and interface eligibility
along with scores. Existing reports are rejected by default; `--overwrite true`
explicitly replaces them when recomputation is intended. This command does not
rename structures or update generation `result.csv`.

## Relax saved structures with OpenMM

Read `docs/relax.md`. Supply the original matching task YAML and its source
inputs, plus the existing **run root**, not an individual task directory.
Selected tasks need `postprocess: true` and `override: true`; the command skips
tasks with either false rather than enabling them. It does not load generation
weights.

For an existing `ligand_fixture` run whose YAML enables refinement, a bounded
plan and CPU relaxation command are:

```bash
partidesign relax configs/examples/ligand.yaml --task ligand_fixture \
  --out inference/ligand_example --platform CPU --limit 1 --plan

partidesign relax configs/examples/ligand.yaml --task ligand_fixture \
  --out inference/ligand_example --platform CPU --limit 1
```

Use the requested sample scope; `--limit 1` here is a diagnostic example. CPU
refinement is sequential; CUDA uses Ray on visible GPUs. Successful refinement
replaces selected CIFs and updates `result.csv`; failures retain the usable base
structure. Completed rows are normally skipped. `--rerun` includes completed
samples and permits replacement of their refined staging files. If task scoring
is enabled, refinement may update scores, ranking, and filenames too; consult
the updated report.

## Postfilter successful protein and peptide complexes

Use [postfilter.md](postfilter.md) for CPU relaxation and structural screening
of passing protein–protein and standard peptide–protein complex refolds,
including antibody–antigen assemblies and physical head-to-tail cyclic peptides.
Inputs can come from named validation successes or accepted pipeline final
representatives. Keep their saved acceptance records and complete complex
structures together. Nucleic acids, ligands, and monomers are unsupported.
Clustering is a separate step.

## Cluster passing polymer backbones

Read `docs/cluster.md` after validation has produced `success.csv` with suitable
criteria. Accepted inputs are run/task directories, validation method directories,
or their `success/` directories. Ligand-only tasks are unsupported.

```bash
partidesign cluster inference/first_run --task protein_fixture --plan
partidesign cluster inference/first_run --task protein_fixture \
  --tm-threshold 0.5 --workers 4
```

`--plan` inspects selected passing backbones and comparison counts without
alignment or report writes. Execution runs US-align on CPU; use `--usalign FILE`
when discovery in the active environment/PATH fails. Clustering compares the
original generated polymer backbones referenced by the current `result.csv`,
not refold coordinates. Duplicate passing sequence slots from one backbone
collapse; tasks, methods, criteria, and polymer compositions stay separate.

Each selected `validate/<method>/` receives `clusters.csv` and
`cluster_summary.csv`. Reruns recalculate and replace these reports. Empty
success tables yield zero-count reports; no clusters does not itself mean the
alignment command failed.
