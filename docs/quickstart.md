# Quickstart

Run the installed CLI from the repository root. The CLI resolves the repository
as the default model root, so the examples do not need a model-directory flag.

```bash
cd /path/to/partidesign-inference
```

All task, structure, and output paths below are relative to this directory.

## 1. Check a task on CPU

The external protein fixture is a compact binder task. `check` compiles the task
structure and performs a read-only CPU check without generating structures:

```bash
partidesign check configs/examples/protein.yaml \
  --task protein_fixture
```

Use `--model` or `--all` when you also want to inspect registered model assets.
`--load` adds strict CPU loading; `--smoke` runs bounded real models and is a
separate, resource-consuming check.

## 2. Resolve the design configuration

`design --show-config` selects the task and prints the resolved configuration
without generating structures or loading the checkpoint:

```bash
partidesign design configs/examples/protein.yaml \
  --task protein_fixture \
  --batch-size 1 \
  --run-name guide_protein \
  --show-config
```

The positional task file is a top-level mapping from names to task dictionaries.
Repeat `--task NAME` to select several tasks, or omit it to use every task in file
order. The selected tasks must all use the same format (`normal` or `antibody`).

## 3. Generate a small fixture run

The fixture uses
`configs/examples/structures/protein_design/prot_bind_prot.pdb` and one 40-residue
length plan:

```bash
CUDA_VISIBLE_DEVICES=0 partidesign design \
  configs/examples/protein.yaml \
  --task protein_fixture \
  --batch-size 1 \
  --run-name guide_protein
```

The task leaf is written under `./inference/guide_protein/protein_fixture/`.
Each selected leaf must be new or empty; use another `--run-name` for a second
attempt.

The default normal run uses the generated FM sequence and structure. Public tasks
default to `rm_token_type: [CYS]` for free protein identities; RNA `C` and DNA `DC`
are different token names, and fixed or contextual identities are retained. The
run does not invoke MPNN, refolding, PartiScore, or OpenMM automatically. Add
explicit search and reward overrides when candidates should be generated and ranked:

```bash
CUDA_VISIBLE_DEVICES=0 partidesign design \
  configs/examples/protein.yaml \
  --task protein_fixture \
  --batch-size 1 \
  --run-name guide_protein_ranked \
  --override search=best_of_n \
  --override reward=esmfold2 \
  --override search.best_of_n.replicas=2 \
  --override reward.sequence.source=design
```

Fold and PartiScore rewards require `search=best_of_n`. A reward ranks candidates;
task-level `score: true` writes a separate PartiScore result, and
`postprocess: true` selects the separate OpenMM stage. Best-of-N writes every
successful ranked replica. With `replicas=2`, inspect two rank-qualified structure
files and two ranked rows in `result.csv`; `rank1` is the highest-scoring row.

## 4. Inspect the saved run

Each selected task has a resolved configuration, ordinary log, result table, and
structure files:

```text
inference/guide_protein/protein_fixture/
  resolved_config.yaml
  result.csv
  logs/design.log
  sample*.cif
```

See [Outputs](outputs.md) for row fields, status values, multi-chain filenames,
and validation files.

## 5. Plan and validate

Plan validation reads the existing run and reports what would be selected without
loading a validation model or writing a report:

```bash
partidesign validate ./inference/guide_protein \
  --method esmfold2-fast \
  --source gt:1 \
  --plan
```

After reviewing the plan, run a model-backed validation:

```bash
CUDA_VISIBLE_DEVICES=0 partidesign validate ./inference/guide_protein \
  --method esmfold2-fast \
  --source gt:1 \
  --criteria af3_basic
```

The current methods are `af2_ig`, `esmfold2-fast`, `af3`, `af3-la`, `partiscore`,
and `vina_dock`. Generic protein and cyclic tasks default to `gt:1` plus
`solublempnn:1`; nucleic-acid tasks default to `gt:1` plus `na_mpnn:1`. An
explicit source list replaces that default. `auto` is a reward-only source;
design rewards keep the generated sequence with `reward.sequence.source=design`,
while `reward.sequence.source=auto` can select the compatible NA backend. See
[Validation](validation.md) for criteria, maintenance, and reruns.

## 6. Assets and relaxation

Inspect or prepare registered model assets with the installed CLI. These commands
use the repository-default model root unless `--model-dir` is supplied:

```bash
partidesign download --list
partidesign download proteinmpnn ligandmpnn
```

The source column marks PartiScore and CyclicMPNN as `with git repo`: their
weights arrive with the clone. `download URL` identifies registered remote
sources, and `not configured` means a compatible URL has not been registered.
The native bundle currently needs to be placed at `ckpts/partidesign.ckpt`
manually; it has no published download URL yet.

OpenMM relaxation operates on an existing run and task file. First run the
[ligand fixture](tasks/ligand.md), which creates `./inference/ligand_fixture`.
Then plan the relaxation, and repeat without `--plan` when the selected samples
and platform are correct:

```bash
partidesign relax configs/examples/ligand.yaml \
  --task ligand_fixture \
  --out ./inference/ligand_fixture \
  --platform CPU \
  --plan

partidesign relax configs/examples/ligand.yaml \
  --task ligand_fixture \
  --out ./inference/ligand_fixture \
  --platform CUDA
```

`relax` retries the OpenMM stage for existing successful inference samples; it does
not generate new FM structures. Use `--rerun` to retry completed samples explicitly.

## 7. Compatibility module entry points

The installed CLI is the primary interface. A checkout also retains the direct
module entrances for compatibility; run them only when a workflow explicitly
requires Hydra syntax:

```bash
python -m src.design \
  tasks_file=configs/examples/protein.yaml \
  'task_names=[protein_fixture]' \
  run_name=guide_module --show-config

python -m src.validate ./inference/guide_protein \
  --method esmfold2-fast --source gt:1 --plan
```

The module examples use the same external task file and resolve relative paths
from the repository root.
