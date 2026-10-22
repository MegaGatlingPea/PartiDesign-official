# `partidesign relax`

[Command index](commands.md)

Retry OpenMM refinement of existing PartiDesign CIF results without generating
new molecules or loading the generation model. Use the original task names,
source inputs, and chemistry to interpret the saved results correctly.

## Inputs and a minimal example

The input is a task YAML plus an existing **run root** containing
`<task-name>/result.csv` and its CIF files. The selected task must have
`postprocess: true` and `override: true`; `override` defaults to true. Tasks
with either flag false are skipped. The command does not enable those flags.

For an existing run of `ligand_fixture`, whose shipped YAML enables refinement:

```bash
partidesign relax configs/examples/ligand.yaml --task ligand_fixture \
  --out inference/ligand_example --platform CPU --limit 1 --plan

partidesign relax configs/examples/ligand.yaml --task ligand_fixture \
  --out inference/ligand_example --platform CPU --limit 1
```

The first command counts eligible saved samples without refining them. Remove
`--plan` for actual execution. CPU refinement runs sequentially. To use the
visible GPUs through Ray:

```bash
CUDA_VISIBLE_DEVICES=0 partidesign relax configs/examples/ligand.yaml \
  --task ligand_fixture --out inference/ligand_example --platform CUDA
```

`--out` must point to the existing run root, not the individual task directory.
For example, use `inference/ligand_example`, not its `ligand_fixture` subdirectory.

## Key options

Defaults come from [relax.yaml](../configs/relax.yaml).

| Option | Meaning and default |
| --- | --- |
| `TASK_FILE.yaml` | Task definitions matching the existing outputs. |
| `--task NAME` | Select a task; repeatable. Default: all tasks in the file. |
| `--out PATH` | Existing inference run root; required. |
| `--platform CPU\|CUDA` | Refinement platform; default CUDA. CPU is sequential; CUDA uses Ray. |
| `--charge-method gasteiger\|am1bcc` | Ligand charge method; default `gasteiger`. |
| `--limit N` | Positive maximum number of samples to retry per task. |
| `--rerun` | Include completed samples and allow their refined staging files to be replaced. |
| `--plan` | Count selected samples without refinement. |
| `--model-dir PATH` | Model root for any requested scoring assets. |
| `--config FILE`, `--override KEY=VALUE` | Runtime configuration overlay and repeatable overrides. |

## Outputs and reuse

Successful refinement replaces the selected structure and updates `result.csv`.
Failed refinement retains the usable base structure. Rows whose refinement
already succeeded are normally skipped; use `--rerun` to include them again.

When task scoring is enabled, the command can also update final PartiScore,
ranking, and structure filenames. Read the updated `result.csv` for final file
identities. A task that requests scoring needs the PartiScore resources even
though the generation checkpoint is not loaded.

## Troubleshooting

| Problem | Next step |
| --- | --- |
| The plan selects no samples | Check the task's `postprocess`/`override` flags, the task selector, and whether all eligible rows already succeeded. |
| `result.csv` is missing | Point `--out` at the original run root and select the original task name. |
| A structure identity or chemistry check fails | Use the matching task YAML and source inputs; inspect the reported sample and its `result.csv` row. |
| Refinement fails | Inspect the sample's status/error, verify its atoms and requested chemistry, and try a bounded CPU run for diagnosis. |
| A saved refined staging file is reported invalid | Inspect the affected result, then use `--rerun` if you intend to replace that refinement. |

[Command index](commands.md) · [Generate designs](design.md) · [Score structures](score.md)
