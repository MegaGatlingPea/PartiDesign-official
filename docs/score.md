# `partidesign score`

[Command index](commands.md)

Calculate PartiScore for existing structures. PartiScore is an MDN network for
molecular interface scoring. This command uses the bundled PartiScore weights;
it does not require the PartiDesign generation bundle.

## Inputs and a minimal example

Supply one or more directories containing `.cif` or `.pdb` files as direct
children. The command does not recurse through a run directory. Each directory
must use one format; a single directory cannot mix PDB and CIF files.

For the task directory created in the [first generation example](design.md#a-first-run):

```bash
partidesign score inference/first_run/protein_fixture --show-config
partidesign score inference/first_run/protein_fixture --device cpu
```

The first command previews the resolved configuration without scoring. The
second performs inference on CPU. Supply additional directories as further
positional arguments to score several groups together.

## Key options

Defaults come from [score.yaml](../configs/score.yaml).

| Option | Meaning and default |
| --- | --- |
| `INPUT_DIR ...` | One or more directories with direct-child PDB/CIF files. |
| `--model-dir PATH` | Select the model root containing the PartiScore weights. |
| `--ckpt FILE` | Explicit PartiScore checkpoint; default `ckpts/partiscore.ckpt` under the selected model root. |
| `--device DEVICE` | Scoring device; default `auto`. Use `cpu` for CPU execution. |
| `--batch-size N` | Scorer inference batch size; default 32. |
| `--num-workers N` | Structure-preparation workers; default 16. |
| `--overwrite true\|false` | Replace existing `score.csv` files; default false. |
| `--config FILE` | Score configuration overlay. |
| `--override KEY=VALUE` | Typed score configuration override; repeatable. |
| `--show-config` | Print settings without scoring. |

Use `--show-config` after changing configuration. Avoid conflicting values in a
dedicated flag and an override for the same field.

## Output and reuse

Each input directory receives `score.csv`. It contains structure-level and
interface-level scores, partner identities, status/error information, and model
metadata. Inspect status as well as the score; a structure without an eligible
interface does not provide an interface score.

The standalone report is separate from generation-time scoring in `result.csv`.
It does not rename the structures or update the task's generation report.
An existing `score.csv` is rejected by default rather than resumed. To recompute:

```bash
partidesign score inference/first_run/protein_fixture \
  --device cpu --overwrite true
```

## Troubleshooting

| Problem | Next step |
| --- | --- |
| No input structures are found | Pass the task directory containing the CIF/PDB files, not its parent run directory. |
| PDB and CIF files are mixed | Put the two formats in separate input directories. |
| `score.csv` already exists | Keep the saved report, or explicitly recompute with `--overwrite true`. |
| PartiScore weights are missing | Run `partidesign check --model partiscore`; for an external model root, copy its Git-supplied weights to the matching location. |
| A structure fails or has no eligible interface | Inspect `status` and `error` in `score.csv`, then check the structure and chain composition. |
| GPU scoring exhausts memory | Reduce `--batch-size`, or choose CPU execution with sufficient host RAM. |

[Command index](commands.md) · [Check assets](check.md) · [Generate designs](design.md)
