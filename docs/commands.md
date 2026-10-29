# Command reference

Run commands from the repository root after activating `partidesign` (CUDA 12)
or `partidesign_cu13` (CUDA 13). A [task YAML](task_schema.md) describes the
molecules and conditions; command options select inputs, model assets, and
execution settings.

```bash
partidesign --help
partidesign <command> --help
```

| Command | Use it to |
| --- | --- |
| [`partidesign download`](download.md) | Prepare generation and optional model assets |
| [`partidesign check`](check.md) | Check tasks, model assets, and the installation |
| [`partidesign design`](design.md) | Generate molecular structures and identities |
| [`partidesign pipeline`](pipeline.md) | Run design, refinement, and validation in one command |
| [`partidesign score`](score.md) | Score interfaces in saved structures with PartiScore |
| [`partidesign relax`](relax.md) | Refine existing structures with OpenMM |
| [`partidesign validate`](validate.md) | Evaluate saved designs with selected sequences, predictors, and criteria |
| [`partidesign postfilter`](postfilter.md) | Relax and screen passing protein and peptide complexes |
| [`partidesign cluster`](cluster.md) | Group passing designs by backbone similarity |

For a first run, follow the [main README](../README.md#inference). After generation,
use the [validation guide](validate.md) to choose the method, sequence budget, and
passing criteria before running an evaluation. The [performance guide](performance.md)
explains the paper's computational budgets and evaluation protocols.

## Pipeline

`partidesign pipeline TASKS.yaml` runs design, refinement, and validation in one
command. It generates once, then refines sequences and structures through MPNN,
refolding, selection, and partial diffusion. OpenMM is optional. Final validation
uses AF3 for protein/nuc presets or paired AF2-IG predictions for explicit `ppi`.
See the [complete pipeline guide](pipeline.md) for
presets, round/task overrides, the insulin receptor demo, outputs, and resume.
For single-reference sampling with `partidesign design`, see
[partial diffusion](partial.md).
