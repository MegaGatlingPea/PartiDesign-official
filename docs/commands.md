# Command reference

Run commands from the repository root after activating the `partidesign`
environment. A [task YAML](task_schema.md) describes the molecules and conditions;
command options select inputs, model assets, and execution settings.

```bash
partidesign --help
partidesign <command> --help
```

| Command | Guide |
| --- | --- |
| [`partidesign download`](download.md) | Model inventory, downloads, optional backends, and model storage |
| [`partidesign check`](check.md) | Task and resource checks, CPU loading, smoke checks, and troubleshooting |
| [`partidesign design`](design.md) | Generation, task selection, model routing, ranking, and output files |
| [`partidesign score`](score.md) | PartiScore interface scoring of saved structures |
| [`partidesign relax`](relax.md) | OpenMM refinement of existing designs |
| **[`partidesign validate`](validate.md)** | **Sequence redesign, structure prediction, criteria, reports, reuse, and reruns** |
| [`partidesign cluster`](cluster.md) | Structural clustering of passing designs |

For a first run, follow the [main README](../README.md#inference). After generation,
use the [validation guide](validate.md) to choose the method, sequence budget, and
passing criteria before running an evaluation. The [performance guide](performance.md)
explains the paper's computational budgets and evaluation protocols.
