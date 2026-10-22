# `partidesign cluster`

[Command index](commands.md)

Cluster the saved design-polymer backbones selected by validation success
tables. Clustering runs on CPU with US-align. It compares the generated
structures referenced by the task's current `result.csv`, rather than the
validation refold coordinates. Ligand-only tasks are unsupported.

## Inputs and a minimal example

Run [validation](validate.md) with suitable criteria first. The input can be one
or more run roots, task directories, validation method directories, or their
`success/` directories. The command discovers existing `success.csv` tables at
these documented directory levels.

```bash
partidesign cluster inference/first_run --task protein_fixture --plan

partidesign cluster inference/first_run --task protein_fixture \
  --tm-threshold 0.5 --workers 4
```

`--plan` lists selected successful backbones and pair-comparison counts without
alignment or report writes. The second command performs clustering. It requires
US-align in the active Python environment or on `PATH`; select a different
executable with `--usalign FILE` if needed.

## Key options

| Option | Meaning and default |
| --- | --- |
| `RUN_TASK_OR_SUCCESS ...` | One or more existing input directories as described above. |
| `--task NAME` | Select task names; repeatable. Default: all discovered tasks. |
| `--method NAME` | Select existing validation methods; repeatable. |
| `--criteria NAME` | Select passing criteria; repeatable. Each criterion stays separate. |
| `--tm-threshold VALUE` | Mean of the two length-normalized TM-scores; default 0.5, valid range `(0, 1]`. |
| `--workers N` | Concurrent success directories; default 16, allowed range 1–16. Each worker uses one CPU thread. |
| `--usalign FILE` | Explicit US-align executable; default is the active Python environment, then `PATH`. |
| `--plan` | Inspect selected inputs and comparison counts without clustering. |

Actual concurrency is bounded by available CPUs and the number of selected
directories. Task, method, criterion, and design-polymer composition stay
separate. Duplicate passing sequence slots from one generated backbone are
collapsed before clustering.

## Outputs and reruns

Each selected `validate/<method>/` directory receives:

| File | Contents |
| --- | --- |
| `clusters.csv` | Per-backbone cluster assignment, representative, cluster size, and TM-score to the representative. |
| `cluster_summary.csv` | Per-criterion counts of successful backbones, clusters, singletons, and largest cluster. |

A nonempty set of passing backbones is needed to form clusters. Existing empty
success tables produce reports with zero counts. Rerunning recalculates and
replaces these two reports; it does not reuse previous cluster assignments.

## Troubleshooting

| Problem | Next step |
| --- | --- |
| No `success.csv` tables match | Check the run/task path and selectors, then complete validation with suitable criteria. |
| Reports contain no clusters | Inspect the validation passing results. Empty success tables correctly yield zero clusters. |
| US-align is not found | Use the supplied environment's `USalign`, check `PATH`, or pass an executable with `--usalign FILE`. |
| A saved structure cannot be found | Keep the task's `result.csv` and its referenced generated structures together. |
| A task contains only ligands | Ligand-only clustering is unsupported; use the command for generated protein, RNA, or DNA backbones. |

[Command index](commands.md) · [Validate designs](validate.md) · [Generate designs](design.md)
