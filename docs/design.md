# `partidesign design`

[Command index](commands.md)

Generate molecular identity and structure from a task YAML. The YAML specifies
the molecules, sample count, lengths, context, and constraints; command options
select tasks and execution settings. See the [task reference](task_schema.md)
and [example index](../configs/examples/README.md).

## A first run

After [installation](../README.md#installation) and generation-weight setup, run
from the repository root. The included `protein_fixture` requests five samples
at length 40, with scoring and refinement disabled:

```bash
partidesign design configs/examples/protein.yaml \
  --task protein_fixture --run-name first_run \
  --batch-size 1 --compile false --show-config

partidesign check configs/examples/protein.yaml --task protein_fixture

CUDA_VISIBLE_DEVICES=0 partidesign design configs/examples/protein.yaml \
  --task protein_fixture --run-name first_run \
  --batch-size 1 --compile false
```

`--show-config` prints the resolved settings without loading weights or
generating structures. Remove it to generate after checking the inputs and
assets. The generation bundle must include the model, autoencoder, and
experiment configuration; see [model setup](download.md#generation-weights-and-bundled-assets).

## Routing and task selection

Use `--task NAME` to select one task; repeat it for several. Omitting `--task`
selects every task in the file, which may request many samples. Batch size does
not change the sample count or length plan.

Automatic routing chooses one model member for the complete selected
submission. Normal and antibody tasks must be run in separate commands. For an
antibody request, for example:

```bash
partidesign design configs/examples/antibody.yaml \
  --task antibody_fixture --run-name antibody_run \
  --batch-size 1 --compile false --show-config
```

Routing is enabled by default. `--route-ckpt` can select `omni`, `prot`, `pep`,
`nuc`, `ligand`, or `antibody`; `null` keeps automatic selection. The `pep`
member requires a version-2 six-member bundle. `--route false` loads a
compatible checkpoint directly. Use explicit routing only with a matching
checkpoint, and run [check](check.md) with the same settings.

## Key options

These defaults come from the supplied [runtime configuration](../configs/design.yaml).

| Option | Meaning and default |
| --- | --- |
| `--task NAME` | Select a named task; repeatable. Default: all tasks in the file. |
| `--run-name NAME` | Run label; default `design`. |
| `--out PATH` | Run output root; default `inference/<run-name>`. |
| `--model-dir PATH` | Select the model root; see [model directories](download.md#use-a-separate-model-directory). |
| `--ckpt FILE` | Explicit compatible checkpoint; default `ckpts/partidesign.ckpt` under the selected model root. |
| `--route true\|false` | Enable bundle routing; default true. |
| `--route-ckpt MEMBER` | Force a bundle member; default `null` for automatic selection. |
| `--batch-size N` | Positive generation batch size; default 16. |
| `--num-workers N` | Nonnegative data-worker count; default 4. |
| `--seed N` | Nonnegative global seed; default 2026. |
| `--compile true\|false` | Compile the model trunk; default true. |
| `--config FILE` | Main runtime configuration overlay. |
| `--override KEY=VALUE` | Runtime override; repeatable. |
| `--show-config` | Print resolved settings without generation. |

Set sample counts, lengths, and molecular conditions in the task YAML.
`--config` is a runtime overlay, not a second task file. Dedicated flags override
the corresponding configured values; do not set a conflicting `--override` for
the same field. Preview changes with `--show-config`.

## Optional candidate ranking

Search and reward are off by default. Best-of-N can generate several candidates
per request and rank them with a reward. This example uses clash scoring, which
needs no additional learned reward weights:

```bash
partidesign design configs/examples/protein.yaml \
  --task protein_fixture --run-name ranked_run \
  --batch-size 1 --compile false \
  --override search=best_of_n \
  --override reward=clash \
  --override search.best_of_n.replicas=4
```

Add `--show-config` to preview. Four replicas for each of the five base requests
plan 20 candidates. Successful replicas are retained with ranking information in
their filenames and `result.csv`. A reward determines candidate ordering;
task `score: true` separately requests final PartiScore, and
`postprocess: true` requests OpenMM refinement. See
[task fields](task_schema.md#top-level-task-fields) for their defaults.

## Outputs and reruns

Each selected task receives its own directory:

```text
inference/first_run/protein_fixture/
  resolved_config.yaml
  result.csv
  logs/design.log
  sample1.cif
  ...
```

Each task directory must be new or empty. Generation does not resume a nonempty
task directory: choose a fresh run name or `--out` for another run. `--out`
names the run root; the task name is appended beneath it.

Use `result.csv` to identify the final structure files and per-sample status.
Normal-task rows include `error` and `postprocess_status`; antibody tasks have
their own result columns. Failed samples receive no placeholder structures.
When refinement or scoring is enabled, final filenames may also contain ranking
information.

## Troubleshooting

For missing checkpoints, input paths, or CUDA problems, start with
[installation and first-run checks](check.md#installation-and-first-run-troubleshooting).
If a sample fails, inspect its row and `logs/design.log`. To reduce a trial,
copy the task and reduce `sample_num` or its lengths; lowering batch size alone
does not reduce the requested work. Existing usable CIFs can be rescored with
[score](score.md) or refined with [relax](relax.md).

[Command index](commands.md) · [Check inputs](check.md) · [Evaluate designs](validate.md)
