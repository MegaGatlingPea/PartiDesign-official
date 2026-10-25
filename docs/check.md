# `partidesign check`

[Command index](commands.md)

Check the environment, task inputs, model routing, and required assets before
generation, or check individual model resources. The default check does not run
model inference.

## Check a task

Run from the repository root with the `partidesign` environment active:

```bash
partidesign check configs/examples/protein.yaml --task protein_fixture
```

The checker inspects the selected configuration, source structures, routing,
generation checkpoint, and any scoring/refinement resources requested by the
task. Omit `--task` to check every task in the YAML, or repeat it to select
several. Normal and antibody tasks must be selected in separate submissions.
Use the same `--ckpt`, routing settings, and configuration overlays intended for
[generation](design.md).

Task checks accept the same execution request as generation:

```bash
partidesign check configs/examples/protein.yaml --task protein_fixture \
  --execution-backend auto --compile auto
```

`--execution-backend auto|native|bioir`, `--compile auto|true|false`, and
`--compile-strategy auto|dynamic|bucket` follow the
[generation flag rules](design.md#execution-and-compilation). Ordinary checks
and CPU loading do not probe GPU capabilities; automatic backend resolution
remains pending. An explicit BioIR check inspects its dependencies and CPU task
preparation, while GPU execution requires a real job.

## Check model resources

Without a task file, select repeatable `--model NAME` or `--all`. With neither,
the default resource is `partidesign`:

```bash
partidesign check --model chemistry --model partiscore
partidesign check --model esmfold2-fast --load
partidesign check --model af3 --msa off
```

The AF3 example checks its installation without requiring MSA databases. To
check a different model root, add `--model-dir PATH`; see
[model directories](download.md#use-a-separate-model-directory).

Task-file checking cannot be combined with `--model` or `--all`. Task-specific
options such as `--task` and `--ckpt` require a task file. AF3 MSA options require
an explicit `--model af3` or `--model af3-la`.

## CPU loading and real execution

| Mode | What runs |
| --- | --- |
| No execution flag | Configuration, environment, source, and asset checks; no model inference. |
| `--load` | The checks above plus strict model loading on CPU. This can require substantial host RAM and does not test GPU inference. |
| `--smoke` | CPU loading followed by bounded, real model jobs with output readback. GPU backends require CUDA; PartiScore can run its check on CPU. |

For a small real inference check using the bundled PartiScore weights:

```bash
partidesign check --model partiscore --smoke
```

This check does not need the generation checkpoint. Smoke outputs use a
temporary directory and are removed when the check finishes. Retain them with:

```bash
partidesign check --model partiscore --smoke --out checks/partiscore
```

`--out` is accepted only with `--smoke`. Use an output location suitable for the
requested check. Native-generation smoke checks require the generation bundle.

## Key options and output

| Option | Meaning |
| --- | --- |
| `TASK_FILE.yaml` | Check a generation task file; optional when checking model resources. |
| `--task NAME` | Select a named task; repeatable. |
| `--model NAME` | Select a registered model/resource; repeatable. See [download targets](download.md#optional-evaluation-models). |
| `--all` | Check every registered resource, including optional models. |
| `--model-dir PATH` | Select the model root. |
| `--ckpt FILE` | Select a compatible generation checkpoint for a task check. |
| `--config FILE`, `--override KEY=VALUE` | Use the intended runtime settings for a task check. |
| `--route`, `--route-ckpt` | Use the intended model-routing settings; see [routing](design.md#routing-and-task-selection). |
| `--execution-backend auto\|native\|bioir` | Request an execution backend for a task check; default `auto`. |
| `--compile auto\|true\|false`, `--compile-strategy auto\|dynamic\|bucket` | Use the intended task compilation settings; see [execution rules](design.md#execution-and-compilation). |
| `--load`, `--smoke` | Opt into CPU loading or real execution as described above. |
| `--msa off` | Check explicit AF3/AF3-LA resources without MSA databases. |

The terminal report shows `PASS`, `FAIL`, `SKIP`, and `NOT_RUN` entries. Unused
optional models are skipped. A failed check returns a nonzero status; no
generation result is produced unless a smoke job was requested. Repeat the
check after correcting the reported issue. Passing a CPU check does not
establish GPU availability or validate every possible stochastic length/crop.

## Installation and first-run troubleshooting

The [main installation guide](../README.md#installation) covers the Linux/NVIDIA
environment, bundled resources, generation bundle, and CPU checks.

| Symptom | Check |
| --- | --- |
| `partidesign: command not found` | Run `conda activate partidesign` and confirm environment creation completed successfully. |
| A package or backend path fails during installation | Run `conda env create -f partidesign_cu12.yaml` from the repository root; the environment definition installs relative source paths. Keep the supplied versions and bundled dependencies together. |
| The native checkpoint is missing | Download the [release bundle](download.md#generation-weights-and-bundled-assets) to `ckpts/partidesign.ckpt` under the selected model root, or use `--ckpt` with a task check. The Drive release is downloaded separately from the CLI asset downloader. |
| CUDA is unavailable | Check `nvidia-smi`, the activated Python/PyTorch environment, and `CUDA_VISIBLE_DEVICES`. CPU checks do not establish GPU availability. |
| An example input cannot be found | Run from the repository root. Input paths resolve relative to the invocation directory; `--model-dir` does not change input/output resolution. |
| An existing generation output is rejected | Choose a fresh `design --run-name` or `--out` directory. Generation does not resume a nonempty task directory. |
| An optional evaluator reports missing assets | [Download](download.md) and check its exact target. With an external model root, copy any required Git-supplied weights into it. |
| AF3 checking reports missing MSA resources | Use `partidesign check --model af3 --msa off`, or prepare resources for the intended MSA policy. |
| A generated sample has no CIF | Read `error` and `postprocess_status` in its `result.csv`, then inspect `logs/design.log`. Failed samples have no placeholder structure. |

Check the active PyTorch build and driver with:

```bash
python -c "import sys, torch; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
nvidia-smi
```

Before retrying generation, inspect the task with `design --show-config`. For a
smaller trial, copy the YAML and reduce `sample_num` and its length plan.
With the default automatic backend, `--batch-size 1 --compile false` is an
intentional native-eager diagnostic that preserves the requested sample count.
Retry refinement of existing usable structures with [relax](relax.md), without
regenerating them.

[Command index](commands.md) · [Download assets](download.md) · [Generate designs](design.md)
