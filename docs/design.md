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
  --batch-size 1 --show-config

partidesign check configs/examples/protein.yaml --task protein_fixture

CUDA_VISIBLE_DEVICES=0 partidesign design configs/examples/protein.yaml \
  --task protein_fixture --run-name first_run \
  --batch-size 1
```

`--show-config` prints configuration without loading weights, probing the GPU,
or generating structures. Automatic execution remains pending until runtime.
Remove it to generate after checking the inputs and assets. The generation
bundle must include the model, autoencoder, and
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
  --batch-size 1 --show-config
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
| `--execution-backend auto\|native\|bioir` | Execution backend; default `auto`. |
| `--compile auto\|true\|false` | Native FM-denoiser compilation; default `auto`. |
| `--compile-strategy auto\|dynamic\|bucket` | Automatic selection, dynamic shapes, or static token buckets. |
| `--config FILE` | Main runtime configuration overlay. |
| `--override KEY=VALUE` | Runtime override; repeatable. |
| `--show-config` | Print resolved settings without generation. |

Set sample counts, lengths, and molecular conditions in the task YAML.
`--config` is a runtime overlay, not a second task file. Dedicated flags override
the corresponding configured values; do not set a conflicting `--override` for
the same field. Preview changes with `--show-config`.

## Execution and compilation

The public defaults are `model.execution_backend: auto` and
`model.compile.enabled: auto`. Runtime chooses once for the complete selected
submission, using its expanded candidate count: sum `sample_num` over the
selected tasks and length plans, then include any Best-of-N replica multiplier.
For example, five base requests with four replicas count as 20 candidates.
Batch size does not change this count.

| Requested settings | Runtime selection |
| --- | --- |
| Backend `auto`, compilation `auto`, at least 100 candidates | Native dynamic compilation |
| Backend `auto`, compilation `auto`, fewer than 100 candidates | Usable BioIR (CUDA Graphs for eligible inputs, eager otherwise), or native eager when unavailable |
| Backend `native`, compilation `auto` | Native dynamic compilation at 100 candidates or more; native eager below 100 |
| Backend `bioir`, compilation `auto` | BioIR regardless of count; compilation resolves to false |
| Backend `auto`, explicit `--compile true` or `false` | Native compilation with the configured strategy (dynamic by default), or native eager, respectively |

An explicit native strategy takes precedence over the automatic count policy.
`--compile-strategy dynamic` or `bucket` selects that native strategy
and enables compilation, including with `--compile auto`. Combining it with
`--compile false` keeps compilation and bucket padding disabled.
`--compile-strategy auto` resets `enabled: auto`, `dynamic: true`, and
`token_buckets: null`; it cannot be combined with `--compile true` or `false`.
Explicit BioIR cannot be combined with `--compile true` or the `dynamic`/`bucket`
strategy selectors. It accepts `--compile false` or `auto` and the `auto`
strategy selector.

For a small automatic submission, a controlled BioIR dependency or startup
incompatibility selects native eager execution. This startup selection does not
suppress out-of-memory or input errors or retry a failed generation job. Explicit
`--execution-backend bioir` reports unavailable or incompatible dependencies
as errors. Within BioIR, graph-ineligible inputs use eager kernels. Its graph
manager can also discard a graph and retry that signature eagerly after a
recoverable capture/replay error, including an allocation failure; device-fatal
CUDA errors propagate. This graph fallback does not rerun the entire job.

CPU task checks and `--show-config` validate the request without probing GPU
capability; an automatic choice remains pending. At execution, the log and
saved `resolved_config.yaml` record the requested settings, resolved backend
and compilation, selection reason, and expanded candidate count. The saved
`model.execution_policy` block contains `requested_backend`, `requested_compile`,
`resolved_backend`, `resolved_compile`, `reason`, and `fm_samples` (that count).
Saved public backend and compilation fields are concrete; configuration
previews retain `auto` until runtime resolution. The execution-policy block
is output metadata; when preparing a configuration overlay, copy only the
public settings you intend to override, rather than the entire saved snapshot.

Native compilation applies only to the whole FM denoiser (`model.nn`); the VAE
stays in eager mode. Both native strategies use Inductor, `mode: default`,
`fullgraph: true`, and compiler `shape_padding: false`. Dynamic compilation uses
`model.compile.dynamic: true` with `token_buckets: null`.

`--compile-strategy bucket` resolves to `dynamic: false` and pads each batch's
largest actual token count to the smallest fitting bucket. The default buckets
are `[128,160,192,224,256,288,320,352,384,448,512,640,768,896,1024]`.
All tokens and pairs use the existing masked padding, including cyclic metadata;
sample order and sample counts are preserved. Enabled bucket compilation
reserves Dynamo specialization capacity for the configured buckets and batch
size regimes; dynamic and disabled modes keep the existing cache limits.
A batch larger than the largest bucket fails clearly; it is never truncated.
Supply a larger custom list when needed:

```bash
partidesign design configs/examples/protein.yaml --compile-strategy bucket \
  --override 'model.compile.token_buckets=[128,256,512,1024,1536]' --show-config
```

The optional `model.compile.token_buckets` config field must be null for
`dynamic: true`, or a nonempty, strictly increasing list of positive integers
for `dynamic: false`. Booleans are rejected. A config overlay can select either
policy directly; the bucket CLI selector retains a configured non-null list,
and the dynamic or auto selector resets it to null. `--compile false` disables
both compilation and bucket padding while keeping the selected policy in the
resolved config.

## Experimental BioIR execution

BioIR is the automatic choice for eligible small submissions and can also be
selected explicitly. It is an optional execution path for the existing FM
network, using NVIDIA's
pairwise attention, AdaLN and SwiGLU primitives and optional CUDA Graph replay;
the PartiDesign checkpoint, sampler, self-conditioning, conditions and VAE remain
the same.

Use the full CUDA 13 environment from the repository root:

```bash
conda env create -f partidesign_cu13.yaml
conda activate partidesign_cu13
partidesign check configs/examples/protein.yaml \
  --task protein_fixture --execution-backend bioir
partidesign design configs/examples/protein.yaml \
  --task protein_fixture --execution-backend bioir --out ./inference/bioir
```

The full `partidesign_cu13` environment pins PyTorch 2.11.0 cu130, Triton 3.6.0
and JAX 0.10.2 with CUDA 13 plugins, alongside all community folding/redesign,
MSA, OpenMM, AmberTools and docking dependencies. The regular cu128 environment
keeps its existing versions. Both environments install the `partidesign` command.

For BioIR, the CUDA 13 environment installs the separately named
`partidesign-bioir-kernels` runtime, derived from NVIDIA BioIR 0.2.0 with its
kernel files and licenses preserved. Its dependency contract covers the kernels
called by PartiDesign; NVIDIA's other model pipelines are outside this contract.
Install exactly one provider: this runtime and the complete official
`bionemo-ir` package share the same Python namespace.
See [environment setup](environments.md) and the [runtime package notes](../runtime/bioir-kernels/README.md).

`--execution-backend bioir` disables native `torch.compile` and token bucket
padding, subject to the [flag combinations above](#execution-and-compilation).
A configuration overlay can set `model.execution_backend: bioir` with
`model.compile.enabled: auto` or `false` and `model.compile.token_buckets: null`.
Explicit BioIR checks report missing dependencies; a real design run is needed
to validate GPU execution. `check --execution-backend` requires a task YAML and
accepts `auto`, `native`, or `bioir` together with the intended compile settings.

The settings under `model.bioir` are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `cuda_graph` | `true` | Use BioIR's capture and replay around one denoiser evaluation. |
| `max_graphs` | `4` | Maximum resident graph signatures; older signatures can be evicted. |
| `max_tokens` | `512` | Largest token count eligible for graph replay. Longer inputs run the BioIR kernels eagerly. |

The graph tracker also uses a 4 GiB estimated residency budget. It can evict
signatures before reaching `max_graphs`; this is not a hard peak-memory cap,
and one graph can exceed it. A separate free-memory check can keep a signature
on eager execution instead of capturing it. Check the reported fallback counts
when comparing performance on large inputs.

For example, `--override model.bioir.cuda_graph=false` measures the kernel path
without graph replay. Each distinct input signature, including batch size and
token count, may need its own capture. CUDA Graphs are process-local; starting a
new process captures again. Packaged CuTe kernels can load precompiled code,
while some Triton kernels still need first-use JIT or a disk-cache load. These
are separate from CUDA Graph capture: skipping whole-model compilation does
not mean zero first-use preparation. The design log records actual attention
backend choices and graph capture, replay, fallback and eviction counts.

SM89 GPUs have a BioIR CuTeDSL pairwise-attention path. On SM120 GPUs, BioIR
selects SDPA for this operation; 50-series hardware has not been validated here.
On the tested SM89 device, BioIR reduced first-batch startup in most BS8/BS16
cases, while native compilation had better sustained throughput and lower
reserved memory. These observations used the earlier minimal BioIR environment;
the complete CUDA 13 stack is validated for functionality rather than a new
matched performance benchmark. Automatic selection uses native dynamic
compilation for submissions of at least 100 candidates. Performance and kernel
coverage depend on the GPU and workload.

See NVIDIA's [support matrix](https://docs.nvidia.com/bionemo/inference-runtime/latest/references/support-matrix/)
and [custom model integration guide](https://docs.nvidia.com/bionemo/inference-runtime/latest/references/accelerate-custom-model).

## Optional candidate ranking

Search and reward are off by default. Best-of-N can generate several candidates
per request and rank them with a reward. This example uses clash scoring, which
needs no additional learned reward weights:

```bash
partidesign design configs/examples/protein.yaml \
  --task protein_fixture --run-name ranked_run \
  --batch-size 1 \
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
