# Choose execution settings

Use this reference when choosing resources for an authorized run or answering a
performance question. Resolve the active CLI/config first; the values below are
starting defaults, not measured optima for every input or machine.

## Decide before running

1. Choose the operation from the inputs: generate new designs with design or
   pipeline, evaluate saved designs with validate, and screen supplied sequences
   or SMILES with screen. Reuse complete outputs, MSA/templates and supported
   resume/refilter/recompute paths before scheduling repeated model work.
2. Distinguish a few requested results from a sustained batch campaign. Infer
   first-result latency versus whole-job throughput from the request when clear;
   ask only when an unresolved objective or compute budget changes the choice.
3. Inspect ordinary `check` and the assigned devices: current free VRAM and
   utilization, usable host RAM, CPU affinity/cgroup allowance, and other jobs.
   Use available/assigned capacity rather than host totals. Model loading is not
   needed to inspect these resources.
4. Inspect `--show-config` or `--plan`: candidate counts, length plans, context
   size, modalities, sequence proposals, prediction samples and rounds. A small
   ligand with a long receptor can still be a large prediction request.
5. Preserve explicit user settings. For unset execution controls, choose a
   supported configuration and briefly state its reason and expected workload.
   Do not add fixed GPU/batch flags merely to make a command look complete.

## Know which controls apply

| Operation | Controls | Scope |
| --- | --- | --- |
| Design and direct partial | `--batch-size`, `--num-workers`, assigned `CUDA_VISIBLE_DEVICES` | Native batch and loader workers are per distributed rank. Native execution can use the assigned visible GPUs. |
| Pipeline generation/partial | `generation.batch_size`, `generation.num_workers`, matching CLI flags | Partial has its own pool; prediction worker flags do not resize it. |
| Pipeline refolding | `iteration.refold.validation.worker_gpus/worker_cpus`, `final.validation.worker_gpus/worker_cpus`, round/task overrides | Global CLI worker flags override stage settings; all `final.methods` share the final resource block. |
| Validate | Top-level `worker_gpus/worker_cpus` or CLI flags | A supplied override applies across selected methods. |
| Screen | `--worker-gpus`, `--worker-cpus` | Global across selected tasks/methods; no per-method resource YAML or generation-style batch-size flag. |

Current native generation defaults are batch **16**, loader workers **4** and
`pin_memory: false`. Prediction workers default to **1 CPU** each, with GPU
reservations **0.25 for AF2**, **0.5 for AF3/AF3-LA**, and **1 for ESMFold2,
Mol-1 and Boltz-2**. Omit a global override to retain model-specific defaults.
An unset/null `worker_gpus` inherits the method default. Pipeline and screen
limit active workers by CPU/GPU capacity. Standalone GPU validation chooses its
worker count from GPU reservations and pending jobs, then requests CPUs for
those workers; a larger CPU reservation does not reduce its worker count.
Check that total CPU demand fits the process's available allowance.
Pipeline MPNN workers reserve 1 CPU/0.0625 GPU; partial workers reserve one GPU
and `generation.num_workers + 1` CPUs. Neither has a public prediction-worker
reservation override. Method groups execute sequentially; workers within an
active group can execute concurrently.

## Select resources for the workload

- **Generation:** start from resolved defaults and matching prior measurements.
  For a sustained homogeneous workload, avoid inheriting batch one from a small
  example. Choose batch size from input/token shape and free VRAM, with RAM for
  prepared inputs and loader prefetch. Four loader workers per rank can multiply
  across GPUs; more workers can increase CPU/RAM pressure without raising GPU
  throughput. Batch size does not change total requested samples.
  For full PartiDesign generation, use the [token-to-batch guide](partidesign-memory.md):
  historical batch-16 data anchors estimates for candidates 4, 8, 12, 16, 24, 32,
  48 and 64. Its other batches are unmeasured and its curve does not calibrate
  partial diffusion or cofolding backends.
- **Prediction:** a GPU fraction reserves scheduling capacity, not a fraction of
  VRAM. A 0.5 reservation can allow two resident workers on one GPU; each still
  needs its own model and input memory. Consider host RAM and CPU preparation as
  well as VRAM. Long contexts or larger complexes can favor fewer workers even
  when many short inputs would benefit from concurrency. Keep method defaults
  unless task/hardware evidence supports another choice.
- **Small jobs versus large queues:** avoid unnecessary worker startup and
  compilation for a few results. For a large screen or validation queue, consider
  concurrent resident workers and shared context reuse. Screen `samples` changes
  the number of structures per candidate; it is not a candidate batch size.
- **Measurement:** compare total elapsed time and completed requested work;
  distinguish cold startup/compilation from sustained model timing. Published
  batch-16 generation timings and passing-output/GPU-day tables are not a
  concurrency sweep. Do not claim an unmeasured fastest batch or GPU fraction.
  Keep any tuning trial within the authorized compute/output budget; do not run
  profiling to answer a usage or configuration question.

Retain automatic backend/compile selection unless the request or matching
evidence calls for an override. An OOM diagnostic using batch one/eager execution
is not a general throughput profile. Use only controls supported by the active
release: do not invent a total-worker cap, per-method screen reservation, VRAM
limit or automatic mid-run batch tuning. If supported controls cannot enforce
the available CPU/memory budget, state that limitation before an unsuitable run.

## Choose iteration budgets explicitly

When authoring a workflow with unspecified budgets, start from the matching
preset and explain its workload against the user's goal/time budget. Ordinary
protein/nuc/antibody pipelines use fast iteration and standard final validation;
explicit PPI uses AF2-IG final validation. Ligand defaults to one Boltz-2 round
with eligible final affinity; another round adds a partial-diffusion transition.

Count initial candidates and sequence proposals, then each transition's selected
parents, retained parents/partial replicas, subsequent refolds and final samples.
Include paired AF2 predictions or native affinity work where applicable, and
account for supported reuse of matching completed structure predictions.
Filters, failures and deduplication can reduce realized counts; show the planned
work without assuming a favorable pass rate. `max_pass` limits survivors after
refolding and cannot save the refolds already submitted in that round.

Recycles, partial fraction/replicas, MPNN draws, prediction samples, methods,
MSA evidence and selection criteria change the search/evaluation protocol.
Preserve explicit choices; make proposed budget changes visible rather than
treating them as scheduling tweaks. Native FM steps remain 400; partial fraction
changes the retained schedule and optimization behavior. For a reproduction,
also match batching, devices, seed and execution policy rather than promising
bitwise-identical results after resource changes.

Before launch, report the chosen operation, estimated work, device allocation,
generation batch/loaders and applicable prediction worker reservations, with a
short reason. For a plan, report configuration only; for execution, report actual
completion/timing and any observed bottleneck. Read the active checkout's
`docs/design.md`, `docs/pipeline.md`, `docs/validate.md` and `docs/screen.md` for
mode-specific syntax and limits.
