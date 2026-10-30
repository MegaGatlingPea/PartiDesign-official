# Pipeline

Use `partidesign pipeline` to go from initial designs to validated candidates
in one command:

```bash
partidesign pipeline configs/examples/pipeline_tasks.yaml --task protdbench_insulin_receptor --preset ppi --run-name insulin_receptor
```

This insulin receptor binder task requests 32 initial backbones. The command
uses the general PPI workflow below. Add `--config configs/pipeline/demo.yaml`
for strict screening with the same production candidate limits.

Run from the repository root with the generation, MPNN, ESMFold2-fast, and AF2
assets prepared; see [model setup](download.md#optional-evaluation-models).
Add `--plan` for a short task/workflow preview or `--show-config` for the full
configuration used for diagnostics. Neither loads models nor creates run output.

The pipeline supports protein, peptide, RNA, and DNA design within each
backend's capabilities, including supported ligand context. Pure ligand design
uses [generation](design.md).

## Default flow

The default is three rounds, including the first validation round (`recycle: 3`):

```text
generate once using each task's sample_num
→ round 1: current sequence + MPNN1 → refold → filter → cluster
→ selected parent + four partial replicas
→ round 2: current sequence + MPNN4 → refold → filter → select
→ selected parent + four partial replicas
→ round 3: current sequence + MPNN4 → refold → filter → select
→ historical candidate pool → final validation
```

Here GT means the candidate's current sequence. Default iteration refolding is
ESMFold2-fast with one conformer per sequence. The first selection deduplicates
sequences and uses average-linkage clustering with an exact K=200 cut, keeping
the best candidate in each cluster; if fewer unique sequences pass, all remain.
Later rounds select the best child per immediate parent and deduplicate across
the task. Middle rounds forwarding candidates retain at most **two per partial
family**, then apply the task-wide cap of 200. A family is the previous selected
refold and its partial children. The last round has no family cap. Iteration
ranking uses **design-chain pLDDT**; context confidence is excluded.

Transitions use `fraction: 0.625`, four replicas and parent retention by default;
OpenMM is off. The last round has no transition, including when `recycle: 1`.
Original task conditions, fixed identities, motifs, context, and bonds remain
active as the current complex supplies the next partial-diffusion reference.

## Control candidate counts

The 32 initial InsR backbones are the start of the run. With the default
four partial replicas and `include_parent: true`, each selected refold supplies
up to **five input structures** to the next round. Each of those structures
then proposes GT1 plus MPNN4: up to **25 sequence proposals per selected refold**.
Partial failures, sequence deduplication and prediction-cache reuse can reduce
this work.

Later `per_parent: 1` selection applies to each immediate input structure, so
the retained refold and its four partial children are five separate parents.
In middle rounds, `per_family: 2` then keeps the two highest-scoring candidates
from that family after task-wide sequence deduplication. Set `per_family: null`
to disable this extra cap. First-round clustering and last/final selection do
not apply the family cap.

`max_pass: 200` caps the selected survivors **after refolding and filtering**;
it does not cap the number of sequence proposals submitted by that round.
With these defaults, 200 selected refolds can supply 1,000 structures and up to
5,000 sequence proposals to the next round. No partial replicas are generated
after the final round. Proposal totals describe the whole round; GPU workers
process them in batches or scheduled requests.

Use `summary.csv` to inspect proposal, prediction, selection, and partial
counts for your run.

## Final validation

By default, final validation uses the best eligible candidate seen across all
rounds for each lineage and refold profile. A **lineage** starts with a selected
round-1 refold and includes all its descendants. It stays the same across rounds;
a **partial family** groups a selected refold with its children for the next
round only. The family cap therefore does not set the final pool size.

The pool removes duplicate design sequences and keeps at most 200 candidates.
Each refold profile ranks its own candidates; the pool takes one from each
profile in turn until it reaches the cap or runs out. Set `final.max_pass` to
change this cap, or `final.pool.source: last` to use last-round survivors.

| Profile | Final prediction and acceptance |
| --- | --- |
| `protein`, `nuc` | AF3, five samples per sequence. Select the highest native `ranking_score`, then apply final criteria to that prediction. |
| `ppi` | AF2-IG models `[1, 2]`, five matched complex/monomer pairs per model. Select each model's best complex by `0.8 * ipTM + 0.2 * pTM`, keep its same-sample monomer, and require both selected model pairs to pass. |

`preset: auto` chooses protein or nuc from each task; select `ppi` explicitly
for an eligible standard protein/peptide binder with protein context. PPI final
criteria are `af2_ig_protdbench` for proteins and `af2_ig_peptide` for peptides.
PPI produces ten pairs (twenty structures) per candidate with its default two
models. Prediction selection precedes criteria; a lower-ranked passing sample
does not replace a selected failing sample. All final structures remain in
`predictions/`. Each accepted candidate has one representative complex in
`success/`; the metrics and selection records retain both PPI model winners
and their matching monomer evidence.

## Results

New runs group results by task:

```text
inference/<run_name>/
  config.yaml                 # workflow configuration
  tasks.yaml                  # native task snapshot
  summary.csv
  <task>/
    design/                   # initial generation: result.csv and structures
    round_1/
      result.csv
      selection.csv
      refold/                 # authoritative passing candidate CIFs
      partial/<parent>/sample*.cif
      relaxed/                # optional whole-complex OpenMM
    round_2/...
    round_3/...               # no transition after the final iteration
    final/
      result.csv
      selection.csv
      predictions/            # all final AF3 samples or AF2 pairs
      success/                # accepted final representatives
  .work/                      # inputs, snapshots and resumable state
```

`config.yaml` is the pure workflow snapshot; `tasks.yaml` contains the selected
native tasks. `.work/` directories contain internal runtime files, including
`.work/resolved_config.yaml` for strict resume. Start with `summary.csv` for phase
counts, then each phase's `result.csv` and `selection.csv` for candidate
outcomes and selection. For per-prediction metrics, final model winners, and
rejection reasons, open `final/predictions/proposal*.json`; its
`prediction_selection` and `decisions` explain final acceptance.
`result.csv` points to visible candidate CIFs. Iteration prediction artifacts
are under the round's `.work/predictions/`. To use a candidate as a
standalone partial start, supply its one `refold/*.cif` and define the new task's
conditions in YAML. To recover a previous run's resolved locators, pass its
round-level `result.csv` explicitly in `task.partial`; the pipeline handles this
metadata internally. Standalone validation can target a specific
`<task>/round_N` or `<task>/final` directory.

Check partial success/failure counts and final passing counts: a completed run
can have fewer rounds or no accepted final structure. With parent retention,
a selected parent can continue even when its partial children fail.

Completed runs remove raw prediction caches by default, preserving all
structures, final predictions, small metric records, resumable state, and input
snapshots. Add `--keep-cache` to retain raw prediction data and the request
cache for later PAE/PDE metric recomputation. This CLI flag controls retention
and does not belong in workflow YAML. Failed or interrupted runs retain caches.
A completed-stage `--resume` still reuses saved results; an invalidated stage
needs new predictions when its raw cache has been removed.

For eligible protein/peptide complexes, pass the run, task, `final/`, or
`final/success/` to [postfilter](postfilter.md). It screens only each published
final representative and checks its final acceptance evidence, including both
PPI model winners. The other final predictions and isolated monomers remain
validation evidence.

## Tasks, presets, and workflow files

A task file describes the molecules and conditions. Keep it in the normal
[task schema](task_schema.md), for example `tasks.yaml`:

```yaml
my_protein:
  sample_num: 4
  design_chains: [{P: protein}]
  design_length: {P: [80]}
```

Use the same file and task selector for generation and the pipeline:

```bash
partidesign design tasks.yaml --task my_protein --run-name my_design
partidesign pipeline tasks.yaml --task my_protein --preset protein --run-name my_pipeline
```

Keep each task's usual `input`, conditions, crop, hotspots and bonds in this
native task file. `--task NAME` selects a task; repeat it to select several.
Omitting `--task` selects all tasks, each with its own candidates, counts and
outputs. The selected task's `sample_num` sets its initial generation budget.
Relative paths resolve from the invocation directory.

`--preset` chooses the workflow; `--config FILE.yaml` supplies separate workflow
overrides. Workflow presets and configuration examples live under
`configs/pipeline/`. The workflow file sets `preset`, `recycle`, operations under
`iteration` and `rounds`, and `final` controls. It is a plain top-level mapping,
without a `pipeline:` wrapper. Optional `tasks` entries in that workflow file
contain per-task workflow overrides, indexed by existing task names; molecular
task definitions stay in the native task file.

## PPI demos

The native task collection
[pipeline_tasks.yaml](../configs/examples/pipeline_tasks.yaml) works with both
`partidesign design` and `partidesign pipeline`. It contains only molecular
tasks; choose the pipeline workflow with `--preset ppi`:

| Task | Request |
| --- | --- |
| `protdbench_insulin_receptor` | Length-80 insulin receptor protein binder. |
| `protdbench_pdl1` | Length-80 PD-L1 protein binder. |
| `protdbench_bhrf1` | Length-80 BHRF1 protein binder. |
| `pepdbench_mdm2_linear` | Length-13 MDM2 linear peptide. |
| `pepdbench_mdm2_cyclic` | Length-13 MDM2 head-to-tail cyclic peptide. |
| `pepdbench_mdm2_disulfide` | Length-13 MDM2 disulfide peptide. |

These tasks retain the native benchmark crops and hotspots and each request
32 initial backbones. `--preset ppi` uses the general defaults: three rounds,
automatically routed iteration criteria, K=200 first selection, later
max-pass=200 and final max-pass=200.

The optional [demo workflow](../configs/pipeline/demo.yaml) uses production
candidate limits and stricter screening: strict `af3_basic` iteration filtering,
K=200 first selection, later max-pass=200 and final max-pass=200. It retains three rounds,
partial fraction 0.625 with four replicas and parent retention, and relaxation
off. Iteration and final prediction workers each request one GPU. Both the
preset and demo workflow use PPI final AF2 validation. Strict criteria may end
a task early or leave no final pass.

```bash
partidesign pipeline configs/examples/pipeline_tasks.yaml --task protdbench_insulin_receptor \
  --preset ppi --config configs/pipeline/demo.yaml --run-name insulin_receptor_demo --plan
partidesign pipeline configs/examples/pipeline_tasks.yaml --task protdbench_pdl1 \
  --preset ppi --config configs/pipeline/demo.yaml --run-name pdl1 --plan
partidesign pipeline configs/examples/pipeline_tasks.yaml --task protdbench_bhrf1 \
  --preset ppi --config configs/pipeline/demo.yaml --run-name bhrf1 --plan
partidesign pipeline configs/examples/pipeline_tasks.yaml \
  --task pepdbench_mdm2_linear --task pepdbench_mdm2_cyclic --task pepdbench_mdm2_disulfide \
  --preset ppi --config configs/pipeline/demo.yaml --run-name mdm2 --plan
```

Remove `--plan` to run. Select one MDM2 task with its individual `--task` flag.
The cyclic task authors a real terminal C–N bond and cyclic offsets; the
disulfide task fixes both cysteines and their SG–SG bond.

## Advanced workflow configuration

A workflow file needs only the settings you want to change. Leave other fields
out to inherit the preset. Use `null` only where it has a documented meaning,
such as disabling `per_family`; required numbers and booleans cannot be blank.
See `configs/pipeline/demo.yaml` for a runnable example. Molecular inputs belong
in the separate task YAML.

Put these operations under `iteration` in the workflow configuration; use the
same fields in `rounds.N` to change a particular round:

| Operation | Accepted input |
| --- | --- |
| `mpnn` | A proposal count, or `{source: auto, count: 4, settings: {...}}` with native MPNN settings. |
| `include_gt` | Boolean: propose the current sequence as well as MPNN sequences. |
| `refold` | A method string, or a mapping with `method`, `conformers`, native `af2`/`esmfold2`/`af3` settings and `worker_gpus`. |
| `filter` | A list of existing validation criterion names, or `{criteria: [...], criteria_route: auto, criteria_mode: all}`. |
| `selection` | `mode: cluster` or `top`, plus the limits described below. |
| `partial` | Boolean, or `{enabled: true, fraction: 0.625, replicas: 4, include_parent: true}`. |
| `relax` | Boolean: run whole-complex OpenMM before partial diffusion. |

Set `partial: false` to disable partial diffusion or `relax: true` to enable
whole-complex relaxation. The default fraction 0.625 runs the last 250 intervals
of the native 400-interval grid, starting at index 150; it is not an RMSD bound.
The pipeline fills `task.partial` from the selected refold and uses the same
runtime as [single-reference partial generation](partial.md). Existing
coordinate-condition sites take their restart coordinates from that complex;
original identity permissions and motif references are preserved.

MPNN `source` can be `auto`, `protein_mpnn`, `soluble_mpnn`, `ligand_mpnn`,
`cyclic_mpnn` or `na_mpnn`. Automatic routing follows native chemistry and model
capabilities. Fixed identities and motif permissions remain in force. Unsupported
redesign does not count repeated current sequences as new diversity.

For example, save this standalone binder workflow as
`configs/pipeline/custom.yaml`:

```yaml
preset: ppi
recycle: 3
iteration:
  mpnn: 2
  refold: esmfold2-fast
  filter: [af3_basic]
  selection: {max_pass: 16}
  partial: {replicas: 2}
rounds:
  1:
    mpnn: 1
    selection: {mode: cluster, num_clusters: 16}
  2:
    mpnn: {source: auto, count: 3}
final:
  max_pass: 8
```

Apply it to a task from the native collection:

```bash
partidesign pipeline configs/examples/pipeline_tasks.yaml --task protdbench_insulin_receptor \
  --config configs/pipeline/custom.yaml --run-name my_binder
```

For a per-task workflow change, add an override to that same workflow file:

```yaml
tasks:
  protdbench_insulin_receptor:
    iteration:
      mpnn: 2
```

The workflow's `tasks` mapping contains only workflow overrides. Keep molecular
fields in the native task YAML, without duplicating them here.

`final` accepts `method`, native `af2`/`af3` settings, `max_pass`, `criteria` and
the native final-selection controls. For example,
`final: {method: af3, af3: {samples: 3}, max_pass: 8}` changes final prediction
sampling without changing iteration refolding. Native predictor settings are
listed in the [validation guide](validate.md).

Settings resolve in this order:

```text
internal defaults → profile → common workflow → common round
→ task workflow → task round → explicit CLI within its documented scope
```

Most mappings merge field by field, and lists replace earlier lists.
**A `selection` block containing `mode` replaces the whole earlier selection
block**, even when the mode is unchanged. Omitted limits then use that mode's
defaults. For example, `{mode: top, per_family: null}` does not retain an earlier
`max_pass: 16`; include `max_pass: 16` again if you want to keep it.

To change just one limit, leave out `mode`. This round-2 override disables the
family cap while preserving the inherited `max_pass` and other selection fields:

```yaml
rounds:
  2:
    selection: {per_family: null}
```

`--method` and `--criteria` affect iteration;
`--final-criteria` affects final validation. `--first-mpnn-count` and
`--num-clusters` affect round 1; `--mpnn-count`, `--max-pass` and
`--max-pass-ratio` affect later rounds. `--recycle` sets the total round count.
Generation flags such as `--ckpt`, `--batch-size` and `--compile` keep their
native meanings.

## Selection controls

`selection.num_clusters` sets K for `mode: cluster`. For `mode: top`,
`per_parent` limits children per immediate input structure. Middle rounds also
use `per_family: 2` across a retained refold and its partial children; positive
integers change that cap and `null` disables it. First and last rounds omit the
family cap. `max_pass` caps task-wide survivors after these limits.
`max_pass_ratio` caps the fraction of unique proposed sequences before filtering;
when both limits are set, the smaller applies. There is no minimum-pass target
or backfill after a duplicate or transition failure.

Deduplication preserves chain boundaries, modality and modified-residue tokens.
Legal cyclic rotations are equivalent only when motif and connection mappings
remain consistent. Clustering uses normalized token edit distance; multiple
design chains use their largest corresponding-chain distance.

## Criteria and ranking

Criterion names and native definitions come from
[`configs/criteria.yaml`](../configs/criteria.yaml), as in `partidesign validate`.
Automatic iteration screening uses relaxed versions of those rules. The
protein-binder screen uses `binder_iptm > 0.6`, `binder_ptm > 0.7` and
`complex_design_rmsd < 4 Å`; selected other geometric limits are 1.5 times their
native values. Identity, discrete and clash checks retain their native limits.
Explicit `filter: [af3_basic]` requests that existing strict criterion instead.
Its name does not select the prediction backend.

The pipeline normally requires all requested criteria on the same prediction.
Use `criteria_mode: any` in a filter mapping for a union. Additional native
`presets:` definitions can be supplied with `--presets FILE.yaml`. The PPI
protein final route preserves its intentional `all_and_mean` reduction for
`af2_ig_protdbench`; it is not an exact copy of every standalone reduction.

For coordinate motifs, new runs require the original-reference motif RMSD to
pass at **round 1 and final only**, against the original motif rather than the
current refold. Protein/peptide uses validate `af3_ame`'s
`motif_rmsd_reference <= 1.5 Å` rule; RNA/DNA uses validate `af3_na_motif`'s
`motif_rmsd_reference <= 3 Å` rule, selected by the design modality.
The gate remains mandatory with `criteria_mode: any`. If a task overrides
the applicable native rule, the gate uses its resolved value. Later iterations skip motif
RMSD computation, source-reference reads and motif-geometry filter rules; normal
confidence and self-consistency filters continue. Fixed motif identities,
conditions, role mappings and chemistry remain active in every round. A
sequence-only identity condition does not create a coordinate-motif gate.

Motif tasks retain design-only pLDDT ranking. Final prediction ranking happens
before the strict motif check, with the same no-fallback behavior described
under [final validation](#final-validation). Motif RMSD is a pass/fail check;
it does not change the ranking score.

Special polymer–polymer connections, including physical head-to-tail closures,
disulfides and staples, force ESMFold2-fast iteration. Ordinary backbone edges
and offset-only intent do not trigger this override. Final validation keeps its
configured backend and [native chemistry limits](validate.md#antibodies-cyclic-peptides-and-modified-chemistry).
An iteration override does not extend AF2 final capabilities or add a final
bond-geometry criterion.

## Resume and reuse

To replay saved task and workflow snapshots, use a fresh run name from the same
invocation directory as the original run:

```bash
partidesign pipeline inference/my_run/tasks.yaml \
  --config inference/my_run/config.yaml --run-name replay
```

To resume, repeat the same command with `--resume`. Effective task and workflow
settings must match; changed settings need a new run. Retained prediction
evidence is reused for equivalent requests, while parent-relative metrics are recomputed.
Existing runs retain their legacy layout and saved checking policy on resume;
the first/final motif policy applies to new runs.

To start from saved generation, add `--from-run PATH` with the same native tasks;
upstream structures remain unchanged. Use a new run name for the pipeline's
outputs. See [partial generation](partial.md) when restarting from one actual
reference CIF without an iterative workflow.
