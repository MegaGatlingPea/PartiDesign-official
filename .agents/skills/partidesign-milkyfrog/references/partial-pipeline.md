# Partial diffusion and iterative pipelines

`partidesign pipeline` is the one-command runner for generation, iterative
sequence/structure refinement, and final validation. Start with the command
and final results before explaining advanced workflow overrides.

Read the selected checkout's `docs/partial.md` for one reference restart, or
`docs/pipeline.md` for iteration. Locate the active checkout as in SKILL.md;
these paths are not relative to the installed skill.

## Inputs and entry points

- `partidesign design`: supply one actual reference CIF and define the design
  chains and conditions in the task YAML. Add
  `partial: {reference: actual.cif, fraction: 0.625}`; no previous run or input
  CSV is required. Without `result_csv`, the loader reads only the CIF and never
  looks for a sibling CSV. An explicit `result_csv` restores saved locators and
  motif mappings; it must contain exactly one successful row for that CIF.
  Directory input is unsupported. `sample_num` sets the replica count. Actual
  lengths come from the reference, so omit `design_length` when chain sizes
  are resolved. For CIF-only contigs, use task `input` for source spans and
  specify exact gap lengths when the reference does not determine them uniquely.
  Reference geometry does not fix every design identity. Use a flex-context
  checkpoint.
- `partidesign pipeline`: read the same native task YAML and `--task` names as
  design. Keep recycle, iteration and final settings in separate workflow files
  under `configs/pipeline/`, selected by `--preset` or `--config`. Per-task
  workflow overrides belong in that workflow file, not the molecular task.
- `--plan` shows a short human preview; `--show-config` shows full internal
  diagnostics. Neither loads models nor creates output. Preserve the requested
  counts and criteria when execution is authorized.

Document configuration choices and defaults in docs and this skill; do not
create an empty production YAML template. Omit unused workflow options to inherit
preset defaults, rather than writing null values. Keep example comments in English.

Common iteration operations are `mpnn`, `include_gt`, `refold`, `filter`,
`selection`, `partial` and `relax`, under workflow `iteration`. `rounds.N` uses
those same operations. `mpnn` accepts a count or `{source, count, settings}`;
`refold` accepts a method or `{method, conformers, af2/esmfold2/af3/mol1 settings,
worker_gpus}`. Filters reuse existing validation criterion names. `partial`
accepts a boolean or `{enabled, fraction, replicas, include_parent}`; `relax` is
a boolean. `final` accepts `method` or `methods`, native predictor settings, criteria and
`max_pass`. Follow the guide rather than copying full model/criterion defaults.

## Defaults, criteria and ranking

Generate once using native task budgets. The first round counts toward
`recycle: 3`: GT1 + auto MPNN1 first, GT1 + MPNN4 later, ESMFold2-fast with one
conformer. GT means the current candidate sequence. Auto MPNN follows native
chemistry routing; NA-MPNN with ligand context preserves contacting identities
through self_fi.

First selection deduplicates and clusters to K=200. Later rounds take the best
design-chain pLDDT candidate per immediate input parent, then deduplicate task-wide.
Middle rounds that continue keep at most two per partial family before the
global 200-candidate cap. A family is the latest transition's retained refold
plus its partial children. Set `selection: {per_family: null}` to disable only
this cap while preserving the other selection settings; including `mode`
replaces the whole selection block. First, last and final selection do not use
this family cap. Context pLDDT is excluded, and removed candidates are not
backfilled.

The history-best final pool instead tracks first-round lineages: each selected
first-round refold starts a lineage inherited by its descendants. It takes the
best candidate per lineage and scoring profile, deduplicates sequences and
keeps at most 200. This can yield fewer final candidates than the last round
selected.

Between rounds, partial defaults to fraction 0.625 (last 250 of 400 intervals,
start index 150), four replicas and parent retention. OpenMM is off. The final
round has no transition. `partial: false` disables partial; `relax: true` enables
whole-complex relaxation before partial.

| Profile | Final validation |
| --- | --- |
| protein, nuc | Mol-1, five samples per sequence; select native ranking_score best first, then apply criteria. |
| ppi, explicitly selected | AF2-IG models `[1, 2]`, five matched complex/monomer pairs per model; each model chooses its best complex by 0.8ipTM+0.2pTM and its same-sample monomer; both model pairs must pass. |

Auto selects protein/nuc, never ppi. PPI requires eligible standard
protein/peptide design with protein context and native AF2 chemistry limits.
PPI final criteria are af2_ig_protdbench and af2_ig_peptide. Criteria definitions
come from `configs/criteria.yaml`, with intentional iteration relaxations derived
privately; the PPI protein final route retains its all_and_mean reduction.
Explicit names such as `filter: [af3_basic]` select the existing strict filter.
All final selectors choose before criteria and never backfill a failing winner
with a lower-ranked passing sample. All prediction evidence remains available.

ESMFold2, AF3, and Mol-1 also report `ipsae`, `ipsae_design_to_context`, and
`ipsae_context_to_design` in [0, 1], plus `ipsae_pae_cutoff` and `ipsae_pairs`.
Round and single-method final result rows retain these fields for passed and
filtered proposals. Scores use the maximum over design–context polymer pairs,
excluding design–design/context–context pairs. Protein and RNA/DNA pairs,
including atomized residues with identifiable protein Cα or nucleic-acid C1′
representatives, are supported;
ligand/ion pairs are not applicable. Missing required PAE leaves scores
unavailable. AF2 does not calculate ipSAE. Filters and ranking are unchanged.
Multi-method finals retain these scores in `<method>_metrics` in
`final/result.csv`, and as direct columns per published CIF in
`final/success/result.csv`.
Use `iteration.refold.validation.ipsae: {pae_cutoff: 10.0}` and
`final.validation.ipsae: {pae_cutoff: 10.0}` in workflow YAML to set the default
10.0 Å cutoff; use the same refold override under `rounds.N` for one round.

A final filter can use `method: af2_ig`, `method: mol1`, or `method: af3`.
Use `methods: [af2_ig, mol1, af3]` instead of `method` to run all three. A
candidate passes when any method passes its own criteria; AF2 still requires
both configured model pairs internally. Do not turn this into an intersection
or stop after the first method passes. Per-method criteria use the routes in
`configs/pipeline/demo.yaml`.

Multi-method output has only one final success directory. For each accepted
candidate, publish every executed method's best-ranked complex, including a
method that did not pass. AF2 contributes each model's best complex; matching
monomers remain prediction evidence. Keep candidate, method and prediction key
in filenames. `success/result.csv` summarizes published CIFs and method status;
`final/filter_results.csv` records the actual metric, comparison, threshold,
criterion status, method pass and overall union pass. Do not imply that every
published CIF passed. Single-method output stays unchanged.

Mol-1 passes explicit terminal C–N, disulfide SG–SG and other supported links
through native covalent-bond features, using ESMFold2's endpoint rules. It keeps
upstream positional encoding without an added cyclic-offset transform; offset
metadata alone does not create a closure bond.

Mol-1 shares AF3's context MSA/template cache. The default `source: colabfold`
searches only fixed protein context and provides MSA plus templates together;
RNA/DNA remain query-only. `source: af3` also enables fixed-RNA MSA search,
without RNA/DNA structural templates. Designed chains always stay query-only.
`--msa off` disables both alignments and templates. The pipeline prepares the
required task-input queries before GPU work starts; queries with the same search
profile share preparation across tasks, rounds and final methods. Their cache
generations stay pinned for the run and `--resume`. `--msa-refresh` refreshes each
query/profile once for a new run, not per candidate or model; a resumed run keeps
its saved references.

Use `partidesign msa` to prepare selected tasks separately, and shared workflow
`msa` settings or `--msa-source` to choose the route. The cache remains
`<model_dir>/.cache/af3_msa/`. Each generation's `manifest.json` records template
mappings and provenance. The run's compact `msa.csv` lists task/chain, source,
cache references, MSA depths and template counts without copying search data
into round or final-method directories. Keep the shared cache available when
resuming a run. Read `docs/msa.md` for commands, source resources, cache-only use
and refresh.

Motif tasks use design-only pLDDT ranking plus applicable native motif filters.
Do not introduce a separate motif ranking score.

## Geometry and execution

Preserve original conditions, motifs, context, bonds and offsets. The current
complete reference supplies the restart state and coordinates at existing
condition sites under the same native mapping, centering, units and affine.
It adds no conditions at free sites. The original source remains the motif
evaluation reference; consistency RMSDs compare against the current parent.
Pipeline builds this same task.partial in memory and calls native design,
passing its resolved internal CSV metadata explicitly so constraints and mappings
remain intact. Standalone CIF-only input requires no input CSV. Native children
still emit normal result.csv, with partial_result_csv blank when no input CSV
was supplied. Fraction is a grid suffix, not a displacement bound or shared
numeric time.

Special polymer–polymer bonds switch AF2-IG/AF3 iteration to ESMFold2-fast;
explicit Mol-1 iteration keeps its native covalent-bond input. Ordinary backbone
edges and offset-only intent do not trigger an override. Final chemistry limits remain in force;
no automatic final bond-geometry gate is added. Pure ligand pipeline design is
unsupported. Equivalent prediction requests reuse evidence; parent-relative
metrics are recomputed. `--resume` needs identical effective settings; changed
settings need a new run. `--from-run` preserves upstream generation.

## Demos and outputs

Use the plain native task collection `configs/examples/pipeline_tasks.yaml`
with `--task`, just as in design. Workflow presets live in `configs/pipeline/`
and contain recycle/iteration/final settings only. The example contains InsR,
PDL1, BHRF1 and MDM2 linear/cyclic/disulfide tasks, 32 initial samples each.
`configs/examples/partial_diffusion.yaml` is the single-reference partial template,
with only protein_refold.cif bundled as its input structure.

```bash
partidesign pipeline configs/examples/pipeline_tasks.yaml --task protdbench_insulin_receptor --preset ppi --run-name insulin_receptor
partidesign pipeline configs/examples/pipeline_tasks.yaml --task protdbench_insulin_receptor --config configs/pipeline/demo.yaml --run-name insulin_receptor_demo
```

The demo workflow uses strict `af3_basic` screening, 200 initial clusters,
a 200-candidate limit in later rounds and the final pool, and the preset worker
budgets. Its final methods are AF2-IG, Mol-1, and AF3 with union acceptance.
Prepare all three predictors. The standard `ppi` profile retains AF2-only final
validation and its general screening defaults.

New output is `inference/<run>/<task>/design/`, `round_1/`, `round_2/`,
`round_3/` and `final/`. Start with root `summary.csv` for phase counts,
`msa.csv` for shared search inputs when used, and phase `result.csv` /
`selection.csv` for candidates. Each round's `refold/` contains the candidate
structures named in its result table; raw prediction evidence is under
`.work/predictions`.
`partial/<parent>/sample*.cif` holds children; `relaxed/` is optional;
`final/success/` holds accepted representatives for a single method, or each
executed method's best complex per accepted candidate in a multi-method final.
Multi-method prediction evidence is under `final/predictions/<method>/`;
`final/filter_results.csv` explains each method's thresholds and overall acceptance. Root `config.yaml` contains
workflow settings; `tasks.yaml` contains the native task snapshot. Hidden
`.work/` directories hold inputs, cache, snapshots and resumable state.
Completed runs prune raw prediction caches by default. `--keep-cache` preserves
raw confidence arrays and prediction reuse; failed/interrupted runs keep them.
Structures, metrics, final predictions, metadata snapshots and completed-stage
resume are retained. Never remove all `.work` directories: native input metadata
is still needed. `partidesign postfilter` accepts pipeline run/task/final/success
inputs and selects only accepted final representatives with their saved policy,
not all raw predictions or monomers.

Existing runs retain their legacy layout and resume behavior.

Completed pipeline stages keep saved metrics on resume. Existing standalone
validation evaluations require explicit `--recompute-metrics` to add ipSAE
using their saved plan and confidence data. Without raw PAE, only
saved ipSAE with a matching definition, chain-pair scope, and cutoff can be
retained; absent evidence or a changed cutoff stays unavailable. Recomputation
does not start new predictions. See [validation.md](validation.md) for saved-plan
cutoff rules.

A completed run can have fewer rounds, failed partial children or no final
passes. Inspect counts and row errors before reporting success. A reference
compilation failure produces no child. Parent retention can permit continuation
when children fail. Preserve requested criteria and constraints during debugging.

## Coordinate-motif checking policy

For new runs, motif geometry is checked strictly at round 1 and final, not at
intermediate iterations. Both compare against the original motif, never the
current refold. Protein/peptide derives validate af3_ame's
motif_rmsd_reference <= 1.5 Angstrom rule; RNA/DNA derives af3_na_motif's <= 3
Angstrom rule, chosen by design modality. Respect the applicable resolved task
override. The gate is ANDed with the ordinary filter even when criteria_mode is any. Only
record.motif_positions signals coordinate motifs; fixed identity alone does not.
Later metrics skip source recovery and motif RMSDs, while identity permissions,
_sequence_metrics, scaffold/self-consistency metrics and assert_invariants stay
active. Per-call private metric controls must never enter prediction inputs or
resident-worker identity. Legacy runs keep their saved checking policy. Ranking
continues to use design-only pLDDT, with native final selection before checking.
