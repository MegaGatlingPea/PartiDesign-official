# Partial diffusion and iterative pipelines

`partidesign pipeline` is the one-command runner for de novo generation or
optimization of existing complexes, iterative sequence/structure refinement,
and final validation. Start with the command
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
  design, or an existing-structure directory with annotation.csv. Keep recycle, iteration and final settings in separate workflow files
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


### Optimize a directory of existing complexes

Use `partidesign pipeline DIRECTORY` with `DIRECTORY/annotation.csv`, not
`design DIRECTORY`. Each listed CIF/PDB is one actual full-complex starting
conformation. Read only the rows selected by the user's task scope; do not
search the directory for other candidates or filter by an old pass/reject label.
Rows in one `task` share selection and final-pool limits while retaining their
own native task conditions and original motif sources.

Metadata columns are `filename`, optional `task` (default `optimize`),
`sample_id` (default file stem) and `source_status`. Other columns are the
complete native task fields, with structured JSON cells; do not invent an
annotation-only motif, fixed-sequence or bond syntax. Empty cells omit fields,
and explicit null keeps native semantics. Paths resolve beside annotation.csv.
Never request a manufactured input result.csv. Existing fixed ligand context
uses a complete molecular graph with known bond orders together with its native
ligand condition.

Normal tasks use native design_chains/condition/cyclic fields; antibodies use
format/antibody/antigen. Keep FR/CDR permissions, role mappings and soft
framework templates. Antibody partial accepts cdr_lengths by VHH or VH/VL role,
with all three actual CDR counts per supplied role when layout is ambiguous.
A scaffold and chain name alone do not establish an arbitrary antibody's
region/source mapping.

Directory initialization runs partial before round 1: four children plus the
retained seed by default. It does not count toward recycle. Initial settings
inherit the first-round transition, then initial.partial and per-row sample_num
/ partial.fraction; explicit CLI partial-replicas/fraction win. Per-row sampling
overrides apply only to initialization. Disabling initial.partial starts directly
with the seeds. Recycle=1 still includes that configured initial step.

Use configs/pipeline/optimize.yaml; it inherits the selected preset's final
methods and validation criteria. The bundled
configs/examples/optimization/annotation.csv lists real linear and cyclic
complexes. Use --preset ppi for eligible AF2-IG, Mol-1, AF3 and
ESMFold2-standard final validation, or omit it for automatic protein/nuc profiles
with Mol-1, AF3 and ESMFold2-standard. Native antibody rows automatically select
the antibody profile; use `--preset antibody` explicitly for clarity.

The fixed five-seed GLP1R example is `configs/examples/antibody_opt`. Its columns
are filename, task, sample_id, source_status, format, antibody, antigen and
partial, with JSON objects for the last three. Use native `format: antibody`,
VHH mapped to B, scaffold bindcraft2_vhh, all-CDR official lengths, selected
framework permissions and the independent paratope declaration from
[task-recipes.md](task-recipes.md). Per-row `partial.cdr_lengths.VHH` counts are
(7,5,25), (7,7,10), (7,6,8), (7,5,21), (7,7,6). The full current seed is the
reference; antigen input is that same full CIF cropped to A. Antigen label 1–119
corresponds to author 21–139. Scaffold soft template, partial reference and
retained antigen remain distinct. See `<repo>/docs/pipeline.md#antibody-pipelines`
for the JSON-cell example.

## Defaults, criteria and ranking

For YAML input, generate once using native task budgets; directory input starts
with partial around the listed complexes. The first refolding round counts toward
`recycle: 3`: GT1 + auto MPNN1 first, GT1 + MPNN4 later, ESMFold2-fast with one
conformer. GT means the current candidate sequence. Auto MPNN follows native
chemistry routing; NA-MPNN with ligand context preserves contacting identities
through self_fi. All presets use relaxed pipeline screening by default, with
original-reference motif checks still strict at round 1 and final.

First selection deduplicates and clusters to K=200. Later rounds take the best
design-chain pLDDT candidate per immediate input parent, then deduplicate task-wide.
Middle rounds that continue keep at most two per partial family before the
global 200-candidate cap. A family is the latest transition's retained refold
plus its partial children. Set `selection: {per_family: null}` to disable only
this cap while preserving the other selection settings; including `mode`
replaces the whole selection block. First and last refolding rounds omit this
cap by default; `rounds.N.selection.per_family` can enable it explicitly. Final
validation uses `final.pool` instead. Context pLDDT is excluded, and removed candidates are not
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
| protein, nuc | Mol-1, AF3 and ESMFold2-standard, five samples per model; each selects its native best-ranked prediction before applying its criteria. |
| antibody | ESMFold2-standard only, five GT-only samples per selected candidate; select the best by 0.8 ipTM + 0.2 pTM, then apply final criteria. |
| ppi, explicitly selected | AF2-IG, Mol-1, AF3 and ESMFold2-standard. Mol-1, AF3 and ESMFold2-standard each use five samples; AF2-IG models `[1, 2]` each use five matched complex/monomer pairs. Each AF2 model chooses its best complex by 0.8ipTM+0.2pTM and its same-sample monomer; both model pairs must pass AF2's internal filter. |

Auto selects antibody for native antibody tasks, otherwise protein/nuc, never
ppi. PPI requires eligible standard
protein/peptide design with protein context and native AF2 chemistry limits.
PPI's AF2 final criteria are af2_ig_protdbench and af2_ig_peptide. Criteria
definitions come from `configs/criteria.yaml`, with intentional iteration
relaxations derived privately; the PPI protein AF2 final route retains its
all_and_mean reduction.
Explicit names such as `filter: [af3_basic]` select the existing strict filter.

For iteration without general confidence or self-consistency quality thresholds,
use `iteration: {filter: [blank]}` or `--criteria blank`. This native preset uses
one complex-scoped `plddt >= 0` rule with `any_prediction`; completed predictions
and finite pLDDT remain required. Original-reference motif gates at round 1 and
final, mandatory antibody contacts, native identity and bond constraints,
deduplication, ranking and selection budgets still apply. Final criteria remain
those of the selected profile unless explicitly changed with `--final-criteria`
or `final.criteria`.

All final selectors choose before criteria and never backfill a failing winner
with a lower-ranked passing sample. All prediction evidence remains available.

All antibody pipelines require `antibody_contacts` on the same selected
prediction as the configured criteria, even with `criteria_mode: any` or another
explicit preset. Its defaults require `antibody_interface_residues >= 7` and
`antibody_off_paratope_contact_fraction <= 0.20`. Contacts are unique binder
residues with any antigen heavy atom ≤4 Å away; VH–VL internal contacts do not
count. The antibody profile requires RFD4 min iPAE and target-aligned binder
RMSD each ≤3 Å during iteration and each ≤2 Å final. Customize contact thresholds only through
the existing native `presets.antibody_contacts` rules. Standalone validate
criteria keep their defaults; contact screening there is optional.

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

To override the preset with one final filter, set `method: af2_ig`,
`method: mol1`, `method: af3`, or `method: esmfold2-standard` under `final`.
Standard uses its own checkpoint, query-only input, and native
`0.8*ipTM+0.2*pTM` best-sample ordering before filtering; it does not
consume external MSA/templates. Its default five samples are configured by
`esmfold2: {samples: 5}` under `final`. ESMFold2-fast remains the default iteration
model. The default final lists are `methods: [mol1, af3, esmfold2-standard]` for
protein/nuc and `methods: [af2_ig, mol1, af3, esmfold2-standard]` for ppi.
A candidate passes when any method passes its own criteria; AF2 still requires
both configured model pairs internally. Do not turn this into an intersection
or stop after the first method passes. Per-method criteria use `criteria_routes`
in workflow YAML.

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
`--msa off` disables both alignments and templates. Explicit
`--msa auto --msa-templates false` keeps alignments and skips template work;
a compatible template-on cache can supply a template-free view, never the
reverse. The pipeline prepares the
required task-input queries before GPU work starts; queries with the same search
profile share preparation across tasks, rounds and final methods. Their cache
generations stay pinned for the run and `--resume`. `--msa-refresh` refreshes each
query/profile once for a new run, not per candidate or model; a resumed run keeps
its saved references.

Use `partidesign msa` with native YAML or the same annotated directory to
prepare selected task contexts separately, and shared workflow
`msa` settings or `--msa-source` to choose the route. The cache remains
`<model_dir>/.cache/af3_msa/`. Each generation's `manifest.json` records template
mappings and provenance. The run's compact `msa.csv` lists task/chain, source,
cache references, MSA depths and template counts without copying search data
into round or final-method directories. Keep the shared cache available when
resuming a run. Read `docs/msa.md` for commands, source resources, cache-only use
and refresh.

Motif tasks use design-only pLDDT ranking plus applicable native motif filters.
Do not introduce a separate motif ranking score.

Strict validation screens can be selected by their native names in `final.criteria`.
Their `pipeline_` variants can be selected in iteration filters, including from
the PPI profile. Derived PAE/RMSD upper bounds are 1.5 times the native value;
pTM/ipTM, sequence and clash conditions stay fixed. Keep automatic routes unless
the user selects a different criterion. The extra original-reference motif gate
still applies only at first round and final; middle rounds omit both legacy and
strict motif RMSD rules while retaining other strict conditions. Read
`docs/validate.md` for exact thresholds and atom/alignment conventions.

MPNN proposals remain eligible for strict partner-aligned RMSDs. Same-type
residues retain all observed reference atoms; redesigned positions use the
conserved protein backbone or nucleic-acid sugar-phosphate framework. Preserve
the target-only fit and original fixed/motif/chemistry constraints, and report
the comparison selection in metric provenance. Do not turn sequence redesign
into an unavailable metric merely because a new sidechain has no old coordinates.

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

The pipeline uses one Ray runtime with independent partial-diffusion, MPNN
and refolding worker pools. MPNN workers request 1 CPU and 0.0625 GPU each,
allowing up to 16 per GPU, further limited by parent count and available CPUs.
Each phase closes its workers and model cache before the next phase, so later
phases reload their models. `--worker-gpus` sets refolding resources; partial
workers independently request one GPU and `generation.num_workers + 1` CPUs
each. Both initial and inter-round partial diffusion retain the effective
`generation.batch_size` (set by `--batch-size`), regardless of parent count or
refolding budgets. For example, one parent with 20 replicas and batch size 16
runs as batches of 16 and 4.

Special polymer–polymer bonds switch AF2-IG/AF3 iteration to ESMFold2-fast;
explicit Mol-1 iteration keeps its native covalent-bond input. Ordinary backbone
edges and offset-only intent do not trigger an override. Final chemistry limits remain in force;
no automatic final bond-geometry gate is added. Pure ligand pipeline design is
unsupported. Equivalent prediction requests reuse evidence; parent-relative
metrics are recomputed. `--resume` needs identical effective settings; changed
settings need a new run. `--from-run` preserves upstream generation.

## Antibody commands

```bash
partidesign pipeline configs/examples/antibody.yaml --task glp1r_vhh_bindcraft2 --preset antibody --run-name glp1r_vhh_denovo
partidesign pipeline configs/examples/antibody_opt --preset antibody --run-name glp1r_vhh_optimization
```

Keep the normal workflow budgets: `recycle: 3`, first GT1+MPNN1, later GT1+MPNN4,
ESMFold2-fast iteration, partial fraction 0.625 with four replicas and parent
retention, `relax: false` and five GT-only ESMFold2-standard final predictions.
Rank by 0.8 ipTM + 0.2 pTM before filtering, without replacing a failing winner.
ESMFold2 currently stays query-only; shared ColabFold MSA is not consumed. Use `--plan`
for preview; do not reduce production examples merely to create a smoke test.

## Demos and outputs

Use the plain native task collection `configs/examples/pipeline_tasks.yaml`
with `--task`, just as in design. Workflow presets live in `configs/pipeline/`
and contain recycle/iteration/final settings only. The example contains InsR,
PDL1, BHRF1 and MDM2 linear/cyclic/disulfide tasks, 32 initial samples each.
`configs/examples/partial_diffusion.yaml` is the single-reference partial template,
with only protein_refold.cif bundled as its input structure.

```bash
partidesign pipeline configs/examples/pipeline_tasks.yaml --task protdbench_insulin_receptor --preset ppi --run-name insulin_receptor
partidesign pipeline configs/examples/pipeline_tasks.yaml --task protdbench_insulin_receptor --preset ppi --config configs/pipeline/demo.yaml --run-name insulin_receptor_demo
```

The lighter demo uses ESMFold2-fast iteration with relaxed
`pipeline_protein_binder` / `pipeline_peptide_binder` screening, 200 initial
clusters, a 200-candidate limit in later rounds and the final pool, and the preset
worker budgets. Its final validation uses only Mol-1 with five samples and native
ranking before criteria. Prepare ESMFold2-fast and Mol-1 alongside generation and
auto MPNN. Standard `ppi` final validation runs all four configured methods.

YAML output begins at `inference/<run>/<task>/design/`; directory input uses
`start/` for copied seeds, resolved result.csv and initial partial/<sample_id>/
children, with root annotation.csv. Both continue through `round_1/`, `round_2/`,
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
