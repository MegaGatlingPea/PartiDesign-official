# Postfilter successful protein and peptide complexes

`partidesign postfilter` relaxes successful refold complexes and applies the
structural rules saved in its preset configuration. It consumes passing
results from `partidesign validate` or the published final representatives from
`partidesign pipeline`, preserving upstream structures, validation results,
reward values, and rankings. It performs no sequence redesign, folding
predictions, or PartiScore scoring.

## Select passing inputs

```bash
partidesign postfilter inference/RUN --plan
partidesign postfilter inference/RUN/TASK/validate/af2_ig/success --plan
partidesign postfilter inference/PIPELINE_RUN/TASK/final/success --plan
partidesign postfilter inference/RUN --task protein_fixture --task peptide_fixture \
  --config postfilter-local.yaml --out postfilter/RUN
```

Supply a run, task, validation method directory, pipeline `final/`, or its
`success/` directory. You can combine several input paths. Keep published
success records beside their selected complete refold files; a loose structure
or success CSV does not carry enough acceptance metadata.

For standalone validation, keep the validation output; the original generation
CIFs and `result.csv` are unnecessary. Older records may also need saved task
configuration or `.state/plan.json` to recover partner roles. For a pipeline,
keep the complete `final/` directory, including `result.csv`, `selection.csv`,
`predictions/`, `success/`, and `.work/`. These records identify the accepted
representatives and the final validation policy.

For standalone validation, selection uses every saved named passing criterion's
`passing_prediction_keys`. It keeps the union of their actual complete complex
predictions, scoring duplicate nominations once. Different sequence/model slots
remain distinct. Linked monomers and predictions that did not individually pass
do not establish eligibility. Completion-only `unfiltered` decisions are ineligible.

For a pipeline, postfilter screens the published final representative for each
accepted candidate, using the saved final validation policy. The default PPI
preset requires the selected winners from AF2 models 1 and 2 to pass, with
their matching monomer comparisons. Other final samples remain supporting
prediction evidence. A missing representative or incomplete final acceptance
record is reported as an input-selection failure.

Repeat `--task NAME` to select tasks. Use `--plan` to preview selected samples,
exclusions, and dependency availability. It reads metadata without parsing
candidate CIFs, initializing PyRosetta, or creating postfilter output; normal
package startup may create cache directories. Workers check the actual
chemistry when they read each selected structure. `--out` chooses the output
directory, `--config` supplies execution settings, and `--rerun` reprocesses
previously completed samples.

Supported inputs use the twenty standard amino acids in protein–protein or
peptide–protein complexes. Linear and physical head-to-tail cyclic peptides
are supported. All declared binder chains form one assembly; antibody heavy
and light chains stay together against the antigen. Monomers, nucleic acids,
ligands, nonstandard residues, mixed protein/peptide binder assemblies, and
ambiguous partner roles are excluded or reported precisely.

Workers resolve saved chain labels against exact sequences in the actual
refold. Existing matching labels retain their identity; renamed chains require
unique sequence matches. Ambiguous or changed sequences are rejected. CIF
polymer order follows authoritative `label_seq_id`, including reordered atom
rows, while public atom identities and float64 source coordinates are retained.
PDB files with ambiguous, non-increasing author residue locators require an
explicit CIF mapping. Saved peptide modality/subtype or an actual binder
head-to-tail bond selects the peptide path; chain length does not classify it.

A physical ring requires an explicit single bond from the last residue's
carbonyl C to the first residue's N, with an intact standard peptide backbone.
Neither a short distance nor `cyclic_offset` establishes that bond. A declared
`input_metadata.cyclic_chains` closure missing from the actual refold is
rejected after chain mapping, for binder and target chains alike. Rosetta
installs terminal, disulfide, and cutpoint variants before the final atom map,
then restores source coordinates once. Head-to-tail overlap penalties apply
only during relaxation and are removed before metric calculation. Initial
identity/graph verification and final connection/closure geometry checks
protect the retained chemistry.

## Dependencies and execution settings

Both Linux x86-64 environment definitions, `partidesign_cu12.yaml` and
`partidesign_cu13.yaml`, install Ray and the native postfilter dependencies.
Use the same environment as the other PartiDesign commands. For a new CUDA 12
install, run from the checkout root:

```bash
conda env create -f partidesign_cu12.yaml
conda activate partidesign
```

Use `partidesign_cu13.yaml` / `partidesign_cu13` for CUDA 13; see the
[environment guide](environments.md). Both definitions install PyRosetta
`2026.39+release.49e05e797c` from its
[official source](https://www.pyrosetta.org/downloads), DSSP `4.6.1`, compatible
`libmcfp 2.0.1`, and Ray `2.53.0`. The SDK pin targets CPython 3.12 on Linux
x86-64. Postfilter runs on CPU and allocates no GPU.

For manual setup in an existing compatible Python 3.12 Linux x86-64 environment,
install the same tested SDK and command-line dependencies:

```bash
conda install -c conda-forge \
  'dssp=4.6.1=np2py312h4f94bcb_0' 'libmcfp=2.0.1=h171cf75_0'
python -m pip install 'packaging==24.2' 'ray==2.53.0' \
  'https://west.rosettacommons.org/pyrosetta/release/release/PyRosetta4.Release.python312.ubuntu.wheel/pyrosetta-2026.39%2Brelease.49e05e797c-cp312-cp312-linux_x86_64.whl'
python -m pip check
```

These commands target the tested Python 3.12 Linux x86-64 setup. PyRosetta's
license terms apply. Its SDK, database, and downloaded wheel are not included
in Git or the PartiDesign wheel.

The backend also needs DAlphaBall. A source checkout provides the small Linux
x86-64 executable and full corresponding LGPL source in
`community_models/rosetta/`; see its [tool guide](../community_models/rosetta/README.md)
for rebuilding. A wheel-only installation ships the source and license;
provision the executable in the active environment or set `dalphaball`
explicitly. Both environment definitions install DSSP automatically.

Tool resolution checks an explicit config path, checkout/packaged
`community_models/rosetta/bin`, the active interpreter's `bin`, then `PATH`.
Loading or scoring never downloads tools or models implicitly.

Runtime settings have exactly six fields in
[`configs/postfilter.yaml`](../configs/postfilter.yaml): `seed`, `workers`,
`threads`, `timeout`, `dalphaball`, and `dssp`. The default seed is `2026`,
`workers: null` automatically uses the allocated CPU budget, `threads: 1`
sets each actor's thread budget, and `timeout: 1800` limits each sample in
seconds. Explicit worker counts are capped by the CPU allocation and job count.
A small execution overlay is:

```yaml
# postfilter-local.yaml
seed: 2026
workers: 8
threads: 1
timeout: 1800
# Set tool paths only when automatic resolution cannot find them.
dalphaball: null
dssp: null
```

```bash
partidesign postfilter inference/RUN --config postfilter-local.yaml
```

A shared pool of persistent Ray CPU actors processes candidates across tasks.
Each actor initializes PyRosetta once, restores its in-memory RNG state before
each sample, and owns parsing, relaxation, metrics, and publication for that
sample. Each CIF is parsed once; mapping and metric views reuse the document
in memory. A lost or timed-out actor is replaced for later samples. An already
published completion can be recovered; otherwise that sample is reported as
failed and is retried by a later command invocation.

## Scientific defaults and required metrics

[`configs/postfilter_presets.yaml`](../configs/postfilter_presets.yaml) is the
authority for binder-kind defaults, protocol, presets, and thresholds.
Protein binders require `bc1_protein_structural` and `bc2_protein_geometry` on
the same concrete refold; peptide binders use `bc1_peptide_structural`.
Separate protein and peptide candidates select their corresponding defaults.
Every selected preset must pass. The shipped relaxed-threshold profiles remain
in that configuration; threshold tables are maintained there.

The protocol uses `beta_nov16`, one FastRelax repeat, and 200 maximum minimizer
iterations. Backbone and sidechains can move, rigid-body jumps remain fixed,
and packing only repacks existing identities. It applies no fixed-coordinate
or motif restraints. Physical ring penalties are relaxation-only. Cysteine-containing
and disulfide-design refolds use this same protocol and are evaluated as
supplied. Existing structural connections are preserved; saved design
disulfide constraints are not added to a refold.

Only metrics required by the selected presets and their calculation
dependencies run. Measurements use two coordinate views:

| Calculation | Coordinates and outcome |
| --- | --- |
| Binder energy, surface hydrophobicity, interface complementarity, binding energy, buried surface, contacts, and hydrogen bonds (`bc1_*`) | The relaxed complex supplies the Rosetta measurements. Floating values are rounded to two decimals before threshold comparison. |
| Binder loop percentage (`bc1_binder_loop_percent`) | One DSSP assignment of the input successful refold measures the percentage outside helix and sheet, rounded to two decimals. |
| Backbone clashes (`bc2_backbone_clashes`) | The input refold supplies the count of interchain Cα pairs within 2.5 Å. The default protein preset requires zero. |
| Interface size (`bc2_interface_residues`) | The input refold supplies the count of binder residues with target-atom contacts within 4 Å. The default protein preset requires at least seven. |

Buried unsatisfied hydrogen bonds count heavy atoms that become unsatisfied in
the bound assembly. Two- or three-chain complexes use Rosetta's XML filter;
assemblies with four or more chains use an atomwise calculation of the complete
bound and separated partners. All partner chains and their chemical context
remain present.

Geometry metrics use raw finite values. Unused PackStat, auxiliary SASA ratios,
composition reports, and terminal/sequence properties are not calculated.
Missing, NaN, or infinite required metrics produce `unavailable` and cannot pass.
A measured value outside its threshold is a failed preset; the relaxed structure
and measurements remain available. Prediction confidence, reference RMSDs,
binder-alone refolds, and prediction averages come from validation.

## Outputs and sample-level resume

Each task defaults to its own `postfilter/` directory:

```text
postfilter/
  metrics.csv
  success.csv
  summary.json
  resolved_config.json
  records/
  structures/
```

`--out DIR` uses that literal directory for one selected task. With multiple
tasks, output goes to `DIR/TASK_NAME`; selected task names must be distinct.
Choose an output location independent of upstream validation.

`metrics.csv` has one row per selected complex refold, retaining its original
method/sample/sequence/prediction identity, required measured metrics, and each
preset's status/reason. `success.csv` contains only complete candidates passing
every required preset and points to their relaxed CIFs. `summary.json` reports
selection, completed relaxation, metric/execution failures, preset outcomes,
and success counts. Threshold failure keeps the relaxed structure and measured
values. Missing required metrics remain distinct from a scientific threshold
failure.

Small records under `records/` are published after each sample's CIF. CIF output
replaces only XYZ columns in the retained source document and is published once
with atomic replacement, preserving noncoordinate categories. Sample records contain
compact metrics, errors, timings, and scalar details.

```bash
# Reuse matching completed samples; execute missing, failed, or incomplete work.
partidesign postfilter inference/RUN --config postfilter-local.yaml
# Explicitly reprocess selected samples in this compatible output directory.
partidesign postfilter inference/RUN --config postfilter-local.yaml --rerun
# Older postfilter output stays untouched; choose a new destination.
partidesign postfilter inference/RUN --config postfilter-local.yaml \
  --out postfilter/RUN-ray
```

Resume requires a matching complete sample record and its existing relaxed CIF.
It reuses the completed sample as a unit; interrupted samples run again from
their selected input refold. Keep `resolved_config.json`, records, and relaxed
structures together. Changes requiring new calculations use `--rerun` within
a compatible output directory. Unrecognized or older output formats are
rejected untouched, including with `--rerun`; use a fresh `--out` instead.
An empty legitimate success set produces empty reports without native work.

Exit status `0` means the requested work completed, even if every candidate
fails quality thresholds. Status `2` means incomplete candidate execution,
selection errors, or unavailable required metrics. Status `1` means a
configuration or run-level error.

Clustering independently compares original generated polymer backbones;
postfilter relaxed refolds do not replace those inputs.

## Third-party licenses

See the [third-party licenses and notices](../licenses/) for source attribution
and usage terms, including the [structural metric license](../licenses/BindCraft-LICENSE)
and [geometry metric license](../licenses/BindCraft2-LICENSE).
