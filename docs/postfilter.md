# Postfilter successful protein and peptide complexes

`partidesign postfilter` relaxes successful refold complexes and applies the
structural rules saved in its preset configuration. It consumes named passing
decisions from `partidesign validate` and preserves upstream structures,
validation results, reward values, and rankings. It performs no sequence
redesign, folding predictions, or PartiScore scoring.

## Select validated inputs

```bash
partidesign postfilter inference/RUN --plan
partidesign postfilter inference/RUN/TASK/validate/af2_ig/success --plan
partidesign postfilter inference/RUN --task protein_fixture --task peptide_fixture \
  --config postfilter-local.yaml --out postfilter/RUN
```

An input may be a run, task, validation method directory, or its `success/`
directory. Multiple input paths can be supplied together. Keep the published
success sequence records and their selected complete refold files together.
Saved task configuration or `.state/plan.json` is read only when older records
need additional partner-role metadata. A loose PDB/CIF or success CSV cannot
establish eligibility. Original generated structures and `result.csv` are not
required.

Selection uses every saved named passing criterion's
`passing_prediction_keys`. It keeps the union of their actual complete complex
predictions, scoring duplicate nominations once. Different sequence/model slots
remain distinct. A success directory can also contain linked monomers or
predictions that did not individually pass; those links do not establish
eligibility. Completion-only `unfiltered` decisions are ineligible.

Repeat `--task NAME` to select task names. The command's options are `--out`,
`--config`, `--task`, `--plan`, and `--rerun`. `--plan` reports metadata selection,
exclusions, and dependency availability without parsing candidate CIFs,
initializing PyRosetta, or creating postfilter output. Actual chemistry checks
run when a worker reads the selected structure. Normal package startup may
initialize its cache directories.

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
dependencies run. BC1 Rosetta metrics use the relaxed Pose. BC1 loop percentage
uses one DSSP assignment of the input success refold; required BC2 geometry
also uses that input structure. Unused PackStat, auxiliary SASA ratios,
composition reports, and terminal/sequence properties are not calculated.
Missing, NaN, or infinite required metrics produce `unavailable` and cannot pass.

For two or three chains, buried unsatisfied hydrogen bonds use the pinned BC1
XML filter. For four or more chains, the pinned SDK's XML filter rejects the
assembly, so the native atomwise calculator uses the complete bound and
separated partner assemblies and counts heavy atoms unsatisfied only in the
bound state. All partner chains and their chemical context remain present.

BC1 floats and DSSP percentages use the upstream two-decimal comparison
values. BC2 compares raw finite values. The separate `bc1_` and `bc2_` metric
families retain their own definitions: the BC2 CA clash count uses interchain
pairs within 2.5 Å, and its binder interface residue count uses 4 Å contacts.
These structural projections preserve the pinned upstream rules; AF confidence,
reference RMSDs, binder-alone predictions, and prediction averages remain
upstream validation responsibilities.

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

## Upstream definitions and licenses

The reference definitions are pinned to
[BC1 7713aa0](https://github.com/martinpacesa/BindCraft/tree/7713aa0d0d351e4117a8befeb8541f3a8ebd3368)
and [BC2 b6fb3ae](https://github.com/PacesaLab/BindCraft2/tree/b6fb3ae15ab6a702f1156cffc11bcc3323c2fd06).
BC1 is MIT; BC2's adapted definitions carry its hosting-restricted source
license. See the separate notices in `licenses/`. This command does not imply
official association with either project.
