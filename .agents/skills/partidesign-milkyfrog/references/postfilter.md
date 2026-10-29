# Postfilter successful protein and peptide complexes

Use this reference for requested CPU relaxation and structural screening after
validation. Read the active checkout's `docs/postfilter.md` for the maintained
selection contract, dependencies, output format, and resume behavior. Consult
`configs/postfilter_presets.yaml` for scientific defaults and thresholds, and
`configs/postfilter.yaml` for runtime settings. Do not start postfiltering merely
because validation finished.

## Preserve the validation handoff

`partidesign postfilter` accepts saved successes from validation and accepted
final representatives from a pipeline. Supply a run, task, validation method,
pipeline `final/`, or the corresponding `success/` directory. A loose success
CSV or PDB/CIF does not carry enough information to establish acceptance.

For standalone validation, keep the published success records and their
complete complex refolds together. Older records may also need the saved task
configuration or `.state/plan.json` for partner roles. Original generated
structures and the generation `result.csv` are not required. Selection takes
the union of named criteria's `passing_prediction_keys`: a linked monomer, a
model that did not individually pass, or completion-only `unfiltered` status
is insufficient. Duplicate nominations are scored once; different
sequence/model slots stay separate.

For pipeline inputs, keep `final/` intact, including its result and selection
CSVs, prediction records, `success/` structures, and hidden state. Postfilter
uses the saved final selection policy and screens only each accepted published
representative. With the PPI preset, both selected AF2 model winners and their
matching monomers must have passed; retain both selected pairs as evidence.
Other raw predictions and isolated monomers do not become postfilter inputs.
Workers map saved chain labels to the exact sequences in the selected complex
and reject ambiguous mappings.

Supported structures use the twenty standard amino acids in protein–protein
or peptide–protein complexes. Antibody heavy/light chains form one binder
assembly against the antigen. Linear and physical head-to-tail cyclic peptides
are supported. Monomers, nucleic acids, ligands, nonstandard residues, mixed
protein/peptide binder assemblies, and ambiguous roles are excluded or reported
precisely. Saved peptide roles or the actual binder closure select the peptide
path; chain length and `cyclic_offset` do not establish physical closure.
A declared closure missing from the refold is rejected. Preserve actual
identities and bonds; do not reconstruct a ring from proximity or generated
coordinates.

## Plan and run the requested scope

```bash
partidesign postfilter inference/RUN --task protein_fixture --plan
partidesign postfilter inference/RUN/TASK/validate/af2_ig/success --plan
partidesign postfilter inference/PIPELINE_RUN/TASK/final/success --plan
partidesign postfilter inference/RUN --task protein_fixture \
  --config postfilter-local.yaml --out postfilter/protein_fixture
```

The CLI accepts input paths and `--out`, `--config`, repeatable `--task`,
`--plan`, and `--rerun`. Plan mode reads metadata and reports selection and
tool availability without parsing candidate CIFs, initializing PyRosetta, or
writing postfilter output. It does not prove native execution or chemistry.
After prerequisites pass, remove `--plan` for authorized execution with the
same requested scope.

Both `partidesign_cu12.yaml` / `partidesign` and
`partidesign_cu13.yaml` / `partidesign_cu13` install Ray, CPU PyRosetta, and
DSSP. The checkout provides DAlphaBall; wheel-only users must provision its
executable. Use the maintained guide for pinned manual SDK/DSSP/libmcfp
installation and tool resolution. Do not install tools for a usage question.

Runtime overlays accept exactly `seed`, `workers`, `threads`, `timeout`,
`dalphaball`, and `dssp`. Defaults use seed `2026`, automatic CPU allocation
with `workers: null`, one thread per actor, and a 1800-second per-sample timeout.
For a small explicit pool, use `workers: 8` and `threads: 1`. Persistent Ray
CPU actors each initialize PyRosetta once and process complete samples across
tasks. One source parse feeds in-memory mapping and metric views. No GPU is
allocated.

Protein defaults require `bc1_protein_structural` and `bc2_protein_geometry`
on the same refold; peptide defaults use `bc1_peptide_structural`. Every
selected preset is required, unlike validation's OR across named criteria.
Read the preset YAML for thresholds and protocol rather than copying tables
from another release. The protocol remains `beta_nov16`, one repeat and
200 maximum minimizer iterations, with movable backbone/sidechains, fixed
jumps, repacking only, and no fixed-coordinate restraints. Physical closure
penalties are used only during relaxation.

Only selected preset metrics and their dependencies run. BC1 Rosetta metrics
use the relaxed Pose; BC1 loop content uses one DSSP assignment of the input
success refold, and required BC2 geometry uses that same input. Missing required
metrics are `unavailable` and cannot pass. Do not fill missing values with
zeros or request unused composition/terminal-property reports.

## Read results and resume samples

Each task defaults to its own `postfilter/` directory. `--out DIR` is literal
for one task; multiple tasks use `DIR/TASK_NAME` and require distinct task names.
Outputs include `metrics.csv`, `success.csv`, `summary.json`,
`resolved_config.json`, small `records/`, and relaxed CIFs in `structures/`.
Rows keep method/sample/sequence/prediction identities and preset statuses.
A scientific threshold failure retains its structure and metrics. Distinguish
that result from incomplete execution or an unavailable required metric.

An ordinary repeat reuses matching complete records with existing CIFs and
runs missing, failed, or interrupted samples from the selected input. Explicit
`--rerun` reprocesses selected samples in a compatible output directory.
Older or unrecognized output formats are rejected untouched; select a fresh
`--out` rather than altering them. Keep the run configuration, records, and
structures together. Resume operates on completed samples and does not restore
partial native geometry.

Exit `0` can include zero passing candidates. Exit `1` reports configuration
or run-level errors; exit `2` reports incomplete execution, selection errors,
or unavailable required metrics.

Postfilter preserves upstream results and rankings and runs no folding,
sequence redesign, or PartiScore scoring. Clustering in
[workflows.md](workflows.md) independently compares original generated polymer
backbones and does not consume these relaxed refolds.
