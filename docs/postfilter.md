# Postfilter successful protein and peptide complexes

`partidesign postfilter` relaxes **successful refold complexes** and measures
protein–protein or peptide–protein structural properties. It consumes the named quality decisions
saved by `partidesign validate`. It does not use generated design coordinates,
run new folding predictions, redesign sequences, or change upstream results.

## Select validated inputs

```bash
partidesign postfilter inference/RUN --plan
partidesign postfilter inference/RUN/TASK/validate/af2_ig/success --plan
partidesign postfilter inference/RUN --method af2_ig \
  --upstream-criteria af2_ig_protdbench
```

An input may be a run, task, validation method directory, or its `success/`
directory. The saved `.state/plan.json`, sequence records under `refolds/`, and
the selected refold files must remain available. A loose PDB/CIF or a success
CSV alone cannot establish eligibility. Generated files and `result.csv` need
not exist.

The selector reads each named passed decision's `passing_prediction_keys` and
keeps its actual complete **complex** predictions. A success directory can also
contain linked monomers or predictions that did not individually pass; these
are not automatically selected. Completion-only `unfiltered` decisions do not
count. With no `--upstream-criteria`, all saved named passing criteria contribute
to the selected union. Duplicate nominations of the same prediction are scored
once; different sequence/model slots remain distinct.

Use repeatable `--task`, `--method`, and `--upstream-criteria` to restrict the
selection. `--limit N` limits concrete refolds per task, after selection and
deduplication; the report records that the run is limited.

Standard protein–protein and peptide–protein inputs are supported, including
linear peptides and head-to-tail cyclic peptides. Monomers, nucleic acids,
ligands, nonstandard residues, mixed protein/peptide binder assemblies, and
ambiguous partner roles are excluded or reported with a precise error.
Saved peptide modality or subtype metadata identifies peptide tasks; a real
head-to-tail bond in a binder refold also identifies its peptide path
automatically. Chain length does not decide whether a task is a peptide.
All declared binder chains form one assembly;
antibody heavy and light chains belong together against the antigen. Metrics
use the actual saved redesigned sequence, not the original generation sequence.
CIF polymer order follows its authoritative `label_seq_id`, including reordered
atom rows; public atom identities and input coordinates are retained. PDB files
with ambiguous, non-increasing author residue locators require an explicit CIF
mapping rather than an inferred chain order.

Head-to-tail closure must be an explicit single bond from the last residue's
carbonyl C to the first residue's N in the actual success refold. The command
does not infer that bond from a short distance or `cyclic_offset`. For an intact
standard peptide chain, Rosetta removes the free-terminal variants, restores
that chemical bond, and maintains its geometry during FastRelax with native
cutpoint overlap constraints. It checks the closed graph and final junction
geometry before publishing the relaxed CIF. Source refold coordinates and
connections remain untouched. No cyclic-specific CLI option is required.
If saved `input_metadata.cyclic_chains` declares a physical closure but the
actual refold omits it, postfilter rejects that historical refold rather than
relaxing it as a linear peptide. The check applies to declared binder and target
cycles after chain mapping. Current topology-preserving refold CIFs carry the
closure themselves; the command never reconstructs a missing bond from offsets
or original generated coordinates.

`--plan` lists candidate identities, reasons for exclusions, and missing tools.
It performs no native scoring and creates no postfilter output. The common
package startup may initialize its normal cache directories.

## Dependencies and execution settings

Both Linux x86-64 environment definitions, `partidesign_cu12.yaml` and
`partidesign_cu13.yaml`, install the native postfilter dependencies. Use the
same environment as the other PartiDesign commands. For a new CUDA 12 install,
run from the checkout root:

```bash
conda env create -f partidesign_cu12.yaml
conda activate partidesign
```

Use `partidesign_cu13.yaml` / `partidesign_cu13` for CUDA 13; see the
[environment guide](environments.md). Both definitions install PyRosetta
`2026.39+release.49e05e797c` from its
[official source](https://www.pyrosetta.org/downloads), DSSP `4.6.1`, and its
compatible `libmcfp 2.0.1`. The SDK pin targets CPython 3.12 on Linux x86-64.
Postfilter runs on CPU and does not require a different CUDA stack.

For manual setup in an existing compatible Python 3.12 Linux x86-64 environment,
install the same tested SDK and command-line dependencies:

```bash
conda install -c conda-forge \
  'dssp=4.6.1=np2py312h4f94bcb_0' 'libmcfp=2.0.1=h171cf75_0'
python -m pip install 'packaging==24.2' \
  'https://west.rosettacommons.org/pyrosetta/release/release/PyRosetta4.Release.python312.ubuntu.wheel/pyrosetta-2026.39%2Brelease.49e05e797c-cp312-cp312-linux_x86_64.whl'
python -m pip check
```

These commands target the tested Python 3.12 Linux x86-64 setup. Native worker
recovery currently requires Linux process identity. PyRosetta's license terms
apply. Its SDK, database and downloaded wheel are not included in Git or the
PartiDesign wheel.

The backend also needs DAlphaBall. A source checkout provides the small Linux
x86-64 executable and full corresponding LGPL source in
`community_models/rosetta/`; see its [tool guide](../community_models/rosetta/README.md)
for rebuilding. A wheel-only installation ships the source and license; provision
the executable in the active environment or set `dalphaball` explicitly. The
two environment definitions install DSSP automatically.

Tool resolution checks an explicit config path, checkout/packaged
`community_models/rosetta/bin`, the active interpreter's `bin`, then `PATH`.
Tools and models are never downloaded implicitly while loading or scoring.

```yaml
# postfilter-local.yaml: only override fields that need changing.
workers: 1
threads: 1
timeout: 1800
dalphaball: /path/to/DAlphaBall.gcc
dssp: /path/to/mkdssp
```

```bash
partidesign postfilter --show-config
partidesign postfilter inference/RUN --config postfilter-local.yaml
```

The complete defaults are in `configs/postfilter.yaml`. Every native candidate
runs in an isolated CPU process with an explicit timeout and thread budget.
`workers` controls parallel candidates within each task; tasks are processed in
order. This command does not allocate a GPU.

The pinned BC1 protocol uses `beta_nov16`, one FastRelax repeat with
200 maximum minimizer iterations, movable backbone/sidechains, fixed rigid-body
jumps, and repacking without sequence design. It does not inherit PartiDesign
motif or fixed-coordinate constraints. BC1's own soft starting-coordinate
option is recorded explicitly; coordinates can move during relaxation.

## Automatic criteria

With no `--criteria`, protein binders require both `bc1_protein_structural` and
`bc2_protein_geometry` to pass for the same concrete refold. Peptide binders
automatically use `bc1_peptide_structural`. A run containing separate protein
and peptide tasks selects the appropriate default for each candidate. Rules
are stored in `configs/postfilter_presets.yaml`.

### Protein rules

| BC1 protein structural metric | Default rule | Unit |
| --- | --- | --- |
| Binder energy score | ≤ 0 | Rosetta energy units |
| Surface hydrophobicity | ≤ 0.35 | LayerSelector residue fraction |
| Shape complementarity | ≥ 0.55 | dimensionless |
| Interface dG | ≤ 0 | Rosetta energy units |
| Interface dSASA | ≥ 1 | Å², IAM interface area |
| Binder interface residues | ≥ 7 | count |
| Interface hydrogen bonds | ≥ 3 | count |
| Buried unsatisfied hydrogen bonds | ≤ 4 | count |
| Binder loop content | ≤ 90 | percent, input-refold DSSP |

`bc1_protein_structural_relaxed` explicitly selects the upstream protein
profile's more permissive thresholds. Its name changes thresholds, not the
relaxation protocol. The default BC2 geometry rules are CA-only interchain
clashes ≤ 0 at 2.5 Å and binder interface residues ≥ 7 at 4 Å.

```bash
partidesign postfilter inference/RUN \
  --criteria bc1_protein_structural_relaxed \
  --criteria bc2_protein_geometry
```

Repeat `--criteria` to require every selected preset. `--presets FILE.yaml`
selects a custom preset file using the shipped `presets → methods/reduction/rules`
format. Unknown names, metrics, methods, operators, and scopes are rejected
before native work. Missing, NaN, or infinite required metrics produce
`unavailable` and cannot pass.

### Peptide rules

`bc1_peptide_structural` projects the per-prediction structural rules from
BindCraft's pinned [peptide filter profile](https://github.com/martinpacesa/BindCraft/blob/7713aa0d0d351e4117a8befeb8541f3a8ebd3368/settings_filters/peptide_filters.json).
It uses the same metric definitions and relax protocol as the protein profile.

| BC1 peptide structural metric | Default rule | Unit |
| --- | --- | --- |
| Binder energy score | ≤ 0 | Rosetta energy units |
| Surface hydrophobicity | ≤ 0.50 | LayerSelector residue fraction |
| Shape complementarity | ≥ 0.50 | dimensionless |
| Interface dG | ≤ 0 | Rosetta energy units |
| Interface dSASA | ≥ 1 | Å², IAM interface area |
| Binder interface residues | ≥ 4 | count |
| Interface hydrogen bonds | ≥ 1 | count |
| Buried unsatisfied hydrogen bonds | ≤ 3 | count |
| Binder loop content | ≤ 90 | percent, input-refold DSSP |

`bc1_peptide_structural_relaxed` projects the pinned
[relaxed peptide profile](https://github.com/martinpacesa/BindCraft/blob/7713aa0d0d351e4117a8befeb8541f3a8ebd3368/settings_filters/peptide_relaxed_filters.json):
binder energy ≤ 0, surface hydrophobicity ≤ 0.50, dG ≤ 0, dSASA ≥ 1, and binder
interface residues ≥ 2. Upstream SC, hydrogen-bond, unsatisfied-hydrogen-bond,
and loop thresholds are unset in that profile and do not become required rules.
The relaxed profile changes thresholds, not the relaxation protocol.

```bash
# Peptide rules are selected automatically from saved roles and real topology.
partidesign postfilter inference/PEPTIDE_RUN/TASK/validate/af2_ig/success
# Select the upstream relaxed peptide profile explicitly when desired.
partidesign postfilter inference/PEPTIDE_RUN --criteria bc1_peptide_structural_relaxed
```

Explicit `--criteria` overrides automatic defaults for every selected candidate.
Peptide tasks do not automatically inherit the BC2 protein interface-size gate.

These are structural projections of the pinned upstream rules, not complete
BindCraft pipelines. AF confidence, trajectory-reference RMSD, binder-alone
predictions, and multi-prediction averages remain upstream responsibilities.
BC1's protein single-structure SC threshold is 0.55 and its peptide threshold
is 0.50; Average SC thresholds are not substituted here. Average-only amino-acid rules are not turned into per-refold
defaults.

## Metric stages and interpretation

BC1 Rosetta metrics use the relaxed refold. BC1 loop percentage uses DSSP on
the input success refold. BC2 coordinate and sequence metrics also use that
input refold, matching its pre-extra-relax evaluation stage. This staging is
recorded explicitly and does not combine the best values of different models.

For two or three chains, buried unsatisfied hydrogen bonds use the pinned BC1
XML filter. For four or more chains, the pinned SDK's XML filter rejects the
assembly. This command uses its native atomwise calculator on complete bound
and separated partner assemblies, counting heavy atoms unsatisfied only in the
bound state. The full chain and chemical context is retained; unavailable
calculator output cannot pass a required rule.

BC1 interface floats and DSSP percentages use the upstream two-decimal
comparison values. BC2 compares raw finite values, without display rounding.
BC1 and BC2 names have separate `bc1_` and `bc2_` prefixes.

Additional BC2 reports include atom37 interchain clashes, interface composition,
binder-side buried area, surface hydrophobicity, chain breaks, secondary
structure fractions, exposed loops/termini, and standard sequence estimates.
BC2 hydrophobicity uses relative SASA and the isolated binder assembly; it is
not BC1's LayerSelector value. BC2 buried area is binder-side and cannot replace
IAM dSASA. The CA/atom37 clash counts are geometric counts, not a chemically
complete clash score. Sequence mass, pI, and charge follow BC2's concatenated
assembly convention; they are not exact masses of linked chemical species.
For a head-to-tail cyclic binder, free-terminal exposure and the linear
free-terminal mass, pI, and charge estimates are unavailable. Their values
are not replaced with zeros or used to pass a required rule.
Metrics requiring additional property inputs or prediction states are not
fabricated and do not become default gates.

## Outputs and recovery

Each task receives a separate `postfilter/` directory:

```text
postfilter/
  metrics.csv
  success.csv
  summary.json
  resolved_config.json
  records/
  structures/
  .state/
```

`metrics.csv` has one row per selected complex refold, with its original
method/sample/sequence/prediction identity, both metric families, and separate
preset statuses/reasons. `summary.json` separates criterion nominations,
deduplicated predictions, non-PPI exclusions, valid predictions before the limit,
limited-out predictions, and file/metadata diagnostic events. It also reports
completed relaxation, complete metrics, each preset, and the joint success count.
These are concrete prediction counts; error events are labelled separately.
`success.csv` lists only candidates passing all selected
presets and points to their actual relaxed structures. Threshold failure keeps
the structure and all measured values. It is distinct from execution failure
or an unavailable required metric.

Input refolds, validation decisions, generated `result.csv`, reward values and
rankings remain unchanged. `--out DIR` chooses an independent output directory
when exactly one task is selected. Retain the owned records and private native
geometry state with the run; these preserve the atom state needed for exact
metric recovery without repeating relaxation.

```bash
# Ordinary rerun resumes complete, unchanged candidates and repairs missing work.
partidesign postfilter inference/RUN
# New thresholds: use saved metrics without native model work.
partidesign postfilter inference/RUN --presets updated-presets.yaml --refilter
# Changed metric settings: reuse completed geometry, without another relax.
partidesign postfilter inference/RUN --recompute-metrics
# Changed source/roles/relax settings: explicitly replace only owned outputs.
partidesign postfilter inference/RUN --rerun
```

Maintenance modes are mutually exclusive. Reuse the same execution overlay when
resuming a configured run. Changes to input, partner roles or relax settings
require explicit `--rerun`; changed metric settings require explicit
`--recompute-metrics`. The old plan is compared before replacement. A withdrawn
upstream success is removed from the current success report. An empty legitimate
success set produces empty tables and zero counts without native work.

For relocated validation data, `path_mappings` can explicitly map an old absolute
task root to its new absolute root. Preserve relative ownership and saved
identities; the selector does not search for similarly named files. Historical
records cannot prove that old coordinates were never changed since upstream
validation; new postfilter state checks apply from selection onward.

Exit status 0 means the requested work completed, even if every candidate fails
quality thresholds. Status 1 signals a configuration or run-level error; status
2 signals incomplete candidate execution or unavailable required metrics.
Reports use atomic replacement, with publication status under `.state/`;
interrupted work or report publication requires a resume before consuming success output.

Current clustering consumes generated designs. Direct clustering of these
postfilter relaxed refolds is not included in this command.

## Upstream definitions and licenses

The reference definitions are pinned to
[BC1 7713aa0](https://github.com/martinpacesa/BindCraft/tree/7713aa0d0d351e4117a8befeb8541f3a8ebd3368)
and [BC2 b6fb3ae](https://github.com/PacesaLab/BindCraft2/tree/b6fb3ae15ab6a702f1156cffc11bcc3323c2fd06).
BC1 is MIT; BC2's adapted definitions carry its hosting-restricted source license.
See the separate notices in `licenses/`. This command does not imply official
association with either project.
