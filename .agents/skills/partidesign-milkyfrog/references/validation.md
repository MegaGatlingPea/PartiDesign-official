# Validate saved PartiDesign designs

Use this reference for method/source/criteria choices, AF3 MSA policy, chemistry
limits, result interpretation, and reuse. Read the active checkout's
`docs/validate.md` for detailed flags, profiles, and custom metric rules; inspect
`configs/criteria.yaml` for exact preset thresholds. Paths here are relative to
the detected checkout, not this skill's installation directory.

## Establish the evaluation request

Validation takes existing **run or task directories** containing `result.csv`
and its saved CIF/PDB files, not a task YAML. Keep `resolved_config.yaml`, source
inputs, and result metadata available: chemistry, constraints, and planned
sample counts can depend on them. Several input paths and repeatable `--task`
selectors are supported.

Choose method, sequence sources, and passing criteria independently. Preserve
the user's choices; a predictor name does not define a sequence budget or a
quality threshold. Start with a plan matching those choices. For a bounded
original-sequence protein-binder example:

```bash
partidesign validate inference/first_run --task protein_fixture \
  --method esmfold2-fast --source gt:1 --criteria af3_basic --limit 2 --plan
```

`--plan` inspects inventory/config without loading models, searching databases,
or writing validation results. Inspect input errors and
`sequence_sources_by_sample`. It is not a full chemistry/backend compatibility
test. `--limit 2` caps result rows per task, including failed rows; it does not
promise two successful predictions. Use the requested scope rather than
silently adding a limit. Remove `--plan` for authorized execution after assets
and compatibility are checked.

## Select methods and assets

| Method | Use and main limits | Asset target |
| --- | --- | --- |
| `esmfold2-fast`, `esmfold2-standard` | Protein/RNA/DNA and ligand-containing refolding; check component/chemistry support | Matching method name |
| `af2_ig` | One standard-protein design chain with protein context; matched complex and isolated-chain predictions | `af2` |
| `af3`, `af3-la` | Multimodal complex prediction with the selected MSA policy | Matching method name |
| `partiscore` | Interface scoring on original saved structures; no refolding/redesign | Git-supplied `partiscore` |
| `vina_dock` | CPU chemical evaluation/docking of eligible refined pocket-ligand designs; runs alone | Environment Vina/Meeko/RDKit |

With no method/profile, runtime defaults to **`esmfold2-fast`**. A `--config`
profile can change this: `configs/validate.yaml` explicitly selects `af3`.
Repeated `--method` runs independent methods; `vina_dock` cannot be combined.
Explicit method/source/criteria flags replace the corresponding profile lists.

Prepare and check only required assets; e.g. `partidesign download
esmfold2-fast` then `partidesign check --model esmfold2-fast` when installation
is requested. Sequence-redesign download targets are `proteinmpnn`,
`solublempnn`, `ligandmpnn`, and `nampnn`; CyclicMPNN is Git-supplied. Downloads
do not prepare full AF3 MSA databases. Respect the selected model root and
upstream model access terms. For structure directories lacking `result.csv`,
use standalone `partidesign score` as described in [workflows.md](workflows.md).

## Make the sequence budget explicit

`gt:1` means original generated identities, reported as branch `self`; it does
not mean experimental ground truth. Each MPNN draw redesigns identities on the
generated backbone before refolding.

| Source | Meaning |
| --- | --- |
| `gt:1` | One original-identity slot |
| `proteinmpnn:N`, `solublempnn:N` | Protein redesign with the named model |
| `ligandmpnn:N` | Protein redesign with molecular context |
| `cyclicmpnn:N` | Cyclic peptide redesign |
| `nampnn:N` | RNA/DNA redesign |
| Any MPNN name with `-fi:N` | Retain existing interface identities during that redesign |

Counts are positive integers. Original `gt:1` is always retained, including
when an explicit source list names only MPNN. Underscore aliases are accepted;
there is no public `auto` source spelling.

Without an explicit list, fold methods use original + one SolubleMPNN draw for
protein/peptide designs, original + one NA-MPNN draw for RNA/DNA designs, and
original only when no polymer design modality is resolved, including ligand-only
inputs. Standalone PartiScore and Vina use original only. A source list inside a
profile is explicit and replaces these defaults: `configs/validate.yaml` names
SolubleMPNN, so do not assume that profile will substitute NA-MPNN for RNA/DNA.

For example, `--source gt:1 --source proteinmpnn:2` requests three slots per
backbone. `--source gt:1` alone avoids extra redesign for a requested
original-only evaluation. Inspect the plan rather than inferring costs from
the number of generated structures.

The `-fi` suffix preserves identities at design residues whose heavy atoms lie
within 6 Å of selected molecular partners; it does not freeze coordinates.
Existing identity/motif constraints remain protected. Profiles may customize
the cutoff/partners. NA-MPNN uses polymer context and does **not** directly
encode small-molecule ligands. For ligand-binding RNA/DNA, `nampnn-fi` can retain
generated ligand-contacting identities, but does not make NA-MPNN ligand-aware.

## Choose and interpret passing criteria

Without criteria, ordinary validation collects metrics, leaves `success.csv`
without data rows, and publishes no passing candidates. Completion alone is
not a quality screen. Explicit `unfiltered_success: true` profiles report
completed evaluations without thresholds and cannot combine with named criteria.

Common starting presets, subject to input compatibility:

| Request | Preset |
| --- | --- |
| Protein binder with ESMFold2/AF3 | `af3_basic` or `af3_strict` |
| Protein binder or eligible peptide with AF2 | `af2_ig_protein` or `af2_ig_peptide` |
| Protein or RNA/DNA monomer | `af3_protein_monomer` or `af3_na_monomer` |
| Joint protein–nucleic-acid geometry | `af3_protein_na_complex` |
| Protein or RNA/DNA binding a ligand | `af3_protein_bind_ligand` or `af3_na_bind_ligand` |
| Antibody with antigen context | `af3_antibody` |
| Atomic motif scaffolding | `af3_ame` |
| Refined ligand generation in a protein pocket | `odesign_denovo_ligand` |

Despite its name, `af3_basic` also supports both ESMFold2 variants. Read exact
rules before selecting a preset; missing required metrics are `unavailable`,
not a pass. Multiple `--criteria` are an **OR**: a row enters `success.csv` if
at least one screen passes. Conditions that must all hold belong in one preset.
For `any_prediction`, all its rules must hold in the same complete prediction.
`--presets FILE` replaces the preset catalog; follow the custom-rule examples
in `docs/validate.md` rather than inventing metric names or reductions.

Preserve metric units and alignment semantics: normalized confidence metrics
use [0, 1], RMSD/angstrom PAE use Å, and `ipae_normalized` differs from
`ipae_angstrom`. Joint-, self-, and partner-aligned RMSDs answer different
questions. PartiScore (`--score`) adds learned interface scores on original and
successful refolds, requires its weights, and does not replace criteria.

## AF3 MSA and resource policy

AF3/AF3-LA defaults are **`--msa auto`**, templates enabled, maximum template
date **2021-09-30**. Searches target fixed protein/RNA context chains; designed
chains stay query-only and DNA has no MSA search. Generation coordinates are
not automatically prediction templates.

- `auto` reuses exact matching cache entries and searches misses with prepared
  databases/HMMER. Check those resources before execution.
- `cache-only` requires exact ready entries and never fills misses.
- `off` performs no MSA/template searches. Choose it explicitly for a requested
  search-free evaluator trial; do not silently replace a requested MSA policy.

For a search-free AF3 plan, use the first example with `--method af3 --msa off`.
`partidesign check --model af3 --msa off` checks AF3 weights/chemistry without
MSA databases; it does not change later validation's defaults. Read
`docs/validate.md` for database/cache paths, search CPU limits, and templates.
`--msa-profile` needs `cache-only`; `--msa-refresh` needs `auto`. A changed
cache/profile for saved predictions requires an intentional refold/full rerun.

GPU methods use visible CUDA devices. Fractional `--worker-gpus` reserves
scheduling capacity, not a memory cap. Account for concurrent resident models
and additional MSA CPUs when setting a requested resource budget.

## Check chemistry compatibility

- AF2 requires exactly one standard-protein design chain. It can evaluate a
  standard-residue head-to-tail cyclic design using cyclic offsets, but rejects
  modified design residues and unsupported crosslinks. Paired VH/VL cannot use
  that adapter.
- ESMFold2 input preparation retains resolvable fixed modifications and explicit
  cyclic, disulfide, cross-chain, and ligand bonds; backend component support
  still applies. Fixed modified identities must remain resolvable.
- AF3 retains supported modifications/ligand covalent links but omits explicit
  polymer–polymer links, including disulfides. Protein head-to-tail intent uses
  cyclic offsets. Refolding therefore does not independently verify every
  requested peptide crosslink; inspect generated/refined bond geometry too.
- Antibody antigen metrics need antigen context. Preserve VH/VL/VHH and mutable
  CDR metadata; use the shipped antibody validation profile when appropriate.
- `vina_dock` needs a connected free designed ligand with authoritative SMILES,
  fixed protein receptor context, a usable source pocket ligand, task
  `postprocess: true`/`override: true`, and explicit successful OpenMM status.
  Failed refinement, covalent ligands, and nucleic-acid receptors are ineligible.
  It runs alone on CPU with original only, without `--score` or
  `--recompute-metrics`. The shipped criterion requires RDKit validity, docking
  success, QED > 0.25, and Vina < −8.18 kcal/mol. This evaluates generated pocket
  ligands; it is not the general fixed-identity docking workflow.

## Read reports and reuse work

Each task writes `validate/<method>/metrics.csv`, `success.csv`, `summary.json`,
`refolds/`, `success/`, logs, and `.state/`. Metrics have one row per sequence
slot. `success.csv` is the passing subset and may have multiple rows per
generated backbone; `input_file` identifies the original generated structure,
while prediction columns identify refolds. `success/` contains links.

Inspect `status`, `failure_stage`, `error`, and criterion status/reasons.
Distinguish unavailable/incompatible work from a completed threshold failure.
`summary.json` separates planned, completed, passing-sequence, and
passing-backbone counts; missing generation results are not silently removed
from the denominator. A limited trial is not a full-run success rate. Computational
passing does not establish experimental binding or activity.

| Requested operation | Reuse and effect |
| --- | --- |
| Repeat original command | Reuse sequences/complete predictions, finish incomplete work, rebuild reports |
| `--refilter` | Apply criteria to saved metrics; no models or new metrics |
| `--recompute-metrics` | Recalculate available metrics from saved structures/confidence; no new samples or PartiScore rerun |
| `--rerun-refold` | AF3/AF3-LA only: reuse all requested saved sequence slots and replace predictions; no missing MPNN generation |
| `--rerun` | Regenerate selected sequence/prediction slots from original design inputs |

Keep source order/settings consistent for resume; intentionally changed sequence
generation settings need a full rerun. Keep `refolds/` and `.state/` for reuse.
Refiltering needs the saved plan and per-sequence records in those directories;
`metrics.csv` alone is insufficient. Use the existing run/task directory as input.
Specify the intended method: each method has separate reports. Rerun flags
replace selected validation outputs, not original generated models.

Refilter/recompute use the saved plan; `--limit` does not narrow it. Select tasks
explicitly. `--plan`, `--refilter`, and `--recompute-metrics` are mutually
exclusive. Refilter/recompute cannot accompany `--rerun` or `--score`;
`--rerun-refold` cannot accompany plan/refilter/recompute or full rerun.
`--msa-refresh` cannot accompany plan/refilter/recompute. Use `--refilter` for
new Vina thresholds or `--rerun` to dock again. After applying criteria, route
passing original polymer backbones to clustering in [workflows.md](workflows.md).
