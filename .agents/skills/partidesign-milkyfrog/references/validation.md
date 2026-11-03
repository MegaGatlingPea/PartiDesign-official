# Validate saved PartiDesign designs

Use this reference for method/source/criteria choices, shared AF3/Mol-1 MSA policy, chemistry
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
| `mol1` | Multimodal prediction with shared context MSA/template cache | `mol1` |
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
do not prepare full MSA search databases. Respect the selected model root and
upstream model access terms. For structure directories lacking `result.csv`,
use standalone `partidesign score` for scoring as described in
[workflows.md](workflows.md), or pipeline with annotation.csv for sequence/structure
optimization as described in [partial-pipeline.md](partial-pipeline.md). Do not
invent result.csv rows as an input workaround.

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
| Protein binder with ESMFold2/AF3/Mol-1 | `af3_basic` or `af3_strict` |
| Linear/cyclic peptide binder with ESMFold2/AF3/Mol-1 | `af3_peptide` |
| Protein binder or eligible peptide with AF2 | `af2_ig_protein` or `af2_ig_peptide` |
| Protein or RNA/DNA monomer | `af3_protein_monomer` or `af3_na_monomer` |
| Joint protein–nucleic-acid geometry | `af3_protein_na_complex` |
| Protein or RNA/DNA binding a ligand | `af3_protein_bind_ligand` or `af3_na_bind_ligand` |
| Antibody with antigen context | `af3_antibody` |
| Atomic motif scaffolding | `af3_ame` |
| Refined ligand generation in a protein pocket | `odesign_denovo_ligand` |

`af3_peptide` requires `binder_iptm > 0.8`, `complex_design_rmsd < 2.5 Å`
and `rfd4_min_ipae_angstrom < 1.5 Å`,
without a binder pTM threshold. It applies to both linear and cyclic peptides.

Despite its name, `af3_basic` also supports both ESMFold2 variants and Mol-1. Read exact
rules before selecting a preset; missing required metrics are `unavailable`,
not a pass. Multiple `--criteria` are an **OR**: a row enters `success.csv` if
at least one screen passes. Conditions that must all hold belong in one preset.
For `any_prediction`, all its rules must hold in the same complete prediction.
`--presets FILE` replaces the preset catalog; follow the custom-rule examples
in `docs/validate.md` rather than inventing metric names or reductions.

Optional strict screens are `af3_ame_strict`, `af3_protein_bind_na_strict`,
`af3_na_bind_protein_strict`, `af3_protein_bind_ligand_strict`, and
`af3_na_bind_ligand_strict`. They include min iPAE: directional design–target PAE
minima are averaged per chain pair, then reduced to the minimum across eligible
pairs. Do not include design–design or context–context interfaces. The updated
`af3_alphaproteo` and `af3_antibody` also use this convention and target-aligned
binder partner-aligned geometry. Read the preset table in `docs/validate.md` for exact
thresholds and the reverse nucleic-acid extensions. Strict AME clash checks use
the refold, exclude directly bonded and truly coordinate-conditioned pairs,
and use distance <1.5 Å. Fixed residue identity alone is not a coordinate condition.
Legacy nonstrict ligand/NA/AME screens remain available. Older saved validation
requires `--recompute-metrics` for the new fields; refiltering does not create
missing evidence.

MPNN residue changes are supported in these partner-aligned metrics. Unchanged
residues use all observed reference atoms. Authorized changed protein positions
use N/CA/C/O and OXT present on both sides; changed RNA/DNA bases use sugar-phosphate atoms.
Do not discard the whole mutated residue or invent new sidechain coordinates.
Record selection and comparison counts in metric provenance. Same-sequence
comparisons remain all-atom. Fixed/motif/modified identities, context and
identity-dependent bond anchors must still match; missing required backbone
geometry is unavailable, while a normal permitted sequence change is not.
Normalize an unrequested ordinary terminal OXT absent from a predictor output;
retain explicit motif/chemical endpoint requirements.

Preserve metric units and alignment semantics: normalized confidence metrics
use [0, 1], RMSD/angstrom PAE use Å, and `ipae_normalized` differs from
`ipae_angstrom`. Joint-, self-, and partner-aligned RMSDs answer different
questions. PartiScore (`--score`) adds learned interface scores on original and
successful refolds, requires its weights, and does not replace criteria.

ESMFold2-fast/standard, AF3/AF3-LA, and Mol-1 also report `ipsae`,
`ipsae_design_to_context`, and `ipsae_context_to_design` in [0, 1], plus
`ipsae_pae_cutoff` (Å) and per-pair `ipsae_pairs`. The default profile setting is
`ipsae: {pae_cutoff: 10.0}`. Each directional score is the maximum over
design–context polymer pairs; `ipsae` is the maximum of the two directions.
Exclude design–design/context–context pairs. Protein and RNA/DNA, including
atomized residues with identifiable protein Cα or nucleic-acid C1′
representatives, are supported;
ligand/ion pairs are not applicable. Missing required PAE leaves scores
unavailable. AF2/PartiScore/Vina do not calculate ipSAE, and shipped filtering
and ranking remain unchanged.

## Antibody contact metrics

Preserve the native role, source-to-output and intended paratope mappings.
`antibody_interface_residues` counts unique antibody residues contacting antigen
through heavy atoms at ≤4 Å. `antibody_off_paratope_contact_fraction` is the
fraction outside `antibody.paratope`; omit VH–VL internal contacts. Omitted
paratope defaults to all CDRs and no framework independently of identity design
and soft template visibility. Missing mappings/context are unavailable evidence.

Standalone `antibody_contacts` is an optional screen with ≥7 interface residues
and ≤0.20 off-paratope fraction; `af3_antibody` and other existing standalone
defaults are unchanged. Multiple standalone criteria are OR, so combine contact,
confidence and geometry rules within one native preset when all are required.
Every antibody pipeline adds the contact rules on the same prediction even with
`criteria_mode: any`. The antibody pipeline profile uses ESMFold2-fast iteration,
ESMFold2-standard-only final with five GT samples, 0.8 ipTM + 0.2 pTM best selection before filtering,
RFD4 min iPAE and target-aligned binder RMSD each ≤3 Å during iteration and
≤2 Å at final validation.
Use native `presets.antibody_contacts` overrides for custom contact thresholds;
see `<repo>/docs/validate.md#antibody-contact-metrics`.

## Mol-1 and shared MSA policy

Use `partidesign download mol1` and `partidesign check --model mol1 --msa off`
for setup. The asset layout is `community_models/mol1/{source,checkpoint,common}`;
source is bundled, parameters and common resources are downloaded. External
model roots move checkpoint/common assets, not installed source. Check the
upstream parameter terms separately from the source-code license.

`--method mol1` uses the existing validation commands, sequence sources, criteria,
reports and resume behavior. Native profile settings belong under `mol1`:
`samples: 1`, `recycles: 20`, `steps: 200`, `seed: 101`, `dtype: bf16` by default.
It supports `mode: complex|monomer|both`. Pipeline protein/nuc final requests
five samples and filters the highest native ranking_score prediction. Explicit
`checkpoint` overrides the default checkpoint path. See docs/validate.md for
kernel and precision controls.

Standalone repeated `--method` runs retain separate method reports/success
folders. Pipeline `final.methods` instead uses one shared candidate pool and
one union success directory; read partial-pipeline.md for that contract.

AF3/AF3-LA/Mol-1 defaults are `--msa auto`, `--msa-source colabfold`, and maximum
protein-template release date `2021-09-30`. ColabFold searches fixed protein
context, providing MSA and templates together; it does not search nucleic acids.
`source: af3` uses local databases for fixed protein/RNA context. Structural
templates are protein-only. All designed chains, including antibody VH/VL/VHH,
remain query-only. `off` disables both MSA and templates; do not expose a separate
required template toggle. Generation and partial-reference coordinates do not
become prediction templates.

Use the same task selection as design/pipeline to prepare fixed-context inputs:

```bash
partidesign msa configs/examples/pipeline_tasks.yaml \
  --task protdbench_insulin_receptor --source colabfold --plan
```

Remove `--plan` for the authorized search. The command does not generate designs
or load prediction models. Its shared cache is
`<model_dir>/.cache/af3_msa/`, retaining source/profile/generation identity.
Different sources are not interchangeable. Pipeline `msa.csv` records which
entries its predictions used; do not duplicate the cache under every model/run.
A ready generation can be used by both AF3 and Mol-1.

`auto` searches misses, `cache-only` requires ready entries without searching,
and `off` stays query-only. `--msa-profile` needs `cache-only`; independent
`partidesign msa --refresh` and prediction `--msa-refresh` create a new generation.
Changed prediction evidence requires an intentional refold/full rerun. The
ColabFold route uses its service and HHsearch from `colabfold-binaries`; the local
AF3 route needs its existing databases/HMMER tools. Public service requests are
serialized; never substitute another source after a failure. See `docs/msa.md`
for commands, endpoints, cache paths, date filtering and search resources.

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
- Mol-1 uses native covalent-bond input for explicit terminal C–N, disulfide SG–SG
  and other supported links, with ESMFold2's bond endpoint rules. It keeps upstream
  positional encoding without an added cyclic-offset transform; `cyclic_offset`
  alone does not declare a closure bond. Explicit inter-residue links support single or
  unspecified bond orders. Input support does not guarantee final bond geometry.
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
| `--rerun-refold` | AF3/AF3-LA/Mol-1: reuse all requested saved sequence slots and replace predictions; no missing MPNN generation |
| `--rerun` | Regenerate selected sequence/prediction slots from original design inputs |

Keep source order/settings consistent for resume; intentionally changed sequence
generation settings need a full rerun. Keep `refolds/` and `.state/` for reuse.
Refiltering needs the saved plan and per-sequence records in those directories;
`metrics.csv` alone is insufficient. Use the existing run/task directory as input.
Specify the intended method: each method has separate reports. Rerun flags
replace selected validation outputs, not original generated models.

Older completed results need explicit `--recompute-metrics` to add ipSAE.
Recompute keeps the saved cutoff (default 10.0 Å for older plans) unless the
`--config` profile explicitly supplies `ipsae: {pae_cutoff: ...}`. Without raw
PAE, only saved ipSAE with the same definition, chain-pair scope, and cutoff can
survive recomputation; absent evidence or a changed cutoff stays unavailable.
This operation does not create new predictions.

Refilter/recompute use the saved plan; `--limit` does not narrow it. Select tasks
explicitly. `--plan`, `--refilter`, and `--recompute-metrics` are mutually
exclusive. Refilter/recompute cannot accompany `--rerun` or `--score`;
`--rerun-refold` cannot accompany plan/refilter/recompute or full rerun.
`--msa-refresh` cannot accompany plan/refilter/recompute. Use `--refilter` for
new Vina thresholds or `--rerun` to dock again.

## Hand off passing results

Keep the saved named decisions, `.state/` plan, sequence records, and refold
files when handing eligible protein–protein complexes, including
antibody–antigen assemblies, or standard peptide–protein complexes to optional
[postfiltering](postfilter.md) with persistent Ray CPU actors.
Postfilter selects the union of actual complete predictions passing saved named
upstream criteria; a success link or CSV alone is insufficient. Use
`partidesign postfilter inference/RUN --plan` for metadata/tool checks before
requested execution. Scientific defaults and thresholds live in
`configs/postfilter_presets.yaml`; runtime resources/tools use
`configs/postfilter.yaml`. Complete samples resume together with their relaxed
CIFs, while older output formats require a fresh `--out`. Linear and physical
head-to-tail cyclic peptides are supported automatically, while nonstandard
residues, nucleic acids, and ligands remain unsupported. Upstream results are
preserved.

Independently, route passing original generated polymer backbones to clustering
in [workflows.md](workflows.md). Postfilter relaxed refolds do not replace
those clustering inputs. Start neither additional step unless requested.
