# `partidesign validate`

[Command guide](commands.md) · [Task YAML](task_schema.md) · [Installation](../README.md#installation)

Validate saved designs, compare refolded structures with the generated models, and select candidates
using explicit computational criteria. Validation starts from an existing **run or task directory
containing `result.csv` and its CIF/PDB files**; it does not take a task YAML as its positional
input.

## Start with one original sequence

Run from the repository root in the `partidesign` environment. The commands below assume that
`inference/first_run/protein_fixture/` already contains design results.

```bash
partidesign validate inference/first_run \
  --method esmfold2-fast --source gt:1 --limit 2 --plan

CUDA_VISIBLE_DEVICES=0 partidesign validate inference/first_run \
  --method esmfold2-fast --source gt:1 --limit 2
```

`--plan` reads the input inventory and resolved configuration without loading models, searching
databases, or writing validation results. It is a lightweight plan, not a complete chemistry or
backend compatibility check. Inspect missing input errors, the selected method, and
`sequence_sources_by_sample` before running. `--limit 2` selects at most two result rows **per
task**, including failed rows; it does not promise two successful predictions. Remove it for the
complete run.

A run path selects its task directories; a task path selects that task directly. Repeat `--task
NAME` to select tasks, or supply several run/task paths together. A bare run name can also resolve
under `inference/`. Keep the original task snapshot (`resolved_config.yaml`), input structures, and
result metadata available: ligand chemistry, constraints, and planned sample counts can depend on
them.

These first commands collect metrics. To publish passing candidates, add criteria as described under
[Success criteria](#success-criteria).

## Choose a method and prepare its assets

| `--method` | Suitable inputs and behavior | Model asset target |
| --- | --- | --- |
| `esmfold2-fast` | Protein, RNA, DNA, and ligand-containing structures; query-only fast refolding | `esmfold2-fast` |
| `esmfold2-standard` | The same input modalities with standard weights and shared fixed protein-context MSA; no templates | `esmfold2-standard` |
| `af2_ig` | One standard protein design chain with protein context; predicts both complex and isolated design chain | `af2` |
| `mol1` | Protein, RNA, DNA, and ligand-containing structures; shared context MSA/template cache | `mol1` |
| `af3` | Protein, RNA, DNA, and ligand-containing structures; complex prediction by default | `af3` |
| `af3-la` | The AF3-LA variant, with its own weights and the same validation interface | `af3-la` |
| `partiscore` | Interface scoring of saved original structures; no refolding or sequence redesign | `download partiscore` |
| `vina_dock` | CPU docking and chemical evaluation of eligible, refined small-molecule designs in protein pockets | Vina/Meeko/RDKit from the environment |

With no `--method` or profile, the command selects **`esmfold2-fast`**. A profile passed through
`--config` can change the method: the shipped
[`configs/validate.yaml`](../configs/validate.yaml), for example, selects `af3`.
Explicit method/source/criteria flags override the corresponding profile lists; they do not append
to those lists. Repeat `--method` for independent method runs, except that `vina_dock` must run
alone. This standalone command retains separate reports and success directories
per method. Use [pipeline multi-method final filtering](pipeline.md#combine-final-filters)
for a shared final pool and one union success directory.

Prepare only the backends you need, observing their upstream access terms:

```bash
partidesign download esmfold2-fast
partidesign check --model esmfold2-fast
partidesign download solublempnn   # Needed by default protein sequence redesign
partidesign check --model solublempnn
```

For other workflows, use the asset target in the table. ProteinMPNN, LigandMPNN, and NA-MPNN use the
download targets `proteinmpnn`, `ligandmpnn`, and `nampnn`; CyclicMPNN is supplied with the clone.
Add `--model-dir MODEL_ROOT`, or set `PARTIDESIGN_MODEL_DIR`, when using a shared model store. See
[Download](download.md) and [Check](check.md) for asset setup and checks.

### Antibodies, cyclic peptides, and modified chemistry

Antibody inference and peptide inference across the supported topologies are available. Choose
validation according to the actual chemistry and evaluator:

- Antibody result metadata identifies VH/VL or VHH, mutable CDR positions, and
  antigen context. The shipped [antibody profile](../configs/benchmarks/antibody/validate.yaml)
  evaluates the original sequence with AF3 and `af3_antibody` criteria. Antigen
  metrics require antigen context; a paired VH/VL design cannot use the AF2
  adapter, which requires exactly one design chain.
- AF2 supports a standard-residue head-to-tail cyclic design through cyclic
  offsets. It rejects modified design residues and unsupported crosslinks;
  it is not a general evaluator for stapled or multiply linked peptides.
- ESMFold2 input preparation retains resolvable fixed modifications and explicit
  cyclic, disulfide, cross-chain, and ligand bonds. Backend chemistry and component
  support still apply. Fixed modified residues must retain a resolvable identity.
- AF3 retains supported modifications and ligand-related covalent links, but its
  input adapter omits explicit **polymer–polymer links**, including disulfides.
  Protein head-to-tail cyclic intent is represented through cyclic offsets.
  An AF3 refold therefore does not independently verify every requested peptide
  crosslink. Inspect the generated/refined structure and bond geometry as well.
- Mol-1 passes explicit terminal C–N, disulfide SG–SG and other supported links
  through its native covalent-bond input, using the same bond endpoint rules as
  ESMFold2. It keeps upstream positional encoding without an added cyclic-offset
  transform; `cyclic_offset` alone does not declare a closure bond. Explicit
  inter-residue links support single or unspecified bond orders. Input support does not
  guarantee final bond geometry.

## Choose sequence sources

`gt:1` means the original generated identities on the saved design structure; “GT” here does **not**
mean an experimentally known sequence. It is reported as branch `self`. Each MPNN draw redesigns
identities on the generated backbone before the selected predictor refolds the result.

| Source spelling | Use |
| --- | --- |
| `gt:1` | Original identities; exactly one slot |
| `proteinmpnn:N` | ProteinMPNN protein redesign |
| `solublempnn:N` | SolubleMPNN protein redesign |
| `ligandmpnn:N` | Protein redesign with molecular context through LigandMPNN |
| `cyclicmpnn:N` | Cyclic peptide sequence redesign |
| `nampnn:N` | RNA/DNA sequence redesign through NA-MPNN |
| Any MPNN name with `-fi:N` | The same model, retaining identities at the existing interface |

Underscore spellings such as `protein_mpnn`, `ligand_mpnn`, and `na_mpnn` are also accepted. Counts
must be positive integers. The original `gt:1` slot is always retained, even if omitted from an
explicit MPNN list.

With **no explicit source list**, fold methods use original + one SolubleMPNN draw for
protein/peptide designs, original + one NA-MPNN draw for RNA/DNA designs, and original only when a
polymer design modality cannot be resolved (including ligand-only inputs). Standalone `partiscore`
and `vina_dock` use original only. A source list in a profile is explicit too: it replaces these
modality-based defaults.

```bash
# Three sequence slots per backbone: original + two ProteinMPNN draws.
partidesign validate inference/first_run \
  --method esmfold2-fast --source gt:1 --source proteinmpnn:2 --plan

# Two slots: original + one interface-preserving LigandMPNN draw.
partidesign validate inference/first_run \
  --method esmfold2-standard --source ligandmpnn-fi:1 --plan
```

### Fixed-interface redesign

The `-fi` suffix fixes a design residue's identity when any of its heavy atoms lies within **6 Å**
of a selected molecular partner. Coordinates are not frozen by this option. Partner chains are
selected automatically from molecular context (including ligands, excluding ions); profiles can set
a different cutoff or explicit `target_chains`. Existing fixed identities and motifs remain
protected. The selected positions and partner chains appear in the reports.

NA-MPNN uses polymer context and **does not directly encode small-molecule ligands**. For
ligand-binding RNA/DNA, unrestricted NA-MPNN redesign can discard ligand-recognition identities. The
shipped ligand-binding aptamer profile uses original + `nampnn-fi` to retain the generated
ligand-contacting identities and redesign the remaining positions. This does not make NA-MPNN
ligand-aware.

## Success criteria

**Metrics describe a result; criteria decide whether it passes.** Without named criteria, normal
validation writes metrics but leaves `success.csv` with no data rows and publishes no passing
candidates. Execution completion alone is not a structural quality screen. A profile can explicitly
opt into `unfiltered_success: true`; that reports completed evaluations without a quality threshold
and cannot be combined with named criteria.

```bash
partidesign validate inference/first_run \
  --method esmfold2-fast --source gt:1 --criteria af3_basic --limit 2 --plan
```

Remove `--plan` to evaluate and select. Despite its name, `af3_basic` supports both ESMFold2
variants as well as AF3/AF3-LA and Mol-1. Its required metrics must be present for that input. A missing
metric is `unavailable`, not a pass. Repeat `--criteria` for several screens; a row enters
`success.csv` when it passes **at least one** selected screen. Put all required conditions in a
single preset when they must hold together.

Useful shipped presets in [configs/criteria.yaml](../configs/criteria.yaml):

| Task | Preset |
| --- | --- |
| Protein binder, AF2 complex/monomer screen | `af2_ig_protein` or benchmark-specific `af2_ig_protdbench` |
| Standard/cyclic peptide with eligible AF2 chemistry | `af2_ig_peptide` |
| Protein binder, AF3/Mol-1/ESMFold2 | `af3_basic`, `af3_strict` |
| Linear/cyclic peptide binder, AF3/Mol-1/ESMFold2 | `af3_peptide` |
| Protein or RNA/DNA monomer consistency | `af3_protein_monomer`, `af3_na_monomer` |
| Joint protein–nucleic acid geometry | `af3_protein_na_complex` |
| Protein binding a small molecule | `af3_protein_bind_ligand` |
| RNA/DNA binding a small molecule | `af3_na_bind_ligand` |
| Antibody with antigen context | `af3_antibody` |
| Atomic motif scaffolding | `af3_ame` |
| Refined ligand generation in a protein pocket | `odesign_denovo_ligand` |

`af3_peptide` requires `binder_iptm > 0.8`, `complex_design_rmsd < 2.5 Å`
and `rfd4_min_ipae_angstrom < 1.5 Å`,
without a binder pTM threshold. It applies to both linear and cyclic peptides.

For stricter interface and geometry checks, choose one of these presets explicitly:

| Preset | Required checks (distances in Å) |
| --- | --- |
| `af3_ame_strict` | min iPAE <1.5; protein backbone RMSD ≤2; reference-motif RMSD ≤1.5; correct motif identities; no eligible refold polymer–ligand clash below 1.5 Å. Requires a target interface. |
| `af3_protein_bind_na_strict` | min iPAE ≤1.5; nucleic-acid-target-aligned protein RMSD ≤8. Applies to RNA and DNA context. |
| `af3_na_bind_protein_strict` | min iPAE ≤1.5; protein-target-aligned designed nucleic-acid RMSD ≤8. |
| `af3_protein_bind_ligand_strict` | min iPAE ≤1.5; binder ipTM ≥0.8; protein backbone RMSD ≤1.5; backbone-aligned ligand RMSD ≤5. |
| `af3_na_bind_ligand_strict` | min iPAE ≤1.5; designed-nucleic-acid phosphate RMSD ≤5; nucleic-acid-aligned ligand RMSD ≤5. |

`af3_alphaproteo` uses min iPAE ≤1.5, binder pTM ≥0.8, and target-backbone-aligned
binder RMSD ≤2.5. `af3_antibody` uses min iPAE ≤2 and
antigen-aligned antibody RMSD ≤2. Antibody heavy/light internal interfaces
do not replace the antigen interface.

Min iPAE takes the minimum valid PAE in each direction of a design–target chain
pair, averages the available directions, then takes the minimum across those
pairs. Context–context and design–design pairs are excluded. The strict presets
use `rfd4_min_ipae_angstrom`; `min_interchain_pae_angstrom` has the same definition.
The legacy `min_design_target_pae_angstrom` retains its original calculation.
The reverse nucleic-acid and nucleic-acid–ligand strict screens extend these
interface conventions to their respective design directions.

Partner-aligned RMSDs preserve the target fit when measuring the binder; they
differ from whole-complex Cα RMSD. With unchanged residue types, all observed
reference atoms are compared. MPNN redesign is supported: changed protein
positions compare their conserved N/CA/C/O backbone, plus OXT when present in
both structures; changed
RNA/DNA bases compare the conserved sugar-phosphate framework. Unchanged
positions keep their sidechain/base atoms. Every mapped residue still contributes,
and no coordinates are invented for new sidechain atoms. Per-prediction metric
provenance records the comparison selection and atom/residue counts.

Fixed identities, motifs, modified CCD residues and identity-dependent bond
anchors remain protected. Motif checks use the original source reference.
An unrequested ordinary terminal OXT omitted by a predictor is normalized out;
an explicitly required bond endpoint or motif atom is not omitted.
Missing required PAE, backbone atoms, correspondence, or motif evidence is
unavailable and cannot pass; an ordinary authorized MPNN mutation alone is not
an error.
For older validation results, use
`--recompute-metrics` to calculate the new metrics from retained evidence;
`--refilter` alone does not calculate them.

### Antibody contact metrics

Antibody–antigen evaluations also report `antibody_interface_residues` and
`antibody_off_paratope_contact_fraction`. A contact is any binder–antigen
heavy-atom pair at distance ≤4 Å. The first metric counts unique contacting
antibody residues across the complete VHH or VH/VL binder. The second is the
fraction of those residues outside the declared `antibody.paratope`.
VH–VL internal contacts do not count as antigen contacts. These metrics use the
refolded prediction and require native role/paratope mappings and antigen
context; absent evidence is unavailable rather than a passing value.

The optional standalone criterion `antibody_contacts` requires at least seven
contacting antibody residues and an off-paratope fraction ≤0.20. Existing
standalone criteria, including `af3_antibody`, retain their defaults. Select
`antibody_contacts` explicitly for a contact-only screen; to require confidence,
geometry and contacts together in standalone validation, put their rules in
one custom preset. Repeating `--criteria` is a union, not a conjunction.
[Antibody pipelines](pipeline.md#antibody-pipelines) automatically conjoin the
contact rules with the selected prediction's criteria, including
`criteria_mode: any`. Native `presets.antibody_contacts` overrides can change
the two thresholds without a separate contact configuration format.

### Define your own thresholds

Save the following as `binder_presets.yaml`. This reproduces the shipped `af3_basic` screen under a
custom name, so you can edit thresholds explicitly:

```yaml
presets:
  my_binder:
    methods: [af3, af3-la, mol1, esmfold2-fast, esmfold2-standard]
    reduction: any_prediction
    rules:
      - {metric: binder_iptm, op: '>', value: 0.8}
      - {metric: binder_ptm, op: '>', value: 0.8}
      - {metric: complex_design_rmsd, op: '<', value: 2.5}
```

```bash
partidesign validate inference/first_run \
  --method esmfold2-fast --source gt:1 \
  --presets binder_presets.yaml --criteria my_binder --plan
```

`--presets FILE` replaces the preset catalog; include every selected preset in that file. For
`any_prediction`, every rule must pass in the **same complete prediction**. Supported alternatives
are `best_ranked` (native 0.8 iPTM + 0.2 pTM ranking), `any_af2_pair` (one matched complex/monomer
pair), and `all_and_mean` (all configured predictions and their means). Rules accept `>`, `>=`, `<`,
`<=`, and `==`, with scopes `complex` (default), `monomer`, `paired`, or `prediction`.

RMSD and PAE metrics with Å/angstrom semantics are in Å; normalized pLDDT, pTM, and iPTM are
fractions in [0, 1]. `ipae_normalized` is distinct from `ipae_angstrom`. Vina affinity is in
kcal/mol; QED is dimensionless. Preserve the alignment choice in metric names: joint-complex RMSD,
self-aligned RMSD, and partner-aligned RMSD answer different questions.

### ipSAE interface confidence

ESMFold2-fast/standard, AF3/AF3-LA, and Mol-1 report `ipsae`,
`ipsae_design_to_context`, and `ipsae_context_to_design` in [0, 1].
`ipsae_pae_cutoff` records the PAE cutoff in Å (default 10.0), and `ipsae_pairs`
contains each chain pair's scores and availability. The calculation uses
[DunbrackLab/IPSAE](https://github.com/DunbrackLab/IPSAE) v4 d0res with PAE
values strictly below that cutoff. This cutoff controls the calculation;
an acceptance threshold belongs in a criteria rule on `ipsae`.

Scores cover design–context polymer chain pairs only: each directional score
is the maximum across eligible pairs, and `ipsae` is the maximum of the two
directions. Design–design and context–context pairs are excluded. Protein and
RNA/DNA pairs are supported, including atomized residues with identifiable
protein Cα or nucleic-acid C1′ representative atoms. Ligand/ion pairs are not
applicable. Missing PAE or an
unresolved required polymer pair leaves the overall scores unavailable with a
reason. AF2, PartiScore, and Vina do not calculate ipSAE. These metrics do not
change the shipped filters or ranking.

### Save a repeatable validation profile

A validation profile is separate from the task YAML. For example, save this as
`binder_validation.yaml` and inspect it with `--config binder_validation.yaml`:

```yaml
methods: [esmfold2-standard]
sequence:
  sources:
    - {source: gt, count: 1}
    - source: ligandmpnn-fi
      count: 1
      settings:
        temperature: 0.1
        fixed_interface: {enabled: true, cutoff: 6.0}
criteria: [af3_basic]
scorer: {enabled: false}
ipsae: {pae_cutoff: 10.0}
```

The profile requests two sequence slots per backbone. CLI `--source` and `--criteria` replace its
lists. Use the same profile and source order when resuming; use `--rerun` when intentionally
changing sequence generation settings.

## Read the results

Each task has a separate directory for each method:

```text
inference/first_run/protein_fixture/validate/esmfold2-fast/
  metrics.csv
  success.csv
  summary.json
  refolds/<sample>/             # Sequence records and predicted structures
  success/<sample>/             # Links to passing records and refold structures
  logs/validate.log
  logs/worker_*.log
  .state/                      # Saved plan and resume state
```

`metrics.csv` contains one row per **sequence slot**, not one per backbone. It includes
source/branch, sequences, completion status, errors, prediction-specific metrics, and each
criterion's status/reasons. AF2's complex, monomer, and paired metrics stay distinguishable.
`success.csv` is the passing subset, and can contain several rows for the same generated backbone.
Its structure columns point to associated predictions; the original generated structure remains at
`input_file`. `success/` contains links, rather than a second set of independent model outputs.

`summary.json` separates planned structures/sequences, completed and unavailable results, and
passing sequence/backbone counts. A passed backbone means at least one sequence slot passed. Planned
counts can come from the generation snapshot, so missing generation results are not silently dropped
from the denominator. A limited trial has a limited denominator and should not be reported as the
full run's success rate. Inspect `status`, `failure_stage`, and `error` before judging an
unsuccessful screen. Missing criteria metrics and incompatible inputs are reported separately from
an evaluated threshold failure.

Keep `refolds/` and `.state/` for reuse or recomputation. After applying criteria, use
[Cluster](cluster.md) to group the **original generated polymer backbones** selected by
`success.csv`.

## Resume, refilter, recompute, or rerun

| Operation | Reuses | Performs |
| --- | --- | --- |
| Repeat the original command | Saved sequences and complete predictions | Finishes missing/incomplete work and rebuilds reports |
| `--refilter` | Saved metrics | Applies selected criteria and rebuilds reports/links; no models |
| `--recompute-metrics` | Saved sequences, predictions, and confidence data | Recalculates available geometry/metrics and rebuilds reports; no new model samples |
| `--rerun-refold` | All requested saved sequence slots | ESMFold2-standard/AF3/AF3-LA/Mol-1 predictions again; never generates missing MPNN sequences |
| `--rerun` | Original design inputs | Regenerates requested sequence/prediction slots |

```bash
partidesign validate inference/first_run --method esmfold2-fast \
  --criteria af3_strict --refilter

partidesign validate inference/first_run --method esmfold2-fast \
  --criteria af3_basic --recompute-metrics
```

Keep the validation `.state/` plan and per-sequence records under `refolds/` when reusing results;
`metrics.csv` alone is insufficient for refiltering. Supply the existing run/task directory.
Refiltering changes selection only; it cannot add a missing metric. Recomputation needs the saved
structures and relevant confidence data and does not rerun PartiScore. Use the intended method
explicitly: reports for another method are stored separately. Rerun flags replace their selected
validation outputs, not the original generated models. Refiltering and recomputation use the saved
plan; `--limit` does not narrow that existing plan. Select tasks explicitly when needed.

Completed older outputs gain ipSAE only through an explicit `--recompute-metrics`;
ordinary resume and `--refilter` reuse saved metrics. Recomputation uses the saved
plan's ipSAE cutoff, or 10.0 Å for an older plan. To change it, include
`ipsae: {pae_cutoff: 8.0}` in a profile passed with `--config`. Saved ipSAE values
can be retained without raw PAE only when their definition, chain-pair scope, and
cutoff match. If old outputs lack both PAE and matching saved ipSAE evidence,
the score remains unavailable; recomputation does not run new predictions.

`--plan`, `--refilter`, and `--recompute-metrics` are mutually exclusive. `--rerun` and
`--rerun-refold` are mutually exclusive. Neither refiltering nor recomputation can accompany
`--rerun` or `--score`. Recomputation is unsupported for `vina_dock`.
`--rerun-refold` supports ESMFold2-standard/AF3/AF3-LA/Mol-1 and cannot accompany plan/refilter/recompute. `--msa-refresh` also cannot accompany plan/refilter/recompute.

## ESMFold2 standard settings

ESMFold2-standard uses shared fixed protein-context MSA by default. Configure it
in a validation profile, for example:

```yaml
methods: [esmfold2-standard]
esmfold2:
  msa: {mode: auto, source: colabfold}
  msa_max_depth: 1024
  msa_column_mask_rate: 0.1
  msa_subsample_at_inference: true
```

`esmfold2.msa` accepts the same settings as the shared top-level `msa` mapping.
Model-specific values override shared YAML values; common CLI options such as
`--msa` and `--msa-source` override the configured values. Templates are always
disabled for ESMFold2. Fast remains query-only even when the shared policy is
`auto`; `--msa off` gives standard its historical query-only path.

`msa_max_depth` is a positive integer or `null`; `null` removes the row limit,
subject to native feature caps. `msa_column_mask_rate` is finite and between 0
and 1, and `msa_subsample_at_inference` is boolean. Each prediction's existing
`confidence.msa` metadata reports its actual MSA features and effective depth
after native selection and featurization. This uses the ordinary prediction
evidence, without a separate output directory.

## Mol-1 predictor settings

Mol-1 uses the same `validate` command, sequence sources, criteria, reports and
resume interface as the other folding backends:

```bash
partidesign download mol1
partidesign check --model mol1 --msa off
partidesign validate inference/first_run --method mol1 --source gt:1 \
  --criteria af3_basic --msa off --limit 2 --plan
```

Remove `--plan` to predict. For repeatable settings, place the native predictor
group in a validation profile passed through `--config`:

```yaml
methods: [mol1]
sequence:
  sources: [{source: gt, count: 1}]
criteria: [af3_basic]
mol1:
  samples: 5
  recycles: 20
  steps: 200
  seed: 101
  dtype: bf16
  msa: {mode: auto}
```

Standalone Mol-1 validation defaults to one sample. When Mol-1 is explicitly
selected for pipeline final validation, its default remains five samples and
filters the highest native `ranking_score` sample. Ordinary pipeline final
validation instead uses ESMFold2-standard. `mode` accepts
`complex`, `monomer`, or `both`. `dtype` accepts `bf16`, `fp32`, or `fp16`.
`triangle_multiplicative` and `triangle_attention` accept `auto`, `torch`, or
`cuequivariance`; `auto` uses available kernels. An explicit `checkpoint` path
overrides the default `community_models/mol1/checkpoint/mol-1-base.pt`.
These settings belong under `mol1`; pipeline uses `final.mol1` or
`iteration.refold.mol1` with `method: mol1`.

## Shared MSA, templates, and worker resources

ESMFold2-standard, AF3, AF3-LA and Mol-1 share one context MSA/template cache. The defaults are
`--msa auto`, `--msa-source colabfold`, and maximum template release date
**2021-09-30**. ESMFold2-standard searches only fixed protein context and consumes
its MSA without templates. AF3/AF3-LA/Mol-1 use protein templates by default;
`--msa-templates false` disables their template work. `--msa off` disables both
alignments and templates. Designed chains remain query-only, including original
GT sequences, MPNN sequences, peptides, and antibody VHH/VH/VL chains.

ColabFold does not search RNA or DNA. Select `--msa-source af3` to search fixed
RNA context for AF3/AF3-LA/Mol-1 with the existing local tools. ESMFold2 keeps
RNA and DNA query-only with either source. RNA/DNA have no structural templates.
Generation coordinates are not automatically prediction templates.

```bash
# Preview the default ColabFold route.
partidesign validate inference/first_run --method esmfold2-standard --source gt:1 \
  --criteria af3_basic --msa-source colabfold --limit 2 --plan

# Use existing local AF3 search resources instead.
partidesign validate inference/first_run --method af3 --source gt:1 \
  --criteria af3_basic --msa-source af3 \
  --msa-database-dir /path/to/af3_databases --msa-binaries-dir /path/to/hmmer/bin \
  --msa-template-date 2021-09-30 --plan
```

Remove `--plan` to run. `auto` reuses a compatible ready entry and searches
missing queries; `cache-only` requires cached results; `off` uses query-only
input. `--msa-profile` selects a recorded profile with `cache-only`.
`--msa-refresh` requires `auto` and publishes a new generation. When new MSA/template
evidence would change saved prediction inputs, use `--rerun-refold` to keep saved
sequences, or `--rerun` to regenerate them too.

The shared cache remains `<model_dir>/.cache/af3_msa/` with the existing
source/profile/generation references. A compatible template-on generation can
supply ESMFold2's template-off view while preserving templates for AF3/Mol-1.
No separate ESMFold2 search cache is created.

See [MSA and templates](msa.md) for pre-search commands, shared workflow settings,
cache paths, source-specific resources and template handling. Mol-1 consumes the
same normalized evidence as AF3 and does not run a separate online search.

GPU methods use visible CUDA devices; select them with `CUDA_VISIBLE_DEVICES`. Per-worker defaults
are 0.25 GPU for AF2, 0.5 GPU for AF3/AF3-LA, and 1 GPU for Mol-1/ESMFold2/PartiScore, with 1 CPU per
worker. Fractions reserve scheduling capacity, not a memory limit. Increase `--worker-gpus` if
concurrent resident models exceed available memory. MSA search CPUs are additional; `--msa-workers`,
`--msa-cpus`, `--jackhmmer-cpus`, `--nhmmer-cpus`, and `--msa-timeout` bound the search resources.

## Short recipes for other tasks

```bash
# RNA/DNA binding protein: original plus one NA-MPNN redesign, AF3 geometry.
partidesign validate inference/aptamer_run --method af3 --msa off \
  --source gt:1 --source nampnn:1 --criteria af3_protein_na_complex --plan

# Ligand-binding RNA/DNA: retain the generated interface during redesign.
partidesign validate inference/ligand_aptamer_run --method af3 --msa off \
  --source gt:1 --source nampnn-fi:1 --criteria af3_na_bind_ligand --plan

# Add PartiScore on original structures and successful refolds.
partidesign validate inference/first_run --method esmfold2-fast \
  --source gt:1 --criteria af3_basic --score --plan

# Score saved original structures without a structure predictor.
partidesign validate inference/first_run --method partiscore --source gt:1 --plan

# Chemical validity, QED, and CPU Vina docking of refined pocket ligands.
partidesign validate inference/ligand_run --method vina_dock \
  --source gt:1 --criteria odesign_denovo_ligand --plan
```

`--score` uses PartiScore, an MDN network for molecular interface scoring, in addition to refolding.
It requires the PartiScore checkpoint and does not replace the selected criteria. The standalone
`partiscore` method accepts original only; use `--score` with a refold method when evaluating
redesigned sequences. For structure directories without `result.csv`, use [Score](score.md) instead.

`vina_dock` requires a designed, connected free ligand with authoritative SMILES, fixed protein
receptor context, and a usable source pocket ligand. The task must have `postprocess: true`,
`override: true`, and an explicit successful OpenMM result in `result.csv`. Failed refinement,
covalently attached ligands, and nucleic-acid receptors do not satisfy this workflow. It runs on
CPU, alone, with original only; neither `--score` nor `--recompute-metrics` is supported. Use
`--refilter` for new thresholds or `--rerun` to dock again. The shipped ligand criterion requires
RDKit validity, docking success, QED > 0.25, and Vina < −8.18 kcal/mol. This evaluates generated
pocket ligands; it is not the general fixed-identity docking workflow.

## Optional protein-complex postfiltering

After named validation criteria pass, [postfilter](postfilter.md) can relax and
screen successful protein–protein and standard peptide–protein complexes on
CPU, including eligible antibody–antigen and physical head-to-tail cyclic
peptide assemblies. Retain the published success records and their complete
refold files for the handoff. Saved plans provide partner metadata for older
records when needed. Nucleic acids, ligands, and nonstandard residues are excluded.

For standalone validation, postfilter selects the saved named criteria's passing
complex predictions. For [pipeline](pipeline.md) results, it selects only the
published final representative and verifies final acceptance evidence, including
both PPI model winners. Pass the pipeline run, task, `final/`, or `final/success/`.

```bash
partidesign postfilter inference/first_run --task protein_fixture --plan
```

Remove `--plan` only for requested execution. Postfiltering adds no new folding
predictions and preserves upstream results. It is independent of
[clustering](cluster.md), which compares original generated backbones.

[Back to the command guide](commands.md)
