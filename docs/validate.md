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
| `esmfold2-fast` | Protein, RNA, DNA, and ligand-containing structures; fast refolding | `esmfold2-fast` |
| `esmfold2-standard` | The same input modalities with standard ESMFold2 weights | `esmfold2-standard` |
| `af2_ig` | One standard protein design chain with protein context; predicts both complex and isolated design chain | `af2` |
| `af3` | Protein, RNA, DNA, and ligand-containing structures; complex prediction by default | `af3` |
| `af3-la` | The AF3-LA variant, with its own weights and the same validation interface | `af3-la` |
| `partiscore` | Interface scoring of saved original structures; no refolding or sequence redesign | Git-supplied `partiscore` |
| `vina_dock` | CPU docking and chemical evaluation of eligible, refined small-molecule designs in protein pockets | Vina/Meeko/RDKit from the environment |

With no `--method` or profile, the command selects **`esmfold2-fast`**. A profile passed through
`--config` can change the method: the shipped
[`configs/validate.yaml`](../configs/validate.yaml), for example, selects `af3`.
Explicit method/source/criteria flags override the corresponding profile lists; they do not append
to those lists. Repeat `--method` for independent method runs, except that `vina_dock` must run
alone.

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
variants as well as AF3/AF3-LA. Its required metrics must be present for that input. A missing
metric is `unavailable`, not a pass. Repeat `--criteria` for several screens; a row enters
`success.csv` when it passes **at least one** selected screen. Put all required conditions in a
single preset when they must hold together.

Useful shipped presets in [configs/criteria.yaml](../configs/criteria.yaml):

| Task | Preset |
| --- | --- |
| Protein binder, AF2 complex/monomer screen | `af2_ig_protein` or benchmark-specific `af2_ig_protdbench` |
| Standard/cyclic peptide with eligible AF2 chemistry | `af2_ig_peptide` |
| Protein binder, AF3/ESMFold2 | `af3_basic`, `af3_strict` |
| Protein or RNA/DNA monomer consistency | `af3_protein_monomer`, `af3_na_monomer` |
| Joint protein–nucleic acid geometry | `af3_protein_na_complex` |
| Protein binding a small molecule | `af3_protein_bind_ligand` |
| RNA/DNA binding a small molecule | `af3_na_bind_ligand` |
| Antibody with antigen context | `af3_antibody` |
| Atomic motif scaffolding | `af3_ame` |
| Refined ligand generation in a protein pocket | `odesign_denovo_ligand` |

### Define your own thresholds

Save the following as `binder_presets.yaml`. This reproduces the shipped `af3_basic` screen under a
custom name, so you can edit thresholds explicitly:

```yaml
presets:
  my_binder:
    methods: [af3, af3-la, esmfold2-fast, esmfold2-standard]
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
| `--rerun-refold` | All requested saved sequence slots | AF3/AF3-LA predictions again; never generates missing MPNN sequences |
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

`--plan`, `--refilter`, and `--recompute-metrics` are mutually exclusive. `--rerun` and
`--rerun-refold` are mutually exclusive. Neither refiltering nor recomputation can accompany
`--rerun` or `--score`. Recomputation is unsupported for `vina_dock`.
`--rerun-refold` is AF3/AF3-LA only and cannot accompany plan/refilter/recompute. `--msa-refresh` also cannot accompany plan/refilter/recompute.

## AF3 MSA, templates, and worker resources

The current AF3/AF3-LA defaults are **`--msa auto`**, templates enabled, and maximum template date
**2021-09-30**. MSA/template searches target fixed protein/RNA context chains; designed chains
remain query-only, and DNA has no MSA search. Generation coordinates are not automatically
prediction templates.

| Policy | Behavior |
| --- | --- |
| `--msa off` | No MSA/template searches; useful for a first evaluator trial |
| `--msa auto` | Reuse matching cache entries; search missing entries using prepared databases and HMMER binaries |
| `--msa cache-only` | Require exact ready cache entries; never search to fill a miss |

```bash
# A bounded AF3 trial without database searches.
partidesign validate inference/first_run --method af3 \
  --source gt:1 --criteria af3_basic --msa off --limit 2 --plan

# Database-backed evaluation; choose paths containing your prepared resources.
partidesign validate inference/first_run --method af3 --source gt:1 \
  --criteria af3_basic --msa auto --msa-cache-dir msa_cache \
  --msa-database-dir /path/to/af3_databases --msa-binaries-dir /path/to/hmmer/bin \
  --msa-templates true --msa-template-date 2021-09-30 \
  --msa-workers 2 --msa-cpus 16 --worker-gpus 0.5 --worker-cpus 1 --plan
```

Remove `--plan` after checking the resources. `partidesign download af3` prepares model assets, not
the full search databases. `--msa-profile` selects a recorded cache profile and requires
`--msa cache-only`. `--msa-refresh` requires `--msa auto` and refreshes matching cache
generations. If an existing prediction used a different MSA/cache profile
or generation, the runtime requires an explicit rerun decision; add `--rerun-refold` to retain
sequences and replace AF3 predictions, or `--rerun` to regenerate sequences too.

GPU methods use visible CUDA devices; select them with `CUDA_VISIBLE_DEVICES`. Per-worker defaults
are 0.25 GPU for AF2, 0.5 GPU for AF3/AF3-LA, and 1 GPU for ESMFold2/PartiScore, with 1 CPU per
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
