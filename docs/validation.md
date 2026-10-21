# Validation

Validation consumes the structures and `result.csv` written by a design run. It is
invoked separately so generation and evaluation can be repeated with different
methods or criteria. Run it from the repository root:

```bash
cd /path/to/partidesign-inference
partidesign validate RUN_OR_TASK_PATH [RUN_OR_TASK_PATH ...] [options]
```

An explicit run/task path may be supplied. Default generation outputs are under
`./inference/RUN_NAME`; pass that path to validation. Run names under
`inference/` are also accepted. Use
`--task NAME` to select task leaves.

## Notebook validation

In **6 Runtime** of [the notebook](../partidesign.ipynb), choose an existing
run/task directory containing actual results. **Prepare validation** writes a
native configuration and command without starting validation. **Execute** and
**Modify command** default off: copy the selectable read-only command for manual
use, or enable Execute and explicitly click **Execute command**. Editing or
preparing a command never launches it automatically.

The notebook defaults to **AF3**, with **Context MSA = Auto / reuse cache** and
**Use templates** enabled; native cache defaults remain in effect. Original
sequence 1 is mandatory, while ordinary and fixed-interface MPNN branches have
independent counts and temperatures. Prediction methods use direct checkboxes; AF2 uses model 1. Success criteria
are selected from project or custom presets. **No criteria** opts into the
complete-only unfiltered cohort described below. Prepare and Execute capture
the visible settings together, including quick selection or command edits.

## Cluster successful designs

Use the independent [cluster command](clustering.md) after validation:

```bash
partidesign cluster RUN_OR_TASK_PATH --plan
partidesign cluster RUN_OR_TASK_PATH
```

Each validation method and each passing criterion is clustered separately.
The success table selects original generated backbones; multiple successful
MPNN/refold records for one backbone do not become extra clustering samples.

## Methods and sources

The current method names are:

| Method | Use |
| --- | --- |
| `af2_ig` | AF2 initial-guess complex/monomer comparison for applicable protein binders. |
| `esmfold2-fast` | Fast ESMFold2 refolding and structure metrics. |
| `esmfold2-standard` | Standard ESMFold2 refolding and structure metrics. |
| `af3` | AF3 complex, monomer, or paired (`both`) prediction and metrics. |
| `af3-la` | AF3-LA complex, monomer, or paired (`both`) prediction and metrics. |
| `partiscore` | Score existing GT coordinates; no sequence generation. |
| `vina_dock` | Vina/QED evaluation for eligible postprocessed de novo ligand designs. |

When neither the config nor CLI supplies sources, fold validation uses one
original sequence (`gt:1`) plus one redesign sequence selected per input:
protein and cyclic-protein designs use SolubleMPNN (`solublempnn:1`,
`v_48_020`, temperature `0.1`), while RNA/DNA designs use NA-MPNN
(`nampnn:1`). Ligand-only designs retain GT only. Standalone `partiscore`
and `vina_dock` also remain GT-only because they consume existing coordinates.
Explicit benchmark source lists take precedence. Use `--source gt:1` to validate
only the original sequence, or an explicit source list for a different redesign
protocol.

Configured validation sequence sources are ordered:
`gt`, `protein_mpnn`, `soluble_mpnn`, `ligand_mpnn`, `cyclic_mpnn`, and `na_mpnn`.
Compact aliases (`proteinmpnn`, `solublempnn`, `ligandmpnn`, `cyclicmpnn`, and
`nampnn`) are accepted. Each MPNN source also has a `-fi` fixed-interface form.
`auto` is a reward-only source and is rejected by validation.

The validation parser inserts exactly one `gt`/`self` slot when the requested list
contains only MPNN sources. An explicit `gt:1` requests the same original slot;
`gt:2` is invalid:

```bash
CUDA_VISIBLE_DEVICES=0 partidesign validate ./inference/guide_protein \
  --method esmfold2-fast \
  --source gt:1 \
  --source proteinmpnn:2 \
  --criteria af3_basic
```

`--source` order defines the sequence slots after the parser's original `self`
slot. MPNN backend
settings, including the canonical `rm_aa` (default `C` for every backend), remain in
the selected validation config;
the CLI source list replaces the configured source list itself. See
[MPNN](mpnn.md) for the precedence between `rm_aa`, legacy `omit_aas`, backend
blocks, and source-local settings.

## AF3 native MSA search

Use standard AF3 with the checked-in profile:

~~~bash
partidesign validate ./inference/my_run \
  --config configs/validate_af3_msa.yaml \
  --method af3
~~~

The profile uses the shared cache at .cache/af3_msa, two concurrent search
workers, seven HMMER threads per process, a four-hour search timeout, and
native AF3 templates through 2021-09-30. Set --msa-database-dir and
--msa-binaries-dir when automatic discovery is unavailable. CLI values take
precedence over the selected YAML profile; the same flags are available through
python -m src.validate.

Ordinary validation YAMLs show `af3.samples`, `recycles`, and `seed`, plus
`msa.mode`, `search_workers`, `templates`, and `max_template_date`. Runtime
defaults supply the shared cache, automatic tool/database discovery, CPU budget,
Triton attention, and token buckets. Buckets pad native inputs for compilation;
they do not crop the structure. Specify advanced options only to override them.
Omitting `af3.timeout_seconds` retains the 1800-second limit for starting a
resident worker or waiting for one prediction, including feature preparation
and compilation. Omitting `af3.msa.timeout_seconds` retains the 14400-second
limit for one uncached sequence search, including templates. These limits do
not cap an entire validation run, and omitting them does not disable timeouts.

--msa auto first reuses exact cached target/context queries and searches only
cache misses. --msa cache-only uses the same automatic cache matching and
fails for a missing target entry. Both modes can reuse a copied cache without
the original database, HMMER installation, or machine paths. --msa off keeps
the MSA-free behavior and requires no MSA cache or search resources.

Cache matching uses the exact modality/sequence and compatible search and
protein-template settings. Database/tool paths, file modification times and
installation versions are retained as provenance, not cache-hit requirements.
An explicit database_revision remains a constraint. When several compatible
published entries exist, the newest readable valid entry wins; a corrupt entry
can fall back to an older valid one. Protein template enable/date changes remain
separate requests; RNA matching ignores protein-only template settings.

The default cache root is .cache/af3_msa under the selected model root. After
search writers finish, copy the complete cache directory, including its SQLite
index, profiles, protein and RNA generation directories, to another machine.
Alignment and template files use relative paths inside the cache; no original
database files need to accompany them. A cache relocation alone does not
invalidate saved AF3 prediction identity.

--msa-profile optionally pins a recorded profile in cache-only mode; it is not
required for offline reuse or when multiple profiles exist. --msa-refresh with
--msa auto deliberately searches again and publishes a new generation, so it
requires the search databases and tools. Resource checks report available
compatible cached queries; exact coverage is checked for each validation input.
Fresh searches require the selected database and HMMER resources. HMMER is
included in partidesign.yaml from Bioconda, and automatic discovery prefers the
current Python environment's complete HMMER tool set after explicit overrides.

The validation profile leaves sequence sources unset, so validation retains the existing modality-aware original-plus-one-MPNN
default. Benchmark source counts and temperatures remain explicitly configured.

Protein and RNA context chains are searchable. Binder and designed chains,
including antibody or fixed-interface binder chains, remain query-only; DNA,
ligand, ion, and CCD-ligand chains do not use native AF3 MSA search. Each
homomer copy receives its own native chain mapping even when it shares one
cache entry. Cached output stores compact query/profile/generation references;
native AF3 confidence data needed for metric recomputation remains available.

Search is staged outside the resident GPU workers, so prepared samples can
search on CPU while hot AF3 workers refold ready samples. --msa-workers,
--msa-cpus, --jackhmmer-cpus, --nhmmer-cpus, --msa-timeout,
--msa-templates, and --msa-template-date tune the run without changing
the selected AF3 weights. --plan reports known roles and saved queries but
performs no search, cache write, MPNN generation, model load, or output write;
future unsampled MPNN queries are reported as unknown.

To regenerate only AF3/AF3-LA refolds while reusing saved sequence slots, use:

~~~bash
partidesign validate ./inference/my_run \
  --method af3 --msa cache-only --rerun-refold
~~~

--rerun-refold is mutually exclusive with --rerun and with
--plan, --refilter, or --recompute-metrics. --msa-refresh is also
rejected by those no-search maintenance modes; it reuses saved sequence
slots and does not redraw MPNN sequences. A read-only resource check
uses the same policy without starting search or GPU work:

~~~bash
partidesign check --model af3 --msa auto \
  --msa-database-dir /mnt/software/AlphaFold3/alphafold3/public_databases \
  --msa-binaries-dir /mnt/software/miniforge3/envs/partidesign/bin
~~~

## Ligand chemistry and identity

AF3 reuses a native CCD component only when the complete named heavy-atom
inventory, elements, bonds and every explicit source formal charge agree.
CCD-supported aromatic and Kekule representations are compared using the
CCD's concrete bond orders; a different known bond order is still rejected.
Explicit SMILES and trusted molecules retain their charges, stereochemistry
and source atom names through named user CCDs and structure restoration.
AF3 prepare and prediction also carry explicitly authored context SMILES from
the resolved task snapshot; design-chain sequence planning remains separate.
Each request's user CCD overlays the shared native CCD without mutating the
cached base or another request's definitions. The native pickle and resident
model caches remain reusable.

The reader preserves explicit PDB and CIF formal charges. Unknown charges
remain unknown, while an explicit zero remains a constraint. Conflicting
explicit charge columns are errors. Trusted SMILES and CCD construction
supply their own formal charges, and structure export retains explicit values.
These are chemistry annotations, not additional model input channels.

A source component ID represents one chemical definition. Repeated occurrences
can share a native or user CCD when their definitions agree. Mixed native and
custom definitions, or repeated trusted definitions with different charge or
stereochemistry, are rejected rather than replacing one occurrence with another.
Named-CCD atom recovery uses the actual native atom names; compiler ordinal
names are retained only as legacy mapping metadata.

## Criteria and presets

`--criteria NAME` selects a named criterion from `configs/criteria.yaml` or a
selected benchmark profile. With no selected criteria and the native default
`unfiltered_success: false`, validation collects metrics and leaves
`success.csv` empty. The preset YAML contains only `methods`, `reduction`, and
`rules`; rules contain `metric`, `op`, `value`, and an explicit `scope` where
needed. Sources, units, and protocol interpretations are documented here rather
than repeated as executable configuration.

The notebook explicitly writes `criteria: []` and `unfiltered_success: true`
when its criteria field is empty. This opt-in publishes the `unfiltered`
success cohort only for complete records with valid prediction state and all
expected predictions, metrics, and required artifacts completed. Failed,
incomplete, or malformed records never count as success. This cohort applies
no scientific quality threshold; it does not change the native metrics-only
default.

`unfiltered_success` is a strict boolean, and `true` requires empty criteria.
The notebook disables the flag when named criteria are selected. CLI
`--criteria NAME` also disables it when reusing a notebook configuration,
including with `--refilter`. Refiltering rebuilds decisions for named criteria
from saved records without retaining the unfiltered cohort or loading
prediction models.

A rule without `scope` uses `complex`. `monomer` selects monomer metrics,
`paired` selects the comparison for the same AF2 model/sample, and `prediction`
allows either complex or monomer structure protocols. Each required value must
be a finite number. An absent, null, nonnumeric, NaN, or infinite required metric
is unavailable and never passes; the configured comparison operator also applies
at the exact threshold. Method/criterion incompatibility is `not_applicable`.
All requested prediction and metric stages must be complete before filtering.

| Reduction | Protocol |
| --- | --- |
| `any_prediction` | Every rule must pass together on one prediction. At least one passing prediction succeeds; metrics from different predictions are never combined. If none passes, a missing required metric makes the decision unavailable rather than successful. |
| `all_and_mean` | Every configured prediction and each scope's metric mean must pass. Missing values are not averaged around. |
| `any_af2_pair` | At least one same-model/sample complex, monomer, and paired-comparison group must pass every rule. Metrics from different pairs are never combined. |
| `best_ranked` | Select the prediction with the highest native `0.8 * iptm + 0.2 * ptm`, then test all its rules. Equal ranking values use the prediction key for a stable tie break. Every candidate needs finite native `iptm` and `ptm`. |

Sequence/prediction counts and MPNN/backend temperatures come from the selected
benchmark's `validate.yaml`; presets supply no global count or temperature.
Decisions are per backbone. Benchmark success rates use all planned backbones
as the denominator, with incomplete or unavailable entries excluded from the
success numerator. These computational screens do not establish experimental
binding or calibrated affinity.

Metric units are shared across presets:

| Unit | Rule metrics |
| --- | --- |
| Fraction | `plddt`, `plddt_total`, `ptm`, `iptm`, `binder_ptm`, `binder_iptm`, `correct_motif_sequence`, `qed`. Confidence is stored as a fraction rather than a 0–100 score; binder confidence retains the selected backend's native definition. |
| PAE / 31 | `ipae_normalized`; this is the existing normalized AF2 PAE metric. |
| Å | All RMSD metrics; `ipae_angstrom`, `pae_interaction`, `min_interchain_pae_angstrom`, `min_design_target_pae_angstrom`, `min_target_design_pae_angstrom`, and `antibody_antigen_pae_angstrom`. AF3/ESMFold2 minimum-PAE gates use native Å values rather than scaled normalized PAE. |
| Boolean 0/1 | `has_ligand_clashes`, `rdkit_valid`, `docking_success`. |
| kcal/mol | `vina_dock`, the best successfully docked affinity. |

The four AF2 criteria retain their existing definitions and thresholds:

| Criterion | Source and computational protocol |
| --- | --- |
| `af2_ig_protein` | [BindCraft default filters](https://github.com/martinpacesa/BindCraft/blob/main/settings_filters/default_filters.json), using the AF2 confidence subset. Complex `plddt` is the binder-residue mean. Align the paired monomer to the AF2 complex binder on CA, then measure all-heavy RMSD at most 3.5 Å. Every configured prediction and the mean must pass. |
| `af2_ig_peptide` | [BindCraft peptide filters](https://github.com/martinpacesa/BindCraft/blob/main/settings_filters/peptide_filters.json), with the same paired CA alignment and all-heavy RMSD at most 2.5 Å. Linear, head-tail cyclic, and disulfide peptides share this filter; AF2 treats disulfide designs as cysteine sequences. Every configured prediction and the mean must pass. |
| `af2_ig_protdbench` | [ProtDBench AF2-IG-Easy evaluation](https://github.com/congliuUvA/ProtDBench/blob/6cd7ac9df0621d0950b6e6bc65eb5f6e0d392bfd/protdbench/protd_configs/eval.py), using paired binder CA-only RMSD below 3.5 Å. Local model 1 corresponds to upstream `model_id=0`. Values retain full precision; upstream rounds before filtering. At least one same-model/sample pair must pass every rule. |
| `af2_ig_alphaproteo` | [AlphaProteo Appendix S2.2](https://arxiv.org/html/2409.08022v1#A2.SS2.p7) optimized AF2-initial-guess filter: total complex pLDDT above 90, bidirectional binder–target PAE below 7 Å, and binder-aligned CA RMSD to the generated design below 1.5 Å. The rules pass together on one complete prediction; multi-prediction OR is PartiDesign policy. |

`af2_ig_alphaproteo` follows the
[AF2ig implementation cited by the paper](https://github.com/nrbennet/dl_binder_design/blob/main/af2_initial_guess/predict.py).
Its `plddt_total` uses all complex residues, while AF2 `plddt` remains binder-only.
Its `pae_interaction` averages binder-to-target and target-to-binder PAE; native
`ipae_angstrom` remains separate. The binder CA RMSD uses the binder chain in the
complex prediction and the generated binder as reference, following the
[cited alignment implementation](https://github.com/nrbennet/dl_binder_design/blob/main/af2_initial_guess/af2_util.py).
The paper's S2.1 original benchmark uses different thresholds (pLDDT above 80,
PAE below 10 Å, RMSD below 1 Å). Older AF2 results without the two paper-defined
confidence scalars are unavailable for the optimized filter; binder-only
confidence is not substituted. New predictions preserve those scalars for
filtering and metric-only recomputation.

All `af3_*` presets support `af3`, `af3-la`, `esmfold2-fast`, and
`esmfold2-standard`. Their names identify filtering families; the method still
determines the refold model and native confidence. Transferring a threshold to
another backend does not establish the source model's calibration.

| Criterion | Source and computational protocol |
| --- | --- |
| `af3_basic`, `af3_strict` | [PXDesignBench evaluation](https://github.com/bytedance/PXDesignBench/blob/master/pxdbench/pxd_configs/eval.py), following PTX-basic/PTX-strict thresholds. Native binder confidence and generated/refolded complex protein CA RMSD. |
| `af3_alphaproteo` | [AlphaProteo Appendix S2.2](https://arxiv.org/html/2409.08022v1#A2.SS2.p7): minimum across all interchain PAE entries below 1.5 Å, binder pTM above 0.8, and complex CA RMSD below 2.5 Å, using the native best-ranked prediction. The paper used target templates and five diffusion samples; actual run settings remain explicit in the validation config. |
| `af3_ame` | `Proteina-Complexa/src/proteinfoundation/result_analysis/motif_binder_analysis_utils.py:DEFAULT_MOTIF_LIGAND_BINDER_SUCCESS`. Shared AME backbone, actual-reference motif, motif identity, and clash gate. Motif coordinates are compared to the supplied reference rather than substituted with generated coordinates. Clash uses Complexa's 1.5 Å distance check; there is no added ligand-pose threshold. |
| `af3_protein_monomer` | [La-Proteina](https://arxiv.org/html/2507.09466). User-selected N/CA/C backbone self-consistency below 2 Å is a project proxy for the published all-atom gate, not an exact reproduction. The rule is prediction-scoped. |
| `af3_na_monomer` | [ODesign](https://arxiv.org/html/2510.22304). RNA/DNA C1-prime token-center self-consistency below 5 Å adapts the published C4-prime gate to this model's C1-prime center atom. The rule is prediction-scoped. |
| `af3_protein_na_complex` | **PartiDesign project-custom standard** for protein bind RNA/DNA and RNA/DNA bind protein. Collect every protein CA and every observed valid RNA/DNA P from the generated and refolded structures, perform one joint unweighted rigid fit, and require `complex_ca_p_rmsd < 5.0 Å`. AF3 is the default; AF3-LA and ESMFold2 are explicit transfers. |
| `af3_protein_bind_na` (legacy directional screen) | [RFD3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12458353/). Fit context DNA phosphate atoms and measure the full designed protein CA RMSD below 5 Å, with no PAE gate. RFD3 trims terminal loops; the project's explicit protocol retains every loop, so this is not exact published preprocessing. The four current protein–NA binder benchmarks use `af3_protein_na_complex` instead. |
| `af3_na_bind_protein` (legacy directional screen) | [ODesign](https://arxiv.org/html/2510.22304). Fit context protein CA atoms and measure designed RNA/DNA phosphate RMSD below 5 Å. The source used AF3; AF3-LA and ESMFold2 are explicit transfers. |
| `af3_ligand_iptm_screen` | [AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w). Provisional native interface-confidence screen for de novo or partial-identity ligand design. It does not evaluate recovery of a fixed-SMILES docking pose; missing native confidence is unavailable. |
| `af3_protein_bind_ligand` | `Proteina-Complexa/src/proteinfoundation/result_analysis/binder_analysis_utils.py:DEFAULT_LIGAND_BINDER_THRESHOLDS`, transferred from RF3/LigandMPNN. Minimum native token PAE over both designed-protein–ligand directions below 2 Å, generated/refolded protein CA RMSD below 2 Å, and ligand heavy-atom RMSD below 5 Å. |
| `af3_antibody` | [RFAntibody filtering suggestions](https://github.com/RosettaCommons/RFantibody/blob/3e5a5bb1cb98befde9fc049ae3ff58cabb491f67/README.md#filtering-strategies), adapted as a provisional project screen: antigen-interface mean PAE below 10 Å and antigen-aligned antibody CA RMSD below 2 Å. |
| `af3_na_bind_ligand` | **PartiDesign project-custom standard.** The same prediction must satisfy `na_p_self_rmsd < 5.0 Å` and `na_aligned_ligand_rmsd < 5.0 Å`. ODesign/RFD3 do not publish this reverse-direction NA-on-ligand protocol. |
| `odesign_denovo_ligand` | [ODesign ligand design evaluation](https://arxiv.org/html/2510.22304v2). Successful postprocessed molecule only: RDKit recognition and successful Vina Dock, then strict QED and best-affinity thresholds. No refold, PAE, SA, or uniqueness gate. This preset uses `vina_dock`, not a fold backend. |

`af3_protein_bind_ligand` uses `min_design_target_pae_angstrom`, the minimum
over both designed-protein-to-ligand and ligand-to-designed-protein token PAE
blocks. Context–context pairs are excluded. The separately reported
`min_target_design_pae_angstrom` remains directional and is not the gate.

`af3_protein_bind_ligand` fits identity-matched receptor heavy atoms excluding
OXT, with Complexa's CA fallback; the CA-only ligand RMSD remains diagnostic.
Protein CA and ligand heavy-atom RMSDs compare generated and refolded CIFs rather
than the original conditioning structure. Apply the gate separately to fixed-
ligand and joint-SMILES sampling; the transfer does not establish RF3-calibrated
success rates. The additional NA/protein minimum-PAE gate remains removed.

For `af3_antibody`, PAE is the bidirectional mean over all declared antibody
heavy/light versus antigen token pairs, excluding VH/VL. RMSD fits antigen
anchors and measures every antibody CA. RFAntibody emits whole-complex `pae`,
`interaction_pae`, and multiple RMSDs, and its README does not unambiguously
identify which fields are gated. This field selection, nonprotein-antigen support,
and backend transfer are project policy. Missing maps or confidence are
unavailable; total-complex pLDDT and internal VH/VL confidence do not replace
antigen confidence. This shared screen does not reproduce BoltzGen's
antibody-anything or BindCraft2's complete or type-specific protocols. The
[antibody reference comparison](tasks/antibody.md#boltzgen-antibody-anything-and-rfantibody-references)
records the sources, masks, ranking, and confidence normalization.

For `af3_protein_na_complex`, `complex_ca_p_rmsd` includes all protein and
nucleic-acid polymer chains in the complex and compares the actual generated
CIF against its refold. Protein contributes CA, RNA/DNA contributes P, and every
selected point has equal weight; longer chains therefore contribute more
points. It is one fit across the concatenated points, without independent
chain fitting or averaging per-chain RMSDs. A genuinely absent source 5-prime
P is excluded; missing predicted selected atoms, incomplete chain mapping,
or nonfinite coordinates leave the metric unavailable and cannot pass.
The boundary is strict: 5.0 Angstrom does not pass. One complete prediction
below the threshold passes `any_prediction`. Original, FI and ordinary
MPNN branches remain independently reported. Directional aligned RMSDs and
native confidence metrics remain available as diagnostics; no extra iPTM or
PAE condition is added. This replaces the current four benchmark defaults,
not the recorded historical ODesign/RFD3 directional protocols.

For `af3_na_bind_ligand`, `src/validate/benchmark_metrics.py` computes phosphate
self-fit RMSD for the designed nucleic acid, then ligand heavy-atom RMSD after
that same phosphate alignment, both against the generated structure. Required
phosphate and authoritative ligand-atom correspondences must exist. One
prediction passing only the NA rule and another passing only the ligand rule
cannot pass the standard. For nucleic-acid designs, generic protein RMSD
columns are inapplicable: `protein_aligned_na_p_rmsd` remains the directional NA-on-protein diagnostic,
while `na_p_self_rmsd` reports nucleic-acid self-consistency. The shipped
`configs/benchmarks/nuc_ligand_binder/validate.yaml` evaluates one original
sequence plus one fixed-interface NA-MPNN sequence (temperature `0.1`, cutoff
`6.0 Å`); it does not add ordinary NA-MPNN sequences. NA-MPNN does not consume
ligand atom features. Fixed-identity docking tasks retain no validation profile.

For a fixed context ligand with a source-resolved SMILES, metric mapping can
compare aromatic and concrete Kekule bond representations only after both
complete heavy-atom graphs are proved against that same trusted molecule.
Elements, explicit charges, stereochemistry and external attachments retain
their existing checks; SMILES does not authorize missing or extra atoms.
Generated or joint ligands without this context authority retain strict graph
matching. This validation-only mapping does not alter structure coordinates or
the dump path. After a mapping repair, use `--recompute-metrics` on the affected
tasks to update cached decisions, CSVs, success links and summaries from existing
refold structures without regenerating sequences or loading prediction models.

Completed AF3 records reuse the compact native PAE/PDE matrices and their
source-token projection when validating saved confidence. Legacy projected
matrix artifacts remain supported; neither format requires duplicating matrices
in structure dumps or regenerating a complete prediction.

The current local round has three ligands, each with RNA fixed at 30 nt and
DNA fixed at 20 nt. These lengths were selected from historical ESMFold2
results; the new round uses original AF3, with Original and FI rates reported
separately and their union also reported. This is a local targeted assessment,
not the official ODesign target collection.

The DNA-to-protein design profile uses NA-MPNN for the designed DNA sequence;
LigandMPNN redesigns protein and cannot supply that branch. MotifBench sampling
tasks remain available, while its official external evaluation owns success
thresholds. No internal MotifBench validation profile or criterion is shipped.

Protein and RNA/DNA monomers use separate profiles and explicit task selectors:

```bash
partidesign validate ./inference/monomer \
  --config configs/benchmarks/monomer/validate.yaml \
  --task protein_monomer_benchmark

partidesign validate ./inference/monomer \
  --config configs/benchmarks/monomer/validate_na.yaml \
  --task rna_monomer_benchmark --task dna_monomer_benchmark
```

The protein profile keeps original1 plus ProteinMPNN8, and the nucleic-acid
profile uses original1 plus NA-MPNN8; both retain temperature 0.1 and have no
fixed-interface redesign. Monomer rules evaluate a complete same-prediction
conjunction for either a complex or monomer structure protocol. Use CLI `--task`;
`task_names` is not an accepted validation configuration field.

Use a custom threshold file without modifying the packaged preset; RMSD values
in this example are in Å:

```yaml
presets:
  guide_protein:
    methods: [esmfold2-fast]
    reduction: any_prediction
    rules:
      - {metric: design_rmsd, op: '<', value: 2.5}
```

```bash
partidesign validate ./inference/guide_protein \
  --method esmfold2-fast \
  --source gt:1 \
  --presets configs/examples/custom_presets.yaml \
  --criteria guide_protein
```

## Plan, maintenance, and regeneration

Use `--plan` before a model run. It reads the existing task inventory and resolved
configuration, reports the selected methods/sources, and exits before model work or
new validation output:

```bash
partidesign validate ./inference/guide_protein \
  --method esmfold2-fast --source gt:1 --plan
```

The maintenance modes have distinct meanings:

```bash
# Rebuild reports from metrics already persisted by this validation run.
partidesign validate ./inference/guide_protein \
  --method esmfold2-fast --criteria guide_protein \
  --presets configs/examples/custom_presets.yaml --refilter

# Recompute metrics from saved refold structures; no refold model is loaded.
partidesign validate ./inference/guide_protein \
  --method esmfold2-fast --recompute-metrics

# Explicitly regenerate the requested sequence/prediction slots.
CUDA_VISIBLE_DEVICES=0 partidesign validate ./inference/guide_protein \
  --method esmfold2-fast --source proteinmpnn:2 --rerun
```

`--plan`, `--refilter`, and `--recompute-metrics` are mutually exclusive.
`--refilter` and `--recompute-metrics` are no-model operations; they require the
corresponding persisted validation records. `--rerun` is the explicit regeneration
switch and may load sequence or structure models. Vina metrics cannot be recomputed
from refold structures; use `--refilter` to reuse saved docking metrics or `--rerun`
to dock again.

## Method-specific constraints

Standalone `partiscore` uses GT coordinates and therefore requires a GT-only source.
To score MPNN/refold structures with PartiScore, use a refold method and add
`--score`; successful refolds are then scored in the same validation run.

`af3` and `af3-la` share the `af3` block. The default `af3.mode: complex`
predicts design and context chains together and also accepts a single chain.
`monomer` predicts only the design chains, which can include multiple chains;
`both` runs both protocols. This setting selects chains, independently of the
MSA search mode. Ordinary configs omit the default mode. Set `af3.samples`
to any positive integer. `both` plans both
protocols for every sample index, so a task produces
`complex_sample0...complex_sampleN` and `monomer_sample0...monomer_sampleN`.
These validation samples are independent prediction slots; they are different
from design reward execution, where the AF3 reward runs one sample per candidate.

`vina_dock` is CPU-only, must be the only selected method, accepts GT only, and
requires a task whose native postprocess completed successfully. It does not use
`--score` or `--recompute-metrics`.

AF2 uses the `af2` configuration block; the method name is `af2_ig`. AF3 and AF3-LA
use the shared `af3` block with its configured mode and sample count. ESMFold2
uses `esmfold2-fast` or `esmfold2-standard` and the configured fast/standard
fold settings. Missing or
inapplicable capability data is reported per prediction rather than silently
counted as a pass.

ESMFold2-fast deliberately uses `lm_mask_pct: 0.0` instead of the official
SDK's documented `0.1` default, to efficiently reuse deterministic ESM-C
embeddings for fixed protein context. This is a project-specific protocol
choice; ESMFold2-standard's official SDK default is already `0.0`.
See the [official FoldingConfig documentation](https://github.com/Biohub/esm/blob/main/esm/sdk/api.py).

`lm_mask_pct` masks protein residues before computing ESM-C embeddings.
It is separate from `lm_dropout: 0.3`, which remains enabled on LM pair
features inside the folding model. Fixed-context embedding reuse requires
zero input masking, protein context, and non-protein design chains;
protein-design requests compute their embeddings afresh. Explicit nonzero
`lm_mask_pct` disables this embedding cache without changing resident model
loading or the model cache lifetime.

## Worker lifetime and GPU ownership

GPU validation starts a Ray worker queue sized from the visible GPUs and the method's
worker fraction. Each worker owns one validation backend, stays hot while the owner
queue feeds it samples, and is closed before the validation process shuts down. A
worker queue may therefore reuse a loaded model across its samples; this does not
claim that every backend or the entire validation inferencer is resident at once.

Set `CUDA_VISIBLE_DEVICES` to choose the GPU pool. `--worker-gpus` controls the
fraction assigned to each worker and `--worker-cpus` controls its CPU allocation.
`vina_dock` bypasses the Ray GPU queue and runs on CPU.

When `--worker-gpus` is omitted, the method defaults are:

| Method | Default worker GPU fraction | JAX preallocation |
| --- | ---: | --- |
| `af2_ig` | `0.25` | `false` by default |
| `af3`, `af3-la` | `0.5` | `false` by default |
| Other GPU-backed validation methods | `1.0` | method-dependent |

The default worker CPU allocation is `1.0`. AF2, AF3, and AF3-LA Ray workers pass
`XLA_PYTHON_CLIENT_PREALLOCATE=false` unless that environment variable was already
set by the caller.

For example, this AF3 validation uses the default `0.5` worker GPU fraction and
default JAX allocation policy; add `--worker-gpus` only when you need another
fraction:

```bash
CUDA_VISIBLE_DEVICES=0,1 partidesign validate ./inference/guide_protein \
  --method af3 \
  --source gt:1
```

Normal and antibody task batches remain format-isolated during validation. A
validation command can select several homogeneous task leaves, but source,
checkpoint, and capability checks still apply to each task's saved resolved config.

## Reports

Each selected task/method writes under
`<task>/validate/<method>/`. `metrics.csv` contains per-slot metrics.
`success.csv` contains rows passing a requested criterion, or complete valid
records in the explicitly enabled unfiltered cohort. The method log records
execution and failure messages. Validation never imports a
different inference run's cached reward result as its own prediction.

See [Outputs](outputs.md) for the complete run tree and status semantics.

For new tasks with `score: true`, `result.csv` also publishes `final_partiscore`
and `score_rank` after the entire task's final scoring has finished. The score
is the existing `design_groups` total over selected contacts, rather than an
average of interface rows. Antibodies use the antigen total and retain separate
VH/VL scores. A successful relaxation uses the optimized structure's score;
failure uses the actual retained structure's score. Missing or invalid scores
have a blank rank. The order is descending score, then original sample number,
original reward rank, and original filename for ties.

Ranked new structures have an additional `_score_rankN` suffix. Original sample
numbers and search `_rankN` suffixes remain. Result and interface tables refer to
the same final filenames, which validation reads directly. A search
`partiscore_score` is a reward and is not substituted for the final score.
Recovery clears a managed task's old ranks before changing geometry, reuses
unchanged valid totals, and publishes the complete task order after scoring.
Legacy task files are not retroactively renamed. `score: false` keeps its
existing output schema and filenames.


## Docking versus design

The condition determines the task semantics. Full ligand SMILES fix chemical
identity: generating only its position/conformation is docking. Free chemistry
and partially fixed scaffold/motif chemistry are ligand design; they may use
`af3_ligand_iptm_screen` with the four fold models or the separate ODesign/Vina
design criterion. Full sequence conditions make ssDNA/ssRNA pose sampling
fixed-sequence docking as well.

Docking tasks run generation only. They do not have shipped `validate` profiles,
refold-based success gates, or an internal docking RMSD validation method.
The task input need not contain an observed bound pose. `nuc_ligand`,
`posebusters`, `ssdna_docking` and `ssrna_docking` retain their sampling task
files and inputs; their internal validation profiles have been removed.
`ligand_design` is de novo chemistry and retains its design evaluation, while
`nuc_ligand_binder` designs new nucleic-acid sequences and retains refolding
validation. The distinction does not change inference or optional task-level
PartiScore/OpenMM settings.

## Native AF3 confidence and failure recovery

Native AF3 can report undefined confidence for a single-atom ion with no
valid local frame. The adapter checks the actual native frame and chain masks
before converting only SDK-defined missing chain-confidence entries to null.
Protein and ordinary multi-atom ligand confidence still require valid finite
values. Infinity, incorrect array shapes, invalid ranges, and unexplained NaN
remain errors; full PAE/PDE and atom pLDDT retain their finite checks. Ions,
ADP, 3PG and other ligand chains remain in the prediction. This policy does not
change criteria thresholds or missing-value behavior.

Native chain-confidence arrays and pair matrices use native chain identifiers.
Explicit source-to-native mappings take precedence when a restored source label
collides with another native label (for example source protein A maps to native
D while native A is Mg). Source-keyed pLDDT and atom-count fields retain their
source identities; split native fragments retain token-weighted reductions.

Recoverable prediction-output errors fail the affected request while retaining
the healthy resident model. A genuinely broken native service is closed and
may be replaced once; repeated infrastructure failure aborts validation rather
than recording the rest of the queue as design failures. Healthy requests keep
the same service and model cache, and recovery does not clear MPNN/scorer owners.

Records publish an explicit failure_stage at the producing boundary; prediction
errors retain error_stage. Input, MPNN, refold, metrics and PartiScore failures
are reported separately. Legacy AF3 errors with saved sequences and no
predictions are recognized as refold failures. Successful recovery removes
stale failure fields.
Error-free pending/running records are counted as unfinished work, not input
or MPNN failures.

Ordinary validation resume reuses saved sequence slots, including those in
failed records, and retains compatible completed predictions. Use the original
AME profile to resume unfinished refolds after repair. --rerun regenerates
sequences; --rerun-refold keeps sequences but deliberately replaces all selected
refolds. Neither is required for ordinary recovery. Refiltering saved records
updates criteria and summaries without running models.
