# Benchmark generation and validation

The maintained task maps and validation profiles are under `configs/benchmarks`.
The [benchmark index](../configs/benchmarks/README.md) lists every retained suite.
Run the commands below from the repository root after installing model assets.
Routing is enabled by default; use `--route-ckpt MEMBER` to force one bundle member,
or `--ckpt FILE --route false` to run a standalone checkpoint.

## Current round: generation budgets and fresh outputs

The current round contains 38,440 generated structures across normal/alpha/beta,
or 18,360 when only the normal ProtDBench/PepDBench variants are counted.
These totals exclude sequence redesign and refold multipliers. The exact family
and sequence allocations are in the [current-round table](../configs/benchmarks/README.md)
and the manifest's `current_round` section. These are configuration budgets, not
a queue of unfinished jobs. The user reports existing ProtDBench alpha/beta
sampling; retain those outputs without resampling. Historical directories remain intact.
Use the independent `inference/benchmark_round_20261012/<family>_<variant>` paths
shown below, without pooling historical structures into this round.

Nuc monomer includes only RNA/DNA: five existing lengths each, 100 structures per
length, 1,000 total. The shared task file also contains an unrelated protein
monomer, which is excluded from the baseline NA budget by explicit task selection:

```bash
partidesign check configs/benchmarks/monomer/tasks.yaml --task rna_monomer_benchmark --task dna_monomer_benchmark
```

Protein bind RNA/DNA now uses the five DISCO-derived targets detailed below:
two RNA and three DNA, with all integer protein lengths 50–80 aa and 4 per
length. This gives 124 per target, 248 RNA and 372 DNA structures. Complete
original NA coordinates and locally chosen compact hotspots form a fixed-context
benchmark; DISCO co-folds the NA instead. AME retains 41×100=4,100. Formal protein–ligand benchmarking
has only OQO, IAI, FAD and SAM at length100, 200 each, 800 total. The other five
runnable tasks moved to [examples](../configs/examples/protein_ligand_additional.yaml);
their input assets remain. Historical 13/9 source inventory counts are not the
current formal task count.

This round excludes docking, antibody, external MotifBench and unlisted suites;
their existing configurations remain. Configuration and CPU compilation do not
start sampling or refolding. Complete the AF3 break/index mapping acceptance
before formal submission. CPU compilation is not GPU or scientific acceptance.

## Protein design against DISCO-derived RNA/DNA targets

These five source targets come from the [DISCO published release](https://huggingface.co/datasets/DISCO-Design/DISCO_benchmark_data).
[DISCO A.8.5](https://arxiv.org/html/2604.05181v1#A8.SS5) generates 100 total designs
per target, samples protein lengths 50–80, crops each DNA chain to at most 10 nt,
and evaluates Chai refolds after an NA-anchor fit. Its NA coordinates co-fold.
Here protein length range matches that range, while our 124-per-target budget,
complete fixed original NA, local hotspots and existing AF3 validation are distinct.
Both families use our project-custom `af3_protein_na_complex` criterion: joint protein-CA / NA-P complex RMSD <5 Å.

| Case | Prepared context | Source model | Task hotspot selectors |
| --- | --- | ---: | --- |
| 6YMC | RNA A, 26 nt | 1 | `A12/A14-16` |
| 8BWT | RNA A, 26 nt | 6 | `A12/A14-16` |
| 9JPJ | DNA A+B, 27+27 nt | 1 | `A14-15/B14-15` |
| 7S03 | DNA B+C, 26+26 nt | 1 | `B17-18/C8-9` |
| 7EDS | DNA F+R, 30+30 nt | 1 | `F8-9/R22-23` |

Inputs contain only the listed complete NA chains from one original model.
8BWT model 6 is an NMR ensemble medoid selected by NA P-RMSD, independently
of successful designs. Heavy coordinates, residue identities and author
numbers are preserved.

Hotspots are four-nt local patches, with native C1-prime spans 9.16–12.65 Å.
They are project choices, not publicly released DISCO hotspot configurations.
RNA and 9JPJ use co-occurring released design-contact candidates.
7S03 and 7EDS retain two DISCO-supported anchors and add two
opposite-strand native contacts to form one spatially coherent region. Released
passing flags and short distances establish neither experimental binding nor
affinity; source/model/hotspot provenance is retained in the existing manifest.

Each target expands to exactly 4 requests at every integer length 50–80.
Use `inference/benchmark_round_20261012/protein_bind_rna_disco` and
`inference/benchmark_round_20261012/protein_bind_dna_disco` for fresh outputs.
Previous generated structures, refolds and histories remain available separately.

## Sources and protocol provenance

Target provenance and success criteria are separate. Historical paths in
`configs/benchmarks/manifest.yaml` identify migrated files; they are not paper
citations. Input metadata or the manifest records source structures and preparation.

| Family | Target/task provenance | Success protocol provenance |
| --- | --- | --- |
| ProtDBench | Ten targets and the paper backbone budget, with local input/crop preparation | [ProtDBench AF2-IG-Easy](https://github.com/congliuUvA/ProtDBench/blob/6cd7ac9df0621d0950b6e6bc65eb5f6e0d392bfd/protdbench/protd_configs/eval.py) |
| PepDBench | Six local targets expanded to linear, disulfide and head-tail cyclic tasks; current metadata does not establish a standalone official target-set release | [BindCraft peptide confidence subset](https://github.com/martinpacesa/BindCraft/blob/main/settings_filters/peptide_filters.json); the separately named ProtDBench filter is optional |
| AME | [41-case AME archive](https://zenodo.org/records/17401001), with local motif index preparation | [Complexa AME defaults](https://github.com/NVIDIA-BioNeMo/Proteina-Complexa/blob/dev/src/proteinfoundation/result_analysis/motif_binder_analysis_utils.py) |
| MotifBench | [Official v1.0 30 cases](https://github.com/blt2114/MotifBench) | Official external evaluation, not a local threshold preset |
| Protein binder for ligand | Complexa/NVIDIA input assets; local SMILES/context and design grammar | Complexa ligand-binder defaults |
| Protein binder for DNA | DISCO-selected 9JPJ/7S03/7EDS with complete native DNA and local hotspots | Project-custom `af3_protein_na_complex`: joint protein CA + DNA P complex RMSD <5 Angstrom; all loops retained, no PAE gate |
| Protein binder for RNA | DISCO-selected 6YMC/8BWT with complete native RNA and local hotspots | Project-custom `af3_protein_na_complex`: joint protein CA + RNA P complex RMSD <5 Angstrom |
| RNA/DNA binder for protein | [NACraft NA-12](../configs/benchmarks/manifest.yaml): six RNA and six DNA source targets | Project-custom `af3_protein_na_complex`: joint protein CA + RNA/DNA P complex RMSD <5 Angstrom; target set and protocol differ from ODesign |
| RNA/DNA binder for ligand | Imidacloprid nitroimine, Ethyl maltol and Vanillin × RNA30/DNA20; six local groups | Project-custom `af3_na_bind_ligand` dual RMSD gate; not an official ODesign target set |
| Protein / RNA / DNA monomers | Unconditional generation length grids, not a fixed target collection | Existing user-selected La-Proteina backbone proxy and ODesign C1-prime token-center adaptation; unchanged this round |
| De novo ligand design | Local PDB pocket panel, not a verified complete official ODesign target set | ODesign QED >0.25 and Vina score <-8.18 kcal/mol, after valid recognition/docking |
| Antibody | Nine local RFAntibody-inspired target/scaffold tasks | Existing RFAntibody-inspired shared AF3/ESMFold screen, not full RF2 reproduction |
| CPS / ligand docking / ssDNA / ssRNA docking | Local PDB-derived cohorts; NA-ligand comes from the historical test archive and ssNA from a recentPDB shortlist | Docking is sampling-only under the accepted policy; do not apply design refold gates |

PoseBusters input files are materialized PDB/SMILES cases. Current input metadata
identifies their RCSB source structures but does not establish exact membership
in a pinned official PoseBusters release. The same provenance limitation applies
to the CPSet and historical NA-ligand docking archives. Keep that distinction
when reporting an official benchmark comparison.

## Nucleic-acid binder design

The RNA/DNA bind-protein profiles use NACraft NA-12: six targets per family,
replacing the former three-target panels and their input files.

| Polymer | Task | Published design length | Backbones |
| --- | --- | ---: | ---: |
| RNA | `rna_bind_protein_8x0n` | 12 | 100 |
| RNA | `rna_bind_protein_8swc` | 20 | 100 |
| RNA | `rna_bind_protein_9rvp` | 34 | 100 |
| RNA | `rna_bind_protein_9xzr` | 71 | 100 |
| RNA | `rna_bind_protein_9ny9` | 19 | 100 |
| RNA | `rna_bind_protein_8swb` | 20 | 100 |
| DNA | `dna_bind_protein_8c58` | 14 | 100 |
| DNA | `dna_bind_protein_8u0p` | 11 | 100 |
| DNA | `dna_bind_protein_7uv7` | 21 | 100 |
| DNA | `dna_bind_protein_8vx4` | 35 | 100 |
| DNA | `dna_bind_protein_7uxy` | 32 | 100 |
| DNA | `dna_bind_protein_8sln` | 29 | 100 |

The input folders contain exactly twelve files, six RNA and six DNA cases:
`inputs/rna_bind_protein/<PDB>_protein.pdb` and
`inputs/dna_bind_protein/<PDB>_protein.pdb`. Only the task-ready protein inputs
are retained; duplicate native complexes and unrelated antigen/selective/OGA
structures are removed. The existing manifest links each original source at
commit `2ec87dbb64a3ab5209795aaa4f70ac9dd152c1dc` and records chain selection,
published hotspots and preparation.

Ten raw inputs fail unique SEQRES/coordinate alignment, including an excluded
native chain in 9RVP. Prepared files omit SEQRES and retain only selected protein
ATOM records, original coordinates, author residue numbers, insertion codes,
altlocs, TER and internal CONECT edges where present. Waters, ions, native NA and
other additives are removed. Observed protein sequence lengths differ from the
published sequence-based NACraft inputs, e.g. 8SWC 149/294 aa and 9XZR 176/293 aa.
No missing residues are reconstructed; this is observed-fragment context preparation.

This is 1,200 protein-target NA backbones total. [The manifest](../configs/benchmarks/manifest.yaml)
records exact original source links, author hotspots, observed/full-sequence differences
and preparation. Tasks use the selected observed protein fragment; native NA and
additives are excluded, and missing residues are not reconstructed. NACraft optimizes
100 parents per target per mode, then two NA-MPNN variants per parent for 300 AF3
validations. Our 100 structures match one mode's parent count. Local validation
remains Original1 + FI NA-MPNN1 + ordinary NA-MPNN4, FI cutoff6 Å, temperature0.1;
this is distinct from NACraft optimization and the official ODesign target set.
Old outputs remain; new runs use `rna_bind_protein_nacraft` /
`dna_bind_protein_nacraft` round leaves.

The six `nuc_ligand_binder` groups use RNA30 or DNA20, 100 each, 600 total.
They contain Imidacloprid nitroimine, Ethyl maltol and Vanillin; Menthol and
other lengths are excluded without deleting input assets. Historical ESMFold2
results informed the fixed lengths; the new round is validated with original
AF3, once per sequence, using Original1 + FI NA-MPNN1 only. There is no ordinary
NA-MPNN branch. The project-custom criterion requires the same prediction to
satisfy both `na_p_self_rmsd < 5.0 Å` and `na_aligned_ligand_rmsd < 5.0 Å`.
Report Original, FI and their union independently. This local assessment is
not the official ODesign target collection.

AF3 validation uses MSA mode `auto`, templates enabled, cache
`.cache/af3_msa`, and two search workers. New NA-12 context sequences may
require searches; this update runs neither MSA nor AF3. ESMFold2-fast remains
available as an explicit comparison override. Baseline monomer grids and docking
configurations remain unchanged; the new coarse-condition files below are separate.

Protein bind RNA/DNA and RNA/DNA bind protein share the project-custom
`af3_protein_na_complex` success gate: `complex_ca_p_rmsd < 5.0 Angstrom`.
All protein CA and observed valid NA P points undergo one unweighted joint
fit against the generated CIF; no independent chain fit or RMSD averaging is
used. Missing predicted points or chain correspondence cannot pass, and 5.0
does not pass. All terminal loops remain included. Native confidence and
directional placement metrics stay reported, without an added PAE gate.
This is a local standard rather than an ODesign/RFD3/DISCO reproduction.
Ligand-target RNA/DNA keeps the separate dual gate above.

## Additional paired and helical sampling

`manifest.yaml.additional_sampling` records six requested cohorts with unchanged
baseline lengths and per-length/per-task budgets. They add **7,400** generated
structures: 2,800 paired NA designs, 500 helical protein monomers, and 4,100
helical AME designs. Together with the original 38,440 budget this is **45,840**;
counting only normal ProtDBench/PepDBench gives **25,760**. Existing sampling is
retained; these numbers do not state how many jobs remain.

Use `tasks_paired.yaml` in `monomer`, `rna_bind_protein`, `dna_bind_protein` and
`nuc_ligand_binder` for 1,000/600/600/600 structures respectively. Each task adds
only a design-chain coarse `paired` condition. No per-nucleotide states or pairing
partners are fixed, and original input/Hotspot/chemistry settings are preserved.
Both RNA and DNA monomer grids are identical to their original five lengths.

`monomer/tasks_alpha.yaml` contains only `protein_monomer_benchmark`: five lengths
100/200/300/400/500 aa, 100 structures at each, with coarse `alpha`. The existing
`ame/tasks_alpha.yaml` is AME-helical: 41 cases ×100, keeping every normal 180-aa
Contig, motif atom, fixed identity and ligand constraint. Reuse it directly;
`ame/tasks_beta.yaml` remains optional outside these additional totals.

All these cohorts use AF3, with existing criteria and sequence branches unchanged.
RNA/DNA monomer uses `monomer/validate_na.yaml`; protein monomer uses
`monomer/validate.yaml`. Binder and AME variants reuse their own family's
`validate.yaml`. For example:

```bash
partidesign design configs/benchmarks/monomer/tasks_paired.yaml --out inference/benchmark_round_20261012/nuc_monomer_paired
partidesign validate inference/benchmark_round_20261012/nuc_monomer_paired --config configs/benchmarks/monomer/validate_na.yaml
partidesign design configs/benchmarks/monomer/tasks_alpha.yaml --out inference/benchmark_round_20261012/protein_monomer_alpha
partidesign validate inference/benchmark_round_20261012/protein_monomer_alpha --config configs/benchmarks/monomer/validate.yaml
partidesign design configs/benchmarks/ame/tasks_alpha.yaml --out inference/benchmark_round_20261012/ame_alpha
partidesign validate inference/benchmark_round_20261012/ame_alpha --config configs/benchmarks/ame/validate.yaml
```

The manifest lists distinct directories for every paired family. Keep normal and
conditioned results separate even though the task names match. Coarse guidance
does not guarantee all generated positions are paired or helical. No sampling,
MSA search or validation job is started by this configuration update.

Protein bind RNA/DNA uses Original1 + FI LigandMPNN1 + ordinary LigandMPNN4.
Nuc monomer uses Original1 + ordinary NA-MPNN8 with no FI. AME keeps Original1 +
FI LigandMPNN1 + ordinary LigandMPNN8; protein–ligand keeps Original1 + FI
LigandMPNN1 + ordinary LigandMPNN1. All branches are reported separately.
Ordinary `@N` counts only the `mpnn` branch, excluding `self`/`self_fi`; its
full-budget rate represents the configured N. Generic `first1/bestN` column names
are not available for all families; AME retains explicit `first1/best8` fields.
A pooled union does not replace Original success, and metrics-only families
do not acquire a binary gate by pooling.


The following commands are runnable examples; they are shown for planning only.
The `--plan` validation calls inspect configuration without launching model
execution.

```bash
partidesign design configs/benchmarks/rna_bind_protein/tasks.yaml --out inference/benchmark_round_20261012/rna_bind_protein_nacraft
partidesign check configs/benchmarks/rna_bind_protein/tasks.yaml
partidesign validate inference/benchmark_round_20261012/rna_bind_protein_nacraft --config configs/benchmarks/rna_bind_protein/validate.yaml --plan

partidesign design configs/benchmarks/dna_bind_protein/tasks.yaml --out inference/benchmark_round_20261012/dna_bind_protein_nacraft
partidesign check configs/benchmarks/dna_bind_protein/tasks.yaml
partidesign validate inference/benchmark_round_20261012/dna_bind_protein_nacraft --config configs/benchmarks/dna_bind_protein/validate.yaml --plan

partidesign design configs/benchmarks/nuc_ligand_binder/tasks.yaml --out inference/benchmark_round_20261012/nuc_ligand_binder
partidesign check configs/benchmarks/nuc_ligand_binder/tasks.yaml
partidesign validate inference/benchmark_round_20261012/nuc_ligand_binder --config configs/benchmarks/nuc_ligand_binder/validate.yaml --plan
```

## ProtDBench

Each variant contains 10 targets and 2,840 backbones, with four backbones per
integer binder length. Alpha/beta add only coarse secondary-structure conditioning.

```bash
partidesign design configs/benchmarks/protdbench/tasks.yaml --out inference/benchmark_round_20261012/protdbench
partidesign design configs/benchmarks/protdbench/tasks_alpha.yaml --out inference/benchmark_round_20261012/protdbench_alpha
partidesign design configs/benchmarks/protdbench/tasks_beta.yaml --out inference/benchmark_round_20261012/protdbench_beta

partidesign validate inference/benchmark_round_20261012/protdbench --config configs/benchmarks/protdbench/validate.yaml --worker-gpus 0.5 --worker-cpus 1
```

Use the same validation config for alpha/beta run directories. It selects
ProteinMPNN, original1 + fixed-interface1 + MPNN8, model1 AF2-IG with three
recycles, and `af2_ig_protdbench`.

## PepDBench

Each variant contains 18 tasks: six targets times linear/disulfide/head-tail
cyclic topology. Every task has 40 backbones at each length 8-17: 400 per task,
7,200 per variant. The base, alpha and beta task maps preserve the same bonds,
fixed identities and cyclic offset.

```bash
partidesign design configs/benchmarks/pepdbench/tasks.yaml --out inference/benchmark_round_20261012/pepdbench
partidesign design configs/benchmarks/pepdbench/tasks_alpha.yaml --out inference/benchmark_round_20261012/pepdbench_alpha
partidesign design configs/benchmarks/pepdbench/tasks_beta.yaml --out inference/benchmark_round_20261012/pepdbench_beta
```

Choose the validation profile by topology; avoid applying the linear profile to
cyclic tasks. This complete example validates the base run:

```bash
linear_disulfide=()
cyclic=()
for target in dll3 gabarap keap1 klk2 mdm2 xiap; do
  linear_disulfide+=(--task "pepdbench_${target}_linear" --task "pepdbench_${target}_disulfide")
  cyclic+=(--task "pepdbench_${target}_cyclic")
done

partidesign validate inference/benchmark_round_20261012/pepdbench "${linear_disulfide[@]}"   --config configs/benchmarks/pepdbench/validate.yaml   --worker-gpus 0.5 --worker-cpus 1

partidesign validate inference/benchmark_round_20261012/pepdbench "${cyclic[@]}"   --config configs/benchmarks/pepdbench/validate_cyclic.yaml   --worker-gpus 0.5 --worker-cpus 1
```

For alpha/beta, replace `inference/benchmark_round_20261012/pepdbench` with the corresponding run directory.
Both profiles use original1 + fixed-interface1 + MPNN4 at 0.0001, FI cutoff6,
model1 AF2-IG with three recycles and `af2_ig_peptide`. The cyclic profile uses
`community_models/cyclicmpnn/checkpoint/cyclicmpnn_48_010.pkl`; the other uses
ProteinMPNN. Disulfide identities remain fixed when free cysteine is omitted.

## Check before submitting

```bash
partidesign check configs/benchmarks/protdbench/tasks_alpha.yaml
partidesign check configs/benchmarks/pepdbench/tasks_beta.yaml
partidesign validate inference/benchmark_round_20261012/pepdbench --config configs/benchmarks/pepdbench/validate_cyclic.yaml --task pepdbench_dll3_cyclic --plan
```

The default check compiles one representative request per task on CPU, without
sampling. A validation `--plan` inspects saved outputs/configuration without model
execution. Each MPNN budget is separate from the generation budget; do not multiply
backbone counts by sequence slots when comparing generation budgets.
