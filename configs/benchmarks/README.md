# Current benchmark round

The `current_round` section of [manifest.yaml](manifest.yaml) specifies this
round. Budgets count generated structures before sequence redesign and refold.
These are allocation budgets, not a list of remaining jobs. Retain completed
outputs; ProtDBench alpha/beta sampling already exists according to the user.
New condition comparisons use separate directories under
`inference/benchmark_round_20261012/<family>_<variant>`; CPU checks create none.

| Benchmark | Generation allocation | Structures | Validation / criterion |
| --- | --- | ---: | --- |
| Nuc monomer | RNA/DNA: five existing lengths each, 100 per length | 1,000 | AF3 / `af3_na_monomer` |
| RNA bind protein | Six NACraft NA-12 RNA targets, 100 each | 600 | AF3 / `af3_protein_na_complex` |
| DNA bind protein | Six NACraft NA-12 DNA targets, 100 each | 600 | AF3 / `af3_protein_na_complex` |
| RNA/DNA bind ligand | Three ligands × RNA30/DNA20, 100 per group | 600 | AF3 / `af3_na_bind_ligand` |
| Protein bind RNA | Two DISCO RNA targets × 31 lengths × 4 | 248 | AF3 / `af3_protein_na_complex` |
| Protein bind DNA | Three DISCO DNA targets × 31 lengths × 4 | 372 | AF3 / `af3_protein_na_complex` |
| ProtDBench | Full integer ranges, four per length; three variants | 8,520 | AF2-IG model1 / `af2_ig_protdbench` |
| PepDBench | Six targets × three topologies × ten lengths × 40; three variants | 21,600 | AF2-IG model1 / `af2_ig_peptide` |
| AME | 41 tasks, 100 each | 4,100 | AF3 / `af3_ame` |
| Protein bind ligand | OQO/IAI/FAD/SAM, length100, 200 each | 800 | AF3 / `af3_protein_bind_ligand` |

Baseline plan (ProtDBench/PepDBench normal/alpha/beta): **38,440 structures**. Counting only normal
ProtDBench/PepDBench: **18,360**. ProtDBench has 2,840 and PepDBench 7,200 per
variant. Alpha/beta add only coarse secondary-structure conditioning; targets,
length ranges, budgets, chemistry, fixed identities and bonds stay identical.

## Additional condition comparisons

The manifest's `additional_sampling` section adds **7,400 structures** to the
baseline configuration budget: **45,840** overall, or **25,760** when only normal
ProtDBench/PepDBench is counted. These numbers describe configured sample
budgets; completion must be checked in the run results.

| Added comparison | Task file | Structures | Shared AF3 validation |
| --- | --- | ---: | --- |
| RNA/DNA monomer paired | [monomer/tasks_paired.yaml](monomer/tasks_paired.yaml) | 1,000 | [monomer/validate_na.yaml](monomer/validate_na.yaml) |
| RNA bind protein paired | [rna_bind_protein/tasks_paired.yaml](rna_bind_protein/tasks_paired.yaml) | 600 | [rna_bind_protein/validate.yaml](rna_bind_protein/validate.yaml) |
| DNA bind protein paired | [dna_bind_protein/tasks_paired.yaml](dna_bind_protein/tasks_paired.yaml) | 600 | [dna_bind_protein/validate.yaml](dna_bind_protein/validate.yaml) |
| RNA/DNA bind ligand paired | [nuc_ligand_binder/tasks_paired.yaml](nuc_ligand_binder/tasks_paired.yaml) | 600 | [nuc_ligand_binder/validate.yaml](nuc_ligand_binder/validate.yaml) |
| Protein monomer helical | [monomer/tasks_alpha.yaml](monomer/tasks_alpha.yaml) | 500 | [monomer/validate.yaml](monomer/validate.yaml) |
| AME helical (existing file reused) | [ame/tasks_alpha.yaml](ame/tasks_alpha.yaml) | 4,100 | [ame/validate.yaml](ame/validate.yaml) |

Paired adds only `condition.secondary_structure.coarse.<design chain>: paired`;
it leaves per-position states and pairing partners unspecified. Helical adds only
coarse `alpha`. These are conditioning comparisons, not guarantees about predicted
secondary structure. Inputs, length grids, sample counts, Hotspots, motifs, fixed
identities, chemistry, scoring/relax settings and validation branches match their
baselines. All six comparisons use the shared original AF3 profiles, one prediction
per sequence, with existing criteria unchanged.

Protein monomer helical keeps lengths 100/200/300/400/500 aa and 100 structures per
length. AME helical reuses all 41 cases at 100 each; no duplicate helical task file
is needed. AME beta remains available separately and is outside the added budget.
Task names stay unchanged within variant files; use their distinct planned output
directories in the manifest, rather than the baseline output directory.

Protein bind RNA/DNA uses all integer lengths 50–80 aa: 4 structures per
length, 124 per target. The panel is 6YMC/8BWT RNA and
9JPJ/7S03/7EDS DNA, derived from the DISCO release. Inputs retain
complete original NA coordinates; local compact hotspots are documented in
the manifest. Both directions and both nucleic-acid types use the project-custom
`af3_protein_na_complex` gate: one joint protein-CA / NA-P RMSD below 5 Angstrom.
DISCO co-folds the NA and uses shorter DNA crops, so this is a local fixed-NA
benchmark with its own budget and existing AF3 protocol. New output leaves are
`protein_bind_rna_disco` and `protein_bind_dna_disco` under the round root.

RNA/DNA bind-protein now use NACraft NA-12 instead of the old three-target
panels. [Target sources, author hotspots and preparation](manifest.yaml)
are available for inspection. 100 structures per target matches one NACraft mode's
parent count; Original1/FI1/ordinary4 remains our local validation protocol.
New output leaves use `rna_bind_protein_nacraft` and `dna_bind_protein_nacraft`
under the round root. The opposite-direction protein-design panels are listed above.

## Sequence and prediction budgets

| Benchmark | Original | Fixed interface | Ordinary redesign |
| --- | ---: | --- | --- |
| Nuc monomer | 1 | None | NA-MPNN8 |
| RNA/DNA bind protein | 1 | NA-MPNN1 | NA-MPNN4 |
| Protein bind RNA/DNA | 1 | LigandMPNN1 | LigandMPNN4 |
| RNA/DNA bind ligand | 1 | NA-MPNN1 | None |
| ProtDBench | 1 | ProteinMPNN1 | ProteinMPNN8 |
| PepDBench | 1 | MPNN1 | MPNN4 |
| AME | 1 | LigandMPNN1 | LigandMPNN8 |
| Protein bind ligand | 1 | LigandMPNN1 | LigandMPNN1 |

Reports keep `self` (Original), `self_fi` (FI), `mpnn` and the union separate.
Ordinary `@N` refers only to the configured `mpnn` pool, excluding Original/FI.
Its full-budget rate represents `@N`; generic `first1/bestN` field names are
not available for every family. AME retains explicit `first1/best8`. Union
success never substitutes for Original success. Metrics-only tasks do not
gain a binary success gate through pooling.

AF3 predicts once per sequence. Context MSA stays `auto`, templates enabled,
cache `.cache/af3_msa`, two search workers; design chains do not search MSA.
Existing temperatures, recycles, performance optimizations and persistent model
cache remain. Criteria contain executable methods/reductions/rules; sources,
units and missing-value semantics are in the [validation guide](../../docs/validate.md).

PepDBench linear/disulfide uses `validate.yaml`; head-tail cyclic uses
`validate_cyclic.yaml` and the supplied CyclicMPNN checkpoint. Both use model1
AF2-IG and `af2_ig_peptide`; cyclic validation enables the design-only AF2
offset. MPNN keeps fixed and motif identities; head-tail bonds remain explicit.

## Nucleic acid–ligand assessment

The six groups are Imidacloprid nitroimine, Ethyl maltol and Vanillin, each
with RNA30 and DNA20, 100 structures per group. Menthol and other lengths are
outside this benchmark; their existing input assets remain.

Lengths were chosen from historical ESMFold2 results. This local assessment
uses AF3 for the new round and is not the official ODesign target set. Each
structure has Original1 and FI NA-MPNN1 only; FI cutoff6 Å and temperature0.1.
The project-custom `af3_na_bind_ligand` requires **the same prediction** to
satisfy both `na_p_self_rmsd < 5.0 Å` and `na_aligned_ligand_rmsd < 5.0 Å`.
Report Original, FI and their union independently.

## Round scope and acceptance

Formal protein–ligand benchmarking has four tasks only. Five additional
runnable tasks are in [protein_ligand_additional.yaml](../examples/protein_ligand_additional.yaml),
with inputs retained. The historical 13/9 counts in the manifest describe source
inventories; the current formal protein–ligand count is four.

Docking, antibody, external MotifBench and unlisted suites are outside this
round; their configurations and assets remain. Nuc monomer selects only
`rna_monomer_benchmark` and `dna_monomer_benchmark`; the unconditional protein
monomer remains separately available. Its helical counterpart is in the added
condition comparisons above.

See the [validation command](../../docs/validate.md) to
preview and run these profiles. Task sources and budgets are recorded in
[manifest.yaml](manifest.yaml).
