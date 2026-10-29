# Example tasks

These YAML files contain runnable requests for the supported inference workflows.
Run commands from the repository root so their relative structure paths resolve
correctly. Complete the [installation](../../README.md#installation) first.

Choose a **named task** with `--task`; omitting it selects every task in that
file. Preview the settings before generation:

```bash
partidesign design configs/examples/protein.yaml \
  --task protein_fixture --run-name protein_example \
  --batch-size 1 --compile false --show-config

partidesign check configs/examples/protein.yaml --task protein_fixture

CUDA_VISIBLE_DEVICES=0 partidesign design configs/examples/protein.yaml \
  --task protein_fixture --run-name protein_example \
  --batch-size 1 --compile false
```

Each selected task writes to `inference/<run-name>/<task-name>/`. Use a fresh run
name or output root for another generation run. Read the task's `sample_num` and
length plan before running it: one named task can request several lengths and
multiple samples. `--batch-size` controls execution, not the total sample count.

[Protein](#protein) · [Partial diffusion](#partial-diffusion) · [Pipeline demo](#pipeline-demo) ·
[Nucleic acids](#nucleic-acids) · [Ligands](#ligands) ·
[Motifs](#motifs) · [Peptide topologies](#peptide-topologies-preview) ·
[Docking](#docking-preview) · [Antibodies](#antibodies-preview)

## Protein

Tasks in [protein.yaml](protein.yaml):

| Task | Request |
| --- | --- |
| `protein_fixture` | Protein binder around a protein receptor with specified hotspots; five length-40 samples, scoring and refinement disabled. |
| `protein_monomer` | De novo protein generation at several requested lengths. |
| `protein_bind_protein_alpha_guided` | Protein binder with an α-rich secondary-structure condition. |
| `protein_bind_protein_beta_guided` | Protein binder with a β-rich secondary-structure condition. |
| `protein_binds_ligand` | Protein design around a retained small-molecule ligand. |
| `protein_bind_rna` | Protein design around an RNA receptor. |
| `protein_bind_dna` | Protein design around a DNA receptor. |

Protein atomic-motif examples from the same file appear under [Motifs](#motifs).

[protein_ligand_additional.yaml](protein_ligand_additional.yaml) contains further
protein designs around fixed ligand contexts. These examples are outside the
current formal benchmark:

| Task | Retained ligand context |
| --- | --- |
| `imidacloprid_nitroimine` | Nitroimine form of imidacloprid. |
| `imidacloprid_ring_imine` | Ring-imine form of imidacloprid. |
| `ligand_binder_menthol_fixed` | Menthol. |
| `ligand_binder_ethyl_maltol_fixed` | Ethyl maltol. |
| `ligand_binder_vanillin_fixed` | Vanillin. |

## Partial diffusion

[partial_diffusion.yaml](partial_diffusion.yaml) restarts from one actual reference
CIF bundled under [structures/partial_diffusion/](structures/partial_diffusion/).
The task YAML supplies its molecular conditions. Run directly:

```bash
partidesign design configs/examples/partial_diffusion.yaml --run-name protein_partial
```

The example restarts the bundled 100-residue ESMFold2-fast refold, retaining
its alpha secondary-structure condition, and requests four replicas of chain `P` at
`fraction: 0.625`: the last 250 of 400 native intervals, starting at index 150.
Actual lengths come from the reference. For your own structure, define the
conditions, motifs, context and bonds in its task YAML, then set `partial.reference`
to your CIF. This route requires a flex-context checkpoint. No input CSV is
required; new children still receive the normal output `result.csv`. See the
[partial guide](../../docs/partial.md) for conditions, exact contig gaps and
optional saved metadata.

## Pipeline demo

[pipeline_tasks.yaml](pipeline_tasks.yaml) is a plain native task collection,
usable by both `partidesign design` and `partidesign pipeline`. It contains no
workflow settings:

| Task | Request |
| --- | --- |
| `protdbench_insulin_receptor` | Length-80 insulin receptor protein binder. |
| `protdbench_pdl1` | Length-80 PD-L1 protein binder. |
| `protdbench_bhrf1` | Length-80 BHRF1 protein binder. |
| `pepdbench_mdm2_linear` | Length-13 MDM2 linear peptide. |
| `pepdbench_mdm2_cyclic` | Length-13 MDM2 peptide with terminal C-N bond and cyclic offsets. |
| `pepdbench_mdm2_disulfide` | Length-13 MDM2 peptide with fixed cysteines and an SG-SG bond. |

Each task requests 32 initial backbones. Select tasks exactly as in design:

```bash
partidesign pipeline configs/examples/pipeline_tasks.yaml \
  --task protdbench_insulin_receptor --preset ppi --run-name insulin_receptor
```

Workflow presets live in `configs/pipeline/`: protein/nuc use AF3 final;
ppi uses AF2-IG models 1 and 2, five matched pairs per model, requiring both
winners to pass. The normal first/later selection caps are 200.

For the previous small, strict demo behavior, select
[configs/pipeline/demo.yaml](../pipeline/demo.yaml) instead:

```bash
partidesign pipeline configs/examples/pipeline_tasks.yaml \
  --task protdbench_insulin_receptor --config configs/pipeline/demo.yaml \
  --run-name insulin_receptor_demo
```

The demo workflow uses production sequence counts (GT1+MPNN1 first,
GT1+MPNN4 later), three rounds, strict native `af3_basic` filtering,
first-round K=200, later/final max-pass=200, partial fraction 0.625 with four replicas
and parent retention, and no OpenMM. Middle rounds that continue retain at most
two candidates from each parent/partial family before the task-wide cap. Add `--plan` to preview without loading
models or creating output. Repeat `--task` to select several; omitting it selects
all six tasks. Strict criteria can end a task early or produce no final pass.
See the [pipeline guide](../../docs/pipeline.md).

## Nucleic acids

Tasks in [nucleic_acids.yaml](nucleic_acids.yaml):

| Task | Request |
| --- | --- |
| `rna_fixture`, `dna_fixture` | RNA or DNA aptamer design around a protein receptor with specified hotspots. |
| `rna_bind_protein`, `dna_bind_protein` | Design 20-nucleotide RNA or DNA chains with protein chain E as context and specified hotspots. |
| `rna_bind_ligand`, `dna_bind_ligand` | RNA or DNA aptamer design around a retained small molecule. |
| `rna_monomer`, `dna_monomer` | De novo RNA or DNA generation at several requested lengths. |
| `rna_monomer_folding`, `dna_monomer_folding` | Structure generation for a fully specified nucleotide sequence. |
| `rna_invfold`, `dna_invfold` | Identity redesign conditioned on a supplied backbone. |
| `rna_ss_guided`, `dna_ss_guided` | Secondary-structure guidance combining chain-level, nucleotide-level, and pair conditions. |
| `rna_motif_all_atom` | RNA scaffolding around a retained all-atom motif. |
| `rna_motif_base_atoms` | RNA scaffolding around selected base atoms. |
| `rna_sequence_prediction`, `rna_ss_prediction` | Predict a fixed 12-nucleotide RNA sequence, with optional nucleotide and pair secondary-structure conditions. |
| `dna_sequence_prediction`, `dna_ss_prediction` | Predict a fixed 16-nucleotide DNA sequence, with optional nucleotide and pair secondary-structure conditions. |
| `rna_bind_protein_unpaired` | Design a 20-nucleotide unpaired RNA chain around the supplied protein chain A. |
| `dna_bind_protein_unpaired` | Design a 35-nucleotide unpaired DNA chain around the supplied protein chain A. |
| `rna_ligand_binder_ethyl_maltol_unpaired` | Design a 30-nucleotide unpaired RNA chain around fixed ethyl maltol, defined by SMILES. |
| `dna_ligand_binder_vanillin_unpaired` | Design a 20-nucleotide unpaired DNA chain around fixed vanillin, defined by SMILES. |

The sequence-only and SS-annotated controls fix nucleotide identities and supply
no reference coordinates. In the unpaired aptamer cases, nucleotide identities
remain free; `unpaired` is their secondary-structure condition. The sequence-control
and unpaired aptamer tasks request five samples with OpenMM refinement. Their inputs reuse structures
already included under `configs/benchmarks/inputs/`.

For example:

```bash
partidesign design configs/examples/nucleic_acids.yaml \
  --task rna_ss_prediction --run-name rna_ss_prediction \
  --batch-size 1 --compile false

partidesign design configs/examples/nucleic_acids.yaml \
  --task dna_ligand_binder_vanillin_unpaired --run-name dna_vanillin \
  --batch-size 1 --compile false
```

Fixed-sequence RNA docking examples from this file are listed under
[Docking](#docking-preview).

## Ligands

Tasks in [ligand.yaml](ligand.yaml):

| Task | Request |
| --- | --- |
| `ligand_fixture` | Pocket-conditioned generation of ligand chemistry and coordinates, with ligand size sampled from the configured heavy-atom count distribution. |
| `ligand_scaffold_hopping` | Ligand redesign conditioned on the supplied ligand backbone. |
| `ligand_motif_scaffolding` | Ligand generation around selected source atoms. |
| `ligand_decoration` | Retain a six-atom scaffold and grow from C9, sampling 12, 13, or 14 total heavy atoms before fragment selection. |
| `ligand_decoration_two_sites` | Use the same sizes with simultaneous attachment sites at C12 and C10. |

The fixed-identity ligand tasks from this file are listed under
[Docking](#docking-preview). See
[ligand identity modes](../../docs/task_schema.md#ligand-identity-modes) for the
distinction between fixed chemistry, partial chemistry, and unconstrained design.

Run the decoration example with:

```bash
partidesign design configs/examples/ligand.yaml \
  --task ligand_decoration --task ligand_decoration_two_sites \
  --run-name ligand_decoration \
  --batch-size 1 --compile false
```

Each task requests five samples at each size: 15 samples per task, with scoring
and OpenMM relaxation. Keep just one `--task` flag to run only that example.
The two-site task requires both scaffold-to-new bonds; bonds among new atoms
remain generated, so the two sites can form separate branches or a bridge.
Unattached fragments may be discarded while retaining the motif and required
connections, so the final molecule can be smaller than the sampling size. See
[ligand decoration](../../docs/task_schema.md#ligand-decoration) for attachment
indexing and geometry.

## Motifs

| Task | File | Request |
| --- | --- | --- |
| `motif_fixture` | [motif.yaml](motif.yaml) | Protein scaffolding around a contig with selected atoms retained. |
| `protein_monomer_motif_tip` | [protein.yaml](protein.yaml) | Protein scaffolding around selected side-chain tip atoms. |
| `protein_binds_ligand_ame` | [protein.yaml](protein.yaml) | Atomic motif scaffolding around a ligand, retaining specified residue atoms. |

For RNA motifs, use the two `rna_motif_*` tasks listed above. The
[backbone and contig reference](../../docs/task_schema.md#backbone-and-contig)
explains how to combine retained segments with newly generated regions.

## Peptide topologies (Preview)

**Inference is available** for peptides with the topologies below. Peptide design
remains under active development, and output quality may be inconsistent across
targets and constraints. These examples demonstrate task authoring and inference
support; they are not a performance comparison.

Tasks in [peptide.yaml](peptide.yaml):

| Task | Topology and conditions |
| --- | --- |
| `peptide_fixture` | Head-to-tail cyclic peptide with an explicit terminal bond. |
| `peptide_bind_protein_disulfide` | One specified disulfide bond. |
| `peptide_bind_protein_betahairpin` | Disulfide-constrained peptide with β-hairpin guidance. |
| `peptide_bind_protein_double_disulfide` | Two specified disulfide bonds. |
| `peptide_bind_protein_cyclotide` | Head-to-tail closure and three specified disulfide bonds. |
| `peptide_bind_protein_isopeptide` | Lys–Asp side-chain isopeptide linkage. |
| `peptide_bind_protein_lactam` | N-terminal-to-Glu side-chain lactam linkage. |
| `peptide_bind_protein_hydrocarbon_staple` | One MK8–MK8 hydrocarbon staple with helical guidance. |
| `peptide_bind_protein_whl_staple` | Cysteines linked through a fixed WHL linker. |
| `peptide_bind_protein_tbmb_bicyclic` | Three cysteines linked through a fixed TBMB linker. |
| `peptide_bind_protein_double_hydrocarbon_staple` | Two MK8–MK8 hydrocarbon staples with helical guidance. |

For example, preview or run one topology:

```bash
partidesign design configs/examples/peptide.yaml \
  --task peptide_bind_protein_hydrocarbon_staple --run-name peptide_example \
  --batch-size 1 --compile false --show-config
```

Remove `--show-config` to generate after checking the selected task and assets.
Explicit bonds specify the requested chemistry; `cyclic_offset` alone does not
create a covalent closure. See [explicit bonds](../../docs/task_schema.md#explicit-bonds).

## Docking (Preview)

**Inference is available** for fixed-identity docking. Docking remains under
active development, and pose quality may be inconsistent. The examples below
illustrate supported inputs without claiming benchmark performance.

| Task | File | Request |
| --- | --- | --- |
| `ligand_smiles_fixture` | [ligand.yaml](ligand.yaml) | Dock the ligand specified by a complete SMILES into a protein receptor. |
| `ligand_covalent_docking` | [ligand.yaml](ligand.yaml) | Dock with the source ligand's atom types and bonds fixed by `mode: type_bond`. |
| `peptide_dock_protein` | [peptide.yaml](peptide.yaml) | Dock a fully specified peptide with an explicit linker and bonds into a protein receptor. |
| `peptide_dock_protein_cyclic_disulfide` | [peptide.yaml](peptide.yaml) | Dock a fixed peptide with head-to-tail closure, a disulfide bond and cyclic offsets into the supplied protein receptor. |
| `rna_dock_protein` | [nucleic_acids.yaml](nucleic_acids.yaml) | Dock fixed RNA sequence `UAAUUU` into the supplied protein receptor. |
| `rna_dock_larp1` | [nucleic_acids.yaml](nucleic_acids.yaml) | Dock fixed RNA sequence `AAAA` into the LARP1 receptor. |
| `ssrna_dock_7tuv_b0_1_4` | [ssRNA task file](../benchmarks/ssrna_docking/tasks.yaml) | Dock a specified ssRNA sequence into a protein receptor. |
| `nuc_ligand_3gx5` | [Ligand–nucleic acid task file](../benchmarks/nuc_ligand/tasks.yaml) | Dock a fixed ligand into an RNA receptor. |

Fixed full ligand chemistry specifies docking; free or partial chemistry
specifies ligand design. A full fixed polymer sequence with receptor context
likewise specifies pose docking. These tasks sample poses and can use the
requested scoring/refinement; they do not apply an internal refolding pass gate.
The two task files under `configs/benchmarks/` request 40 samples per selected
case, so copy and edit a task if a smaller trial is needed.

```bash
partidesign design configs/examples/ligand.yaml \
  --task ligand_smiles_fixture --run-name docking_example \
  --batch-size 1 --compile false --show-config
```

## Antibodies (Preview)

**Inference is available** for paired VH/VL variable-domain scaffolds and
single-domain VHH scaffolds, including selected-CDR design with optional antigen
context. Antibody design remains under active development; output quality may
be inconsistent, and generated structures do not establish experimental binding.

Tasks in [antibody.yaml](antibody.yaml):

| Task | Request |
| --- | --- |
| `antibody_fixture` | Paired VH/VL scaffold with native-length VH CDR3 design, fixed VL identity, and PD-L1 context. |
| `vhh_fixture` | VHH CDR design with antigen context; framework identity remains fixed. |
| `pdl1_adalimumab_selective_cdr3_official` | Paired VH/VL scaffold with VH CDR3 length sampled from the scaffold's allowed range. |
| `penguinpox_7eow_selective_cdr3_official` | VHH scaffold with CDR3 length sampled from the scaffold's allowed range. |
| `pdl1_adalimumab_framework_optin` | Paired VH/VL CDR3 design with one selected framework position also mutable. |
| `penguinpox_7eow_framework_optin` | VHH CDR3 design with one selected framework position also mutable. |
| `pdl1_adalimumab_reference` | Paired VH/VL reference request with all sequence identities fixed. |
| `penguinpox_7eow_reference` | VHH reference request with all sequence identities fixed. |

Submit antibody and normal tasks in **separate commands**. For example:

```bash
partidesign design configs/examples/antibody.yaml \
  --task antibody_fixture --run-name antibody_example \
  --batch-size 1 --compile false --show-config
```

Antibody-specific controls use the `antibody` schema; normal contig and bond
fields do not apply. See [antibody tasks](../../docs/task_schema.md#antibody-tasks)
for CDR selection, length ranges, framework mutability, and scaffold templates.

## Customize an example

Copy a task into your own YAML and adjust its sample count, lengths, retained
identities, motifs, secondary structure, hotspots, and bonds. Then select the
copied task explicitly with `--task`. Input structures for the primary example
files are under [structures/](structures/); additional task files use the paths
shown in their YAML.

- [Task YAML reference](../../docs/task_schema.md): all supported input fields.
- [Command guide](../../docs/commands.md): model setup, generation flags,
  scoring, refinement, validation, and clustering.
- [Benchmark configurations](../benchmarks/): task collections and validation
  settings used for evaluation.
- [Custom criteria example](custom_presets.yaml): success criteria for
  `partidesign validate --presets`, rather than a generation task file.
