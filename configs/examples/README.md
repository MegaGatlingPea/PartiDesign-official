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

[Protein](#protein) · [Nucleic acids](#nucleic-acids) · [Ligands](#ligands) ·
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

## Nucleic acids

Tasks in [nucleic_acids.yaml](nucleic_acids.yaml):

| Task | Request |
| --- | --- |
| `rna_fixture`, `dna_fixture` | RNA or DNA aptamer design around a protein receptor with specified hotspots. |
| `rna_bind_ligand`, `dna_bind_ligand` | RNA or DNA aptamer design around a retained small molecule. |
| `rna_monomer`, `dna_monomer` | De novo RNA or DNA generation at several requested lengths. |
| `rna_monomer_folding`, `dna_monomer_folding` | Structure generation for a fully specified nucleotide sequence. |
| `rna_invfold`, `dna_invfold` | Identity redesign conditioned on a supplied backbone. |
| `rna_ss_guided`, `dna_ss_guided` | Secondary-structure guidance combining chain-level, nucleotide-level, and pair conditions. |
| `rna_motif_all_atom` | RNA scaffolding around a retained all-atom motif. |
| `rna_motif_base_atoms` | RNA scaffolding around selected base atoms. |

## Ligands

Tasks in [ligand.yaml](ligand.yaml):

| Task | Request |
| --- | --- |
| `ligand_fixture` | Pocket-conditioned generation of ligand chemistry and coordinates, with ligand size sampled from the configured heavy-atom count distribution. |
| `ligand_scaffold_hopping` | Ligand redesign conditioned on the supplied ligand backbone. |
| `ligand_motif_scaffolding` | Ligand generation around selected source atoms. |

The fixed-identity ligand tasks from this file are listed under
[Docking](#docking-preview). See
[ligand identity modes](../../docs/task_schema.md#ligand-identity-modes) for the
distinction between fixed chemistry, partial chemistry, and unconstrained design.

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
