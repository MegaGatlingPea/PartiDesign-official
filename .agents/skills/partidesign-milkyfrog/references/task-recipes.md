# Choose and prepare a task

Locate the runtime checkout first. Here `<repo>` means that checkout, even when
this skill is installed elsewhere. Read its current
`<repo>/configs/examples/README.md` and `<repo>/docs/task_schema.md`; the YAML in
that checkout is the source of truth. Run bundled examples from `<repo>` so
their source-structure paths resolve.

## Resolve the request before copying an example

- Identify the generated modality and goal: new identities and structure,
  fixed-identity folding/docking, backbone-conditioned redesign, or motif
  scaffolding. A supplied input chain is not automatically fixed.
- For context-dependent work, obtain the local PDB/mmCIF, the chains to retain,
  the design chain IDs, and any user-specified interface residues. Do not invent
  a target, hotspot, sequence, motif, bond, or success threshold.
- Make the budget explicit: `sample_num`, each length or joint length row,
  optional Best-of-N replicas, scoring/refinement, and any later validation.
  Ask for missing lengths/counts or a usable compute budget before generation.
- Record which identities, atoms, secondary structure, or chemistry must stay
  fixed. For antibodies, obtain the scaffold, mutable roles/CDRs and lengths,
  optional framework positions, and optional antigen context.

Source selectors for crop, hotspots, contigs, and context bond endpoints use
the input identifiers: mmCIF `label_asym_id`/`label_seq_id`, or PDB chain/residue
IDs and insertion codes. Sparse sequences, secondary-structure ranges, and
design bond endpoints use final design numbering, starting at 1; negative
positions count from the end. Antibody framework selectors use source
`label_seq_id` within the scaffold role. Keep these numbering systems separate.

## Protein, nucleic acid, and ligand recipes

All paths in this table are beneath `<repo>/configs/examples/`.

| User goal | File and named starting task | Resolve before adapting |
| --- | --- | --- |
| Protein binder for a protein | `protein.yaml`: `protein_fixture` | Receptor chains, intended interface if supplied, binder lengths, sample budget. The fixture is five length-40 samples with scoring/refinement off. |
| Protein without a receptor | `protein.yaml`: `protein_monomer` | Length plan and count; retain `input: null` for a de novo monomer. |
| Protein with α/β guidance | `protein.yaml`: `protein_bind_protein_alpha_guided`, `protein_bind_protein_beta_guided` | Whether coarse guidance or specific final residue blocks are intended; guidance is not a structural guarantee. |
| Protein binding RNA or DNA | `protein.yaml`: `protein_bind_rna`, `protein_bind_dna` | Exact retained nucleic-acid chains and binder length plan. |
| Protein binding a ligand | `protein.yaml`: `protein_binds_ligand` | Retained ligand identity/atoms, surrounding context, design chain ID and lengths. Additional fixed-ligand examples are indexed in `protein_ligand_additional.yaml`, outside the current formal benchmark. |
| Protein motif scaffolding | `motif.yaml`: `motif_fixture`; `protein.yaml`: `protein_monomer_motif_tip` | Source motif residues/atoms, generated gaps, final length limits, and identities to preserve. |
| Atomic motif scaffolding around a ligand (AME) | `protein.yaml`: `protein_binds_ligand_ame` | Ligand context and exact motif atoms; evaluate agreement afterward because conditioning does not overwrite predictions with exact source coordinates. |
| RNA/DNA aptamer for a protein | `nucleic_acids.yaml`: `rna_fixture`, `dna_fixture` | RNA versus DNA, receptor chains, interface if specified, lengths/count. Both fixtures request five length-20 samples. |
| RNA/DNA aptamer for a ligand | `nucleic_acids.yaml`: `rna_bind_ligand`, `dna_bind_ligand` | Retained ligand identity and context; each shipped task requests five length-24 samples. |
| RNA/DNA monomer | `nucleic_acids.yaml`: `rna_monomer`, `dna_monomer` | Length plan and count without receptor context. |
| Fold a supplied RNA/DNA sequence | `nucleic_acids.yaml`: `rna_monomer_folding`, `dna_monomer_folding` | Complete user sequence and alphabet; full identity determines length. Adding receptor context changes the goal to pose docking. |
| RNA/DNA identity redesign on a backbone | `nucleic_acids.yaml`: `rna_invfold`, `dna_invfold` | Source design chain and representative-center constraints; backbone conditioning allows identities to change. |
| RNA/DNA secondary-structure guidance | `nucleic_acids.yaml`: `rna_ss_guided`, `dna_ss_guided` | Coarse paired/unpaired intent, final token ranges, and any explicit compatible pair relations. |
| Fixed RNA/DNA sequence controls | `nucleic_acids.yaml`: `rna_sequence_prediction`, `dna_sequence_prediction`; `rna_ss_prediction`, `dna_ss_prediction` add explicit pairing | Full sequence, optional pairing, and whether OpenMM is wanted; the examples enable it. |
| Unpaired RNA/DNA aptamer for a protein | `nucleic_acids.yaml`: `rna_bind_protein_unpaired`, `dna_bind_protein_unpaired` | Retained receptor and unpaired regions; unpaired guidance does not fix the final conformation. |
| Unpaired aptamer around a SMILES-defined ligand | `nucleic_acids.yaml`: `rna_ligand_binder_ethyl_maltol_unpaired`, `dna_ligand_binder_vanillin_unpaired` | Fixed ligand SMILES and aptamer length; the ligand remains context rather than new chemistry to generate. |
| RNA motif scaffolding | `nucleic_acids.yaml`: `rna_motif_all_atom`, `rna_motif_base_atoms` | Source motif and atom subset, generated gaps and length limits. |
| New ligand chemistry in a protein pocket | `ligand.yaml`: `ligand_fixture` | Pocket/context and heavy-atom budget. `from_distribution` needs input pocket context and exactly one ligand design chain. |
| Ligand scaffold hopping | `ligand.yaml`: `ligand_scaffold_hopping` | Source ligand chain; `condition.backbone` retains centers while chemistry can change. |
| Ligand growth around selected atoms | `ligand.yaml`: `ligand_motif_scaffolding` | Stable source atom names and final heavy-atom count; the example uses 41 heavy atoms. |
| Ligand decoration at one or more retained atoms | `ligand.yaml`: `ligand_decoration`, `ligand_decoration_two_sites` | Grow from C9 alone, or C12 and C10 together, at 12, 13, or 14 total heavy atoms. `attachments` requires the declared retained-to-new bonds and forbids others. New atom identities and the remaining graph are generated; no complete product SMILES is needed. |

For decoration, read the `atom_motif` attachment fields in
`<repo>/docs/task_schema.md` and adapt `ligand_decoration` rather than inventing
another bond syntax. `new_atom` indexes a generated endpoint from 1; it does
not set the number of atoms in a branch;
`design_length` counts retained and new heavy atoms together; `sample_num` applies
to each listed size. Multiple attachments can use different retained atoms.
New-to-new bonds remain free, so two sites may form separate branches or a bridge.
Disconnected products reuse valid-fragment selection: retain all motif atoms
and required attachment endpoints, and discard unattached fragments. Sampling
count can exceed the retained output count. Attachments fix connectivity, not
final bond lengths or chemical quality.

For several generated chains, consult the length-plan section of
`<repo>/docs/task_schema.md`. Explicit `design_length.joint` rows are complete
polymer-chain combinations, not a Cartesian product; ligand counts come from
their chemistry conditions. Do not create a new schema to express a joint task.

## Docking — Preview

Complete fixed ligand chemistry plus receptor context specifies pose docking.
Free or partial ligand chemistry specifies design. A full fixed polymer
sequence with receptor context likewise specifies docking. These workflows
support inference, with variable pose quality; they do not apply an internal
refolding pass gate.

| Goal | File beneath `<repo>/configs/` and task | Important distinction |
| --- | --- | --- |
| Known SMILES ligand in a protein receptor | `examples/ligand.yaml`: `ligand_smiles_fixture` | Complete connected SMILES fixes the graph and determines the heavy-atom count. |
| Source ligand with its graph fixed | `examples/ligand.yaml`: `ligand_covalent_docking` | `mode: type_bond` fixes source atom types/bonds. This example declares no receptor covalent bond; its name alone does not create one. |
| Fixed peptide with a linker in a protein receptor | `examples/peptide.yaml`: `peptide_dock_protein` | Preserve only the user's intended full sequence, linker and explicit bonds; the shipped example includes them. |
| Fixed cyclic peptide with a disulfide | `examples/peptide.yaml`: `peptide_dock_protein_cyclic_disulfide` | Full sequence, explicit terminal closure and disulfide, with offset encoding also enabled. |
| Fixed short RNA in a protein receptor | `examples/nucleic_acids.yaml`: `rna_dock_protein`, `rna_dock_larp1` | Full sequence determines length; receptor context makes this docking rather than a monomer folding task. |
| Fixed ssRNA in a protein receptor | `benchmarks/ssrna_docking/tasks.yaml`: `ssrna_dock_7tuv_b0_1_4` | Requests 40 samples as shipped. Copy/edit the selected task for a smaller trial. |
| Known ligand in an RNA receptor | `benchmarks/nuc_ligand/tasks.yaml`: `nuc_ligand_3gx5` | Requests 40 samples as shipped, with scoring/refinement enabled. |

## Peptide topologies — Preview

Use explicit `peptide` design intent; length alone does not turn `protein` into
`peptide`. All named peptide examples below are in
`<repo>/configs/examples/peptide.yaml` and request five base samples as shipped.
Clarify topology, sequence/modified components, endpoint identities and atom
names, linkers, lengths, and budget before adapting them.

| Requested topology | Starting task |
| --- | --- |
| Linear | No dedicated linear-only example is shipped. Use the normal peptide schema; do not carry the cyclic fixture's terminal bond into a linear request. |
| Head-to-tail cyclic | `peptide_fixture` |
| One or two disulfides | `peptide_bind_protein_disulfide`, `peptide_bind_protein_double_disulfide` |
| Disulfide with β-hairpin guidance | `peptide_bind_protein_betahairpin` |
| Head-to-tail closure plus three disulfides | `peptide_bind_protein_cyclotide` |
| Lys–Asp isopeptide / N-terminal–Glu lactam | `peptide_bind_protein_isopeptide`, `peptide_bind_protein_lactam` |
| One or two MK8 hydrocarbon staples | `peptide_bind_protein_hydrocarbon_staple`, `peptide_bind_protein_double_hydrocarbon_staple` |
| WHL-linked cysteines / TBMB bicyclic peptide | `peptide_bind_protein_whl_staple`, `peptide_bind_protein_tbmb_bicyclic` |

`cyclic_offset` changes relative-position encoding; an explicit terminal bond
creates head-to-tail connectivity. Side-chain bonds need known endpoint
identities. Bond declarations do not guarantee optimized geometry or
stereochemistry. Some examples enable OpenMM processing. Read the current
explicit-bond schema and `<repo>/docs/validate.md` before choosing an evaluator:
backend support differs for modifications and crosslinks.

## Antibody CDR design — Preview

Use `<repo>/configs/examples/antibody.yaml` with `format: antibody`; submit it
separately from normal tasks. Every shipped antibody task requests five samples.

| Goal | Starting task or choice |
| --- | --- |
| Paired VH/VL, native-length VH CDR3, fixed VL | `antibody_fixture` |
| VHH with antigen context | `vhh_fixture`; omitted `antibody.design` makes all CDRs mutable while framework identities stay fixed. |
| Sample the scaffold's allowed CDR3 range | `pdl1_adalimumab_selective_cdr3_official`; the example index also lists the VHH counterpart. |
| Open selected framework positions | `pdl1_adalimumab_framework_optin`; obtain exact source framework positions instead of guessing antibody numbering. |
| Keep all identities fixed | `pdl1_adalimumab_reference`; `antibody.design: {}` fixes sequences, while coordinates are still generated. |

Choose a built-in scaffold and its exact roles (`VH`/`VL` or `VHH`), unique
output chain IDs, mutable CDRs/lengths, template setting, and antigen chains if
present. The schema does not accept normal contigs, bonds, or cyclic fields.
Framework templates are soft geometry guidance. See the antibody section of
`<repo>/docs/task_schema.md` and the scaffold definitions under
`<repo>/configs/resources/antibody/scaffolds/`.

## Confirm and run the selected request

Read the copied YAML's actual count and length plan. The shipped monomer tasks
for protein, RNA and DNA, the two guided protein tasks, `protein_binds_ligand`,
`protein_bind_rna`, and `protein_bind_dna` each use `sample_num: 1` across five
lengths, for five base requests. Other named example tasks above use five base
samples; the two benchmark docking tasks use 40. Best-of-N replicas multiply
candidate work. `--batch-size` does not change these counts.

From `<repo>`, this real fixture preview requests five length-40 proteins:

```bash
partidesign design configs/examples/protein.yaml \
  --task protein_fixture --run-name protein_trial \
  --batch-size 1 --show-config

partidesign check configs/examples/protein.yaml --task protein_fixture
```

Use the same checkpoint, routing settings and overlays for checking and
generation. Select tasks explicitly: repeated `--task` selects several, while
omitting it selects every task in the file. Automatic routing chooses one
member for the whole submission; keep it enabled unless a matching checkpoint
requires an explicit choice. The `pep` member needs a version-2 six-member
bundle. See `<repo>/docs/design.md` for execution options and routing.

After inputs, assets and requested work are settled, remove `--show-config` to
generate into a fresh run/task directory. Inspect `result.csv` and
`logs/design.log`; later scoring, refinement and validation are separate choices
described in `<repo>/docs/score.md`, `<repo>/docs/relax.md`, and
`<repo>/docs/validate.md`. Docking, antibody and peptide topology workflows are
Preview features; their availability and images do not establish binding or
benchmark performance.
