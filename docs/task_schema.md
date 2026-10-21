# Task schema

A normal task YAML is a closed mapping from a task name to one task dictionary.
The task name may contain ASCII letters, digits, `_`, and `-`. Unknown keys,
duplicate keys, duplicate chain names, invalid scalar types, and non-finite numeric
values are errors before model loading.

```yaml
protein_fixture:
  input: ./configs/examples/structures/protein_design/prot_bind_prot.pdb
  sample_num: 1
  design_chains:
    - {P: protein}
  design_length:
    P: [40]
  hotspot: E64/E88/E96
  score: false
  postprocess: false
```

Use [Configuration](configuration.md) for run-wide options. This page defines the
fields shared by [protein](tasks/protein.md), [peptide](tasks/peptide.md),
[nucleic-acid](tasks/nucleic_acids.md), [ligand](tasks/ligand.md), and
[motif](tasks/motif.md) tasks. Antibody tasks use `format: antibody` and the
additional schema described in [Antibody](tasks/antibody.md).

## Top-level task fields

| Field | Type | Meaning |
| --- | --- | --- |
| `format` | string, optional | `normal` (default) or `antibody`. Selected tasks must share one format. |
| `input` | local `.pdb` or `.cif` path, optional | Source structure. Omit for de novo tasks. Paths are relative to the invocation directory. |
| `sample_num` | positive integer | Number of samples for each expanded length plan. |
| `design_chains` | non-empty list | One-entry maps such as `{P: protein}`. Modalities are `protein`, `peptide`, `rna`, `dna`, and `ligand`. `peptide` is an authoring alias for protein chemistry. |
| `design_length` | mapping, optional | One-chain length list/grid, an explicit multi-chain `joint` table, or ligand `from_distribution`. Conditions can provide the length instead. |
| `condition` | mapping, optional | Sequence, backbone, contig, secondary-structure, ligand, and bond constraints. |
| `hotspot` | selector string, optional | Context residues exposed as the target interface. It does not redesign or freeze the selected context. |
| `crop` | mapping, optional | Context selection by `include`, `exclude`, and/or residue `proximity`. |
| `score` | boolean, optional | Write a final PartiScore result for successful structures. Default `false`. |
| `postprocess` | boolean, optional | Run the configured OpenMM postprocess for successful structures. Default `false`. |
| `override` | boolean, optional | Allow the requested postprocess to replace the base structure. Default `true`. |
| `cyclic_offset` | boolean, optional | Use the cyclic protein offset path. It does not create a ring by itself. CyclicMPNN eligibility can also come from a required-output head-to-tail single bond. Default `false`. |
| `cyclic_offset_chains` | list of protein chain IDs, optional | Required when several design protein chains use `cyclic_offset`. |
| `rm_token_type` | list of canonical token names or `null`, optional | Task-level free-polymer identity omission policy. Omitted or `null` uses `[CYS]`; `[]` disables it. See [Configuration](configuration.md). |

`sample_num` multiplies every explicit length plan. For example, two lengths and
`sample_num: 3` create six base requests. Best-of-N replicas are added after task,
length, and condition resolution. The runtime writes successful ranked replicas
with rank-qualified filenames; replicas do not change the base task sample budget.

## Design chains and source structures

`design_chains` declares the output entities. A chain that exists in `input` is
redesigned from its normalized source; a declared chain absent from the input is
created de novo. Other source chains remain rigid or flexible context according to
the selected model context mode. A normal task may declare several design chains,
but mixed modalities require an explicit `design_length.joint` plan.

PDB and mmCIF are read as the first model. Input paths are local, and the reader
keeps the source topology and concrete bond orders it can read. Water and hydrogen
rows may be dropped during normalization. A PDB `TER` boundary creates separate
internal entities even when the author chain ID is repeated.

## Length plans

For one design chain, use an explicit list or an aligned inclusive grid:

```yaml
design_length:
  P: [80, 100, 120]

# Equivalent grid form:
design_length:
  P: {min: 80, max: 120, step: 20}
```

For multiple design chains, every row must name every polymer design chain in the
same declared order:

```yaml
design_chains: [{P: protein}, {R: rna}]
design_length:
  joint:
    - {P: 80, R: 24}
    - {P: 100, R: 30}
```

Joint rows are not a Cartesian product. Missing or extra chain keys are rejected.
Ligand-only tasks use a separate public heavy-atom count. A single free ligand may
request a sampled count with `L: from_distribution`; a fixed ligand mode (`smiles`,
`ccd`, `type_bond`, or `backbone`) cannot also request a sampled atom count.

If a full sequence, backbone, or a contig fixes the final length, do not provide a
conflicting `design_length`. A contig may instead use
`motif_min_length`/`motif_max_length` to sample one feasible final length.

## Conditions

`condition` is additive, but two conditions that describe the same identity or
coordinate fact must agree. The compiler resolves conditions before model execution
and fails a contradictory task explicitly.

### Sequence identity

Use a full string or a sparse list. Protein strings use the standard one-letter
alphabet; RNA uses `A/C/G/U`; DNA uses `A/C/G/T`. A sparse row's `resid` is
one-based, and negative positions count from the end of a design chain (`-1` is its
last position):

```yaml
condition:
  sequence:
    P: "ACDEFGHIKLMNPQRSTVWY"
    # Or only fix two positions:
    # P:
    #   - {resid: 2, residue: C}
    #   - {resid: -2, residue: C}
```

Bracketed CCD identities such as `[MK8]` or `[BAL]` retain a known heavy-atom
carrier when its chemistry is available. They are private chemical components, not
extra model letters. A fixed sequence position is exempt from token omission and
remains fixed even when its identity is omitted globally.

### Backbone and contig

`backbone: [P]` copies the source design chain's representative centers while
leaving identity to the decoder. It requires that the design chain exists in the
input and cannot be combined with a contig for that chain.

`contig` copies selected source residues and samples the gap lengths. Each slash
separates a gap (`N` or `N-M`) from a source segment (`P19-27` or `P19`). Source
segments must belong to the same design chain:

```yaml
condition:
  contig:
    P:
      token: "10-40/P19-27/10-40"
      polymer_atom: null
```

The copied segment is a hard coordinate motif. The `polymer_atom` field is either
`null` or a mapping keyed by copied residue selectors; a scalar such as
`polymer_atom: tip` is invalid. `polymer_atom: null` keeps all available copied
atoms. A mapping such as `{P19: [CA, C, O]}` keeps only those atom names, while
`{P19: tip}` uses the residue-specific side-chain tip preset. The source identity in a copied contig is
also fixed. `motif_min_length` and `motif_max_length` constrain the final sampled
length across all gaps.

### Interactive task authoring

The [notebook](../partidesign.ipynb) opens
`peptide/peptide_bind_protein_betahairpin`, a 15-residue peptide task with fixed
Cys2/Cys14, beta-sheet segments 1–6 and 10–15, adjacent strand pairing, and an
explicit SG–SG bond. Its fixed **Nord Light** workbench has six steps:
**1 Input & design**, **2 Motif**, **3 Hotspot**, **4 Advanced conditions**,
**5 Save**, and **6 Runtime**. Mol* remains mounted throughout,
including before loading Input and for **De novo** tasks. The first five
steps keep source and applied-condition previews; all inference, validation,
clustering, and result-opening controls are in **6 Runtime**.

Choose a complete task preset in **1 Input & design**, before reviewing its
conditions. All **46 non-antibody presets** under `configs/examples/` load the input,
modality, lengths, Motif, Hotspots, and advanced settings together, update Mol*,
and return to the first step. Loading another preset replaces the whole current
task rather than adding conditions to it. Antibody examples are excluded from the notebook. Visual native forms expose
existing joint design, legacy Motif, and context-ligand fields. Existing mapping
and list values are editable while native source paths and chain IDs remain
read-only. **Full task YAML** supports adding/removing entries and other
structural changes; apply YAML edits before checking or saving.

The original loaded structure, active authoring Input, and exported task input
are separate files. Chain filters apply on Enter or when leaving the field,
updating the active Input and viewer while preserving the original in `raw/`.
A failed filter leaves the current Input unchanged. Review roles and applied
marks after changing active chains.

Active roles are **Context** (target/environment) and **Reference** (geometry
for optional Motifs and contact selection). Assigning Reference does not
select Motif residues. The entire retained Reference geometry can identify
contacting Context residues independently of Motif selection. Preview candidates
at an adjustable cutoff, default **6 Å**, then explicitly Merge or Replace
applied Hotspots. A Reference used only for contact selection supplies no model
coordinates and does not become Context.

Motifs are optional, and **Selection**, **Contig**, and **Atoms** areas remain
visible when disabled. A compact residue list shows the current atom restriction
for each copied residue; only the focused residue's atom editor is displayed.
List and editor appear side by side when space allows. A chain selector switches
between multiple Contigs without changing their values. Loaded presets retain authored residues and atom restrictions.
Newly applied residues default to all atoms observed in their source CIF;
clearing a protein atom selection retains that full observed inventory.
Protein subsets can only use observed atoms. Canonical RNA/DNA offer complete
observed atoms or the supported combined base/non-backbone group. Modified
nucleotides offer all observed atoms. Legacy nucleotide subsets stay visible
and read-only until explicitly replaced with a supported group; the form does
not offer arbitrary nucleotide atom subsets or rewrite native syntax on load.

Mol* has **Chain**, **Residue**, and **Atom** scope controls in its upper-right
corner. These select a whole chain, a whole residue, or an atom/bond's endpoint
atoms, respectively. Picks follow the current page and the supported Motif
atom choices above. Mol* and selection lists highlight the same residues or atoms. Motif and
Context-only Hotspot picks have independent pending lists. Apply selections
explicitly and check the applied summaries; later per-residue Motif atom edits
update the current task directly. For a source containing B12–20,
`10-20/B12-20/15` retains nine source residues and adds 10–20 plus 15 new
residues, totaling 34–44. Source selectors use the displayed chain IDs and
numbering. Contig source residues and per-residue atom restrictions must agree.

Working normalized source is temporary under `.cache/`. Exported input omits
unused References, retaining Context and complete source chains needed by
active Motif, backbone, or source-graph conditions. Only selected Motif atoms
supply Motif coordinates; visible reference-only atoms add no conditions.
**4 Advanced conditions** contains sequence, fine secondary structure, RNA/DNA
pairing, explicit bonds, native condition examples, and complete-YAML editing.
**5 Save** checks and exports the current task. Free ligand length accepts an
explicit count/range or `from_distribution`; supplied chemistry/backbone can
determine the count instead.

Question-circle help appears on hover or keyboard focus. Click the icon to pin
the help while inspecting a field; press Esc to close it.

**6 Runtime** prepares native commands and immutable task/input/custom scaffold
snapshots under `.cache/`. Preparation never starts a job. **Execute** and
**Modify command** default off; copy the selectable read-only command for manual
use, or enable Execute and explicitly click **Execute command**. Modify enables
editing of the actual command arguments. Editing, toggling, and preparing remain
idle. There is no implicit shell pipeline; use an explicit `bash -lc '...'`
command when shell syntax is needed. Owned tmux jobs expose logs, refresh,
stop, and reconnect controls.

Notebook model compilation defaults off, with an optional toggle for larger
jobs. The output root defaults to `configs/customs/inference`; native generation
adds `<task name>/` once. Validation takes an existing result directory, retains
**Original · 1**, and exposes independent ordinary/FI MPNN counts and
temperatures. FI also exposes its cutoff. AF3 defaults retain Context MSA
**Auto / reuse cache**, templates on, and native cache settings. Empty notebook
criteria enable unfiltered success for complete validation only; optional named
criteria add thresholds. Clustering handles original successful backbones
separately for each method/cohort, with TM threshold **0.5** and concurrency
at most **16**. Runtime also selects raw samples and successful method/source/
prediction slots for preview. Validated refolds retain native chain coloring
and confidence values; those values are not interpreted as task-condition
categories.

Custom originals, exported structures, and task YAML default to ignored
`configs/customs/raw/`, `input/`, and `tasks/`. See
[notebook outputs and Runtime](outputs.md#notebook-runtime-and-result-preview)
for prepared commands, result selection, and recovery.

### Secondary structure

Secondary-structure guidance is split into a coarse chain label, token ranges, and
pair relations. Protein/peptide coarse labels are `alpha` and `beta`; RNA/DNA
coarse labels are `paired` and `unpaired`. `loop` is not a coarse label. Protein
token labels are `helix`, `sheet`, and `loop`; RNA/DNA token labels are `paired`
and `unpaired`. Ligands do not accept secondary-structure conditions. Token ranges
use final one-based positions; pair entries use chain selectors and ranges:

```yaml
condition:
  secondary_structure:
    coarse: {P: beta}
    token:
      P:
        sheet: "1-8,12-19"
        loop: "9-11"
    pair:
      adjacent: [["P1-8", "P12-19"]]
      not_adjacent: [["P1-8", "P25-32"]]
```

Use the modality-specific examples in [Protein](tasks/protein.md) and [Nucleic
acids](tasks/nucleic_acids.md) for paired/unpaired nucleic labels.

### Ligand identity modes

The public ligand modes are mutually exclusive per design ligand chain:

| Mode | Effect |
| --- | --- |
| `smiles` | Fixes one connected molecular graph from a SMILES string and generates its pose. |
| `ccd` | Uses one uppercase CCD component ID and its complete known heavy-atom graph. |
| `backbone` | Keeps an existing ligand atom inventory and source positions while identity is hidden. |
| `type_bond` | Keeps an existing ligand atom inventory, element types, internal graph, and source covalent facts. |
| `atom_motif` | Keeps selected existing atom names as a hard motif and samples the remaining heavy-atom count. |

SMILES must contain exactly one connected component. Public ligand lengths count
heavy atoms; hydrogens in a SMILES are chemistry input and never model tokens. A
source or supplied ligand graph is authoritative: reconstruction cannot overwrite
its hard bonds. See [Ligand tasks](tasks/ligand.md).

### Explicit bonds

Use a list of atom endpoint mappings. Polymer endpoints require `chain`, `resid`,
and `atom`; ligand endpoints use stable `chain` and `atom` names. Design-chain
negative residue positions are allowed. Bond order is a concrete `single`,
`double`, `triple`, or `aromatic` value supported by the input chemistry:

```yaml
condition:
  bond:
    - atom1: {chain: P, resid: 1, atom: N}
      atom2: {chain: P, resid: -1, atom: C}
      order: single
```

An explicit bond is a hard output fact and must have addressable endpoint identity.
Duplicate declarations with the same unordered endpoints and order collapse;
conflicting orders fail. `omit_atoms` may remove non-representative atoms from a
design polymer endpoint residue, or named atoms from a complete fixed design
ligand (`ccd`/`type_bond`), to describe the product of a reaction. The omitted
atom must belong to one of that bond's endpoint components. Context, bond
endpoints, hard coordinate motifs, and free/partial ligand chemistry cannot be
removed this way. Fixed ligand products are matched to their full CCD precursor
before projection; a changed inventory uses a local CIF component ID and its
actual internal graph rather than claiming the complete precursor CCD identity.

## Selectors, crop, and hotspots

Context selectors use the source chain and source `label_seq_id`, with insertion
codes when needed. Examples are `A33`, `A95-97`, `A10A`, and `[RNA1]9-16` for a
multi-character chain. `hotspot` is a slash-separated union of residue selectors.

```yaml
crop:
  include: [A]
  proximity:
    - {anchor: P1-20, radius: 12.0}
  exclude: [W]
hotspot: A33/A95-97
```

`include` and `exclude` select context; `proximity` retains context around an
anchor. Only retained context is used for routing and reward applicability. A
hotspot marks the intended target interface and does not alter source coordinates.

## Precedence and omission

The most specific hard fact wins only when it is compatible with the source graph:
source topology and explicit bond declarations are retained, copied contig atoms
retain their coordinates, and explicit sequence/ligand identity fixes the decoder
identity. `rm_token_type` controls only free polymer identity decoding; it does not
remove context, motif, fixed sequence, explicit identity, or explicit bond endpoint
facts. An unknown motif identity still contributes its inferred motif coordinates
when its token is omitted.

The task omission default is `[CYS]` when the field is omitted or `null`. An explicit
task list, including `[]`, replaces that default; a non-null global list overrides
every task. The default affects free protein CYS only: RNA C and DNA DC are distinct
token names, and context, motifs, fixed sequences, explicit identities, and bond
endpoints remain exempt. The full precedence and all 32 canonical names are
centralized in [Configuration](configuration.md). Sequence-backend letter omission
is separate; see [MPNN](mpnn.md).
