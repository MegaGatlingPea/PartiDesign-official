# Writing task YAML

A task file describes **what to generate and what to retain**. Run settings such
as the checkpoint, GPU batch size, and output directory belong to the
[command line](commands.md). Run the bundled examples from the repository root.

Start with a complete task:

```yaml
my_protein:
  sample_num: 2
  design_chains: [{P: protein}]
  design_length: {P: [100]}
```

Save this as `my_tasks.yaml`. It requests two proteins of length 100, with no
source structure or fixed sequence. A file can contain several named tasks;
select one with `--task my_protein`, or omit `--task` to run all of them.
Names may contain ASCII letters, digits, `_`, and `-`.

## Choose an example

Copy a task from the following files, then change its input, chain IDs, lengths,
and conditions. The example files contain several tasks; select a name explicitly
when you only want one. Their `*_fixture` tasks request **five samples** each.

| Goal | Example file | Starting task |
| --- | --- | --- |
| Protein binder or monomer | [protein.yaml](../configs/examples/protein.yaml) | `protein_fixture` or `protein_monomer` |
| Peptide with head-to-tail closure | [peptide.yaml](../configs/examples/peptide.yaml) | `peptide_fixture` |
| RNA or DNA | [nucleic_acids.yaml](../configs/examples/nucleic_acids.yaml) | `rna_fixture` or `dna_fixture` |
| New ligand chemistry or known-ligand pose | [ligand.yaml](../configs/examples/ligand.yaml) | `ligand_fixture` or `ligand_smiles_fixture` |
| Preserve a source motif and generate its scaffold | [motif.yaml](../configs/examples/motif.yaml) | `motif_fixture` |
| Resample near an existing candidate | [partial_diffusion.yaml](../configs/examples/partial_diffusion.yaml) | `partial_protein` (bundled reference) |
| Antibody CDR design | [antibody.yaml](../configs/examples/antibody.yaml) | `antibody_fixture` |

`protein_monomer` has five lengths and one sample per length. Most fixture tasks
leave scoring and postprocessing off; the ligand fixtures enable both and need
those additional assets. Inspect the copied task before running it.

## Top-level task fields

Normal tasks use the fields below. [Antibody tasks](#antibody-tasks) use a
separate schema.

| Field | Required/default | Meaning |
| --- | --- | --- |
| `sample_num` | Required, positive integer | Base samples **per length plan**; partial tasks request replicas of one resolved reference. |
| `design_chains` | Required, nonempty list | One-entry maps from chain ID to `protein`, `peptide`, `rna`, `dna`, or `ligand`. |
| `input` | Optional; omitted/`null` for de novo | Local `.pdb` or `.cif` source. |
| `design_length` | Needed unless conditions or a partial reference determine every chain's size | Polymer residue counts or ligand heavy-atom counts. |
| `condition` | Optional | Sequence, backbone, contig, secondary structure, ligand chemistry, and bonds. |
| `partial` | Optional; absent for ordinary generation | `{reference, result_csv?, fraction}` for one actual candidate CIF; see the [partial guide](partial.md). |
| `crop` | Optional | Select source context to retain. |
| `hotspot` | Optional | Mark source context residues as the intended interface. |
| `format` | `normal` | Use `antibody` only with its separate schema. |
| `score` | `false` | Evaluate final structures with PartiScore; does not enable search or MPNN. |
| `postprocess` | `false` | Request OpenMM processing of successful structures. |
| `override` | `true` | Allow requested postprocessing to replace the base CIF; `false` skips that stage. |
| `cyclic_offset` | `false` | Use cyclic relative-position encoding for protein/peptide chains; adds no bond by itself. |
| `cyclic_offset_chains` | Optional | Required to select protein/peptide chains when several are designed and cyclic offsets are enabled. |
| `rm_token_type` | Task default `[CYS]` | Omit identities at free polymer positions, subject to the global policy below. |

Use YAML booleans and numbers, not quoted strings. Design chain IDs must be unique,
nonempty strings of at most 32 characters. Unknown fields, duplicate YAML keys,
and invalid types are errors. `--show-config` resolves configuration and basic
task fields; source atoms and detailed conditions are checked during compilation.

## Design chains and source structures

Declared chains are generated; other input chains become context. A design chain
absent from the input is de novo. If a design chain exists in the input, **its
presence alone preserves neither sequence nor coordinates**. Add conditions to
state what should be retained.

```yaml
design_chains: [{P: protein}]   # generate P; other input chains are context
```

`peptide` uses protein chemistry and the protein sequence alphabet, while recording
explicit peptide intent for model routing. A short `protein` chain is not
reclassified as a peptide by its length. RNA and DNA are separate modalities.

Input paths are local and resolve from the directory where you invoke the
command, **not the task YAML's directory**. Use absolute paths when moving a task
file outside the repository. The reader uses the first PDB/mmCIF model.

Source selectors use structural chain labels and residue identifiers. For mmCIF,
use `label_asym_id` and `label_seq_id`; for PDB, use the source chain/residue
identifiers. Insertion codes can disambiguate residues. A design chain matching
several PDB `TER` segments is ambiguous; disambiguate its input chain IDs first.
Context coordinates follow the checkpoint's rigid/flexible context mode.

## Length plans

For one design chain, give a list:

```yaml
design_length:
  P: [80, 100, 120]
```

Or use the equivalent aligned inclusive grid:

```yaml
design_length:
  P: {min: 80, max: 120, step: 20}
```

Counts and step must be positive integers; `max - min` must be divisible by
`step`. Three lengths with `sample_num: 2` request six base samples. Best-of-N
replicas multiply this budget later.

For several design chains, use explicit `joint` rows when specifying lengths:

```yaml
design_chains: [{P: protein}, {R: rna}]
design_length:
  joint:
    - {P: 80, R: 24}
    - {P: 100, R: 30}
```

Each row names every polymer design chain, in declaration order, with no extra
keys. These are explicit combinations, not a Cartesian product. Ligands obtain
counts from chemistry conditions and do not appear in `joint` rows. Multiple
ligand-only design chains also need conditions to determine their counts.

Omit `design_length` when full sequences, source backbones, contigs, fixed
ligand graphs, or a partial reference determine every chain's size. A sparse sequence does not determine
length. If independent conditions provide lengths, they must agree.

## Partial reference tasks

Write a normal task with its molecular conditions, then add `partial` to
start from one reference CIF. No earlier task file or input CSV is required.
The bundled example starts from a 100-residue protein chain `P`:

```yaml
partial_protein:
  sample_num: 4
  design_chains: [{P: protein}]
  condition:
    secondary_structure:
      coarse: {P: alpha}
  partial:
    reference: ./configs/examples/structures/partial_diffusion/protein_refold.cif
    fraction: 0.625
```

| Partial field | Meaning |
| --- | --- |
| `reference` | Required single actual candidate CIF with complete current complex geometry; directory input is unsupported. |
| `result_csv` | Optional explicit native CSV for saved locators and motif mappings. Omitted: use the task and CIF directly, without sibling CSV lookup. Supplied: exactly one successful row must resolve to the reference filename. |
| `fraction` | Required finite number in `(0, 1]` with `round(fraction * 400) >= 1`; retained fraction of native sampling intervals. |

These are the only fields inside `partial`. Paths resolve from the invocation
directory. The bundled demo uses one reference CIF and its task YAML. CIF-only
input uses the task's design-chain labels, fixed identities, motifs, required
bonds and offsets. Keep labels consistent with the reference. To restore native
saved locator or motif mappings, supply `result_csv` explicitly; invalid or
missing explicit metadata is an error.

The task YAML supplies permissions, conditions, motifs, context and topology.
The CIF supplies the current complete complex for both the reference state and
coordinates at existing condition sites, using the same native centering and
unit transform. Free design positions gain no extra conditions. The original
source remains the independent motif evaluation reference; see
[conditions and reference geometry](partial.md#conditions-and-reference-geometry).

`sample_num` requests replicas of one reference. Actual lengths come from the
reference, so `design_length` is not required when every chain's size is resolved.
CIF-only contigs use the task's source input, exact source spans and gap lengths;
resolve ambiguous random gaps in the task YAML. Explicit native CSV
metadata can restore saved placement. No length or gap draw is repeated.
This route requires a flex-context checkpoint.
`fraction: 0.625` runs the last 250 intervals of the native 400-interval grid,
starting at index 150; it is not a numeric start time or an RMSD bound.
The [partial guide](partial.md) gives the command, optional metadata and outputs.

`partidesign pipeline` reads this same native task YAML and selects names with
`--task`, just like `design`. Keep workflow settings in a separate
`configs/pipeline/` YAML selected through `--preset` or `--config`.
The iteration operation
`partial: {fraction: 0.625, replicas: 4, include_parent: true}` controls selected
parents between rounds; the pipeline creates the same task-level `partial`
reference input in memory. See the [pipeline guide](pipeline.md).

## Conditions

`condition` accepts `sequence`, `backbone`, `contig`, `secondary_structure`,
`ligand`, and `bond`. Combine compatible conditions to state identity, geometry,
and topology separately. Contradictory conditions fail rather than silently
replacing one another. The snippets below belong inside a normal task.

### Sequence identity

A complete string fixes sequence and determines length:

```yaml
condition:
  sequence:
    R: GGCUCUCACUGGCC
```

Use standard one-letter amino acids for protein/peptide, `A/C/G/U` for RNA, and
`A/C/G/T` for DNA. A sparse list fixes only the named positions:

```yaml
condition:
  sequence:
    P:
      - {resid: 2, residue: C}
      - {resid: -2, residue: C}
```

`resid` addresses the **final design chain**, starting at 1; `-1` is the last
residue and `-2` the penultimate residue. Leave positions out of a sparse list
when their identities should be generated. Fixing identity does not fix geometry.

Bracketed CCD components, such as `A[MK8]G`, express known modified residues.
Each bracketed component counts as **one residue**, and its required chemistry
must be available locally. See the modified-residue examples in
[peptide.yaml](../configs/examples/peptide.yaml).

### Backbone and contig

A backbone condition requires an existing input design chain:

```yaml
condition:
  backbone: [B]
```

It retains representative centers (`CA` for protein, `C1'` for RNA/DNA) and
allows standard residue identities to change. It does not retain every atom's
coordinates. Source modified residues retain their known chemistry.

Use a contig to copy selected source residues as coordinate motifs:

```yaml
condition:
  contig:
    B:
      token: "10-40/B19-27/10-40"
      polymer_atom: {B19: [CA, C, O]}
```

Segments separated by `/` are generated gaps (`10` or inclusive `10-40`) or
source residues (`B19`, `B19-27`) from the same design chain. This example copies
nine residues and adds 10–40 residues on either side, for a total of 29–89.
To restrict the final length, add **both** `motif_min_length` and
`motif_max_length` inside the chain's contig mapping.

`polymer_atom: null` keeps all available copied atoms. A mapping restricts only
the named copied residues; other copied residues retain all available atoms.
Each mapping key must identify one copied residue, and each atom must exist and
have valid source coordinates. Protein residues may use `tip` as the mapping
value for their side-chain tip preset; `polymer_atom: tip` by itself is invalid.
Backbone and contig are mutually exclusive for one chain.

Motif coordinates guide generation; the selected motif atoms are written at
the model-predicted positions in the source frame. They are not overwritten
with exact source coordinates, so evaluate motif agreement in the results.
Known copied identities are fixed even when only some atoms are conditioned.
For a source `UNK` motif, the model predicts amino-acid identity while retaining
its coordinate conditions. Coordinate retention does not fix identity to `UNK`.
The [motif fixture](../configs/examples/motif.yaml) and
[RNA motif examples](../configs/examples/nucleic_acids.yaml) show full task files.

### Secondary structure

| Channel | Protein/peptide | RNA/DNA |
| --- | --- | --- |
| `coarse` | Whole-chain `alpha` or `beta` | Whole-chain `paired` or `unpaired` |
| `token` | Final ranges labelled `helix`, `sheet`, `loop` | Final ranges labelled `paired`, `unpaired` |
| `pair` | `adjacent`, `not_adjacent` between declared helix/sheet blocks | `paired`, `unpaired` between declared token ranges |

```yaml
condition:
  secondary_structure:
    coarse: {P: beta}
    token:
      P:
        sheet: "1-6,10-15"
        loop: "7-9"
    pair:
      adjacent: [["P1-6", "P10-15"]]
```

Pair endpoints require compatible token conditions. Protein pair selectors must
exactly match complete declared helix/sheet blocks on the same chain; loops
cannot be pair endpoints. RNA/DNA paired relations may span chains, but
`unpaired` relations stay within one chain. Ligands do not accept these conditions.
These are fold guidance, not guarantees. See `rna_ss_guided` and `dna_ss_guided`
in [nucleic_acids.yaml](../configs/examples/nucleic_acids.yaml).

### Ligand identity modes

Ligand lengths count **heavy atoms**, excluding hydrogens. Choose one condition
per design ligand chain:

| Goal | Authoring | Length requirement |
| --- | --- | --- |
| Generate chemistry and pose | No ligand condition | `L: [count]`, a grid, or `L: from_distribution`. |
| Generate a known SMILES ligand's pose | `ligand: {L: {mode: smiles, smiles: "O=C(O)C1=CC=CC=C1"}}` | Graph determines count; omit `design_length`. SMILES must be connected. |
| Generate a known CCD component's pose | `ligand: {L: {mode: ccd, ccd: WHL}}` | Graph determines count; chain must be absent from input. |
| Regenerate a source ligand's pose | `ligand: {L: {mode: type_bond}}` | Retain source elements/graph; omit `design_length`. |
| Retain source centers, generate chemistry | `backbone: [L]` | Source fixes count; omit `design_length`. |
| Retain selected source atoms, generate the remainder | `ligand: {L: {mode: atom_motif, atoms: [C24, C21, O3]}}` | Sampling heavy-atom count, at least the selected atom count; use an explicit count. |
| Grow a fragment at specified scaffold atoms | `atom_motif` with `attachments`; see below. | Explicit sampling heavy-atom count, including retained atoms. |

Source-based ligand modes and decoration require the design ligand in `input`.
Ligand backbone is expressed
through `condition.backbone`, not a ligand mode named `backbone`.
`from_distribution` samples an unresolved count for exactly one ligand design
chain and requires an input structure for pocket sizing. It cannot be combined
with a fixed graph/backbone or multiple design chains. Source/supplied bonds
remain authoritative during reconstruction. See
[ligand.yaml](../configs/examples/ligand.yaml) for full tasks.

### Ligand decoration

Use `atom_motif` with `attachments` to retain a source scaffold and grow new
chemistry from one or more specified atoms. A complete product SMILES is not
required. For example, grow from two sites on a six-atom ring:

```yaml
design_length: {L: [12, 13, 14]}
condition:
  ligand:
    L:
      mode: atom_motif
      atoms: [C9, C13, C12, C11, N5, C10]
      attachments:
        - atom: C12
          new_atom: 1
          order: single
        - atom: C10
          new_atom: 2
          order: single
```

Supply the scaffold structure through `input` and declare `L` as a ligand
design chain. `design_length` sets the sampling heavy-atom count: these three
sizes retain six atoms and generate six, seven, or eight more. `sample_num` applies
to each size. Use one attachment for single-site growth; the single-site
example uses C9. `atom` must name one of the retained atoms.
`new_atom` is a **1-based endpoint index among the new atoms**, excluding the
retained scaffold. It is not the number of atoms to grow at that site; per-site
branch sizes are not configurable. It does not choose an element or require a
generated atom name. If
omitted, it defaults to the attachment's position in the list. `order` defaults
to `single`; supported values are `single`, `double`, `triple`, and `aromatic`.

With `attachments`, the selected scaffold's internal bonds and non-bonds are
preserved. Only the declared scaffold-to-new connections are allowed. Bonds
among new atoms, their elements, and their coordinates remain generated. With
two attachment sites, the generated atoms may form separate substituents or
bridge the sites; `new_atom` does not divide them into independent fragments.
Disconnected products use the existing valid-fragment selection. The retained
fragment must contain every motif atom and required bond endpoint, including
the declared attachment atoms. Unattached fragments can be discarded, so the
output may contain fewer heavy atoms than requested for sampling. With no
protected atoms, the largest chemically valid fragment is retained. Output
SMILES and `result.csv` describe the actual retained molecule.

The checkpoint must support bond-order and non-bond conditioning; incompatible
checkpoints are rejected. Preserving the output graph does not guarantee bond
geometry, motif placement, or useful molecular properties. Inspect generated
structures and their relaxation results. The runnable `ligand_decoration` and
`ligand_decoration_two_sites` tasks are in
[ligand.yaml](../configs/examples/ligand.yaml).

### Fixed SMILES ligand context

A ligand omitted from `design_chains` is retained context. Declare its trusted
chemistry using `condition.ligand.<chain>: {mode: smiles, smiles: "..."}` and keep
its actual pose in `input`. OpenMM uses that declaration when its complete
element and bond graph matches the retained component, including during
recovery of saved CIFs. A custom residue name such as `LIG01` does not require a
CCD entry. Keep the original SMILES declaration in the task YAML used for
[relaxation](relax.md); a mismatched graph is an error.

### Explicit bonds

Declare connections that must appear in output using addressable atoms:

```yaml
condition:
  bond:
    - atom1: {chain: P, resid: 1, atom: N}
      atom2: {chain: P, resid: -1, atom: C}
      order: single
```

Polymer endpoints need `chain`, `resid`, and `atom`; optional `ins_code`
disambiguates source residues. Design positions use final numbering, including
negative positions; context positions use positive source identifiers. Ligand
endpoints use `chain` and a stable `atom` name, with no `resid`.

Side-chain endpoints need known identity, usually fixed by `sequence` or a
contig. Identity-independent design backbone atoms can be used without fixing
sequence. Designed ligand endpoints must be known atoms in `type_bond`, `ccd`, or
the selected part of `atom_motif`; generated atoms and SMILES indices are not
stable endpoint names. For a connection from a retained ligand atom to a new
free atom, use [decoration attachments](#ligand-decoration).

Use `single`, `double`, `triple`, or `aromatic` for known order. Omitted order,
`any`, or `unknown` asserts existence without choosing an exact order; unresolved
order can be written as unknown in CIF. Identical endpoint/order declarations
collapse; conflicting orders fail. Bond topology does not guarantee optimized
geometry or stereochemistry.

For head-to-tail peptides, the terminal C–N bond and `cyclic_offset: true` are
independent settings. For disulfides, fix the cysteines and declare SG–SG bonds.
The [peptide examples](../configs/examples/peptide.yaml) also show staples and
bridges. Bond-local `omit_atoms` uses endpoint-style selectors for leaving atoms
in eligible design endpoint components; context, bond endpoints, hard motif
atoms, and free/partial ligand chemistry cannot be removed this way.

## Selectors, crop, and hotspots

Source selectors address chains (`A`), residues (`A33`), ranges (`A95-97`),
insertion codes (`A10A`), or multi-character chains (`[RNA1]9-16`). `/` joins a
union. `hotspot: A33/A95-97` marks the intended context interface without fixing
coordinates or redesigning those residues.

```yaml
crop:
  include: [A]
  proximity:
    - {anchor: A33/A95-97, radius: 12.0}
  exclude: [W]
hotspot: A33/A95-97
```

`include` selects the starting set; omitted `include` starts with the whole source.
`proximity` **adds** residues within the radius in ångströms of source heavy atoms;
it does not narrow an omitted include. `exclude` removes residues afterward.
Existing design chains, hotspots, and context bond endpoints are mandatory.
Connected chemistry may retain additional residues. Excluding mandatory residues
or cutting required chemistry is an error. A de novo chain cannot be a source
proximity anchor. Only retained context reaches the model.

## Token omission policy

The shipped **global** `rm_token_type: [CYS]` avoids cysteine at free protein
positions and overrides task lists. RNA `C` and DNA `DC` are distinct tokens.
Set global `rm_token_type: null` to respect each task's policy: an omitted/`null`
task value then uses `[CYS]`, an explicit list replaces it, and `[]` disables
omission. Global `[]` disables omission for every selected task.

Names are canonical tokens, such as `CYS`, `MET`, `C`, and `DC`, and are
case-insensitive. Fixed identities, context, modified components, and
coordinate-conditioned positions are protected. Unknown motif identities remain
predictions despite that exemption. MPNN redesign has a separate one-letter
`rm_aa` policy; changing native token omission does not change MPNN omission.

## Antibody tasks

Antibodies use `format: antibody` and `antibody` instead of normal
`design_chains`, `design_length`, and `condition` fields. Select antibody tasks
in a separate command from normal tasks. This complete example designs VH CDR3
on a built-in variable-domain scaffold:

```yaml
my_antibody:
  format: antibody
  sample_num: 2
  antibody:
    scaffold: adalimumab.6cr1
    chains: {VH: VH, VL: VL}
    template: true
    design:
      VH:
        cdrs: [CDR3]
        lengths: {CDR3: native}
      VL: {}
  antigen:
    input: ./configs/examples/structures/antibody_design/7uxq.cif
    crop: {include: [A]}
    hotspot: A20/A30-32
```

`format`, `sample_num`, and `antibody.scaffold` are required. `antigen` is
optional; crop/hotspot require its local `input`. Use the full built-in scaffold
name, such as `adalimumab.6cr1`. Built-in paired scaffolds contain VH/VL variable
domains; they do not add constant domains or a scFv linker.

`chains` maps the scaffold's exact roles to unique output chain IDs. Paired
scaffolds use `VH`/`VL`; single-domain scaffolds use `VHH`. If omitted, output IDs
are `H` for VH/VHH and `L` for VL. They must not collide with retained antigen
chains. Built-in scaffold definitions are in
[`configs/resources/antibody/scaffolds/`](../configs/resources/antibody/scaffolds/).

| Authoring choice | Effect |
| --- | --- |
| Omit `design` | All three CDRs of every role are mutable; framework identity stays fixed. |
| `design: {}` | Fix all role sequences at native lengths. |
| `VH: {cdrs: [CDR3]}` | Only VH CDR3 is mutable; unmentioned roles are fixed. |
| `VL: {}` | Fix the full VL sequence. |
| `framework: true` within a role | Also allow its framework identities to change. |
| `framework: "5/10-12"` within a role | Open only these source framework positions; CDR positions cannot be selected here. |

`lengths` applies only to selected CDRs. Values are `native` (default), `official`
(the scaffold's insertion/deletion range), a positive final count, or an inclusive
`{min: N, max: M}` interval sampled per candidate. Framework selectors use source
`label_seq_id` within the selected role, not antibody numbering conventions or
final output positions. Mutability allows a change; it does not require one.

`template: true` (default) supplies framework geometry as a soft template;
`false` removes it. Neither setting fixes framework coordinates. CDR coordinates
are generated, including CDRs whose sequence is fixed. Top-level `score`,
`postprocess`, `override`, and `rm_token_type` follow the normal task defaults.
Normal contig, bond, and cyclic fields are not accepted by this schema. Compare
`antibody_fixture`, `pdl1_adalimumab_selective_cdr3_official`, and
`pdl1_adalimumab_reference` in [antibody.yaml](../configs/examples/antibody.yaml).
