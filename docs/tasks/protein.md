# Protein tasks

Protein tasks use `design_chains: [{P: protein}]` (or another chain ID) and the
shared fields in [Task schema](../task_schema.md). A protein chain can be de novo
when it is absent from the input or redesigned when it is present. Other input
chains remain context.

## Runnable binder fixture

The repository includes a protein receptor and an external task fixture. From the
repository root, inspect and run one length:

```bash
cd /path/to/partidesign-inference
partidesign design configs/examples/protein.yaml \
  --task protein_fixture \
  --batch-size 1 \
  --run-name protein_fixture \
  --show-config

CUDA_VISIBLE_DEVICES=0 partidesign design configs/examples/protein.yaml \
  --task protein_fixture \
  --batch-size 1 \
  --run-name protein_fixture
```

The input is `configs/examples/structures/protein_design/prot_bind_prot.pdb`, and its
target hotspot is `E64/E88/E96`. The generated design chain is `P`; the receptor
chains stay context. The same task is available as an external mapping in
`configs/examples/protein.yaml`.

## De novo and redesign

An input-free monomer needs an explicit length:

```yaml
protein_monomer:
  input: null
  sample_num: 2
  design_chains: [{P: protein}]
  design_length: {P: [100]}
```

For a source-backed redesign, keep the design chain in the input and use
`condition.backbone` for inverse folding or `condition.contig` for a coordinate
motif. A complete sequence condition fixes identity while the model generates
structure; a sparse sequence condition fixes only the listed positions:

```yaml
protein_inverse_fold:
  input: ./configs/examples/structures/protein_design/prot_monomer_motif_aa.pdb
  sample_num: 2
  design_chains: [{B: protein}]
  condition:
    backbone: [B]
    sequence:
      B:
        - {resid: 2, residue: C}
        - {resid: -2, residue: C}
```

The backbone condition reveals representative centers and hides sequence identity.
The contig condition reveals copied coordinates and identity for its motif. Use
`polymer_atom` to narrow copied coordinates to a selected atom list or `tip`.

## Secondary structure and interfaces

Protein secondary-structure guidance can be coarse or token/pair level. The
condition applies to the declared design chain:

```yaml
condition:
  secondary_structure:
    coarse: {P: alpha}
```

Use `hotspot` for receptor residues that should define the target interface. A
hotspot is an interface annotation, not a fixed sequence or coordinate condition.
For protein–RNA/DNA or protein–ligand tasks, keep the non-design entity in `input`
and declare its design chain separately; routing then accounts for the retained
context.

## Length plans and routing

One chain can use a list or aligned `{min, max, step}` grid. Multiple design chains
must use an explicit `design_length.joint` table with one row for every polymer
chain in declaration order. A protein-only task normally prefers `prot`; a protein
coordinate motif, mixed polymer joint, or retained ligand context prefers `omni`.
An explicit peptide authoring alias is documented on the [Peptide](peptide.md) page
and selects `pep`.

Normal protein generation does not call ProteinMPNN automatically. Add Best-of-N
and a reward explicitly when you want sequence redesign or structure ranking; see
[Rewards](../reward.md) and [MPNN](../mpnn.md).
