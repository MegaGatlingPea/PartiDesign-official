# Nucleic-acid tasks

Declare RNA and DNA design chains explicitly as `rna` and `dna`:

```yaml
design_chains:
  - {R: rna}
  - {D: dna}
```

The public RNA alphabet is `A/C/G/U`; DNA uses `A/C/G/T`. Use `N` for unknown RNA
and `DN` for unknown DNA where the task requires an explicit unknown token. The
shared length, sequence, backbone, secondary-structure, bond, crop, and hotspot
fields are defined in [Task schema](../task_schema.md).

## Runnable RNA fixture

The repository fixture `configs/examples/structures/nuc_design/nuc_bind_prot.pdb`
contains a protein context. The external task designs chain `R` at length 20:

```bash
cd /path/to/partidesign-inference
partidesign design configs/examples/nucleic_acids.yaml \
  --task rna_fixture \
  --batch-size 1 \
  --run-name rna_fixture \
  --show-config

CUDA_VISIBLE_DEVICES=0 partidesign design configs/examples/nucleic_acids.yaml \
  --task rna_fixture \
  --batch-size 1 \
  --run-name rna_fixture
```

The target hotspot is `E64/E88/E96`. The equivalent external mapping is in
`configs/examples/nucleic_acids.yaml`; change `R: rna` to `D: dna` and use
the DNA alphabet for a DNA design.

## Sequence and inverse folding

A complete sequence fixes every identity while the FM model generates structure:

```yaml
rna_sequence:
  input: null
  sample_num: 2
  design_chains: [{R: rna}]
  condition:
    sequence:
      R: GGCUCUCACUGGCC
```

For inverse folding, provide an existing design chain and `condition.backbone`:

```yaml
rna_inverse_fold:
  input: ./configs/examples/structures/nuc_design/rna_backbone_motif.pdb
  sample_num: 2
  design_chains: [{A: rna}]
  condition:
    backbone: [A]
```

Backbone conditions retain token-center geometry while allowing identity decoding.
They require an existing source chain and cannot be combined with a contig on the
same chain.

## Paired and unpaired guidance

Nucleic-acid secondary structure uses coarse, token, and pair guidance. For example,
the following makes a paired stem with an unpaired middle loop:

```yaml
condition:
  secondary_structure:
    coarse: {R: paired}
    token:
      R:
        paired: "1-4,9-12"
        unpaired: "5-8"
    pair:
      paired:
        - ["R1", "R12"]
        - ["R2", "R11"]
        - ["R3", "R10"]
        - ["R4", "R9"]
      unpaired:
        - ["R5", "R8"]
```

Pair entries are guidance, not a replacement for a concrete covalent bond. Declare
noncanonical or cross-chain bonds in `condition.bond` with addressable atom names.

## Protein/nucleic and ligand context

Protein–RNA/DNA joint design uses `design_length.joint` with complete declared
polymer chain keys. A protein context with a nucleic-acid design chain is routed to
`omni` when modalities are mixed. Nucleic-acid-only tasks prefer `nuc`. A retained
small-molecule context can also change routing and reward applicability; only the
ligand that survives the actual crop counts as context.

The NA-MPNN backend is available for explicit reward or validation sequence slots.
Validation source selection remains explicit (`na_mpnn`/`nampnn`); reward `auto` may
choose it when the task is compatible. See [MPNN](../mpnn.md) and
[Validation](../validation.md).
