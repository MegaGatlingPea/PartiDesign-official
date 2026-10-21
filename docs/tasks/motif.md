# Motif and scaffold tasks

Motif tasks keep selected source coordinates while generating the rest of a design
chain. They use the normal task schema; the motif-specific field is
`condition.contig`. Use [Task schema](../task_schema.md) for the complete selector,
length, bond, crop, and omission rules.

## Runnable residue motif fixture

The repository fixture
`configs/examples/structures/protein_design/prot_monomer_motif_aa.pdb` contains a
source motif on chain `B`; the external task is:

```bash
cd /path/to/partidesign-inference
partidesign design configs/examples/motif.yaml \
  --task motif_fixture \
  --batch-size 1 \
  --run-name motif_fixture \
  --show-config

CUDA_VISIBLE_DEVICES=0 partidesign design configs/examples/motif.yaml \
  --task motif_fixture \
  --batch-size 1 \
  --run-name motif_fixture
```

The external task in `configs/examples/motif.yaml` uses contig
`10-40/B19-27/10-40`, which samples the two gaps and copies source residues
B19–B27. It keeps the `CA`, `C`, and `O` coordinates for copied residue B19 via
the valid `polymer_atom` mapping.

## Contig grammar

Each contig segment is one of:

| Segment | Meaning |
| --- | --- |
| `N` | Fixed gap of `N` newly generated residues. |
| `N-M` | Inclusive sampled gap range. |
| `B19` or `B19-27` | Source residue or source range copied from the same design chain. |

Join segments with `/`:

```yaml
condition:
  contig:
    B:
      token: "10-40/B19-27/10-40"
      motif_min_length: 45
      motif_max_length: 65
      polymer_atom: null
```

The final length window applies to the sum of fixed source segments and all sampled
gaps. A contig requires an existing input design chain. It cannot be combined with
`condition.backbone` for that chain or with a conflicting `design_length`.

## Atom scope

`polymer_atom` controls which copied coordinates become hard motif coordinates. The
field itself is `null` or a mapping; `polymer_atom: tip` is invalid because `tip`
must be the value for a copied-residue selector:

```yaml
polymer_atom: null                 # all copied source atoms
polymer_atom:
  B19: [CA, C, O]
  B20: [CB, CG]

polymer_atom:
  B19: tip                         # residue-specific side-chain tip preset
  B20: tip
```

Known copied token identities remain fixed for a contig. For an unknown source
residue, the decoder infers a standard identity while keeping its coordinate motif
conditions. Coordinate motif positions are exempt from global/task `rm_token_type`
and MPNN `rm_aa`, including unknown-identity motifs. This exemption leaves the
identity mutable; it does not freeze it as `UNK`.

## Hotspots and crop

Motifs and hotspots address different objects. `condition.contig` supplies design-chain
coordinate conditions; `hotspot` marks retained context residues for interface guidance. Use
`crop.include`, `crop.proximity`, and `crop.exclude` to define which context reaches
the model. Context that is removed by the actual crop does not affect routing or
reward applicability.

Coordinate motifs normally select `omni` routing because they require source
coordinate preservation. A motif-free protein task can use `prot`; an explicit
peptide motif uses the peptide intent and may select `pep` or `omni` according to
its complete design unit.
