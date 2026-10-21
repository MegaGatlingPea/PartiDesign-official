# Peptide tasks

`peptide` is an explicit authoring modality for protein chemistry with peptide
intent. It keeps the protein sequence alphabet and structural ABI, but routes the
design unit to the `pep` checkpoint member when automatic routing is enabled. A
short `protein` chain does not become a peptide by length, filename, disulfide, or
secondary-structure condition.

Use the shared [Task schema](../task_schema.md) for lengths, sequences, selectors,
and bonds. The peptide-specific choices are the cyclic offset and explicit peptide
topology.

## Runnable cyclic fixture

The external fixture uses a protein receptor at
`configs/examples/structures/peptide_design/pep_bind_prot.pdb`. The head-to-tail bond
and cyclic offset are independent settings and are both explicit in this example:

```bash
cd /path/to/partidesign-inference
partidesign design configs/examples/peptide.yaml \
  --task peptide_fixture \
  --route true --route-ckpt pep \
  --batch-size 1 \
  --run-name peptide_fixture \
  --show-config

CUDA_VISIBLE_DEVICES=0 partidesign design configs/examples/peptide.yaml \
  --task peptide_fixture \
  --route true --route-ckpt pep \
  --batch-size 1 \
  --run-name peptide_fixture
```

The equivalent external task is in `configs/examples/peptide.yaml`.

```yaml
peptide_fixture:
  input: ./configs/examples/structures/peptide_design/pep_bind_prot.pdb
  sample_num: 1
  design_chains: [{P: peptide}]
  cyclic_offset: true
  design_length: {P: [13]}
  condition:
    bond:
      - atom1: {chain: P, resid: 1, atom: N}
        atom2: {chain: P, resid: -1, atom: C}
        order: single
```

An explicit terminal bond is retained as a hard output fact. `cyclic_offset: true`
changes the cyclic sampling coordinate convention; it does not create a bond. A
linear peptide leaves `cyclic_offset: false` and omits the terminal bond.

## Disulfides, staples, and bridges

Declare disulfides as explicit `SG`–`SG` bonds and fix the corresponding residues:

```yaml
condition:
  sequence:
    P:
      - {resid: 2, residue: C}
      - {resid: -2, residue: C}
  bond:
    - atom1: {chain: P, resid: 2, atom: SG}
      atom2: {chain: P, resid: -2, atom: SG}
      order: single
```

Side-chain isopeptides, hydrocarbon staples, and CCD bridge carriers use the same
bond endpoint grammar. A bracketed CCD carrier such as `[WHL]` is a fixed chemical
component and may be a second `ligand` design chain. Declare every covalent bridge
that the output must retain; a topology condition does not guarantee the generated
coordinates are geometrically closed or chemically optimized.

Head-to-tail closure, disulfides, and staples are separate intents. Do not infer one
from another, and do not assume a cyclic offset alone produces a ring.

## Additional covalent product examples

The following two tasks are appended to `configs/examples/peptide.yaml`. Each
requests five samples. The original ten examples and the formal PepDBench tasks
retain their existing settings.

| Task | Fixed chemistry and required product bonds |
| --- | --- |
| `peptide_bind_protein_tbmb_bicyclic` | 18-residue peptide; Cys3/9/15 `SG` attach by single bonds to ligand T `C1/C2/CD`, respectively. |
| `peptide_bind_protein_double_hydrocarbon_staple` | 18-residue peptide; `[MK8]` at 3/7/11/15, with `CE=CE` bonds at 3–7 and 11–15; alpha conditioning. |

TBMB is a three-cysteine bicyclization chemistry
([Heinis et al., 2009](https://www.nature.com/articles/nchembio.184)).
This example authors its final product using the named carbon scaffold
[CCD SEZ, mesitylene](https://www.rcsb.org/ligand/SEZ): three benzylic carbon
anchors connect to the cysteines. Bromine leaving groups are absent from the
product. The benzene ring is retained in addition to the two peptide loops.

The double staple declares two carbon–carbon double bonds. Both examples
request postprocessing and use `cyclic_offset: false`; their side-chain bridges
do not create a head-to-tail peptide bond. The declared graphs do not establish
generated bond geometry or alkene stereochemistry. Model sampling and full
receptor–peptide minimization require separate runtime acceptance.

### Refold scope

AF2's existing eligibility checks exclude ligand carriers and modified polymer
residues, so the TBMB and MK8 examples do not enter the ordinary AF2 lane.
The current AF3 adapter can encode ligand–polymer bonds, but does not forward
explicit polymer–polymer crosslinks. AF3 refolds of the hydrocarbon staples
therefore cannot yet be treated as topology-preserving validation. No new
automatic refold branch is enabled by these examples.

## Peptide routing and rewards

Automatic routing selects `pep` for explicit peptide design units, including a
peptide plus a covalently connected known chemical carrier when its design bonds
make that relationship explicit. Mixed peptide/protein or peptide/nucleic joints
can require `omni`. Force a member with `--route-ckpt pep` only when the checkpoint
bundle contains that member and the task is compatible.

Normal peptide generation keeps the FM sequence by default. ProteinMPNN or
CyclicMPNN sequence redesign requires explicit Best-of-N reward/search configuration.
The CyclicMPNN eligibility check accepts either explicit `cyclic_offset: true` intent
or a required-output head-to-tail single bond; supplying both is allowed and keeps
the sampling convention and chemistry declaration independently visible.
The CyclicMPNN default temperature is `0.0001`; an explicit backend temperature
override remains authoritative. See [MPNN](../mpnn.md).
