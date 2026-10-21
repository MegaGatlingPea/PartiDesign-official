# Ligand tasks

Ligand design chains use `design_chains: [{L: ligand}]`. Public ligand lengths are
heavy-atom counts. Hydrogen atoms written in a SMILES string are chemistry input,
never model tokens or a public endpoint. Use the shared fields in
[Task schema](../task_schema.md) for source selection, crop, hotspots, and bonds.

## Runnable ligand fixture

The repository includes a KRAS pocket at
`configs/examples/structures/ligand_design/lig_bind_prot.cif`. The external task uses a
sampled heavy-atom count and an explicit interface hotspot:

```bash
cd /path/to/partidesign-inference
partidesign design configs/examples/ligand.yaml \
  --task ligand_fixture \
  --batch-size 1 \
  --run-name ligand_fixture \
  --show-config

CUDA_VISIBLE_DEVICES=0 partidesign design configs/examples/ligand.yaml \
  --task ligand_fixture \
  --batch-size 1 \
  --run-name ligand_fixture
```

OpenMM is a separate stage; the external fixture sets `postprocess: false` for a
base-generation smoke run. Enable it when the OpenMM dependencies and chemistry
assets are available. The equivalent task mapping is in
`configs/examples/ligand.yaml`.

## Identity modes

Choose one mode for a design ligand chain:

```yaml
condition:
  ligand:
    L:
      mode: smiles
      smiles: "O=C(O)C1=CC=CC=C1"
```

The supported modes are:

- `smiles` fixes one connected RDKit molecular graph and generates its pose. Salts
  and disconnected SMILES fragments are rejected.
- `ccd` fixes one uppercase CCD component and its known heavy-atom graph.
- `backbone` retains an existing ligand atom inventory and source positions while
  hiding element identity for reconstruction.
- `type_bond` retains the complete source element and internal bond graph, including
  source covalent edges. It is the pose-only mode for a known ligand.
- `atom_motif` retains selected source atom names and samples the remaining heavy
  atom count. Supply `design_length` for the final count.

`smiles`, `ccd`, and fixed-graph modes are mutually exclusive. A fixed mode fixes
the atom count, so it cannot be paired with `design_length`; free ligand generation
uses `L: from_distribution` or an explicit heavy-atom count. A design ligand chain
must resolve to one connected component.

## Bonds and covalent ligands

Source and supplied ligand bonds are hard output facts. A graph reconstruction method
cannot replace an explicit type/bond graph or an input covalent edge. For an explicit
new covalent connection, expose both endpoint identities and declare the order:

```yaml
condition:
  ligand:
    L: {mode: type_bond}
  bond:
    - atom1: {chain: A, resid: 145, atom: SG}
      atom2: {chain: L, atom: C7}
      order: single
```

Ligand endpoints use stable atom names and do not take `resid`. Polymer endpoints
use one-based design positions (negative positions count from the design-chain end)
or source identifiers for context. Duplicate unordered endpoint declarations with the
same order collapse; order conflicts fail. For a complete fixed design ligand
(`ccd`/`type_bond`), bond-local `omit_atoms` may remove named leaving atoms
from the ligand endpoint component. Compilation first matches the complete
precursor to trusted CCD chemistry, then projects its explicitly declared
product. Assembly keeps the reduced inventory; output uses a local `LIGxx`
component ID and its actual internal bonds. An unmodified fixed ligand keeps
its existing CCD identity. Context, bond endpoints, hard coordinate motifs,
and free/partial ligand chemistry remain ineligible for omission.

Private isolated fragments use hydrogen caps for parameterization; whole-product
preparation consumes those hydrogens at the explicit external bonds.
Existing-output recovery reads the original task's bond-local omissions and
checks the retained graph against the same projected CCD component.

## Reconstruction and postprocess

The ligand graph method is selected in the sampling config. The current packaged
default is `obabel`, as shown by `--show-config`; no other graph method is a silent
fallback. A supplied graph remains authoritative under the configured method.

OpenMM postprocess is optional and runs after successful base CIF output. The
production path is CUDA plus Ray; CPU plus no Ray is a focused diagnostic. Use
`gasteiger` by default or select `am1bcc` explicitly in the top-level postprocess
configuration. See [Configuration](../configuration.md) and [Outputs](../outputs.md).

## Ligand validation

Use ESMFold2/AF3 methods for compatible ligand binder structures. `vina_dock` is a
separate CPU-only validation method: it accepts GT-only input, requires successful
native postprocess, cannot be mixed with another method, and does not reuse refold
metrics. See [Validation](../validation.md).
