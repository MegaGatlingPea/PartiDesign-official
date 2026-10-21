# MPNN sequence backends

MPNNs are sequence-generation backends for reward search and offline validation.
They are not part of plain FM inference. Choose them explicitly through a reward
configuration or a validation `--source` entry.

Every MPNN backend starts with `rm_aa: C`. This is a protein one-letter omission
policy: the NA-MPNN backend carries the same protein-only default for a uniform
backend contract, but it does not remove RNA `C` or DNA `DC`. A task's fixed
sequence, motif, context, or explicit identity remains exempt from free-position
omission.

## Reward and validation names

Reward and validation use different public source namespaces. For design rewards,
the common `reward.sequence.source` values are `design`, `auto`, `proteinmpnn`,
`ligandmpnn`, and `cyclicmpnn`. ESMFold2 and AF3 reward validators also expose
`na_mpnn` for compatible nucleic-acid candidates; PartiScore does not. The reward
namespace does not accept the validation-only `soluble_mpnn` or `solublempnn`
source aliases.

The shipped reward files use `sequence.proteinmpnn` with `model_type: protein_mpnn`,
`use_soluble_model: false`, and the original/vanilla `v_48_020.pt` checkpoint.
Ordinary ProteinMPNN defaults to temperature `0.0001`, matching the
[ProtDBench binder configuration](https://github.com/congliuUvA/ProtDBench/blob/52e30cb6c9e6580286e5e014f887038c00368d23/protdbench/protd_configs/eval.py#L39-L45).
SolubleMPNN remains an explicit alternative: select `soluble_mpnn`, set
`use_soluble_model: true`, and use the checkpoint below `checkpoint/soluble/`.

Validation takes canonical sources `protein_mpnn`, `soluble_mpnn`, `ligand_mpnn`,
`cyclic_mpnn`, and `na_mpnn`, plus compact aliases such as `proteinmpnn` and
`solublempnn`. Each source also has a `-fi` fixed-interface form. In validation,
`proteinmpnn` selects vanilla ProteinMPNN and `solublempnn` selects soluble
ProteinMPNN. The `protein_mpnn` and `soluble_mpnn` backend blocks use the same
default `rm_aa: C`. The validation backend block is built for the selected source
and is separate from the reward backend block.

The backend block supplies the checkpoint, temperature, noise, bias, PSSM/tied
controls, and any applicable packing flags. A validation alias selects the
corresponding backend family; it does not silently switch to another source.

## Unified amino-acid omission

The canonical backend option is `rm_aa`. It is a string of uppercase standard
protein one-letter codes, such as `C`:

```yaml
reward:
  sequence:
    proteinmpnn:
      rm_aa: C
    ligandmpnn:
      rm_aa: C
    cyclicmpnn:
      rm_aa: C
    na_mpnn:
      rm_aa: C
```

`omit_aas` remains accepted for existing YAML. Resolution within one settings layer
is deliberately simple:

| `rm_aa` | Effective omission |
| --- | --- |
| Present with a non-null string | That string; it wins over `omit_aas`. |
| Present as `""` | Empty omission; it clears a legacy `omit_aas` value. |
| Omitted | The inherited backend policy, normally `C`; a same-layer legacy-only `omit_aas` overrides it. |
| Explicit `null` | Fall back to effective legacy `omit_aas`; an empty legacy string disables omission. |

The same rule is used by reward and validation backend settings. Values are
case-sensitive uppercase standard protein letters; `rm_aa: c` and unsupported
letters are invalid. For the nucleic-acid backend, `rm_aa` remains a protein-only
omission control and is not translated into `A/C/G/U` or `A/C/G/T` removal.

`rm_aa: ""` explicitly disables the default C omission. This setting applies to
free sequence positions. It does not remove task-fixed identities, contig residues,
explicit CCD carriers, or Cys positions required by a bond. It also does not change
the global/task `rm_token_type` policy used by the FM token decoder; see
[Configuration](configuration.md).

## Reward configuration

The selected reward config retains backend-level settings when a source is chosen:

```yaml
reward:
  _target_: src.design.rewards.esmfold2.ESMFold2Reward
  structure_source: fold
  sequence:
    source: proteinmpnn
    proteinmpnn:
      model_type: protein_mpnn
      checkpoint: ./community_models/proteinmpnn/checkpoint/vanilla/v_48_020.pt
      temperature: 0.0001
      backbone_noise: 0.0
      omit_aas: ""
      rm_aa: C
      bias_aa: {}
```

The shipped reward configs use `source: design`, so a reward keeps the generated FM
sequence unless you override it. `reward.sequence.source=auto` is also supported;
it selects an applicable configured backend while retaining that backend's own
settings. Every profile carries the explicit `sequence.proteinmpnn` slot; the
runtime records the selected effective settings in `sequence.auto_proteinmpnn`
before applying them to the automatic backend. Therefore an explicit
`reward.sequence.proteinmpnn.model_type`, checkpoint, temperature, or `rm_aa`
override remains effective for `source=auto`.

Native FM omission and MPNN omission are separate policies. A native task or
global `rm_token_type: []` means no token identities are omitted. An MPNN
`rm_aa: ""` means no one-letter amino acids are omitted; it clears inherited
legacy `omit_aas`. Fixed identities, motifs, and explicit chemistry remain
protected in both cases. Use `reward.sequence.source=design` explicitly when you
want that behavior to be visible in a command. Fold and PartiScore rewards
require Best-of-N; a design reward still needs `search=best_of_n` when it ranks
replicas.

For an explicit protein redesign from the installed CLI, select the reward backend
and override its canonical omission field directly:

```bash
partidesign design configs/examples/protein.yaml \
  --task protein_fixture \
  --run-name protein_mpnn_ranked \
  --override search=best_of_n \
  --override reward=esmfold2 \
  --override search.best_of_n.replicas=2 \
  --override reward.sequence.source=proteinmpnn \
  --override reward.sequence.proteinmpnn.rm_aa=C
```

## Validation source selection

Fold validation defaults to one original sequence plus one SolubleMPNN sequence
for generic protein and cyclic tasks. Nucleic-acid tasks use one original
sequence plus one NA-MPNN sequence. Use an explicit ordered source list to
replace those defaults. `auto` is not supported by validation; it is a reward
source for automatic backend selection (including compatible NA design tasks):

```bash
cd /path/to/partidesign-inference
CUDA_VISIBLE_DEVICES=0 partidesign validate ./inference/guide_protein \
  --method esmfold2-fast \
  --source gt:1 \
  --source solublempnn:1
```

The validation parser inserts one `gt`/`self` slot when the requested list contains
only MPNN sources. An explicit `gt:1` requests that same single original slot.
MPNN slots follow in the order given and receive stable source labels. GT must be
exactly one slot; `gt:2` is invalid. `-fi` enables the fixed-interface variant, for example
`proteinmpnn-fi:2`, with the default 6 Å heavy-atom cutoff. It fixes design residues
near the selected partner while leaving non-interface design residues eligible for
redesign.

The CLI `--source` list replaces the complete configured source list. It does not
replace backend blocks: backend-level checkpoint, temperature, `rm_aa`, PSSM, and
packing settings remain available. Source-local settings embedded in the old YAML
source list are replaced by the explicit CLI source entries. Put a setting that
must survive `--source` in the backend block; use a validation config file when a
source needs its own settings.

Validation resolves omission layers in this order: the backend default `C`, then
the selected backend block, then the source-local settings. At each layer, a non-null `rm_aa`
wins over that layer's legacy `omit_aas`; `rm_aa: null` uses that layer's
`omit_aas`. A higher-layer legacy `omit_aas` is itself authoritative, so a source
with `omit_aas: M` overrides an inherited backend `rm_aa: C`. A source-local
`rm_aa: ""` explicitly clears the inherited omission. A source-local `rm_aa: null`
alone falls through to the merged legacy value and does not clear it.

The equivalent YAML keeps source-specific labels and fixed-interface settings:

```yaml
methods: [esmfold2-fast]
sequence:
  protein_mpnn:
    rm_aa: C
    omit_aas: M
  sources:
    - {source: gt, label: self, count: 1}
    - {source: proteinmpnn, label: mpnn, count: 2}
    - source: proteinmpnn-fi
      label: interface
      count: 2
      settings:
        omit_aas: M
        fixed_interface:
          enabled: true
          cutoff: 6.0
          target_chains: [A]
```

The backend block resolves to `C`; the source-local `omit_aas: M` resolves that
source to `M`, because the higher source layer supplied the legacy field.

The normalized source aliases are `protein_mpnn`, `soluble_mpnn`, `ligand_mpnn`,
`cyclic_mpnn`, and `na_mpnn`, with compact aliases accepted. Duplicate methods or
normalized source labels are rejected.

## Temperatures and defaults

Ordinary ProteinMPNN defaults to temperature `0.0001`, matching ProtDBench's
binder protocol. SolubleMPNN and NA-MPNN retain their independent `0.1` defaults.
LigandMPNN uses `0.1` with ligand-atom and catalytic side-chain context enabled,
matching the [AME sequence sampler](https://github.com/RosettaCommons/RFdiffusion2/blob/main/rf_diffusion/benchmark/mpnn_designs_v2.py#L177-L190).
Packing remains a separate downstream operation for the paths that consume it.
CyclicMPNN defaults to `0.0001`, matching its PartiFlow-style refolding route.
Its dedicated JAX checkpoint is Git-tracked at
`community_models/cyclicmpnn/checkpoint/cyclicmpnn_48_010.pkl` and is supplied by
cloning PartiDesign; no additional CyclicMPNN checkpoint download is needed for a
source-checkout installation. It does not reuse the vanilla or soluble
ProteinMPNN weights. An explicit backend temperature always wins; these defaults do
not override a user setting.

Backbone noise defaults to zero for the pinned sequence routes. LigandMPNN's atom
context and side-chain context are enabled by its backend config, while side-chain
packing and full repacking remain opt-in where the selected reward path supports
them. A task's hard sequence, topology, and motif conditions are passed to the
backend as constraints.

## Residency and ownership

For design rewards, a fold backend may remain resident only for the owner batch that
is currently being processed; it is closed between owner batches. This avoids
claiming that the whole inference model stack stays resident for the entire run.

Validation has a separate queue: each Ray validation worker owns one backend, stays
hot while it receives its owner queue, and closes at the end of the validation
run. See [Validation](validation.md).

## Sampling batches, seeds and resume

ProteinMPNN (including Soluble and CA-only) and LigandMPNN sample compatible
copies of a backbone in one native batch. Their grouping checks coordinates and
the applicable fixed, omission, PSSM or tied controls; serialization names do
not split equivalent requests. Different constraints still form separate groups.
Within a LigandMPNN call, symmetry must resolve to one shared contract.
NA-MPNN and CyclicMPNN instead batch compatible shapes and seeds,
retaining each row's own constraints. Each returned slot receives its own native
draw and its original source index. A low temperature can legitimately produce
repeated sequences, so `count: 8` requests eight draws rather than eight guaranteed
unique sequences.

Validation uses seed `101` unless configured otherwise. Its precedence is the
source-local `settings.seed`, selected backend `seed`, then `sequence.seed` and
the default. Zero is a valid explicit seed. Repeating the same source, structure,
constraints, count and batch layout reproduces its ordered draws, including the
first cold call and later hot calls. This is not a promise of sample-index-stable
results after changing the count or grouping.

A partial validation resume replays the original source count and saves only the
missing slots. This preserves the missing slot's original draw instead of
assigning it draw zero. A fully completed source does no sampling. Existing
completed slots are reused; they are not automatically invalidated by an update.
To replace results generated with the former batching bug, explicitly rerun the
requested validation sources, for example:

```bash
partidesign validate ./inference/my_designs \
  --method af2_ig \
  --source gt:1 --source proteinmpnn:8 \
  --rerun
```

This regenerates the requested sequence slots and their validation, without
resampling FM structures. Keep the benchmark's original backend, temperature and
criteria when rerunning its results.

Antibody validation reconstructs VH/VL/VHH chain roles and mutable positions from
the final result CSV. Its MPNN plan protects unselected framework residues, other
unselected regions, context and explicit user-fixed positions while jointly
designing the selected mutable chains. Explicitly selected framework positions
can still be redesigned. The existing shared planner still rejects antibody
`-fi`; this update does not add that mode or paired-antibody AF2-IG support.

Persistent workers keep their model and checkpoint resident throughout their
owner run. Sampling seed, temperature and omission are request controls; changing
those does not reload the same model. Model/weight changes still select the
appropriate resident model. Constructor randomness is isolated from the CPU
sampling stream, and cyclic per-row retries retain their original draw keys.
These fixes do not change FM inference seeding or the configured temperatures.
