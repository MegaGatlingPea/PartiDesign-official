# Rewards and search

Generation and candidate ranking are separate stages. A plain normal task runs the
FM sampler and writes its generated sequence and structure. Add a search algorithm
and a reward when several candidates should be generated for each base request and
the best candidate should be published.

## Best-of-N

Best-of-N is selected explicitly:

```bash
cd /path/to/partidesign-inference
CUDA_VISIBLE_DEVICES=0 partidesign design \
  configs/examples/protein.yaml \
  --task protein_fixture \
  --batch-size 1 \
  --run-name reward_demo \
  --override search=best_of_n \
  --override reward=esmfold2 \
  --override search.best_of_n.replicas=4 \
  --override reward.sequence.source=design
```

`replicas` is applied after task, condition, and length resolution. One base request
with four replicas produces four candidates. The current runtime scores and writes
all successful ranked replicas in the task leaf, with rank-qualified structure names
and `global_rank` values (`rank1` is the highest score); it does not collapse the
result to one winner row. Search has one selection scope: the task plus its
resolved length plan. It does not pool candidates across unrelated tasks or length
plans.

The same submission uses one routed checkpoint member for all selected tasks. A
submission requiring several member preferences is routed to `omni` as one whole
submission; it is not split automatically by model member. `search.max_batch_size`
can bound reward work, but it does not change replica counts or ranked-row
accounting.

## Reward choices

The supported design reward targets are:

| Reward | Typical use | Structure source |
| --- | --- | --- |
| `clash` | Fast continuous geometry ranking with Best-of-N | Generated structure |
| `esmfold2` | Refolded protein, peptide, or compatible mixed-chemistry candidate ranking | `fold` or explicit `design` |
| `antibody_esmfold2` | Refolded Fab or VHH candidate ranking | `fold` |
| `af2` | AF2 initial-guess binder ranking | Refolded structure |
| `af3` / `af3-la` | AF3 or AF3-LA complex ranking | Refolded structure |
| `partiscore` | Learned structure likelihood ranking | Refolded structure |

ESMFold2 (including `antibody_esmfold2`), AF2, AF3, AF3-LA, and PartiScore rewards
require `search=best_of_n`. A null search must use a null reward. The clash reward is the
lightweight Best-of-N option for continuous geometry ranking.

The shipped reward configs default to `reward.sequence.source: design`, so a reward
uses the generated FM sequence unless you explicitly select a sequence backend.
`reward.sequence.source=auto` is also supported when you want the configured backend
router to choose an applicable sequence model. The shared sequence policy routes
protein and cyclic requests to SolubleMPNN and keeps nucleic requests on the FM
`design` sequence unless NA-MPNN is explicitly selected. Validation source selection
is different: validation requires an explicit ordered source list and does not use
`auto`; see [Validation](validation.md).

An ESMFold2-style reward can be configured with a design structure source only when
the sequence source is also `design`:

```yaml
search:
  algorithm: best_of_n
  selection_scope: task_length_plan
  max_batch_size: null
  best_of_n: {replicas: 4}
reward:
  _target_: src.design.rewards.esmfold2.ESMFold2Reward
  structure_source: fold
  sequence:
    source: design
  esmfold2:
    mode: fast
    loops: 20
    steps: 100
```

The full MPNN backend blocks remain in the selected reward config. A CLI override of
`reward.sequence.source` chooses which backend is used; it does not silently reset
the backend's checkpoint, temperature, fixed positions, omission, packing, or
interface settings. See [MPNN](mpnn.md).

The AF3 design reward evaluates one complex prediction sample per candidate. The
validation `af3.samples` setting is independent and may request multiple samples
per candidate for offline evaluation.

## Clash ranking

The clash reward is a continuous pairwise VDW distance ranking signal. It excludes
bonded 1-2 and angle-neighbor 1-3 pairs, retains 1-4 and farther pairs, and keeps
design-design and design-context terms according to their configured nonnegative
weights. It is a ranking score, not a hard pass/fail threshold and never turns a
clash observation into a generation failure by itself.

## Sequence source and packing

The FM candidate is the base structure. A reward may request a sequence backend and,
where that backend supports it, the applicable side-chain packing step. Packing is
used only where the selected reward path requires it; it is not a universal step
between FM generation and every reward. Direct MPNN-to-PartiScore ranking uses the
required packing path before scoring. Refold rewards pass the selected sequence and
context to the fold backend, then score the returned structure.

Fixed positions, task sequence conditions, contig identities, and explicit topology
remain hard constraints. A sequence omission policy cannot change a fixed Cys or an
explicit CCD identity. A reward failure is recorded against its candidate or sample
according to the normal result accounting; it does not silently fall back to a
different reward backend.

## Worker lifetime

Search work is queued by owner and candidate. A design fold reward may keep its fold
backend resident for the owner batch that is currently being processed, then closes
it between owner batches. This is a resource optimization for that backend; it does
not mean the whole inference process or every model remains resident for the entire
run. Full validation has a separate Ray worker lifecycle described in
[Validation](validation.md).

## Final scoring and postprocessing are separate

These three options have different meanings:

```yaml
search: {algorithm: best_of_n, best_of_n: {replicas: 4}}
reward: {_target_: src.design.rewards.esmfold2.ESMFold2Reward, ...}

my_task:
  score: true        # final PartiScore file for each successful base result
  postprocess: true  # optional OpenMM refinement
```

`score: true` does not enable search. `postprocess: true` does not rank candidates.
OpenMM refinement runs after base output and updates `postprocess_status`; it does
not change the reward definition. See [Outputs](outputs.md) for the resulting files.
