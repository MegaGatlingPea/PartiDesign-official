# Antibody tasks

Antibody tasks use `format: antibody` and the `antibody` mapping. They describe a
VH/VL Fab variable region or a VHH single-domain design around a named scaffold.
The scaffold supplies the variable-domain layout and CDR windows; the antigen is
an ordinary structure input that can be cropped and annotated with hotspots.

Antibody and normal tasks are separate inference formats. Every task selected in
one submission must use the same format, so do not select a normal protein task
with an antibody task. Automatic routing chooses the `antibody` member for an
antibody-only submission. Use `--route-ckpt antibody` when that choice should be
visible in the command. The selected routing member is used for the whole
submission; it is not split into per-task model batches.

## Runnable Fab fixture

The repository contains an adalimumab variable-region scaffold and a PD-L1
fixture antigen. Run from the repository root:

```bash
cd /path/to/partidesign-inference
partidesign design configs/examples/antibody.yaml \
  --task antibody_fixture \
  --route true --route-ckpt antibody \
  --batch-size 1 \
  --run-name antibody_fixture \
  --show-config

CUDA_VISIBLE_DEVICES=0 partidesign design configs/examples/antibody.yaml \
  --task antibody_fixture \
  --route true --route-ckpt antibody \
  --batch-size 1 \
  --run-name antibody_fixture
```

`--show-config` checks task resolution without loading model weights or creating
structures. The external task uses
`configs/examples/structures/antibody_design/7uxq.cif`, retains antigen chain `A`, and
uses the `adalimumab.6cr1` scaffold. The corresponding external task is
`configs/examples/antibody.yaml`.

For a single-domain antibody, select `vhh_fixture` from the same external file.
It uses `chains: {VHH: VHH}` and an antigen crop on chain `B`:

```bash
partidesign design configs/examples/antibody.yaml \
  --task vhh_fixture \
  --route true --route-ckpt antibody \
  --batch-size 1 \
  --run-name vhh_fixture \
  --show-config
```

## Scaffold, roles, and design scope

The public task shape is:

```yaml
my_antibody:
  format: antibody
  sample_num: 1
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
  score: false
  postprocess: false
```

`scaffold` is the complete built-in scaffold name, such as
`adalimumab.6cr1` or `7eow`; the loader does not infer a shortened name.
`chains` maps antibody roles to output chain IDs. A Fab uses `VH` and `VL`; a
single-domain task uses `VHH` without `VL`. The output chain IDs must not collide
with retained antigen chains.

The scaffold's template YAML defines the source chains and CDR windows. The
numbers in a role-scoped framework selector are source `label_seq_id` positions,
not positions in the output chain. The built-in `adalimumab.6cr1` scaffold maps
VH to source chain B and VL to source chain A. Users select roles (`VH`, `VL`, or
`VHH`) and do not need to copy those internal source IDs into the task.

The `design` mapping controls sequence identity and length:

| Authoring | Effect |
| --- | --- |
| Omit `design` | All three CDRs of every role may be designed; framework identity stays fixed. |
| `design: {}` | Fix all role sequences and native lengths. |
| `VH: {cdrs: [CDR3]}` | Design only VH CDR3; unmentioned roles are fixed. |
| `VL: {}` | Fix the complete VL sequence. |
| `cdrs: []`, `framework: true` | Keep CDRs fixed and open framework sequence editing for that role. |

`lengths` applies only to selected CDRs. A length can be `native`, `official`, a
positive integer, or an inclusive `{min, max}` range. `native` preserves the
length while allowing the selected CDR sequence to change. `official` follows the
scaffold's official insertion/deletion range. An integer is the final CDR length,
not an insertion count.

Framework editing is opt-in. `framework: false` fixes the role's framework
identity. `framework: true` opens all framework positions for that role. A role
scoped selector such as `framework: "5/10-12"` opens only those source positions;
it does not request that those residues mutate. CDR and framework design choices
affect sequence identity while the complete variable-domain structure still goes
through the model.

## Antigen, crop, hotspots, and bonds

`antigen.input` is a local PDB or mmCIF path resolved from the command's working
directory. `antigen.crop` uses the shared `include`, `exclude`, and `proximity`
selectors. `crop: {include: [A]}` retains chain A but does not select an epitope.
Use `antigen.hotspot`, for example `A20/A30-32`, to mark target residues.
Hotspots guide interface intent; they do not freeze or redesign the antigen.

The antibody task schema accepts the scaffold, role design, antigen input, crop,
and hotspot fields shown here. It does not accept the normal task's
`condition.bond`, contig, or cyclic fields. Use a normal protein/peptide/ligand
task when those topology conditions are required; their endpoint grammar is in
[Task schema](../task_schema.md).

## Search, omission, and output

The default antibody task preserves the FM sequence and does not invoke MPNN,
packing, or refolding. Its free protein identity policy defaults to
`rm_token_type: [CYS]`. Fixed scaffold identities, unselected CDRs and frameworks,
context, and coordinate motifs remain exempt. Selected mutable CDR positions still
use the omission policy even when a soft template is visible. To redesign and rank
candidates, set an explicit
Best-of-N search and reward, for example:

```bash
CUDA_VISIBLE_DEVICES=0 partidesign design configs/examples/antibody.yaml \
  --task antibody_fixture \
  --route true --route-ckpt antibody \
  --batch-size 1 \
  --run-name antibody_ranked \
  --override reward=antibody_esmfold2 \
  --override search=best_of_n \
  --override search.best_of_n.replicas=2
```

Best-of-N writes each successful ranked antibody replica to the antibody task leaf;
`result.csv` contains one ranked row per emitted structure, with `rank1` as the
highest-scoring candidate.

Use the canonical `rm_aa` field for sequence-backend residue omission. A missing
or `null` `rm_aa` follows the backend's existing `omit_aas` policy; a present
string, including the empty string, is authoritative. Fixed scaffold residues,
framework positions, explicit CDR identities, motif/context tokens, and explicit
bond endpoints are not removed by free-polymer omission. See [MPNN](../mpnn.md)
for source-specific precedence.

Set `score: true` for a final PartiScore table and `postprocess: true` for the
separate OpenMM stage. These are independent of reward ranking. Antibody tasks
write the normal run files plus an antibody-specific `result.csv` with heavy/light
roles, CDR/framework ranges, sequence and structure sources, reward fields, and
status columns. See [Outputs](../outputs.md).

## Batch isolation

Select several Fab and VHH tasks together only when all are `format: antibody`.
They share the antibody routing member and remain one homogeneous submission.
Normal tasks such as `protein_bind_protein` or `peptide_bind_protein_cyclic` must
run in another command. This format boundary is checked before model loading and
prevents a normal task from being silently routed through an antibody checkpoint.


## Validation and BindCraft2 comparison

Select the folding method separately from the success criterion:

```bash
partidesign validate ./inference/antibody_fixture \
  --method esmfold2-standard --source gt:1 --criteria af3_antibody --plan

CUDA_VISIBLE_DEVICES=0 partidesign validate ./inference/antibody_fixture \
  --method esmfold2-standard --source gt:1 --criteria af3_antibody \
  --worker-gpus 0.5 --worker-cpus 1
```

Replace `esmfold2-standard` with `esmfold2-fast`, `af3`, or `af3-la` to select
another supported model. The criterion name stays `af3_antibody`; it identifies
the shared filter family rather than forcing a particular backend. Without an
explicit source list, folding validation includes GT plus one applicable MPNN
sequence; `--source gt:1` above intentionally evaluates only the design sequence.
This is independent of the `reward=antibody_esmfold2` generation reward identifier.

Our criterion requires both of these metrics on the **same complete prediction**:

- Mean antibody–antigen PAE **<10 Å**, over both directions and all declared
  antibody/antigen token pairs. VH–VL internal pairs are excluded.
- Antigen-aligned antibody CA RMSD **<2 Å**, against the generated structure.
  Fit all declared antigen anchors, then measure all antibody CA atoms without
  an additional antibody fit.

Once the configured prediction stages are complete, a candidate passes if one
prediction passes both rules. Missing mappings,
incomplete reference coverage or required confidence yield `unavailable`.
Antigens can be protein, RNA, DNA, ligand, or a mixture: alignment uses protein
CA, nucleic-acid C1′, and uniquely mapped ligand heavy atoms. At least three
noncollinear antigen anchors are necessary; a one-atom or collinear antigen
cannot define this pose metric. These are project filters, not binding-affinity
predictions or a reproduction of BindCraft2's complete screening pipeline.

### What BindCraft2 actually uses

The audited official revision is
[`5bac63c`](https://github.com/PacesaLab/BindCraft2/commit/5bac63cd2d4e7ed3d5077ba4fc5966e0352038e4)
(2026-10-04). Its antibody presets are
[VHH](https://github.com/PacesaLab/BindCraft2/blob/5bac63cd2d4e7ed3d5077ba4fc5966e0352038e4/settings/modality/VHH.json),
[scFv](https://github.com/PacesaLab/BindCraft2/blob/5bac63cd2d4e7ed3d5077ba4fc5966e0352038e4/settings/modality/scFv.json), and
[Fab](https://github.com/PacesaLab/BindCraft2/blob/5bac63cd2d4e7ed3d5077ba4fc5966e0352038e4/settings/modality/Fab.json).
VHH is one variable domain. Its scFv preset models VH and VL as **two separate
chains**, without a linker. Its Fab includes constant domains. PartiDesign's
built-in Fab-labelled scaffolds contain VH/VL variable domains; they do not add
constant domains or a scFv linker. BindCraft2 does not ship separate complete-IgG
or standalone-VH presets in this revision.

All three select **AF2 Multimer**, default `initial_guess=false`, three design
models and two held-out validation models, with three validation recycles. The
unbound prediction contains the entire antibody assembly; for scFv/Fab it is not
a single isolated chain. See the
[default settings](https://github.com/PacesaLab/BindCraft2/blob/5bac63cd2d4e7ed3d5077ba4fc5966e0352038e4/settings/core/default.json)
and [model selection](https://github.com/PacesaLab/BindCraft2/blob/5bac63cd2d4e7ed3d5077ba4fc5966e0352038e4/bindcraft/settings.py#L570-L586).

| Official default filter | VHH | scFv | Fab |
|---|---:|---:|---:|
| Bound selected-chain pLDDT | ≥0.7 | ≥0.7 | ≥0.7 |
| Unbound assembly mean pLDDT | ≥0.7 | ≥0.7 | ≥0.7 |
| Complex pTM | ≥0.55 | ≥0.55 | ≥0.55 |
| Binder–target iPTM | ≥0.7 | ≥0.7 | ≥0.7 |
| Selected-chain interface mean PAE /31 | ≤0.35 | ≤0.35 | ≤0.35 |
| Interchain CA clashes at ≤2.5 Å | 0 | 0 | 0 |
| Binder residues contacting target at ≤4 Å | ≥7 | ≥7 | ≥7 |
| Fixed framework sequence retained fraction | 1.0 | 1.0 | 1.0 |
| Framework CA RMSD, per-chain fit | No hard cutoff | ≤3 Å | ≤3 Å |
| Off-paratope fraction of contacting residues | ≤0.2 | ≤0.2 | ≤0.2 |

The interface PAE cutoff is 10.85 Å after multiplying by 31. For the double-chain
formats, selected-chain pLDDT and interface PAE use the first scaffold binder
chain by default; iPTM groups all antibody chains together and excludes their
internal interface. Unbound pLDDT averages the full antibody assembly. Paratope
membership comes from scaffold CONTACT annotations, rather than a generic rule
that every CDR residue must contact. There is no default hard cutoff on
`Binder_RMSD` or `Target_pLDDT` in these antibody presets.
Metric definitions and inclusive comparison operators are in
[filters.py](https://github.com/PacesaLab/BindCraft2/blob/5bac63cd2d4e7ed3d5077ba4fc5966e0352038e4/bindcraft/filters.py)
and [PAE loss.py](https://github.com/PacesaLab/BindCraft2/blob/5bac63cd2d4e7ed3d5077ba4fc5966e0352038e4/bindcraft/loss.py#L216-L226).

BindCraft2 first requires each validation model to pass configured native
confidence checks. It then averages native confidence arrays/scalars and runs
final screening with the **first model's coordinates**. Thus PAE is checked on
the ensemble mean and geometry on the first model; this is neither our
`any_prediction` reduction nor the existing AF2 `all_and_mean` policy. See
[MPNN_stage.py](https://github.com/PacesaLab/BindCraft2/blob/5bac63cd2d4e7ed3d5077ba4fc5966e0352038e4/bindcraft/MPNN_stage.py#L87-L131).

The existing PartiDesign `af2_ig` method remains unchanged: it uses pTM model
weights, initial guess, and exactly one protein design chain. A protein-antigen
VHH can enter that method, but its result is not the official BindCraft2 protocol.
A VH/VL task has two antibody design chains and cannot use this backend; select
one of the four methods above. No BindCraft2-labelled AF2 antibody preset is
advertised by this release.


### BoltzGen antibody-anything and RFAntibody references

There are three separate screening ideas to compare; their metric names are not
interchangeable. Our current two-rule gate remains an **RF2-inspired project
screen**, while the alternative thresholds below are reference definitions.

| Reference | Default computational selection | Difference from `af3_antibody` |
|---|---|---|
| BoltzGen `antibody-anything` | Whole-complex backbone RMSD ≤2.5 Å and designed-region backbone RMSD ≤2.5 Å; designed-sequence composition checks | Self-fit N/CA/C/O measurements, with the designed mask from the specification; not antigen-aligned whole-antibody CA RMSD. No default mean-PAE or fixed iPTM pass cutoff. |
| BindCraft2 antibody presets | Bound/unbound pLDDT ≥0.7, complex pTM ≥0.55, antibody–target iPTM ≥0.7, normalized interface PAE ≤0.35, plus the scaffold/geometry checks above | Native AF2 Multimer confidence, a different multi-model policy and additional unbound prediction; not the same inference protocol. |
| RFAntibody README | Suggested RF2 pAE <10 Å and design/refold RMSD <2 Å; optional Rosetta ddG <−20 | Emits several PAE/RMSD fields. The README does not identify the selected fields precisely; our explicit interface/pose selection is a project choice. |

BoltzGen's audited release is
[v0.3.2](https://github.com/HannesStark/boltzgen/releases/tag/v0.3.2),
revision `31d9d9b` (2026-05-01). Current main is
[`a07fea1`](https://github.com/HannesStark/boltzgen/commit/a07fea1153111ef453aaea7a15697ee05de9291f)
(2026-10-06). Both use the structural and sequence checks above. Designed
residues must contain no unknown amino acids or cysteine, and the fraction of
each of ALA/GLY/GLU/LEU/VAL must be ≤0.3. Fixed framework cysteines are excluded
from this designed-sequence test. The default antibody protocol does not require
a separate binder-alone refold. See the
[current filter constructor](https://github.com/HannesStark/boltzgen/blob/a07fea1153111ef453aaea7a15697ee05de9291f/src/boltzgen/task/filter/filter.py#L185)
and [released filter](https://github.com/HannesStark/boltzgen/blob/31d9d9b9c72245b4ed6fe8742d6fbf4e1a3552a0/src/boltzgen/task/filter/filter.py#L149).

For a CDR-only specification, BoltzGen's designed-region RMSD fits and measures
only those CDRs, pooling all selected loops/chains. Opening framework residues
changes that mask. Backbone protein atoms are N, CA, C and O. The optional
target-aligned filter is disabled by default. These masks follow the
[design specification](https://github.com/HannesStark/boltzgen/blob/a07fea1153111ef453aaea7a15697ee05de9291f/src/boltzgen/data/parse/schema.py#L2231),
[backbone atom mask](https://github.com/HannesStark/boltzgen/blob/a07fea1153111ef453aaea7a15697ee05de9291f/src/boltzgen/data/feature/featurizer.py#L1067),
and [RMSD implementation](https://github.com/HannesStark/boltzgen/blob/a07fea1153111ef453aaea7a15697ee05de9291f/src/boltzgen/model/loss/validation.py#L1651).

BoltzGen `pass_filters` annotates and penalizes ranking; its final budget can
still contain failed rows. It is not identical to PartiDesign's strict passed
criterion. Released ranking uses designed-residue-to-target iPTM, binder pTM,
negative minimum PAE, H-bonds, salt bridges and ΔSASA. Current main instead ranks
minimum-directional ESMFold2 ipSAE alongside binder pTM and the physical terms.
The 10 Å PAE cutoff *inside ipSAE* selects scoring partners; it is not a mean-PAE
success threshold. See
[filtering and budget selection](https://github.com/HannesStark/boltzgen/blob/a07fea1153111ef453aaea7a15697ee05de9291f/src/boltzgen/task/filter/filter.py#L491)
and [ESMFold2 scoring](https://github.com/HannesStark/boltzgen/blob/a07fea1153111ef453aaea7a15697ee05de9291f/docs/esmfold2.md).

The RFAntibody reference is
[revision `3e5a5bb`](https://github.com/RosettaCommons/RFantibody/blob/3e5a5bb1cb98befde9fc049ae3ff58cabb491f67/README.md#filtering-strategies).
Its RF2 code emits `interaction_pae` for antibody–target pairs, excluding VH–VL,
and also a whole-complex `pae`. The serializer reduces the latter matrix to a
mean. It emits target-aligned antibody RMSD and framework-aligned RMSD variants.
Our protein-antigen pose geometry matches the exposed target-aligned CA
calculation, but this does not prove which RMSD the README's generic filter used.
See [RF2 scores](https://github.com/RosettaCommons/RFantibody/blob/3e5a5bb1cb98befde9fc049ae3ff58cabb491f67/src/rfantibody/rf2/modules/model_runner.py#L157)
and [target alignment](https://github.com/RosettaCommons/RFantibody/blob/3e5a5bb1cb98befde9fc049ae3ff58cabb491f67/src/rfantibody/rf2/modules/rmsd.py#L7).

The later [RFAntibody paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12727541/)
reports retrospective AF3 ipTM observations at >0.6 for VHH and >0.85 for scFv,
using target MSA/templates and antibody templates; scFv used the maximum over
ten seeds. Those are not established cutoffs for our empty-MSA/template,
single-refold default or an antigen-only chain-pair score.

A future BindCraft2-style confidence screen should first compute confidence for
**all declared antibody chains**, excluding their internal interface. The current
validation `plddt` is total-complex confidence, so it cannot substitute for
antibody-only pLDDT. A derived mean of native chain-pair iPTMs is also distinct
from native combined-antibody iPTM. No extra confidence thresholds, unbound
refolds, BoltzGen composition gates or ipSAE implementation are silently added by
this cleanup. Choosing those would change the definition of success and requires
an explicit protocol decision; source thresholds alone do not establish
cross-model calibration.
