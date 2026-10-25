# Practical FAQ

Locate the runtime checkout first; `<repo>` below means that checkout, not this
skill's installation directory. Read the referenced local page and current
configuration for the user's installed version.

## 1. Where should I install and run PartiDesign?

Use `<repo>/README.md` and `<repo>/docs/environments.md`: Linux with an
NVIDIA GPU, using `partidesign_cu12.yaml` / `partidesign` or
`partidesign_cu13.yaml` / `partidesign_cu13`. Create the environment from the
checkout root, activate it, then run `partidesign --help`. Both support CPU
postfiltering with PyRosetta, DSSP, and checkout-provided DAlphaBall. For CUDA
issues, inspect
`nvidia-smi` and the active Python/PyTorch build using
`<repo>/docs/check.md`. A CPU resource check does not prove GPU readiness.

## 2. Where are the native generation weights?

The manual release download is the
[Google Drive checkpoint](https://drive.google.com/file/d/19j8aaysxeQlT0RI5URrQm3A9uWdqSE5e/view?usp=sharing)
(4,909,825,762 bytes). Place it at `ckpts/partidesign.ckpt` in the selected
model root, beside `ckpts/partiscore.ckpt`, or select a compatible bundle with
`--ckpt`. The bundle needs model, autoencoder and experiment configuration;
a flow-matching-only training checkpoint is insufficient. The current native
`partidesign` download-registry entry has no automatic source, so
`partidesign download --all` does not fetch this manual Drive release. Check the
current inventory with `partidesign download --list`; see
`<repo>/docs/download.md` for setup.

## 3. Why are weights missing after I chose another model directory?

Root priority is `--model-dir`, then `PARTIDESIGN_MODEL_DIR`, then the detected
checkout. That root holds `ckpts/` and `community_models/`; copy required
Git-supplied PartiScore/CyclicMPNN weights into matching locations as well as
the generation bundle. Static chemistry and bundled source stay with the
installed package. A model-root change does not relocate task inputs or
outputs. See `<repo>/docs/download.md`, Use a separate model directory.

## 4. Which preview or check should I use?

`design --show-config` resolves settings without loading weights, probing the
GPU, or generating. Automatic backend selection remains pending until runtime.
For the real protein fixture, `check configs/examples/protein.yaml --task protein_fixture`
checks selected inputs, routing and required assets without inference.
`check --load` adds strict CPU loading and may need
substantial RAM. `check --smoke` runs bounded real jobs and reads outputs;
GPU backends require CUDA. `check --model partiscore --smoke` can run on CPU
without generation weights. See `<repo>/docs/check.md` for task/model option
compatibility and retained smoke outputs.

## 5. Why did one command request more samples than expected?

Without `--task`, every named task in the file is selected; repeat `--task` to
select several explicitly. `sample_num` is per length plan, so three lengths
with two samples request six base samples. Joint rows are explicit combinations,
and Best-of-N replicas multiply candidates. `--batch-size 1` changes execution,
not the count. Read `<repo>/configs/examples/README.md` and
`<repo>/docs/task_schema.md` before choosing a budget.

## 6. Why can't the task find my structure or residue?

Input paths resolve from the invocation directory, not the YAML directory;
run bundled examples from `<repo>` or use absolute input paths. mmCIF source
selectors use `label_asym_id`/`label_seq_id`; PDB uses source chain/residue IDs
and insertion codes. Final design positions start at 1 and may use negative
indices. Confirm which numbering a field expects in
`<repo>/docs/task_schema.md`; a matching input design chain alone fixes neither
its sequence nor coordinates.

## 7. How do I keep a sequence, backbone, or motif fixed?

Use full/sparse `condition.sequence` for identities, `condition.backbone` for
source representative centers, and `condition.contig` with optional
`polymer_atom` for source motifs. Identity constraints do not fix geometry;
backbone constraints allow identity redesign. Motif coordinates condition the
prediction rather than being copied exactly into the output. These controls
have different effects; choose from `<repo>/docs/task_schema.md` instead of
assuming one preserves everything.

## 8. Does a hotspot crop the target or fix the interface?

No. A hotspot marks intended context residues. Crop `include` selects a starting
set; omitted `include` starts with the whole source, and `proximity` adds nearby
residues before `exclude` removes them. Proximity alone therefore does not
narrow the whole source. Hotspots, existing design chains and context bond
endpoints are mandatory retained content. Read the selectors/crop section of
`<repo>/docs/task_schema.md` before changing the pocket.

## 9. Is `cyclic_offset` enough to make a cyclic or linked peptide?

It changes relative-position encoding and adds no bond. Declare the required
terminal or side-chain connections, with known endpoint identities and stable
atom names. Use the actual topology example in
`<repo>/configs/examples/peptide.yaml`; do not invent chemistry from the task
name. Bonds do not guarantee optimized geometry or stereochemistry, and
validation backends differ in retained crosslinks. See
`<repo>/docs/task_schema.md` and `<repo>/docs/validate.md`.

## 10. Why did routing choose a different model member?

Automatic routing considers the complete selected submission, retained context
and conditions. A short `protein` is not automatically a `peptide`; author
peptide intent explicitly. Protein motifs or retained ligand context can require
`omni`, and mixed task preferences can select it for the submission. Antibody
and normal tasks require separate commands. Avoid forcing a member to fix an
input error; check the task using the intended checkpoint/settings. See
`<repo>/docs/design.md`, Routing and task selection.

## 11. What should I change after an out-of-memory error?

Inspect the log and active GPU first. Try a smaller batch, commonly
`--batch-size 1 --compile false` as an intentional native-eager diagnostic;
this leaves the requested sample budget unchanged. Automatic execution does
not retry arbitrary out-of-memory failures. To reduce work, copy the task and
explicitly reduce `sample_num` or the length plan. Retain scientifically
required context when adjusting a crop. Validation models have their own memory
demands and options. Start with
`<repo>/docs/check.md` and the relevant command page.

## 12. The command ended, but a sample has no CIF. Where do I look?

Each task has `resolved_config.yaml`, `result.csv`, and `logs/design.log` beneath
the run root. Read per-sample status and errors; normal results also report
`postprocess_status`. Failed samples have no placeholder structure. Ranking or
refinement can change final filenames, so use the report to identify them.
See `<repo>/docs/design.md`, Outputs and reruns.

## 13. Does a passing check or a successful run prove a good binder?

No. Checks establish operational readiness at their stated level; generation
completion establishes an output. Neither establishes structural accuracy or
experimental binding. Choose appropriate computational validation and explicit
criteria, and inspect chemistry/geometry where needed. Validation without named
criteria normally collects metrics without publishing passing candidates.
See `<repo>/docs/check.md` and `<repo>/docs/validate.md`.

## 14. Should I use `score`, `validate`, refinement, or `postfilter`?

`score` applies PartiScore to existing structures and writes `score.csv`; it
does not refold or redesign sequences. `validate` evaluates saved run/task
results with selected methods, sequence sources and criteria; `gt:1` means the
original generated identities, not experimental ground truth. `relax` retries
OpenMM refinement. `postfilter` uses actual successful protein–protein complex
refolds for optional CPU Rosetta relaxation and structural screening; it does
not refold or alter upstream results. Generation task `score: true` and
Best-of-N reward ranking are also separate choices. Read `<repo>/docs/score.md`,
`<repo>/docs/validate.md`, `<repo>/docs/relax.md`, and `<repo>/docs/postfilter.md`
for the intended step and its assets.

## 15. Can I reuse an output directory or change saved results?

Generation requires new or empty task directories; choose a fresh run name or
`--out`. Standalone scoring writes a separate report and requires
`--overwrite true` to replace an existing `score.csv`. Successful `relax`
replaces selected CIFs and updates `result.csv`, potentially scores/rankings and
filenames too; inspect `--plan` first. Validation normally reuses compatible
state; its refilter, recompute and rerun modes have different effects.
Keep `.state/` and the per-sequence `refolds/` records: `metrics.csv` alone cannot
be refiltered by the CLI. Read
`<repo>/docs/design.md`, `<repo>/docs/score.md`, `<repo>/docs/relax.md`, and
`<repo>/docs/validate.md` before mutating an existing run.

## 16. How should I interpret the published budget and Preview labels?

The budget in `<repo>/docs/performance.md` estimates computationally passing
outputs per 24 hours on one RTX 4090-48G per case. Polymer budgets include the
stated validation/redesign; ligand budgets count GPU generation while CPU
chemical scoring/docking is separate. They exclude several workflow costs and
do not measure binding rates or service throughput. Docking, antibody CDR
design, and peptide design across topologies support inference as Preview
features; quality can vary, and separate performance budgets are not reported
there. Start with a bounded trial matched to the user's hardware and criteria.

## 17. Why did automatic execution select BioIR or native?

The public backend and compilation defaults are `auto`. Count the selected
tasks' expanded length plans and `sample_num`, including Best-of-N replicas.
At least 100 candidates selects native dynamic compilation; smaller submissions
use BioIR when usable, otherwise native eager. BioIR uses graph replay for
eligible inputs and eager kernels otherwise. This is one choice for the
submission, not per batch. An explicit native request keeps the
same automatic compilation cutoff; explicit BioIR takes precedence over count
and reports dependency errors. CPU checks do not prove GPU suitability.
Read the requested/resolved settings, reason, and count in the runtime log and
saved config. See `<repo>/docs/design.md` for explicit compile/strategy rules.

## 18. Can postfilter consume a success CSV or replace clustering?

It needs saved named passing decisions, the validation plan and sequence records,
and the actual complete complex refolds; a loose CSV, PDB, or CIF is insufficient.
Eligible protein–protein complexes include antibody–antigen assemblies, while
explicit peptides, nucleic acids, ligands, and monomers are excluded.
`postfilter --plan` checks selection and tools without native scoring.
Postfilter criteria are all required, unlike validation's OR across criteria.
Its relaxed refolds are separate from clustering, which compares original
generated polymer backbones. Use [postfilter.md](postfilter.md) for the workflow
and `<repo>/docs/postfilter.md` for maintained criteria and recovery details.
