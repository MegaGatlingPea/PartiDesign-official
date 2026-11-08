# Sequence screening

Use `partidesign screen INPUT.yaml --out RUN`; start with `--plan` for a schema
check without downloads, searches, models or output creation. Read the selected
checkout's `docs/screen.md` and `configs/examples/screen.yaml`. Use
[resource and budget choices](resources.md) when choosing worker reservations
and prediction sample counts.

Each named YAML task has one joint `contexts` mapping: chain ID → `type` and
`sequence` (a ligand context uses SMILES in `sequence`). `candidates` names a CSV with exactly
`id,modality,seq`. A row is one candidate chain: protein, peptide, RNA, DNA, or a
connected ligand SMILES in `seq`. Paths resolve from the invocation directory.
Do not invent `target1` groups, `design_chains`, source structures or RMSD targets.
VHH/scFv can use a single protein sequence; paired VH/VL rows are not supported.

`methods` accepts esmfold2-fast, esmfold2-standard, af3, af3-la, mol1 and boltz2;
AF2-IG is excluded. `samples` defaults to 1. Each model first chooses its sample
using native confidence, then ranks candidates within each modality. Polymer
ranking defaults to symmetric `design_ipsae`. Ligand Boltz ranking uses native
binder probability; other models use scoped `ligand_iptm`. Missing metrics stay
missing, never fall back to global iPTM. No cross-model score averaging occurs.
`ranking: {metric: NAME, order: desc}` explicitly overrides ranking. All structures
remain under `predictions/`; read scores.csv, samples.csv and rankings/.

Context MSA/template settings use the existing shared cache. Set task `msa`,
optionally override a context chain's `msa`, then explicit CLI flags take priority.
Candidates remain query-only. Fast disables MSA/templates; standard uses fixed
protein-context MSA without templates. `--msa cache-only` cannot search.
Do not import local A3M/template files through this first-version schema.

A polymer context may specify `pocket: [25, 48]`, using 1-based positions in its
supplied sequence. Boltz applies native 6 Å, force=false conditioning. Other
methods run with an explicit unapplied status; no pocket postfilter is added.
Ligand context SMILES character positions are not residue/atom indices.

`--worker-cpus` and `--worker-gpus` are global CLI reservations per resident
worker through the same helper and native backend classes as validate/pipeline.
Screen has no task/method YAML resource overrides or candidate `batch_size`
control. `samples` sets the prediction budget. Method/configuration groups run
serially, with parallel workers within the active group. `--resume`
reuses complete predictions, requests only missing structure slots and retries
missing representative affinity evidence. Do not change inputs in the same run.
Boltz affinity performs internal sampling; its scored-pose NPZ is distinct from
the public confidence-stage CIF. Predicted scores do not establish experimental binding.

Group ligand outputs with `partidesign cluster RUN --group-by identity|murcko`.
Murcko retains analog members; acyclic molecules use exact identity rather than
one shared empty scaffold. `--method` still selects a prediction method.
