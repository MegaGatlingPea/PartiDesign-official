# PartiDesign community models

This directory is the only runtime asset root for optional sequence design and
ESMFold2 reward/validation. Model and sequence workers must not search user
caches, sibling checkouts, Hugging Face, model hubs, MSA services, or the public
network. Validation and pipeline drivers prepare shared context MSA separately.

## Layout

- `esmfold2/`: Biohub ESMFold2 fast/standard weights, ESM-C 6B weights, source,
  and CCD snapshots. Copied from PipeTools
  commit `8b4e143f40ccb2076eceba32814cacb43df2d8f2`.
- `proteinmpnn/`: official ProteinMPNN source and vanilla/CA/soluble weights.
- `ligandmpnn/`: official LigandMPNN source, sequence-model variants and
  side-chain packer.
- `cyclicmpnn/`: PartiFlow-v2 CyclicMPNN wrapper and project checkpoint. It also uses
  the exact tracked ColabDesign `v_48_020.pkl` bootstrap recorded in both inventories;
  runtime validates its size/SHA before FM for explicit CyclicMPNN or auto sequence
  source with any cyclic-offset task.
- `colabdesign/`: local ColabDesign 1.1.3 source at upstream commit
  `e31a56fe1d9b4de25c8697f3a28b75892941cc72`, plus the complete local
  AlphaFold2 model-1..5 monomer/PTM/multimer-v3 parameter set copied from
  PartiFlow-v2. The current AF2Reward adapter runs pTM or Multimer strictly
  offline, caches the process-local model/compiled function, and passes a
  circular relative-position offset (never a chemical-bond tensor) for
  task-local cyclic protein requests.
- `mol1/`: bundled Mol-1 source at upstream
  [`9e85ad16b3d7031dd92c34d99f895ee65547038b`](https://github.com/molence/Mol-1/tree/9e85ad16b3d7031dd92c34d99f895ee65547038b),
  shared `source/` and `common/`, and separate `checkpoint/mol-1-base.pt` and
  `checkpoint/mol-1-constraint.pt`. Use `partidesign download mol1` or
  `partidesign download mol1-constraint`; official checkpoints and common data
  download separately from bundled source. The constraint variant uses native
  SITE guidance; standard Mol-1 remains available.
- `pipetools/`: only the copied ESMFold2/MPNN compatibility modules required by
  the runtime. RF3, Protenix, orchestration, MSA-search, publication, and
  downloader modules are deliberately absent. Its shared runtime helper resolves one
  physical owner GPU consistently for ESMFold2 and all MPNN children.
- `offline_site/`: Python `sitecustomize` that rejects AF_INET/AF_INET6 sockets
  in every community-model child process.

Checkpoint directories and Python environments are intentionally ignored by
Git. Source, licenses, concise provenance, and configuration are tracked; local
SHA256 manifests and smoke-test logs are not release artifacts. Installation
may use a network while constructing an environment. Model execution stays
local. ESMFold2-standard validation and pipeline prediction can consume the
existing shared ColabFold or local AF3 cache for fixed protein context, with no
templates. Fast stays query-only, and designed GT/MPNN/VHH/VH/VL chains are not
automatically searched. ESMFold2 keeps RNA/DNA query-only with either source.
The driver performs authorized searches before prediction workers start;
`--msa off` restores standard's query-only path.

`esmfold2.msa` uses the shared MSA schema and common `--msa`/`--msa-source`
options. Native defaults are `msa_max_depth: 1024`, `msa_column_mask_rate: 0.1`,
and `msa_subsample_at_inference: true`; a `null` depth removes the row limit
subject to native feature caps. Cache paths, source/profile/generation pins,
and prediction directories stay unchanged. A compatible template-on generation
can supply ESMFold2's template-off view while retaining AF3/Mol-1 templates.
The existing prediction `confidence.msa` metadata reports actual features and
effective depth.

Community backend load, asset, timeout, device-owner, and worker-protocol failures are
systemic. They never silently fall back to another graph/sequence/reward backend.
CyclicMPNN child failure text includes only a sanitized bounded process-log tail so the
earliest backend error remains diagnosable without leaking arbitrary stderr.

`bash install.sh` defaults to the uv/CUDA12 environment (`.venv` paired with a
native Conda prefix); `--installer conda` selects the Conda route and `--cuda 13`
selects CUDA13. Both Mol-1 variants use the same environment. Environments and
weights are local assets; see the current [installation guide](../docs/environments.md)
and [model downloads](../docs/download.md#mol-1-assets).
