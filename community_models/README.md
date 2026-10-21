# PartiDesign community models

This directory is the only runtime asset root for optional sequence design and
ESMFold2 reward.  Runtime code must not search user caches, sibling checkouts,
Hugging Face, model hubs, MSA services, or the public network.

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
- `pipetools/`: only the copied ESMFold2/MPNN compatibility modules required by
  the reward runtime. RF3, Protenix, orchestration, MSA, publication, and
  downloader modules are deliberately absent. Its shared runtime helper resolves one
  physical owner GPU consistently for ESMFold2 and all MPNN children.
- `offline_site/`: Python `sitecustomize` that rejects AF_INET/AF_INET6 sockets
  in every community-model child process.

Checkpoint directories and Python environments are intentionally ignored by
Git. Source, licenses, concise provenance, and configuration are tracked; local
SHA256 manifests and smoke-test logs are not release artifacts. Installation
may use a network while constructing an environment;
inference is entirely local and every MSA search path is forbidden.

Community backend load, asset, timeout, device-owner, and worker-protocol failures are
systemic. They never silently fall back to another graph/sequence/reward backend.
CyclicMPNN child failure text includes only a sanitized bounded process-log tail so the
earliest backend error remains diagnosable without leaking arbitrary stderr.

The recommended unified Conda environment is `partidesign_env` (canonical
A800 prefix `/home/jinzichang/miniforge3/envs/partidesign_env`; Python 3.12.6,
PyTorch 2.9.1+cu128).  It is local build output outside this asset tree and
must never enter Git; root `partidesign.yaml` is the only reproducible
environment declaration.
