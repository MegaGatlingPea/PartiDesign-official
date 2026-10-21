# PartiDesign NA-MPNN v1 offline asset inventory

This directory contains the minimal, immutable NA-MPNN design assets vendored
for the standalone PartiDesign reward worker. It is intentionally independent
of the PipeTools checkout and of any model-specific Python environment.

- Upstream: `https://github.com/baker-laboratory/NA-MPNN.git`
- Pinned revision: `9fabc2482092b725e067969fba21297a806b6fda`
- Capability: backbone-conditioned DNA/RNA sequence design only
- Checkpoint: `checkpoint/s_19137.pt`
- Checkpoint SHA256: `5fb12951a0ad3ee44b91f33b7dd19114e755b49eb577d9eb1be71b24b7e49e4c`
- Source entrypoint: `source/inference/run.py`
- Runtime interpreter: the active PartiDesign interpreter (`sys.executable`)
- Runtime policy: offline; no MSA, template, database, or network access

Only the inference source files, design configuration, upstream notices, and
exact hash manifest are distributed in the tracked source release. The pinned
checkpoint remains a gitignored external model asset: it must be provisioned
separately at `checkpoint/s_19137.pt` with the exact recorded SHA256 before
NA-MPNN can run. Missing or mismatched checkpoint assets fail closed. No upstream
environment, examples, logs, training code, specificity assets, or PipeTools
modules are part of this deployment.

`source/LICENSE` and `source/README.md` retain the upstream notices. The
validator checks this inventory, every regular-file hash, the pinned upstream
metadata, and import compatibility before a model is loaded.
