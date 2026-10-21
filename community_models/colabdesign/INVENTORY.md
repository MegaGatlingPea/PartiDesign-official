# ColabDesign offline source

- Upstream: https://github.com/sokrypton/ColabDesign.git
- Commit: e31a56fe1d9b4de25c8697f3a28b75892941cc72
- Installed distribution version: 1.1.3
- Runtime policy: local source and local weights only; no network or MSA search.
- AlphaFold2 parameters: all 15 model-1..5 monomer, PTM and multimer-v3 NPZ
  files copied byte-for-byte from
  `/home/jinzichang/workdir/PartiFlow-v2/PXDesignBench/tool_weights/af2`.
- Weights are ignored by Git and must be installed under `checkpoint/af2/`;
  release acceptance checks the complete 15-file inventory and runs an offline
  initial-guess fold instead of retaining a local checksum/smoke artifact.
- One narrow exception is the tracked CyclicMPNN bootstrap at
  `source/colabdesign/mpnn/weights/v_48_020.pkl`: 6,650,698 bytes, SHA256
  `993194d9cbe9334f6c2c81bddeb1404ea831711bca8e38b591b4116060484ac3`.
  It is the pinned upstream base model used to load the project-specific cyclic
  checkpoint, not an AF2 parameter or a user-selectable path. Release preflight rejects
  missing, symlinked, size-drifted, or SHA-drifted copies before FM.
- Bootstrap provenance is recorded in `PROVENANCE.md`. The exact binary and
  digest were recovered from a locally cached ColabDesign 1.1.3 distribution
  RECORD; no PyPI URL or more specific upstream training commit is asserted.
  Its ProteinMPNN lineage is attributed under the retained MIT notice in
  `LICENSE-ProteinMPNN.txt`, while the enclosing ColabDesign distribution keeps
  its Beer-Ware notice in `LICENSE.txt`.
- Licenses are component-scoped: upstream ColabDesign Beer-Ware is `LICENSE.txt`; AF2
  parameter CC BY 4.0 is `LICENSE-AF2.txt`; AlphaFold-derived code Apache-2.0 is
  `LICENSE-AlphaFold-Code-Apache-2.0.txt`; embedded Facebook ESM MIT is
  `LICENSE-ESM-Facebook-MIT.txt`; Levinthal ESM-MSA headers are retained in
  `NOTICE-ESM-MSA.txt`; ProteinMPNN MIT is `LICENSE-ProteinMPNN.txt`. Exact provenance
  and the boundary between code and parameter notices are recorded in `PROVENANCE.md`.
- A800 gate: Python 3.12.6/JAX 0.4.35 CUDA loaded the local multimer-v3 model
  and completed one offline initial-guess fold of the 13-residue peptide in
  `pep_dock_prot.pdb` (`i_ptm=0.1326864`, `i_pae=0.7724189`).
- Product boundary: `src.design.rewards.af2` provides the offline pTM/Multimer Best-of-N
  adapter. AF2 cyclic guidance is a design-chain relative-position offset only; the
  private AF2 input carries no BondList, and PartiDesign restores authoritative special
  topology after folding. On Ubuntu staging, integrated pTM and Multimer rewards each
  passed 1/1. A pTM replicas-64 run completed 64/64 with durable
  `model_cache_misses=1` and `model_cache_hits=63` (wall `1:43.31`, peak RSS
  `4,976,220 KiB`, zero socket use) under
  `/home/megagatlingpea/workdir/PartiDesign_release_20260814/inference/formal_all_cases`.
  This closes the Ubuntu single-GPU adapter/cache smoke, not the still-pending A800
  integrated or true two-GPU reward gates.
