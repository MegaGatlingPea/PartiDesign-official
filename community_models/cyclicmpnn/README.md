# CyclicMPNN offline assets

- Source workspace: PartiFlow-v2 commit `2d4ddd7d978b4d6bc27e05c9c373521bbcde83a0`.
- Weight-introducing commit: `8845c1bf9c317c3953f1999db963f177f094deae`.
- Runtime checkpoint: `checkpoint/cyclicmpnn_48_010.pkl`.
- The curated JAX checkpoint is Git-tracked and supplied by a PartiDesign clone.
  No separate CyclicMPNN weight download or PyTorch-to-JAX conversion is needed
  for the source-checkout installation.
- The copied wrapper contains Apache-2.0 ByteDance code, MIT ProteinMPNN
  adaptations, and one MIT BindCraft-derived helper.  Exact license texts and
  file-level attribution are in `LICENSE-Apache-2.0.txt`,
  `LICENSE-ProteinMPNN.txt`, `LICENSE-BindCraft.txt`, and `PROVENANCE.md`.
- These source licenses do not establish a separate redistribution license for
  the project-specific checkpoint.  Redistribute the weight only after that
  provenance gap is resolved.
- Runtime wrapper: the minimal `pxdbench/tools/protmpnn` source copied from the same workspace, with only import isolation and explicit local-checkpoint handling patched for PartiDesign.
- ColabDesign source: `community_models/colabdesign/source`, upstream commit recorded in its inventory.
- ColabDesign loads the configured Cyclic checkpoint directly once. The ordinary upstream ProteinMPNN checkpoint is not a bootstrap dependency for CyclicMPNN.

MPNN and ESMFold2 children share the same physical-owner resolver. A parent exposing one
GPU keeps that entry even with a nonzero local rank; a multi-GPU parent selects the
entry indexed by `LOCAL_RANK`; an unset `CUDA_VISIBLE_DEVICES` selects local rank or
standalone GPU 0. Empty, malformed, duplicate, or out-of-range ownership is systemic.
Cyclic child model-load/timeout/protocol failure is also systemic and reports
the sanitized final 2,000 printable characters of its combined process log; it never
silently falls back to ProteinMPNN or the FM sequence.

The backend designs one standard-protein cyclic chain and may condition on
standard-protein receptor chains.  Ligands, nucleic acids, and modified
polymers are deliberately ineligible.  The real A800 cyclic-monomer plus
protein-context smoke passed with fixed-residue preservation.

On the synchronized Ubuntu staging checkout,
`inference/formal_all_cases/smoke_cyclic_owner_final` ran with both
`CUDA_VISIBLE_DEVICES` and `LOCAL_RANK` unset. The resolver selected physical GPU 0 and
ESMFold2+auto/CyclicMPNN completed 1/1 candidate in `1:35.93`; the 792-atom CIF passed
project/Biotite readers, metrics were finite, and socket use was zero. This is an owner/
bootstrap/runtime smoke, not the full protein or scientific-quality gate.

Runtime is strictly offline.  No weight, CCD, MSA, or model download is
permitted.

The PartiDesign resident worker batches independently prepared structures with compatible tensor shapes and preserves each input's fixed positions, bias and temperature. This path is enabled by default; it does not reuse features from a different structure or impose sequence-length/batch-size cutoffs. Warm forward and preparation-inclusive benchmarks are retained in docs/benchmarks/inference-cleanup-af2-mpnn-20260921.md.
