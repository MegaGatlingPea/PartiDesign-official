# CyclicMPNN source provenance

The minimal runtime wrapper was copied from the local PartiFlow-v2 workspace at
commit `2d4ddd7d978b4d6bc27e05c9c373521bbcde83a0`; the checkpoint first appears in
that workspace at commit `8845c1bf9c317c3953f1999db963f177f094deae`.

File-level provenance is mixed and must remain explicit:

- `source/pxdbench/tools/protmpnn/main_mpnn.py`, `mpnn_predictor.py`,
  `vanilla_mpnn_predictor.py`, and `source/pxdbench/utils.py` carry ByteDance
  Apache-2.0 headers.  `LICENSE-Apache-2.0.txt` is copied byte-for-byte from the
  local PartiFlow-v2 `third_party/LICENSE`.
- `parse_multiple_chains.py`, `protein_mpnn_run.py`, and
  `protein_mpnn_utils.py` state that they are adapted from the official
  ProteinMPNN project.  The matching MIT text is copied byte-for-byte from
  PartiDesign's vendored ProteinMPNN snapshot as `LICENSE-ProteinMPNN.txt`.
- `source/pxdbench/tools/biopython_utils.py` names the exact BindCraft source
  commit `b8642f3ff27e535615176f3611e298261c95eb3b` in its header.  The matching
  `LICENSE-BindCraft.txt` is copied byte-for-byte from the local BindCraft
  checkout and retains `Copyright (c) 2024 Martin Pacesa`.

These licenses cover the attributed source files.  The source workspace does
not contain a separate license declaration for `cyclicmpnn_48_010.pkl`, so this
release records the checkpoint provenance without claiming that the source
licenses grant checkpoint redistribution rights.
