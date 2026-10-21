# ColabDesign and CyclicMPNN bootstrap provenance

## Vendored source

- Project: ColabDesign
- Upstream repository: `https://github.com/sokrypton/ColabDesign.git`
- Source commit: `e31a56fe1d9b4de25c8697f3a28b75892941cc72`
- Recovered distribution metadata: ColabDesign 1.1.3
- License: Beer-Ware Revision 42; retained in `LICENSE.txt`

The runtime source is vendored for strictly offline use. The version statement
above comes from the locally recovered distribution metadata and the source
inventory; it is not a claim that runtime imports an installed wheel.

## Embedded source notices

The enclosing Beer-Ware notice is not the only license in this source tree:

- Facebook-authored files in `source/colabdesign/esm_msa` carry MIT headers.
  Their exact locally installed ESM license is retained as
  `LICENSE-ESM-Facebook-MIT.txt`.
- `esm_msa/__init__.py` and `axial_attention.py` carry a Levinthal, Inc. MIT
  header, while `esm_msa/config.py` carries a Levinthal Limited Apache-2.0
  header.  Those source notices are transcribed in `NOTICE-ESM-MSA.txt`.
  The vendored snapshot contains no separate Levinthal root-license file, so
  no additional copyright line is invented here.
- Files below `source/colabdesign/af/alphafold` carry 2021 DeepMind
  Apache-2.0 headers.  The official upstream AlphaFold license is retained as
  `LICENSE-AlphaFold-Code-Apache-2.0.txt`.

These are code licenses.  `LICENSE-AF2.txt` is separately the CC BY 4.0 notice
for the installed, Git-ignored AlphaFold parameter files; it is not used as the
license for the AlphaFold-derived Python source.

## Pinned ProteinMPNN bootstrap

Exactly one normally ignored model asset is eligible for source-release
tracking:

```text
source/colabdesign/mpnn/weights/v_48_020.pkl
size: 6650698 bytes
sha256: 993194d9cbe9334f6c2c81bddeb1404ea831711bca8e38b591b4116060484ac3
```

The locally cached ColabDesign 1.1.3 wheel RECORD contains this exact entry:

```text
colabdesign/mpnn/weights/v_48_020.pkl,sha256=mTGU2cvpM09sLIG93rFATqgxcRvKjji1kbQRYGBISsM,6650698
```

The URL-safe base64 RECORD digest decodes to the SHA256 stated above. The
binary is the ProteinMPNN `v_48_020` base checkpoint used by ColabDesign's MPNN
module and by PartiDesign's project-specific CyclicMPNN bootstrap. ProteinMPNN
is MIT licensed; its notice is retained in `LICENSE-ProteinMPNN.txt`. No more
specific training-data or upstream checkpoint commit is asserted because the
recovered wheel RECORD does not establish one.

All AF2 NPZ parameters and every other PT/PKL/NPZ/Safetensors asset remain
installed, ignored runtime assets and are forbidden from the tracked source
release.
