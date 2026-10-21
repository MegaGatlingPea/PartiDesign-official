# AlphaFold 3 shared runtime

`source` vendors the official `google-deepmind/alphafold3` v3.0.4 release.
Upstream: https://github.com/google-deepmind/alphafold3/tree/v3.0.4

PartiDesign changes are limited to local Conda/CMake dependency discovery and
relative library paths, current runtime dependency pins, the shared CCD resource
location, and chain-local cyclic residue-relative encoding. Cyclic encoding adds
input features without changing model parameter shapes and is applied by both
trunk and diffusion embedding calls. The upstream source license and parameter
terms remain in `source/LICENSE` and `source/WEIGHTS_TERMS_OF_USE.md`.

The `partidesign` environment builds and installs this native SDK from source.
Stock and leaving-atom models share this SDK. Stock weights belong in
`checkpoints/af3.bin.zst`; AF3-LA weights belong in the sibling
`../af3-la/checkpoints/af3_leaving_atom.bin.zst`. Shared CCD assets belong in `ccd`.
Use `partidesign download af3 af3-la` to populate them. Parameter archives,
generated CCD files and build outputs are local assets and do not enter source
packages or Git commits. See `docs/AF3_VALIDATION.md` for usage and semantics.
