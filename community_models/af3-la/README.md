# AlphaFold 3 leaving-atom parameters

This target uses the shared native SDK in `../af3/source` and CCD in `../af3/ccd`.
Its separate parameters are `checkpoints/af3_leaving_atom.bin.zst`, prepared by
`partidesign download af3-la`. Parameter archives are local assets ignored by Git.
Validation selects `--method af3-la`; stock parameters select `--method af3`.
See `docs/AF3_VALIDATION.md` and the upstream parameter terms in
`../af3/source/WEIGHTS_TERMS_OF_USE.md`.
