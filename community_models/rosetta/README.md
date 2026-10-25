# Postfilter native tools

`bin/DAlphaBall.gcc` is a small Linux x86-64 executable built from
[outpace-bio/DAlphaBall](https://github.com/outpace-bio/DAlphaBall/tree/7b9dc05fa2a40f7ea36c6d89973d150eaed459d9).
Its corresponding unmodified sources and LGPL-2.1 license are in
`source/dalphaball/`. It runs as a separate subprocess tool and is not the
PyRosetta SDK. Dynamic dependencies are `libgmp.so.10`, `libgfortran.so.5`,
`libm`, and the C runtime. It has no build-environment RPATH.

Rebuild with GNU C/Fortran and GMP development headers/libraries:

```bash
cd community_models/rosetta/source/dalphaball
make CC=gcc FC=gfortran FFLAGS=-fallow-argument-mismatch
mkdir -p ../../bin
install -m 755 DAlphaBall.gcc ../../bin/DAlphaBall.gcc
make clean
```

Build in a separate directory when maintaining a clean checkout. The bundled
executable is selected automatically only on Linux x86-64. Other platforms
should build their own tool or configure an explicit path. Real native scoring
verifies library compatibility; file presence alone is insufficient.

Both [CUDA 12](../../partidesign_cu12.yaml) and
[CUDA 13](../../partidesign_cu13.yaml) environment definitions install DSSP and
the compatible native dependencies. An explicit `dssp` config path can select
an externally provisioned executable.

Both definitions also install the pinned CPU PyRosetta SDK from the official
distribution under its own terms. For manual installation into an existing
environment, see the
[postfilter installation guide](../../docs/postfilter.md#dependencies-and-execution-settings).
Its SDK, database and downloaded wheel do not belong in Git.

Portable wheels and source archives ship the corresponding source. Provision
the native executable in the active environment or configure its explicit path
when using these packages. The prebuilt Linux executable is retained in Git.
