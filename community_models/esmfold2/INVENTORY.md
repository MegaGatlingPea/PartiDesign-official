# ESMFold2 offline assets

- Source provenance: copied from PipeTools commit
  `8b4e143f40ccb2076eceba32814cacb43df2d8f2`, using the vendored Biohub ESM
  and Transformers source trees under `source/`.
- Required local variants: `checkpoint/fast`, `checkpoint/standard`, and
  `checkpoint/esmc-6b`.
- Required local chemistry data: `common/ccd.pkl`, `common/components.cif`,
  and `common/ccd_bonds.sqlite3`.
- `common/ccd.pkl` has a second, narrow role for ligand `type_bond`: runtime opens the
  exact 417 MB release snapshot once in an isolated spawned child, using one
  `O_NOFOLLOW` descriptor for regular-file/owner/mode/stability checks, SHA256 bytes,
  and pickle decode. Only requested component IDs plus immutable RDKit molecule binaries
  return to the parent. Asset drift or loader/protocol failure is pre-output run-fatal;
  runs without `type_bond` do not open it.
- Runtime policy: query-only input, no MSA search, no network access, no model
  or CCD download, and one persistent owner-local ESMFold2 worker per batch.
- Checkpoints, CCD snapshots, caches, environments, and executable smoke logs
  are installed assets ignored by Git. Acceptance validates the complete
  required-file set and performs an offline fold; it does not retain local
  checksum or smoke-test artifacts in the release tree.
- Fresh Ubuntu staging evidence under
  `/home/megagatlingpea/workdir/PartiDesign_release_20260814/inference/formal_all_cases`
  used the persistent offline worker for protein 350/350 and nucleic 270/270 Best-of-5
  folds. Coherent ligand retained 89 valid folds plus 36 structured candidate failures
  in 125 rows; the separate hard type-bond/covalent subset passed 40/40, and one
  CyclicMPNN protein fold passed 1/1. Metrics were finite, readers passed, and socket use
  was zero. Actual headless Mol* loaded the 711 successful formal CIFs, including the
  coherent ligand 89/89. This is matching-modality Ubuntu evidence; A800 integrated
  reward and the three-checkpoint × all-task cross-product remain pending.
