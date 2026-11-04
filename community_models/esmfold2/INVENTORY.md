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
- Runtime policy: local model/CCD assets, no worker network access or downloads,
  and one persistent owner-local ESMFold2 worker per batch. Fast uses query-only
  input. Standard validation/pipeline prediction can consume shared fixed
  protein-context MSA prepared by the driver; designed GT/MPNN/VHH/VH/VL chains
  are not automatically searched, and RNA/DNA remain query-only. ESMFold2 never
  consumes templates. A compatible template-on cache generation provides its
  template-off view while retaining templates for AF3/Mol-1.
- MSA controls: `esmfold2.msa` uses the existing shared schema and common
  `--msa`/`--msa-source` flags; `--msa off` restores query-only standard input.
  Native defaults are `msa_max_depth: 1024`, `msa_column_mask_rate: 0.1`, and
  `msa_subsample_at_inference: true`. A `null` depth removes the row limit subject
  to native feature caps. The shared cache and pins stay unchanged; prediction
  `confidence.msa` records actual features and effective depth in the existing
  output tree.
- Checkpoints, CCD snapshots, caches, environments, and executable smoke logs
  are installed assets ignored by Git. Acceptance validates the complete
  required-file set and performs an offline fold; it does not retain local
  checksum or smoke-test artifacts in the release tree.
