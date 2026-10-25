# Postfilter successful protein complexes

Use this reference for optional CPU relaxation and structural screening after
validation. Read the active checkout's `docs/postfilter.md` for the maintained
selection contract, exact metric definitions and criteria, dependencies, outputs,
and recovery flags. Do not copy thresholds from another release or run a
postfilter merely because validation finished.

## Preserve the validation handoff

`partidesign postfilter` consumes named passing decisions from
`partidesign validate` and their actual complete complex refolds. Inputs can
be a run, task, validation method directory, or its `success/` directory.
Keep `.state/plan.json`, per-sequence records under `refolds/`, and selected
refold files. A loose success CSV or PDB/CIF cannot establish eligibility;
original generated files and `result.csv` are not required.

The supported scope is protein–protein complexes, including antibody–antigen
assemblies with heavy/light chains kept together as the binder. Explicit
peptide tasks, monomers, nucleic acids, ligands, unsupported modifications,
and ambiguous roles are excluded or reported precisely. Length alone does not
define peptide intent. Metrics use each saved refold's actual redesigned
sequence.

The selector uses each named criterion's `passing_prediction_keys`. It does
not score every link in `success/`: a linked monomer or a model that did not
individually pass is insufficient, and completion-only `unfiltered` decisions
are ineligible. Duplicate nominations of one prediction are scored once;
different sequence/model slots remain separate.

## Plan and run the requested scope

```bash
partidesign postfilter inference/RUN --task protein_fixture --plan
partidesign postfilter inference/RUN --method af2_ig \
  --upstream-criteria af2_ig_protdbench --plan
```

Use repeatable `--task`, `--method`, and `--upstream-criteria` for the requested
selection. With no upstream criterion selector, all saved named passing
criteria contribute a union. `--limit N` limits concrete deduplicated refolds
per task; add it only for a requested bounded trial. `--plan` lists candidate
identities, exclusions, and missing tools without native scoring or postfilter
output. It does not prove that native execution will complete.

Both `partidesign_cu12.yaml` / `partidesign` and
`partidesign_cu13.yaml` / `partidesign_cu13` support CPU PyRosetta, DSSP, and
DAlphaBall. The checkout provides DAlphaBall; wheel-only users must provision
it. Inspect `docs/postfilter.md` and `configs/postfilter.yaml` for explicit tool
paths, workers, threads, and timeout. Do not install tools to answer a usage
question. Execution uses isolated CPU workers and allocates no GPU.

After prerequisites pass, remove `--plan` for authorized execution with the
same selection and execution overlay. The default requires both
`bc1_protein_structural` and `bc2_protein_geometry` on the same concrete
refold. Repeated `--criteria` are an AND, unlike validation's OR. Preserve
requested criteria and consult `configs/postfilter_presets.yaml` rather than
inventing thresholds. Missing required metrics are unavailable and cannot pass.

## Read results and recover saved work

Each task writes a separate `postfilter/` with `metrics.csv`, `success.csv`,
`summary.json`, `resolved_config.json`, records, relaxed structures, and private
state. Rows retain method/sample/sequence/prediction identities and separate
preset statuses/reasons. Distinguish threshold failure from incomplete execution
or unavailable metrics. Success requires every selected preset on one refold;
the output points to its actual relaxed structure.

An ordinary repeat resumes compatible work. Use `--refilter` for new thresholds,
`--recompute-metrics` for changed metric settings using saved geometry, and
explicit `--rerun` for changed input, roles, or relaxation settings. These modes
are mutually exclusive; inspect the current guide before replacing outputs.
Retain owned records and private geometry state for recovery. Exit 0 can include
zero passing candidates; exit 1 reports configuration or run-level errors, and
exit 2 means incomplete execution or unavailable required metrics.

Postfilter preserves validation decisions, input refolds, generated structures,
and rankings. It runs no folding predictions or sequence redesign. Clustering
in [workflows.md](workflows.md) independently compares original generated
polymer backbones; it does not consume these relaxed refolds.
