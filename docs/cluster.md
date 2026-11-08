# `partidesign cluster`

[Command index](commands.md)

Cluster the saved design-polymer backbones selected by validation success
tables. Clustering runs on CPU with US-align. It compares the generated
structures referenced by the task's current `result.csv`, rather than the
validation refold coordinates. Explicit `--group-by identity` and
`--group-by murcko` group designed ligands using authoritative SMILES instead.
Both chemical modes run on CPU and do not resolve or run US-align.

## Inputs and a minimal example

Run [validation](validate.md) with suitable criteria first. The input can be one
or more run roots, task directories, validation method directories, or their
`success/` directories. The command discovers existing `success.csv` tables at
these documented directory levels.

```bash
partidesign cluster inference/first_run --task protein_fixture --plan

partidesign cluster inference/first_run --task protein_fixture \
  --tm-threshold 0.5 --workers 4
```

`--plan` lists selected successful backbones and pair-comparison counts without
alignment or report writes. The second command performs clustering. It requires
US-align in the active Python environment or on `PATH`; select a different
executable with `--usalign FILE` if needed.

## Key options

| Option | Meaning and default |
| --- | --- |
| `RUN_TASK_OR_SUCCESS ...` | One or more existing input directories as described above. |
| `--task NAME` | Select task names; repeatable. Default: all discovered tasks. |
| `--method NAME` | Select existing validation or screen methods; repeatable. |
| `--criteria NAME` | Select passing criteria; repeatable. Each criterion stays separate. |
| `--group-by identity\|murcko` | Explicit designed-ligand grouping. Omit for the existing polymer TM clustering. |
| `--tm-threshold VALUE` | Mean of the two length-normalized TM-scores; default 0.5, valid range `(0, 1]`. |
| `--workers N` | Concurrent success directories; default 16, allowed range 1–16. Each worker uses one CPU thread. |
| `--usalign FILE` | Explicit US-align executable; default is the active Python environment, then `PATH`. |
| `--plan` | Inspect selected inputs and comparison counts without clustering. |

Actual concurrency is bounded by available CPUs and the number of selected
directories. Task, method, criterion, and design-polymer composition stay
separate. Duplicate passing sequence slots from one generated backbone are
collapsed before clustering.

## Designed-ligand chemical grouping

Use validation outputs with passing candidates, including pipeline candidate
tasks that contain the usual `result.csv` and `validate/<method>/success.csv`:

```bash
partidesign cluster inference/ligands --group-by identity --method vina --plan
partidesign cluster inference/ligands --group-by murcko --method vina
```

The selected candidate's `result.csv` must supply each designed ligand's SMILES
in `design_chains[].sequence`, with `modality: ligand`. Chemical grouping reads
this metadata directly. It does not reconstruct chemistry from coordinates or
read source, GT, or predicted structure contents. Candidates without a designed
ligand are excluded. A missing or invalid designed-ligand SMILES is an error.
Validation task, method, and passing-criterion selectors keep their existing
meaning; different criteria remain separate cohorts.

`identity` uses canonical isomeric SMILES. It retains formal charge,
stereochemistry, isotopes, disconnected components, and the authored tautomer
and protonation state. It does not neutralize molecules or standardize tautomers.
`murcko` uses the RDKit Murcko scaffold with its heteroatoms and bond types;
it does not convert the scaffold into a generic carbon skeleton. Molecules with
an empty scaffold, including acyclic molecules, use their exact canonical
isomeric molecular identity as the group key. They do not share one empty group.
When a candidate designs multiple ligands, each chain's identity or scaffold
contributes to the key, preserving the chain assignments.

All candidates remain in the detail report. Different molecules with the same
scaffold remain distinct group members with their own exact identities. Each
group's representative is the first candidate in natural ID order (for example,
`sample2` before `sample10`, with literal ID order breaking ties). The command
does not score representatives, delete candidates, or remove analogs from
downstream inputs. `--tm-threshold` and `--usalign` apply only to polymer clustering.

### Screen outputs

Chemical grouping also accepts a [screen](screen.md) task directory or an output
root containing immediate child screen task directories:

```bash
partidesign cluster screen_results --task ligand_panel \
  --method boltz2 --group-by murcko
```

The task's `.screen.json` must have `kind: partidesign.screen.v1` and retain its
`task.rows` candidate manifest. Eligible `scores.csv` rows have
`status: complete` and `modality: ligand`; their IDs select the authoritative
SMILES from `task.rows[].seq`. Candidate IDs are labels, never structure paths.
The existing screen-owned `structure_path` is copied into report provenance
without reading its file. Methods remain separate. These cohorts are labeled
`screen_complete` and have an empty criterion: completion does not establish
scientific passing criteria. `--criteria` is therefore rejected for screen
inputs. Discovery checks only the documented task levels, without recursively
walking predictions, sources, model assets, or other directories.

## Outputs and reruns

Each selected `validate/<method>/` directory receives:

| File | Contents |
| --- | --- |
| `clusters.csv` | Per-backbone cluster assignment, representative, cluster size, and TM-score to the representative. |
| `cluster_summary.csv` | Per-criterion counts of successful backbones, clusters, singletons, and largest cluster. |

A nonempty set of passing backbones is needed to form clusters. Existing empty
success tables produce reports with zero counts. Rerunning recalculates and
replaces these two reports; it does not reuse previous cluster assignments.

Chemical reports use separate filenames, leaving the polymer TM reports intact:

| File | Contents |
| --- | --- |
| `chemical_identity_clusters.csv` or `chemical_murcko_clusters.csv` | Every eligible candidate, its canonical ligand identities, chemical group key, group size, and representative. |
| `chemical_identity_summary.csv` or `chemical_murcko_summary.csv` | Per-cohort candidate, exact-identity, group, singleton, and largest-group counts. |

For validation inputs, chemical reports are written to `validate/<method>/`.
For screen inputs, they are written to `<screen-task>/cluster/<method>/`.
Reruns replace only the selected chemical mode's two reports. `--plan` validates
the metadata and reports candidate/group counts without writing reports.

## Troubleshooting

| Problem | Next step |
| --- | --- |
| No `success.csv` tables match | Check the run/task path and selectors, then complete validation with suitable criteria. |
| Reports contain no clusters | Inspect the validation passing results. Empty success tables correctly yield zero clusters. |
| US-align is not found | Use the supplied environment's `USalign`, check `PATH`, or pass an executable with `--usalign FILE`. |
| A saved structure cannot be found | Keep the task's `result.csv` and its referenced generated structures together. |
| A task contains only ligands | Select `--group-by identity` or `--group-by murcko` with authoritative designed-ligand SMILES metadata. |
| Ligand SMILES is missing or invalid | Retain the candidate's authoritative `result.csv` design-chain metadata or screen `.screen.json`; coordinates are not a chemistry fallback. |

[Command index](commands.md) · [Validate designs](validate.md) · [Generate designs](design.md)
