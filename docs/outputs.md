# Outputs

Without `--out`, `run_name=guide_protein` writes this stable user-facing tree
under the configured default `./inference/${run_name}`:

```text
inference/guide_protein/
└── protein_fixture/
    ├── resolved_config.yaml
    ├── result.csv
    ├── logs/
    │   └── design.log
    ├── sample*.cif
    ├── partiscore.csv              # only when task score=true
    └── validate/
        └── esmfold2-fast/
            ├── metrics.csv
            ├── success.csv
            └── logs/validate.log
```

The exact set of structure and validation files depends on selected stages. Private
temporary worker files are not part of the authoring or result contract.

## Generation files

`resolved_config.yaml` is the normalized configuration actually used by the task,
including the resolved task, route, checkpoint, reward, and output settings.
`logs/design.log` records the ordinary run log, one global seed, progress, and
sample-level failures. It is the first place to look when a row has `status=failed`.

`result.csv` is the per-task result table. Normal rows include these stable fields:

| Field | Meaning |
| --- | --- |
| `filename` | Relative or task-local structure filename. |
| `design_chains` | JSON list describing final chain label, modality, sequence, and residue count. |
| `base_sample_id` | Task-local base sample identity. |
| `design_modality`, `design_chain_id` | Legacy single-chain summary fields. |
| `token_length` | Model-token length for the row. |
| `reward_method`, `global_rank` | Ranking method and candidate rank when search is enabled. |
| `design_sequence` | Sequence summary for the primary design chain. |
| `contig_string`, `tip_atoms` | Resolved motif summary when a contig/partial-atom condition was used. |
| `status`, `failure_stage`, `failure_message` | Base generation status and failure details. |
| `fixed_restypes`, `bond_constraints` | Resolved hard identity/topology summaries for successful rows. |
| `postprocess_status` | `disabled`, `not_run`, `pending`, `succeeded`, or `failed`. |
| `error` | Compact combined base, postprocess, or score error text. |

Additional metric columns are appended when a reward or final score is requested.
The exact metric set is method-dependent. Failed rows retain accounting fields and
do not invent a structure filename or design-chain payload.

For joint tasks, `design_chains` is the authoritative chain list. Its entries retain
the peptide sub-modality and cyclic-offset metadata when those were authored. A
single-chain legacy row may still use the scalar summary fields.

Antibody tasks use the antibody-specific result schema. It reports the heavy/light
roles, CDR and framework ranges, sequence/structure sources, reward and PartiScore
fields, status, and postprocess status. See [Antibody tasks](tasks/antibody.md).

## Filenames and ranks

Unranked single-chain outputs use names such as `sample1.cif`. Ranked outputs
include the chain labels when needed and a rank, such as
`sample1_B-C_rank2.cif`. Best-of-N writes every successful ranked replica in the
task leaf; `rank1` is the highest-scoring row for its task/length selection group.
The candidate pool is published alongside ordinary task outputs rather than in a
second candidate tree.

`base_sample_id` is task-local. When comparing rows across a run, use
`(task_name, base_sample_id)` or the final task-relative filename. A global rank is
an ordering field, not a globally unique sample identity.

## Structure formats and chemistry

mmCIF is the default and authoritative output. The writer preserves design and
context chain identities, concrete noncanonical bonds, ligand graphs, and the stable
terminal OXT policy for protein chains. It reads the written file back before
accepting the structure.

PDB output is bounded. It fails when the requested identity, chain namespace, ligand
graph, or noncanonical bond order cannot be represented without loss. Use CIF for
ligands, peptide staples, nucleic acids with multi-character chain IDs, and covalent
designs.

Hydrogens are not model atoms. Public ligand counts are heavy-atom counts, and the
written ligand graph carries only the retained model/source heavy atoms and concrete
bond orders.

## Condition colors in CIF

New native CIF output writes `atom_site.B_iso_or_equiv` as an atom-level task
condition category, not pLDDT or an experimental displacement parameter:
context `0`, free design `20`, fixed identity `40`, supplied token center `60`,
supplied motif atom coordinates `80`, and hotspot `100`. Within a design atom,
geometry categories override identity; hotspot overrides all categories.
Unselected atoms in a partial motif retain their own category. Context keeps
`0` despite the generic known-identity/coordinate flags attached to context.

The notebook's fixed **Nord Light** workbench keeps Mol* mounted across all
six steps, including an empty Input or **De novo** task. Complete task presets
are selected in **1 Input & design**, loading input, modality, lengths, Motif,
Hotspots, and advanced settings together. Loading returns to that first step;
**4 Advanced conditions** edits the loaded task's additional settings. The first five steps
show source and applied-condition previews, using muted Context/Reference
geometry with stronger condition and selection highlights. Inference,
validation, clustering, and opening output structures are all in **6 Runtime**.
Reference is an authoring role, not another B-factor category.

The Input view includes full retained References. Their geometry can identify
Context Hotspot contacts independently of Motif selection, without supplying
model coordinates. Exported input omits unused References and retains Context
plus complete source chains needed by active conditions. Only selected Motif
atoms supply Motif coordinates; condition previews omit coordinate-free design
placeholders. Motif **Selection**, **Contig**, and **Atoms** areas remain visible
when Motif is off. A compact residue list and one current atom editor replace
the per-residue stack of editors; tasks with multiple Contigs have a chain selector.
Mol* offers **Chain**, **Residue**, and **Atom** scope controls in its upper-right
corner. Atom scope can pick an atom or a bond's endpoint atoms; condition-specific
Motif atom restrictions still apply. Question-circle help appears on hover or
keyboard focus, pins on click, and closes with Esc.

Native raw CIF results use the task-condition categories above. Validated
refolds retain native chain coloring and confidence values; those values are
not interpreted as task-condition categories. Raw category values must not be
read as pLDDT.

Existing files are not retroactively colored, and legacy files without this
column do not contain enough information to recover the authored categories
by rereading alone. Coordinate-only native refinement preserves the existing
CIF color column.

Notebook custom files are grouped under ignored `configs/customs/`:
`raw/` keeps original structures, `input/` keeps exported task structures,
and `tasks/` keeps exported YAML. Applied chain filtering updates the active
authoring Input without changing the raw original. Working normalized source,
condition previews, and prepared runtime snapshots remain under `.cache/`.
The notebook output root defaults to `configs/customs/inference`, with output
in `<output root>/<task name>/` exactly once. CLI output follows `--out` or its
configured `inference/<run_name>/` root.

## PartiScore and postprocess files

`score: true` writes `partiscore.csv` for successful base structures. A task-level
PartiScore row describes the structure actually selected for final scoring; it is
separate from a Best-of-N reward score. If scoring finds no applicable contact, the
status explains that condition instead of fabricating a numeric score.

`postprocess: true` enables the configured OpenMM stage. Its status is merged into
`result.csv`; a successful base row can therefore have `status=ok` and
`postprocess_status=failed`. Postprocess failures do not rewrite the base structure
unless `override: true` and the refinement succeeds.

## Validation files

Validation writes under the task leaf so several methods can be run independently:

```text
<task>/validate/<method>/
  metrics.csv
  success.csv
  logs/validate.log
  ...saved prediction structures and private state...
```

`metrics.csv` is the per-source/per-prediction record. `success.csv` includes rows
passing at least one requested criterion. Ordinary CLI validation without
criteria leaves it empty. The notebook explicitly enables unfiltered success
when its criteria field is empty: complete validation enters the unfiltered
cohort, while failed or incomplete validation remains unsuccessful. Optional
named criteria apply scientific thresholds instead.
`--refilter` rebuilds reports from saved metrics without loading a model.
`--recompute-metrics` reads saved prediction structures and recomputes metrics
without refolding. `--rerun` explicitly regenerates requested slots and may replace
the owned validation outputs.

## Notebook Runtime and result preview

**6 Runtime** contains Inference, Validate, Cluster, and Results & preview.
The first five steps retain source/condition previews and task export. All
54 native presets load complete tasks in **1 Input & design** and have visual forms for their existing values, including
joint and antibody tasks; native source paths and chain IDs are read-only,
and adding/removing entries uses applied Full task YAML edits.

The three **Prepare** buttons generate commands and required snapshots without
running them. Prepared inference uses a unique `.cache/notebook_runtime/inputs/`
directory containing task YAML, input structure, and any custom scaffold YAML
and referenced CIF. Later edits to exported/custom sources do not change those
prepared copies. Notebook compilation defaults off; larger jobs can optionally
enable **Compile model**.

**Execute** and **Modify command** default off. Commands are read-only and
selectable for manual copying. Enable Execute and click **Execute command** to
launch the displayed command in an owned detached tmux session. Modify command
enables editing of the actual arguments. Preparation, edits, and toggles never
launch automatically. Commands have no implicit shell; explicitly invoke
`bash -lc '...'` for shell syntax. **Background jobs & logs** refreshes owned
jobs/logs, stops the selected job, and reconnects after notebook or SSH loss.

Validate an existing task directory or its parent run directory containing
actual results. **Original · 1** is mandatory; ordinary and FI MPNN branches
have independent counts and temperatures, and FI has its own cutoff. AF3 keeps
Context MSA **Auto / reuse cache**, templates enabled, and native cache defaults.
Empty notebook criteria use the complete-only unfiltered cohort described
above. Cluster uses original successful design backbones, handles each method
and success cohort separately, defaults to TM threshold **0.5**, and limits
concurrency to **16** directories.

In **Results & preview**, refresh the selected result directory, choose a raw
sample and method, then select a successful refold by source/sequence and
prediction slot. The choices follow native result tables and saved prediction
paths. **Preview raw sample** and **Preview refold** reuse Mol* without starting
a model. Runtime also offers **Open a result file** for an existing or uploaded
PDB/CIF. Validated refolds retain native chain coloring and confidence values;
those values are not interpreted as task-condition categories. Native raw
structures retain their task-condition categories.

## Collisions and recovery

The selected task output leaf must be absent or empty before generation. A run can
reuse a container that already holds completed sibling task leaves. A non-empty
selected leaf is a run-fatal collision; change `run_name` rather than deleting a
previous result. Validation and its maintenance modes operate on an existing task
leaf and do not silently import another run's caches.
