# PartiDesign user guide

Start with the [installation and first run](../README.md#installation), then
choose a named request from the [example index](../configs/examples/README.md).
The [Task YAML reference](task_schema.md) explains every input field and how to
combine molecular context, identities, motifs, secondary structure, and bonds.

See [environments](environments.md) for complete CUDA 12/CUDA 13 installs,
migration, and BioIR requirements. The [generation guide](design.md#execution-and-compilation)
explains automatic execution and compilation.

## Commands

Each command has a separate guide:

| Command | Contents |
| --- | --- |
| [`download`](download.md) | Generation and optional model assets, registered downloads, and model directories |
| [`check`](check.md) | Input checks, strict CPU loading, real smoke jobs, and installation troubleshooting |
| [`design`](design.md) | Generation, routing, Best-of-N ranking, and outputs |
| [`score`](score.md) | PartiScore scoring for saved structures |
| [`relax`](relax.md) | OpenMM refinement of existing results |
| **[`validate`](validate.md)** | **A complete evaluation workflow: methods, sequences, criteria, reports, and reruns** |
| [`postfilter`](postfilter.md) | Ray CPU relaxation, automatic protein/peptide screening, physical cyclic closure, and sample-level resume |
| [`cluster`](cluster.md) | Clustering passing design backbones |

After generation, start with [validate](validate.md). Choosing a structure
predictor, choosing sequence sources, and deciding what passes are separate
steps; the guide explains how to make those choices and inspect the results.
Successful protein–protein and standard peptide–protein complex refolds,
including antibody–antigen and head-to-tail cyclic peptide complexes, can then
receive optional [postfiltering](postfilter.md) through persistent Ray CPU actors.
Clustering independently compares original generated polymer backbones.

## Tasks and examples

- [PartiDesign MilkyFrog skill](../.agents/skills/partidesign-milkyfrog/SKILL.md):
  invoke `$partidesign-milkyfrog` in Codex for task preparation, workflows,
  troubleshooting, and usage questions.
- [Task YAML](task_schema.md): schema, lengths, structures, constraints, hotspots,
  chemistry, and antibody-specific fields.
- [Example index](../configs/examples/README.md): named runnable tasks by modality
  and peptide topology.
- [Command index](commands.md): all CLI entry points.
- [Performance](performance.md): complete compute-budget tables, sequence
  protocols, task-level figures, and timing boundaries.

Docking, antibody CDR design, and peptide design with multiple topologies are
supported for inference. These Preview workflows remain under active development,
and results can vary across targets. Run normal and antibody tasks in separate
submissions, as described in the [generation guide](design.md).
