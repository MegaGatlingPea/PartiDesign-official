# PartiDesign user guide

After [installation](../README.md#installation), run design, refinement, and
validation in one command:

```bash
partidesign pipeline configs/examples/pipeline_tasks.yaml --task protdbench_insulin_receptor --preset ppi --run-name insulin_receptor
```

The [pipeline guide](pipeline.md) explains the default flow, final results,
workflow presets, and resume. Add `--plan` to preview the command. To author your
own molecules, choose a request from the [example index](../configs/examples/README.md)
and use the [Task YAML reference](task_schema.md) for context, identities, motifs,
secondary structure, and bonds.

See [environments](environments.md) for complete CUDA 12/CUDA 13 installs,
migration, and BioIR requirements. The [generation guide](design.md#execution-and-compilation)
explains automatic execution and compilation.

## Commands

Each command has a separate guide:

| Command | Use it to |
| --- | --- |
| [`download`](download.md) | Prepare generation and optional model assets |
| [`check`](check.md) | Check tasks, model assets, and the installation |
| [`design`](design.md) | Generate molecular structures and identities |
| [`msa`](msa.md) | Prepare shared context MSA and templates for selected tasks |
| [`pipeline`](pipeline.md) | Run design, refinement, and validation in one command |
| [`screen`](screen.md) | Predict and rank supplied sequences or SMILES against shared contexts |
| [`score`](score.md) | Score interfaces in saved structures with PartiScore |
| [`relax`](relax.md) | Refine existing structures with OpenMM |
| [`validate`](validate.md) | Evaluate saved designs with selected sequences, predictors, and criteria |
| [`postfilter`](postfilter.md) | Relax and screen passing protein and peptide complexes |
| [`cluster`](cluster.md) | Group passing designs by backbone similarity |

For saved generation results, use [validate](validate.md). Choosing a structure
predictor, choosing sequence sources, and deciding what passes are separate
steps; the guide explains how to make those choices and inspect the results.
For supplied candidates in an `id,modality,seq` CSV, use [screen](screen.md) with
the named tasks in [screen.yaml](../configs/examples/screen.yaml).
Successful protein–protein and standard peptide–protein complex refolds,
including antibody–antigen and head-to-tail cyclic peptide complexes, can then
receive optional [postfiltering](postfilter.md) through persistent Ray CPU actors.
Clustering independently compares original generated polymer backbones.

## Tasks and examples

- [Capabilities](../README.md#capabilities): illustrated protein, DNA, RNA, and
  ligand design.
- [A first run](design.md#a-first-run): check a task and generate initial structures.
- [PartiDesign MilkyFrog skill](../.agents/skills/partidesign-milkyfrog/SKILL.md):
  invoke `$partidesign-milkyfrog` in Codex for task preparation, workflows,
  troubleshooting, and usage questions.
- [Task YAML](task_schema.md): schema, lengths, structures, constraints, hotspots,
  chemistry, and antibody-specific fields.
- [Partial diffusion](partial.md): one reference CIF with task-defined conditions,
  schedule semantics, and optional native metadata recovery.
- [Pipeline](pipeline.md): fast iterations and standard final validation with
  fixed protein-context MSA. Explicit PPI uses only paired AF2-IG final validation,
  including the insulin receptor demo. AF3 and Mol-1 remain explicitly selectable.
  [Antibody pipelines](pipeline.md#antibody-pipelines) include GLP1R VHH de novo
  and annotated-complex optimization with standard final validation.
- [Ligand pipelines](pipeline.md#ligand-pipelines): Boltz-2 prediction,
  optional partial-diffusion rounds, and native final affinity.
- [Screen](screen.md): shared sequence contexts, native representatives and
  per-method candidate rankings.
- [Example index](../configs/examples/README.md): named runnable tasks by modality
  and peptide topology.
- [Command index](commands.md): all CLI entry points.
- [Performance](performance.md): complete compute-budget tables, sequence
  protocols, task-level figures, and timing boundaries.

Docking, antibody CDR design, and peptide design with multiple topologies are
supported for inference. Results can vary across targets. Run normal and
antibody tasks in separate submissions, as described in the
[generation guide](design.md).
