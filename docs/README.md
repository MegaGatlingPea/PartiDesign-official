# PartiDesign inference

This guide is the user-facing reference for PartiDesign generation, search, and
validation. It follows the same small, task-first workflow used by PartiFlow-v2:
write a named YAML task, inspect the resolved plan, run the selected tasks, and
validate the structures that were written.

The maintained entry point is the installed `partidesign` CLI:

```bash
partidesign design configs/examples/protein.yaml \
  --task protein_fixture --run-name demo --out ./inference/demo
partidesign check configs/examples/protein.yaml --task protein_fixture
partidesign validate ./inference/demo --config configs/benchmarks/protdbench/validate.yaml --plan
```

Run commands from the repository root unless a command says otherwise. The CLI
uses that checkout as the default model root, and relative task, checkpoint, CCD,
config, and output paths are resolved against the captured working directory. A
task file is a mapping from task name to task dictionary; selected tasks must all
use the same format (`normal` or `antibody`).

## Start here

Use [Model downloads](download.md) for interactive selection and installation status.
[Benchmark commands](benchmarks.md) covers base/alpha/beta and cyclic validation.
[Refold numbering and chain-break repair plan](refold_breaks_plan.md) records the
code audit, proposed fixes, compatibility requirements, and real acceptance gates.


1. Read [Quickstart](quickstart.md) and run `--show-config` or `check` before
   loading a checkpoint.
2. Use [Task schema](task_schema.md) for the shared fields and condition grammar.
3. Choose a task-specific page for a runnable example:
   [protein](tasks/protein.md), [peptide](tasks/peptide.md),
   [nucleic acids](tasks/nucleic_acids.md), [ligand](tasks/ligand.md),
   [motif](tasks/motif.md), or [antibody](tasks/antibody.md).
4. Set routing, checkpoint, solver, omission, output, and device options in
   [Configuration](configuration.md).
5. Add explicit Best-of-N search and a reward only when candidate ranking is
   wanted; see [Rewards and search](reward.md).
6. Use [Score, relax, and validation protocols](protocols.md) for command YAML,
   benchmark profiles, criteria and precedence.
7. Run [Validation](validation.md) against the saved `result.csv` and structures.
   The stable output tree is described in [Outputs](outputs.md).
8. Cluster successful original design polymers on CPU with
   [Clustering](clustering.md).

The repository contains small fixture structures under
`configs/examples/structures/` and benchmark structures under
`configs/benchmarks/inputs/`. The YAML files in
`configs/examples/` use those repository-relative paths and are intended
for smoke runs after the required model assets have been installed.

## What a normal run does

Normal inference generates the model sequence and structure. It does not invoke
ProteinMPNN, LigandMPNN, CyclicMPNN, NA-MPNN, refolding, PartiScore, or OpenMM just
because those packages are installed. Select search and reward explicitly, or set
the task's `score`/`postprocess` flags for their separate final stages.

Free protein identity decoding uses task-level `rm_token_type: [CYS]` by default;
the global field remains `null` so a task can replace that policy or set `[]` to
disable it. This removes only free protein CYS identities; it does not remove RNA
`C`, DNA `DC`, context, motifs, or fixed sequence identities.

The selected task set is routed once to one checkpoint member for the whole
submission. Automatic routing can choose `omni`, `prot`, `pep`, `nuc`, `ligand`, or
`antibody`; a submission is never silently split into per-task model batches.
Normal and antibody tasks cannot share one submission. See
[Configuration](configuration.md) and [Antibody tasks](tasks/antibody.md).

## Scope of this guide

This guide documents the public CLI, YAML task grammar, sequence backends,
rewards, validation methods, and files users inspect after a run. It does not
describe training internals or the private temporary files used by model workers.
The executable source and packaged configs remain authoritative when adding a new
task or backend.
