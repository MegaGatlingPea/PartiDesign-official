---
name: partidesign-milkyfrog
description: "Use PartiDesign for molecular inference: write task YAML, plan or run generation, score and refine structures, validate saved designs, postfilter successful protein and peptide complexes, cluster generated backbones, and troubleshoot usage. For end users rather than model training or software development."
---

# PartiDesign MilkyFrog

Turn the user's molecular design goal into the appropriate PartiDesign request,
or answer their usage question from the active release. Keep model training,
runtime implementation, and software changes outside this skill's scope.

## Read only what the task needs

| Need | Read |
| --- | --- |
| Environment/assets, generation, Best-of-N, existing-structure scoring, refinement retry, clustering, outputs | [workflows.md](references/workflows.md) |
| Select an example or author molecular inputs, identities, motifs, topology, and antibody tasks | [task-recipes.md](references/task-recipes.md) |
| Choose validation methods, sequence sources, criteria, MSA policy, or reuse saved evaluation | [validation.md](references/validation.md) |
| Relax and screen successful protein–protein or standard peptide–protein refolds, including physical head-to-tail cycles, with persistent Ray CPU actors | [postfilter.md](references/postfilter.md) |
| Explain behavior, interpret a failure, or answer a common usage question | [faq.md](references/faq.md) |

Start with the relevant reference; load another only when the request crosses
that boundary. The references point to specific repository guides for detail.

## Find the active installation

Use the user's selected checkout, environment, and working directory. Locate
the matching PartiDesign CLI/package, configs, and repository docs dynamically;
never derive them from this skill's installation directory. A checkout contains
`pyproject.toml`, `partidesign/cli.py`, and `src/design/`. Documentation paths
such as `docs/design.md` in these references are relative to that checkout.
The current CLI, schema, and configs take precedence over historical or private
deployment conventions. If only an installed package is available, inspect its
CLI and packaged resources and identify any missing matching documentation.

The model root is separate: `--model-dir` takes priority over
`PARTIDESIGN_MODEL_DIR`, then the detected checkout. Task inputs, explicit
checkpoint paths, and outputs resolve from the invocation directory.

## Preserve intent and execution scope

- A usage question calls for an answer. A request for a plan or config check
  calls for those checks. Neither starts downloads, model loading, or inference.
- For requested execution, proceed within the existing authorization after
  checking prerequisites. Do not add an approval ritual merely because a GPU
  is involved. Keep environment changes, downloads, output replacement, and
  compute within the requested scope.
- Preserve the selected tasks, modalities, chains, lengths, samples, conditions,
  sequence sources, criteria, MSA policy, and output choices. State assumptions
  that affect the result. Do not silently substitute a smaller trial, add MPNN
  draws or Best-of-N, or weaken chemistry or passing thresholds.
- Start with bounded inspection: `design --show-config`, ordinary `check`, or
  the relevant `--plan`. Explain what was checked; these do not establish
  successful model execution or quality. Run the requested job with the same
  intended settings once prerequisites are met.
- Use explicit task selectors when a file contains additional requests. Batch
  size changes execution memory, not the task's total sample budget. Submit
  normal and antibody tasks separately.
- Inspect existing outputs before retries. Use each command's documented resume
  or rerun behavior; validation also has refilter/recompute operations. Postfilter
  resumes complete samples and requires a fresh output directory for older
  output formats. Preserve requested replacement boundaries; generation requires
  new or empty task directories.

Report the answer or concrete artifact first. For execution, include the actual
command/settings, output paths, observed completion/failures, and any remaining
blocker. Distinguish a parsed plan, completed computation, and passing
computational criteria; none establishes experimental binding or activity.
