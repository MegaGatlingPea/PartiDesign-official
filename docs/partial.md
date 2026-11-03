# Partial diffusion from one structure

[Design command](design.md) · [Task YAML](task_schema.md) · [Pipeline](pipeline.md)

Partial diffusion generates new candidates around **one reference CIF**. Write
a native task YAML describing what to design and what to retain, then add
`partial` to choose the starting structure and how much sampling to run.

The task defines design and context roles, fixed identities, motifs, bonds and
offsets. The CIF supplies the complete current complex geometry. You do not need
a task file from an earlier run or an input `result.csv`. This route supports
normal and antibody tasks with a flex-context checkpoint; the reference must be one CIF file.

## Input task and command

[configs/examples/partial_diffusion.yaml](../configs/examples/partial_diffusion.yaml) shows a minimal
100-residue protein example with a bundled ESMFold2-fast refold. The demo needs
only that reference CIF and its task YAML, and runs directly from a checkout:

```yaml
partial_protein:
  input: null
  sample_num: 4
  design_chains: [{P: protein}]
  condition:
    secondary_structure:
      coarse: {P: alpha}
  partial:
    reference: ./configs/examples/structures/partial_diffusion/protein_refold.cif
    fraction: 0.625
```

Set `sample_num` to the number of new structures to generate around the
reference. The bundled example runs with:

```bash
partidesign design configs/examples/partial_diffusion.yaml --run-name protein_partial --show-config
partidesign design configs/examples/partial_diffusion.yaml --run-name protein_partial
```

`--show-config` previews settings without loading models or creating output.
Children and their `result.csv` go to
`inference/protein_partial/partial_protein/`. Use a fresh output task directory
for each generation run.

| Partial field | Input contract |
| --- | --- |
| `reference` | Required path to one actual candidate `.cif` with complete current complex geometry. |
| `result_csv` | Optional explicit native output CSV for saved locator and motif metadata. Omit it to use the CIF and task directly; no sibling CSV is loaded. |
| `fraction` | Required finite number `f` in `(0, 1]`, with `round(f * 400) >= 1`. |
| `cdr_lengths` | Antibody-only actual CDR counts by role, used when the reference layout is ambiguous; see below. |

These are the fields accepted inside `partial`. Booleans are rejected for the
fraction. Relative paths resolve from the invocation directory.

With `result_csv` omitted, chain roles and permissions come from the task and
actual lengths come from the CIF. Use matching design-chain labels in the task
and reference. Fixed identities, motifs, required bonds and cyclic offsets
come from the task YAML; loading a complete reference does not fix its whole
sequence. For contigs, provide the source input and exact gap lengths in the
task. Resolve ambiguous gaps there before running.

To restore native saved locator or motif mappings, set `result_csv` explicitly.
That CSV must contain exactly one successful row whose `filename` resolves to
the reference CIF; relative filenames resolve beside the CSV. Preserve its
`design_chains`, `fixed_restypes`, `bond_constraints`, `contig_string` and
`tip_atoms`. Known-empty fixed identities and bonds use `[]`, not blank cells.
An explicit missing, invalid or unmatched CSV is an error. For example, a saved
pipeline refold can use its visible CIF and round-level metadata:

```yaml
partial:
  reference: ./inference/RUN/TASK/round_1/refold/r001_c0000001.cif
  result_csv: ./inference/RUN/TASK/round_1/result.csv
  fraction: 0.625
```

`sample_num` counts replicas of this one reference. Actual design lengths come
from its atoms, so `design_length` is not required when the reference determines
every chain's size. Contigs use the task's exact source spans and gaps, or
explicit saved mappings. The runtime does not draw new lengths or gap sizes.
It batches different partial fractions and ordinary generation separately.

## Antibody references

VHH and paired VH/VL tasks use the same single-CIF `partial` entry through
`partidesign design` and through directory-start pipeline optimization.
Keep `format: antibody`, the native `antibody.scaffold`, role-to-chain mapping,
CDR/framework design choices and any `antigen` conditions. The reference must
contain the full current binder and retained context with matching chain roles.
An antigen-only source or an unrelated antibody structure is not a partial
reference for the chosen scaffold.

Partial uses the reference's actual FR/CDR layout and checks it against the
authored lengths. It does not draw new loop lengths. Saved native region
metadata can restore that layout; otherwise it must be uniquely determined
from the declared scaffold and lengths. When it is ambiguous, add the actual
lengths under `partial.cdr_lengths`, alongside your real `reference` and
`fraction`. For a VHH, the extra field has this form:

```yaml
cdr_lengths:
  VHH: {CDR1: 9, CDR2: 8, CDR3: 21}
```

Use `VH` and `VL` for paired antibodies. Every supplied role contains all three
positive CDR lengths; a role whose layout is already unambiguous may be omitted.
The counts must describe the actual reference and lie within the task's allowed
ranges. This field is specific to antibody partial and is not an antibody
numbering or automatic IMGT assignment.

The bundled [GLP1R seeds](../configs/examples/antibody_opt/annotation.csv) use
`bindcraft2_vhh` and actual VHH CDR lengths `(7,5,25)`, `(7,7,10)`, `(7,6,8)`,
`(7,5,21)` and `(7,7,6)`. Each row declares these counts in its native
`partial` object, and derives the current full-complex reference from `filename`.
Its antigen input is that same complete seed cropped to label chain `A`.
Label residues 1–119 correspond to author residues 21–139 on the antigen;
source scaffold positions and final binder positions use their own label axes.

Framework and unselected CDR identities remain fixed unless the native task
opens them for redesign. `template: true` retains soft framework geometry
guidance; it does not freeze coordinates or expose CDR coordinates as a
template. The complete current complex supplies the partial starting state,
while scaffold template geometry and antigen conditions keep their own roles.
VH and VL remain one paired binder throughout MPNN, refolding and final
evaluation. Select a multi-chain predictor for paired antibodies; AF2-IG
requires a single design chain.

## Conditions and reference geometry

Define `input`, crop, design chains, conditions, hotspots, cyclic offsets and
required chemistry as you would for ordinary design. If you are continuing an
existing task, keep the conditions you want to preserve. Source files used by
motifs or context, and any required CCD definitions, must remain available.
If you supply saved CSV metadata, its identities, bonds, motifs and chain roles
must agree with the task.

The reference supplies the starting geometry and the coordinates at existing
condition sites, including the current context pose. The complete complex goes
through the same atom/token mapping, centering and unit conversion as ordinary
inference, keeping design and context in the same coordinate frame. Free design
positions remain free; loading the reference adds no new coordinate conditions.

The motif evaluation reference remains the original source, with the task's
resolved placement or an explicit saved motif mapping. It is separate from the
current coordinates used at motif condition sites. Consistency RMSDs compare
against the current parent.

Incomplete required geometry, ambiguous mappings, incompatible fixed identities
or unavailable chemistry stop the request before sampling. Read the failed
row's `error` and `logs/design.log` for the cause.

## Meaning of the fraction

The native sampler uses 400 intervals. For fraction `f`:

```text
retained intervals = round(400 * f)
start index = 400 - retained intervals
```

`fraction: 0.625` runs the last **250 intervals**, from index **150** through
399. The VAE reference latent is sampled from its posterior; token centers use
the current geometry. Each component mixes its native prior with its reference
at that common index and continues on the unchanged native grid.

The two components have separate schedules, so their numeric start times can
differ. Fraction is neither a shared numeric time nor an RMSD/displacement
bound. Explicit component start times and `couple_partial_t` are not task fields.
At the full-generation boundary, including `fraction: 1`, the runtime delegates
to ordinary full generation after checking the reference inputs.

## Outputs and pipeline use

Partial output uses the native design directory and `result.csv`. Ordinary
partial rows keep native identity, motif and bond metadata, plus:

| Metadata | Meaning |
| --- | --- |
| `partial_reference`, `partial_fraction` | Resolved reference CIF and requested fraction. |
| `partial_result_csv` | Explicit metadata CSV path; blank for CIF-only input. |
| `partial_start_index`, `partial_intervals` | Actual start index and retained interval count. |
| `partial_t_center`, `partial_t_latent`, `partial_latent_strategy` | Component start times and posterior/prior initialization. |
| `partial_rng_seed`, `partial_rng_stream` | Native RNG seed and rank-local stream. |
| `partial_checkpoint`, `partial_checkpoint_member`, `partial_context_mode` | Model actually used. |

Antibody output retains its native heavy/light sequences, FR/CDR ranges and
mutable-position fields. Its partial provenance and resolved constraints remain
in the run metadata under `.work/antibody/`; the public table keeps the antibody
schema rather than adding ordinary-task JSON columns.

The [pipeline](pipeline.md) also accepts an optimization directory with
`annotation.csv`; it first applies partial to each listed complete complex,
retaining each original by default. See [directory optimization](pipeline.md#optimize-existing-structures).
The initial partial step is separate from the refold round count.

Between rounds, the pipeline fills the same `task.partial` in memory for each
selected refold and calls the same native design runtime. Its iteration
`partial` operation controls `fraction`, `replicas` and parent retention; task
`partial` supplies one reference CIF and optional explicit metadata to
`partidesign design`. Pipeline calls pass their resolved internal CSV metadata
explicitly, preserving task constraints and mappings.

Pipeline defaults use fraction 0.625, four replicas and `include_parent: true`;
relaxation is off, and the last round has no transition. Middle rounds that
continue keep at most two candidates per parent/partial family after per-input
selection and sequence deduplication. Pipeline motif RMSD is a strict
round-1/final check; intermediate rounds retain motif permissions without
recalculating motif geometry. Pipeline outputs group
partial samples under `<task>/round_N/partial/<parent>/sample*.cif`. Final
validation is configured separately from partial generation; see the
[pipeline output and final-validation guide](pipeline.md#final-validation).
