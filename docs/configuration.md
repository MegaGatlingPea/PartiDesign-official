# Configuration

PartiDesign composes a small closed inference configuration from
`configs/design.yaml`. The installed CLI takes an external task file as
its positional argument. Packaged task groups remain available to repository
workflows, but the user-facing examples use explicit task files.

```bash
cd /path/to/partidesign-inference
partidesign design configs/examples/protein.yaml \
  --task protein_fixture --show-config
```

The CLI resolves the repository as the default model root. See
[Quickstart](quickstart.md) for frontend checks and asset download commands.

## Main configuration

The effective top-level configuration has this shape:

```yaml
route: true
route_ckpt: null       # null | omni | prot | pep | nuc | ligand | antibody
task_names: null
tasks_file: null
tasks: {}
seed: 2026
batch_size: 16
num_workers: 4
pin_memory: false
run_name: design
output_dir: ./inference/${run_name}
output_format: cif
ccd_path: null
rm_token_type: [CYS]
model:
  checkpoint: ./ckpts/partidesign.ckpt
  precision: bf16-mixed
  context_constraint_mode: auto
  compile: {enabled: true, backend: inductor, mode: default, dynamic: true, fullgraph: true}
sampling:
  args: {nsteps: 400, self_cond: true}
  ligand_graph: {method: obabel}
search: {algorithm: null, selection_scope: task_length_plan, max_batch_size: null, best_of_n: {replicas: 1}}
reward: {_target_: null}
scorer: {checkpoint: ./ckpts/partiscore.ckpt, weights: raw, device: auto, batch_size: 32, max_pair_elements: 2097152, num_workers: 16}
postprocess: {force_field: amber19sb, charge_method: gasteiger, platform: CUDA, use_ray: true}
```

The field list is closed. Boolean values are not accepted as integers, numeric
values must be finite, and public YAML files must end in `.yaml`.

The released config files are grouped by command: `design.yaml` contains the
inline model/sampling/default-disabled search and reward policy; `score.yaml`
selects the shared scorer; `relax.yaml` is a closed DesignRunner postprocess
overlay; `validate.yaml` carries general validation defaults; and
`scorer/partiscore.yaml`, `search/best_of_n.yaml`, `sequence.yaml`, and
`criteria.yaml` hold reusable definitions. CLI flags and typed overrides apply
after the selected file, while task fields and explicit benchmark profiles take
precedence over generic defaults. The inline model and sampling blocks are authoritative
for the standalone design config; model and sampling defaults live directly in configs/design.yaml.


`seed` is one global seed for the run. It is recorded once in each task's ordinary
`logs/design.log`. A fixed seed does not promise bitwise-identical structures after
changing batch size, worker count, GPU count, or generation chunking.

`batch_size` is the FM batch size per rank. `num_workers` controls data workers.
Choose visible devices with `CUDA_VISIBLE_DEVICES`; do not put a GPU list or a
`distributed` block in the YAML:

```bash
CUDA_VISIBLE_DEVICES=0,1 partidesign design \
  configs/examples/protein.yaml \
  --task protein_fixture --batch-size 2 --run-name two_gpu_demo
```

Lightning uses the visible devices and assigns DDP rank streams from the global
seed plus global rank. One checkpoint member is selected for the entire submission;
tasks are not auto-batched into separate model-member groups. A submission that
needs more than one routing preference is sent to `omni` as one submission.

## Paths, tasks, and output

Capture the invocation directory before composition. Relative paths in the main
config, an external task YAML, and the paths nested inside that YAML all resolve
against that invocation directory. They do not resolve against the task file's
parent or Hydra's temporary directory. URI download paths are not accepted.

`task_names` is either `null` (all tasks in YAML order) or a non-empty list of unique
task names. `output_dir` defaults to `./inference/${run_name}` after interpolation.
It is a reusable run container: unrelated completed task leaves may be present, but
each selected `<output_dir>/<task_name>` leaf must be absent or empty. There is no
overwrite, resume, archive, or skip-existing mode.

`output_format: cif` is the default and authoritative format. `pdb` is available
only when the selected structure is representable in PDB; multi-character chain
IDs, ligand chemistry, or noncanonical cross-chain bond orders can make PDB output
invalid. Postprocessing requires CIF.

`ccd_path` is an optional local CCD/mmCIF component dictionary used when a source
ligand needs missing topology. File-declared graph information takes precedence;
CCD is a fallback for absent topology and never invents a bond from a coordinate
distance.

## Model and sampling

The model config selects one combined checkpoint and the fixed solver contract:

```yaml
model:
  checkpoint: ./ckpts/partidesign.ckpt
  precision: bf16-mixed
  context_constraint_mode: auto  # task-aware context policy
  compile:
    enabled: true
    backend: inductor
    mode: default
    dynamic: true
    fullgraph: true
```

The checkpoint path names one combined FM/VAE artifact. The loader validates both
namespaces strictly. It does not pair an FM file with a second VAE file or search
for a preferred filename. Compile is enabled by default and applies to the FM
denoiser trunk; the VAE remains eager. `backend=inductor`, `mode=default`,
`dynamic=true`, and `fullgraph=true` are fixed even when `enabled=false`; disabling
compile is an eager diagnostic and does not provide an automatic fallback.

The default sampling group is 400 steps with `self_cond: true`. Token-center and
local-latent schedules remain separate; polymer updates use the SC path and ligand
updates use the configured ligand solver path. The selected context mode is applied at each step,
and the previous clean prediction is detached before reuse. `tsr_k` and `tsr_sigma`
may appear in the resolved config for compatibility but do not change the current
validated solver output.

## Automatic routing

`route: true, route_ckpt: null` selects the member from the selected tasks only.
`route_ckpt` can force `omni`, `prot`, `pep`, `nuc`, `ligand`, or `antibody` while
retaining strict compatibility checks. `route: false` loads the explicitly named
standalone checkpoint and ignores a valid selector; it does not load a multi-member
bundle.

Routing preferences are resolved before generation:

1. `format: antibody` selects `antibody` and is isolated from normal tasks.
2. Explicit peptide design units select `pep`.
3. Independent cross-modality design or coordinate motifs select `omni`.
4. Ordinary protein, nucleic-acid, and ligand-only tasks select `prot`, `nuc`, and
   `ligand` when no higher-priority context applies.

Known ligand context retained by the source crop and mixed modality joints are
therefore resolved as one `omni` submission. Routing is execution selection, not a
quality guarantee. It does not unload a resident model between normal batches or
switch network members halfway through a selected submission.

## Token omission policy

`rm_token_type` is a list of canonical 32-token names and applies only to free
polymer identity decoding. Every public FM task uses `[CYS]` by default. That
task-local default is applied when the task omits `rm_token_type` or sets it to
`null`; it does not require a global YAML value. The main config keeps the global
field `null` so tasks can retain that local policy. The parsed task/API view reports
`[CYS]` for an omitted or `null` task field even though the raw global field remains
`null`. The field has two scopes:

```yaml
# Top-level design.yaml
rm_token_type: [CYS, C, DC]

# A task may provide its own list when the global value is null.
protein_task:
  rm_token_type: [CYS]
```

The same global policy can be supplied on the installed CLI with a typed override:

```bash
partidesign design configs/examples/protein.yaml \
  --task protein_fixture \
  --override 'rm_token_type=[CYS,MET]' \
  --show-config
```

The precedence is:

| Global value | Task value | Effective policy |
| --- | --- | --- |
| `null` or omitted | omitted/`null` | `[CYS]`, the public task default. |
| `null` or omitted | list | That task list, including `[]` to disable omission. |
| non-null list | any task value | Global list overrides every task list. |
| `[]` | any task value | Empty global list clears omission for every task. |

The accepted names are:

```text
Protein: ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL UNK
RNA:     A G C U N
DNA:     DA DG DC DT DN
Other:   -
```

Names are case-insensitive in YAML and normalized to their canonical uppercase
spelling. The default `[CYS]` removes only the free protein `CYS` identity; it does
not remove RNA `C` or DNA `DC`. Omission does not remove context, motif, fixed
sequence, explicit ligand identity, or an atom required by a declared bond. An
unknown motif identity still contributes its inferred motif coordinates; omission
affects only the inferred free identity. This policy is separate from MPNN
one-letter omission, which is described in [MPNN](mpnn.md).

## Search and reward composition

The search group is closed and has one selection scope: task plus resolved length
plan. The default is no search and no reward:

```yaml
search:
  algorithm: null
  selection_scope: task_length_plan
  max_batch_size: null
  best_of_n:
    replicas: 1
reward:
  _target_: null
```

The public search option is the `configs/search/best_of_n.yaml` overlay with one selected reward profile.
`replicas` expands each resolved base request; the current runtime writes all
successful ranked replicas to the task leaf. See [Rewards and search](reward.md)
for backend choices, score scope, and worker lifecycle.

When a reward profile is selected, its sequence source defaults to `design`: the
generated FM sequence is used and no MPNN call is made. Set
`reward.sequence.source=auto` or an explicit backend name only when sequence
redesign is intended. The shared `configs/sequence.yaml` policy routes
auto protein/cyclic requests to SolubleMPNN and keeps nucleic requests on design.

Every shipped reward profile keeps an explicit `sequence.proteinmpnn` slot,
including checkpoint, temperature, `use_soluble_model`, and omission fields. For
`source: auto`, those fields are applied to the selected automatic backend after
the runtime captures the effective settings in its internal `auto_proteinmpnn`
provenance block. This keeps an explicit `model_type: protein_mpnn` or scalar
override authoritative without introducing a second public backend schema.

Native FM omission is controlled by task/global `rm_token_type`: use `[]` to
disable native token omission for the submission. MPNN omission is independent:
`sequence.*mpnn.rm_aa: ""` explicitly disables one-letter amino-acid omission,
whereas a non-empty string such as `C` omits those free MPNN positions. Neither
setting changes fixed identities, motifs, or explicit ligand chemistry.

## Postprocessing

The optional OpenMM stage is selected per task with `postprocess: true` and uses the
run-wide block:

```yaml
postprocess:
  force_field: amber19sb
  charge_method: gasteiger   # gasteiger | am1bcc
  platform: CUDA             # CUDA | CPU
  use_ray: true
```

The production pairing is CUDA plus Ray; the local diagnostic pairing is CPU plus
no Ray. `gasteiger` is the fast default route. `am1bcc` is an explicit slower
compatibility route. Postprocessing acts on successful base structures and writes
its status to `result.csv`; it is not part of the base FM sampler.
