# Prepare shared MSA and templates

`partidesign msa` searches the fixed context in selected YAML tasks or annotated
optimization directories before a pipeline run. It uses the same task selection
and crop rules as `design` and `pipeline`,
without generating designs or loading prediction models.

The pipeline also prepares these inputs automatically before GPU work starts.
For [sequence screening](screen.md), task and per-context `msa` settings in
`SCREEN.yaml` prepare the same shared context evidence before prediction.
The separate command lets you prepare and inspect the shared cache in advance.
Identical queries with the same search profile are prepared once, even when
several tasks, rounds or final prediction methods need them.

```bash
partidesign msa configs/examples/pipeline_tasks.yaml \
  --task protdbench_insulin_receptor --source colabfold --plan

partidesign msa configs/examples/pipeline_tasks.yaml \
  --task protdbench_insulin_receptor --source colabfold
```

The first command previews queries and cache reuse. The second prepares missing
results. Use the same ordinary workflow settings as the pipeline. The explicit
PPI preset has no default MSA consumer, so it needs no separate pre-search.
The standalone `msa` command is an explicit pre-search request: invoking it
with a PPI workflow still uses its shared MSA policy, including the shared
template default. Use `--plan` to inspect such optional pre-searches first.
`--task` is repeatable; without it, all tasks in the input are selected.
YAML task paths resolve from the invocation directory, as they do for `design`.

For an optimization directory, use the same annotation input and task groups:

```bash
partidesign msa configs/examples/optimization \
  --task cyclic_peptides --config configs/pipeline/optimize.yaml --plan
```

Remove `--plan` to prepare missing inputs. Each listed seed supplies its own
retained context and authoring conditions; identical queries with compatible
settings share the cache. Annotation paths resolve relative to `annotation.csv`.

## Defaults and search scope

Ordinary pipeline final uses ESMFold2-standard and **ColabFold by default** for
fixed protein-context MSA, without templates. Fast iterations and designed
chains remain query-only. Explicit AF3/AF3-LA/Mol-1 can also request protein
templates; their default maximum template release date is **2021-09-30**.
Boltz-2 in validate, screen and the ligand pipeline preset uses the same fixed
protein-context MSA/template policy.
`--msa off` disables MSA and templates. Without eligible fixed protein context,
standard naturally has no MSA query.

| Source | Fixed protein context | Fixed RNA context | DNA |
| --- | --- | --- | --- |
| `colabfold` (default) | Search MSA; templates only for template-enabled consumers | Query-only | Query-only |
| `af3` | Local protein MSA; templates only for template-enabled consumers | AF3/Mol-1 can search RNA locally; standard remains query-only | Query-only |

Designed chains remain query-only for every source, including peptides,
antibody heavy/light chains, RNA and DNA. Ligands are not MSA queries. Original
generation coordinates, motifs and partial-diffusion references are not
automatically prediction templates.

Search source and prediction method are separate choices. AF3, Mol-1 and
Boltz-2 can read the same ColabFold result. Choosing `source: af3` selects the local search
route; it does not choose the prediction model. AF2-IG keeps its existing
initial-guess protocol.

If a query depends on a structure produced later in the pipeline, it is prepared
when that actual input becomes available. The pre-search command does not invent
design sequences or sample their lengths.

## Use the same settings in a workflow

```yaml
preset: protein
msa:
  mode: auto
  source: colabfold
  max_template_date: "2021-09-30"
```

Shared settings apply to ESMFold2-standard, AF3/AF3-LA, Mol-1 and Boltz-2. Standard
consumes fixed protein-context MSA without templates. Model-specific MSA settings
remain supported and can override the shared workflow settings. Corresponding
CLI options take precedence. Use `--show-config` to inspect the resolved values.

```bash
partidesign pipeline configs/examples/pipeline_tasks.yaml \
  --task protdbench_insulin_receptor --preset protein \
  --msa-source colabfold

partidesign validate inference/RUN --method mol1 --source gt:1 \
  --msa-source colabfold --msa auto

partidesign validate inference/ligand_run --method boltz2 --source gt:1 \
  --msa-source colabfold --msa auto --plan

partidesign screen configs/examples/screen.yaml --task ligand_screen \
  --msa-source colabfold --msa auto --msa-templates true --plan
```

In `validate`, `--source gt:1` selects sequences, while `--msa-source` selects
their context-search source. Boltz-2 uses `boltz2.msa` in a validation profile
and the same `auto`, `cache-only`, `off` and template CLI controls below.
Screen applies task-level settings to all shared context chains, then applies
each context's own `msa` overrides and finally CLI overrides. Candidates remain
query-only. Boltz-2 and ESMFold2 consume protein-context searches only.

| MSA mode | Behavior |
| --- | --- |
| `auto` (default) | Reuse a compatible ready entry; search when missing |
| `cache-only` | Require a compatible ready entry; do not search |
| `off` | Use query-only input and no structural templates |

To keep protein MSA but omit structural templates, set `msa.templates: false`
or pass:

```bash
partidesign pipeline configs/examples/optimization \
  --preset protein --config configs/pipeline/optimize.yaml --msa auto --msa-templates false
```

This requests alignments only: no HHsearch or structural-template preparation
is performed. A compatible cached MSA+template result can supply an MSA-only
view without another search; its templates are not passed to the predictor.
The cached result remains unchanged. The reverse is not sufficient: an
MSA-only entry does not satisfy a template-enabled request.

## Cache location and contents

The default cache is:

```text
<model_dir>/.cache/af3_msa/
  index.sqlite
  profiles/
  protein/entry000001/generation000001/
    manifest.json
    unpaired.a3m
    paired.a3m
    template000001.mmcif
  rna/entry000002/generation000001/
    manifest.json
    unpaired.a3m
  .locks/
  .work/
```

The historical `af3_msa` directory name is retained for compatibility. It is
shared across prediction methods, tasks and runs; there are no separate Mol-1,
Boltz-2 or ColabFold copies. Select another model root with `--model-dir`, or set
`msa.cache_dir` in the workflow. The run's compact `msa.csv` records the task
and chain, search source, cache entry/profile/generation, MSA depth and template
count. MSA and template files are shared rather than copied into each round or
final-method directory.

Entries match the exact query and search profile, including source, endpoint,
search parameters and template-date policy. An AF3-source entry is not silently
used for a ColabFold request. Old entries remain readable with `source: af3`.
Each generation's `manifest.json` records its template files, query-to-template
residue mappings and source/date information.

For the current ColabFold route, `unpaired.a3m` contains search hits and
`paired.a3m` contains only the query. Cross-chain paired MSA search is not
provided by this shared result.

The pipeline pins the selected cache generation for the run. Later rounds and
prediction methods reuse it; `--resume` restores the same references even if
another run has since published a newer generation. `--msa-refresh` prepares
one new generation per unique query/search profile for a new run, rather than
refreshing for every candidate or prediction method. Resuming an existing run
keeps its pinned generations. For separate preparation, use
`partidesign msa --refresh`.

Concurrent duplicate requests share a producer. Refresh publishes new generations
without overwriting old ones. A cache-only miss is an error, while a successful
search with no usable hits is a reusable query-only result. Search or download
failures are reported as failures rather than recorded as no-hit results.

Do not edit a published generation's files. Keep the cache when moving or
resuming a run that still needs its MSA/template inputs; restore a missing pinned
generation before resuming. The pipeline's default prediction-cache cleanup does
not remove this shared search cache.

## Templates and search resources

Selected protein templates use one protein chain, its complete polymer sequence
and a residue mapping that includes positions without observed atoms. AF3,
Mol-1 and Boltz-2 each construct their own model features from that
common evidence. Boltz-2 preserves the exact mapping across insertions, deletions
and missing coordinates; the generated pose is not used as a template. At most
four templates are used. Date filtering uses the structure's first release date,
not the download time or a placeholder supplied by a prediction backend.

ColabFold requests use its MSA service and template-profile results; HHsearch
builds the template alignments. The supplied environments include
`colabfold-binaries` for HHsearch without installing the full ColabFold prediction
environment. Public-service requests are serialized. For a private compatible
service, use `--msa-server-url URL`; its endpoint is recorded in the cache profile.

The `af3` source needs the existing local sequence/template databases and HMMER
tools. `partidesign download` installs model assets, not those search databases.
Inspect the selected route before running:

```bash
partidesign check --model mol1 --msa-source colabfold
partidesign check --model boltz2 --msa-source colabfold
partidesign check --model af3 --msa-source af3
```

Preparation does not load prediction models or reserve a local GPU. Network
errors, missing tools and template-preparation errors do not cause an automatic
switch to another source or to MSA-off input.

[Pipeline](pipeline.md) · [Validation](validate.md) · [Command index](commands.md)
