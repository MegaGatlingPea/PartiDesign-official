# Screen sequence candidates

`partidesign screen` predicts supplied candidates against shared sequence-defined
contexts, selects each candidate's native representative, then ranks candidates.
Start with the bundled [screen tasks](../configs/examples/screen.yaml):

```bash
partidesign screen configs/examples/screen.yaml --plan
partidesign screen configs/examples/screen.yaml --task ligand_screen \
  --out screen_results/ligand --plan
```

Remove `--plan` to predict after preparing the selected models. Plans validate
the input tables and show sample counts, resources, MSA policy and pocket support
without loading prediction models or creating output directories.

## Task YAML and candidate CSV

Each named task has one candidate CSV and contexts shared by all of its rows.
Different tasks can use different contexts, methods and ranking settings:

```yaml
ligand_screen:
  methods: [boltz2]
  samples: 1
  msa: {mode: auto, source: colabfold, templates: true}
  contexts:
    A:
      type: protein
      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
      pocket: [8, 44, 68]
  candidates: ./configs/examples/screen/ligand_candidates.csv
```

CSV paths resolve from the invocation directory. The table must have exactly
`id,modality,seq` columns, with a nonempty unique ID for each candidate:

```csv
id,modality,seq
ethanol,ligand,CCO
propanol,ligand,CCCO
```

Allowed modalities are `protein`, `peptide`, `rna`, `dna` and `ligand`.
Polymer candidates use standard sequence letters; ligand candidates use one
connected SMILES. Context `type` uses the same modalities, with `sequence`
holding either letters or a ligand SMILES. Context chain IDs are independent
of candidate IDs. Invalid candidate rows retain a reported reason.

Choose any supported cofold method: `esmfold2-fast`, `esmfold2-standard`, `af3`,
`af3-la`, `mol1` or `boltz2`. Native method capabilities still apply to the
requested chemistry. AF2-IG requires a structure initial guess and is not a
sequence-only screening method.

Boltz-2 builds native sequence-defined inputs for protein, RNA, DNA and ligand
chains; peptide inputs use protein chemistry. Structure-based
[`validate --method boltz2`](validate.md#boltz-2-proteinligand-validation) retains
its narrower initial source-restoration support for protein context plus one
free designed ligand. Native affinity is conditional on an eligible designed
ligand/protein-context request. Input plans establish schema behavior;
GPU execution and prediction quality require separate evidence.

## Native representative, then candidate ranking

For each candidate and method, `samples` controls the number of complex
predictions. The highest native confidence selects its representative before
candidate ranking:

| Method                 | Representative score        |
| ---------------------- | --------------------------- |
| ESMFold2 fast/standard | `0.8 * ipTM + 0.2 * pTM`  |
| AF3, AF3-LA, Mol-1     | Native `ranking_score`    |
| Boltz-2                | Native `confidence_score` |

Candidate rankings are separate for each method and modality. Defaults are
descending `design_ipsae` for polymers, descending
`boltz2_affinity_probability_binary` for Boltz-2 ligand candidates, and descending
`ligand_iptm` for ligand candidates using other methods. Missing metrics remain
unavailable and unranked. Override the candidate ordering within a task with:

```yaml
ranking: {metric: design_plddt, order: desc}
```

`design_plddt` averages native design-atom confidence, excluding context atoms.
Sequence `design_ipsae` uses symmetric ipSAE for each eligible design/context
polymer pair and retains the highest pair score; directional values remain in
the prediction evidence. `binder_iptm` and `ligand_iptm` use scoped native
chain-pair confidence. These metrics use a 0–1 scale.

Boltz-2 affinity is evaluated for the selected structure representative when
the request has one designed ligand and eligible protein context. Its native
affinity calculation internally samples cropped poses and retains the selected
internal sample and pose evidence. The binding probability is in [0, 1]; value
fields retain native `log10(IC50 in µM)` units. See
[Boltz-2 validation](validate.md#boltz-2-proteinligand-validation).

## Context MSA, templates and pockets

Task-level `msa` settings are shared by the context chains. A context's own
`msa` mapping overrides those settings; CLI controls take precedence. Use
`--msa auto|cache-only|off`, `--msa-source colabfold|af3`,
`--msa-templates true|false` and `--msa-template-date DATE` as in pipeline and
validation. Search results use the same context cache. Designed candidates
remain query-only.

Fast uses no MSA or templates. Standard consumes fixed protein-context MSA
without templates. AF3/AF3-LA/Mol-1/Boltz-2 can consume shared protein templates;
the default maximum template date is 2021-09-30. Boltz-2 searches protein context
only. With the AF3 search source, AF3/AF3-LA/Mol-1 can also search fixed RNA
context. DNA remains query-only and ligands are not MSA queries. `--msa off`
disables both MSA and templates. See [MSA and templates](msa.md).

`pocket` lists **1-based residue positions in the supplied context sequence**.
Boltz-2 passes supported pockets to its native conditioning. Other methods
explicitly report the requested pocket as unapplied; `--plan` shows
`pocket_will_apply: false` for them. Pocket annotations do not add a proximity
postfilter or a new candidate acceptance cutoff.

## Resources, outputs and resume

Screen, pipeline and validate use the same resident prediction backends and
the same `--worker-cpus` and `--worker-gpus` controls. They reserve CPU/GPU
capacity per prediction worker; a GPU fraction is not a memory limit. Defaults
are one CPU per worker, 0.5 GPU for AF3/AF3-LA and one GPU for the other screen
methods. Omit the flags to retain these model defaults, and inspect the
resolved reservations with `--plan`. Explicit flags apply across all selected
tasks and methods; screen YAML has no task/method resource overrides.

Choose reservations for candidate count, total complex lengths and shapes,
available VRAM, host RAM and the CPUs available to the process. A small screen
may have too few requests to use extra workers; bulk screens need enough
memory for concurrent resident model copies. Increase reservations when fewer
concurrent workers are needed to fit memory, and check representative inputs
before reducing GPU fractions. More CPU per worker also reduces the number
of workers that the available CPU budget can support. There is no universal
best fraction for short turnaround or bulk throughput.

Method/configuration groups run serially; workers within the active group
process candidates in parallel. Screen has no candidate `batch_size` setting.
`samples` controls predictions per candidate and method, changing the
evaluation budget; retain the requested value while choosing worker resources.

Each task writes beneath `<out>/<task>/`:

- `scores.csv`: each candidate/method representative, metrics, status and reason.
- `samples.csv`: every completed native structure sample and its metrics.
- `rankings/<method>/<modality>.csv`: ranked representatives and unranked rows.
- `ligand_ranking.csv`: the Boltz-2 ligand ranking, when present.
- `predictions/`: structures, raw confidence and native affinity pose evidence.

`summary.json` records counts and metric units; `msa.csv` records prepared
context searches. Resume matching inputs with:

```bash
partidesign screen configs/examples/screen.yaml --task ligand_screen \
  --out screen_results/ligand --resume
```

Completed samples survive interruptions and failed siblings. Use a new output
directory when changing scientific inputs. `--msa-refresh` also requires a new
run and cannot accompany `--resume`.
