# Validation runtime

The maintained user interface is `partidesign validate`. See the
[user guide](../../docs/inference/validation.md) for methods, sequence sources,
criteria, threshold overrides, workers, resume, and result files.

```bash
partidesign validate inference/demo --method af2_ig --plan
partidesign validate inference/demo --method af2_ig
python -m src.validate inference/demo --method af2_ig --plan
```

Main defaults live in `configs/validate.yaml`; success presets live in
`configs/criteria.yaml`. Explicit benchmark tasks/protocols live under
`configs/benchmarks/<family>/`. Shared coordinates live under `configs/benchmarks/inputs/`.
Normal and antibody generation retain their result.csv/structure contracts.

Protein/cyclic fold validation defaults to GT1+SolubleMPNN1. Nucleic sequence
validation defaults to GT1+NA-MPNN1. Explicit source counts, settings and labels
preserve benchmark budgets. AF2 uses model1 by default; AF3/AF3-LA do not search
MSA/templates and do not preallocate JAX GPU memory by default. GPU-backed
validation uses persistent Ray owners; neural backends stay resident for the
owner lifetime. Missing/unsupported metrics cannot silently satisfy a preset.

CPU-only metrics and reporting can be recomputed from persisted structures.
Use `--refilter` after criteria changes, or `--recompute-metrics` after metric
implementation changes. Neither option regenerates model samples.
