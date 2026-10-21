# Score, relax, and validation protocols

The command-level YAML files are intentionally separate from design tasks:

- configs/score.yaml selects scorer/partiscore.yaml for standalone
  coordinate scoring. Input directories and overwrite policy are explicit CLI
  inputs.
- configs/relax.yaml is a closed DesignRunner overlay. It changes
  only the postprocess stage to OpenMM and keeps the fixed amber19sb,
  Gasteiger-by-default, CUDA+Ray production pairing (CPU+no-Ray is the local
  diagnostic).
- configs/validate.yaml carries general validation defaults. A
  benchmark profile under configs/benchmarks/<family>/validate.yaml supplies its
  explicit method, source list, backend temperature, AF2 model, structure stage,
  and criteria.

Examples:

    partidesign score --config configs/score.yaml ./configs/examples/structures/score
    partidesign relax configs/examples/protein.yaml --out inference/demo --platform CPU
    partidesign validate inference/demo --config configs/benchmarks/protdbench/validate.yaml --plan

Precedence is explicit CLI flags, then typed command overrides, then the selected
command YAML, then the common scorer/sequence/criteria definitions. A task-local
protocol or benchmark profile wins over a generic default. Validation source
labels and counts are independent from reward sequence settings: reward defaults
to the FM design sequence, auto selects SolubleMPNN for protein/cyclic
systems and stays on design for nucleic systems, while validation profiles
explicitly select gt, SolubleMPNN, NA-MPNN, LigandMPNN, or CyclicMPNN.

The criteria file contains executable methods, reductions, and rule thresholds.
[Validation](validation.md) records source authorities, units, and protocols; rules are conjunctive within a prediction
and the named reduction combines predictions. Missing required metrics produce an
unavailable status. Benchmark sample budgets and temperatures remain in each
task/profile and are recorded in configs/benchmarks/manifest.yaml.
