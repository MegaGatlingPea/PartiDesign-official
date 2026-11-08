# Boltz-2 prediction backend

Boltz-2 is available through `partidesign validate`, sequence
[`screen`](../../docs/screen.md), and the
[ligand pipeline](../../docs/pipeline.md#ligand-pipelines).

```bash
partidesign download boltz2
partidesign check --model boltz2 --msa off

partidesign validate inference/ligand_run --method boltz2 --source gt:1 \
  --msa-source colabfold --msa auto --limit 1 --plan
partidesign screen configs/examples/screen.yaml --task ligand_screen \
  --out screen_results/ligand --plan
partidesign pipeline configs/examples/ligand.yaml --task ligand_fixture \
  --preset ligand --run-name ligand_pipeline --plan
```

Remove `--plan` to predict after preparing the selected models. The optional
CUDA12 route requires Numba **0.68.1** and llvmlite **0.50.0** in the existing
environment; installer defaults do not add them. Use the bundled source:
installing upstream `boltz` would apply conflicting dependency pins.

Structure-based `validate` supports a fixed standard protein context and one
free designed small molecule. It predicts a new complex from the saved ligand
chemistry, retaining the original structure for comparison. It rejects
modified/cyclic polymers, unsupported covalent links, attached or
ambiguous/multiple designed ligands, and conflicting context/design MSA policies
for identical protein sequences. See the
[validation guide](../../docs/validate.md#boltz-2-proteinligand-validation).

Sequence screening accepts protein, peptide, RNA, DNA and ligand candidates;
peptides use protein chemistry. The ligand pipeline defaults to one round with
Boltz-2, ranks iteration candidates by design-only pLDDT, and performs native
affinity evaluation at final. Affinity requires one eligible designed ligand
with protein context; context-free ligand pipelines retain structure validation
and report affinity as not applicable.

Fixed protein contexts use the shared MSA/template cache; designed chains stay
query-only, RNA/DNA remain query-only in Boltz-2, and ligands have no MSA.
Boltz-2 does not launch a separate online search. Supported pocket annotations
condition prediction without adding a pocket postfilter. See
[MSA and templates](../../docs/msa.md) and the workflow guides for input controls.

Structure prediction defaults to one sample, three recycles and 200 steps.
Native affinity defaults to five samples and 200 steps; set
`boltz2.affinity: false` in a validation profile for structure-only prediction.
Screen and pipeline select the structure representative by native
`confidence_score` before their later ranking or affinity stage. Native affinity
and confidence fields are preserved, with no Boltz-specific acceptance
thresholds enabled by default. Existing Vina behavior and ordinary validation
defaults are unchanged.

The [upstream source](https://github.com/jwohlwend/boltz) is Boltz 2.2.1 at
`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`. The original
[MIT license](source/LICENSE) and [provenance](UPSTREAM.json) are retained.
Model downloads pin the official HF revision
`6fdef46d763fee7fbb83ca5501ccceff43b85607`; see the
[download guide](../../docs/download.md#boltz-2-assets) for the asset layout.
