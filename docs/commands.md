# Command reference

Run commands from the repository root after activating your environment.
The default uv installation uses `source .venv/bin/activate`; the explicit
Conda route uses `conda activate partidesign` (CUDA 12) or
`conda activate partidesign_cu13` (CUDA 13). See [installation](environments.md)
for separate CUDA profiles and environment paths. A [task YAML](task_schema.md)
describes the molecules and conditions; command options select inputs, model
assets, and execution settings.

```bash
partidesign --help
partidesign <command> --help
```

| Command | Use it to |
| --- | --- |
| [`partidesign download`](download.md) | Prepare generation and optional model assets |
| [`partidesign check`](check.md) | Check tasks, model assets, and the installation |
| [`partidesign design`](design.md) | Generate molecular structures and identities |
| [`partidesign msa`](msa.md) | Prepare shared context MSA and templates for selected tasks |
| [`partidesign pipeline`](pipeline.md) | Run design, refinement, and validation in one command |
| [`partidesign score`](score.md) | Score interfaces in saved structures with PartiScore |
| [`partidesign relax`](relax.md) | Refine existing structures with OpenMM |
| [`partidesign validate`](validate.md) | Evaluate saved designs with selected sequences, predictors, and criteria |
| [`partidesign postfilter`](postfilter.md) | Relax and screen passing protein and peptide complexes |
| [`partidesign cluster`](cluster.md) | Group passing designs by backbone similarity |

For a first run, follow the [generation guide](design.md#a-first-run). After generation,
use the [validation guide](validate.md) to choose the method, sequence budget, and
passing criteria before running an evaluation. The [performance guide](performance.md)
explains the paper's computational budgets and evaluation protocols.

## Terminal appearance

Interactive help uses a compact rounded header, yellow text accents, and
bordered tables with centered cells. Section titles remain left aligned;
emoji retain the terminal's native appearance.
The four `check` sections use the same title size: 🖥️ Machine overview,
🧪 PartiDesign models, 🧬 Sequence design, and 🔬 Structure prediction.
Kitty can also display section titles at about 1.2 times the normal size.
Other terminals keep the bold title; the compact header keeps its size.
`partidesign --help` always uses the standing MilkyFrog image. Other human-facing
commands select one of the six bundled images once per process. Image selection
is independent of model random seeds and never downloads an image at runtime.

The bundled PNG and JPEG images appear in terminals identified as supporting Kitty graphics
or the iTerm2 image protocol, including Ghostty and WezTerm. Across SSH, that
terminal identity must reach the remote shell; otherwise the header stays text
only. Narrow or short terminals also use the text header. There is no pixel-art
substitute. Redirected output, `NO_COLOR`, and `TERM=dumb` retain plain output.
JSON plans and saved scientific outputs keep their existing formats.

## Pipeline

`partidesign pipeline TASKS.yaml` runs design, refinement, and validation in one
command. It generates once, then refines sequences and structures through MPNN,
refolding, selection, and partial diffusion. All presets default to ESMFold2-fast
with relaxed pipeline screening during iteration; OpenMM is optional.

By default, final validation for `protein`, `nuc`, and `antibody`, including
profiles selected by `auto`, uses only ESMFold2-standard with five samples per
selected candidate.
Standard uses shared fixed protein-context MSA without templates; designed chains
remain query-only. Without fixed protein context, standard also stays query-only.

Native antibody tasks automatically choose the `antibody` preset. Its five
standard predictions are ranked by 0.8 ipTM + 0.2 pTM before final criteria and
mandatory antigen-contact checks. See [antibody pipelines](pipeline.md#antibody-pipelines)
for de novo and annotated-complex commands.

The explicit `ppi` preset uses only AF2-IG models `[1, 2]`, with `samples: 1` and
`mode: both`: one matched complex/monomer pair per model, four structures total.
Both selected model pairs must pass the AF2 internal filter. The demo uses this
same PPI route.

Use `--method` or `iteration.refold` to explicitly override iteration, and
`final.method` or `final.methods` to select optional final backends such as AF3 or
Mol-1. With an explicit multi-method final, every configured method runs and any
passing method accepts the candidate. See the [complete pipeline guide](pipeline.md)
for presets, round/task overrides, the insulin receptor demo, outputs, and resume.
For single-reference sampling with `partidesign design`, see
[partial diffusion](partial.md).

## MSA and templates

`partidesign msa TASKS.yaml --task NAME` prepares fixed-context alignments and
protein templates without generating designs or loading a prediction model.
Protein searches default to ColabFold. `pipeline` and `validate` use the same
cache and accept `--msa-source colabfold|af3`. See [MSA and templates](msa.md).
