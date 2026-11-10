# `partidesign download`

[Command index](commands.md)

List available model resources and prepare the weights needed by a workflow.
The command manages model assets; the [main installation guide](../README.md#installation)
covers the Python environment and bundled backend code.

## Usage

Run from the repository root with the `partidesign` environment active:

```bash
partidesign download --list
partidesign download esmfold2-fast
partidesign check --model esmfold2-fast
```

`--list` inspects the inventory without network access. Name one or more targets
to download them. In an interactive terminal, `partidesign download` opens a
model selector. Noninteractive jobs must specify targets, `--list`, or `--all`.

The selector shows a short use, size and status for each model. A **✓** marks
local files; the checkbox selects a download. Missing or damaged files are marked
separately. Known project checkpoints are checked before reuse and replacement;
a failed download keeps the existing file. `~` sizes are reference disk footprints,
including shared files, rather than an additional-download budget.

## Generation weights and bundled assets

PartiDesign and PartiScore checkpoints are distributed from the
[project Hugging Face model repository](https://huggingface.co/MegaGatlingPea/PartiDesign).
The downloader pins both files to immutable revision `a0ccb3b7b2e775d1364314f8bf35644ba0719e71`;
neither checkpoint is supplied by a new source clone.

| Checkpoint | Purpose |
| --- | --- |
| [`partidesign.ckpt`](https://huggingface.co/MegaGatlingPea/PartiDesign/resolve/a0ccb3b7b2e775d1364314f8bf35644ba0719e71/ckpts/partidesign.ckpt?download=true) | Generation bundle: model, autoencoder and experiment configuration |
| [`partiscore.ckpt`](https://huggingface.co/MegaGatlingPea/PartiDesign/resolve/a0ccb3b7b2e775d1364314f8bf35644ba0719e71/ckpts/partiscore.ckpt?download=true) | Molecular interface scoring |

With the project environment active, download both unchanged `.ckpt` files:

```bash
partidesign download partidesign partiscore
partidesign check --model partidesign --model partiscore
```

The resulting model root contains:

```text
ckpts/
├── partidesign.ckpt
└── partiscore.ckpt
```

The generation bundle is 4,909,825,762 bytes; PartiScore is 25,350,931 bytes.
Keep the existing binary formats and filenames intact. These project-owned
weights use Apache-2.0; third-party backend weights retain their upstream terms.
CyclicMPNN weights come with the source checkout. PartiDesign's native component
dictionary is included in the package and needs no separate download. Model
weights are not included in wheels or source archives.

For a different location, pass `--model-dir` to download/check and runtime
commands, or select the checkpoint with `design --ckpt FILE`. A compatible
generation release includes the model, autoencoder and experiment configuration;
a flow-matching-only training checkpoint is insufficient.

ESMFold2-fast and ESMFold2-standard share the ESM-C safetensors under
`community_models/esmfold2/checkpoint/esmc-6b/`; they are stored once.

## Optional evaluation models

Basic generation does not require a separate prediction or sequence-redesign
model. Download only the backends used by your chosen validation workflow:

| Workflow | Download command |
| --- | --- |
| Ordinary protein/nucleic-acid pipeline | `partidesign download esmfold2-fast esmfold2-standard` plus the selected MPNN |
| Antibody pipeline | `partidesign download esmfold2-fast mol1-constraint proteinmpnn` |
| Explicit PPI pipeline | `partidesign download esmfold2-fast af2` plus the selected MPNN |
| Protein sequence redesign | `partidesign download proteinmpnn solublempnn ligandmpnn` |
| Nucleic acid sequence redesign | `partidesign download nampnn` |
| Fast ESMFold2 prediction | `partidesign download esmfold2-fast` |
| Standard ESMFold2 evaluation | `partidesign download esmfold2-standard` |
| AlphaFold2 initial-guess evaluation | `partidesign download af2` |
| Standard Mol-1 evaluation | `partidesign download mol1` |
| Hotspot-guided Mol-1 evaluation or screening | `partidesign download mol1-constraint` |
| Boltz-2 validation, sequence screening or ligand pipeline | `partidesign download boltz2` |
| AlphaFold3 / AF3-LA evaluation | `partidesign download af3 af3-la` |

The target name for NA-MPNN is `nampnn`. CyclicMPNN is already included. Each
upstream model's access and usage terms apply to its source and parameters.

Check only the backends you installed:

```bash
partidesign check --model proteinmpnn --model ligandmpnn
partidesign check --model af3 --msa off
partidesign check --model mol1 --msa off
partidesign check --model mol1-constraint --msa off
partidesign check --model boltz2 --msa off
```

These AF3, Mol-1 and Boltz-2 commands check weights and chemistry without requiring MSA databases.
ColabFold is the default protein MSA/template source and needs no local AF3 search databases.
For `source: af3`, prepare local databases or a cache separately. Downloads
do not install the full MSA search databases. See [checks](check.md) and
[validation](validate.md) for the next steps.

### Mol-1 assets

Standard Mol-1 and Mol-1-constraint share source and common data, with separate weights:

```text
community_models/mol1/
  source/                    # bundled backend code
  checkpoint/mol-1-base.pt    # standard model parameters
  checkpoint/mol-1-constraint.pt  # hotspot-guided model parameters
  common/                    # downloaded chemistry and reference metadata
```

`partidesign download mol1` or `partidesign download mol1-constraint` prepares
the selected checkpoint and shared common resources; backend code is bundled.
Downloading both targets together, including with `--force`, prepares shared
data once. The constraint checkpoint is **2,311,106,509 bytes** (about 2.15 GiB),
plus shared common data. It comes from the
[official Mol-1 release](https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-constraint.pt).
Bundled source is pinned to
[`9e85ad16b3d7031dd92c34d99f895ee65547038b`](https://github.com/molence/Mol-1/tree/9e85ad16b3d7031dd92c34d99f895ee65547038b).
These third-party weights are not hosted in PartiDesign's Hugging Face repository.

With an external model root,
`checkpoint/` and `common/` go under that root while source stays with the
package. Mol-1 reuses the AF3 context MSA/template cache and search resources;
it does not need a separate Mol-1 search database. The environments provide
`colabfold-binaries` for HHsearch; no full ColabFold prediction installation is needed.
See [MSA and templates](msa.md) for pre-search commands and cache policy. Check upstream parameter
terms separately from the source-code license.

### Boltz-2 assets

`partidesign download boltz2` prepares optional structure/affinity checkpoints
and safely extracts their molecule library:

```text
community_models/boltz/
  source/                    # bundled source; stays with the code installation
  checkpoint/boltz2_conf.ckpt
  checkpoint/boltz2_aff.ckpt
  common/mols.tar
  common/mols/                # extracted molecule library
```

The three downloads total **6,204,362,719 bytes**; keeping the extracted library
brings their installed footprint to about **7.47 GiB**. Downloads use the official
[Boltz-2 Hugging Face repository](https://huggingface.co/boltz-community/boltz-2)
at immutable revision `6fdef46d763fee7fbb83ca5501ccceff43b85607`, with registered
sizes and SHA256 checksums. The release uses MIT terms. Incomplete extraction
is not reported as ready. `check` can verify the extracted files against the
pinned archive; replaced incomplete ordinary directories are preserved as hidden
backups.

The single `boltz2` target supplies structure and affinity assets for
[`validate --method boltz2`](validate.md#boltz-2-proteinligand-validation),
[sequence screening](screen.md) and the [ligand pipeline](pipeline.md#ligand-pipelines).
Vina and the ordinary polymer defaults remain unchanged. Boltz-2 shares the existing ColabFold
MSA/template cache and needs no separate search database. The current optional
route requires Numba 0.68.1 and llvmlite 0.50.0 in the existing environment;
installer defaults have not been expanded for this integration. See the
[validation guide](validate.md#boltz-2-proteinligand-validation) for input limits
and native affinity metrics.

## Use a separate model directory

The model-root priority is `--model-dir`, then `PARTIDESIGN_MODEL_DIR`, then the
detected PartiDesign checkout:

```bash
partidesign download --list --model-dir /path/to/partidesign-models
partidesign download esmfold2-fast --model-dir /path/to/partidesign-models
partidesign check --model esmfold2-fast --model-dir /path/to/partidesign-models
```

For repeated use, set the environment variable instead:

```bash
export PARTIDESIGN_MODEL_DIR=/path/to/partidesign-models
partidesign download --list
```

This root contains `ckpts/` and `community_models/`. Download the two project
checkpoints into it with `partidesign download partidesign partiscore --model-dir PATH`.
Copy any needed Git-supplied CyclicMPNN files into their matching locations;
missing Git-supplied resources still cannot be fetched by the downloader.
Bundled code and static chemistry resources stay with the installed package.
Task inputs, explicit checkpoint paths, and outputs still resolve from the
invocation directory. Selecting a model root does not change that directory.

## Options, output, and reuse

| Option | Behavior |
| --- | --- |
| `TARGET ...` | Prepare the named resources; no targets opens the interactive selector in a terminal. |
| `--list` | Print asset status and sources without downloading. Can be restricted to named targets. |
| `--model-dir PATH` | Select the model root. |
| `--all` | Prepare all models; cannot be combined with explicit targets. |
| `--force` | Refresh downloadable assets, replacing them only after complete preparation. Cannot be combined with `--list`. |

Files are placed at the paths shown by the inventory. Repeating a download
reuses a complete installation and prepares missing assets. `--force` cannot
refresh Git-supplied files or supply a resource without a registered source.
The command returns a nonzero status if any selected target fails or remains
unavailable; other independent targets can still complete.

## Troubleshooting

| Problem | Next step |
| --- | --- |
| A target is unavailable | Inspect `download --list`. Download `partidesign`/`partiscore` explicitly from the pinned HF source, or restore remaining Git-supplied resources such as CyclicMPNN from a complete checkout. |
| A download fails | Check the reported source and network access, then repeat the explicit target command. |
| Files appear to be missing after using another disk | Check `--model-dir` and `PARTIDESIGN_MODEL_DIR`; copy needed Git-supplied weights into the selected root. |
| AF3 reports missing MSA resources | Use `check --model af3 --msa off` for the weights-and-chemistry check, or prepare the intended MSA resources. |

[Command index](commands.md) · [Check assets](check.md) · [Generate designs](design.md)
