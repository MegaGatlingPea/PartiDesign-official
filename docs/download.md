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

## Generation weights and bundled assets

PartiScore weights, CyclicMPNN weights, and native chemistry resources come with
the clone. Download the PartiDesign generation bundle from
**[Google Drive](https://drive.google.com/file/d/19j8aaysxeQlT0RI5URrQm3A9uWdqSE5e/view?usp=sharing)**
(4,909,825,762 bytes, about 4.91 GB). In the browser, choose **Download** and
save it as `ckpts/partidesign.ckpt`, beside the Git-supplied `partiscore.ckpt`.

Alternatively, download from a terminal at the repository root:

```bash
python -m pip install gdown
gdown 'https://drive.google.com/file/d/19j8aaysxeQlT0RI5URrQm3A9uWdqSE5e/view?usp=sharing' \
  -O ckpts/partidesign.ckpt
```

The resulting directory should contain:

```text
ckpts/
├── partidesign.ckpt
└── partiscore.ckpt
```

Keep the `.ckpt` file intact rather than extracting it. The `partidesign download`
CLI has no automatic source registered for this Drive release, so neither
`download partidesign` nor `download --all` retrieves it yet. Use the link or
`gdown` above. If Drive reports an access or download-quota error, use the browser
link and retry after access is available; an HTML page is not a checkpoint.

For a different location, select the file with `design --ckpt FILE` or use the
[model directory](#use-a-separate-model-directory). A compatible release bundle
includes the model, autoencoder, and experiment configuration; a flow-matching-only
training checkpoint is insufficient.

## Optional evaluation models

Basic generation does not require a separate prediction or sequence-redesign
model. Download only the backends used by your chosen validation workflow:

| Workflow | Download command |
| --- | --- |
| Protein sequence redesign | `partidesign download proteinmpnn solublempnn ligandmpnn` |
| Nucleic acid sequence redesign | `partidesign download nampnn` |
| Fast ESMFold2 prediction | `partidesign download esmfold2-fast` |
| Standard ESMFold2 evaluation | `partidesign download esmfold2-standard` |
| AlphaFold2 initial-guess evaluation | `partidesign download af2` |
| AlphaFold3 / AF3-LA evaluation | `partidesign download af3 af3-la` |

The target name for NA-MPNN is `nampnn`. CyclicMPNN is already included. Each
upstream model's access and usage terms apply to its source and parameters.

Check only the backends you installed:

```bash
partidesign check --model proteinmpnn --model ligandmpnn
partidesign check --model af3 --msa off
```

The AF3 command checks weights and chemistry without requiring MSA databases.
Prepare databases or a cache separately for MSA-enabled validation. Downloads
do not install the full MSA search databases. See [checks](check.md) and
[validation](validate.md) for the next steps.

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

This root contains `ckpts/` and `community_models/`. Copy the generation bundle
and any needed Git-supplied PartiScore/CyclicMPNN weights into their matching
locations there; the downloader cannot fetch missing Git-supplied files.
Bundled code and static chemistry resources stay with the installed package.
Task inputs, explicit checkpoint paths, and outputs still resolve from the
invocation directory. Selecting a model root does not change that directory.

## Options, output, and reuse

| Option | Behavior |
| --- | --- |
| `TARGET ...` | Prepare the named resources; no targets opens the interactive selector in a terminal. |
| `--list` | Print asset status and sources without downloading. Can be restricted to named targets. |
| `--model-dir PATH` | Select the model root. |
| `--all` | Attempt every registered target; cannot be combined with explicit targets. |
| `--force` | Refresh downloadable assets, replacing them only after complete preparation. Cannot be combined with `--list`. |

Files are placed at the paths shown by the inventory. Repeating a download
reuses a complete installation and prepares missing assets. `--force` cannot
refresh Git-supplied files or supply a resource without a registered source.
The command returns a nonzero status if any selected target fails or remains
unavailable; other independent targets can still complete.

## Troubleshooting

| Problem | Next step |
| --- | --- |
| A target is unavailable | Inspect `download --list`. Obtain the generation bundle from the [Drive link](#generation-weights-and-bundled-assets), or restore Git-supplied resources from a complete checkout. |
| A download fails | Check the reported source and network access, then repeat the explicit target command. |
| Files appear to be missing after using another disk | Check `--model-dir` and `PARTIDESIGN_MODEL_DIR`; copy needed Git-supplied weights into the selected root. |
| AF3 reports missing MSA resources | Use `check --model af3 --msa off` for the weights-and-chemistry check, or prepare the intended MSA resources. |

[Command index](commands.md) · [Check assets](check.md) · [Generate designs](design.md)
