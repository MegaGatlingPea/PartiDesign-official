# Model downloads

From the repository root, open the interactive model-selection dialog:

```bash
partidesign download
```

Use Up/Down to move, Space to toggle, Enter to continue, or Esc to cancel.
`a` selects all available entries; `n` clears the selection. No model is selected
by default. An empty selection keeps the dialog open. Ctrl-C interrupts with exit
130. Cancel/EOF exit without downloading or creating the selected model root.

The dialog shows ready/download/manual states and resource origin separately.
Ready resources are kept; missing resources with configured URLs can be fetched.
Manual rows cannot be selected when required download URLs are absent. Git-supplied
files need to be copied/restored from the repository. This menu does not make a
missing weight URL available; PartiDesign weights still need a local authorized copy.

Weights go under `ROOT/ckpts` and `ROOT/community_models`, where ROOT is the checkout
unless `--model-dir PATH` or `PARTIDESIGN_MODEL_DIR` overrides it. Static source and
`configs/resources` stay with the installed package.

## Explicit and scripted commands

```bash
partidesign download --list
partidesign download proteinmpnn solublempnn
partidesign download esmfold2-fast --model-dir /path/to/model-root
partidesign download --all
partidesign check --model proteinmpnn --load
```

Explicit targets, `--all` and `--list` never open the dialog. A no-target invocation
with redirected input/output or `TERM=dumb` fails promptly and suggests these
commands. `NO_COLOR` removes color; in a real terminal the selection dialog still
works in monochrome. `--list` is read-only; local presence does not prove a model
strict-loads, which is what `check --load` verifies.

`--force` stages and validates replacement downloads before publishing them. An
interactive forced download disables entries that cannot be refreshed from a URL.
aria2c uses up to eight connections per file; if unavailable or failing, fetching
falls back to built-in HTTP. Unknown transfer sizes remain indeterminate until
completion, and publication failures stay failures in the final summary.

## Chemistry and ESMFold2 sources

`chemistry` is the single native `configs/resources/chemistry/ccd.json.gz`
shipped with Git and the wheel. It stays with the code when `--model-dir` moves
weights. It is not the ESMFold2 external CCD bundle.

ESMFold2 fast/standard model weights, configs, ESMC-6B shards/tokenizer and
`ccd.pkl` use pinned official Biohub Hugging Face sources. Their catalog reports
`download URL`. The shared `components.cif` is fetched from the official wwPDB
monomer source as gzip and decoded during staging. The downloader then builds the
shared `ccd_bonds.sqlite3` cache from that CIF with the vendored CCD builder and
validates it before publication. Fast and standard reuse the same common and ESMC
files under the selected model root.

A fresh ESMFold2 target is selectable because both model assets and chemistry
provisioning are automated. Existing complete installs are reused without fetches;
a missing or invalid derived SQLite cache is rebuilt from the existing CIF, while a
missing or refreshed CIF is fetched and causes the cache to be rebuilt. Files are
staged and validated before publication; publication failures roll back the previous
files. `--list` reports source origin separately from completeness; `check --load`
verifies installed assets.
