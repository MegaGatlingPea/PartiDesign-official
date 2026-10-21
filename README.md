# PartiDesign inference

Independent inference distribution for multimodal structure design, antibody
CDR design, PartiScore, refinement, and validation. The CLI is `partidesign`;
`python -m src.design`, `src.validate`, and `src.scoring` remain available.

## Install from a clone

Run these commands from this repository root on Linux with NVIDIA CUDA support:

```bash
conda env create -f partidesign.yaml
conda activate partidesign
pip install -e .
partidesign --help
partidesign check --help
partidesign download  # Choose weights with arrow keys and Space.
```

The environment file provisions the pinned runtime and maintained backend code;
the explicit editable install above registers this checkout as `partidesign`.
After editing Python/configuration files, restart the
active CLI/worker processes to use the changes. Reinstall when packaging,
dependencies, entry points, or native extensions change.

Weights belong under `ROOT/ckpts` and `ROOT/community_models`, where `ROOT` is
this checkout by default. Use `--model-dir PATH` or `PARTIDESIGN_MODEL_DIR` for a
separate asset root. PartiScore and the curated CyclicMPNN assets come with Git.
Model assets are supplied with the checkout or placed under the documented
`ROOT/ckpts` and `ROOT/community_models` paths. The native six-branch
`ckpts/partidesign.ckpt` has no public URL yet; copy an existing authorized
bundle into that path. The SDK uses the single native CCD shipped under
`configs/resources/chemistry/ccd.json.gz`. Static runtime data, including
scaffolds, ligand atom-count distributions, the autoencoder configuration and
OpenMM tables, all live under `configs/resources`; `src` contains the loaders.

## Interactive task notebook

Open [partidesign.ipynb](partidesign.ipynb) and run its setup cell. The native
`peptide/peptide_bind_protein_betahairpin` preset opens in the fixed **Nord Light**
workbench with six steps: **1 Input & design**, **2 Motif**, **3 Hotspot**,
**4 Advanced conditions**, **5 Save**, and **6 Runtime**. Mol* stays mounted in the same frame,
including an empty Input or a **De novo** task. The first five steps retain
source and applied-condition previews. Inference, validation, clustering, and
opening generated or refolded results are all in **6 Runtime**. Hover or focus
the question-circle icons for help; click to pin the help and press Esc to close it.

Choose a complete task preset at the start of **1 Input & design**. The library
loads **46 non-antibody tasks** under `configs/examples/`. Selecting a preset automatically loads and
replaces the current task's input, design modality, lengths, Motif, Hotspots,
and advanced conditions together, updates Mol*, and returns to the first step.
Protein, Peptide, Nucleic acids, Ligand, Motif, and Additional protein-ligand
presets retain their complete native task. Antibody examples are excluded from
this notebook. Joint, legacy Motif, and context-ligand tasks have visual native forms for their
existing fields. Source paths and chain IDs are read-only in those forms;
use **Full task YAML** to add or remove entries, and apply YAML edits before
checking or saving.

Assign source chains to **Context** or **Reference**. Chain filters apply on
Enter or when leaving the field; original files stay in `raw/`. Full Reference
geometry can identify Context Hotspots independently of Motif selection:
preview contacts at the default **6 Å** cutoff, then Merge or Replace applied
Hotspots. Reference-only contact selection supplies no model coordinates.

Motif **Selection**, **Contig**, and **Atoms** areas remain visible when Motif is off.
A compact residue list shows each residue's atom restriction; only the selected
residue's atom editor is open. The list and editor sit side by side when space
allows, with a chain selector for tasks containing several Contigs.
New Motif residues keep all atoms observed in their source CIF; clearing a
protein atom restriction keeps all observed atoms. Protein options use the
actual residue inventory. RNA/DNA offer all observed atoms or the trained
base/non-backbone group. Existing legacy nucleotide subsets remain visible
until explicitly replaced by a supported choice. Check Contig lengths and
applied selections before saving. Exported input keeps Context and complete
source chains needed by active conditions, omitting unused References.

Mol* has **Chain**, **Residue**, and **Atom** selection scopes in its upper-right
corner. Chain selects the whole chain, Residue selects a whole residue, and Atom
selects an atom or a bond's endpoint atoms. Selections follow the active page;
RNA/DNA Motif atom restrictions still use the supported groups above. The default
beta-hairpin task demonstrates fine beta-sheet segments, adjacent pairing,
fixed cysteines, and an explicit Cys2–Cys14 SG–SG bond.

In **6 Runtime**, **Prepare inference**, **Prepare validation**, and **Prepare
clustering** only create a command and any required input/configuration
snapshots under `.cache/`. **Execute** and **Modify command** both default to
**off**. The command is read-only and selectable for copying and manual use.
Enable **Execute**, then click **Execute command** to start an owned background
tmux job. **Modify command** enables editing of the actual argument list;
preparing, editing, and toggling controls never launch a job automatically.
Commands do not use an implicit shell; shell syntax requires an explicit
`bash -lc '...'` invocation. Use **Background jobs & logs** to refresh logs,
stop the selected owned job, or reconnect after a notebook/SSH disconnection.

Notebook compilation defaults to **off**; larger jobs can optionally enable
**Compile model**. The default output root is `configs/customs/inference`, and
native generation adds the task leaf once: `<output root>/<task name>/`.
Prepared task YAML, input structures, and custom scaffold YAML/CIF dependencies
are copied into a unique snapshot so later authoring edits do not change the
prepared inputs.

For validation, select an existing task or run directory containing actual
results. **Original · 1** is always retained. Ordinary and fixed-interface MPNN
redesign have independent counts and temperatures; FI also exposes its cutoff.
The default method is **AF3**, with native **Context MSA = Auto / reuse cache**,
**Use templates** on, and native cache defaults. Empty notebook criteria
explicitly accept completed validation into the unfiltered success cohort;
failed or incomplete validation remains unsuccessful. Enter optional named
criteria to apply scientific thresholds. Clustering uses original successful
design backbones, keeps methods/cohorts separate, defaults to TM threshold
**0.5**, and allows at most **16** concurrent directories.

In **Results & preview**, choose the result directory, refresh raw samples,
select a method, and choose a successful refold by its source/sequence and
prediction slot. Raw samples and refolds share Mol*. Validated refolds retain
native chain coloring and confidence values; those values are not interpreted
as task-condition categories. Raw native CIFs retain task-condition colors.

Custom files default to ignored `configs/customs/raw/`, `input/`, and `tasks/`;
working source and condition previews stay under `.cache/`. See
[task conditions](docs/task_schema.md) and
[outputs and colors](docs/outputs.md#condition-colors-in-cif).

## Use the CLI

```bash
partidesign check configs/examples/protein.yaml --task protein_fixture
partidesign design configs/examples/protein.yaml --task protein_fixture --out inference/demo
partidesign score inference/demo/protein_fixture
partidesign validate inference/demo --method af2_ig --plan
partidesign cluster inference/demo --plan
```

Task files contain named tasks. Keep normal and antibody tasks in separate
submissions. Automatic routing selects one of `prot`, `pep`, `nuc`, `ligand`,
`omni`, or `antibody` for the complete selected batch. Design produces its own
sequence by default. Reward/search are opt-in. Protein/cyclic fold validation
defaults to one original sequence plus one SolubleMPNN sequence; nucleic
sequence validation defaults to one original plus one NA-MPNN sequence.
Explicit benchmark profiles keep their documented sequence budgets/backends.

Start with the [inference guide](docs/README.md),
[configuration](docs/configuration.md),
[reward/search](docs/reward.md),
[validation](docs/validation.md), and
[successful-backbone clustering](docs/clustering.md).
Examples live under `configs/examples/`; benchmark tasks/protocols under `configs/benchmarks/`.
The protocol manifest records all retained, merged, and excluded source suites.
RNA/DNA bind-protein now use the [NACraft NA-12 sources and task mapping](configs/benchmarks/manifest.yaml).
Protein-bind-RNA/DNA now uses eight DISCO-derived fixed-native-NA cases (124 structures per target); see the [benchmark protocol](docs/benchmarks.md#protein-design-against-disco-derived-rnadna-targets).
This repository contains inference code only; training remains in PartiDesign.

## Source and acceptance

Third-party code licenses remain with their source directories; scaffold and
resource licenses are kept beside the data under `configs/resources`.
Model weights have their own upstream terms. See the backend documentation
before obtaining external assets.

[Acceptance evidence](docs/acceptance.md) records the tested
installation, source boundary, defaults, chemistry parity, backend matrix, and
performance. Work in progress is recorded explicitly rather than represented
as a completed release.
