# LigandMPNN asset inventory

Status: deployed for PartiDesign's offline reward pipeline. `run.py`/`score.py`, core imports, every checkpoint, and all five sequence-model families load. A real A800 LigandMPNN run on the repository's AME/BCA case produced exactly one validated 180-residue design sequence.

## Layout and provenance

- Official upstream: https://github.com/dauparas/LigandMPNN (MIT; see `LICENSE`).
- Copied from the tracked LigandMPNN snapshot in NVIDIA-BioNeMo/Proteina-Complexa checkout `916eaaedce5b07c205efb6ef32370c01d366591e`; the vendored model files trace to its `abf53e953640d574ccfd687967513676f6ea7197` release commit.
- The vendored snapshot contains no nested upstream Git metadata, so no unverified LigandMPNN commit is claimed. Runtime acceptance uses the local source/checkpoint inventory and an executable offline case.
- `source/` contains unchanged `run.py`, `score.py`, model/data/side-chain utilities, and LigandMPNN's own OpenFold subset. `checkpoint/` contains all runtime weights.
- There are no symlinks or absolute references to the Proteina-Complexa checkout. `UPSTREAM_README.md` is retained for the original CLI examples.

## Adapter entry

- Design CLI: `community_models/ligandmpnn/source/run.py`
- Score CLI: `community_models/ligandmpnn/source/score.py`
- Checkpoint root: `community_models/ligandmpnn/checkpoint`.

The adapter must explicitly pass the checkpoint flag for the selected `--model_type`; the untouched upstream defaults still point to `./model_params`. For side-chain packing it must also pass `--checkpoint_path_sc` to `ligandmpnn_sc_v_32_002_16.pt`. PipeTools should own `--pdb_path`, `--out_folder`, and the one-chain `--chains_to_design` value.

The supported PartiDesign runtime is the repository's pinned Python 3.12.6 / PyTorch 2.9.1+cu128 environment. Runtime subprocesses receive local assets only and an IP-socket-denying `sitecustomize`.

## Available behavior parameters

- Model family: `--model_type` supports `protein_mpnn`, `ligand_mpnn`, `soluble_mpnn`, `per_residue_label_membrane_mpnn`, and `global_label_membrane_mpnn`.
- Sampling: `--seed`, `--temperature`, `--batch_size`, `--number_of_batches`.
- Single/multi input: `--pdb_path`, `--pdb_path_multi`, `--parse_these_chains_only`, `--chains_to_design`, `--force_hetatm`, `--parse_atoms_with_zero_occupancy`.
- Design masks: `--fixed_residues`, `--fixed_residues_multi`, `--redesigned_residues`, `--redesigned_residues_multi`.
- Amino-acid constraints: `--omit_AA`, `--omit_AA_per_residue`, `--omit_AA_per_residue_multi`, `--bias_AA`, `--bias_AA_per_residue`, `--bias_AA_per_residue_multi`.
- Symmetry: `--homo_oligomer`, `--symmetry_residues`, `--symmetry_weights`.
- Atom/side-chain context: `--ligand_mpnn_use_atom_context`, `--ligand_mpnn_cutoff_for_score`, `--ligand_mpnn_use_side_chain_context`.
- Membrane labels: `--global_transmembrane_label`, `--transmembrane_buried`, `--transmembrane_interface`.
- Side-chain packing: `--pack_side_chains`, `--pack_with_ligand_context`, `--repack_everything`, `--number_of_packs_per_design`, `--sc_num_samples`, `--sc_num_denoising_steps`.
- Output behavior: `--save_stats`, `--file_ending`, `--packed_suffix`, `--fasta_seq_separation`, `--verbose`, `--zero_indexed`.

## Checkpoints

| Behavior | Files | Size per file (bytes) |
|---|---|---:|
| ligand-aware | `ligandmpnn_v_32_005_25.pt`, `_010_25.pt`, `_020_25.pt`, `_030_25.pt` | 10,541,943 |
| ProteinMPNN | `proteinmpnn_v_48_002.pt`, `_010.pt`, `_020.pt`, `_030.pt` | 6,681,301 |
| soluble MPNN | `solublempnn_v_48_002.pt`, `_010.pt`, `_020.pt`, `_030.pt` | 6,681,301 |
| per-residue membrane | `per_residue_label_membrane_mpnn_v_48_020.pt` | 6,751,499 |
| global membrane | `global_label_membrane_mpnn_v_48_020.pt` | 6,751,499 |
| side-chain packer | `ligandmpnn_sc_v_32_002_16.pt` | 14,359,379 |

Total checkpoint bytes: 123,480,557. Verification is executable and does not retain a smoke-test log.
