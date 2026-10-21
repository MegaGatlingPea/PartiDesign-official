# ProteinMPNN asset inventory

Status: deployed for PartiDesign's offline reward pipeline. Source import, every checkpoint load, and representative vanilla/CA/soluble loads pass. A real A800 ProteinMPNN one-sequence design using the soluble checkpoint also passed.

## Layout and provenance

- Official upstream: https://github.com/dauparas/ProteinMPNN (MIT; see `LICENSE`).
- Copied from the tracked ProteinMPNN snapshot in NVIDIA-BioNeMo/Proteina-Complexa checkout `916eaaedce5b07c205efb6ef32370c01d366591e`; the vendored model files trace to its `abf53e953640d574ccfd687967513676f6ea7197` release commit.
- The vendored snapshot contains no nested upstream Git metadata, so no unverified ProteinMPNN commit is claimed. Runtime acceptance uses the local source/checkpoint inventory and an executable offline case.
- `source/` is unchanged inference code. `checkpoint/` contains all runtime weights. There are no symlinks or absolute references to the Proteina-Complexa checkout.
- `checkpoint/soluble/` is populated from the official soluble MPNN weights bundled with the matching LigandMPNN snapshot; each was verified against this ProteinMPNN class and renamed to the filename contract expected by `protein_mpnn_run.py`.

## Adapter entry

- CLI: `community_models/proteinmpnn/source/protein_mpnn_run.py`
- Vanilla weights: `community_models/proteinmpnn/checkpoint/vanilla`.
- CA weights: `community_models/proteinmpnn/checkpoint/ca` plus `--ca_only`.
- Soluble weights: `community_models/proteinmpnn/checkpoint/soluble` plus `--use_soluble_model`.

The adapter must always pass the weight directory explicitly because the untouched upstream CLI defaults still describe its original sibling-directory layout. It should also own `--pdb_path`, `--pdb_path_chains`, and `--out_folder`; these are pipeline plumbing rather than user model behavior.

The supported PartiDesign runtime is the repository's pinned Python 3.12.6 / PyTorch 2.9.1+cu128 environment. Runtime subprocesses receive local assets only and an IP-socket-denying `sitecustomize`.

## Available behavior parameters

- Model behavior: `--model_name`, `--ca_only`, `--use_soluble_model`, `--backbone_noise`, `--max_length`.
- Sampling: `--seed`, `--num_seq_per_target`, `--batch_size`, `--sampling_temp`.
- Design masks and coupling: `--chain_id_jsonl`, `--fixed_positions_jsonl`, `--tied_positions_jsonl`.
- Amino-acid constraints: `--omit_AAs`, `--omit_AA_jsonl`, `--bias_AA_jsonl`, `--bias_by_res_jsonl`.
- PSSM: `--pssm_jsonl`, `--pssm_multi`, `--pssm_threshold`, `--pssm_log_odds_flag`, `--pssm_bias_flag`.
- Score/probability modes: `--score_only`, `--path_to_fasta`, `--conditional_probs_only`, `--conditional_probs_only_backbone`, `--unconditional_probs_only`, `--save_score`, `--save_probs`.
- Alternate parsed input is available through `--jsonl_path`; PipeTools should normally use one PDB at a time and set the single design chain itself.

## Checkpoints

| Family | Files | Size per file (bytes) |
|---|---|---:|
| vanilla | `v_48_002.pt`, `v_48_010.pt`, `v_48_020.pt`, `v_48_030.pt` | 6,681,301 |
| CA-only | `v_48_002.pt`, `v_48_010.pt`, `v_48_020.pt` | 6,624,011 |
| soluble | `v_48_002.pt`, `v_48_010.pt`, `v_48_020.pt`, `v_48_030.pt` | 6,681,301 |

Total checkpoint bytes: 73,322,441. Verification is executable and does not retain a smoke-test log.
