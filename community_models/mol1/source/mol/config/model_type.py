# Copyright 2024 ByteDance and/or its affiliates.
# Modifications Copyright 2026 Molence Bio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Molence Bio
# Based on Protenix: https://github.com/bytedance/Protenix
# Upstream revision: bd54a05d047b8925a241056f36d619700604068a
# Upstream file: configs/configs_model_type.py

# Architecture verified against the released checkpoint. Runtime settings remain in
# mol/config/base.py and mol/config/inference.py and can be overridden by the CLI.
model_configs = {
    "mol-1-base": {
        "c_s": 384,
        "c_z": 448,
        "c_z_pairmixer": 448,
        "c_z_pairformer": 320,
        "c_s_inputs": 449,
        "c_atom": 128,
        "c_atompair": 16,
        "c_token": 384,
        "n_blocks": 48,
        "n_pairmixer_blocks": 8,
        "hidden_scale_up": True,
        "max_atoms_per_token": 24,
        "no_bins": 64,
        "model": {
            "template_embedder": {
                "c": 64,
                "n_blocks": 2,
            },
            "msa_module": {
                "c_m": 192,
                "n_blocks": 4,
            },
            "pairformer": {
                "n_heads": 16,
            },
            "diffusion_module": {
                "c_token": 768,
                "atom_encoder": {
                    "n_blocks": 3,
                    "n_heads": 4,
                },
                "transformer": {
                    "n_blocks": 24,
                    "n_heads": 16,
                },
                "atom_decoder": {
                    "n_blocks": 3,
                    "n_heads": 4,
                },
            },
            "confidence_head": {
                "n_blocks": 4,
            },
        },
    },
}
