# Modifications Copyright 2026 PartiDesign contributors
# Inference-only integration: omit training/content-digest helpers.
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
# Upstream file: protenix/utils/file_io.py

import gzip
import json
import pickle
import time
from typing import Any, TYPE_CHECKING, Union

# Only import lmdb for type checking (no runtime cost)
if TYPE_CHECKING:
    import lmdb

import os
from pathlib import Path

import pandas as pd
import torch
from biotite import structure
from biotite.structure.io import pdbx

from mol.data.constants import mmcif_restype_1to3
from mol.utils.logger import get_logger
from mol.utils.torch_utils import map_values_to_list, to_device

logger = get_logger(__name__)

PANDAS_NA_VALUES = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    # "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
]

_INDICES_CSV_CACHE: dict[str, pd.DataFrame] = {}
_JSON_FILE_CACHE: dict[str, Any] = {}


def read_indices_csv(
    csv: Union[str, Path], load_indice_mode: str = "full"
) -> pd.DataFrame:
    """
    Read a csv file without the content changing.

    Args:
        csv (Union[str, Path]): A csv file path.
        load_indice_mode (str): Mode to load indices. Defaults to "full".

    Returns:
        pd.DataFrame : A pandas DataFrame.
    """
    csv_str = str(csv)

    if csv_str in _INDICES_CSV_CACHE:
        df_cached = _INDICES_CSV_CACHE[csv_str]
        logger.info(
            "[Mol-1 IO] read_indices_csv cache hit: path=%s, shape=%s",
            csv_str,
            tuple(df_cached.shape),
        )
        # Always return a copy so per-dataset filtering does not mutate the cache.
        return df_cached.copy(deep=True)

    t0 = time.time()
    if csv_str.endswith(".parquet"):
        if load_indice_mode == "simple":
            df = pd.read_parquet(
                csv_str,
                columns=["pdb_id", "type", "chain_1_id", "chain_2_id", "cluster_id"],
            )
        else:
            df = pd.read_parquet(csv_str)
        df = df.astype(str)
    else:
        df = pd.read_csv(
            csv_str, na_values=PANDAS_NA_VALUES, keep_default_na=False, dtype=str
        )

    def _normalize_entity_id(entity_id):
        if pd.isna(entity_id):
            return None
        try:
            return str(int(float(entity_id)))
        except (ValueError, TypeError):
            return None

    if "entity_1_id" in df.columns:
        df["entity_1_id"] = df["entity_1_id"].apply(_normalize_entity_id)
    if "entity_2_id" in df.columns:
        df["entity_2_id"] = df["entity_2_id"].apply(_normalize_entity_id)

    _INDICES_CSV_CACHE[csv_str] = df
    logger.info(
        "[Mol-1 IO] read_indices_csv finished in %.3fs: path=%s, shape=%s",
        time.time() - t0,
        csv_str,
        tuple(df.shape),
    )
    # Return a copy so callers can safely modify.
    return df.copy(deep=True)



def load_json_cached(path: Union[str, Path]) -> Any:
    """
    Load a JSON/Parquet/LMDB file with a simple in-process cache.

    - For JSON: Loads entire file into RAM (returns dict).
    - For Parquet: Loads into DataFrame (returns DataFrame).
    - For LMDB: Opens connection handle (not used by this inference distribution).
    """
    path_str = str(Path(path))

    # 1. Cache JSON results per process; LMDB always returns a fresh wrapper.
    if path_str.endswith(".json") and path_str in _JSON_FILE_CACHE:
        logger.info("[Mol-1 IO] load_json_cached cache hit: path=%s", path_str)
        return _JSON_FILE_CACHE[path_str]

    t0 = time.time()

    # 2. Load according to suffix
    if path_str.endswith(".lmdb"):
        # LMDB branch: construct a new LMDBDict instance (not cached in _JSON_FILE_CACHE).
        raise ValueError("Mol-1 inference accepts JSON indexes, not training LMDB indexes")

    elif path_str.endswith(".json"):
        # JSON branch: load full JSON into memory and cache it.
        with open(path_str, "r") as f:
            try:
                import orjson

                data = orjson.loads(f.read())
            except ImportError:
                data = json.load(f)
        _JSON_FILE_CACHE[path_str] = data
    elif path_str == ".":
        # Empty JSON file
        data = {}
    else:
        raise ValueError(f"Unsupported file format: {path_str}")

    logger.info(
        "[Mol-1 IO] load_json_cached finished in %.3fs: path=%s",
        time.time() - t0,
        path_str,
    )
    return data


def load_gzip_pickle(pkl: Union[str, Path]) -> Any:
    """
    Load a gzip pickle file.

    Args:
        pkl (Union[str, Path]): A gzip pickle file path.

    Returns:
        Any: The loaded data.
    """
    with gzip.open(pkl, "rb") as f:
        data = pickle.load(f)
    return data


def dump_gzip_pickle(data: Any, pkl: Union[str, Path]):
    """
    Dump a gzip pickle file.

    Args:
        data (Any): The data to be dumped.
        pkl (Union[str, Path]): A gzip pickle file path.
    """
    with gzip.open(pkl, "wb") as f:
        pickle.dump(data, f)


def pdb_to_cif(pdb_file: str, cif_file: str, sequence=None):
    """
    Convert monomer protein pdb to CIF, and fill "entity_poly" and "entity_poly_seq"category.

    pdb_file (str): PDB file path.
    cif_file (str): output CIF file path.
    """
    entity_poly = pdbx.CIFCategory({"entity_id": "0", "type": "polypeptide(L)"})
    block_dict = {"entity_poly": entity_poly}

    if sequence is not None:
        entity_id = ["0"] * len(sequence)
        num = []
        mon_id = []
        for idx, i in enumerate(sequence):
            num.append(str(idx + 1))
            mon_id.append(mmcif_restype_1to3.get(i, "UNK"))
            assert (
                i in mmcif_restype_1to3 or i == "X"
            ), f"{i} is not in mmcif_restype_1to3"

        block_dict["entity_poly_seq"] = pdbx.CIFCategory(
            {
                "entity_id": entity_id,
                "num": num,
                "mon_id": mon_id,
            }
        )

    block = pdbx.CIFBlock(block_dict)
    pdb_file = Path(pdb_file)
    block_name = pdb_file.stem
    cif = pdbx.CIFFile({block_name: block})
    atom_array = structure.io.load_structure(pdb_file, extra_fields=["b_factor"])
    pdbx.set_structure(cif, atom_array)
    cif.write(cif_file)


class FloatEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        self.precision = kwargs.pop("precision", 2)
        super(FloatEncoder, self).__init__(*args, **kwargs)

    def encode(self, obj):
        def float_converter(o):
            if isinstance(o, float):
                return format(o, f".{self.precision}f")
            if isinstance(o, list):
                return [float_converter(i) for i in o]
            if isinstance(o, dict):
                return {k: float_converter(v) for k, v in o.items()}
            return o

        return super(FloatEncoder, self).encode(float_converter(obj))


def save_json(data, output_fpath, indent=4):
    data_json = data.copy()
    data_json = map_values_to_list(data_json)
    with open(output_fpath, "w") as f:
        if indent is not None:
            json.dump(data_json, f, indent=indent)
        else:
            json.dump(data_json, f)


def save_tensor(data, output_fpath):
    torch.save(to_device(data, device=None), output_fpath)
