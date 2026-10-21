"""Offline, fail-closed runtime for the vendored NA-MPNN model.

The upstream package is deliberately imported by file path from the immutable
``community_models/na_mpnn`` tree.  This module does not import PipeTools or
depend on its environment; the worker is the only public execution boundary.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from partidesign.paths import community_asset, community_source
from types import ModuleType
from typing import Any


class NAMPNNError(RuntimeError):
    """Base class for the standalone NA-MPNN adapter."""


class NAMPSystemError(NAMPNNError):
    """An asset, interpreter, checkpoint, or model-load failure."""


class NAMPSequenceError(NAMPNNError):
    """A single candidate could not be parsed, sampled, or validated."""


ASSET_RELATIVE_ROOT = Path("community_models/na_mpnn")


_CUDA_VISIBLE_DEVICE_TOKEN = re.compile(
    r"(?:[0-9]+|GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}|"
    r"MIG-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12})"
)


def _repo_root(repo_root: Path | str | None) -> Path:
    if repo_root is None:
        return Path(__file__).resolve().parents[3]
    return Path(repo_root).expanduser().resolve(strict=True)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise NAMPSystemError(f"cannot import vendored NA-MPNN module: {path}")
    module = importlib.util.module_from_spec(spec)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise NAMPSystemError(
            "cannot import vendored NA-MPNN module: "
            f"{path}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    return module


@dataclass(frozen=True)
class NAMPNNVocab:
    restype_to_int: dict[str, int]
    restype_int_to_str: dict[int, str]
    restype_1_to_3: dict[str, str]
    dna_char_to_rna_char: dict[str, str]
    atom_dict: dict[str, int]
    polytype_to_int: dict[str, int]
    alphabet: tuple[str, ...]
    num_letters: int = 33
    vocab: int = 33


def make_na_mpnn_vocab() -> NAMPNNVocab:
    restypes = (
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL", "UNK", "DA", "DC", "DG", "DT", "DX", "A", "C",
        "G", "U", "RX", "MAS", "PAD",
    )
    restype_to_int = {name: index for index, name in enumerate(restypes)}
    restype_to_int.update({"A": restype_to_int["DA"], "C": restype_to_int["DC"]})
    restype_to_int.update({"G": restype_to_int["DG"], "U": restype_to_int["DT"]})
    restype_to_int["RX"] = restype_to_int["DX"]
    restype_3_to_1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "UNK": "X", "DA": "a", "DC": "c", "DG": "g", "DT": "t",
        "DX": "x", "A": "b", "C": "d", "G": "h", "U": "u", "RX": "y",
        "MAS": "-", "PAD": "+",
    }
    restype_int_to_str: dict[int, str] = {}
    for name in restypes:
        index = restype_to_int[name]
        restype_int_to_str.setdefault(index, restype_3_to_1[name])
    restype_1_to_3 = {value: key for key, value in restype_3_to_1.items()}
    atom_types = (
        "N", "CA", "C", "O", "OP1", "OP2", "P", "O5'", "C5'", "C4'",
        "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    )
    polytypes = ("PP", "DNA", "RNA", "UNK", "MAS", "PAD")
    return NAMPNNVocab(
        restype_to_int=restype_to_int,
        restype_int_to_str=restype_int_to_str,
        restype_1_to_3=restype_1_to_3,
        dna_char_to_rna_char={"a": "b", "c": "d", "g": "h", "t": "u", "x": "y"},
        atom_dict={name: index for index, name in enumerate(atom_types)},
        polytype_to_int={name: index for index, name in enumerate(polytypes)},
        alphabet=tuple(restype_3_to_1[name] for name in restypes),
    )


@dataclass(frozen=True)
class LoadedNAMPNN:
    model: Any
    data_utils: ModuleType
    device: Any
    vocab: NAMPNNVocab


def _resolve_local_cuda_device(torch_module: Any) -> Any:
    """Resolve the CUDA device owned by this worker process.

    ``CUDA_VISIBLE_DEVICES`` names physical devices, while PyTorch addresses
    the resulting visible set with local ordinals.  Consequently a worker
    inheriting ``CUDA_VISIBLE_DEVICES=3,7`` with ``LOCAL_RANK=1`` must use
    ``cuda:1`` (the second visible device, physical GPU 7), not ``cuda:7``.
    A parent that has already narrowed the process to one device always owns
    local ``cuda:0`` regardless of its rank.
    """

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        raise NAMPSystemError("NA-MPNN requires a CUDA-capable PyTorch runtime")
    try:
        available = bool(cuda.is_available())
        visible_count = int(cuda.device_count())
    except Exception as exc:
        raise NAMPSystemError(
            f"cannot query CUDA devices for NA-MPNN: {type(exc).__name__}: {exc}"
        ) from exc
    if not available or visible_count <= 0:
        raise NAMPSystemError(
            "NA-MPNN requires at least one visible CUDA device; none is available"
        )

    raw_rank = os.environ.get("LOCAL_RANK")
    if raw_rank is None:
        local_rank = 0
    else:
        rank_text = raw_rank.strip()
        if not rank_text.isdigit():
            raise NAMPSystemError(
                f"invalid LOCAL_RANK for NA-MPNN owner GPU: {raw_rank!r}"
            )
        local_rank = int(rank_text)

    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_visible is None:
        local_index = local_rank
    else:
        if not raw_visible.strip():
            raise NAMPSystemError(
                "CUDA_VISIBLE_DEVICES explicitly hides every NA-MPNN owner GPU"
            )
        visible = [value.strip() for value in raw_visible.split(",")]
        if any(
            not value or _CUDA_VISIBLE_DEVICE_TOKEN.fullmatch(value) is None
            for value in visible
        ):
            raise NAMPSystemError(
                "CUDA_VISIBLE_DEVICES does not name a usable NA-MPNN GPU set"
            )
        normalized_visible = [
            str(int(value)) if value.isdigit() else value.casefold()
            for value in visible
        ]
        if len(set(normalized_visible)) != len(normalized_visible):
            raise NAMPSystemError(
                "CUDA_VISIBLE_DEVICES contains duplicate NA-MPNN GPU owners"
            )
        # A single visible physical GPU is always local cuda:0.  For a
        # multi-device list LOCAL_RANK selects the corresponding local slot.
        local_index = 0 if len(visible) == 1 else local_rank

    if local_index < 0 or local_index >= visible_count:
        raise NAMPSystemError(
            "LOCAL_RANK does not resolve one visible NA-MPNN owner GPU: "
            f"local_index={local_index}, visible_count={visible_count}"
        )
    try:
        return torch_module.device(f"cuda:{local_index}")
    except Exception as exc:
        raise NAMPSystemError(
            "cannot construct the resolved NA-MPNN CUDA device: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def load_na_mpnn_model(
    repo_root: Path | str | None = None,
    *,
    device: str | None = None,
) -> LoadedNAMPNN:
    """Load the vendored design model exactly once per worker batch."""

    asset_root = community_asset("na_mpnn", _repo_root(repo_root))
    source_root = community_source("na_mpnn", _repo_root(repo_root))
    try:
        import torch
    except ImportError as exc:
        raise NAMPSystemError("PartiDesign interpreter lacks PyTorch for NA-MPNN") from exc
    if device is None:
        chosen_device = _resolve_local_cuda_device(torch)
    else:
        try:
            chosen_device = torch.device(device)
        except Exception as exc:
            raise NAMPSystemError(
                f"invalid NA-MPNN device {device!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    data_utils = _load_module(
        source_root / "inference/data_utils.py",
        "partidesign_na_mpnn_data_utils",
    )
    model_utils = _load_module(
        source_root / "inference/model_utils.py",
        "partidesign_na_mpnn_model_utils",
    )
    checkpoint_path = asset_root / "checkpoint/s_19137.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=chosen_device, weights_only=False)
        vocab = make_na_mpnn_vocab()
        model = model_utils.ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=32,
            model_type="na_mpnn",
            vocab=vocab.vocab,
            num_letters=vocab.num_letters,
            atom_dict=vocab.atom_dict,
            restype_to_int=vocab.restype_to_int,
            polytype_to_int=vocab.polytype_to_int,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(chosen_device)
        model.eval()
    except NAMPSystemError as exc:
        raise NAMPSystemError(
            f"failed to load the pinned NA-MPNN model: {exc}"
        ) from exc
    except Exception as exc:
        raise NAMPSystemError(
            "failed to load the pinned NA-MPNN model: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return LoadedNAMPNN(model=model, data_utils=data_utils, device=chosen_device, vocab=vocab)
