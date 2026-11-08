"""Resident backend factory shared by validation, pipeline and screening."""
from __future__ import annotations


def create_backend(repo, method, full_config, device="cuda:0"):
    """Create an existing method service; its owner controls resident lifetime."""
    from .config import config_for_method
    config = config_for_method(full_config, method)
    if method == "af2_ig":
        from .af2 import AF2Backend
        return AF2Backend(repo, config["af2"], device)
    if method in {"esmfold2-fast", "esmfold2-standard"}:
        from .esmfold2 import ESMFold2Backend
        return ESMFold2Backend(repo, config["esmfold2"], device)
    if method == "mol1":
        from .mol1 import Mol1Backend
        return Mol1Backend(repo, config["mol1"], device)
    if method == "boltz2":
        from .boltz2 import Boltz2Backend
        return Boltz2Backend(repo, config["boltz2"], device)
    if method in {"af3", "af3-la"}:
        from .af3 import AF3Backend, AF3Validator
        return AF3Validator(AF3Backend(repo, config["af3"], method=method, device=device),
                            require_structure=True)
    raise ValueError(f"Unsupported resident prediction method: {method!r}")


__all__ = ["create_backend"]
