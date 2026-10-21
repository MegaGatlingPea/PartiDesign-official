"""Small exact protein chi-angle kernel used by the frozen VAE encoder.

This is the inference-relevant part of OpenFold's atom37 torsion transform.  The
existing encoder consumes only the four side-chain chi angles, so copying the full
training transform and its unrelated structure stack would add no checkpoint-facing
behavior.  The signed-dihedral convention is numerically equivalent to the OpenFold
chi output and is locked by a reference test.
"""

from collections.abc import Mapping

import torch

from src.design.model.angle_utils import signed_dihedral_angle
from src.design.model.constants import PROTEIN_ATOM37, TOKEN_TYPE32


_CHI_ATOMS: Mapping[str, tuple[tuple[str, str, str, str], ...]] = {
    "ALA": (),
    "ARG": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "NE"),
        ("CG", "CD", "NE", "CZ"),
    ),
    "ASN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "ASP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "CYS": (("N", "CA", "CB", "SG"),),
    "GLN": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ),
    "GLU": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ),
    "GLY": (),
    "HIS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
    "ILE": (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")),
    "LEU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "LYS": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "CE"),
        ("CG", "CD", "CE", "NZ"),
    ),
    "MET": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "SD"),
        ("CB", "CG", "SD", "CE"),
    ),
    "PHE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "PRO": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "SER": (("N", "CA", "CB", "OG"),),
    "THR": (("N", "CA", "CB", "OG1"),),
    "TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "TYR": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "VAL": (("N", "CA", "CB", "CG1"),),
}

_ATOM_INDEX = {name: index for index, name in enumerate(PROTEIN_ATOM37)}


def protein_chi_angles(
    token_type32: torch.Tensor,
    atom37_coords: torch.Tensor,
    atom37_mask: torch.Tensor,
    protein_token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return four chi angles and their validity mask for every token."""

    if atom37_coords.shape[:-1] != atom37_mask.shape:
        raise ValueError("atom37 coordinate and mask shapes are inconsistent")
    if atom37_coords.shape[:-2] != token_type32.shape:
        raise ValueError("atom37 coordinates and token types are inconsistent")
    if protein_token_mask.shape != token_type32.shape:
        raise ValueError("protein token mask and token types are inconsistent")

    torsion_dtype = (
        torch.float32 if atom37_coords.device.type == "mps" else torch.float64
    )
    coords = atom37_coords.to(dtype=torsion_dtype)
    valid_atoms = atom37_mask.bool() & protein_token_mask[..., None]
    angles = coords.new_zeros(token_type32.shape + (4,))
    valid = torch.zeros_like(angles, dtype=torch.bool)

    for residue_name, residue_chis in _CHI_ATOMS.items():
        residue_mask = protein_token_mask & (
            token_type32 == TOKEN_TYPE32[residue_name]
        )
        for chi_index, atom_names in enumerate(residue_chis):
            atom_indices = tuple(_ATOM_INDEX[name] for name in atom_names)
            chi_valid = residue_mask
            for atom_index in atom_indices:
                chi_valid = chi_valid & valid_atoms[..., atom_index]
            chi_angle = signed_dihedral_angle(
                *(coords[..., atom_index, :] for atom_index in atom_indices)
            )
            angles[..., chi_index] = torch.where(
                chi_valid,
                chi_angle,
                angles[..., chi_index],
            )
            valid[..., chi_index] = valid[..., chi_index] | chi_valid

    return angles, valid
