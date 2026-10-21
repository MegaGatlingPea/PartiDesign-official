"""Pure model-token projection over the source-faithful authoring AtomArray."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from biotite.structure import AtomArray

from .records import ModifiedPolymerCarrier


@dataclass(frozen=True, slots=True)
class ModelTokenLayout:
    """Immutable correspondence between authoring and checkpoint token axes.

    Ordinary authoring tokens remain one model token.  A modified-polymer carrier
    expands its one source polymer token into one model token per source atom, ordered
    by stable AtomUID.  Every row and AtomUID still refers to the sole authoring
    AtomArray; this object owns no coordinates, annotations, or bonds.
    """

    rows_by_model_token: tuple[tuple[int, ...], ...]
    model_token_to_row: tuple[int, ...]
    model_token_to_authoring_token: tuple[int, ...]
    model_token_by_row: tuple[int, ...]
    model_token_by_atom_uid: Mapping[int, int]
    authoring_token_to_model_tokens: tuple[tuple[int, ...], ...]
    carrier_token_mask: tuple[bool, ...]
    carrier_residue_uid_by_model_token: tuple[int | None, ...]
    model_molecule_uid_by_model_token: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rows_by_model_token",
            tuple(tuple(int(row) for row in rows) for rows in self.rows_by_model_token),
        )
        for field_name in (
            "model_token_to_row",
            "model_token_to_authoring_token",
            "model_token_by_row",
            "model_molecule_uid_by_model_token",
        ):
            object.__setattr__(
                self,
                field_name,
                tuple(int(value) for value in getattr(self, field_name)),
            )
        object.__setattr__(
            self,
            "authoring_token_to_model_tokens",
            tuple(
                tuple(int(token_uid) for token_uid in token_uids)
                for token_uids in self.authoring_token_to_model_tokens
            ),
        )
        object.__setattr__(
            self,
            "carrier_token_mask",
            tuple(bool(value) for value in self.carrier_token_mask),
        )
        object.__setattr__(
            self,
            "carrier_residue_uid_by_model_token",
            tuple(
                None if value is None else int(value)
                for value in self.carrier_residue_uid_by_model_token
            ),
        )
        atom_uid_mapping = {
            int(atom_uid): int(token_uid)
            for atom_uid, token_uid in self.model_token_by_atom_uid.items()
        }
        object.__setattr__(
            self,
            "model_token_by_atom_uid",
            MappingProxyType(dict(sorted(atom_uid_mapping.items()))),
        )

        token_count = len(self.rows_by_model_token)
        token_axis_fields: tuple[Sequence[object], ...] = (
            self.model_token_to_row,
            self.model_token_to_authoring_token,
            self.carrier_token_mask,
            self.carrier_residue_uid_by_model_token,
            self.model_molecule_uid_by_model_token,
        )
        if any(len(values) != token_count for values in token_axis_fields):
            raise ValueError("ModelTokenLayout token-axis fields disagree")
        if any(not rows for rows in self.rows_by_model_token):
            raise ValueError("every model token must own at least one authoring row")

        row_count = len(self.model_token_by_row)
        seen_rows: set[int] = set()
        for model_token, rows in enumerate(self.rows_by_model_token):
            if self.model_token_to_row[model_token] not in rows:
                raise ValueError("model_token_to_row must select a row owned by its token")
            for row in rows:
                if row < 0 or row >= row_count or row in seen_rows:
                    raise ValueError("model-token rows must partition the authoring rows")
                seen_rows.add(row)
                if self.model_token_by_row[row] != model_token:
                    raise ValueError("model_token_by_row disagrees with rows_by_model_token")
        if seen_rows != set(range(row_count)):
            raise ValueError("model-token rows do not cover the authoring AtomArray")
        if len(self.model_token_by_atom_uid) != row_count:
            raise ValueError("AtomUID mapping must cover every authoring row")

        seen_model_tokens: set[int] = set()
        for authoring_token, model_tokens in enumerate(
            self.authoring_token_to_model_tokens
        ):
            if not model_tokens:
                raise ValueError("every authoring token must map to a model token")
            for model_token in model_tokens:
                if (
                    model_token < 0
                    or model_token >= token_count
                    or model_token in seen_model_tokens
                ):
                    raise ValueError(
                        "authoring-to-model tokens must partition the model-token axis"
                    )
                seen_model_tokens.add(model_token)
                if self.model_token_to_authoring_token[model_token] != authoring_token:
                    raise ValueError(
                        "model_token_to_authoring_token disagrees with its inverse"
                    )
        if seen_model_tokens != set(range(token_count)):
            raise ValueError("authoring tokens do not cover the model-token axis")

        for model_token, is_carrier in enumerate(self.carrier_token_mask):
            residue_uid = self.carrier_residue_uid_by_model_token[model_token]
            if is_carrier != (residue_uid is not None):
                raise ValueError("carrier token mask disagrees with carrier residue UID")
            if is_carrier and len(self.rows_by_model_token[model_token]) != 1:
                raise ValueError("each carrier model token must own exactly one row")
        if set(self.model_token_by_atom_uid.values()).difference(range(token_count)):
            raise ValueError("AtomUID mapping references an invalid model token")

    @property
    def num_model_tokens(self) -> int:
        return len(self.rows_by_model_token)

    @property
    def num_authoring_tokens(self) -> int:
        return len(self.authoring_token_to_model_tokens)


def _normalized_carriers(
    carriers: Mapping[int, ModifiedPolymerCarrier],
) -> dict[int, ModifiedPolymerCarrier]:
    if not isinstance(carriers, Mapping):
        raise TypeError("modified-polymer carriers must be a mapping")
    result: dict[int, ModifiedPolymerCarrier] = {}
    model_molecule_uids: set[int] = set()
    for raw_residue_uid, carrier in carriers.items():
        if isinstance(raw_residue_uid, bool) or not isinstance(
            raw_residue_uid, (int, np.integer)
        ):
            raise TypeError("carrier keys must be ResidueUID integers")
        residue_uid = int(raw_residue_uid)
        if not isinstance(carrier, ModifiedPolymerCarrier):
            raise TypeError("carrier values must be ModifiedPolymerCarrier records")
        if carrier.residue_uid != residue_uid:
            raise ValueError("carrier key disagrees with its residue_uid")
        if carrier.model_molecule_uid in model_molecule_uids:
            raise ValueError("carrier model_molecule_uid values must be unique")
        model_molecule_uids.add(carrier.model_molecule_uid)
        result[residue_uid] = carrier
    return dict(sorted(result.items()))


def build_model_token_layout(
    atoms: AtomArray,
    carriers: Mapping[int, ModifiedPolymerCarrier],
) -> ModelTokenLayout:
    """Derive the checkpoint token axis without mutating or copying ``atoms``."""

    if not isinstance(atoms, AtomArray):
        raise TypeError("atoms must be a Biotite AtomArray")
    normalized_carriers = _normalized_carriers(carriers)
    row_count = len(atoms)
    atom_uids = np.asarray(atoms.atom_uid, dtype=np.int64)
    if len(np.unique(atom_uids)) != row_count or bool((atom_uids < 0).any()):
        raise ValueError("model-token layout requires unique finalized AtomUIDs")

    authoring_token_uids = np.asarray(atoms.token_uid, dtype=np.int64)
    unique_authoring_tokens = np.unique(authoring_token_uids)
    if not np.array_equal(
        unique_authoring_tokens,
        np.arange(len(unique_authoring_tokens), dtype=np.int64),
    ):
        raise ValueError("authoring token_uid values must be contiguous from zero")

    source_molecule_uids = {
        int(value) for value in np.asarray(atoms.molecule_uid, dtype=np.int64)
    }
    if any(value < 0 for value in source_molecule_uids):
        raise ValueError("model-token layout requires finalized source molecule UIDs")
    collisions = source_molecule_uids.intersection(
        carrier.model_molecule_uid for carrier in normalized_carriers.values()
    )
    if collisions:
        raise ValueError(
            "carrier model molecule IDs collide with source molecule IDs: "
            f"{sorted(collisions)}"
        )

    rows_by_model_token: list[tuple[int, ...]] = []
    model_token_to_row: list[int] = []
    model_token_to_authoring_token: list[int] = []
    model_token_by_row = [-1] * row_count
    model_token_by_atom_uid: dict[int, int] = {}
    authoring_token_to_model_tokens: list[tuple[int, ...]] = []
    carrier_token_mask: list[bool] = []
    carrier_residue_uid_by_model_token: list[int | None] = []
    model_molecule_uid_by_model_token: list[int] = []
    seen_carrier_residue_uids: set[int] = set()

    for authoring_token in unique_authoring_tokens.tolist():
        token_rows_array = np.where(authoring_token_uids == int(authoring_token))[0]
        token_rows = tuple(int(row) for row in token_rows_array.tolist())
        if not token_rows:
            raise AssertionError(authoring_token)
        modalities = {str(atoms.mol_type[row]) for row in token_rows}
        if len(modalities) != 1:
            raise ValueError("one authoring token spans multiple source modalities")
        modality = next(iter(modalities))
        if modality == "ligand" and len(token_rows) != 1:
            raise ValueError("each source ligand authoring token must own one atom")

        token_residue_uids = {
            int(atoms.residue_uid[row]) for row in token_rows
        }
        matching_carrier_uids = token_residue_uids.intersection(normalized_carriers)
        if len(matching_carrier_uids) > 1:
            raise ValueError("one authoring token spans multiple carrier residues")
        emitted_model_tokens: list[int] = []

        if matching_carrier_uids:
            residue_uid = next(iter(matching_carrier_uids))
            if residue_uid in seen_carrier_residue_uids:
                raise ValueError(
                    "one carrier residue spans multiple authoring tokens"
                )
            if token_residue_uids != {residue_uid}:
                raise ValueError(
                    "carrier residue shares its authoring token with another residue"
                )
            if modality not in {"protein", "rna", "dna"}:
                raise ValueError("carrier source modality must remain polymer")
            carrier = normalized_carriers[residue_uid]
            ordered_rows = tuple(
                sorted(token_rows, key=lambda row: (int(atom_uids[row]), row))
            )
            for row in ordered_rows:
                model_token = len(rows_by_model_token)
                emitted_model_tokens.append(model_token)
                rows_by_model_token.append((row,))
                model_token_to_row.append(row)
                model_token_to_authoring_token.append(int(authoring_token))
                model_token_by_row[row] = model_token
                model_token_by_atom_uid[int(atom_uids[row])] = model_token
                carrier_token_mask.append(True)
                carrier_residue_uid_by_model_token.append(residue_uid)
                model_molecule_uid_by_model_token.append(carrier.model_molecule_uid)
            seen_carrier_residue_uids.add(residue_uid)
        else:
            molecule_uids = {int(atoms.molecule_uid[row]) for row in token_rows}
            if len(molecule_uids) != 1:
                raise ValueError("one authoring token spans multiple source molecules")
            center_rows = [row for row in token_rows if bool(atoms.centre_atom_mask[row])]
            if len(center_rows) != 1:
                raise ValueError(
                    "ordinary authoring token requires exactly one representative row"
                )
            model_token = len(rows_by_model_token)
            emitted_model_tokens.append(model_token)
            rows_by_model_token.append(token_rows)
            model_token_to_row.append(center_rows[0])
            model_token_to_authoring_token.append(int(authoring_token))
            for row in token_rows:
                model_token_by_row[row] = model_token
                model_token_by_atom_uid[int(atom_uids[row])] = model_token
            carrier_token_mask.append(False)
            carrier_residue_uid_by_model_token.append(None)
            model_molecule_uid_by_model_token.append(next(iter(molecule_uids)))

        authoring_token_to_model_tokens.append(tuple(emitted_model_tokens))

    missing_carriers = set(normalized_carriers).difference(seen_carrier_residue_uids)
    if missing_carriers:
        raise ValueError(
            "carrier residues are absent from the authoring token axis: "
            f"{sorted(missing_carriers)}"
        )
    return ModelTokenLayout(
        rows_by_model_token=tuple(rows_by_model_token),
        model_token_to_row=tuple(model_token_to_row),
        model_token_to_authoring_token=tuple(model_token_to_authoring_token),
        model_token_by_row=tuple(model_token_by_row),
        model_token_by_atom_uid=MappingProxyType(model_token_by_atom_uid),
        authoring_token_to_model_tokens=tuple(authoring_token_to_model_tokens),
        carrier_token_mask=tuple(carrier_token_mask),
        carrier_residue_uid_by_model_token=tuple(
            carrier_residue_uid_by_model_token
        ),
        model_molecule_uid_by_model_token=tuple(model_molecule_uid_by_model_token),
    )


__all__ = ["ModelTokenLayout", "build_model_token_layout"]
