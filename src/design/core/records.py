from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from biotite.structure import AtomArray


_STANDARD_PARENT_BY_MODALITY = {
    "protein": frozenset(
        {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
            "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
            "THR", "TRP", "TYR", "VAL",
        }
    ),
    "rna": frozenset({"A", "C", "G", "U"}),
    "dna": frozenset({"DA", "DC", "DG", "DT"}),
}


class DesignError(Exception):
    """Base class for standalone inference errors."""


class DesignConfigError(DesignError, ValueError):
    """The public configuration or task grammar is invalid."""


class DesignInputError(DesignError, ValueError):
    """A structure cannot be normalized under the reader contract."""


class DesignSampleError(DesignError, ValueError):
    """One expanded sample cannot be compiled under the task contract."""


@dataclass(frozen=True)
class DesignRunResult:
    output_dir: Path
    succeeded: int
    failed: int
    postprocessed: int = 0
    postprocess_failed: int = 0
    scored: int = 0
    score_failed: int = 0


@dataclass(frozen=True)
class DesignChainSpec:
    chain_id: str
    modality: str
    sub_modality: str | None = None

    def __post_init__(self) -> None:
        """Keep semantic design aliases separate from chemistry modality.

        Peptide is an authoring alias for a protein design chain. The compiler
        and model routing must continue to see protein while reporting and
        recovery can retain the peptide intent in the optional sub-modality
        field. The two-field constructor remains source-compatible because
        sub_modality defaults to None.
        """

        modality = self.modality
        sub_modality = self.sub_modality
        if not isinstance(modality, str):
            raise DesignConfigError("design chain modality must be a string")
        if modality == "peptide":
            if sub_modality not in (None, "peptide"):
                raise DesignConfigError(
                    "peptide design chains may only use sub_modality=peptide"
                )
            modality = "protein"
            sub_modality = "peptide"
        elif modality not in {"protein", "rna", "dna", "ligand"}:
            raise DesignConfigError(f"unsupported design chain modality: {modality!r}")

        if sub_modality is not None:
            if not isinstance(sub_modality, str):
                raise DesignConfigError("design chain sub_modality must be a string or null")
            if sub_modality != "peptide":
                raise DesignConfigError(
                    f"unsupported design chain sub_modality: {sub_modality!r}"
                )
            if modality != "protein":
                raise DesignConfigError(
                    "sub_modality=peptide requires chemistry modality=protein"
                )

        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "sub_modality", sub_modality)


@dataclass(frozen=True)
class LengthPlan:
    index: int
    lengths: Mapping[str, int]


@dataclass(frozen=True)
class CropProximitySpec:
    """One source-structure proximity selector for a task crop.

    Selectors are intentionally kept as public strings here.  Parsing them
    into AtomUID/residue ranges belongs to the later crop compiler, not task
    configuration loading.
    """

    anchor: str
    radius: float


@dataclass(frozen=True)
class CropSpec:
    """Parsed, closed task crop configuration.

    ``include=None`` means that the full input is eligible before the
    proximity/exclude operations are applied.  The compiler consumes this
    immutable record to project a fully read source ``AtomArray`` while
    preserving the task's crop contract.
    """

    include: tuple[str, ...] | None = None
    proximity: tuple[CropProximitySpec, ...] = ()
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class PartialSpec:
    """One complete candidate structure used as a partial-denoising reference."""

    reference: Path
    result_csv: Path | None
    fraction: float


@dataclass(frozen=True)
class TaskSpec:
    name: str
    input_path: Path | None
    sample_num: int
    design_chains: tuple[DesignChainSpec, ...]
    length_plans: tuple[LengthPlan, ...]
    condition: Mapping[str, Any]
    hotspot: str | None
    crop: CropSpec | None = None
    score: bool = False
    cyclic_offset: bool = False
    cyclic_offset_chains: tuple[str, ...] = ()
    postprocess: bool = False
    override: bool = True
    length_distribution_chain: str | None = None
    format: str = "normal"
    antibody: Any | None = None
    rm_token_type: tuple[str, ...] | None = None
    partial: PartialSpec | None = None


@dataclass(frozen=True)
class ExpandedRequest:
    task: TaskSpec
    plan: LengthPlan
    sample_id: str
    task_slot_ordinal: int
    global_ordinal: int
    # Resolved once per base request, before replica expansion; reporting only.
    length_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyCondition:
    atom_uid_a: int
    atom_uid_b: int
    # Public compiler records use an exact order. Internal/legacy carriers may
    # use ``any``/``unknown`` to preserve a positive edge without inventing an
    # exact class in the five-state condition ABI.
    order: str
    required_output: bool

    def canonical_uids(self) -> tuple[int, int]:
        return tuple(sorted((int(self.atom_uid_a), int(self.atom_uid_b))))


@dataclass(frozen=True)
class TokenPairCondition:
    token_uid_a: int
    token_uid_b: int
    value: int

    def canonical_tokens(self) -> tuple[int, int]:
        return tuple(sorted((int(self.token_uid_a), int(self.token_uid_b))))


@dataclass(frozen=True)
class MotifOutputFields:
    """Resolved motif metadata for one final design-polymer chain.

    Both strings are derived while compiling the sampled contig.  They are
    owner-local output metadata, not task YAML, model input, or a distributed
    candidate-selection payload.
    """

    contig_string: str
    tip_atoms: str


@dataclass(frozen=True)
class EdgePolicy:
    """Hard topology facts scoped to one component or molecule.

    ``complete_graph`` makes every unlisted atom pair in the selected scope a
    hard non-edge.  ``hard_positive_edges`` always contains canonical AtomUID
    pairs; bond order remains authoritative in ``TopologyCondition`` and the
    AtomArray bond list rather than being duplicated here. Explicit
    ``hard_negative_edges`` forbid only the listed pairs, leaving all other
    unlisted pairs free unless ``complete_graph`` is set.
    """

    component_uid: int | None = None
    molecule_uid: int | None = None
    complete_graph: bool = False
    hard_positive_edges: tuple[tuple[int, int], ...] = ()
    required_output: bool = False
    # Appended to preserve the positional constructor used by existing callers.
    hard_negative_edges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if (self.component_uid is None) == (self.molecule_uid is None):
            raise ValueError(
                "EdgePolicy requires exactly one of component_uid or molecule_uid"
            )
        for field_name in ("hard_positive_edges", "hard_negative_edges"):
            normalized: set[tuple[int, int]] = set()
            for edge in getattr(self, field_name):
                if len(edge) != 2:
                    raise ValueError(f"{field_name} entries must be AtomUID pairs")
                atom_uid_a, atom_uid_b = sorted((int(edge[0]), int(edge[1])))
                if atom_uid_a == atom_uid_b:
                    raise ValueError(f"{field_name} cannot contain self-edges")
                normalized.add((atom_uid_a, atom_uid_b))
            object.__setattr__(self, field_name, tuple(sorted(normalized)))
        conflict = set(self.hard_positive_edges).intersection(self.hard_negative_edges)
        if conflict:
            raise ValueError(
                f"hard edge policy has positive/negative conflicts: {sorted(conflict)}"
            )


@dataclass(frozen=True, slots=True)
class ModifiedPolymerCarrier:
    """Model-only atomized view of one source-faithful polymer residue.

    Structural identity and topology remain authoritative in ``CompiledSample.atoms``.
    This record carries only the metadata that cannot be derived from that AtomArray.
    Formal charges are a deterministic sparse tuple; an absent AtomUID has charge zero.
    """

    residue_uid: int
    component_id: str
    model_molecule_uid: int
    formal_charge_by_atom_uid: tuple[tuple[int, int], ...]
    provenance: tuple[str, ...]
    report_residue_name: str
    # Optional canonical polymer parent used only for owner-local sequence
    # generator proxy views.  The carrier itself and every structural output
    # retain ``component_id``; this field is never a chemistry fallback.
    standard_parent: str | None = None

    def __post_init__(self) -> None:
        for field_name, raw_value in (
            ("residue_uid", self.residue_uid),
            ("model_molecule_uid", self.model_molecule_uid),
        ):
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, np.integer)):
                raise TypeError(f"{field_name} must be an integer")
            value = int(raw_value)
            if value < 0:
                raise ValueError(f"{field_name} must be nonnegative")
            object.__setattr__(self, field_name, value)

        for field_name, raw_value in (
            ("component_id", self.component_id),
            ("report_residue_name", self.report_residue_name),
        ):
            if (
                not isinstance(raw_value, str)
                or not raw_value
                or raw_value != raw_value.strip()
                or "[" in raw_value
                or "]" in raw_value
            ):
                raise ValueError(
                    f"{field_name} must be a non-empty bare component identifier"
                )

        if self.standard_parent is not None:
            if (
                not isinstance(self.standard_parent, str)
                or not self.standard_parent
                or self.standard_parent != self.standard_parent.strip()
                or "[" in self.standard_parent
                or "]" in self.standard_parent
            ):
                raise ValueError(
                    "standard_parent must be a bare canonical component identifier"
                )

        if isinstance(self.provenance, str):
            raise TypeError("provenance must be a tuple of source labels")
        normalized_provenance: set[str] = set()
        for value in self.provenance:
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("provenance labels must be non-empty stripped strings")
            normalized_provenance.add(value)
        if not normalized_provenance:
            raise ValueError("modified-polymer carrier provenance may not be empty")
        object.__setattr__(self, "provenance", tuple(sorted(normalized_provenance)))

        raw_charges = self.formal_charge_by_atom_uid
        charge_items = (
            raw_charges.items()
            if isinstance(raw_charges, MappingABC)
            else raw_charges
        )
        normalized_charges: dict[int, int] = {}
        for raw_item in charge_items:
            if not isinstance(raw_item, (tuple, list)) or len(raw_item) != 2:
                raise TypeError(
                    "formal_charge_by_atom_uid entries must be (AtomUID, charge) pairs"
                )
            raw_atom_uid, raw_charge = raw_item
            if isinstance(raw_atom_uid, bool) or not isinstance(
                raw_atom_uid, (int, np.integer)
            ):
                raise TypeError("formal-charge AtomUIDs must be integers")
            if isinstance(raw_charge, bool) or not isinstance(
                raw_charge, (int, np.integer)
            ):
                raise TypeError("formal charges must be integers")
            atom_uid, charge = int(raw_atom_uid), int(raw_charge)
            if atom_uid < 0:
                raise ValueError("formal-charge AtomUIDs must be nonnegative")
            if atom_uid in normalized_charges:
                raise ValueError(f"duplicate formal charge for AtomUID {atom_uid}")
            normalized_charges[atom_uid] = charge
        object.__setattr__(
            self,
            "formal_charge_by_atom_uid",
            tuple(sorted(normalized_charges.items())),
        )

    def formal_charge(self, atom_uid: int) -> int:
        """Return the sparse CCD formal charge, defaulting to neutral."""

        target = int(atom_uid)
        for candidate, charge in self.formal_charge_by_atom_uid:
            if candidate == target:
                return charge
        return 0


@dataclass(frozen=True, slots=True)
class ValidatedLigandIdentity:
    """Immutable proof that one ligand graph has one exact CCD identity.

    The record is deliberately coordinate-free.  The atom_signature contains
    the normalized heavy-atom (name, element) inventory and edge_signature
    contains the complete internal heavy-atom graph with concrete bond classes.
    Writers must re-check the record against the current AtomArray before using
    component_id; this makes a stale record downgrade to a file-local LIGxx
    name instead of claiming chemistry it no longer proves.
    """

    component_id: str
    atom_signature: tuple[tuple[str, str], ...]
    edge_signature: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        component_id = str(self.component_id).strip().upper()
        if (
            not component_id
            or component_id in {".", "?"}
            or "[" in component_id
            or "]" in component_id
        ):
            raise ValueError("validated ligand component_id must be a bare CCD ID")
        object.__setattr__(self, "component_id", component_id)

        normalized_atoms: list[tuple[str, str]] = []
        atom_names: set[str] = set()
        for raw in self.atom_signature:
            if not isinstance(raw, (tuple, list)) or len(raw) != 2:
                raise TypeError(
                    "validated ligand atom_signature entries must be (name, element)"
                )
            name, element = (str(value).strip() for value in raw)
            element = element.upper()
            if (
                not name
                or name in {".", "?"}
                or not element
                or element in {".", "?"}
            ):
                raise ValueError("validated ligand atom signatures must be non-empty")
            if name in atom_names:
                raise ValueError("validated ligand atom names must be unique")
            atom_names.add(name)
            normalized_atoms.append((name, element))
        if not normalized_atoms:
            raise ValueError("validated ligand identity requires heavy atoms")
        normalized_atoms.sort()
        object.__setattr__(self, "atom_signature", tuple(normalized_atoms))

        order_names = {
            "1": "single",
            "single": "single",
            "sing": "single",
            "2": "double",
            "double": "double",
            "doub": "double",
            "3": "triple",
            "triple": "triple",
            "trip": "triple",
            "arom": "aromatic",
            "aromatic": "aromatic",
        }
        normalized_edges: set[tuple[str, str, str]] = set()
        for raw in self.edge_signature:
            if not isinstance(raw, (tuple, list)) or len(raw) != 3:
                raise TypeError(
                    "validated ligand edge_signature entries must be (a, b, order)"
                )
            left, right = sorted((str(raw[0]).strip(), str(raw[1]).strip()))
            order = order_names.get(str(raw[2]).strip().lower())
            if (
                not left
                or not right
                or left == right
                or left not in atom_names
                or right not in atom_names
                or order is None
            ):
                raise ValueError("validated ligand edge signature is invalid")
            normalized_edges.add((left, right, order))
        object.__setattr__(self, "edge_signature", tuple(sorted(normalized_edges)))


@dataclass
class CompiledSample:
    request: ExpandedRequest
    atoms: AtomArray
    source_to_model_affine: np.ndarray
    model_to_source_affine: np.ndarray
    topology_conditions: tuple[TopologyCondition, ...] = ()
    edge_policies: tuple[EdgePolicy, ...] = ()
    model_parent_residue: dict[int, str] = field(default_factory=dict)
    modified_polymer_carriers: Mapping[int, ModifiedPolymerCarrier] = field(
        default_factory=dict,
        repr=False,
    )
    coarse_ss: dict[int, int] = field(default_factory=dict)
    ss_token: dict[int, int] = field(default_factory=dict)
    ss_pair: tuple[TokenPairCondition, ...] = ()
    motif_output_fields: Mapping[str, MotifOutputFields] = field(default_factory=dict)
    # Owner-local private chemistry matched to the complete installed CCD
    # precursor, or its explicitly declared atom-omission product fragment.
    trusted_ligand_components: Mapping[int, Any] = field(default_factory=dict, repr=False)
    # Coordinate-free proof for source/design ligand component IDs.  This is
    # intentionally separate from the private RDKit chemistry sidecar above:
    # output naming may use this record without importing RDKit.
    validated_ligand_identities: Mapping[int, ValidatedLigandIdentity] = field(
        default_factory=dict,
        repr=False,
    )
    # Compiler-side polymer atom omissions, keyed by the final canonical token UID.
    # Values are sorted atom-name tuples so assembly can apply the same
    # per-token exclusion after decoding without changing the batch ABI.
    omitted_atom_names_by_token_uid: Mapping[int, tuple[str, ...]] = field(
        default_factory=dict,
        repr=False,
    )
    # Authoring token UIDs ending a retained context segment after crop.
    # Empty for crop=None, preserving the canonical tensor path exactly.
    forced_chain_break_token_uids: tuple[int, ...] = ()
    antibody_metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    # Same authoring UID/inventory axis as atoms; current candidate identities and
    # complete geometry are kept separate from original task conditions.
    reference_atoms: AtomArray | None = field(default=None, repr=False)
    partial_fraction: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("source_to_model_affine", self.source_to_model_affine),
            ("model_to_source_affine", self.model_to_source_affine),
        ):
            array = np.asarray(value)
            if array.shape != (4, 4) or array.dtype != np.float64:
                raise TypeError(f"{name} must be float64[4,4]")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")

        if not isinstance(self.validated_ligand_identities, MappingABC):
            raise TypeError("validated_ligand_identities must be a mapping")
        normalized_identities: dict[int, ValidatedLigandIdentity] = {}
        component_uids = {
            int(value)
            for value in np.asarray(self.atoms.component_uid, dtype=np.int64)
        }
        for raw_uid, identity in self.validated_ligand_identities.items():
            if isinstance(raw_uid, bool) or not isinstance(raw_uid, (int, np.integer)):
                raise TypeError(
                    "validated ligand identity keys must be component UID integers"
                )
            component_uid = int(raw_uid)
            if component_uid < 0 or component_uid not in component_uids:
                raise ValueError(
                    f"validated ligand identity references absent component UID {component_uid}"
                )
            if not isinstance(identity, ValidatedLigandIdentity):
                raise TypeError(
                    "validated_ligand_identities values must be ValidatedLigandIdentity"
                )
            rows = np.flatnonzero(
                np.asarray(self.atoms.component_uid, dtype=np.int64) == component_uid
            )
            modalities = {str(value) for value in self.atoms.mol_type[rows]}
            if modalities != {"ligand"}:
                raise ValueError(
                    "validated ligand identity must reference ligand rows only"
                )
            names = {str(value).strip().upper() for value in self.atoms.res_name[rows]}
            if names != {identity.component_id}:
                raise ValueError(
                    "validated ligand identity component_id disagrees with AtomArray res_name"
                )
            if "identity_condition" not in self.atoms.get_annotation_categories():
                raise ValueError(
                    "validated ligand identity requires identity_condition annotation"
                )
            if not bool(np.asarray(self.atoms.identity_condition[rows], dtype=bool).all()):
                raise ValueError(
                    "validated ligand identity requires identity_condition=True"
                )
            normalized_identities[component_uid] = identity
        self.validated_ligand_identities = MappingProxyType(
            dict(sorted(normalized_identities.items()))
        )

        if not isinstance(self.modified_polymer_carriers, MappingABC):
            raise TypeError("modified_polymer_carriers must be a mapping")
        if not self.modified_polymer_carriers:
            self.modified_polymer_carriers = MappingProxyType({})
            return
        proxy_residue_uids = {int(value) for value in self.model_parent_residue}
        source_molecule_uids = {
            int(value) for value in np.asarray(self.atoms.molecule_uid, dtype=np.int64)
        }
        normalized_carriers: dict[int, ModifiedPolymerCarrier] = {}
        model_molecule_uids: set[int] = set()
        for raw_residue_uid, carrier in self.modified_polymer_carriers.items():
            if isinstance(raw_residue_uid, bool) or not isinstance(
                raw_residue_uid, (int, np.integer)
            ):
                raise TypeError("modified-polymer carrier keys must be ResidueUID integers")
            residue_uid = int(raw_residue_uid)
            if not isinstance(carrier, ModifiedPolymerCarrier):
                raise TypeError(
                    "modified_polymer_carriers values must be ModifiedPolymerCarrier"
                )
            if carrier.residue_uid != residue_uid:
                raise ValueError(
                    "modified-polymer carrier key disagrees with its residue_uid"
                )
            if residue_uid in proxy_residue_uids:
                raise ValueError(
                    f"modified polymer residue UID {residue_uid} has both carrier and proxy"
                )
            if carrier.model_molecule_uid in source_molecule_uids:
                raise ValueError(
                    "carrier model_molecule_uid collides with a source molecule_uid"
                )
            if carrier.model_molecule_uid in model_molecule_uids:
                raise ValueError("carrier model_molecule_uid values must be unique")
            model_molecule_uids.add(carrier.model_molecule_uid)

            rows = np.where(
                np.asarray(self.atoms.residue_uid, dtype=np.int64) == residue_uid
            )[0]
            if len(rows) == 0:
                raise ValueError(
                    f"modified-polymer carrier residue UID {residue_uid} is absent"
                )
            source_modalities = {str(value) for value in self.atoms.mol_type[rows]}
            if len(source_modalities) != 1 or not source_modalities <= {
                "protein",
                "rna",
                "dna",
            }:
                raise ValueError(
                    "modified-polymer carrier source modality must remain polymer"
                )
            source_modality = next(iter(source_modalities))
            if (
                carrier.standard_parent is not None
                and carrier.standard_parent
                not in _STANDARD_PARENT_BY_MODALITY[source_modality]
            ):
                raise ValueError(
                    "modified-polymer carrier standard_parent changes modality "
                    "or is noncanonical"
                )
            for annotation in (
                "chain_id",
                "res_id",
                "ins_code",
                "res_name",
                "mol_type",
                "component_uid",
                "molecule_uid",
                "token_uid",
            ):
                values = np.unique(self.atoms.get_annotation(annotation)[rows])
                if len(values) != 1:
                    raise ValueError(
                        "modified-polymer carrier source residue is not uniquely aggregated: "
                        f"{annotation}"
                    )
            token_uid = int(self.atoms.token_uid[rows[0]])
            if token_uid < 0:
                raise ValueError("modified-polymer carrier requires authoring tokenization")
            token_rows = np.where(
                np.asarray(self.atoms.token_uid, dtype=np.int64) == token_uid
            )[0]
            if not np.array_equal(np.sort(token_rows), np.sort(rows)):
                raise ValueError(
                    "modified-polymer carrier must occupy exactly one authoring token"
                )
            source_residue_name = str(self.atoms.res_name[rows[0]])
            if source_residue_name != carrier.component_id:
                raise ValueError(
                    "carrier component_id disagrees with source residue identity"
                )
            if source_residue_name != carrier.report_residue_name:
                raise ValueError(
                    "carrier report_residue_name disagrees with source residue identity"
                )

            atom_names = [str(value) for value in self.atoms.atom_name[rows]]
            if any(not value for value in atom_names) or len(atom_names) != len(
                set(atom_names)
            ):
                raise ValueError(
                    "modified-polymer carrier atom names must be non-empty and unique"
                )
            elements = [str(value).strip().upper() for value in self.atoms.element[rows]]
            if any(not value for value in elements):
                raise ValueError("modified-polymer carrier elements must be non-empty")
            if not any(value not in {"H", "D", "T"} for value in elements):
                raise ValueError(
                    "modified-polymer carrier requires at least one heavy atom"
                )

            atom_uids = [int(value) for value in self.atoms.atom_uid[rows]]
            if len(atom_uids) != len(set(atom_uids)) or any(value < 0 for value in atom_uids):
                raise ValueError(
                    "modified-polymer carrier AtomUIDs must be unique and finalized"
                )
            all_atom_uids = np.asarray(self.atoms.atom_uid, dtype=np.int64)
            for atom_uid in atom_uids:
                if int(np.count_nonzero(all_atom_uids == atom_uid)) != 1:
                    raise ValueError(
                        f"modified-polymer carrier AtomUID {atom_uid} is ambiguous"
                    )
            unknown_charge_uids = {
                atom_uid for atom_uid, _ in carrier.formal_charge_by_atom_uid
            }.difference(atom_uids)
            if unknown_charge_uids:
                raise ValueError(
                    "modified-polymer formal charge references an AtomUID outside its "
                    f"source residue: {sorted(unknown_charge_uids)}"
                )
            normalized_carriers[residue_uid] = carrier

        self.modified_polymer_carriers = MappingProxyType(
            dict(sorted(normalized_carriers.items()))
        )

    def __getstate__(self) -> dict[str, Any]:
        """Keep the immutable carrier map compatible with DataLoader workers."""

        state = dict(self.__dict__)
        state["modified_polymer_carriers"] = dict(self.modified_polymer_carriers)
        state["validated_ligand_identities"] = dict(self.validated_ligand_identities)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(state)
        if "validated_ligand_identities" not in self.__dict__:
            self.validated_ligand_identities = {}
        if "antibody_metadata" not in self.__dict__:
            self.antibody_metadata = {}
        self.__dict__.setdefault("reference_atoms", None)
        self.__dict__.setdefault("partial_fraction", None)
        self.__post_init__()


@dataclass
class SearchCandidate:
    candidate_id: str
    candidate_ordinal: int
    base_request: ExpandedRequest
    actual_joint_length: tuple[int, ...]
    atoms: AtomArray
    topology_conditions: tuple[TopologyCondition, ...] = ()
    # Owner-local private chemistry carrier for an explicitly requested later
    # postprocess.  These fields never enter CIF/PDB or distributed summaries.
    private_ligand_molecules: Mapping[int, Any] = field(default_factory=dict, repr=False)
    private_ligand_errors: Mapping[int, str] = field(default_factory=dict, repr=False)
    # Molecule UID -> original/retained inventory and fragment-selection audit.
    # No model tensors or private chemistry objects belong in this ledger.
    ligand_fragment_metadata: Mapping[int, Mapping[str, Any]] = field(
        default_factory=dict, repr=False,
    )
    motif_output_fields: Mapping[str, MotifOutputFields] = field(
        default_factory=dict,
        repr=False,
    )
    # ResidueUID -> model-visible standard/unknown proxy.  Final atoms keep the
    # source residue name; this sidecar is used only where a one-letter model
    # identity is required (for example result.csv sequence reporting).
    model_parent_residue: Mapping[int, str] = field(default_factory=dict, repr=False)
    # ResidueUID -> bare CCD component ID for a source-faithful modified-polymer
    # carrier.  This is reporting metadata only: the candidate AtomArray remains
    # on the residue-level authoring token axis.
    fixed_modified_residue_names: Mapping[int, str] = field(
        default_factory=dict,
        repr=False,
    )
    # ResidueUID -> unambiguous canonical parent for a fixed modified carrier.
    # This is a private MPNN proxy ledger, not an alternate output identity.
    fixed_modified_parent_residue: Mapping[int, str] = field(
        default_factory=dict,
        repr=False,
    )
    # Coordinate-free CCD identity proofs propagated through search and final
    # output.  The writer revalidates the proof against the current atoms so a
    # fold-output signature drift safely downgrades to LIGxx.
    validated_ligand_identities: Mapping[int, ValidatedLigandIdentity] = field(
        default_factory=dict,
        repr=False,
    )

    antibody_metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    partial_metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("antibody_metadata", {})
        self.__dict__.setdefault("partial_metadata", {})

    def __post_init__(self) -> None:
        if not isinstance(self.fixed_modified_residue_names, MappingABC):
            raise TypeError("fixed_modified_residue_names must be a mapping")
        proxy_residue_uids = {int(value) for value in self.model_parent_residue}
        normalized: dict[int, str] = {}
        modality_by_residue_uid: dict[int, str] = {}
        for raw_residue_uid, raw_component_id in self.fixed_modified_residue_names.items():
            if isinstance(raw_residue_uid, bool) or not isinstance(
                raw_residue_uid, (int, np.integer)
            ):
                raise TypeError("fixed-modified report keys must be ResidueUID integers")
            residue_uid = int(raw_residue_uid)
            if residue_uid < 0:
                raise ValueError("fixed-modified report ResidueUIDs must be nonnegative")
            if (
                not isinstance(raw_component_id, str)
                or not raw_component_id
                or raw_component_id != raw_component_id.strip()
                or "[" in raw_component_id
                or "]" in raw_component_id
            ):
                raise ValueError(
                    "fixed-modified report identities must be bare CCD component IDs"
                )
            if residue_uid in proxy_residue_uids:
                raise ValueError(
                    f"modified polymer residue UID {residue_uid} has both report carrier and proxy"
                )
            rows = np.flatnonzero(
                np.asarray(self.atoms.residue_uid, dtype=np.int64) == residue_uid
            )
            if len(rows) == 0:
                raise ValueError(
                    f"fixed-modified report residue UID {residue_uid} is absent"
                )
            for annotation in (
                "chain_id",
                "res_id",
                "ins_code",
                "res_name",
                "mol_type",
                "token_uid",
            ):
                if len(np.unique(self.atoms.get_annotation(annotation)[rows])) != 1:
                    raise ValueError(
                        "fixed-modified report residue is not uniquely aggregated: "
                        f"{annotation}"
                    )
            modality = str(self.atoms.mol_type[rows[0]])
            if modality not in {"protein", "rna", "dna"}:
                raise ValueError("fixed-modified report residue must remain polymer")
            modality_by_residue_uid[residue_uid] = modality
            source_residue_name = str(self.atoms.res_name[rows[0]])
            if source_residue_name != raw_component_id:
                raise ValueError(
                    "fixed-modified report identity disagrees with candidate source res_name"
                )
            token_uid = int(self.atoms.token_uid[rows[0]])
            if token_uid < 0:
                raise ValueError("fixed-modified report residue requires authoring tokenization")
            token_rows = np.flatnonzero(
                np.asarray(self.atoms.token_uid, dtype=np.int64) == token_uid
            )
            if not np.array_equal(np.sort(token_rows), np.sort(rows)):
                raise ValueError(
                    "fixed-modified report residue must occupy exactly one authoring token"
                )
            design_rows = rows[np.asarray(self.atoms.is_design[rows], dtype=bool)]
            if len(design_rows) and not bool(
                np.asarray(
                    self.atoms.identity_condition[design_rows], dtype=bool
                ).all()
            ):
                raise ValueError(
                    "fixed modified design residue must be hard identity-fixed"
                )
            normalized[residue_uid] = raw_component_id
        self.fixed_modified_residue_names = dict(sorted(normalized.items()))
        if not isinstance(self.fixed_modified_parent_residue, MappingABC):
            raise TypeError("fixed_modified_parent_residue must be a mapping")
        normalized_parents: dict[int, str] = {}
        for raw_residue_uid, raw_parent in self.fixed_modified_parent_residue.items():
            if isinstance(raw_residue_uid, bool) or not isinstance(
                raw_residue_uid, (int, np.integer)
            ):
                raise TypeError(
                    "fixed-modified parent keys must be ResidueUID integers"
                )
            residue_uid = int(raw_residue_uid)
            if residue_uid not in normalized:
                raise ValueError(
                    "fixed-modified parent must reference a fixed modified residue"
                )
            if (
                not isinstance(raw_parent, str)
                or not raw_parent
                or raw_parent != raw_parent.strip()
                or "[" in raw_parent
                or "]" in raw_parent
            ):
                raise ValueError(
                    "fixed-modified parent identities must be bare component IDs"
                )
            modality = modality_by_residue_uid[residue_uid]
            if raw_parent not in _STANDARD_PARENT_BY_MODALITY[modality]:
                raise ValueError(
                    "fixed-modified parent changes modality or is noncanonical"
                )
            normalized_parents[residue_uid] = raw_parent
        self.fixed_modified_parent_residue = dict(sorted(normalized_parents.items()))
        if not isinstance(self.validated_ligand_identities, MappingABC):
            raise TypeError("validated_ligand_identities must be a mapping")
        component_uids = {
            int(value)
            for value in np.asarray(self.atoms.component_uid, dtype=np.int64)
        }
        normalized_identities: dict[int, ValidatedLigandIdentity] = {}
        for raw_uid, identity in self.validated_ligand_identities.items():
            if isinstance(raw_uid, bool) or not isinstance(raw_uid, (int, np.integer)):
                raise TypeError(
                    "validated ligand identity keys must be component UID integers"
                )
            component_uid = int(raw_uid)
            if component_uid < 0 or component_uid not in component_uids:
                raise ValueError(
                    f"validated ligand identity references absent component UID {component_uid}"
                )
            if not isinstance(identity, ValidatedLigandIdentity):
                raise TypeError(
                    "validated_ligand_identities values must be ValidatedLigandIdentity"
                )
            rows = np.flatnonzero(
                np.asarray(self.atoms.component_uid, dtype=np.int64) == component_uid
            )
            if {str(value) for value in self.atoms.mol_type[rows]} != {"ligand"}:
                raise ValueError(
                    "validated ligand identity must reference ligand rows only"
                )
            names = {str(value).strip().upper() for value in self.atoms.res_name[rows]}
            if names != {identity.component_id}:
                raise ValueError(
                    "validated ligand identity component_id disagrees with AtomArray res_name"
                )
            normalized_identities[component_uid] = identity
        self.validated_ligand_identities = dict(sorted(normalized_identities.items()))


@dataclass(frozen=True)
class SampleFailure:
    task_name: str
    sample_id: str
    stage: str
    message: str
