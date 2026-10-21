"""Source-preserving all-modality structure preparation (coordinates in nm)."""
from dataclasses import dataclass, field
from .records import ScoreInputError
import numpy as np
import torch
from biotite import structure as struc
from .structure import AddAtomArrayAnnot as Annot, AtomArrayTokenizer
from src.scoring.ccd import canonical_residue_names
from .projection import canonicalize_modified_polymer_carrier
from .canonical import build_canonical

SCORE_FIELDS = ("token_mask", "source_mol_id", "asym_id", "residue_index", "mol_type_id",
                "token_type32", "token_center_coords_nm", "token_bond_type",
                "token_bond_pair_mask", "ligand_mask", "ligand_element_type",
                "polymer_atom91_coords", "polymer_atom91_mask",
                "_partiscore_design_group_id",
                "_partiscore_covalent_bond_pair_mask")

@dataclass(frozen=True)
class PreparedStructure:
    canonical: dict
    partner_labels: dict[int, str]
    design_source_ids: frozenset[int]
    # One token-local connected-design-group ID per canonical token; -1 is
    # context.  The tuple is intentionally ordinary Python data so it remains
    # cheap and explicit across the CPU preparation process boundary.
    design_group_ids: tuple[int, ...] = ()
    design_group_labels: dict[int, str] = field(default_factory=dict)


def _source_atoms(atoms):
    atoms = atoms.copy()
    if not len(atoms):
        raise ScoreInputError("structure has no atoms")
    mol = np.asarray(atoms.mol_type)
    if not set(mol) <= {"protein", "rna", "dna", "ligand"}:
        raise ScoreInputError("structure contains an unsupported modality")
    author = np.asarray(atoms.chain_id).copy()
    prior = None
    if hasattr(atoms, "source_mol_id"):
        prior = np.asarray(atoms.source_mol_id)
        if prior.shape != (len(atoms),) or not np.issubdtype(prior.dtype, np.integer) or (prior < 0).any():
            raise ScoreInputError("source_mol_id must be a nonnegative integer per source atom")
        prior = prior.copy()
    # Private chains prevent independent ligands sharing an author chain from
    # collapsing in source molecule assignment. Polymer TER boundaries remain.
    asym = np.asarray(getattr(atoms, "label_asym_id", author))
    comp = np.asarray(getattr(atoms, "component_uid", atoms.res_id))
    keys = [("ligand", str(asym[i]), str(comp[i])) if mol[i] == "ligand"
            else ("polymer", str(asym[i])) for i in range(len(atoms))]
    ids = {key: n for n, key in enumerate(dict.fromkeys(keys))}
    atoms.set_annotation("chain_id", np.asarray([f"S{ids[key]}" for key in keys], dtype="U16"))
    if not hasattr(atoms, "label_entity_id"):
        atoms.set_annotation("label_entity_id", asym.astype("U128"))
    atoms.set_annotation("is_resolved", np.ones(len(atoms), dtype=bool))
    if atoms.bonds is None:
        atoms.bonds = struc.BondList(len(atoms))
    Annot.unique_chain_and_add_ids(atoms)
    types = {"protein": "polypeptide(L)", "rna": "polyribonucleotide", "dna": "polydeoxyribonucleotide"}
    entity_types = {}
    for entity, modality in zip(atoms.label_entity_id, mol):
        if modality in types:
            entity_types[str(entity)] = types[modality]
    Annot.find_equiv_mol_and_assign_ids(atoms, entity_types or None)
    if prior is not None:
        atoms.set_annotation("mol_id", prior)
    atoms.set_annotation("source_mol_id", np.asarray(atoms.mol_id, dtype=np.int64).copy())
    return atoms, author


def _selected_design_mask(atoms, author, design_chains):
    selected = {str(value) for value in (design_chains or ())}
    author_values = np.asarray(author).astype(str)
    label_values = np.asarray(getattr(atoms, "label_asym_id", author)).astype(str)
    label_namespace = set(label_values.tolist())
    label_selected = selected & label_namespace
    author_selected = selected - label_namespace
    return np.isin(label_values, list(label_selected)) | np.isin(author_values, list(author_selected))


def _split_design_context_sources(atoms, author, design_chains):
    """Split mixed source IDs at the design/context ownership boundary.

    The source molecule assignment is retained everywhere else.  A source
    molecule that crosses the requested design/context boundary must be split
    before canonical feature construction so the network can represent that
    interface as an inter-molecule pair.  If one prior source ID contains
    multiple independent design components attached to the same context, those
    design components retain distinct source IDs as well; context connectivity
    is never allowed to bridge them.  Coordinates, atom rows, and the bond
    list are untouched.
    """

    if not design_chains:
        return atoms
    source = np.asarray(atoms.source_mol_id, dtype=np.int64).copy()
    design_mask = _selected_design_mask(atoms, author, design_chains)
    next_id = int(source.max(initial=-1)) + 1
    # Compute design-side components before changing source IDs.  String source
    # labels are sufficient here; the actual labels are rebuilt afterwards.
    atom_groups, _ = _design_group_metadata(
        atoms,
        author,
        design_chains,
        {int(value): str(int(value)) for value in np.unique(source)},
    )
    changed = False
    for source_id in np.unique(source):
        members = source == int(source_id)
        design_groups = sorted(set(int(value) for value in atom_groups[members & design_mask]))
        mixed = bool((members & design_mask).any()) and bool((members & ~design_mask).any())
        if mixed or len(design_groups) > 1:
            # Keep the original ID for the first design group and allocate
            # private IDs for each additional group in this source.
            for group_id in design_groups[1:]:
                source[members & design_mask & (atom_groups == group_id)] = next_id
                next_id += 1
            if mixed:
                source[members & ~design_mask] = next_id
                next_id += 1
            changed = True
    if changed:
        # The canonical compiler reads mol_id as the model-facing molecule
        # assignment and later exposes it as source_mol_id.
        atoms.set_annotation("mol_id", source)
        atoms.set_annotation("source_mol_id", source)
    return atoms


def _design_group_metadata(atoms, author, design_chains, labels):
    """Build covalent design groups and their token-facing labels.

    Design units start from component ownership (polymer source/asym units or
    ligand components), then only true BondList edges whose two endpoints are
    on the design side can merge units.  Context atoms are never traversed, so
    a receptor cannot bridge independent designs.
    """

    if isinstance(design_chains, str):
        raise ScoreInputError("design_chains must be a sequence of chain IDs")
    n = len(atoms)
    design_mask = _selected_design_mask(atoms, author, design_chains)
    atom_groups = np.full(n, -1, dtype=np.int64)
    if not bool(design_mask.any()):
        return atom_groups, {}

    modality = np.asarray(atoms.mol_type)
    asym = np.asarray(getattr(atoms, "label_asym_id", author))
    component = np.asarray(getattr(atoms, "component_uid", atoms.res_id))
    source = np.asarray(atoms.source_mol_id, dtype=np.int64)

    # Component-aware units keep same-author ligand/protein instances distinct
    # while retaining one unit for an ordinary polymer source.  True bonds can
    # then merge units across a ligand/polymer boundary.
    unit_ids: dict[tuple[object, ...], int] = {}
    atom_units = np.full(n, -1, dtype=np.int64)
    for atom_index in np.flatnonzero(design_mask):
        atom_index = int(atom_index)
        if modality[atom_index] == "ligand":
            key = ("ligand", str(asym[atom_index]), str(component[atom_index]), int(source[atom_index]))
        else:
            key = ("polymer", str(modality[atom_index]), str(asym[atom_index]), int(source[atom_index]))
        unit = unit_ids.setdefault(key, len(unit_ids))
        atom_units[atom_index] = unit

    parent = list(range(len(unit_ids)))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    if atoms.bonds is not None:
        coordination = int(struc.BondType.COORDINATION)
        for atom_i, atom_j, bond_type in atoms.bonds.as_array():
            atom_i, atom_j = int(atom_i), int(atom_j)
            if (design_mask[atom_i] and design_mask[atom_j]
                    and int(bond_type) != coordination):
                union(int(atom_units[atom_i]), int(atom_units[atom_j]))

    root_to_group: dict[int, int] = {}
    for atom_index in np.flatnonzero(design_mask):
        atom_index = int(atom_index)
        root = find(int(atom_units[atom_index]))
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        atom_groups[atom_index] = root_to_group[root]

    group_sources: dict[int, set[int]] = {}
    for atom_index in np.flatnonzero(design_mask):
        atom_index = int(atom_index)
        group_sources.setdefault(int(atom_groups[atom_index]), set()).add(int(source[atom_index]))
    group_labels: dict[int, str] = {}
    used_labels: set[str] = set()
    for group_id in sorted(group_sources):
        base = "+".join(labels[source_id] for source_id in sorted(group_sources[group_id]))
        label = base or f"design_group_{group_id}"
        if label in used_labels:
            label = f"{label}@group{group_id}"
        used_labels.add(label)
        group_labels[group_id] = label
    return atom_groups, group_labels


def _partners(atoms, author, design_chains, *, scope=None):
    source = np.asarray(atoms.source_mol_id)
    labels = {int(s): "+".join(dict.fromkeys(str(x) for x in author[source == s]))
              for s in np.unique(source)}
    duplicates = {label for label in labels.values() if list(labels.values()).count(label) > 1}
    labels = {s: f"{label}#{s}" if label in duplicates else label for s, label in labels.items()}
    if isinstance(design_chains, str):
        raise ScoreInputError("design_chains must be a sequence of chain IDs")
    if scope == "design_groups":
        selected = {str(value) for value in (design_chains or ())}
        author_values = np.asarray(author).astype(str)
        label_values = np.asarray(getattr(atoms, "label_asym_id", author)).astype(str)
        missing = selected - (set(author_values.tolist()) | set(label_values.tolist()))
        if missing:
            raise ScoreInputError(f"unknown design chains: {sorted(missing)}")
        design_mask = _selected_design_mask(atoms, author, design_chains)
    else:
        selected = set(design_chains or ())
        missing = selected - set(author.tolist())
        if missing:
            raise ScoreInputError(f"unknown design chains: {sorted(missing)}")
        design_mask = np.isin(author, list(selected))
    design = frozenset(int(s) for s in source[design_mask])
    return labels, design


def source_partners(atoms, design_chains=None, *, scope=None):
    """Inspect source ownership without requiring resolved design coordinates."""
    source, author = _source_atoms(atoms)
    if scope == "design_groups":
        source = _split_design_context_sources(source, author, design_chains)
    return _partners(source, author, design_chains, scope=scope)


def prepare_structure(atoms, identity="structure", design_chains=None, *, scope=None):
    """Build canonical scoring fields without crop, augmentation, or VAE."""
    atoms, author = _source_atoms(atoms)
    if scope == "design_groups":
        atoms = _split_design_context_sources(atoms, author, design_chains)
    labels, design = _partners(atoms, author, design_chains, scope=scope)
    atom_groups, group_labels = _design_group_metadata(atoms, author, design_chains, labels)
    if not np.isfinite(atoms.coord).all():
        raise ScoreInputError("structure coordinates must be finite")
    if np.isin(np.char.upper(atoms.element), ["H", "D", "T"]).any():
        raise ScoreInputError("score_structures expects reader-normalized heavy atoms")
    n = len(atoms)
    ligand = np.asarray(atoms.mol_type) == "ligand"
    atoms.set_annotation("source_atom_index", np.arange(n, dtype=np.int64))
    atoms.set_annotation("source_entity_id", np.asarray(atoms.label_entity_id, dtype="U128"))
    atoms.set_annotation("source_ligand_component_id", np.where(ligand, getattr(atoms, "component_uid", atoms.mol_id), -1).astype(np.int64))
    for name in ("was_modified_polymer", "modified_res_mask", "modified_polymer_carrier_mask", "modified_polymer_proxy_mask"):
        atoms.set_annotation(name, np.zeros(n, dtype=bool))
    # The controlled reader has already resolved all four modalities. Do not
    # overwrite RNA/DNA with the PPI benchmark protein-only annotation helper.
    Annot.add_centre_atom_mask(atoms)
    atoms.set_annotation("cano_seq_resname", canonical_residue_names(atoms))
    Annot.add_modified_res_mask(atoms)
    Annot.add_tokatom_idx(atoms)
    if not hasattr(atoms, "charge"):
        atoms.set_annotation("charge", np.zeros(n, dtype=np.int32))
    Annot.add_ref_space_uid(atoms)
    tokens = AtomArrayTokenizer(atoms).get_token_array()
    coords, sources = atoms.coord.copy(), atoms.source_mol_id.copy()
    bonds = atoms.bonds.as_array().copy()
    normalized = bonds.copy()
    modified = atoms.modified_res_mask
    for edge in normalized:
        if modified[int(edge[0])] and modified[int(edge[1])]:
            if edge[2] == struc.BondType.AROMATIC_SINGLE:
                edge[2] = struc.BondType.SINGLE
            elif edge[2] == struc.BondType.AROMATIC_DOUBLE:
                edge[2] = struc.BondType.DOUBLE
    atoms.bonds = struc.BondList(n, normalized)
    tokens, atoms, _, projection = canonicalize_modified_polymer_carrier(tokens, atoms)
    if (len(atoms) != n or not np.array_equal(atoms.source_atom_index, np.arange(n))
            or not np.array_equal(atoms.coord, coords) or not np.array_equal(atoms.source_mol_id, sources)):
        raise ScoreInputError(f"modified-polymer conversion changed source atom coverage: {projection}")
    original = {tuple(sorted((int(i), int(j)))): int(order) for i, j, order in bonds}
    restored = atoms.bonds.as_array().copy()
    if not original.keys() <= {tuple(sorted((int(i), int(j)))) for i, j, _ in restored}:
        raise ScoreInputError("modified-polymer conversion dropped source bonds")
    for edge in restored:
        key = tuple(sorted((int(edge[0]), int(edge[1]))))
        if key in original:
            edge[2] = original[key]
    atoms.bonds = struc.BondList(n, restored)
    token_centres = np.asarray(tokens.get_annotation("centre_atom_index"), dtype=np.int64)
    if token_centres.shape != (len(tokens),) or (token_centres < 0).any() or (token_centres >= n).any():
        raise ScoreInputError("token centres do not map to source atoms")
    token_groups = torch.as_tensor(atom_groups[token_centres], dtype=torch.long)
    atom_to_token = np.full(n, -1, dtype=np.int64)
    for token_index, token in enumerate(tokens):
        atom_indices = np.asarray(token.atom_indices, dtype=np.int64)
        atom_to_token[atom_indices] = token_index
    if (atom_to_token < 0).any():
        raise ScoreInputError("canonical tokens do not cover source atoms")
    covalent_bonds = torch.zeros((len(tokens), len(tokens)), dtype=torch.bool)
    coordination = int(struc.BondType.COORDINATION)
    for atom_i, atom_j, bond_type in restored:
        if int(bond_type) == coordination:
            continue
        token_i, token_j = int(atom_to_token[int(atom_i)]), int(atom_to_token[int(atom_j)])
        if token_i != token_j:
            covalent_bonds[token_i, token_j] = True
            covalent_bonds[token_j, token_i] = True
    canonical = build_canonical(atoms, tokens, str(identity))
    canonical["_partiscore_design_group_id"] = token_groups
    canonical["_partiscore_covalent_bond_pair_mask"] = covalent_bonds
    represented = int(canonical["polymer_atom91_mask"].sum() + canonical["ligand_mask"].sum())
    if represented != n:
        raise ScoreInputError(f"canonical scoring geometry represents {represented} of {n} source atoms")
    return PreparedStructure(canonical, labels, design,
                             tuple(int(value) for value in token_groups.tolist()),
                             group_labels)
