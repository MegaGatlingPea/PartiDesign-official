"""Coordinate-free AF3 ligand graph and CCD materialization helpers.

The AF3 input adapter owns source-row identity.  This module only turns an
already-authoritative graph or trusted RDKit molecule into a named user CCD;
it never reads source coordinates and never infers missing graph edges.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import numpy as np


class ChemistryError(ValueError):
  """Raised when an AF3 ligand graph cannot be represented exactly."""


@dataclass(frozen=True, slots=True)
class AtomGraph:
  """Heavy-atom graph in source-row or molecule atom order."""

  names: tuple[str, ...]
  elements: tuple[str, ...]
  charges: tuple[int | None, ...]
  bonds: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class MaterializedMolecule:
  """A trusted molecule with names assigned in its own atom order."""

  molecule: Any
  atom_names: tuple[str, ...]


def _rdkit() -> tuple[Any, Any]:
  try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
  except Exception as exc:  # pragma: no cover - dependency is environment-owned
    raise ChemistryError("RDKit is required for AF3 ligand materialization") from exc
  return Chem, AllChem


def _element(raw: Any) -> str:
  Chem, _ = _rdkit()
  text = str(raw).strip()
  if not text or text in {".", "?"} or any(char.isspace() for char in text):
    raise ChemistryError("ligand graph has an unknown atom element")
  symbol = text[:1].upper() + text[1:].lower()
  try:
    atomic_number = int(Chem.GetPeriodicTable().GetAtomicNumber(symbol))
  except (RuntimeError, ValueError) as exc:
    raise ChemistryError(f"ligand graph has unsupported atom element {raw!r}") from exc
  if atomic_number <= 1:
    raise ChemistryError(f"ligand graph has unsupported atom element {raw!r}")
  return Chem.GetPeriodicTable().GetElementSymbol(atomic_number).upper()


def _charge_values(atoms: Any, rows: Sequence[int] | None = None) -> np.ndarray | None:
  from ...common.chemistry.charges import formal_charge_values

  try:
    return formal_charge_values(atoms, rows=rows)
  except ValueError as exc:
    raise ChemistryError(str(exc)) from exc


def _charge(values: np.ndarray | None, row: int) -> int | None:
  if values is None:
    return None
  raw = values[row]
  if raw is None or str(raw).strip().lower() in {"", ".", "?", "none", "nan"}:
    return None
  try:
    value = float(raw)
  except (TypeError, ValueError) as exc:
    raise ChemistryError(f"ligand graph has invalid atom charge {raw!r}") from exc
  if not np.isfinite(value) or not value.is_integer():
    raise ChemistryError(f"ligand graph has non-integral atom charge {raw!r}")
  return int(value)


def _bond_info(raw_order: Any) -> tuple[int, str, str, Any]:
  """Return canonical order, CCD value_order/aromatic flag, and RDKit type."""
  Chem, _ = _rdkit()
  try:
    order = int(raw_order)
  except (TypeError, ValueError) as exc:
    raise ChemistryError(f"unsupported bond order {raw_order!r}") from exc
  # Biotite 1.4: 1/2/3 are S/D/T, 5/6/7 are aromatic subtypes, and 9 is
  # generic aromatic.  4 is QUADRUPLE and must never be silently rewritten.
  mapping = {
      1: (1, "SING", "N", Chem.BondType.SINGLE),
      2: (2, "DOUB", "N", Chem.BondType.DOUBLE),
      3: (3, "TRIP", "N", Chem.BondType.TRIPLE),
      5: (9, "SING", "Y", Chem.BondType.SINGLE),
      6: (9, "DOUB", "Y", Chem.BondType.DOUBLE),
      7: (9, "TRIP", "Y", Chem.BondType.TRIPLE),
      9: (9, "SING", "Y", Chem.BondType.AROMATIC),
  }
  try:
    return mapping[order]
  except KeyError as exc:
    raise ChemistryError(f"unsupported bond order {raw_order!r}") from exc


def source_graph(
    atoms: Any,
    rows: Sequence[int],
    *,
    preserve_aromatic_orders: bool = False,
) -> AtomGraph:
  """Extract a heavy-atom graph, optionally retaining aromatic base orders."""
  normalized_rows = [int(value) for value in np.asarray(rows).tolist()]
  if not normalized_rows or len(set(normalized_rows)) != len(normalized_rows):
    raise ChemistryError("ligand graph rows must be non-empty and unique")
  names = tuple(str(atoms.atom_name[row]).strip() for row in normalized_rows)
  if any(
      not name or len(name) > 4 or any(char.isspace() for char in name)
      for name in names
  ) or len(set(names)) != len(names):
    raise ChemistryError("ligand graph atom names must be unique and <=4 characters")
  elements_array = np.asarray(
      getattr(atoms, "element", np.full(len(atoms), "", dtype=object)), dtype=object
  )
  elements = tuple(_element(elements_array[row]) for row in normalized_rows)
  values = _charge_values(atoms, normalized_rows)
  charges = tuple(_charge(values, index) for index in range(len(normalized_rows)))
  row_to_local = {row: index for index, row in enumerate(normalized_rows)}
  seen: dict[tuple[int, int], int] = {}
  if getattr(atoms, "bonds", None) is not None:
    for raw_i, raw_j, raw_order in atoms.bonds.as_array().tolist():
      row_i, row_j = int(raw_i), int(raw_j)
      if row_i not in row_to_local or row_j not in row_to_local:
        continue
      if row_i == row_j:
        raise ChemistryError("ligand graph contains a self-bond")
      canonical_order, _, _, _ = _bond_info(raw_order)
      graph_order = int(raw_order) if preserve_aromatic_orders else canonical_order
      key = tuple(sorted((row_to_local[row_i], row_to_local[row_j])))
      previous = seen.get(key)
      if previous is not None and previous != graph_order:
        raise ChemistryError("ligand graph contains conflicting bond orders")
      seen[key] = graph_order
  bonds = tuple((left, right, order) for (left, right), order in sorted(seen.items()))
  return AtomGraph(names, elements, charges, bonds)


def molecule_graph(
    molecule: Any,
    *,
    atom_names: Sequence[str] | None = None,
) -> AtomGraph:
  """Extract a heavy-atom graph from a trusted RDKit molecule."""
  Chem, _ = _rdkit()
  try:
    molecule = Chem.RemoveHs(Chem.Mol(molecule), sanitize=True)
  except Exception as exc:
    raise ChemistryError("trusted ligand molecule cannot be hydrogen-normalized") from exc
  if molecule.GetNumAtoms() == 0:
    raise ChemistryError("trusted ligand molecule has no heavy atoms")
  if len(Chem.GetMolFrags(molecule)) != 1:
    raise ChemistryError("trusted ligand molecule must be one connected component")
  if atom_names is None:
    names = []
    for atom in molecule.GetAtoms():
      if atom.HasProp("atom_name"):
        names.append(atom.GetProp("atom_name").strip())
      elif atom.HasProp("name"):
        names.append(atom.GetProp("name").strip())
      else:
        names.append(f"{atom.GetSymbol().upper()}{atom.GetIdx() + 1}")
  else:
    names = [str(value).strip() for value in atom_names]
  if len(names) != molecule.GetNumAtoms() or any(
      not name or len(name) > 4 or any(char.isspace() for char in name)
      for name in names
  ) or len(set(names)) != len(names):
    raise ChemistryError("trusted ligand atom names must be unique and <=4 characters")
  elements = tuple(atom.GetSymbol().upper() for atom in molecule.GetAtoms())
  charges = tuple(int(atom.GetFormalCharge()) for atom in molecule.GetAtoms())
  bonds: list[tuple[int, int, int]] = []
  for bond in molecule.GetBonds():
    if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
      order = 9
    elif bond.GetBondType() == Chem.BondType.SINGLE:
      order = 1
    elif bond.GetBondType() == Chem.BondType.DOUBLE:
      order = 2
    elif bond.GetBondType() == Chem.BondType.TRIPLE:
      order = 3
    else:
      raise ChemistryError(f"trusted ligand has unsupported bond type {bond.GetBondType()}")
    left, right = sorted((int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())))
    bonds.append((left, right, order))
  return AtomGraph(tuple(names), elements, charges, tuple(sorted(bonds)))


def trusted_smiles(smiles: str) -> Any:
  """Parse one authoritative, connected SMILES molecule without source pose data."""
  Chem, _ = _rdkit()
  molecule = Chem.MolFromSmiles(str(smiles))
  if molecule is None:
    raise ChemistryError("SMILES cannot be parsed")
  if len(Chem.GetMolFrags(molecule)) != 1:
    raise ChemistryError("SMILES must contain one connected component")
  try:
    molecule = Chem.RemoveHs(Chem.Mol(molecule), sanitize=True)
  except Exception as exc:
    raise ChemistryError("SMILES molecule cannot be hydrogen-normalized") from exc
  if molecule.GetNumAtoms() == 0:
    raise ChemistryError("SMILES molecule has no heavy atoms")
  return molecule


def _trusted_copy(molecule: Any) -> Any:
  """Copy a caller-owned RDKit molecule and discard caller conformers."""
  Chem, _ = _rdkit()
  try:
    copied = Chem.RemoveHs(Chem.Mol(molecule), sanitize=True)
  except Exception as exc:
    raise ChemistryError("trusted ligand molecule cannot be hydrogen-normalized") from exc
  if copied.GetNumAtoms() == 0:
    raise ChemistryError("trusted ligand molecule has no heavy atoms")
  if len(Chem.GetMolFrags(copied)) != 1:
    raise ChemistryError("trusted ligand molecule must be one connected component")
  # The input conformer is not an identity authority.  Rendering below embeds
  # a fresh conformer with the fixed AF3 materialization seed instead.
  copied.RemoveAllConformers()
  return copied


def _explicit_atom_names(molecule: Any) -> tuple[str, ...] | None:
  """Return caller-provided atom names, rejecting a partially named molecule."""
  names: list[str | None] = []
  found = False
  for atom in molecule.GetAtoms():
    name: str | None = None
    for property_name in ("atom_name", "name"):
      if atom.HasProp(property_name):
        found = True
        name = atom.GetProp(property_name).strip()
        break
    names.append(name)
  if not found:
    return None
  if any(
      not name or len(name) > 4 or any(char.isspace() for char in str(name))
      for name in names
  ) or len(set(names)) != len(names):
    raise ChemistryError("trusted ligand atom-name properties are incomplete or ambiguous")
  return tuple(str(name) for name in names)


def _mapping_requires_charge_match(
    source: AtomGraph,
    mapping: Sequence[int],
    trusted: AtomGraph,
    *,
    matcher: _TrustedGraphMatcher | None = None,
) -> None:
  """Compare every source charge that is explicitly annotated."""
  valid = (
      _mapping_valid(source, trusted, mapping) if matcher is None
      else matcher.mapping_valid(source, mapping)
  )
  if not valid:
    raise ChemistryError(
        "trusted ligand molecule does not match source elements, bonds, or charges"
    )


def materialize_trusted_molecule(
    source: AtomGraph,
    molecule: Any,
) -> tuple[MaterializedMolecule, tuple[int, ...]]:
  """Materialize an authoritative RDKit molecule without SMILES round-tripping.

  A molecule with complete atom_name/name properties is mapped by those exact
  names.  Otherwise compiler A1 order is used only after a complete
  graph/charge check; all other source names require one unique graph
  bijection.  Ambiguous or drifting graphs fail closed.
  """
  trusted = _trusted_copy(molecule)
  explicit_names = _explicit_atom_names(trusted)
  matcher = _TrustedGraphMatcher(trusted, atom_names=explicit_names)
  trusted_graph = matcher.graph
  if len(source.names) != len(trusted_graph.names):
    raise ChemistryError("trusted molecule atom count differs from source ligand graph")
  if explicit_names is not None and set(source.names) == set(explicit_names):
    trusted_by_name = {name: index for index, name in enumerate(explicit_names)}
    mapping = tuple(trusted_by_name[name] for name in source.names)
    _mapping_requires_charge_match(source, mapping, trusted_graph, matcher=matcher)
  else:
    # map_source_to_molecule enforces canonical A1 order before attempting
    # graph search.  Passing the copied molecule keeps the no-guess contract
    # while allowing a unique noncanonical graph bijection when appropriate.
    mapping = _map_source_to_graph(source, matcher)
    _mapping_requires_charge_match(source, mapping, trusted_graph, matcher=matcher)
  atom_names: list[str | None] = [None] * trusted.GetNumAtoms()
  for source_index, trusted_index in enumerate(mapping):
    atom_names[trusted_index] = source.names[source_index]
  if any(name is None for name in atom_names):
    raise ChemistryError("trusted molecule source-row atom mapping is incomplete")
  return (
      MaterializedMolecule(trusted, tuple(str(name) for name in atom_names)),
      mapping,
  )


def materialize_smiles(source: AtomGraph, smiles: str) -> tuple[MaterializedMolecule, tuple[int, ...]]:
  """Map source rows to a trusted SMILES graph and carry source atom names."""
  molecule = trusted_smiles(smiles)
  mapping = map_source_to_molecule(source, molecule)
  atom_names: list[str | None] = [None] * molecule.GetNumAtoms()
  for source_index, trusted_index in enumerate(mapping):
    atom_names[trusted_index] = source.names[source_index]
  if any(name is None for name in atom_names):
    raise ChemistryError("SMILES source-row atom mapping is incomplete")
  return (
      MaterializedMolecule(molecule, tuple(str(name) for name in atom_names)),
      mapping,
  )


def _edge_signature(graph: AtomGraph) -> tuple[tuple[str, str, int], ...]:
  return tuple(sorted(
      (min(graph.names[left], graph.names[right]), max(graph.names[left], graph.names[right]), order)
      for left, right, order in graph.bonds
  ))


def _mapping_valid(source: AtomGraph, trusted: AtomGraph, mapping: Sequence[int]) -> bool:
  if len(mapping) != len(source.names) or len(set(mapping)) != len(mapping):
    return False
  for source_index, trusted_index in enumerate(mapping):
    if source.elements[source_index] != trusted.elements[trusted_index]:
      return False
    source_charge = source.charges[source_index]
    if source_charge is not None and source_charge != trusted.charges[trusted_index]:
      return False
  observed = tuple(sorted(
      (min(mapping[left], mapping[right]), max(mapping[left], mapping[right]), order)
      for left, right, order in source.bonds
  ))
  return observed == trusted.bonds


class _TrustedGraphMatcher:
  """Exact graph matcher with one lazy, trusted Kekule bond representation."""

  def __init__(self, molecule: Any, *, atom_names: Sequence[str] | None = None) -> None:
    self.graph = molecule_graph(molecule, atom_names=atom_names)
    self._molecule = molecule
    self._edges = {(left, right): order for left, right, order in self.graph.bonds}
    self._incident = [[] for _ in self.graph.names]
    for left, right in self._edges:
      self._incident[left].append((left, right))
      self._incident[right].append((left, right))
    self._kekule_edges: dict[tuple[int, int], int] | None = None

  def _kekule_orders(self) -> dict[tuple[int, int], int]:
    if self._kekule_edges is None:
      Chem, _ = _rdkit()
      try:
        copied = Chem.RemoveHs(Chem.Mol(self._molecule), sanitize=False)
        Chem.Kekulize(copied, clearAromaticFlags=True)
        self._kekule_edges = {
            tuple(sorted((int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())))):
                _rdkit_bond_order(bond)
            for bond in copied.GetBonds()
        }
      except Exception as exc:
        raise ChemistryError("trusted ligand aromatic bonds cannot be Kekulized") from exc
    return self._kekule_edges

  def bond_matches(self, key: tuple[int, int], source_order: int | None) -> bool:
    trusted_order = self._edges.get(key)
    if source_order == trusted_order:
      return True
    if trusted_order != 9 or source_order not in {1, 2, 3, 5, 6, 7}:
      return False
    base = source_order - 4 if source_order in {5, 6, 7} else source_order
    return self._kekule_orders()[key] == base

  def mapping_valid(self, source: AtomGraph, mapping: Sequence[int]) -> bool:
    if _mapping_valid(source, self.graph, mapping):
      return True
    if len(mapping) != len(source.names) or len(set(mapping)) != len(mapping):
      return False
    if len(source.bonds) != len(self.graph.bonds):
      return False
    for source_index, trusted_index in enumerate(mapping):
      if source.elements[source_index] != self.graph.elements[trusted_index]:
        return False
      charge = source.charges[source_index]
      if charge is not None and charge != self.graph.charges[trusted_index]:
        return False
    observed = [
        (tuple(sorted((mapping[left], mapping[right]))), order)
        for left, right, order in source.bonds
    ]
    if {key for key, _order in observed} != self._edges.keys():
      return False
    return all(self.bond_matches(key, order) for key, order in observed)

  def degree_matches(self, source_orders: Sequence[int], trusted_index: int) -> bool:
    incident = self._incident[trusted_index]
    if len(source_orders) != len(incident):
      return False
    if sorted(source_orders) == sorted(self._edges[key] for key in incident):
      return True
    candidates = [
        [index for index, key in enumerate(incident) if self.bond_matches(key, order)]
        for order in source_orders
    ]
    if any(not candidate for candidate in candidates):
      return False
    candidates.sort(key=len)

    def assign(index: int, used: set[int]) -> bool:
      if index == len(candidates):
        return True
      return any(
          assign(index + 1, used | {edge})
          for edge in candidates[index] if edge not in used
      )

    return assign(0, set())


def map_source_to_molecule(source: AtomGraph, molecule: Any) -> tuple[int, ...]:
  """Return source-local -> trusted-molecule atom mapping, fail-closed."""
  return _map_source_to_graph(source, _TrustedGraphMatcher(molecule))


def _map_source_to_graph(source: AtomGraph, matcher: _TrustedGraphMatcher) -> tuple[int, ...]:
  trusted = matcher.graph
  if len(source.names) != len(trusted.names):
    raise ChemistryError("SMILES atom count differs from source ligand graph")
  if len(source.bonds) != len(trusted.bonds):
    raise ChemistryError("SMILES bond count differs from source ligand graph")

  # The compiler's A1/A2/... names are an explicit atom-order contract.  Use
  # that order only after checking the complete graph.
  canonical_indices: list[int] = []
  for name in source.names:
    match = re.fullmatch(r"A([1-9][0-9]*)", name)
    if match is None:
      canonical_indices = []
      break
    canonical_indices.append(int(match.group(1)) - 1)
  if canonical_indices and set(canonical_indices) == set(range(len(source.names))):
    if matcher.mapping_valid(source, canonical_indices):
      return tuple(canonical_indices)
    raise ChemistryError(
        "canonical compiler A1 atom mapping does not match trusted SMILES"
    )

  # Otherwise require one exact graph bijection.  Do not choose an arbitrary
  # member of a symmetric graph automorphism class.
  candidates: dict[int, list[int]] = {}
  for source_index, source_element in enumerate(source.elements):
    source_degree = sorted(
        order for left, right, order in source.bonds
        if left == source_index or right == source_index
    )
    candidates[source_index] = [
        trusted_index
        for trusted_index, trusted_element in enumerate(trusted.elements)
        if trusted_element == source_element
        and matcher.degree_matches(source_degree, trusted_index)
        and (
            source.charges[source_index] is None
            or source.charges[source_index] == trusted.charges[trusted_index]
        )
    ]
    if not candidates[source_index]:
      raise ChemistryError("SMILES has no exact source-row atom mapping")
  order = sorted(candidates, key=lambda index: (len(candidates[index]), index))
  found: list[tuple[int, ...]] = []
  mapping: dict[int, int] = {}
  used: set[int] = set()

  def visit(depth: int) -> None:
    if len(found) > 1:
      return
    if depth == len(order):
      candidate = tuple(mapping[index] for index in range(len(source.names)))
      if matcher.mapping_valid(source, candidate):
        found.append(candidate)
      return
    source_index = order[depth]
    for trusted_index in candidates[source_index]:
      if trusted_index in used:
        continue
      # Check every already assigned endpoint before recursing.
      valid = True
      for previous_source, previous_trusted in mapping.items():
        source_order = next(
            (bond_order for left, right, bond_order in source.bonds
             if {left, right} == {source_index, previous_source}),
            None,
        )
        key = tuple(sorted((trusted_index, previous_trusted)))
        if not matcher.bond_matches(key, source_order):
          valid = False
          break
      if not valid:
        continue
      mapping[source_index] = trusted_index
      used.add(trusted_index)
      visit(depth + 1)
      used.remove(trusted_index)
      mapping.pop(source_index)

  visit(0)
  if len(found) != 1:
    raise ChemistryError(
        "SMILES source-row atom mapping is ambiguous" if len(found) > 1
        else "SMILES source-row atom mapping is unavailable"
    )
  return found[0]


def _graph_molecule(graph: AtomGraph) -> Any:
  Chem, _ = _rdkit()
  editable = Chem.RWMol()
  for element, charge in zip(graph.elements, graph.charges, strict=True):
    atom = Chem.Atom(element.title())
    atom.SetFormalCharge(0 if charge is None else int(charge))
    editable.AddAtom(atom)
  for left, right, order in graph.bonds:
    _, _, aromatic, rdkit_order = _bond_info(order)
    bond_index = editable.AddBond(left, right, rdkit_order)
    if bond_index <= 0:
      raise ChemistryError("ligand graph contains an invalid bond")
    if aromatic == "Y":
      bond = editable.GetBondWithIdx(bond_index - 1)
      bond.SetIsAromatic(True)
      editable.GetAtomWithIdx(left).SetIsAromatic(True)
      editable.GetAtomWithIdx(right).SetIsAromatic(True)
  molecule = editable.GetMol()
  try:
    Chem.SanitizeMol(molecule)
  except Exception as exc:
    raise ChemistryError("ligand graph cannot be sanitized") from exc
  return molecule


def _embedded_molecule(
    molecule: Any,
    *,
    add_hydrogens: bool,
    remove_hydrogens: bool,
) -> tuple[Any, np.ndarray]:
  Chem, AllChem = _rdkit()
  try:
    working = Chem.Mol(molecule)
    if add_hydrogens:
      working = Chem.AddHs(working)
    status = AllChem.EmbedMolecule(
        working,
        randomSeed=101,
        clearConfs=True,
        useRandomCoords=False,
    )
    if status != 0 or not working.GetNumConformers():
      raise ValueError(f"RDKit conformer embedding returned status {status}")
    if AllChem.MMFFHasAllMoleculeParams(working):
      AllChem.MMFFOptimizeMolecule(working, maxIters=500)
    elif AllChem.UFFHasAllMoleculeParams(working):
      AllChem.UFFOptimizeMolecule(working, maxIters=500)
    if remove_hydrogens:
      working = Chem.RemoveHs(working)
    conformer = working.GetConformer()
    coordinates = np.asarray(
        [
            [float(position.x), float(position.y), float(position.z)]
            for position in (
                conformer.GetAtomPosition(i) for i in range(working.GetNumAtoms())
            )
        ],
        dtype=np.float64,
    )
  except Exception as exc:
    raise ChemistryError("ligand graph has no valid deterministic conformer") from exc
  if not np.isfinite(coordinates).all():
    raise ChemistryError("ligand graph conformer has non-finite coordinates")
  if len(coordinates) == 1 and float(np.max(np.abs(coordinates))) <= 1e-3:
    coordinates[0] = np.asarray([1.0, 0.125, -0.25], dtype=np.float64)
  if not np.any(np.abs(coordinates) > 1e-7):
    raise ChemistryError("ligand graph conformer is all-origin")
  if len(coordinates) > 1 and float(np.ptp(coordinates, axis=0).max()) <= 1e-7:
    raise ChemistryError("ligand graph conformer is collapsed")
  return working, coordinates


def _conformer(molecule: Any) -> np.ndarray:
  _, coordinates = _embedded_molecule(
      molecule, add_hydrogens=True, remove_hydrogens=True
  )
  return coordinates


def _rdkit_bond_order(bond: Any, *, preserve_aromatic_orders: bool = False) -> int:
  Chem, _ = _rdkit()
  if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
    if not preserve_aromatic_orders:
      return 9
    # The renderer Kekulizes its private molecule before requesting a concrete
    # aromatic witness.  Other graph/matching callers retain generic aromatic.
    if bond.GetBondType() == Chem.BondType.SINGLE:
      return 5
    if bond.GetBondType() == Chem.BondType.DOUBLE:
      return 6
    raise ChemistryError("trusted ligand aromatic bond has no concrete Kekule order")
  if bond.GetBondType() == Chem.BondType.SINGLE:
    return 1
  if bond.GetBondType() == Chem.BondType.DOUBLE:
    return 2
  if bond.GetBondType() == Chem.BondType.TRIPLE:
    return 3
  raise ChemistryError(f"trusted ligand has unsupported bond type {bond.GetBondType()}")


def _stereo_name(bond: Any) -> str:
  stereo = str(bond.GetStereo())
  if stereo in {"STEREOE", "STEREOTRANS"}:
    return "E"
  if stereo in {"STEREOZ", "STEREOCIS"}:
    return "Z"
  return "N"

def _quote_name(name: str) -> str:
  if any(char.isspace() for char in name) or not name or len(name) > 4:
    raise ChemistryError("CCD atom names must be non-empty and <=4 characters")
  return name


def render_component_ccd(
    component_id: str,
    molecule: Any,
    atom_names: Sequence[str],
    *,
    include_hydrogens: bool = False,
) -> str:
  """Render trusted molecule chemistry as one official AF3 CCD block.

  Explicit SMILES materialization includes stereodefining hydrogens so the
  native RDKit/AF3 parser can restore tetrahedral stereo from the generated
  conformer.  Generic graph-only custom CCDs remain heavy-atom-only.
  """
  Chem, _ = _rdkit()
  molecule = Chem.RemoveHs(Chem.Mol(molecule), sanitize=True)
  if len(atom_names) != molecule.GetNumAtoms():
    raise ChemistryError("trusted ligand atom names do not match heavy-atom count")
  # Embed/AddHs can select another valid Kekule phase.  Establish the named
  # heavy-atom witness before embedding so the CCD retains the same concrete
  # aromatic orders as the trusted graph matcher.  Only this copy is changed.
  bond_molecule = Chem.Mol(molecule)
  try:
    Chem.Kekulize(bond_molecule, clearAromaticFlags=False)
  except Exception as exc:
    raise ChemistryError("trusted ligand aromatic bonds cannot be Kekulized") from exc
  if include_hydrogens:
    output_molecule, coordinates = _embedded_molecule(
        molecule, add_hydrogens=True, remove_hydrogens=False
    )
    output_names = [str(name) for name in atom_names]
    used_names = set(output_names)
    hydrogen_ordinal = 1
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def _hydrogen_name(ordinal: int) -> str:
      value = int(ordinal)
      digits = "0"
      if value:
        digits = ""
        while value:
          value, remainder = divmod(value, len(alphabet))
          digits = alphabet[remainder] + digits
      return "H" + digits[-3:].rjust(3, "0")

    for atom in list(output_molecule.GetAtoms())[molecule.GetNumAtoms():]:
      name = _hydrogen_name(hydrogen_ordinal)
      while name in used_names:
        hydrogen_ordinal += 1
        name = _hydrogen_name(hydrogen_ordinal)
      hydrogen_ordinal += 1
      output_names.append(name)
      used_names.add(name)
  else:
    output_molecule, coordinates = _embedded_molecule(
        molecule, add_hydrogens=True, remove_hydrogens=True
    )
    output_names = [str(name) for name in atom_names]
  if len(output_names) != output_molecule.GetNumAtoms():
    raise ChemistryError("trusted ligand output atom names are incomplete")
  if len(set(output_names)) != len(output_names):
    raise ChemistryError("trusted ligand output atom names are not unique")
  for name in output_names:
    _quote_name(name)
  atom_records = [
      (
          output_names[index],
          str(atom.GetSymbol()).upper(),
          int(atom.GetFormalCharge()),
          coordinates[index],
      )
      for index, atom in enumerate(output_molecule.GetAtoms())
  ]
  lines = [
      f"data_{component_id}",
      "loop_",
      "_chem_comp.id",
      "_chem_comp.name",
      "_chem_comp.type",
      "_chem_comp.formula",
      "_chem_comp.mon_nstd_parent_comp_id",
      "_chem_comp.pdbx_synonyms",
      "_chem_comp.formula_weight",
      f"{component_id} {component_id} NON-POLYMER ? ? ? ?",
      "loop_",
      "_chem_comp_atom.comp_id",
      "_chem_comp_atom.atom_id",
      "_chem_comp_atom.type_symbol",
      "_chem_comp_atom.charge",
      "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
      "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
      "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
  ]
  for name, element, charge, coordinate in atom_records:
    x, y, z = (float(value) for value in coordinate)
    lines.append(
        f"{component_id} {name} {element} {charge} "
        f"{x:.6f} {y:.6f} {z:.6f}"
    )
  lines.extend([
      "loop_",
      "_chem_comp_bond.comp_id",
      "_chem_comp_bond.atom_id_1",
      "_chem_comp_bond.atom_id_2",
      "_chem_comp_bond.value_order",
      "_chem_comp_bond.pdbx_aromatic_flag",
      "_chem_comp_bond.pdbx_stereo_config",
  ])
  for bond in output_molecule.GetBonds():
    left = int(bond.GetBeginAtomIdx())
    right = int(bond.GetEndAtomIdx())
    chemistry_bond = (
        bond_molecule.GetBondBetweenAtoms(left, right)
        if left < molecule.GetNumAtoms() and right < molecule.GetNumAtoms()
        else bond
    )
    order = _rdkit_bond_order(chemistry_bond, preserve_aromatic_orders=True)
    _, value_order, aromatic, _ = _bond_info(order)
    lines.append(
        f"{component_id} {output_names[left]} {output_names[right]} "
        f"{value_order} {aromatic} {_stereo_name(chemistry_bond)}"
    )
  return "\n".join(lines) + "\n"

def render_graph_ccd(component_id: str, graph: AtomGraph) -> str:
  return render_component_ccd(component_id, _graph_molecule(graph), graph.names)


def render_user_ccd(
    atoms: Any,
    components: Mapping[str, Sequence[Mapping[str, Any]]],
    molecule_overrides: Mapping[str, MaterializedMolecule] | None = None,
) -> str | None:
  """Render custom components once, validating every repeated occurrence."""
  if not components:
    return None
  blocks: list[str] = []
  overrides = molecule_overrides or {}
  for component_id, entries in components.items():
    if not entries:
      raise ChemistryError(f"custom CCD component {component_id!r} has no atom rows")
    override = overrides.get(component_id)
    graph = source_graph(atoms, entries[0]["rows"], preserve_aromatic_orders=override is not None)
    if override is None:
      signature = (tuple(sorted(zip(graph.names, graph.elements, graph.charges))), _edge_signature(graph))
      for entry in entries[1:]:
        other = source_graph(atoms, entry["rows"])
        if (tuple(sorted(zip(other.names, other.elements, other.charges))), _edge_signature(other)) != signature:
          raise ChemistryError(f"custom CCD component {component_id!r} has inconsistent repeated graphs")
      blocks.append(render_graph_ccd(component_id, graph))
    else:
      matcher = _TrustedGraphMatcher(override.molecule, atom_names=override.atom_names)
      override_graph = matcher.graph
      native_by_name = {name: index for index, name in enumerate(override_graph.names)}
      for index, entry in enumerate(entries):
        occurrence = graph if index == 0 else source_graph(
            atoms, entry["rows"], preserve_aromatic_orders=True
        )
        mapping = tuple(native_by_name.get(name, -1) for name in occurrence.names)
        if any(index < 0 for index in mapping) or not matcher.mapping_valid(occurrence, mapping):
          raise ChemistryError(f"custom CCD component {component_id!r} does not match its source graph")
      blocks.append(render_component_ccd(component_id, override.molecule, override.atom_names, include_hydrogens=True))
  return "".join(blocks)


def _ccd_bond_columns(
    entry: Mapping[str, Any],
) -> tuple[list[Any], list[Any], list[Any], list[Any], list[Any]] | None:
  """Read one CCD bond loop without silently truncating malformed columns."""
  columns: list[list[Any]] = []
  for field_name in (
      "_chem_comp_bond.atom_id_1",
      "_chem_comp_bond.atom_id_2",
      "_chem_comp_bond.value_order",
      "_chem_comp_bond.pdbx_aromatic_flag",
      "_chem_comp_bond.pdbx_stereo_config",
  ):
    value = entry.get(field_name, ())
    if value is None:
      value = ()
    if isinstance(value, (str, bytes)):
      return None
    try:
      column = list(value)
    except TypeError:
      return None
    columns.append(column)
  starts, ends, orders, aromatic, stereo = columns
  bond_count = len(starts)
  # Aromatic and stereo columns are optional in older/native CCD records, but
  # when present they describe the complete same-length bond loop.
  if any(len(column) not in {0, bond_count} for column in (ends, orders, aromatic, stereo)):
    return None
  if bond_count and (len(ends) != bond_count or len(orders) != bond_count):
    return None
  return starts, ends, orders, aromatic, stereo


def ccd_has_explicit_stereo(entry: Any) -> bool:
  """Return whether a CCD block asserts stereo beyond an unspecified N."""
  if not (isinstance(entry, Mapping) or (hasattr(entry, "get") and hasattr(entry, "keys"))):
    return False
  columns = _ccd_bond_columns(entry)
  if columns is None:
    raise ChemistryError("CCD bond loop columns have inconsistent lengths")
  _starts, _ends, _orders, _aromatic, stereo = columns
  atom_stereo = entry.get("_chem_comp_atom.pdbx_stereo_config", ())
  atom_names = entry.get("_chem_comp_atom.atom_id", ())
  if len(atom_stereo) not in {0, len(atom_names)}:
    raise ChemistryError("CCD atom stereo column has inconsistent length")
  return any(
      str(value).strip().upper() not in {"", ".", "?", "N"}
      for value in (*stereo, *atom_stereo)
  )


def _ccd_graph(entry: Any) -> AtomGraph | None:
  if isinstance(entry, Mapping) or (hasattr(entry, "get") and hasattr(entry, "keys")):
    atom_names_raw = entry.get("_chem_comp_atom.atom_id")
    elements_raw = entry.get("_chem_comp_atom.type_symbol")
    if atom_names_raw is None or elements_raw is None:
      return None
    if isinstance(atom_names_raw, (str, bytes)) or isinstance(elements_raw, (str, bytes)):
      return None
    try:
      atom_names = list(atom_names_raw)
      elements = list(elements_raw)
    except TypeError:
      return None
    if len(atom_names) != len(elements) or not atom_names:
      return None
    charges_raw = entry.get("_chem_comp_atom.charge", ())
    if charges_raw is None:
      charges_raw = ()
    if isinstance(charges_raw, (str, bytes)):
      return None
    try:
      charges_raw = list(charges_raw)
    except TypeError:
      return None
    if charges_raw and len(charges_raw) != len(elements):
      return None
    normalized_names = tuple(str(value).strip() for value in atom_names)
    if (
        any(not name or len(name) > 4 or any(char.isspace() for char in name)
            for name in normalized_names)
        or len(set(normalized_names)) != len(normalized_names)
    ):
      return None
    heavy_indices = [
        index for index, element in enumerate(elements)
        if str(element).strip().upper() not in {"H", "D"}
    ]
    if not heavy_indices:
      return None
    name_to_all = {
        normalized_names[index]: index for index in range(len(normalized_names))
    }
    name_to_local = {
        normalized_names[index]: local for local, index in enumerate(heavy_indices)
    }
    names = tuple(normalized_names[index] for index in heavy_indices)
    charges: list[int | None] = []
    for index in heavy_indices:
      raw = charges_raw[index] if charges_raw else None
      try:
        charges.append(_charge(np.asarray([raw], dtype=object), 0))
      except ChemistryError:
        return None

    columns = _ccd_bond_columns(entry)
    if columns is None:
      return None
    starts, ends, orders, aromatic, _stereo = columns
    bonds: list[tuple[int, int, int]] = []
    for bond_index, (start, end, order) in enumerate(zip(starts, ends, orders, strict=True)):
      start_name, end_name = str(start).strip(), str(end).strip()
      if start_name not in name_to_all or end_name not in name_to_all:
        return None
      # Hydrogen endpoints are part of valid CCD records but are omitted from
      # the heavy-atom identity graph.
      if start_name not in name_to_local or end_name not in name_to_local:
        continue
      normalized_order = str(order).strip().upper()
      raw_order = {"SING": 1, "DOUB": 2, "TRIP": 3, "AROM": 9}.get(normalized_order)
      if raw_order is None:
        return None
      if aromatic:
        aromatic_flag = str(aromatic[bond_index]).strip().upper()
        if aromatic_flag not in {"Y", "N"}:
          return None
        if aromatic_flag == "Y":
          raw_order = 9
      left, right = sorted((name_to_local[start_name], name_to_local[end_name]))
      bonds.append((left, right, raw_order))
    try:
      normalized_elements = tuple(
          _element(value) for value in np.asarray(elements, dtype=object)[heavy_indices]
      )
    except ChemistryError:
      return None
    return AtomGraph(names, normalized_elements, tuple(charges), tuple(sorted(bonds)))
  try:
    return molecule_graph(entry)
  except ChemistryError:
    return None


def native_component_matches(atoms: Any, rows: Sequence[int], entry: Any) -> bool:
  """Match named source chemistry, retaining CCD-supported aromatic bases.

  Native CCD aromatic bonds carry both an aromatic flag and, when known,
  their SING/DOUB/TRIP base.  Accept that base, its Biotite aromatic subtype,
  or generic aromatic; never use aromatic normalization to hide a changed
  known base.  This comparison does not alter the source graph or molecule.
  """
  native = _ccd_graph(entry)
  if native is None:
    return False
  source = source_graph(atoms, rows)
  if set(source.names) != set(native.names) or len(source.names) != len(native.names):
    return False
  native_by_name = {name: index for index, name in enumerate(native.names)}
  for index, name in enumerate(source.names):
    native_index = native_by_name[name]
    if source.elements[index] != native.elements[native_index]:
      return False
    if source.charges[index] is not None and source.charges[index] != native.charges[native_index]:
      return False

  allowed: dict[tuple[str, str], set[int]] = {}
  for left, right, order in native.bonds:
    key = tuple(sorted((native.names[left], native.names[right])))
    if key in allowed:
      return False
    allowed[key] = {order}
  if isinstance(entry, Mapping) or (hasattr(entry, "get") and hasattr(entry, "keys")):
    columns = _ccd_bond_columns(entry)
    if columns is None:
      return False
    starts, ends, orders, _aromatic, _stereo = columns
    for start, end, order in zip(starts, ends, orders, strict=True):
      key = tuple(sorted((str(start).strip(), str(end).strip())))
      if allowed.get(key) != {9}:
        continue
      base = {"SING": 1, "DOUB": 2, "TRIP": 3}.get(str(order).strip().upper())
      if base is not None:
        allowed[key].update((base, base + 4))

  # source_graph deliberately normalizes aromatic subtypes for other callers.
  # Read raw orders only here so 5/6/7 retain their authored base for matching.
  row_names = {int(row): name for row, name in zip(rows, source.names, strict=True)}
  observed: dict[tuple[str, str], int] = {}
  if getattr(atoms, "bonds", None) is not None:
    for left, right, order in atoms.bonds.as_array().tolist():
      if int(left) not in row_names or int(right) not in row_names:
        continue
      key = tuple(sorted((row_names[int(left)], row_names[int(right)])))
      raw_order = int(order)
      if key in observed and observed[key] != raw_order:
        return False
      observed[key] = raw_order
  return observed.keys() == allowed.keys() and all(
      order in allowed[key] for key, order in observed.items()
  )


@lru_cache(maxsize=4)
def _load_native_ccd(root_text: str) -> Mapping[str, Any]:
  root = Path(root_text)
  candidates = [root]
  if root.is_dir():
    candidates = [
        root / "ccd.pickle",
        root / "ccd" / "ccd.pickle",
        root / "constants" / "converters" / "ccd.pickle",
    ]
  path = next((candidate for candidate in candidates if candidate.is_file()), None)
  if path is None:
    raise ChemistryError(f"native AF3 CCD pickle is missing under {root}")
  try:
    from alphafold3.constants.chemical_components import Ccd
    return Ccd(ccd_pickle_path=path)
  except Exception as exc:
    raise ChemistryError(f"native AF3 CCD cannot be loaded from {path}") from exc


def load_native_ccd(asset_root: str | Path) -> Mapping[str, Any]:
  """Load one native CCD mapping for an asset root and cache it process-wide."""
  root_text = str(Path(asset_root).expanduser().resolve())
  return _load_native_ccd(root_text)


__all__ = [
    "AtomGraph",
    "ChemistryError",
    "ccd_has_explicit_stereo",
    "MaterializedMolecule",
    "load_native_ccd",
    "materialize_trusted_molecule",
    "materialize_smiles",
    "map_source_to_molecule",
    "molecule_graph",
    "native_component_matches",
    "render_component_ccd",
    "render_graph_ccd",
    "render_user_ccd",
    "source_graph",
    "trusted_smiles",
]
