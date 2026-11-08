"""Bind declared pocket residues to the exact prepared native chain axis.

Source selectors are resolved once per immutable authoring snapshot by an
explicit owner cache. Only canonical source occurrences are cached; prepared
native positions are always checked on the current candidate payload.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class PocketResolutionError(ValueError):
    """A declared pocket has no exact retained native context occurrence."""


_POLYMERS = frozenset({"protein", "rna", "dna"})


@dataclass(frozen=True)
class _SourcePocket:
    selector: str | None
    occurrences: tuple[tuple[str, int, str], ...] = ()
    snapshot: str | None = None
    owner: str | None = None


class SourcePocketCache:
    """Run/owner-local source membership cache; never stores native positions."""

    def __init__(self):
        self._entries: dict[tuple[str, str], _SourcePocket | str] = {}

    def resolve(self, input_record, repo_root=None):
        from .metadata import authoring_dir, snapshot_path
        owner = authoring_dir(input_record)
        if owner is None:
            return _SourcePocket(None)
        owner = owner.expanduser().resolve()
        snapshot = snapshot_path(owner)
        if snapshot is None:
            return _SourcePocket(None)
        snapshot = snapshot.resolve()
        key = (str(snapshot), str(owner))
        if key not in self._entries:
            try:
                self._entries[key] = _resolve_source_membership(owner, snapshot, repo_root)
            except PocketResolutionError as exc:
                self._entries[key] = str(exc)
        entry = self._entries[key]
        if isinstance(entry, str):
            raise PocketResolutionError(entry)
        return entry


def _design_ids(task):
    raw = task.get("design_chains", ())
    if isinstance(raw, Mapping):
        return set(map(str, raw))
    return {str(chain) for item in raw if isinstance(item, Mapping) for chain in item}


def _source_root(owner, repo_root):
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    for parent in (owner, *owner.parents):
        if (parent / "src").is_dir() and (parent / "configs").is_dir():
            return parent
    return owner


def _resolve_source_membership(owner, snapshot, repo_root):
    from .config import load_yaml
    from .metadata import task_config
    try:
        config = load_yaml(snapshot)
        task = task_config(config, owner)
    except Exception as exc:
        raise PocketResolutionError(f"cannot read pocket authoring snapshot {snapshot}: {exc}") from exc
    if task is None:
        return _SourcePocket(None, snapshot=str(snapshot), owner=str(owner))
    selector = task.get("hotspot")
    if selector is None:
        return _SourcePocket(None, snapshot=str(snapshot), owner=str(owner))
    if not isinstance(selector, str) or not selector.strip():
        raise PocketResolutionError("declared source hotspot must be a non-empty selector")
    source_text = task.get("input")
    if not isinstance(source_text, str) or not source_text.strip():
        raise PocketResolutionError("declared source hotspot requires an authoritative input structure")
    root = _source_root(owner, repo_root)
    source_path = Path(source_text).expanduser()
    if not source_path.is_absolute():
        source_path = root / source_path
    ccd_path = config.get("ccd_path")
    if ccd_path is not None:
        ccd_path = Path(ccd_path).expanduser()
        if not ccd_path.is_absolute():
            ccd_path = root / ccd_path
    # No source structure is read or scanned unless a hotspot was declared.
    from src.design.utils.reader import read_structure
    from src.design.core.crop import source_selector
    from src.design.utils.source_mapping import source_locator
    import numpy as np
    try:
        source = read_structure(source_path.resolve(), ccd_path=ccd_path)
        selected = source_selector(source, selector)
    except Exception as exc:
        raise PocketResolutionError(f"cannot resolve declared source hotspot {selector!r}: {exc}") from exc
    design = _design_ids(task)
    occurrences, seen = [], set()
    for uid in selected:
        rows = np.flatnonzero(source.residue_uid == uid)
        if not len(rows):
            raise PocketResolutionError("declared hotspot source residue is unresolved")
        first = int(rows[0])
        locator = source_locator(source, first)
        key = (str(locator["chain_id"]), int(locator["res_id"]), str(locator["ins_code"]))
        if key[0] in design:
            raise PocketResolutionError(f"hotspot {key!r} must select context residues only")
        if str(source.mol_type[first]) not in _POLYMERS:
            raise PocketResolutionError("Boltz-2 residue pockets support protein/RNA/DNA context only")
        if key in seen:
            raise PocketResolutionError(f"hotspot {key!r} has an ambiguous canonical source occurrence")
        seen.add(key)
        occurrences.append(key)
    if not occurrences:
        raise PocketResolutionError("declared hotspot matched no source residues")
    return _SourcePocket(selector, tuple(occurrences), str(snapshot), str(owner))


def _chain(payload, native_id):
    found = [chain for chain in payload["chains"] if str(chain["id"]) == native_id]
    if len(found) != 1:
        raise PocketResolutionError(f"pocket native chain {native_id!r} is missing or ambiguous")
    return found[0]


def _position(value):
    if type(value) is not int or value <= 0:
        raise PocketResolutionError("pocket positions must be positive 1-based integers")
    return value


def _context_position(payload, native_id, position, source_id=None):
    position = _position(position)
    chain = _chain(payload, native_id)
    if chain.get("role") != "context" or chain.get("modality") not in _POLYMERS:
        raise PocketResolutionError("pocket residues must belong to protein/RNA/DNA context")
    sequence = chain.get("sequence")
    if not isinstance(sequence, str) or position > len(sequence):
        raise PocketResolutionError(f"pocket position {native_id}:{position} exceeds native sequence bounds")
    source_id = str(chain.get("source_id")) if source_id is None else source_id
    forward = payload.get("source_to_native_chain", {}).get(source_id, ())
    if (str(chain.get("source_id")) != source_id
            or payload.get("native_to_source_chain", {}).get(native_id) != source_id
            or native_id not in forward):
        raise PocketResolutionError(f"pocket native chain {native_id!r} has an incorrect source binding")
    return chain


def bind_source_pocket(input_record, payload, *, repo_root=None, cache=None):
    """Resolve saved source selectors and bind only their exact retained keys."""
    declaration = (cache if cache is not None else SourcePocketCache()).resolve(input_record, repo_root)
    if declaration.selector is None:
        return payload
    contacts, records = [], []
    index = payload.get("source_residue_native_index", {})
    for source_id, residue, insertion in declaration.occurrences:
        key = (source_id, residue, insertion)
        mapped = index.get(source_id, {}).get(str(residue), {}).get(insertion)
        if not isinstance(mapped, (tuple, list)) or len(mapped) != 2 or not isinstance(mapped[0], str):
            raise PocketResolutionError(f"hotspot {key!r} has no unique retained native context occurrence")
        native_id, position = mapped
        _context_position(payload, native_id, position, source_id)
        native_residues = payload.get("source_residue_map", {}).get(native_id, ())
        if position > len(native_residues):
            raise PocketResolutionError(f"hotspot {key!r} exceeds the prepared source residue map")
        occurrence = native_residues[position - 1]
        actual = None if not isinstance(occurrence, Mapping) else (
            str(occurrence.get("source_chain_id")), occurrence.get("source_res_id"),
            str(occurrence.get("source_ins_code", "")))
        if actual != key:
            raise PocketResolutionError(f"hotspot {key!r} is unresolved or has an incorrect native source occurrence")
        contacts.append([native_id, position])
        records.append({"source_chain_id": source_id, "source_res_id": residue,
                        "source_ins_code": insertion, "native_chain_id": native_id,
                        "native_residue_index": position})
    return dict(payload, boltz2_pocket={"source": "source", "selector": declaration.selector,
        "authoring_snapshot": declaration.snapshot, "task_owner": declaration.owner,
        "source_occurrences": records, "contacts": contacts})


def bind_sequence_pocket(request, payload):
    """Bind native logical sequence positions without fabricating source atoms."""
    selected = [chain for chain in request["chains"] if chain.get("pocket")]
    if not selected:
        return payload
    contacts, records, seen = [], [], set()
    for authored in selected:
        source_id = str(authored["id"])
        if authored.get("role") != "context" or authored.get("modality") not in _POLYMERS:
            raise PocketResolutionError("sequence pockets require protein/RNA/DNA context residue lists")
        positions = authored["pocket"]
        if not isinstance(positions, (list, tuple)):
            raise PocketResolutionError("sequence pocket must be a list of 1-based residue positions")
        native_ids = payload.get("source_to_native_chain", {}).get(source_id, ())
        if len(native_ids) != 1 or not isinstance(native_ids[0], str):
            raise PocketResolutionError(f"sequence pocket chain {source_id!r} has no unique native sequence mapping")
        native_id = native_ids[0]
        for position in positions:
            native = _context_position(payload, native_id, position, source_id)
            if native["sequence"] != authored.get("sequence"):
                raise PocketResolutionError(f"sequence pocket chain {source_id!r} differs from its native sequence binding")
            contact = (native_id, position)
            if contact in seen:
                continue
            seen.add(contact)
            contacts.append(list(contact))
            records.append({"source_chain_id": source_id, "sequence_position": position,
                            "native_chain_id": native_id, "native_residue_index": position})
    return dict(payload, boltz2_pocket={"source": "sequence", "source_occurrences": records, "contacts": contacts})


def native_pocket_constraints(payload):
    """Render one automatic binder pocket after narrow native validation."""
    pocket = payload.get("boltz2_pocket")
    if pocket is None:
        return []
    if not isinstance(pocket, Mapping) or not isinstance(pocket.get("contacts"), (tuple, list)) or not pocket["contacts"]:
        raise PocketResolutionError("resolved Boltz-2 pocket requires nonempty native contacts")
    designed = [chain for chain in payload["chains"] if chain.get("role") == "design"]
    # Sequence-native requests have one candidate of any supported modality.
    # Source ligand tasks retain their automatic designed-ligand binder rule.
    candidates = designed if pocket.get("source") == "sequence" else [
        chain for chain in designed if chain.get("modality") == "ligand"]
    if not candidates and len(designed) == 1:
        candidates = designed
    if len(candidates) != 1:
        raise PocketResolutionError("Boltz-2 pocket requires one unambiguous designed binder chain")
    binders = [str(candidates[0]["id"])]
    contacts = []
    for contact in pocket["contacts"]:
        if not isinstance(contact, (tuple, list)) or len(contact) != 2 or not isinstance(contact[0], str):
            raise PocketResolutionError("resolved Boltz-2 pocket contact must name a native chain and residue")
        _context_position(payload, contact[0], contact[1])
        contacts.append(list(contact))
    # Omit optional fields to preserve Boltz's 6 A / force=false defaults.
    return [{"pocket": {"binder": binders[0], "contacts": contacts}}]


def pocket_feature_evidence(tokenized, features, payload):
    """Check actual native contact channels only when a pocket was declared."""
    constraints = native_pocket_constraints(payload)
    if not constraints:
        return {"declared": False, "applied": False}
    import numpy as np
    from boltz.data import const
    pocket = constraints[0]["pocket"]
    by_chain = {str(chain["name"]): int(chain["asym_id"]) for chain in tokenized.structure.chains}
    tokens = tokenized.tokens
    if pocket["binder"] not in by_chain:
        raise PocketResolutionError("native Boltz-2 features lost the pocket binder chain")
    binder = np.flatnonzero(tokens["asym_id"] == by_chain[pocket["binder"]])
    if not len(binder):
        raise PocketResolutionError("native Boltz-2 features contain no pocket binder tokens")
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    conditioning = array(features["contact_conditioning"])
    threshold = array(features["contact_threshold"])
    for native_id, position in pocket["contacts"]:
        if native_id not in by_chain:
            raise PocketResolutionError(f"native Boltz-2 features lost pocket chain {native_id}")
        rows = np.flatnonzero((tokens["asym_id"] == by_chain[native_id]) & (tokens["res_idx"] == position - 1))
        if len(rows) != 1:
            raise PocketResolutionError(f"native Boltz-2 features lost pocket residue {native_id}:{position}")
        row = int(rows[0])
        if (not np.all(conditioning[binder, row, const.contact_conditioning_info["BINDER>POCKET"]] == 1)
                or not np.all(conditioning[row, binder, const.contact_conditioning_info["POCKET>BINDER"]] == 1)
                or not np.all(threshold[binder, row] == 6.0)
                or not np.all(threshold[row, binder] == 6.0)):
            raise PocketResolutionError(f"native Boltz-2 features did not include pocket residue {native_id}:{position}")
    return {"declared": True, "applied": True, "source": payload["boltz2_pocket"].get("source"),
            "binder": pocket["binder"], "contacts": pocket["contacts"],
            "max_distance": 6.0, "force": False, "feature_contact_pairs": len(binder) * len(pocket["contacts"])}
