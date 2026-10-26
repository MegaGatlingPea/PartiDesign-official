"""Identify physical standard-peptide head-to-tail links from a saved atom graph."""
from __future__ import annotations

_STANDARD = frozenset((
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
))


def head_tail_chains(snapshot: dict) -> tuple[str, ...]:
    """Return chains with a real last-C to first-N single bond and intact backbone.

    Residues follow authoritative positive ``label_seq_id`` order when present,
    otherwise the controlled reader's residue occurrence order. Neither distances
    nor prediction-relative offsets establish a chemical connection.
    """
    annotations = snapshot["annotations"]
    chains = annotations["label_asym_id"]
    uids = annotations["residue_uid"]
    names = annotations["atom_name"]
    if not (len(chains) == len(uids) == len(names)):
        raise ValueError("inconsistent source topology annotations")
    labels = annotations.get("label_seq_id")
    residues: dict[tuple[str, int], dict] = {}
    chain_residues: dict[str, list[tuple[str, int]]] = {}
    atom_residues = []
    for index, (chain, uid, name) in enumerate(zip(chains, uids, names, strict=True)):
        key = (str(chain), int(uid))
        atom_residues.append(key)
        if key not in residues:
            residues[key] = {"atoms": {}, "label": labels[index] if labels is not None else None}
            chain_residues.setdefault(key[0], []).append(key)
        if labels is not None and str(residues[key]["label"]) != str(labels[index]):
            raise ValueError("a residue occurrence has inconsistent sequence labels")
        if str(name) in residues[key]["atoms"]:
            raise ValueError("ambiguous atom occurrence in source topology")
        residues[key]["atoms"][str(name)] = index
    for ordered in chain_residues.values():
        if labels is not None:
            try:
                numbers = [int(residues[key]["label"]) for key in ordered]
            except (TypeError, ValueError) as exc:
                raise ValueError("source peptide order requires integral sequence labels") from exc
            if not numbers or min(numbers) < 1 or len(set(numbers)) != len(numbers):
                raise ValueError("source peptide order requires unique positive sequence labels")
            ordered.sort(key=lambda key: int(residues[key]["label"]))
    positions = {key: position for ordered in chain_residues.values()
                 for position, key in enumerate(ordered)}
    links: dict[str, set[tuple[int, int]]] = {chain: set() for chain in chain_residues}
    closures: set[str] = set()
    for raw_left, raw_right, raw_order in snapshot.get("bonds", ()):
        left, right, order = int(raw_left), int(raw_right), int(raw_order)
        if not (0 <= left < len(names) and 0 <= right < len(names)):
            raise ValueError("source bond endpoint is outside the atom inventory")
        a, b = atom_residues[left], atom_residues[right]
        if a == b or a[0] != b[0] or {str(names[left]), str(names[right])} != {"C", "N"}:
            continue
        carbon, nitrogen = (a, b) if names[left] == "C" else (b, a)
        c, n = positions[carbon], positions[nitrogen]
        chain = carbon[0]
        if order != 1:
            raise ValueError("inter-residue peptide bonds must be single bonds")
        if n == c + 1:
            links[chain].add((c, n))
        elif c == len(chain_residues[chain]) - 1 and n == 0 and c >= 1:
            closures.add(chain)
        else:
            raise ValueError("nonterminal or reversed C-N connections are unsupported")
    result = []
    residue_names = annotations.get("res_name")
    for chain, ordered in chain_residues.items():
        if chain not in closures:
            continue
        if links[chain] != {(i, i + 1) for i in range(len(ordered) - 1)}:
            raise ValueError("head-to-tail closure requires an intact peptide backbone")
        for key in ordered:
            atoms = residues[key]["atoms"]
            if not {"N", "CA", "C", "O"}.issubset(atoms):
                raise ValueError("head-to-tail closure requires complete standard backbone atoms")
            if "OXT" in atoms:
                raise ValueError("a closed peptide cannot retain a terminal OXT atom")
            if residue_names is not None and any(str(residue_names[i]) not in _STANDARD for i in atoms.values()):
                raise ValueError("head-to-tail postfilter supports standard amino acids only")
        result.append(chain)
    return tuple(result)
