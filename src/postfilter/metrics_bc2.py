"""Requested BC2 coordinate metrics and BC1 DSSP on the parsed source view.

Definitions adapted from BindCraft2. See licenses/BindCraft2-LICENSE for terms.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
import tempfile

MAX_SASA = dict(zip("ARNDCQEGHILKMFPSTWYV", (121., 265., 187., 187., 148., 214., 214., 97.,
    216., 195., 191., 230., 203., 228., 154., 143., 163., 264., 255., 165.)))
HYDROPHOBIC = frozenset("ACVILMPFWY")
_DSSP_NAMES = frozenset(("bc1_binder_helix_percent", "bc1_binder_sheet_percent", "bc1_binder_loop_percent"))
_SSE_NAMES = frozenset("bc2_" + scope + "_" + state + "_fraction"
    for scope in ("binder", "interface") for state in ("helix", "beta_sheet", "loop"))
_SASA_NAMES = frozenset(("bc2_surface_hydrophobicity", "bc2_interface_buried_area",
    "bc2_interface_buried_area_fraction", "bc2_exposed_loop_fraction"))
_CONTACT_NAMES = frozenset(("bc2_interface_residues", "bc2_receptor_chains_contacted",
    "bc2_interface_hydrophobicity")) | frozenset("bc2_interface_aa_" + aa for aa in MAX_SASA)
_BC2_NAMES = _SSE_NAMES | _SASA_NAMES | _CONTACT_NAMES | frozenset((
    "bc2_backbone_clashes", "bc2_all_atom_clashes", "bc2_binder_chain_breaks"))


def _dssp(atoms, binder_chains, executable):
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile
    from Bio.PDB import PDBParser, DSSP
    if not executable:
        raise ValueError("DSSP executable is missing")
    chains = list(dict.fromkeys(atoms.chain_id.tolist()))
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    if len(chains) > len(alphabet):
        raise ValueError("too many chains for the DSSP adapter")
    aliases = dict(zip(chains, alphabet))
    private = atoms.copy()
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    wanted = []
    numbers = {chain: 0 for chain in chains}
    for start, stop in zip(starts[:-1], starts[1:]):
        chain = str(atoms.chain_id[start])
        numbers[chain] += 1
        private.res_id[start:stop] = numbers[chain]
        private.ins_code[start:stop] = ""
        private.chain_id[start:stop] = aliases[chain]
        if chain in binder_chains:
            wanted.append((aliases[chain], (" ", numbers[chain], " ")))
    file = PDBFile()
    file.set_structure(private)
    stream = StringIO()
    file.write(stream)
    text = "HEADER    POSTFILTER INPUT\n" + stream.getvalue()
    model = PDBParser(QUIET=True).get_structure("input", StringIO(text))[0]
    with tempfile.TemporaryDirectory(prefix="pd-postfilter-dssp-") as directory:
        path = Path(directory) / "input.pdb"
        path.write_text(text)
        assignments = DSSP(model, str(path), dssp=executable)
        if not wanted or any(key not in assignments for key in wanted):
            raise ValueError("DSSP did not assign every binder residue")
        states = [assignments[key][2] for key in wanted]
    helix = sum(s in "HGI" for s in states)
    sheet = sum(s == "E" for s in states)
    return {"bc1_binder_helix_percent": round(100 * helix / len(states), 2),
            "bc1_binder_sheet_percent": round(100 * sheet / len(states), 2),
            "bc1_binder_loop_percent": round(100 * (len(states) - helix - sheet) / len(states), 2)}


def compute_input_metrics_from_source(source, required_metrics, config):
    """Reuse exact source maps; run only requested coordinate or DSSP actions."""
    import numpy as np
    wanted = set(required_metrics)
    metrics, errors = {}, {}
    details = {"coordinate_stage": "input_success_refold"}
    dssp_names = wanted & _DSSP_NAMES
    if dssp_names:
        try:
            assignments = _dssp(source.atoms, source.binder_chains, config.get("dssp"))
            metrics.update({name: assignments[name] for name in dssp_names})
        except Exception as exc:
            errors.update({name: str(exc) for name in dssp_names})
    bc2 = wanted & _BC2_NAMES
    for name in wanted:
        if name.startswith("bc2_") and name not in _BC2_NAMES:
            errors[name] = "unsupported BC2 source geometry metric"
    if not bc2:
        return {"metrics": metrics, "errors": errors, "details": details}
    import biotite.structure as struc
    from scipy.spatial import cKDTree
    from src.design.utils.atoms import PROTEIN_ATOM37
    from .pose_mapping import _AA
    atoms = source.atoms
    order = []
    for chain in dict.fromkeys(atoms.chain_id.tolist()):
        chain_mask = atoms.chain_id == chain
        for uid in dict.fromkeys(atoms.residue_uid[chain_mask].tolist()):
            indices = np.flatnonzero(chain_mask & (atoms.residue_uid == uid))
            order.extend(sorted(indices, key=lambda i: str(atoms.atom_name[i])))
    atoms = atoms[order]
    coordinates = source.coordinates[order]
    binder = np.isin(atoms.chain_id, source.binder_chains)
    target = np.isin(atoms.chain_id, source.target_chains)
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    residue_id = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    residue_chains = atoms.chain_id[starts[:-1]]
    binder_res = np.flatnonzero(np.isin(residue_chains, source.binder_chains))
    binder_sequence = np.asarray([_AA[str(name)] for name in atoms.res_name[starts[:-1]][binder_res]])
    hydrophobic = np.isin(binder_sequence, list(HYDROPHOBIC))
    for name, atom_names in (("bc2_backbone_clashes", ("CA",)), ("bc2_all_atom_clashes", PROTEIN_ATOM37)):
        if name not in bc2:
            continue
        mask = np.isin(atoms.atom_name, atom_names)
        pairs = np.asarray(list(cKDTree(coordinates[mask]).query_pairs(2.5)), dtype=int).reshape(-1, 2)
        chains = atoms.chain_id[mask]
        metrics[name] = int(np.sum(chains[pairs[:, 0]] != chains[pairs[:, 1]]))
    contact_mask = None
    if bc2 & (_CONTACT_NAMES | {name for name in _SSE_NAMES if name.startswith("bc2_interface_")}):
        contacts = cKDTree(coordinates[binder]).query_ball_tree(cKDTree(coordinates[target]), 4.)
        binder_indices = np.flatnonzero(binder)
        target_indices = np.flatnonzero(target)
        contacted = {residue_id[binder_indices[i]] for i, indices in enumerate(contacts) if indices}
        contact_mask = np.asarray([rid in contacted for rid in binder_res])
        if "bc2_interface_residues" in bc2:
            metrics["bc2_interface_residues"] = int(contact_mask.sum())
        if "bc2_receptor_chains_contacted" in bc2:
            metrics["bc2_receptor_chains_contacted"] = len({str(atoms.chain_id[target_indices[j]])
                for indices in contacts for j in indices})
        for aa in MAX_SASA:
            name = "bc2_interface_aa_" + aa
            if name in bc2:
                metrics[name] = int(np.sum(contact_mask & (binder_sequence == aa)))
        if "bc2_interface_hydrophobicity" in bc2:
            if contact_mask.any():
                metrics["bc2_interface_hydrophobicity"] = float(hydrophobic[contact_mask].mean())
            else:
                errors["bc2_interface_hydrophobicity"] = "no binder-target contacts"
    states = None
    sse_names = bc2 & (_SSE_NAMES | {"bc2_exposed_loop_fraction"})
    if sse_names:
        try:
            states = np.concatenate([struc.annotate_sse(atoms[atoms.chain_id == chain])
                for chain in dict.fromkeys(residue_chains[binder_res].tolist())])
            for state, mask in (("helix", states == "a"), ("beta_sheet", states == "b"),
                                ("loop", ~np.isin(states, ("a", "b")))):
                name = "bc2_binder_" + state + "_fraction"
                if name in bc2:
                    metrics[name] = float(mask.mean())
                name = "bc2_interface_" + state + "_fraction"
                if name in bc2:
                    if contact_mask.any():
                        metrics[name] = float(mask[contact_mask].mean())
                    else:
                        errors[name] = "no binder-target contacts"
        except Exception as exc:
            errors.update({name: str(exc) for name in sse_names})
            states = None
    sasa_names = bc2 & _SASA_NAMES
    if sasa_names:
        details.update({"sasa_probe_radius": 1.4, "sasa_points": 100})
        def residue_sasa(array):
            try:
                per_atom = struc.sasa(array, probe_radius=1.4, point_number=100)
            except KeyError:
                details["sasa_radii"] = "Single (ProtOr unavailable)"
                per_atom = struc.sasa(array, probe_radius=1.4, point_number=100, vdw_radii="Single")
            if not np.isfinite(per_atom).all():
                raise ValueError("nonfinite solvent accessible area")
            return struc.apply_residue_wise(array, per_atom, np.sum)
        try:
            maximum = np.asarray([MAX_SASA[aa] for aa in binder_sequence])
            free_names = sasa_names - {"bc2_exposed_loop_fraction"}
            free = residue_sasa(atoms[binder]) if free_names else None
            bound_names = sasa_names - {"bc2_surface_hydrophobicity"}
            bound = residue_sasa(atoms)[binder_res] if bound_names else None
            if "bc2_surface_hydrophobicity" in bc2:
                exposed = free / maximum >= .2
                metrics["bc2_surface_hydrophobicity"] = float(hydrophobic[exposed].mean()) if exposed.any() else 0.
            if "bc2_interface_buried_area" in bc2:
                metrics["bc2_interface_buried_area"] = max(0., float(free.sum() - bound.sum()))
            if "bc2_interface_buried_area_fraction" in bc2:
                if free.sum() <= 0:
                    raise ValueError("zero isolated binder surface")
                metrics["bc2_interface_buried_area_fraction"] = max(0., float((free.sum() - bound.sum()) / free.sum()))
            if "bc2_exposed_loop_fraction" in bc2 and states is not None:
                loops = ~np.isin(states, ("a", "b"))
                metrics["bc2_exposed_loop_fraction"] = float(np.mean((bound / maximum >= .2)[loops])) if loops.any() else 0.
        except Exception as exc:
            for name in sasa_names:
                errors[name] = str(exc)
                metrics.pop(name, None)
    if "bc2_binder_chain_breaks" in bc2:
        breaks, observed = 0, False
        for chain in source.binder_chains:
            ca = coordinates[(atoms.chain_id == chain) & (atoms.atom_name == "CA")]
            if len(ca) > 1:
                breaks += int(np.sum(np.linalg.norm(np.diff(ca, axis=0), axis=1) > 4.3))
                observed = True
        if observed:
            metrics["bc2_binder_chain_breaks"] = breaks
        else:
            errors["bc2_binder_chain_breaks"] = "no adjacent resolved CA pair"
    for name, value in list(metrics.items()):
        if not np.isfinite(value):
            errors[name] = "nonfinite metric"
            del metrics[name]
    return {"metrics": metrics, "errors": errors, "details": details}
