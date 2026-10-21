"""Lightweight source metadata checks; structural preparation belongs to runtime."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from biotite import InvalidFileError

from .records import ExpandedRequest

PreflightKey = tuple[str, int, tuple[tuple[str, int], ...]]


def preflight_key(request: ExpandedRequest) -> PreflightKey:
    # Distribution-resolved requests may share plan.index but have different lengths.
    return (request.task.name, request.plan.index,
            tuple(sorted(getattr(request.plan, "lengths", {}).items())))


def unique_preflight_requests(requests: Sequence[ExpandedRequest]) -> tuple[ExpandedRequest, ...]:
    """Keep the first sample as the deterministic representative of each plan."""
    unique: dict[PreflightKey, ExpandedRequest] = {}
    for request in requests:
        unique.setdefault(preflight_key(request), request)
    return tuple(unique.values())


def _source_residue_summary(task: Any) -> set[tuple[str, str]] | None:
    """Read chain/component identifiers only; never reconstruct chemistry here."""
    if task.input_path is None:
        return set()
    try:
        path = Path(task.input_path)
        rows: set[tuple[str, str]] = set()
        if path.suffix.lower() == '.cif':
            from biotite.structure.io.pdbx import CIFFile
            cif = CIFFile.read(path)
            block = cif[next(iter(cif))]
            # Cross-chain source links can merge scoring partners.
            if 'struct_conn' in block:
                links = block['struct_conn']
                left = links['ptnr1_label_asym_id'].as_array(str)
                right = links['ptnr2_label_asym_id'].as_array(str)
                design_ids = {chain.chain_id for chain in getattr(task, 'design_chains', ())}
                if any(a != b or a in design_ids for a, b in zip(left, right, strict=True)):
                    return None
            site = block['atom_site']
            chains = site['label_asym_id'].as_array(str)
            names = site['label_comp_id'].as_array(str)
            elements = site['type_symbol'].as_array(str)
            models = site['pdbx_PDB_model_num'].as_array(str) if 'pdbx_PDB_model_num' in site else ['1'] * len(chains)
            first = models[0] if len(models) else '1'
            for chain, name, element, model in zip(chains, names, elements, models, strict=True):
                if model == first and name.upper() not in {'HOH', 'WAT', 'DOD', 'H2O'} and element.upper() not in {'H', 'D', 'T'}:
                    if not chain or chain in {'.', '?'} or not element.strip():
                        return None
                    rows.add((str(chain), str(name).upper()))
        elif path.suffix.lower() == '.pdb':
            lines = path.read_text().splitlines()
            if any(line.startswith(('LINK  ', 'CONECT', 'SSBOND')) for line in lines):
                return None
            for line in lines:
                if line.startswith('ENDMDL'):
                    break
                if line[:6].strip() not in {'ATOM', 'HETATM'}:
                    continue
                name, element = line[17:20].strip().upper(), line[76:78].strip().upper()
                if not element or not line[21:22].strip():
                    return None
                if name not in {'HOH', 'WAT', 'DOD', 'H2O'} and element not in {'H', 'D', 'T'}:
                    rows.add((line[21], name))
        else:
            return None
        return rows
    except (OSError, ValueError, KeyError, IndexError, StopIteration, InvalidFileError):
        # The authoritative reader/compiler owns malformed input reporting.
        return None


def source_interfaces(
    requests: Sequence[ExpandedRequest],
) -> dict[PreflightKey, bool | None]:
    """Prove unlinked design interfaces; defer covalent group decisions to scoring.

    Multiple design chains imply an eligible interface only in the simple case
    with no declared/source links. Linked cases remain unknown here rather than
    treating every joint task as a design-context-only or inter-design interface.
    """
    interfaces = {}
    summaries = {}
    for request in unique_preflight_requests(requests):
        task = request.task
        if task.name not in summaries:
            summaries[task.name] = _source_residue_summary(task)
        summary = summaries[task.name]
        known = None
        if (task.crop is None and summary is not None
                and not task.condition.get("bond")
                and all(chain.modality != "ligand" for chain in task.design_chains)):
            design = {chain.chain_id for chain in task.design_chains}
            known = len(design) >= 2 or any(chain not in design for chain, _ in summary)
        interfaces[preflight_key(request)] = known
    return interfaces
