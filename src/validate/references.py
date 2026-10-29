"""Recover source motif coordinates without resampling a generation contig."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping

from .config import load_yaml


class ReferenceUnavailable(ValueError):
    pass


def ordered_source_positions(token: str, source=None) -> list[tuple[str, int, str]]:
    from .inputs import _parse_contig_positions, _SOURCE_SEGMENT
    if not isinstance(token, str):
        raise ReferenceUnavailable("saved source contig token is not a string")
    positions = []
    # Numeric gap ranges are deliberately ignored. Their sampled lengths come
    # only from result.csv, never from another compiler or random draw.
    for segment in token.split("/"):
        match = _SOURCE_SEGMENT.fullmatch(segment.strip())
        if source is not None and match is not None:
            chain = match.group(1) or match.group(2)
            # Match the compiler's structural numbering and residue-order
            # ranges, including gaps and insertion codes in source files.
            residues, seen = [], set()
            for row in range(len(source)):
                if str(source.chain_id[row]) != chain or int(source.residue_uid[row]) in seen:
                    continue
                seen.add(int(source.residue_uid[row]))
                try:
                    number = int(source.label_seq_id[row])
                except ValueError:
                    number = int(source.res_id[row])
                residues.append((chain, number, str(source.ins_code[row])))
            def endpoint(number, insertion):
                matches = [i for i, item in enumerate(residues) if item[1] == number and (not insertion or item[2] == insertion)]
                if len(matches) != 1:
                    raise ReferenceUnavailable(f"source selector endpoint is missing/ambiguous: {chain}{number}{insertion}")
                return matches[0]
            start = endpoint(int(match.group(3)), match.group(4) or "")
            end = endpoint(int(match.group(5)), match.group(6) or "") if match.group(5) else start
            if end < start:
                raise ReferenceUnavailable("reversed source residue selector")
            positions.extend(residues[start:end + 1])
            continue
        for chain, values in _parse_contig_positions(segment).items():
            positions.extend((chain, position, ins) for position, ins in values)
    return positions


def source_mapping(input, task_config: dict, source=None) -> list[tuple[tuple, tuple]]:
    specs = task_config.get("condition", {}).get("contig", {})
    if not isinstance(specs, Mapping):
        raise ReferenceUnavailable("saved task has no source contig mapping")
    result = []
    for output_chain, final_positions in input.motif_positions.items():
        spec = specs.get(output_chain)
        if not isinstance(spec, Mapping):
            raise ReferenceUnavailable(f"no saved source contig for output chain {output_chain}")
        sources = ordered_source_positions(spec.get("token"), source)
        atom_selection = spec.get("polymer_atom")
        if isinstance(atom_selection, Mapping):
            excluded = set()
            for selector, atoms in atom_selection.items():
                if atoms == []:
                    excluded.update(ordered_source_positions(str(selector), source))
            sources = [source for source in sources if source not in excluded]
        if len(sources) != len(final_positions):
            raise ReferenceUnavailable(
                f"source/final motif counts differ for {output_chain}: {len(sources)} != {len(final_positions)}")
        result.extend(((output_chain, int(position), str(ins)), source)
                      for (position, ins), source in zip(final_positions, sources, strict=True))
    if not result:
        raise ReferenceUnavailable("no final motif positions were declared")
    if len({destination for destination, _ in result}) != len(result):
        raise ReferenceUnavailable("duplicate final motif residue locator")
    return result


def recover_reference(input):
    """Return source-coordinate atoms relabeled into final motif positions.

    The reference contains the complete SOURCE atom inventory for each copied
    residue. Thus a wrong/missing generated sidechain cannot erase required
    reference atoms and accidentally improve the motif RMSD.
    """
    import numpy as np
    from ..design.utils.reader import read_structure
    from ..design.utils.atoms import assign_identity_uids

    from .metadata import snapshot_path, task_config as saved_task
    snapshot = snapshot_path(input.task_dir)
    if snapshot is None:
        raise ReferenceUnavailable("no saved resolved_config.yaml for source motif recovery")
    config = load_yaml(snapshot)
    task_config = saved_task(config, input.task_dir)
    if not isinstance(task_config, Mapping):
        raise ReferenceUnavailable("saved configuration does not identify this task")
    raw_path = task_config.get("input")
    if not isinstance(raw_path, str) or not raw_path:
        raise ReferenceUnavailable("saved task has no source input path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path = path.resolve()
    source = read_structure(path)
    mapping = source_mapping(input, task_config, source)
    rows, chains, positions, insertions = [], [], [], []
    source_chains = np.asarray(source.chain_id).astype(str)
    source_positions = []
    for row in range(len(source)):
        try:
            source_positions.append(int(source.label_seq_id[row]))
        except ValueError:
            source_positions.append(int(source.res_id[row]))
    source_positions = np.asarray(source_positions)
    source_ins = np.asarray(source.ins_code).astype(str)
    for destination, origin in mapping:
        mask = (source_chains == origin[0]) & (source_positions == origin[1]) & (source_ins == origin[2])
        selected = np.flatnonzero(mask)
        if not len(selected):
            raise ReferenceUnavailable(f"source motif residue absent: {origin}")
        if len(set(source.residue_uid[selected].tolist())) != 1:
            raise ReferenceUnavailable(f"source motif residue is ambiguous: {origin}")
        rows.extend(selected.tolist())
        chains.extend([destination[0]] * len(selected))
        positions.extend([destination[1]] * len(selected))
        insertions.extend([destination[2]] * len(selected))
    reference = source[np.asarray(rows, dtype=int)].copy()
    reference.set_annotation("chain_id", np.asarray(chains, dtype=str))
    reference.set_annotation("label_asym_id", np.asarray(chains, dtype=str))
    reference.set_annotation("res_id", np.asarray(positions, dtype=int))
    reference.set_annotation("label_seq_id", np.asarray(positions, dtype=str))
    reference.set_annotation("ins_code", np.asarray(insertions, dtype=str))
    assign_identity_uids(reference)
    return reference, {"path": str(path), "snapshot": str(snapshot),
                       "mapping": [{"output": list(out), "source": list(src)} for out, src in mapping]}
