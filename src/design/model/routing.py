"""Submission-wide model preferences from authored roles and retained conditions."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.records import DesignConfigError, DesignInputError, DesignSampleError, TaskSpec

_MEMBERS = {"protein": "prot", "rna": "nuc", "dna": "nuc", "ligand": "ligand"}
_KNOWN_LIGAND_MODES = {"smiles", "ccd", "type_bond"}


@dataclass(frozen=True)
class TaskModelPreference:
    task_name: str
    member: str
    reason: str


def _known_ligands(task: TaskSpec) -> set[str]:
    conditions = task.condition.get("ligand", {}) or {}
    if not isinstance(conditions, Mapping):
        return set()
    return {str(chain.chain_id) for chain in task.design_chains
            if chain.modality == "ligand"
            and isinstance(conditions.get(chain.chain_id), Mapping)
            and isinstance(conditions[chain.chain_id].get("mode"), str)
            and conditions[chain.chain_id]["mode"] in _KNOWN_LIGAND_MODES}


def _peptide_chains(task: TaskSpec) -> set[str]:
    return {str(chain.chain_id) for chain in task.design_chains
            if chain.modality == "protein" and getattr(chain, "sub_modality", None) == "peptide"}


def _design_links(task: TaskSpec, prepared_source: Any = None) -> set[tuple[str, str]]:
    from ..core.compiler import _ORDER_TO_BIOTITE
    design_ids = {str(chain.chain_id) for chain in task.design_chains}
    covalent_orders = set(_ORDER_TO_BIOTITE.values()) - {0, 8}
    result: set[tuple[str, str]] = set()
    for bond in task.condition.get("bond", ()) or ():
        if not isinstance(bond, Mapping):
            continue  # The compiler owns malformed topology diagnostics.
        order = str(bond.get("order", "any")).strip().lower()
        if _ORDER_TO_BIOTITE.get(order) not in covalent_orders:
            continue
        left, right = bond.get("atom1"), bond.get("atom2")
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            a, b = left.get("chain"), right.get("chain")
            if a in design_ids and b in design_ids and a != b:
                result.add(tuple(sorted((str(a), str(b)))))
    source = getattr(prepared_source, "source", None)
    if source is not None:
        for i, j, order in prepared_source.source_declared_bonds:
            if int(order) not in covalent_orders:
                continue
            a, b = str(source.chain_id[i]), str(source.chain_id[j])
            if a in design_ids and b in design_ids and a != b:
                result.add(tuple(sorted((a, b))))
    return result


def _peptide_unit(task: TaskSpec, prepared_source: Any = None) -> bool:
    peptides = _peptide_chains(task)
    if not peptides:
        return False
    other = {str(chain.chain_id) for chain in task.design_chains} - peptides
    if not other:
        return True
    # Only complete fixed chemistry connected through design-owned bonds can
    # be an auxiliary peptide fragment. A context path cannot join units.
    if not other <= _known_ligands(task):
        return False
    reached = set(peptides)
    links = _design_links(task, prepared_source)
    while True:
        expanded = reached | {b for a, b in links if a in reached} | {a for a, b in links if b in reached}
        if expanded == reached:
            break
        reached = expanded
    return other <= reached


def needs_routing_source(task: TaskSpec) -> bool:
    if getattr(task, "format", "normal") != "normal" or getattr(task, "input_path", None) is None:
        return False
    modalities = {chain.modality for chain in task.design_chains}
    if modalities == {"protein"}:
        return not _peptide_unit(task)
    # A source-declared attachment may prove a known peptide linker.
    peptides = _peptide_chains(task)
    others = {str(chain.chain_id) for chain in task.design_chains} - peptides
    return bool(peptides and others and others <= _known_ligands(task) and not _peptide_unit(task))


def prepare_routing_sources(tasks: Sequence[TaskSpec], *, ccd_path: Path | None = None) -> dict[str, Any]:
    """Prepare each needed source once; the native compiler consumes it later."""
    selected = [task for task in tasks if needs_routing_source(task)]
    if not selected:
        return {}
    from ..core.compiler import prepare_task_source
    sources: dict[str, Any] = {}
    for task in selected:
        try:
            sources[task.name] = prepare_task_source(task, ccd_path=ccd_path)
        except (DesignInputError, DesignSampleError, OSError) as error:
            # Preserve native candidate/input failure behavior. Uncertain source
            # preferences use Omni; compilation still reports the original error.
            sources[task.name] = error
    return sources


def _protein_coordinate_motif(task: TaskSpec, prepared_source: Any = None) -> bool:
    contigs = task.condition.get("contig", {}) or {}
    if not contigs or not isinstance(contigs, Mapping):
        return False
    from ..core.compiler import _SOURCE_SEGMENT, _source_residue_term, _source_selector
    source = getattr(prepared_source, "source", None)
    for chain in task.design_chains:
        if chain.modality != "protein":
            continue
        entry = contigs.get(chain.chain_id)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("token"), str):
            continue
        segments = [s.strip() for s in entry["token"].split("/") if _SOURCE_SEGMENT.fullmatch(s.strip())]
        if not segments:
            continue
        # Direct callers without a source snapshot still retain explicit authored
        # motif intent. Production runtime/check both provide the shared snapshot.
        if source is None:
            return True
        selected_uids: set[int] = set()
        for segment in segments:
            selected_uids.update(_source_residue_term(source, segment))
        partial = entry.get("polymer_atom") or {}
        names_by_uid = {}
        if isinstance(partial, Mapping):
            for selector, names in partial.items():
                uids = _source_selector(source, selector)
                if len(uids) == 1:
                    if names == "tip":
                        from ..utils.atoms import SIDECHAIN_TIP_ATOMS
                        rows = source.residue_uid == uids[0]
                        names = SIDECHAIN_TIP_ATOMS.get(str(source.res_name[rows][0]), ())
                    if isinstance(names, (list, tuple)):
                        names_by_uid[int(uids[0])] = set(names)
        from ..utils.atoms import PROTEIN_RES_ATOMS
        import numpy as np
        carrier_uids = set(prepared_source.source_carrier_candidates)
        for uid in selected_uids:
            residue_rows = source.residue_uid == uid
            rows = residue_rows & source.coord_valid
            if int(uid) not in carrier_uids:
                residue_name = str(source.res_name[residue_rows][0])
                inventory = PROTEIN_RES_ATOMS.get(residue_name, ("N", "CA", "C", "O"))
                rows &= np.isin(source.atom_name, list(inventory))
            names = names_by_uid.get(int(uid))
            if names is not None:
                import numpy as np
                rows &= np.isin(source.atom_name, list(names))
            if rows.any():
                return True
    return False


def _retained_ligand_context(prepared_source: Any) -> bool:
    source = getattr(prepared_source, "source", None)
    if source is None:
        return False
    import numpy as np
    rows = (~source.is_design) & (source.mol_type == "ligand")
    # Atomized modified polymer carriers retain their polymer ownership.
    carriers = set(prepared_source.source_carrier_candidates)
    if carriers:
        rows &= ~np.isin(source.residue_uid, list(carriers))
    return bool(rows.any())


def task_model_preference(task: TaskSpec, *, prepared_source: Any = None) -> TaskModelPreference:
    if not task.design_chains:
        raise DesignConfigError(f"task {task.name!r} has no design chains for model routing")
    if task.format == "antibody":
        return TaskModelPreference(task.name, "antibody", "isolated antibody format")
    modalities = {chain.modality for chain in task.design_chains}
    unknown = modalities - set(_MEMBERS)
    if unknown:
        raise DesignConfigError(f"unsupported design modality: {sorted(unknown)!r}")
    if _peptide_unit(task, prepared_source):
        return TaskModelPreference(task.name, "pep", "explicit peptide design, including fixed covalent auxiliary fragments")
    if len(modalities) > 1:
        return TaskModelPreference(task.name, "omni", "independent cross-modality joint design")
    if _peptide_chains(task):
        return TaskModelPreference(task.name, "omni", "joint peptide and ordinary protein design roles")
    modality = next(iter(modalities))
    if modality == "protein":
        if isinstance(prepared_source, Exception):
            return TaskModelPreference(task.name, "omni", "source unavailable; native compiler reports the input failure")
        try:
            has_motif = _protein_coordinate_motif(task, prepared_source)
        except (DesignInputError, DesignSampleError, OSError, IndexError):
            return TaskModelPreference(task.name, "omni", "motif preparation unresolved; native compiler validates it")
        if has_motif:
            return TaskModelPreference(task.name, "omni", "coordinate motif on a designed protein")
        if _retained_ligand_context(prepared_source):
            return TaskModelPreference(task.name, "omni", "protein design with retained ligand context")
    return TaskModelPreference(task.name, _MEMBERS[modality], f"{modality}-only design roles")


def task_model_preferences(tasks: Sequence[TaskSpec], *, prepared_sources: Mapping[str, Any] | None = None,
                           ccd_path: Path | None = None) -> tuple[TaskModelPreference, ...]:
    if not tasks:
        raise DesignConfigError("model routing requires at least one task")
    formats = {task.format for task in tasks}
    if formats not in ({"normal"}, {"antibody"}):
        raise DesignConfigError("cannot mix antibody and normal task formats")
    sources = dict(prepared_sources or {})
    missing = [task for task in tasks if task.name not in sources]
    sources.update(prepare_routing_sources(missing, ccd_path=ccd_path))
    return tuple(task_model_preference(task, prepared_source=sources.get(task.name)) for task in tasks)


def select_model_variant(tasks: Sequence[TaskSpec], *, prepared_sources: Mapping[str, Any] | None = None,
                         ccd_path: Path | None = None) -> str:
    preferences = task_model_preferences(tasks, prepared_sources=prepared_sources, ccd_path=ccd_path)
    members = {entry.member for entry in preferences}
    return next(iter(members)) if len(members) == 1 else "omni"


def resolve_model_route(tasks: Sequence[TaskSpec], *, enabled: bool, override: str | None,
                        prepared_sources: Mapping[str, Any] | None = None,
                        ccd_path: Path | None = None) -> tuple[str | None, str, str]:
    if {task.format for task in tasks} not in ({"normal"}, {"antibody"}):
        raise DesignConfigError("cannot mix antibody and normal task formats")
    if not enabled:
        return None, "direct", "route=false; load the exact standalone checkpoint"
    if override is not None:
        if override == "antibody" and {task.format for task in tasks} != {"antibody"}:
            raise DesignConfigError("route_ckpt=antibody requires format: antibody; normal tasks do not provide the antibody template feature ABI")
        return override, "forced", "route_ckpt explicitly selects this branch"
    preferences = task_model_preferences(tasks, prepared_sources=prepared_sources, ccd_path=ccd_path)
    members = {entry.member for entry in preferences}
    selected = next(iter(members)) if len(members) == 1 else "omni"
    if len(members) == 1:
        reason = f"all selected tasks prefer {selected}"
    else:
        reason = f"mixed task preferences {','.join(sorted(members))}; one submission uses omni"
    return selected, "auto", reason
