"""Pure source-residue mapping and design/context aggregation for ipSAE."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .ipsae import IPSAE_METRIC_NAMES, ipsae_pair

_POLYMERS = frozenset({"protein", "rna", "dna"})
_SCORE_NAMES = IPSAE_METRIC_NAMES[:3]
_MISSING = "ipSAE requires raw PAE and exact polymer residue token mapping"
_NO_PAIRS = "ipSAE has no eligible design/context polymer chain pair"


def initialize_ipsae(
    metrics: dict[str, Any], reasons: dict[str, str], provenance: dict[str, Any] | None,
    reference: Any, design_chains: Sequence[str], config: Mapping[str, Any],
) -> dict[str, Any]:
    """Initialize scores and their definition before confidence can be missing.

    The reference polymer occurrence groups are cached once per chain. All row
    selection later uses explicit source identity rather than PAE/CIF order.
    """
    from .metrics import _ordered_chains, _chain_mol_type, _residue_rows

    cutoff = float(config.get("pae_cutoff", 10.0))
    chains = _ordered_chains(reference)
    design = list(dict.fromkeys(str(chain) for chain in design_chains))
    context = [chain for chain in chains if chain not in design]
    kinds = {chain: _chain_mol_type(reference, chain) for chain in chains}
    residues = {chain: _residue_rows(reference, chain, polymer_only=True)
                for chain in chains if kinds[chain] in _POLYMERS}
    pairs = []
    for left in design:
        for right in context:
            eligible = kinds.get(left) in _POLYMERS and kinds.get(right) in _POLYMERS
            pair_type = ("nucleic_acid" if kinds.get(left) in {"rna", "dna"}
                         or kinds.get(right) in {"rna", "dna"} else "protein") if eligible else None
            pairs.append({"design_chain": left, "context_chain": right, "pair_type": pair_type})
    eligible = [pair for pair in pairs if pair["pair_type"] is not None]
    definition = {"implementation": "DunbrackLab/IPSAE v4 d0res", "pae_cutoff": cutoff,
                  "aggregation": "max_design_context_pair",
                  "chain_pairs": [[pair["design_chain"], pair["context_chain"]] for pair in eligible],
                  "pair_types": [pair["pair_type"] for pair in eligible]}
    for name in IPSAE_METRIC_NAMES:
        metrics[name] = cutoff if name == "ipsae_pae_cutoff" else None
        if name == "ipsae_pae_cutoff":
            reasons.pop(name, None)
        else:
            reasons[name] = _MISSING if eligible else _NO_PAIRS
        if provenance is not None:
            provenance[name] = {**definition,
                "chain_pairs": [list(pair) for pair in definition["chain_pairs"]],
                "pair_types": list(definition["pair_types"])}

    # A polymer source chain may also retain heterogens. Their exact source
    # locators can exclude them without mistaking an empty anchor for a ligand.
    categories = reference.get_annotation_categories()
    chain_values = np.asarray(reference.label_asym_id if "label_asym_id" in categories
                              else reference.chain_id).astype(str)
    nonpolymer: dict[str, set[tuple[int, str]]] = {chain: set() for chain in chains}
    for row in np.flatnonzero(~np.isin(np.asarray(reference.mol_type).astype(str), tuple(_POLYMERS))):
        nonpolymer[str(chain_values[row])].add((int(reference.res_id[row]), str(reference.ins_code[row])))
    locator_index = {}
    number_index = {}
    for chain, groups in residues.items():
        locators: dict[tuple[int, str], list[int | None]] = {}
        numbers: dict[int, list[int | None]] = {}
        for index, group in enumerate(groups):
            locators.setdefault((group["res_id"], group["ins_code"]), []).append(index)
            numbers.setdefault(group["res_id"], []).append(index)
        for number, insertion in nonpolymer[chain]:
            locators.setdefault((number, insertion), []).append(None)
            numbers.setdefault(number, []).append(None)
        locator_index[chain], number_index[chain] = locators, numbers
    return {"pae_cutoff": cutoff, "pairs": pairs, "chain_types": kinds,
            "residues": residues, "nonpolymer_locators": nonpolymer,
            "locator_index": locator_index, "number_index": number_index}


def _metadata(prediction: Mapping[str, Any], payload: Mapping[str, Any], name: str) -> Any:
    value = prediction.get(name)
    return payload.get(name) if value is None else value


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"ipSAE {name} must match the PAE token axis")
    return array


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, str) and value and value.strip() == value:
        digits = value[1:] if value[:1] in {"+", "-"} else value
        if digits.isascii() and digits.isdigit():
            return int(value)
    raise ValueError(f"ipSAE {name} must be an integer source identifier")


def _insertion(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("ipSAE source insertion code must be a string")
    return value


def _source_locator(item: Any, chain: str) -> tuple[int, str]:
    if not isinstance(item, Mapping):
        raise ValueError("ipSAE source token residue entry is not a source locator")
    for primary, alias in (("chain_id", "source_chain_id"), ("res_id", "source_res_id"),
                           ("ins_code", "source_ins_code")):
        if primary in item and alias in item and str(item[primary]) != str(item[alias]):
            raise ValueError("ipSAE source token residue locator is inconsistent")
    source_chain = item.get("chain_id", item.get("source_chain_id"))
    if not isinstance(source_chain, str) or source_chain != chain:
        raise ValueError(f"ipSAE source token residue chain differs from {chain!r}")
    residue = _integer(item.get("res_id", item.get("source_res_id")), "source residue number")
    insertion = _insertion(item.get("ins_code", item.get("source_ins_code")))
    return residue, insertion


def _token_mapping(
    prediction: Mapping[str, Any], payload: Mapping[str, Any], size: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    anchors = _metadata(prediction, payload, "token_residue_atom_names")
    if anchors is not None:
        anchors = _vector(anchors, size, "residue-anchor labels")
        if any(not isinstance(value, str) or value not in {"", "CA", "C1'", "C1*"}
               for value in anchors.tolist()):
            raise ValueError("ipSAE residue-anchor labels must be CA, C1', or empty strings")
        result["anchors"] = np.asarray(["C1'" if value == "C1*" else value for value in anchors])
    else:
        result["anchors"] = None
    source = _metadata(prediction, payload, "source_token_residue_map")
    ordinal = _metadata(prediction, payload, "residue_index")
    residue = _metadata(prediction, payload, "token_res_ids")
    insertions = _metadata(prediction, payload, "token_ins_codes")
    if source is not None:
        result["mode"] = "source"
        result["source"] = _vector(source, size, "source token residue map")
    elif ordinal is not None:
        result["mode"] = "ordinal"
        result["ordinal"] = _vector(ordinal, size, "observed residue indices")
    elif residue is not None:
        result["mode"] = "locator"
    else:
        raise ValueError("ipSAE PAE tokens have no exact source residue mapping")
    # Source occurrence records take priority over weaker native identifiers.
    # In ordinal mode, source locators are needed only to identify heterogens
    # outside the explicit observed polymer index range.
    if result["mode"] != "source" and residue is not None:
        result["residue"] = residue
        result["insertions"] = insertions
        result["size"] = size
        if result["mode"] == "locator":
            result["residue"] = _vector(residue, size, "source residue numbers")
            result["insertions"] = None if insertions is None else _vector(insertions, size, "source insertion codes")
    return result


def _locator_candidates(
    chain: str, row: int, mapping: Mapping[str, Any], state: Mapping[str, Any],
) -> list[int | None]:
    residue_values = _vector(mapping["residue"], mapping["size"], "source residue numbers")
    residue = _integer(residue_values[row], "source residue number")
    insertions = mapping["insertions"]
    if insertions is None:
        return state["number_index"][chain].get(residue, [])
    insertion_values = _vector(insertions, mapping["size"], "source insertion codes")
    insertion = _insertion(insertion_values[row])
    return state["locator_index"][chain].get((residue, insertion), [])


def _representative_rows(
    chain: str, rows: np.ndarray, mapping: Mapping[str, Any], state: Mapping[str, Any],
) -> np.ndarray:
    groups = state["residues"][chain]
    if not groups:
        raise ValueError(f"ipSAE chain {chain!r} has no observed polymer residues")
    by_residue: list[list[int]] = [[] for _ in groups]
    for raw_row in rows:
        row = int(raw_row)
        if mapping["mode"] == "source":
            item = mapping["source"][row]
            locator = _source_locator(item, chain)
            if locator in state["nonpolymer_locators"][chain]:
                if state["locator_index"][chain][locator] != [None]:
                    raise ValueError(f"ipSAE chain {chain!r} has an ambiguous polymer/heterogen source locator")
                continue
            index = _integer(item.get("source_observed_index"), "source observed residue index")
            if not 0 <= index < len(groups) or locator != (groups[index]["res_id"], groups[index]["ins_code"]):
                raise ValueError(f"ipSAE chain {chain!r} source occurrence differs from the reference")
        elif mapping["mode"] == "ordinal":
            index = _integer(mapping["ordinal"][row], "observed residue index")
            if not 0 <= index < len(groups):
                if "residue" in mapping and _locator_candidates(chain, row, mapping, state) == [None]:
                    continue
                raise ValueError(f"ipSAE chain {chain!r} observed residue index is outside the reference polymer")
        else:
            candidates = _locator_candidates(chain, row, mapping, state)
            if len(candidates) != 1:
                raise ValueError(f"ipSAE chain {chain!r} source residue locator is missing or ambiguous")
            index = candidates[0]
            if index is None:
                continue
        by_residue[index].append(row)
    selected = []
    expected = "CA" if state["chain_types"][chain] == "protein" else "C1'"
    anchors = mapping["anchors"]
    for index, owned in enumerate(by_residue):
        if not owned:
            raise ValueError(f"ipSAE chain {chain!r} lacks a PAE token for observed polymer residue {index}")
        if anchors is None:
            if len(owned) != 1:
                raise ValueError(f"ipSAE chain {chain!r} atomized residue {index} needs residue-anchor metadata")
            representatives = owned
        else:
            representatives = [row for row in owned if anchors[row] == expected]
            if len(representatives) != 1:
                raise ValueError(f"ipSAE chain {chain!r} residue {index} needs exactly one {expected} anchor token")
        selected.append(representatives[0])
    return np.asarray(selected, dtype=np.int64)


def _context_is_omitted(positions: Mapping[str, Any], chain: str) -> bool:
    raw = positions.get(chain) if isinstance(positions, Mapping) else None
    if raw is None:
        return True
    if isinstance(raw, (list, tuple)):
        return not raw
    return isinstance(raw, np.ndarray) and raw.shape == (0,)


def compute_ipsae(
    state: Mapping[str, Any], matrix: Any, positions: Mapping[str, Any],
    payload: Mapping[str, Any], prediction: Mapping[str, Any],
    metrics: dict[str, Any], reasons: dict[str, str],
) -> None:
    """Score exact residue representatives and retain every design/context pair.

    ``positions`` already maps source chains to exact PAE token rows. This
    function never derives token identity from file order or chain lengths.
    """
    for name in _SCORE_NAMES:
        metrics[name] = None
    failure = None
    mapping = None
    pae = None
    try:
        if matrix is None:
            raise ValueError("ipSAE prediction confidence has no raw PAE matrix")
        pae = np.asarray(matrix)
        if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or pae.dtype.kind not in "iuf":
            raise ValueError("ipSAE PAE must be a square real numeric token matrix")
        if not np.isfinite(pae).all() or np.any(pae < 0):
            raise ValueError("ipSAE PAE must contain finite nonnegative values")
        mapping = _token_mapping(prediction, payload, pae.shape[0])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        failure = str(exc)

    row_cache: dict[str, np.ndarray] = {}
    chain_errors: dict[str, str] = {}
    if failure is None:
        for chain in state["residues"]:
            try:
                raw = positions.get(chain) if isinstance(positions, Mapping) else None
                if raw is None:
                    raise ValueError(f"ipSAE chain {chain!r} has no exact PAE token chain mapping")
                rows = np.asarray(raw)
                if (rows.ndim != 1 or not rows.size or rows.dtype.kind not in "iu"
                        or np.any(rows < 0) or np.any(rows >= pae.shape[0])
                        or np.unique(rows).size != rows.size):
                    raise ValueError(f"ipSAE chain {chain!r} PAE token rows are missing or ambiguous")
                row_cache[chain] = rows.astype(np.int64, copy=False)
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                chain_errors[chain] = str(exc)
        owners: dict[int, str] = {}
        for chain, rows in row_cache.items():
            for row in rows:
                previous = owners.setdefault(int(row), chain)
                if previous != chain:
                    reason = "ipSAE PAE token rows are assigned to multiple source chains"
                    chain_errors[chain] = chain_errors[previous] = reason
        for chain, rows in list(row_cache.items()):
            if chain not in chain_errors:
                try:
                    row_cache[chain] = _representative_rows(chain, rows, mapping, state)
                except (IndexError, KeyError, TypeError, ValueError) as exc:
                    chain_errors[chain] = str(exc)

    pair_rows = []
    complete = []
    unavailable = []
    for pair in state["pairs"]:
        left, right = pair["design_chain"], pair["context_chain"]
        result = {"design_chain": left, "context_chain": right, "ipsae": None,
                  "design_to_context": None, "context_to_design": None,
                  "status": "unavailable", "reason": None}
        if pair["pair_type"] is None:
            result.update(status="not_applicable", reason="ipSAE applies only to protein/RNA/DNA chain pairs")
        elif prediction.get("protocol") == "monomer" and _context_is_omitted(positions, right):
            result.update(status="not_applicable", reason="ipSAE monomer protocol omits the context chain")
        else:
            reason = failure or chain_errors.get(left) or chain_errors.get(right)
            if reason is None:
                try:
                    left_rows, right_rows = row_cache[left], row_cache[right]
                    values = ipsae_pair(pae[np.ix_(left_rows, right_rows)], pae[np.ix_(right_rows, left_rows)],
                                        pae_cutoff=state["pae_cutoff"],
                                        d0_min=2.0 if pair["pair_type"] == "nucleic_acid" else 1.0)
                    result.update(ipsae=values["symmetric"], design_to_context=values["forward"],
                                  context_to_design=values["reverse"], status="complete")
                    complete.append(result)
                except (IndexError, KeyError, TypeError, ValueError) as exc:
                    reason = str(exc)
            if reason is not None:
                result["reason"] = reason
                unavailable.append(result)
        pair_rows.append(result)
    metrics["ipsae_pairs"] = pair_rows
    if pair_rows:
        reasons.pop("ipsae_pairs", None)
    else:
        reasons["ipsae_pairs"] = _NO_PAIRS
    if complete and not unavailable:
        metrics["ipsae"] = max(pair["ipsae"] for pair in complete)
        metrics["ipsae_design_to_context"] = max(pair["design_to_context"] for pair in complete)
        metrics["ipsae_context_to_design"] = max(pair["context_to_design"] for pair in complete)
        for name in _SCORE_NAMES:
            reasons.pop(name, None)
    else:
        if unavailable:
            pair = unavailable[0]
            reason = f"ipSAE required pair {pair['design_chain']!r}/{pair['context_chain']!r} is unavailable: {pair['reason']}"
        elif any(pair["pair_type"] is not None for pair in state["pairs"]):
            reason = "ipSAE monomer protocol omits the context chain"
        else:
            reason = _NO_PAIRS
        for name in _SCORE_NAMES:
            reasons[name] = reason
