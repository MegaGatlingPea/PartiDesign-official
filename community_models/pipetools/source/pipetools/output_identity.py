from __future__ import annotations

from typing import Any


def rewrite_output_entity_identity(
    block: Any,
    spec: dict[str, Any],
    pdbx: Any,
    np: Any,
) -> None:
    """Restore the private label-asym/entity ledger in a model output CIF.

    Model writers are allowed to choose their own internal numeric entity IDs,
    but PipeTools outputs are not.  The private model-neutral input is the
    authority for exact ``label_asym_id`` order, variant-aware entity grouping,
    and the public ``label_entity_id`` values.  This helper writes all entity
    categories from that authority without consulting author-chain fields.
    """

    sequences = spec.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise RuntimeError("Output entity rewrite lacks a private sequence ledger")
    chain_to_entry: dict[str, dict[str, Any]] = {}
    for entry in sequences:
        if not isinstance(entry, dict):
            raise RuntimeError("Output entity entry is malformed")
        chain_id = entry.get("chain_id")
        output_entity = entry.get("output_entity_id")
        if (
            not isinstance(chain_id, str)
            or not chain_id
            or chain_id in chain_to_entry
            or not isinstance(output_entity, str)
            or not output_entity
        ):
            raise RuntimeError("Output entity identity is malformed")
        chain_to_entry[chain_id] = entry

    if "atom_site" not in block:
        raise RuntimeError("Output CIF lacks atom_site")
    atom_site = block["atom_site"]
    label_asym = atom_site["label_asym_id"].as_array(str)
    observed_chains = list(dict.fromkeys(map(str, label_asym)))
    if observed_chains != list(chain_to_entry):
        raise RuntimeError("Output changed label_asym_id before entity rewrite")
    entity_values = np.array(
        [chain_to_entry[str(chain)]["output_entity_id"] for chain in label_asym],
        dtype=np.str_,
    )
    atom_site["label_entity_id"] = pdbx.CIFColumn(
        data=pdbx.CIFData(array=entity_values)
    )

    block["struct_asym"] = _cif_category(
        pdbx,
        np,
        [
            {"id": chain_id, "entity_id": entry["output_entity_id"]}
            for chain_id, entry in chain_to_entry.items()
        ],
    )

    entries_by_entity: dict[str, list[dict[str, Any]]] = {}
    for entry in sequences:
        entries_by_entity.setdefault(entry["output_entity_id"], []).append(entry)
    entity_rows: list[dict[str, str]] = []
    entity_poly_rows: list[dict[str, str]] = []
    entity_poly_seq_rows: list[dict[str, str]] = []
    label_seq = atom_site["label_seq_id"].as_array(str)
    label_comp = atom_site["label_comp_id"].as_array(str)
    for entity_id, entries in entries_by_entity.items():
        modality = entries[0].get("modality")
        if any(entry.get("modality") != modality for entry in entries):
            raise RuntimeError("One output entity mixes modalities")
        is_polymer = modality in {"protein", "rna", "dna"}
        entity_rows.append(
            {
                "id": entity_id,
                "type": "polymer" if is_polymer else "non-polymer",
                "pdbx_description": f"PipeTools {modality} entity {entity_id}",
            }
        )
        if not is_polymer:
            continue
        sequence = entries[0].get("sequence")
        if not isinstance(sequence, str) or not sequence or any(
            entry.get("sequence") != sequence for entry in entries
        ):
            raise RuntimeError("One output polymer entity has multiple sequences")
        polymer_type = {
            "protein": "polypeptide(L)",
            "rna": "polyribonucleotide",
            "dna": "polydeoxyribonucleotide",
        }[modality]
        entity_poly_rows.append(
            {
                "entity_id": entity_id,
                "type": polymer_type,
                "pdbx_strand_id": ",".join(entry["chain_id"] for entry in entries),
                "pdbx_seq_one_letter_code_can": sequence,
            }
        )
        representative_chain = entries[0]["chain_id"]
        components: dict[int, str] = {}
        for row, observed_chain in enumerate(label_asym):
            if str(observed_chain) != representative_chain:
                continue
            try:
                position = int(str(label_seq[row]))
            except ValueError as exc:
                raise RuntimeError(
                    "Output polymer has non-numeric label_seq_id"
                ) from exc
            component = str(label_comp[row])
            if components.setdefault(position, component) != component:
                raise RuntimeError("Output residue has multiple component IDs")
        if set(components) != set(range(1, len(sequence) + 1)):
            raise RuntimeError("Output polymer residue coverage differs")
        entity_poly_seq_rows.extend(
            {
                "entity_id": entity_id,
                "num": str(position),
                "mon_id": components[position],
                "hetero": "n",
            }
            for position in range(1, len(sequence) + 1)
        )
    block["entity"] = _cif_category(pdbx, np, entity_rows)
    if entity_poly_rows:
        block["entity_poly"] = _cif_category(pdbx, np, entity_poly_rows)
        block["entity_poly_seq"] = _cif_category(
            pdbx, np, entity_poly_seq_rows
        )
    else:
        for category_name in ("entity_poly", "entity_poly_seq"):
            if category_name in block:
                del block[category_name]


def _cif_category(pdbx: Any, np: Any, rows: list[dict[str, str]]) -> Any:
    if not rows:
        raise RuntimeError("Cannot construct an empty CIF category")
    columns = {
        name: pdbx.CIFColumn(
            data=pdbx.CIFData(
                array=np.array([row[name] for row in rows], dtype=np.str_)
            )
        )
        for name in rows[0]
    }
    return pdbx.CIFCategory(columns=columns)


__all__ = ["rewrite_output_entity_identity"]
