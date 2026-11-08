"""Lightweight descriptions and honestly labelled resource footprint references.

Reference footprints include shared files and are not download-byte budgets or
integrity standards. Only a downloader asset's explicit expected_bytes/sha256
can establish a trusted size or checksum comparison.
"""
from __future__ import annotations


_REFERENCE_LABEL = "approx. installed footprint; shared files included; varies"

_SHORT_DESCRIPTIONS = {
    "partidesign": "Generates molecular structures and sequences from tasks.",
    "partiscore": "Ranks molecular interfaces from supplied structures.",
    "proteinmpnn": "Redesigns protein sequences for a fixed backbone.",
    "solublempnn": "Designs soluble-protein sequences for a fixed backbone.",
    "ligandmpnn": "Designs protein sequences around ligand context.",
    "cyclicmpnn": "Redesigns sequences for cyclic peptide backbones.",
    "nampnn": "Redesigns DNA and RNA sequences for fixed structures.",
    "esmfold2-fast": "Iterates designs using shared ESM-C safetensors.",
    "esmfold2-standard": "Validates with MSA and shared ESM-C safetensors.",
    "af2": "Runs AF2-IG final validation for PPI designs.",
    "mol1": "Optionally predicts molecular complex structures.",
    "boltz2": "Predicts protein–ligand structures and binding affinity.",
    "af3": "Optionally predicts full-complex structures.",
    "af3-la": "Optionally predicts complexes with AF3-LA.",
    "chemistry": "Provides molecular definitions for structure workflows."
}

_TARGET_INFO = {
    "partidesign": ("PartiDesign generation bundle (model, autoencoder and experiment configuration).",
                   ("design", "pipeline"), 4_909_825_762, "exact published checkpoint size",
                   "Required for generation and partial diffusion; Apache-2.0 project HF release."),
    "partiscore": ("PartiScore molecular-interface scorer; independent of generation weights.",
                  ("score", "validate --score", "design (configured scoring/reward)", "pipeline (configured scoring)"),
                  25_350_931, "exact published checkpoint size", "Apache-2.0 project HF release."),
    "proteinmpnn": ("ProteinMPNN vanilla and CA protein-sequence redesign weights.",
                    ("design (sequence search/reward)", "pipeline", "validate (sequence source)"),
                    46_597_237, _REFERENCE_LABEL, "Seven registered parameter files; select this sequence source explicitly."),
    "solublempnn": ("SolubleMPNN sequence-redesign weights for soluble proteins and peptides.",
                    ("design (sequence search/reward)", "pipeline", "validate (sequence source)"),
                    26_725_204, _REFERENCE_LABEL, "Four registered parameter files; ordinary protein fold validation uses this default sequence source."),
    "ligandmpnn": ("LigandMPNN protein sequence redesign with ligand context and side-chain packing weights.",
                   ("design (sequence search/reward)", "pipeline (selected sequence source)", "validate (sequence source)"),
                   56_527_151, _REFERENCE_LABEL, "Five registered parameter files; use when its context-aware redesign is selected."),
    "cyclicmpnn": ("CyclicMPNN sequence-redesign assets supplied by the source checkout.",
                   ("design (sequence search/reward)", "pipeline (selected sequence source)", "validate (sequence source)"),
                   13_302_187, _REFERENCE_LABEL, "Two Git-maintained curated files; restore missing files from the checkout rather than download."),
    "nampnn": ("NA-MPNN sequence design for nucleic-acid tasks.",
               ("design (sequence search/reward)", "pipeline (nucleic-acid sequence source)", "validate (sequence source)"),
               27_667_482, _REFERENCE_LABEL, "The download target is nampnn; nucleic-acid fold validation uses it as its default sequence source."),
    "esmfold2-fast": ("ESMFold2-fast structure prediction with shared ESMC-6B and chemistry assets.",
                      ("pipeline (iteration)", "validate (selected method)", "screen (selected method)", "design (configured fold reward/search)"),
                      27_146_137_588, _REFERENCE_LABEL, "Default ordinary iteration backend; shares ESMC-6B/CCD files with standard, so footprints must not be summed as new download size."),
    "esmfold2-standard": ("ESMFold2-standard structure prediction with fixed protein-context MSA support.",
                          ("pipeline (ordinary final)", "validate (selected method)", "screen (selected method)", "design (configured fold reward/search)"),
                          27_330_225_891, _REFERENCE_LABEL, "Default ordinary final; explicit PPI instead uses AF2-IG. Shares ESMC-6B/CCD files with fast."),
    "af2": ("AlphaFold2 initial-guess (AF2-IG) structure-evaluation parameters.",
            ("pipeline (PPI final)", "validate (af2 method)", "design (configured fold reward/search)"),
            5_587_937_914, _REFERENCE_LABEL, "Fifteen registered parameter files; the PPI preset selects models 1 and 2 with matched complex/monomer predictions."),
    "boltz2": ("Boltz-2 optional protein–ligand structure and affinity weights with molecule library.",
               ("validate --method boltz2", "pipeline (ligand default)", "screen (boltz2 method)"), 8_025_061_538,
               "weights, archive and extracted molecule library",
               "Default ligand pipeline backend; optional in validate and screen. Official MIT release; shared ColabFold MSA/templates."),
    "mol1": ("Mol-1 optional biomolecular structure-evaluation assets.",
             ("validate (mol1 method)", "pipeline (explicit custom method)", "screen (selected method)", "design (configured fold reward/search)"),
             3_099_999_035, _REFERENCE_LABEL, "Optional backend; not required by the shipped default final routes."),
    "af3": ("AlphaFold3 optional structure-evaluation weights and shared derived CCD assets.",
            ("validate (af3 method)", "pipeline (explicit custom method)", "screen (selected method)", "design (configured fold reward/search)"),
            2_175_129_772, _REFERENCE_LABEL, "Optional, upstream parameter/output terms apply. Shares CCD files with af3-la; derived CCD size can vary."),
    "af3-la": ("AlphaFold3-LA optional structure-evaluation variant with shared AF3 chemistry assets.",
               ("validate (af3-la method)", "pipeline (explicit custom method)", "screen (selected method)", "design (configured fold reward/search)"),
               2_175_093_308, _REFERENCE_LABEL, "Optional variant; shares CCD files with af3. Derived CCD size can vary."),
    "chemistry": ("Bundled chemical-component dictionary used by molecular structure workflows.",
                  ("design", "pipeline", "score", "validate", "relax", "postfilter"),
                  9_575_765, _REFERENCE_LABEL, "Packaged static chemistry resource; environment/native tools are checked separately."),
}


def format_bytes(value) -> str:
    """Format actual or reference byte counts; do not invent unknown sizes."""
    if value is None:
        return "unknown"
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return "unknown"
    if count < 0:
        return "unknown"
    if count < 1024:
        return f"{count} B"
    amount = float(count)
    for unit in ("KiB", "MiB", "GiB", "TiB", "PiB"):
        amount /= 1024
        if amount < 1024 or unit == "PiB":
            return f"{amount:.2f} {unit}"
    return "unknown"


def target_metadata(name: str) -> dict[str, object]:
    """Return fresh metadata containers, including for synthetic/unknown targets."""
    data = _TARGET_INFO.get(name)
    if data is None:
        return {"target": name, "description": "Resource target.", "short_description": "Resource", "used_by": [],
                "size_hint_bytes": None, "size_hint_label": "unknown", "notes": ""}
    description, used_by, size, label, notes = data
    return {"target": name, "description": description,
            "short_description": _SHORT_DESCRIPTIONS[name], "used_by": list(used_by),
            "size_hint_bytes": size, "size_hint_label": label, "notes": notes}


def target_catalog() -> list[dict[str, object]]:
    """Static help catalog: importing it performs no file/network/model checks."""
    from .download import TARGETS
    return [target_metadata(name) for name in TARGETS]


__all__ = ["format_bytes", "target_metadata", "target_catalog"]
