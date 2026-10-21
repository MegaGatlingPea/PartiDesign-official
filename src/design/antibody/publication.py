"""Compatibility wrapper for antibody result publication.

Antibody output is represented by the normal authoritative CIF and one
primitive result row. FASTA and JSON sidecars are deliberately not written.
"""

from .results import publish_antibody_outputs

__all__ = ["publish_antibody_outputs"]
