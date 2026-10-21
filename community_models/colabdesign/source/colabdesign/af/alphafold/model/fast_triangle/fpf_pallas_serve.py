"""Minimal carrier shim for the isolated ``+bias`` triangle spike.

The upstream bias module asks its serving adapter for the carried triangle
and attention helper modules.  This spike exercises triangle multiplication
only, so both helper slots point at the copied ``trimul_pallas`` module.
"""

from . import trimul_pallas


def kernels():
    return trimul_pallas, trimul_pallas
