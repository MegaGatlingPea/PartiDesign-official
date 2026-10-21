"""Deny IP networking inside PartiDesign community-model workers.

Unix-domain sockets remain available for local runtime internals.  The reward
stack receives every model, checkpoint, CCD snapshot, and sequence input from
explicit local paths; an AF_INET/AF_INET6 connection is therefore always a
contract violation rather than a recoverable fallback.
"""

from __future__ import annotations

import socket as _socket


_Socket = _socket.socket


class _OfflineSocket(_Socket):
    def connect(self, address):  # type: ignore[no-untyped-def]
        if self.family in {_socket.AF_INET, _socket.AF_INET6}:
            raise OSError("PartiDesign community-model workers are strictly offline")
        return super().connect(address)

    def connect_ex(self, address):  # type: ignore[no-untyped-def]
        if self.family in {_socket.AF_INET, _socket.AF_INET6}:
            raise OSError("PartiDesign community-model workers are strictly offline")
        return super().connect_ex(address)


def _offline_create_connection(*args, **kwargs):  # type: ignore[no-untyped-def]
    del args, kwargs
    raise OSError("PartiDesign community-model workers are strictly offline")


_socket.socket = _OfflineSocket
_socket.create_connection = _offline_create_connection
