# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Socket fault injection for descriptor-transfer tests."""

import errno
import socket
from typing import Any


class SocketSendFaults:
    """Inject quota rejection or short writes and count rights-bearing sends."""

    def __init__(
        self, sock: socket.socket, *, reject: bool = False, short: bool = False
    ) -> None:
        self.sock, self.reject, self.short = sock, reject, short
        self.rights_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sock, name)

    def sendmsg(self, buffers: Any, ancillary: Any, flags: int) -> int:
        if ancillary:
            self.rights_calls += 1
            if self.reject:
                raise OSError(errno.ETOOMANYREFS, "injected quota pressure")
        if self.short:
            buffers = [buffers[0][:3]]
        return self.sock.sendmsg(buffers, ancillary, flags)

    def sendall(self, *args: Any) -> None:
        return self.sock.sendall(*args)
