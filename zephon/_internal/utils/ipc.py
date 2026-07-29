# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""IPC transport helpers for multiprocessing queues.

AF_UNIX socketpair alternative to the ``os.pipe()`` under
``multiprocessing.Queue``. Unlike pipes, socket buffers are not charged
against the shared per-UID pipe budget (``fs.pipe-user-pages-soft``; once
exhausted, new pipes silently drop from 64 KiB to 8 KiB capacity) and are
sized per-socket with an unprivileged ``setsockopt``.
"""

from __future__ import annotations

import array
import fcntl
import socket
import sys
import termios
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zephon.options import IpcTransport

DEFAULT_IPC_TRANSPORT: IpcTransport = "socketpair"

# See RuntimeOptions.ipc_buffer_bytes / mtp_buffer_bytes for semantics.
DEFAULT_IPC_BUFFER_BYTES = 512 * 1024
DEFAULT_MTP_BUFFER_BYTES = 8 * 1024 * 1024

# Requested sizes already reported, so a fleet of queues warns once.
_clamp_warned: set[int] = set()


def _warn_buffer_shortfall(granted: int, requested: int, exc: OSError | None) -> None:
    if requested in _clamp_warned:
        return
    _clamp_warned.add(requested)
    cause = f"rejected ({exc})" if exc is not None else f"clamped to {granted} bytes"
    print(
        f"[zephon] socketpair buffer request of {requested} bytes was "
        f"{cause} by the kernel (Linux reports granted sizes with "
        f"bookkeeping overhead included). Raise net.core.wmem_max and "
        f"net.core.rmem_max (Linux) or kern.ipc.maxsockbuf (macOS) to "
        f"honor larger requests.",
        file=sys.stderr,
        flush=True,
    )


def readable_bytes(fd: int) -> int:
    """Bytes queued for reading on ``fd`` (``FIONREAD``); works for pipes and sockets."""
    buf = array.array("i", [0])
    fcntl.ioctl(fd, termios.FIONREAD, buf)
    return buf[0]


def socketpair_connections(
    buffer_bytes: int | None = None,
) -> tuple[Connection, Connection]:
    """Return a socketpair-backed ``(reader, writer)`` Connection pair.

    Mirrors ``multiprocessing.connection.Pipe``: ``duplex=False`` semantics
    over the socketpair transport of ``duplex=True``.

    ``buffer_bytes`` requests kernel buffering, best-effort: Linux clamps to
    ``2 * net.core.wmem_max``, macOS honors it up to ``kern.ipc.maxsockbuf``.
    Both ends are sized because in-flight bytes count against the sender's
    ``SO_SNDBUF`` on Linux but the receiver's ``SO_RCVBUF`` on macOS (8 KiB
    default there).  A request the kernel won't fully grant is reported to
    stderr once per size.
    """
    read_sock, write_sock = socket.socketpair()
    if buffer_bytes is not None:
        setsockopt_exc: OSError | None = None
        try:
            write_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_bytes)
            read_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_bytes)
        except OSError as exc:
            # Linux and macOS silently clamp oversized requests (caught by
            # the readback below); BSD kernels may reject with ENOBUFS.
            setsockopt_exc = exc
        granted = min(
            write_sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF),
            read_sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF),
        )
        if granted < buffer_bytes:
            _warn_buffer_shortfall(granted, buffer_bytes, setsockopt_exc)
    # Undo socket.setdefaulttimeout() (it sets O_NONBLOCK): Connection does
    # raw os.read/os.write with no EAGAIN handling. Mirrors stdlib Pipe().
    read_sock.setblocking(True)
    write_sock.setblocking(True)
    reader = Connection(read_sock.detach(), writable=False)
    writer = Connection(write_sock.detach(), readable=False)
    return reader, writer
