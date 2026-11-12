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
import math
import select
import socket
import struct
import sys
import termios
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zephon.options import IpcTransport

DEFAULT_IPC_TRANSPORT: IpcTransport = "socketpair"

# See RuntimeOptions.ipc_buffer_bytes / mtp_buffer_bytes for semantics.
DEFAULT_IPC_BUFFER_BYTES = 512 * 1024
DEFAULT_MTP_BUFFER_BYTES = 8 * 1024 * 1024

# Requested sizes already reported, so a fleet of queues warns once.
_clamp_warned: set[int] = set()
_LENGTH = struct.Struct("!i")


class QueueTransportError(RuntimeError):
    """A queue stream is no longer safe to reuse."""


def _read_exact(sock: socket.socket, size: int) -> bytes:
    if not size:
        return b""
    result = sock.recv(size, socket.MSG_WAITALL)
    if len(result) == size:
        return result
    chunks = []
    while len(result) < size:
        if not result:
            raise QueueTransportError("Queue ended inside a message")
        chunks.append(result)
        size -= len(result)
        result = sock.recv(size, socket.MSG_WAITALL)
    chunks.append(result)
    return b"".join(chunks)


def read_frame(
    sock: socket.socket, *, nonblocking_start: bool = False, fd_frame: Any = None
) -> Any:
    """Read one native frame, optionally delegating an FD extension to its decoder.

    nonblocking_start returns None only if the first read consumes no bytes.
    After any progress the reader must finish the frame or fail the stream;
    returning an empty-queue result would lose the next frame boundary.
    """
    try:
        header = sock.recv(4, socket.MSG_DONTWAIT if nonblocking_start else 0)
    except BlockingIOError:
        if nonblocking_start:
            return None  # No bytes consumed; the queue can wait outside its lock.
        raise
    if not header:
        raise QueueTransportError("Queue ended inside a message")
    if len(header) < 4:
        header += _read_exact(sock, 4 - len(header))
    size = _LENGTH.unpack(header)[0]
    if size == -2 and fd_frame is not None:
        return fd_frame(sock)
    if size == -1:
        size = struct.unpack("!Q", _read_exact(sock, 8))[0]
    if size < 0 or size > sys.maxsize:
        raise QueueTransportError("Invalid queue frame length")
    return _read_exact(sock, size)


class SocketConnection(Connection):
    """Expose a socket through Connection while keeping a single FD owner.

    Connection's byte methods and the socket's ancillary methods use the same
    descriptor; only the socket closes it. A framed reader holds close_guard
    during a receive. Concurrent close requests are deferred until that reader
    unwinds, preventing descriptor reuse underneath an active socket operation.
    """

    def __init__(
        self, sock: socket.socket, *, readable: bool = True, writable: bool = True
    ) -> None:
        sock.setblocking(True)
        self.socket = sock
        self.close_guard: Any = None
        self.close_requested = False
        super().__init__(sock.fileno(), readable=readable, writable=writable)

    def _close(self) -> None:
        self.socket.close()

    def poll_readable(self, timeout: float | None = 0.0) -> bool:
        """Poll readability, including hangups/errors that the next read reports.

        Each caller owns its poll object so concurrent readers can wait without
        sharing a poll instance that rejects concurrent poll() calls.
        """
        poller = select.poll()
        poller.register(self.fileno(), select.POLLIN)
        milliseconds = None if timeout is None else max(0, math.ceil(timeout * 1000))
        return bool(poller.poll(milliseconds))

    def close(self) -> None:
        """Defer feeder close until an active framed reader has unwound."""
        self.close_requested = True
        if self.close_guard is None:
            super().close()
        elif self.close_guard.acquire(False):
            try:
                super().close()
            finally:
                self.close_guard.release()

    def __getstate__(self) -> tuple[socket.socket, bool, bool]:
        return self.socket, self.readable, self.writable

    def __setstate__(self, state: tuple[socket.socket, bool, bool]) -> None:
        sock, readable, writable = state
        self.__init__(sock, readable=readable, writable=writable)


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
    reader = SocketConnection(read_sock, writable=False)
    writer = SocketConnection(write_sock, readable=False)
    return reader, writer
