# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Pickle framing and CPU storage descriptor ownership for socketpair queues.

Ordinary messages have a native length prefix and pickle. A -2 prefix introduces
an FD subframe whose header carries SCM_RIGHTS, followed by optional legacy
tickets and the pickle. Only that subframe is read with recvmsg: an ordinary
read at a rights-bearing boundary would discard the descriptors.

Storage references in the pickle are indices into a message-local handle table.
The sender can replace direct FDs with tickets after an initial quota rejection
without invoking user reducers again. The receiver owns each delivered FD until
Torch detaches it for reconstruction or message cleanup closes it.
"""

from __future__ import annotations

import array
import errno
import io
import os
import pickle
import socket
import struct
import sys
from functools import partial
from multiprocessing.reduction import DupFd, ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Any

from zephon._internal.utils.ipc import (
    _LENGTH,
    QueueTransportError,
    _read_exact,
    read_frame,
)

MAX_DIRECT_FDS = 128
_FD_PREFIX = _LENGTH.pack(-2)
_HEADER = struct.Struct(
    "!4sIIQQ"
)  # magic, direct count, total count, table/payload bytes
_MAGIC = b"ZFD1"
_CONTROL_BYTES = socket.CMSG_SPACE(MAX_DIRECT_FDS * array.array("i").itemsize)
_SEND_FLAGS = getattr(socket, "MSG_NOSIGNAL", 0)
_RECV_FLAGS = getattr(socket, "MSG_CMSG_CLOEXEC", 0)


def _storage_from_slot(cls: type, index: int, size: int) -> Any:
    """Pickle marker resolved by the owning message's unpickler to rebuild()."""
    raise QueueTransportError("Storage slots require a queue message decoder")


class _FdTable:
    """Own duplicated descriptors and deduplicate storage backings in one pickle."""

    def __init__(self) -> None:
        self.fds: list[int] = []
        self.indices: dict[tuple[int, int], int] = {}

    def add(self, fd: int, key: tuple[int, int]) -> int:
        index = self.indices.get(key)
        if index is None:
            if not self.fds:
                register_after_fork(self, _FdTable.close)
            index = len(self.fds)
            self.fds.append(os.dup(fd))
            self.indices[key] = index
        return index

    def close(self) -> None:
        for index, fd in enumerate(self.fds):
            if fd >= 0:
                self.fds[index] = -1
                os.close(fd)

    def __del__(self) -> None:
        self.close()


class _LazyDispatch:
    """Snapshot the normal reducer registry on the first custom-type lookup.

    Builtin-only messages never need a dispatch table. Once materialized, all
    subsequent lookups use the normal dictionary directly in the C pickler.
    """

    def __init__(self, pickler: Any, message: EncodedMessage) -> None:
        self.pickler, self.message = pickler, message

    def __getitem__(self, cls: type) -> Any:
        dispatch = ForkingPickler._copyreg_dispatch_table.copy()
        dispatch.update(ForkingPickler._extra_reducers)
        torch = sys.modules.get("torch")
        reductions = sys.modules.get("torch.multiprocessing.reductions")
        if (
            torch is not None
            and reductions is not None
            and dispatch.get(torch.UntypedStorage) is reductions.reduce_storage
        ):
            dispatch[torch.UntypedStorage] = partial(
                _reduce_storage, self.message, reductions
            )
        self.pickler.dispatch_table = dispatch
        return dispatch[cls]


def _dump(value: Any, buffer: io.BytesIO, message: EncodedMessage) -> None:
    pickler = pickle.Pickler(buffer)
    pickler.dispatch_table = _LazyDispatch(pickler, message)
    try:
        pickler.dump(value)
    finally:
        pickler.dispatch_table = (
            ForkingPickler._copyreg_dispatch_table
        )  # Break the owner cycle.


def _reduce_storage(message: EncodedMessage, reductions: Any, storage: Any) -> tuple:
    mp = sys.modules["torch.multiprocessing"]
    if (
        storage.device.type != "cpu"
        or mp.get_sharing_strategy() != "file_descriptor"
        or storage.size() == 0
    ):
        return reductions.reduce_storage(storage)
    fd, size = storage._share_fd_cpu_()
    key = reductions.fd_id(fd)
    index = message.table.add(fd, key)
    reductions.shared_cache[key] = reductions.StorageWeakRef(storage)
    return _storage_from_slot, (type(storage), index, size)


def _send_remaining(sock: socket.socket, buffers: list[memoryview], sent: int) -> None:
    for buffer in buffers:
        if sent >= len(buffer):
            sent -= len(buffer)
        else:
            sock.sendall(buffer[sent:], _SEND_FLAGS)
            sent = 0


class EncodedMessage:
    """Pickled data plus the storage descriptors needed to reconstruct it.

    CPU storage reducers put owned FD duplicates in a lazy table and encode
    table indices in the pickle. send() chooses direct rights or legacy tickets
    for those indices; close() releases the duplicates after sending or failure.
    Messages without storage FDs use native framing and allocate no FD table.
    """

    def __init__(self, value: Any) -> None:
        self._table: _FdTable | None = None
        buffer = io.BytesIO()
        buffer.seek(4)  # Reserve native framing without copying the pickle later.
        try:
            _dump(value, buffer, self)
            self.buffer = buffer.getbuffer()
            self.payload = self.buffer[4:]
        except BaseException:
            self.close()
            raise

    @property
    def table(self) -> _FdTable:
        if self._table is None:
            self._table = _FdTable()
        return self._table

    def close(self) -> None:
        """Release only owned direct descriptors, never Torch's original handles."""
        if self._table is not None:
            self._table.close()

    def send(self, sock: socket.socket) -> None:
        """Send a frame, converting only its handle table on initial FD pressure."""
        if self._table is None or not self._table.fds:
            size = len(self.payload)
            if size > 0x7FFFFFFF:
                sock.sendall(_LENGTH.pack(-1) + struct.pack("!Q", size), _SEND_FLAGS)
                sock.sendall(self.payload, _SEND_FLAGS)
            else:
                _LENGTH.pack_into(self.buffer, 0, size)
                sock.sendall(self.buffer, _SEND_FLAGS)
            return
        assert self._table is not None
        # The native-size prefix never carries rights. Only the following
        # subframe uses recvmsg, so ordinary messages keep the native wire path.
        sock.sendall(_FD_PREFIX, _SEND_FLAGS)
        fds = self._table.fds
        direct = min(len(fds), MAX_DIRECT_FDS)
        tickets: dict[int, Any] = {}

        def prepare(first: int) -> list[memoryview]:
            # As with stock Queue, tickets left by a failed send live until
            # resource_sharer/process cleanup; there is no public cancel API.
            for index in range(first, len(fds)):
                if index not in tickets:
                    tickets[index] = DupFd(fds[index])
                    os.close(fds[index])
                    fds[index] = -1
            table = ForkingPickler.dumps(tickets) if tickets else b""
            header = _HEADER.pack(
                _MAGIC, first, len(fds), len(table), len(self.payload)
            )
            return [memoryview(header), memoryview(table), self.payload]

        buffers = prepare(direct)
        rights = (
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds[:direct]))]
            if direct
            else []
        )
        try:
            sent = sock.sendmsg(buffers, rights, _SEND_FLAGS)
        except OSError as exc:
            # Linux rejects the first rights-bearing buffer before enqueueing
            # anything. A positive short send must NEVER resend its rights.
            if exc.errno != errno.ETOOMANYREFS or not direct:
                raise
            buffers = prepare(0)  # Existing overflow tickets are reused.
            sent = sock.sendmsg(buffers, [], _SEND_FLAGS)
        if not sent:
            raise QueueTransportError("Zero-progress socket send")
        _send_remaining(sock, buffers, sent)


class ReceivedMessage:
    """Own a complete frame's bytes and FDs until storage reconstruction finishes.

    Receiving consumes the full wire transaction under the queue's read lock.
    decode() runs afterwards: unpickling may execute user code or contact a
    legacy ticket's sender, and must not prevent another consumer reading.
    """

    def __init__(self, sock: socket.socket) -> None:
        self.fds: list[int] = []
        self.storages: dict[int, tuple[type, int, Any]] = {}
        self.tickets: dict[int, Any] = {}
        register_after_fork(self, ReceivedMessage.close)
        try:
            header = bytearray()
            while len(header) < _HEADER.size:
                data, ancillary, flags, _ = sock.recvmsg(
                    _HEADER.size - len(header), _CONTROL_BYTES, _RECV_FLAGS
                )
                unexpected = False
                for level, kind, raw in ancillary:
                    if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                        received = array.array("i")
                        received.frombytes(
                            raw[: len(raw) - len(raw) % received.itemsize]
                        )
                        self.fds.extend(received)
                    else:
                        unexpected = True
                if unexpected:
                    raise QueueTransportError("Unexpected ancillary data")
                if flags & socket.MSG_CTRUNC or not data:
                    raise QueueTransportError("Truncated queue header or descriptors")
                header.extend(data)
            magic, direct, self.total, table_bytes, payload_bytes = _HEADER.unpack(
                header
            )
            if (
                magic != _MAGIC
                or direct > MAX_DIRECT_FDS
                or direct > self.total
                or direct != len(self.fds)
                or (direct == self.total) != (table_bytes == 0)
                or max(table_bytes, payload_bytes) > sys.maxsize
            ):
                raise QueueTransportError("Invalid queue frame")
            self.ticket_bytes = _read_exact(sock, table_bytes)
            self.payload = _read_exact(sock, payload_bytes)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close only descriptors not already handed to Torch's reconstruction."""
        for index, fd in enumerate(self.fds):
            if fd >= 0:
                self.fds[index] = -1
                os.close(fd)

    def __del__(self) -> None:
        self.close()

    def decode(self) -> Any:
        """Resolve local storage slots or demand-driven legacy tickets."""
        try:
            if not self.total:
                return ForkingPickler.loads(self.payload)
            self.tickets = (
                ForkingPickler.loads(self.ticket_bytes) if self.ticket_bytes else {}
            )
            if (
                not isinstance(self.tickets, dict)
                or len(self.tickets) != self.total - len(self.fds)
                or set(self.tickets) != set(range(len(self.fds), self.total))
            ):
                raise QueueTransportError("Invalid queue ticket table")
            return _Unpickler(self).load()
        finally:
            self.close()

    def rebuild(self, cls: type, index: int, size: int) -> Any:
        """Detach each slot exactly once and preserve storage aliases."""
        if not 0 <= index < self.total:
            raise QueueTransportError("Invalid storage index")
        if index in self.storages:
            previous_cls, previous_size, storage = self.storages[index]
            if cls is not previous_cls or size != previous_size:
                raise QueueTransportError("Inconsistent storage alias")
            return storage
        from torch.multiprocessing.reductions import rebuild_storage_fd

        storage = rebuild_storage_fd(cls, _Handle(self, index), size)
        self.storages[index] = cls, size, storage
        return storage


class _Handle:
    """Torch's detach() interface, transferring one table slot's FD ownership."""

    def __init__(self, message: ReceivedMessage, index: int) -> None:
        self.message, self.index = message, index

    def detach(self) -> int:
        if self.index < len(self.message.fds):
            fd = self.message.fds[self.index]
            self.message.fds[self.index] = -1
            if fd < 0:
                raise QueueTransportError("Storage descriptor already consumed")
            return fd
        return self.message.tickets.pop(self.index).detach()


class _Unpickler(pickle.Unpickler):
    """Resolve storage markers against this frame's descriptors and alias memo."""

    def __init__(self, message: ReceivedMessage) -> None:
        super().__init__(io.BytesIO(message.payload))
        self.message = message

    def find_class(self, module: str, name: str) -> Any:
        if module == __name__ and name == "_storage_from_slot":
            return self.message.rebuild
        return super().find_class(module, name)


receive = partial(read_frame, fd_frame=ReceivedMessage)
