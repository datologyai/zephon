# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Socket-backed Connection framing, buffering and descriptor ownership."""

from __future__ import annotations

import errno
import os
import socket
import threading
from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.reduction import ForkingPickler

import pytest

from zephon._internal.utils import ipc
from zephon._internal.utils.ipc import SocketConnection, socketpair_connections


class TestSocketpairConnections:
    def test_framing_round_trip(self) -> None:
        reader, writer = socketpair_connections(buffer_bytes=1024 * 1024)
        try:
            msg = b"y" * (64 * 1024)
            writer.send_bytes(msg)
            assert reader.recv_bytes() == msg
        finally:
            reader.close()
            writer.close()

    def test_buffer_bytes_applied(self) -> None:
        def sndbuf_of(conn: Connection) -> int:
            sock = socket.socket(fileno=os.dup(conn.fileno()))
            try:
                return sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            finally:
                sock.close()

        default_pair = socketpair_connections()
        sized_pair = socketpair_connections(buffer_bytes=512 * 1024)
        try:
            # Strictly larger than an unsized pair's platform default
            # (208 KiB Linux / 8 KiB macOS): proves setsockopt was applied,
            # not merely that the platform default clears some floor.
            assert sndbuf_of(sized_pair[1]) > sndbuf_of(default_pair[1])
            # Linux clamps to 2 * net.core.wmem_max (>= 416 KiB stock);
            # macOS honors the request exactly — either way above 128 KiB.
            assert sndbuf_of(sized_pair[1]) >= 128 * 1024
        finally:
            for conn in (*default_pair, *sized_pair):
                conn.close()


def test_close_defers_descriptor_disposal_until_receive_finishes() -> None:
    reader, writer = socketpair_connections()
    reader.close_guard = threading.Lock()
    try:
        with reader.close_guard:
            reader.close()
            assert reader.close_requested and not reader.closed
            writer.send_bytes(b"in progress")
            assert reader.recv_bytes() == b"in progress"
        reader.close()
        assert reader.closed and reader.socket.fileno() == -1
    finally:
        reader.close()
        writer.close()


def test_poll_reports_hangup_for_the_reader_to_handle() -> None:
    reader, writer = socketpair_connections()
    try:
        assert not reader.poll(0)
        writer.close()
        assert reader.poll(1)
        assert reader.socket.recv(1) == b""
    finally:
        reader.close()
        writer.close()


def _send_on_connection(writer: SocketConnection) -> None:
    writer.send_bytes(b"spawned")
    writer.close()


def test_spawn_transfers_socket_ownership_without_local_thread_lock() -> None:
    reader, writer = socketpair_connections()
    writer.close_guard = threading.Lock()
    child = get_context("spawn").Process(target=_send_on_connection, args=(writer,))
    child.start()
    writer.close()
    try:
        assert reader.poll(15)
        assert reader.recv_bytes() == b"spawned"
        child.join(5)
        assert child.exitcode == 0
    finally:
        reader.close()
        if child.is_alive():
            child.kill()
            child.join(5)


def test_would_block_after_partial_header_is_not_empty():
    class PartialSocket:
        calls = 0

        def recv(self, size, flags=0):
            self.calls += 1
            if self.calls == 1:
                assert flags & socket.MSG_DONTWAIT
                return b"Z"
            assert not flags & socket.MSG_DONTWAIT
            raise BlockingIOError(errno.EAGAIN, "injected after progress")

    with pytest.raises(BlockingIOError):
        ipc.read_frame(PartialSocket(), nonblocking_start=True)


@pytest.mark.parametrize("extended", [False, True])
def test_fragmented_native_prefix_and_extended_length(extended):
    reader, writer = socket.socketpair()
    payload = bytes(ForkingPickler.dumps({"payload": 123}))
    prefix = (
        ipc._LENGTH.pack(ipc._LARGE_FRAME) + len(payload).to_bytes(8, "big")
        if extended
        else ipc._LENGTH.pack(len(payload))
    )

    def produce():
        for value in prefix + payload:
            writer.sendall(bytes([value]))
        writer.close()

    thread = threading.Thread(target=produce)
    thread.start()
    try:
        assert ForkingPickler.loads(ipc.read_frame(reader)) == {"payload": 123}
    finally:
        reader.close()
        thread.join(3)


@pytest.mark.parametrize("marker", [-3, -2])
def test_native_reader_rejects_unsupported_frame_types(marker: int) -> None:
    reader, writer = socket.socketpair()
    try:
        writer.sendall(ipc._LENGTH.pack(marker))
        with pytest.raises(ipc.QueueTransportError, match="Invalid queue frame length"):
            ipc.read_frame(reader)
    finally:
        reader.close()
        writer.close()
