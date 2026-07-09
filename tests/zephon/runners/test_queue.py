# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for NamedQueue IPC transports (socketpair vs pipe)."""

from __future__ import annotations

import multiprocessing
import os
import socket
import stat
import time
from multiprocessing.connection import Connection
from typing import cast

import pytest

from zephon.runners.queue import NamedQueue
from zephon.utils.ipc import IpcTransport, socketpair_connections


def _produce(q: NamedQueue, n: int, payload_size: int) -> None:
    payload = b"x" * payload_size
    for i in range(n):
        q.put((i, payload))
    q.put(None)


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


class TestNamedQueueTransports:
    @pytest.mark.parametrize("transport", ["socketpair", "pipe"])
    def test_cross_process_round_trip(self, transport: IpcTransport) -> None:
        ctx = multiprocessing.get_context("spawn")
        q = NamedQueue(
            "xproc",
            maxsize=8,
            ctx=ctx,
            transport=transport,
            buffer_bytes=512 * 1024,
        )
        proc = ctx.Process(target=_produce, args=(q, 50, 10_240))
        proc.start()
        try:
            for expected in range(50):
                i, payload = q.get(timeout=30)
                assert i == expected
                assert len(payload) == 10_240
            assert q.get(timeout=30) is None
        finally:
            proc.join(timeout=30)
            q.close()
        assert proc.exitcode == 0

    def test_socketpair_is_default(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        q = NamedQueue("default", ctx=ctx)
        try:
            mode = os.fstat(q._reader.fileno()).st_mode
            assert stat.S_ISSOCK(mode)
        finally:
            q.close()

    def test_pipe_transport_uses_pipe(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        q = NamedQueue("stock", ctx=ctx, transport="pipe")
        try:
            mode = os.fstat(q._reader.fileno()).st_mode
            assert stat.S_ISFIFO(mode)
        finally:
            q.close()

    def test_unknown_transport_raises(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        with pytest.raises(ValueError, match="Unknown queue transport"):
            NamedQueue("bad", ctx=ctx, transport=cast(IpcTransport, "carrier-pigeon"))


class TestStagedBytes:
    @pytest.mark.parametrize("transport", ["socketpair", "pipe"])
    def test_tracks_write_and_drain(self, transport: IpcTransport) -> None:
        ctx = multiprocessing.get_context("spawn")
        q = NamedQueue("staged", maxsize=8, ctx=ctx, transport=transport)
        try:
            assert q.staged_bytes() == 0

            payload = b"x" * 1024
            q.put(payload)
            # The feeder thread pickles and writes asynchronously.
            deadline = time.monotonic() + 10
            while q.staged_bytes() == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            # Pickled payload plus length framing exceeds the raw size.
            assert q.staged_bytes() > len(payload)

            assert q.get(timeout=10) == payload
            assert q.staged_bytes() == 0
        finally:
            q.close()
            q.join_thread()

    def test_closed_queue_reports_minus_one(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        q = NamedQueue("staged-closed", maxsize=2, ctx=ctx)
        q.close()
        q.join_thread()
        assert q.staged_bytes() == -1
