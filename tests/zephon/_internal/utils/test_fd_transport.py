# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Storage descriptor framing, reconstruction, ownership and quota fallback."""

from __future__ import annotations

import array
import multiprocessing as mp
import os
import socket
import sys
import tempfile
import weakref
from multiprocessing.reduction import ForkingPickler

import pytest

from tests.helpers.ipc import SocketSendFaults
from zephon._internal.utils import fd_transport as wire


class _Counted:
    calls = 0

    def __reduce__(self):
        self.calls += 1
        return str, ("payload",)


def test_encoding_does_not_retain_payload_until_cyclic_gc():
    pytest.importorskip("torch")
    value = _Counted()
    ref = weakref.ref(value)
    message = wire.EncodedMessage(value)
    del value
    assert ref() is None
    message.close()


@pytest.mark.parametrize(
    "reject,short,cap",
    [
        (False, False, 128),
        (False, True, 128),
        (True, False, 128),
        (False, False, 1),
        (True, False, 1),
    ],
)
def test_same_payload_direct_overflow_and_pressure_fallback(
    monkeypatch, reject, short, cap
):
    monkeypatch.setattr(wire, "MAX_DIRECT_FDS", cap)
    reader, writer = socket.socketpair()
    counted = _Counted()
    message = wire.EncodedMessage(counted)
    before = bytes(message.payload)
    with tempfile.TemporaryFile() as backing:
        backing.write(b"shared data")
        backing.flush()
        identity = os.fstat(backing.fileno())
        # Different keys intentionally test separate table slots for this FD.
        for i in range(3):
            message.table.add(backing.fileno(), (identity.st_dev, identity.st_ino + i))
        proxy = SocketSendFaults(writer, reject=reject, short=short)
        frame = None
        try:
            message.send(proxy)
            frame = wire.receive(reader)
            assert bytes(frame.payload) == before
            assert ForkingPickler.loads(frame.payload) == "payload"
            assert counted.calls == 1
            assert len(frame.fds) == (0 if reject else min(3, cap))
            for fd in frame.fds:
                assert os.pread(fd, 11, 0) == b"shared data"
                if sys.platform == "linux":
                    assert not os.get_inheritable(fd)
            tickets = (
                ForkingPickler.loads(frame.ticket_bytes) if frame.ticket_bytes else {}
            )
            for ticket in tickets.values():
                fd = ticket.detach()
                try:
                    assert os.pread(fd, 11, 0) == b"shared data"
                finally:
                    os.close(fd)
            assert proxy.rights_calls == 1
        finally:
            if frame is not None:
                frame.close()
            message.close()
            reader.close()
            writer.close()


def test_fragmented_fd_marker_does_not_discard_rights():
    reader, writer = socket.socketpair()
    with tempfile.TemporaryFile() as backing:
        message = wire.EncodedMessage("with fd")
        message.table.add(backing.fileno(), (1, 1))

        class Fragmented(SocketSendFaults):
            def sendall(self, data, *args):
                if data == wire._FD_PREFIX:
                    for value in data:
                        self.sock.sendall(bytes([value]))
                else:
                    self.sock.sendall(data, *args)

        frame = None
        try:
            message.send(Fragmented(writer))
            frame = wire.receive(reader)
            assert len(frame.fds) == 1 and frame.decode() == "with fd"
        finally:
            if frame is not None:
                frame.close()
            message.close()
            reader.close()
            writer.close()


def test_invalid_header_closes_received_descriptors():
    with tempfile.TemporaryFile() as backing:
        fd = os.dup(backing.fileno())

        class BadSocket:
            def recvmsg(self, *args):
                import array

                return (
                    wire._HEADER.pack(b"BAD!", 1, 1, 0, 0),
                    [
                        (
                            socket.SOL_SOCKET,
                            socket.SCM_RIGHTS,
                            array.array("i", [fd]).tobytes(),
                        )
                    ],
                    0,
                    None,
                )

        with pytest.raises(wire.QueueTransportError):
            wire.ReceivedMessage(BadSocket())
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(fd)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux SCM_RIGHTS truncation")
def test_native_ancillary_truncation_closes_delivered_descriptors():
    reader, writer = socket.socketpair()
    with tempfile.TemporaryFile() as backing:
        identity = os.fstat(backing.fileno())

        def matching_fds():
            result = set()
            for name in os.listdir("/proc/self/fd"):
                try:
                    stat = os.fstat(int(name))
                    if (stat.st_ino, stat.st_dev) == (identity.st_ino, identity.st_dev):
                        result.add(int(name))
                except OSError:
                    pass
            return result

        before = matching_fds()
        try:
            writer.sendmsg(
                [wire._HEADER.pack(wire._MAGIC, 128, 128, 0, 0)],
                [
                    (
                        socket.SOL_SOCKET,
                        socket.SCM_RIGHTS,
                        array.array("i", [backing.fileno()] * 160),
                    )
                ],
            )
            with pytest.raises(wire.QueueTransportError, match="Truncated"):
                wire.ReceivedMessage(reader)
            assert matching_fds() == before
        finally:
            reader.close()
            writer.close()


def test_custom_storage_reducer_is_preserved(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(
        ForkingPickler._extra_reducers,
        torch.UntypedStorage,
        lambda storage: (str, ("custom",)),
    )
    message = wire.EncodedMessage(torch.UntypedStorage(8))
    try:
        assert not message.table.fds
        assert ForkingPickler.loads(message.payload) == "custom"
    finally:
        message.close()


def _kernel_quota_probe(control):
    import resource

    reader, writer = socket.socketpair()
    with tempfile.TemporaryFile() as backing:
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
        for index in range(200):
            message = wire.EncodedMessage(index)
            message.table.add(backing.fileno(), (1, 1))
            try:
                message.send(writer)
                fallback = message.table.fds[0] == -1
            finally:
                message.close()
            if fallback:
                break
        else:
            raise AssertionError("Kernel did not enforce the unprivileged FD quota")
        for expected in range(index + 1):
            frame = wire.receive(reader)
            try:
                assert ForkingPickler.loads(frame.payload) == expected
                if frame.ticket_bytes:
                    for ticket in ForkingPickler.loads(frame.ticket_bytes).values():
                        os.close(ticket.detach())
            finally:
                frame.close()
        control.send((fallback, index))
    reader.close()
    writer.close()


@pytest.mark.skipif(
    sys.platform != "linux" or os.geteuid() == 0,
    reason="Linux unprivileged in-flight FD quota",
)
def test_real_kernel_fd_quota_falls_back_without_queue_credits():
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_kernel_quota_probe, args=(child,))
    proc.start()
    child.close()
    try:
        assert parent.poll(15)
        # Quota is per UID; another process may already occupy its headroom.
        fallback, index = parent.recv()
        assert fallback and 0 <= index < 200
        proc.join(5)
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        parent.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Torch FD storage is Linux-only")
@pytest.mark.parametrize("cap,reject", [(128, False), (1, False), (1, True)])
def test_numpy_tensor_and_bytes_storage_roundtrip(monkeypatch, cap, reject):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from zephon._internal.utils.shm_coalesce import _shared_numpy_view, _ShmBytes

    monkeypatch.setattr(wire, "MAX_DIRECT_FDS", cap)
    reader, writer = socket.socketpair()
    tensor = torch.arange(16)
    array = _shared_numpy_view(torch.arange(16).share_memory_())[::-1]
    value = [
        tensor[2:],
        tensor[4:],
        array,
        _ShmBytes(torch.tensor(list(b"test bytes"), dtype=torch.uint8)),
        torch.empty(0),
    ]
    message = wire.EncodedMessage(value)
    try:
        # Before 3.12 the fixed-size bytearray fallback copies its tensor input.
        assert len(message.table.fds) == (3 if sys.version_info >= (3, 12) else 2)
        message.send(SocketSendFaults(writer, reject=reject))
        result = wire.receive(reader).decode()
        assert torch.equal(result[0], tensor[2:])
        assert (
            result[0].untyped_storage().data_ptr()
            == result[1].untyped_storage().data_ptr()
        )
        np.testing.assert_array_equal(result[2], array)
        assert bytes(result[3]) == b"test bytes"
        assert result[4].numel() == 0
    finally:
        message.close()
        reader.close()
        writer.close()
