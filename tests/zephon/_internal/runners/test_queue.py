# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""NamedQueue serialization, backpressure, transports and process lifecycle."""

from __future__ import annotations

import errno
import inspect
import multiprocessing as mp
import os
import stat
import sys
import threading
import time
import weakref
from multiprocessing.reduction import ForkingPickler
from queue import Empty, Full
from typing import Any, cast

import pytest

from tests.helpers.ipc import SocketSendFaults
from zephon._internal.runners import queue
from zephon._internal.runners.queue import NamedQueue
from zephon._internal.utils import fd_transport as wire
from zephon._internal.utils import ipc
from zephon.options import IpcTransport


def _produce(q: NamedQueue, n: int, payload_size: int) -> None:
    payload = b"x" * payload_size
    for i in range(n):
        q.put((i, payload))
    q.put(None)


class TestNamedQueueTransports:
    @pytest.mark.parametrize("transport", ["socketpair", "pipe"])
    def test_cross_process_round_trip(self, transport: IpcTransport) -> None:
        ctx = mp.get_context("spawn")
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
        ctx = mp.get_context("spawn")
        q = NamedQueue("default", ctx=ctx)
        try:
            mode = os.fstat(q._reader.fileno()).st_mode
            assert stat.S_ISSOCK(mode)
        finally:
            q.close()

    def test_pipe_transport_uses_pipe(self) -> None:
        ctx = mp.get_context("spawn")
        q = NamedQueue("stock", ctx=ctx, transport="pipe")
        try:
            mode = os.fstat(q._reader.fileno()).st_mode
            assert stat.S_ISFIFO(mode)
        finally:
            q.close()

    def test_unknown_transport_raises(self) -> None:
        ctx = mp.get_context("spawn")
        with pytest.raises(ValueError, match="Unknown queue transport"):
            NamedQueue("bad", ctx=ctx, transport=cast(IpcTransport, "carrier-pigeon"))


class TestStagedBytes:
    @pytest.mark.parametrize("transport", ["socketpair", "pipe"])
    def test_tracks_write_and_drain(self, transport: IpcTransport) -> None:
        ctx = mp.get_context("spawn")
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
        ctx = mp.get_context("spawn")
        q = NamedQueue("staged-closed", maxsize=2, ctx=ctx)
        q.close()
        q.join_thread()
        assert q.staged_bytes() == -1


@pytest.fixture
def framed_queue(monkeypatch):
    # Exercise framing on macOS too; Torch's FD reducer tests remain Linux-only.
    monkeypatch.setattr(queue, "_use_socket_transport", lambda transport: True)
    q = queue.NamedQueue("framed-test", maxsize=2, ctx=mp.get_context("spawn"))
    yield q
    q.abort()
    q.close()
    q.join_thread()
    q._reader.close()
    q._writer.close()


class _Counted:
    calls = 0

    def __reduce__(self):
        self.calls += 1
        return str, ("payload",)


def test_idle_feeder_releases_original_payload(framed_queue):
    value = _Counted()
    ref = weakref.ref(value)
    framed_queue.put(value)
    del value
    assert framed_queue.get(timeout=5) == "payload"
    assert ref() is None


def test_uses_unmodified_stdlib_feeder(framed_queue):
    value = _Counted()
    framed_queue.put(value)
    assert framed_queue._thread._target.__code__ is queue.mp_queues.Queue._feed.__code__
    assert (
        framed_queue._thread._target.__globals__["_ForkingPickler"]
        is framed_queue._frame_writer
    )
    assert framed_queue.get(timeout=5) == "payload"
    assert value.calls == 1


class _ReducerFailure:
    def __init__(self, error):
        self.error = error

    def __reduce__(self):
        raise self.error


def test_index_error_in_reducer_is_reported(framed_queue):
    framed_queue.put(_ReducerFailure(IndexError("injected reducer failure")))
    with pytest.raises(queue.QueueFeederError, match="injected reducer failure"):
        framed_queue.get(timeout=5)
    framed_queue.put("after-error")
    assert framed_queue.get(timeout=5) == "after-error"


def test_epipe_in_reducer_aborts_instead_of_leaving_a_dead_feeder(framed_queue):
    framed_queue.put(
        _ReducerFailure(BrokenPipeError(errno.EPIPE, "injected reducer failure"))
    )
    with pytest.raises(queue.QueueTransportError):
        framed_queue.get(timeout=5)
    framed_queue._thread.join(3)
    assert not framed_queue._thread.is_alive()
    with pytest.raises(queue.QueueTransportError):
        framed_queue.put("after-abort")


def test_pickler_isolation_for_unrelated_and_nested_pickles(framed_queue):
    original = queue.mp_queues._ForkingPickler

    class Nested:
        def __reduce__(self):
            payload = queue.mp_queues._ForkingPickler.dumps({"nested": 123}, 2)
            return ForkingPickler.loads, (bytes(payload),)

    framed_queue.put(Nested())
    assert framed_queue.get(timeout=5) == {"nested": 123}
    assert queue.mp_queues._ForkingPickler is original
    q = mp.get_context("spawn").Queue()
    try:
        q.put({"ordinary": "queue"})
        assert q.get(timeout=5) == {"ordinary": "queue"}
    finally:
        q.close()
        q.join_thread()


def test_registered_reducers_are_refreshed_between_messages(framed_queue, monkeypatch):
    value = _Counted()
    monkeypatch.setitem(
        ForkingPickler._copyreg_dispatch_table,
        _Counted,
        lambda value: (str, ("copyreg",)),
    )
    framed_queue.put([value, value])
    assert framed_queue.get(timeout=5) == ["copyreg", "copyreg"]
    monkeypatch.setitem(
        ForkingPickler._extra_reducers, _Counted, lambda value: (str, ("override",))
    )
    framed_queue.put([value, value])
    assert framed_queue.get(timeout=5) == ["override", "override"]
    assert value.calls == 0


def test_no_fd_messages_use_native_framing_without_fd_bookkeeping(
    framed_queue, monkeypatch
):
    def unexpected(*args, **kwargs):
        raise AssertionError("Zero-FD message entered the FD path")

    monkeypatch.setattr(wire, "_FdTable", unexpected)
    monkeypatch.setattr(wire, "register_after_fork", unexpected)
    monkeypatch.setattr(wire, "ReceivedMessage", unexpected)
    for value in [None, [], {"data": [1, "two", b"three"]}, b"x" * 65536]:
        framed_queue.put(value)
        assert framed_queue.get(timeout=5) == value


def test_framed_queue_capacity_and_feeder_error(framed_queue):
    q = framed_queue
    assert q.empty()
    with pytest.raises(Empty):
        q.get_nowait()
    q.put("first")
    q.put("second")
    with pytest.raises(Full):
        q.put_nowait("full")
    assert q.get(timeout=5) == "first"
    assert q.get(timeout=5) == "second"
    assert q.empty()
    q.put(iter(i for i in range(1)))  # Generator serialization fails in the feeder.
    q.put("after-error")
    values, failures = [], 0
    for _ in range(2):
        try:
            values.append(q.get(timeout=5))
        except queue.QueueFeederError:
            failures += 1
    assert values == ["after-error"] and failures == 1
    q.put("capacity-restored")
    assert q.get(timeout=5) == "capacity-restored"


class _RetryShmOnce(_Counted):
    def __reduce__(self):
        self.calls += 1
        if self.calls == 1:
            raise OSError(errno.ENOSPC, "injected SHM exhaustion")
        return str, ("retried",)


def test_shm_retry_preserves_message_and_capacity(framed_queue, monkeypatch):
    waits = []
    monkeypatch.setattr(
        queue, "wait_for_shm_space", lambda *a, **k: waits.append(1) or 0
    )
    value = _RetryShmOnce()
    framed_queue.put(value)
    assert framed_queue.get(timeout=5) == "retried"
    assert value.calls == 2 and waits == [1]
    assert framed_queue._frame_writer.pending is None
    framed_queue.put("after-retry")
    assert framed_queue.get(timeout=5) == "after-retry"


def test_abort_wakes_partial_frame_reader(framed_queue):
    q = framed_queue
    failures = []
    q._writer.socket.sendall(b"Z")

    def consume():
        try:
            q.get(timeout=0.1)
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=consume)
    thread.start()
    deadline = time.monotonic() + 2
    while not q._transport_state[2] and time.monotonic() < deadline:
        time.sleep(0.001)
    assert q._transport_state[2] == os.getpid()
    q.abort()
    thread.join(2)
    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], queue.QueueTransportError)
    assert q._reader.socket.fileno() == -1


def test_abort_wakes_capacity_waiter(framed_queue):
    q = framed_queue
    q.put("first")
    q.put("second")
    failures = []

    def produce():
        try:
            q.put("blocked")
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=produce)
    thread.start()
    q.abort()
    thread.join(2)
    assert not thread.is_alive()
    assert isinstance(failures[0], queue.QueueTransportError)


def test_abort_cancels_error_reporting_wait(framed_queue):
    q = framed_queue
    q.put("first")
    q.put("second")
    thread = threading.Thread(
        target=q._enqueue_sentinel, args=("serialization failed",)
    )
    thread.start()
    q.abort()
    thread.join(2)
    assert not thread.is_alive()


def test_broken_pipe_drops_buffered_roots_and_stops_stdlib_feeder(
    framed_queue, monkeypatch
):
    entered, proceed = threading.Event(), threading.Event()

    def fail_send(message, sock):
        entered.set()
        assert proceed.wait(3)
        raise BrokenPipeError(errno.EPIPE, "injected peer loss")

    q = framed_queue
    monkeypatch.setattr(wire.EncodedMessage, "send", fail_send)
    q.put("sending")
    assert entered.wait(3)
    value = _Counted()
    ref = weakref.ref(value)
    q.put(value)
    del value
    proceed.set()
    q._thread.join(3)
    assert not q._thread.is_alive()
    assert ref() is None
    with pytest.raises(queue.QueueTransportError):
        q.get(timeout=0.1)


@pytest.mark.parametrize("slot", [1, 2])
def test_dead_or_unknown_lock_owner_aborts(framed_queue, slot):
    q = framed_queue
    lock = q._wlock if slot == 1 else q._rlock
    lock.acquire()
    q._transport_state[slot] = (
        0  # Simulate death between lock acquisition and PID publication.
    )
    try:
        assert not q.check_dead_process(123456789)
        with pytest.raises(queue.QueueTransportError):
            q.put("cannot reuse")
    finally:
        lock.release()


def test_live_writer_is_not_force_unlocked(framed_queue):
    q = framed_queue
    q._wlock.acquire()
    q._transport_state[1] = os.getpid()
    try:
        assert q.check_dead_process(123456789)
        assert not q._wlock.acquire(block=False)
    finally:
        q._transport_state[1] = 0
        q._wlock.release()


def test_idle_reader_holds_no_process_lock(framed_queue):
    q = framed_queue
    thread = threading.Thread(target=lambda: q.get(timeout=2))
    thread.start()
    try:
        assert q._rlock.acquire(timeout=1)
        q._rlock.release()
        q.put("wake")
    finally:
        thread.join(3)
    assert not thread.is_alive()


def test_cpython_feeder_contract():
    """Pin the CPython internals that socketpair NamedQueues hook into.

    They keep CPython's feeder thread but hand it our write lock and send hook,
    and abort() stops it with CPython's sentinel. If a new Python changes this
    wiring, fail here instead of letting the queue misbehave silently.
    """
    stock = queue.mp_queues
    feeder = stock.Queue._feed
    assert " ".join(inspect.signature(feeder).parameters) == (
        "buffer notempty send_bytes writelock reader_close writer_close "
        "ignore_epipe onerror queue_sem"
    )
    assert "_sentinel" in feeder.__code__.co_names
    hooks = {"_buffer", "_notempty", "_send_bytes", "_wlock", "_on_queue_feeder_error"}
    assert hooks <= set(stock.Queue._start_thread.__code__.co_names)


def test_buffered_receive_avoids_readiness_polls(framed_queue, monkeypatch):
    q = framed_queue
    q.put("buffered")
    deadline = time.monotonic() + 3
    while q.staged_bytes() <= 0 and time.monotonic() < deadline:
        time.sleep(0.001)

    def unexpected_poll(*args):
        raise AssertionError("buffered receive should not need a selector")

    monkeypatch.setattr(q, "_poll", unexpected_poll)
    assert q.get(timeout=1) == "buffered"


def test_initial_would_block_keeps_queue_usable(framed_queue):
    q = framed_queue
    with pytest.raises(Empty):
        q.get_nowait()
    started = time.monotonic()
    with pytest.raises(Empty):
        q.get(timeout=0.02)
    assert time.monotonic() - started < 1
    assert not q._transport_state[0]
    q.put("after-empty")
    assert q.get(timeout=1) == "after-empty"


def test_local_close_after_poll_does_not_abort_peers(framed_queue, monkeypatch):
    q = framed_queue

    def close_after_poll(timeout):
        q._reader.close()
        return True

    monkeypatch.setattr(q, "_poll", close_after_poll)
    with pytest.raises(ValueError, match="is closed"):
        q.get(timeout=1)
    assert not q._transport_state[0]
    assert q._transport_state[2] == 0


def test_nonblocking_probe_completes_fragmented_frame(framed_queue, monkeypatch):
    q = framed_queue
    results = []
    parked = threading.Event()
    original = q._poll

    def poll(timeout):
        parked.set()
        return original(timeout)

    monkeypatch.setattr(q, "_poll", poll)
    thread = threading.Thread(target=lambda: results.append(q.get(timeout=5)))
    thread.start()
    assert parked.wait(3)  # The initial empty probe has released its lock.
    payload = bytes(ForkingPickler.dumps("fragmented"))
    header = ipc._LENGTH.pack(len(payload))
    q._sem.acquire()
    try:
        q._writer.socket.sendall(header[:3])
        deadline = time.monotonic() + 3
        while not q._transport_state[2] and time.monotonic() < deadline:
            time.sleep(0.001)
        assert q._transport_state[2] == os.getpid()
        q._writer.socket.sendall(header[3:] + payload)
        thread.join(3)
        assert not thread.is_alive()
        assert results == ["fragmented"]
    finally:
        thread.join(3)


def test_poll_timeouts_do_not_repeat_empty_probes(framed_queue, monkeypatch):
    calls = []
    original = queue.read_frame

    def receive(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(queue, "read_frame", receive)
    with pytest.raises(Empty):
        framed_queue.get(timeout=0.21)
    assert calls == [1]


def test_abort_wakes_reader_parked_in_poll(framed_queue, monkeypatch):
    q = framed_queue
    parked, failures = threading.Event(), []
    original = q._poll

    def poll(timeout):
        parked.set()
        return original(timeout)

    def consume():
        try:
            q.get(timeout=5)
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(q, "_poll", poll)
    thread = threading.Thread(target=consume)
    thread.start()
    assert parked.wait(3)
    q.abort()
    thread.join(3)
    assert not thread.is_alive()
    assert len(failures) == 1 and isinstance(failures[0], queue.QueueTransportError)


def _produce_plain(q, producer):
    for i in range(12):
        q.put((producer, i, b"x" * 70_000))
    q.close()
    q.join_thread()


@pytest.mark.parametrize("start", mp.get_all_start_methods())
def test_multiple_writers(framed_queue, start):
    q = framed_queue
    children = [
        mp.get_context(start).Process(target=_produce_plain, args=(q, i))
        for i in range(3)
    ]
    for child in children:
        child.start()
    try:
        seen = set()
        for _ in range(36):
            producer, i, payload = q.get(timeout=15)
            assert payload == b"x" * 70_000
            assert (producer, i) not in seen
            seen.add((producer, i))
        for child in children:
            child.join(5)
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join(5)


def _hold_partial_frame(q, role, control):
    if role == "writer":
        q._wlock.acquire()
        q._transport_state[1] = os.getpid()
        q._writer.socket.sendall(b"Z")
        control.send("ready")
        control.recv()
    else:
        control.send("ready")
        q.get()


def _publish_owner(q, role, control):
    if role == "writer":
        q._wlock.acquire()
        control.send("ready")
        control.recv()
        q._wlock.release()
    else:
        control.send("ready")
        try:
            q.get()
        except queue.QueueTransportError:
            pass


def _spawn_then_fork(q, role, control):
    parent, child = mp.get_context("fork").Pipe()
    proc = mp.get_context("fork").Process(target=_publish_owner, args=(q, role, child))
    proc.start()
    child.close()
    assert parent.recv() == "ready"
    control.send(proc.pid)
    control.recv()
    if role == "writer":
        parent.send("release")
    proc.join(5)
    control.send(proc.exitcode)
    parent.close()


@pytest.mark.parametrize("role,slot", [("writer", 1), ("reader", 2)])
def test_owner_pid_after_forking_a_spawned_queue(framed_queue, role, slot):
    q = framed_queue
    if role == "reader":
        q._writer.socket.sendall(b"Z")
    parent, child = mp.get_context("spawn").Pipe()
    proc = mp.get_context("spawn").Process(
        target=_spawn_then_fork, args=(q, role, child)
    )
    proc.start()
    child.close()
    try:
        assert parent.poll(15)
        owner = parent.recv()
        deadline = time.monotonic() + 3
        while not q._transport_state[slot] and time.monotonic() < deadline:
            time.sleep(0.001)
        assert q._transport_state[slot] == owner
        if role == "reader":
            q.abort()
        parent.send("release")
        assert parent.poll(10) and parent.recv() == 0
        proc.join(5)
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            q.abort()
            parent.send("release")
            proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        parent.close()


def _put_and_flush(q, value):
    q.put(value)
    q.close()
    q.join_thread()


def _spawn_feeder_then_fork(q, control):
    q.put("spawn")  # Start this process's feeder before forking.
    control.recv()  # The root has consumed it.
    proc = mp.get_context("fork").Process(target=_put_and_flush, args=(q, "fork"))
    proc.start()
    proc.join(5)
    assert proc.exitcode == 0
    q.close()
    q.join_thread()


def test_fork_after_spawn_resets_an_existing_feeder(framed_queue):
    q = framed_queue
    parent, child = mp.get_context("spawn").Pipe()
    proc = mp.get_context("spawn").Process(
        target=_spawn_feeder_then_fork, args=(q, child)
    )
    proc.start()
    child.close()
    try:
        assert q.get(timeout=15) == "spawn"
        parent.send("consumed")
        assert q.get(timeout=10) == "fork"
        proc.join(10)
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        parent.close()


def _consume_plain(q, control):
    control.send([q.get(timeout=10) for _ in range(8)])


def test_multiple_consumers(framed_queue):
    ctx = mp.get_context("spawn")
    workers = []
    for _ in range(3):
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_consume_plain, args=(framed_queue, child))
        proc.start()
        child.close()
        workers.append((proc, parent))
    try:
        for index in range(24):
            framed_queue.put((index, b"x" * 70_000))
        received = []
        for proc, control in workers:
            assert control.poll(15)
            received.extend(
                index for index, payload in control.recv() if payload == b"x" * 70_000
            )
            proc.join(5)
            assert proc.exitcode == 0
        assert sorted(received) == list(range(24))
    finally:
        for proc, control in workers:
            if proc.is_alive():
                proc.kill()
                proc.join(5)
            control.close()


def _produce_mixed(q, producer):
    import torch

    storage = torch.arange(4).share_memory_()
    for index in range(18):
        value = storage if index % 2 else {"plain": b"x" * 70000}
        q.put((producer, index, value))
    q.close()
    q.join_thread()


def _consume_mixed(q, control):
    import torch

    seen = []
    for _ in range(18):
        producer, index, value = q.get(timeout=15)
        if index % 2:
            assert torch.equal(value, torch.arange(4)) and value.is_shared()
        else:
            assert value == {"plain": b"x" * 70000}
        seen.append((producer, index))
    control.send(seen)


@pytest.mark.skipif(sys.platform != "linux", reason="Torch FD storage is Linux-only")
def test_native_and_fd_frames_with_multiple_producers_and_consumers(framed_queue):
    ctx = mp.get_context("spawn")
    producers = [
        ctx.Process(target=_produce_mixed, args=(framed_queue, i)) for i in range(3)
    ]
    consumers = []
    for _ in range(3):
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_consume_mixed, args=(framed_queue, child))
        proc.start()
        child.close()
        consumers.append((proc, parent))
    for proc in producers:
        proc.start()
    try:
        seen = []
        for proc, control in consumers:
            assert control.poll(30)
            seen.extend(control.recv())
        assert sorted(seen) == [(i, j) for i in range(3) for j in range(18)]
        for proc in producers + [proc for proc, _ in consumers]:
            proc.join(5)
            assert proc.exitcode == 0
    finally:
        for proc in producers + [proc for proc, _ in consumers]:
            if proc.is_alive():
                proc.kill()
                proc.join(5)
        for _, control in consumers:
            control.close()


@pytest.mark.parametrize("role,slot", [("writer", 1), ("reader", 2)])
def test_killed_transaction_owner_invalidates_stream(framed_queue, role, slot):
    q = framed_queue
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_hold_partial_frame, args=(q, role, child))
    if role == "reader":
        q._writer.socket.sendall(b"Z")
    proc.start()
    child.close()
    try:
        assert parent.poll(10)
        assert parent.recv() == "ready"
        deadline = time.monotonic() + 3
        while q._transport_state[slot] != proc.pid and time.monotonic() < deadline:
            time.sleep(0.001)
        assert q._transport_state[slot] == proc.pid
        proc.kill()
        proc.join(5)
        assert not q.check_dead_process(proc.pid)
        with pytest.raises(queue.QueueTransportError):
            q.get(timeout=0.1)
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        parent.close()


def _produce_tensors(q):
    import torch

    tensor = torch.arange(32)
    q.put([tensor[1:7], tensor[3:10], torch.arange(8, dtype=torch.float32)])
    q.close()
    q.join_thread()  # Successful put alone is not a completed transfer.


@pytest.mark.skipif(
    sys.platform != "linux", reason="Torch file_descriptor is Linux-only"
)
def test_tensors_survive_sender_exit_without_resource_sharer(monkeypatch):
    torch = pytest.importorskip("torch")
    import multiprocessing.resource_sharer as sharer

    ctx = mp.get_context("spawn")
    q = queue.NamedQueue("sender-exit", ctx=ctx)
    child = ctx.Process(target=_produce_tensors, args=(q,))
    child.start()
    try:
        child.join(15)
        assert child.exitcode == 0

        def unexpected(*args, **kwargs):
            raise AssertionError("direct receive must not contact resource_sharer")

        monkeypatch.setattr(sharer._resource_sharer, "get_connection", unexpected)
        values = q.get(timeout=5)
        assert torch.equal(values[0], torch.arange(1, 7))
        assert torch.equal(values[1], torch.arange(3, 10))
        assert (
            values[0].untyped_storage().data_ptr()
            == values[1].untyped_storage().data_ptr()
        )
        assert all(value.is_shared() for value in values)
    finally:
        q.close()
        if child.is_alive():
            child.kill()
            child.join(5)


@pytest.mark.skipif(sys.platform != "linux", reason="Torch FD storage is Linux-only")
@pytest.mark.parametrize("cap,reject", [(128, False), (1, False), (1, True)])
def test_storage_overflow_and_pressure_through_stdlib_feeder(
    framed_queue, monkeypatch, cap, reject
):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(wire, "MAX_DIRECT_FDS", cap)
    q = framed_queue
    proxy = SocketSendFaults(q._writer.socket, reject=reject)
    q._writer.socket = proxy
    values = [torch.arange(4), torch.arange(6)]
    q.put(values)
    q.put(values)  # Each put has its own direct handle allowance.
    for _ in range(2):
        received = q.get(timeout=10)
        assert all(torch.equal(a, b) for a, b in zip(received, values))
    assert proxy.rights_calls == 2


@pytest.mark.skipif(
    sys.platform != "linux", reason="Torch file_descriptor is Linux-only"
)
def test_storage_transfer_uses_selected_codec(
    framed_queue: NamedQueue,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    import multiprocessing.resource_sharer as sharer

    torch = pytest.importorskip("torch")
    strategy = torch.multiprocessing.get_sharing_strategy()
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
    request.addfinalizer(lambda: torch.multiprocessing.set_sharing_strategy(strategy))
    connections = []
    original = sharer._resource_sharer.get_connection

    def get_connection(ticket: Any) -> Any:
        connections.append(ticket)
        return original(ticket)

    monkeypatch.setattr(sharer._resource_sharer, "get_connection", get_connection)
    framed_queue.put(torch.arange(8))
    received = framed_queue.get(timeout=5)
    assert torch.equal(received, torch.arange(8)) and received.is_shared()
    assert len(connections) == 0


def test_abort_stops_feeder_blocked_in_large_native_send(
    framed_queue: NamedQueue,
) -> None:
    framed_queue.put(b"x" * (8 * 1024 * 1024))
    deadline = time.monotonic() + 3
    while not framed_queue.staged_bytes() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert framed_queue.staged_bytes() > 0
    assert framed_queue._transport_state[1] == os.getpid()
    framed_queue.abort()
    framed_queue._thread.join(3)
    assert not framed_queue._thread.is_alive()


@pytest.mark.parametrize("role", ["reader", "writer"])
def test_abort_wakes_process_lock_waiter(
    framed_queue: NamedQueue, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    q = framed_queue
    lock = q._rlock if role == "reader" else q._wlock
    lock.acquire()
    waiting, failures = threading.Event(), []
    original = queue._acquire

    def acquire(*args: Any, **kwargs: Any) -> bool:
        waiting.set()
        return original(*args, **kwargs)

    def consume() -> None:
        try:
            q.get(timeout=5)
        except queue.QueueTransportError as exc:
            failures.append(exc)

    monkeypatch.setattr(queue, "_acquire", acquire)
    if role == "reader":
        waiter = threading.Thread(target=consume)
        waiter.start()
    else:
        q.put("waiting for lock")
        waiter = q._thread
    try:
        assert waiting.wait(3)
        q.abort()
        waiter.join(3)
        assert not waiter.is_alive()
        if role == "reader":
            assert len(failures) == 1
    finally:
        lock.release()
