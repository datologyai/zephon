"""Focused unit tests for ``zephon.runners.watchdog``.

These verify the Linux ``/proc``-based introspection helpers in isolation
of the watchdog/runner machinery.  Most importantly they cover the
inode-based fd lookup that resolves the parent's pipe fd to the worker's
(different) fd number for the same kernel pipe — a key correctness
property when the recovery chain reaches strategy 3 with parallelism > 1.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pytest

# NOTE: keep zephon imports function-local.  Forkserver children
# re-import this module; a module-level zephon import adds seconds of
# child startup that the timed /proc-sampling tests race against.

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="/proc/PID/syscall is Linux-specific",
)

#: Both NamedQueue transports.  The /proc introspection must identify a
#: live writer over either — a transport it can't resolve makes every
#: worker look like a non-writer, and the watchdog would force-release
#: ``_wlock`` while a live writer holds it.
TRANSPORTS = ["pipe", "socketpair"]


def _connection_pair(
    transport: str, ctx: mp.context.BaseContext
) -> tuple["mp.connection.Connection", "mp.connection.Connection"]:
    """Return a (reader, writer) Connection pair over the given transport."""
    from zephon.utils.ipc import socketpair_connections

    if transport == "socketpair":
        return socketpair_connections()
    return ctx.Pipe(duplex=False)


def _writer_loop(writer: "mp.connection.Connection") -> None:  # pragma: no cover
    """Child target: write to ``writer`` forever.

    Writes 4 KiB chunks with no sleep.  The reader end is intentionally
    *not* drained in the parent, so the kernel buffer (64 KiB for a
    pipe, ``SO_SNDBUF`` for a socketpair) fills within microseconds and
    subsequent ``os.write`` calls block until space is available —
    which is never, under this test.  The child therefore spends nearly
    100 % of its wall time inside the ``write`` syscall, which is
    exactly the signature the watchdog's ``/proc/PID/syscall``
    introspection is designed to catch.

    Connection objects are passed across process boundaries via
    ``SCM_RIGHTS`` (so the child receives a *fresh* fd number for the
    same underlying kernel object — exactly the case our inode-based
    lookup needs to handle).
    """
    try:
        fd = writer.fileno()
        payload = b"x" * 4096
        while True:
            os.write(fd, payload)
    except OSError:
        return


def _no_op() -> None:  # pragma: no cover - target for the dead-sentinel
    return


@linux_only
def test_get_pipe_write_inode_matches_fstat() -> None:
    """The recovery helper's inode lookup must agree with ``os.fstat``."""
    from zephon.runners.watchdog import _get_pipe_write_inode

    r, w = os.pipe()
    try:

        class _FakeQueue:
            class _Writer:
                def __init__(self, fd: int) -> None:
                    self._fd = fd

                def fileno(self) -> int:
                    return self._fd

            _writer = _Writer(w)

        inode = _get_pipe_write_inode(_FakeQueue())
        assert inode is not None
        assert inode == os.fstat(w).st_ino
    finally:
        os.close(r)
        os.close(w)


def test_get_pipe_write_inode_matches_fstat_socketpair() -> None:
    """Socket write ends qualify too — NamedQueue's default transport."""
    from zephon.runners.watchdog import _get_pipe_write_inode
    from zephon.utils.ipc import socketpair_connections

    reader, writer = socketpair_connections()
    try:

        class _FakeQueue:
            _writer = writer

        inode = _get_pipe_write_inode(_FakeQueue())
        assert inode is not None
        assert inode == os.fstat(writer.fileno()).st_ino
    finally:
        reader.close()
        writer.close()


def test_get_pipe_write_inode_fails_closed_on_unknown_fd_type(
    tmp_path: Path,
) -> None:
    """A write end that is neither a pipe nor a socket must yield None.

    ``_find_worker_fds_for_inode`` can only resolve ``pipe:[inode]`` /
    ``socket:[inode]`` symlinks, so an unrecognized fd type would make
    every worker look like a non-writer and the recovery chain would
    force-release ``_wlock`` under a live writer.  ``None`` instead
    degrades to "can't introspect — leave the lock alone".
    """
    from zephon.runners.watchdog import _get_pipe_write_inode

    with open(tmp_path / "regular_file", "wb") as f:

        class _FakeQueue:
            _writer = f

        assert _get_pipe_write_inode(_FakeQueue()) is None


@linux_only
@pytest.mark.parametrize("transport", TRANSPORTS)
def test_find_worker_fds_for_inode_resolves_inherited_writer(transport: str) -> None:
    """A child that received a Connection across SCM_RIGHTS has a
    *different* fd number for the same kernel object; the helper must
    still resolve the worker fd by matching inodes — over both the
    ``pipe:[inode]`` and ``socket:[inode]`` symlink forms."""
    from zephon.runners.watchdog import _find_worker_fds_for_inode

    ctx = mp.get_context("forkserver")
    reader, writer = _connection_pair(transport, ctx)
    try:
        # Inode of the parent's view of the write end.
        parent_inode = os.fstat(writer.fileno()).st_ino
        proc = ctx.Process(target=_writer_loop, args=(writer,), daemon=True)
        proc.start()
        try:
            # Wait up to 5 s for the child's syscall view to be populated.
            deadline = time.monotonic() + 5.0
            found: set[int] = set()
            ok = False
            while time.monotonic() < deadline:
                assert proc.pid is not None
                found, ok = _find_worker_fds_for_inode(proc.pid, parent_inode)
                if found:
                    break
                time.sleep(0.05)
            assert ok, (
                "Helper returned ok=False — /proc/PID/fd was unreadable "
                "for what was supposed to be a live worker."
            )
            assert found, (
                "Inode-based lookup failed to find ANY worker fd matching "
                f"parent inode {parent_inode}. The pre-fix bug was "
                "comparing parent fd numbers directly; this test catches "
                "regressions of that.  Note: whether the worker's fd "
                "number coincides with the parent's is "
                "implementation-defined (depends on the forkserver's "
                "fd-table state at fork time) and is NOT asserted on — "
                "the inode is the canonical match key."
            )
        finally:
            proc.terminate()
            proc.join(timeout=2.0)
    finally:
        reader.close()
        writer.close()


@linux_only
@pytest.mark.parametrize("transport", TRANSPORTS)
def test_pid_any_thread_in_write_to_fds_detects_active_writer(transport: str) -> None:
    """``_pid_any_thread_in_write_to_fds`` must observe the child in a
    write syscall to one of its target fds at least once across a short
    sample window.  Connection does raw ``os.write`` on socket fds too,
    so both transports surface as write-family syscalls."""
    from zephon.runners.watchdog import (
        _find_worker_fds_for_inode,
        _pid_any_thread_in_write_to_fds,
    )

    ctx = mp.get_context("forkserver")
    reader, writer = _connection_pair(transport, ctx)
    try:
        proc = ctx.Process(target=_writer_loop, args=(writer,), daemon=True)
        proc.start()
        try:
            # Wait for the child's first chunk rather than racing its
            # startup; poll() peeks without draining, so the buffer
            # still fills and the child still blocks in write().
            assert reader.poll(timeout=30.0), "child never wrote its first chunk"
            inode = os.fstat(writer.fileno()).st_ino
            assert proc.pid is not None
            child_fds: set[int] = set()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                child_fds, ok = _find_worker_fds_for_inode(proc.pid, inode)
                assert ok
                if child_fds:
                    break
                time.sleep(0.05)
            assert child_fds, "child must have at least one matching pipe fd"

            # Child blocks in write() against a full pipe.  Sample for
            # up to 1 s — should detect on the first or second sample.
            saw_in_write = False
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                writer_seen, ok = _pid_any_thread_in_write_to_fds(proc.pid, child_fds)
                assert ok
                if writer_seen:
                    saw_in_write = True
                    break
                time.sleep(0.005)
            assert saw_in_write, (
                "Expected to observe the child in a write syscall to "
                "the target pipe at least once across ~1 s of sampling."
            )
        finally:
            proc.terminate()
            proc.join(timeout=2.0)
    finally:
        reader.close()
        writer.close()


@linux_only
@pytest.mark.parametrize("transport", TRANSPORTS)
def test_proc_syscall_no_live_writer_detects_active_writer(transport: str) -> None:
    """The integrated multi-sample helper should return False (live
    writer present) for a child actively writing to the write-end inode.

    This is the integration-of-the-helpers test: any one of inode
    lookup, multi-sampling, or syscall parsing being broken would let
    this fail — i.e. ``_proc_syscall_no_live_writer`` would falsely
    return True and the watchdog would force-release a non-wedged
    lock.  The socketpair leg is the sharp edge: a lookup that only
    understands ``pipe:[inode]`` returns an empty fd set for every live
    worker and positively confirms "no live writer" over the default
    transport."""
    from zephon.runners.watchdog import _proc_syscall_no_live_writer

    ctx = mp.get_context("forkserver")
    reader, writer = _connection_pair(transport, ctx)
    try:
        live_proc = ctx.Process(target=_writer_loop, args=(writer,), daemon=True)
        live_proc.start()
        # Spawn a sentinel to act as the dead worker.
        dead_proc = ctx.Process(target=_no_op, daemon=True)
        dead_proc.start()
        dead_proc.join(timeout=5.0)
        assert not dead_proc.is_alive(), "sentinel should have exited"
        try:
            # As above: wait for the first chunk, don't race startup.
            assert reader.poll(timeout=30.0), "child never wrote its first chunk"
            inode = os.fstat(writer.fileno()).st_ino
            result = _proc_syscall_no_live_writer(
                workers=[live_proc, dead_proc],
                dead_proc=dead_proc,
                pipe_inode=inode,
            )
            assert result is False, (
                "Live worker actively writing to the pipe inode was "
                "NOT detected.  The watchdog would have force-released "
                "a non-wedged lock and risked pipe-stream corruption.  "
                "Likely a regression in inode lookup or multi-sample "
                "syscall introspection."
            )
        finally:
            live_proc.terminate()
            live_proc.join(timeout=2.0)
    finally:
        reader.close()
        writer.close()


@linux_only
def test_proc_syscall_no_live_writer_returns_true_when_only_dead() -> None:
    """No live workers → no one to discriminate → return True (the
    dead worker must have been the lone writer; force-release is safe).
    """
    from zephon.runners.watchdog import _proc_syscall_no_live_writer

    ctx = mp.get_context("forkserver")
    sentinel = ctx.Process(target=_no_op, daemon=True)
    sentinel.start()
    sentinel.join(timeout=5.0)
    assert not sentinel.is_alive()
    result = _proc_syscall_no_live_writer(
        workers=[sentinel], dead_proc=sentinel, pipe_inode=1
    )
    assert result is True


@linux_only
def test_proc_syscall_no_live_writer_returns_true_when_no_one_writing() -> None:
    """A live worker that is *not* writing to the target pipe inode
    (here: no fd at all matching the inode) should yield True — there's
    no live writer to falsely dispossess."""
    from zephon.runners.watchdog import _proc_syscall_no_live_writer

    ctx = mp.get_context("forkserver")
    # Live worker doing nothing relevant — sleeping in a separate process.
    live = ctx.Process(target=_idle_sleep, daemon=True)
    live.start()
    dead = ctx.Process(target=_no_op, daemon=True)
    dead.start()
    dead.join(timeout=5.0)
    try:
        # Use an inode value that no one has a fd for.  The helper must
        # iterate every live worker, fail to find any matching fd, and
        # therefore return True.
        result = _proc_syscall_no_live_writer(
            workers=[live, dead], dead_proc=dead, pipe_inode=2**31 - 1
        )
        assert result is True
    finally:
        live.terminate()
        live.join(timeout=2.0)


@linux_only
def test_pid_any_thread_in_write_to_fds_returns_false_for_unrelated_fd() -> None:
    """``_pid_any_thread_in_write_to_fds`` must NOT match a write
    syscall whose fd argument isn't in the target set."""
    from zephon.runners.watchdog import _pid_any_thread_in_write_to_fds

    ctx = mp.get_context("forkserver")
    reader, writer = ctx.Pipe(duplex=False)
    try:
        proc = ctx.Process(target=_writer_loop, args=(writer,), daemon=True)
        proc.start()
        try:
            time.sleep(0.2)
            assert proc.pid is not None
            # Pretend we care about a fd number that the worker is
            # definitely NOT writing to.  Helper must return
            # (saw_writer=False, ok=True).
            saw, ok = _pid_any_thread_in_write_to_fds(proc.pid, target_fds={9999})
            assert ok
            assert saw is False
        finally:
            proc.terminate()
            proc.join(timeout=2.0)
    finally:
        reader.close()
        writer.close()


@linux_only
def test_find_worker_fds_for_inode_returns_empty_for_nonexistent_inode() -> None:
    """A process that exists but has no fd for the given inode must
    yield an empty set with ok=True (process was reachable)."""
    from zephon.runners.watchdog import _find_worker_fds_for_inode

    # Use our own pid; we definitely don't have a fd for inode 0.
    fds, ok = _find_worker_fds_for_inode(os.getpid(), inode=0)
    assert ok
    assert fds == set()


@linux_only
def test_find_worker_fds_for_inode_handles_dead_process() -> None:
    """A pid that doesn't exist must yield ``(set(), True)`` —
    process-gone is a definite signal, not an unreadable failure."""
    from zephon.runners.watchdog import _find_worker_fds_for_inode

    # Pick a high pid that very likely doesn't exist.
    fds, ok = _find_worker_fds_for_inode(2**30, inode=12345)
    assert ok
    assert fds == set()


@linux_only
def test_pid_any_thread_in_write_to_fds_handles_dead_process() -> None:
    """A pid that doesn't exist must yield
    ``(saw_writer=False, ok=True)`` — gone process is definitely not
    writing."""
    from zephon.runners.watchdog import _pid_any_thread_in_write_to_fds

    saw, ok = _pid_any_thread_in_write_to_fds(2**30, target_fds={1, 2, 3})
    assert ok
    assert saw is False


@linux_only
def test_find_worker_fds_for_inode_fails_closed_on_unreadable_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-fd ``OSError`` from ``os.readlink`` (e.g.
    ``PermissionError``) must propagate as ``ok=False`` rather than
    silently skipping the fd.  If the unreadable fd happens to be the
    result-pipe fd, silently skipping would let the caller conclude
    "no matching fd → not a writer" and force-release while a live
    writer holds the lock.
    """
    from zephon.runners import watchdog as mod

    real_readlink = os.readlink

    def flaky_readlink(path: str) -> str:
        # Raise for the first fd we readlink in /proc/PID/fd; pass
        # everything else through.  This simulates "the result-pipe fd
        # happens to be the one with the access issue".
        if "/proc/" in path and "/fd/" in path:
            raise PermissionError(f"simulated permission denied for {path}")
        return real_readlink(path)

    monkeypatch.setattr(os, "readlink", flaky_readlink)

    fds, ok = mod._find_worker_fds_for_inode(os.getpid(), inode=12345)
    assert ok is False, (
        "Per-fd PermissionError must propagate as ok=False (fail closed). "
        "Silently skipping unreadable fds risks force-releasing the lock "
        "while a live writer holds it."
    )
    assert fds == set()


def test_unwrap_wlock_handles_safesem_wrapper() -> None:
    """The helper should peel a SafeSemLock-style wrapper."""
    from zephon.runners.watchdog import _unwrap_wlock

    class _InnerLock:
        pass

    class _Wrapper:
        def __init__(self, sem: _InnerLock) -> None:
            self._sem = sem

    class _FakeQueue:
        def __init__(self, lock: _Wrapper) -> None:
            self._wlock = lock

    inner = _InnerLock()
    fake = _FakeQueue(_Wrapper(inner))
    assert _unwrap_wlock(fake) is inner


def test_unwrap_wlock_handles_unwrapped_lock() -> None:
    """If ``_wlock`` is the lock itself (no wrapper) the helper should
    return it as-is."""
    from zephon.runners.watchdog import _unwrap_wlock

    class _PlainLock:
        pass

    class _FakeQueue:
        def __init__(self, lock: _PlainLock) -> None:
            self._wlock = lock

    plain = _PlainLock()
    fake = _FakeQueue(plain)
    assert _unwrap_wlock(fake) is plain


def test_unwrap_wlock_handles_missing_attribute() -> None:
    """If the queue has no ``_wlock`` attribute the helper must return
    None rather than raise."""
    from zephon.runners.watchdog import _unwrap_wlock

    class _NoLock:
        pass

    assert _unwrap_wlock(_NoLock()) is None


def test_get_pipe_write_inode_handles_missing_writer() -> None:
    """No ``_writer`` → None, not an exception."""
    from zephon.runners.watchdog import _get_pipe_write_inode

    class _NoWriter:
        pass

    assert _get_pipe_write_inode(_NoWriter()) is None


def test_rotate_result_queue_swaps_and_records_old() -> None:
    """``_rotate_result_queue`` must (a) call the queue factory with a
    label derived from stage+op names, (b) install the new queue on the
    state, and (c) append the old queue to ``_abandoned_result_queues``.
    """
    from zephon.runners.watchdog import _rotate_result_queue

    captured_labels: list[str] = []
    new_queue = object()

    def factory(label: str) -> object:
        captured_labels.append(label)
        return new_queue

    class _Node:
        name = "myop"

    class _State:
        result_queue = "OLD_QUEUE_SENTINEL"
        node = _Node()
        op_index = 7
        stage_name = "mystage"
        _abandoned_result_queues: list[object] = []

    state = _State()
    _rotate_result_queue(state, dead_pid=1234, make_ipc_queue=factory)

    assert captured_labels == ["result:mystage:myop"]
    assert state.result_queue is new_queue
    assert state._abandoned_result_queues == ["OLD_QUEUE_SENTINEL"]


def _idle_sleep() -> None:  # pragma: no cover - test helper
    """Sleep in a syscall that's not write-family."""
    time.sleep(60)
