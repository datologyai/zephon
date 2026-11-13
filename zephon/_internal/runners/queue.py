# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Named multiprocessing queue with feeder-error detection and SHM backpressure.

CPython's ``multiprocessing.Queue`` silently drops items when the background
``_feed`` thread fails to serialize them.  :class:`NamedQueue` catches these
failures and either retries (for transient ``/dev/shm`` exhaustion) or surfaces
them as :class:`QueueFeederError` exceptions so the pipeline can fail loudly.
"""

from __future__ import annotations

import errno
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing import queues as mp_queues
from multiprocessing.context import BaseContext
from multiprocessing.util import register_after_fork
from queue import Empty, Full
from types import FunctionType
from typing import TYPE_CHECKING, Any, Protocol, get_args

from zephon._internal.utils.fd_transport import (
    EncodedMessage,
    ReceivedMessage,
)
from zephon._internal.utils.ipc import (
    DEFAULT_IPC_BUFFER_BYTES,
    DEFAULT_IPC_TRANSPORT,
    QueueTransportError,
    SocketConnection,
    read_frame,
    readable_bytes,
    socketpair_connections,
)
from zephon._internal.utils.semaphore import SafeSemLock
from zephon._internal.utils.shm import is_shm_error, shm_usage_str, wait_for_shm_space
from zephon.options import IpcTransport

if TYPE_CHECKING:
    import ctypes
    from collections import deque
    from collections.abc import Callable

    # Shared per-queue state, indexed by _BROKEN, _WRITER_PID and _READER_PID.
    _TransportState = ctypes.Array[ctypes.c_longlong]

# Timeouts for the escalating sentinel-enqueue in _enqueue_sentinel.
_FEEDER_SHORT_TIMEOUT = 5.0
_FEEDER_LONG_TIMEOUT = 120.0
# Semaphore waits cannot be woken by socket shutdown, so they re-check abort this often.
_ABORT_POLL_S = 0.1
# Slots of a socketpair queue's shared _transport_state array.
_BROKEN, _WRITER_PID, _READER_PID = 0, 1, 2
# Cached because os.getpid() is a syscall and every frame records it; refreshed on fork.
_PID = os.getpid()


def _refresh_pid() -> None:
    global _PID
    _PID = os.getpid()


os.register_at_fork(after_in_child=_refresh_pid)


def _use_socket_transport(transport: IpcTransport) -> bool:
    return sys.platform == "linux" and transport == "socketpair"


def _check_transport(state: _TransportState) -> None:
    if state[_BROKEN]:
        raise QueueTransportError(
            "Queue transport aborted; the stream cannot be reused"
        )


class _Lock(Protocol):
    """The acquire/release surface shared by multiprocessing locks and semaphores."""

    def acquire(self, block: bool = ..., timeout: float | None = ...) -> bool: ...

    def release(self) -> None: ...


def _acquire(
    lock: _Lock,
    state: _TransportState,
    block: bool = True,
    timeout: float | None = None,
) -> bool:
    """Acquire a semaphore while checking whether its queue was aborted.

    Socket shutdown cannot wake a semaphore waiter. Bounded waits let a feeder
    or caller exit even when a dead process left the semaphore locked.
    """
    _check_transport(state)
    if lock.acquire(False):
        return True
    if not block:
        return False
    deadline = None if timeout is None else time.monotonic() + max(0, timeout)
    while True:
        _check_transport(state)
        wait = (
            _ABORT_POLL_S
            if deadline is None
            else min(_ABORT_POLL_S, max(0, deadline - time.monotonic()))
        )
        if lock.acquire(True, wait):
            return True
        if deadline is not None and time.monotonic() >= deadline:
            return False


class _WriteLock:
    """Add abort checks and owner tracking to the queue's multiprocessing lock.

    The underlying lock still excludes concurrent writers across processes.
    This adapter checks shared abort state during acquisition so a feeder does
    not wait forever on a lock abandoned by a dead writer. While held, it also
    publishes the writer's PID for the watchdog: a writer dying mid-frame makes
    the stream unsafe to reuse, even if the lock could be force-released.
    """

    def __init__(self, lock: _Lock, state: _TransportState) -> None:
        self.lock, self.state = lock, state

    def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
        try:
            if self.state[_BROKEN]:
                _check_transport(self.state)
            if not self.lock.acquire(False) and not _acquire(
                self.lock, self.state, block, timeout
            ):
                return False
        except BaseException:
            self.state[_BROKEN] = 1
            raise
        self.state[_WRITER_PID] = _PID
        return True

    def release(self) -> None:
        self.state[_WRITER_PID] = 0
        self.lock.release()


class _SocketWriter:
    """Abort the shared stream before a failed feeder releases its write lock."""

    def __init__(
        self,
        reader: SocketConnection,
        writer: SocketConnection,
        state: _TransportState,
        buffer: deque[object],
        condition: threading.Condition,
    ) -> None:
        self.reader, self.writer, self.state = reader, writer, state
        self.buffer, self.condition = buffer, condition

    def abort(self) -> None:
        """Invalidate the stream and wake peers before discarding buffered items."""
        self.state[_BROKEN] = 1
        for endpoint in (self.reader, self.writer):
            try:
                endpoint.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.reader.close()  # Local guard defers close until active recv unwinds.
        with self.condition:
            self.buffer.clear()
            self.buffer.append(mp_queues._sentinel)  # type: ignore[attr-defined]
            self.condition.notify_all()

    def send(self, payload: memoryview) -> None:
        try:
            if self.state[_BROKEN]:
                _check_transport(self.state)
            self.writer.send_bytes(payload)
        except BaseException:
            self.abort()
            raise


class _FrameWriter(_SocketWriter):
    """Carry a serialized message and its owned FDs between the feeder's hooks.

    Queue._feed calls dumps() before acquiring its write lock, then passes the
    returned bytes to send(). Bytes cannot carry descriptor ownership, so
    dumps() stages an EncodedMessage here and returns an empty placeholder.
    send() transmits that staged message under the feeder's write lock and
    releases its descriptors on success or failure. The placeholder never goes
    on the wire and prevents the idle feeder from retaining a large payload.
    """

    def __init__(
        self,
        reader: SocketConnection,
        writer: SocketConnection,
        state: _TransportState,
        buffer: deque[object],
        condition: threading.Condition,
    ) -> None:
        super().__init__(reader, writer, state, buffer, condition)
        self.pending: EncodedMessage | None = None

    def clear(self) -> None:
        if self.pending is not None:
            self.pending.close()
            self.pending = None

    def dumps(self, value: object) -> bytes:
        assert self.pending is None, "Previous queue message was not sent"
        try:
            self.pending = EncodedMessage(value)
        except BaseException as exc:
            # Stock _feed exits before its error hook for these failures.
            if (
                not isinstance(exc, Exception)
                or getattr(exc, "errno", None) == errno.EPIPE
            ):
                self.abort()
            elif isinstance(exc, IndexError):
                # Stock _feed reads IndexError as "buffer empty" and would drop the item.
                raise RuntimeError("Queue serialization failed") from exc
            raise
        # _feed hands this placeholder to send(); the real frame is self.pending.
        return b""

    def send(self, payload: memoryview) -> None:
        message = self.pending
        self.pending = None
        try:
            if self.state[_BROKEN]:
                _check_transport(self.state)
            if message is None:
                raise QueueTransportError("Queue feeder bypassed its codec")
            message.send(self.writer.socket)
        except BaseException:
            # Publish failure before the stock feeder releases its write lock.
            self.abort()
            raise
        finally:
            if message is not None:
                message.close()


def _with_globals(function: FunctionType, **bindings: object) -> FunctionType:
    """Bind a function to a private copy of its global namespace.

    Queue startup and feeding resolve their serializer through module globals.
    Binding those names per queue selects its codec without changing unrelated
    queues or maintaining a separate copy of CPython's function bodies.
    """
    rebound = FunctionType(
        function.__code__,
        {**function.__globals__, **bindings},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    rebound.__kwdefaults__ = function.__kwdefaults__
    return rebound


class _FeederGlobals:
    """Namespace used by Queue._start_thread to select a queue's feeder.

    Startup resolves Queue._feed and Queue's finalizer callbacks. Only _feed
    needs a private serializer; the remaining attributes delegate to Queue.
    """

    def __init__(self, writer: _FrameWriter) -> None:
        self._feed = _with_globals(mp_queues.Queue._feed, _ForkingPickler=writer)

    def __getattr__(self, name: str) -> object:
        return getattr(mp_queues.Queue, name)


@dataclass(frozen=True, slots=True)
class FeederError:
    """Sentinel enqueued when the feeder thread fails to serialize an item.

    Placed on a queue's internal ``_buffer`` when serialization fails (e.g.
    ``/dev/shm`` exhaustion with torch tensors).  Contains only strings so it
    serializes without touching ``/dev/shm``.  The consumer's ``get()`` override
    checks for it and raises :class:`QueueFeederError`.
    """

    queue_name: str
    traceback: str
    shm_usage: str


class QueueFeederError(RuntimeError):
    """Raised when a queue's background feeder thread fails to serialize an item.

    CPython's ``Queue._feed`` thread silently drops items on serialization
    failure.  This exception surfaces that failure with the original traceback
    so the pipeline can fail loudly instead of silently losing batches.
    """


class NamedQueue(mp_queues.Queue):
    """Multiprocessing queue with named feeders and explicit transport failures.

    Uses SafeSemLock for all internal semaphores to ensure proper cleanup in
    free-threaded Python where GC finalizers run in background threads and can
    race with explicit cleanup.

    Overrides ``_on_queue_feeder_error`` so that serialization failures in the
    background feeder thread (e.g. ``/dev/shm`` exhaustion when pickling torch
    tensors) are handled as follows:

    * **SHM errors** (``ENOSPC``): retry indefinitely with exponential-jittered
      backoff, blocking the ``_feed`` thread to provide natural backpressure.
    * **Other errors**: enqueue a :class:`FeederError` sentinel so the
      consumer's ``get()`` raises :class:`QueueFeederError`.

    ``transport="socketpair"`` (default) swaps the stock ``os.pipe()`` for an
    AF_UNIX socketpair carrying ``buffer_bytes`` of kernel buffering (see
    :mod:`zephon._internal.utils.ipc` for why); ``"pipe"`` keeps stock behavior.
    Linux socketpairs carry CPU storage FDs with the pickle. A partial frame
    cannot be skipped safely, so wire failures abort the queue for all peers.
    """

    def __init__(
        self,
        name: str,
        maxsize: int = 0,
        *,
        ctx: BaseContext,
        transport: IpcTransport = DEFAULT_IPC_TRANSPORT,
        buffer_bytes: int | None = DEFAULT_IPC_BUFFER_BYTES,
    ) -> None:
        if transport not in get_args(IpcTransport):
            raise ValueError(
                f"Unknown queue transport {transport!r}; expected 'socketpair' or 'pipe'"
            )
        super().__init__(maxsize, ctx=ctx)
        self._frame_writer: _FrameWriter | None = None
        self._send_bytes: Callable[[memoryview], None] = self._writer.send_bytes
        self._poll: Callable[[float | None], bool] = self._reader.poll
        self._ignore_epipe = True
        self._name_label = name
        # Three aligned shared integers: broken, active writer PID, active reader
        # PID. Only the lock holder writes its PID; zero while a lock is held
        # means an ambiguous crash window, never permission to reclaim it.
        self._transport_state: _TransportState | None = (
            ctx.RawArray("q", 3) if _use_socket_transport(transport) else None
        )

        if transport == "socketpair":
            self._reader.close()
            self._writer.close()
            self._reader, self._writer = socketpair_connections(buffer_bytes)
            # Rebinds _send_bytes/_recv_bytes/_poll to the new pair.
            self._reset()

        # Wrap all semaphores with SafeSemLock for coordinated cleanup
        # in free-threaded Python where GC finalizers run in background threads
        self._sem = SafeSemLock.wrap(self._sem, source=f"queue:{name}:_sem")
        self._rlock = SafeSemLock.wrap(self._rlock, source=f"queue:{name}:_rlock")
        self._wlock = SafeSemLock.wrap(self._wlock, source=f"queue:{name}:_wlock")
        if self._transport_state is not None:
            self._wlock = _WriteLock(self._wlock, self._transport_state)

    def _reset(self, after_fork: bool = False) -> None:
        super()._reset(after_fork)  # type: ignore[attr-defined]
        state = getattr(self, "_transport_state", None)
        if state is not None:
            reader, writer = self._reader, self._writer
            # Only socketpair-backed queues carry transport state.
            assert isinstance(reader, SocketConnection)
            assert isinstance(writer, SocketConnection)
            self._frame_writer = _FrameWriter(
                reader, writer, state, self._buffer, self._notempty
            )
            self._send_bytes = self._frame_writer.send
            reader.close_guard = threading.Lock()

    # -- Feeder error detection ---------------------------------------------

    def _enqueue_sentinel(self, tb_str: str) -> None:
        """Enqueue a :class:`FeederError` sentinel with escalating timeouts.

        Uses a 3-stage strategy to get the sentinel into the queue so that the
        consumer's ``get()`` override can raise :class:`QueueFeederError`:

        1. Short wait (5 s) — should succeed immediately since ``_feed``
           already released a semaphore slot.
        2. Warn to stderr and block longer (120 s).
        3. Give up — log aggressively to stderr.
        """
        shm_str = shm_usage_str()
        sentinel = FeederError(self._name_label, tb_str, shm_str)
        try:
            # Stage 1: short wait — _feed already released a semaphore slot.
            if self._acquire_capacity(_FEEDER_SHORT_TIMEOUT):
                with self._notempty:
                    self._buffer.append(sentinel)
                    self._notempty.notify()
                return

            # Stage 2: warn and block longer.
            print(
                f"WARNING: Queue '{self._name_label}' feeder thread failed to "
                f"serialize an item and cannot re-enqueue the error sentinel "
                f"(queue full for {_FEEDER_SHORT_TIMEOUT}s). Retrying for "
                f"{_FEEDER_LONG_TIMEOUT}s before dropping. "
                f"{shm_str}\n{tb_str}",
                file=sys.stderr,
                flush=True,
            )
            if self._acquire_capacity(_FEEDER_LONG_TIMEOUT):
                with self._notempty:
                    self._buffer.append(sentinel)
                    self._notempty.notify()
                return

            # Stage 3: give up after 2+ minutes — log aggressively.
            msg = (
                f"\n{'!' * 72}\n"
                f"CRITICAL: Queue '{self._name_label}' feeder thread DROPPED "
                f"an item after failing to serialize it AND failing to enqueue "
                f"the error sentinel for "
                f"{_FEEDER_SHORT_TIMEOUT + _FEEDER_LONG_TIMEOUT}s.\n"
                f"Training results may be SILENTLY INCORRECT. "
                f"{shm_str}\n"
                f"Original error:\n{tb_str}"
                f"\n{'!' * 72}\n"
            )
            print(msg, file=sys.stderr, flush=True)
        except QueueTransportError:
            self.abort()  # A peer may have broken the stream; wake this local feeder.
            return
        except Exception:
            # Last resort — something is deeply wrong.
            print(
                f"CRITICAL: Queue '{self._name_label}' feeder error AND failed "
                f"to report it: {shm_str}\n{tb_str}",
                file=sys.stderr,
                flush=True,
            )

    def _on_queue_feeder_error(self, e: BaseException, obj: object) -> None:
        """Called by CPython's ``Queue._feed`` thread on serialization failure.

        For SHM-related errors (``ENOSPC``), retries indefinitely with
        exponential-jittered backoff — blocking the ``_feed`` thread provides
        natural backpressure.  The item is re-queued via ``appendleft`` so
        ``_feed`` re-attempts serialization on its next loop iteration.

        For all other errors, enqueues a :class:`FeederError` sentinel so the
        consumer's ``get()`` raises :class:`QueueFeederError`.
        """
        import traceback as tb_mod

        tb_str = tb_mod.format_exc()
        if self._transport_state is not None:
            assert self._frame_writer is not None
            self._frame_writer.clear()
            if self._transport_state[_BROKEN]:
                self.abort()
                return

        if not is_shm_error(e):
            self._enqueue_sentinel(tb_str)
            return

        # --- SHM error: wait for space, then re-queue for serialization ----
        label = f"Queue '{self._name_label}'"
        total_waits = 0
        transport = self._transport_state
        while True:
            try:
                total_waits += wait_for_shm_space(
                    label,
                    check_cancelled=(lambda: _check_transport(transport))
                    if transport is not None
                    else None,
                )
            except QueueTransportError:
                self.abort()
                return

            # SHM looks available — re-queue the original item for _feed to
            # re-attempt serialization.  The semaphore slot was released by
            # _feed when serialization failed, so we re-acquire it here.
            try:
                acquired = self._acquire_capacity(_FEEDER_SHORT_TIMEOUT)
                if acquired:
                    with self._notempty:
                        self._buffer.appendleft(obj)
                        self._notempty.notify()
                    if total_waits > 0:
                        print(
                            f"INFO: {label} SHM retry succeeded after "
                            f"{total_waits} waits — item re-queued for "
                            f"serialization. {shm_usage_str()}",
                            file=sys.stderr,
                            flush=True,
                        )
                    return
            except QueueTransportError:
                self.abort()
                return
            except Exception as retry_exc:
                print(
                    f"WARNING: {label} SHM retry re-queue failed: "
                    f"{retry_exc!r} {shm_usage_str()}",
                    file=sys.stderr,
                    flush=True,
                )
            # Queue full or re-queue failed — wait again

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """Retrieve an item, raising on feeder-thread errors."""
        state = self._transport_state
        if state is None:
            item = super().get(block, timeout)
        else:
            try:
                frame = self._get_frame(block, timeout)
            except (OSError, ValueError, TypeError):
                # After an abort another thread may close our reader under us;
                # report the abort rather than whichever error that close caused.
                _check_transport(state)
                raise
            item = (
                frame.decode()
                if isinstance(frame, ReceivedMessage)
                else mp_queues._ForkingPickler.loads(frame)
            )
        if isinstance(item, FeederError):
            raise QueueFeederError(
                f"Queue '{item.queue_name}' feeder thread failed to serialize "
                f"an item (likely /dev/shm exhaustion).  The item was silently "
                f"dropped by CPython's Queue._feed thread. "
                f"{item.shm_usage}\n\n"
                f"Original traceback from feeder thread:\n{item.traceback}"
            )
        return item

    def put(
        self, obj: object, block: bool = True, timeout: float | None = None
    ) -> None:
        """Enqueue asynchronously, observing terminal abort during capacity waits."""
        state = self._transport_state
        if state is None:
            return super().put(obj, block, timeout)
        if self._closed:
            raise ValueError(f"Queue {self!r} is closed")
        if state[_BROKEN]:
            _check_transport(state)
        if not self._sem.acquire(False) and not _acquire(
            self._sem, state, block, timeout
        ):
            raise Full
        try:
            with self._notempty:
                if state[_BROKEN]:
                    _check_transport(state)
                if self._thread is None:
                    self._start_thread()
                self._buffer.append(obj)
                self._notempty.notify()
        except BaseException:
            self._sem.release()
            raise

    def _get_frame(self, block: bool, timeout: float | None) -> ReceivedMessage | bytes:
        if self._closed:
            raise ValueError(f"Queue {self!r} is closed")
        state = self._transport_state
        assert state is not None
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        while True:
            if not self._rlock.acquire(False):
                remaining = (
                    None if deadline is None else max(0, deadline - time.monotonic())
                )
                if not _acquire(self._rlock, state, block, remaining):
                    raise Empty
            state[_READER_PID] = _PID
            frame = None
            guard = self._reader.close_guard
            assert guard is not None  # _reset installs it on socketpair queues.
            try:
                with guard:
                    # A local close before any bytes arrive leaves the shared
                    # stream intact; only an interrupted transaction aborts it.
                    if self._reader.closed:
                        _check_transport(state)
                        raise ValueError(f"Queue {self!r} is closed")
                    try:
                        if state[_BROKEN]:
                            _check_transport(state)
                        # No selector calls when a frame is already buffered.
                        # Only the first receive is nonblocking; once any bytes
                        # arrive we own the transaction and finish it normally.
                        frame = read_frame(
                            self._reader.socket,
                            nonblocking_start=True,
                            fd_frame=ReceivedMessage,
                        )
                        if state[_BROKEN]:
                            _check_transport(state)
                        if frame is not None:
                            self._sem.release()
                    except BaseException:
                        self.abort()
                        if isinstance(frame, ReceivedMessage):
                            frame.close()
                        raise
            finally:
                state[_READER_PID] = 0
                self._rlock.release()
                if state[_BROKEN] or self._reader.close_requested:
                    self._reader.close()
            if frame is not None:
                return frame
            if not block:
                raise Empty
            # Only the entry probe can take an idle lock. After that, readiness
            # is required before probing again; timeout ticks just check abort.
            while True:
                _check_transport(state)
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise Empty
                if self._poll(
                    _ABORT_POLL_S
                    if remaining is None
                    else min(_ABORT_POLL_S, remaining)
                ):
                    break

    @property
    def supports_abort(self) -> bool:
        """Whether this queue uses the managed Linux socketpair stream abort() stops."""
        return self._transport_state is not None

    def abort(self) -> None:
        """Invalidate a managed socket stream and wake I/O/semaphore waits."""
        if self._transport_state is not None:
            assert self._frame_writer is not None
            self._frame_writer.abort()

    def _acquire_capacity(self, timeout: float) -> bool:
        if self._transport_state is None:
            return self._sem.acquire(block=True, timeout=timeout)
        return _acquire(self._sem, self._transport_state, timeout=timeout)

    def check_dead_process(self, pid: int) -> bool:
        """Return whether the stream is safe after a process has died.

        A dead transaction owner may have sent or consumed only part of a frame.
        Abort for that PID or a held lock whose owner remains unknown after the
        grace period. Releasing such a lock could expose corrupted framing.
        """
        if self._transport_state is None:
            return True
        wlock = self._wlock.lock if isinstance(self._wlock, _WriteLock) else self._wlock
        for lock, slot in ((wlock, _WRITER_PID), (self._rlock, _READER_PID)):
            deadline = time.monotonic() + 1.0
            while True:
                if lock.acquire(block=False):
                    lock.release()
                    break
                owner = self._transport_state[slot]
                if owner and owner != pid:
                    break
                if owner == pid or time.monotonic() >= deadline:
                    self.abort()
                    return False
                # A live owner may be entering/leaving its critical section.
                time.sleep(0.005)
        return not self._transport_state[_BROKEN]

    def staged_bytes(self) -> int:
        """Serialized bytes sitting in the transport buffer, not yet read.

        Consumable by ``get()`` without the producer's feeder thread being
        scheduled.  Returns -1 once the queue is closed.
        """
        if self._closed:
            return -1
        try:
            return readable_bytes(self._reader.fileno())
        except OSError:  # close() raced us and took the fd
            return -1

    # -- Thread naming & lifecycle ------------------------------------------

    def _start_thread(self) -> None:
        if self._transport_state is None:
            super()._start_thread()  # type: ignore[attr-defined]
        else:
            assert self._frame_writer is not None
            # Run CPython's own feeder loop, but with its `_ForkingPickler` name
            # resolving to our _FrameWriter (FD-aware dumps() and send()).
            _with_globals(
                mp_queues.Queue._start_thread, Queue=_FeederGlobals(self._frame_writer)
            )(self)
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.name = f"QueueFeederThread[{self._name_label}]"

    def close(self) -> None:
        """Close the queue and clean up all internal semaphores."""
        try:
            if self._transport_state is not None and self._transport_state[_BROKEN]:
                self.abort()
            super().close()
        finally:
            wlock = (
                self._wlock.lock if isinstance(self._wlock, _WriteLock) else self._wlock
            )
            for sem in (self._sem, self._rlock, wlock):
                # Spawn deliberately unwraps SafeSemLock; the child must not
                # unlink the parent's named semaphore.
                if isinstance(sem, SafeSemLock):
                    sem.cleanup()

    # -- Pickling (process spawning only) -----------------------------------

    def __getstate__(self) -> Any:  # noqa: D401 - custom pickle payload
        base_state = super().__getstate__()  # type: ignore[attr-defined]
        return (self._name_label, self._transport_state, base_state)

    def __setstate__(self, state: Any) -> None:
        self._name_label, self._transport_state, base_state = state
        super().__setstate__(base_state)  # type: ignore[attr-defined]
        if self._transport_state is not None:
            register_after_fork(self, mp_queues.Queue._after_fork)
