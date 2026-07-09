# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""MTP mode: subprocess-based pipeline execution with seq-number ACK protocol.

The Engine runs in a non-daemon subprocess for GIL isolation.  Items flow
from the subprocess to the main process via a bounded ``NamedQueue``.
Acknowledgements flow back via a ``mp.Pipe`` as lightweight ``(tag, seq)``
tuples — the full ``NotifyPayload`` never crosses the process boundary.

Protocol overview::

    [Subprocess]                             [Main Process]
    produce item
      _pending[seq] = item
      data_q.put((item, seq))      ────────> data_q.get() → (item, seq)
                                             ctrl_conn.send((_ACK, seq))
      ctrl_conn.recv() → (_ACK, seq)  <────  yield item to training loop
      _notify_item(engine, _pending.pop(seq), use_monotone)

Prefetch thread
---------------
A low-priority ``_Prefetcher`` thread in the main process drains ``data_q``
into a small local buffer so ``next()`` pops an already-deserialized item
instead of paying the IPC recv + unpickle inline.  The protocol above is
unchanged: the thread never ACKs, so buffered items are still un-ACK'd —
absent from ``state_dict()``, replayed on restore — and ACKs happen at pop
time on the training thread.  ``RuntimeOptions.mtp_prefetch``: None = auto,
0 = disabled.

Shutdown
--------
The subprocess only checks ``ctrl_conn`` for ``_SHUTDOWN`` messages between
items (via ``_drain_ctrl``).  When the main process calls ``close()`` during
partial iteration the subprocess is typically blocked inside
``next(iterator)`` producing the next item and cannot see the pipe message.

This is different from ``ProcessStageRunner``, where stop signals travel on
the *same* queues workers read data from, so they are always visible.  Here
the control channel is separate from the data path.

To handle this we use a SIGTERM escalation strategy:

1. Send ``_SHUTDOWN`` on ``ctrl_conn`` (works if subprocess is idle or in
   ``_wait_for_shutdown``).
2. Short 200 ms grace period for the pipe path.
3. ``terminate()`` → SIGTERM.  The subprocess installs a handler that raises
   ``SystemExit``, unwinding into the ``finally`` block which calls
   ``engine.close()`` for clean runner shutdown.
4. 5 s grace for ``engine.close()`` to finish.
5. ``kill()`` → SIGKILL as last resort.

``multiprocessing.util.Finalize`` (with ``exitpriority=10``) ensures this
sequence runs during interpreter exit *before* ``multiprocessing`` tries to
join non-daemon children — preventing the process from blocking exit.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import queue as _queue_mod
import signal
import sys
import threading
import time
import traceback
import warnings
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from multiprocessing.util import Finalize
from typing import Any, Iterator

from zephon.core.constants import SampleRecord, StreamItem
from zephon.core.mtp_stats import MTPQueueStats
from zephon.core.notify import (
    NotifyArgs,
    _apply_notify_args,
    _extract_notify_args,
    is_sentinel,
)
from zephon.runners.queue import NamedQueue, QueueFeederError
from zephon.utils.fault_handling import setup_faulthandler
from zephon.utils.ipc import (
    DEFAULT_IPC_TRANSPORT,
    DEFAULT_MTP_BUFFER_BYTES,
    IpcTransport,
)
from zephon.utils.rank import rank_ctx


def _resolve_mp_context(spec: BaseContext | str | None) -> BaseContext:
    """Normalize a ``RuntimeOptions.mp_context`` spec to a context (default spawn).

    Mirrors ``Engine._resolve_mp_context``; keep the two in sync.
    """
    if spec is None:
        return mp.get_context("spawn")
    if isinstance(spec, str):
        return mp.get_context(spec)
    return spec


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Timeout for checkpoint operations and subprocess shutdown polls (5 min).
_CHECKPOINT_TIMEOUT_S: float = 300.0

# ---------------------------------------------------------------------------
# IPC message tags
# ---------------------------------------------------------------------------

_ACK = 0
_CHECKPOINT = 1
_SHUTDOWN = 2
_STATE_DICT = 3


# ---------------------------------------------------------------------------
# Sentinels placed on data_q by the subprocess
# ---------------------------------------------------------------------------


class _StopSentinel:
    """Normal end-of-iteration marker."""

    __slots__ = ()


@dataclass(slots=True)
class _ErrorSentinel:
    """Subprocess-side exception propagation."""

    tb: str


# ---------------------------------------------------------------------------
# Prefetch thread
# ---------------------------------------------------------------------------

# Auto capacity for mtp_prefetch=None: a small slice of the IPC queue depth.
_PREFETCH_CAPACITY_DIVISOR = 10
_PREFETCH_MIN_CAPACITY = 4

# +10 ≈ 1/10 CFS weight — training wins CPU; lower would risk GIL priority inversion.
_PREFETCH_NICE_DELTA = 10


def _prefetch_capacity(buffer_size: int) -> int:
    """Local prefetch buffer size for a given IPC queue depth."""
    return max(_PREFETCH_MIN_CAPACITY, buffer_size // _PREFETCH_CAPACITY_DIVISOR)


def _lower_thread_priority() -> None:
    """Best-effort renice of the calling thread (Linux threads are kernel tasks)."""
    if sys.platform != "linux":
        return
    try:
        tid = threading.get_native_id()
        nice = os.getpriority(os.PRIO_PROCESS, tid)
        os.setpriority(os.PRIO_PROCESS, tid, min(19, nice + _PREFETCH_NICE_DELTA))
    except OSError:
        pass


class _Prefetcher:
    """Single reader of ``data_q``: a bounded relay in front of the consumer.

    With ``capacity > 0`` a drain thread pre-pops messages into a local deque
    so ``get()`` returns already-deserialized items; with ``capacity <= 0``
    there is no thread and ``get()`` reads ``data_q`` directly.  Both modes
    are transparent: messages come out in order, and exceptions from
    ``data_q.get()`` (e.g. one-shot ``QueueFeederError`` poisons) re-raise at
    their stream position.

    Never sends ACKs, so buffered messages stay checkpoint-invisible:
    un-ACK'd, absent from ``state_dict()``, replayed on restore.

    The thread parks when the buffer is full and lowers its own scheduling
    priority so the training loop wins CPU contention (Linux only).
    """

    # data_q poll interval — bounds stop() latency, not throughput.
    _GET_POLL_S = 0.5

    def __init__(self, data_q: mp.Queue, capacity: int) -> None:  # type: ignore[type-arg]
        self._data_q = data_q
        self._capacity = capacity
        self._buf: deque[Any] = deque()
        lock = threading.Lock()
        self._not_empty = threading.Condition(lock)
        self._not_full = threading.Condition(lock)
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        if capacity > 0:
            self._thread = threading.Thread(
                target=self._loop,
                name="zephon-mtp-prefetch",
                daemon=True,
            )
            self._thread.start()

    def _loop(self) -> None:
        _lower_thread_priority()
        while not self._stop_evt.is_set():
            with self._not_full:
                while len(self._buf) >= self._capacity and not self._stop_evt.is_set():
                    self._not_full.wait()
            if self._stop_evt.is_set():
                return
            try:
                msg = self._data_q.get(timeout=self._GET_POLL_S)
            except _queue_mod.Empty:
                continue
            except QueueFeederError as exc:
                # One-shot poison message — the queue stream continues past it.
                msg, terminal = exc, False
            except BaseException as exc:  # noqa: BLE001 — re-raised on get()
                msg, terminal = exc, True
            else:
                terminal = isinstance(msg, (_StopSentinel, _ErrorSentinel))
            with self._not_empty:
                self._buf.append(msg)
                self._not_empty.notify()
            if terminal:
                return

    @property
    def buffered(self) -> int:
        """Buffer depth, sampled without locking — may lag by an item."""
        return len(self._buf)

    def get(self, timeout: float) -> Any:
        """Pop the next message; raises ``queue.Empty`` on timeout."""
        if self._thread is None:
            return self._data_q.get(timeout=timeout)
        deadline = time.monotonic() + timeout
        with self._not_empty:
            while not self._buf:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _queue_mod.Empty
                self._not_empty.wait(timeout=remaining)
            msg = self._buf.popleft()
            self._not_full.notify()
        # The drain thread stores exceptions it hit as buffer entries so they
        # surface here in arrival order.  Genuine data messages are (item, seq)
        # tuples or sentinel objects, so this check cannot misfire on data.
        if isinstance(msg, BaseException):
            raise msg
        return msg

    def stop(self) -> None:
        """Stop the drain thread, discarding buffered (never-ACK'd) messages."""
        if self._thread is None:
            return
        self._stop_evt.set()
        with self._not_full:
            self._not_full.notify()
        self._thread.join(timeout=self._GET_POLL_S + 5.0)
        with self._not_empty:
            self._buf.clear()


# ---------------------------------------------------------------------------
# MTP worker (runs in child process)
# ---------------------------------------------------------------------------


def _mtp_worker(
    pipeline_bytes: bytes,
    restore_ckpt: dict[str, Any] | None,
    data_q: mp.Queue,  # type: ignore[type-arg]
    ctrl_conn: Connection,
    buffer_size: int,
    inflight_shm: ctypes.Array[ctypes.c_int] | None = None,
) -> None:
    """Entry point for the MTP worker subprocess.

    Runs in a non-daemon child process.  Deserializes the pipeline,
    builds the Engine, iterates items, and communicates via IPC.
    """
    # Ignore SIGINT in the subprocess — let the main process handle Ctrl-C
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    setup_faulthandler()

    # Handle SIGTERM gracefully: raise SystemExit so the finally block
    # runs engine.close() (clean runner shutdown).  Without this the
    # default SIGTERM handler hard-kills the process and any child
    # workers become orphans.
    def _sigterm_handler(_signum: int, _frame: Any) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    import cloudpickle

    engine = None
    try:
        pipeline = cloudpickle.loads(pipeline_bytes)
        # We cannot iterate the pipeline directly (``for item in pipeline``)
        # because (a) with mtp_mode=True the pipeline's __iter__ would
        # recursively spawn another MTP subprocess, and (b) the inline path
        # applies notifications immediately on yield, whereas MTP defers
        # them until the main process ACKs each item.  _build_raw_iter()
        # gives us the engine iterator without notification wrapping.
        engine, iterator, use_monotone = pipeline._build_raw_iter(
            restore_ckpt=restore_ckpt,
        )

        if inflight_shm is not None:
            engine.attach_inflight_counter(inflight_shm)

        # Pending notify args indexed by sequence number.  Only lightweight
        # NamedTuples of scalars/cursors — never retains the full item payload.
        pending: dict[int, NotifyArgs] = {}
        seq = 0

        shutdown = False
        for item in iterator:
            # Extract lightweight notify args now (before the item leaves
            # scope), so pending never holds full payloads.  All items
            # including tombstones go through the ACK path to preserve
            # notification ordering.
            # Flush sentinels carry dummy cursor data — skip tracking to
            # avoid corrupting engine state.  When the consumer ACKs a
            # sentinel's seq, pending.pop returns None and the notify is
            # simply skipped.
            if isinstance(item, SampleRecord) and item.meta.is_flush_sentinel:
                pass  # don't add to pending
            else:
                pending[seq] = _extract_notify_args(item, use_monotone)

            # Put item on queue, draining ctrl while waiting if queue is full
            while True:
                try:
                    data_q.put((item, seq), timeout=0.05)
                    break
                except _queue_mod.Full:
                    if _drain_ctrl(ctrl_conn, pending, engine):
                        shutdown = True
                        break
            if shutdown:
                break

            seq += 1

            # Opportunistically drain control messages without blocking
            if _drain_ctrl(ctrl_conn, pending, engine):
                break

        if not shutdown:
            # Iteration complete — send stop sentinel.
            # Use timeout+drain like the item put loop: the main process may
            # not have consumed all items yet, so the queue could be full.
            sentinel_sent = False
            while not sentinel_sent:
                try:
                    data_q.put(_StopSentinel(), timeout=0.05)
                    sentinel_sent = True
                except _queue_mod.Full:
                    if _drain_ctrl(ctrl_conn, pending, engine):
                        break
            # Wait for remaining ACKs + commands until SHUTDOWN
            _wait_for_shutdown(ctrl_conn, pending, engine)

    except Exception:
        tb = traceback.format_exc()
        try:
            data_q.put(_ErrorSentinel(tb=tb))
        except Exception:
            pass
    finally:
        # Ignore further SIGTERM during cleanup so engine.close() isn't
        # interrupted by a second terminate() from the main process.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass


def _handle_ctrl_msg(
    tag: int,
    val: Any,
    pending: dict[int, NotifyArgs],
    engine: Any,
    ctrl_conn: Connection,
) -> bool:
    """Dispatch a single control message. Returns True if SHUTDOWN."""
    if tag == _ACK:
        notify = pending.pop(val, None)
        if notify is not None:
            _apply_notify_args(engine, notify)
    elif tag == _CHECKPOINT:
        state = engine.state_dict()
        ctrl_conn.send((_STATE_DICT, state))
    elif tag == _SHUTDOWN:
        return True
    return False


def _drain_ctrl(
    ctrl_conn: Connection,
    pending: dict[int, NotifyArgs],
    engine: Any,
) -> bool:
    """Non-blocking drain of control messages. Returns True if SHUTDOWN seen."""
    while ctrl_conn.poll(0):
        tag, val = ctrl_conn.recv()
        if _handle_ctrl_msg(tag, val, pending, engine, ctrl_conn):
            return True
    return False


def _wait_for_shutdown(
    ctrl_conn: Connection,
    pending: dict[int, NotifyArgs],
    engine: Any,
) -> None:
    """Block on control connection until SHUTDOWN.

    Uses a 5-minute poll timeout per message so the subprocess exits
    even if the main process never sends SHUTDOWN (e.g. unclean exit).
    """
    while True:
        try:
            if not ctrl_conn.poll(timeout=_CHECKPOINT_TIMEOUT_S):
                return  # no message for 5 min — assume main process is gone
            tag, val = ctrl_conn.recv()
        except (EOFError, OSError):
            return
        if _handle_ctrl_msg(tag, val, pending, engine, ctrl_conn):
            return


# ---------------------------------------------------------------------------
# Shutdown helper — shared by MTPPipeline.close() and Finalize
# ---------------------------------------------------------------------------


def _shutdown_process(
    process: mp.Process,
    main_conn: Connection,
    data_q: mp.Queue,  # type: ignore[type-arg]
    already_exhausted: bool = False,
) -> None:
    """Send SHUTDOWN, drain queue, escalate to SIGTERM/SIGKILL, close IPC."""
    try:
        main_conn.send((_SHUTDOWN, None))
    except (BrokenPipeError, OSError):
        pass

    # Short drain to unblock subprocess puts.
    if not already_exhausted:
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            try:
                msg = data_q.get(timeout=0.05)
                if isinstance(msg, (_StopSentinel, _ErrorSentinel)):
                    break
            except Exception:
                break

    # Short grace for pipe-based SHUTDOWN.
    process.join(timeout=0.2)

    # SIGTERM → SystemExit → finally → engine.close()
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)

    # Last resort.
    if process.is_alive():
        process.kill()
        process.join(timeout=2)

    try:
        main_conn.close()
    except Exception:
        pass
    try:
        data_q.close()
        data_q.join_thread()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# MTPPipeline — main-process handle
# ---------------------------------------------------------------------------


class MTPPipeline:
    """Main-process handle managing the MTP worker subprocess and IPC.

    Usage::

        sp = MTPPipeline(pipeline, restore_ckpt=None)
        for item in sp:
            train(item)
        sp.close()

    A ``_Prefetcher`` thread keeps a small local buffer ahead of the
    consumer so ``next()`` normally returns without touching IPC
    (``prefetch``: None = auto-size from ``buffer_size``, 0 = disabled).
    """

    # Shared-memory array size for per-lane inflight counts.
    # Lane IDs are small integers (typically 0..dp_degree*replicas).
    _INFLIGHT_SHM_LANES: int = 256

    def __init__(
        self,
        pipeline: Any,
        *,
        buffer_size: int = 16,
        buffer_bytes: int = DEFAULT_MTP_BUFFER_BYTES,
        transport: IpcTransport = DEFAULT_IPC_TRANSPORT,
        prefetch: int | None = None,
        restore_ckpt: dict[str, Any] | None = None,
    ) -> None:
        import cloudpickle

        self._buffer_size = buffer_size

        # Use the engine's start method (RuntimeOptions.mp_context, default
        # "spawn"), never fork: the parent's live CUDA and obstore tokio runtime
        # don't survive fork(), so a forked child segfaults. The IPC primitives
        # below must share this context with the Process.
        ctx = _resolve_mp_context(pipeline._options.mp_context)

        # NamedQueue over plain ctx.Queue: feeder serialization failures
        # raise QueueFeederError on get() instead of silently dropping
        # items, and the socketpair transport gives the feeder buffer_bytes
        # of run-ahead (a pipe caps at 64 KiB).
        self._data_q: NamedQueue = NamedQueue(
            "mtp-data",
            maxsize=buffer_size,
            ctx=ctx,
            transport=transport,
            buffer_bytes=buffer_bytes,
        )

        # Control pipe: main_conn (main process) ↔ sub_conn (subprocess)
        self._main_conn, sub_conn = ctx.Pipe()

        # Shared-memory inflight counter: subprocess writes, main reads.
        # RawArray (no lock) — single-int writes are atomic on x86/ARM and
        # we tolerate reading a slightly stale value.
        self._inflight_shm: ctypes.Array[ctypes.c_int] = ctx.RawArray(
            ctypes.c_int, self._INFLIGHT_SHM_LANES
        )

        pipeline_bytes = cloudpickle.dumps(pipeline)

        self._process = ctx.Process(
            target=_mtp_worker,
            args=(
                pipeline_bytes,
                restore_ckpt,
                self._data_q,
                sub_conn,
                buffer_size,
                self._inflight_shm,
            ),
            daemon=False,
            name="zephon-mtp-worker",
        )
        self._process.start()
        # Close subprocess end of the pipe in the main process
        sub_conn.close()

        self._closed = False
        self._exhausted = False
        self._last_state: dict[str, Any] | None = None

        # All data_q reads go through the prefetcher (single reader).  With
        # capacity 0 it is a plain passthrough; otherwise a low-priority
        # thread pre-pops messages so __iter__ never blocks on IPC.
        capacity = _prefetch_capacity(buffer_size) if prefetch is None else prefetch
        self._prefetch = _Prefetcher(self._data_q, capacity)

        # multiprocessing.util.Finalize runs inside _exit_function's
        # _run_finalizers(0) — guaranteed BEFORE non-daemon children are
        # joined, regardless of atexit registration order.
        self._finalizer = Finalize(
            self,
            MTPPipeline._static_close,
            args=(self._process, self._main_conn, self._data_q, self._prefetch),
            exitpriority=10,
        )

        # Reports unexpected subprocess death even when the consumer
        # isn't iterating (e.g. training is busy on the GPU); the
        # ``is_alive()`` check in ``__iter__`` only runs while the
        # consumer is blocked on ``data_q``.
        self._watchdog_stop = threading.Event()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="zephon-mtp-watchdog",
            daemon=True,
        )
        self._watchdog.start()

    def _watchdog_loop(self) -> None:
        """Poll the subprocess and shout to stderr on unexpected death."""
        while not self._watchdog_stop.wait(timeout=5.0):
            if self._closed or self._exhausted or sys.is_finalizing():
                return
            if not self._process.is_alive():
                exitcode = self._process.exitcode
                print(
                    "\n"
                    + "!" * 78
                    + "\n"
                    + f"[zephon] MTP subprocess (sub_pid={self._process.pid}) "
                    + f"DIED UNEXPECTEDLY with exit code {exitcode}. "
                    + f"({rank_ctx()})\n"
                    + "         The main process is still running; subsequent "
                    + "iteration will raise RuntimeError.\n"
                    + "         Common causes: OOM (exit=-9/-6), segfault "
                    + "(exit=-11), unhandled exception inside the pipeline.\n"
                    + "!" * 78,
                    file=sys.stderr,
                    flush=True,
                )
                return

    @staticmethod
    def _static_close(
        process: mp.Process,
        main_conn: Connection,
        data_q: mp.Queue,  # type: ignore[type-arg]
        prefetch: _Prefetcher,
    ) -> None:
        """Ensure subprocess is shut down — used by Finalize."""
        prefetch.stop()
        _shutdown_process(process, main_conn, data_q)

    def __iter__(self) -> Iterator[StreamItem]:
        """Yield items from the prefetch buffer, ACKing each on dequeue.

        ACKs are sent here — not in the prefetch thread — so an item is
        marked consumed only when the training loop actually receives it.
        """
        while True:
            # Poll with timeout so we can detect a dead subprocess
            # instead of blocking forever.
            while True:
                try:
                    msg = self._prefetch.get(timeout=5.0)
                    break
                except _queue_mod.Empty:
                    if not self._process.is_alive():
                        self._exhausted = True
                        exitcode = self._process.exitcode
                        raise RuntimeError(
                            f"MTP subprocess exited unexpectedly (exit code {exitcode})"
                        )
                    # Subprocess alive but slow — retry.

            if isinstance(msg, _StopSentinel):
                self._exhausted = True
                return

            if isinstance(msg, _ErrorSentinel):
                self._exhausted = True
                raise RuntimeError(f"MTP pipeline failed:\n{msg.tb}")

            item, seq = msg
            # ACK immediately on dequeue — seq number only, no payload
            self._main_conn.send((_ACK, seq))
            if not is_sentinel(item):
                yield item

    def queue_stats(self) -> MTPQueueStats:
        """Sample the hand-off queue occupancy."""
        try:
            depth = self._data_q.qsize()
        except NotImplementedError:  # macOS: sem_getvalue is unsupported
            depth = -1
        return MTPQueueStats(
            depth=depth,
            capacity=self._buffer_size,
            staged_bytes=self._data_q.staged_bytes(),
            prefetch_depth=self._prefetch.buffered,
        )

    def inflight_summary(self) -> dict[int, int]:
        """Read per-lane inflight chunk counts from shared memory.

        Non-blocking — reads whatever the subprocess last wrote.
        Returns ``{lane_id: count}`` for lanes with count > 0.
        """
        return {
            lane: count
            for lane in range(self._INFLIGHT_SHM_LANES)
            if (count := self._inflight_shm[lane]) > 0
        }

    def checkpoint(self, *, timeout: float = _CHECKPOINT_TIMEOUT_S) -> dict[str, Any]:
        """Request a checkpoint from the subprocess Engine.

        Returns the ``engine.state_dict()`` from the subprocess.

        Correctness note: ``_CHECKPOINT`` and all prior ``_ACK`` messages
        travel on the **same** ``mp.Pipe`` (FIFO).  The subprocess reads
        them sequentially in ``_drain_ctrl``, so every ACK sent before
        ``_CHECKPOINT`` is guaranteed to be applied to the engine before
        ``state_dict()`` is called — there is no ACK–CHECKPOINT ordering
        gap despite the asynchronous production loop.
        """
        if self._closed:
            # Subprocess shut down — return cached state if available.
            if self._last_state is not None:
                return self._last_state
            raise RuntimeError(
                "MTP subprocess shut down but no checkpoint was captured. "
                "In MTP mode, call pipe.checkpoint() explicitly "
                "*before* breaking out of the iteration loop to guarantee "
                "a checkpoint is available."
            )
        self._main_conn.send((_CHECKPOINT, None))
        if not self._main_conn.poll(timeout):
            raise RuntimeError(
                f"Checkpoint timed out after {timeout}s; subprocess may be stuck or dead"
            )
        try:
            tag, val = self._main_conn.recv()
        except EOFError:
            raise RuntimeError("Subprocess exited before responding to checkpoint")
        if tag != _STATE_DICT:
            raise RuntimeError(f"Expected _STATE_DICT response, got tag={tag}")
        self._last_state = val
        return val

    def capture_final_state(self, timeout: float = _CHECKPOINT_TIMEOUT_S) -> None:
        """Best-effort capture of engine state for post-break checkpoint.

        Sends CHECKPOINT, then drains ``data_q`` (through the prefetcher,
        its sole reader) **without ACKs** to unblock the subprocess's
        put-retry loop (which calls ``_drain_ctrl`` every 50 ms).  When
        the subprocess sees CHECKPOINT it responds with STATE_DICT.

        The resulting state reflects only ACK'd items — items drained
        from the queue here are discarded without ACKs, so the engine
        doesn't count them as consumed.

        If the subprocess is stuck in ``next(iterator)`` and can't
        reach ``_drain_ctrl``, this times out and ``_last_state``
        keeps its previous value (from any earlier explicit
        ``checkpoint()`` call, or None).
        """
        if self._closed:
            return
        try:
            self._main_conn.send((_CHECKPOINT, None))
        except (BrokenPipeError, OSError):
            return

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Subprocess died — no point waiting.
            if not self._process.is_alive():
                break

            # Check for STATE_DICT response on ctrl pipe.
            if self._main_conn.poll(0):
                try:
                    tag, val = self._main_conn.recv()
                except EOFError:
                    return
                if tag == _STATE_DICT:
                    self._last_state = val
                    return
                # Ignore unexpected tags (e.g. stale responses).

            # Drain one item (no ACK) to free queue space and unblock the
            # subprocess's put-retry → _drain_ctrl loop.
            try:
                msg = self._prefetch.get(timeout=0.05)
                if isinstance(msg, (_StopSentinel, _ErrorSentinel)):
                    # Subprocess finished — wait for STATE_DICT response.
                    remaining = max(0.0, deadline - time.monotonic())
                    if self._main_conn.poll(remaining):
                        try:
                            tag, val = self._main_conn.recv()
                            if tag == _STATE_DICT:
                                self._last_state = val
                        except EOFError:
                            pass
                    return
            except _queue_mod.Empty:
                pass
            except QueueFeederError as exc:
                # Raising would escape generator teardown into user code —
                # demote to a warning; STATE_DICT may still arrive.
                warnings.warn(
                    f"[zephon] MTP mode: data-queue feeder error while "
                    f"draining for final-state capture: {exc}",
                    stacklevel=2,
                )

        if self._last_state is None:
            warnings.warn(
                "[zephon] MTP mode: failed to capture checkpoint — the "
                "child process was likely blocked inside a slow operation and "
                "could not respond within the timeout. If you need a checkpoint "
                "after early break, call pipe.checkpoint() explicitly *before* "
                "breaking out of the iteration loop.",
                stacklevel=2,
            )

    def close(self) -> None:
        """Shut down the subprocess cleanly."""
        if self._closed:
            return
        self._closed = True
        self._watchdog_stop.set()
        self._finalizer.cancel()
        self._prefetch.stop()
        _shutdown_process(self._process, self._main_conn, self._data_q, self._exhausted)
