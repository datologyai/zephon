# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""MTP mode: subprocess-based pipeline execution with seq-number ACK protocol.

The Engine runs in a non-daemon subprocess for GIL isolation.  Items flow
from the subprocess to the main process via a bounded ``mp.Queue``.
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

import multiprocessing as mp
import queue as _queue_mod
import signal
import time
import traceback
import warnings
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.util import Finalize
from typing import Any, Iterator

from zephon.core.constants import SampleRecord, StreamItem
from zephon.core.notify import (
    NotifyArgs,
    _apply_notify_args,
    _extract_notify_args,
    is_sentinel,
)

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
# MTP worker (runs in child process)
# ---------------------------------------------------------------------------


def _mtp_worker(
    pipeline_bytes: bytes,
    restore_ckpt: dict[str, Any] | None,
    data_q: mp.Queue,  # type: ignore[type-arg]
    ctrl_conn: Connection,
    buffer_size: int,
) -> None:
    """Entry point for the MTP worker subprocess.

    Runs in a non-daemon child process.  Deserializes the pipeline,
    builds the Engine, iterates items, and communicates via IPC.
    """
    # Ignore SIGINT in the subprocess — let the main process handle Ctrl-C
    signal.signal(signal.SIGINT, signal.SIG_IGN)

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
    """

    def __init__(
        self,
        pipeline: Any,
        *,
        buffer_size: int = 16,
        restore_ckpt: dict[str, Any] | None = None,
    ) -> None:
        import cloudpickle

        self._buffer_size = buffer_size
        self._data_q: mp.Queue = mp.Queue(maxsize=buffer_size)  # type: ignore[type-arg]

        # Control pipe: main_conn (main process) ↔ sub_conn (subprocess)
        self._main_conn, sub_conn = mp.Pipe()

        pipeline_bytes = cloudpickle.dumps(pipeline)

        self._process = mp.Process(
            target=_mtp_worker,
            args=(pipeline_bytes, restore_ckpt, self._data_q, sub_conn, buffer_size),
            daemon=False,
            name="zephon-mtp-worker",
        )
        self._process.start()
        # Close subprocess end of the pipe in the main process
        sub_conn.close()

        self._closed = False
        self._exhausted = False
        self._last_state: dict[str, Any] | None = None

        # multiprocessing.util.Finalize runs inside _exit_function's
        # _run_finalizers(0) — guaranteed BEFORE non-daemon children are
        # joined, regardless of atexit registration order.
        self._finalizer = Finalize(
            self,
            MTPPipeline._static_close,
            args=(self._process, self._main_conn, self._data_q),
            exitpriority=10,
        )

    @staticmethod
    def _static_close(
        process: mp.Process,
        main_conn: Connection,
        data_q: mp.Queue,  # type: ignore[type-arg]
    ) -> None:
        """Ensure subprocess is shut down — used by Finalize."""
        _shutdown_process(process, main_conn, data_q)

    def __iter__(self) -> Iterator[StreamItem]:
        """Yield items from the subprocess, ACKing each on dequeue."""
        import queue as _queue

        while True:
            # Poll with timeout so we can detect a dead subprocess
            # instead of blocking forever on data_q.get().
            while True:
                try:
                    msg = self._data_q.get(timeout=5.0)
                    break
                except _queue.Empty:
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

        Sends CHECKPOINT, then drains ``data_q`` **without ACKs** to
        unblock the subprocess's put-retry loop (which calls
        ``_drain_ctrl`` every 50 ms).  When the subprocess sees
        CHECKPOINT it responds with STATE_DICT.

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

        import queue as _queue

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

            # Drain one item from data_q (no ACK) to free queue space
            # and unblock the subprocess's put-retry → _drain_ctrl loop.
            try:
                msg = self._data_q.get(timeout=0.05)
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
            except _queue.Empty:
                pass

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
        self._finalizer.cancel()
        _shutdown_process(self._process, self._main_conn, self._data_q, self._exhausted)
