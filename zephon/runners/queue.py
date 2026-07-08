# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Named multiprocessing queue with feeder-error detection and SHM backpressure.

CPython's ``multiprocessing.Queue`` silently drops items when the background
``_feed`` thread fails to serialize them.  :class:`NamedQueue` catches these
failures and either retries (for transient ``/dev/shm`` exhaustion) or surfaces
them as :class:`QueueFeederError` exceptions so the pipeline can fail loudly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from multiprocessing import queues as mp_queues
from multiprocessing.context import BaseContext
from typing import Any, get_args

from zephon.utils.ipc import (
    DEFAULT_IPC_BUFFER_BYTES,
    DEFAULT_IPC_TRANSPORT,
    IpcTransport,
    socketpair_connections,
)
from zephon.utils.semaphore import SafeSemLock
from zephon.utils.shm import is_shm_error, shm_usage_str, wait_for_shm_space

# Timeouts for the escalating sentinel-enqueue in _enqueue_sentinel.
_FEEDER_SHORT_TIMEOUT = 5.0
_FEEDER_LONG_TIMEOUT = 120.0


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
    """Queue that tags its feeder thread with a friendly name.

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
    :mod:`zephon.utils.ipc` for why); ``"pipe"`` keeps stock behavior.
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
        self._ignore_epipe = True
        self._name_label = name

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
            if self._sem.acquire(block=True, timeout=_FEEDER_SHORT_TIMEOUT):
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
            if self._sem.acquire(block=True, timeout=_FEEDER_LONG_TIMEOUT):
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

        if not is_shm_error(e):
            self._enqueue_sentinel(tb_str)
            return

        # --- SHM error: wait for space, then re-queue for serialization ----
        label = f"Queue '{self._name_label}'"
        total_waits = 0
        while True:
            total_waits += wait_for_shm_space(label)

            # SHM looks available — re-queue the original item for _feed to
            # re-attempt serialization.  The semaphore slot was released by
            # _feed when serialization failed, so we re-acquire it here.
            try:
                if self._sem.acquire(block=True, timeout=_FEEDER_SHORT_TIMEOUT):
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
        item = super().get(block, timeout)
        if isinstance(item, FeederError):
            raise QueueFeederError(
                f"Queue '{item.queue_name}' feeder thread failed to serialize "
                f"an item (likely /dev/shm exhaustion).  The item was silently "
                f"dropped by CPython's Queue._feed thread. "
                f"{item.shm_usage}\n\n"
                f"Original traceback from feeder thread:\n{item.traceback}"
            )
        return item

    # -- Thread naming & lifecycle ------------------------------------------

    def _start_thread(self) -> None:
        super()._start_thread()  # type: ignore[attr-defined]
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.name = f"QueueFeederThread[{self._name_label}]"

    def close(self) -> None:
        """Close the queue and clean up all internal semaphores."""
        try:
            super().close()
        finally:
            self._sem.cleanup()
            self._rlock.cleanup()
            self._wlock.cleanup()

    # -- Pickling (process spawning only) -----------------------------------

    def __getstate__(self) -> Any:  # noqa: D401 - custom pickle payload
        base_state = super().__getstate__()  # type: ignore[attr-defined]
        return (self._name_label, base_state)

    def __setstate__(self, state: Any) -> None:
        self._name_label, base_state = state
        super().__setstate__(base_state)  # type: ignore[attr-defined]
