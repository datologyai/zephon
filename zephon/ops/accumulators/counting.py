# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Counting accumulator that batches elements by count, optionally with time.

Elements are routed to per-lane buffers (keyed by :func:`lane_of`), so each
emitted batch is lane-pure.  This is critical for deterministic shuffle:
``batch_seed`` must not depend on the cross-lane interleaving order from the
source stream, and it lets a per-lane flush sentinel reset one lane alone.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TypeVar

from zephon._internal.stream import RunnerStreamIn, lane_of
from zephon.ops.accumulators.base import Accumulator, ReadyBatch

# Bounded to RunnerStreamIn: the accumulator only batches pipeline elements,
# which is exactly what lane_of can key by (SampleRecord/SampleBatch/EngineSample).
T = TypeVar("T", bound=RunnerStreamIn)


class CountingAccumulator(Accumulator[T]):
    """Accumulator that batches elements by count, optionally with time-based flushing.

    Elements are routed to per-lane buffers keyed by :func:`lane_of`, so each
    emitted batch is lane-pure.  ``lane_of`` extracts the lane from whichever
    element type the accumulator carries (a ``SampleRecord``, or a raw
    ``EngineSample`` tuple at the fetch stage), so the buffer key is always the
    lane id — which is what lets a per-lane flush sentinel reset one lane alone.

    When ``max_latency_ms`` is None, this is a pure count-based accumulator
    suitable for deterministic execution.  When set, batches are also flushed
    after the timeout expires (per lane).

    When ``drop_last`` is True, ``flush()`` discards partial batches instead
    of emitting them.
    """

    def __init__(
        self,
        max_batch: int,
        max_latency_ms: int | None = None,
        *,
        drop_last: bool = False,
    ) -> None:
        if max_batch <= 0:
            raise ValueError("max_batch must be positive")
        self._max_batch = max_batch
        self._max_latency_ms = max_latency_ms
        self._drop_last = drop_last
        self._buffers: dict[int, list[T]] = {}
        self._first_ts_ns: dict[int, int] = {}

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        """Return True if buffered elements remain (in ``lane_id`` if given)."""
        if lane_id is None:
            return any(self._buffers.values())
        return bool(self._buffers.get(lane_id))

    def push_many(self, elems: Sequence[T]) -> list[ReadyBatch[T]]:
        """Accumulate elements and emit based on count or time thresholds."""
        if self._max_latency_ms is None:
            return self._push_many_count_only(elems)
        return self._push_many_timed(elems)

    def _push_many_count_only(self, elems: Sequence[T]) -> list[ReadyBatch[T]]:
        """Fast path for deterministic (count-only) mode -- no clock reads."""
        ready: list[ReadyBatch[T]] = []
        bufs = self._buffers
        max_batch = self._max_batch
        lane_fn = lane_of
        for elem in elems:
            k = lane_fn(elem)
            buf = bufs.get(k)
            if buf is None:
                buf = []
                bufs[k] = buf
            buf.append(elem)
            if len(buf) >= max_batch:
                ready.append((buf, 0))
                bufs[k] = []
        return ready

    def _push_many_timed(self, elems: Sequence[T]) -> list[ReadyBatch[T]]:
        """Latency-aware path -- amortises perf_counter_ns calls."""
        ready: list[ReadyBatch[T]] = []
        now_ns = time.perf_counter_ns()
        max_latency_ns = self._max_latency_ms * 1_000_000  # type: ignore[operator]
        lane_fn = lane_of
        for elem in elems:
            k = lane_fn(elem)
            buf = self._buffers.get(k)
            if buf is None:
                buf = []
                self._buffers[k] = buf
            if not buf:
                self._first_ts_ns[k] = now_ns
            buf.append(elem)

            if len(buf) >= self._max_batch:
                now_ns = time.perf_counter_ns()
                ready.append(self._emit_batch_for_key(k, now_ns))
            else:
                first = self._first_ts_ns.get(k)
                if first is not None and (now_ns - first) >= max_latency_ns:
                    now_ns = time.perf_counter_ns()
                    ready.append(self._emit_batch_for_key(k, now_ns))

        return ready

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[T]]:
        """Emit any remaining buffered elements (or discard if drop_last).

        Buffers are keyed by lane, so ``lane_id`` is the buffer key for a
        per-lane flush.
        """
        keys = [lane_id] if lane_id is not None else list(self._buffers)
        ready: list[ReadyBatch[T]] = []
        for k in keys:
            if not self._drop_last and self._buffers.get(k):
                ready.append(self._emit_batch_for_key(k))
            self._buffers.pop(k, None)
            self._first_ts_ns.pop(k, None)
        return ready

    def _emit_batch_for_key(self, key: int, now_ns: int | None = None) -> ReadyBatch[T]:
        """Emit the buffer for a single lane as a batch."""
        buf = self._buffers.get(key, [])
        first = self._first_ts_ns.pop(key, None)
        if now_ns is None:
            now_ns = time.perf_counter_ns()
        wait_ns = now_ns - first if first is not None else 0
        self._buffers[key] = []
        return (buf, wait_ns)
