# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Counting accumulator that batches elements by count, optionally with time.

Elements are routed to per-lane buffers via a caller-supplied ``key_fn``,
ensuring each emitted batch is lane-pure.  This is critical for
deterministic shuffle: ``batch_seed`` must not depend on the cross-lane
interleaving order from the source stream.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Sequence
from typing import TypeVar

from zephon.core.accumulators.base import Accumulator, ReadyBatch
from zephon.core.constants import lane_of

T = TypeVar("T")


class CountingAccumulator(Accumulator[T]):
    """Accumulator that batches elements by count, optionally with time-based flushing.

    Elements are routed to per-key buffers via ``key_fn`` so each emitted
    batch is key-pure (typically lane-pure).  By default ``key_fn`` uses
    :func:`~zephon.core.constants.lane_of` to extract the lane id.

    When ``max_latency_ms`` is None, this is a pure count-based accumulator
    suitable for deterministic execution.  When set, batches are also flushed
    after the timeout expires (per key).

    When ``drop_last`` is True, ``flush()`` discards partial batches instead
    of emitting them.
    """

    def __init__(
        self,
        max_batch: int,
        max_latency_ms: int | None = None,
        *,
        key_fn: Callable[[T], Hashable] = lane_of,
        drop_last: bool = False,
    ) -> None:
        if max_batch <= 0:
            raise ValueError("max_batch must be positive")
        self._max_batch = max_batch
        self._max_latency_ms = max_latency_ms
        self._key_fn = key_fn
        self._drop_last = drop_last
        self._buffers: dict[Hashable, list[T]] = {}
        self._first_ts_ns: dict[Hashable, int] = {}

    def has_pending_data(self) -> bool:
        """Return True if there are elements in any buffer."""
        return any(self._buffers.values())

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
        key_fn = self._key_fn
        for elem in elems:
            k = key_fn(elem)
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
        key_fn = self._key_fn
        for elem in elems:
            k = key_fn(elem)
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

    def flush(self, *, reset: bool = False) -> list[ReadyBatch[T]]:
        """Emit any remaining buffered elements (or discard if drop_last)."""
        if self._drop_last:
            self._buffers.clear()
            self._first_ts_ns.clear()
            return []

        ready: list[ReadyBatch[T]] = []
        for k in list(self._buffers):
            buf = self._buffers[k]
            if buf:
                ready.append(self._emit_batch_for_key(k))
        self._buffers.clear()
        self._first_ts_ns.clear()
        return ready

    def _emit_batch_for_key(
        self, key: Hashable, now_ns: int | None = None
    ) -> ReadyBatch[T]:
        """Emit the buffer for a single key as a batch."""
        buf = self._buffers.get(key, [])
        first = self._first_ts_ns.pop(key, None)
        if now_ns is None:
            now_ns = time.perf_counter_ns()
        wait_ns = now_ns - first if first is not None else 0
        self._buffers[key] = []
        return (buf, wait_ns)
