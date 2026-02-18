# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Counting accumulator that batches elements by count, optionally with time."""

from __future__ import annotations

import time
from typing import Optional, Sequence, TypeVar

from zephon.core.accumulators.base import Accumulator, ReadyBatch

T = TypeVar("T")


class CountingAccumulator(Accumulator[T]):
    """Accumulator that batches elements by count, optionally with time-based flushing.

    When max_latency_ms is None, this is a pure count-based accumulator suitable
    for deterministic execution. When max_latency_ms is set, batches are also
    flushed after the timeout expires.
    """

    def __init__(self, max_batch: int, max_latency_ms: Optional[int] = None) -> None:
        """Initialize the counting accumulator.

        Args:
            max_batch: Maximum number of elements per batch. When reached,
                a batch is emitted immediately.
            max_latency_ms: Maximum time to wait before flushing (milliseconds).
                If None, only count-based flushing is used (deterministic mode).
        """
        if max_batch <= 0:
            raise ValueError("max_batch must be positive")
        self._max_batch = max_batch
        self._max_latency_ms = max_latency_ms
        self._buffer: list[T] = []
        self._first_ts_ns: Optional[int] = None

    def has_pending_data(self) -> bool:
        """Return True if there are elements in the buffer."""
        return bool(self._buffer)

    def push_many(self, elems: Sequence[T]) -> list[ReadyBatch[T]]:
        """Accumulate elements and emit based on count or time thresholds."""
        if self._max_latency_ms is None:
            return self._push_many_count_only(elems)
        return self._push_many_timed(elems)

    def _push_many_count_only(self, elems: Sequence[T]) -> list[ReadyBatch[T]]:
        """Fast path for deterministic (count-only) mode -- no clock reads."""
        ready: list[ReadyBatch[T]] = []
        buf = self._buffer
        max_batch = self._max_batch
        for elem in elems:
            buf.append(elem)
            if len(buf) >= max_batch:
                ready.append((buf, 0))
                buf = []
        self._buffer = buf
        return ready

    def _push_many_timed(self, elems: Sequence[T]) -> list[ReadyBatch[T]]:
        """Latency-aware path -- amortises perf_counter_ns calls."""
        ready: list[ReadyBatch[T]] = []
        now_ns = time.perf_counter_ns()
        max_latency_ns = self._max_latency_ms * 1_000_000  # type: ignore[operator]
        for elem in elems:
            if not self._buffer:
                self._first_ts_ns = now_ns
            self._buffer.append(elem)

            if len(self._buffer) >= self._max_batch:
                now_ns = time.perf_counter_ns()
                ready.append(self._emit_batch(now_ns))
            elif (
                self._first_ts_ns is not None
                and (now_ns - self._first_ts_ns) >= max_latency_ns
            ):
                now_ns = time.perf_counter_ns()
                ready.append(self._emit_batch(now_ns))

        return ready

    def flush(self) -> list[ReadyBatch[T]]:
        """Emit any remaining buffered elements."""
        if not self._buffer:
            return []
        return [self._emit_batch()]

    def _emit_batch(self, now_ns: Optional[int] = None) -> ReadyBatch[T]:
        """Emit the current buffer as a batch."""
        current_ns = now_ns if now_ns is not None else time.perf_counter_ns()
        wait_ns = current_ns - (self._first_ts_ns or current_ns)
        result = (self._buffer, wait_ns)
        self._buffer = []
        self._first_ts_ns = None
        return result
