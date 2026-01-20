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
        ready: list[ReadyBatch[T]] = []
        now_ns = time.perf_counter_ns()

        for elem in elems:
            if not self._buffer:
                self._first_ts_ns = now_ns
            self._buffer.append(elem)

            # Check count threshold
            if len(self._buffer) >= self._max_batch:
                ready.append(self._emit_batch(now_ns))
            # Check time threshold (only if enabled)
            elif (
                self._max_latency_ms is not None
                and self._first_ts_ns is not None
                and (now_ns - self._first_ts_ns) / 1_000_000 >= self._max_latency_ms
            ):
                ready.append(self._emit_batch(now_ns))

            now_ns = time.perf_counter_ns()

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
