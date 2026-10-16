# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Base accumulator abstraction for deterministic parallel operator execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar

T = TypeVar("T")

# Type alias for a batch ready for worker invocation.
# Tuple of (batch elements, wait_ns for metrics).
ReadyBatch = tuple[list[T], int]


class Accumulator(ABC, Generic[T]):
    """Base class for accumulators that run on the pump thread.

    Accumulators consume inputs in deterministic order, maintain any
    cross-invocation state (buffers/bins), and emit ready invocation batches.

    Contract for deterministic parallelism:
    - All cross-invocation, output-producing state must live in the accumulator
    - Worker `process_many` must be deterministic given its input batch
    - Accumulators are invoked only on the pump thread (no concurrent calls)
    """

    @abstractmethod
    def push_many(self, elems: Sequence[T]) -> list[ReadyBatch[T]]:
        """Consume elements in stream order and return 0..N ready invocation batches.

        Args:
            elems: Input elements to accumulate.

        Returns:
            List of ready batches to dispatch to workers. May be empty if
            more elements are needed before a batch is ready.
        """

    @abstractmethod
    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[T]]:
        """Emit any remaining ready batches.

        Called both at upstream close (default, ``reset=False``) and
        mid-stream by flush sentinels (``reset=True``).  Mid-stream flushes
        must reset epoch-local state so replay reproduces the same output.
        Keep source-exhaustion state across resets; notifications are not repeated
        (see :meth:`on_source_exhausted`).

        Flush sentinels are injected **per lane**, so a mid-stream flush is
        scoped to ``lane_id``: only that lane's state is emitted and reset,
        leaving other lanes' epochs untouched.  Flushing every lane at one
        lane's epoch boundary corrupts the others and breaks deterministic
        replay across checkpoint/restore.  ``lane_id=None`` flushes all lanes
        (used at upstream close, when every lane is done).

        Args:
            reset: True for a mid-stream flush: reset epoch-local state but keep
                source-exhaustion state. False at upstream close.
            lane_id: When set, flush only this lane.  When None, flush all
                lanes.

        Returns:
            List of remaining batches. These are typically partial batches
            that were waiting for more elements.
        """

    @property
    def reads_payload(self) -> bool:
        """Whether ``push_many()`` may access element payloads.

        When True, the runner resolves lazy SHM payloads before calling
        ``push_many()``.  Default False — most accumulators only read
        metadata (e.g. ``lane_id``).
        """
        return False

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        """Return True if buffered data would be emitted on flush().

        Default implementation returns False. Subclasses with internal buffers
        should override this to return True when they have pending data.

        Args:
            lane_id: When set, report only this lane's pending state.  When
                None, report whether any lane has pending data.
        """
        return False

    def on_source_exhausted(
        self, lane_id: int, component_id: int
    ) -> list[ReadyBatch[T]]:
        """Handle an EOD notification for a component in this lane.

        The source has permanently run out, but buffered records may still arrive,
        even through order-preserving operators. The default leaves buffers untouched.

        Called at most once per component per lane stream. Keep exhaustion state
        across ``flush(reset=True)``; some lanes close without a notification.
        Restore replays notifications, possibly in a different order across
        components. Reactions must be lane-scoped, idempotent, and deterministic
        for a given input order.

        The ``continues_after_exhaustion`` context entry describes the source's
        continuation policy; it does not suppress notifications. See
        :doc:`/pipelines/user_defined_operators` for usage.

        Return ready batches as in :meth:`push_many`. Keep tombstones in separate
        batches from data.
        """
        return []

    def try_epoch_reset(
        self, boundary_chunk_id: int, lane_id: int | None = None
    ) -> bool:
        """Attempt a delayed epoch reset for stalled flush sentinels.

        Called when a flush sentinel was stalled (held behind buffered data)
        and the runner wants to check whether the accumulator can now be reset.

        Return True only after resetting state so future outputs match a fresh
        accumulator fed the retained records, with the same source-exhaustion
        state. This includes clearing ordering history such as SWRR counters.

        The default implementation is conservative: it returns True only when
        there is no pending data at all.  Accumulators that buffer
        ``SampleRecord`` elements can override this to check ``chunk_id``
        directly and return True as soon as no pre-boundary records remain,
        even if post-boundary records are still buffered.

        Args:
            boundary_chunk_id: The ``_boundary_cid`` from the stalled flush
                sentinel.  Records with ``chunk_id < boundary_chunk_id`` are
                pre-boundary (old epoch).
            lane_id: The stalled sentinel's lane.  When set, only this lane's
                buffered records gate the reset; other lanes are independent.

        Returns:
            True if the reset was performed and the sentinel can be released.
            False if pre-boundary records still exist in the buffer.
        """
        return not self.has_pending_data(lane_id)


__all__ = ["Accumulator", "ReadyBatch"]
