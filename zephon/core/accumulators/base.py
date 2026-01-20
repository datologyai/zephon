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
    def flush(self) -> list[ReadyBatch[T]]:
        """Called on upstream close; return any remaining ready batches.

        Returns:
            List of remaining batches. These are typically partial batches
            that were waiting for more elements.
        """

    def has_pending_data(self) -> bool:
        """Return True if the accumulator has buffered data that would be emitted on flush().

        Default implementation returns False. Subclasses with internal buffers
        should override this to return True when they have pending data.
        """
        return False
