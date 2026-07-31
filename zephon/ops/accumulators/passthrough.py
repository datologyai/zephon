# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Passthrough accumulator that emits input microbatches as-is."""

from __future__ import annotations

from typing import Sequence, TypeVar

from zephon.ops.accumulators.base import Accumulator, ReadyBatch

T = TypeVar("T")


class PassthroughAccumulator(Accumulator[T]):
    """Accumulator that emits exactly the incoming microbatch as one invocation batch.

    This is the default accumulator for stateless operators. It preserves the
    upstream batching without any additional buffering.
    """

    def push_many(self, elems: Sequence[T]) -> list[ReadyBatch[T]]:
        """Pass through the input microbatch as-is."""
        if not elems:
            return []
        batch = elems if isinstance(elems, list) else list(elems)
        return [(batch, 0)]

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[T]]:
        """No buffered state to flush."""
        return []


__all__ = ["PassthroughAccumulator"]
