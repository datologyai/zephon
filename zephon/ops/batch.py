# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Batching operator for grouping sample records (lane-pure)."""

from typing import Any

from zephon.core.accumulators import Accumulator, CountingAccumulator
from zephon.core.constants import SampleBatch, SampleRecord, lane_of
from zephon.core.op_base import DefaultSetup
from zephon.core.traits import OpTraits


class Batch(DefaultSetup):
    """Collect sample records into mini-batches, one lane per batch.

    Uses a lane-keyed CountingAccumulator to group records into lane-pure
    batches on the pump thread.  The process_many method receives lane-pure
    batches and wraps them into SampleBatch objects.

    Since batch formation happens in the accumulator (on the serial pump thread),
    the process_many method is stateless and can safely run with multiple workers.
    """

    def __init__(
        self,
        microbatch_size: int,
        *,
        drop_last: bool = True,
        parallelism: int = 1,
    ) -> None:
        DefaultSetup.__init__(self)

        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        self.microbatch_size = int(microbatch_size)
        self.drop_last = drop_last
        self._parallelism = parallelism

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=False,
            preserves_cursor_order=True,
            batch_shape_sensitive=False,
            requires_serial_state=False,
            parallelism=self._parallelism,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return CountingAccumulator[SampleRecord](
            max_batch=self.microbatch_size, key_fn=lane_of, drop_last=self.drop_last
        )

    def process_one(self, elem: SampleRecord) -> list[SampleBatch | SampleRecord]:
        return self.process_many([elem])

    def process_many(
        self, elems: list[SampleRecord]
    ) -> list[SampleBatch | SampleRecord]:
        """Wrap lane-pure batch into SampleBatch.

        The accumulator provides lane-pure batches of regular records.
        """
        if not elems:
            return []

        # Sanity check: ensure lane purity
        if __debug__:
            lane_ids = {r.meta.lane_id for r in elems}
            assert len(lane_ids) == 1, f"mixed lanes in batch: {lane_ids}"

        batch = SampleBatch(records=tuple(elems))
        return [batch]
