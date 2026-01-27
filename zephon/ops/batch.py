# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Batching operator for grouping sample records (lane-pure)."""

from collections import defaultdict
from typing import Any, Sequence

from zephon.core.accumulators import Accumulator, ReadyBatch
from zephon.core.constants import SampleBatch, SampleRecord
from zephon.core.op_base import DefaultSetup
from zephon.core.traits import OpTraits


class BatchAccumulator(Accumulator[SampleRecord]):
    """Accumulator that groups records into lane-pure batches.

    This accumulator runs on the pump thread and maintains per-lane buffers.
    It emits ready batches when a lane reaches the microbatch size. Tombstones
    are handled specially: they trigger a flush of the lane's buffer and are
    then emitted in their own batch.

    The Batch operator receives these lane-pure batches and wraps them
    into SampleBatch objects. Tombstones are passed through unchanged.
    """

    def __init__(self, microbatch_size: int, drop_last: bool = True) -> None:
        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        self.microbatch_size = microbatch_size
        self.drop_last = drop_last
        # lane_id -> list[SampleRecord]
        self._buffers: defaultdict[int, list[SampleRecord]] = defaultdict(list)

    def has_pending_data(self) -> bool:
        """Return True if there are any buffered records."""
        return any(self._buffers.values())

    def push_many(
        self, elems: Sequence[SampleRecord]
    ) -> list[ReadyBatch[SampleRecord]]:
        """Accumulate records and emit lane-pure batches when ready."""
        ready: list[ReadyBatch[SampleRecord]] = []

        for elem in elems:
            lane_id = elem.meta.lane_id
            buf = self._buffers[lane_id]

            # Tombstones must not affect batch shapes; flush full batches
            # then emit the tombstone in its own batch.
            if elem.meta.tombstone:
                ready.extend(self._flush_full_batches(lane_id))
                # Emit tombstone in its own batch so operator can handle it
                ready.append(([elem], 0))
            else:
                buf.append(elem)
                ready.extend(self._flush_full_batches(lane_id))

        return ready

    def flush(self) -> list[ReadyBatch[SampleRecord]]:
        """Emit any remaining buffered records."""
        if not self._buffers:
            return []

        if self.drop_last:
            # Drop any residual partial microbatches for all lanes
            self._buffers.clear()
            return []

        # Emit remaining (possibly smaller) batches per lane
        ready: list[ReadyBatch[SampleRecord]] = []
        for buf in self._buffers.values():
            if buf:
                ready.append((list(buf), 0))
        self._buffers.clear()
        return ready

    def _flush_full_batches(self, lane_id: int) -> list[ReadyBatch[SampleRecord]]:
        """Emit all full batches for a lane."""
        buf = self._buffers[lane_id]
        if not buf or len(buf) < self.microbatch_size:
            return []

        ready: list[ReadyBatch[SampleRecord]] = []
        while len(buf) >= self.microbatch_size:
            chunk = buf[: self.microbatch_size]
            ready.append((list(chunk), 0))
            del buf[: self.microbatch_size]

        if buf:
            self._buffers[lane_id] = buf
        else:
            self._buffers.pop(lane_id, None)

        return ready


class Batch(DefaultSetup):
    """Collect sample records into mini-batches, one lane per batch.

    This operator uses a BatchAccumulator to group records into lane-pure
    batches on the pump thread. The process_many method receives lane-pure
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
            # No longer needs serial state - accumulator handles it
            requires_serial_state=False,
            parallelism=self._parallelism,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return BatchAccumulator(
            microbatch_size=self.microbatch_size,
            drop_last=self.drop_last,
        )

    def process_one(self, elem: SampleRecord) -> list[SampleBatch | SampleRecord]:
        return self.process_many([elem])

    def process_many(
        self, elems: list[SampleRecord]
    ) -> list[SampleBatch | SampleRecord]:
        """Wrap lane-pure batch into SampleBatch.

        The accumulator ensures we receive either:
        - A lane-pure batch of regular records -> wrap into SampleBatch
        - A single tombstone record -> pass through unchanged
        """
        if not elems:
            return []

        # Check if this is a tombstone batch (single tombstone record)
        if len(elems) == 1 and elems[0].meta.tombstone:
            return [elems[0]]

        # Sanity check: ensure lane purity
        if __debug__:
            lane_ids = {r.meta.lane_id for r in elems}
            assert len(lane_ids) == 1, f"mixed lanes in batch: {lane_ids}"

        batch = SampleBatch(records=tuple(elems))
        return [batch]
