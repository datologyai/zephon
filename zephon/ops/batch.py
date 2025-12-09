# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Batching operator for grouping sample records (lane-pure)."""

from collections import defaultdict
from typing import Optional

from zephon.core.constants import SampleBatch, SampleRecord
from zephon.core.op_base import DefaultFinalize, DefaultSetup
from zephon.core.traits import Buffering, OpTraits


class Batch(DefaultSetup, DefaultFinalize[SampleBatch]):
    """Collect sample records into mini-batches, one lane per batch."""

    def __init__(self, microbatch_size: int, *, drop_last: bool = True) -> None:
        DefaultSetup.__init__(self)

        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        self.microbatch_size = int(microbatch_size)
        self.drop_last = drop_last
        # lane_id -> list[SampleRecord]
        self._buffers: defaultdict[int, list[SampleRecord]] = defaultdict(list)

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=False,
            preserves_cursor_order=True,
            batch_shape_sensitive=False,
            requires_serial_state=True,
        )

    def buffering(self) -> Optional[Buffering]:
        return None

    def _emit(self, chunk: list[SampleRecord]) -> list[SampleBatch]:
        # Sanity: ensure lane purity inside the batch
        if __debug__:
            lane_ids = {r.meta.lane_id for r in chunk}
            assert len(lane_ids) == 1, f"mixed lanes in batch: {lane_ids}"
        batch = SampleBatch(records=tuple(chunk))
        return [batch]

    def _flush_full_batches(self, lane_id: int) -> list[SampleBatch]:
        buf = self._buffers[lane_id]
        if not buf or len(buf) < self.microbatch_size:
            return []

        out: list[SampleBatch] = []
        while len(buf) >= self.microbatch_size:
            chunk = buf[: self.microbatch_size]  # slice copy (unchanged vs previous)
            out.extend(self._emit(chunk))
            del buf[: self.microbatch_size]  # keep buffer tight
        if buf:
            self._buffers[lane_id] = buf
        else:
            self._buffers.pop(lane_id, None)
        return out

    def process_one(self, elem: SampleRecord) -> list[SampleBatch | SampleRecord]:
        assert isinstance(elem, SampleRecord)
        lane_id = elem.meta.lane_id
        buf = self._buffers[lane_id]

        outputs: list[SampleBatch | SampleRecord] = []

        # Tombstones must not affect batch shapes; forward them directly after
        # emitting any ready full batches. Partial buffers stay untouched.
        if elem.meta.tombstone:
            outputs.extend(self._flush_full_batches(lane_id))
            outputs.append(elem)  # forwarded as a record, not batched
        else:
            buf.append(elem)
            outputs.extend(self._flush_full_batches(lane_id))

        return outputs

    def process_many(
        self, elems: list[SampleRecord]
    ) -> list[SampleBatch | SampleRecord]:
        outputs: list[SampleBatch | SampleRecord] = []
        for e in elems:
            outputs.extend(self.process_one(e))
        return outputs

    def finalize(self) -> list[SampleBatch]:
        if not self._buffers:
            return []

        if self.drop_last:
            # Drop any residual partial microbatches for all lanes
            self._buffers.clear()
            return []

        # Emit remaining (possibly smaller) batches per lane
        outputs: list[SampleBatch] = []
        for _, buf in list(self._buffers.items()):
            if buf:
                outputs.extend(self._emit(buf))
        self._buffers.clear()
        return outputs
