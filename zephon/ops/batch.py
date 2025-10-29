# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Batching operator for grouping sample records (lane-pure)."""

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
        self._buffers: dict[int, list[SampleRecord]] = {}

    def traits(self) -> OpTraits:
        return OpTraits(indexable=False, batch_shape_sensitive=False)

    def buffering(self) -> Optional[Buffering]:
        return None

    def _emit(self, chunk: list[SampleRecord]) -> list[SampleBatch]:
        # Sanity: ensure lane purity inside the batch
        if __debug__:
            lane_ids = {r.meta.lane_id for r in chunk}
            assert len(lane_ids) == 1, f"mixed lanes in batch: {lane_ids}"
        batch = SampleBatch(records=tuple(chunk))
        return [batch]

    def _flush_exact(self, lane_id: int) -> list[SampleBatch]:
        buf = self._buffers.get(lane_id, [])
        if len(buf) < self.microbatch_size:
            return []
        out = self._emit(buf[: self.microbatch_size])
        del buf[: self.microbatch_size]
        if not buf:
            # keep dict tidy to minimize finalize work
            self._buffers.pop(lane_id, None)
        else:
            self._buffers[lane_id] = buf
        return out

    def process_one(self, elem: SampleRecord) -> list[SampleBatch]:
        assert isinstance(elem, SampleRecord)
        lane_id = elem.meta.lane_id
        buf = self._buffers.setdefault(lane_id, [])
        buf.append(elem)

        outputs: list[SampleBatch] = []
        # Flush as many full microbatches as are now available for this lane
        while len(buf) >= self.microbatch_size:
            outputs.extend(self._flush_exact(lane_id))
            buf = self._buffers.get(lane_id, [])
            if not buf:
                break
        return outputs

    def process_many(self, elems: list[SampleRecord]) -> list[SampleBatch]:
        outputs: list[SampleBatch] = []
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
