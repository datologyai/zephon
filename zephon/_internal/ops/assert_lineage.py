# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Debug operators that help validate lineage ordering guarantees."""

from collections import defaultdict
from typing import Any

from zephon._internal.op_base import DefaultSetup
from zephon.ops.accumulators import Accumulator, PassthroughAccumulator
from zephon.ops.traits import OpTraits
from zephon.types import SampleBatch, SampleCursor, SampleRecord, StreamItem


class AssertLineageOrder(DefaultSetup):
    """Verifies that records for each lane arrive in strictly increasing order.

    This operator is intended for debugging pipelines that use fan-out or complex
    filtering. It raises ``AssertionError`` if any incoming record is not greater
    than the last cursor observed for its lane. Consumers can insert it anywhere in
    the graph; it is a pure pass-through when the ordering is correct.
    """

    def __init__(self) -> None:
        DefaultSetup.__init__(self)
        self._last_per_lane: dict[int, SampleCursor | None] = defaultdict(lambda: None)

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, preserves_cursor_order=True)

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[StreamItem]:
        return PassthroughAccumulator[StreamItem]()

    def _check_cursor(self, lane: int, cursor: SampleCursor) -> None:
        last = self._last_per_lane[lane]
        if last is not None and not (cursor > last):
            raise AssertionError(
                f"Non-monotonic lineage detected on lane {lane}: {cursor.as_key()} <= {last.as_key()}"
            )
        self._last_per_lane[lane] = cursor

    def _check_record(self, record: SampleRecord) -> None:
        self._check_cursor(record.meta.lane_id, record.meta.cursor)

    def _check_batch(self, batch: SampleBatch) -> None:
        lane_ids = set(batch.lane_ids)
        if len(lane_ids) != 1:
            raise AssertionError(
                f"AssertLineageOrder expects lane-pure batches, got {lane_ids}"
            )
        for record in batch.records:
            self._check_record(record)

    def process_one(self, elem: StreamItem) -> list[StreamItem]:
        self._visit(elem)
        return [elem]

    def process_many(self, elems: list[StreamItem]) -> list[StreamItem]:
        for elem in elems:
            self._visit(elem)
        return elems

    def _visit(self, elem: StreamItem) -> None:
        if isinstance(elem, SampleBatch):
            self._check_batch(elem)
            return
        self._check_record(elem)
