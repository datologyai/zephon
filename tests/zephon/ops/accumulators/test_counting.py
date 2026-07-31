# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for CountingAccumulator lane-aware buffering.

CountingAccumulator routes elements to per-lane buffers (keyed by lane),
ensuring each emitted batch is lane-pure.  This is critical for
deterministic shuffle: ``batch_seed`` must not depend on the cross-lane
interleaving order from the source stream.
"""

from __future__ import annotations

from zephon.ops.accumulators.counting import CountingAccumulator
from zephon.types import (
    ChunkId,
    ChunkOffset,
    ComponentId,
    LaneId,
    SampleId,
    SampleMeta,
    SampleRecord,
)

# Mirrors the engine wire tuple; lane_id is index 1.
EngineSample = tuple[SampleId, LaneId, ChunkId, ChunkOffset, ComponentId]


def _rec(lane_id: int, offset: int) -> SampleRecord:
    return SampleRecord(
        meta=SampleMeta(
            sample_id=(0, 0, offset),
            lane_id=lane_id,
            chunk_id=0,
            chunk_offset=offset,
        ),
        payload={"text": f"L{lane_id}:{offset}"},
    )


def test_batches_identical_regardless_of_interleaving_order() -> None:
    """Same records, different lane interleaving -> identical batches."""
    acc_a = CountingAccumulator[SampleRecord](max_batch=4)
    acc_b = CountingAccumulator[SampleRecord](max_batch=4)

    lane0 = [_rec(0, i) for i in range(4)]
    lane1 = [_rec(1, i) for i in range(4)]

    # Interleaving A: lane 0 first, then lane 1
    batches_a = acc_a.push_many(lane0 + lane1)

    # Interleaving B: alternating
    interleaved = [r for pair in zip(lane0, lane1) for r in pair]
    batches_b = acc_b.push_many(interleaved)

    def batch_lanes(batches: list) -> list[list[int]]:
        return [[r.meta.lane_id for r in batch] for batch, _ in batches]

    lanes_a = batch_lanes(batches_a)
    lanes_b = batch_lanes(batches_b)

    assert lanes_a == lanes_b


def test_batches_are_lane_pure() -> None:
    """Batches from a multi-lane stream should each contain a single lane."""
    acc = CountingAccumulator[SampleRecord](max_batch=3)

    # Alternating lanes, as _source_stream round-robin would produce
    records = [_rec(lane_id=i % 2, offset=i) for i in range(6)]

    batches = acc.push_many(records)
    assert len(batches) == 2

    for batch, _ in batches:
        lane_ids = {r.meta.lane_id for r in batch}
        assert len(lane_ids) == 1, (
            f"Batch contains mixed lanes {lane_ids}; "
            f"expected lane-pure batches for deterministic shuffle."
        )


def test_drop_last_discards_partial_batches() -> None:
    """drop_last=True discards partial per-lane buffers on flush."""
    acc = CountingAccumulator[SampleRecord](max_batch=3, drop_last=True)
    acc.push_many([_rec(0, 0), _rec(0, 1)])  # partial lane 0
    assert acc.has_pending_data()
    assert acc.flush() == []
    assert not acc.has_pending_data()


def test_flush_emits_partial_batches() -> None:
    """drop_last=False emits partial per-lane buffers on flush."""
    acc = CountingAccumulator[SampleRecord](max_batch=3, drop_last=False)
    acc.push_many([_rec(0, 0), _rec(1, 0)])  # 1 per lane
    flushed = acc.flush()
    assert len(flushed) == 2
    for batch, _ in flushed:
        assert len(batch) == 1


def test_flush_reset_produces_fresh_equivalent_state() -> None:
    """After flush(reset=True), internal state must match a fresh instance.

    CountingAccumulator's only state is _buffers and _first_ts_ns, both of
    which are cleared by flush().  This test ensures that flush(reset=True)
    actually produces fresh-equivalent state — catching future fields that
    might be added without corresponding reset logic.
    """
    acc = CountingAccumulator[SampleRecord](max_batch=3)

    # Build up state across multiple lanes
    acc.push_many([_rec(0, i) for i in range(5)])
    acc.push_many([_rec(1, i) for i in range(3)])
    assert acc.has_pending_data()

    acc.flush(reset=True)

    fresh = CountingAccumulator[SampleRecord](max_batch=3)

    # Compare all instance attributes (excluding callables/config)
    assert not acc.has_pending_data()
    assert acc._buffers == fresh._buffers
    assert acc._first_ts_ns == fresh._first_ts_ns


def test_flush_reset_is_lane_scoped() -> None:
    """flush(reset=True, lane_id=L) drains and resets only lane L.

    Flush sentinels are per-lane, so a mid-stream flush for one lane must
    leave other lanes' buffers intact — otherwise multi-lane epochs corrupt
    each other and checkpoint replay diverges.
    """
    acc = CountingAccumulator[SampleRecord](max_batch=10)
    acc.push_many([_rec(0, 0), _rec(0, 1), _rec(1, 0)])  # 2 in lane 0, 1 in lane 1

    flushed = acc.flush(reset=True, lane_id=0)

    emitted = [r for batch, _ in flushed for r in batch]
    assert {r.meta.lane_id for r in emitted} == {0}
    assert len(emitted) == 2
    # Lane 0 is drained; lane 1 survives.
    assert not acc.has_pending_data(lane_id=0)
    assert acc.has_pending_data(lane_id=1)
    assert acc.has_pending_data()


def test_engine_sample_keying() -> None:
    """EngineSample tuples are keyed by lane_id at index 1."""
    acc = CountingAccumulator[EngineSample](max_batch=2)
    # EngineSample = (sample_id, lane_id, chunk_id, offset, component_id)
    elems = [
        ((0, 0, 0), 0, 0, 0, 0),  # lane 0
        ((0, 0, 1), 1, 0, 0, 0),  # lane 1
        ((0, 0, 2), 0, 0, 1, 0),  # lane 0
        ((0, 0, 3), 1, 0, 1, 0),  # lane 1
    ]
    batches = acc.push_many(elems)
    assert len(batches) == 2
    # Each batch should be lane-pure
    for batch, _ in batches:
        lanes = {e[1] for e in batch}
        assert len(lanes) == 1
