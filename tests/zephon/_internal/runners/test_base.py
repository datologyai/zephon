# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for BaseOperatorState and StageRunnerBase (via InlineStageRunner)."""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from tests.zephon._internal.runners._helpers import _ctx_services
from zephon._internal.graph import Node, Stage
from zephon._internal.op_base import DefaultSetup
from zephon._internal.ops.batch import Batch
from zephon._internal.ops.pack_sequences import PackSequences
from zephon._internal.runners.inline import InlineStageRunner
from zephon.ops.accumulators.base import Accumulator, ReadyBatch
from zephon.ops.traits import OpTraits
from zephon.types import SampleBatch, SampleMeta, SampleRecord


def _rec(value: int, *, chunk_id: int = 0, lane_id: int = 0) -> SampleRecord:
    """Create a record with a specific chunk_id."""
    meta = SampleMeta(
        sample_id=(0, 0, value),
        lane_id=lane_id,
        chunk_id=chunk_id,
        chunk_offset=value,
    )
    return SampleRecord(meta=meta, payload={"value": value, "length": 1})


def _flush_sentinel(*, lane_id: int = 0, boundary_cid: int = 0) -> SampleRecord:
    """Create a flush sentinel with boundary_cid."""
    meta = SampleMeta(
        sample_id=(0, 0, 0),
        lane_id=lane_id,
        chunk_id=0,
        chunk_offset=0,
        tags={"_flush_sentinel": True, "_boundary_cid": boundary_cid},
    )
    return SampleRecord(meta=meta, payload={})


# ---------------------------------------------------------------------------
# Runner / state helpers
# ---------------------------------------------------------------------------


def _make_pack_runner() -> InlineStageRunner:
    """Create a runner with PackSequences (preserves_cursor_order=False)."""
    op = PackSequences(
        max_length=10, num_bins=2, length_fn=lambda r: r.payload["length"]
    )
    node = Node(name="pack", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")
    return InlineStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )


def _make_batch_runner(batch_size: int = 3) -> InlineStageRunner:
    """Create a runner with Batch (preserves_cursor_order=True)."""
    op = Batch(batch_size, drop_last=True)
    node = Node(name="batch", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")
    return InlineStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )


def _sentinel_batches(
    batches: list[tuple[list[SampleRecord], int]],
) -> list[list[SampleRecord]]:
    """Extract flush-sentinel batches from enqueue output."""
    return [b for b, _ in batches if len(b) == 1 and b[0].meta.is_flush_sentinel]


# ---------------------------------------------------------------------------
# BaseOperatorState._epoch_floor tracking
# ---------------------------------------------------------------------------


def test_epoch_floor_tracks_min_chunk_id() -> None:
    """Floor tracks the minimum chunk_id from regular elements."""
    state = _make_pack_runner().ops[0]
    assert state._epoch_floor is None

    state.enqueue([_rec(0, chunk_id=3), _rec(1, chunk_id=1), _rec(2, chunk_id=5)])
    assert state._epoch_floor == 1


def test_epoch_floor_none_for_preserves_cursor_order() -> None:
    """Floor stays None for preserves_cursor_order=True ops (e.g. Batch)."""
    state = _make_batch_runner().ops[0]
    assert state._preserves_cursor_order is True

    state.enqueue([_rec(0, chunk_id=3), _rec(1, chunk_id=1)])
    assert state._epoch_floor is None


def test_epoch_floor_none_initially() -> None:
    """Floor is None before any records are pushed."""
    runner = _make_pack_runner()
    assert runner.ops[0]._epoch_floor is None
    assert runner.epoch_floor() is None


def test_reset_epoch_floor_clears() -> None:
    """reset_epoch_floor() sets floor back to None."""
    state = _make_pack_runner().ops[0]
    state.enqueue([_rec(0, chunk_id=5)])
    assert state._epoch_floor == 5

    state.reset_epoch_floor()
    assert state._epoch_floor is None


def test_epoch_floor_includes_tombstones() -> None:
    """Tombstones contribute to epoch floor — their chunk_id constrains eviction."""
    state = _make_pack_runner().ops[0]
    tomb_meta = SampleMeta(sample_id=(0, 0, 99), lane_id=0, chunk_id=3).with_tombstone(
        True
    )
    tomb = SampleRecord(meta=tomb_meta, payload={})

    state.enqueue([tomb, _rec(0, chunk_id=5)])
    assert state._epoch_floor == 3


# ---------------------------------------------------------------------------
# StageRunnerBase.epoch_floor() — min across ops
# ---------------------------------------------------------------------------


def test_runner_epoch_floor_min_across_ops() -> None:
    """Runner.epoch_floor() returns the minimum floor across all ops in the stage."""
    op1 = PackSequences(
        max_length=10, num_bins=2, length_fn=lambda r: r.payload["length"]
    )
    op2 = PackSequences(
        max_length=10, num_bins=2, length_fn=lambda r: r.payload["length"]
    )
    node1 = Node(name="pack1", op=op1)
    node2 = Node(name="pack2", op=op2)
    stage = Stage(name="s", nodes=[node1, node2], placement="auto", break_reason="test")

    runner = InlineStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    runner.ops[0].enqueue([_rec(0, chunk_id=5)])
    runner.ops[1].enqueue([_rec(0, chunk_id=3)])

    assert runner.epoch_floor() == 3


def test_runner_epoch_floor_skips_none_ops() -> None:
    """Runner.epoch_floor() skips ops with None floor (preserves_cursor_order)."""
    op1 = Batch(3)  # preserves_cursor_order=True
    op2 = PackSequences(
        max_length=10, num_bins=2, length_fn=lambda r: r.payload["length"]
    )
    node1 = Node(name="batch", op=op1)
    node2 = Node(name="pack", op=op2)
    stage = Stage(name="s", nodes=[node1, node2], placement="auto", break_reason="test")

    runner = InlineStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    runner.ops[0].enqueue([_rec(0, chunk_id=2)])  # Batch: floor stays None
    runner.ops[1].enqueue([_rec(0, chunk_id=7)])  # Pack: floor = 7

    assert runner.ops[0]._epoch_floor is None
    assert runner.epoch_floor() == 7


# ---------------------------------------------------------------------------
# Flush sentinel handling in enqueue()
# ---------------------------------------------------------------------------


def test_flush_sentinel_triggers_accumulator_flush() -> None:
    """Flush sentinel flushes the accumulator for non-preserves_cursor_order ops."""
    state = _make_pack_runner().ops[0]
    batches = state.enqueue(
        [_rec(0, chunk_id=1), _rec(1, chunk_id=1), _flush_sentinel()]
    )
    assert len(_sentinel_batches(batches)) == 1, "Flush sentinel should pass through"


def test_flush_sentinel_advances_epoch_floor() -> None:
    """Flush sentinel advances epoch floor to boundary_cid."""
    state = _make_pack_runner().ops[0]
    state.enqueue([_rec(0, chunk_id=3)])
    assert state._epoch_floor == 3

    state.enqueue([_flush_sentinel(boundary_cid=5)])
    assert state._epoch_floor == 5


def test_flush_sentinel_stalls_behind_batch_accumulator() -> None:
    """Batch(drop_last=True) explicitly stalls and releases later."""
    state = _make_batch_runner().ops[0]
    assert state._preserves_cursor_order is True

    batches = state.enqueue(
        [_rec(0, chunk_id=1), _rec(1, chunk_id=1), _flush_sentinel()]
    )
    assert batches == [], "Nothing should be emitted yet"
    assert len(state._stalled_sentinels) == 1

    # Push a third record — Batch(3) fills and emits.
    batches = state.enqueue([_rec(2, chunk_id=2)])
    assert len(batches) == 2
    assert len(batches[0][0]) == 3, "Full batch of 3 records"
    assert batches[1][0][0].meta.is_flush_sentinel, "Sentinel released after data"
    assert len(state._stalled_sentinels) == 0


def test_flush_sentinel_released_on_force_flush() -> None:
    """Stalled sentinels are released on force flush (end of stream)."""
    state = _make_batch_runner().ops[0]

    batches = state.enqueue([_rec(0, chunk_id=1), _flush_sentinel()])
    assert batches == []
    assert len(state._stalled_sentinels) == 1

    batches = state.enqueue([], force=True)
    assert len(_sentinel_batches(batches)) == 1, (
        "Stalled sentinel released on force flush"
    )


def test_multiple_stalled_sentinels_released_in_order() -> None:
    """Multiple sentinels accumulate and release in order when the buffer fills."""
    state = _make_batch_runner(batch_size=5).ops[0]

    flush1 = _flush_sentinel(boundary_cid=2)
    batches = state.enqueue([_rec(0, chunk_id=1), _rec(1, chunk_id=1), flush1])
    assert batches == []
    assert len(state._stalled_sentinels) == 1

    flush2 = _flush_sentinel(boundary_cid=4)
    batches = state.enqueue([_rec(2, chunk_id=3), flush2])
    assert batches == []
    assert len(state._stalled_sentinels) == 2

    # Push 2 more → buffer hits 5, emits batch, releases both sentinels.
    batches = state.enqueue([_rec(3, chunk_id=5), _rec(4, chunk_id=5)])
    assert len(batches) == 3, (
        f"Expected [data, sentinel1, sentinel2], got {len(batches)}"
    )
    assert len(batches[0][0]) == 5
    assert batches[1][0][0].meta.is_flush_sentinel
    assert batches[2][0][0].meta.is_flush_sentinel
    assert len(state._stalled_sentinels) == 0


def test_multiple_stalled_sentinels_released_on_force() -> None:
    """Force flush releases all stalled sentinels even when data is dropped."""
    state = _make_batch_runner(batch_size=5).ops[0]

    state.enqueue(
        [
            _rec(0, chunk_id=1),
            _flush_sentinel(boundary_cid=2),
            _rec(1, chunk_id=3),
            _flush_sentinel(boundary_cid=4),
        ]
    )
    assert len(state._stalled_sentinels) == 2

    batches = state.enqueue([], force=True)
    assert len(_sentinel_batches(batches)) == 2, "Both stalled sentinels released"
    assert len(state._stalled_sentinels) == 0


def test_stalled_sentinel_release_is_per_lane() -> None:
    """A blocked lane must not hold back another lane's ready sentinel.

    Both lanes stall at boundary_cid=2.  Lane 1 then drains its pre-boundary
    record (its batch fills and emits), so lane 1's sentinel becomes
    releasable while lane 0 — still holding a chunk_id=1 record — stays
    blocked.  The old all-lane ``try_epoch_reset()`` checked every buffer at
    once and stopped at the FIFO head, so lane 0 starved lane 1's release.
    """
    state = _make_batch_runner(batch_size=2).ops[0]

    # Each lane buffers one pre-boundary (chunk_id=1) record, then stalls.
    state.enqueue(
        [_rec(0, chunk_id=1, lane_id=0), _flush_sentinel(lane_id=0, boundary_cid=2)]
    )
    state.enqueue(
        [_rec(1, chunk_id=1, lane_id=1), _flush_sentinel(lane_id=1, boundary_cid=2)]
    )
    assert len(state._stalled_sentinels) == 2

    # A post-boundary record fills lane 1's batch (size 2), draining its
    # pre-boundary record; lane 1's sentinel releases, lane 0's stays stalled.
    batches = state.enqueue([_rec(2, chunk_id=2, lane_id=1)])

    released = _sentinel_batches(batches)
    assert len(released) == 1
    assert released[0][0].meta.lane_id == 1
    assert [s.meta.lane_id for s, _ in state._stalled_sentinels] == [0]


def test_stalled_batch_produces_cross_epoch_batches() -> None:
    """When Batch stalls, cross-epoch mixing in output is expected.

    The batch that triggers sentinel release naturally contains records
    from both epochs.  This is inherent to stalling — the whole point
    is to avoid dropping the partial batch at the boundary.
    """
    state = _make_batch_runner().ops[0]

    batches = state.enqueue(
        [_rec(0, chunk_id=1), _rec(1, chunk_id=1), _flush_sentinel(boundary_cid=2)]
    )
    assert batches == []

    batches = state.enqueue([_rec(2, chunk_id=2)])
    flat = [record for batch, _ in batches for record in batch]
    sentinel_idx = next(i for i, rec in enumerate(flat) if rec.meta.is_flush_sentinel)
    post_boundary = [rec for rec in flat[:sentinel_idx] if rec.meta.chunk_id >= 2]

    assert post_boundary, (
        "stalling Batch should produce cross-epoch output: post-boundary "
        "records appear before the sentinel (this is expected behavior)"
    )


def test_flush_sentinel_order_preservation() -> None:
    """enqueue([regular, flush_sentinel, regular]) processes in order."""
    state = _make_pack_runner().ops[0]

    batches = state.enqueue(
        [_rec(0, chunk_id=1), _flush_sentinel(boundary_cid=2), _rec(1, chunk_id=2)]
    )

    assert len(_sentinel_batches(batches)) == 1, "Flush sentinel must appear in output"
    assert state._epoch_floor == 2


def test_flush_sentinel_dummy_chunk_id_does_not_corrupt_floor() -> None:
    """Flush sentinels have dummy chunk_id=0 but use boundary_cid for floor."""
    state = _make_pack_runner().ops[0]

    state.enqueue([_rec(0, chunk_id=5)])
    assert state._epoch_floor == 5

    # Flush sentinel has chunk_id=0 in its meta, but boundary_cid=6 in tags.
    state.enqueue([_flush_sentinel(boundary_cid=6)])
    assert state._epoch_floor == 6


# ---------------------------------------------------------------------------
# Per-lane flush sentinel scoping
# ---------------------------------------------------------------------------


def _emitted_lanes(batches: list[tuple[list[SampleRecord], int]]) -> set[int]:
    """Lane ids of the non-sentinel records emitted by enqueue()."""
    return {
        r.meta.lane_id for b, _ in batches for r in b if not r.meta.is_flush_sentinel
    }


def test_flush_sentinel_only_flushes_its_own_lane() -> None:
    """A per-lane flush sentinel must not flush other lanes' buffered state.

    The engine injects one flush sentinel per lane.  When lane 0's sentinel
    arrives, only lane 0's accumulator state may flush; lane 1's buffered
    records must survive until lane 1's own sentinel (or end of stream).
    Flushing every lane corrupts the other lanes' epochs mid-stream and
    breaks deterministic replay across checkpoint/restore.
    """
    state = _make_pack_runner().ops[0]

    # Each lane buffers a partial bin (length 1 << max_length=10).
    state.enqueue([_rec(0, chunk_id=1, lane_id=0), _rec(0, chunk_id=1, lane_id=1)])
    assert state.accumulator_impl.has_pending_data()

    batches = state.enqueue([_flush_sentinel(lane_id=0, boundary_cid=2)])

    assert _emitted_lanes(batches) == {0}, "lane-0 sentinel must flush only lane 0"
    assert len(_sentinel_batches(batches)) == 1, "sentinel still passes through"
    # Lane 1's bin is untouched and still pending.
    assert state.accumulator_impl.has_pending_data()


def test_each_lane_sentinel_flushes_independently() -> None:
    """Two lanes flush at their own sentinels, in order, exactly once each."""
    state = _make_pack_runner().ops[0]
    state.enqueue([_rec(0, chunk_id=1, lane_id=0), _rec(0, chunk_id=1, lane_id=1)])

    b0 = state.enqueue([_flush_sentinel(lane_id=0, boundary_cid=2)])
    assert _emitted_lanes(b0) == {0}

    b1 = state.enqueue([_flush_sentinel(lane_id=1, boundary_cid=2)])
    assert _emitted_lanes(b1) == {1}
    assert not state.accumulator_impl.has_pending_data()


# ---------------------------------------------------------------------------
# stall_on_epoch_boundary trait
# ---------------------------------------------------------------------------


class _FlushTrackingAccumulator(Accumulator[SampleRecord]):
    """Accumulator that buffers records and tracks flush calls."""

    def __init__(self) -> None:
        self._buffer: list[SampleRecord] = []
        self.mid_stream_flush_called = False

    def push_many(
        self, elems: Sequence[SampleRecord]
    ) -> list[ReadyBatch[SampleRecord]]:
        self._buffer.extend(elems)
        ready: list[ReadyBatch[SampleRecord]] = []
        while len(self._buffer) >= 3:
            batch = self._buffer[:3]
            self._buffer = self._buffer[3:]
            ready.append((batch, 0))
        return ready

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[SampleRecord]]:
        if reset:
            self.mid_stream_flush_called = True
        result: list[ReadyBatch[SampleRecord]] = []
        if self._buffer:
            result.append((list(self._buffer), 0))
            self._buffer.clear()
        return result

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return bool(self._buffer)

    def try_epoch_reset(
        self, boundary_chunk_id: int, lane_id: int | None = None
    ) -> bool:
        for rec in self._buffer:
            if rec.meta.chunk_id < boundary_chunk_id:
                return False
        return True


class _StallTrackingOp(DefaultSetup):
    """Minimal op that uses _FlushTrackingAccumulator and configurable traits."""

    def __init__(self, *, stall_on_epoch_boundary: bool = False) -> None:
        DefaultSetup.__init__(self)
        self._stall = stall_on_epoch_boundary

    def traits(self) -> OpTraits:
        return OpTraits(
            preserves_cursor_order=True,
            stall_on_epoch_boundary=self._stall,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return _FlushTrackingAccumulator()

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        return [elem]

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        return elems


class _BrokenFlushAccumulator(Accumulator[SampleRecord]):
    """Accumulator that violates the mid-stream flush contract."""

    def __init__(self) -> None:
        self._buffer: list[SampleRecord] = []

    def push_many(
        self, elems: Sequence[SampleRecord]
    ) -> list[ReadyBatch[SampleRecord]]:
        self._buffer.extend(elems)
        return []

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[SampleRecord]]:
        return []

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return bool(self._buffer)


class _BrokenFlushOp(DefaultSetup):
    """Op whose accumulator leaves data behind after mid-stream flush."""

    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return _BrokenFlushAccumulator()

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        return [elem]

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        return elems


def _make_tracking_runner(*, stall_on_epoch_boundary: bool) -> InlineStageRunner:
    op = _StallTrackingOp(stall_on_epoch_boundary=stall_on_epoch_boundary)
    node = Node(name="stall_test", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")
    return InlineStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )


def test_nonstalling_accumulator_flushes_on_epoch_boundary() -> None:
    """Non-stalling accumulators still flush and pass the sentinel through."""
    runner = _make_tracking_runner(stall_on_epoch_boundary=False)
    state = runner.ops[0]
    acc = state.accumulator_impl
    assert isinstance(acc, _FlushTrackingAccumulator)

    # Push 2 records (not enough for a batch of 3) + flush sentinel.
    batches = state.enqueue(
        [_rec(0, chunk_id=1), _rec(1, chunk_id=1), _flush_sentinel(boundary_cid=5)]
    )

    assert acc.mid_stream_flush_called, "flush should be called when not stalling"
    assert len(_sentinel_batches(batches)) == 1, (
        "Sentinel should pass through after flush"
    )
    assert len(state._stalled_sentinels) == 0
    assert not acc.has_pending_data()


def test_epoch_boundary_flush_raises_if_pending_data_remains() -> None:
    """Non-Batch operators must not silently stall after mid-stream flush."""
    op = _BrokenFlushOp()
    node = Node(name="broken_flush", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")
    runner = InlineStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    with pytest.raises(RuntimeError, match="left pending data at an epoch boundary"):
        runner.ops[0].enqueue([_rec(0, chunk_id=1), _flush_sentinel(boundary_cid=2)])


def test_only_batch_supports_stall_on_epoch_boundary() -> None:
    """Non-Batch operators that request stalling are rejected."""
    with pytest.raises(RuntimeError, match="only supported for Batch"):
        _make_tracking_runner(stall_on_epoch_boundary=True)


def test_batch_drop_last_uses_stall_trait() -> None:
    """Batch(drop_last=True) declares stall_on_epoch_boundary=True."""
    assert Batch(3, drop_last=True).traits().stall_on_epoch_boundary is True
    assert Batch(3, drop_last=False).traits().stall_on_epoch_boundary is False


class _NonMonotoneFlushOp(DefaultSetup):
    """Non-stalling, non-monotone op for testing downstream sentinel ordering."""

    def __init__(self) -> None:
        DefaultSetup.__init__(self)

    def traits(self) -> OpTraits:
        return OpTraits(
            preserves_cursor_order=False,
            stall_on_epoch_boundary=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return _FlushTrackingAccumulator()

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        return [elem]

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        return elems


def test_stalled_batch_sentinel_arrives_after_cross_epoch_output() -> None:
    """Downstream non-monotone op sees post-boundary records before the sentinel.

    When Batch stalls, the cross-epoch batch flows to the downstream op
    before the sentinel is released.  The downstream op processes new-epoch
    records with old-epoch state, then sees the sentinel and resets.  This
    is expected: checkpoint/restore correctness is ensured by the
    ReplayFilter (placed before Batch) and the batch-aligned cursor, not
    by strict sentinel ordering within the stage.
    """
    op1 = Batch(3, drop_last=True)
    op2 = _NonMonotoneFlushOp()
    node1 = Node(name="stalling_batch", op=op1)
    node2 = Node(name="downstream_flush", op=op2)
    stage = Stage(name="s", nodes=[node1, node2], placement="auto", break_reason="test")

    runner = InlineStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    outputs = runner._process_pipeline(
        [_rec(0, chunk_id=1), _rec(1, chunk_id=1), _flush_sentinel(boundary_cid=2)],
        force=False,
    )
    assert outputs == []

    outputs = runner._process_pipeline([_rec(2, chunk_id=2)], force=False)
    # Flatten: outputs may contain SampleBatch or SampleRecord or sentinels.
    flat: list[SampleRecord] = []
    for item in outputs:
        if isinstance(item, SampleBatch):
            flat.extend(item.records)
        else:
            flat.append(item)
    sentinel_idx = next(i for i, rec in enumerate(flat) if rec.meta.is_flush_sentinel)
    post_boundary = [rec for rec in flat[:sentinel_idx] if rec.meta.chunk_id >= 2]

    assert post_boundary, (
        "with stalling Batch upstream, the downstream op sees post-boundary "
        "records before the sentinel (this is expected behavior)"
    )
