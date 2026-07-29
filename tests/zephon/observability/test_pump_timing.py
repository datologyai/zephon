"""Tests for PumpTimingDelta / PumpTimingSummary aggregation."""

from __future__ import annotations

from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.stats import (
    PumpTimingDelta,
    PumpTimingSummary,
)


def _delta(**overrides) -> PumpTimingDelta:
    base = dict(
        stage_index=0,
        op_index=0,
        stage_name="stage0",
        name="op0",
        input_wait_ns=1_000,
        dispatch_wait_ns=2_000,
        dispatch_active_ns=3_000,
        result_wait_ns=4_000,
        result_collect_ns=5_000,
        result_handle_ns=6_000,
        idle_drain_ns=7_000,
        batches_submitted=2,
        batches_completed=2,
        capacity_stalls=1,
    )
    base.update(overrides)
    return PumpTimingDelta(**base)


def test_summary_apply_aggregates_per_op() -> None:
    summary = PumpTimingSummary(tracking_mode=ExecutionTrackingMode.NODES)
    summary.apply(_delta())
    summary.apply(_delta(input_wait_ns=500, batches_submitted=3))

    records = summary.to_records()
    assert len(records) == 1
    rec = records[0]
    assert rec["stage"] == 0
    assert rec["op"] == 0
    assert rec["input_wait_ns"] == 1_500
    assert rec["dispatch_active_ns"] == 6_000
    assert rec["batches_submitted"] == 5
    assert rec["batches_completed"] == 4
    assert rec["capacity_stalls"] == 2
    assert rec["total_ns"] == sum(
        rec[name]
        for name in (
            "input_wait_ns",
            "dispatch_wait_ns",
            "dispatch_active_ns",
            "result_wait_ns",
            "result_collect_ns",
            "result_handle_ns",
            "idle_drain_ns",
        )
    )


def test_summary_apply_splits_by_stage_and_op() -> None:
    summary = PumpTimingSummary(tracking_mode=ExecutionTrackingMode.NODES)
    summary.apply(_delta())
    summary.apply(_delta(op_index=1, name="op1"))
    summary.apply(_delta(stage_index=1, stage_name="stage1", op_index=0, name="opA"))

    records = summary.to_records()
    assert len(records) == 3
    keys = sorted((r["stage"], r["op"]) for r in records)
    assert keys == [(0, 0), (0, 1), (1, 0)]


def test_summary_merge_combines_two_summaries() -> None:
    a = PumpTimingSummary(tracking_mode=ExecutionTrackingMode.NODES)
    a.apply(_delta())
    b = PumpTimingSummary(tracking_mode=ExecutionTrackingMode.NODES)
    b.apply(_delta(input_wait_ns=100))
    b.apply(_delta(stage_index=1, stage_name="stage1", name="op_in_stage1"))

    a.merge(b)
    records = a.to_records()
    assert len(records) == 2
    by_stage = {r["stage"]: r for r in records}
    assert by_stage[0]["input_wait_ns"] == 1_100  # a's 1000 + b's 100
    assert by_stage[1]["input_wait_ns"] == 1_000


def test_summary_clone_is_independent() -> None:
    summary = PumpTimingSummary(tracking_mode=ExecutionTrackingMode.NODES)
    summary.apply(_delta())
    clone = summary.clone()
    summary.apply(_delta(input_wait_ns=999_999))
    # The clone should not reflect the post-clone apply.
    rec = clone.to_records()[0]
    assert rec["input_wait_ns"] == 1_000
