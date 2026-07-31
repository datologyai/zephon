from __future__ import annotations

import math
from typing import Any

from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.stats import (
    FetchTimingDelta,
    FetchTimingSummary,
    NodeMetricsDelta,
    PipelineSummary,
    PrefetchTimingDelta,
    PrefetchTimingSummary,
)


def _make_delta(**overrides: Any) -> NodeMetricsDelta:
    base = dict(
        stage_index=0,
        op_index=0,
        stage_name="stage0",
        name="op0",
        processed_ns=1_000_000,
        produced_elements=4,
        consumed_elements=4,
        produced_bytes=1024,
        consumed_bytes=512,
        wait_ns=200_000,
        max_queue_depth=3,
        min_processing_ns=200_000,
        max_processing_ns=300_000,
    )
    base.update(overrides)
    return NodeMetricsDelta(**base)


def test_node_summary_aggregation_and_wait_ratio() -> None:
    summary = PipelineSummary(plan_id="plan", tracking_mode=ExecutionTrackingMode.NODES)
    summary.apply(_make_delta())
    summary.apply(
        _make_delta(processed_ns=2_000_000, produced_elements=6, wait_ns=400_000)
    )

    records = summary.to_records()
    assert len(records) == 1
    rec = records[0]
    assert rec["processed_ns"] == 3_000_000
    assert rec["produced_elements"] == 10
    assert rec["consumed_elements"] == 8
    assert rec["produced_bytes"] == 2048
    assert rec["consumed_bytes"] == 1024
    assert rec["max_queue_depth"] == 3
    assert rec["min_processing_ns"] == 200_000
    assert rec["max_processing_ns"] == 300_000

    summary.compute_wait_ratios()
    records = summary.to_records()
    assert math.isclose(records[0]["wait_ratio"], 1.0)


def test_summary_merge_and_snapshot() -> None:
    delta_a = _make_delta(op_index=0, name="opA")
    delta_b = _make_delta(
        op_index=1, name="opB", processed_ns=500_000, stage_name="stage0"
    )

    summary_1 = PipelineSummary(plan_id="p", tracking_mode=ExecutionTrackingMode.NODES)
    summary_1.apply(delta_a)
    summary_2 = PipelineSummary(plan_id="p", tracking_mode=ExecutionTrackingMode.NODES)
    summary_2.apply(delta_b)

    summary_1.merge(summary_2)
    records = summary_1.to_records()
    assert len(records) == 2
    names = {rec["name"] for rec in records}
    assert names == {"opA", "opB"}


def _fetch_delta(dataset_id: int, shard_id: int, samples: int) -> FetchTimingDelta:
    return FetchTimingDelta(
        stage_index=2,
        dataset_id=dataset_id,
        shard_id=shard_id,
        samples=samples,
        group_ns=10,
        resolve_ns=1,
        open_ns=2,
        read_ns=3,
        close_ns=4,
        retries=0,
        cache_hits=1,
        cache_misses=0,
    )


def test_fetch_summary_keeps_dataset_and_shard_identity() -> None:
    summary = FetchTimingSummary(tracking_mode=ExecutionTrackingMode.NODES)
    summary.apply(_fetch_delta(dataset_id=0, shard_id=7, samples=2))
    summary.apply(_fetch_delta(dataset_id=1, shard_id=7, samples=3))

    stage = summary.stages[2]
    assert stage.totals.samples == 5
    assert set(stage.shard_totals) == {(0, 7), (1, 7)}

    shard_records = {
        (record["dataset_id"], record["shard_id"]): record
        for record in summary.to_records()
        if record["dataset_id"] is not None
    }
    assert shard_records[(0, 7)]["samples"] == 2
    assert shard_records[(1, 7)]["samples"] == 3


def test_fetch_summary_clone_is_isolated() -> None:
    summary = FetchTimingSummary(tracking_mode=ExecutionTrackingMode.NODES)
    summary.apply(_fetch_delta(dataset_id=0, shard_id=1, samples=2))
    clone = summary.clone()

    summary.apply(_fetch_delta(dataset_id=0, shard_id=1, samples=3))

    assert clone.stages[2].totals.samples == 2
    assert clone.stages[2].shard_totals[(0, 1)].samples == 2


def test_prefetch_success_rate_aggregates_stages() -> None:
    summary = PrefetchTimingSummary(tracking_mode=ExecutionTrackingMode.NODES)
    summary.apply(
        PrefetchTimingDelta(
            stage_index=0,
            batch_size=4,
            prefetch_requests=4,
            prefetch_succeeded=3,
            prefetch_failed=1,
        )
    )
    summary.apply(
        PrefetchTimingDelta(
            stage_index=1,
            batch_size=6,
            prefetch_requests=6,
            prefetch_succeeded=2,
            prefetch_failed=4,
        )
    )

    assert summary.success_rate == 0.5
    assert PrefetchTimingSummary().success_rate == 0.0
