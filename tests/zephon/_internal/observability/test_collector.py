"""Tests for the internal PipelineCollector."""

from __future__ import annotations

from typing import Any

import pytest

from zephon._internal.observability.collector import CollectorConfig, PipelineCollector
from zephon.observability.config import (
    ExecutionTrackingMode,
    MetricsSinkConfig,
    MetricsSinkMode,
)
from zephon.observability.stats import (
    FetchTimingDelta,
    NodeMetricsDelta,
    PrefetchTimingDelta,
    PumpTimingDelta,
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


def _pump_delta(**overrides: Any) -> PumpTimingDelta:
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


def _fetch_delta() -> FetchTimingDelta:
    return FetchTimingDelta(
        stage_index=0,
        dataset_id=1,
        shard_id=2,
        samples=3,
        group_ns=10,
        resolve_ns=1,
        open_ns=2,
        read_ns=3,
        close_ns=4,
        retries=0,
        cache_hits=1,
        cache_misses=0,
    )


def _prefetch_delta() -> PrefetchTimingDelta:
    return PrefetchTimingDelta(
        stage_index=0,
        batch_size=3,
        prefetch_requests=2,
        prefetch_succeeded=1,
        prefetch_failed=1,
    )


def test_pipeline_collector_snapshot_isolated() -> None:
    collector = PipelineCollector(
        CollectorConfig(
            tracking_mode=ExecutionTrackingMode.NODES,
            sink=MetricsSinkConfig(mode=MetricsSinkMode.LOG),
            report_interval_s=1.0,
            plan_id="plan123",
        )
    )
    collector.record(_make_delta())
    snapshot = collector.snapshot()
    records = snapshot.to_records()
    assert len(records) == 1
    assert records[0]["plan_id"] == "plan123"
    collector.reset()
    assert not collector.snapshot().to_records()


def test_collector_records_pump_timing() -> None:
    config = CollectorConfig(
        tracking_mode=ExecutionTrackingMode.NODES,
        sink=None,
        report_interval_s=5.0,
        plan_id="plan-1",
    )
    collector = PipelineCollector(config)
    collector.record_pump_timing(_pump_delta())
    collector.record_pump_timing(_pump_delta(input_wait_ns=42))
    snapshot = collector.snapshot_pump_timing()
    rec = snapshot.to_records()[0]
    assert rec["input_wait_ns"] == 1_042
    assert rec["plan_id"] == "plan-1"


def test_collector_pump_timing_no_op_when_tracking_off() -> None:
    config = CollectorConfig(
        tracking_mode=ExecutionTrackingMode.OFF,
        sink=None,
        report_interval_s=5.0,
        plan_id=None,
    )
    collector = PipelineCollector(config)
    collector.record_pump_timing(_pump_delta())
    snapshot = collector.snapshot_pump_timing()
    assert snapshot.to_records() == []


@pytest.mark.parametrize(
    ("tracking_mode", "has_samples"),
    [
        (ExecutionTrackingMode.OFF, False),
        (ExecutionTrackingMode.STAGES, False),
        (ExecutionTrackingMode.NODES, True),
    ],
)
def test_fetch_and_prefetch_require_node_tracking(
    tracking_mode: ExecutionTrackingMode, has_samples: bool
) -> None:
    collector = PipelineCollector(CollectorConfig(tracking_mode=tracking_mode))
    collector.record_fetch(_fetch_delta())
    collector.record_prefetch(_prefetch_delta())

    assert collector.snapshot_fetch().has_samples() is has_samples
    assert collector.snapshot_prefetch().has_samples() is has_samples


def test_fetch_and_prefetch_snapshots_are_isolated() -> None:
    collector = PipelineCollector(
        CollectorConfig(tracking_mode=ExecutionTrackingMode.NODES)
    )
    collector.record_fetch(_fetch_delta())
    collector.record_prefetch(_prefetch_delta())
    fetch_snapshot = collector.snapshot_fetch()
    prefetch_snapshot = collector.snapshot_prefetch()

    collector.record_fetch(_fetch_delta())
    collector.record_prefetch(_prefetch_delta())

    assert fetch_snapshot.stages[0].totals.samples == 3
    assert prefetch_snapshot.stages[0].samples == 3
