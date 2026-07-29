"""Tests for the internal PipelineCollector."""

from __future__ import annotations

from typing import Any

from zephon._internal.observability.collector import CollectorConfig, PipelineCollector
from zephon.observability.config import (
    ExecutionTrackingMode,
    MetricsSinkConfig,
    MetricsSinkMode,
)
from zephon.observability.stats import NodeMetricsDelta, PumpTimingDelta


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
