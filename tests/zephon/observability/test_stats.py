from __future__ import annotations

import math
from typing import Any

import pytest

from zephon.observability.collector import CollectorConfig, PipelineCollector
from zephon.observability.config import (
    ExecutionTrackingMode,
    MetricsSinkConfig,
    MetricsSinkMode,
)
from zephon.observability.size_estimator import estimate_bytes
from zephon.observability.stats import NodeMetricsDelta, PipelineSummary


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


@pytest.mark.parametrize(
    "value,minimum",
    [
        (None, 0),
        (b"bytes", 5),
        ("text", len("text".encode("utf-8"))),
    ],
)
def test_estimate_bytes_handles_simple_types(value: Any, minimum: int) -> None:
    size = estimate_bytes(value)
    assert size >= minimum


def test_estimate_bytes_handles_nested_containers() -> None:
    nested = {"a": [1, 2, 3], "b": {"inner": b"payload"}}
    size = estimate_bytes(nested)
    assert size >= len(b"payload")
