"""End-to-end test: pump timing flows from a thread runner to the collector."""

from __future__ import annotations

from tests.zephon._internal.runners._helpers import (
    _ctx_services,
    _make_stage,
    _mk_records,
)
from zephon._internal.observability.collector import CollectorConfig, PipelineCollector
from zephon._internal.runners.threads import ThreadStageRunner
from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.stats import NodeMetricsDelta


def _noop_metrics(_: NodeMetricsDelta) -> None:
    return None


def test_thread_runner_emits_pump_timing_to_collector() -> None:
    collector = PipelineCollector(
        CollectorConfig(
            tracking_mode=ExecutionTrackingMode.NODES,
            sink=None,
            report_interval_s=0.05,
            plan_id="plan",
        )
    )

    ctx_services = {
        "record_node_metrics": collector.record,
        "emit_pump_metrics": collector.record_pump_timing,
        "pump_flush_interval_s": 0.05,
    }

    stage = _make_stage(max_delay_ms=2.0, parallelism=2)
    runner = ThreadStageRunner(
        stage,
        ctx_services=ctx_services,
        max_workers=4,
        deterministic=True,
        tracking_mode=ExecutionTrackingMode.NODES,
        stage_output_mode="stream_items",
    )

    records = _mk_records(range(40))
    list(runner.run(iter(records)))

    snapshot = collector.snapshot_pump_timing()
    pump_records = snapshot.to_records()
    assert pump_records, "expected at least one pump-timing record"

    rec = pump_records[0]
    # The pump did dispatch and complete some batches.
    assert rec["batches_submitted"] > 0
    assert rec["batches_completed"] > 0
    # At least one bucket accumulated real wall time.
    assert rec["total_ns"] > 0


def test_thread_runner_no_pump_timing_when_tracking_off() -> None:
    collector = PipelineCollector(
        CollectorConfig(
            tracking_mode=ExecutionTrackingMode.OFF,
            sink=None,
            report_interval_s=5.0,
            plan_id=None,
        )
    )

    calls: list[object] = []

    def emit(delta: object) -> None:
        calls.append(delta)

    ctx_services = {
        "record_node_metrics": _noop_metrics,
        # Even with an emit callable wired, the timer should be disabled
        # because collect_stats is False under tracking_mode=OFF.
        "emit_pump_metrics": emit,
        "pump_flush_interval_s": 0.05,
    }

    stage = _make_stage(max_delay_ms=0.5, parallelism=2)
    runner = ThreadStageRunner(
        stage,
        ctx_services=ctx_services,
        max_workers=2,
        deterministic=False,
        tracking_mode=ExecutionTrackingMode.OFF,
        stage_output_mode="stream_items",
    )

    records = _mk_records(range(10))
    list(runner.run(iter(records)))

    assert calls == []


def _collector() -> PipelineCollector:
    return PipelineCollector(
        CollectorConfig(
            tracking_mode=ExecutionTrackingMode.NODES,
            sink=None,
            report_interval_s=0.05,
            plan_id="plan",
        )
    )


def _pump_services(collector: PipelineCollector) -> dict[str, object]:
    return _ctx_services(
        {
            "record_node_metrics": collector.record,
            "emit_pump_metrics": collector.record_pump_timing,
            "pump_flush_interval_s": 0.05,
        }
    )


def test_thread_runner_charges_instance_starvation_to_dispatch_wait() -> None:
    """With one instance, the pump's blocking acquire counts as a capacity
    stall in dispatch_wait, not dispatch_active.
    """
    collector = _collector()
    # One instance + more work than it can absorb forces repeated acquire stalls.
    stage = _make_stage(max_delay_ms=3.0, parallelism=1)
    runner = ThreadStageRunner(
        stage,
        ctx_services=_pump_services(collector),
        max_workers=2,
        deterministic=True,
        tracking_mode=ExecutionTrackingMode.NODES,
        stage_output_mode="stream_items",
    )

    records = _mk_records(range(40))
    out = list(runner.run(iter(records)))
    assert len(out) == 40

    rec = collector.snapshot_pump_timing().to_records()[0]
    assert rec["capacity_stalls"] > 0, "instance starvation should be counted"
    assert rec["dispatch_wait_ns"] > 0, "acquire wait should land in dispatch_wait"
