import logging

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon import Pipeline as PublicPipeline
from zephon.observability import (
    ExecutionTrackingMode,
    MetricsSinkConfig,
    MetricsSinkMode,
)


def test_pipeline_metrics_snapshot_contains_stage_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rows = [{"text": f"row{i}"} for i in range(6)]
    ds = make_inmem_dataset("tiny", rows)
    ws = FakeIndexableWorkSource(ds, chunk_size=3)

    caplog.set_level(logging.INFO)

    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .batch(microbatch_size=2, drop_last=False)
        .enable_observability(
            tracking=ExecutionTrackingMode.NODES,
            sink=MetricsSinkConfig(
                mode=MetricsSinkMode.LOG,
                flush_interval_s=0.1,
                json_logs=True,
            ),
        )
        .options(
            deterministic=True,
            max_workers=1,
            default_stage_prefetch=0,
            prefetch_batches=0,
            # Metrics collector lives inside the Engine process; needs inline mode.
            mtp_mode=False,
        )
    )

    it = iter(pipe)
    try:
        # Drain entire pipeline to ensure metrics are recorded.
        list(it)
    finally:
        it.close()

    summary = pipe.metrics_snapshot()
    assert summary is not None
    records = summary.to_records()
    assert records, "expected at least one metrics record"
    assert pipe._plan is not None
    assert summary.plan_id == pipe._plan.plan_id
    assert any(rec["produced_elements"] > 0 for rec in records)
    summary.compute_wait_ratios()
    assert all("wait_ratio" in rec for rec in summary.to_records())


def test_timing_snapshots_are_unavailable_without_an_inline_engine() -> None:
    ds = make_inmem_dataset("tiny", [{"text": "row"}])
    pipe = PublicPipeline(FakeIndexableWorkSource(ds, chunk_size=1))

    assert pipe.fetch_timing_snapshot() is None
    assert pipe.prefetch_timing_snapshot() is None

    pipe._sp = object()
    assert pipe.fetch_timing_snapshot() is None
    assert pipe.prefetch_timing_snapshot() is None


@pytest.mark.parametrize(
    ("measure_bytes", "expect_bytes"), [(None, False), (True, True), (False, False)]
)
def test_measure_bytes_controls_in_process_byte_metrics(
    measure_bytes: bool | None, expect_bytes: bool
) -> None:
    rows = [{"text": f"row{i}"} for i in range(6)]
    ws = FakeIndexableWorkSource(make_inmem_dataset("tiny", rows), chunk_size=3)
    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .batch(microbatch_size=2, drop_last=False)
        .enable_observability(
            tracking=ExecutionTrackingMode.NODES, measure_bytes=measure_bytes
        )
        .options(
            deterministic=True,
            max_workers=1,
            default_stage_prefetch=0,
            prefetch_batches=0,
            mtp_mode=False,
        )
    )
    it = iter(pipe)
    try:
        list(it)
    finally:
        it.close()

    summary = pipe.metrics_snapshot()
    assert summary is not None
    records = summary.to_records()
    assert any(rec["produced_elements"] > 0 for rec in records)
    measured = any(rec["produced_bytes"] or rec["consumed_bytes"] for rec in records)
    assert measured is expect_bytes


def test_enable_observability_keeps_measure_bytes_set_via_options() -> None:
    ds = make_inmem_dataset("tiny", [{"text": "row"}])
    pipe = PublicPipeline(FakeIndexableWorkSource(ds, chunk_size=1))

    pipe.options(measure_bytes=True).enable_observability()
    assert pipe._options.measure_bytes is True

    pipe.enable_observability(measure_bytes=False)
    assert pipe._options.measure_bytes is False
