import logging

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon.api import Pipeline as PublicPipeline
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
        )
    )

    it = iter(pipe)
    try:
        # Drain entire pipeline to ensure metrics are recorded.
        list(it)
    finally:
        it.close()

    engine = pipe._engine
    assert engine is not None
    summary = engine.metrics_snapshot()
    assert summary is not None
    records = summary.to_records()
    assert records, "expected at least one metrics record"
    assert pipe._plan is not None
    assert summary.plan_id == pipe._plan.plan_id
    assert any(rec["produced_elements"] > 0 for rec in records)
    summary.compute_wait_ratios()
    assert all("wait_ratio" in rec for rec in summary.to_records())
