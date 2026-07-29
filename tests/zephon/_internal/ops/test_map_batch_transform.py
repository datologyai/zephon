# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import pytest

from zephon._internal.ops.map_transform import MapBatchTransform
from zephon.ops.base import OpContext
from zephon.types import SampleBatch, SampleMeta, SampleRecord


def _noop(*args, **kwargs) -> None:  # pragma: no cover - trivial helper
    return None


def _setup(
    op: MapBatchTransform,
    ctx_data: dict[str, object] | None = None,
    *,
    collect_stats: bool = False,
) -> MapBatchTransform:
    ctx = {"record_node_metrics": _noop}
    if ctx_data:
        ctx.update(ctx_data)
    op.setup(
        OpContext(ctx),
        stage_index=0,
        stage_name="stage0",
        op_index=0,
        collect_stats=collect_stats,
    )
    return op


def _rec(payload: dict, *, sample_id: tuple[int, int, int] = (0, 0, 0)) -> SampleRecord:
    meta = SampleMeta(sample_id=sample_id, lane_id=0, chunk_id=0)
    return SampleRecord(meta=meta, payload=payload)


def test_map_batch_transform_basic() -> None:
    """Test transforming a SampleBatch passes the whole batch to transform_fn."""

    def transform_batch(batch: SampleBatch) -> SampleBatch:
        new_records = []
        for record in batch.records:
            payload = record.payload
            assert isinstance(payload, dict)
            new_payload = {
                "text": payload.get("text", "").upper(),
                "batch_processed": True,
            }
            new_records.append(SampleRecord(meta=record.meta, payload=new_payload))
        return SampleBatch(records=tuple(new_records))

    op = _setup(MapBatchTransform(transform_batch))
    records = tuple(_rec({"text": f"hello{i}"}, sample_id=(0, 0, i)) for i in range(3))
    batch = SampleBatch(records=records)

    out = op.process_one(batch)
    assert len(out) == 1
    result_batch = out[0]
    assert isinstance(result_batch, SampleBatch)
    assert len(result_batch.records) == 3
    for i, record in enumerate(result_batch.records):
        payload = record.payload
        assert isinstance(payload, dict)
        assert payload["text"] == f"HELLO{i}"
        assert payload["batch_processed"] is True
        assert record.meta.sample_id == (0, 0, i)


def test_map_batch_transform_filtering() -> None:
    """Test that returning None from a batch transform emits tombstones."""

    def transform_batch(batch: SampleBatch) -> SampleBatch | None:
        for record in batch.records:
            payload = record.payload
            if isinstance(payload, dict) and "drop" in payload.get("text", ""):
                return None
        return batch

    op = _setup(MapBatchTransform(transform_batch, drop_none=True))

    # Batch that should be kept
    records1 = tuple(_rec({"text": f"keep{i}"}, sample_id=(0, 0, i)) for i in range(2))
    batch1 = SampleBatch(records=records1)
    out1 = op.process_one(batch1)
    assert len(out1) == 1

    # Batch that should be dropped — emits tombstones for each record
    records2 = (
        _rec({"text": "drop_me"}, sample_id=(0, 0, 0)),
        _rec({"text": "also_gone"}, sample_id=(0, 0, 1)),
    )
    batch2 = SampleBatch(records=records2)
    out2 = op.process_one(batch2)
    assert len(out2) == 2
    assert all(r.meta.tombstone is True for r in out2)
    assert all(r.payload is None for r in out2)


def test_map_batch_transform_keep_none() -> None:
    """Test keeping batch when transform returns None and drop_none=False."""
    op = _setup(MapBatchTransform(lambda b: None, drop_none=False))
    records = tuple(_rec({"text": f"hello{i}"}, sample_id=(0, 0, i)) for i in range(2))
    batch = SampleBatch(records=records)
    out = op.process_one(batch)
    assert len(out) == 1
    assert isinstance(out[0], SampleBatch)


def test_map_batch_transform_process_many() -> None:
    """Test process_many works with SampleBatch inputs."""

    def transform_batch(batch: SampleBatch) -> SampleBatch:
        new_records = []
        for record in batch.records:
            payload = record.payload
            assert isinstance(payload, dict)
            new_payload = {"text": payload.get("text", "").upper()}
            new_records.append(SampleRecord(meta=record.meta, payload=new_payload))
        return SampleBatch(records=tuple(new_records))

    op = _setup(MapBatchTransform(transform_batch))
    batches = [
        SampleBatch(
            records=tuple(
                _rec({"text": f"a{i}"}, sample_id=(0, 0, i)) for i in range(2)
            )
        ),
        SampleBatch(
            records=tuple(
                _rec({"text": f"b{i}"}, sample_id=(0, 1, i)) for i in range(2)
            )
        ),
    ]
    results = op.process_many(batches)
    assert len(results) == 2
    for result in results:
        assert isinstance(result, SampleBatch)


def test_map_batch_transform_process_many_passes_through_records() -> None:
    """process_many passes SampleRecord elements through unchanged (tombstone handling)."""
    op = _setup(MapBatchTransform(lambda b: b))
    record = _rec({"text": "hello"})
    batch = SampleBatch(
        records=tuple(_rec({"text": f"t{i}"}, sample_id=(0, 0, i)) for i in range(2))
    )
    results = op.process_many([record, batch])
    # record passed through, batch processed
    assert len(results) == 2
    assert isinstance(results[0], SampleRecord)
    assert isinstance(results[1], SampleBatch)


def test_map_batch_transform_passes_through_tombstone_record() -> None:
    """Tombstone SampleRecords should pass through MapBatchTransform unchanged."""
    op = _setup(MapBatchTransform(lambda b: b))
    meta = SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0).with_tombstone()
    tombstone = SampleRecord(meta=meta, payload=None)
    out = op.process_one(tombstone)
    assert len(out) == 1
    assert out[0].meta.tombstone is True


def test_map_batch_transform_traits_and_accumulator() -> None:
    """Test operator traits and accumulator configuration."""
    op = _setup(MapBatchTransform(lambda b: b, max_batch=32, max_latency_ms=100))
    traits = op.traits()

    assert traits.indexable is True
    assert traits.parallelism == 4

    acc_det = op.accumulator(deterministic=True, ctx={})
    assert acc_det._max_batch == 32
    assert acc_det._max_latency_ms is None

    acc_nondet = op.accumulator(deterministic=False, ctx={})
    assert acc_nondet._max_batch == 32
    assert acc_nondet._max_latency_ms == 100


def test_map_batch_transform_invalid_callable() -> None:
    """Test that non-callable transform_fn raises TypeError."""
    with pytest.raises(TypeError, match="must be callable"):
        MapBatchTransform("not a function")
