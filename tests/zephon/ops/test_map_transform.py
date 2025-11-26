# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import pytest

from zephon.core.constants import SampleMeta, SampleRecord
from zephon.core.op_base import OpContext
from zephon.ops.map_transform import MapTransform


def _noop(*args, **kwargs) -> None:  # pragma: no cover - trivial helper
    return None


def _setup(
    op: MapTransform,
    ctx_data: dict[str, object] | None = None,
    *,
    collect_stats: bool = False,
) -> MapTransform:
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


def _payload_dict(record: SampleRecord) -> dict:
    payload = record.payload
    assert isinstance(payload, dict)
    return payload


def test_map_transform_full_payload() -> None:
    """Test transforming entire payload."""

    def transform(payload: dict) -> dict:
        return {"text": payload.get("text", "").upper(), "processed": True}

    op = _setup(MapTransform(transform))
    r = _rec({"text": "hello", "other": "data"})
    out = op.process_one(r)[0]
    payload = _payload_dict(out)
    assert payload["text"] == "HELLO"
    assert payload["processed"] is True
    assert "other" not in payload  # Full payload replacement


def test_map_transform_filtering_drop_none() -> None:
    """Test filtering samples by returning None."""

    def transform(payload: dict) -> dict | None:
        if payload.get("text") == "":
            return None  # Drop empty
        return {"text": payload["text"], "kept": True}

    op = _setup(MapTransform(transform, drop_none=True))

    # Should be kept
    r1 = _rec({"text": "hello"})
    out1 = op.process_one(r1)
    assert len(out1) == 1
    assert _payload_dict(out1[0])["kept"] is True

    # Should be dropped
    r2 = _rec({"text": ""})
    out2 = op.process_one(r2)
    assert len(out2) == 0


def test_map_transform_filtering_keep_none() -> None:
    """Test keeping samples when transform returns None and drop_none=False."""

    def transform(payload: dict) -> dict | None:
        return None

    op = _setup(MapTransform(transform, drop_none=False))
    r = _rec({"text": "hello"})
    out = op.process_one(r)
    assert len(out) == 1
    # Original payload preserved when None returned and drop_none=False
    assert _payload_dict(out[0])["text"] == "hello"


def test_map_transform_non_dict_result() -> None:
    """Test that non-dict results are used directly (no wrapping)."""

    def transform(payload: dict) -> str:
        return payload.get("text", "").upper()

    op = _setup(MapTransform(transform))
    r = _rec({"text": "hello"})
    out = op.process_one(r)[0]
    # Non-dict results are used directly (must match SamplePayload typing)
    assert isinstance(out.payload, str)
    assert out.payload == "HELLO"


def test_map_transform_process_many() -> None:
    """Test batch processing."""

    def transform(payload: dict) -> dict:
        return {"text": payload.get("text", "").upper()}

    op = _setup(MapTransform(transform))
    records = [_rec({"text": "hello"}, sample_id=(0, 0, i)) for i in range(3)]
    results = op.process_many(records)
    assert len(results) == 3
    for i, result in enumerate(results):
        assert _payload_dict(result)["text"] == "HELLO"
        # Metadata preserved
        assert result.meta.sample_id == (0, 0, i)


def test_map_transform_preserves_metadata() -> None:
    """Test that metadata and lineage are preserved."""

    def transform(payload: dict) -> dict:
        return {"transformed": True}

    op = _setup(MapTransform(transform))
    original_meta = SampleMeta(
        sample_id=(1, 2, 3),
        lane_id=5,
        chunk_id=10,
        chunk_offset=20,
        lineage=(0, 1, 2),
        tags={"tag": "value"},
    )
    r = SampleRecord(meta=original_meta, payload={"text": "hello"})
    out = op.process_one(r)[0]

    # All metadata preserved
    assert out.meta.sample_id == (1, 2, 3)
    assert out.meta.lane_id == 5
    assert out.meta.chunk_id == 10
    assert out.meta.chunk_offset == 20
    assert out.meta.lineage == (0, 1, 2)
    assert out.meta.tags == {"tag": "value"}


def test_map_transform_traits_and_buffering() -> None:
    """Test operator traits and buffering configuration."""

    def transform(payload: dict) -> dict:
        return payload

    op = _setup(MapTransform(transform))
    traits = op.traits()
    buffering = op.buffering()

    assert traits.indexable is True
    assert traits.parallelism == 4
    assert buffering is not None
    assert buffering.max_batch == 64
    assert buffering.max_latency_ms == 3


def test_map_transform_custom_buffering() -> None:
    """Test custom buffering configuration."""
    from zephon.core.traits import Buffering

    def transform(payload: dict) -> dict:
        return payload

    custom_buffering = Buffering(max_batch=128, max_latency_ms=5)
    op = _setup(MapTransform(transform, buffering=custom_buffering))
    assert op.buffering() == custom_buffering


def test_map_transform_invalid_callable() -> None:
    """Test that non-callable transform_fn raises TypeError."""
    with pytest.raises(TypeError, match="must be callable"):
        MapTransform("not a function")


def test_map_transform_non_dict_payload() -> None:
    """Test that non-dict payloads work fine."""

    def transform(payload: str) -> str:
        return payload.upper()

    op = _setup(MapTransform(transform))
    # Create record with non-dict payload - should work fine
    meta = SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0)
    r = SampleRecord(meta=meta, payload="hello")
    out = op.process_one(r)[0]
    assert out.payload == "HELLO"


def test_map_transform_filtering_with_process_many() -> None:
    """Test filtering works correctly in batch processing."""

    def transform(payload: dict) -> dict | None:
        if payload.get("text", "").startswith("drop"):
            return None
        return {"text": payload["text"], "kept": True}

    op = _setup(MapTransform(transform, drop_none=True))
    records = [
        _rec({"text": "keep1"}),
        _rec({"text": "drop_me"}),
        _rec({"text": "keep2"}),
    ]
    results = op.process_many(records)
    # Should have 2 results (one dropped)
    assert len(results) == 2
    assert all(_payload_dict(r)["kept"] is True for r in results)


def test_map_transform_empty_payload() -> None:
    """Test transforming an empty payload."""

    def transform(payload: dict) -> dict:
        return {"empty": True}

    op = _setup(MapTransform(transform))
    r = _rec({})
    out = op.process_one(r)[0]
    payload = _payload_dict(out)
    assert payload["empty"] is True
