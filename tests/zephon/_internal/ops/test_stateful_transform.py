# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the StatefulTransformOp operator."""

import pytest

from zephon._internal.ops.stateful_transform import (
    StatefulTransformAccumulator,
    StatefulTransformOp,
)
from zephon.ops.base import OpContext, StageInfo
from zephon.types import SampleMeta, SampleRecord


def _noop(*args, **kwargs) -> None:  # pragma: no cover - trivial helper
    return None


def _setup(
    op: StatefulTransformOp,
    ctx_data: dict[str, object] | None = None,
    *,
    collect_stats: bool = False,
) -> StatefulTransformOp:
    ctx = {"record_node_metrics": _noop}
    if ctx_data:
        ctx.update(ctx_data)
    op.setup(
        OpContext(
            ctx,
            StageInfo(
                stage_index=0,
                stage_name="stage0",
                op_index=0,
                collect_stats=collect_stats,
            ),
        )
    )
    return op


def _rec(payload: dict, *, sample_id: tuple[int, int, int] = (0, 0, 0)) -> SampleRecord:
    meta = SampleMeta(sample_id=sample_id, lane_id=0, chunk_id=0)
    return SampleRecord(meta=meta, payload=payload)


def _payload_dict(record: SampleRecord) -> dict:
    payload = record.payload
    assert isinstance(payload, dict)
    return payload


# =============================================================================
# StatefulTransformAccumulator Tests
# =============================================================================


class TestStatefulTransformAccumulator:
    """Tests for the StatefulTransformAccumulator class."""

    def test_basic_passthrough(self) -> None:
        """Test accumulator that just passes through items."""

        def push(
            state: int, items: list[SampleRecord]
        ) -> tuple[int, list[SampleRecord]]:
            return state + len(items), items

        acc = StatefulTransformAccumulator(
            init_state=lambda: 0,
            push_fn=push,
            flush_fn=None,
            should_flush_fn=None,
        )

        records = [_rec({"value": i}) for i in range(3)]
        batches = acc.push_many(records)

        assert len(batches) == 1
        items, count = batches[0]
        assert len(items) == 3
        assert count == 3

    def test_buffering_accumulator(self) -> None:
        """Test accumulator that buffers until threshold."""

        def push(
            state: dict, items: list[SampleRecord]
        ) -> tuple[dict, list[SampleRecord]]:
            buffer = state["buffer"] + items
            outputs = []
            while len(buffer) >= 3:
                outputs.extend(buffer[:3])
                buffer = buffer[3:]
            return {"buffer": buffer}, outputs

        acc = StatefulTransformAccumulator(
            init_state=lambda: {"buffer": []},
            push_fn=push,
            flush_fn=lambda s: s["buffer"],
            should_flush_fn=None,
        )

        # Push 2 items - should buffer, no output
        records1 = [_rec({"value": i}) for i in range(2)]
        batches1 = acc.push_many(records1)
        assert batches1 == []

        # Push 2 more items - should emit 3, buffer 1
        records2 = [_rec({"value": i}) for i in range(2, 4)]
        batches2 = acc.push_many(records2)
        assert len(batches2) == 1
        items, _ = batches2[0]
        assert len(items) == 3

        # Flush remaining
        flush_batches = acc.flush()
        assert len(flush_batches) == 1
        flush_items, _ = flush_batches[0]
        assert len(flush_items) == 1

    def test_filtering_accumulator(self) -> None:
        """Test accumulator that filters items based on state."""

        def push(
            seen: set, items: list[SampleRecord]
        ) -> tuple[set, list[SampleRecord]]:
            outputs = []
            for item in items:
                item_id = item.payload["id"]
                if item_id not in seen:
                    seen.add(item_id)
                    outputs.append(item)
            return seen, outputs

        acc = StatefulTransformAccumulator(
            init_state=lambda: set(),
            push_fn=push,
            flush_fn=None,
            should_flush_fn=None,
        )

        # Push items with some duplicates
        records = [
            _rec({"id": 1, "value": "a"}),
            _rec({"id": 2, "value": "b"}),
            _rec({"id": 1, "value": "c"}),  # duplicate id
            _rec({"id": 3, "value": "d"}),
        ]
        batches = acc.push_many(records)

        assert len(batches) == 1
        items, count = batches[0]
        assert len(items) == 3  # 1 duplicate filtered
        assert count == 3

    def test_should_flush_trigger(self) -> None:
        """Test that should_flush triggers early flush and state reset."""
        flush_calls: list[int] = []

        def push(
            state: dict, items: list[SampleRecord]
        ) -> tuple[dict, list[SampleRecord]]:
            state["count"] += len(items)
            return state, items

        def should_flush(state: dict) -> bool:
            return state["count"] >= 5

        def flush(state: dict) -> list[SampleRecord]:
            flush_calls.append(state["count"])
            return []

        acc = StatefulTransformAccumulator(
            init_state=lambda: {"count": 0},
            push_fn=push,
            flush_fn=flush,
            should_flush_fn=should_flush,
        )

        # Push 3 items - below threshold
        records1 = [_rec({"value": i}) for i in range(3)]
        acc.push_many(records1)
        assert flush_calls == []

        # Push 3 more - exceeds threshold (count=6), triggers flush
        records2 = [_rec({"value": i}) for i in range(3)]
        acc.push_many(records2)
        assert flush_calls == [6]

        # Verify state was reset - push 2 more, should not trigger flush
        records3 = [_rec({"value": i}) for i in range(2)]
        acc.push_many(records3)
        assert flush_calls == [6]  # No new flush

        # Push 4 more - exceeds threshold again (count=6), triggers second flush
        records4 = [_rec({"value": i}) for i in range(4)]
        acc.push_many(records4)
        assert flush_calls == [6, 6]  # Second flush with count=6

    def test_has_pending_data(self) -> None:
        """Test has_pending_data returns correct state through full lifecycle."""
        # Without flush_fn, never has pending data
        acc_no_flush = StatefulTransformAccumulator(
            init_state=lambda: {},
            push_fn=lambda s, items: (s, items),
            flush_fn=None,
            should_flush_fn=None,
        )
        assert acc_no_flush.has_pending_data() is False
        acc_no_flush.push_many([_rec({"x": 1})])
        assert acc_no_flush.has_pending_data() is False  # Still false, no flush_fn

        # With flush_fn, tracks pending data correctly
        acc_with_flush = StatefulTransformAccumulator(
            init_state=lambda: {"buffer": []},
            push_fn=lambda s, items: (s, items),
            flush_fn=lambda s: s["buffer"],
            should_flush_fn=None,
        )
        # Before initialization
        assert acc_with_flush.has_pending_data() is False

        # After push - has pending data
        acc_with_flush.push_many([_rec({"x": 1})])
        assert acc_with_flush.has_pending_data() is True

        # After flush - no longer has pending data
        acc_with_flush.flush()
        assert acc_with_flush.has_pending_data() is False


# =============================================================================
# StatefulTransformOp Tests
# =============================================================================


class TestStatefulTransformOp:
    """Tests for the StatefulTransformOp operator."""

    def test_basic_op_creation(self) -> None:
        """Test basic operator creation and traits."""
        op = StatefulTransformOp(
            init_state=lambda: {},
            push_fn=lambda s, items: (s, items),
        )
        _setup(op)

        traits = op.traits()
        assert traits.indexable is False  # Default: stateful ops break indexability
        assert traits.preserves_cursor_order is True
        assert traits.batch_shape_sensitive is True
        # requires_serial_state=False because accumulator already runs serially
        assert traits.requires_serial_state is False
        assert traits.parallelism == 1

    def test_indexable_configurable(self) -> None:
        """Test that indexable trait can be configured."""
        op = StatefulTransformOp(
            init_state=lambda: {},
            push_fn=lambda s, items: (s, items),
            indexable=True,
        )
        _setup(op)

        traits = op.traits()
        assert traits.indexable is True

    def test_preserves_cursor_order_configurable(self) -> None:
        """Test that preserves_cursor_order can be set to False."""
        op = StatefulTransformOp(
            init_state=lambda: {},
            push_fn=lambda s, items: (s, items),
            preserves_cursor_order=False,
        )
        _setup(op)

        traits = op.traits()
        assert traits.preserves_cursor_order is False

    def test_preserves_cursor_order_defaults_true(self) -> None:
        """Test that preserves_cursor_order defaults to True."""
        op = StatefulTransformOp(
            init_state=lambda: {},
            push_fn=lambda s, items: (s, items),
        )
        traits = op.traits()
        assert traits.preserves_cursor_order is True

    def test_custom_parallelism(self) -> None:
        """Test that parallelism can be customized."""
        op = StatefulTransformOp(
            init_state=lambda: {},
            push_fn=lambda s, items: (s, items),
            parallelism=4,
        )
        _setup(op)

        traits = op.traits()
        assert traits.parallelism == 4

    def test_process_one_passthrough(self) -> None:
        """Test process_one just passes through (work is done in accumulator)."""
        op = StatefulTransformOp(
            init_state=lambda: {},
            push_fn=lambda s, items: (s, items),
        )
        _setup(op)

        record = _rec({"value": 42})
        result = op.process_one(record)
        assert len(result) == 1
        assert result[0] is record

    def test_process_many_passthrough(self) -> None:
        """Test process_many just passes through."""
        op = StatefulTransformOp(
            init_state=lambda: {},
            push_fn=lambda s, items: (s, items),
        )
        _setup(op)

        records = [_rec({"value": i}) for i in range(5)]
        result = op.process_many(records)
        assert len(result) == 5
        assert result == records

    @pytest.mark.parametrize(
        "kwarg",
        ["init_state", "push_fn", "flush_fn", "should_flush_fn", "transform_fn"],
    )
    def test_invalid_callable_raises(self, kwarg: str) -> None:
        """Non-callable arguments must raise TypeError."""
        defaults: dict[str, object] = {
            "init_state": lambda: {},
            "push_fn": lambda s, items: (s, items),
        }
        defaults[kwarg] = "not callable"
        with pytest.raises(TypeError, match=f"{kwarg} must be callable"):
            StatefulTransformOp(**defaults)  # type: ignore[arg-type]

    def test_accumulator_creation(self) -> None:
        """Test that accumulator is created correctly."""
        op = StatefulTransformOp(
            init_state=lambda: {"test": True},
            push_fn=lambda s, items: (s, items),
            flush_fn=lambda s: [],
            should_flush_fn=lambda s: False,
        )
        _setup(op)

        acc = op.accumulator(deterministic=True, ctx={})
        assert isinstance(acc, StatefulTransformAccumulator)


# =============================================================================
# Example Use Case Tests
# =============================================================================


class TestStatefulTransformExamples:
    """Tests demonstrating real-world use cases."""

    def test_example_deduplication(self) -> None:
        """Example: Deduplicate items by ID."""

        def push(
            seen: set, items: list[SampleRecord]
        ) -> tuple[set, list[SampleRecord]]:
            outputs = []
            for item in items:
                item_id = item.payload["id"]
                if item_id not in seen:
                    seen.add(item_id)
                    outputs.append(item)
            return seen, outputs

        op = StatefulTransformOp(
            init_state=lambda: set(),
            push_fn=push,
        )
        _setup(op)

        acc = op.accumulator(deterministic=True, ctx={})
        records = [
            _rec({"id": "a", "value": 1}),
            _rec({"id": "b", "value": 2}),
            _rec({"id": "a", "value": 3}),  # duplicate
            _rec({"id": "c", "value": 4}),
            _rec({"id": "b", "value": 5}),  # duplicate
        ]

        batches = acc.push_many(records)
        assert len(batches) == 1
        items, _ = batches[0]
        assert len(items) == 3
        ids = [_payload_dict(item)["id"] for item in items]
        assert ids == ["a", "b", "c"]

    def test_example_running_statistics(self) -> None:
        """Example: Add running mean to each item."""

        def push(
            state: dict, items: list[SampleRecord]
        ) -> tuple[dict, list[SampleRecord]]:
            outputs = []
            total = state["total"]
            count = state["count"]

            for item in items:
                value = item.payload["value"]
                total += value
                count += 1
                running_mean = total / count

                # Create new record with added running_mean
                new_payload = dict(item.payload)
                new_payload["running_mean"] = running_mean
                item.payload = new_payload
                outputs.append(item)

            return {"total": total, "count": count}, outputs

        op = StatefulTransformOp(
            init_state=lambda: {"total": 0.0, "count": 0},
            push_fn=push,
        )
        _setup(op)

        acc = op.accumulator(deterministic=True, ctx={})
        records = [
            _rec({"value": 10}),
            _rec({"value": 20}),
            _rec({"value": 30}),
        ]

        batches = acc.push_many(records)
        items, _ = batches[0]

        assert _payload_dict(items[0])["running_mean"] == 10.0
        assert _payload_dict(items[1])["running_mean"] == 15.0
        assert _payload_dict(items[2])["running_mean"] == 20.0

    def test_example_batch_by_size(self) -> None:
        """Example: Emit batches when total size exceeds threshold."""

        def push(
            state: dict, items: list[SampleRecord]
        ) -> tuple[dict, list[SampleRecord]]:
            buffer = state["buffer"]
            total_size = state["total_size"]
            max_size = 100
            outputs = []

            for item in items:
                item_size = item.payload["size"]
                if total_size + item_size > max_size and buffer:
                    # Emit current buffer
                    outputs.extend(buffer)
                    buffer = []
                    total_size = 0
                buffer.append(item)
                total_size += item_size

            return {"buffer": buffer, "total_size": total_size}, outputs

        def flush(state: dict) -> list[SampleRecord]:
            return state["buffer"]

        op = StatefulTransformOp(
            init_state=lambda: {"buffer": [], "total_size": 0},
            push_fn=push,
            flush_fn=flush,
        )
        _setup(op)

        acc = op.accumulator(deterministic=True, ctx={})

        # Items with sizes that will trigger batch emission
        records = [
            _rec({"size": 40, "id": 1}),  # total: 40
            _rec({"size": 50, "id": 2}),  # total: 90
            _rec({"size": 30, "id": 3}),  # exceeds 100, emit first 2
            _rec({"size": 60, "id": 4}),  # total: 90
        ]

        batches = acc.push_many(records)
        # Should have emitted items 1 and 2
        assert len(batches) == 1
        emitted, _ = batches[0]
        assert len(emitted) == 2
        assert [_payload_dict(e)["id"] for e in emitted] == [1, 2]

        # Flush remaining
        flush_batches = acc.flush()
        assert len(flush_batches) == 1
        remaining, _ = flush_batches[0]
        assert len(remaining) == 2
        assert [_payload_dict(r)["id"] for r in remaining] == [3, 4]

    def test_example_windowed_transform(self) -> None:
        """Example: Apply transform with sliding window context."""

        def push(
            state: dict, items: list[SampleRecord]
        ) -> tuple[dict, list[SampleRecord]]:
            window = state["window"]  # List of recent values
            window_size = 3
            outputs = []

            for item in items:
                value = item.payload["value"]
                window.append(value)
                if len(window) > window_size:
                    window.pop(0)

                # Add window context to payload
                new_payload = dict(item.payload)
                new_payload["window"] = list(window)
                new_payload["window_sum"] = sum(window)
                item.payload = new_payload
                outputs.append(item)

            return {"window": window}, outputs

        op = StatefulTransformOp(
            init_state=lambda: {"window": []},
            push_fn=push,
        )
        _setup(op)

        acc = op.accumulator(deterministic=True, ctx={})
        records = [
            _rec({"value": 1}),
            _rec({"value": 2}),
            _rec({"value": 3}),
            _rec({"value": 4}),
            _rec({"value": 5}),
        ]

        batches = acc.push_many(records)
        items, _ = batches[0]

        # Check windowed sums
        assert _payload_dict(items[0])["window_sum"] == 1  # [1]
        assert _payload_dict(items[1])["window_sum"] == 3  # [1, 2]
        assert _payload_dict(items[2])["window_sum"] == 6  # [1, 2, 3]
        assert _payload_dict(items[3])["window_sum"] == 9  # [2, 3, 4]
        assert _payload_dict(items[4])["window_sum"] == 12  # [3, 4, 5]

    def test_preserves_metadata(self) -> None:
        """Test that sample metadata is preserved through stateful transform."""

        def push(
            state: int, items: list[SampleRecord]
        ) -> tuple[int, list[SampleRecord]]:
            for item in items:
                item.payload = {"transformed": True, **item.payload}
            return state + 1, items

        op = StatefulTransformOp(
            init_state=lambda: 0,
            push_fn=push,
        )
        _setup(op)

        acc = op.accumulator(deterministic=True, ctx={})

        original_meta = SampleMeta(
            sample_id=(1, 2, 3),
            lane_id=5,
            chunk_id=10,
            chunk_offset=20,
            lineage=(0, 1, 2),
            tags={"tag": "value"},
        )
        record = SampleRecord(meta=original_meta, payload={"data": "test"})

        batches = acc.push_many([record])
        items, _ = batches[0]
        result = items[0]

        # All metadata preserved
        assert result.meta.sample_id == (1, 2, 3)
        assert result.meta.lane_id == 5
        assert result.meta.chunk_id == 10
        assert result.meta.chunk_offset == 20
        assert result.meta.lineage == (0, 1, 2)
        assert result.meta.tags == {"tag": "value"}

        # Payload transformed
        assert result.payload["transformed"] is True
        assert result.payload["data"] == "test"


# =============================================================================
# Parallel Transform Tests (Serial State + Parallel Batch Processing)
# =============================================================================


class TestParallelTransform:
    """Tests for the transform_fn feature that enables parallel batch processing."""

    def test_transform_fn_applied_in_process_one(self) -> None:
        """Test that transform_fn is applied in process_one (wraps single item)."""

        def transform(batch: list[SampleRecord]) -> list[SampleRecord]:
            for record in batch:
                record.payload = {"value": record.payload["value"] * 2}
            return batch

        op = StatefulTransformOp(
            init_state=lambda: {},
            push_fn=lambda s, items: (s, items),
            transform_fn=transform,
        )
        _setup(op)

        record = _rec({"value": 21})
        result = op.process_one(record)
        assert len(result) == 1
        assert _payload_dict(result[0])["value"] == 42

    def test_transform_fn_applied_in_process_many(self) -> None:
        """Test that transform_fn is applied to whole batch in process_many."""

        def transform(batch: list[SampleRecord]) -> list[SampleRecord]:
            for record in batch:
                record.payload = {"value": record.payload["value"].upper()}
            return batch

        op = StatefulTransformOp(
            init_state=lambda: {},
            push_fn=lambda s, items: (s, items),
            transform_fn=transform,
        )
        _setup(op)

        records = [
            _rec({"value": "hello"}),
            _rec({"value": "world"}),
        ]
        results = op.process_many(records)
        assert len(results) == 2
        assert _payload_dict(results[0])["value"] == "HELLO"
        assert _payload_dict(results[1])["value"] == "WORLD"

    def test_transform_fn_batch_filtering(self) -> None:
        """Test that transform_fn can filter items from batch."""

        def transform(batch: list[SampleRecord]) -> list[SampleRecord]:
            results = []
            for record in batch:
                if record.payload["value"] >= 0:
                    record.payload = {"value": record.payload["value"] * 2}
                    results.append(record)
            return results

        op = StatefulTransformOp(
            init_state=lambda: {},
            push_fn=lambda s, items: (s, items),
            transform_fn=transform,
        )
        _setup(op)

        records = [
            _rec({"value": 5}),
            _rec({"value": -3}),  # should be filtered
            _rec({"value": 10}),
        ]
        results = op.process_many(records)
        assert len(results) == 2
        assert _payload_dict(results[0])["value"] == 10
        assert _payload_dict(results[1])["value"] == 20

    def test_no_transform_fn_passthrough(self) -> None:
        """Test that without transform_fn, process_many is passthrough."""
        op = StatefulTransformOp(
            init_state=lambda: {},
            push_fn=lambda s, items: (s, items),
            transform_fn=None,
        )
        _setup(op)

        records = [_rec({"value": i}) for i in range(3)]
        results = op.process_many(records)
        assert results == records  # Same objects

    def test_example_dedupe_then_batch_encode(self) -> None:
        """Example: Deduplicate (serial) then batch encode (parallel)."""

        # Serial state: track seen IDs
        def push(
            seen: set, items: list[SampleRecord]
        ) -> tuple[set, list[SampleRecord]]:
            outputs = []
            for item in items:
                item_id = item.payload["id"]
                if item_id not in seen:
                    seen.add(item_id)
                    outputs.append(item)
            return seen, outputs

        # Parallel transform: batch encoding (could be GPU batched)
        def batch_encode(batch: list[SampleRecord]) -> list[SampleRecord]:
            # In real usage, this could batch texts for GPU encoding
            for record in batch:
                text = record.payload["text"]
                record.payload = {
                    **record.payload,
                    "encoded": [ord(c) for c in text],  # fake encoding
                    "length": len(text),
                }
            return batch

        op = StatefulTransformOp(
            init_state=lambda: set(),
            push_fn=push,
            transform_fn=batch_encode,
            parallelism=4,  # Would run in parallel in real pipeline
        )
        _setup(op)

        # Test the accumulator (serial dedupe)
        acc = op.accumulator(deterministic=True, ctx={})
        records = [
            _rec({"id": "a", "text": "hello"}),
            _rec({"id": "b", "text": "world"}),
            _rec({"id": "a", "text": "duplicate"}),  # filtered by accumulator
        ]
        batches = acc.push_many(records)
        deduped, _ = batches[0]
        assert len(deduped) == 2

        # Test the transform (parallel batch encode)
        encoded = op.process_many(deduped)
        assert len(encoded) == 2
        assert _payload_dict(encoded[0])["encoded"] == [104, 101, 108, 108, 111]
        assert _payload_dict(encoded[1])["length"] == 5

    def test_batch_structure_preserved(self) -> None:
        """Test that batch structure from accumulator is preserved for transform."""
        batch_sizes_seen: list[int] = []

        def transform(batch: list[SampleRecord]) -> list[SampleRecord]:
            batch_sizes_seen.append(len(batch))
            return batch

        op = StatefulTransformOp(
            init_state=lambda: {},
            push_fn=lambda s, items: (s, items),
            transform_fn=transform,
        )
        _setup(op)

        # Process batches of different sizes
        op.process_many([_rec({"x": i}) for i in range(5)])
        op.process_many([_rec({"x": i}) for i in range(3)])
        op.process_many([_rec({"x": i}) for i in range(10)])

        assert batch_sizes_seen == [5, 3, 10]

    def test_parallelism_with_transform(self) -> None:
        """Test that parallelism is correctly set when transform_fn is provided."""
        op = StatefulTransformOp(
            init_state=lambda: {},
            push_fn=lambda s, items: (s, items),
            transform_fn=lambda batch: batch,
            parallelism=8,
        )
        _setup(op)

        traits = op.traits()
        assert traits.parallelism == 8
        # requires_serial_state=False - accumulator handles serial state,
        # transform runs in parallel workers
        assert traits.requires_serial_state is False

    def test_example_gpu_batch_inference(self) -> None:
        """Example: Custom batching (serial) then GPU inference (parallel batch)."""

        def push(
            state: dict, items: list[SampleRecord]
        ) -> tuple[dict, list[SampleRecord]]:
            buffer = state["buffer"] + items
            outputs = []
            while len(buffer) >= 4:
                outputs.extend(buffer[:4])
                buffer = buffer[4:]
            return {"buffer": buffer}, outputs

        # Parallel: "GPU" inference on batch (simulated)
        def gpu_inference(batch: list[SampleRecord]) -> list[SampleRecord]:
            # In real usage: stack tensors, run model, split results
            for record in batch:
                record.payload["prediction"] = sum(record.payload["features"])
            return batch

        op = StatefulTransformOp(
            init_state=lambda: {"buffer": []},
            push_fn=push,
            flush_fn=lambda s: s["buffer"],
            transform_fn=gpu_inference,
            parallelism=4,
        )
        _setup(op)

        acc = op.accumulator(deterministic=True, ctx={})

        # Push 6 items
        records = [_rec({"features": [i, i * 2]}) for i in range(6)]
        batches = acc.push_many(records)

        # Should emit 4 items (one batch)
        assert len(batches) == 1
        emitted, _ = batches[0]
        assert len(emitted) == 4

        # Run through transform
        transformed = op.process_many(emitted)
        assert all("prediction" in _payload_dict(r) for r in transformed)

        # Flush remaining 2
        flush_batches = acc.flush()
        assert len(flush_batches) == 1
        remaining, _ = flush_batches[0]
        assert len(remaining) == 2


# =============================================================================
# Mid-stream flush: flush(reset=True) must reset state
# =============================================================================


class TestStatefulTransformAccumulatorMidStreamFlush:
    """Verify that flush(reset=True) resets state for continued use.

    After a mid-stream flush the accumulator must be indistinguishable from
    a freshly constructed instance.  The current implementation has three
    compounding bugs:
    1. ``_flushed = True`` permanently — ``has_pending_data()`` returns False
    2. ``_state = None`` without resetting ``_initialized``
    3. No re-initialization — next ``push_many()`` calls ``push_fn(None, ...)``
    """

    def test_mid_stream_flush_allows_continued_use(self) -> None:
        """After flush(reset=True), push_many must work (not crash).

        Uses a dedup accumulator: push [a,b], flush, push [c,d] → should
        emit [c,d] (fresh dedup state, not carrying over seen set from epoch 1).
        """

        def push(
            seen: set, items: list[SampleRecord]
        ) -> tuple[set, list[SampleRecord]]:
            outputs = []
            for item in items:
                item_id = item.payload["id"]
                if item_id not in seen:
                    seen.add(item_id)
                    outputs.append(item)
            return seen, outputs

        acc = StatefulTransformAccumulator(
            init_state=lambda: set(),
            push_fn=push,
            flush_fn=lambda s: [],
            should_flush_fn=None,
        )

        # Epoch 1
        epoch1 = [_rec({"id": "a"}), _rec({"id": "b"})]
        acc.push_many(epoch1)
        acc.flush(reset=True)

        # Epoch 2 — must not crash, must produce output
        epoch2 = [_rec({"id": "c"}), _rec({"id": "d"})]
        batches = acc.push_many(epoch2)
        output = [rec for batch, _ in batches for rec in batch]
        assert len(output) == 2, (
            f"Expected 2 records after mid-stream flush, got {len(output)}"
        )

    def test_mid_stream_flush_resets_has_pending_data(self) -> None:
        """After flush(reset=True) + push, has_pending_data must be True."""
        acc = StatefulTransformAccumulator(
            init_state=lambda: {"buffer": []},
            push_fn=lambda s, items: (s, items),
            flush_fn=lambda s: s["buffer"],
            should_flush_fn=None,
        )

        acc.push_many([_rec({"x": 1})])
        assert acc.has_pending_data() is True

        acc.flush(reset=True)
        assert acc.has_pending_data() is False, (
            "has_pending_data() should be False immediately after a mid-stream "
            "flush resets the accumulator"
        )

        # Push new data — has_pending_data should be True again
        acc.push_many([_rec({"x": 2})])
        assert acc.has_pending_data() is True, (
            "has_pending_data() should become True again after pushing data post-flush"
        )

    def test_mid_stream_flush_resets_user_state(self) -> None:
        """After flush(reset=True), user state must be fresh.

        Dedup accumulator: push [a,b], flush, push [a,c] → fresh state means
        'a' is NOT in the seen set → both [a,c] emitted.  With stale state,
        'a' would be filtered → only [c] emitted.
        """

        def push(
            seen: set, items: list[SampleRecord]
        ) -> tuple[set, list[SampleRecord]]:
            outputs = []
            for item in items:
                item_id = item.payload["id"]
                if item_id not in seen:
                    seen.add(item_id)
                    outputs.append(item)
            return seen, outputs

        acc = StatefulTransformAccumulator(
            init_state=lambda: set(),
            push_fn=push,
            flush_fn=lambda s: [],
            should_flush_fn=None,
        )

        # Epoch 1: see 'a' and 'b'
        epoch1 = [_rec({"id": "a"}), _rec({"id": "b"})]
        acc.push_many(epoch1)
        acc.flush(reset=True)

        # Epoch 2: push 'a' again — with fresh state, 'a' should pass through
        epoch2 = [_rec({"id": "a"}), _rec({"id": "c"})]
        batches = acc.push_many(epoch2)
        output = [rec for batch, _ in batches for rec in batch]
        ids = [r.payload["id"] for r in output]
        assert ids == ["a", "c"], (
            f"Expected ['a', 'c'] (fresh dedup state), got {ids}. "
            f"Stale seen set from epoch 1 leaked across flush boundary."
        )


def _lane_rec(lane: int, item_id: str, *, offset: int = 0) -> SampleRecord:
    """A record on a specific lane (the default ``_rec`` is lane 0 only)."""
    meta = SampleMeta(
        sample_id=(0, lane, offset), lane_id=lane, chunk_id=0, chunk_offset=offset
    )
    return SampleRecord(meta=meta, payload={"id": item_id})


def _dedup_push(seen: set, items: list[SampleRecord]) -> tuple[set, list[SampleRecord]]:
    """Per-lane dedup by ``payload['id']`` — state is the seen-id set."""
    outputs = []
    for item in items:
        if item.payload["id"] not in seen:
            seen.add(item.payload["id"])
            outputs.append(item)
    return seen, outputs


class TestStatefulTransformAccumulatorPerLane:
    """State is partitioned by lane: a flush sentinel for one lane must not
    disturb another lane's epoch, and push_fn must see one lane at a time."""

    def test_per_lane_flush_resets_only_target_lane(self) -> None:
        """flush(reset=True, lane_id=L) resets lane L's state and no other's.

        Feed 'a' to lane 0 and 'x' to lane 1, then flush only lane 0.
        Re-feeding both: lane 0 forgot 'a' (emitted again), lane 1 still
        remembers 'x' (filtered) — so the flush touched only lane 0.  Flushing
        every lane would re-emit 'x' too.
        """
        acc = StatefulTransformAccumulator(
            init_state=set,
            push_fn=_dedup_push,
            flush_fn=lambda s: [],
            should_flush_fn=None,
        )

        acc.push_many([_lane_rec(0, "a"), _lane_rec(1, "x")])
        acc.flush(reset=True, lane_id=0)

        out = [
            r
            for b, _ in acc.push_many([_lane_rec(0, "a"), _lane_rec(1, "x")])
            for r in b
        ]
        emitted = {(r.meta.lane_id, r.payload["id"]) for r in out}
        assert emitted == {(0, "a")}, (
            f"only lane 0 should forget its epoch; got {emitted}"
        )

    def test_has_pending_data_is_per_lane(self) -> None:
        """has_pending_data(lane) tracks each lane independently."""
        acc = StatefulTransformAccumulator(
            init_state=lambda: {"buf": []},
            push_fn=lambda s, items: ({"buf": s["buf"] + items}, []),
            flush_fn=lambda s: s["buf"],
            should_flush_fn=None,
        )

        acc.push_many([_lane_rec(0, "a"), _lane_rec(1, "x")])
        assert acc.has_pending_data(0) is True
        assert acc.has_pending_data(1) is True
        assert acc.has_pending_data() is True

        acc.flush(reset=True, lane_id=0)
        assert acc.has_pending_data(0) is False
        assert acc.has_pending_data(1) is True, "lane 1's buffer must survive"
        assert acc.has_pending_data() is True

    def test_push_fn_receives_one_lane_at_a_time(self) -> None:
        """push_fn is called per lane, never with a mixed-lane batch."""
        seen_lane_sets: list[set[int]] = []

        def push(
            state: None, items: list[SampleRecord]
        ) -> tuple[None, list[SampleRecord]]:
            seen_lane_sets.append({i.meta.lane_id for i in items})
            return state, items

        acc = StatefulTransformAccumulator(
            init_state=lambda: None, push_fn=push, flush_fn=None, should_flush_fn=None
        )
        batches = acc.push_many(
            [_lane_rec(0, "a"), _lane_rec(1, "x"), _lane_rec(0, "b"), _lane_rec(1, "y")]
        )

        assert all(len(lanes) == 1 for lanes in seen_lane_sets), (
            f"push_fn must see lane-pure batches, got {seen_lane_sets}"
        )
        for batch, _ in batches:
            assert len({r.meta.lane_id for r in batch}) == 1, "output must be lane-pure"
