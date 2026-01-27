# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from tests.zephon.runners._helpers import _ctx_services
from zephon.core.constants import SampleBatch, SampleMeta, SampleRecord
from zephon.core.graph import Node, Stage
from zephon.core.op_base import Op
from zephon.ops.batch import Batch, BatchAccumulator
from zephon.ops.delay import DelayById
from zephon.runners.threads import ThreadStageRunner


def _rec(i: int, *, lane: int = 0, chunk: int = 0, **payload: Any) -> SampleRecord:
    meta = SampleMeta(sample_id=(0, 0, i), lane_id=lane, chunk_id=chunk)
    return SampleRecord(meta=meta, payload={"value": i, **payload})


def test_batch_invalid_microbatch_size_raises() -> None:
    with pytest.raises(ValueError):
        Batch(0)
    with pytest.raises(ValueError):
        Batch(-2)


def test_batch_accumulator_invalid_microbatch_size_raises() -> None:
    with pytest.raises(ValueError):
        BatchAccumulator(0)
    with pytest.raises(ValueError):
        BatchAccumulator(-2)


def test_batch_accumulator_single_lane_drop_last_true() -> None:
    acc = BatchAccumulator(2, drop_last=True)
    # First record - not enough for a batch
    ready = acc.push_many([_rec(0)])
    assert ready == []
    # Second record - now we have a full batch
    ready = acc.push_many([_rec(1)])
    assert len(ready) == 1
    assert len(ready[0][0]) == 2  # ready[0] is (batch, wait_ns), [0] is batch
    # Third record - partial batch
    ready = acc.push_many([_rec(2)])
    assert ready == []
    # Flush drops remainder when drop_last=True
    ready = acc.flush()
    assert ready == []


def test_batch_accumulator_single_lane_drop_last_false() -> None:
    acc = BatchAccumulator(3, drop_last=False)
    # Feed 5 -> one batch of 3, remainder 2 emitted on flush
    ready = acc.push_many([_rec(i) for i in range(5)])
    assert len(ready) == 1
    assert len(ready[0][0]) == 3
    # Flush emits remainder
    ready = acc.flush()
    assert len(ready) == 1
    assert len(ready[0][0]) == 2


def test_batch_accumulator_multi_lane_interleaving_lane_purity() -> None:
    acc = BatchAccumulator(2, drop_last=False)
    # Interleave lanes 0 and 1
    items = [_rec(0, lane=0), _rec(1, lane=1), _rec(2, lane=0), _rec(3, lane=1)]
    ready = acc.push_many(items)
    # Expect two batches, each lane-pure
    assert len(ready) == 2
    # Check lane purity - ready[i] is (batch, wait_ns), [0] is batch
    batch0_lanes = {r.meta.lane_id for r in ready[0][0]}
    batch1_lanes = {r.meta.lane_id for r in ready[1][0]}
    assert batch0_lanes == {0}
    assert batch1_lanes == {1}
    # No remainder
    assert acc.flush() == []


def test_batch_accumulator_process_equivalence() -> None:
    """Test that batch size and accumulation works correctly."""
    acc_a = BatchAccumulator(2, drop_last=False)
    acc_b = BatchAccumulator(2, drop_last=False)
    items = [_rec(i) for i in range(4)]

    # Process one-by-one
    seq_batches = []
    for it in items:
        seq_batches.extend(acc_a.push_many([it]))
    seq_batches.extend(acc_a.flush())

    # Process in bulk
    bulk_batches = acc_b.push_many(items)
    bulk_batches.extend(acc_b.flush())

    # Same number of batches
    assert len(seq_batches) == len(bulk_batches)
    # Same batch contents - [0] is batch from (batch, wait_ns)
    for seq_b, bulk_b in zip(seq_batches, bulk_batches):
        seq_ids = tuple(r.meta.sample_id for r in seq_b[0])
        bulk_ids = tuple(r.meta.sample_id for r in bulk_b[0])
        assert seq_ids == bulk_ids


def test_batch_traits_and_accumulator() -> None:
    op = Batch(2, drop_last=True)
    traits = op.traits()
    assert traits.indexable is False
    assert traits.batch_shape_sensitive is False

    # Verify accumulator configuration is passed through correctly
    acc = op.accumulator(deterministic=False, ctx={})
    assert acc.microbatch_size == 2
    assert acc.drop_last is True

    # Verify drop_last=False is also passed through
    op_keep = Batch(4, drop_last=False)
    acc_keep = op_keep.accumulator(deterministic=False, ctx={})
    assert acc_keep.microbatch_size == 4
    assert acc_keep.drop_last is False


def test_batch_accumulator_tombstone_handling() -> None:
    """Test that tombstones flush current batch and are emitted in their own batch."""
    acc = BatchAccumulator(3, drop_last=False)
    tomb_meta = SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0).with_tombstone(
        True
    )
    tomb = SampleRecord(meta=tomb_meta, payload={})
    kept = _rec(1)

    # Push tombstone - should emit in its own batch
    ready = acc.push_many([tomb, kept])
    assert len(ready) == 1  # Just the tombstone batch
    assert len(ready[0][0]) == 1  # [0] is (batch, wait_ns), [0] is batch
    assert ready[0][0][0].meta.tombstone

    # Flush remaining
    ready = acc.flush()
    assert len(ready) == 1  # The kept record
    assert len(ready[0][0]) == 1
    assert ready[0][0][0] is kept


def test_batch_operator_process_many_wraps_batch() -> None:
    """Test that the operator wraps lane-pure batches into SampleBatch."""
    op = Batch(2)
    # Operator receives pre-batched records from accumulator
    records = [_rec(0), _rec(1)]
    result = op.process_many(records)
    assert len(result) == 1
    batch = result[0]
    assert hasattr(batch, "records")
    assert len(batch.records) == 2


def test_batch_operator_passes_through_tombstone() -> None:
    """Test that tombstones are passed through unchanged by the operator."""
    op = Batch(2)
    tomb_meta = SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0).with_tombstone(
        True
    )
    tomb = SampleRecord(meta=tomb_meta, payload={})
    result = op.process_many([tomb])
    assert len(result) == 1
    assert result[0] is tomb


def test_batch_has_pending_data() -> None:
    """Test has_pending_data method."""
    acc = BatchAccumulator(3, drop_last=False)
    assert not acc.has_pending_data()

    acc.push_many([_rec(0)])
    assert acc.has_pending_data()

    # After flush, should have no pending data
    acc.flush()
    assert not acc.has_pending_data()


def test_batch_parallelism_configurable() -> None:
    """Test that parallelism can be configured."""
    op = Batch(2, parallelism=4)
    assert op.traits().parallelism == 4

    # Default should be 1 for backwards compatibility
    op_default = Batch(2)
    assert op_default.traits().parallelism == 1


# ---------------------------------------------------------------------------
# Batch determinism tests (with parallel workers and delays)
# ---------------------------------------------------------------------------


def _mk_record(value: int, lane: int = 0) -> SampleRecord:
    """Create a SampleRecord with the given value as its local ID."""
    meta = SampleMeta(
        sample_id=(0, 0, value),
        lane_id=lane,
        chunk_id=0,
        chunk_offset=value,
    )
    return SampleRecord(meta=meta, payload={"value": value})


def _collect_batches(runner: ThreadStageRunner, records: list[SampleRecord]) -> list:
    """Collect all outputs from the runner."""
    return list(runner.run(iter(records)))


def _fit_to_ops_workers(*ops: Op) -> int:
    """Compute max_workers as sum of operator parallelisms (fit_to_ops mode)."""
    return sum(op.traits().parallelism for op in ops)


def test_batch_deterministic_with_delay_and_parallel_workers() -> None:
    """Test that batching is deterministic even with delays and parallel workers.

    This test uses DelayById to introduce timing variation, runs the Batch
    operator with multiple workers, and verifies the exact same batches
    are produced in deterministic mode.
    """
    # Create a pipeline: DelayById -> Batch
    delay_op = DelayById(max_delay_ms=3.0)
    batch_op = Batch(microbatch_size=5, drop_last=False, parallelism=4)

    delay_node = Node(name="delay", op=delay_op)
    batch_node = Node(name="batch", op=batch_op)

    stage = Stage(
        name="test_stage",
        nodes=[delay_node, batch_node],
        placement="auto",
        break_reason="test",
    )

    # Create 1000 records for robust testing
    records = [_mk_record(i) for i in range(1000)]

    # Run multiple times with deterministic=True
    # Use fit_to_ops: max_workers = sum of operator parallelisms
    max_workers = _fit_to_ops_workers(delay_op, batch_op)
    results: list[list] = []
    for _ in range(3):
        runner = ThreadStageRunner(
            stage,
            ctx_services=_ctx_services(),
            max_workers=max_workers,
            deterministic=True,
            stage_output_mode="stream_items",
        )
        result = _collect_batches(runner, records)
        results.append(result)

    # All runs should produce identical results
    for i, result in enumerate(results[1:], 1):
        assert len(result) == len(results[0]), f"Run {i} produced different batch count"
        for j, (batch_a, batch_b) in enumerate(zip(results[0], result)):
            assert isinstance(batch_a, SampleBatch), (
                f"Expected SampleBatch, got {type(batch_a)}"
            )
            assert isinstance(batch_b, SampleBatch), (
                f"Expected SampleBatch, got {type(batch_b)}"
            )
            ids_a = [r.meta.sample_id for r in batch_a.records]
            ids_b = [r.meta.sample_id for r in batch_b.records]
            assert ids_a == ids_b, f"Run {i} batch {j} differs: {ids_a} != {ids_b}"


def test_batch_output_order_preserved_with_parallelism() -> None:
    """Test that batch output order matches input order with parallel workers."""
    delay_op = DelayById(max_delay_ms=2.0)
    batch_op = Batch(microbatch_size=3, drop_last=False, parallelism=4)

    delay_node = Node(name="delay", op=delay_op)
    batch_node = Node(name="batch", op=batch_op)

    stage = Stage(
        name="test_stage",
        nodes=[delay_node, batch_node],
        placement="auto",
        break_reason="test",
    )

    num_records = 1024
    records = [_mk_record(i) for i in range(num_records)]

    runner = ThreadStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=_fit_to_ops_workers(delay_op, batch_op),
        deterministic=True,
        stage_output_mode="stream_items",
    )

    batches = _collect_batches(runner, records)

    # Extract all values in order
    all_values = []
    for batch in batches:
        assert isinstance(batch, SampleBatch)
        for rec in batch.records:
            all_values.append(rec.payload["value"])

    # Values should be in original order
    assert all_values == list(range(num_records))


def test_batch_lane_purity_preserved_with_parallelism() -> None:
    """Test that batches remain lane-pure even with parallel workers."""
    batch_op = Batch(microbatch_size=3, drop_last=False, parallelism=4)

    batch_node = Node(name="batch", op=batch_op)
    stage = Stage(
        name="test_stage",
        nodes=[batch_node],
        placement="auto",
        break_reason="test",
    )

    # Interleave lanes 0 and 1 with 1000 records
    records = []
    for i in range(1000):
        records.append(_mk_record(i, lane=i % 2))

    runner = ThreadStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=_fit_to_ops_workers(batch_op),
        deterministic=True,
        stage_output_mode="stream_items",
    )

    batches = _collect_batches(runner, records)

    # Each batch should be lane-pure
    for batch in batches:
        assert isinstance(batch, SampleBatch)
        lanes = {r.meta.lane_id for r in batch.records}
        assert len(lanes) == 1, f"Batch has mixed lanes: {lanes}"


def test_batch_produces_consistent_sizes_with_parallelism() -> None:
    """Test that batch sizes are consistent across multiple runs with parallelism."""
    delay_op = DelayById(max_delay_ms=1.5)
    batch_op = Batch(microbatch_size=7, drop_last=True, parallelism=4)

    delay_node = Node(name="delay", op=delay_op)
    batch_node = Node(name="batch", op=batch_op)

    stage = Stage(
        name="test_stage",
        nodes=[delay_node, batch_node],
        placement="auto",
        break_reason="test",
    )

    # 1024 records with batch size 7, drop_last=True -> 146 full batches (1022 records)
    num_records = 1024
    records = [_mk_record(i) for i in range(num_records)]

    # Run multiple times with fit_to_ops worker allocation
    max_workers = _fit_to_ops_workers(delay_op, batch_op)
    all_sizes: list[list[int]] = []
    for _ in range(3):
        runner = ThreadStageRunner(
            stage,
            ctx_services=_ctx_services(),
            max_workers=max_workers,
            deterministic=True,
            stage_output_mode="stream_items",
        )
        batches = _collect_batches(runner, records)
        sizes = [len(b.records) for b in batches]
        all_sizes.append(sizes)

    # All runs should produce the same batch sizes
    for i, sizes in enumerate(all_sizes[1:], 1):
        assert sizes == all_sizes[0], f"Run {i} has different batch sizes"

    # With drop_last=True and batch_size=7, 1024 records -> 146 batches of 7 (1022 used)
    expected_batches = num_records // 7
    assert all_sizes[0] == [7] * expected_batches
