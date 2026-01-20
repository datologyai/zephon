# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from zephon.core.constants import SampleMeta, SampleRecord
from zephon.ops.pack_sequences import PackingAccumulator, PackSequences


def _rec(
    i: int, length: int, *, lane: int = 0, chunk: int = 0, **payload: Any
) -> SampleRecord:
    """Create a sample record with a length field."""
    meta = SampleMeta(sample_id=(0, 0, i), lane_id=lane, chunk_id=chunk)
    return SampleRecord(meta=meta, payload={"value": i, "length": length, **payload})


def _simple_length_fn(rec: SampleRecord) -> int:
    """Simple length function for tests."""
    payload = rec.payload
    if isinstance(payload, dict):
        return payload.get("length", 0)
    return 0


def _identity_pack_fn(payloads: list[Any]) -> list[Any]:
    """Identity pack function that keeps list as-is."""
    return payloads


def test_pack_sequences_invalid_max_length_raises() -> None:
    """Test that invalid max_length raises ValueError."""
    with pytest.raises(ValueError):
        PackSequences(0, num_bins=10)
    with pytest.raises(ValueError):
        PackSequences(-5, num_bins=10)


def test_pack_sequences_traits() -> None:
    """Test operator traits."""
    op = PackSequences(
        max_length=10, length_fn="length", algorithm="first_fit", num_bins=10
    )
    traits = op.traits()
    assert traits.indexable is False
    assert traits.batch_shape_sensitive is False
    # With accumulator, requires_serial_state is now False
    assert traits.requires_serial_state is False
    assert traits.preserves_cursor_order is False


def test_pack_sequences_accumulator() -> None:
    """Test that accumulator configuration is passed through correctly."""
    op = PackSequences(
        max_length=10, length_fn="length", algorithm="first_fit", num_bins=5
    )
    acc = op.accumulator(deterministic=False)

    # Verify configuration is passed through correctly
    assert acc.max_length == 10
    assert acc.num_bins == 5
    assert acc.algorithm == "first_fit"


def test_packing_accumulator_first_fit_basic() -> None:
    """Test basic first-fit packing via accumulator."""
    acc = PackingAccumulator(
        max_length=10,
        num_bins=10,
        length_fn=_simple_length_fn,
        algorithm="first_fit",
        drop_oversized=False,
        min_sequence_length=1,
        shuffle_strategy=None,
        shuffle_seed=0,
        pack_payloads_fn=_identity_pack_fn,
        flush_strategy="fifo",
    )

    # Sequences: [3, 4, 2, 5]
    # First bin: 3 + 4 + 2 = 9 (not full, kept)
    # When 5 arrives, it doesn't fit in first bin -> create new bin
    ready = acc.push_many([_rec(0, 3), _rec(1, 4), _rec(2, 2), _rec(3, 5)])

    # No bins emitted yet (neither is full)
    assert len(ready) == 0

    # Finalize should emit both bins
    tail = acc.flush()
    assert len(tail) == 2

    # Check that we have both bins - find them by sample count
    sample_counts = [len(rb[0][0].payload["packed_samples"]) for rb in tail]
    assert sorted(sample_counts) == [1, 3]


def test_packing_accumulator_full_bin_emits() -> None:
    """Test that a full bin emits immediately."""
    acc = PackingAccumulator(
        max_length=10,
        num_bins=10,
        length_fn=_simple_length_fn,
        algorithm="first_fit",
        drop_oversized=False,
        min_sequence_length=1,
        shuffle_strategy=None,
        shuffle_seed=0,
        pack_payloads_fn=_identity_pack_fn,
        flush_strategy="fifo",
    )

    # A single sequence that exactly fills a bin
    ready = acc.push_many([_rec(0, 10)])
    assert len(ready) == 1  # Should emit immediately

    packed = ready[0][0][0]
    assert packed.meta.tags["_packing_metadata"]["total_length"] == 10
    assert packed.meta.tags["_packing_metadata"]["packing_efficiency"] == 1.0


def test_packing_accumulator_oversized_drop() -> None:
    """Test that oversized sequences are dropped when drop_oversized=True."""
    acc = PackingAccumulator(
        max_length=10,
        num_bins=10,
        length_fn=_simple_length_fn,
        algorithm="first_fit",
        drop_oversized=True,
        min_sequence_length=1,
        shuffle_strategy=None,
        shuffle_seed=0,
        pack_payloads_fn=_identity_pack_fn,
        flush_strategy="fifo",
    )

    # Mix of normal and oversized sequences
    ready = acc.push_many([_rec(0, 5), _rec(1, 15), _rec(2, 5)])

    # Should emit one bin with [5, 5] = 10 (oversized dropped)
    assert len(ready) == 1
    packed = ready[0][0][0]
    assert len(packed.payload["packed_samples"]) == 2
    assert packed.meta.tags["_packing_metadata"]["total_length"] == 10


def test_packing_accumulator_oversized_raise() -> None:
    """Test that oversized sequences raise when drop_oversized=False."""
    acc = PackingAccumulator(
        max_length=10,
        num_bins=10,
        length_fn=_simple_length_fn,
        algorithm="first_fit",
        drop_oversized=False,
        min_sequence_length=1,
        shuffle_strategy=None,
        shuffle_seed=0,
        pack_payloads_fn=_identity_pack_fn,
        flush_strategy="fifo",
    )

    with pytest.raises(ValueError, match="exceeds max_length"):
        acc.push_many([_rec(0, 15)])


def test_packing_accumulator_lane_isolation() -> None:
    """Test that sequences from different lanes are not mixed."""
    acc = PackingAccumulator(
        max_length=10,
        num_bins=10,
        length_fn=_simple_length_fn,
        algorithm="first_fit",
        drop_oversized=False,
        min_sequence_length=1,
        shuffle_strategy=None,
        shuffle_seed=0,
        pack_payloads_fn=_identity_pack_fn,
        flush_strategy="fifo",
    )

    # Lane 0: [5, 5] -> bin fills immediately
    # Lane 1: [5, 5] -> bin fills immediately
    items = [
        _rec(0, 5, lane=0),
        _rec(1, 5, lane=1),
        _rec(2, 5, lane=0),
        _rec(3, 5, lane=1),
    ]
    ready = acc.push_many(items)

    # Should have 2 full bins (one per lane)
    assert len(ready) == 2

    # Check lane purity - extract lane_id from packed records
    for rb in ready:
        packed = rb[0][0]
        # All samples in packed bin should be from same lane
        samples = packed.payload["packed_samples"]
        assert len(samples) == 2


def test_packing_accumulator_has_pending_data() -> None:
    """Test has_pending_data method."""
    acc = PackingAccumulator(
        max_length=10,
        num_bins=10,
        length_fn=_simple_length_fn,
        algorithm="first_fit",
        drop_oversized=False,
        min_sequence_length=1,
        shuffle_strategy=None,
        shuffle_seed=0,
        pack_payloads_fn=_identity_pack_fn,
        flush_strategy="fifo",
    )

    assert not acc.has_pending_data()

    acc.push_many([_rec(0, 3)])
    assert acc.has_pending_data()

    acc.flush()
    assert not acc.has_pending_data()


def test_packing_accumulator_contributors() -> None:
    """Test that packed records properly set contributors for eviction tracking."""
    acc = PackingAccumulator(
        max_length=10,
        num_bins=10,
        length_fn=_simple_length_fn,
        algorithm="first_fit",
        drop_oversized=False,
        min_sequence_length=1,
        shuffle_strategy=None,
        shuffle_seed=0,
        pack_payloads_fn=_identity_pack_fn,
        flush_strategy="fifo",
    )

    # Pack two sequences that exactly fill a bin
    rec1 = _rec(0, 5)
    rec2 = _rec(1, 5)
    ready = acc.push_many([rec1, rec2])

    assert len(ready) == 1
    packed = ready[0][0][0]

    # Verify contributors are set
    assert packed.meta.contributors is not None
    assert len(packed.meta.contributors) == 2

    # Verify each contributor references the correct base sample
    contributor_cursors = {ref.cursor for ref in packed.meta.contributors}
    assert rec1.meta.cursor in contributor_cursors
    assert rec2.meta.cursor in contributor_cursors

    # All should be marked as last_child (since these are 1:1 inputs)
    assert all(ref.is_last_child for ref in packed.meta.contributors)


def test_packing_accumulator_best_fit() -> None:
    """Test best-fit algorithm."""
    acc = PackingAccumulator(
        max_length=10,
        num_bins=10,
        length_fn=_simple_length_fn,
        algorithm="best_fit",
        drop_oversized=False,
        min_sequence_length=1,
        shuffle_strategy=None,
        shuffle_seed=0,
        pack_payloads_fn=_identity_pack_fn,
        flush_strategy="fifo",
    )

    # Sequences: [3, 4, 2, 5]
    ready = acc.push_many([_rec(0, 3), _rec(1, 4), _rec(2, 2), _rec(3, 5)])

    # No bins emitted yet (neither is full)
    assert len(ready) == 0

    tail = acc.flush()
    assert len(tail) == 2


def test_packing_accumulator_shuffle_length() -> None:
    """Test shuffle_strategy='length' sorts by descending length."""
    acc = PackingAccumulator(
        max_length=10,
        num_bins=10,
        length_fn=_simple_length_fn,
        algorithm="first_fit",
        drop_oversized=False,
        min_sequence_length=1,
        shuffle_strategy="length",
        shuffle_seed=0,
        pack_payloads_fn=_identity_pack_fn,
        flush_strategy="fifo",
    )

    # Sequences in ascending order: [2, 3, 4, 5]
    # After sorting descending: [5, 4, 3, 2]
    # Expected packing: bin1=[5,4] (9), bin2=[3,2] (5)
    items = [_rec(0, 2), _rec(1, 3), _rec(2, 4), _rec(3, 5)]
    ready = acc.push_many(items)

    # No immediate emissions (no full bins)
    assert len(ready) == 0

    tail = acc.flush()
    assert len(tail) == 2

    # Find the bin with total_length=9 (the [5,4] bin)
    total_lengths = [
        rb[0][0].meta.tags["_packing_metadata"]["total_length"] for rb in tail
    ]
    assert sorted(total_lengths) == [5, 9]


def test_pack_sequences_operator_passthrough() -> None:
    """Test that operator process_many passes through pre-packed records."""
    op = PackSequences(
        max_length=10, length_fn="length", algorithm="first_fit", num_bins=10
    )

    # Create a mock packed record (as would be produced by accumulator)
    mock_packed = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"packed_samples": [{"value": 1}]},
    )

    result = op.process_many([mock_packed])
    assert len(result) == 1
    assert result[0] is mock_packed


def test_packing_accumulator_num_bins_limit() -> None:
    """Test that num_bins limit triggers flush of completed bins."""
    acc = PackingAccumulator(
        max_length=10,
        num_bins=2,
        length_fn=_simple_length_fn,
        algorithm="first_fit",
        drop_oversized=False,
        min_sequence_length=1,
        shuffle_strategy=None,
        shuffle_seed=0,
        pack_payloads_fn=_identity_pack_fn,
        flush_strategy="fifo",
    )

    # [3, 4, 2] creates bin1 with remaining=1 (complete)
    # [5] creates bin2 with remaining=5
    # [6] can't fit in either, triggers flush
    ready = []
    ready.extend(acc.push_many([_rec(0, 3)]))  # bin1: [3] remaining=7
    ready.extend(acc.push_many([_rec(1, 4)]))  # bin1: [3,4] remaining=3
    ready.extend(acc.push_many([_rec(2, 2)]))  # bin1: [3,4,2] remaining=1
    ready.extend(acc.push_many([_rec(3, 5)]))  # bin2: [5] remaining=5, at limit
    ready.extend(acc.push_many([_rec(4, 6)]))  # triggers flush of completed

    # Should have flushed bin1 when creating bin3
    assert len(ready) == 1
    assert len(ready[0][0][0].payload["packed_samples"]) == 3

    # Flush remaining
    tail = acc.flush()
    assert len(tail) == 2  # bin2 and bin3
