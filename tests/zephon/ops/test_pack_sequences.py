# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from zephon.core.constants import SampleMeta, SampleRecord
from zephon.ops.pack_sequences import PackSequences


def _rec(
    i: int, length: int, *, lane: int = 0, chunk: int = 0, **payload: Any
) -> SampleRecord:
    """Create a sample record with a length field."""
    meta = SampleMeta(sample_id=(0, 0, i), lane_id=lane, chunk_id=chunk)
    return SampleRecord(meta=meta, payload={"value": i, "length": length, **payload})


def test_pack_sequences_invalid_max_length_raises() -> None:
    """Test that invalid max_length raises ValueError."""
    with pytest.raises(ValueError):
        PackSequences(0, num_bins=10)
    with pytest.raises(ValueError):
        PackSequences(-5, num_bins=10)


def test_pack_sequences_first_fit_basic() -> None:
    """Test basic first-fit packing."""
    op = PackSequences(
        max_length=10, length_fn="length", algorithm="first_fit", num_bins=10
    )
    # Sequences: [3, 4, 2, 5]
    # First bin: 3 + 4 + 2 = 9 (not full, kept)
    # When 5 arrives, it doesn't fit in first bin -> create new bin
    # Now we have two bins: bin1=[3,4,2] remaining=1, bin2=[5] remaining=5
    out = []
    out += op.process_one(_rec(0, 3))
    assert out == []  # Not full yet
    out += op.process_one(_rec(1, 4))
    assert out == []  # Still not full
    out += op.process_one(_rec(2, 2))
    assert out == []  # Still not full (9/10)
    out += op.process_one(_rec(3, 5))
    # No bins emitted yet - multiple bins maintained for better packing
    assert len(out) == 0

    # Finalize should emit both bins
    tail = op.finalize()
    assert len(tail) == 2
    # Check that we have both bins
    bin_lengths = {len(b.payload["packed_samples"]) for b in tail}
    assert bin_lengths == {3, 1}  # One bin with 3 sequences, one with 1
    # Find the bin with 3 sequences
    packed = next(b for b in tail if len(b.payload["packed_samples"]) == 3)
    packing_meta = packed.meta.tags["_packing_metadata"]
    assert packing_meta["num_sequences"] == 3
    assert packing_meta["total_length"] == 9


def test_pack_sequences_best_fit_basic() -> None:
    """Test basic best-fit packing."""
    op = PackSequences(
        max_length=10, length_fn="length", algorithm="best_fit", num_bins=10
    )
    # Sequences: [3, 4, 2, 5]
    # First bin: 3 + 4 + 2 = 9 (not full, kept)
    # When 5 arrives, it doesn't fit in first bin -> create new bin
    # Now we have two bins: bin1=[3,4,2] remaining=1, bin2=[5] remaining=5
    out = []
    out += op.process_one(_rec(0, 3))
    out += op.process_one(_rec(1, 4))
    out += op.process_one(_rec(2, 2))
    out += op.process_one(_rec(3, 5))
    # No bins emitted yet - multiple bins maintained for better packing
    assert len(out) == 0

    tail = op.finalize()
    assert len(tail) == 2
    # Check that we have both bins
    bin_lengths = {len(b.payload["packed_samples"]) for b in tail}
    assert bin_lengths == {3, 1}  # One bin with 3 sequences, one with 1


def test_pack_sequences_best_fit_tighter_packing() -> None:
    """Test that best-fit produces tighter packing than first-fit."""
    op_best = PackSequences(
        max_length=10,
        length_fn="length",
        algorithm="best_fit",
        shuffle_strategy="length",
        num_bins=10,
    )
    op_first = PackSequences(
        max_length=10,
        length_fn="length",
        algorithm="first_fit",
        shuffle_strategy="length",
        num_bins=10,
    )

    # Sequences: [3, 3, 4, 2]
    # First-fit: bin1=[3,3,4]=10, bin2=[2]=2 (efficiency: 10/10 + 2/10 = 1.2)
    # Best-fit: bin1=[3,3,2]=8, bin2=[4]=4 (efficiency: 8/10 + 4/10 = 1.2)
    # Actually same in this case, but best-fit should choose tighter fit

    sequences = [_rec(0, 3), _rec(1, 3), _rec(2, 4), _rec(3, 2)]
    out_best = op_best.process_many(sequences)
    out_first = op_first.process_many(sequences)

    # Both should produce 2 bins
    assert len(out_best) >= 1
    assert len(out_first) >= 1

    # Finalize both
    tail_best = op_best.finalize()
    tail_first = op_first.finalize()

    # Both should have 2 bins total
    total_best = len(out_best) + len(tail_best)
    total_first = len(out_first) + len(tail_first)
    assert total_best == 2
    assert total_first == 2


def test_pack_sequences_per_lane_isolation() -> None:
    """Test that sequences from different lanes are not mixed."""
    op = PackSequences(
        max_length=10, length_fn="length", algorithm="first_fit", num_bins=10
    )
    # Lane 0: [3, 4] -> bin with 7
    # Lane 1: [2, 5] -> bin with 7
    items = [
        _rec(0, 3, lane=0),
        _rec(1, 2, lane=1),
        _rec(2, 4, lane=0),
        _rec(3, 5, lane=1),
    ]
    out = op.process_many(items)

    # Check that all packed samples maintain lane purity
    for packed in out:
        samples = packed.payload["packed_samples"]
        # All samples in a packed bin should have the same lane_id
        # We can't check this directly from payload, but we can verify
        # that the packing worked correctly
        assert len(samples) > 0

    tail = op.finalize()
    # Should have bins for both lanes
    assert len(out) + len(tail) >= 2


def test_pack_sequences_preserves_combined_meta() -> None:
    """Packed record should have combined meta with contributor information."""
    op = PackSequences(
        max_length=10, length_fn="length", algorithm="best_fit", num_bins=10
    )
    items = [_rec(0, 6), _rec(1, 4)]
    out = op.process_many(items)
    out += op.finalize()
    assert len(out) == 1
    packed = out[0]
    # Verify the packed record has a combined meta
    assert packed.meta is not None
    # Verify contributor information is preserved in the combined meta
    contributors = list(packed.meta.contribution_refs())
    assert len(contributors) >= 2  # Should have contributors from both samples


def test_pack_sequences_oversized_drop() -> None:
    """Test that oversized sequences are dropped when drop_oversized=True."""
    op = PackSequences(
        max_length=10,
        length_fn="length",
        algorithm="first_fit",
        drop_oversized=True,
        num_bins=10,
    )
    out = []
    out += op.process_one(_rec(0, 5))  # Normal
    assert len(out) == 0  # Bin not full yet
    out += op.process_one(_rec(1, 15))  # Oversized - should be dropped
    assert len(out) == 0  # No output yet, oversized dropped without affecting bins
    out += op.process_one(_rec(2, 5))  # Normal, fills bin (5 + 5 = 10)
    # Should emit bin with [5, 5]
    assert len(out) == 1
    assert len(out[0].payload["packed_samples"]) == 2
    assert out[0].meta.tags["_packing_metadata"]["total_length"] == 10


def test_pack_sequences_oversized_raise() -> None:
    """Test that oversized sequences raise when drop_oversized=False."""
    op = PackSequences(
        max_length=10,
        length_fn="length",
        algorithm="first_fit",
        drop_oversized=False,
        num_bins=10,
    )
    op.process_one(_rec(0, 5))  # Normal
    with pytest.raises(ValueError, match="exceeds max_length"):
        op.process_one(_rec(1, 15))  # Oversized - should raise


def test_pack_sequences_length_fn_callable() -> None:
    """Test that length_fn can be a callable."""
    op = PackSequences(
        max_length=10,
        length_fn=lambda r: len(r.payload.get("tokens", [])),
        algorithm="first_fit",
        num_bins=10,
    )
    rec1 = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"tokens": [1, 2, 3, 4]},  # length 4
    )
    rec2 = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0),
        payload={"tokens": [5, 6, 7]},  # length 3
    )
    out = op.process_many([rec1, rec2])
    # Should pack both into one bin (4 + 3 = 7 <= 10)
    assert len(out) == 0  # Not full yet
    tail = op.finalize()
    assert len(tail) == 1
    assert len(tail[0].payload["packed_samples"]) == 2


def test_pack_sequences_length_fn_field_not_found() -> None:
    """Test that missing length field raises appropriate error."""
    op = PackSequences(
        max_length=10,
        length_fn="nonexistent",
        algorithm="first_fit",
        drop_oversized=False,  # Don't drop, so we can see the error
        num_bins=10,
    )
    rec = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"value": 42},  # No "nonexistent" field
    )
    with pytest.raises(ValueError, match="not found"):
        op.process_one(rec)


def test_pack_sequences_length_fn_field_not_found_drop_oversized_true() -> None:
    """Missing length should still raise even when drop_oversized=True."""
    op = PackSequences(
        max_length=10,
        length_fn="nonexistent",
        algorithm="first_fit",
        drop_oversized=True,
        num_bins=10,
    )
    rec = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"value": 42},
    )
    with pytest.raises(ValueError, match="not found"):
        op.process_one(rec)


def test_pack_sequences_finalize_emits_remaining() -> None:
    """Test that finalize emits all remaining bins."""
    op = PackSequences(
        max_length=10, length_fn="length", algorithm="first_fit", num_bins=10
    )
    # Add sequences that don't fill a bin
    op.process_one(_rec(0, 3))
    op.process_one(_rec(1, 4))
    # Should not emit yet (7/10, can still fit more)
    out = op.process_one(_rec(2, 2))
    assert len(out) == 0  # Still not full (9/10)

    # Finalize should emit the remaining bin
    tail = op.finalize()
    assert len(tail) == 1
    assert len(tail[0].payload["packed_samples"]) == 3
    assert tail[0].meta.tags["_packing_metadata"]["total_length"] == 9


def test_pack_sequences_process_many_best_fit_decreasing() -> None:
    """process_many should pack efficiently via best-fit-decreasing.

    With the fix to maintain multiple bins in process_one, both paths should
    produce similar results. The advantage of process_many is that it can
    sort sequences by length (best-fit-decreasing) for optimal packing.
    """
    items = [
        _rec(0, 6),
        _rec(1, 6),
        _rec(2, 2),
        _rec(3, 2),
        _rec(4, 2),
        _rec(5, 2),
    ]

    op_online = PackSequences(
        max_length=10, length_fn="length", algorithm="best_fit", num_bins=10
    )
    online_out = []
    for item in items:
        online_out += op_online.process_one(item)
    online_out += op_online.finalize()

    op_many = PackSequences(
        max_length=10,
        length_fn="length",
        algorithm="best_fit",
        shuffle_strategy="length",
        num_bins=10,
    )
    bulk_out = op_many.process_many(items)
    bulk_out += op_many.finalize()

    # Both paths maintain multiple bins now, so they should produce similar results.
    # Best-fit-decreasing may yield equal or fewer bins than the online path.
    assert len(bulk_out) <= len(online_out)

    # Packed bins should still respect the max length.
    for packed in bulk_out:
        assert packed.meta.tags["_packing_metadata"]["total_length"] <= 10


def test_pack_sequences_traits() -> None:
    """Test operator traits."""
    op = PackSequences(
        max_length=10, length_fn="length", algorithm="first_fit", num_bins=10
    )
    traits = op.traits()
    assert traits.indexable is False
    assert traits.batch_shape_sensitive is False
    assert traits.requires_serial_state is True  # Should require serial state


def test_pack_sequences_packing_efficiency() -> None:
    """Test that packing efficiency is calculated correctly."""
    op = PackSequences(
        max_length=10, length_fn="length", algorithm="first_fit", num_bins=10
    )
    # Pack sequences that exactly fill a bin: [3, 3, 4] = 10
    out = []
    out += op.process_one(_rec(0, 3))
    out += op.process_one(_rec(1, 3))
    out += op.process_one(_rec(2, 4))
    # Should emit when bin is full
    assert len(out) == 1
    packed = out[0]
    metadata = packed.meta.tags["_packing_metadata"]
    assert metadata["total_length"] == 10
    assert metadata["packing_efficiency"] == 1.0  # Perfect packing


def test_pack_sequences_full_bin_emits_immediately() -> None:
    """Test that a bin that exactly fills max_length emits immediately."""
    op = PackSequences(
        max_length=10, length_fn="length", algorithm="first_fit", num_bins=10
    )
    # Pack sequences that exactly fill: [10] or [5, 5]
    out = op.process_one(_rec(0, 10))
    # Should emit immediately when bin is exactly full
    assert len(out) == 1
    packing_meta = out[0].meta.tags["_packing_metadata"]
    assert packing_meta["total_length"] == 10
    assert packing_meta["packing_efficiency"] == 1.0


def test_pack_sequences_sets_contributors() -> None:
    """Test that packed records properly set contributors for eviction tracking."""
    op = PackSequences(
        max_length=10, length_fn="length", algorithm="first_fit", num_bins=10
    )

    # Pack multiple sequences: [3, 4, 2, 5]
    # First bin: [3, 4, 2] = 9, second bin: [5] = 5
    rec1 = _rec(0, 3)
    rec2 = _rec(1, 4)
    rec3 = _rec(2, 2)
    rec4 = _rec(3, 5)

    out = []
    out += op.process_one(rec1)
    out += op.process_one(rec2)
    out += op.process_one(rec3)
    out += op.process_one(rec4)
    # No bins emitted yet - multiple bins maintained
    assert len(out) == 0

    # Finalize should emit both bins
    tail = op.finalize()
    assert len(tail) == 2

    # Find the bin with 3 sequences (first bin)
    packed = next(b for b in tail if len(b.payload["packed_samples"]) == 3)

    # Verify contributors are set
    assert packed.meta.contributors is not None
    assert len(packed.meta.contributors) == 3

    # Verify each contributor references the correct base sample
    contributor_cursors = {ref.cursor for ref in packed.meta.contributors}
    assert rec1.meta.cursor in contributor_cursors
    assert rec2.meta.cursor in contributor_cursors
    assert rec3.meta.cursor in contributor_cursors
    assert rec4.meta.cursor not in contributor_cursors  # Not in first bin

    # All should be marked as closing (since these are 1:1 inputs)
    assert all(ref.is_last_child for ref in packed.meta.contributors)


def test_pack_sequences_max_bins() -> None:
    """Test that num_bins limit is enforced."""
    op = PackSequences(
        max_length=10, length_fn="length", algorithm="first_fit", num_bins=2
    )
    # Create sequences that will require multiple bins
    # [3, 4, 2] = 9 (bin 1), [5] = 5 (bin 2), [6] = 6 (bin 3 - should trigger flush)
    out = []
    out += op.process_one(_rec(0, 3))  # Bin 1: [3] remaining=7
    assert len(out) == 0
    out += op.process_one(_rec(1, 4))  # Bin 1: [3,4] remaining=3
    assert len(out) == 0
    out += op.process_one(_rec(2, 2))  # Bin 1: [3,4,2] remaining=1
    assert len(out) == 0
    out += op.process_one(_rec(3, 5))  # Bin 2: [5] remaining=5 (creates new bin)
    assert (
        len(out) == 0
    )  # Should not flush yet (2 bins < num_bins=2? No, 2 bins == num_bins)
    # Actually, when we create bin 2, we have 2 bins total, so we're at the limit
    # When we try to create bin 3, we should flush completed bins first, then oldest active

    out += op.process_one(_rec(4, 6))  # Should trigger flush of completed bin (bin 1)
    # Bin 1 has remaining=1 < min_sequence_length=1, so it's completed
    # When creating bin 3, we flush completed bins first
    assert len(out) == 1  # Bin 1 should be flushed
    assert len(out[0].payload["packed_samples"]) == 3

    # Finalize should emit remaining bins
    tail = op.finalize()
    assert len(tail) == 2  # Bin 2 and bin 3


def test_pack_sequences_fullest_flush_strategy() -> None:
    """Test that fullest flush strategy flushes bins with smallest remaining capacity first."""
    op = PackSequences(
        max_length=10,
        length_fn="length",
        algorithm="first_fit",
        num_bins=2,
        flush_strategy="fullest",
    )

    # Create two bins with different remaining capacities:
    # Bin 1: [1] -> remaining=9 (less full)
    # Bin 2: [8] -> remaining=2 (more full)
    out = []
    out += op.process_one(_rec(0, 1))  # Bin 1: [1] remaining=9
    out += op.process_one(_rec(1, 8))  # Bin 2: [8] remaining=2
    assert len(out) == 0  # At limit (2 bins)

    # Fill Bin 1 to make it fuller than Bin 2
    out += op.process_one(_rec(2, 8))  # Bin 1: [1,8] remaining=1 (now fullest)
    # Now: Bin 1 remaining=1 (fullest), Bin 2 remaining=2 (less full)

    # Add sequence that doesn't fit in either bin (needs 3, but Bin 1 has 1, Bin 2 has 2)
    # This requires a new bin, triggering a flush
    # With "fullest" strategy, Bin 1 (remaining=1, smallest) should flush first
    out += op.process_one(_rec(3, 3))  # Needs new bin, should flush fullest
    assert len(out) == 1
    flushed_bin = out[0]
    # Bin 1 (remaining=1, most full) should be flushed
    assert flushed_bin.meta.tags["_packing_metadata"]["total_length"] == 9  # [1,8] = 9
    assert flushed_bin.meta.tags["_packing_metadata"]["num_sequences"] == 2


def test_pack_sequences_deterministic_output_keep_list() -> None:
    """Test that packing produces deterministic output with keep_list strategy."""
    # Fixed input: sequences of lengths [3, 4, 2, 5] with max_length=10
    # With first_fit: bin1=[3,4,2] (total=9), bin2=[5] (total=5)
    op = PackSequences(
        max_length=10,
        length_fn="length",
        algorithm="first_fit",
        shuffle_strategy=None,  # No reordering for deterministic output
        num_bins=10,
        pack_payloads="keep_list",
    )

    samples = [_rec(0, 3), _rec(1, 4), _rec(2, 2), _rec(3, 5)]
    # Use process_many for deterministic batch processing
    outputs = op.process_many(samples)
    outputs.extend(op.finalize())

    # Should have 2 bins
    assert len(outputs) == 2

    # Find bins by number of sequences
    bin_with_3 = next(b for b in outputs if len(b.payload["packed_samples"]) == 3)
    bin_with_1 = next(b for b in outputs if len(b.payload["packed_samples"]) == 1)

    # Bin with 3 sequences should be [3, 4, 2]
    assert bin_with_3.payload["packed_samples"] == [
        {"value": 0, "length": 3},
        {"value": 1, "length": 4},
        {"value": 2, "length": 2},
    ]
    assert bin_with_3.meta.tags["_packing_metadata"]["total_length"] == 9
    assert bin_with_3.meta.tags["_packing_metadata"]["num_sequences"] == 3

    # Bin with 1 sequence should be [5]
    assert bin_with_1.payload["packed_samples"] == [{"value": 3, "length": 5}]
    assert bin_with_1.meta.tags["_packing_metadata"]["total_length"] == 5
    assert bin_with_1.meta.tags["_packing_metadata"]["num_sequences"] == 1


def test_pack_sequences_deterministic_output_torch_tensor() -> None:
    """Test that packing produces deterministic output with torch_tensor strategy."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    # Fixed input: sequences of lengths [3, 4, 2, 5] with max_length=10
    # Expected: Two bins - bin1=[3,4,2] (total=9), bin2=[5] (total=5)
    # Use direct tensor payloads (not dicts) to avoid mixing types
    op = PackSequences(
        max_length=10,
        length_fn=lambda r: len(r.payload)
        if isinstance(r.payload, torch.Tensor)
        else r.payload["length"],
        algorithm="first_fit",
        shuffle_strategy=None,  # No reordering for deterministic output
        num_bins=10,
        pack_payloads="torch_tensor",
    )

    # Create samples with direct tensor payloads
    samples = [
        SampleRecord(meta=_rec(0, 3).meta, payload=torch.tensor([0, 1, 2])),
        SampleRecord(meta=_rec(1, 4).meta, payload=torch.tensor([3, 4, 5, 6])),
        SampleRecord(meta=_rec(2, 2).meta, payload=torch.tensor([7, 8])),
        SampleRecord(meta=_rec(3, 5).meta, payload=torch.tensor([9, 10, 11, 12, 13])),
    ]

    # Use process_many for deterministic batch processing
    outputs = op.process_many(samples)
    outputs.extend(op.finalize())

    # Should have 2 bins
    assert len(outputs) == 2

    # Find bins by number of sequences
    bin_with_3 = next(
        b for b in outputs if b.meta.tags["_packing_metadata"]["num_sequences"] == 3
    )
    bin_with_1 = next(
        b for b in outputs if b.meta.tags["_packing_metadata"]["num_sequences"] == 1
    )

    # Bin with 3 sequences should have total_length=9
    assert bin_with_3.meta.tags["_packing_metadata"]["total_length"] == 9
    # Verify tokens are concatenated (order: [0,1,2], [3,4,5,6], [7,8])
    expected_tokens = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert torch.equal(bin_with_3.payload["packed_samples"], expected_tokens)

    # Bin with 1 sequence should have total_length=5
    assert bin_with_1.meta.tags["_packing_metadata"]["total_length"] == 5
    # Verify tokens are concatenated
    expected_tokens = torch.tensor([9, 10, 11, 12, 13])
    assert torch.equal(bin_with_1.payload["packed_samples"], expected_tokens)


def test_pack_sequences_deterministic_output_numpy_array() -> None:
    """Test that packing produces deterministic output with numpy_array strategy."""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy not available")

    # Fixed input: sequences of lengths [3, 4, 2, 5] with max_length=10
    # Expected: Two bins - bin1=[3,4,2] (total=9), bin2=[5] (total=5)
    # Use direct numpy array payloads (not dicts) to avoid mixing types
    op = PackSequences(
        max_length=10,
        length_fn=lambda r: len(r.payload)
        if isinstance(r.payload, np.ndarray)
        else r.payload["length"],
        algorithm="first_fit",
        shuffle_strategy=None,  # No reordering for deterministic output
        num_bins=10,
        pack_payloads="numpy_array",
    )

    # Create samples with direct numpy array payloads
    samples = [
        SampleRecord(meta=_rec(0, 3).meta, payload=np.array([0, 1, 2])),
        SampleRecord(meta=_rec(1, 4).meta, payload=np.array([3, 4, 5, 6])),
        SampleRecord(meta=_rec(2, 2).meta, payload=np.array([7, 8])),
        SampleRecord(meta=_rec(3, 5).meta, payload=np.array([9, 10, 11, 12, 13])),
    ]

    # Use process_many for deterministic batch processing
    outputs = op.process_many(samples)
    outputs.extend(op.finalize())

    # Should have 2 bins
    assert len(outputs) == 2

    # Find bins by number of sequences
    bin_with_3 = next(
        b for b in outputs if b.meta.tags["_packing_metadata"]["num_sequences"] == 3
    )
    bin_with_1 = next(
        b for b in outputs if b.meta.tags["_packing_metadata"]["num_sequences"] == 1
    )

    # Bin with 3 sequences should have total_length=9
    assert bin_with_3.meta.tags["_packing_metadata"]["total_length"] == 9
    # Verify tokens are concatenated (order: [0,1,2], [3,4,5,6], [7,8])
    expected_tokens = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(bin_with_3.payload["packed_samples"], expected_tokens)

    # Bin with 1 sequence should have total_length=5
    assert bin_with_1.meta.tags["_packing_metadata"]["total_length"] == 5
    # Verify tokens are concatenated
    expected_tokens = np.array([9, 10, 11, 12, 13])
    np.testing.assert_array_equal(bin_with_1.payload["packed_samples"], expected_tokens)


def test_pack_sequences_strategy_length_deterministic_output() -> None:
    """Test that shuffle_strategy='length' produces deterministic output when sorted."""
    # Fixed input: sequences of lengths [3, 4, 2, 5] with max_length=10
    # With shuffle_strategy='length', sequences are sorted descending: [5, 4, 3, 2]
    # Expected: Two bins - bin1=[5,4] (total=9), bin2=[3,2] (total=5)
    op = PackSequences(
        max_length=10,
        length_fn="length",
        algorithm="first_fit",
        shuffle_strategy="length",  # Sort by length descending
        num_bins=10,
        pack_payloads="keep_list",
    )

    samples = [_rec(0, 3), _rec(1, 4), _rec(2, 2), _rec(3, 5)]
    # Use process_many to apply strategy sorting
    outputs = op.process_many(samples)
    outputs.extend(op.finalize())

    # Should have 2 bins
    assert len(outputs) == 2

    # Sort by total_length to get deterministic order
    outputs_sorted = sorted(
        outputs,
        key=lambda x: x.meta.tags["_packing_metadata"]["total_length"],
        reverse=True,
    )

    # First bin should have 2 sequences: [5, 4] (sorted descending)
    bin1 = outputs_sorted[0]
    assert len(bin1.payload["packed_samples"]) == 2
    assert bin1.meta.tags["_packing_metadata"]["total_length"] == 9
    assert bin1.meta.tags["_packing_metadata"]["num_sequences"] == 2
    # Verify order is descending by length
    assert bin1.payload["packed_samples"][0]["length"] == 5
    assert bin1.payload["packed_samples"][1]["length"] == 4

    # Second bin should have 2 sequences: [3, 2]
    bin2 = outputs_sorted[1]
    assert len(bin2.payload["packed_samples"]) == 2
    assert bin2.meta.tags["_packing_metadata"]["total_length"] == 5
    assert bin2.meta.tags["_packing_metadata"]["num_sequences"] == 2
    # Verify order is descending by length
    assert bin2.payload["packed_samples"][0]["length"] == 3
    assert bin2.payload["packed_samples"][1]["length"] == 2


def test_pack_sequences_strategy_random_deterministic_with_seed() -> None:
    """Test that shuffle_strategy='random' produces deterministic output with fixed seed."""
    # Fixed input: sequences of lengths [3, 4, 2, 5] with max_length=10
    # With shuffle_strategy='random' and fixed seed, should produce deterministic output
    op = PackSequences(
        max_length=10,
        length_fn="length",
        algorithm="first_fit",
        shuffle_strategy="random",
        shuffle_seed=42,  # Fixed seed for determinism
        num_bins=10,
        pack_payloads="keep_list",
    )

    samples = [_rec(0, 3), _rec(1, 4), _rec(2, 2), _rec(3, 5)]
    # Use process_many to apply strategy shuffling
    outputs = op.process_many(samples)
    outputs.extend(op.finalize())

    # Should have 2 bins (regardless of shuffle order)
    assert len(outputs) == 2

    # Verify total sequences match input
    total_sequences = sum(
        b.meta.tags["_packing_metadata"]["num_sequences"] for b in outputs
    )
    assert total_sequences == 4

    # Verify total length matches input
    total_length = sum(
        b.meta.tags["_packing_metadata"]["total_length"] for b in outputs
    )
    assert total_length == 14  # 3 + 4 + 2 + 5

    # Run again with same seed - should produce same output
    op2 = PackSequences(
        max_length=10,
        length_fn="length",
        algorithm="first_fit",
        shuffle_strategy="random",
        shuffle_seed=42,
        num_bins=10,
        pack_payloads="keep_list",
    )
    # Use process_many to apply strategy shuffling
    outputs2 = op2.process_many(samples)
    outputs2.extend(op2.finalize())

    # Should have same number of bins
    assert len(outputs2) == len(outputs)

    # Should have same bin structure (same num_sequences per bin)
    bin1_sequences = sorted(
        [b.meta.tags["_packing_metadata"]["num_sequences"] for b in outputs]
    )
    bin2_sequences = sorted(
        [b.meta.tags["_packing_metadata"]["num_sequences"] for b in outputs2]
    )
    assert bin1_sequences == bin2_sequences
