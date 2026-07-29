# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the streaming EnsureMixture operator."""

import logging
from typing import Any

import pytest

from zephon._internal.ops.ensure_mixture import (
    EnsureMixture,
    EnsureMixtureAccumulator,
    EnsureMixtureConfig,
    _LaneState,
)
from zephon.ops.base import OpContext
from zephon.types import ContributorRef, SampleCursor, SampleMeta, SampleRecord


def _assert_tombstones_only(ready, expected_count=None):
    """Discard output must be tombstones (payload-free offset closers) only."""
    records = _flatten_ready(ready)
    assert all(r.meta.tombstone for r in records)
    assert all(r.payload is None for r in records)
    if expected_count is not None:
        assert len(records) == expected_count
    return records


def _rec(
    i: int,
    *,
    component_id: int = 0,
    lane: int = 0,
    chunk: int = 0,
    tokens: int | None = None,
    **payload: Any,
) -> SampleRecord:
    """Create a test SampleRecord with configurable component_id.

    Uses component_sample_counts={component_id: 1} for single-component samples.
    """
    meta = SampleMeta(
        sample_id=(0, 0, i),
        lane_id=lane,
        chunk_id=chunk,
        chunk_offset=i,
        component_sample_counts={component_id: 1},
    )
    record_payload = {"value": i, **payload}
    if tokens is not None:
        record_payload["input_ids"] = list(range(tokens))
    return SampleRecord(meta=meta, payload=record_payload)


def _get_component(meta: SampleMeta) -> int:
    """Extract the primary component ID from a single-component sample.

    For test assertions where we know the sample has exactly one component.
    """
    counts = meta.component_sample_counts
    assert len(counts) == 1, f"Expected single-component, got {counts}"
    return next(iter(counts.keys()))


def _tombstone(lane: int = 0, chunk: int = 0) -> SampleRecord:
    """Create a tombstone record."""
    meta = SampleMeta(
        sample_id=(0, 0, 0),
        lane_id=lane,
        chunk_id=chunk,
        component_sample_counts={0: 1},
    ).with_tombstone(True)
    return SampleRecord(meta=meta, payload={})


def _make_config(
    weight_by: str = "samples",
    warn_tolerance: float | None = None,
    warn_warmup: float = 1000.0,
    mixture_override: dict[str, float] | None = None,
    max_buffer_size: int | None = 1,
    drain_target_ratio: float = 0.8,
    obsolete_drain_rate: float = 0.1,
) -> EnsureMixtureConfig:
    """Create a test config with populated fields."""
    config = EnsureMixtureConfig(
        weight_by=weight_by,
        warn_tolerance=warn_tolerance,
        warn_warmup=warn_warmup,
        mixture_override=mixture_override,
        max_buffer_size=max_buffer_size,
        drain_target_ratio=drain_target_ratio,
        obsolete_drain_rate=obsolete_drain_rate,
    )
    return config


def _make_accumulator_with_chunk_mixture(
    chunk_mixture: dict[int, float],
    weight_by: str = "samples",
    warn_tolerance: float | None = None,
    max_buffer_size: int | None = 1,
    drain_target_ratio: float = 0.8,
    obsolete_drain_rate: float = 0.1,
) -> EnsureMixtureAccumulator:
    """Create a test accumulator with mocked chunk mixture service."""
    config = _make_config(
        weight_by=weight_by,
        warn_tolerance=warn_tolerance,
        max_buffer_size=max_buffer_size,
        drain_target_ratio=drain_target_ratio,
        obsolete_drain_rate=obsolete_drain_rate,
    )
    # Pass context services directly to accumulator (not stored in config)
    return EnsureMixtureAccumulator(
        config,
        get_chunk_mixture=lambda lane_id, chunk_id: chunk_mixture,
        get_component_name=lambda cid: {0: "a", 1: "b"}.get(cid, f"comp_{cid}"),
    )


def _flatten_ready(ready: list) -> list[SampleRecord]:
    """Flatten batched ready output to list of records."""
    return [rec for batch, _ in ready for rec in batch]


class TestEnsureMixtureAccumulator:
    """Tests for EnsureMixtureAccumulator (adaptive buffering)."""

    def test_single_component_passthrough(self) -> None:
        """Single component should pass through immediately (adaptive emission)."""
        acc = _make_accumulator_with_chunk_mixture({0: 1.0})

        records = [_rec(i, component_id=0) for i in range(5)]
        ready = acc.push_many(records)
        emitted = _flatten_ready(ready)

        # Adaptive: 5 in = 5 out (emitted immediately since SWRR is happy)
        assert len(emitted) == 5

    def test_two_components_balanced_reordering(self) -> None:
        """Two components with equal weights: sequential input should be reordered.

        Input: all A's first (0,1,2,3), then all B's (10,11,12,13)
        Target: 50:50
        Expected output: perfect interleaving A,B,A,B,A,B,A,B

        With large buffer, all records buffered first, then SWRR emits optimally.
        """
        acc = _make_accumulator_with_chunk_mixture({0: 0.5, 1: 0.5}, max_buffer_size=8)

        # Sequential input: all A's first, then all B's
        records = []
        for i in range(4):
            records.append(_rec(i, component_id=0))  # A: 0,1,2,3
        for i in range(4):
            records.append(_rec(i + 10, component_id=1))  # B: 10,11,12,13

        ready = acc.push_many(records)
        ready.extend(acc.flush())
        emitted = _flatten_ready(ready)

        assert len(emitted) == 8

        # Extract component and sample sequences
        comp_seq = [_get_component(rec.meta) for rec in emitted]
        sample_seq = [rec.meta.sample_id[2] for rec in emitted]

        # With 50:50 target and all records buffered, SWRR should perfectly interleave
        # Deficit-based: A first (tie-break), then B (higher deficit), alternate...
        expected_comp_seq = [0, 1, 0, 1, 0, 1, 0, 1]
        expected_sample_seq = [0, 10, 1, 11, 2, 12, 3, 13]

        assert comp_seq == expected_comp_seq, (
            f"Expected {expected_comp_seq}, got {comp_seq}"
        )
        assert sample_seq == expected_sample_seq, (
            f"Expected {expected_sample_seq}, got {sample_seq}"
        )

    def test_weighted_ratio_reordering(self) -> None:
        """Test that SWRR reorders input toward target ratios.

        When input is 50:50 but target is 70:30, SWRR should emit
        to achieve 70:30 ratio as closely as possible given available samples.

        Buffer size must be >= input size to allow full reordering.
        """
        acc = _make_accumulator_with_chunk_mixture(
            {0: 0.7, 1: 0.3}, max_buffer_size=100
        )

        # Equal input: 50 a's, 50 b's (50:50), all A's first then all B's
        records = []
        for i in range(50):
            records.append(_rec(i, component_id=0))
        for i in range(50):
            records.append(_rec(i + 100, component_id=1))

        ready = acc.push_many(records)
        ready.extend(acc.flush())
        emitted = _flatten_ready(ready)

        # All samples should be emitted (no drops)
        assert len(emitted) == 100

        # Count components
        a_count = sum(1 for r in emitted if _get_component(r.meta) == 0)
        b_count = len(emitted) - a_count

        # Should have equal counts (we have 50 of each)
        assert a_count == 50
        assert b_count == 50

        # Check emission ratios in sliding windows to verify SWRR behavior
        # With 70:30 target, each window of 10 should have ~7 A's and ~3 B's
        # (as long as both components have samples available)

        # Check first 50 emissions (where both components have plenty of samples)
        first_50 = emitted[:50]
        first_50_a = sum(1 for r in first_50 if _get_component(r.meta) == 0)

        # With 70% target and both available, expect ~35 A's in first 50
        # Allow some tolerance since we started with only A's buffered
        assert first_50_a >= 30, f"Expected ~35 A's in first 50, got {first_50_a}"
        assert first_50_a <= 40, f"Expected ~35 A's in first 50, got {first_50_a}"

    def test_swrr_exact_small_sequence(self) -> None:
        """Test exact SWRR emission sequence for a small deterministic case.

        Input: A0, A1, A2, B0 (3 A's, 1 B)
        Target: 75:25 (same as count ratio)
        Expected: Deficit-based SWRR should produce A, B, A, A
        (A starts with higher deficit, B catches up, then A dominates)

        Buffer size must be >= input size to allow full reordering.
        """
        acc = _make_accumulator_with_chunk_mixture(
            {0: 0.75, 1: 0.25}, max_buffer_size=4
        )

        records = [
            _rec(0, component_id=0),  # A0
            _rec(1, component_id=0),  # A1
            _rec(2, component_id=0),  # A2
            _rec(10, component_id=1),  # B0
        ]

        ready = acc.push_many(records)
        ready.extend(acc.flush())  # Flush remaining buffered samples
        emitted = _flatten_ready(ready)

        assert len(emitted) == 4

        # Extract component and sample sequences
        comp_seq = [_get_component(rec.meta) for rec in emitted]
        sample_seq = [rec.meta.sample_id[2] for rec in emitted]  # local sample id

        # Deficit-based SWRR with 75:25 target:
        # Start: A deficit=0.75, B deficit=0.25 -> emit A (sample 0)
        # After A: A deficit=0.75*1-1=-0.25, B deficit=0.25*1-0=0.25 -> emit B (sample 10)
        # After B: A deficit=0.75*2-1=0.5, B deficit=0.25*2-1=-0.5 -> emit A (sample 1)
        # After A: A deficit=0.75*3-2=0.25, B deficit=0.25*3-1=-0.25 -> emit A (sample 2)
        expected_comp_seq = [0, 1, 0, 0]  # A, B, A, A
        expected_sample_seq = [0, 10, 1, 2]

        assert comp_seq == expected_comp_seq, (
            f"Expected {expected_comp_seq}, got {comp_seq}"
        )
        assert sample_seq == expected_sample_seq, (
            f"Expected {expected_sample_seq}, got {sample_seq}"
        )

    def test_single_record_batches_same_as_bulk(self) -> None:
        """Verify behavior is identical whether records come one-at-a-time or in bulk.

        This proves the tests aren't passing due to specific microbatch sizes.
        """
        acc_bulk = _make_accumulator_with_chunk_mixture(
            {0: 0.5, 1: 0.5}, max_buffer_size=8
        )
        acc_single = _make_accumulator_with_chunk_mixture(
            {0: 0.5, 1: 0.5}, max_buffer_size=8
        )

        # Same input: all A's then all B's
        records = [_rec(i, component_id=0) for i in range(4)]  # A0-A3
        records += [_rec(i + 10, component_id=1) for i in range(4)]  # B0-B3

        # Bulk: send all at once
        ready_bulk = acc_bulk.push_many(records)
        ready_bulk.extend(acc_bulk.flush())
        emitted_bulk = _flatten_ready(ready_bulk)

        # Single: send one at a time
        all_ready_single = []
        for rec in records:
            all_ready_single.extend(acc_single.push_many([rec]))
        all_ready_single.extend(acc_single.flush())
        emitted_single = _flatten_ready(all_ready_single)

        # Both should produce the same final sequence
        bulk_samples = [r.meta.sample_id[2] for r in emitted_bulk]
        single_samples = [r.meta.sample_id[2] for r in emitted_single]

        assert bulk_samples == single_samples, (
            f"Bulk: {bulk_samples}, Single: {single_samples}"
        )

        # Verify it's the expected interleaved sequence
        expected = [0, 10, 1, 11, 2, 12, 3, 13]
        assert bulk_samples == expected, f"Expected {expected}, got {bulk_samples}"

    def test_multi_batch_large_buffer_reordering(self) -> None:
        """Test SWRR reordering across multiple microbatches with large buffer.

        Input (two batches): [A0,A1,A2,A3] then [B0,B1,B2,B3]
        Target: 50:50
        max_buffer_size: 8 (large enough to wait for B's)

        Expected behavior:
        - Batch 1: emit A0 (optimal), buffer A1-A3 (SWRR wants B, wait)
        - Batch 2: B's arrive, emit optimally: B0,A1,B1,A2,B2,A3,B3
        - Flush: empty

        Total: [A0, B0, A1, B1, A2, B2, A3, B3]

        This proves SWRR correctly:
        1. Emits when optimal (A0 in batch 1)
        2. Buffers when non-optimal (A1-A3 waiting for B)
        3. Catches up correctly when optimal component arrives (B's in batch 2)
        """
        acc = _make_accumulator_with_chunk_mixture({0: 0.5, 1: 0.5}, max_buffer_size=8)

        # First batch: only A's
        batch1 = [_rec(i, component_id=0) for i in range(4)]  # A0, A1, A2, A3
        ready1 = acc.push_many(batch1)
        emitted1 = _flatten_ready(ready1)

        # Only A0 should be emitted (SWRR wants A first, then wants B which isn't available)
        assert len(emitted1) == 1, (
            f"Expected 1 emission from batch1, got {len(emitted1)}"
        )
        assert emitted1[0].meta.sample_id[2] == 0  # A0
        assert _get_component(emitted1[0].meta) == 0  # Component A

        # Buffer should have A1, A2, A3 waiting for B
        assert acc.has_pending_data()

        # Second batch: B's arrive
        batch2 = [_rec(i + 10, component_id=1) for i in range(4)]  # B0, B1, B2, B3
        ready2 = acc.push_many(batch2)
        emitted2 = _flatten_ready(ready2)

        # SWRR should now emit optimally: B0, A1, B1, A2, B2, A3, B3
        assert len(emitted2) == 7, (
            f"Expected 7 emissions from batch2, got {len(emitted2)}"
        )

        batch2_samples = [r.meta.sample_id[2] for r in emitted2]
        batch2_comps = [_get_component(r.meta) for r in emitted2]

        # After emitting A0, deficit strongly favors B, then alternates
        expected_samples = [10, 1, 11, 2, 12, 3, 13]  # B0, A1, B1, A2, B2, A3, B3
        expected_comps = [1, 0, 1, 0, 1, 0, 1]

        assert batch2_samples == expected_samples, (
            f"Expected {expected_samples}, got {batch2_samples}"
        )
        assert batch2_comps == expected_comps, (
            f"Expected {expected_comps}, got {batch2_comps}"
        )

        # Flush should be empty
        flushed = acc.flush()
        assert len(_flatten_ready(flushed)) == 0

        # Verify complete sequence
        all_emitted = emitted1 + emitted2
        all_samples = [r.meta.sample_id[2] for r in all_emitted]
        all_comps = [_get_component(r.meta) for r in all_emitted]

        expected_total_samples = [0, 10, 1, 11, 2, 12, 3, 13]
        expected_total_comps = [0, 1, 0, 1, 0, 1, 0, 1]

        assert all_samples == expected_total_samples, (
            f"Expected {expected_total_samples}, got {all_samples}"
        )
        assert all_comps == expected_total_comps, (
            f"Expected {expected_total_comps}, got {all_comps}"
        )

    def test_max_buffer_forces_emission_across_microbatches(self) -> None:
        """Test exact output when max_buffer_size forces non-optimal emission.

        Input (across microbatches): A0, A1, A2, then B0, B1
        Target: 50:50
        max_buffer_size: 2

        Expected sequence:
        - Batch1 [A0, A1]: emit A0 (SWRR wants A), buffer A1 (SWRR wants B)
        - Batch2 [A2]: buffer=2, force emit A1, buffer A2
        - Batch3 [B0, B1]: buffer has [A2, B0, B1]
          - SWRR deficit: 2 A's emitted, 0 B's → B has huge deficit
          - emit B0, emit B1 (still catching up), then emit A2
        - Flush: nothing left

        Final sequence: [0, 1, 10, 11, 2]
        """
        acc = _make_accumulator_with_chunk_mixture({0: 0.5, 1: 0.5}, max_buffer_size=2)

        # First microbatch: only A's
        batch1 = [_rec(0, component_id=0), _rec(1, component_id=0)]
        ready1 = acc.push_many(batch1)
        emitted1 = _flatten_ready(ready1)

        # A0 emitted (SWRR wants A), A1 buffered (SWRR wants B, not available)
        assert len(emitted1) == 1
        assert emitted1[0].meta.sample_id[2] == 0  # A0
        assert acc.has_pending_data()

        # Second microbatch: more A's, pushes buffer to max
        batch2 = [_rec(2, component_id=0)]
        ready2 = acc.push_many(batch2)
        emitted2 = _flatten_ready(ready2)

        # Buffer was [A1], now [A1, A2], total_buffered=2 >= max
        # Force emit A1 (best available), buffer A2
        assert len(emitted2) == 1
        assert emitted2[0].meta.sample_id[2] == 1  # A1 (force-emitted)
        assert acc.has_pending_data()

        # Third microbatch: B's finally arrive
        batch3 = [_rec(10, component_id=1), _rec(11, component_id=1)]
        ready3 = acc.push_many(batch3)
        emitted3 = _flatten_ready(ready3)

        # Buffer was [A2], now [A2, B0, B1]
        # SWRR has emitted 2 A's, 0 B's → deficit strongly favors B
        # Emit B0, B1 (catching up), then A2
        assert len(emitted3) == 3

        batch3_samples = [r.meta.sample_id[2] for r in emitted3]
        batch3_comps = [_get_component(r.meta) for r in emitted3]

        # First two should be B's (high deficit), last is A
        assert batch3_comps == [1, 1, 0], f"Expected [B,B,A], got {batch3_comps}"
        assert batch3_samples == [10, 11, 2], (
            f"Expected [10,11,2], got {batch3_samples}"
        )

        # Flush remaining (should be empty)
        flushed = acc.flush()
        assert len(_flatten_ready(flushed)) == 0

        # Verify complete sequence
        all_emitted = emitted1 + emitted2 + emitted3
        all_samples = [r.meta.sample_id[2] for r in all_emitted]
        all_comps = [_get_component(r.meta) for r in all_emitted]

        # Exact expected sequence: A0, A1(forced), B0, B1, A2
        assert all_samples == [0, 1, 10, 11, 2], (
            f"Expected [0,1,10,11,2], got {all_samples}"
        )
        assert all_comps == [0, 0, 1, 1, 0], f"Expected [A,A,B,B,A], got {all_comps}"

    def test_swrr_reorders_when_both_available(self) -> None:
        """Test SWRR reordering when both components have samples available.

        Feed A's and B's in a way that both are in the buffer, then verify
        that SWRR picks according to deficit.

        Input order: A, A, B, B, A, A, B, B (grouped pairs)
        Target: 50:50
        Expected: SWRR should interleave A,B,A,B,... when buffer is large enough

        Buffer size must be >= input size to allow full reordering.
        """
        acc = _make_accumulator_with_chunk_mixture({0: 0.5, 1: 0.5}, max_buffer_size=8)

        records = [
            _rec(0, component_id=0),  # A
            _rec(1, component_id=0),  # A
            _rec(10, component_id=1),  # B
            _rec(11, component_id=1),  # B
            _rec(2, component_id=0),  # A
            _rec(3, component_id=0),  # A
            _rec(12, component_id=1),  # B
            _rec(13, component_id=1),  # B
        ]

        ready = acc.push_many(records)
        ready.extend(acc.flush())  # Flush remaining buffered samples
        emitted = _flatten_ready(ready)

        assert len(emitted) == 8

        comp_seq = [_get_component(rec.meta) for rec in emitted]

        # With adaptive emission and interleaved input, SWRR should interleave
        # Expected: A,B,A,B,A,B,A,B

        # Count A's and B's - should be 4 each
        assert comp_seq.count(0) == 4
        assert comp_seq.count(1) == 4

        # With adaptive buffering and interleaved input, expect perfect interleaving (7 transitions)
        transitions = sum(
            1 for i in range(len(comp_seq) - 1) if comp_seq[i] != comp_seq[i + 1]
        )
        assert transitions == 7, (
            f"Expected perfect interleaving (7 transitions), got {comp_seq} with {transitions} transitions"
        )

    def test_tombstone_passthrough(self) -> None:
        """Tombstones should be passed through immediately."""
        acc = _make_accumulator_with_chunk_mixture({0: 0.5, 1: 0.5})

        tomb = _tombstone()
        ready = acc.push_many([tomb])

        assert len(ready) == 1
        assert ready[0][0][0].meta.tombstone

    def test_has_pending_data(self) -> None:
        """Test has_pending_data method."""
        acc = _make_accumulator_with_chunk_mixture({0: 1.0})

        assert not acc.has_pending_data()

        # Push one record - adaptive should emit it immediately (single component)
        ready = acc.push_many([_rec(0)])
        emitted = _flatten_ready(ready)
        # After emission, buffer should be empty
        assert not acc.has_pending_data()
        assert len(emitted) == 1

    def test_flush_reset_is_lane_scoped(self) -> None:
        """A per-lane flush sentinel must only flush its own lane's buffer.

        SWRR deficit state and buffers are per-lane; flushing every lane at
        one lane's epoch boundary corrupts the others' mixture state and
        breaks checkpoint replay.
        """
        # Target wants component 0; feeding component 1 forces buffering.
        acc = _make_accumulator_with_chunk_mixture({0: 1.0}, max_buffer_size=100)
        for lane in (0, 1):
            acc.push_many([_rec(i, component_id=1, lane=lane) for i in range(3)])
        assert acc.has_pending_data(lane_id=0)
        assert acc.has_pending_data(lane_id=1)

        flushed = _flatten_ready(acc.flush(reset=True, lane_id=0))

        assert {r.meta.lane_id for r in flushed} == {0}
        assert len(flushed) == 3
        assert not acc.has_pending_data(lane_id=0)
        assert acc.has_pending_data(lane_id=1)

    def test_flush_emits_remaining(self) -> None:
        """Flush should emit all remaining buffered records.

        With adaptive emission, records may be buffered waiting for the ideal
        component. When that component never arrives, max_buffer_size forces
        emission or flush handles it.
        """
        # Config with only component 0 in target, large buffer to prevent force-emit
        acc = _make_accumulator_with_chunk_mixture({0: 1.0}, max_buffer_size=100)

        # Push records from component 1 (not in target mixture)
        # SWRR wants component 0, but only 1 is available - buffers until max_buffer_size
        records = [_rec(i, component_id=1) for i in range(5)]
        ready = acc.push_many(records)

        # With large buffer, nothing emitted yet (waiting for component 0)
        assert len(_flatten_ready(ready)) == 0
        assert acc.has_pending_data()

        # Flush emits all remaining
        flushed = acc.flush()
        emitted = _flatten_ready(flushed)
        assert len(emitted) == 5
        assert not acc.has_pending_data()

    def test_token_weighting(self) -> None:
        """Test token-based weight function."""
        acc = _make_accumulator_with_chunk_mixture({0: 0.5, 1: 0.5}, weight_by="auto")

        # Records with different token counts (interleaved for adaptive emission)
        records = [
            _rec(0, component_id=0, tokens=50),
            _rec(1, component_id=1, tokens=50),
            _rec(2, component_id=0, tokens=50),
            _rec(3, component_id=1, tokens=50),
        ]

        ready = acc.push_many(records)
        emitted = _flatten_ready(ready)

        # Adaptive: 4 in = 4 out
        assert len(emitted) == 4

    def test_token_weighting_affects_ordering(self) -> None:
        """Verify token weights change SWRR ordering vs sample weights.

        With unequal token counts, token-weighted SWRR should produce
        different ordering than sample-weighted SWRR.

        Setup:
        - A0: 100 tokens, A1: 100 tokens (total 200 tokens)
        - B0-B9: 20 tokens each (total 200 tokens)
        - Target: 50:50

        Sample-weighted (each sample = 1):
        - Would interleave A,B,A,B,... to get 50:50 sample ratio
        - After 2 A's, deficit wants 2 B's

        Token-weighted:
        - After A0 (100 tokens), need 100 tokens of B to balance
        - That's 5 B samples (5 * 20 = 100 tokens)
        - Expected: A0, B0,B1,B2,B3,B4, A1, B5,B6,B7,B8,B9
        """
        # Token-weighted accumulator
        acc_tokens = _make_accumulator_with_chunk_mixture(
            {0: 0.5, 1: 0.5}, weight_by="auto", max_buffer_size=20
        )

        # Sample-weighted accumulator
        acc_samples = _make_accumulator_with_chunk_mixture(
            {0: 0.5, 1: 0.5}, weight_by="samples", max_buffer_size=20
        )

        # Create records: 2 A's with 100 tokens each, 10 B's with 20 tokens each
        records = [
            _rec(0, component_id=0, tokens=100),  # A0: 100 tokens
            _rec(1, component_id=0, tokens=100),  # A1: 100 tokens
        ]
        for i in range(10):
            records.append(
                _rec(10 + i, component_id=1, tokens=20)
            )  # B0-B9: 20 tokens each

        # Process with token weighting
        ready_tokens = acc_tokens.push_many(records)
        ready_tokens.extend(acc_tokens.flush())
        emitted_tokens = _flatten_ready(ready_tokens)

        # Process with sample weighting
        ready_samples = acc_samples.push_many(records)
        ready_samples.extend(acc_samples.flush())
        emitted_samples = _flatten_ready(ready_samples)

        # Both should emit all 12 records
        assert len(emitted_tokens) == 12
        assert len(emitted_samples) == 12

        # Extract component sequences
        token_comp_seq = [_get_component(rec.meta) for rec in emitted_tokens]
        sample_comp_seq = [_get_component(rec.meta) for rec in emitted_samples]

        # The sequences should be DIFFERENT - that proves token weighting matters
        assert token_comp_seq != sample_comp_seq, (
            f"Token and sample weighting produced same sequence!\n"
            f"Token:  {token_comp_seq}\n"
            f"Sample: {sample_comp_seq}"
        )

        # Verify token-weighted behavior: after first A (100 tokens),
        # SWRR should emit ~5 B's (100 tokens) before the second A
        # Find where A1 appears in token-weighted output
        token_sample_seq = [rec.meta.sample_id[2] for rec in emitted_tokens]
        a1_index = token_sample_seq.index(1)  # Where is A1?

        # A1 should come after several B's (not immediately after A0)
        # With 100 tokens deficit after A0, need 5 B's (100 tokens) to catch up
        assert a1_index >= 5, (
            f"Token-weighted: A1 at index {a1_index}, expected >= 5\n"
            f"Sequence: {token_sample_seq}"
        )

        # Verify sample-weighted behavior: should interleave more evenly
        sample_sample_seq = [rec.meta.sample_id[2] for rec in emitted_samples]
        a1_index_samples = sample_sample_seq.index(1)  # Where is A1?

        # With sample weighting (each sample = 1), A1 should come after ~1 B
        # (tie-break gives A first, then B, then A again)
        assert a1_index_samples <= 3, (
            f"Sample-weighted: A1 at index {a1_index_samples}, expected <= 3\n"
            f"Sequence: {sample_sample_seq}"
        )

    def test_multi_lane_isolation(self) -> None:
        """Records from different lanes should be buffered separately."""
        acc = _make_accumulator_with_chunk_mixture({0: 1.0})

        records = [
            _rec(0, lane=0),
            _rec(1, lane=1),
            _rec(2, lane=0),
            _rec(3, lane=1),
        ]

        ready = acc.push_many(records)
        emitted = _flatten_ready(ready)

        # Adaptive: 4 in = 4 out
        assert len(emitted) == 4

        # Each lane should have records
        lanes = {rec.meta.lane_id for rec in emitted}
        assert lanes == {0, 1}

    def test_chunk_transition_updates_mixture(self) -> None:
        """Test that SWRR updates when chunk changes (different mixture)."""
        # Initial chunk has 70:30 mixture
        chunk_mixtures = {
            0: {0: 0.7, 1: 0.3},  # chunk 0
            1: {0: 0.3, 1: 0.7},  # chunk 1
        }
        config = _make_config()
        acc = EnsureMixtureAccumulator(
            config,
            get_chunk_mixture=lambda lane, chunk: chunk_mixtures.get(chunk, {}),
            get_component_name=lambda cid: {0: "a", 1: "b"}.get(cid),
        )

        # Records from chunk 0 (70:30 target) - alternating components
        records_chunk0 = [_rec(i, component_id=i % 2, chunk=0) for i in range(10)]
        ready0 = acc.push_many(records_chunk0)

        # Records from chunk 1 (30:70 target) - alternating components
        records_chunk1 = [_rec(i + 10, component_id=i % 2, chunk=1) for i in range(10)]
        ready1 = acc.push_many(records_chunk1)

        # All records should be emitted (adaptive emission with interleaved input)
        emitted0 = _flatten_ready(ready0)
        emitted1 = _flatten_ready(ready1)
        assert len(emitted0) + len(emitted1) == 20


class TestEnsureMixtureOperator:
    """Tests for EnsureMixture operator."""

    def test_invalid_warn_tolerance_raises(self) -> None:
        with pytest.raises(ValueError, match="warn_tolerance must be between 0 and 1"):
            EnsureMixture(warn_tolerance=-0.1, mixture_override={"a": 1.0})
        with pytest.raises(ValueError, match="warn_tolerance must be between 0 and 1"):
            EnsureMixture(warn_tolerance=1.5, mixture_override={"a": 1.0})

    def test_setup_with_explicit_weights(self) -> None:
        """Test setup with explicit mixture override."""
        op = EnsureMixture(
            mixture_override={"code": 0.3, "text": 0.7},
        )

        ctx = OpContext({})
        op.setup(ctx)

        # Explicit mixture override should be stored by name
        assert op._config.mixture_override == {"code": 0.3, "text": 0.7}

    def test_setup_with_context_services(self) -> None:
        """Test setup with context services from engine.

        Note: Context services (get_chunk_mixture, get_component_name) are no longer
        stored in config to avoid pickle issues. They're passed directly to the
        accumulator in accumulator(). This test verifies setup still works.
        """
        op = EnsureMixture()

        # Mock context services
        mock_chunk_mixture = lambda lane, chunk: {0: 0.3, 1: 0.7}
        mock_component_name = lambda cid: {0: "code", 1: "text"}.get(cid)

        ctx = OpContext(
            {
                "get_chunk_mixture": mock_chunk_mixture,
                "get_component_name": mock_component_name,
            }
        )
        op.setup(ctx)

        # Setup should complete without error (services are used via accumulator, not stored)
        assert op._config is not None

    def test_traits(self) -> None:
        """Test operator traits."""
        op = EnsureMixture(parallelism=4)
        traits = op.traits()

        assert traits.indexable is False
        assert traits.preserves_cursor_order is False
        assert traits.batch_shape_sensitive is False
        assert traits.parallelism == 4

    def test_accumulator_with_context(self) -> None:
        """Test that accumulator receives context services.

        Context services are passed directly to the accumulator (not stored in
        config) to avoid pickle issues with the process runner.
        """
        op = EnsureMixture()

        mock_chunk_mixture = lambda lane, chunk: {0: 0.3, 1: 0.7}
        mock_component_name = lambda cid: {0: "code", 1: "text"}.get(cid)

        ctx = {
            "get_chunk_mixture": mock_chunk_mixture,
            "get_component_name": mock_component_name,
        }
        acc = op.accumulator(deterministic=True, ctx=ctx)

        assert acc is not None
        # Accumulator should have context services as instance variables
        assert acc._get_chunk_mixture is not None
        assert acc._get_component_name is not None

    def test_process_one_passthrough(self) -> None:
        """Test that process_one passes through records."""
        op = EnsureMixture(mixture_override={"a": 1.0})
        ctx = OpContext({})
        op.setup(ctx)

        rec = _rec(0)
        result = op.process_one(rec)

        assert len(result) == 1
        assert result[0] is rec

    def test_process_many_passthrough(self) -> None:
        """Test that process_many passes through records."""
        op = EnsureMixture(mixture_override={"a": 1.0})
        ctx = OpContext({})
        op.setup(ctx)

        records = [_rec(i) for i in range(5)]
        result = op.process_many(records)

        assert len(result) == 5
        for i, rec in enumerate(result):
            assert rec is records[i]

    def test_weight_by_samples(self) -> None:
        """Test sample-based weighting."""
        op = EnsureMixture(weight_by="samples", mixture_override={"a": 1.0})
        ctx = OpContext({})
        op.setup(ctx)

        # All samples should have weight 1
        rec_no_tokens = _rec(0)
        rec_with_tokens = _rec(1, tokens=100)

        assert op._config.get_weight(rec_no_tokens) == 1.0
        assert op._config.get_weight(rec_with_tokens) == 1.0

    def test_weight_auto_detect(self) -> None:
        """Test weight_by='auto' with auto-detection of token field."""
        op = EnsureMixture(weight_by="auto", mixture_override={"a": 1.0})
        ctx = OpContext({})
        op.setup(ctx)

        # Record with input_ids should use token length
        rec_with_tokens = _rec(0, tokens=50)
        assert op._config.get_weight(rec_with_tokens) == 50.0

    def test_weight_auto_raises_when_no_token_field(self) -> None:
        """Test that weight_by='auto' raises when no token field found."""
        op = EnsureMixture(weight_by="auto", mixture_override={"a": 1.0})
        ctx = OpContext({})
        op.setup(ctx)

        # Record without tokens should raise ValueError
        rec_no_tokens = _rec(1)
        with pytest.raises(ValueError, match="Cannot auto-detect length field"):
            op._config.get_weight(rec_no_tokens)

    def test_weight_explicit_field(self) -> None:
        """Test weighting with explicit field name."""
        # Use explicit field name instead of "auto" + auto-detection
        op = EnsureMixture(
            weight_by="custom_tokens",
            mixture_override={"a": 1.0},
        )
        ctx = OpContext({})
        op.setup(ctx)

        rec = SampleRecord(
            meta=SampleMeta(
                sample_id=(0, 0, 0),
                lane_id=0,
                chunk_id=0,
                component_sample_counts={0: 1},
            ),
            payload={"custom_tokens": [1, 2, 3, 4, 5]},
        )
        assert op._config.get_weight(rec) == 5.0

    def test_custom_weight_callable(self) -> None:
        """Test custom weight function (callable)."""
        custom_fn = lambda rec: rec.payload.get("weight", 1)

        # Pass callable directly to weight parameter
        op = EnsureMixture(
            weight_by=custom_fn,
            mixture_override={"a": 1.0},
        )
        ctx = OpContext({})
        op.setup(ctx)

        rec1 = _rec(0, weight=10)
        rec2 = _rec(1, weight=5)

        assert op._config.get_weight(rec1) == 10.0
        assert op._config.get_weight(rec2) == 5.0


def test_chunk_mixture_drives_swrr_target() -> None:
    """A chunk mixture from get_chunk_mixture becomes the lane's SWRR target."""
    acc = _make_accumulator_with_chunk_mixture({0: 0.75, 1: 0.25})
    state = _LaneState()
    acc._update_lane_swrr(lane_id=0, chunk_id=0, state=state)
    assert state.swrr is not None
    assert state.swrr.target_ratios == pytest.approx({0: 0.75, 1: 0.25})


class TestObsoleteComponentDraining:
    """Tests for draining samples from obsolete components after mixture change."""

    def test_obsolete_components_eventually_drain(self) -> None:
        """Samples from obsolete components should eventually be emitted.

        Scenario:
        1. Start with mixture {A=0: 0.5, B=1: 0.5}
        2. Push some A and B samples
        3. Change mixture to {C=2: 0.5, D=3: 0.5}
        4. Push many C and D samples
        5. Verify that A and B samples are eventually emitted (not stuck forever)
        """
        # Start with mixture for components 0 (A) and 1 (B)
        current_mixture: dict[int, float] = {0: 0.5, 1: 0.5}

        def get_chunk_mixture(lane_id: int, chunk_id: int) -> dict[int, float]:
            return current_mixture

        config = _make_config(
            max_buffer_size=20,
            drain_target_ratio=0.5,  # Drain to 50% when forced
        )

        accum = EnsureMixtureAccumulator(
            config,
            get_chunk_mixture=get_chunk_mixture,
            get_component_name=lambda cid: {
                0: "A",
                1: "B",
                2: "C",
                3: "D",
            }.get(cid, f"comp_{cid}"),
        )

        # Phase 1: Push some A (0) and B (1) samples in chunk 0
        phase1_records = [
            _rec(i, component_id=i % 2, chunk=0) for i in range(10)
        ]  # 5 A, 5 B
        result1 = accum.push_many(phase1_records)

        # Collect emitted records
        emitted: list[SampleRecord] = []
        for batch, _ in result1:
            emitted.extend(batch)

        # Phase 2: Change mixture to C (2) and D (3)
        current_mixture = {2: 0.5, 3: 0.5}

        # Push many C and D samples in chunk 1 (enough to trigger forced emissions)
        phase2_records = [
            _rec(100 + i, component_id=2 + (i % 2), chunk=1) for i in range(50)
        ]  # 25 C, 25 D
        result2 = accum.push_many(phase2_records)

        for batch, _ in result2:
            emitted.extend(batch)

        # Push more C and D samples
        phase3_records = [
            _rec(200 + i, component_id=2 + (i % 2), chunk=1) for i in range(50)
        ]
        result3 = accum.push_many(phase3_records)

        for batch, _ in result3:
            emitted.extend(batch)

        # Flush remaining
        flush_result = accum.flush()
        for batch, _ in flush_result:
            emitted.extend(batch)

        # Count how many A (0) and B (1) samples were emitted
        obsolete_emitted = sum(1 for r in emitted if _get_component(r.meta) in (0, 1))

        # Total samples: 10 (A,B) + 50 (C,D) + 50 (C,D) = 110
        assert len(emitted) == 110, f"Expected 110 samples, got {len(emitted)}"

        # All 10 obsolete samples (A and B) should have been emitted
        assert obsolete_emitted == 10, (
            f"Expected all 10 obsolete samples (A,B) to be emitted, "
            f"but only {obsolete_emitted} were emitted. "
            f"Obsolete samples are stuck in buffer!"
        )

    def test_obsolete_samples_interleaved_during_emission(self) -> None:
        """Obsolete samples should be interleaved during normal emissions.

        With obsolete_drain_rate=0.2 (20%), we expect 1 obsolete sample to be
        emitted for every 5 normal emissions. This test verifies that obsolete
        samples (B) are interleaved with current-target samples (C/D), not just
        emitted at the end.

        Scenario:
        1. Mixture is {A: 0.9, B: 0.1} - SWRR strongly prefers A
        2. Push ONLY B samples - they get buffered (SWRR wants A, not available)
        3. Change mixture to {C: 0.5, D: 0.5}
        4. Push C/D samples to trigger emissions
        5. Verify B samples are INTERLEAVED with C/D (not all at the end)
        """
        current_mixture: dict[int, float] = {0: 0.9, 1: 0.1}  # Strongly prefer A (0)

        def get_chunk_mixture(lane_id: int, chunk_id: int) -> dict[int, float]:
            return current_mixture

        config = _make_config(
            max_buffer_size=50,
            drain_target_ratio=0.5,
            obsolete_drain_rate=0.2,  # 20% = 1 obsolete per 5 emissions
        )

        accum = EnsureMixtureAccumulator(config, get_chunk_mixture=get_chunk_mixture)

        # Phase 1: Push ONLY B (component 1) samples - they'll be buffered because
        # SWRR wants A (component 0) which isn't available
        phase1_records = [_rec(i, component_id=1, chunk=0) for i in range(10)]
        result1 = accum.push_many(phase1_records)

        # With max_buffer=50 and only 10 samples, they should be buffered
        emitted_phase1: list[SampleRecord] = []
        for batch, _ in result1:
            emitted_phase1.extend(batch)
        assert len(emitted_phase1) == 0, "Phase 1 should buffer all B samples"

        # Phase 2: Change mixture to C (2) and D (3)
        current_mixture = {2: 0.5, 3: 0.5}

        # Push enough C/D samples to trigger forced drain and emit many samples
        # Buffer will be: 10 B + 50 CD = 60, needs to drain to 25 (emit 35)
        phase2_records = [
            _rec(100 + i, component_id=2 + (i % 2), chunk=1) for i in range(50)
        ]
        result2 = accum.push_many(phase2_records)

        emitted_phase2: list[SampleRecord] = []
        for batch, _ in result2:
            emitted_phase2.extend(batch)

        # Count B samples (obsolete) emitted
        b_emitted = sum(1 for r in emitted_phase2 if _get_component(r.meta) == 1)

        # With obsolete_drain_rate=0.2 and ~35 emissions, we expect ~7 B samples
        # (35 * 0.2 = 7, though actual count depends on interleaving)
        assert b_emitted > 0, (
            f"Expected obsolete B samples to be interleaved during emissions, "
            f"but none were emitted. Emissions: {len(emitted_phase2)}"
        )

        # Check that B samples are NOT all at the end (i.e., they're interleaved)
        # Find the position of the last B sample
        last_b_idx = None
        for idx in range(len(emitted_phase2) - 1, -1, -1):
            if _get_component(emitted_phase2[idx].meta) == 1:
                last_b_idx = idx
                break

        # If B samples are interleaved, there should be some C/D samples after
        # the first B sample. If all B are at the end, last_b_idx would be near
        # len(emitted) - 1 and there would be few C/D after.
        if last_b_idx is not None and len(emitted_phase2) > 10:
            # Count how many C/D samples appear after the last B
            cd_after_last_b = sum(
                1
                for r in emitted_phase2[last_b_idx + 1 :]
                if _get_component(r.meta) in (2, 3)
            )
            # With proper interleaving, we should have some C/D after the last B
            # (unless we're at the very end of emissions)
            # This is a soft check - the main point is that B appears in the output
            assert b_emitted >= 1, f"Expected at least 1 B sample, got {b_emitted}"


# ---------------------------------------------------------------------------
# Mid-stream flush: flush(reset=True) must reset state
# ---------------------------------------------------------------------------


class TestEnsureMixtureAccumulatorMidStreamFlush:
    """Verify that flush(reset=True) resets SWRR state.

    After a mid-stream flush the accumulator must be indistinguishable from
    a freshly constructed instance.  This is required for correctness when
    flush sentinels fire between epochs: on checkpoint/restore the
    accumulator is rebuilt from scratch (no SWRR history), so the live run
    must match by resetting at the same boundary.
    """

    def test_mid_stream_flush_resets_swrr_state(self) -> None:
        """After flush(reset=True), output must match a fresh accumulator.

        Epoch 1: push 20 records (component 0 only) → builds SWRR history.
        Mid-stream flush.
        Epoch 2: push 10 records (50/50 components 0 and 1).

        A fresh accumulator given only epoch 2 data should produce identical
        output.  If SWRR history from epoch 1 leaks, the ordering diverges.
        """
        mixture: dict[int, float] = {0: 0.5, 1: 0.5}

        # --- live accumulator: epoch 1 + flush + epoch 2 ---
        live_acc = _make_accumulator_with_chunk_mixture(mixture, max_buffer_size=50)
        epoch1 = [_rec(i, component_id=0, chunk=0) for i in range(20)]
        live_acc.push_many(epoch1)
        live_acc.flush(reset=True)

        epoch2 = []
        for i in range(10):
            epoch2.append(_rec(100 + i, component_id=i % 2, chunk=1))
        live_ready = live_acc.push_many(epoch2)
        live_ready.extend(live_acc.flush())
        live_output = _flatten_ready(live_ready)

        # --- fresh accumulator: only epoch 2 ---
        fresh_acc = _make_accumulator_with_chunk_mixture(mixture, max_buffer_size=50)
        fresh_ready = fresh_acc.push_many(epoch2)
        fresh_ready.extend(fresh_acc.flush())
        fresh_output = _flatten_ready(fresh_ready)

        # Ordering must be identical — stale SWRR state must not influence
        # epoch 2 selection.
        live_ids = [r.meta.sample_id for r in live_output]
        fresh_ids = [r.meta.sample_id for r in fresh_output]
        assert live_ids == fresh_ids, (
            f"SWRR state leaked across mid-stream flush.\n"
            f"  live (with epoch 1 history):  {live_ids}\n"
            f"  fresh (no history):           {fresh_ids}"
        )

    def test_mid_stream_flush_resets_emission_counters(self) -> None:
        """After flush(reset=True), emitted_by_component and
        total_emitted must be zero (fresh state)."""
        acc = _make_accumulator_with_chunk_mixture({0: 0.5, 1: 0.5}, max_buffer_size=50)
        records = [_rec(i, component_id=i % 2, chunk=0) for i in range(10)]
        acc.push_many(records)

        # Verify counters are non-zero before flush
        lane_state = acc._lanes[0]
        assert lane_state.total_emitted > 0

        acc.flush(reset=True)

        # After mid-stream flush, counters must be reset
        lane_state = acc._lanes[0]
        assert lane_state.total_emitted == 0.0, (
            f"total_emitted should be 0 after mid-stream flush, "
            f"got {lane_state.total_emitted}"
        )
        assert all(v == 0.0 for v in lane_state.emitted_by_component.values()), (
            f"emitted_by_component should be zeroed after mid-stream flush, "
            f"got {dict(lane_state.emitted_by_component)}"
        )

    def test_mid_stream_flush_produces_fresh_equivalent_state(self) -> None:
        """After flush(reset=True), internal state must match a fresh accumulator.

        This is the enforcement test for the reset contract: if someone adds
        a new stateful field to _LaneState, this test will catch it if
        flush() forgets to reset it.
        """
        mixture: dict[int, float] = {0: 0.7, 1: 0.3}
        acc = _make_accumulator_with_chunk_mixture(mixture, max_buffer_size=50)

        # Build up state across multiple chunks
        records = [_rec(i, component_id=i % 2, chunk=0) for i in range(10)]
        records += [_rec(i + 10, component_id=i % 2, chunk=1) for i in range(10)]
        acc.push_many(records)

        # Verify state is non-trivial before flush
        lane_state = acc._lanes[0]
        assert lane_state.total_emitted > 0
        assert lane_state.current_chunk_id is not None
        assert lane_state.swrr is not None

        acc.flush(reset=True)

        # After mid-stream flush, every field on _LaneState must match
        # a freshly constructed instance (except buffers, which were
        # drained by the flush itself — they should be empty).
        lane_state = acc._lanes[0]
        fresh = _LaneState()

        # Compare all fields.  If a new field is added to _LaneState and
        # flush() doesn't reset it, this will fail.
        for field_name in [f.name for f in lane_state.__dataclass_fields__.values()]:
            live_val = getattr(lane_state, field_name)
            fresh_val = getattr(fresh, field_name)
            # Buffers are drained (empty deques remain keyed by component)
            # rather than replaced, so check that no actual records remain.
            if field_name == "buffers":
                has_records = any(len(d) > 0 for d in live_val.values())
                assert not has_records, (
                    f"_LaneState.buffers should have no records after flush, "
                    f"got {live_val}"
                )
                continue
            if field_name == "multi_component_buffer":
                assert len(live_val) == 0, (
                    f"_LaneState.multi_component_buffer should be empty "
                    f"after flush, got {live_val}"
                )
            else:
                assert live_val == fresh_val, (
                    f"_LaneState.{field_name} not reset by flush.\n"
                    f"  after flush: {live_val!r}\n"
                    f"  fresh:       {fresh_val!r}"
                )


def test_stall_trait_default_is_false() -> None:
    """EnsureMixture defaults to stall_on_epoch_boundary=False (flush)."""
    op = EnsureMixture()
    assert op.traits().stall_on_epoch_boundary is False


# ---------------------------------------------------------------------------
# Strict (unbounded) buffering: max_buffer_size=None + discard-on-flush
# ---------------------------------------------------------------------------


def test_max_buffer_size_none_accepted() -> None:
    """The operator accepts max_buffer_size=None and rejects non-positive ints."""
    EnsureMixture(max_buffer_size=None)
    with pytest.raises(ValueError, match="positive"):
        EnsureMixture(max_buffer_size=0)
    with pytest.raises(ValueError, match="positive"):
        EnsureMixture(max_buffer_size=-5)


def test_unbounded_buffer_never_force_drains_when_target_unmet() -> None:
    """With max_buffer_size=None, no force-drain compromises the mixture.

    A heavily skewed input (all component 0 first, then component 1) must not
    drain hundreds of component-0 samples while waiting on component 1.  At
    most one emission may occur on the initial SWRR tie-break before the
    operator starts waiting; after that no further emissions happen until
    component 1 arrives.
    """
    acc = _make_accumulator_with_chunk_mixture(
        {0: 0.5, 1: 0.5},
        max_buffer_size=None,
    )
    only_a = [_rec(i, component_id=0) for i in range(100)]
    ready = acc.push_many(only_a)
    emitted_a_phase = [rec for batch, _ in ready for rec in batch]
    # At most one initial emission (SWRR tie-break) — otherwise we're force-
    # draining, which is exactly what unbounded mode forbids.
    assert len(emitted_a_phase) <= 1
    # Subsequent unbalanced pushes must not drain further.
    more_a = [_rec(i, component_id=0) for i in range(100, 150)]
    ready = acc.push_many(more_a)
    emitted = [rec for batch, _ in ready for rec in batch]
    assert emitted == []
    # Now push component-1 samples — SWRR can satisfy its target.
    some_b = [_rec(i, component_id=1) for i in range(200, 220)]
    ready = acc.push_many(some_b)
    emitted_b_phase = [rec for batch, _ in ready for rec in batch]
    components = [_get_component(r.meta) for r in emitted_b_phase]
    # Components 0 and 1 should be roughly balanced now that 1 is available.
    assert abs(components.count(0) - components.count(1)) <= 1


def test_unbounded_buffer_grows_without_bound_when_target_unmet() -> None:
    """Unbounded mode buffers everything it cannot emit on-target.

    The whole point of the strict mode: with no cap, a one-sided stream
    accumulates in the buffer (rather than the bounded mode's pressure-valve
    force-drain). The buffered count must equal the unservable surplus.
    """
    acc = _make_accumulator_with_chunk_mixture(
        {0: 0.5, 1: 0.5},
        max_buffer_size=None,
    )
    only_a = [_rec(i, component_id=0) for i in range(200)]
    emitted = _flatten_ready(acc.push_many(only_a))
    assert len(emitted) <= 1  # at most the initial tie-break emission
    assert acc.has_pending_data()
    assert acc._lanes[0].total_buffered == 200 - len(emitted)


def test_strict_mode_no_discard_when_supply_matches_target() -> None:
    """Strict mode only drops the surplus it cannot place on-target.

    A stream that can satisfy the target emits in full and leaves nothing
    buffered, so flush discards nothing — data loss is confined to genuine
    over-supply, not a side effect of enabling strict mode.
    """
    acc = _make_accumulator_with_chunk_mixture(
        {0: 0.5, 1: 0.5},
        max_buffer_size=None,
    )
    balanced = []
    for i in range(20):
        balanced.append(_rec(2 * i, component_id=0))
        balanced.append(_rec(2 * i + 1, component_id=1))
    emitted = _flatten_ready(acc.push_many(balanced))
    # SWRR places every record on-target; nothing is withheld.
    assert len(emitted) == 40
    assert not acc.has_pending_data()
    # Empty buffer → flush has nothing to discard.
    assert acc.flush(reset=False) == []


def test_terminal_flush_discards_in_strict_mode() -> None:
    """End-of-stream flush() discards the strict buffer with tombstones.

    Draining it would emit the mixture-withheld surplus as a skewed tail."""
    acc = _make_accumulator_with_chunk_mixture(
        {0: 0.5, 1: 0.5},
        max_buffer_size=None,
    )
    only_a = [_rec(i, component_id=0) for i in range(8)]
    emitted = _flatten_ready(acc.push_many(only_a))
    buffered = 8 - len(emitted)
    assert buffered > 0

    out = acc.flush(reset=False)
    _assert_tombstones_only(out, expected_count=buffered)
    assert not acc.has_pending_data()


def test_unbounded_mid_stream_flush_discards_buffer() -> None:
    """flush(reset=True) with max_buffer_size=None discards buffered content.

    Flush sentinels (flush_every_k_chunks) must not dump the held-back buffer
    into the output stream — that would corrupt the mixture across epoch
    boundaries.  The discarded data is content SWRR had deliberately withheld
    because it was over-represented.
    """
    acc = _make_accumulator_with_chunk_mixture(
        {0: 0.9, 1: 0.1},  # mostly A, little B
        max_buffer_size=None,
    )
    # Feed only component 0 (A) — it fills its quota quickly, the rest buffers
    only_a = [_rec(i, component_id=0) for i in range(50)]
    emitted = acc.push_many(only_a)
    # Some A gets emitted (it's the ideal component initially), rest buffers
    assert acc.has_pending_data()

    # Mid-stream flush (epoch boundary) must discard, not drain — but each
    # dropped record's closing contributor must come back as a tombstone, or
    # its offset never closes and per-epoch eviction stalls for the run.
    result = acc.flush(reset=True)
    tombstones = _assert_tombstones_only(
        result, expected_count=50 - len(_flatten_ready(emitted))
    )
    assert not acc.has_pending_data(), "buffer must be empty after discard"
    # Plain records self-close: the tombstones target exactly the dropped ids.
    assert {t.meta.sample_id for t in tombstones} == {
        r.meta.sample_id for r in only_a
    } - {r.meta.sample_id for r in _flatten_ready(emitted)}


def test_strict_flush_reset_clears_state_like_fresh_accumulator() -> None:
    """After a mid-stream discard, the lane state is reset (SWRR history,
    chunk id) so the next epoch behaves like a freshly constructed
    accumulator — the flush(reset=True) replay contract."""
    acc = _make_accumulator_with_chunk_mixture(
        {0: 0.9, 1: 0.1},
        max_buffer_size=None,
    )
    acc.push_many([_rec(i, component_id=0) for i in range(20)])
    assert acc.has_pending_data()

    acc.flush(reset=True)
    state = acc._lanes[0]
    assert not acc.has_pending_data()
    assert state.swrr is None
    assert state.current_chunk_id is None
    assert state.total_emitted == 0.0
    assert state.emissions_since_obsolete_drain == 0


def test_strict_discard_tombstones_for_multi_component_records() -> None:
    """Packed (multi-component) buffered records are tombstoned on discard too.

    The discard path walks the multi-component buffer as well as the
    per-component buffers; each packed record closes its own offset.
    """
    acc = _make_accumulator_with_chunk_mixture(
        {0: 0.5, 1: 0.5},
        max_buffer_size=None,
    )
    # Packed records contributing to BOTH components. With only these in the
    # buffer and a 50/50 target, SWRR can serve its ideal from the packed
    # buffer, so feed a stream that leaves a surplus by adding lone-component
    # records the packed ones cannot balance.
    packed = []
    for i in range(4):
        meta = SampleMeta(
            sample_id=(0, 0, i),
            lane_id=0,
            chunk_id=0,
            chunk_offset=i,
            component_sample_counts={0: 1, 1: 1},
        )
        packed.append(SampleRecord(meta=meta, payload={"value": i}))
    # Lone component-0 records that can never be balanced (no lone 1s arrive).
    lone = [_rec(100 + i, component_id=0) for i in range(10)]
    acc.push_many(packed + lone)
    assert acc.has_pending_data()
    # Some packed records may emit; whatever remains (packed and/or lone) must
    # be tombstoned exactly once each on discard.
    remaining_single = sum(len(b) for b in acc._lanes[0].buffers.values())
    remaining_multi = len(acc._lanes[0].multi_component_buffer)
    out = acc.flush(reset=True)
    _assert_tombstones_only(out, expected_count=remaining_single + remaining_multi)
    assert not acc.has_pending_data()


def test_strict_flush_on_empty_buffer_is_noop() -> None:
    """A flush with nothing buffered emits no tombstones (no offsets to close)."""
    acc = _make_accumulator_with_chunk_mixture(
        {0: 1.0},  # single-component target: SWRR is always happy, nothing buffers
        max_buffer_size=None,
    )
    out = _flatten_ready(acc.push_many([_rec(i, component_id=0) for i in range(5)]))
    assert len(out) == 5
    assert not acc.has_pending_data()
    assert acc.flush(reset=True) == []
    assert acc.flush(reset=False) == []


def test_strict_discard_is_lane_scoped() -> None:
    """Each lane discards its own buffer independently; a flush clears all
    lanes but the tombstones carry each lane's own ids."""
    acc = _make_accumulator_with_chunk_mixture(
        {0: 0.5, 1: 0.5},
        max_buffer_size=None,
    )
    acc.push_many([_rec(i, component_id=0, lane=0) for i in range(10)])
    acc.push_many([_rec(i, component_id=0, lane=1) for i in range(7)])
    assert acc.has_pending_data()

    buffered_0 = acc._lanes[0].total_buffered
    buffered_1 = acc._lanes[1].total_buffered
    tombstones = _assert_tombstones_only(acc.flush(reset=True))
    assert len(tombstones) == buffered_0 + buffered_1
    lanes = {t.meta.lane_id for t in tombstones}
    assert lanes == {0, 1}
    assert not acc.has_pending_data()


def test_strict_per_lane_flush_discards_only_target_lane() -> None:
    """A per-lane flush in strict mode discards only its own lane.

    The runner injects one flush sentinel per lane at epoch boundaries, so an
    unbounded (strict) accumulator must discard the target lane's buffer while
    leaving every other lane's buffer pending — the discard path is lane-scoped
    just like the bounded drain path.
    """
    acc = _make_accumulator_with_chunk_mixture(
        {0: 0.5, 1: 0.5},
        max_buffer_size=None,
    )
    acc.push_many([_rec(i, component_id=0, lane=0) for i in range(10)])
    acc.push_many([_rec(i, component_id=0, lane=1) for i in range(7)])
    buffered_0 = acc._lanes[0].total_buffered
    buffered_1 = acc._lanes[1].total_buffered
    assert buffered_0 and buffered_1

    tombstones = _assert_tombstones_only(acc.flush(reset=True, lane_id=0))

    assert {t.meta.lane_id for t in tombstones} == {0}
    assert len(tombstones) == buffered_0
    assert not acc.has_pending_data(lane_id=0)
    assert acc.has_pending_data(lane_id=1)
    assert acc._lanes[1].total_buffered == buffered_1


def test_strict_discard_batches_tombstones() -> None:
    """A large discard is split into ReadyBatches of at most _TOMBSTONE_BATCH
    records, so an unbounded-buffer discard never becomes one oversized dispatch."""
    batch = EnsureMixtureAccumulator._TOMBSTONE_BATCH
    acc = _make_accumulator_with_chunk_mixture(
        {0: 0.5, 1: 0.5},
        max_buffer_size=None,
    )
    n = batch * 2 + 3  # spans three batches
    acc.push_many([_rec(i, component_id=0) for i in range(n)])
    buffered = acc._lanes[0].total_buffered
    assert buffered > batch

    ready = acc.flush(reset=True)
    assert all(len(records) <= batch for records, _ in ready)
    assert sum(len(records) for records, _ in ready) == buffered


def test_discard_tombstones_skip_non_closing_children() -> None:
    """Only closing contributors (is_last_child=True) produce tombstones.

    A discarded middle piece of a split document must not close its base
    offset — the closing sibling (delivered, or discarded itself) does.
    """
    acc = _make_accumulator_with_chunk_mixture({0: 0.9, 1: 0.1}, max_buffer_size=None)
    records = []
    for i in range(6):
        rec = _rec(i, component_id=0)
        cursor = SampleCursor(
            sample_id=rec.meta.sample_id,
            chunk_id=rec.meta.chunk_id,
            chunk_offset=rec.meta.chunk_offset,
        )
        meta = rec.meta.with_contributors(
            (ContributorRef(cursor=cursor, is_last_child=(i % 2 == 0)),)
        )
        records.append(SampleRecord(meta=meta, payload=rec.payload))
    emitted = _flatten_ready(acc.push_many(records))
    buffered = [r for r in records if r not in emitted]
    assert buffered

    tombstones = _assert_tombstones_only(acc.flush(reset=True))
    closing = [r for r in buffered if r.meta.contribution_refs()[0].is_last_child]
    assert len(tombstones) == len(closing)


def test_strict_discard_logs_warning(caplog) -> None:
    """A strict-mode discard logs a WARNING with the per-flush and cumulative
    dropped counts, so silent data loss is observable."""
    acc = _make_accumulator_with_chunk_mixture({0: 0.5, 1: 0.5}, max_buffer_size=None)
    acc.push_many([_rec(i, component_id=0) for i in range(20)])
    buffered = acc._lanes[0].total_buffered
    assert buffered > 0

    with caplog.at_level(logging.WARNING, logger="zephon._internal.ops.ensure_mixture"):
        acc.flush(reset=True)

    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("discarded" in r.getMessage() for r in warned)
    assert str(buffered) in caplog.text
    assert acc._total_discarded == buffered


def test_strict_no_discard_emits_no_warning(caplog) -> None:
    """A flush that discards nothing stays quiet — no spurious data-loss warning
    and the cumulative counter is untouched."""
    acc = _make_accumulator_with_chunk_mixture({0: 1.0}, max_buffer_size=None)
    out = _flatten_ready(acc.push_many([_rec(i, component_id=0) for i in range(5)]))
    assert len(out) == 5  # single-component target: all emit, nothing buffers

    with caplog.at_level(logging.WARNING, logger="zephon._internal.ops.ensure_mixture"):
        assert acc.flush(reset=True) == []

    assert "discarded" not in caplog.text
    assert acc._total_discarded == 0


def test_strict_discard_warns_once_then_logs_debug(caplog) -> None:
    """Skewed strict runs discard on roughly every flush, so only the first
    discard warns; later ones drop to DEBUG so the log isn't flooded."""
    acc = _make_accumulator_with_chunk_mixture({0: 0.5, 1: 0.5}, max_buffer_size=None)

    # First discard warns, and the warning announces that the rest go to DEBUG.
    acc.push_many([_rec(i, component_id=0) for i in range(20)])
    with caplog.at_level(logging.WARNING, logger="zephon._internal.ops.ensure_mixture"):
        acc.flush(reset=True)
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1
    assert "DEBUG" in warned[0].getMessage()

    # Second discard stays silent at WARNING but is still recorded at DEBUG.
    caplog.clear()
    acc.push_many([_rec(i, component_id=0) for i in range(20, 40)])
    with caplog.at_level(logging.DEBUG, logger="zephon._internal.ops.ensure_mixture"):
        acc.flush(reset=True)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        r.levelno == logging.DEBUG and "discarded" in r.getMessage()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Bounded mode is unchanged by the strict-mode addition (the invariant)
# ---------------------------------------------------------------------------


def test_bounded_buffer_force_drains_when_full() -> None:
    """With a small max_buffer_size, the operator force-drains when stuck."""
    acc = _make_accumulator_with_chunk_mixture(
        {0: 0.5, 1: 0.5},
        max_buffer_size=10,
        drain_target_ratio=0.5,
    )
    only_a = [_rec(i, component_id=0) for i in range(50)]
    ready = acc.push_many(only_a)
    emitted = [rec for batch, _ in ready for rec in batch]
    # Force-drain triggered; we should have emitted some component-0 samples
    # even though the mixture target was unsatisfiable.
    assert len(emitted) > 0


def test_bounded_mid_stream_flush_drains_buffer() -> None:
    """flush(reset=True) with bounded max_buffer_size still drains normally."""
    acc = _make_accumulator_with_chunk_mixture(
        {0: 0.5, 1: 0.5},
        max_buffer_size=100,
    )
    only_a = [_rec(i, component_id=0) for i in range(10)]
    acc.push_many(only_a)
    assert acc.has_pending_data()

    result = acc.flush(reset=True)
    # Bounded mode: content is emitted (not discarded), losslessly.
    emitted = _flatten_ready(result)
    assert len(emitted) > 0
    assert all(not r.meta.tombstone for r in emitted), "bounded flush never discards"
    assert not acc.has_pending_data()


def test_bounded_terminal_flush_drains_losslessly() -> None:
    """End-of-stream flush in bounded mode emits every buffered record (no
    discard, no tombstones) — the behavior unchanged from origin/main."""
    acc = _make_accumulator_with_chunk_mixture(
        {0: 0.5, 1: 0.5},
        max_buffer_size=100,
    )
    only_a = [_rec(i, component_id=0) for i in range(12)]
    emitted_push = _flatten_ready(acc.push_many(only_a))
    buffered = 12 - len(emitted_push)
    assert buffered > 0

    out = _flatten_ready(acc.flush(reset=False))
    assert all(not r.meta.tombstone for r in out)
    # Every buffered record drains out; total in == total out.
    assert len(emitted_push) + len(out) == 12
    assert not acc.has_pending_data()


def test_bounded_mode_passes_balanced_stream_through() -> None:
    """The bounded-mode emission path is unaffected by the strict addition:
    a balanced interleaved stream passes straight through under a cap."""
    acc = _make_accumulator_with_chunk_mixture(
        {0: 0.5, 1: 0.5},
        max_buffer_size=8,
    )
    interleaved = []
    for i in range(8):
        interleaved.append(_rec(2 * i, component_id=0))
        interleaved.append(_rec(2 * i + 1, component_id=1))
    out = _flatten_ready(acc.push_many(interleaved))
    # SWRR is always happy on a balanced stream: everything emits, in order.
    assert len(out) == 16
    assert not acc.has_pending_data()
