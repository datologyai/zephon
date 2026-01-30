# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for shard prefetching operator."""

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from zephon.core.accumulators import CountingAccumulator
from zephon.core.op_base import OpContext
from zephon.io.dataset import Dataset
from zephon.io.types import LocalShardFile, LocalShardRef, ShardLocator
from zephon.ops.prefetch import PrefetchOp


class MockResolver:
    """Mock resolver that tracks prefetch requests and simulates download latency."""

    def __init__(self, base_latency_ms: float = 10.0):
        self.base_latency_ms = base_latency_ms
        # Track (dataset_id, shard_id) tuples that were resolved
        self.resolve_calls: list[
            tuple[int, int, float]
        ] = []  # (dataset_id, shard_id, timestamp)
        self.cached_shards: set[tuple[int, int]] = set()
        self.lock = threading.Lock()

    def resolve(self, locator: ShardLocator, *, blocking: bool = True) -> LocalShardRef:
        """Simulate resolving a shard with download latency."""
        # Extract dataset_id from locator.dataset (which is the dataset name)
        # We'll use the shard_id as key since that's what we have
        key = (hash(locator.dataset) % 1000, locator.shard_id)

        with self.lock:
            self.resolve_calls.append((key[0], locator.shard_id, time.time()))
            is_cached = key in self.cached_shards

        if not is_cached and blocking:
            # Simulate download time
            time.sleep(self.base_latency_ms / 1000.0)

        with self.lock:
            self.cached_shards.add(key)

        return LocalShardRef(
            raw=LocalShardFile(path=Path(f"/tmp/shard_{locator.shard_id}"), bytes=1000),
            cache_hit=is_cached,
        )

    def get_resolved_shards(self) -> list[int]:
        """Get list of shard IDs that were resolved (in order)."""
        with self.lock:
            return [shard_id for _, shard_id, _ in self.resolve_calls]

    def get_unique_resolved_shards(self) -> set[int]:
        """Get set of unique shard IDs that were resolved."""
        with self.lock:
            return {shard_id for _, shard_id, _ in self.resolve_calls}

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about prefetch behavior."""
        with self.lock:
            return {
                "resolve_count": len(self.resolve_calls),
                "unique_shards": len({(d, s) for d, s, _ in self.resolve_calls}),
                "cached_count": len(self.cached_shards),
            }


def _setup_prefetch_with_mock(
    op: PrefetchOp,
    resolver: MockResolver,
    num_datasets: int = 1,
    num_shards_per_dataset: int = 100,
) -> None:
    """Set up a PrefetchOp with a mock resolver and locators.

    This bypasses the normal setup() flow to inject a mock resolver,
    allowing tests to verify prefetch behavior.
    """
    op._resolver = resolver
    op._locators = {}

    # Create mock locators for all dataset/shard combinations
    for dataset_id in range(num_datasets):
        for shard_id in range(num_shards_per_dataset):
            key = (dataset_id, shard_id)
            op._locators[key] = ShardLocator(
                dataset=f"dataset_{dataset_id}",
                shard_id=shard_id,
                format="jsonl",
                root="/tmp",
                raw=MagicMock(),
            )


@pytest.fixture
def mock_datasets():
    """Create mock dataset descriptors for testing."""
    return {
        0: Dataset(
            name="dataset_0",
            shard_index={},
            backend={"kind": "inmem", "shards": {}},  # In-memory for testing
            path=None,
        ),
        1: Dataset(
            name="dataset_1",
            shard_index={},
            backend={"kind": "inmem", "shards": {}},
            path=None,
        ),
    }


def test_prefetch_basic_setup(mock_datasets):
    """Test basic prefetch operator setup and configuration."""
    op = PrefetchOp(buffer_size=512)

    assert op.buffer_size == 512


def test_prefetch_invalid_config():
    """Test that invalid configurations raise errors."""
    with pytest.raises(ValueError, match="buffer_size must be positive"):
        PrefetchOp(buffer_size=0)


def test_prefetch_traits():
    """Test operator traits are correct."""
    op = PrefetchOp()
    traits = op.traits()

    assert traits.indexable is True
    assert traits.preserves_cursor_order is True
    assert traits.parallelism == 4  # Default parallelism for concurrent downloads


def test_prefetch_accumulator():
    """Test accumulator configuration."""
    op = PrefetchOp(buffer_size=1024)
    acc = op.accumulator(deterministic=True, ctx={})

    # Should return a counting accumulator with the configured buffer size
    assert isinstance(acc, CountingAccumulator)
    assert acc._max_batch == 1024
    assert acc._max_latency_ms is None  # None when deterministic=True


def test_prefetch_lookahead_simple():
    """Test that prefetch looks ahead and triggers downloads for unique shards."""
    op = PrefetchOp(buffer_size=10)
    resolver = MockResolver(base_latency_ms=1.0)
    _setup_prefetch_with_mock(op, resolver, num_datasets=1, num_shards_per_dataset=10)

    # Create a batch of samples from 3 different shards (4 samples each)
    # SampleId = (dataset_id, shard_id, sample_idx)
    samples = [
        (
            (0, shard_id, idx),
            0,
            0,
            idx,
            0,
        )  # EngineSample format: (SampleId, lane_id, chunk_id, chunk_offset, component_id)
        for shard_id in range(3)
        for idx in range(4)
    ]

    # Process the batch
    result = op.process_many(samples)

    # Should pass through unchanged
    assert result == samples

    # Should have resolved exactly 3 unique shards (0, 1, 2)
    resolved = resolver.get_unique_resolved_shards()
    assert resolved == {0, 1, 2}, f"Expected shards {{0, 1, 2}}, got {resolved}"

    # Total resolve calls should be 3 (one per unique shard, not 12 for each sample)
    stats = resolver.get_stats()
    assert stats["resolve_count"] == 3, (
        f"Expected 3 resolve calls, got {stats['resolve_count']}"
    )


@pytest.mark.parametrize("num_shards", [10, 50, 100])
@pytest.mark.parametrize("buffer_size", [128, 512, 1024])
def test_prefetch_stress_many_shards(num_shards, buffer_size):
    """Stress test with many shards and various buffer sizes.

    This test simulates a realistic scenario with:
    - Multiple datasets
    - Many shards per dataset
    - Sequential and interleaved access patterns
    - Various buffer sizes to test lookahead effectiveness
    """
    op = PrefetchOp(buffer_size=buffer_size)
    resolver = MockResolver(base_latency_ms=0.1)  # Fast for stress test
    _setup_prefetch_with_mock(
        op, resolver, num_datasets=2, num_shards_per_dataset=num_shards
    )

    # Create samples from many shards across multiple datasets
    samples = []
    expected_shards: set[tuple[int, int]] = set()
    for dataset_id in range(2):
        for shard_id in range(num_shards):
            expected_shards.add((dataset_id, shard_id))
            for sample_idx in range(10):  # 10 samples per shard
                sample_id = (dataset_id, shard_id, sample_idx)
                engine_sample = (sample_id, 0, 0, len(samples), 0)
                samples.append(engine_sample)

    # Process in batches to simulate streaming
    batch_size = min(buffer_size, len(samples))
    processed = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        result = op.process_many(batch)
        assert len(result) == len(batch), (
            f"Batch {i // batch_size}: expected {len(batch)} samples, got {len(result)}"
        )
        processed += len(result)

    assert processed == len(samples), (
        f"Expected to process {len(samples)} samples, but processed {processed}"
    )

    # Verify all unique shards were resolved
    stats = resolver.get_stats()
    expected_unique = num_shards * 2  # 2 datasets
    assert stats["unique_shards"] == expected_unique, (
        f"Expected {expected_unique} unique shards resolved, got {stats['unique_shards']}"
    )


def test_prefetch_interleaved_access():
    """Test prefetching with interleaved shard access patterns.

    This simulates a mixture dataset where samples from different shards
    are interleaved, which should trigger prefetch for each unique shard.
    """
    op = PrefetchOp(buffer_size=100)
    resolver = MockResolver(base_latency_ms=1.0)
    _setup_prefetch_with_mock(op, resolver, num_datasets=1, num_shards_per_dataset=20)

    # Create interleaved pattern: alternating between 4 specific shards
    target_shards = {0, 5, 10, 15}
    samples = []
    for round_idx in range(20):
        for shard_id in target_shards:
            sample_id = (0, shard_id, round_idx)
            engine_sample = (sample_id, 0, 0, len(samples), 0)
            samples.append(engine_sample)

    result = op.process_many(samples)
    assert len(result) == len(samples), (
        f"Expected {len(samples)} samples in result, got {len(result)}"
    )

    # Should have resolved exactly the 4 unique shards, despite 80 samples
    resolved = resolver.get_unique_resolved_shards()
    assert resolved == target_shards, f"Expected shards {target_shards}, got {resolved}"

    stats = resolver.get_stats()
    assert stats["resolve_count"] == 4, (
        f"Expected 4 resolve calls, got {stats['resolve_count']}"
    )


def test_prefetch_sequential_access():
    """Test prefetching with sequential shard access.

    This simulates reading a single dataset sequentially, shard by shard.
    Prefetch should download each shard exactly once.
    """
    op = PrefetchOp(buffer_size=200)
    resolver = MockResolver(base_latency_ms=1.0)
    _setup_prefetch_with_mock(op, resolver, num_datasets=1, num_shards_per_dataset=10)

    # Create sequential pattern: all samples from shard 0, then 1, then 2, etc.
    samples = []
    for shard_id in range(10):
        for sample_idx in range(20):
            sample_id = (0, shard_id, sample_idx)
            engine_sample = (sample_id, 0, 0, len(samples), 0)
            samples.append(engine_sample)

    # Process in smaller batches to simulate streaming
    for i in range(0, len(samples), 50):
        batch = samples[i : i + 50]
        result = op.process_many(batch)
        assert len(result) == len(batch), (
            f"Batch at offset {i}: expected {len(batch)} samples, got {len(result)}"
        )

    # Should have resolved all 10 unique shards
    # Note: resolve_count may be > 10 because deduplication only happens within
    # each batch, not across batches. A shard at the boundary of two batches
    # may be resolved twice.
    resolved = resolver.get_unique_resolved_shards()
    assert resolved == set(range(10)), f"Expected shards 0-9, got {resolved}"

    stats = resolver.get_stats()
    assert stats["unique_shards"] == 10, (
        f"Expected 10 unique shards, got {stats['unique_shards']}"
    )
    # resolve_count >= 10 (may have some re-resolves at batch boundaries)
    assert stats["resolve_count"] >= 10, (
        f"Expected at least 10 resolve calls, got {stats['resolve_count']}"
    )


def test_prefetch_determinism():
    """Test that prefetch operator is deterministic.

    The operator should yield samples in the exact same order across runs,
    regardless of prefetch timing.
    """
    op = PrefetchOp(buffer_size=100)

    samples = [
        ((0, shard_id, idx), 0, 0, i, 0)
        for i, (shard_id, idx) in enumerate([(0, 0), (1, 5), (0, 3), (2, 1)])
    ]

    # Run multiple times
    results = []
    for _ in range(3):
        result = op.process_many(samples.copy())
        results.append(result)

    # All results should be identical
    for i, result in enumerate(results[1:], start=1):
        assert result == results[0], (
            f"Run {i} produced different results than run 0. "
            f"Expected deterministic output but got: {result} vs {results[0]}"
        )


def test_prefetch_deduplication():
    """Test that prefetch doesn't download the same shard multiple times.

    If the same shard appears multiple times in the lookahead window,
    it should only be prefetched once per batch.
    """
    op = PrefetchOp(buffer_size=50)
    resolver = MockResolver(base_latency_ms=1.0)
    _setup_prefetch_with_mock(op, resolver, num_datasets=1, num_shards_per_dataset=10)

    # Create samples where the same shard appears many times
    samples = [
        ((0, 0, idx), 0, 0, idx, 0)  # All 100 samples from shard 0
        for idx in range(100)
    ]

    # Process first batch (50 samples, all from shard 0)
    result = op.process_many(samples[:50])
    assert len(result) == 50, f"Expected 50 samples in result, got {len(result)}"

    # Should have only resolved shard 0 ONCE, not 50 times
    stats = resolver.get_stats()
    assert stats["resolve_count"] == 1, (
        f"Expected 1 resolve call for 50 samples from same shard, got {stats['resolve_count']}"
    )
    resolved = resolver.get_unique_resolved_shards()
    assert resolved == {0}, (
        f"Expected only shard 0 to be resolved (deduplication), got {resolved}"
    )

    # Process second batch (another 50 samples from shard 0)
    result = op.process_many(samples[50:])
    assert len(result) == 50, f"Expected 50 samples in second batch, got {len(result)}"

    # Should have resolved shard 0 again (once per batch), total = 2
    stats = resolver.get_stats()
    assert stats["resolve_count"] == 2, (
        f"Expected 2 resolve calls total (one per batch), got {stats['resolve_count']}"
    )


def test_prefetch_with_real_resolver_mock():
    """Integration test: PrefetchOp with MockResolver simulating realistic latency.

    This test demonstrates the full prefetch workflow and verifies that
    PrefetchOp correctly triggers resolver.resolve() for all unique shards.
    """
    resolver = MockResolver(base_latency_ms=5.0)  # 5ms simulated download time
    op = PrefetchOp(buffer_size=100)
    _setup_prefetch_with_mock(op, resolver, num_datasets=1, num_shards_per_dataset=20)

    # Simulate 10 shards, each appearing multiple times in sequence
    num_shards = 10
    samples_per_shard = 20

    # Create samples: sequential access pattern
    samples = []
    for shard_id in range(num_shards):
        for sample_idx in range(samples_per_shard):
            sample_id = (0, shard_id, sample_idx)
            engine_sample = (sample_id, 0, 0, len(samples), 0)
            samples.append(engine_sample)

    # Process all samples through PrefetchOp
    start_time = time.time()
    result = op.process_many(samples)
    elapsed = time.time() - start_time

    # Verify all samples passed through unchanged
    assert result == samples, (
        f"Prefetch should pass samples through unchanged, but got different results"
    )

    # Verify prefetch resolved all 10 unique shards
    stats = resolver.get_stats()
    assert stats["resolve_count"] == num_shards, (
        f"Expected {num_shards} resolve calls, got {stats['resolve_count']}"
    )
    assert stats["unique_shards"] == num_shards

    # Verify timing: should have taken at least 10 * 5ms = 50ms for downloads
    # (This is a sanity check that the mock latency is being applied)
    expected_min_time = num_shards * 0.005  # 5ms per shard
    assert elapsed >= expected_min_time * 0.5, (  # Allow some slack
        f"Expected at least {expected_min_time * 0.5:.3f}s, took {elapsed:.3f}s"
    )


def test_prefetch_setup_with_inmem_datasets(mock_datasets):
    """Test prefetch operator setup with in-memory datasets.

    In-memory datasets should be skipped by prefetch (no files to download).
    """
    from zephon.io.options import StoreOptions

    op = PrefetchOp(buffer_size=256)

    # Context with in-memory datasets
    ctx = OpContext(
        {
            "datasets_by_id": mock_datasets,
            "io_options": StoreOptions(),
        }
    )

    # Setup should succeed even with in-memory datasets
    # (they're just skipped for prefetch)
    op.setup(ctx, stage_index=0, stage_name="prefetch", op_index=0, collect_stats=False)

    # Should have no locators for in-memory datasets
    assert len(op._locators) == 0, (
        f"In-memory datasets should not create locators (nothing to prefetch), "
        f"but found {len(op._locators)} locators"
    )


if __name__ == "__main__":
    # Run stress test directly for profiling
    print("Running prefetch stress test...")
    test_prefetch_stress_many_shards(num_shards=100, buffer_size=1024)
    print("Stress test completed successfully!")
