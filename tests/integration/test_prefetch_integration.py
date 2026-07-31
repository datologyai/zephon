# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for prefetch operator cache behavior.

These tests verify prefetch/cache interactions with full pipelines,
including cache eviction, backpressure, and pathological access patterns.
"""

import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from zephon import Pipeline
from zephon._internal.observability.collector import PipelineCollector
from zephon._internal.ops.fetch import FetchOp
from zephon._internal.ops.prefetch import PrefetchOp
from zephon.io.dataset import Dataset
from zephon.observability import (
    ExecutionTrackingMode,
    FetchTimingTotals,
    PrefetchTimingTotals,
)
from zephon.work import MixtureSpec, StaticMixtureWorkSource


def _fetch_stats(pipe: Pipeline) -> tuple[FetchTimingTotals, set[tuple[int, int]]]:
    summary = pipe.fetch_timing_snapshot()
    assert summary is not None, "requires execution_tracking and a live engine"
    totals = FetchTimingTotals()
    shard_keys: set[tuple[int, int]] = set()
    for stage in summary.iter_stages():
        totals.merge(stage.totals)
        shard_keys.update(stage.shard_totals)
    return totals, shard_keys


def _prefetch_totals(pipe: Pipeline) -> PrefetchTimingTotals:
    summary = pipe.prefetch_timing_snapshot()
    assert summary is not None, "requires execution_tracking and a live engine"
    totals = PrefetchTimingTotals()
    for stage_totals in summary.stages.values():
        totals.merge(stage_totals)
    return totals


@pytest.mark.integration
def test_dataset_structure():
    """Validate the test dataset structure - shard sizes and sample counts.

    This test ensures the jsonl_prefetch_demo dataset has the expected structure
    that other tests depend on.
    """
    dataset_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "data"
        / "jsonl_prefetch_demo"
    )
    assert dataset_path.exists(), f"Dataset not found: {dataset_path}"

    # Check that we have exactly 4 shards
    shard_files = sorted(dataset_path.glob("shard*.jsonl"))
    assert len(shard_files) == 4, f"Expected 4 shards, found {len(shard_files)}"

    # Check each shard has 20 samples and ~870 bytes
    for i, shard_file in enumerate(shard_files):
        # Check file size
        file_size = shard_file.stat().st_size
        assert file_size == 870, f"Shard {i} expected 870 bytes, got {file_size}"

        # Check sample count
        with open(shard_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 20, f"Shard {i} expected 20 samples, got {len(lines)}"

    print(f"✓ Dataset validated: 4 shards × 20 samples × 870 bytes")


@pytest.mark.integration
def test_prefetch_with_sufficient_cache(tmp_path):
    """Test prefetch with sufficient cache - verify 100% cache hit rate.

    Setup:
    - 4 shards, each 870 bytes, 20 samples each, total=3480 bytes
    - Cache sized for all shards (4000 bytes > 3480 bytes needed)
    - Sequential access: s0, s1, s2, s3
    - Prefetch enabled with sufficient lookahead

    Expected behavior:
    - Prefetch scans buffer and identifies all 4 shards within prefetch_distance
    - Prefetch downloads all shards and BLOCKS until complete
    - Only after downloads finish, samples pass to fetch
    - Fetch sees 100% cache hits (all shards pre-cached by prefetch)

    This demonstrates blocking prefetch effectiveness: with sufficient cache space,
    fetch sees guaranteed cache hits because prefetch completed downloads first.
    """
    # Use the consolidated prefetch demo dataset
    dataset_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "data"
        / "jsonl_prefetch_demo"
    )
    assert dataset_path.exists(), f"Dataset not found: {dataset_path}"

    # Create cache directory
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    # Load the dataset
    dataset = Dataset.from_path("jsonl_prefetch_demo", str(dataset_path))

    # Create work source that reads shards sequentially
    # Each shard has 20 samples, 4 shards = 80 total samples
    work_source = StaticMixtureWorkSource(
        [dataset],
        mixture=MixtureSpec({"jsonl_prefetch_demo": 1.0}),
        chunk_size=80,  # One chunk with all samples
        seed=42,
        shuffle_shards=False,  # Sequential shard access
    )

    # Build pipeline with prefetch
    pipe = (
        Pipeline(work_source)
        .prefetch(buffer_size=80)  # Use default parallelism=4
        .fetch(parallelism=1)  # Single-threaded fetch for determinism
        .decode_text()
    )

    # Configure cache with enough space for all shards
    # 4 shards × 870 bytes = 3480 bytes total
    # Use 4000 bytes to fit all shards comfortably
    pipe = pipe.options(
        io_options={
            "cache": {
                "enabled": True,
                "root": str(cache_root),
                "limit_bytes": 4000,  # 4000 bytes - fits all 4 shards (3480 bytes)
                "min_slack_bytes": 0,  # No slack for testing
                "max_slack_bytes": 0,
            }
        },
        execution_tracking=ExecutionTrackingMode.NODES,
        # Snapshot access requires an in-process engine.
        mtp_mode=False,
    )

    pipe = pipe.batch(microbatch_size=80, drop_last=False)

    # Run the pipeline
    sample_count = 0
    for batch in pipe:
        sample_count += len(batch)

    assert sample_count == 80, f"Expected 80 samples, got {sample_count}"

    # Aggregate fetch statistics from the public timing snapshots
    fetch_totals, unique_shards = _fetch_stats(pipe)
    prefetch_totals = _prefetch_totals(pipe)
    print(
        f"DEBUG: Cache hits: {fetch_totals.cache_hits}/{fetch_totals.groups} fetch "
        f"groups, {len(unique_shards)} unique shards"
    )

    assert fetch_totals.samples == 80, (
        f"Expected 80 samples fetched, got {fetch_totals.samples}"
    )

    # Check prefetch statistics
    assert prefetch_totals.prefetch_requests == 4, (
        f"Expected 4 prefetch requests, got {prefetch_totals.prefetch_requests}"
    )

    # With blocking prefetch and sufficient cache, expect 100% cache hit rate
    # Prefetch downloads ALL shards within prefetch_distance BEFORE returning samples to fetch
    # Count unique shards accessed (a shard may appear in multiple fetch groups/batches)
    assert len(unique_shards) == 4, (
        f"Expected 4 unique shards accessed, got {len(unique_shards)}"
    )
    # Every fetch group should be a cache hit (prefetch downloaded everything)
    assert fetch_totals.cache_misses == 0, (
        f"Expected 0 cache misses with blocking prefetch and sufficient cache, "
        f"got {fetch_totals.cache_misses}"
    )
    assert fetch_totals.cache_hits == fetch_totals.groups, (
        f"Expected all {fetch_totals.groups} fetch groups to be cache hits, "
        f"got {fetch_totals.cache_hits}"
    )


@pytest.mark.integration
def test_sequential_shard_prefetch_insufficient_cache(tmp_path):
    """Test sequential shard reads with insufficient cache - verify deterministic cache eviction.

    Setup:
    - 4 shards: s0-s3, each 870 bytes, 20 samples each, total=3480 bytes
    - Cache sized for only 2 shards (2000 bytes fits 2×870=1740, not all 3480)
    - Mixed access via block shuffling (deterministic with seed=42)
    - Prefetch with buffer_size=80 (sees all 4 shards at once)
    - Prefetch with max_concurrent=1 (sequential downloads for deterministic eviction order)
    - Fetch with parallelism=1 (deterministic access order)

    Expected behavior (deterministic with max_concurrent=1):
    1. Prefetch downloads all 4 shards sequentially (buffer sees all 80 samples):
       - Downloads in order they appear in block-shuffled buffer
       - Cache can only hold 2 shards (2000 bytes)
       - Sequential downloads cause LRU evictions: earlier shards evicted
       - After all downloads: only last 2 downloaded shards remain in cache
    2. Fetch accesses shards in block-shuffled order (1, 0, 2, 3):
       - First access to each shard: MISS (shard not in cache or evicted)
       - Shard 3 split across batches: second access is HIT (reuse)
    3. Result: 4 cache misses (one per unique shard), 1 hit (s3 reuse across batches)

    Key insight: With insufficient cache (2000 bytes < 3480 bytes total) and
    sequential prefetch downloads (max_concurrent=1), we get deterministic cache
    eviction behavior that reliably demonstrates cache insufficiency.
    """
    # Use the consolidated dataset (will only access shards 0 and 1)
    dataset_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "data"
        / "jsonl_prefetch_demo"
    )
    assert dataset_path.exists(), f"Dataset not found: {dataset_path}"

    # Create cache directory
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    # Load the dataset
    dataset = Dataset.from_path("jsonl_prefetch_demo", str(dataset_path))

    # Create work source with block-level shuffling to create cross-shard mixing
    work_source = StaticMixtureWorkSource(
        [dataset],
        mixture=MixtureSpec({"jsonl_prefetch_demo": 1.0}),
        chunk_size=80,  # All 80 samples (4 shards × 20 samples)
        seed=42,
        shuffle_shards=False,
        shuffle_within_shard=False,
        shuffle_block_size=30,  # Large block to mix shards together
    )

    # Build pipeline with prefetch
    # CRITICAL:
    # - buffer_size=80 ensures all 4 shards are seen at once in the prefetch buffer
    # - max_concurrent=1 forces sequential downloads for deterministic cache eviction order
    # - This forces shards to compete for insufficient cache (2000 bytes fits only 2 shards)
    pipe = (
        Pipeline(work_source)
        .prefetch(buffer_size=80, parallelism=1)
        .fetch(parallelism=1)  # Single-threaded fetch for determinism
        .decode_text()
    )

    # Configure cache with space for ~2 shards (insufficient for all 4)
    # 4 shards × 870 bytes = 3480 bytes total
    # Use 2000 bytes cache - fits only 2 shards (2×870=1740), not all 4
    pipe = pipe.options(
        io_options={
            "cache": {
                "enabled": True,
                "root": str(cache_root),
                "limit_bytes": 2000,  # 2000 bytes - fits 2 shards, not all 4
                "min_slack_bytes": 0,  # No slack for testing
                "max_slack_bytes": 0,
            }
        },
        execution_tracking=ExecutionTrackingMode.NODES,
        # Snapshot access requires an in-process engine.
        mtp_mode=False,
    )

    # Use small batches to create multiple process_many calls
    # This forces multiple shard open operations, triggering cache behavior
    pipe = pipe.batch(microbatch_size=10, drop_last=False)

    # Run the pipeline
    sample_count = 0
    for batch in pipe:
        sample_count += len(batch)

    assert sample_count == 80, f"Expected 80 samples, got {sample_count}"

    # With blocking prefetch and severely insufficient cache, expect constant thrashing
    fetch_totals, unique_shards = _fetch_stats(pipe)
    total_accesses = fetch_totals.cache_hits + fetch_totals.cache_misses

    print(
        f"DEBUG: Cache performance - hits: {fetch_totals.cache_hits}/{total_accesses} "
        f"accesses, {len(unique_shards)} unique shards (insufficient cache)"
    )

    # With insufficient cache (2000 bytes for 2 shards, but 4 shards total):
    # - Prefetch downloads all 4 shards sequentially (buffer_size=80, max_concurrent=1)
    # - Cache can only hold 2 of 4, so LRU eviction happens during prefetch
    # - After prefetch completes, only last 2 downloaded shards remain cached
    # - Fetch accesses all 4 shards in block-shuffled order (deterministic)
    # - First access to each unique shard causes a cache miss: 4 misses total
    # - Shard 3 appears in 2 fetch batches: second access is a hit (reuse)
    assert len(unique_shards) == 4, (
        f"Expected 4 unique shards accessed, got {len(unique_shards)}"
    )
    assert fetch_totals.cache_misses == 4, (
        f"Expected exactly 4 cache misses (one per unique shard), "
        f"got {fetch_totals.cache_misses}"
    )
    assert fetch_totals.cache_hits == 1, (
        f"Expected exactly 1 cache hit (shard 3 reuse across batches), "
        f"got {fetch_totals.cache_hits}"
    )


@pytest.mark.integration
def test_alternating_shard_prefetch_insufficient_cache(tmp_path):
    """Test alternating shard access with insufficient cache - pathological case with 100% miss rate.

    Setup:
    - 2 shards with 20 samples each (s0: 870 bytes, s1: 870 bytes, total: 1740 bytes)
    - Cache sized for only 1 shard (870 bytes < 1740 bytes total needed)
    - Perfect alternating access pattern: s0[0], s1[0], s0[1], s1[1], ..., s0[19], s1[19]
    - Coordinated 2:2 pattern: prefetch, prefetch, fetch, fetch (using mock.patch on process_many)
    - Single-threaded fetch (parallelism=1) to prevent optimistic file handle reuse
    - Microbatch size=1 to create 40 separate fetch operations

    Expected behavior with 2:2 coordination pattern:
    - Prefetch 0, Prefetch 1: Downloads both shards, second evicts first → cache: [shard 1]
    - Fetch 0: Accesses shard 0 (MISS - was evicted), downloads it, evicts shard 1 → cache: [shard 0]
    - Fetch 1: Accesses shard 1 (MISS - was just evicted), downloads it, evicts shard 0 → cache: [shard 1]
    - Pattern repeats with 100% misses (40/40 misses, 0/40 hits)

    Key insight: The 2:2 coordination pattern creates MAXIMUM cache thrashing. Every fetch
    sees an evicted shard because the two prefetches download both shards (evicting the first),
    then the two fetches alternate access, causing mutual eviction on every operation.
    """
    # Use the consolidated dataset (AlternatingWorkSource only uses shards 0 and 1)
    dataset_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "data"
        / "jsonl_prefetch_demo"
    )
    assert dataset_path.exists(), f"Dataset not found: {dataset_path}"

    # Create cache directory
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    # Load the dataset
    dataset = Dataset.from_path("jsonl_prefetch_demo", str(dataset_path))

    # Create a custom work source that emits samples in PERFECT alternation
    # s0[0], s1[0], s0[1], s1[1], ..., s0[19], s1[19]
    from zephon.work.base import WorkChunk, WorkSource

    class AlternatingWorkSource(WorkSource):
        """Custom work source that alternates samples between two shards."""

        def __init__(self, dataset: Dataset, chunk_size: int = 40):
            super().__init__()
            self._dataset = dataset
            self._chunk_size = chunk_size
            self._emitted = False
            self._datasets_by_id = {0: dataset}

        @property
        def datasets_by_id(self):
            return self._datasets_by_id

        def component_ids(self) -> dict[str, int]:
            return {"jsonl_prefetch_demo": 0}

        def next_chunk(self) -> WorkChunk | None:
            if self._emitted:
                return None

            # Create perfectly alternating sequence: s0[0], s1[0], s0[1], s1[1], ...
            samples = []
            for i in range(20):  # 20 samples per shard
                samples.append((0, 0, i))  # (dataset_id=0, shard_id=0, sample_idx=i)
                samples.append((0, 1, i))  # (dataset_id=0, shard_id=1, sample_idx=i)

            self._emitted = True
            return WorkChunk(components={"jsonl_prefetch_demo": samples}, seed=42)

        def __len__(self) -> int:
            return 40 if not self._emitted else 0

        def chunk_size_hint(self) -> int | None:
            return self._chunk_size

        def supports_indexing(self) -> bool:
            return False

        def sample_id_at(self, index: int):
            raise NotImplementedError()

    work_source = AlternatingWorkSource(dataset, chunk_size=40)

    # Build pipeline with prefetch
    # CRITICAL: buffer_size=1 forces prefetch to process 1 sample at a time (alternation)
    # CRITICAL: max_batch=1 forces fetch to process 1 sample at a time (40 separate calls)
    pipe = (
        Pipeline(work_source)
        .prefetch(buffer_size=1, parallelism=1)
        .fetch(parallelism=1, max_batch=1)
        .decode_text()
    )

    # Configure cache with space for ONLY 1 shard (pathological scenario)
    # Both shards are now 870 bytes each, cache is 900 bytes
    pipe = pipe.options(
        io_options={
            "cache": {
                "enabled": True,
                "root": str(cache_root),
                "limit_bytes": 900,  # 900 bytes - fits one 870-byte shard, not both (1740 bytes)
                "min_slack_bytes": 0,  # No slack for testing
                "max_slack_bytes": 0,
            }
        },
        execution_tracking=ExecutionTrackingMode.NODES,
        # Metrics patches live in this process; need inline mode.
        mtp_mode=False,
    )

    # CRITICAL: microbatch_size=1 creates 40 separate batches (one per sample)
    # Combined with alternating pattern, this demonstrates maximum cache thrashing
    pipe = pipe.batch(microbatch_size=1, drop_last=False)

    # Coordination: Use threading events to enforce strict 2:2 alternation between prefetch and fetch
    # Pattern: prefetch, prefetch, fetch, fetch, prefetch, prefetch, fetch, fetch, ...
    # This forces MAXIMUM cache thrashing by ensuring every fetch sees an evicted shard
    prefetch_can_proceed = threading.Event()
    fetch_can_proceed = threading.Event()
    coordination_lock = threading.Lock()
    prefetch_count = 0
    fetch_count = 0

    # Start with prefetch allowed to proceed
    prefetch_can_proceed.set()

    # Store original methods
    original_prefetch_process_many = PrefetchOp.process_many
    original_fetch_process_many = FetchOp.process_many

    def coordinated_prefetch_process_many(self, elems):
        nonlocal prefetch_count
        # Wait for permission to proceed
        prefetch_can_proceed.wait()
        # Call original
        print(f" {threading.current_thread().name} prefetch.process_many start")
        result = original_prefetch_process_many(self, elems)
        print(f" {threading.current_thread().name} prefetch.process_many end")
        # After prefetch: block prefetch, allow fetch
        with coordination_lock:
            prefetch_count += 1
            # Only let fetch go after two prefetches complete
            if prefetch_count % 2 == 0:
                prefetch_can_proceed.clear()
                fetch_can_proceed.set()
        return result

    def coordinated_fetch_process_many(self, elems):
        nonlocal fetch_count
        # Wait for permission to proceed
        fetch_can_proceed.wait()
        # Call original
        print(f" {threading.current_thread().name} fetch.process_many start")
        result = original_fetch_process_many(self, elems)
        print(f" {threading.current_thread().name} fetch.process_many end")
        # After fetch: block fetch, allow prefetch
        with coordination_lock:
            fetch_count += 1
            # Only let prefetch go after two fetches complete
            if fetch_count % 2 == 0:
                fetch_can_proceed.clear()
                prefetch_can_proceed.set()
        return result

    # Run the pipeline with patched coordination
    with patch.object(PrefetchOp, "process_many", coordinated_prefetch_process_many):
        with patch.object(FetchOp, "process_many", coordinated_fetch_process_many):
            sample_count = 0
            for batch in pipe:
                sample_count += len(batch)

    assert sample_count == 40, f"Expected 40 samples (20 per shard), got {sample_count}"

    # Analyze cache performance
    fetch_totals, shard_ids_accessed = _fetch_stats(pipe)
    total_accesses = fetch_totals.cache_hits + fetch_totals.cache_misses

    print(
        f"\nDEBUG: Pathological alternating case - hits: "
        f"{fetch_totals.cache_hits}/{total_accesses} accesses"
    )
    print(f"DEBUG: Number of fetch groups: {fetch_totals.groups}")

    # With pathological scenario + 1:1 alternation via mock.patch coordination:
    # Pattern: prefetch → fetch → prefetch → fetch → ...
    # - Prefetch s0[0] → downloads shard 0, blocks
    # - Fetch s0[0] → accesses shard 0 (HIT from prefetch), blocks
    # - Prefetch s1[0] → downloads shard 1, evicts s0, blocks
    # - Fetch s1[0] → accesses shard 1 (HIT from prefetch), blocks
    # - Prefetch s0[1] → downloads shard 0, evicts s1, blocks
    # - Fetch s0[1] → accesses shard 0 (MISS - s0 was just downloaded by prefetch before this fetch, but may be evicted by timing)
    # - Tight coordination creates near-maximum thrashing: 95-100% miss rate
    # - Result: 40 accesses (forced by alternation), 2 hits (first two samples), 38 misses (95% miss rate)

    miss_rate = fetch_totals.cache_misses / total_accesses if total_accesses > 0 else 0
    print(f"DEBUG: Miss rate: {miss_rate * 100:.1f}%")

    # Assert exact expected behavior for pathological case with 1:1 coordination
    assert len(shard_ids_accessed) == 2, (
        f"Expected both shards to be accessed, got {shard_ids_accessed}"
    )

    # Exact expectations with 2:2 coordination pattern (prefetch, prefetch, fetch, fetch):
    # - 40 total accesses (one per sample, forced by parallelism=1 on fetch)
    # - 0 cache hits (100% miss rate)
    # - 40 cache misses
    #
    # Why 100% miss rate?
    # 1. Prefetch 0, Prefetch 1: Downloads both shards, second evicts first → cache: [shard 1]
    # 2. Fetch 0: Accesses shard 0 (MISS - was evicted), downloads it, evicts shard 1 → cache: [shard 0]
    # 3. Fetch 1: Accesses shard 1 (MISS - was just evicted), downloads it, evicts shard 0 → cache: [shard 1]
    # 4. Pattern repeats with 100% misses throughout
    #
    # This is maximum cache thrashing - the 2:2 pattern ensures every fetch sees an evicted shard
    assert fetch_totals.groups == 40, (
        f"Expected exactly 40 fetch groups (one per sample), got {fetch_totals.groups}"
    )
    assert total_accesses == 40, (
        f"Expected exactly 40 shard accesses (forced by coordination), got {total_accesses}"
    )
    assert fetch_totals.cache_hits == 0, (
        f"Expected 0 cache hits (100% miss rate with 2:2 pattern), "
        f"got {fetch_totals.cache_hits}"
    )
    assert fetch_totals.cache_misses == 40, (
        f"Expected 40 cache misses (pathological case with 2:2 pattern), "
        f"got {fetch_totals.cache_misses}"
    )
    assert miss_rate == 1.0, (
        f"Expected 100% miss rate (pathological case with 2:2 pattern), got {miss_rate * 100:.1f}%"
    )


@pytest.mark.integration
def test_backpressure_callback_simple(tmp_path):
    """Test backpressure callback mechanism.

    Verifies that on_backpressure_event fires when the pump thread can't
    forward results downstream (because FetchOp is blocked).

    The test works by:
    1. Patching FetchOp to block until backpressure is detected
    2. PrefetchOp processes samples and its pump tries to forward to FetchOp
    3. FetchOp.input_queue fills (capacity=1) since FetchOp is blocked
    4. PrefetchOp pump blocks in _put_into_queue → callback fires
    5. Callback unblocks FetchOp → pipeline completes
    """
    dataset_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "data"
        / "jsonl_prefetch_demo"
    )
    assert dataset_path.exists(), f"Dataset not found: {dataset_path}"

    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    dataset = Dataset.from_path("jsonl_prefetch_demo", str(dataset_path))

    work_source = StaticMixtureWorkSource(
        [dataset],
        mixture=MixtureSpec({"jsonl_prefetch_demo": 1.0}),
        chunk_size=80,
        seed=42,
    )

    from zephon.observability.stats import BackpressureDelta

    # Track backpressure events and coordinate with FetchOp
    backpressure_occurred = threading.Event()
    backpressure_events: list[BackpressureDelta] = []
    callback_lock = threading.Lock()

    pipe = Pipeline(work_source)
    pipe = pipe.options(
        io_options={
            "cache": {
                "enabled": True,
                "root": str(cache_root),
                "limit_bytes": 10000000,
            }
        },
        op_queue_capacity=4,  # Queue capacity to trigger backpressure
        default_stage_prefetch=4,
        max_workers=20,
        execution_tracking=ExecutionTrackingMode.NODES,
        # Metrics patches live in this process; need inline mode.
        mtp_mode=False,
    )
    pipe = pipe.prefetch(buffer_size=1, parallelism=4, placement="local")
    pipe = pipe.fetch(max_batch=1, parallelism=1)
    pipe = pipe.decode_text()
    pipe = pipe.batch(microbatch_size=10, drop_last=False)

    # Patch collector to intercept backpressure events
    original_record_backpressure = PipelineCollector.record_backpressure

    def patched_record_backpressure(self, delta: BackpressureDelta):
        with callback_lock:
            backpressure_events.append(delta)
            if len(backpressure_events) == 1:
                print(
                    f"🔴 BACKPRESSURE DETECTED! stage={delta.stage_index}, count=1",
                    file=sys.stderr,
                )
                backpressure_occurred.set()
        return original_record_backpressure(self, delta)

    # Patch FetchOp to block until backpressure is detected
    original_fetch_process_many = FetchOp.process_many

    def blocked_fetch_process_many(self, elems):
        # Block FetchOp until backpressure callback sets the event
        print(f"  FetchOp: Waiting for backpressure event...", file=sys.stderr)
        success = backpressure_occurred.wait()
        if success:
            print(
                f"  FetchOp: Backpressure event received! Proceeding...",
                file=sys.stderr,
            )
        result = original_fetch_process_many(self, elems)
        return result

    # Run pipeline - FetchOp blocks until backpressure callback fires
    total_samples = 0

    with (
        patch.object(
            PipelineCollector, "record_backpressure", patched_record_backpressure
        ),
        patch.object(FetchOp, "process_many", blocked_fetch_process_many),
    ):
        for batch in pipe:
            total_samples += len(batch)

    # Assertions
    assert total_samples == 80, f"Expected 80 samples, got {total_samples}"
    # At least 1 backpressure event (may get more due to retry loop timing)
    assert len(backpressure_events) >= 1, (
        f"Expected at least 1 backpressure event, got {len(backpressure_events)}"
    )

    # Verify backpressure event count
    first_event = backpressure_events[0]
    assert first_event.put_into_queue_backpressure_events == 1, (
        "Expected one backpressure event from _put_into_queue"
    )
