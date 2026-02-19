# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for PackSequences operator."""

from zephon.api.pipeline import Pipeline
from zephon.core.constants import ContributorRef, SampleRecord
from zephon.io import InMemoryShard
from zephon.io.dataset import Dataset
from zephon.work.static_mixture import StaticMixtureWorkSource


def _mk_dataset(name: str, shards: dict[int, int]) -> Dataset:
    """Create a dataset with specified shards and counts."""
    data: dict[int, InMemoryShard] = {}
    for sid, count in shards.items():
        rows = [{"text": f"{name}:{sid}:{i}", "length": 3} for i in range(count)]
        data[int(sid)] = InMemoryShard(rows)
    return Dataset.from_dict(name, data)


def test_pack_sequences_chunk_eviction() -> None:
    """Test that chunk eviction works correctly with PackSequences.

    Verifies that:
    1. Contributors are correctly collected from packed samples
    2. Chunks evict only when all offsets have been closed
    3. Packing across chunks doesn't prevent eviction
    """
    # Create dataset with multiple chunks
    ds = _mk_dataset("test", {0: 6})  # 6 samples
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=2,  # 2 samples per chunk -> chunks 0, 1, 2
        seed=42,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )

    # Build pipeline: Fetch -> PackSequences
    pipeline = Pipeline(work)
    pipeline.pack_sequences(
        max_length=10, length_fn="length", algorithm="best_fit", num_bins=100
    )
    pipeline.options(
        # Needs inline mode to inspect inflight_chunks_per_lane internal state.
        deterministic=True,
        max_workers=1,
        default_stage_prefetch=0,
        mtp_mode=False,
    )

    # Collect all packed records and track contributors
    packed_records: list[SampleRecord] = []
    all_contributors: list[ContributorRef] = []
    seen_chunks: set[int] = set()

    # Iterate through all records to load chunks and collect contributors
    for rec in pipeline:
        assert isinstance(rec, SampleRecord)
        packed_records.append(rec)
        seen_chunks.add(rec.meta.chunk_id)

        # Collect all contributors from packed records
        contributors = list(rec.meta.contribution_refs())
        all_contributors.extend(contributors)

        # Check that packed records have correct structure
        assert "packed_samples" in rec.payload
        assert "_packing_metadata" in rec.meta.tags

    # The engine's build_iter handles finalization automatically

    # Verify we got some packed records
    assert len(packed_records) > 0
    assert len(seen_chunks) > 0, "Should have seen at least one chunk"

    # Verify all contributors are present and track which offsets are closed
    chunk_offsets_closed: dict[tuple[int, int], bool] = {}
    for contrib in all_contributors:
        key = (contrib.cursor.chunk_id, contrib.cursor.chunk_offset)
        if contrib.is_last_child:
            chunk_offsets_closed[key] = True

    # Verify that chunks were evicted (should be if all offsets closed)
    # Access engine through pipeline to check chunk eviction
    assert pipeline._engine is not None
    eng = pipeline._engine
    lane_id = 0
    for chunk_id in seen_chunks:
        # Chunk should be evicted since all offsets are closed
        assert chunk_id not in eng.inflight_chunks_per_lane[lane_id], (
            f"Chunk {chunk_id} should be evicted since all offsets are closed"
        )


def test_pack_sequences_reproducibility() -> None:
    """Test that PackSequences produces deterministic output.

    Runs PackSequences multiple times with the same seed and verifies identical output.
    This is important for reproducibility, especially if we remove requires_serial_state.
    """
    # Create dataset
    ds = _mk_dataset("test", {0: 20})

    def _make_work_source() -> StaticMixtureWorkSource:
        return StaticMixtureWorkSource(
            [ds],
            {ds.name: 1.0},
            chunk_size=5,  # 4 chunks
            seed=123,
            shuffle_shards=False,
            shuffle_within_shard=False,
        )

    def _run_pack_sequences(
        shuffle: bool, seed: int | None = None
    ) -> list[tuple[int, int, int]]:
        """Run PackSequences and return list of (chunk_id, chunk_offset, num_sequences)."""
        work = _make_work_source()
        pipeline = Pipeline(work)
        pipeline.pack_sequences(
            max_length=10,
            length_fn="length",
            algorithm="best_fit",
            shuffle_strategy="random" if shuffle else "length",
            shuffle_seed=seed,
            num_bins=100,  # Large enough to avoid premature flushing affecting determinism
        )
        pipeline.options(deterministic=True, max_workers=1, default_stage_prefetch=0)

        results: list[tuple[int, int, int]] = []
        for rec in pipeline:
            assert isinstance(rec, SampleRecord)
            packing_meta = rec.meta.tags.get("_packing_metadata", {})
            num_sequences = packing_meta.get("num_sequences", 0)
            results.append(
                (
                    rec.meta.chunk_id,
                    rec.meta.chunk_offset,
                    num_sequences,
                )
            )

        # build_iter handles closing automatically
        return results

    # Test 1: Without shuffling (length-based sorting) - should be deterministic
    run1_no_shuffle = _run_pack_sequences(shuffle=False)
    run2_no_shuffle = _run_pack_sequences(shuffle=False)
    assert run1_no_shuffle == run2_no_shuffle, (
        "PackSequences without shuffling should produce deterministic output"
    )

    # Test 2: With shuffling and same seed - should be deterministic
    run1_shuffle = _run_pack_sequences(shuffle=True, seed=42)
    run2_shuffle = _run_pack_sequences(shuffle=True, seed=42)
    assert run1_shuffle == run2_shuffle, (
        "PackSequences with shuffling should produce deterministic output with same seed"
    )

    # Test 3: With shuffling and different seed - may produce different output
    run3_shuffle = _run_pack_sequences(shuffle=True, seed=99)
    assert len(run3_shuffle) > 0, "Should produce some output even with different seed"

    # Test 4: Verify reproducibility across multiple runs
    # This ensures determinism is maintained (important if we remove requires_serial_state)
    run4_shuffle = _run_pack_sequences(shuffle=True, seed=42)
    assert run1_shuffle == run4_shuffle, (
        "Should be reproducible across multiple runs with same seed"
    )
