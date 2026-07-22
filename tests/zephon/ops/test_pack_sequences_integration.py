# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for PackSequences operator."""

from typing import Any, get_args

import pytest

from tests.zephon.ops.conftest import mk_dataset
from zephon.api.pipeline import Pipeline
from zephon.core.constants import ContributorRef, SampleRecord
from zephon.io import InMemoryShard
from zephon.io.dataset import Dataset
from zephon.ops import PackingAlgorithm
from zephon.work.static_mixture import StaticMixtureWorkSource

_PACKING_ALGORITHMS: tuple[PackingAlgorithm, ...] = get_args(PackingAlgorithm)


def _mk_varlen_dataset(name: str, lengths: list[int]) -> Dataset:
    """Create a single-shard dataset with per-sample lengths."""
    rows = [
        {"text": f"{name}:0:{i}", "length": lengths[i]} for i in range(len(lengths))
    ]
    return Dataset.from_dict(name, {0: InMemoryShard(rows)})


@pytest.mark.parametrize(
    "runner_kind",
    ["inline", "threads", "process"],
    ids=["inline", "threads", "process"],
)
def test_pack_sequences_chunk_eviction(runner_kind: str) -> None:
    """Test that chunk eviction works correctly with PackSequences.

    Verifies that:
    1. Contributors are correctly collected from packed samples
    2. Chunks evict only when all offsets have been closed
    3. Packing across chunks doesn't prevent eviction
    """
    # Create dataset with multiple chunks
    ds = mk_dataset("test", {0: 6})  # 6 samples
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
        max_length=10, tokens_field="length", algorithm="best_fit", num_bins=100
    )
    pipeline.options(
        deterministic=True,
        max_workers=1,
        default_stage_prefetch=16,
        runner=runner_kind,
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


@pytest.mark.parametrize(
    "runner_kind",
    ["inline", "threads", "process"],
    ids=["inline", "threads", "process"],
)
def test_pack_sequences_reproducibility(runner_kind: str) -> None:
    """Test that PackSequences produces deterministic output.

    Runs PackSequences multiple times with the same seed and verifies identical output.
    This is important for reproducibility, especially if we remove requires_serial_state.
    """
    # Create dataset
    ds = mk_dataset("test", {0: 20})

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
            tokens_field="length",
            algorithm="best_fit",
            shuffle_strategy="random" if shuffle else "length",
            shuffle_seed=seed,
            num_bins=100,  # Large enough to avoid premature flushing affecting determinism
        )
        pipeline.options(
            deterministic=True,
            max_workers=1,
            default_stage_prefetch=16,
            runner=runner_kind,
        )

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


# ---------------------------------------------------------------------------
# Checkpoint / resume + eviction
# ---------------------------------------------------------------------------


def _make_pack_pipeline(
    ds: Dataset,
    chunk_size: int = 4,
    *,
    runner: str = "inline",
    flush_every_k_chunks: int | None = None,
) -> Pipeline:
    """Build a deterministic pipeline with pack_sequences."""
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=chunk_size,
        seed=42,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = Pipeline(work)
    pipeline.pack_sequences(
        max_length=10,
        tokens_field="length",
        algorithm="best_fit",
        num_bins=100,
    )
    opts: dict[str, Any] = dict(
        deterministic=True,
        max_workers=1,
        default_stage_prefetch=16,
        runner=runner,
    )
    if flush_every_k_chunks is not None:
        opts["flush_every_k_chunks"] = flush_every_k_chunks
    pipeline.options(**opts)
    return pipeline


@pytest.mark.parametrize(
    "runner_kind",
    ["inline", "threads", "process"],
    ids=["inline", "threads", "process"],
)
def test_pack_sequences_chunk_eviction_after_checkpoint(runner_kind: str) -> None:
    """Chunks must evict after checkpoint/restore with pack_sequences.

    Regression test for the issue where ReplayFilter drops the already-consumed
    prefix on resume but does not emit tombstones for their closing
    contributors.  Without tombstones the engine's per-offset bitmaps are never
    rebuilt for those offsets, so the oldest inflight chunk from the previous
    run can never complete → blocks eviction of all subsequent chunks.

    The test:
    1. Creates a pipeline with pack_sequences (preserves_cursor_order=False →
       general eviction path using per-offset bitmaps).
    2. Consumes some records, checkpoints.
    3. Restores on a fresh pipeline and fully drains it.
    4. Asserts that **no** inflight chunks remain (all evicted).
    """
    # 20 samples, chunk_size=4 → 5 chunks (cid 0..4), each with 4 offsets.
    # length=3 per sample, max_length=10 → ~3 samples per packed record.
    # We use enough data so that:
    #  - checkpoint captures several inflight chunks with partially-consumed offsets
    #  - the resume run has to process both replayed and new chunks
    ds = mk_dataset("ckpt", {0: 20})

    # -- baseline: full run without checkpoint, verify eviction works ----------
    baseline_pipe = _make_pack_pipeline(ds, runner=runner_kind)
    baseline_records: list[SampleRecord] = []
    for rec in baseline_pipe:
        assert isinstance(rec, SampleRecord)
        baseline_records.append(rec)

    assert baseline_pipe._engine is not None
    baseline_inflight = baseline_pipe._engine.inflight_chunks_per_lane.get(0, {})
    assert len(baseline_inflight) == 0, (
        f"Baseline (no checkpoint) should evict all chunks, "
        f"but {len(baseline_inflight)} remain: {list(baseline_inflight.keys())}"
    )
    n_baseline = len(baseline_records)
    assert n_baseline > 0, "Baseline must produce records"

    # -- run 1: consume only 1 record and checkpoint ---------------------------
    # Consuming just 1 packed record means we are mid-stream: some chunk offsets
    # have been consumed (via the packed record's contributors), but most chunks
    # still have incomplete bitmaps.  This maximises the number of inflight
    # chunks that will need tombstone-based bitmap rebuild on resume.
    pipe1 = _make_pack_pipeline(ds, runner=runner_kind)
    it = iter(pipe1)
    try:
        first_rec = next(it)
        assert isinstance(first_rec, SampleRecord)
        ckpt: dict[str, Any] = pipe1.checkpoint()
    finally:
        it.close()

    # Sanity: checkpoint contains inflight chunks
    inflight_ckpt = ckpt.get("inflight", {})
    assert inflight_ckpt, "Checkpoint should have inflight chunks"

    # -- run 2: restore and drain ----------------------------------------------
    pipe2 = _make_pack_pipeline(ds, runner=runner_kind)
    pipe2.restore(ckpt)

    suffix: list[SampleRecord] = []
    for rec in pipe2:
        assert isinstance(rec, SampleRecord)
        suffix.append(rec)

    # Sanity: we got records after resume
    assert len(suffix) > 0, "Should produce records after resume"

    # -- the critical assertion: all chunks must have evicted -------------------
    assert pipe2._engine is not None
    eng = pipe2._engine
    inflight_after = eng.inflight_chunks_per_lane.get(0, {})

    assert len(inflight_after) == 0, (
        f"After checkpoint/restore and full drain, all chunks should be "
        f"evicted, but {len(inflight_after)} chunks remain inflight: "
        f"{sorted(inflight_after.keys())}. "
        f"This indicates ReplayFilter dropped records without emitting "
        f"tombstones for their closing contributors."
    )


@pytest.mark.parametrize(
    "runner_kind",
    ["inline", "threads", "process"],
    ids=["inline", "threads", "process"],
)
def test_pack_sequences_cross_chunk_data_correctness_after_checkpoint(
    runner_kind: str,
) -> None:
    """Checkpoint/restore must not duplicate samples from cross-chunk packed records.

    Setup (12 samples, chunk_size=4, length=3, max_length=10 → 3 per pack):

        Baseline packing:
          P0 = [s0, s1, s2]          ← all from chunk 0
          P1 = [s3, s4, s5]          ← s3 from chunk 0, s4/s5 from chunk 1
          P2 = [s6, s7, s8]          ← s6/s7 from chunk 1, s8 from chunk 2
          P3 = [s9, s10, s11]        ← all from chunk 2

    After consuming P0 + P1 and checkpointing:
      - P1's contributors close chunk 0 offset 3 → chunk 0 fully done → evicted
      - Checkpoint inflight = {chunk 1, chunk 2}
      - Replay cursor = P1's cursor (chunk_id=0) → evicted chunk

    On restore:
      - _publish_replay_snapshot sees cursor.chunk_id=0 not in inflight
        → sets snapshot[lane]=None → ReplayFilter becomes a no-op
      - PackSequences re-packs only chunks 1+2 from scratch → different bins
      - All re-packed records pass through (no filtering)
      - Samples s4, s5 appear in BOTH the prefix (P1) and the suffix
    """
    ds = mk_dataset("xchunk", {0: 12})

    # -- baseline: full run without checkpoint ---------------------------------
    baseline_pipe = _make_pack_pipeline(ds, runner=runner_kind)
    baseline_texts: list[list[str]] = []
    for rec in baseline_pipe:
        assert isinstance(rec, SampleRecord)
        baseline_texts.append([s["text"] for s in rec.payload["packed_samples"]])

    n_baseline = len(baseline_texts)
    assert n_baseline == 4, f"Expected 4 packed records, got {n_baseline}"

    # -- run 1: consume 2 records → triggers cross-chunk eviction --------------
    pipe1 = _make_pack_pipeline(ds, runner=runner_kind)
    it = iter(pipe1)
    prefix_texts: list[list[str]] = []
    try:
        for _ in range(2):
            rec = next(it)
            assert isinstance(rec, SampleRecord)
            prefix_texts.append([s["text"] for s in rec.payload["packed_samples"]])
        ckpt: dict[str, Any] = pipe1.checkpoint()
    finally:
        it.close()

    # Sanity: cursor pinning keeps chunk 0 in inflight (the cursor c0:3
    # references it), which is exactly the fix — ReplayFilter stays enabled.
    inflight_ckpt = ckpt.get("inflight", {}).get(
        0, ckpt.get("inflight", {}).get("0", {})
    )
    inflight_cids = sorted(int(c) for c in inflight_ckpt.keys())
    assert 0 in inflight_cids, (
        f"Cursor pinning should keep chunk 0 in inflight (cursor references it), "
        f"but inflight contains: {inflight_cids}"
    )

    # -- run 2: restore and drain ----------------------------------------------
    pipe2 = _make_pack_pipeline(ds, runner=runner_kind)
    pipe2.restore(ckpt)

    suffix_texts: list[list[str]] = []
    for rec in pipe2:
        assert isinstance(rec, SampleRecord)
        suffix_texts.append([s["text"] for s in rec.payload["packed_samples"]])

    # -- data correctness: no duplicates, exact match with baseline ------------
    all_prefix_samples = [s for packed in prefix_texts for s in packed]
    all_suffix_samples = [s for packed in suffix_texts for s in packed]
    all_combined = all_prefix_samples + all_suffix_samples
    all_baseline = [s for packed in baseline_texts for s in packed]

    assert all_combined == all_baseline, (
        f"prefix + suffix should reconstruct the baseline sample stream.\n"
        f"  prefix samples:  {all_prefix_samples}\n"
        f"  suffix samples:  {all_suffix_samples}\n"
        f"  baseline:        {all_baseline}\n"
        f"  combined:        {all_combined}"
    )


# ---------------------------------------------------------------------------
# Cross-chunk packing: variable-length data (Layer 2 bug)
# ---------------------------------------------------------------------------


def _make_varlen_pack_pipeline(
    ds: Dataset,
    chunk_size: int = 4,
    max_length: int = 6,
    num_bins: int = 2,
    flush_strategy: str = "fullest",
    *,
    runner: str = "inline",
    flush_every_k_chunks: int | None = None,
) -> Pipeline:
    """Build a deterministic pipeline with variable-length packing."""
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=chunk_size,
        seed=42,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = Pipeline(work)
    pipeline.pack_sequences(
        max_length=max_length,
        tokens_field="length",
        algorithm="best_fit",
        num_bins=num_bins,
        flush_strategy=flush_strategy,
    )
    opts: dict[str, Any] = dict(
        deterministic=True,
        max_workers=1,
        default_stage_prefetch=16,
        runner=runner,
    )
    if flush_every_k_chunks is not None:
        opts["flush_every_k_chunks"] = flush_every_k_chunks
    pipeline.options(**opts)
    return pipeline


@pytest.mark.parametrize(
    "runner_kind",
    ["inline", "threads", "process"],
    ids=["inline", "threads", "process"],
)
def test_pack_sequences_varlen_cross_chunk_data_correctness_after_checkpoint(
    runner_kind: str,
) -> None:
    """Variable-length packing: atomic eviction prevents data corruption.

    Setup (12 samples, chunk_size=4, lengths=[5,5,5,5, 1,1,1,1, 5,5,5,5]):
      - max_length=6, num_bins=2, flush_strategy="fullest"
      - Chunk 0: large items (len=5), Chunk 1: small items (len=1), Chunk 2: large (len=5)

    Cross-chunk packing (P2=[s2,s4], P3=[s3,s5]) means chunk 0 and chunk 1
    items share bin state.  Atomic eviction (gated by flush sentinel epoch
    floor) prevents chunk 0 from being evicted while chunk 1 is still
    inflight, so on restore all three chunks are replayed → same accumulator
    state → same packing → no data corruption.
    """
    # Large-Small-Large pattern
    lengths = [5, 5, 5, 5, 1, 1, 1, 1, 5, 5, 5, 5]
    ds = _mk_varlen_dataset("vlxc", lengths)

    # -- baseline: full run without checkpoint ---------------------------------
    baseline_pipe = _make_varlen_pack_pipeline(ds, runner=runner_kind)
    baseline_texts: list[list[str]] = []
    for rec in baseline_pipe:
        assert isinstance(rec, SampleRecord)
        baseline_texts.append([s["text"] for s in rec.payload["packed_samples"]])

    n_baseline = len(baseline_texts)
    assert n_baseline > 0, "Baseline must produce records"

    # Verify non-monotonic cursors (the key precondition for this bug)
    baseline_pipe2 = _make_varlen_pack_pipeline(ds, runner=runner_kind)
    cursor_cids = []
    for rec in baseline_pipe2:
        assert isinstance(rec, SampleRecord)
        cursor_cids.append(rec.meta.cursor.chunk_id)
    is_monotone = all(
        cursor_cids[i] <= cursor_cids[i + 1] for i in range(len(cursor_cids) - 1)
    )
    assert not is_monotone, (
        f"Expected non-monotonic cursors for this config, got {cursor_cids}"
    )

    # -- run 1: consume 5 records → cursor IS in inflight but data corrupts ----
    consume_count = 5
    pipe1 = _make_varlen_pack_pipeline(ds, runner=runner_kind)
    it = iter(pipe1)
    prefix_texts: list[list[str]] = []
    try:
        for _ in range(consume_count):
            rec = next(it)
            assert isinstance(rec, SampleRecord)
            prefix_texts.append([s["text"] for s in rec.payload["packed_samples"]])
        ckpt: dict[str, Any] = pipe1.checkpoint()
    finally:
        it.close()

    # Verify all chunks remain inflight — atomic eviction keeps them because
    # the epoch floor hasn't been advanced past them yet (only 5 of 9 records
    # consumed, so the flush sentinel from end-of-stream was never reached).
    assert pipe1._engine is not None
    inflight_ckpt = ckpt.get("inflight", {}).get(
        0, ckpt.get("inflight", {}).get("0", {})
    )
    inflight_cids = sorted(int(c) for c in inflight_ckpt.keys())
    assert 0 in inflight_cids, (
        f"Chunk 0 must remain inflight (atomic eviction prevents premature "
        f"eviction of chunks whose data influenced accumulator state). "
        f"Inflight: {inflight_cids}"
    )

    # -- run 2: restore and drain ----------------------------------------------
    pipe2 = _make_varlen_pack_pipeline(ds, runner=runner_kind)
    pipe2.restore(ckpt)

    suffix_texts: list[list[str]] = []
    for rec in pipe2:
        assert isinstance(rec, SampleRecord)
        suffix_texts.append([s["text"] for s in rec.payload["packed_samples"]])

    # -- data correctness: no duplicates, exact match with baseline ------------
    all_prefix_samples = [s for packed in prefix_texts for s in packed]
    all_suffix_samples = [s for packed in suffix_texts for s in packed]
    all_combined = all_prefix_samples + all_suffix_samples
    all_baseline = [s for packed in baseline_texts for s in packed]

    # Check for duplicates specifically (the symptom of this bug)
    dupes = set(all_prefix_samples) & set(all_suffix_samples)
    assert not dupes, (
        f"Samples duplicated across prefix and suffix: {sorted(dupes)}\n"
        f"  prefix samples: {all_prefix_samples}\n"
        f"  suffix samples: {all_suffix_samples}"
    )

    assert all_combined == all_baseline, (
        f"prefix + suffix should reconstruct the baseline sample stream.\n"
        f"  prefix samples:  {all_prefix_samples}\n"
        f"  suffix samples:  {all_suffix_samples}\n"
        f"  baseline:        {all_baseline}\n"
        f"  combined:        {all_combined}"
    )


# ---------------------------------------------------------------------------
# Mid-stream eviction + data correctness with flush sentinels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "runner_kind",
    ["inline", "threads", "process"],
    ids=["inline", "threads", "process"],
)
def test_pack_sequences_varlen_mid_stream_eviction_after_checkpoint(
    runner_kind: str,
) -> None:
    """Mid-stream eviction + checkpoint/restore with variable-length packing.

    Verifies both:
    1. Chunks from completed epochs evict during iteration (inflight stays
       bounded rather than growing to total_chunks)
    2. Checkpoint/restore after eviction produces identical output to baseline

    Setup (32 samples, chunk_size=4, flush_every_k_chunks=2):
      - 8 chunks total, sentinels after every 2 chunks
      - Repeating Large-Small length pattern for cross-chunk packing

    Note on observability: cursor pinning in notify() prevents eviction of the
    epoch containing the current record's cursor.  Eviction of epoch N happens
    atomically when the first record from epoch N+1 is delivered — in the same
    next() call that loads epoch N+1's chunks.  So the user always observes a
    stable inflight count (~flush_every_k_chunks) rather than a visible dip.
    The proof that mid-stream eviction works is max_inflight < total_chunks.
    """
    chunk_size = 4
    # Repeating Large-Small pattern across 8 chunks
    lengths = [5, 5, 5, 5, 1, 1, 1, 1] * 4  # 32 samples
    total_chunks = len(lengths) // chunk_size  # 8
    ds = _mk_varlen_dataset("vlms", lengths)

    # -- baseline: full run without checkpoint ---------------------------------
    baseline_pipe = _make_varlen_pack_pipeline(
        ds, runner=runner_kind, flush_every_k_chunks=2
    )
    baseline_texts: list[list[str]] = []
    for rec in baseline_pipe:
        assert isinstance(rec, SampleRecord)
        baseline_texts.append([s["text"] for s in rec.payload["packed_samples"]])

    assert len(baseline_texts) > 0, "Baseline must produce records"

    # -- run 1: iterate partway, verify mid-stream eviction, checkpoint --------
    pipe1 = _make_varlen_pack_pipeline(ds, runner=runner_kind, flush_every_k_chunks=2)
    it = iter(pipe1)

    max_inflight = 0
    eviction_decrease_seen = False
    prefix_texts: list[list[str]] = []
    ckpt: dict[str, Any] | None = None
    checkpoint_after = len(baseline_texts) // 2

    try:
        for rec in it:
            assert isinstance(rec, SampleRecord)
            prefix_texts.append([s["text"] for s in rec.payload["packed_samples"]])

            inflight = pipe1._engine.inflight_chunks_per_lane.get(0, {})
            n = len(inflight)
            if n > max_inflight:
                max_inflight = n
            elif n < max_inflight and max_inflight > 0:
                eviction_decrease_seen = True

            if len(prefix_texts) >= checkpoint_after:
                ckpt = pipe1.checkpoint()
                break
    finally:
        it.close()

    # Mid-stream eviction manifests differently by runner:
    # - Inline: feeder is synchronous, so eviction + chunk loading happen in
    #   the same next() call — inflight stays bounded (max < total_chunks)
    #   but never visibly decreases.
    # - Threads/process: feeder runs ahead loading all chunks, then eviction
    #   reduces inflight as records are consumed — visible decrease.
    # Either condition proves mid-stream eviction is working.
    assert max_inflight < total_chunks or eviction_decrease_seen, (
        f"Expected mid-stream eviction: either bounded inflight "
        f"(max_inflight={max_inflight} < total_chunks={total_chunks}) "
        f"or visible decrease (seen={eviction_decrease_seen}). "
        f"Consumed {len(prefix_texts)} records."
    )
    assert ckpt is not None

    # -- run 2: restore and drain ----------------------------------------------
    pipe2 = _make_varlen_pack_pipeline(ds, runner=runner_kind, flush_every_k_chunks=2)
    pipe2.restore(ckpt)

    suffix_texts: list[list[str]] = []
    for rec in pipe2:
        assert isinstance(rec, SampleRecord)
        suffix_texts.append([s["text"] for s in rec.payload["packed_samples"]])

    assert len(suffix_texts) > 0, "Should produce records after resume"

    # -- data correctness: no duplicates, exact match with baseline ------------
    all_prefix = [s for packed in prefix_texts for s in packed]
    all_suffix = [s for packed in suffix_texts for s in packed]
    all_combined = all_prefix + all_suffix
    all_baseline = [s for packed in baseline_texts for s in packed]

    dupes = set(all_prefix) & set(all_suffix)
    assert not dupes, (
        f"Samples duplicated across prefix and suffix: {sorted(dupes)}\n"
        f"  prefix: {all_prefix}\n"
        f"  suffix: {all_suffix}"
    )

    assert all_combined == all_baseline, (
        f"prefix + suffix should reconstruct baseline.\n"
        f"  prefix:   {all_prefix}\n"
        f"  suffix:   {all_suffix}\n"
        f"  baseline: {all_baseline}\n"
        f"  combined: {all_combined}"
    )


# ---------------------------------------------------------------------------
# The output matrix, end to end through real runners:
#   every registered algorithm x output {envelope, flat(+/-positions)}
# ---------------------------------------------------------------------------

_MATRIX_SEQS = [[10, 11, 12], [20, 21], [30, 31, 32, 33], [40]]
_MATRIX_MAXLEN = 4


def _mk_token_dataset(name: str, seqs: list[list[int]]) -> Dataset:
    """Single-shard dataset of token sequences under ``input_ids``."""
    rows: list[dict[str, Any]] = [{"input_ids": list(s)} for s in seqs]
    return Dataset.from_dict(name, {0: InMemoryShard(rows)})


def _matrix_work() -> StaticMixtureWorkSource:
    ds = _mk_token_dataset("m", _MATRIX_SEQS)
    return StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=2,
        seed=42,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )


@pytest.mark.parametrize(
    "runner_kind",
    ["inline", "threads", "process"],
    ids=["inline", "threads", "process"],
)
@pytest.mark.parametrize("algorithm", _PACKING_ALGORITHMS)
def test_pack_envelope_matrix_end_to_end(
    runner_kind: str, algorithm: PackingAlgorithm
) -> None:
    """Envelope output across the algorithm axis: each record is a list of
    per-segment dicts carrying the token field (boundaries preserved)."""
    pipeline = Pipeline(_matrix_work())
    pipeline.pack_sequences(
        max_length=_MATRIX_MAXLEN,
        num_bins=8 if algorithm in ("first_fit", "best_fit") else None,
        algorithm=algorithm,
    )
    pipeline.options(
        deterministic=True, max_workers=1, default_stage_prefetch=16, runner=runner_kind
    )

    records = list(pipeline)
    assert records
    for rec in records:
        assert set(rec.payload) == {"packed_samples"}
        segs = rec.payload["packed_samples"]
        assert isinstance(segs, list) and segs
        assert all("input_ids" in seg for seg in segs)


@pytest.mark.parametrize(
    "runner_kind",
    ["inline", "threads", "process"],
    ids=["inline", "threads", "process"],
)
@pytest.mark.parametrize("algorithm", _PACKING_ALGORITHMS)
@pytest.mark.parametrize("emit_positions", [True, False], ids=["pos", "nopos"])
def test_pack_flat_matrix_end_to_end(
    runner_kind: str, algorithm: PackingAlgorithm, emit_positions: bool
) -> None:
    """Flat output across algorithm × positions: every record is a fixed-length
    ``{input_ids[, positions]}`` the trainer can stack directly."""
    pipeline = Pipeline(_matrix_work())
    pipeline.pack_flat(
        max_length=_MATRIX_MAXLEN,
        num_bins=8 if algorithm in ("first_fit", "best_fit") else None,
        algorithm=algorithm,
        pad_token_id=-1,
        emit_positions=emit_positions,
    )
    pipeline.options(
        deterministic=True, max_workers=1, default_stage_prefetch=16, runner=runner_kind
    )

    records = list(pipeline)
    assert records
    expected_keys = {"input_ids", "positions"} if emit_positions else {"input_ids"}
    for rec in records:
        assert set(rec.payload) == expected_keys
        assert len(rec.payload["input_ids"]) == _MATRIX_MAXLEN
        if emit_positions:
            assert len(rec.payload["positions"]) == _MATRIX_MAXLEN
            assert rec.payload["positions"][0] == 0  # each bin starts a document


@pytest.mark.parametrize("algorithm", _PACKING_ALGORITHMS)
def test_pack_flat_equals_flatten_envelope_end_to_end(
    algorithm: PackingAlgorithm,
) -> None:
    """flat tokens (minus pad) reconstruct the envelope's concatenated segment
    tokens for the same input and algorithm — flat is a serialization of the
    same packing, not a different one."""

    def env_tokens() -> list[int]:
        p = Pipeline(_matrix_work())
        p.pack_sequences(
            max_length=_MATRIX_MAXLEN,
            num_bins=8 if algorithm in ("first_fit", "best_fit") else None,
            algorithm=algorithm,
        )
        p.options(deterministic=True, max_workers=1, runner="inline")
        return [
            t
            for rec in p
            for seg in rec.payload["packed_samples"]
            for t in seg["input_ids"]
        ]

    def flat_tokens() -> list[int]:
        p = Pipeline(_matrix_work())
        p.pack_flat(
            max_length=_MATRIX_MAXLEN,
            num_bins=8 if algorithm in ("first_fit", "best_fit") else None,
            algorithm=algorithm,
            pad_token_id=-1,
        )
        p.options(deterministic=True, max_workers=1, runner="inline")
        return [t for rec in p for t in rec.payload["input_ids"] if t != -1]

    assert flat_tokens() == env_tokens()


# ---------------------------------------------------------------------------
# Homogeneous packing (end-to-end through the engine)
# ---------------------------------------------------------------------------


def _multi_component_work() -> StaticMixtureWorkSource:
    """Two datasets (mixing domains) blended 50/50 into one interleaved lane."""
    a = _mk_token_dataset("a", [[1, 2, 3]] * 6)
    b = _mk_token_dataset("b", [[4, 5, 6]] * 6)
    return StaticMixtureWorkSource(
        [a, b],
        {"a": 0.5, "b": 0.5},
        chunk_size=4,
        seed=42,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )


@pytest.mark.parametrize(
    "algorithm", ["first_fit", "best_fit", "wrap", "best_fit_wrap"]
)
def test_pack_homogeneous_full_end_to_end(algorithm: str) -> None:
    """Through the real engine, homogeneity='full' keeps each packed sample to one
    mixing domain, across every algorithm."""
    pipeline = Pipeline(_multi_component_work())
    pipeline.pack_flat(
        max_length=10,
        num_bins=8 if algorithm in ("first_fit", "best_fit") else None,
        pad_token_id=0,
        algorithm=algorithm,
        homogeneity="full",
        drop_oversized=False,
    )
    pipeline.options(
        deterministic=True, max_workers=1, default_stage_prefetch=16, runner="threads"
    )

    records = list(pipeline)
    assert records
    # wrap counts only closing slices in sample counts, so use token counts.
    if algorithm in ("wrap", "best_fit_wrap"):
        counts = [rec.meta.component_token_counts for rec in records]
    else:
        counts = [rec.meta.component_sample_counts for rec in records]
    assert all(c is not None and len(c) == 1 for c in counts)
    # Require both datasets to reach packing, so single-domain packing is a real
    # constraint here rather than trivially true.
    seen = {cid for c in counts if c for cid in c}
    assert len(seen) >= 2


def test_pack_homogeneous_none_end_to_end_mixes() -> None:
    """Back-compat: the default homogeneity='none' still mixes domains end-to-end."""
    pipeline = Pipeline(_multi_component_work())
    pipeline.pack_flat(
        max_length=10,
        num_bins=8,
        pad_token_id=0,
        homogeneity="none",
        drop_oversized=False,
    )
    pipeline.options(deterministic=True, max_workers=1, runner="inline")

    records = list(pipeline)
    assert records
    assert any(len(rec.meta.component_sample_counts) == 2 for rec in records)


def test_group_unknown_component_name_raises_at_build() -> None:
    pipeline = Pipeline(_multi_component_work())  # datasets "a", "b"
    with pytest.raises(ValueError, match="not in the mixture"):
        pipeline.pack_flat(
            max_length=10,
            num_bins=8,
            pad_token_id=0,
            homogeneity="group",
            groups={"g": ["a", "nope"]},  # "nope" is not a dataset in the mixture
            drop_oversized=False,
        )


def test_group_mode_mismatch_reports_precise_error() -> None:
    pipeline = Pipeline(_multi_component_work())
    with pytest.raises(ValueError, match="only valid with homogeneity='group'"):
        pipeline.pack_flat(
            max_length=10,
            num_bins=8,
            pad_token_id=0,
            homogeneity="full",
            groups={"g": ["typo"]},  # bogus member must not mask the mode error
            drop_oversized=False,
        )


def test_group_two_groups_mapping_rejected_by_pipeline() -> None:
    pipeline = Pipeline(_multi_component_work())
    with pytest.raises(ValueError, match="multiple groups"):
        pipeline.pack_flat(
            max_length=10,
            num_bins=8,
            pad_token_id=0,
            homogeneity="group",
            groups={"g1": ["a"], "g2": ["a"]},  # "a" in two groups
            drop_oversized=False,
        )


def test_group_mode_without_groups_raises_at_build() -> None:
    pipeline = Pipeline(_multi_component_work())
    with pytest.raises(ValueError, match="requires groups"):
        pipeline.pack_flat(
            max_length=10,
            num_bins=8,
            pad_token_id=0,
            homogeneity="group",
            drop_oversized=False,
        )


def test_group_incomplete_grouping_warns_at_build() -> None:
    pipeline = Pipeline(_multi_component_work())  # components "a", "b"
    with pytest.warns(UserWarning, match="in no group"):
        pipeline.pack_flat(
            max_length=10,
            num_bins=8,
            pad_token_id=0,
            homogeneity="group",
            groups={"g": ["a"]},  # "b" omitted -> silent singleton without the warning
            drop_oversized=False,
        )


def _grouped_component_work() -> StaticMixtureWorkSource:
    """Three datasets with disjoint token ranges (a->1xx, b->2xx, c->3xx) so a
    bin's datasets are readable from its tokens."""
    a = _mk_token_dataset("a", [[100, 101, 102]] * 8)
    b = _mk_token_dataset("b", [[200, 201, 202]] * 8)
    c = _mk_token_dataset("c", [[300, 301, 302]] * 8)
    return StaticMixtureWorkSource(
        [a, b, c],
        {"a": 1.0, "b": 1.0, "c": 1.0},
        chunk_size=6,
        seed=13,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )


@pytest.mark.parametrize("runner_kind", ["inline", "threads", "process"])
def test_pack_homogeneous_group_end_to_end(runner_kind: str) -> None:
    """End-to-end: each packed sample stays within one group, with real
    ``get_component_id`` resolution. Parametrized over runners so the process
    path exercises cloudpickling the op's ``DomainGroups``; per-algorithm group
    packing is covered by the unit tests."""
    pipeline = Pipeline(_grouped_component_work())
    pipeline.pack_flat(
        max_length=12,
        num_bins=8,
        pad_token_id=0,
        algorithm="first_fit",
        homogeneity="group",
        groups={"code": ["a", "b"]},  # c is ungrouped -> its own singleton domain
        drop_oversized=False,
    )
    pipeline.options(
        deterministic=True,
        max_workers=1,
        default_stage_prefetch=16,
        runner=runner_kind,
    )

    def bin_datasets(rec: SampleRecord) -> set[int]:
        ids = rec.payload["input_ids"]
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        # token // 100: 1=a, 2=b (both in group "code"), 3=c (ungrouped). Pad is 0.
        return {int(t) // 100 for t in ids if t != 0}

    per_bin = [bin_datasets(r) for r in pipeline]
    assert per_bin
    # No bin crosses the group boundary: within {a,b} OR c-only.
    assert all(cs <= {1, 2} or cs <= {3} for cs in per_bin)
    # Grouping widens beyond fully-homogeneous: some bin mixes a and b.
    assert any(cs == {1, 2} for cs in per_bin)
    # All three datasets reached packing, so the constraint is non-trivial.
    assert set().union(*per_bin) == {1, 2, 3}


# ---------------------------------------------------------------------------
# Parallel materialization
# ---------------------------------------------------------------------------


def _big_token_work() -> StaticMixtureWorkSource:
    """Dataset large enough to fan many bins across worker threads."""
    seqs = [list(range(i, i + (i % 7) + 1)) for i in range(200)]
    ds = _mk_token_dataset("big", seqs)
    return StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=8,
        seed=7,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )


def _norm_payload(payload: Any) -> Any:
    """Normalize a packed payload for equality (numpy arrays -> lists)."""
    import numpy as np

    def norm(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        if isinstance(v, list):
            return [norm(x) for x in v]
        return v

    return norm(payload)


def test_pack_parallelism_plumbed_to_node() -> None:
    """The pack_flat/pack_sequences ``parallelism`` arg reaches the graph node."""
    p = Pipeline(_matrix_work())
    p.pack_flat(max_length=4, algorithm="wrap", drop_oversized=False, parallelism=4)
    node = p._graph.nodes[-1]
    assert node.name == "pack_flat" and node.parallelism == 4


@pytest.mark.parametrize("algorithm", _PACKING_ALGORITHMS)
@pytest.mark.parametrize("output", ["envelope", "flat"], ids=["envelope", "flat"])
def test_pack_parallelism_invariant(algorithm: PackingAlgorithm, output: str) -> None:
    """parallelism>1 produces identical payloads, order, and lineage as
    parallelism=1: bin assignment stays serial, only materialization fans out."""

    def run(parallelism: int) -> list[tuple[Any, Any]]:
        p = Pipeline(_big_token_work())
        if output == "flat":
            p.pack_flat(
                max_length=8,
                num_bins=8 if algorithm in ("first_fit", "best_fit") else None,
                algorithm=algorithm,
                pad_token_id=-1,
                parallelism=parallelism,
            )
        else:
            p.pack_sequences(
                max_length=8,
                num_bins=8 if algorithm in ("first_fit", "best_fit") else None,
                algorithm=algorithm,
                parallelism=parallelism,
            )
        p.options(deterministic=True, runner="threads", default_stage_prefetch=16)
        return [(r.meta.cursor.as_key(), _norm_payload(r.payload)) for r in p]

    serial = run(1)
    assert serial
    assert run(4) == serial


def test_best_fit_wrap_checkpoint_resume_matches_baseline() -> None:
    def make_pipeline() -> Pipeline:
        pipeline = Pipeline(_big_token_work())
        pipeline.pack_sequences(
            max_length=8,
            algorithm="best_fit_wrap",
            candidate_pool_size=8,
        )
        pipeline.options(deterministic=True, runner="inline")
        return pipeline

    baseline = [
        (record.meta.cursor.as_key(), _norm_payload(record.payload))
        for record in make_pipeline()
    ]

    first = make_pipeline()
    iterator = iter(first)
    prefix = []
    try:
        for _ in range(10):
            record = next(iterator)
            prefix.append((record.meta.cursor.as_key(), _norm_payload(record.payload)))
        checkpoint = first.checkpoint()
    finally:
        iterator.close()

    resumed = make_pipeline()
    resumed.restore(checkpoint)
    suffix = [
        (record.meta.cursor.as_key(), _norm_payload(record.payload))
        for record in resumed
    ]
    assert prefix + suffix == baseline
