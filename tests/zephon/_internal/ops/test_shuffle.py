from collections import defaultdict
from typing import Any

import pytest

from tests.zephon._internal.ops.conftest import mk_dataset
from zephon._internal.ops.shuffle_buffer import (
    ShuffleBuffer,
    StreamingShuffleAccumulator,
    WarmupBlockAccumulator,
    _seed_from_cursor,
    _stable_index_scalar,
    _stable_index_vec,
    _victim_indices,
)
from zephon.io.dataset import Dataset
from zephon.ops.accumulators import CountingAccumulator
from zephon.ops.base import OpContext
from zephon.ops.children import spawn_child
from zephon.pipeline import Pipeline
from zephon.types import SampleMeta, SampleRecord
from zephon.work.static_mixture import StaticMixtureWorkSource


def _records(n: int, *, lane_id: int = 0) -> list[SampleRecord]:
    return [
        SampleRecord(
            meta=SampleMeta(
                sample_id=(0, lane_id, i),
                lane_id=lane_id,
                chunk_id=i // 4,
                chunk_offset=i % 4,
            ),
            payload={"text": f"L{lane_id}:{i}"},
        )
        for i in range(n)
    ]


def _drain_accumulator(
    acc: StreamingShuffleAccumulator, records: list[SampleRecord]
) -> list[SampleRecord]:
    out: list[SampleRecord] = []
    out.extend(record for batch, _ in acc.push_many(records) for record in batch)
    out.extend(record for batch, _ in acc.flush(reset=True) for record in batch)
    return out


def test_shuffle_buffer_is_deterministic() -> None:
    recs1 = _records(10)
    recs2 = _records(10)
    original_order = [r.meta.cursor for r in recs1]
    op1 = ShuffleBuffer(buffer_size=3, seed=123, algorithm="block")
    op2 = ShuffleBuffer(buffer_size=3, seed=123, algorithm="block")
    ctx = OpContext({"record_node_metrics": lambda *args, **kwargs: None})
    op1.setup(ctx)
    op2.setup(ctx)

    out1 = op1.process_many(recs1)
    out2 = op2.process_many(recs2)

    assert [r.meta.cursor for r in out1] == [r.meta.cursor for r in out2]
    assert [r.meta.cursor for r in out1] != original_order


def test_shuffle_buffer_flushes_tail() -> None:
    recs = _records(2)
    op = ShuffleBuffer(buffer_size=4, seed=7, algorithm="block")
    ctx = OpContext({"record_node_metrics": lambda *args, **kwargs: None})
    op.setup(ctx)

    # buffer smaller than window → no immediate output from process_many
    shuffled = op.process_many(recs)
    assert sorted(r.meta.chunk_offset for r in shuffled) == [0, 1]


def test_shuffle_buffer_keeps_closer_after_non_closer_for_same_base() -> None:
    base = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0, chunk_offset=5)
    first = spawn_child(base, 0, is_last_child=False)
    last = spawn_child(base, 1, is_last_child=True)
    recs = [SampleRecord(meta=first, payload={}), SampleRecord(meta=last, payload={})]

    op = ShuffleBuffer(buffer_size=2, seed=99, algorithm="block")
    ctx = OpContext({"record_node_metrics": lambda *args, **kwargs: None})
    op.setup(ctx)

    out = op.process_many(recs)
    # The closing contributor must be on the last occurrence for the base offset.
    last_idx = max(
        idx
        for idx, rec in enumerate(out)
        if rec.meta.chunk_id == base.chunk_id
        and rec.meta.chunk_offset == base.chunk_offset
    )
    for idx, rec in enumerate(out):
        closes = any(ref.is_last_child for ref in rec.meta.contribution_refs())
        if idx == last_idx:
            assert closes
        else:
            assert not closes


def test_streaming_shuffle_emits_before_full_buffer() -> None:
    acc = StreamingShuffleAccumulator(buffer_size=8, seed=123)
    batches = acc.push_many(_records(6))
    emitted = [rec for batch, _ in batches for rec in batch]

    assert emitted, "streaming shuffle should emit during warmup"
    assert len(emitted) < 6, "streaming shuffle should retain a growing reservoir"
    assert acc.has_pending_data()


def test_streaming_shuffle_deterministic_across_push_boundaries() -> None:
    acc_a = StreamingShuffleAccumulator(buffer_size=8, seed=123)
    out_a = _drain_accumulator(acc_a, _records(32))

    acc_b = StreamingShuffleAccumulator(buffer_size=8, seed=123)
    out_b: list[SampleRecord] = []
    for record in _records(32):
        out_b.extend(
            emitted for batch, _ in acc_b.push_many([record]) for emitted in batch
        )
    out_b.extend(emitted for batch, _ in acc_b.flush(reset=True) for emitted in batch)

    assert [rec.meta.cursor for rec in out_a] == [rec.meta.cursor for rec in out_b]


def test_streaming_shuffle_reset_matches_fresh_accumulator() -> None:
    acc = StreamingShuffleAccumulator(buffer_size=8, seed=123)
    _drain_accumulator(acc, _records(24))
    assert not acc.has_pending_data()

    fresh = StreamingShuffleAccumulator(buffer_size=8, seed=123)
    out_after_reset = _drain_accumulator(acc, _records(24))
    out_fresh = _drain_accumulator(fresh, _records(24))

    assert [rec.meta.cursor for rec in out_after_reset] == [
        rec.meta.cursor for rec in out_fresh
    ]


def test_streaming_shuffle_lane_order_independent_of_interleaving() -> None:
    lane0 = _records(24, lane_id=0)
    lane1 = _records(24, lane_id=1)
    interleaved = [rec for pair in zip(lane0, lane1) for rec in pair]

    combined = _drain_accumulator(
        StreamingShuffleAccumulator(buffer_size=8, seed=99), interleaved
    )
    isolated0 = _drain_accumulator(
        StreamingShuffleAccumulator(buffer_size=8, seed=99), _records(24, lane_id=0)
    )
    isolated1 = _drain_accumulator(
        StreamingShuffleAccumulator(buffer_size=8, seed=99), _records(24, lane_id=1)
    )

    combined_by_lane: dict[int, list[str]] = defaultdict(list)
    for rec in combined:
        combined_by_lane[rec.meta.lane_id].append(str(rec.payload["text"]))

    assert combined_by_lane[0] == [str(rec.payload["text"]) for rec in isolated0]
    assert combined_by_lane[1] == [str(rec.payload["text"]) for rec in isolated1]


def test_streaming_shuffle_moves_closer_to_last_emitted_child() -> None:
    """Online closer rewriting keeps exactly one closer per base offset."""
    base = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0, chunk_offset=5)
    first = spawn_child(base, 0, is_last_child=False)
    last = spawn_child(base, 1, is_last_child=True)
    recs = [SampleRecord(meta=first, payload={}), SampleRecord(meta=last, payload={})]

    out = _drain_accumulator(StreamingShuffleAccumulator(buffer_size=2, seed=99), recs)

    last_idx = max(
        idx
        for idx, rec in enumerate(out)
        if rec.meta.chunk_id == base.chunk_id
        and rec.meta.chunk_offset == base.chunk_offset
    )
    for idx, rec in enumerate(out):
        closes = any(ref.is_last_child for ref in rec.meta.contribution_refs())
        if idx == last_idx:
            assert closes
        else:
            assert not closes


def test_stable_index_scalar_matches_vectorized() -> None:
    seed_mixed = 0x123456789ABCDEF
    lanes = [0, 1, 7, 3, 0, 255, 1, 2, 9, 4]
    salts = [0, 1, 2**32 - 1, 12345, 999, 2**31, 7, 8, 9, 2**16]
    counters = [0, 1, 2, 5, 100, 8191, 8192, 999999, 3, 4]
    sizes = [1, 8, 8193, 8192, 4096, 2, 8192, 8192, 17, 33]

    vec = _stable_index_vec(
        seed_mixed=seed_mixed,
        lanes=lanes,
        salts=salts,
        counters=counters,
        sizes=sizes,
    )
    scalar = [
        _stable_index_scalar(seed_mixed, lane, salt, counter, size)
        for lane, salt, counter, size in zip(lanes, salts, counters, sizes)
    ]
    assert vec == scalar
    assert all(0 <= idx < size for idx, size in zip(vec, sizes))


def test_streaming_shuffle_vectorized_matches_small_batches() -> None:
    """Large single push (vector path) matches per-record pushes (scalar path)."""
    big = StreamingShuffleAccumulator(buffer_size=64, seed=7)
    out_big = _drain_accumulator(big, _records(4096))

    small = StreamingShuffleAccumulator(buffer_size=64, seed=7)
    out_small: list[SampleRecord] = []
    for record in _records(4096):
        out_small.extend(rec for batch, _ in small.push_many([record]) for rec in batch)
    out_small.extend(rec for batch, _ in small.flush(reset=True) for rec in batch)

    assert [r.meta.cursor for r in out_big] == [r.meta.cursor for r in out_small]


def test_seed_from_cursor_is_stable_and_cursor_sensitive() -> None:
    a = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=2, chunk_offset=3)
    a_again = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=2, chunk_offset=3)
    by_offset = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=2, chunk_offset=4)
    by_chunk = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=3, chunk_offset=3)

    assert _seed_from_cursor(a) == _seed_from_cursor(a_again)
    assert _seed_from_cursor(a) != _seed_from_cursor(by_offset)
    assert _seed_from_cursor(a) != _seed_from_cursor(by_chunk)
    assert 0 <= _seed_from_cursor(a) <= 0xFFFFFFFF


def test_victim_indices_broadcast_matches_lists_and_stays_in_range() -> None:
    for n in (3, 40):  # 3 -> scalar path, 40 -> vector path (>= _VEC_MIN)
        counters = list(range(n))
        sizes = list(range(n, 0, -1))
        broadcast = _victim_indices(
            seed_mixed=12345, lanes=3, salts=99, counters=counters, sizes=sizes
        )
        explicit = _victim_indices(
            seed_mixed=12345,
            lanes=[3] * n,
            salts=[99] * n,
            counters=counters,
            sizes=sizes,
        )
        assert broadcast == explicit
        assert all(0 <= idx < size for idx, size in zip(broadcast, sizes))


# ---------------------------------------------------------------------------
# block_warmup
# ---------------------------------------------------------------------------


def _drain_blocks(
    acc: WarmupBlockAccumulator, records: list[SampleRecord]
) -> list[list[SampleRecord]]:
    blocks = [batch for batch, _ in acc.push_many(records)]
    blocks.extend(batch for batch, _ in acc.flush(reset=True))
    return blocks


def test_warmup_block_ramps_block_size() -> None:
    acc = WarmupBlockAccumulator(buffer_size=64, growth=2.0)
    blocks = _drain_blocks(acc, _records(400))
    sizes = [len(b) for b in blocks]
    assert sizes[:5] == [4, 8, 16, 32, 64]
    assert all(s <= 64 for s in sizes)
    assert sum(sizes) == 400


def test_warmup_block_gentle_growth_is_smaller_steps() -> None:
    acc = WarmupBlockAccumulator(buffer_size=64, growth=1.5)
    blocks = _drain_blocks(acc, _records(400))
    sizes = [len(b) for b in blocks]
    assert sizes[:7] == [4, 6, 9, 14, 21, 32, 48]
    assert sizes.count(64) >= 1 and all(s <= 64 for s in sizes)
    assert sum(sizes) == 400


def test_warmup_block_growth_must_exceed_one() -> None:
    with pytest.raises(ValueError):
        WarmupBlockAccumulator(buffer_size=64, growth=1.0)


def test_warmup_block_first_block_is_small() -> None:
    acc = WarmupBlockAccumulator(buffer_size=8192)
    blocks = [batch for batch, _ in acc.push_many(_records(200))]
    assert blocks, "should emit during warmup, not wait for 8192"
    assert len(blocks[0]) == 128  # _streaming_warmup_min_buffer(8192)


def test_warmup_block_flush_resets_schedule() -> None:
    acc = WarmupBlockAccumulator(buffer_size=64)
    _drain_blocks(acc, _records(400))
    assert not acc.has_pending_data()
    blocks = [batch for batch, _ in acc.push_many(_records(20))]
    assert len(blocks[0]) == 4  # ramp restarted, not continued at 64


def test_warmup_block_is_lane_pure() -> None:
    lane0 = _records(40, lane_id=0)
    lane1 = _records(40, lane_id=1)
    interleaved = [rec for pair in zip(lane0, lane1) for rec in pair]
    acc = WarmupBlockAccumulator(buffer_size=64)
    blocks = _drain_blocks(acc, interleaved)
    for block in blocks:
        lanes = {rec.meta.lane_id for rec in block}
        assert len(lanes) == 1, f"block mixes lanes: {lanes}"


# ---------------------------------------------------------------------------
# Constructor validation, accumulator dispatch, and op surface
# ---------------------------------------------------------------------------


def test_streaming_accumulator_rejects_nonpositive_buffer() -> None:
    with pytest.raises(ValueError):
        StreamingShuffleAccumulator(buffer_size=0)


def test_warmup_block_rejects_nonpositive_buffer() -> None:
    with pytest.raises(ValueError):
        WarmupBlockAccumulator(buffer_size=0)


@pytest.mark.parametrize(
    "kwargs",
    [{"buffer_size": 0}, {"algorithm": "nonsense"}, {"warmup_growth": 1.0}],
)
def test_shuffle_buffer_rejects_bad_args(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ShuffleBuffer(**kwargs)


def test_streaming_buffer_size_one_is_passthrough() -> None:
    acc = StreamingShuffleAccumulator(buffer_size=1, seed=5)
    recs = _records(10)
    out = _drain_accumulator(acc, recs)
    assert [r.meta.cursor for r in out] == [r.meta.cursor for r in recs]
    assert not acc.has_pending_data()


def test_shuffle_buffer_accumulator_dispatch() -> None:
    ctx: dict[str, Any] = {}
    streaming = ShuffleBuffer(buffer_size=16, algorithm="streaming").accumulator(
        deterministic=True, ctx=ctx
    )
    warmup = ShuffleBuffer(buffer_size=16, algorithm="block_warmup").accumulator(
        deterministic=True, ctx=ctx
    )
    block = ShuffleBuffer(buffer_size=16, algorithm="block").accumulator(
        deterministic=True, ctx=ctx
    )
    assert isinstance(streaming, StreamingShuffleAccumulator)
    assert isinstance(warmup, WarmupBlockAccumulator)
    assert isinstance(block, CountingAccumulator)


def test_shuffle_buffer_traits_and_process_one() -> None:
    op = ShuffleBuffer(buffer_size=4, seed=0, algorithm="block")
    traits = op.traits()
    assert traits.parallelism == 1
    assert traits.preserves_cursor_order is False

    [rec] = _records(1)
    assert op.process_one(rec) == [rec]


# ---------------------------------------------------------------------------
# Checkpoint / resume + cross-chunk eviction
# ---------------------------------------------------------------------------


def _make_shuffle_pipeline(
    ds: Dataset,
    buffer_size: int = 5,
    seed: int = 0,
    *,
    flush_every_k_chunks: int | None = None,
    runner: str | None = None,
    algorithm: str = "streaming",
) -> Pipeline:
    """Build a deterministic pipeline with shuffle."""
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=4,
        seed=42,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = Pipeline(work)
    pipeline.shuffle(buffer_size=buffer_size, seed=seed, algorithm=algorithm)
    opts: dict[str, Any] = dict(
        deterministic=True,
        max_workers=1,
        default_stage_prefetch=16,
        mtp_mode=False,
    )
    if flush_every_k_chunks is not None:
        opts["flush_every_k_chunks"] = flush_every_k_chunks
    if runner is not None:
        opts["runner"] = runner
    pipeline.options(**opts)
    return pipeline


@pytest.mark.parametrize(
    "runner_kind",
    ["inline", "threads", "process"],
    ids=["inline", "threads", "process"],
)
def test_shuffle_cross_chunk_data_correctness_after_checkpoint(
    runner_kind: str,
) -> None:
    """Checkpoint/restore must not duplicate samples from cross-chunk shuffle buffers.

    Setup (12 samples, chunk_size=4, buffer_size=5):

        Buffer 1: [s0, s1, s2, s3, s4]  -- chunk 0 (all 4) + chunk 1 (1)
        Buffer 2: [s5, s6, s7, s8, s9]  -- chunk 1 (3) + chunk 2 (2)
        Buffer 3: [s10, s11]            -- chunk 2 (2), flushed at end

    With atomic eviction (gated by flush sentinel epoch floor), chunks that
    share shuffle-buffer state are never evicted independently.  The cursor
    never references an evicted chunk, and on restore all chunks are replayed
    → same buffer state → same shuffle output → no data corruption.
    """
    ds = mk_dataset("shuf", {0: 12})

    # -- baseline: full run without checkpoint ---------------------------------
    baseline_pipe = _make_shuffle_pipeline(ds, runner=runner_kind)
    baseline_texts: list[str] = []
    for rec in baseline_pipe:
        assert isinstance(rec, SampleRecord)
        baseline_texts.append(rec.payload["text"])

    assert len(baseline_texts) == 12, f"Expected 12 records, got {len(baseline_texts)}"

    # -- run 1: consume until cursor references evicted chunk ------------------
    # With atomic eviction the cursor never lands on an evicted chunk.
    # We iterate the full stream: if no eviction-triggered checkpoint fires,
    # we checkpoint after the last record instead.
    pipe1 = _make_shuffle_pipeline(ds, runner=runner_kind)
    it = iter(pipe1)
    prefix_texts: list[str] = []
    ckpt: dict[str, Any] | None = None
    try:
        for rec in it:
            assert isinstance(rec, SampleRecord)
            prefix_texts.append(rec.payload["text"])

            eng = pipe1._engine
            assert eng is not None
            lane_cursor = eng._lane_last_cursor.get(0)
            inflight = eng.inflight_chunks_per_lane.get(0, {})
            if lane_cursor is not None and lane_cursor.chunk_id not in inflight:
                ckpt = pipe1.checkpoint()
                break
    finally:
        it.close()

    if ckpt is None:
        # Atomic eviction prevented the cursor from ever referencing an
        # evicted chunk — this is the expected (fixed) behavior.  Checkpoint
        # mid-stream instead to verify restore correctness.
        pipe1b = _make_shuffle_pipeline(ds, runner=runner_kind)
        it2 = iter(pipe1b)
        prefix_texts = []
        try:
            for _ in range(6):
                rec = next(it2)
                assert isinstance(rec, SampleRecord)
                prefix_texts.append(rec.payload["text"])
            ckpt = pipe1b.checkpoint()
        finally:
            it2.close()

    # -- run 2: restore and drain ----------------------------------------------
    pipe2 = _make_shuffle_pipeline(ds, runner=runner_kind)
    pipe2.restore(ckpt)

    suffix_texts: list[str] = []
    for rec in pipe2:
        assert isinstance(rec, SampleRecord)
        suffix_texts.append(rec.payload["text"])

    assert len(suffix_texts) > 0, "Should produce records after resume"

    # -- data correctness: no duplicates, exact match with baseline ------------
    all_combined = prefix_texts + suffix_texts

    assert all_combined == baseline_texts, (
        f"prefix + suffix should reconstruct the baseline sample stream.\n"
        f"  prefix:   {prefix_texts}\n"
        f"  suffix:   {suffix_texts}\n"
        f"  baseline: {baseline_texts}\n"
        f"  combined: {all_combined}"
    )


# ---------------------------------------------------------------------------
# Mid-stream eviction + data correctness with flush sentinels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "runner_kind",
    ["inline", "threads", "process"],
    ids=["inline", "threads", "process"],
)
def test_shuffle_mid_stream_eviction_after_checkpoint(runner_kind: str) -> None:
    """Mid-stream eviction + checkpoint/restore with shuffle.

    Like test_shuffle_cross_chunk_data_correctness_after_checkpoint but with
    flush_every_k_chunks=2 and more data.  Verifies both:
    1. Chunks from completed epochs evict DURING iteration (not cleanup)
    2. Checkpoint/restore after eviction produces identical output to baseline

    Setup (32 samples, chunk_size=4, buffer_size=5, flush_every_k_chunks=2):
      - 8 chunks, sentinels after chunks 1, 3, 5, 7
      - Shuffle buffer spans chunk boundaries → cross-chunk state
    """
    ds = mk_dataset("shms", {0: 32})
    total_chunks = 32 // 4  # 8

    # -- baseline: full run without checkpoint ---------------------------------
    baseline_pipe = _make_shuffle_pipeline(
        ds, flush_every_k_chunks=2, runner=runner_kind
    )
    baseline_texts: list[str] = []
    for rec in baseline_pipe:
        assert isinstance(rec, SampleRecord)
        baseline_texts.append(rec.payload["text"])

    assert len(baseline_texts) == 32, f"Expected 32 records, got {len(baseline_texts)}"

    # -- run 1: iterate partway, verify mid-stream eviction, checkpoint --------
    pipe1 = _make_shuffle_pipeline(ds, flush_every_k_chunks=2, runner=runner_kind)
    it = iter(pipe1)

    max_inflight = 0
    eviction_decrease_seen = False
    prefix_texts: list[str] = []
    checkpoint_after = len(baseline_texts) // 2
    ckpt: dict[str, Any] | None = None

    try:
        for rec in it:
            assert isinstance(rec, SampleRecord)
            prefix_texts.append(rec.payload["text"])

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

    assert max_inflight < total_chunks or eviction_decrease_seen, (
        f"Expected mid-stream eviction: either bounded inflight "
        f"(max_inflight={max_inflight} < total_chunks={total_chunks}) "
        f"or visible decrease (seen={eviction_decrease_seen}). "
        f"Consumed {len(prefix_texts)} records."
    )
    assert ckpt is not None

    # -- run 2: restore and drain ----------------------------------------------
    pipe2 = _make_shuffle_pipeline(ds, flush_every_k_chunks=2, runner=runner_kind)
    pipe2.restore(ckpt)

    suffix_texts: list[str] = []
    for rec in pipe2:
        assert isinstance(rec, SampleRecord)
        suffix_texts.append(rec.payload["text"])

    assert len(suffix_texts) > 0, "Should produce records after resume"

    # -- data correctness: no duplicates, exact match with baseline ------------
    all_combined = prefix_texts + suffix_texts

    dupes = set(prefix_texts) & set(suffix_texts)
    assert not dupes, (
        f"Samples duplicated across prefix and suffix: {sorted(dupes)}\n"
        f"  prefix: {prefix_texts}\n"
        f"  suffix: {suffix_texts}"
    )

    assert all_combined == baseline_texts, (
        f"prefix + suffix should reconstruct baseline.\n"
        f"  prefix:   {prefix_texts}\n"
        f"  suffix:   {suffix_texts}\n"
        f"  baseline: {baseline_texts}\n"
        f"  combined: {all_combined}"
    )


# ---------------------------------------------------------------------------
# flush_every_k_chunks cadence change across checkpoint/restore
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "runner_kind",
    ["inline", "threads", "process"],
    ids=["inline", "threads", "process"],
)
def test_flush_cadence_change_across_checkpoint(runner_kind: str) -> None:
    """Changing flush_every_k_chunks across checkpoint/restore must not lose data.

    Replay (Phase 1) re-injects sentinels at the stored epoch boundary
    positions from the checkpoint — it does not use the new K value.
    Phase 2 (fresh chunks) uses the new K value.

    Setup (64 samples, chunk_size=4, buffer_size=5):
      - K_old=2: 16 chunks, sentinels after every 2 chunks
      - Checkpoint after mid-stream eviction (epoch 0 evicted)
      - K_new=4: Phase 2 fires sentinels every 4 fresh chunks

    Verifies:
    1. Mid-stream eviction happens before checkpoint (epoch 0 removed)
    2. prefix + suffix covers the exact same records as baseline (no loss)
    3. No duplicates across prefix and suffix
    """
    ds = mk_dataset("cadence", {0: 64})

    # -- baseline: full run with K=2 ------------------------------------------
    baseline_pipe = _make_shuffle_pipeline(
        ds, flush_every_k_chunks=2, runner=runner_kind
    )
    baseline_texts: list[str] = []
    for rec in baseline_pipe:
        assert isinstance(rec, SampleRecord)
        baseline_texts.append(rec.payload["text"])
    assert len(baseline_texts) == 64

    # -- run 1: iterate partway, verify mid-stream eviction, checkpoint (K=2) -
    total_chunks = 64 // 4  # 16
    pipe1 = _make_shuffle_pipeline(ds, flush_every_k_chunks=2, runner=runner_kind)
    it = iter(pipe1)

    max_inflight = 0
    eviction_decrease_seen = False
    prefix_texts: list[str] = []
    checkpoint_after = len(baseline_texts) // 2
    ckpt: dict[str, Any] | None = None

    try:
        for rec in it:
            assert isinstance(rec, SampleRecord)
            prefix_texts.append(rec.payload["text"])

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

    assert max_inflight < total_chunks or eviction_decrease_seen, (
        f"Expected mid-stream eviction: either bounded inflight "
        f"(max_inflight={max_inflight} < total_chunks={total_chunks}) "
        f"or visible decrease (seen={eviction_decrease_seen}). "
        f"Consumed {len(prefix_texts)} records."
    )
    assert ckpt is not None

    # -- run 2: restore with DIFFERENT K=4, drain ------------------------------
    pipe2 = _make_shuffle_pipeline(ds, flush_every_k_chunks=4, runner=runner_kind)
    pipe2.restore(ckpt)

    suffix_texts: list[str] = []
    for rec in pipe2:
        assert isinstance(rec, SampleRecord)
        suffix_texts.append(rec.payload["text"])

    assert len(suffix_texts) > 0, "Should produce records after resume"

    # -- no duplicates across prefix and suffix --------------------------------
    all_combined = prefix_texts + suffix_texts
    dupes = set(prefix_texts) & set(suffix_texts)
    assert not dupes, (
        f"Samples duplicated across prefix and suffix: {sorted(dupes)}\n"
        f"  prefix ({len(prefix_texts)}): {prefix_texts[:10]}...\n"
        f"  suffix ({len(suffix_texts)}): {suffix_texts[:10]}..."
    )

    # -- same set of records as baseline (ordering may differ in suffix due
    #    to different sentinel cadence in Phase 2) -----------------------------
    assert sorted(all_combined) == sorted(baseline_texts), (
        f"All records from baseline must appear in prefix + suffix.\n"
        f"  combined ({len(all_combined)}): {sorted(all_combined)[:10]}...\n"
        f"  baseline ({len(baseline_texts)}): {sorted(baseline_texts)[:10]}..."
    )


# ---------------------------------------------------------------------------
# block_warmup checkpoint/restore correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "runner_kind",
    ["inline", "threads", "process"],
    ids=["inline", "threads", "process"],
)
def test_block_warmup_checkpoint_reconstructs_baseline(runner_kind: str) -> None:
    """block_warmup satisfies the replay-from-boundary checkpoint contract."""
    ds = mk_dataset("bwck", {0: 32})

    baseline_pipe = _make_shuffle_pipeline(
        ds, flush_every_k_chunks=2, runner=runner_kind, algorithm="block_warmup"
    )
    baseline_texts = [rec.payload["text"] for rec in baseline_pipe]
    assert len(baseline_texts) == 32

    pipe1 = _make_shuffle_pipeline(
        ds, flush_every_k_chunks=2, runner=runner_kind, algorithm="block_warmup"
    )
    it = iter(pipe1)
    prefix_texts: list[str] = []
    ckpt: dict[str, Any] | None = None
    try:
        for rec in it:
            assert isinstance(rec, SampleRecord)
            prefix_texts.append(rec.payload["text"])
            if len(prefix_texts) >= 16:
                ckpt = pipe1.checkpoint()
                break
    finally:
        it.close()
    assert ckpt is not None

    pipe2 = _make_shuffle_pipeline(
        ds, flush_every_k_chunks=2, runner=runner_kind, algorithm="block_warmup"
    )
    pipe2.restore(ckpt)
    suffix_texts = [rec.payload["text"] for rec in pipe2]

    all_combined = prefix_texts + suffix_texts
    assert not (set(prefix_texts) & set(suffix_texts)), "duplicate across prefix/suffix"
    assert all_combined == baseline_texts


def test_streaming_shuffle_per_lane_flush_only_drains_that_lane() -> None:
    acc = StreamingShuffleAccumulator(buffer_size=8, seed=99)
    acc.push_many(_records(24, lane_id=0))
    acc.push_many(_records(24, lane_id=1))
    assert acc.has_pending_data(0)
    assert acc.has_pending_data(1)

    flushed = [rec for batch, _ in acc.flush(reset=True, lane_id=0) for rec in batch]

    assert all(rec.meta.lane_id == 0 for rec in flushed)
    assert not acc.has_pending_data(0), "lane 0 should be drained and reset"
    assert acc.has_pending_data(1), "lane 1 must be untouched by a lane-0 flush"


def test_warmup_block_per_lane_flush_only_drains_that_lane() -> None:
    acc = WarmupBlockAccumulator(buffer_size=64)
    acc.push_many(_records(3, lane_id=0))  # below first target (4): stays buffered
    acc.push_many(_records(3, lane_id=1))
    assert acc.has_pending_data(0)
    assert acc.has_pending_data(1)

    flushed = [rec for batch, _ in acc.flush(reset=True, lane_id=0) for rec in batch]

    assert [rec.meta.lane_id for rec in flushed] == [0, 0, 0]
    assert not acc.has_pending_data(0)
    assert acc.has_pending_data(1), "lane 1 must be untouched by a lane-0 flush"


def test_streaming_shuffle_per_lane_flush_restarts_ramp_without_touching_neighbor() -> (
    None
):
    """Lane-specific streaming flush restarts that lane only."""
    lane0 = _records(48, lane_id=0)
    lane1 = _records(48, lane_id=1)

    acc = StreamingShuffleAccumulator(buffer_size=8, seed=99)
    emitted: list[SampleRecord] = []
    phase1 = [rec for pair in zip(lane0[:24], lane1[:24]) for rec in pair]
    emitted += [rec for batch, _ in acc.push_many(phase1) for rec in batch]
    emitted += [rec for batch, _ in acc.flush(reset=True, lane_id=0) for rec in batch]
    assert not acc.has_pending_data(0)
    assert acc.has_pending_data(1), "lane 1's reservoir must survive a lane-0 flush"
    phase2 = [rec for pair in zip(lane0[24:], lane1[24:]) for rec in pair]
    emitted += [rec for batch, _ in acc.push_many(phase2) for rec in batch]
    emitted += [rec for batch, _ in acc.flush(reset=True) for rec in batch]

    by_lane: dict[int, list[str]] = defaultdict(list)
    for rec in emitted:
        by_lane[rec.meta.lane_id].append(str(rec.payload["text"]))

    def fresh(records: list[SampleRecord]) -> list[str]:
        ref = StreamingShuffleAccumulator(buffer_size=8, seed=99)
        return [str(rec.payload["text"]) for rec in _drain_accumulator(ref, records)]

    assert by_lane[0] == fresh(lane0[:24]) + fresh(lane0[24:])
    assert by_lane[1] == fresh(lane1)


def test_warmup_block_per_lane_flush_restarts_ramp_without_touching_neighbor() -> None:
    """Lane-specific warmup flush restarts that lane's block ramp only."""
    lane0 = _records(200, lane_id=0)
    lane1 = _records(200, lane_id=1)

    acc = WarmupBlockAccumulator(buffer_size=64, growth=1.5)
    ready = acc.push_many([r for pair in zip(lane0[:100], lane1[:100]) for r in pair])
    ready += acc.flush(reset=True, lane_id=0)
    assert not acc.has_pending_data(0)
    assert acc.has_pending_data(1), "lane 1's partial block must survive"
    ready += acc.push_many([r for pair in zip(lane0[100:], lane1[100:]) for r in pair])
    ready += acc.flush(reset=True)

    sizes: dict[int, list[int]] = defaultdict(list)
    seen: dict[int, list[str]] = defaultdict(list)
    for batch, _ in ready:
        lane = batch[0].meta.lane_id
        sizes[lane].append(len(batch))
        seen[lane].extend(str(rec.payload["text"]) for rec in batch)

    def fresh_sizes(records: list[SampleRecord]) -> list[int]:
        ref = WarmupBlockAccumulator(buffer_size=64, growth=1.5)
        return [len(b) for b, _ in ref.push_many(records) + ref.flush(reset=True)]

    pre, post = fresh_sizes(lane0[:100]), fresh_sizes(lane0[100:])
    assert post[0] == 4, "post-flush ramp restarts from the small initial block"
    assert sizes[0] == pre + post
    assert sizes[1] == fresh_sizes(lane1)
    assert seen[0] == [str(rec.payload["text"]) for rec in lane0]
    assert seen[1] == [str(rec.payload["text"]) for rec in lane1]
