from typing import Any

import pytest

from tests.zephon.ops.conftest import mk_dataset
from zephon.api.pipeline import Pipeline
from zephon.core.children import spawn_child
from zephon.core.constants import SampleMeta, SampleRecord
from zephon.core.op_base import OpContext
from zephon.io.dataset import Dataset
from zephon.ops.shuffle_buffer import ShuffleBuffer
from zephon.work.static_mixture import StaticMixtureWorkSource


def _records(n: int) -> list[SampleRecord]:
    return [
        SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, i), lane_id=0, chunk_id=0, chunk_offset=i),
            payload={"text": str(i)},
        )
        for i in range(n)
    ]


def test_shuffle_buffer_is_deterministic() -> None:
    recs1 = _records(10)
    recs2 = _records(10)
    original_order = [r.meta.cursor for r in recs1]
    op1 = ShuffleBuffer(buffer_size=3, seed=123)
    op2 = ShuffleBuffer(buffer_size=3, seed=123)
    ctx = OpContext({"record_node_metrics": lambda *args, **kwargs: None})
    op1.setup(ctx, 0, "s", 0, False)
    op2.setup(ctx, 0, "s", 0, False)

    out1 = op1.process_many(recs1)
    out2 = op2.process_many(recs2)

    assert [r.meta.cursor for r in out1] == [r.meta.cursor for r in out2]
    assert [r.meta.cursor for r in out1] != original_order


def test_shuffle_buffer_flushes_tail() -> None:
    recs = _records(2)
    op = ShuffleBuffer(buffer_size=4, seed=7)
    ctx = OpContext({"record_node_metrics": lambda *args, **kwargs: None})
    op.setup(ctx, 0, "s", 0, False)

    # buffer smaller than window → no immediate output from process_many
    shuffled = op.process_many(recs)
    assert sorted(r.meta.chunk_offset for r in shuffled) == [0, 1]


def test_shuffle_buffer_keeps_closer_after_non_closer_for_same_base() -> None:
    base = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0, chunk_offset=5)
    first = spawn_child(base, 0, is_last_child=False)
    last = spawn_child(base, 1, is_last_child=True)
    recs = [SampleRecord(meta=first, payload={}), SampleRecord(meta=last, payload={})]

    op = ShuffleBuffer(buffer_size=2, seed=99)
    ctx = OpContext({"record_node_metrics": lambda *args, **kwargs: None})
    op.setup(ctx, 0, "s", 0, False)

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
    pipeline.shuffle(buffer_size=buffer_size, seed=seed)
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
