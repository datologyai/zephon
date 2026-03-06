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
    ds: Dataset, buffer_size: int = 5, seed: int = 0
) -> Pipeline:
    """Build a deterministic inline pipeline with shuffle."""
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
    pipeline.options(
        deterministic=True,
        max_workers=1,
        default_stage_prefetch=0,
        mtp_mode=False,
    )
    return pipeline


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Cross-chunk shuffle buffer causes data duplication after resume when "
        "the replay cursor references an evicted chunk. "
        "Same root cause as the packing variant: "
        "see docs/_internal/cross_chunk_packing_bug.md for full analysis."
    ),
)
def test_shuffle_cross_chunk_data_correctness_after_checkpoint() -> None:
    """Checkpoint/restore must not duplicate samples from cross-chunk shuffle buffers.

    Setup (12 samples, chunk_size=4, buffer_size=5):

        Buffer 1: [s0, s1, s2, s3, s4]  -- chunk 0 (all 4) + chunk 1 (1)
        Buffer 2: [s5, s6, s7, s8, s9]  -- chunk 1 (3) + chunk 2 (2)
        Buffer 3: [s10, s11]            -- chunk 2 (2), flushed at end

    After consuming all records from buffer 1, all chunk 0 offsets are done
    so chunk 0 evicts.  The cursor at that point references whichever sample
    was consumed last; if it was a chunk 0 sample the cursor references an
    evicted chunk.  The test dynamically finds the first record after which
    the cursor references an evicted chunk and checkpoints there.
    """
    ds = mk_dataset("shuf", {0: 12})

    # -- baseline: full run without checkpoint ---------------------------------
    baseline_pipe = _make_shuffle_pipeline(ds)
    baseline_texts: list[str] = []
    for rec in baseline_pipe:
        assert isinstance(rec, SampleRecord)
        baseline_texts.append(rec.payload["text"])

    assert len(baseline_texts) == 12, f"Expected 12 records, got {len(baseline_texts)}"

    # -- run 1: consume until cursor references evicted chunk ------------------
    pipe1 = _make_shuffle_pipeline(ds)
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

    assert ckpt is not None, (
        "Could not trigger cross-chunk eviction — the shuffle order for this "
        "seed never leaves the cursor on an evicted chunk."
    )

    # Sanity: confirm cross-chunk eviction actually happened
    inflight_ckpt = ckpt.get("inflight", {}).get(
        0, ckpt.get("inflight", {}).get("0", {})
    )
    inflight_cids = sorted(int(c) for c in inflight_ckpt.keys())
    assert 0 not in inflight_cids, (
        f"Chunk 0 should have been evicted before checkpoint, "
        f"but inflight contains: {inflight_cids}"
    )

    # -- run 2: restore and drain ----------------------------------------------
    pipe2 = _make_shuffle_pipeline(ds)
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
