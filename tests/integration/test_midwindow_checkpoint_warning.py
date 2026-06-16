# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the mid-window checkpoint warning (observability).

A "global window" is one delivered batch per canonical lane. Checkpoints cut
mid-window (unequal per-lane emitted-batch counts) and resumed into a
DIFFERENT topology permute the window remainder — a documented non-goal of
``_refresh_rr_from_progress`` (no samples are lost or duplicated). These
tests cover the detection-only counters and warnings:

* window-ALIGNED cuts stay silent,
* MID-window cuts warn at checkpoint time,
* resuming a mid-window checkpoint into a different topology warns at load
  time while the existing full-drain multiset guarantee still holds.

All pipelines run inline (no MTP) so warnings surface in the test process.
"""

import warnings
from collections import Counter
from pathlib import Path

import pytest

from tests.integration.test_elastic_continuation import (
    consume_until,
    make_dataset,
)
from zephon.api import Pipeline as PublicPipeline
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration

LANES = 4
CHUNK_SIZE = 16
BATCH_SIZE = 8


def _build_pipe(
    *,
    with_shuffle: bool,
    sample_count: int = 512,
    world_size: int = 1,
    global_rank: int = 0,
    dp_degree: int = 1,
    dp_group_id: int = 0,
    aggregate_dir: str | None = None,
) -> PublicPipeline:
    ds = make_dataset("alpha", sample_count)
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=CHUNK_SIZE,
        seed=7,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipe = PublicPipeline(work).decode_text()
    if with_shuffle:
        # Non-monotone pipeline: tail counting must stay exact downstream of
        # the shuffle (LanePtr progress is only approximate on this path).
        pipe = pipe.shuffle(buffer_size=16, seed=1234)
    pipe = pipe.tokenize(
        tokenizer_id="__fallback__",
        field="text",
        parallelism=2,
        preserve_upstream_payload=True,
    ).batch(BATCH_SIZE, drop_last=False)
    return pipe.options(
        deterministic=True,
        canonical_replicas=LANES,
        world_size=world_size,
        global_rank=global_rank,
        dp_degree=dp_degree,
        dp_group_id=dp_group_id,
        default_stage_prefetch=0,
        prefetch_batches=0,
        max_workers=4,
        aggregate_dir=aggregate_dir,
        mtp_mode=False,
    )


def _midwindow_warnings(recwarn: pytest.WarningsRecorder) -> list:
    return [w for w in recwarn.list if "mid-window" in str(w.message)]


@pytest.mark.parametrize("with_shuffle", [False, True], ids=["plain", "shuffle"])
def test_window_aligned_checkpoint_no_warning(
    with_shuffle: bool, recwarn: pytest.WarningsRecorder
) -> None:
    """Cutting at a multiple of `lanes` batches is window-aligned: silent."""
    pipe = _build_pipe(with_shuffle=with_shuffle)
    cut = 2 * LANES  # 8 batches → 2 per lane
    elems, ckpt = consume_until(pipe, elem_limit=cut, return_elems=True)
    assert ckpt is not None and len(elems) == cut
    assert ckpt["lane_emitted"] == dict.fromkeys(range(LANES), cut // LANES)
    assert not _midwindow_warnings(recwarn)


@pytest.mark.parametrize("with_shuffle", [False, True], ids=["plain", "shuffle"])
def test_mid_window_checkpoint_warns(with_shuffle: bool) -> None:
    """Cutting after 5 batches with 4 lanes leaves counts (2,1,1,1): warn."""
    pipe = _build_pipe(with_shuffle=with_shuffle)
    cut = LANES + 1  # not a multiple of LANES
    with pytest.warns(RuntimeWarning, match=r"Checkpoint taken mid-window"):
        elems, ckpt = consume_until(pipe, elem_limit=cut, return_elems=True)
    assert ckpt is not None and len(elems) == cut
    # Counters are exact per delivered batch at the tail.
    expected = Counter(lane for lane, _ in elems)
    assert ckpt["lane_emitted"] == dict(expected)
    assert sorted(ckpt["lane_emitted"].values()) == [1, 1, 1, 2]


def test_counters_exact_across_resume_no_double_count_shuffle() -> None:
    """Resume from a mid-window cut under a ShuffleBuffer: replayed-and-dropped
    records must not increment the counters — only newly delivered batches do."""
    cut = LANES + 1
    p1 = _build_pipe(with_shuffle=True)
    with pytest.warns(RuntimeWarning, match="mid-window"):
        elems1, ckpt1 = consume_until(p1, elem_limit=cut, return_elems=True)
    assert ckpt1 is not None
    counts1 = Counter(lane for lane, _ in elems1)
    assert ckpt1["lane_emitted"] == dict(counts1)
    assert sum(ckpt1["lane_emitted"].values()) == cut

    p2 = _build_pipe(with_shuffle=True)
    p2.restore(ckpt1)
    # Same topology: the load itself must not warn. Consuming 3 more batches
    # re-aligns the window (5 + 3 = 8 = 2 per lane), so the second checkpoint
    # must be silent too — proof the counters carried over exactly.
    extra = 3
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        elems2, ckpt2 = consume_until(p2, elem_limit=extra, return_elems=True)
    assert not [w for w in caught if "mid-window" in str(w.message)]
    assert ckpt2 is not None and len(elems2) == extra

    # Exact continuation: checkpointed values + one increment per newly
    # delivered batch, with no contribution from the replayed prefix.
    expected = counts1 + Counter(lane for lane, _ in elems2)
    assert ckpt2["lane_emitted"] == dict(expected)
    assert sum(ckpt2["lane_emitted"].values()) == cut + extra
    # The second cut at 5 + 3 = 8 batches is window-aligned again (2 per
    # lane) — asserted directly so the no-warning check above cannot pass
    # vacuously with broken counters.
    assert sorted(ckpt2["lane_emitted"].values()) == [2] * LANES


@pytest.mark.parametrize("with_shuffle", [False, True], ids=["plain", "shuffle"])
def test_mid_window_resume_into_different_topology_warns(
    with_shuffle: bool, tmp_path: Path
) -> None:
    """Loading a mid-window checkpoint into a different topology fires the
    load-time warning, and the existing guarantee still holds: the full drain
    across the new topology preserves the global multiset (no samples lost or
    duplicated)."""
    # Baseline: single rank owning all lanes, full drain.
    baseline_flat, _ = consume_until(_build_pipe(with_shuffle=with_shuffle))
    assert baseline_flat

    # Phase 1: single rank, cut mid-window.
    p1 = _build_pipe(with_shuffle=with_shuffle)
    with pytest.warns(RuntimeWarning, match="Checkpoint taken mid-window"):
        prefix_elems, ckpt = consume_until(p1, elem_limit=LANES + 1, return_elems=True)
    assert ckpt is not None
    prefix_flat = [t for _, texts in prefix_elems for t in texts]

    # Phase 2: resume into a DIFFERENT topology (1 rank → 2 ranks).
    per_rank_flat: list[str] = []
    for rank in range(2):
        rp = _build_pipe(
            with_shuffle=with_shuffle,
            world_size=2,
            global_rank=rank,
            dp_degree=2,
            dp_group_id=rank,
            aggregate_dir=str(tmp_path),
        )
        rp.restore(ckpt)
        # load_state_dict runs lazily at iteration start, so the warning
        # surfaces while consuming.
        with pytest.warns(RuntimeWarning, match="Resuming a mid-window checkpoint"):
            flat, _ = consume_until(rp)
        per_rank_flat.extend(flat)

    # Existing guarantee: full-drain multiset equality across the resume.
    assert Counter(prefix_flat + per_rank_flat) == Counter(baseline_flat)


def test_window_aligned_resume_into_different_topology_silent(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    """A window-aligned checkpoint resumed into a different topology is the
    supported elastic-continuation flow — it must stay silent."""
    p1 = _build_pipe(with_shuffle=False)
    _, ckpt = consume_until(p1, elem_limit=2 * LANES, return_elems=True)
    assert ckpt is not None

    rp = _build_pipe(
        with_shuffle=False,
        world_size=2,
        global_rank=0,
        dp_degree=2,
        dp_group_id=0,
        aggregate_dir=str(tmp_path),
    )
    rp.restore(ckpt)
    flat, _ = consume_until(rp)
    assert flat
    assert not _midwindow_warnings(recwarn)
