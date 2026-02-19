# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path
from queue import Empty

import pytest

from zephon.api import Pipeline as PublicPipeline
from zephon.io import Dataset, InMemoryShard
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def make_dataset(name: str, sample_count: int) -> Dataset:
    rows = [{"text": f"{name}-{i}"} for i in range(sample_count)]
    return Dataset.from_dict(name, {0: InMemoryShard(rows)})


def _extract_texts(item) -> list[str]:
    """Return a flat list of 'text' payloads from a SampleRecord or SampleBatch."""
    from zephon.core.constants import SampleBatch, SampleRecord

    if isinstance(item, SampleRecord):
        payload = item.payload
        assert isinstance(payload, dict)
        return [str(payload.get("text", ""))]
    assert isinstance(item, SampleBatch)
    texts: list[str] = []
    for record in item.records:
        payload = record.payload
        assert isinstance(payload, dict)
        texts.append(str(payload.get("text", "")))
    return texts


def consume_until(
    pipe: PublicPipeline,
    *,
    flat_limit: int | None = None,
    elem_limit: int | None = None,
    return_elems: bool = False,
) -> tuple[list, dict | None]:
    """
    Iterate a pipeline until a limit and collect outputs.
    - When return_elems=False, returns flat list[str] of texts.
    - When return_elems=True, returns a list[(lane_id, [texts...])].
    If a limit is provided, capture a checkpoint at the cut.
    Returns (collection, checkpoint_or_None).
    """
    from zephon.core.constants import SampleBatch, SampleRecord

    out: list = []
    ckpt: dict | None = None
    it = iter(pipe)
    try:
        for item in it:
            texts = _extract_texts(item)
            if return_elems:
                if isinstance(item, SampleRecord):
                    lane = int(item.meta.lane_id)
                else:
                    assert isinstance(item, SampleBatch)
                    lids = item.lane_ids
                    assert lids, "empty batch"
                    lane = int(lids[0])
                out.append((lane, texts))
                if elem_limit is not None and len(out) >= elem_limit:
                    ckpt = pipe.checkpoint()
                    break
            else:
                out.extend(texts)
                if flat_limit is not None and len(out) >= flat_limit:
                    ckpt = pipe.checkpoint()
                    break
    finally:
        it.close()
    return out, ckpt


# Small helper that lets us vary dp_degree/dp_group_id/mapping_strategy.
def _build_pipe_params(
    ds: Dataset,
    *,
    chunk_size: int,
    canonical_replicas: int,
    with_batch: bool,
    batch_size: int = 8,
    stage_prefetch: int = 0,
    final_prefetch: int = 0,
    seed: int = 7,
    world_size: int = 1,
    global_rank: int = 0,
    dp_degree: int = 1,
    dp_group_id: int = 0,
    mapping_strategy: str = "contiguous",
    allow_latency_flush_in_deterministic: bool = True,
    aggregate_dir: str | None = None,
    run_id: str | None = None,
    op_queue_capacity: int | None = None,
    mtp_mode: bool = False,
) -> PublicPipeline:
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=chunk_size,
        seed=seed,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipe = (
        PublicPipeline(work)
        .decode_text()
        ._delay(max_delay_ms=1.0, parallelism=4)
        .tokenize(
            tokenizer_id="__fallback__",
            parallelism=4,
            preserve_upstream_payload=True,
        )
    )
    if with_batch:
        pipe = pipe.batch(batch_size, drop_last=False)

    pipe = pipe.options(
        deterministic=True,
        canonical_replicas=canonical_replicas,
        world_size=world_size,
        global_rank=global_rank,
        dp_degree=dp_degree,
        dp_group_id=dp_group_id,
        mapping_strategy=mapping_strategy,
        default_stage_prefetch=stage_prefetch,
        prefetch_batches=final_prefetch,
        max_workers=8,
        allow_latency_flush_in_deterministic=allow_latency_flush_in_deterministic,
        aggregate_dir=aggregate_dir,
        mtp_mode=mtp_mode,
        **({"run_id": run_id} if run_id is not None else {}),
        **(
            {"op_queue_capacity": op_queue_capacity}
            if op_queue_capacity is not None
            else {}
        ),
    )
    return pipe


def rr_merge(
    per_rank: list[list], *, mode: str, limit: int, chunk_size: int | None = None
) -> list:
    """
    Round-robin merge across ranks.
    - mode='elem': take 1 element per rank per turn (works for str or (lane, texts)).
    - mode='chunk': take `chunk_size` elements per rank per turn (used for batching).
    Returns a flat list of elements; caller decides element shape.
    """
    iters = [iter(seq) for seq in per_rank]
    out: list = []
    finished = [False] * len(iters)

    if mode == "elem":
        while not all(finished) and len(out) < limit:
            for i, it in enumerate(iters):
                if finished[i]:
                    continue
                try:
                    out.append(next(it))
                    if len(out) >= limit:
                        break
                except StopIteration:
                    finished[i] = True
        return out

    assert mode == "chunk" and chunk_size is not None and chunk_size > 0
    while not all(finished) and len(out) < limit:
        for i, it in enumerate(iters):
            if finished[i]:
                continue
            taken = 0
            while taken < chunk_size and len(out) < limit:
                try:
                    out.append(next(it))
                    taken += 1
                except StopIteration:
                    finished[i] = True
                    break
    return out


def _is_cyclic_rotation_elems(
    a: list[tuple[int, list[str]]], b: list[tuple[int, list[str]]]
) -> bool:
    """
    True iff sequence `a` equals `b` up to a cyclic rotation at the *element* level.
    Element identity includes (lane_id, texts).
    """
    if len(a) != len(b):
        return False
    A = [(lane, tuple(txts)) for lane, txts in a]
    B = [(lane, tuple(txts)) for lane, txts in b]
    if not A:
        return True
    # search for a rotation K where A[i] == B[(i+K) % n]
    n = len(A)
    # anchor on A[0] to cut search space a bit
    candidates = [k for k in range(n) if B[k] == A[0]]
    for k in candidates:
        ok = True
        for i in range(1, n):
            if B[(k + i) % n] != A[i]:
                ok = False
                break
        if ok:
            return True
    return False


@pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
@pytest.mark.parametrize("lanes", [1, 2, 4])
@pytest.mark.parametrize("stage_prefetch,final_prefetch", [(0, 0), (4, 16)])
def test_resume_equivalence_no_batch(
    lanes: int, stage_prefetch: int, final_prefetch: int, mtp_mode: bool
) -> None:
    """Resume mid-run without batching across lane counts and prefetch settings."""
    ds = make_dataset("alpha", 256)
    chunk_size = 8

    kw = dict(mtp_mode=mtp_mode)
    baseline = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=lanes,
        with_batch=False,
        stage_prefetch=stage_prefetch,
        final_prefetch=final_prefetch,
        **kw,
    )
    baseline_flat, _ = consume_until(baseline)
    assert baseline_flat

    cut = len(baseline_flat) // 2

    p1 = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=lanes,
        with_batch=False,
        stage_prefetch=stage_prefetch,
        final_prefetch=final_prefetch,
        **kw,
    )
    prefix_flat, ckpt = consume_until(p1, flat_limit=cut)
    assert ckpt is not None
    assert prefix_flat == baseline_flat[:cut]

    p2 = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=lanes,
        with_batch=False,
        stage_prefetch=stage_prefetch,
        final_prefetch=final_prefetch,
        **kw,
    )
    p2.restore(ckpt)
    suffix_flat, _ = consume_until(p2)
    assert prefix_flat + suffix_flat == baseline_flat


@pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
@pytest.mark.parametrize("workers_a,workers_b", [(4, 8), (8, 2)])
def test_checkpoint_resume_with_workers_change_no_batch(
    workers_a: int, workers_b: int, mtp_mode: bool
) -> None:
    """
    No batching, single lane; resume with a different number of workers.
    Deterministic mode should make the outputs identical.
    """
    ds = make_dataset("alpha", 96)
    chunk_size = 8

    kw = dict(mtp_mode=mtp_mode)
    # Ground truth (any worker count is fine; use A for consistency)
    base = _build_pipe_params(
        ds,
        # f"crwcnb-{workers_a}-{workers_b}-baseline",
        chunk_size=chunk_size,
        canonical_replicas=1,
        with_batch=False,
        stage_prefetch=4,
        **kw,
    )
    base = base.options(max_workers=workers_a)
    baseline_flat, _ = consume_until(base)
    cut = len(baseline_flat) // 3 * 2  # cut at ~2/3

    p1 = _build_pipe_params(
        ds,
        # f"crwcnb-{workers_a}-{workers_b}-p1",
        chunk_size=chunk_size,
        canonical_replicas=1,
        with_batch=False,
        stage_prefetch=4,
        **kw,
    ).options(max_workers=workers_a)
    prefix_flat, ckpt = consume_until(p1, flat_limit=cut)
    assert ckpt is not None
    assert prefix_flat == baseline_flat[:cut]

    p2 = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=1,
        with_batch=False,
        stage_prefetch=4,
        **kw,
    ).options(max_workers=workers_b)
    p2.restore(ckpt)
    suffix_flat, _ = consume_until(p2)

    assert prefix_flat + suffix_flat == baseline_flat


@pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
@pytest.mark.parametrize("lanes", [1, 4])
@pytest.mark.parametrize("stage_prefetch,final_prefetch", [(0, 0), (4, 16)])
def test_resume_equivalence_with_batch(
    lanes: int, stage_prefetch: int, final_prefetch: int, mtp_mode: bool
) -> None:
    """Resume mid-run with batching across lane counts and prefetch settings."""
    ds = make_dataset("alpha", 512)
    chunk_size = 16  # divisible by batch_size
    batch_size = 8

    kw = dict(mtp_mode=mtp_mode)
    baseline = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=lanes,
        with_batch=True,
        batch_size=batch_size,
        stage_prefetch=stage_prefetch,
        final_prefetch=final_prefetch,
        **kw,
    )
    baseline_flat, _ = consume_until(baseline)
    assert baseline_flat

    # Cut on RR-turn boundary to avoid rotation ambiguities
    rr_turn = lanes * batch_size
    half = len(baseline_flat) // 2
    cut = (half // rr_turn) * rr_turn
    assert cut > 0 and (cut % rr_turn == 0)

    p1 = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=lanes,
        with_batch=True,
        batch_size=batch_size,
        stage_prefetch=stage_prefetch,
        final_prefetch=final_prefetch,
        **kw,
    )
    prefix_flat, ckpt = consume_until(p1, flat_limit=cut)
    assert ckpt is not None
    assert prefix_flat == baseline_flat[:cut]

    p2 = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=lanes,
        with_batch=True,
        batch_size=batch_size,
        stage_prefetch=stage_prefetch,
        final_prefetch=final_prefetch,
        **kw,
    )
    p2.restore(ckpt)
    suffix_flat, _ = consume_until(p2)

    assert prefix_flat + suffix_flat == baseline_flat


@pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
def test_checkpoint_records_inflight_pruned(mtp_mode: bool) -> None:
    """
    Sanity: When we checkpoint mid-run, inflight chunks recorded in the engine state
    must not include any chunk_id strictly older than the lane's current progress.
    """
    ds = make_dataset("alpha", 64)
    chunk_size = 8

    kw = dict(mtp_mode=mtp_mode)
    p1 = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=1,
        with_batch=False,
        stage_prefetch=8,  # allow multiple chunks in flight
        **kw,
    )
    # Consume a handful of records then checkpoint
    prefix_flat, ckpt = consume_until(p1, flat_limit=10)
    assert ckpt is not None and prefix_flat

    inflight = ckpt.get("inflight", {})
    progress = ckpt.get("progress", {})
    # Single lane (0), but keys are ints by construction
    assert 0 in progress
    cur_chunk = int(progress[0]["chunk_id"])

    if 0 in inflight:
        for cid_str in inflight[0].keys():
            cid = int(cid_str)
            assert cid >= cur_chunk, (
                f"inflight contains chunk {cid} older than progress {cur_chunk}"
            )


# ---- 2) Resume mid-chunk, no batch (explicitly cut off boundary)
@pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
def test_checkpoint_resume_mid_chunk_no_batch(mtp_mode: bool) -> None:
    ds = make_dataset("alpha", 200)
    chunk_size = 8

    kw = dict(mtp_mode=mtp_mode)
    baseline = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=2,
        with_batch=False,
        stage_prefetch=0,
        **kw,
    )
    baseline_flat, _ = consume_until(baseline)
    assert baseline_flat, "Baseline produced no output"

    # Intentionally not a multiple of chunk_size to cut mid-chunk
    cut = (len(baseline_flat) // 2) + 3
    assert cut % chunk_size != 0

    p1 = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=2,
        with_batch=False,
        stage_prefetch=0,
        **kw,
    )
    prefix_flat, ckpt = consume_until(p1, flat_limit=cut)
    assert ckpt is not None
    assert prefix_flat == baseline_flat[:cut]

    p2 = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=2,
        with_batch=False,
        stage_prefetch=0,
        **kw,
    )
    p2.restore(ckpt)
    suffix_flat, _ = consume_until(p2)

    assert prefix_flat + suffix_flat == baseline_flat


# ---- 3) Scale-down equivalence, no batch (simulate N ranks vs. 1 rank)
@pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
@pytest.mark.parametrize("with_batch", [False, True])
@pytest.mark.parametrize("stage_prefetch,final_prefetch", [(0, 0), (4, 16)])
def test_scale_down_equivalence_truth_checkpnts(
    with_batch: bool,
    stage_prefetch: int,
    final_prefetch: int,
    tmp_path: Path,
    mtp_mode: bool,
) -> None:
    """Compare N-rank merged stream to 1-rank truth across batch/prefetch modes."""
    N = 4
    if with_batch:
        ds = make_dataset("alpha", 256)
        chunk_size = 16
        batch_size = 8
    else:
        ds = make_dataset("alpha", 180)
        chunk_size = 8
        batch_size = None  # type: ignore[assignment]

    kw = dict(mtp_mode=mtp_mode)
    pipe_all = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=N,
        with_batch=with_batch,
        batch_size=(batch_size or 8),
        stage_prefetch=stage_prefetch,
        final_prefetch=final_prefetch,
        **kw,
    ).options(
        dp_degree=1,
        dp_group_id=0,
    )
    all_flat, _ = consume_until(pipe_all)
    total = len(all_flat)

    per_rank_flats: list[list[str]] = []
    for r in range(N):
        pipe = _build_pipe_params(
            ds,
            chunk_size=chunk_size,
            canonical_replicas=N,
            with_batch=with_batch,
            batch_size=(batch_size or 8),
            stage_prefetch=stage_prefetch,
            final_prefetch=final_prefetch,
            aggregate_dir=str(tmp_path),
            **kw,
        ).options(
            world_size=N,
            global_rank=r,
            dp_degree=N,
            dp_group_id=r,
            mtp_auto_checkpoint=False,
        )
        flat, _ = consume_until(pipe)
        per_rank_flats.append(flat)

    if with_batch:
        merged = rr_merge(
            per_rank_flats, mode="chunk", chunk_size=(batch_size or 8), limit=total
        )
    else:
        merged = rr_merge(per_rank_flats, mode="elem", limit=total)
    assert merged == all_flat


# ---- 6) Multiple resumes (same size segments), no batch
@pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
def test_multiple_resumes_same_size_no_batch_current_api(mtp_mode: bool) -> None:
    ds = make_dataset("alpha", 400)
    chunk_size = 8
    canonical_replicas = 4

    kw = dict(mtp_mode=mtp_mode)
    # Canonical baseline
    baseline = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=canonical_replicas,
        with_batch=False,
        stage_prefetch=0,
        **kw,
    )
    baseline_flat, _ = consume_until(baseline, flat_limit=240)
    assert baseline_flat

    seg = 80  # multiple of chunk_size * canonical_replicas
    collected: list[str] = []
    ckpt: dict | None = None

    for _ in range(3):
        pipe = _build_pipe_params(
            ds,
            chunk_size=chunk_size,
            canonical_replicas=canonical_replicas,
            with_batch=False,
            stage_prefetch=0,
            **kw,
        )
        if ckpt is not None:
            pipe.restore(ckpt)
        part, ckpt = consume_until(pipe, flat_limit=seg)
        collected.extend(part)

    assert collected == baseline_flat[: len(collected)]


# ---- 7) Multiple resizes across phases (ranks/strategy change), with batch
@pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
def test_multiple_resizes_equivalence_with_batch_truthchkpnts(
    tmp_path: Path, mtp_mode: bool
) -> None:
    """
    Stitch phases with different world sizes & mappings; contract is:
    - element-level RR across lanes
    - at each phase boundary the starting lane is unconstrained
    Therefore for each phase, the multi-rank merged stream should equal the
    single-rank "truth" for that phase up to a cyclic rotation (at element granularity).
    """
    ds = make_dataset("alpha", 512)
    chunk_size = 16
    batch_size = 8
    lanes = 4  # canonical replicas

    # Phase spec: (num_ranks, mapping_strategy, FLAT element budget for phase)
    # (values are multiples of batch_size so element counts are integral)
    phases = [
        (4, "contiguous", 64),
        (2, "interleaved", 64),
        (4, "interleaved", 128),
    ]

    kw = dict(mtp_mode=mtp_mode)
    ckpt: dict | None = None  # carry checkpoint across phases

    for phase_idx, (ranks, strat, flat_budget) in enumerate(phases):
        phase_elem_budget = flat_budget // batch_size
        assert phase_elem_budget > 0, (
            "phase budget must be a positive multiple of batch size"
        )

        # ---- Single-rank "truth" for this phase (owning all lanes) ----
        truth_pipe = _build_pipe_params(
            ds,
            chunk_size=chunk_size,
            canonical_replicas=lanes,
            with_batch=True,
            batch_size=batch_size,
            stage_prefetch=2,
            final_prefetch=2,
            run_id=f"mrrb-truth-p{phase_idx}",
            **kw,
        ).options(
            dp_degree=1,
            dp_group_id=0,
        )
        if ckpt is not None:
            truth_pipe.restore(ckpt)
        truth_phase_elems, ckpt_next = consume_until(
            truth_pipe, elem_limit=phase_elem_budget, return_elems=True
        )
        assert ckpt_next is not None

        # ---- Multi-rank simulation for this phase (merge ranks element-RR) ----
        per_rank_elems: list[list[tuple[int, list[str]]]] = []
        for r in range(ranks):
            rank_pipe = _build_pipe_params(
                ds,
                chunk_size=chunk_size,
                canonical_replicas=lanes,
                with_batch=True,
                batch_size=batch_size,
                stage_prefetch=0,
                final_prefetch=0,
                aggregate_dir=str(tmp_path),
                run_id=f"mrrb-sim-p{phase_idx}",
                **kw,
            ).options(
                world_size=ranks,
                global_rank=r,
                dp_degree=ranks,
                dp_group_id=r,
                mapping_strategy=strat,
                mtp_auto_checkpoint=False,
            )
            if ckpt is not None:
                rank_pipe.restore(ckpt)
            per_rank_elems.append(consume_until(rank_pipe, return_elems=True)[0])

        merged_phase = rr_merge(
            per_rank_elems, mode="elem", limit=len(truth_phase_elems)
        )

        # Allow cyclic rotation at element level for this phase
        assert _is_cyclic_rotation_elems(merged_phase, truth_phase_elems)

        # advance
        ckpt = ckpt_next


def _flatten_elems(elems: list[tuple[int, list[str]]]) -> list[str]:
    """Flatten [(lane, [texts...]), ...] to a flat list of texts."""
    out: list[str] = []
    for _, texts in elems:
        out.extend(texts)
    return out


@pytest.mark.parametrize(
    "chunk_size",
    [
        pytest.param(7, id="cs=7"),
        pytest.param(10, id="cs=10"),
        pytest.param(16, id="cs=16"),
        pytest.param(32, id="cs=32"),
    ],
)
def test_resize_and_microbatch_change_equivalence_current_api_using_truthcheckpoints(
    chunk_size: int, tmp_path: Path
) -> None:
    """
    Change both DP (num_ranks) and microbatch size across phases, while resuming.

    Contract: for each phase, the multi-rank merged stream (element-level RR)
    equals the single-rank "truth" up to a cyclic rotation at the element level.

    We HAVE TO use truth checkpoints here, otherwise we need to do multiprocessing. We also have a MP test at the end of this file.
    """
    ds = make_dataset("alpha", 512)
    lanes = 4

    # Phase spec: (num_ranks, mapping_strategy, batch_size, FLAT element budget)
    # flat budgets are multiples of that phase's batch_size so elem counts are integral
    phases: list[tuple[int, str, int, int]] = [
        (4, "contiguous", 8, 64),  # 8 batches this phase
        (2, "interleaved", 4, 64),  # 16 batches this phase (halved microbatch)
    ]
    total_flat_budget = sum(b for _, _, _, b in phases)

    # ---- Global oracle: single-rank, single pass, NO batching, no checkpoint ----
    oracle_pipe = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=lanes,
        with_batch=False,  # <— important: per-element oracle
        stage_prefetch=2,
        final_prefetch=2,
        aggregate_dir=str(tmp_path),
        run_id=f"resize-mb-cs{chunk_size}-truth",
    ).options(
        dp_degree=1,
        dp_group_id=0,
    )
    # First N elements of the true canonical stream (no resume, no batching)
    oracle_flat, _ = consume_until(
        oracle_pipe, flat_limit=total_flat_budget, return_elems=False
    )
    assert len(oracle_flat) == total_flat_budget

    ckpt: dict | None = None
    observed_flat: list[str] = []

    for phase_idx, (ranks, strat, batch_size, flat_budget) in enumerate(phases):
        phase_elem_budget = flat_budget // batch_size
        assert phase_elem_budget > 0 and (flat_budget % batch_size == 0)

        # ---- Single-rank truth for this phase ----
        truth_pipe = _build_pipe_params(
            ds,
            chunk_size=chunk_size,
            canonical_replicas=lanes,
            with_batch=True,
            batch_size=batch_size,
            stage_prefetch=2,
            final_prefetch=2,
            aggregate_dir=str(tmp_path),
            run_id=f"resize-mb-cs{chunk_size}-ph{phase_idx}-truth",
        ).options(
            dp_degree=1,
            dp_group_id=0,
        )
        if ckpt is not None:
            truth_pipe.restore(ckpt)
        truth_phase_elems, ckpt_next = consume_until(
            truth_pipe, elem_limit=phase_elem_budget, return_elems=True
        )
        assert ckpt_next is not None

        # ---- Multi-rank per-rank sequences for this phase ----
        per_rank_elems: list[list[tuple[int, list[str]]]] = []
        for r in range(ranks):
            rank_pipe = _build_pipe_params(
                ds,
                chunk_size=chunk_size,
                canonical_replicas=lanes,
                with_batch=True,
                batch_size=batch_size,
                stage_prefetch=0,
                final_prefetch=0,
                aggregate_dir=str(tmp_path),
                run_id=f"resize-mb-cs{chunk_size}-ph{phase_idx}-simul",
            ).options(
                world_size=ranks,
                global_rank=r,
                dp_degree=ranks,
                dp_group_id=r,
                mapping_strategy=strat,
            )
            if ckpt is not None:
                rank_pipe.restore(ckpt)
            per_rank_elems.append(consume_until(rank_pipe, return_elems=True)[0])

        merged_phase = rr_merge(
            per_rank_elems, mode="elem", limit=len(truth_phase_elems)
        )

        # Allow cyclic rotation at element level for this phase
        assert _is_cyclic_rotation_elems(merged_phase, truth_phase_elems)
        observed_flat.extend(_flatten_elems(merged_phase))

        # advance to next phase
        ckpt = ckpt_next

    # Sanity: consumed exactly what we asked for across phases
    assert len(observed_flat) == total_flat_budget

    # Global check: no duplicate replays across resume boundaries.
    # Compare as multisets to ignore harmless reordering from batching.
    assert Counter(observed_flat) == Counter(oracle_flat)


@pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
@pytest.mark.parametrize("where", ["boundary", "within"])  # checkpoint location
def test_checkpoint_at_chunk_boundary_and_within_with_batch_current_api(
    where: str,
    mtp_mode: bool,
) -> None:
    """
    Single-lane, batched pipeline: resume from a checkpoint taken exactly at a
    chunk boundary vs within a chunk. Both must reproduce the baseline.
    """
    ds = make_dataset("alpha", 512)
    chunk_size = 16
    batch_size = 8  # divides chunk_size so boundary in terms of batches is integral

    kw = dict(mtp_mode=mtp_mode)
    # Canonical baseline (full run)
    baseline = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=1,
        with_batch=True,
        batch_size=batch_size,
        stage_prefetch=0,
        final_prefetch=0,
        **kw,
    )
    baseline_flat, _ = consume_until(baseline)
    assert baseline_flat

    # Number of batches per chunk in single lane
    batches_per_chunk = chunk_size // batch_size
    assert batches_per_chunk > 0
    # Choose a modest number of full chunks to advance before checkpointing
    full_chunks = 3
    elem_limit = full_chunks * batches_per_chunk
    if where == "within":
        elem_limit -= 1  # cut one batch before the boundary
        assert elem_limit > 0

    # Phase 1: run until elem_limit batches, capture checkpoint and prefix
    p1 = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=1,
        with_batch=True,
        batch_size=batch_size,
        stage_prefetch=0,
        final_prefetch=0,
        **kw,
    )
    prefix_elems, ckpt = consume_until(p1, elem_limit=elem_limit, return_elems=True)
    assert ckpt is not None and prefix_elems
    # Flatten texts from (lane, [texts...])
    prefix_flat = [t for _, texts in prefix_elems for t in texts]
    assert prefix_flat == baseline_flat[: len(prefix_flat)]

    # Phase 2: resume and drain the rest
    p2 = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=1,
        with_batch=True,
        batch_size=batch_size,
        stage_prefetch=0,
        final_prefetch=0,
        **kw,
    )
    p2.restore(ckpt)
    suffix_flat, _ = consume_until(p2)

    assert prefix_flat + suffix_flat == baseline_flat


@pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
def test_scale_down_then_up_with_microbatch_change_current_api(
    tmp_path: Path, mtp_mode: bool
) -> None:
    """
    Three phases with resume:
    - Phase A: 4 ranks, microbatch 8
    - Phase B: 2 ranks, microbatch 4 (scale down + halve microbatch)
    - Phase C: 4 ranks, microbatch 8 (scale up + restore microbatch)
    For each phase, compare merged multi-rank to single-rank truth up to a cyclic rotation.
    """
    ds = make_dataset("alpha", 512)
    chunk_size = 16
    lanes = 4

    phases: list[tuple[int, str, int, int]] = [
        (4, "contiguous", 8, 64),
        (2, "interleaved", 4, 64),
        (4, "contiguous", 8, 64),
    ]

    kw = dict(mtp_mode=mtp_mode)
    ckpt: dict | None = None

    for phase_idx, (ranks, strat, batch_size, flat_budget) in enumerate(phases):
        phase_elem_budget = flat_budget // batch_size
        assert phase_elem_budget > 0 and (flat_budget % batch_size == 0)

        # Single-rank truth for this phase
        truth = _build_pipe_params(
            ds,
            chunk_size=chunk_size,
            canonical_replicas=lanes,
            with_batch=True,
            batch_size=batch_size,
            stage_prefetch=2,
            final_prefetch=2,
            aggregate_dir=str(tmp_path),
            run_id=f"scaleudownnomp-p{phase_idx}-truth",
            **kw,
        ).options(
            dp_degree=1,
            dp_group_id=0,
        )
        if ckpt is not None:
            truth.restore(ckpt)
        truth_elems, ckpt_next = consume_until(
            truth, elem_limit=phase_elem_budget, return_elems=True
        )
        assert ckpt_next is not None

        # Multi-rank per-rank sequences for this phase
        per_rank_elems: list[list[tuple[int, list[str]]]] = []
        for r in range(ranks):
            rp = _build_pipe_params(
                ds,
                chunk_size=chunk_size,
                canonical_replicas=lanes,
                with_batch=True,
                batch_size=batch_size,
                stage_prefetch=0,
                final_prefetch=0,
                aggregate_dir=str(tmp_path),
                run_id=f"scaleudownnomp-p{phase_idx}-simulation",
                **kw,
            ).options(
                world_size=ranks,
                global_rank=r,
                dp_degree=ranks,
                dp_group_id=r,
                mapping_strategy=strat,
                mtp_auto_checkpoint=False,
            )
            if ckpt is not None:
                rp.restore(ckpt)
            per_rank_elems.append(consume_until(rp, return_elems=True)[0])

        merged = rr_merge(per_rank_elems, mode="elem", limit=len(truth_elems))
        assert _is_cyclic_rotation_elems(merged, truth_elems)

        ckpt = ckpt_next


####### Multiprocessing tests without truth checkpoints


def _truth_windows_no_dl(
    *,
    ds: Dataset,
    run_id: str,
    canonical_replicas: int,
    chunk_size: int,
    with_batch: bool,
    microbatch_size: int,
    global_batch_size: int,
    total_windows: int,
    op_queue_capacity: int | None = None,
) -> list[list[str]]:
    """Single-rank 'truth' windows — no checkpoints, just iterate."""
    pipe = _build_pipe_params(
        ds,
        with_batch=with_batch,
        batch_size=microbatch_size,
        chunk_size=chunk_size,
        canonical_replicas=canonical_replicas,
        dp_degree=1,
        dp_group_id=0,
        mapping_strategy="contiguous",
        aggregate_dir=None,  # single rank → auto tmp dir
        run_id=run_id,
        op_queue_capacity=op_queue_capacity,
    )
    wins: list[list[str]] = []
    it = iter(pipe)
    try:
        for _ in range(total_windows):
            buf: list[str] = []
            while len(buf) < global_batch_size:
                buf.extend(_extract_texts(next(it)))
            wins.append(buf)
    finally:
        del it
    return wins


def _rank_worker_proc(
    rank: int,
    ranks: int,
    run_id: str,
    *,
    ds: Dataset,
    canonical_replicas: int,
    with_batch: bool,
    microbatch_size: int,
    chunk_size: int,
    acc_steps: int,
    windows: int,
    mapping_strategy: str,
    tmp_path_str: str,  # shared dir only when ranks > 1
    start_ckpt: dict | None,
    out_q: "mp.Queue",  # emits (rank, window_idx, [texts...])
    win_barrier: "mp.Barrier",  # sync at window boundaries
    ckpt_barrier: "mp.Barrier",
    ckpt_q: "mp.Queue",  # emits (rank, merged_ckpt)
    op_queue_capacity: int | None = None,
) -> None:
    pipe = _build_pipe_params(
        ds,
        with_batch=with_batch,
        batch_size=microbatch_size,
        chunk_size=chunk_size,
        canonical_replicas=canonical_replicas,
        world_size=ranks,
        global_rank=rank,
        dp_degree=ranks,
        dp_group_id=rank,
        mapping_strategy=mapping_strategy,
        aggregate_dir=tmp_path_str,
        run_id=run_id,
        op_queue_capacity=op_queue_capacity,
    )
    # Already running in a spawned subprocess worker — disable nested MTP mode.
    pipe = pipe.options(mtp_mode=False)
    if start_ckpt is not None:
        pipe.restore(start_ckpt)

    it = iter(pipe)
    try:
        for w in range(windows):
            buf: list[str] = []
            for _ in range(acc_steps):
                buf.extend(_extract_texts(next(it)))
            # win_barrier.wait()
            out_q.put((rank, w, buf), timeout=45.0)
            win_barrier.wait()
        ckpt_barrier.wait()
        merged = pipe.checkpoint()  # triggers file aggregation for multi-rank
        ckpt_q.put((rank, merged))
    finally:
        ckpt_barrier.wait(timeout=45.0)  # wait that everybody is done at the end
        try:
            it.close()
        except Exception:
            pass


def _phase_run_and_checkpoint_mp(
    *,
    ds: Dataset,
    run_id: str,
    ranks: int,
    canonical_replicas: int,
    with_batch: bool,
    microbatch_size: int,
    chunk_size: int,
    acc_steps: int,
    windows: int,
    mapping_strategy: str,
    tmp_path: Path | None,
    start_ckpt: dict | None,
    op_queue_capacity: int | None = None,
) -> tuple[list[list[str]], dict]:
    """
    Run one phase concurrently across `ranks`; return (windows_out, merged_ckpt).

    NOTE ON ORDERING W/ MULTI-PRODUCER QUEUE:
      We synchronize workers with a Barrier so they *call* put() for window w
      before proceeding to window w+1. However, multiprocessing.Queue is only
      FIFO per producer. Each process has a local feeder/buffer; a fast producer
      can enqueue (and have its feeder flush) items for window w+1 before a slow
      producer’s window w item has actually reached the shared queue. The parent
      can therefore observe (fast, w+1) before (slow, w).

      To make the test deterministic, we demultiplex by window index in the
      parent: we keep a small stash of out-of-window items and only assemble
      window w when we've collected exactly `ranks` contributions for w.
    """
    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue(maxsize=max(4 * ranks, ranks * windows) + 16)
    ckpt_q: mp.Queue = ctx.Queue()
    win_barrier = ctx.Barrier(ranks)
    ckpt_barrier = ctx.Barrier(ranks)

    procs: list[mp.Process] = []
    for r in range(ranks):
        p = ctx.Process(
            target=_rank_worker_proc,
            args=(r, ranks, run_id),
            kwargs=dict(
                ds=ds,
                canonical_replicas=canonical_replicas,
                with_batch=with_batch,
                microbatch_size=microbatch_size,
                chunk_size=chunk_size,
                acc_steps=acc_steps,
                windows=windows,
                mapping_strategy=mapping_strategy,
                tmp_path_str=(str(tmp_path) if (tmp_path and ranks > 1) else None),
                start_ckpt=start_ckpt,
                out_q=out_q,
                win_barrier=win_barrier,
                ckpt_barrier=ckpt_barrier,
                ckpt_q=ckpt_q,
                op_queue_capacity=op_queue_capacity,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    try:
        # collect per-window outputs from all ranks; concatenate in rank order
        windows_out: list[list[str]] = []

        # Map: window_idx -> list[(rankid, buf)]
        stash: dict[int, list[tuple[int, list[str]]]] = defaultdict(list)

        # ---- Collect per-window outputs from all ranks (order-agnostic) ----
        for w in range(windows):
            per_window = stash.pop(w, [])
            while len(per_window) < ranks:
                try:
                    rr, w_idx, buf = out_q.get(timeout=60.0)
                except Empty:
                    crashed = {
                        p.pid: p.exitcode
                        for p in procs
                        if not p.is_alive() and p.exitcode not in (None, 0)
                    }
                    if crashed:
                        raise RuntimeError(
                            f"Rank worker crashed while collecting window {w}: {crashed}"
                        )
                    continue
                if w_idx == w:
                    # Append directly to the current window's accumulator.
                    per_window.append((rr, buf))
                else:
                    # Stash for a future window.
                    stash[w_idx].append((rr, buf))

            # (optional) stable rank order for reproducible merges
            per_window.sort(key=lambda x: x[0])

            merged: list[str] = []
            for _, chunk in per_window:
                merged.extend(chunk)
            windows_out.append(merged)

        merged_ckpt = None
        for _ in range(ranks):
            try:
                _rid, ckpt = ckpt_q.get(timeout=60.0)
            except Empty:
                crashed = {
                    p.pid: p.exitcode
                    for p in procs
                    if not p.is_alive() and p.exitcode not in (None, 0)
                }
                if crashed:
                    raise RuntimeError(
                        f"Rank worker crashed before emitting checkpoint: {crashed}"
                    )
                raise
            if merged_ckpt is None:
                merged_ckpt = ckpt
        assert merged_ckpt is not None
        return windows_out, merged_ckpt
    finally:
        for p in procs:
            p.join(timeout=60)
        for p in procs:
            if p.is_alive():
                p.terminate()


# ---------- the one-for-one torchdata-style test (no DataLoader) ----------


@pytest.mark.parametrize("op_queue_capacity", [4, 256])
def test_mp_scale_down_then_up_with_microbatch_change_no_dataloader(
    tmp_path: Path, op_queue_capacity: int
) -> None:
    """
    Mirror tests/integration/test_dataloader_elasticity.py but without the DataLoader:
      Phase A: ranks=4, mapping=contiguous,  micro=8, acc=2  → GLOBAL=64
      Phase B: ranks=2, mapping=interleaved, micro=4, acc=8  → GLOBAL=64
      Phase C: ranks=4, mapping=contiguous,  micro=8, acc=2  → GLOBAL=64

    - Build 'truth' windows once from a single-rank pipeline (no checkpoints).
    - Run each phase under multiprocessing, take a merged checkpoint, and resume.
    - Compare each MP window to the truth window using multiset equality.
    """
    GLOBAL = 64
    WINDOWS_PER_PHASE = 4
    TOTAL_WINDOWS = WINDOWS_PER_PHASE * 3

    ds = make_dataset("alpha", 4096)
    canonical_lanes = 4
    chunk_size = 16  # divisible by micro=8 and 4 → deterministic resume safety

    # ---- Truth windows (single rank, no checkpoints) ----
    truth_windows = _truth_windows_no_dl(
        ds=ds,
        run_id="mpscaling-truth",
        canonical_replicas=canonical_lanes,
        chunk_size=chunk_size,
        with_batch=True,
        microbatch_size=8,  # truth microbatch doesn't matter, we just buffer GLOBAL=64
        global_batch_size=GLOBAL,
        total_windows=TOTAL_WINDOWS,
        op_queue_capacity=op_queue_capacity,
    )
    assert len(truth_windows) == TOTAL_WINDOWS

    phases = [
        (4, "contiguous", 8, 2),
        (2, "interleaved", 4, 8),
        (4, "contiguous", 8, 2),
    ]

    got_all: list[list[str]] = []
    start_ckpt: dict | None = None
    for idx, (ranks, mapping, micro, acc) in enumerate(phases):
        phase_tmp = tmp_path / f"phase_{idx}"
        phase_tmp.mkdir(parents=True, exist_ok=True)
        phase_windows, start_ckpt = _phase_run_and_checkpoint_mp(
            ds=ds,
            run_id=f"mpscaling-{idx}",
            ranks=ranks,
            canonical_replicas=canonical_lanes,
            with_batch=True,
            microbatch_size=micro,
            chunk_size=chunk_size,
            acc_steps=acc,
            windows=WINDOWS_PER_PHASE,
            mapping_strategy=mapping,
            tmp_path=phase_tmp,  # required only because ranks>1 in all phases here
            start_ckpt=start_ckpt,
            op_queue_capacity=op_queue_capacity,
        )
        got_all.extend(phase_windows)

    # ---- Compare window-by-window to truth (multiset equality) ----
    assert len(got_all) == TOTAL_WINDOWS
    for w_got, w_truth in zip(got_all, truth_windows):
        assert len(w_got) == len(w_truth) == GLOBAL
        assert Counter(w_got) == Counter(w_truth)
