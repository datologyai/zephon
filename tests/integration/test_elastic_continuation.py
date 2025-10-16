# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

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
        return [str(item.payload.get("text", ""))]
    assert isinstance(item, SampleBatch)
    return [str(r.payload.get("text", "")) for r in item.records]


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

    pipe._ensure()
    assert pipe._engine is not None
    engine = pipe._engine

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
                    ckpt = engine.state_dict()
                    break
            else:
                out.extend(texts)
                if flat_limit is not None and len(out) >= flat_limit:
                    ckpt = engine.state_dict()
                    break
    finally:
        engine.close()
    return out, ckpt


# Small helper that lets us vary num_ranks/physical_rank/mapping_strategy.
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
    num_ranks: int = 1,
    physical_rank: int = 0,
    mapping_strategy: str = "contiguous",
    allow_latency_flush_in_deterministic: bool = True,
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
        .tokenize(tokenizer_id="__fallback__", parallelism=4)
    )
    if with_batch:
        pipe = pipe.batch(batch_size, drop_last=False)

    pipe = pipe.options(
        deterministic=True,
        canonical_replicas=canonical_replicas,
        num_ranks=num_ranks,
        physical_rank=physical_rank,
        mapping_strategy=mapping_strategy,
        default_stage_prefetch=stage_prefetch,
        prefetch_batches=final_prefetch,
        max_workers=8,
        allow_latency_flush_in_deterministic=allow_latency_flush_in_deterministic,
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


@pytest.mark.parametrize("lanes", [1, 2, 4])
@pytest.mark.parametrize("stage_prefetch,final_prefetch", [(0, 0), (4, 16)])
def test_resume_equivalence_no_batch(
    lanes: int, stage_prefetch: int, final_prefetch: int
) -> None:
    """Resume mid-run without batching across lane counts and prefetch settings."""
    ds = make_dataset("alpha", 256)
    chunk_size = 8

    baseline = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=lanes,
        with_batch=False,
        stage_prefetch=stage_prefetch,
        final_prefetch=final_prefetch,
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
    )
    p2._ensure()
    assert p2._engine is not None
    p2._engine.load_state_dict(ckpt, replay=True)
    suffix_flat, _ = consume_until(p2)
    assert prefix_flat + suffix_flat == baseline_flat


@pytest.mark.parametrize("workers_a,workers_b", [(4, 8), (8, 2)])
def test_checkpoint_resume_with_workers_change_no_batch(
    workers_a: int, workers_b: int
) -> None:
    """
    No batching, single lane; resume with a different number of workers.
    Deterministic mode should make the outputs identical.
    """
    ds = make_dataset("alpha", 96)
    chunk_size = 8

    # Ground truth (any worker count is fine; use A for consistency)
    base = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=1,
        with_batch=False,
        stage_prefetch=4,
    )
    base = base.options(max_workers=workers_a)
    baseline_flat, _ = consume_until(base)
    cut = len(baseline_flat) // 3 * 2  # cut at ~2/3

    p1 = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=1,
        with_batch=False,
        stage_prefetch=4,
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
    ).options(max_workers=workers_b)
    p2._ensure()
    assert p2._engine is not None
    p2._engine.load_state_dict(ckpt, replay=True)
    suffix_flat, _ = consume_until(p2)

    assert prefix_flat + suffix_flat == baseline_flat


@pytest.mark.parametrize("lanes", [1, 4])
@pytest.mark.parametrize("stage_prefetch,final_prefetch", [(0, 0), (4, 16)])
def test_resume_equivalence_with_batch(
    lanes: int, stage_prefetch: int, final_prefetch: int
) -> None:
    """Resume mid-run with batching across lane counts and prefetch settings."""
    ds = make_dataset("alpha", 512)
    chunk_size = 16  # divisible by batch_size
    batch_size = 8

    baseline = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=lanes,
        with_batch=True,
        batch_size=batch_size,
        stage_prefetch=stage_prefetch,
        final_prefetch=final_prefetch,
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
    )
    p2._ensure()
    assert p2._engine is not None
    p2._engine.load_state_dict(ckpt, replay=True)
    suffix_flat, _ = consume_until(p2)

    assert prefix_flat + suffix_flat == baseline_flat


def test_checkpoint_records_inflight_pruned() -> None:
    """
    Sanity: When we checkpoint mid-run, inflight chunks recorded in the engine state
    must not include any chunk_id strictly older than the lane's current progress.
    """
    ds = make_dataset("alpha", 64)
    chunk_size = 8

    p1 = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=1,
        with_batch=False,
        stage_prefetch=8,  # allow multiple chunks in flight
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
def test_checkpoint_resume_mid_chunk_no_batch() -> None:
    ds = make_dataset("alpha", 200)
    chunk_size = 8

    baseline = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=2,
        with_batch=False,
        stage_prefetch=0,
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
    )
    p2._ensure()
    assert p2._engine is not None
    p2._engine.load_state_dict(ckpt, replay=True)
    suffix_flat, _ = consume_until(p2)

    assert prefix_flat + suffix_flat == baseline_flat


# ---- 3) Scale-down equivalence, no batch (simulate N ranks vs. 1 rank)
@pytest.mark.parametrize("with_batch", [False, True])
@pytest.mark.parametrize("stage_prefetch,final_prefetch", [(0, 0), (4, 16)])
def test_scale_down_equivalence_current_api(
    with_batch: bool, stage_prefetch: int, final_prefetch: int
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

    pipe_all = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=N,
        with_batch=with_batch,
        batch_size=(batch_size or 8),
        stage_prefetch=stage_prefetch,
        final_prefetch=final_prefetch,
    ).options(
        num_ranks=1,
        physical_rank=0,
        allow_latency_flush_in_deterministic=False,
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
        ).options(
            num_ranks=N,
            physical_rank=r,
            allow_latency_flush_in_deterministic=False,
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
def test_multiple_resumes_same_size_no_batch_current_api() -> None:
    ds = make_dataset("alpha", 400)
    chunk_size = 8
    canonical_replicas = 4

    # Canonical baseline
    baseline = _build_pipe_params(
        ds,
        chunk_size=chunk_size,
        canonical_replicas=canonical_replicas,
        with_batch=False,
        stage_prefetch=0,
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
        )
        if ckpt is not None:
            pipe._ensure()
            assert pipe._engine is not None
            pipe._engine.load_state_dict(ckpt, replay=True)
        part, ckpt = consume_until(pipe, flat_limit=seg)
        collected.extend(part)

    assert collected == baseline_flat[: len(collected)]


# ---- 7) Multiple resizes across phases (ranks/strategy change), with batch
def test_multiple_resizes_equivalence_with_batch_current_api() -> None:
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

    ckpt: dict | None = None  # carry checkpoint across phases

    for ranks, strat, flat_budget in phases:
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
        ).options(
            num_ranks=1,
            physical_rank=0,
            allow_latency_flush_in_deterministic=False,
        )
        truth_pipe._ensure()
        assert truth_pipe._engine is not None
        if ckpt is not None:
            truth_pipe._engine.load_state_dict(ckpt, replay=True)
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
            ).options(
                num_ranks=ranks,
                physical_rank=r,
                mapping_strategy=strat,
                allow_latency_flush_in_deterministic=False,
            )
            rank_pipe._ensure()
            assert rank_pipe._engine is not None
            if ckpt is not None:
                rank_pipe._engine.load_state_dict(ckpt, replay=True)
            per_rank_elems.append(consume_until(rank_pipe, return_elems=True)[0])

        merged_phase = rr_merge(
            per_rank_elems, mode="elem", limit=len(truth_phase_elems)
        )

        # Allow cyclic rotation at element level for this phase
        assert _is_cyclic_rotation_elems(merged_phase, truth_phase_elems)

        # advance
        ckpt = ckpt_next
