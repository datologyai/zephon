# tests/integration/test_dataloader_elasticity.py

import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path
from queue import Empty

import pytest

torch = pytest.importorskip("torch")

from zephon import Pipeline as PublicPipeline
from zephon.io import Dataset, InMemoryShard
from zephon.work.static_mixture import StaticMixtureWorkSource

try:
    from torchdata.stateful_dataloader import (
        StatefulDataLoader,  # type: ignore[attr-defined]
    )
except Exception:
    StatefulDataLoader = None  # pragma: no cover

pytestmark = pytest.mark.integration


def _make_dataset(name: str, n: int) -> Dataset:
    rows = [{"text": f"{name}-{i}"} for i in range(n)]
    return Dataset.from_dict(name, {0: InMemoryShard(rows)})


def _extract_elem(item) -> tuple[int, list[str]]:
    from zephon.types import SampleBatch, SampleRecord

    if isinstance(item, SampleRecord):
        payload = item.payload
        assert isinstance(payload, dict)
        return int(item.meta.lane_id), [str(payload.get("text", ""))]
    assert isinstance(item, SampleBatch)
    lids = item.lane_ids
    assert lids, "empty batch"
    lane = int(lids[0])
    texts = []
    for record in item.records:
        payload = record.payload
        assert isinstance(payload, dict)
        texts.append(str(payload.get("text", "")))
    return lane, texts


def _build_pipe(
    ds: Dataset,
    run_id: str,
    *,
    with_batch: bool,
    microbatch_size: int = 8,
    chunk_size: int = 16,
    canonical_replicas: int = 4,
    world_size: int = 1,
    global_rank: int = 0,
    dp_degree: int = 1,
    dp_group_id: int = 0,
    mapping_strategy: str = "contiguous",
    tmp_path: Path = Path("/tmp/pytest_zephon"),
) -> PublicPipeline:
    # Keep prefetch off and threads minimal to avoid shutdown races on partial consumption.
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=chunk_size,
        seed=7,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipe = (
        PublicPipeline(work)
        .decode_text()
        .tokenize(
            tokenizer_id="__fallback__",
            field="text",
            preserve_upstream_payload=True,
        )
    )
    if with_batch:
        pipe = pipe.batch(microbatch_size, drop_last=False)
    return pipe.options(
        deterministic=True,
        canonical_replicas=canonical_replicas,
        world_size=world_size,
        global_rank=global_rank,
        dp_degree=dp_degree,
        dp_group_id=dp_group_id,
        mapping_strategy=mapping_strategy,
        default_stage_prefetch=2,
        prefetch_batches=2,
        max_workers=8,
        aggregate_dir=tmp_path,
        run_id=run_id,
    )


def _make_stateful_dl(pipe: PublicPipeline, num_workers: int) -> "StatefulDataLoader":
    assert StatefulDataLoader is not None
    return StatefulDataLoader(
        pipe.to_torch_dataset(),
        batch_size=None,  # unbatched: Zephon controls batching
        num_workers=num_workers,  # single-process to keep state simple
        persistent_workers=False,
        in_order=True,
    )


def _truth_windows_only(
    dl: "StatefulDataLoader",
    *,
    global_batch_size: int,
    total_windows: int,
) -> list[list[str]]:
    windows: list[list[str]] = []
    it = iter(dl)
    try:
        for _ in range(total_windows):
            buf: list[str] = []
            while len(buf) < global_batch_size:
                next_item = next(it)
                _, texts = _extract_elem(next_item)
                buf.extend(texts)
            windows.append(buf)
    finally:
        del it
    return windows


def _node_worker_proc(
    run_id: str,
    rank: int,
    ranks: int,
    ds: Dataset,
    canonical_replicas: int,
    microbatch_size: int,
    acc_steps: int,
    windows: int,
    mapping_strategy: str,
    num_workers: int,
    tmp_path_str: str,
    start_ckpt: dict | None,
    out_q: "mp.Queue",  # emits (rank, window_idx, [texts...])
    win_barrier: "mp.Barrier",  # sync at window boundaries
    ckpt_barrier: "mp.Barrier",  # sync before checkpoint
    ckpt_q: "mp.Queue",  # emits (rank, merged_ckpt_dict)
) -> None:
    tmp_path = Path(tmp_path_str)
    pipe = _build_pipe(
        ds,
        run_id,
        with_batch=True,
        microbatch_size=microbatch_size,
        chunk_size=16,
        canonical_replicas=canonical_replicas,
        world_size=ranks,
        global_rank=rank,
        dp_degree=ranks,
        dp_group_id=rank,
        mapping_strategy=mapping_strategy,
        tmp_path=tmp_path,
    )
    dl = _make_stateful_dl(pipe, num_workers)
    if start_ckpt is not None:
        dl.load_state_dict(start_ckpt)

    it = iter(dl)
    try:
        for w in range(windows):
            buf: list[str] = []
            for _ in range(acc_steps):
                item = next(it)
                _, texts = _extract_elem(item)
                assert len(texts) == microbatch_size
                buf.extend(texts)
            # win_barrier.wait()
            out_q.put((rank, w, buf), timeout=45.0)
            win_barrier.wait()  # align with other nodes at window boundary

        # Phase checkpoint: every node participates concurrently
        ckpt_barrier.wait()
        merged = dl.state_dict()  # triggers file-based aggregation in your Engine
        ckpt_q.put((rank, merged))
    finally:
        ckpt_barrier.wait(timeout=45.0)  # wait that everybody is done at the end
        try:
            del it
        except Exception:
            pass


def _phase_run_and_checkpoint_mp(
    *,
    ds: Dataset,
    run_id: str,
    ranks: int,
    canonical_replicas: int,
    microbatch_size: int,
    acc_steps: int,
    start_ckpt: dict | None,
    windows: int,
    mapping_strategy: str,
    num_workers: int,
    tmp_path: Path,
) -> tuple[list[list[str]], dict]:
    """
    Run one phase across `ranks` processes, return (windows, merged_ckpt).

    NOTE ON ORDERING:
      We synchronize workers with a Barrier so they *call* put() for window w
      before proceeding to window w+1. However, multiprocessing.Queue is only
      FIFO per producer. Each process has a local feeder/buffer; a fast producer
      can enqueue (and have its feeder flush) items for window w+1 before a slow
      producer’s window w item has actually reached the shared queue. The parent
      can therefore observe (fast, w+1) before (slow, w).

      To make the test deterministic, we *demultiplex by window index* in the
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
            target=_node_worker_proc,
            args=(
                run_id,
                r,
                ranks,
                ds,
                canonical_replicas,
                microbatch_size,
                acc_steps,
                windows,
                mapping_strategy,
                num_workers,
                str(tmp_path),
                start_ckpt,
                out_q,
                win_barrier,
                ckpt_barrier,
                ckpt_q,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Collect per-window buffers from all ranks
    windows_out: list[list[str]] = []
    # Map: window_idx -> list[(rankid, buf)]
    stash: dict[int, list[tuple[int, list[str]]]] = defaultdict(list)

    try:
        # ---- Collect per-window buffers from all ranks (order-agnostic) ----
        for w in range(windows):
            per_window = stash.pop(w, [])
            while len(per_window) < ranks:
                try:
                    rankid, w_idx, buf = out_q.get(timeout=90.0)
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
                    per_window.append((rankid, buf))
                else:
                    # Stash for a future window.
                    stash[w_idx].append((rankid, buf))

            merged: list[str] = []
            for _, chunk in per_window:
                merged.extend(chunk)
            windows_out.append(merged)

        # Gather phase checkpoint from all ranks (identical merged dicts)
        merged_ckpt = None
        for _ in range(ranks):
            try:
                rankid, ckpt = ckpt_q.get(timeout=90.0)
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


@pytest.mark.skipif(
    StatefulDataLoader is None, reason="torchdata.StatefulDataLoader not available"
)
@pytest.mark.parametrize("num_workers", [0, 1, 2, 3, 5, 8])
def test_stateful_dataloader_scale_down_then_up_with_microbatch_change(
    tmp_path: Path, num_workers: int
) -> None:
    """
    Three phases. Fixed GLOBAL=64. Compare each phase's merged multi-rank stream
    (with gradient accumulation) against a single-rank UNBATCHED truth,
    allowing only a permutation within each global window (multiset equality).

      Phase A: ranks=4, mapping=contiguous,  micro=8, acc=2  → 4*8*2 = 64
      Phase B: ranks=2, mapping=interleaved, micro=4, acc=8  → 2*4*8 = 64
      Phase C: ranks=4, mapping=contiguous,  micro=8, acc=2  → 4*8*2 = 64
    """
    GLOBAL = 64
    WINDOWS_PER_PHASE = 4
    phases = [
        (4, "contiguous", 8, 2),
        (2, "interleaved", 4, 8),
        (4, "contiguous", 8, 2),
    ]
    total_windows = WINDOWS_PER_PHASE * len(phases)

    # Dataset large enough to cover all windows.
    ds = _make_dataset("alpha", 4096)
    canonical_lanes = 4

    # ---- Precompute truth windows + boundary checkpoints once ----
    # truth_tmp = tmp_path / "truth"
    # truth_tmp.mkdir()
    truth_pipe = _build_pipe(
        ds,
        "truth",
        with_batch=True,
        microbatch_size=8,
        canonical_replicas=canonical_lanes,
        dp_degree=1,
        dp_group_id=0,
        mapping_strategy="contiguous",
        tmp_path=tmp_path,
    )

    truth_dl = _make_stateful_dl(truth_pipe, num_workers)
    truth_windows = _truth_windows_only(
        truth_dl, global_batch_size=GLOBAL, total_windows=total_windows
    )
    assert len(truth_windows) == total_windows

    got_all: list[list[str]] = []
    start_ckpt: dict | None = None
    phase_idx = 0
    for ranks, strat, micro, acc in phases:
        phase_windows, start_ckpt = _phase_run_and_checkpoint_mp(
            ds=ds,
            run_id=f"phase{phase_idx}",
            ranks=ranks,
            canonical_replicas=canonical_lanes,
            microbatch_size=micro,
            acc_steps=acc,
            start_ckpt=start_ckpt,
            windows=WINDOWS_PER_PHASE,
            mapping_strategy=strat,
            num_workers=num_workers,
            tmp_path=tmp_path,
        )
        got_all.extend(phase_windows)
        phase_idx += 1

    assert len(got_all) == total_windows

    # ---- Compare window-by-window (multiset equality) to truth ----
    offset = 0
    for w_got, w_truth in zip(got_all, truth_windows):
        assert len(w_got) == GLOBAL and len(w_truth) == GLOBAL
        # print(f"\n\n\nWINDOW {offset} ---- \n\n\n{w_got}\n\n{w_truth}\n\n")
        assert Counter(w_got) == Counter(w_truth), f"window {offset}"
        offset += 1
