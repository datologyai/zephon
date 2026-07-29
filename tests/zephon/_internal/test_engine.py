import gc
import logging
import re
import threading
import time
import types
import weakref
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from tests._helpers import mk_dataset
from zephon._internal.engine import Engine
from zephon._internal.graph import Graph
from zephon._internal.ops.batch import Batch
from zephon._internal.ops.delay import DelayById
from zephon._internal.ops.fetch import FetchOp
from zephon._internal.planner import Planner
from zephon._internal.runners.inline import InlineStageRunner
from zephon._internal.runners.process import ProcessStageRunner
from zephon._internal.runners.threads import ThreadStageRunner
from zephon._internal.runtime_spec import resolve_runtime_spec
from zephon._internal.utils.disk import InsufficientCacheSpaceError
from zephon.io.dataset import Dataset
from zephon.io.options import CacheOptions, StoreOptions
from zephon.options import RuntimeOptions
from zephon.types import ContributorRef, SampleCursor, SampleRecord
from zephon.work.base import MixtureReadConfig, MixtureReadMode, WorkChunk, WorkSource
from zephon.work.static_mixture import StaticMixtureWorkSource


def _run_engine(
    deterministic: bool, workers: int, mode: MixtureReadMode
) -> list[tuple[int, int, int]]:
    # Two small in-memory datasets with two shards each
    ds_a = mk_dataset("A", {0: 20, 1: 20})
    ds_b = mk_dataset("B", {0: 15, 1: 15})

    mixture = {"A": 0.6, "B": 0.4}
    work = StaticMixtureWorkSource(
        [ds_a, ds_b],
        mixture,
        chunk_size=32,
        seed=123,
        shuffle_shards=True,
        shuffle_within_shard=True,
    )

    # Build a simple graph: FetchOp -> DelayById (delay after fetch to act on SampleRecord)
    g = Graph()
    g.add("fetch", FetchOp())
    g.add("delay", DelayById(max_delay_ms=2.0), *g.nodes)

    plan = Planner().make_plan(g)

    opts = RuntimeOptions(
        deterministic=deterministic,
        max_workers=workers,
        mixture_config=MixtureReadConfig(mode=mode, seed=999),
        default_stage_prefetch=16,
    )
    spec = resolve_runtime_spec(plan, opts)
    eng = Engine(plan, opts, work, spec)

    # Collect a fixed number of outputs to make the test bounded
    out_ids: list[tuple[int, int, int]] = []
    for rec in eng.build_iter():
        assert isinstance(rec, SampleRecord)
        out_ids.append(rec.meta.sample_id)
        if len(out_ids) >= 120:
            break
    eng.close()
    return out_ids


def test_engine_reproducible_wrr() -> None:
    first = _run_engine(
        deterministic=True, workers=4, mode=MixtureReadMode.WEIGHTED_ROUND_ROBIN
    )
    second = _run_engine(
        deterministic=True, workers=16, mode=MixtureReadMode.WEIGHTED_ROUND_ROBIN
    )
    assert first == second


def test_engine_reproducible_weighted_random() -> None:
    first = _run_engine(
        deterministic=True, workers=2, mode=MixtureReadMode.WEIGHTED_RANDOM
    )
    second = _run_engine(
        deterministic=True, workers=8, mode=MixtureReadMode.WEIGHTED_RANDOM
    )
    assert first == second


def test_engine_cleanup_finalizer_removes_previous_merged(
    monkeypatch, tmp_path
) -> None:
    # Avoid building runners to keep setup minimal for this finalizer test.
    monkeypatch.setattr(Engine, "_build_runners", lambda self: None)
    g = Graph()
    g.add("noop", DelayById(max_delay_ms=0.0))
    plan = Planner().make_plan(g)
    work = _DummyWorkSource()
    opts = RuntimeOptions(aggregate_dir=str(tmp_path))
    spec = resolve_runtime_spec(plan, opts)
    eng = Engine(plan, opts, work, spec)

    called = {"flag": False}

    def patched_clean(self: Engine) -> None:
        called["flag"] = True

    eng._clean_merged = types.MethodType(patched_clean, eng)
    fin = eng._cleanup_finalizer
    eng_ref = weakref.ref(eng)
    eng = None

    for _ in range(50):
        if not fin.alive:
            break
        gc.collect()
        time.sleep(0.01)

    assert called["flag"] is True
    assert fin.alive is False
    assert eng_ref() is None


class _DummyWorkSource(WorkSource):
    """Minimal WorkSource stub for constructing the Engine in tests.

    We don't iterate the engine here; we only need a `datasets_by_id` mapping
    for operator setup context.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def datasets_by_id(self) -> dict[int, Any]:  # type: ignore[override]
        return {}

    def component_ids(self) -> dict[str, int]:
        return {"default": 0, "short": 1, "long": 2}

    # Methods below satisfy the WorkSource protocol at runtime if accessed.
    def next_chunk(self) -> Any:  # pragma: no cover - not used in these tests
        return None

    # Use default WorkSource.state_dict for lane/canon if needed.

    # load_state_dict falls back to WorkSource.load_state_dict

    def supports_indexing(self) -> bool:  # pragma: no cover - not used
        return False

    def chunk_size_hint(self) -> int | None:  # pragma: no cover - deterministic tests
        return 1

    def __len__(self) -> int:  # pragma: no cover - not used
        return 0

    def sample_id_at(self, index: int) -> Any:  # pragma: no cover - not used
        raise NotImplementedError


# -- Cache disk-space preflight ---------------------------------------------

_GiB = 1024**3


class _FileBackedWorkSource(_DummyWorkSource):
    """`datasets_by_id` with a file-backed entry so the cache preflight runs."""

    @property
    def datasets_by_id(self) -> dict[int, Any]:  # type: ignore[override]
        return {0: Dataset(name="d", backend={"kind": "jsonl"}, path="/data")}


def _cache_preflight_engine_args(tmp_path, limit_bytes: int):
    g = Graph()
    g.add("noop", DelayById(max_delay_ms=0.0))
    plan = Planner().make_plan(g)
    opts = RuntimeOptions(
        io_options=StoreOptions(
            cache=CacheOptions(
                enabled=True, root=tmp_path / "cache", limit_bytes=limit_bytes
            )
        )
    )
    return plan, opts, resolve_runtime_spec(plan, opts)


def test_engine_cache_preflight_raises_on_insufficient_space(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(Engine, "_build_runners", lambda self: None)
    monkeypatch.setattr(
        "zephon._internal.utils.disk.device_space", lambda path: (100 * _GiB, 10 * _GiB)
    )
    plan, opts, spec = _cache_preflight_engine_args(tmp_path, 50 * _GiB)
    with pytest.raises(InsufficientCacheSpaceError):
        Engine(plan, opts, _FileBackedWorkSource(), spec)


def test_engine_cache_preflight_warns_on_low_headroom(
    monkeypatch, tmp_path, caplog
) -> None:
    monkeypatch.setattr(Engine, "_build_runners", lambda self: None)
    monkeypatch.setattr(
        "zephon._internal.utils.disk.device_space",
        lambda path: (1000 * _GiB, 1000 * _GiB),
    )
    plan, opts, spec = _cache_preflight_engine_args(tmp_path, 970 * _GiB)
    with caplog.at_level(logging.WARNING, logger="zephon._internal.utils.disk"):
        Engine(plan, opts, _FileBackedWorkSource(), spec)
    assert sum("headroom" in r.getMessage() for r in caplog.records) == 1


def test_engine_cache_preflight_skipped_without_file_backed_datasets(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(Engine, "_build_runners", lambda self: None)
    check = Mock()
    monkeypatch.setattr("zephon._internal.engine.check_cache_disk_space", check)
    plan, opts, spec = _cache_preflight_engine_args(tmp_path, 50 * _GiB)
    Engine(plan, opts, _DummyWorkSource(), spec)
    check.assert_not_called()


def _mk_three_stage_plan(
    pars: tuple[int, int, int] = (2, 5, 1),
) -> tuple[Engine, list[int]]:
    """Build a 3-stage plan with placements forcing stage breaks and return (engine, expected_caps_fit_to_ops)."""
    g = Graph()
    # Stage 0 (auto)
    g.add("op0", DelayById(max_delay_ms=0.0), parallelism=pars[0])
    # Stage 1 (placement change to force new stage)
    g.add(
        "op1",
        DelayById(max_delay_ms=0.0),
        *g.nodes,
        placement="local",
        parallelism=pars[1],
    )
    # Stage 2 (another placement change to force new stage)
    g.add(
        "op2",
        DelayById(max_delay_ms=0.0),
        *g.nodes,
        placement="local2",
        parallelism=pars[2],
    )

    plan = Planner().make_plan(g)
    work = _DummyWorkSource()
    # Options are test-specific; callers will set allocation knobs later
    opts = RuntimeOptions()
    spec = resolve_runtime_spec(plan, opts)
    eng = Engine(plan, opts, work, spec)
    expected = [pars[0], pars[1], pars[2]]
    return eng, expected


def _extract_caps_from_explain(explain: str) -> list[int]:
    caps: list[int] = []
    for line in explain.splitlines():
        m = re.search(r"cap=(\d+)", line)
        if m:
            caps.append(int(m.group(1)))
    return caps


def _rebuild_with_opts(eng: Engine, **opts: Any) -> Engine:
    # Recreate the engine with new options but same plan/work
    plan = eng._plan  # type: ignore[attr-defined]
    work = eng._work  # type: ignore[attr-defined]
    new_opts = RuntimeOptions(**opts)
    spec = resolve_runtime_spec(plan, new_opts)
    return Engine(plan, new_opts, work, spec)


def test_fit_to_ops_caps_and_explain() -> None:
    eng, expected = _mk_three_stage_plan((2, 5, 1))
    eng = _rebuild_with_opts(
        eng,
        worker_allocation="fit_to_ops",
        max_workers=999,  # ignored in this mode
    )
    # Inspect internal runner caps
    caps = [getattr(r, "_max_workers") for r in eng._runners]  # type: ignore[attr-defined]
    assert caps == expected
    # Explain header reflects mode
    exp = eng.explain()
    assert "Allocation=fit_to_ops" in exp
    assert _extract_caps_from_explain(exp) == expected


def test_per_stage_fixed_caps_and_explain() -> None:
    eng, _ = _mk_three_stage_plan((2, 5, 1))
    eng = _rebuild_with_opts(
        eng,
        worker_allocation="per_stage_fixed",
        max_workers=7,
    )
    caps = [getattr(r, "_max_workers") for r in eng._runners]  # type: ignore[attr-defined]
    assert caps == [7, 7, 7]
    exp = eng.explain()
    assert "Allocation=per_stage_fixed per_stage=7" in exp
    assert _extract_caps_from_explain(exp) == [7, 7, 7]


def test_global_allocation_caps() -> None:
    eng, expected_fit = _mk_three_stage_plan((2, 5, 1))
    assert expected_fit == [2, 5, 1]
    eng = _rebuild_with_opts(
        eng,
        worker_allocation="global",
        max_workers=16,
    )
    # weights proportional to [2,5,1] with total=16 → [4,10,2]
    caps = [getattr(r, "_max_workers") for r in eng._runners]  # type: ignore[attr-defined]
    assert caps == [4, 10, 2]
    exp = eng.explain()
    assert "Allocation=global total=16" in exp
    assert _extract_caps_from_explain(exp) == [4, 10, 2]


def test_global_total_less_than_stages_warns_and_bumps() -> None:
    eng, _ = _mk_three_stage_plan((2, 5, 1))
    with pytest.warns(RuntimeWarning) as rec:
        eng = _rebuild_with_opts(
            eng,
            worker_allocation="global",
            max_workers=2,  # less than number of stages (=3)
        )
    # Warning message should mention bumping to number of stages
    msgs = [str(w.message) for w in rec]
    assert any("bumping to 3" in m or "max_workers_total=2" in m for m in msgs)
    caps = [getattr(r, "_max_workers") for r in eng._runners]  # type: ignore[attr-defined]
    assert caps == [1, 1, 1]


def test_autotune_mode_not_implemented() -> None:
    eng, _ = _mk_three_stage_plan((2, 5, 1))
    with pytest.raises(NotImplementedError):
        _ = _rebuild_with_opts(
            eng,
            worker_allocation="autotune",
            max_workers=8,
        )


def test_engine_forces_inline_runner_for_terminal_batch_stage() -> None:
    """When batch is the last op (no post-batch ops), it stays inline."""
    g = Graph()
    src = g.add("src", DelayById(max_delay_ms=0.0))
    g.add("batch", Batch(microbatch_size=4), src)

    plan = Planner().make_plan(g)
    work = _DummyWorkSource()
    opts = RuntimeOptions(
        runner="process", worker_allocation="per_stage_fixed", max_workers=1
    )
    spec = resolve_runtime_spec(plan, opts)
    eng = Engine(plan, opts, work, spec)
    try:
        assert len(eng._runners) == 2
        assert isinstance(eng._runners[0], ProcessStageRunner)
        assert isinstance(eng._runners[1], InlineStageRunner)
    finally:
        eng.close()


def test_engine_batch_with_post_ops_not_inline() -> None:
    """When post-batch ops exist, batch + post-batch merge into one non-inline stage."""
    g = Graph()
    src = g.add("src", DelayById(max_delay_ms=0.0))
    batch = g.add("batch", Batch(microbatch_size=4), src)
    g.add("tail", DelayById(max_delay_ms=0.0), batch)

    plan = Planner().make_plan(g)
    work = _DummyWorkSource()
    opts = RuntimeOptions(
        runner="process", worker_allocation="per_stage_fixed", max_workers=1
    )
    spec = resolve_runtime_spec(plan, opts)
    eng = Engine(plan, opts, work, spec)
    try:
        assert len(eng._runners) == 2
        assert isinstance(eng._runners[0], ProcessStageRunner)
        assert isinstance(eng._runners[1], ThreadStageRunner)
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# max_worker_retries plumbing (RuntimeOptions -> Engine -> ProcessStageRunner)
# ---------------------------------------------------------------------------


def test_max_worker_retries_reaches_process_stage_runner() -> None:
    """RuntimeOptions.max_worker_retries flows into ProcessStageRunner.

    Regression guard for the full plumbing path:
    ``RuntimeOptions -> StageRuntimeSpec -> Engine._build_runners ->
    ProcessStageRunner.__init__``.  A single missing ``kwargs=`` on any
    hop would cause ``runner._max_worker_retries`` to silently default
    back to 0.
    """
    g = Graph()
    g.add("op", DelayById(max_delay_ms=0.0))
    plan = Planner().make_plan(g)
    work = _DummyWorkSource()
    opts = RuntimeOptions(
        runner="process",
        worker_allocation="per_stage_fixed",
        max_workers=1,
        max_worker_retries=7,
    )
    spec = resolve_runtime_spec(plan, opts)
    eng = Engine(plan, opts, work, spec)
    try:
        assert len(eng._runners) == 1
        runner = eng._runners[0]
        assert isinstance(runner, ProcessStageRunner)
        assert runner._max_worker_retries == 7
    finally:
        eng.close()


def test_max_worker_retries_zero_disables_resilience_at_engine_level() -> None:
    """``max_worker_retries=0`` produces a runner with the watchdog-resilient
    actor path disabled (diagnostic-only)."""
    g = Graph()
    g.add("op", DelayById(max_delay_ms=0.0))
    plan = Planner().make_plan(g)
    work = _DummyWorkSource()
    opts = RuntimeOptions(
        runner="process",
        worker_allocation="per_stage_fixed",
        max_workers=1,
        max_worker_retries=0,
    )
    spec = resolve_runtime_spec(plan, opts)
    eng = Engine(plan, opts, work, spec)
    try:
        runner = eng._runners[0]
        assert isinstance(runner, ProcessStageRunner)
        assert runner._max_worker_retries == 0
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# flush_every_k_chunks validation
# ---------------------------------------------------------------------------


def _make_non_monotone_engine(flush_k: int | None) -> Engine:
    """Build an Engine with a non-monotone pipeline for validation tests."""
    from zephon._internal.ops.shuffle_buffer import ShuffleBuffer

    ds = mk_dataset("V", {0: 8})
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=4,
        seed=0,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    g = Graph()
    g.add("shuf", ShuffleBuffer(buffer_size=4, seed=0))
    plan = Planner().make_plan(g)
    opts = RuntimeOptions(
        deterministic=True,
        max_workers=1,
        default_stage_prefetch=16,
        flush_every_k_chunks=flush_k,
    )
    spec = resolve_runtime_spec(plan, opts)
    return Engine(plan, opts, work, spec)


def test_flush_every_k_chunks_rejects_negative() -> None:
    """Negative flush_every_k_chunks must raise ValueError."""
    with pytest.raises(ValueError, match="non-negative integer"):
        _make_non_monotone_engine(flush_k=-1)


def _make_monotone_engine(flush_k: int | None) -> Engine:
    """Build an Engine with a monotone pipeline for validation tests."""
    ds = mk_dataset("V", {0: 8})
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=4,
        seed=0,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    g = Graph()
    g.add("d", DelayById(max_delay_ms=0.0))
    plan = Planner().make_plan(g)
    opts = RuntimeOptions(
        deterministic=True,
        max_workers=1,
        default_stage_prefetch=16,
        flush_every_k_chunks=flush_k,
    )
    spec = resolve_runtime_spec(plan, opts)
    return Engine(plan, opts, work, spec)


def test_flush_every_k_chunks_rejects_negative_monotone() -> None:
    """Negative flush_every_k_chunks is rejected even for monotone pipelines."""
    with pytest.raises(ValueError, match="non-negative integer"):
        _make_monotone_engine(flush_k=-1)


def test_flush_every_k_chunks_zero_non_monotone_raises() -> None:
    """flush_every_k_chunks=0 with non-monotone ops must raise ValueError."""
    with pytest.raises(ValueError, match="flush_every_k_chunks=0 is not allowed"):
        _make_non_monotone_engine(flush_k=0)


def test_flush_every_k_chunks_zero_monotone_ok() -> None:
    """flush_every_k_chunks=0 is fine for monotone pipelines."""
    eng = _make_monotone_engine(flush_k=0)
    eng.close()


def test_flush_every_k_chunks_positive_monotone_warns_and_clamps() -> None:
    """flush_every_k_chunks>0 with a monotone pipeline warns and clamps to 0."""
    with pytest.warns(UserWarning, match="ignored for monotonic"):
        eng = _make_monotone_engine(flush_k=4)
    assert eng._flush_every_k_chunks == 0, (
        "flush_every_k_chunks should be clamped to 0 for monotonic pipelines"
    )
    eng.close()


# ---------------------------------------------------------------------------
# _epoch_boundaries pruning race in notify()
# ---------------------------------------------------------------------------


class _InterceptBoundaryDict(defaultdict):
    """``_epoch_boundaries`` replacement that pauses ``__setitem__`` mid-prune.

    When *armed*, the next ``__setitem__`` with a list value will:

    1. Set ``entered`` (signaling the list comprehension is done).
    2. Wait on ``resume`` (or time out) before completing the write.

    This opens a window where a concurrent feeder can append to the **old**
    list.  If the prune doesn't hold ``_checkpoint_lock``, that append is
    silently overwritten by the new list.
    """

    def __init__(self) -> None:
        super().__init__(list)
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.armed = False

    def __setitem__(self, key: Any, value: Any) -> None:
        if self.armed and isinstance(value, list):
            self.armed = False
            self.entered.set()
            # Timeout prevents deadlock after the fix: when notify() holds
            # _checkpoint_lock, the feeder blocks on the lock and can never
            # set ``resume``.  The 200 ms timeout lets __setitem__ complete
            # so the lock is released and the feeder can proceed.
            self.resume.wait(timeout=0.2)
        super().__setitem__(key, value)


def test_notify_epoch_boundary_prune_concurrent_append(monkeypatch: Any) -> None:
    """notify() must not lose boundaries appended under ``_checkpoint_lock``.

    engine.py:1555-1558 prunes ``_epoch_boundaries[lane]`` via a
    read-filter-replace **without** ``_checkpoint_lock``.  A concurrent
    feeder append (under the lock) can be lost when the list-comprehension
    replacement overwrites the list containing the append.

    Before fix: prune is unlocked -> feeder appends during the window ->
        append is overwritten -> test FAILS.
    After fix: prune holds _checkpoint_lock -> feeder blocks until prune
        finishes -> append goes to the post-prune list -> test PASSES.
    """
    from zephon.work.base import WorkChunk

    monkeypatch.setattr(Engine, "_build_runners", lambda self: None)

    g = Graph()
    g.add("noop", DelayById(max_delay_ms=0.0))
    plan = Planner().make_plan(g)
    work = _DummyWorkSource()
    opts = RuntimeOptions()
    spec = resolve_runtime_spec(plan, opts)
    eng = Engine(plan, opts, work, spec)

    LANE = 0
    NEW_BOUNDARY = 99

    # 4 single-sample chunks forming 2 epochs: [0,2) and [2,4).
    for cid in range(4):
        eng.inflight_chunks_per_lane[LANE][cid] = WorkChunk(
            components={"default": [(0, 0, cid)]}
        )
    eng._lane_next_cid[LANE] = 4

    # Swap in the intercepting dict.
    intercept = _InterceptBoundaryDict()
    intercept[LANE] = [2, 4]  # boundary after each 2-chunk epoch
    eng._epoch_boundaries = intercept

    def feeder() -> None:
        intercept.entered.wait()  # prune comprehension is done
        with eng._checkpoint_lock:
            eng._epoch_boundaries[LANE].append(NEW_BOUNDARY)
        intercept.resume.set()

    intercept.armed = True
    t = threading.Thread(target=feeder, name="boundary-feeder")
    t.start()

    # Complete all 4 chunks in one notify → triggers eviction + prune.
    entries = [
        ContributorRef(
            cursor=SampleCursor(chunk_id=cid, chunk_offset=0, sample_id=(0, 0, cid)),
            is_last_child=True,
        )
        for cid in range(4)
    ]
    eng.notify(LANE, entries)
    t.join(timeout=5.0)

    assert NEW_BOUNDARY in eng._epoch_boundaries[LANE], (
        f"Epoch boundary {NEW_BOUNDARY} was lost by notify() prune. "
        f"Final boundaries: {list(eng._epoch_boundaries[LANE])}"
    )


# ---------------------------------------------------------------------------
# Phase 2 chunks_in_epoch seeding after restore
# ---------------------------------------------------------------------------


class _ScriptedLaneWS:
    """Minimal per-lane WorkSource that yields a fixed list of chunks then None."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)
        self._i = 0

    def next_chunk(self) -> Any:
        if self._i < len(self._chunks):
            chunk = self._chunks[self._i]
            self._i += 1
            return chunk
        return None


def _build_minimal_engine(monkeypatch: Any) -> Engine:
    """Build an Engine with runners stubbed out, for driving `_lane_stream`."""
    monkeypatch.setattr(Engine, "_build_runners", lambda self: None)
    g = Graph()
    g.add("noop", DelayById(max_delay_ms=0.0))
    plan = Planner().make_plan(g)
    work = _DummyWorkSource()
    opts = RuntimeOptions()
    spec = resolve_runtime_spec(plan, opts)
    return Engine(plan, opts, work, spec)


def _lane_chunk(cid: int) -> WorkChunk:
    return WorkChunk(components={"default": [(0, 0, cid)]})


def _phase2_boundary_cids(eng: Engine, lane: int) -> list[int]:
    """Drive ``_lane_stream`` to exhaustion, returning the flush-boundary cids."""
    return [
        int(item.meta.tags["_boundary_cid"])
        for item in eng._lane_stream(lane)
        if isinstance(item, SampleRecord)
    ]


def test_lane_stream_seeds_first_epoch_count_from_admission_counter(
    monkeypatch: Any,
) -> None:
    """First epoch (no boundary recorded yet): chunks consumed and evicted
    before the checkpoint are gone from inflight but still belong to the open
    epoch, so the count must come from the admission counter.

    Restored with ``flush_every_k_chunks=4``: cids 0,1,2 were admitted (still
    the first epoch, 3 < 4), the accumulator evicted cid 0 before the
    checkpoint, so inflight is {1,2} while ``_lane_next_cid`` is 3. The open
    epoch holds 3 chunks, so the first boundary fires after one more chunk (cid
    4) — not after the two live inflight chunks (which would land it at cid 5).
    """
    eng = _build_minimal_engine(monkeypatch)
    LANE, K = 0, 4
    eng._flush_every_k_chunks = K

    for cid in (1, 2):
        eng.inflight_chunks_per_lane[LANE][cid] = _lane_chunk(cid)
    eng._lane_next_cid[LANE] = 3  # cids 0,1,2 admitted; cid 0 already evicted

    eng._lane_ws[LANE] = _ScriptedLaneWS([_lane_chunk(c) for c in range(3, 9)])  # type: ignore[assignment]

    boundary_cids = _phase2_boundary_cids(eng, LANE)
    assert boundary_cids and boundary_cids[0] == 4, (
        f"first boundary must fire at cid 4 (open epoch holds 3 admitted "
        f"chunks); got {boundary_cids}"
    )


def test_lane_stream_seeds_open_epoch_after_recorded_boundary(
    monkeypatch: Any,
) -> None:
    """Open epoch after a recorded boundary: the seed counts only chunks above
    the last boundary.

    Restored with ``flush_every_k_chunks=4``: epoch 0 = {0,1,2,3} closed at cid
    4, open epoch = {4,5}; inflight is {2,3,4,5} (epoch-0 chunks 0,1 evicted)
    and ``_lane_next_cid`` is 6. The open epoch already holds 2 chunks, so
    Phase 1 re-injects the recorded boundary at cid 4 and the next boundary
    fires after two more chunks (cid 8).
    """
    eng = _build_minimal_engine(monkeypatch)
    LANE, K = 0, 4
    eng._flush_every_k_chunks = K

    for cid in (2, 3, 4, 5):
        eng.inflight_chunks_per_lane[LANE][cid] = _lane_chunk(cid)
    eng._lane_next_cid[LANE] = 6
    eng._epoch_boundaries[LANE] = [4]

    eng._lane_ws[LANE] = _ScriptedLaneWS([_lane_chunk(c) for c in range(6, 12)])  # type: ignore[assignment]

    boundary_cids = _phase2_boundary_cids(eng, LANE)
    assert boundary_cids[:2] == [4, 8], (
        f"expected boundaries [4, 8], got {boundary_cids}"
    )


def test_store_chunk_mixture_prefers_target_mixture(monkeypatch, tmp_path) -> None:
    """A stamped target_mixture (token-aware sources) wins over the counted
    composition; untargeted chunks keep deriving the target from counts."""
    from zephon.work.base import WorkChunk

    monkeypatch.setattr(Engine, "_build_runners", lambda self: None)
    g = Graph()
    g.add("noop", DelayById(max_delay_ms=0.0))
    plan = Planner().make_plan(g)
    opts = RuntimeOptions(aggregate_dir=str(tmp_path))
    spec = resolve_runtime_spec(plan, opts)
    eng = Engine(plan, opts, _DummyWorkSource(), spec)
    try:
        # 80/20 counted composition, 50/50 stamped target.
        components = {
            "short": [(0, 0, i) for i in range(8)],
            "long": [(1, 0, i) for i in range(2)],
        }
        targeted = WorkChunk(
            components=components, seed=0, target_mixture={"short": 0.5, "long": 0.5}
        )
        eng._store_chunk_mixture(0, 0, targeted)
        by_id = eng._get_chunk_mixture(0, 0)
        short_id = eng._get_component_id("short")
        long_id = eng._get_component_id("long")
        assert by_id == {short_id: 0.5, long_id: 0.5}

        untargeted = WorkChunk(components=components, seed=0)
        eng._store_chunk_mixture(0, 1, untargeted)
        assert eng._get_chunk_mixture(0, 1) == {short_id: 0.8, long_id: 0.2}

        # The replay path stores from a deserialized chunk: the stamped target
        # must survive the state round-trip.
        restored = WorkChunk.from_state(targeted.state_dict())
        eng._store_chunk_mixture(0, 2, restored)
        assert eng._get_chunk_mixture(0, 2) == {short_id: 0.5, long_id: 0.5}
    finally:
        eng.close()


def test_component_ids_follow_worksource_vocabulary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _VocabWorkSource(_DummyWorkSource):
        def component_ids(self) -> dict[str, int]:
            return {"rare": 0, "common": 7}

    monkeypatch.setattr(Engine, "_build_runners", lambda self: None)
    g = Graph()
    g.add("noop", DelayById(max_delay_ms=0.0))
    plan = Planner().make_plan(g)
    opts = RuntimeOptions(aggregate_dir=str(tmp_path))
    spec = resolve_runtime_spec(plan, opts)
    eng = Engine(plan, opts, _VocabWorkSource(), spec)
    try:
        # Lookup order does not affect declared ids; sparse ids remain valid.
        assert eng._get_component_id("common") == 7
        assert eng._get_component_id("rare") == 0
        assert eng._get_component_name(7) == "common"
        assert eng._get_component_name(3) is None
        with pytest.raises(ValueError, match="not in the work source's"):
            eng._get_component_id("undeclared")
    finally:
        eng.close()


@pytest.mark.parametrize(
    "vocabulary",
    [
        pytest.param({"a": 0, "b": 0}, id="duplicate-id"),
        pytest.param({"a": -1}, id="negative-id"),
        pytest.param({"a": True}, id="bool-id"),
        pytest.param({"a": []}, id="unhashable-id"),
        pytest.param({1: 0}, id="non-string-name"),
    ],
)
def test_component_ids_reject_invalid_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    vocabulary: dict[Any, Any],
) -> None:
    class _InvalidVocab(_DummyWorkSource):
        def component_ids(self) -> Any:
            return vocabulary

    monkeypatch.setattr(Engine, "_build_runners", lambda self: None)
    g = Graph()
    g.add("noop", DelayById(max_delay_ms=0.0))
    plan = Planner().make_plan(g)
    opts = RuntimeOptions(aggregate_dir=str(tmp_path))
    spec = resolve_runtime_spec(plan, opts)
    with pytest.raises(ValueError, match="unique non-negative"):
        Engine(plan, opts, _InvalidVocab(), spec)
