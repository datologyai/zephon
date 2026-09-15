import types
from typing import Any

import pytest

from zephon._internal.checkpoint._schemas import EngineStateV1
from zephon._internal.engine import Engine, get_torch_worker_info, inside_torch_worker
from zephon._internal.graph import Graph
from zephon._internal.ops.delay import DelayById
from zephon._internal.planner import Planner
from zephon._internal.runtime_spec import resolve_runtime_spec
from zephon._internal.stream import LanePtr
from zephon.options import RuntimeOptions
from zephon.types import ContributorRef, SampleBatch, SampleMeta, SampleRecord
from zephon.work.base import WorkChunk, WorkSource


class _DummyWorkSource(WorkSource):
    def __init__(self) -> None:
        super().__init__()

    @property
    def datasets_by_id(self) -> dict[int, Any]:  # type: ignore[override]
        return {}

    def component_ids(self) -> dict[str, int]:
        return {"X": 0, "Y": 1}

    def next_chunk(self) -> Any:  # pragma: no cover - not used here
        return None

    def state_dict(self) -> dict[str, Any]:  # pragma: no cover - minimal
        return {"ws": 1}

    def load_state_dict(self, state: dict[str, Any]) -> None:  # pragma: no cover
        assert "ws" in state or state == {}

    def supports_indexing(self) -> bool:  # pragma: no cover
        return False

    def __len__(self) -> int:  # pragma: no cover
        return 0

    def sample_id_at(self, index: int) -> Any:  # pragma: no cover
        raise NotImplementedError


def _mk_engine_with_opts(**opts: Any) -> Engine:
    g = Graph()
    g.add("delay", DelayById(max_delay_ms=0.0))
    plan = Planner().make_plan(g)
    work = _DummyWorkSource()
    # Provide a default aggregate_dir when simulating multi-rank setups
    o = dict(opts)
    try:
        ws = int(o.get("world_size", 1))
    except Exception:
        ws = 1
    if ws > 1 and not o.get("aggregate_dir"):
        import os
        import tempfile

        o["aggregate_dir"] = os.path.join(tempfile.gettempdir(), "zephon_test_agg")
    opts = RuntimeOptions(**o)
    spec = resolve_runtime_spec(plan, opts)
    return Engine(plan, opts, work, spec)


def test_world_mapping_contiguous_and_interleaved() -> None:
    # Contiguous mapping
    eng_c = _mk_engine_with_opts(
        canonical_replicas=8, world_size=3, dp_degree=3, mapping_strategy="contiguous"
    )
    mapping_c = eng_c._world.lanes_for_dp_group  # type: ignore[attr-defined]
    assert mapping_c == {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7]}

    # Interleaved mapping
    eng_i = _mk_engine_with_opts(
        canonical_replicas=8, world_size=3, dp_degree=3, mapping_strategy="interleaved"
    )
    mapping_i = eng_i._world.lanes_for_dp_group  # type: ignore[attr-defined]
    assert mapping_i == {0: [0, 3, 6], 1: [1, 4, 7], 2: [2, 5]}


def _mk_rec(lane: int, chunk: int, local: int = 0) -> SampleRecord:
    sid = (0, 0, local)
    return SampleRecord(
        meta=SampleMeta(sample_id=sid, lane_id=lane, chunk_id=chunk),
        payload={"text": str(local)},
    )


def test_lane_rr_iter_round_robin_order() -> None:
    eng = _mk_engine_with_opts(canonical_replicas=2, dp_degree=1)
    eng._world.lanes_for_dp_group = {0: [0, 1]}  # type: ignore[attr-defined]

    # Upstream yields several items only for lane 0 initially, then lane 1 later
    upstream: list[SampleRecord | SampleBatch] = [
        _mk_rec(0, 0, 0),
        _mk_rec(0, 0, 1),
        _mk_rec(1, 0, 2),
        _mk_rec(1, 0, 3),
    ]

    out = list(eng._lane_rr_iter(iter(upstream)))  # type: ignore[attr-defined]
    # Should alternate per-lane once both have items, preserving per-lane order
    assert [r.meta.lane_id for r in out] == [0, 1, 0, 1]


def test_torch_worker_helpers_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provide stub torch modules without get_worker_info to trigger fallback path
    import sys

    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")
    # Intentionally do NOT set get_worker_info on data_mod
    utils_mod.data = data_mod
    torch_mod.utils = utils_mod
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data_mod)

    assert inside_torch_worker() is False
    assert get_torch_worker_info() == (0, 1)


def test_torch_worker_helpers_with_stubbed_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Build a lightweight stub for torch.utils.data.get_worker_info
    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")

    class _Info:
        def __init__(self) -> None:
            self.id = 3
            self.num_workers = 7

    def _gwi():
        return _Info()

    data_mod.get_worker_info = _gwi
    utils_mod.data = data_mod
    torch_mod.utils = utils_mod
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_mod)
    monkeypatch.setitem(__import__("sys").modules, "torch.utils", utils_mod)
    monkeypatch.setitem(__import__("sys").modules, "torch.utils.data", data_mod)

    assert inside_torch_worker() is True
    assert get_torch_worker_info() == (3, 7)


def _set_accum_floor(eng: Engine, floor: int) -> None:
    """Simulate a shared runner watermark advanced by a peer lane's flush."""
    for runner in eng._runners:  # type: ignore[attr-defined]
        for op_state in runner.ops:
            op_state._epoch_floor = floor


def test_notify_retains_open_epoch_when_another_lane_flushes() -> None:
    eng = _mk_engine_with_opts(canonical_replicas=2, dp_degree=1)
    lane = 1
    cursors = []
    for cid in range(2):
        sample_id = (0, 0, cid)
        eng.inflight_chunks_per_lane[lane][cid] = WorkChunk(
            components={"X": [sample_id]}
        )
        cursors.append(
            SampleMeta(sample_id=sample_id, lane_id=lane, chunk_id=cid).cursor
        )

    # A peer lane's flush advances the shared runner floor, but lane 1's
    # shuffle state still depends on both of its chunks, even when all their
    # offsets have completed. It has no replay-safe boundary of its own yet.
    eng._epoch_boundaries[0] = [2]
    _set_accum_floor(eng, 2)
    try:
        eng.notify(
            lane, [ContributorRef(cursor=c, is_last_child=True) for c in cursors]
        )
        assert set(eng.inflight_chunks_per_lane[lane]) == {0, 1}

        # Once this lane has a boundary, the completed epoch is eligible for
        # eviction, subject to the usual replay cursor pin.
        eng._epoch_boundaries[lane] = [2]
        eng.notify(lane, [], record_cursor=cursors[-1])
        assert set(eng.inflight_chunks_per_lane[lane]) == {0, 1}
        eng.notify(lane, [], record_cursor=None)
        assert eng.inflight_chunks_per_lane[lane] == {}
    finally:
        eng.close()


def test_notify_updates_progress_and_cursor() -> None:
    eng = _mk_engine_with_opts(canonical_replicas=1, dp_degree=1)

    lane = 0
    eng.inflight_chunks_per_lane[lane][0] = WorkChunk(
        components={"X": [(1, 2, 3), (4, 5, 6)]}, seed=None
    )
    # Close this lane's epoch so atomic eviction can proceed.
    eng._epoch_boundaries[lane] = [1]

    cursor0 = SampleMeta(
        sample_id=(0, 0, 0), lane_id=lane, chunk_id=0, chunk_offset=0
    ).cursor
    cursor1 = (
        SampleMeta(sample_id=(0, 0, 1), lane_id=lane, chunk_id=0, chunk_offset=1)
        .child(0)
        .cursor
    )

    eng.notify(
        lane,
        entries=[
            ContributorRef(cursor=cursor0, is_last_child=True),
            ContributorRef(cursor=cursor1, is_last_child=True),
        ],
        record_cursor=cursor1,
    )
    assert eng._lane_progress[lane].chunk_id == 0  # type: ignore[attr-defined]
    assert eng._lane_progress[lane].offset == 2  # type: ignore[attr-defined]
    assert eng._lane_last_cursor[lane] == cursor1  # type: ignore[attr-defined]
    # Cursor pinning keeps chunk 0 (record_cursor references it); flush to release.
    assert 0 in eng.inflight_chunks_per_lane[lane]
    eng.notify(lane, [], record_cursor=None)
    assert eng.inflight_chunks_per_lane[lane] == {}

    eng.inflight_chunks_per_lane[lane][1] = WorkChunk(
        components={"Y": [(7, 8, 9)]}, seed=None
    )
    eng._epoch_boundaries[lane].append(2)

    cursor2 = SampleMeta(
        sample_id=(0, 0, 2), lane_id=lane, chunk_id=1, chunk_offset=0
    ).cursor
    eng.notify(
        lane,
        entries=[ContributorRef(cursor=cursor2, is_last_child=True)],
        record_cursor=cursor2,
    )
    assert eng._lane_progress[lane].chunk_id == 1  # type: ignore[attr-defined]
    assert eng._lane_progress[lane].offset == 1  # type: ignore[attr-defined]
    assert eng._lane_last_cursor[lane] == cursor2  # type: ignore[attr-defined]
    # Cursor pinning keeps chunk 1; flush to release.
    assert 1 in eng.inflight_chunks_per_lane[lane]
    eng.notify(lane, [], record_cursor=None)
    assert eng.inflight_chunks_per_lane[lane] == {}


def test_replay_snapshot_ignores_evicted_cursor() -> None:
    eng = _mk_engine_with_opts(canonical_replicas=1, dp_degree=1)
    lane = 0

    # Cursor whose chunk is not inflight should not appear as target.
    evicted_cursor = SampleMeta(
        sample_id=(0, 0, 0), lane_id=lane, chunk_id=5, chunk_offset=0
    ).cursor
    eng._lane_last_cursor[lane] = evicted_cursor  # type: ignore[attr-defined]
    eng.inflight_chunks_per_lane[lane] = {}
    eng._publish_replay_snapshot()  # type: ignore[attr-defined]
    assert eng._replay_config.snapshot()[lane] is None  # type: ignore[attr-defined]

    # When the chunk is inflight, the cursor is preserved.
    eng.inflight_chunks_per_lane[lane][5] = WorkChunk(components={"X": [(1, 2, 3)]})
    eng._publish_replay_snapshot()  # type: ignore[attr-defined]
    assert eng._replay_config.snapshot()[lane] == evicted_cursor  # type: ignore[attr-defined]


def test_chunk_eviction_waits_for_all_offsets() -> None:
    eng = _mk_engine_with_opts(canonical_replicas=1, dp_degree=1)
    lane = 0
    eng.inflight_chunks_per_lane[lane][0] = WorkChunk(
        components={"X": [(1,), (2,)]}
    )  # two offsets
    eng.inflight_chunks_per_lane[lane][1] = WorkChunk(components={"Y": [(3,)]})
    # Both chunks belong to one closed epoch and must evict atomically.
    eng._epoch_boundaries[lane] = [2]

    # Close only offset 0 of chunk 0 -> no eviction (all_done fails)
    c0_0 = SampleMeta(sample_id=(0, 0, 0), lane_id=lane, chunk_id=0, chunk_offset=0)
    eng.notify(
        lane,
        entries=[ContributorRef(cursor=c0_0.cursor, is_last_child=True)],
        record_cursor=c0_0.cursor,
    )
    assert 0 in eng.inflight_chunks_per_lane[lane]

    # Close offset of chunk 1 -> chunk 0 still present (chunk 0 incomplete)
    c1_0 = SampleMeta(sample_id=(0, 0, 1), lane_id=lane, chunk_id=1, chunk_offset=0)
    eng.notify(
        lane,
        entries=[ContributorRef(cursor=c1_0.cursor, is_last_child=True)],
        record_cursor=c1_0.cursor,
    )
    assert 0 in eng.inflight_chunks_per_lane[lane]  # chunk 0 not evicted yet

    # Close remaining offset of chunk 0 -> both chunks all_done, atomic eviction fires
    c0_1 = SampleMeta(sample_id=(0, 0, 2), lane_id=lane, chunk_id=0, chunk_offset=1)
    eng.notify(
        lane,
        entries=[ContributorRef(cursor=c0_1.cursor, is_last_child=True)],
        record_cursor=c0_1.cursor,
    )
    # Cursor pinning keeps chunk 0 (record_cursor references it) → aborts eviction.
    assert 0 in eng.inflight_chunks_per_lane[lane]
    eng.notify(lane, [], record_cursor=None)
    assert eng.inflight_chunks_per_lane[lane] == {}


def test_state_round_trip_reconstructs_inflight_and_progress() -> None:
    eng = _mk_engine_with_opts(canonical_replicas=1, dp_degree=1)
    lane = 0
    # Add two chunks as inflight
    eng.inflight_chunks_per_lane[lane][0] = WorkChunk(
        components={"X": [(1, 2, 3)]}, seed=13
    )
    eng.inflight_chunks_per_lane[lane][2] = WorkChunk(
        components={"Y": [(4, 5, 6), (4, 5, 7)]}, seed=None
    )
    eng._lane_progress[lane] = LanePtr(chunk_id=2, offset=1)  # type: ignore[index]
    eng._lane_next_cid[lane] = 3  # type: ignore[attr-defined]

    state = eng.state_dict()

    eng_restored = _mk_engine_with_opts(canonical_replicas=1, dp_degree=1)
    eng_restored.load_state_dict(state, replay=False)

    # Inflight reconstructed with WorkChunk instances
    assert set(eng_restored.inflight_chunks_per_lane[lane].keys()) == {0, 2}
    assert isinstance(eng_restored.inflight_chunks_per_lane[lane][0], WorkChunk)
    assert isinstance(eng_restored.inflight_chunks_per_lane[lane][2], WorkChunk)

    # Progress and next cid restored
    assert eng_restored._lane_progress[lane].chunk_id == 2  # type: ignore[attr-defined]
    assert eng_restored._lane_progress[lane].offset == 1  # type: ignore[attr-defined]
    assert eng_restored._lane_next_cid[lane] == 3  # type: ignore[attr-defined]


# =============================================================================
# Parallelism Parameter Validation Tests
# =============================================================================


class TestParallelismValidation:
    """Tests for parallelism parameter validation in _resolve_parallelism_params."""

    # --- Strict validation (errors) ---

    def test_world_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="world_size.*must be >= 1"):
            _mk_engine_with_opts(world_size=0, global_rank=0)

    def test_global_rank_must_be_in_range(self) -> None:
        with pytest.raises(ValueError, match="global_rank.*must be in"):
            _mk_engine_with_opts(world_size=4, global_rank=4)
        with pytest.raises(ValueError, match="global_rank.*must be in"):
            _mk_engine_with_opts(world_size=4, global_rank=-1)

    def test_dp_degree_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="dp_degree.*must be >= 1"):
            _mk_engine_with_opts(
                world_size=4, global_rank=0, dp_degree=0, dp_group_id=0
            )

    def test_dp_degree_cannot_exceed_world_size(self) -> None:
        with pytest.raises(ValueError, match="dp_degree.*cannot exceed world_size"):
            _mk_engine_with_opts(
                world_size=4, global_rank=0, dp_degree=8, dp_group_id=0
            )

    def test_world_size_must_be_divisible_by_dp_degree(self) -> None:
        # 5 nodes with dp_degree=3 -> mp_degree would be 1.67
        with pytest.raises(
            ValueError, match="world_size.*must be divisible by dp_degree"
        ):
            _mk_engine_with_opts(
                world_size=5, global_rank=0, dp_degree=3, dp_group_id=0
            )
        # 8 nodes with dp_degree=3 -> mp_degree would be 2.67
        with pytest.raises(
            ValueError, match="world_size.*must be divisible by dp_degree"
        ):
            _mk_engine_with_opts(
                world_size=8, global_rank=0, dp_degree=3, dp_group_id=0
            )

    def test_dp_group_id_must_be_in_range(self) -> None:
        with pytest.raises(ValueError, match="dp_group_id.*must be in"):
            _mk_engine_with_opts(
                world_size=4, global_rank=0, dp_degree=2, dp_group_id=2
            )
        with pytest.raises(ValueError, match="dp_group_id.*must be in"):
            _mk_engine_with_opts(
                world_size=4, global_rank=0, dp_degree=2, dp_group_id=-1
            )

    def test_canonical_replicas_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="canonical_replicas.*must be >= 1"):
            _mk_engine_with_opts(
                world_size=4,
                global_rank=0,
                dp_degree=2,
                dp_group_id=0,
                canonical_replicas=0,
            )

    def test_canonical_replicas_must_be_at_least_dp_degree(self) -> None:
        with pytest.raises(
            ValueError, match="canonical_replicas.*must be >= dp_degree"
        ):
            _mk_engine_with_opts(
                world_size=4,
                global_rank=0,
                dp_degree=4,
                dp_group_id=0,
                canonical_replicas=2,
            )

    # --- Valid configurations (no errors) ---

    def test_valid_1d_parallelism(self) -> None:
        # Pure data parallelism: world_size=4, dp_degree=4, mp_degree=1
        eng = _mk_engine_with_opts(world_size=4, global_rank=0)
        assert eng._world.dp_degree == 4
        assert eng._world.canonical_replicas == 4

    def test_valid_3d_parallelism(self) -> None:
        # 3D: world_size=8, dp_degree=2, mp_degree=4 (e.g., TP=2, PP=2)
        eng = _mk_engine_with_opts(
            world_size=8,
            global_rank=0,
            dp_degree=2,
            dp_group_id=0,
            canonical_replicas=2,
        )
        assert eng._world.dp_degree == 2
        assert eng._world.world_size == 8

    def test_valid_with_more_canonical_replicas(self) -> None:
        # Elasticity: more lanes than dp_degree for future scale-up
        eng = _mk_engine_with_opts(
            world_size=4,
            global_rank=0,
            dp_degree=2,
            dp_group_id=0,
            canonical_replicas=8,
        )
        assert eng._world.canonical_replicas == 8
        assert eng._world.dp_degree == 2

    # --- Warnings (valid but unusual) ---

    def test_warns_when_mp_degree_not_power_of_2(self) -> None:
        # world_size=6, dp_degree=2 -> mp_degree=3 (not power of 2)
        with pytest.warns(RuntimeWarning, match="mp_degree.*is not a power of 2"):
            _mk_engine_with_opts(
                world_size=6,
                global_rank=0,
                dp_degree=2,
                dp_group_id=0,
                canonical_replicas=2,
            )

    def test_no_warning_when_mp_degree_is_1(self) -> None:
        # mp_degree=1 should not warn (pure DP is common)
        import warnings as w

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            _mk_engine_with_opts(
                world_size=4, global_rank=0, dp_degree=4, dp_group_id=0
            )
            mp_warnings = [x for x in caught if "mp_degree" in str(x.message)]
            assert len(mp_warnings) == 0

    def test_no_warning_when_mp_degree_power_of_2(self) -> None:
        # mp_degree=4 should not warn
        import warnings as w

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            _mk_engine_with_opts(
                world_size=8,
                global_rank=0,
                dp_degree=2,
                dp_group_id=0,
                canonical_replicas=2,
            )
            mp_warnings = [x for x in caught if "mp_degree" in str(x.message)]
            assert len(mp_warnings) == 0

    def test_warns_when_canonical_replicas_not_divisible_by_dp_degree(self) -> None:
        # canonical_replicas=5, dp_degree=2 -> uneven lane distribution
        with pytest.warns(RuntimeWarning, match="canonical_replicas.*is not divisible"):
            _mk_engine_with_opts(
                world_size=4,
                global_rank=0,
                dp_degree=2,
                dp_group_id=0,
                canonical_replicas=5,
            )

    def test_no_warning_when_canonical_replicas_divisible(self) -> None:
        # canonical_replicas=8, dp_degree=2 -> even distribution
        import warnings as w

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            _mk_engine_with_opts(
                world_size=4,
                global_rank=0,
                dp_degree=2,
                dp_group_id=0,
                canonical_replicas=8,
            )
            canon_warnings = [
                x for x in caught if "canonical_replicas" in str(x.message)
            ]
            assert len(canon_warnings) == 0

    # --- Lane distribution tests for uneven configs ---

    def test_uneven_lane_distribution_interleaved(self) -> None:
        # canonical_replicas=5, dp_degree=2 -> dp0: [0,2,4], dp1: [1,3]
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("ignore")  # Suppress the expected warning
            eng = _mk_engine_with_opts(
                world_size=4,
                global_rank=0,
                dp_degree=2,
                dp_group_id=0,
                canonical_replicas=5,
                mapping_strategy="interleaved",
            )
        mapping = eng._world.lanes_for_dp_group
        assert mapping[0] == [0, 2, 4]  # 3 lanes
        assert mapping[1] == [1, 3]  # 2 lanes

    def test_uneven_lane_distribution_contiguous(self) -> None:
        # canonical_replicas=5, dp_degree=2 -> dp0: [0,1,2], dp1: [3,4]
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("ignore")  # Suppress the expected warning
            eng = _mk_engine_with_opts(
                world_size=4,
                global_rank=0,
                dp_degree=2,
                dp_group_id=0,
                canonical_replicas=5,
                mapping_strategy="contiguous",
            )
        mapping = eng._world.lanes_for_dp_group
        assert mapping[0] == [0, 1, 2]  # 3 lanes
        assert mapping[1] == [3, 4]  # 2 lanes


# =============================================================================
# _merge_state_dicts: shared DP group / model-parallel aggregation
# =============================================================================


def _seed_inflight_state(eng: Engine, *, chunk_id: int = 0) -> None:
    """Populate inflight chunks, progress, and lane_next_cid for owned lanes."""
    for lane in eng._world.lanes_for_dp_group[eng._world.dp_group_id]:
        eng.inflight_chunks_per_lane[lane][chunk_id] = WorkChunk(
            components={"X": [(1, 2, 3)]}, seed=lane * 10 + chunk_id
        )
        eng._lane_progress[lane] = LanePtr(chunk_id=chunk_id, offset=0)  # type: ignore[index]
        eng._lane_next_cid[lane] = chunk_id + 1  # type: ignore[attr-defined]


class TestMergeStateDictsSharedDPGroup:
    """``_merge_state_dicts`` handles same-DP-group peer state files.

    Under 3D parallelism (``dp_degree >= 2 AND world_size > dp_degree``) at
    ``workers_per_rank=1``, every rank in a DP group owns the same canonical
    lanes and writes a state file. The merge must collapse those peer files
    to a single representative per ``(dp_group_id, owned-lane-set)``;
    otherwise the lane-uniqueness invariant on ``inflight`` / ``progress`` /
    ``lane_next_cid`` / ``lane_ws_state`` is violated.

    The ``dp_degree == 1`` peer-duplicate case is intentionally not covered
    here: it's only reachable with ``workers_per_rank > 1`` (DataLoader),
    which these unit tests don't exercise.
    """

    def test_dp2_tp2_each_lane_appears_twice(self) -> None:
        """world_size=4, dp_degree=2: ranks 0-1 share lane 0; ranks 2-3 share lane 1."""
        engs: list[Engine] = []
        # Rank-to-dp_group: contiguous so ranks [0,1] -> dp 0; [2,3] -> dp 1.
        for rank in range(4):
            dp_id = rank // 2
            e = _mk_engine_with_opts(
                canonical_replicas=2,
                world_size=4,
                global_rank=rank,
                dp_degree=2,
                dp_group_id=dp_id,
            )
            engs.append(e)
            _seed_inflight_state(e)

        states = [e._state_dict_local() for e in engs]
        # Each lane should appear twice across the four states.
        lane_counts: dict[int, int] = {}
        for s in states:
            for lane in s["inflight"]:
                lane_counts[int(lane)] = lane_counts.get(int(lane), 0) + 1
        assert lane_counts == {0: 2, 1: 2}

        merged = engs[0]._merge_state_dicts(states)
        assert {int(k) for k in merged["inflight"]} == {0, 1}

    def test_dp2_tp3_each_lane_appears_three_times(self) -> None:
        """world_size=6, dp_degree=2: 3 same-DP-group peers per lane.

        Exercises the >2-peer case to make sure any fix doesn't accidentally
        only handle exact pairs.
        """
        engs: list[Engine] = []
        for rank in range(6):
            dp_id = rank // 3
            e = _mk_engine_with_opts(
                canonical_replicas=2,
                world_size=6,
                global_rank=rank,
                dp_degree=2,
                dp_group_id=dp_id,
            )
            engs.append(e)
            _seed_inflight_state(e)

        states = [e._state_dict_local() for e in engs]
        lane_counts: dict[int, int] = {}
        for s in states:
            for lane in s["inflight"]:
                lane_counts[int(lane)] = lane_counts.get(int(lane), 0) + 1
        assert lane_counts == {0: 3, 1: 3}

        merged = engs[0]._merge_state_dicts(states)
        assert {int(k) for k in merged["inflight"]} == {0, 1}

    def test_dp_equals_world_size_unique_lanes_merge_ok(self) -> None:
        """Sanity / regression guard: when each rank has its own dp_group_id
        (pure DP), the merge already works today and must keep working."""
        engs: list[Engine] = []
        for rank in range(4):
            e = _mk_engine_with_opts(
                canonical_replicas=4,
                world_size=4,
                global_rank=rank,
                dp_degree=4,
                dp_group_id=rank,
            )
            engs.append(e)
            _seed_inflight_state(e)

        states = [e._state_dict_local() for e in engs]
        merged = engs[0]._merge_state_dicts(states)
        assert {int(k) for k in merged["inflight"]} == {0, 1, 2, 3}
        assert {int(k) for k in merged["progress"]} == {0, 1, 2, 3}

    def test_pure_dp_merge_preserves_per_rank_progress(self) -> None:
        """Sanity guard: pure-DP merge result preserves per-rank progress."""
        engs: list[Engine] = []
        for rank in range(2):
            e = _mk_engine_with_opts(
                canonical_replicas=2,
                world_size=2,
                global_rank=rank,
                dp_degree=2,
                dp_group_id=rank,
            )
            engs.append(e)
            # Advance rank `rank` to chunk_id=rank, offset=rank*2.
            for lane in e._world.lanes_for_dp_group[rank]:
                e.inflight_chunks_per_lane[lane][rank] = WorkChunk(
                    components={"X": [(1, 2, 3)]}, seed=lane
                )
                e._lane_progress[lane] = LanePtr(chunk_id=rank, offset=rank * 2)  # type: ignore[index]
                e._lane_next_cid[lane] = rank + 1  # type: ignore[attr-defined]

        states = [e._state_dict_local() for e in engs]
        merged = engs[0]._merge_state_dicts(states)
        assert merged["progress"][0]["chunk_id"] == 0
        assert merged["progress"][1]["chunk_id"] == 1
        assert merged["progress"][1]["offset"] == 2


# =============================================================================
# _dedupe_dp_group_peers: direct unit tests on hand-built typed states
# =============================================================================


def _mk_typed_state(
    *,
    global_rank: int,
    dp_group_id: int,
    lanes: list[int],
    world_size: int = 4,
    canonical_replicas: int = 2,
    dp_degree: int = 2,
    progress: dict[Any, Any] | None = None,
    inflight: dict[Any, Any] | None = None,
    lane_next_cid: dict[Any, Any] | None = None,
    lane_ws_state: dict[Any, Any] | None = None,
    replay_cursors: dict[Any, Any] | None = None,
    epoch_boundaries: dict[Any, Any] | None = None,
    last_round_id: str | None = "rid-1",
    checkpoint_reload_count: int = 0,
) -> EngineStateV1:
    """Minimal valid EngineStateV1 for testing _dedupe_dp_group_peers in isolation."""
    if progress is None:
        progress = {l: {"chunk_id": 0, "offset": 0} for l in lanes}
    if inflight is None:
        inflight = {}
    if lane_next_cid is None:
        lane_next_cid = dict.fromkeys(lanes, 0)
    if lane_ws_state is None:
        lane_ws_state = {l: {"ws": l} for l in lanes}
    if replay_cursors is None:
        replay_cursors = dict.fromkeys(lanes)
    if epoch_boundaries is None:
        epoch_boundaries = {}
    return EngineStateV1(
        world={
            "canonical_replicas": canonical_replicas,
            "world_size": world_size,
            "global_rank": global_rank,
            "dp_degree": dp_degree,
            "dp_group_id": dp_group_id,
        },
        progress=progress,
        lane_next_cid=lane_next_cid,
        lane_ws_state=lane_ws_state,
        last_round_id=last_round_id,
        checkpoint_reload_count=checkpoint_reload_count,
        inflight=inflight,
        replay_cursors=replay_cursors,
        epoch_boundaries=epoch_boundaries,
    )


@pytest.fixture
def dedupe_engine() -> Engine:
    """Engine instance solely for calling ``_dedupe_dp_group_peers`` — the
    method uses no instance state, so any engine works as the receiver."""
    return _mk_engine_with_opts(canonical_replicas=1, dp_degree=1)


class TestDedupeDpGroupPeers:
    """Direct unit tests for ``Engine._dedupe_dp_group_peers``."""

    def test_progress_divergence_raises_with_actionable_message(
        self, dedupe_engine: Engine
    ) -> None:
        """Same-DP-group peers with different progress trigger the barrier hint."""
        rep = _mk_typed_state(global_rank=0, dp_group_id=0, lanes=[0, 1])
        diverged = _mk_typed_state(
            global_rank=1,
            dp_group_id=0,
            lanes=[0, 1],
            progress={
                0: {"chunk_id": 5, "offset": 0},  # rep has chunk_id=0
                1: {"chunk_id": 0, "offset": 0},
            },
        )
        with pytest.raises(RuntimeError) as excinfo:
            dedupe_engine._dedupe_dp_group_peers([rep, diverged])
        msg = str(excinfo.value)
        assert "DP group 0" in msg
        assert "'progress'" in msg
        assert "rank 0" in msg and "rank 1" in msg
        assert "barrier" in msg

    def test_replay_cursors_divergence_raises(self, dedupe_engine: Engine) -> None:
        """Replay-cursor divergence across same-DP-group peers is also flagged."""
        rep = _mk_typed_state(
            global_rank=0,
            dp_group_id=0,
            lanes=[0, 1],
            replay_cursors={0: "cursor-a", 1: "cursor-b"},
        )
        diverged = _mk_typed_state(
            global_rank=1,
            dp_group_id=0,
            lanes=[0, 1],
            replay_cursors={0: "cursor-a", 1: "DIFFERENT"},
        )
        with pytest.raises(RuntimeError) as excinfo:
            dedupe_engine._dedupe_dp_group_peers([rep, diverged])
        assert "'replay_cursors'" in str(excinfo.value)

    def test_production_side_divergence_accepted_rep_wins(
        self, dedupe_engine: Engine
    ) -> None:
        """Same delivery, different prefetch state → dedupe succeeds and the
        lowest-rank rep's production fields are kept wholesale."""
        rep = _mk_typed_state(
            global_rank=0,
            dp_group_id=0,
            lanes=[0, 1],
            inflight={0: {3: {"k": "rep"}}, 1: {}},
            lane_next_cid={0: 4, 1: 1},
            epoch_boundaries={0: [2]},
        )
        faster_peer = _mk_typed_state(
            global_rank=1,
            dp_group_id=0,
            lanes=[0, 1],
            inflight={0: {3: {"k": "peer"}, 4: {"k": "peer-extra"}}, 1: {}},
            lane_next_cid={0: 5, 1: 1},
            epoch_boundaries={0: [2, 4]},
        )
        chosen = dedupe_engine._dedupe_dp_group_peers([rep, faster_peer])
        assert len(chosen) == 1
        kept = chosen[0]
        assert int(kept.world["global_rank"]) == 0
        # rep's production-side snapshot is preserved verbatim
        assert kept.inflight == {0: {3: {"k": "rep"}}, 1: {}}
        assert kept.lane_next_cid == {0: 4, 1: 1}
        assert kept.epoch_boundaries == {0: [2]}

    def test_disjoint_lane_partitions_not_collapsed(
        self, dedupe_engine: Engine
    ) -> None:
        """``workers_per_rank > 1`` partitions a DP group's lanes across workers;
        those state files have the same ``dp_group_id`` but disjoint lane sets
        and must survive dedupe as separate reps."""
        worker_a = _mk_typed_state(
            global_rank=0, dp_group_id=0, lanes=[0], canonical_replicas=2
        )
        worker_b = _mk_typed_state(
            global_rank=0, dp_group_id=0, lanes=[1], canonical_replicas=2
        )
        chosen = dedupe_engine._dedupe_dp_group_peers([worker_a, worker_b])
        assert len(chosen) == 2
        kept_lane_sets = {frozenset(int(k) for k in st.progress) for st in chosen}
        assert kept_lane_sets == {frozenset({0}), frozenset({1})}

    def test_three_peers_collapse_to_single_rep(self, dedupe_engine: Engine) -> None:
        """All three TP/PP peers (matching delivery) collapse to the lowest-rank rep."""
        peers = [
            _mk_typed_state(global_rank=r, dp_group_id=0, lanes=[0, 1])
            for r in (0, 1, 2)
        ]
        chosen = dedupe_engine._dedupe_dp_group_peers(peers)
        assert len(chosen) == 1
        assert int(chosen[0].world["global_rank"]) == 0

    def test_cross_dp_group_peers_preserved(self, dedupe_engine: Engine) -> None:
        """Peers in different DP groups own disjoint lane sets and survive intact."""
        dp0_peer = _mk_typed_state(global_rank=0, dp_group_id=0, lanes=[0])
        dp1_peer = _mk_typed_state(global_rank=1, dp_group_id=1, lanes=[1])
        chosen = dedupe_engine._dedupe_dp_group_peers([dp0_peer, dp1_peer])
        assert len(chosen) == 2
        kept_dp_ids = {int(st.world["dp_group_id"]) for st in chosen}
        assert kept_dp_ids == {0, 1}

    def test_universal_field_divergence_caught_before_dedupe(
        self, dedupe_engine: Engine
    ) -> None:
        """Cross-shard global-invariant checks in ``_merge_state_dicts`` run
        BEFORE dedupe, so divergence in a universal field between same-DP-group
        peers is surfaced rather than masked by dropping the divergent peer."""
        rep = _mk_typed_state(global_rank=0, dp_group_id=0, lanes=[0, 1])
        diverged = _mk_typed_state(
            global_rank=1, dp_group_id=0, lanes=[0, 1], last_round_id="DIFFERENT"
        )
        with pytest.raises(RuntimeError) as excinfo:
            dedupe_engine._merge_state_dicts([rep.to_dict(), diverged.to_dict()])
        assert "last_round_id mismatch" in str(excinfo.value)


# =============================================================================
# Per-lane delivery counters + mid-window checkpoint warning (observability)
# =============================================================================


def _no_midwindow_warnings(recwarn: Any) -> bool:
    return not [w for w in recwarn.list if "mid-window" in str(w.message)]


class TestLaneEmittedCounters:
    def test_record_delivery_increments_per_lane(self) -> None:
        eng = _mk_engine_with_opts(canonical_replicas=2, dp_degree=1)
        eng.record_delivery(0)
        eng.record_delivery(0)
        eng.record_delivery(1)
        assert eng._lane_emitted == {0: 2, 1: 1}

    def test_record_delivery_persists_next_rr_lane_before_checkpoint(self) -> None:
        """A checkpoint immediately after delivery saves the next lane."""
        eng = _mk_engine_with_opts(canonical_replicas=4, dp_degree=1)
        rr = eng._lane_rr_iter(iter([_mk_rec(0, 0), _mk_rec(1, 0)]))
        first = next(rr)
        eng.record_delivery(first.meta.lane_id)
        second = next(rr)
        eng.record_delivery(second.meta.lane_id)
        state = eng.state_dict()
        assert state["rr_next_idx"] == {"0:0/1:0,1,2,3": 2}

    def test_rr_prefetch_does_not_persist_undelivered_lanes(self) -> None:
        """Producer read-ahead cannot advance the checkpoint's RR pointer."""
        eng = _mk_engine_with_opts(canonical_replicas=4, dp_degree=1)
        rr = eng._lane_rr_iter(iter([_mk_rec(lane, 0) for lane in range(4)]))

        first = next(rr)
        next(rr)
        eng.record_delivery(first.meta.lane_id)
        next(rr)  # producer advances again while only lane 0 is acknowledged

        state = eng.state_dict()
        assert state["rr_next_idx"] == {"0:0/1:0,1,2,3": 1}

    def test_state_dict_serializes_counters_for_owned_lanes(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        eng = _mk_engine_with_opts(canonical_replicas=2, dp_degree=1)
        eng.record_delivery(0)
        eng.record_delivery(1)
        state = eng.state_dict()
        assert state["lane_emitted"] == {0: 1, 1: 1}
        # Equal counts across all canonical lanes → window-aligned, no warning.
        assert _no_midwindow_warnings(recwarn)

    def test_single_lane_fast_path_counts_and_never_warns(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        """canonical_replicas=1: the single-lane fast path (no RR multiplexer)
        still counts deliveries, and one lane can never be mid-window."""
        eng = _mk_engine_with_opts(canonical_replicas=1, dp_degree=1)
        for _ in range(3):
            eng.record_delivery(0)
        state = eng.state_dict()
        assert state["lane_emitted"] == {0: 3}
        assert _no_midwindow_warnings(recwarn)

    def test_round_trip_restores_counters_and_continues_counting(self) -> None:
        eng = _mk_engine_with_opts(canonical_replicas=2, dp_degree=1)
        for _ in range(3):
            eng.record_delivery(0)
            eng.record_delivery(1)
        state = eng.state_dict()

        eng2 = _mk_engine_with_opts(canonical_replicas=2, dp_degree=1)
        eng2.load_state_dict(state, replay=False)
        # Invariant: after load_state_dict, counters equal their
        # checkpointed values...
        assert dict(eng2._lane_emitted) == {0: 3, 1: 3}
        assert eng2._lane_emitted_valid is True
        # ...and increase only for newly delivered items.
        eng2.record_delivery(0)
        assert dict(eng2._lane_emitted) == {0: 4, 1: 3}

    def test_legacy_state_without_counters_loads_cleanly(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        """Backward compat: a pre-counter checkpoint loads fine — counters are
        treated as unknown (no warning, no crash) and stay unknown on re-save."""
        eng = _mk_engine_with_opts(canonical_replicas=2, dp_degree=1)
        eng.record_delivery(0)
        state = eng._state_dict_local()
        state.pop("lane_emitted")  # simulate an old checkpoint

        eng2 = _mk_engine_with_opts(canonical_replicas=2, dp_degree=1)
        eng2.load_state_dict(state, replay=False)
        assert dict(eng2._lane_emitted) == {}
        assert eng2._lane_emitted_valid is False
        assert _no_midwindow_warnings(recwarn)

        # Unknown propagates: the next save must not pretend to know counts.
        restate = eng2.state_dict()
        assert restate["lane_emitted"] == {}
        assert _no_midwindow_warnings(recwarn)


class TestMidWindowCheckpointWarning:
    def test_unequal_counts_warn_on_single_rank_state_dict(self) -> None:
        eng = _mk_engine_with_opts(canonical_replicas=4, dp_degree=1)
        # 5 deliveries across 4 lanes → mid-window (counts 2,1,1,1).
        for lane in (0, 1, 2, 3, 0):
            eng.record_delivery(lane)
        with pytest.warns(RuntimeWarning, match=r"mid-window.*\[1\.\.2\]"):
            eng.state_dict()

    def test_equal_counts_do_not_warn(self, recwarn: pytest.WarningsRecorder) -> None:
        eng = _mk_engine_with_opts(canonical_replicas=4, dp_degree=1)
        for _ in range(2):
            for lane in range(4):
                eng.record_delivery(lane)
        eng.state_dict()
        assert _no_midwindow_warnings(recwarn)

    def test_merge_collects_counters_and_warns_on_mid_window(self) -> None:
        engs: list[Engine] = []
        for rank in range(2):
            e = _mk_engine_with_opts(
                canonical_replicas=2,
                world_size=2,
                global_rank=rank,
                dp_degree=2,
                dp_group_id=rank,
            )
            engs.append(e)
            _seed_inflight_state(e)
        engs[0].record_delivery(0)
        engs[0].record_delivery(0)
        engs[1].record_delivery(1)

        states = [e._state_dict_local() for e in engs]
        with pytest.warns(RuntimeWarning, match=r"mid-window.*\[1\.\.2\]"):
            merged = engs[0]._merge_state_dicts(states)
        assert merged["lane_emitted"] == {0: 2, 1: 1}

    def test_merge_with_equal_counts_does_not_warn(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        engs: list[Engine] = []
        for rank in range(2):
            e = _mk_engine_with_opts(
                canonical_replicas=2,
                world_size=2,
                global_rank=rank,
                dp_degree=2,
                dp_group_id=rank,
            )
            engs.append(e)
            _seed_inflight_state(e)
            e.record_delivery(rank)  # one delivery on each rank's lane
        merged = engs[0]._merge_state_dicts([e._state_dict_local() for e in engs])
        assert merged["lane_emitted"] == {0: 1, 1: 1}
        assert _no_midwindow_warnings(recwarn)

    def test_merge_with_partial_coverage_does_not_warn(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        """Shards whose counters are unknown leave coverage incomplete — the
        merged checkpoint must stay silent rather than guess."""
        engs: list[Engine] = []
        for rank in range(2):
            e = _mk_engine_with_opts(
                canonical_replicas=2,
                world_size=2,
                global_rank=rank,
                dp_degree=2,
                dp_group_id=rank,
            )
            engs.append(e)
            _seed_inflight_state(e)
        engs[0].record_delivery(0)
        engs[1]._lane_emitted_valid = False  # rank 1 resumed from a legacy ckpt

        states = [e._state_dict_local() for e in engs]
        merged = engs[0]._merge_state_dicts(states)
        assert merged["lane_emitted"] == {0: 1}
        assert _no_midwindow_warnings(recwarn)

    def test_load_warns_on_mid_window_into_different_topology(self) -> None:
        src = _mk_engine_with_opts(canonical_replicas=2, dp_degree=1)
        src.record_delivery(0)
        src.record_delivery(0)
        src.record_delivery(1)
        state = src._state_dict_local()  # world_size=1, owns both lanes

        dst = _mk_engine_with_opts(
            canonical_replicas=2,
            world_size=2,
            global_rank=0,
            dp_degree=2,
            dp_group_id=0,
        )
        with pytest.warns(RuntimeWarning, match="Resuming a mid-window checkpoint"):
            dst.load_state_dict(state, replay=False)
        # Counters for the (now smaller) owned lane set are still restored.
        assert dict(dst._lane_emitted) == {0: 2}
        assert dst._lane_emitted_valid is True

    def test_load_same_topology_does_not_warn(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        src = _mk_engine_with_opts(canonical_replicas=2, dp_degree=1)
        src.record_delivery(0)
        src.record_delivery(0)
        src.record_delivery(1)
        state = src._state_dict_local()

        dst = _mk_engine_with_opts(canonical_replicas=2, dp_degree=1)
        dst.load_state_dict(state, replay=False)
        assert not [
            w for w in recwarn.list if "Resuming a mid-window" in str(w.message)
        ]

    def test_load_equal_counts_into_different_topology_does_not_warn(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        src = _mk_engine_with_opts(canonical_replicas=2, dp_degree=1)
        src.record_delivery(0)
        src.record_delivery(1)
        state = src._state_dict_local()  # window-aligned

        dst = _mk_engine_with_opts(
            canonical_replicas=2,
            world_size=2,
            global_rank=0,
            dp_degree=2,
            dp_group_id=0,
        )
        dst.load_state_dict(state, replay=False)
        assert not [
            w for w in recwarn.list if "Resuming a mid-window" in str(w.message)
        ]


# =============================================================================
# Reload must SCATTER the tail round-robin pointer to its owner, not BROADCAST
# the full merged pointer-set into every rank.
# =============================================================================
#
# The tail-RR pointer is physical/owner-scoped: its key encodes the owning rank
# ("{global_rank}:{worker}/{active}:{lanes}"). A fresh run merges cleanly
# because each rank contributes only its own key (one author per key). The
# merged checkpoint, however, holds every rank's key. ``load_state_dict`` used
# to copy that whole dict into *every* rank, so each rank then carried all
# peers' pointers; only its own moved as it trained, while the peers' copies
# went stale. The next aggregation merge saw the owner's advanced value
# conflict with the stale copies and raised
# ``cannot merge N aggregation state shards: ... rr_next_idx[...] conflicts``.
#
# This is invisible until ALL of: world_size > 1 (merge path runs), >= 2 lanes
# per rank (else the pointer is pinned to 0), a same-topology reload (so the
# owner's key matches one the peers also hold), and a checkpoint AFTER that
# reload. No existing test hits that intersection, which is why the crash
# shipped.


# The bug needs >= 2 lanes per rank (a single-lane rank pins the pointer to 0);
# these tests use exactly two, the minimal trigger.
LANES_PER_RANK = 2


def _owned_pair(eng: Engine) -> tuple[int, int]:
    lanes = eng._world.lanes_for_dp_group[eng._world.dp_group_id]
    assert len(lanes) == LANES_PER_RANK, (
        f"test assumes {LANES_PER_RANK} lanes/rank, got {lanes}"
    )
    return lanes[0], lanes[1]


def _set_least_advanced(eng: Engine, behind_lane: int) -> None:
    """Make ``behind_lane`` the least-advanced of the rank's two lanes so
    ``_refresh_rr_from_progress`` resolves the pointer to that lane's index."""
    lo, hi = _owned_pair(eng)
    for lane in (lo, hi):
        eng._lane_progress[lane] = LanePtr(  # type: ignore[index]
            chunk_id=0, offset=0 if lane == behind_lane else 1
        )


def _build_pure_dp_engines(world_size: int, agg_dir: str) -> list[Engine]:
    """``world_size`` ranks, two canonical lanes per rank, pure data parallel."""
    engs: list[Engine] = []
    for r in range(world_size):
        e = _mk_engine_with_opts(
            canonical_replicas=LANES_PER_RANK * world_size,  # 2 lanes per rank
            world_size=world_size,
            global_rank=r,
            dp_degree=world_size,
            dp_group_id=r,
            aggregate_dir=agg_dir,
            run_id="rr-pointer-regression",
        )
        _seed_inflight_state(e)
        engs.append(e)
    return engs


def test_reload_then_checkpoint_does_not_conflict_on_rr_pointer(tmp_path) -> None:
    """Fresh merge -> reload into every rank -> advance -> merge again.

    The second merge must succeed. Before the fix it raised
    ``RuntimeError: cannot merge ... rr_next_idx[...] conflicts`` because every
    rank had loaded (and re-emitted) all peers' pointers.
    """
    world_size = 4
    engs = _build_pure_dp_engines(world_size, str(tmp_path))

    # Fresh checkpoint: each rank's lower-id lane is least-advanced, so every
    # pointer resolves to index 0. Fresh => exactly one author per key.
    for e in engs:
        _set_least_advanced(e, behind_lane=_owned_pair(e)[0])
    merged0 = engs[0]._merge_state_dicts([e._state_dict_local() for e in engs])
    assert len(merged0["rr_next_idx"]) == world_size  # one key per rank
    assert set(merged0["rr_next_idx"].values()) == {0}

    # Reload the merged checkpoint into every rank (identical topology).
    # replay=False keeps the test focused on pointer scoping, not replay.
    for e in engs:
        e.load_state_dict(merged0, replay=False)

    # Deliver from the lower-id lane so the acknowledged next pointer flips
    # 0 -> 1 for every owner. Peer pointers were discarded during load.
    for e in engs:
        lo, hi = _owned_pair(e)
        rr = e._lane_rr_iter(iter([_mk_rec(lo, 0), _mk_rec(hi, 0)]))
        first = next(rr)
        assert first.meta.lane_id == lo
        e.record_delivery(first.meta.lane_id)

    # Must NOT raise (pre-fix: rr_next_idx conflict across the shards).
    merged1 = engs[0]._merge_state_dicts([e._state_dict_local() for e in engs])
    assert len(merged1["rr_next_idx"]) == world_size
    assert set(merged1["rr_next_idx"].values()) == {1}


def test_load_state_dict_scopes_rr_pointer_to_owner(tmp_path) -> None:
    """After a reload a rank keeps only its OWN rr pointer key (never peers'),
    and that key's value is preserved (progress unchanged => no spurious flip)."""
    world_size = 4
    engs = _build_pure_dp_engines(world_size, str(tmp_path))
    for e in engs:
        _set_least_advanced(e, behind_lane=_owned_pair(e)[0])
    merged0 = engs[0]._merge_state_dicts([e._state_dict_local() for e in engs])
    # The merged checkpoint legitimately holds every rank's key...
    assert len(merged0["rr_next_idx"]) == world_size

    # ...but each rank must restore only the key it owns, value intact.
    for r, e in enumerate(engs):
        e.load_state_dict(merged0, replay=False)
        rr = e._rr_next_idx
        assert {int(k.split(":")[0]) for k in rr} == {r}, (
            f"rank {r} restored peers' rr pointers: {sorted(rr)}"
        )
        (own_key,) = rr  # exactly one key for this single-worker rank
        assert rr[own_key] == merged0["rr_next_idx"][own_key]
