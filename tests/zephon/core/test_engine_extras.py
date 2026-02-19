import types
from typing import Any

import pytest

from zephon.core.constants import (
    ContributorRef,
    LanePtr,
    SampleBatch,
    SampleMeta,
    SampleRecord,
)
from zephon.core.engine import (
    Engine,
    RuntimeOptions,
    get_torch_worker_info,
    inside_torch_worker,
)
from zephon.core.graph import Graph
from zephon.core.planner import Planner
from zephon.core.runtime_spec import resolve_runtime_spec
from zephon.ops.delay import DelayById
from zephon.work.base import WorkChunk, WorkSource


class _DummyWorkSource(WorkSource):
    def __init__(self) -> None:
        super().__init__()

    @property
    def datasets_by_id(self) -> dict[int, Any]:  # type: ignore[override]
        return {}

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


def test_notify_updates_progress_and_cursor() -> None:
    eng = _mk_engine_with_opts(canonical_replicas=1, dp_degree=1)

    lane = 0
    eng.inflight_chunks_per_lane[lane][0] = WorkChunk(
        components={"X": [(1, 2, 3), (4, 5, 6)]}, seed=None
    )
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
    assert eng.inflight_chunks_per_lane[lane] == {}

    eng.inflight_chunks_per_lane[lane][1] = WorkChunk(
        components={"Y": [(7, 8, 9)]}, seed=None
    )
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

    # Close only offset 0 of chunk 0 -> no eviction
    c0_0 = SampleMeta(sample_id=(0, 0, 0), lane_id=lane, chunk_id=0, chunk_offset=0)
    eng.notify(
        lane,
        entries=[ContributorRef(cursor=c0_0.cursor, is_last_child=True)],
        record_cursor=c0_0.cursor,
    )
    assert 0 in eng.inflight_chunks_per_lane[lane]

    # Close offset of chunk 1 -> chunk 0 still present
    c1_0 = SampleMeta(sample_id=(0, 0, 1), lane_id=lane, chunk_id=1, chunk_offset=0)
    eng.notify(
        lane,
        entries=[ContributorRef(cursor=c1_0.cursor, is_last_child=True)],
        record_cursor=c1_0.cursor,
    )
    assert 0 in eng.inflight_chunks_per_lane[lane]  # chunk 0 not evicted yet

    # Close remaining offset of chunk 0 -> both chunks can evict in order
    c0_1 = SampleMeta(sample_id=(0, 0, 2), lane_id=lane, chunk_id=0, chunk_offset=1)
    eng.notify(
        lane,
        entries=[ContributorRef(cursor=c0_1.cursor, is_last_child=True)],
        record_cursor=c0_1.cursor,
    )
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
