import types
from typing import Any

import pytest

from zephon.core.constants import LanePtr, SampleBatch, SampleMeta, SampleRecord
from zephon.core.engine import (
    Engine,
    RuntimeOptions,
    get_torch_worker_info,
    inside_torch_worker,
)
from zephon.core.graph import Graph
from zephon.core.planner import Planner
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
        nr = int(o.get("num_ranks", 1))
    except Exception:
        nr = 1
    if nr > 1 and not o.get("aggregate_dir"):
        import os
        import tempfile

        o["aggregate_dir"] = os.path.join(tempfile.gettempdir(), "zephon_test_agg")
    return Engine(plan, RuntimeOptions(**o), work)


def test_world_mapping_contiguous_and_interleaved() -> None:
    # Contiguous mapping
    eng_c = _mk_engine_with_opts(
        canonical_replicas=8, num_ranks=3, mapping_strategy="contiguous"
    )
    mapping_c = eng_c._world.lanes_for_rank  # type: ignore[attr-defined]
    assert mapping_c == {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7]}

    # Interleaved mapping
    eng_i = _mk_engine_with_opts(
        canonical_replicas=8, num_ranks=3, mapping_strategy="interleaved"
    )
    mapping_i = eng_i._world.lanes_for_rank  # type: ignore[attr-defined]
    assert mapping_i == {0: [0, 3, 6], 1: [1, 4, 7], 2: [2, 5]}


def test_apportion_edge_cases_and_fairness() -> None:
    ap = Engine._apportion

    assert ap(0, [1, 2, 3]) == [0, 0, 0]
    assert ap(5, []) == []
    # Non-positive weights → equal split
    assert ap(5, [0, 0, -1, 0]) in (
        [2, 1, 1, 1],
        [1, 1, 2, 1],
        [1, 1, 1, 2],
    )  # distribution may pick first slots

    # Largest remainder proportional split
    parts = ap(16, [2, 5, 1])
    assert parts == [4, 10, 2]


def _mk_rec(lane: int, chunk: int, local: int = 0) -> SampleRecord:
    sid = (0, 0, local)
    return SampleRecord(
        meta=SampleMeta(sample_id=sid, lane_id=lane, chunk_id=chunk),
        payload={"text": str(local)},
    )


def test_lane_rr_iter_round_robin_order() -> None:
    eng = _mk_engine_with_opts(canonical_replicas=2, num_ranks=1)
    eng._world.lanes_for_rank = {0: [0, 1]}  # type: ignore[attr-defined]

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


def test_notify_replay_and_live_progress() -> None:
    eng = _mk_engine_with_opts(canonical_replicas=1, num_ranks=1)

    lane = 0
    # Live mode: no replay yet
    should_emit = eng.notify(
        lane, max_chunk_id=0, max_chunk_samples=[(0, 0, 0), (0, 0, 1)]
    )
    assert should_emit is True
    assert eng._lane_progress[lane].chunk_id == 0  # type: ignore[attr-defined]
    assert eng._lane_progress[lane].offset == 2  # type: ignore[attr-defined]

    # Enter replay mode by loading a state dict with progress at chunk 1, offset 1
    # Prime inflight with a chunk to also exercise inflight serialization
    eng.inflight_chunks_per_lane[lane][1] = WorkChunk(components={"A": [(0, 0, 0)]})
    # Maintain invariant: next_cid must be max(inflight)+1 for state_dict()
    eng._lane_next_cid[lane] = 2  # type: ignore[attr-defined]
    state = eng.state_dict()
    # Mutate saved progress to emulate a later checkpoint
    state["progress"] = {str(lane): {"chunk_id": 1, "offset": 1}}

    eng2 = _mk_engine_with_opts(canonical_replicas=1, num_ranks=1)
    eng2.load_state_dict(state, replay=True)

    # Before reaching target chunk 1: drop everything
    assert eng2.notify(lane, max_chunk_id=0, max_chunk_samples=[(0, 0, 0)]) is False
    # At target chunk 1, but cumulative <= offset → still drop
    assert eng2.notify(lane, max_chunk_id=1, max_chunk_samples=[(0, 0, 0)]) is False
    # Crossing the offset with additional items → accept and switch to live
    assert (
        eng2.notify(lane, max_chunk_id=1, max_chunk_samples=[(0, 0, 1), (0, 0, 2)])
        is True
    )


def test_state_round_trip_reconstructs_inflight_and_progress() -> None:
    eng = _mk_engine_with_opts(canonical_replicas=1, num_ranks=1)
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

    eng_restored = _mk_engine_with_opts(canonical_replicas=1, num_ranks=1)
    eng_restored.load_state_dict(state, replay=False)

    # Inflight reconstructed with WorkChunk instances
    assert set(eng_restored.inflight_chunks_per_lane[lane].keys()) == {0, 2}
    assert isinstance(eng_restored.inflight_chunks_per_lane[lane][0], WorkChunk)
    assert isinstance(eng_restored.inflight_chunks_per_lane[lane][2], WorkChunk)

    # Progress and next cid restored
    assert eng_restored._lane_progress[lane].chunk_id == 2  # type: ignore[attr-defined]
    assert eng_restored._lane_progress[lane].offset == 1  # type: ignore[attr-defined]
    assert eng_restored._lane_next_cid[lane] == 3  # type: ignore[attr-defined]
