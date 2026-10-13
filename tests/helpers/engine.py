"""Engine fixtures shared by unit and integration tests."""

import os
import tempfile
from typing import Any

from zephon._internal.engine import Engine
from zephon._internal.graph import Graph
from zephon._internal.ops.delay import DelayById
from zephon._internal.planner import Planner
from zephon._internal.runtime_spec import resolve_runtime_spec
from zephon._internal.stream import LanePtr
from zephon.options import RuntimeOptions
from zephon.work.base import WorkChunk, WorkSource


class DummyWorkSource(WorkSource):
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


def make_engine(**opts: Any) -> Engine:
    g = Graph()
    g.add("delay", DelayById(max_delay_ms=0.0))
    plan = Planner().make_plan(g)
    work = DummyWorkSource()
    # Provide a default aggregate_dir when simulating multi-rank setups
    o = dict(opts)
    try:
        ws = int(o.get("world_size", 1))
    except Exception:
        ws = 1
    if ws > 1 and not o.get("aggregate_dir"):
        o["aggregate_dir"] = os.path.join(tempfile.gettempdir(), "zephon_test_agg")
    opts = RuntimeOptions(**o)
    spec = resolve_runtime_spec(plan, opts)
    return Engine(plan, opts, work, spec)


def seed_inflight_state(eng: Engine, *, chunk_id: int = 0) -> None:
    """Populate inflight chunks, progress, and lane_next_cid for owned lanes."""
    for lane in eng._world.lanes_for_dp_group[eng._world.dp_group_id]:
        eng.inflight_chunks_per_lane[lane][chunk_id] = WorkChunk(
            components={"X": [(1, 2, 3)]}, seed=lane * 10 + chunk_id
        )
        eng._lane_progress[lane] = LanePtr(chunk_id=chunk_id, offset=0)  # type: ignore[index]
        eng._lane_next_cid[lane] = chunk_id + 1  # type: ignore[attr-defined]
