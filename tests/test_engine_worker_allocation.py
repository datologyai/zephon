import re
from typing import Any

import pytest

from zephon.core.engine import Engine, RuntimeOptions
from zephon.core.graph import Graph
from zephon.core.planner import Planner
from zephon.ops.delay import DelayById


class _DummyWorkSource:
    """Minimal WorkSource stub for constructing the Engine in tests.

    We don't iterate the engine here; we only need a `datasets_by_id` mapping
    for operator setup context.
    """

    @property
    def datasets_by_id(self) -> dict[int, Any]:  # type: ignore[override]
        return {}

    # Methods below satisfy the WorkSource protocol at runtime if accessed.
    def next_chunk_for(
        self,
        lane: int,
        *,
        worker_id: int = 0,
        workers_per_rank: int = 1,
        canonical_replicas: int = 1,
    ) -> Any:  # pragma: no cover - not used in these tests
        return None

    def state_dict(self) -> dict[str, Any]:  # pragma: no cover - not used
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:  # pragma: no cover
        return None

    def supports_indexing(self) -> bool:  # pragma: no cover - not used
        return False

    def __len__(self) -> int:  # pragma: no cover - not used
        return 0

    def sample_id_at(self, index: int) -> Any:  # pragma: no cover - not used
        raise NotImplementedError


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
    eng = Engine(plan, opts, work)
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
    new = Engine(plan, RuntimeOptions(**opts), work)
    return new


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


def test_global_equal_weighting_caps() -> None:
    eng, _ = _mk_three_stage_plan((2, 5, 1))
    eng = _rebuild_with_opts(
        eng,
        worker_allocation="global",
        stage_weighting="equal",
        max_workers=9,
    )
    caps = [getattr(r, "_max_workers") for r in eng._runners]  # type: ignore[attr-defined]
    assert caps == [3, 3, 3]
    exp = eng.explain()
    assert "Allocation=global total=9 weighting=equal" in exp
    assert _extract_caps_from_explain(exp) == [3, 3, 3]


def test_global_by_declared_parallelism_caps() -> None:
    eng, expected_fit = _mk_three_stage_plan((2, 5, 1))
    assert expected_fit == [2, 5, 1]
    eng = _rebuild_with_opts(
        eng,
        worker_allocation="global",
        stage_weighting="by_declared_parallelism",
        max_workers=16,
    )
    # weights proportional to [2,5,1] with total=16 → [4,10,2]
    caps = [getattr(r, "_max_workers") for r in eng._runners]  # type: ignore[attr-defined]
    assert caps == [4, 10, 2]
    exp = eng.explain()
    assert "Allocation=global total=16 weighting=by_declared_parallelism" in exp
    assert _extract_caps_from_explain(exp) == [4, 10, 2]


def test_global_total_less_than_stages_warns_and_bumps() -> None:
    eng, _ = _mk_three_stage_plan((2, 5, 1))
    with pytest.warns(RuntimeWarning) as rec:
        eng = _rebuild_with_opts(
            eng,
            worker_allocation="global",
            stage_weighting="equal",
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
