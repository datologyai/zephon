from zephon.core.graph import Graph
from zephon.core.planner import Planner
from zephon.ops.delay import DelayById
from zephon.ops.materialize import Materialize


def test_planner_single_stage_explain_and_indexable_true() -> None:
    g = Graph()
    g.add("op0", DelayById(max_delay_ms=0.0))
    g.add("op1", DelayById(max_delay_ms=0.0), *g.nodes)
    plan = Planner().make_plan(g)

    assert len(plan.stages) == 1
    assert plan.indexable is True
    exp = plan.explain.splitlines()
    assert len(exp) == 1
    assert "Stage[0]" in exp[0]
    assert "place=auto" in exp[0]
    assert "ops=['op0@p8', 'op1@p8']" in exp[0]


def test_planner_splits_on_placement_and_sets_indexable_false() -> None:
    g = Graph()
    g.add("a", DelayById(max_delay_ms=0.0))
    g.add("b", DelayById(max_delay_ms=0.0), *g.nodes, placement="local")
    # Add a non-indexable op to flip indexable flag
    g.add("c", Materialize(), *g.nodes, placement="local")
    plan = Planner().make_plan(g)

    assert len(plan.stages) == 2
    assert plan.stages[0].placement == "auto"
    assert plan.stages[1].placement == "local"
    # First stage break reason is from initial state (start), then placement-hint
    assert plan.stages[1].break_reason in {"placement-hint", "barrier"}
    assert plan.indexable is False
