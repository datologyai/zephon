import pytest

from zephon.core.accumulators import Accumulator, PassthroughAccumulator
from zephon.core.graph import Graph
from zephon.core.op_base import DefaultSetup, Op
from zephon.core.planner import Planner
from zephon.core.traits import OpTraits
from zephon.ops.batch import Batch
from zephon.ops.delay import DelayById
from zephon.ops.map_transform import MapBatchTransform, MapTransform
from zephon.ops.materialize import Materialize
from zephon.ops.replay_filter import ReplayFilter


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
    stage0_names = [nd.name for nd in plan.stages[0].nodes]
    assert stage0_names == ["op0", "op1", "op1_replay_filter"]


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
    assert plan.stages[1].nodes[-1].name == "c_replay_filter"


def test_planner_injects_filter_before_batch() -> None:
    g = Graph()
    upstream = g.add("upstream", DelayById(max_delay_ms=0.0))
    g.add("batch", Batch(microbatch_size=4), upstream)

    plan = Planner().make_plan(g)
    batch_stage = next(
        stage
        for stage in plan.stages
        if any(isinstance(nd.op, Batch) for nd in stage.nodes)
    )
    names = [nd.name for nd in batch_stage.nodes]
    assert names == ["batch_replay_filter", "batch"]
    assert isinstance(batch_stage.nodes[0].op, ReplayFilter)


def test_planner_merges_batch_with_post_batch_ops() -> None:
    """When post-batch ops exist, they merge into the batch stage (not inline)."""
    g = Graph()
    upstream = g.add("upstream", DelayById(max_delay_ms=0.0))
    batch = g.add("batch", Batch(microbatch_size=4), upstream)
    g.add("tail", DelayById(max_delay_ms=0.0), batch)

    plan = Planner().make_plan(g)
    assert len(plan.stages) == 2

    head_nodes = [nd.name for nd in plan.stages[0].nodes]
    batch_nodes = [nd.name for nd in plan.stages[1].nodes]

    assert head_nodes == ["upstream"]
    assert batch_nodes == ["batch_replay_filter", "batch", "tail"]

    # Threads (not inline/process) because post-batch ops create tensors
    assert plan.stages[1].runner_hint == "threads"


def test_planner_batch_terminal_stays_inline() -> None:
    """When batch is the last op (no post-batch ops), it stays inline."""
    g = Graph()
    upstream = g.add("upstream", DelayById(max_delay_ms=0.0))
    g.add("batch", Batch(microbatch_size=4), upstream)

    plan = Planner().make_plan(g)
    assert len(plan.stages) == 2

    head_nodes = [nd.name for nd in plan.stages[0].nodes]
    batch_nodes = [nd.name for nd in plan.stages[1].nodes]

    assert head_nodes == ["upstream"]
    assert batch_nodes == ["batch_replay_filter", "batch"]

    assert plan.stages[1].runner_hint == "inline"


class _OrderedOp(DefaultSetup, Op[int, int]):
    def __init__(self) -> None:
        DefaultSetup.__init__(self)

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, preserves_cursor_order=True, parallelism=1)

    def accumulator(self, *, deterministic: bool) -> Accumulator[int]:
        return PassthroughAccumulator[int]()

    def process_one(self, elem: int) -> list[int]:
        return [elem]


class _ReorderingOp(_OrderedOp):
    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, preserves_cursor_order=False, parallelism=1)


def test_planner_marks_plan_cursor_order_when_all_ops_preserve() -> None:
    g = Graph()
    a = g.add("a", _OrderedOp())
    g.add("b", _OrderedOp(), a)
    plan = Planner().make_plan(g)
    assert plan.preserves_cursor_order is True


def test_planner_marks_plan_non_cursor_order_when_any_op_reorders() -> None:
    g = Graph()
    a = g.add("a", _OrderedOp())
    g.add("b", _ReorderingOp(), a)
    plan = Planner().make_plan(g)
    assert plan.preserves_cursor_order is False


def test_optraits_requires_preserves_cursor_order() -> None:
    """OpTraits enforces the trait declaration at construction time."""
    with pytest.raises(TypeError):
        OpTraits()  # type: ignore[call-arg]


def test_planner_rejects_map_transform_after_batch() -> None:
    """map_transform() after batch() must raise ValueError."""
    g = Graph()
    upstream = g.add("upstream", DelayById(max_delay_ms=0.0))
    batch = g.add("batch", Batch(microbatch_size=4), upstream)
    g.add("map_transform", MapTransform(lambda p: p), batch)

    with pytest.raises(ValueError, match="map_transform.*after batch"):
        Planner().make_plan(g)


def test_planner_rejects_map_batch_before_batch() -> None:
    """map_batch() without a preceding batch() must raise ValueError."""
    g = Graph()
    upstream = g.add("upstream", DelayById(max_delay_ms=0.0))
    g.add("map_batch", MapBatchTransform(lambda b: b), upstream)

    with pytest.raises(ValueError, match="map_batch.*preceding batch"):
        Planner().make_plan(g)


def test_planner_accepts_map_transform_before_batch() -> None:
    """map_transform() before batch() is valid."""
    g = Graph()
    upstream = g.add("upstream", DelayById(max_delay_ms=0.0))
    mt = g.add("map_transform", MapTransform(lambda p: p), upstream)
    g.add("batch", Batch(microbatch_size=4), mt)

    plan = Planner().make_plan(g)
    assert plan is not None


def test_planner_accepts_map_batch_after_batch() -> None:
    """map_batch() after batch() is valid."""
    g = Graph()
    upstream = g.add("upstream", DelayById(max_delay_ms=0.0))
    batch = g.add("batch", Batch(microbatch_size=4), upstream)
    g.add("map_batch", MapBatchTransform(lambda b: b), batch)

    plan = Planner().make_plan(g)
    assert plan is not None
