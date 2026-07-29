from zephon._internal.graph import Graph
from zephon._internal.ops.delay import DelayById
from zephon._internal.planner import Planner
from zephon.ops.base import BaseOp


def test_graph_add_defaults_and_ordering() -> None:
    g = Graph()
    n0 = g.add("a", DelayById())
    n1 = g.add("b", DelayById(), placement="local", parallelism=3, *g.nodes)

    # Graph preserves insertion order
    assert [n.name for n in g.nodes] == ["a", "b"]

    # Defaults: placement auto, parallelism from op.traits() (DelayById.parallelism=8)
    assert n0.placement == "auto"
    assert n0.parallelism == 8

    # Overrides respected
    assert n1.placement == "local"
    assert n1.parallelism == 3


def _make_plan(
    *,
    second_delay_ms: float = 2.0,
    second_placement: str = "auto",
) -> tuple[str, str]:
    graph = Graph()
    first = graph.add("first", DelayById(max_delay_ms=1.0))
    graph.add(
        "second",
        DelayById(max_delay_ms=second_delay_ms),
        first,
        placement=second_placement,
    )
    plan = Planner().make_plan(graph)
    return plan.fingerprint(), plan.plan_id


def test_plan_identifier_is_stable() -> None:
    fp_a, id_a = _make_plan()
    fp_b, id_b = _make_plan()

    assert fp_a == fp_b
    assert id_a == id_b
    assert id_a == fp_a[:16]


def test_plan_identifier_changes_for_different_plans() -> None:
    baseline_fp, baseline_id = _make_plan()
    different_op_fp, different_op_id = _make_plan(second_delay_ms=5.0)
    different_stage_fp, different_stage_id = _make_plan(second_placement="remote")

    assert baseline_fp != different_op_fp
    assert baseline_fp != different_stage_fp
    assert len({baseline_id, different_op_id, different_stage_id}) == 3


def _single_op_fingerprint(op: BaseOp) -> str:
    graph = Graph()
    graph.add("op", op)
    return Planner().make_plan(graph).fingerprint()


def test_plan_identifier_ignores_zephon_module_moves() -> None:
    moved_delay_type = type(
        "DelayById",
        (DelayById,),
        {"__module__": "zephon._internal.moved"},
    )

    assert DelayById.plan_identity() == "zephon.builtin.DelayById"
    assert _single_op_fingerprint(moved_delay_type()) == _single_op_fingerprint(
        DelayById()
    )


def test_plan_identifier_distinguishes_custom_modules() -> None:
    first_type = type("CustomDelay", (DelayById,), {"__module__": "first"})
    second_type = type("CustomDelay", (DelayById,), {"__module__": "second"})

    assert _single_op_fingerprint(first_type()) != _single_op_fingerprint(second_type())
