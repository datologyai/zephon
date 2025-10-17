from zephon.core.graph import Graph
from zephon.ops.delay import DelayById


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
