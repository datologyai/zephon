from zephon._internal.world import World


def test_world_defaults() -> None:
    w = World()
    assert w.canonical_replicas == 1
    assert w.worker_id == 0
    assert w.workers_per_rank == 1
    assert w.global_rank == 0
    assert w.world_size == 1
    assert w.dp_degree == 1
    assert w.dp_group_id == 0
    assert w.lanes_for_dp_group == {0: [0]}
