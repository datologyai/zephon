from zephon.core.world import World


def test_world_defaults() -> None:
    w = World()
    assert w.canonical_replicas == 1
    assert w.worker_id == 0
    assert w.workers_per_rank == 1
    assert w.physical_rank == 0
    assert w.num_ranks == 1
    assert w.lanes_for_rank == {0: [0]}
