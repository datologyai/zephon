from zephon.io.memory import (
    InMemoryDatasetStore,
    InMemoryMultiDatasetStore,
    InMemoryShard,
)


def test_inmemory_shard_and_stores() -> None:
    shard = InMemoryShard([{"v": 1}, {"v": 2}])
    assert len(shard) == 2
    assert shard[0] == {"v": 1}
    shard.close()

    ds_view = InMemoryDatasetStore({0: shard})
    opened, reused = ds_view.open(0)
    assert opened is shard
    assert reused is True

    multi = InMemoryMultiDatasetStore({5: ds_view})
    assert multi.for_dataset(5) is ds_view
