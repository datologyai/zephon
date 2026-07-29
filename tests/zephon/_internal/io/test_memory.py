from zephon._internal.io.memory import (
    InMemoryDatasetStore,
    InMemoryMultiDatasetStore,
)
from zephon.io.memory import InMemoryShard


def test_inmemory_stores_open_and_lookup() -> None:
    shard = InMemoryShard([{"v": 1}, {"v": 2}])
    ds_view = InMemoryDatasetStore({0: shard})
    opened, reused = ds_view.open(0)
    assert opened is shard
    assert reused is True

    multi = InMemoryMultiDatasetStore({5: ds_view})
    assert multi.for_dataset(5) is ds_view
