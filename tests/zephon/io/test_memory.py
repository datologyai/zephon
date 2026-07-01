from zephon.io.memory import (
    InMemoryDatasetStore,
    InMemoryMultiDatasetStore,
    InMemoryShard,
)


def test_inmemory_shard_and_stores() -> None:
    shard = InMemoryShard([{"v": 1}, {"v": 2}])
    assert len(shard) == 2
    assert shard[0] == {"v": 1}
    # Bulk path returns list-only and preserves order with duplicates
    assert [r["v"] for r in shard.getsamples([1, 0, 1])] == [2, 1, 2]
    shard.close()

    ds_view = InMemoryDatasetStore({0: shard})
    opened, reused = ds_view.open(0)
    assert opened is shard
    assert reused is True

    multi = InMemoryMultiDatasetStore({5: ds_view})
    assert multi.for_dataset(5) is ds_view


def test_inmemory_shard_raw_bytes() -> None:
    shard = InMemoryShard([{"text": "abcd"} for _ in range(10)])
    assert shard.raw_bytes == 40
    assert shard.raw_bytes == 40


def test_inmemory_shard_raw_bytes_empty() -> None:
    assert InMemoryShard([]).raw_bytes == 0
