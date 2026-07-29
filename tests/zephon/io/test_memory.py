from zephon.io.memory import InMemoryShard


def test_inmemory_shard() -> None:
    shard = InMemoryShard([{"v": 1}, {"v": 2}])
    assert len(shard) == 2
    assert shard[0] == {"v": 1}
    # Bulk path returns list-only and preserves order with duplicates
    assert [r["v"] for r in shard.getsamples([1, 0, 1])] == [2, 1, 2]
    shard.close()


def test_inmemory_shard_raw_bytes() -> None:
    shard = InMemoryShard([{"text": "abcd"} for _ in range(10)])
    assert shard.raw_bytes == 40
    assert shard.raw_bytes == 40


def test_inmemory_shard_raw_bytes_empty() -> None:
    assert InMemoryShard([]).raw_bytes == 0
