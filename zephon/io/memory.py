"""In-memory implementations of shard protocols."""

from typing import Mapping

from .protocols import DatasetShardView, MultiDatasetShardStore, RandomAccessShard


class InMemoryShard:
    """List-backed shard useful for tests and quickstarts."""

    def __init__(self, rows: list[dict[str, object]]):
        self._rows = rows

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._rows[index]

    def __len__(self) -> int:
        return len(self._rows)

    def close(self) -> None:
        return None


class InMemoryDatasetStore(DatasetShardView):
    """Dictionary-backed shard view for a single dataset."""

    def __init__(self, shards: Mapping[int, RandomAccessShard]):
        self._shards = dict(shards)

    def open(self, shard_id: int) -> tuple[RandomAccessShard, bool]:
        return self._shards[shard_id], True


class InMemoryMultiDatasetStore(MultiDatasetShardStore):
    """Simple multi-dataset store backed by in-memory shard views."""

    def __init__(self, datasets: Mapping[int, DatasetShardView]):
        self._datasets = dict(datasets)

    def for_dataset(self, dataset_id: int) -> DatasetShardView:
        return self._datasets[dataset_id]


__all__ = [
    "InMemoryDatasetStore",
    "InMemoryMultiDatasetStore",
    "InMemoryShard",
]
