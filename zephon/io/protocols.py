"""Core shard IO protocols."""

from typing import Protocol


class RandomAccessShard(Protocol):
    """A shard that supports len/index style random access."""

    def __getitem__(self, index: int) -> dict[str, object]: ...

    def __len__(self) -> int: ...

    def close(self) -> None: ...


class DatasetShardView(Protocol):
    """Shards belonging to a specific dataset."""

    def open(self, shard_id: int) -> RandomAccessShard: ...


class MultiDatasetShardStore(Protocol):
    """Shard store that spans multiple datasets."""

    def for_dataset(self, dataset_id: int) -> DatasetShardView: ...


__all__ = [
    "RandomAccessShard",
    "DatasetShardView",
    "MultiDatasetShardStore",
]
