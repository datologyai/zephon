"""Core shard IO protocols."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class SampleLoadStats:
    """Timing information captured while loading a single sample."""

    resolve_ns: int = 0
    open_ns: int = 0
    read_ns: int = 0
    close_ns: int = 0
    touch_ns: int = 0
    retries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    optimistic_reuses: int = 0


class RandomAccessShard(Protocol):
    """A shard that supports len/index style random access."""

    def __getitem__(
        self, index: int
    ) -> dict[str, object] | tuple[dict[str, object], SampleLoadStats]: ...

    def __len__(self) -> int: ...

    def close(self) -> None: ...


class DatasetShardView(Protocol):
    """Shards belonging to a specific dataset."""

    def open(self, shard_id: int) -> tuple[RandomAccessShard, bool]: ...


class MultiDatasetShardStore(Protocol):
    """Shard store that spans multiple datasets."""

    def for_dataset(self, dataset_id: int) -> DatasetShardView: ...


__all__ = [
    "RandomAccessShard",
    "DatasetShardView",
    "MultiDatasetShardStore",
]
