# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shard interfaces and in-memory helpers for feeding pipelines."""

from typing import Mapping, Protocol


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

    def __init__(self, shards: Mapping[int, InMemoryShard]):
        self._shards = dict(shards)

    def open(self, shard_id: int) -> RandomAccessShard:
        return self._shards[shard_id]


class InMemoryMultiDatasetStore(MultiDatasetShardStore):
    """Simple multi-dataset store backed by in-memory shard views."""

    def __init__(self, datasets: Mapping[int, DatasetShardView]):
        self._datasets = dict(datasets)

    def for_dataset(self, dataset_id: int) -> DatasetShardView:
        return self._datasets[dataset_id]
