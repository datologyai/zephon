# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""In-memory shard stores backing ``Dataset.from_dict``."""

from typing import Mapping

from zephon._internal.io.protocols import (
    DatasetShardView,
    MultiDatasetShardStore,
    RandomAccessShard,
)


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
