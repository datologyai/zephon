# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Static work source implementation over an in-memory index.

Will be removed soon after finalizing StaticMixtureWorkSource.
"""

import random

from zephon.core.constants import SampleId
from zephon.work.base import WorkChunk, WorkSource


class StaticWorkSource(WorkSource):
    """Simple indexable work source over a fixed shard index."""

    def __init__(
        self,
        shard_index: dict[int, int],
        *,
        dataset_id: int = 0,
        chunk_size: int = 256,
        seed: int = 1234,
        shuffle: bool = True,  # TODO(MaxiBoether): probably do not want shuffling on the source level?
        component: str = "default",
    ) -> None:
        self._dataset_id = dataset_id
        self._all_ids: list[SampleId] = []
        for shard_id, count in shard_index.items():
            self._all_ids.extend((self._dataset_id, shard_id, i) for i in range(count))
        if shuffle:
            random.Random(seed).shuffle(self._all_ids)
        self._position = 0
        self._chunk_size = chunk_size
        self._seed = seed
        self._component = component

    def next_chunk(self) -> WorkChunk | None:
        if self._position >= len(self._all_ids):
            return None
        end = min(self._position + self._chunk_size, len(self._all_ids))
        ids = self._all_ids[self._position : end]
        self._position = end
        components = {self._component: list(ids)}
        return WorkChunk(components=components, seed=self._seed)

    def checkpoint(self) -> bytes:
        return b""

    def restore(self, state: bytes) -> None:
        return None

    def supports_indexing(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self._all_ids)

    def sample_id_at(self, index: int) -> SampleId:
        return self._all_ids[index]
