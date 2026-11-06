"""Core shard IO protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from zephon._internal.observability.size_estimator import content_bytes
from zephon.types import SamplePayload

#: Rows decoded per ``getsamples`` call when sizing, bounding payloads held at once.
_SIZE_BATCH = 2048


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
    """A shard that supports len/index style random access with bulk reads.

    Implementations MUST implement ``getsamples``. The return type mirrors the
    pattern used by ``__getitem__``: implementations may return just the rows or
    a tuple of (rows, per-sample stats). Implementations that do not surface
    per-sample stats can simply loop over ``__getitem__`` and return the rows.

    ``getsamples`` accepts indices in arbitrary order, including duplicates.
    Returned rows (and per-sample stats, when present) must preserve that order
    and include one entry per requested index without modifying ``indices``.
    Rows follow ``SamplePayload``, including scalar and array payloads.
    """

    def __getitem__(
        self, index: int
    ) -> SamplePayload | tuple[SamplePayload, SampleLoadStats]: ...

    def getsamples(
        self, indices: list[int]
    ) -> list[SamplePayload] | tuple[list[SamplePayload], list[SampleLoadStats]]: ...

    def __len__(self) -> int: ...

    def close(self) -> None: ...

    def content_sizes(self, indices: list[int]) -> np.ndarray:
        """``content_bytes`` of each requested row, in request order.

        Formats with columnar buffers override this to size rows unmaterialized.
        """
        sizes = np.empty(len(indices), dtype=np.int64)
        for start in range(0, len(indices), _SIZE_BATCH):
            got = self.getsamples(indices[start : start + _SIZE_BATCH])
            rows = got[0] if isinstance(got, tuple) else got
            sizes[start : start + len(rows)] = [content_bytes(row) for row in rows]
        return sizes


class DatasetShardView(Protocol):
    """Shards belonging to a specific dataset."""

    def open(self, shard_id: int) -> tuple[RandomAccessShard, bool]: ...


class MultiDatasetShardStore(Protocol):
    """Shard store that spans multiple datasets."""

    def for_dataset(self, dataset_id: int) -> DatasetShardView: ...

    def close(self) -> None:
        """Release owned resources; repeated calls must be safe."""
        ...


__all__ = [
    "RandomAccessShard",
    "DatasetShardView",
    "MultiDatasetShardStore",
]
