"""Shard view for file-backed datasets."""

from zephon.io.dataset import Dataset
from zephon.io.formats.base import FormatHandler
from zephon.io.protocols import DatasetShardView, RandomAccessShard
from zephon.io.resolvers import ShardResolver
from zephon.io.stores.resilient import ResilientShard


class FileBackedDatasetShardView(DatasetShardView):
    """Dataset shard view backed by a format handler and resolver."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        handler: FormatHandler,
        resolver: ShardResolver,
        retry_attempts: int,
        retry_initial_backoff: float,
        retry_max_backoff: float,
    ) -> None:
        self._dataset = dataset
        self._handler = handler
        self._resolver = resolver
        self._locators = dict(handler.build_locators(dataset))
        self._lengths = {
            int(sid): int(count) for sid, count in dataset.shard_index.items()
        }
        self._retry_attempts = max(1, int(retry_attempts))
        self._retry_initial_backoff = max(0.0, float(retry_initial_backoff))
        self._retry_max_backoff = max(
            self._retry_initial_backoff, float(retry_max_backoff)
        )
        self._shards: dict[int, RandomAccessShard] = {}

    def open(self, shard_id: int) -> RandomAccessShard:
        try:
            locator = self._locators[shard_id]
        except KeyError as exc:
            raise KeyError(
                f"Shard {shard_id} not found in dataset '{self._dataset.name}'"
            ) from exc
        shard = self._shards.get(shard_id)
        if shard is None:
            length = self._lengths.get(shard_id, 0)
            # TODO(MaxiBoether): right now we also use the ResilientShard for runs without a cache. Is this a problem?
            shard = ResilientShard(
                locator=locator,
                resolver=self._resolver,
                handler=self._handler,
                length=length,
                retry_attempts=self._retry_attempts,
                retry_initial_backoff=self._retry_initial_backoff,
                retry_max_backoff=self._retry_max_backoff,
            )
            self._shards[shard_id] = shard
        return shard


__all__ = [
    "FileBackedDatasetShardView",
]
