"""Shard view for file-backed datasets."""

from __future__ import annotations

from zephon.io.catalog import ShardCatalog
from zephon.io.dataset import Dataset
from zephon.io.formats.base import FormatHandler
from zephon.io.protocols import DatasetShardView, RandomAccessShard
from zephon.io.resolvers import ShardResolver
from zephon.io.stores.resilient import ResilientShard
from zephon.io.types import ShardLocator


class FileBackedDatasetShardView(DatasetShardView):
    """Dataset shard view backed by a format handler and the node-local catalog.

    Lengths come from the catalog's ``num_rows`` column and locators are
    synthesized per ``open`` via ``locator_at`` (no resident locator dict). The
    catalog is supplied explicitly or attached from the dataset's handle.
    """

    def __init__(
        self,
        *,
        dataset: Dataset,
        handler: FormatHandler,
        resolver: ShardResolver,
        retry_attempts: int,
        retry_initial_backoff: float,
        retry_max_backoff: float,
        catalog: ShardCatalog | None = None,
    ) -> None:
        self._name = dataset.name
        self._handler = handler
        self._resolver = resolver

        if catalog is None:
            handle = dataset.catalog_handle
            if handle is None:
                raise ValueError(
                    f"File-backed dataset {dataset.name!r} has no shard catalog "
                    "handle; build it via Dataset.from_path."
                )
            catalog = handle.ensure_attached()
        self._catalog = catalog

        self._retry_attempts = max(1, int(retry_attempts))
        self._retry_initial_backoff = max(0.0, float(retry_initial_backoff))
        self._retry_max_backoff = max(
            self._retry_initial_backoff, float(retry_max_backoff)
        )
        self._shards: dict[int, RandomAccessShard] = {}

    def _resolve_shard(self, shard_id: int) -> tuple[ShardLocator, int]:
        try:
            slot = self._catalog.slot_of(shard_id)
        except KeyError as exc:
            raise KeyError(
                f"Shard {shard_id} not found in dataset '{self._name}'"
            ) from exc
        locator = self._catalog.locator_at(slot, dataset_name=self._name)
        return locator, int(self._catalog.num_rows()[slot])

    def open(self, shard_id: int) -> tuple[RandomAccessShard, bool]:
        shard = self._shards.get(shard_id)
        if shard is not None:
            return shard, True
        locator, length = self._resolve_shard(shard_id)
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
        return shard, False


__all__ = [
    "FileBackedDatasetShardView",
]
