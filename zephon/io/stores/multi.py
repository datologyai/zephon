"""Unified builder for multi-dataset shard stores."""

from pathlib import Path
from typing import Mapping, cast

from zephon.io.dataset import Dataset
from zephon.io.formats import ensure_builtin_formats
from zephon.io.formats.base import get_format
from zephon.io.memory import InMemoryDatasetStore
from zephon.io.options import StoreOptions
from zephon.io.protocols import (
    MultiDatasetShardStore,
    RandomAccessShard,
)
from zephon.io.resolvers import CacheManager, DirectResolver, ShardResolver
from zephon.io.storage import LocalFSBackend, RouterStorageBackend, StorageBackend
from zephon.io.stores.file_backed import FileBackedDatasetShardView
from zephon.io.stores.registry import DatasetStoreRegistry


def build_multi_dataset_store(
    datasets: Mapping[int, Dataset],
    *,
    storage: StorageBackend | None = None,
    options: StoreOptions | None = None,
) -> MultiDatasetShardStore:
    """Construct a registry-backed multi-dataset store mixing inmem and file-backed.

    Creates a single resolver (cache-backed or direct) and a per-dataset view
    using registered format handlers.
    """
    store_opts = StoreOptions.from_any(options)

    resolver: ShardResolver
    if store_opts.cache.enabled:
        # Default to router for discovery and cache-backed downloads; for direct
        # (no-cache) runs we still require local files.
        storage = RouterStorageBackend() if storage is None else storage
        cache_root = Path(store_opts.cache.root).expanduser()
        cache_manager = CacheManager(
            cache_root,
            storage,
            limit_bytes=store_opts.cache.limit_bytes,
            keep_zip=store_opts.cache.keep_zip,
            validate_hash=store_opts.cache.validate_hash,
            download_retry=store_opts.cache.download_retry,
            download_timeout=store_opts.cache.download_timeout,
        )
        resolver = cache_manager
    else:
        # No need for RouterStorageBackend in the no-cache case.
        storage = LocalFSBackend(root=Path("/")) if storage is None else storage
        assert isinstance(storage, LocalFSBackend)
        resolver = DirectResolver(
            storage,
            validate_hash=store_opts.cache.validate_hash,
        )

    registry = DatasetStoreRegistry()
    for dataset_id, dataset in datasets.items():
        backend = dataset.backend
        kind = backend.get("kind") if isinstance(backend, dict) else None
        if kind == "inmem":
            shards_obj = backend.get("shards", {}) if isinstance(backend, dict) else {}
            view = InMemoryDatasetStore(
                cast(Mapping[int, RandomAccessShard], shards_obj)
            )
        elif isinstance(kind, str):
            # Defer to any registered file-backed format handler (e.g. jsonl, mds, litdata)
            ensure_builtin_formats()
            try:
                handler = get_format(kind)
            except KeyError as exc:
                raise ValueError(
                    f"Dataset '{dataset.name}' missing or unsupported backend kind for multi-store"
                ) from exc
            view = FileBackedDatasetShardView(
                dataset=dataset,
                handler=handler,
                resolver=resolver,
                retry_attempts=store_opts.cache.open_retry_attempts,
                retry_initial_backoff=store_opts.cache.open_retry_initial_backoff,
                retry_max_backoff=store_opts.cache.open_retry_max_backoff,
            )
        else:
            raise ValueError(
                f"Dataset '{dataset.name}' missing or unsupported backend kind for multi-store"
            )
        registry.register(int(dataset_id), view)

    return registry


__all__ = ["build_multi_dataset_store"]
