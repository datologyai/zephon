"""Unified builder for multi-dataset shard stores."""

from pathlib import Path
from typing import Mapping, cast

from zephon.io.dataset import Dataset
from zephon.io.formats import ensure_builtin_formats
from zephon.io.formats.base import get_format
from zephon.io.memory import InMemoryDatasetStore
from zephon.io.options import StoreOptions
from zephon.io.protocols import MultiDatasetShardStore, RandomAccessShard
from zephon.io.resolvers import CacheManager, DirectResolver, ShardResolver
from zephon.io.storage import LocalFSBackend, RouterStorageBackend, StorageBackend
from zephon.io.stores.file_backed import FileBackedDatasetShardView
from zephon.io.stores.registry import DatasetStoreRegistry
from zephon.io.types import ShardLocator


def collect_cacheable_locators(
    datasets: Mapping[int, Dataset],
) -> tuple[
    dict[tuple[int, int], ShardLocator],
    dict[int, dict[int, ShardLocator]],
]:
    """Build locators once for every file-backed dataset.

    Skips in-memory datasets and unknown format kinds. Returns both a flat
    map keyed by ``(dataset_id, shard_id)`` (what PrefetchOp iterates) and a
    per-dataset nested map keyed by ``shard_id`` (what each view needs).

    This centralises the ``handler.build_locators(dataset)`` call so
    downstream builders can thread pre-built locators into both the
    ``ShardResolver`` and the per-dataset views without recomputing them.
    """
    kinds: set[str] = {
        kind
        for d in datasets.values()
        if isinstance((backend := d.backend), dict)
        and isinstance((kind := backend.get("kind")), str)
        and kind != "inmem"
    }
    ensure_builtin_formats(required=kinds)

    flat: dict[tuple[int, int], ShardLocator] = {}
    per_dataset: dict[int, dict[int, ShardLocator]] = {}

    for dataset_id, dataset in datasets.items():
        backend = dataset.backend
        kind = backend.get("kind") if isinstance(backend, dict) else None
        # Skip in-memory datasets (no file-backed shards) and entries whose
        # backend kind is absent or malformed.
        if kind == "inmem" or not isinstance(kind, str):
            continue
        try:
            handler = get_format(kind)
        except KeyError:
            # Skip datasets with unknown formats.
            continue

        # Build locators for all shards in this dataset.
        shard_locators = dict(handler.build_locators(dataset))
        per_dataset[int(dataset_id)] = shard_locators
        for shard_id, locator in shard_locators.items():
            flat[(int(dataset_id), int(shard_id))] = locator

    return flat, per_dataset


def build_resolver(
    locators: Mapping[tuple[int, int], ShardLocator],
    datasets: Mapping[int, Dataset],
    options: StoreOptions | None = None,
    storage: StorageBackend | None = None,
) -> ShardResolver:
    """Build a shard resolver based on store options.

    Creates either a ``CacheManager`` (for cache-enabled configs) or a
    ``DirectResolver`` (for direct local file access). The resolver can be
    shared across multiple operators (e.g., PrefetchOp and FetchOp) to
    ensure consistent cache behavior.

    Args:
        locators: Flat ``(dataset_id, shard_id) → ShardLocator`` map for all
            cacheable shards. Used by ``CacheManager`` to build its dense
            runtime index and to compute the session fingerprint.
        datasets: Mapping of dataset id to ``Dataset`` objects, used for
            fingerprint inputs (name, path).
        options: Store configuration options. If None, uses defaults.
        storage: Optional storage backend override.

    Returns:
        A ``ShardResolver`` instance (``CacheManager`` or ``DirectResolver``).
    """
    store_opts = StoreOptions.from_any(options)

    # When there are no cacheable shards (e.g. all datasets are in-memory),
    # there is nothing for the cache layer to manage. Fall back to a
    # DirectResolver regardless of ``cache.enabled`` — this avoids trying
    # to allocate a zero-sized SHM cache (which would raise ValueError in
    # CacheSharedState) and keeps the downstream contract intact because
    # the resolver is unused for inmem datasets anyway.
    if store_opts.cache.enabled and locators:
        # Default to router for discovery and cache-backed downloads.
        storage = RouterStorageBackend() if storage is None else storage
        cache_root = Path(store_opts.cache.root).expanduser()
        return CacheManager(
            cache_root,
            storage,
            locators=locators,
            datasets=datasets,
            limit_bytes=store_opts.cache.limit_bytes,
            keep_zip=store_opts.cache.keep_zip,
            validate_hash=store_opts.cache.validate_hash,
            download_retry=store_opts.cache.download_retry,
            download_timeout=store_opts.cache.download_timeout,
            min_slack_bytes=store_opts.cache.min_slack_bytes,
            max_slack_bytes=store_opts.cache.max_slack_bytes,
        )
    else:
        # No need for RouterStorageBackend in the no-cache case.
        storage = LocalFSBackend(root=Path("/")) if storage is None else storage
        assert isinstance(storage, LocalFSBackend)
        return DirectResolver(
            storage,
            validate_hash=store_opts.cache.validate_hash,
        )


def build_resolver_with_locators(
    datasets: Mapping[int, Dataset],
    *,
    storage: StorageBackend | None = None,
    options: StoreOptions | None = None,
    skip_inmem: bool = True,
) -> tuple[ShardResolver, dict[tuple[int, int], ShardLocator]]:
    """Build a resolver and extract locators for all file-backed datasets.

    Args:
        datasets: Mapping of dataset ID to Dataset objects.
        storage: Optional storage backend override.
        options: Store configuration options.
        skip_inmem: Retained for API compatibility; in-memory datasets are
            always excluded from the locator map since they have no
            file-backed shards.

    Returns:
        Tuple of (resolver, locators_dict) where locators_dict maps
        ``(dataset_id, shard_id)`` to ``ShardLocator``.
    """
    del skip_inmem  # inmem datasets are always excluded by collect_cacheable_locators
    store_opts = StoreOptions.from_any(options)
    flat_locators, _ = collect_cacheable_locators(datasets)
    resolver = build_resolver(
        flat_locators, datasets, options=store_opts, storage=storage
    )
    return resolver, flat_locators


def build_multi_dataset_store(
    datasets: Mapping[int, Dataset],
    *,
    storage: StorageBackend | None = None,
    options: StoreOptions | None = None,
) -> MultiDatasetShardStore:
    """Construct a registry-backed multi-dataset store mixing inmem and file-backed.

    Creates a single resolver (cache-backed or direct) and a per-dataset view
    using registered format handlers.

    Args:
        datasets: Mapping of dataset ID to Dataset objects.
        storage: Optional storage backend override.
        options: Store configuration options.

    Returns:
        A MultiDatasetShardStore registry.
    """
    store_opts = StoreOptions.from_any(options)
    flat_locators, per_dataset_locators = collect_cacheable_locators(datasets)
    resolver = build_resolver(
        flat_locators, datasets, options=store_opts, storage=storage
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
            # Defer to any registered file-backed format handler (e.g. jsonl,
            # mds, litdata). collect_cacheable_locators already called
            # ensure_builtin_formats above for every non-inmem kind seen,
            # so get_format() resolves without an extra registration step.
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
                locators=per_dataset_locators.get(int(dataset_id)),
            )
        else:
            raise ValueError(
                f"Dataset '{dataset.name}' missing or unsupported backend kind for multi-store"
            )
        registry.register(int(dataset_id), view)

    return registry


__all__ = [
    "build_multi_dataset_store",
    "build_resolver",
    "build_resolver_with_locators",
    "collect_cacheable_locators",
]
