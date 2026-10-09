"""Unified builder for multi-dataset shard stores."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Iterator, Mapping, cast

import numpy as np

from zephon._internal.io.catalog import CatalogSet, ShardCatalog, set_catalog_dir
from zephon._internal.io.formats import ensure_builtin_formats
from zephon._internal.io.formats.base import ShardOpener, get_format
from zephon._internal.io.formats.parquet_cache.runtime import (
    validate_parquet_cache_disk_space,
)
from zephon._internal.io.memory import InMemoryDatasetStore
from zephon._internal.io.protocols import (
    DatasetShardView,
    RandomAccessShard,
)
from zephon._internal.io.resolvers import CacheManager, DirectResolver, ShardResolver
from zephon._internal.io.resolvers.cache.layout import (
    validate_raw_cache_dataset_names,
)
from zephon._internal.io.storage import (
    LocalFSBackend,
    RouterStorageBackend,
    StorageBackend,
)
from zephon._internal.io.stores.file_backed import FileBackedDatasetShardView
from zephon._internal.io.stores.registry import DatasetStoreRegistry
from zephon._internal.io.types import ShardLocator
from zephon._internal.utils.disk import check_cache_disk_space
from zephon.io.dataset import Dataset
from zephon.io.options import StoreOptions


def _cross_check_counts(dataset: Dataset, catalog: ShardCatalog) -> None:
    """Guard the ``discover_counts`` <-> ``discover`` seam.

    The work source's cursor is sized from ``from_path``'s count arrays, while
    the catalog's slots/locators come from the full ``discover``. A divergent
    shard ordering would silently mis-map counts to files, so when both are
    present in the same process we assert they agree.
    """
    ids = dataset._ids
    counts = dataset._counts
    if ids is None or counts is None:
        return
    if not (
        np.array_equal(catalog.ids(), ids)
        and np.array_equal(catalog.num_rows(), counts)
    ):
        raise RuntimeError(
            f"Catalog for dataset {dataset.name!r} disagrees with count-only "
            "discovery on (shard_id, num_rows); discover_counts() and discover() "
            "must assign the same shard ordering."
        )


def _ensure_attached(dataset: Dataset) -> ShardCatalog:
    """Attach the catalog for a file-backed dataset, finalizing if needed."""
    handle = dataset._catalog_handle
    assert handle is not None
    catalog = handle.ensure_attached()
    _cross_check_counts(dataset, catalog)
    return catalog


def finalize_dataset_catalogs(
    datasets: Mapping[int, Dataset], options: StoreOptions | None = None
) -> None:
    """Set the process-global catalog dir and finalize each file-backed handle.

    Called by the Engine before runners spawn: this is the one full build per
    node (source-key-locked in ``finalize``), and it bakes each handle's
    ``fingerprint`` onto the shared handle that ships in ctx.
    """
    set_catalog_dir(StoreOptions.from_any(options))
    for dataset in datasets.values():
        if dataset._catalog_handle is None:
            continue  # in-memory dataset
        _ensure_attached(dataset)


def _attached_entries(
    datasets: Mapping[int, Dataset],
) -> dict[int, tuple[str, ShardCatalog]]:
    """Attach every file-backed dataset's catalog, keyed by dataset id.

    Rejects duplicate dataset names across the whole map (in-memory included):
    names key the cache slot space, disk layout, and fingerprint, so a collision
    is ambiguous for any by-name lookup.
    """
    seen: set[str] = set()
    for dataset in datasets.values():
        if dataset.name in seen:
            raise ValueError(f"Duplicate dataset name {dataset.name!r} across datasets")
        seen.add(dataset.name)

    # Register the format modules for the file-backed kinds (their handler AND
    # extra codec) before attaching catalogs / synthesizing locators. A freshly
    # spawned worker never ran from_path, so its registry is otherwise empty.
    kinds = {
        kind
        for d in datasets.values()
        if isinstance((b := d.backend), dict)
        and isinstance((kind := b.get("kind")), str)
        and kind != "inmem"
    }
    ensure_builtin_formats(required=kinds)

    entries: dict[int, tuple[str, ShardCatalog]] = {}
    for dataset_id, dataset in datasets.items():
        if dataset._catalog_handle is None:
            continue  # in-memory dataset
        entries[int(dataset_id)] = (dataset.name, _ensure_attached(dataset))
    return entries


def build_catalog_set(datasets: Mapping[int, Dataset]) -> CatalogSet | None:
    """Compose a :class:`CatalogSet` over the file-backed datasets, if any.

    Attaches (finalizing first if the handle is not yet baked, e.g. on a Ray
    actor or in direct/non-Engine use) each file-backed dataset's catalog. Skips
    in-memory datasets; a file-backed dataset that reaches the view without a
    catalog handle raises there (``from_path`` always builds one).
    """
    entries = _attached_entries(datasets)
    return CatalogSet(entries) if entries else None


class CatalogLocators(Mapping[tuple[int, int], ShardLocator]):
    """Lazy ``(dataset_id, shard_id) -> ShardLocator`` view over catalogs.

    PrefetchOp consults this by key per shard (never a full scan), so a locator
    is synthesized only when actually requested — no resident per-shard dict.
    """

    def __init__(self, entries: Mapping[int, tuple[str, ShardCatalog]]) -> None:
        self._entries = dict(entries)

    def __getitem__(self, key: tuple[int, int]) -> ShardLocator:
        dataset_id, shard_id = key
        entry = self._entries.get(dataset_id)
        if entry is None:
            raise KeyError(key)
        name, catalog = entry
        try:
            slot = catalog.slot_of(shard_id)
        except KeyError:
            raise KeyError(key) from None
        return catalog.locator_at(slot, dataset_name=name)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for dataset_id, (_name, catalog) in self._entries.items():
            for shard_id in catalog.ids().tolist():
                yield (dataset_id, shard_id)

    def __len__(self) -> int:
        return sum(catalog.shard_count for _name, catalog in self._entries.values())


def has_cacheable_dataset(datasets: Mapping[int, Dataset]) -> bool:
    """Return ``True`` if any dataset is file-backed (a non-inmem string kind)."""
    cacheable, _parquet = _cacheable_dataset_flags(datasets)
    return cacheable


def _cacheable_dataset_flags(
    datasets: Mapping[int, Dataset],
) -> tuple[bool, bool]:
    cacheable = False
    parquet = False
    for dataset in datasets.values():
        backend = dataset.backend
        kind = backend.get("kind") if isinstance(backend, dict) else None
        if not isinstance(kind, str) or kind == "inmem":
            continue
        cacheable = True
        parquet = parquet or kind == "parquet"
    return cacheable, parquet


def validate_store_cache_disk_space(
    datasets: Mapping[int, Dataset],
    options: StoreOptions | None = None,
) -> None:
    """Preflight the complete on-disk cache budget before workers start."""
    store_opts = StoreOptions.from_any(options)
    cacheable, parquet = _cacheable_dataset_flags(datasets)
    if not cacheable:
        return
    if parquet:
        validate_parquet_cache_disk_space(store_opts)
        return
    if store_opts.cache.enabled:
        check_cache_disk_space(
            Path(store_opts.cache.root).expanduser().resolve(),
            store_opts.cache.limit_bytes,
        )


def build_resolver(
    catalog_set: CatalogSet | None,
    options: StoreOptions | None = None,
    storage: StorageBackend | None = None,
) -> ShardResolver:
    """Build a shard resolver (``CacheManager`` or ``DirectResolver``).

    A cache-backed ``CacheManager`` over the ``CatalogSet`` when the cache is
    enabled and there are shards, else a stateless ``DirectResolver``.
    """
    store_opts = StoreOptions.from_any(options)
    if store_opts.cache.enabled and catalog_set is not None and catalog_set.num_shards:
        validate_raw_cache_dataset_names(
            Path(store_opts.cache.root), catalog_set.cacheable_names
        )
        storage = RouterStorageBackend() if storage is None else storage
        cache_root = Path(store_opts.cache.root).expanduser()
        return CacheManager(
            cache_root,
            storage,
            catalog_set=catalog_set,
            limit_bytes=store_opts.cache.limit_bytes,
            keep_zip=store_opts.cache.keep_zip,
            validate_hash=store_opts.cache.validate_hash,
            download_retry=store_opts.cache.download_retry,
            download_timeout=store_opts.cache.download_timeout,
            min_slack_bytes=store_opts.cache.min_slack_bytes,
            max_slack_bytes=store_opts.cache.max_slack_bytes,
        )

    storage = LocalFSBackend(root=Path("/")) if storage is None else storage
    assert isinstance(storage, LocalFSBackend)
    return DirectResolver(storage, validate_hash=store_opts.cache.validate_hash)


def build_resolver_with_locators(
    datasets: Mapping[int, Dataset],
    *,
    storage: StorageBackend | None = None,
    options: StoreOptions | None = None,
    skip_inmem: bool = True,
) -> tuple[ShardResolver, Mapping[tuple[int, int], ShardLocator]]:
    """Build a resolver plus a lazy ``(dataset_id, shard_id) -> ShardLocator`` map.

    Used by PrefetchOp; the map synthesizes locators on demand, never all at once.
    """
    del skip_inmem  # in-memory datasets are always excluded
    store_opts = StoreOptions.from_any(options)
    entries = _attached_entries(datasets)
    catalog_set = CatalogSet(entries) if entries else None
    resolver = build_resolver(catalog_set, options=store_opts, storage=storage)
    return resolver, CatalogLocators(entries)


def build_multi_dataset_store(
    datasets: Mapping[int, Dataset],
    *,
    storage: StorageBackend | None = None,
    options: StoreOptions | None = None,
) -> DatasetStoreRegistry:
    """Construct a registry-backed multi-dataset store mixing inmem and file-backed.

    Creates a single resolver (cache-backed or direct) and a per-dataset view
    backed by the node-local catalog.
    """
    store_opts = StoreOptions.from_any(options)
    catalog_set = build_catalog_set(datasets)
    resolver = build_resolver(catalog_set, options=store_opts, storage=storage)
    resources = ExitStack()
    if isinstance(resolver, CacheManager):
        resources.callback(resolver.close)
    openers: dict[str, ShardOpener] = {}
    registry = DatasetStoreRegistry(
        on_close=resources.close,
    )
    try:
        for dataset_id, dataset in datasets.items():
            backend = dataset.backend
            kind = backend.get("kind") if isinstance(backend, dict) else None
            if kind == "inmem":
                assert isinstance(backend, dict)
                shards_obj = backend.get("shards", {})
                view: DatasetShardView = InMemoryDatasetStore(
                    cast(Mapping[int, RandomAccessShard], shards_obj)
                )
            elif isinstance(kind, str):
                try:
                    handler = get_format(kind)
                except KeyError as exc:
                    raise ValueError(
                        f"Dataset '{dataset.name}' missing or unsupported backend kind for multi-store"
                    ) from exc
                if kind not in openers:
                    openers[kind] = resources.enter_context(
                        handler.create_opener(catalog_set, store_opts)
                    )
                opener = openers[kind]
                catalog: ShardCatalog | None = None
                if catalog_set is not None and dataset._catalog_handle is not None:
                    catalog = catalog_set.catalog_for(dataset.name)
                view = FileBackedDatasetShardView(
                    dataset=dataset,
                    opener=opener,
                    resolver=resolver,
                    retry_attempts=store_opts.cache.open_retry_attempts,
                    retry_initial_backoff=store_opts.cache.open_retry_initial_backoff,
                    retry_max_backoff=store_opts.cache.open_retry_max_backoff,
                    catalog=catalog,
                )
            else:
                raise ValueError(
                    f"Dataset '{dataset.name}' missing or unsupported backend kind for multi-store"
                )
            registry.register(int(dataset_id), view)
    except Exception:
        registry.close()
        raise

    return registry


__all__ = [
    "CatalogLocators",
    "build_catalog_set",
    "build_multi_dataset_store",
    "build_resolver",
    "build_resolver_with_locators",
    "finalize_dataset_catalogs",
    "has_cacheable_dataset",
    "validate_store_cache_disk_space",
]
