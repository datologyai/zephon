# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers used across unit and integration tests."""

from collections.abc import Mapping
from pathlib import Path

from zephon.io import InMemoryShard
from zephon.io.catalog import (
    CatalogSet,
    DatasetHeader,
    ShardCatalog,
    ShardCatalogHandle,
)
from zephon.io.catalog import io as _catalog_io
from zephon.io.catalog.builder import pack_locators
from zephon.io.dataset import Dataset
from zephon.io.types import ShardLocator


def catalog_set_from_locators(
    locators: list[ShardLocator],
    catalog_dir: Path,
    counts: Mapping[int, int] | None = None,
) -> CatalogSet:
    """Build a :class:`CatalogSet` from a flat list of synthetic locators.

    Lets tests exercise the production catalog-backed path without a real dataset
    on disk: it groups locators by dataset name, columnarizes each via
    :func:`pack_locators`, writes the artifact under ``catalog_dir``, and composes
    the per-dataset catalogs. ``counts`` maps ``shard_id -> num_rows`` (default 1).
    """
    by_name: dict[str, list[ShardLocator]] = {}
    for loc in locators:
        by_name.setdefault(loc.dataset, []).append(loc)

    entries: dict[int, tuple[str, ShardCatalog]] = {}
    for dataset_id, name in enumerate(sorted(by_name)):
        locs = by_name[name]
        header = DatasetHeader(
            name=name,
            root=locs[0].root,
            format=locs[0].format,
            path=locs[0].root,
        )
        per_shard = {int(loc.shard_id): loc for loc in locs}
        num_rows = {
            int(loc.shard_id): int((counts or {}).get(int(loc.shard_id), 1))
            for loc in locs
        }
        built = pack_locators(header, per_shard, num_rows)
        path = catalog_dir / f"{name}-{built.fingerprint[7:23]}"
        _catalog_io.write_atomic(path, built.file_bytes)
        entries[dataset_id] = (name, ShardCatalog(_catalog_io.load_mmap(path)))
    return CatalogSet(entries)


def attach_catalog(dataset: Dataset) -> ShardCatalog:
    """Finalize (if needed) and attach a file-backed dataset's shard catalog.

    Test-side equivalent of what the Engine / store builder does, for tests that
    want to inspect synthesized locators directly.

    When the dataset carries no ``catalog_handle``, one is synthesized from the
    backend descriptor (format kind + path).
    """
    handle = getattr(dataset, "catalog_handle", None)
    if handle is None:
        kind = dataset.backend.get("kind")
        assert isinstance(kind, str) and dataset.path is not None, (
            "attach_catalog requires a file-backed dataset"
        )
        header = DatasetHeader(
            name=dataset.name, root=dataset.path, format=kind, path=dataset.path
        )
        handle = ShardCatalogHandle(dataset=header)
    if handle.fingerprint is None:
        handle.finalize()
    return handle.attach()


def catalog_locators(
    dataset: Dataset,
) -> tuple[ShardCatalog, dict[int, ShardLocator]]:
    """Return ``(catalog, {shard_id -> synthesized ShardLocator})`` for a dataset.

    Locators are synthesized on demand from the dataset's node-local catalog.
    """
    catalog = attach_catalog(dataset)
    locators = {
        sid: catalog.locator_at(slot, dataset_name=dataset.name)
        for slot, sid in enumerate(catalog.ids().tolist())
    }
    return catalog, locators


def mk_dataset(name: str, shards: dict[int, int]) -> Dataset:
    """Create a Dataset backed by in-memory shards.

    Args:
        name: Dataset name (used as prefix in generated text payloads).
        shards: Mapping of shard ID to row count.

    Returns:
        A ``Dataset`` with ``InMemoryShard`` entries whose rows contain
        ``{"text": "<name>:<shard_id>:<row_index>"}``.
    """
    data: dict[int, InMemoryShard] = {}
    for sid, count in shards.items():
        rows = [{"text": f"{name}:{sid}:{i}"} for i in range(count)]
        data[int(sid)] = InMemoryShard(rows)
    return Dataset.from_dict(name, data)
