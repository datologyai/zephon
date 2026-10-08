from pathlib import Path

import numpy as np
import pytest

from zephon._internal.io.resolvers.cache.manager import CacheManager
from zephon._internal.io.resolvers.direct import DirectResolver
from zephon._internal.io.stores.file_backed import FileBackedDatasetShardView
from zephon._internal.io.stores.multi import (
    build_catalog_set,
    build_multi_dataset_store,
    build_resolver_with_locators,
    finalize_dataset_catalogs,
    has_cacheable_dataset,
    validate_store_cache_disk_space,
)
from zephon._internal.io.stores.registry import DatasetStoreRegistry
from zephon._internal.utils.disk import InsufficientCacheSpaceError
from zephon.io import InMemoryShard
from zephon.io.dataset import Dataset
from zephon.io.options import CacheOptions, ParquetRGCacheOptions, StoreOptions

GiB = 1024**3


def _mk_jsonl_dataset(tmp_path: Path, *, name: str = "demo") -> Dataset:
    tmp_path.mkdir(parents=True, exist_ok=True)
    f0 = tmp_path / "s0.jsonl"
    f1 = tmp_path / "s1.jsonl"
    f0.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    f1.write_text('{"a": 3}\n', encoding="utf-8")
    return Dataset.from_path(name, str(tmp_path))


def test_build_multi_dataset_store_inmem_only() -> None:
    ds = Dataset.from_dict("mem", {0: InMemoryShard([{"x": 1}])})
    store = build_multi_dataset_store({7: ds})
    assert isinstance(store, DatasetStoreRegistry)
    view = store.for_dataset(7)
    # opening inmem shard returns the same shard object
    shard, reused = view.open(0)
    assert reused is True
    assert len(shard) == 1


def test_build_multi_dataset_store_file_backed_resolvers(tmp_path: Path) -> None:
    ds = _mk_jsonl_dataset(tmp_path)

    # No cache -> DirectResolver
    store = build_multi_dataset_store(
        {0: ds}, options=StoreOptions(cache=CacheOptions(enabled=False))
    )
    view = store.for_dataset(0)
    assert isinstance(view, FileBackedDatasetShardView)
    assert isinstance(getattr(view, "_resolver"), DirectResolver)

    # With cache -> CacheManager
    store_cached = build_multi_dataset_store(
        {0: ds}, options=StoreOptions(cache=CacheOptions(enabled=True, root=tmp_path))
    )
    view_cached = store_cached.for_dataset(0)
    assert isinstance(view_cached, FileBackedDatasetShardView)
    assert isinstance(getattr(view_cached, "_resolver"), CacheManager)


def test_build_multi_dataset_store_unknown_backend_kind_raises(tmp_path: Path) -> None:
    ds = Dataset(name="x", backend={"kind": "unknown"}, path=str(tmp_path))
    with pytest.raises(ValueError):
        _ = build_multi_dataset_store({0: ds})


def test_build_multi_dataset_store_mixed_inmem_and_file_backed(tmp_path: Path) -> None:
    file_ds = _mk_jsonl_dataset(tmp_path)
    mem_ds = Dataset.from_dict("mem", {0: InMemoryShard([{"x": 1}])})
    store = build_multi_dataset_store({0: file_ds, 1: mem_ds})

    file_view = store.for_dataset(0)
    assert isinstance(file_view, FileBackedDatasetShardView)
    shard, reused = file_view.open(0)
    assert reused is False
    assert len(shard) == 2  # length served from the catalog's num_rows column

    mem_shard, mem_reused = store.for_dataset(1).open(0)
    assert mem_reused is True
    assert len(mem_shard) == 1


def test_finalize_dataset_catalogs_bakes_fingerprints(tmp_path: Path) -> None:
    ds = _mk_jsonl_dataset(tmp_path)
    handle = ds._catalog_handle
    assert handle is not None
    assert handle.fingerprint is None

    finalize_dataset_catalogs({0: ds}, StoreOptions())
    baked = handle.fingerprint
    assert baked is not None

    # Idempotent: a second call (e.g. another rank on the node) keeps the bake.
    finalize_dataset_catalogs({0: ds}, StoreOptions())
    assert handle.fingerprint == baked


def test_finalize_dataset_catalogs_skips_inmem() -> None:
    ds = Dataset.from_dict("mem", {0: InMemoryShard([{"x": 1}])})
    finalize_dataset_catalogs({0: ds})  # no handle -> nothing to build
    assert ds._catalog_handle is None


def test_build_catalog_set_cross_checks_counts(tmp_path: Path) -> None:
    """A discover_counts()/discover() divergence must fail loud, not mis-map."""
    ds = _mk_jsonl_dataset(tmp_path)
    object.__setattr__(ds, "_counts", np.array([99, 99], dtype=np.int64))
    with pytest.raises(RuntimeError, match="disagrees with count-only"):
        build_catalog_set({0: ds})


def test_catalog_locators_lazy_map(tmp_path: Path) -> None:
    ds = _mk_jsonl_dataset(tmp_path)
    _resolver, locators = build_resolver_with_locators(
        {3: ds}, options=StoreOptions(cache=CacheOptions(enabled=False))
    )

    assert len(locators) == 2
    assert set(locators) == {(3, 0), (3, 1)}

    loc = locators[(3, 0)]
    assert loc.dataset == "demo"
    assert loc.shard_id == 0
    assert loc.raw.basename.endswith("s0.jsonl")

    # Misses: unknown shard in a known dataset, and unknown dataset.
    assert locators.get((3, 99)) is None
    assert locators.get((9, 0)) is None
    with pytest.raises(KeyError):
        _ = locators[(3, 99)]


def test_cache_manager_blocks_on_insufficient_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fresh cache (existing=0): the limit must fit in free space alone, and
    # 10GiB free < 50GiB limit, so CacheManager construction fails fast.
    ds = _mk_jsonl_dataset(tmp_path)
    monkeypatch.setattr(
        "zephon._internal.utils.disk.device_space", lambda path: (100 * GiB, 10 * GiB)
    )
    with pytest.raises(InsufficientCacheSpaceError):
        build_multi_dataset_store(
            {0: ds},
            options=StoreOptions(
                cache=CacheOptions(
                    enabled=True, root=tmp_path / "cache", limit_bytes=50 * GiB
                )
            ),
        )


def test_inmem_only_skips_disk_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # In-memory only: no CacheManager is built, so the disk check never runs —
    # even with the device reported nearly full.
    monkeypatch.setattr(
        "zephon._internal.utils.disk.device_space", lambda path: (100 * GiB, 1 * GiB)
    )
    ds = Dataset.from_dict("mem", {0: InMemoryShard([{"x": 1}])})
    store = build_multi_dataset_store(
        {0: ds},
        options=StoreOptions(cache=CacheOptions(enabled=True, limit_bytes=50 * GiB)),
    )
    _shard, reused = store.for_dataset(0).open(0)
    assert reused is True


def test_store_preflight_combines_derived_rg_limit_on_shared_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "zephon._internal.utils.disk.device_space", lambda path: (100 * GiB, 10 * GiB)
    )
    dataset = Dataset(name="pq", backend={"kind": "parquet"}, path="/data")
    options = StoreOptions(
        cache=CacheOptions(
            enabled=True,
            root=tmp_path / "cache",
            limit_bytes=8 * GiB,
        )
    )

    assert options.resolved_parquet_rg_cache().limit_bytes == 4 * GiB
    with pytest.raises(InsufficientCacheSpaceError):
        validate_store_cache_disk_space({0: dataset}, options)


def test_decoded_only_preflight_never_probes_raw_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probed: list[Path] = []
    monkeypatch.setattr(
        "zephon._internal.utils.disk.device_id",
        lambda path: probed.append(Path(path)) or 1,
    )
    monkeypatch.setattr(
        "zephon._internal.utils.disk.device_space", lambda path: (100 * GiB, 10 * GiB)
    )
    dataset = Dataset(name="pq", backend={"kind": "parquet"}, path="/data")
    options = StoreOptions(
        parquet_rg_cache=ParquetRGCacheOptions(
            enabled=True,
            root=tmp_path / "decoded",
            limit_bytes=4 * GiB,
        )
    )

    validate_store_cache_disk_space({0: dataset}, options)
    assert probed == [options.parquet_rg_cache.root]


def test_has_cacheable_dataset(tmp_path: Path) -> None:
    inmem = Dataset.from_dict("mem", {0: InMemoryShard([{"x": 1}])})
    file_backed = _mk_jsonl_dataset(tmp_path)
    assert has_cacheable_dataset({}) is False
    assert has_cacheable_dataset({0: inmem}) is False
    assert has_cacheable_dataset({0: inmem, 1: file_backed}) is True


def test_jsonl_store_ignores_parquet_cache_root(tmp_path: Path) -> None:
    ds = _mk_jsonl_dataset(tmp_path / "dataset")
    raw_root = tmp_path / "raw-cache"
    # Overlap validation is intentionally Parquet-only: JSONL never opens the
    # decoded row-group cache.
    store = build_multi_dataset_store(
        {0: ds},
        options=StoreOptions(
            cache=CacheOptions(enabled=True, root=raw_root),
            parquet_rg_cache=ParquetRGCacheOptions(root=raw_root / "custom-child"),
        ),
    )
    store.close()


def test_remote_dataset_needs_cache_when_format_cannot_read_in_place() -> None:
    from types import SimpleNamespace

    from zephon._internal.io.formats.base import get_format
    from zephon._internal.io.formats.vortex import VortexFormat
    from zephon._internal.io.stores.multi import _check_readable_in_place

    remote = SimpleNamespace(name="remote", backend={"path": "s3://bucket/data"})
    local = SimpleNamespace(name="local", backend={"path": "/data"})
    jsonl = get_format("jsonl")
    uncached = StoreOptions()
    cached = StoreOptions(cache=CacheOptions(enabled=True))

    with pytest.raises(ValueError, match="cache.enabled=True"):
        _check_readable_in_place(remote, jsonl, uncached)  # type: ignore[arg-type]

    _check_readable_in_place(remote, jsonl, cached)  # type: ignore[arg-type]
    _check_readable_in_place(local, jsonl, uncached)  # type: ignore[arg-type]
    _check_readable_in_place(remote, VortexFormat(), uncached)  # type: ignore[arg-type]
