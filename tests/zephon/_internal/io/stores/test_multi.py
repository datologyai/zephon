from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import Mock

import numpy as np
import pytest

from tests._catalog_helpers import catalog_set_from_locators
from tests.helpers.storage import MemoryStorage
from zephon._internal.io.catalog import CatalogSet
from zephon._internal.io.formats.base import ShardOpener
from zephon._internal.io.formats.jsonl import JsonlFormat
from zephon._internal.io.resolvers.cache.manager import CacheManager
from zephon._internal.io.resolvers.direct import DirectResolver
from zephon._internal.io.storage import RouterStorageBackend
from zephon._internal.io.stores import multi as multi_store
from zephon._internal.io.stores.file_backed import FileBackedDatasetShardView
from zephon._internal.io.stores.multi import (
    build_catalog_set,
    build_multi_dataset_store,
    build_resolver,
    build_resolver_with_locators,
    finalize_dataset_catalogs,
    has_cacheable_dataset,
    validate_store_cache_disk_space,
)
from zephon._internal.io.stores.registry import DatasetStoreRegistry
from zephon._internal.io.types import ShardFile, ShardLocator
from zephon._internal.utils.disk import InsufficientCacheSpaceError
from zephon.io import InMemoryShard
from zephon.io.dataset import Dataset
from zephon.io.options import (
    CacheOptions,
    ParquetOptions,
    ParquetRGCacheOptions,
    StoreOptions,
)

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


@pytest.mark.parametrize(
    "root", ["s3://bucket/data", "gs://bucket/data", "hf://datasets/org/repo"]
)
@pytest.mark.parametrize("explicit_storage", [False, True])
def test_remote_root_requires_shard_cache(
    tmp_path: Path, root: str, explicit_storage: bool
) -> None:
    # Include a local dataset first: every catalog root must be checked.
    catalogs = catalog_set_from_locators(
        [
            ShardLocator(
                dataset=name,
                shard_id=0,
                format="jsonl",
                root=path,
                raw=ShardFile("data.jsonl", 2, {}),
            )
            for name, path in (("local", str(tmp_path)), ("remote", root))
        ],
        tmp_path / "catalogs",
    )
    storage = MemoryStorage(root, {}) if explicit_storage else None

    with pytest.raises(ValueError, match=r"set cache\.enabled=True") as exc:
        build_resolver(
            catalogs,
            options=StoreOptions(cache=CacheOptions(enabled=False)),
            storage=storage,
        )
    assert root in str(exc.value)


def test_nonlocal_storage_requires_shard_cache() -> None:
    with pytest.raises(ValueError, match=r"set cache\.enabled=True"):
        build_resolver(
            None,
            options=StoreOptions(cache=CacheOptions(enabled=False)),
            storage=RouterStorageBackend(),
        )


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
        parquet=ParquetOptions(
            rg_cache=ParquetRGCacheOptions(
                enabled=True,
                root=tmp_path / "decoded",
                limit_bytes=4 * GiB,
            )
        )
    )

    validate_store_cache_disk_space({0: dataset}, options)
    assert probed == [options.parquet.rg_cache.root]


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
            parquet=ParquetOptions(
                rg_cache=ParquetRGCacheOptions(root=raw_root / "custom-child")
            ),
        ),
    )
    store.close()


@pytest.fixture
def opener_dependencies(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock, Mock]:
    """Isolate store construction from catalogs, storage, and shard readers."""
    catalog_set = Mock(spec=CatalogSet)
    resolver = Mock(spec=CacheManager)
    views = Mock(side_effect=lambda **kwargs: Mock(spec=FileBackedDatasetShardView))
    monkeypatch.setattr(multi_store, "build_catalog_set", lambda datasets: catalog_set)
    monkeypatch.setattr(multi_store, "build_resolver", lambda *args, **kwargs: resolver)
    monkeypatch.setattr(multi_store, "FileBackedDatasetShardView", views)
    return catalog_set, resolver, views


def test_format_opener_shared_and_closed_once(
    monkeypatch: pytest.MonkeyPatch, opener_dependencies: tuple[Mock, Mock, Mock]
) -> None:
    catalog_set, resolver, views = opener_dependencies
    datasets = {
        i: Dataset(name=name, backend={"kind": "jsonl"})
        for i, name in enumerate(("first", "second"))
    }
    opener = Mock(spec=ShardOpener)
    events: list[str] = []

    @contextmanager
    def create_opener(
        self: JsonlFormat, catalogs: CatalogSet | None, options: StoreOptions
    ) -> Iterator[ShardOpener]:
        assert catalogs is catalog_set
        events.append("open")
        try:
            yield opener
        finally:
            events.append("close")

    monkeypatch.setattr(JsonlFormat, "create_opener", create_opener)
    store = build_multi_dataset_store(datasets)
    try:
        assert events == ["open"]
        assert views.call_count == 2
        assert all(call.kwargs["opener"] is opener for call in views.call_args_list)
        assert all(call.kwargs["resolver"] is resolver for call in views.call_args_list)
    finally:
        store.close()
        store.close()
    assert events == ["open", "close"]
    resolver.close.assert_called_once_with()


@pytest.mark.parametrize("failed_kind", ["jsonl", "parquet"])
def test_failed_opener_initialization_closes_resources(
    failed_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    opener_dependencies: tuple[Mock, Mock, Mock],
) -> None:
    """Failure closes the resolver and any previously initialized format opener."""
    _, resolver, _ = opener_dependencies
    datasets = {
        i: Dataset(name=kind, backend={"kind": kind})
        for i, kind in enumerate(("jsonl", "parquet"))
    }
    events: list[str] = []
    resolver.close.side_effect = lambda: events.append("resolver.close")

    def get_format(kind: str) -> SimpleNamespace:
        @contextmanager
        def create_opener(
            catalogs: CatalogSet | None, options: StoreOptions
        ) -> Iterator[ShardOpener]:
            if kind == failed_kind:
                raise RuntimeError("opener initialization failed")
            events.append(f"{kind}.open")
            try:
                yield Mock(spec=ShardOpener)
            finally:
                events.append(f"{kind}.close")

        return SimpleNamespace(create_opener=create_opener)

    monkeypatch.setattr(multi_store, "get_format", get_format)
    with pytest.raises(RuntimeError, match="opener initialization failed"):
        build_multi_dataset_store(datasets)
    expected = [] if failed_kind == "jsonl" else ["jsonl.open", "jsonl.close"]
    assert events == [*expected, "resolver.close"]
    resolver.close.assert_called_once_with()
