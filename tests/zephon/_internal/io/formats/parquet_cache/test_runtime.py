# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from zephon._internal.io.formats.parquet import ParquetShardOpener
from zephon._internal.io.formats.parquet_cache import runtime as runtime_module
from zephon._internal.io.formats.parquet_cache.session import (
    ParquetRGSessionInUseError,
)
from zephon._internal.io.resolvers.direct import DirectResolver
from zephon._internal.io.stores.file_backed import FileBackedDatasetShardView
from zephon._internal.io.stores.multi import build_multi_dataset_store
from zephon.io.dataset import Dataset
from zephon.io.options import CacheOptions, ParquetRGCacheOptions, StoreOptions


def test_repeated_initialization_failure_warns_once_per_process_and_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingCache:
        def __init__(self, **_kwargs: object) -> None:
            raise RuntimeError("simulated initialization failure")

    monkeypatch.setattr(
        runtime_module,
        "ParquetRGIndex",
        lambda _catalog_set: SimpleNamespace(num_row_groups=1),
    )
    monkeypatch.setattr(runtime_module, "ParquetRGCache", FailingCache)
    monkeypatch.setattr(
        runtime_module,
        "validate_parquet_cache_disk_space",
        lambda _options: None,
    )
    runtime_module._cache_failure_warning_keys.clear()
    options = StoreOptions(
        cache=CacheOptions(enabled=True, root=tmp_path / "raw-cache"),
        parquet_rg_cache=ParquetRGCacheOptions(
            limit_bytes=16 * 1024 * 1024,
            min_free_bytes=0,
        ),
    )
    caplog.set_level(logging.WARNING, logger=runtime_module.__name__)

    first = runtime_module.build_parquet_cache_runtime(object(), options)  # type: ignore[arg-type]
    second = runtime_module.build_parquet_cache_runtime(object(), options)  # type: ignore[arg-type]

    assert first.cache is None
    assert second.cache is None
    failures = [
        record
        for record in caplog.records
        if "cache initialization failed" in record.getMessage()
    ]
    assert len(failures) == 1
    assert failures[0].exc_info is not None


def _parquet_dataset(
    tmp_path: Path,
    *,
    name: str = "parquet-data",
) -> Dataset:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
    pq.write_table(
        pa.table({"value": [10, 20, 30, 40, 50, 60]}),
        dataset_root / "part-000.parquet",
        row_group_size=3,
    )
    return Dataset.from_path(name, str(dataset_root), fmt="parquet")


def test_no_options_leave_decoded_cache_disabled(tmp_path: Path) -> None:
    store = build_multi_dataset_store({0: _parquet_dataset(tmp_path)})
    try:
        view = store.for_dataset(0)
        assert isinstance(view, FileBackedDatasetShardView)
        assert isinstance(getattr(view, "_resolver"), DirectResolver)
        opener = getattr(view, "_opener")
        assert isinstance(opener, ParquetShardOpener)
        assert getattr(opener, "_decoded_cache") is None
    finally:
        store.close()


def test_parquet_datasets_share_one_store_opener(tmp_path: Path) -> None:
    first = _parquet_dataset(tmp_path / "first", name="first")
    second = _parquet_dataset(tmp_path / "second", name="second")
    store = build_multi_dataset_store(
        {0: first, 1: second},
        options=StoreOptions(
            parquet_rg_cache=ParquetRGCacheOptions(
                root=tmp_path / "decoded-cache",
                limit_bytes=16 * 1024 * 1024,
                min_free_bytes=0,
            )
        ),
    )
    try:
        first_view = store.for_dataset(0)
        second_view = store.for_dataset(1)
        assert isinstance(first_view, FileBackedDatasetShardView)
        assert isinstance(second_view, FileBackedDatasetShardView)
        assert getattr(first_view, "_opener") is getattr(second_view, "_opener")

        first_shard, _ = first_view.open(0)
        second_shard, _ = second_view.open(0)
        first_rows, _ = first_shard.getsamples([0])
        second_rows, _ = second_shard.getsamples([1])
        assert int(first_rows[0]["value"]) == 10
        assert int(second_rows[0]["value"]) == 20
    finally:
        store.close()


def test_shard_cache_enables_and_locates_decoded_cache(tmp_path: Path) -> None:
    dataset = _parquet_dataset(tmp_path)
    raw_root = tmp_path / "raw-cache"
    store = build_multi_dataset_store(
        {0: dataset},
        options=StoreOptions(
            cache=CacheOptions(enabled=True, root=raw_root),
            parquet_rg_cache=ParquetRGCacheOptions(
                limit_bytes=16 * 1024 * 1024,
                min_free_bytes=0,
            ),
        ),
    )
    try:
        view = store.for_dataset(0)
        assert isinstance(view, FileBackedDatasetShardView)
        opener = getattr(view, "_opener")
        assert isinstance(opener, ParquetShardOpener)
        decoded_cache = getattr(opener, "_decoded_cache")
        assert decoded_cache.root == (raw_root / ".parquet-rg-cache").resolve()

        shard, _ = view.open(0)
        first_rows, _ = shard.getsamples([2, 0])
        second_rows, _ = shard.getsamples([1])

        assert [int(row["value"]) for row in first_rows] == [30, 10]
        assert [int(row["value"]) for row in second_rows] == [20]
        stats = decoded_cache.stats()
        assert stats["publications"] == 1
        assert stats["hits"] == 1
        assert stats["ready_count"] == 1
    finally:
        store.close()


def test_explicit_root_works_without_shard_cache(tmp_path: Path) -> None:
    dataset = _parquet_dataset(tmp_path)
    decoded_root = tmp_path / "decoded-cache"
    store = build_multi_dataset_store(
        {0: dataset},
        options=StoreOptions(
            parquet_rg_cache=ParquetRGCacheOptions(
                root=decoded_root,
                limit_bytes=16 * 1024 * 1024,
                min_free_bytes=0,
            )
        ),
    )
    try:
        view = store.for_dataset(0)
        assert isinstance(view, FileBackedDatasetShardView)
        assert isinstance(getattr(view, "_resolver"), DirectResolver)
        opener = getattr(view, "_opener")
        assert isinstance(opener, ParquetShardOpener)
        decoded_cache = getattr(opener, "_decoded_cache")
        assert decoded_cache.root == decoded_root.resolve()

        shard, _ = view.open(0)
        first_rows, _ = shard.getsamples([2])
        second_rows, _ = shard.getsamples([1])
        assert int(first_rows[0]["value"]) == 30
        assert int(second_rows[0]["value"]) == 20
        assert decoded_cache.stats()["publications"] == 1
        assert decoded_cache.stats()["hits"] == 1
    finally:
        store.close()


def test_live_configuration_mismatch_fails(tmp_path: Path) -> None:
    dataset = _parquet_dataset(tmp_path)
    decoded_root = tmp_path / "decoded-cache"
    first = build_multi_dataset_store(
        {0: dataset},
        options=StoreOptions(
            parquet_rg_cache=ParquetRGCacheOptions(
                root=decoded_root,
                limit_bytes=16 * 1024 * 1024,
                min_free_bytes=0,
            )
        ),
    )
    try:
        with pytest.raises(ParquetRGSessionInUseError, match="live incompatible"):
            build_multi_dataset_store(
                {0: dataset},
                options=StoreOptions(
                    parquet_rg_cache=ParquetRGCacheOptions(
                        root=decoded_root,
                        limit_bytes=32 * 1024 * 1024,
                        min_free_bytes=0,
                    )
                ),
            )
    finally:
        first.close()


def test_custom_root_must_not_overlap_raw_cache(tmp_path: Path) -> None:
    dataset = _parquet_dataset(tmp_path)
    raw_root = tmp_path / "raw-cache"
    with pytest.raises(ValueError, match="must not equal, contain, or be contained"):
        build_multi_dataset_store(
            {0: dataset},
            options=StoreOptions(
                cache=CacheOptions(enabled=True, root=raw_root),
                parquet_rg_cache=ParquetRGCacheOptions(root=raw_root / "custom-child"),
            ),
        )
