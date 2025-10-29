from pathlib import Path

import pytest

from zephon.io import InMemoryShard
from zephon.io.dataset import Dataset
from zephon.io.options import CacheOptions, StoreOptions
from zephon.io.resolvers.cache.manager import CacheManager
from zephon.io.resolvers.direct import DirectResolver
from zephon.io.stores.file_backed import FileBackedDatasetShardView
from zephon.io.stores.multi import build_multi_dataset_store
from zephon.io.stores.registry import DatasetStoreRegistry


def _mk_jsonl_dataset(tmp_path: Path) -> Dataset:
    f0 = tmp_path / "s0.jsonl"
    f1 = tmp_path / "s1.jsonl"
    f0.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    f1.write_text('{"a": 3}\n', encoding="utf-8")
    return Dataset.from_path("demo", str(tmp_path))


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
    ds = Dataset(
        name="x", shard_index={0: 1}, backend={"kind": "unknown"}, path=str(tmp_path)
    )
    with pytest.raises(ValueError):
        _ = build_multi_dataset_store({0: ds})
