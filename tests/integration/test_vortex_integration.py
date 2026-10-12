# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex pipeline reads, discovery, store wiring, and recovery."""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("vortex.io", reason="vortex-data not installed")
import vortex

from zephon import Pipeline
from zephon._internal.io.formats import vortex as vortex_format
from zephon._internal.io.formats.vortex import VortexFormat
from zephon._internal.io.resolvers.base import ShardResolver
from zephon._internal.io.stores.multi import build_multi_dataset_store
from zephon._internal.io.stores.resilient import ResilientShard
from zephon._internal.io.types import (
    LocalShardFile,
    LocalShardRef,
    ShardFile,
    ShardLocator,
)
from zephon.build_index import build_index
from zephon.io import CacheOptions, Dataset, StoreOptions, VortexOptions
from zephon.work import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def _write(path: Path, start: int = 0) -> None:
    vortex.io.write(
        vortex.array([{"value": i} for i in range(start, start + 8)]), str(path)
    )


def test_store_reuses_caches_across_fetch_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = {}
    for dataset_id in (0, 1):
        root = tmp_path / str(dataset_id)
        root.mkdir()
        _write(root / "data.vortex", dataset_id * 10)
        datasets[dataset_id] = Dataset.from_path(
            str(dataset_id), str(root), fmt="vortex"
        )
    calls: list[dict[str, Any]] = []
    paths: list[Path] = []
    native_open = vortex.open

    def record_open(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        paths.append(Path(args[0]))
        return native_open(*args, **kwargs)

    monkeypatch.setattr(vortex, "open", record_open)
    store = build_multi_dataset_store(
        datasets,
        options=StoreOptions(
            cache=CacheOptions(
                enabled=True,
                root=tmp_path / "cache",
                metadata_cache_entries=2,
            ),
            vortex=VortexOptions(segment_cache_bytes=4096),
        ),
    )
    try:
        for dataset_id in (0, 1):
            shard, _ = store.for_dataset(dataset_id).open(0)
            for indices in ([2, 0, 2], [1]):
                rows, _ = shard.getsamples(indices)
                assert rows == [dataset_id * 10 + index for index in indices]
                if len(indices) > 1:
                    # Re-downloading identical bytes keeps the immutable shard's key.
                    path = paths[-1]
                    replacement = path.with_suffix(".replacement")
                    replacement.write_bytes(path.read_bytes())
                    replacement.replace(path)
        assert len(calls) == 4  # Separate readers share caches.
        assert ["footer" in call for call in calls] == [False, True, False, True]
        cache = calls[0]["segment_cache"]
        assert all(call["segment_cache"] is cache for call in calls)
        assert calls[0]["cache_key"] == calls[1]["cache_key"]
        assert calls[0]["cache_key"] != calls[2]["cache_key"]
        assert cache.entry_count > 0
    finally:
        store.close()
    assert calls[0]["segment_cache"].size_bytes == 0


def test_native_open_eviction_is_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native IO error translation reaches ResilientShard's eviction recovery."""
    path = tmp_path / "rows.vortex"
    rows = [{"value": 0}, {"value": 1}]
    vortex.io.write(vortex.array(rows), str(path))
    data = path.read_bytes()
    locator = ShardLocator(
        dataset="test",
        shard_id=0,
        format="vortex",
        root=str(tmp_path),
        raw=ShardFile(basename=path.name, bytes=len(data), hashes={}),
    )
    ref = LocalShardRef(raw=LocalShardFile(path=path, bytes=len(data)))

    class Resolver(ShardResolver):
        def __init__(self) -> None:
            self.calls = 0

        def resolve(
            self, locator: ShardLocator, *, blocking: bool = True
        ) -> LocalShardRef:
            self.calls += 1
            if self.calls > 1:
                path.write_bytes(data)
            return ref

        def touch(self, locator: ShardLocator) -> None:
            pass

    native_open = vortex.open
    opens = 0

    def open_after_eviction(*args: Any, **kwargs: Any) -> Any:
        nonlocal opens
        opens += 1
        if opens == 1:
            # Simulate the cache changing after resolution but before native open.
            path.unlink()
        return native_open(*args, **kwargs)

    monkeypatch.setattr(vortex, "open", open_after_eviction)
    resolver = Resolver()
    shard = ResilientShard(
        locator=locator,
        resolver=resolver,
        opener=VortexFormat(),
        length=len(rows),
        retry_attempts=2,
        retry_initial_backoff=0,
        retry_max_backoff=0,
    )
    try:
        indices = [1, 0, 1]
        actual, stats = shard.getsamples(indices)
        assert actual == [rows[index]["value"] for index in indices]
        assert all(item.retries == 1 for item in stats)
        assert resolver.calls == opens == 2
    finally:
        shard.close()


def test_vortex_pipeline_preserves_fields_and_nested_metadata(tmp_path: Path) -> None:
    """Discovery, fetch, and batching preserve complete samples across shards."""
    expected = {}
    for shard_id in range(2):
        rows = [
            {
                "text": f"sample {shard_id}/{row_id}: café",
                "meta": {
                    "language": "en",
                    "source": {"shard": shard_id, "license": "CC-BY"},
                },
            }
            for row_id in range(3)
        ]
        vortex.io.write(vortex.array(rows), str(tmp_path / f"{shard_id}.vortex"))
        expected.update({(0, shard_id, row_id): row for row_id, row in enumerate(rows)})

    dataset = Dataset.from_path("samples", str(tmp_path), fmt="vortex")
    work = StaticMixtureWorkSource(
        [dataset],
        {"samples": 1.0},
        chunk_size=2,
        seed=7,
        shuffle_shards=True,
        shuffle_within_shard=True,
    )
    pipeline = (
        Pipeline(work)
        .decode_text()
        .batch(microbatch_size=2, drop_last=False)
        .options(
            runner="threads",
            max_workers=2,
            deterministic=True,
            io_options={"cache": {"enabled": False}},
        )
    )
    iterator = iter(pipeline)
    try:
        records = [record for batch in iterator for record in batch.records]
    finally:
        iterator.close()

    assert len(records) == len(expected)
    assert {record.meta.sample_id for record in records} == set(expected)
    for record in records:
        assert record.payload == expected[record.meta.sample_id]


def test_vortex_discovery_index_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index preserves discovered counts and removes the need to read footers."""
    for name, count in (("b.vortex", 3), ("a.vortex", 2)):
        vortex.io.write(
            vortex.array([{"value": i} for i in range(count)]), str(tmp_path / name)
        )

    discovered = Dataset.from_path("discovered", str(tmp_path))
    assert discovered.backend["kind"] == "vortex"
    assert discovered.shard_count() == 2
    assert discovered.total() == 5
    assert len(discovered) == 5
    target = tmp_path / "custom-index.json"
    assert build_index("vortex", tmp_path, output_path=target, progress=False) == target
    index = json.loads(target.read_text())
    assert [shard["basename"] for shard in index["shards"]] == ["a.vortex", "b.vortex"]
    assert [shard["extra"] for shard in index["shards"]] == [
        {"length": 2},
        {"length": 3},
    ]
    target.rename(tmp_path / "index.json")

    monkeypatch.setattr(vortex_format, "_vortex", None)
    indexed = Dataset.from_path("indexed", str(tmp_path), fmt="vortex")
    assert indexed.shard_count() == discovered.shard_count()
    assert indexed.total() == discovered.total()
