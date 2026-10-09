# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex cache ownership through real store fetches and file replacements."""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("vortex.io", reason="vortex-data not installed")
import vortex

from zephon._internal.io.formats.vortex import VortexShardOpener
from zephon._internal.io.stores.multi import build_multi_dataset_store
from zephon._internal.io.types import (
    LocalShardFile,
    LocalShardRef,
    ShardFile,
    ShardLocator,
)
from zephon.io import Dataset, StoreOptions, VortexOptions

pytestmark = pytest.mark.integration


def _write(path: Path, start: int = 0) -> None:
    vortex.io.write(
        vortex.array([{"value": i} for i in range(start, start + 8)]), str(path)
    )


def _refs(path: Path) -> tuple[ShardLocator, LocalShardRef]:
    size = path.stat().st_size
    return (
        ShardLocator(
            dataset="test",
            shard_id=0,
            format="vortex",
            root=str(path.parent),
            raw=ShardFile(basename=path.name, bytes=size, hashes={}),
        ),
        LocalShardRef(raw=LocalShardFile(path=path, bytes=size)),
    )


@pytest.mark.parametrize("segment_bytes", [0, 4096])
@pytest.mark.parametrize("metadata_entries", [0, 2])
def test_store_reuses_caches_across_fetch_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    segment_bytes: int,
    metadata_entries: int,
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
    native_open = vortex.open

    def record_open(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return native_open(*args, **kwargs)

    monkeypatch.setattr(vortex, "open", record_open)
    store = build_multi_dataset_store(
        datasets,
        options=StoreOptions(
            vortex=VortexOptions(
                segment_cache_bytes=segment_bytes,
                metadata_cache_entries=metadata_entries,
            )
        ),
    )
    try:
        for dataset_id in (0, 1):
            shard, _ = store.for_dataset(dataset_id).open(0)
            for indices in ([2, 0, 2], [1], [3, 1]):
                rows, _ = shard.getsamples(indices)
                assert [row["value"] for row in rows] == [
                    dataset_id * 10 + index for index in indices
                ]
        assert len(calls) == 6  # Readers are still temporary; their caches persist.
        assert ["footer" in call for call in calls] == (
            [False, True, True, False, True, True] if metadata_entries else [False] * 6
        )
        if segment_bytes:
            cache = calls[0]["segment_cache"]
            assert all(call["segment_cache"] is cache for call in calls)
            assert calls[0]["cache_key"] == calls[1]["cache_key"]
            assert calls[0]["cache_key"] != calls[3]["cache_key"]
            assert cache.size_bytes <= segment_bytes
            if metadata_entries:
                assert cache.entry_count > 0
        else:
            assert all(call["without_segment_cache"] for call in calls)
            assert all("segment_cache" not in call for call in calls)
    finally:
        store.close()
    if segment_bytes:
        assert calls[0]["segment_cache"].size_bytes == 0


def test_footer_eviction_and_replaced_file_invalidate_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "first.vortex", tmp_path / "second.vortex"
    _write(first)
    _write(second, 10)
    calls: list[dict[str, Any]] = []
    native_open = vortex.open

    def record_open(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return native_open(*args, **kwargs)

    monkeypatch.setattr(vortex, "open", record_open)
    opener = VortexShardOpener(
        VortexOptions(metadata_cache_entries=1, segment_cache_bytes=1)
    )

    def read(path: Path) -> int:
        reader = opener.open_shard(*_refs(path))
        try:
            return reader[0]["value"]
        finally:
            reader.close()

    try:
        assert read(first) == read(first) == 0
        assert read(second) == 10
        assert read(first) == 0
        assert ["footer" in call for call in calls] == [False, True, False, False]
        old_key = calls[-1]["cache_key"]
        replacement = tmp_path / "replacement.vortex"
        _write(replacement, 20)
        replacement.replace(first)
        assert read(first) == 20
        assert calls[-1]["cache_key"] != old_key
        assert "footer" not in calls[-1]
        assert read(first) == 20
        # Segments larger than the configured byte budget are not retained.
        assert calls[-1]["segment_cache"].size_bytes <= 1
    finally:
        opener.close()
