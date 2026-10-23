# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shard readers stay open between reads of a real store, for each format."""

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from zephon._internal.io.stores.multi import build_multi_dataset_store
from zephon._internal.io.stores.open_shards import OpenShardCache
from zephon.io import CacheOptions, Dataset, StoreOptions

pytestmark = pytest.mark.integration


def _write_jsonl(root: Path) -> None:
    rows = (json.dumps({"value": i}) for i in range(8))
    (root / "data.jsonl").write_text("\n".join(rows), encoding="utf-8")


def _write_parquet(root: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    pq.write_table(pa.table({"value": list(range(8))}), root / "data.parquet")


def _write_vortex(root: Path) -> None:
    vortex = pytest.importorskip("vortex")
    pytest.importorskip("vortex.io")
    vortex.io.write(
        vortex.array([{"value": i} for i in range(8)]), str(root / "data.vortex")
    )


@pytest.mark.parametrize(
    ("fmt", "write"),
    [("jsonl", _write_jsonl), ("parquet", _write_parquet), ("vortex", _write_vortex)],
)
def test_store_keeps_readers_open_until_their_file_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fmt: str,
    write: Callable[[Path], None],
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    write(root)
    store = build_multi_dataset_store(
        {0: Dataset.from_path("data", str(root), fmt=fmt)},
        options=StoreOptions(
            cache=CacheOptions(enabled=True, root=tmp_path / "cache"),
            max_open_shards=4,
        ),
    )
    try:
        view = store.for_dataset(0)
        cache = getattr(view, "_opener")
        assert isinstance(cache, OpenShardCache)

        # Count the opens of the format's own opener.
        format_opener = getattr(cache, "_opener")
        paths: list[Path] = []
        native_open = format_opener.open_shard

        def record_open(locator: Any, local_ref: Any) -> Any:
            paths.append(local_ref.raw.path)
            return native_open(locator, local_ref)

        monkeypatch.setattr(format_opener, "open_shard", record_open)

        shard, _ = view.open(0)
        for _ in range(3):
            rows, _ = shard.getsamples([2, 0, 2])
            assert [row["value"] for row in rows] == [2, 0, 2]
        assert len(paths) == 1

        # A new download of the shard is a new file, so it gets a new reader.
        replacement = paths[0].with_suffix(".replacement")
        replacement.write_bytes(paths[0].read_bytes())
        replacement.replace(paths[0])
        rows, _ = shard.getsamples([1, 3])
        assert [row["value"] for row in rows] == [1, 3]
        assert len(paths) == 2
    finally:
        store.close()
