# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Index creation and discovery across format readers and storage routing."""

import json
import random
from pathlib import Path

import pytest

from tests.helpers.storage import MemoryStorage
from zephon._internal.io.storage import router
from zephon.build_index import build_index
from zephon.io import Dataset

pytestmark = pytest.mark.integration


def _dataset_files(fmt: str, tmp_path: Path) -> dict[str, bytes]:
    """Real files for each registered index format, with known record counts."""
    files: dict[str, bytes] = {}
    rng = random.Random(0)
    for name, count in (("a", 2), ("b", 3)):
        path = tmp_path / f"{name}.{fmt}"
        if fmt == "jsonl":
            path.write_text(
                "\n".join(json.dumps({"value": i}) for i in range(count)),
                encoding="utf-8",
            )
        else:
            rows = [{"payload": rng.randbytes(40 * 1024)} for _ in range(count)]
            if fmt == "parquet":
                pa = pytest.importorskip("pyarrow")
                pq = pytest.importorskip("pyarrow.parquet")
                pq.write_table(pa.Table.from_pylist(rows), path)
            else:
                vx = pytest.importorskip("vortex")
                vx.io.write(vx.array(rows), str(path))
        files[path.name] = path.read_bytes()
    return files


@pytest.mark.parametrize("fmt", ["jsonl", "parquet", "vortex"])
def test_remote_index_roundtrip_all_formats(
    fmt: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public API writes remote indexes that discovery can use without shard IO."""
    files = _dataset_files(fmt, tmp_path)
    root = "s3://bucket/data"
    storage = MemoryStorage(root, {**files, f"nested/ignored.{fmt}": b"invalid"})
    monkeypatch.setattr(router, "_make_s3_backend", lambda: storage)
    target = build_index(fmt, root, progress=False, max_workers=2)
    assert target == f"{root}/index.json"
    index = json.loads(storage.files[target])
    assert [shard["basename"] for shard in index["shards"]] == sorted(files)
    assert [shard["num_rows"] for shard in index["shards"]] == [2, 3]
    if fmt != "jsonl":
        assert 0 < storage.read_bytes < sum(map(len, files.values()))
        assert storage.opened == []

    read_bytes = storage.read_bytes
    opened = len(storage.opened)
    dataset = Dataset.from_path("indexed", root, fmt=fmt)
    assert dataset.shard_count() == 2
    assert dataset.total() == 5
    assert storage.read_bytes == read_bytes
    assert storage.opened[opened:] == [target]


@pytest.mark.parametrize("remote_source", [True, False])
def test_index_output_can_use_a_different_backend(
    remote_source: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both remote-to-local and local-to-remote index output go through the router."""
    fmt = "jsonl"
    files = _dataset_files(fmt, tmp_path)
    storage = MemoryStorage("gs://bucket/data", files)
    monkeypatch.setattr(router, "_make_gcs_backend", lambda: storage)
    source = storage.root if remote_source else tmp_path
    target = tmp_path / "output.json" if remote_source else "gs://bucket/index.json"
    assert build_index(fmt, source, output_path=target, progress=False) == target
    data = target.read_bytes() if isinstance(target, Path) else storage.files[target]
    assert [shard["num_rows"] for shard in json.loads(data)["shards"]] == [2, 3]
