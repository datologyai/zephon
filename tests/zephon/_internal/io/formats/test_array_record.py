# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""ArrayRecord discovery, indexing, and shard-read unit tests."""

import json
import sys
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import pytest

from zephon._internal.io.formats import array_record as array_record_format
from zephon._internal.io.formats.array_record import (
    ArrayRecordFormat,
    ArrayRecordShard,
)
from zephon._internal.io.storage.local import LocalFSBackend
from zephon._internal.io.types import LocalShardFile, LocalShardRef
from zephon.io.dataset import Dataset


@pytest.fixture
def write_array_record() -> Callable[[Path, list[bytes]], None]:
    module = pytest.importorskip(
        "array_record.python.array_record_module",
        reason="array-record not installed",
    )

    def write(path: Path, records: list[bytes], options: str = "group_size:1") -> None:
        writer = module.ArrayRecordWriter(str(path), options)
        for record in records:
            writer.write(record)
        writer.close()

    return write


def _records(n: int, prefix: str = "row") -> list[bytes]:
    return [f"{prefix}-{i}".encode() for i in range(n)]


@pytest.mark.parametrize(
    "indices",
    [
        pytest.param([], id="empty"),
        pytest.param([3], id="single"),
        pytest.param([3, 3, 3], id="all-identical"),
        pytest.param([0, 1, 4, 5], id="sorted"),
        pytest.param([4, 1, 4, 0, 1, 5, 0], id="unsorted-duplicates"),
    ],
)
def test_getsamples_preserves_order_and_duplicates(
    tmp_path: Path, indices: list[int], write_array_record: Any
) -> None:
    path = tmp_path / "shard.array_record"
    records = _records(6)
    write_array_record(path, records)

    shard = ArrayRecordShard(path)
    original = indices.copy()
    try:
        assert len(shard) == 6
        assert shard.getsamples(indices) == [{"data": records[i]} for i in original]
        assert indices == original
    finally:
        shard.close()


@pytest.mark.parametrize("options", ["group_size:1", "group_size:64"])
def test_reads_grouped_records(
    tmp_path: Path, options: str, write_array_record: Any
) -> None:
    """Records packed many-per-chunk read back the same as one-per-chunk."""
    path = tmp_path / "grouped.array_record"
    records = _records(500)
    write_array_record(path, records, options)

    shard = ArrayRecordShard(path)
    try:
        assert shard[0] == {"data": records[0]}
        assert shard.getsamples([499, 7, 250]) == [
            {"data": records[i]} for i in (499, 7, 250)
        ]
    finally:
        shard.close()


def test_rejects_out_of_range_and_closed_reads(
    tmp_path: Path, write_array_record: Any
) -> None:
    path = tmp_path / "shard.array_record"
    write_array_record(path, _records(3))
    shard = ArrayRecordShard(path)
    with pytest.raises(IndexError):
        shard.getsamples([3])
    with pytest.raises(IndexError):
        shard.getsamples([-1])
    shard.close()
    shard.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        shard.getsamples([0])


@pytest.mark.usefixtures("write_array_record")
def test_missing_shard_raises_os_error(tmp_path: Path) -> None:
    """The native reader opens a missing file as empty; the shard must not."""
    with pytest.raises(FileNotFoundError):
        ArrayRecordShard(tmp_path / "missing.array_record")


@pytest.mark.usefixtures("write_array_record")
def test_corrupt_shard_raises(tmp_path: Path) -> None:
    path = tmp_path / "junk.array_record"
    path.write_bytes(b"not an array record file" * 10)
    with pytest.raises(ValueError, match="Not a readable ArrayRecord file"):
        ArrayRecordShard(path)


def test_discover_by_scanning(tmp_path: Path, write_array_record: Any) -> None:
    write_array_record(tmp_path / "b.array_record", _records(3))
    write_array_record(tmp_path / "a.array_record", _records(5))
    (tmp_path / "notes.txt").write_text("ignored")

    handler = ArrayRecordFormat()
    storage = LocalFSBackend(tmp_path)
    shard_index, shard_meta = handler.discover(str(tmp_path), storage)
    ids, counts = handler.discover_counts(str(tmp_path), storage)

    # Shards are numbered in filename order.
    assert dict(shard_index) == {0: 5, 1: 3}
    assert shard_meta[0]["raw"]["basename"] == "a.array_record"
    assert shard_meta[0]["extra"] == {"length": 5}
    assert ids.tolist() == [0, 1]
    assert counts.tolist() == [5, 3]


@pytest.mark.usefixtures("write_array_record")
def test_discover_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No .array_record shards"):
        ArrayRecordFormat().discover(str(tmp_path), LocalFSBackend(tmp_path))


def test_index_build_and_discovery_use_index(
    tmp_path: Path, write_array_record: Any
) -> None:
    from zephon.build_index import build_index

    write_array_record(tmp_path / "s0.array_record", _records(4))
    write_array_record(tmp_path / "s1.array_record", _records(2))
    build_index("array_record", tmp_path, progress=False)

    index = json.loads((tmp_path / "index.json").read_text())
    assert [(s["basename"], s["num_rows"]) for s in index["shards"]] == [
        ("s0.array_record", 4),
        ("s1.array_record", 2),
    ]

    # With an index present, discovery must not open any shard.
    handler = ArrayRecordFormat()
    storage = LocalFSBackend(tmp_path)
    with mock.patch.object(
        array_record_format, "_open_reader", side_effect=AssertionError("opened")
    ):
        shard_index, _ = handler.discover(str(tmp_path), storage)
        _, counts = handler.discover_counts(str(tmp_path), storage)
    assert dict(shard_index) == {0: 4, 1: 2}
    assert counts.tolist() == [4, 2]


def test_remote_row_count_downloads_shard(
    tmp_path: Path, write_array_record: Any
) -> None:
    """A remote shard is fetched through the backend before it is opened."""
    source = tmp_path / "src.array_record"
    write_array_record(source, _records(7))
    storage = mock.Mock()
    storage.download.side_effect = lambda src, dst: Path(dst).write_bytes(
        source.read_bytes()
    )

    count = array_record_format._read_array_record_row_count(
        "s3://bucket/data/src.array_record", storage
    )

    assert count == 7
    storage.download.assert_called_once()
    assert storage.download.call_args.args[0] == "s3://bucket/data/src.array_record"


@pytest.mark.parametrize("with_index", [False, True], ids=["scan", "index"])
def test_from_path_auto_detects_and_reads(
    tmp_path: Path, with_index: bool, write_array_record: Any
) -> None:
    from zephon.build_index import build_index
    from zephon.debug import DatasetInspector

    records = _records(3, prefix="sample")
    write_array_record(tmp_path / "shard0.array_record", records)
    if with_index:
        build_index("array_record", tmp_path, progress=False)

    dataset = Dataset.from_path("ar", str(tmp_path))

    assert dataset.backend["kind"] == "array_record"
    assert dataset.counts().tolist() == [3]
    with DatasetInspector(dataset) as inspector:
        assert inspector.read(shard_id=0, sample_index=2) == {"data": records[2]}


@pytest.mark.parametrize(
    "bad_shards",
    [
        {0: {}},
        {0: {"raw": {"bytes": 1}}},
        {0: {"raw": {"basename": "s.array_record"}}},
        {0: {"raw": {"basename": "s.array_record", "bytes": object()}}},
    ],
)
def test_build_locators_rejects_bad_metadata(tmp_path: Path, bad_shards: Any) -> None:
    ds = Dataset(
        name="bad",
        backend={"kind": "array_record", "path": str(tmp_path), "shards": bad_shards},
        path=str(tmp_path),
    )
    with pytest.raises(ValueError):
        ArrayRecordFormat().build_locators(ds)


def test_open_shard_uses_length_hint(tmp_path: Path, write_array_record: Any) -> None:
    path = tmp_path / "s.array_record"
    write_array_record(path, _records(4))
    handler = ArrayRecordFormat()
    _, shard_meta = handler.discover(str(tmp_path), LocalFSBackend(tmp_path))
    ds = Dataset(
        name="demo",
        backend={"kind": "array_record", "path": str(tmp_path), "shards": shard_meta},
        path=str(tmp_path),
    )
    locator = handler.build_locators(ds)[0]
    ref = LocalShardRef(
        raw=LocalShardFile(path=path, bytes=path.stat().st_size),
        extra={"length": "4"},
    )
    shard = handler.open_shard(locator, ref)
    try:
        assert len(shard) == 4
        assert shard[3] == {"data": b"row-3"}
    finally:
        shard.close()


def test_missing_dependency_explains_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "array_record", None)
    monkeypatch.setitem(sys.modules, "array_record.python", None)
    monkeypatch.setitem(sys.modules, "array_record.python.array_record_module", None)
    with pytest.raises(ImportError, match=r"zephon\[array_record\]"):
        ArrayRecordShard(tmp_path / "x.array_record")
