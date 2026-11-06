# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""ArrayRecord native reader adaptation and format-specific wiring."""

import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from zephon._internal.io.formats import array_record as array_record_format
from zephon._internal.io.formats.array_record import (
    ArrayRecordFormat,
    ArrayRecordShard,
)
from zephon._internal.io.storage.local import LocalFSBackend
from zephon._internal.io.types import LocalShardFile, LocalShardRef


def _records(n: int, prefix: str = "row") -> list[bytes]:
    return [f"{prefix}-{i}".encode() for i in range(n)]


@pytest.mark.parametrize(
    "indices",
    [
        pytest.param([], id="empty"),
        pytest.param([3], id="single"),
        pytest.param([3, 3, 3], id="all-identical"),
        pytest.param([0, 1, 4, 5], id="sorted"),
        pytest.param([1, 2, 3, 4], id="contiguous-range"),
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
        assert shard.getsamples(indices) == [records[i] for i in original]
        assert indices == original
    finally:
        shard.close()


@pytest.mark.parametrize("options", ["group_size:1", "group_size:2", "group_size:64"])
def test_reads_grouped_records(
    tmp_path: Path, options: str, write_array_record: Any
) -> None:
    """Records packed many-per-chunk read back the same as one-per-chunk."""
    path = tmp_path / "grouped.array_record"
    records = _records(500)
    write_array_record(path, records, options)

    shard = ArrayRecordShard(path)
    try:
        assert shard[0] == records[0]
        assert shard.getsamples([499, 7, 250]) == [records[i] for i in (499, 7, 250)]
        assert shard.getsamples([62, 63, 64, 65]) == records[62:66]
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
    """Missing files remain filesystem errors across native reader versions."""
    with pytest.raises(FileNotFoundError):
        ArrayRecordShard(tmp_path / "missing.array_record")


@pytest.mark.parametrize("exists", [False, True])
def test_native_constructor_failure(tmp_path: Path, exists: bool) -> None:
    """Normalize old native exceptions without hiding missing-file errors."""
    path = tmp_path / "shard.array_record"
    if exists:
        path.touch()
    with mock.patch.object(array_record_format, "_require_array_record") as module:
        module.return_value.ArrayRecordReader.side_effect = RuntimeError(
            "native failure"
        )
        if exists:
            with pytest.raises(ValueError, match="native failure"):
                ArrayRecordShard(path)
        else:
            with pytest.raises(FileNotFoundError):
                ArrayRecordShard(path)


@pytest.mark.usefixtures("write_array_record")
def test_corrupt_shard_raises(tmp_path: Path) -> None:
    path = tmp_path / "junk.array_record"
    path.write_bytes(b"not an array record file" * 10)
    with pytest.raises(ValueError, match="Not a readable ArrayRecord file"):
        ArrayRecordShard(path)


def test_row_count_delegates_backend_path_and_size() -> None:
    storage = mock.Mock()
    storage.local_path = mock.Mock(return_value=None)
    path = "custom://bucket/src.array_record"
    with mock.patch.object(
        array_record_format, "read_row_count", return_value=7
    ) as read:
        assert array_record_format._read_array_record_row_count(path, storage, 123) == 7
    read.assert_called_once_with(path, storage, 123)
    storage.stat.assert_not_called()
    storage.download.assert_not_called()


def test_native_count_uses_backend_local_path_and_closes_reader(tmp_path: Path) -> None:
    storage = mock.Mock()
    local = tmp_path / "part.array_record"
    storage.local_path = mock.Mock(return_value=local)
    reader = mock.Mock()
    reader.num_records.return_value = 7
    with mock.patch.object(
        array_record_format, "_open_reader", return_value=reader
    ) as open_reader:
        assert (
            array_record_format._read_array_record_row_count(
                "custom://dataset/part", storage, 123
            )
            == 7
        )
    open_reader.assert_called_once_with(str(local))
    reader.close.assert_called_once_with()
    storage.read_range.assert_not_called()
    storage.stat.assert_not_called()


def test_native_count_closes_reader_when_counting_fails(tmp_path: Path) -> None:
    storage = LocalFSBackend(tmp_path)
    reader = mock.Mock()
    reader.num_records.side_effect = RuntimeError("bad metadata")
    with mock.patch.object(array_record_format, "_open_reader", return_value=reader):
        with pytest.raises(RuntimeError, match="bad metadata"):
            array_record_format._read_array_record_row_count("part", storage, 123)
    reader.close.assert_called_once_with()


def test_open_shard_uses_length_hint(tmp_path: Path, write_array_record: Any) -> None:
    path = tmp_path / "s.array_record"
    write_array_record(path, _records(4))
    handler = ArrayRecordFormat()
    locator = mock.Mock()
    ref = LocalShardRef(
        raw=LocalShardFile(path=path, bytes=path.stat().st_size),
        extra={"length": "4"},
    )
    shard = handler.open_shard(locator, ref)
    try:
        assert len(shard) == 4
        assert shard[3] == b"row-3"
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


@pytest.mark.parametrize(
    ("index", "expected_args"),
    [
        (3, (3, 4)),
        ((1 << 31) - 2, ((1 << 31) - 2, (1 << 31) - 1)),
        ((1 << 31) - 1, ([(1 << 31) - 1],)),
        (1 << 32, ([1 << 32],)),
    ],
)
def test_singleton_native_range_boundary(
    tmp_path: Path, index: int, expected_args: tuple[Any, ...]
) -> None:
    reader = mock.Mock()
    reader.read.return_value = [b"record"]
    with mock.patch.object(array_record_format, "_open_reader", return_value=reader):
        shard = ArrayRecordShard(tmp_path / "s.array_record", length=(1 << 32) + 1)
    try:
        assert shard.getsamples([index]) == [b"record"]
        reader.read.assert_called_once_with(*expected_args)
    finally:
        shard.close()


@pytest.mark.parametrize(
    ("indices", "calls"),
    [
        ([5, 6, 2, 2, 3], [([5, 6, 2, 2, 3],)]),
        ([2, 3, 4], [(2, 5)]),
        ([2, 4, 4], [([2, 4, 4],)]),
        ([(1 << 31) - 3, (1 << 31) - 2], [((1 << 31) - 3, (1 << 31) - 1)]),
        (
            [(1 << 31) - 2, (1 << 31) - 1],
            [([(1 << 31) - 2, (1 << 31) - 1],)],
        ),
        ([1 << 31, 0], [([1 << 31, 0],)]),
    ],
)
def test_bulk_native_dispatch_preserves_order_and_duplicates(
    tmp_path: Path, indices: list[int], calls: list[tuple[Any, ...]]
) -> None:
    reader = mock.Mock()

    def read(*args: Any) -> list[bytes]:
        requested = args[0] if len(args) == 1 else range(*args)
        return [str(index).encode() for index in requested]

    reader.read.side_effect = read
    with mock.patch.object(array_record_format, "_open_reader", return_value=reader):
        shard = ArrayRecordShard(tmp_path / "part", length=(1 << 32) + 1)
    try:
        assert shard.getsamples(indices) == [str(index).encode() for index in indices]
        assert reader.read.call_args_list == [mock.call(*args) for args in calls]
    finally:
        shard.close()


@pytest.mark.parametrize(
    "update",
    [
        {"basename": "wrong.jsonl"},
        {"extra": {"length": 2}},
        {"extra": {"length": True}},
    ],
)
def test_index_rejects_format_specific_metadata(
    tmp_path: Path, update: dict[str, Any]
) -> None:
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "shards": [
                    {"basename": "a.array_record", "bytes": 10, "num_rows": 3, **update}
                ],
            }
        )
    )
    handler = ArrayRecordFormat()
    for discover in (handler.discover, handler.discover_counts):
        with pytest.raises(ValueError, match="index"):
            discover(".", LocalFSBackend(tmp_path))


def test_index_preserves_extra_without_reading_shards(tmp_path: Path) -> None:
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "shards": [
                    {
                        "basename": "a.array_record",
                        "bytes": 10,
                        "num_rows": 3,
                        "extra": {"source": "test"},
                    }
                ],
            }
        )
    )
    storage = mock.Mock(wraps=LocalFSBackend(tmp_path))
    handler = ArrayRecordFormat()
    counts, metadata = handler.discover(".", storage)
    assert counts == {0: 3}
    assert handler.discover_counts(".", storage)[1].tolist() == [3]
    assert metadata[0]["extra"] == {"source": "test", "length": 3}
    storage.read_range.assert_not_called()
    storage.listdir.assert_not_called()


def test_scan_sorts_shards_and_passes_backend_paths_and_sizes(tmp_path: Path) -> None:
    (tmp_path / "b.array_record").write_bytes(b"bb")
    (tmp_path / "a.array_record").write_bytes(b"a")
    (tmp_path / "notes.txt").write_text("ignored")
    storage = LocalFSBackend(tmp_path)
    with mock.patch.object(
        array_record_format,
        "_read_array_record_row_count",
        side_effect=lambda path, backend, size: size + 2,
    ) as read:
        counts, metadata = ArrayRecordFormat().discover(".", storage)
    assert counts == {0: 3, 1: 4}
    assert metadata[0]["raw"]["basename"] == "a.array_record"
    assert metadata[0]["extra"] == {"length": 3}
    assert sorted(call.args[0] for call in read.call_args_list) == [
        "./a.array_record",
        "./b.array_record",
    ]
    assert all(call.args[1] is storage for call in read.call_args_list)
    assert sorted(call.args[2] for call in read.call_args_list) == [1, 2]


def test_scan_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No .array_record shards"):
        ArrayRecordFormat().discover(".", LocalFSBackend(tmp_path))


def test_scan_preserves_metadata_error_context(tmp_path: Path) -> None:
    (tmp_path / "a.array_record").touch()
    with mock.patch.object(
        array_record_format,
        "_read_array_record_row_count",
        side_effect=OSError("missing"),
    ):
        with pytest.raises(ValueError, match="a.array_record: missing") as error:
            ArrayRecordFormat().discover(".", LocalFSBackend(tmp_path))
    assert isinstance(error.value.__cause__, OSError)
