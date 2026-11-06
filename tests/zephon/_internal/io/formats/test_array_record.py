# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""ArrayRecord native reader adaptation and format-specific wiring."""

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


@pytest.mark.parametrize("size", [None, 123])
def test_row_count_delegates_backend_path_and_size(size: int | None) -> None:
    storage = mock.Mock()
    storage.stat.return_value = {"size": 123}
    path = "custom://bucket/src.array_record"
    with mock.patch.object(
        array_record_format, "read_row_count", return_value=7
    ) as read:
        assert (
            array_record_format._read_array_record_row_count(path, storage, size) == 7
        )
    read.assert_called_once_with(path, storage, 123)
    if size is None:
        storage.stat.assert_called_once_with(path)
    else:
        storage.stat.assert_not_called()
    storage.download.assert_not_called()


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
        assert shard.getsamples([index]) == [{"data": b"record"}]
        reader.read.assert_called_once_with(*expected_args)
    finally:
        shard.close()
