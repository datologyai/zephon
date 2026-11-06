# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""ArrayRecord footer parsing, range bounds, and corruption checks."""

from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from zephon._internal.io.formats.array_record_support.metadata import (
    _fields,
    _varint,
    read_row_count,
)
from zephon._internal.io.storage.local import LocalFSBackend


@pytest.mark.parametrize("codec", ["uncompressed", "zstd:3", "brotli:5", "snappy"])
def test_footer_counts_match_native_files(
    tmp_path: Path, write_array_record: Any, codec: str
) -> None:
    count = 257
    path = tmp_path / "s.array_record"
    write_array_record(path, [b"record"] * count, f"group_size:64,{codec}")
    assert (
        read_row_count(path.name, LocalFSBackend(tmp_path), path.stat().st_size)
        == count
    )


@pytest.mark.parametrize(("group", "count"), [(1, 0), (1, 1), (1, 257), (64, 257)])
def test_empty_and_grouped_files(
    tmp_path: Path, write_array_record: Any, group: int, count: int
) -> None:
    path = tmp_path / "s.array_record"
    write_array_record(path, [b"record"] * count, f"group_size:{group}")
    assert (
        read_row_count(path.name, LocalFSBackend(tmp_path), path.stat().st_size)
        == count
    )


def test_footer_crosses_block_boundaries(
    tmp_path: Path, write_array_record: Any
) -> None:
    path = tmp_path / "s.array_record"
    write_array_record(path, [b"x"] * 20000, "group_size:1,uncompressed")
    assert (
        read_row_count(path.name, LocalFSBackend(tmp_path), path.stat().st_size)
        == 20000
    )


@pytest.mark.parametrize(
    "damage", ["block", "header", "postscript", "footer", "truncate"]
)
def test_rejects_corrupt_metadata(
    tmp_path: Path, write_array_record: Any, damage: str
) -> None:
    path = tmp_path / "s.array_record"
    write_array_record(path, [b"x"] * 257, "group_size:1,uncompressed")
    contents = bytearray(path.read_bytes())
    postscript = len(contents) - 65536
    storage = mock.Mock(wraps=LocalFSBackend(tmp_path))
    read_row_count(path.name, storage, len(contents))
    footer_offset = storage.read_range.call_args_list[1].args[1]
    if damage == "truncate":
        contents.pop()
    else:
        # The uncompressed footer fits in the block immediately before the postscript.
        offset = {
            "block": postscript,
            "header": postscript + 32,
            "postscript": postscript + 75,
            "footer": footer_offset + 80,
        }[damage]
        contents[offset] ^= 1
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        read_row_count(path.name, LocalFSBackend(tmp_path), len(contents))


def test_incomplete_range_is_not_an_empty_shard(
    tmp_path: Path, write_array_record: Any
) -> None:
    path = tmp_path / "s.array_record"
    write_array_record(path, [b"x"])
    storage = mock.Mock(wraps=LocalFSBackend(tmp_path))
    storage.read_range.return_value = b""
    with pytest.raises(ValueError, match="Incomplete"):
        read_row_count(path.name, storage, path.stat().st_size)


def test_varint_uint64_boundary() -> None:
    assert _varint(b"\xff" * 9 + b"\x01", 0) == ((1 << 64) - 1, 10)
    with pytest.raises(ValueError, match="Invalid"):
        _varint(b"\xff" * 9 + b"\x02", 0)
    with pytest.raises(ValueError, match="Truncated"):
        _varint(b"\x80", 0)


@pytest.mark.parametrize("data", [b"\x00", b"\x0b", b"\x0a\x02x"])
def test_rejects_invalid_protobuf_fields(data: bytes) -> None:
    with pytest.raises(ValueError):
        list(_fields(data))
