# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the single-file catalog artifact: pack/load, validation, pointers."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from zephon.io.catalog import SCHEMA_VERSION
from zephon.io.catalog import io as catalog_io


def _raw_artifact(header: dict, data_region: bytes = b"") -> bytes:
    """Assemble a file with an arbitrary header, for ``is_valid`` edge cases."""
    header_bytes = json.dumps(header).encode("utf-8")
    prefix = len(catalog_io.MAGIC) + struct.calcsize("<Q") + len(header_bytes)
    pad = (-prefix) % 8
    return (
        catalog_io.MAGIC
        + struct.pack("<Q", len(header_bytes))
        + header_bytes
        + b"\x00" * pad
        + data_region
    )


def test_pack_load_roundtrip_preserves_header_and_columns(tmp_path: Path) -> None:
    cols = {
        "shard_id": np.arange(4, dtype=np.int64),
        "num_rows": np.array([3, 5, 7, 11], dtype=np.int64),
        "weight": np.array([0.5, 1.5], dtype=np.float64),
        "blob": np.frombuffer(b"hello-world", dtype=np.uint8),
    }
    fields = {
        "schema_version": SCHEMA_VERSION,
        "format": "jsonl",
        "fingerprint": "deadbeef",
    }
    path = tmp_path / "art"
    catalog_io.write_atomic(path, catalog_io.pack_catalog(fields, cols))

    loaded = catalog_io.load_mmap(path)
    assert loaded.header["schema_version"] == SCHEMA_VERSION
    assert loaded.header["format"] == "jsonl"
    assert loaded.header["fingerprint"] == "deadbeef"
    for name, arr in cols.items():
        np.testing.assert_array_equal(loaded.columns[name], arr)
        assert loaded.columns[name].dtype == arr.dtype
    assert loaded.columns["blob"].tobytes() == b"hello-world"
    assert loaded._mmap is not None
    assert not loaded.columns["shard_id"].flags.writeable  # zero-copy mmap view


def test_empty_column_roundtrips(tmp_path: Path) -> None:
    cols = {
        "present": np.array([1, 2, 3], dtype=np.int64),
        "absent": np.empty(0, dtype=np.int64),
    }
    path = tmp_path / "art"
    catalog_io.write_atomic(
        path, catalog_io.pack_catalog({"schema_version": SCHEMA_VERSION}, cols)
    )

    loaded = catalog_io.load_mmap(path)
    assert loaded.columns["absent"].shape == (0,)
    assert loaded.columns["absent"].dtype == np.int64
    np.testing.assert_array_equal(loaded.columns["present"], cols["present"])


def test_ragged_column_forces_alignment(tmp_path: Path) -> None:
    cols = {
        "head": np.frombuffer(b"abc", dtype=np.uint8),  # 3 bytes -> pads to 8
        "tail": np.array([10, 20, 30], dtype=np.int64),
    }
    path = tmp_path / "art"
    catalog_io.write_atomic(
        path, catalog_io.pack_catalog({"schema_version": SCHEMA_VERSION}, cols)
    )

    loaded = catalog_io.load_mmap(path)
    offsets = [col["offset"] for col in loaded.header["columns"]]
    assert all(off % 8 == 0 for off in offsets)
    np.testing.assert_array_equal(loaded.columns["tail"], cols["tail"])
    assert loaded.columns["head"].tobytes() == b"abc"


def test_non_contiguous_input_is_materialized(tmp_path: Path) -> None:
    strided = np.arange(20, dtype=np.int64)[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    path = tmp_path / "art"
    catalog_io.write_atomic(
        path,
        catalog_io.pack_catalog({"schema_version": SCHEMA_VERSION}, {"x": strided}),
    )

    loaded = catalog_io.load_mmap(path)
    np.testing.assert_array_equal(loaded.columns["x"], strided)


def test_pack_rejects_multidimensional_column() -> None:
    with pytest.raises(ValueError, match="1-D"):
        catalog_io.pack_catalog(
            {"schema_version": SCHEMA_VERSION},
            {"bad": np.zeros((2, 2), dtype=np.int64)},
        )


def test_load_mmap_rejects_bad_magic(tmp_path: Path) -> None:
    path = tmp_path / "art"
    path.write_bytes(b"NOTMAGIC" + b"\x00" * 16)
    with pytest.raises(ValueError, match="bad magic"):
        catalog_io.load_mmap(path)


def test_is_valid_accepts_roundtrip_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "art"
    catalog_io.write_atomic(
        path,
        catalog_io.pack_catalog(
            {"schema_version": SCHEMA_VERSION}, {"x": np.arange(4, dtype=np.int64)}
        ),
    )
    assert catalog_io.is_valid(path, schema_version=SCHEMA_VERSION)
    assert not catalog_io.is_valid(path, schema_version=SCHEMA_VERSION + 1)


def test_is_valid_rejects_truncated_columns(tmp_path: Path) -> None:
    data = catalog_io.pack_catalog(
        {"schema_version": SCHEMA_VERSION}, {"x": np.arange(8, dtype=np.int64)}
    )
    path = tmp_path / "art"
    path.write_bytes(data[:-4])
    assert not catalog_io.is_valid(path, schema_version=SCHEMA_VERSION)


def test_is_valid_rejects_missing_bad_magic_and_corrupt_header(tmp_path: Path) -> None:
    assert not catalog_io.is_valid(tmp_path / "absent", schema_version=SCHEMA_VERSION)

    bad_magic = tmp_path / "magic"
    bad_magic.write_bytes(b"XXXXXXXX" + struct.pack("<Q", 2) + b"{}")
    assert not catalog_io.is_valid(bad_magic, schema_version=SCHEMA_VERSION)

    corrupt = tmp_path / "corrupt"
    corrupt.write_bytes(catalog_io.MAGIC + struct.pack("<Q", 5) + b"not{j")
    assert not catalog_io.is_valid(corrupt, schema_version=SCHEMA_VERSION)


def test_is_valid_rejects_negative_or_oversized_extent(tmp_path: Path) -> None:
    negative = _raw_artifact(
        {
            "schema_version": SCHEMA_VERSION,
            "columns": [
                {"name": "x", "dtype": "<i8", "count": 1, "offset": -8, "nbytes": 8}
            ],
        }
    )
    (tmp_path / "neg").write_bytes(negative)
    assert not catalog_io.is_valid(tmp_path / "neg", schema_version=SCHEMA_VERSION)

    oversized = _raw_artifact(
        {
            "schema_version": SCHEMA_VERSION,
            "columns": [
                {
                    "name": "x",
                    "dtype": "<i8",
                    "count": 1,
                    "offset": 0,
                    "nbytes": 1 << 30,
                }
            ],
        }
    )
    (tmp_path / "big").write_bytes(oversized)
    assert not catalog_io.is_valid(tmp_path / "big", schema_version=SCHEMA_VERSION)


def test_pointer_write_read_roundtrip(tmp_path: Path) -> None:
    ptr = tmp_path / "ptr"
    catalog_io.write_pointer(ptr, "abc123")
    assert catalog_io.read_pointer(ptr) == "abc123"


def test_read_pointer_missing_or_empty_is_none(tmp_path: Path) -> None:
    assert catalog_io.read_pointer(tmp_path / "absent") is None
    empty = tmp_path / "empty"
    empty.write_text("   \n", encoding="utf-8")
    assert catalog_io.read_pointer(empty) is None


def test_read_pointer_strips_whitespace(tmp_path: Path) -> None:
    ptr = tmp_path / "ptr"
    ptr.write_text("  fingerprint-xyz\n", encoding="utf-8")
    assert catalog_io.read_pointer(ptr) == "fingerprint-xyz"
