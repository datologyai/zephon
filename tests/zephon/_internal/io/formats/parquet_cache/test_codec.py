# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from zephon._internal.io.formats.parquet_cache.codec import (
    ArrowRGFileCodec,
    DecodedRGIdentity,
)

_IDENTITY = DecodedRGIdentity("sha256:" + "bc" * 32)


def _representative_table():
    fixed = pa.FixedSizeListArray.from_arrays(
        pa.array([1, 2, 3, 4, 5, 6], type=pa.int32()),
        list_size=2,
    )
    dictionary = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, 0]),
        pa.array(["red", "blue"]),
    )
    return pa.table(
        {
            "number": pa.array([10, 20, 30], type=pa.int64()),
            "nullable": pa.array([1, None, 3], type=pa.int32()),
            "text": ["a", "bb", "ccc"],
            "tokens": pa.array([[1, 2], [], [3, 4, 5]], type=pa.list_(pa.int32())),
            "fixed": fixed,
            "nested": pa.array([{"x": 1, "y": "u"}, {"x": 2, "y": "v"}, None]),
            "dictionary": dictionary,
        }
    )


def test_exact_measure_stream_write_and_mapped_owned_rows(tmp_path: Path) -> None:
    codec = ArrowRGFileCodec()
    table = _representative_table()
    measured = codec.measure(table, _IDENTITY)
    destination = tmp_path / "row-group.arrow"

    assert codec.write(table, _IDENTITY, destination, exact_bytes=measured) == measured
    assert destination.stat().st_size == measured

    mapped = codec.open_mapped(
        destination,
        _IDENTITY,
        exact_bytes=measured,
    )
    rows = mapped.take_and_materialize([2, 0, 2])
    mapped.close()
    destination.unlink()

    with pytest.raises(RuntimeError, match="mapping is closed"):
        mapped.take_and_materialize([0])
    assert [row["number"] for row in rows] == [30, 10, 30]
    assert [row["text"] for row in rows] == ["ccc", "a", "ccc"]
    assert np.array_equal(rows[0]["tokens"], np.array([3, 4, 5]))
    assert np.array_equal(rows[1]["fixed"], np.array([1, 2]))
    assert rows[1]["nested"] == {"x": 1, "y": "u"}
    assert rows[0]["dictionary"] == "red"
    rows[0]["tokens"][0] = 99
    assert rows[2]["tokens"][0] == 3
    rows[0]["fixed"][0] = 99
    assert rows[2]["fixed"][0] == 5


def test_identity_mismatch_is_rejected_before_row_materialization(
    tmp_path: Path,
) -> None:
    codec = ArrowRGFileCodec()
    table = _representative_table()
    measured = codec.measure(table, _IDENTITY)
    destination = tmp_path / "row-group.arrow"
    codec.write(table, _IDENTITY, destination, exact_bytes=measured)

    with pytest.raises(RuntimeError, match="identity mismatch"):
        codec.open_mapped(
            destination,
            DecodedRGIdentity("sha256:" + "de" * 32),
            exact_bytes=measured,
        )


def test_measurement_mismatch_removes_private_temp(tmp_path: Path) -> None:
    codec = ArrowRGFileCodec()
    table = _representative_table()
    destination = tmp_path / "row-group.tmp"

    with pytest.raises(RuntimeError, match="measurement mismatch"):
        codec.write(
            table,
            _IDENTITY,
            destination,
            exact_bytes=codec.measure(table, _IDENTITY) + 1,
        )
    assert not destination.exists()


def test_write_never_replaces_an_existing_path(tmp_path: Path) -> None:
    codec = ArrowRGFileCodec()
    table = _representative_table()
    destination = tmp_path / "row-group.arrow"
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        codec.write(
            table,
            _IDENTITY,
            destination,
            exact_bytes=codec.measure(table, _IDENTITY),
        )
    assert destination.read_bytes() == b"existing"


def test_payload_size_and_closed_mapping_are_validated(tmp_path: Path) -> None:
    codec = ArrowRGFileCodec()
    table = _representative_table()
    measured = codec.measure(table, _IDENTITY)
    destination = tmp_path / "row-group.arrow"
    codec.write(table, _IDENTITY, destination, exact_bytes=measured)

    with pytest.raises(RuntimeError, match="size mismatch"):
        codec.open_mapped(destination, _IDENTITY, exact_bytes=measured + 1)

    mapped = codec.open_mapped(destination, _IDENTITY, exact_bytes=measured)
    mapped.close()
    with pytest.raises(RuntimeError, match="mapping is closed"):
        mapped.take_and_materialize([0])
