# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Typed row conversion, null preservation, and buffer ownership."""

import gc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("vortex.io", reason="vortex-data not installed")
import vortex

from zephon._internal.io.formats.vortex import VortexShard, _column_values


@pytest.mark.parametrize(
    "kind", [pa.list_, pa.large_list, pa.list_view, pa.large_list_view]
)
def test_numeric_lists_preserve_offsets_nulls_and_dtype(kind) -> None:
    array = pa.array([[99], [1, 2], None, [], [3]], type=kind(pa.int16())).slice(1)
    rows = _column_values(array)
    np.testing.assert_array_equal(rows[0], np.array([1, 2], dtype=np.int16))
    assert rows[0].dtype == np.int16
    assert rows[1] is None
    assert rows[2].shape == (0,)
    assert rows[3].tolist() == [3]
    assert not rows[0].flags.writeable
    assert rows[0].base is not None


def test_fixed_size_shapes_and_null_parents() -> None:
    matrix_type = pa.list_(pa.list_(pa.float32(), 2), 2)
    array = pa.array(
        [[[0, 0], [0, 0]], [[1, 2], [3, 4]], None, [[5, 6], [7, 8]]],
        type=matrix_type,
    ).slice(1)
    rows = _column_values(array)
    assert rows[0].shape == (2, 2)
    assert rows[0].dtype == np.float32
    np.testing.assert_array_equal(rows[0], [[1, 2], [3, 4]])
    assert rows[1] is None
    np.testing.assert_array_equal(rows[2], [[5, 6], [7, 8]])


def test_fixed_size_slice_uses_only_selected_rows() -> None:
    array = pa.array([[9, 9], [1, 2], [3, 4]], type=pa.list_(pa.int32(), 2)).slice(1)
    rows = _column_values(array)
    np.testing.assert_array_equal(rows, [[1, 2], [3, 4]])
    assert rows.dtype == np.int32
    assert not rows.flags.writeable


def test_inner_nulls_and_nonnumeric_lists_are_lossless() -> None:
    numbers = _column_values(pa.array([[1, None], [2, 3]], type=pa.list_(pa.int16())))
    assert numbers[0] == [1, None]
    assert numbers[1].tolist() == [2, 3]
    assert numbers[1].dtype == np.int16
    assert _column_values(pa.array([["a", "b"], None, []])) == [["a", "b"], None, []]
    assert _column_values(pa.array([1, None], type=pa.int64())) == [1, None]


def test_boolean_lists_unpack_to_readonly_numpy() -> None:
    rows = _column_values(pa.array([[True, False], []], type=pa.list_(pa.bool_())))
    assert rows[0].dtype == np.bool_
    assert rows[0].tolist() == [True, False]
    assert not rows[0].flags.writeable


def test_row_order_across_different_column_chunk_boundaries(tmp_path: Path) -> None:
    table = pa.table(
        {
            "id": pa.chunked_array([[0], [1, 2]], type=pa.int32()),
            "vector": pa.chunked_array(
                [[[1, 2], []], [[3]]], type=pa.list_(pa.int16())
            ),
        }
    )
    batch = SimpleNamespace(to_arrow_table=lambda: table)
    file = SimpleNamespace(
        scan=lambda **_kwargs: SimpleNamespace(read_all=lambda: batch)
    )
    shard = VortexShard.__new__(VortexShard)
    shard._file = file
    shard._path = tmp_path / "unused.vortex"
    shard._length = 3
    rows = shard.getsamples([2, 0, 2, 1])
    assert [row["id"] for row in rows] == [2, 0, 2, 1]
    assert [row["vector"].tolist() for row in rows] == [[3], [1, 2], [3], []]
    assert all(row["vector"].dtype == np.int16 for row in rows)
    shard.close()


def test_vortex_rows_survive_close_and_preserve_duplicate_order(tmp_path: Path) -> None:
    path = tmp_path / "typed.vortex"
    table = pa.table(
        {
            "id": pa.array([0, 1, 2], type=pa.int32()),
            "vector": pa.array([[1, 2], [], [3, 4, 5]], type=pa.list_(pa.float32())),
            "matrix": pa.array(
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
                type=pa.list_(pa.list_(pa.int16(), 2), 2),
            ),
            "text": ["zero", "one", "two"],
            "binary": [b"a", b"b", b"c"],
        }
    )
    vortex.io.write(table, str(path))
    shard = VortexShard(path)
    rows = shard.getsamples([2, 0, 2, 1])
    single = shard[2]
    shard.close()
    del shard
    gc.collect()
    assert [row["id"] for row in rows] == [2, 0, 2, 1]
    assert rows[0]["id"].dtype == np.int32
    assert rows[0]["vector"].dtype == np.float32
    assert rows[0]["vector"].tolist() == [3, 4, 5]
    np.testing.assert_array_equal(single["vector"], rows[0]["vector"])
    assert rows[0]["matrix"].shape == (2, 2)
    assert rows[0]["matrix"].dtype == np.int16
    assert rows[0]["text"] == "two"
    assert rows[0]["binary"] == b"c"
    assert rows[3]["vector"].shape == (0,)
    assert np.shares_memory(rows[0]["vector"], rows[2]["vector"])
    assert not rows[0]["vector"].flags.writeable
