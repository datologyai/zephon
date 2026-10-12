# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Typed row conversion, null preservation, and buffer ownership."""

import gc
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("vortex.io", reason="vortex-data not installed")
import vortex

from zephon._internal.io.formats import vortex as vortex_format
from zephon._internal.io.formats.vortex import (
    VortexShard,
    _column_values,
    _list_view_type,
    _rows_from_table,
)


@pytest.mark.parametrize("indices", [[73], [73, 73, 73], [76, 74, 75, 73, 74]])
def test_contiguous_read_limits_the_decoded_list_values(
    tmp_path: Path, indices: list[int]
) -> None:
    """Compressed token columns expose only the selected decoded child values."""
    tokens = np.random.default_rng(0).integers(0, 256, size=(256, 32), dtype=np.uint32)
    array = pa.ListArray.from_arrays(
        pa.array(np.arange(0, tokens.size + 1, 32), type=pa.int32()),
        pa.array(tokens.reshape(-1)),
    )
    path = tmp_path / "tokens.vortex"
    vortex.io.write(pa.table({"tokens": array}), str(path))
    shard = VortexShard(path)
    try:
        rows = shard.getsamples(indices)
    finally:
        shard.close()
    for index, row in zip(indices, rows):
        values = row["tokens"]
        np.testing.assert_array_equal(values, tokens[index])
        assert values.dtype == np.uint32
        assert not values.flags.writeable
        owner = values
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        assert owner.nbytes == values.nbytes * len(set(indices))


def test_list_view_export_preserves_field_metadata_and_fixed_shapes() -> None:
    child = pa.field("element", pa.int16(), nullable=False, metadata={b"unit": b"x"})
    field = pa.field("values", pa.list_(child), metadata={b"label": b"tokens"})
    kind = pa.struct([field, pa.field("matrix", pa.list_(pa.list_(child), 2))])
    transformed = _list_view_type(kind, pa)
    assert transformed[0].metadata == field.metadata
    assert transformed[0].nullable == field.nullable
    assert transformed[0].type.value_field == child
    assert pa.types.is_list_view(transformed[0].type)
    assert transformed[1].type.list_size == 2
    assert pa.types.is_list_view(transformed[1].type.value_type)


@pytest.mark.parametrize("indices", [[1], [1, 1], [2, 1, 2, 0], [0, 2]])
def test_text_only_reads_preserve_nulls_bytes_and_duplicate_order(
    tmp_path: Path, indices: list[int]
) -> None:
    table = pa.table({"text": ["é", None, "漢字"], "bytes": [b"\x00", b"", None]})
    path = tmp_path / "text.vortex"
    vortex.io.write(table, str(path))
    shard = VortexShard(path)
    rows = shard.getsamples(indices)
    shard.close()
    expected = table.to_pylist()
    assert rows == [expected[index] for index in indices]
    if len(indices) > 1:
        assert rows[0] is not rows[1]


@pytest.mark.parametrize("indices", [[1], [1, 1], [2, 1, 0]])
def test_nullable_native_struct_uses_arrow_table_row_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, indices: list[int]
) -> None:
    """A null struct scalar must not fail or depend on request batch size."""
    source = vortex.array(
        pa.array(
            [{"text": "ok", "bytes": b"x"}, None, {"text": None, "bytes": None}],
            type=pa.struct([("text", pa.string()), ("bytes", pa.binary())]),
        )
    )

    def scan(*, indices: Any) -> Any:
        selected = source.take(indices)
        return vortex.ArrayIterator.from_iter(selected.dtype, iter([selected]))

    # Vortex 0.88's file statistics writer rejects nullable root structs. Feed a
    # real native nullable array through the reader to exercise conversion alone.
    file = SimpleNamespace(dtype=source.dtype, scan=scan)
    monkeypatch.setattr(vortex_format, "_open_vortex_file", lambda *_args: file)
    shard = VortexShard(tmp_path / "nullable.vortex", length=len(source))
    try:
        rows = shard.getsamples(indices)
    finally:
        shard.close()
    expected = source.to_arrow_table().to_pylist()
    assert rows == [expected[index] for index in indices]


@pytest.mark.parametrize("indices", [[1], [3, 1, 2, 0, 1], [0, 2]])
def test_list_export_preserves_inner_nulls_and_nested_fallbacks(
    tmp_path: Path, indices: list[int]
) -> None:
    table = pa.table(
        {
            "numbers": pa.array(
                [[1, None], [2, 3], None, []], type=pa.list_(pa.int16())
            ),
            "nested": pa.array([[[1], [2, 3]], None, [], [[4]]]),
            "strings": pa.array([["a"], None, [], ["b", "c"]]),
            "structs": pa.array([[{"a": 1}], [], None, [{"a": None}]]),
        }
    )
    path = tmp_path / "nested.vortex"
    vortex.io.write(table, str(path))
    shard = VortexShard(path)
    rows = shard.getsamples(indices)
    shard.close()
    expected = table.to_pylist()
    for index, row in zip(indices, rows):
        numbers = row["numbers"]
        if isinstance(numbers, np.ndarray):
            assert numbers.dtype == np.int16
            assert not numbers.flags.writeable
            row["numbers"] = numbers.tolist()
        assert row == expected[index]


@pytest.mark.parametrize(
    "kind", [pa.list_, pa.large_list, pa.list_view, pa.large_list_view]
)
def test_numeric_lists_preserve_offsets_nulls_and_dtype(kind: Any) -> None:
    array = pa.array([[99], [1, 2], None, [], [3]], type=kind(pa.int16())).slice(1)
    rows = _column_values(array, pa)
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
    rows = _column_values(array, pa)
    assert rows[0].shape == (2, 2)
    assert rows[0].dtype == np.float32
    np.testing.assert_array_equal(rows[0], [[1, 2], [3, 4]])
    assert rows[1] is None
    np.testing.assert_array_equal(rows[2], [[5, 6], [7, 8]])


def test_fixed_size_slice_uses_only_selected_rows() -> None:
    array = pa.array([[9, 9], [1, 2], [3, 4]], type=pa.list_(pa.int32(), 2)).slice(1)
    rows = _column_values(array, pa)
    np.testing.assert_array_equal(rows, [[1, 2], [3, 4]])
    assert isinstance(rows, np.ndarray)
    assert rows.dtype == np.int32
    assert not rows.flags.writeable


def test_inner_nulls_and_nonnumeric_lists_preserve_values() -> None:
    numbers = _column_values(
        pa.array([[1, None], [2, 3]], type=pa.list_(pa.int16())), pa
    )
    assert numbers[0] == [1, None]
    assert numbers[1].tolist() == [2, 3]
    assert numbers[1].dtype == np.int16
    assert _column_values(pa.array([["a", "b"], None, []]), pa) == [
        ["a", "b"],
        None,
        [],
    ]
    assert _column_values(pa.array([1, None], type=pa.int64()), pa) == [1, None]


def test_boolean_lists_unpack_to_readonly_numpy() -> None:
    rows = _column_values(pa.array([[True, False], []], type=pa.list_(pa.bool_())), pa)
    assert rows[0].dtype == np.bool_
    assert rows[0].tolist() == [True, False]
    assert not rows[0].flags.writeable


@pytest.mark.parametrize(
    "array",
    [
        pa.array([["a", "b"], None, []]),
        pa.array([[{"a": 1, "b": "x"}], None, []]),
        pa.array([[[1, 2], [3]], None, []]),
        pa.array([["a", "b"], None, ["c", "d"]], type=pa.list_(pa.string(), 2)),
    ],
)
def test_nonnumeric_lists_skip_per_row_numeric_conversion(
    array: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_numeric(*_args: Any) -> None:
        raise AssertionError("nonnumeric lists must use whole-column conversion")

    monkeypatch.setattr(vortex_format, "_numeric_values", reject_numeric)
    assert _column_values(array, pa) == array.to_pylist()


@pytest.mark.parametrize("dtype", [pa.int32(), pa.uint64(), pa.float32(), pa.float64()])
def test_numeric_scalar_types_do_not_depend_on_null_neighbours(
    tmp_path: Path, dtype: Any
) -> None:
    path = tmp_path / "nullable.vortex"
    vortex.io.write(pa.table({"v": pa.array([1, 2, None, 4], type=dtype)}), str(path))
    shard = VortexShard(path)
    try:
        single = shard[0]["v"]
        assert isinstance(single, np.generic)
        for indices in ([0, 1], [0, 2], [2, 0, 0], [0, 1, 2, 3]):
            rows = shard.getsamples(indices)
            for index, row in zip(indices, rows):
                if index == 2:
                    assert row["v"] is None
                else:
                    assert type(row["v"]) is type(single)
                    assert row["v"] == [1, 2, None, 4][index]
    finally:
        shard.close()


def test_nullable_scalars_across_chunks_keep_dtype_and_distinguish_nan() -> None:
    table = pa.table(
        {
            "v": pa.chunked_array([[1], [None, 2]], type=pa.int16()),
            "f": pa.chunked_array([[float("nan"), None], [3]], type=pa.float32()),
        }
    )
    rows = _rows_from_table(table, [2, 0, 1])
    assert type(rows[0]["v"]) is np.int16
    assert type(rows[1]["v"]) is np.int16
    assert rows[2]["v"] is None
    assert type(rows[1]["f"]) is np.float32
    assert np.isnan(rows[1]["f"])
    assert rows[2]["f"] is None


def test_row_order_across_different_column_chunk_boundaries() -> None:
    table = pa.table(
        {
            "id": pa.chunked_array([[0], [1, 2]], type=pa.int32()),
            "vector": pa.chunked_array(
                [[[1, 2], []], [[3]]], type=pa.list_(pa.int16())
            ),
        }
    )
    rows = _rows_from_table(table, [2, 0, 2, 1])
    assert [row["id"] for row in rows] == [2, 0, 2, 1]
    assert [row["vector"].tolist() for row in rows] == [[3], [1, 2], [3], []]
    assert all(row["vector"].dtype == np.int16 for row in rows)


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
