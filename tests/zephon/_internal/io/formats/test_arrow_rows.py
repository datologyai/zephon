# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared Arrow conversion contract for Parquet and Vortex."""

import gc
from typing import Any

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

from zephon._internal.io.formats import arrow_rows
from zephon._internal.io.formats.arrow_rows import (
    _column_values,
    arrow_table_to_samples,
    take_and_materialize,
)


def test_pyarrow_thread_cap_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def count_call() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(arrow_rows, "_pa", None)
    monkeypatch.setattr(arrow_rows, "cap_arrow_threads", count_call)

    assert arrow_rows.require_pyarrow() is pa
    assert arrow_rows.require_pyarrow() is pa
    assert calls == 1


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


def test_inner_nulls_use_python_only_for_affected_rows() -> None:
    numbers = _column_values(
        pa.array([[1, None], [2, 3]], type=pa.list_(pa.int16())), pa
    )
    assert numbers[0] == [1, None]
    assert numbers[1].tolist() == [2, 3]
    assert numbers[1].dtype == np.int16


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

    monkeypatch.setattr(arrow_rows, "_numeric_values", reject_numeric)
    assert _column_values(array, pa) == array.to_pylist()


def test_nullable_scalars_across_chunks_keep_dtype_and_distinguish_nan() -> None:
    table = pa.table(
        {
            "v": pa.chunked_array([[1], [None, 2]], type=pa.int16()),
            "f": pa.chunked_array([[float("nan"), None], [3]], type=pa.float32()),
        }
    )
    rows = arrow_table_to_samples(table, [2, 0, 1])
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
    rows = arrow_table_to_samples(table, [2, 0, 2, 1])
    assert [row["id"] for row in rows] == [2, 0, 2, 1]
    assert [row["vector"].tolist() for row in rows] == [[3], [1, 2], [3], []]
    assert all(row["vector"].dtype == np.int16 for row in rows)
    for row in rows:
        chunk = table.column("vector").chunk(1 if row["id"] == 2 else 0)
        if row["vector"].size:
            assert np.shares_memory(row["vector"], chunk.values.to_numpy())


@pytest.mark.parametrize("dtype", [pa.int16(), pa.uint64(), pa.float32()])
def test_single_numeric_column_returns_typed_scalars(dtype: Any) -> None:
    rows = arrow_table_to_samples(pa.table({"value": pa.array([1, 2], type=dtype)}))
    assert all(isinstance(value, np.generic) for value in rows)
    assert all(
        value.dtype == pa.array([], type=dtype).to_numpy(zero_copy_only=False).dtype
        for value in rows
    )
    assert rows == [1, 2]


@pytest.mark.parametrize(
    "array",
    [
        pa.array(["a", None, "b"]),
        pa.array([b"a", None, b"b"]),
        pa.array([True, None, False]),
        pa.array([{"x": 1}, None, {"x": 2}]),
    ],
)
def test_single_column_returns_its_value(array: Any) -> None:
    rows = arrow_table_to_samples(pa.table({"value": array}))
    expected = array.to_pylist()
    assert rows == expected
    assert [type(value) for value in rows] == [type(value) for value in expected]


def test_single_numeric_list_column_keeps_readonly_views_after_table_release() -> None:
    array = pa.array([[1, 2], [], [3]], type=pa.list_(pa.uint32()))
    source = array.values.to_numpy()
    table = pa.table({"tokens": array})
    rows = arrow_table_to_samples(table, [2, 0, 2, 1])
    del table, array
    gc.collect()
    assert [row.tolist() for row in rows] == [[3], [1, 2], [3], []]
    assert all(row.dtype == np.uint32 and not row.flags.writeable for row in rows)
    assert np.shares_memory(rows[0], source)
    assert np.shares_memory(rows[0], rows[2])


@pytest.mark.parametrize("multiple_columns", [False, True])
def test_duplicate_mutable_fallbacks_are_independent(multiple_columns: bool) -> None:
    fields = {"nested": pa.array([[{"x": [1, None]}], [{"x": [2]}]])}
    if multiple_columns:
        fields["tokens"] = pa.array([[1, 2], [3]], type=pa.list_(pa.int32()))
    rows = arrow_table_to_samples(pa.table(fields), [0, 1, 0])
    first = rows[0]["nested"] if multiple_columns else rows[0]
    duplicate = rows[2]["nested"] if multiple_columns else rows[2]
    first[0]["x"][0] = 99
    assert duplicate == [{"x": [1, None]}]
    if multiple_columns:
        assert np.shares_memory(rows[0]["tokens"], rows[2]["tokens"])
        assert not rows[2]["tokens"].flags.writeable


def test_nullable_integers_never_pass_through_float() -> None:
    maximum = np.iinfo(np.int64).max
    table = pa.table(
        {
            "scalar": pa.array([maximum, None], type=pa.int64()),
            "list": pa.array([[maximum, None], [3, 4]], type=pa.list_(pa.int64())),
        }
    )
    rows = arrow_table_to_samples(table)
    assert type(rows[0]["scalar"]) is np.int64
    assert rows[0]["scalar"] == maximum
    assert rows[1]["scalar"] is None
    assert rows[0]["list"] == [maximum, None]
    assert rows[1]["list"].dtype == np.int64


def test_parquet_take_retains_selection_but_returns_readonly_arrays() -> None:
    table = pa.table({"tokens": pa.array([[1, 2], [3, 4]], type=pa.list_(pa.int32()))})
    rows = take_and_materialize(table, [1, 0, 1])
    assert [row.tolist() for row in rows] == [[3, 4], [1, 2], [3, 4]]
    assert all(not row.flags.writeable for row in rows)
    assert not np.shares_memory(rows[0], table.column(0).chunk(0).values.to_numpy())
    with pytest.raises(ValueError, match="read-only"):
        rows[0][0] = 99


def test_empty_column_and_empty_selection() -> None:
    table = pa.table({"tokens": pa.chunked_array([], type=pa.list_(pa.int32()))})
    assert arrow_table_to_samples(table) == []
    assert take_and_materialize(table, []) == []
