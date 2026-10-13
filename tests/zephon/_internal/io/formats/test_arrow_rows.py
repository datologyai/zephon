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
from zephon.io import ArrowOptions


def test_pyarrow_import_does_not_resize_pools(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def record_call(*args: Any) -> None:
        calls.append(args)

    monkeypatch.setattr(arrow_rows, "_pa", None)
    monkeypatch.setattr(arrow_rows, "_arrow_options", None)
    monkeypatch.setattr(arrow_rows, "configure_arrow_threads", record_call)

    assert arrow_rows.require_pyarrow() is pa
    assert arrow_rows.require_pyarrow() is pa
    assert calls == []


def test_pyarrow_store_configuration_survives_later_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(arrow_rows, "_pa", None)
    monkeypatch.setattr(arrow_rows, "_arrow_options", None)
    monkeypatch.setattr(
        arrow_rows, "configure_arrow_threads", lambda *args: calls.append(args)
    )
    options = ArrowOptions(cpu_threads=2, io_threads=None)
    arrow_rows.configure_pyarrow(options)
    assert arrow_rows.require_pyarrow() is pa
    assert arrow_rows.require_pyarrow() is pa
    assert calls == [(pa, options)]


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
        pa.array([["a", "b"], None, []], type=pa.list_view(pa.string())),
        pa.array(
            [[{"a": 1}], None, []],
            type=pa.large_list_view(pa.struct([("a", pa.int32())])),
        ),
        pa.array(
            [[[1, None], [2]], None, []], type=pa.list_view(pa.list_view(pa.int64()))
        ),
        pa.array([[[1], [2]], None, []], type=pa.list_(pa.list_view(pa.int64()))),
        pa.array(
            [[[1], [2]], None, [[3], []]], type=pa.list_(pa.list_view(pa.int64()), 2)
        ),
        pa.ListViewArray.from_arrays([2, 0, 1], [2, 2, 2], ["a", "b", "c", "d"]),
        pa.array(
            [{"items": ["a", None]}, None, {"items": []}],
            type=pa.struct([("items", pa.list_view(pa.string()))]),
        ),
        pa.array(
            [[[0], [0]], [[1], [2]], None, [[3], []]],
            type=pa.list_(pa.list_view(pa.int64()), 2),
        ).slice(1),
    ],
)
def test_nonnumeric_columns_skip_per_row_numeric_conversion(
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
    assert all(isinstance(row["value"], np.generic) for row in rows)
    assert all(
        row["value"].dtype
        == pa.array([], type=dtype).to_numpy(zero_copy_only=False).dtype
        for row in rows
    )
    assert rows == [{"value": 1}, {"value": 2}]


@pytest.mark.parametrize(
    "array",
    [
        pa.array(["a", None, "b"]),
        pa.array([b"a", None, b"b"]),
        pa.array([True, None, False]),
        pa.array([{"x": 1}, None, {"x": 2}]),
    ],
)
def test_single_column_preserves_field_name(array: Any) -> None:
    rows = arrow_table_to_samples(pa.table({"value": array}))
    expected = array.to_pylist()
    assert rows == [{"value": value} for value in expected]
    assert [type(row["value"]) for row in rows] == [type(value) for value in expected]


def test_single_numeric_list_column_keeps_readonly_views_after_table_release() -> None:
    array = pa.array([[1, 2], [], [3]], type=pa.list_(pa.uint32()))
    source = array.values.to_numpy()
    table = pa.table({"tokens": array})
    rows = arrow_table_to_samples(table, [2, 0, 2, 1])
    del table, array
    gc.collect()
    assert [row["tokens"].tolist() for row in rows] == [[3], [1, 2], [3], []]
    assert all(
        row["tokens"].dtype == np.uint32 and not row["tokens"].flags.writeable
        for row in rows
    )
    assert np.shares_memory(rows[0]["tokens"], source)
    assert np.shares_memory(rows[0]["tokens"], rows[2]["tokens"])


def test_duplicate_mutable_fallbacks_are_independent() -> None:
    fields = {
        "nested": pa.array([[{"x": [1, None]}], [{"x": [2]}]]),
        "tokens": pa.array([[1, 2], [3]], type=pa.list_(pa.int32())),
    }
    rows = arrow_table_to_samples(pa.table(fields), [0, 1, 0])
    first = rows[0]["nested"]
    duplicate = rows[2]["nested"]
    first[0]["x"][0] = 99
    assert duplicate == [{"x": [1, None]}]
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
    assert [row["tokens"].tolist() for row in rows] == [[3, 4], [1, 2], [3, 4]]
    assert all(not row["tokens"].flags.writeable for row in rows)
    assert not np.shares_memory(
        rows[0]["tokens"], table.column(0).chunk(0).values.to_numpy()
    )
    with pytest.raises(ValueError, match="read-only"):
        rows[0]["tokens"][0] = 99


def test_empty_column_and_empty_selection() -> None:
    table = pa.table({"tokens": pa.chunked_array([], type=pa.list_(pa.int32()))})
    assert arrow_table_to_samples(table) == []
    assert take_and_materialize(table, []) == []


def test_arrow_configuration_is_lazy_and_applied_after_import() -> None:
    import subprocess
    import sys
    import textwrap

    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
        import sys
        from zephon._internal.io.formats.arrow_rows import configure_pyarrow, require_pyarrow
        from zephon.io import ArrowOptions

        assert "pyarrow" not in sys.modules
        configure_pyarrow(ArrowOptions(cpu_threads=2, io_threads=3))
        assert "pyarrow" not in sys.modules
        pa = require_pyarrow()
        assert (pa.cpu_count(), pa.io_thread_count()) == (2, 3)
        configure_pyarrow(ArrowOptions(cpu_threads=None, io_threads=1))
        assert (pa.cpu_count(), pa.io_thread_count()) == (2, 1)
        assert require_pyarrow() is pa
        assert (pa.cpu_count(), pa.io_thread_count()) == (2, 1)
    """),
        ],
        check=True,
        timeout=30,
    )
