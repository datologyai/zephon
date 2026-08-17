# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

from zephon._internal.io.formats import arrow_rows
from zephon._internal.io.formats.arrow_rows import (
    arrow_table_to_numpy,
    extract_row,
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


def test_scalar_columns() -> None:
    table = pa.table({"x": pa.array([1, 2, 3]), "y": pa.array([1.5, 2.5, 3.5])})
    result, decoded_bytes = arrow_table_to_numpy(table)

    assert set(result) == {"x", "y"}
    np.testing.assert_array_equal(result["x"], [1, 2, 3])
    np.testing.assert_array_equal(result["y"], [1.5, 2.5, 3.5])
    assert result["x"].shape == (3,)
    assert decoded_bytes == table.nbytes


def test_fixed_size_list_column() -> None:
    inner = pa.array([10, 20, 30, 40, 50, 60], type=pa.uint32())
    fixed = pa.FixedSizeListArray.from_arrays(inner, list_size=3)
    result, _ = arrow_table_to_numpy(pa.table({"tokens": fixed}))

    assert result["tokens"].shape == (2, 3)
    np.testing.assert_array_equal(result["tokens"][0], [10, 20, 30])
    np.testing.assert_array_equal(result["tokens"][1], [40, 50, 60])


def test_nullable_fixed_size_list_preserves_null() -> None:
    fixed = pa.array([[1, 2], None, [5, 6]], type=pa.list_(pa.int32(), 2))
    table = pa.table({"tokens": fixed})
    result, _ = arrow_table_to_numpy(table)

    assert result["tokens"].dtype == object
    np.testing.assert_array_equal(result["tokens"][0], [1, 2])
    assert result["tokens"][1] is None
    np.testing.assert_array_equal(result["tokens"][2], [5, 6])
    assert take_and_materialize(table, [1]) == [{"tokens": None}]


def test_variable_length_list_column() -> None:
    ragged = pa.array([[1, 2], [3, 4, 5], [6]], type=pa.list_(pa.int64()))
    result, _ = arrow_table_to_numpy(pa.table({"ragged": ragged}))

    assert result["ragged"].dtype == object
    np.testing.assert_array_equal(result["ragged"][0], [1, 2])
    np.testing.assert_array_equal(result["ragged"][1], [3, 4, 5])
    np.testing.assert_array_equal(result["ragged"][2], [6])


def test_nullable_variable_length_list_distinguishes_null_from_empty() -> None:
    ragged = pa.array([[1, 2], None, [], [3]], type=pa.list_(pa.int64()))
    table = pa.table({"ragged": ragged})
    result, _ = arrow_table_to_numpy(table)

    np.testing.assert_array_equal(result["ragged"][0], [1, 2])
    assert result["ragged"][1] is None
    np.testing.assert_array_equal(result["ragged"][2], [])
    np.testing.assert_array_equal(result["ragged"][3], [3])
    rows = take_and_materialize(table, [1, 2])
    assert rows[0]["ragged"] is None
    np.testing.assert_array_equal(rows[1]["ragged"], [])


def test_taken_duplicate_list_rows_are_independent_writable_views() -> None:
    fixed = pa.array([[1, 2], [3, 4], [5, 6]], type=pa.list_(pa.int32(), 2))
    ragged = pa.array([[1], [2, 3], [4, 5, 6]], type=pa.list_(pa.int32()))
    rows = take_and_materialize(
        pa.table({"fixed": fixed, "ragged": ragged}),
        [2, 0, 2],
    )

    for name in ("fixed", "ragged"):
        first = rows[0][name]
        duplicate = rows[2][name]
        assert isinstance(first, np.ndarray)
        assert isinstance(duplicate, np.ndarray)
        assert first.flags.writeable
        assert not first.flags.owndata
        assert not np.shares_memory(first, duplicate)

    rows[0]["fixed"][0] = 99
    rows[0]["ragged"][0] = 99
    assert rows[2]["fixed"][0] == 5
    assert rows[2]["ragged"][0] == 4


def test_string_column_fallback() -> None:
    result, _ = arrow_table_to_numpy(pa.table({"text": ["hello", "world"]}))

    assert result["text"].tolist() == ["hello", "world"]


def test_extract_row_from_mixed_table() -> None:
    inner = pa.array(list(range(12)), type=pa.uint32())
    fixed = pa.FixedSizeListArray.from_arrays(inner, list_size=4)
    columns, _ = arrow_table_to_numpy(
        pa.table({"id": pa.array([10, 20, 30]), "tokens": fixed})
    )

    row = extract_row(columns, 1)
    assert row["id"] == 20
    np.testing.assert_array_equal(row["tokens"], [4, 5, 6, 7])
