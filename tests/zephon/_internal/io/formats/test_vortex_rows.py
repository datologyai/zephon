# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex export layouts, scalar reads and output lifetime."""

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
    _list_view_type,
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
        values = row
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


@pytest.mark.parametrize("dtype", [pa.int32(), pa.uint64(), pa.float32(), pa.float64()])
def test_numeric_scalar_types_do_not_depend_on_null_neighbours(
    tmp_path: Path, dtype: Any
) -> None:
    path = tmp_path / "nullable.vortex"
    vortex.io.write(pa.table({"v": pa.array([1, 2, None, 4], type=dtype)}), str(path))
    shard = VortexShard(path)
    try:
        single = shard[0]
        assert isinstance(single, np.generic)
        for indices in ([0, 1], [0, 2], [2, 0, 0], [0, 1, 2, 3]):
            rows = shard.getsamples(indices)
            for index, row in zip(indices, rows):
                if index == 2:
                    assert row is None
                else:
                    assert type(row) is type(single)
                    assert row == [1, 2, None, 4][index]
    finally:
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
