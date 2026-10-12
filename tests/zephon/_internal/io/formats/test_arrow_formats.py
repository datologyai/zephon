# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Parquet and Vortex sample conversion parity."""

import gc
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from zephon._internal.io.formats.parquet import ParquetShard


def _write_and_open(tmp_path: Path, fmt: str, table: Any) -> Any:
    path = tmp_path / f"data.{fmt}"
    if fmt == "vortex":
        vortex = pytest.importorskip("vortex.io")
        from zephon._internal.io.formats.vortex import VortexShard

        vortex.write(table, str(path))
        return VortexShard(path)
    pq.write_table(table, path, row_group_size=2)
    metadata = pq.read_metadata(path)
    return ParquetShard(
        path,
        [
            {"num_rows": metadata.row_group(i).num_rows}
            for i in range(metadata.num_row_groups)
        ],
    )


@pytest.mark.parametrize("fmt", ["parquet", "vortex"])
@pytest.mark.parametrize(
    "array",
    [
        pa.array([np.iinfo(np.int64).max, None, 3, 4], type=pa.int64()),
        pa.array([True, None, False, True]),
        pa.array(["é", None, "漢字", ""]),
        pa.array([b"\x00", None, b"xyz", b""]),
        pa.array([[1, 2], None, [3], []], type=pa.list_(pa.uint32())),
        pa.array(
            [[np.iinfo(np.int64).max, None], [2, 3], None, []],
            type=pa.list_(pa.int64()),
        ),
        pa.array([[[1, 2], [3, 4]]] * 4, type=pa.list_(pa.list_(pa.int16(), 2), 2)),
        pa.array([{"x": [1, None]}, None, {"x": [2]}, {"x": []}]),
    ],
    ids=[
        "integer",
        "boolean",
        "text",
        "binary",
        "tokens",
        "inner-null",
        "matrix",
        "struct",
    ],
)
@pytest.mark.parametrize("multiple_columns", [False, True])
def test_payload_contract_across_single_and_bulk_reads(
    tmp_path: Path, fmt: str, array: Any, multiple_columns: bool
) -> None:
    fields = {"value": array}
    if multiple_columns:
        fields["index"] = pa.array(range(4), type=pa.int32())
    shard = _write_and_open(tmp_path, fmt, pa.table(fields))
    requests = [[0], [1], [0, 0], [3, 0, 3, 1, 2], [0, 2], []]
    try:
        outputs = [(indices, shard.getsamples(indices)) for indices in requests]
        outputs.extend(([i], [shard[i]]) for i in range(4))
    finally:
        shard.close()
    del shard
    gc.collect()
    for indices, rows in outputs:
        assert len(rows) == len(indices)
        for index, row in zip(indices, rows, strict=True):
            value = row["value"] if multiple_columns else row
            expected = array[index].as_py()
            if multiple_columns:
                assert row["index"] == index
                assert type(row["index"]) is np.int32
            if isinstance(value, np.ndarray):
                assert value.tolist() == expected
                assert not value.flags.writeable
                dtype = array.type
                while pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype):
                    dtype = dtype.value_type
                assert (
                    value.dtype
                    == pa.array([], type=dtype).to_numpy(zero_copy_only=False).dtype
                )
            else:
                assert value == expected
                if expected is not None and pa.types.is_integer(array.type):
                    assert type(value) is np.int64
                if expected is not None and pa.types.is_boolean(array.type):
                    assert type(value) is bool


@pytest.mark.parametrize("fmt", ["parquet", "vortex"])
def test_duplicate_fallbacks_do_not_share_mutable_containers(
    tmp_path: Path, fmt: str
) -> None:
    table = pa.table({"value": [[{"x": [1, None]}], [{"x": [2]}]]})
    shard = _write_and_open(tmp_path, fmt, table)
    try:
        rows = shard.getsamples([0, 1, 0])
    finally:
        shard.close()
    rows[0][0]["x"][0] = 99
    assert rows[2] == [{"x": [1, None]}]
