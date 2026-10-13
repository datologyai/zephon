# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Single-column Arrow payloads survive fetch and process transport."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from zephon import Pipeline
from zephon.io import Dataset
from zephon.work import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def _write_dataset(root: Path, fmt: str, table: Any) -> Dataset:
    root.mkdir()
    path = root / f"data.{fmt}"
    if fmt == "parquet":
        pq = pytest.importorskip("pyarrow.parquet")
        pq.write_table(table, path, row_group_size=3)
    else:
        vortex = pytest.importorskip("vortex.io")
        vortex.write(table, str(path))
    return Dataset.from_path("samples", str(root), fmt=fmt)


@pytest.mark.parametrize("fmt", ["parquet", "vortex"])
@pytest.mark.parametrize("runner", ["threads", "process"])
@pytest.mark.parametrize("array_length", [2048, 256 * 1024], ids=["inline", "shm"])
def test_single_column_payloads_through_fetch_and_transport(
    tmp_path: Path, fmt: str, runner: str, array_length: int
) -> None:
    if runner == "process":
        pytest.importorskip(
            "torch", reason="Read-only payload transport requires Torch"
        )
    pa = pytest.importorskip("pyarrow")
    # Exercise both inline payloads and allocations above the SHM threshold.
    values = [
        None if i == 1 else np.arange(i, i + array_length, dtype=np.int32)
        for i in range(8)
    ]
    array = pa.array(values, type=pa.list_(pa.int32()))
    table = pa.table({"value": array})
    dataset = _write_dataset(tmp_path / "data", fmt, table)
    work = StaticMixtureWorkSource(
        [dataset],
        {"samples": 1.0},
        chunk_size=4,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = (
        Pipeline(work)
        .batch(microbatch_size=4, drop_last=False)
        .options(
            runner=runner,
            max_workers=2,
            deterministic=True,
            io_options={"cache": {"enabled": False}},
        )
    )
    iterator = iter(pipeline)
    try:
        records = [record for batch in iterator for record in batch.records]
    finally:
        iterator.close()
    assert len(records) == len(values)
    for record in records:
        expected = values[record.meta.sample_id[2]]
        if expected is None:
            assert record.payload == {"value": None}
        else:
            assert isinstance(record.payload, dict)
            value = record.payload["value"]
            assert isinstance(value, np.ndarray)
            assert value.dtype == np.int32
            np.testing.assert_array_equal(value, expected)
            assert not value.flags.writeable


@pytest.mark.parametrize("fmt", ["parquet", "vortex"])
@pytest.mark.parametrize("operation", ["pack_sequences", "decode_text"])
def test_single_column_payloads_support_named_field_ops(
    tmp_path: Path, fmt: str, operation: str
) -> None:
    pa = pytest.importorskip("pyarrow")
    tokens = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10]]
    text = [b"Hello\r\nworld", "café".encode()]
    table = (
        pa.table({"input_ids": pa.array(tokens, type=pa.list_(pa.int32()))})
        if operation == "pack_sequences"
        else pa.table({"text": text})
    )
    dataset = _write_dataset(tmp_path / "data", fmt, table)
    work = StaticMixtureWorkSource(
        [dataset],
        {"samples": 1.0},
        chunk_size=2,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = Pipeline(work)
    if operation == "pack_sequences":
        pipeline.pack_sequences(max_length=12, num_bins=4)
    else:
        pipeline.decode_text()
    pipeline.options(runner="inline", max_workers=1, deterministic=True)
    records = list(pipeline)
    if operation == "pack_sequences":
        actual = [
            sample["input_ids"].tolist()
            for record in records
            for sample in record.payload["packed_samples"]
        ]
        assert sorted(actual) == sorted(tokens)
    else:
        assert len(records) == len(text)
        for record in records:
            expected = text[record.meta.sample_id[2]].decode().replace("\r\n", "\n")
            assert record.payload == {"text": expected}
