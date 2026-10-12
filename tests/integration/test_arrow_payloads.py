# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Single-column Arrow payloads survive fetch and process transport."""

from pathlib import Path

import numpy as np
import pytest

from zephon import Pipeline
from zephon.io import Dataset
from zephon.work import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("fmt", ["parquet", "vortex"])
@pytest.mark.parametrize("runner", ["threads", "process"])
def test_single_column_payloads_through_fetch_and_transport(
    tmp_path: Path, fmt: str, runner: str
) -> None:
    pa = pytest.importorskip("pyarrow")
    # One representative schema covers views, missing values, and SHM transport.
    values = [
        None if i == 1 else np.arange(i, i + 2048, dtype=np.int32) for i in range(8)
    ]
    array = pa.array(values, type=pa.list_(pa.int32()))
    table = pa.table({"value": array})
    root = tmp_path / "data"
    root.mkdir()
    path = root / f"data.{fmt}"
    if fmt == "parquet":
        pq = pytest.importorskip("pyarrow.parquet")
        pq.write_table(table, path, row_group_size=3)
    else:
        vortex = pytest.importorskip("vortex.io")
        vortex.write(table, str(path))
    dataset = Dataset.from_path("samples", str(root), fmt=fmt)
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
            assert record.payload is None
        else:
            assert isinstance(record.payload, np.ndarray)
            assert record.payload.dtype == np.int32
            np.testing.assert_array_equal(record.payload, expected)
            if runner == "threads":
                assert not record.payload.flags.writeable
