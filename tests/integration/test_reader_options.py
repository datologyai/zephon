# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Reader options reach real fetch workers on both local runners."""

from pathlib import Path

import pytest

from tests.helpers.litdata_chunks import write_litdata_fixture
from zephon import Pipeline
from zephon._internal import pipeline as pipeline_module
from zephon._internal.ops.fetch import FetchOp
from zephon._internal.stream import EngineSample
from zephon.io import ArrowOptions, Dataset, ParquetOptions, StoreOptions, VortexOptions
from zephon.types import SampleRecord
from zephon.work import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("runner", ["threads", "process"])
@pytest.mark.parametrize("format_name", ["parquet", "vortex", "litdata"])
def test_reader_options_inside_fetch_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: str, format_name: str
) -> None:
    pa = pytest.importorskip("pyarrow")
    rows = [{"text": f"sample {i}"} for i in range(12)]
    previous_arrow = (pa.cpu_count(), pa.io_thread_count())
    vortex = None
    previous_vortex = None
    if format_name == "parquet":
        pq = pytest.importorskip("pyarrow.parquet")
        pq.write_table(
            pa.Table.from_pylist(rows), tmp_path / "data.parquet", row_group_size=4
        )
    elif format_name == "litdata":
        pytest.importorskip("optree")
        write_litdata_fixture(tmp_path, rows, layout="file")
    else:
        pytest.importorskip("vortex.io")
        vortex = pytest.importorskip("vortex")

        previous_vortex = vortex.worker_threads()
        vortex.io.write(vortex.array(rows), str(tmp_path / "data.vortex"))

    class InspectFetchOp(FetchOp):
        def process_many(self, elems: list[EngineSample]) -> list[SampleRecord]:
            records = super().process_many(elems)
            import pyarrow as arrow

            assert arrow.cpu_count() == 2
            assert arrow.io_thread_count() == 3
            if format_name == "vortex":
                import vortex

                assert vortex.worker_threads() == 2
            return records

    monkeypatch.setattr(pipeline_module, "FetchOp", InspectFetchOp)
    dataset = Dataset.from_path("samples", str(tmp_path), fmt=format_name)
    work = StaticMixtureWorkSource(
        [dataset],
        {"samples": 1.0},
        chunk_size=2,
        seed=7,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = (
        Pipeline(work)
        .fetch_parallelism(2, max_batch=2)
        .batch(microbatch_size=2, drop_last=False)
        .options(
            runner=runner,
            max_workers=2,
            deterministic=True,
            io_options=StoreOptions(
                arrow=ArrowOptions(cpu_threads=2, io_threads=3),
                parquet=ParquetOptions(use_threads=True),
                vortex=VortexOptions(worker_threads=2),
            ),
        )
    )
    iterator = iter(pipeline)
    try:
        records = [record for batch in iterator for record in batch.records]
        assert sorted(record.payload["text"] for record in records) == sorted(
            row["text"] for row in rows
        )
    finally:
        iterator.close()
        pa.set_cpu_count(previous_arrow[0])
        pa.set_io_thread_count(previous_arrow[1])
        if vortex is not None:
            vortex.set_worker_threads(previous_vortex)
