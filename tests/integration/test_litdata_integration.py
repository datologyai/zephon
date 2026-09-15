"""End-to-end LitData writer, inspector, and pipeline compatibility tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.litdata_chunks import write_litdata_fixture
from zephon import Pipeline
from zephon.debug import DatasetInspector
from zephon.io import Dataset
from zephon.work import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def _copy_row(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row)


@pytest.mark.parametrize("runner", ["threads", "process"])
@pytest.mark.parametrize("cache_enabled", [False, True])
def test_pipeline_reads_mixed_litdata_layouts(
    tmp_path: Path, runner: str, cache_enabled: bool
) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("litdata")
    datasets = []
    expected = []
    for group, layout in enumerate(["legacy", "file"]):
        rows = [{"id": group * 10 + i, "text": f"{layout}-{i}"} for i in range(6)]
        root = tmp_path / layout
        write_litdata_fixture(root, rows, layout=layout)
        datasets.append(Dataset.from_path(layout, str(root), fmt="litdata"))
        expected.extend(rows)
    work = StaticMixtureWorkSource(
        datasets,
        {dataset.name: 1.0 for dataset in datasets},
        chunk_size=3,
        shuffle_shards=False,
        shuffle_within_shard=False,
        seed=7,
    )
    pipeline = (
        Pipeline(work)
        .map_transform(_copy_row, parallelism=2)
        .batch(microbatch_size=4, drop_last=False)
        .options(
            runner=runner,
            max_workers=2,
            deterministic=True,
            io_options={
                "cache": {"enabled": cache_enabled, "root": str(tmp_path / "cache")}
            },
        )
    )
    iterator = iter(pipeline)
    try:
        observed = [record.payload for batch in iterator for record in batch.records]
    finally:
        iterator.close()
    assert sorted(observed, key=lambda row: row["id"]) == expected


@pytest.mark.parametrize("kind", ["text", "multimodal"])
@pytest.mark.parametrize("compression", [None, "zstd"])
def test_installed_writer_dictionary_rows(
    tmp_path: Path, kind: str, compression: str | None
) -> None:
    pytest.importorskip("litdata")
    pytest.importorskip("pyarrow")
    from litdata.streaming.writer import BinaryWriter

    # Cross the new writer's 256-row IPC batch boundary, using the shapes
    # emitted by Universe. Older installed writers exercise the legacy path.
    rows = []
    for index in range(300):
        row: dict[str, Any] = {"text": {"content": f"row-{index}"}}
        if kind == "multimodal":
            row.update(images=b"\x00\xffimage", metadata={"document_id": str(index)})
        rows.append(row)
    writer = BinaryWriter(
        cache_dir=str(tmp_path), chunk_size=len(rows), compression=compression
    )
    for index, row in enumerate(rows):
        writer.add_item(index, row)
    writer.done()
    writer.merge()
    with DatasetInspector(
        Dataset.from_path("writer", str(tmp_path), fmt="litdata")
    ) as reader:
        assert reader.read(0, 256) == rows[256]
        indices = [299, 0, 255, 256, 299]
        assert reader.read_many(0, indices) == [rows[index] for index in indices]


def test_arrow_read_does_not_import_pytree_decoder(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    write_litdata_fixture(tmp_path, [{"id": 0, "text": "row"}], layout="file")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from zephon.debug import DatasetInspector
from zephon.io import Dataset

with DatasetInspector(Dataset.from_path('arrow', sys.argv[1], fmt='litdata')) as reader:
    assert reader.read(0, 0) == {'id': 0, 'text': 'row'}
assert 'zephon._internal.io.formats.litdata_support.pytree' not in sys.modules
assert 'litdata' not in sys.modules
""",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
