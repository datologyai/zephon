# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end ArrayRecord reads through the public Pipeline."""

from __future__ import annotations

import random
import runpy
from pathlib import Path
from typing import Any
from unittest import mock

import msgpack
import pytest

from zephon import Pipeline
from zephon._internal.io.formats.array_record import ArrayRecordFormat
from zephon._internal.io.index.array_record_index import ArrayRecordIndexBuilder
from zephon._internal.io.storage.local import LocalFSBackend
from zephon.build_index import build_index
from zephon.debug import DatasetInspector
from zephon.io.dataset import Dataset
from zephon.types import SampleRecord
from zephon.work.base import MixtureReadConfig, MixtureReadMode
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration

array_record_module = pytest.importorskip(
    "array_record.python.array_record_module", reason="array-record not installed"
)


def _write_dataset(root: Path, values: list[int], shards: int, *, index: bool) -> None:
    root.mkdir(parents=True)
    per = -(-len(values) // shards)
    for shard in range(shards):
        writer = array_record_module.ArrayRecordWriter(
            str(root / f"part-{shard:03d}.array_record"), "group_size:4"
        )
        for value in values[shard * per : (shard + 1) * per]:
            writer.write(msgpack.packb({"value": value}))
        writer.close()
    if index:
        build_index("array_record", root, progress=False)


def _decode(payload: Any) -> dict[str, Any]:
    return msgpack.unpackb(payload)


def _run(
    datasets: list[Dataset],
    weights: dict[str, float],
    *,
    runner: str,
    workers: int,
    cache_root: Path | None,
    fetch_max_batch: int = 64,
    exhausted_policy: str | None = None,
) -> list[tuple[str, int]]:
    work = StaticMixtureWorkSource(
        datasets,
        weights,
        chunk_size=16,
        seed=11,
        shuffle_shards=True,
        shuffle_within_shard=True,
        exhausted_policy=exhausted_policy,
    )
    pipe = (
        Pipeline(work)
        .fetch(max_batch=fetch_max_batch)
        .map_transform(_decode, parallelism=4)
        .options(
            deterministic=True,
            runner=runner,
            max_workers=workers,
            mixture_config=MixtureReadConfig(
                mode=MixtureReadMode.WEIGHTED_ROUND_ROBIN, seed=5
            ),
            io_options={
                "cache": {
                    "enabled": cache_root is not None,
                    "root": str(cache_root) if cache_root else None,
                }
            },
        )
    )
    names = {i: ds.name for i, ds in pipe.ws.datasets_by_id.items()}
    out: list[tuple[str, int]] = []
    for rec in pipe:
        assert isinstance(rec, SampleRecord)
        assert isinstance(rec.payload, dict)
        value = rec.payload["value"]
        assert isinstance(value, int)
        out.append((names[int(rec.meta.sample_id[0])], value))
    return out


@pytest.mark.parametrize(
    ("runner", "cached", "fetch_max_batch"),
    [
        pytest.param("inline", False, 1, id="inline-singleton"),
        pytest.param("threads", True, 64, id="threads-cached-bulk"),
        pytest.param("process", False, 64, id="process-bulk"),
    ],
)
def test_reads_every_record_once(
    tmp_path: Path, runner: str, cached: bool, fetch_max_batch: int
) -> None:
    values = list(range(250))
    _write_dataset(tmp_path / "ds", values, shards=4, index=False)
    dataset = Dataset.from_path("ds", str(tmp_path / "ds"))
    assert dataset.backend["kind"] == "array_record"

    out = _run(
        [dataset],
        {"ds": 1.0},
        runner=runner,
        workers=2,
        cache_root=tmp_path / "cache" if cached else None,
        fetch_max_batch=fetch_max_batch,
        # Read the dataset once, including the short final chunk.
        exhausted_policy="redistribute",
    )

    assert sorted(value for _, value in out) == values


def test_mixture_order_is_independent_of_worker_count(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "even", list(range(0, 400, 2)), shards=3, index=True)
    _write_dataset(tmp_path / "odd", list(range(1, 400, 2)), shards=5, index=False)
    datasets = [
        Dataset.from_path("even", str(tmp_path / "even")),
        Dataset.from_path("odd", str(tmp_path / "odd")),
    ]
    weights = {"even": 0.7, "odd": 0.3}

    runs = [
        _run(datasets, weights, runner="threads", workers=w, cache_root=None)
        for w in (1, 4, 8)
    ]

    assert runs[0], "pipeline produced no samples"
    assert runs[0] == runs[1] == runs[2]
    assert all((v % 2 == 0) == (name == "even") for name, v in runs[0])


def test_index_build_discovery_and_inspector(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _write_dataset(root, list(range(6)), shards=2, index=True)
    storage = mock.Mock(wraps=LocalFSBackend(root))
    storage.read_range.side_effect = AssertionError("opened shard metadata")
    assert ArrayRecordFormat().discover(".", storage)[0] == {0: 3, 1: 3}
    assert ArrayRecordFormat().discover_counts(".", storage)[1].tolist() == [3, 3]
    dataset = Dataset.from_path("ds", str(root))
    assert dataset.counts().tolist() == [3, 3]
    with DatasetInspector(dataset) as inspector:
        payload = inspector.read(shard_id=1, sample_index=2)
        assert isinstance(payload, bytes)
        assert _decode(payload) == {"value": 5}


def test_guide_writer_and_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = Path(__file__).resolve().parents[2] / "examples" / "guide" / "datasets"
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(examples / "array_record_write_shards.py"))
    runpy.run_path(str(examples / "array_record_pipeline.py"))
    assert capsys.readouterr().out.splitlines() == [
        str({"text": text})
        for text in (
            "The ocean covers most of Earth's surface.",
            "Plants use sunlight to turn water and carbon dioxide into sugars.",
            "Written language allows knowledge to be shared across generations.",
        )
    ]


def test_discovery_and_indexing_use_only_metadata_ranges(tmp_path: Path) -> None:
    path = tmp_path / "s.array_record"
    rng = random.Random(10)
    writer = array_record_module.ArrayRecordWriter(
        str(path), "group_size:1,uncompressed"
    )
    for _ in range(100):
        writer.write(rng.randbytes(8192))
    writer.close()
    storage = mock.Mock(wraps=LocalFSBackend(tmp_path))
    storage.local_path = mock.Mock(
        return_value=None
    )  # Model a range-only remote backend.
    size = path.stat().st_size
    # A complete user-data chunk lives before the metadata tail. No request may touch it.
    for action in (
        lambda: ArrayRecordFormat().discover(".", storage)[0][0],
        lambda: ArrayRecordIndexBuilder(storage).build(".", progress=False)["shards"][
            0
        ]["num_rows"],
    ):
        storage.reset_mock()
        assert action() == 100
        calls = storage.read_range.call_args_list
        assert calls and all(call.args[1] > size // 2 for call in calls)
        assert sum(call.kwargs["length"] for call in calls) <= 2 * 65536
        storage.download.assert_not_called()
        storage.open.assert_not_called()
