# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end ArrayRecord reads through the public Pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgpack
import pytest

from zephon import Pipeline
from zephon.build_index import build_index
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
    return msgpack.unpackb(payload["data"])


def _run(
    datasets: list[Dataset],
    weights: dict[str, float],
    *,
    runner: str,
    workers: int,
    cache_root: Path | None,
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
        out.append((names[int(rec.meta.sample_id[0])], int(rec.payload["value"])))
    return out


@pytest.mark.parametrize("runner", ["inline", "threads", "process"])
@pytest.mark.parametrize("cached", [False, True], ids=["direct", "cached"])
def test_reads_every_record_once(tmp_path: Path, runner: str, cached: bool) -> None:
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
