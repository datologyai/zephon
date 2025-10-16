# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from zephon.api import Pipeline as PublicPipeline
from zephon.core.constants import EngineSample
from zephon.core.op_base import OpContext
from zephon.io import Dataset
from zephon.ops.fetch import FetchOp
from zephon.work import MixtureSpec, StaticMixtureWorkSource


@pytest.fixture()
def jsonl_dataset(tmp_path: Path) -> Dataset:
    shard0 = tmp_path / "shard0.jsonl"
    shard1 = tmp_path / "shard1.jsonl"
    shard0.write_text(
        "\n".join(json.dumps({"text": f"sample {i}", "value": i}) for i in range(3)),
        encoding="utf-8",
    )
    shard1.write_text(
        "\n".join(json.dumps({"text": f"sample {i}", "value": i}) for i in range(3, 5)),
        encoding="utf-8",
    )
    return Dataset.from_path("demo", str(tmp_path))


def test_dataset_from_path_detects_jsonl(jsonl_dataset: Dataset) -> None:
    assert jsonl_dataset.backend["kind"] == "jsonl"
    assert set(jsonl_dataset.shard_index.keys()) == {0, 1}
    assert sum(jsonl_dataset.shard_index.values()) == 5


def make_engine_sample(
    dataset_id: int, shard_id: int, sample_idx: int, lane_id: int = 0, chunk_id: int = 0
) -> EngineSample:
    """Helper to build EngineSample tuples for direct FetchOp invocation in tests."""
    return ((dataset_id, shard_id, sample_idx), lane_id, chunk_id)


def test_fetch_op_reads_jsonl(jsonl_dataset: Dataset) -> None:
    ctx = OpContext({"datasets_by_id": {0: jsonl_dataset}})
    op = FetchOp()
    op.setup(ctx)
    record = op.process_one(make_engine_sample(0, 0, 1))[0]
    assert record.payload["value"] == 1
    batch = op.process_many(
        [
            make_engine_sample(0, 1, 0, lane_id=0, chunk_id=0),
            make_engine_sample(0, 1, 1, lane_id=0, chunk_id=0),
        ]
    )
    assert [elem.payload["value"] for elem in batch] == [3, 4]


def test_fetch_op_reads_with_cache(jsonl_dataset: Dataset, tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    ctx = OpContext(
        {
            "datasets_by_id": {0: jsonl_dataset},
            "io_options": {"cache": {"enabled": True, "root": cache_root}},
        }
    )
    op = FetchOp()
    op.setup(ctx)
    _ = op.process_many(
        [
            make_engine_sample(0, 0, 0, lane_id=0, chunk_id=0),
            make_engine_sample(0, 0, 1, lane_id=0, chunk_id=0),
        ]
    )
    assert cache_root.exists()


def test_pipeline_with_cache(tmp_path: Path, jsonl_dataset: Dataset) -> None:
    cache_root = tmp_path / "cache"
    work_source = StaticMixtureWorkSource(
        [jsonl_dataset],
        mixture=MixtureSpec({jsonl_dataset.name: 1.0}).weights,
        chunk_size=1,
        seed=11,
        shuffle_shards=False,
    )
    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .options(io_options={"cache": {"enabled": True, "root": cache_root}})
        .batch(microbatch_size=2, drop_last=False)
    )
    iterator = iter(pipe)
    try:
        next(iterator)
    finally:
        iterator.close()
    assert cache_root.exists()
