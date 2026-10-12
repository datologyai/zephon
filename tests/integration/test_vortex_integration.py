# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Public Vortex discovery, index creation, and pipeline reads."""

import json
from pathlib import Path

import pytest

pytest.importorskip("vortex.io", reason="vortex-data not installed")
import vortex

from zephon import Pipeline
from zephon._internal.io.formats import vortex as vortex_format
from zephon.build_index import build_index
from zephon.io import Dataset
from zephon.work import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def test_vortex_pipeline_preserves_fields_and_nested_metadata(tmp_path: Path) -> None:
    """Discovery, fetch, and batching preserve complete samples across shards."""
    expected = {}
    for shard_id in range(2):
        rows = [
            {
                "text": f"sample {shard_id}/{row_id}: café",
                "meta": {
                    "language": "en",
                    "source": {"shard": shard_id, "license": "CC-BY"},
                },
            }
            for row_id in range(3)
        ]
        vortex.io.write(vortex.array(rows), str(tmp_path / f"{shard_id}.vortex"))
        expected.update({(0, shard_id, row_id): row for row_id, row in enumerate(rows)})

    dataset = Dataset.from_path("samples", str(tmp_path), fmt="vortex")
    work = StaticMixtureWorkSource(
        [dataset],
        {"samples": 1.0},
        chunk_size=2,
        seed=7,
        shuffle_shards=True,
        shuffle_within_shard=True,
    )
    pipeline = (
        Pipeline(work)
        .decode_text()
        .batch(microbatch_size=2, drop_last=False)
        .options(
            runner="threads",
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

    assert len(records) == len(expected)
    assert {record.meta.sample_id for record in records} == set(expected)
    for record in records:
        assert record.payload == expected[record.meta.sample_id]


def test_vortex_discovery_index_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index preserves discovered counts and removes the need to read footers."""
    for name, count in (("b.vortex", 3), ("a.vortex", 2)):
        vortex.io.write(
            vortex.array([{"value": i} for i in range(count)]), str(tmp_path / name)
        )

    discovered = Dataset.from_path("discovered", str(tmp_path))
    assert discovered.backend["kind"] == "vortex"
    assert discovered.shard_count() == 2
    assert discovered.total() == 5
    assert len(discovered) == 5
    target = tmp_path / "custom-index.json"
    assert build_index("vortex", tmp_path, output_path=target, progress=False) == target
    index = json.loads(target.read_text())
    assert [shard["basename"] for shard in index["shards"]] == ["a.vortex", "b.vortex"]
    assert [shard["extra"] for shard in index["shards"]] == [
        {"length": 2},
        {"length": 3},
    ]
    target.rename(tmp_path / "index.json")

    monkeypatch.setattr(vortex_format, "_vortex", None)
    indexed = Dataset.from_path("indexed", str(tmp_path), fmt="vortex")
    assert indexed.shard_count() == discovered.shard_count()
    assert indexed.total() == discovered.total()
