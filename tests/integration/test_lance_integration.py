# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Lance pipeline reads through the runners, with and without the shard cache."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("lance", reason="pylance not installed")
import lance  # noqa: E402

from zephon import Pipeline  # noqa: E402
from zephon._internal.io.stores.multi import build_multi_dataset_store  # noqa: E402
from zephon.io import CacheOptions, Dataset, StoreOptions  # noqa: E402
from zephon.work import StaticMixtureWorkSource  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture
def pa() -> Any:
    return pytest.importorskip("pyarrow")


_ROWS = 12


def _write(path: Path, pa: Any) -> None:
    """Write _ROWS rows over three fragments, so shard boundaries are exercised."""
    lance.write_dataset(
        pa.table(
            {
                "id": pa.array(range(_ROWS), pa.int64()),
                "vec": pa.array(
                    [[float(i), float(i + 1)] for i in range(_ROWS)],
                    pa.list_(pa.float32(), 2),
                ),
            }
        ),
        str(path),
        max_rows_per_file=4,
    )


def _run(path: Path, runner: str, cache_root: Path | None) -> list[Any]:
    dataset = Dataset.from_path("ds", str(path))
    work = StaticMixtureWorkSource(
        [dataset],
        {"ds": 1.0},
        chunk_size=2,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    io_options = (
        {"cache": {"enabled": True, "root": str(cache_root)}}
        if cache_root is not None
        else {"cache": {"enabled": False}}
    )
    pipeline = (
        Pipeline(work)
        .batch(microbatch_size=3, drop_last=False)
        .options(runner=runner, max_workers=2, io_options=io_options)
    )
    iterator = iter(pipeline)
    try:
        return [record for batch in iterator for record in batch.records]
    finally:
        iterator.close()


@pytest.mark.parametrize("runner", ["inline", "threads", "process"])
@pytest.mark.parametrize("cached", [False, True])
def test_pipeline_reads_every_row_once(
    tmp_path: Path, pa: Any, runner: str, cached: bool
) -> None:
    path = tmp_path / "ds.lance"
    _write(path, pa)

    records = _run(path, runner, tmp_path / "cache" if cached else None)

    assert sorted(record.payload["id"] for record in records) == list(range(_ROWS))
    for record in records:
        vector = record.payload["vec"]
        assert isinstance(vector, np.ndarray)
        identifier = int(record.payload["id"])
        assert vector.tolist() == [float(identifier), float(identifier + 1)]


@pytest.mark.parametrize("runner", ["threads", "process"])
def test_deleted_rows_never_reach_the_pipeline(
    tmp_path: Path, pa: Any, runner: str
) -> None:
    path = tmp_path / "ds.lance"
    _write(path, pa)
    lance.dataset(str(path)).delete("id % 3 = 0")

    records = _run(path, runner, tmp_path / "cache")

    assert sorted(record.payload["id"] for record in records) == [
        i for i in range(_ROWS) if i % 3
    ]


def test_cached_shards_land_under_the_data_root(tmp_path: Path, pa: Any) -> None:
    """The pinned manifest keys the catalog; shards still resolve under ``data/``."""
    path = tmp_path / "ds.lance"
    _write(path, pa)
    cache_root = tmp_path / "cache"
    dataset = Dataset.from_path("ds", str(path))

    store = build_multi_dataset_store(
        {0: dataset},
        options=StoreOptions(cache=CacheOptions(enabled=True, root=cache_root)),
    )
    try:
        shard, _ = store.for_dataset(0).open(0)
        rows = shard.getsamples([2, 0, 2])
    finally:
        store.close()

    payloads = rows[0] if isinstance(rows, tuple) else rows
    assert [int(row["id"]) for row in payloads] == [2, 0, 2]
    cached = sorted(p.name for p in (cache_root / "ds").iterdir() if p.is_file())
    basenames = [
        f.data_files()[0].path for f in lance.dataset(str(path)).get_fragments()
    ]
    basenames = [name() if callable(name) else name for name in basenames]
    assert cached == [basenames[0]]
