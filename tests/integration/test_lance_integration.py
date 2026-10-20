# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Lance pipeline reads through the runners, with and without the shard cache."""

import json
import os
import subprocess
import sys
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


# Runs in a fresh interpreter: pylance sizes its pools once per process, and
# this one has already started them.
_POOL_PROBE = """
import json, os, subprocess, sys
from pathlib import Path

from zephon._internal.io.formats.lance import LanceFormat, LanceShardOpener
from zephon._internal.io.storage import RouterStorageBackend
from zephon._internal.io.types import LocalShardFile, LocalShardRef
from zephon.io import Dataset, LanceOptions, StoreOptions


def native_threads():
    if sys.platform == "linux":
        return len(os.listdir("/proc/self/task"))
    out = subprocess.run(["ps", "-M", str(os.getpid())], capture_output=True, text=True)
    return len(out.stdout.strip().splitlines()) - 1


path, count = sys.argv[1], json.loads(sys.argv[2])
storage = RouterStorageBackend()
handler = LanceFormat()
manifest = handler.resolve_root(path, storage)
_counts, meta = handler.discover(manifest, storage)
dataset = Dataset(name="ds", backend={"kind": "lance", "path": manifest, "shards": meta})
locator = handler.build_locators(dataset)[0]
data = Path(path) / "data" / locator.raw.basename
ref = LocalShardRef(
    raw=LocalShardFile(path=data, bytes=locator.raw.bytes), extra=locator.extra
)
opener = LanceShardOpener(
    StoreOptions(lance=LanceOptions(worker_threads=count)), storage
)
opener.open_shard(locator, ref).getsamples([0, 1])
print(json.dumps({
    "threads": native_threads(),
    "env": {name: os.environ.get(name) for name in ("TOKIO_WORKER_THREADS", "LANCE_CPU_THREADS")},
}))
"""


def _pool_probe(path: Path, count: int | None) -> dict[str, Any]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("TOKIO_WORKER_THREADS", "LANCE_CPU_THREADS")
    }
    result = subprocess.run(
        [sys.executable, "-c", _POOL_PROBE, str(path), json.dumps(count)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.skipif(
    sys.platform not in ("linux", "darwin"), reason="counts native threads"
)
def test_worker_threads_caps_pylance_pools_without_leaking_the_environment(
    tmp_path: Path, pa: Any
) -> None:
    """The cap relies on a failed open starting pylance's pools; a pylance that
    stops doing so would leave them silently uncapped."""
    path = tmp_path / "ds.lance"
    _write(path, pa)

    one = _pool_probe(path, 1)
    four = _pool_probe(path, 4)

    assert one["env"] == {"TOKIO_WORKER_THREADS": None, "LANCE_CPU_THREADS": "1"}
    # Two explicit sizes rather than pylance's default, whose size follows the
    # CPUs available to the process (a container quota), not os.cpu_count().
    assert four["threads"] - one["threads"] >= 3
