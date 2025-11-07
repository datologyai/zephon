import json
from pathlib import Path

import pytest

from zephon.io.dataset import Dataset
from zephon.io.formats.jsonl import JsonlFormat, JsonlShard
from zephon.io.storage.local import LocalFSBackend
from zephon.io.types import LocalShardFile, LocalShardRef


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_jsonl_discover_local(tmp_path: Path) -> None:
    shard0 = tmp_path / "a.jsonl"
    shard1 = tmp_path / "b.jsonl"
    _write_jsonl(shard0, [{"i": i} for i in range(2)])
    _write_jsonl(shard1, [{"i": i} for i in range(3)])

    handler = JsonlFormat()
    shard_index, shard_meta = handler.discover(str(tmp_path), LocalFSBackend(tmp_path))

    # Enumerate in storage.listdir order; just assert counts sum and per-shard entries present
    assert sum(int(v) for v in shard_index.values()) == 5
    assert len(shard_meta) == len(shard_index)
    # Ensure basic metadata captured
    for meta in shard_meta.values():
        assert isinstance(meta.get("raw"), dict)
        assert isinstance(meta.get("extra"), dict)


def test_jsonl_build_locators_and_open(tmp_path: Path) -> None:
    shard0 = tmp_path / "shard0.jsonl"
    _write_jsonl(shard0, [{"x": 1}, {"x": 2}])

    handler = JsonlFormat()
    shard_index, shard_meta = handler.discover(str(tmp_path), LocalFSBackend(tmp_path))

    ds = Dataset(
        name="demo",
        shard_index=shard_index,
        backend={"kind": "jsonl", "path": str(tmp_path), "shards": shard_meta},
        path=str(tmp_path),
    )

    locators = handler.build_locators(ds)
    assert set(locators.keys()) == set(shard_index.keys())
    loc = next(iter(locators.values()))

    # Open without extra length: __len__ must scan and then cache
    ref = LocalShardRef(raw=LocalShardFile(path=shard0, bytes=shard0.stat().st_size))
    shard = handler.open_shard(loc, ref)
    assert len(shard) == 2
    # __getitem__ returns parsed JSON
    assert shard[0] == {"x": 1}
    assert shard[1] == {"x": 2}
    with pytest.raises(IndexError):
        _ = shard[-1]
    with pytest.raises(IndexError):
        _ = shard[2]
    shard.close()

    # Open with extra length hint
    ref2 = LocalShardRef(
        raw=LocalShardFile(path=shard0, bytes=shard0.stat().st_size),
        extra={"length": 2},
    )
    shard2 = handler.open_shard(loc, ref2)
    assert len(shard2) == 2
    shard2.close()


@pytest.mark.parametrize(
    "bad_shards",
    [
        # Missing raw
        {0: {}},
        # Missing basename
        {0: {"raw": {"bytes": 1}}},
        # Missing bytes
        {0: {"raw": {"basename": "shard0.jsonl"}}},
        # Invalid bytes type
        {0: {"raw": {"basename": "shard0.jsonl", "bytes": object()}}},
    ],
)
def test_jsonl_build_locators_rejects_bad_metadata(tmp_path: Path, bad_shards) -> None:
    ds = Dataset(
        name="bad",
        shard_index={0: 1},
        backend={"kind": "jsonl", "path": str(tmp_path), "shards": bad_shards},
        path=str(tmp_path),
    )
    handler = JsonlFormat()
    with pytest.raises(ValueError):
        _ = handler.build_locators(ds)


def test_jsonl_shard_getsamples_unsorted_and_duplicates(tmp_path: Path) -> None:
    p = tmp_path / "shard.jsonl"
    rows = [{"v": i} for i in range(6)]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    shard = JsonlShard(p)

    out = shard.getsamples([4, 1, 4, 0])
    assert [r["v"] for r in out] == [4, 1, 4, 0]

    assert shard.getsamples([]) == []

    with pytest.raises(IndexError):
        _ = shard.getsamples([10])
