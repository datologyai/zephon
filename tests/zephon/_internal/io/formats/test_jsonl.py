import bz2
import gzip
import json
import lzma
from pathlib import Path

import pytest

import zephon._internal.io.resolvers.cache.manager as manager_mod
import zephon._internal.io.resolvers.direct as direct_mod
from tests._helpers import counts_dict
from zephon._internal.io.formats.jsonl import JsonlFormat, JsonlShard
from zephon._internal.io.index.jsonl_index import JsonlIndexBuilder
from zephon._internal.io.storage.local import LocalFSBackend
from zephon._internal.io.stores.multi import build_multi_dataset_store
from zephon._internal.io.types import LocalShardFile, LocalShardRef
from zephon._internal.utils.compression import decompress_file, zstd
from zephon.io.dataset import Dataset
from zephon.io.options import CacheOptions, StoreOptions

_WRITERS = {"gzip": gzip.open, "bz2": bz2.open, "xz": lzma.open, "zstd": zstd.open}
_SUFFIXES = {"gzip": ".gz", "bz2": ".bz2", "xz": ".xz", "zstd": ".zst"}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _write_compressed_jsonl(
    path: Path, rows: list[dict[str, object]], compression: str, *, blank: bool = False
) -> None:
    """Write ``rows`` as two streamed frames/members, optionally with a blank line."""
    lines = [json.dumps(r) + "\n" for r in rows]
    if blank:
        lines.insert(len(lines) // 2, "\n")
    half = len(lines) // 2
    with _WRITERS[compression](path, "wt", encoding="utf-8") as fh:
        fh.write("".join(lines[:half]))
    with _WRITERS[compression](path, "at", encoding="utf-8") as fh:
        fh.write("".join(lines[half:]))


def test_jsonl_create_index(tmp_path: Path) -> None:
    """The builder writes sorted shard metadata and ignores blank lines."""
    (tmp_path / "b.jsonl").write_text('{"i": 1}\n\n  \n{"i": 2}\n', encoding="utf-8")
    _write_jsonl(tmp_path / "a.jsonl", [{"i": 0}])

    result = JsonlIndexBuilder().create_index(tmp_path, progress=False)
    data = json.loads(result.read_text(encoding="utf-8"))

    assert data["format_version"] == 1
    assert [shard["basename"] for shard in data["shards"]] == [
        "a.jsonl",
        "b.jsonl",
    ]
    assert [shard["num_rows"] for shard in data["shards"]] == [1, 2]
    assert [shard["extra"] for shard in data["shards"]] == [
        {"length": 1},
        {"length": 2},
    ]
    assert data["shards"][1]["bytes"] == (tmp_path / "b.jsonl").stat().st_size


def test_jsonl_create_index_rejects_empty_directory(tmp_path: Path) -> None:
    """Building an index requires at least one JSONL shard."""
    with pytest.raises(ValueError, match=r"No \*\.jsonl files found"):
        JsonlIndexBuilder().create_index(tmp_path, progress=False)


def test_jsonl_discover_local(tmp_path: Path) -> None:
    shard0 = tmp_path / "a.jsonl"
    shard1 = tmp_path / "b.jsonl"
    _write_jsonl(shard0, [{"i": i} for i in range(2)])
    _write_jsonl(shard1, [{"i": i} for i in range(3)])

    handler = JsonlFormat()
    shard_index, shard_meta = handler.discover(str(tmp_path), LocalFSBackend(tmp_path))

    # Assert counts sum and per-shard entries present
    assert sum(int(v) for v in shard_index.values()) == 5
    assert len(shard_meta) == len(shard_index)
    # Ensure basic metadata captured
    for meta in shard_meta.values():
        assert isinstance(meta.get("raw"), dict)
        assert isinstance(meta.get("extra"), dict)


def test_jsonl_discover_assigns_shard_ids_in_sorted_basename_order(
    tmp_path: Path,
) -> None:
    """Shard ids follow sorted basenames regardless of directory order."""
    # Created out of lexical order on purpose; distinct row counts identify files.
    _write_jsonl(tmp_path / "c.jsonl", [{"i": i} for i in range(1)])
    _write_jsonl(tmp_path / "a.jsonl", [{"i": i} for i in range(2)])
    _write_jsonl(tmp_path / "b.jsonl", [{"i": i} for i in range(3)])

    handler = JsonlFormat()
    shard_index, shard_meta = handler.discover(str(tmp_path), LocalFSBackend(tmp_path))

    basenames = [shard_meta[sid]["raw"]["basename"] for sid in sorted(shard_meta)]
    assert basenames == ["a.jsonl", "b.jsonl", "c.jsonl"]
    assert dict(shard_index) == {0: 2, 1: 3, 2: 1}


def test_jsonl_discover_and_counts_use_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Indexed discovery obtains counts without scanning shard contents again."""
    _write_jsonl(tmp_path / "a.jsonl", [{"i": 0}, {"i": 1}])
    _write_jsonl(tmp_path / "b.jsonl", [{"i": 2}])
    JsonlIndexBuilder().create_index(tmp_path, progress=False)

    handler = JsonlFormat()
    monkeypatch.setattr(
        handler,
        "_discover_from_files",
        lambda *_args: pytest.fail("indexed discovery scanned JSONL shards"),
    )
    storage = LocalFSBackend(tmp_path)

    shard_index, shard_meta = handler.discover(str(tmp_path), storage)
    ids, counts = handler.discover_counts(str(tmp_path), storage)

    assert dict(shard_index) == {0: 2, 1: 1}
    assert ids.tolist() == [0, 1]
    assert counts.tolist() == [2, 1]
    assert shard_meta[0]["raw"]["basename"] == "a.jsonl"
    assert shard_meta[0]["extra"] == {"length": 2}


def test_jsonl_discover_ignores_non_jsonl_shard_index(tmp_path: Path) -> None:
    """An index for another format does not replace JSONL file discovery."""
    _write_jsonl(tmp_path / "data.jsonl", [{"i": 0}, {"i": 1}])
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "shards": [
                    {
                        "basename": "data.parquet",
                        "bytes": 1,
                        "num_rows": 99,
                        "hashes": {},
                        "extra": {"row_groups": []},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    shard_index, _ = JsonlFormat().discover(str(tmp_path), LocalFSBackend(tmp_path))

    assert dict(shard_index) == {0: 2}


def test_jsonl_auto_detects_from_index(tmp_path: Path) -> None:
    """A Zephon shard index with JSONL basenames identifies the format."""
    _write_jsonl(tmp_path / "data.jsonl", [{"i": 0}, {"i": 1}])
    JsonlIndexBuilder().create_index(tmp_path, progress=False)

    dataset = Dataset.from_path("indexed", str(tmp_path))

    assert dataset.backend["kind"] == "jsonl"
    assert dataset.ids().tolist() == [0]
    assert dataset.counts().tolist() == [2]


def test_jsonl_build_locators_and_open(tmp_path: Path) -> None:
    shard0 = tmp_path / "shard0.jsonl"
    _write_jsonl(shard0, [{"x": 1}, {"x": 2}])

    handler = JsonlFormat()
    shard_index, shard_meta = handler.discover(str(tmp_path), LocalFSBackend(tmp_path))

    ds = Dataset(
        name="demo",
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


@pytest.mark.parametrize("compression", sorted(_WRITERS))
def test_jsonl_discover_compressed_shard(tmp_path: Path, compression: str) -> None:
    name = f"a.jsonl{_SUFFIXES[compression]}"
    _write_compressed_jsonl(
        tmp_path / name, [{"i": i} for i in range(5)], compression, blank=True
    )

    shard_index, shard_meta = JsonlFormat().discover(
        str(tmp_path), LocalFSBackend(tmp_path)
    )

    with _WRITERS[compression](tmp_path / name, "rb") as fh:
        decoded = fh.read()
    assert dict(shard_index) == {0: 5}  # the blank line is not a record
    # zip = the compressed file; raw = its decoded copy, as for MDS/LitData.
    assert shard_meta[0]["zip"] == {
        "basename": name,
        "bytes": (tmp_path / name).stat().st_size,
        "hashes": {},
    }
    assert shard_meta[0]["raw"] == {
        "basename": f"{name}.raw",
        "bytes": len(decoded),
        "hashes": {},
    }
    assert shard_meta[0]["compression"] == compression
    assert shard_meta[0]["extra"] == {"length": 5}


def test_jsonl_discover_mixes_plain_and_compressed_shards(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "b.jsonl", [{"i": i} for i in range(2)])
    _write_compressed_jsonl(
        tmp_path / "a.jsonl.zst", [{"i": i} for i in range(3)], "zstd"
    )
    (tmp_path / "c.json").write_text('{"i": 0}\n', encoding="utf-8")  # not JSONL

    shard_index, shard_meta = JsonlFormat().discover(
        str(tmp_path), LocalFSBackend(tmp_path)
    )

    assert [shard_meta[sid]["raw"]["basename"] for sid in sorted(shard_meta)] == [
        "a.jsonl.zst.raw",
        "b.jsonl",
    ]
    assert dict(shard_index) == {0: 3, 1: 2}
    assert "zip" not in shard_meta[1]


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_compressed_jsonl_reads_through_store(
    tmp_path: Path, cache_enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    rows = [{"i": i} for i in range(6)]
    _write_compressed_jsonl(data / "s0.jsonl.zst", rows[:4], "zstd")
    _write_compressed_jsonl(data / "s1.jsonl.gz", rows[4:], "gzip")

    decoded_shards: list[str] = []

    def _counting_decompress(src: Path, dst: Path, compression: str) -> None:
        decoded_shards.append(src.name.split(".")[0])
        decompress_file(src, dst, compression)

    for resolver_mod in (manager_mod, direct_mod):
        monkeypatch.setattr(resolver_mod, "decompress_file", _counting_decompress)

    ds = Dataset.from_path("demo", str(data))
    cache_root = tmp_path / "cache"
    store = build_multi_dataset_store(
        {0: ds},
        options=StoreOptions(
            cache=CacheOptions(enabled=cache_enabled, root=cache_root)
        ),
    )
    try:
        view = store.for_dataset(0)
        # The store wraps shards in ResilientShard, which returns (rows, stats).
        shard0, _ = view.open(0)
        assert len(shard0) == 4
        assert shard0.getsamples([3, 0, 3])[0] == [rows[3], rows[0], rows[3]]
        assert shard0[1][0] == rows[1]
        shard1, _ = view.open(1)
        assert shard1.getsamples([1])[0] == [rows[5]]
    finally:
        store.close()

    # Each shard is decoded once, however often it is read, into its raw file:
    # in the cache when enabled, otherwise next to the data (as for MDS/LitData).
    assert sorted(decoded_shards) == ["s0", "s1"]
    decoded_dir = cache_root / "demo" if cache_enabled else data
    decoded = sorted(p.name for p in decoded_dir.iterdir() if p.suffix == ".raw")
    assert decoded == ["s0.jsonl.zst.raw", "s1.jsonl.gz.raw"]
    assert (decoded_dir / "s0.jsonl.zst.raw").read_text(encoding="utf-8") == "".join(
        json.dumps(r) + "\n" for r in rows[:4]
    )
    # Decoded copies are not shards: rediscovery sees the same dataset.
    assert counts_dict(Dataset.from_path("demo", str(data))) == {0: 4, 1: 2}
