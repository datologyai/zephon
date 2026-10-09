# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""All index builders use the same storage and concurrency infrastructure."""

import bz2
import gzip
import importlib
import io
import json
import lzma
import random
from collections.abc import Iterator
from pathlib import Path
from threading import Event, Lock
from typing import IO, Any

import pytest

from zephon._internal.io.formats.base import get_format
from zephon._internal.io.index.index_builder import IndexBuilder, ShardInfo
from zephon._internal.io.index.jsonl_index import JsonlIndexBuilder
from zephon._internal.io.storage import StorageBackend, router
from zephon._internal.utils.compression import require_zstd
from zephon.build_index import build_index


class _MemoryStorage(StorageBackend):
    """An authenticated backend stand-in; no native format reader can access it."""

    def __init__(self, root: str, files: dict[str, bytes]) -> None:
        self.root = root
        self.files = {f"{root}/{name}": data for name, data in files.items()}
        self.read_bytes = 0
        self.opened: list[str] = []
        self.deny_shard_reads = False

    def walk(self, path: str) -> Iterator[tuple[str, int]]:
        prefix = path.rstrip("/") + "/"
        for name, data in reversed(self.files.items()):
            if name.startswith(prefix):
                yield name[len(prefix) :], len(data)

    def exists(self, path: str) -> bool:
        return path in self.files

    def listdir(self, path: str) -> list[str]:
        return [name for name, _ in self.walk(path) if "/" not in name]

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> IO[bytes] | IO[str]:
        if not path.endswith(".json"):
            assert not self.deny_shard_reads
            assert ".jsonl" in path, "Columnar index builders must use range reads"
        self.opened.append(path)
        data = self.files[path]
        return io.BytesIO(data) if "b" in mode else io.StringIO(data.decode("utf-8"))

    def read_range(
        self,
        path: str,
        start: int,
        *,
        end: int | None = None,
        length: int | None = None,
    ) -> bytes:
        assert not self.deny_shard_reads
        assert length is not None and end is None
        data = self.files[path][start : start + length]
        self.read_bytes += len(data)
        return data

    def put(self, path: str, data: bytes) -> None:
        self.files[path] = data


@pytest.fixture(params=["jsonl", "parquet", "vortex"])
def dataset_files(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[str, dict[str, bytes]]:
    """Real files for each registered index format, with known record counts."""
    fmt = request.param
    files: dict[str, bytes] = {}
    rng = random.Random(0)
    for name, count in (("a", 2), ("b", 3)):
        path = tmp_path / f"{name}.{fmt}"
        if fmt == "jsonl":
            path.write_text(
                "\n".join(json.dumps({"value": i}) for i in range(count)),
                encoding="utf-8",
            )
        else:
            rows = [{"payload": rng.randbytes(40 * 1024)} for _ in range(count)]
            if fmt == "parquet":
                pa = pytest.importorskip("pyarrow")
                pq = pytest.importorskip("pyarrow.parquet")
                pq.write_table(pa.Table.from_pylist(rows), path)
            else:
                vx = pytest.importorskip("vortex")
                vx.io.write(vx.array(rows), str(path))
        files[path.name] = path.read_bytes()
    return fmt, files


@pytest.mark.parametrize("scheme", ["s3", "gs"])
def test_remote_index_roundtrip_all_formats(
    dataset_files: tuple[str, dict[str, bytes]],
    scheme: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public API writes remote indexes that discovery can use without shard IO."""
    fmt, files = dataset_files
    root = f"{scheme}://bucket/data"
    storage = _MemoryStorage(root, {**files, f"nested/ignored.{fmt}": b"invalid"})
    factory = "_make_s3_backend" if scheme == "s3" else "_make_gcs_backend"
    monkeypatch.setattr(router, factory, lambda: storage)
    target = build_index(fmt, root, progress=False, max_workers=2)
    assert target == f"{root}/index.json"
    index = json.loads(storage.files[target])
    assert [shard["basename"] for shard in index["shards"]] == sorted(files)
    assert [shard["num_rows"] for shard in index["shards"]] == [2, 3]
    if fmt != "jsonl":
        assert 0 < storage.read_bytes < sum(map(len, files.values()))
        assert storage.opened == []

    storage.deny_shard_reads = True
    importlib.import_module(f"zephon._internal.io.formats.{fmt}")
    counts, _ = get_format(fmt).discover(root, storage)
    assert dict(counts) == {0: 2, 1: 3}


@pytest.mark.parametrize("remote_source", [True, False])
def test_index_output_can_use_a_different_backend(
    dataset_files: tuple[str, dict[str, bytes]],
    remote_source: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both remote-to-local and local-to-remote index output go through the router."""
    fmt, files = dataset_files
    storage = _MemoryStorage("s3://bucket/data", files)
    monkeypatch.setattr(router, "_make_s3_backend", lambda: storage)
    source = storage.root if remote_source else tmp_path
    target = tmp_path / "output.json" if remote_source else "s3://bucket/index.json"
    assert build_index(fmt, source, output_path=target, progress=False) == target
    data = target.read_bytes() if isinstance(target, Path) else storage.files[target]
    assert [shard["num_rows"] for shard in json.loads(data)["shards"]] == [2, 3]


@pytest.mark.parametrize("suffix", ["gz", "bz2", "xz", "zst"])
def test_remote_compressed_jsonl_records_raw_size(suffix: str) -> None:
    """Compression metadata remains valid when the input is a backend-owned stream."""
    raw = '{"text":"café"}\n\nnull'.encode("utf-8")
    compress = {
        "gz": gzip.compress,
        "bz2": bz2.compress,
        "xz": lzma.compress,
        "zst": require_zstd().compress,
    }[suffix]
    storage = _MemoryStorage(
        "s3://bucket/data", {f"data.jsonl.{suffix}": compress(raw)}
    )
    index = JsonlIndexBuilder(storage).build(storage.root, progress=False)
    assert index["shards"][0]["num_rows"] == 2
    assert index["shards"][0]["extra"] == {"raw_bytes": len(raw)}


def test_failed_read_preserves_existing_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """An error on any shard prevents publishing a partial replacement index."""
    storage = _MemoryStorage(
        "s3://bucket/data",
        {"a.jsonl": b"{}", "b.jsonl": b"{}", "index.json": b"existing index"},
    )
    original_open = storage.open

    def open_with_failure(
        path: str, mode: str = "rb", **kwargs: Any
    ) -> IO[bytes] | IO[str]:
        if path.endswith("b.jsonl"):
            raise PermissionError("backend denied access")
        return original_open(path, mode, **kwargs)

    monkeypatch.setattr(storage, "open", open_with_failure)
    with pytest.raises(
        ValueError, match="Failed to index shard s3://bucket/data/b.jsonl"
    ):
        JsonlIndexBuilder(storage).create_index(storage.root, progress=False)
    assert storage.files[f"{storage.root}/index.json"] == b"existing index"


def test_parallel_indexing_is_bounded_and_keeps_progress_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Slow reads keep stable IDs and progress while respecting the configured bound."""
    storage = _MemoryStorage(
        "s3://bucket/data", {f"{name}.jsonl": b"{}" for name in "abcdef"}
    )
    second_finished = Event()
    lock = Lock()
    active = peak = 0

    class OrderedBuilder(IndexBuilder):
        suffixes = (".jsonl",)

        def extract_shard_info(self, path: str, file_size: int) -> ShardInfo:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if path.endswith("a.jsonl"):
                    assert second_finished.wait(timeout=5)
                if path.endswith("b.jsonl"):
                    second_finished.set()
                return ShardInfo(path.rsplit("/", 1)[-1], file_size, 1)
            finally:
                with lock:
                    active -= 1

    index = OrderedBuilder(storage, max_workers=2).build(
        storage.root, progress_interval=2
    )
    assert peak == 2
    assert [shard["basename"] for shard in index["shards"]] == [
        f"{name}.jsonl" for name in "abcdef"
    ]
    output = capsys.readouterr().out
    assert "Processed 2/6 files" in output
    assert "Processed 4/6 files" in output
    assert "Processed 6/6 files" in output


def test_sequential_indexing_and_invalid_limits() -> None:
    """Callers can serialize content scans, and invalid limits fail before IO."""
    storage = _MemoryStorage("s3://bucket/data", {"b.jsonl": b"{}", "a.jsonl": b"{}"})
    builder = JsonlIndexBuilder(storage, max_workers=1)
    builder.build(storage.root, progress=False)
    assert storage.opened == [f"{storage.root}/a.jsonl", f"{storage.root}/b.jsonl"]
    with pytest.raises(ValueError, match="max_workers"):
        JsonlIndexBuilder(storage, max_workers=0)
    with pytest.raises(ValueError, match="progress_interval"):
        builder.build(storage.root, progress_interval=0)
