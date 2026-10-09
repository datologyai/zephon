# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex discovery must use the supplied storage backend for metadata IO."""

import io
import json
import random
from pathlib import Path
from threading import Event
from typing import IO, Any

import pytest

vortex = pytest.importorskip("vortex", reason="vortex-data not installed")

from zephon._internal.io.formats import vortex as vortex_format
from zephon._internal.io.formats.vortex import VortexFormat
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.storage.local import LocalFSBackend
from zephon.build_index import build_index


class _RangeOnlyStorage(StorageBackend):
    """Serve shard ranges and control files, rejecting whole-shard access."""

    def __init__(self, root: str, files: dict[str, bytes]) -> None:
        self.root = root
        self.files = {f"{root}/{name}": data for name, data in files.items()}
        self.reads: list[tuple[str, int, int]] = []
        self.max_read: int | None = None

    def exists(self, path: str) -> bool:
        return path in self.files

    def listdir(self, path: str) -> list[str]:
        assert path == self.root
        return [key.rsplit("/", 1)[-1] for key in reversed(self.files)]

    def stat(self, path: str) -> dict[str, int]:
        return {"size": len(self.files[path])}

    def read_range(
        self,
        path: str,
        start: int,
        *,
        end: int | None = None,
        length: int | None = None,
    ) -> bytes | memoryview:
        assert length is not None and end is None
        if self.max_read is not None:
            length = min(length, self.max_read)
        self.reads.append((path, start, length))
        return memoryview(self.files[path])[start : start + length]

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> IO[bytes] | IO[str]:
        assert path.endswith(".json"), "Discovery must not open a whole shard"
        return io.StringIO(self.files[path].decode("utf-8"))

    def download(self, src: str, dst: str, timeout: float | None = None) -> None:
        raise AssertionError("Discovery must not download shards")

    def put(self, path: str, data: bytes) -> None:
        self.files[path] = data


@pytest.fixture
def shard_files(tmp_path: Path) -> dict[str, bytes]:
    """Small real files with payloads larger than Vortex's footer prefetch."""
    rng = random.Random(0)
    files = {}
    for name, count in (("a.vortex", 2), ("b.vortex", 3)):
        path = tmp_path / name
        rows = [{"payload": rng.randbytes(32 * 1024)} for _ in range(count)]
        vortex.io.write(vortex.array(rows), str(path))
        files[name] = path.read_bytes()
    return files


@pytest.mark.parametrize(
    "root", ["s3://bucket/data", "gs://bucket/data", "custom://data", "relative/data"]
)
def test_discovery_uses_supplied_backend(
    root: str, shard_files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paths and permissions belong to the backend, not to Vortex's native stores."""
    storage = _RangeOnlyStorage(root, shard_files)

    def reject_native_read(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Metadata discovery must use the supplied backend")

    monkeypatch.setattr(vortex, "open", reject_native_read)
    monkeypatch.setattr(vortex.io, "read_url", reject_native_read)
    monkeypatch.setattr(vortex.VortexFile, "scan", reject_native_read)

    counts, metadata = VortexFormat().discover(root, storage)

    assert counts == {0: 2, 1: 3}
    assert [metadata[i]["raw"]["basename"] for i in counts] == sorted(shard_files)
    for name, data in shard_files.items():
        reads = [
            (start, length)
            for path, start, length in storage.reads
            if path == f"{root}/{name}"
        ]
        assert reads
        assert sum(length for _, length in reads) < len(data)


def test_discovery_honors_local_backend_root(
    tmp_path: Path, shard_files: dict[str, bytes]
) -> None:
    """Relative paths resolve against the supplied backend, not the process cwd."""
    storage = LocalFSBackend(tmp_path)
    counts, _ = VortexFormat().discover(".", storage)
    assert counts == {0: 2, 1: 3}


def test_metadata_reader_accepts_short_reads(shard_files: dict[str, bytes]) -> None:
    """Vortex completes short range reads without a whole-file fallback."""
    storage = _RangeOnlyStorage("custom://data", shard_files)
    storage.max_read = 127
    counts, _ = VortexFormat().discover(storage.root, storage)
    assert counts == {0: 2, 1: 3}


def test_parallel_discovery_keeps_filename_order(
    shard_files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slower first shard must retain its ID when later reads finish first."""
    files = dict(shard_files)
    files.update({f"{name}.vortex": files["b.vortex"] for name in "cdef"})
    storage = _RangeOnlyStorage("custom://parallel", files)
    second_finished = Event()
    read_count = vortex_format._read_vortex_row_count

    def read_out_of_order(path: str, backend: StorageBackend, size: int) -> int:
        if path.endswith("/a.vortex"):
            assert second_finished.wait(timeout=5), "The second shard did not finish"
        result = read_count(path, backend, size)
        if path.endswith("/b.vortex"):
            second_finished.set()
        return result

    monkeypatch.setattr(vortex_format, "_read_vortex_row_count", read_out_of_order)
    counts, metadata = VortexFormat().discover(storage.root, storage)
    assert counts == {0: 2, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3}
    assert [metadata[i]["raw"]["basename"] for i in counts] == sorted(files)


def test_discovery_preserves_storage_errors(
    shard_files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed backend reads report the shard path instead of bypassing its auth."""
    storage = _RangeOnlyStorage("custom://data", shard_files)

    def deny_read(*args: Any, **kwargs: Any) -> bytes:
        raise PermissionError("test backend denied access")

    monkeypatch.setattr(storage, "read_range", deny_read)
    with pytest.raises(
        ValueError, match="Failed to read Vortex shard custom://data/a.vortex"
    ) as error:
        VortexFormat().discover(storage.root, storage)
    assert "test backend denied access" in str(error.value)


def test_local_build_index_entrypoint(
    tmp_path: Path, shard_files: dict[str, bytes]
) -> None:
    """The public index builder keeps local Path results and explicit output paths."""
    target = tmp_path / "custom-index.json"
    assert build_index("vortex", tmp_path, output_path=target, progress=False) == target
    index = json.loads(target.read_text())
    assert [shard["basename"] for shard in index["shards"]] == sorted(shard_files)
    assert [shard["extra"] for shard in index["shards"]] == [
        {"length": 2},
        {"length": 3},
    ]


def test_truncated_shard_reports_path() -> None:
    """Corrupt metadata cannot silently become an empty dataset."""
    storage = _RangeOnlyStorage("custom://data", {"bad.vortex": b"bad"})
    with pytest.raises(
        ValueError, match="Failed to read Vortex shard custom://data/bad.vortex"
    ):
        VortexFormat().discover(storage.root, storage)
