# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex discovery and shard-reader unit tests."""

import importlib
import io
import json
import random
from pathlib import Path
from threading import Event
from types import ModuleType
from typing import IO, Any, Callable

import pytest

from zephon._internal.io.formats import vortex as vortex_format
from zephon._internal.io.formats.vortex import (
    VortexFormat,
    VortexShard,
)
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.storage.local import LocalFSBackend
from zephon._internal.io.types import (
    LocalShardFile,
    LocalShardRef,
)
from zephon.io.dataset import Dataset


@pytest.fixture
def vortex() -> Any:
    # The unrelated PyPI package named "vortex" does not provide vortex.io.
    pytest.importorskip("vortex.io", reason="vortex-data not installed")
    return importlib.import_module("vortex")


@pytest.fixture
def write_vortex(vortex: Any) -> Callable[[Path, Any], None]:
    def write(path: Path, rows: Any) -> None:
        vortex.io.write(vortex.array(rows), str(path))

    return write


def test_vortex_build_locators_and_open(tmp_path: Path, write_vortex: Any) -> None:
    """Test building locators and opening a Vortex shard."""
    shard0 = tmp_path / "shard0.vortex"
    write_vortex(shard0, [{"x": 1}, {"x": 2}])

    handler = VortexFormat()
    shard_index, shard_meta = handler.discover(str(tmp_path), LocalFSBackend(tmp_path))

    ds = Dataset(
        name="demo",
        backend={"kind": "vortex", "path": str(tmp_path), "shards": shard_meta},
        path=str(tmp_path),
    )

    locators = handler.build_locators(ds)
    assert set(locators.keys()) == set(shard_index.keys())
    loc = next(iter(locators.values()))

    # Open and verify shard contents
    ref = LocalShardRef(raw=LocalShardFile(path=shard0, bytes=shard0.stat().st_size))
    shard = handler.open_shard(loc, ref)
    assert len(shard) == 2
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
        {0: {"raw": {"basename": "shard0.vortex"}}},
        # Invalid bytes type
        {0: {"raw": {"basename": "shard0.vortex", "bytes": object()}}},
    ],
)
def test_vortex_build_locators_rejects_bad_metadata(tmp_path: Path, bad_shards) -> None:
    """Test that build_locators rejects invalid metadata."""
    ds = Dataset(
        name="bad",
        backend={"kind": "vortex", "path": str(tmp_path), "shards": bad_shards},
        path=str(tmp_path),
    )
    handler = VortexFormat()
    with pytest.raises(ValueError):
        _ = handler.build_locators(ds)


@pytest.mark.parametrize(
    "indices",
    [
        pytest.param([], id="empty"),
        pytest.param([3], id="single"),
        pytest.param([3, 3, 3], id="all-identical"),
        pytest.param([0, 1, 4, 5], id="sorted"),
        pytest.param([3, 2, 3], id="contiguous-duplicates"),
        pytest.param([4, 1, 4, 0, 1, 5, 0], id="unsorted-duplicates"),
    ],
)
def test_vortex_shard_getsamples(
    tmp_path: Path, indices: list[int], write_vortex: Any
) -> None:
    """Bulk reads preserve request order and duplicates without mutating indices."""
    p = tmp_path / "shard.vortex"
    rows = [{"v": i, "text": f"row_{i}"} for i in range(6)]
    write_vortex(p, rows)

    shard = VortexShard(p)
    original_indices = indices.copy()
    try:
        assert shard.getsamples(indices) == [rows[i] for i in original_indices]
        assert indices == original_indices
    finally:
        shard.close()


def test_vortex_shard_getsamples_random_order(
    tmp_path: Path, write_vortex: Any
) -> None:
    """Regression for #302: a shuffled bulk read preserves order on a larger shard."""
    rows = [{"v": i, "text": f"row_{i}"} for i in range(5000)]
    path = tmp_path / "shard.vortex"
    write_vortex(path, rows)
    indices = list(range(len(rows)))
    random.Random(0).shuffle(indices)
    shard = VortexShard(path)
    try:
        assert shard.getsamples(indices) == [rows[i] for i in indices]
    finally:
        shard.close()


@pytest.mark.parametrize("indices", [[4, -1, 0], [4, 6, 0]])
def test_vortex_shard_out_of_bounds(
    tmp_path: Path, indices: list[int], write_vortex: Any
) -> None:
    """Reject invalid indices before passing the request to Vortex."""
    p = tmp_path / "shard.vortex"
    write_vortex(p, [{"v": i} for i in range(6)])
    shard = VortexShard(p)
    try:
        with pytest.raises(IndexError):
            shard.getsamples(indices)
        with pytest.raises(IndexError):
            shard[indices[1]]
    finally:
        shard.close()


@pytest.mark.parametrize(
    "installed", [None, ModuleType("vortex")], ids=["missing", "old"]
)
def test_vortex_import_error_handling(
    monkeypatch: pytest.MonkeyPatch, installed: ModuleType | None
) -> None:
    """Missing and outdated installations fail before touching storage."""
    monkeypatch.setattr(vortex_format, "_vortex", installed)
    with pytest.raises(ImportError, match=r"vortex-data>=0\.88\.0.*pip install -U"):
        VortexShard(Path("/fake/path.vortex"))
    storage = _RangeOnlyStorage("custom://data", {})
    with pytest.raises(ImportError, match=r"vortex-data>=0\.88\.0"):
        VortexFormat().discover(storage.root, storage)
    with pytest.raises(ImportError, match=r"vortex-data>=0\.88\.0"):
        vortex_format._read_vortex_row_count("missing.vortex", storage, 1)


@pytest.mark.usefixtures("vortex")
def test_vortex_discover_no_shards(tmp_path: Path) -> None:
    """Test that discover raises ValueError when no .vortex files found."""
    # Create a directory with no .vortex files
    (tmp_path / "other.txt").write_text("hello")

    handler = VortexFormat()

    # If vortex is installed, should raise ValueError for no shards
    with pytest.raises(ValueError, match="No .vortex shards found"):
        handler.discover(str(tmp_path), LocalFSBackend(tmp_path))


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


def test_vortex_reader_forwards_ranges_and_skips_empty_reads() -> None:
    """Keep backend buffers intact and avoid unsupported empty range requests."""
    storage = _RangeOnlyStorage("custom://data", {"a.vortex": b"abcdef"})
    reader = vortex_format._VortexReader(storage, "custom://data/a.vortex", 6)
    assert reader.size() == 6
    assert reader.read_at(3, 0) == b""
    assert storage.reads == []
    result = reader.read_at(2, 3)
    assert isinstance(result, memoryview)
    assert result == b"cde"
    assert storage.reads == [(reader.path, 2, 3)]


@pytest.fixture
def shard_files(tmp_path: Path, vortex: Any) -> dict[str, bytes]:
    """Small real files with payloads larger than Vortex's footer prefetch."""
    rng = random.Random(0)
    files = {}
    for name, count in (("a.vortex", 2), ("b.vortex", 3)):
        path = tmp_path / name
        rows = [{"payload": rng.randbytes(32 * 1024)} for _ in range(count)]
        vortex.io.write(vortex.array(rows), str(path))
        files[name] = path.read_bytes()
    return files


@pytest.mark.parametrize("root", ["custom://data", "relative/data"])
def test_discovery_uses_supplied_backend(
    root: str,
    shard_files: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    vortex: Any,
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
    """Vortex retries short reads; the adapter need not loop or fetch whole files."""
    storage = _RangeOnlyStorage("custom://data", shard_files)
    storage.max_read = 127
    counts, _ = VortexFormat().discover(storage.root, storage)
    assert counts == {0: 2, 1: 3}


def test_parallel_discovery_keeps_filename_order(
    shard_files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slower first shard must retain its ID when later reads finish first."""
    files = dict(shard_files)
    # Six files enable two workers under the shared Parquet/Vortex heuristic.
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
    denied = PermissionError("test backend denied access")

    def deny_read(*args: Any, **kwargs: Any) -> bytes:
        raise denied

    monkeypatch.setattr(storage, "read_range", deny_read)
    with pytest.raises(
        ValueError, match="Failed to read Vortex shard custom://data/a.vortex"
    ) as error:
        VortexFormat().discover(storage.root, storage)
    assert "test backend denied access" in str(error.value)
    assert error.value.__cause__ is denied


@pytest.mark.usefixtures("vortex")
def test_truncated_shard_reports_path() -> None:
    """Corrupt metadata cannot silently become an empty dataset."""
    storage = _RangeOnlyStorage("custom://data", {"bad.vortex": b"bad"})
    with pytest.raises(
        ValueError, match="Failed to read Vortex shard custom://data/bad.vortex"
    ):
        VortexFormat().discover(storage.root, storage)


def _write_index(tmp_path: Path, shards: list[dict[str, object]]) -> None:
    (tmp_path / "index.json").write_text(
        json.dumps({"format_version": 1, "shards": shards}), encoding="utf-8"
    )


def test_discover_counts_matches_discover_on_index_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counts agree with ``discover`` on ``(shard_id, num_rows)``."""
    monkeypatch.setattr(vortex_format, "_vortex", None)
    _write_index(
        tmp_path,
        [
            {"basename": "a.vortex", "bytes": 10, "num_rows": 3},
            {"basename": "b.vortex", "bytes": 20, "num_rows": 5},
        ],
    )
    handler = VortexFormat()
    storage = LocalFSBackend(tmp_path)

    shard_index, _ = handler.discover(str(tmp_path), storage)
    ids, counts = handler.discover_counts(str(tmp_path), storage)

    assert ids.tolist() == sorted(shard_index)
    assert counts.tolist() == [shard_index[sid] for sid in sorted(shard_index)]


def test_discover_counts_agrees_with_discover_on_empty_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index with no shards fails count-only discovery just like ``discover``."""
    monkeypatch.setattr(vortex_format, "_vortex", None)
    _write_index(tmp_path, [])
    handler = VortexFormat()
    storage = LocalFSBackend(tmp_path)

    with pytest.raises(ValueError, match="contains no shards"):
        handler.discover(str(tmp_path), storage)
    with pytest.raises(ValueError, match="contains no shards"):
        handler.discover_counts(str(tmp_path), storage)


def test_vortex_mixed_column_types(tmp_path: Path, write_vortex: Any) -> None:
    """Test handling of various column types."""
    rows = [
        {
            "int_col": 42,
            "float_col": 3.14159,
            "str_col": "hello world",
            "bool_col": True,
            "none_col": None,
        },
        {
            "int_col": -100,
            "float_col": 2.71828,
            "str_col": "goodbye",
            "bool_col": False,
            "none_col": None,
        },
    ]
    p = tmp_path / "mixed.vortex"
    write_vortex(p, rows)

    shard = VortexShard(p)
    assert len(shard) == 2

    row0 = shard[0]
    assert row0["int_col"] == 42
    assert abs(row0["float_col"] - 3.14159) < 0.0001
    assert row0["str_col"] == "hello world"
    assert row0["bool_col"] is True

    row1 = shard[1]
    assert row1["int_col"] == -100
    assert row1["bool_col"] is False

    shard.close()
