# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex discovery, native export, and reader-cache unit tests."""

import gc
import importlib
import io
import json
import random
import subprocess
import sys
import textwrap
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace
from typing import IO, Any, Callable
from unittest import mock

import numpy as np
import pytest

from zephon._internal.io.formats import vortex as vortex_format
from zephon._internal.io.formats.vortex import (
    VortexFormat,
    VortexShard,
    VortexShardOpener,
    _list_view_type,
)
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.storage.local import LocalFSBackend
from zephon._internal.io.types import (
    LocalShardFile,
    LocalShardRef,
    ShardFile,
    ShardLocator,
)
from zephon.io import StoreOptions, VortexOptions
from zephon.io.dataset import Dataset


@pytest.fixture
def vortex() -> Any:
    # The unrelated PyPI package named "vortex" does not provide vortex.io.
    pytest.importorskip("vortex.io", reason="vortex-data not installed")
    return importlib.import_module("vortex")


@pytest.fixture
def pa() -> Any:
    return pytest.importorskip("pyarrow")


@pytest.mark.parametrize(
    "make_array,expected",
    [
        pytest.param(
            lambda pa: pa.array([10, None, 30], pa.int16()),
            [np.int16(10), None, np.int16(30)],
            id="integer",
        ),
        pytest.param(
            lambda pa: pa.array([1.5, None, 3.5], pa.float32()),
            [np.float32(1.5), None, np.float32(3.5)],
            id="float",
        ),
        pytest.param(
            lambda pa: pa.array([True, None, False]), [True, None, False], id="boolean"
        ),
        pytest.param(
            lambda pa: pa.array(["zero", None, "two"]), ["zero", None, "two"], id="text"
        ),
        pytest.param(
            lambda pa: pa.array([b"zero", None, b"two"]),
            [b"zero", None, b"two"],
            id="binary",
        ),
        pytest.param(
            lambda pa: pa.array([[1, 2], None, []], pa.list_(pa.int16())),
            [np.array([1, 2], dtype=np.int16), None, np.array([], dtype=np.int16)],
            id="numeric-list",
        ),
        pytest.param(
            lambda pa: pa.array([[1, None], None, [3, None]], pa.list_(pa.int16())),
            [[1, None], None, [3, None]],
            id="inner-nulls",
        ),
        pytest.param(
            lambda pa: pa.array(
                [[[1, 2], [3, 4]], None, [[5, 6], [7, 8]]],
                pa.list_(pa.list_(pa.int16(), 2), 2),
            ),
            [
                np.array([[1, 2], [3, 4]], dtype=np.int16),
                None,
                np.array([[5, 6], [7, 8]], dtype=np.int16),
            ],
            id="fixed-nesting",
        ),
        pytest.param(
            lambda pa: pa.array(
                [[[1], [2, 3]], None, [[4, 5], []]], pa.list_(pa.list_(pa.int16()))
            ),
            [[[1], [2, 3]], None, [[4, 5], []]],
            id="variable-nesting",
        ),
        pytest.param(
            lambda pa: pa.array(
                [1_700_000_000_000_000_001, None, -1], pa.timestamp("ns")
            ),
            [
                np.datetime64(1_700_000_000_000_000_001, "ns"),
                None,
                np.datetime64(-1, "ns"),
            ],
            id="timestamp",
        ),
    ],
)
def test_non_struct_roots_preserve_values_and_ownership(
    tmp_path: Path,
    vortex: Any,
    pa: Any,
    make_array: Callable[[Any], Any],
    expected: list[Any],
) -> None:
    """Bare roots retain their types, order and buffers, including cached reopen."""
    path = tmp_path / "values.vortex"
    vortex.io.write(vortex.array(make_array(pa)), str(path))
    metadata_cache = vortex_format.MetadataCache(2)
    requested = [0, 2, 0, 1]
    for _ in range(2):
        shard = VortexShard(path, metadata_cache=metadata_cache)
        try:
            rows = shard.getsamples(requested)
            single = shard[0]
        finally:
            shard.close()
        del shard
        gc.collect()
        for actual, index in zip([*rows, single], [*requested, 0]):
            value = expected[index]
            assert type(actual) is type(value)
            if isinstance(value, np.ndarray):
                assert actual.dtype == value.dtype
                assert actual.shape == value.shape
                assert actual.tolist() == value.tolist()
                assert not actual.flags.writeable
            else:
                assert actual == value
        if isinstance(rows[0], np.ndarray):
            assert np.shares_memory(rows[0], rows[2])
        elif isinstance(rows[0], list):
            rows[0].append("changed")
            assert rows[2] == expected[0]


@pytest.fixture
def write_vortex(vortex: Any) -> Callable[[Path, Any], None]:
    def write(path: Path, rows: Any) -> None:
        vortex.io.write(vortex.array(rows), str(path))

    return write


def test_vortex_caps_threads_at_initialization_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_vortex: Any
) -> None:
    """Only reader setup caps Vortex; imports, discovery and reads do not."""
    write_vortex(tmp_path / "data.vortex", [{"value": 1}, {"value": 2}])
    monkeypatch.setenv("VORTEX_MAX_THREADS", "2")
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from unittest.mock import patch

        import vortex

        assert vortex.worker_threads() == 2
        with patch.object(vortex, "set_worker_threads", wraps=vortex.set_worker_threads) as cap:
            from zephon._internal.io.formats.vortex import VortexFormat, VortexShard
            from zephon._internal.io.storage.local import LocalFSBackend
            from zephon.io import StoreOptions, VortexOptions

            cap.assert_not_called()
            root = Path(sys.argv[1])
            counts, _ = VortexFormat().discover(str(root), LocalFSBackend(root))
            assert sum(counts.values()) == 2
            cap.assert_not_called()
            with VortexFormat().create_opener(
                None,
                StoreOptions(vortex=VortexOptions(worker_threads=1)),
                LocalFSBackend(root),
            ):
                cap.assert_called_once_with(1)
            for _ in range(2):
                shard = VortexShard(root / "data.vortex")
                try:
                    assert shard[0] == {"value": 1}
                finally:
                    shard.close()
            cap.assert_called_once_with(1)
            assert vortex.worker_threads() == 1
        """
    )
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], check=True, timeout=30
    )


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
    indices = random.Random(0).choices(range(len(rows)), k=64)
    # Include distant boundaries and guarantee non-consecutive duplicates.
    indices.extend([len(rows) - 1, 0, indices[0]])
    shard = VortexShard(path)
    try:
        for requested in (sorted(indices), indices):
            assert shard.getsamples(requested) == [rows[i] for i in requested]
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
    with pytest.raises(ImportError, match=r"vortex-data>=0\.90\.0.*pip install -U"):
        VortexShard(Path("/fake/path.vortex"))
    storage = _RangeOnlyStorage("custom://data", {})
    with pytest.raises(ImportError, match=r"vortex-data>=0\.90\.0"):
        VortexFormat().discover(storage.root, storage)
    with pytest.raises(ImportError, match=r"vortex-data>=0\.90\.0"):
        vortex_format._read_vortex_row_count("missing.vortex", storage, 1)


def test_vortex_read_failure_preserves_error_unless_file_is_missing(
    tmp_path: Path, write_vortex: Any
) -> None:
    """Only a filesystem failure makes a native read error retryable."""
    path = tmp_path / "rows.vortex"
    write_vortex(path, [{"value": 1}])
    shard = VortexShard(path)
    failure = RuntimeError("invalid encoded data")
    try:
        with mock.patch.object(shard._file, "scan", side_effect=failure):
            with pytest.raises(RuntimeError) as caught:
                shard[0]
            assert caught.value is failure
            path.unlink()
            with pytest.raises(FileNotFoundError):
                shard[0]
    finally:
        shard.close()


@pytest.mark.usefixtures("vortex")
def test_vortex_discover_no_shards(tmp_path: Path) -> None:
    """Test that discover raises ValueError when no .vortex files found."""
    # Create a directory with no .vortex files
    (tmp_path / "other.txt").write_text("hello")

    handler = VortexFormat()

    # If vortex is installed, should raise ValueError for no shards
    with pytest.raises(ValueError, match="No .vortex shards found"):
        handler.discover(str(tmp_path), LocalFSBackend(tmp_path))


def test_vortex_single_row_shard(tmp_path: Path, write_vortex: Any) -> None:
    """Test handling of shards with a single row."""
    p = tmp_path / "single.vortex"
    write_vortex(p, [{"only": "row"}])

    shard = VortexShard(p)
    assert len(shard) == 1
    assert shard[0] == {"only": "row"}

    # getsamples with single item
    result = shard.getsamples([0])
    assert len(result) == 1
    assert result[0] == {"only": "row"}

    # Out of bounds
    with pytest.raises(IndexError):
        _ = shard[1]

    shard.close()


@pytest.mark.parametrize("indices", [[73], [76, 74, 75, 73, 74]])
def test_contiguous_read_limits_the_decoded_list_values(
    tmp_path: Path, indices: list[int], vortex: Any, pa: Any
) -> None:
    """Compressed token columns expose only the selected decoded child values."""
    tokens = np.random.default_rng(0).integers(0, 256, size=(256, 32), dtype=np.uint32)
    array = pa.ListArray.from_arrays(
        pa.array(np.arange(0, tokens.size + 1, 32), type=pa.int32()),
        pa.array(tokens.reshape(-1)),
    )
    path = tmp_path / "tokens.vortex"
    vortex.io.write(pa.table({"tokens": array}), str(path))
    # Without the copy, the views show the size of the exported buffer.
    shard = VortexShard(path, copy_mapped_arrays=False)
    try:
        rows = shard.getsamples(indices)
    finally:
        shard.close()
    for index, row in zip(indices, rows):
        values = row["tokens"]
        np.testing.assert_array_equal(values, tokens[index])
        assert values.dtype == np.uint32
        assert not values.flags.writeable
        owner = values
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        assert owner.nbytes == values.nbytes * len(set(indices))


def test_list_view_export_preserves_field_metadata_and_fixed_shapes(pa: Any) -> None:
    child = pa.field("element", pa.int16(), nullable=False, metadata={b"unit": b"x"})
    field = pa.field("values", pa.list_(child), metadata={b"label": b"tokens"})
    kind = pa.struct([field, pa.field("matrix", pa.list_(pa.list_(child), 2))])
    transformed = _list_view_type(kind, pa)
    assert transformed[0].metadata == field.metadata
    assert transformed[0].nullable == field.nullable
    assert transformed[0].type.value_field == child
    assert pa.types.is_list_view(transformed[0].type)
    assert transformed[1].type.list_size == 2
    assert pa.types.is_list_view(transformed[1].type.value_type)


@pytest.mark.parametrize("indices", [[0], [1, 1], [2, 1, 0], [0, 2]])
def test_text_only_reads_preserve_nulls_bytes_and_duplicate_order(
    tmp_path: Path, indices: list[int], vortex: Any, pa: Any
) -> None:
    table = pa.table({"text": ["é", None, "漢字"], "bytes": [b"\x00", b"", None]})
    path = tmp_path / "text.vortex"
    vortex.io.write(table, str(path))
    shard = VortexShard(path)
    rows = shard.getsamples(indices)
    shard.close()
    expected = table.to_pylist()
    assert rows == [expected[index] for index in indices]
    if len(indices) > 1:
        assert rows[0] is not rows[1]


def test_nullable_native_struct_uses_arrow_table_row_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vortex: Any, pa: Any
) -> None:
    """A null struct scalar must not fail or depend on request batch size."""
    source = vortex.array(
        pa.array(
            [{"text": "ok", "bytes": b"x"}, None, {"text": None, "bytes": None}],
            type=pa.struct([("text", pa.string()), ("bytes", pa.binary())]),
        )
    )

    def scan(*, indices: Any) -> Any:
        selected = source.take(indices)
        return vortex.ArrayIterator.from_iter(selected.dtype, iter([selected]))

    # Vortex's file statistics writer rejects nullable root structs. Feed a
    # real native nullable array through the reader to exercise conversion alone.
    file = SimpleNamespace(dtype=source.dtype, scan=scan)
    metadata = vortex_format._VortexMetadata(None, source.dtype.to_arrow_type(), True)
    monkeypatch.setattr(
        vortex_format, "_open_vortex_file", lambda *_args, **_kwargs: (file, metadata)
    )
    shard = VortexShard(tmp_path / "nullable.vortex", length=len(source))
    try:
        rows = shard.getsamples([1, 1])
    finally:
        shard.close()
    expected = source.to_arrow_table().to_pylist()
    assert rows == [expected[1], expected[1]]


@pytest.mark.parametrize("indices", [[3, 1, 2, 0, 1], [0, 2]])
def test_list_export_preserves_inner_nulls_and_nested_fallbacks(
    tmp_path: Path, indices: list[int], vortex: Any, pa: Any
) -> None:
    table = pa.table(
        {
            "numbers": pa.array(
                [[1, None], [2, 3], None, []], type=pa.list_(pa.int16())
            ),
            "nested": pa.array([[[1], [2, 3]], None, [], [[4]]]),
            "strings": pa.array([["a"], None, [], ["b", "c"]]),
            "structs": pa.array([[{"a": 1}], [], None, [{"a": None}]]),
        }
    )
    path = tmp_path / "nested.vortex"
    vortex.io.write(table, str(path))
    shard = VortexShard(path)
    rows = shard.getsamples(indices)
    shard.close()
    expected = table.to_pylist()
    for index, row in zip(indices, rows):
        numbers = row["numbers"]
        if isinstance(numbers, np.ndarray):
            assert numbers.dtype == np.int16
            assert not numbers.flags.writeable
            row["numbers"] = numbers.tolist()
        assert row == expected[index]


def test_vortex_rows_survive_close_and_preserve_duplicate_order(
    tmp_path: Path, vortex: Any, pa: Any
) -> None:
    path = tmp_path / "typed.vortex"
    table = pa.table(
        {
            "id": pa.array([0, 1, 2], type=pa.int32()),
            "vector": pa.array([[1, 2], [], [3, 4, 5]], type=pa.list_(pa.float32())),
            "matrix": pa.array(
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
                type=pa.list_(pa.list_(pa.int16(), 2), 2),
            ),
            "text": ["zero", "one", "two"],
            "binary": [b"a", b"b", b"c"],
        }
    )
    vortex.io.write(table, str(path))
    shard = VortexShard(path)
    rows = shard.getsamples([2, 0, 2, 1])
    single = shard[2]
    shard.close()
    with pytest.raises(RuntimeError, match="closed"):
        shard[0]
    with pytest.raises(RuntimeError, match="closed"):
        shard.getsamples([0, 1])
    del shard
    gc.collect()
    assert [row["id"] for row in rows] == [2, 0, 2, 1]
    assert rows[0]["id"].dtype == np.int32
    assert rows[0]["vector"].dtype == np.float32
    assert rows[0]["vector"].tolist() == [3, 4, 5]
    np.testing.assert_array_equal(single["vector"], rows[0]["vector"])
    assert rows[0]["matrix"].shape == (2, 2)
    assert rows[0]["matrix"].dtype == np.int16
    assert rows[0]["text"] == "two"
    assert rows[0]["binary"] == b"c"
    assert rows[3]["vector"].shape == (0,)
    assert np.shares_memory(rows[0]["vector"], rows[2]["vector"])
    assert not rows[0]["vector"].flags.writeable


def test_nullable_scalar_single_and_bulk_exports_agree(
    tmp_path: Path, pa: Any, write_vortex: Any
) -> None:
    """Native scalar, contiguous, and sparse reads share the numeric contract."""
    path = tmp_path / "scalar.vortex"
    maximum = np.iinfo(np.uint64).max
    write_vortex(
        path, pa.table({"value": pa.array([maximum, None, 3], type=pa.uint64())})
    )
    shard = VortexShard(path)
    try:
        single = shard[0]
        contiguous = shard.getsamples([0, 1, 2])
        sparse = shard.getsamples([2, 0])
        assert shard[1] == {"value": None}
    finally:
        shard.close()
    assert contiguous[1] == {"value": None}
    for value in (single, contiguous[0], sparse[1]):
        assert type(value["value"]) is np.uint64
        assert value["value"] == maximum


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
        # Unsorted on purpose: discovery must sort.
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


@pytest.mark.parametrize("metadata_entries", [0, 1])
@pytest.mark.parametrize("segment_bytes", [0, 4096])
def test_opener_retains_only_enabled_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_entries: int,
    segment_bytes: int,
    pa: Any,
) -> None:
    """Cache options control footer/schema reuse and native segment arguments."""
    to_arrow_type = mock.Mock(return_value=pa.struct([("value", pa.int64())]))
    native_open = mock.Mock(
        side_effect=lambda *_args, **_kwargs: SimpleNamespace(
            footer=object(), dtype=SimpleNamespace(to_arrow_type=to_arrow_type)
        )
    )
    segment_cache = mock.Mock()
    make_cache = mock.Mock(return_value=segment_cache)
    monkeypatch.setattr(
        vortex_format,
        "_require_vortex",
        lambda: SimpleNamespace(
            open=native_open, SegmentCache=make_cache, set_worker_threads=mock.Mock()
        ),
    )
    opener = VortexShardOpener(
        StoreOptions(
            metadata_cache_entries=metadata_entries,
            # Segment caches apply to file reads; mapped reads are tested below.
            vortex=VortexOptions(memory_map=False, segment_cache_bytes=segment_bytes),
        )
    )
    try:
        for name in ("a.vortex", "a.vortex", "b.vortex", "a.vortex"):
            path = tmp_path / name
            locator = ShardLocator(
                dataset="test",
                shard_id=0,
                format="vortex",
                root=str(tmp_path),
                raw=ShardFile(basename=name, bytes=1, hashes={}),
            )
            ref = LocalShardRef(
                raw=LocalShardFile(path=path, bytes=1), extra={"length": 1}
            )
            reader = opener.open_shard(locator, ref)
            reader.close()
        calls = native_open.call_args_list
        assert len(calls) == 4  # A footer miss must not open the file twice.
        assert to_arrow_type.call_count == (3 if metadata_entries else 4)
        assert ["footer" in call.kwargs for call in calls] == [
            False,
            bool(metadata_entries),
            False,
            False,
        ]
        if segment_bytes:
            make_cache.assert_called_once_with(segment_bytes)
            assert all(call.kwargs["segment_cache"] is segment_cache for call in calls)
            assert calls[0].kwargs["cache_key"] == calls[1].kwargs["cache_key"]
            assert calls[0].kwargs["cache_key"] != calls[2].kwargs["cache_key"]
        else:
            make_cache.assert_not_called()
            assert all(call.kwargs["without_segment_cache"] for call in calls)
            assert all("segment_cache" not in call.kwargs for call in calls)
    finally:
        opener.close()
    if segment_bytes:
        segment_cache.clear.assert_called_once_with()


@pytest.mark.parametrize(
    ("memory_map", "copy_mapped_arrays", "owned"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_mapped_reads_copy_arrays_unless_disabled(
    tmp_path: Path,
    vortex: Any,
    pa: Any,
    memory_map: bool,
    copy_mapped_arrays: bool,
    owned: bool,
) -> None:
    """Mapped reads return owned arrays; duplicate rows keep sharing one array."""
    path = tmp_path / "tokens.vortex"
    table = pa.table(
        {"tokens": pa.array([[1, 2], [3, 4, 5], [6]], type=pa.list_(pa.int32()))}
    )
    vortex.io.write(table, str(path))
    shard = VortexShard(
        path, memory_map=memory_map, copy_mapped_arrays=copy_mapped_arrays
    )
    try:
        rows = shard.getsamples([1, 0, 1])
    finally:
        shard.close()

    assert [row["tokens"].tolist() for row in rows] == [[3, 4, 5], [1, 2], [3, 4, 5]]
    assert [row["tokens"].base is None for row in rows] == [owned] * 3
    assert rows[0]["tokens"] is rows[2]["tokens"] or not owned
    assert all(not row["tokens"].flags.writeable for row in rows)


@pytest.mark.parametrize("struct", [False, True], ids=["bare", "struct"])
@pytest.mark.parametrize("nested", [False, True], ids=["vector", "matrix"])
def test_mapped_fixed_size_duplicates_share_one_owned_array(
    tmp_path: Path, vortex: Any, pa: Any, struct: bool, nested: bool
) -> None:
    """Repeating a 4 KiB fixed-size row must not allocate 64 separate copies."""
    values = np.arange(2 * 1024, dtype=np.int32).reshape(2, 1024)
    kind = pa.list_(pa.int32(), 1024)
    if nested:
        values = values.reshape(2, 32, 32)
        kind = pa.list_(pa.list_(pa.int32(), 32), 32)
    array = pa.array(values.tolist(), type=kind)
    source = pa.table({"tokens": array}) if struct else vortex.array(array)
    path = tmp_path / "fixed.vortex"
    vortex.io.write(source, str(path))
    shard = VortexShard(path, memory_map=True)
    try:
        rows = shard.getsamples([1] * 64)
    finally:
        shard.close()

    arrays = [row["tokens"] if struct else row for row in rows]
    assert len(arrays) == 64
    assert all(array is arrays[0] for array in arrays)
    assert arrays[0].nbytes == 4096
    assert arrays[0].base is None
    assert not arrays[0].flags.writeable
    np.testing.assert_array_equal(arrays[0], values[1])
    if struct:
        assert len({id(row) for row in rows}) == 64


@pytest.mark.parametrize(
    ("options", "memory_map", "copy_arrays"),
    [
        ({}, False, False),
        ({"memory_map": True}, True, True),
        ({"memory_map": True, "safe_mmap": False}, True, False),
    ],
)
def test_opener_uses_regular_reads_unless_mmap_is_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pa: Any,
    options: dict[str, bool],
    memory_map: bool,
    copy_arrays: bool,
) -> None:
    """Regular reads cache segments; opt-in mmap copies array views by default."""
    to_arrow_type = mock.Mock(return_value=pa.struct([("value", pa.int64())]))
    native_open = mock.Mock(
        side_effect=lambda *_args, **_kwargs: SimpleNamespace(
            footer=object(), dtype=SimpleNamespace(to_arrow_type=to_arrow_type)
        )
    )
    make_cache = mock.Mock()
    monkeypatch.setattr(
        vortex_format,
        "_require_vortex",
        lambda: SimpleNamespace(
            open=native_open, SegmentCache=make_cache, set_worker_threads=mock.Mock()
        ),
    )
    path = tmp_path / "a.vortex"
    path.write_bytes(b"data")
    locator = ShardLocator(
        dataset="test",
        shard_id=0,
        format="vortex",
        root=str(tmp_path),
        raw=ShardFile(basename="a.vortex", bytes=4, hashes={}),
    )
    ref = LocalShardRef(raw=LocalShardFile(path=path, bytes=4), extra={"length": 1})

    opener = VortexShardOpener(StoreOptions.from_any({"vortex": options}))
    try:
        shards = [opener.open_shard(locator, ref) for _ in range(2)]
        assert all(shard._copy_arrays is copy_arrays for shard in shards)
        assert [
            call.kwargs.get("memory_map", False) for call in native_open.call_args_list
        ] == [memory_map] * 2
        if memory_map:
            make_cache.assert_not_called()
        else:
            make_cache.assert_called_once_with(64 * 1024**2)
            assert all(
                call.kwargs["segment_cache"] is make_cache.return_value
                for call in native_open.call_args_list
            )
    finally:
        opener.close()
