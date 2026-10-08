from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from zephon._internal.io.formats import vortex as vortex_format
from zephon._internal.io.formats.vortex import (
    VortexFormat,
    VortexShard,
    segment_cache_key,
)
from zephon._internal.io.formats.vortex_readable import StorageReadAt
from zephon._internal.io.storage import RouterStorageBackend, StorageBackend
from zephon._internal.io.storage.local import LocalFSBackend
from zephon._internal.io.storage.obstore import ObstoreBackend
from zephon._internal.io.stores.multi import (
    build_multi_dataset_store,
    build_resolver_with_locators,
)
from zephon._internal.io.types import RemoteShardRef, ShardFile, ShardLocator
from zephon.io import Dataset, StoreOptions


def plain(rows: Any) -> Any:
    """Turn the NumPy arrays of Vortex rows into lists, to compare with the input."""
    if isinstance(rows, list):
        return [plain(row) for row in rows]
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in rows.items()}


@pytest.fixture(autouse=True)
def fresh_segment_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test its own process caches; test files reuse cache keys."""
    monkeypatch.setattr(vortex_format, "_segment_caches", {})
    monkeypatch.setattr(vortex_format, "_vortex_stores", {})


class RangeOnlyBackend(LocalFSBackend):
    """A remote namespace without obstore, which reads shard payloads through ranges."""

    def __init__(self, root: Path, *, chunk: int | None = None) -> None:
        super().__init__(root)
        self.chunk = chunk
        self.ranges: list[tuple[int, int]] = []

    def _abspath(self, path: str) -> Path:
        return self.root / path.removeprefix("s3://test/data").lstrip("/")

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        if path.endswith(".vortex"):
            raise AssertionError("Shard opened instead of range-read")
        return super().open(path, mode, **kwargs)

    def download(self, src: str, dst: str, timeout: float | None = None) -> None:
        raise AssertionError("Shard downloaded instead of range-read")

    def read_range(
        self,
        path: str,
        start: int,
        *,
        end: int | None = None,
        length: int | None = None,
    ) -> bytes:
        assert length is not None and end is None
        self.ranges.append((start, length))
        if self.chunk is not None:
            length = min(length, self.chunk)
        return bytes(super().read_range(path, start, length=length))


def test_storage_read_at_concurrent_ranges_and_eof(tmp_path: Path) -> None:
    data = bytes(range(256)) * 100
    (tmp_path / "data").write_bytes(data)
    backend = RangeOnlyBackend(tmp_path)
    source = StorageReadAt(backend, "s3://test/data/data", len(data))

    def read(offset: int) -> bytes:
        return bytes(source.read_at(offset, 100))

    offsets = [1000, 0, 500, len(data) - 10, len(data), len(data) + 1] * 10
    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(read, offsets)) == [data[i : i + 100] for i in offsets]
    assert source.size() == len(data)


def test_storage_read_at_returns_backend_buffer(tmp_path: Path) -> None:
    backend = LocalFSBackend(tmp_path)
    source = StorageReadAt(backend, "data", 3)
    data = b"abc"

    with patch.object(backend, "read_range", return_value=data):
        assert source.read_at(0, 3) is data


def test_storage_read_at_retries_and_rejects_overlong_result(tmp_path: Path) -> None:
    backend = LocalFSBackend(tmp_path)
    source = StorageReadAt(
        backend, "data", 3, retry_attempts=2, retry_initial_backoff=0
    )
    with patch.object(
        backend, "read_range", side_effect=[OSError("retry"), b"abc"]
    ) as read:
        assert source.read_at(0, 3) == b"abc"
        assert read.call_count == 2
    with patch.object(backend, "read_range", return_value=b"abcd"):
        with pytest.raises(ValueError, match="more bytes"):
            source.read_at(0, 3)


@pytest.fixture
def vortex_file(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    vx = pytest.importorskip("vortex")
    if not hasattr(vx.io, "ReadBytesAt"):
        pytest.skip("requires experimental Vortex ReadBytesAt bindings")
    rows = [{"id": i, "text": f"row-{i}", "values": [i, i + 1]} for i in range(1000)]
    path = tmp_path / "part.vortex"
    vx.io.write(vx.array(rows), str(path))
    return path, rows


@pytest.mark.parametrize("chunk", [None, 37])
def test_python_source_single_bulk_and_close(vortex_file, chunk: int | None) -> None:
    path, rows = vortex_file
    backend = RangeOnlyBackend(path.parent, chunk=chunk)
    source = StorageReadAt(backend, "s3://test/data/part.vortex", path.stat().st_size)
    shard = VortexShard("s3://test/data/part.vortex", source=source, concurrency=2)
    try:
        assert plain(shard[500]) == rows[500]
        indices = [999, 0, 500, 500, 9]
        assert plain(shard.getsamples(indices)) == [rows[i] for i in indices]
        assert plain(shard.getsamples([])) == []
        with pytest.raises(IndexError):
            shard.getsamples([1000])
        assert backend.ranges
    finally:
        shard.close()
    with pytest.raises(RuntimeError, match="closed"):
        shard[0]


def test_remote_without_obstore_uses_python_ranges(
    vortex_file: tuple[Path, list[dict[str, Any]]],
) -> None:
    path, rows = vortex_file
    backend = RangeOnlyBackend(path.parent)
    # The catalog reconstructs its storage router independently of from_path.
    with (
        patch("zephon.io.dataset._RouterStorageBackend", return_value=backend),
        patch(
            "zephon._internal.io.storage.RouterStorageBackend",
            return_value=backend,
        ),
    ):
        dataset = Dataset.from_path("remote", "s3://test/data/", fmt="vortex")
        assert len(dataset) == len(rows)
        # Prefetch has nothing to download; it resolves without IO.
        resolver, prefetch_locators = build_resolver_with_locators(
            {0: dataset}, storage=backend, options=StoreOptions()
        )
        assert isinstance(resolver.resolve(prefetch_locators[(0, 0)]), RemoteShardRef)
        backend.ranges.clear()
        store = build_multi_dataset_store(
            {0: dataset},
            storage=backend,
            options=StoreOptions(),
        )
        try:
            shard, reused = store.for_dataset(0).open(0)
            assert not reused
            result, _stats = shard.getsamples([900, 1, 900])
            assert plain(result) == [rows[i] for i in [900, 1, 900]]
            assert plain(shard[42][0]) == rows[42]
            assert store.for_dataset(0).open(0)[1]
        finally:
            store.close()
    assert backend.ranges


def _remote_dataset(backend: "RangeOnlyBackend", name: str = "remote") -> Dataset:
    with (
        patch("zephon.io.dataset._RouterStorageBackend", return_value=backend),
        patch(
            "zephon._internal.io.storage.RouterStorageBackend",
            return_value=backend,
        ),
    ):
        return Dataset.from_path(name, "s3://test/data/", fmt="vortex")


def test_uncached_store_without_segment_cache_reads_footer_once(
    vortex_file: tuple[Path, list[dict[str, Any]]],
) -> None:
    path, rows = vortex_file
    size = path.stat().st_size
    backend = RangeOnlyBackend(path.parent)
    with patch(
        "zephon._internal.io.storage.RouterStorageBackend", return_value=backend
    ):
        dataset = _remote_dataset(backend)
        options = StoreOptions.from_any({"vortex_segment_cache_bytes": 0})
        store = build_multi_dataset_store(
            {0: dataset}, storage=backend, options=options
        )
        try:
            shard, _ = store.for_dataset(0).open(0)
            backend.ranges.clear()
            assert plain(shard.getsamples([3])[0]) == [rows[3]]
            # The footer sits at the end of the file.
            assert any(start + length == size for start, length in backend.ranges)

            backend.ranges.clear()
            assert plain(shard.getsamples([3])[0]) == [rows[3]]
            assert backend.ranges
            assert all(start + length < size for start, length in backend.ranges)
        finally:
            store.close()


def test_uncached_store_reuses_segments_across_batches_and_stores(
    vortex_file: tuple[Path, list[dict[str, Any]]],
) -> None:
    path, rows = vortex_file
    backend = RangeOnlyBackend(path.parent)
    with patch(
        "zephon._internal.io.storage.RouterStorageBackend", return_value=backend
    ):
        dataset = _remote_dataset(backend, name="shared-cache")
        options = StoreOptions.from_any({"vortex_segment_cache_bytes": "64mb"})
        first = build_multi_dataset_store(
            {0: dataset}, storage=backend, options=options
        )
        second = build_multi_dataset_store(
            {0: dataset}, storage=backend, options=options
        )
        try:
            shard, _ = first.for_dataset(0).open(0)
            backend.ranges.clear()
            assert plain(shard.getsamples([3, 700])[0]) == [rows[3], rows[700]]
            assert backend.ranges

            # A later batch reopens the file but reads nothing.
            backend.ranges.clear()
            assert plain(shard.getsamples([700, 3])[0]) == [rows[700], rows[3]]
            assert not backend.ranges

            # Another store of the same process, such as a second FetchOp,
            # shares the cache; it reads only the footer it has not seen.
            other, _ = second.for_dataset(0).open(0)
            assert plain(other.getsamples([3])[0]) == [rows[3]]
            size = path.stat().st_size
            assert all(start + length == size for start, length in backend.ranges)
        finally:
            first.close()
            second.close()


def test_segment_cache_key_identifies_contents() -> None:
    def locator(root: str, size: int, hashes: dict[str, str]) -> ShardLocator:
        return ShardLocator(
            dataset="d",
            shard_id=0,
            format="vortex",
            root=root,
            raw=ShardFile(basename="part.vortex", bytes=size, hashes=hashes),
            zip=None,
            compression=None,
            extra=None,
        )

    base = segment_cache_key(locator("s3://a/data", 10, {}))
    assert base == segment_cache_key(locator("s3://a/data", 10, {}))
    assert base != segment_cache_key(locator("s3://b/data", 10, {}))
    assert base != segment_cache_key(locator("s3://a/data", 11, {}))
    assert base != segment_cache_key(locator("s3://a/data", 10, {"xxh64": "1"}))


def _remote_locator(size: int) -> ShardLocator:
    return ShardLocator(
        dataset="remote",
        shard_id=0,
        format="vortex",
        root="s3://test/data",
        raw=ShardFile(basename="part.vortex", bytes=size, hashes={}),
    )


def test_remote_shard_keeps_footer_after_read_failure(
    vortex_file: tuple[Path, list[dict[str, Any]]],
) -> None:
    path, rows = vortex_file
    size = path.stat().st_size
    backend = RangeOnlyBackend(path.parent)
    opener = VortexFormat()
    locator = _remote_locator(size)
    ref = RemoteShardRef(backend, "s3://test/data/part.vortex", size)

    def read(index: int) -> list[dict[str, Any]]:
        shard = opener.open_remote_shard(locator, ref)
        try:
            return shard.getsamples([index])
        finally:
            shard.close()

    assert plain(read(3)) == [rows[3]]

    with patch.object(
        backend, "read_range", side_effect=OSError("storage unavailable")
    ):
        with pytest.raises(Exception, match="storage unavailable"):
            read(3)

    backend.ranges.clear()
    assert plain(read(3)) == [rows[3]]
    assert backend.ranges
    assert all(start + length < size for start, length in backend.ranges)


class StoreBackend(ObstoreBackend):
    """An ``s3://test`` bucket that is an obstore store."""

    valid_schemes = frozenset({"s3"})

    def __init__(self, store: object) -> None:
        self._store = store

    def _get_store(self, bucket: str) -> object:
        assert bucket == "test"
        return self._store


class StoreRouter(RouterStorageBackend):
    """Send ``s3://test`` to an obstore store, and other paths to the usual backends."""

    def __init__(self, store: object) -> None:
        super().__init__()
        self.store_backend = StoreBackend(store)

    def _backend_for(self, path: str) -> StorageBackend:
        if path.startswith("s3://"):
            return self.store_backend

        return super()._backend_for(path)


@contextmanager
def _routed(router: StoreRouter) -> Iterator[None]:
    """Use ``router`` wherever Zephon makes a storage router, as for catalog builds."""
    with (
        patch("zephon.io.dataset._RouterStorageBackend", return_value=router),
        patch("zephon._internal.io.storage.RouterStorageBackend", return_value=router),
    ):
        yield


def test_obstore_backend_is_read_natively(
    vortex_file: tuple[Path, list[dict[str, Any]]], tmp_path: Path
) -> None:
    import vortex as vx
    from obstore.store import LocalStore

    path, rows = vortex_file
    (tmp_path / "data").mkdir()
    path.rename(tmp_path / "data" / "part.vortex")
    router = StoreRouter(LocalStore(tmp_path))

    with (
        _routed(router),
        patch.object(vx, "open", wraps=vx.open) as open_file,
        patch.object(vx, "open_readable") as open_readable,
    ):
        dataset = Dataset.from_path("remote", "s3://test/data/", fmt="vortex")
        assert len(dataset) == len(rows)

        store = build_multi_dataset_store(
            {0: dataset}, storage=router, options=StoreOptions()
        )
        try:
            shard, _ = store.for_dataset(0).open(0)
            result, _stats = shard.getsamples([900, 1, 900])
            assert plain(result) == [rows[i] for i in [900, 1, 900]]

            assert open_file.call_args.args[0] == "data/part.vortex"
            converted = open_file.call_args.kwargs["store"]
            assert isinstance(converted, vx.store.LocalStore)

            # A later batch reuses the converted store and the footer.
            shard.getsamples([1])
            assert open_file.call_args.kwargs["store"] is converted
            assert open_file.call_args.kwargs["footer"] is not None
        finally:
            store.close()

        open_readable.assert_not_called()


def test_obstore_memory_store_falls_back_to_python_ranges(
    vortex_file: tuple[Path, list[dict[str, Any]]],
) -> None:
    import obstore
    import vortex as vx
    from obstore.store import MemoryStore

    path, rows = vortex_file
    memory = MemoryStore()
    obstore.put(memory, "data/part.vortex", path.read_bytes())
    router = StoreRouter(memory)

    with (
        _routed(router),
        patch.object(vx, "open_readable", wraps=vx.open_readable) as open_readable,
    ):
        dataset = Dataset.from_path("remote", "s3://test/data/", fmt="vortex")
        store = build_multi_dataset_store(
            {0: dataset}, storage=router, options=StoreOptions()
        )
        try:
            shard, _ = store.for_dataset(0).open(0)
            assert plain(shard.getsamples([3, 700])[0]) == [rows[3], rows[700]]
        finally:
            store.close()

    assert isinstance(open_readable.call_args.args[0], StorageReadAt)


def test_uncached_local_dataset_is_read_natively(
    vortex_file: tuple[Path, list[dict[str, Any]]],
) -> None:
    import vortex as vx

    path, rows = vortex_file
    dataset = Dataset.from_path("local", str(path.parent), fmt="vortex")
    store = build_multi_dataset_store({0: dataset}, options=StoreOptions())
    try:
        with (
            patch.object(vx, "open", wraps=vx.open) as open_file,
            patch.object(vx, "open_readable") as open_readable,
        ):
            shard, _ = store.for_dataset(0).open(0)
            assert plain(shard.getsamples([2, 0])[0]) == [rows[2], rows[0]]

            assert Path(open_file.call_args.args[0]) == path.resolve()
            assert open_file.call_args.kwargs["store"] is None
            open_readable.assert_not_called()
    finally:
        store.close()


def test_python_io_requires_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    from zephon._internal.io.formats import vortex

    monkeypatch.setattr(vortex, "_vortex", SimpleNamespace(io=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="ReadAt bindings"):
        VortexShard(Path("/unused.vortex"))

    monkeypatch.setattr(
        vortex,
        "_vortex",
        SimpleNamespace(io=SimpleNamespace(ReadBytesAt=object)),
    )
    with pytest.raises(RuntimeError, match="ReadAt bindings"):
        VortexShard(Path("/unused.vortex"))


def test_python_source_reports_backend_errors(
    vortex_file: tuple[Path, list[dict[str, Any]]],
) -> None:
    path, _rows = vortex_file
    backend = LocalFSBackend(path.parent)
    source = StorageReadAt(backend, str(path), path.stat().st_size)
    with patch.object(
        backend, "read_range", side_effect=OSError("storage unavailable")
    ):
        with pytest.raises(Exception, match="storage unavailable"):
            VortexShard(str(path), source=source)
    with patch.object(backend, "read_range", return_value=b""):
        with pytest.raises(Exception, match="0 bytes"):
            VortexShard(str(path), source=source)


def test_cached_store_opens_resolved_local_file_natively(
    vortex_file: tuple[Path, list[dict[str, Any]]], tmp_path: Path
) -> None:
    import vortex as vx

    path, rows = vortex_file
    cache_root = tmp_path / "cache"
    dataset = Dataset.from_path("cached", str(path.parent), fmt="vortex")
    options = StoreOptions.from_any(
        {"cache": {"enabled": True, "root": str(cache_root)}}
    )
    store = build_multi_dataset_store({0: dataset}, options=options)
    try:
        with (
            patch.object(vx, "open", wraps=vx.open) as open_file,
            patch.object(vx, "open_readable") as open_readable,
        ):
            shard, _ = store.for_dataset(0).open(0)
            result, _stats = shard.getsamples([2, 0, 2])
            assert plain(result) == [rows[i] for i in [2, 0, 2]]

            opened = Path(open_file.call_args.args[0])
            assert opened.is_relative_to(cache_root)
            assert open_file.call_args.kwargs["segment_cache"] is not None
            assert open_file.call_args.kwargs["cache_key"]
            open_readable.assert_not_called()

            # The second batch reuses the footer from the first open.
            shard.getsamples([1])
            assert open_file.call_args.kwargs["footer"] is not None
    finally:
        store.close()


def test_numeric_lists_are_numpy_views(tmp_path: Path) -> None:
    import pyarrow as pa
    import vortex as vx

    tokens = np.arange(4 * 8, dtype=np.int32).reshape(4, 8)
    table = pa.table(
        {
            "tokens": pa.FixedSizeListArray.from_arrays(pa.array(tokens.ravel()), 8),
            "lengths": pa.array([[1.5], [], None, [2.5, 3.5]], pa.list_(pa.float32())),
            "gaps": pa.array([[1, None], [2], [3], [4]], pa.list_(pa.int64())),
            "text": pa.array(["a", "b", "c", "d"]),
            "blobs": pa.array([[b"x"], [b"y"], [], [b"z"]], pa.list_(pa.binary())),
        }
    )
    path = tmp_path / "typed.vortex"
    vx.io.write(table, str(path))

    shard = VortexShard(path)
    try:
        rows = shard.getsamples([3, 0, 2, 3])
        assert [row["tokens"].tolist() for row in rows] == tokens[[3, 0, 2, 3]].tolist()
        assert rows[0]["tokens"].dtype == np.int32
        assert not rows[0]["tokens"].flags.writeable

        assert rows[0]["lengths"].tolist() == [2.5, 3.5]
        assert rows[1]["lengths"].dtype == np.float32
        assert rows[2]["lengths"] is None

        # Values with nulls, strings and bytes stay Python objects.
        assert rows[1]["gaps"] == [1, None]
        assert rows[0]["text"] == "d"
        assert rows[0]["blobs"] == [b"z"]

        # A single row has the same types as a batch.
        assert isinstance(shard[1]["tokens"], np.ndarray)
    finally:
        shard.close()
