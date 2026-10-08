import inspect
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("litdata")
from litdata.streaming.writer import BinaryWriter

from tests._catalog_helpers import catalog_locators
from tests._helpers import counts_dict
from tests.helpers.litdata_chunks import write_litdata_fixture
from tests.helpers.storage import _install_obstore_stubs
from zephon._internal.io.formats import ensure_builtin_formats
from zephon._internal.io.formats.base import get_format
from zephon._internal.io.formats.litdata import (
    LitDataFormat,
    _LitDataExtraCodec,
    _LitDataShard,
)
from zephon._internal.io.formats.litdata_support import arrow, dependencies
from zephon._internal.io.formats.litdata_support.arrow import ArrowLoader
from zephon._internal.io.formats.litdata_support.pytree import (
    PyTreeLoader,
    TokensLoader,
)
from zephon._internal.io.resolvers import DirectResolver
from zephon._internal.io.storage import LocalFSBackend
from zephon._internal.io.storage.base import StorageBackend
from zephon._internal.io.storage.s3 import S3Backend
from zephon._internal.io.types import LocalShardFile, LocalShardRef
from zephon.io.dataset import Dataset


@pytest.mark.parametrize("loader_kind", ["pytree", "tokens"])
@pytest.mark.parametrize("compression", [False, True])
def test_litdata_reader_handles_dataset(
    tmp_path: Path, loader_kind: str, compression: bool
) -> None:
    dataset_dir = (
        tmp_path / f"litdata_{loader_kind}_{'compressed' if compression else 'plain'}"
    )
    kwargs: dict[str, object] = {"loader": loader_kind}
    block_size: int | None = None
    if loader_kind == "tokens":
        block_size = 4
        samples = [
            np.arange(i * block_size, (i + 1) * block_size, dtype=np.int32)
            for i in range(4)
        ]
    else:
        samples = [(idx, f"sample-{idx}") for idx in range(4)]
    compression_value = "zstd:3" if compression else None
    writer = _binary_writer_for_samples(
        dataset_dir,
        len(samples),
        loader_kind,
        block_size=block_size,
        compression=compression_value,
    )
    for idx, sample in enumerate(samples):
        writer.add_item(idx, sample)
    writer.done()
    writer.merge()

    dataset = Dataset.from_path(name="lit", path=str(dataset_dir))
    ensure_builtin_formats(required={"litdata"})
    handler = get_format("litdata")
    _, locators = catalog_locators(dataset)
    assert dataset.path is not None
    resolver = DirectResolver(LocalFSBackend(root=Path(dataset.path)))

    observed: list[tuple[int, str]] = []
    counts = counts_dict(dataset)
    for shard_id, locator in locators.items():
        local_ref = resolver.resolve(locator)
        chunk_meta: dict[str, object] = {}
        config_meta: dict[str, object] = {}
        if isinstance(locator.extra, Mapping):
            candidate = locator.extra.get("chunk")
            if isinstance(candidate, Mapping):
                chunk_meta = dict(candidate)
            config_candidate = locator.extra.get("config")
            if isinstance(config_candidate, Mapping):
                config_meta = dict(config_candidate)
        if compression:
            assert locator.compression == "zstd:3"
            assert locator.zip is not None
            assert str(config_meta.get("compression")) == "zstd:3"
            assert config_meta.get("compression_level") in {None, "chunk"}
            assert locator.zip.basename != locator.raw.basename
            # Verify the fixture is a whole-file Zstd frame, not merely labelled
            # compressed in its index (a regression in newer LitData writers).
            with (dataset_dir / locator.zip.basename).open("rb") as compressed_file:
                assert compressed_file.read(4) == b"\x28\xb5\x2f\xfd"
        else:
            assert locator.compression is None
            assert locator.zip is None
            assert not config_meta.get("compression")
        shard = handler.open_shard(locator, local_ref)
        try:
            assert len(shard) == counts[shard_id]
            for row_idx in range(len(shard)):
                row = shard[row_idx]
                if loader_kind == "tokens":
                    assert isinstance(row, np.ndarray)
                    observed.append(np.array(row))
                else:
                    observed.append((int(row[0]), str(row[1])))
        finally:
            shard.close()

    if loader_kind == "tokens":
        for expected, actual in zip(samples, observed, strict=True):
            assert isinstance(actual, np.ndarray)
            assert np.array_equal(actual, expected)
    else:
        expected = [(idx, f"sample-{idx}") for idx in range(len(samples))]
        assert observed == expected


def test_litdata_reader_handles_dict_with_numpy_ints(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "litdata_numpy_dict"
    samples = [
        {"payload": np.arange(i * 5, (i + 1) * 5, dtype=np.int32)} for i in range(3)
    ]
    writer = _binary_writer_for_samples(dataset_dir, len(samples), loader_kind="pytree")
    for idx, sample in enumerate(samples):
        writer.add_item(idx, sample)
    writer.done()
    writer.merge()

    dataset = Dataset.from_path(name="lit-numpy", path=str(dataset_dir))
    ensure_builtin_formats(required={"litdata"})
    handler = get_format("litdata")
    _, locators = catalog_locators(dataset)
    assert dataset.path is not None
    resolver = DirectResolver(LocalFSBackend(root=Path(dataset.path)))

    observed: list[Mapping[str, object]] = []
    counts = counts_dict(dataset)
    for shard_id, locator in locators.items():
        local_ref = resolver.resolve(locator)
        shard = handler.open_shard(locator, local_ref)
        try:
            assert len(shard) == counts[shard_id]
            for row_idx in range(len(shard)):
                row = shard[row_idx]
                assert isinstance(row, Mapping)
                observed.append(row)
        finally:
            shard.close()

    for expected, actual in zip(samples, observed, strict=True):
        assert set(actual.keys()) == {"payload"}
        value = actual["payload"]
        assert isinstance(value, np.ndarray)
        assert value.dtype == expected["payload"].dtype
        assert np.array_equal(value, expected["payload"])


def test_litdata_reader_handles_no_header_numpy(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "litdata_no_header_numpy"
    samples = [
        {"payload": np.arange(i * 4, (i + 1) * 4, dtype=np.uint32)} for i in range(3)
    ]
    writer = BinaryWriter(cache_dir=str(dataset_dir), chunk_size=len(samples))
    for idx, sample in enumerate(samples):
        writer.add_item(idx, sample)
    writer.done()
    writer.merge()

    dataset = Dataset.from_path(name="lit-no-header", path=str(dataset_dir))
    ensure_builtin_formats(required={"litdata"})
    handler = get_format("litdata")
    _, locators = catalog_locators(dataset)
    assert dataset.path is not None
    resolver = DirectResolver(LocalFSBackend(root=Path(dataset.path)))

    observed: list[Mapping[str, object]] = []
    counts = counts_dict(dataset)
    for shard_id, locator in locators.items():
        local_ref = resolver.resolve(locator)
        shard = handler.open_shard(locator, local_ref)
        try:
            assert len(shard) == counts[shard_id]
            for row_idx in range(len(shard)):
                row = shard[row_idx]
                assert isinstance(row, Mapping)
                observed.append(row)
        finally:
            shard.close()

    for expected, actual in zip(samples, observed, strict=True):
        value = actual["payload"]
        assert isinstance(value, np.ndarray)
        assert value.dtype == expected["payload"].dtype
        assert np.array_equal(value, expected["payload"])


def test_discover_from_files_s3_does_not_download_full_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test cloud discovery does not rely on full-object downloads.

    When index.json is absent, LitData discover() uses read_range() to read
    only chunk headers (~40KB per chunk) instead of downloading full .bin files.
    """
    pytest.importorskip("torch")

    state = _install_obstore_stubs(monkeypatch)

    dataset_dir = tmp_path / "litdata_dataset"
    dataset_dir.mkdir()
    block_size = 4
    samples = [
        np.arange(i * block_size, (i + 1) * block_size, dtype=np.int32)
        for i in range(4)
    ]
    writer = _binary_writer_for_samples(
        dataset_dir, len(samples), "tokens", block_size=block_size
    )
    for idx, sample in enumerate(samples):
        writer.add_item(idx, sample)
    writer.done()
    writer.merge()

    # Use only the first chunk file; do NOT add index.json to simulate fallback
    chunk_path = dataset_dir / "chunk-0-0.bin"
    assert chunk_path.exists()
    state["objects"][("bucket", "dataset/chunk-0-0.bin")] = chunk_path.read_bytes()

    backend = S3Backend()

    def fail_download(src: str, dst: str, timeout: float | None = None) -> None:
        del dst, timeout
        if "index.json" in src:
            raise FileNotFoundError(f"Missing LitData index: {src}")
        pytest.fail("discover() performed a full-object download")

    backend.download = fail_download  # type: ignore[assignment]

    format_handler = LitDataFormat()
    shard_index, shard_meta = format_handler.discover("s3://bucket/dataset", backend)

    # Discovery succeeded using only read_range (no full-object download)
    assert len(shard_index) == 1
    assert len(shard_meta) == 1
    assert shard_meta[0]["chunk"]["filename"] == "chunk-0-0.bin"


def test_discover_from_files_reads_large_headers_via_range_reads(
    tmp_path: Path,
) -> None:
    """Large chunk headers (many items) are read via targeted range reads, not full-file reads."""
    pytest.importorskip("torch")

    dataset_dir = tmp_path / "litdata_large_header"
    dataset_dir.mkdir()
    block_size = 4
    num_samples = 4
    samples = [
        np.arange(i * block_size, (i + 1) * block_size, dtype=np.int32)
        for i in range(num_samples)
    ]
    writer = _binary_writer_for_samples(
        dataset_dir, num_samples, "tokens", block_size=block_size
    )
    for idx, sample in enumerate(samples):
        writer.add_item(idx, sample)
    writer.done()
    writer.merge()

    (dataset_dir / "index.json").unlink(missing_ok=True)

    class _TrackingStorage(StorageBackend):
        """Delegates to LocalFSBackend but records all read_range calls."""

        def __init__(self, root: Path):
            self._local = LocalFSBackend(root=root)
            self.range_reads: list[tuple[str, int, int | None]] = []

        def read_range(
            self,
            path: str,
            start: int,
            *,
            end: int | None = None,
            length: int | None = None,
        ) -> bytes:
            self.range_reads.append((path, start, length))
            return self._local.read_range(path, start, end=end, length=length)

        def open(self, path: str, mode: str = "rb", **kwargs):
            return self._local.open(path, mode, **kwargs)

        def exists(self, path: str) -> bool:
            return self._local.exists(path)

        def download(self, src: str, dst: str, timeout: float | None = None) -> None:
            return self._local.download(src, dst, timeout)

        def listdir(self, path: str) -> list[str]:
            return self._local.listdir(path)

        def stat(self, path: str) -> Mapping:
            return self._local.stat(path)

        def put(self, path: str, data: bytes) -> None:
            return self._local.put(path, data)

        def delete(self, path: str) -> None:
            return self._local.delete(path)

        def glob(self, pattern: str) -> list[str]:
            return self._local.glob(pattern)

        def mkdir(
            self, path: str, parents: bool = False, exist_ok: bool = False
        ) -> None:
            return self._local.mkdir(path, parents, exist_ok)

    backend = _TrackingStorage(tmp_path)
    format_handler = LitDataFormat()
    shard_index, shard_meta = format_handler.discover(
        str(dataset_dir.relative_to(tmp_path)), backend
    )

    assert len(shard_index) >= 1
    assert shard_meta[0]["chunk"]["filename"] == "chunk-0-0.bin"

    # Every range read should be a targeted header read (4 bytes for num_items,
    # then the offset table), never a full-file read.
    for path, start, length in backend.range_reads:
        if not path.endswith(".bin"):
            continue
        file_size = int(backend._local.stat(path).get("size", 0))
        assert length is not None and length < file_size, (
            f"read_range fetched the full file ({length} bytes) for {path}; "
            "expected a targeted header-only read"
        )


def _binary_writer_for_samples(
    out: Path,
    chunk_size: int,
    loader_kind: str,
    *,
    block_size: int | None = None,
    compression: str | None = None,
) -> BinaryWriter:
    kwargs: dict[str, object] = {"cache_dir": str(out), "chunk_size": chunk_size}
    if compression:
        kwargs["compression"] = compression
        # Exercise whole-chunk compression for both pytree and token fixtures.
        # LitData 0.2.74+ defaults to batch compression, which rejects TokensLoader;
        # earlier writers do not accept the compression_level keyword.
        if "compression_level" in inspect.signature(BinaryWriter).parameters:
            kwargs["compression_level"] = "chunk"
    if loader_kind == "tokens":
        pytest.importorskip("torch")
        from litdata.streaming.item_loader import TokensLoader as StreamingTokensLoader

        if block_size is None:
            raise ValueError("block_size must be provided for tokens loader")
        kwargs["item_loader"] = StreamingTokensLoader(block_size=block_size)
    writer = BinaryWriter(**kwargs)
    if compression:
        _ensure_single_thread_zstd(writer)
    return writer


def test_litdata_shard_getsamples_batch_loading(tmp_path: Path) -> None:
    """Test that getsamples correctly loads multiple items in batch."""
    dataset_dir = tmp_path / "litdata_getsamples"
    samples = [(idx, f"sample-{idx}") for idx in range(6)]
    writer = _binary_writer_for_samples(dataset_dir, len(samples), loader_kind="pytree")
    for idx, sample in enumerate(samples):
        writer.add_item(idx, sample)
    writer.done()
    writer.merge()

    dataset = Dataset.from_path(name="lit-getsamples", path=str(dataset_dir))
    ensure_builtin_formats(required={"litdata"})
    handler = get_format("litdata")
    _, locators = catalog_locators(dataset)
    assert dataset.path is not None
    resolver = DirectResolver(LocalFSBackend(root=Path(dataset.path)))

    for locator in locators.values():
        local_ref = resolver.resolve(locator)
        shard = handler.open_shard(locator, local_ref)
        try:
            # Test batch loading with unsorted and duplicate indices
            out = shard.getsamples([4, 1, 4, 0])
            assert len(out) == 4
            assert [int(r[0]) for r in out] == [4, 1, 4, 0]
            assert [str(r[1]) for r in out] == [
                "sample-4",
                "sample-1",
                "sample-4",
                "sample-0",
            ]

            # Test empty list
            assert shard.getsamples([]) == []

            # Test out of bounds
            with pytest.raises(IndexError):
                _ = shard.getsamples([10])

            # Test that single item works
            single = shard.getsamples([2])
            assert len(single) == 1
            assert int(single[0][0]) == 2
            assert str(single[0][1]) == "sample-2"

            # Test loading all items
            all_indices = list(range(len(shard)))
            all_items = shard.getsamples(all_indices)
            assert len(all_items) == len(shard)
            for idx, item in enumerate(all_items):
                assert int(item[0]) == idx
                assert str(item[1]) == f"sample-{idx}"
        finally:
            shard.close()


def test_litdata_codec_rejects_non_constant_config() -> None:
    """``encode`` fails loud when shards carry msgpack-distinct configs."""
    codec = _LitDataExtraCodec()
    metas = [
        {"config": {"data_spec": None, "chunk_bytes": 1}},
        {"config": {"data_spec": None, "chunk_bytes": 2}},
    ]
    with pytest.raises(ValueError, match="not constant across shards"):
        codec.encode(metas, np.array([1, 1], dtype=np.int64))


def test_litdata_codec_rejects_partial_config() -> None:
    """``config`` on some shards but not all would be fabricated at decode."""
    codec = _LitDataExtraCodec()
    metas = [
        {"chunk": {"chunk_size": 1}},
        {"config": {"data_spec": None}, "chunk": {"chunk_size": 1}},
    ]
    with pytest.raises(ValueError, match="some shards but not all"):
        codec.encode(metas, np.array([1, 1], dtype=np.int64))


def test_litdata_codec_rejects_partial_interval() -> None:
    """``interval`` on some shards but not all would be fabricated at decode."""
    from zephon._internal.io.formats.litdata_support.support import Interval

    codec = _LitDataExtraCodec()
    metas = [
        {"interval": Interval(0, 0, 4, 4)},
        {"chunk": {"chunk_size": 1}},
    ]
    with pytest.raises(ValueError, match="some shards but not all"):
        codec.encode(metas, np.array([4, 1], dtype=np.int64))


def _ensure_single_thread_zstd(writer: BinaryWriter) -> None:
    compressor = getattr(writer, "_compressor", None)
    name = getattr(compressor, "name", "") if compressor is not None else ""
    if not compressor or "zstd" not in str(name).lower():
        return
    try:
        import zstd  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return
    level = getattr(compressor, "level", None)
    if level is None:
        return
    compress = zstd.compress

    def _patched(data: bytes, *, _level: int = level, _compress=compress) -> bytes:
        return _compress(data, _level, 1)

    compressor.compress = _patched


def _open_shard(root: Path, shard_index: int = 0) -> _LitDataShard:
    # Exercise the format handler directly, without catalog or resolver behavior.
    handler = LitDataFormat()
    _, metadata = handler.discover(str(root), LocalFSBackend(root=root))
    dataset = Dataset("lit", {"kind": "litdata", "path": str(root), "shards": metadata})
    locator = handler.build_locators(dataset)[shard_index]
    path = root / locator.raw.basename
    shard = handler.open_shard(
        locator, LocalShardRef(raw=LocalShardFile(path, path.stat().st_size))
    )
    assert isinstance(shard, _LitDataShard)
    return shard


@pytest.mark.parametrize("layout", ["legacy", "file", "stream", "hybrid"])
def test_dispatch_uses_file_layout(tmp_path: Path, layout: str) -> None:
    if layout != "legacy":
        pytest.importorskip("pyarrow")
    rows = [{"id": i, "text": f"row-{i}"} for i in range(6)]
    write_litdata_fixture(tmp_path, rows, layout=layout)
    shard = _open_shard(tmp_path)
    try:
        assert type(shard) is _LitDataShard
        assert type(shard._loader) is (
            PyTreeLoader if layout == "legacy" else ArrowLoader
        )
        assert len(shard) == len(rows)
        assert shard[3] == rows[3]
        indices = [5, 0, 2, 5, 1]
        assert shard.getsamples(indices) == [rows[i] for i in indices]
        assert shard.getsamples([]) == []
        for index in [-1, len(rows)]:
            with pytest.raises(IndexError):
                shard[index]
            with pytest.raises(IndexError):
                shard.getsamples([0, index])
    finally:
        shard.close()
    shard.close()  # Repeated cleanup remains harmless for both layouts.


@pytest.mark.parametrize(
    "layouts",
    [("legacy", "file"), ("file", "legacy"), ("file", "file"), ("legacy", "legacy")],
)
def test_chunks_share_shard_and_intervals(
    tmp_path: Path, layouts: tuple[str, str]
) -> None:
    pytest.importorskip("pyarrow")
    chunks = []
    expected = []
    for group, layout in enumerate(layouts):
        rows = [{"id": group * 10 + i, "text": f"{layout}-{i}"} for i in range(3)]
        source = tmp_path / str(group)
        path = write_litdata_fixture(source, rows, layout=layout)
        index = json.loads((source / "index.json").read_text())
        chunk = index["chunks"][0]
        chunk["filename"] = f"chunk-{group}.bin"
        path.rename(tmp_path / chunk["filename"])
        chunks.append(chunk)
        expected.append(rows)
    config = index["config"]
    (tmp_path / "index.json").write_text(
        json.dumps({"config": config, "chunks": chunks})
    )

    # The second shard has a nonzero global interval, regardless of its layout.
    for group, rows in enumerate(expected):
        shard = _open_shard(tmp_path, group)
        try:
            assert shard[1] == rows[1]
            assert shard.getsamples([2, 0, 2]) == [rows[i] for i in [2, 0, 2]]
        finally:
            shard.close()


def test_arrow_discovery_does_not_need_new_litdata_serializers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pyarrow")
    rows = [{"id": i, "text": str(i)} for i in range(3)]
    write_litdata_fixture(tmp_path, rows, layout="file")
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text())
    index["config"]["data_format"] = ["new_serializer_not_installed"]
    index["config"]["return_flat_leaves"] = True
    index_path.write_text(json.dumps(index))

    def fail() -> None:
        pytest.fail(
            "Arrow discovery/reading attempted to initialize LitData serializers"
        )

    monkeypatch.setattr(dependencies, "_get_serializers", fail)
    shard = _open_shard(tmp_path)
    try:
        result = shard.getsamples([2, 0, 2])
        assert [row.materialize() for row in result] == [rows[i] for i in [2, 0, 2]]
    finally:
        shard.close()


def test_legacy_read_does_not_require_pyarrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [{"id": i, "text": str(i)} for i in range(3)]
    write_litdata_fixture(tmp_path, rows)

    def fail() -> None:
        pytest.fail("Legacy read attempted to import PyArrow")

    monkeypatch.setattr(arrow, "require_pyarrow", fail)
    shard = _open_shard(tmp_path)
    try:
        assert shard.getsamples([2, 0, 2]) == [rows[i] for i in [2, 0, 2]]
    finally:
        shard.close()


@pytest.mark.parametrize("loader_spec", ["TokensLoader", {"name": "tokens"}])
def test_tokens_shard_open_does_not_access_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader_spec: object
) -> None:
    path = tmp_path / "not-yet-downloaded.bin"
    config = {
        "item_loader": loader_spec,
        "block_size": 4,
        "data_format": ["no_header_tensor:0"],
    }
    chunk = {"chunk_size": 4, "dim": 16, "chunk_bytes": 100}

    def fail(*args: object, **kwargs: object) -> None:
        pytest.fail("Token shard opening must not inspect the chunk file")

    # Initialize optional packages before checking the chunk's lazy-open path.
    dependencies.ensure_litdata_deps()
    monkeypatch.setattr(arrow, "arrow_footer_span", fail)
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(Path, "stat", fail)
    shard = _LitDataShard(
        str(tmp_path), config, chunk, LocalShardRef(LocalShardFile(path, 100))
    )
    try:
        assert isinstance(shard._loader, TokensLoader)
        assert len(shard) == 4
    finally:
        shard.close()


@pytest.mark.parametrize("kind", ["numpy", "tensor"])
def test_token_blocks_own_only_their_bytes_and_survive_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    torch = pytest.importorskip("torch")
    dependencies.ensure_litdata_deps()
    # The first item has no complete block. Subsequent items have two and
    # three blocks, respectively, exercising repeated cumulative offsets.
    items = [np.arange(n, dtype=np.int32) + i * 100 for i, n in enumerate([2, 8, 12])]
    header_bytes = (len(items) + 2) * 4
    offsets = np.cumsum(
        [header_bytes, *(item.nbytes for item in items)], dtype=np.uint32
    )
    path = tmp_path / "tokens.bin"
    path.write_bytes(
        np.array([len(items)], dtype=np.uint32).tobytes()
        + offsets.tobytes()
        + b"".join(item.tobytes() for item in items)
    )
    mapping = (
        dependencies._NUMPY_DTYPES_MAPPING
        if kind == "numpy"
        else dependencies._TORCH_DTYPES_MAPPING
    )
    dtype = np.dtype("int32") if kind == "numpy" else torch.int32
    dtype_index = next(key for key, value in mapping.items() if value == dtype)
    shard = _LitDataShard(
        str(tmp_path),
        {
            "item_loader": "TokensLoader",
            "block_size": 4,
            "data_format": [f"no_header_{kind}:{dtype_index}"],
        },
        {"chunk_size": len(items), "dim": 22, "chunk_bytes": path.stat().st_size},
        LocalShardRef(LocalShardFile(path, path.stat().st_size)),
    )
    loader = shard._loader
    # Build the lookup once, then forbid recomputing it for every block.
    loader._load_chunk(0, str(path))

    def reject_cumsum(*args: object, **kwargs: object) -> None:
        pytest.fail("Block lookup must reuse the chunk's cumulative index")

    monkeypatch.setattr(np, "cumsum", reject_cumsum)
    indices = [4, 0, 2, 4, 1]
    expected_blocks = [
        items[1][:4],
        items[1][4:],
        items[2][:4],
        items[2][4:8],
        items[2][8:],
    ]
    outputs = shard.getsamples(indices)
    if kind == "numpy":
        assert all(len(row.base) == 4 * 4 for row in outputs)
    else:
        assert all(row.untyped_storage().nbytes() == 4 * 4 for row in outputs)
    # Eviction explicitly closes the mmap; returned values must remain valid.
    loader.delete(0, str(path))
    assert not path.exists()
    for row, index in zip(outputs, indices, strict=True):
        np.testing.assert_array_equal(np.asarray(row), expected_blocks[index])


def test_truncated_arrow_chunk_does_not_dispatch_to_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pyarrow")
    path = write_litdata_fixture(tmp_path, [{"id": 0, "text": "row"}], layout="file")
    path.write_bytes(path.read_bytes()[:-12])

    def fail() -> None:
        pytest.fail("Incomplete Arrow chunk was dispatched to the binary decoder")

    monkeypatch.setattr(dependencies, "_get_serializers", fail)
    with pytest.raises(FileNotFoundError, match="not found or incomplete"):
        _open_shard(tmp_path)


def test_binary_serializer_validation_happens_at_shard_open(tmp_path: Path) -> None:
    write_litdata_fixture(tmp_path, [{"id": 0, "text": "row"}])
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text())
    index["config"]["data_format"] = ["unknown_serializer"]
    index_path.write_text(json.dumps(index))
    handler = LitDataFormat()
    sizes, _ = handler.discover(str(tmp_path), LocalFSBackend(root=tmp_path))
    assert sizes == {0: 1}
    with pytest.raises(KeyError, match="unknown_serializer"):
        _open_shard(tmp_path)
