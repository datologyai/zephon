from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from litdata.streaming.writer import BinaryWriter

from zephon.io.dataset import Dataset
from zephon.io.formats import ensure_builtin_formats
from zephon.io.formats.base import get_format
from zephon.io.resolvers import DirectResolver
from zephon.io.storage import LocalFSBackend


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
    locators = handler.build_locators(dataset)
    assert dataset.path is not None
    resolver = DirectResolver(LocalFSBackend(root=Path(dataset.path)))

    observed: list[tuple[int, str]] = []
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
        else:
            assert locator.compression is None
            assert locator.zip is None
            assert not config_meta.get("compression")
        shard = handler.open_shard(locator, local_ref)
        try:
            assert len(shard) == dataset.shard_index[shard_id]
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
    locators = handler.build_locators(dataset)
    assert dataset.path is not None
    resolver = DirectResolver(LocalFSBackend(root=Path(dataset.path)))

    observed: list[Mapping[str, object]] = []
    for shard_id, locator in locators.items():
        local_ref = resolver.resolve(locator)
        shard = handler.open_shard(locator, local_ref)
        try:
            assert len(shard) == dataset.shard_index[shard_id]
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
    locators = handler.build_locators(dataset)
    assert dataset.path is not None
    resolver = DirectResolver(LocalFSBackend(root=Path(dataset.path)))

    observed: list[Mapping[str, object]] = []
    for shard_id, locator in locators.items():
        local_ref = resolver.resolve(locator)
        shard = handler.open_shard(locator, local_ref)
        try:
            assert len(shard) == dataset.shard_index[shard_id]
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
    locators = handler.build_locators(dataset)
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
