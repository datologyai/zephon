"""Unit tests for binary pytree and token chunk readers."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("optree")
pytest.importorskip("litdata")
import optree
from litdata.streaming.item_loader import TokensLoader as StreamingTokensLoader
from litdata.streaming.writer import BinaryWriter

from tests.helpers.litdata_chunks import write_litdata_fixture
from zephon._internal.io.formats.litdata_support.dependencies import _get_serializers
from zephon._internal.io.formats.litdata_support.pytree import (
    PyTreeLoader,
    TokensLoader,
)
from zephon._internal.io.formats.litdata_support.support import FlatPyTree


def test_pytree_loader_deserialize_roundtrip():
    config = {
        "data_format": ["str", "int"],
        "data_spec": optree.tree_structure(("a", "b")),
    }
    chunks = [{"chunk_size": 2}, {"chunk_size": 3}]
    region = [(0, 1), (1, 3)]

    loader = PyTreeLoader()
    serializers = _get_serializers()
    loader.setup(config, chunks, serializers, region)

    payloads: list[bytes] = []
    sizes: list[int] = []
    for fmt, value in zip(config["data_format"], ("payload", 7), strict=True):
        serializer = loader._serializers[fmt]  # type: ignore[attr-defined]
        serialized, _ = serializer.serialize(value)
        payloads.append(serialized)
        sizes.append(len(serialized))

    encoded, _ = PyTreeLoader.encode_data(payloads, sizes, ["payload", 7])
    restored = loader.deserialize(encoded, chunk_index=0)
    assert restored == ("payload", 7)


def test_pytree_loader_flat_mode_returns_lazy_bundle():
    config = {
        "data_format": ["str", "int"],
        "data_spec": optree.tree_structure(("a", "b")),
        "return_flat_leaves": True,
    }
    chunk = {"chunk_size": 1}

    loader = PyTreeLoader(return_flat_leaves=True)
    serializers = _get_serializers()
    loader.setup(config, [chunk], serializers, None)

    payloads: list[bytes] = []
    sizes: list[int] = []
    for fmt, value in zip(config["data_format"], ("payload", 7), strict=True):
        serializer = loader._serializers[fmt]  # type: ignore[attr-defined]
        serialized, _ = serializer.serialize(value)
        payloads.append(serialized)
        sizes.append(len(serialized))

    encoded, _ = PyTreeLoader.encode_data(payloads, sizes, ["payload", 7])
    flat = loader.deserialize(encoded, chunk_index=0)
    assert isinstance(flat, FlatPyTree)
    assert flat.leaves == ["payload", 7]
    assert flat.materialize() == ("payload", 7)


def test_tokens_loader_reads_blocks(tmp_path):
    pytest.importorskip("torch")

    block_size = 4
    dataset_dir = tmp_path / "tokens"
    samples = [
        np.arange(i * block_size, (i + 1) * block_size, dtype=np.int32)
        for i in range(3)
    ]

    writer = BinaryWriter(
        cache_dir=str(dataset_dir),
        chunk_size=len(samples),
        item_loader=StreamingTokensLoader(block_size=block_size),
    )
    for idx, sample in enumerate(samples):
        writer.add_item(idx, sample)
    writer.done()
    writer.merge()

    index_data = json.loads((dataset_dir / "index.json").read_text())
    config = index_data["config"]
    chunks = index_data["chunks"]

    loader = TokensLoader(block_size=block_size)
    serializers = _get_serializers()
    loader.setup(config, chunks, serializers, None)

    intervals = loader.generate_intervals()
    assert intervals[-1].chunk_end == len(samples)
    assert len(intervals) == len(index_data["chunks"])

    for chunk_index, (chunk_entry, interval) in enumerate(
        zip(index_data["chunks"], intervals, strict=True)
    ):
        chunk_basename = chunk_entry.get("chunk_path") or chunk_entry.get("filename")
        assert chunk_basename is not None
        chunk_path = dataset_dir / chunk_basename
        for block_idx in range(interval.chunk_start, interval.chunk_end):
            item = loader.load_item_from_chunk(
                block_idx,
                chunk_index=chunk_index,
                chunk_filepath=str(chunk_path),
                begin=interval.chunk_start,
                filesize_bytes=int(chunk_entry["chunk_bytes"]),
            )
            assert isinstance(item, np.ndarray)
            assert np.array_equal(item, samples[block_idx])

        loader.close(chunk_index)


def test_pytree_loader_reuses_files_with_global_intervals(tmp_path: Path) -> None:
    paths = []
    chunks = []
    expected = []
    for group in range(2):
        rows = [{"id": group * 10 + i, "text": f"row-{group}-{i}"} for i in range(3)]
        path = write_litdata_fixture(tmp_path / str(group), rows, layout="legacy")
        paths.append(path)
        chunks.append({"chunk_size": len(rows), "chunk_bytes": path.stat().st_size})
        expected.append(rows)
    loader = PyTreeLoader()
    loader.setup(
        {
            "data_format": ["int", "str"],
            "data_spec": optree.tree_structure(expected[0][0]),
        },
        chunks,
        _get_serializers(),
    )
    try:
        for group in [0, 1, 0]:
            begin = 3 * group
            rows = loader.load_items_from_chunk(
                [begin + i for i in [2, 0, 2]],
                group,
                str(paths[group]),
                begin,
                chunks[group]["chunk_bytes"],
            )
            assert rows == [expected[group][i] for i in [2, 0, 2]]
    finally:
        loader.close(0)
