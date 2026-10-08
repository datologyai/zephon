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
from zephon._internal.io.formats.litdata_support import dependencies
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
    loader = TokensLoader(block_size=4)
    loader.setup(
        {
            "item_loader": "TokensLoader",
            "block_size": 4,
            "data_format": [f"no_header_{kind}:{dtype_index}"],
        },
        [{"chunk_size": len(items), "dim": 22, "chunk_bytes": path.stat().st_size}],
        _get_serializers(),
    )
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
    outputs = [
        loader.load_item_from_chunk(index, 0, str(path), 0, path.stat().st_size)
        for index in indices
    ]
    if kind == "numpy":
        assert all(len(row.base) == 4 * 4 for row in outputs)
    else:
        assert all(row.untyped_storage().nbytes() == 4 * 4 for row in outputs)
    # Eviction explicitly closes the mmap; returned values must remain valid.
    loader.delete(0, str(path))
    assert not path.exists()
    for row, index in zip(outputs, indices, strict=True):
        np.testing.assert_array_equal(np.asarray(row), expected_blocks[index])
