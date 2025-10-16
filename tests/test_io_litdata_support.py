import json

import numpy as np
import optree
import pytest
from litdata.streaming.item_loader import TokensLoader as StreamingTokensLoader
from litdata.streaming.writer import BinaryWriter

from zephon.io.formats.litdata_support import (
    PyTreeLoader,
    Serializer,
    TokensLoader,
    _get_serializers,
    treespec_dumps,
    treespec_loads,
)


class _EchoSerializer(Serializer):
    def serialize(self, data):
        return str(data).encode("utf-8"), None

    def deserialize(self, data):
        return data.decode("utf-8")

    def can_serialize(self, data):
        return True


def test_litdata_serializers_roundtrip_basic_types():
    serializers = _get_serializers()
    samples = {
        "str": "hello world",
        "bool": True,
        "int": 123,
        "float": 3.14159,
        "bytes": b"\x00\x01payload",
        "numpy": np.arange(5, dtype=np.float32),
        "pickle": {"alpha": 1, "beta": [1, 2, 3]},
    }

    for key, value in samples.items():
        serializer = serializers[key]
        payload, _ = serializer.serialize(value)
        restored = serializer.deserialize(payload)
        if isinstance(value, np.ndarray):
            assert isinstance(restored, np.ndarray)
            assert np.array_equal(restored, value)
        else:
            assert restored == value


def test_litdata_serializers_can_be_overridden():
    override = {"str": _EchoSerializer()}
    serializers = _get_serializers(override)
    assert isinstance(serializers["str"], _EchoSerializer)
    payload, _ = serializers["str"].serialize(42)
    assert serializers["str"].deserialize(payload) == "42"


def test_pytree_loader_intervals_and_deserialize_roundtrip():
    config = {
        "data_format": ["str", "int"],
        "data_spec": optree.tree_structure(("a", "b")),
    }
    chunks = [{"chunk_size": 2}, {"chunk_size": 3}]
    region = [(0, 1), (1, 3)]

    loader = PyTreeLoader()
    serializers = _get_serializers()
    loader.setup(config, chunks, serializers, region)

    intervals = loader.generate_intervals()
    assert intervals[0].chunk_start == 0
    assert intervals[0].roi_start_idx == 0
    assert intervals[0].roi_end_idx == 1
    assert intervals[0].chunk_end == 2

    assert intervals[1].chunk_start == 2
    assert intervals[1].roi_start_idx == 3
    assert intervals[1].roi_end_idx == 5
    assert intervals[1].chunk_end == 5

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


@pytest.mark.parametrize(
    "structure",
    [
        ("a", 1),
        {"x": [1, 2], "y": {"z": 3}},
        [{"alpha": 1}, {"beta": (2, 3)}],
    ],
)
def test_treespec_roundtrip(structure):
    spec = optree.tree_structure(structure)
    serialized = treespec_dumps(spec)
    restored_spec = treespec_loads(serialized)
    leaves = optree.tree_leaves(structure)
    reconstructed = optree.tree_unflatten(restored_spec, leaves)
    assert reconstructed == optree.tree_unflatten(spec, leaves)
