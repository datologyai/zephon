import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from tests._catalog_helpers import attach_catalog, catalog_locators
from tests._helpers import counts_dict, zstd
from tests.helpers.storage import _install_obstore_stubs
from zephon.io.dataset import Dataset


@pytest.mark.parametrize("backend_kind", ["local", "s3", "gcs"])  # MDS discovery
def test_dataset_from_path_mds_parametrized(
    backend_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = {
        "shards": [
            {
                "samples": 3,
                "raw": {
                    "basename": "shard0.mds",
                    "bytes": 12,
                    "hashes": {"md5": "abc"},
                },
            }
        ]
    }

    if backend_kind == "local":
        (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
        path = str(tmp_path)
    elif backend_kind == "s3":
        state = _install_obstore_stubs(monkeypatch)
        state["objects"][("bucket", "dataset/index.json")] = json.dumps(index).encode(
            "utf-8"
        )
        path = "s3://bucket/dataset"
    else:  # gcs
        state = _install_obstore_stubs(monkeypatch)
        state["objects"][("bucket", "dataset/index.json")] = json.dumps(index).encode(
            "utf-8"
        )
        path = "gs://bucket/dataset"

    dataset = Dataset.from_path("demo", path)
    assert dataset.backend["kind"] == "mds"
    assert counts_dict(dataset) == {0: 3}
    _, locators = catalog_locators(dataset)
    assert locators[0].raw.basename == "shard0.mds"


@pytest.mark.parametrize("backend_kind", ["local", "s3", "gcs"])  # JSONL discovery
def test_dataset_from_path_jsonl_parametrized(
    backend_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if backend_kind == "local":
        root = tmp_path
        (root / "shard0.jsonl").write_text('{"id":1}\n{"id":2}\n', encoding="utf-8")
        (root / "shard1.jsonl").write_text('{"id":3}\n{"id":4}\n', encoding="utf-8")
        path = str(root)
    elif backend_kind == "s3":
        state = _install_obstore_stubs(monkeypatch)
        state["objects"][("bucket", "json/shard0.jsonl")] = b'{"id":1}\n{"id":2}\n'
        state["objects"][("bucket", "json/shard1.jsonl")] = b'{"id":3}\n{"id":4}\n'
        path = "s3://bucket/json"
    else:  # gcs
        state = _install_obstore_stubs(monkeypatch)
        state["objects"][("bucket", "json/shard0.jsonl")] = b'{"id":1}\n{"id":2}\n'
        state["objects"][("bucket", "json/shard1.jsonl")] = b'{"id":3}\n{"id":4}\n'
        path = "gs://bucket/json"

    dataset = Dataset.from_path("jsonl", path)
    assert dataset.backend["kind"] == "jsonl"
    assert counts_dict(dataset) == {0: 2, 1: 2}
    _, locators = catalog_locators(dataset)
    assert locators[0].raw.basename.endswith("shard0.jsonl")


def test_dataset_from_path_remote_gcs_jsonl(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_obstore_stubs(monkeypatch)
    state["objects"][("bucket", "json/shard0.jsonl")] = '{"id": 1}\n{"id": 2}\n'.encode(
        "utf-8"
    )
    state["objects"][("bucket", "json/shard1.jsonl")] = '{"id": 3}\n{"id": 4}\n'.encode(
        "utf-8"
    )

    dataset = Dataset.from_path("remote-jsonl", "gs://bucket/json")

    assert dataset.backend["kind"] == "jsonl"
    assert counts_dict(dataset) == {0: 2, 1: 2}
    _, locators = catalog_locators(dataset)
    assert locators[0].raw.basename == "shard0.jsonl"


@pytest.mark.parametrize("backend_kind", ["local", "s3"])
def test_dataset_from_path_detects_compressed_jsonl(
    backend_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shards = {
        "shard0.jsonl.gz": gzip.compress(b'{"id":1}\n{"id":2}\n'),
        "shard1.jsonl.zst": zstd.compress(b'{"id":3}\n'),
    }
    if backend_kind == "local":
        for name, data in shards.items():
            (tmp_path / name).write_bytes(data)
        path = str(tmp_path)
    else:
        state = _install_obstore_stubs(monkeypatch)
        for name, data in shards.items():
            state["objects"][("bucket", f"json/{name}")] = data
        path = "s3://bucket/json"

    dataset = Dataset.from_path("compressed", path)

    assert dataset.backend["kind"] == "jsonl"
    assert counts_dict(dataset) == {0: 2, 1: 1}
    _, locators = catalog_locators(dataset)
    assert [locators[i].compression for i in (0, 1)] == ["gzip", "zstd"]
    for i, (name, data) in enumerate(shards.items()):
        zip_file = locators[i].zip
        assert zip_file is not None
        assert (zip_file.basename, zip_file.bytes) == (name, len(data))
        assert locators[i].raw.basename == f"{name}.raw"
    assert [locators[i].raw.bytes for i in (0, 1)] == [18, 9]  # decoded sizes


def test_from_path_dataset_pickle_is_small(tmp_path: Path) -> None:
    """A file-backed Dataset must pickle to KB (handle only; no shard graph)."""
    import pickle

    root = tmp_path
    for i in range(50):
        (root / f"shard{i:03d}.jsonl").write_text(
            '{"id":1}\n{"id":2}\n{"id":3}\n', encoding="utf-8"
        )
    ds = Dataset.from_path("many", str(root))
    assert ds.total() == 150
    blob = pickle.dumps(ds)
    assert len(blob) < 4096, f"Dataset pickle too large: {len(blob)} bytes"

    restored = pickle.loads(blob)
    assert "shards" not in restored.backend  # no per-shard graph travels
    assert restored._ids is None  # count arrays dropped
    assert restored._catalog_handle is not None


def test_dataset_from_path_detects_jsonl(tmp_path: Path) -> None:
    shard0 = tmp_path / "shard0.jsonl"
    shard1 = tmp_path / "shard1.jsonl"
    shard0.write_text(
        "\n".join(json.dumps({"text": f"sample {i}", "value": i}) for i in range(3)),
        encoding="utf-8",
    )
    shard1.write_text(
        "\n".join(json.dumps({"text": f"sample {i}", "value": i}) for i in range(3, 5)),
        encoding="utf-8",
    )
    ds = Dataset.from_path("demo", str(tmp_path))
    assert ds.backend["kind"] == "jsonl"
    # from_path is count-only: no per-shard metadata graph, just the handle.
    assert "shards" not in ds.backend
    assert ds._catalog_handle is not None
    assert ds._catalog_handle.fingerprint is None  # baked later by the Engine
    # The array fast path is ordered, aligned, and consistent with the counts.
    assert ds.ids().tolist() == [0, 1]
    assert ds.counts().tolist() == [3, 2]
    assert counts_dict(ds) == {0: 3, 1: 2}
    assert ds.total() == 5
    # The accessors hand out shared arrays; they must be frozen.
    assert not ds.ids().flags.writeable
    assert not ds.counts().flags.writeable


def test_dataset_accessors_dict_backed() -> None:
    """ids/counts/total/max_count/shard_count/__len__ on a dict-backed Dataset."""
    from zephon.io import InMemoryShard

    shards = {
        2: InMemoryShard([{"v": i} for i in range(3)]),
        0: InMemoryShard([{"v": i} for i in range(5)]),
        1: InMemoryShard([{"v": i} for i in range(2)]),
    }
    ds = Dataset.from_dict("demo", shards)

    # ids() is sorted and int64.
    ids = ds.ids()
    assert ids.dtype == np.int64
    assert ids.tolist() == [0, 1, 2]

    # counts() is aligned with ids() (sorted order), int64.
    counts = ds.counts()
    assert counts.dtype == np.int64
    assert counts.tolist() == [5, 2, 3]

    # Shared arrays are frozen — same contract as the from_path arrays.
    assert not ids.flags.writeable
    assert not counts.flags.writeable

    assert ds.total() == 10
    assert ds.max_count() == 5
    assert ds.shard_count() == 3
    assert len(ds) == 10


def test_dataset_accessors_empty() -> None:
    """Accessors are well-defined for an empty dict-backed Dataset."""
    ds = Dataset.from_dict("empty", {})
    assert ds.ids().dtype == np.int64
    assert ds.ids().tolist() == []
    assert ds.counts().tolist() == []
    assert ds.total() == 0
    assert ds.max_count() == 0
    assert ds.shard_count() == 0
    assert len(ds) == 0


def test_dataset_raw_bytes_dict_backed() -> None:
    from zephon.io import InMemoryShard

    shards = {
        2: InMemoryShard([{"text": "ab"} for _ in range(3)]),  # 2 * 3
        0: InMemoryShard([{"text": "abcd"} for _ in range(5)]),  # 4 * 5
        1: InMemoryShard([{"text": "x"} for _ in range(2)]),  # 1 * 2
    }
    ds = Dataset.from_dict("demo", shards)

    raw = ds.raw_bytes()
    assert raw.dtype == np.int64
    assert ds.ids().tolist() == [0, 1, 2]
    assert raw.tolist() == [20, 2, 6]


def test_dataset_raw_bytes_empty() -> None:
    ds = Dataset.from_dict("empty", {})
    assert ds.raw_bytes().dtype == np.int64
    assert ds.raw_bytes().tolist() == []


def test_dataset_raw_bytes_file_backed_matches_catalog(tmp_path: Path) -> None:
    (tmp_path / "shard0.jsonl").write_text('{"id":1}\n{"id":2}\n', encoding="utf-8")
    (tmp_path / "shard1.jsonl").write_text('{"id":3}\n', encoding="utf-8")
    ds = Dataset.from_path("demo", str(tmp_path))

    catalog = attach_catalog(ds)
    raw = ds.raw_bytes()
    assert raw.dtype == np.int64
    assert ds.ids().tolist() == catalog.ids().tolist()
    assert raw.tolist() == catalog.raw_bytes().tolist()
    # jsonl shard bytes are the on-disk file sizes.
    assert raw.tolist() == [
        (tmp_path / "shard0.jsonl").stat().st_size,
        (tmp_path / "shard1.jsonl").stat().st_size,
    ]
