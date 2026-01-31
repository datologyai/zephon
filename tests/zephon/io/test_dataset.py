import json
from pathlib import Path

import pytest

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
    assert dataset.shard_index == {0: 3}
    assert dataset.backend["shards"][0]["raw"]["basename"] == "shard0.mds"


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
    assert dataset.shard_index == {0: 2, 1: 2}
    shards = dataset.backend["shards"]
    assert shards[0]["raw"]["basename"].endswith("shard0.jsonl")


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
    assert dataset.shard_index == {0: 2, 1: 2}
    assert dataset.backend["shards"][0]["raw"]["basename"] == "shard0.jsonl"


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
    assert set(ds.shard_index.keys()) == {0, 1}
    assert sum(ds.shard_index.values()) == 5
