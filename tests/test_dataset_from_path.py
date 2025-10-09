import json
from pathlib import Path

import pytest

from zephon.io.dataset import Dataset
from zephon.io.storage.gcs import GCSBackend
from zephon.io.storage.s3 import S3Backend

from .test_storage_gcs import _FakeGCSClient
from .test_storage_s3 import _FakeS3Client, _install_boto3_stubs


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
        client_err = _install_boto3_stubs(monkeypatch)
        s3 = S3Backend()
        fake = _FakeS3Client(client_err)
        fake.objects[("bucket", "dataset/index.json")] = json.dumps(index).encode(
            "utf-8"
        )
        s3._client = fake
        s3._ensure_client = lambda timeout=None, unsigned_ok=True: None
        monkeypatch.setattr("zephon.io.storage.router._make_s3_backend", lambda: s3)
        path = "s3://bucket/dataset"
    else:  # gcs
        gcs = GCSBackend()
        fake = _FakeGCSClient()
        fake.store["bucket/dataset/index.json"] = json.dumps(index).encode("utf-8")
        gcs._client = fake
        gcs._mode = "gcs"
        gcs._ensure_client = lambda: None
        monkeypatch.setattr("zephon.io.storage.router._make_gcs_backend", lambda: gcs)
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
        client_err = _install_boto3_stubs(monkeypatch)
        s3 = S3Backend()
        fake = _FakeS3Client(client_err)
        # Paginator returns files under prefix
        fake.pages = [
            {"Contents": [{"Key": "json/shard0.jsonl"}, {"Key": "json/shard1.jsonl"}]}
        ]
        fake.objects[("bucket", "json/shard0.jsonl")] = b'{"id":1}\n{"id":2}\n'
        fake.objects[("bucket", "json/shard1.jsonl")] = b'{"id":3}\n{"id":4}\n'
        s3._client = fake
        s3._ensure_client = lambda timeout=None, unsigned_ok=True: None
        monkeypatch.setattr("zephon.io.storage.router._make_s3_backend", lambda: s3)
        path = "s3://bucket/json"
    else:  # gcs
        gcs = GCSBackend()
        fake = _FakeGCSClient()
        fake.store["bucket/json/shard0.jsonl"] = b'{"id":1}\n{"id":2}\n'
        fake.store["bucket/json/shard1.jsonl"] = b'{"id":3}\n{"id":4}\n'
        fake.listing = ["json/shard0.jsonl", "json/shard1.jsonl"]
        gcs._client = fake
        gcs._mode = "gcs"
        gcs._ensure_client = lambda: None
        monkeypatch.setattr("zephon.io.storage.router._make_gcs_backend", lambda: gcs)
        path = "gs://bucket/json"

    dataset = Dataset.from_path("jsonl", path)
    assert dataset.backend["kind"] == "jsonl"
    assert dataset.shard_index == {0: 2, 1: 2}
    shards = dataset.backend["shards"]
    assert shards[0]["raw"]["basename"].endswith("shard0.jsonl")


def test_dataset_from_path_remote_gcs_jsonl(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = GCSBackend()
    fake_client = _FakeGCSClient()
    fake_client.store["bucket/json/shard0.jsonl"] = '{"id": 1}\n{"id": 2}\n'.encode(
        "utf-8"
    )
    fake_client.store["bucket/json/shard1.jsonl"] = '{"id": 3}\n{"id": 4}\n'.encode(
        "utf-8"
    )
    fake_client.listing = ["json/shard0.jsonl", "json/shard1.jsonl"]
    backend._client = fake_client
    backend._mode = "gcs"
    backend._ensure_client = lambda: None
    monkeypatch.setattr("zephon.io.storage.router._make_gcs_backend", lambda: backend)

    dataset = Dataset.from_path("remote-jsonl", "gs://bucket/json")

    assert dataset.backend["kind"] == "jsonl"
    assert dataset.shard_index == {0: 2, 1: 2}
    assert dataset.backend["shards"][0]["raw"]["basename"] == "shard0.jsonl"
