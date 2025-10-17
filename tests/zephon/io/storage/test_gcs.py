from pathlib import Path

import pytest

from zephon.io.storage.gcs import GCSBackend


class _FakeBlob:
    def __init__(self, name: str, store: dict[str, bytes]) -> None:
        self.name = name
        self._store = store

    def download_to_filename(self, dst: str) -> None:
        Path(dst).write_bytes(self._store[self.name])

    def exists(self) -> bool:
        return self.name in self._store

    @property
    def size(self) -> int:
        return len(self._store[self.name])


class _FakeBucket:
    def __init__(self, bucket: str, store: dict[str, bytes]) -> None:
        self._bucket = bucket
        self._store = store

    def blob(self, key: str) -> _FakeBlob:
        return _FakeBlob(f"{self._bucket}/{key}", self._store)

    def get_blob(self, key: str) -> _FakeBlob | None:
        name = f"{self._bucket}/{key}"
        if name not in self._store:
            return None
        return _FakeBlob(name, self._store)


class _FakeGCSClient:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.listing: list[str] = []

    def bucket(self, bucket: str) -> _FakeBucket:
        return _FakeBucket(bucket, self.store)

    def list_blobs(self, bucket: str, prefix: str, delimiter: str):
        base = prefix or ""
        results = []
        for key in self.listing:
            if key.startswith(base) and "/" not in key[len(base) :]:
                results.append(_ListBlobView(key))
        return results


class _ListBlobView:
    def __init__(self, name: str) -> None:
        self.name = name


def test_gcs_native_download_stat(tmp_path: Path) -> None:
    backend = GCSBackend()
    fake = _FakeGCSClient()
    fake.store["bucket/folder/file.bin"] = b"payload"
    backend._client = fake
    backend._mode = "gcs"
    backend._ensure_client = lambda: None

    out = tmp_path / "file.bin"
    backend.download("gs://bucket/folder/file.bin", str(out))
    assert out.read_bytes() == b"payload"

    assert backend.exists("gs://bucket/folder/file.bin") is True
    assert backend.exists("gs://bucket/folder/missing.bin") is False

    info = backend.stat("gs://bucket/folder/file.bin")
    assert info["size"] == len(b"payload")


def test_gcs_native_listdir(tmp_path: Path) -> None:
    backend = GCSBackend()
    fake = _FakeGCSClient()
    fake.store["bucket/prefix/a.jsonl"] = b"{}"
    fake.listing = ["prefix/a.jsonl", "prefix/sub/b.jsonl"]
    backend._client = fake
    backend._mode = "gcs"
    backend._ensure_client = lambda: None

    assert backend.listdir("gs://bucket/prefix") == ["a.jsonl"]


class _FakeS3CompatClient:
    def __init__(self, client_error_cls: type[Exception]) -> None:
        self._err = client_error_cls
        self.objects: dict[tuple[str, str], bytes] = {}

    def download_file(self, bucket, key, dst, Config=None):
        payload = self.objects.get((bucket, key))
        if payload is None:
            raise self._err({"Error": {"Code": "404"}})
        Path(dst).write_bytes(payload)

    def head_object(self, Bucket, Key):
        payload = self.objects.get((Bucket, Key))
        if payload is None:
            raise self._err({"Error": {"Code": "404"}})
        return {"ContentLength": len(payload)}

    def get_paginator(self, operation_name):
        return _FakePaginator(
            [
                {"Contents": [{"Key": "prefix/file.jsonl"}]},
            ]
        )


class _FakePaginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kwargs):
        return list(self._pages)


def _install_boto3_stubs(monkeypatch: pytest.MonkeyPatch) -> type[Exception]:
    import sys
    import types

    boto3_mod = types.ModuleType("boto3")
    s3_mod = types.ModuleType("boto3.s3")
    transfer_mod = types.ModuleType("boto3.s3.transfer")

    class TransferConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    transfer_mod.TransferConfig = TransferConfig

    botocore_mod = types.ModuleType("botocore")
    botocore_mod.UNSIGNED = "unsigned"
    config_mod = types.ModuleType("botocore.config")

    class Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    config_mod.Config = Config
    exceptions_mod = types.ModuleType("botocore.exceptions")

    class ClientError(Exception):
        def __init__(self, response):
            super().__init__("ClientError")
            self.response = response

    exceptions_mod.ClientError = ClientError
    exceptions_mod.NoCredentialsError = type("NoCredentialsError", (Exception,), {})

    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)
    monkeypatch.setitem(sys.modules, "boto3.s3", s3_mod)
    monkeypatch.setitem(sys.modules, "boto3.s3.transfer", transfer_mod)
    monkeypatch.setitem(sys.modules, "botocore", botocore_mod)
    monkeypatch.setitem(sys.modules, "botocore.config", config_mod)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions_mod)

    return ClientError


def test_gcs_s3_compatible_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client_error_cls = _install_boto3_stubs(monkeypatch)
    backend = GCSBackend()
    fake = _FakeS3CompatClient(client_error_cls)
    fake.objects[("bucket", "prefix/file.jsonl")] = b"{}"
    backend._client = fake
    backend._mode = "s3compat"
    backend._ensure_client = lambda: None

    out = tmp_path / "file.jsonl"
    backend.download("gs://bucket/prefix/file.jsonl", str(out))
    assert out.read_bytes() == b"{}"

    info = backend.stat("gs://bucket/prefix/file.jsonl")
    assert info["size"] == len(b"{}")
    assert backend.listdir("gs://bucket/prefix") == ["file.jsonl"]


def test_gcs_open_downloads_to_temp(tmp_path: Path) -> None:
    backend = GCSBackend()
    paths: list[Path] = []

    def fake_download(src: str, dst: str, timeout: float | None = None) -> None:
        path = Path(dst)
        path.write_text("hello", encoding="utf-8")
        paths.append(path)

    backend.download = fake_download  # type: ignore[assignment]

    with backend.open("gs://bucket/file.txt", "r", encoding="utf-8") as handle:
        assert handle.read() == "hello"

    assert not paths[0].exists()
