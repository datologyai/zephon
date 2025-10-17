import sys
import types
from pathlib import Path

import pytest

from zephon.io.storage.s3 import S3Backend


def _install_boto3_stubs(monkeypatch: pytest.MonkeyPatch) -> type[Exception]:
    """Register lightweight boto3/botocore stubs for unit testing."""

    client_error_cls: type[Exception]

    # boto3.session.Session (unused in tests but patched for safety)
    boto3_mod = types.ModuleType("boto3")
    session_mod = types.ModuleType("boto3.session")

    class _UnusedSession:
        def client(self, *args, **kwargs):  # pragma: no cover - not used
            raise RuntimeError("Session.client should not be called in tests")

    session_mod.Session = lambda: _UnusedSession()

    s3_mod = types.ModuleType("boto3.s3")
    transfer_mod = types.ModuleType("boto3.s3.transfer")

    class TransferConfig:  # pragma: no cover - simple data holder
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    transfer_mod.TransferConfig = TransferConfig

    botocore_mod = types.ModuleType("botocore")
    botocore_mod.UNSIGNED = "unsigned"

    config_mod = types.ModuleType("botocore.config")

    class Config:  # pragma: no cover - simple data holder
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    config_mod.Config = Config

    exceptions_mod = types.ModuleType("botocore.exceptions")

    class ClientError(Exception):
        def __init__(self, response):
            super().__init__("ClientError")
            self.response = response

    class NoCredentialsError(Exception): ...

    exceptions_mod.ClientError = ClientError
    exceptions_mod.NoCredentialsError = NoCredentialsError
    client_error_cls = ClientError

    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)
    monkeypatch.setitem(sys.modules, "boto3.session", session_mod)
    monkeypatch.setitem(sys.modules, "boto3.s3", s3_mod)
    monkeypatch.setitem(sys.modules, "boto3.s3.transfer", transfer_mod)
    monkeypatch.setitem(sys.modules, "botocore", botocore_mod)
    monkeypatch.setitem(sys.modules, "botocore.config", config_mod)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions_mod)

    return client_error_cls


class _FakePaginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kwargs):
        return list(self._pages)


class _FakeS3Client:
    def __init__(self, client_error_cls: type[Exception]):
        self._err = client_error_cls
        self.objects: dict[tuple[str, str], bytes] = {}
        self.download_calls: list[dict[str, object]] = []
        self.pages: list[dict[str, object]] = []

    def download_file(self, bucket, key, dst, ExtraArgs=None, Config=None):
        payload = self.objects.get((bucket, key))
        if payload is None:
            raise self._err({"Error": {"Code": "404"}})
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(payload)
        self.download_calls.append(
            {
                "bucket": bucket,
                "key": key,
                "extra": ExtraArgs,
                "config": Config,
            }
        )

    def head_object(self, Bucket, Key):
        payload = self.objects.get((Bucket, Key))
        if payload is None:
            raise self._err({"Error": {"Code": "404"}})
        return {"ContentLength": len(payload)}

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"
        return _FakePaginator(self.pages)


def test_s3_download_and_stat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client_error_cls = _install_boto3_stubs(monkeypatch)
    monkeypatch.setenv("ZEPHON_AWS_REQUESTER_PAYS", "bucket")

    backend = S3Backend()
    fake = _FakeS3Client(client_error_cls)
    fake.objects[("bucket", "prefix/file.bin")] = b"payload"
    backend._client = fake
    backend._ensure_client = lambda timeout=None, unsigned_ok=True: None

    out = tmp_path / "out.bin"
    backend.download("s3://bucket/prefix/file.bin", str(out))
    assert out.read_bytes() == b"payload"

    # Transfer args include requester pays when configured.
    assert fake.download_calls[0]["extra"] == {"RequestPayer": "requester"}

    info = backend.stat("s3://bucket/prefix/file.bin")
    assert info["size"] == len(b"payload")


def test_s3_download_missing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client_error_cls = _install_boto3_stubs(monkeypatch)
    backend = S3Backend()
    fake = _FakeS3Client(client_error_cls)
    backend._client = fake
    backend._ensure_client = lambda timeout=None, unsigned_ok=True: None

    with pytest.raises(FileNotFoundError):
        backend.download("s3://bucket/missing.bin", str(tmp_path / "out.bin"))


def test_s3_exists_and_listdir(monkeypatch: pytest.MonkeyPatch) -> None:
    client_error_cls = _install_boto3_stubs(monkeypatch)
    backend = S3Backend()
    fake = _FakeS3Client(client_error_cls)
    fake.objects[("bucket", "root/file.jsonl")] = b"{}"
    fake.pages = [
        {
            "Contents": [
                {"Key": "root/file.jsonl"},
                {"Key": "root/subdir/ignored"},
            ]
        }
    ]
    backend._client = fake
    backend._ensure_client = lambda timeout=None, unsigned_ok=True: None

    assert backend.exists("s3://bucket/root/file.jsonl") is True
    assert backend.exists("s3://bucket/root/missing.jsonl") is False
    assert backend.listdir("s3://bucket/root") == ["file.jsonl"]


def test_s3_open_downloads_to_temp(tmp_path: Path) -> None:
    backend = S3Backend()
    invoked: list[Path] = []

    def fake_download(src: str, dst: str, timeout: float | None = None) -> None:
        path = Path(dst)
        path.write_text("hello", encoding="utf-8")
        invoked.append(path)

    backend.download = fake_download  # type: ignore[assignment]

    with backend.open("s3://bucket/file.txt", "r", encoding="utf-8") as handle:
        assert handle.read() == "hello"

    temp_path = invoked[0]
    assert not temp_path.exists()
