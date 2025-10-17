import sys
import types
from pathlib import Path


def _install_boto3_stubs(monkeypatch) -> type[Exception]:
    """Register lightweight boto3/botocore stubs for unit testing."""

    client_error_cls: type[Exception]

    # boto3.session.Session (unused but patched for safety)
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
            {"bucket": bucket, "key": key, "extra": ExtraArgs, "config": Config}
        )

    def head_object(self, Bucket, Key):
        payload = self.objects.get((Bucket, Key))
        if payload is None:
            raise self._err({"Error": {"Code": "404"}})
        return {"ContentLength": len(payload)}

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"
        return _FakePaginator(self.pages)


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


__all__ = [
    "_install_boto3_stubs",
    "_FakeS3Client",
    "_FakePaginator",
    "_FakeGCSClient",
]
