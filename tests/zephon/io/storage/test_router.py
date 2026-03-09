from io import BytesIO
from pathlib import Path

import pytest

from zephon.io.storage.router import RouterStorageBackend


class _DummyBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def open(self, path: str, mode: str = "rb", **kwargs):
        self.calls.append(("open", path))
        return path

    def exists(self, path: str) -> bool:
        self.calls.append(("exists", path))
        return True

    def download(self, src: str, dst: str, timeout: float | None = None) -> None:
        self.calls.append(("download", src))
        Path(dst).write_text("data", encoding="utf-8")

    def listdir(self, path: str) -> list[str]:
        self.calls.append(("listdir", path))
        return ["entry"]

    def stat(self, path: str):
        self.calls.append(("stat", path))
        return {"size": 1}


class _RangeCapableBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def read_range(
        self,
        path: str,
        start: int,
        *,
        end: int | None = None,
        length: int | None = None,
    ) -> bytes:
        self.calls.append(("read_range", path))
        data = b"0123456789"
        if end is not None:
            return data[start:end]
        if length is not None:
            return data[start : start + length]
        return data[start:]


class _NoRangeBackend:
    def __init__(self, payload: bytes = b"abcdefghij") -> None:
        self.calls: list[tuple[str, str]] = []
        self.payload = payload

    def open(self, path: str, mode: str = "rb", **kwargs):
        del kwargs
        self.calls.append(("open", path))
        assert mode == "rb"
        return BytesIO(self.payload)


class _NotImplementedRangeBackend(_NoRangeBackend):
    def read_range(
        self,
        path: str,
        start: int,
        *,
        end: int | None = None,
        length: int | None = None,
    ) -> bytes:
        del path, start, end, length
        raise NotImplementedError


def test_router_selects_s3(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _DummyBackend()
    monkeypatch.setattr("zephon.io.storage.router._make_s3_backend", lambda: dummy)

    router = RouterStorageBackend()
    router.exists("s3://bucket/file")
    router.listdir("s3://bucket/")
    assert dummy.calls == [("exists", "s3://bucket/file"), ("listdir", "s3://bucket/")]


def test_router_selects_gcs(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _DummyBackend()
    monkeypatch.setattr("zephon.io.storage.router._make_gcs_backend", lambda: dummy)

    router = RouterStorageBackend()
    router.stat("gs://bucket/file")
    assert dummy.calls == [("stat", "gs://bucket/file")]


def test_router_defaults_to_local(tmp_path: Path) -> None:
    router = RouterStorageBackend(local_root=tmp_path)
    local_file = tmp_path / "file.txt"
    local_file.write_text("local", encoding="utf-8")

    with router.open(str(local_file), "r", encoding="utf-8") as handle:
        assert handle.read() == "local"


def test_router_unknown_scheme_falls_back_to_local(tmp_path: Path) -> None:
    router = RouterStorageBackend(local_root=tmp_path)
    # Access the backend selection method directly to assert fallback behaviour
    backend = router._backend_for("ftp://host/path/file.bin")  # type: ignore[attr-defined]
    from zephon.io.storage.local import LocalFSBackend

    assert isinstance(backend, LocalFSBackend)


def test_router_read_range_uses_backend_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _RangeCapableBackend()
    monkeypatch.setattr("zephon.io.storage.router._make_s3_backend", lambda: dummy)
    router = RouterStorageBackend()

    out = router.read_range("s3://bucket/file.bin", 2, length=4)

    assert out == b"2345"
    assert dummy.calls == [("read_range", "s3://bucket/file.bin")]


def test_router_read_range_falls_back_when_method_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _NoRangeBackend()
    monkeypatch.setattr("zephon.io.storage.router._make_s3_backend", lambda: dummy)
    router = RouterStorageBackend()

    out = router.read_range("s3://bucket/payload.bin", 3, length=3)

    assert out == b"def"
    assert dummy.calls == [("open", "s3://bucket/payload.bin")]


def test_router_read_range_falls_back_when_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _NotImplementedRangeBackend()
    monkeypatch.setattr("zephon.io.storage.router._make_s3_backend", lambda: dummy)
    router = RouterStorageBackend()

    out = router.read_range("s3://bucket/payload2.bin", 4, end=8)

    assert out == b"efgh"
    assert dummy.calls == [("open", "s3://bucket/payload2.bin")]


def test_router_read_range_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _RangeCapableBackend()
    monkeypatch.setattr("zephon.io.storage.router._make_s3_backend", lambda: dummy)
    router = RouterStorageBackend()

    with pytest.raises(ValueError, match="start must be non-negative"):
        router.read_range("s3://bucket/file.bin", -1, length=2)

    with pytest.raises(ValueError, match="Specify at most one of end or length"):
        router.read_range("s3://bucket/file.bin", 0, end=2, length=1)
