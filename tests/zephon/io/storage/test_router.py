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
