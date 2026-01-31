"""Tests for S3Backend using obstore."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _install_obstore_stubs(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Install obstore stubs for testing without the actual package."""
    # Create mock obstore module
    obstore_mod = types.ModuleType("obstore")
    store_mod = types.ModuleType("obstore.store")

    # Mock classes
    class MockS3Store:
        @classmethod
        def from_url(cls, url: str, config: dict = None, client_options: dict = None):
            store = MagicMock()
            store._config = config or {}
            store._url = url
            store._client_options = client_options or {}
            return store

    store_mod.S3Store = MockS3Store

    # State to control mock behavior
    state = {
        "objects": {},  # (bucket, key) -> bytes
        "configs": [],  # track configs passed to from_url
    }

    class MockGetResult:
        def __init__(self, data: bytes):
            self._data = data

        def bytes(self) -> bytes:
            return self._data

    def mock_get(store, key):
        # Extract bucket from store URL
        url = getattr(store, "_url", "s3://unknown")
        bucket = url.replace("s3://", "")
        data = state["objects"].get((bucket, key))
        if data is None:
            raise Exception(f"404 NotFound: {key}")
        return MockGetResult(data)

    def mock_head(store, key):
        url = getattr(store, "_url", "s3://unknown")
        bucket = url.replace("s3://", "")
        data = state["objects"].get((bucket, key))
        if data is None:
            raise Exception(f"404 NotFound: {key}")
        return {"size": len(data), "path": key}

    def mock_list(store, prefix: str = ""):
        url = getattr(store, "_url", "s3://unknown")
        bucket = url.replace("s3://", "")
        results = []
        for (b, k), data in state["objects"].items():
            if b == bucket and k.startswith(prefix):
                results.append({"path": k, "size": len(data)})
        return iter([results])

    obstore_mod.get = mock_get
    obstore_mod.head = mock_head
    obstore_mod.list = mock_list

    # Patch from_url to track configs
    original_from_url = MockS3Store.from_url

    @classmethod
    def tracking_from_url(
        cls, url: str, config: dict = None, client_options: dict = None
    ):
        state["configs"].append(config or {})
        return original_from_url(url, config, client_options)

    store_mod.S3Store.from_url = tracking_from_url

    monkeypatch.setitem(sys.modules, "obstore", obstore_mod)
    monkeypatch.setitem(sys.modules, "obstore.store", store_mod)

    return state


def test_s3_download_and_stat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test S3 download and stat operations."""
    state = _install_obstore_stubs(monkeypatch)
    monkeypatch.setenv("ZEPHON_AWS_REQUESTER_PAYS", "bucket")

    from zephon.io.storage.s3 import S3Backend

    state["objects"][("bucket", "prefix/file.bin")] = b"payload"

    backend = S3Backend()

    out = tmp_path / "out.bin"
    backend.download("s3://bucket/prefix/file.bin", str(out))
    assert out.read_bytes() == b"payload"

    # Verify requester pays config was set
    assert state["configs"][0].get("request_payer") is True

    info = backend.stat("s3://bucket/prefix/file.bin")
    assert info["size"] == 7


def test_s3_download_missing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that downloading a missing file raises FileNotFoundError."""
    _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.s3 import S3Backend

    backend = S3Backend()

    with pytest.raises(FileNotFoundError):
        backend.download("s3://bucket/missing.bin", str(tmp_path / "out.bin"))


def test_s3_exists_and_listdir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test S3 exists and listdir operations."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.s3 import S3Backend

    state["objects"][("bucket", "root/file.jsonl")] = b"{}"
    state["objects"][("bucket", "root/subdir/ignored")] = b"{}"

    backend = S3Backend()

    assert backend.exists("s3://bucket/root/file.jsonl") is True
    assert backend.exists("s3://bucket/root/missing.jsonl") is False
    assert backend.listdir("s3://bucket/root") == ["file.jsonl"]


def test_s3_stat_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that stat on missing file raises FileNotFoundError."""
    _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.s3 import S3Backend

    backend = S3Backend()

    with pytest.raises(FileNotFoundError):
        backend.stat("s3://bucket/missing.bin")


def test_s3_open_downloads_to_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that open() downloads to temp file and cleans up."""
    _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.s3 import S3Backend

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


def test_s3_invalid_url_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of invalid S3 URLs."""
    _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.s3 import S3Backend

    backend = S3Backend()

    # exists returns False for invalid URLs
    assert backend.exists("not-s3://bucket/file") is False
    assert backend.exists("s3://") is False

    # download raises for invalid URLs
    with pytest.raises(ValueError):
        backend.download("not-s3://bucket/file", "/tmp/out")

    # stat raises for invalid URLs
    with pytest.raises(FileNotFoundError):
        backend.stat("not-s3://bucket")


def test_s3_custom_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test S3 with custom endpoint (e.g., MinIO)."""
    state = _install_obstore_stubs(monkeypatch)
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:9000")

    from zephon.io.storage.s3 import S3Backend

    state["objects"][("bucket", "file.bin")] = b"data"

    backend = S3Backend()
    backend.exists("s3://bucket/file.bin")

    assert state["configs"][0].get("aws_endpoint") == "http://localhost:9000"
