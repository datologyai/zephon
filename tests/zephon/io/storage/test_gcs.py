"""Tests for GCSBackend using obstore."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _install_obstore_stubs(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Install obstore stubs for testing without the actual package."""
    obstore_mod = types.ModuleType("obstore")
    store_mod = types.ModuleType("obstore.store")

    # State to control mock behavior
    state = {
        "objects": {},  # (bucket, key) -> bytes
        "configs": [],  # track configs passed to from_url
        "store_type": [],  # track which store type was used
    }

    class MockGCSStore:
        @classmethod
        def from_url(cls, url: str, config: dict = None, client_options: dict = None):
            store = MagicMock()
            store._config = config or {}
            store._url = url
            store._client_options = client_options or {}
            state["configs"].append(config or {})
            state["store_type"].append("gcs")
            return store

    class MockS3Store:
        @classmethod
        def from_url(cls, url: str, config: dict = None, client_options: dict = None):
            store = MagicMock()
            store._config = config or {}
            store._url = url
            store._client_options = client_options or {}
            state["configs"].append(config or {})
            state["store_type"].append("s3")
            return store

    store_mod.GCSStore = MockGCSStore
    store_mod.S3Store = MockS3Store

    class MockGetResult:
        def __init__(self, data: bytes):
            self._data = data

        def bytes(self) -> bytes:
            return self._data

    def mock_get(store, key):
        url = getattr(store, "_url", "gs://unknown")
        # Handle both gs:// and s3:// URLs
        bucket = url.replace("gs://", "").replace("s3://", "")
        data = state["objects"].get((bucket, key))
        if data is None:
            raise Exception(f"404 NotFound: {key}")
        return MockGetResult(data)

    def mock_head(store, key):
        url = getattr(store, "_url", "gs://unknown")
        bucket = url.replace("gs://", "").replace("s3://", "")
        data = state["objects"].get((bucket, key))
        if data is None:
            raise Exception(f"404 NotFound: {key}")
        return {"size": len(data), "path": key}

    def mock_list(store, prefix: str = ""):
        url = getattr(store, "_url", "gs://unknown")
        bucket = url.replace("gs://", "").replace("s3://", "")
        results = []
        for (b, k), data in state["objects"].items():
            if b == bucket and k.startswith(prefix):
                results.append({"path": k, "size": len(data)})
        return iter([results])

    obstore_mod.get = mock_get
    obstore_mod.head = mock_head
    obstore_mod.list = mock_list

    monkeypatch.setitem(sys.modules, "obstore", obstore_mod)
    monkeypatch.setitem(sys.modules, "obstore.store", store_mod)

    return state


def test_gcs_native_download_stat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test GCS download and stat operations in native mode."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.gcs import GCSBackend

    state["objects"][("bucket", "folder/file.bin")] = b"payload"

    backend = GCSBackend()

    out = tmp_path / "file.bin"
    backend.download("gs://bucket/folder/file.bin", str(out))
    assert out.read_bytes() == b"payload"

    assert backend.exists("gs://bucket/folder/file.bin") is True

    info = backend.stat("gs://bucket/folder/file.bin")
    assert info["size"] == 7

    # Verify native GCS store was used
    assert state["store_type"][0] == "gcs"


def test_gcs_native_exists_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test GCS exists returns False for missing files."""
    _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.gcs import GCSBackend

    backend = GCSBackend()
    assert backend.exists("gs://bucket/folder/missing.bin") is False


def test_gcs_native_listdir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test GCS listdir operation."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.gcs import GCSBackend

    state["objects"][("bucket", "prefix/a.jsonl")] = b"{}"
    state["objects"][("bucket", "prefix/sub/b.jsonl")] = b"{}"  # Should be filtered

    backend = GCSBackend()
    assert backend.listdir("gs://bucket/prefix") == ["a.jsonl"]


def test_gcs_s3_compatible_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test GCS S3-compatible mode (deprecated but supported)."""
    state = _install_obstore_stubs(monkeypatch)
    monkeypatch.setenv("GCS_KEY", "test-key")
    monkeypatch.setenv("GCS_SECRET", "test-secret")

    from zephon.io.storage.gcs import GCSBackend

    state["objects"][("bucket", "prefix/file.jsonl")] = b"{}"

    backend = GCSBackend()

    out = tmp_path / "file.jsonl"
    backend.download("gs://bucket/prefix/file.jsonl", str(out))
    assert out.read_bytes() == b"{}"

    info = backend.stat("gs://bucket/prefix/file.jsonl")
    assert info["size"] == 2

    # Verify S3-compatible store was used with correct config
    assert state["store_type"][0] == "s3"
    assert state["configs"][0]["aws_access_key_id"] == "test-key"
    assert state["configs"][0]["aws_secret_access_key"] == "test-secret"
    assert state["configs"][0]["aws_endpoint"] == "https://storage.googleapis.com"


def test_gcs_s3_compatible_listdir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test GCS listdir in S3-compatible mode."""
    state = _install_obstore_stubs(monkeypatch)
    monkeypatch.setenv("GCS_KEY", "test-key")
    monkeypatch.setenv("GCS_SECRET", "test-secret")

    from zephon.io.storage.gcs import GCSBackend

    state["objects"][("bucket", "prefix/file.jsonl")] = b"{}"

    backend = GCSBackend()
    assert backend.listdir("gs://bucket/prefix") == ["file.jsonl"]


def test_gcs_open_downloads_to_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that open() downloads to temp file and cleans up."""
    _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.gcs import GCSBackend

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


def test_gcs_invalid_url_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of invalid GCS URLs."""
    _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.gcs import GCSBackend

    backend = GCSBackend()

    # exists returns False for invalid URLs
    assert backend.exists("not-gs://bucket/file") is False
    assert backend.exists("gs://") is False

    # download raises for invalid URLs
    with pytest.raises(ValueError):
        backend.download("not-gs://bucket/file", "/tmp/out")

    # stat raises for invalid URLs
    with pytest.raises(FileNotFoundError):
        backend.stat("gs://bucket")


def test_gcs_download_missing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that downloading a missing file raises FileNotFoundError."""
    _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.gcs import GCSBackend

    backend = GCSBackend()

    with pytest.raises(FileNotFoundError):
        backend.download("gs://bucket/missing.bin", str(tmp_path / "out.bin"))


def test_gcs_stat_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that stat on missing file raises FileNotFoundError."""
    _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.gcs import GCSBackend

    backend = GCSBackend()

    with pytest.raises(FileNotFoundError):
        backend.stat("gs://bucket/missing.bin")


def test_gcs_scheme_gcs_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that gcs:// scheme works (alias for gs://)."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.gcs import GCSBackend

    state["objects"][("bucket", "file.bin")] = b"data"

    backend = GCSBackend()
    assert backend.exists("gcs://bucket/file.bin") is True
