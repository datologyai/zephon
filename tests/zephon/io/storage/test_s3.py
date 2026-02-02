"""Tests for S3Backend using obstore."""

from pathlib import Path

import pytest

from tests.helpers.storage import _install_obstore_stubs


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


def test_s3_put_and_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test S3 put and delete operations."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.s3 import S3Backend

    backend = S3Backend()

    # Test put creates object
    backend.put("s3://bucket/new/file.txt", b"hello world")
    assert state["objects"][("bucket", "new/file.txt")] == b"hello world"

    # Test put overwrites
    backend.put("s3://bucket/new/file.txt", b"new content")
    assert state["objects"][("bucket", "new/file.txt")] == b"new content"

    # Test delete removes object
    backend.delete("s3://bucket/new/file.txt")
    assert ("bucket", "new/file.txt") not in state["objects"]

    # Test delete is idempotent
    backend.delete("s3://bucket/new/file.txt")  # Should not raise

    # Test invalid URL handling
    with pytest.raises(ValueError):
        backend.put("not-s3://bucket/file", b"data")

    # delete with invalid URL should be no-op (idempotent)
    backend.delete("not-s3://bucket/file")  # Should not raise


def test_s3_glob(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test S3 glob pattern matching."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.s3 import S3Backend

    # Create test objects
    state["objects"][("bucket", "prefix/state_r0_w0_123.json")] = b"{}"
    state["objects"][("bucket", "prefix/state_r1_w0_123.json")] = b"{}"
    state["objects"][("bucket", "prefix/state_r0_w0_456.json")] = b"{}"
    state["objects"][("bucket", "prefix/merged_123.json")] = b"{}"

    backend = S3Backend()

    # Test glob with wildcards
    matches = backend.glob("s3://bucket/prefix/state_r*_w*_123.json")
    assert len(matches) == 2
    assert all("123.json" in m for m in matches)
    assert all("state_r" in m for m in matches)

    # Test glob with no matches
    matches = backend.glob("s3://bucket/prefix/nonexistent_*.json")
    assert matches == []

    # Test glob without wildcards (exact match)
    matches = backend.glob("s3://bucket/prefix/merged_123.json")
    assert len(matches) == 1
    assert matches[0] == "s3://bucket/prefix/merged_123.json"

    # Test glob with invalid URL
    matches = backend.glob("not-s3://bucket/file*")
    assert matches == []


def test_s3_stat_includes_mtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that stat returns mtime."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon.io.storage.s3 import S3Backend

    state["objects"][("bucket", "file.bin")] = b"data"

    backend = S3Backend()
    info = backend.stat("s3://bucket/file.bin")

    assert "size" in info
    assert "mtime" in info
    assert isinstance(info["mtime"], float)


def test_s3_stat_access_denied_raises_permission_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that stat raises PermissionError on 403 Access Denied."""
    _install_obstore_stubs(monkeypatch)

    import sys

    # Patch the mock head to raise 403 for a specific key
    obstore_mod = sys.modules["obstore"]
    original_head = obstore_mod.head

    def mock_head_with_403(store, key):
        if key == "forbidden.bin":
            raise Exception("403 AccessDenied: Access denied")
        return original_head(store, key)

    obstore_mod.head = mock_head_with_403

    from zephon.io.storage.s3 import S3Backend

    backend = S3Backend()

    with pytest.raises(PermissionError, match="Access denied"):
        backend.stat("s3://bucket/forbidden.bin")
