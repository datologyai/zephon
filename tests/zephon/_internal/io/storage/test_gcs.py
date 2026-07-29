"""Tests for GCSBackend using obstore."""

import sys
from pathlib import Path

import pytest

from tests.helpers.storage import _install_obstore_stubs


def test_gcs_native_download_stat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test GCS download and stat operations in native mode."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon._internal.io.storage.gcs import GCSBackend

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

    from zephon._internal.io.storage.gcs import GCSBackend

    backend = GCSBackend()
    assert backend.exists("gs://bucket/folder/missing.bin") is False


def test_gcs_native_listdir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test GCS listdir operation."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon._internal.io.storage.gcs import GCSBackend

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

    from zephon._internal.io.storage.gcs import GCSBackend

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

    from zephon._internal.io.storage.gcs import GCSBackend

    state["objects"][("bucket", "prefix/file.jsonl")] = b"{}"

    backend = GCSBackend()
    assert backend.listdir("gs://bucket/prefix") == ["file.jsonl"]


def test_gcs_open_downloads_to_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that open() downloads to temp file and cleans up."""
    _install_obstore_stubs(monkeypatch)

    from zephon._internal.io.storage.gcs import GCSBackend

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


def test_gcs_read_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test GCS byte-range reads."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon._internal.io.storage.gcs import GCSBackend

    state["objects"][("bucket", "prefix/file.bin")] = b"abcdef"

    backend = GCSBackend()

    assert backend.read_range("gs://bucket/prefix/file.bin", 0, length=2) == b"ab"
    assert backend.read_range("gs://bucket/prefix/file.bin", 3, end=6) == b"def"
    assert state["range_calls"] == [
        {
            "bucket": "bucket",
            "key": "prefix/file.bin",
            "start": 0,
            "end": None,
            "length": 2,
        },
        {
            "bucket": "bucket",
            "key": "prefix/file.bin",
            "start": 3,
            "end": 6,
            "length": None,
        },
    ]


def test_gcs_invalid_url_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of invalid GCS URLs."""
    _install_obstore_stubs(monkeypatch)

    from zephon._internal.io.storage.gcs import GCSBackend

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

    from zephon._internal.io.storage.gcs import GCSBackend

    backend = GCSBackend()

    with pytest.raises(FileNotFoundError):
        backend.download("gs://bucket/missing.bin", str(tmp_path / "out.bin"))


def test_gcs_stat_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that stat on missing file raises FileNotFoundError."""
    _install_obstore_stubs(monkeypatch)

    from zephon._internal.io.storage.gcs import GCSBackend

    backend = GCSBackend()

    with pytest.raises(FileNotFoundError):
        backend.stat("gs://bucket/missing.bin")


def test_gcs_scheme_gcs_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that gcs:// scheme works (alias for gs://)."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon._internal.io.storage.gcs import GCSBackend

    state["objects"][("bucket", "file.bin")] = b"data"

    backend = GCSBackend()
    assert backend.exists("gcs://bucket/file.bin") is True


def test_gcs_put_and_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test GCS put and delete operations."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon._internal.io.storage.gcs import GCSBackend

    backend = GCSBackend()

    # Test put creates object
    backend.put("gs://bucket/new/file.txt", b"hello world")
    assert state["objects"][("bucket", "new/file.txt")] == b"hello world"

    # Test put overwrites
    backend.put("gs://bucket/new/file.txt", b"new content")
    assert state["objects"][("bucket", "new/file.txt")] == b"new content"

    # Test delete removes object
    backend.delete("gs://bucket/new/file.txt")
    assert ("bucket", "new/file.txt") not in state["objects"]

    # Test delete is idempotent
    backend.delete("gs://bucket/new/file.txt")  # Should not raise

    # Test invalid URL handling
    with pytest.raises(ValueError):
        backend.put("not-gs://bucket/file", b"data")

    # delete with invalid URL should be no-op (idempotent)
    backend.delete("not-gs://bucket/file")  # Should not raise


def test_gcs_glob(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test GCS glob pattern matching."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon._internal.io.storage.gcs import GCSBackend

    # Create test objects
    state["objects"][("bucket", "prefix/state_r0_w0_123.json")] = b"{}"
    state["objects"][("bucket", "prefix/state_r1_w0_123.json")] = b"{}"
    state["objects"][("bucket", "prefix/state_r0_w0_456.json")] = b"{}"
    state["objects"][("bucket", "prefix/merged_123.json")] = b"{}"

    backend = GCSBackend()

    # Test glob with wildcards
    matches = backend.glob("gs://bucket/prefix/state_r*_w*_123.json")
    assert len(matches) == 2
    assert all("123.json" in m for m in matches)
    assert all("state_r" in m for m in matches)

    # Test glob with no matches
    matches = backend.glob("gs://bucket/prefix/nonexistent_*.json")
    assert matches == []

    # Test glob without wildcards (exact match)
    matches = backend.glob("gs://bucket/prefix/merged_123.json")
    assert len(matches) == 1
    assert matches[0] == "gs://bucket/prefix/merged_123.json"

    # Test glob with invalid URL
    matches = backend.glob("not-gs://bucket/file*")
    assert matches == []


def test_gcs_stat_includes_mtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that stat returns mtime."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon._internal.io.storage.gcs import GCSBackend

    state["objects"][("bucket", "file.bin")] = b"data"

    backend = GCSBackend()
    info = backend.stat("gs://bucket/file.bin")

    assert "size" in info
    assert "mtime" in info
    assert isinstance(info["mtime"], float)


# ---------- walk() — exercises the base ObstoreBackend implementation ---------- #


def test_gcs_walk_yields_recursive_rel_paths_with_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercises the base ``ObstoreBackend.walk`` (GCS does not override it)."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon._internal.io.storage.gcs import GCSBackend

    state["objects"][("bucket", "tok/tokenizer.json")] = b"\x00" * 8
    state["objects"][("bucket", "tok/subdir/config.json")] = b"\x00" * 4
    state["objects"][("bucket", "outside/leak.json")] = b"\x00" * 99

    backend = GCSBackend()
    out = sorted(backend.walk("gs://bucket/tok/"))

    assert out == [
        ("subdir/config.json", 4),
        ("tokenizer.json", 8),
    ]


def test_gcs_walk_skips_folder_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-byte keys ending in ``/`` (folder markers) are skipped."""
    state = _install_obstore_stubs(monkeypatch)

    from zephon._internal.io.storage.gcs import GCSBackend

    state["objects"][("bucket", "tok/real.json")] = b"\x00" * 5
    state["objects"][("bucket", "tok/subdir/")] = b""

    backend = GCSBackend()
    assert list(backend.walk("gs://bucket/tok/")) == [("real.json", 5)]


def test_gcs_walk_accepts_prefix_without_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_obstore_stubs(monkeypatch)

    from zephon._internal.io.storage.gcs import GCSBackend

    state["objects"][("bucket", "tok/file.json")] = b"x"

    backend = GCSBackend()
    assert sorted(backend.walk("gs://bucket/tok")) == [("file.json", 1)]


def test_gcs_walk_invalid_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_obstore_stubs(monkeypatch)

    from zephon._internal.io.storage.gcs import GCSBackend

    backend = GCSBackend()
    with pytest.raises(ValueError, match="Invalid URL"):
        list(backend.walk("not-gs://bucket/prefix/"))


def test_gcs_walk_propagates_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """No unsigned-fallback on GCS — errors must propagate."""
    _install_obstore_stubs(monkeypatch)
    obstore_mod = sys.modules["obstore"]

    def always_fail(store, prefix=""):  # noqa: ARG001
        raise RuntimeError("GCS access denied")

    obstore_mod.list = always_fail

    from zephon._internal.io.storage.gcs import GCSBackend

    backend = GCSBackend()
    with pytest.raises(RuntimeError, match="GCS access denied"):
        list(backend.walk("gs://bucket/prefix/"))
