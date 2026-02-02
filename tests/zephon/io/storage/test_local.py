import shutil
from pathlib import Path

import pytest

from zephon.io.storage.local import LocalFSBackend


def test_local_open_and_exists_relative_and_absolute(tmp_path: Path) -> None:
    backend = LocalFSBackend(root=tmp_path)
    rel_file = tmp_path / "rel.txt"
    abs_file = tmp_path / "abs.txt"
    rel_file.write_text("hello", encoding="utf-8")
    abs_file.write_text("world", encoding="utf-8")

    with backend.open("rel.txt", "r", encoding="utf-8") as fh:
        assert fh.read() == "hello"
    assert backend.exists("rel.txt") is True

    # Absolute path should ignore the backend root
    other = tmp_path / "other"
    other.mkdir()
    backend2 = LocalFSBackend(root=other)
    with backend2.open(str(abs_file), "r", encoding="utf-8") as fh:
        assert fh.read() == "world"


def test_local_download_variants_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = LocalFSBackend(root=tmp_path)
    src = tmp_path / "src.bin"
    dst = tmp_path / "out" / "dst.bin"
    buf = b"x" * (64 * 1024)
    src.write_bytes(buf)

    # timeout=None path
    backend.download(str(src), str(dst), timeout=None)
    assert dst.read_bytes() == buf

    # overwrite via chunked copy path (timeout>0)
    backend.download(str(src), str(dst), timeout=1.0)
    assert dst.read_bytes() == buf

    # Simulate failure during copyfile to test cleanup of target
    monkeypatch.setattr(
        shutil, "copyfile", lambda _s, _d: (_ for _ in ()).throw(OSError("fail"))
    )
    dst.unlink(missing_ok=True)
    with pytest.raises(OSError):
        backend.download(str(src), str(dst), timeout=None)
    assert not dst.exists()


def test_local_listdir_and_stat(tmp_path: Path) -> None:
    backend = LocalFSBackend(root=tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bb", encoding="utf-8")
    names = backend.listdir(str(tmp_path))
    assert names == sorted(["a.txt", "b.txt"])  # sorted by backend

    info = backend.stat(str(tmp_path / "b.txt"))
    assert int(info.get("size", -1)) == 2
    assert "mtime" in info  # mtime should be present
    assert isinstance(info["mtime"], float)

    with pytest.raises(NotADirectoryError):
        _ = backend.listdir(str(tmp_path / "a.txt"))


def test_local_put_and_delete(tmp_path: Path) -> None:
    """Test put and delete operations."""
    backend = LocalFSBackend(root=tmp_path)

    # Test put creates file and parent directories
    target = tmp_path / "subdir" / "file.txt"
    backend.put(str(target), b"hello world")
    assert target.exists()
    assert target.read_bytes() == b"hello world"

    # Test put overwrites existing file
    backend.put(str(target), b"new content")
    assert target.read_bytes() == b"new content"

    # Test delete removes file
    backend.delete(str(target))
    assert not target.exists()

    # Test delete is idempotent (no error for missing file)
    backend.delete(str(target))  # Should not raise


def test_local_glob(tmp_path: Path) -> None:
    """Test glob pattern matching."""
    backend = LocalFSBackend(root=tmp_path)

    # Create test files
    (tmp_path / "state_r0_w0_123.json").write_text("{}", encoding="utf-8")
    (tmp_path / "state_r1_w0_123.json").write_text("{}", encoding="utf-8")
    (tmp_path / "state_r0_w0_456.json").write_text("{}", encoding="utf-8")
    (tmp_path / "merged_123.json").write_text("{}", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "state_r2_w0_123.json").write_text("{}", encoding="utf-8")

    # Test glob with wildcards
    pattern = str(tmp_path / "state_r*_w*_123.json")
    matches = backend.glob(pattern)
    assert len(matches) == 2
    assert all("123.json" in m for m in matches)
    assert all("state_r" in m for m in matches)

    # Test glob with no matches
    pattern = str(tmp_path / "nonexistent_*.json")
    matches = backend.glob(pattern)
    assert matches == []

    # Test glob without wildcards (exact match)
    pattern = str(tmp_path / "merged_123.json")
    matches = backend.glob(pattern)
    assert len(matches) == 1
    assert matches[0] == pattern

    # Test glob with relative pattern
    matches = backend.glob("state_r0_*.json")
    assert len(matches) == 2
