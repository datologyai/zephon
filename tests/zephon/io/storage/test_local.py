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

    with pytest.raises(NotADirectoryError):
        _ = backend.listdir(str(tmp_path / "a.txt"))
