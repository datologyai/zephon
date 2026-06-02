# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``zephon.utils.atomic.atomic_write_bytes``."""

import os
import threading
from pathlib import Path

import pytest

from zephon.utils.atomic import atomic_write_bytes


def test_writes_bytes_and_creates_parent_dirs(tmp_path: Path) -> None:
    dst = tmp_path / "nested" / "dir" / "file.bin"
    atomic_write_bytes(str(dst), b"payload")
    assert dst.read_bytes() == b"payload"


def test_overwrites_existing_file(tmp_path: Path) -> None:
    dst = tmp_path / "file.bin"
    dst.write_bytes(b"old")
    atomic_write_bytes(dst, b"new")
    assert dst.read_bytes() == b"new"


def test_writes_empty_payload(tmp_path: Path) -> None:
    dst = tmp_path / "empty.bin"
    atomic_write_bytes(dst, b"")
    assert dst.read_bytes() == b""


def test_leaves_no_temp_file_after_success(tmp_path: Path) -> None:
    dst = tmp_path / "file.bin"
    atomic_write_bytes(dst, b"data")
    assert [p.name for p in tmp_path.iterdir()] == ["file.bin"]


def test_temp_name_is_pid_thread_unique_or_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = os.replace
    seen: list[str] = []

    def spy(src, dst):
        seen.append(Path(src).name)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    monkeypatch.setattr(os, "getpid", lambda: 4242)
    monkeypatch.setattr(threading, "get_ident", lambda: 99)

    dst = tmp_path / "art.bin"
    atomic_write_bytes(dst, b"a", unique_tmp=True)
    atomic_write_bytes(dst, b"b", unique_tmp=False)
    assert seen == ["art.bin.4242.99.tmp", "art.bin.tmp"]
    assert dst.read_bytes() == b"b"


def test_concurrent_writers_use_distinct_temp_files(tmp_path: Path) -> None:
    """Two threads writing the same destination must not share a temp file.

    The ``unique_tmp`` suffix embeds both the PID and the thread id, so
    concurrent writers in one process get distinct temp siblings and neither
    truncates or renames the other's in-flight temp. With a PID-only suffix
    both threads would target one temp name and the second ``os.replace``
    would hit ``FileNotFoundError`` after the first renamed it away.
    """
    dst = tmp_path / "shared.bin"
    seen_tmps: list[str] = []
    seen_lock = threading.Lock()
    # Force both threads to be mid-write (temp created, not yet renamed)
    # simultaneously so a shared temp name would actually collide.
    rendezvous = threading.Barrier(2, timeout=5)
    real_replace = os.replace

    def spy(src, dst_):
        with seen_lock:
            seen_tmps.append(Path(src).name)
        rendezvous.wait()
        real_replace(src, dst_)

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            atomic_write_bytes(dst, b"payload")
        except BaseException as exc:  # noqa: BLE001 - surfaced via assert below
            errors.append(exc)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "replace", spy)
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    assert not errors, f"concurrent writers raised: {errors}"
    assert len(set(seen_tmps)) == 2, f"temp names collided: {seen_tmps}"
    assert dst.read_bytes() == b"payload"


def test_fsync_only_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsynced: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda fd: fsynced.append(fd))

    dst = tmp_path / "f.bin"
    atomic_write_bytes(dst, b"x", fsync=False)
    assert fsynced == []
    atomic_write_bytes(dst, b"y", fsync=True)
    assert len(fsynced) == 1
    assert dst.read_bytes() == b"y"


def test_replace_failure_cleans_temp_and_preserves_dest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dst = tmp_path / "f.bin"
    dst.write_bytes(b"original")

    def boom(src, dst_):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_bytes(dst, b"new", unique_tmp=False)
    assert dst.read_bytes() == b"original"
    assert [p.name for p in tmp_path.iterdir()] == ["f.bin"]
