# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing
import shutil
from pathlib import Path

import pytest

from zephon._internal.io.formats.parquet_cache.session import (
    ParquetRGSession,
    ParquetRGSessionError,
    ParquetRGSessionInUseError,
)

_CATALOG = "sha256:" + "12" * 32
_CONFIGURATION = "sha256:" + "34" * 32


def _open_session(root: Path, *, configuration: str = _CONFIGURATION):
    return ParquetRGSession(
        root,
        slot_count=4,
        limit_bytes=1_000,
        catalog_fingerprint=_CATALOG,
        configuration_fingerprint=configuration,
    )


def _join_in_child(root: str, joined, release, output) -> None:
    with _open_session(Path(root)) as session:
        output.put(session.session_id)
        joined.set()
        release.wait(timeout=10)


def test_same_process_attachments_share_and_release_lifetime_leases(
    tmp_path: Path,
) -> None:
    root = tmp_path / "decoded"
    first = _open_session(root)
    second = _open_session(root)
    try:
        assert second.session_id == first.session_id
        first.close()
        assert second.control.header().slot_count == 4
        assert not (root / "session.json").exists()
        assert (root / "locks.ofd").exists()
        assert (root / "v1" / "control.bin").is_file()
    finally:
        second.close()


def test_spawned_process_joins_live_generation(tmp_path: Path) -> None:
    root = tmp_path / "decoded"
    parent = _open_session(root)
    context = multiprocessing.get_context("spawn")
    joined = context.Event()
    release = context.Event()
    output = context.Queue()
    process = context.Process(
        target=_join_in_child,
        args=(str(root), joined, release, output),
    )
    process.start()
    try:
        assert joined.wait(timeout=10)
        assert output.get(timeout=1) == parent.session_id
    finally:
        release.set()
        process.join(timeout=10)
        parent.close()
    assert process.exitcode == 0


def test_live_incompatible_configuration_never_resets_peer(tmp_path: Path) -> None:
    root = tmp_path / "decoded"
    first = _open_session(root)
    control_path = root / "v1" / "control.bin"
    try:
        with pytest.raises(ParquetRGSessionInUseError, match="live incompatible"):
            _open_session(root, configuration="sha256:" + "56" * 32)
        assert control_path.exists()
        assert first.control.header().slot_count == 4
    finally:
        first.close()


def test_no_owner_restart_resets_v1_but_keeps_lock_inode(tmp_path: Path) -> None:
    root = tmp_path / "decoded"
    first = _open_session(root)
    first_session_id = first.session_id
    lock_stat = (root / "locks.ofd").stat()
    stale = first.entries_dir / "000000" / "0000000000000001.arrow"
    stale.parent.mkdir()
    stale.write_bytes(b"stale")
    first.close()

    with _open_session(root) as second:
        assert second.session_id != first_session_id
        assert not stale.exists()
        assert (root / "locks.ofd").stat().st_ino == lock_stat.st_ino


def test_no_owner_restart_refuses_symlinked_generation(tmp_path: Path) -> None:
    root = tmp_path / "decoded"
    session = _open_session(root)
    session.close()

    generation = root / "v1"
    shutil.rmtree(generation)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    sentinel = foreign / "keep-me"
    sentinel.write_bytes(b"safe")
    generation.symlink_to(foreign, target_is_directory=True)

    with pytest.raises(ParquetRGSessionError, match="symlinked"):
        _open_session(root)
    assert sentinel.read_bytes() == b"safe"


def test_unmarked_nonempty_root_is_never_claimed_or_deleted(tmp_path: Path) -> None:
    root = tmp_path / "decoded"
    root.mkdir()
    foreign = root / "foreign-data"
    foreign.write_bytes(b"keep me")
    with pytest.raises(ParquetRGSessionError, match="unowned non-empty"):
        _open_session(root)
    assert foreign.read_bytes() == b"keep me"


def test_missing_stable_lock_with_published_control_is_not_reset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "decoded"
    session = _open_session(root)
    control_path = root / "v1" / "control.bin"
    session.close()
    (root / "locks.ofd").unlink()

    with pytest.raises(ParquetRGSessionError, match="stable lock file is missing"):
        _open_session(root)
    assert control_path.exists()


def test_incomplete_first_bootstrap_rebuilds_stable_lock(tmp_path: Path) -> None:
    root = tmp_path / "decoded"
    session = _open_session(root)
    session.close()
    control_path = root / "v1" / "control.bin"
    control_path.unlink()
    (root / "locks.ofd").write_bytes(b"")

    with _open_session(root) as recovered:
        assert control_path.exists()


def test_replaced_stable_lock_is_not_treated_as_no_live_owners(tmp_path: Path) -> None:
    root = tmp_path / "decoded"
    session = _open_session(root)
    control_path = root / "v1" / "control.bin"
    session.close()
    (root / "locks.ofd").write_text("different-lock-id\n", encoding="ascii")

    with pytest.raises(ParquetRGSessionError, match="lock identity changed"):
        _open_session(root)
    assert control_path.exists()
