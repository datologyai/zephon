# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty

import pytest

from zephon._internal.io import ofd_lock as ofd_lock_module
from zephon._internal.io.ofd_lock import (
    OFDLockFile,
    OFDLockMode,
    OFDLockUnavailable,
    ofd_backend_info,
    probe_ofd_support,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def require_ofd_support(tmp_path_factory: pytest.TempPathFactory) -> None:
    probe_dir = tmp_path_factory.mktemp("ofd-support")
    result = probe_ofd_support(probe_dir)
    if not result.supported:
        pytest.skip(result.reason or "OFD locks are unavailable")


def _create_lock_file(path: Path) -> None:
    path.touch(mode=0o600)


def _spawn_try_lock(path: str, start: int, results: Queue[bool]) -> None:
    with OFDLockFile(path) as lock_file:
        lease = lock_file.try_acquire(start=start, mode=OFDLockMode.EXCLUSIVE)
        results.put(lease is not None)
        if lease is not None:
            lease.close()


def test_probe_checks_actual_filesystem(tmp_path: Path) -> None:
    result = probe_ofd_support(tmp_path)

    assert result.supported, result.reason
    assert result.backend == ofd_backend_info()
    assert not list(tmp_path.iterdir())


def test_anchor_open_waits_for_fork_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "locks.ofd"
    _create_lock_file(path)
    started = threading.Event()
    opened = threading.Event()
    original_open = ofd_lock_module._open_regular_file

    def observed_open(
        open_path: str | os.PathLike[str],
    ) -> tuple[int, os.stat_result]:
        opened.set()
        return original_open(open_path)

    monkeypatch.setattr(ofd_lock_module, "_open_regular_file", observed_open)
    created: list[OFDLockFile] = []
    errors: list[BaseException] = []

    def create_lock_file() -> None:
        started.set()
        try:
            created.append(OFDLockFile(path))
        except BaseException as exc:
            errors.append(exc)

    ofd_lock_module._before_fork()
    worker = threading.Thread(target=create_lock_file)
    try:
        worker.start()
        assert started.wait(timeout=1)
        assert not opened.wait(timeout=0.05)
    finally:
        ofd_lock_module._after_fork_parent()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert opened.is_set()
    assert not errors
    created[0].close()


def test_exact_ranges_do_not_serialize_each_other(tmp_path: Path) -> None:
    path = tmp_path / "locks.ofd"
    _create_lock_file(path)

    with OFDLockFile(path) as lock_file:
        first = lock_file.try_acquire(start=10, mode=OFDLockMode.EXCLUSIVE)
        second = lock_file.try_acquire(start=11, mode=OFDLockMode.EXCLUSIVE)
        conflicting = lock_file.try_acquire(start=10, mode=OFDLockMode.SHARED)

        assert first is not None
        assert second is not None
        assert conflicting is None
        second.close()
        first.close()


def test_shared_leases_coexist_and_block_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "locks.ofd"
    _create_lock_file(path)

    with OFDLockFile(path) as lock_file:
        first = lock_file.try_acquire(start=42, mode=OFDLockMode.SHARED)
        second = lock_file.try_acquire(start=42, mode=OFDLockMode.SHARED)
        exclusive = lock_file.try_acquire(start=42, mode=OFDLockMode.EXCLUSIVE)

        assert first is not None
        assert second is not None
        assert exclusive is None
        second.close()
        first.close()

        exclusive = lock_file.try_acquire(start=42, mode=OFDLockMode.EXCLUSIVE)
        assert exclusive is not None
        exclusive.close()


def test_independent_spawned_process_observes_exact_range(tmp_path: Path) -> None:
    path = tmp_path / "locks.ofd"
    _create_lock_file(path)
    context = multiprocessing.get_context("spawn")

    with OFDLockFile(path) as lock_file:
        held = lock_file.try_acquire(start=5, mode=OFDLockMode.EXCLUSIVE)
        assert held is not None

        for start, expected in ((5, False), (6, True)):
            results = context.Queue()
            process = context.Process(
                target=_spawn_try_lock,
                args=(str(path), start, results),
            )
            process.start()
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                pytest.fail("spawned OFD-lock probe did not terminate")
            assert process.exitcode == 0
            try:
                observed = results.get(timeout=1)
            except Empty:
                pytest.fail("spawned OFD-lock probe returned no result")
            finally:
                results.close()
                results.join_thread()
            assert observed is expected

        held.close()


def test_deadline_wait_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "locks.ofd"
    _create_lock_file(path)

    with OFDLockFile(path) as lock_file:
        held = lock_file.try_acquire(start=7, mode=OFDLockMode.EXCLUSIVE)
        assert held is not None

        started = time.monotonic()
        blocked = lock_file.acquire(
            start=7,
            mode=OFDLockMode.SHARED,
            timeout=0.02,
            initial_backoff=0.001,
            max_backoff=0.004,
        )
        elapsed = time.monotonic() - started

        assert blocked is None
        assert 0.015 <= elapsed < 0.2
        held.close()


def test_deadline_wait_reuses_one_open_file_description(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "locks.ofd"
    _create_lock_file(path)

    with OFDLockFile(path) as lock_file:
        held = lock_file.try_acquire(start=7, mode=OFDLockMode.EXCLUSIVE)
        assert held is not None
        original_open = ofd_lock_module._open_regular_file
        open_calls = 0

        def counted_open(
            open_path: str | os.PathLike[str],
        ) -> tuple[int, os.stat_result]:
            nonlocal open_calls
            open_calls += 1
            return original_open(open_path)

        monkeypatch.setattr(ofd_lock_module, "_open_regular_file", counted_open)
        blocked = lock_file.acquire(
            start=7,
            mode=OFDLockMode.SHARED,
            timeout=0.02,
            initial_backoff=0.001,
            max_backoff=0.004,
        )

        assert blocked is None
        assert open_calls == 1
        held.close()


def test_replacing_lock_path_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "locks.ofd"
    replacement = tmp_path / "replacement"
    _create_lock_file(path)

    with OFDLockFile(path) as lock_file:
        _create_lock_file(replacement)
        replacement.replace(path)

        with pytest.raises(OFDLockUnavailable, match="inode changed"):
            lock_file.try_acquire(start=0, mode=OFDLockMode.EXCLUSIVE)


def test_replacing_lock_path_during_retry_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "locks.ofd"
    replacement = tmp_path / "replacement"
    _create_lock_file(path)

    with OFDLockFile(path) as lock_file:
        held = lock_file.try_acquire(start=0, mode=OFDLockMode.EXCLUSIVE)
        assert held is not None
        _create_lock_file(replacement)
        real_sleep = ofd_lock_module.time.sleep
        replaced = False

        def replace_during_sleep(delay: float) -> None:
            nonlocal replaced
            if not replaced:
                replacement.replace(path)
                replaced = True
            real_sleep(delay)

        monkeypatch.setattr(ofd_lock_module.time, "sleep", replace_during_sleep)
        with pytest.raises(OFDLockUnavailable, match="inode changed"):
            lock_file.acquire(
                start=0,
                mode=OFDLockMode.SHARED,
                timeout=0.1,
                initial_backoff=0.001,
            )
        held.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_child_closes_inherited_descriptors_without_unlocking_parent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "locks.ofd"
    _create_lock_file(path)
    lock_file = OFDLockFile(path)
    held = lock_file.try_acquire(start=3, mode=OFDLockMode.EXCLUSIVE)
    assert held is not None

    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - assertions are observed by exit code
        try:
            assert held.closed
            assert lock_file.closed
            held.close()
            try:
                lock_file.try_acquire(start=3, mode=OFDLockMode.SHARED)
            except OFDLockUnavailable:
                os._exit(0)
            os._exit(2)
        except BaseException:
            os._exit(3)

    _, status = os.waitpid(child_pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0

    independent = OFDLockFile(path)
    assert independent.try_acquire(start=3, mode=OFDLockMode.SHARED) is None
    held.close()
    released = independent.try_acquire(start=3, mode=OFDLockMode.SHARED)
    assert released is not None
    released.close()
    independent.close()
    lock_file.close()


def test_exclusive_lease_converts_to_shared_without_a_gap(tmp_path: Path) -> None:
    path = tmp_path / "locks.ofd"
    _create_lock_file(path)
    with OFDLockFile(path) as lock_file:
        lease = lock_file.try_acquire(start=2, mode=OFDLockMode.EXCLUSIVE)
        assert lease is not None
        assert lease.mode is OFDLockMode.EXCLUSIVE
        assert lease.start == 2
        assert lease.length == 1

        lease.convert(OFDLockMode.SHARED)
        assert lease.mode is OFDLockMode.SHARED
        peer = lock_file.try_acquire(start=2, mode=OFDLockMode.SHARED)
        blocked = lock_file.try_acquire(start=2, mode=OFDLockMode.EXCLUSIVE)
        assert peer is not None
        assert blocked is None
        peer.close()
        lease.close()


def test_failed_upgrade_reports_contention_and_preserves_shared_lease(
    tmp_path: Path,
) -> None:
    path = tmp_path / "locks.ofd"
    _create_lock_file(path)
    with OFDLockFile(path) as lock_file:
        lease = lock_file.try_acquire(start=2, mode=OFDLockMode.SHARED)
        peer = lock_file.try_acquire(start=2, mode=OFDLockMode.SHARED)
        assert lease is not None
        assert peer is not None

        assert not lease.try_convert(OFDLockMode.EXCLUSIVE)
        assert lease.mode is OFDLockMode.SHARED
        with pytest.raises(BlockingIOError, match="conversion conflicted"):
            lease.convert(OFDLockMode.EXCLUSIVE)
        assert lease.mode is OFDLockMode.SHARED
        assert lock_file.try_acquire(start=2, mode=OFDLockMode.EXCLUSIVE) is None

        peer.close()
        assert lease.try_convert(OFDLockMode.EXCLUSIVE)
        assert lease.mode is OFDLockMode.EXCLUSIVE
        assert lock_file.try_acquire(start=2, mode=OFDLockMode.SHARED) is None
        lease.close()
