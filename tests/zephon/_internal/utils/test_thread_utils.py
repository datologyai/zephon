# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex's runtime cap is lazy and scoped to the current process."""

import os
import sys
import threading
from types import SimpleNamespace

import pytest

from zephon._internal.utils import thread_utils


def test_vortex_cap_once_per_process_and_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setitem(
        sys.modules, "vortex", SimpleNamespace(set_worker_threads=calls.append)
    )
    monkeypatch.setattr(thread_utils, "_vortex_thread_settings", None)
    process = SimpleNamespace(environ=os.environ, getpid=lambda: 100)
    monkeypatch.setattr(thread_utils, "os", process)
    monkeypatch.delenv("ZEPHON_VORTEX_THREADS", raising=False)
    thread_utils.cap_vortex_threads()
    thread_utils.cap_vortex_threads()
    assert calls == [1]
    process.getpid = lambda: 101
    thread_utils.cap_vortex_threads()
    monkeypatch.setenv("ZEPHON_VORTEX_THREADS", "2")
    thread_utils.cap_vortex_threads()
    assert calls == [1, 1, 2]
    monkeypatch.setenv("ZEPHON_VORTEX_THREADS", "0")
    thread_utils.cap_vortex_threads()
    assert calls == [1, 1, 2]
    monkeypatch.setenv("ZEPHON_VORTEX_THREADS", "-1")
    with pytest.raises(ValueError, match="negative"):
        thread_utils.cap_vortex_threads()


def test_vortex_cap_does_not_import_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "vortex", raising=False)
    monkeypatch.setattr(thread_utils, "_vortex_thread_settings", None)
    thread_utils.cap_vortex_threads()
    assert "vortex" not in sys.modules
    assert thread_utils._vortex_thread_settings is None


def test_vortex_cap_waits_for_module_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SimpleNamespace()
    calls: list[int] = []
    monkeypatch.setitem(sys.modules, "vortex", module)
    monkeypatch.setattr(thread_utils, "_vortex_thread_settings", None)
    monkeypatch.delenv("ZEPHON_VORTEX_THREADS", raising=False)
    thread_utils.cap_vortex_threads()
    assert thread_utils._vortex_thread_settings is None
    module.set_worker_threads = calls.append
    thread_utils.cap_vortex_threads()
    assert calls == [1]


def test_fork_reset_discards_inherited_locked_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setitem(
        sys.modules, "vortex", SimpleNamespace(set_worker_threads=calls.append)
    )
    monkeypatch.delenv("ZEPHON_VORTEX_THREADS", raising=False)
    inherited_lock = threading.Lock()
    monkeypatch.setattr(thread_utils, "_vortex_thread_lock", inherited_lock)
    monkeypatch.setattr(thread_utils, "_vortex_thread_settings", (os.getpid(), 1))
    with inherited_lock:
        thread_utils._reset_vortex_threads_after_fork()
        assert thread_utils._vortex_thread_settings is None
        assert thread_utils._vortex_thread_lock.acquire(blocking=False)
        thread_utils._vortex_thread_lock.release()
        thread_utils.cap_vortex_threads()
    assert calls == [1]
