# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex's runtime cap is lazy and scoped to the current process."""

import sys
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
    monkeypatch.setattr(thread_utils.os, "getpid", lambda: 100)
    monkeypatch.delenv("ZEPHON_VORTEX_THREADS", raising=False)
    thread_utils.cap_vortex_threads()
    thread_utils.cap_vortex_threads()
    assert calls == [1]
    monkeypatch.setattr(thread_utils.os, "getpid", lambda: 101)
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
