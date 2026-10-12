# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex worker limits and environment overrides."""

import builtins
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from zephon._internal.utils import thread_utils


def test_vortex_cap_applies_configured_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setitem(
        sys.modules, "vortex", SimpleNamespace(set_worker_threads=calls.append)
    )
    monkeypatch.delenv("ZEPHON_VORTEX_THREADS", raising=False)
    monkeypatch.delenv("VORTEX_MAX_THREADS", raising=False)
    thread_utils.cap_vortex_threads()
    assert calls == [1]
    monkeypatch.setenv("ZEPHON_VORTEX_THREADS", "2")
    thread_utils.cap_vortex_threads()
    assert calls == [1, 2]
    monkeypatch.setenv("ZEPHON_VORTEX_THREADS", "0")
    thread_utils.cap_vortex_threads()
    assert calls == [1, 2]
    monkeypatch.setenv("ZEPHON_VORTEX_THREADS", "-1")
    thread_utils.cap_vortex_threads()
    assert calls == [1, 2]


def test_vortex_native_thread_setting_and_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setitem(
        sys.modules, "vortex", SimpleNamespace(set_worker_threads=calls.append)
    )
    monkeypatch.delenv("ZEPHON_VORTEX_THREADS", raising=False)
    monkeypatch.setenv("VORTEX_MAX_THREADS", "2")
    thread_utils.cap_vortex_threads()
    assert calls == []
    monkeypatch.setenv("ZEPHON_VORTEX_THREADS", "1")
    thread_utils.cap_vortex_threads()
    assert calls == [1]


def test_vortex_invalid_thread_limit_names_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZEPHON_VORTEX_THREADS", "auto")
    with pytest.raises(ValueError, match="ZEPHON_VORTEX_THREADS must be an integer"):
        thread_utils.cap_vortex_threads()


def test_vortex_cap_without_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "vortex", raising=False)
    monkeypatch.delenv("VORTEX_MAX_THREADS", raising=False)
    monkeypatch.setenv("ZEPHON_VORTEX_THREADS", "1")
    native_import = builtins.__import__

    def reject_vortex(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name != "vortex", "thread suppression must not import optional Vortex"
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_vortex)
    thread_utils.cap_vortex_threads()
