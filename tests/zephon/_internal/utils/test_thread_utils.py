# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex thread limits follow the optional library suppression helpers."""

import sys
from types import SimpleNamespace

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
    thread_utils.cap_vortex_threads()
    assert calls == [1]
    monkeypatch.setenv("ZEPHON_VORTEX_THREADS", "2")
    thread_utils.cap_vortex_threads()
    assert calls == [1, 2]
    monkeypatch.setenv("ZEPHON_VORTEX_THREADS", "0")
    thread_utils.cap_vortex_threads()
    assert calls == [1, 2]
    monkeypatch.setenv("ZEPHON_VORTEX_THREADS", "-1")
    with pytest.raises(ValueError, match="negative"):
        thread_utils.cap_vortex_threads()


def test_vortex_cap_without_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "vortex", None)
    thread_utils.cap_vortex_threads()
