# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import weakref
from unittest import mock

import pytest

from zephon._internal.utils import gc_freeze
from zephon._internal.utils.gc_freeze import GCFreeze, resolve_gc_freeze


@pytest.fixture
def gc_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(gc_freeze.gc, "get_freeze_count", lambda: 0)
    monkeypatch.setattr(gc_freeze.gc, "collect", lambda: calls.append("collect"))
    monkeypatch.setattr(gc_freeze.gc, "freeze", lambda: calls.append("freeze"))
    monkeypatch.setattr(gc_freeze.gc, "unfreeze", lambda: calls.append("unfreeze"))
    return calls


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("option", [None, False, True])
def test_resolve_uses_actual_execution_mode(option: bool | None, mtp: bool) -> None:
    assert resolve_gc_freeze(option, mtp=mtp) is (mtp if option is None else option)


@pytest.mark.parametrize("value", [0, 1, "true", []])
def test_resolve_rejects_non_boolean_values(value: object) -> None:
    with pytest.raises(ValueError, match="gc_freeze must be"):
        resolve_gc_freeze(value, mtp=True)


def test_disabled_scope_never_changes_gc(gc_calls: list[str]) -> None:
    guard = GCFreeze(False)
    guard.close()
    del guard
    assert gc_calls == []


def test_overlapping_scopes_unfreeze_only_after_last_close(gc_calls: list[str]) -> None:
    first, second = GCFreeze(True), GCFreeze(True)
    try:
        assert gc_calls == ["collect", "freeze"]
        first.close()
        first.close()
        assert gc_calls == ["collect", "freeze"]
    finally:
        first.close()
        second.close()
    assert gc_calls == ["collect", "freeze", "unfreeze"]


def test_destructor_releases_an_abandoned_scope(gc_calls: list[str]) -> None:
    guard = GCFreeze(True)
    ref = weakref.ref(guard)
    del guard
    assert ref() is None
    assert gc_calls == ["collect", "freeze", "unfreeze"]


def test_existing_application_freeze_is_preserved(
    monkeypatch: pytest.MonkeyPatch, gc_calls: list[str]
) -> None:
    monkeypatch.setattr(gc_freeze.gc, "get_freeze_count", lambda: 42)
    guard = GCFreeze(True)
    guard.close()
    del guard
    assert gc_calls == []


@pytest.mark.parametrize("operation", ["collect", "freeze"])
def test_initialization_failure_does_not_leave_a_scope(
    monkeypatch: pytest.MonkeyPatch, gc_calls: list[str], operation: str
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(
            gc_freeze.gc, operation, mock.Mock(side_effect=RuntimeError("fail"))
        )
        with pytest.raises(RuntimeError, match="fail"):
            GCFreeze(True)
    gc_calls.clear()
    guard = GCFreeze(True)
    guard.close()
    assert gc_calls == ["collect", "freeze", "unfreeze"]


def test_fork_does_not_adopt_parent_ownership(
    monkeypatch: pytest.MonkeyPatch, gc_calls: list[str]
) -> None:
    parent = GCFreeze(True)
    pid = os.getpid()
    gc_freeze._after_fork()
    monkeypatch.setattr(gc_freeze.os, "getpid", lambda: pid + 1)
    monkeypatch.setattr(gc_freeze.gc, "get_freeze_count", lambda: 42)
    child = GCFreeze(True)
    parent.close()
    child.close()
    # Child scopes preserve the inherited frozen generation.
    assert gc_calls == ["collect", "freeze"]
