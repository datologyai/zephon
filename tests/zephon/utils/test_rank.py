# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``zephon.utils.rank.rank_ctx``."""

from __future__ import annotations

import importlib
import os
import re

import pytest

from zephon.utils import rank as rank_mod


def test_rank_ctx_format_matches_rank_local_pid_triple() -> None:
    """``rank_ctx()`` returns ``rank=<r> local=<l> pid=<n>`` exactly."""
    text = rank_mod.rank_ctx()
    assert re.fullmatch(r"rank=\S+ local=\S+ pid=\d+", text), (
        f"unexpected rank_ctx format: {text!r}"
    )


def test_rank_ctx_pid_is_current_process() -> None:
    """``pid=`` is read on each call, not captured at import time."""
    text = rank_mod.rank_ctx()
    assert f"pid={os.getpid()}" in text


def test_rank_ctx_defaults_to_question_mark_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent RANK/LOCAL_RANK env vars, ``rank_ctx`` emits ``?`` placeholders."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    # _RANK / _LOCAL_RANK are captured at import time, so reload to pick up
    # the scrubbed environment.
    reloaded = importlib.reload(rank_mod)
    try:
        text = reloaded.rank_ctx()
        assert text.startswith("rank=? local=? pid=")
    finally:
        # Restore the original module so later tests see the process env.
        importlib.reload(rank_mod)


def test_rank_ctx_reads_env_at_import_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reloading with RANK/LOCAL_RANK set threads those values through."""
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    reloaded = importlib.reload(rank_mod)
    try:
        text = reloaded.rank_ctx()
        assert text.startswith("rank=2 local=1 pid=")
    finally:
        importlib.reload(rank_mod)


def test_rank_ctx_ignores_runtime_env_mutation_after_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutating RANK after import does not change ``rank_ctx``.

    Design invariant: rank is frozen at import time so every spawned
    worker inherits the parent's rank even if some later code path
    mutates the env.  Only the PID refreshes per call.
    """
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "3")
    reloaded = importlib.reload(rank_mod)
    try:
        baseline = reloaded.rank_ctx()
        assert "rank=7 local=3" in baseline

        monkeypatch.setenv("RANK", "99")
        monkeypatch.setenv("LOCAL_RANK", "99")

        after = reloaded.rank_ctx()
        assert "rank=7 local=3" in after, (
            "rank_ctx should reflect the import-time env, not the current env"
        )
    finally:
        importlib.reload(rank_mod)
