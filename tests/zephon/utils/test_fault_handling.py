# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``zephon.utils.fault_handling.setup_faulthandler``.

The function modifies process-global state (``faulthandler.enable()``)
and writes its startup banner to ``sys.stderr``.  We use ``capfd``
(file-descriptor-level capture) instead of ``capsys`` because
``faulthandler.enable`` requires a file with a real ``fileno()``,
which ``capsys``'s fake ``sys.stderr`` does not expose.

An autouse fixture captures and restores the prior enablement so the
suite stays clean for later tests that may or may not want faulthandler
installed.
"""

from __future__ import annotations

import faulthandler
import multiprocessing as mp
import os
import sys

import pytest

from zephon.utils.fault_handling import setup_faulthandler


@pytest.fixture(autouse=True)
def _restore_faulthandler_state():
    """Save/restore faulthandler enablement around each test."""
    was_enabled = faulthandler.is_enabled()
    yield
    if was_enabled and not faulthandler.is_enabled():
        target = sys.__stderr__ or sys.stderr
        try:
            faulthandler.enable(file=target, all_threads=True)
        except (OSError, ValueError):
            pass  # best-effort restore on exotic stderr setups
    elif not was_enabled and faulthandler.is_enabled():
        faulthandler.disable()


def _spawned_worker_probes_faulthandler(q: mp.Queue) -> None:
    """Module-level helper: mimic what a ``spawn``-started worker sees.

    Runs in a subprocess whose ``current_process().name`` is not
    ``MainProcess``; prior to the resilient-workers branch, an early
    return gated on that name would leave faulthandler disabled in
    workers.
    """
    import faulthandler as _fh
    import os as _os

    _os.environ["ZEPHON_FAULTHANDLER"] = "1"
    from zephon.utils.fault_handling import setup_faulthandler as _setup

    _setup()
    q.put(_fh.is_enabled())


def test_no_op_without_env_var(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture
) -> None:
    """With ``ZEPHON_FAULTHANDLER`` unset, ``setup_faulthandler`` is a no-op."""
    monkeypatch.delenv("ZEPHON_FAULTHANDLER", raising=False)
    was_enabled = faulthandler.is_enabled()
    setup_faulthandler()
    assert faulthandler.is_enabled() == was_enabled
    captured = capfd.readouterr()
    assert "faulthandler enabled" not in captured.err


def test_no_op_when_env_var_is_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty string is treated as falsy (``os.environ.get`` returns ``""``)."""
    monkeypatch.setenv("ZEPHON_FAULTHANDLER", "")
    was_enabled = faulthandler.is_enabled()
    setup_faulthandler()
    assert faulthandler.is_enabled() == was_enabled


def test_enables_and_prints_banner(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture
) -> None:
    """First call enables faulthandler and emits the rank-tagged banner."""
    monkeypatch.setenv("ZEPHON_FAULTHANDLER", "1")
    if faulthandler.is_enabled():
        faulthandler.disable()
    setup_faulthandler()
    assert faulthandler.is_enabled()
    banner = capfd.readouterr().err
    assert "faulthandler enabled" in banner
    # Banner embeds rank_ctx() which includes the current pid.
    assert f"pid={os.getpid()}" in banner


def test_idempotent_second_call_is_noop(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture
) -> None:
    """Second call short-circuits via ``faulthandler.is_enabled()``.

    Regression guard for the worker re-import path: a spawned worker
    imports ``zephon.runners.process``, which imports (transitively) the
    fault-handler setup — and Python re-exec of the test suite should
    not produce duplicate banners or register handlers twice.
    """
    monkeypatch.setenv("ZEPHON_FAULTHANDLER", "1")
    if faulthandler.is_enabled():
        faulthandler.disable()
    setup_faulthandler()
    _ = capfd.readouterr()  # drain banner from first call
    setup_faulthandler()
    second = capfd.readouterr().err
    assert "faulthandler enabled" not in second, (
        "second call should not re-print the banner"
    )
    assert faulthandler.is_enabled()


@pytest.mark.timeout(30)
def test_enables_in_spawned_worker_process() -> None:
    """Non-MainProcess workers must also enable faulthandler.

    The resilient-workers branch removed the old
    ``current_process().name != "MainProcess"`` early return so spawned
    ``ProcessStageRunner`` workers (and the MTP subprocess) can dump
    stack traces on SIGUSR1/SIGUSR2 too.  Regression guard that it
    doesn't come back.
    """
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_spawned_worker_probes_faulthandler, args=(q,))
    proc.start()
    proc.join(timeout=20)
    assert proc.exitcode == 0, f"worker exited with code {proc.exitcode}"
    assert q.get(timeout=5) is True, (
        "faulthandler should be enabled in spawned (non-MainProcess) workers"
    )
