# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Fault handling utilities for debugging hangs and crashes.

This module provides utilities for diagnosing process hangs and crashes:
- setup_faulthandler(): Enable crash diagnosis via signal handlers
- dump_all_threads(): Dump stack traces of all threads
- ShutdownWatchdog: Context manager that dumps stacks if shutdown takes too long

Environment variables:
- ZEPHON_FAULTHANDLER=1: Enable faulthandler for crash signal handling
- ZEPHON_SHUTDOWN_WATCHDOG=<seconds>: Enable watchdog with specified timeout
"""

from __future__ import annotations

import faulthandler
import os
import signal
import sys
import threading
from typing import Any

from zephon.utils.rank import rank_ctx


def setup_faulthandler() -> None:
    """Set up faulthandler for crash diagnosis.

    When ``ZEPHON_FAULTHANDLER=1`` is set:

    - Enables automatic traceback dump on SIGSEGV, SIGFPE, SIGABRT, SIGBUS, SIGILL
    - Registers SIGUSR1/SIGUSR2 to manually trigger a full traceback dump
      (``kill -USR1 <pid>``)

    This has zero runtime overhead — signal handlers only fire on actual
    signals.  Runs in every process that imports (or explicitly invokes)
    this setup: the main process, ProcessStageRunner workers (via
    module-level import of ``zephon.runners.process``), and the MTP
    subprocess (explicit call from ``_mtp_worker``).  Idempotent thanks
    to ``faulthandler.is_enabled()``.
    """
    if not os.environ.get("ZEPHON_FAULTHANDLER"):
        return

    # Idempotent: repeated calls (e.g. during spawn-worker re-import) are no-ops.
    if faulthandler.is_enabled():
        return

    # Enable default crash signal handlers (SIGSEGV, SIGFPE, SIGABRT, SIGBUS, SIGILL)
    faulthandler.enable(file=sys.stderr, all_threads=True)

    # Register SIGUSR1 for manual stack dump (Unix only)
    if hasattr(signal, "SIGUSR1"):
        try:
            faulthandler.register(
                signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False
            )
        except (OSError, AttributeError):
            pass  # May fail in some environments

    # Register SIGUSR2 as well for convenience
    if hasattr(signal, "SIGUSR2"):
        try:
            faulthandler.register(
                signal.SIGUSR2, file=sys.stderr, all_threads=True, chain=False
            )
        except (OSError, AttributeError):
            pass

    print(
        f"[Zephon] faulthandler enabled ({rank_ctx()}). "
        + "Send SIGUSR1/SIGUSR2 to dump all thread stacks.",
        file=sys.stderr,
        flush=True,
    )


def dump_all_threads(reason: str) -> None:  # pragma: no cover - diagnostics helper
    """Dump stack traces of all threads to stderr.

    Args:
        reason: A message explaining why the dump was triggered.
    """
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"WATCHDOG: {reason}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    print(f"{'=' * 60}\n", file=sys.stderr, flush=True)


class ShutdownWatchdog:
    """Watchdog that dumps thread stacks if shutdown takes too long.

    Use as a context manager around shutdown code to automatically dump
    all thread stacks if the shutdown exceeds the specified timeout.

    Example:
        with ShutdownWatchdog(30.0, "close()"):
            # shutdown code here
            ...

    Args:
        timeout: Timeout in seconds. If <= 0, the watchdog is disabled.
        label: A label describing the operation being monitored.
    """

    def __init__(self, timeout: float, label: str):
        self._timeout = timeout
        self._label = label
        self._timer: threading.Timer | None = None

    def __enter__(self) -> "ShutdownWatchdog":
        if self._timeout > 0:
            self._timer = threading.Timer(
                self._timeout,
                dump_all_threads,
                args=[f"{self._label} exceeded {self._timeout}s"],
            )
            self._timer.daemon = True
            self._timer.start()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
