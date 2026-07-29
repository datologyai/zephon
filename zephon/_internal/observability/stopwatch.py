# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Lightweight helpers for collecting perf counters when instrumentation is enabled."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar

__all__ = ["Stopwatch"]

_T = TypeVar("_T")


def _noop_start() -> int:
    """Pickle-friendly no-op start function."""
    return 0


def _noop_elapsed(_start_ns: int = 0) -> int:
    """Pickle-friendly no-op elapsed function."""
    return 0


class Stopwatch(Generic[_T]):
    """Utility that elides ``perf_counter_ns`` calls when metrics are disabled.

    The class is intentionally tiny: constructing it has negligible overhead, and
    every method short-circuits when ``enabled`` is ``False`` so hot paths can keep
    a single branch per measurement site.
    """

    __slots__ = ("enabled", "elapsed", "start")

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.elapsed = self._elapsed_enabled if enabled else _noop_elapsed
        self.start = time.perf_counter_ns if enabled else _noop_start

    def _elapsed_enabled(self, start_ns: int) -> int:
        """Return the elapsed nanoseconds from ``start_ns`` ."""
        return time.perf_counter_ns() - start_ns

    def time_call(
        self, func: Callable[..., _T], *args: Any, **kwargs: Any
    ) -> tuple[_T, int]:
        """Execute ``func`` and return ``(result, elapsed_ns)``.

        When disabled the elapsed component is always ``0``.
        """
        if not self.enabled:
            return func(*args, **kwargs), 0
        begin = time.perf_counter_ns()
        result = func(*args, **kwargs)
        return result, time.perf_counter_ns() - begin
