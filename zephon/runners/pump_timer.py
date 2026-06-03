# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Pump-thread timer used by concurrent stage runners.

Centralizes phase-timing and counter bookkeeping for an operator's pump
thread. Call sites use the :meth:`PumpTimer.measure` /
:meth:`measure_excluding` context managers and the small :meth:`note`
helper so the surrounding runner code reads as straight control flow.

When the timer is disabled (``collect_stats=False`` on the owning
operator state) every method short-circuits to a no-op so the overhead
is a single ``if not self.enabled: return`` plus one
``perf_counter_ns`` read at the start/end of each ``with``.

The seven ``*_ns`` buckets are designed to be disjoint slices of pump
wall time. Outer measurements that wrap inner ones (e.g. ``idle_drain``
wraps the sweep that may grow ``result_wait``) use
:meth:`measure_excluding` so the residual stays in the outer bucket and
the buckets remain additive. The bucket and counter field set is defined
once on :class:`PumpCounts`; this class inherits those fields so the
shape is named in exactly one place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from zephon.observability.stats import PumpCounts, PumpTimingDelta


@dataclass
class PumpTimer(PumpCounts):
    """Per-operator pump-thread timer.

    Owned by ``ConcurrentOperatorState`` and accessed by the operator's
    pump thread. Bucket / counter fields are inherited from
    :class:`PumpCounts`; this class adds the enable flag, the cadence
    bookkeeping, and the static descriptors that get copied into each
    emitted delta. When ``enabled`` is False every public method is a
    no-op and the inherited integer fields stay at zero.
    """

    enabled: bool = False
    last_flush_ns: int = 0
    stage_index: int = field(default=0, repr=False)
    op_index: int = field(default=0, repr=False)
    stage_name: str = field(default="", repr=False)
    op_name: str = field(default="", repr=False)

    def start_window(self, now_ns: int | None = None) -> None:
        """Initialize the window-start timestamp."""
        if not self.enabled:
            return
        self.last_flush_ns = now_ns if now_ns is not None else time.perf_counter_ns()

    def measure(self, bucket: str) -> "_Measure":
        """Context manager that accumulates wall time into ``<bucket>_ns``."""
        if not self.enabled:
            return _NULL_MEASURE
        return _Measure(self, bucket, ())

    def measure_excluding(self, bucket: str, *exclude: str) -> "_Measure":
        """Like :meth:`measure`, minus any growth in the named excluded buckets.

        Used when an outer measurement wraps inner ones that already
        account for their own time (e.g. ``idle_drain`` wraps a sweep
        that may grow ``result_wait_ns``; the outer bucket should hold
        only the residual time so the seven buckets stay disjoint).
        """
        if not self.enabled:
            return _NULL_MEASURE
        return _Measure(self, bucket, exclude)

    def note(self, counter: str, count: int = 1) -> None:
        """Bump a named counter (no-op when the timer is disabled)."""
        if not self.enabled:
            return
        setattr(self, counter, getattr(self, counter) + count)

    def should_flush(self, now_ns: int, interval_ns: int) -> bool:
        """Return True if at least ``interval_ns`` has elapsed since last flush."""
        if not self.enabled:
            return False
        return (now_ns - self.last_flush_ns) >= interval_ns

    def flush(self, now_ns: int | None = None) -> PumpTimingDelta | None:
        """Snapshot current accumulators into a delta and reset the window.

        Returns ``None`` when the timer is disabled. Otherwise always
        returns a delta — callers that want to suppress empty windows
        can check counters / bucket sums themselves.
        """
        if not self.enabled:
            return None
        delta = PumpTimingDelta(
            stage_index=self.stage_index,
            op_index=self.op_index,
            stage_name=self.stage_name,
            name=self.op_name,
        )
        self.copy_to(delta)
        self.reset_counts()
        self.last_flush_ns = now_ns if now_ns is not None else time.perf_counter_ns()
        return delta


class _Measure:
    """Context manager returned by :meth:`PumpTimer.measure[_excluding]`.

    On exit, adds ``elapsed - sum(excluded_deltas)`` nanoseconds to the
    named ``<bucket>_ns`` field. A no-op when the parent timer is
    disabled.
    """

    __slots__ = ("_p", "_bucket_attr", "_exclude_attrs", "_t0", "_exclude_before")

    def __init__(self, timer: PumpTimer, bucket: str, exclude: tuple[str, ...]):
        self._p = timer
        self._bucket_attr = f"{bucket}_ns"
        self._exclude_attrs = tuple(f"{e}_ns" for e in exclude)
        self._t0 = 0
        self._exclude_before: tuple[int, ...] = ()

    def __enter__(self) -> "_Measure":
        if not self._p.enabled:
            return self
        self._t0 = time.perf_counter_ns()
        if self._exclude_attrs:
            self._exclude_before = tuple(
                getattr(self._p, attr) for attr in self._exclude_attrs
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._p.enabled:
            return
        elapsed = time.perf_counter_ns() - self._t0
        if self._exclude_attrs:
            deltas = sum(
                getattr(self._p, attr) - before
                for attr, before in zip(self._exclude_attrs, self._exclude_before)
            )
            elapsed = max(0, elapsed - deltas)
        setattr(
            self._p, self._bucket_attr, getattr(self._p, self._bucket_attr) + elapsed
        )


# Singleton returned by PumpTimer.measure/measure_excluding when disabled,
# so the hot path never allocates a _Measure object.
_NULL_MEASURE = _Measure(PumpTimer(enabled=False), "", ())
