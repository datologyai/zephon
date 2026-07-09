# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe semaphore wrapper for free-threaded Python.

In free-threaded Python (PEP 703), GC finalizers run in background threads with
deferred finalization. This creates race conditions when both explicit cleanup
and GC try to clean up the same semaphore concurrently.

Python's SemLock._cleanup() is not idempotent - if sem_unlink fails (because
another thread already unlinked), unregister is never called, leading to
resource tracker warnings at shutdown.

SafeSemLock wraps any SemLock-based object (Lock, Semaphore, BoundedSemaphore)
with coordinated, idempotent cleanup that handles these race conditions gracefully.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import threading
import traceback
from multiprocessing import util
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext

from _multiprocessing import sem_unlink
from multiprocessing.resource_tracker import unregister

# Import debug registry functions
from zephon.debug.semaphore_registry import (
    _mark_semaphore_cleaned,
    _register_semaphore,
)

# Type variable for the wrapped semaphore type
T = TypeVar("T")


class SafeSemLock(Generic[T]):
    """
    Thread-safe wrapper for any SemLock-based synchronization object.

    Wraps Lock, Semaphore, and BoundedSemaphore with coordinated, idempotent
    cleanup that handles race conditions in free-threaded Python (PEP 703).

    The underlying mp synchronization objects register a Finalize callback that
    isn't idempotent - if sem_unlink fails, unregister is never called. In free-
    threaded Python where GC runs in background threads, this causes race
    conditions leading to FileNotFoundError and leaked semaphore warnings.

    This wrapper:
    1. Cancels the auto-registered Finalize immediately after creation
    2. Registers our own Finalize with idempotent cleanup
    3. Uses a lock to coordinate between explicit cleanup() and GC

    Thread safety:
    - _cleanup_lock: Prevents concurrent cleanup() and GC finalization
    - _state[0]: Mutable container allows both instance and static method to set flag
    - First caller (explicit or GC) sets state[0] = True and does cleanup
    - Subsequent callers see state[0] = True and skip

    Example flow when explicit cleanup races with GC:
        Time    cleanup()                      _static_cleanup() (GC)
        ────    ─────────                      ──────────────────────
         1      with lock: ← acquires lock
         2      check state[0] == False
         3      state[0] = True                with lock: ← BLOCKS (waiting)
         4      _do_unlink(name)                    │
         5      ← releases lock                     │
         6                                     ← acquires lock (unblocked)
         7                                     check state[0] == True
         8                                     return (skip cleanup)
    """

    def __init__(self, sem: T, *, source: str = ""):
        """Wrap an existing SemLock-based object with safe cleanup coordination.

        Args:
            sem: The Lock, Semaphore, or BoundedSemaphore to wrap.
            source: Debug identifier for tracking (used with ZEPHON_SEMAPHORE_DEBUG=1).
        """
        self._cleanup_lock = threading.Lock()
        # Use mutable container so static method and instance method can share state
        self._state: list[bool] = [False]  # [cleaned_up]
        self._sem: T = sem
        self._name: str | None = getattr(getattr(sem, "_semlock", None), "name", None)

        if self._name is not None:
            self._cancel_original_finalizer()
            self._register_safe_finalizer()
            _register_semaphore(self._name, source or "SafeSemLock")

    @classmethod
    def wrap(cls, sem: T, source: str = "") -> "SafeSemLock[T]":
        """Wrap an existing SemLock-based object (Lock, Semaphore, BoundedSemaphore).

        Use this to wrap semaphores created by other code (like Queue's internal
        locks) so they get coordinated cleanup in free-threaded Python.

        Args:
            sem: The Lock, Semaphore, or BoundedSemaphore to wrap.
            source: Debug identifier for tracking.

        Returns:
            A SafeSemLock wrapping the provided object.
        """
        return cls(sem, source=source)

    @classmethod
    def new_semaphore(
        cls, value: int = 1, *, ctx: BaseContext | None = None, source: str = ""
    ) -> "SafeSemLock":
        """Create a new Semaphore with safe cleanup.

        Args:
            value: Initial semaphore value.
            ctx: Multiprocessing context (defaults to spawn).
            source: Debug identifier for tracking.

        Returns:
            A SafeSemLock wrapping a new Semaphore.
        """
        ctx = ctx or mp.get_context("spawn")
        return cls(ctx.Semaphore(value), source=source or "SafeSemLock.new_semaphore")

    @classmethod
    def new_bounded_semaphore(
        cls, value: int = 1, *, ctx: BaseContext | None = None, source: str = ""
    ) -> "SafeSemLock":
        """Create a new BoundedSemaphore with safe cleanup.

        Args:
            value: Initial semaphore value.
            ctx: Multiprocessing context (defaults to spawn).
            source: Debug identifier for tracking.

        Returns:
            A SafeSemLock wrapping a new BoundedSemaphore.
        """
        ctx = ctx or mp.get_context("spawn")
        return cls(
            ctx.BoundedSemaphore(value),
            source=source or "SafeSemLock.new_bounded_semaphore",
        )

    @classmethod
    def new_lock(
        cls, *, ctx: BaseContext | None = None, source: str = ""
    ) -> "SafeSemLock":
        """Create a new Lock with safe cleanup.

        Args:
            ctx: Multiprocessing context (defaults to spawn).
            source: Debug identifier for tracking.

        Returns:
            A SafeSemLock wrapping a new Lock.
        """
        ctx = ctx or mp.get_context("spawn")
        return cls(ctx.Lock(), source=source or "SafeSemLock.new_lock")

    def _cancel_original_finalizer(self) -> None:
        """Cancel the auto-registered finalizer from the wrapped object."""
        registry = getattr(util, "_finalizer_registry", None)
        if registry is None:
            return

        for finalizer in list(registry.values()):
            args = getattr(finalizer, "_args", ())
            if args and args[0] == self._name:
                finalizer.cancel()
                break

    def _register_safe_finalizer(self) -> None:
        """Register our idempotent finalizer."""
        # Pass mutable state container so static method can set flag
        util.Finalize(
            self,
            self._static_cleanup,
            args=(self._name, self._cleanup_lock, self._state),
            exitpriority=0,
        )

    @staticmethod
    def _static_cleanup(name: str, lock: threading.Lock, state: list[bool]) -> None:
        """Static cleanup called by GC finalization - idempotent."""
        with lock:
            if state[0]:
                return
            state[0] = True
            SafeSemLock._do_unlink(name)

    @staticmethod
    def _do_unlink(name: str) -> None:
        """Actually perform the unlink and unregister."""
        # Unregister FIRST - prevents resource tracker warnings even if unlink fails
        try:
            unregister(name, "semaphore")
        except Exception:
            print(
                f"Error unregistering semaphore {name!r}.",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)

        # Unlink - may fail if already unlinked by another thread, that's fine
        try:
            sem_unlink(name)
        except FileNotFoundError:
            pass
        except Exception:
            print(
                f"Error unlinking semaphore {name!r}.",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)

    def cleanup(self) -> None:
        """Explicit cleanup - idempotent, can be called multiple times safely."""
        if self._name is None:
            return

        with self._cleanup_lock:
            if self._state[0]:
                return
            self._state[0] = True
            self._do_unlink(self._name)
            _mark_semaphore_cleaned(self._name)

    # Delegate semaphore interface
    def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
        """Acquire the semaphore."""
        return self._sem.acquire(block, timeout)

    def release(self) -> None:
        """Release the semaphore."""
        self._sem.release()

    def get_value(self) -> int:
        """Current semaphore value (``Queue.qsize()`` reads this)."""
        # Pure sem_getvalue read — no cleanup coordination to protect. 3.14's
        # qsize() calls get_value(); 3.12's uses _semlock._get_value() via the
        # _semlock property.
        return self._sem.get_value()

    def __enter__(self) -> "SafeSemLock[T]":
        self._sem.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self._sem.release()

    @property
    def _semlock(self):
        """Access to underlying semlock for compatibility with existing code."""
        return self._sem._semlock

    def __reduce__(self):
        """Make SafeSemLock picklable for multiprocessing.

        When passed to a worker process, the underlying semaphore is sent.
        The worker doesn't need cleanup coordination - only the parent process
        manages cleanup. The worker receives a regular mp object.
        """
        # Return the underlying semaphore - it has its own __reduce__ implementation
        return self._sem.__reduce__()
