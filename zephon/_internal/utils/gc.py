"""
GC helpers used as a defensive guard around CPython GC quirks.

See https://github.com/python/cpython/issues/142531 for background on why
we sometimes disable the collector in tight loops or startup paths.
"""

import gc
import sys
import sysconfig
import time
import traceback
from contextlib import contextmanager

_GIL_DISABLED: bool | None = None


def is_gil_disabled() -> bool:
    """Return True if running on a free-threaded (no-GIL) Python build."""
    global _GIL_DISABLED
    if _GIL_DISABLED is not None:
        return _GIL_DISABLED

    check = getattr(sys, "_is_gil_enabled", None)
    if callable(check):
        try:
            _GIL_DISABLED = not bool(check())
            return _GIL_DISABLED
        except Exception:
            print("Error checking sys._is_gil_enabled.", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    try:
        gil_disabled = sysconfig.get_config_var("Py_GIL_DISABLED")
        if gil_disabled is not None:
            _GIL_DISABLED = bool(gil_disabled)
            return _GIL_DISABLED
    except Exception:
        print("Error checking Py_GIL_DISABLED sysconfig var.", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    _GIL_DISABLED = False
    return _GIL_DISABLED


def cleanup_semaphores(semaphores: list) -> None:
    """
    Explicitly clean up semaphores to avoid resource tracker warnings.

    In free-threaded Python, semaphore finalizers are deferred and may not run
    before the resource tracker checks at shutdown. This function handles both
    SafeSemaphore (preferred) and regular mp.Semaphore objects.

    For SafeSemaphore: Calls the cleanup() method which coordinates with GC
    via a lock to ensure idempotent, race-free cleanup.

    For regular Semaphore: Falls back to finding and invoking the Finalize
    callback directly (best effort, may still race with GC).

    Args:
        semaphores: List of Semaphore or SafeSemaphore objects to clean up.
    """
    if not semaphores:
        return

    for sem in semaphores:
        try:
            # SafeSemaphore has a cleanup() method that handles coordination
            if hasattr(sem, "cleanup") and callable(sem.cleanup):
                sem.cleanup()
            else:
                # Fallback for regular mp.Semaphore (best effort)
                _cleanup_regular_semaphore(sem)
        except Exception:
            print(
                f"Error cleaning up semaphore {sem!r}.",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)


def _cleanup_regular_semaphore(sem: object) -> None:
    """Best-effort cleanup for regular mp.Semaphore objects.

    This may still race with GC in free-threaded Python, but is kept as a
    fallback for any code that creates semaphores without using SafeSemaphore.
    """
    try:
        from multiprocessing import util
    except ImportError:
        return

    registry = getattr(util, "_finalizer_registry", None)
    if registry is None:
        return

    name = getattr(getattr(sem, "_semlock", None), "name", None)
    if name is None:
        return

    # Find and call the finalizer for this semaphore
    for finalizer in list(registry.values()):
        args = getattr(finalizer, "_args", ())
        if args and args[0] == name:
            finalizer()
            break


def collect_with_finalizers(cycles: int = 3, yield_ms: float = 1.0) -> None:
    """
    Run garbage collection, ensuring deferred finalizers complete.

    On free-threaded Python (PEP 703), finalizers are deferred and run after
    gc.collect() returns. This function runs multiple cycles with small yields
    to give finalizer threads time to execute.

    On standard Python (with GIL), this is equivalent to a single gc.collect().

    Args:
        cycles: Number of GC cycles to run on free-threaded Python.
        yield_ms: Milliseconds to sleep between cycles to allow finalizers to run.
    """
    gc.collect()
    if is_gil_disabled() and cycles > 1:
        for _ in range(cycles - 1):
            time.sleep(yield_ms / 1000.0)
            gc.collect()


@contextmanager
def disable_gc():
    """
    Context manager that temporarily disables the Garbage Collector.

    It checks the current state upon entry and only re-enables the GC
    on exit if it was originally enabled. This is safe to nest or use
    in environments where GC might already be disabled.
    """
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
