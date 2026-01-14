# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Debug tool for tracking semaphore registration and identifying leaks.

This module monkey-patches Python's resource_tracker to capture stack traces
when semaphores are registered. It compares this against zephon's SafeSemLock
registry to identify semaphores that bypass our tracking.

Enable with: ZEPHON_SEMAPHORE_LEAK_DEBUG=1

Call dump_semaphore_leak_report() to print the analysis, or it will be
printed automatically via a weakref finalizer when the tracker is garbage
collected (typically at interpreter shutdown).

Example output:
    === Semaphore Leak Analysis ===
    Registered with resource tracker: 62
    Tracked by SafeSemLock: 23
    Unregistered (cleaned): 23
    LEAKED (registered but not unregistered): 39

    Leaked semaphores not in SafeSemLock registry:
    /mp-abc123 registered at 14:32:01.234
      File "zephon/runners/process.py", line 105, in __init__
        super().__init__(maxsize, ctx=ctx)
      ...
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
import weakref
from dataclasses import dataclass
from typing import Callable

# Only activate if env var is set
_DEBUG_ENABLED = os.environ.get("ZEPHON_SEMAPHORE_LEAK_DEBUG", "").lower() in (
    "1",
    "true",
    "yes",
)

_lock = threading.Lock()


@dataclass
class RegistrationInfo:
    """Information captured when a semaphore is registered."""

    name: str
    timestamp: float
    stack_trace: str
    thread_name: str
    thread_id: int


# Global registries - these are the actual data stores
_registrations: dict[str, RegistrationInfo] = {}
_unregistrations: set[str] = set()
_original_register: Callable | None = None
_original_unregister: Callable | None = None
_patched = False
_report_printed = False


def _capture_registration(name: str, rtype: str) -> None:
    """Capture info when a semaphore is registered."""
    if rtype != "semaphore":
        return

    info = RegistrationInfo(
        name=name,
        timestamp=time.time(),
        stack_trace="".join(traceback.format_stack()[:-2]),  # Exclude this frame
        thread_name=threading.current_thread().name,
        thread_id=threading.get_ident(),
    )

    with _lock:
        _registrations[name] = info


def _capture_unregistration(name: str, rtype: str) -> None:
    """Capture when a semaphore is unregistered."""
    if rtype != "semaphore":
        return

    with _lock:
        _unregistrations.add(name)


def _patched_register(name: str, rtype: str) -> None:
    """Patched register function that captures info then calls original."""
    _capture_registration(name, rtype)
    if _original_register:
        _original_register(name, rtype)


def _patched_unregister(name: str, rtype: str) -> None:
    """Patched unregister function that captures info then calls original."""
    _capture_unregistration(name, rtype)
    if _original_unregister:
        _original_unregister(name, rtype)


def _format_time(ts: float) -> str:
    """Format timestamp as HH:MM:SS.mmm."""
    local = time.localtime(ts)
    ms = int((ts % 1) * 1000)
    return f"{local.tm_hour:02d}:{local.tm_min:02d}:{local.tm_sec:02d}.{ms:03d}"


def dump_semaphore_leak_report(*, force: bool = False) -> None:
    """Print the semaphore leak analysis report.

    Can be called manually at any point to see current state.
    Also called automatically via weakref finalizer at shutdown.

    Args:
        force: If True, print even from worker processes with no leaks.
    """
    global _report_printed

    import multiprocessing

    proc_name = multiprocessing.current_process().name
    is_main = proc_name == "MainProcess"

    # Avoid printing multiple times per process
    with _lock:
        if _report_printed:
            return
        _report_printed = True

    # Import here to avoid circular imports
    try:
        from zephon.debug.semaphore_registry import _semaphore_registry
    except ImportError:
        _semaphore_registry = {}

    with _lock:
        registered = set(_registrations.keys())
        unregistered = _unregistrations.copy()
        safesemlock_tracked = set(_semaphore_registry.keys())
        safesemlock_cleaned = {
            name for name, (_, cleaned) in _semaphore_registry.items() if cleaned
        }

    leaked = registered - unregistered

    # Skip workers with no leaks (unless forced) to reduce noise
    if not is_main and len(leaked) == 0 and not force:
        return

    prefix = f"[{proc_name}] "
    print(f"\n{prefix}" + "=" * 50, file=sys.stderr)
    print(
        f"{prefix}=== Semaphore Leak Analysis (ZEPHON_SEMAPHORE_LEAK_DEBUG) ===",
        file=sys.stderr,
    )
    print(f"{prefix}" + "=" * 50, file=sys.stderr)
    print(
        f"{prefix}Registered with resource tracker: {len(registered)}", file=sys.stderr
    )
    print(f"{prefix}Unregistered (cleaned up): {len(unregistered)}", file=sys.stderr)
    print(
        f"{prefix}LEAKED (registered but not unregistered): {len(leaked)}",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    print(f"{prefix}SafeSemLock tracked: {len(safesemlock_tracked)}", file=sys.stderr)
    print(f"{prefix}SafeSemLock cleaned: {len(safesemlock_cleaned)}", file=sys.stderr)

    # Find leaked semaphores that aren't in SafeSemLock registry
    leaked_not_tracked = leaked - safesemlock_tracked

    if leaked_not_tracked:
        print(file=sys.stderr)
        print(
            f"{prefix}Leaked semaphores NOT in SafeSemLock registry ({len(leaked_not_tracked)}):",
            file=sys.stderr,
        )
        print(f"{prefix}" + "-" * 50, file=sys.stderr)

        for name in sorted(leaked_not_tracked):
            info = _registrations.get(name)
            if info:
                print(file=sys.stderr)
                print(
                    f"{prefix}{name} registered at {_format_time(info.timestamp)} [thread: {info.thread_name}]",
                    file=sys.stderr,
                )
                # Print condensed stack trace (last 10 frames)
                lines = info.stack_trace.strip().split("\n")
                # Show last 20 lines (10 frames, 2 lines each)
                for line in lines[-20:]:
                    print(f"{prefix}  {line}", file=sys.stderr)
            else:
                print(
                    f"{prefix}{name} - no registration info captured", file=sys.stderr
                )

    # Find leaked semaphores that ARE in SafeSemLock but weren't cleaned
    leaked_tracked_not_cleaned = leaked & safesemlock_tracked - safesemlock_cleaned

    if leaked_tracked_not_cleaned:
        print(file=sys.stderr)
        print(
            f"{prefix}Leaked semaphores in SafeSemLock but NOT cleaned ({len(leaked_tracked_not_cleaned)}):",
            file=sys.stderr,
        )
        print(f"{prefix}" + "-" * 50, file=sys.stderr)

        for name in sorted(leaked_tracked_not_cleaned):
            source, _ = _semaphore_registry.get(name, ("unknown", False))
            info = _registrations.get(name)
            ts = _format_time(info.timestamp) if info else "unknown"
            print(
                f"{prefix}  {name} source={source} registered_at={ts}", file=sys.stderr
            )

    print(f"{prefix}" + "=" * 50, file=sys.stderr)


def _invoke_report() -> None:
    """Static callback for weakref.finalize - no instance reference needed."""
    try:
        dump_semaphore_leak_report()
    except Exception:
        pass


class _DebugTrackerAnchor:
    """Prevent reference holder - destructor triggers leak report.

    We use weakref.finalize instead of __del__ because it's more reliable
    and doesn't keep references that could prevent cleanup.
    """

    pass


# Module-level reference to prevent GC until shutdown
_anchor: _DebugTrackerAnchor | None = None
_finalizer: weakref.finalize | None = None


def install_debug_hooks() -> None:
    """Install monkey-patches on resource_tracker.

    This should be called early in the program, before any multiprocessing
    objects are created.
    """
    global _original_register, _original_unregister, _patched, _anchor, _finalizer

    if _patched:
        return

    if not _DEBUG_ENABLED:
        return

    try:
        import multiprocessing
        import multiprocessing.resource_tracker as rt_module
        from multiprocessing.resource_tracker import (
            _resource_tracker,
        )

        # Save originals
        _original_register = _resource_tracker.register
        _original_unregister = _resource_tracker.unregister

        # Patch instance methods
        _resource_tracker.register = _patched_register  # type: ignore[method-assign]
        _resource_tracker.unregister = _patched_unregister  # type: ignore[method-assign]

        # Also patch module-level functions (in case code imports them directly)
        _original_module_register = rt_module.register
        _original_module_unregister = rt_module.unregister

        def _patched_module_register(name, rtype):
            _capture_registration(name, rtype)
            return _original_module_register(name, rtype)

        def _patched_module_unregister(name, rtype):
            _capture_unregistration(name, rtype)
            return _original_module_unregister(name, rtype)

        rt_module.register = _patched_module_register  # type: ignore[assignment]
        rt_module.unregister = _patched_module_unregister  # type: ignore[assignment]

        _patched = True

        # Set up finalizer in ALL processes to capture worker leaks
        # Use weakref.finalize to trigger report at shutdown
        _anchor = _DebugTrackerAnchor()
        _finalizer = weakref.finalize(_anchor, _invoke_report)

        # Only print verbose messages in main process to reduce noise
        if multiprocessing.current_process().name == "MainProcess":
            # Verification
            print(
                f"[ZEPHON_SEMAPHORE_LEAK_DEBUG] Patched _resource_tracker.register: "
                f"{_resource_tracker.register is _patched_register}",
                file=sys.stderr,
            )
            print(
                f"[ZEPHON_SEMAPHORE_LEAK_DEBUG] Patched rt_module.register: "
                f"{rt_module.register is _patched_module_register}",
                file=sys.stderr,
            )
            print(
                "[ZEPHON_SEMAPHORE_LEAK_DEBUG] Semaphore tracking installed",
                file=sys.stderr,
            )

    except Exception as e:
        print(
            f"[ZEPHON_SEMAPHORE_LEAK_DEBUG] Failed to install hooks: {e}",
            file=sys.stderr,
        )


# Auto-install if debug is enabled
if _DEBUG_ENABLED:
    install_debug_hooks()
