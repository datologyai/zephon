# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Compatibility patches for litdata library.

litdata creates a module-level Lock in litdata/utilities/breakpoint.py for
its MPPdb debugger. This Lock is only used when calling breakpoint() in workers,
but it's created eagerly at import time.

When worker processes are terminated (rather than exiting cleanly), this Lock's
finalizer never runs, leaking the semaphore and causing resource_tracker warnings.

This module patches litdata to use lazy Lock creation - the Lock is only created
if debugging is actually used. This eliminates the leak for normal workloads.
"""

from __future__ import annotations

import sys
import threading
from types import ModuleType
from typing import Any

from wrapt.importer import register_post_import_hook

_LOCK = threading.Lock()
_WATCH = ("litdata.utilities.breakpoint",)


class _LazyLock:
    """A lazily-created multiprocessing Lock.

    Only creates the actual Lock when acquire() or __enter__() is called.
    This avoids semaphore allocation for code that never uses the lock.
    """

    def __init__(self) -> None:
        self._real_lock: Any = None
        self._init_lock = threading.Lock()

    def _ensure_lock(self) -> Any:
        if self._real_lock is None:
            with self._init_lock:
                if self._real_lock is None:
                    import multiprocessing

                    self._real_lock = multiprocessing.Lock()
        return self._real_lock

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        return self._ensure_lock().acquire(*args, **kwargs)

    def release(self) -> None:
        if self._real_lock is not None:
            self._real_lock.release()

    def __enter__(self) -> _LazyLock:
        self._ensure_lock().acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._real_lock is not None:
            self._real_lock.release()


def _apply_patch(module: ModuleType) -> None:
    """Replace litdata's _stdin_lock with a lazy version."""
    with _LOCK:
        if getattr(module, "_zephon_patched", False):
            return

        # Only patch if the lock exists and is a real mp.Lock
        lock = getattr(module, "_stdin_lock", None)
        if lock is None:
            return

        # Check if it's already our lazy lock
        if isinstance(lock, _LazyLock):
            return

        # Clean up the existing lock
        semlock = getattr(lock, "_semlock", None)
        if semlock is not None:
            name = getattr(semlock, "name", None)
            if name is not None:
                # Cancel the original Lock's Finalize callback first
                # Otherwise it will try to sem_unlink after we already did
                from multiprocessing import util

                registry = getattr(util, "_finalizer_registry", None)
                if registry is not None:
                    for finalizer in list(registry.values()):
                        args = getattr(finalizer, "_args", ())
                        if args and args[0] == name:
                            finalizer.cancel()
                            break

                # Unregister and unlink the existing lock
                from multiprocessing.resource_tracker import unregister

                try:
                    unregister(name, "semaphore")
                except Exception:
                    pass
                try:
                    from _multiprocessing import sem_unlink

                    sem_unlink(name)
                except Exception:
                    pass

        # Replace with lazy lock
        module._stdin_lock = _LazyLock()  # type: ignore[attr-defined]
        module._zephon_patched = True  # type: ignore[attr-defined]


def install_litdata_patch() -> None:
    """Install the litdata Lock cleanup patch."""
    # If module is already loaded, patch immediately
    for n in _WATCH:
        mod = sys.modules.get(n)
        if mod is not None:
            _apply_patch(mod)

    # Hook future imports
    if register_post_import_hook is not None:
        for n in _WATCH:
            register_post_import_hook(_apply_patch, n)


__all__ = ["install_litdata_patch"]
