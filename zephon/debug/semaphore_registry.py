# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Debug registry for tracking SafeSemLock instances.

This module provides a registry that tracks semaphore creation and cleanup
for debugging purposes. Enable with ZEPHON_SEMAPHORE_DEBUG=1 environment variable.

The registry is used by:
- SafeSemLock.__init__(): registers semaphores on creation
- SafeSemLock.cleanup(): marks semaphores as cleaned
- dump_semaphore_registry(): prints debug info at shutdown
- debug/semaphore_tracker.py: reads for leak analysis
"""

from __future__ import annotations

import os
import threading

# Debug registry: maps semaphore name -> (source, cleaned_up)
_semaphore_registry: dict[str, tuple[str, bool]] = {}
_registry_lock = threading.Lock()


def _is_semaphore_debug_enabled() -> bool:
    """Check if semaphore debug tracking is enabled via env var."""
    return os.environ.get("ZEPHON_SEMAPHORE_DEBUG", "").lower() in ("1", "true", "yes")


def _register_semaphore(name: str, source: str) -> None:
    """Register a semaphore for debug tracking."""
    if not _is_semaphore_debug_enabled() or name is None:
        return
    with _registry_lock:
        _semaphore_registry[name] = (source, False)


def _mark_semaphore_cleaned(name: str) -> None:
    """Mark a semaphore as cleaned up."""
    if not _is_semaphore_debug_enabled() or name is None:
        return
    with _registry_lock:
        if name in _semaphore_registry:
            source, _ = _semaphore_registry[name]
            _semaphore_registry[name] = (source, True)


def dump_semaphore_registry() -> None:
    """Print debug info about tracked semaphores."""
    if not _is_semaphore_debug_enabled():
        return
    with _registry_lock:
        if not _semaphore_registry:
            return
        print(f"\n=== Semaphore Registry ({len(_semaphore_registry)} total) ===")
        not_cleaned = [(n, s) for n, (s, c) in _semaphore_registry.items() if not c]
        cleaned = [(n, s) for n, (s, c) in _semaphore_registry.items() if c]
        print(f"Cleaned: {len(cleaned)}, Not cleaned: {len(not_cleaned)}")
        if not_cleaned:
            print("\nNOT CLEANED:")
            for name, source in sorted(not_cleaned, key=lambda x: x[1]):
                print(f"  {name}: {source}")


__all__ = [
    "_semaphore_registry",
    "_register_semaphore",
    "_mark_semaphore_cleaned",
    "dump_semaphore_registry",
]
