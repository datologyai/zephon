# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Process-wide, lifetime-scoped ownership of Python's permanent GC generation."""

from __future__ import annotations

import gc
import os
import threading

_lock = threading.RLock()
_users = 0
_owns_freeze = False


def _after_fork() -> None:
    """Inherited freezes belong to the parent, not to new child scopes."""
    global _lock, _users, _owns_freeze
    _lock = threading.RLock()
    _users = 0
    _owns_freeze = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def resolve_gc_freeze(value: object, *, mtp: bool) -> bool:
    """Default to freezing only when the engine actually runs in an MTP hub."""
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"gc_freeze must be True, False, or None; got {value!r}")
    return mtp if value is None else value


class GCFreeze:
    """Keep a Zephon-owned freeze until the last overlapping scope closes.

    An existing permanent generation is left alone: Python cannot selectively
    unfreeze only the objects Zephon added to it. A caller must not manually
    freeze/unfreeze while a scope is active. This guard never changes automatic
    GC's enabled state or thresholds, and unfreezing does not force collection.
    """

    def __init__(self, enabled: bool) -> None:
        self._active = False
        self._pid = os.getpid()
        if not enabled:
            return
        global _users, _owns_freeze
        with _lock:
            if _users == 0 and gc.get_freeze_count() == 0:
                gc.collect()
                gc.freeze()
                _owns_freeze = True
            _users += 1
            self._active = True

    def close(self) -> None:
        """Release this scope, unfreezing only a generation Zephon owns."""
        global _users, _owns_freeze
        with _lock:
            if not self._active:
                return
            self._active = False
            if self._pid != os.getpid():
                return
            _users -= 1
            if _users == 0 and _owns_freeze:
                _owns_freeze = False
                gc.unfreeze()

    def __del__(self) -> None:
        # Normal execution closes explicitly. This also handles abandoned guards
        # and partially initialized objects, including interpreter shutdown.
        try:
            self.close()
        except Exception:
            pass
