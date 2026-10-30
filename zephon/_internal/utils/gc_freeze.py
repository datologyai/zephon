# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""GC freezing shared by overlapping pipeline iterations."""

from __future__ import annotations

import gc
import os
import threading

_lock = threading.RLock()
_users = 0
_owns_freeze = False


def _after_fork() -> None:
    """Reset the child lock and freeze ownership without unfreezing objects."""
    global _lock, _users, _owns_freeze
    _lock = threading.RLock()
    _users = 0
    _owns_freeze = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def resolve_gc_freeze(value: object, *, mtp: bool) -> bool:
    """Resolve GC freezing, enabled by default only in MTP mode."""
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"gc_freeze must be True, False, or None; got {value!r}")
    return mtp if value is None else value


class GCFreeze:
    """Share a GC freeze until the last active scope closes.

    Existing freezes are preserved. Callers must not call ``gc.freeze()`` or
    ``gc.unfreeze()`` while a scope is active.
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
        """Unfreeze Zephon-owned objects when the last scope closes."""
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
        # Best-effort cleanup during destruction and interpreter shutdown.
        try:
            self.close()
        except Exception:
            pass
