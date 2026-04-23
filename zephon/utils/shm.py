# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory (``/dev/shm``) pressure detection and diagnostics.

All functions in this module use only the standard library — no torch or other
optional dependencies are required.  This makes the module safe to import in
any environment, including those without GPU support.
"""

from __future__ import annotations

import errno
import os
import random as _random
import sys
import time

_SHM_FREE_THRESHOLD = 0.05  # require 5% free before re-queuing for serialization

# Backpressure retry tuning — shared by NamedQueue and SHM coalescing.
_SHM_RETRY_BASE_BACKOFF = 0.5  # initial sleep seconds
_SHM_RETRY_MAX_BACKOFF = 30.0  # cap on exponential backoff component
_SHM_RETRY_MAX_JITTER = 5.0  # uniform random jitter added to each sleep
_SHM_RETRY_LOG_EVERY = 10  # log warning every N attempts (always log first)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def is_shm_error(e: BaseException) -> bool:
    """Return ``True`` if *e* (or a chained cause) indicates ``/dev/shm`` exhaustion.

    Walks ``__cause__`` and ``__context__`` chains because pickle and torch
    often wrap the underlying ``OSError(ENOSPC)`` in a ``RuntimeError``.
    """
    seen: set[int] = set()
    to_check: list[BaseException] = [e]
    while to_check:
        exc = to_check.pop()
        eid = id(exc)
        if eid in seen:
            continue
        seen.add(eid)
        if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
            return True
        if "No space left on device" in str(exc):
            return True
        if exc.__cause__ is not None:
            to_check.append(exc.__cause__)
        if exc.__context__ is not None:
            to_check.append(exc.__context__)
    return False


# ---------------------------------------------------------------------------
# /dev/shm capacity checks (cgroup-aware)
# ---------------------------------------------------------------------------


def read_cgroup_memory_limit() -> int:
    """Return cgroup v2 ``memory.max`` in bytes, or ``-1`` if unavailable."""
    try:
        with open("/proc/self/cgroup") as f:
            cgroup_line = f.read().strip()
        parts = cgroup_line.split("::", 1)
        if len(parts) < 2:
            return -1
        cg_path = parts[1].strip()
        with open(f"/sys/fs/cgroup{cg_path}/memory.max") as f:
            val = f.read().strip()
            if val == "max":
                return -1
            return int(val)
    except (OSError, ValueError, IndexError):
        return -1


def shm_has_free_space(threshold: float = _SHM_FREE_THRESHOLD) -> bool:
    """Return ``True`` if ``/dev/shm`` has at least *threshold* fraction free.

    Accounts for cgroup v2 memory limits: when a cgroup cap is smaller than
    the tmpfs mount size, uses the cgroup limit as the effective total.  This
    handles K8s pods, ``docker --memory``, and SLURM cgroup setups where
    ``statvfs`` alone would report misleading free space.
    """
    try:
        st = os.statvfs("/dev/shm")
        total = st.f_blocks * st.f_frsize
        if total == 0:
            return True
        free = st.f_bavail * st.f_frsize
        cgroup_limit = read_cgroup_memory_limit()
        if cgroup_limit > 0 and cgroup_limit < total:
            used = total - free
            total = cgroup_limit
            free = max(0, total - used)
        return (free / total) >= threshold
    except OSError:
        return True  # Non-Linux or /dev/shm unavailable — don't block


def wait_for_shm_space(label: str, *, threshold: float = _SHM_FREE_THRESHOLD) -> int:
    """Block with exponential-jittered backoff until ``/dev/shm`` has free space.

    Intended for use after a ``share_memory_()`` or ``shm_open()`` failure.
    The caller should catch the ``ENOSPC`` exception, call this function to
    wait, then retry the allocation.

    Returns the number of backoff sleeps performed (``0`` if space was
    already available on the first check).
    """
    attempt = 0
    while not shm_has_free_space(threshold):
        backoff = min(
            _SHM_RETRY_BASE_BACKOFF * (2**attempt),
            _SHM_RETRY_MAX_BACKOFF,
        )
        jitter = _random.uniform(0, _SHM_RETRY_MAX_JITTER)
        sleep_time = backoff + jitter

        if attempt == 0 or (attempt + 1) % _SHM_RETRY_LOG_EVERY == 0:
            print(
                f"WARNING: {label} — SHM pressure (attempt {attempt + 1}), "
                f"waiting {sleep_time:.1f}s for /dev/shm space. "
                f"{shm_usage_str()}",
                file=sys.stderr,
                flush=True,
            )

        time.sleep(sleep_time)
        attempt += 1
    return attempt


def shm_usage_str() -> str:
    """Return a compact string like ``shm=954.2/1000.0GB(95.4%)`` for log messages.

    When a cgroup v2 memory limit is smaller than the tmpfs mount, reports
    usage against the effective (cgroup-adjusted) total — matching the view
    that :func:`shm_has_free_space` uses — and appends the raw tmpfs size
    for context.
    """
    try:
        st = os.statvfs("/dev/shm")
        tmpfs_total = st.f_blocks * st.f_frsize
        if tmpfs_total == 0:
            return "shm=N/A"
        free = st.f_bavail * st.f_frsize
        used = tmpfs_total - free
        cgroup_limit = read_cgroup_memory_limit()
        suffix = ""
        if cgroup_limit > 0 and cgroup_limit < tmpfs_total:
            # Use cgroup-adjusted values (mirrors shm_has_free_space logic).
            total = cgroup_limit
            free = max(0, total - used)
            suffix = f" [tmpfs={tmpfs_total / (1024**3):.1f}GB]"
        else:
            total = tmpfs_total
        pct = (used / total) * 100
        return (
            f"shm={used / (1024**3):.1f}/{total / (1024**3):.1f}GB({pct:.1f}%){suffix}"
        )
    except OSError:
        return "shm=N/A"
