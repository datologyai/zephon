# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Disk-capacity preflight for Zephon's on-disk caches.

Stdlib-only (like :mod:`zephon._internal.utils.shm`) so it imports anywhere.

No cgroup handling (unlike shm): cgroup v2 has no disk-space controller, and
container space limits (XFS project quotas) already show through the filesystem.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

# Warn when filling the cache to its limit would leave less than this
# fraction of the device's total capacity free for other writers.
_CACHE_HEADROOM_FRACTION = 0.05

# Escape hatch for filesystems where the free-space query succeeds but misreports
# (e.g. NFS/Lustre quotas, thin provisioning).
_DISABLE_ENV = "ZEPHON_DISABLE_CACHE_SPACE_CHECK"

_GiB = 1024**3


class InsufficientCacheSpaceError(RuntimeError):
    """Configured cache limit cannot fit on the device holding the cache root."""

    def __init__(
        self,
        *,
        root: Path,
        limit_bytes: int,
        free_bytes: int,
        existing_bytes: int,
        device_total: int,
    ) -> None:
        self.root = root
        self.limit_bytes = limit_bytes
        self.free_bytes = free_bytes
        self.existing_bytes = existing_bytes
        self.device_total = device_total
        usable = free_bytes + existing_bytes
        super().__init__(
            "Configured cache limit does not fit on the device holding the "
            "cache root.\n"
            f"  cache root          : {root}\n"
            f"  configured limit    : {limit_bytes / _GiB:.1f}GiB\n"
            f"  device free         : {free_bytes / _GiB:.1f}GiB\n"
            f"  reclaimable (cache) : {existing_bytes / _GiB:.1f}GiB "
            "(bytes already under the cache root)\n"
            f"  usable (free+cache) : {usable / _GiB:.1f}GiB\n"
            f"  device total        : {device_total / _GiB:.1f}GiB\n"
            f"Reduce the configured cache limits to at most {usable / _GiB:.1f}GiB, free "
            "space on the device, or move cache.root to a larger device. Set "
            f"{_DISABLE_ENV}=1 to bypass this check (e.g. on filesystems that "
            "misreport free space)."
        )


def _existing_ancestor(path: Path) -> tuple[Path, os.stat_result]:
    probe = path
    while True:
        try:
            return probe, probe.stat()
        except FileNotFoundError:
            if probe == probe.parent:
                raise
            probe = probe.parent


def device_id(path: Path) -> int | None:
    """Return the device containing *path*, probing its nearest existing ancestor."""
    try:
        _ancestor, info = _existing_ancestor(path)
        return info.st_dev
    except OSError:
        return None


def device_space(path: Path) -> tuple[int, int] | None:
    """Return ``(device_total_bytes, free_bytes)`` for the device holding *path*.

    Inspect the nearest existing ancestor — the cache root may not exist yet,
    and the ancestor's filesystem is where its bytes would land. Free is
    ``f_bavail`` (non-root-reserved). ``None`` on ``OSError``.
    """
    try:
        ancestor, _info = _existing_ancestor(path)
        usage = shutil.disk_usage(ancestor)
    except OSError:
        return None
    return usage.total, usage.free


def dir_usage_bytes(path: Path) -> int:
    """Sum on-disk bytes of all files under *path* (0 if missing).

    Counts allocation (the quantity statvfs free space moves by) rather than
    ``st_size``. Files vanishing mid-walk (concurrent eviction) are skipped.
    """
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                # st_blocks counts 512-byte units by POSIX convention on every
                # platform, independent of the filesystem's own block size.
                total += os.lstat(os.path.join(dirpath, name)).st_blocks * 512
            except OSError:
                continue
    return total


def check_cache_disk_space(
    root: Path,
    limit_bytes: int | None,
    *,
    existing_bytes: int | Callable[[], int] | None = None,
    warn_fraction: float | None = _CACHE_HEADROOM_FRACTION,
) -> None:
    """Validate that the configured cache limit fits on the device holding *root*.

    Warm-restart aware: bytes already under the cache root count toward the
    limit (a warm restart reuses them, a cache reset reclaims them), so::

        usable = device_free + bytes_currently_under_root
        fail   if limit_bytes > usable
        warn   if usable - limit_bytes < warn_fraction * device_total

    Transient overshoot from in-flight downloads (``.tmp`` + zip/raw pairs)
    is deliberately not modeled; the headroom warning is the guard rail for
    cutting it that close.

    Args:
        root: Cache root directory (may not exist yet).
        limit_bytes: Configured on-disk cache limit; ``None`` skips the check.
        existing_bytes: Bytes already under the cache root, or a lazy function
            computing them when the caller needs a multi-root tally. A concrete
            value skips the ``os.walk``; ``None`` walks ``root`` when needed.
        warn_fraction: Device-total fraction below which the headroom warning
            fires. ``None`` disables the warning branch (per-worker callers,
            so only the engine preflight warns once per rank).

    Raises:
        InsufficientCacheSpaceError: If the limit cannot fit even after
            reclaiming everything already under the cache root.
    """
    if limit_bytes is None:
        return
    if os.environ.get(_DISABLE_ENV):
        logger.debug("Cache disk-space check disabled via %s", _DISABLE_ENV)
        return
    space = device_space(root)
    if space is None:
        # Permissive: never block a working setup over a statvfs failure.
        logger.debug(
            "Cache disk-space check skipped (statvfs unavailable for %s)", root
        )
        return
    device_total, free = space

    # The walk only ever adds to usable, so when free alone clears the limit
    # plus warning headroom neither branch can fire — skip the walk.
    headroom_needed = (
        int(warn_fraction * device_total) if warn_fraction is not None else 0
    )
    if free - limit_bytes >= headroom_needed:
        return

    if callable(existing_bytes):
        existing = existing_bytes()
    else:
        existing = (
            existing_bytes if existing_bytes is not None else dir_usage_bytes(root)
        )
    usable = free + existing
    if limit_bytes > usable:
        raise InsufficientCacheSpaceError(
            root=root,
            limit_bytes=limit_bytes,
            free_bytes=free,
            existing_bytes=existing,
            device_total=device_total,
        )
    headroom = usable - limit_bytes
    if warn_fraction is not None and headroom < headroom_needed:
        logger.warning(
            "Cache limit %.1fGiB fits on %s but leaves only %.1fGiB (%.1f%%) "
            "of the %.1fGiB device free once the cache fills — under the "
            "%.0f%% headroom threshold. Other writers on this device (logs, "
            "checkpoints, /tmp) may run out of space.",
            limit_bytes / _GiB,
            root,
            headroom / _GiB,
            100.0 * headroom / device_total,
            device_total / _GiB,
            100.0 * warn_fraction,
        )


def check_cache_disk_budgets(
    budgets: Iterable[tuple[Path, int]],
) -> None:
    """Validate cache budgets together when their roots share a filesystem.

    Limits on one device are additive. Existing bytes under disjoint roots are
    also additive, while nested roots are walked only once through their
    outermost root.
    """
    resolved = []
    for root, limit_bytes in budgets:
        canonical_root = Path(root).expanduser().resolve()
        resolved.append((canonical_root, limit_bytes, device_id(canonical_root)))
    groups: list[list[tuple[Path, int, int | None]]] = []
    for budget in resolved:
        root, _limit_bytes, device = budget
        matching = [
            index
            for index, group in enumerate(groups)
            if any(
                root == other_root
                or root in other_root.parents
                or other_root in root.parents
                or (device is not None and device == other_device)
                for other_root, _other_limit, other_device in group
            )
        ]
        if not matching:
            groups.append([budget])
            continue
        merged = []
        for index in matching:
            merged.extend(groups[index])
        merged.append(budget)
        for index in reversed(matching):
            groups.pop(index)
        groups.append(merged)

    for group in groups:
        if len(group) == 1:
            root, limit_bytes, _device = group[0]
            check_cache_disk_space(root, limit_bytes)
            continue

        roots = list(dict.fromkeys(root for root, _limit, _device in group))
        outer_roots = [
            root
            for root in roots
            if not any(other != root and other in root.parents for other in roots)
        ]

        def existing_bytes() -> int:
            return sum(dir_usage_bytes(root) for root in outer_roots)

        check_cache_disk_space(
            outer_roots[0],
            sum(limit_bytes for _root, limit_bytes, _device in group),
            existing_bytes=existing_bytes,
        )


__all__ = [
    "InsufficientCacheSpaceError",
    "check_cache_disk_budgets",
    "check_cache_disk_space",
    "device_id",
    "device_space",
    "dir_usage_bytes",
]
