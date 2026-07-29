"""SHM-only shared bookkeeping for the cache resolver.

This module owns exclusively the shared-memory lifecycle and numeric array
accessors that multiple processes consult to track shard residency. All
higher-level concerns — dense indexing, fingerprinting, session-state
decisions — live in :class:`zephon._internal.io.resolvers.cache.manager.CacheManager`.
"""

import contextlib
import weakref
from enum import IntEnum
from multiprocessing import shared_memory
from typing import Mapping

import numpy as np

_SHM_KEYS = ("states", "access", "sizes", "usage")


def _cleanup_shared_memory_regions(shms: list) -> None:
    """Release this process's mappings of the shared-memory regions.

    Designed for ``weakref.finalize``: takes the resources by value so it
    can run even when the parent object is already being collected.

    Deliberately does **not** call ``unlink()``. Unlinking removes the
    segment name OS-wide and would invalidate peer processes that are
    still attached via ``session.json``. Names are released only when the
    last owner drops the session (see
    :func:`zephon._internal.io.resolvers.cache.manager._release_session_owner`) or
    on the next manager init via :meth:`CacheSharedState.unlink_by_names`.
    """
    for shm in shms:
        if shm is None:
            continue
        with contextlib.suppress(Exception):
            shm.close()


class _ShardState(IntEnum):
    """Per-shard cache residency state shared across processes."""

    INVALID = 0
    REMOTE = 1
    PREPARING = 2
    LOCAL = 3


class CacheSharedState:
    """Owner of the four SHM arrays backing the cache.

    - ``states``   : uint8  per-shard residency state (see :class:`_ShardState`).
    - ``access``   : uint64 per-shard last-access time in ns (LRU).
    - ``sizes``    : int64  per-shard bytes currently on disk.
    - ``usage``    : int64  scalar, total bytes used across all LOCAL shards.

    Creation vs. attachment is chosen by the caller via ``shm_names``:

    - ``shm_names=None`` allocates fresh SHM and records the generated
      names; the caller can read them via :pyattr:`shm_names` and persist
      them (e.g. into ``session.json``).
    - Passing a dict attaches to existing SHM created by a peer process.
    """

    def __init__(
        self,
        *,
        capacity: int,
        shm_names: Mapping[str, str] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Shared cache capacity must be positive")
        self._capacity = int(capacity)
        self._states_mem: shared_memory.SharedMemory | None = None
        self._access_mem: shared_memory.SharedMemory | None = None
        self._sizes_mem: shared_memory.SharedMemory | None = None
        self._usage_mem: shared_memory.SharedMemory | None = None
        self._states_view: np.ndarray | None = None
        self._access_view: np.ndarray | None = None
        self._sizes_view: np.ndarray | None = None
        self._usage_view: np.ndarray | None = None
        # Creator vs joiner is tracked purely informationally; it does NOT
        # control unlinking. Unlinking happens in the manager layer when
        # the last session owner departs.
        self._created_regions = False
        self._closed = False
        self._close_finalizer: weakref.finalize | None = None

        if shm_names is None:
            self._names = self._generate_shm_names()
            self._open(create=True)
        else:
            self._names = {key: str(shm_names[key]) for key in _SHM_KEYS}
            self._open(create=False)

    # ------------------------------------------------------------------
    # SHM name generation / lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_shm_names() -> dict[str, str]:
        make_filename = getattr(shared_memory, "_make_filename", None)
        if make_filename is None:
            raise RuntimeError(
                "multiprocessing.shared_memory lacks _make_filename helper"
            )
        used: set[str] = set()
        names: dict[str, str] = {}
        for key in _SHM_KEYS:
            while True:
                candidate = str(make_filename())
                if candidate not in used:
                    names[key] = candidate
                    used.add(candidate)
                    break
        return names

    def _open(self, *, create: bool) -> None:
        state_size = self._capacity * np.uint8().nbytes
        access_size = self._capacity * np.uint64().nbytes
        sizes_size = self._capacity * np.int64().nbytes
        usage_size = np.int64().nbytes

        created_regions = False
        if create:
            try:
                self._states_mem = shared_memory.SharedMemory(
                    name=self._names["states"], create=True, size=state_size
                )
                self._access_mem = shared_memory.SharedMemory(
                    name=self._names["access"], create=True, size=access_size
                )
                self._sizes_mem = shared_memory.SharedMemory(
                    name=self._names["sizes"], create=True, size=sizes_size
                )
                self._usage_mem = shared_memory.SharedMemory(
                    name=self._names["usage"], create=True, size=usage_size
                )
                created_regions = True
            except FileExistsError:
                # Name collision: close whatever we opened and attach instead.
                self._close_partial()
                create = False

        if not create:
            self._states_mem = shared_memory.SharedMemory(
                name=self._names["states"], create=False
            )
            self._access_mem = shared_memory.SharedMemory(
                name=self._names["access"], create=False
            )
            self._sizes_mem = shared_memory.SharedMemory(
                name=self._names["sizes"], create=False
            )
            self._usage_mem = shared_memory.SharedMemory(
                name=self._names["usage"], create=False
            )

        assert self._states_mem is not None
        assert self._access_mem is not None
        assert self._sizes_mem is not None
        assert self._usage_mem is not None
        self._states_view = np.ndarray(
            (self._capacity,), dtype=np.uint8, buffer=self._states_mem.buf
        )
        self._access_view = np.ndarray(
            (self._capacity,), dtype=np.uint64, buffer=self._access_mem.buf
        )
        self._sizes_view = np.ndarray(
            (self._capacity,), dtype=np.int64, buffer=self._sizes_mem.buf
        )
        self._usage_view = np.ndarray((1,), dtype=np.int64, buffer=self._usage_mem.buf)

        if create:
            self._states_view[:] = _ShardState.INVALID
            self._access_view[:] = 0
            self._sizes_view[:] = 0
            self._usage_view[:] = 0

        self._created_regions = created_regions
        self._closed = False
        # Finalizer closes local handles only. Unlinking is the manager's
        # job when the session loses its last owner — see module docstring.
        self._close_finalizer = weakref.finalize(
            self,
            _cleanup_shared_memory_regions,
            [self._states_mem, self._access_mem, self._sizes_mem, self._usage_mem],
        )

    def _close_partial(self) -> None:
        """Close any handles already opened during a failed ``_open``."""
        for attr in ("_states_mem", "_access_mem", "_sizes_mem", "_usage_mem"):
            mem = getattr(self, attr, None)
            if mem is None:
                continue
            with contextlib.suppress(Exception):
                mem.close()
            setattr(self, attr, None)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close_finalizer is not None:
            self._close_finalizer()
        self._states_mem = None
        self._access_mem = None
        self._sizes_mem = None
        self._usage_mem = None
        self._states_view = None
        self._access_view = None
        self._sizes_view = None
        self._usage_view = None
        self._created_regions = False

    @staticmethod
    def unlink_by_names(shm_names: Mapping[str, str]) -> None:
        """Best-effort unlink of SHM segments recorded in *shm_names*.

        Called on stale-session / fingerprint-mismatch reset paths to
        guarantee the previous session's SHM segments are released before
        the new session creates fresh ones. Missing segments are silently
        ignored — this is safe to call on any session-dict payload.
        """
        if not isinstance(shm_names, Mapping):
            return
        for key in _SHM_KEYS:
            name = shm_names.get(key)
            if not isinstance(name, str) or not name:
                continue
            try:
                shm = shared_memory.SharedMemory(name=name, create=False)
            except FileNotFoundError:
                continue
            except Exception:
                continue
            with contextlib.suppress(Exception):
                shm.unlink()
            with contextlib.suppress(Exception):
                shm.close()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def shm_names(self) -> dict[str, str]:
        return dict(self._names)

    @property
    def created_regions(self) -> bool:
        """Whether this instance originally created the SHM segments.

        Informational only. Unlinking is gated on session ownership, not
        on creator identity — see the module docstring and the manager's
        last-owner unlink path in ``_release_session_owner``.
        """
        return self._created_regions

    @property
    def shard_states(self) -> np.ndarray:
        view = self._states_view
        if view is None:
            raise RuntimeError("Shared state not initialized")
        return view

    @property
    def shard_access_ns(self) -> np.ndarray:
        view = self._access_view
        if view is None:
            raise RuntimeError("Shared state not initialized")
        return view

    @property
    def shard_sizes(self) -> np.ndarray:
        view = self._sizes_view
        if view is None:
            raise RuntimeError("Shared state not initialized")
        return view

    def get_cache_usage(self) -> int:
        if self._usage_view is None:
            raise RuntimeError("Shared state not initialized")
        return int(self._usage_view[0])

    def set_cache_usage(self, value: int) -> None:
        if self._usage_view is None:
            raise RuntimeError("Shared state not initialized")
        self._usage_view[0] = int(value)

    def add_cache_usage(self, delta: int) -> None:
        if self._usage_view is None:
            raise RuntimeError("Shared state not initialized")
        self._usage_view[0] = int(self._usage_view[0] + int(delta))

    def count_local(self) -> int:
        return int(np.count_nonzero(self.shard_states == _ShardState.LOCAL))

    def set_access_time(self, index: int, timestamp_ns: int) -> None:
        self.shard_access_ns[index] = np.uint64(timestamp_ns)


__all__ = [
    "CacheSharedState",
    "_ShardState",
]
