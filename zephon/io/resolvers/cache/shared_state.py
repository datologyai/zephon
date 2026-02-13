"""Shared-memory bookkeeping for the cache-enabled resolver."""

import contextlib
import json
import weakref
from dataclasses import dataclass
from enum import IntEnum
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional

import numpy as np
from filelock import FileLock

from zephon.io.types import ShardLocator

_CACHE_STATE_DIR = ".zephon_cache_state"
_CACHE_META_FILENAME = "meta.json"
_CACHE_META_LOCK_FILENAME = "meta.lock"
_SHM_KEYS = ("states", "access", "sizes", "usage")


def _close_cache_shared_state(
    state_ref: weakref.ReferenceType["CacheSharedState"],
) -> None:
    state = state_ref()
    if state is None:
        return
    state.close()


class _ShardState(IntEnum):
    """Per-shard cache residency state shared across processes."""

    INVALID = 0
    REMOTE = 1
    PREPARING = 2
    LOCAL = 3


@dataclass(frozen=True)
class CacheEntry:
    """Metadata describing a shard tracked in shared cache state."""

    dataset: str
    shard_id: int
    raw: str
    zip: str | None
    index: int


class CacheSharedState:
    """Process-shared bookkeeping for shard residency and usage metrics."""

    def __init__(self, root: Path, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Shared cache capacity must be positive")

        self._root = root
        self._capacity = int(capacity)
        self._state_dir = self._root / _CACHE_STATE_DIR
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._state_dir / _CACHE_META_FILENAME
        self._meta_lock_path = self._state_dir / _CACHE_META_LOCK_FILENAME
        self._meta_lock = FileLock(str(self._meta_lock_path))
        self._names: dict[str, str] = {}
        self._entries: dict[tuple[str, int], CacheEntry] = {}
        self._entries_by_index: dict[int, CacheEntry] = {}
        self._next_index = 0
        self._states_mem: shared_memory.SharedMemory | None = None
        self._access_mem: shared_memory.SharedMemory | None = None
        self._sizes_mem: shared_memory.SharedMemory | None = None
        self._usage_mem: shared_memory.SharedMemory | None = None
        self._states_view: np.ndarray | None = None
        self._access_view: np.ndarray | None = None
        self._sizes_view: np.ndarray | None = None
        self._usage_view: np.ndarray | None = None
        self._owns_regions = False
        self._closed = False
        self._close_finalizer: weakref.finalize | None = None

        with self._meta_lock:
            if self._meta_path.exists():
                meta = self._load_meta()
                stored_capacity = int(meta.get("capacity", self._capacity))
                if self._capacity > stored_capacity:
                    # Capacity grew beyond what was stored (e.g., dataset changed
                    # between runs with persist_state=True). Recreate shared
                    # memory from scratch at the new size. All workers
                    # synchronize through this FileLock so subsequent processes
                    # will see the updated meta.json and attach to the new
                    # correctly-sized regions.
                    self._unlink_old_regions_locked(meta)
                    self._names = self._generate_shm_names()
                    self._attach_shared(create=True)
                    meta = {
                        "capacity": self._capacity,
                        "next_index": 0,
                        "mapping": {},
                        "names": dict(self._names),
                    }
                    self._write_meta(meta)
                else:
                    self._capacity = stored_capacity
                    self._sync_next_index_locked(meta)
                    self._entries = {}
                    self._entries_by_index = {}
                    meta = self._ensure_names_locked(meta)
                    self._reload_meta_locked(meta)
                    try:
                        self._attach_shared(create=False)
                    except FileNotFoundError:
                        self._attach_shared(create=True)
            else:
                self._next_index = 0
                self._entries = {}
                self._entries_by_index = {}
                self._names = self._generate_shm_names()
                self._attach_shared(create=True)
                meta = {
                    "capacity": self._capacity,
                    "next_index": self._next_index,
                    "mapping": {},
                    "names": dict(self._names),
                }
                self._write_meta(meta)

    # ------------------------------------------------------------------
    # Shared memory accessors
    # ------------------------------------------------------------------

    def _unlink_old_regions_locked(self, meta: dict) -> None:
        """Unlink shared memory regions referenced by *meta*."""
        old_names = meta.get("names", {})
        if not isinstance(old_names, dict):
            return
        for key in _SHM_KEYS:
            name = old_names.get(key)
            if not isinstance(name, str):
                continue
            try:
                shm = shared_memory.SharedMemory(name=name, create=False)
                shm.unlink()
                shm.close()
            except FileNotFoundError:
                pass

    def _generate_shm_names(self) -> dict[str, str]:
        make_filename = getattr(shared_memory, "_make_filename", None)
        if make_filename is None:
            raise RuntimeError(
                "multiprocessing.shared_memory lacks _make_filename helper"
            )
        names: dict[str, str] = {}
        used: set[str] = set()
        for key in _SHM_KEYS:
            while True:
                candidate = str(make_filename())
                if candidate not in used:
                    names[key] = candidate
                    used.add(candidate)
                    break
        return names

    def _ensure_names_locked(self, meta: dict) -> dict:
        raw_names = meta.get("names")
        if isinstance(raw_names, dict):
            names: dict[str, str] = {
                str(key): str(value)
                for key, value in raw_names.items()
                if isinstance(key, str) and isinstance(value, str)
            }
        else:
            names = {}
        make_filename = getattr(shared_memory, "_make_filename", None)
        if make_filename is None:
            raise RuntimeError(
                "multiprocessing.shared_memory lacks _make_filename helper"
            )
        changed = False
        used: set[str] = set(names.values())
        for key in _SHM_KEYS:
            name = names.get(key)
            if isinstance(name, str):
                max_len = getattr(shared_memory, "_SHM_SAFE_NAME_LENGTH", None)
                if max_len is not None and len(name) > max_len:
                    name = None
            else:
                name = None
            valid = name is not None
            if not valid:
                while True:
                    candidate = str(make_filename())
                    if candidate not in used:
                        names[key] = candidate
                        used.add(candidate)
                        changed = True
                        break
        if len(set(names.values())) != len(names.values()):
            names = self._generate_shm_names()
            changed = True
        meta["names"] = names
        self._names = dict(names)
        if changed:
            self._write_meta(meta)
        return meta

    def _attach_shared(self, *, create: bool) -> None:
        state_size = self._capacity * np.uint8().nbytes
        access_size = self._capacity * np.uint64().nbytes
        sizes_size = self._capacity * np.int64().nbytes
        usage_size = np.int64().nbytes

        states_name = self._names["states"]
        access_name = self._names["access"]
        sizes_name = self._names["sizes"]
        usage_name = self._names["usage"]

        owns_regions = False
        try:
            if create:
                try:
                    self._states_mem = shared_memory.SharedMemory(
                        name=states_name, create=True, size=state_size
                    )
                    self._access_mem = shared_memory.SharedMemory(
                        name=access_name, create=True, size=access_size
                    )
                    self._sizes_mem = shared_memory.SharedMemory(
                        name=sizes_name, create=True, size=sizes_size
                    )
                    self._usage_mem = shared_memory.SharedMemory(
                        name=usage_name, create=True, size=usage_size
                    )
                    owns_regions = True
                except FileExistsError:
                    create = False
            if not create:
                self._states_mem = shared_memory.SharedMemory(
                    name=states_name, create=False
                )
                self._access_mem = shared_memory.SharedMemory(
                    name=access_name, create=False
                )
                self._sizes_mem = shared_memory.SharedMemory(
                    name=sizes_name, create=False
                )
                self._usage_mem = shared_memory.SharedMemory(
                    name=usage_name, create=False
                )
        except FileNotFoundError:
            if create:
                raise
            self._attach_shared(create=True)
            return

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

        self._owns_regions = owns_regions
        self._closed = False
        # Avoid atexit strong refs so GC can reclaim shared state instances.
        self._close_finalizer = weakref.finalize(
            self, _close_cache_shared_state, weakref.ref(self)
        )

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict:
        with self._meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_meta(self, meta: dict) -> None:
        tmp_path = self._meta_path.with_suffix(".tmp")
        payload = json.dumps(meta, sort_keys=True)
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self._meta_path)

    def _reload_meta_locked(self, meta: dict) -> None:
        names = meta.get("names")
        if isinstance(names, dict) and all(key in names for key in _SHM_KEYS):
            self._names = dict(names)
        mapping = meta.get("mapping")
        if not isinstance(mapping, dict):
            return
        datasets: dict[tuple[str, int], CacheEntry] = {}
        by_index: dict[int, CacheEntry] = {}
        for dataset, shards in mapping.items():
            if not isinstance(shards, dict):
                continue
            for shard_id, payload in shards.items():
                if not isinstance(payload, dict):
                    continue
                try:
                    index = int(payload["index"])
                except Exception:
                    continue
                entry = CacheEntry(
                    dataset=str(dataset),
                    shard_id=int(shard_id),
                    raw=str(payload.get("raw", "")),
                    zip=payload.get("zip"),
                    index=index,
                )
                datasets[(entry.dataset, entry.shard_id)] = entry
                by_index[index] = entry
        self._entries = datasets
        self._entries_by_index = by_index
        self._sync_next_index_locked(meta)

    def _sync_next_index_locked(self, meta: dict) -> None:
        # ``next_index`` is persisted so independent processes can allocate
        # unique indices without coordinating through additional shared-memory
        # primitives. When we see a newer on-disk value we simply adopt it. This
        # piggybacks on metadata I/O that is already required during cache
        # warm-up and therefore does not add extra filesystem cost in the steady
        # state where lookups hit the in-memory maps.
        raw_next = meta.get("next_index")
        if raw_next is None:
            return

        try:
            next_index = int(raw_next)
        except Exception:
            return

        if next_index < 0:
            next_index = 0
        if next_index > self._next_index:
            self._next_index = next_index

    # ------------------------------------------------------------------
    # Public API used by CacheManager
    # ------------------------------------------------------------------

    def ensure_entry(self, locator: ShardLocator) -> CacheEntry:
        key = (locator.dataset, int(locator.shard_id))
        entry = self._entries.get(key)
        if entry is not None:
            self._maybe_update_entry_metadata(entry, locator)
            return entry

        with self._meta_lock:
            # NOTE: Disk I/O via ``_load_meta`` only occurs while registering a
            # shard that is not already known locally. Once an entry has been
            # created, repeated lookups are served from ``_entries`` without
            # touching the filesystem, so steady-state cache hits remain fast.
            meta = self._load_meta()
            self._sync_next_index_locked(meta)
            mapping = meta.setdefault("mapping", {})
            dataset_map = mapping.setdefault(locator.dataset, {})
            shard_key = str(int(locator.shard_id))
            payload = dataset_map.get(shard_key)
            if payload is None:
                index = self._next_index
                if index >= self._capacity:
                    raise IndexError(
                        f"Cache shard index {index} exceeds capacity "
                        f"{self._capacity}. The dataset has more unique "
                        f"shards than the cache can track."
                    )
                self._next_index += 1
                payload = {
                    "index": index,
                    "raw": locator.raw.basename,
                    "zip": locator.zip.basename if locator.zip else None,
                }
                dataset_map[shard_key] = payload
                meta["next_index"] = self._next_index
                self._write_meta(meta)
                self._reload_meta_locked(meta)
            else:
                self._reload_meta_locked(meta)
            entry = self._entries.get(key)
        if entry is None:
            raise RuntimeError("Failed to register shard in shared cache state")
        self._maybe_update_entry_metadata(entry, locator)
        return entry

    def _maybe_update_entry_metadata(
        self, entry: CacheEntry, locator: ShardLocator
    ) -> None:
        raw_name = locator.raw.basename
        zip_name = locator.zip.basename if locator.zip is not None else None
        if entry.raw == raw_name and entry.zip == zip_name:
            return
        with self._meta_lock:
            meta = self._load_meta()
            mapping = meta.setdefault("mapping", {})
            dataset_map = mapping.setdefault(locator.dataset, {})
            payload = dataset_map.setdefault(
                str(int(locator.shard_id)), {"index": entry.index}
            )
            changed = False
            if payload.get("raw") != raw_name:
                payload["raw"] = raw_name
                changed = True
            if payload.get("zip") != zip_name:
                payload["zip"] = zip_name
                changed = True
            if changed:
                self._write_meta(meta)
                self._reload_meta_locked(meta)

    def lookup(self, dataset: str, shard_id: int) -> Optional[CacheEntry]:
        key = (dataset, int(shard_id))
        entry = self._entries.get(key)
        if entry is not None:
            return entry
        with self._meta_lock:
            meta = self._load_meta()
            self._reload_meta_locked(meta)
        return self._entries.get(key)

    def entry_by_index(self, index: int) -> Optional[CacheEntry]:
        entry = self._entries_by_index.get(index)
        if entry is not None:
            return entry
        with self._meta_lock:
            meta = self._load_meta()
            self._reload_meta_locked(meta)
        return self._entries_by_index.get(index)

    # ------------------------------------------------------------------
    # Numeric state helpers (callers must coordinate external locking)
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

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

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        shms = [self._states_mem, self._access_mem, self._sizes_mem, self._usage_mem]
        if self._owns_regions:
            for shm in shms:
                if shm is None:
                    continue
                with contextlib.suppress(Exception):
                    shm.unlink()
        for shm in shms:
            if shm is None:
                continue
            with contextlib.suppress(Exception):
                shm.close()
        self._states_mem = None
        self._access_mem = None
        self._sizes_mem = None
        self._usage_mem = None
        self._states_view = None
        self._access_view = None
        self._sizes_view = None
        self._usage_view = None
        self._owns_regions = False


__all__ = [
    "CacheEntry",
    "CacheSharedState",
    "_ShardState",
]
