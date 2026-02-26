"""Cache-backed implementation of the shard resolver protocol."""

import bz2
import contextlib
import errno
import gzip
import io
import json
import logging
import lzma
import os
import shutil
import time
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Optional, cast

import numpy as np
from filelock import BaseFileLock, FileLock

from zephon.io.resolvers.base import ShardResolver
from zephon.io.resolvers.cache.errors import PermanentSourceMissing, ShardNotReady
from zephon.io.resolvers.cache.shared_state import CacheSharedState, _ShardState
from zephon.io.resolvers.utils import compute_file_hash
from zephon.io.storage import StorageBackend
from zephon.io.types import LocalShardFile, LocalShardRef, ShardFile, ShardLocator

try:  # LZ4 is optional
    import lz4.frame as lz4frame
except Exception:
    lz4frame = None

try:  # zstd is optional
    import zstd as zstd_mod
except Exception:
    zstd_mod = None

logger = logging.getLogger(__name__)

_CACHE_LOCK_FILENAME = ".cache.lock"
_TICK_SECONDS = float(os.environ.get("ZEPHON_CACHE_TICK", "0.05"))
Opener = Callable[[Path], BinaryIO]


def _close_cache_manager_resources(
    persist_state: bool,
    shared: "CacheSharedState",
    reset_lock: "BaseFileLock",
    session_path: Path,
    pid: int,
) -> None:
    """Release all resources owned by a ``CacheManager``.

    Designed for use as a ``weakref.finalize`` callback: receives the
    resources directly so cleanup succeeds even when the
    ``CacheManager`` is already being garbage-collected.
    """
    if not persist_state:
        with contextlib.suppress(Exception):
            _release_session_owner(reset_lock, session_path, pid)
    shared.close()


def _release_session_owner(
    reset_lock: "BaseFileLock",
    session_path: Path,
    pid: int,
) -> None:
    """Remove *pid* from the session owner file.

    Module-level helper so it can be called from a ``weakref.finalize``
    callback (where ``self`` is already dead).
    """
    with reset_lock:
        if not session_path.exists():
            return
        try:
            session = json.loads(session_path.read_text(encoding="utf-8"))
        except Exception:
            return
        owners = session.get("owners", {})
        owner_key = str(pid)
        entry = owners.get(owner_key)
        if isinstance(entry, dict):
            instances = int(entry.get("instances", 1))
            if instances > 1:
                entry["instances"] = instances - 1
                owners[owner_key] = entry
                session["owners"] = owners
                tmp = session_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(session, sort_keys=True), encoding="utf-8")
                tmp.replace(session_path)
                return
        owners.pop(owner_key, None)
        if owners:
            session["owners"] = owners
            tmp = session_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(session, sort_keys=True), encoding="utf-8")
            tmp.replace(session_path)
        else:
            session_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class CacheStats:
    """Snapshot of cache usage metrics."""

    bytes_used: int
    shards: int


class CacheManager(ShardResolver):
    """Resolve shards locally with eviction-aware coordination."""

    def __init__(
        self,
        root: Path,
        storage: StorageBackend,
        *,
        num_shards: int,
        limit_bytes: int | None = None,
        keep_zip: bool = False,
        validate_hash: str | None = None,
        download_retry: int = 2,
        download_timeout: float = 60.0,
        persist_state: bool = False,
        min_slack_bytes: int = 512 * 1024,
        max_slack_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self._root = root
        self._storage = storage
        self._limit_bytes = limit_bytes
        self._keep_zip = keep_zip
        self._validate_hash = validate_hash
        self._download_retry = max(1, int(download_retry))
        self._download_timeout = download_timeout
        self._persist_state = bool(persist_state)
        self._min_slack_bytes = int(min_slack_bytes)
        self._max_slack_bytes = int(max_slack_bytes)

        self._root.mkdir(parents=True, exist_ok=True)
        reset_lock_path = self._root / ".reset.lock"
        self._reset_lock_path = reset_lock_path
        self._reset_lock = reset_lock = FileLock(str(reset_lock_path))
        self._session_path = self._root / ".cache.session.json"
        self._pid = os.getpid()
        if not self._persist_state:
            with reset_lock:
                self._reset_if_first_owner(reset_lock_path)
        self._shared = CacheSharedState(self._root, capacity=num_shards)
        logger.debug(
            "Initialized cache at %s (capacity=%d shards, limit=%s)",
            self._root,
            self._shared.capacity,
            f"{self._limit_bytes:,} bytes"
            if self._limit_bytes is not None
            else "unlimited",
        )
        self._cache_lock = FileLock(str(self._root / _CACHE_LOCK_FILENAME))
        # Ensure all resources are released even if close() is never called.
        # Pass the resources directly so the callback works after the manager
        # has been collected (weakref.ref(self) would be dead).
        self._close_finalizer = weakref.finalize(
            self,
            _close_cache_manager_resources,
            self._persist_state,
            self._shared,
            self._reset_lock,
            self._session_path,
            self._pid,
        )

    def stats(self) -> CacheStats:
        with self._cache_lock:
            return CacheStats(
                bytes_used=self._shared.get_cache_usage(),
                shards=self._shared.count_local(),
            )

    def close(self) -> None:
        # Delegate to the finalizer which handles session owner release +
        # shared state cleanup.  Calling it is idempotent — a second call
        # (or GC triggering it later) is a harmless no-op.
        self._close_finalizer()

    def touch(self, locator: ShardLocator) -> None:
        """Record a shard access without taking the global lock."""
        entry = self._shared.lookup(locator.dataset, int(locator.shard_id))
        if entry is None:
            return
        if _ShardState(self._shared.shard_states[entry.index]) == _ShardState.LOCAL:
            self._shared.set_access_time(entry.index, time.time_ns())

    def resolve(self, locator: ShardLocator, *, blocking: bool = True) -> LocalShardRef:
        """Return a local reference, downloading and evicting as required."""
        entry = self._shared.ensure_entry(locator)
        dataset_root = self._root / entry.dataset
        dataset_root.mkdir(parents=True, exist_ok=True)

        raw_path = dataset_root / entry.raw
        zip_path = dataset_root / entry.zip if entry.zip else None
        required = self._required_bytes(locator)

        shard_lock = self._shard_lock(dataset_root, locator.shard_id)

        while True:
            wait_only = False
            with self._cache_lock:
                state_value = _ShardState(self._shared.shard_states[entry.index])
                if state_value == _ShardState.LOCAL:
                    if raw_path.is_file():
                        self._shared.set_access_time(entry.index, time.time_ns())
                        return self._build_ref(
                            raw_path, zip_path, locator, cache_hit=True
                        )
                    self._mark_remote_locked(entry.index)
                    continue

                if state_value in (_ShardState.INVALID, _ShardState.REMOTE):
                    current_size = int(self._shared.shard_sizes[entry.index])
                    additional = max(0, required - current_size)
                    if self._limit_bytes is not None:
                        self._ensure_capacity_locked(additional, skip_index=entry.index)
                    self._shared.shard_states[entry.index] = _ShardState.PREPARING
                    self._shared.set_access_time(entry.index, time.time_ns())
                    break

                if state_value == _ShardState.PREPARING:
                    wait_only = True

            if not wait_only:
                continue
            if not blocking:
                raise ShardNotReady(locator.dataset, int(locator.shard_id))
            time.sleep(_TICK_SECONDS)

        with shard_lock:
            part_path = self._part_path(raw_path)
            part_path.parent.mkdir(parents=True, exist_ok=True)
            part_path.write_text(f"PREPARING {time.time()}\n", encoding="utf-8")
            success = False
            try:
                self._prepare(locator, raw_path, zip_path)
                self._validate(raw_path, locator.raw)
                success = True
            finally:
                with contextlib.suppress(Exception):
                    part_path.unlink()
                if not success:
                    with self._cache_lock:
                        self._mark_remote_locked(entry.index)

        entry_size = raw_path.stat().st_size
        actual_zip: Optional[Path] = None
        if zip_path and zip_path.exists():
            if self._keep_zip:
                entry_size += zip_path.stat().st_size
                actual_zip = zip_path
            else:
                zip_path.unlink(missing_ok=True)
        with self._cache_lock:
            old_size = int(self._shared.shard_sizes[entry.index])
            delta = entry_size - old_size
            if delta:
                self._shared.add_cache_usage(delta)
            self._shared.shard_sizes[entry.index] = entry_size
            self._shared.shard_states[entry.index] = _ShardState.LOCAL
            self._shared.set_access_time(entry.index, time.time_ns())

        return self._build_ref(raw_path, actual_zip, locator, cache_hit=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_remote_locked(self, index: int) -> None:
        size = int(self._shared.shard_sizes[index])
        if size:
            self._shared.add_cache_usage(-size)
        self._shared.shard_sizes[index] = 0
        self._shared.shard_states[index] = _ShardState.REMOTE
        self._shared.shard_access_ns[index] = 0

    def _ensure_capacity_locked(self, additional: int, *, skip_index: int) -> None:
        if self._limit_bytes is None or additional <= 0:
            return
        if self._limit_bytes < additional:
            raise ValueError(
                "Cache limit smaller than shard; increase cache_limit to proceed"
            )
        while self._shared.get_cache_usage() + additional > self._limit_bytes:
            victim = self._select_coldest_index_locked(skip_index)
            if victim is None:
                raise RuntimeError("Cache limit reached but no shards to evict")
            self._evict_index_locked(victim)

    def _select_coldest_index_locked(self, skip_index: int) -> Optional[int]:
        states = self._shared.shard_states
        access_times = self._shared.shard_access_ns
        mask = states == _ShardState.LOCAL
        if skip_index >= 0:
            mask[skip_index] = False
        if not mask.any():
            return None
        # Set non-LOCAL slots to max so argmin ignores them.
        candidates = np.where(mask, access_times, np.iinfo(np.uint64).max)
        return int(np.argmin(candidates))

    def _evict_index_locked(self, index: int) -> None:
        entry = self._shared.entry_by_index(index)
        if entry is not None:
            dataset_root = self._root / entry.dataset
            raw_path = dataset_root / entry.raw
            zip_path = dataset_root / entry.zip if entry.zip else None
            raw_path.unlink(missing_ok=True)
            if zip_path is not None:
                zip_path.unlink(missing_ok=True)
        self._mark_remote_locked(index)

    def _prepare(
        self,
        locator: ShardLocator,
        raw_path: Path,
        zip_path: Optional[Path],
    ) -> None:
        raw_tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
        raw_tmp.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)

        if locator.zip and locator.compression:
            if zip_path is None:
                raise RuntimeError("Zip metadata missing while compression provided")
            zip_tmp = zip_path.with_suffix(zip_path.suffix + ".tmp")
            zip_tmp.parent.mkdir(parents=True, exist_ok=True)
            self._download_file(locator.root, locator.zip, zip_tmp)
            try:
                self._decompress_stream(zip_tmp, raw_tmp, locator.compression)
            finally:
                if self._keep_zip:
                    with contextlib.suppress(Exception):
                        zip_tmp.replace(zip_path)
                else:
                    zip_tmp.unlink(missing_ok=True)
            raw_tmp.replace(raw_path)
        else:
            raw_tmp.parent.mkdir(parents=True, exist_ok=True)
            self._download_file(locator.root, locator.raw, raw_tmp)
            raw_tmp.replace(raw_path)

    def _download_file(
        self, root: str, shard_file: ShardFile, target_tmp: Path
    ) -> None:
        src = os.path.join(root, shard_file.basename)
        target_tmp.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(self._download_retry):
            try:
                self._assert_source_exists(src)
                self._storage.download(
                    src, str(target_tmp), timeout=self._download_timeout
                )
                return
            except PermanentSourceMissing:
                target_tmp.unlink(missing_ok=True)
                raise
            except Exception:
                target_tmp.unlink(missing_ok=True)
                if attempt + 1 == self._download_retry:
                    raise
                time.sleep(min(1.0 * (attempt + 1), 5.0))

    def _decompress_stream(self, src: Path, dst_tmp: Path, compression: str) -> None:
        algo = (compression or "").lower()
        dst_tmp.parent.mkdir(parents=True, exist_ok=True)
        opener: Opener
        if algo in {"gz", "gzip"}:

            def _open_gzip(p: Path) -> BinaryIO:
                return cast(BinaryIO, gzip.open(p, "rb"))

            opener = _open_gzip
        elif algo in {"bz2", "bzip2"}:

            def _open_bz2(p: Path) -> BinaryIO:
                return cast(BinaryIO, bz2.open(p, "rb"))

            opener = _open_bz2
        elif algo in {"lzma", "xz"}:

            def _open_lzma(p: Path) -> BinaryIO:
                return cast(BinaryIO, lzma.open(p, "rb"))

            opener = _open_lzma
        elif algo in {"zst", "zstd", "zstandard"}:
            if zstd_mod is None:
                raise RuntimeError("zstd compression requires the 'zstd' package")

            def _open_zstd(p: Path) -> BinaryIO:
                compressed = p.read_bytes()
                decompressed = zstd_mod.decompress(compressed)
                return cast(BinaryIO, io.BytesIO(decompressed))

            opener = _open_zstd
        elif algo in {"lz4"}:
            lz4_mod = lz4frame
            if lz4_mod is None:
                raise RuntimeError("lz4 compression requires the 'lz4' package")

            def _open_lz4(p: Path) -> BinaryIO:
                return cast(BinaryIO, lz4_mod.open(p, mode="rb"))

            opener = _open_lz4
        else:
            raise ValueError(f"Unsupported compression: {compression}")

        with opener(src) as in_f, dst_tmp.open("wb") as out_f:
            while True:
                chunk = in_f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)

    def _validate(self, raw_path: Path, raw_meta: ShardFile) -> None:
        if not self._validate_hash:
            return
        expected = raw_meta.hashes.get(self._validate_hash)
        if not expected:
            return
        actual = compute_file_hash(raw_path, self._validate_hash)
        if actual != expected:
            raw_path.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {raw_path.name}: {actual} != {expected}"
            )

    def _build_ref(
        self,
        raw_path: Path,
        zip_path: Optional[Path],
        locator: ShardLocator,
        cache_hit: bool,
    ) -> LocalShardRef:
        raw_file = LocalShardFile(path=raw_path, bytes=raw_path.stat().st_size)
        zip_file = None
        if zip_path and zip_path.exists():
            zip_file = LocalShardFile(path=zip_path, bytes=zip_path.stat().st_size)
        return LocalShardRef(
            raw=raw_file,
            zip=zip_file,
            compression=locator.compression,
            extra=locator.extra,
            cache_hit=cache_hit,
        )

    def _shard_lock(self, dataset_root: Path, shard_id: int) -> BaseFileLock:
        locks_dir = dataset_root / ".locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        return FileLock(str(locks_dir / f"{shard_id}.lock"))

    def _part_path(self, raw_path: Path) -> Path:
        return raw_path.with_suffix(raw_path.suffix + ".part")

    def _required_bytes(self, locator: ShardLocator) -> int:
        required = int(locator.raw.bytes)
        if locator.zip is not None:
            required += int(locator.zip.bytes)
        slack = max(self._min_slack_bytes, int(required * 0.005))  # min slack or 0.5%
        slack = min(self._max_slack_bytes, slack)  # cap slack at max
        return required + slack

    def _assert_source_exists(self, src: str) -> None:
        if self._storage.exists(src):
            return
        raise PermanentSourceMissing(f"Source missing: {src}")

    def _reset_cache_root(self, lock_path: Path) -> None:
        for child in list(self._root.iterdir()):
            if child == lock_path:
                continue
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            else:
                shutil.rmtree(child, ignore_errors=True)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Session coordination helpers
    # ------------------------------------------------------------------

    def _reset_if_first_owner(self, lock_path: Path) -> None:
        """Ensure only the first live owner resets the cache.

        Because ``persist_state`` is ``False``, managers are expected to wipe the
        cache directory when starting fresh. In a multi-process setup that shares
        the same cache root, blindly resetting would cause one process to delete
        data that another is actively preparing. To avoid data loss while still
        guaranteeing a clean slate when *nobody* is using the cache, we maintain
        a small session descriptor file recording which process IDs currently own
        the cache. Initialization proceeds as follows:

        1. Read the descriptor (if present) and drop owners whose processes are no
           longer alive. Crashes and unclean exits are therefore handled when the
           *next* manager starts rather than requiring explicit shutdown hooks.
        2. If no live owners remain, reset the cache directory and start a brand
           new session. Otherwise we simply join the existing session.
        3. Register the current PID as an owner (tracking how many managers this
           PID currently has open) and persist the descriptor.

        This logic ensures that the cache is wiped exactly once at the beginning
        of a session and that concurrent managers never stomp on each other's
        data. Shutdown is best-effort: :meth:`close` removes the PID from the
        descriptor, decrementing the per-PID reference count. Even if shutdown
        fails, the next initializer will notice the stale PID and clean things
        up before deciding whether to reset.
        """
        session = self._load_session()
        owners = self._prune_dead_owners(session.get("owners", {}))
        if not owners:
            self._reset_cache_root(lock_path)
            session = {
                "session_id": str(uuid.uuid4()),
                "session_started_ns": time.time_ns(),
                "owners": {},
            }
        else:
            session.setdefault("session_id", str(uuid.uuid4()))
            session.setdefault("session_started_ns", time.time_ns())
            session["owners"] = owners

        owners = session["owners"]
        owner_key = str(self._pid)
        owner_entry = owners.get(owner_key)
        if not isinstance(owner_entry, dict):
            owner_entry = {}
        owner_entry.setdefault("started_ns", time.time_ns())
        instances = int(owner_entry.get("instances", 0)) + 1
        owner_entry["instances"] = instances
        owners[owner_key] = owner_entry
        session["owners"] = owners
        self._write_session(session)

    def _load_session(self) -> dict:
        if not self._session_path.exists():
            return {}
        try:
            return json.loads(self._session_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_session(self, payload: dict) -> None:
        tmp = self._session_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(self._session_path)

    def _prune_dead_owners(self, owners: dict) -> dict:
        alive: dict[str, dict] = {}
        for pid_str, meta in owners.items():
            try:
                pid = int(pid_str)
            except Exception:
                continue
            if self._pid_alive(pid):
                entry = meta if isinstance(meta, dict) else {}
                if int(entry.get("instances", 0)) <= 0:
                    entry["instances"] = 1
                entry.setdefault("started_ns", time.time_ns())
                alive[pid_str] = entry
        return alive

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError as exc:
            # ESRCH: process does not exist; EPERM: process exists but not ours
            if exc.errno == errno.ESRCH:
                return False
            if exc.errno == errno.EPERM:
                return True
            return False
        else:
            return True


__all__ = [
    "CacheManager",
    "CacheStats",
]
