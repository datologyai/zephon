"""Cache-backed implementation of the shard resolver protocol."""

import bz2
import contextlib
import errno
import gzip
import hashlib
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
from typing import BinaryIO, Callable, Mapping, Optional, cast

import numpy as np
from filelock import BaseFileLock, FileLock
from filelock import Timeout as FileLockTimeout

from zephon.io.dataset import Dataset
from zephon.io.resolvers.base import ShardResolver
from zephon.io.resolvers.cache.errors import (
    CacheInUseError,
    PermanentSourceMissing,
    ShardNotReady,
)
from zephon.io.resolvers.cache.shared_state import CacheSharedState, _ShardState
from zephon.io.resolvers.utils import compute_file_hash
from zephon.io.storage import StorageBackend
from zephon.io.types import LocalShardFile, LocalShardRef, ShardFile, ShardLocator
from zephon.utils.atomic import atomic_write_bytes

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
_STATE_DIR_NAME = ".zephon_cache_state"
_SESSION_FILENAME = "session.json"
_RESET_LOCK_FILENAME = ".reset.lock"
_TICK_SECONDS = float(os.environ.get("ZEPHON_CACHE_TICK", "0.05"))
Opener = Callable[[Path], BinaryIO]


def _parse_proc_stat_starttime(data: bytes) -> int | None:
    """Extract field 22 (``starttime``) from ``/proc/<pid>/stat`` bytes.

    ``comm`` (field 2) is rendered in parentheses and may itself contain
    spaces or parens, so we split *after* the rightmost ``)`` to land
    in the whitespace-delimited tail starting at field 3 (``state``).
    Field 22 is then index 19 in that tail. Returns ``None`` if the
    buffer doesn't conform to the expected layout, which is treated as
    "unknown" by the caller.
    """
    rparen = data.rfind(b")")
    if rparen == -1:
        return None
    fields = data[rparen + 2 :].split()
    if len(fields) < 20:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def _process_start_time_ns(pid: int) -> int | None:
    """Return *pid*'s OS-level start time in ns, or ``None`` if unknown.

    Used with ``os.kill(pid, 0)`` to distinguish a still-live owner from
    a recycled PID.  Linux reads ``/proc/<pid>/stat``; macOS/BSD falls
    back to ``psutil`` when available.
    """
    if pid <= 0:
        return None
    # Linux fast path: /proc/<pid>/stat
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
        ticks = _parse_proc_stat_starttime(data)
        if ticks is not None:
            tck = os.sysconf("SC_CLK_TCK")
            return int(ticks * 1_000_000_000 / tck)
    except (FileNotFoundError, PermissionError, OSError):
        pass
    # Non-Linux fallback.
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process(pid).create_time() * 1_000_000_000)
    except Exception:
        return None


def _make_owner_key(pid: int, start_time_ns: int | None) -> str:
    """Build the compound owner key used in ``session.json``.

    Format is ``f"{pid}-{start_time_ns}"`` so PID recycling produces a
    different key; falls back to ``str(pid)`` when start-time is
    unavailable (non-Linux without psutil).
    """
    if start_time_ns is None:
        return str(pid)
    return f"{pid}-{start_time_ns}"


def _parse_owner_key(key: str) -> tuple[int, int | None]:
    """Parse ``f"{pid}-{start_time_ns}"`` back into its components.

    Accepts the legacy ``str(pid)`` form for backwards-compatible reads
    of pre-upgrade session files; returns ``(pid, None)`` in that case
    so the caller treats missing start-time as "can't verify" rather
    than "PID recycled".
    """
    if "-" in key:
        pid_str, _, start_str = key.partition("-")
        try:
            return int(pid_str), int(start_str)
        except ValueError:
            pass
    try:
        return int(key), None
    except ValueError:
        return -1, None


def _close_cache_manager_resources(
    shared: "CacheSharedState | None",
    reset_lock: "BaseFileLock",
    session_path: Path,
    owner_key: str,
) -> None:
    """Release all resources owned by a ``CacheManager``.

    Designed for use as a ``weakref.finalize`` callback: receives the
    resources directly so cleanup succeeds even when the
    ``CacheManager`` is already being garbage-collected.
    """
    with contextlib.suppress(Exception):
        _release_session_owner(reset_lock, session_path, owner_key)
    if shared is not None:
        shared.close()


def _release_session_owner(
    reset_lock: "BaseFileLock",
    session_path: Path,
    owner_key: str,
) -> None:
    """Remove *owner_key* from the session owner list.

    Module-level helper so it can be called from a ``weakref.finalize``
    callback (where ``self`` is already dead). Owner key format is
    ``f"{pid}-{start_time_ns}"`` (or legacy ``str(pid)`` for sessions
    written by older versions).

    When removal drops the last live owner, the session's SHM segments
    are unlinked here but ``session.json`` itself is preserved (with an
    empty ``owners`` list) so the next manager init can see the
    fingerprint and route to RESUME via the ``persist_state=True``
    path. This is the ONLY place the cache normally unlinks SHM —
    ``CacheSharedState.close`` deliberately leaves segment names alive
    so peer managers can keep attaching while any owner remains.
    """
    with reset_lock:
        if not session_path.exists():
            return
        try:
            session = json.loads(session_path.read_text(encoding="utf-8"))
        except Exception:
            return
        owners = session.get("owners", {})
        entry = owners.get(owner_key)
        if isinstance(entry, dict):
            instances = int(entry.get("instances", 1))
            if instances > 1:
                entry["instances"] = instances - 1
                owners[owner_key] = entry
                session["owners"] = owners
                _atomic_write_json(session_path, session)
                return
        owners.pop(owner_key, None)
        session["owners"] = owners
        if owners:
            _atomic_write_json(session_path, session)
            return
        # Last owner: free the OS-level SHM names promptly so /dev/shm
        # doesn't carry them between runs. Deliberately KEEP session.json
        # (with empty owners) so the next manager init can see the
        # fingerprint and route to RESUME instead of FIRST_INIT, which is
        # what makes persist_state=True survive process exits. The stored
        # shm_names become stale here but unlink_by_names is idempotent
        # on missing segments, so the next init's reset paths remain safe.
        shm_names = session.get("shm_names")
        if isinstance(shm_names, dict):
            CacheSharedState.unlink_by_names(shm_names)
        _atomic_write_json(session_path, session)


def _atomic_write_json(path: Path, payload: dict) -> None:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, data, unique_tmp=False)


def _diff_session_summary(
    stored: object, current: list[dict[str, object]]
) -> list[str]:
    """Diff two per-dataset summaries for inclusion in ``CacheInUseError``.

    Each entry is ``{name, path, shard_count, total_raw_bytes}``. The
    diff bullets cover the operationally interesting cases — datasets
    added or removed, shard count deltas, byte deltas, and path moves —
    which are the cases the operator can act on without re-running the
    prior job.

    When ``stored`` isn't a list (legacy ``session.json`` written before
    this schema landed), a single ``"summary missing"`` line is returned
    so the caller can report that the structured data wasn't recorded.
    When the summary is unchanged but the fingerprint still differs, the
    mismatch is at the per-shard level (raw bytes, hashes, or zip
    metadata) — those aren't preserved in the summary, so we say so
    rather than emit an empty diff.
    """
    if not isinstance(stored, list):
        return ["summary missing — reporting hashes only."]

    stored_by_name: dict[str, dict] = {}
    for entry in stored:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            stored_by_name[entry["name"]] = entry
    current_by_name: dict[str, dict[str, object]] = {str(d["name"]): d for d in current}

    lines: list[str] = []
    added = sorted(set(current_by_name) - set(stored_by_name))
    removed = sorted(set(stored_by_name) - set(current_by_name))
    common = sorted(set(stored_by_name) & set(current_by_name))

    for name in added:
        d = current_by_name[name]
        lines.append(
            f"  + dataset added: {name!r} "
            f"(shards={d.get('shard_count')}, "
            f"raw_bytes={d.get('total_raw_bytes')})"
        )
    for name in removed:
        d = stored_by_name[name]
        lines.append(
            f"  - dataset removed: {name!r} "
            f"(shards={d.get('shard_count')}, "
            f"raw_bytes={d.get('total_raw_bytes')})"
        )
    for name in common:
        s, c = stored_by_name[name], current_by_name[name]
        for field in ("path", "shard_count", "total_raw_bytes"):
            if s.get(field) != c.get(field):
                lines.append(
                    f"  ~ dataset {name!r}: "
                    f"{field} {s.get(field)!r} -> {c.get(field)!r}"
                )
    if not lines:
        lines.append(
            "per-dataset summary unchanged — fingerprint differs at "
            "shard level (per-shard bytes, hashes, or zip metadata)."
        )
    return lines


@dataclass(frozen=True)
class CacheStats:
    """Snapshot of cache usage metrics."""

    bytes_used: int
    shards: int


class CacheManager(ShardResolver):
    """Resolve shards locally with eviction-aware coordination.

    State layout on disk under ``root``:

    - ``<dataset.name>/<raw_basename>`` — shard data files (one per shard).
    - ``.locks/<shard_id>.lock`` — per-shard download locks.
    - ``.cache.lock`` — global state-transition lock.
    - ``.reset.lock`` — session-init lock (held across wipe + SHM create +
      reconciliation + session.json write).
    - ``.zephon_cache_state/session.json`` — handshake file recording
      fingerprint, SHM segment names, and live owners.

    Dense index mapping ``(dataset.name, shard_id) → slot`` is built
    in-memory at ``__init__`` from the caller-provided ``locators`` dict.
    No persistent index mapping lives on disk; recovery uses the
    fingerprint to decide whether existing files are trustworthy.
    """

    def __init__(
        self,
        root: Path,
        storage: StorageBackend,
        *,
        locators: Mapping[tuple[int, int], ShardLocator],
        datasets: Mapping[int, Dataset],
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
        self._reset_lock_path = self._root / _RESET_LOCK_FILENAME
        self._reset_lock = FileLock(str(self._reset_lock_path))
        self._state_dir = self._root / _STATE_DIR_NAME
        self._session_path = self._state_dir / _SESSION_FILENAME
        self._pid = os.getpid()
        # Capture our OS-level start time now so the session entry encodes
        # ``(pid, start_time_ns)``. Prevents stale entries from masquerading
        # as live when a PID is recycled between a crash and the next init.
        self._start_time_ns = _process_start_time_ns(self._pid)
        self._owner_key = _make_owner_key(self._pid, self._start_time_ns)

        # In-memory dense index and per-index locator map (used by resolve/
        # touch and eviction). Identity map keys by (dataset_name, basename)
        # for disk-scan reconciliation; covers both raw and zip basenames.
        (
            self._index_map,
            self._locators_by_index,
            self._identity_to_index,
        ) = self._build_index_maps(locators, datasets)
        self._num_shards = len(self._locators_by_index)
        self._fingerprint, self._summary = self._compute_fingerprint(datasets, locators)
        self._cacheable_dataset_names: tuple[str, ...] = tuple(
            sorted({loc.dataset for loc in self._locators_by_index.values()})
        )

        # Full init (wipe? create SHM? reconcile? register owner?) runs
        # under _reset_lock so joiner processes cannot observe a partial
        # session.json. The lock is released only after session.json names
        # the newly-created SHM and our pid appears in the owners list.
        self._shared: CacheSharedState | None = None
        with self._reset_lock:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._run_session_state_machine()

        assert self._shared is not None
        logger.debug(
            "Initialized cache at %s (num_shards=%d, limit=%s)",
            self._root,
            self._num_shards,
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
            self._shared,
            self._reset_lock,
            self._session_path,
            self._owner_key,
        )

    # ------------------------------------------------------------------
    # Dense index map + fingerprint
    # ------------------------------------------------------------------

    @staticmethod
    def _build_index_maps(
        locators: Mapping[tuple[int, int], ShardLocator],
        datasets: Mapping[int, Dataset],
    ) -> tuple[
        dict[tuple[str, int], int],
        dict[int, ShardLocator],
        dict[tuple[str, str], tuple[int, str]],
    ]:
        """Build the three in-memory maps used at runtime.

        Canonical ordering: dataset names sorted, then shard ids within
        each dataset. Every worker process feeds the same ``datasets`` and
        ``locators`` in, so every process produces identical indices.

        Dataset names must be unique among cacheable (non-inmem) datasets
        because the dense map, disk layout (``{root}/{dataset.name}/``),
        and fingerprint all key by name. Collisions are refused loudly
        rather than silently overwriting — StaticMixtureWorkSource happens
        to overwrite duplicates by name on its own, but we do not rely on
        that and catch the problem at the cache boundary.
        """
        by_name: dict[str, list[tuple[int, ShardLocator]]] = {}
        for (_did, sid), loc in locators.items():
            by_name.setdefault(loc.dataset, []).append((int(sid), loc))

        # Reject duplicate dataset names. Scope is ALL datasets, not just
        # cacheable ones: even though in-memory datasets never touch the
        # cache, an inmem dataset sharing a name with a file-backed one
        # would make it ambiguous which dataset the cache's `(name, …)`
        # keys and on-disk `{cache_root}/{name}/` layout refer to from
        # the caller's perspective. Upstream (StaticMixtureWorkSource)
        # already overwrites duplicates silently, so we catch it here.
        names_seen: dict[str, int] = {}
        for did, ds in datasets.items():
            prev = names_seen.get(ds.name)
            if prev is not None and prev != did:
                raise ValueError(
                    f"Duplicate dataset name {ds.name!r} in dataset set "
                    f"(dataset_ids {prev} and {did}). Each dataset — "
                    f"in-memory or file-backed — must have a unique .name "
                    f"because the cache layout, dense index, and "
                    f"fingerprint all key on it."
                )
            names_seen[ds.name] = did

        # Only include datasets whose name is in the locators set (this
        # implicitly excludes in-memory datasets which don't produce
        # locators, matching collect_cacheable_locators' behavior).
        cacheable_names = sorted(by_name.keys())

        index_map: dict[tuple[str, int], int] = {}
        locators_by_index: dict[int, ShardLocator] = {}
        identity_to_index: dict[tuple[str, str], tuple[int, str]] = {}

        next_idx = 0
        for name in cacheable_names:
            shards_for_name = sorted(by_name[name], key=lambda pair: pair[0])
            for shard_id, loc in shards_for_name:
                index_map[(name, shard_id)] = next_idx
                locators_by_index[next_idx] = loc
                identity_to_index[(name, loc.raw.basename)] = (next_idx, "raw")
                if loc.zip is not None:
                    identity_to_index[(name, loc.zip.basename)] = (next_idx, "zip")
                next_idx += 1

        # Reference datasets to make pyright happy (they're used indirectly
        # via fingerprint computation elsewhere — here we just validate
        # that every locator belongs to a known dataset name).
        known_names = {ds.name for ds in datasets.values()}
        for name in cacheable_names:
            if name not in known_names:
                raise ValueError(f"Locator references unknown dataset name {name!r}")

        return index_map, locators_by_index, identity_to_index

    @staticmethod
    def _compute_fingerprint(
        datasets: Mapping[int, Dataset],
        locators: Mapping[tuple[int, int], ShardLocator],
    ) -> tuple[str, list[dict[str, object]]]:
        """SHA-256 fingerprint plus a per-dataset summary.

        The fingerprint is the canonical hash of the
        ``(dataset, locator)`` set — bytes, hashes, paths included — and
        remains the authoritative match key. The summary is a small
        ``{name, path, shard_count, total_raw_bytes}`` list per dataset
        persisted alongside the hash in ``session.json`` so the
        HARD_ERROR path can describe *what* changed rather than emit an
        opaque hex digest. Returns the pair ``(fingerprint, summary)``;
        the summary is ordered by dataset name.
        """
        by_name: dict[str, list[dict]] = {}
        raw_byte_totals: dict[str, int] = {}
        for (_did, sid), loc in locators.items():
            entry = {
                "shard_id": int(sid),
                "raw_basename": loc.raw.basename,
                "raw_bytes": int(loc.raw.bytes),
                "raw_hashes": (
                    {k: str(v) for k, v in dict(loc.raw.hashes).items()}
                    if loc.raw.hashes
                    else None
                ),
                "zip_basename": loc.zip.basename if loc.zip is not None else None,
                "zip_bytes": int(loc.zip.bytes) if loc.zip is not None else None,
                "zip_hashes": (
                    {k: str(v) for k, v in dict(loc.zip.hashes).items()}
                    if loc.zip is not None and loc.zip.hashes
                    else None
                ),
                "compression": loc.compression,
            }
            by_name.setdefault(loc.dataset, []).append(entry)
            raw_byte_totals[loc.dataset] = raw_byte_totals.get(loc.dataset, 0) + int(
                loc.raw.bytes
            )

        ds_by_name = {ds.name: ds for ds in datasets.values()}
        payload = [
            {
                "name": name,
                "path": ds_by_name[name].path if name in ds_by_name else None,
                "shards": sorted(by_name[name], key=lambda e: e["shard_id"]),
            }
            for name in sorted(by_name.keys())
        ]
        text = json.dumps(payload, sort_keys=True, default=str)
        fingerprint = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

        summary: list[dict[str, object]] = [
            {
                "name": name,
                "path": ds_by_name[name].path if name in ds_by_name else None,
                "shard_count": len(by_name[name]),
                "total_raw_bytes": int(raw_byte_totals.get(name, 0)),
            }
            for name in sorted(by_name.keys())
        ]
        return fingerprint, summary

    # ------------------------------------------------------------------
    # Session state machine
    # ------------------------------------------------------------------

    def _run_session_state_machine(self) -> None:
        """Dispatch one of six startup branches under ``_reset_lock``.

        Branches:

        - FIRST_INIT: no ``session.json`` — wipe, create SHM, write session.
        - JOIN: matching fingerprint + live owners — attach to existing SHM.
        - HARD_ERROR: fingerprint mismatch + live owners — raise
          ``CacheInUseError``.
        - FRESH_RESET: ``persist_state=False`` + stale session — wipe and
          create fresh SHM.
        - RESUME: ``persist_state=True`` + matching fingerprint + stale
          session — preserve files, reconcile LOCAL state from disk.
        - COLD_RESET: ``persist_state=True`` + fingerprint mismatch + stale
          session — wipe and create fresh SHM.
        """
        session = self._load_session()

        if session is None:
            # FIRST_INIT: no session file means no prior state we can
            # verify. Wipe regardless of persist_state because unverified
            # files on disk could be from a crashed run with different
            # content — reconciliation without a fingerprint is unsafe.
            self._wipe_cache_root_locked()
            self._create_fresh_session_locked()
            return

        live_owners = self._prune_dead_owners(session.get("owners", {}))
        existing_fp = str(session.get("fingerprint", ""))
        fp_match = existing_fp == self._fingerprint

        if live_owners:
            if fp_match:
                try:
                    self._join_session_locked(session, live_owners)
                    return
                except FileNotFoundError:
                    # session.json claims live owners but the SHM names are
                    # gone from /dev/shm. The old meta.json implementation
                    # silently recovered via an attach→create fallback; we
                    # reproduce that resilience here by downgrading the
                    # session to "stale" and continuing through the
                    # no-live-owners branches below. Plausible causes:
                    # - a pid-reuse collision where the recorded owner pid
                    #   refers to an unrelated process;
                    # - OS-level /dev/shm cleanup (containers, reboots);
                    # - a crashed creator that unlinked before peers detached.
                    logger.warning(
                        "session.json at %s names live owners but SHM is "
                        "missing; treating session as stale and rebuilding.",
                        self._session_path,
                    )
                    live_owners = {}
            else:
                stored_summary = session.get("summary")
                diff_lines = _diff_session_summary(stored_summary, self._summary)
                raise CacheInUseError(
                    str(self._root),
                    existing_fp,
                    current_fingerprint=self._fingerprint,
                    diff_lines=diff_lines,
                )

        # No live owners — session is stale, safe to reset or resume.
        old_shm_names = session.get("shm_names")
        if isinstance(old_shm_names, dict):
            CacheSharedState.unlink_by_names(old_shm_names)

        if self._persist_state and fp_match:
            # RESUME: preserve existing files, reconcile LOCAL state from disk.
            self._shared = CacheSharedState(capacity=self._num_shards, shm_names=None)
            self._reconcile_local_files_locked()
            self._write_new_session_record_locked()
            return

        # FRESH_RESET (persist_state=False) or COLD_RESET (fingerprint mismatch).
        self._wipe_cache_root_locked()
        self._create_fresh_session_locked()

    def _create_fresh_session_locked(self) -> None:
        self._shared = CacheSharedState(capacity=self._num_shards, shm_names=None)
        self._write_new_session_record_locked()

    def _write_new_session_record_locked(self) -> None:
        assert self._shared is not None
        owner_entry: dict[str, object] = {
            "instances": 1,
            "started_ns": time.time_ns(),
        }
        if self._start_time_ns is not None:
            owner_entry["process_start_ns"] = int(self._start_time_ns)
        session = {
            "session_id": str(uuid.uuid4()),
            "session_started_ns": time.time_ns(),
            "fingerprint": self._fingerprint,
            "summary": self._summary,
            "capacity": int(self._num_shards),
            "shm_names": self._shared.shm_names,
            "owners": {self._owner_key: owner_entry},
        }
        _atomic_write_json(self._session_path, session)

    def _join_session_locked(self, session: dict, live_owners: dict) -> None:
        shm_names_raw = session.get("shm_names")
        if not isinstance(shm_names_raw, dict):
            raise RuntimeError(
                f"session.json at {self._session_path} missing shm_names"
            )
        shm_names = {str(k): str(v) for k, v in shm_names_raw.items()}
        capacity = int(session.get("capacity", self._num_shards))
        self._shared = CacheSharedState(capacity=capacity, shm_names=shm_names)
        owners = dict(live_owners)
        entry = owners.get(self._owner_key)
        if not isinstance(entry, dict):
            entry = {"started_ns": time.time_ns()}
        entry.setdefault("started_ns", time.time_ns())
        if self._start_time_ns is not None:
            entry["process_start_ns"] = int(self._start_time_ns)
        entry["instances"] = int(entry.get("instances", 0)) + 1
        owners[self._owner_key] = entry
        session["owners"] = owners
        # Backfill summary for sessions written before the schema landed.
        # Safe because JOIN implies fingerprint match, so our local summary
        # is identical to what the original creator would have written.
        session.setdefault("summary", self._summary)
        _atomic_write_json(self._session_path, session)

    def _load_session(self) -> Optional[dict]:
        if not self._session_path.exists():
            return None
        try:
            return json.loads(self._session_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _prune_dead_owners(self, owners: object) -> dict:
        """Drop owners whose process is gone or whose pid has been recycled.

        Defense-in-depth: each owner key is ``f"{pid}-{start_time_ns}"``
        (compound), AND the entry value carries ``process_start_ns``.
        Both encode the same identity; we cross-check them against the
        live process's procfs start-time so a stale entry that survived
        a pid recycle is identified as dead from either signal alone.
        Legacy single-pid keys (and entries without
        ``process_start_ns``) keep working — they degrade to the
        kill-only check via :func:`_pid_alive`.
        """
        alive: dict[str, dict] = {}
        if not isinstance(owners, dict):
            return alive
        for key, meta in owners.items():
            pid, key_start = _parse_owner_key(key)
            if pid <= 0:
                logger.info("Dropping malformed cache-session owner %r", key)
                continue
            if not self._pid_alive(pid):
                logger.info("Dropping dead cache-session owner pid=%d key=%r", pid, key)
                continue
            actual_start = _process_start_time_ns(pid)
            entry = meta if isinstance(meta, dict) else {}
            recorded_start = entry.get("process_start_ns")
            if (
                actual_start is not None
                and isinstance(recorded_start, (int, float))
                and int(recorded_start) != int(actual_start)
            ):
                logger.info(
                    "Dropping recycled-PID cache-session owner pid=%d "
                    "recorded_start=%s actual_start=%s key=%r",
                    pid,
                    recorded_start,
                    actual_start,
                    key,
                )
                continue
            if (
                key_start is not None
                and actual_start is not None
                and key_start != actual_start
            ):
                logger.info(
                    "Dropping stale-compound-key cache-session owner "
                    "pid=%d key_start=%d actual_start=%d key=%r",
                    pid,
                    key_start,
                    actual_start,
                    key,
                )
                continue
            if int(entry.get("instances", 0)) <= 0:
                entry["instances"] = 1
            entry.setdefault("started_ns", time.time_ns())
            alive[key] = entry
        return alive

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Return True if *pid* names a live process (kill-only check).

        Recycle-detection lives in :meth:`_prune_dead_owners`, which
        compares the recorded ``process_start_ns`` against the live
        process's procfs start-time before trusting this answer.
        """
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
        return True

    def _wipe_cache_root_locked(self) -> None:
        """Remove everything under ``root`` except the lock files.

        Must be called under ``_reset_lock``. Preserves ``.reset.lock`` so
        the lock itself stays valid through the wipe.
        """
        preserve = {self._reset_lock_path, self._root / _CACHE_LOCK_FILENAME}
        for child in list(self._root.iterdir()):
            if child in preserve:
                continue
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                else:
                    shutil.rmtree(child, ignore_errors=True)
            except Exception:
                logger.warning("Failed to remove %s during cache reset", child)
        self._root.mkdir(parents=True, exist_ok=True)
        self._state_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Disk-scan reconciliation
    # ------------------------------------------------------------------

    def _reconcile_local_files_locked(self) -> None:
        """Scan the cache root and mark existing files as LOCAL.

        Invoked only on the RESUME branch (fingerprint matched, no live
        owners). Aggregates raw/zip presence per shard index and marks
        LOCAL only when the raw file exists — zip-only shards remain
        REMOTE to avoid materialising a new state.
        """
        assert self._shared is not None
        raw_present: dict[int, bool] = {}
        zip_present: dict[int, bool] = {}
        raw_bytes: dict[int, int] = {}
        zip_bytes: dict[int, int] = {}

        skip_dirs = {".locks", _STATE_DIR_NAME}
        skip_suffixes = (".tmp", ".part")

        for dataset_name in self._cacheable_dataset_names:
            ds_dir = self._root / dataset_name
            if not ds_dir.is_dir():
                continue
            for entry in ds_dir.iterdir():
                if entry.is_dir():
                    if entry.name in skip_dirs:
                        continue
                    continue  # unexpected nested dir — ignore
                if not entry.is_file():
                    continue
                name = entry.name
                if name.endswith(skip_suffixes):
                    continue
                key = (dataset_name, name)
                idx_role = self._identity_to_index.get(key)
                if idx_role is None:
                    # Orphan file from a prior run or unknown content.
                    continue
                idx, role = idx_role
                try:
                    size = entry.stat().st_size
                except OSError:
                    continue
                if role == "raw":
                    raw_present[idx] = True
                    raw_bytes[idx] = raw_bytes.get(idx, 0) + int(size)
                elif role == "zip":
                    zip_present[idx] = True
                    zip_bytes[idx] = zip_bytes.get(idx, 0) + int(size)

        states = self._shared.shard_states
        sizes = self._shared.shard_sizes
        access = self._shared.shard_access_ns
        now_ns = time.time_ns()
        total_usage = 0
        for idx, has_raw in raw_present.items():
            if not has_raw:
                continue
            size = raw_bytes.get(idx, 0)
            if zip_present.get(idx, False):
                if self._keep_zip:
                    size += zip_bytes.get(idx, 0)
                else:
                    # Delete orphan zip — we only keep what the current run would keep.
                    locator = self._locators_by_index.get(idx)
                    if locator is not None and locator.zip is not None:
                        zip_file = self._root / locator.dataset / locator.zip.basename
                        with contextlib.suppress(Exception):
                            zip_file.unlink(missing_ok=True)
            states[idx] = _ShardState.LOCAL
            sizes[idx] = size
            access[idx] = np.uint64(now_ns)
            total_usage += size
        self._shared.set_cache_usage(total_usage)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stats(self) -> CacheStats:
        assert self._shared is not None
        with self._cache_lock:
            return CacheStats(
                bytes_used=self._shared.get_cache_usage(),
                shards=self._shared.count_local(),
            )

    def close(self) -> None:
        # Delegate to the finalizer which handles session owner release +
        # shared state cleanup. Calling it is idempotent — a second call
        # (or GC triggering it later) is a harmless no-op.
        self._close_finalizer()

    def touch(self, locator: ShardLocator) -> None:
        """Record a shard access without taking the global lock."""
        index = self._index_for(locator)
        if index is None:
            return
        assert self._shared is not None
        if _ShardState(self._shared.shard_states[index]) == _ShardState.LOCAL:
            self._shared.set_access_time(index, time.time_ns())

    def resolve(self, locator: ShardLocator, *, blocking: bool = True) -> LocalShardRef:
        """Return a local reference, downloading and evicting as required.

        With ``blocking=False``, raises :class:`ShardNotReady` instead of
        waiting on a peer actively preparing the same shard. Orphaned
        downloads from dead peers are taken over rather than waited on.
        """
        assert self._shared is not None
        index = self._index_for(locator)
        if index is None:
            raise KeyError(
                f"Shard not registered with cache: dataset={locator.dataset} "
                f"shard={locator.shard_id}. The cache manager must be built with "
                f"locators covering every shard that resolve() is called for."
            )

        dataset_root = self._root / locator.dataset
        dataset_root.mkdir(parents=True, exist_ok=True)

        raw_path = dataset_root / locator.raw.basename
        zip_path = dataset_root / locator.zip.basename if locator.zip else None
        required = self._required_bytes(locator)

        shard_lock = self._shard_lock(dataset_root, locator.shard_id)

        while True:
            with self._cache_lock:
                state_value = _ShardState(self._shared.shard_states[index])
                if state_value == _ShardState.LOCAL:
                    if raw_path.is_file():
                        self._shared.set_access_time(index, time.time_ns())
                        return self._build_ref(
                            raw_path, zip_path, locator, cache_hit=True
                        )
                    self._mark_remote_locked(index)
                    continue

            # Not LOCAL. Try to become the active preparer via non-
            # blocking flock. ``timeout=0`` → ``fcntl.flock(LOCK_NB)``:
            #   - Acquired → previous preparer is dead (or nobody had
            #     started yet); we may proceed.
            #   - Timeout  → a live peer holds the lock and is actively
            #     downloading; wait and retry.
            try:
                shard_lock.acquire(timeout=0)
            except FileLockTimeout:
                if not blocking:
                    raise ShardNotReady(locator.dataset, int(locator.shard_id))
                time.sleep(_TICK_SECONDS)
                continue

            # We hold shard_lock. Re-check state under cache_lock to
            # handle races where the prior preparer finished between
            # our state observation and the flock acquisition.
            try:
                with self._cache_lock:
                    state_value = _ShardState(self._shared.shard_states[index])
                    if state_value == _ShardState.LOCAL:
                        if raw_path.is_file():
                            self._shared.set_access_time(index, time.time_ns())
                            return self._build_ref(
                                raw_path, zip_path, locator, cache_hit=True
                            )
                        # LOCAL but file missing — treat as REMOTE and prepare.
                        self._mark_remote_locked(index)
                        state_value = _ShardState.REMOTE

                    if state_value == _ShardState.PREPARING:
                        # We hold the flock but state is PREPARING → the
                        # previous owner died mid-download. Normalize
                        # stale on-disk state and proceed to prepare.
                        logger.warning(
                            "Cache: taking over orphaned PREPARING shard "
                            "dataset=%s shard_id=%s (previous preparer died "
                            "mid-download)",
                            locator.dataset,
                            int(locator.shard_id),
                        )
                        self._mark_remote_locked(index)

                    current_size = int(self._shared.shard_sizes[index])
                    additional = max(0, required - current_size)
                    if self._limit_bytes is not None:
                        self._ensure_capacity_locked(additional, skip_index=index)
                    self._shared.shard_states[index] = _ShardState.PREPARING
                    self._shared.set_access_time(index, time.time_ns())

                # Download while holding shard_lock. If this worker dies,
                # the kernel releases shard_lock, unblocking a peer to
                # take over via the LOCK_NB path above.
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
                            self._mark_remote_locked(index)

                entry_size = raw_path.stat().st_size
                actual_zip: Optional[Path] = None
                if zip_path and zip_path.exists():
                    if self._keep_zip:
                        entry_size += zip_path.stat().st_size
                        actual_zip = zip_path
                    else:
                        zip_path.unlink(missing_ok=True)
                with self._cache_lock:
                    old_size = int(self._shared.shard_sizes[index])
                    delta = entry_size - old_size
                    if delta:
                        self._shared.add_cache_usage(delta)
                    self._shared.shard_sizes[index] = entry_size
                    self._shared.shard_states[index] = _ShardState.LOCAL
                    self._shared.set_access_time(index, time.time_ns())

                return self._build_ref(raw_path, actual_zip, locator, cache_hit=False)
            finally:
                with contextlib.suppress(Exception):
                    shard_lock.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_for(self, locator: ShardLocator) -> Optional[int]:
        return self._index_map.get((locator.dataset, int(locator.shard_id)))

    def _mark_remote_locked(self, index: int) -> None:
        assert self._shared is not None
        size = int(self._shared.shard_sizes[index])
        if size:
            self._shared.add_cache_usage(-size)
        self._shared.shard_sizes[index] = 0
        self._shared.shard_states[index] = _ShardState.REMOTE
        self._shared.shard_access_ns[index] = 0

    def _ensure_capacity_locked(self, additional: int, *, skip_index: int) -> None:
        assert self._shared is not None
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
        assert self._shared is not None
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
        locator = self._locators_by_index.get(index)
        if locator is not None:
            dataset_root = self._root / locator.dataset
            raw_path = dataset_root / locator.raw.basename
            zip_path = dataset_root / locator.zip.basename if locator.zip else None
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


__all__ = [
    "CacheManager",
    "CacheStats",
]
