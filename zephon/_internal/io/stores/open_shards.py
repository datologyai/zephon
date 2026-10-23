# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shard readers that stay open between batches."""

from __future__ import annotations

import contextlib
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass

from zephon._internal.io.formats.base import ShardOpener
from zephon._internal.io.protocols import RandomAccessShard, SampleLoadStats
from zephon._internal.io.types import LocalShardRef, ShardLocator
from zephon.types import SamplePayload

_FileIdentity = tuple[int, int, int, int]
_ShardKey = tuple[str, int, str]


def _file_identity(path: str) -> _FileIdentity:
    """Return values that change when a file is replaced or rewritten."""
    stats = os.stat(path)
    return (stats.st_dev, stats.st_ino, stats.st_size, stats.st_mtime_ns)


def _close_quietly(shards: list[RandomAccessShard]) -> None:
    for shard in shards:
        with contextlib.suppress(Exception):
            shard.close()


@dataclass(eq=False)
class _OpenShard:
    key: _ShardKey
    shard: RandomAccessShard
    identity: _FileIdentity
    leases: int = 0
    retired: bool = False


class OpenShardCache:
    """Keep a store's most recently used shard readers open between batches.

    ``ResilientShard`` opens and closes a reader for each read. Opening again
    can cost more than a small read, for example a file map or a header read.
    This opener wraps a format's opener and gives out leases on open readers.
    Closing a lease leaves its reader open.

    Each open checks the file's identity with ``stat``. A replaced file, such
    as a shard that the cache downloaded again, gets a new reader. A removed
    file raises ``FileNotFoundError``, so ``ResilientShard`` resolves it again.
    Each new reader also closes the readers of files that were removed or
    replaced since they were opened. An open reader of an evicted shard keeps
    its disk space until then. A reader that raises an error is not used again.
    A reader leaves the cache when it is the least recently used of more than
    ``max_shards``, and it closes when its last lease closes.
    """

    def __init__(self, opener: ShardOpener, max_shards: int) -> None:
        if max_shards < 1:
            raise ValueError("max_shards must be a positive integer")

        self._opener = opener
        self._max_shards = max_shards
        self._shards: OrderedDict[_ShardKey, _OpenShard] = OrderedDict()
        self._lock = threading.Lock()

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard:
        """Return a lease on the open reader, opening one for a new or changed file."""
        path = str(local_ref.raw.path)
        key = (locator.dataset, int(locator.shard_id), path)

        # Read the identity before the open. If the file changes between the
        # two, the next open sees an identity change and opens it again.
        try:
            identity = _file_identity(path)
        except OSError:
            self._discard_key(key)
            raise

        with self._lock:
            entry = self._shards.get(key)
            if entry is not None and entry.identity == identity:
                entry.leases += 1
                self._shards.move_to_end(key)
                return _ShardLease(self, entry)

        self._close_changed_files()

        # Opening does IO, so it runs outside the lock.
        entry = _OpenShard(
            key, self._opener.open_shard(locator, local_ref), identity, leases=1
        )
        with self._lock:
            removed = [self._shards.pop(key, None)]
            self._shards[key] = entry
            while len(self._shards) > self._max_shards:
                removed.append(self._shards.popitem(last=False)[1])
            to_close = [
                shard
                for old in removed
                if old is not None and (shard := self._retire(old)) is not None
            ]

        _close_quietly(to_close)
        return _ShardLease(self, entry)

    def close(self) -> None:
        """Close every reader now, or when its last lease closes."""
        with self._lock:
            entries = list(self._shards.values())
            self._shards.clear()
            to_close = [
                shard for entry in entries if (shard := self._retire(entry)) is not None
            ]

        _close_quietly(to_close)

    def _release(self, entry: _OpenShard) -> None:
        with self._lock:
            entry.leases -= 1
            close = entry.retired and entry.leases == 0

        if close:
            _close_quietly([entry.shard])

    def _discard(self, entry: _OpenShard) -> None:
        """Remove ``entry`` from the cache; its open leases keep it until they close."""
        with self._lock:
            if self._shards.get(entry.key) is entry:
                del self._shards[entry.key]
            shard = self._retire(entry)

        if shard is not None:
            _close_quietly([shard])

    def _discard_key(self, key: _ShardKey) -> None:
        with self._lock:
            entry = self._shards.get(key)
        if entry is not None:
            self._discard(entry)

    def _close_changed_files(self) -> None:
        """Discard readers whose files were removed or replaced."""
        with self._lock:
            entries = list(self._shards.values())

        for entry in entries:
            try:
                changed = _file_identity(entry.key[2]) != entry.identity
            except OSError:
                changed = True
            if changed:
                self._discard(entry)

    def _retire(self, entry: _OpenShard) -> RandomAccessShard | None:
        """Mark ``entry`` closed and return its reader when no lease uses it.

        The caller holds the lock and closes the returned reader after it
        releases the lock.
        """
        if entry.retired:
            return None

        entry.retired = True
        return entry.shard if entry.leases == 0 else None


class _ShardLease(RandomAccessShard):
    """One use of an open reader. Closing it leaves the reader open."""

    def __init__(self, cache: OpenShardCache, entry: _OpenShard) -> None:
        self._cache = cache
        self._entry = entry
        self._released = False

    def __getitem__(
        self, index: int
    ) -> SamplePayload | tuple[SamplePayload, SampleLoadStats]:
        try:
            return self._entry.shard[index]
        except BaseException:
            self._cache._discard(self._entry)
            raise

    def getsamples(
        self, indices: list[int]
    ) -> list[SamplePayload] | tuple[list[SamplePayload], list[SampleLoadStats]]:
        try:
            return self._entry.shard.getsamples(indices)
        except BaseException:
            self._cache._discard(self._entry)
            raise

    def __len__(self) -> int:
        return len(self._entry.shard)

    def close(self) -> None:
        if self._released:
            return

        self._released = True
        self._cache._release(self._entry)


__all__ = ["OpenShardCache"]
