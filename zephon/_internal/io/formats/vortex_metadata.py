# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Lazy Vortex metadata reads through Zephon's storage abstraction."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat

from zephon._internal.io.index import ShardInfo, warn_missing_index
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.suffixes import VORTEX_SUFFIXES

try:
    import vortex as _vortex
except ImportError:
    _vortex = None

_VORTEX_DISCOVERY_MAX_WORKERS = 32


@dataclass(frozen=True)
class _VortexReader:
    """Adapt an existing backend to Vortex's ``ReadBytesAt`` protocol.

    No file handle or client is created here: the supplied backend owns path
    resolution and authentication, just as it does for Parquet footer reads.
    """

    storage: StorageBackend
    path: str
    file_size: int

    def size(self) -> int:
        """Return the size already obtained during discovery."""
        return self.file_size

    def read_at(self, offset: int, length: int) -> bytes | memoryview:
        """Read a positional range without sharing a mutable seek position."""
        if offset < 0 or length < 0:
            raise ValueError("offset and length must be non-negative")
        length = min(length, max(0, self.file_size - offset))
        if length == 0:
            return b""
        return self.storage.read_range(self.path, offset, length=length)


def read_vortex_shard_info(
    path: str, storage: StorageBackend, file_size: int | None = None
) -> ShardInfo:
    """Read a shard's row count without scanning its data or staging the file."""
    if _vortex is None:
        raise ImportError(
            "Vortex discovery requires vortex-data; "
            + 'install it with: pip install "zephon[vortex]"'
        )
    try:
        if file_size is None:
            file_size = int(storage.stat(path)["size"])
        reader = _VortexReader(storage, path, file_size)
        count = len(_vortex.open_readable(reader, without_segment_cache=True))
        return ShardInfo(
            basename=path.rsplit("/", 1)[-1],
            bytes=file_size,
            num_rows=count,
            extra={"length": count},
        )
    except Exception as exc:
        raise ValueError(f"Failed to read Vortex shard {path}: {exc}") from exc


def scan_vortex_shard_metadata(
    path: str, storage: StorageBackend, *, warn_if_unindexed: bool
) -> list[ShardInfo]:
    """Read metadata with bounded parallelism and deterministic shard ordering."""
    entries = sorted(
        name for name in storage.listdir(path) if name.endswith(VORTEX_SUFFIXES)
    )
    if not entries:
        raise ValueError(f"No .vortex shards found under {path}")
    if warn_if_unindexed:
        warn_missing_index(path, "vortex", num_shards=len(entries))

    paths = [os.path.join(path, name) for name in entries]
    max_workers = min(_VORTEX_DISCOVERY_MAX_WORKERS, max(1, (len(entries) + 4) // 5))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # map preserves filename order regardless of request completion order.
        return list(executor.map(read_vortex_shard_info, paths, repeat(storage)))
