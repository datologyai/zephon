# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex shard format integration.

Vortex is a next-generation columnar file format designed for high-performance
data processing with zero-copy Arrow integration and GPU-friendly design.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from zephon._internal.io.formats.base import FormatHandler, register_format
from zephon._internal.io.index import find_and_load_index, warn_missing_index
from zephon._internal.io.index.index_types import ShardIndex, is_shard_index
from zephon._internal.io.protocols import RandomAccessShard
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.suffixes import VORTEX_SUFFIXES
from zephon._internal.io.types import LocalShardRef, ShardFile, ShardLocator

if TYPE_CHECKING:
    from zephon.io.dataset import Dataset

try:
    import vortex as _vortex
except ImportError:
    _vortex = None


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


def _read_vortex_row_count(path: str, storage: StorageBackend, size: int) -> int:
    """Read the row count from Vortex metadata through the supplied backend."""
    if _vortex is None:
        raise ImportError(
            "Vortex discovery requires vortex-data; "
            + 'install it with: pip install "zephon[vortex]"'
        )
    reader = _VortexReader(storage, path, size)
    return len(_vortex.open_readable(reader, without_segment_cache=True))


class VortexShard(RandomAccessShard):
    """Random access shard backed by a Vortex file."""

    def __init__(self, path: Path, *, length: int | None = None) -> None:
        if _vortex is None:
            raise RuntimeError(
                "Opening Vortex shards requires the 'vortex-data' package. "
                + 'Install with: pip install "zephon[vortex]"'
            )
        self._path = path
        try:
            self._file = _vortex.open(str(path))
        except RuntimeError:
            # Vortex wraps native IO errors. Surface a missing/inaccessible local
            # file as OSError so ResilientShard can resolve it again after eviction.
            # If it still exists, preserve the original Vortex error.
            path.stat()
            raise
        self._length = length if length is not None else len(self._file)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.getsamples([index])[0]

    def __len__(self) -> int:
        return self._length

    def close(self) -> None:
        self._file = None

    def getsamples(self, indices: list[int]) -> list[dict[str, Any]]:
        """Bulk read rows in request order, preserving duplicate indices."""
        if not indices:
            return []
        for i in indices:
            if i < 0 or i >= self._length:
                raise IndexError(i)
        if self._file is None or _vortex is None:
            raise RuntimeError("VortexShard has been closed")

        # Vortex scans require strictly increasing indices, even when callers
        # request rows out of order or more than once.
        sorted_unique = sorted(set(indices))
        index_to_pos = {idx: pos for pos, idx in enumerate(sorted_unique)}

        try:
            batch = self._file.scan(indices=_vortex.array(sorted_unique)).read_all()
        except RuntimeError:
            # Apply the same eviction check if the file disappears during a read.
            self._path.stat()
            raise
        arrow_table = batch.to_arrow_table()
        batch_dict = arrow_table.to_pydict()

        # Restore the caller's order and duplicate rows.
        return [
            {k: v[index_to_pos[idx]] for k, v in batch_dict.items()} for idx in indices
        ]


class VortexFormat(FormatHandler):
    """Format handler for Vortex datasets."""

    kind = "vortex"

    def discover(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, Any]]]:
        """Scan ``path`` and return shard statistics and metadata.

        Vortex files use the ``.vortex`` extension. Each file is treated as a
        single shard.

        If an ``index.json`` file exists (created by
        ``python -m zephon.build_index vortex``), it will be used for O(1)
        discovery instead of opening each file.
        """
        result = find_and_load_index(path, storage)
        if result is not None and is_shard_index(result):
            return self._discover_from_index_data(result)

        return self._discover_by_scanning(path, storage)

    def _discover_from_index_data(
        self, index: ShardIndex
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, Any]]]:
        """Load shard metadata from pre-built index data."""
        shards = index.get("shards", [])
        if not shards:
            raise ValueError(f"index.json contains no shards")

        shard_index: dict[int, int] = {}
        shard_meta: dict[int, dict[str, Any]] = {}

        for shard_id, shard in enumerate(shards):
            count = shard.get("num_rows", 0)
            shard_index[shard_id] = count
            shard_meta[shard_id] = {
                "raw": {
                    "basename": shard.get("basename", ""),
                    "bytes": shard.get("bytes", 0),
                    "hashes": shard.get("hashes", {}),
                },
                "extra": shard.get("extra", {"length": count}),
            }

        return shard_index, shard_meta

    def _discover_by_scanning(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, Any]]]:
        """Discover row counts through the supplied storage backend."""
        if _vortex is None:
            raise RuntimeError(
                "Discovering Vortex datasets requires the 'vortex-data' package. "
                + 'Install with: pip install "zephon[vortex]"'
            )

        entries = sorted(
            name for name in storage.listdir(path) if name.endswith(VORTEX_SUFFIXES)
        )
        if not entries:
            raise ValueError(f"No .vortex shards found under {path}")
        warn_missing_index(path, self.kind, num_shards=len(entries))

        def read_metadata(name: str) -> tuple[int, int]:
            full = os.path.join(path, name)
            try:
                size = int(storage.stat(full)["size"])
                return size, _read_vortex_row_count(full, storage, size)
            except Exception as exc:
                raise ValueError(f"Failed to read Vortex shard {full}: {exc}") from exc

        shard_index: dict[int, int] = {}
        shard_meta: dict[int, dict[str, Any]] = {}
        max_workers = min(32, max(1, (len(entries) + 4) // 5))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map keeps filename order even when later reads finish first.
            for shard_id, (size, count) in enumerate(
                executor.map(read_metadata, entries)
            ):
                shard_index[shard_id] = count
                shard_meta[shard_id] = {
                    "raw": {"basename": entries[shard_id], "bytes": size, "hashes": {}},
                    "extra": {"length": count},
                }

        return shard_index, shard_meta

    def discover_counts(
        self, path: str, storage: StorageBackend
    ) -> tuple[np.ndarray, np.ndarray]:
        """Count-only discovery: read per-shard ``num_rows`` from the index.

        Avoids building the per-shard ``shard_meta`` graph at ``from_path`` time
        and, on the index path, opening any ``.vortex`` files. Falls back to the
        full ``discover`` when no index is present — or when it has no shards,
        which ``discover`` rejects. Must agree with ``discover`` on
        ``(shard_id, num_rows)``.
        """
        result = find_and_load_index(path, storage)
        if is_shard_index(result) and result.get("shards"):
            counts = [shard.get("num_rows", 0) for shard in result["shards"]]
            return (
                np.arange(len(counts), dtype=np.int64),
                np.array(counts, dtype=np.int64),
            )
        return super().discover_counts(path, storage)

    def build_locators(self, dataset: "Dataset") -> Mapping[int, ShardLocator]:
        """Build shard locators from dataset metadata."""
        backend = dataset.backend
        path = backend.get("path")
        if not isinstance(path, str):
            raise ValueError("Vortex dataset missing 'path' in backend metadata")
        shards = backend.get("shards")
        if not isinstance(shards, Mapping):
            raise ValueError("Vortex dataset missing 'shards' metadata")

        locators: dict[int, ShardLocator] = {}
        for shard_id_obj, shard_meta in shards.items():
            shard_id = int(shard_id_obj)
            if not isinstance(shard_meta, Mapping):
                raise ValueError(f"Invalid shard metadata for shard {shard_id}")
            raw_meta = shard_meta.get("raw")
            if not isinstance(raw_meta, Mapping):
                raise ValueError(f"Shard {shard_id} missing Vortex raw metadata")
            basename = raw_meta.get("basename")
            if not isinstance(basename, str):
                raise ValueError(f"Shard {shard_id} missing basename for Vortex shard")
            bytes_value = raw_meta.get("bytes")
            if isinstance(bytes_value, (int, float)):
                raw_bytes = int(bytes_value)
            elif isinstance(bytes_value, str):
                raw_bytes = int(bytes_value)
            else:
                raise ValueError(
                    f"Shard {shard_id} missing or invalid byte size for Vortex shard"
                )
            raw = ShardFile(
                basename=basename,
                bytes=raw_bytes,
                hashes={
                    str(k): str(v) for k, v in dict(raw_meta.get("hashes", {})).items()
                },
            )
            extra = shard_meta.get("extra")
            locators[shard_id] = ShardLocator(
                dataset=dataset.name,
                shard_id=shard_id,
                format=self.kind,
                root=path,
                raw=raw,
                zip=None,
                compression=None,
                extra=extra if isinstance(extra, Mapping) else None,
            )
        return locators

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard:
        """Open a Vortex shard for random access reading."""
        length = None
        if local_ref.extra and "length" in local_ref.extra:
            length_value = local_ref.extra["length"]
            if isinstance(length_value, (int, float)):
                length = int(length_value)
            elif isinstance(length_value, str):
                try:
                    length = int(length_value)
                except ValueError:
                    length = None
        return VortexShard(local_ref.raw.path, length=length)


register_format(VortexFormat())

__all__ = ["VortexFormat", "VortexShard"]
