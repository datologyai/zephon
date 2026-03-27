# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Parquet shard format support.

Provides efficient random access and bulk reading of Parquet datasets with
support for index.json preprocessing (recommended for large datasets) and
fallback direct metadata reading.

Key features:
- Binary search for O(log n) row group lookup
- Bulk read optimization grouping by row group
- Dual discovery: index.json (fast) or direct metadata reads (fallback)
"""

import bisect
import os
import struct
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from zephon.io.formats.base import FormatHandler, register_format
from zephon.io.index import find_and_load_index
from zephon.io.index.index_types import ShardIndex, is_shard_index
from zephon.io.protocols import RandomAccessShard
from zephon.io.storage.base import StorageBackend
from zephon.io.types import LocalShardRef, ShardFile, ShardLocator

if TYPE_CHECKING:
    from zephon.io.dataset import Dataset

# Lazy import of PyArrow to avoid hard dependency
_pa = None
_pq = None
_PARQUET_FOOTER_PREFETCH_BYTES = 64 * 1024


def _ensure_pyarrow():
    """Lazy import of PyArrow with helpful error message."""
    global _pa, _pq
    if _pa is None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            _pa = pa
            _pq = pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for Parquet format support. "
                "Install with: pip install zephon[parquet]"
            ) from exc
    return _pa, _pq


_CachedRG = dict[str, np.ndarray]


def _arrow_table_to_numpy(table: Any) -> _CachedRG:
    """Convert an Arrow table to a dict of numpy arrays, zero-copy when possible."""
    pa, _ = _ensure_pyarrow()
    result: _CachedRG = {}
    for name in table.column_names:
        col = table.column(name)
        arr = col.chunk(0) if col.num_chunks == 1 else col.combine_chunks()
        col_type = arr.type

        if isinstance(col_type, pa.lib.FixedSizeListType):
            flat = arr.values.to_numpy(zero_copy_only=False)
            result[name] = flat.reshape(len(arr), col_type.list_size)
        elif isinstance(col_type, pa.lib.ListType):
            offsets = arr.offsets.to_numpy(zero_copy_only=False)
            values = arr.values.to_numpy(zero_copy_only=False)
            rows = np.empty(len(arr), dtype=object)
            for i in range(len(arr)):
                rows[i] = values[offsets[i] : offsets[i + 1]]
            result[name] = rows
        else:
            try:
                result[name] = arr.to_numpy(zero_copy_only=False)
            except Exception:
                result[name] = np.array(arr.to_pylist(), dtype=object)
    return result


def _extract_row(columns: _CachedRG, idx: int) -> dict[str, object]:
    return {name: arr[idx] for name, arr in columns.items()}


_DEFAULT_RG_CACHE_BYTES = 2 * 1024**3  # 2 GiB


def _rg_size_bytes(columns: _CachedRG) -> int:
    return sum(a.nbytes for a in columns.values())


class _RowGroupCache:
    """Thread-safe LRU cache keyed by (file_path, row_group_id) → _CachedRG.

    Evicts LRU entries when total numpy buffer usage exceeds ``max_bytes``.
    Set ``max_bytes=0`` to disable caching (low-memory mode).
    """

    def __init__(self, max_bytes: int = _DEFAULT_RG_CACHE_BYTES) -> None:
        self._cache: OrderedDict[tuple[str, int], _CachedRG] = OrderedDict()
        self._lock = threading.Lock()
        self._max_bytes = max_bytes
        self._used_bytes = 0

    def get(self, path: str, rg_id: int) -> _CachedRG | None:
        key = (path, rg_id)
        with self._lock:
            val = self._cache.get(key)
            if val is not None:
                self._cache.move_to_end(key)
            return val

    def get_many(self, path: str, rg_ids: list[int]) -> dict[int, _CachedRG] | None:
        """Return all requested row groups if all cached, else None."""
        with self._lock:
            result: dict[int, _CachedRG] = {}
            for rg_id in rg_ids:
                key = (path, rg_id)
                val = self._cache.get(key)
                if val is None:
                    return None
                self._cache.move_to_end(key)
                result[rg_id] = val
            return result

    def put(self, path: str, rg_id: int, columns: _CachedRG) -> None:
        if self._max_bytes == 0:
            return
        key = (path, rg_id)
        entry_bytes = _rg_size_bytes(columns)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return
            self._cache[key] = columns
            self._used_bytes += entry_bytes
            while self._used_bytes > self._max_bytes and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                self._used_bytes -= _rg_size_bytes(evicted)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._used_bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes


class ParquetShard(RandomAccessShard):
    """Random access shard backed by a Parquet file.

    Optimized for training workloads with:
    - Binary search row group lookup
    - Bulk read optimization (group by row group)
    - Shared row group cache across open/close cycles
    """

    def __init__(
        self,
        path: Path,
        row_groups: list[dict],
        rg_cache: _RowGroupCache,
        metadata: Any | None = None,
    ) -> None:
        _ensure_pyarrow()

        self._path = path
        self._row_groups = row_groups
        self._metadata = metadata
        self._rg_cache = rg_cache

        self._rg_boundaries = self._build_cumulative_index(row_groups)
        self._length = self._rg_boundaries[-1] if self._rg_boundaries else 0

    def _build_cumulative_index(self, row_groups: list[dict]) -> list[int]:
        """Build cumulative row count index for O(log n) lookup.

        Args:
            row_groups: List of dicts with num_rows field

        Returns:
            List of cumulative row counts: [0, n0, n0+n1, n0+n1+n2, ...]
        """
        boundaries = [0]
        for rg in row_groups:
            boundaries.append(boundaries[-1] + rg["num_rows"])
        return boundaries

    def _locate_row_group(self, index: int) -> tuple[int, int]:
        """Map global record index to (row_group_id, local_row_index).

        Uses binary search for O(log n) complexity.

        Args:
            index: Global record index (0-based)

        Returns:
            Tuple of (row_group_id, local_row_index)
        """
        rg_id = bisect.bisect_right(self._rg_boundaries, index) - 1
        local_idx = index - self._rg_boundaries[rg_id]
        return rg_id, local_idx

    def _read_row_group(self, rg_id: int) -> _CachedRG:
        path_key = str(self._path)
        cached = self._rg_cache.get(path_key, rg_id)
        if cached is not None:
            return cached

        pq_file = _pq.ParquetFile(self._path, metadata=self._metadata)
        table = pq_file.read_row_group(rg_id)
        del pq_file

        columns = _arrow_table_to_numpy(table)
        self._rg_cache.put(path_key, rg_id, columns)
        return columns

    def __getitem__(self, index: int) -> dict[str, object]:
        """Single record random access.

        Args:
            index: Record index (0-based)

        Returns:
            Record as dictionary

        Raises:
            IndexError: If index is out of bounds
        """
        if index < 0 or index >= self._length:
            raise IndexError(index)

        rg_id, local_idx = self._locate_row_group(index)
        columns = self._read_row_group(rg_id)
        return _extract_row(columns, local_idx)

    def getsamples(self, indices: list[int]) -> list[dict[str, object]]:
        """Bulk read: groups by row group, single file open for cache misses."""
        if not indices:
            return []

        for idx in indices:
            if idx < 0 or idx >= self._length:
                raise IndexError(idx)

        rg_groups: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        for orig_pos, idx in enumerate(indices):
            rg_id, local_idx = self._locate_row_group(idx)
            rg_groups[rg_id].append((orig_pos, local_idx))

        path_key = str(self._path)
        missing_rg_ids = [
            rg_id for rg_id in rg_groups if self._rg_cache.get(path_key, rg_id) is None
        ]
        if missing_rg_ids:
            pq_file = _pq.ParquetFile(self._path, metadata=self._metadata)
            for rg_id in missing_rg_ids:
                table = pq_file.read_row_group(rg_id)
                self._rg_cache.put(path_key, rg_id, _arrow_table_to_numpy(table))
            del pq_file

        results: list[dict[str, object] | None] = [None] * len(indices)
        for rg_id, items in rg_groups.items():
            columns = self._rg_cache.get(path_key, rg_id)
            assert columns is not None, (
                f"row group {rg_id} unexpectedly missing from cache"
            )
            for orig_pos, local_idx in items:
                results[orig_pos] = _extract_row(columns, local_idx)

        return results  # type: ignore[return-value]

    def __len__(self) -> int:
        """Return total number of records in shard."""
        return self._length

    def close(self) -> None:
        pass


class ParquetFormat(FormatHandler):
    """Format handler for Parquet datasets.

    Supports two discovery modes:
    1. Fast path: Read index.json (created by preprocessing tool)
    2. Fallback: Read Parquet metadata directly (with parallel reads)
    """

    kind = "parquet"
    _METADATA_CACHE_MAX_SIZE = 256

    def __init__(self) -> None:
        self._metadata_cache: OrderedDict[str, Any] = OrderedDict()
        self._metadata_lock = threading.Lock()
        max_bytes = int(
            os.environ.get("ZEPHON_PARQUET_RG_CACHE_BYTES", _DEFAULT_RG_CACHE_BYTES)
        )
        self._rg_cache = _RowGroupCache(max_bytes=max_bytes)

    def _read_metadata_only(
        self, path: str, storage: StorageBackend, *, size: int
    ) -> Any:
        """Read Parquet metadata without downloading the entire object when possible.

        Because the footer size is unknown, you can't do a single "read footer" call
        without first knowing its length. The prefetch is an optimization:

        1. Read up to 64 KB from the end (_PARQUET_FOOTER_PREFETCH_BYTES).
        2. Parse the last 8 bytes – get metadata_len and verify "PAR1".
        3. Check if the footer fits in the prefetch – if footer_size <= prefetch_size,
           use the prefetched bytes.
        4. Otherwise, do a second read – if the footer is larger than 64 KB, read
           exactly footer_size bytes from the correct offset.

        For most Parquet files, the footer is well under 64 KB, so one read is enough.
        The prefetch avoids:
        - A tiny read of just 8 bytes (which can be inefficient on some backends).
        - A second read in the common case.
        """
        assert _pa is not None and _pq is not None

        if size < 8:
            raise ValueError(f"File too small to be a valid Parquet file: {path}")

        # 1. Read up to 64 KB from the end.
        prefetch_size = min(size, _PARQUET_FOOTER_PREFETCH_BYTES)
        footer_start = size - prefetch_size
        footer = storage.read_range(path, footer_start, length=prefetch_size)
        if len(footer) != prefetch_size:
            raise ValueError(f"Incomplete Parquet footer prefetch for {path}")

        # 2. Parse the last 8 bytes – get metadata_len and verify "PAR1".
        trailer = footer[-8:]
        if trailer[4:] != b"PAR1":
            raise ValueError(f"Invalid Parquet footer trailer for {path}")

        metadata_len = struct.unpack("<I", trailer[:4])[0]
        footer_size = metadata_len + 8
        if footer_size > size:
            raise ValueError(f"Invalid Parquet footer size for {path}")

        if footer_size > len(footer):
            # 4. Footer larger than prefetch – read exactly footer_size bytes.
            footer = storage.read_range(path, size - footer_size, length=footer_size)
            if len(footer) != footer_size:
                raise ValueError(f"Incomplete Parquet footer read for {path}")
        else:
            # 3. Use the prefetched bytes.
            footer = footer[-footer_size:]

        return _pq.read_metadata(_pa.BufferReader(footer))

    def discover(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, object]]]:
        """Discover Parquet dataset and return shard metadata.

        Tries index.json first (O(1)), falls back to direct metadata reads (O(N)).

        Args:
            path: Dataset directory path
            storage: Storage backend for file access

        Returns:
            Tuple of (shard_index, shard_meta) where:
            - shard_index: Maps shard_id -> record count
            - shard_meta: Maps shard_id -> metadata dict

        Raises:
            ValueError: If no Parquet files or index found
        """
        _ensure_pyarrow()

        result = find_and_load_index(path, storage)
        if result is not None and is_shard_index(result):
            if self._is_valid_parquet_index(result):
                return self._discover_from_index_data(result)

        return self._discover_from_files(path, storage)

    def _is_valid_parquet_index(self, data: ShardIndex) -> bool:
        """Check if data is a valid Parquet index (has shards with row_groups)."""
        if not data["shards"]:
            return False
        first_shard = data["shards"][0]
        extra = first_shard.get("extra", {})
        return isinstance(extra, dict) and "row_groups" in extra

    def _discover_from_index_data(
        self, data: ShardIndex
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, object]]]:
        """Fast O(1) discovery from preprocessed index data."""
        shards = data.get("shards", [])
        if not shards:
            raise ValueError("Empty shards list in Parquet index")

        shard_index: dict[int, int] = {}
        shard_meta: dict[int, dict[str, object]] = {}

        for shard_id, shard in enumerate(shards):
            num_rows = shard.get("num_rows", 0)
            shard_index[shard_id] = num_rows

            extra = shard.get("extra", {})

            shard_meta[shard_id] = {
                "raw": {
                    "basename": shard["basename"],
                    "bytes": shard.get("bytes", 0),
                    "hashes": shard.get("hashes", {}),
                },
                "extra": {
                    "num_rows": num_rows,
                    "num_row_groups": extra.get("num_row_groups", 0),
                    "row_groups": extra.get("row_groups", []),
                },
            }

        return shard_index, shard_meta

    def _discover_from_files(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, object]]]:
        """Fallback: read Parquet metadata from each file.

        Uses parallel reads (ThreadPoolExecutor) to speed up discovery.

        Args:
            path: Dataset directory path
            storage: Storage backend

        Returns:
            Tuple of (shard_index, shard_meta)

        Raises:
            ValueError: If no .parquet files found
        """
        entries = sorted(
            name for name in storage.listdir(path) if name.endswith(".parquet")
        )
        if not entries:
            raise ValueError(f"No .parquet shards found under {path}")

        shard_index: dict[int, int] = {}
        shard_meta: dict[int, dict[str, object]] = {}

        def read_metadata(shard_id: int, name: str):
            """Read metadata from a single Parquet file."""
            full_path = os.path.join(path, name)
            stats = storage.stat(full_path)
            size = int(stats.get("size", 0))

            metadata = self._read_metadata_only(full_path, storage, size=size)

            # Extract row group info
            row_groups = []
            for rg_idx in range(metadata.num_row_groups):
                rg = metadata.row_group(rg_idx)
                row_groups.append(
                    {
                        "num_rows": rg.num_rows,
                        "total_byte_size": rg.total_byte_size,
                    }
                )

            meta: dict[str, object] = {
                "raw": {
                    "basename": name,
                    "bytes": size,
                    "hashes": {},
                },
                "extra": {
                    "num_rows": metadata.num_rows,
                    "num_row_groups": metadata.num_row_groups,
                    "row_groups": row_groups,
                },
            }

            return (
                shard_id,
                metadata.num_rows,
                meta,
            )

        # Read metadata in parallel for better performance
        max_workers = min(32, (len(entries) + 4) // 5)  # Reasonable parallelism
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(read_metadata, shard_id, name): shard_id
                for shard_id, name in enumerate(entries)
            }

            for future in as_completed(futures):
                shard_id, num_rows, meta = future.result()
                shard_index[shard_id] = num_rows
                shard_meta[shard_id] = meta

        return shard_index, shard_meta

    def build_locators(self, dataset: "Dataset") -> Mapping[int, ShardLocator]:
        """Convert discovered metadata into ShardLocators.

        Args:
            dataset: Dataset instance with backend metadata

        Returns:
            Mapping from shard_id to ShardLocator
        """
        backend = dataset.backend
        path = backend.get("path")
        if not isinstance(path, str):
            raise ValueError("Parquet dataset missing 'path' in backend metadata")
        shards = backend.get("shards")
        if not isinstance(shards, Mapping):
            raise ValueError("Parquet dataset missing 'shards' metadata")

        locators: dict[int, ShardLocator] = {}
        for shard_id_obj, shard_meta in shards.items():
            shard_id = int(shard_id_obj)
            if not isinstance(shard_meta, Mapping):
                raise ValueError(f"Invalid shard metadata for shard {shard_id}")

            raw_meta = shard_meta.get("raw")
            if not isinstance(raw_meta, Mapping):
                raise ValueError(f"Shard {shard_id} missing raw metadata")

            basename = raw_meta.get("basename")
            if not isinstance(basename, str):
                raise ValueError(f"Shard {shard_id} missing basename")

            bytes_value = raw_meta.get("bytes")
            if isinstance(bytes_value, Integral):
                raw_bytes = int(bytes_value)
            elif isinstance(bytes_value, str):
                raw_bytes = int(bytes_value)
            else:
                raise ValueError(f"Shard {shard_id} missing or invalid byte size")

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
        """Open a Parquet shard for reading.

        Args:
            locator: Shard location metadata
            local_ref: Local file reference after caching/download

        Returns:
            ParquetShard instance with cached metadata
        """
        # Extract metadata from local_ref.extra
        extra = local_ref.extra or {}
        row_groups = extra.get("row_groups", [])

        if not row_groups:
            raise ValueError(
                f"Missing row_groups metadata for shard {locator.shard_id}"
            )

        metadata = self._get_cached_metadata(local_ref.raw.path)

        return ParquetShard(
            path=local_ref.raw.path,
            row_groups=row_groups,
            metadata=metadata,
            rg_cache=self._rg_cache,
        )

    def _get_cached_metadata(self, path: Path) -> Any:
        cache_key = str(path)
        with self._metadata_lock:
            metadata = self._metadata_cache.get(cache_key)
            if metadata is not None:
                self._metadata_cache.move_to_end(cache_key)
                return metadata

        _, pq = _ensure_pyarrow()
        metadata = pq.read_metadata(path)

        with self._metadata_lock:
            existing = self._metadata_cache.get(cache_key)
            if existing is not None:
                self._metadata_cache.move_to_end(cache_key)
                return existing
            self._metadata_cache[cache_key] = metadata
            while len(self._metadata_cache) > self._METADATA_CACHE_MAX_SIZE:
                self._metadata_cache.popitem(last=False)
            return metadata


# Register format
register_format(ParquetFormat())

__all__ = ["ParquetFormat", "ParquetShard"]
