# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex shard format integration.

Vortex is a next-generation columnar file format designed for high-performance
data processing with zero-copy Arrow integration and GPU-friendly design.
"""

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from zephon._internal.io.formats.base import FormatHandler, register_format
from zephon._internal.io.formats.vortex_readable import StorageReadAt
from zephon._internal.io.index import find_and_load_index, warn_missing_index
from zephon._internal.io.index.index_types import ShardIndex, is_shard_index
from zephon._internal.io.protocols import RandomAccessShard
from zephon._internal.io.storage import StorageBackend, is_remote_path
from zephon._internal.io.suffixes import VORTEX_SUFFIXES
from zephon._internal.io.types import (
    LocalShardRef,
    RemoteShardRef,
    ShardFile,
    ShardLocator,
)

if TYPE_CHECKING:
    import pyarrow as pa
    from vortex import SegmentCache
    from vortex.file import Footer

    from zephon.io.dataset import Dataset
    from zephon.io.options import StoreOptions

try:
    import vortex as _vortex
except ImportError:
    _vortex = None


# Caches shared by every Vortex shard of a process, by size. Vortex caches cannot
# be pickled or used across a fork, so each process makes its own on first use.
_segment_caches: dict[tuple[int, int], "SegmentCache"] = {}
_segment_caches_lock = threading.Lock()


def process_segment_cache(max_bytes: int) -> "SegmentCache | None":
    """Return this process's shared segment cache, or ``None`` when ``max_bytes`` is 0."""
    if max_bytes == 0 or _vortex is None:
        return None

    key = (os.getpid(), max_bytes)
    with _segment_caches_lock:
        cache = _segment_caches.get(key)
        if cache is None:
            cache = _vortex.SegmentCache(max_bytes)
            _segment_caches[key] = cache
        return cache


def segment_cache_key(locator: ShardLocator) -> str:
    """Identify a shard's contents within a shared segment cache.

    Datasets must not change during a run, so the location, size and any known
    hashes identify the contents.
    """
    raw = locator.raw
    hashes = ",".join(f"{k}={v}" for k, v in sorted(raw.hashes.items()))
    return f"{locator.root}/{raw.basename}#{raw.bytes}#{hashes}"


# Vortex stores rebuilt from obstore stores, by process and obstore store. Each
# entry keeps its obstore store, so that its id is not used again.
_vortex_stores: dict[tuple[int, int], tuple[object, Any]] = {}
_vortex_stores_lock = threading.Lock()


def _vortex_store(store: object) -> Any:
    """Return the Vortex store with the configuration of an obstore ``store``.

    Vortex accepts only its own store classes, so the store is rebuilt from its
    configuration, once per process, and then shared by every open. The result
    is ``None`` for a store without a configuration, such as ``MemoryStore``.
    """
    import vortex.store

    key = (os.getpid(), id(store))
    with _vortex_stores_lock:
        entry = _vortex_stores.get(key)
        if entry is not None:
            return entry[1]

        make = getattr(vortex.store, type(store).__name__, None)
        get_args = getattr(store, "__getnewargs_ex__", None)
        converted = None
        if make is not None and get_args is not None:
            args, kwargs = get_args()
            converted = make(*args, **kwargs)

        _vortex_stores[key] = (store, converted)
        return converted


def _native_store(storage: StorageBackend, path: str) -> tuple[Any, str] | None:
    """Return a Vortex store and key that read ``path`` natively, if there is one.

    There is one when the backend of ``path`` uses obstore, as the cloud
    backends do. Vortex then does the IO itself, without the GIL.
    """
    object_store = getattr(storage, "object_store", None)
    found = object_store(path) if object_store is not None else None
    if found is None:
        return None

    store, key = found
    converted = _vortex_store(store)
    return (converted, key) if converted is not None else None


class VortexShard(RandomAccessShard):
    """Random access shard backed by a Vortex file.

    With a ``store``, Vortex reads the key ``path`` from it natively. With a
    ``source``, Vortex reads through Python positional reads, limited by
    ``concurrency``. With neither, it opens the local ``path`` natively.

    A ``segment_cache`` with a ``cache_key`` keeps decoded segments after this
    shard closes, so a later open of the same file does not read them again.
    Without one, the file has a private cache unless ``without_segment_cache``.

    Rows are dicts. A list column of numbers without nulls in its values gives
    a read-only NumPy array per row, which views the batch's Arrow memory.
    Other columns give Python objects, as ``pyarrow.Array.to_pylist`` does.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        length: int | None = None,
        store: Any = None,
        source: StorageReadAt | None = None,
        concurrency: int | None = None,
        footer: "Footer | None" = None,
        segment_cache: "SegmentCache | None" = None,
        cache_key: str | None = None,
        without_segment_cache: bool = False,
    ) -> None:
        if _vortex is None:
            raise RuntimeError(
                "Opening Vortex shards requires the 'vortex-data' package. "
                + 'Install with: pip install "zephon[vortex]"'
            )
        self._path = path
        require_read_at()

        if source is None:
            self._file = _vortex.open(
                str(path),
                store=store,
                footer=footer,
                without_segment_cache=without_segment_cache,
                segment_cache=segment_cache,
                cache_key=cache_key,
            )
        else:
            self._file = _vortex.open_readable(
                source,
                footer=footer,
                concurrency=concurrency,
                without_segment_cache=without_segment_cache,
                segment_cache=segment_cache,
                cache_key=cache_key,
            )
        self._footer = self._file.footer
        self._length = length if length is not None else len(self._file)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # One path for single rows and batches, so both give the same types.
        return self.getsamples([index])[0]

    def __len__(self) -> int:
        return self._length

    @property
    def footer(self) -> "Footer":
        """The parsed Vortex footer, which reopens this file without reading it again."""
        return self._footer

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

        batch = self._file.scan(indices=_vortex.array(sorted_unique)).read_all()

        # Restore the caller's order and duplicate rows. Arrow has no take for
        # the view types that Vortex exports, so Vortex takes the rows.
        positions = _vortex.array([index_to_pos[idx] for idx in indices])
        table = batch.take(positions).to_arrow_table(combine_chunks=True)

        columns = {
            name: _column_values(table.column(name).chunk(0))
            for name in table.column_names
        }

        return [
            {name: values[row] for name, values in columns.items()}
            for row in range(len(indices))
        ]


def _column_values(array: "pa.Array") -> Sequence[Any]:
    """Export a column as one value per row.

    A list of numbers becomes NumPy arrays, the same as MDS and LitData give.
    Each is a view of one array for the whole column, so no values are copied
    when they are already contiguous. Any other column becomes Python objects.
    """
    import pyarrow as pa

    kind = array.type
    is_list = (
        pa.types.is_fixed_size_list(kind)
        or pa.types.is_list(kind)
        or pa.types.is_large_list(kind)
    )
    if not is_list:
        return array.to_pylist()

    item = kind.value_type
    is_number = (
        pa.types.is_integer(item)
        or pa.types.is_floating(item)
        or pa.types.is_boolean(item)
    )
    # Values that hold nulls have no exact NumPy form, so they stay Python objects.
    values = array.flatten()
    if not is_number or values.null_count:
        return array.to_pylist()

    numbers = values.to_numpy(zero_copy_only=False)
    if pa.types.is_fixed_size_list(kind) and not array.null_count:
        return numbers.reshape(len(array), kind.list_size)

    # ``flatten`` drops the values of null rows, so offsets are recomputed from
    # the row lengths rather than taken from the array.
    lengths = array.value_lengths().fill_null(0).to_numpy(zero_copy_only=False)
    ends = np.cumsum(lengths)
    starts = ends - lengths
    valid = array.is_valid().to_numpy(zero_copy_only=False)
    return [
        numbers[start:end] if is_valid else None
        for start, end, is_valid in zip(starts, ends, valid)
    ]


def require_read_at() -> None:
    """Fail explicitly when the experimental bindings have not been installed."""
    if (
        _vortex is None
        or not hasattr(_vortex.io, "ReadBytesAt")
        or not hasattr(_vortex, "SegmentCache")
    ):
        raise RuntimeError(
            "Reading Vortex shards requires Vortex's Python ReadAt bindings; "
            + "install vortex-data 0.88.0 or later."
        )


class VortexFormat(FormatHandler):
    """Format handler for Vortex datasets.

    A local shard is opened natively. Without the shard cache, a remote shard
    is read in place. When its backend uses obstore, Vortex reads it natively
    from the same store. Otherwise Vortex reads it through Python positional
    reads on the backend, at most ``read_concurrency`` at once per file. Each
    of these reads retries ``OSError`` as the ``read_retry_*`` arguments set,
    because an error that crosses into Vortex is no longer an ``OSError``.

    All shards share the process's segment cache of ``segment_cache_bytes``,
    so that segments outlive the per-batch open; ``0`` disables it.
    """

    kind = "vortex"

    def __init__(
        self,
        *,
        segment_cache_bytes: int = 0,
        read_concurrency: int | None = None,
        read_retry_attempts: int = 1,
        read_retry_initial_backoff: float = 0.1,
        read_retry_max_backoff: float = 2.0,
    ) -> None:
        self._segment_cache_bytes = segment_cache_bytes
        self._read_concurrency = read_concurrency
        self._read_retry_attempts = read_retry_attempts
        self._read_retry_initial_backoff = read_retry_initial_backoff
        self._read_retry_max_backoff = read_retry_max_backoff

        # Shards are opened again for each batch. Their footers are kept, keyed
        # by location, so that only the first open reads one. A cached file is
        # hash-validated, so a download after eviction has the same footer.
        # A failed open drops the footer, so that a retry does not reuse a bad
        # one.
        self._footers: dict[tuple[str, str], Footer] = {}

    def __getstate__(self) -> dict[str, Any]:
        # Vortex footers are not picklable, and a new process reads its own.
        state = self.__dict__.copy()
        state["_footers"] = {}
        return state

    def opener(self, options: "StoreOptions") -> "VortexFormat":
        """Return an opener that applies the Vortex options of ``options``."""
        return VortexFormat(
            segment_cache_bytes=options.vortex_segment_cache_bytes,
            read_concurrency=options.vortex_read_concurrency,
            read_retry_attempts=options.cache.open_retry_attempts,
            read_retry_initial_backoff=options.cache.open_retry_initial_backoff,
            read_retry_max_backoff=options.cache.open_retry_max_backoff,
        )

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
        """Scan directory and open each file to get metadata."""
        if _vortex is None:
            raise RuntimeError(
                "Discovering Vortex datasets requires the 'vortex-data' package. "
                + 'Install with: pip install "zephon[vortex]"'
            )

        require_read_at()
        entries = [
            name for name in storage.listdir(path) if name.endswith(VORTEX_SUFFIXES)
        ]
        if not entries:
            raise ValueError(f"No .vortex shards found under {path}")
        warn_missing_index(path, self.kind, num_shards=len(entries))

        shard_index: dict[int, int] = {}
        shard_meta: dict[int, dict[str, Any]] = {}

        for shard_id, name in enumerate(sorted(entries)):
            full = os.path.join(path, name)
            stats = storage.stat(full)
            size = int(stats.get("size", 0))

            try:
                count = _row_count(storage, full, size)
            except Exception as exc:
                raise ValueError(f"Failed to read Vortex shard {full}: {exc}") from exc

            shard_index[shard_id] = count
            shard_meta[shard_id] = {
                "raw": {
                    "basename": name,
                    "bytes": size,
                    "hashes": {},
                },
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
        """Open a local Vortex shard natively."""
        return self._open(locator, local_ref.raw.path, _length(local_ref.extra))

    def open_remote_shard(
        self, locator: ShardLocator, remote_ref: RemoteShardRef
    ) -> RandomAccessShard:
        """Open a remote Vortex shard that is read in place through its storage."""
        length = _length(remote_ref.extra)

        native = _native_store(remote_ref.storage, remote_ref.path)
        if native is not None:
            store, key = native
            return self._open(locator, key, length, store=store)

        source = StorageReadAt(
            remote_ref.storage,
            remote_ref.path,
            remote_ref.bytes,
            retry_attempts=self._read_retry_attempts,
            retry_initial_backoff=self._read_retry_initial_backoff,
            retry_max_backoff=self._read_retry_max_backoff,
        )
        return self._open(locator, remote_ref.path, length, source=source)

    def _open(
        self,
        locator: ShardLocator,
        path: Path | str,
        length: int | None,
        *,
        store: Any = None,
        source: StorageReadAt | None = None,
    ) -> VortexShard:
        # The file lives for one batch, so only a shared cache is worth filling.
        segment_cache = process_segment_cache(self._segment_cache_bytes)
        key = (locator.root, locator.raw.basename)

        try:
            shard = VortexShard(
                path,
                length=length,
                store=store,
                source=source,
                concurrency=self._read_concurrency,
                footer=self._footers.get(key),
                segment_cache=segment_cache,
                cache_key=segment_cache_key(locator) if segment_cache else None,
                without_segment_cache=segment_cache is None,
            )
        except Exception:
            self._footers.pop(key, None)
            raise

        self._footers[key] = shard.footer
        return shard


def _row_count(storage: StorageBackend, path: str, size: int) -> int:
    """Read the row count of a Vortex file, natively when it can be."""
    # Only the row count is read, so a segment cache has nothing to reuse.
    if not is_remote_path(path):
        return len(_vortex.open(path, without_segment_cache=True))

    native = _native_store(storage, path)
    if native is not None:
        store, key = native
        return len(_vortex.open(key, store=store, without_segment_cache=True))

    source = StorageReadAt(storage, path, size)
    return len(_vortex.open_readable(source, without_segment_cache=True))


def _length(extra: Mapping[str, object] | None) -> int | None:
    """Return the row count that discovery recorded, if it is valid."""
    if not extra or "length" not in extra:
        return None

    value = extra["length"]
    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None

    return None


register_format(VortexFormat())

__all__ = ["VortexFormat", "VortexShard"]
