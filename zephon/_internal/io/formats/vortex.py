# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex shard discovery and indexed sample reads."""

import posixpath
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping

import numpy as np

from zephon._internal.io.formats.arrow_rows import (
    arrow_table_to_samples,
    require_pyarrow,
    samples_from_columns,
)
from zephon._internal.io.formats.base import FormatHandler, ShardOpener, register_format
from zephon._internal.io.formats.metadata_cache import MetadataCache, file_cache_key
from zephon._internal.io.index import find_and_load_index, warn_missing_index
from zephon._internal.io.index.index_types import ShardIndex, is_shard_index
from zephon._internal.io.protocols import RandomAccessShard
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.suffixes import VORTEX_SUFFIXES
from zephon._internal.io.types import LocalShardRef, ShardFile, ShardLocator
from zephon._internal.utils.thread_utils import cap_vortex_threads
from zephon.types import SamplePayload

if TYPE_CHECKING:
    from zephon._internal.io.catalog import CatalogSet
    from zephon.io.dataset import Dataset
    from zephon.io.options import StoreOptions

try:
    import vortex as _vortex
except ImportError:
    _vortex = None
else:
    cap_vortex_threads()


def _require_vortex() -> ModuleType:
    """Return Vortex with custom-reader support or explain how to upgrade."""
    if _vortex is None or not hasattr(_vortex, "open_readable"):
        raise ImportError(
            "Vortex support requires vortex-data>=0.88.0; "
            + 'install or upgrade with: pip install -U "zephon[vortex]"'
        )
    return _vortex


@dataclass(slots=True)
class _VortexReader:
    """Adapt an existing backend to Vortex's ``ReadBytesAt`` protocol.

    The backend owns path resolution and authentication. Vortex can call
    ``read_at`` concurrently, so its ``read_range`` must be safe across threads.
    """

    storage: StorageBackend
    path: str
    file_size: int
    error: Exception | None = field(default=None, init=False)

    def size(self) -> int:
        """Return the size already obtained during discovery."""
        return self.file_size

    def read_at(self, offset: int, length: int) -> bytes | memoryview:
        """Read a positional range without sharing a mutable seek position."""
        # Some backends (including obstore) reject empty range requests.
        if length == 0:
            return b""
        try:
            return self.storage.read_range(self.path, offset, length=length)
        except Exception as exc:
            # Vortex's native bridge otherwise retains only the error's text.
            self.error = exc
            raise


def _read_vortex_row_count(path: str, storage: StorageBackend, file_size: int) -> int:
    """Read the row count from Vortex metadata through the supplied backend."""
    vortex = _require_vortex()
    reader = _VortexReader(storage=storage, path=path, file_size=file_size)
    try:
        return len(vortex.open_readable(reader, without_segment_cache=True))
    except Exception:
        if reader.error is not None:
            raise reader.error
        raise


def _open_vortex_file(
    path: Path,
    metadata_cache: MetadataCache[str, Any] | None,
    segment_cache: Any | None,
) -> Any:
    """Open a reader using the store's footer and segment caches."""
    vortex = _require_vortex()
    if metadata_cache is None and segment_cache is None:
        return vortex.open(str(path), without_segment_cache=True)
    key = file_cache_key(path)
    options: dict[str, Any] = (
        {"segment_cache": segment_cache, "cache_key": key}
        if segment_cache is not None
        else {"without_segment_cache": True}
    )
    if metadata_cache is None:
        return vortex.open(str(path), **options)

    file = None

    def load_footer() -> Any:
        nonlocal file
        file = vortex.open(str(path), **options)
        return file.footer

    footer = metadata_cache.get_or_load(key, load_footer)
    # Reuse the reader opened on a footer-cache miss.
    return (
        file if file is not None else vortex.open(str(path), footer=footer, **options)
    )


@contextmanager
def _local_io_errors(path: Path) -> Iterator[None]:
    """Expose filesystem errors for missing or inaccessible shards."""
    try:
        yield
    except RuntimeError as exc:
        # Vortex wraps IO errors in RuntimeError; only path failures are retryable.
        try:
            path.stat()
        except OSError as io_error:
            raise io_error from exc
        raise


class VortexShard(RandomAccessShard):
    """Random access shard backed by a Vortex file."""

    def __init__(
        self,
        path: Path,
        *,
        length: int | None = None,
        metadata_cache: MetadataCache[str, Any] | None = None,
        segment_cache: Any | None = None,
    ) -> None:
        self._path = path
        with _local_io_errors(path):
            self._file = _open_vortex_file(path, metadata_cache, segment_cache)
        self._length = length if length is not None else len(self._file)
        self._export_type: Any | None = None
        self._text_only = False

    def __getitem__(self, index: int) -> "SamplePayload":
        return self.getsamples([index])[0]

    def __len__(self) -> int:
        return self._length

    def close(self) -> None:
        self._file = None
        self._export_type = None

    def getsamples(self, indices: list[int]) -> list["SamplePayload"]:
        """Bulk read rows in request order, preserving duplicate indices."""
        if not indices:
            return []
        for i in indices:
            if i < 0 or i >= self._length:
                raise IndexError(i)
        if self._file is None:
            raise RuntimeError("VortexShard has been closed")

        # Vortex scans require strictly increasing indices, even when callers
        # request rows out of order or more than once.
        positions: Iterable[int]
        if all(a < b for a, b in zip(indices, indices[1:])):
            sorted_unique = indices
            positions = range(len(indices))
        else:
            sorted_unique = sorted(set(indices))
            index_to_pos = {idx: pos for pos, idx in enumerate(sorted_unique)}
            positions = map(index_to_pos.__getitem__, indices)

        # Native ranges avoid converting Python index lists through Arrow.
        contiguous = sorted_unique[-1] - sorted_unique[0] + 1 == len(sorted_unique)
        selected = (
            range(sorted_unique[0], sorted_unique[-1] + 1)
            if contiguous
            else sorted_unique
        )
        with _local_io_errors(self._path):
            batch = self._file.scan(
                indices=_require_vortex().array(selected)
            ).read_all()
            pa = require_pyarrow()
            if self._export_type is None:
                export_type = _list_view_type(self._file.dtype.to_arrow_type(), pa)
                self._export_type = export_type
                self._text_only = all(
                    pa.types.is_string(field.type)
                    or pa.types.is_large_string(field.type)
                    or pa.types.is_string_view(field.type)
                    or pa.types.is_binary(field.type)
                    or pa.types.is_large_binary(field.type)
                    or pa.types.is_binary_view(field.type)
                    for field in export_type
                )
            if self._text_only and len(sorted_unique) == 1:
                # Scalar access avoids exporting a whole string dictionary.
                row = batch.scalar_at(0).as_py()
                if row is not None:
                    return samples_from_columns(
                        {name: [value] for name, value in row.items()}, positions
                    )
                # Null structs need Arrow's per-field null conversion.
            # ListView export can trim unselected child values before decoding.
            arrow_table = pa.Table.from_struct_array(
                batch.to_arrow_array(arrow_type=self._export_type)
            )
        return arrow_table_to_samples(arrow_table, positions)


def _list_view_type(kind: Any, pa: Any) -> Any:
    """Use ListView for variable-length lists, preserving fields and fixed shapes."""
    if pa.types.is_struct(kind):
        return pa.struct(
            [field.with_type(_list_view_type(field.type, pa)) for field in kind]
        )
    if pa.types.is_list(kind) or pa.types.is_large_list(kind):
        child = kind.value_field.with_type(_list_view_type(kind.value_type, pa))
        return (
            pa.list_view(child) if pa.types.is_list(kind) else pa.large_list_view(child)
        )
    if pa.types.is_fixed_size_list(kind):
        return pa.list_(
            kind.value_field.with_type(_list_view_type(kind.value_type, pa)),
            kind.list_size,
        )
    return kind


def _shard_length(local_ref: LocalShardRef) -> int | None:
    if local_ref.extra and "length" in local_ref.extra:
        length_value = local_ref.extra["length"]
        if isinstance(length_value, (int, float)):
            return int(length_value)
        if isinstance(length_value, str):
            try:
                return int(length_value)
            except ValueError:
                pass
    return None


class VortexShardOpener:
    """Share footer and segment caches across a store's Vortex readers."""

    def __init__(self, options: "StoreOptions") -> None:
        vortex = _require_vortex()
        self._metadata_cache = (
            MetadataCache[str, Any](options.cache.metadata_cache_entries)
            if options.cache.metadata_cache_entries
            else None
        )
        self._segment_cache = (
            vortex.SegmentCache(options.vortex.segment_cache_bytes)
            if options.vortex.segment_cache_bytes
            else None
        )

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard:
        """Open a shard with this store's caches."""
        return VortexShard(
            local_ref.raw.path,
            length=_shard_length(local_ref),
            metadata_cache=self._metadata_cache,
            segment_cache=self._segment_cache,
        )

    def close(self) -> None:
        """Clear the store's Vortex caches."""
        if self._metadata_cache is not None:
            self._metadata_cache.clear()
        if self._segment_cache is not None:
            self._segment_cache.clear()


class VortexFormat(FormatHandler):
    """Format handler for Vortex datasets."""

    kind = "vortex"

    @contextmanager
    def create_opener(
        self, catalog_set: "CatalogSet | None", options: "StoreOptions"
    ) -> Iterator[ShardOpener]:
        """Provide a store-scoped opener and clear its caches on exit."""
        opener = VortexShardOpener(options)
        try:
            yield opener
        finally:
            opener.close()

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
            raise ValueError("index.json contains no shards")

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
        _require_vortex()

        entries = sorted(
            name for name in storage.listdir(path) if name.endswith(VORTEX_SUFFIXES)
        )
        if not entries:
            raise ValueError(f"No .vortex shards found under {path}")
        warn_missing_index(path, self.kind, num_shards=len(entries))

        def read_metadata(name: str) -> tuple[int, int]:
            full = posixpath.join(path, name)
            try:
                size = int(storage.stat(full)["size"])
                return size, _read_vortex_row_count(full, storage, size)
            except Exception as exc:
                raise ValueError(f"Failed to read Vortex shard {full}: {exc}") from exc

        shard_index: dict[int, int] = {}
        shard_meta: dict[int, dict[str, Any]] = {}
        max_workers = min(32, (len(entries) + 4) // 5)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map keeps filename order even when later reads finish first.
            for shard_id, (name, (size, count)) in enumerate(
                zip(entries, executor.map(read_metadata, entries))
            ):
                shard_index[shard_id] = count
                shard_meta[shard_id] = {
                    "raw": {"basename": name, "bytes": size, "hashes": {}},
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
        return VortexShard(local_ref.raw.path, length=_shard_length(local_ref))


register_format(VortexFormat())

__all__ = ["VortexFormat", "VortexShard"]
