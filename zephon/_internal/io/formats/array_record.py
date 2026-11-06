# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""ArrayRecord metadata discovery and native sample reads.

Each file is one shard. Records are returned as bytes; decode them with
``Pipeline.map_transform`` using the serializer chosen when writing the data.
"""

import importlib
import os
import posixpath
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from zephon._internal.io.formats.array_record_support.metadata import read_row_count
from zephon._internal.io.formats.base import FormatHandler, register_format
from zephon._internal.io.formats.locators import build_file_locators, shard_length
from zephon._internal.io.index import find_and_load_index, warn_missing_index
from zephon._internal.io.index.index_types import (
    ShardIndex,
    ShardInfoDict,
    is_shard_index,
)
from zephon._internal.io.index.shard_index import (
    shard_index_counts,
    shard_index_metadata,
    validate_shard_index,
)
from zephon._internal.io.protocols import RandomAccessShard
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.storage.base import LocalPathBackend
from zephon._internal.io.suffixes import ARRAY_RECORD_SUFFIXES
from zephon._internal.io.types import LocalShardRef, ShardLocator

if TYPE_CHECKING:
    from zephon.io.dataset import Dataset

_READER_OPTIONS = "readahead_buffer_size:0"
# The Python binding declares read(begin, end) with signed int32 endpoints.
_MAX_RANGE_END = (1 << 31) - 1


def _require_array_record() -> ModuleType:
    """Load the optional native reader, with an actionable installation hint."""
    try:
        return importlib.import_module("array_record.python.array_record_module")
    except ImportError as exc:
        raise ImportError(
            'ArrayRecord support requires pip install "zephon[array_record]"'
        ) from exc


def _open_reader(path: str) -> Any:
    """Raise when the native reader opens a missing/corrupt file as empty."""
    detail = ""
    try:
        reader = _require_array_record().ArrayRecordReader(path, _READER_OPTIONS)
    except RuntimeError as exc:
        # Older releases raise during construction instead of returning a failed reader.
        detail = f": {exc}"
    else:
        if reader.ok():
            return reader
        try:
            reader.close()
        except RuntimeError as exc:
            detail = f": {exc}"
    # Filesystem failures remain retryable by the existing shard store.
    os.stat(path)
    raise ValueError(f"Not a readable ArrayRecord file: {path}{detail}")


def _read_array_record_row_count(path: str, storage: StorageBackend, size: int) -> int:
    """Use native local metadata reads, or bounded ranges through the backend."""
    local = storage.local_path(path) if isinstance(storage, LocalPathBackend) else None
    if local is not None:
        reader = _open_reader(str(local))
        try:
            return reader.num_records()
        finally:
            reader.close()
    return read_row_count(path, storage, size)


class ArrayRecordShard(RandomAccessShard):
    """Random access through a short-lived native ArrayRecord reader."""

    def __init__(self, path: Path, *, length: int | None = None) -> None:
        self._reader: Any = _open_reader(str(path))
        self._length = length if length is not None else self._reader.num_records()

    def __getitem__(self, index: int) -> bytes:
        return self.getsamples([index])[0]

    def __len__(self) -> int:
        return self._length

    def close(self) -> None:
        if self._reader is not None:
            reader, self._reader = self._reader, None
            reader.close()

    def getsamples(self, indices: list[int]) -> list[bytes]:
        """Read in request order, retaining duplicates and releasing the GIL."""
        if not indices:
            return []
        for index in indices:
            if index < 0 or index >= self._length:
                raise IndexError(index)
        if self._reader is None:
            raise RuntimeError("ArrayRecordShard has been closed")
        # Singletons and consecutive requests avoid indexed all-chunk bookkeeping.
        # The exclusive range end must fit in int32; indexed reads accept uint64.
        end = indices[-1] + 1
        if end <= _MAX_RANGE_END and all(
            a + 1 == b for a, b in zip(indices, indices[1:])
        ):
            return self._reader.read(indices[0], end)
        return self._reader.read(indices)


class ArrayRecordFormat(FormatHandler):
    """Directories of ArrayRecord shards with optional Zephon indexes."""

    kind = "array_record"

    def _load_index(self, path: str, storage: StorageBackend) -> ShardIndex | None:
        index = find_and_load_index(path, storage)
        if not is_shard_index(index):
            return None
        validate_shard_index(index)
        for i, shard in enumerate(index["shards"]):
            if not shard["basename"].endswith(ARRAY_RECORD_SUFFIXES):
                raise ValueError(f"Invalid ArrayRecord index basename for shard {i}")
            length = shard.get("extra", {}).get("length", shard["num_rows"])
            if type(length) is not int or length != shard["num_rows"]:
                raise ValueError(f"Inconsistent ArrayRecord index length for shard {i}")
        return index

    def _index(self, path: str, storage: StorageBackend) -> ShardIndex:
        index = self._load_index(path, storage)
        return self._scan_files(path, storage) if index is None else index

    def discover(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, Any]]]:
        """Read the index, or inspect each shard's metadata."""
        return shard_index_metadata(self._index(path, storage))

    def discover_counts(
        self, path: str, storage: StorageBackend
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read counts without constructing locator metadata."""
        return shard_index_counts(self._index(path, storage))

    def _scan_files(self, path: str, storage: StorageBackend) -> ShardIndex:
        entries = sorted(
            name
            for name in storage.listdir(path)
            if name.endswith(ARRAY_RECORD_SUFFIXES)
        )
        if not entries:
            raise ValueError(f"No .array_record shards found under {path}")
        warn_missing_index(path, self.kind, num_shards=len(entries))

        def read_metadata(name: str) -> ShardInfoDict:
            full = posixpath.join(path, name)
            try:
                size = int(storage.stat(full)["size"])
                count = _read_array_record_row_count(full, storage, size)
                return ShardInfoDict(
                    basename=name, bytes=size, num_rows=count, hashes={}, extra={}
                )
            except Exception as exc:
                raise ValueError(
                    f"Failed to read ArrayRecord shard {full}: {exc}"
                ) from exc

        with ThreadPoolExecutor(
            max_workers=min(32, (len(entries) + 4) // 5)
        ) as executor:
            shards = list(executor.map(read_metadata, entries))
        return ShardIndex(format_version=1, shards=shards)

    def build_locators(self, dataset: "Dataset") -> Mapping[int, ShardLocator]:
        """Build locators from discovered shard metadata."""
        return build_file_locators(dataset, format_kind=self.kind)

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard:
        """Open a local shard with the native reader."""
        return ArrayRecordShard(local_ref.raw.path, length=shard_length(local_ref))


register_format(ArrayRecordFormat())

__all__ = ["ArrayRecordFormat", "ArrayRecordShard"]
