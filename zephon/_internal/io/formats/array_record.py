# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""ArrayRecord metadata discovery and native sample reads.

Each file is one shard. Records remain opaque bytes under the ``data`` key;
decode them with ``Pipeline.map_transform``.
"""

import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from zephon._internal.io.formats.array_record_support.metadata import read_row_count
from zephon._internal.io.formats.base import register_format
from zephon._internal.io.formats.single_file import SingleFileFormat, shard_length
from zephon._internal.io.protocols import RandomAccessShard
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.suffixes import ARRAY_RECORD_SUFFIXES
from zephon._internal.io.types import LocalShardRef, ShardLocator
from zephon.types import SamplePayload

RECORD_FIELD = "data"
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


def _read_array_record_row_count(
    path: str, storage: StorageBackend, size: int | None = None
) -> int:
    """Read postscript/footer ranges using the backend's path semantics."""
    if size is None:
        size = int(storage.stat(path)["size"])
    return read_row_count(path, storage, size)


class ArrayRecordShard(RandomAccessShard):
    """Random access through a short-lived native ArrayRecord reader."""

    def __init__(self, path: Path, *, length: int | None = None) -> None:
        self._reader: Any = _open_reader(str(path))
        self._length = length if length is not None else self._reader.num_records()

    def __getitem__(self, index: int) -> SamplePayload:
        return self.getsamples([index])[0]

    def __len__(self) -> int:
        return self._length

    def close(self) -> None:
        if self._reader is not None:
            reader, self._reader = self._reader, None
            reader.close()

    def getsamples(self, indices: list[int]) -> list[SamplePayload]:
        """Read in request order, retaining duplicates and releasing the GIL."""
        if not indices:
            return []
        for index in indices:
            if index < 0 or index >= self._length:
                raise IndexError(index)
        if self._reader is None:
            raise RuntimeError("ArrayRecordShard has been closed")
        # read(begin, end) skips the indexed API's all-chunk bookkeeping.
        # Its exclusive end must fit in int32; read(indices) accepts uint64.
        if len(indices) == 1 and indices[0] + 1 <= _MAX_RANGE_END:
            records = self._reader.read(indices[0], indices[0] + 1)
        else:
            records = self._reader.read(indices)
        return [{RECORD_FIELD: record} for record in records]


class ArrayRecordFormat(SingleFileFormat):
    """Directories of ArrayRecord shards with optional Zephon indexes."""

    kind = "array_record"
    label = "ArrayRecord"
    suffixes = ARRAY_RECORD_SUFFIXES

    def _read_row_count(self, path: str, storage: StorageBackend, size: int) -> int:
        return _read_array_record_row_count(path, storage, size)

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard:
        """Open a local shard with the native reader."""
        return ArrayRecordShard(local_ref.raw.path, length=shard_length(local_ref))


register_format(ArrayRecordFormat())

__all__ = ["RECORD_FIELD", "ArrayRecordFormat", "ArrayRecordShard"]
