# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""ArrayRecord shard discovery and indexed record reads.

Each ``.array_record`` file is one shard. Records are opaque byte strings, so
every sample is ``{"data": <bytes>}``; decode it (``tf.train.Example``,
msgpack, pickle, ...) with ``Pipeline.map_transform``.
"""

import importlib
import os
import posixpath
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from zephon._internal.io.formats.base import FormatHandler, register_format
from zephon._internal.io.index import find_and_load_index, warn_missing_index
from zephon._internal.io.index.index_types import ShardIndex, is_shard_index
from zephon._internal.io.protocols import RandomAccessShard
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.suffixes import ARRAY_RECORD_SUFFIXES
from zephon._internal.io.types import LocalShardRef, ShardFile, ShardLocator
from zephon.types import SamplePayload

if TYPE_CHECKING:
    from zephon.io.dataset import Dataset

# Payload key holding each record's bytes.
RECORD_FIELD = "data"

# Zephon reads scattered indices, which read-ahead (sized for sequential
# scans) only slows down.
_READER_OPTIONS = "readahead_buffer_size:0"


def _require_array_record() -> ModuleType:
    """Return the native ArrayRecord module, or explain how to install it."""
    try:
        # A native extension without type stubs.
        return importlib.import_module("array_record.python.array_record_module")
    except ImportError as exc:
        raise ImportError(
            "ArrayRecord support requires array-record (Python 3.11+); "
            + 'install with: pip install "zephon[array_record]"'
        ) from exc


def _open_reader(path: str) -> Any:
    """Open a reader, raising instead of returning an empty reader on failure.

    The native reader does not raise for a missing or corrupt file: it reports
    ``ok() == False`` and zero records, which would read as an empty shard.
    """
    module = _require_array_record()
    reader = module.ArrayRecordReader(path, _READER_OPTIONS)
    if reader.ok():
        return reader
    # Closing a failed reader raises its stored status, which says why.
    detail = ""
    try:
        reader.close()
    except RuntimeError as exc:
        detail = f": {exc}"
    # Surface filesystem errors as OSError so the shard store can retry them.
    os.stat(path)
    raise ValueError(f"Not a readable ArrayRecord file: {path}{detail}")


def _local_row_count(path: str) -> int:
    reader = _open_reader(path)
    try:
        return int(reader.num_records())
    finally:
        reader.close()


def _read_array_record_row_count(path: str, storage: StorageBackend) -> int:
    """Return the record count of the shard at ``path`` on ``storage``.

    The native reader only opens local files, so a remote shard is downloaded
    to a temporary directory first. Build an ``index.json`` to avoid this.
    """
    if "://" not in path:
        return _local_row_count(path)
    with tempfile.TemporaryDirectory(prefix="zephon-array-record-") as tmp:
        local = os.path.join(tmp, posixpath.basename(path))
        storage.download(path, local)
        return _local_row_count(local)


class ArrayRecordShard(RandomAccessShard):
    """Random access shard backed by one ArrayRecord file."""

    def __init__(self, path: Path, *, length: int | None = None) -> None:
        self._reader: Any = _open_reader(str(path))
        self._length = length if length is not None else self._reader.num_records()

    def __getitem__(self, index: int) -> SamplePayload:
        return self.getsamples([index])[0]

    def __len__(self) -> int:
        return self._length

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def getsamples(self, indices: list[int]) -> list[SamplePayload]:
        """Read records in request order, preserving duplicate indices."""
        if not indices:
            return []
        for i in indices:
            if i < 0 or i >= self._length:
                raise IndexError(i)
        if self._reader is None:
            raise RuntimeError("ArrayRecordShard has been closed")
        # The native read accepts unordered and repeated indices directly and
        # releases the GIL while it reads.
        return [{RECORD_FIELD: record} for record in self._reader.read(indices)]


def _shard_length(local_ref: LocalShardRef) -> int | None:
    length = (local_ref.extra or {}).get("length")
    if isinstance(length, (int, float, str)):
        try:
            return int(length)
        except ValueError:
            return None
    return None


class ArrayRecordFormat(FormatHandler):
    """Format handler for directories of ``.array_record`` shards."""

    kind = "array_record"

    def discover(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, Any]]]:
        """Return shard counts and metadata, from ``index.json`` when present.

        Without an index, every shard is opened to read its record count, and a
        remote shard is downloaded in full to do so. Build an index with
        ``zephon index array_record <path>`` for large or remote datasets.
        """
        result = find_and_load_index(path, storage)
        if result is not None and is_shard_index(result):
            return self._discover_from_index_data(result)
        return self._discover_by_scanning(path, storage)

    def _discover_from_index_data(
        self, index: ShardIndex
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, Any]]]:
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
        _require_array_record()
        entries = sorted(
            name
            for name in storage.listdir(path)
            if name.endswith(ARRAY_RECORD_SUFFIXES)
        )
        if not entries:
            raise ValueError(f"No .array_record shards found under {path}")
        warn_missing_index(path, self.kind, num_shards=len(entries))

        def read_metadata(name: str) -> tuple[int, int]:
            full = posixpath.join(path, name)
            try:
                size = int(storage.stat(full)["size"])
                return size, _read_array_record_row_count(full, storage)
            except Exception as exc:
                raise ValueError(
                    f"Failed to read ArrayRecord shard {full}: {exc}"
                ) from exc

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
        """Read per-shard counts from ``index.json`` without opening shards.

        Falls back to the full ``discover`` when there is no index or it has no
        shards; either way the result agrees with ``discover``.
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
            raise ValueError("ArrayRecord dataset missing 'path' in backend metadata")
        shards = backend.get("shards")
        if not isinstance(shards, Mapping):
            raise ValueError("ArrayRecord dataset missing 'shards' metadata")

        locators: dict[int, ShardLocator] = {}
        for shard_id_obj, shard_meta in shards.items():
            shard_id = int(shard_id_obj)
            raw_meta = (
                shard_meta.get("raw") if isinstance(shard_meta, Mapping) else None
            )
            if not isinstance(raw_meta, Mapping):
                raise ValueError(f"Shard {shard_id} missing ArrayRecord raw metadata")
            basename = raw_meta.get("basename")
            if not isinstance(basename, str):
                raise ValueError(f"Shard {shard_id} missing basename")
            bytes_value = raw_meta.get("bytes")
            if not isinstance(bytes_value, (int, float, str)):
                raise ValueError(f"Shard {shard_id} missing or invalid byte size")
            extra = shard_meta.get("extra")
            locators[shard_id] = ShardLocator(
                dataset=dataset.name,
                shard_id=shard_id,
                format=self.kind,
                root=path,
                raw=ShardFile(
                    basename=basename,
                    bytes=int(bytes_value),
                    hashes={
                        str(k): str(v)
                        for k, v in dict(raw_meta.get("hashes", {})).items()
                    },
                ),
                zip=None,
                compression=None,
                extra=extra if isinstance(extra, Mapping) else None,
            )
        return locators

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard:
        """Open an ArrayRecord shard for random access reading."""
        return ArrayRecordShard(local_ref.raw.path, length=_shard_length(local_ref))


register_format(ArrayRecordFormat())

__all__ = ["RECORD_FIELD", "ArrayRecordFormat", "ArrayRecordShard"]
