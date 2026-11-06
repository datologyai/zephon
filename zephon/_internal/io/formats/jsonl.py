# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""JSONL shard format support."""

import io
import json
import os
from collections import defaultdict
from collections.abc import Iterator
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Mapping, cast

import numpy as np

from zephon._internal.io.formats.base import FormatHandler, register_format
from zephon._internal.io.formats.locators import build_file_locators
from zephon._internal.io.index import find_and_load_index, warn_missing_index
from zephon._internal.io.index.index_types import ShardIndex, is_shard_index
from zephon._internal.io.index.shard_index import (
    is_nonnegative_int64,
    shard_index_counts,
    validate_shard_index,
)
from zephon._internal.io.protocols import RandomAccessShard
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.suffixes import JSONL_SUFFIXES
from zephon._internal.io.types import LocalShardRef, ShardLocator
from zephon._internal.utils.compression import compression_for_name, open_decompressed

if TYPE_CHECKING:
    from zephon.io.dataset import Dataset


class JsonlShard(RandomAccessShard):
    """Random access shard backed by a JSONL file."""

    def __init__(self, path: Path, *, length: int | None = None) -> None:
        self._path = path
        self._length = length

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            raise IndexError(index)
        if self._length is not None and index >= self._length:
            raise IndexError(index)

        target = index
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.isspace():
                    continue
                if target == 0:
                    # TODO: Build and use a byte-offset index so we can seek directly.
                    return json.loads(line.strip())
                target -= 1
        raise IndexError(index)

    def __len__(self) -> int:
        if self._length is not None:
            return self._length
        with self._path.open("r", encoding="utf-8") as handle:
            self._length = sum(1 for line in handle if not line.isspace())
        return self._length

    def close(self) -> None:
        return None

    def getsamples(self, indices: list[int]) -> list[dict[str, object]]:
        if not indices:
            return []
        # Validate and prepare scatter targets for duplicates and arbitrary order.
        for i in indices:
            if i < 0 or (self._length is not None and i >= self._length):
                raise IndexError(i)
        out: list[dict[str, object] | None] = [None] * len(indices)
        waiting: defaultdict[int, list[int]] = defaultdict(list)
        for pos, idx in enumerate(indices):
            waiting[int(idx)].append(pos)

        # Single pass over file collecting requested rows.
        logical_index = 0
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.isspace():
                    continue
                positions = waiting.pop(logical_index, None)
                if positions is not None:
                    obj = json.loads(line.strip())
                    for pos in positions:
                        out[pos] = obj
                    if not waiting:
                        break
                logical_index += 1

        if waiting:
            # Some indices were out of range; report the smallest missing one.
            missing = min(waiting.keys())
            raise IndexError(missing)
        # Every slot is filled here; a JSON ``null`` row is a legit None value.
        return cast(list[dict[str, object]], out)


# Suffix of the decoded copy of a compressed shard. The local resolver writes it
# next to the data, so it must not be a JSONL suffix that discovery would list.
_DECODED_SUFFIX = ".raw"


def _count_records(stream: BinaryIO) -> int:
    """Count the non-blank lines :class:`JsonlShard` indexes as records."""
    text = io.TextIOWrapper(stream, encoding="utf-8")
    try:
        return sum(1 for line in text if not line.isspace())
    finally:
        text.detach()  # leave ``stream`` open for the caller


class JsonlFormat(FormatHandler):
    """Format handler for JSON Lines datasets, optionally compressed per shard."""

    kind = "jsonl"

    def discover(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, object]]]:
        """Return indexed shard metadata, falling back to scanning files."""
        result = find_and_load_index(path, storage)
        if is_shard_index(result) and self._is_valid_jsonl_index(result):
            return self._discover_from_index_data(result)

        return self._discover_from_files(path, storage)

    def _is_valid_jsonl_index(self, data: ShardIndex) -> bool:
        """Check JSONL suffixes and decoded sizes in a Zephon shard index."""
        try:
            validate_shard_index(data)
        except ValueError:
            return False
        for shard in data["shards"]:
            name = shard["basename"]
            if not name.endswith(JSONL_SUFFIXES):
                return False
            if compression_for_name(name) is not None:
                if not is_nonnegative_int64(shard.get("extra", {}).get("raw_bytes")):
                    return False
        return True

    def _discover_from_index_data(
        self, data: ShardIndex
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, object]]]:
        """Build discovery metadata without opening the indexed shards."""
        shard_index: dict[int, int] = {}
        shard_meta: dict[int, dict[str, object]] = {}

        for shard_id, shard in enumerate(data["shards"]):
            count = shard["num_rows"]
            shard_index[shard_id] = count
            name = shard["basename"]
            compression = compression_for_name(name)
            file_meta = {
                "basename": name,
                "bytes": shard["bytes"],
                "hashes": shard.get("hashes", {}),
            }
            meta: dict[str, object] = {
                "raw": file_meta,
                "extra": {"length": count},
            }
            if compression is not None:
                meta["raw"] = {
                    "basename": name + _DECODED_SUFFIX,
                    "bytes": shard["extra"]["raw_bytes"],
                    "hashes": {},
                }
                meta["zip"] = file_meta
                meta["compression"] = compression
            shard_meta[shard_id] = meta

        return shard_index, shard_meta

    def _discover_from_files(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, object]]]:
        """Scan JSONL files and retain their counts and locator metadata."""
        shard_index: dict[int, int] = {}
        shard_meta: dict[int, dict[str, object]] = {}
        for shard_id, (count, meta) in enumerate(self._scan_files(path, storage)):
            shard_index[shard_id] = count
            shard_meta[shard_id] = meta
        return shard_index, shard_meta

    def _scan_files(
        self, path: str, storage: StorageBackend
    ) -> Iterator[tuple[int, dict[str, object]]]:
        """Yield each shard's count and metadata without retaining prior shards."""
        # Sort so shard_id assignment is reproducible across processes/machines
        # (cache JOIN, cross-node Ray). Re-numbers existing jsonl datasets once.
        entries = sorted(
            name for name in storage.listdir(path) if name.endswith(JSONL_SUFFIXES)
        )
        if not entries:
            raise ValueError(
                f"No JSONL shards ({', '.join(JSONL_SUFFIXES)}) found under {path}"
            )
        warn_missing_index(path, self.kind, num_shards=len(entries))

        # Without a usable index, exact row counts require reading every shard.
        # Build index.json to avoid this scan on subsequent discoveries.
        for name in entries:
            full = os.path.join(path, name)
            stats = storage.stat(full)
            size = int(stats.get("size", 0))
            compression = compression_for_name(name)
            with storage.open(full, "rb") as handle:
                if compression is None:
                    count = _count_records(cast(BinaryIO, handle))
                    raw_bytes = size
                else:
                    with open_decompressed(cast(BinaryIO, handle), compression) as src:
                        count = _count_records(src)
                        raw_bytes = src.tell()
            meta: dict[str, object] = {
                "raw": {
                    "basename": name if compression is None else name + _DECODED_SUFFIX,
                    "bytes": raw_bytes,
                    "hashes": {},
                },
                "extra": {"length": count},
            }
            if compression is not None:
                # As for compressed MDS/LitData shards: resolvers decode the archive
                # once into the raw file that JsonlShard reads.
                meta["zip"] = {"basename": name, "bytes": size, "hashes": {}}
                meta["compression"] = compression
            yield count, meta

    def discover_counts(
        self, path: str, storage: StorageBackend
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read per-shard row counts directly from an index when available."""
        result = find_and_load_index(path, storage)
        if is_shard_index(result) and self._is_valid_jsonl_index(result):
            return shard_index_counts(result)
        counts = np.fromiter(
            (count for count, _ in self._scan_files(path, storage)),
            dtype=np.int64,
        )
        return np.arange(counts.size, dtype=np.int64), counts

    def build_locators(self, dataset: "Dataset") -> Mapping[int, ShardLocator]:
        """Build locators from normalized shard metadata."""
        return build_file_locators(dataset, format_kind=self.kind)

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard:
        length = None
        if local_ref.extra and "length" in local_ref.extra:
            length_value = local_ref.extra["length"]
            if isinstance(length_value, Integral):
                length = int(length_value)
            elif isinstance(length_value, str):
                try:
                    length = int(length_value)
                except ValueError:
                    length = None
        return JsonlShard(local_ref.raw.path, length=length)


register_format(JsonlFormat())

__all__ = ["JsonlFormat", "JsonlShard"]
