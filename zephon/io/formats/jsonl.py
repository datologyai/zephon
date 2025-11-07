# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""JSONL shard format support."""

import json
import os
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from zephon.io.formats.base import FormatHandler, register_format
from zephon.io.protocols import RandomAccessShard
from zephon.io.storage import StorageBackend
from zephon.io.types import LocalShardRef, ShardFile, ShardLocator

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
                stripped = line.strip()
                if not stripped:
                    continue
                if target == 0:
                    # TODO: Build and use a byte-offset index so we can seek directly.
                    return json.loads(stripped)
                target -= 1
        raise IndexError(index)

    def __len__(self) -> int:
        if self._length is not None:
            return self._length
        count = 0
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        self._length = count
        return count

    def close(self) -> None:
        return None

    def getsamples(self, indices: list[int]) -> list[dict[str, object]]:
        if not indices:
            return []
        # Validate and prepare scatter targets for duplicates and arbitrary order.
        for i in indices:
            if i < 0:
                raise IndexError(i)
        out: list[dict[str, object] | None] = [None] * len(indices)
        waiting: dict[int, list[int]] = {}
        for pos, idx in enumerate(indices):
            waiting.setdefault(int(idx), []).append(pos)

        # Single pass over file collecting requested rows.
        with self._path.open("r", encoding="utf-8") as handle:
            for lnum, line in enumerate(handle):
                if not waiting:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                if lnum in waiting:
                    obj = json.loads(stripped)
                    for pos in waiting[lnum]:
                        out[pos] = obj
                    del waiting[lnum]

        if waiting:
            # Some indices were out of range; report the smallest missing one.
            missing = min(waiting.keys())
            raise IndexError(missing)
        rows = [x for x in out if x is not None]
        return rows


class JsonlFormat(FormatHandler):
    """Format handler for JSON Lines datasets."""

    kind = "jsonl"

    def discover(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, object]]]:
        """Scan ``path`` and return shard statistics and metadata."""
        entries = [name for name in storage.listdir(path) if name.endswith(".jsonl")]
        if not entries:
            raise ValueError(f"No .jsonl shards found under {path}")

        shard_index: dict[int, int] = {}
        shard_meta: dict[int, dict[str, object]] = {}

        for shard_id, name in enumerate(entries):
            full = os.path.join(path, name)
            stats = storage.stat(full)
            size = int(stats.get("size", 0))
            # TODO(MaxiBoether): Counting lines by opening every shard is expensive on
            # remote/cloud storage. Consider storing counts in metadata or lazily
            # computing lengths during shard open.
            count = 0
            with storage.open(full, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        count += 1
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

    def build_locators(self, dataset: "Dataset") -> Mapping[int, ShardLocator]:
        backend = dataset.backend
        path = backend.get("path")
        if not isinstance(path, str):
            raise ValueError("JSONL dataset missing 'path' in backend metadata")
        shards = backend.get("shards")
        if not isinstance(shards, Mapping):
            raise ValueError("JSONL dataset missing 'shards' metadata")

        locators: dict[int, ShardLocator] = {}
        for shard_id_obj, shard_meta in shards.items():
            shard_id = int(shard_id_obj)
            if not isinstance(shard_meta, Mapping):
                raise ValueError(f"Invalid shard metadata for shard {shard_id}")
            raw_meta = shard_meta.get("raw")
            if not isinstance(raw_meta, Mapping):
                raise ValueError(f"Shard {shard_id} missing JSONL raw metadata")
            basename = raw_meta.get("basename")
            if not isinstance(basename, str):
                raise ValueError(f"Shard {shard_id} missing basename for JSONL shard")
            bytes_value = raw_meta.get("bytes")
            if isinstance(bytes_value, Integral):
                raw_bytes = int(bytes_value)
            elif isinstance(bytes_value, str):
                raw_bytes = int(bytes_value)
            else:
                raise ValueError(
                    f"Shard {shard_id} missing or invalid byte size for JSONL shard"
                )
            raw = ShardFile(
                basename=basename,
                bytes=raw_bytes,
                hashes={
                    str(k): str(v) for k, v in dict(raw_meta.get("hashes", {})).items()
                },
            )
            zip_meta = shard_meta.get("zip")
            zip_file = None
            if isinstance(zip_meta, Mapping):
                zip_basename = zip_meta.get("basename")
                if not isinstance(zip_basename, str):
                    raise ValueError(
                        f"Shard {shard_id} missing basename for compressed JSONL shard"
                    )
                zip_bytes = zip_meta.get("bytes")
                if isinstance(zip_bytes, Integral):
                    compressed_bytes = int(zip_bytes)
                elif isinstance(zip_bytes, str):
                    compressed_bytes = int(zip_bytes)
                else:
                    raise ValueError(
                        f"Shard {shard_id} missing or invalid byte size for compressed JSONL shard"
                    )
                zip_file = ShardFile(
                    basename=zip_basename,
                    bytes=compressed_bytes,
                    hashes={
                        str(k): str(v)
                        for k, v in dict(zip_meta.get("hashes", {})).items()
                    },
                )
            compression = shard_meta.get("compression")
            extra = shard_meta.get("extra")
            locators[shard_id] = ShardLocator(
                dataset=dataset.name,
                shard_id=shard_id,
                format=self.kind,
                root=path,
                raw=raw,
                zip=zip_file,
                compression=str(compression) if compression else None,
                extra=extra if isinstance(extra, Mapping) else None,
            )
        return locators

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
