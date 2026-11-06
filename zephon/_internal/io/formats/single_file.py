# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Discovery and locators for uncompressed, self-describing shard files."""

import posixpath
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from zephon._internal.io.formats.base import FormatHandler
from zephon._internal.io.index import find_and_load_index, warn_missing_index
from zephon._internal.io.index.index_types import (
    ShardIndex,
    ShardInfoDict,
    is_shard_index,
)
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.types import LocalShardRef, ShardFile, ShardLocator

if TYPE_CHECKING:
    from zephon.io.dataset import Dataset


def _nonnegative_int64(value: object) -> bool:
    return type(value) is int and 0 <= value < 1 << 63


def shard_length(local_ref: LocalShardRef) -> int | None:
    """Return the optional locator length hint, including legacy strings."""
    value = (local_ref.extra or {}).get("length")
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (ValueError, OverflowError):
            pass
    return None


class SingleFileFormat(FormatHandler):
    """Share discovery for formats whose files each hold one complete shard.

    Subclasses supply suffixes, a row-count reader and a native shard opener.
    Internal chunk compression is handled by the native format itself.
    """

    suffixes: tuple[str, ...]
    label: str

    @abstractmethod
    def _read_row_count(self, path: str, storage: StorageBackend, size: int) -> int:
        raise NotImplementedError

    def _load_index(self, path: str, storage: StorageBackend) -> ShardIndex | None:
        data = find_and_load_index(path, storage)
        if not is_shard_index(data):
            return None
        raw: Any = data
        if raw.get("format_version") != 1:
            raise ValueError(f"Unsupported {self.label} index version")
        shards = raw.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("index.json contains no shards")
        for i, shard in enumerate(shards):
            if not isinstance(shard, dict):
                raise ValueError(f"Invalid {self.label} index shard {i}")
            name = shard.get("basename")
            hashes, extra = shard.get("hashes", {}), shard.get("extra", {})
            if (
                not isinstance(name, str)
                or not name.endswith(self.suffixes)
                or not _nonnegative_int64(shard.get("num_rows"))
                or not _nonnegative_int64(shard.get("bytes"))
                or not isinstance(hashes, dict)
                or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in hashes.items()
                )
                or not isinstance(extra, dict)
            ):
                raise ValueError(f"Invalid {self.label} index metadata for shard {i}")
            if "length" in extra and (
                not _nonnegative_int64(extra["length"])
                or extra["length"] != shard["num_rows"]
            ):
                raise ValueError(
                    f"Inconsistent {self.label} index length for shard {i}"
                )
        return data

    def discover(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, Any]]]:
        """Read the index, or inspect each file's metadata through storage."""
        index = self._load_index(path, storage)
        if index is not None:
            return self._discover_from_index_data(index)
        return self._discover_by_scanning(path, storage)

    def _discover_from_index_data(
        self, index: ShardIndex
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, Any]]]:
        counts: dict[int, int] = {}
        metadata: dict[int, dict[str, Any]] = {}
        for i, shard in enumerate(index["shards"]):
            count = shard["num_rows"]
            counts[i] = count
            metadata[i] = {
                "raw": {
                    "basename": shard["basename"],
                    "bytes": shard["bytes"],
                    "hashes": shard.get("hashes", {}),
                },
                "extra": {**shard.get("extra", {}), "length": count},
            }
        return counts, metadata

    def _discover_by_scanning(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, Any]]]:
        entries = sorted(
            name for name in storage.listdir(path) if name.endswith(self.suffixes)
        )
        if not entries:
            raise ValueError(f"No {'/'.join(self.suffixes)} shards found under {path}")
        warn_missing_index(path, self.kind, num_shards=len(entries))

        def read_metadata(name: str) -> ShardInfoDict:
            full = posixpath.join(path, name)
            try:
                size = int(storage.stat(full)["size"])
                count = self._read_row_count(full, storage, size)
                return ShardInfoDict(
                    basename=name, bytes=size, num_rows=count, hashes={}, extra={}
                )
            except Exception as exc:
                raise ValueError(
                    f"Failed to read {self.label} shard {full}: {exc}"
                ) from exc

        with ThreadPoolExecutor(
            max_workers=min(32, (len(entries) + 4) // 5)
        ) as executor:
            shards = list(executor.map(read_metadata, entries))
        return self._discover_from_index_data(
            ShardIndex(format_version=1, shards=shards)
        )

    def discover_counts(
        self, path: str, storage: StorageBackend
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read validated index counts without building locator metadata."""
        index = self._load_index(path, storage)
        if index is not None:
            counts = [shard["num_rows"] for shard in index["shards"]]
        else:
            discovered, _ = self._discover_by_scanning(path, storage)
            counts = list(discovered.values())
        return np.arange(len(counts), dtype=np.int64), np.array(counts, dtype=np.int64)

    def build_locators(self, dataset: "Dataset") -> Mapping[int, ShardLocator]:
        """Build plain-file locators from discovered metadata."""
        path, shards = dataset.backend.get("path"), dataset.backend.get("shards")
        if not isinstance(path, str) or not isinstance(shards, Mapping):
            raise ValueError(f"{self.label} dataset missing path or shards metadata")
        locators = {}
        for shard_id, meta in shards.items():
            raw = meta.get("raw") if isinstance(meta, Mapping) else None
            if not isinstance(raw, Mapping) or not isinstance(raw.get("basename"), str):
                raise ValueError(
                    f"Shard {shard_id} missing {self.label} raw metadata or basename"
                )
            size = raw.get("bytes")
            if not isinstance(size, (int, float, str)):
                raise ValueError(f"Shard {shard_id} missing or invalid byte size")
            extra = meta.get("extra")
            locators[int(shard_id)] = ShardLocator(
                dataset=dataset.name,
                shard_id=int(shard_id),
                format=self.kind,
                root=path,
                raw=ShardFile(
                    basename=raw["basename"],
                    bytes=int(size),
                    hashes=dict(raw.get("hashes", {})),
                ),
                zip=None,
                compression=None,
                extra=extra if isinstance(extra, Mapping) else None,
            )
        return locators
