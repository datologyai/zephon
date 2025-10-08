# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Core IO datatypes shared across storage, caching, and formats.

These structures provide the glue between dataset descriptors, storage backends,
cache managers, and format-specific shard readers. They intentionally avoid
referencing heavyweight runtime objects so they can be serialized, logged, and
passed between processes safely.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ShardFile:
    """Metadata describing a single file that belongs to a shard.

    Produced during dataset discovery (before any local caching). Contains the
    remote basename, expected byte size, and optional hashes used for
    validation. These values are stable across runs and shared between
    transport, caching, and format layers.
    """

    basename: str
    bytes: int
    hashes: Mapping[str, str]


@dataclass(frozen=True)
class ShardLocator:
    """Immutable metadata required to locate and prepare a shard.

    Attributes:
        dataset: Logical dataset name for logging and cache namespaces.
        shard_id: Integer identifier of the shard within the dataset.
        format: Format identifier (``mds``, ``parquet``, etc.).
        root: Root directory or remote prefix containing shard files.
        raw: Metadata for the raw shard payload. Always present.
        zip: Metadata for a compressed companion file, when applicable.
        compression: Name of the compression algorithm used for ``zip``.
        extra: Format-specific metadata (column schemas, dtype info, ...).
    """

    dataset: str
    shard_id: int
    format: str
    root: str
    raw: ShardFile
    zip: ShardFile | None = None
    compression: str | None = None
    extra: Mapping[str, object] | None = None


@dataclass(frozen=True)
class LocalShardFile:
    """Reference to a shard file on the local filesystem.

    This reflects the actual on-disk path chosen by the resolver/cache and the
    observed size after any decompression.
    """

    path: Path
    bytes: int


@dataclass(frozen=True)
class LocalShardRef:
    """Local view of a shard after storage/caching has prepared it.

    Mirrors :class:`ShardLocator` but with concrete local paths so format
    handlers can build ``RandomAccessShard`` instances without knowing about
    caching or remote storage.
    """

    raw: LocalShardFile
    zip: LocalShardFile | None = None
    compression: str | None = None
    extra: Mapping[str, object] | None = None


__all__ = [
    "LocalShardFile",
    "LocalShardRef",
    "ShardFile",
    "ShardLocator",
]
