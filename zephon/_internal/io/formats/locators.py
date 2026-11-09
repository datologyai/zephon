# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Locators for normalized raw files and optional compressed companions.

Format handlers decide how to discover and normalize their shard metadata.
These helpers only consume the common raw/zip/compression/extra shape; they
do not inspect native files or impose a discovery or reader lifecycle.
"""

from numbers import Integral
from typing import TYPE_CHECKING, Any, Mapping

from zephon._internal.io.types import LocalShardRef, ShardFile, ShardLocator

if TYPE_CHECKING:
    from zephon.io.dataset import Dataset


def shard_length(local_ref: LocalShardRef) -> int | None:
    """Return the optional locator length hint, including legacy strings."""
    value = (local_ref.extra or {}).get("length")
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (ValueError, OverflowError):
            pass
    return None


def _shard_file(meta: Any, shard_id: int, kind: str) -> ShardFile:
    if not isinstance(meta, Mapping) or not isinstance(meta.get("basename"), str):
        raise ValueError(f"Shard {shard_id} missing {kind} metadata or basename")
    size = meta.get("bytes")
    if isinstance(size, bool) or not isinstance(size, (Integral, str)):
        raise ValueError(f"Shard {shard_id} invalid byte size for {kind} file")
    try:
        size = int(size)
    except ValueError as exc:
        raise ValueError(f"Shard {shard_id} invalid byte size for {kind} file") from exc
    if not 0 <= size < 1 << 63:
        raise ValueError(f"Shard {shard_id} invalid byte size for {kind} file")
    hashes = meta.get("hashes") or {}
    if not isinstance(hashes, Mapping):
        raise ValueError(f"Shard {shard_id} invalid hashes for {kind} file")
    return ShardFile(
        basename=meta["basename"],
        bytes=size,
        hashes={str(k): str(v) for k, v in hashes.items()},
    )


def build_file_locators(
    dataset: "Dataset", *, format_kind: str, root: str | None = None
) -> dict[int, ShardLocator]:
    """Build locators from normalized metadata, preserving compression and extras.

    The default root is the dataset's backend path. Manifest-based handlers may
    pass the resolved data directory explicitly.
    """
    path = dataset.backend.get("path") if root is None else root
    shards = dataset.backend.get("shards")
    if not isinstance(path, str) or not isinstance(shards, Mapping):
        raise ValueError(f"{format_kind} dataset missing path or shards metadata")
    locators = {}
    for shard_id_obj, meta in shards.items():
        shard_id = int(shard_id_obj)
        if not isinstance(meta, Mapping):
            raise ValueError(f"Invalid shard metadata for shard {shard_id}")
        raw = _shard_file(meta.get("raw"), shard_id, "raw")
        zipped = meta.get("zip")
        zip_file = _shard_file(zipped, shard_id, "zip") if zipped is not None else None
        compression, extra = meta.get("compression"), meta.get("extra")
        locators[shard_id] = ShardLocator(
            dataset=dataset.name,
            shard_id=shard_id,
            format=format_kind,
            root=path,
            raw=raw,
            zip=zip_file,
            compression=str(compression) if compression else None,
            extra=extra if isinstance(extra, Mapping) else None,
        )
    return locators
