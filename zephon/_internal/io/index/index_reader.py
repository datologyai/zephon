# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared utility for finding and loading dataset index JSON files.

This module owns the list of candidate filenames (e.g., index.json, _index.json)
and provides a single function to find and load the first existing index file
in a dataset directory.
"""

from __future__ import annotations

import json
import logging

from zephon._internal.io.index.index_types import IndexData, is_index_data
from zephon._internal.io.storage.base import StorageBackend

logger = logging.getLogger(__name__)

INDEX_FILENAMES: list[str] = ["index.json", "_index.json"]

# Command that materializes an index.json for each indexable format.
_INDEX_BUILD_COMMANDS: dict[str, str] = {
    "parquet": "python -m zephon.build_index parquet {path}",
    "vortex": "python -m zephon.build_index vortex {path}",
}

# Paths already warned about, so we shout once per dataset rather than per scan.
_warned_paths: set[str] = set()

# Shard-count heuristic for escalating the missing-index hint to WARNING.
_LOUD_SCAN_MIN_SHARDS = 64


def find_and_load_index(dir_path: str, storage: StorageBackend) -> IndexData | None:
    """Find and load the first existing index file in a dataset directory.

    Tries each candidate in INDEX_FILENAMES in order. Returns parsed JSON data
    for the first existing file, or None if none exist.
    JSON parse errors propagate to the caller.

    Args:
        dir_path: Dataset directory path (local or remote, e.g. s3://bucket/dir).
        storage: Storage backend for file access.

    Returns:
        Parsed :class:`IndexData` for the first existing file, or ``None``.
    """
    base = dir_path.rstrip("/") or dir_path
    for filename in INDEX_FILENAMES:
        path = f"{base}/{filename}"
        if storage.exists(path):
            with storage.open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if is_index_data(data):
                    return data
                raise ValueError(f"Invalid index format: {path}")
    return None


def warn_missing_index(path: str, fmt: str, *, num_shards: int | None = None) -> None:
    """Loudly warn that ``path`` has no ``index.json`` and is being scanned shard-by-shard.

    Discovery without an index opens every shard to read its metadata — one
    network/disk round-trip per shard, which is painfully slow for large
    datasets. Building an index once makes later discovery O(1). Warns at most
    once per ``path`` so it shouts per dataset, not per scan, and logs at INFO
    when the shard count is known to be small.

    Args:
        path: Dataset directory being scanned.
        fmt: Format kind (``"parquet"``, ``"vortex"``); selects the build hint.
        num_shards: Number of shards about to be scanned, if known.
    """
    if path in _warned_paths:
        return
    _warned_paths.add(path)

    build_cmd = _INDEX_BUILD_COMMANDS.get(fmt, "").format(path=path)
    if num_shards is not None and num_shards < _LOUD_SCAN_MIN_SHARDS:
        logger.info(
            "No index.json for %s dataset at %s; reading metadata from %d shard(s)%s.",
            fmt,
            path,
            num_shards,
            f" (build one with: {build_cmd})" if build_cmd else "",
        )
        return

    scope = f"{num_shards} shards" if num_shards is not None else "every shard"
    hint = f"\n  ==> Build one once with:  {build_cmd}" if build_cmd else ""
    logger.warning(
        """
================================ SLOW DATASET DISCOVERY ================================
  No usable index.json found for %s dataset at:
    %s
  Falling back to opening %s to read metadata (one read per shard).
  This can be VERY slow for large datasets.%s
=======================================================================================""",
        fmt,
        path,
        scope,
        hint,
    )


__all__ = ["INDEX_FILENAMES", "find_and_load_index", "warn_missing_index"]
