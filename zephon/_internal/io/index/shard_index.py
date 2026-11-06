# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Validation and projections for Zephon's shard index schema.

These helpers apply to Zephon-written indexes, not upstream MDS or LitData
indexes. Callers retain suffix checks, format-specific extras and the policy
for handling an unusable index.
"""

from typing import Any, Mapping

import numpy as np

from zephon._internal.io.index.index_types import ShardIndex


def is_nonnegative_int64(value: object) -> bool:
    """Accept counts and sizes representable by the catalog's int64 columns."""
    return type(value) is int and 0 <= value < 1 << 63


def validate_shard_index(data: Mapping[str, Any]) -> None:
    """Validate common fields, leaving format-specific metadata to the caller."""
    version = data.get("format_version")
    if type(version) is not int or version != 1:
        raise ValueError("Unsupported shard index version")
    shards = data.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("index.json contains no shards")
    for i, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(f"Invalid index shard {i}")
        hashes = shard.get("hashes", {})
        if (
            not isinstance(shard.get("basename"), str)
            or not is_nonnegative_int64(shard.get("num_rows"))
            or not is_nonnegative_int64(shard.get("bytes"))
            or not isinstance(hashes, dict)
            or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in hashes.items()
            )
            or not isinstance(shard.get("extra", {}), dict)
        ):
            raise ValueError(f"Invalid index metadata for shard {i}")


def shard_index_counts(index: ShardIndex) -> tuple[np.ndarray, np.ndarray]:
    """Extract counts from a validated index without building locator metadata."""
    counts = np.fromiter(
        (shard["num_rows"] for shard in index["shards"]),
        dtype=np.int64,
        count=len(index["shards"]),
    )
    return np.arange(counts.size, dtype=np.int64), counts


def shard_index_metadata(
    index: ShardIndex,
) -> tuple[dict[int, int], dict[int, dict[str, Any]]]:
    """Project validated index entries to raw files with a record-length hint.

    Use this when entries describe the files read directly by the native reader.
    Formats needing decompression or a different extra schema build their own
    discovery metadata instead.
    """
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
