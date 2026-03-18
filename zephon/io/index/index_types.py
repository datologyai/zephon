# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Typed representations of the index.json schemas.

Example index data structures:

- **ShardIndex**: written by zephon's IndexBuilder.
  Discriminated by the ``format_version`` key.
- **MdsIndex**: upstream MosaicML Streaming format.  Has ``shards`` but
  no ``format_version``.
- **LitDataIndex**: upstream Lightning LitData format.  Has ``config``
  and ``chunks``.
"""

from __future__ import annotations

from typing import Any, TypedDict, TypeGuard, Union


class ShardInfoDict(TypedDict):
    """Serialised form of :class:`~zephon.io.index.index_builder.ShardInfo`."""

    basename: str
    bytes: int
    num_rows: int
    hashes: dict[str, str]
    extra: dict[str, Any]


class ShardIndex(TypedDict):
    """Index written by :class:`~zephon.io.index.index_builder.IndexBuilder`."""

    format_version: int
    shards: list[ShardInfoDict]


class MdsIndex(TypedDict):
    """Upstream MosaicML Streaming ``index.json`` schema."""

    shards: list[dict[str, Any]]


class LitDataIndex(TypedDict):
    """Upstream Lightning LitData ``index.json`` schema."""

    config: dict[str, Any]
    chunks: list[dict[str, Any]]


IndexData = Union[ShardIndex, LitDataIndex, MdsIndex]


# ---------------------------------------------------------------------------
# TypeGuard helpers
# ---------------------------------------------------------------------------


def is_shard_index(data: IndexData) -> TypeGuard[ShardIndex]:
    """True when *data* matches the zephon-written ShardIndex shape."""
    return isinstance(data, dict) and "format_version" in data and "shards" in data


def is_litdata_index(data: IndexData) -> TypeGuard[LitDataIndex]:
    """True when *data* matches the LitData index shape."""
    return isinstance(data, dict) and "config" in data and "chunks" in data


def is_mds_index(data: IndexData) -> TypeGuard[MdsIndex]:
    """True when *data* matches the MDS index shape."""
    return isinstance(data, dict) and "shards" in data and "format_version" not in data


def is_index_data(data: Any) -> TypeGuard[IndexData]:
    """True when *data* matches one of the known index schemas."""
    return is_shard_index(data) or is_litdata_index(data) or is_mds_index(data)


__all__ = [
    "IndexData",
    "LitDataIndex",
    "MdsIndex",
    "ShardIndex",
    "ShardInfoDict",
    "is_index_data",
    "is_litdata_index",
    "is_mds_index",
    "is_shard_index",
]
