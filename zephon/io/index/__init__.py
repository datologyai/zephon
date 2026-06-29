# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Index utilities for zephon datasets.

Provides utilities for finding/loading index files (index_reader) and
building index.json for fast dataset discovery (index_builder, vortex_index,
parquet_index).
"""

from zephon.io.index.index_builder import IndexBuilder, ShardInfo, create_index
from zephon.io.index.index_reader import find_and_load_index, warn_missing_index
from zephon.io.index.index_types import (
    IndexData,
    LitDataIndex,
    MdsIndex,
    ShardIndex,
    ShardInfoDict,
    is_litdata_index,
    is_mds_index,
    is_shard_index,
)

__all__ = [
    "IndexBuilder",
    "IndexData",
    "LitDataIndex",
    "MdsIndex",
    "ShardIndex",
    "ShardInfo",
    "ShardInfoDict",
    "create_index",
    "find_and_load_index",
    "warn_missing_index",
    "is_litdata_index",
    "is_mds_index",
    "is_shard_index",
]
