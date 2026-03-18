# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Index utilities for zephon datasets.

Provides utilities for finding/loading index files (index_reader) and
building index.json for fast dataset discovery (index_builder, vortex_index,
parquet_index).
"""

from zephon.io.index.index_builder import IndexBuilder, ShardInfo, create_index
from zephon.io.index.index_reader import INDEX_FILENAMES, find_and_load_index

__all__ = [
    "INDEX_FILENAMES",
    "IndexBuilder",
    "ShardInfo",
    "create_index",
    "find_and_load_index",
]
