# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Parquet index builder.

Registers the ``parquet`` ``IndexBuilder`` on import; build an index via the
``zephon.build_index`` CLI.
"""

from zephon._internal.io.index.index_builder import (
    IndexBuilder,
    ShardInfo,
    register_builder,
)
from zephon._internal.io.suffixes import PARQUET_SUFFIXES

try:
    import pyarrow.parquet as _pq
except ImportError:
    _pq = None


class ParquetIndexBuilder(IndexBuilder):
    """Index builder for Parquet datasets."""

    suffixes = PARQUET_SUFFIXES

    def extract_shard_info(self, path: str, file_size: int) -> ShardInfo:
        """Extract row count and metadata from a Parquet file."""
        if _pq is None:
            raise ImportError(
                "pyarrow is required for Parquet index building. "
                "Install with: pip install zephon[parquet]"
            )

        # Read metadata only (footer), not the full file
        metadata = _pq.read_metadata(path)

        # Extract row group information
        row_groups = []
        for rg_idx in range(metadata.num_row_groups):
            rg = metadata.row_group(rg_idx)
            row_groups.append(
                {
                    "num_rows": rg.num_rows,
                    "total_byte_size": rg.total_byte_size,
                }
            )

        return ShardInfo(
            basename=path.rsplit("/", 1)[-1],
            bytes=file_size,
            num_rows=metadata.num_rows,
            extra={
                "num_row_groups": metadata.num_row_groups,
                "row_groups": row_groups,
            },
        )


register_builder("parquet", ParquetIndexBuilder)


__all__ = ["ParquetIndexBuilder"]
