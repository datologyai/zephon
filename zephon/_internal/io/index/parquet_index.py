# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Parquet index builder.

Registers the ``parquet`` ``IndexBuilder`` on import; build an index via the
``zephon.build_index`` CLI.
"""

from zephon._internal.io.formats.parquet import _read_parquet_metadata
from zephon._internal.io.index.index_builder import (
    IndexBuilder,
    ShardInfo,
    register_builder,
)
from zephon._internal.io.suffixes import PARQUET_SUFFIXES


class ParquetIndexBuilder(IndexBuilder):
    """Index builder for Parquet datasets."""

    suffixes = PARQUET_SUFFIXES

    def extract_shard_info(self, path: str, file_size: int) -> ShardInfo:
        """Extract row count and metadata from a Parquet file."""
        metadata = _read_parquet_metadata(path, self._storage, size=file_size)

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
