# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Parquet index builder.

Creates index.json for Parquet datasets to avoid reading metadata from every
file during discovery.

Usage:
    python -m zephon.io.index.parquet_index <dataset_dir>

Or from Python:
    from zephon.io.index.parquet_index import ParquetIndexBuilder
    builder = ParquetIndexBuilder()
    builder.create_index('/path/to/parquet/dataset')
"""

from zephon.io.index.index_builder import IndexBuilder, ShardInfo, register_builder

try:
    import pyarrow.parquet as _pq
except ImportError:
    _pq = None


class ParquetIndexBuilder(IndexBuilder):
    """Index builder for Parquet datasets."""

    file_pattern = "*.parquet"

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


def main() -> None:
    """CLI entry point for Parquet index building."""
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m zephon.io.index.parquet_index <dataset_dir>")
        print()
        print("Creates index.json for fast Parquet dataset discovery.")
        print("This is optional but recommended for datasets with 100+ shards.")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    builder = ParquetIndexBuilder()

    try:
        builder.create_index(dataset_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ["ParquetIndexBuilder"]
