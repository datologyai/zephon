# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex index builder.

Creates index.json for Vortex datasets to avoid opening every file
during discovery.

Usage:
    python -m zephon.tools.vortex_index <dataset_dir>

Or from Python:
    from zephon.tools.vortex_index import VortexIndexBuilder
    builder = VortexIndexBuilder()
    builder.create_index('/path/to/vortex/dataset')
"""

from typing import Any

from zephon.tools.index_builder import IndexBuilder, ShardInfo, register_builder

try:
    import vortex as _vortex
except ImportError:
    _vortex = None


class VortexIndexBuilder(IndexBuilder):
    """Index builder for Vortex datasets."""

    file_pattern = "*.vortex"

    def extract_shard_info(self, path: str, file_size: int) -> ShardInfo:
        """Extract row count and metadata from a Vortex file."""
        if _vortex is None:
            raise ImportError(
                "vortex-data is required for Vortex index building. "
                "Install with: pip install zephon[vortex]"
            )

        url = f"file://{path}"
        reader = _vortex.io.read_url(url)
        count = len(reader)

        # Extract schema info if available
        extra: dict[str, Any] = {"length": count}

        return ShardInfo(
            basename=path.rsplit("/", 1)[-1],
            bytes=file_size,
            num_rows=count,
            extra=extra,
        )


register_builder("vortex", VortexIndexBuilder)


def main() -> None:
    """CLI entry point for Vortex index building."""
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m zephon.tools.vortex_index <dataset_dir>")
        print()
        print("Creates index.json for fast Vortex dataset discovery.")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    builder = VortexIndexBuilder()

    try:
        builder.create_index(dataset_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ["VortexIndexBuilder"]
