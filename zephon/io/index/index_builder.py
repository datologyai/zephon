# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Generic index builder for dataset preprocessing.

This module provides a base class for building index.json files that enable
O(1) dataset discovery instead of O(N) file scanning. Each format can
implement its own IndexBuilder subclass to extract format-specific metadata.

Usage:
    python -m zephon.io.index.index_builder <format> <dataset_dir>

Or from Python:
    from zephon.io.index.index_builder import create_index
    create_index('jsonl', '/path/to/dataset')
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, cast

from zephon.io.index.index_types import ShardIndex, ShardInfoDict


@dataclass
class ShardInfo:
    """Metadata for a single shard file."""

    basename: str
    bytes: int
    num_rows: int
    hashes: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class IndexBuilder(ABC):
    """Base class for format-specific index builders.

    Subclasses must implement:
    - file_pattern: glob pattern for matching shard files (e.g., "*.jsonl")
    - extract_shard_info: extract metadata from a single file
    """

    file_pattern: str

    @abstractmethod
    def extract_shard_info(
        self,
        path: str,
        file_size: int,
    ) -> ShardInfo:
        """Extract metadata from a shard file.

        Args:
            path: Full path to the shard file
            file_size: File size in bytes (already retrieved)

        Returns:
            ShardInfo with row count and any format-specific metadata
        """
        ...

    def scan_directory(self, dataset_dir: str | Path) -> list[str]:
        """Find all matching shard files in directory.

        Args:
            dataset_dir: Directory to scan

        Returns:
            Sorted list of matching filenames (not full paths)
        """
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            raise ValueError(f"Directory does not exist: {dataset_dir}")

        entries = [
            f.name
            for f in dataset_dir.iterdir()
            if f.is_file() and fnmatch(f.name, self.file_pattern)
        ]

        if not entries:
            raise ValueError(f"No {self.file_pattern} files found in {dataset_dir}")

        return sorted(entries)

    def build(
        self,
        dataset_dir: str | Path,
        *,
        progress: bool = True,
        progress_interval: int = 100,
    ) -> ShardIndex:
        """Build the index structure for a dataset.

        Args:
            dataset_dir: Directory containing shard files
            progress: Whether to print progress messages
            progress_interval: Print progress every N files

        Returns:
            A :class:`ShardIndex` ready to be written as JSON.
        """
        dataset_dir = Path(dataset_dir)
        entries = self.scan_directory(dataset_dir)

        if progress:
            print(
                f"Found {len(entries)} {self.file_pattern} files, reading metadata..."
            )

        shards: list[ShardInfoDict] = []
        for i, name in enumerate(entries, 1):
            full_path = dataset_dir / name
            file_size = os.path.getsize(full_path)

            info = self.extract_shard_info(str(full_path), file_size)
            shards.append(cast(ShardInfoDict, asdict(info)))

            if progress and i % progress_interval == 0:
                print(f"  Processed {i}/{len(entries)} files...")

        return ShardIndex(format_version=1, shards=shards)

    def create_index(
        self,
        dataset_dir: str | Path,
        *,
        output_path: str | Path | None = None,
        progress: bool = True,
    ) -> Path:
        """Build and write index.json for a dataset.

        Args:
            dataset_dir: Directory containing shard files
            output_path: Where to write index.json (default: dataset_dir/index.json)
            progress: Whether to print progress messages

        Returns:
            Path to the created index.json file
        """
        dataset_dir = Path(dataset_dir)
        index = self.build(dataset_dir, progress=progress)

        if output_path is None:
            output_path = dataset_dir / "index.json"
        else:
            output_path = Path(output_path)

        with open(output_path, "w") as f:
            json.dump(index, f, indent=2)

        if progress:
            total_rows = sum(s["num_rows"] for s in index["shards"])
            total_size_mb = sum(s["bytes"] for s in index["shards"]) / (1024 * 1024)
            print(f"Created {output_path}")
            print(
                f"  {len(index['shards'])} shards, "
                f"{total_rows:,} total rows, "
                f"{total_size_mb:.1f} MB"
            )

        return output_path


# Registry of format -> IndexBuilder
_BUILDERS: dict[str, type[IndexBuilder]] = {}


def register_builder(format_name: str, builder_class: type[IndexBuilder]) -> None:
    """Register an IndexBuilder class for a format."""
    _BUILDERS[format_name] = builder_class


def get_builder(format_name: str) -> IndexBuilder:
    """Get an IndexBuilder instance for the given format."""
    if format_name not in _BUILDERS:
        available = ", ".join(sorted(_BUILDERS.keys())) or "(none)"
        raise ValueError(
            f"No index builder registered for format '{format_name}'. "
            f"Available: {available}"
        )
    return _BUILDERS[format_name]()


def create_index(
    format_name: str,
    dataset_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    progress: bool = True,
) -> Path:
    """Create index.json for a dataset using the appropriate builder.

    Args:
        format_name: Format type (e.g., 'jsonl', 'parquet', 'vortex')
        dataset_dir: Directory containing shard files
        output_path: Where to write index.json (default: dataset_dir/index.json)
        progress: Whether to print progress messages

    Returns:
        Path to the created index.json file
    """
    builder = get_builder(format_name)
    return builder.create_index(dataset_dir, output_path=output_path, progress=progress)


def main() -> None:
    """CLI entry point."""
    import sys

    if len(sys.argv) < 3:
        available = ", ".join(sorted(_BUILDERS.keys())) or "(none registered)"
        print("Usage: python -m zephon.io.index.index_builder <format> <dataset_dir>")
        print()
        print("Creates index.json for fast dataset discovery.")
        print(f"Available formats: {available}")
        print()
        print("To register format builders, import the format module first:")
        print("  from zephon.io.index import jsonl_index  # registers 'jsonl'")
        sys.exit(1)

    format_name = sys.argv[1]
    dataset_dir = sys.argv[2]

    try:
        create_index(format_name, dataset_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "IndexBuilder",
    "ShardInfo",
    "create_index",
    "get_builder",
    "register_builder",
]
