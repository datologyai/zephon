# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Generic index-builder infrastructure.

Base ``IndexBuilder`` class and a format registry for building ``index.json``
files (O(1) dataset discovery instead of O(N) file scanning). Each format
registers an ``IndexBuilder`` subclass; the user-facing CLI is
``zephon.build_index``.
"""

from __future__ import annotations

import json
import posixpath
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from zephon._internal.io.index.index_types import ShardIndex, ShardInfoDict
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.storage.router import RouterStorageBackend


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
    - suffixes: data-file suffixes to index, from :mod:`zephon._internal.io.suffixes`
    - extract_shard_info: extract metadata from a single file
    """

    suffixes: tuple[str, ...]

    def __init__(
        self, storage: StorageBackend | None = None, *, max_workers: int = 8
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._storage = (
            storage
            if storage is not None
            else RouterStorageBackend(local_root=Path.cwd())
        )
        self._max_workers = max_workers

    @property
    def _file_patterns(self) -> str:
        return ", ".join(f"*{suffix}" for suffix in self.suffixes)

    @abstractmethod
    def extract_shard_info(
        self,
        path: str,
        file_size: int,
    ) -> ShardInfo:
        """Extract metadata from a shard file.

        Args:
            path: Local path or remote URL for the shard file
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
        return [name for name, _ in self._scan_shards(str(dataset_dir))]

    def _scan_shards(self, dataset_dir: str) -> list[tuple[str, int]]:
        """List direct child shard files and their sizes through the backend."""
        entries = sorted(
            (name, size)
            for name, size in self._storage.walk(dataset_dir)
            if "/" not in name and name.endswith(self.suffixes)
        )
        if not entries:
            raise ValueError(f"No {self._file_patterns} files found in {dataset_dir}")
        return entries

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
        if progress_interval < 1:
            raise ValueError("progress_interval must be at least 1")
        dataset_dir = str(dataset_dir)
        entries = self._scan_shards(dataset_dir)

        if progress:
            print(
                f"Found {len(entries)} {self._file_patterns} files, reading metadata..."
            )

        def read_shard(entry: tuple[str, int]) -> ShardInfo:
            name, size = entry
            path = posixpath.join(dataset_dir, name)
            try:
                return self.extract_shard_info(path, size)
            except Exception as exc:
                raise ValueError(f"Failed to index shard {path}: {exc}") from exc

        shards: list[ShardInfoDict] = []
        workers = min(self._max_workers, len(entries))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Consume in filename order so index shard IDs never depend on IO timing.
            for i, info in enumerate(executor.map(read_shard, entries), 1):
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
    ) -> Path | str:
        """Build and write index.json for a dataset.

        Args:
            dataset_dir: Directory containing shard files
            output_path: Where to write index.json (default: dataset_dir/index.json)
            progress: Whether to print progress messages

        Returns:
            Path to a local index or URL string to a remote index.
        """
        index = self.build(dataset_dir, progress=progress)

        target = (
            str(output_path)
            if output_path is not None
            else posixpath.join(str(dataset_dir), "index.json")
        )
        self._storage.put(target, json.dumps(index, indent=2).encode("utf-8"))

        if progress:
            total_rows = sum(s["num_rows"] for s in index["shards"])
            total_size_mb = sum(s["bytes"] for s in index["shards"]) / (1024 * 1024)
            print(f"Created {target}")
            print(
                f"  {len(index['shards'])} shards, "
                f"{total_rows:,} total rows, "
                f"{total_size_mb:.1f} MB"
            )

        return target if "://" in target else Path(target)


# Registry of format -> IndexBuilder
_BUILDERS: dict[str, type[IndexBuilder]] = {}


def register_builder(format_name: str, builder_class: type[IndexBuilder]) -> None:
    """Register an IndexBuilder class for a format."""
    _BUILDERS[format_name] = builder_class


def get_builder(format_name: str, *, max_workers: int = 8) -> IndexBuilder:
    """Get an IndexBuilder instance for the given format."""
    if format_name not in _BUILDERS:
        available = ", ".join(sorted(_BUILDERS.keys())) or "(none)"
        raise ValueError(
            f"No index builder registered for format '{format_name}'. "
            f"Available: {available}"
        )
    return _BUILDERS[format_name](max_workers=max_workers)


def create_index(
    format_name: str,
    dataset_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    progress: bool = True,
    max_workers: int = 8,
) -> Path | str:
    """Create index.json for a dataset using the appropriate builder.

    Args:
        format_name: Format type (e.g., 'jsonl', 'parquet', 'vortex')
        dataset_dir: Directory containing shard files
        output_path: Where to write index.json (default: dataset_dir/index.json)
        progress: Whether to print progress messages
        max_workers: Maximum number of shards to inspect concurrently.

    Returns:
        Path to a local index or URL string to a remote index.
    """
    builder = get_builder(format_name, max_workers=max_workers)
    return builder.create_index(dataset_dir, output_path=output_path, progress=progress)


__all__ = [
    "IndexBuilder",
    "ShardInfo",
    "create_index",
    "get_builder",
    "register_builder",
]
