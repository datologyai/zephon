# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex index builder.

Registers the ``vortex`` ``IndexBuilder`` on import; build an index via the
``zephon.build_index`` CLI.
"""

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import cast

from zephon._internal.io.formats.vortex_metadata import (
    read_vortex_shard_info,
    scan_vortex_shard_metadata,
)
from zephon._internal.io.index.index_builder import (
    IndexBuilder,
    ShardInfo,
    register_builder,
)
from zephon._internal.io.index.index_types import ShardIndex, ShardInfoDict
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.storage.router import RouterStorageBackend
from zephon._internal.io.suffixes import VORTEX_SUFFIXES


class VortexIndexBuilder(IndexBuilder):
    """Index builder for Vortex datasets."""

    suffixes = VORTEX_SUFFIXES

    def __init__(self, storage: StorageBackend | None = None) -> None:
        self._storage = (
            storage
            if storage is not None
            else RouterStorageBackend(local_root=Path.cwd())
        )

    def extract_shard_info(self, path: str, file_size: int) -> ShardInfo:
        """Extract row count and metadata from a Vortex file."""
        return read_vortex_shard_info(path, self._storage, file_size)

    def build(
        self,
        dataset_dir: str | Path,
        *,
        progress: bool = True,
        progress_interval: int = 100,
    ) -> ShardIndex:
        """Build a local or remote index using the same metadata reads as discovery."""
        del progress_interval
        shards = scan_vortex_shard_metadata(
            str(dataset_dir), self._storage, warn_if_unindexed=False
        )
        if progress:
            print(f"Found {len(shards)} {self._file_patterns} files, read metadata")
        return ShardIndex(
            format_version=1,
            shards=[cast(ShardInfoDict, asdict(shard)) for shard in shards],
        )

    def create_index(
        self,
        dataset_dir: str | Path,
        *,
        output_path: str | Path | None = None,
        progress: bool = True,
    ) -> Path | str:
        """Build and write index.json through the configured storage backend."""
        index = self.build(dataset_dir, progress=progress)
        target = (
            str(output_path)
            if output_path is not None
            else os.path.join(str(dataset_dir), "index.json")
        )
        self._storage.put(target, json.dumps(index, indent=2).encode("utf-8"))
        if progress:
            total_rows = sum(shard["num_rows"] for shard in index["shards"])
            total_size_mb = sum(shard["bytes"] for shard in index["shards"]) / (
                1024 * 1024
            )
            print(f"Created {target}")
            print(
                f"  {len(index['shards'])} shards, "
                + f"{total_rows:,} total rows, "
                + f"{total_size_mb:.1f} MB"
            )
        return target if "://" in target else Path(target)


register_builder("vortex", VortexIndexBuilder)


__all__ = ["VortexIndexBuilder"]
